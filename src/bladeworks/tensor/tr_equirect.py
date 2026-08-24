"""F2 transition port: the eight 360° equirectangular transitions (handler ``equirectangular``).

Architecture map
================

    plan.build_tensor_plan -> transitions.lower_transition (handler "equirectangular")
        -> _lower_equirectangular(item, LowerContext)
             transitions.equirectangular.build_equirectangular_transition_plan(xfade_id,
                 parameter_values)            -- the SAME registry resolution the CPU builder
                                                 runs (``ffmpeg._build_stock_transition_groups``),
                                                 parameter-aware (direction / speed menu /
                                                 soften edges / border / slices / spacing)
               prefilter None + admitted id  -> kind "xfade_custom" (transitions.XfadePayload:
                                                 the wrap-aware ``mod``-longitude expression is
                                                 evaluated by tensor/expr.py exactly like the
                                                 flat registry expressions)
               prefilter equirectangular_gaussian
                                             -> kind "equirect_gaussian_blur" (this module:
                                                 the stock ``gblur`` graph, see below)
               prefilter equirectangular_bloom
                                             -> loud reject (see below)

Six ids are pure ``xfade=custom`` expressions in the reference and lower to ``xfade_custom``:
``equirect_circle_wipe``, ``equirect_divide``, ``equirect_push``, ``equirect_reveal_wipe``,
``equirect_slide``, ``equirect_wipe`` (``ADMITTED_EQUIRECT_IDS``; goldens in
``experimental_tests/core/test_tensor_transition_golden.py`` -- default parameter values plus
direction / speed+soften / border / slices variants, float64 CPU bit-exact -- and an
end-to-end SSIM render in ``test_tensor_transition_golden_f1f2.py``).

The two blur variants are NOT expressions in the reference: ``equirect_gaussian_blur`` and
``equirect_bloom_default`` resolve to ``prefilter`` in {``equirectangular_gaussian``,
``equirectangular_bloom``} and the CPU builder runs ``ffmpeg._build_equirectangular_blur_group``
(the plan's ``expression`` field for those ids is the dead five-tap expression the builder
never emits, so evaluating it would be a silent approximation).

360° Gaussian Blur (ported, kind ``equirect_gaussian_blur``) -- per side, in the 8-bit
straight ``gbrap`` code domain of the reference sides (``color.premultiplied_to_code``):

    clean = the side
    strip = [side | side | side]                       (``split`` + ``hstack=inputs=3``)
    strong = gblur(strip, sigma, sigmaV, steps=3)[centre crop]
    mild   = gblur(strip, 0.28 sigma, 0.28 sigmaV, steps=2)[centre crop]
    shoulder = trunc(clean*(1-sw) + mild*sw)         (``blend=all_expr``, ``sw = min(1, 2*activity)``)
    side_out = trunc(shoulder*(1-activity) + strong*activity)
    activity = transitions.equirectangular.equirectangular_blur_activity("gaussian_skew", duration) at T = k*tb

then the two blurred sides go through the small ownership ``xfade=custom`` expression
(``gaussian_lead``, ``ffmpeg._build_equirectangular_blur_group``) evaluated by
``expr.xfade_custom_rgba`` like every other xfade.  ``gblur`` is FFmpeg's Alvarez-Mazorra
recursive Gaussian (``vf_gblur.c``): per axis ``steps`` passes of a causal then anti-causal
first-order IIR ``y[n] = x[n] + nu*y[n-1]`` whose boundary scale ``1/(1-nu)`` is exactly an
infinite replicate pad, ``postscale = (nu/lambda)^steps`` per axis, float32 buffers, clip to
[0, 255], ``lrintf`` (round half even).  Here each IIR pass is the same recursion evaluated
in blocks (lower-Toeplitz matmul + carry, ``_iir_pass_blocked``) on the triple strip -- so the
longitude seam wraps exactly like the reference's ``hstack`` and latitude clamps at the poles.  The envelope,
sigma texts (``ffmpeg._number`` -> float32 AVOption) and the ownership / activity expression
strings are taken from ``ffmpeg.py`` itself, never
re-typed, so the port cannot drift from the emitted graph.  Golden:
``test_tensor_transition_golden_f1f2.py`` (gblur vs the ``ffmpeg`` CLI on plates, the whole
side graph vs the ``ffmpeg`` CLI, and the e2e SSIM render vs the CPU reference).

360° Bloom stays a loud reject (``support.reject("transition (other)", ...)``): its side graph
adds ``colorlevels`` + ``eq`` (a YUV round trip through swscale) + a ``curves`` natural
spline luminance mask + ``alphamerge`` + two alpha ``fade`` ramps + two ``overlay`` passes;
six stock filters with their own conversions and rounding, each needing its own golden.

Contract: see ``tensor/transitions.py`` (``Lowered`` / ``XfadePayload`` / apply ports).

Main callers:
- ``transitions.lower_transition`` (via ``register_handler``); ``transitions.apply_transition``
  (via ``register`` of the ``EquirectGaussian`` transition).
- ``experimental_tests/core/test_tensor_transition_golden.py`` (``resolve_equirectangular_expression``),
  ``test_tensor_transition_golden_f1f2.py`` (``gblur_gbrap``, ``gaussian_blur_side``).

Why this exists: the 360° ids are a separate registry (``transitions.equirectangular``) with
its own parameter contract and its own plan type; keeping the lowering here (one owner per
file) leaves ``tensor/transitions.py``'s xfade path untouched while sharing its apply kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final, Mapping, Optional

import numpy as np
import torch

from ..core.filter_text import format_number as _number, format_seconds as _seconds
from ..transitions.equirectangular import equirectangular_blur_activity
from ..core.model import RenderTransition
from ..transitions.equirectangular import build_equirectangular_transition_plan
from .color import code_to_premultiplied, premultiplied_to_code
from .errors import TensorRenderUnsupported
from .expr import Environment, evaluate, parse, quantize_uint8, xfade_custom_rgba, xfade_progress
from .support import reject
from .transitions import (
    ApplyContext,
    LowerContext,
    Lowered,
    Transition,
    XfadePayload,
    register,
    register_handler,
)

# 360° ids the plan admits (append-only; every entry has a golden -- see module doc).
ADMITTED_EQUIRECT_IDS: Final[tuple[str, ...]] = (
    "equirect_circle_wipe",
    "equirect_divide",
    "equirect_push",
    "equirect_reveal_wipe",
    "equirect_slide",
    "equirect_wipe",
    "equirect_gaussian_blur",
)

# The reference's ``ffmpeg._build_equirectangular_blur_group`` shape for the Gaussian id
# (asserted at lowering so a registry change is loud, not silently re-interpreted).
_GAUSSIAN_ACTIVITY_PROFILE: Final = "gaussian_skew"
_GAUSSIAN_OWNERSHIP_PROFILE: Final = "gaussian_lead"
_STRONG_STEPS: Final = 3
_MILD_STEPS: Final = 2
_MILD_SIGMA_SCALE: Final = 0.28


def resolve_equirectangular_expression(xfade_id: str, parameter_values: Optional[Mapping[str, Any]] = None) -> str:
    """The reference's resolved 360° ``xfade=custom`` expression for an admitted expression id, loudly.

    Runs the very registry resolution the CPU builder runs
    (``build_equirectangular_transition_plan(xfade_id, parameter_values)``); refuses ids
    outside ``ADMITTED_EQUIRECT_IDS`` and plans that carry a blur prefilter (their
    ``expression`` is dead text, see module doc -- the Gaussian id lowers to its own kind).
    """

    if xfade_id not in ADMITTED_EQUIRECT_IDS:
        raise TensorRenderUnsupported(
            f"360° transition {xfade_id!r} is not ported to the tensor renderer "
            f"(admitted: {', '.join(ADMITTED_EQUIRECT_IDS)})"
        )
    plan = build_equirectangular_transition_plan(xfade_id, dict(parameter_values or {}))
    if plan.mode != "custom" or plan.prefilter is not None:
        raise TensorRenderUnsupported(
            f"360° transition {xfade_id!r} resolves to mode={plan.mode!r} prefilter={plan.prefilter!r}, "
            "not a pure custom expression (the Gaussian blur is its own kind, see tr_equirect)"
        )
    return plan.expression


# ---------------------------------------------------------------------------
# 360° Gaussian Blur: the stock gblur side graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquirectGaussianPayload:
    xfade_id: str
    horizontal_strength: float     # plan.strength: sigma = max(1e-4, min(W, H) * strength)
    vertical_strength: float       # plan.spread
    duration_text: str             # ffmpeg._seconds(owned window) as the graph writes it
    ownership_expression: str      # the small ownership xfade text (evaluated by expr)


def gaussian_ownership_expression() -> str:
    """The ``gaussian_lead`` ownership ``xfade=custom`` text of ``ffmpeg._build_equirectangular_blur_group``.

    Copied character-for-character from the builder (it is an inline literal there);
    ``test_tensor_transition_golden_f1f2.py`` asserts the emitted filter script carries it.
    """

    q = "(1-P)"
    normalized = f"clip((({q})-0.02)/0.825,0,1)"
    weight = f"(({normalized})*({normalized})*(3-2*({normalized})))"
    return f"if(gte(P,1),A,if(lte(P,0),B,A*(1-({weight}))+B*({weight})))"


def gblur_params(sigma: float, steps: int) -> tuple[float, float]:
    """``vf_gblur.c`` ``set_params``: ``(nu, postscale)`` for one axis, in the C precisions.

    ``sigma`` is the value the AVOption parsed (float32 of the ``_number`` text); ``lambda``
    and ``dnu`` are doubles, ``nu`` is stored as float32, ``postscale = pow(dnu/lambda, steps)``.
    """

    sigma32 = np.float32(sigma)
    lam = float(sigma32 * sigma32) / (2.0 * steps)   # C: float * float, then promoted
    dnu = (1.0 + 2.0 * lam - math.sqrt(1.0 + 4.0 * lam)) / (2.0 * lam)
    postscale = float(np.float32(math.pow(dnu / lam, steps)))
    nu = float(np.float32(dnu))
    if not math.isfinite(postscale) or postscale == 0.0:
        postscale = 1.0
    if not math.isfinite(nu):
        nu = 0.0
    return nu, postscale


_IIR_BLOCK: Final = 128


def _iir_pass_blocked(rows: torch.Tensor, nu: float, *, causal: bool) -> torch.Tensor:
    """One first-order IIR pass ``y[n] = x[n] + nu*y[n-1]`` along the last axis with the C boundary
    (``ptr[0] *= 1/(1-nu)``, i.e. ``y[-1] = x[0]/(1-nu)``), evaluated in blocks of ``_IIR_BLOCK``:

        y_block = x_block @ T + carry * nu^(1..B)      T[j, i] = nu^(i-j) for i >= j (lower Toeplitz)
        carry   = y_block[-1]

    -- the exact recursion, one matmul + one outer product per block instead of ``L`` scalar
    steps (fast on MPS and CPU) and no unfolded convolution buffers (``conv1d`` on the CPU
    unfolds ``rows x taps x L`` floats: gigabytes at 1080p).  ``causal=False`` mirrors the axis
    (``ptr[W-1] *= bscale; ptr[x-1] += nu*ptr[x]``).
    """

    if not causal:
        return _iir_pass_blocked(rows.flip(-1), nu, causal=True).flip(-1)
    length = rows.shape[-1]
    block = min(_IIR_BLOCK, length)
    device, dtype = rows.device, rows.dtype
    powers = torch.pow(torch.full((block,), nu, device=device, dtype=dtype), torch.arange(1, block + 1, device=device, dtype=dtype))
    index = torch.arange(block, device=device)
    exponent = (index.view(1, block) - index.view(block, 1)).to(dtype)         # i - j
    toeplitz = torch.where(exponent >= 0, torch.pow(torch.full_like(exponent, nu), exponent.clamp(min=0)), torch.zeros_like(exponent))
    carry = rows[:, :1] * float(np.float32(1.0 / (1.0 - nu)))                     # y[-1] = x[0] * bscale
    pieces = []
    for start in range(0, length, block):
        chunk = rows[:, start:start + block]
        width = chunk.shape[-1]
        y = chunk @ toeplitz[:width, :width] + carry * powers[:width].view(1, width)
        carry = y[:, -1:]
        pieces.append(y)
    return torch.cat(pieces, dim=-1)


def _iir_pass_recursion(rows: torch.Tensor, nu: float, *, causal: bool) -> torch.Tensor:
    """The C recursion itself (``horiz_slice_c``), vectorized across rows, sequential along the
    axis: ``ptr[0] *= bscale; ptr[x] += nu*ptr[x-1]`` (and mirrored).  Reference implementation
    for the goldens (same float32 evaluation order as ffmpeg); ``L`` vector ops per pass."""

    out = rows.clone()
    length = out.shape[-1]
    bscale = float(np.float32(1.0 / (1.0 - nu)))
    columns = out.unbind(-1)     # views into ``out``; in-place updates keep the recursion
    if causal:
        columns[0].mul_(bscale)
        for x in range(1, length):
            columns[x].add_(columns[x - 1], alpha=nu)
    else:
        columns[length - 1].mul_(bscale)
        for x in range(length - 1, 0, -1):
            columns[x - 1].add_(columns[x], alpha=nu)
    return out


def _iir_axis(planes: torch.Tensor, nu: float, steps: int, method: str = "blocked") -> torch.Tensor:
    """``steps`` passes of causal + anti-causal IIR along the last axis (``horiz_slice_c``).

    ``method``: ``"blocked"`` (the runtime, any device) or ``"recursion"`` (the C loop, goldens);
    both are goldened against ``ffmpeg gblur`` in ``test_tensor_transition_golden_f1f2.py``.
    """

    if nu <= 0.0:
        return planes
    one_pass = {"blocked": _iir_pass_blocked, "recursion": _iir_pass_recursion}[method]
    rows = planes.reshape(-1, planes.shape[-1])
    for _ in range(steps):
        rows = one_pass(rows, nu, causal=True)
        rows = one_pass(rows, nu, causal=False)
    return rows.reshape(planes.shape)


def gblur_gbrap(
    code: torch.Tensor, *, sigma: float, sigma_v: float, steps: int,
    crop_x: Optional[tuple[int, int]] = None, method: str = "blocked",
) -> torch.Tensor:
    """``gblur=sigma=..:sigmaV=..:steps=..`` on an 8-bit code frame ``[planes, H, W]`` (all planes).

    Input values are code values (any float dtype); output is the C result: float32 IIR passes
    horizontally then vertically, ``* postscale_h * postscale_v``, clip to [0, 255], ``lrintf``.
    ``sigma`` / ``sigma_v`` are the AVOption values (callers pass the ``_number`` text's float).
    ``crop_x = (x0, x1)`` returns columns ``x0:x1`` (the reference's centre ``crop`` after the
    triple-strip blur); columns are independent in the vertical pass, so it is applied first.
    ``method`` selects the IIR implementation (see ``_iir_axis``).
    """

    nu_h, post_h = gblur_params(sigma, steps)
    nu_v, post_v = gblur_params(sigma_v, steps)
    work = code.to(torch.float32)
    work = _iir_axis(work, nu_h, steps, method)
    if crop_x is not None:
        work = work[..., crop_x[0]:crop_x[1]]
    work = _iir_axis(work.transpose(-1, -2).contiguous(), nu_v, steps, method).transpose(-1, -2)
    work = (work * float(np.float32(post_h) * np.float32(post_v))).clamp(0.0, 255.0)
    return torch.round(work).to(code.dtype)


def _sigma_options(width: int, height: int, strength: float) -> tuple[float, float]:
    """The strong and mild sigmas the graph writes for one axis, as the ``gblur`` AVOption reads them.

    ``ffmpeg._equirectangular_blur_side``: ``sigma = max(0.0001, min(W, H) * strength)`` (double),
    strong text ``_number(sigma)``, mild text ``_number(sigma * 0.28)``; AVOption floats are float32.
    """

    sigma = max(0.0001, min(width, height) * strength)
    return float(np.float32(float(_number(sigma)))), float(np.float32(float(_number(sigma * _MILD_SIGMA_SCALE))))


def _scalar(text: str, **variables: float) -> float:
    value = evaluate(parse(text), Environment(variables=dict(variables), samplers={}, device=torch.device("cpu"), dtype=torch.float64))
    if isinstance(value, torch.Tensor):
        raise AssertionError(f"expected a scalar envelope, got a tensor for {text!r}")
    return float(value)


def gaussian_blur_side(
    code: torch.Tensor,
    *,
    width: int,
    height: int,
    horizontal_strength: float,
    vertical_strength: float,
    activity: float,
) -> torch.Tensor:
    """One side of ``ffmpeg._equirectangular_blur_side`` (Gaussian branch) on an 8-bit code frame.

    ``activity`` is the envelope value at this frame (``transitions.equirectangular.equirectangular_blur_activity``
    at ``T``); the two ``blend=all_expr`` stages truncate like the C ``uint8`` store.
    """

    strong_h, mild_h = _sigma_options(width, height, horizontal_strength)
    strong_v, mild_v = _sigma_options(width, height, vertical_strength)
    strip = torch.cat((code, code, code), dim=-1)
    centre = (width, 2 * width)
    strong = gblur_gbrap(strip, sigma=strong_h, sigma_v=strong_v, steps=_STRONG_STEPS, crop_x=centre)
    mild = gblur_gbrap(strip, sigma=mild_h, sigma_v=mild_v, steps=_MILD_STEPS, crop_x=centre)
    shoulder_weight = min(1.0, 2.0 * activity)
    shoulder = quantize_uint8(code * (1.0 - shoulder_weight) + mild * shoulder_weight)
    return quantize_uint8(shoulder * (1.0 - activity) + strong * activity)


def blur_activity(duration_text: str, *, frame_index: int, frame_duration: Fraction) -> float:
    """``transitions.equirectangular.equirectangular_blur_activity("gaussian_skew", duration)`` at the k-th frame:
    ``T = pts * av_q2d(tb)`` with ``pts = k`` and ``tb`` the frame duration (``settb=expr=1/fps``)."""

    time_seconds = frame_index * (frame_duration.numerator / frame_duration.denominator)
    return _scalar(equirectangular_blur_activity(_GAUSSIAN_ACTIVITY_PROFILE, duration_text), T=time_seconds)


class EquirectGaussian(Transition):
    kind = "equirect_gaussian_blur"

    def apply(self, payload: EquirectGaussianPayload, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
        progress = xfade_progress(ctx.frame_index, ctx.frame_count)
        if progress >= 1.0:
            return a
        activity = blur_activity(payload.duration_text, frame_index=ctx.frame_index, frame_duration=ctx.frame_duration)
        sides = []
        for side in (a, b):
            sides.append(gaussian_blur_side(
                premultiplied_to_code(side), width=ctx.width, height=ctx.height,
                horizontal_strength=payload.horizontal_strength, vertical_strength=payload.vertical_strength,
                activity=activity,
            ))
        out_code = xfade_custom_rgba(parse(payload.ownership_expression), sides[0], sides[1], progress=progress)
        return code_to_premultiplied(out_code)


register(EquirectGaussian())


# ---------------------------------------------------------------------------
# lowering
# ---------------------------------------------------------------------------


def _lower_equirectangular(item: RenderTransition, ctx: LowerContext) -> Lowered:
    if item.xfade_id is None:
        raise reject("transition (other)", f"{item.path}: equirectangular transition without a registry id")
    plan = build_equirectangular_transition_plan(item.xfade_id, dict(item.parameter_values or {}))
    if plan.prefilter == "equirectangular_bloom":
        raise reject(
            "transition (other)",
            f"{item.path}: transition {item.name!r} xfade_id={item.xfade_id!r} resolves to the stock "
            "bloom graph (ffmpeg._build_equirectangular_blur_group: hstack-wrapped gblur + colorlevels + eq "
            "+ curves luminance mask + alphamerge + alpha fades + overlays), not an expression; F2 bloom port pending",
        )
    if plan.prefilter == "equirectangular_gaussian":
        if item.xfade_id not in ADMITTED_EQUIRECT_IDS:
            raise reject("transition (other)", f"{item.path}: {item.xfade_id!r} gaussian graph is not admitted")
        if (plan.activity_profile, plan.ownership_profile) != (_GAUSSIAN_ACTIVITY_PROFILE, _GAUSSIAN_OWNERSHIP_PROFILE):
            raise reject(
                "transition (other)",
                f"{item.path}: {item.xfade_id!r} resolves to profiles ({plan.activity_profile!r}, "
                f"{plan.ownership_profile!r}); the tensor port covers ({_GAUSSIAN_ACTIVITY_PROFILE!r}, "
                f"{_GAUSSIAN_OWNERSHIP_PROFILE!r}) only",
            )
        return Lowered(
            kind="equirect_gaussian_blur",
            xfade_id=item.xfade_id,
            payload=EquirectGaussianPayload(
                xfade_id=item.xfade_id,
                horizontal_strength=plan.strength,
                vertical_strength=plan.spread,
                duration_text=_seconds(ctx.frame_count * ctx.frame_duration),
                ownership_expression=gaussian_ownership_expression(),
            ),
        )
    if plan.prefilter is not None:
        raise reject(
            "transition (other)",
            f"{item.path}: transition {item.name!r} xfade_id={item.xfade_id!r} resolves to prefilter={plan.prefilter!r}",
        )
    if item.xfade_id not in ADMITTED_EQUIRECT_IDS:
        raise reject(
            "transition (other)",
            f"{item.path}: transition {item.name!r} xfade_id={item.xfade_id!r} is a 360° custom "
            "expression but is not in tr_equirect.ADMITTED_EQUIRECT_IDS (needs its golden)",
        )
    return Lowered(kind="xfade_custom", xfade_id=item.xfade_id, payload=XfadePayload(item.xfade_id, plan.expression))


register_handler("equirectangular", _lower_equirectangular)
