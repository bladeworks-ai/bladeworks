"""F4 transition ports: handler transitions (owner: F4 batch).

Architecture map
================

    handler "fade_color"  -> kind "xfade_custom"   (transitions.XfadePayload; the registry's
                             3-phase expression from transitions.stock.build_fade_color_plan)
    handler "wipe"        -> kind "wipe"           (WipePayload: direction + geq divisor)
    handler "slide_push"  -> kind "slide_push"     (SlidePushPayload: direction, push, motion
                             origin, avgblur radius)

Every port validates its authored parameters the way the CPU emitter reads them
(``ffmpeg._parameter_int`` / ``_transition_color_components``), rejects loudly through
``support.reject`` where the emitter would silently default or clamp, and lowers to a frozen
payload.  Apply kernels receive the two PLACED sides (premultiplied linear RGBA on the project
canvas, ``a`` outgoing / ``b`` incoming) and return the pair's composed canvas.

Fade to Color (reference: ``ffmpeg._build_stock_transition_groups``)
--------------------------------------------------------------------
A stock transition group: both sides are composed, adapted to 8-bit straight ``gbrap`` and run
through ``xfade=transition=custom`` with ``build_fade_color_plan(rgb).expression`` (reach the
colour at 11/30 progress, hold to 19/30, reveal).  Lowered to the existing ``xfade_custom`` kind:
the very registry string is evaluated by ``tensor/expr.py`` (GBRA plane order, float32 ``P``,
uint8 store), so the port cannot drift from the reference's expression text.  Colour handling
mirrors ``ffmpeg._transition_color_components`` (key ``"3"`` / name ``"Color"``, absent -> black,
``round(c * 255)`` clamped to 0..255) except that a malformed value -- which the emitter silently
turns into black -- is a loud reject here.

*Plane-order history (ledger row 8, RESOLVED):* the registry literal used to be authored as
``if(eq(PLANE,0),R,if(eq(PLANE,1),G,if(eq(PLANE,2),B,255)))`` while the xfade runs on ``gbrap``
planes (0=G, 1=B, 2=R), so an authored ``(R, G, B)`` rendered as ``(B, R, G)`` -- a red-orange
``1,0.2,0.1`` fade came out bright green.  Only grey survived the permutation, which is why every
calibrated fixture missed it.  ``build_fade_color_plan`` now permutes the triplet into plane order
and both backends moved together, because the tensor port evaluates that same emitted string.

Wipe (reference: ``ffmpeg._video_chain``, ``clip.transition_in.handler == "wipe"``)
------------------------------------------------------------------------------------
NOT a stock group: the incoming leaf gets, after its spatial stage and in its own 8-bit
``rgba`` stream, ``geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='alpha(X,Y)*(<visible>)'`` with
``progress = min(max(T/(duration),0),1)`` and

    direction 0 (Left):  visible = gte(X/W, 1-progress)
    direction 1 (Up):    visible = gte(Y/H, 1-progress)
    direction 2 (Right): visible = lte(X/W, progress)
    direction 3 (Down):  visible = lte(Y/H, progress)

then the ordinary calibrated overlay pass reveals it over the outgoing leaf.  ``T`` is the
incoming stream's frame time: its frame 0 is the transition's owned first frame
(``_expanded_schedule``), so ``T = k * av_q2d(frame_duration)`` for the k-th owned window frame
and geq evaluates everything in doubles.  ``wipe_visible_range`` reproduces that double
arithmetic on the CPU (numpy float64) and yields the first / last visible pixel index, so the
mask is exact on every device (no float32 knife-edge drift on MPS).  Kernel:
``over(a, b * mask)`` -- equivalent to the reference's per-clip alpha because the two
participants are consecutive in the stack (premultiplied source-over is associative).

Slide / Push (reference: ``ffmpeg._overlay_position`` + ``_slide_push_motion_blur_filters``)
--------------------------------------------------------------------------------------------
Two per-clip manipulations on ordinary layers:

* placement: the incoming leaf's overlay position gets ``-main_w*(1-progress)`` (direction 0),
  ``+main_h*(1-progress)`` (1), ``+main_w*(1-progress)`` (2), ``-main_h*(1-progress)`` (3) with
  ``progress = min(max((t-start)/(duration),0),1)`` on the *timeline* clock (``t`` of the
  project-canvas base at output frame ``n``); in Push mode (key ``"5"`` == 2) the outgoing
  leaf additionally moves ``+main_w*progress`` / ``-main_h*progress`` / ``-main_w*progress`` /
  ``+main_h*progress``.  ``vf_overlay`` truncates the evaluated position toward zero
  (``normalize_xy``: ``(int)d``), so the shift is an exact integer per frame.
* motion blur: ``avgblur=sizeX=K:sizeY=1`` (directions 0/2) or ``sizeX=1:sizeY=K`` (1/3) with
  ``planes=0x7`` on the moving leaf's 8-bit straight ``rgba`` stream, enabled over the owned
  window; ``K = max(2, round(dimension / max(1, duration*fps) * 0.75))`` -- resolution-scaled.
  NB ``sizeY=1`` is a real 3-tap box on the other axis (avgblur radius semantics; verified
  against the CLI); FFmpeg n8.0's ``vf_avgblur`` is one integer 2-D box with edge-replicated
  taps and a floor-division store (``avgblur_code`` is bit-exact against it).

*Reference regression (ledger candidate; NOT followed):* since ``35c44314b`` ("unify spatial
geometry and composition") ``ffmpeg._compose_item_batch`` overrides every ordinary layer's
overlay position with ``layer.surface.origin`` (``x='0':y='0'``), so the placement motion
above is dead code in the emitted graph: the emitted reference renders Slide/Push as a hard
cut to the (blurred) incoming leaf.  That is not the calibrated behaviour
(FFMPEG_BACKEND_SUPPORT.md: "All eight directions use correct ownership and resolution-scaled
motion blur; luma SSIM 0.802-0.908" vs Final Cut, measured before the regression), so this port
implements the placement the emitter's source text specifies and the golden test compares
against that graph (the emitted filter script with the dead-code override undone).

Time base of a port: ``ctx.frame_index`` is ``k`` of ``ctx.frame_count`` ``F`` owned window
frames; ``SlidePushPayload.first_frame`` carries the window's first output frame (computed at
lower time with the same ``resolve_owned_frame_window`` the plan uses) so the timeline clock
``t = (first_frame + k) * frame_duration`` is available at apply time.

Main callers:
- ``tensor/transitions.py`` imports this module (registration on import);
  ``plan.build_tensor_plan`` lowers through ``lower_transition``; ``renderer._FrameComposer``
  applies through ``apply_transition``.
- ``experimental_tests/core/test_tensor_transition_handlers.py`` (goldens).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import numpy as np
import torch

from ..core.model import Parameter, RenderTransition
from ..core.retime_execution import resolve_owned_frame_window
from ..transitions.stock import build_fade_color_plan
from .color import code_to_premultiplied, premultiplied_to_code
from .composite import over
from .support import reject
from .transitions import (
    ApplyContext,
    Lowered,
    LowerContext,
    Transition,
    XfadePayload,
    register,
    register_handler,
)

# ---------------------------------------------------------------------------
# Parameter access (mirrors ffmpeg._parameter_values / _parameter_int, loud where it defaults)
# ---------------------------------------------------------------------------


def _parameter_values(params: tuple[Parameter, ...]) -> dict[str, str]:
    """``ffmpeg._parameter_values``: authored values by FCPXML key and by name."""

    values: dict[str, str] = {}
    for param in params:
        if param.value is None:
            continue
        if param.key:
            values[param.key] = param.value
        if param.name:
            values[param.name] = param.value
    return values


def _parameter_choice(item: RenderTransition, *, key: str, label: str, allowed: tuple[int, ...], default: int) -> int:
    """An integer popup parameter read like ``ffmpeg._parameter_int(params, keys=(key,), default)``.

    The emitter falls back to ``default`` for an unparsable value and lets any other integer
    through to its ``else`` branch; both are loud rejects here.
    """

    values = _parameter_values(item.params)
    if key not in values:
        return default
    raw = values[key]
    try:
        value = int(raw.split()[0])
    except (ValueError, IndexError):
        raise reject(
            "transition (other)",
            f"{item.path}: {item.name!r} {label} (key {key}) value {raw!r} is not an integer",
        ) from None
    if value not in allowed:
        raise reject(
            "transition (other)",
            f"{item.path}: {item.name!r} {label} (key {key}) value {value} is outside the calibrated set {allowed}",
        )
    return value


# ---------------------------------------------------------------------------
# Fade to Color
# ---------------------------------------------------------------------------


def fade_color_components(item: RenderTransition) -> tuple[int, int, int]:
    """The 8-bit RGB triplet ``ffmpeg._transition_color_components`` derives, loud on malformed input.

    Key ``"3"`` (else name ``"Color"``), absent -> ``"0,0,0"``; comma or space separated
    floats, first three components, ``round(c * 255)`` clamped to 0..255 (values outside 0..1
    clamp exactly like the emitter; the compiler reports that clamp).  A non-numeric or short
    value -- black in the emitter -- rejects.
    """

    values = _parameter_values(item.params)
    raw = values.get("3") or values.get("Color") or "0,0,0"
    try:
        components = [float(piece) for piece in raw.replace(",", " ").split()[:3]]
    except ValueError:
        raise reject("transition (other)", f"{item.path}: Fade to Color value {raw!r} is not numeric") from None
    if len(components) < 3:
        raise reject("transition (other)", f"{item.path}: Fade to Color value {raw!r} has fewer than 3 components")
    red, green, blue = (max(0, min(255, round(component * 255))) for component in components)
    return red, green, blue


def _lower_fade_color(item: RenderTransition, ctx: LowerContext) -> Lowered:
    plan = build_fade_color_plan(fade_color_components(item))
    assert plan.mode == "custom" and plan.expression is not None and plan.prefilter is None
    return Lowered(kind="xfade_custom", payload=XfadePayload(xfade_id="fade_color", expression=plan.expression))


register_handler("fade_color", _lower_fade_color)


# ---------------------------------------------------------------------------
# Shared clocks (the doubles ffmpeg computes)
# ---------------------------------------------------------------------------


def _q2d(value: Fraction) -> float:
    """``av_q2d`` / the expression parser's ``num/den``: one correctly rounded double division."""

    return value.numerator / value.denominator


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Wipe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WipePayload:
    direction: int          # key "13": 0 Left, 1 Up, 2 Right, 3 Down
    duration: Fraction      # the geq's ``T/(duration)`` divisor (transition.duration)


def wipe_visible_range(payload: WipePayload, *, frame_index: int, frame_duration: Fraction, width: int, height: int) -> tuple[int, int, int]:
    """``(axis, first, end)``: the visible half-open index range ``[first, end)`` along ``axis`` (0 = x, 1 = y).

    Reproduces the reference geq in doubles: ``T = k * av_q2d(frame_duration)``,
    ``progress = min(max(T/duration, 0), 1)``, then ``gte(X/W, 1-progress)`` /
    ``lte(X/W, progress)`` (or ``Y/H``) per pixel index -- so the knife-edge columns land on
    ffmpeg's side of the comparison on every device.
    """

    frame_time = np.float64(frame_index) * np.float64(_q2d(frame_duration))
    progress = np.float64(_clamp01(float(frame_time / np.float64(_q2d(payload.duration)))))
    axis = 0 if payload.direction in (0, 2) else 1
    size = width if axis == 0 else height
    coordinate = np.arange(size, dtype=np.float64) / np.float64(size)
    if payload.direction in (0, 1):
        visible = coordinate >= (np.float64(1.0) - progress)
    else:
        visible = coordinate <= progress
    indices = np.flatnonzero(visible)
    if indices.size == 0:
        return axis, 0, 0
    first, last = int(indices[0]), int(indices[-1])
    assert last - first + 1 == indices.size, "wipe visibility is a contiguous range by construction"
    return axis, first, last + 1


class Wipe(Transition):
    kind = "wipe"

    def apply(self, payload: WipePayload, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
        axis, first, end = wipe_visible_range(
            payload, frame_index=ctx.frame_index, frame_duration=ctx.frame_duration, width=ctx.width, height=ctx.height
        )
        size = ctx.width if axis == 0 else ctx.height
        index = torch.arange(size, device=b.device)
        mask = ((index >= first) & (index < end)).to(b.dtype)
        mask = mask.view(1, 1, size) if axis == 0 else mask.view(1, size, 1)
        return over(a, b * mask)


def _lower_wipe(item: RenderTransition, ctx: LowerContext) -> Lowered:
    direction = _parameter_choice(item, key="13", label="Direction", allowed=(0, 1, 2, 3), default=0)
    return Lowered(kind="wipe", payload=WipePayload(direction=direction, duration=item.duration))


register(Wipe())
register_handler("wipe", _lower_wipe)


# ---------------------------------------------------------------------------
# Slide / Push
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlidePushPayload:
    direction: int              # key "4": 0 Left, 1 Up, 2 Right, 3 Down
    push: bool                  # key "5" == 2: the outgoing leaf moves too
    absolute_start: Fraction    # transition start: origin of the overlay progress clock
    duration: Fraction
    first_frame: int            # owned window's first output frame (t = (first_frame + k) * frame_duration)
    kernel: int                 # avgblur radius (resolution-scaled)


def slide_push_kernel(*, duration: Fraction, frame_duration: Fraction, width: int, height: int, direction: int) -> int:
    """``ffmpeg._slide_push_motion_blur_filters``: ``max(2, round(dimension / max(1, duration*fps) * 0.75))``."""

    dimension = width if direction in (0, 2) else height
    travel_frames = max(1.0, float(duration * (1 / frame_duration)))
    return max(2, round(dimension / travel_frames * 0.75))


def slide_push_progress(payload: SlidePushPayload, *, frame_index: int, frame_duration: Fraction) -> float:
    """The overlay's ``min(max((t-start)/(duration),0),1)`` at output frame ``first_frame + k`` (doubles)."""

    frame_time = float(payload.first_frame + frame_index) * _q2d(frame_duration)
    return _clamp01((frame_time - _q2d(payload.absolute_start)) / _q2d(payload.duration))


def slide_push_shifts(payload: SlidePushPayload, progress: float, *, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """``((incoming_dx, incoming_dy), (outgoing_dx, outgoing_dy))`` in whole pixels for one frame.

    The emitter's overlay expressions (base position 0 on the project canvas) followed by
    ``vf_overlay``'s ``(int)`` truncation toward zero; the outgoing shift is (0, 0) in Slide mode.
    """

    w, h, d = float(width), float(height), payload.direction
    remaining = 1.0 - progress
    if d == 0:
        incoming, outgoing = (-(w * remaining), 0.0), (w * progress, 0.0)
    elif d == 1:
        incoming, outgoing = (0.0, h * remaining), (0.0, -(h * progress))
    elif d == 2:
        incoming, outgoing = (w * remaining, 0.0), (-(w * progress), 0.0)
    else:
        incoming, outgoing = (0.0, -(h * remaining)), (0.0, h * progress)
    if not payload.push:
        outgoing = (0.0, 0.0)
    return (int(incoming[0]), int(incoming[1])), (int(outgoing[0]), int(outgoing[1]))


def shift_canvas(canvas: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate a ``[C, H, W]`` canvas by whole pixels; uncovered pixels are transparent (zeros)."""

    if dx == 0 and dy == 0:
        return canvas
    _, height, width = canvas.shape
    out = torch.zeros_like(canvas)
    if abs(dx) >= width or abs(dy) >= height:
        return out
    src_x = slice(max(0, -dx), width - max(0, dx))
    dst_x = slice(max(0, dx), width - max(0, -dx))
    src_y = slice(max(0, -dy), height - max(0, dy))
    dst_y = slice(max(0, dy), height - max(0, -dy))
    out[:, dst_y, dst_x] = canvas[:, src_y, src_x]
    return out


def avgblur_code(code_rgb: torch.Tensor, *, size_x: int, size_y: int) -> torch.Tensor:
    """``avgblur=sizeX=size_x:sizeY=size_y`` on 8-bit code planes ``[3, H, W]`` (FFmpeg n8.0 ``vf_avgblur.c``).

    One integer 2-D box: radii ``min(size_x, W // 2)`` / ``min(size_y, H // 2)`` (``sizeY <= 0``
    inherits ``sizeX``), edge-replicated taps at the borders (the ``col_sum`` pads clamp to the
    first / last column and row), area ``(2rx+1)(2ry+1)`` and a floor division store (``lut[sum]``
    = ``sum // area``).  Integer inputs keep the box sums exact in float32 (< 2**24), so the
    result is bit-exact against the CLI (``test_tensor_transition_handlers.py``).
    """

    if code_rgb.dim() != 3:
        raise ValueError(f"expected [C, H, W] code planes, got {tuple(code_rgb.shape)}")
    _, height, width = code_rgb.shape
    radius_x = min(size_x, width // 2)
    radius_y = min(size_y if size_y > 0 else size_x, height // 2)
    padded = torch.nn.functional.pad(
        code_rgb.unsqueeze(0), (radius_x, radius_x, radius_y, radius_y), mode="replicate"
    )[0]
    integral = torch.nn.functional.pad(padded.cumsum(dim=1).cumsum(dim=2), (1, 0, 1, 0))
    span_x, span_y = 2 * radius_x + 1, 2 * radius_y + 1
    sums = (
        integral[:, span_y:, span_x:] - integral[:, :-span_y, span_x:]
        - integral[:, span_y:, :-span_x] + integral[:, :-span_y, :-span_x]
    )
    area = float(span_x * span_y)
    return (sums / area).floor().clamp(0.0, 255.0)


def motion_blur(canvas: torch.Tensor, *, direction: int, kernel: int) -> torch.Tensor:
    """The moving leaf's shutter blur on a placed premultiplied linear canvas.

    Round trip through the reference's 8-bit straight ``rgba`` leaf domain
    (``premultiplied_to_code``, rounded to the 8-bit store the leaf stream is at that stage),
    ``avgblur`` on the RGB planes only (``planes=0x7``; alpha untouched), back to premultiplied
    linear.
    """

    code = premultiplied_to_code(canvas)
    rgb = code[:3].round()
    if direction in (0, 2):
        rgb = avgblur_code(rgb, size_x=kernel, size_y=1)
    else:
        rgb = avgblur_code(rgb, size_x=1, size_y=kernel)
    return code_to_premultiplied(torch.cat((rgb, code[3:4]), dim=0))


class SlidePush(Transition):
    kind = "slide_push"

    def apply(self, payload: SlidePushPayload, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
        progress = slide_push_progress(payload, frame_index=ctx.frame_index, frame_duration=ctx.frame_duration)
        (in_dx, in_dy), (out_dx, out_dy) = slide_push_shifts(payload, progress, width=ctx.width, height=ctx.height)
        incoming = shift_canvas(motion_blur(b, direction=payload.direction, kernel=payload.kernel), in_dx, in_dy)
        outgoing = a
        if payload.push:
            outgoing = shift_canvas(motion_blur(a, direction=payload.direction, kernel=payload.kernel), out_dx, out_dy)
        return over(outgoing, incoming)


def _lower_slide_push(item: RenderTransition, ctx: LowerContext) -> Lowered:
    direction = _parameter_choice(item, key="4", label="Direction", allowed=(0, 1, 2, 3), default=0)
    # The registry admits Mode "2" (Push); the calibration's Slide cases author no Mode at all
    # (emitter default 0).  Any other authored value is outside both and rejects.
    mode = _parameter_choice(item, key="5", label="Mode", allowed=(0, 2), default=0)
    window = resolve_owned_frame_window(item.absolute_start, item.end, frame_duration=ctx.frame_duration)
    if window.frame_count != ctx.frame_count:
        raise AssertionError(
            f"{item.path}: owned window {window.frame_count} frames differs from the plan's {ctx.frame_count}"
        )
    return Lowered(
        kind="slide_push",
        payload=SlidePushPayload(
            direction=direction,
            push=(mode == 2),
            absolute_start=item.absolute_start,
            duration=item.duration,
            first_frame=window.first_frame,
            kernel=slide_push_kernel(
                duration=item.duration, frame_duration=ctx.frame_duration,
                width=ctx.width, height=ctx.height, direction=direction,
            ),
        ),
    )


register(SlidePush())
register_handler("slide_push", _lower_slide_push)


__all__ = [
    "SlidePushPayload",
    "WipePayload",
    "avgblur_code",
    "fade_color_components",
    "motion_blur",
    "shift_canvas",
    "slide_push_kernel",
    "slide_push_progress",
    "slide_push_shifts",
    "wipe_visible_range",
]
