"""Phase 5 ports for the prioritized flat blur and movement transitions.

Architecture map
================

    stock.build_stock_transition_plan(item.xfade_id, item.parameter_values)
        -> strict plan-shape and parameter validation at lowering time
        -> Phase5Payload containing only registry-owned constants
        -> one bounded per-frame tensor kernel
             native hblur                 Directional Blur, Cross Blur
             isotropic recursive blur     Gaussian
             bounded rotational taps      Radial
             bounded displacement / shift Flashback, Earthquake
             moving panel                 Drop In
             bounded crop-and-stretch     Smear

Both inputs are already recursively composed project-sized transition surfaces.
The kernels therefore operate on the complete outgoing and incoming sides, not
on one leaf clip. Every working tensor is O(width * height), tap counts are
fixed, and temporal kernels receive an explicit three-frame raw-side window.

Reference boundary
------------------
The implementation consumes the same ``StockTransitionPlan`` that drives the
legacy FFmpeg renderer.  Its timing envelopes and constants are direct ports of
``legacy_ffmpeg/ffmpeg.py``'s flat transition builders.  Directional and Cross
Blur reproduce FFmpeg n8.0's native ``xfade=hblur`` loop. Flashback and
Earthquake reproduce FFmpeg's three-frame ``tmix`` history, including
first-frame cloning and oldest-to-newest weights.

Main callers:
- ``tensor.transitions.lower_transition`` through the registered ``xfade``
  handler delegation below.
- ``tensor.transitions.apply_transition`` through kind ``phase5_transition``.
- ``experimental_tests/core/test_tensor_transition_phase5.py``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any, Callable, Final, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from ..core.model import RenderTransition
from ..core.errors import FCPXMLCompileError
from ..transitions import cohort
from ..transitions.stock import build_stock_transition_plan
from .color import code_to_premultiplied, premultiplied_to_code
from .composite import over
from .expr import quantize_uint8, xfade_progress
from .transitions import (
    ApplyContext,
    LowerContext,
    Lowered,
    Transition,
    register,
    register_xfade_ids,
)
from .support import reject


PHASE5_IDS: Final[tuple[str, ...]] = (
    "cohort_flashback_default",
    "cohort_gaussian_default",
    "cohort_radial_default",
    "cohort_earthquake_default",
    "cohort_drop_in_default",
    "cohort_smear_default",
    "directional_blur_default",
    "cross_blur_default",
)

_NO_PARAMETER_IDS: Final[frozenset[str]] = frozenset(
    {
        "cohort_flashback_default",
        "cohort_gaussian_default",
        "cohort_radial_default",
        "directional_blur_default",
        "cross_blur_default",
    }
)


# The Phase 5 prefilters whose kernels (``_flashback`` / ``_earthquake``) consume
# a three-frame raw side history instead of only the current frame.  This is the
# single source of truth for the temporal capability: ``_lower_phase5_xfade`` turns
# it into ``Lowered.needs_history`` at plan time, and the renderer builds
# ``a_history``/``b_history`` purely off that flag -- it never needs to know which
# prefilters are temporal (that coupling used to live in a ``is_temporal_phase5``
# predicate the renderer imported).
TEMPORAL_PREFILTERS: Final[frozenset[str]] = frozenset(
    {"liquid_ripple", "earthquake_shake", "earthquake_shake_no_smoke"}
)


@dataclass(frozen=True)
class Phase5Payload:
    """One fully validated registry plan, free of source-authored syntax."""

    xfade_id: str
    mode: str
    prefilter: str | None
    strength: float
    spread: float


def _strict_parameter_values(xfade_id: str, values: Mapping[str, Any]) -> None:
    """Reject keys and values outside the exact parameter branches this port runs.

    Main callers: ``_lower_phase5_xfade`` before consulting the stock plan.
    Why this exists: the stock builders assume compiler validation.  Tensor's
    lowerer is also called directly by tests and other Python entry points, so
    it must fail closed instead of treating unknown values as defaults.
    """

    keys = set(values)
    if xfade_id in _NO_PARAMETER_IDS:
        allowed: set[str] = set()
    elif xfade_id == "cohort_earthquake_default":
        allowed = {cohort.EARTHQUAKE_SMOKE_KEY}
    elif xfade_id == "cohort_drop_in_default":
        allowed = {cohort.DROP_IN_SMOKE_KEY}
    elif xfade_id == "cohort_smear_default":
        allowed = {cohort.SMEAR_DIRECTION_KEY}
    else:
        raise AssertionError(f"not a Phase 5 id: {xfade_id}")
    unexpected = keys - allowed
    if unexpected:
        raise reject(
            "transition (other)",
            f"transition {xfade_id!r} has unsupported parameter key(s): {', '.join(sorted(unexpected))}",
        )
    if xfade_id in {"cohort_earthquake_default", "cohort_drop_in_default"} and keys:
        value = next(iter(values.values()))
        if not isinstance(value, bool):
            raise reject(
                "transition (other)",
                f"transition {xfade_id!r} Smoke must be a resolved boolean, got {value!r}",
            )
    if xfade_id == "cohort_smear_default" and keys:
        value = values[cohort.SMEAR_DIRECTION_KEY]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) not in (0.0, 1.0):
            raise reject(
                "transition (other)",
                f"transition {xfade_id!r} direction must be resolved numeric 0 or 1, got {value!r}",
            )


def _lower_phase5_xfade(item: RenderTransition, ctx: LowerContext) -> Lowered:
    assert item.xfade_id in PHASE5_IDS
    values = dict(item.parameter_values or {})
    _strict_parameter_values(item.xfade_id, values)
    try:
        plan = build_stock_transition_plan(item.xfade_id, values)
    except FCPXMLCompileError as exc:
        raise reject("transition (other)", f"{item.path}: {exc}") from exc

    expected = {
        "cohort_flashback_default": ("fade", {"liquid_ripple"}),
        "cohort_gaussian_default": ("fade", {"gaussian_blur"}),
        "cohort_radial_default": ("fade", {"radial_spin"}),
        "cohort_earthquake_default": ("fade", {"earthquake_shake", "earthquake_shake_no_smoke"}),
        "cohort_drop_in_default": ("fade", {"drop_in_panel", "drop_in_panel_no_smoke"}),
        "cohort_smear_default": ("fade", {"smear_streak", "smear_streak_left"}),
        "directional_blur_default": ("hblur", {None}),
        "cross_blur_default": ("hblur", {None}),
    }[item.xfade_id]
    if plan.mode != expected[0] or plan.prefilter not in expected[1] or plan.expression is not None:
        raise reject(
            "transition (other)",
            f"{item.path}: {item.xfade_id!r} resolved to unported plan shape "
            f"mode={plan.mode!r} prefilter={plan.prefilter!r}",
        )
    return Lowered(
        kind="phase5_transition",
        xfade_id=item.xfade_id,
        payload=Phase5Payload(
            xfade_id=item.xfade_id,
            mode=plan.mode,
            prefilter=plan.prefilter,
            strength=plan.strength,
            spread=plan.spread,
        ),
        needs_history=plan.prefilter in TEMPORAL_PREFILTERS,
    )


def _code(canvas: torch.Tensor) -> torch.Tensor:
    return premultiplied_to_code(canvas).round().clamp(0.0, 255.0)


def _from_code(code: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return code_to_premultiplied(code.to(dtype=like.dtype, device=like.device))


def _q(ctx: ApplyContext) -> float:
    return float(np.float32(1.0 - xfade_progress(ctx.frame_index, ctx.frame_count)))


def _smoothstep(raw: float) -> float:
    value = min(1.0, max(0.0, raw))
    return value * value * (3.0 - 2.0 * value)


def _blend_code(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
    return quantize_uint8(a * (1.0 - weight) + b * weight)


def _tmix(history: tuple[torch.Tensor, ...], weights: tuple[float, ...]) -> torch.Tensor:
    """FFmpeg ``tmix`` weighted path for one oldest-to-newest frame window.

    FFmpeg accumulates float samples, scales by the reciprocal weight sum,
    then stores with ``lrintf`` and uint8 clipping. ``torch.round`` supplies
    the same nearest-even tie rule. The renderer owns startup duplication.

    Main callers: ``_flashback_side`` and ``_earthquake_side``.
    """

    if len(history) != len(weights):
        raise ValueError(f"tmix needs {len(weights)} frames, got {len(history)}")
    total = torch.zeros_like(history[0], dtype=torch.float32)
    for frame, weight in zip(history, weights):
        total.add_(frame.to(torch.float32), alpha=weight)
    return torch.round(total * (1.0 / sum(weights))).clamp(0.0, 255.0)


def _history(ctx: ApplyContext, *, incoming: bool) -> tuple[torch.Tensor, ...]:
    history = ctx.b_history if incoming else ctx.a_history
    if len(history) != 3:
        side = "incoming" if incoming else "outgoing"
        raise ValueError(f"temporal transition requires three {side} history frames, got {len(history)}")
    return history


def _history_qs(ctx: ApplyContext) -> tuple[float, float, float]:
    """Local transition progress for the clamped tmix history frames."""

    return tuple(
        float(np.float32(max(0, ctx.frame_index - delta) / ctx.frame_count))
        for delta in (2, 1, 0)
    )


# Rolling cache of processed temporal-history frames, shared across output frames.
#
# Why this exists
# ---------------
# ``_flashback_side`` and ``_earthquake_side`` each turn a three-frame RAW side
# history into three PROCESSED frames (displacement / shake warps plus blurs)
# before ``_tmix`` folds them.  The renderer hands the temporal kernels the SAME
# raw side tensors on consecutive output frames (they come straight out of its
# ``_transition_side_cache``), and a given raw frame is always processed at the
# same local progress ``q`` -- so output frame k and k+1 share two of their three
# processed frames per side.  Rebuilding all three every output frame therefore
# redoes ~2/3 of the (expensive) grid_sample / gblur work.  This cache hands the
# overlapping processed frames back instead of recomputing them.
#
# What makes a cached processed frame reusable
# --------------------------------------------
# A processed frame is a pure function of: the exact raw side tensor, the side it
# belongs to (outgoing vs incoming), the local progress ``q`` it is built at, the
# payload constants, and which kernel built it (flashback vs earthquake).  The raw
# tensor's identity already co-varies with ``q`` (the renderer caches each history
# slot at a fixed local frame, hence a fixed ``q``), but we still key on all of
# these so a cached frame can never be returned for a different side, progress, or
# kernel.
#
# Correctness / invalidation
# ---------------------------
# We key on ``id(canvas)`` for O(1) lookup, but the cache VALUE also stores a
# reference to that raw tensor and every hit re-checks ``stored is canvas``.
# Holding the reference keeps the raw tensor alive while cached, so its ``id``
# cannot be recycled by a different object; if a stale id ever collides, the
# ``is`` check misses and we recompute exactly as the no-cache path did -- output
# stays byte-identical.  The cache is a small bounded LRU: once a raw frame scrolls
# out of the live three-frame window it is evicted and its processed frame dropped.
_PROCESSED_HISTORY_CACHE: "OrderedDict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

# The live working set is 3 frames x 2 sides = 6 per output frame, rising to 4 per
# side (8 total) while two consecutive output frames overlap.  16 comfortably holds
# that window even with a couple of temporal transitions in flight through nested
# scopes; a smaller bound only lowers reuse, never changes any pixel.
_PROCESSED_HISTORY_LIMIT: Final[int] = 16


def _processed_history(
    history: tuple[torch.Tensor, ...],
    qs: tuple[float, ...],
    *,
    kind: str,
    incoming: bool,
    xfade_id: str,
    build: Callable[[torch.Tensor, float], torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Return the processed history frames, reusing overlaps across output frames.

    Step by step: for each ``(raw_frame, q)`` pair, look for an already-built
    processed frame keyed by the raw frame's identity plus its side, progress,
    kernel, and transition id.  If the stored entry really is this raw frame,
    reuse it (and mark it most-recently-used); otherwise run ``build(raw_frame, q)``
    -- the identical per-frame kernel the caller would have run anyway -- cache the
    result under a live reference to the raw frame, and evict the oldest entries
    past the bound.

    Main callers: ``_flashback_side`` and ``_earthquake_side``.
    """

    out: list[torch.Tensor] = []
    for canvas, q in zip(history, qs):
        key = (kind, incoming, id(canvas), q, xfade_id)
        cached = _PROCESSED_HISTORY_CACHE.get(key)
        if cached is not None and cached[0] is canvas:
            _PROCESSED_HISTORY_CACHE.move_to_end(key)
            out.append(cached[1])
            continue
        processed = build(canvas, q)
        _PROCESSED_HISTORY_CACHE[key] = (canvas, processed)
        _PROCESSED_HISTORY_CACHE.move_to_end(key)
        while len(_PROCESSED_HISTORY_CACHE) > _PROCESSED_HISTORY_LIMIT:
            _PROCESSED_HISTORY_CACHE.popitem(last=False)
        out.append(processed)
    return tuple(out)


def _gblur(code: torch.Tensor, *, sigma: float, sigma_v: float, steps: int) -> torch.Tensor:
    """Call the already-goldened recursive blur after transition modules finish importing.

    ``tr_equirect`` imports the transition registry to register its own kind, so
    importing its helper at module load time would form a cycle.  Runtime calls
    happen only after all registrations are complete.
    """

    from .tr_equirect import gblur_gbrap

    return gblur_gbrap(code, sigma=sigma, sigma_v=sigma_v, steps=steps)


def hblur_code(a: torch.Tensor, b: torch.Tensor, *, progress: float) -> torch.Tensor:
    """FFmpeg n8.0 ``xfade=hblur`` on integer code planes.

    ``progress`` is FFmpeg's remaining-outgoing value P.  Its moving window is
    forward-looking and becomes shorter at the right edge.  Accumulation,
    division, and mixing are float32 before the uint8 truncating store.
    """

    if a.shape != b.shape or a.dim() != 3:
        raise ValueError("hblur sides must have equal [C,H,W] shapes")
    width = a.shape[-1]
    p = float(np.float32(progress))
    phase = float(np.float32(p * 2.0 if p <= 0.5 else (1.0 - p) * 2.0))
    size = 1 + int((width // 2) * phase)

    def moving_average(side: torch.Tensor) -> torch.Tensor:
        work = side.to(torch.float32)
        prefix = F.pad(work.cumsum(dim=-1), (1, 0))
        x = torch.arange(width, device=side.device)
        end = (x + size).clamp(max=width)
        sums = prefix.index_select(-1, end) - prefix.index_select(-1, x)
        count = (end - x).to(torch.float32).view(1, 1, width)
        return sums / count

    first = moving_average(a)
    second = moving_average(b)
    return (first * p + second * float(np.float32(1.0 - p))).trunc().clamp(0.0, 255.0)


def _gaussian(payload: Phase5Payload, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    q = _q(ctx)
    if q == 0.0:
        return a
    sigma = max(0.0001, min(ctx.width, ctx.height) * payload.strength)
    activity = math.pow(max(0.0, math.sin(math.pi * q)), 0.65)

    def side(canvas: torch.Tensor) -> torch.Tensor:
        clean = _code(canvas)
        low = _gblur(clean, sigma=sigma * 0.45, sigma_v=sigma * 0.45, steps=3)
        high = _gblur(clean, sigma=sigma, sigma_v=sigma, steps=3)
        shoulder = _blend_code(clean, low, min(1.0, 2.0 * activity))
        return _blend_code(shoulder, high, min(1.0, max(0.0, 2.0 * activity - 1.0)))

    return _from_code(_blend_code(side(a), side(b), q), a)


def _sample_border(code: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    height, width = code.shape[-2:]
    gx = x * (2.0 / max(1, width - 1)) - 1.0
    gy = y * (2.0 / max(1, height - 1)) - 1.0
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
    return F.grid_sample(
        code.to(torch.float32).unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0]


def _coordinate_grid(code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = code.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=code.device, dtype=torch.float32),
        torch.arange(width, device=code.device, dtype=torch.float32),
        indexing="ij",
    )
    return x, y


def _rotation_grid(code: torch.Tensor, angle: float) -> torch.Tensor:
    x, y = _coordinate_grid(code)
    cx, cy = (code.shape[-1] - 1) / 2.0, (code.shape[-2] - 1) / 2.0
    cosine, sine = math.cos(angle), math.sin(angle)
    dx, dy = x - cx, y - cy
    source_x = cosine * dx + sine * dy + cx
    source_y = -sine * dx + cosine * dy + cy
    gx = source_x * (2.0 / max(1, code.shape[-1] - 1)) - 1.0
    gy = source_y * (2.0 / max(1, code.shape[-2] - 1)) - 1.0
    return torch.stack((gx, gy), dim=-1)


def _rotated_tap_sum(code: torch.Tensor, angles: tuple[float, ...]) -> torch.Tensor:
    """Sum fixed rotation taps in bounded batches of four.

    Four 1080p RGBA outputs plus grids stay bounded while reducing eight MPS
    ``grid_sample`` launches to two.  The source expansion is a view.
    """

    total = torch.zeros_like(code, dtype=torch.float32)
    source = code.to(torch.float32).unsqueeze(0)
    for start in range(0, len(angles), 4):
        grids = torch.stack([_rotation_grid(code, angle) for angle in angles[start : start + 4]])
        batch = source.expand(grids.shape[0], -1, -1, -1)
        sampled = F.grid_sample(batch, grids, mode="bilinear", padding_mode="border", align_corners=True)
        total.add_(sampled.sum(dim=0))
    return total


def _radial_side(canvas: torch.Tensor, strength: float, q: float) -> torch.Tensor:
    clean = _code(canvas)
    activity = math.pow(max(0.0, math.sin(math.pi * q)), 0.45)
    # Holding all nine 1080p RGBA taps would add roughly 300 MB without changing
    # the result.  Process the eight rotated taps in bounded groups of four.
    averaged = clean.to(torch.float32).clone()
    angles = tuple(
        multiplier * strength * activity
        for multiplier in (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)
    )
    averaged.add_(_rotated_tap_sum(clean, angles))
    averaged.mul_(1.0 / 9.0)
    blurred = _gblur(averaged, sigma=3.2, sigma_v=3.2, steps=2)
    envelope = math.pow(max(0.0, math.sin(math.pi * q)), 0.35)
    return _blend_code(clean, blurred, envelope)


def _radial(payload: Phase5Payload, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    q = _q(ctx)
    if q == 0.0:
        return a
    handoff = _smoothstep((q - 0.60) / 0.18)
    return _from_code(_blend_code(_radial_side(a, payload.strength, q), _radial_side(b, payload.strength, q), handoff), a)


def _shift_border(code: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    x, y = _coordinate_grid(code)
    return _sample_border(code, x + dx, y + dy)


def _earthquake_envelope(q: float, incoming: bool) -> float:
    if incoming:
        rise_start, rise_duration, fall_start, fall_duration = 0.48, 0.10, 0.66, 0.16
    else:
        rise_start, rise_duration, fall_start, fall_duration = 0.34, 0.12, 0.52, 0.14
    return _smoothstep((q - rise_start) / rise_duration) * (1.0 - _smoothstep((q - fall_start) / fall_duration))


def _earthquake_processed(canvas: torch.Tensor, payload: Phase5Payload, q: float, incoming: bool) -> torch.Tensor:
    """Build one pre-tmix primary/echo frame at its own transition time."""

    clean = _code(canvas)
    envelope = _earthquake_envelope(q, incoming)
    activity = math.pow(max(0.0, envelope), 0.72)
    amplitude_x = max(2, round(clean.shape[-1] * payload.strength))
    amplitude_y = max(2, round(clean.shape[-2] * payload.spread))
    phase = q * math.pi
    primary = _shift_border(clean, amplitude_x * activity * math.sin(19 * phase), amplitude_y * activity * math.sin(27 * phase))
    echo = _shift_border(clean, 0.68 * amplitude_x * activity * math.sin(19 * phase + 0.85), 0.68 * amplitude_y * activity * math.sin(27 * phase + 0.85))
    echo = _gblur(echo, sigma=2.2, sigma_v=2.2, steps=2)
    return torch.round((primary * 4.0 + echo) * 0.2).clamp(0.0, 255.0)


def _earthquake_side(payload: Phase5Payload, ctx: ApplyContext, incoming: bool) -> torch.Tensor:
    history = _history(ctx, incoming=incoming)
    processed = _processed_history(
        history,
        _history_qs(ctx),
        kind="earthquake",
        incoming=incoming,
        xfade_id=payload.xfade_id,
        build=lambda canvas, q: _earthquake_processed(canvas, payload, q, incoming),
    )
    clean = _code(history[-1])
    return _blend_code(clean, _tmix(processed, (1.0, 2.0, 1.0)), _earthquake_envelope(_q(ctx), incoming))


def _screen_white_fog(code: torch.Tensor, q: float) -> torch.Tensor:
    rise = _smoothstep((q - 0.52) / 0.08)
    fall = 1.0 - _smoothstep((q - 0.68) / 0.20)
    amount = rise * fall
    if amount <= 0.0:
        return code
    height = code.shape[-2]
    y = torch.arange(height, device=code.device, dtype=code.dtype).view(1, height, 1)
    alpha = (145.0 / 255.0) * torch.clamp((y / height - 0.62) / 0.38, min=0.0).pow(1.4) * amount
    rgb = 255.0 - (255.0 - code[:3]) * (1.0 - alpha)
    return torch.cat((rgb, code[3:4]), dim=0).clamp(0.0, 255.0)


def _earthquake(payload: Phase5Payload, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    q = _q(ctx)
    if q == 0.0:
        return a
    handoff = _smoothstep((q - 0.50) / 0.10)
    out = _blend_code(_earthquake_side(payload, ctx, False), _earthquake_side(payload, ctx, True), handoff)
    if payload.prefilter == "earthquake_shake":
        out = _screen_white_fog(out, q)
    return _from_code(out, a)


def _bounce_y(q: float, height: int) -> float:
    if q < 0.13:
        return -height * (1.0 - _smoothstep(q / 0.13))
    if q < 0.20:
        return 0.45 * height * _smoothstep((q - 0.13) / 0.07)
    if q < 0.33:
        return height * (0.45 - 0.63 * _smoothstep((q - 0.20) / 0.13))
    if q < 0.45:
        return -0.18 * height * (1.0 - _smoothstep((q - 0.33) / 0.12))
    return 0.0


def _translate_transparent(canvas: torch.Tensor, dy: int) -> torch.Tensor:
    out = torch.zeros_like(canvas)
    height = canvas.shape[-2]
    if abs(dy) >= height:
        return out
    if dy >= 0:
        out[:, dy:] = canvas[:, : height - dy]
    else:
        out[:, : height + dy] = canvas[:, -dy:]
    return out


def _drop_in(payload: Phase5Payload, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    q = _q(ctx)
    if q == 0.0:
        return a
    raw = min(1.0, max(0.0, q / payload.strength))
    blur_weight = math.pow(max(0.0, math.sin(math.pi * raw)), 0.70)
    code = _code(b)
    blurred = _gblur(code, sigma=1.2, sigma_v=12.0, steps=2)
    panel = _from_code(_blend_code(code, blurred, blur_weight), b)
    dy = int(_bounce_y(q, ctx.height))
    background = a
    if payload.prefilter == "drop_in_panel":
        echo_code = _gblur(code, sigma=1.4, sigma_v=18.0, steps=2)
        echo_fade = 1.0 - _smoothstep((q - 0.22) / 0.18)
        echo = _from_code(torch.cat((echo_code[:3], echo_code[3:4] * (0.28 * echo_fade)), dim=0), b)
        background = over(background, _translate_transparent(echo, dy - round(ctx.height * 0.10)))
        shadow = torch.cat((torch.zeros_like(b[:3]), b[3:4] * 0.30), dim=0)
        background = over(background, _translate_transparent(shadow, dy + round(ctx.height * 0.025)))
    result = over(background, _translate_transparent(panel, dy))
    if payload.prefilter == "drop_in_panel":
        fog_in = _smoothstep((q - 0.15) / 0.06)
        fog_out = 1.0 - _smoothstep((q - 0.35) / 0.20)
        fog_amount = fog_in * fog_out
        if fog_amount > 0.0:
            result_code = _code(result)
            y = torch.arange(ctx.height, device=result.device, dtype=result_code.dtype).view(1, ctx.height, 1)
            fog_alpha = ((55.0 + 80.0 * torch.clamp((y / ctx.height - 0.45) / 0.55, min=0.0).pow(1.2)) / 255.0) * fog_amount
            rgb = result_code[:3] * (1.0 - fog_alpha) + 255.0 * fog_alpha
            result = _from_code(torch.cat((rgb, result_code[3:4]), dim=0), result)
    return result


def _flashback_displaced(canvas: torch.Tensor, payload: Phase5Payload, q: float) -> torch.Tensor:
    """Build the calibrated displacement frame before FFmpeg's temporal mix."""

    clean = _code(canvas)
    x, y = _coordinate_grid(clean)
    activity = math.pow(max(0.0, math.sin(math.pi * q)), 0.75)
    amp_x = min(96.0, clean.shape[-1] * payload.strength)
    amp_y = min(96.0, clean.shape[-2] * payload.spread)
    dx = amp_x * activity * (0.65 * torch.sin(0.045 * y + 8 * math.pi * q) + 0.35 * torch.sin(0.025 * (x + y) - 6 * math.pi * q))
    dy = amp_y * activity * (0.62 * torch.sin(0.042 * x - 9 * math.pi * q) + 0.38 * torch.sin(0.022 * (x - y) + 7 * math.pi * q))
    # ``displace,format=rgba`` materializes an 8-bit frame before tmix.
    return torch.round(_sample_border(clean, x + dx, y + dy)).clamp(0.0, 255.0)


def _flashback_side(payload: Phase5Payload, ctx: ApplyContext, incoming: bool) -> torch.Tensor:
    history = _history(ctx, incoming=incoming)
    displaced = _processed_history(
        history,
        _history_qs(ctx),
        kind="flashback",
        incoming=incoming,
        xfade_id=payload.xfade_id,
        build=lambda canvas, q: _flashback_displaced(canvas, payload, q),
    )
    clean = _code(history[-1])
    activity = math.pow(max(0.0, math.sin(math.pi * _q(ctx))), 0.75)
    return _blend_code(clean, _tmix(displaced, (3.0, 2.0, 1.0)), activity)


def _flashback(payload: Phase5Payload, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    q = _q(ctx)
    if q == 0.0:
        return a
    handoff = _smoothstep((q - 0.32) / 0.18)
    out = _blend_code(_flashback_side(payload, ctx, False), _flashback_side(payload, ctx, True), handoff)
    x, y = _coordinate_grid(out)
    radius = (0.08 + 0.62 * q) * min(ctx.width, ctx.height)
    cx, cy = ctx.width * (0.32 + 0.25 * q), ctx.height * (0.58 - 0.20 * q)
    ring = 86.0 * torch.exp(-torch.abs(torch.hypot(x - cx, y - cy) - radius) / max(1.0, 0.055 * min(ctx.width, ctx.height))) * math.pow(max(0.0, math.sin(math.pi * q)), 1.2)
    rgb = 255.0 - (255.0 - out[:3]) * (1.0 - ring.clamp(0.0, 255.0).unsqueeze(0) / 255.0)
    return _from_code(torch.cat((rgb, out[3:4]), dim=0), a)


def _smear_side(canvas: torch.Tensor, active_fraction: float, q: float, left: bool) -> torch.Tensor:
    clean = _code(canvas)
    width, height = clean.shape[-1], clean.shape[-2]
    preserved_width = max(2, round(width * 0.34))
    slice_width = max(2, round(width * 0.055))
    streak_width = width - preserved_width
    if left:
        preserved = clean[..., width - preserved_width :]
        slice_x = max(0, width - preserved_width - slice_width)
    else:
        preserved = clean[..., :preserved_width]
        slice_x = min(width - slice_width, preserved_width)
    streak = clean[..., slice_x : slice_x + slice_width]
    streak = F.interpolate(streak.unsqueeze(0), size=(height, streak_width), mode="bilinear", align_corners=False)[0]
    streak = _gblur(streak, sigma=10.0, sigma_v=0.8, steps=2)
    maximum = torch.cat((streak, preserved), dim=-1) if left else torch.cat((preserved, streak), dim=-1)
    local = min(1.0, max(0.0, q / active_fraction))
    if local < 0.68:
        envelope = _smoothstep((local - 0.25) / 0.35)
    else:
        envelope = _smoothstep((1.0 - local) / 0.32)
    return _blend_code(clean, maximum, envelope)


def _smear(payload: Phase5Payload, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    q = _q(ctx)
    if q == 0.0:
        return a
    handoff = _smoothstep((q - max(0.0, payload.strength - 0.05)) / 0.05)
    outgoing = _smear_side(a, payload.strength, q, payload.prefilter == "smear_streak_left")
    return _from_code(_blend_code(outgoing, _code(b), handoff), a)


class Phase5(Transition):
    """The prioritized blur / warp cohort: one apply that dispatches on the lowered prefilter.

    The ``mode``/``prefilter`` -> kernel selection below is intra-family subroutine dispatch
    (every branch is the same ``(payload, A, B, ctx)`` kernel), not a separate transition path.
    Temporal prefilters read ``ctx.a_history``/``b_history``; the renderer supplies those only
    because ``_lower_phase5_xfade`` set ``Lowered.needs_history`` for them.
    """

    kind = "phase5_transition"

    def apply(self, payload: Phase5Payload, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
        if payload.mode == "hblur":
            progress = xfade_progress(ctx.frame_index, ctx.frame_count)
            if progress >= 1.0:
                return a
            return _from_code(hblur_code(_code(a), _code(b), progress=progress), a)
        return {
            "gaussian_blur": _gaussian,
            "radial_spin": _radial,
            "earthquake_shake": _earthquake,
            "earthquake_shake_no_smoke": _earthquake,
            "drop_in_panel": _drop_in,
            "drop_in_panel_no_smoke": _drop_in,
            "liquid_ripple": _flashback,
            "smear_streak": _smear,
            "smear_streak_left": _smear,
        }[payload.prefilter](payload, a, b, ctx)


register(Phase5())

# Phase-5 specializes the ``xfade`` handler for its own ids.  This is an explicit
# id-route registration (transitions._xfade_handler consults XFADE_ID_ROUTES), replacing
# the former monkeypatch that reassigned HANDLER_LOWERERS["xfade"]; the non-Phase-5 xfade
# ids stay on the stock custom-expression path.
register_xfade_ids(PHASE5_IDS, _lower_phase5_xfade)
