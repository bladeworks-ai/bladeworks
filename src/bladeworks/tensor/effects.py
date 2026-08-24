"""Effect ports: the append-only registry of ported effect handlers (E1-E6).

Architecture map
================

    plan.build_tensor_plan
        -> lower_effect(effect, ctx)      : ResolvedEffect -> EffectSpec (loud reject when no port
                                            is registered for the handler, or the port refuses the
                                            authored parameters)
    renderer._FrameComposer.placed
        -> apply_effects(canvas, specs, frame=n)
             -> port.apply(payload, canvas, ApplyContext)   in registration order per layer

    fx_basic.py  (E1: simple + LUT + blur/sharpen/vignette + Color Curves no-ops + Color Wheels)
    fx_warp.py   (E2: directional/radial blur, fisheye, droplet, crop&feather, vignette mask,
                  kaleidoscope, perspective tile, vibrancy)
    fx_color.py  (E4: Color Adjustments through the BT.601 limited-range YUVA444P bridge,
                  Color Board)
    fx_mask.py   (portable mask mattes and inside/outside branch compositing)
    fx_keyer.py  (Green Screen Keyer colorkey/despill approximation)
    this module  (Earthquake, the first port; the registry itself)

Where an effect sits in the layer pipeline (E6, mirrors ``ffmpeg._video_chain``): the
reference emits ``_ordered_effect_filters`` *after* crop/conform (``initial_filters``) and
*before* the spatial tail (corner pin / transform / animation), on the conformed
project-space canvas.  ``renderer.placed`` therefore warps conform -> effects -> composed
whenever ``layer.effects`` is non-empty.  Group effects (Yunah's Earthquake on a compound)
are folded onto each leaf's conformed canvas the same way (``plan._effect_specs``).

Port contract
-------------
* ``lower(effect, ctx) -> payload``: runs at plan time; reads ``effect.params`` /
  ``effect.calibration`` / ``effect.parameter_values`` exactly like the CPU emitter for the
  same handler (``ffmpeg._effect_filters`` -> ``basic_effects.py`` / ``cohort_effects.py`` /
  ``effects/color_adjustments.py``), and raises ``support.reject("effect (unsupported
  parameters)", ...)`` for any authored parameter it cannot honour.  The payload is the
  port's own frozen dataclass (hashable, no torch objects).
* ``apply(payload, canvas, ctx) -> canvas``: ``canvas`` is a premultiplied *linear* RGBA
  ``[4, H, W]`` float32 tensor (the working space; see ``tensor/__init__``).  Ports that
  emulate 8-bit code-space filters (almost all of them: ``geq``, ``lut``, ``gblur``,
  ``eq`` ...) round-trip through ``color.premultiplied_to_code`` /
  ``color.code_to_premultiplied`` (straight 0..255 encoded RGBA, the ``format=rgba``
  domain the reference feeds them) and may ``expr.quantize_uint8`` at their exit to
  mirror the 8-bit link between chained filters.  ``ctx.frame`` is the layer-local frame
  counter (``N`` in ``geq``), ``ctx.seconds`` its time (``T``).
* Ports never edit this file; they ``register(EffectPort(...))`` from their own module and
  this module imports the port modules at the bottom.  Registering a handler twice is an
  error (one owner per handler).

Main callers:
- ``plan.build_tensor_plan`` (``lower_effect`` for leaf effects, ``_effect_specs`` for
  folded group effects).
- ``renderer._FrameComposer.placed`` (``apply_effects``).

Why this exists:
The skeleton hard-coded one effect kind (Earthquake) in the plan and the renderer; the
E1/E2/E4 batches each add ~10 handlers in parallel.  A registry keyed by the CPU handler
id keeps every batch in its own file, keeps "what is ported" answerable from one table
(``EFFECT_PORTS`` + ``support.py``), and keeps the reject loud and per-handler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Callable, Final, Mapping, Optional

import torch
import torch.nn.functional as F

from ..core.model import ResolvedEffect
from ..core.retime import RetimeMap
from .fx_mask import MaskEffectPayload, apply_masked_effect
from .fx_keyer import lower_green_screen_keyer, green_screen_key
from .support import reject


@dataclass(frozen=True)
class LowerContext:
    """What a port may know about the layer at plan time."""

    clip_path: str
    width: int          # conformed canvas width the effect runs on (project / container width)
    height: int         # conformed canvas height
    frame_duration: Fraction
    clip_duration: Fraction
    # Mask keyframes are authored on the clip's source clock. Runtime effect
    # time is local output time, so masked ports need this exact affine map.
    source_start: Fraction = Fraction(0)
    playback_rate: Fraction = Fraction(1)
    retime_map: Optional[RetimeMap] = None
    coordinate_scale_x: float = 1.0
    coordinate_scale_y: float = 1.0
    # Source stream colour tags (ffprobe names; "unknown" for rasters / untagged): libavfilter
    # negotiates the yuva444p link around YUV-native filters (eq / hue / unsharp / colorize)
    # from the SOURCE's tags, so bridge ports need them (E1 / E4).
    source_colorspace: str = "unknown"
    source_color_range: str = "unknown"
    # The pixel link libavfilter negotiates for the reference's effect stack on this layer,
    # ``"<pix_fmt>:<colorspace>:<range>"`` (``fx_color.reference_effect_link``: source tags +
    # whether a ``perspective`` geometry stage precedes the effects).  ``None`` only in direct
    # port unit tests; the plan always sets it.
    reference_effect_link: Optional[str] = None


@dataclass(frozen=True)
class ApplyContext:
    """Per-frame facts for ``EffectPort.apply``."""

    frame: int          # layer-local frame counter (``N``): 0 at the layer's frame origin
    seconds: float      # ``T`` = frame * frame_duration
    width: int
    height: int


@dataclass(frozen=True)
class EffectSpec:
    """One lowered effect on a layer: the port key plus its port-owned payload."""

    handler: str
    path: str
    frame_origin: int   # frame on the layer's local grid (``plan.LayerSpec.local_frame``) that ``N`` counts from
    payload: Any


@dataclass(frozen=True)
class EffectPort:
    handler: str
    lower: Callable[[ResolvedEffect, LowerContext], Any]
    apply: Callable[[Any, torch.Tensor, ApplyContext], torch.Tensor]


EFFECT_PORTS: Final[dict[str, EffectPort]] = {}


def register(port: EffectPort) -> EffectPort:
    if port.handler in EFFECT_PORTS:
        raise AssertionError(f"effect port {port.handler!r} registered twice")
    EFFECT_PORTS[port.handler] = port
    return port


def effect_scalar(effect: ResolvedEffect, key: str, default: float) -> float:
    """Authored parameter value by FCPXML key, else the registry calibration default, else ``default``.

    Same lookup as the CPU emitter's ``_effect_scalar`` (``ffmpeg.py``): authored ``params``
    win, then ``calibration[key]["default"]``.
    """

    for parameter in effect.params:
        if parameter.key == key and parameter.value is not None:
            return float(parameter.value)
    calibration = effect.calibration.get(key) if isinstance(effect.calibration, Mapping) else None
    if isinstance(calibration, Mapping) and "default" in calibration:
        return float(calibration["default"])
    return default


def lower_effect(effect: ResolvedEffect, ctx: LowerContext, *, frame_origin: int) -> EffectSpec:
    """Lower one applied effect through its registered port or reject loudly (see module doc)."""

    if effect.mask is not None:
        # A masked effect is a small branch graph: lower the inside effect,
        # lower the optional outside effect, then let fx_mask perform matte
        # generation and premultiplied branch compositing.
        inside_effect = replace(effect, mask=None, outside_effect=None)
        inside = lower_effect(inside_effect, ctx, frame_origin=frame_origin)
        outside = (
            lower_effect(effect.outside_effect, ctx, frame_origin=frame_origin)
            if effect.outside_effect is not None and effect.outside_effect.execution == "apply"
            else None
        )
        return EffectSpec(
            handler="masked_effect",
            path=effect.path,
            frame_origin=frame_origin,
            payload=MaskEffectPayload(
                group=effect.mask,
                inside=inside,
                outside=outside,
                source_start=ctx.source_start,
                playback_rate=ctx.playback_rate,
                retime_map=ctx.retime_map,
                coordinate_scale_x=ctx.coordinate_scale_x,
                coordinate_scale_y=ctx.coordinate_scale_y,
            ),
        )
    port = EFFECT_PORTS.get(effect.handler or "")
    if port is None:
        raise reject(
            "effect (unported handler)",
            f"{ctx.clip_path}: {effect.name or '?'} handler={effect.handler!r}",
        )
    payload = port.lower(effect, ctx)
    return EffectSpec(handler=port.handler, path=effect.path, frame_origin=frame_origin, payload=payload)


def apply_effects(
    canvas: torch.Tensor,
    effects: tuple[EffectSpec, ...],
    *,
    frame: int,
    frame_duration: Fraction,
) -> torch.Tensor:
    """Run the layer's lowered effects in order on its conformed premultiplied-linear canvas."""

    _, height, width = canvas.shape
    for spec in effects:
        local = frame - spec.frame_origin
        ctx = ApplyContext(
            frame=local,
            seconds=float(local * frame_duration),
            width=width,
            height=height,
        )
        if spec.handler == "masked_effect":
            canvas = apply_masked_effect(
                spec.payload,
                canvas,
                frame=frame,
                seconds=ctx.seconds,
                apply_effect=lambda nested, pixels, nested_frame: _apply_one(
                    nested, pixels, nested_frame, frame_duration
                ),
            )
        else:
            canvas = _apply_one(spec, canvas, frame, frame_duration)
    return canvas


def _apply_one(spec: EffectSpec, canvas: torch.Tensor, frame: int, frame_duration: Fraction) -> torch.Tensor:
    """Apply one non-branching port, or recurse for a nested mask branch."""

    width, height = int(canvas.shape[2]), int(canvas.shape[1])
    local = frame - spec.frame_origin
    ctx = ApplyContext(
        frame=local,
        seconds=float(local * frame_duration),
        width=width,
        height=height,
    )
    if spec.handler == "masked_effect":
        return apply_masked_effect(
            spec.payload,
            canvas,
            frame=frame,
            seconds=ctx.seconds,
            apply_effect=lambda nested, pixels, nested_frame: _apply_one(
                nested, pixels, nested_frame, frame_duration
            ),
        )
    port = EFFECT_PORTS[spec.handler]
    return port.apply(spec.payload, canvas, ctx)


# ---------------------------------------------------------------------------
# Earthquake (``cohort_earthquake``): the first port, kept here as the worked example.
#
# The legacy ``geq`` samples the frame at
#   (X + W*amp*(sin(N*1.71+px) + 0.35*sin(N*4.13)),
#    Y + H*amp*(cos(N*1.37+py) + 0.35*sin(N*3.29)))
# clamped to the frame -- a per-frame constant sub-pixel translation with bilinear
# sampling and edge clamping.  Here: one ``grid_sample`` with ``padding_mode="border"``
# on the premultiplied-linear canvas (the calibrated Yunah port; see
# ``cohort_effects._earthquake_filter`` for the reference string).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarthquakePayload:
    amplitude: float
    phase_x: float
    phase_y: float


def _lower_earthquake(effect: ResolvedEffect, ctx: LowerContext) -> EarthquakePayload:
    amount = effect_scalar(effect, "9999/10062/100/10063/2/100", 0.0979)
    layers = effect_scalar(effect, "9999/10039/100/10044/4", 3.0)
    # Epicenter vector default (0.5, 0.5) -> phase pi/2 on both axes.
    return EarthquakePayload(
        amplitude=amount * (0.0034 + 0.0003 * layers),
        phase_x=0.5 * math.pi,
        phase_y=0.5 * math.pi,
    )


def _apply_earthquake(payload: EarthquakePayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    _, height, width = canvas.shape
    n = ctx.frame
    dx = width * payload.amplitude * (math.sin(n * 1.71 + payload.phase_x) + 0.35 * math.sin(n * 4.13))
    dy = height * payload.amplitude * (math.cos(n * 1.37 + payload.phase_y) + 0.35 * math.sin(n * 3.29))
    # Normalized sampling grid: destination pixel (x, y) reads source (x+dx, y+dy).
    ys = torch.arange(height, device=canvas.device, dtype=canvas.dtype)
    xs = torch.arange(width, device=canvas.device, dtype=canvas.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    src_x = ((grid_x + dx).clamp(0, width - 1) / (width - 1)) * 2.0 - 1.0
    src_y = ((grid_y + dy).clamp(0, height - 1) / (height - 1)) * 2.0 - 1.0
    grid = torch.stack((src_x, src_y), dim=-1).unsqueeze(0)
    return F.grid_sample(
        canvas.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
    ).squeeze(0)


register(EffectPort(handler="cohort_earthquake", lower=_lower_earthquake, apply=_apply_earthquake))


register(
    EffectPort(
        handler="green_screen_keyer",
        lower=lower_green_screen_keyer,
        apply=lambda payload, canvas, _ctx: green_screen_key(canvas, payload),
    )
)


# Port modules register on import (one owner per file; append-only).
from . import fx_basic as _fx_basic  # noqa: E402,F401  (E1)
from . import fx_branched as _fx_branched  # noqa: E402,F401  (E3: Callout)
from . import fx_cohort as _fx_cohort  # noqa: E402,F401  (Phase 4 approved effect cohort)
from . import fx_color as _fx_color  # noqa: E402,F401  (E4)
from . import fx_warp as _fx_warp  # noqa: E402,F401  (E2)
