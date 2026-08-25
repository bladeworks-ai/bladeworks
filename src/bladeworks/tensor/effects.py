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

Effect surface vs clip canvas (overscan)
----------------------------------------
The staged leaf path may hand ``apply_effects`` a surface LARGER than the clip canvas: the
conform's overscan (a portrait ``fill``, a Crop / Ken-Burns camera) is kept so a later pan or
zoom samples real image (``renderer._overscan_surface``).  Every reference filter, however,
was calibrated on the CLIP CANVAS raster: vignette / mask centres, noise row shifts, the
pixelize grid, blur boundary handling, HUD placement ... all read the canvas geometry.  So
``apply_effects`` takes a ``CanvasPlacement`` (where the canvas sits on the surface) and:

* ``ApplyContext.width`` / ``height`` are ALWAYS the clip canvas; ``origin_x`` / ``origin_y``
  is the clip-canvas coordinate of surface pixel (0, 0) (``<= 0``).  A port reads the
  surface size from the tensor and its coordinate system from the context
  (``ApplyContext.pixel_axes``).
* each ``EffectPort`` declares an ``overscan`` policy (see ``EffectPort``): ``"extend"`` runs
  the port once on the whole surface (the port proves its canvas region is byte-identical to a
  canvas-only run), ``"splice"`` runs the port on the canvas crop (byte-exact, the pre-overscan
  path) AND on the whole surface, keeping the surface run only outside the canvas, and
  ``"crop"`` runs the canvas crop only (the overscan passes through unchanged).

Invariant (guarded by ``test_tensor_overscan_surface.py``): the canvas region of the output is
byte-identical to running the same chain on the canvas crop, for every policy.  When the
surface IS the canvas (the default placement) every policy collapses to one plain run.

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
from .expr import Environment, Expr, bilinear_sampler, evaluate_image, geq_rgba, pixel_grid, quantize_uint8
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
class CanvasPlacement:
    """Where the clip canvas sits on the surface ``apply_effects`` is given (see module doc).

    ``origin_x`` / ``origin_y`` is the clip-canvas coordinate of the surface's top-left pixel,
    so surface column ``i`` is clip-canvas column ``i + origin_x``.  The surface must contain
    the whole canvas (``origin <= 0`` and ``-origin + canvas <= surface``); ``apply_effects``
    checks that loudly.  ``renderer._overscan_surface`` produces exactly such placements.
    """

    width: int      # clip canvas width
    height: int     # clip canvas height
    origin_x: int = 0
    origin_y: int = 0


@dataclass(frozen=True)
class ApplyContext:
    """Per-frame facts for ``EffectPort.apply``.

    ``width`` / ``height`` are the CLIP CANVAS the reference filter was calibrated on, never
    the surface size (read that from the tensor).  ``origin_x`` / ``origin_y`` place the
    surface in clip-canvas coordinates (``CanvasPlacement``); both are 0 in the common case
    where the surface is the canvas.
    """

    frame: int          # layer-local frame counter (``N``): 0 at the layer's frame origin
    seconds: float      # ``T`` = frame * frame_duration
    width: int
    height: int
    origin_x: int = 0
    origin_y: int = 0

    def is_whole_surface(self, canvas: torch.Tensor) -> bool:
        """True when ``canvas`` is exactly the clip canvas (no overscan on this surface)."""

        return (
            self.origin_x == 0
            and self.origin_y == 0
            and int(canvas.shape[1]) == self.height
            and int(canvas.shape[2]) == self.width
        )

    def canvas_slices(self) -> tuple[slice, slice]:
        """``(rows, cols)`` of the clip canvas region on the surface."""

        return (
            slice(-self.origin_y, -self.origin_y + self.height),
            slice(-self.origin_x, -self.origin_x + self.width),
        )

    def on_canvas(self) -> "ApplyContext":
        """The context for a run on the canvas crop itself (origin 0)."""

        return replace(self, origin_x=0, origin_y=0)

    def pixel_axes(self, canvas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Clip-canvas pixel coordinates of the surface's columns and rows.

        Returns ``(xs, ys)`` shaped ``[1, surface_w]`` and ``[surface_h, 1]`` in the tensor's
        dtype/device: ``xs[0, i] == i + origin_x``.  On the canvas itself this is the plain
        ``arange`` every port used before overscan existed, so an integer-origin offset keeps
        the canvas region's values bit-identical.
        """

        surface_h, surface_w = int(canvas.shape[1]), int(canvas.shape[2])
        xs = torch.arange(surface_w, device=canvas.device, dtype=canvas.dtype) + self.origin_x
        ys = torch.arange(surface_h, device=canvas.device, dtype=canvas.dtype) + self.origin_y
        return xs.view(1, -1), ys.view(-1, 1)


@dataclass(frozen=True)
class EffectSpec:
    """One lowered effect on a layer: the port key plus its port-owned payload."""

    handler: str
    path: str
    frame_origin: int   # frame on the layer's local grid (``plan.LayerSpec.local_frame``) that ``N`` counts from
    payload: Any


# ``EffectPort.overscan`` policies (module doc, "Effect surface vs clip canvas").
OVERSCAN_EXTEND: Final[str] = "extend"   # one run on the whole surface; the port honours ctx canvas coords
OVERSCAN_SPLICE: Final[str] = "splice"   # canvas crop run (exact) + surface run for the overscan pixels
OVERSCAN_CROP: Final[str] = "crop"       # canvas crop run only; overscan pixels pass through untouched
_OVERSCAN_POLICIES: Final[frozenset[str]] = frozenset((OVERSCAN_EXTEND, OVERSCAN_SPLICE, OVERSCAN_CROP))


@dataclass(frozen=True)
class EffectPort:
    """One ported handler.

    ``overscan`` says how the port behaves on a surface larger than the clip canvas:

    * ``"extend"`` -- the port's output at canvas pixels is bit-identical whether it runs on
      the crop or on the surface (pointwise kernels, or kernels re-expressed in clip-canvas
      coordinates through ``ApplyContext``).  One run on the whole surface.
    * ``"splice"`` -- the canvas output depends on the canvas boundary (blur recursions,
      replicate padding, clamped taps, edge blocks ...) so it cannot be reproduced on the
      surface.  The crop run supplies the canvas region, a second, canvas-coordinate-aware
      surface run supplies the overscan.  Costs two runs when overscan is present.
    * ``"crop"`` -- the port cannot sensibly continue into the overscan at all (it paints a
      canvas-relative composition).  Crop run only; the overscan keeps the input pixels.

    The safe default is ``"crop"``: never wrong on the canvas, never paints artefacts outside
    it.  Every registered port sets its policy explicitly.
    """

    handler: str
    lower: Callable[[ResolvedEffect, LowerContext], Any]
    apply: Callable[[Any, torch.Tensor, ApplyContext], torch.Tensor]
    overscan: str = OVERSCAN_CROP


EFFECT_PORTS: Final[dict[str, EffectPort]] = {}


def register(port: EffectPort) -> EffectPort:
    if port.handler in EFFECT_PORTS:
        raise AssertionError(f"effect port {port.handler!r} registered twice")
    if port.overscan not in _OVERSCAN_POLICIES:
        raise AssertionError(f"effect port {port.handler!r}: unknown overscan policy {port.overscan!r}")
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
    placement: Optional[CanvasPlacement] = None,
) -> torch.Tensor:
    """Run the layer's lowered effects in order on its conformed premultiplied-linear canvas.

    ``placement`` says where the clip canvas sits when ``canvas`` is an overscan-preserving
    surface (module doc); ``None`` means the tensor IS the clip canvas.
    """

    surface_h, surface_w = int(canvas.shape[1]), int(canvas.shape[2])
    if placement is None:
        placement = CanvasPlacement(width=surface_w, height=surface_h)
    elif (
        placement.origin_x > 0
        or placement.origin_y > 0
        or -placement.origin_x + placement.width > surface_w
        or -placement.origin_y + placement.height > surface_h
    ):
        raise ValueError(
            f"effect surface {surface_w}x{surface_h} does not contain the clip canvas "
            f"{placement.width}x{placement.height} at origin ({placement.origin_x}, {placement.origin_y})"
        )
    for spec in effects:
        canvas = _apply_one(spec, canvas, frame, frame_duration, placement)
    return canvas


def _apply_one(
    spec: EffectSpec,
    canvas: torch.Tensor,
    frame: int,
    frame_duration: Fraction,
    placement: CanvasPlacement,
) -> torch.Tensor:
    """Apply one port under its overscan policy, or recurse for a nested mask branch."""

    local = frame - spec.frame_origin
    ctx = ApplyContext(
        frame=local,
        seconds=float(local * frame_duration),
        width=placement.width,
        height=placement.height,
        origin_x=placement.origin_x,
        origin_y=placement.origin_y,
    )
    if spec.handler == "masked_effect":
        # The matte is geometric on the clip canvas (fx_mask takes the placement); the
        # branches run on the same surface through this dispatcher, policy included.
        return apply_masked_effect(
            spec.payload,
            canvas,
            frame=frame,
            seconds=ctx.seconds,
            apply_effect=lambda nested, pixels, nested_frame: _apply_one(
                nested, pixels, nested_frame, frame_duration, placement
            ),
            canvas_rect=(placement.width, placement.height, placement.origin_x, placement.origin_y),
        )
    return _apply_port(EFFECT_PORTS[spec.handler], spec.payload, canvas, ctx)


def _apply_port(port: EffectPort, payload: Any, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    """Run one port on ``canvas`` under ``port.overscan`` (see ``EffectPort``).

    Pythonese:
    1. If the surface is the clip canvas, or the port extends into overscan exactly, run it
       once on the whole tensor.
    2. Otherwise run it on the canvas crop with an origin-0 context: that is byte-for-byte the
       pre-overscan computation, so the canvas region is exact by construction.
    3. ``"splice"``: also run it on the whole surface (canvas-coordinate context) and take the
       overscan pixels from that run.  ``"crop"``: the overscan keeps the input pixels.
    4. Paste the exact canvas result into a fresh tensor (never in place: a mask branch still
       reads the unmodified input for its outside branch).
    """

    if port.overscan == OVERSCAN_EXTEND or ctx.is_whole_surface(canvas):
        return port.apply(payload, canvas, ctx)
    rows, cols = ctx.canvas_slices()
    exact = port.apply(payload, canvas[:, rows, cols], ctx.on_canvas())
    if port.overscan == OVERSCAN_SPLICE:
        out = port.apply(payload, canvas, ctx).clone()
    else:
        out = canvas.clone()
    out[:, rows, cols] = exact
    return out


def geq_rgba_canvas(
    expressions: Mapping[str, Expr],
    source_rgba: torch.Tensor,
    ctx: ApplyContext,
) -> torch.Tensor:
    """``expr.geq_rgba`` in CLIP-CANVAS coordinates on a possibly larger surface.

    On the canvas itself this IS ``geq_rgba`` (same call, byte-identical).  On an overscan
    surface the expressions see ``X``/``Y`` as clip-canvas coordinates and ``W``/``H`` as the
    clip canvas, and the ``r/g/b/alpha/p`` samplers take clip-canvas positions (translated onto
    the surface, then clamped to the surface like ``vf_geq.c`` clamps to its frame).  So a
    centre-relative warp (fisheye, kaleidoscope), a frame-relative matte (crop & feather,
    vignette mask) or a mirror keeps its canvas geometry and simply continues outward, reading
    the real overscan image where its taps fall outside the canvas.

    Canvas pixels whose taps fall outside the canvas read real pixels here instead of the
    clamped canvas edge, so this is NOT canvas-exact for such ports -- they register as
    ``"splice"`` (the crop run supplies the canvas).  Ports whose taps never leave the canvas
    (threshold, mirror) are exact and register as ``"extend"``.

    Main callers: ``fx_basic`` (threshold / mirror) and ``fx_warp`` (the E2 geq cohort).
    """

    if ctx.is_whole_surface(source_rgba):
        return geq_rgba(expressions, source_rgba, frame_number=ctx.frame, time_seconds=ctx.seconds)
    missing = [key for key in ("r", "g", "b", "a") if key not in expressions]
    if missing:
        raise ValueError(f"geq needs expressions for r, g, b and a; missing {missing}")
    _, surface_h, surface_w = source_rgba.shape
    device, dtype = source_rgba.device, source_rgba.dtype
    grid_x, grid_y = pixel_grid(surface_h, surface_w, device=device, dtype=dtype)
    grid_x = grid_x + ctx.origin_x
    grid_y = grid_y + ctx.origin_y

    def canvas_sampler(plane: torch.Tensor) -> Callable[[Any, Any], torch.Tensor]:
        surface_sample = bilinear_sampler(plane)

        def sample(x: Any, y: Any) -> torch.Tensor:
            return surface_sample(x - ctx.origin_x, y - ctx.origin_y)

        return sample

    channel_samplers = {
        "r": canvas_sampler(source_rgba[0]),
        "g": canvas_sampler(source_rgba[1]),
        "b": canvas_sampler(source_rgba[2]),
        "alpha": canvas_sampler(source_rgba[3]),
    }
    out = torch.empty_like(source_rgba)
    for channel_index, key in enumerate(("r", "g", "b", "a")):
        samplers = dict(channel_samplers)
        samplers["p"] = channel_samplers["alpha" if key == "a" else key]
        env = Environment(
            variables={
                "X": grid_x, "Y": grid_y, "W": float(ctx.width), "H": float(ctx.height),
                "N": float(ctx.frame), "T": float(ctx.seconds), "SW": 1.0, "SH": 1.0,
            },
            samplers=samplers,
            device=device,
            dtype=dtype,
        )
        out[channel_index] = quantize_uint8(evaluate_image(expressions[key], env, (surface_h, surface_w)))
    return out


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
    # The displacement is a fraction of the CLIP CANVAS (``W*amp`` in the geq), the sampling
    # grid spans whatever surface is given and clamps to ITS edges -- so on an overscan
    # surface the shake reads the real image beyond the canvas.  Canvas pixels near the canvas
    # edge therefore differ from the canvas-only run (real pixels instead of the clamped edge):
    # the port is ``"splice"`` and the crop run supplies the canvas region.
    _, height, width = canvas.shape
    n = ctx.frame
    dx = ctx.width * payload.amplitude * (math.sin(n * 1.71 + payload.phase_x) + 0.35 * math.sin(n * 4.13))
    dy = ctx.height * payload.amplitude * (math.cos(n * 1.37 + payload.phase_y) + 0.35 * math.sin(n * 3.29))
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


register(
    EffectPort(
        handler="cohort_earthquake",
        lower=_lower_earthquake,
        apply=_apply_earthquake,
        overscan=OVERSCAN_SPLICE,
    )
)


register(
    EffectPort(
        handler="green_screen_keyer",
        lower=lower_green_screen_keyer,
        apply=lambda payload, canvas, _ctx: green_screen_key(canvas, payload),
        overscan=OVERSCAN_EXTEND,  # pointwise key / despill
    )
)


# Port modules register on import (one owner per file; append-only).
from . import fx_basic as _fx_basic  # noqa: E402,F401  (E1)
from . import fx_branched as _fx_branched  # noqa: E402,F401  (E3: Callout)
from . import fx_cohort as _fx_cohort  # noqa: E402,F401  (Phase 4 approved effect cohort)
from . import fx_color as _fx_color  # noqa: E402,F401  (E4)
from . import fx_warp as _fx_warp  # noqa: E402,F401  (E2)
