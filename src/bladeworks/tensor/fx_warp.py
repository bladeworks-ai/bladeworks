"""E2 effect ports: the warp / matte cohort (owner: E2 batch).

Architecture map
================

    ResolvedEffect (handler in cohort_effects.COHORT_EFFECT_HANDLERS)
        -> cohort_effects.cohort_effect_filters(effect)      : the CPU reference's exact filter
                                                               strings (read-only, pure function
                                                               of the effect's validated params)
        -> lower_*  (plan time)                              : parse those strings into a frozen
                                                               payload; anything not one of the two
                                                               shapes below is a loud reject
        -> apply_*  (per frame, premultiplied linear RGBA)   : color.premultiplied_to_code
                                                               -> 8-bit code-space kernel
                                                               -> color.code_to_premultiplied

    Shape A -- one ``geq`` filter (8 handlers)
        cohort_directional_blur, cohort_radial_blur (17 taps in ONE geq, summed left to right),
        cohort_fisheye, cohort_droplet, cohort_crop_feather, cohort_vignette_mask,
        cohort_kaleidoscope, cohort_perspective_tile
        -> ``expr.parse`` each of the four channel expressions (``r/g/b/a``) taken verbatim from
           the CPU string, ``expr.geq_rgba`` (vf_geq.c loop: bilinear clamped ``r/g/b/alpha``
           samplers, ``N``/``T``, uint8 store).  Nothing is hand-ported: the port evaluates the
           same text the reference feeds ffmpeg.

    Shape B -- ``[eq=brightness:contrast,] vibrance=intensity:rbal:gbal:bbal`` (cohort_vibrancy)
        -> the reference chain is ``format=rgba -> eq -> vibrance``; ``eq`` only accepts YUV, so
           libavfilter auto-inserts swscale ``rgba -> yuva444p`` before it and ``yuva444p -> rgba``
           after it (both BT.601 limited-range, verified with ``-loglevel verbose``).  The port
           reproduces that bridge with swscale's own integer arithmetic (``_rgba_to_yuva444p`` /
           ``_yuva444p_to_rgba``, bit-exact against the ffmpeg CLI on random plates), ``eq``'s
           ``process_c`` integer luma LUT, then ``vf_vibrance.c``'s float32 kernel.

Pixel domain (E6): the reference emits these filters after crop/conform and before the spatial
tail on 8-bit ``format=rgba`` straight alpha (``ffmpeg._video_chain`` -> ``_ordered_effect_filters``);
``renderer.placed`` hands the port the conformed premultiplied-linear canvas, so every port here
round-trips through ``color.premultiplied_to_code`` (straight 0..255 encoded) and rounds the input
to whole codes first -- the reference's link *is* an 8-bit frame, and the kernels below store as
uint8 (truncating), so feeding them fractional codes would bias pass-through channels by -0.5.

Reachable parameter surface: the capability registry admits only Radial Blur ``Amount``,
Crop & Feather ``Width/Height/Feather``, Droplet ``Intensity`` and Vibrancy ``Amount``
(``effect_parameters.unsupported_parameter_reason`` fails the effect closed for anything else,
animated controls included), so the other handlers only ever reach a port at their defaults.
``cohort_effect_filters`` is still consulted per effect so an authored value can never drift from
the reference string.

Main callers:
- ``tensor/effects.py`` imports this module for its ``register`` side effects
  (``lower_effect`` / ``apply_effects``).
- ``experimental_tests/core/test_tensor_fx_warp.py`` (per-handler synthetic goldens vs the
  ffmpeg CLI + end-to-end SSIM vs the CPU ``reference`` render).

Not here: ``cohort_earthquake`` is ported in ``tensor/effects.py``; the branched handlers
``cohort_focus_blur`` / ``cohort_drop_shadow`` / ``cohort_camcorder`` / ``cohort_cartoon``
are ported in ``fx_cohort.py``; ``cohort_callout`` is in ``fx_branched.py``; and the Color
Curves / Hue-Sat Curves no-ops are in ``fx_basic.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import torch

from ..core.cohort_effects import cohort_effect_filters
from ..core.model import ResolvedEffect
from .color import code_to_premultiplied, premultiplied_to_code
from .effects import ApplyContext, EffectPort, LowerContext, register
from .expr import ExpressionError, geq_rgba, parse
from .support import reject


def _rounded_code(canvas: torch.Tensor) -> torch.Tensor:
    """Premultiplied linear RGBA -> straight 0..255 codes rounded to whole 8-bit values (see module doc)."""

    return premultiplied_to_code(canvas).round().clamp(0.0, 255.0)


# =============================================================================
# Shape A: one ``geq`` filter, evaluated verbatim through tensor/expr.py
# =============================================================================

GEQ_HANDLERS: Final[tuple[str, ...]] = (
    "cohort_directional_blur",
    "cohort_radial_blur",
    "cohort_fisheye",
    "cohort_droplet",
    "cohort_crop_feather",
    "cohort_vignette_mask",
    "cohort_kaleidoscope",
    "cohort_perspective_tile",
)

_GEQ_CHANNEL = re.compile(r"([rgba])='([^']*)'")


@dataclass(frozen=True)
class GeqPayload:
    """The CPU ``geq`` string split into its four channel expressions (texts; ``expr.parse`` caches)."""

    filter_text: str
    r: str
    g: str
    b: str
    a: str

    def expressions(self) -> dict[str, object]:
        return {key: parse(getattr(self, key)) for key in ("r", "g", "b", "a")}


def geq_channels(filter_text: str) -> dict[str, str]:
    """Split ``geq=r='..':g='..':b='..':a='..'`` (the shape ``cohort_effects._sample_filter`` /
    ``_identity_rgb_with_alpha`` emit) into ``{"r": .., "g": .., "b": .., "a": ..}``.

    Raises ``ValueError`` for any other shape -- the caller turns that into a loud reject.
    """

    if not filter_text.startswith("geq="):
        raise ValueError(f"not a geq filter: {filter_text[:60]!r}")
    found = _GEQ_CHANNEL.findall(filter_text[len("geq=") :])
    keys = [key for key, _ in found]
    if keys != ["r", "g", "b", "a"]:
        raise ValueError(f"geq filter does not name exactly r, g, b, a in order: {keys}")
    rebuilt = "geq=" + ":".join(f"{key}='{text}'" for key, text in found)
    if rebuilt != filter_text:
        raise ValueError("geq filter has options outside the four channel expressions")
    return dict(found)


def _lower_geq(effect: ResolvedEffect, ctx: LowerContext) -> GeqPayload:
    """Take the reference filter string for this effect and keep its four expressions.

    ``cohort_effect_filters`` is the CPU emitter's own builder (``ffmpeg._effect_filters`` calls
    it for every cohort handler), so the payload text is exactly what the reference runs.
    """

    filters = cohort_effect_filters(effect)
    label = f"{ctx.clip_path}: {effect.name or effect.handler}"
    if len(filters) != 1:
        raise reject(
            "effect (unsupported parameters)",
            f"{label}: expected one geq filter for {effect.handler}, the reference emits {len(filters)}",
        )
    try:
        channels = geq_channels(filters[0])
        for text in channels.values():
            parse(text)
    except (ValueError, ExpressionError) as exc:
        raise reject("effect (unsupported parameters)", f"{label}: {exc}") from exc
    return GeqPayload(filter_text=filters[0], **channels)


def _apply_geq(payload: GeqPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    code = _rounded_code(canvas)
    out = geq_rgba(payload.expressions(), code, frame_number=ctx.frame, time_seconds=ctx.seconds)
    return code_to_premultiplied(out)


for _handler in GEQ_HANDLERS:
    register(EffectPort(handler=_handler, lower=_lower_geq, apply=_apply_geq))


# =============================================================================
# Shape B: Vibrancy = [eq] + vibrance through the swscale rgba <-> yuva444p bridge
# =============================================================================
#
# swscale (libswscale, FFmpeg n8.0, default flags = bicubic, no SWS_ACCURATE_RND) for an
# unscaled rgba -> yuva444p conversion runs the generic scaler path:
#   input.c  rgb32ToY_c / rgb32ToUV_c : 15-bit ``(RY*r + GY*g + BY*b + 0x80100) >> 9`` (BT.601
#            limited coefficients from utils.c fill_rgb2yuv_table's SWS_CS_DEFAULT branch),
#            chroma with ``0x400100``; alpha ``a<<6 | a>>2``
#   swscale.c hScale16To15_c (sh = 13 for RGB sources, one 1<<14 tap): value * 2
#   output.c yuv2plane1_8_c with the flat sws_pb_64 dither: ``(v + 64) >> 7`` clipped to uint8
# and yuva444p -> rgba:
#   hScale8To15_c: ``src * 128``; yuv2rgb_full_1_c_template: ``Y = y15*4``, ``U = (u15-16384)*4``,
#   ``A = (a15+64)>>7``; yuv2rgb_write_full with ff_yuv2rgb_c_init_tables' 601 limited constants
#   (y_coeff 9539, y_offset 8192, v2r 13075, v2g -6660, u2g -3209, u2b 16525): clip to 30 bits,
#   ``>> 22``.
# Both directions verified bit-exact against ``ffmpeg -vf format=rgba,eq=...,format=rgba`` on
# 131k random pixels (test_tensor_fx_warp.py::test_vibrancy_bridge_golden).

_RY, _GY, _BY = 8414, 16519, 3208
_RU, _GU, _BU = -4865, -9528, 14392
_RV, _GV, _BV = 14392, -12061, -2332
_Y_COEFF, _Y_OFFSET = 9539, 8192
_V2R, _V2G, _U2G, _U2B = 13075, -6660, -3209, 16525


def _plane_to_uint8(v15: torch.Tensor) -> torch.Tensor:
    """``yuv2plane1_8_c`` with the flat 64 dither: ``av_clip_uint8((v + 64) >> 7)``."""

    return ((v15 + 64) >> 7).clamp(0, 255)


def _rgba_to_yuva444p(code: torch.Tensor) -> torch.Tensor:
    """swscale ``rgba -> yuva444p`` on integer 0..255 codes ``[4, H, W]`` (int64) -> ``[4, H, W]`` int64."""

    r, g, b, a = code[0], code[1], code[2], code[3]
    y = ((_RY * r + _GY * g + _BY * b + 0x80100) >> 9) * 2
    u = ((_RU * r + _GU * g + _BU * b + 0x400100) >> 9) * 2
    v = ((_RV * r + _GV * g + _BV * b + 0x400100) >> 9) * 2
    alpha = ((a << 6) | (a >> 2)) * 2
    return torch.stack((_plane_to_uint8(y), _plane_to_uint8(u), _plane_to_uint8(v), _plane_to_uint8(alpha)))


def _yuva444p_to_rgba(yuva: torch.Tensor) -> torch.Tensor:
    """swscale ``yuva444p -> rgba`` on integer planes ``[4, H, W]`` (int64) -> integer codes ``[4, H, W]``."""

    y8, u8, v8, a8 = yuva[0], yuva[1], yuva[2], yuva[3]
    luma = (y8 * 512 - _Y_OFFSET) * _Y_COEFF + (1 << 21)
    u = (u8 - 128) * 512
    v = (v8 - 128) * 512
    red = luma + v * _V2R
    green = luma + v * _V2G + u * _U2G
    blue = luma + u * _U2B
    limit = (1 << 30) - 1
    return torch.stack(
        (
            red.clamp(0, limit) >> 22,
            green.clamp(0, limit) >> 22,
            blue.clamp(0, limit) >> 22,
            a8,
        )
    )


def eq_luma_lut(brightness: float, contrast: float) -> tuple[int, ...]:
    """``vf_eq.h process_c`` (the gamma == 1, |contrast| < 7.9 path) as a 256-entry luma LUT.

    ``vf_eq.c`` clips the evaluated option values with ``av_clipf`` (returns *float*), so both
    controls are float32-rounded before the integer quantisation -- ``brightness=0.16`` is
    0.15999999 and ``(int)(100*b + 100)`` is 115, not 116 (verified against ffmpeg on a luma ramp).
    """

    brightness = float(torch.tensor(brightness, dtype=torch.float32))
    contrast = float(torch.tensor(contrast, dtype=torch.float32))
    contrast_q = int(contrast * 256 * 16)
    brightness_q = (int(100.0 * brightness + 100.0) * 511) // 200 - 128 - int(contrast_q / 32)
    table = []
    for code in range(256):
        pel = ((code * contrast_q) >> 12) + brightness_q
        table.append(0 if pel < 0 else 255 if pel > 255 else pel)
    return tuple(table)


def _vibrance_uint8(code: torch.Tensor, *, intensity: float, rbal: float, gbal: float, bbal: float) -> torch.Tensor:
    """``vf_vibrance.c vibrance_slice8p`` (the packed rgba path) in float32, alpha copied.

    Same operation order as the C loop; the C build contracts some products into FMAs, which
    leaves ~4e-6 of the stored values one code away (documented in the golden).
    """

    f32 = torch.float32
    scale = float(torch.tensor(1.0, dtype=f32) / torch.tensor(255.0, dtype=f32))  # ``1.f / 255.f``
    x = code.to(f32) * scale
    r, g, b = x[0], x[1], x[2]
    intensity32 = float(torch.tensor(intensity, dtype=f32))
    gint = float(torch.tensor(intensity32 * float(torch.tensor(gbal, dtype=f32)), dtype=f32))
    bint = float(torch.tensor(intensity32 * float(torch.tensor(bbal, dtype=f32)), dtype=f32))
    rint = float(torch.tensor(intensity32 * float(torch.tensor(rbal, dtype=f32)), dtype=f32))
    alternate = -1.0  # ``alternate`` option is 0 in every reference string
    sg = alternate * _ffsign(gint)
    sb = alternate * _ffsign(bint)
    sr = alternate * _ffsign(rint)
    saturation = torch.maximum(torch.maximum(r, g), b) - torch.minimum(torch.minimum(r, g), b)
    luma = g * 0.715158 + r * 0.212656 + b * 0.072186
    cg = 1.0 + gint * (1.0 - sg * saturation)
    cb = 1.0 + bint * (1.0 - sb * saturation)
    cr = 1.0 + rint * (1.0 - sr * saturation)
    g2 = luma + (g - luma) * cg
    b2 = luma + (b - luma) * cb
    r2 = luma + (r - luma) * cr
    rgb = torch.stack((r2, g2, b2)) * 255.0
    # ``av_clip_uint8((int)(v * 255.f))``: C float -> int conversion truncates toward zero.
    rgb = rgb.trunc().clamp(0.0, 255.0)
    return torch.cat((rgb, code[3:4].to(f32)), dim=0)


def _ffsign(value: float) -> float:
    return 1.0 if value > 0 else -1.0 if value < 0 else 0.0


_EQ_FILTER = re.compile(r"^eq=brightness=([-0-9.]+):contrast=([-0-9.]+)$")
_VIBRANCE_FILTER = re.compile(r"^vibrance=intensity=([-0-9.]+):rbal=([-0-9.]+):gbal=([-0-9.]+):bbal=([-0-9.]+)$")


@dataclass(frozen=True)
class VibrancyPayload:
    filters: tuple[str, ...]
    eq_lut: tuple[int, ...] | None    # None when the reference emits no eq stage
    intensity: float
    rbal: float
    gbal: float
    bbal: float


def _lower_vibrancy(effect: ResolvedEffect, ctx: LowerContext) -> VibrancyPayload:
    """Parse the reference's ``[eq,] vibrance`` strings (the only two shapes ``cohort_vibrancy`` emits)."""

    filters = tuple(cohort_effect_filters(effect))
    label = f"{ctx.clip_path}: {effect.name or effect.handler}"
    if not filters or len(filters) > 2:
        raise reject("effect (unsupported parameters)", f"{label}: unexpected vibrancy chain {filters}")
    eq_lut = None
    if len(filters) == 2:
        match = _EQ_FILTER.match(filters[0])
        if match is None:
            raise reject("effect (unsupported parameters)", f"{label}: unexpected eq stage {filters[0]!r}")
        eq_lut = eq_luma_lut(float(match.group(1)), float(match.group(2)))
    match = _VIBRANCE_FILTER.match(filters[-1])
    if match is None:
        raise reject("effect (unsupported parameters)", f"{label}: unexpected vibrance stage {filters[-1]!r}")
    intensity, rbal, gbal, bbal = (float(match.group(index)) for index in range(1, 5))
    return VibrancyPayload(filters=filters, eq_lut=eq_lut, intensity=intensity, rbal=rbal, gbal=gbal, bbal=bbal)


def apply_vibrancy_code(payload: VibrancyPayload, code: torch.Tensor) -> torch.Tensor:
    """The whole reference chain on integer-valued 0..255 codes ``[4, H, W]`` (any float/int dtype)."""

    ints = code.to(torch.int64)
    if payload.eq_lut is not None:
        yuva = _rgba_to_yuva444p(ints)
        lut = torch.tensor(payload.eq_lut, dtype=torch.int64, device=code.device)
        yuva = torch.stack((lut[yuva[0]], yuva[1], yuva[2], yuva[3]))
        ints = _yuva444p_to_rgba(yuva)
    return _vibrance_uint8(ints, intensity=payload.intensity, rbal=payload.rbal, gbal=payload.gbal, bbal=payload.bbal)


def _apply_vibrancy(payload: VibrancyPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    out = apply_vibrancy_code(payload, _rounded_code(canvas))
    return code_to_premultiplied(out.to(canvas.dtype))


register(EffectPort(handler="cohort_vibrancy", lower=_lower_vibrancy, apply=_apply_vibrancy))
