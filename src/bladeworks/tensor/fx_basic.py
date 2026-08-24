"""E1 effect ports: simple + LUT + blur / sharpen / vignette + Color Curves no-ops + Color Wheels.

Architecture map
================

    effects.lower_effect(effect, ctx)                (plan time)
        -> _lower_<handler>(effect, ctx) -> frozen payload   (parameters read exactly like the
                                                              CPU emitter for the same handler)
    effects.apply_effects(canvas, specs, frame)      (per frame, premultiplied LINEAR RGBA in/out)
        -> _apply_<handler>(payload, canvas, ctx)
             1. ``_code8(canvas)``   premultiplied linear -> straight 0..255 code, rounded to
                                     integers: the ``format=rgba`` 8-bit link the reference filter
                                     reads (``ffmpeg._video_chain`` puts ``format=rgba`` in
                                     ``initial_filters`` before ``_ordered_effect_filters``)
             2. the filter's own arithmetic, mirrored from the FFmpeg n8.0 libavfilter source,
                in the pixel format libavfilter negotiates for it (see the table below); the
                two auto-inserted swscale conversions the reference pays for filters that do
                not take ``rgba`` are mirrored too (``_rgba_to_yuva444p`` /
                ``_yuva444p_to_rgba``: limited-range fixed point, BT.709 for 709-tagged
                sources / ITU601 for untagged ones -- the link colourspace libavfilter
                negotiates -- bit-exact against ffmpeg; ``rgba -> gbrap`` and ``rgba ->
                rgb24`` are exact byte shuffles)
             3. ``_from_code8(...)``  the C store (truncation / lrint per filter) back to
                                     premultiplied linear

    handler                       reference filter string (``ffmpeg._effect_filters``)              negotiated pix_fmt
    ---------------------------   ---------------------------------------------------------------   ------------------
    negative                      lutrgb=r/g/b='pow(clip(1-pow(clip(val/maxval,0,1),2.2),0,1),0.45)*maxval'   rgba
    threshold                     geq (two-band luma smoothstep, calibrated)                        gbrap
    colorize                      colorchannelmixer=rr=0.69:rg=0.27:rb=0.20:gr=0.087:gg=0.47:gb=0.08:br=0:bg=0:bb=0.51   rgba
    tint                          colorize=hue=245.54457:saturation=0.324662:lightness=0.270485:mix=1,eq=brightness=-0.15   yuva444p (YUV bridge)
    flipped                       hflip                                                             rgba
    mirror                        geq (right half reflected across the centre seam)                gbrap
    add_noise                     noise=c0s=6:c1s=6:c2s=6:c3s=0:c0_seed=424242:c1_seed=424243:c2_seed=424244   gbrap
    pixellate_default             pixelize=width=4:height=4:mode=avg:planes=0x7                    gbrap
    gaussian                      gblur=sigma=<amount*boost*ffmpeg_sigma>                          gbrap (all 4 planes)
    sharpen                       unsharp=5:5:<amount*ffmpeg_scale>:5:5:0                          yuva444p (YUV bridge, Y only)
    vignette                      vignette=angle=<clamp(strength*scale/size)>:eval=frame           rgb24 (ALPHA DROPPED -> opaque)
    cohort_color_curves           curves=master='0/0 1/1'                                          identity (verified by golden)
    cohort_hue_saturation_curves  huesaturation=colors=r+y+m:saturation=0:strength=0:lightness=0    identity (verified by golden)
    color_wheels                  [hue=h=<deg>] [colorbalance=rs..bh]                              yuva444p bridge (hue) then gbrap / rgba

Invariants
----------
* Parameters: ``effects.effect_scalar`` (authored key, else calibration default) clamped to the
  registry ``minimum`` / ``maximum`` exactly like ``ffmpeg._effect_scalar``; option strings go
  through ``ffmpeg._number`` so the port parses the very float the CPU filter string carries
  (``AV_OPT_TYPE_FLOAT`` options are float32, expression options are doubles).
* Anything a port cannot honour raises ``support.reject("effect (unsupported parameters)")``
  at plan time; nothing is approximated silently.  The BASIC handlers accept only the
  no-parameter default (``basic_effects.unsupported_basic_effect_reason`` already omits
  parameterised instances before they reach the IR; the ports re-check).
* Each port round-trips through the RGBA link on its own.  When the reference chains two
  YUV-only filters back to back (Tint then Sharpen, Color Wheels' ``hue`` then a Sharpen ...)
  libavfilter keeps the yuva444p link *between* them (no gbrap round trip), so such chains
  differ from these independent ports by up to a few tens of codes at the effect stage
  (measured: Tint -> Sharpen 1.0, max 28 / mean 0.17 code, delivered SSIM 0.9957); the fix is
  a core change (ports declare their native link, ``apply_effects`` fuses adjacent YUV ports),
  requested in the E1 report -- not approximated here.
* Reference departures that are *reproduced* here (ledger candidates, see the report):
  Vignette drops alpha (libavfilter negotiates ``rgb24``), so the clip becomes opaque after it;
  the yuva444p bridge nudges alpha 128..254 up by one (``rgbaToA_c``'s ``a<<6 | a>>2``).
* Measured on the goldens (arm64 ffmpeg 8.0.1): CPU float64 is bit-exact for every port --
  including gblur (float32 recurrence with fmadd in the reference, emulated step by step) and
  colorbalance (float32 math), whose gates stay at <= 1 code because x86 builds take SIMD paths
  with a different summation order; the default device (float32) is within one code.

Main callers:
- ``tensor/effects.py`` imports this module for its ``register`` side effects; the registry
  dispatches ``lower`` / ``apply``.

Why this exists:
The E1 batch is the "simple" effects, but each one is a real libavfilter kernel with its own
pixel format, fixed-point rounding and store rule; keeping them in one file (one owner) with
the negotiated-format table above is what lets the goldens prove them one by one against
the ``ffmpeg`` CLI on the same bytes.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import torch

from ..core.filter_text import format_number as cpu_number
from ..core.model import ResolvedEffect
from .color import code_to_premultiplied, premultiplied_to_code
from .effects import ApplyContext, EffectPort, LowerContext, effect_scalar, register
from .expr import geq_rgba, parse
from .support import reject
from .fx_color import LINK_COLORSPACE_ALIASES, BridgeLink, rgba8_to_yuva444p8, yuva444p8_to_rgba8


# --------------------------------------------------------------------------- shared helpers


def _calibration(effect: ResolvedEffect, key: str, name: str, default: float) -> float:
    """``ffmpeg._calibration``: registry calibration value ``name`` under parameter ``key``."""

    raw = effect.calibration.get(key, {}) if isinstance(effect.calibration, Mapping) else {}
    if not isinstance(raw, Mapping):
        return default
    try:
        return float(raw.get(name, default))
    except (TypeError, ValueError):
        return default


def _cpu_scalar(effect: ResolvedEffect, key: str, default: float) -> float:
    """``ffmpeg._effect_scalar``: authored value (else default) clamped to the registry min/max."""

    value = effect_scalar(effect, key, default)
    raw = effect.calibration.get(key, {}) if isinstance(effect.calibration, Mapping) else {}
    if not isinstance(raw, Mapping):
        return value
    minimum = float(raw.get("minimum", value))
    maximum = float(raw.get("maximum", value))
    return max(minimum, min(maximum, value))


def _option_float32(value: float) -> float:
    """An ``AV_OPT_TYPE_FLOAT`` option as ffmpeg stores it: the CPU string re-parsed, then cast to float."""

    return float(np.float32(float(cpu_number(value))))


def _require_no_params(effect: ResolvedEffect) -> None:
    if effect.params:
        names = sorted({p.name or p.key or "?" for p in effect.params})
        raise reject(
            "effect (unsupported parameters)",
            f"{effect.path}: {effect.name or effect.handler} default-only port received controls {names}",
        )


def _code8(canvas: torch.Tensor) -> torch.Tensor:
    """Premultiplied linear -> straight RGBA code values rounded to the 8-bit link (float tensor)."""

    return premultiplied_to_code(canvas).round().clamp(0.0, 255.0)


def _from_code8(code: torch.Tensor) -> torch.Tensor:
    return code_to_premultiplied(code)


def _lut_apply(code_plane: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """``lut[code]`` for an integer-valued float plane (LUT already on the plane's device/dtype)."""

    return lut[code_plane.long()]


def _floor_div(x: torch.Tensor, divisor: int) -> torch.Tensor:
    """C ``>>`` on signed ints (arithmetic shift == floor division by a power of two)."""

    return torch.div(x, divisor, rounding_mode="floor")


# --------------------------------------------------------------------------- YUV bridge (swscale)
# libavfilter auto-inserts ``scale`` (swscale, csp/range negotiated per link) around filters that
# only take YUV: rgba -> yuva444p before, yuva444p -> gbrap after.  The negotiated colourspace of
# that yuva444p link follows the *source's* tags: a BT.709-tagged source gives ``csp:bt709
# range:tv`` (Yunah-class footage), an untagged source gives ``csp:unknown`` = swscale's default
# ITU601 (verified with ``-v debug``: ``auto_scale_N ... -> fmt:yuva444p csp:bt709 range:tv``).
# Fixed-point transcription of swscale n8.0 for the 1:1 case, verified bit-exact against ffmpeg
# on all 256 ramps + a random plate for both matrices (test_tensor_fx_basic.py):
#   input.c   bgr32ToY_c/UV_c (RGB2YUV_SHIFT 15; the ITU601 table is special-cased in
#             utils.c fill_rgb2yuv_table, BT.709 comes from its generic ROUNDED_DIV derivation)
#             -> 14-bit ((S + off<<15 + 256) >> 9), hScale16To15 (RGB sh=13, 1:1 tap 1<<14) -> *2,
#             yuv2plane1_8_c: (v + 64) >> 7 (no dither: 8-bit source)
#   alpha     rgbaToA_c: a<<6 | a>>2 (14-bit) through the same path -> a + 1 for 128 <= a < 255
#   output.c  yuv2gbrp_full_X_c with hScale8To15 (Y8<<7), 12-bit vertical tap 4096,
#             yuv2rgb.c ff_yuv2rgb_c_init_tables (cy 76309, oy 16<<16, per-matrix crv/cbu/cgu/cgv)


# --------------------------------------------------------------------------- yuva444p bridge (shared)
#
# YUV-native filters (``colorize`` / ``eq`` for Tint, ``unsharp`` for Sharpen, ``hue`` for Color
# Wheels) run on the link libavfilter negotiates from the graph -- ``<pix_fmt>:<colorspace>:<range>``,
# see ``fx_color.reference_effect_link`` (source tags + whether a ``perspective`` stage precedes the
# effects) -- so the bridge is E4's swscale-exact ``rgba8_to_yuva444p8`` / ``yuva444p8_to_rgba8``
# (bit-exact on every triple for all eight links).  Payloads carry the link text.


def _bridge_link(effect: ResolvedEffect, ctx: LowerContext) -> str:
    """The negotiated bridge link for this layer, validated (rejects unported matrices / links)."""

    text = ctx.reference_effect_link
    if text is None:
        # No plan-side negotiation (direct port tests): a bare ``format=rgba`` link on the
        # source's tags, the way ``reference_effect_link`` resolves a non-resampling layer.
        csp = LINK_COLORSPACE_ALIASES.get(str(ctx.source_colorspace).lower(), str(ctx.source_colorspace).lower())
        rng = "pc" if str(ctx.source_color_range).lower() in ("pc", "jpeg", "full") else "tv"
        text = f"rgba:{csp}:{rng}"
    try:
        return str(BridgeLink.parse(text))
    except ValueError as error:
        raise reject(
            "effect (unsupported parameters)",
            f"{effect.path}: {effect.name or effect.handler} on a {text} link: {error}",
        ) from error


def _rgba_to_yuva444p(code: torch.Tensor, link: str) -> torch.Tensor:
    """Straight 0..255 integer-valued RGBA code [4,H,W] -> int64 YUVA planes [4,H,W] on ``link``."""

    if ":" not in link:  # bare colourspace (unit goldens): the limited-range rgba link
        link = f"rgba:{link}:tv"
    return rgba8_to_yuva444p8(code.to(torch.int32), BridgeLink.parse(link)).long()


def _yuva444p_to_rgba(yuva: torch.Tensor, dtype: torch.dtype, link: str) -> torch.Tensor:
    """int64 YUVA planes [4,H,W] -> straight 0..255 RGBA code [4,H,W] on ``link``."""

    if ":" not in link:
        link = f"rgba:{link}:tv"
    return yuva444p8_to_rgba8(yuva.to(torch.int32), BridgeLink.parse(link)).to(dtype)


# --------------------------------------------------------------------------- Negative (lutrgb)


@dataclass(frozen=True)
class NegativePayload:
    pass


_LUTS: dict[tuple, torch.Tensor] = {}


def _negative_lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``vf_lut.c``: ``lut[val] = av_clip((int)expr, 0, 255)`` with the expression folded in double."""

    key = ("negative", str(device), dtype)
    lut = _LUTS.get(key)
    if lut is None:
        values = []
        for val in range(256):
            inner = min(max(val / 255.0, 0.0), 1.0) ** 2.2
            curve = min(max(1.0 - inner, 0.0), 1.0) ** 0.45 * 255.0
            values.append(float(min(max(int(curve), 0), 255)))
        lut = torch.tensor(values, device=device, dtype=dtype)
        _LUTS[key] = lut
    return lut


def _lower_negative(effect: ResolvedEffect, ctx: LowerContext) -> NegativePayload:
    _require_no_params(effect)
    return NegativePayload()


def _apply_negative(payload: NegativePayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    code = _code8(canvas)
    lut = _negative_lut(canvas.device, canvas.dtype)
    rgb = _lut_apply(code[:3], lut)
    return _from_code8(torch.cat((rgb, code[3:4]), dim=0))


register(EffectPort(handler="negative", lower=_lower_negative, apply=_apply_negative))


# --------------------------------------------------------------------------- Threshold / Mirror (geq)
# The exact ``geq`` strings of ``basic_effects._threshold_filter`` / ``_mirror_filter``,
# evaluated by ``expr.geq_rgba`` (vf_geq.c on gbrap: bilinear samplers, uint8 truncation).

_THRESHOLD_LUMA = "0.2126*r(X,Y)+0.7152*g(X,Y)+0.0722*b(X,Y)"
_THRESHOLD_PROGRESS = f"clip((({_THRESHOLD_LUMA})-70.125)/63.75,0,1)"
_THRESHOLD_SMOOTH = f"pow({_THRESHOLD_PROGRESS},2)*(3-2*{_THRESHOLD_PROGRESS})"
THRESHOLD_GEQ: Mapping[str, str] = {
    "r": f"r(X,Y)*0.375+255*0.625*({_THRESHOLD_SMOOTH})",
    "g": f"g(X,Y)*0.375+255*0.625*({_THRESHOLD_SMOOTH})",
    "b": f"b(X,Y)*0.375+255*0.625*({_THRESHOLD_SMOOTH})",
    "a": "alpha(X,Y)",
}
_MIRROR_X = "min(W-1,W/2+abs(X-W/2))"
MIRROR_GEQ: Mapping[str, str] = {
    "r": f"r({_MIRROR_X},Y)",
    "g": f"g({_MIRROR_X},Y)",
    "b": f"b({_MIRROR_X},Y)",
    "a": f"alpha({_MIRROR_X},Y)",
}


@dataclass(frozen=True)
class GeqPayload:
    kind: str  # "threshold" | "mirror"


def _lower_geq(kind: str):
    def lower(effect: ResolvedEffect, ctx: LowerContext) -> GeqPayload:
        _require_no_params(effect)
        return GeqPayload(kind=kind)

    return lower


def _apply_geq(payload: GeqPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    strings = THRESHOLD_GEQ if payload.kind == "threshold" else MIRROR_GEQ
    expressions = {key: parse(text) for key, text in strings.items()}
    out = geq_rgba(expressions, _code8(canvas), frame_number=ctx.frame, time_seconds=ctx.seconds)
    return _from_code8(out)


register(EffectPort(handler="threshold", lower=_lower_geq("threshold"), apply=_apply_geq))
register(EffectPort(handler="mirror", lower=_lower_geq("mirror"), apply=_apply_geq))


# --------------------------------------------------------------------------- Colorize (colorchannelmixer)
# ``vf_colorchannelmixer.c``: per-channel LUTs ``lrint(i * coeff)`` summed, ``av_clip_uintp2(.., 8)``;
# alpha is ``lut[A][A][a] = lrint(a * 1.0)`` = identity (ra/ga/ba/aa defaults).

_COLORIZE_MATRIX = ((0.69, 0.27, 0.20), (0.087, 0.47, 0.08), (0.0, 0.0, 0.51))


@dataclass(frozen=True)
class ColorizePayload:
    pass


def _colorize_luts(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = ("colorize", str(device), dtype)
    luts = _LUTS.get(key)
    if luts is None:
        table = [[[float(round(i * coeff)) for i in range(256)] for coeff in row] for row in _COLORIZE_MATRIX]
        luts = torch.tensor(table, device=device, dtype=dtype)  # [out 3, in 3, 256]
        _LUTS[key] = luts
    return luts


def _lower_colorize(effect: ResolvedEffect, ctx: LowerContext) -> ColorizePayload:
    _require_no_params(effect)
    return ColorizePayload()


def _apply_colorize(payload: ColorizePayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    code = _code8(canvas)
    luts = _colorize_luts(canvas.device, canvas.dtype)
    idx = code[:3].long()
    outs = []
    for out_channel in range(3):
        acc = luts[out_channel][0][idx[0]] + luts[out_channel][1][idx[1]] + luts[out_channel][2][idx[2]]
        outs.append(acc.clamp(0.0, 255.0))
    return _from_code8(torch.cat((torch.stack(outs), code[3:4]), dim=0))


register(EffectPort(handler="colorize", lower=_lower_colorize, apply=_apply_colorize))


# --------------------------------------------------------------------------- Tint (colorize + eq, 601 bridge)
# ``vf_colorize.c`` with mix=1 keeps Y and writes constant U/V from hsl2rgb (float32) ->
# rgb2yuv (double, truncated to int); ``vf_eq.c`` process_c with contrast 1 / brightness -0.15
# adds a constant to Y (chroma + alpha copied).


def _hue2rgb(p: np.float32, q: np.float32, t: np.float32) -> np.float32:
    one, six = np.float32(1.0), np.float32(6.0)
    if t < 0:
        t = np.float32(t + one)
    if t > 1:
        t = np.float32(t - one)
    if t < np.float32(1.0 / 6.0):
        return np.float32(p + np.float32(np.float32(np.float32(q - p) * six) * t))
    if t < np.float32(0.5):
        return q
    if t < np.float32(2.0 / 3.0):
        return np.float32(p + np.float32(np.float32(np.float32(q - p) * np.float32(np.float32(2.0 / 3.0) - t)) * six))
    return p


def _colorize_yuv_constants(hue: float, saturation: float, lightness: float) -> tuple[int, int, int]:
    """``vf_colorize.c`` hsl2rgb (float32) + rgb2yuv (double, ``(int)`` store) for depth 8."""

    h = np.float32(np.float32(hue) / np.float32(360.0))
    s = np.float32(saturation)
    light = np.float32(lightness)
    if s == 0:
        r = g = b = light
    else:
        q = np.float32(light * np.float32(np.float32(1.0) + s)) if light < np.float32(0.5) else np.float32(np.float32(light + s) - np.float32(light * s))
        p = np.float32(np.float32(2.0) * light - q)
        r = _hue2rgb(p, q, np.float32(h + np.float32(1.0 / 3.0)))
        g = _hue2rgb(p, q, h)
        b = _hue2rgb(p, q, np.float32(h - np.float32(1.0 / 3.0)))
    r, g, b = float(r), float(g), float(b)
    y = int(((0.21260 * 219.0 / 255.0) * r + (0.71520 * 219.0 / 255.0) * g + (0.07220 * 219.0 / 255.0) * b) * 255)
    u = int((-(0.11457 * 224.0 / 255.0) * r - (0.38543 * 224.0 / 255.0) * g + (0.50000 * 224.0 / 255.0) * b + 0.5) * 255)
    v = int(((0.50000 * 224.0 / 255.0) * r - (0.45415 * 224.0 / 255.0) * g - (0.04585 * 224.0 / 255.0) * b + 0.5) * 255)
    return y, u, v


def _eq_brightness_offset(brightness: float) -> int:
    """``vf_eq.h`` process_c constant for contrast 1: ``((int)(100*b + 100) * 511) / 200 - 128 - 4096/32``."""

    brightness = float(np.float32(brightness))  # set_brightness: av_clipf(...) returns float
    contrast = int(1.0 * 256 * 16)
    return (int(100.0 * brightness + 100.0) * 511) // 200 - 128 - contrast // 32


@dataclass(frozen=True)
class TintPayload:
    u: int
    v: int
    y_offset: int
    link: str                    # negotiated yuva444p bridge link (fx_color.BridgeLink text)


def _lower_tint(effect: ResolvedEffect, ctx: LowerContext) -> TintPayload:
    _require_no_params(effect)
    _, u, v = _colorize_yuv_constants(_option_float32(245.54457), _option_float32(0.324662), _option_float32(0.270485))
    return TintPayload(u=u, v=v, y_offset=_eq_brightness_offset(-0.15), link=_bridge_link(effect, ctx))


def _apply_tint(payload: TintPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    yuva = _rgba_to_yuva444p(_code8(canvas), payload.link)
    y = (yuva[0] + payload.y_offset).clamp(0, 255)
    u = torch.full_like(y, payload.u)
    v = torch.full_like(y, payload.v)
    return _from_code8(_yuva444p_to_rgba(torch.stack((y, u, v, yuva[3])), canvas.dtype, payload.link))


register(EffectPort(handler="tint", lower=_lower_tint, apply=_apply_tint))


# --------------------------------------------------------------------------- Flipped (hflip)


@dataclass(frozen=True)
class FlippedPayload:
    pass


def _lower_flipped(effect: ResolvedEffect, ctx: LowerContext) -> FlippedPayload:
    _require_no_params(effect)
    return FlippedPayload()


def _apply_flipped(payload: FlippedPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    # A byte-exact horizontal flip on rgba; premultiplied linear flips identically, and the
    # 8-bit link is a no-op for a permutation, so no code round trip is needed.
    return canvas.flip(-1)


register(EffectPort(handler="flipped", lower=_lower_flipped, apply=_apply_flipped))


# --------------------------------------------------------------------------- Add Noise (noise, gbrap)
# ``vf_noise.c`` on gbrap (rgba -> gbrap is an exact shuffle): per plane an ``AVLFG`` seeded
# ``123457 + comp*31415`` (comp = gbrap plane index: 0 G, 1 B, 2 R) fills a 5120-entry int8
# gaussian table, then 3*4096 draws (prev_shift) and 4096 line shifts (first frame, non-temporal
# so every frame reuses them); row y adds ``noise[rand_shift[y & 4095] + x]``.  Alpha untouched.
# NOTE the seed: ``vf_noise.c`` init() overwrites every ``param[i].seed`` with ``all_seed`` when
# it is set, else with the constant 123457 -- the ``c0_seed=424242:c1_seed=..:c2_seed=..`` the CPU
# string carries never reach the generator (verified by the golden; ffmpeg n8.0 behaviour).

_MAX_NOISE, _MAX_SHIFT = 5120, 1024
_MAX_RES = _MAX_NOISE - _MAX_SHIFT
_NOISE_BASE_SEED = 123457
_NOISE_PLANES = ("c0", "c1", "c2")  # gbrap planes G, B, R
_NOISE_STRENGTH = 6
_MASK32 = 0xFFFFFFFF


class _AVLFG:
    """``libavutil/lfg.c``: 64-word lagged Fibonacci generator seeded through chained MD5 blocks."""

    def __init__(self, seed: int) -> None:
        self.state = [0] * 64
        tmp = bytearray(16)
        for i in range(8, 64, 4):
            tmp[0:4] = (seed & _MASK32).to_bytes(4, "little")
            tmp[4] = i
            digest = hashlib.md5(bytes(tmp)).digest()
            tmp[:] = digest
            for k in range(4):
                self.state[i + k] = int.from_bytes(digest[4 * k: 4 * k + 4], "little")
        self.index = 0

    def get(self) -> int:
        a = (self.state[(self.index - 24) & 63] + self.state[(self.index - 55) & 63]) & _MASK32
        self.state[self.index & 63] = a
        self.index += 1
        return a


def _noise_plane_tables(seed: int, strength: int) -> tuple[np.ndarray, np.ndarray]:
    """(noise table int8[5120], rand_shift int32[4096]) for one plane, exactly ``init_noise`` + frame 0."""

    lfg = _AVLFG(seed)
    noise = np.zeros(_MAX_NOISE, dtype=np.int64)
    uint_max_f = float(np.float32(0xFFFFFFFF))  # (float)UINT_MAX == 4294967296.0f
    for i in range(_MAX_NOISE):
        while True:
            x1 = 2.0 * lfg.get() / uint_max_f - 1.0
            x2 = 2.0 * lfg.get() / uint_max_f - 1.0
            w = x1 * x1 + x2 * x2
            if w < 1.0:
                break
        w = math.sqrt((-2.0 * math.log(w)) / w)
        y1 = x1 * w
        y1 *= strength / math.sqrt(3.0)
        y1 = float(np.float32(min(max(y1, -128.0), 127.0)))  # av_clipf returns float
        noise[i] = int(y1)
        lfg.get()  # RAND_N(6) (pattern bookkeeping only; the draw still advances the state)
    for _ in range(_MAX_RES * 3):
        lfg.get()  # prev_shift init
    rand_shift = np.array([lfg.get() & (_MAX_SHIFT - 1) for _ in range(_MAX_RES)], dtype=np.int64)
    return noise, rand_shift


_NOISE_CACHE: dict[tuple, torch.Tensor] = {}


def _noise_field(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """[3, H, W] additive noise for the R, G, B code planes (cached; the pattern is static per frame)."""

    key = (height, width, str(device), dtype)
    field = _NOISE_CACHE.get(key)
    if field is None:
        if width > _MAX_RES:
            raise ValueError(f"noise port: width {width} exceeds the single-chunk MAX_RES {_MAX_RES}")
        planes = {}
        for comp, option in enumerate(_NOISE_PLANES):
            table, shifts = _noise_plane_tables(_NOISE_BASE_SEED + comp * 31415, _NOISE_STRENGTH)
            rows = shifts[np.arange(height) & (_MAX_RES - 1)]
            planes[option] = table[rows[:, None] + np.arange(width)[None, :]]
        # gbrap plane order: c0 = G, c1 = B, c2 = R -> stack as R, G, B
        stacked = np.stack((planes["c2"], planes["c0"], planes["c1"])).astype(np.float64)
        field = torch.from_numpy(stacked).to(device=device, dtype=dtype)
        _NOISE_CACHE[key] = field
    return field


@dataclass(frozen=True)
class NoisePayload:
    pass


def _lower_noise(effect: ResolvedEffect, ctx: LowerContext) -> NoisePayload:
    _require_no_params(effect)
    if ctx.width > _MAX_RES:
        raise reject("effect (unsupported parameters)", f"{effect.path}: Add Noise canvas width {ctx.width} > {_MAX_RES}")
    return NoisePayload()


def _apply_noise(payload: NoisePayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    code = _code8(canvas)
    _, height, width = code.shape
    rgb = (code[:3] + _noise_field(height, width, canvas.device, canvas.dtype)).clamp(0.0, 255.0)
    return _from_code8(torch.cat((rgb, code[3:4]), dim=0))


register(EffectPort(handler="add_noise", lower=_lower_noise, apply=_apply_noise))


# --------------------------------------------------------------------------- Pixellate (pixelize avg 4x4)
# ``vf_pixelize.c`` on gbrap, planes 0x7 (G, B, R; alpha copied): every 4x4 block (edge blocks
# clipped to the plane) is filled with ``sum / (w*h)`` in integer arithmetic (floor).

_PIXELIZE_BLOCK = 4


@dataclass(frozen=True)
class PixellatePayload:
    pass


def _lower_pixellate(effect: ResolvedEffect, ctx: LowerContext) -> PixellatePayload:
    _require_no_params(effect)
    return PixellatePayload()


def _block_average_floor(planes: torch.Tensor, block: int) -> torch.Tensor:
    """Integer block mean (floor) with clipped edge blocks, expanded back to the plane size."""

    _, height, width = planes.shape
    ones = torch.ones((1, height, width), device=planes.device, dtype=planes.dtype)
    sums = torch.nn.functional.avg_pool2d(planes.unsqueeze(0), block, stride=block, ceil_mode=True, count_include_pad=False, divisor_override=1)
    counts = torch.nn.functional.avg_pool2d(ones.unsqueeze(0), block, stride=block, ceil_mode=True, count_include_pad=False, divisor_override=1)
    fill = torch.div(sums.round(), counts.round(), rounding_mode="floor")
    expanded = fill.repeat_interleave(block, dim=2).repeat_interleave(block, dim=3)
    return expanded[0, :, :height, :width]


def _apply_pixellate(payload: PixellatePayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    code = _code8(canvas)
    rgb = _block_average_floor(code[:3], _PIXELIZE_BLOCK)
    return _from_code8(torch.cat((rgb, code[3:4]), dim=0))


register(EffectPort(handler="pixellate_default", lower=_lower_pixellate, apply=_apply_pixellate))


# --------------------------------------------------------------------------- Gaussian (gblur, gbrap, all planes)
# ``vf_gblur.c`` (Alvarez-Mazorra recursive filter, ``steps=1``): float32 buffer, per row
# ``p[0]*=bscale; p[x] += nu*p[x-1] (right); p[W-1]*=bscale; p[x-1] += nu*p[x] (left)``, the
# same down/up per column, ``*= postscale*postscaleV``, clip, ``lrintf``.  Parameters from
# ``set_params`` (double, from the float sigma).  Emulated step by step in float32 (each
# ``+=`` rounded to float32 like the reference's fmadd) so CPU float64 stays within one code.


@dataclass(frozen=True)
class GaussianPayload:
    sigma: float          # the float32 option value ffmpeg parses from the CPU string
    nu: float
    boundaryscale: float
    postscale: float      # postscale * postscaleV (float32 product)


def _gblur_params(sigma32: float, steps: int = 1) -> tuple[float, float, float]:
    """``vf_gblur.c`` set_params: (nu, boundaryscale, postscale) as the float32 fields ffmpeg keeps."""

    sigma = np.float32(sigma32)
    lam = float(np.float32(sigma * sigma)) / (2.0 * steps)  # float*float in single, then double division
    if lam == 0.0:
        return 0.0, 1.0, 1.0  # dnu = NaN -> !isnormal -> nu 0, scales 1 (identity)
    dnu = (1.0 + 2.0 * lam - math.sqrt(1.0 + 4.0 * lam)) / (2.0 * lam)
    postscale = float(np.float32(math.pow(dnu / lam, steps)))
    boundaryscale = float(np.float32(1.0 / (1.0 - dnu)))
    nu = float(np.float32(dnu))
    if not (math.isfinite(postscale) and postscale != 0.0):
        postscale = 1.0
    if not (math.isfinite(boundaryscale) and boundaryscale != 0.0):
        boundaryscale = 1.0
    if not (math.isfinite(nu) and nu != 0.0):
        nu = 0.0
    return nu, boundaryscale, postscale


def _lower_gaussian(effect: ResolvedEffect, ctx: LowerContext) -> GaussianPayload:
    amount = _cpu_scalar(effect, "9999/986883370/100/986883376/2/100", 0.0)
    boost = _cpu_scalar(effect, "9999/986883370/100/986884620/2/100", 1.0)
    sigma_scale = _calibration(effect, "9999/986883370/100/986883376/2/100", "ffmpeg_sigma", 20.0)
    sigma = _option_float32(max(0.0, amount * boost * sigma_scale))
    nu, boundaryscale, postscale = _gblur_params(sigma)
    return GaussianPayload(sigma=sigma, nu=nu, boundaryscale=boundaryscale, postscale=float(np.float32(np.float32(postscale) * np.float32(postscale))))


def _f32_step(value: torch.Tensor) -> torch.Tensor:
    """Round a float64 tensor to float32 precision (float32 tensors pass through)."""

    return value.float().to(value.dtype) if value.dtype == torch.float64 else value


def _gblur_axis(buffer: torch.Tensor, nu: float, bscale: float, axis: int) -> torch.Tensor:
    """One causal + anti-causal recursive pass along ``axis`` (1 = rows/vertical, 2 = columns/horizontal)."""

    work = buffer.movedim(axis, -1).contiguous()  # [..., L] (a copy; the loop writes into it)
    length = work.shape[-1]
    lines = work.view(-1, length)
    lines[:, 0] = _f32_step(lines[:, 0] * bscale)
    for x in range(1, length):
        lines[:, x] = _f32_step(lines[:, x] + nu * lines[:, x - 1])
    lines[:, length - 1] = _f32_step(lines[:, length - 1] * bscale)
    for x in range(length - 1, 0, -1):
        lines[:, x - 1] = _f32_step(lines[:, x - 1] + nu * lines[:, x])
    return lines.view(work.shape).movedim(-1, axis)


def gblur_planes(code_planes: torch.Tensor, payload: GaussianPayload) -> torch.Tensor:
    """``vf_gblur.c`` filter_frame on integer-valued 8-bit planes ``[P, H, W]`` -> ``lrintf`` result."""

    if payload.sigma <= 0.0:
        return code_planes
    buffer = _f32_step(code_planes.clone())
    buffer = _gblur_axis(buffer, payload.nu, payload.boundaryscale, axis=2)  # horizontal (rows)
    buffer = _gblur_axis(buffer, payload.nu, payload.boundaryscale, axis=1)  # vertical (columns)
    buffer = _f32_step(buffer * payload.postscale).clamp(0.0, 255.0)
    return torch.round(buffer)  # lrintf: half to even == torch.round


def _apply_gaussian(payload: GaussianPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    return _from_code8(gblur_planes(_code8(canvas), payload))


register(EffectPort(handler="gaussian", lower=_lower_gaussian, apply=_apply_gaussian))


# --------------------------------------------------------------------------- Sharpen (unsharp luma, 601 bridge)
# ``vf_unsharp.c`` ``unsharp=5:5:A:5:5:0`` on yuva444p: only plane 0 (Y) is filtered (chroma
# and alpha amounts are 0 -> plane copy).  The sr/sc accumulator cascade is the separable
# binomial [1 4 6 4 1] (sum 256 = 1 << scalebits) with edge replication;
#   res = y + (((y - ((blur + 128) >> 8)) * amount) >> 16),  amount = (int)(A * 65536.0)


@dataclass(frozen=True)
class SharpenPayload:
    amount: int  # ``fp->amount`` (16.16 fixed point, truncated like the C double->int store)
    link: str                    # negotiated yuva444p bridge link (fx_color.BridgeLink text)


def _lower_sharpen(effect: ResolvedEffect, ctx: LowerContext) -> SharpenPayload:
    amount = _cpu_scalar(effect, "9999/986883553/100/986883554/2/100", 0.0)
    scale = _calibration(effect, "9999/986883553/100/986883554/2/100", "ffmpeg_scale", 2.0)
    option = _option_float32(max(0.0, amount * scale))
    return SharpenPayload(amount=int(option * 65536.0), link=_bridge_link(effect, ctx))


_BINOMIAL5 = (1.0, 4.0, 6.0, 4.0, 1.0)


def unsharp_luma(y8: torch.Tensor, amount: int) -> torch.Tensor:
    """``vf_unsharp.c`` slice on one int64 8-bit plane ``[H, W]``."""

    if amount == 0:
        return y8
    kernel_1d = torch.tensor(_BINOMIAL5, device=y8.device, dtype=torch.float32)
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, 5, 5)
    padded = torch.nn.functional.pad(y8.to(torch.float32).view(1, 1, *y8.shape), (2, 2, 2, 2), mode="replicate")
    # integer sums <= 255 * 256 are exact in float32
    blur = torch.nn.functional.conv2d(padded, kernel).view(y8.shape).round().long()
    blurred = _floor_div(blur + 128, 256)
    res = y8 + _floor_div((y8 - blurred) * amount, 1 << 16)
    return res.clamp(0, 255)


def _apply_sharpen(payload: SharpenPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    yuva = _rgba_to_yuva444p(_code8(canvas), payload.link)
    y = unsharp_luma(yuva[0], payload.amount)
    return _from_code8(_yuva444p_to_rgba(torch.stack((y, yuva[1], yuva[2], yuva[3])), canvas.dtype, payload.link))


register(EffectPort(handler="sharpen", lower=_lower_sharpen, apply=_apply_sharpen))


# --------------------------------------------------------------------------- Vignette (rgb24: alpha dropped)
# ``vf_vignette.c`` negotiates rgb24 (no alpha format in its list), so the reference clip is
# opaque afterwards (rgb24 -> gbrap fills alpha with 255).  Per pixel ``f = fmap[y][x]``
# (float32 of ``cos(angle*dnorm)^4``, 0 outside the ellipse), each channel
# ``(uint8_t)(int)(src*f + dv)`` where ``dv`` is the default dither: an LCG
# ``s = s*1664525 + 1013904223`` (uint32, seeded 0, one draw per channel sample, never reset)
# so frame n consumes samples [3WHn, 3WH(n+1)).  ``eval=frame`` re-evaluates the constant
# angle each frame (identical map).

_LCG_A, _LCG_C = 1664525, 1013904223


@dataclass(frozen=True)
class VignettePayload:
    angle: float  # float32 of clip(angle, 0, pi/2) (av_clipf returns float)


def _lower_vignette(effect: ResolvedEffect, ctx: LowerContext) -> VignettePayload:
    strength = _cpu_scalar(effect, "9999/987213582/3001385021/1/200/202", 0.65)
    size = _cpu_scalar(effect, "9999/987213582/3001385021/3/987213589/1", 1.5)
    angle_scale = _calibration(effect, "9999/987213582/3001385021/1/200/202", "ffmpeg_angle_scale", math.pi / 4)
    angle = max(0.01, min(math.pi / 2, strength * angle_scale / max(size, 0.25)))
    parsed = float(cpu_number(angle))  # expression option: parsed as a double
    clipped = min(max(np.float32(parsed), np.float32(0.0)), np.float32(math.pi / 2))
    return VignettePayload(angle=float(clipped))


_VIGNETTE_CACHE: dict[tuple, torch.Tensor] = {}


def vignette_fmap(height: int, width: int, angle: float, device: torch.device) -> torch.Tensor:
    """``get_natural_factor`` over the frame as the float32 map ffmpeg keeps (returned as float32)."""

    key = (height, width, angle, str(device))
    fmap = _VIGNETTE_CACHE.get(key)
    if fmap is None:
        x0, y0 = width / 2.0, height / 2.0
        dmax = math.hypot(width / 2.0, height / 2.0)
        xs = np.trunc((np.arange(width) - x0) * 1.0).astype(np.float64)   # (int)((x - x0) * xscale)
        ys = np.trunc((np.arange(height) - y0) * 1.0).astype(np.float64)
        dnorm = np.hypot(xs[None, :], ys[:, None]) / dmax
        c = np.cos(angle * dnorm)
        factor = np.where(dnorm > 1.0, 0.0, (c * c) * (c * c)).astype(np.float32)
        fmap = torch.from_numpy(factor).to(device)
        _VIGNETTE_CACHE[key] = fmap
    return fmap


def _lcg_affine_power(steps: int) -> tuple[int, int]:
    """(A, C) with ``state_{k+steps} = A*state_k + C (mod 2^32)`` for the vignette LCG."""

    a, c = 1, 0
    base_a, base_c = _LCG_A, _LCG_C
    while steps:
        if steps & 1:
            a, c = (base_a * a) & _MASK32, (base_a * c + base_c) & _MASK32
        base_a, base_c = (base_a * base_a) & _MASK32, (base_a * base_c + base_c) & _MASK32
        steps >>= 1
    return a, c


def _lcg_block(count: int, device: torch.device) -> torch.Tensor:
    """States 0..count-1 of the LCG from seed 0 as int64 (cached per count/device)."""

    key = ("lcg", count, str(device))
    block = _VIGNETTE_CACHE.get(key)
    if block is None:
        states = np.zeros(1, dtype=np.uint64)
        length = 1
        while length < count:
            a, c = _lcg_affine_power(length)
            nxt = (np.uint64(a) * states + np.uint64(c)) & np.uint64(_MASK32)
            states = np.concatenate((states, nxt))
            length *= 2
        block = torch.from_numpy(states[:count].astype(np.int64)).to(device)
        _VIGNETTE_CACHE[key] = block
    return block


def _mulmod32(a: int, states: torch.Tensor) -> torch.Tensor:
    """``(a * states) mod 2^32`` on int64 tensors without overflowing (a, states < 2^32)."""

    lo, hi = a & 0xFFFF, a >> 16
    return torch.remainder(lo * states + torch.remainder(hi * states, 1 << 16) * (1 << 16), 1 << 32)


def vignette_dither(frame: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """The dither values ``dv`` for layer-local frame ``frame`` as ``[3, H, W]`` (R, G, B sample order)."""

    count = 3 * height * width
    a, c = _lcg_affine_power(frame * count)
    states = torch.remainder(_mulmod32(a, _lcg_block(count, device)) + c, 1 << 32)
    return (states.to(dtype) / float(1 << 32)).view(height, width, 3).permute(2, 0, 1)


def vignette_rgb(code_rgb: torch.Tensor, angle: float, frame: int) -> torch.Tensor:
    """``vf_vignette.c`` RGB path on integer-valued code planes ``[3, H, W]``."""

    _, height, width = code_rgb.shape
    fmap = vignette_fmap(height, width, angle, code_rgb.device)
    scaled = (code_rgb.float() * fmap).to(code_rgb.dtype)  # ``srcp[0] * f`` is a float32 product
    dither = vignette_dither(frame, height, width, code_rgb.device, code_rgb.dtype)
    return (scaled + dither).trunc().clamp(0.0, 255.0)  # av_clip_uint8((int)double)


def _apply_vignette(payload: VignettePayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    code = _code8(canvas)
    rgb = vignette_rgb(code[:3], payload.angle, ctx.frame)
    opaque = torch.full_like(code[3:4], 255.0)  # rgb24 -> gbrap: alpha becomes 255
    return _from_code8(torch.cat((rgb, opaque), dim=0))


register(EffectPort(handler="vignette", lower=_lower_vignette, apply=_apply_vignette))


# --------------------------------------------------------------------------- Color Curves / Hue-Sat Curves (no-ops)
# ``cohort_effects.cohort_effect_filters`` emits ``curves=master='0/0 1/1'`` and
# ``huesaturation=colors=r+y+m:saturation=0:strength=0:lightness=0``; both are byte-exact
# identities on rgba (golden in test_tensor_fx_basic.py), so the ports pass the canvas through.


@dataclass(frozen=True)
class IdentityPayload:
    handler: str


def _lower_identity(effect: ResolvedEffect, ctx: LowerContext) -> IdentityPayload:
    if effect.data:
        raise reject("effect (unsupported parameters)", f"{effect.path}: {effect.name or effect.handler} carries opaque filter data")
    return IdentityPayload(handler=effect.handler or "")


def _apply_identity(payload: IdentityPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    return canvas


register(EffectPort(handler="cohort_color_curves", lower=_lower_identity, apply=_apply_identity))
register(EffectPort(handler="cohort_hue_saturation_curves", lower=_lower_identity, apply=_apply_identity))


# --------------------------------------------------------------------------- Color Wheels (hue + colorbalance)
# ``color._color_wheels_filters``: ``hue=h=<deg>`` when hue != 0 (vf_hue.c: 16.16 rotation LUT on
# U/V, Y and alpha copied, through the yuva444p bridge) and ``colorbalance=rs=..:bh=..`` when the
# temperature / tint deltas are non-zero (vf_colorbalance.c float32 math on gbrap or rgba;
# preserve_lightness 0, alpha copied).


@dataclass(frozen=True)
class ColorWheelsPayload:
    hue_deg: Optional[float]      # None when the hue stage is not emitted
    red: Optional[float]          # colorbalance rs=rm=rh (float32 option) or None when not emitted
    green: Optional[float]
    blue: Optional[float]
    link: str                    # negotiated yuva444p bridge link for the hue stage


def _lower_color_wheels(effect: ResolvedEffect, ctx: LowerContext) -> ColorWheelsPayload:
    temperature = _cpu_scalar(effect, "8890", 5000.0)
    tint = _cpu_scalar(effect, "8891", 0.0)
    hue = _cpu_scalar(effect, "8892", 0.0)
    temperature_delta = (temperature - 5000.0) * _calibration(effect, "8890", "ffmpeg_scale", 0.00006)
    tint_delta = tint * _calibration(effect, "8891", "ffmpeg_scale", 0.006)
    hue_deg = float(cpu_number(hue)) if abs(hue) > 1e-6 else None
    if abs(temperature_delta) > 1e-6 or abs(tint_delta) > 1e-6:
        red = _option_float32(max(-1.0, min(1.0, temperature_delta)))
        green = _option_float32(max(-1.0, min(1.0, tint_delta)))
        blue = _option_float32(max(-1.0, min(1.0, -temperature_delta)))
    else:
        red = green = blue = None
    return ColorWheelsPayload(hue_deg=hue_deg, red=red, green=green, blue=blue, link=_bridge_link(effect, ctx))


def hue_rotate_uv(u8: torch.Tensor, v8: torch.Tensor, hue_deg: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``vf_hue.c`` create_chrominance_lut / apply_lut on int64 U/V planes (saturation 1)."""

    hue = hue_deg * math.pi / 180
    s = int(round(math.sin(hue) * (1 << 16) * 1.0))  # lrint (half-even; the argument is never a tie)
    c = int(round(math.cos(hue) * (1 << 16) * 1.0))
    u = u8 - 128
    v = v8 - 128
    new_u = _floor_div((c * u) - (s * v) + (1 << 15) + (128 << 16), 1 << 16).clamp(0, 255)
    new_v = _floor_div((s * u) + (c * v) + (1 << 15) + (128 << 16), 1 << 16).clamp(0, 255)
    return new_u, new_v


def colorbalance_rgb(code_rgb: torch.Tensor, red: float, green: float, blue: float) -> torch.Tensor:
    """``vf_colorbalance.c`` (pl=0) with equal shadows/midtones/highlights per channel, float32 math."""

    rgb = code_rgb.float() / np.float32(255.0)
    light = rgb.amax(dim=0) + rgb.amin(dim=0)  # ``l = FFMAX3 + FFMIN3``
    a, b, scale = np.float32(4.0), np.float32(0.333), np.float32(0.7)
    shadow_w = ((b - light) * a + np.float32(0.5)).clamp(0.0, 1.0) * scale
    mid_w = ((light - b) * a + np.float32(0.5)).clamp(0.0, 1.0) * ((np.float32(1.0) - light - b) * a + np.float32(0.5)).clamp(0.0, 1.0) * scale
    high_w = ((light + b - np.float32(1.0)) * a + np.float32(0.5)).clamp(0.0, 1.0) * scale
    outs = []
    for channel, amount in enumerate((red, green, blue)):
        v = rgb[channel]
        v = v + np.float32(amount) * shadow_w
        v = v + np.float32(amount) * mid_w
        v = v + np.float32(amount) * high_w
        outs.append((v.clamp(0.0, 1.0) * np.float32(255.0)).round().clamp(0.0, 255.0))  # lrintf
    return torch.stack(outs).to(code_rgb.dtype)


def _apply_color_wheels(payload: ColorWheelsPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    if payload.hue_deg is None and payload.red is None:
        return canvas
    code = _code8(canvas)
    if payload.hue_deg is not None:
        yuva = _rgba_to_yuva444p(code, payload.link)
        u, v = hue_rotate_uv(yuva[1], yuva[2], payload.hue_deg)
        code = _yuva444p_to_rgba(torch.stack((yuva[0], u, v, yuva[3])), canvas.dtype, payload.link)
    if payload.red is not None:
        assert payload.green is not None and payload.blue is not None
        code = torch.cat((colorbalance_rgb(code[:3], payload.red, payload.green, payload.blue), code[3:4]), dim=0)
    return _from_code8(code)


register(EffectPort(handler="color_wheels", lower=_lower_color_wheels, apply=_apply_color_wheels))
