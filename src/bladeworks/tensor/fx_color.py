"""E4 effect ports: the colour pipeline -- Color Adjustments and Color Board (owner: E4 batch).

Architecture map
================

    plan.build_tensor_plan -> effects.lower_effect -> _lower_color(effect, ctx)
        color.color_effect_filters(effect)          : the CPU emitter's EXACT filter strings
                                                      (same parameter lookup, clamps, ``.12g``
                                                      formatting the ffmpeg CLI would parse)
        ctx.reference_effect_link                   : which pixel link the reference feeds the
                                                      stack through (``BridgeLink``, computed by
                                                      the plan via ``reference_effect_link``)
        _parse_stage(filter_string, link)           : one ffmpeg filter -> one frozen stage
                                                      (unknown filter / option -> loud reject)
        ColorPipelinePayload(stages)                : the port payload (hashable, no torch)

    renderer._FrameComposer.placed -> effects.apply_effects -> _apply_color(payload, canvas)
        color.premultiplied_to_code                 : premultiplied linear -> straight 0..255 code
        round to the uint8 frame the reference sees (``format=rgba``)
        stages in order:
            YuvBridgeStage   : swscale RGB -> yuva444p (8-bit, lossy; matrix / range / alpha
                               path per the negotiated ``BridgeLink``) -> ``eq`` per-plane
                               LUTs -> swscale yuva444p -> RGB
            LutStage         : per-channel 8-bit LUT (``colorlevels``, ``curves``)
            ColorBalanceStage: ``colorbalance`` float32 three-zone math (``pl=0``)
        color.code_to_premultiplied                 : straight code -> premultiplied linear

Reference filter chains (``src/bladeworks/core/color.py``; ffmpeg 8.0.1 auto-negotiates
the pixel formats, ``ffmpeg -v verbose`` shows the two ``auto_scale`` insertions):

    Color Adjustments (always emits ``eq``):
        RGB --auto_scale--> yuva444p  eq=brightness:contrast:saturation:gamma
            --auto_scale--> RGB       [colorlevels=rimin=gimin=bimin=k]  (black point > 0)
                                      [colorbalance=rs:bs:rm:bm:rh:bh:gs:gm:gh]  (warmth/tint)
    Color Board (every filter optional):
        [eq=saturation=s]  (same yuva444p bridge)   [curves=master='5 points']   [colorbalance=...]

Which bridge: the negotiated ``eq`` link (documented here so E1/E2 can reuse the pattern)
--------------------------------------------------------------------------------------
``eq`` only accepts 8-bit YUV formats, so libavfilter inserts ``scale`` (libswscale) around
it.  What that scale computes depends on three facts of the *reference graph*, all of which
were read off ``ffmpeg -v verbose`` on real reference graphs and are modelled by
``BridgeLink`` = ``<pix_fmt>:<colorspace>:<range>``:

* ``pix_fmt`` of the RGB frame entering ``eq``: ``rgba`` after a bare ``format=rgba`` chain
  (same-size clip, trim crop, ``conform none``, folded group effects) or ``gbrap`` after a
  ``perspective`` stage (conform resample, camera crop, Ken Burns) -- ``perspective`` only
  supports 8-bit planar formats.  It decides the alpha path (``rgbaToA_c`` replicates bit 7:
  A >= 128 gains one code; ``planar_rgb_to_a`` is exact).
* ``colorspace`` of the YUV link: libavfilter propagates the *source stream's* colorspace tag
  through the RGB filters to the YUV link -- ``bt709`` for tagged camera media, unknown /
  ``gbr`` (PNG rasters, titles) or ``bt470bg`` (JPEG) -> the ``SWS_CS_DEFAULT`` (601) table.
  Others (bt2020nc, smpte240m, fcc, ycgco) are not ported and reject loudly.
* ``range`` of the YUV link: the source's range tag (``tv`` / unknown -> limited; ``pc`` for
  JPEG / yuvj sources -> full), or ``pc`` whenever ``setparams=range=full`` (every
  ``perspective`` stage) precedes the effects.

``reference_effect_link(...)`` derives the link from plan facts; the plan passes it as
``LowerContext.reference_effect_link``.  All eight links were verified bit-exact against
``ffmpeg`` on all 2**24 RGB triples and all 2**24 YUV triples (``probe_swscale`` run,
2026-08-17; the test file re-checks each on a 1M-pixel plate).  No dithering anywhere (8-bit
source: ``ff_sws_pb_64`` -> ``+64 >> 7``).

RGB -> YUV (``input.c`` ``rgb32ToY_c`` / ``rgb32ToUV_c`` = ``planar_rgb_to_y`` / ``_uv``,
``hScale16To15`` (x2, FFMIN 32767), ``lumRangeToJpeg_c`` / ``chrRangeToJpeg_c`` for ``pc``,
``yuv2plane1_8_c``):

    RY..BV = rgb2yuv_table(colorspace)          # fill_rgb2yuv_table (utils.c), 15-bit, ROUNDED_DIV
      bt601: 8414 16519 3208 | -4865 -9528 14392 | 14392 -12061 -2332   (fixed constants)
      bt709: 5983 20127 2032 | -3298 -11094 14392 | 14392 -13073 -1320
    y14 = (RY*R + GY*G + BY*B + (32  << 14) + (1 << 8)) >> 9      # 15-bit sum -> 14-bit, offset 16
    u14 = (RU*R + GU*G + BU*B + (256 << 14) + (1 << 8)) >> 9      # offset 128
    v14 = (RV*R + GV*G + BV*B + (256 << 14) + (1 << 8)) >> 9
    x15 = min(2*x14, 32767)
    pc:  x15 = min((x15*coeff + offset) >> 14, 32767)  with (coeff, offset) = solve_range_convert
         (n8.0.1: luma (19078, -39084288), chroma (18652, -38207488); n8.0 lacked the +2**13
         rounding term and lands 0.4% of codes one lower)
    X8  = clip((x15 + 64) >> 7)
    A8' = rgba: min(A8 + (A8 >= 128), 255) ; gbrap: A8

YUV -> RGB (``output.c`` ``yuv2rgb_write_full`` for ``rgba``, ``yuv2gbrp_full_X_c`` for
``gbrap`` -- identical arithmetic; ``ff_yuv2rgb_c_init_tables`` per (colorspace, range)):

    (y_coeff, y_off>>9, v2r, v2g, u2g, u2b) = yuv2rgb_coefficients(colorspace, full_range)
      bt601 tv: 9539 16 13075 -6660 -3209 16525      bt601 pc: 8192 0 11485 -5850 -2819 14516
      bt709 tv: 9539 16 14686 -4366 -1747 17305      bt709 pc: 8192 0 12901 -3835 -1534 15201
    Y  = Y8 << 9 ; U = (U8 - 128) << 9 ; V = (V8 - 128) << 9
    t  = (Y - (y_off << 9)) * y_coeff + (1 << 21)
    R  = t + V*v2r ; G = t + V*v2g + U*u2g ; B = t + U*u2b       (32-bit int arithmetic)
    if (R|G|B) & 0xC0000000: each is av_clip_uintp2(x, 30)      -> negative OR wrapped-past-2**31 -> 0
    R8 = R >> 22 (etc.);  A8 copied exactly.
    In the >>9-reduced form used here: T = (Y8-y_off)*y_coeff + 4096 + chroma terms;
    R8 = 0 if T < 0 or T >= 2**22 (the B plane really reaches the 2**31 wrap for Y8=255, U8=255
    in the limited sets and swscale returns 0 there -- reproduced, not "fixed"), 255 if
    T >= 2**21, else T >> 13.

Post-``eq`` precision: when the effect stack is followed by a spatial tail (static transform /
corner pin: ``format=rgba64le,lutrgb...``), libavfilter negotiates the post-``eq`` RGB link to
``rgba64le`` and ``colorlevels`` / ``colorbalance`` / ``curves`` run at 16 bits until the
tail's ``perspective`` quantises to 8-bit ``gbrap``; the port always runs them at 8 bits (the
no-tail negotiation), which is within one code of that -- soft-match territory, recorded in
the E4 report as a ledger candidate.

``eq`` (``vf_eq.c``, per plane, alpha plane copied): luma uses ``process_c`` when gamma == 1
(``c = int(contrast*4096)``, ``b = ((int)(100*brightness+100)*511)/200 - 128 - c/32`` in C
(truncating) division, controls float32-rounded like ``av_clipf`` -- ``eq_process_lut``,
``pel = clip(((src*c) >> 12) + b)``) and the double LUT ``256.0 * pow(contrast*(v-0.5)+0.5+
brightness, 1/gamma)`` (truncated, 0 for v <= 0, 255 for v >= 1) otherwise; chroma uses
``process_c`` with ``contrast = saturation`` and ``brightness = 0``, or is copied when
saturation == 1.0.  All three are 8-bit LUTs, computed at plan time in Python integers /
doubles.  The C x86 SIMD path (``psllw 4; pmulhw``) equals ``process_c`` bit for bit.

``colorlevels`` (8-bit RGB, packed or planar): ``imin = lrint(k*255)``, ``coeff = float32(255/(255-imin))``,
``dst = clip(trunc(float32(src - imin) * coeff))``; alpha coeff 1.0 -> copied.
``colorbalance`` (8-bit RGB, float32, ``pl=0``): ``l = max(r,g,b) + min(r,g,b)`` on ``x/255``;
zone weights ``clipf((0.333 - l)*4 + 0.5) * 0.7`` (shadows), ``clipf((l-0.333)*4+0.5) *
clipf((1-l-0.333)*4+0.5) * 0.7`` (midtones), ``clipf((l+0.333-1)*4+0.5) * 0.7`` (highlights);
``v = clipf(v + s + m + h)``; ``lrintf(v*255)`` (half-to-even); alpha copied.  The
reference binary contracts ``x*a + 0.5f`` into FMA on arm64, so this stage is <= 1 code, not
bit-exact, and the goldens gate it at 1.
``curves=master=...`` (8-bit RGB): natural cubic spline through the points in double
(``interpolate()``), ``clip((int)(y*255))`` per code, applied to R, G, B; alpha copied.

Precision contract: every stage is 8-bit-in / 8-bit-out exactly like the reference link
between filters (the reference's bridge loss IS the calibrated look and is reproduced, not
avoided); the input frame is the working canvas' straight code rounded to the nearest
integer (the uint8 ``format=rgba`` / ``gbrap`` frame the reference feeds ``eq``).  Integer
stages run in int32 on the canvas device (MPS-safe: no int64, no float64).

Main callers:
- ``effects.lower_effect`` / ``effects.apply_effects`` through the registry
  (``EFFECT_PORTS["color_adjustments"]``, ``EFFECT_PORTS["color_board"]``).

Why this exists:
Color Adjustments is the one calibrated colour handler in the corpus, and its look is
inseparable from the accidental 8-bit YUV bridge the CPU graph pays for ``eq``.  A
"cleaner" float port would not soft-match the reference (the oracle in
``evidence/color_adjustments_shared_plan_v3.json`` is +-1 code against that bridge), so
the port reproduces libswscale / libavfilter integer semantics exactly and keeps the whole
parameter lowering on the CPU emitter (``color.color_effect_filters``) so both backends can
never disagree about what the authored controls mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Mapping, Sequence

import numpy as np
import torch

from ..core.color import color_effect_filters
from ..core.model import ResolvedEffect
from .color import code_to_premultiplied, premultiplied_to_code
from .effects import OVERSCAN_EXTEND, ApplyContext, EffectPort, LowerContext, register
from .support import reject

# --------------------------------------------------------------------------- swscale model
#
# The pixel link libavfilter negotiates around ``eq`` (see the module doc "which bridge"):
# input pixel format (alpha path), the YUV colour matrix (the *source stream's* colorspace
# tag propagates to the YUV link) and the YUV range (source range tag, or ``pc`` after
# ``setparams=range=full``).  Everything below is libswscale n8.0.1 integer arithmetic,
# verified bit-exact against ``ffmpeg`` on all 2**24 triples for every variant.

LINK_FORMATS: Final = ("rgba", "gbrap")
LINK_COLORSPACES: Final = ("bt601", "bt709")   # bt601 = the SWS_CS_DEFAULT table (unknown / gbr / bt470bg / smpte170m)
LINK_RANGES: Final = ("tv", "pc")

# ff_yuv2rgb_coeffs (yuv2rgb.c): {crv, cbu, cgu, cgv} per colourspace; the 601 row is SWS_CS_DEFAULT.
_CSP_TABLES: Final[Mapping[str, tuple[int, int, int, int]]] = {
    "bt601": (104597, 132201, 25675, 53279),
    "bt709": (117489, 138438, 13975, 34925),
}
# ffprobe / libavfilter colourspace names -> the swscale table they select (sws_getCoefficients:
# RGB(0) / unspecified(2) / reserved(3) / bt470bg(5) / smpte170m(6) / >10 all use the default row).
LINK_COLORSPACE_ALIASES: Final[Mapping[str, str]] = {
    "bt709": "bt709",
    "bt601": "bt601", "bt470bg": "bt601", "smpte170m": "bt601", "unknown": "bt601",
    "unspecified": "bt601", "reserved": "bt601", "gbr": "bt601", "rgb": "bt601",
}


@dataclass(frozen=True)
class BridgeLink:
    """The negotiated ``eq`` link: ``<pix_fmt>:<colorspace>:<range>``, e.g. ``rgba:bt709:tv``."""

    pix_fmt: str        # "rgba" (rgbaToA_c alpha quirk) | "gbrap" (alpha exact)
    colorspace: str     # "bt601" | "bt709"
    color_range: str    # "tv" (limited, also unknown) | "pc" (full)

    @staticmethod
    def parse(text: str) -> "BridgeLink":
        """``<pix_fmt>:<colorspace>:<range>`` -> link; ``ValueError`` names what is not portable."""

        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError(f"reference effect link {text!r} is not '<pix_fmt>:<colorspace>:<range>'")
        pix_fmt, colorspace, color_range = parts
        if pix_fmt not in LINK_FORMATS:
            raise ValueError(f"reference effect link {text!r}: pixel format {pix_fmt!r} not in {LINK_FORMATS}")
        if colorspace not in LINK_COLORSPACES:
            raise ValueError(
                f"reference effect link {text!r}: colourspace {colorspace!r} is not one of the ported eq "
                f"bridge matrices {LINK_COLORSPACES} (aliases {sorted(set(LINK_COLORSPACE_ALIASES))})"
            )
        if color_range not in LINK_RANGES:
            raise ValueError(f"reference effect link {text!r}: range {color_range!r} not in {LINK_RANGES}")
        return BridgeLink(pix_fmt, colorspace, color_range)

    @property
    def full_range(self) -> bool:
        return self.color_range == "pc"

    def __str__(self) -> str:
        return f"{self.pix_fmt}:{self.colorspace}:{self.color_range}"


def _c_div(a: int, b: int) -> int:
    """C integer division (truncates toward zero)."""

    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _rounded_div(a: int, b: int) -> int:
    """FFmpeg ``ROUNDED_DIV`` (b > 0): ``(a +- b/2) / b`` with C truncation."""

    return _c_div(a + (b >> 1) if a >= 0 else a - (b >> 1), b)


def _round_to_int16(value: int) -> int:
    """``roundToInt16`` (yuv2rgb.c) without the +-0x7FFF saturation (never reached here)."""

    return (value + (1 << 15)) >> 16


@lru_cache(maxsize=None)
def rgb2yuv_table(colorspace: str) -> tuple[int, int, int, int, int, int, int, int, int]:
    """``fill_rgb2yuv_table`` (utils.c): (RY, GY, BY, RU, GU, BU, RV, GV, BV), 15-bit fixed point.

    ``dstRange`` is forced to 0 in that function (range conversion happens elsewhere), so the
    table is always the limited-range one; the default (601) table is overwritten with the
    fixed ``(int)(coef * 219/255 [Y] or 224/255 [UV] * 32768 + 0.5)`` constants.
    """

    table = _CSP_TABLES[colorspace]
    vr, ub, ug, vg = table[0], table[1], -table[2], -table[3]
    one = 65536
    cy = one * 255 // 219
    w = _rounded_div(one * one * ug, ub)
    v = _rounded_div(one * one * vg, vr)
    z = one * one - w - v
    c_y, c_u, c_v = _rounded_div(cy * z, one), _rounded_div(ub * z, one), _rounded_div(vr * z, one)
    s = 1 << 15
    ry, gy, by = -_rounded_div(s * v, c_y), _rounded_div(s * one * one, c_y), -_rounded_div(s * w, c_y)
    ru, gu, bu = _rounded_div(s * v, c_u), -_rounded_div(s * one * one, c_u), _rounded_div(s * (z + w), c_u)
    rv, gv, bv = _rounded_div(s * (v + z), c_v), -_rounded_div(s * one * one, c_v), _rounded_div(s * w, c_v)
    if table == _CSP_TABLES["bt601"]:
        by, bv, bu = int(0.114 * 219 / 255 * s + 0.5), -int(0.081 * 224 / 255 * s + 0.5), int(0.500 * 224 / 255 * s + 0.5)
        gy, gv, gu = int(0.587 * 219 / 255 * s + 0.5), -int(0.419 * 224 / 255 * s + 0.5), -int(0.331 * 224 / 255 * s + 0.5)
        ry, rv, ru = int(0.299 * 219 / 255 * s + 0.5), int(0.500 * 224 / 255 * s + 0.5), -int(0.169 * 224 / 255 * s + 0.5)
    return ry, gy, by, ru, gu, bu, rv, gv, bv


@dataclass(frozen=True)
class _Yuv2Rgb:
    """``ff_yuv2rgb_c_init_tables`` (yuv2rgb.c) for one (colourspace, source range): the six int16
    coefficients ``yuv2rgb_write_full`` / ``yuv2gbrp_full_X_c`` use (brightness 0, contrast 1,
    saturation 1)."""

    y_coeff: int
    y_offset: int   # already >> 9 (the port works on the accumulator >> 9)
    v2r: int
    v2g: int
    u2g: int
    u2b: int


@lru_cache(maxsize=None)
def yuv2rgb_coefficients(colorspace: str, full_range: bool) -> _Yuv2Rgb:
    """``ff_yuv2rgb_c_init_tables`` for ``ff_yuv2rgb_coeffs[colorspace]`` and the YUV source range."""

    crv, cbu, cgu, cgv = _CSP_TABLES[colorspace]
    cgu, cgv = -cgu, -cgv
    cy, oy = 1 << 16, 0
    if full_range:
        crv, cbu, cgu, cgv = (_c_div(v * 224, 255) for v in (crv, cbu, cgu, cgv))
    else:
        cy = (cy * 255) // 219
        oy = 16 << 16
    return _Yuv2Rgb(
        y_coeff=_round_to_int16(cy << 13),
        y_offset=_round_to_int16(oy << 9) >> 9,
        v2r=_round_to_int16(crv << 13), v2g=_round_to_int16(cgv << 13),
        u2g=_round_to_int16(cgu << 13), u2b=_round_to_int16(cbu << 13),
    )


def _solve_range_convert(src_min: int, src_max: int, dst_min: int, dst_max: int) -> tuple[int, int]:
    """``solve_range_convert`` (swscale.c, n8.0.1) for an 8-bit destination: 15-bit source,
    ``src_shift`` 7, ``mult_shift`` 14; the ``+ (1 << 13)`` rounding term in the offset is
    the n8.0.1 fix (n8.0 lacked it and lands 0.4% of codes one lower)."""

    src_shift, mult_shift = 7, 14
    total_shift = mult_shift + src_shift
    coeff = -((-(((dst_max - dst_min) << total_shift) // (src_max - src_min))) >> src_shift)
    offset = (dst_max << total_shift) - (src_max << src_shift) * coeff + (1 << (mult_shift - 1))
    return coeff, offset


_LUM_TO_JPEG: Final = _solve_range_convert(16, 235, 0, 255)   # (19078, -39084288)
_CHR_TO_JPEG: Final = _solve_range_convert(16, 240, 0, 255)   # (18652, -38207488)


# --------------------------------------------------------------------------- payload types


@dataclass(frozen=True)
class YuvBridgeStage:
    """RGB -> yuva444p -> ``eq`` -> RGB: the bridge variant plus three per-plane 8-bit LUTs."""

    link: BridgeLink     # the negotiated eq link (pix_fmt / colourspace / range)
    y_lut: tuple[int, ...]
    u_lut: tuple[int, ...]
    v_lut: tuple[int, ...]


@dataclass(frozen=True)
class LutStage:
    """Per-channel 8-bit LUT on straight RGB (``colorlevels``, ``curves``); alpha copied."""

    r_lut: tuple[int, ...]
    g_lut: tuple[int, ...]
    b_lut: tuple[int, ...]


@dataclass(frozen=True)
class ColorBalanceStage:
    """``colorbalance`` option values as float32-rounded floats (rs gs bs rm gm bm rh gh bh)."""

    shadows: tuple[float, float, float]
    midtones: tuple[float, float, float]
    highlights: tuple[float, float, float]


Stage = YuvBridgeStage | LutStage | ColorBalanceStage


@dataclass(frozen=True)
class ColorPipelinePayload:
    filters: tuple[str, ...]      # the exact CPU filter strings (diagnostics)
    stages: tuple[Stage, ...]


# --------------------------------------------------------------------------- lowering


def _split_options(handler_path: str, name: str, args: str) -> dict[str, str]:
    """``k=v:k=v`` -> dict; quoted values (``master='...'``) unquoted.  Positional -> reject."""

    options: dict[str, str] = {}
    for item in _split_top_level(args):
        if not item:
            continue
        if "=" not in item:
            raise reject("effect (unsupported parameters)", f"{handler_path}: {name} positional option {item!r}")
        key, value = item.split("=", 1)
        options[key.strip()] = value.strip().strip("'\"")
    return options


def _split_top_level(args: str) -> list[str]:
    """Split on ``:`` outside single quotes (``curves=master='0/0 0.2/0.21 ...'``)."""

    parts: list[str] = []
    current: list[str] = []
    quoted = False
    for char in args:
        if char == "'":
            quoted = not quoted
        if char == ":" and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def _option(options: dict[str, str], key: str, default: float) -> float:
    return float(options.pop(key)) if key in options else default


def _reject_leftovers(handler_path: str, name: str, options: dict[str, str]) -> None:
    if options:
        raise reject(
            "effect (unsupported parameters)",
            f"{handler_path}: {name} option(s) {sorted(options)} are not ported",
        )


def _clipf(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clipf32(value: float, low: float, high: float) -> float:
    """``av_clipf`` semantics: the double is rounded to float32 (the parameter type) and clipped."""

    return _clipf(float(np.float32(value)), low, high)


# ---- eq ------------------------------------------------------------------------------------


def eq_process_lut(contrast: float, brightness: float) -> tuple[int, ...]:
    """``process_c`` (vf_eq.h) as a LUT: ``((src * c) >> 12) + b`` clipped.

    ``vf_eq.c`` stores every control through ``av_clipf`` (a *float* function), so the LUT is
    built from the float32-rounded values promoted back to double: ``brightness=-0.28`` makes
    ``100*b + 100`` = 71.9999998... -> ``(int)`` 71 (Y-75), where a plain double gives 72 (Y-73).
    Callers that already went through ``_clipf32`` are unaffected (rounding is idempotent);
    direct callers (``fx_branched`` Callout constants) rely on it.  Divisions use C truncation
    (``c`` is negative for negative contrast).

    Main callers: ``_eq_stage`` (E4 eq bridge), ``fx_branched`` (Callout field / chroma LUTs).
    """

    contrast = float(np.float32(contrast))
    brightness = float(np.float32(brightness))
    c = int(contrast * 256 * 16)
    b = _c_div(int(100.0 * brightness + 100.0) * 511, 200) - 128 - _c_div(c, 32)
    return tuple(min(255, max(0, ((src * c) >> 12) + b)) for src in range(256))


def _eq_gamma_lut(contrast: float, brightness: float, gamma: float, gamma_weight: float = 1.0) -> tuple[int, ...]:
    """``create_lut`` (vf_eq.c): double math, ``(uint8_t)(256.0 * v)`` truncation."""

    g = 1.0 / gamma
    lw = 1.0 - gamma_weight
    lut = []
    for i in range(256):
        v = i / 255.0
        v = contrast * (v - 0.5) + 0.5 + brightness
        if v <= 0.0:
            lut.append(0)
            continue
        v = v * lw + math.pow(v, g) * gamma_weight
        lut.append(255 if v >= 1.0 else int(256.0 * v))
    return tuple(lut)


_IDENTITY_LUT: Final = tuple(range(256))


def _eq_stage(handler_path: str, options: dict[str, str], link: BridgeLink) -> YuvBridgeStage:
    """``eq=...`` -> per-plane LUTs with vf_eq.c's ``check_values`` dispatch (option clamps included)."""

    # ``av_clipf`` takes *float* arguments: every eq control is rounded to float32 on its way
    # into the double fields (``brightness=-0.4`` really is -0.4000000059604645, which is what
    # makes ``(int)(100*b + 100)`` = 59, not 60).
    contrast = _clipf32(_option(options, "contrast", 1.0), -1000.0, 1000.0)
    brightness = _clipf32(_option(options, "brightness", 0.0), -1.0, 1.0)
    saturation = _clipf32(_option(options, "saturation", 1.0), 0.0, 3.0)
    gamma = _clipf32(_option(options, "gamma", 1.0), 0.1, 10.0)
    _reject_leftovers(handler_path, "eq", options)   # gamma_r/g/b/weight/eval are never emitted
    # Luma: param[0] = (contrast, brightness, gamma * gamma_g=1).
    if contrast == 1.0 and brightness == 0.0 and gamma == 1.0:
        y_lut = _IDENTITY_LUT
    elif gamma == 1.0 and abs(contrast) < 7.9:
        y_lut = eq_process_lut(contrast, brightness)
    else:
        y_lut = _eq_gamma_lut(contrast, brightness, gamma)
    # Chroma: param[1..2] = (saturation, 0, sqrt(gamma_b/gamma_g) = 1.0).
    if saturation == 1.0:
        chroma_lut = _IDENTITY_LUT
    elif abs(saturation) < 7.9:
        chroma_lut = eq_process_lut(saturation, 0.0)
    else:  # unreachable (saturation is clipped to 3.0) but keeps the C dispatch complete
        chroma_lut = _eq_gamma_lut(saturation, 0.0, 1.0)
    return YuvBridgeStage(link=link, y_lut=y_lut, u_lut=chroma_lut, v_lut=chroma_lut)


# ---- colorlevels ---------------------------------------------------------------------------


def _colorlevels_lut(in_min: float, in_max: float, out_min: float, out_max: float) -> tuple[int, ...]:
    """8-bit ``colorlevels`` channel LUT: ``clip((int)(float32(src - imin) * coeff + omin))``."""

    imin = int(round(in_min * 255))   # lrint (half-even; the emitted values never sit on .5)
    imax = int(round(in_max * 255))
    omin = int(round(out_min * 255))
    omax = int(round(out_max * 255))
    if imin < 0 or imax < 0:
        raise ValueError("colorlevels auto (negative) input levels are not emitted by the reference")
    coeff = np.float32((omax - omin) / float(imax - imin))
    values = np.float32(np.arange(256) - imin) * coeff + np.float32(omin)
    return tuple(int(v) for v in np.clip(np.trunc(values.astype(np.float64)), 0, 255))


def _colorlevels_stage(handler_path: str, options: dict[str, str]) -> LutStage:
    luts = []
    for channel in "rgb":
        luts.append(_colorlevels_lut(
            _option(options, f"{channel}imin", 0.0), _option(options, f"{channel}imax", 1.0),
            _option(options, f"{channel}omin", 0.0), _option(options, f"{channel}omax", 1.0),
        ))
    # Alpha levels default to identity (aimin=0 aimax=1 aomin=0 aomax=1); the emitter never sets them.
    for key in ("aimin", "aimax", "aomin", "aomax", "preserve"):
        if key in options:
            raise reject("effect (unsupported parameters)", f"{handler_path}: colorlevels {key} is not ported")
    _reject_leftovers(handler_path, "colorlevels", options)
    return LutStage(*luts)


# ---- colorbalance --------------------------------------------------------------------------


def _colorbalance_stage(handler_path: str, options: dict[str, str]) -> ColorBalanceStage:
    def zone(suffix: str) -> tuple[float, float, float]:
        return tuple(  # type: ignore[return-value]
            float(np.float32(_clipf(_option(options, f"{channel}{suffix}", 0.0), -1.0, 1.0)))
            for channel in "rgb"
        )

    shadows, midtones, highlights = zone("s"), zone("m"), zone("h")
    if _option(options, "pl", 0.0) != 0.0:
        raise reject("effect (unsupported parameters)", f"{handler_path}: colorbalance pl=1 (preserve lightness) is not ported")
    _reject_leftovers(handler_path, "colorbalance", options)
    return ColorBalanceStage(shadows=shadows, midtones=midtones, highlights=highlights)


# ---- curves --------------------------------------------------------------------------------


def _curves_lut(points: Sequence[tuple[float, float]]) -> tuple[int, ...]:
    """``interpolate()`` (vf_curves.c, natural cubic spline, 8-bit): the master LUT."""

    scale = 255
    n = len(points)
    if n == 0:
        return _IDENTITY_LUT
    if n == 1:
        return tuple([min(255, max(0, int(points[0][1] * scale)))] * 256)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    r = [0.0] * n
    for i in range(1, n - 1):
        r[i] = 6 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
    bd = [0.0] * n
    md = [0.0] * n
    ad = [0.0] * n
    md[0] = md[n - 1] = 1.0
    for i in range(1, n - 1):
        bd[i] = h[i - 1]
        md[i] = 2 * (h[i - 1] + h[i])
        ad[i] = h[i]
    for i in range(1, n):
        den = md[i] - bd[i] * ad[i - 1]
        k = 1.0 / den if den else 1.0
        ad[i] *= k
        r[i] = (r[i] - bd[i] * r[i - 1]) * k
    for i in range(n - 2, -1, -1):
        r[i] = r[i] - ad[i] * r[i + 1]
    lut = [0] * 256

    def clip8(value: float) -> int:
        return min(255, max(0, int(value)))   # av_clip_uint8((int) v): C truncation

    for i in range(int(xs[0] * scale)):
        lut[i] = clip8(ys[0] * scale)
    for i in range(n - 1):
        yc, yn = ys[i], ys[i + 1]
        a = yc
        b = (yn - yc) / h[i] - h[i] * r[i] / 2.0 - h[i] * (r[i + 1] - r[i]) / 6.0
        c = r[i] / 2.0
        d = (r[i + 1] - r[i]) / (6.0 * h[i])
        x_start = int(xs[i] * scale)
        x_end = int(xs[i + 1] * scale)
        for x in range(x_start, x_end + 1):
            xx = (x - x_start) * 1.0 / scale
            yy = a + b * xx + c * xx * xx + d * xx * xx * xx
            lut[x] = clip8(yy * scale)
    for i in range(int(xs[-1] * scale), 256):
        lut[i] = clip8(ys[-1] * scale)
    return tuple(lut)


def _curves_stage(handler_path: str, options: dict[str, str]) -> LutStage:
    master = options.pop("master", options.pop("m", None))
    if master is None:
        raise reject("effect (unsupported parameters)", f"{handler_path}: curves without a master curve")
    _reject_leftovers(handler_path, "curves", options)   # r/g/b/all/preset/psfile/interp never emitted
    points: list[tuple[float, float]] = []
    for token in master.split():
        x_text, y_text = token.split("/")
        x, y = float(x_text), float(y_text)
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise reject("effect (unsupported parameters)", f"{handler_path}: curves point {token!r} outside [0, 1]")
        if points and int(points[-1][0] * 255) >= int(x * 255):
            raise reject("effect (unsupported parameters)", f"{handler_path}: curves points not strictly increasing ({token!r})")
        points.append((x, y))
    lut = _curves_lut(points)
    return LutStage(lut, lut, lut)   # component curves are identity, composed with master


_STAGE_PARSERS: Final = {
    "colorlevels": _colorlevels_stage,
    "colorbalance": _colorbalance_stage,
    "curves": _curves_stage,
}


def _parse_stage(handler_path: str, filter_string: str, link: BridgeLink | None) -> Stage:
    name, _, args = filter_string.partition("=")
    if name == "eq":
        if link is None:
            raise reject(
                "effect (unsupported parameters)",
                f"{handler_path}: the plan did not resolve which pixel link the reference feeds "
                "this effect stack (LowerContext.reference_effect_link is unset; see "
                "fx_color.reference_effect_link)",
            )
        return _eq_stage(handler_path, _split_options(handler_path, name, args), link)
    parser = _STAGE_PARSERS.get(name)
    if parser is None:
        raise reject("effect (unsupported parameters)", f"{handler_path}: reference filter {name!r} is not ported")
    return parser(handler_path, _split_options(handler_path, name, args))


def reference_effect_link(
    *,
    source_color_space: str | None,
    source_color_range: str | None,
    source_width: int,
    source_height: int,
    canvas_width: int,
    canvas_height: int,
    conform: str,
    crop_mode: str | None,
    folded_group_effect: bool = False,
) -> str:
    """Which pixel link the reference graph feeds an effect stack through (``<fmt>:<csp>:<range>``).

    Read from ``ffmpeg._video_chain`` / ``_geometry_stage_filters`` / ``GeometryPlan._conform_filters``
    and verified with ``ffmpeg -v verbose`` on real reference graphs (see the module doc):

    * pixel format + range: the effect stack follows the pre-effect geometry stages, which
      contain a ``perspective`` (always preceded by ``setparams=range=full`` and negotiated in
      8-bit planar ``gbrap``) exactly when the conform resamples (Fit / Fill with a source size
      different from the canvas), the crop is a camera crop, or the crop is an animated pan
      (Ken Burns) -> ``gbrap`` / ``pc``.  Otherwise the stack follows a bare ``format=rgba``
      (+ ``crop`` / ``pad`` for trim / ``conform none``) -> ``rgba`` with the *source's* range
      tag (``tv`` for tv / unknown, ``pc`` for full-range sources such as JPEG rasters).  Static
      transforms / corner pins live in the spatial *tail* (after effects).  Folded *group*
      effects run on the group surface after a bare ``format=rgba`` -> ``rgba`` / source range.
    * colourspace: libavfilter propagates the source stream's colorspace tag to the YUV link
      (``bt709`` for tagged camera media, unknown / ``gbr`` for PNG rasters -> the 601 default,
      ``bt470bg`` for JPEG); anything outside the two ported matrices rejects loudly.

    Main callers:
    - ``plan._effect_specs`` (requested wiring: passes the result as
      ``LowerContext.reference_effect_link``); tests.
    """

    # Unported matrices are carried through by name (e.g. ``rgba:bt2020nc:tv``) and reject only
    # when a chain actually needs the bridge (``_lower_color``), so a bt2020 source without
    # colour effects never trips the plan.
    csp_name = (source_color_space or "unknown").lower()
    colorspace = LINK_COLORSPACE_ALIASES.get(csp_name, csp_name)
    resamples = (not folded_group_effect) and (
        crop_mode in ("crop", "pan")
        or (conform in ("fit", "fill") and (source_width, source_height) != (canvas_width, canvas_height))
    )
    if resamples:
        return str(BridgeLink("gbrap", colorspace, "pc"))
    range_name = (source_color_range or "tv").lower()
    color_range = "pc" if range_name in ("pc", "jpeg", "full") else "tv"
    return str(BridgeLink("rgba", colorspace, color_range))


def _lower_color(effect: ResolvedEffect, ctx: LowerContext) -> ColorPipelinePayload:
    """Lower one Color Adjustments / Color Board effect from the CPU emitter's filter strings.

    Everything about *what the authored controls mean* (key/name lookup, registry
    ``ffmpeg_scale`` / clamps / defaults, ``.12g`` formatting) stays in
    ``color.color_effect_filters`` -- the port only interprets the emitted ffmpeg filters, so
    the two backends cannot drift apart on parameter semantics.  Any emitted filter or option
    the port does not implement is a loud ``effect (unsupported parameters)`` reject.

    The ``eq`` bridge link comes from ``ctx.reference_effect_link`` (see
    ``reference_effect_link``).  TODO(E4 -> core): ``effects.LowerContext`` does not carry that
    field yet; until ``plan._effect_specs`` passes it, any chain that needs the bridge rejects
    loudly rather than guess (the links differ by whole tone-curve pivots and matrices, not by
    a rounding code).  The requested diff is in the E4 report.
    """

    handler_path = f"{ctx.clip_path}: {effect.name or effect.handler}"
    filters = tuple(color_effect_filters(effect))
    link_text = getattr(ctx, "reference_effect_link", None)
    link: BridgeLink | None = None
    if link_text is not None and any(item.startswith("eq=") for item in filters):
        try:
            link = BridgeLink.parse(link_text)
        except ValueError as error:
            raise reject("effect (unsupported parameters)", f"{handler_path}: {error}") from None
    stages = tuple(_parse_stage(handler_path, item, link) for item in filters)
    return ColorPipelinePayload(filters=filters, stages=stages)


# --------------------------------------------------------------------------- apply


@lru_cache(maxsize=256)
def _lut_tensor(lut: tuple[int, ...], device: str) -> torch.Tensor:
    return torch.tensor(lut, dtype=torch.int32, device=device)


def _floor_shift(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Arithmetic ``>>`` on int32 tensors (floor division; MPS-safe)."""

    return torch.div(values, 1 << bits, rounding_mode="floor")


def rgba8_to_yuva444p8(rgba: torch.Tensor, link: BridgeLink) -> torch.Tensor:
    """swscale 8-bit RGB -> ``yuva444p`` on ``[4, H, W]`` int32 (RGBA plane order in; see the module doc).

    ``rgb32ToY_c`` / ``planar_rgb_to_y`` (same arithmetic) with ``rgb2yuv_table(link.colorspace)``,
    ``hScale16To15`` (x2, FFMIN 32767), then for a ``pc`` link ``lumRangeToJpeg_c`` /
    ``chrRangeToJpeg_c`` on the 15-bit intermediate, then ``yuv2plane1_8_c`` (``+64 >> 7``, no
    dither).  Alpha: ``rgba`` follows ``rgbaToA_c`` (+1 for A >= 128, saturating at 255),
    ``gbrap`` follows ``planar_rgb_to_a`` (exact).  Bit-exact against ffmpeg 8.0.1 on every RGB
    triple for all eight links.
    """

    ry, gy, by, ru, gu, bu, rv, gv, bv = rgb2yuv_table(link.colorspace)
    r, g, b, a = rgba[0], rgba[1], rgba[2], rgba[3]

    def plane(cr: int, cg: int, cb: int, offset: int, to_jpeg: tuple[int, int]) -> torch.Tensor:
        x14 = _floor_shift(cr * r + cg * g + cb * b + (offset << 14) + (1 << 8), 9)
        x15 = torch.clamp(x14 * 2, max=32767)
        if link.full_range:
            coeff, shift_offset = to_jpeg
            x15 = torch.clamp(_floor_shift(x15 * coeff + shift_offset, 14), max=32767)
        return torch.clamp(_floor_shift(x15 + 64, 7), 0, 255)

    y = plane(ry, gy, by, 32, _LUM_TO_JPEG)
    u = plane(ru, gu, bu, 256, _CHR_TO_JPEG)
    v = plane(rv, gv, bv, 256, _CHR_TO_JPEG)
    alpha = a if link.pix_fmt == "gbrap" else torch.clamp(a + (a >= 128).to(torch.int32), max=255)
    return torch.stack((y, u, v, alpha))


def yuva444p8_to_rgba8(yuva: torch.Tensor, link: BridgeLink) -> torch.Tensor:
    """swscale ``yuva444p`` -> 8-bit RGB on ``[4, H, W]`` int32 (RGBA plane order out).

    ``yuv2rgb_write_full`` (``rgba``) / ``yuv2gbrp_full_X_c`` (``gbrap``) -- the same arithmetic --
    with ``yuv2rgb_coefficients(link.colorspace, link.full_range)``.  Bit-exact against ffmpeg
    8.0.1 on every YUV triple for all eight links, including the 32-bit wrap of the B
    accumulator in the limited-range sets (Y8=255 with U8 near 255 -> 0); alpha copied.
    """

    k = yuv2rgb_coefficients(link.colorspace, link.full_range)
    y, u, v, a = yuva[0], yuva[1], yuva[2], yuva[3]
    t = (y - k.y_offset) * k.y_coeff + (1 << 21 >> 9)
    du, dv = u - 128, v - 128

    def channel(total: torch.Tensor) -> torch.Tensor:
        # total is the C accumulator >> 9 (exact: every term is a multiple of 2**9).
        wrapped_or_negative = (total < 0) | (total >= (1 << 22))
        saturated = total >= (1 << 21)
        return torch.where(wrapped_or_negative, torch.zeros_like(total),
                           torch.where(saturated, torch.full_like(total, 255), _floor_shift(total, 13)))

    r = channel(t + dv * k.v2r)
    g = channel(t + dv * k.v2g + du * k.u2g)
    b = channel(t + du * k.u2b)
    return torch.stack((r, g, b, a))


def _apply_lut(values: torch.Tensor, lut: tuple[int, ...]) -> torch.Tensor:
    if lut is _IDENTITY_LUT:
        return values
    return _lut_tensor(lut, str(values.device))[values.long()]


def _apply_bridge(stage: YuvBridgeStage, rgba: torch.Tensor) -> torch.Tensor:
    yuva = rgba8_to_yuva444p8(rgba, stage.link)
    yuva = torch.stack((
        _apply_lut(yuva[0], stage.y_lut), _apply_lut(yuva[1], stage.u_lut),
        _apply_lut(yuva[2], stage.v_lut), yuva[3],
    ))
    return yuva444p8_to_rgba8(yuva, stage.link)


def _apply_luts(stage: LutStage, rgba: torch.Tensor) -> torch.Tensor:
    return torch.stack((
        _apply_lut(rgba[0], stage.r_lut), _apply_lut(rgba[1], stage.g_lut),
        _apply_lut(rgba[2], stage.b_lut), rgba[3],
    ))


def _apply_colorbalance(stage: ColorBalanceStage, rgba: torch.Tensor) -> torch.Tensor:
    """``color_balance8`` (vf_colorbalance.c) in float32, ``pl=0``."""

    x = rgba[:3].to(torch.float32) / 255.0
    lightness = torch.amax(x, dim=0) + torch.amin(x, dim=0)   # ``l`` in vf_colorbalance.c
    a, b, scale = 4.0, 0.333, 0.7
    w_s = torch.clamp((b - lightness) * a + 0.5, 0.0, 1.0) * scale
    w_m = torch.clamp((lightness - b) * a + 0.5, 0.0, 1.0) * torch.clamp((1.0 - lightness - b) * a + 0.5, 0.0, 1.0) * scale
    w_h = torch.clamp((lightness + b - 1) * a + 0.5, 0.0, 1.0) * scale
    outputs = []
    for index in range(3):
        v = x[index]
        v = v + stage.shadows[index] * w_s
        v = v + stage.midtones[index] * w_m
        v = v + stage.highlights[index] * w_h
        v = torch.clamp(v, 0.0, 1.0)
        outputs.append(torch.clamp(torch.round(v * 255.0), 0.0, 255.0).to(torch.int32))
    return torch.stack((*outputs, rgba[3]))


def apply_stages(stages: Sequence[Stage], rgba8: torch.Tensor) -> torch.Tensor:
    """Run the parsed reference stages on an int32 ``[4, H, W]`` straight-RGBA 0..255 frame."""

    for stage in stages:
        if isinstance(stage, YuvBridgeStage):
            rgba8 = _apply_bridge(stage, rgba8)
        elif isinstance(stage, LutStage):
            rgba8 = _apply_luts(stage, rgba8)
        elif isinstance(stage, ColorBalanceStage):
            rgba8 = _apply_colorbalance(stage, rgba8)
        else:  # pragma: no cover - the payload is built by _parse_stage only
            raise TypeError(f"unknown colour stage {stage!r}")
    return rgba8


def _apply_color(payload: ColorPipelinePayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    if not payload.stages:
        return canvas
    code = premultiplied_to_code(canvas)
    rgba8 = torch.clamp(torch.round(code), 0.0, 255.0).to(torch.int32)
    rgba8 = apply_stages(payload.stages, rgba8)
    return code_to_premultiplied(rgba8.to(canvas.dtype))


# Per-pixel LUT / bridge / balance stages: bit-identical on any surface (``"extend"``).
register(EffectPort(handler="color_adjustments", lower=_lower_color, apply=_apply_color, overscan=OVERSCAN_EXTEND))
register(EffectPort(handler="color_board", lower=_lower_color, apply=_apply_color, overscan=OVERSCAN_EXTEND))
