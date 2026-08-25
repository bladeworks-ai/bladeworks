"""E3 effect port: Callout (``cohort_callout``, the Reframe/Callout Motion template).

The Phase 4 Cartoon, Camcorder, Focus Blur, and Drop Shadow ports live in
``fx_cohort.py`` so this module remains the calibrated Callout implementation.

Architecture map
================

    effects.lower_effect(effect, ctx)  -> _lower_callout      : default-only (any authored control
                                                                 rejects); resolves the YUV link the
                                                                 reference negotiates for this layer
                                                                 (``ctx.reference_effect_link``, pixel
                                                                 format forced to ``rgba`` by the graph's
                                                                 own ``format=rgba``); freezes the ``eq``
                                                                 LUTs and the layer time base
    effects.apply_effects(...)         -> _apply_callout      : the reference graph, stage by stage, on
                                                                 the straight 0..255 RGBA code frame

The reference graph (``cohort_effects._callout_graph_lines``, emitted through
``ffmpeg._branched_effect_graph`` at the SAME stage as ordinary effects: after crop / conform, before
the spatial tail) and the pixel links libavfilter negotiates for it (read off ``ffmpeg -v debug`` on
real reference graphs, 320x180 bt709 sources, with and without a ``perspective`` conform stage):

    [in]format=rgba,split=3[original][fieldsource][inset]                       rgba
    [fieldsource] gblur=sigma=4:steps=2:planes=0x7, eq=brightness=-0.28:saturation=0.6   [field]
                  -> auto_scale rgba->yuva444p (E4 bridge, ``rgba8_to_yuva444p8``), gblur on Y/U/V
                     (float32 IIR, ``tr_equirect.gblur_gbrap``), eq = ``process_c`` LUTs (luma
                     brightness -0.28 -> Y-75; chroma contrast 0.6), alpha copied
    [original][field] blend=all_expr='A*(1-s(T))+B*s(T)'                        [builtin]  yuva444p
                  -> ``original`` goes through the same auto_scale; ``blend_expr_8bit`` evaluates the
                     expression per plane (Y, U, V, A) in double and stores with C truncation;
                     ``T`` = pts * av_q2d(link time base) with pts counted from the layer's stream
                     start (``setpts=PTS-STARTPTS`` after ``fps``), i.e. from the transition-expanded
                     render start = ``EffectSpec.frame_origin`` (``ctx.frame`` here)
    [inset] crop=w=0.24*iw:h=0.46*ih:x=0.39*iw:y=0.18*ih                         rgba (lrint sizes)
            drawbox=x=0:y=0:w=iw:h=ih:c=white@0.65:t=1                            rgba (1-px border,
                     float32 ``(1-a)*v + a*255`` truncated, alpha kept; a = 165/255, ``white@0.65``
                     parses to alpha 165 by truncation)
            scale=w=iw*(0.68+0.68*clip(t/0.8,0,1)):h=ih*(0.775+0.775*clip(t/0.8,0,1)):eval=frame
                     -> ONE swscale pass rgba -> yuva444p at (int)w x (int)h: bicubic (B=0, C=0.6)
                        integer filters (``initFilter``, 14-bit horizontal / 12-bit vertical taps,
                        NEON/x86 filter alignment 4/2), RGB->YUV 14-bit input converters,
                        ``hScale16To15`` (>>13), ``lumRangeToJpeg`` on ``pc`` links,
                        ``yuv2planeX_8`` (>>19, flat dither 64); alpha ``rgbaToA_c`` through the luma
                        filters -- ``sws_scale_rgba8_to_yuva444p8`` below, bit-exact vs the CLI
            fade=t=in:st=0:d=0.3:alpha=1                                          yuva444p alpha plane:
                     ``a' = (a*factor + 32768) >> 16``, factor = pts*65535/duration_pts (integer),
                     duration_pts = av_rescale_q(0.3 s, 1/1e6, tb) (9 at 30 fps)
    [builtin][window] overlay=x=W*(0.42+0.18*s(t/0.8)):y=0.23*H:format=auto     yuva444p (main has
                     alpha): ``blend_plane_8_8bits`` straight path per plane with the
                     ``UNPREMULTIPLY_ALPHA`` correction where the main alpha is partial, then
                     ``alpha_composite_8_8bits``; x / y are ``(int)`` truncated per frame
    [out] -> the spatial tail / compositor: yuva444p -> 8-bit RGB (``yuva444p8_to_rgba8``)

Time: ``T`` / ``t`` = ``frame * (tb.num / tb.den)`` in double exactly like ``av_q2d``; the link time
base after ``fps=<project rate>`` is the project frame duration.

Precision (goldens in ``experimental_tests/core/test_tensor_fx_branched.py``, arm64 ffmpeg 8.0.1):
every integer stage (swscale scaled conversion incl. the chroma pixel-pair path, drawbox, fade,
overlay, crop, the E4 bridge in/out) is bit-exact against the ``ffmpeg`` CLI; ``gblur`` is the
shared float32 IIR (<= 1 code by contract, 4 values in 576k off by one measured); the ``blend``
expression is evaluated in float64 on CPU (bit-exact) and float32 on MPS (<= 1 code: the
reference's own double evaluation lands e.g. ``100*(1-s)+100*s = 99.999...`` -> 99 on some frames,
which float32 cannot always mirror).  Whole graph, 36 frames x 3 links: CPU float64 99.9996-99.9998 %
exact, max 2; MPS 99.71-99.76 % exact, 99.9945-99.9999 % within 2, max 4 (one value).  Reference
departures inherited from E4's ledger rows: with a spatial tail the reference's exit link is 16-bit
``rgba64le`` while this port exits at 8 bits (<= 1 code, row 16); the yuva444p bridge's alpha +1
quirk applies (row 13).  x86 references differ by +-1 in the ``yuv2planeX`` vertical scaler (the
SSE path without ``accurate_rnd`` uses ``pmulhw``); the arm64 NEON paths equal the C arithmetic.

Main callers:
- ``tensor/effects.py`` imports this module for its ``register`` side effect; the registry
  dispatches ``lower`` / ``apply``.
- ``test_tensor_fx_branched.py`` calls the public stage kernels directly for the per-filter goldens.

Why this exists:
The E1/E2/E4 ports are one-filter chains; Callout is the first ported effect whose reference is a
five-branch complex graph with a *time-varying* per-frame ``scale`` (libswscale's bicubic scaler
inside the effect) -- the swscale port here (``sws_bicubic_filter`` / ``sws_scale_rgba8_to_yuva444p8``)
is the reusable piece other branched or scaled effects will need.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Final

import torch

from ..core.model import ResolvedEffect
from .color import code_to_premultiplied, premultiplied_to_code
from .effects import OVERSCAN_CROP, ApplyContext, EffectPort, LowerContext, register
from .fx_color import (
    BridgeLink,
    eq_process_lut,
    rgb2yuv_table,
    rgba8_to_yuva444p8,
    yuva444p8_to_rgba8,
)
from .support import reject
from .swscale_fixedpoint import c_div as _c_div, finalize_swscale_filter
from .tr_equirect import gblur_gbrap

CALLOUT_HANDLER: Final = "cohort_callout"

# --------------------------------------------------------------------------- the graph's constants
# (``cohort_effects._callout_graph_lines``: default-only, no XML control reaches the graph)
FIELD_SIGMA: Final = 4.0          # gblur=sigma=4 (AVOption float; 4.0 is exact in float32)
FIELD_STEPS: Final = 2            # gblur steps=2
FIELD_BRIGHTNESS: Final = -0.28   # eq=brightness=-0.28
FIELD_SATURATION: Final = 0.6     # eq=saturation=0.6
BLEND_SPAN: Final = 0.6           # T/0.6 in the blend expression
CROP_W, CROP_H, CROP_X, CROP_Y = 0.24, 0.46, 0.39, 0.18
BOX_ALPHA_CODE: Final = 165       # white@0.65 -> av_parse_color: rgba[3] = (uint8_t)(0.65 * 255) = 165 (truncated)
SCALE_W0, SCALE_W1 = 0.68, 0.68   # w = iw*(0.68 + 0.68*clip(t/0.8,0,1))
SCALE_H0, SCALE_H1 = 0.775, 0.775
MOTION_SPAN: Final = 0.8          # t/0.8 in scale / overlay
FADE_DURATION_US: Final = 300000  # fade d=0.3 (AV_OPT_TYPE_DURATION, microseconds)
OVERLAY_X0, OVERLAY_X1, OVERLAY_Y = 0.42, 0.18, 0.23

# swscale (libswscale/utils.c ``initFilter``) constants for the default ``scale`` filter:
# ``sws_flags`` = SWS_BICUBIC (vf_scale passes no flags), ``scaler_params`` = SWS_PARAM_DEFAULT
# -> B = 0, C = 0.6; ``SWS_MAX_REDUCE_CUTOFF`` 0.002; NEON / x86 filter alignment 4 (horizontal)
# and 2 (vertical, dropped to 1 for the unscaled special case).
_SWS_BICUBIC_B: Final = int(0.0 * (1 << 24))
_SWS_BICUBIC_C: Final = int(0.6 * (1 << 24))
_SWS_H_ALIGN: Final = 4
_SWS_V_ALIGN: Final = 2
_SWS_ONE_H: Final = 1 << 14      # horizontal taps are 14-bit
_SWS_ONE_V: Final = 1 << 12      # vertical taps are 12-bit
_SWS_POS: Final = 128            # get_local_pos(c, 0, 0|-513, dir) for luma / chroma / alpha of a 4:4:4 pair
# swscale ``ff_sws_init_range_convert`` -> ``lumRangeToJpeg_c`` / ``chrRangeToJpeg_c`` (coeff, offset)
# = solve_range_convert(16, 235 | 240, 0, 255) as n8.0.1 rounds it (fx_color module doc).
_LUM_TO_JPEG: Final = (19078, -39084288)
_CHR_TO_JPEG: Final = (18652, -38207488)


# --------------------------------------------------------------------------- payload


@dataclass(frozen=True)
class CalloutPayload:
    """One lowered Callout: the negotiated YUV link, the layer time base and the ``eq`` LUTs."""

    link: str          # ``rgba:<bt601|bt709>:<tv|pc>`` (pixel format is always rgba: the graph's own format=rgba)
    tb_num: int        # link time base after ``fps`` = the project frame duration (T = frame * tb)
    tb_den: int
    y_lut: tuple[int, ...]   # eq luma: process_c(contrast 1, brightness -0.28)
    c_lut: tuple[int, ...]   # eq chroma: process_c(contrast = saturation 0.6, brightness 0)

    @property
    def bridge(self) -> BridgeLink:
        return BridgeLink.parse(self.link)

    @property
    def tb_seconds(self) -> float:
        """``av_q2d(time_base)``: num / (double) den."""

        return self.tb_num / self.tb_den


# --------------------------------------------------------------------------- C arithmetic helpers


def _av_log2(value: int) -> int:
    """``av_log2``: floor(log2(value)); 0 for 0 (the C table lookup)."""

    return value.bit_length() - 1 if value > 0 else 0


def _floor_shift(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Arithmetic ``>>`` on int32 tensors (floor division; MPS-safe)."""

    return torch.div(values, 1 << bits, rounding_mode="floor")


def _fast_div255(values: torch.Tensor) -> torch.Tensor:
    """``FAST_DIV255(x) = ((x + 128) * 257) >> 16`` (vf_overlay.c) on non-negative int32."""

    return _floor_shift((values + 128) * 257, 16)


def _unpremultiply_alpha(alpha: torch.Tensor, alpha_d: torch.Tensor) -> torch.Tensor:
    """``UNPREMULTIPLY_ALPHA(x, y)`` (vf_overlay.c), C integer division on positive operands.

    ``((x<<16) - (x<<9) + x) / (((x+y)<<8) - (x+y) - y*x)``: the overlay alpha divided by the
    straight-alpha coverage the two layers reach together.
    """

    numerator = (alpha * 65536) - (alpha * 512) + alpha
    total = alpha + alpha_d
    denominator = (total * 256) - total - alpha_d * alpha
    # The C code only evaluates this for 0 < alpha < 255 (denominator > 0); callers select those
    # pixels with ``torch.where``, which still evaluates every element -- keep the rest finite.
    denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    return torch.div(numerator, denominator, rounding_mode="floor")


def av_expr_smoothstep(seconds: float, span: float) -> float:
    """``3*pow(clip(t/span,0,1),2)-2*pow(clip(t/span,0,1),3)`` evaluated exactly like ``av_expr_eval``
    (double arithmetic in the written order, libm ``pow``)."""

    clipped = seconds / span
    clipped = 0.0 if clipped < 0.0 else (1.0 if clipped > 1.0 else clipped)
    return 3.0 * math.pow(clipped, 2.0) - 2.0 * math.pow(clipped, 3.0)


def link_seconds(frame: int, payload: CalloutPayload) -> float:
    """``T`` / ``t`` for the layer-local frame: ``pts * av_q2d(time_base)`` with pts = frame."""

    return float(frame) * payload.tb_seconds


# --------------------------------------------------------------------------- lower


def _callout_link(effect: ResolvedEffect, ctx: LowerContext) -> BridgeLink:
    """The yuva444p link the reference negotiates inside the graph: colourspace / range from the
    plan (``fx_color.reference_effect_link``: source tags, ``pc`` after a resampling ``perspective``
    stage), pixel format ``rgba`` because the graph starts with its own ``format=rgba``."""

    text = ctx.reference_effect_link
    if text is None:
        # Same contract as E4's eq bridge (``fx_color._parse_stage``): the plan must say which
        # link the reference negotiates for this layer; never guess one from the source tags.
        raise reject(
            "effect (unsupported parameters)",
            f"{effect.path}: the plan did not resolve which pixel link the reference feeds this "
            "Callout graph (LowerContext.reference_effect_link is unset; see fx_color.reference_effect_link)",
        )
    try:
        negotiated = BridgeLink.parse(text)
    except ValueError as error:
        raise reject(
            "effect (unsupported parameters)",
            f"{effect.path}: {effect.name or effect.handler} on a {text} link: {error}",
        ) from error
    return BridgeLink("rgba", negotiated.colorspace, negotiated.color_range)


# ``eq_process_lut`` lives in fx_color (E4) -- shared, float32-rounded like vf_eq.c.
def _lower_callout(effect: ResolvedEffect, ctx: LowerContext) -> CalloutPayload:
    if effect.params:
        names = sorted({parameter.name or parameter.key or "?" for parameter in effect.params})
        raise reject(
            "effect (unsupported parameters)",
            f"{effect.path}: {effect.name or effect.handler} is a default-only graph but received controls {names}",
        )
    link = _callout_link(effect, ctx)
    frame_duration = Fraction(ctx.frame_duration)
    return CalloutPayload(
        link=str(link),
        tb_num=frame_duration.numerator,
        tb_den=frame_duration.denominator,
        y_lut=eq_process_lut(1.0, FIELD_BRIGHTNESS),
        c_lut=eq_process_lut(FIELD_SATURATION, 0.0),
    )


# --------------------------------------------------------------------------- swscale: bicubic filters
#
# ``initFilter`` (libswscale/utils.c) transcribed for the SWS_BICUBIC default (no src/dst
# convolution filters, no SWS_BITEXACT / ACCURATE_RND): 64-bit fixed point coefficients, the
# reduce / align / border-fix passes and the error-diffused normalisation to ``one``.  Every C
# division that can see a negative operand uses C truncation (``_c_div``).


@lru_cache(maxsize=1024)
def sws_bicubic_filter(
    inc: int, src_len: int, dst_len: int, filter_align: int, one: int, src_pos: int = _SWS_POS, dst_pos: int = _SWS_POS,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """One axis of swscale's default (bicubic) scaler: ``(filterPos, filter)`` for ``dst_len``
    outputs -- ``filter[i]`` are the ``filterSize`` integer taps (sum ``one``, +-1 error diffusion)
    applied at source indices ``filterPos[i] + j``.

    ``inc`` is the 16.16 step (``lumXInc`` = ``((src<<16) + (dst>>1)) / dst``), ``filter_align`` the
    host's SIMD alignment (4 horizontal / 2 vertical on NEON and x86), ``one`` the tap scale
    (``1<<14`` horizontal, ``1<<12`` vertical), positions the ``get_local_pos`` centre offsets
    (128 for every plane of an rgba -> 4:4:4 pair).
    """

    fone = 1 << (54 - min(_av_log2(src_len // dst_len), 8))
    positions: list[int] = []
    filt: list[list[int]] = []
    if abs(inc - 0x10000) < 10 and src_pos == dst_pos:      # unscaled: one tap
        filter_size = 1
        positions = list(range(dst_len))
        filt = [[fone] for _ in range(dst_len)]
    else:
        size_factor = 4                                       # scale_algorithms[SWS_BICUBIC]
        filter_size = 1 + size_factor if inc <= (1 << 16) else 1 + (size_factor * src_len + dst_len - 1) // dst_len
        filter_size = max(min(filter_size, src_len - 2), 1)
        x_dst_in_src = ((dst_pos * inc) >> 7) - ((src_pos * 0x10000) >> 7)
        b_coef, c_coef = _SWS_BICUBIC_B, _SWS_BICUBIC_C
        for _ in range(dst_len):
            xx = _c_div(x_dst_in_src - (filter_size - 2) * (1 << 16), 1 << 17)
            positions.append(xx)
            row: list[int] = []
            for _tap in range(filter_size):
                d = abs(xx * (1 << 17) - x_dst_in_src) << 13
                if inc > (1 << 16):
                    d = d * dst_len // src_len
                if d >= (1 << 31):
                    coeff = 0
                else:
                    dd = (d * d) >> 30
                    ddd = (dd * d) >> 30
                    if d < (1 << 30):
                        coeff = ((12 * (1 << 24) - 9 * b_coef - 6 * c_coef) * ddd
                                 + (-18 * (1 << 24) + 12 * b_coef + 6 * c_coef) * dd
                                 + (6 * (1 << 24) - 2 * b_coef) * (1 << 30))
                    else:
                        coeff = ((-b_coef - 6 * c_coef) * ddd
                                 + (6 * b_coef + 30 * c_coef) * dd
                                 + (-12 * b_coef - 48 * c_coef) * d
                                 + (8 * b_coef + 24 * c_coef) * (1 << 30))
                    coeff = _c_div(coeff, (1 << 54) // fone)
                row.append(coeff)
                xx += 1
            filt.append(row)
            x_dst_in_src += 2 * inc

    # No srcFilter / dstFilter (filter2 == filter): run the shared ``initFilter``
    # tail on the raw bicubic taps -- reduce near-zero heads/tails, align to the
    # host SIMD width, reform, fold out-of-raster support, normalise to ``one``.
    # The bicubic ``scale`` can be 1:1 on an axis, so it keeps swscale's size-1
    # vertical align quirk (``apply_align_quirk=True``) and its faithful
    # truncate-or-pad reform (``reform_truncates=True``).
    return finalize_swscale_filter(
        positions,
        filt,
        filter_size,
        fone,
        filter_align=filter_align,
        one=one,
        apply_align_quirk=True,
        reform_truncates=True,
        src_len=src_len,
    )


def _filter_tensors(
    positions: tuple[int, ...], taps: tuple[tuple[int, ...], ...], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(positions, dtype=torch.long, device=device),
        torch.tensor(taps, dtype=torch.int32, device=device),
    )


def _hscale16to15(plane14: torch.Tensor, positions: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """``hScale16To15_c`` for an RGB-derived 14-bit plane ``[H, W]`` -> ``[H, dstW]`` int32:
    ``min(sum_j src[pos+j]*filter[j] >> 13, 32767)`` (RGB sources shift by 13)."""

    value = torch.zeros((plane14.shape[0], positions.shape[0]), dtype=torch.int32, device=plane14.device)
    for j in range(taps.shape[1]):
        value = value + torch.index_select(plane14, 1, positions + j) * taps[:, j].view(1, -1)
    return torch.clamp(_floor_shift(value, 13), max=32767)


def _range_to_jpeg(plane15: torch.Tensor, coeff_offset: tuple[int, int]) -> torch.Tensor:
    """``lumRangeToJpeg_c`` / ``chrRangeToJpeg_c`` on the 15-bit horizontally scaled line."""

    coeff, offset = coeff_offset
    return torch.clamp(_floor_shift(plane15 * coeff + offset, 14), max=32767)


def _vscale8(plane15: torch.Tensor, positions: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """``yuv2planeX_8_c`` (flat dither 64: ``(64<<12 + sum_j src[pos+j]*filter[j]) >> 19``, clip) or
    ``yuv2plane1_8_c`` (``(src + 64) >> 7``) when the vertical filter has one tap; ``[H, W]`` -> ``[dstH, W]``."""

    if taps.shape[1] == 1:
        rows = torch.index_select(plane15, 0, positions)
        return torch.clamp(_floor_shift(rows + 64, 7), 0, 255)
    value = torch.full((positions.shape[0], plane15.shape[1]), 64 << 12, dtype=torch.int32, device=plane15.device)
    for j in range(taps.shape[1]):
        value = value + torch.index_select(plane15, 0, positions + j) * taps[:, j].view(-1, 1)
    return torch.clamp(_floor_shift(value, 19), 0, 255)


def sws_scale_rgba8_to_yuva444p8(rgba8: torch.Tensor, dst_w: int, dst_h: int, link: BridgeLink) -> torch.Tensor:
    """libswscale ``scale`` of a straight 8-bit RGBA frame to ``yuva444p`` at ``dst_w x dst_h`` with
    the default flags (bicubic): int32 ``[4, H, W]`` in (RGBA plane order), int32 ``[4, dst_h, dst_w]``
    out (YUVA).  ``link.colorspace`` selects the RGB->YUV table, ``link.color_range`` whether the
    limited-range 15-bit lines are expanded to full range (``pc`` links).

    Pipeline per plane (swscale.c ``hyscale`` / ``hcscale`` / ``lum_planar_vscale``): rgb32ToY/UV
    14-bit input converters, ``hScale16To15`` with the bicubic taps, ``lumRangeToJpeg`` /
    ``chrRangeToJpeg`` on ``pc`` links, ``yuv2planeX_8`` (or ``yuv2plane1_8``); alpha uses
    ``rgbaToA_c`` (``a<<6 | a>>2``) through the luma filters and no range conversion; chroma of an
    even-width source downscaled to <= half its width comes from pixel pairs (``rgb32ToUV_half_c``,
    ``chrSrcHSubSample = 1``) with a half-width filter.  Bit-exact against ffmpeg 8.0.1 (arm64) for
    up- and down-scales on both matrices and ranges.  At 1:1 it reduces to
    ``fx_color.rgba8_to_yuva444p8`` (verified in the tests).
    """

    if rgba8.dim() != 3 or rgba8.shape[0] != 4:
        raise ValueError(f"sws_scale_rgba8_to_yuva444p8 expects [4, H, W], got {tuple(rgba8.shape)}")
    _, src_h, src_w = rgba8.shape
    if src_w < 3 or src_h < 3 or dst_w < 1 or dst_h < 1:
        raise ValueError(f"sws_scale_rgba8_to_yuva444p8: degenerate scale {src_w}x{src_h} -> {dst_w}x{dst_h}")
    device = rgba8.device
    rgba8 = rgba8.to(torch.int32)
    r, g, b, a = rgba8[0], rgba8[1], rgba8[2], rgba8[3]
    ry, gy, by, ru, gu, bu, rv, gv, bv = rgb2yuv_table(link.colorspace)
    y14 = _floor_shift(ry * r + gy * g + by * b + (32 << 14) + (1 << 8), 9)
    a14 = a * 64 + _floor_shift(a, 2)          # a<<6 | a>>2 (disjoint bits)
    # "drop every other pixel for chroma calculation" (utils.c): an even-width RGB source scaled
    # to at most half its width converts chroma from PIXEL PAIRS (``rgb32ToUV_half_c``: sums of the
    # pair, offset (256<<15) + (1<<9), >> 10) on a half-width chroma line with its own filter.
    chroma_half = src_w % 2 == 0 and dst_w <= (src_w >> 1)
    if chroma_half:
        r2, g2, b2 = (r[:, 0::2] + r[:, 1::2]), (g[:, 0::2] + g[:, 1::2]), (b[:, 0::2] + b[:, 1::2])
        u14 = _floor_shift(ru * r2 + gu * g2 + bu * b2 + (256 << 15) + (1 << 9), 10)
        v14 = _floor_shift(rv * r2 + gv * g2 + bv * b2 + (256 << 15) + (1 << 9), 10)
        chr_src_w = src_w >> 1
    else:
        u14 = _floor_shift(ru * r + gu * g + bu * b + (256 << 14) + (1 << 8), 9)
        v14 = _floor_shift(rv * r + gv * g + bv * b + (256 << 14) + (1 << 8), 9)
        chr_src_w = src_w
    x_inc = ((src_w << 16) + (dst_w >> 1)) // dst_w
    chr_x_inc = ((chr_src_w << 16) + (dst_w >> 1)) // dst_w
    y_inc = ((src_h << 16) + (dst_h >> 1)) // dst_h
    h_pos, h_taps = _filter_tensors(*sws_bicubic_filter(x_inc, src_w, dst_w, _SWS_H_ALIGN, _SWS_ONE_H), device)
    c_pos, c_taps = _filter_tensors(*sws_bicubic_filter(chr_x_inc, chr_src_w, dst_w, _SWS_H_ALIGN, _SWS_ONE_H), device)
    v_pos, v_taps = _filter_tensors(*sws_bicubic_filter(y_inc, src_h, dst_h, _SWS_V_ALIGN, _SWS_ONE_V), device)

    def plane(p14: torch.Tensor, to_jpeg: tuple[int, int] | None, pos: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
        p15 = _hscale16to15(p14, pos, taps)
        if to_jpeg is not None and link.full_range:
            p15 = _range_to_jpeg(p15, to_jpeg)
        return _vscale8(p15, v_pos, v_taps)

    return torch.stack((
        plane(y14, _LUM_TO_JPEG, h_pos, h_taps),
        plane(u14, _CHR_TO_JPEG, c_pos, c_taps),
        plane(v14, _CHR_TO_JPEG, c_pos, c_taps),
        plane(a14, None, h_pos, h_taps),
    ))


# --------------------------------------------------------------------------- stage kernels


@lru_cache(maxsize=64)
def _lut_tensor(lut: tuple[int, ...], device: str) -> torch.Tensor:
    return torch.tensor(lut, dtype=torch.int32, device=device)


def callout_field(yuva: torch.Tensor, payload: CalloutPayload) -> torch.Tensor:
    """``gblur=sigma=4:steps=2:planes=0x7,eq=brightness=-0.28:saturation=0.6`` on an int32 yuva444p
    frame ``[4, H, W]``: the float32 IIR Gaussian on Y/U/V (alpha untouched), then the ``eq``
    ``process_c`` LUTs (luma brightness, chroma contrast = saturation), alpha copied."""

    blurred = gblur_gbrap(yuva[:3], sigma=FIELD_SIGMA, sigma_v=FIELD_SIGMA, steps=FIELD_STEPS).to(torch.int32)
    device = str(yuva.device)
    y_lut = _lut_tensor(payload.y_lut, device)
    c_lut = _lut_tensor(payload.c_lut, device)
    return torch.stack((y_lut[blurred[0].long()], c_lut[blurred[1].long()], c_lut[blurred[2].long()], yuva[3]))


def callout_blend(top: torch.Tensor, bottom: torch.Tensor, seconds: float) -> torch.Tensor:
    """``blend=all_expr='A*(1-s)+B*s'`` with ``s = smoothstep(clip(T/0.6, 0, 1))`` per plane
    (``blend_expr_8bit``: double evaluation, ``dst[x] = (uint8_t) value`` = truncation).
    ``top`` = the first input (``A``, the original), ``bottom`` = the second (``B``, the field).
    Float64 on devices that have it (bit-exact), float32 on MPS (<= 1 code)."""

    weight = av_expr_smoothstep(seconds, BLEND_SPAN)
    dtype = torch.float32 if top.device.type == "mps" else torch.float64
    a = top.to(dtype)
    b = bottom.to(dtype)
    value = a * (1.0 - weight) + b * weight
    return torch.trunc(value).to(torch.int32)


def crop_window(width: int, height: int) -> tuple[int, int, int, int]:
    """``crop=w=0.24*iw:h=0.46*ih:x=0.39*iw:y=0.18*ih`` on an rgba frame -> ``(x, y, w, h)``:
    ``vf_crop.c`` ``normalize_double`` = ``lrint`` (half to even), then the position clamps
    (``x + w > iw -> x = iw - w``); no chroma alignment for rgba."""

    w = int(round(CROP_W * width))
    h = int(round(CROP_H * height))
    x = int(round(CROP_X * width))
    y = int(round(CROP_Y * height))
    x = max(x, 0)
    y = max(y, 0)
    if x + w > width:
        x = width - w
    if y + h > height:
        y = height - h
    return x, y, w, h


def drawbox_white_border(rgba8: torch.Tensor) -> torch.Tensor:
    """``drawbox=x=0:y=0:w=iw:h=ih:c=white@0.65:t=1`` on a packed-rgba int32 frame ``[4, h, w]``:
    ``draw_region_rgb_packed`` (no ``replace``): the one-pixel border gets
    ``(uint8_t)((1.f - alpha) * v + alpha * 255.f)`` in float32 with ``alpha = 165 / 255.f`` on
    R, G, B; alpha untouched."""

    _, h, w = rgba8.shape
    alpha = torch.tensor(BOX_ALPHA_CODE, dtype=torch.float32) / torch.tensor(255.0, dtype=torch.float32)
    one_minus = torch.tensor(1.0, dtype=torch.float32) - alpha
    white = alpha * torch.tensor(255.0, dtype=torch.float32)
    device = rgba8.device
    rgb = rgba8[:3].to(torch.float32)
    mixed = one_minus.to(device) * rgb + white.to(device)
    mixed = torch.trunc(mixed).clamp(0.0, 255.0).to(torch.int32)
    ys = torch.arange(h, device=device).view(-1, 1)
    xs = torch.arange(w, device=device).view(1, -1)
    border = (xs < 1) | (w - 1 - xs < 1) | (ys < 1) | (h - 1 - ys < 1)
    out = rgba8.clone()
    out[:3] = torch.where(border.unsqueeze(0), mixed, rgba8[:3])
    return out


def scale_window_size(crop_w: int, crop_h: int, seconds: float) -> tuple[int, int]:
    """``scale=w=iw*(0.68+0.68*clip(t/0.8,0,1)):h=ih*(0.775+0.775*clip(t/0.8,0,1)):eval=frame`` ->
    ``(int)`` truncated sizes (``ff_scale_eval_dimensions``: a 0 result means the input size)."""

    clipped = seconds / MOTION_SPAN
    clipped = 0.0 if clipped < 0.0 else (1.0 if clipped > 1.0 else clipped)
    w = int(crop_w * (SCALE_W0 + SCALE_W1 * clipped))
    h = int(crop_h * (SCALE_H0 + SCALE_H1 * clipped))
    return (w or crop_w), (h or crop_h)


def fade_duration_pts(payload: CalloutPayload) -> int:
    """``fade`` ``duration_pts = av_rescale_q(300000 us, 1/1e6, time_base)`` (AV_ROUND_NEAR_INF)."""

    b = payload.tb_den
    c = payload.tb_num * 1000000
    return (FADE_DURATION_US * b + c // 2) // c


def fade_in_factor(frame_pts: int, duration_pts: int) -> int:
    """``vf_fade.c`` ``s->factor`` for the frame with ``pts = frame_pts`` (``st=0``, fading by
    duration): ``pts * 65535 / duration_pts`` in integer arithmetic, 65535 once ``pts > duration_pts``."""

    if duration_pts <= 0:
        raise ValueError(f"fade duration of {duration_pts} pts: frame-count fading is not modelled")
    if frame_pts > duration_pts:
        return 65535
    return min(65535, (frame_pts * 65535) // duration_pts)


def fade_in_alpha(yuva: torch.Tensor, factor: int) -> torch.Tensor:
    """``fade=t=in:alpha=1`` on an int32 yuva444p frame: ``a' = (a * factor + 32768) >> 16`` when
    ``factor < 65535`` (``filter_slice_alpha``, black level 0), untouched otherwise."""

    if factor >= 65535:
        return yuva
    alpha = _floor_shift(yuva[3] * factor + 32768, 16)
    return torch.cat((yuva[:3], alpha.unsqueeze(0)))


def overlay_position(width: int, height: int, seconds: float) -> tuple[int, int]:
    """``overlay=x='W*(0.42+0.18*s(t/0.8))':y='0.23*H'`` -> ``normalize_xy``: ``(int)`` truncation."""

    x = width * (OVERLAY_X0 + OVERLAY_X1 * av_expr_smoothstep(seconds, MOTION_SPAN))
    y = OVERLAY_Y * height
    return int(x), int(y)


def overlay_yuva444p(main: torch.Tensor, over: torch.Tensor, x: int, y: int) -> torch.Tensor:
    """``overlay=...:format=auto`` with both inputs yuva444p (``blend_slice_yuva444``,
    ``main_has_alpha``, straight alpha) on int32 ``[4, H, W]`` / ``[4, h, w]`` frames at integer
    ``(x, y)`` (negative / overhanging positions are clipped like the C ``jmin/jmax/kmin/kmax``).

    Per colour plane (``blend_plane_8_8bits``): ``alpha`` = overlay alpha, replaced by
    ``UNPREMULTIPLY_ALPHA(alpha, main_alpha)`` when ``0 < alpha < 255``; ``d = FAST_DIV255(d*(255-alpha)
    + s*alpha)``.  Alpha (``alpha_composite_8_8bits``): the same corrected alpha decides ``d = s``
    (== 255) or ``d += FAST_DIV255((255 - d) * s)`` (> 0), where ``s`` is the raw overlay alpha.
    """

    _, dst_h, dst_w = main.shape
    _, src_h, src_w = over.shape
    j_min, j_max = max(-y, 0), min(-y + dst_h, src_h, dst_h, y + src_h)
    k_min, k_max = max(-x, 0), min(-x + dst_w, src_w)
    if j_max <= j_min or k_max <= k_min:
        return main
    region = main[:, y + j_min:y + j_max, x + k_min:x + k_max]
    source = over[:, j_min:j_max, k_min:k_max]
    s_alpha = source[3]
    d_alpha = region[3]
    partial = (s_alpha != 0) & (s_alpha != 255)
    alpha = torch.where(partial, _unpremultiply_alpha(s_alpha, d_alpha), s_alpha)
    colour = _fast_div255(region[:3] * (255 - alpha).unsqueeze(0) + source[:3] * alpha.unsqueeze(0))
    composed = d_alpha + _fast_div255((255 - d_alpha) * s_alpha)
    new_alpha = torch.where(alpha == 255, s_alpha, torch.where(alpha > 0, composed, d_alpha))
    out = main.clone()
    out[:3, y + j_min:y + j_max, x + k_min:x + k_max] = colour
    out[3, y + j_min:y + j_max, x + k_min:x + k_max] = new_alpha
    return out


def callout_window(rgba8: torch.Tensor, payload: CalloutPayload, frame: int) -> torch.Tensor:
    """The inset branch for the layer-local ``frame``: crop, drawbox border, per-frame swscale to
    yuva444p, alpha fade-in.  Returns int32 yuva444p ``[4, h_t, w_t]``."""

    _, height, width = rgba8.shape
    x, y, w, h = crop_window(width, height)
    inset = drawbox_white_border(rgba8[:, y:y + h, x:x + w])
    seconds = link_seconds(frame, payload)
    w_t, h_t = scale_window_size(w, h, seconds)
    window = sws_scale_rgba8_to_yuva444p8(inset, w_t, h_t, payload.bridge)
    return fade_in_alpha(window, fade_in_factor(frame, fade_duration_pts(payload)))


def callout_graph_yuva(rgba8: torch.Tensor, payload: CalloutPayload, frame: int) -> torch.Tensor:
    """The whole reference graph on the straight int32 RGBA code frame for the layer-local ``frame``,
    up to (and excluding) the exit conversion: int32 yuva444p ``[4, H, W]``."""

    link = payload.bridge
    original = rgba8_to_yuva444p8(rgba8, link)
    field = callout_field(original, payload)
    seconds = link_seconds(frame, payload)
    built_in = callout_blend(original, field, seconds)
    window = callout_window(rgba8, payload, frame)
    x, y = overlay_position(rgba8.shape[2], rgba8.shape[1], seconds)
    return overlay_yuva444p(built_in, window, x, y)


def callout_graph_rgba8(rgba8: torch.Tensor, payload: CalloutPayload, frame: int) -> torch.Tensor:
    """The whole graph including the exit ``yuva444p -> 8-bit RGB`` (the compositor / spatial-tail
    link): int32 straight RGBA ``[4, H, W]``."""

    return yuva444p8_to_rgba8(callout_graph_yuva(rgba8, payload, frame), payload.bridge)


def _apply_callout(payload: CalloutPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    code = premultiplied_to_code(canvas)
    rgba8 = torch.clamp(torch.round(code), 0.0, 255.0).to(torch.int32)
    out = callout_graph_rgba8(rgba8, payload, ctx.frame)
    return code_to_premultiplied(out.to(canvas.dtype))


# ``"crop"``: the Callout graph is a canvas-relative COMPOSITION (a crop window sized and
# placed by ``lrint`` of the canvas dimensions, rescaled and overlaid at a canvas-normalized
# position over the blurred field).  Run on an overscan surface it would cut and overlay a
# second inset relative to the surface, which a pan would reveal; it cannot be re-expressed
# in canvas coordinates without re-deriving the whole calibrated graph.  So only the canvas
# crop is processed and the overscan keeps its input pixels (a pan past the canvas shows the
# un-called-out image there rather than black).
register(EffectPort(handler=CALLOUT_HANDLER, lower=_lower_callout, apply=_apply_callout, overscan=OVERSCAN_CROP))
