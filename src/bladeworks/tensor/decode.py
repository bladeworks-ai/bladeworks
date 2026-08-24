"""PyAV clip decoding with exact source-frame ownership and the colour-in policy (A3 / X10).

Architecture map
----------------
    probe_video(path)                 : ffprobe facts (raster, rotation, SAR, duration)
                                        + the stream colour tags (pix_fmt, range, matrix,
                                        transfer, primaries, bit depth) -> ``VideoProbe``
    check_source_color(probe)         : plan-time verdict on those tags (loud reject)
    ClipDecoder.frame_at(t)           : decoded frame owning ``t`` -> RGB [3,H,W] float 0..1
                                        (device) = packed_at(t) uploaded + planes_to_rgb
    ClipDecoder.packed_at(t)          : the same frame as raw planes on the CPU
                                        (``SourceFrame``: one flat tensor + layout + colour);
                                        with ``decode_size`` (preview only) the planes
                                        are libswscale-downscaled first, same pixel format
    planes_to_rgb(planes, layout, c)  : the ONE yuv -> RGB kernel (device), per source class:
                                          8-bit  : exact float BT.709/601/2020 matrix, nearest
                                                   chroma, round to 8-bit codes
                                          10-bit : swscale's generic-path integer arithmetic
                                                   (bicubic vertical chroma, C lookup tables)
                                          4:4:4  : swscale's full-chroma 16-bit fixed-point writer
    RasterSource                      : stills / titles / solids decoded once (straight alpha)

Colour-in policy (X10) -- what the CPU reference does and what this module mirrors
---------------------------------------------------------------------------------
The reference feeds every source through ``format=rgba`` (``ffmpeg._video_chain``),
i.e. libswscale (FFmpeg n8.0) with the *frame's* colour tags:

* matrix from ``colorspace``: bt709 -> BT.709; bt470bg / smpte170m -> BT.601;
  bt2020nc -> BT.2020 non-constant; **unspecified -> BT.601** (swscale's default,
  regardless of resolution); range from ``color_range`` (unspecified -> limited;
  ``yuvj*`` formats are full range by definition).  Verified empirically against
  ``ffmpeg -vf format=rgba`` (see ``test_tensor_color_io.py``); the tensor renderer
  honours exactly the same tags and defaults.
* 8-bit yuv420p / yuv422p at even sizes: swscale's *unscaled* converters
  (NEON on aarch64 for width % 16 == 0, SSSE3 on x86): nearest chroma
  (each chroma sample covers its 2x2 / 2x1 luma block), 16-bit fixed point that
  rounds like exact float math (max 1 code apart).  This module uses the exact
  float matrix and rounds to 8-bit codes.  On aarch64 a width that is not a
  multiple of 16 (e.g. 1080-wide vertical video) drops swscale into its C
  lookup-table converter, ~1.7 codes darker (white 235 -> 253); the tensor
  renderer keeps the exact math there (ledger candidate; SSIM-invisible).
* 10-bit (yuv420p10le HEVC, yuv422p10le ProRes 422): swscale's *generic* path,
  which is C on every architecture: luma and chroma are reduced to 8 bits
  ((x10 + 2) >> 2 for identity taps), 4:2:0 chroma is upsampled vertically with
  the default bicubic (B=0, C=0.6) 4-tap filter quantised to int16 (sum 4096),
  horizontal chroma is shared per luma pair, and RGB comes out of the yuv2rgb
  lookup tables (``ff_yuv2rgb_c_init_tables``: chroma folded into an integer
  luma-table offset *before* the range scaling, luma table anchored at 326 not
  384 so limited-range white 940 lands on 253).  ``planes_to_rgb`` reproduces
  that integer arithmetic exactly (bit-exact vs the CLI on the fixtures).
* HLG / PQ transfer: accepted only with Rec.2020 matrix and primaries, then
  mapped through the renderer-owned Rec.2020-to-Rec.709 SDR LUT on the render
  device. Matrices other than the three above (fcc, smpte240m, ycgco,
  bt2020c, rgb, chroma-derived, ictcp), pixel formats outside
  {yuv420p, yuvj420p, yuv422p, yuvj422p, yuv444p, yuvj444p, yuv420p10le,
  yuv422p10le, yuv444p10le, yuv444p12le} (nv12 / p010 hwaccel formats, alpha
  sources, gray, rgb), odd rasters of subsampled formats.
* 4:4:4 (yuv444p, yuv444p10le / yuv444p12le -- ProRes 4444 without alpha, the
  XYZT landmark plate): swscale forces full chroma interpolation for
  non-subsampled input and
  writes RGB with its 16-bit fixed-point matrix (``yuv2rgb_write_full``);
  ``planes_to_rgb`` reproduces that integer arithmetic exactly too.

Alpha stays with the raster sources; ``ClipDecoder`` returns opaque RGB.

Main callers:
- ``plan.build_tensor_plan`` (``probe_video`` per media file at plan time; it
  should call ``check_source_color`` for video clips -- see its docstring).
- ``pipeline.DirectSources`` / ``pipeline.PrefetchSources`` (``open_source``,
  ``frame_at`` / ``packed_at`` + ``planes_to_rgb``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

import av
import numpy as np
import torch

from .errors import TensorRenderError
from .hdr import HDRTransfer, hdr_to_sdr
from .support import reject
from .swscale_fixedpoint import c_div as _c_div, rounded_div as _rounded_div

if TYPE_CHECKING:  # pragma: no cover
    from .plan import LayerSpec


# --------------------------------------------------------------------------- colour tags


# Canonical matrix names -> (Kr, Kb) and swscale's ``ff_yuv2rgb_coeffs`` inverse table
# (crv, cbu, cgu, cgv in 16.16 for full-range chroma; ``sws_getCoefficients``).
_MATRICES: dict[str, tuple[float, float, tuple[int, int, int, int]]] = {
    "bt709": (0.2126, 0.0722, (117489, 138438, 13975, 34925)),
    "bt601": (0.299, 0.114, (104597, 132201, 25675, 53279)),
    "bt2020nc": (0.2627, 0.0593, (110013, 140363, 12276, 42626)),
}
# ffprobe / libavutil names -> canonical matrix (``None`` = unspecified -> swscale default 601).
_MATRIX_BY_TAG: dict[str, Optional[str]] = {
    "bt709": "bt709",
    "bt470bg": "bt601",
    "smpte170m": "bt601",
    "bt2020nc": "bt2020nc",
    "unknown": None,
    "unspecified": None,
    "": None,
}
# libavutil enum AVColorSpace / AVColorRange / AVColorTransferCharacteristic values
# (what PyAV exposes as ints on frames and codec contexts) -> ffprobe names.
_AV_COLORSPACE_NAMES = {
    0: "gbr", 1: "bt709", 2: "unknown", 4: "fcc", 5: "bt470bg", 6: "smpte170m", 7: "smpte240m",
    8: "ycgco", 9: "bt2020nc", 10: "bt2020c", 11: "smpte2085", 12: "chroma-derived-nc",
    13: "chroma-derived-c", 14: "ictcp",
}
_AV_COLOR_RANGE_NAMES = {0: "unknown", 1: "tv", 2: "pc"}
_AV_TRANSFER_NAMES = {
    1: "bt709", 2: "unknown", 4: "gamma22", 5: "gamma28", 6: "smpte170m", 7: "smpte240m",
    8: "linear", 9: "log100", 10: "log316", 11: "iec61966-2-4", 12: "bt1361e",
    13: "iec61966-2-1", 14: "bt2020-10", 15: "bt2020-12", 16: "smpte2084", 17: "smpte428",
    18: "arib-std-b67",
}
_AV_PRIMARIES_NAMES = {
    1: "bt709", 2: "unknown", 4: "bt470m", 5: "bt470bg", 6: "smpte170m", 7: "smpte240m",
    8: "film", 9: "bt2020", 10: "smpte428", 11: "smpte431", 12: "smpte432", 22: "jedec-p22",
}
_HDR_TRANSFERS = {"smpte2084": "PQ", "arib-std-b67": "HLG"}

# libswscale filter for the preview's decoder-side downscale (``ClipDecoder``
# ``decode_size``).  ``BILINEAR`` is swscale's support-scaled triangle filter --
# the same family the renderer's calibrated whole-raster minification port
# (``sampler.resize_exact_aspect_opaque``) reproduces -- so a downscaled leaf
# looks like the native path's minified leaf rather than a point-sampled one.
DECODE_SCALE_INTERPOLATION = "BILINEAR"

# Supported decoded pixel formats -> (bit depth, horizontal chroma shift, vertical chroma
# shift, full range implied by the format).
_PIXEL_FORMATS: dict[str, tuple[int, int, int, bool]] = {
    "yuv420p": (8, 1, 1, False),
    "yuvj420p": (8, 1, 1, True),
    "yuv422p": (8, 1, 0, False),
    "yuvj422p": (8, 1, 0, True),
    "yuv420p10le": (10, 1, 1, False),
    "yuv422p10le": (10, 1, 0, False),
    "yuv444p": (8, 0, 0, False),
    "yuvj444p": (8, 0, 0, True),
    "yuv444p10le": (10, 0, 0, False),
    "yuv444p12le": (12, 0, 0, False),  # ProRes 4444 profiles decode as 12-bit
}


@dataclass(frozen=True)
class SourceColor:
    """The resolved colour-in policy of one decoded video stream (what swscale would do).

    ``matrix`` is one of ``_MATRICES``; ``full_range`` selects the 0..255 /
    16..235 luma scale; ``pixel_format`` / ``bit_depth`` / chroma shifts describe
    the plane layout the kernels expect.
    """

    pixel_format: str
    bit_depth: int
    chroma_shift_x: int
    chroma_shift_y: int
    matrix: str
    full_range: bool
    hdr_transfer: Optional[HDRTransfer] = None

    def describe(self) -> str:
        suffix = f" {self.hdr_transfer}->sdr" if self.hdr_transfer is not None else ""
        return f"{self.pixel_format} {self.matrix} {'full' if self.full_range else 'limited'}{suffix}"


def resolve_source_color(
    pixel_format: str,
    *,
    matrix_tag: str,
    range_tag: str,
    transfer_tag: str,
    primaries_tag: str,
    subject: str,
) -> SourceColor:
    """Turn a stream's (pix_fmt, colorspace, color_range, color_trc) tags into a ``SourceColor``.

    Pythonese: recognize PQ / HLG only with their required Rec.2020 matrix and
    primaries; reject a pixel format outside the supported planar 8/10-bit
    4:2:0 / 4:2:2 set; map the matrix tag (unspecified ->
    BT.601, exactly like swscale) or reject an exotic one; the range is full when the
    tag says ``pc`` or the format is a ``yuvj`` one, limited otherwise (unspecified
    -> limited, like swscale).

    Main callers: ``check_source_color`` (plan time, ffprobe strings) and
    ``ClipDecoder`` (decode time, the first frame's tags) -- both must agree, so
    both go through here.
    """

    layout = _PIXEL_FORMATS.get(pixel_format)
    if layout is None:
        raise reject(
            "source pixel format (unsupported)",
            f"{subject}: pix_fmt={pixel_format!r} (supported: {', '.join(sorted(_PIXEL_FORMATS))})",
        )
    bit_depth, shift_x, shift_y, format_full = layout
    if matrix_tag not in _MATRIX_BY_TAG:
        raise reject(
            "source colour matrix (unsupported)",
            f"{subject}: colorspace={matrix_tag!r} (supported: bt709, bt470bg, smpte170m, bt2020nc, unspecified)",
        )
    matrix = _MATRIX_BY_TAG[matrix_tag] or "bt601"
    hdr_transfer: Optional[HDRTransfer] = None
    if transfer_tag in _HDR_TRANSFERS:
        if matrix != "bt2020nc" or primaries_tag != "bt2020":
            raise reject(
                "source HDR metadata (malformed)",
                f"{subject}: color_trc={transfer_tag} requires colorspace=bt2020nc and "
                f"color_primaries=bt2020; got colorspace={matrix_tag}, "
                f"color_primaries={primaries_tag}",
            )
        hdr_transfer = "hlg" if transfer_tag == "arib-std-b67" else "pq"
    if range_tag not in {"tv", "pc", "unknown", ""}:
        raise TensorRenderError(f"{subject}: unexpected color_range tag {range_tag!r}")
    full_range = format_full or range_tag == "pc"
    return SourceColor(
        pixel_format=pixel_format,
        bit_depth=bit_depth,
        chroma_shift_x=shift_x,
        chroma_shift_y=shift_y,
        matrix=matrix,
        full_range=full_range,
        hdr_transfer=hdr_transfer,
    )


# --------------------------------------------------------------------------- probe


@dataclass(frozen=True)
class VideoProbe:
    """Facts about a media file's first video stream that the plan needs before decoding.

    ``rotation_degrees`` comes from the container display matrix (what the
    FFmpeg CLI autorotates by; PyAV neither exposes nor applies it), so the
    plan can refuse rotated sources loudly until spatial intrinsics are ported.

    The colour tags are ffprobe's stream-level strings (``"unknown"`` when the
    stream carries none): ``check_source_color`` turns them into a plan-time
    verdict; the decoder re-derives the same policy from the first decoded
    frame (whose tags are what swscale actually reads -- ProRes, for one,
    carries its matrix in the frame header, which ffprobe reports as unknown).
    """

    width: int
    height: int
    pixel_format: str
    sample_aspect_ratio: tuple[int, int]
    rotation_degrees: int
    duration: Optional[Fraction]
    codec_name: str = ""
    color_range: str = "unknown"
    color_space: str = "unknown"
    color_transfer: str = "unknown"
    color_primaries: str = "unknown"
    bit_depth: int = 8


def _pixel_format_bit_depth(pixel_format: str) -> int:
    """Bits per sample of the first component of ``pixel_format`` (0 when unknown to libav)."""

    try:
        return int(av.VideoFormat(pixel_format).components[0].bits)
    except (ValueError, IndexError, AttributeError):
        return 0


def _first_frame_color_tags(media_path: Path) -> tuple[str, str]:
    """(color_range, color_space) names of the first decoded frame ("unknown" when absent / undecodable)."""

    try:
        with av.open(str(media_path)) as container:
            for frame in container.decode(video=0):
                return (
                    _AV_COLOR_RANGE_NAMES.get(int(frame.color_range), "unknown"),
                    _AV_COLORSPACE_NAMES.get(int(frame.colorspace), "unknown"),
                )
    except (av.AVError, OSError, ValueError, IndexError):  # pragma: no cover - probe stays "unknown"
        pass
    return "unknown", "unknown"


def probe_video(media_path: Path) -> VideoProbe:
    """Read the first video stream's raster facts and colour tags with ffprobe (JSON), loudly."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise TensorRenderError("ffprobe is required to probe source media for the tensor plan")
    completed = subprocess.run(
        (ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", str(media_path)),
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise TensorRenderError(f"ffprobe could not read {media_path}: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise TensorRenderError(f"{media_path} has no video stream")
    stream = streams[0]
    rotation = 0
    for side in stream.get("side_data_list") or ():
        if "rotation" in side:
            rotation = int(round(float(side["rotation"]))) % 360
    tag_rotation = (stream.get("tags") or {}).get("rotate")
    if tag_rotation not in (None, "0"):
        rotation = int(round(float(tag_rotation))) % 360
    sar_raw = stream.get("sample_aspect_ratio") or "1:1"
    if sar_raw in {"0:1", "N/A"}:
        sar_raw = "1:1"
    sar_num, sar_den = (int(part) for part in sar_raw.split(":"))
    duration_raw = stream.get("duration") or (payload.get("format") or {}).get("duration")
    duration = Fraction(str(duration_raw)) if duration_raw not in (None, "N/A") else None
    pixel_format = str(stream.get("pix_fmt") or "")

    def tag(name: str) -> str:
        value = stream.get(name)
        return str(value) if value not in (None, "", "N/A") else "unknown"

    color_range, color_space = tag("color_range"), tag("color_space")
    if pixel_format.startswith("yuv") and (color_range == "unknown" or color_space == "unknown"):
        # ProRes (and other in-frame-header codecs) carry the tags on the decoded frame, not
        # on the container stream ffprobe reports; libavfilter -- hence the reference -- and
        # ``ClipDecoder._resolve_color`` see the FRAME tags, so the plan must too (effect
        # ports negotiate their yuva444p bridge from them).
        frame_range, frame_space = _first_frame_color_tags(media_path)
        color_range = frame_range if color_range == "unknown" else color_range
        color_space = frame_space if color_space == "unknown" else color_space

    return VideoProbe(
        width=int(stream["width"]),
        height=int(stream["height"]),
        pixel_format=pixel_format,
        sample_aspect_ratio=(sar_num, sar_den),
        rotation_degrees=rotation,
        duration=duration,
        codec_name=str(stream.get("codec_name") or ""),
        color_range=color_range,
        color_space=color_space,
        color_transfer=tag("color_transfer"),
        color_primaries=tag("color_primaries"),
        bit_depth=_pixel_format_bit_depth(pixel_format),
    )


def check_source_color(probe: VideoProbe, *, subject: str) -> SourceColor:
    """Plan-time colour-in verdict for a *video* source (not for PNG rasters).

    Raises ``TensorRenderUnsupported`` naming the tag when the stream's pixel
    format, matrix or transfer is outside the supported class, so a project
    fails before any frame is decoded; returns the resolved policy otherwise.
    Odd rasters of chroma-subsampled formats are refused here too (the kernels
    need whole chroma blocks; the reference's swscale would silently scale).

    Main callers: ``plan._reject_unsupported_source`` should call this for
    every layer whose ``source_kind == "clip"`` (requested core diff); until
    then ``ClipDecoder`` applies the same rule to the first decoded frame.
    """

    color = resolve_source_color(
        probe.pixel_format,
        matrix_tag=probe.color_space,
        range_tag=probe.color_range,
        transfer_tag=probe.color_transfer,
        primaries_tag=probe.color_primaries,
        subject=subject,
    )
    _check_raster_parity(probe.width, probe.height, color, subject=subject)
    return color


def _check_raster_parity(width: int, height: int, color: SourceColor, *, subject: str) -> None:
    if (color.chroma_shift_x and width % 2) or (color.chroma_shift_y and height % 2):
        raise reject(
            "source pixel format (unsupported)",
            f"{subject}: {color.pixel_format} raster {width}x{height} is not chroma-block aligned",
        )


# --------------------------------------------------------------------------- planes


@dataclass(frozen=True)
class PlaneLayout:
    """Geometry of one packed planar frame: Y ``[H, W]`` then U, V ``[Hc, Wc]`` back to back."""

    height: int
    width: int
    chroma_height: int
    chroma_width: int

    @property
    def luma_size(self) -> int:
        return self.height * self.width

    @property
    def chroma_size(self) -> int:
        return self.chroma_height * self.chroma_width

    @property
    def total(self) -> int:
        return self.luma_size + 2 * self.chroma_size


@dataclass(frozen=True)
class SourceFrame:
    """One decoded frame as raw planes on the CPU, ready for a single upload + ``planes_to_rgb``.

    ``planes`` is a flat 1-D tensor: ``uint8`` for 8-bit sources, ``int16``
    holding the 0..1023 codes for 10-bit ones (torch has no arithmetic uint16;
    the values fit).  Its byte layout for 8-bit 4:2:0 is the one
    ``av.VideoFrame.from_ndarray(..., format="yuv420p")`` expects when reshaped
    to ``[H*3/2, W]``.
    """

    planes: torch.Tensor
    layout: PlaneLayout
    color: SourceColor


@dataclass(frozen=True)
class HDRFrame:
    """High-precision code-space Rec.2020 RGB awaiting the device-side SDR LUT.

    Main callers:
    - ``ClipDecoder.packed_at`` for HLG/PQ frames.
    - ``pipeline._LayerWorker.pop`` after its one host-to-device upload.

    Why this exists:
    The ordinary path intentionally quantizes through the reference's 8-bit
    ``format=rgba`` link. HDR conform instead enters ``format=gbrpf32le``
    before the 3D LUT, so preserving that float32 boundary avoids clipping and
    large errors around out-of-gamut edges.
    """

    rgb: torch.Tensor
    transfer: HDRTransfer


def pack_planes(frame: av.VideoFrame, color: SourceColor) -> SourceFrame:
    """Copy a decoded planar frame's Y, U, V planes (stride removed) into one flat CPU tensor."""

    height, width = frame.height, frame.width
    layout = PlaneLayout(
        height=height,
        width=width,
        chroma_height=-(-height >> color.chroma_shift_y) if color.chroma_shift_y else height,
        chroma_width=-(-width >> color.chroma_shift_x) if color.chroma_shift_x else width,
    )
    dtype = np.uint8 if color.bit_depth == 8 else np.uint16
    flat = np.empty(layout.total, dtype=dtype)

    def plane(index: int, rows: int, cols: int) -> np.ndarray:
        raw = frame.planes[index]
        stride = raw.line_size
        buffer = np.frombuffer(raw, dtype=np.uint8, count=stride * rows).reshape(rows, stride)
        return buffer.view(dtype)[:, :cols]

    flat[:layout.luma_size] = plane(0, height, width).reshape(-1)
    flat[layout.luma_size:layout.luma_size + layout.chroma_size] = plane(1, layout.chroma_height, layout.chroma_width).reshape(-1)
    flat[layout.luma_size + layout.chroma_size:] = plane(2, layout.chroma_height, layout.chroma_width).reshape(-1)
    if color.bit_depth != 8:
        flat = flat.view(np.int16)  # 0..1023 codes fit; torch has no arithmetic uint16
    return SourceFrame(planes=torch.from_numpy(flat), layout=layout, color=color)


def _split_planes(planes: torch.Tensor, layout: PlaneLayout) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = planes.reshape(-1)
    y = flat[:layout.luma_size].view(layout.height, layout.width)
    u = flat[layout.luma_size:layout.luma_size + layout.chroma_size].view(layout.chroma_height, layout.chroma_width)
    v = flat[layout.luma_size + layout.chroma_size:].view(layout.chroma_height, layout.chroma_width)
    return y, u, v


def planes_to_rgb(planes: torch.Tensor, layout: PlaneLayout, color: SourceColor) -> torch.Tensor:
    """Packed planes (any device) -> RGB ``[3, H, W]`` float32 0..1 code space, the reference's ``format=rgba``.

    Dispatches on the source class (see the module docstring for why the two
    kernels differ): 8-bit sources take the exact-float path, 10-bit sources
    the swscale integer path.  Both quantise to 8-bit codes because that is
    what the reference's ``rgba`` link carries.

    Main callers: ``ClipDecoder.frame_at`` (serial) and ``pipeline._LayerWorker.pop``
    (GPU thread of the pipelined loop) -- one kernel, so both loops are identical.
    """

    if planes.numel() != layout.total:
        raise TensorRenderError(f"packed planes hold {planes.numel()} samples; layout needs {layout.total}")
    if color.chroma_shift_x == 0:
        rgb = _rgb_from_planes_swscale_full(planes, layout, color)
    elif color.bit_depth == 8:
        rgb = _rgb_from_planes_8bit(planes, layout, color)
    else:
        rgb = _rgb_from_planes_swscale_tables(planes, layout, color)
    return hdr_to_sdr(rgb, color.hdr_transfer) if color.hdr_transfer is not None else rgb


class _DeviceConstants:
    """Per-(device, key) cache of the small constant tensors the kernels need."""

    def __init__(self) -> None:
        self._cache: dict[tuple, object] = {}

    def get(self, key: tuple, like: torch.Tensor, build):
        full_key = (str(like.device), like.device.index or 0) + key
        cached = self._cache.get(full_key)
        if cached is None:
            cached = self._cache[full_key] = build(like.device)
        return cached


_CONSTANTS = _DeviceConstants()


def _rgb_matrix(color: SourceColor) -> tuple[list[list[float]], list[float]]:
    """Row-major 3x3 ``M`` and offset ``o`` so that RGB(0..1) = M @ [Y, Cb, Cr] + o on 8-bit codes."""

    kr, kb, _ = _MATRICES[color.matrix]
    kg = 1.0 - kr - kb
    if color.full_range:
        y_scale, c_scale, y_offset = 1.0 / 255.0, 1.0 / 255.0, 0.0
    else:
        y_scale, c_scale, y_offset = 1.0 / 219.0, 1.0 / 224.0, 16.0
    r_cr = 2.0 * (1.0 - kr)
    b_cb = 2.0 * (1.0 - kb)
    g_cb = -2.0 * kb * (1.0 - kb) / kg
    g_cr = -2.0 * kr * (1.0 - kr) / kg
    matrix = [
        [y_scale, 0.0, r_cr * c_scale],
        [y_scale, g_cb * c_scale, g_cr * c_scale],
        [y_scale, b_cb * c_scale, 0.0],
    ]
    offset = [
        -y_offset * y_scale - r_cr * 128.0 * c_scale,
        -y_offset * y_scale - (g_cb + g_cr) * 128.0 * c_scale,
        -y_offset * y_scale - b_cb * 128.0 * c_scale,
    ]
    return matrix, offset


def _rgb_from_planes_8bit(planes: torch.Tensor, layout: PlaneLayout, color: SourceColor) -> torch.Tensor:
    """8-bit planar -> RGB: nearest chroma, exact matrix, round to 8-bit codes (swscale's unscaled converters)."""

    def build(device: torch.device):
        matrix, offset = _rgb_matrix(color)
        return (
            torch.tensor(matrix, dtype=torch.float32, device=device),
            torch.tensor(offset, dtype=torch.float32, device=device).view(3, 1, 1),
        )

    matrix, offset = _CONSTANTS.get(("matrix8", color.matrix, color.full_range), planes, build)
    y, u, v = _split_planes(planes, layout)
    y = y.float()
    if color.chroma_shift_x:
        u = u.repeat_interleave(2, dim=1)
        v = v.repeat_interleave(2, dim=1)
    if color.chroma_shift_y:
        u = u.repeat_interleave(2, dim=0)
        v = v.repeat_interleave(2, dim=0)
    ycbcr = torch.stack((y, u.float(), v.float()), dim=0).reshape(3, layout.luma_size)
    rgb = (torch.matmul(matrix, ycbcr).view(3, layout.height, layout.width) + offset) * 255.0
    return rgb.round_().clamp_(0.0, 255.0).div_(255.0)


# ---- swscale generic-path integer arithmetic (10-bit sources) ----------------------------


def swscale_yuv2rgb_tables(color: SourceColor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Port of ``ff_yuv2rgb_c_init_tables`` (libswscale n8.0, 32 bpp) for one matrix / range.

    Returns ``(y_table, delta_r_of_v, delta_gu_of_u, delta_gv_of_v, delta_b_of_u, yoffs)``:
    ``R = y_table[yoffs + Y8 + delta_r[V8]]``, ``G = y_table[yoffs + Y8 + delta_gu[U8] + delta_gv[V8]]``,
    ``B = y_table[yoffs + Y8 + delta_b[U8]]`` (indices clamped to the table), where the
    deltas are the integer *luma-table* offsets swscale folds the chroma into.
    """

    _, _, inv_table = _MATRICES[color.matrix]
    crv, cbu, cgu, cgv = inv_table[0], inv_table[1], -inv_table[2], -inv_table[3]
    cy, oy = 1 << 16, 0
    if not color.full_range:
        cy = _c_div(cy * 255, 219)
        oy = 16 << 16
    else:
        crv, cbu, cgu, cgv = (_c_div(c * 224, 255) for c in (crv, cbu, cgu, cgv))
    # contrast = saturation = 1<<16, brightness = 0: cy / crv / ... unchanged.
    crv, cbu, cgu, cgv = (_c_div((c << 16) + 0x8000, max(cy, 1)) for c in (crv, cbu, cgu, cgv))
    luma_headroom = 512  # YUVRGB_TABLE_LUMA_HEADROOM; the chroma tables' HEADROOM clamps the 8-bit index
    yoffs = (384 if color.full_range else 326) + luma_headroom
    table_plane_size = 1024 + 2 * luma_headroom
    yb = -(384 << 16) - luma_headroom * cy - oy
    y_table = np.clip((yb + cy * np.arange(table_plane_size, dtype=np.int64) + 0x8000) >> 16, 0, 255).astype(np.int64)

    def delta(inc: int) -> np.ndarray:
        # fill_table / fill_gv_table over the 8-bit index (the ±512 headroom clamps to 0..255).
        codes = np.arange(256, dtype=np.int64)
        return ((codes * inc) >> 16) - (inc >> 9)

    return y_table, delta(crv), delta(cgu), delta(cgv), delta(cbu), yoffs


def swscale_vertical_filter(src_len: int, dst_len: int, *, src_pos: int = 128, dst_pos: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Port of libswscale ``initFilter`` for the vertical *chroma* pass with default flags (bicubic).

    Returns ``(pos[dst_len], coeff[dst_len, K])`` (int64): output row ``r`` =
    ``sum_j coeff[r, j] * src[pos[r] + j]`` with coefficients summing to 4096
    (``one = 1 << 12``), borders folded into the edge samples exactly as swscale
    does, filter width aligned to 2 (``filterAlign`` on NEON / MMX hosts).
    Positions are the chroma-plane centre convention swscale derives from the
    default ``*_chr_pos = -513`` (``get_local_pos``): 128 for both grids.
    """

    one = 1 << 12
    filter_align = 2
    x_inc = _c_div((src_len << 16) + (dst_len >> 1), dst_len)
    ratio = src_len // dst_len
    fone = 1 << (54 - min(ratio.bit_length() - 1 if ratio > 0 else 0, 8))  # av_log2(0) == 0
    if abs(x_inc - 0x10000) < 10 and src_pos == dst_pos:
        filter_size = 1
        filt = np.full((dst_len, 1), fone, dtype=np.int64)
        pos = np.arange(dst_len, dtype=np.int64)
    else:
        size_factor = 4  # SWS_BICUBIC
        filter_size = 1 + size_factor if x_inc <= (1 << 16) else 1 + _c_div(size_factor * src_len + dst_len - 1, dst_len)
        filter_size = max(min(filter_size, src_len - 2), 1)
        filt = np.zeros((dst_len, filter_size), dtype=np.int64)
        pos = np.zeros(dst_len, dtype=np.int64)
        b_param = 0
        c_param = int(0.6 * (1 << 24))
        x_dst_in_src = ((dst_pos * x_inc) >> 7) - ((src_pos * 0x10000) >> 7)
        for i in range(dst_len):
            xx = _c_div(x_dst_in_src - (filter_size - 2) * (1 << 16), 1 << 17)
            pos[i] = xx
            for j in range(filter_size):
                d = abs((xx << 17) - x_dst_in_src) << 13
                if x_inc > (1 << 16):
                    d = _c_div(d * dst_len, src_len)
                if d >= 1 << 31:
                    coeff = 0
                else:
                    dd = (d * d) >> 30
                    ddd = (dd * d) >> 30
                    if d < 1 << 30:
                        coeff = ((12 * (1 << 24) - 9 * b_param - 6 * c_param) * ddd
                                 + (-18 * (1 << 24) + 12 * b_param + 6 * c_param) * dd
                                 + (6 * (1 << 24) - 2 * b_param) * (1 << 30))
                    else:
                        coeff = ((-b_param - 6 * c_param) * ddd
                                 + (6 * b_param + 30 * c_param) * dd
                                 + (-12 * b_param - 48 * c_param) * d
                                 + (8 * b_param + 24 * c_param) * (1 << 30))
                    coeff = _c_div(coeff, _c_div(1 << 54, fone))
                filt[i, j] = coeff
                xx += 1
            x_dst_in_src += 2 * x_inc
    # reduce the filter size (swscale reads the FIRST tap for the left cut-off; keep that quirk)
    cutoff_limit = 0.002 * fone
    min_filter_size = 0
    for i in range(dst_len - 1, -1, -1):
        size = filter_size
        cutoff = 0
        for _ in range(filter_size):
            cutoff += abs(int(filt[i, 0]))
            if cutoff > cutoff_limit:
                break
            if i < dst_len - 1 and pos[i] >= pos[i + 1]:
                break
            filt[i, :-1] = filt[i, 1:]
            filt[i, -1] = 0
            pos[i] += 1
        cutoff = 0
        for j in range(filter_size - 1, 0, -1):
            cutoff += abs(int(filt[i, j]))
            if cutoff > cutoff_limit:
                break
            size -= 1
        min_filter_size = max(min_filter_size, size)
    if min_filter_size == 1 and filter_align == 2:
        filter_align = 1
    out_size = (min_filter_size + filter_align - 1) & ~(filter_align - 1)
    reduced = np.zeros((dst_len, out_size), dtype=np.int64)
    reduced[:, :min(out_size, filter_size)] = filt[:, :min(out_size, filter_size)]
    # fix borders (fold out-of-range taps into the edge samples)
    for i in range(dst_len):
        if pos[i] < 0:
            for j in range(1, out_size):
                left = max(j + int(pos[i]), 0)
                reduced[i, left] += reduced[i, j]
                reduced[i, j] = 0
            pos[i] = 0
        if pos[i] + out_size > src_len:
            shift = int(pos[i]) + min(out_size - src_len, 0)
            acc = 0
            for j in range(out_size - 1, -1, -1):
                if pos[i] + j >= src_len:
                    acc += reduced[i, j]
                    reduced[i, j] = 0
            for j in range(out_size - 1, -1, -1):
                reduced[i, j] = 0 if j < shift else reduced[i, j - shift]
            pos[i] -= shift
            reduced[i, src_len - 1 - int(pos[i])] += acc
    # normalise with error diffusion so every row sums to ``one``
    coeff = np.zeros((dst_len, out_size), dtype=np.int64)
    for i in range(dst_len):
        total = int(reduced[i].sum())
        total = _c_div(total + one // 2, one)
        if total == 0:
            total = 1
        error = 0
        for j in range(out_size):
            v = int(reduced[i, j]) + error
            int_v = _rounded_div(v, total)
            coeff[i, j] = int_v
            error = v - int_v * total
    return pos, coeff


def _rgb_from_planes_swscale_tables(planes: torch.Tensor, layout: PlaneLayout, color: SourceColor) -> torch.Tensor:
    """10-bit planar -> RGB exactly as swscale's generic path (see module docstring)."""

    if color.chroma_shift_x != 1:
        raise TensorRenderError(f"swscale table path expects 4:2:0 / 4:2:2 chroma; got {color.pixel_format}")

    def build_tables(device: torch.device):
        y_table, d_r, d_gu, d_gv, d_b, yoffs = swscale_yuv2rgb_tables(color)
        as_t = lambda a: torch.from_numpy(a.astype(np.int32)).to(device)  # noqa: E731
        return as_t(y_table), as_t(d_r), as_t(d_gu), as_t(d_gv), as_t(d_b), yoffs

    def build_filter(device: torch.device):
        pos, coeff = swscale_vertical_filter(layout.chroma_height, layout.height)
        taps = coeff.shape[1]
        index = pos[:, None] + np.arange(taps)[None, :]
        return (
            torch.from_numpy(index.astype(np.int64)).to(device),
            torch.from_numpy(coeff.astype(np.int32)).to(device).view(layout.height, taps, 1),
        )

    y_table, d_r, d_gu, d_gv, d_b, yoffs = _CONSTANTS.get(("sws_tables", color.matrix, color.full_range), planes, build_tables)
    y10, u10, v10 = _split_planes(planes, layout)
    y8 = torch.div(y10.to(torch.int32) + 2, 4, rounding_mode="floor")
    if color.chroma_shift_y:
        index, coeff = _CONSTANTS.get(("sws_vfilter", layout.chroma_height, layout.height), planes, build_filter)
        # 15-bit chroma (x << 5), int16 taps summing to 4096, then (acc + 2^18) >> 19.
        u15 = u10.to(torch.int32) * 32
        v15 = v10.to(torch.int32) * 32
        u8 = torch.div((u15[index] * coeff).sum(dim=1) + (1 << 18), 1 << 19, rounding_mode="floor")
        v8 = torch.div((v15[index] * coeff).sum(dim=1) + (1 << 18), 1 << 19, rounding_mode="floor")
    else:
        u8 = torch.div(u10.to(torch.int32) + 2, 4, rounding_mode="floor")
        v8 = torch.div(v10.to(torch.int32) + 2, 4, rounding_mode="floor")
    u8 = u8.clamp_(0, 255)
    v8 = v8.clamp_(0, 255)
    # Chroma is shared per horizontal luma pair (swscale's chrDstW = ceil(W/2) for RGB output).
    off_r = d_r[v8].repeat_interleave(2, dim=1)
    off_g = (d_gu[u8] + d_gv[v8]).repeat_interleave(2, dim=1)
    off_b = d_b[u8].repeat_interleave(2, dim=1)
    base = y8 + yoffs
    limit = int(y_table.numel()) - 1
    r = y_table[(base + off_r).clamp_(0, limit)]
    g = y_table[(base + off_g).clamp_(0, limit)]
    b = y_table[(base + off_b).clamp_(0, limit)]
    return torch.stack((r, g, b), dim=0).float().div_(255.0)


def _round_to_int16(value: int) -> int:
    """libswscale ``roundToInt16``: ``(f + 2^15) >> 16`` clipped to the int16 range."""

    return max(-0x8000, min(0x7FFF, (value + (1 << 15)) >> 16))


def swscale_full_chroma_coefficients(color: SourceColor) -> tuple[int, int, int, int, int, int]:
    """``ff_yuv2rgb_c_init_tables``' 16-bit coefficients used by the *full chroma* RGB writers.

    Returns ``(y_offset, y_coeff, v2r, v2g, u2g, u2b)`` -- ``c->yuv2rgb_*`` in
    ``SwsInternal`` -- for the 4:4:4 path (``yuv2rgb_write_full``): 17-bit Y' /
    signed chroma in, ``R = ((Y - y_offset) * y_coeff + 2^21 + V * v2r) >> 22``.
    """

    _, _, inv_table = _MATRICES[color.matrix]
    crv, cbu, cgu, cgv = inv_table[0], inv_table[1], -inv_table[2], -inv_table[3]
    cy, oy = 1 << 16, 0
    if not color.full_range:
        cy = _c_div(cy * 255, 219)
        oy = 16 << 16
    else:
        crv, cbu, cgu, cgv = (_c_div(c * 224, 255) for c in (crv, cbu, cgu, cgv))
    return (
        _round_to_int16(oy * (1 << 9)),
        _round_to_int16(cy * (1 << 13)),
        _round_to_int16(crv * (1 << 13)),
        _round_to_int16(cgv * (1 << 13)),
        _round_to_int16(cgu * (1 << 13)),
        _round_to_int16(cbu * (1 << 13)),
    )


def _rgb_from_planes_swscale_full(planes: torch.Tensor, layout: PlaneLayout, color: SourceColor) -> torch.Tensor:
    """4:4:4 planar (8 / 10-bit) -> RGB exactly as swscale's full-chroma writer.

    A non-subsampled source forces ``SWS_FULL_CHR_H_INT`` (``utils.c``: "input
    having non subsampled chroma"), so the reference does not use the lookup
    tables: ``yuv2rgb_full_1_c_template`` scales the 15-bit samples to 17 bits
    (``* 4``), centres chroma on 128, and ``yuv2rgb_write_full`` applies the
    16-bit fixed-point matrix with a ``+ 2^21`` rounding term, clips to 30 bits
    and shifts by 22.  Integer arithmetic here reproduces it bit-exactly.
    """

    def build(device: torch.device):
        return tuple(torch.tensor(c, dtype=torch.int32, device=device) for c in swscale_full_chroma_coefficients(color))

    y_offset, y_coeff, v2r, v2g, u2g, u2b = _CONSTANTS.get(("sws_full", color.matrix, color.full_range), planes, build)
    y, u, v = _split_planes(planes, layout)
    shift = 15 - color.bit_depth
    y17 = y.to(torch.int32) * (4 << shift)
    u17 = (u.to(torch.int32) * (1 << shift) - (128 << 7)) * 4
    v17 = (v.to(torch.int32) * (1 << shift) - (128 << 7)) * 4
    luma = (y17 - y_offset) * y_coeff + (1 << 21)
    limit = (1 << 30) - 1
    r = (luma + v17 * v2r).clamp_(0, limit)
    g = (luma + v17 * v2g + u17 * u2g).clamp_(0, limit)
    b = (luma + u17 * u2b).clamp_(0, limit)
    rgb = torch.stack((r, g, b), dim=0)
    return torch.div(rgb, 1 << 22, rounding_mode="floor").float().div_(255.0)


# --------------------------------------------------------------------------- sources


class FrameSource(Protocol):
    """What the renderer needs from any per-layer pixel source (A0 seam).

    ``frame_at(instant)`` returns the source raster owning ``instant`` (a
    layer-local source time) as a float32 code-space tensor ``[C, H, W]`` in
    0..1 on the render device: ``C == 3`` for opaque video, ``C == 4`` for
    straight-alpha rasters (titles, generators, still images with alpha).
    Implementations: ``ClipDecoder`` (video, X1-X3), ``RasterSource`` (stills /
    titles / captions / Custom Solid generators, X5).
    """

    frames_decoded: int

    def frame_at(self, source_time: Fraction) -> torch.Tensor: ...

    def close(self) -> None: ...


def open_source(layer: "LayerSpec", *, device: torch.device, threads: int = 4) -> FrameSource:
    """Open the pixel source for ``layer``.

    Why this exists: the renderer loop must not know how a layer's pixels are
    produced (a decoded video stream, or one image held for the layer's whole
    window), so the dispatch on the layer's source kind lives here beside the
    sources.

    Main callers: ``renderer.render_document`` (when a layer becomes active).
    """

    if layer.source_kind == "raster":
        return RasterSource(layer.media_path, device=device)
    raster = layer.decode_raster
    if raster is None:
        raise TensorRenderError(f"{layer.path}: video layer carries no decode raster (plan lowering bug)")
    return ClipDecoder(
        layer.media_path,
        device=device,
        threads=threads,
        reverse_cache_bytes=128 * 1024 * 1024 if layer.reverse_decode_cache else 0,
        reverse_cache_frames=128,
        decode_size=None if raster.is_native else raster.encoded_size,
    )


class RasterSource:
    """One image decoded once and returned for every instant of a layer (X5).

    What it does (Pythonese)
    ------------------------
    On construction: open the file with PyAV, decode its single frame, convert
    it to ``rgba``, upload it to the render device as float 0..1, and keep it.
    ``frame_at`` then ignores the instant and hands back that tensor.
    ``frames_decoded`` stays 1 for the layer's whole life.

    Why this exists
    ---------------
    Stills, titles, captions and Custom Solid generators have no source
    timeline: the reference feeds each of them to FFmpeg as a ``-loop 1
    -framerate fps`` image input and only trims the looped stream
    (``ffmpeg._append_video_input``, ``ffmpeg.py:1338``; ``_video_chain``,
    ``ffmpeg.py:9853``).  Re-decoding that image per output frame would be pure
    waste, and re-rasterizing a title per frame would be *wrong* -- the text
    raster is produced exactly once, before the render, by
    ``text.resolve_text_clip_raster``.

    Colour and alpha
    ----------------
    PyAV (not PIL) does the decode, and the conversion is unconditionally to
    ``rgba``, so it is the same swscale call the reference's ``format=rgba``
    makes (``ffmpeg.py:9935``): an RGBA PNG passes through untouched, an opaque
    one gains a solid alpha plane, a palette PNG keeps its palette alpha, and a
    JPEG goes through swscale's full-range BT.601 matrix.  Sniffing the pixel
    format to skip alpha would be a second, divergent decision -- an opaque
    alpha plane costs one extra channel on one image.

    The result is *code space* (the same space ``ClipDecoder`` returns), which
    ``renderer.placed`` linearizes; alpha stays **straight** (libpng and the
    reference's overlay both treat RGBA as unassociated) and the renderer
    premultiplies after the warp.

    Main callers: ``open_source`` for layers with ``source_kind == "raster"``.
    """

    def __init__(self, media_path: Path, *, device: torch.device) -> None:
        self.media_path = media_path
        self.device = device
        self.frames_decoded = 0
        self._tensor = self._decode_once(media_path, device=device)
        self.frames_decoded = 1

    @staticmethod
    def _decode_once(media_path: Path, *, device: torch.device) -> torch.Tensor:
        with av.open(str(media_path)) as container:
            streams = [s for s in container.streams if s.type == "video"]
            if not streams:
                raise TensorRenderError(f"{media_path} has no video stream to raster from")
            frame = next(iter(container.decode(streams[0])), None)
            if frame is None:
                raise TensorRenderError(f"{media_path} decoded no frame for a raster layer")
            array = frame.reformat(format="rgba").to_ndarray()  # [H, W, 4] uint8
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device)
        return tensor.permute(2, 0, 1).float().div_(255.0)

    @property
    def width(self) -> int:
        return int(self._tensor.shape[2])

    @property
    def height(self) -> int:
        return int(self._tensor.shape[1])

    def frame_at(self, source_time: Fraction) -> torch.Tensor:
        """Return the held raster; ``source_time`` is constant for raster layers."""

        return self._tensor

    def close(self) -> None:
        return None


class ClipDecoder:
    """PyAV decoder with directional ownership and a bounded reverse GOP cache.

    What it does (Pythonese)
    ------------------------
    ``frame_at(t)`` finds the last decoded frame whose presentation time is
    <= ``t``. Reverse schedules reuse frames retained from the most recently
    decoded GOP before seeking backward again. It then packs
    its planes (``pack_planes``), uploads them once and converts on the device
    with ``planes_to_rgb``.  ``packed_at(t)`` stops before the upload so a
    decode worker can hand the CPU planes to the GPU thread (``pipeline``).

    The colour policy (``SourceColor``) is resolved from the *first decoded
    frame's* tags -- the same per-frame ``colorspace`` / ``color_range`` swscale
    reads in the reference -- plus the codec context's transfer for the HDR
    reject, then pinned: a later frame with a different format or tags is a
    loud ``TensorRenderError`` (mid-stream tag changes are not a supported
    class either).

    Decoder-side downscale (``decode_size``, preview only)
    ------------------------------------------------------
    ``decode_size=(width, height)`` -- in the ENCODED orientation, chosen by
    ``decode_policy.resolve_decode_raster`` -- makes ``packed_at`` hand on the
    owning frame's planes downscaled by libswscale (``VideoFrame.reformat`` with
    the same pixel format and ``DECODE_SCALE_INTERPOLATION``) instead of the
    native planes.  Ownership, seeking, the reverse GOP cache and the colour
    policy are all resolved on the NATIVE frame first; only the planes that
    leave this class are smaller.  ``None`` (export) hands on the native planes.
    """

    def __init__(
        self,
        media_path: Path,
        *,
        device: torch.device,
        threads: int = 4,
        reverse_cache_bytes: int = 0,
        reverse_cache_frames: int = 128,
        decode_size: Optional[tuple[int, int]] = None,
    ) -> None:
        self.media_path = media_path
        self.device = device
        if decode_size is not None:
            width, height = decode_size
            if width < 2 or height < 2:
                raise TensorRenderError(f"{media_path}: decode size {width}x{height} is too small")
        self._decode_size = decode_size
        self._container = av.open(str(media_path))
        streams = [s for s in self._container.streams if s.type == "video"]
        if not streams:
            raise TensorRenderError(f"{media_path} has no video stream")
        self._stream = streams[0]
        self._stream.thread_type = "AUTO"
        self._stream.thread_count = threads
        self._time_base = Fraction(self._stream.time_base)
        start_time = self._stream.start_time or 0
        self._origin = Fraction(start_time) * self._time_base
        # Nominal container frame duration (seconds), used by the reverse cache
        # to decide which retained frame OWNS a requested instant. This is the
        # stream's average frame rate inverted -- e.g. 1/30 for CFR-30, or
        # 1001/30000 for NTSC 29.97. We derive it from the container rather than
        # from any FCPXML-declared cadence, because reverse lookups compare a
        # requested time against the frames' true container PTS keys, and the
        # ownership window is one *container* frame wide. Fail loudly (house
        # rule: no silent magic default) if the stream reports no usable rate.
        average_rate = self._stream.average_rate
        if average_rate is None or average_rate <= 0:
            raise TensorRenderError(
                f"{media_path}: video stream reports no usable average_rate "
                f"({average_rate!r}); cannot determine nominal frame duration"
            )
        self._frame_duration = 1 / Fraction(average_rate)
        self._iterator = self._container.decode(self._stream)
        self._current: Optional[av.VideoFrame] = None
        self._next: Optional[av.VideoFrame] = None
        self._sought = False
        self._color: Optional[SourceColor] = None
        if reverse_cache_bytes < 0 or reverse_cache_frames < 0:
            raise TensorRenderError("reverse decoder cache limits cannot be negative")
        self._reverse_cache_bytes_limit = reverse_cache_bytes
        self._reverse_cache_frames_limit = reverse_cache_frames
        self._reverse_cache_bytes = 0
        self._reverse_cache: "OrderedDict[Fraction, tuple[av.VideoFrame, int]]" = OrderedDict()
        self.frames_decoded = 0

    @property
    def width(self) -> int:
        return int(self._stream.codec_context.width)

    @property
    def height(self) -> int:
        return int(self._stream.codec_context.height)

    @property
    def color(self) -> Optional[SourceColor]:
        """The resolved colour-in policy (``None`` until the first frame is decoded)."""

        return self._color

    @property
    def decode_size(self) -> Optional[tuple[int, int]]:
        """The encoded-orientation raster ``packed_at`` hands on (``None`` = native)."""

        return self._decode_size

    def _scaled(self, frame: av.VideoFrame, color: SourceColor) -> av.VideoFrame:
        """Return ``frame`` downscaled to ``decode_size`` (same pixel format), or ``frame`` itself.

        Why this exists: the one place the preview's decoder-side downscale
        happens.  Never upscales (the policy caps at native and this refuses
        louder than trusting it); keeps the frame's own colour tags because
        ``reformat`` copies them; re-checks chroma-block parity on the smaller
        raster so ``pack_planes`` / ``planes_to_rgb`` see whole chroma blocks.
        """

        if self._decode_size is None:
            return frame
        width, height = self._decode_size
        if (frame.width, frame.height) == (width, height):
            return frame
        if width > frame.width or height > frame.height:
            raise TensorRenderError(
                f"{self.media_path}: decode size {width}x{height} exceeds the native raster "
                f"{frame.width}x{frame.height}"
            )
        if (color.chroma_shift_x and width % 2) or (color.chroma_shift_y and height % 2):
            # The policy rounds to even; an odd request here is a caller bug, not a media verdict.
            raise TensorRenderError(
                f"{self.media_path}: decode size {width}x{height} is not chroma-block aligned for {color.pixel_format}"
            )
        return frame.reformat(
            width=width,
            height=height,
            format=frame.format.name,
            interpolation=DECODE_SCALE_INTERPOLATION,
        )

    def _frame_time(self, frame: av.VideoFrame) -> Fraction:
        if frame.pts is None:
            raise TensorRenderError(f"{self.media_path}: decoded frame without pts")
        return Fraction(frame.pts) * self._time_base - self._origin

    def _decode_next(self) -> Optional[av.VideoFrame]:
        try:
            frame = next(self._iterator)
        except StopIteration:
            return None
        self.frames_decoded += 1
        self._remember(frame)
        return frame

    @staticmethod
    def _frame_bytes(frame: av.VideoFrame) -> int:
        """Approximate retained decoder memory from the actual plane buffers."""

        sizes = [int(getattr(plane, "buffer_size", 0)) for plane in frame.planes]
        if all(size > 0 for size in sizes):
            return sum(sizes)
        # Conservative fallback: over-count chroma planes as full-height. The
        # cache may evict early, but it cannot exceed its byte budget.
        return sum(int(plane.line_size) * frame.height for plane in frame.planes)

    def _remember(self, frame: av.VideoFrame) -> None:
        """Retain a decoded suffix under strict byte and frame limits.

        Main callers:
        - ``_decode_next`` for a reverse-capable layer.

        Why this exists:
        Seeking once per reverse output frame repeatedly decodes the same GOP.
        Retaining its recent suffix makes the common reverse walk linear while
        the two limits bound memory for 4K sources and unusually long GOPs.
        """

        if self._reverse_cache_bytes_limit == 0 or self._reverse_cache_frames_limit == 0:
            return
        key = self._frame_time(frame)
        size = self._frame_bytes(frame)
        previous = self._reverse_cache.pop(key, None)
        if previous is not None:
            self._reverse_cache_bytes -= previous[1]
        self._reverse_cache[key] = (frame, size)
        self._reverse_cache_bytes += size
        while self._reverse_cache and (
            len(self._reverse_cache) > self._reverse_cache_frames_limit
            or self._reverse_cache_bytes > self._reverse_cache_bytes_limit
        ):
            _, (_, evicted_size) = self._reverse_cache.popitem(last=False)
            self._reverse_cache_bytes -= evicted_size

    def _cached_frame_at(self, source_time: Fraction) -> Optional[av.VideoFrame]:
        """Return the retained frame that OWNS ``source_time``, or ``None`` on a miss.

        What it does (Pythonese)
        ------------------------
        Walk the retained frames, keep the one with the LARGEST container key
        (its true ``_frame_time``) that is still ``<= source_time``, and return
        it -- but only if the requested instant falls inside that frame's own
        one-frame window, i.e. ``source_time - key < frame_duration``. If the
        nearest earlier key is more than a frame away (or there is none),
        report a miss (``None``) so the caller seeks, exactly as before.

        Why the guard, and why NOT an exact-key match
        ----------------------------------------------
        A frame decoded at container time ``key`` owns the half-open instant
        window ``[key, key + frame_duration)``. Reverse requests arrive on the
        timeline's DECLARED frame grid (a clean multiple of the FCPXML
        ``frameDuration``), which for a 29.97fps (30000/1001) or VFR source
        never lands exactly on the container's true PTS keys (k*1001/30000).
        An exact-dict lookup therefore missed on EVERY reverse frame and
        silently re-decoded the whole GOP per output frame -- a ~16x cliff and
        a silent fallback. Matching by owning window is exact for constant-rate
        sources (delta 0 when the grid coincides) and conservative for VFR (it
        may occasionally miss and re-seek, but it never hands back a frame whose
        window does not contain the instant).

        The ``< frame_duration`` guard is also what preserves disjoint-GOP
        safety: after several backward seeks the cache can hold suffixes of two
        non-adjacent GOPs with the true owner's GOP never decoded between them.
        In that case the largest key ``<= source_time`` belongs to the earlier,
        wrong GOP and is more than one ``frame_duration`` away, so the guard
        fails and we correctly miss (seek) instead of returning a wrong frame.
        Do not drop this guard -- it is the whole safety argument.

        Main callers:
        - ``owning_frame`` on the reverse branch (request time before the
          current decoded frame).
        """

        # Bounded (~128 frame) cache, so a plain linear scan for the largest
        # owning key is fine and stays readable. The cache may hold disjoint
        # GOP suffixes, so we cannot assume global key order -- scan them all.
        best_frame: Optional[av.VideoFrame] = None
        best_key: Optional[Fraction] = None
        for key, (frame, _size) in self._reverse_cache.items():
            if key <= source_time and (best_key is None or key > best_key):
                best_key = key
                best_frame = frame
        if best_key is None:
            return None
        # Guarded owning-window check: only return the frame if the instant
        # falls within its one-frame window. Otherwise it belongs to a frame
        # whose GOP was never decoded -> real miss -> caller seeks.
        if source_time - best_key < self._frame_duration:
            return best_frame
        return None

    def _seek(self, source_time: Fraction) -> None:
        target = int((source_time + self._origin) / self._time_base)
        self._container.seek(target, stream=self._stream, backward=True, any_frame=False)
        self._iterator = self._container.decode(self._stream)
        self._current = None
        self._next = None
        self._sought = True

    def owning_frame(self, source_time: Fraction) -> av.VideoFrame:
        """Return the decoded ``av.VideoFrame`` that owns ``source_time`` (forward ownership)."""

        if self._current is not None and source_time < self._frame_time(self._current):
            cached = self._cached_frame_at(source_time)
            if cached is not None:
                return cached
            self._seek(source_time)
        if self._current is None:
            # First request (or just after a seek): land on the keyframe at or
            # before the instant instead of stepping from the file head --
            # this is the decoder-level ``-ss`` of the legacy path.
            if not self._sought:
                self._seek(source_time)
            self._current = self._decode_next()
            if self._current is None:
                raise TensorRenderError(f"{self.media_path}: no frames at {float(source_time):.4f}s")
            self._next = self._decode_next()
            if self._frame_time(self._current) > source_time:
                # Seek landed after the instant (should not happen with a
                # backward keyframe seek); fail loudly rather than drift.
                raise TensorRenderError(
                    f"{self.media_path}: first decoded frame {float(self._frame_time(self._current)):.4f}s "
                    f"is after requested {float(source_time):.4f}s"
                )
        # Advance while the lookahead frame still owns the instant.
        while self._next is not None and self._frame_time(self._next) <= source_time:
            self._current = self._next
            self._next = self._decode_next()
        return self._current

    def _resolve_color(self, frame: av.VideoFrame) -> SourceColor:
        """Pin the colour policy from the first frame's tags; verify every later frame against it."""

        matrix_tag = _AV_COLORSPACE_NAMES.get(int(frame.colorspace), f"enum{int(frame.colorspace)}")
        range_tag = _AV_COLOR_RANGE_NAMES.get(int(frame.color_range), f"enum{int(frame.color_range)}")
        if self._color is not None:
            if frame.format.name != self._color.pixel_format:
                raise TensorRenderError(
                    f"{self.media_path}: pixel format changed mid-stream ({self._color.pixel_format} -> {frame.format.name})"
                )
            return self._color
        codec_context = self._stream.codec_context
        transfer_tag = _AV_TRANSFER_NAMES.get(int(getattr(codec_context, "color_trc", 2)), "unknown")
        primaries_tag = _AV_PRIMARIES_NAMES.get(
            int(getattr(codec_context, "color_primaries", 2)), "unknown"
        )
        color = resolve_source_color(
            frame.format.name,
            matrix_tag=matrix_tag,
            range_tag=range_tag,
            transfer_tag=transfer_tag,
            primaries_tag=primaries_tag,
            subject=str(self.media_path),
        )
        _check_raster_parity(frame.width, frame.height, color, subject=str(self.media_path))
        self._color = color
        return color

    def packed_at(self, source_time: Fraction) -> SourceFrame | HDRFrame:
        """Return raw SDR planes or high-precision HDR RGB, with no device work."""

        frame = self.owning_frame(source_time)
        color = self._resolve_color(frame)
        if color.hdr_transfer is not None:
            if self._decode_size is not None:
                # ``decode_policy`` keeps HDR sources native (rule "hdr"); a
                # downscaled request here means a caller bypassed the policy.
                raise TensorRenderError(f"{self.media_path}: decoder-side downscale is not supported for HDR sources")
            # Match ``format=gbrpf32le`` before the shared LUT: FFmpeg's
            # libswscale graph uses bicubic chroma expansion and a full-range
            # floating RGB destination. PyAV defaults to bilinear and leaves
            # the destination range unspecified unless both are named.
            array = frame.reformat(
                format="gbrpf32le",
                dst_color_range="JPEG",
                interpolation="BICUBIC",
            ).to_ndarray()
            rgb = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
            return HDRFrame(rgb=rgb, transfer=color.hdr_transfer)
        return pack_planes(self._scaled(frame, color), color)

    def frame_at(self, source_time: Fraction) -> torch.Tensor:
        """Return the owning frame for ``source_time`` as RGB float [3,H,W] on the device."""

        packed = self.packed_at(source_time)
        if isinstance(packed, HDRFrame):
            return hdr_to_sdr(packed.rgb.to(self.device), packed.transfer)
        planes = packed.planes.to(self.device)
        return planes_to_rgb(planes, packed.layout, packed.color)

    def close(self) -> None:
        self._container.close()
