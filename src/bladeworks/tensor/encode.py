"""Encoder exit: GPU canvas -> planes, PyAV encoder, optional encoder thread (A7 / A7-lite).

Architecture map
----------------
    canvas [4,H,W] premultiplied linear (renderer)
        -> VideoEncoder.canvas_to_planes  : the exit for the encoder's pixel policy (GPU):
             opaque  canvas_to_yuv420p    : encode() to code space -> 8-bit RGB codes ->
                                            BT.709 limited-range Y/Cb/Cr, 4:2:0 chroma
                                            subsampling that mirrors swscale's default
                                            (pair-average horizontally, 4-tap triangle
                                            vertically) -> ONE flat uint8 tensor on device
             alpha    canvas_to_yuva444p10: unpremultiply -> straight RGB code + alpha ->
                                            BT.709 limited-range 10-bit Y/Cb/Cr 4:4:4 +
                                            full-range 10-bit straight alpha -> ONE flat
                                            int16 tensor on device
        -> .cpu()                         : the single device->host copy per frame (renderer)
        -> VideoEncoder.write_planes      : av.VideoFrame -> encode -> mux            (serial)
           EncoderThread.write_planes     : bounded queue -> worker thread does the same (pipelined)

Pixel policies (PYTORCH_MVP_PLAN.md: "two pixel policies (opaque / alpha)")
--------------------------------------------------------------------------
``opaque`` (default): Rec.709 limited-range yuv420p, libx264 (preset/crf
configurable) or an alternative encoder name (``h264_videotoolbox``), constant
frame rate, colour tags written -- unchanged from the legacy CPU delivery.

``alpha`` (A7-lite): the root canvas is transparent (renderer), the exit
unpremultiplies at export only and writes **straight** alpha into ProRes 4444
(``prores_ks`` profile 4, ``yuva444p10le``) in a ``.mov``, Rec.709 limited
tags asserted, alpha plane full-range 10-bit.  There is no swscale between the
renderer and the encoder: this module owns the only format conversion.  The
CPU path has no alpha-carrying delivery (``oracle_rgba`` is a Vulkan rawvideo
oracle without audio), so the exit math here is exact float BT.709, rounded --
verified by ``test_tensor_color_io.py`` (planes vs a float64 oracle, and a
PyAV decode round-trip: alpha exact, RGB within 1 code).

Why the conversion moved to the GPU
-----------------------------------
The former exit handed 8-bit RGB to swscale (``VideoFrame.reformat``) on the
render thread: ~3 ms @720p / 5 ms @1080p of CPU work per frame in the one
thread that must never stall, plus a 3 B/px download instead of 1.5 B/px.
``canvas_to_yuv420p`` does the same matrix and subsampling as elementwise
tensor ops and downloads half the bytes.  Both render paths (serial and
pipelined) call this one function, so their outputs are identical by
construction; the residual drift vs swscale is fixed-point coefficient
rounding only (measured on random + gradient frames: |diff| <= 1 code,
Y differs on ~0.04 % of pixels, Cb/Cr on ~0.3 % of samples).

Main callers: ``renderer.render_document`` (both loops), the executor's
``--backend tensor`` branch (``_render_tensor_video`` picks the policy), the
bench and tests.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal, Optional

import av
import numpy as np
import torch
import torch.nn.functional as F

from .color import encode, unpremultiply
from .errors import TensorRenderError

PixelPolicy = Literal["opaque", "alpha"]

# BT.709 luma weights (Rec. ITU-R BT.709-6) and the limited-range scale/offsets.
_KR, _KB = 0.2126, 0.0722
_KG = 1.0 - _KR - _KB
# swscale's rgb24ToY_c adds ``1 << (RGB2YUV_SHIFT - 7)`` before its shift, a
# +1/128-code luma bias; matching it moves the round-boundary mismatches from
# ~0.9 % of pixels to ~0.04 % (measured against libswscale 9.1 / FFmpeg 8).
_SWSCALE_LUMA_BIAS = 1.0 / 128.0


# Row-major 3x3: [Y', Cb, Cr] = M @ [R, G, B] on 0..255 codes (Y' scaled to 219/255,
# Cb/Cr to 224/255), then + [16 + bias, 128, 128].
_RGB_TO_YCBCR = (
    (_KR * 219.0 / 255.0, _KG * 219.0 / 255.0, _KB * 219.0 / 255.0),
    (-_KR * 0.5 / (1.0 - _KB) * 224.0 / 255.0, -_KG * 0.5 / (1.0 - _KB) * 224.0 / 255.0, 0.5 * 224.0 / 255.0),
    (0.5 * 224.0 / 255.0, -_KG * 0.5 / (1.0 - _KR) * 224.0 / 255.0, -_KB * 0.5 / (1.0 - _KR) * 224.0 / 255.0),
)
_YCBCR_OFFSET = (16.0 + _SWSCALE_LUMA_BIAS, 128.0, 128.0)
# swscale's 4:2:0 chroma taps at equal size: pair-average horizontally, [1/8, 3/8, 3/8, 1/8]
# vertically over rows 2j-1..2j+2 -> one depthwise conv2d (2 x 4 taps, stride 2).
_CHROMA_TAPS = tuple(0.5 * weight for weight in (0.125, 0.375, 0.375, 0.125))


class _ExitConstants:
    """Per-device cache of the exit's small constant tensors (matrix, offset, chroma kernel)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def get(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (str(like.device), like.device.index or 0)
        cached = self._cache.get(key)
        if cached is None:
            matrix = torch.tensor(_RGB_TO_YCBCR, dtype=torch.float32, device=like.device)
            offset = torch.tensor(_YCBCR_OFFSET, dtype=torch.float32, device=like.device).view(3, 1, 1)
            taps = torch.tensor(_CHROMA_TAPS, dtype=torch.float32, device=like.device).view(1, 1, 4, 1)
            kernel = taps.expand(2, 1, 4, 2).contiguous()   # [out=2, in/groups=1, kh=4, kw=2]
            cached = self._cache[key] = (matrix, offset, kernel)
        return cached


_EXIT_CONSTANTS = _ExitConstants()


def rgb_codes_to_yuv420p(rgb_codes: torch.Tensor) -> torch.Tensor:
    """8-bit RGB codes ``[3, H, W]`` (float 0..255, already rounded) -> flat yuv420p uint8.

    Pythonese: compute full-resolution Y'CbCr from the codes with the BT.709
    matrix (one matmul + one bias add); Y' lands on 16..235 and Cb/Cr on
    16..240 around 128; subsample the chroma the way swscale's default
    (bilinear, non-accurate-rounding) path does for rgb24 -> yuv420p at equal
    size -- average each horizontal pixel pair (its ``rgb24ToUV_half``), then
    filter vertically with the 4-tap triangle [1/8, 3/8, 3/8, 1/8] over rows
    2j-1..2j+2 (edge rows replicated) -- as one replicate pad + one stride-2
    depthwise ``conv2d``; round half up, clamp, and lay out Y, U, V planes
    back to back in one uint8 buffer (the memory layout
    ``av.VideoFrame.from_ndarray(..., format="yuv420p")`` expects when
    reshaped to ``[H * 3 / 2, W]``).

    Why matmul + conv2d rather than per-channel arithmetic: eager MPS is
    launch-bound (~0.1-0.2 ms per kernel at 720p), so the exit's cost is its
    kernel count -- ~10 launches here vs ~35 written out per channel
    (measured 3.2 ms/frame -> see stats).

    Main callers: ``canvas_to_yuv420p``; tests measuring the drift vs swscale.
    """

    channels, height, width = rgb_codes.shape
    if channels != 3 or height % 2 or width % 2:
        raise TensorRenderError(f"yuv420p exit needs [3, even H, even W]; got {tuple(rgb_codes.shape)}")
    matrix, offset, kernel = _EXIT_CONSTANTS.get(rgb_codes)
    ycbcr = torch.matmul(matrix, rgb_codes.reshape(3, height * width)).view(3, height, width) + offset
    padded = F.pad(ycbcr[1:].unsqueeze(0), (0, 0, 1, 2), mode="replicate")   # rows -1 .. H+1
    chroma = F.conv2d(padded, kernel, stride=2, groups=2).squeeze(0)          # [2, H/2, W/2]
    # Round half up via truncation of the (non-negative, clamped) value + 0.5.
    y_u8 = (ycbcr[0] + 0.5).clamp_(0.0, 255.0).to(torch.uint8)
    uv_u8 = (chroma + 0.5).clamp_(0.0, 255.0).to(torch.uint8)
    return torch.cat((y_u8.reshape(-1), uv_u8.reshape(-1)))


def canvas_to_yuv420p(canvas: torch.Tensor) -> torch.Tensor:
    """Premultiplied linear canvas ``[4, H, W]`` -> flat yuv420p uint8 tensor (on device).

    The root canvas is opaque, so the RGB channels are the final light: encode
    to code space, quantize to 8-bit codes exactly as the former rgb24 exit
    did (round, clamp), then convert.  Quantizing first keeps the yuv exit a
    pure replacement of swscale on the same 8-bit input.
    """

    codes = (encode(canvas[:3]) * 255.0).round_().clamp_(0.0, 255.0)
    return rgb_codes_to_yuv420p(codes)


def yuv420p_ndarray(flat_planes_cpu: torch.Tensor, *, height: int, width: int) -> np.ndarray:
    """View a downloaded flat plane buffer as the ``[H*3/2, W]`` array PyAV wants."""

    return flat_planes_cpu.numpy().reshape(height * 3 // 2, width)


# ---- alpha exit: straight RGBA -> yuva444p10le (BT.709 limited 10-bit + full-range alpha) ----

# Row-major 3x3 on straight RGB in 0..1: [Y', Cb, Cr] = M @ [R, G, B] * 1023-scale, then
# + [64, 512, 512] (10-bit limited: Y' 64..940, Cb/Cr 64..960 around 512).
_RGB_TO_YCBCR_10 = (
    (_KR * 876.0, _KG * 876.0, _KB * 876.0),
    (-_KR * 0.5 / (1.0 - _KB) * 896.0, -_KG * 0.5 / (1.0 - _KB) * 896.0, 0.5 * 896.0),
    (0.5 * 896.0, -_KG * 0.5 / (1.0 - _KR) * 896.0, -_KB * 0.5 / (1.0 - _KR) * 896.0),
)
_YCBCR_OFFSET_10 = (64.0, 512.0, 512.0)


class _AlphaExitConstants:
    """Per-device cache of the alpha exit's matrix / offset."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(like.device), like.device.index or 0)
        cached = self._cache.get(key)
        if cached is None:
            matrix = torch.tensor(_RGB_TO_YCBCR_10, dtype=torch.float32, device=like.device)
            offset = torch.tensor(_YCBCR_OFFSET_10, dtype=torch.float32, device=like.device).view(3, 1, 1)
            cached = self._cache[key] = (matrix, offset)
        return cached


_ALPHA_EXIT_CONSTANTS = _AlphaExitConstants()


def canvas_to_yuva444p10(canvas: torch.Tensor) -> torch.Tensor:
    """Premultiplied linear canvas ``[4, H, W]`` -> flat ``yuva444p10le`` int16 tensor (on device).

    Pythonese: unpremultiply (colour 0 where alpha is 0), encode the straight
    colour to code space, take the straight alpha as is; Y'CbCr = BT.709
    limited-range 10-bit (one matmul + bias), alpha = full-range 10-bit
    (``alpha * 1023``); round half up, clamp to 0..1023, and lay the four planes
    Y, Cb, Cr, A back to back (``[4*H*W]``, int16 because torch has no
    arithmetic uint16 -- the codes fit).

    Why unpremultiply here and nowhere else: the whole tensor pipeline is
    premultiplied (over, dissolves, opacity); ProRes 4444 carries straight
    alpha, so the division happens exactly once, at the exit.  Alpha is
    quantised straight (no association), which is what QuickTime / Final Cut
    expect from ``ap4h``.

    Main callers: ``VideoEncoder.canvas_to_planes`` for ``pixel_policy="alpha"``.
    """

    channels, height, width = canvas.shape
    if channels != 4:
        raise TensorRenderError(f"alpha exit needs a [4, H, W] canvas; got {tuple(canvas.shape)}")
    matrix, offset = _ALPHA_EXIT_CONSTANTS.get(canvas)
    straight = unpremultiply(canvas)
    rgb = encode(straight[:3]).reshape(3, height * width)
    ycbcr = torch.matmul(matrix, rgb).view(3, height, width) + offset
    alpha = straight[3:4].clamp(0.0, 1.0) * 1023.0
    planes = torch.cat((ycbcr, alpha), dim=0)
    return (planes + 0.5).clamp_(0.0, 1023.0).to(torch.int16).reshape(-1)


def yuva444p10_ndarray(flat_planes_cpu: torch.Tensor, *, height: int, width: int) -> np.ndarray:
    """View a downloaded flat alpha-exit buffer as the ``[4, H, W]`` uint16 array PyAV wants."""

    return flat_planes_cpu.numpy().view(np.uint16).reshape(4, height, width)


def _planar_frame_from_planes(planes: np.ndarray, pixel_format: str) -> av.VideoFrame:
    """Build an ``av.VideoFrame`` from a ``[planes, H, W]`` array for formats PyAV cannot ``from_ndarray``.

    Copies each plane into the frame's own buffer honouring the plane stride
    (``line_size`` may exceed ``W * bytes_per_sample``), so any planar format
    with equally sized planes works (used for ``yuva444p10le``).
    """

    count, height, width = planes.shape
    frame = av.VideoFrame(width, height, pixel_format)
    if len(frame.planes) != count:
        raise TensorRenderError(f"{pixel_format} has {len(frame.planes)} planes; got {count}")
    for index, plane in enumerate(frame.planes):
        row_bytes = width * planes.dtype.itemsize
        if plane.line_size == row_bytes:
            plane.update(np.ascontiguousarray(planes[index]).tobytes())
            continue
        padded = np.zeros((height, plane.line_size), dtype=np.uint8)
        padded[:, :row_bytes] = np.ascontiguousarray(planes[index]).view(np.uint8).reshape(height, row_bytes)
        plane.update(padded.tobytes())
    return frame


@dataclass
class EncoderAudio:
    """The finished delivery audio to mux beside the video, in ONE container.

    ``frames`` are float-planar (``fltp``) ``av.AudioFrame`` at ``sample_rate`` /
    ``ffmpeg_layout`` -- exactly what ``tensor.audio_pyav`` yields and what the
    AAC encoder takes directly.  This is how the tensor renderer replaces the old
    second ffmpeg mux process: the audio graph runs in-process and its frames are
    interleaved into the encoder's own container.
    """

    frames: list  # list[av.AudioFrame]
    sample_rate: int
    ffmpeg_layout: str
    codec: str = "aac"


class VideoEncoder:
    """Synchronous PyAV encoder: ``canvas_to_planes`` (GPU exit) + ``write_planes`` per frame, ``close`` flushes.

    ``pixel_policy`` selects the exit and the container/codec contract:

    * ``"opaque"``: ``codec`` (``libx264`` default, or ``h264_videotoolbox`` /
      any encoder taking ``yuv420p``) into the given path (``.mp4`` by the
      executor), Rec.709 limited yuv420p.
    * ``"alpha"``: ``codec`` must be ``"prores_ks"`` (loud otherwise) into a
      ``.mov``, ProRes 4444 (profile 4) ``yuva444p10le`` straight alpha,
      Rec.709 limited tags; ``preset`` / ``crf`` do not apply.

    ``write_yuv420p`` remains for callers that already hold the ``[H*3/2, W]``
    uint8 array (the current renderer exit); ``write_planes`` takes the flat
    host tensor of either exit.
    """

    def __init__(
        self,
        output_path: Path,
        *,
        width: int,
        height: int,
        frame_duration: Fraction,
        codec: str = "libx264",
        preset: str = "medium",
        crf: int = 18,
        threads: int = 0,
        bit_rate: int = 0,
        pixel_policy: PixelPolicy = "opaque",
        audio: Optional["EncoderAudio"] = None,
    ) -> None:
        if pixel_policy not in ("opaque", "alpha"):
            raise TensorRenderError(f"unknown pixel policy {pixel_policy!r} (opaque | alpha)")
        if pixel_policy == "alpha" and codec != "prores_ks":
            raise TensorRenderError(
                f"the alpha exit writes ProRes 4444 through prores_ks; got codec={codec!r}"
            )
        if pixel_policy == "alpha" and output_path.suffix.lower() != ".mov":
            raise TensorRenderError(f"the alpha exit needs a .mov container; got {output_path.name}")
        self.output_path = output_path
        self.width = width
        self.height = height
        self.pixel_policy: PixelPolicy = pixel_policy
        self.pixel_format = "yuva444p10le" if pixel_policy == "alpha" else "yuv420p"
        self.codec = codec
        threads = threads or max(2, min(4, (os.cpu_count() or 2) // 2))
        self._container = av.open(str(output_path), mode="w", options={"movflags": "+faststart+write_colr"})
        stream = self._container.add_stream(codec, rate=Fraction(1) / frame_duration)
        stream.width = width
        stream.height = height
        stream.pix_fmt = self.pixel_format
        stream.time_base = frame_duration
        options = {}
        if pixel_policy == "alpha":
            # ProRes 4444 (profile 4) with 16-bit alpha coding; the encoder's
            # own rate control (bits_per_mb per profile) -- no crf/preset.
            options = {"profile": "4", "alpha_bits": "16", "vendor": "apl0"}
        elif codec == "libx264":
            options = {
                "preset": preset,
                "crf": str(crf),
                "x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709",
            }
        else:
            # Hardware / other encoders have no crf: target a bitrate in the
            # same class as x264 crf 18 (~10 Mb/s at 720p, ~20 Mb/s at 1080p).
            stream.bit_rate = bit_rate or int(10_000_000 * (width * height) / (720 * 1280))
            if codec.endswith("_videotoolbox"):
                options = {"allow_sw": "0", "realtime": "0"}
        stream.codec_context.options = options
        codec_context = stream.codec_context
        # libavcodec defaults to ONE encoder thread unless the caller asks
        # (the ffmpeg CLI sets auto).  x264 frame-threading is what makes
        # ``medium`` affordable; use every core, like the legacy path does.
        codec_context.thread_type = "FRAME"
        codec_context.thread_count = threads
        for name, value in (
            ("color_range", 1),        # MPEG / limited
            ("colorspace", 1),         # BT.709
            ("color_primaries", 1),    # BT.709
            ("color_trc", 1),          # BT.709
        ):
            try:
                setattr(codec_context, name, value)
            except (AttributeError, TypeError, ValueError):
                pass
        self._stream = stream
        self._frame_duration = frame_duration
        self.frames_written = 0
        self.busy_seconds = 0.0
        self._closed = False

        # ---- audio (single-container, interleaved by dts) ---------------------
        # The audio is small, so it is pre-encoded to a packet list and the SMALL
        # side is buffered: during the streaming video mux we flush audio packets
        # up to each video frame's presentation time, keeping the muxer's
        # interleave window bounded (never buffer the large video side).  All of
        # this runs on whichever thread owns the container (the encoder worker for
        # the pipelined path), because PyAV containers are not thread-safe.
        self._audio = audio
        self._audio_stream = None
        self._audio_packets: list = []
        self._audio_index = 0
        self._audio_primed = False
        if audio is not None:
            audio_stream = self._container.add_stream(audio.codec, rate=audio.sample_rate)
            audio_stream.codec_context.layout = audio.ffmpeg_layout
            audio_stream.codec_context.format = "fltp"
            self._audio_stream = audio_stream

    # ---- exit (GPU) -------------------------------------------------------------------

    def canvas_to_planes(self, canvas: torch.Tensor) -> torch.Tensor:
        """The exit for this encoder's pixel policy: canvas -> ONE flat plane tensor on device.

        Main callers: ``renderer.render_document`` (once per frame, before the
        single device->host download).
        """

        if self.pixel_policy == "alpha":
            return canvas_to_yuva444p10(canvas)
        return canvas_to_yuv420p(canvas)

    # ---- encode (CPU) -----------------------------------------------------------------

    def write_planes(self, host_planes: torch.Tensor) -> None:
        """Encode one downloaded flat plane tensor of this encoder's exit (``canvas_to_planes``)."""

        if self.pixel_policy == "alpha":
            self.write_yuva444p10(yuva444p10_ndarray(host_planes, height=self.height, width=self.width))
        else:
            self.write_yuv420p(yuv420p_ndarray(host_planes, height=self.height, width=self.width))

    def write_yuv420p(self, planes: np.ndarray) -> None:
        """Encode one ``[H*3/2, W]`` uint8 yuv420p frame (Y rows, then U, then V rows)."""

        if self.pixel_policy != "opaque":
            raise TensorRenderError("write_yuv420p on an alpha-policy encoder")
        expected = (self.height * 3 // 2, self.width)
        if planes.dtype != np.uint8 or planes.shape != expected:
            raise TensorRenderError(f"yuv420p frame must be uint8 {expected}; got {planes.dtype} {planes.shape}")
        self._encode(av.VideoFrame.from_ndarray(planes, format="yuv420p"))

    def write_yuva444p10(self, planes: np.ndarray) -> None:
        """Encode one ``[4, H, W]`` uint16 yuva444p10le frame (Y, Cb, Cr, straight A)."""

        if self.pixel_policy != "alpha":
            raise TensorRenderError("write_yuva444p10 on an opaque-policy encoder")
        expected = (4, self.height, self.width)
        if planes.dtype != np.uint16 or planes.shape != expected:
            raise TensorRenderError(f"yuva444p10 frame must be uint16 {expected}; got {planes.dtype} {planes.shape}")
        self._encode(_planar_frame_from_planes(planes, "yuva444p10le"))

    def _prime_audio(self) -> None:
        """Encode the whole delivery audio to a dts-ordered packet list once.

        Runs on the container-owning thread the first time a frame is encoded (or
        at close for a zero-frame render).  ``encode()`` does not touch the
        muxer, so this only fills ``self._audio_packets``; muxing happens
        interleaved in ``_flush_audio_until``.
        """

        self._audio_primed = True
        if self._audio_stream is None or self._audio is None:
            return
        sample_index = 0
        rate = self._audio.sample_rate
        for audio_frame in self._audio.frames:
            audio_frame.pts = sample_index
            audio_frame.time_base = Fraction(1, rate)
            sample_index += audio_frame.samples
            self._audio_packets.extend(self._audio_stream.encode(audio_frame))
        self._audio_packets.extend(self._audio_stream.encode(None))

    def _flush_audio_until(self, video_seconds: Optional[float]) -> None:
        """Mux buffered audio packets up to ``video_seconds`` (all if ``None``)."""

        while self._audio_index < len(self._audio_packets):
            packet = self._audio_packets[self._audio_index]
            if (
                video_seconds is not None
                and packet.pts is not None
                and float(packet.pts * packet.time_base) > video_seconds
            ):
                break
            self._container.mux(packet)
            self._audio_index += 1

    def _encode(self, frame: av.VideoFrame) -> None:
        started = time.perf_counter()
        frame.pts = self.frames_written
        frame.time_base = self._frame_duration
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frames_written += 1
        if self._audio_stream is not None:
            if not self._audio_primed:
                self._prime_audio()
            # Keep audio muxed just behind the video's presentation time so the
            # interleaver never holds the large video side waiting for audio.
            self._flush_audio_until(float(self.frames_written * self._frame_duration))
        self.busy_seconds += time.perf_counter() - started

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        started = time.perf_counter()
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        if self._audio_stream is not None:
            if not self._audio_primed:
                self._prime_audio()
            # Video is fully written; flush the remaining (small) audio tail.
            self._flush_audio_until(None)
        self._container.close()
        self.busy_seconds += time.perf_counter() - started


@dataclass
class EncoderThreadStats:
    frames: int = 0
    wait_seconds: float = 0.0        # producer time blocked because the queue was full
    busy_seconds: float = 0.0        # worker time inside encode + mux (incl. flush)
    max_queue_depth: int = 0         # deepest backlog seen at submit time
    queue_depth_sum: int = 0         # for the mean backlog


class EncoderThread:
    """Bounded queue in front of ``VideoEncoder`` so x264 never blocks the GPU thread.

    Pythonese: ``write_planes`` (flat host tensor of either exit) or
    ``write_yuv420p`` (the ``[H*3/2, W]`` array) puts the frame on a queue of at
    most ``queue_depth`` frames (blocking, and re-raising the worker's exception
    if it died); one worker thread pops frames in order and calls the matching
    ``VideoEncoder`` writer; ``close`` sends the end sentinel, joins the worker,
    and re-raises whatever it failed with, so a broken encoder fails the render
    loudly instead of hanging or truncating silently.

    Why a thread and not a process: PyAV releases the GIL inside
    ``avcodec_send_frame`` and x264 runs its own frame threads, so the encoder
    stage overlaps with GPU work without pickling frames.
    """

    _DONE = object()

    def __init__(self, encoder: VideoEncoder, *, queue_depth: int = 6) -> None:
        if queue_depth < 1:
            raise TensorRenderError("encoder queue depth must be >= 1")
        self.encoder = encoder
        self.stats = EncoderThreadStats()
        self._queue: queue.Queue = queue.Queue(maxsize=queue_depth)
        self._failure: Optional[BaseException] = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="tensor-encoder", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is self._DONE:
                    return
                if isinstance(item, torch.Tensor):
                    self.encoder.write_planes(item)
                else:
                    self.encoder.write_yuv420p(item)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the producer thread
            self._failure = exc
            # Drain so a blocked producer wakes up and sees the failure.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return

    def _check_failure(self) -> None:
        if self._failure is not None:
            raise TensorRenderError(f"encoder thread failed: {self._failure!r}") from self._failure

    def canvas_to_planes(self, canvas: torch.Tensor) -> torch.Tensor:
        """The wrapped encoder's exit (GPU); see ``VideoEncoder.canvas_to_planes``."""

        return self.encoder.canvas_to_planes(canvas)

    def write_planes(self, host_planes: torch.Tensor) -> None:
        self._submit(host_planes)

    def write_yuv420p(self, planes: np.ndarray) -> None:
        self._submit(planes)

    def _submit(self, item: object) -> None:
        depth = self._queue.qsize()
        self.stats.max_queue_depth = max(self.stats.max_queue_depth, depth)
        self.stats.queue_depth_sum += depth
        started = time.perf_counter()
        while True:
            self._check_failure()
            if not self._thread.is_alive():
                raise TensorRenderError("encoder thread exited before the render finished")
            try:
                self._queue.put(item, timeout=0.5)
                break
            except queue.Full:
                continue
        self.stats.wait_seconds += time.perf_counter() - started
        self.stats.frames += 1

    def close(self) -> None:
        """Flush: wait for the queue to drain, close the encoder, re-raise worker failures."""

        if self._closed:
            return
        self._closed = True
        started = time.perf_counter()
        if self._thread.is_alive():
            while True:
                try:
                    self._queue.put(self._DONE, timeout=0.5)
                    break
                except queue.Full:
                    if self._failure is not None or not self._thread.is_alive():
                        break
            self._thread.join()
        self.stats.wait_seconds += time.perf_counter() - started
        try:
            self.encoder.close()
        except Exception:
            if self._failure is None:
                raise
            # The worker's failure is the root cause; report that one.
        finally:
            self.stats.busy_seconds = self.encoder.busy_seconds
        self._check_failure()
