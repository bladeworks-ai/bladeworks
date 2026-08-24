"""Transient, low-resolution media samples for the Studio timeline.

Architecture map
================

authenticated Studio request
    -> validated file below ``<bundle>/Media``
    -> bounded PyAV video seeks and sequential audio peak scan
    -> small JPEG data URLs plus normalized audio bands
    -> browser-owned in-memory LRU

No sample is written to disk or retained by the server. The browser controls
request concurrency and cache size; this module independently caps every
request so a malformed client cannot ask for unbounded decode work.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from .media_library import MediaLibraryError


MAX_THUMBNAILS = 12
MAX_AUDIO_BANDS = 256
MAX_SAMPLE_DURATION = 300.0
MIN_THUMBNAIL_WIDTH = 48
MAX_THUMBNAIL_WIDTH = 160


@dataclass(frozen=True)
class MediaVisualRequest:
    start: float
    duration: float
    thumbnail_count: int
    thumbnail_width: int
    audio_bands: int


def sample_media(path: Path, request: MediaVisualRequest) -> dict[str, object]:
    """Decode a deliberately small visual summary and immediately discard it."""

    _validate_request(request)
    try:
        with av.open(str(path), mode="r") as container:
            has_video = bool(container.streams.video)
            has_audio = bool(container.streams.audio)
        thumbnails = _video_thumbnails(path, request) if has_video and request.thumbnail_count else []
        bands = _audio_bands(path, request) if has_audio and request.audio_bands else []
    except MediaLibraryError:
        raise
    except Exception as error:
        raise MediaLibraryError(
            "media_visual_decode_failed",
            f"Could not decode timeline visuals for {path.name}: {error}",
            status=422,
        ) from error
    return {
        "relativePath": path.name,
        "thumbnails": thumbnails,
        "audioBands": bands,
        "hasVideo": has_video,
        "hasAudio": has_audio,
    }


def _validate_request(request: MediaVisualRequest) -> None:
    if request.start < 0 or request.duration <= 0 or request.duration > MAX_SAMPLE_DURATION:
        raise MediaLibraryError(
            "invalid_request",
            f"start must be non-negative and duration must be between 0 and {MAX_SAMPLE_DURATION:g} seconds.",
            status=400,
        )
    if not 0 <= request.thumbnail_count <= MAX_THUMBNAILS:
        raise MediaLibraryError("invalid_request", f"thumbnailCount must be 0 through {MAX_THUMBNAILS}.", status=400)
    if not MIN_THUMBNAIL_WIDTH <= request.thumbnail_width <= MAX_THUMBNAIL_WIDTH:
        raise MediaLibraryError(
            "invalid_request",
            f"thumbnailWidth must be {MIN_THUMBNAIL_WIDTH} through {MAX_THUMBNAIL_WIDTH}.",
            status=400,
        )
    if not 0 <= request.audio_bands <= MAX_AUDIO_BANDS:
        raise MediaLibraryError("invalid_request", f"audioBands must be 0 through {MAX_AUDIO_BANDS}.", status=400)


def _video_thumbnails(path: Path, request: MediaVisualRequest) -> list[str]:
    targets = [
        request.start + request.duration * (index + 0.5) / request.thumbnail_count
        for index in range(request.thumbnail_count)
    ]
    output: list[str] = []
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # FFmpeg exposes still images through the image2 demuxer. Seeking that
        # one-frame stream raises EPERM on macOS, so decode it once and reuse
        # the same thumbnail across the requested filmstrip slots.
        if container.format.name == "image2":
            frame = next(container.decode(stream), None)
            if frame is None:
                return []
            thumbnail = _thumbnail_data_url(frame, request.thumbnail_width)
            return [thumbnail] * request.thumbnail_count
        if stream.start_time is not None:
            stream_origin = float(stream.start_time * stream.time_base)
        elif container.start_time is not None:
            stream_origin = float(container.start_time / av.time_base)
        else:
            stream_origin = 0.0
        for target in targets:
            # PyAV's container-level seek offset is relative to the media
            # timeline, while decoded frame times retain the stream origin.
            container.seek(int(target * av.time_base), backward=True, any_frame=False)
            chosen = None
            for frame in container.decode(stream):
                chosen = frame
                timestamp = frame.time
                if timestamp is None or timestamp - stream_origin >= target:
                    break
            if chosen is None:
                continue
            output.append(_thumbnail_data_url(chosen, request.thumbnail_width))
    return output


def _thumbnail_data_url(frame: av.VideoFrame, width: int) -> str:
    """Encode one decoded frame as the small JPEG consumed by Studio."""

    image = frame.to_image()
    height = max(27, round(width * image.height / max(1, image.width)))
    image.thumbnail((width, height))
    encoded = io.BytesIO()
    image.convert("RGB").save(encoded, format="JPEG", quality=38, optimize=False)
    return f"data:image/jpeg;base64,{base64.b64encode(encoded.getvalue()).decode('ascii')}"


def _audio_bands(path: Path, request: MediaVisualRequest) -> list[float]:
    peaks = np.zeros(request.audio_bands, dtype=np.float32)
    end = request.start + request.duration
    with av.open(str(path), mode="r") as container:
        stream = container.streams.audio[0]
        if stream.start_time is not None:
            stream_origin = float(stream.start_time * stream.time_base)
        elif container.start_time is not None:
            stream_origin = float(container.start_time / av.time_base)
        else:
            stream_origin = 0.0
        container.seek(int(request.start * av.time_base), backward=True, any_frame=False)
        cursor = request.start
        for frame in container.decode(stream):
            frame_start = float(frame.time) - stream_origin if frame.time is not None else cursor
            sample_rate = frame.sample_rate or stream.rate
            if not sample_rate:
                continue
            values = frame.to_ndarray()
            source_dtype = values.dtype
            if values.ndim == 1:
                values = values.reshape(1, -1)
            channel_count = max(1, len(frame.layout.channels))
            if not frame.format.is_planar:
                values = values.reshape(-1, channel_count).transpose()
            frame_samples = frame.samples
            frame_end = frame_start + frame_samples / sample_rate
            cursor = frame_end
            if frame_end < request.start:
                continue
            if frame_start >= end:
                break
            normalized = values.astype("float32", copy=False)
            if np.issubdtype(source_dtype, np.integer):
                limits = np.iinfo(source_dtype)
                if limits.min == 0:
                    midpoint = float(1 << (limits.bits - 1))
                    normalized = (normalized - midpoint) / midpoint
                else:
                    normalized /= float(max(abs(limits.min), abs(limits.max)))
            absolute = np.abs(normalized).max(axis=0)
            _accumulate_audio_peaks(
                peaks,
                absolute,
                frame_start=frame_start,
                sample_rate=float(sample_rate),
                request=request,
            )
    # Keep levels referenced to digital full scale. Normalizing each request by
    # its own loudest sample makes every clip reach 100% and destroys the level
    # differences the waveform is meant to communicate.
    return np.round(np.clip(peaks, 0.0, 1.0), decimals=4).tolist()


def _accumulate_audio_peaks(
    peaks: np.ndarray,
    absolute_samples: np.ndarray,
    *,
    frame_start: float,
    sample_rate: float,
    request: MediaVisualRequest,
) -> None:
    """Reduce one decoded frame into waveform bands without Python sample loops.

    ``numpy.maximum.at`` performs the many-to-one peak reduction in compiled
    code. Python work is therefore proportional to decoded audio frames, not
    sample count (14.4 million samples for five minutes of 48 kHz audio).
    """

    if absolute_samples.size == 0:
        return
    first = max(0, int(np.ceil((request.start - frame_start) * sample_rate)))
    last = min(
        absolute_samples.size,
        int(np.ceil((request.start + request.duration - frame_start) * sample_rate)),
    )
    if first >= last:
        return
    indexes = np.arange(first, last, dtype=np.float64)
    timestamps = frame_start + indexes / sample_rate
    bands = ((timestamps - request.start) / request.duration * request.audio_bands).astype(np.intp)
    np.clip(bands, 0, request.audio_bands - 1, out=bands)
    np.maximum.at(peaks, bands, absolute_samples[first:last])
