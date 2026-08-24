"""PyAV live-audio producer for the standalone Bladeworks server."""

from __future__ import annotations

import shutil
from fractions import Fraction
from typing import Callable, Iterator

import av
import numpy as np

from ..core.errors import RenderCapabilityError
from ..core.model import RenderDocument
from ..core.report import CompatibilityReport
from .contracts import ComposedAudioFrame
from ..tensor import audio_pyav
from ..tensor.audio_delivery import audio_delivery_layout, resolve_audio_delivery


class FFmpegPreviewAudioProducer:
    """Expose the tensor audio graph as seekable planar PCM preview chunks."""

    def __init__(
        self,
        document: RenderDocument,
        *,
        report: CompatibilityReport,
        ffprobe_path: str,
        chunk_milliseconds: int = 20,
        **_: object,
    ) -> None:
        self.document = document
        self.report = report
        self.ffprobe_path = ffprobe_path
        self.chunk_milliseconds = chunk_milliseconds
        self._closed = False

    def _samples(self) -> tuple[np.ndarray, int, str]:
        resolution = resolve_audio_delivery(
            self.document,
            ffprobe=self.ffprobe_path,
            report=self.report,
        )
        rate = resolution.output_sample_rate
        layout = audio_delivery_layout(self.document)
        if resolution.mode == "render" and resolution.execution is not None:
            frames = audio_pyav.render_execution_frames(resolution.execution)
        else:
            frames = audio_pyav.render_silence_frames(
                sample_rate=rate,
                ffmpeg_layout=layout,
                duration=resolution.output_duration,
            )
        resampler = av.AudioResampler(format="s16p", layout=layout, rate=rate)
        chunks: list[np.ndarray] = []
        for frame in frames:
            for converted in resampler.resample(frame):
                chunks.append(np.ascontiguousarray(converted.to_ndarray()))
        for converted in resampler.resample(None):
            chunks.append(np.ascontiguousarray(converted.to_ndarray()))
        channels = 1 if layout == "mono" else 2
        samples = np.concatenate(chunks, axis=1) if chunks else np.empty((channels, 0), dtype=np.int16)
        return samples, rate, layout

    def frames(self, start_time: Fraction, *, is_cancelled) -> Iterator[ComposedAudioFrame]:
        samples, rate, layout = self._samples()
        start = max(0, int(start_time * rate))
        chunk = max(1, rate * self.chunk_milliseconds // 1000)
        for offset in range(start, samples.shape[1], chunk):
            if self._closed or is_cancelled():
                return
            yield ComposedAudioFrame(
                time=Fraction(offset, rate),
                sample_rate=rate,
                layout=layout,
                samples=np.ascontiguousarray(samples[:, offset : offset + chunk]),
            )

    def close(self) -> None:
        self._closed = True


class FFmpegPreviewAudioFactory:
    """Compatibility-named factory backed entirely by tensor/PyAV audio."""

    def __init__(
        self,
        *,
        report_for: Callable[[RenderDocument], CompatibilityReport],
        ffprobe_path: str | None = None,
        **_: object,
    ) -> None:
        resolved = ffprobe_path or shutil.which("ffprobe")
        if resolved is None:
            raise RenderCapabilityError("ffprobe is required for Bladeworks preview audio")
        self.report_for = report_for
        self.ffprobe_path = resolved

    def create(self, document: RenderDocument) -> FFmpegPreviewAudioProducer:
        return FFmpegPreviewAudioProducer(
            document,
            report=self.report_for(document),
            ffprobe_path=self.ffprobe_path,
        )
