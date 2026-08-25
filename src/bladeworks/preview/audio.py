"""Incremental live-audio production for the standalone Bladeworks server.

Architecture map
================

    RenderDocument + compatibility report
        -> public audio-delivery resolver
        -> calibrated stock-FFmpeg filter graph
        -> seek trim at the requested preview time
        -> raw signed 16-bit PCM streamed from stdout
        -> small planar ``ComposedAudioFrame`` chunks

The producer never builds the complete preview timeline in memory. Closing or
cancelling a scan terminates its one FFmpeg child, so a new seek can begin
without waiting for the previous timeline to finish decoding.
"""

from __future__ import annotations

import selectors
import shutil
import subprocess
import tempfile
import threading
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from ..core.errors import RenderCapabilityError
from ..core.model import RenderDocument
from ..core.report import CompatibilityReport
from .contracts import ComposedAudioFrame
from ..tensor.audio_delivery import resolve_audio_delivery


def _seconds(value: Fraction) -> str:
    return f"{float(value):.12f}".rstrip("0").rstrip(".") or "0"


def _layout_channels(layout: str) -> int:
    channels = {"mono": 1, "stereo": 2}.get(layout)
    if channels is None:
        raise RenderCapabilityError(
            f"unsupported Bladeworks preview audio layout {layout!r}"
        )
    return channels


class FFmpegPreviewAudioProducer:
    """Stream seekable PCM chunks from the same graph used for final export."""

    def __init__(
        self,
        document: RenderDocument,
        *,
        report: CompatibilityReport,
        ffmpeg_path: str,
        ffprobe_path: str,
        chunk_milliseconds: int = 20,
        **_: object,
    ) -> None:
        self.document = document
        self.report = report
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.chunk_milliseconds = chunk_milliseconds
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._closed = False

    def frames(self, start_time: Fraction, *, is_cancelled) -> Iterator[ComposedAudioFrame]:
        """Yield small audio chunks until the timeline ends or the scan stops.

        Main callers:
        - The preview session starts this iterator for each play or seek scan.

        Why this exists: preview must return its first chunk promptly and remain
        cancellable. The former path decoded and concatenated every audio frame
        before yielding anything.
        """

        resolution = resolve_audio_delivery(
            self.document,
            ffprobe=self.ffprobe_path,
            report=self.report,
        )
        rate = resolution.output_sample_rate
        layout = resolution.output_layout
        channels = _layout_channels(layout)
        duration = resolution.output_duration
        if start_time >= duration or self._closed or is_cancelled():
            return

        chunk_samples = max(1, rate * self.chunk_milliseconds // 1000)
        chunk_bytes = chunk_samples * channels * 2
        with tempfile.TemporaryDirectory(prefix="bladeworks-preview-audio-") as directory:
            script_path = Path(directory) / "audio.ffmpeg.txt"
            argv: list[str] = [self.ffmpeg_path, "-hide_banner", "-loglevel", "error"]
            execution = resolution.execution
            if execution is None:
                lines = [
                    f"anullsrc=r={rate}:cl={layout}:d={_seconds(duration)}[aout]"
                ]
                output_label = "aout"
            else:
                for input_path in execution.inputs:
                    argv.extend(("-i", str(input_path)))
                lines = execution.filter_complex.split(";")
                output_label = execution.output_label
            lines.append(
                f"[{output_label}]atrim=start={_seconds(start_time)},"
                "asetpts=PTS-STARTPTS[previewa]"
            )
            script_path.write_text(";\n".join(lines) + "\n", encoding="utf-8")
            argv.extend(
                (
                    "-filter_complex_script",
                    str(script_path),
                    "-map",
                    "[previewa]",
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(rate),
                    "-ac",
                    str(channels),
                    "pipe:1",
                )
            )
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with self._lock:
                if self._closed:
                    process.terminate()
                self._process = process
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            emitted_samples = 0
            pending = bytearray()
            consumer_stopped = False
            try:
                while not self._closed and not is_cancelled():
                    if not selector.select(timeout=0.1):
                        if process.poll() is not None:
                            break
                        continue
                    data = process.stdout.read(chunk_bytes - len(pending))
                    if not data:
                        break
                    pending.extend(data)
                    if len(pending) < chunk_bytes:
                        continue
                    interleaved = np.frombuffer(bytes(pending), dtype="<i2").reshape(
                        chunk_samples, channels
                    )
                    yield ComposedAudioFrame(
                        time=start_time + Fraction(emitted_samples, rate),
                        sample_rate=rate,
                        layout=layout,
                        samples=np.ascontiguousarray(interleaved.T),
                    )
                    emitted_samples += chunk_samples
                    pending.clear()
                if pending and not self._closed and not is_cancelled():
                    whole_samples = len(pending) // (channels * 2)
                    if whole_samples:
                        interleaved = np.frombuffer(
                            bytes(pending[: whole_samples * channels * 2]), dtype="<i2"
                        ).reshape(whole_samples, channels)
                        yield ComposedAudioFrame(
                            time=start_time + Fraction(emitted_samples, rate),
                            sample_rate=rate,
                            layout=layout,
                            samples=np.ascontiguousarray(interleaved.T),
                        )
            except GeneratorExit:
                consumer_stopped = True
                raise
            finally:
                selector.close()
                stopping = consumer_stopped or self._closed or is_cancelled()
                if process.poll() is None and stopping:
                    process.terminate()
                try:
                    return_code = process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    return_code = process.wait(timeout=2.0)
                with self._lock:
                    if self._process is process:
                        self._process = None
                if return_code != 0 and not stopping and process.stderr is not None:
                    message = process.stderr.read().decode("utf-8", errors="replace")[-2000:]
                    raise RuntimeError(
                        f"FFmpeg live audio failed with exit {return_code}: {message}"
                    )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()


class FFmpegPreviewAudioFactory:
    """Resolve the two external tools required by incremental preview audio."""

    def __init__(
        self,
        *,
        report_for: Callable[[RenderDocument], CompatibilityReport],
        ffmpeg_path: str | None = None,
        ffprobe_path: str | None = None,
        **_: object,
    ) -> None:
        resolved_ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
        resolved_ffprobe = ffprobe_path or shutil.which("ffprobe")
        if resolved_ffmpeg is None:
            raise RenderCapabilityError("ffmpeg is required for Bladeworks preview audio")
        if resolved_ffprobe is None:
            raise RenderCapabilityError("ffprobe is required for Bladeworks preview audio")
        self.report_for = report_for
        self.ffmpeg_path = resolved_ffmpeg
        self.ffprobe_path = resolved_ffprobe

    def create(self, document: RenderDocument) -> FFmpegPreviewAudioProducer:
        return FFmpegPreviewAudioProducer(
            document,
            report=self.report_for(document),
            ffmpeg_path=self.ffmpeg_path,
            ffprobe_path=self.ffprobe_path,
        )
