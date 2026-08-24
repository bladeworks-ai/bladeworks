"""Framework-neutral contracts for the local Bladeworks preview API.

Architecture map
================

    HTTP JSON
        -> request dataclasses and validated public enum values
        -> PreviewService / PreviewSession
        -> result dataclasses
        -> HTTP JSON or typed SSE records

No class in this file knows about FastAPI, WebRTC implementations, GPU state,
or project storage. This makes command behavior testable with ordinary fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Protocol

import numpy as np

from ..core.model import RenderDocument
from ..tensor.renderer import ComposedFrame, FrameWindow
from ..tensor.resolution import OutputResolution, ResolutionProfile

if TYPE_CHECKING:
    from .source import LoadedProject


class PreviewAPIError(RuntimeError):
    """A stable public API failure with an HTTP mapping."""

    def __init__(self, code: str, message: str, *, status: int, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.retryable = retryable

    def body(self) -> dict[str, object]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            }
        }


@dataclass(frozen=True)
class SessionQuality:
    resolution: ResolutionProfile


@dataclass(frozen=True)
class SessionDescription:
    type: str
    sdp: str


@dataclass(frozen=True)
class PreviewEvent:
    id: int
    event: str
    data: Mapping[str, object]


@dataclass(frozen=True)
class ComposedAudioFrame:
    """Planar signed 16-bit PCM with an exact project-time origin."""

    time: Fraction
    sample_rate: int
    layout: str
    samples: np.ndarray


class SourceDocumentProvider(Protocol):
    """Resolve one Project inside an exact complete-library content hash."""

    def require_current(self, source_version: str, project_ref: str) -> LoadedProject: ...


class FrameProducer(Protocol):
    """One warm, resolution-specific Bladeworks composition lifetime."""

    @property
    def frame_duration(self) -> Fraction: ...

    @property
    def frame_count(self) -> int: ...

    def seek(self, frame: int, *, is_cancelled) -> ComposedFrame: ...

    def frames(self, window: FrameWindow, *, is_cancelled) -> Iterator[ComposedFrame]: ...

    def close(self) -> None: ...


class FrameProducerFactory(Protocol):
    def create(self, document: RenderDocument, resolution: OutputResolution) -> FrameProducer: ...


class PreviewAudioProducer(Protocol):
    def frames(self, start_time: Fraction, *, is_cancelled) -> Iterator[ComposedAudioFrame]: ...

    def close(self) -> None: ...


class PreviewAudioProducerFactory(Protocol):
    def create(self, document: RenderDocument) -> PreviewAudioProducer: ...


class PreviewMediaSink(Protocol):
    """A negotiated live-media connection owned by one preview session."""

    def write_video(self, frame: ComposedFrame) -> None: ...

    def write_still(self, frame: ComposedFrame) -> None: ...

    def write_audio(self, frame: ComposedAudioFrame) -> None: ...

    def flush(self) -> None:
        """Discard media that was enqueued but not yet delivered to the client.

        Called at each playback control boundary (pause, seek, quality change,
        source swap, Project switch) after the old generation's scan has been
        stopped, so the viewer never receives frames or audio from the
        superseded generation ahead of the new still or playback stream.
        """
        ...

    def close(self) -> None: ...


class PreviewMediaFactory(Protocol):
    """Negotiate an SDP offer and return the live media sink plus SDP answer."""

    def negotiate(self, offer: SessionDescription) -> tuple[SessionDescription, PreviewMediaSink]: ...


def fraction_from_seconds(value: Any, *, field: str = "time") -> Fraction:
    """Parse public JSON seconds without inheriting binary-float drift."""

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise PreviewAPIError("time_out_of_range", f"{field} must be numeric seconds", status=400)
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise PreviewAPIError("time_out_of_range", f"{field} must be numeric seconds", status=400) from error
    return result


def seconds_json(value: Fraction) -> float:
    return float(value)
