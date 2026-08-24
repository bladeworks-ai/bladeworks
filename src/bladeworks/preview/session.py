"""One preview session: source-version state, cancellation, seek, and paced scan.

Architecture map
================

        command thread                      scan worker
        sync/seek/pause/play                sequential Bladeworks frames
        -> generation counter               -> paced media writes
        -> cancel prior work                 -> SSE time/buffering
                 both share one renderer lock and media sink
                  one renderer lock + one media sink

Important rules
---------------
* A generation number guards publication, not just computation. Old work may
  finish, but it can never write a stale frame after a newer command.
* Bladeworks is single-caller within a session. The renderer lock protects its
  decoder and temporal caches.
* Preview resolution is fixed by the user's Better Quality / Performance
  selection. Slow graphs buffer instead of silently changing image quality.
"""

from __future__ import annotations

import math
import threading
import time
from fractions import Fraction
from typing import Optional

from ..core.model import RenderDocument
from ..tensor.errors import TensorRenderError
from ..tensor.renderer import FrameWindow
from ..tensor.resolution import OutputResolution, ResolutionProfile, resolve_output_resolution
from .contracts import (
    FrameProducerFactory,
    PreviewAPIError,
    PreviewAudioProducerFactory,
    PreviewMediaSink,
    SessionQuality,
    seconds_json,
)
from .events import EventStream


class PreviewSession:
    def __init__(
        self,
        *,
        session_id: str,
        source_version: str,
        project_ref: str,
        document: RenderDocument,
        quality: SessionQuality,
        producer_factory: FrameProducerFactory,
        media_sink: PreviewMediaSink,
        audio_factory: PreviewAudioProducerFactory | None,
        playhead: Fraction,
    ) -> None:
        self.session_id = session_id
        self.source_version = source_version
        self.project_ref = project_ref
        self.document = document
        self.quality = quality
        self.producer_factory = producer_factory
        self.media_sink = media_sink
        self.audio_factory = audio_factory
        self.events = EventStream()
        self._state_lock = threading.RLock()
        self._renderer_lock = threading.Lock()
        self._generation = 0
        self._closed = False
        self._playing = False
        self._scan_thread: Optional[threading.Thread] = None
        self._scan_stop = threading.Event()
        self._buffering_reported = False
        self.selected_profile = quality.resolution
        self.output_resolution = self._resolution(self.selected_profile)
        self.current_frame = self._resolve_frame(playhead)
        self.producer = producer_factory.create(document, self.output_resolution)
        self._publish(
            "ready",
            {
                "resolution": self.selected_profile.value,
                "width": self.output_resolution.width,
                "height": self.output_resolution.height,
            },
        )

    def _publish(self, event: str, data: dict[str, object]) -> None:
        """Attach the pinned library/Project identity to every SSE event."""

        self.events.publish(
            event,
            {
                "sourceVersion": self.source_version,
                "projectRef": self.project_ref,
                **data,
            },
        )

    @property
    def playing(self) -> bool:
        with self._state_lock:
            return self._playing

    @property
    def current_time(self) -> Fraction:
        with self._state_lock:
            return self.current_frame * self.document.frame_duration

    def _resolution(self, profile: ResolutionProfile) -> OutputResolution:
        return resolve_output_resolution(self.document.width, self.document.height, profile)

    def _resolve_frame(self, requested: Fraction) -> int:
        return self._resolve_frame_for_document(self.document, requested)

    @staticmethod
    def _resolve_frame_for_document(document: RenderDocument, requested: Fraction) -> int:
        """Resolve a public time without consulting mutable session state.

        Main callers:
        - session creation through ``_resolve_frame``;
        - ``sync`` before it stops or replaces the current source version.

        Why this exists:
        A bad playhead for a new source version must fail before any producer
        or session state changes. That keeps replacement transactional.
        """

        if document.duration == 0:
            if requested != 0:
                raise PreviewAPIError(
                    "time_out_of_range",
                    "time 0 is the only valid playhead on an empty timeline",
                    status=400,
                )
            return 0
        if requested < 0 or requested >= document.duration:
            raise PreviewAPIError(
                "time_out_of_range",
                f"time {float(requested)} is outside [0, {float(document.duration)})",
                status=400,
            )
        frames = requested / document.frame_duration
        frame = (2 * frames.numerator + frames.denominator) // (2 * frames.denominator)
        return min(frame, document.frame_count - 1)

    def _superseded(self, generation: int) -> bool:
        with self._state_lock:
            return self._closed or self._generation != generation

    def _scan_cancelled(self, generation: int) -> bool:
        return self._superseded(generation) or self._scan_stop.is_set()

    def _stop_scan(self) -> None:
        with self._state_lock:
            thread = self._scan_thread
            self._scan_stop.set()
            self._generation += 1
            self._playing = False
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._state_lock:
            if self._scan_thread is thread:
                self._scan_thread = None
        # The scan worker is now guaranteed stopped, so nothing else is writing
        # to the sink. Drop whatever the old generation already enqueued before
        # the caller (seek/play/sync/pause) publishes the new still or playback
        # stream, so the client never sees stale frames or audio from the
        # superseded generation. Every control boundary routes through here.
        self.media_sink.flush()

    def sync(
        self,
        *,
        source_version: str,
        project_ref: str,
        document: RenderDocument,
        playhead: Fraction,
        quality: SessionQuality | None = None,
    ) -> dict[str, object]:
        """Atomically replace the compiled source after old work is stopped."""

        new_frame = self._resolve_frame_for_document(document, playhead)
        selected_quality = quality or self.quality
        new_resolution = resolve_output_resolution(document.width, document.height, selected_quality.resolution)
        self._stop_scan()
        with self._renderer_lock:
            with self._state_lock:
                if self._closed:
                    raise PreviewAPIError("preview_not_found", "Preview session is closed.", status=404)
                self._generation += 1
            new_producer = self.producer_factory.create(document, new_resolution)
            old_producer = self.producer
            with self._state_lock:
                self.source_version = source_version
                self.project_ref = project_ref
                self.document = document
                self.quality = selected_quality
                self.selected_profile = selected_quality.resolution
                self.output_resolution = new_resolution
                self.producer = new_producer
                self.current_frame = new_frame
            old_producer.close()
        return self.state_payload()

    def seek(self, *, requested_time: Fraction, request_id: str) -> dict[str, object]:
        self._stop_scan()
        with self._state_lock:
            if self._closed:
                raise PreviewAPIError("preview_not_found", "Preview session is closed.", status=404)
            self._generation += 1
            generation = self._generation
            frame_number = self._resolve_frame(requested_time)
        with self._renderer_lock:
            try:
                frame = self.producer.seek(
                    frame_number,
                    is_cancelled=lambda: self._superseded(generation),
                )
            except TensorRenderError as error:
                if self._superseded(generation):
                    raise PreviewAPIError(
                        "seek_superseded",
                        f"Seek {request_id} was superseded by a newer command.",
                        status=409,
                        retryable=True,
                    ) from error
                raise
            with self._state_lock:
                if self._superseded(generation):
                    raise PreviewAPIError(
                        "seek_superseded",
                        f"Seek {request_id} was superseded by a newer command.",
                        status=409,
                        retryable=True,
                    )
                self.media_sink.write_still(frame)
                self.current_frame = frame.frame
        self._publish("time", {"time": seconds_json(frame.time), "frame": frame.frame})
        return {
            "sourceVersion": self.source_version,
            "projectRef": self.project_ref,
            "requestId": request_id,
            "requestedTime": seconds_json(requested_time),
            "actualTime": seconds_json(frame.time),
            "frame": frame.frame,
            "resolution": self.selected_profile.value,
            "width": frame.width,
            "height": frame.height,
        }

    def play(self, *, requested_time: Optional[Fraction]) -> dict[str, object]:
        self._stop_scan()
        with self._state_lock:
            if self._closed:
                raise PreviewAPIError("preview_not_found", "Preview session is closed.", status=404)
            if requested_time is not None:
                self.current_frame = self._resolve_frame(requested_time)
            start_frame = self.current_frame
            self._scan_stop = threading.Event()
            self._generation += 1
            generation = self._generation
            self._playing = True
            self._buffering_reported = False
            thread = threading.Thread(
                target=self._scan_loop,
                args=(start_frame, generation),
                name=f"tensor-preview:{self.session_id}",
                daemon=True,
            )
            self._scan_thread = thread
            self._publish("playing", {"playing": True})
            thread.start()
        return {
            "sourceVersion": self.source_version,
            "projectRef": self.project_ref,
            "playing": True,
            "startTime": seconds_json(start_frame * self.document.frame_duration),
            "startFrame": start_frame,
            "resolution": self.selected_profile.value,
        }

    def _scan_loop(self, start_frame: int, generation: int) -> None:
        wall_origin = time.monotonic()
        clock_lock = threading.Lock()
        emitted = 0
        next_frame = start_frame
        time_event_stride = max(
            1,
            math.ceil(float(1 / self.document.frame_duration) / 30.0),
        )
        audio_producer = self.audio_factory.create(self.document) if self.audio_factory is not None else None
        audio_thread: Optional[threading.Thread] = None

        def playback_origin() -> float:
            with clock_lock:
                return wall_origin

        def reset_playback_origin(value: float) -> None:
            nonlocal wall_origin
            with clock_lock:
                wall_origin = value

        def wait_for_audio_slot(frame_time: Fraction) -> bool:
            """Pace PCM on the video clock and cap its lead during render lag.

            FFmpeg can produce PCM much faster than realtime. Without this
            gate aiortc would receive future audio immediately while video is
            deliberately wall-clock paced. The 300 ms lead permits a small
            media buffer but makes audio stall when composition falls behind.
            """

            start_time = start_frame * self.document.frame_duration
            while not self._scan_cancelled(generation):
                target_wall = playback_origin() + float(frame_time - start_time)
                with self._state_lock:
                    published_video_time = self.current_frame * self.document.frame_duration
                wall_delay = target_wall - time.monotonic()
                within_video_buffer = frame_time <= published_video_time + Fraction(3, 10)
                if wall_delay <= 0 and within_video_buffer:
                    return True
                wait_seconds = min(0.02, max(0.001, wall_delay))
                if self._scan_stop.wait(wait_seconds):
                    return False
            return False

        def pump_audio() -> None:
            assert audio_producer is not None
            try:
                start_time = start_frame * self.document.frame_duration
                for audio_frame in audio_producer.frames(
                    start_time,
                    is_cancelled=lambda: self._scan_cancelled(generation),
                ):
                    if self._scan_cancelled(generation):
                        return
                    if not wait_for_audio_slot(audio_frame.time):
                        return
                    self.media_sink.write_audio(audio_frame)
            except Exception as error:  # noqa: BLE001 - live failure is an SSE state
                if not self._scan_cancelled(generation):
                    self._publish(
                        "error",
                        {"code": "preview_failed", "message": str(error), "retryable": False},
                    )
                    self._scan_stop.set()

        if audio_producer is not None:
            audio_thread = threading.Thread(
                target=pump_audio,
                name=f"tensor-preview-audio:{self.session_id}",
                daemon=True,
            )
            audio_thread.start()
        try:
            while next_frame < self.document.frame_count and not self._scan_cancelled(generation):
                with self._renderer_lock:
                    stream = self.producer.frames(
                        FrameWindow(next_frame, self.document.frame_count),
                        is_cancelled=lambda: self._scan_cancelled(generation),
                    )
                    for frame in stream:
                        if self._scan_cancelled(generation):
                            return
                        target_wall = playback_origin() + float(emitted * self.document.frame_duration)
                        delay = target_wall - time.monotonic()
                        if delay > 0:
                            time.sleep(delay)
                        if self._scan_cancelled(generation):
                            return
                        self.media_sink.write_video(frame)
                        with self._state_lock:
                            self.current_frame = frame.frame
                        if emitted % time_event_stride == 0:
                            self._publish(
                                "time",
                                {"time": seconds_json(frame.time), "frame": frame.frame},
                            )
                        emitted += 1
                        next_frame = frame.frame + 1
                        lag = time.monotonic() - target_wall
                        if lag >= 1.0:
                            if not self._buffering_reported:
                                self._buffering_reported = True
                                elapsed = max(0.000001, time.monotonic() - playback_origin())
                                render_rate = float(emitted * self.document.frame_duration) / elapsed
                                self._publish(
                                    "buffering",
                                    {
                                        "bufferedSeconds": 0.0,
                                        "renderRate": round(render_rate, 3),
                                    },
                                )
                    break
            if not self._scan_cancelled(generation) and next_frame >= self.document.frame_count:
                self._publish("ended", {"time": seconds_json(self.document.duration)})
        except TensorRenderError:
            if not self._scan_cancelled(generation):
                self._publish(
                    "error",
                    {"code": "preview_failed", "message": "Bladeworks scan failed.", "retryable": False},
                )
        except Exception as error:  # noqa: BLE001 - session error must reach SSE
            if not self._scan_cancelled(generation):
                self._publish(
                    "error",
                    {"code": "preview_failed", "message": str(error), "retryable": False},
                )
        finally:
            if audio_producer is not None:
                audio_producer.close()
            if audio_thread is not None and audio_thread is not threading.current_thread():
                audio_thread.join(timeout=3.0)
            with self._state_lock:
                if self._generation == generation:
                    self._playing = False
                    self._scan_thread = None
                    self._publish("playing", {"playing": False})

    def pause(self) -> dict[str, object]:
        was_playing = self.playing
        self._stop_scan()
        if was_playing:
            self._publish("playing", {"playing": False})
        return {
            "sourceVersion": self.source_version,
            "projectRef": self.project_ref,
            "playing": False,
            "time": seconds_json(self.current_time),
            "frame": self.current_frame,
        }

    def state_payload(self) -> dict[str, object]:
        return {
            "sourceVersion": self.source_version,
            "projectRef": self.project_ref,
            "playhead": seconds_json(self.current_time),
            "selectedResolution": self.selected_profile.value,
            "width": self.output_resolution.width,
            "height": self.output_resolution.height,
        }

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        # Wake an open SSE response before waiting on renderer, decoder, or
        # WebRTC cleanup. This lets Uvicorn finish the client connection while
        # the remaining session resources close in parallel during SIGINT.
        self.events.close()
        self._stop_scan()
        with self._renderer_lock:
            self.producer.close()
            self.media_sink.close()
