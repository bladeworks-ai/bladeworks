"""Framework-neutral preview orchestration and thin HTTP contract tests."""

from __future__ import annotations

import asyncio
import struct
import threading
import time
import shutil
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from bladeworks.core.model import MissingMediaReference, RenderDocument
from bladeworks.core.compiler import compile_fcpxml
from bladeworks.preview.contracts import PreviewAPIError, SessionDescription
from bladeworks.preview.routes import create_app
from bladeworks.preview.audio import FFmpegPreviewAudioProducer
from bladeworks.preview.render_jobs import RenderJobService
from bladeworks.preview.rawframe import RawFrameMediaFactory, RawFrameMediaSink
from bladeworks.preview.provider import RegisteredSourceProvider
from bladeworks.preview.producer import TensorFrameProducerFactory
from bladeworks.preview.service import PreviewService, validate_quality
from bladeworks.preview.contracts import ComposedAudioFrame
from bladeworks.preview.webrtc import AiortcMediaFactory, TensorAudioTrack, TensorVideoTrack
from bladeworks.tensor.errors import TensorRenderError
from bladeworks.core.report import CompatibilityReport
from bladeworks.tensor.renderer import ComposedFrame, FrameWindow
from bladeworks.tensor.resolution import ResolutionProfile, resolve_output_resolution


PROJECT_REF = "library[1]/event[1]/project[1]"
MANUAL_QA = Path(__file__).parents[1] / "src" / "bladeworks" / "examples" / "manual_qa"


def _document(*, revision_name: str = "test") -> RenderDocument:
    return RenderDocument(
        schema_version=1,
        source_sha256="sha",
        source_path=Path("/tmp/test.fcpxml"),
        project_name=revision_name,
        width=1920,
        height=1080,
        frame_duration=Fraction(1, 30),
        duration=Fraction(1),
        tc_start=Fraction(0),
        clips=(),
        transitions=(),
        asset_bindings=(),
        font_bindings=(),
    )


class FakeDocuments:
    def __init__(self) -> None:
        self.values = {"sha256:v1": _document()}

    def require_current(self, source_version: str, project_ref: str):
        assert project_ref == PROJECT_REF
        try:
            document = self.values[source_version]
        except KeyError as error:
            raise PreviewAPIError("source_version_conflict", "Source version does not exist.", status=409) from error
        return SimpleNamespace(
            version=source_version,
            project_ref=project_ref,
            document=document,
            report=CompatibilityReport(project_name=document.project_name),
        )


class MultiProjectDocuments(FakeDocuments):
    SECOND_PROJECT_REF = "library[1]/event[2]/project[1]"

    def require_current(self, source_version: str, project_ref: str):
        if source_version != "sha256:v1":
            raise PreviewAPIError(
                "source_version_conflict",
                "Source version does not exist.",
                status=409,
            )
        document = (
            _document()
            if project_ref == PROJECT_REF
            else _document(revision_name="second-project")
            if project_ref == self.SECOND_PROJECT_REF
            else None
        )
        if document is None:
            raise PreviewAPIError("project_not_found", "Project does not exist.", status=404)
        return SimpleNamespace(
            version=source_version,
            project_ref=project_ref,
            document=document,
            report=CompatibilityReport(project_name=document.project_name),
        )


class FakeProducer:
    def __init__(
        self,
        document: RenderDocument,
        resolution,
        *,
        block_first_seek: bool = False,
        frame_delay: float = 0.0,
    ) -> None:
        self.document = document
        self.resolution = resolution
        self.closed = False
        self.block_first_seek = block_first_seek
        self.seek_started = threading.Event()
        self.frame_delay = frame_delay

    @property
    def frame_duration(self) -> Fraction:
        return self.document.frame_duration

    @property
    def frame_count(self) -> int:
        return self.document.frame_count

    def _frame(self, frame: int) -> ComposedFrame:
        return ComposedFrame(
            frame=frame,
            time=frame * self.frame_duration,
            duration=self.frame_duration,
            width=self.resolution.width,
            height=self.resolution.height,
            yuv420p=np.zeros((self.resolution.height * 3 // 2, self.resolution.width), dtype=np.uint8),
        )

    def seek(self, frame: int, *, is_cancelled) -> ComposedFrame:
        if self.block_first_seek:
            self.block_first_seek = False
            self.seek_started.set()
            while not is_cancelled():
                time.sleep(0.001)
            raise TensorRenderError("cancelled")
        return self._frame(frame)

    def frames(self, window: FrameWindow, *, is_cancelled):
        for frame in range(window.start_frame, window.end_frame):
            if is_cancelled():
                raise TensorRenderError("cancelled")
            if self.frame_delay:
                time.sleep(self.frame_delay)
            yield self._frame(frame)

    def close(self) -> None:
        self.closed = True


class FakeProducers:
    def __init__(self, *, block_first_seek: bool = False, frame_delay: float = 0.0) -> None:
        self.block_first_seek = block_first_seek
        self.frame_delay = frame_delay
        self.created: list[FakeProducer] = []

    def create(self, document, resolution) -> FakeProducer:
        producer = FakeProducer(
            document,
            resolution,
            block_first_seek=self.block_first_seek,
            frame_delay=self.frame_delay,
        )
        self.block_first_seek = False
        self.created.append(producer)
        return producer


class RecordingSink:
    def __init__(self) -> None:
        self.frames: list[ComposedFrame] = []
        self.audio_frames: list[ComposedAudioFrame] = []
        self.audio_write_times: list[float] = []
        self.flush_count = 0
        self.closed = False

    def write_video(self, frame: ComposedFrame) -> None:
        self.frames.append(frame)

    def write_still(self, frame: ComposedFrame) -> None:
        self.frames.append(frame)

    def write_audio(self, frame: ComposedAudioFrame) -> None:
        self.audio_frames.append(frame)
        self.audio_write_times.append(time.monotonic())

    def flush(self) -> None:
        # The real sinks drop unsent queued media here; this double only records
        # that a control boundary asked for a flush so tests can assert it.
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


class FakeMedia:
    def __init__(self) -> None:
        self.sinks: list[RecordingSink] = []

    def negotiate(self, offer: SessionDescription):
        assert offer.type == "offer"
        sink = RecordingSink()
        self.sinks.append(sink)
        return SessionDescription(type="answer", sdp="answer-sdp"), sink


class BlockingCloseSink(RecordingSink):
    """Hold media cleanup so tests can observe earlier SSE shutdown."""

    def __init__(self) -> None:
        super().__init__()
        self.close_started = threading.Event()
        self.release_close = threading.Event()

    def close(self) -> None:
        self.close_started.set()
        self.release_close.wait(timeout=2)
        super().close()


class BlockingCloseMedia:
    def __init__(self) -> None:
        self.sink = BlockingCloseSink()

    def negotiate(self, offer: SessionDescription):
        assert offer.type == "offer"
        return SessionDescription(type="answer", sdp="answer-sdp"), self.sink


class FakeAudioProducer:
    def frames(self, start_time, *, is_cancelled):
        for index in range(5):
            if is_cancelled():
                return
            yield ComposedAudioFrame(
                time=start_time + Fraction(index, 50),
                sample_rate=48_000,
                layout="stereo",
                samples=np.zeros((2, 960), dtype=np.int16),
            )

    def close(self) -> None:
        pass


class FakeAudioFactory:
    def create(self, document) -> FakeAudioProducer:
        return FakeAudioProducer()


class FakeExportRunner:
    def run(
        self,
        document,
        *,
        output_path,
        output_resolution,
        output_profile,
        video_only,
        progress,
        is_cancelled,
    ) -> None:
        assert output_resolution.profile.value in {"1080p", "720p", "540p", "480p"}
        assert output_profile in {"delivery", "delivery_alpha"}
        assert isinstance(video_only, bool)
        for completed in range(1, document.frame_count + 1):
            if is_cancelled():
                return
            progress(completed, document.frame_count)
        output_path.write_bytes(b"fake-mp4")


class BlockingExportRunner:
    def __init__(self) -> None:
        self.started = threading.Event()

    def run(
        self,
        document,
        *,
        output_path,
        output_resolution,
        output_profile,
        video_only,
        progress,
        is_cancelled,
    ) -> None:
        self.started.set()
        while not is_cancelled():
            time.sleep(0.001)


class CompletedExportRunner:
    """Publish an artifact, then expose the narrow post-export cancel race."""

    def __init__(self) -> None:
        self.artifact_ready = threading.Event()
        self.return_allowed = threading.Event()

    def run(self, document, *, output_path, progress, **_kwargs) -> None:
        output_path.write_bytes(b"complete")
        progress(document.frame_count, document.frame_count)
        self.artifact_ready.set()
        self.return_allowed.wait(timeout=2)


def _service(
    *,
    block_first_seek: bool = False,
    frame_delay: float = 0.0,
) -> tuple[PreviewService, FakeProducers, FakeMedia]:
    producers = FakeProducers(
        block_first_seek=block_first_seek,
        frame_delay=frame_delay,
    )
    media = FakeMedia()
    return PreviewService(documents=FakeDocuments(), producers=producers, media=media), producers, media


def test_quality_defaults_and_rejects_automatic_fallbacks() -> None:
    quality = validate_quality(None, None)
    assert quality.resolution.value == "720p"
    assert validate_quality("540p", []).resolution.value == "540p"
    assert validate_quality("480p", []).resolution.value == "480p"
    with pytest.raises(PreviewAPIError, match="Automatic preview fallback"):
        validate_quality("540p", ["720p"])


def test_registered_provider_requires_exact_source_version_and_copies_report() -> None:
    provider = RegisteredSourceProvider()
    document = _document()
    report = CompatibilityReport(project_name="test")
    provider.register("sha256:v7", SimpleNamespace(render=document, report=report))

    assert provider.require_current("sha256:v7", PROJECT_REF).document is document
    first = provider.report_for(document)
    second = provider.report_for(document)
    assert first is not second and first.project_name == "test"
    with pytest.raises(PreviewAPIError) as missing:
        provider.require_current("sha256:v8", PROJECT_REF)
    assert missing.value.code == "source_version_conflict"


def test_webrtc_track_accepts_composed_yuv_and_uses_monotonic_transport_time() -> None:
    async def exercise() -> None:
        track = TensorVideoTrack(queue_depth=2)
        source = ComposedFrame(
            frame=300,
            time=Fraction(10),
            duration=Fraction(1, 30),
            width=16,
            height=8,
            yuv420p=np.zeros((12, 16), dtype=np.uint8),
        )
        await track.frames.put(source)
        await track.frames.put(source)
        first = await track.recv()
        second = await track.recv()
        assert first.format.name == "yuv420p"
        assert first.pts == 0 and second.pts == 3000
        assert first.time_base == Fraction(1, 90_000)
        await track.close_queue()

    asyncio.run(exercise())


def test_webrtc_audio_track_accepts_planar_pcm() -> None:
    async def exercise() -> None:
        track = TensorAudioTrack(queue_depth=1)
        source = ComposedAudioFrame(
            time=Fraction(5),
            sample_rate=48_000,
            layout="stereo",
            samples=np.zeros((2, 960), dtype=np.int16),
        )
        await track.frames.put(source)
        frame = await track.recv()
        assert frame.format.name == "s16"
        assert frame.layout.name == "stereo"
        assert frame.samples == 960 and frame.pts == 0
        assert frame.time_base == Fraction(1, 48_000)
        await track.close_queue()

    asyncio.run(exercise())


def test_live_audio_uses_project_origin_and_twenty_millisecond_pcm_chunks() -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("needs ffmpeg/ffprobe")
    producer = FFmpegPreviewAudioProducer(
        _document(),
        report=CompatibilityReport(),
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
    )
    frames = list(producer.frames(Fraction(9, 10), is_cancelled=lambda: False))
    producer.close()

    assert frames
    assert frames[0].time == Fraction(9, 10)
    assert frames[0].sample_rate == 48_000
    assert frames[0].layout == "stereo"
    assert frames[0].samples.shape == (2, 960)
    assert np.count_nonzero(frames[0].samples) == 0


def test_live_audio_executes_audible_timeline_graph_from_nonzero_time(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("needs ffmpeg/ffprobe")
    from tests._tensor_backend_helpers import _project

    compiled = compile_fcpxml(_project(tmp_path, ffmpeg))
    producer = FFmpegPreviewAudioProducer(
        compiled.render,
        report=compiled.report,
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
    )
    frames = list(producer.frames(Fraction(1, 2), is_cancelled=lambda: False))
    producer.close()

    assert frames[0].time == Fraction(1, 2)
    assert frames[0].samples.shape == (2, 960)
    assert any(np.count_nonzero(frame.samples) for frame in frames)


def test_live_preview_prepares_basic_title_raster() -> None:
    """The preview path must prepare title pixels just like batch export."""

    compiled = compile_fcpxml(
        MANUAL_QA / "studio_showcase.fcpxmld",
        project="qa-launch-main",
    )
    document = compiled.render
    resolution = resolve_output_resolution(
        document.width,
        document.height,
        ResolutionProfile.P480,
    )
    producer = TensorFrameProducerFactory(device="cpu", decoder_threads=1).create(
        document,
        resolution,
    )
    temporary = producer._runtime_raster_directory
    assert temporary is not None
    temporary_path = Path(temporary.name)
    assert list(temporary_path.glob("*-text.png"))

    frame = producer.seek(15, is_cancelled=lambda: False)
    assert frame.width == 640 and frame.height == 360
    producer.close()
    assert not temporary_path.exists()


def test_seek_resolves_canonical_frame_and_publishes_to_media() -> None:
    service, _, media = _service()
    session, answer = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer-sdp"),
        quality=validate_quality(None, None),
    )
    result = session.seek(requested_time=Fraction(51, 100), request_id="seek-1")

    assert answer.type == "answer"
    assert result["frame"] == 15
    assert result["actualTime"] == 0.5
    assert result["resolution"] == "720p"
    assert media.sinks[0].frames[-1].frame == 15
    service.close(session.session_id)
    assert media.sinks[0].closed


def test_control_boundaries_flush_pending_media_before_publishing() -> None:
    """Every control boundary must drop the old generation's queued media.

    Seek/pause/play/sync route through ``_stop_scan``, which advances the sink
    generation after the old scan stops and before the new still or playback
    stream is published. Queue draining removes buffered items; the generation
    check also rejects an old item already dequeued by the WebSocket sender.
    """

    service, _, media = _service()
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer-sdp"),
        quality=validate_quality(None, None),
    )
    sink = media.sinks[0]

    assert sink.flush_count == 0
    session.seek(requested_time=Fraction(51, 100), request_id="seek-1")
    assert sink.flush_count == 1
    session.pause()
    assert sink.flush_count == 2
    session.seek(requested_time=Fraction(0), request_id="seek-2")
    assert sink.flush_count == 3
    service.close(session.session_id)


def test_create_and_sync_publish_missing_media_for_the_pinned_source() -> None:
    service, _, _ = _service()
    reference = MissingMediaReference(
        locator="/offline/interview.mov",
        fcpxml_path="/fcpxml/project/sequence/spine/asset-clip",
        timeline_start=Fraction(0),
        timeline_duration=Fraction(1),
        has_video=True,
        has_audio=True,
    )
    service.documents.values["sha256:v1"] = replace(
        _document(),
        missing_media_references=(reference,),
    )
    service.documents.values["sha256:v2"] = replace(
        _document(revision_name="revision-2"),
        missing_media_references=(replace(reference, locator="/offline/replacement.mov"),),
    )

    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality(None, None),
    )
    created = [event for event in session.events.snapshot() if event.event == "missing_media"]
    assert created[-1].data["sourceVersion"] == "sha256:v1"
    assert created[-1].data["paths"][0]["basename"] == "interview.mov"

    service.sync(
        session.session_id,
        source_version="sha256:v2",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
    )
    synced = [event for event in session.events.snapshot() if event.event == "missing_media"]
    assert synced[-1].data["sourceVersion"] == "sha256:v2"
    assert synced[-1].data["paths"][0]["basename"] == "replacement.mov"
    service.close(session.session_id)


def test_invalid_initial_playhead_closes_media_without_allocating_a_producer() -> None:
    service, producers, media = _service()

    with pytest.raises(PreviewAPIError) as failure:
        service.create_session(
            source_version="sha256:v1",
            project_ref=PROJECT_REF,
            playhead=Fraction(1),
            offer=SessionDescription(type="offer", sdp="offer"),
            quality=validate_quality(None, None),
        )

    assert failure.value.code == "time_out_of_range"
    assert producers.created == []
    assert len(media.sinks) == 1 and media.sinks[0].closed


def test_new_seek_supersedes_old_seek_before_media_publication() -> None:
    service, producers, media = _service(block_first_seek=True)
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality(None, None),
    )
    failures: list[PreviewAPIError] = []

    def old_seek() -> None:
        try:
            session.seek(requested_time=Fraction(1, 10), request_id="old")
        except PreviewAPIError as error:
            failures.append(error)

    thread = threading.Thread(target=old_seek)
    thread.start()
    assert producers.created[0].seek_started.wait(timeout=1)
    new_result = session.seek(requested_time=Fraction(2, 10), request_id="new")
    thread.join(timeout=1)

    assert [error.code for error in failures] == ["seek_superseded"]
    assert new_result["frame"] == 6
    assert [frame.frame for frame in media.sinks[0].frames] == [6]
    service.close(session.session_id)


def test_sync_replaces_exact_source_version_but_keeps_media_connection() -> None:
    service, producers, media = _service()
    service.documents.values["sha256:v2"] = _document(revision_name="revision-2")
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality(None, None),
    )
    first_producer = producers.created[0]
    result = service.sync(
        session.session_id,
        source_version="sha256:v2",
        project_ref=PROJECT_REF,
        playhead=Fraction(1, 5),
    )
    sought = session.seek(requested_time=Fraction(1, 5), request_id="revision-2")

    assert result["sourceVersion"] == "sha256:v2"
    assert sought["sourceVersion"] == "sha256:v2"
    assert first_producer.closed
    assert len(media.sinks) == 1 and not media.sinks[0].closed
    service.close(session.session_id)


def test_sync_switches_project_under_same_library_version_and_keeps_webrtc_sink() -> None:
    producers = FakeProducers()
    media = FakeMedia()
    service = PreviewService(
        documents=MultiProjectDocuments(),
        producers=producers,
        media=media,
    )
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality(None, None),
    )

    result = service.sync(
        session.session_id,
        source_version="sha256:v1",
        project_ref=MultiProjectDocuments.SECOND_PROJECT_REF,
        playhead=Fraction(0),
    )

    assert result["sourceVersion"] == "sha256:v1"
    assert result["projectRef"] == MultiProjectDocuments.SECOND_PROJECT_REF
    assert session.document.project_name == "second-project"
    assert len(media.sinks) == 1 and not media.sinks[0].closed
    assert len(producers.created) == 2 and producers.created[0].closed
    service.close(session.session_id)


def test_sync_switches_fixed_preview_quality_without_replacing_webrtc_sink() -> None:
    service, producers, media = _service()
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality("720p"),
    )

    result = service.sync(
        session.session_id,
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        quality=validate_quality("480p"),
    )

    assert result["selectedResolution"] == "480p"
    assert session.selected_profile.value == "480p"
    assert producers.created[0].closed
    assert producers.created[-1].resolution.height == 480
    assert len(media.sinks) == 1 and not media.sinks[0].closed
    service.close(session.session_id)


def test_invalid_sync_playhead_leaves_the_existing_revision_usable() -> None:
    service, producers, media = _service()
    service.documents.values["sha256:v2"] = _document(revision_name="revision-2")
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality(None, None),
    )
    first_producer = producers.created[0]

    with pytest.raises(PreviewAPIError) as failure:
        service.sync(
            session.session_id,
            source_version="sha256:v2",
            project_ref=PROJECT_REF,
            playhead=Fraction(1),
        )

    assert failure.value.code == "time_out_of_range"
    assert session.source_version == "sha256:v1" and session.document.project_name == "test"
    assert session.producer is first_producer and not first_producer.closed
    assert len(producers.created) == 1
    sought = session.seek(requested_time=Fraction(1, 5), request_id="still-revision-1")
    assert sought["sourceVersion"] == "sha256:v1" and media.sinks[0].frames[-1].frame == 6
    service.close(session.session_id)


def test_scan_starts_on_seek_equivalent_frame_and_emits_ended() -> None:
    service, _, media = _service()
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality(None, None),
    )
    sought = session.seek(requested_time=Fraction(1, 2), request_id="seek")
    played = session.play(requested_time=Fraction(1, 2))
    for _ in range(100):
        if not session.playing:
            break
        time.sleep(0.01)

    scan_frames = media.sinks[0].frames[1:]
    assert played["startFrame"] == sought["frame"] == 15
    assert scan_frames[0].frame == sought["frame"]
    event_names = [event.event for event in session.events.snapshot()]
    assert "ended" in event_names
    assert event_names.index("playing", 1) < event_names.index("ended")
    assert event_names[-1] == "playing"
    service.close(session.session_id)


def test_scan_paces_audio_from_the_same_nonzero_project_origin() -> None:
    documents = FakeDocuments()
    producers = FakeProducers()
    media = FakeMedia()
    service = PreviewService(
        documents=documents,
        producers=producers,
        media=media,
        audio=FakeAudioFactory(),
    )
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality(None, None),
    )

    session.play(requested_time=Fraction(1, 2))
    for _ in range(200):
        if not session.playing:
            break
        time.sleep(0.01)

    sink = media.sinks[0]
    assert [frame.time for frame in sink.audio_frames] == [
        Fraction(1, 2) + Fraction(index, 50) for index in range(5)
    ]
    assert sink.audio_write_times[-1] - sink.audio_write_times[0] >= 0.05
    service.close(session.session_id)


def test_fixed_720p_scan_reports_buffering_instead_of_falling_back() -> None:
    service, _, _ = _service(frame_delay=0.2)
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality(None, None),
    )
    session.play(requested_time=Fraction(0))
    for _ in range(200):
        if any(event.event == "buffering" for event in session.events.snapshot()):
            break
        time.sleep(0.01)
    session.pause()

    events = session.events.snapshot()
    assert session.selected_profile.value == "720p"
    assert any(event.event == "buffering" for event in events)
    assert not any(event.event == "quality" for event in events)
    service.close(session.session_id)


def test_fixed_480p_scan_reports_buffering_instead_of_degrading_semantics() -> None:
    service, _, _ = _service(frame_delay=0.2)
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality("480p", []),
    )
    session.play(requested_time=Fraction(0))
    for _ in range(200):
        if any(event.event == "buffering" for event in session.events.snapshot()):
            break
        time.sleep(0.01)
    session.pause()

    events = session.events.snapshot()
    assert session.selected_profile.value == "480p"
    assert any(event.event == "buffering" for event in events)
    assert not any(event.event == "quality" for event in events)
    service.close(session.session_id)


def test_http_commands_return_scoped_results_and_fail_loud_errors() -> None:
    service, _, media = _service()
    renders = RenderJobService(
        documents=service.documents,
        runner=FakeExportRunner(),
        artifact_directory=Path("/tmp") / f"tensor-preview-test-{time.time_ns()}",
    )
    client = TestClient(create_app(service, renders=renders))
    missing_project = client.post(
        "/api/editor/preview/sessions",
        json={
            "sourceVersion": "sha256:v1",
            "offer": {"type": "offer", "sdp": "offer"},
        },
    )
    assert missing_project.status_code == 400
    assert missing_project.json()["error"]["code"] == "invalid_request"
    created = client.post(
        "/api/editor/preview/sessions",
        json={
            "sourceVersion": "sha256:v1",
            "projectRef": PROJECT_REF,
            "playhead": 0,
            "offer": {"type": "offer", "sdp": "offer"},
            "quality": {"resolution": "540p"},
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["sourceVersion"] == "sha256:v1"
    assert body["projectRef"] == PROJECT_REF
    assert body["answer"] == {"type": "answer", "sdp": "answer-sdp"}
    assert body["selectedResolution"] == "540p"

    synced = client.post(
        f"/api/editor/preview/sessions/{body['sessionId']}/sync",
        json={
            "sourceVersion": "sha256:v1",
            "projectRef": PROJECT_REF,
            "playhead": 0,
            "quality": {"resolution": "480p"},
        },
    )
    assert synced.status_code == 200
    assert synced.json()["selectedResolution"] == "480p"

    preserved = client.post(
        f"/api/editor/preview/sessions/{body['sessionId']}/sync",
        json={
            "sourceVersion": "sha256:v1",
            "projectRef": PROJECT_REF,
            "playhead": 0,
        },
    )
    assert preserved.status_code == 200
    assert preserved.json()["selectedResolution"] == "480p"

    obsolete_fallback = client.post(
        "/api/editor/preview/sessions",
        json={
            "sourceVersion": "sha256:v1",
            "projectRef": PROJECT_REF,
            "offer": {"type": "offer", "sdp": "offer"},
            "quality": {"preferredResolution": "720p", "fallbackResolutions": ["540p"]},
        },
    )
    assert obsolete_fallback.status_code == 400
    assert obsolete_fallback.json()["error"]["code"] == "invalid_request"

    sought = client.post(
        f"/api/editor/preview/sessions/{body['sessionId']}/seek",
        json={"time": 0.51, "requestId": "seek-http"},
    )
    assert sought.status_code == 200
    assert sought.json()["frame"] == 15
    assert media.sinks[0].frames[-1].frame == 15

    missing = client.post(
        "/api/editor/preview/sessions/missing/seek",
        json={"time": 0, "requestId": "missing"},
    )
    assert missing.status_code == 404
    assert missing.json() == {
        "error": {
            "code": "preview_not_found",
            "message": "Preview session 'missing' does not exist.",
            "retryable": False,
        }
    }

    assert client.delete(f"/api/editor/preview/sessions/{body['sessionId']}").status_code == 204
    assert client.delete(f"/api/editor/preview/sessions/{body['sessionId']}").status_code == 204

    started = client.post(
        "/api/editor/render",
        json={"sourceVersion": "sha256:v1", "projectRef": PROJECT_REF},
    )
    assert started.status_code == 202
    render = started.json()
    assert render["status"] == "queued"
    assert render["projectRef"] == PROJECT_REF
    assert render["resolution"] == "1080p"
    assert (render["width"], render["height"]) == (1920, 1080)
    for _ in range(100):
        status = client.get(f"/api/editor/renders/{render['jobId']}").json()
        if status["status"] == "completed":
            break
        time.sleep(0.005)
    assert status["completedFrames"] == 30
    artifact = client.get(status["artifact"]["url"])
    assert artifact.status_code == 200 and artifact.content == b"fake-mp4"


def test_sse_route_replays_ready_event_with_monotonic_id() -> None:
    service, _, _ = _service()
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality(None, None),
    )
    client = TestClient(create_app(service))

    closer = threading.Thread(
        target=lambda: (time.sleep(0.05), service.close(session.session_id)),
        daemon=True,
    )
    closer.start()
    response = client.get(f"/api/editor/preview/sessions/{session.session_id}/events")
    closer.join(timeout=1)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "id: 1\nevent: ready\ndata:" in response.text


def test_cors_allows_loopback_editor_origins_only() -> None:
    service, _, _ = _service()
    client = TestClient(create_app(service, allowed_origins=("http://localhost:3000",)))
    allowed = client.options(
        "/api/editor/preview/sessions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    rejected = client.options(
        "/api/editor/preview/sessions",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert rejected.status_code == 400
    assert "access-control-allow-origin" not in rejected.headers


def test_preview_routes_require_the_instance_bearer_token() -> None:
    service, _, _ = _service()
    client = TestClient(create_app(service, auth_token="secret"))
    payload = {
        "sourceVersion": "sha256:v1",
        "projectRef": PROJECT_REF,
        "offer": {"type": "offer", "sdp": "offer"},
    }

    missing = client.post("/api/editor/preview/sessions", json=payload)
    wrong = client.post(
        "/api/editor/preview/sessions",
        json=payload,
        headers={"Authorization": "Bearer wrong"},
    )
    accepted = client.post(
        "/api/editor/preview/sessions",
        json=payload,
        headers={"Authorization": "Bearer secret"},
    )

    assert missing.status_code == wrong.status_code == 401
    assert missing.json()["error"]["code"] == "unauthorized"
    assert accepted.status_code == 201
    service.close(accepted.json()["sessionId"])


def test_health_is_public_and_readiness_uses_503_until_ready() -> None:
    service, _, _ = _service()
    client = TestClient(
        create_app(
            service,
            auth_token="secret",
            readiness=lambda: {"ready": False, "compileStatus": "source_invalid"},
        )
    )

    assert client.get("/healthz").json() == {"ok": True}
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["compileStatus"] == "source_invalid"


def test_http_seek_crosses_a_real_webrtc_peer_connection() -> None:
    async def exercise() -> None:
        service = PreviewService(
            documents=FakeDocuments(),
            producers=FakeProducers(),
            media=AiortcMediaFactory(queue_depth=2),
        )
        client = TestClient(create_app(service))
        peer = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        remote_tracks = {}

        @peer.on("track")
        def on_track(track) -> None:
            remote_tracks[track.kind] = track

        peer.addTransceiver("video", direction="recvonly")
        offer = await peer.createOffer()
        await peer.setLocalDescription(offer)
        created = await asyncio.to_thread(
            client.post,
            "/api/editor/preview/sessions",
            json={
                "sourceVersion": "sha256:v1",
                "projectRef": PROJECT_REF,
                "offer": {
                    "type": peer.localDescription.type,
                    "sdp": peer.localDescription.sdp,
                },
            },
        )
        assert created.status_code == 201
        body = created.json()
        await peer.setRemoteDescription(
            RTCSessionDescription(
                type=body["answer"]["type"],
                sdp=body["answer"]["sdp"],
            )
        )
        for _ in range(100):
            if peer.connectionState == "connected":
                break
            await asyncio.sleep(0.01)
        assert peer.connectionState == "connected"

        sought = await asyncio.to_thread(
            client.post,
            f"/api/editor/preview/sessions/{body['sessionId']}/seek",
            json={"time": 0.5, "requestId": "webrtc-seek"},
        )
        assert sought.status_code == 200
        received = await asyncio.wait_for(remote_tracks["video"].recv(), timeout=5.0)
        assert (received.width, received.height) == (1280, 720)

        await asyncio.to_thread(
            client.delete,
            f"/api/editor/preview/sessions/{body['sessionId']}",
        )
        await peer.close()

    asyncio.run(exercise())


def test_http_scan_delivers_synchronized_webrtc_video_and_audio_tracks() -> None:
    async def exercise() -> None:
        service = PreviewService(
            documents=FakeDocuments(),
            producers=FakeProducers(),
            media=AiortcMediaFactory(queue_depth=4),
            audio=FakeAudioFactory(),
        )
        client = TestClient(create_app(service))
        peer = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        remote_tracks = {}

        @peer.on("track")
        def on_track(track) -> None:
            remote_tracks[track.kind] = track

        peer.addTransceiver("video", direction="recvonly")
        peer.addTransceiver("audio", direction="recvonly")
        offer = await peer.createOffer()
        await peer.setLocalDescription(offer)
        created = await asyncio.to_thread(
            client.post,
            "/api/editor/preview/sessions",
            json={
                "sourceVersion": "sha256:v1",
                "projectRef": PROJECT_REF,
                "offer": {
                    "type": peer.localDescription.type,
                    "sdp": peer.localDescription.sdp,
                },
            },
        )
        body = created.json()
        await peer.setRemoteDescription(
            RTCSessionDescription(
                type=body["answer"]["type"],
                sdp=body["answer"]["sdp"],
            )
        )
        for _ in range(100):
            if peer.connectionState == "connected":
                break
            await asyncio.sleep(0.01)
        played = await asyncio.to_thread(
            client.post,
            f"/api/editor/preview/sessions/{body['sessionId']}/play",
            json={"time": 0.9},
        )
        assert played.status_code == 202 and played.json()["startFrame"] == 27
        video, audio = await asyncio.gather(
            asyncio.wait_for(remote_tracks["video"].recv(), timeout=5.0),
            asyncio.wait_for(remote_tracks["audio"].recv(), timeout=5.0),
        )
        assert (video.width, video.height) == (1280, 720)
        assert audio.sample_rate == 48_000 and audio.samples == 960
        await asyncio.to_thread(
            client.delete,
            f"/api/editor/preview/sessions/{body['sessionId']}",
        )
        await peer.close()

    asyncio.run(exercise())


def test_render_job_cancellation_is_server_side_and_removes_partial_artifact(tmp_path: Path) -> None:
    runner = BlockingExportRunner()
    jobs = RenderJobService(
        documents=FakeDocuments(),
        runner=runner,
        artifact_directory=tmp_path / "renders",
    )
    job = jobs.start(source_version="sha256:v1", project_ref=PROJECT_REF, profile="720p")
    assert runner.started.wait(timeout=1)
    cancelling = jobs.cancel(job.job_id)
    assert cancelling.status == "cancelling"
    assert job.thread is not None
    job.thread.join(timeout=1)
    assert jobs.get(job.job_id).status == "cancelled"
    assert not job.output_path.exists()


def test_render_completion_wins_over_cancel_after_artifact_commit(tmp_path: Path) -> None:
    runner = CompletedExportRunner()
    jobs = RenderJobService(
        documents=FakeDocuments(),
        runner=runner,
        artifact_directory=tmp_path / "renders",
    )
    job = jobs.start(source_version="sha256:v1", project_ref=PROJECT_REF, profile="720p")
    assert runner.artifact_ready.wait(timeout=1)

    jobs.cancel(job.job_id)
    runner.return_allowed.set()
    assert job.thread is not None
    job.thread.join(timeout=1)

    assert jobs.get(job.job_id).status == "completed"
    assert job.output_path.read_bytes() == b"complete"


def test_render_rejects_an_empty_project_before_starting_a_worker(tmp_path: Path) -> None:
    documents = FakeDocuments()
    documents.values["sha256:v1"] = replace(_document(), duration=Fraction(0))
    jobs = RenderJobService(
        documents=documents,
        runner=FakeExportRunner(),
        artifact_directory=tmp_path / "renders",
    )

    with pytest.raises(PreviewAPIError) as empty:
        jobs.start(source_version="sha256:v1", project_ref=PROJECT_REF, profile="720p")

    assert empty.value.code == "empty_timeline"
    assert empty.value.status == 422
    assert jobs.active_count == 0


def test_raw_frame_sink_invalidates_an_already_dequeued_payload() -> None:
    sink = RawFrameMediaSink(queue_depth=2)
    sink._enqueue(b"old")
    old = sink.get(timeout=0)

    sink.flush()
    sink._enqueue(b"new")
    new = sink.get(timeout=0)

    assert not sink.is_current(old)
    assert sink.is_current(new)
    assert new.data == b"new"


def test_raw_preview_http_session_produces_binary_video() -> None:
    service = PreviewService(
        documents=FakeDocuments(),
        producers=FakeProducers(),
        raw_media=RawFrameMediaFactory(queue_depth=2),
    )
    with TestClient(create_app(service)) as client:
        created = client.post(
            "/api/editor/preview/sessions/raw",
            json={
                "sourceVersion": "sha256:v1",
                "projectRef": PROJECT_REF,
                "playhead": 0,
                "quality": {"resolution": "480p"},
            },
        )
        assert created.status_code == 201
        body = created.json()
        assert body["selectedResolution"] == "480p"
        assert body["projectRef"] == PROJECT_REF

        sought = client.post(
            f"/api/editor/preview/sessions/{body['sessionId']}/seek",
            json={"time": 0.5, "requestId": "raw-seek"},
        )
        assert sought.status_code == 200
        session = service.session(body["sessionId"])
        assert isinstance(session.media_sink, RawFrameMediaSink)
        item = session.media_sink.get(timeout=0)
        assert session.media_sink.is_current(item)
        kind, frame, width, height = struct.unpack("<BIHH", item.data[:9])
        assert (kind, frame, width, height) == (0, 15, 852, 480)
        assert len(item.data) == 9 + width * height * 4


def test_render_service_rejects_a_competing_export(tmp_path: Path) -> None:
    runner = BlockingExportRunner()
    jobs = RenderJobService(
        documents=FakeDocuments(),
        runner=runner,
        artifact_directory=tmp_path / "renders",
    )
    first = jobs.start(source_version="sha256:v1", project_ref=PROJECT_REF, profile="720p")
    assert runner.started.wait(timeout=1)

    with pytest.raises(PreviewAPIError) as busy:
        jobs.start(source_version="sha256:v1", project_ref=PROJECT_REF, profile="720p")
    assert busy.value.code == "render_busy"
    assert busy.value.status == 409

    jobs.cancel(first.job_id)
    assert first.thread is not None
    first.thread.join(timeout=1)


def test_service_shutdown_closes_sessions_and_cancels_render_jobs(tmp_path: Path) -> None:
    service, producers, media = _service()
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality(None, None),
    )
    runner = BlockingExportRunner()
    jobs = RenderJobService(
        documents=service.documents,
        runner=runner,
        artifact_directory=tmp_path / "renders",
    )
    job = jobs.start(source_version="sha256:v1", project_ref=PROJECT_REF, profile=None)
    assert runner.started.wait(timeout=1)

    service.shutdown()
    jobs.shutdown()

    assert service.active_count == 0
    assert jobs.active_count == 0
    assert producers.created[0].closed
    assert media.sinks[0].closed
    assert jobs.get(job.job_id).status == "cancelled"
    with pytest.raises(PreviewAPIError) as missing:
        service.session(session.session_id)
    assert missing.value.code == "preview_not_found"


def test_session_close_wakes_sse_before_slow_media_cleanup() -> None:
    media = BlockingCloseMedia()
    service = PreviewService(
        documents=FakeDocuments(),
        producers=FakeProducers(),
        media=media,
    )
    session, _ = service.create_session(
        source_version="sha256:v1",
        project_ref=PROJECT_REF,
        playhead=Fraction(0),
        offer=SessionDescription(type="offer", sdp="offer"),
        quality=validate_quality(None, None),
    )
    after_id = session.events.snapshot()[-1].id
    subscriber_finished = threading.Event()

    def consume_until_closed() -> None:
        list(session.events.subscribe(after_id=after_id))
        subscriber_finished.set()

    subscriber = threading.Thread(target=consume_until_closed, daemon=True)
    subscriber.start()
    closer = threading.Thread(target=session.close, daemon=True)
    closer.start()
    assert media.sink.close_started.wait(timeout=1)
    assert subscriber_finished.wait(timeout=1)

    media.sink.release_close.set()
    closer.join(timeout=1)
    subscriber.join(timeout=1)
    assert not closer.is_alive()
    assert not subscriber.is_alive()
