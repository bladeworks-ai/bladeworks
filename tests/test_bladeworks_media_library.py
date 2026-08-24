"""Focused contract tests for explicit bundle-media inventory and import."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image

pytest.importorskip("av")

from bladeworks.preview import media_visuals  # noqa: E402

from bladeworks.examples import EXAMPLES_DIR  # noqa: E402
from bladeworks.preview.media_library import (  # noqa: E402
    MediaLibrary,
    MediaLibraryError,
)
from bladeworks.preview.media_routes import create_media_router  # noqa: E402
from bladeworks.preview.contracts import PreviewAPIError  # noqa: E402
from bladeworks.preview.media_visuals import (  # noqa: E402
    MediaVisualRequest,
    _accumulate_audio_peaks,
    _audio_bands,
    _video_thumbnails,
)


SOURCE_MEDIA = EXAMPLES_DIR / "single_clip.fcpxmld" / "Media" / "a.mp4"


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "Editor.fcpxmld"
    bundle.mkdir()
    (bundle / "Info.fcpxml").write_text("<fcpxml/>", encoding="utf-8")
    return bundle


def test_inventory_is_sorted_and_ignores_hidden_or_partial_files(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    media = bundle / "Media"
    media.mkdir()
    shutil.copy2(SOURCE_MEDIA, media / "z.mp4")
    shutil.copy2(SOURCE_MEDIA, media / "a.mp4")
    shutil.copy2(SOURCE_MEDIA, media / ".hidden.mp4")
    shutil.copy2(SOURCE_MEDIA, media / "unfinished.mp4.partial")

    inventory = MediaLibrary(bundle).inventory()

    assert [item.relative_path for item in inventory.items] == ["Media/a.mp4", "Media/z.mp4"]
    assert all(item.has_video and item.has_audio for item in inventory.items)
    assert all(item.width and item.height and item.duration for item in inventory.items)
    assert inventory.failures == ()


def test_inventory_keeps_readable_files_when_one_probe_fails(tmp_path: Path) -> None:
    """A sidecar or damaged file must not abort the rest of the Media scan."""

    bundle = _bundle(tmp_path)
    media = bundle / "Media"
    media.mkdir()
    shutil.copy2(SOURCE_MEDIA, media / "good.mp4")
    (media / "notes.txt").write_text("not media", encoding="utf-8")
    (media / "broken.mov").write_bytes(b"not media")

    inventory = MediaLibrary(bundle).inventory()

    assert [item.filename for item in inventory.items] == ["good.mp4"]
    assert {failure.relative_path for failure in inventory.failures} == {
        "Media/broken.mov",
        "Media/notes.txt",
    }
    payload = inventory.to_json()
    assert len(payload["failures"]) == 2
    assert all("relativePath" in failure and "message" in failure for failure in payload["failures"])


def test_refresh_adopts_direct_filesystem_changes_only_when_requested(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    library = MediaLibrary(bundle)
    assert library.inventory().items == ()
    (bundle / "Media").mkdir()
    shutil.copy2(SOURCE_MEDIA, bundle / "Media" / "new.mp4")

    assert library.inventory().items == ()
    assert [item.filename for item in library.refresh().items] == ["new.mp4"]


def test_import_is_staged_probed_and_never_edits_fcpxml(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    source_xml = (bundle / "Info.fcpxml").read_bytes()
    library = MediaLibrary(bundle)

    record = library.import_file(SOURCE_MEDIA, "interview.mp4")

    assert record.relative_path == "Media/interview.mp4"
    assert record.has_video and record.has_audio
    assert (bundle / "Info.fcpxml").read_bytes() == source_xml
    assert not list((bundle / "Media").glob("*.partial"))
    assert not list((bundle / "Media").glob(".*.partial"))


def test_import_rejects_collision_traversal_and_missing_source(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    library = MediaLibrary(bundle)
    library.import_file(SOURCE_MEDIA, "same.mp4")

    with pytest.raises(MediaLibraryError, match="already exists") as collision:
        library.import_file(SOURCE_MEDIA, "same.mp4")
    assert collision.value.code == "media_import_conflict"
    assert collision.value.status == 409

    with pytest.raises(MediaLibraryError) as traversal:
        library.import_file(SOURCE_MEDIA, "../escape.mp4")
    assert traversal.value.code == "invalid_request"
    assert not (tmp_path / "escape.mp4").exists()

    with pytest.raises(MediaLibraryError) as missing:
        library.import_file(tmp_path / "absent.mp4", "absent.mp4")
    assert missing.value.code == "media_source_not_found"


def test_failed_probe_removes_published_target_and_partial(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    invalid = tmp_path / "invalid.mov"
    invalid.write_bytes(b"not media")
    library = MediaLibrary(bundle)

    with pytest.raises(MediaLibraryError) as failure:
        library.import_file(invalid, "invalid.mov")

    assert failure.value.code == "media_probe_failed"
    assert not (bundle / "Media" / "invalid.mov").exists()
    assert not list((bundle / "Media").glob(".*.partial"))


def test_media_router_owns_only_the_public_media_routes(tmp_path: Path) -> None:
    router = create_media_router(MediaLibrary(_bundle(tmp_path)))
    assert {(route.path, next(iter(route.methods))) for route in router.routes} == {
        ("/api/editor/media", "GET"),
        ("/api/editor/media/refresh", "POST"),
        ("/api/editor/media/import", "POST"),
        ("/api/editor/media/upload", "POST"),
        ("/api/editor/media/visuals", "POST"),
    }


def test_media_visuals_are_small_transient_samples_and_reject_unsafe_paths(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    media = bundle / "Media"
    media.mkdir()
    shutil.copy2(SOURCE_MEDIA, media / "sample.mp4")
    app = FastAPI()

    @app.exception_handler(PreviewAPIError)
    async def public_error(_request: Request, error: PreviewAPIError) -> JSONResponse:
        return JSONResponse(status_code=error.status, content=error.body())

    app.include_router(create_media_router(MediaLibrary(bundle)))
    client = TestClient(app)
    response = client.post(
        "/api/editor/media/visuals",
        json={
            "relativePath": "Media/sample.mp4",
            "start": 0,
            "duration": 0.5,
            "thumbnailCount": 3,
            "thumbnailWidth": 64,
            "audioBands": 16,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["thumbnails"]) == 3
    assert all(frame.startswith("data:image/jpeg;base64,") for frame in payload["thumbnails"])
    assert len(payload["audioBands"]) == 16
    assert all(0 <= peak <= 1 for peak in payload["audioBands"])
    assert not list(media.glob(".*")), "sampling must not create a disk cache"

    traversal = client.post(
        "/api/editor/media/visuals",
        json={"relativePath": "../Info.fcpxml", "duration": 1},
    )
    oversized = client.post(
        "/api/editor/media/visuals",
        json={"relativePath": "Media/sample.mp4", "duration": 301},
    )
    assert traversal.status_code == 400
    assert traversal.json()["error"]["code"] == "invalid_media_path"
    assert oversized.status_code == 400
    assert oversized.json()["error"]["code"] == "invalid_request"


def test_media_visuals_preserve_nested_inventory_path(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    nested = bundle / "Media" / "nested"
    nested.mkdir(parents=True)
    shutil.copy2(SOURCE_MEDIA, nested / "sample.mp4")
    app = FastAPI()
    app.include_router(create_media_router(MediaLibrary(bundle)))
    client = TestClient(app)

    response = client.post(
        "/api/editor/media/visuals",
        json={"relativePath": "Media/nested/sample.mp4", "duration": 1},
    )

    assert response.status_code == 200
    assert response.json()["relativePath"] == "Media/nested/sample.mp4"


def test_symlinked_media_root_resolves_visible_files(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    external = tmp_path / "external-media"
    external.mkdir()
    shutil.copy2(SOURCE_MEDIA, external / "sample.mp4")
    (bundle / "Media").symlink_to(external, target_is_directory=True)
    library = MediaLibrary(bundle)

    resolved = library.resolve_media_path("Media/sample.mp4")

    assert resolved == (external / "sample.mp4").resolve()


def test_video_thumbnails_normalize_a_nonzero_stream_origin(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the timestamp-origin fixture")
    source = tmp_path / "nonzero-origin.ts"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=2",
            "-c:v", "mpeg2video", "-f", "mpegts", str(source),
        ],
        check=True,
    )

    thumbnails = _video_thumbnails(
        source,
        MediaVisualRequest(start=0, duration=2, thumbnail_count=4, thumbnail_width=80, audio_bands=0),
    )

    assert len(thumbnails) == 4
    assert len(set(thumbnails)) == 4


def test_still_image_visuals_decode_once_without_seeking(tmp_path: Path) -> None:
    source = tmp_path / "still.jpg"
    Image.new("RGB", (160, 90), color=(24, 80, 140)).save(source)

    thumbnails = _video_thumbnails(
        source,
        MediaVisualRequest(start=0, duration=0.04, thumbnail_count=4, thumbnail_width=80, audio_bands=0),
    )

    assert len(thumbnails) == 4
    assert len(set(thumbnails)) == 1
    assert thumbnails[0].startswith("data:image/jpeg;base64,")


def test_audio_peak_reduction_vectorizes_large_sample_arrays() -> None:
    peaks = np.zeros(4, dtype=np.float32)
    samples = np.zeros(1_000_000, dtype=np.float32)
    samples[[0, 249_999, 250_000, 999_999]] = [0.2, 0.7, 0.4, 1.0]
    request = MediaVisualRequest(
        start=0,
        duration=10,
        thumbnail_count=0,
        thumbnail_width=96,
        audio_bands=4,
    )

    _accumulate_audio_peaks(peaks, samples, frame_start=0, sample_rate=100_000, request=request)

    assert peaks.tolist() == pytest.approx([0.7, 0.4, 0.0, 1.0])


def test_audio_bands_preserve_levels_relative_to_digital_full_scale(tmp_path: Path) -> None:
    sample_rate = 8_000
    amplitudes = (0.1, 0.2, 0.4, 0.8)
    samples = np.concatenate([
        np.full(sample_rate // 4, round(amplitude * 32767), dtype=np.int16)
        for amplitude in amplitudes
    ])
    path = tmp_path / "level-steps.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())

    bands = _audio_bands(
        path,
        MediaVisualRequest(
            start=0,
            duration=1,
            thumbnail_count=0,
            thumbnail_width=96,
            audio_bands=4,
        ),
    )

    assert bands == pytest.approx(amplitudes, abs=0.001)
    assert max(bands) < 1.0, "the loudest local band must not be normalized to full height"


def test_audio_bands_subtract_the_stream_timestamp_origin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Decoded timestamps may be absolute even though Studio requests relative media time."""

    class FakeFormat:
        is_planar = True

    class FakeLayout:
        channels = ("mono",)

    class FakeFrame:
        sample_rate = 4
        samples = 2
        format = FakeFormat()
        layout = FakeLayout()

        def __init__(self, time: float, amplitude: float) -> None:
            self.time = time
            self._values = np.full((1, 2), amplitude, dtype=np.float32)

        def to_ndarray(self) -> np.ndarray:
            return self._values

    class FakeStream:
        start_time = 5_000
        time_base = 1 / 1_000
        rate = 4

    class FakeContainer:
        start_time = None

        def __init__(self) -> None:
            self.streams = type("Streams", (), {"audio": [FakeStream()]})()

        def __enter__(self) -> "FakeContainer":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def seek(self, *_args: object, **_kwargs: object) -> None:
            return None

        def decode(self, _stream: FakeStream) -> list[FakeFrame]:
            return [FakeFrame(5.0, 0.25), FakeFrame(5.5, 0.75)]

    monkeypatch.setattr(media_visuals.av, "open", lambda *_args, **_kwargs: FakeContainer())
    bands = _audio_bands(
        tmp_path / "unused.wav",
        MediaVisualRequest(start=0, duration=1, thumbnail_count=0, thumbnail_width=96, audio_bands=2),
    )

    assert bands == pytest.approx([0.25, 0.75])


def test_unsigned_8_bit_silence_is_centered_at_zero(tmp_path: Path) -> None:
    sample_rate = 8_000
    path = tmp_path / "unsigned-silence.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(1)
        output.setframerate(sample_rate)
        output.writeframes(np.full(sample_rate, 128, dtype=np.uint8).tobytes())

    bands = _audio_bands(
        path,
        MediaVisualRequest(
            start=0,
            duration=1,
            thumbnail_count=0,
            thumbnail_width=96,
            audio_bands=4,
        ),
    )

    assert bands == [0.0, 0.0, 0.0, 0.0]


def test_browser_upload_streams_stages_and_publishes_without_xml_mutation(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    source_xml = (bundle / "Info.fcpxml").read_bytes()
    app = FastAPI()

    @app.exception_handler(PreviewAPIError)
    async def public_error(_request: Request, error: PreviewAPIError) -> JSONResponse:
        return JSONResponse(status_code=error.status, content=error.body())

    app.include_router(create_media_router(MediaLibrary(bundle)))
    client = TestClient(app)
    response = client.post(
        "/api/editor/media/upload",
        content=SOURCE_MEDIA.read_bytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Bladeworks-Filename": "browser.mp4",
        },
    )

    assert response.status_code == 201
    assert response.json()["media"]["relativePath"] == "Media/browser.mp4"
    assert (bundle / "Media" / "browser.mp4").read_bytes() == SOURCE_MEDIA.read_bytes()
    assert (bundle / "Info.fcpxml").read_bytes() == source_xml
    assert not list((bundle / "Media").glob(".*.partial"))

    collision = client.post(
        "/api/editor/media/upload",
        content=b"different",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Bladeworks-Filename": "browser.mp4",
        },
    )
    traversal = client.post(
        "/api/editor/media/upload",
        content=b"different",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Bladeworks-Filename": "../escape.mp4",
        },
    )
    assert collision.status_code == 409
    assert collision.json()["error"]["code"] == "media_import_conflict"
    assert traversal.status_code == 400
    assert traversal.json()["error"]["code"] == "invalid_request"
    assert not (tmp_path / "escape.mp4").exists()
