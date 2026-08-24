"""``--backend tensor`` end to end: gate, assembly mux, manifest, A/V probes.

What this covers
----------------

The tensor renderer produces video only.  The executor's tensor branch adds
the rest of a delivery render around it: the capability gate
(``probe_tensor_plan`` / ``select_render_backend``), the single ffmpeg
assembly process that copies the tensor video and runs the calibrated audio
graph beside it (``tensor/assemble.py``), the frame-count and audio probes,
and the manifest.  Every assertion here is about that seam, not about pixels
-- pixel parity is the A/B gate's job (WS-G).

The audio assertion is the load-bearing one: the tensor and CPU backends
splice the *same* ``AudioExecutionPlan`` graph, only at a different input
index offset, so their decoded audio must be bit-identical.  Anything else
means the mux changed the audio, which would desynchronise delivery files.

Skips when torch / PyAV / ffmpeg are unavailable (torch and PyAV are optional
spike dependencies).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("av")

from bladeworks.cli import main  # noqa: E402
from bladeworks.core.compiler import compile_fcpxml  # noqa: E402
from bladeworks.executor import execute_render  # noqa: E402
from bladeworks.preview.export import TensorExecutorExportRunner  # noqa: E402
from bladeworks.preview.provider import (  # noqa: E402
    DEFAULT_REGISTERED_PROJECT_REF,
    RegisteredSourceProvider,
)
from bladeworks.preview.render_jobs import RenderJobService  # noqa: E402
from bladeworks.tensor import resolve_output_resolution  # noqa: E402


def _tools() -> tuple[str, str]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("needs ffmpeg/ffprobe")
    return ffmpeg, ffprobe


def _media(ffmpeg: str, path: Path, *, pattern: str, tone_hz: int) -> Path:
    """Write one 3 s 30 fps 320x180 clip with 48 kHz stereo audio."""

    subprocess.run(
        (
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"{pattern}=s=320x180:r=30:d=3",
            "-f", "lavfi", "-i",
            f"sine=frequency={tone_hz}:sample_rate=48000:duration=3",
            "-ac", "2", "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
            "-shortest", str(path),
        ),
        check=True,
    )
    return path


def _project(
    directory: Path,
    ffmpeg: str,
    *,
    blend: str = "",
    canvas: tuple[int, int] = (320, 180),
) -> Path:
    """Two spine clips (one trimmed) plus a lane-1 connected clip, all audible.

    ``blend`` injects an ``adjust-blend`` used by acceptance and rejection cases.
    """

    a = _media(ffmpeg, directory / "a.mp4", pattern="testsrc2", tone_hz=440)
    b = _media(ffmpeg, directory / "b.mp4", pattern="smptebars", tone_hz=660)
    width, height = canvas
    source = directory / f"av-{width}x{height}.fcpxml"
    source.write_text(
        f'''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
<resources>
  <format id="fmt" frameDuration="1/30s" width="{width}" height="{height}" colorSpace="1-1-1 (Rec. 709)"/>
  <asset id="a" start="0s" duration="3s" hasVideo="1" hasAudio="1" audioSources="1"
         audioChannels="2" audioRate="48000" format="fmt">
    <media-rep kind="original-media" src="{a.as_uri()}"/></asset>
  <asset id="b" start="0s" duration="3s" hasVideo="1" hasAudio="1" audioSources="1"
         audioChannels="2" audioRate="48000" format="fmt">
    <media-rep kind="original-media" src="{b.as_uri()}"/></asset>
</resources>
<library><event name="t"><project name="t">
<sequence format="fmt" duration="2s"><spine>
  <asset-clip ref="a" offset="0s" start="0s" duration="1s" audioRole="dialogue">{blend}</asset-clip>
  <asset-clip ref="b" offset="1s" start="1/2s" duration="1s" audioRole="dialogue">
    <asset-clip ref="a" lane="1" offset="1/2s" start="1s" duration="1/2s" audioRole="music"/>
  </asset-clip>
</spine></sequence></project></event></library></fcpxml>''',
        encoding="utf-8",
    )
    return source


def _short_4k_project(directory: Path, ffmpeg: str) -> Path:
    """One audible frame on a 4K canvas for inexpensive 1080p export proof."""

    media = _media(ffmpeg, directory / "short-a.mp4", pattern="testsrc2", tone_hz=440)
    source = directory / "short-4k.fcpxml"
    source.write_text(
        f'''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
<resources>
  <format id="project" frameDuration="1/30s" width="3840" height="2160" colorSpace="1-1-1 (Rec. 709)"/>
  <format id="asset" frameDuration="1/30s" width="320" height="180" colorSpace="1-1-1 (Rec. 709)"/>
  <asset id="a" start="0s" duration="3s" hasVideo="1" hasAudio="1" audioSources="1"
         audioChannels="2" audioRate="48000" format="asset">
    <media-rep kind="original-media" src="{media.as_uri()}"/></asset>
</resources>
<library><event name="t"><project name="short-4k">
<sequence format="project" duration="1/30s"><spine>
  <asset-clip ref="a" offset="0s" start="0s" duration="1/30s" audioRole="dialogue"/>
</spine></sequence></project></event></library></fcpxml>''',
        encoding="utf-8",
    )
    return source


def _render(source: Path, output: Path, *extra: str) -> int:
    return main(["render", str(source), "--output", str(output), *extra])


def _manifest(output: Path) -> dict:
    return json.loads(output.with_suffix(".manifest.json").read_text())


def _decoded_audio_digest(ffmpeg: str, path: Path, scratch: Path) -> str:
    """SHA-256 of the fully decoded float32 PCM of the output's audio track."""

    subprocess.run(
        (ffmpeg, "-v", "error", "-y", "-i", str(path), "-map", "0:a",
         "-c:a", "pcm_f32le", "-f", "wav", str(scratch)),
        check=True,
    )
    return hashlib.sha256(scratch.read_bytes()).hexdigest()


def _decoded_frame_count(ffprobe: str, path: Path) -> int:
    result = subprocess.run(
        (ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)),
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def _decoded_sample_count(ffprobe: str, path: Path) -> int:
    result = subprocess.run(
        (ffprobe, "-v", "error", "-select_streams", "a:0", "-count_frames",
         "-show_entries", "frame=nb_samples", "-of", "csv=p=0", str(path)),
        capture_output=True, text=True, check=True,
    )
    return sum(int(line) for line in result.stdout.split() if line.isdigit())


def _video_size(ffprobe: str, path: Path) -> tuple[int, int]:
    result = subprocess.run(
        (ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)),
        capture_output=True, text=True, check=True,
    )
    width, height = result.stdout.strip().split("x")
    return int(width), int(height)


def test_tensor_backend_matches_cpu_audio_and_frame_count(tmp_path: Path) -> None:
    ffmpeg, ffprobe = _tools()
    source = _project(tmp_path, ffmpeg)

    cpu_output = tmp_path / "cpu.mp4"
    tensor_output = tmp_path / "tensor.mp4"
    assert _render(source, cpu_output, "--backend", "cpu", "--strict") == 0
    assert _render(source, tensor_output, "--backend", "tensor", "--strict") == 0

    manifest = _manifest(tensor_output)
    assert manifest["status"] == "succeeded"
    assert manifest["render_backend"]["requested"] == "tensor"
    assert manifest["render_backend"]["selected"] == "tensor"
    assert manifest["render_backend"]["tensor"]["available"] is True
    # Three layers: two spine clips plus the lane-1 connected clip.
    assert manifest["render_backend"]["tensor"]["layer_count"] == 3
    assert manifest["render_backend"]["render_profile"]["name"] == "reference"
    assert manifest["render_backend"]["tensor_execution"]["frames"] == 60
    # Ledger admission (U3): the manifest must be sealed with the commit that
    # ran, resolved from this source tree, not guessed.
    assert manifest["commit"] and manifest["commit_unavailable_reason"] is None
    assert len(manifest["commit"]) == 40

    # (b) frame count exact, and verified independently of the executor.
    assert manifest["output"]["expected_frame_count"] == 60
    assert _decoded_frame_count(ffprobe, tensor_output) == 60

    # (c) audio: sample-exact length and bit-identical to the CPU render.
    probe = manifest["output"]["audio_probe"]
    assert probe["sample_rate"] == 48_000 and probe["channel_layout"] == "stereo"
    assert probe["expected_samples"] == 96_000
    decoded = _decoded_sample_count(ffprobe, tensor_output)
    assert decoded == probe["decoded_samples"]
    # One video frame of tolerance plus the codec's own frame granularity.
    assert 96_000 - 1_600 <= decoded <= 96_000 + 1_600 + probe["codec_frame_samples"]
    assert _decoded_audio_digest(ffmpeg, tensor_output, tmp_path / "tensor.wav") == (
        _decoded_audio_digest(ffmpeg, cpu_output, tmp_path / "cpu.wav")
    )

    # The CPU manifest gains the same two U3/U4 fields.
    cpu_manifest = _manifest(cpu_output)
    assert cpu_manifest["commit"] == manifest["commit"]
    assert cpu_manifest["output"]["audio_probe"]["expected_samples"] == 96_000


def test_tensor_executor_applies_fixed_export_resolution_and_keeps_audio(tmp_path: Path) -> None:
    ffmpeg, ffprobe = _tools()
    compiled = compile_fcpxml(_project(tmp_path, ffmpeg, canvas=(1920, 1080)))
    target = resolve_output_resolution(1920, 1080, "480p")
    output = tmp_path / "tensor-480p.mp4"

    execute_render(
        compiled.render,
        compiled.report,
        output_path=output,
        backend="tensor",
        output_resolution=target,
        encoder_preset="veryfast",
    )

    manifest = _manifest(output)
    assert _video_size(ffprobe, output) == (852, 480)
    assert (manifest["output"]["width"], manifest["output"]["height"]) == (852, 480)
    assert manifest["render_backend"]["output_resolution"] == {
        "profile": "480p",
        "width": 852,
        "height": 480,
    }
    assert manifest["output"]["audio_probe"]["sample_rate"] == 48_000


def test_render_job_default_produces_1080p_h264_with_audio(tmp_path: Path) -> None:
    """An omitted API profile reaches the complete executor as fixed 1080p."""

    ffmpeg, ffprobe = _tools()
    compiled = compile_fcpxml(_short_4k_project(tmp_path, ffmpeg))
    provider = RegisteredSourceProvider()
    provider.register("sha256:v1", compiled)
    jobs = RenderJobService(
        documents=provider,
        runner=TensorExecutorExportRunner(
            report_for=provider.report_for,
            encoder_preset="veryfast",
        ),
        artifact_directory=tmp_path / "renders",
    )

    job = jobs.start(
        source_version="sha256:v1",
        project_ref=DEFAULT_REGISTERED_PROJECT_REF,
        profile=None,
    )
    assert job.thread is not None
    job.thread.join(timeout=60)

    completed = jobs.get(job.job_id)
    assert completed.thread is not None and not completed.thread.is_alive()
    assert completed.status == "completed", completed.error
    assert completed.profile.value == "1080p"
    assert _video_size(ffprobe, completed.output_path) == (1920, 1080)
    probe = json.loads(
        subprocess.run(
            (
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name",
                "-of",
                "json",
                str(completed.output_path),
            ),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    codecs = {stream["codec_type"]: stream["codec_name"] for stream in probe["streams"]}
    assert codecs == {"video": "h264", "audio": "aac"}


def test_tensor_backend_rejects_unknown_blend_mode(tmp_path: Path) -> None:
    """(d) A construct outside the supported class fails loudly, by name."""

    ffmpeg, _ = _tools()
    source = _project(tmp_path, ffmpeg, blend='<adjust-blend mode="A New Mystery Mode"/>')
    output = tmp_path / "blend.mp4"

    assert _render(source, output, "--backend", "tensor", "--strict") == 1

    manifest = _manifest(output)
    assert manifest["status"] == "failed"
    assert "blend mode" in manifest["error"]
    assert "A New Mystery Mode" in manifest["error"]
    assert not output.exists()


def test_tensor_backend_accepts_reviewed_rgb_blend_mode(tmp_path: Path) -> None:
    ffmpeg, _ = _tools()
    source = _project(tmp_path, ffmpeg, blend='<adjust-blend mode="Multiply"/>')
    output = tmp_path / "multiply.mp4"

    assert _render(source, output, "--backend", "tensor") == 0
    manifest = _manifest(output)
    assert manifest["status"] == "succeeded"
    assert manifest["render_backend"]["selected"] == "tensor"


def test_tensor_backend_rejects_non_reference_render_profile(tmp_path: Path) -> None:
    """(e) ``fast8`` is a CPU filtergraph pixel policy; tensor must not accept it."""

    ffmpeg, _ = _tools()
    source = _project(tmp_path, ffmpeg)
    output = tmp_path / "fast8.mp4"

    assert _render(
        source, output, "--backend", "tensor", "--render-profile", "fast8"
    ) == 1
    manifest = _manifest(output)
    assert manifest["status"] == "failed"
    assert "render_profile='reference'" in manifest["error"]


def test_tensor_backend_rejects_oracle_mezzanine_and_segments(tmp_path: Path) -> None:
    """Both remaining Day-1 exits are capability errors, not silent downgrades."""

    ffmpeg, _ = _tools()
    source = _project(tmp_path, ffmpeg)

    mezzanine = tmp_path / "mezzanine.mov"
    assert _render(source, mezzanine, "--backend", "tensor", "--oracle-mezzanine") == 1
    assert "delivery output profile" in _manifest(mezzanine)["error"]

    segmented = tmp_path / "segmented.mp4"
    assert _render(source, segmented, "--backend", "tensor", "--cpu-segments") == 1
    assert "--cpu-segments" in _manifest(segmented)["error"]


def test_tensor_video_only_carries_silence_and_fails_strict(tmp_path: Path) -> None:
    """``--video-only`` behaves exactly as on the CPU path: silence + omission."""

    ffmpeg, ffprobe = _tools()
    source = _project(tmp_path, ffmpeg)

    output = tmp_path / "video_only.mp4"
    assert _render(source, output, "--backend", "tensor", "--video-only") == 0
    manifest = _manifest(output)
    assert manifest["status"] == "succeeded"
    assert manifest["compatibility"]["degraded"] is True
    # The output still carries a full-length silent track, so the A/V probe
    # applies unchanged.
    assert manifest["output"]["audio_probe"]["expected_samples"] == 96_000
    assert _decoded_frame_count(ffprobe, output) == 60

    strict_output = tmp_path / "video_only_strict.mp4"
    assert _render(
        source, strict_output, "--backend", "tensor", "--video-only", "--strict"
    ) == 1
    assert _manifest(strict_output)["status"] == "failed"


def test_auto_never_selects_tensor(tmp_path: Path) -> None:
    """``auto`` stays on the CPU reference even when tensor would accept."""

    ffmpeg, _ = _tools()
    source = _project(tmp_path, ffmpeg)
    output = tmp_path / "auto.mp4"

    assert _render(source, output, "--backend", "auto") == 0
    assert _manifest(output)["render_backend"]["selected"] == "cpu"
