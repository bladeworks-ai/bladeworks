"""Stage 1 coverage: the backend-neutral PyAV audio-delivery resolver.

Architecture map
================

Exercises ``tensor/audio_delivery.py`` -- the argv-free copy of the legacy
``_build_audio_execution`` resolution logic (plan:
``pyav-audio-delivery-unification.md``).  These tests pin the behaviours the PyAV
port depends on and that must never silently regress:

- absent / inactive audio short-circuits to a ``silence`` resolution,
- an audible document builds an ``AudioExecutionPlan`` whose inputs start at
  index 0 (no shared argv),
- an asset that declares audio but carries no decodable stream records the loud
  ``omitted`` finding and drops to silent -- no silent failure,
- the channel-layout helper maps all three output layouts.

Main callers: pytest (experimental renderer job).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import shutil
import subprocess

import pytest

from bladeworks.core.errors import RenderCapabilityError
from bladeworks.core.model import AssetBinding, RenderDocument
from bladeworks.core.report import CompatibilityReport
from bladeworks.tensor.audio_delivery import (
    audio_delivery_layout,
    resolve_audio_delivery,
)
from tests._audio_delivery_corpus import (
    item,
    plan,
)


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
_SKIP = pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="stock FFmpeg is unavailable")


def _document(*, audio, bindings: tuple[AssetBinding, ...] = ()) -> RenderDocument:
    """Minimal RenderDocument carrying only what the audio resolver reads."""

    return RenderDocument(
        schema_version=1,
        source_sha256="fixture",
        source_path=Path("/tmp/fixture.fcpxml"),
        project_name="fixture",
        width=1920,
        height=1080,
        frame_duration=Fraction(1, 30),
        duration=Fraction(2),
        tc_start=Fraction(0),
        clips=(),
        transitions=(),
        asset_bindings=bindings,
        font_bindings=(),
        audio=audio,
    )


def _tone(path: Path) -> None:
    assert FFMPEG is not None
    subprocess.run(
        (
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:a", "pcm_s16le", str(path),
        ),
        check=True,
    )


def _video_only(path: Path) -> None:
    assert FFMPEG is not None
    subprocess.run(
        (
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.2:r=30",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ),
        check=True,
    )


def test_absent_and_inactive_audio_resolve_to_silence() -> None:
    absent = resolve_audio_delivery(
        _document(audio=None), ffprobe="ffprobe", report=CompatibilityReport()
    )
    assert absent.execution is None
    assert absent.mode == "silence"
    assert absent.output_duration == Fraction(2)

    inactive = resolve_audio_delivery(
        _document(audio=plan(item(active=False))),
        ffprobe="ffprobe",
        report=CompatibilityReport(),
    )
    assert inactive.execution is None
    assert inactive.mode == "silence"


@_SKIP
def test_audible_document_builds_execution_at_input_offset_zero(tmp_path: Path) -> None:
    media = tmp_path / "tone.wav"
    _tone(media)
    document = _document(
        audio=plan(item()),
        bindings=(AssetBinding(resource_id="asset-a", uid=None, path=media),),
    )
    resolution = resolve_audio_delivery(document, ffprobe=FFPROBE, report=CompatibilityReport())

    assert resolution.mode == "render"
    assert resolution.execution is not None
    # No shared argv: the single input is the media itself and the graph reads
    # source [0:a:0] (offset 0), not [2:a:0] like the legacy assembly.
    assert resolution.execution.inputs == (media,)
    assert "[0:a:0]" in resolution.execution.filter_complex
    assert len(resolution.effective_bindings) == 1


@_SKIP
def test_missing_audio_stream_records_omitted_finding_and_renders_silent(
    tmp_path: Path,
) -> None:
    media = tmp_path / "silent.mp4"
    _video_only(media)
    report = CompatibilityReport()
    document = _document(
        audio=plan(item()),
        bindings=(AssetBinding(resource_id="asset-a", uid=None, path=media),),
    )
    resolution = resolve_audio_delivery(document, ffprobe=FFPROBE, report=report)

    # The only audible item's asset has no decodable audio -> dropped to silence,
    # with a loud omitted finding that fails --strict.  No silent failure.
    assert resolution.execution is None
    assert resolution.mode == "silence"
    findings = [f for f in report.findings if f.construct == "missing audio stream"]
    assert len(findings) == 1
    assert findings[0].outcome == "omitted"
    assert "no decodable audio stream" in findings[0].disposition


def test_audio_delivery_layout_maps_supported_output_layouts() -> None:
    assert audio_delivery_layout(_document(audio=None)) == "stereo"
    assert audio_delivery_layout(_document(audio=plan(item(), layout="mono"))) == "mono"
    assert audio_delivery_layout(_document(audio=plan(item(), layout="stereo"))) == "stereo"


def test_surround_output_rejects_before_graph_build() -> None:
    """A ``surround`` (5.1) sequence must reject cleanly at plan/resolve time.

    Both the layout mapper and the full ``resolve_audio_delivery`` resolver reject
    surround output with a ``RenderCapabilityError`` naming the construct, BEFORE
    any asset is probed or any audio graph node is built -- so surround never
    reaches the cryptic libav ``Errno 22`` (ffmpeg exit 234) its 5.1 ``pan`` upmix
    would otherwise trigger. The reject fires ahead of binding resolution, so no
    real media is required.
    """

    surround_doc = _document(audio=plan(item(), layout="surround"))

    with pytest.raises(RenderCapabilityError, match="surround"):
        audio_delivery_layout(surround_doc)

    with pytest.raises(RenderCapabilityError, match="surround"):
        resolve_audio_delivery(
            surround_doc, ffprobe="ffprobe", report=CompatibilityReport()
        )
