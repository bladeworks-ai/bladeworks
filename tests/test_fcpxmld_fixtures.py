""".fcpxmld bundle input: both resolution paths, end to end through the CLI.

What this covers
----------------

The renderer accepts two input forms and this proves BOTH still work:

1. A plain ``.fcpxml`` FILE whose media ``src`` is an ABSOLUTE ``file://``
   URI. This is the pre-existing behavior; the ``test_absolute_file_url_*``
   case renders one at tmp time (pointing at a committed fixture's media) so a
   regression in the bundle work cannot silently break it.

2. A canonical ``.fcpxmld`` BUNDLE: a directory named ``<Name>.fcpxmld``
   holding ``Info.fcpxml`` at its root plus a ``Media/`` subfolder, with
   BUNDLE-RELATIVE media ``src`` (e.g. ``src="Media/a.mp4"``). Five tiny
   committed bundles under the packaged ``examples/`` directory each exercise
   one core
   mechanic (single AV clip, spine + connected lane, adjust-transform,
   cross-dissolve transition, Color Adjustments). Each renders through
   ``cli.main(["render", <bundle_dir>, ...])`` and is probed for a real
   video stream, an AAC audio stream, and the exact decoded frame count.

The negative case proves the loud failure contract: a directory that is NOT a
valid bundle (no ``Info.fcpxml``) raises ``FCPXMLParseError`` by name -- never
a silent fallback.

Skips when ffmpeg / ffprobe / torch / PyAV are unavailable (the tensor backend
needs torch + PyAV, and the probes need ffmpeg + ffprobe).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("av")

from bladeworks.cli import main  # noqa: E402
from bladeworks.core.compiler import compile_fcpxml  # noqa: E402
from bladeworks.core.errors import FCPXMLParseError  # noqa: E402
from bladeworks.examples import EXAMPLES, EXAMPLES_DIR  # noqa: E402


# The sample bundles were promoted from a private fixtures path to the packaged
# ``examples/`` directory so the CLI (``bladeworks examples``) and this test
# read from ONE source of truth.  ``EXAMPLES_DIR`` holds ``<name>.fcpxmld``.
FIXTURES = EXAMPLES_DIR

# Each committed bundle plus its expected decoded frame count (sequence
# duration x 30 fps), taken straight from the examples manifest.
BUNDLES = {name: example.expected_frames for name, example in EXAMPLES.items()}


def _tools() -> tuple[str, str]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("needs ffmpeg/ffprobe")
    return ffmpeg, ffprobe


def _stream_codecs(ffprobe: str, path: Path, kind: str) -> list[str]:
    """Return the codec names of every stream of ``kind`` ('v' or 'a')."""

    result = subprocess.run(
        (ffprobe, "-v", "error", "-select_streams", kind,
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)),
        capture_output=True, text=True, check=True,
    )
    return [line for line in result.stdout.split() if line]


def _decoded_frame_count(ffprobe: str, path: Path) -> int:
    result = subprocess.run(
        (ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)),
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


@pytest.mark.parametrize("name,expected_frames", sorted(BUNDLES.items()))
def test_fcpxmld_bundle_renders(name: str, expected_frames: int, tmp_path: Path) -> None:
    """Path 2: a canonical .fcpxmld bundle renders with A + V, frame-exact."""

    ffmpeg, ffprobe = _tools()
    bundle = FIXTURES / f"{name}.fcpxmld"
    assert (bundle / "Info.fcpxml").is_file(), f"missing fixture bundle {bundle}"

    output = tmp_path / f"{name}.mp4"
    rc = main(["render", str(bundle), "--backend", "tensor",
               "--output", str(output), "--no-progress"])
    assert rc == 0, f"{name} render failed"
    assert output.exists()

    assert _stream_codecs(ffprobe, output, "v"), f"{name} has no video stream"
    assert "aac" in _stream_codecs(ffprobe, output, "a"), f"{name} has no AAC audio"
    assert _decoded_frame_count(ffprobe, output) == expected_frames


def test_absolute_file_url_plain_fcpxml_still_renders(tmp_path: Path) -> None:
    """Path 1: a plain .fcpxml with an ABSOLUTE file:// media src still works.

    The src points at one committed bundle's media by absolute URI, so this is
    the pre-existing (non-bundle) resolution path proven green alongside the
    new bundle path.
    """

    ffmpeg, ffprobe = _tools()
    media = (FIXTURES / "single_clip.fcpxmld" / "Media" / "a.mp4").resolve()
    assert media.is_file()

    source = tmp_path / "absolute.fcpxml"
    source.write_text(
        f'''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
<resources>
  <format id="fmt" frameDuration="1/30s" width="160" height="90" colorSpace="1-1-1 (Rec. 709)"/>
  <asset id="a" start="0s" duration="2s" hasVideo="1" hasAudio="1" audioSources="1"
         audioChannels="2" audioRate="48000" format="fmt">
    <media-rep kind="original-media" src="{media.as_uri()}"/></asset>
</resources>
<library><event name="t"><project name="absolute">
<sequence format="fmt" duration="1s"><spine>
  <asset-clip ref="a" offset="0s" start="0s" duration="1s" audioRole="dialogue"/>
</spine></sequence></project></event></library></fcpxml>''',
        encoding="utf-8",
    )

    output = tmp_path / "absolute.mp4"
    rc = main(["render", str(source), "--backend", "tensor",
               "--output", str(output), "--no-progress"])
    assert rc == 0
    assert "aac" in _stream_codecs(ffprobe, output, "a")
    assert _decoded_frame_count(ffprobe, output) == 30


def test_directory_without_info_fcpxml_raises_loudly(tmp_path: Path) -> None:
    """Negative: a directory that is not a valid bundle fails by name."""

    not_a_bundle = tmp_path / "broken.fcpxmld"
    (not_a_bundle / "Media").mkdir(parents=True)
    # No Info.fcpxml at the root -> loud FCPXMLParseError, never a silent fallback.
    with pytest.raises(FCPXMLParseError) as excinfo:
        compile_fcpxml(not_a_bundle)
    assert "Info.fcpxml" in str(excinfo.value)
