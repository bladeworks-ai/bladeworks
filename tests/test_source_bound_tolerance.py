"""Source-bound rounding is limited by real media cadence, never an epsilon.

Architecture map
================

FCP asset endpoint + clip endpoint on a different rational timescale
    -> derive the smallest declared audio-sample/video-frame unit
    -> accept sub-unit serialization drift
    -> reject an edit that exceeds that unit.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from bladeworks.core.compiler import compile_fcpxml
from bladeworks.core.errors import FCPXMLCompileError
from bladeworks.core.render_sources import source_bound_tolerance


def _audio_only_fixture(path: Path, *, clip_duration: str) -> Path:
    path.write_text(
        f'''<?xml version="1.0"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="f" frameDuration="1/30s" width="160" height="90"/>
    <asset id="a" start="0s" duration="1125324/48000s"
           hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000"/>
  </resources>
  <library><event><project><sequence format="f" duration="24s"><spine>
    <gap offset="0s" duration="24s">
      <asset-clip ref="a" lane="-1" offset="0s" duration="{clip_duration}"/>
    </gap>
  </spine></sequence></project></event></library>
</fcpxml>
''',
        encoding="utf-8",
    )
    return path


def test_smallest_declared_media_unit_wins_for_av_sources() -> None:
    assert source_bound_tolerance(
        has_video=True,
        frame_duration=Fraction(1, 30),
        has_audio=True,
        audio_rate=48_000,
    ) == Fraction(1, 48_000)


def test_sub_sample_timescale_rounding_compiles(tmp_path: Path) -> None:
    # The clip ends 1/60000 second after its asset: 0.8 of one 48 kHz sample.
    compiled = compile_fcpxml(
        _audio_only_fixture(tmp_path / "sub-sample.fcpxml", clip_duration="43958/1875s")
    )

    assert compiled.render.audio is not None
    assert len(compiled.render.audio.items) == 1


@pytest.mark.parametrize(
    "clip_duration",
    [
        "45013/1920s",  # exactly one 48 kHz sample beyond the asset
        "562663/24000s",  # two samples beyond the asset
    ],
)
def test_source_overrun_of_one_whole_sample_or_more_fails(
    tmp_path: Path,
    clip_duration: str,
) -> None:
    source = _audio_only_fixture(tmp_path / "overrun.fcpxml", clip_duration=clip_duration)

    with pytest.raises(FCPXMLCompileError, match="source range ends.*after asset"):
        compile_fcpxml(source)
