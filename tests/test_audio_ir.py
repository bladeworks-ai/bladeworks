"""Isolated contracts for the video-independent experimental audio IR.

Architecture map
================

Small FCPXML 1.14 documents
    -> the shared secure source parser
    -> ``compile_audio_ir``
    -> exact J/L timing, streams, components, routing, and controls

These tests deliberately stop before FFmpeg.  Their job is to freeze the
audio scheduling contract and its fail-closed behavior without adding the
experimental renderer to production backend CI.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from bladeworks.core.audio_ir import (
    AudioIRAmbiguityError,
    AudioIRReferenceError,
    compile_audio_ir,
)
from bladeworks.core.compiler import compile_fcpxml
from bladeworks.core.multicam_execution import build_multicam_execution_plan
from bladeworks.core.parser import parse_fcpxml


def _document(
    body: str, *, assets: str, duration: str = "12s", layout: str = "stereo"
) -> str:
    return f'''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
    <resources>
      <format id="fmt" frameDuration="1/30s" width="320" height="180" colorSpace="1-1-1 (Rec. 709)"/>
      {assets}
    </resources>
    <library><event name="Audio"><project name="Audio"><sequence format="fmt" duration="{duration}"
      tcStart="0s" audioLayout="{layout}" audioRate="48k"><spine>{body}</spine>
    </sequence></project></event></library></fcpxml>'''


def _asset(
    resource_id: str = "a",
    *,
    duration: str = "12s",
    sources: int = 1,
    channels: int = 2,
) -> str:
    return f'''<asset id="{resource_id}" name="{resource_id}" start="0s" duration="{duration}"
      hasVideo="1" hasAudio="1" videoSources="1" audioSources="{sources}"
      audioChannels="{channels}" audioRate="48000" format="fmt">
      <media-rep kind="original-media" src="file:///tmp/{resource_id}.mov"/>
    </asset>'''


def _parse(tmp_path: Path, xml: str):
    path = tmp_path / "audio.fcpxml"
    path.write_text(xml, encoding="utf-8")
    return parse_fcpxml(path)


def test_split_audio_is_scheduled_independently_with_exact_retime(
    tmp_path: Path,
) -> None:
    source = _parse(
        tmp_path,
        _document(
            """<asset-clip ref="a" offset="4s" start="4s" duration="2s"
                audioStart="3s" audioDuration="4s" audioRole="dialogue.dialogue-1">
                <timeMap preservesPitch="0" frameSampling="floor">
                  <timept time="0s" value="3s" interp="linear"/>
                  <timept time="2s" value="4s" interp="linear"/>
                  <timept time="4s" value="7s" interp="linear"/>
              </timeMap>
              <adjust-volume amount="-6dB"><param name="amount">
                <fadeIn type="linear" duration="1/2s"/>
              </param></adjust-volume>
            </asset-clip>""",
            assets=_asset(),
        ),
    )

    plan = build_multicam_execution_plan(source).audio
    item = plan.items[0]
    assert plan.schema_version == 2
    assert (plan.layout, plan.sample_rate, plan.output_channels) == (
        "stereo",
        48_000,
        ("L", "R"),
    )
    instance = plan.source_instances[0]
    assert instance.source_id == "file:a"
    assert (instance.timing.absolute_start, instance.timing.duration) == (
        Fraction(4),
        Fraction(4),
    )
    assert [segment.rate for segment in instance.timing.retime_map.segments] == [
        Fraction(1, 2),
        Fraction(3, 2),
    ]
    assert (item.absolute_start, item.duration) == (Fraction(4), Fraction(4))
    assert (item.source_start, item.source_duration) == (Fraction(3), Fraction(4))
    assert item.role is not None and item.role.qualified == "dialogue.dialogue-1"
    assert item.retime == ()
    assert instance.preserves_pitch is False
    assert instance.controls[0].gain is not None
    assert instance.controls[0].gain.initial == -6.0
    assert instance.controls[0].gain.fades[0].duration == Fraction(1, 2)


def test_components_keep_routes_controls_enhancements_and_silence(
    tmp_path: Path,
) -> None:
    source = _parse(
        tmp_path,
        _document(
            """<asset-clip ref="a" offset="0s" start="0s" duration="4s" srcEnable="audio">
              <audio-channel-source srcCh="1, 2" outCh="L, R" role="dialogue.dialogue-1">
                <adjust-loudness amount="59" uniformity="3"/>
                <adjust-volume amount="-6dB"><param name="amount">
                  <fadeOut type="easeOut" duration="1s"/>
                  <keyframeAnimation>
                    <keyframe time="0s" value="-6dB" interp="linear" curve="linear"/>
                    <keyframe time="2s" value="0dB" interp="ease" curve="smooth" auxValue="2"/>
                  </keyframeAnimation>
                </param></adjust-volume>
                <adjust-panner mode="stereo" amount="0.25" stereo_spread="0.8">
                  <param name="amount"><keyframeAnimation>
                    <keyframe time="0s" value="-1"/><keyframe time="4s" value="1"/>
                  </keyframeAnimation></param>
                </adjust-panner>
                <mute start="1s" duration="1/2s"><fadeIn type="linear" duration="1/10s"/></mute>
              </audio-channel-source>
              <audio-channel-source srcCh="3, 4" outCh="L, R" role="music.music-1" active="0">
                <adjust-matchEQ><data key="archive">opaque</data></adjust-matchEQ>
              </audio-channel-source>
            </asset-clip>""",
            assets=_asset(channels=4),
        ),
    )

    plan = build_multicam_execution_plan(source).audio
    first, second = plan.items
    assert (first.source_channels, first.output_channels) == ((1, 2), ("L", "R"))
    assert first.audible is True
    assert second.source_channels == (3, 4)
    assert second.audible is False
    layer = first.control_layers[-1]
    assert layer.gain is not None
    assert [point.time for point in layer.gain.keyframes] == [Fraction(0), Fraction(2)]
    assert layer.gain.keyframes[1].aux_value == "2"
    assert layer.panner is not None
    assert layer.panner.parameters["stereo_spread"] == 0.8
    assert layer.mutes[0].source_start == Fraction(1)
    assert layer.mutes[0].fades[0].duration == Fraction(1, 10)
    assert layer.enhancements[0].kind == "adjust-loudness"
    assert any(finding.code == "audio_component_silent" for finding in plan.findings)
    assert any(
        finding.code == "audio_enhancement_not_implemented" for finding in plan.findings
    )


def test_multiple_source_stream_ids_are_preserved_and_missing_choice_fails(
    tmp_path: Path,
) -> None:
    assets = _asset(sources=2, channels=2)
    explicit = _parse(
        tmp_path,
        _document(
            """<audio ref="a" offset="0s" start="0s" duration="2s" srcID="1" srcCh="1" outCh="L" role="dialogue"/>
               <audio ref="a" offset="2s" start="2s" duration="2s" srcID="2" srcCh="2" outCh="R" role="music"/>""",
            assets=assets,
            duration="4s",
        ),
    )
    plan = compile_audio_ir(explicit)
    assert [item.source_stream_id for item in plan.items] == ["1", "2"]
    assert [item.source_channels for item in plan.items] == [(1,), (2,)]

    ambiguous = _parse(
        tmp_path,
        _document(
            '<asset-clip ref="a" offset="0s" start="0s" duration="2s"/>',
            assets=assets,
            duration="2s",
        ),
    )
    with pytest.raises(AudioIRAmbiguityError, match="2 audio streams.*no srcID"):
        compile_audio_ir(ambiguous)


def test_clip_component_override_and_sync_role_controls_are_executed(
    tmp_path: Path,
) -> None:
    source = _parse(
        tmp_path,
        _document(
            """<clip offset="1s" start="1s" duration="2s" audioStart="0s" audioDuration="4s">
              <adjust-volume amount="-1dB"/>
              <audio ref="a" offset="0s" start="0s" duration="4s" role="dialogue.dialogue-1"/>
              <audio-channel-source srcCh="2" outCh="R" role="dialogue.dialogue-1">
                <adjust-volume amount="-3dB"/>
              </audio-channel-source>
            </clip>
            <sync-clip offset="2s" start="0s" duration="2s">
              <audio ref="a" offset="0s" start="0s" duration="2s" role="music.music-1"/>
              <sync-source sourceID="storyline">
                <audio-role-source role="music.music-1" start="0s" duration="2s" active="0"/>
              </sync-source>
            </sync-clip>""",
            assets=_asset(),
            duration="4s",
        ),
    )

    plan = compile_audio_ir(source)
    first, second = plan.items
    assert (first.absolute_start, first.duration) == (Fraction(0), Fraction(4))
    assert (first.source_channels, first.output_channels) == ((2,), ("R",))
    assert [layer.gain.initial for layer in first.control_layers if layer.gain] == [
        -3.0,
        -1.0,
    ]
    assert second.role is not None and second.role.qualified == "music.music-1"
    assert second.audible is False
    selector_layer = next(
        layer for layer in second.control_layers if layer.role_selector
    )
    assert (selector_layer.source_start, selector_layer.source_duration) == (
        Fraction(0),
        Fraction(2),
    )


def test_invalid_channels_and_role_references_fail_explicitly(tmp_path: Path) -> None:
    invalid_channel = _parse(
        tmp_path,
        _document(
            """<asset-clip ref="a" offset="0s" start="0s" duration="2s">
              <audio-channel-source srcCh="3" outCh="L" role="dialogue"/>
            </asset-clip>""",
            assets=_asset(channels=2),
            duration="2s",
        ),
    )
    with pytest.raises(AudioIRReferenceError, match="selects channels.*has 2 channels"):
        compile_audio_ir(invalid_channel)

    invalid_role = _parse(
        tmp_path,
        _document(
            """<sync-clip offset="0s" start="0s" duration="2s">
              <audio ref="a" offset="0s" start="0s" duration="2s" role="dialogue.dialogue-1"/>
              <sync-source sourceID="storyline">
                <audio-role-source role="music.music-1"/>
              </sync-source>
            </sync-clip>""",
            assets=_asset(),
            duration="2s",
        ),
    )
    with pytest.raises(
        AudioIRReferenceError, match="does not match any emitted audio component"
    ):
        compile_audio_ir(invalid_role)


def test_primary_role_matches_its_default_numbered_subrole(tmp_path: Path) -> None:
    """Final Cut serializes bare ``dialogue`` as default ``dialogue-1``."""

    source = _parse(
        tmp_path,
        _document(
            """<sync-clip offset="0s" start="0s" duration="2s">
              <audio ref="a" offset="0s" start="0s" duration="2s" role="dialogue"/>
              <sync-source sourceID="storyline">
                <audio-role-source role="dialogue.dialogue-1"/>
              </sync-source>
            </sync-clip>""",
            assets=_asset(),
            duration="2s",
        ),
    )

    plan = compile_audio_ir(source)

    assert len(plan.items) == 1
    assert plan.items[0].role is not None
    assert plan.items[0].role.qualified == "dialogue"


def test_multiple_channel_components_are_not_video_adjustment_duplicates(
    tmp_path: Path,
) -> None:
    source = _parse(
        tmp_path,
        _document(
            """<asset-clip ref="a" offset="0s" start="0s" duration="2s">
              <audio-channel-source srcCh="1" outCh="L" role="dialogue.dialogue-1"/>
              <audio-channel-source srcCh="2" outCh="R" role="music.music-1"/>
            </asset-clip>""",
            assets=_asset(channels=2),
            duration="2s",
        ),
    )

    compiled = compile_fcpxml(source.source_path)
    assert [item.source_channels for item in compiled.render.audio.items] == [(1,), (2,)]
    assert [item.output_channels for item in compiled.render.audio.items] == [("L",), ("R",)]


def test_canonical_final_cut_stereo_panner_is_normalized_to_unit_range(
    tmp_path: Path,
) -> None:
    source = _parse(
        tmp_path,
        _document(
            """<asset-clip ref="a" offset="0s" start="0s" duration="2s">
              <adjust-panner mode="1 (Stereo Left/Right)" amount="-50">
                <param name="amount"><keyframeAnimation>
                  <keyframe time="0s" value="-100" interp="linear"/>
                  <keyframe time="2s" value="100" interp="linear"/>
                </keyframeAnimation></param>
              </adjust-panner>
            </asset-clip>""",
            assets=_asset(),
            duration="2s",
        ),
    )

    panner = compile_audio_ir(source).items[0].control_layers[0].panner
    assert panner is not None
    assert panner.mode == "stereo"
    assert panner.amount.initial == -0.5
    assert [point.value for point in panner.amount.keyframes] == [-1.0, 1.0]


def test_multicam_audio_resolves_selected_angle_and_role_layer(tmp_path: Path) -> None:
    assets = (
        _asset("a", duration="8s")
        + """
      <media id="mc" name="Multicam"><multicam format="fmt" tcStart="0s" duration="8s">
        <mc-angle name="Camera A" angleID="angle-a">
          <asset-clip ref="a" offset="0s" start="0s" duration="8s" audioRole="dialogue">
            <audio-channel-source srcCh="1, 2" role="dialogue.dialogue-1"/>
          </asset-clip>
        </mc-angle>
      </multicam></media>"""
    )
    source = _parse(
        tmp_path,
        _document(
            """<mc-clip ref="mc" offset="2s" start="2s" duration="2s"
                audioStart="1s" audioDuration="4s">
              <mc-source angleID="angle-a" srcEnable="audio">
                <audio-role-source role="dialogue.dialogue-1">
                  <adjust-volume amount="6dB"/>
                </audio-role-source>
              </mc-source>
            </mc-clip>""",
            assets=assets,
            duration="6s",
        ),
    )

    plan = build_multicam_execution_plan(source).audio
    item = plan.items[0]
    assert (item.absolute_start, item.duration, item.source_start) == (
        Fraction(1),
        Fraction(4),
        Fraction(1),
    )
    assert item.role is not None and item.role.qualified == "dialogue.dialogue-1"
    assert item.ancestor_paths[-1:] == ("spine/mc-clip[1]",)
    assert item.control_layers[-1].gain is not None
    assert item.control_layers[-1].gain.initial == 6.0
