"""Experimental contracts for selected multicam hierarchy and split audio.

These tests intentionally use the isolated renderer suite.  They freeze the
typed MC-1 adapter without adding the experimental FCPXML renderer to backend
production CI.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import wave

import pytest

from bladeworks.core.audio_execution import (
    build_audio_execution_plan,
    probe_audio_asset,
    run_audio_execution_plan,
)
from bladeworks.core.multicam_execution import (
    MulticamSelectionError,
    build_multicam_execution_plan,
)
from bladeworks.core.model import TimeMapPoint
from bladeworks.core.parser import parse_fcpxml
from bladeworks.core.story_containers import (
    CompoundResourceCatalog,
    build_story_container_plan,
)
from bladeworks.core.story_ir import RenderGroup, walk_render_nodes


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = Path(__file__).parent / "fixtures" / "multicam_execution_1_14.fcpxml"
_DTD = Path(__file__).parent / "FCPXMLv1_14.dtd"
_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")


def _source():
    return parse_fcpxml(_FIXTURE)


def test_multicam_execution_fixture_is_fcpxml_1_14_dtd_valid() -> None:
    xmllint = shutil.which("xmllint")
    if xmllint is None:
        pytest.skip("xmllint is unavailable")
    completed = subprocess.run(
        [xmllint, "--noout", "--dtdvalid", str(_DTD), str(_FIXTURE)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_independent_choices_build_selected_video_and_audio() -> None:
    plan = build_multicam_execution_plan(_source())

    resolved = plan.sources["spine/mc-clip[1]"]
    assert (resolved.video_angle_id, resolved.audio_angle_id) == (
        "angle-b",
        "angle-a",
    )
    assert (resolved.source_start, resolved.duration) == (Fraction(0), Fraction(6))

    assert len(plan.audio.items) == 1
    audio = plan.audio.items[0]
    assert audio.name == "Audio Camera A"
    assert (audio.absolute_start, audio.duration, audio.source_start) == (
        Fraction(0),
        Fraction(4),
        Fraction(100),
    )
    assert (audio.source_channels, audio.output_channels) == ((2,), ("R",))
    assert audio.role is not None
    assert audio.role.qualified == "dialogue.dialogue-1"


def test_parser_infers_omitted_multicam_duration_from_angle_extents(
    tmp_path: Path,
) -> None:
    """Real Final Cut exports may express multicam extent only in the angles."""

    xml = _FIXTURE.read_text(encoding="utf-8").replace(
        ' tcStart="0s" duration="6s" tcFormat="NDF"',
        ' tcStart="0s" tcFormat="NDF"',
        1,
    )
    source_path = tmp_path / "duration-in-angles.fcpxml"
    source_path.write_text(xml, encoding="utf-8")

    source = parse_fcpxml(source_path)

    assert source.multicams["multicam"].duration == Fraction(6)
    assert build_multicam_execution_plan(source).sources[
        "spine/mc-clip[1]"
    ].duration == Fraction(6)


def test_scope_order_is_angle_item_then_source_then_clip() -> None:
    plan = build_multicam_execution_plan(_source())
    source = plan.sources["spine/mc-clip[1]"]
    source_scope = source.video_scope
    assert source_scope is not None and source_scope.transform is not None
    assert source_scope.transform.position == (10.0, -6.0)
    assert source_scope.transform.anchor == (1.0, 2.0)
    assert source_scope.crop is not None
    assert source_scope.crop.mode == "crop"
    assert source_scope.conform_type == "fill"
    assert (source_scope.blend_opacity, source_scope.blend_mode) == (
        0.75,
        "multiply",
    )
    assert [effect.ref for effect in source_scope.filters] == ["fx-source"]

    angle_item = source.video_story[0]
    assert angle_item.transform is not None
    assert angle_item.transform.position == (5.0, -2.0)
    assert [effect.ref for effect in angle_item.filters] == ["fx-angle"]

    audio = plan.audio.items[0]
    assert [layer.gain.initial for layer in audio.control_layers if layer.gain] == [
        -1.0,
        6.0,
    ]
    instance = plan.audio.source_instances[0]
    assert [layer.gain.initial for layer in instance.controls if layer.gain] == [-2.0]
    assert [layer.path for layer in audio.control_layers] == [
        "resources/media[@id='multicam']/multicam/mc-angle[1]/asset-clip[1]",
        "spine/mc-clip[1]/mc-source[1]/audio-role-source[1]",
    ]
    assert [layer.path for layer in instance.controls] == ["spine/mc-clip[1]"]


def test_selected_angle_uses_shared_group_compound_connected_and_gap_ir() -> None:
    plan = build_multicam_execution_plan(_source())
    source = plan.sources["spine/mc-clip[1]"]

    # The resolver keeps the selected angle as source-local story. Compound
    # expansion happens recursively only when the ordinary clip-instance
    # compiler consumes this source; no synthetic mc-clip/clip/spine is made.
    assert [(node.kind, node.name, node.start, node.duration) for node in source.video_story] == [
        ("ref-clip", "Compound Camera B", Fraction(20), Fraction(4)),
        ("gap", "Angle Gap", Fraction(0), Fraction(2)),
    ]
    deferred = plan.story.root.children[0]
    assert isinstance(deferred, RenderGroup)
    assert deferred.kind == "mc-clip"
    assert deferred.children == ()
    assert [node.name for node in deferred.connected_children] == [
        "Timeline Connected Overlay"
    ]


def test_connected_storyline_can_extend_past_its_anchor_clip() -> None:
    """An anchor limits source content, not its connected timeline siblings."""

    source = _source()
    anchor = source.spine[0]
    first = replace(
        anchor,
        path=f"{anchor.path}/spine[1]/mc-clip[1]",
        offset=Fraction(0),
        start=Fraction(1),
        duration=Fraction(1),
        audio_start=None,
        audio_duration=None,
        children=(),
    )
    second = replace(
        first,
        path=f"{anchor.path}/spine[1]/mc-clip[2]",
        offset=Fraction(1),
    )
    connected_spine = replace(
        anchor,
        kind="spine",
        path=f"{anchor.path}/spine[1]",
        name=None,
        ref=None,
        lane=1,
        offset=Fraction(4),
        start=Fraction(0),
        duration=Fraction(2),
        audio_start=None,
        audio_duration=None,
        multicam_sources=(),
        children=(first, second),
        raw_xml='<spine lane="1" offset="4s"/>',
    )
    source = replace(
        source,
        spine=(replace(anchor, children=(connected_spine,)),),
    )

    plan = build_multicam_execution_plan(source)
    nested_paths = (
        f"{anchor.path}/spine[1]/mc-clip[1]",
        f"{anchor.path}/spine[1]/mc-clip[2]",
    )

    assert all(path in plan.sources for path in nested_paths)
    assert all(
        path in {node.path for node in walk_render_nodes(plan.story.root)}
        for path in nested_paths
    )
    assert [item.absolute_start for item in plan.audio.items] == [
        Fraction(0),
        Fraction(4),
        Fraction(5),
    ]


@pytest.mark.skipif(
    not _FFMPEG or not _FFPROBE,
    reason="stock FFmpeg and FFprobe are required for the PCM oracle",
)
def test_connected_audio_bypasses_muted_multicam_source_pad(
    tmp_path: Path,
) -> None:
    """Freeze Final Cut's connected-audio module boundary with real samples.

    Main callers:
    - The isolated multicam/audio contract suite.

    Why this exists:
    A connected role component is visually nested under ``mc-clip`` in XML,
    but Final Cut mixes it beside the selected multicam audio pad.  Applying
    the multicam clip's post-mix gain to that connected component turns an
    explicitly authored studio track into silence.
    """

    camera = tmp_path / "camera.wav"
    studio = tmp_path / "studio.wav"
    for path, frequency in ((camera, 233), (studio, 997)):
        subprocess.run(
            (
                str(_FFMPEG),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:sample_rate=48000:duration=1",
                "-c:a",
                "pcm_s16le",
                str(path),
            ),
            check=True,
        )
    xml = f'''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
      <resources>
        <format id="fmt" frameDuration="1/30s" width="16" height="16" colorSpace="1-1-1 (Rec. 709)"/>
        <asset id="camera" name="camera" start="0s" duration="1s" hasVideo="0"
          hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
          <media-rep kind="original-media" src="{camera.as_uri()}"/>
        </asset>
        <asset id="studio" name="studio" start="0s" duration="1s" hasVideo="0"
          hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
          <media-rep kind="original-media" src="{studio.as_uri()}"/>
        </asset>
        <media id="multicam" name="multicam"><multicam format="fmt" tcStart="0s" duration="1s">
          <mc-angle name="camera" angleID="camera-angle">
            <asset-clip ref="camera" offset="0s" start="0s" duration="1s" audioRole="dialogue"/>
          </mc-angle>
        </multicam></media>
      </resources>
      <library><event name="Audio"><project name="Audio"><sequence format="fmt" duration="1s"
        tcStart="0s" audioLayout="stereo" audioRate="48k"><spine>
        <mc-clip ref="multicam" offset="0s" start="0s" duration="1s">
          <adjust-volume amount="-96dB"/>
          <mc-source angleID="camera-angle" srcEnable="audio"/>
          <asset-clip ref="studio" lane="-1" offset="0s" start="0s" duration="1s"
            audioRole="dialogue"><audio-channel-source srcCh="1" role="dialogue.dialogue-1"/>
          </asset-clip>
        </mc-clip>
      </spine></sequence></project></event></library>
    </fcpxml>'''
    source_path = tmp_path / "connected-audio.fcpxml"
    source_path.write_text(xml, encoding="utf-8")
    audio = build_multicam_execution_plan(parse_fcpxml(source_path)).audio

    multicam_path = "spine/mc-clip[1]"
    connected = next(item for item in audio.items if item.asset_id == "studio")
    camera_item = next(item for item in audio.items if item.asset_id == "camera")
    assert multicam_path not in connected.ancestor_paths
    assert multicam_path in camera_item.ancestor_paths

    bindings = {
        "camera": probe_audio_asset("camera", camera, ffprobe_path=str(_FFPROBE)),
        "studio": probe_audio_asset("studio", studio, ffprobe_path=str(_FFPROBE)),
    }
    execution = build_audio_execution_plan(audio, bindings)
    assert len(execution.source_instances) == 1
    assert execution.source_instances[0].input_count == 1
    assert "volume='pow(10,(-96)/20)'" in execution.filter_complex

    output = tmp_path / "connected-output.wav"
    run_audio_execution_plan(
        execution,
        output,
        ffmpeg_path=str(_FFMPEG),
        codec="pcm_s16le",
    )
    with wave.open(str(studio), "rb") as handle:
        studio_frames = handle.readframes(handle.getnframes())
        studio_values = struct.unpack("<" + "h" * (len(studio_frames) // 2), studio_frames)
    with wave.open(str(output), "rb") as handle:
        assert (handle.getframerate(), handle.getnchannels(), handle.getnframes()) == (
            48_000,
            2,
            48_000,
        )
        output_frames = handle.readframes(handle.getnframes())
        interleaved = struct.unpack("<" + "h" * (len(output_frames) // 2), output_frames)
    output_left = interleaved[0::2]
    output_right = interleaved[1::2]
    dot = sum(left * right for left, right in zip(output_left, studio_values))
    output_energy = sum(value * value for value in output_left)
    studio_energy = sum(value * value for value in studio_values)
    correlation = dot / math.sqrt(output_energy * studio_energy)
    rms_ratio = math.sqrt(output_energy / studio_energy)
    assert correlation >= 0.99999
    assert rms_ratio == pytest.approx(1.0, abs=0.001)
    assert max(
        abs(observed - reference)
        for observed, reference in zip(output_left, studio_values)
    ) <= 1
    assert max(
        abs(observed - reference)
        for observed, reference in zip(output_right, studio_values)
    ) <= 1
    quarter_second = 12_000
    assert all(
        math.sqrt(
            sum(value * value for value in output_left[start : start + quarter_second])
            / quarter_second
        )
        > 1_000
        for start in range(0, 48_000, quarter_second)
    )

    probe = subprocess.run(
        (
            str(_FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=time_base,duration_ts:packet=pts,duration",
            "-of",
            "json",
            str(output),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    assert payload["streams"][0] == {"time_base": "1/48000", "duration_ts": 48000}
    assert payload["packets"][0]["pts"] == 0
    assert sum(packet["duration"] for packet in payload["packets"]) == 48_000


def test_selected_audio_can_descend_through_angle_compound_and_group() -> None:
    source = _source()
    containers = build_story_container_plan(source)

    multicam = source.multicams["multicam"]
    angle_b = multicam.angles[1]
    angle_b = replace(
        angle_b,
        story=(replace(angle_b.story[0], src_enable="all"), *angle_b.story[1:]),
    )
    multicam = replace(multicam, angles=(multicam.angles[0], angle_b))
    clip = source.spine[0]
    source = replace(
        source,
        multicams={**source.multicams, "multicam": multicam},
        spine=(
            replace(
                clip,
                multicam_sources=(
                    replace(clip.multicam_sources[0], src_enable="none"),
                    replace(clip.multicam_sources[1], src_enable="all"),
                ),
            ),
        ),
    )

    compound = containers.resources.stories["compound-b"]
    group = compound.story[0]
    group = replace(
        group,
        children=(replace(group.children[0], src_enable="all"), *group.children[1:]),
    )
    compound = replace(compound, story=(group, *compound.story[1:]))
    containers = replace(
        containers,
        resources=CompoundResourceCatalog(
            stories={**containers.resources.stories, "compound-b": compound}
        ),
    )

    plan = build_multicam_execution_plan(source, container_plan=containers)
    assert len(plan.audio.items) == 1
    audio = plan.audio.items[0]
    assert audio.name == "Grouped Camera B"
    assert (audio.absolute_start, audio.duration, audio.source_start) == (
        Fraction(0),
        Fraction(3),
        Fraction(200),
    )
    assert audio.ancestor_paths[-2:] == (
        "resources/media[@id='multicam']/multicam/mc-angle[2]/ref-clip[1]",
        "resources/media[@id='compound-b']/sequence/spine/clip[1]",
    )


def test_missing_and_duplicate_choices_never_auto_select() -> None:
    source = _source()
    clip = source.spine[0]
    none_selected = replace(
        source,
        spine=(replace(clip, multicam_sources=()),),
    )
    plan = build_multicam_execution_plan(none_selected)
    assert plan.audio.items == ()
    assert [(finding.code, finding.stream) for finding in plan.findings] == [
        ("multicam_video_not_selected", "video"),
        ("multicam_audio_not_selected", "audio"),
    ]
    resolved_source = plan.sources["spine/mc-clip[1]"]
    assert not resolved_source.has_video
    assert not resolved_source.has_audio
    deferred = plan.story.root.children[0]
    assert isinstance(deferred, RenderGroup)
    assert deferred.children == ()
    assert len(deferred.connected_children) == 1

    video_choice = clip.multicam_sources[1]
    duplicate = replace(
        source,
        spine=(
            replace(
                clip,
                multicam_sources=(*clip.multicam_sources, video_choice),
            ),
        ),
    )
    with pytest.raises(MulticamSelectionError, match="more than one video angle"):
        build_multicam_execution_plan(duplicate)

    unknown = replace(
        source,
        spine=(
            replace(
                clip,
                multicam_sources=(
                    replace(clip.multicam_sources[0], angle_id="missing-angle"),
                    clip.multicam_sources[1],
                ),
            ),
        ),
    )
    with pytest.raises(MulticamSelectionError, match="unknown angleID"):
        build_multicam_execution_plan(unknown)


def test_selected_audio_interval_inside_explicit_angle_gap_is_silent() -> None:
    source = _source()
    multicam = source.multicams["multicam"]
    audio_angle = multicam.angles[0]
    explicit_gap = replace(
        multicam.angles[1].story[1],
        offset=Fraction(0),
        duration=Fraction(1),
    )
    shifted_audio = replace(audio_angle.story[0], offset=Fraction(1))
    audio_angle = replace(
        audio_angle,
        story=(explicit_gap, shifted_audio),
    )
    clip = source.spine[0]
    source = replace(
        source,
        multicams={
            **source.multicams,
            "multicam": replace(
                multicam,
                angles=(audio_angle, multicam.angles[1]),
            ),
        },
        spine=(
            replace(
                clip,
                start=Fraction(0),
                duration=Fraction(1),
                audio_start=None,
                audio_duration=None,
            ),
        ),
    )

    plan = build_multicam_execution_plan(source)

    assert plan.audio.items == ()
    assert len(plan.audio.source_instances) == 1


def test_multicam_retime_composes_with_independent_split_audio() -> None:
    source = _source()
    clip = source.spine[0]
    timed = replace(
        source,
        spine=(
            replace(
                clip,
                time_map=(
                    TimeMapPoint(Fraction(0), Fraction(0), "linear"),
                    TimeMapPoint(Fraction(2), Fraction(1), "linear"),
                    TimeMapPoint(Fraction(6), Fraction(4), "linear"),
                ),
            ),
        ),
    )
    plan = build_multicam_execution_plan(timed)
    timing = plan.audio.source_instances[0].timing
    assert (timing.source_start, timing.source_duration) == (0, 4)
    assert [segment.rate for segment in timing.retime_map.segments] == [
        Fraction(1, 2),
        Fraction(3, 4),
    ]
