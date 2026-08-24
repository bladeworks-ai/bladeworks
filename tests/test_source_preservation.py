"""Isolated contracts for genuine keyframes and fail-closed adjustments.

Architecture map
================

These tests exercise the source boundary before the v2 execution engines are
integrated.  A documented adjustment must either reach the typed source model
or produce a compatibility finding; invalid synthetic keyframe nesting must
fail instead of creating false support evidence.

The production backend CI does not collect this directory.  The renderer's
isolated test script is the only supported entry point.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from bladeworks.core.compiler import compile_fcpxml
from bladeworks.core.errors import FCPXMLParseError
from bladeworks.core.parser import parse_fcpxml


def _document(body: str, *, clip_attributes: str = "") -> str:
    return f'''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
    <resources>
      <format id="r1" frameDuration="1/30s" width="320" height="180" colorSpace="1-1-1 (Rec. 709)"/>
      <asset id="a1" start="0s" duration="2s" hasVideo="1" hasAudio="1"
             audioSources="1" audioChannels="2" audioRate="48000" format="r1">
        <media-rep kind="original-media" src="file:///tmp/core-source-preservation-missing.mov"/>
      </asset>
    </resources>
    <library><event name="Core"><project name="Core"><sequence format="r1" duration="2s">
      <spine><asset-clip ref="a1" offset="0s" start="0s" duration="2s" {clip_attributes}>{body}</asset-clip></spine>
    </sequence></project></event></library></fcpxml>'''


def test_genuine_nested_transform_animation_and_aux_value_are_preserved(tmp_path: Path) -> None:
    source = tmp_path / "nested.fcpxml"
    source.write_text(
        _document(
            '''<adjust-transform anchor="10 5">
              <param name="position"><keyframeAnimation>
                <keyframe time="0s" value="-10 0" interp="linear"/>
                <keyframe time="1s" value="10 0" interp="ease" curve="smooth" auxValue="2 3"/>
              </keyframeAnimation></param>
              <param name="anchor"><keyframeAnimation>
                <keyframe time="0s" value="0 0" interp="linear"/>
                <keyframe time="1s" value="10 5" interp="ease" curve="smooth"/>
              </keyframeAnimation></param>
            </adjust-transform>'''
        ),
        encoding="utf-8",
    )

    node = parse_fcpxml(source).spine[0]
    assert node.transform is not None
    assert node.transform.anchor == (10.0, 5.0)
    assert [frame.time for frame in node.transform.position_keyframes] == [Fraction(0), Fraction(1)]
    assert node.transform.position_keyframes[1].aux_value == "2 3"

    render_clip = compile_fcpxml(source).render.clips[0]
    assert render_clip.transform_animation is not None
    assert render_clip.transform_animation.position is not None
    assert render_clip.transform_animation.anchor is not None
    assert render_clip.transform_animation.position.value_at(Fraction(0)) == (-10.0, 0.0)
    assert render_clip.transform_animation.position.value_at(Fraction(1)) == (10.0, 0.0)
    assert render_clip.transform_animation.notices[0].code == "uncalibrated_aux_value"
    report = compile_fcpxml(source).report
    assert not any(
        finding.construct == "adjust-transform" and finding.outcome == "omitted"
        for finding in report.findings
    )
    curve = next(
        finding
        for finding in report.findings
        if finding.construct == "nonlinear transform keyframes"
    )
    assert "monotone cubic" in curve.disposition


def test_direct_synthetic_keyframes_fail_instead_of_claiming_support(tmp_path: Path) -> None:
    source = tmp_path / "direct.fcpxml"
    source.write_text(
        _document(
            '''<adjust-transform><param name="position">
              <keyframe time="0s" value="0 0"/>
            </param></adjust-transform>'''
        ),
        encoding="utf-8",
    )

    with pytest.raises(FCPXMLParseError, match="nested inside keyframeAnimation"):
        parse_fcpxml(source)


def test_shear_skew_transform_param_fails_loudly_instead_of_silent_drop(
    tmp_path: Path,
) -> None:
    """A shear/skew request must raise, not vanish into an identity transform.

    Regression guard for the former silent skip in ``_parse_transform``.  Final
    Cut's Transform has no shear degree of freedom, so ``param name="shear"`` used
    to be dropped with ``continue`` and rendered as a wrong-but-quiet identity.  It
    now surfaces as an explicit capability error that names the param and points at
    the supported Distort (adjust-corners) path.
    """

    source = tmp_path / "shear.fcpxml"
    source.write_text(
        _document(
            '''<adjust-transform>
              <param name="shear"><keyframeAnimation>
                <keyframe time="0s" value="0 0" interp="linear"/>
                <keyframe time="1s" value="0.5 0" interp="linear"/>
              </keyframeAnimation></param>
            </adjust-transform>'''
        ),
        encoding="utf-8",
    )

    with pytest.raises(FCPXMLParseError, match=r"shear/skew parameter 'shear'"):
        parse_fcpxml(source)


def test_unknown_transform_param_is_rejected_not_ignored(tmp_path: Path) -> None:
    """Any non-Transform param under adjust-transform is a loud parse error.

    The four valid Transform channels (position/scale/rotation/anchor) still
    parse; an unrecognized name must fail closed rather than be silently skipped.
    """

    source = tmp_path / "unknown-transform-param.fcpxml"
    source.write_text(
        _document(
            '''<adjust-transform position="1 2">
              <param name="wobble"><keyframeAnimation>
                <keyframe time="0s" value="0" interp="linear"/>
              </keyframeAnimation></param>
            </adjust-transform>'''
        ),
        encoding="utf-8",
    )

    with pytest.raises(FCPXMLParseError, match=r"unsupported parameter 'wobble'"):
        parse_fcpxml(source)


def test_genuine_opacity_animation_reaches_retime_aware_render_ir(tmp_path: Path) -> None:
    source = tmp_path / "opacity.fcpxml"
    source.write_text(
        _document(
            '''<adjust-blend amount="0.25"><param name="amount"><keyframeAnimation>
              <keyframe time="0s" value="0.25" interp="linear"/>
              <keyframe time="2s" value="0.75" interp="linear"/>
            </keyframeAnimation></param></adjust-blend>'''
        ),
        encoding="utf-8",
    )

    parsed = parse_fcpxml(source).spine[0]
    assert [frame.value for frame in parsed.blend_keyframes] == ["0.25", "0.75"]
    clip = compile_fcpxml(source).render.clips[0]
    assert clip.opacity_animation is not None
    assert clip.opacity_animation.value_at(Fraction(0)) == 0.25
    assert clip.opacity_animation.value_at(Fraction(2)) == 0.75
    assert not any(
        finding.construct == "adjust-blend" and finding.outcome == "omitted"
        for finding in compile_fcpxml(source).report.findings
    )


def test_opacity_interpolation_belongs_to_the_segment_leaving_its_keyframe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "opacity-segment-ownership.fcpxml"
    source.write_text(
        _document(
            '''<adjust-blend amount="0.84"><param name="amount"><keyframeAnimation>
              <keyframe time="0s" value="0.35"/>
              <keyframe time="1s" value="1" interp="easeIn"/>
              <keyframe time="2s" value="0.84"/>
            </keyframeAnimation></param></adjust-blend>'''
        ),
        encoding="utf-8",
    )

    animation = compile_fcpxml(source).render.clips[0].opacity_animation
    assert animation is not None
    # The first point has no authored interpolation, so the incoming segment
    # is linear. The middle point's ease-in shapes only its outgoing segment.
    assert animation.value_at(Fraction(1, 2)) == pytest.approx(0.675)
    assert animation.value_at(Fraction(3, 2)) == pytest.approx(0.98)


def test_crop_mode_selects_its_named_rect_instead_of_document_order(tmp_path: Path) -> None:
    source = tmp_path / "crop-kinds.fcpxml"
    source.write_text(
        _document(
            '''<adjust-crop mode="trim">
              <crop-rect left="1" top="2" right="3" bottom="4"/>
              <trim-rect left="10" top="20" right="30" bottom="5"/>
              <pan-rect left="6" top="7" right="8" bottom="9"/>
              <pan-rect left="9" top="8" right="7" bottom="6"/>
            </adjust-crop>'''
        ),
        encoding="utf-8",
    )

    crop = parse_fcpxml(source).spine[0].crop
    assert crop is not None
    assert [rect.kind for rect in crop.rects] == [
        "crop-rect",
        "trim-rect",
        "pan-rect",
        "pan-rect",
    ]
    assert [(rect.left, rect.top, rect.right, rect.bottom) for rect in crop.active_rects] == [
        (10.0, 20.0, 30.0, 5.0)
    ]


@pytest.mark.parametrize("mode", ["crop", "trim", "pan"])
def test_empty_active_crop_shell_is_a_valid_identity_adjustment(
    tmp_path: Path,
    mode: str,
) -> None:
    """Apple's optional crop rectangle must not make valid XML unrenderable."""

    source = tmp_path / "empty-crop-shell.fcpxml"
    source.write_text(
        _document(f'<adjust-crop mode="{mode}"/>'),
        encoding="utf-8",
    )

    crop = parse_fcpxml(source).spine[0].crop
    assert crop is not None
    assert crop.mode == mode
    assert crop.rects == ()
    assert crop.enabled is False


def test_active_unimplemented_intrinsics_are_reported_and_noops_are_not(tmp_path: Path) -> None:
    source = tmp_path / "intrinsics.fcpxml"
    source.write_text(
        _document(
            '''<adjust-corners topLeft="2 3"/>
            <adjust-rollingShutter amount="none"/>
            <adjust-colorConform conformType="conformNone" peakNitsOfPQSource="1000" peakNitsOfSDRToPQSource="100"/>
            <audio-channel-source srcCh="1" outCh="L" role="dialogue">
              <adjust-panner mode="stereo" amount="0.5"/>
            </audio-channel-source>'''
        ),
        encoding="utf-8",
    )

    compiled = compile_fcpxml(source)
    omitted = {finding.construct for finding in compiled.report.findings if finding.outcome == "omitted"}
    assert compiled.render.clips[0].corner_pin is not None
    assert compiled.render.clips[0].corner_pin.top_left == (2.0, 3.0)
    assert "adjust-corners" not in omitted
    assert "audio-channel-source" not in omitted
    assert compiled.render.audio is not None
    assert compiled.render.audio.items[0].source_channels == (1,)
    assert compiled.render.audio.items[0].output_channels == ("L",)
    assert compiled.render.audio.items[0].control_layers[-1].panner is not None
    assert "adjust-rollingShutter" not in omitted
    assert "adjust-colorConform" not in omitted


def test_split_audio_and_roles_reach_the_independent_audio_schedule(tmp_path: Path) -> None:
    source = tmp_path / "split-audio.fcpxml"
    source.write_text(
        _document(
            "",
            clip_attributes=(
                'audioStart="1/2s" audioDuration="1s" '
                'audioRole="dialogue.dialogue-1"'
            ),
        ),
        encoding="utf-8",
    )

    compiled = compile_fcpxml(source)
    omitted = {
        finding.construct
        for finding in compiled.report.findings
        if finding.outcome == "omitted"
    }
    assert "split audio edit" not in omitted
    assert "clip roles" not in omitted
    assert compiled.render.audio is not None
    item = compiled.render.audio.items[0]
    assert item.absolute_start == Fraction(1, 2)
    assert item.duration == Fraction(1)
    assert item.role is not None
    assert item.role.qualified == "dialogue.dialogue-1"
    assert compiled.source.schema_version == 2
    assert compiled.render.schema_version == 2
    assert compiled.render.story is not None
    assert compiled.render.story.root.children[0].path.endswith("asset-clip[1]")


def test_render_ir_v2_preserves_piecewise_retime_instead_of_only_its_average(
    tmp_path: Path,
) -> None:
    source = tmp_path / "piecewise.fcpxml"
    source.write_text(
        _document(
            '''<timeMap preservesPitch="1">
              <timept time="0s" value="0s" interp="linear"/>
              <timept time="1s" value="1s" interp="linear"/>
              <timept time="2s" value="1s" interp="linear"/>
            </timeMap>'''
        ),
        encoding="utf-8",
    )

    clip = compile_fcpxml(source).render.clips[0]
    assert clip.retime_map is not None
    assert [segment.kind for segment in clip.retime_map.segments] == [
        "forward",
        "freeze",
    ]
    assert clip.retime_map.timeline_to_source(Fraction(3, 2)) == Fraction(1)
