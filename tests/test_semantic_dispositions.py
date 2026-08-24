"""Regression contracts for authored video operations lowered to identity/cut.

Architecture map
================

Minimal FCPXML fixture
    -> compiler capability resolution
    -> complete ``RenderClip.semantic_effects`` authored order
    -> unchanged ``RenderClip.effects`` executable CPU subset

Storyline transition topology
    -> preserved adjacent story IDs
    -> unsupported hard cut marker
    -> no transition handles attached to either clip

These tests intentionally stop before FFmpeg graph construction.  Their job is
to prove that the shared semantic adapter can see an authored omission without
changing the established CPU reference graph.
"""

from __future__ import annotations

from pathlib import Path

from bladeworks.core.compiler import compile_fcpxml


NEGATIVE_UID = (
    ".../Effects.localized/Basics.localized/Negative.localized/Negative.moef"
)
THRESHOLD_UID = (
    ".../Effects.localized/Basics.localized/Threshold.localized/Threshold.moef"
)
MAGNETIC_MASK_UID = "FFAddAlphaEffectID"


def _write_source(
    tmp_path: Path,
    *,
    effect_resources: str,
    spine: str,
) -> Path:
    """Write one self-contained compiler fixture with a readable media binding."""

    media = tmp_path / "source.mov"
    media.write_bytes(b"semantic disposition fixture")
    source = tmp_path / "semantic-dispositions.fcpxml"
    source.write_text(
        f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="fmt" frameDuration="1/30s" width="96" height="64"
            colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="asset" start="0s" duration="5s" hasVideo="1"
           hasAudio="0" format="fmt">
      <media-rep kind="original-media" src="{media.as_uri()}"/>
    </asset>
    {effect_resources}
  </resources>
  <library><event name="Semantic"><project name="Semantic dispositions">
    <sequence format="fmt" duration="4s"><spine>{spine}</spine></sequence>
  </project></event></library>
</fcpxml>''',
        encoding="utf-8",
    )
    return source


def test_magnetic_mask_identity_retains_authored_effect_order_only_semantically(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path,
        effect_resources=f'''
          <effect id="negative" name="Negative" uid="{NEGATIVE_UID}"/>
          <effect id="magnetic" name="Magnetic Mask"
                  uid="{MAGNETIC_MASK_UID}"/>
          <effect id="threshold" name="Threshold" uid="{THRESHOLD_UID}"/>
        ''',
        spine='''
          <asset-clip ref="asset" offset="0s" start="0s" duration="4s">
            <filter-video ref="negative"/>
            <filter-video ref="magnetic"/>
            <filter-video ref="threshold"/>
          </asset-clip>
        ''',
    )

    compiled = compile_fcpxml(source)
    clip = compiled.render.clips[0]

    assert [effect.name for effect in clip.semantic_effects] == [
        "Negative",
        "Magnetic Mask",
        "Threshold",
    ]
    assert [effect.name for effect in clip.effects] == ["Negative", "Threshold"]
    magnetic = clip.semantic_effects[1]
    assert magnetic.execution == "identity"
    assert magnetic.handler is None
    assert magnetic.uid == MAGNETIC_MASK_UID
    assert magnetic.path.endswith("/filter-video[2]")
    assert magnetic.omission_reason == (
        "Magnetic Mask is explicitly outside the bounded portable mask prototype"
    )
    assert any(
        finding.fcpxml_path == magnetic.path
        and finding.construct == "Magnetic Mask"
        and finding.outcome == "omitted"
        for finding in compiled.report.findings
    )


def test_enabled_unknown_effect_is_an_explicit_semantic_identity(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path,
        effect_resources='''
          <effect id="unknown" name="Unknown Enabled Effect"
                  uid="example.invalid/unknown-effect"/>
        ''',
        spine='''
          <asset-clip ref="asset" offset="0s" start="0s" duration="4s">
            <filter-video ref="unknown"/>
          </asset-clip>
        ''',
    )

    clip = compile_fcpxml(source).render.clips[0]

    assert clip.effects == ()
    assert len(clip.semantic_effects) == 1
    ignored = clip.semantic_effects[0]
    assert ignored.execution == "identity"
    assert ignored.handler is None
    assert ignored.capability_id is None
    assert ignored.omission_reason == (
        "unknown filter omitted; underlying clip remains"
    )


def test_unsupported_transition_keeps_adjacent_story_ids_without_handles(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path,
        effect_resources='''
          <effect id="mystery-transition" name="Mystery Transition"
                  uid="example.invalid/mystery-transition"/>
        ''',
        spine='''
          <asset-clip ref="asset" offset="0s" start="0s" duration="2s"/>
          <transition name="Mystery Transition" offset="3/2s" duration="1s">
            <filter-video ref="mystery-transition"/>
          </transition>
          <asset-clip ref="asset" offset="2s" start="2s" duration="2s"/>
        ''',
    )

    compiled = compile_fcpxml(source)
    outgoing, incoming = compiled.render.clips
    transition = compiled.render.transitions[0]

    assert transition.handler is None
    assert transition.capability_id is None
    assert transition.portable_status == "unsupported"
    assert transition.omission_reason == "unknown transition becomes a hard cut"
    assert transition.previous_story_id == outgoing.id
    assert transition.next_story_id == incoming.id
    assert outgoing.transition_in is None
    assert outgoing.transition_out is None
    assert incoming.transition_in is None
    assert incoming.transition_out is None
    assert outgoing.absolute_start == 0
    assert outgoing.duration == 2
    assert incoming.absolute_start == 2
    assert incoming.duration == 2
    assert any(
        finding.fcpxml_path == transition.path
        and finding.outcome == "omitted"
        and "hard cut" in finding.disposition
        for finding in compiled.report.findings
    )


def test_unsupported_motion_title_has_explicit_transparent_disposition(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path,
        effect_resources='''
          <effect id="motion-title" name="Unportable Motion Title"
                  uid=".../Titles.localized/Unportable.moti"/>
        ''',
        spine='''
          <title ref="motion-title" offset="0s" duration="4s">
            <text>Visible only in Final Cut</text>
          </title>
        ''',
    )

    clip = compile_fcpxml(source).render.clips[0]

    assert not clip.has_video
    assert clip.video_disposition is not None
    assert clip.video_disposition.execution == "omit_transparent"
    assert clip.video_disposition.portable_status == "unsupported"
    assert clip.video_disposition.construct == "Unportable Motion Title"
    assert clip.video_disposition.uid == ".../Titles.localized/Unportable.moti"
    assert clip.video_disposition.reason == (
        "opaque Motion title has no calibrated portable adapter"
    )


def test_missing_video_media_has_explicit_placeholder_disposition(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path,
        effect_resources="",
        spine='''
          <asset-clip ref="asset" offset="0s" start="0s" duration="4s"/>
        ''',
    )
    (tmp_path / "source.mov").unlink()

    clip = compile_fcpxml(source).render.clips[0]

    assert clip.has_video
    assert clip.media_path is None
    assert clip.video_disposition is not None
    assert clip.video_disposition.execution == "composite"
    assert clip.missing_media_locators == (str(tmp_path / "source.mov"),)


def test_disabled_video_has_authored_disabled_disposition(
    tmp_path: Path,
) -> None:
    source = _write_source(
        tmp_path,
        effect_resources="",
        spine='''
          <asset-clip ref="asset" offset="0s" start="0s" duration="4s"
                      enabled="0"/>
        ''',
    )

    clip = compile_fcpxml(source).render.clips[0]

    assert not clip.enabled
    assert clip.video_disposition is not None
    assert clip.video_disposition.execution == "authored_disabled"
    assert clip.video_disposition.reason is None
