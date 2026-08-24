"""Experimental tests for compound, synchronized, clip, and audition stories."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess

import pytest

from bladeworks.core.audio_ir import compile_audio_ir
from bladeworks.core.compiler import compile_fcpxml
from bladeworks.core.errors import FCPXMLCompileError
from bladeworks.core.parser import parse_fcpxml
from bladeworks.core.story_containers import (
    StoryContainerReferenceError,
    StoryContainerResourceError,
    build_story_container_plan,
    parse_compound_resource_stories,
)
from bladeworks.core.story_ir import (
    RenderGroup,
    RenderMedia,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = Path(__file__).parent / "fixtures" / "story_containers_1_14.fcpxml"
_DTD = Path(__file__).parent / "FCPXMLv1_14.dtd"


def _source():
    return parse_fcpxml(_FIXTURE)


def test_genuine_fixture_is_fcpxml_1_14_dtd_valid() -> None:
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


def test_compound_sequence_parses_and_ref_window_is_exact() -> None:
    plan = build_story_container_plan(_source())

    resource = plan.resources.stories["c1"]
    assert (resource.start, resource.duration) == (Fraction(10), Fraction(8))
    assert [node.kind for node in resource.story] == ["asset-clip", "sync-clip"]

    reference = plan.story.root.children[0]
    assert isinstance(reference, RenderGroup)
    assert reference.kind == "ref-clip"
    assert reference.child_timing == "resource"
    assert reference.scope.transform is not None
    assert reference.scope.transform.position == (20.0, -10.0)

    first, synchronized = reference.children
    assert isinstance(first, RenderMedia)
    assert (first.absolute_start, first.duration, first.source_start) == (
        Fraction(0),
        Fraction(2),
        Fraction(22),
    )
    assert isinstance(synchronized, RenderGroup)
    assert (synchronized.absolute_start, synchronized.duration) == (
        Fraction(2),
        Fraction(2),
    )
    assert len(synchronized.children) == 1
    assert len(synchronized.connected_children) == 1


def test_live_drawing_is_preserved_as_one_specific_finding() -> None:
    plan = build_story_container_plan(_source())

    assert [(finding.code, finding.disposition) for finding in plan.story.findings] == [
        ("live_drawing_not_implemented", "not_implemented_yet")
    ]
    reference = plan.story.root.children[0]
    assert isinstance(reference, RenderGroup)
    assert len(reference.connected_children) == 1
    drawing = reference.connected_children[0]
    assert isinstance(drawing, RenderGroup)
    assert drawing.kind == "unknown:live-drawing"
    assert drawing.resolution == "unresolved"


def test_sync_role_hooks_are_typed_and_deterministic() -> None:
    plan = build_story_container_plan(_source())

    hooks = plan.audio_hooks.synchronized_sources
    assert [hook.source_id for hook in hooks] == ["storyline", "connected"]
    assert [hook.role_selectors[0].role for hook in hooks] == ["dialogue", "music"]
    assert all(hook.role_selectors[0].enabled for hook in hooks)
    compound = plan.audio_hooks.compound_references
    assert len(compound) == 1
    assert (
        compound[0].resource_id,
        compound[0].selection_start,
        compound[0].selection_duration,
        compound[0].use_audio_subroles,
    ) == ("c1", Fraction(12), Fraction(4), True)


def test_compound_audio_source_executes_exact_trim_and_sync_sources() -> None:
    source = _source()
    catalog = parse_compound_resource_stories(source)
    audio = compile_audio_ir(source, resource_stories=catalog.stories)
    assert [item.name for item in audio.items] == [
        "Compound dialogue",
        "Synchronized picture",
        "Connected music",
    ]
    assert [item.absolute_start for item in audio.items] == [
        Fraction(0),
        Fraction(2),
        Fraction(2),
    ]
    assert [item.duration for item in audio.items] == [
        Fraction(2),
        Fraction(2),
        Fraction(2),
    ]
    assert [item.source_start for item in audio.items] == [
        Fraction(22),
        Fraction(24),
        Fraction(8),
    ]
    assert all(
        any(
            layer.path == "spine/ref-clip[1]/audio-role-source[1]"
            for layer in item.control_layers
        )
        for item in audio.items[:2]
    )


def test_audition_uses_first_choice_and_preserves_inactive_alternative() -> None:
    source = _source()
    active = replace(
        parse_compound_resource_stories(source).stories["c1"].story[0],
        path="spine/audition[1]/asset-clip[1]",
        name="Active choice",
        offset=Fraction(0),
        start=Fraction(0),
        duration=Fraction(2),
        children=(),
    )
    inactive = replace(
        active,
        path="spine/audition[1]/asset-clip[2]",
        name="Inactive choice",
        ref="a2",
    )
    audition = replace(
        active,
        kind="audition",
        path="spine/audition[1]",
        name="Audition",
        ref=None,
        children=(active, inactive),
        raw_xml="<audition/>",
    )
    audition_source = replace(
        source,
        sequence_duration=Fraction(2),
        spine=(audition,),
    )

    plan = build_story_container_plan(audition_source)
    group = plan.story.root.children[0]
    assert isinstance(group, RenderGroup)
    assert group.children[0].name == "Active choice"
    assert group.inactive_children[0].name == "Inactive choice"
    assert [finding.code for finding in plan.story.findings] == [
        "audition_choice_inactive"
    ]
    audio = compile_audio_ir(
        audition_source,
        resource_stories=plan.resources.stories,
    )
    assert [item.name for item in audio.items] == ["Active choice"]
    assert [finding.code for finding in audio.findings] == [
        "audition_audio_inactive"
    ]


def test_missing_and_out_of_bounds_compound_references_fail_closed() -> None:
    source = _source()
    reference = source.spine[0]

    missing = replace(source, spine=(replace(reference, ref="missing"),))
    with pytest.raises(StoryContainerReferenceError, match="without an inline sequence"):
        build_story_container_plan(missing)

    out_of_bounds = replace(
        source,
        spine=(replace(reference, start=Fraction(17), duration=Fraction(2)),),
        sequence_duration=Fraction(2),
    )
    with pytest.raises(StoryContainerReferenceError, match="source range"):
        build_story_container_plan(out_of_bounds)


def test_zero_duration_compound_is_a_controlled_compile_error(tmp_path: Path) -> None:
    source = tmp_path / "zero-duration-compound.fcpxml"
    source.write_text(
        _FIXTURE.read_text().replace(
            '<sequence format="r1" duration="8s" tcStart="10s"',
            '<sequence format="r1" duration="0s" tcStart="10s"',
            1,
        )
    )

    with pytest.raises(FCPXMLCompileError, match="duration must be positive"):
        compile_fcpxml(source)


def test_unknown_story_kind_fails_instead_of_rendering_its_children() -> None:
    source = _source()
    unknown = replace(
        source.spine[0],
        kind="unknown:future-container",
        ref=None,
        children=(),
    )

    with pytest.raises(StoryContainerResourceError, match="future-container"):
        build_story_container_plan(replace(source, spine=(unknown,)))
