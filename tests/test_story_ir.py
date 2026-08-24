"""Experimental tests for the hierarchical static render-IR contract."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from bladeworks.core.model import SourceDocument, StoryNode, TransformAdjustment
from bladeworks.core.story_ir import (
    RenderGap,
    RenderGroup,
    RenderMedia,
    ResourceStory,
    StoryIRCycleError,
    StoryIRDepthError,
    StoryIRResolutionError,
    build_render_story,
    walk_render_nodes,
)


def _node(
    kind: str,
    path: str,
    *,
    offset: Fraction | None = Fraction(0),
    start: Fraction = Fraction(0),
    duration: Fraction = Fraction(4),
    lane: int = 0,
    ref: str | None = None,
    children: tuple[StoryNode, ...] = (),
    transform: TransformAdjustment | None = None,
) -> StoryNode:
    return StoryNode(
        kind=kind,
        path=path,
        name=path,
        ref=ref,
        lane=lane,
        offset=offset,
        start=start,
        duration=duration,
        enabled=True,
        src_enable=None,
        audio_start=None,
        audio_duration=None,
        role=None,
        video_role=None,
        audio_role=None,
        conform_type="fit",
        transform=transform,
        crop=None,
        blend_opacity=1.0,
        blend_mode=None,
        opacity_fade=None,
        volume_db=None,
        audio_fade=None,
        time_map=(),
        time_map_preserves_pitch=True,
        time_map_frame_sampling=None,
        filters=(),
        params=(),
        text_runs=(),
        text_styles={},
        multicam_sources=(),
        children=children,
        raw_xml=f"<{kind}/>",
    )


def _document(nodes: tuple[StoryNode, ...], *, duration: Fraction = Fraction(20)) -> SourceDocument:
    return SourceDocument(
        schema_version=1,
        source_path=Path("/fixture.fcpxml"),
        source_sha256="0" * 64,
        fcpxml_version="1.14",
        project_name="Hierarchy fixture",
        event_name="Tests",
        sequence_format_id="r1",
        sequence_duration=duration,
        sequence_tc_start=Fraction(0),
        formats={},
        assets={},
        effects={},
        multicams={},
        other_resources=(),
        spine=nodes,
    )


def test_nested_spine_keeps_absolute_times_and_parent_scope() -> None:
    transform = TransformAdjustment(
        position=(20.0, -10.0),
        scale=(120.0, 80.0),
        rotation=8.0,
        enabled=True,
    )
    first = _node("asset-clip", "clip/first", offset=Fraction(10), start=Fraction(20), duration=Fraction(3))
    second = _node("asset-clip", "clip/second", offset=None, start=Fraction(40), duration=Fraction(2))
    spine = _node(
        "spine",
        "clip/spine",
        offset=Fraction(10),
        start=Fraction(10),
        duration=Fraction(8),
        children=(first, second),
    )
    clip = _node(
        "clip",
        "clip",
        offset=Fraction(2),
        start=Fraction(10),
        duration=Fraction(8),
        children=(spine,),
        transform=transform,
    )

    plan = build_render_story(_document((clip,)))

    group = plan.root.children[0]
    assert isinstance(group, RenderGroup)
    assert group.absolute_start == 2
    assert group.scope.transform == transform
    nested_spine = group.children[0]
    assert isinstance(nested_spine, RenderGroup)
    assert nested_spine.child_timing == "sequential"
    assert [child.absolute_start for child in nested_spine.children] == [Fraction(2), Fraction(5)]
    assert [child.source_start for child in nested_spine.children] == [Fraction(20), Fraction(40)]
    assert all(group.id in child.ancestor_group_ids for child in nested_spine.children)


def test_source_range_clipping_updates_source_start_without_float_math() -> None:
    connected = _node(
        "video",
        "asset/video",
        offset=Fraction(9),
        start=Fraction(30),
        duration=Fraction(8),
        lane=1,
    )
    media = _node(
        "asset-clip",
        "asset",
        offset=Fraction(-2),
        start=Fraction(10),
        duration=Fraction(7),
        children=(connected,),
    )

    plan = build_render_story(_document((media,), duration=Fraction(4)))

    rendered = plan.root.children[0]
    assert isinstance(rendered, RenderMedia)
    assert rendered.was_clipped
    assert rendered.absolute_start == 0
    assert rendered.duration == 4
    assert rendered.source_start == 12
    child = rendered.connected_children[0]
    assert child.absolute_start == 0
    assert child.duration == 4
    assert child.source_start == 33
    assert isinstance(child.source_start, Fraction)


def test_compound_resource_is_clipped_to_ref_selection_and_cycles_fail() -> None:
    compound_leaf = _node(
        "asset-clip",
        "resources/c1/leaf",
        offset=Fraction(0),
        start=Fraction(0),
        duration=Fraction(10),
    )
    compound = ResourceStory(
        resource_id="c1",
        path="resources/media[@id='c1']/sequence",
        start=Fraction(0),
        duration=Fraction(10),
        story=(compound_leaf,),
    )
    reference = _node(
        "ref-clip",
        "sequence/ref",
        ref="c1",
        offset=Fraction(2),
        start=Fraction(3),
        duration=Fraction(4),
    )

    plan = build_render_story(_document((reference,)), resource_stories={"c1": compound})

    group = plan.root.children[0]
    assert isinstance(group, RenderGroup)
    assert group.resolution == "resolved"
    internal = group.children[0]
    assert isinstance(internal, RenderMedia)
    assert (internal.absolute_start, internal.duration, internal.source_start) == (Fraction(2), Fraction(4), Fraction(3))

    c1_ref = _node("ref-clip", "resources/c1/ref", ref="c2", duration=Fraction(4))
    c2_ref = _node("ref-clip", "resources/c2/ref", ref="c1", duration=Fraction(4))
    resources = {
        "c1": replace(compound, story=(c1_ref,)),
        "c2": ResourceStory("c2", "resources/c2/sequence", Fraction(0), Fraction(4), (c2_ref,)),
    }
    with pytest.raises(StoryIRCycleError, match="c1 -> c2 -> c1"):
        build_render_story(_document((replace(reference, start=Fraction(0)),)), resource_stories=resources)


def test_unresolved_compound_is_error_by_default_or_explicit_report() -> None:
    reference = _node("ref-clip", "sequence/ref", ref="missing", duration=Fraction(4))

    with pytest.raises(StoryIRResolutionError, match="unparsed media resource"):
        build_render_story(_document((reference,)))

    plan = build_render_story(_document((reference,)), unresolved_policy="report")
    placeholder = plan.root.children[0]
    assert isinstance(placeholder, RenderGroup)
    assert placeholder.resolution == "unresolved"
    assert [finding.code for finding in plan.findings] == ["compound_resource_unresolved"]


def test_sync_audition_and_transition_side_groups_preserve_ownership() -> None:
    sync_primary = _node("asset-clip", "sync/primary", duration=Fraction(6))
    sync_connected = _node("video", "sync/connected", duration=Fraction(2), lane=1)
    sync = _node("sync-clip", "sync", duration=Fraction(6), children=(sync_primary, sync_connected))
    choice_one = _node("asset-clip", "audition/one", offset=Fraction(0), duration=Fraction(4))
    choice_two = _node("asset-clip", "audition/two", offset=Fraction(0), duration=Fraction(4))
    audition = _node(
        "audition",
        "audition",
        offset=Fraction(6),
        duration=Fraction(4),
        children=(choice_one, choice_two),
    )
    transition = _node("transition", "transition", offset=Fraction(5), duration=Fraction(2))

    plan = build_render_story(_document((sync, transition, audition), duration=Fraction(12)))

    sync_group, audition_group = plan.root.children
    assert isinstance(sync_group, RenderGroup)
    assert isinstance(audition_group, RenderGroup)
    assert len(sync_group.children) == 1
    assert len(sync_group.connected_children) == 1
    assert sync_group.id in sync_group.children[0].ancestor_group_ids
    assert sync_group.id not in sync_group.connected_children[0].ancestor_group_ids
    assert sync_group.connected_children[0].lane == 1
    assert len(audition_group.children) == 1
    assert len(audition_group.inactive_children) == 1
    assert audition_group.children[0].path == "audition/one"
    assert audition_group.inactive_children[0].path == "audition/two"
    assert any(finding.code == "audition_choice_inactive" for finding in plan.findings)

    transition_group = plan.transitions[0]
    assert transition_group.kind == "transition"
    assert transition_group.absolute_start == 5
    outgoing, incoming = transition_group.children
    assert isinstance(outgoing, RenderGroup) and isinstance(incoming, RenderGroup)
    assert outgoing.children[0].id == sync_group.id
    assert incoming.children[0].id == audition_group.id


def test_max_depth_and_document_order_are_deterministic() -> None:
    leaf = _node("gap", "deep/gap", duration=Fraction(2))
    inner = _node("clip", "deep/inner", duration=Fraction(2), children=(leaf,))
    outer = _node("clip", "deep/outer", duration=Fraction(2), children=(inner,))
    document = _document((outer,), duration=Fraction(2))

    with pytest.raises(StoryIRDepthError, match="max_depth=2"):
        build_render_story(document, max_depth=2)

    first = build_render_story(document)
    second = build_render_story(document)
    first_nodes = walk_render_nodes(first.root, include_inactive=True)
    second_nodes = walk_render_nodes(second.root, include_inactive=True)
    assert [(node.id, node.document_order) for node in first_nodes] == [
        (node.id, node.document_order) for node in second_nodes
    ]
    assert any(isinstance(node, RenderGap) for node in first_nodes)
