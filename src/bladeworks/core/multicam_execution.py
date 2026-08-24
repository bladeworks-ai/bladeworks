"""Resolve explicit Final Cut multicam choices as renderable A/V sources.

Architecture map
================

``SourceDocument`` plus the story-container catalog
    -> find each deferred ``mc-clip`` group on the absolute project timeline
    -> validate one explicit choice independently for video and audio
    -> expose selected angle stories as one ``RenderableAVSource``
    -> compile source audio through the ordinary clip-instance timing boundary
    -> return one immutable ``MulticamExecutionPlan``

The selected stories remain source-local. The compiler recursively composes
them, then the ordinary clip-instance executor applies the outer trim, split
audio range, retime, controls, geometry, and placement. No synthetic
``mc-clip -> clip -> spine`` rewrite and no intermediate movie exist.

Important invariants
--------------------

* Video and audio choices are resolved independently.  Missing, duplicate, or
  unknown choices are never replaced with the first angle.
* Final Cut's stored synchronization offsets are reused exactly as
  ``Fraction`` values.  This module performs no waveform or timecode sync.
* Angle stories are built by the shared hierarchy engine, so file clips,
  groups, compound references, connected layers, gaps, and auditions have one
  ownership model.
* The source definition contains no timeline-instance timing. One exact
  clip-instance map independently plans video and split audio above it.
* No FFmpeg strings are emitted here.  The root integrator gives ``story`` to
  the shared video compositor and ``audio`` to the stock-FFmpeg audio engine.

Main callers:
- The root compiler after ``build_story_container_plan`` and before flattening
  any video leaves.
- Experimental MC-1 tests and the core compatibility audit.

Why this exists:
Multicam is a media source whose pixels and samples happen to come from a
recursively composed graph. Keeping selection here and editing above the
source prevents multicam-only timing behavior from diverging again.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal, Mapping, Optional
import xml.etree.ElementTree as ET

from .audio_ir import AudioRenderPlan, compile_audio_ir
from .model import (
    MulticamAngle,
    MulticamSource,
    SequenceFormatContext,
    SourceDocument,
    StoryNode,
)
from .parser import _parse_story_node
from .render_sources import RenderableAVSource, resolve_compound_source
from .story_containers import (
    StoryContainerPlan,
    build_story_container_plan,
)
from .story_ir import (
    RenderGroup,
    RenderStoryPlan,
    ResourceStory,
    walk_render_nodes,
)


StreamKind = Literal["video", "audio"]
FindingDisposition = Literal["inactive", "not_implemented_yet"]


class MulticamExecutionError(ValueError):
    """Base class for invalid or unsupported selected-multicam execution."""


class MulticamSelectionError(MulticamExecutionError):
    """Raised when explicit angle choices are unknown or ambiguous."""


class MulticamIntegrationError(MulticamExecutionError):
    """Raised when shared hierarchy output cannot be grafted safely."""


@dataclass(frozen=True)
class SelectedAngleChoice:
    """One explicit stream choice from an ``mc-source`` child."""

    stream: StreamKind
    source_index: int
    source_path: str
    angle_id: str
    angle_name: Optional[str]
    source: MulticamSource
    angle: MulticamAngle


@dataclass(frozen=True)
class MulticamExecutionFinding:
    """One honest inactive or blocked multicam behavior."""

    code: str
    path: str
    stream: StreamKind
    disposition: FindingDisposition
    detail: str


@dataclass(frozen=True)
class MulticamExecutionPlan:
    """The central integration product for all selected multicam edits.

    ``story`` retains timeline ownership and connected siblings. ``sources``
    holds the file/compound/multicam definitions that the ordinary compiler
    consumes at each reference path. ``audio`` is the project-wide schedule
    produced from that same catalog.
    """

    story: RenderStoryPlan
    audio: AudioRenderPlan
    sources: Mapping[str, RenderableAVSource]
    findings: tuple[MulticamExecutionFinding, ...]


def build_multicam_execution_plan(
    source: SourceDocument,
    *,
    container_plan: Optional[StoryContainerPlan] = None,
    max_depth: int = 32,
) -> MulticamExecutionPlan:
    """Resolve every active project/compound ``mc-clip`` without flattening.

    Main callers:
    - The root compiler's shared source-resolution seam.

    The caller may pass its already-built ``StoryContainerPlan`` so compound
    parsing, cycle checks, and hierarchy construction happen exactly once.
    """

    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    containers = container_plan or build_story_container_plan(
        source,
        max_depth=max_depth,
    )
    node_by_path = _source_node_index(source, containers.resources.stories)
    findings: list[MulticamExecutionFinding] = []
    sources: dict[str, RenderableAVSource] = {}

    # Compound references and multicam selections share the same catalog.
    # The source definition is independent of the timeline edit; each path is
    # merely an instance key consumed later by the common clip executor.
    for node in node_by_path.values():
        if node.kind != "ref-clip":
            continue
        resource = containers.resources.stories.get(node.ref or "")
        if resource is None:
            raise MulticamIntegrationError(
                f"{node.path} references unresolved compound source {node.ref!r}"
            )
        has_video, has_audio = _story_stream_capabilities(
            source,
            resource.story,
            containers.resources.stories,
            resource_chain=(resource.resource_id,),
        )
        sources[node.path] = resolve_compound_source(
            resource,
            has_video=has_video,
            has_audio=has_audio,
        )

    deferred_groups = tuple(
        node
        for node in walk_render_nodes(containers.story.root, include_inactive=False)
        if isinstance(node, RenderGroup) and node.kind == "mc-clip"
    )
    for group in deferred_groups:
        source_node = node_by_path.get(group.path)
        if source_node is None or source_node.kind != "mc-clip":
            raise MulticamIntegrationError(
                f"render group {group.id!r} has no unique mc-clip source at {group.path!r}"
            )
        choices = _selected_choices(source, source_node)
        sources[source_node.path] = _resolve_multicam_source(
            source, source_node, choices
        )
        if choices.get("video") is None:
            findings.append(
                MulticamExecutionFinding(
                    code="multicam_video_not_selected",
                    path=group.path,
                    stream="video",
                    disposition="inactive",
                    detail=(
                        "no explicit video mc-source is active; the interval is "
                        "transparent and no angle was guessed"
                    ),
                )
            )
        if choices.get("audio") is None:
            findings.append(
                MulticamExecutionFinding(
                    code="multicam_audio_not_selected",
                    path=group.path,
                    stream="audio",
                    disposition="inactive",
                    detail=(
                        "no explicit audio mc-source is active; the interval is "
                        "silent and no angle was guessed"
                    ),
                )
            )

    audio = compile_audio_ir(
        source,
        resource_stories=containers.resources.stories,
        render_sources=sources,
    )
    return MulticamExecutionPlan(
        story=replace(
            containers.story,
            findings=tuple(
                finding
                for finding in containers.story.findings
                if finding.code != "multicam_selection_deferred"
            ),
        ),
        audio=audio,
        sources=sources,
        findings=tuple(findings),
    )


def _resolve_multicam_source(
    source: SourceDocument,
    node: StoryNode,
    choices: Mapping[StreamKind, SelectedAngleChoice],
) -> RenderableAVSource:
    """Expose selected multicam angles as one ordinary source definition."""

    if not node.ref or node.ref not in source.multicams:
        raise MulticamSelectionError(
            f"{node.path} references unknown multicam resource {node.ref!r}"
        )
    resource = source.multicams[node.ref]
    if resource.duration is None:
        raise MulticamSelectionError(
            f"{node.path} multicam resource has no finite duration"
        )
    format_resource = source.formats.get(resource.format_id)
    if format_resource is None:
        raise MulticamSelectionError(
            f"{node.path} multicam resource references unknown format "
            f"{resource.format_id!r}"
        )
    video = choices.get("video")
    audio = choices.get("audio")
    return RenderableAVSource(
        id=(
            f"multicam:{resource.id}:video={video.angle_id if video else 'none'}:"
            f"audio={audio.angle_id if audio else 'none'}"
        ),
        kind="multicam",
        resource_id=resource.id,
        source_start=resource.tc_start,
        duration=resource.duration,
        format_context=SequenceFormatContext.from_resource(format_resource),
        has_video=video is not None,
        has_audio=audio is not None,
        video_story=video.angle.story if video else (),
        audio_story=audio.angle.story if audio else (),
        video_scope=(
            _mc_source_scope_node(
                video,
                selection_start=resource.tc_start,
                selection_duration=resource.duration,
                children=(),
            )
            if video
            else None
        ),
        audio_scope=(
            _mc_source_scope_node(
                audio,
                selection_start=resource.tc_start,
                selection_duration=resource.duration,
                children=(),
            )
            if audio
            else None
        ),
        video_angle_id=video.angle_id if video else None,
        audio_angle_id=audio.angle_id if audio else None,
    )


def _selected_choices(
    source: SourceDocument,
    node: StoryNode,
) -> dict[StreamKind, SelectedAngleChoice]:
    """Return explicit video/audio choices and reject every ambiguity."""

    if not node.ref or node.ref not in source.multicams:
        raise MulticamSelectionError(
            f"{node.path} references unknown multicam resource {node.ref!r}"
        )
    clip_enable = node.src_enable or "all"
    if clip_enable not in {"all", "audio", "video"}:
        raise MulticamSelectionError(
            f"{node.path} has invalid srcEnable {clip_enable!r}"
        )
    clip_streams: set[StreamKind] = (
        {"video", "audio"} if clip_enable == "all" else {clip_enable}  # type: ignore[assignment]
    )
    resource = source.multicams[node.ref]
    angles = {angle.angle_id: angle for angle in resource.angles}
    selected: dict[StreamKind, SelectedAngleChoice] = {}
    for index, source_choice in enumerate(node.multicam_sources, start=1):
        angle = angles.get(source_choice.angle_id)
        if angle is None:
            raise MulticamSelectionError(
                f"{node.path}/mc-source[{index}] references unknown angleID "
                f"{source_choice.angle_id!r}"
            )
        source_streams: set[StreamKind]
        if source_choice.src_enable == "all":
            source_streams = {"video", "audio"}
        elif source_choice.src_enable == "none":
            source_streams = set()
        else:
            source_streams = {source_choice.src_enable}  # type: ignore[assignment]
        for stream in clip_streams & source_streams:
            if stream in selected:
                raise MulticamSelectionError(
                    f"{node.path} selects more than one {stream} angle; "
                    "an explicit unique choice is required"
                )
            selected[stream] = SelectedAngleChoice(
                stream=stream,
                source_index=index,
                source_path=f"{node.path}/mc-source[{index}]",
                angle_id=angle.angle_id,
                angle_name=angle.name,
                source=source_choice,
                angle=angle,
            )
    return selected


def _mc_source_scope_node(
    choice: SelectedAngleChoice,
    *,
    selection_start: Fraction,
    selection_duration: Fraction,
    children: tuple[StoryNode, ...],
) -> StoryNode:
    """Parse one raw mc-source through the shared intrinsic parser."""

    try:
        element = ET.fromstring(choice.source.raw_xml)
    except ET.ParseError as exc:
        raise MulticamSelectionError(
            f"{choice.source_path} contains malformed preserved XML"
        ) from exc
    element.set("duration", _time_text(selection_duration))
    element.set("start", _time_text(selection_start))
    element.set("offset", _time_text(selection_start))
    parsed = _parse_story_node(element, choice.source_path)
    return replace(
        parsed,
        kind="clip",
        ref=None,
        lane=0,
        offset=selection_start,
        start=selection_start,
        duration=selection_duration,
        audio_start=None,
        audio_duration=None,
        multicam_sources=(),
        children=children,
        raw_xml=choice.source.raw_xml,
    )


def _source_node_index(
    source: SourceDocument,
    resource_stories: Mapping[str, ResourceStory],
) -> Mapping[str, StoryNode]:
    index: dict[str, StoryNode] = {}

    def visit(nodes: tuple[StoryNode, ...]) -> None:
        for node in nodes:
            existing = index.get(node.path)
            if existing is not None and existing is not node:
                raise MulticamIntegrationError(
                    f"source path {node.path!r} is not unique"
                )
            index[node.path] = node
            visit(node.children)

    visit(source.spine)
    for resource in resource_stories.values():
        visit(resource.story)
    return index


def _story_stream_capabilities(
    source: SourceDocument,
    nodes: tuple[StoryNode, ...],
    resources: Mapping[str, ResourceStory],
    *,
    resource_chain: tuple[str, ...],
) -> tuple[bool, bool]:
    """Prove whether a recursively composed source can emit video/audio."""

    has_video = False
    has_audio = False
    for node in nodes:
        enabled = node.enabled and node.src_enable != "none"
        if not enabled:
            continue
        if node.kind in {"asset-clip", "video", "audio"} and node.ref:
            asset = source.assets.get(node.ref)
            if asset is not None:
                has_video = has_video or (
                    asset.has_video
                    and node.kind != "audio"
                    and node.src_enable != "audio"
                )
                has_audio = has_audio or (
                    asset.has_audio
                    and node.kind != "video"
                    and node.src_enable != "video"
                )
        elif node.kind in {"title", "caption"}:
            has_video = True
        elif node.kind == "ref-clip":
            resource = resources.get(node.ref or "")
            if resource is None:
                raise MulticamIntegrationError(
                    f"{node.path} references unknown compound source {node.ref!r}"
                )
            if resource.resource_id in resource_chain:
                chain = " -> ".join((*resource_chain, resource.resource_id))
                raise MulticamIntegrationError(
                    f"recursive render source reference: {chain}"
                )
            child_video, child_audio = _story_stream_capabilities(
                source,
                resource.story,
                resources,
                resource_chain=(*resource_chain, resource.resource_id),
            )
            has_video = has_video or child_video
            has_audio = has_audio or child_audio
        elif node.kind == "mc-clip":
            choices = _selected_choices(source, node)
            video = choices.get("video")
            audio = choices.get("audio")
            if video is not None:
                child_video, _ = _story_stream_capabilities(
                    source,
                    video.angle.story,
                    resources,
                    resource_chain=resource_chain,
                )
                has_video = has_video or child_video
            if audio is not None:
                _, child_audio = _story_stream_capabilities(
                    source,
                    audio.angle.story,
                    resources,
                    resource_chain=resource_chain,
                )
                has_audio = has_audio or child_audio
        if node.children:
            child_video, child_audio = _story_stream_capabilities(
                source,
                node.children,
                resources,
                resource_chain=resource_chain,
            )
            has_video = has_video or child_video
            has_audio = has_audio or child_audio
    return has_video, has_audio


def _time_text(value: Fraction) -> str:
    if value.denominator == 1:
        return f"{value.numerator}s"
    return f"{value.numerator}/{value.denominator}s"
