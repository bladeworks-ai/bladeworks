"""Build the hierarchical, absolute-time story graph used by render IR v2.

Architecture map
================

``StoryNode`` trees and separately parsed compound resources
    -> exact parent/source-domain time mapping
    -> source-range clipping
    -> ``RenderGroup`` / ``RenderMedia`` / ``RenderGap`` hierarchy
    -> transition-side composition views and explicit resolution findings

The main tree owns each source item exactly once.  Transition-side groups are
immutable views of the adjacent main-tree items; they do not flatten or take
ownership away from the main tree.  Every stored timeline coordinate is a
``Fraction`` and is already absolute in the project timeline.

Important invariants
--------------------

* Container scope is never copied onto a child.  ``RenderGroup.scope`` remains
  the one place that owns the container's transform, opacity, filters, and
  audio controls.
* Compound resource resolution is injected explicitly.  The current parser
  preserves compound ``<media><sequence>`` resources as raw XML, so this module
  refuses to pretend an unresolved ``ref-clip`` is ordinary media.
* Recursive resources are rejected with a readable resource chain.  A bounded
  nesting depth also protects the compiler from adversarial documents.
* Source order is stable.  ``document_order`` is assigned before descending
  into a source item and is identical across repeated builds.

Main callers:
- The render-IR v2 integrator, after ``parse_fcpxml`` and compound-resource
  parsing have completed.

Why this exists:
The v1 compiler flattens nested clips while it resolves them.  That loses the
boundary at which Final Cut applies a parent transform, opacity, filter, or
audio control.  This module freezes that boundary before FFmpeg construction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal, Mapping, Optional, TypeAlias

from .model import (
    CropAdjustment,
    FadeEnvelope,
    FilterInstance,
    MaskedFilterInstance,
    Parameter,
    PreservedAdjustment,
    SequenceFormatContext,
    SourceDocument,
    StoryNode,
    TimeMapPoint,
    TransformAdjustment,
)


UnresolvedPolicy: TypeAlias = Literal["error", "report"]
ResolutionState: TypeAlias = Literal["not_applicable", "resolved", "deferred", "unresolved"]
ChildTiming: TypeAlias = Literal["sequential", "anchored", "resource", "choice", "transition_side"]


class StoryIRBuildError(ValueError):
    """Base class for hierarchy construction failures."""


class StoryIRResolutionError(StoryIRBuildError):
    """Raised when an external story resource cannot be resolved safely."""


class StoryIRCycleError(StoryIRBuildError):
    """Raised when compound resources eventually reference themselves."""


class StoryIRDepthError(StoryIRBuildError):
    """Raised when source nesting exceeds the configured safety bound."""


class StoryIRTransitionError(StoryIRBuildError):
    """Raised when a transition does not have both storyline participants."""


@dataclass(frozen=True)
class RenderScope:
    """Controls owned by one item or container, before backend translation.

    Main callers:
    - ``StoryIRBuilder._scope`` for every parsed story item.

    Why this exists:
    Keeping scope separate from timing makes it difficult for a later
    compositor to accidentally apply a parent adjustment to only one leaf.
    """

    transform: Optional[TransformAdjustment] = None
    crop: Optional[CropAdjustment] = None
    conform_type: str = "fit"
    opacity: float = 1.0
    blend_mode: Optional[str] = None
    opacity_fade: Optional[FadeEnvelope] = None
    filters: tuple[FilterInstance | MaskedFilterInstance, ...] = ()
    params: tuple[Parameter, ...] = ()
    preserved_adjustments: tuple[PreservedAdjustment, ...] = ()
    audio_start: Optional[Fraction] = None
    audio_duration: Optional[Fraction] = None
    volume_db: Optional[float] = None
    audio_fade: Optional[FadeEnvelope] = None
    role: Optional[str] = None
    video_role: Optional[str] = None
    audio_role: Optional[str] = None
    src_enable: Optional[str] = None
    time_map: tuple[TimeMapPoint, ...] = ()
    time_map_preserves_pitch: bool = True
    time_map_frame_sampling: Optional[str] = None


@dataclass(frozen=True)
class RenderMedia:
    """One renderable source item on the absolute timeline.

    ``connected_children`` retain anchored content without claiming that the
    media item's own transform/filter scope applies to those children.
    """

    id: str
    kind: str
    path: str
    instance_path: str
    name: Optional[str]
    ref: Optional[str]
    absolute_start: Fraction
    duration: Fraction
    source_start: Fraction
    source_duration: Fraction
    unclipped_absolute_start: Fraction
    unclipped_duration: Fraction
    lane: int
    document_order: int
    enabled: bool
    scope: RenderScope
    ancestor_group_ids: tuple[str, ...]
    connected_children: tuple["RenderNode", ...] = ()

    @property
    def end(self) -> Fraction:
        return self.absolute_start + self.duration

    @property
    def was_clipped(self) -> bool:
        return self.absolute_start != self.unclipped_absolute_start or self.duration != self.unclipped_duration


@dataclass(frozen=True)
class RenderGap:
    """One transparent/silent interval with separately anchored children."""

    id: str
    path: str
    instance_path: str
    name: Optional[str]
    absolute_start: Fraction
    duration: Fraction
    source_start: Fraction
    unclipped_absolute_start: Fraction
    unclipped_duration: Fraction
    lane: int
    document_order: int
    enabled: bool
    scope: RenderScope
    ancestor_group_ids: tuple[str, ...]
    connected_children: tuple["RenderNode", ...] = ()

    @property
    def end(self) -> Fraction:
        return self.absolute_start + self.duration

    @property
    def was_clipped(self) -> bool:
        return self.absolute_start != self.unclipped_absolute_start or self.duration != self.unclipped_duration


@dataclass(frozen=True)
class RenderGroup:
    """A composition boundary whose children all use absolute project time.

    ``children`` are the content composed inside this scope.
    ``connected_children`` are temporally anchored to the group but spatially
    composed beside it, so the group's transform does not implicitly affect
    them. ``inactive_children`` preserve unselected audition choices without
    making them renderable.
    """

    id: str
    kind: str
    path: str
    instance_path: str
    name: Optional[str]
    ref: Optional[str]
    absolute_start: Fraction
    duration: Fraction
    source_start: Fraction
    unclipped_absolute_start: Fraction
    unclipped_duration: Fraction
    lane: int
    document_order: int
    enabled: bool
    scope: RenderScope
    child_timing: ChildTiming
    resolution: ResolutionState
    ancestor_group_ids: tuple[str, ...]
    resource_chain: tuple[str, ...] = ()
    children: tuple["RenderNode", ...] = ()
    connected_children: tuple["RenderNode", ...] = ()
    inactive_children: tuple["RenderNode", ...] = ()

    @property
    def end(self) -> Fraction:
        return self.absolute_start + self.duration

    @property
    def was_clipped(self) -> bool:
        return self.absolute_start != self.unclipped_absolute_start or self.duration != self.unclipped_duration


RenderNode: TypeAlias = RenderMedia | RenderGroup | RenderGap


@dataclass(frozen=True)
class ResourceStory:
    """A parsed reusable ``<media><sequence>`` story supplied to the builder.

    The current source model stores non-multicam ``media`` resources as raw
    XML.  The Wave 3 resource parser should produce this small adapter instead
    of making this module parse XML a second time.
    """

    resource_id: str
    path: str
    start: Fraction
    duration: Fraction
    story: tuple[StoryNode, ...]
    kind: Literal["compound", "sequence"] = "compound"
    format_context: Optional[SequenceFormatContext] = None


@dataclass(frozen=True)
class StoryIRFinding:
    """One explicit non-rendering or invalid-resolution decision."""

    code: str
    path: str
    detail: str
    disposition: Literal["not_implemented_yet", "invalid", "inactive"]


@dataclass(frozen=True)
class RenderStoryPlan:
    """Frozen hierarchical story graph plus transition views and findings."""

    root: RenderGroup
    transitions: tuple[RenderGroup, ...]
    findings: tuple[StoryIRFinding, ...]


@dataclass(frozen=True)
class _BuiltEntry:
    source_index: int
    node: RenderNode


@dataclass(frozen=True)
class _TransitionEntry:
    source_index: int
    source: StoryNode
    id: str
    instance_path: str
    document_order: int
    absolute_start: Fraction
    duration: Fraction
    source_start: Fraction
    lane: int
    ancestor_group_ids: tuple[str, ...]


class StoryIRBuilder:
    """Resolve a ``SourceDocument`` into a bounded hierarchical story plan.

    Main callers:
    - ``build_render_story``.

    Why this exists:
    The mutable counter, transition collection, and resource recursion stack
    are build-lifetime state.  Housing them here keeps the public IR immutable.
    """

    _MEDIA_KINDS = {"asset-clip", "video", "audio", "title", "caption"}

    def __init__(
        self,
        source: SourceDocument,
        *,
        resource_stories: Optional[Mapping[str, ResourceStory]],
        max_depth: int,
        unresolved_policy: UnresolvedPolicy,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if unresolved_policy not in {"error", "report"}:
            raise ValueError("unresolved_policy must be 'error' or 'report'")
        self.source = source
        self.resource_stories = dict(resource_stories or {})
        for key, story in self.resource_stories.items():
            if key != story.resource_id:
                raise ValueError(f"resource story key {key!r} does not match resource_id {story.resource_id!r}")
            if story.duration <= 0:
                raise ValueError(f"resource story {key!r} duration must be positive")
        self.max_depth = max_depth
        self.unresolved_policy = unresolved_policy
        self._document_order = 0
        self._transitions: list[RenderGroup] = []
        self._findings: list[StoryIRFinding] = []

    def build(self) -> RenderStoryPlan:
        """Build one sequence root, preserving exact absolute child times."""

        window = (Fraction(0), self.source.sequence_duration)
        root_id = "render-root"
        children = self._build_storyline(
            self.source.spine,
            parent_absolute=Fraction(0),
            parent_source=self.source.sequence_tc_start,
            inherited_lane=0,
            ancestor_group_ids=(root_id,),
            instance_prefix="sequence",
            sequential=True,
            window=window,
            depth=1,
            resource_chain=(),
        )
        root = RenderGroup(
            id=root_id,
            kind="sequence",
            path="sequence",
            instance_path="sequence",
            name=self.source.project_name,
            ref=self.source.sequence_format_id,
            absolute_start=Fraction(0),
            duration=self.source.sequence_duration,
            source_start=self.source.sequence_tc_start,
            unclipped_absolute_start=Fraction(0),
            unclipped_duration=self.source.sequence_duration,
            lane=0,
            document_order=-1,
            enabled=True,
            scope=RenderScope(),
            child_timing="sequential",
            resolution="resolved",
            ancestor_group_ids=(),
            children=children,
        )
        return RenderStoryPlan(
            root=root,
            transitions=tuple(sorted(self._transitions, key=lambda item: item.document_order)),
            findings=tuple(self._findings),
        )

    def _next_identity(self, instance_path: str) -> tuple[str, int]:
        order = self._document_order
        self._document_order += 1
        return f"render-node-{order}", order

    def _build_storyline(
        self,
        nodes: tuple[StoryNode, ...],
        *,
        parent_absolute: Fraction,
        parent_source: Fraction,
        inherited_lane: int,
        ancestor_group_ids: tuple[str, ...],
        instance_prefix: str,
        sequential: bool,
        window: tuple[Fraction, Fraction],
        depth: int,
        resource_chain: tuple[str, ...],
    ) -> tuple[RenderNode, ...]:
        """Build siblings, then associate transition markers in source order.

        Main callers:
        - ``build`` for the project spine.
        - ``_build_node`` for nested group and connected stories.
        """

        self._check_depth(depth, instance_prefix)
        cursor = parent_source
        built: list[_BuiltEntry] = []
        transitions: list[_TransitionEntry] = []
        for source_index, source_node in enumerate(nodes):
            offset = source_node.offset
            if offset is None:
                offset = cursor if sequential else parent_source
            raw_absolute = parent_absolute + offset - parent_source
            lane = inherited_lane + source_node.lane
            instance_path = f"{instance_prefix}/{source_node.path}"
            node_id, order = self._next_identity(instance_path)
            if source_node.kind == "transition":
                clipped = self._clip_interval(raw_absolute, source_node.duration, window)
                if clipped is not None:
                    visible_start, visible_duration = clipped
                    transitions.append(
                        _TransitionEntry(
                            source_index=source_index,
                            source=source_node,
                            id=node_id,
                            instance_path=instance_path,
                            document_order=order,
                            absolute_start=visible_start,
                            duration=visible_duration,
                            source_start=source_node.start + (visible_start - raw_absolute),
                            lane=lane,
                            ancestor_group_ids=ancestor_group_ids,
                        )
                    )
            else:
                render_node = self._build_node(
                    source_node,
                    node_id=node_id,
                    document_order=order,
                    raw_absolute=raw_absolute,
                    lane=lane,
                    ancestor_group_ids=ancestor_group_ids,
                    instance_path=instance_path,
                    window=window,
                    depth=depth,
                    resource_chain=resource_chain,
                )
                if render_node is not None:
                    built.append(_BuiltEntry(source_index=source_index, node=render_node))
            if sequential:
                cursor = offset + source_node.duration

        for transition in transitions:
            previous = next((entry.node for entry in reversed(built) if entry.source_index < transition.source_index), None)
            following = next((entry.node for entry in built if entry.source_index > transition.source_index), None)
            self._build_transition_group(transition, previous=previous, following=following)
        return tuple(entry.node for entry in built)

    def _build_node(
        self,
        node: StoryNode,
        *,
        node_id: str,
        document_order: int,
        raw_absolute: Fraction,
        lane: int,
        ancestor_group_ids: tuple[str, ...],
        instance_path: str,
        window: tuple[Fraction, Fraction],
        depth: int,
        resource_chain: tuple[str, ...],
    ) -> Optional[RenderNode]:
        self._check_depth(depth, instance_path)
        clipped = self._clip_interval(raw_absolute, node.duration, window)
        if clipped is None:
            return None
        absolute_start, duration = clipped
        source_start = node.start + (absolute_start - raw_absolute)
        visible_window = (absolute_start, absolute_start + duration)
        scope = self._scope(node)

        if node.kind in self._MEDIA_KINDS:
            connected = self._build_storyline(
                node.children,
                parent_absolute=raw_absolute,
                parent_source=node.start,
                inherited_lane=lane,
                ancestor_group_ids=ancestor_group_ids,
                instance_prefix=instance_path,
                sequential=False,
                window=window,
                depth=depth + 1,
                resource_chain=resource_chain,
            ) if node.children else ()
            return RenderMedia(
                id=node_id,
                kind=node.kind,
                path=node.path,
                instance_path=instance_path,
                name=node.name,
                ref=node.ref,
                absolute_start=absolute_start,
                duration=duration,
                source_start=source_start,
                source_duration=duration,
                unclipped_absolute_start=raw_absolute,
                unclipped_duration=node.duration,
                lane=lane,
                document_order=document_order,
                enabled=node.enabled,
                scope=scope,
                ancestor_group_ids=ancestor_group_ids,
                connected_children=connected,
            )

        if node.kind == "gap":
            connected = self._build_storyline(
                node.children,
                parent_absolute=raw_absolute,
                parent_source=node.start,
                inherited_lane=lane,
                ancestor_group_ids=ancestor_group_ids,
                instance_prefix=instance_path,
                sequential=False,
                window=window,
                depth=depth + 1,
                resource_chain=resource_chain,
            ) if node.children else ()
            return RenderGap(
                id=node_id,
                path=node.path,
                instance_path=instance_path,
                name=node.name,
                absolute_start=absolute_start,
                duration=duration,
                source_start=source_start,
                unclipped_absolute_start=raw_absolute,
                unclipped_duration=node.duration,
                lane=lane,
                document_order=document_order,
                enabled=node.enabled,
                scope=scope,
                ancestor_group_ids=ancestor_group_ids,
                connected_children=connected,
            )

        if node.kind == "audition":
            return self._build_audition(
                node,
                node_id=node_id,
                document_order=document_order,
                raw_absolute=raw_absolute,
                absolute_start=absolute_start,
                duration=duration,
                source_start=source_start,
                lane=lane,
                scope=scope,
                ancestor_group_ids=ancestor_group_ids,
                instance_path=instance_path,
                visible_window=visible_window,
                depth=depth,
                resource_chain=resource_chain,
            )

        if node.kind == "ref-clip":
            return self._build_ref_group(
                node,
                node_id=node_id,
                document_order=document_order,
                raw_absolute=raw_absolute,
                absolute_start=absolute_start,
                duration=duration,
                source_start=source_start,
                lane=lane,
                scope=scope,
                ancestor_group_ids=ancestor_group_ids,
                instance_path=instance_path,
                visible_window=visible_window,
                connected_window=window,
                depth=depth,
                resource_chain=resource_chain,
            )

        if node.kind in {"spine", "clip", "sync-clip"}:
            group_ancestors = (*ancestor_group_ids, node_id)
            if node.kind == "spine":
                inner_nodes = node.children
                connected_nodes: tuple[StoryNode, ...] = ()
                sequential = True
                child_timing: ChildTiming = "sequential"
            else:
                inner_nodes = tuple(child for child in node.children if child.lane == 0)
                connected_nodes = tuple(child for child in node.children if child.lane != 0)
                sequential = False
                child_timing = "anchored"
            children = self._build_storyline(
                inner_nodes,
                parent_absolute=raw_absolute,
                parent_source=node.start,
                inherited_lane=lane,
                ancestor_group_ids=group_ancestors,
                instance_prefix=instance_path,
                sequential=sequential,
                window=visible_window,
                depth=depth + 1,
                resource_chain=resource_chain,
            ) if inner_nodes else ()
            connected = self._build_storyline(
                connected_nodes,
                parent_absolute=raw_absolute,
                parent_source=node.start,
                inherited_lane=lane,
                ancestor_group_ids=ancestor_group_ids,
                instance_prefix=f"{instance_path}/connected",
                sequential=False,
                window=window,
                depth=depth + 1,
                resource_chain=resource_chain,
            ) if connected_nodes else ()
            return RenderGroup(
                id=node_id,
                kind=node.kind,
                path=node.path,
                instance_path=instance_path,
                name=node.name,
                ref=node.ref,
                absolute_start=absolute_start,
                duration=duration,
                source_start=source_start,
                unclipped_absolute_start=raw_absolute,
                unclipped_duration=node.duration,
                lane=lane,
                document_order=document_order,
                enabled=node.enabled,
                scope=scope,
                child_timing=child_timing,
                resolution="resolved",
                ancestor_group_ids=ancestor_group_ids,
                resource_chain=resource_chain,
                children=children,
                connected_children=connected,
            )

        # Multicam needs a separate selected-angle adapter.  Unknown story
        # elements likewise stay visible as an explicit unresolved group.
        code = "multicam_selection_deferred" if node.kind == "mc-clip" else "unsupported_story_kind"
        detail = (
            f"{node.path} requires a selected multicam angle adapter"
            if node.kind == "mc-clip"
            else f"{node.path} has unsupported story kind {node.kind!r}"
        )
        self._unresolved(code=code, path=node.path, detail=detail)
        connected = self._build_storyline(
            node.children,
            parent_absolute=raw_absolute,
            parent_source=node.start,
            inherited_lane=lane,
            ancestor_group_ids=ancestor_group_ids,
            instance_prefix=f"{instance_path}/connected",
            sequential=False,
            window=window,
            depth=depth + 1,
            resource_chain=resource_chain,
        ) if node.children else ()
        return RenderGroup(
            id=node_id,
            kind=node.kind,
            path=node.path,
            instance_path=instance_path,
            name=node.name,
            ref=node.ref,
            absolute_start=absolute_start,
            duration=duration,
            source_start=source_start,
            unclipped_absolute_start=raw_absolute,
            unclipped_duration=node.duration,
            lane=lane,
            document_order=document_order,
            enabled=node.enabled,
            scope=scope,
            child_timing="anchored",
            resolution="deferred" if node.kind == "mc-clip" else "unresolved",
            ancestor_group_ids=ancestor_group_ids,
            resource_chain=resource_chain,
            connected_children=connected,
        )

    def _build_ref_group(
        self,
        node: StoryNode,
        *,
        node_id: str,
        document_order: int,
        raw_absolute: Fraction,
        absolute_start: Fraction,
        duration: Fraction,
        source_start: Fraction,
        lane: int,
        scope: RenderScope,
        ancestor_group_ids: tuple[str, ...],
        instance_path: str,
        visible_window: tuple[Fraction, Fraction],
        connected_window: tuple[Fraction, Fraction],
        depth: int,
        resource_chain: tuple[str, ...],
    ) -> RenderGroup:
        """Resolve compound content, keeping ref-level controls on the group.

        Main callers:
        - ``_build_node`` for ``ref-clip``.
        """

        resource = self.resource_stories.get(node.ref or "")
        resolution: ResolutionState = "resolved"
        children: tuple[RenderNode, ...] = ()
        next_chain = resource_chain
        if resource is None:
            resolution = "unresolved"
            self._unresolved(
                code="compound_resource_unresolved",
                path=node.path,
                detail=f"ref-clip references unparsed media resource {node.ref!r}",
            )
        else:
            if resource.resource_id in resource_chain:
                chain = " -> ".join((*resource_chain, resource.resource_id))
                raise StoryIRCycleError(f"compound resource cycle at {node.path}: {chain}")
            next_chain = (*resource_chain, resource.resource_id)
            selection_end = node.start + node.duration
            resource_end = resource.start + resource.duration
            if node.start < resource.start or selection_end > resource_end:
                self._unresolved(
                    code="compound_source_range_out_of_bounds",
                    path=node.path,
                    detail=(
                        f"selected source range [{node.start}, {selection_end}) exceeds "
                        f"resource {resource.resource_id!r} range [{resource.start}, {resource_end})"
                    ),
                    disposition="invalid",
                )
            children = self._build_storyline(
                resource.story,
                parent_absolute=raw_absolute,
                parent_source=node.start,
                inherited_lane=lane,
                ancestor_group_ids=(*ancestor_group_ids, node_id),
                instance_prefix=f"{instance_path}/resource[{resource.resource_id}]",
                sequential=True,
                window=visible_window,
                depth=depth + 1,
                resource_chain=next_chain,
            )

        connected = self._build_storyline(
            node.children,
            parent_absolute=raw_absolute,
            parent_source=node.start,
            inherited_lane=lane,
            ancestor_group_ids=ancestor_group_ids,
            instance_prefix=f"{instance_path}/connected",
            sequential=False,
            window=connected_window,
            depth=depth + 1,
            resource_chain=resource_chain,
        ) if node.children else ()
        return RenderGroup(
            id=node_id,
            kind="ref-clip",
            path=node.path,
            instance_path=instance_path,
            name=node.name,
            ref=node.ref,
            absolute_start=absolute_start,
            duration=duration,
            source_start=source_start,
            unclipped_absolute_start=raw_absolute,
            unclipped_duration=node.duration,
            lane=lane,
            document_order=document_order,
            enabled=node.enabled,
            scope=scope,
            child_timing="resource",
            resolution=resolution,
            ancestor_group_ids=ancestor_group_ids,
            resource_chain=next_chain,
            children=children,
            connected_children=connected,
        )

    def _build_audition(
        self,
        node: StoryNode,
        *,
        node_id: str,
        document_order: int,
        raw_absolute: Fraction,
        absolute_start: Fraction,
        duration: Fraction,
        source_start: Fraction,
        lane: int,
        scope: RenderScope,
        ancestor_group_ids: tuple[str, ...],
        instance_path: str,
        visible_window: tuple[Fraction, Fraction],
        depth: int,
        resource_chain: tuple[str, ...],
    ) -> RenderGroup:
        """Build only the first audition choice as active and retain the rest."""

        active_nodes = node.children[:1]
        inactive_nodes = node.children[1:]
        if not active_nodes:
            self._unresolved(code="audition_without_choice", path=node.path, detail="audition has no active choice", disposition="invalid")
        group_ancestors = (*ancestor_group_ids, node_id)
        active = self._build_storyline(
            active_nodes,
            parent_absolute=raw_absolute,
            parent_source=node.start,
            inherited_lane=lane,
            ancestor_group_ids=group_ancestors,
            instance_prefix=f"{instance_path}/active",
            sequential=False,
            window=visible_window,
            depth=depth + 1,
            resource_chain=resource_chain,
        ) if active_nodes else ()
        inactive = self._build_storyline(
            inactive_nodes,
            parent_absolute=raw_absolute,
            parent_source=node.start,
            inherited_lane=lane,
            ancestor_group_ids=group_ancestors,
            instance_prefix=f"{instance_path}/inactive",
            sequential=False,
            window=visible_window,
            depth=depth + 1,
            resource_chain=resource_chain,
        ) if inactive_nodes else ()
        for choice in inactive_nodes:
            self._findings.append(
                StoryIRFinding(
                    code="audition_choice_inactive",
                    path=choice.path,
                    detail=f"{choice.path} is preserved as an inactive audition alternative",
                    disposition="inactive",
                )
            )
        return RenderGroup(
            id=node_id,
            kind="audition",
            path=node.path,
            instance_path=instance_path,
            name=node.name,
            ref=node.ref,
            absolute_start=absolute_start,
            duration=duration,
            source_start=source_start,
            unclipped_absolute_start=raw_absolute,
            unclipped_duration=node.duration,
            lane=lane,
            document_order=document_order,
            enabled=node.enabled,
            scope=scope,
            child_timing="choice",
            resolution="resolved",
            ancestor_group_ids=ancestor_group_ids,
            resource_chain=resource_chain,
            children=active,
            inactive_children=inactive,
        )

    def _build_transition_group(
        self,
        transition: _TransitionEntry,
        *,
        previous: Optional[RenderNode],
        following: Optional[RenderNode],
    ) -> None:
        if previous is None or following is None:
            detail = f"transition {transition.source.path} requires adjacent outgoing and incoming items"
            if self.unresolved_policy == "error":
                raise StoryIRTransitionError(detail)
            self._findings.append(
                StoryIRFinding(
                    code="transition_participant_missing",
                    path=transition.source.path,
                    detail=detail,
                    disposition="invalid",
                )
            )
        side_ancestors = (*transition.ancestor_group_ids, transition.id)
        side_duration = transition.duration
        outgoing = RenderGroup(
            id=f"{transition.id}-outgoing",
            kind="transition-outgoing",
            path=transition.source.path,
            instance_path=f"{transition.instance_path}/outgoing",
            name="outgoing",
            ref=None,
            absolute_start=transition.absolute_start,
            duration=side_duration,
            source_start=transition.source_start,
            unclipped_absolute_start=transition.absolute_start,
            unclipped_duration=side_duration,
            lane=transition.lane,
            document_order=transition.document_order,
            enabled=True,
            scope=RenderScope(),
            child_timing="transition_side",
            resolution="resolved" if previous is not None else "unresolved",
            ancestor_group_ids=side_ancestors,
            children=(previous,) if previous is not None else (),
        )
        incoming = replace(
            outgoing,
            id=f"{transition.id}-incoming",
            kind="transition-incoming",
            instance_path=f"{transition.instance_path}/incoming",
            name="incoming",
            resolution="resolved" if following is not None else "unresolved",
            children=(following,) if following is not None else (),
        )
        self._transitions.append(
            RenderGroup(
                id=transition.id,
                kind="transition",
                path=transition.source.path,
                instance_path=transition.instance_path,
                name=transition.source.name,
                ref=transition.source.ref,
                absolute_start=transition.absolute_start,
                duration=transition.duration,
                source_start=transition.source_start,
                unclipped_absolute_start=transition.absolute_start,
                unclipped_duration=transition.duration,
                lane=transition.lane,
                document_order=transition.document_order,
                enabled=transition.source.enabled,
                scope=self._scope(transition.source),
                child_timing="transition_side",
                resolution="resolved" if previous is not None and following is not None else "unresolved",
                ancestor_group_ids=transition.ancestor_group_ids,
                children=(outgoing, incoming),
            )
        )

    @staticmethod
    def _scope(node: StoryNode) -> RenderScope:
        return RenderScope(
            transform=node.transform,
            crop=node.crop,
            conform_type=node.conform_type,
            opacity=node.blend_opacity,
            blend_mode=node.blend_mode,
            opacity_fade=node.opacity_fade,
            filters=node.filters,
            params=node.params,
            preserved_adjustments=node.preserved_adjustments,
            audio_start=node.audio_start,
            audio_duration=node.audio_duration,
            volume_db=node.volume_db,
            audio_fade=node.audio_fade,
            role=node.role,
            video_role=node.video_role,
            audio_role=node.audio_role,
            src_enable=node.src_enable,
            time_map=node.time_map,
            time_map_preserves_pitch=node.time_map_preserves_pitch,
            time_map_frame_sampling=node.time_map_frame_sampling,
        )

    @staticmethod
    def _clip_interval(
        absolute_start: Fraction,
        duration: Fraction,
        window: tuple[Fraction, Fraction],
    ) -> Optional[tuple[Fraction, Fraction]]:
        visible_start = max(absolute_start, window[0])
        visible_end = min(absolute_start + duration, window[1])
        if visible_end <= visible_start:
            return None
        return visible_start, visible_end - visible_start

    def _check_depth(self, depth: int, path: str) -> None:
        if depth > self.max_depth:
            raise StoryIRDepthError(f"story nesting at {path} exceeds max_depth={self.max_depth}")

    def _unresolved(
        self,
        *,
        code: str,
        path: str,
        detail: str,
        disposition: Literal["not_implemented_yet", "invalid"] = "not_implemented_yet",
    ) -> None:
        if self.unresolved_policy == "error":
            raise StoryIRResolutionError(detail)
        self._findings.append(
            StoryIRFinding(code=code, path=path, detail=detail, disposition=disposition)
        )


def build_render_story(
    source: SourceDocument,
    *,
    resource_stories: Optional[Mapping[str, ResourceStory]] = None,
    max_depth: int = 32,
    unresolved_policy: UnresolvedPolicy = "error",
) -> RenderStoryPlan:
    """Build render IR v2 hierarchy without mutating the parsed document.

    ``resource_stories`` is deliberately mandatory for every referenced
    compound media object, although the mapping itself may be omitted when the
    document contains no ``ref-clip``.  Set ``unresolved_policy='report'`` only
    when the caller will surface every returned finding in the compatibility
    report.

    Main callers:
    - The future root compiler integration immediately after source parsing.
    """

    return StoryIRBuilder(
        source,
        resource_stories=resource_stories,
        max_depth=max_depth,
        unresolved_policy=unresolved_policy,
    ).build()


def walk_render_nodes(root: RenderNode, *, include_inactive: bool = False) -> tuple[RenderNode, ...]:
    """Return a deterministic pre-order view for audits and test diagnostics."""

    result: list[RenderNode] = []

    def visit(node: RenderNode) -> None:
        result.append(node)
        if isinstance(node, RenderGroup):
            for child in node.children:
                visit(child)
            for child in node.connected_children:
                visit(child)
            if include_inactive:
                for child in node.inactive_children:
                    visit(child)
        else:
            for child in node.connected_children:
                visit(child)

    visit(root)
    return tuple(result)
