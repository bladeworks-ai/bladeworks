"""Build a strict CompositionPlan beside the unchanged CPU renderer.

Architecture map
================

CPU scheduling adapters
    -> validate exact source, scope, layer, and transition membership
    -> materialize every ``composition_ir`` stage without reinterpretation
    -> derive canonical transition-replacement intervals per scope
    -> order nested modules children-before-parents
    -> return one immutable, hashable ``CompositionPlan``

This module is intentionally a shadow compiler.  It does not import the CPU
renderer's private ``_Layer`` type, allocate decoder indices, or emit FFmpeg
text.  The future integration adapter must explicitly supply every semantic
decision that the CPU scheduler has already made.  In particular, resolved
effect stacks, masks, ignored stages, geometry stages, opacity clocks, and
transition participants are required inputs; this module never guesses them.

Important invariants
--------------------

* Every non-transparent source is represented and consumed exactly once.
* Every nested scope is owned by one explicit ``SourceRef(kind="module")``.
* Layers are canonicalized by ``ZOrderKey`` inside their owning scope.
* Transition sides retain all recursively composed participant IDs and replace
  those participants only over the exact ``OwnedFrameWindow``.
* Raster placement, effects/masks, spatial transform, then opacity/blend stay
  as four separate ordered semantic boundaries.
* Warn-and-ignore findings must exactly match the ignored effect stages.

Why this exists
---------------

The migration needs evidence that a backend-neutral plan can mirror the CPU
scheduler before the CPU graph builder consumes that plan.  Keeping this
compiler isolated lets tests compare semantic hashes and interval schedules
without changing one byte of the production FFmpeg graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .composition_ir import (
    CompositionPlan,
    CompositionPlanError,
    DecoderSourcePlan,
    EffectStackPlan,
    FrameRateNormalizationPlan,
    FrameContract,
    HardCutPlan,
    IgnoredEffectFinding,
    IgnoredEffectOp,
    LayerExecution,
    LayerPlan,
    OpacityEnvelopePlan,
    RasterPlacementPlan,
    RasterSpatialBoundaryPlan,
    RasterSourcePlan,
    SourceRef,
    SpatialTransformPlan,
    SurfaceSpec,
    TransitionExtension,
    TransitionPlan,
    TransitionSidePlan,
    VideoDispositionFinding,
    ZOrderKey,
    build_composition_scope_plan,
)
from .pixel_domains import FrameClock
from .retime_execution import OwnedFrameWindow, RetimeExecutionPlan


ScheduledSource: TypeAlias = DecoderSourcePlan | RasterSourcePlan


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompositionPlanError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ScheduledLayerInput:
    """One complete CPU-scheduled layer, before backend graph emission.

    Main callers:
    - The future adapter beside ``ffmpeg.build_invocation``.
    - Focused shadow-plan parity tests.

    ``raster``, ``effects``, and ``spatial`` are deliberately required rather
    than reconstructed from ``RenderClip``.  They preserve the CPU stage order
    and force masked or ignored effects to cross the adapter explicitly.
    """

    scope_id: str
    layer_id: str
    path: str
    source: SourceRef
    window: OwnedFrameWindow
    z_order: ZOrderKey
    raster: RasterPlacementPlan
    effects: EffectStackPlan
    spatial: SpatialTransformPlan
    opacity: OpacityEnvelopePlan
    blend_mode: str | None
    source_contract: FrameContract
    source_retime: RetimeExecutionPlan | None
    input_contract: FrameContract
    output_contract: FrameContract
    execution: LayerExecution
    frame_rate_normalization: FrameRateNormalizationPlan | None
    raster_spatial_boundary: RasterSpatialBoundaryPlan = RasterSpatialBoundaryPlan()

    def __post_init__(self) -> None:
        _text(self.scope_id, name="layer scope_id")
        _text(self.layer_id, name="layer_id")
        _text(self.path, name="layer path")
        if not isinstance(self.source, SourceRef):
            raise CompositionPlanError("scheduled layer source must be SourceRef")
        if not isinstance(self.window, OwnedFrameWindow):
            raise CompositionPlanError("scheduled layer window must be OwnedFrameWindow")
        if not isinstance(self.z_order, ZOrderKey):
            raise CompositionPlanError("scheduled layer z_order must be ZOrderKey")
        if not isinstance(self.raster, RasterPlacementPlan):
            raise CompositionPlanError(
                "scheduled layer requires a resolved RasterPlacementPlan"
            )
        if not isinstance(self.effects, EffectStackPlan):
            raise CompositionPlanError(
                "scheduled layer requires a resolved EffectStackPlan"
            )
        if not isinstance(self.spatial, SpatialTransformPlan):
            raise CompositionPlanError(
                "scheduled layer requires a resolved SpatialTransformPlan"
            )
        if not isinstance(self.opacity, OpacityEnvelopePlan):
            raise CompositionPlanError(
                "scheduled layer requires a resolved OpacityEnvelopePlan"
            )
        if self.execution not in {
            "composite",
            "omit_transparent",
            "authored_disabled",
        }:
            raise CompositionPlanError(
                f"unknown scheduled layer execution {self.execution!r}"
            )
        if self.frame_rate_normalization is not None and not isinstance(
            self.frame_rate_normalization,
            FrameRateNormalizationPlan,
        ):
            raise CompositionPlanError(
                "scheduled frame_rate_normalization must be "
                "FrameRateNormalizationPlan or None"
            )
        if not isinstance(self.raster_spatial_boundary, RasterSpatialBoundaryPlan):
            raise CompositionPlanError(
                "scheduled raster_spatial_boundary must be RasterSpatialBoundaryPlan"
            )

    def to_plan(self) -> LayerPlan:
        """Materialize the four ordered semantic stages without modification."""

        return LayerPlan(
            layer_id=self.layer_id,
            path=self.path,
            source=self.source,
            window=self.window,
            z_order=self.z_order,
            raster=self.raster,
            effects=self.effects,
            spatial=self.spatial,
            opacity=self.opacity,
            blend=LayerPlan.resolve_blend(self.blend_mode),
            source_contract=self.source_contract,
            source_retime=self.source_retime,
            input_contract=self.input_contract,
            output_contract=self.output_contract,
            execution=self.execution,
            frame_rate_normalization=self.frame_rate_normalization,
            raster_spatial_boundary=self.raster_spatial_boundary,
        )


@dataclass(frozen=True)
class ScheduledScopeInput:
    """One CPU ownership scope with its already-planned canvas and clock."""

    scope_id: str
    path: str
    parent_scope_id: str | None
    canvas: SurfaceSpec
    window: OwnedFrameWindow
    output_contract: FrameContract
    requires_transparent_intermediate: bool
    enabled: bool

    def __post_init__(self) -> None:
        _text(self.scope_id, name="scope_id")
        _text(self.path, name="scope path")
        if self.parent_scope_id is not None:
            _text(self.parent_scope_id, name="parent_scope_id")
        if not isinstance(self.canvas, SurfaceSpec):
            raise CompositionPlanError("scheduled scope canvas must be SurfaceSpec")
        if not isinstance(self.window, OwnedFrameWindow):
            raise CompositionPlanError("scheduled scope window must be OwnedFrameWindow")
        if not isinstance(self.output_contract, FrameContract):
            raise CompositionPlanError(
                "scheduled scope requires a resolved output FrameContract"
            )
        if not isinstance(self.requires_transparent_intermediate, bool):
            raise CompositionPlanError(
                "requires_transparent_intermediate must be bool"
            )
        if not isinstance(self.enabled, bool):
            raise CompositionPlanError("scheduled scope enabled must be bool")


@dataclass(frozen=True)
class ScheduledTransitionInput:
    """One exact replacement of two complete, possibly multi-item sides."""

    scope_id: str
    transition_id: str
    path: str
    window: OwnedFrameWindow
    outgoing_layer_ids: tuple[str, ...]
    incoming_layer_ids: tuple[str, ...]
    outgoing_handle: OwnedFrameWindow
    incoming_handle: OwnedFrameWindow
    outgoing_extension: TransitionExtension
    incoming_extension: TransitionExtension
    z_order: ZOrderKey
    handler: str
    parameters: tuple[tuple[str, str], ...]
    input_contract: FrameContract
    output_contract: FrameContract
    artifact_semantic_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.scope_id, name="transition scope_id")
        _text(self.transition_id, name="transition_id")
        _text(self.path, name="transition path")
        _text(self.handler, name="transition handler")
        if not self.outgoing_layer_ids or not self.incoming_layer_ids:
            raise CompositionPlanError(
                "scheduled transition requires non-empty outgoing and incoming sides"
            )
        if tuple(sorted(self.parameters)) != self.parameters:
            raise CompositionPlanError(
                "scheduled transition parameters must be canonical sorted pairs"
            )

    def to_plan(self) -> TransitionPlan:
        """Preserve every side participant and its independently owned handle."""

        return TransitionPlan(
            transition_id=self.transition_id,
            path=self.path,
            window=self.window,
            outgoing=TransitionSidePlan(
                composed_sources=self.outgoing_layer_ids,
                semantic_handle=self.outgoing_handle,
                source_extension=self.outgoing_extension,
            ),
            incoming=TransitionSidePlan(
                composed_sources=self.incoming_layer_ids,
                semantic_handle=self.incoming_handle,
                source_extension=self.incoming_extension,
            ),
            z_order=self.z_order,
            handler=self.handler,
            parameters=self.parameters,
            input_contract=self.input_contract,
            output_contract=self.output_contract,
            artifact_semantic_id=self.artifact_semantic_id,
        )


@dataclass(frozen=True)
class ScheduledHardCutInput:
    """One handler-less authored transition resolved to ordinary hard-cut topology."""

    scope_id: str
    transition_id: str
    path: str
    window: OwnedFrameWindow
    z_order: ZOrderKey
    parameters: tuple[tuple[str, str], ...]
    finding_id: str
    previous_story_id: str | None = None
    next_story_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.scope_id, name="hard-cut scope_id")
        _text(self.transition_id, name="hard-cut transition_id")
        _text(self.path, name="hard-cut path")
        _text(self.finding_id, name="hard-cut finding_id")
        if self.previous_story_id is not None:
            _text(self.previous_story_id, name="hard-cut previous_story_id")
        if self.next_story_id is not None:
            _text(self.next_story_id, name="hard-cut next_story_id")
        if not isinstance(self.window, OwnedFrameWindow):
            raise CompositionPlanError("hard-cut window must be OwnedFrameWindow")
        if not isinstance(self.z_order, ZOrderKey):
            raise CompositionPlanError("hard-cut z_order must be ZOrderKey")
        if tuple(sorted(self.parameters)) != self.parameters:
            raise CompositionPlanError(
                "hard-cut parameters must be canonical sorted pairs"
            )

    def to_plan(self) -> HardCutPlan:
        """Carry topology metadata without creating a composition stack item."""

        return HardCutPlan(
            transition_id=self.transition_id,
            path=self.path,
            window=self.window,
            z_order=self.z_order,
            parameters=self.parameters,
            finding_id=self.finding_id,
            previous_story_id=self.previous_story_id,
            next_story_id=self.next_story_id,
        )


@dataclass(frozen=True)
class ScheduledCompositionInput:
    """Complete public handoff from the CPU scheduler to the shadow compiler."""

    document_source_sha256: str
    project_canvas: SurfaceSpec
    project_clock: FrameClock
    root_scope_id: str
    sources: tuple[ScheduledSource, ...]
    scopes: tuple[ScheduledScopeInput, ...]
    layers: tuple[ScheduledLayerInput, ...]
    transitions: tuple[ScheduledTransitionInput, ...]
    hard_cuts: tuple[ScheduledHardCutInput, ...]
    video_findings: tuple[VideoDispositionFinding, ...]
    ignored_findings: tuple[IgnoredEffectFinding, ...]

    def __post_init__(self) -> None:
        _text(self.root_scope_id, name="root_scope_id")
        if not isinstance(self.project_canvas, SurfaceSpec):
            raise CompositionPlanError("project_canvas must be SurfaceSpec")
        if not isinstance(self.project_clock, FrameClock):
            raise CompositionPlanError("project_clock must be FrameClock")
        if any(
            not isinstance(source, (DecoderSourcePlan, RasterSourcePlan))
            for source in self.sources
        ):
            raise CompositionPlanError(
                "sources must contain only DecoderSourcePlan or RasterSourcePlan"
            )
        if any(not isinstance(item, ScheduledScopeInput) for item in self.scopes):
            raise CompositionPlanError("scopes must contain ScheduledScopeInput")
        if any(not isinstance(item, ScheduledLayerInput) for item in self.layers):
            raise CompositionPlanError("layers must contain ScheduledLayerInput")
        if any(
            not isinstance(item, ScheduledTransitionInput)
            for item in self.transitions
        ):
            raise CompositionPlanError(
                "transitions must contain ScheduledTransitionInput"
            )
        if any(not isinstance(item, ScheduledHardCutInput) for item in self.hard_cuts):
            raise CompositionPlanError("hard_cuts must contain ScheduledHardCutInput")
        if any(
            not isinstance(item, VideoDispositionFinding)
            for item in self.video_findings
        ):
            raise CompositionPlanError(
                "video_findings must contain VideoDispositionFinding"
            )
        if any(
            not isinstance(item, IgnoredEffectFinding)
            for item in self.ignored_findings
        ):
            raise CompositionPlanError(
                "ignored_findings must contain IgnoredEffectFinding"
            )


def _children_before_parents(
    scopes: tuple[ScheduledScopeInput, ...],
    *,
    root_scope_id: str,
) -> tuple[ScheduledScopeInput, ...]:
    """Return deterministic post-order and reject cycles or disconnected scopes."""

    by_id = {scope.scope_id: scope for scope in scopes}
    if len(by_id) != len(scopes):
        raise CompositionPlanError("scheduled scope IDs must be unique")
    if root_scope_id not in by_id:
        raise CompositionPlanError("root_scope_id does not identify a scheduled scope")
    if by_id[root_scope_id].parent_scope_id is not None:
        raise CompositionPlanError("scheduled root scope cannot have a parent")
    input_order = {scope.scope_id: index for index, scope in enumerate(scopes)}
    children: dict[str, list[str]] = {scope_id: [] for scope_id in by_id}
    for scope in scopes:
        parent = scope.parent_scope_id
        if parent is None:
            if scope.scope_id != root_scope_id:
                raise CompositionPlanError(
                    f"non-root scope {scope.scope_id} is disconnected"
                )
            continue
        if parent not in by_id:
            raise CompositionPlanError(
                f"scope {scope.scope_id} has unknown parent {parent}"
            )
        children[parent].append(scope.scope_id)
    for values in children.values():
        values.sort(key=input_order.__getitem__)

    visiting: set[str] = set()
    visited: set[str] = set()
    ordered: list[ScheduledScopeInput] = []

    def visit(scope_id: str) -> None:
        if scope_id in visiting:
            raise CompositionPlanError("scheduled scope graph contains a cycle")
        if scope_id in visited:
            return
        visiting.add(scope_id)
        for child_id in children[scope_id]:
            visit(child_id)
        visiting.remove(scope_id)
        visited.add(scope_id)
        ordered.append(by_id[scope_id])

    visit(root_scope_id)
    if len(visited) != len(scopes):
        missing = sorted(set(by_id) - visited)
        raise CompositionPlanError(
            "scheduled scope graph is disconnected: " + ", ".join(missing)
        )
    return tuple(ordered)


def compile_cpu_shadow_composition(
    scheduled: ScheduledCompositionInput,
) -> CompositionPlan:
    """Compile one strict backend-neutral mirror of CPU scheduling.

    Main callers:
    - Shadow validation beside the current CPU renderer.
    - Migration tests before any FFmpeg emission consumes ``CompositionPlan``.

    Why this exists:
    This routine performs only assembly and cross-checking.  If the adapter has
    not supplied an exact stage, participant, contract, or finding, compilation
    fails rather than substituting a default that could diverge from the CPU.
    """

    if not isinstance(scheduled, ScheduledCompositionInput):
        raise CompositionPlanError(
            "scheduled must be a ScheduledCompositionInput"
        )

    ordered_scopes = _children_before_parents(
        scheduled.scopes,
        root_scope_id=scheduled.root_scope_id,
    )
    scope_ids = {scope.scope_id for scope in ordered_scopes}

    layer_ids = tuple(layer.layer_id for layer in scheduled.layers)
    if len(layer_ids) != len(set(layer_ids)):
        raise CompositionPlanError("scheduled layer IDs must be globally unique")
    transition_ids = tuple(item.transition_id for item in scheduled.transitions)
    if len(transition_ids) != len(set(transition_ids)):
        raise CompositionPlanError(
            "scheduled transition IDs must be globally unique"
        )
    hard_cut_ids = tuple(item.transition_id for item in scheduled.hard_cuts)
    if len(hard_cut_ids) != len(set(hard_cut_ids)):
        raise CompositionPlanError("scheduled hard-cut IDs must be globally unique")
    if (
        set(layer_ids) & set(transition_ids)
        or set(layer_ids) & set(hard_cut_ids)
        or set(transition_ids) & set(hard_cut_ids)
    ):
        raise CompositionPlanError(
            "scheduled layer, transition, and hard-cut IDs must be disjoint"
        )

    layers_by_scope: dict[str, list[LayerPlan]] = {
        scope_id: [] for scope_id in scope_ids
    }
    layer_scope: dict[str, str] = {}
    for item in scheduled.layers:
        if item.scope_id not in scope_ids:
            raise CompositionPlanError(
                f"layer {item.layer_id} references unknown scope {item.scope_id}"
            )
        plan = item.to_plan()
        layers_by_scope[item.scope_id].append(plan)
        layer_scope[plan.layer_id] = item.scope_id
    for scope_id, layers in layers_by_scope.items():
        z_orders = tuple(layer.z_order for layer in layers)
        if len(z_orders) != len(set(z_orders)):
            raise CompositionPlanError(
                f"scope {scope_id} has duplicate layer z-order keys"
            )
        layers.sort(key=lambda layer: (layer.z_order, layer.layer_id))

    transitions_by_scope: dict[str, list[TransitionPlan]] = {
        scope_id: [] for scope_id in scope_ids
    }
    for item in scheduled.transitions:
        if item.scope_id not in scope_ids:
            raise CompositionPlanError(
                f"transition {item.transition_id} references unknown scope "
                f"{item.scope_id}"
            )
        participants = item.outgoing_layer_ids + item.incoming_layer_ids
        if len(participants) != len(set(participants)):
            raise CompositionPlanError(
                f"transition {item.transition_id} repeats a side participant"
            )
        for layer_id in participants:
            owner = layer_scope.get(layer_id)
            if owner != item.scope_id:
                raise CompositionPlanError(
                    f"transition {item.transition_id} participant {layer_id} "
                    "is not a direct layer of its owning scope"
                )
        transitions_by_scope[item.scope_id].append(item.to_plan())
    for transitions in transitions_by_scope.values():
        transitions.sort(
            key=lambda item: (
                item.window.first_frame,
                item.window.end_frame,
                item.z_order,
                item.transition_id,
            )
        )

    hard_cuts_by_scope: dict[str, list[HardCutPlan]] = {
        scope_id: [] for scope_id in scope_ids
    }
    for item in scheduled.hard_cuts:
        if item.scope_id not in scope_ids:
            raise CompositionPlanError(
                f"hard cut {item.transition_id} references unknown scope "
                f"{item.scope_id}"
            )
        hard_cuts_by_scope[item.scope_id].append(item.to_plan())
    for hard_cuts in hard_cuts_by_scope.values():
        hard_cuts.sort(
            key=lambda item: (
                item.window.first_frame,
                item.window.end_frame,
                item.z_order,
                item.transition_id,
            )
        )

    source_ids = tuple(source.identity.source_id for source in scheduled.sources)
    if len(source_ids) != len(set(source_ids)):
        raise CompositionPlanError("scheduled source IDs must be unique")
    referenced_source_ids = tuple(
        layer.source.ref
        for layers in layers_by_scope.values()
        for layer in layers
        if layer.source.kind in {"decoder", "still", "runtime_raster"}
    )
    if len(referenced_source_ids) != len(set(referenced_source_ids)):
        raise CompositionPlanError(
            "each scheduled decoder/raster source must own exactly one layer"
        )
    if set(referenced_source_ids) != set(source_ids):
        missing = sorted(set(referenced_source_ids) - set(source_ids))
        unused = sorted(set(source_ids) - set(referenced_source_ids))
        detail = []
        if missing:
            detail.append("missing sources: " + ", ".join(missing))
        if unused:
            detail.append("unused sources: " + ", ".join(unused))
        raise CompositionPlanError(
            "scheduled source set does not exactly match layer references ("
            + "; ".join(detail)
            + ")"
        )

    ignored_from_stages = {
        (stage.path, stage.handler, stage.reason)
        for layers in layers_by_scope.values()
        for layer in layers
        for stage in layer.effects.stages
        if isinstance(stage, IgnoredEffectOp)
    }
    ignored_from_findings = {
        (finding.path, finding.handler, finding.reason)
        for finding in scheduled.ignored_findings
    }
    general_identity_paths = {
        (finding.path, finding.reason)
        for finding in scheduled.video_findings
        if finding.replacement == "identity"
    }
    uncovered_ignored = {
        item for item in ignored_from_stages if (item[0], item[2]) not in general_identity_paths
    }
    if uncovered_ignored != ignored_from_findings:
        raise CompositionPlanError(
            "scheduled legacy findings do not exactly match identity stages not "
            "covered by general video findings"
        )

    compiled_scopes = tuple(
        build_composition_scope_plan(
            scope_id=scope.scope_id,
            path=scope.path,
            parent_scope_id=scope.parent_scope_id,
            canvas=scope.canvas,
            window=scope.window,
            layers=tuple(layers_by_scope[scope.scope_id]),
            transitions=tuple(transitions_by_scope[scope.scope_id]),
            hard_cuts=tuple(hard_cuts_by_scope[scope.scope_id]),
            output_contract=scope.output_contract,
            requires_transparent_intermediate=(
                scope.requires_transparent_intermediate
            ),
            enabled=scope.enabled,
        )
        for scope in ordered_scopes
    )

    decoders = tuple(
        sorted(
            (
                source
                for source in scheduled.sources
                if isinstance(source, DecoderSourcePlan)
            ),
            key=lambda source: source.identity.source_id,
        )
    )
    rasters = tuple(
        sorted(
            (
                source
                for source in scheduled.sources
                if isinstance(source, RasterSourcePlan)
            ),
            key=lambda source: source.identity.source_id,
        )
    )
    ignored_findings = tuple(
        sorted(
            scheduled.ignored_findings,
            key=lambda item: (item.path, item.handler, item.reason),
        )
    )
    video_findings = tuple(
        sorted(
            scheduled.video_findings,
            key=lambda item: (item.finding_id, item.target_id),
        )
    )
    return CompositionPlan(
        schema_version=1,
        document_source_sha256=scheduled.document_source_sha256,
        project_canvas=scheduled.project_canvas,
        project_clock=scheduled.project_clock,
        decoders=decoders,
        rasters=rasters,
        scopes=compiled_scopes,
        root_scope_id=scheduled.root_scope_id,
        video_findings=video_findings,
        ignored_findings=ignored_findings,
    )


__all__ = [
    "ScheduledCompositionInput",
    "ScheduledHardCutInput",
    "ScheduledLayerInput",
    "ScheduledScopeInput",
    "ScheduledTransitionInput",
    "compile_cpu_shadow_composition",
]
