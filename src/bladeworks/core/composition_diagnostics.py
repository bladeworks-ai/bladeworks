"""Small, deterministic diagnostics for one failing composition interval.

Architecture map
================

``CompositionPlan`` + project frame
    -> select the compiler-owned canonical interval
    -> retain its exact lower-to-upper semantic stack
    -> attach backend observations at named stage boundaries
    -> emit one comparable CPU/Vulkan trace

``reduce_failing_interval``
    -> start from that same active stack
    -> keep only semantic dependencies needed by each candidate
    -> use a caller-supplied failure oracle to delta-debug the stack

This module deliberately does not parse FFmpeg filter strings and does not
render a project.  It gives tiny CPU/GPU probes a common vocabulary so a
95-source timeline can first be reduced to the few layers that actually own a
failing frame.  Backend observations remain explicit inputs: the diagnostic
must never invent a decoder frame, crop rectangle, or pixel hash.

Important invariants
--------------------

* Canonical intervals and z-order come only from ``CompositionPlan``.
* Raster sources own frame zero; decoder frame ownership must be observed by
  the backend that executed the source schedule.
* A strict trace contains one observation for every active layer and no
  observation for an inactive layer.
* Reduction returns a semantic slice, not a mutated ``CompositionPlan``.  The
  caller decides how to lower each candidate and whether the failure remains.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from typing import Callable, Literal

from .composition_ir import (
    CompositionInterval,
    CompositionPlan,
    CompositionScopePlan,
    LayerPlan,
    PlanId,
    StackItem,
    TransitionPlan,
)


class CompositionDiagnosticError(ValueError):
    """The requested trace or reduction contradicts frozen plan semantics."""


Rect = tuple[Fraction, Fraction, Fraction, Fraction]


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _validate_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64:
        raise CompositionDiagnosticError(f"{field_name} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise CompositionDiagnosticError(
            f"{field_name} must be a SHA-256 hex digest"
        ) from error


@dataclass(frozen=True)
class PixelBoundaryObservation:
    """One exact RGBA stage result produced by a tiny backend probe."""

    rgba_sha256: str
    width: int
    height: int
    alpha_min: int
    alpha_max: int
    alpha_nonzero_bbox: tuple[int, int, int, int] | None

    def __post_init__(self) -> None:
        _validate_sha256(self.rgba_sha256, field_name="rgba_sha256")
        if self.width <= 0 or self.height <= 0:
            raise CompositionDiagnosticError("pixel observation dimensions must be positive")
        if not 0 <= self.alpha_min <= self.alpha_max <= 255:
            raise CompositionDiagnosticError("alpha bounds must lie in [0, 255]")
        if self.alpha_nonzero_bbox is None:
            if self.alpha_max != 0:
                raise CompositionDiagnosticError(
                    "a missing alpha bounding box requires a fully transparent frame"
                )
            return
        left, top, right, bottom = self.alpha_nonzero_bbox
        if not (0 <= left < right <= self.width and 0 <= top < bottom <= self.height):
            raise CompositionDiagnosticError(
                "alpha bounding box must be a non-empty rectangle inside the frame"
            )
        if self.alpha_max == 0:
            raise CompositionDiagnosticError(
                "a non-empty alpha bounding box requires nonzero alpha"
            )

    def manifest(self) -> dict[str, object]:
        return {
            "rgba_sha256": self.rgba_sha256,
            "width": self.width,
            "height": self.height,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "alpha_nonzero_bbox": (
                list(self.alpha_nonzero_bbox)
                if self.alpha_nonzero_bbox is not None
                else None
            ),
        }


@dataclass(frozen=True)
class LayerBoundaryObservation:
    """Backend facts at one active layer's semantic stage boundaries.

    Main callers:
    - Low-resolution CPU and Vulkan frame probes.

    ``source_frame_index`` is required for decoded video because only the
    backend execution knows the realized retime/seek ownership.  It must be
    omitted for raster and nested-module sources, whose semantic source frame
    is respectively fixed at zero or represented by the child scope.
    """

    layer_id: PlanId
    source_frame_index: int | None
    crop_rect: Rect
    destination_rect: Rect
    transform_values: tuple[tuple[str, str], ...]
    post_raster: PixelBoundaryObservation
    post_effect: PixelBoundaryObservation

    def __post_init__(self) -> None:
        if not self.layer_id.strip():
            raise CompositionDiagnosticError("layer_id cannot be empty")
        if tuple(sorted(self.transform_values)) != self.transform_values:
            raise CompositionDiagnosticError(
                "transform values must be sorted canonical key/value pairs"
            )
        for name, rect in (
            ("crop_rect", self.crop_rect),
            ("destination_rect", self.destination_rect),
        ):
            if len(rect) != 4 or rect[2] <= rect[0] or rect[3] <= rect[1]:
                raise CompositionDiagnosticError(f"{name} must be a non-empty rectangle")


@dataclass(frozen=True)
class SemanticLayerTrace:
    """Plan semantics and measured pixels for one active layer."""

    layer_id: PlanId
    path: str
    source_kind: str
    source_id: PlanId | None
    source_frame_index: int | None
    project_frame: int
    project_pts: Fraction
    crop_rect: Rect
    destination_rect: Rect
    transform_values: tuple[tuple[str, str], ...]
    z_order: tuple[int, int]
    post_raster: PixelBoundaryObservation
    post_effect: PixelBoundaryObservation

    def manifest(self) -> dict[str, object]:
        def rect(value: Rect) -> list[str]:
            return [_fraction_text(item) for item in value]

        return {
            "layer_id": self.layer_id,
            "path": self.path,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_frame_index": self.source_frame_index,
            "project_frame": self.project_frame,
            "project_pts": _fraction_text(self.project_pts),
            "crop_rect": rect(self.crop_rect),
            "destination_rect": rect(self.destination_rect),
            "transform_values": [list(item) for item in self.transform_values],
            "z_order": list(self.z_order),
            "post_raster": self.post_raster.manifest(),
            "post_effect": self.post_effect.manifest(),
        }


@dataclass(frozen=True)
class CompositionFrameTrace:
    """One complete CPU or Vulkan semantic-boundary trace."""

    backend: Literal["cpu", "vulkan", "reference"]
    plan_sha256: str
    scope_id: PlanId
    project_frame: int
    project_pts: Fraction
    interval: CompositionInterval
    stack: tuple[StackItem, ...]
    layers: tuple[SemanticLayerTrace, ...]
    final_composited: PixelBoundaryObservation

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "fcpxml_composition_frame_trace.v1",
            "backend": self.backend,
            "plan_sha256": self.plan_sha256,
            "scope_id": self.scope_id,
            "project_frame": self.project_frame,
            "project_pts": _fraction_text(self.project_pts),
            "interval": {
                "first_frame": self.interval.window.first_frame,
                "end_frame": self.interval.window.end_frame,
                "frame_duration": _fraction_text(self.interval.window.frame_duration),
            },
            "stack": [
                {
                    "kind": item.kind,
                    "ref": item.ref,
                    "z_order": [item.z_order.lane, item.z_order.document_order],
                }
                for item in self.stack
            ],
            "layers": [layer.manifest() for layer in self.layers],
            "final_composited": self.final_composited.manifest(),
        }

    @property
    def manifest_sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SemanticTraceDivergence:
    """The first semantic boundary where two backend traces disagree."""

    stage: Literal[
        "plan",
        "stack",
        "source_frame",
        "post_raster",
        "geometry",
        "post_effect",
        "final_composite",
    ]
    semantic_id: PlanId
    detail: str


def first_trace_divergence(
    reference: CompositionFrameTrace,
    candidate: CompositionFrameTrace,
) -> SemanticTraceDivergence | None:
    """Locate the first differing plan, source, geometry, or pixel boundary.

    Main callers:
    - The low-resolution title diagnostic after it records matching CPU and
      Vulkan frame traces.

    The comparison order mirrors the semantic pipeline.  It intentionally
    stops at the first mismatch so a downstream composited difference is not
    mistaken for an independent bug.
    """

    root_id = reference.scope_id
    if reference.plan_sha256 != candidate.plan_sha256:
        return SemanticTraceDivergence(
            "plan", root_id, "backends did not consume the same CompositionPlan"
        )
    if (
        reference.scope_id != candidate.scope_id
        or reference.project_frame != candidate.project_frame
        or reference.project_pts != candidate.project_pts
    ):
        return SemanticTraceDivergence(
            "plan", root_id, "backends traced different scope/frame ownership"
        )
    if reference.stack != candidate.stack:
        return SemanticTraceDivergence(
            "stack", root_id, "canonical active stack or z-order differs"
        )
    if tuple(item.layer_id for item in reference.layers) != tuple(
        item.layer_id for item in candidate.layers
    ):
        return SemanticTraceDivergence(
            "stack", root_id, "active layer trace order differs"
        )
    for left, right in zip(reference.layers, candidate.layers):
        if left.source_frame_index != right.source_frame_index:
            return SemanticTraceDivergence(
                "source_frame", left.layer_id, "owned source-frame index differs"
            )
        if left.post_raster.rgba_sha256 != right.post_raster.rgba_sha256:
            return SemanticTraceDivergence(
                "post_raster", left.layer_id, "post-raster RGBA hash differs"
            )
        if (
            left.crop_rect != right.crop_rect
            or left.destination_rect != right.destination_rect
            or left.transform_values != right.transform_values
        ):
            return SemanticTraceDivergence(
                "geometry", left.layer_id, "crop, placement, or transform differs"
            )
        if left.post_effect.rgba_sha256 != right.post_effect.rgba_sha256:
            return SemanticTraceDivergence(
                "post_effect", left.layer_id, "post-effect RGBA hash differs"
            )
    if (
        reference.final_composited.rgba_sha256
        != candidate.final_composited.rgba_sha256
    ):
        return SemanticTraceDivergence(
            "final_composite", root_id, "final composited RGBA hash differs"
        )
    return None


def _root_scope(plan: CompositionPlan) -> CompositionScopePlan:
    return next(scope for scope in plan.scopes if scope.scope_id == plan.root_scope_id)


def _interval_at(scope: CompositionScopePlan, project_frame: int) -> CompositionInterval:
    for interval in scope.intervals:
        if interval.window.first_frame <= project_frame < interval.window.end_frame:
            return interval
    raise CompositionDiagnosticError(
        f"project frame {project_frame} lies outside scope {scope.scope_id}"
    )


def trace_composition_frame(
    plan: CompositionPlan,
    *,
    backend: Literal["cpu", "vulkan", "reference"],
    project_frame: int,
    layer_observations: tuple[LayerBoundaryObservation, ...],
    final_composited: PixelBoundaryObservation,
) -> CompositionFrameTrace:
    """Join one backend's measurements to the exact canonical active stack.

    Main callers:
    - Boundary-frame diagnosis around a known CPU/Vulkan mismatch.

    Why this exists:
    A whole-project SSIM cannot say whether a bad frame came from source
    ownership, alpha, geometry, effects, or z-order.  This routine fails if a
    probe omits an active layer or reports an inactive one, making CPU and
    Vulkan traces directly comparable stage by stage.
    """

    root = _root_scope(plan)
    interval = _interval_at(root, project_frame)
    layer_items = tuple(item for item in interval.stack if item.kind == "layer")
    active_ids = tuple(item.ref for item in layer_items)
    observation_by_id = {item.layer_id: item for item in layer_observations}
    if len(observation_by_id) != len(layer_observations):
        raise CompositionDiagnosticError("layer observations contain duplicate IDs")
    if set(observation_by_id) != set(active_ids):
        missing = sorted(set(active_ids) - set(observation_by_id))
        extra = sorted(set(observation_by_id) - set(active_ids))
        raise CompositionDiagnosticError(
            f"layer observations do not match the active stack; missing={missing}, extra={extra}"
        )
    layer_by_id = {layer.layer_id: layer for layer in root.layers}
    project_pts = (
        root.window.frame_grid_origin
        + project_frame * root.window.frame_duration
    )
    traces: list[SemanticLayerTrace] = []
    for item in layer_items:
        layer = layer_by_id[item.ref]
        observation = observation_by_id[item.ref]
        if layer.source.kind == "decoder":
            if observation.source_frame_index is None:
                raise CompositionDiagnosticError(
                    f"decoder layer {layer.layer_id} requires observed source-frame ownership"
                )
            source_frame_index = observation.source_frame_index
        elif layer.source.kind in {"still", "runtime_raster"}:
            if observation.source_frame_index not in {None, 0}:
                raise CompositionDiagnosticError(
                    f"raster layer {layer.layer_id} can only report source frame zero"
                )
            source_frame_index = 0
        else:
            if observation.source_frame_index is not None:
                raise CompositionDiagnosticError(
                    f"module/transparent layer {layer.layer_id} cannot report a decoder frame"
                )
            source_frame_index = None
        traces.append(
            SemanticLayerTrace(
                layer_id=layer.layer_id,
                path=layer.path,
                source_kind=layer.source.kind,
                source_id=layer.source.ref,
                source_frame_index=source_frame_index,
                project_frame=project_frame,
                project_pts=project_pts,
                crop_rect=observation.crop_rect,
                destination_rect=observation.destination_rect,
                transform_values=observation.transform_values,
                z_order=(layer.z_order.lane, layer.z_order.document_order),
                post_raster=observation.post_raster,
                post_effect=observation.post_effect,
            )
        )
    return CompositionFrameTrace(
        backend=backend,
        plan_sha256=plan.manifest_sha256,
        scope_id=root.scope_id,
        project_frame=project_frame,
        project_pts=project_pts,
        interval=interval,
        stack=interval.stack,
        layers=tuple(traces),
        final_composited=final_composited,
    )


@dataclass(frozen=True)
class DiagnosticDiscriminator:
    """One staged experiment that changes exactly one semantic boundary."""

    name: Literal[
        "id_plates",
        "actual_rasters_identity_placement",
        "restore_raster_placement",
        "restore_spatial",
        "restore_effects",
    ]
    replace_rasters_with_id_plates: bool
    preserve_authored_windows: bool
    preserve_z_order: bool
    apply_raster_placement: bool
    apply_spatial: bool
    apply_effects: bool


DIAGNOSTIC_DISCRIMINATORS = (
    DiagnosticDiscriminator("id_plates", True, True, True, True, True, False),
    DiagnosticDiscriminator(
        "actual_rasters_identity_placement", False, True, True, False, False, False
    ),
    DiagnosticDiscriminator(
        "restore_raster_placement", False, True, True, True, False, False
    ),
    DiagnosticDiscriminator("restore_spatial", False, True, True, True, True, False),
    DiagnosticDiscriminator("restore_effects", False, True, True, True, True, True),
)


def id_plate_rgba(semantic_id: str) -> tuple[int, int, int, int]:
    """Return a stable opaque diagnostic color for one semantic ID."""

    digest = hashlib.sha256(semantic_id.encode("utf-8")).digest()
    # Reserve a visible floor so black or near-transparent failures remain
    # visually distinct from a valid plate.
    return tuple(48 + byte % 192 for byte in digest[:3]) + (255,)


@dataclass(frozen=True)
class CompositionReductionCandidate:
    """One valid dependency slice of a single canonical interval."""

    plan_sha256: str
    scope_id: PlanId
    interval: CompositionInterval
    retained_stack: tuple[StackItem, ...]
    retained_layer_ids: tuple[PlanId, ...]
    retained_transition_ids: tuple[PlanId, ...]
    retained_scope_ids: tuple[PlanId, ...]
    retained_source_ids: tuple[PlanId, ...]


@dataclass(frozen=True)
class CompositionReductionResult:
    original: CompositionReductionCandidate
    minimal: CompositionReductionCandidate
    attempts: int


def _candidate_for_stack(
    plan: CompositionPlan,
    scope: CompositionScopePlan,
    interval: CompositionInterval,
    stack: tuple[StackItem, ...],
) -> CompositionReductionCandidate:
    scope_by_id = {item.scope_id: item for item in plan.scopes}
    retained_layers: set[str] = set()
    retained_transitions: set[str] = set()
    retained_scopes: set[str] = {scope.scope_id}
    retained_sources: set[str] = set()

    def retain_layer(owner_scope: CompositionScopePlan, layer_id: str) -> None:
        layer = next(item for item in owner_scope.layers if item.layer_id == layer_id)
        if layer.layer_id in retained_layers:
            return
        retained_layers.add(layer.layer_id)
        if layer.source.kind == "module":
            assert layer.source.ref is not None
            child = scope_by_id[layer.source.ref]
            retained_scopes.add(child.scope_id)
            for child_interval in child.intervals:
                for child_item in child_interval.stack:
                    if child_item.kind == "layer":
                        retain_layer(child, child_item.ref)
                    else:
                        retain_transition(child, child_item.ref)
        elif layer.source.ref is not None:
            retained_sources.add(layer.source.ref)

    def retain_transition(owner_scope: CompositionScopePlan, transition_id: str) -> None:
        transition = next(
            item for item in owner_scope.transitions if item.transition_id == transition_id
        )
        if transition.transition_id in retained_transitions:
            return
        retained_transitions.add(transition.transition_id)
        for layer_id in (
            transition.outgoing.composed_sources
            + transition.incoming.composed_sources
        ):
            retain_layer(owner_scope, layer_id)

    for item in stack:
        if item.kind == "layer":
            retain_layer(scope, item.ref)
        else:
            retain_transition(scope, item.ref)
    return CompositionReductionCandidate(
        plan_sha256=plan.manifest_sha256,
        scope_id=scope.scope_id,
        interval=interval,
        retained_stack=stack,
        retained_layer_ids=tuple(sorted(retained_layers)),
        retained_transition_ids=tuple(sorted(retained_transitions)),
        retained_scope_ids=tuple(sorted(retained_scopes)),
        retained_source_ids=tuple(sorted(retained_sources)),
    )


def reduce_failing_interval(
    plan: CompositionPlan,
    *,
    project_frame: int,
    failure_persists: Callable[[CompositionReductionCandidate], bool],
) -> CompositionReductionResult:
    """Delta-debug one failing root interval without mutating semantic IR.

    Main callers:
    - A low-resolution runner that can execute an interval slice and answer
      whether the same pixel/timing mismatch remains.

    The algorithm is ordinary ``ddmin`` over the canonical stack.  Each
    candidate includes the recursive module/transition/source dependency
    closure needed by a lowering adapter.  Unrelated historical/future sources
    therefore disappear before any GPU graph is built.
    """

    root = _root_scope(plan)
    interval = _interval_at(root, project_frame)
    original = _candidate_for_stack(plan, root, interval, interval.stack)
    attempts = 1
    if not failure_persists(original):
        raise CompositionDiagnosticError(
            "the supplied failure oracle does not reproduce on the full interval stack"
        )
    current = interval.stack
    granularity = 2
    while len(current) >= 2:
        chunk_size = (len(current) + granularity - 1) // granularity
        reduced = False
        for start in range(0, len(current), chunk_size):
            candidate_stack = current[:start] + current[start + chunk_size :]
            if not candidate_stack:
                continue
            candidate = _candidate_for_stack(
                plan, root, interval, candidate_stack
            )
            attempts += 1
            if failure_persists(candidate):
                current = candidate_stack
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if reduced:
            continue
        if granularity >= len(current):
            break
        granularity = min(len(current), granularity * 2)
    minimal = _candidate_for_stack(plan, root, interval, current)
    return CompositionReductionResult(
        original=original,
        minimal=minimal,
        attempts=attempts,
    )


__all__ = [
    "CompositionDiagnosticError",
    "CompositionFrameTrace",
    "CompositionReductionCandidate",
    "CompositionReductionResult",
    "DIAGNOSTIC_DISCRIMINATORS",
    "DiagnosticDiscriminator",
    "LayerBoundaryObservation",
    "PixelBoundaryObservation",
    "SemanticLayerTrace",
    "SemanticTraceDivergence",
    "first_trace_divergence",
    "id_plate_rgba",
    "reduce_failing_interval",
    "trace_composition_frame",
]
