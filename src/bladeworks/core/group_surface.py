"""Plan expanded RGBA surfaces for nested Final Cut composition groups.

Architecture map
================

``GeometryPlan.snapshot(time).composed_quad``
    -> :class:`QuadTrajectory` enumerates every decoded frame plus declared
       keyframe/retime/transition sentinels
    -> child quads are unioned into one expanded :class:`RenderSurface`
    -> each owner transform maps the *whole previous surface* in order
    -> every destination quad is rebased from project coordinates into the
       next surface's pixel coordinates
    -> the root FFmpeg compositor allocates those surfaces and clips only at
       the owning container or final project canvas

Important invariants
--------------------

* A caller must explicitly certify every trajectory as bounded. Curves whose
  extrema cannot be enumerated fail closed.
* Sampling uses exact :class:`fractions.Fraction` times. It includes every
  decoded frame, both interval boundaries, and caller-supplied sentinel and
  extrema times.
* A child quad is transformed before its parent's surface. Parent transforms
  never start again from the project-canvas quad.
* Surface origins stay explicit. Negative project coordinates are rebased;
  they are not rounded away or clipped.

Main callers:
- The FFmpeg group's deepest-first ``RenderGroup``/``_Layer`` fold.
- XYZT evidence diagnostics that prove an intermediate surface retains pixels
  a later parent transform can move back into view.

Why this exists:
The current compositor begins every nested group on a project-sized canvas.
That destroys off-canvas child pixels before the parent transform sees them.
This module freezes the allocation and coordinate contract independently of
FFmpeg graph syntax so central integration is a small, reviewable adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Callable

from .geometry import (
    FrameGeometry,
    GeometryPlan,
    GeometryValidationError,
    Quad,
    RenderSurface,
    TransformState,
    render_surface_for_quads,
    transform_points,
)


class GroupSurfaceError(ValueError):
    """Base error for an unsafe or ambiguous intermediate group surface."""


class GroupSurfaceValidationError(GroupSurfaceError):
    """Raised when intervals, samples, or hierarchy ownership are invalid."""


class UnboundedGroupCurveError(GroupSurfaceError):
    """Raised when a curve cannot prove that all relevant extrema were seen."""


QuadEvaluator = Callable[[Fraction], Quad]
TransformEvaluator = Callable[[Fraction], TransformState]
SpatialEvaluator = Callable[[Fraction, Quad], Quad]


def _exact_time(value: object, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise GroupSurfaceValidationError(f"{name} must be an exact Fraction, not bool")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        raise GroupSurfaceValidationError(f"{name} must be an exact Fraction, not float")
    raise GroupSurfaceValidationError(
        f"{name} must be an exact Fraction, got {type(value).__name__}"
    )


def _validate_quad(quad: object, *, name: str) -> Quad:
    if not isinstance(quad, tuple) or len(quad) != 4:
        raise GroupSurfaceValidationError(f"{name} must contain exactly four points")
    result: list[tuple[float, float]] = []
    for point_index, point in enumerate(quad):
        if not isinstance(point, tuple) or len(point) != 2:
            raise GroupSurfaceValidationError(
                f"{name}[{point_index}] must contain exactly two coordinates"
            )
        coordinates: list[float] = []
        for axis, value in zip(("x", "y"), point, strict=True):
            if isinstance(value, bool):
                raise GroupSurfaceValidationError(
                    f"{name}[{point_index}].{axis} must be finite"
                )
            try:
                coordinate = float(value)
            except (TypeError, ValueError) as error:
                raise GroupSurfaceValidationError(
                    f"{name}[{point_index}].{axis} must be finite"
                ) from error
            if not math.isfinite(coordinate):
                raise GroupSurfaceValidationError(
                    f"{name}[{point_index}].{axis} must be finite"
                )
            coordinates.append(coordinate)
        result.append((coordinates[0], coordinates[1]))
    xs = tuple(point[0] for point in result)
    ys = tuple(point[1] for point in result)
    if max(xs) <= min(xs) or max(ys) <= min(ys):
        raise GroupSurfaceValidationError(f"{name} is degenerate")
    return tuple(result)  # type: ignore[return-value]


@dataclass(frozen=True)
class SamplingWindow:
    """One finite visible interval sampled on exact decoded-frame boundaries.

    ``sentinel_times`` name semantic boundaries such as keyframes, retime
    segment joins, freezes, reversals, and transition handles.
    ``extrema_times`` name calibrated in-between extrema that decoded frame
    enumeration alone would not prove.

    Main callers:
    - :class:`QuadTrajectory` and :class:`TransformTrajectory`.
    """

    start: Fraction
    end: Fraction
    frame_duration: Fraction
    sentinel_times: tuple[Fraction, ...] = ()
    extrema_times: tuple[Fraction, ...] = ()

    def __post_init__(self) -> None:
        for name in ("start", "end", "frame_duration"):
            object.__setattr__(self, name, _exact_time(getattr(self, name), name=name))
        if self.end <= self.start:
            raise GroupSurfaceValidationError("sampling window must have positive duration")
        if self.frame_duration <= 0:
            raise GroupSurfaceValidationError("frame_duration must be positive")
        for collection_name in ("sentinel_times", "extrema_times"):
            values = tuple(
                _exact_time(value, name=f"{collection_name}[{index}]")
                for index, value in enumerate(getattr(self, collection_name))
            )
            for value in values:
                if value < self.start or value > self.end:
                    raise GroupSurfaceValidationError(
                        f"{collection_name} time {value} falls outside "
                        f"[{self.start}, {self.end}]"
                    )
            object.__setattr__(self, collection_name, values)

    def sample_times(self, *, max_samples: int) -> tuple[Fraction, ...]:
        """Return decoded-frame, endpoint, sentinel, and extrema times.

        The inclusive ``end`` sample is an allocation sentinel, not a claim
        that the half-open media interval decodes a frame there. It preserves
        exact endpoint holds needed by transitions and nested transforms.
        """

        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0:
            raise GroupSurfaceValidationError("max_samples must be a positive integer")
        frame_count = math.ceil((self.end - self.start) / self.frame_duration)
        if frame_count + 2 + len(self.sentinel_times) + len(self.extrema_times) > max_samples:
            raise GroupSurfaceValidationError(
                f"sampling window exceeds max_samples={max_samples}"
            )
        values = {self.start, self.end, *self.sentinel_times, *self.extrema_times}
        time = self.start
        while time < self.end:
            values.add(time)
            if len(values) > max_samples:
                raise GroupSurfaceValidationError(
                    f"sampling window exceeds max_samples={max_samples}"
                )
            time += self.frame_duration
        return tuple(sorted(values))


@dataclass(frozen=True)
class QuadTrajectory:
    """A child layer's already-composed project-space quad over time."""

    id: str
    window: SamplingWindow
    evaluate: QuadEvaluator
    bounded: bool

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise GroupSurfaceValidationError("quad trajectory id must be non-empty")
        if not isinstance(self.window, SamplingWindow):
            raise GroupSurfaceValidationError("quad trajectory requires a SamplingWindow")
        if not callable(self.evaluate):
            raise GroupSurfaceValidationError("quad trajectory evaluate must be callable")
        if not isinstance(self.bounded, bool):
            raise GroupSurfaceValidationError("quad trajectory bounded must be bool")


@dataclass(frozen=True)
class TransformTrajectory:
    """One group owner's affine transform, ordered child then parent."""

    id: str
    window: SamplingWindow
    evaluate: TransformEvaluator
    bounded: bool

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise GroupSurfaceValidationError("transform trajectory id must be non-empty")
        if not isinstance(self.window, SamplingWindow):
            raise GroupSurfaceValidationError("transform trajectory requires a SamplingWindow")
        if not callable(self.evaluate):
            raise GroupSurfaceValidationError("transform trajectory evaluate must be callable")
        if not isinstance(self.bounded, bool):
            raise GroupSurfaceValidationError("transform trajectory bounded must be bool")


@dataclass(frozen=True)
class SpatialTrajectory:
    """One owner's full corner-pin-plus-affine project-space mapping.

    Unlike :class:`TransformTrajectory`, the evaluator accepts arbitrary
    points. That is necessary when an expanded child surface extends outside
    the four project corners and the owner includes a projective corner pin.
    """

    id: str
    window: SamplingWindow
    evaluate: SpatialEvaluator
    bounded: bool

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise GroupSurfaceValidationError("spatial trajectory id must be non-empty")
        if not isinstance(self.window, SamplingWindow):
            raise GroupSurfaceValidationError("spatial trajectory requires a SamplingWindow")
        if not callable(self.evaluate):
            raise GroupSurfaceValidationError("spatial trajectory evaluate must be callable")
        if not isinstance(self.bounded, bool):
            raise GroupSurfaceValidationError("spatial trajectory bounded must be bool")


@dataclass(frozen=True)
class MemberSample:
    """One child's quad after each hierarchy stage at one exact time."""

    member_id: str
    time: Fraction
    project_quads: tuple[Quad, ...]
    surface_quads: tuple[Quad, ...]


@dataclass(frozen=True)
class SurfaceTransformSample:
    """Mapping of a complete intermediate surface through one owner."""

    owner_id: str
    time: Fraction
    source_project_quad: Quad
    destination_project_quad: Quad
    destination_surface_quad: Quad


@dataclass(frozen=True)
class GroupSurfacePlan:
    """Decision-complete allocation and rebasing plan for a nested group.

    ``surfaces[0]`` is the child-composition surface. Each later entry is the
    expanded surface after the owner at the matching hierarchy level.
    ``surface_transforms[level]`` maps the whole ``surfaces[level]`` through
    ``owners[level]`` into ``surfaces[level + 1]``.
    """

    sample_times: tuple[Fraction, ...]
    surfaces: tuple[RenderSurface, ...]
    member_samples: tuple[MemberSample, ...]
    surface_transforms: tuple[tuple[SurfaceTransformSample, ...], ...]

    @property
    def child_surface(self) -> RenderSurface:
        return self.surfaces[0]

    @property
    def output_surface(self) -> RenderSurface:
        return self.surfaces[-1]


def surface_project_quad(surface: RenderSurface) -> Quad:
    """Return a surface's four corners in canonical project coordinates."""

    left = float(surface.origin_x)
    top = float(surface.origin_y)
    right = float(surface.origin_x + surface.width)
    bottom = float(surface.origin_y + surface.height)
    return ((left, top), (right, top), (left, bottom), (right, bottom))


def rebase_quad(surface: RenderSurface, quad: Quad) -> Quad:
    """Translate a project-space quad into one surface's pixel coordinates."""

    checked = _validate_quad(quad, name="quad")
    return tuple(surface.project_to_surface(point) for point in checked)  # type: ignore[return-value]


def trajectory_from_geometry_plan(
    id: str,
    plan: GeometryPlan,
    *,
    frame_duration: Fraction,
    sentinel_times: tuple[Fraction, ...] = (),
    extrema_times: tuple[Fraction, ...] = (),
    bounded: bool,
) -> QuadTrajectory:
    """Adapt the typed geometry snapshot used by the current ``_Layer`` flow.

    Main callers:
    - The root FFmpeg integration after it builds the same ``GeometryPlan``
      currently consumed by ``_geometry_stage_filters``.
    """

    window = SamplingWindow(
        start=plan.window.render_start,
        end=plan.window.render_end,
        frame_duration=frame_duration,
        sentinel_times=sentinel_times,
        extrema_times=extrema_times,
    )
    return QuadTrajectory(
        id=id,
        window=window,
        evaluate=lambda time: plan.snapshot(time).composed_quad,
        bounded=bounded,
    )


def _solve_linear_system(matrix: list[list[float]], values: list[float]) -> tuple[float, ...]:
    """Solve one small finite system with pivoted Gaussian elimination."""

    size = len(values)
    augmented = [row[:] + [value] for row, value in zip(matrix, values, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise GroupSurfaceValidationError("projective owner quad is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                observed - factor * pivot_value
                for observed, pivot_value in zip(
                    augmented[row],
                    augmented[column],
                    strict=True,
                )
            ]
    return tuple(augmented[index][-1] for index in range(size))


def projective_map_points(
    frame: FrameGeometry,
    destination_canvas_quad: Quad,
    points: Quad,
) -> Quad:
    """Map arbitrary project points through a canvas-to-quad homography.

    Main callers:
    - :func:`spatial_trajectory_from_geometry_plan`.

    Why this exists:
    - A parent's corner pin must transform pixels outside the project canvas,
      not merely its original four corners. The homography is therefore
      solved from the canvas corners and evaluated on the complete expanded
      child-surface quad.
    """

    destination = _validate_quad(destination_canvas_quad, name="destination canvas quad")
    source = (
        (0.0, 0.0),
        (float(frame.project_width), 0.0),
        (0.0, float(frame.project_height)),
        (float(frame.project_width), float(frame.project_height)),
    )
    matrix: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    h00, h01, h02, h10, h11, h12, h20, h21 = _solve_linear_system(matrix, values)
    result: list[tuple[float, float]] = []
    for index, (x, y) in enumerate(_validate_quad(points, name="source points")):
        denominator = h20 * x + h21 * y + 1.0
        if abs(denominator) < 1e-12:
            raise GroupSurfaceValidationError(
                f"projective owner maps source point {index} to infinity"
            )
        result.append(
            (
                (h00 * x + h01 * y + h02) / denominator,
                (h10 * x + h11 * y + h12) / denominator,
            )
        )
    return _validate_quad(tuple(result), name="projective mapped quad")


def spatial_trajectory_from_geometry_plan(
    id: str,
    plan: GeometryPlan,
    *,
    frame_duration: Fraction,
    sentinel_times: tuple[Fraction, ...] = (),
    extrema_times: tuple[Fraction, ...] = (),
    bounded: bool,
) -> SpatialTrajectory:
    """Adapt a group owner's full corner-pin-plus-affine geometry mapping."""

    window = SamplingWindow(
        start=plan.window.render_start,
        end=plan.window.render_end,
        frame_duration=frame_duration,
        sentinel_times=sentinel_times,
        extrema_times=extrema_times,
    )

    def evaluate(time: Fraction, points: Quad) -> Quad:
        return projective_map_points(
            plan.frame,
            plan.snapshot(time).composed_quad,
            points,
        )

    return SpatialTrajectory(
        id=id,
        window=window,
        evaluate=evaluate,
        bounded=bounded,
    )


OwnerTrajectory = TransformTrajectory | SpatialTrajectory


def _owner_map(
    frame: FrameGeometry,
    owner: OwnerTrajectory,
    time: Fraction,
    points: Quad,
) -> Quad:
    if isinstance(owner, TransformTrajectory):
        try:
            state = owner.evaluate(time)
        except Exception as error:
            raise GroupSurfaceValidationError(
                f"owner {owner.id!r} could not evaluate at {time}"
            ) from error
        if not isinstance(state, TransformState):
            raise GroupSurfaceValidationError(
                f"owner {owner.id!r} returned {type(state).__name__}, "
                "expected TransformState"
            )
        return transform_points(frame, state, points)
    try:
        return owner.evaluate(time, points)
    except Exception as error:
        if isinstance(error, GroupSurfaceError):
            raise
        raise GroupSurfaceValidationError(
            f"owner {owner.id!r} could not evaluate at {time}"
        ) from error


def plan_group_surfaces(
    frame: FrameGeometry,
    members: tuple[QuadTrajectory, ...],
    *,
    owners: tuple[OwnerTrajectory, ...] = (),
    guard_pixels: int = 2,
    max_samples: int = 200_000,
) -> GroupSurfacePlan:
    """Allocate and rebase every surface in a child-before-parent hierarchy.

    Main callers:
    - The central FFmpeg compositor before ``_compose_item_batch``.

    Why this exists:
    - FFmpeg needs fixed surface dimensions for a filter graph. This routine
      conservatively unions all finite decoded-frame and semantic-boundary
      quads before graph construction, then exposes exact origins for overlay
      and perspective expressions.
    """

    if not isinstance(frame, FrameGeometry):
        raise GroupSurfaceValidationError("frame must be FrameGeometry")
    if not members:
        raise GroupSurfaceValidationError("group surface requires at least one member")
    ids = tuple(member.id for member in members)
    if len(set(ids)) != len(ids):
        raise GroupSurfaceValidationError("group member ids must be unique")
    owner_ids = tuple(owner.id for owner in owners)
    if len(set(owner_ids)) != len(owner_ids):
        raise GroupSurfaceValidationError("group owner ids must be unique")
    for trajectory in (*members, *owners):
        if not trajectory.bounded:
            raise UnboundedGroupCurveError(
                f"trajectory {trajectory.id!r} has no finite extrema proof"
            )

    member_times: dict[str, tuple[Fraction, ...]] = {
        member.id: member.window.sample_times(max_samples=max_samples)
        for member in members
    }
    all_times = set(time for times in member_times.values() for time in times)
    for owner in owners:
        if min(all_times) < owner.window.start or max(all_times) > owner.window.end:
            raise GroupSurfaceValidationError(
                f"owner {owner.id!r} does not cover all visible child samples"
            )
        for time in (*owner.window.sentinel_times, *owner.window.extrema_times):
            if any(
                member.window.start <= time <= member.window.end
                for member in members
            ):
                all_times.add(time)
    if len(all_times) > max_samples:
        raise GroupSurfaceValidationError(
            f"group plan exceeds max_samples={max_samples}"
        )
    ordered_times = tuple(sorted(all_times))

    raw_member_quads: dict[tuple[str, Fraction], Quad] = {}
    stage_member_quads: dict[tuple[str, Fraction], list[Quad]] = {}
    for member in members:
        for time in member_times[member.id]:
            try:
                quad = _validate_quad(
                    member.evaluate(time),
                    name=f"member {member.id!r} quad at {time}",
                )
            except GroupSurfaceError:
                raise
            except Exception as error:
                raise GroupSurfaceValidationError(
                    f"member {member.id!r} could not evaluate at {time}"
                ) from error
            raw_member_quads[(member.id, time)] = quad
            stage_member_quads[(member.id, time)] = [quad]

    surfaces: list[RenderSurface] = [
        render_surface_for_quads(
            frame,
            tuple(raw_member_quads.values()),
            guard_pixels=guard_pixels,
        )
    ]
    surface_transforms: list[tuple[SurfaceTransformSample, ...]] = []

    for owner in owners:
        for member in members:
            for time in member_times[member.id]:
                previous = stage_member_quads[(member.id, time)][-1]
                try:
                    transformed = _validate_quad(
                        _owner_map(frame, owner, time, previous),
                        name=f"owner {owner.id!r} member quad at {time}",
                    )
                except GeometryValidationError as error:
                    raise GroupSurfaceValidationError(str(error)) from error
                stage_member_quads[(member.id, time)].append(transformed)

        source_surface = surfaces[-1]
        source_quad = surface_project_quad(source_surface)
        destination_quads: dict[Fraction, Quad] = {}
        for time in ordered_times:
            try:
                destination_quads[time] = _validate_quad(
                    _owner_map(frame, owner, time, source_quad),
                    name=f"owner {owner.id!r} surface quad at {time}",
                )
            except GeometryValidationError as error:
                raise GroupSurfaceValidationError(str(error)) from error
        destination_surface = render_surface_for_quads(
            frame,
            tuple(destination_quads.values()),
            guard_pixels=guard_pixels,
        )
        surfaces.append(destination_surface)
        surface_transforms.append(
            tuple(
                SurfaceTransformSample(
                    owner_id=owner.id,
                    time=time,
                    source_project_quad=source_quad,
                    destination_project_quad=destination_quads[time],
                    destination_surface_quad=rebase_quad(
                        destination_surface,
                        destination_quads[time],
                    ),
                )
                for time in ordered_times
            )
        )

    member_samples: list[MemberSample] = []
    for member in members:
        for time in member_times[member.id]:
            project_quads = tuple(stage_member_quads[(member.id, time)])
            member_samples.append(
                MemberSample(
                    member_id=member.id,
                    time=time,
                    project_quads=project_quads,
                    surface_quads=tuple(
                        rebase_quad(surface, quad)
                        for surface, quad in zip(surfaces, project_quads, strict=True)
                    ),
                )
            )

    return GroupSurfacePlan(
        sample_times=ordered_times,
        surfaces=tuple(surfaces),
        member_samples=tuple(member_samples),
        surface_transforms=tuple(surface_transforms),
    )


__all__ = [
    "GroupSurfaceError",
    "GroupSurfacePlan",
    "GroupSurfaceValidationError",
    "MemberSample",
    "QuadTrajectory",
    "SamplingWindow",
    "SpatialTrajectory",
    "SurfaceTransformSample",
    "TransformTrajectory",
    "UnboundedGroupCurveError",
    "plan_group_surfaces",
    "projective_map_points",
    "rebase_quad",
    "spatial_trajectory_from_geometry_plan",
    "surface_project_quad",
    "trajectory_from_geometry_plan",
]
