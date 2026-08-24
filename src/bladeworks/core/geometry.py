"""Plan Final Cut crop, conform, corner-pin, and transform geometry.

Architecture map
================

``TransformAdjustment`` / ``CropAdjustment`` / typed animation tracks
    -> validate finite values, exact timing, and non-degenerate scale
    -> select the active crop operation
    -> place the result on a project-sized transparent RGBA canvas
    -> reserve the effects-and-masks insertion point
    -> apply corner pinning
    -> apply the anchor-based transform
    -> emit a typed ``GeometrySnapshot`` or stock-FFmpeg filter fragments

Important invariants
--------------------

* Timeline coordinates remain ``Fraction`` values.  Float time is rejected.
* Transition pre-roll and post-roll hold the clip's first/last geometry; they
  never extrapolate an animation beyond the clip.
* ``crop`` and ``pan`` resolve camera/reference rectangles and make pixels
  outside those windows transparent before the camera warp. ``trim`` remains
  a direct extraction without camera enlargement. A typed full-source quad is
  still retained so placement math stays in original source coordinates.
* Conform always produces a project-sized RGBA frame.  Later composition can
  therefore use one alpha-correct canvas contract for every layer.
* A zero scale, a sign-changing animated scale, or an implicit mirror is a
  hard error.  A caller may opt into a consistently negative component with
  ``allow_mirrored_scale=True``; crossing through zero is still invalid.
* Final Cut spatial values use one percent of project height as a unit.  The
  screen Y axis is inverted only at the FFmpeg/pixel boundary.

Main callers:
- The Wave 2 FFmpeg geometry integration, after parser/compiler construction.
- A/B diagnostics that sample exact animation states at event frames.

Why this exists:
The legacy FFmpeg builder mixes crop, conform, scale, rotation, and overlay
placement across separate helper functions.  That makes anchor pivots and
trim transparency impossible to reason about as one transform.  This module
freezes the geometry contract independently, before the shared builder is
rewired to consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
from typing import Literal, Mapping, Optional, TypeAlias

from .animation import TimelineAnimatedScalar, TimelineAnimatedVec2, ken_burns_progress
from .compositor import FCP_NORMAL_SOURCE_OVER_GAMMA
from .model import CropAdjustment, CropRect, RenderTransformAnimation, TransformAdjustment
from .render_profile import current_render_profile


ConformMode: TypeAlias = Literal["fit", "fill", "none"]
CropMode: TypeAlias = Literal["crop", "trim", "pan"]
Point: TypeAlias = tuple[float, float]
Quad: TypeAlias = tuple[Point, Point, Point, Point]
ExpressionPoint: TypeAlias = tuple[str, str]
ExpressionQuad: TypeAlias = tuple[
    ExpressionPoint,
    ExpressionPoint,
    ExpressionPoint,
    ExpressionPoint,
]
TRANSPARENT_PERSPECTIVE_BORDER = 2


class GeometryError(ValueError):
    """Base error for geometry that cannot be rendered deterministically."""


class GeometryValidationError(GeometryError):
    """Raised when source geometry or an adjustment is invalid."""


class UnsupportedGeometryAnimationError(GeometryError):
    """Raised when a single FFmpeg fragment would change animation semantics."""


def _exact_time(value: object, *, name: str) -> Fraction:
    """Accept exact rational time and reject binary-float timeline values."""

    if isinstance(value, bool):
        raise GeometryValidationError(f"{name} must be an exact Fraction, not bool")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        raise GeometryValidationError(f"{name} must be an exact Fraction, not float")
    raise GeometryValidationError(
        f"{name} must be an exact Fraction, got {type(value).__name__}"
    )


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise GeometryValidationError(f"{name} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise GeometryValidationError(f"{name} must be finite")
    return result


def _pair(value: tuple[float, float], *, name: str) -> Point:
    if not isinstance(value, tuple) or len(value) != 2:
        raise GeometryValidationError(f"{name} must contain exactly two values")
    return (_finite(value[0], name=f"{name}[0]"), _finite(value[1], name=f"{name}[1]"))


def _number(value: float) -> str:
    """Format a finite filter number identically across repeated plans."""

    exact = _finite(value, name="filter value")
    if abs(exact) < 5e-13:
        exact = 0.0
    return format(exact, ".12g")


def _round_half_up(value: float) -> int:
    if value < 0:
        raise GeometryValidationError("pixel extent cannot be negative")
    return int(math.floor(value + 0.5))


def _round_half_up_signed(value: float) -> int:
    """Match FFmpeg placement rounding for positive and negative origins."""

    return int(math.floor(_finite(value, name="pixel origin") + 0.5))


@dataclass(frozen=True)
class GeometryWindow:
    """The clip and expanded render intervals in absolute project time.

    The render interval may begin before the clip for an incoming transition
    and may end after it for an outgoing transition.  ``clip_time`` clamps
    those handles to the clip endpoints so geometry holds during pre-roll.
    """

    clip_start: Fraction
    clip_duration: Fraction
    render_start: Fraction
    render_duration: Fraction

    def __post_init__(self) -> None:
        for name in ("clip_start", "clip_duration", "render_start", "render_duration"):
            object.__setattr__(self, name, _exact_time(getattr(self, name), name=name))
        if self.clip_duration <= 0:
            raise GeometryValidationError("clip_duration must be positive")
        if self.render_duration <= 0:
            raise GeometryValidationError("render_duration must be positive")
        if self.render_start >= self.clip_end or self.render_end <= self.clip_start:
            raise GeometryValidationError("render interval must overlap the clip interval")

    @property
    def clip_end(self) -> Fraction:
        return self.clip_start + self.clip_duration

    @property
    def render_end(self) -> Fraction:
        return self.render_start + self.render_duration

    @property
    def transition_pre_roll(self) -> Fraction:
        return self.clip_start - self.render_start

    @property
    def transition_post_roll(self) -> Fraction:
        return self.render_end - self.clip_end

    def clip_time(self, absolute_time: Fraction) -> Fraction:
        """Convert an absolute render time to a held clip-local coordinate."""

        absolute = _exact_time(absolute_time, name="absolute_time")
        if absolute < self.render_start or absolute > self.render_end:
            raise GeometryValidationError("absolute_time is outside the render interval")
        return min(max(absolute - self.clip_start, Fraction(0)), self.clip_duration)


@dataclass(frozen=True)
class FrameGeometry:
    """Source/project dimensions needed to resolve percentages into pixels."""

    source_width: int
    source_height: int
    project_width: int
    project_height: int

    def __post_init__(self) -> None:
        for name in ("source_width", "source_height", "project_width", "project_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GeometryValidationError(f"{name} must be a positive integer")

    @property
    def spatial_unit(self) -> float:
        """Pixels represented by one Final Cut spatial coordinate unit."""

        return self.project_height / 100.0


@dataclass(frozen=True)
class PixelRect:
    """An integer extraction/reference rectangle in source coordinates.

    Camera-reference Pan rectangles may begin outside the source raster; the
    uncovered portion is transparent.  Directly executable Trim/Crop inputs
    reject negative authored edges before this record is constructed.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if min(self.width, self.height) <= 0:
            raise GeometryValidationError("pixel rectangle must have positive dimensions")


@dataclass(frozen=True)
class SourceRect:
    """A sub-pixel source rectangle in the square-pixel display raster.

    Unlike :class:`PixelRect`, this record deliberately retains fractional
    crop edges.  Final Cut can author non-integer percent-of-height values and
    the camera placement must not round those values before scale and centering
    are resolved.  Pan windows may also extend above or to the left of the
    source raster.  Those negative coordinates intentionally become
    transparent camera support; static Crop validation rejects them earlier.
    """

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if min(self.width, self.height) <= 0:
            raise GeometryValidationError("source rectangle must have positive dimensions")

    @property
    def center(self) -> Point:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


@dataclass(frozen=True)
class CanvasBounds:
    """Unclipped bounds expressed in the project's pixel coordinate system."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if self.right <= self.left or self.bottom <= self.top:
            raise GeometryValidationError("canvas bounds must have positive dimensions")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @classmethod
    def from_quad(cls, quad: Quad) -> "CanvasBounds":
        xs = tuple(point[0] for point in quad)
        ys = tuple(point[1] for point in quad)
        return cls(min(xs), min(ys), max(xs), max(ys))

    def union(self, other: "CanvasBounds") -> "CanvasBounds":
        return CanvasBounds(
            min(self.left, other.left),
            min(self.top, other.top),
            max(self.right, other.right),
            max(self.bottom, other.bottom),
        )


@dataclass(frozen=True)
class RenderSurface:
    """An integer RGBA surface that preserves pixels outside the project.

    ``origin_x``/``origin_y`` locate surface pixel ``(0, 0)`` in project
    coordinates.  They may be negative.  This explicit origin is the missing
    piece in project-sized nested composition: a parent transform can move an
    off-canvas child back into view only if those pixels survive on a larger
    intermediate surface.

    Main callers:
    - The group compositor when it allocates a transparent child surface.
    - Geometry diagnostics that verify child-before-parent composition.
    """

    origin_x: int
    origin_y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("origin_x", "origin_y", "width", "height"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise GeometryValidationError(f"{name} must be an integer")
        if min(self.width, self.height) <= 0:
            raise GeometryValidationError("render surface must have positive dimensions")

    @property
    def bounds(self) -> CanvasBounds:
        return CanvasBounds(
            float(self.origin_x),
            float(self.origin_y),
            float(self.origin_x + self.width),
            float(self.origin_y + self.height),
        )

    def project_to_surface(self, point: Point) -> Point:
        return (point[0] - self.origin_x, point[1] - self.origin_y)

    def surface_to_project(self, point: Point) -> Point:
        return (point[0] + self.origin_x, point[1] + self.origin_y)


@dataclass(frozen=True)
class CameraPlacement:
    """Place a full source by treating one rectangle as the camera reference.

    Crop and Pan use this record.  ``source_quad`` intentionally describes the
    full source, while ``reference_quad`` describes the authored camera window.
    The project canvas is the only clip boundary; pixels outside the reference
    rectangle remain available when their transformed coordinates are visible.
    """

    reference_rect: SourceRect
    conform: ConformMode
    scaled_width: int
    scaled_height: int
    origin_x: int
    origin_y: int
    exact_scale: float
    exact_origin_x: float
    exact_origin_y: float
    source_quad: Quad
    reference_quad: Quad
    base_conform_multiplier: float = 1.0


def pan_base_conform_multiplier(frame: FrameGeometry, conform: str) -> float:
    """Return the ordinary source conform scale used before Pan fill-back.

    Final Cut first Fits or Fills the full source into the project, then enlarges
    the moving retained rectangle by the reciprocal of its smallest retained
    fraction. This helper returns only that first, source-level scale. The
    dynamic reciprocal belongs to the Pan expression itself.

    Main callers:
    - :meth:`GeometryPlan.snapshot` for typed evidence.
    - ``ffmpeg._ken_burns_filter`` for the matching expression graph.
    """

    mode = GeometryPlan._validate_conform(conform)
    source_fit = min(
        frame.project_width / frame.source_width,
        frame.project_height / frame.source_height,
    )
    if mode != "fill":
        return source_fit
    return max(
        frame.project_width / frame.source_width,
        frame.project_height / frame.source_height,
    )


def resolve_crop_camera_placement(
    frame: FrameGeometry,
    reference_rect: SourceRect,
    conform: str,
    *,
    allow_outside_source: bool = False,
) -> CameraPlacement:
    """Apply Final Cut's base conform and native Crop fill-back rule.

    Final Cut first Fits or Fills the full source into its owning clip, then
    enlarges the retained Crop window by the reciprocal of its smallest
    retained source fraction. This is deliberately not equivalent to choosing
    Fit or Fill directly against the Crop rectangle for every source aspect.

    Main callers:
    - :meth:`GeometryPlan.snapshot` for typed camera evidence.
    - The direct and fused FFmpeg Crop execution paths.

    Why this exists:
    A project-sized source makes this rule look like rectangle Fill, while a
    mismatched 3:2 source can make it look like rectangle Fit. One explicit
    formula prevents those two genuine oracle cases from creating divergent
    static and transition implementations.
    """

    authored_mode = GeometryPlan._validate_conform(conform)
    base_mode: ConformMode = "fit" if authored_mode == "none" else authored_mode
    if base_mode == "fit":
        base_scale = min(
            frame.project_width / frame.source_width,
            frame.project_height / frame.source_height,
        )
    else:
        base_scale = max(
            frame.project_width / frame.source_width,
            frame.project_height / frame.source_height,
        )
    retained_fraction = min(
        reference_rect.width / frame.source_width,
        reference_rect.height / frame.source_height,
    )
    if retained_fraction <= 0:
        raise GeometryValidationError("Crop camera retains no source fraction")
    placement = resolve_camera_placement(
        frame,
        reference_rect,
        "none",
        base_conform_multiplier=base_scale / retained_fraction,
        allow_outside_source=allow_outside_source,
    )
    return replace(placement, conform=base_mode)


def resolve_camera_placement(
    frame: FrameGeometry,
    reference_rect: SourceRect,
    conform: str,
    *,
    base_conform_multiplier: float = 1.0,
    allow_outside_source: bool = False,
) -> CameraPlacement:
    """Resolve Crop/Pan camera geometry once, after orientation and SAR bake.

    ``frame.source_width``/``source_height`` are required to be the display
    raster dimensions after orientation and pixel-aspect normalization.  This
    keeps Fill from accidentally using the encoded width before SAR is baked.

    Main callers:
    - :meth:`GeometryPlan.snapshot` for typed camera evidence.
    - The FFmpeg Crop/Pan integration, replacing its duplicate float math.
    """

    mode = GeometryPlan._validate_conform(conform)
    if not isinstance(allow_outside_source, bool):
        raise GeometryValidationError("allow_outside_source must be a boolean")
    if (
        not allow_outside_source
        and (
            reference_rect.x < -1e-9
            or reference_rect.x + reference_rect.width
            > frame.source_width + 1e-9
        )
    ):
        raise GeometryValidationError("camera rectangle exceeds source width")
    if (
        not allow_outside_source
        and (
            reference_rect.y < -1e-9
            or reference_rect.y + reference_rect.height
            > frame.source_height + 1e-9
        )
    ):
        raise GeometryValidationError("camera rectangle exceeds source height")
    if mode == "fit":
        scale = min(
            frame.project_width / reference_rect.width,
            frame.project_height / reference_rect.height,
        )
    elif mode == "fill":
        scale = max(
            frame.project_width / reference_rect.width,
            frame.project_height / reference_rect.height,
        )
    else:
        scale = 1.0

    multiplier = _finite(
        base_conform_multiplier,
        name="camera base conform multiplier",
    )
    if multiplier <= 0:
        raise GeometryValidationError("camera base conform multiplier must be positive")
    scale *= multiplier
    scaled_width = max(1, _round_half_up(frame.source_width * scale))
    scaled_height = max(1, _round_half_up(frame.source_height * scale))
    center_x, center_y = reference_rect.center
    exact_origin_x = frame.project_width / 2.0 - center_x * scale
    exact_origin_y = frame.project_height / 2.0 - center_y * scale
    origin_x = _round_half_up_signed(exact_origin_x)
    origin_y = _round_half_up_signed(exact_origin_y)

    def place(x: float, y: float) -> Point:
        return (exact_origin_x + x * scale, exact_origin_y + y * scale)

    source_quad = (
        place(0.0, 0.0),
        place(float(frame.source_width), 0.0),
        place(0.0, float(frame.source_height)),
        place(float(frame.source_width), float(frame.source_height)),
    )
    right = reference_rect.x + reference_rect.width
    bottom = reference_rect.y + reference_rect.height
    reference_quad = (
        place(reference_rect.x, reference_rect.y),
        place(right, reference_rect.y),
        place(reference_rect.x, bottom),
        place(right, bottom),
    )
    return CameraPlacement(
        reference_rect=reference_rect,
        conform=mode,
        scaled_width=scaled_width,
        scaled_height=scaled_height,
        origin_x=origin_x,
        origin_y=origin_y,
        exact_scale=scale,
        exact_origin_x=exact_origin_x,
        exact_origin_y=exact_origin_y,
        source_quad=source_quad,
        reference_quad=reference_quad,
        base_conform_multiplier=multiplier,
    )


@dataclass(frozen=True)
class TransformState:
    position: Point
    scale: Point
    rotation_degrees: float
    anchor: Point


def transform_points(
    frame: FrameGeometry,
    state: TransformState,
    points: Quad,
) -> Quad:
    """Apply one Final Cut affine state to project-coordinate points.

    The operation is ``C + T + R(-rotation) * S * (point - (C + A))``.
    Final Cut stores position and anchor in project-height units with positive
    Y up; this boundary converts them once into top-left-origin pixels. Anchor
    moves the source origin under the transform; it is not a fixed output
    pivot. That distinction is visible whenever anchor is nonzero.

    Main callers:
    - ``GeometryPlan._transform_quad`` for exact static snapshots.
    - Static-vs-animated parity tests for the FFmpeg expression lowering.

    Why this exists:
    Keeping the numeric transform in one public kernel prevents the static
    planner and the animated FFmpeg bridge from quietly adopting different
    pivot, axis, or scale/rotation-order conventions.
    """

    width = float(frame.project_width)
    height = float(frame.project_height)
    unit = frame.spatial_unit
    center = (width / 2.0, height / 2.0)
    source_origin = (
        width / 2.0 + state.anchor[0] * unit,
        height / 2.0 - state.anchor[1] * unit,
    )
    translation = (state.position[0] * unit, -state.position[1] * unit)
    angle = math.radians(-state.rotation_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)

    def project(point: Point) -> Point:
        scaled_x = (point[0] - source_origin[0]) * state.scale[0]
        scaled_y = (point[1] - source_origin[1]) * state.scale[1]
        return (
            center[0]
            + translation[0]
            + scaled_x * cosine
            - scaled_y * sine,
            center[1]
            + translation[1]
            + scaled_x * sine
            + scaled_y * cosine,
        )

    return tuple(project(point) for point in points)  # type: ignore[return-value]


def transform_quad(frame: FrameGeometry, state: TransformState) -> Quad:
    """Project the four project-canvas corners through one affine state."""

    width = float(frame.project_width)
    height = float(frame.project_height)
    return transform_points(
        frame,
        state,
        ((0.0, 0.0), (width, 0.0), (0.0, height), (width, height)),
    )


def compose_spatial_quad(
    frame: FrameGeometry,
    *,
    corner_quad: Quad,
    transforms: tuple[TransformState, ...],
) -> Quad:
    """Compose corner pin, child affine, then every parent affine numerically.

    Each transform uses Final Cut's project-space pivot convention, even when
    the content currently extends outside the canvas.  No intermediate project
    clip is introduced.  Passing ``(child, parent)`` therefore preserves the
    required child-before-parent order.

    Main callers:
    - :meth:`GeometryPlan.snapshot` for corner-pin-plus-transform composition.
    - The group compositor when lowering nested transforms onto one expanded
      surface.
    """

    result = corner_quad
    for state in transforms:
        result = transform_points(frame, state, result)
    return result


def render_surface_for_quads(
    frame: FrameGeometry,
    quads: tuple[Quad, ...],
    *,
    guard_pixels: int = TRANSPARENT_PERSPECTIVE_BORDER,
) -> RenderSurface:
    """Allocate one deterministic unbounded surface for group composition.

    The project rectangle is always included so identity content retains the
    normal canvas coordinate system.  Bounds expand only where a child quad
    escapes it.  ``guard_pixels`` prevents perspective edge extrapolation from
    turning the outermost retained pixel opaque.
    """

    if isinstance(guard_pixels, bool) or not isinstance(guard_pixels, int):
        raise GeometryValidationError("surface guard_pixels must be an integer")
    if guard_pixels < 0:
        raise GeometryValidationError("surface guard_pixels cannot be negative")
    bounds = CanvasBounds(
        0.0,
        0.0,
        float(frame.project_width),
        float(frame.project_height),
    )
    for quad in quads:
        bounds = bounds.union(CanvasBounds.from_quad(quad))
    origin_x = math.floor(bounds.left) - guard_pixels
    origin_y = math.floor(bounds.top) - guard_pixels
    right = math.ceil(bounds.right) + guard_pixels
    bottom = math.ceil(bounds.bottom) + guard_pixels
    return RenderSurface(origin_x, origin_y, right - origin_x, bottom - origin_y)


def transform_point_expressions(
    frame: FrameGeometry,
    *,
    points: ExpressionQuad,
    position: ExpressionPoint,
    scale: ExpressionPoint,
    rotation_degrees: str,
    anchor: ExpressionPoint,
) -> ExpressionQuad:
    """Apply the affine kernel to arbitrary renderer-owned point expressions.

    All component strings come from the bounded typed animation compiler; XML
    never contributes raw FFmpeg syntax. The returned expressions intentionally
    use no time variable themselves, so leaves and groups can supply one shared
    ``on * frame_duration`` clock before this purely spatial lowering.

    Main callers:
    - ``ffmpeg._animated_affine_transform_filter``.

    Why this exists:
    FFmpeg's old animated path rasterized scale, rotation, and overlay position
    separately. A single destination quad preserves the numeric kernel's pivot
    and non-uniform scale-before-rotation behavior on a fixed project canvas.
    """

    components = (
        *position,
        *scale,
        rotation_degrees,
        *anchor,
    )
    if any(not isinstance(value, str) or not value.strip() for value in components):
        raise GeometryValidationError(
            "affine transform expressions must be non-empty renderer-owned strings"
        )

    width = str(frame.project_width)
    height = str(frame.project_height)
    unit = f"{frame.project_height}/100"
    position_x, position_y = position
    scale_x, scale_y = scale
    anchor_x, anchor_y = anchor
    center_x = f"({width}/2)"
    center_y = f"({height}/2)"
    source_origin_x = f"({center_x}+({anchor_x})*{unit})"
    source_origin_y = f"({center_y}-({anchor_y})*{unit})"
    translation_x = f"(({position_x})*{unit})"
    translation_y = f"(-({position_y})*{unit})"
    radians = f"(-({rotation_degrees})*PI/180)"

    def project(x: str, y: str) -> ExpressionPoint:
        scaled_x = f"(({x}-({source_origin_x}))*({scale_x}))"
        scaled_y = f"(({y}-({source_origin_y}))*({scale_y}))"
        return (
            f"({center_x}+{translation_x}+{scaled_x}*cos({radians})-{scaled_y}*sin({radians}))",
            f"({center_y}+{translation_y}+{scaled_x}*sin({radians})+{scaled_y}*cos({radians}))",
        )

    return tuple(project(x, y) for x, y in points)  # type: ignore[return-value]


def transform_quad_expressions(
    frame: FrameGeometry,
    *,
    position: ExpressionPoint,
    scale: ExpressionPoint,
    rotation_degrees: str,
    anchor: ExpressionPoint,
) -> ExpressionQuad:
    """Return the project-canvas affine quad as FFmpeg frame expressions."""

    width = str(frame.project_width)
    height = str(frame.project_height)
    return transform_point_expressions(
        frame,
        points=(("0", "0"), (width, "0"), ("0", height), (width, height)),
        position=position,
        scale=scale,
        rotation_degrees=rotation_degrees,
        anchor=anchor,
    )


def correct_quad_for_pixel_centers(
    quad: Quad,
    *,
    width: int,
    height: int,
) -> Quad:
    """Shift a destination quad to Final Cut's pixel-center convention.

    FFmpeg's perspective coordinates describe raster edges, while Final Cut's
    affine kernel maps pixel centers.  For a source basis ``(x_axis, y_axis)``
    the measured shared correction is half the transformed pixel diagonal
    minus half the identity pixel diagonal.  Identity geometry therefore
    remains exactly unchanged.

    Main callers:
    - :meth:`GeometryPlan._perspective` for static geometry.
    - Focused numeric tests that prove the expression implementation matches.
    """

    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise GeometryValidationError("perspective width must be positive")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise GeometryValidationError("perspective height must be positive")
    top_left, top_right, bottom_left, _ = quad
    x_axis = (
        (top_right[0] - top_left[0]) / width,
        (top_right[1] - top_left[1]) / width,
    )
    y_axis = (
        (bottom_left[0] - top_left[0]) / height,
        (bottom_left[1] - top_left[1]) / height,
    )
    correction = (
        0.5 * (x_axis[0] + y_axis[0] - 1.0),
        0.5 * (x_axis[1] + y_axis[1] - 1.0),
    )
    return tuple(
        (point[0] + correction[0], point[1] + correction[1])
        for point in quad
    )  # type: ignore[return-value]


def correct_quad_expressions_for_pixel_centers(
    quad: ExpressionQuad,
    *,
    width: int,
    height: int,
) -> ExpressionQuad:
    """Expression equivalent of :func:`correct_quad_for_pixel_centers`.

    Why this exists:
    Static, animated, expanded-child, and expanded-owner perspectives must all
    use the same destination convention.  Keeping the algebra here prevents
    each execution seam from growing a slightly different scale-only offset.
    """

    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise GeometryValidationError("perspective width must be positive")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise GeometryValidationError("perspective height must be positive")
    top_left, top_right, bottom_left, _ = quad
    x_axis = (
        f"((({top_right[0]})-({top_left[0]}))/{width})",
        f"((({top_right[1]})-({top_left[1]}))/{width})",
    )
    y_axis = (
        f"((({bottom_left[0]})-({top_left[0]}))/{height})",
        f"((({bottom_left[1]})-({top_left[1]}))/{height})",
    )
    correction = (
        f"(0.5*(({x_axis[0]})+({y_axis[0]})-1))",
        f"(0.5*(({x_axis[1]})+({y_axis[1]})-1))",
    )
    return tuple(
        (
            f"(({point[0]})+({correction[0]}))",
            f"(({point[1]})+({correction[1]}))",
        )
        for point in quad
    )  # type: ignore[return-value]


def expand_quad_for_transparent_border(
    quad: Quad,
    *,
    width: int,
    height: int,
    border: int = TRANSPARENT_PERSPECTIVE_BORDER,
) -> Quad:
    """Extend an affine quad to include a transparent source border.

    FFmpeg's ``perspective`` extrapolates the source edge outside the authored
    destination quad. Padding before the filter makes that extrapolated edge
    transparent, but the padded canvas has different corners. This function
    extends the same affine mapping from the original corners to those padded
    corners; cropping ``border`` pixels afterward restores the exact project
    coordinates of every original source pixel.

    Main callers:
    - ``GeometryPlan._perspective`` for static transforms and corner pins.
    """

    if isinstance(border, bool) or not isinstance(border, int) or border <= 0:
        raise GeometryValidationError("transparent perspective border must be positive")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise GeometryValidationError("perspective width must be positive")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise GeometryValidationError("perspective height must be positive")
    top_left, top_right, bottom_left, bottom_right = quad
    x_axis = (
        (top_right[0] - top_left[0]) / width,
        (top_right[1] - top_left[1]) / width,
    )
    y_axis = (
        (bottom_left[0] - top_left[0]) / height,
        (bottom_left[1] - top_left[1]) / height,
    )

    def extend(point: Point, x_sign: int, y_sign: int) -> Point:
        return (
            point[0]
            + border
            + x_sign * border * x_axis[0]
            + y_sign * border * y_axis[0],
            point[1]
            + border
            + x_sign * border * x_axis[1]
            + y_sign * border * y_axis[1],
        )

    return (
        extend(top_left, -1, -1),
        extend(top_right, 1, -1),
        extend(bottom_left, -1, 1),
        extend(bottom_right, 1, 1),
    )


def expand_quad_expressions_for_transparent_border(
    quad: ExpressionQuad,
    *,
    width: int,
    height: int,
    border: int = TRANSPARENT_PERSPECTIVE_BORDER,
) -> ExpressionQuad:
    """Expression equivalent of :func:`expand_quad_for_transparent_border`."""

    if isinstance(border, bool) or not isinstance(border, int) or border <= 0:
        raise GeometryValidationError("transparent perspective border must be positive")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise GeometryValidationError("perspective width must be positive")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise GeometryValidationError("perspective height must be positive")
    top_left, top_right, bottom_left, bottom_right = quad
    x_axis = (
        f"((({top_right[0]})-({top_left[0]}))/{width})",
        f"((({top_right[1]})-({top_left[1]}))/{width})",
    )
    y_axis = (
        f"((({bottom_left[0]})-({top_left[0]}))/{height})",
        f"((({bottom_left[1]})-({top_left[1]}))/{height})",
    )

    def extend(
        point: ExpressionPoint,
        x_sign: int,
        y_sign: int,
    ) -> ExpressionPoint:
        x_operator = "+" if x_sign > 0 else "-"
        y_operator = "+" if y_sign > 0 else "-"
        return (
            f"(({point[0]})+{border}{x_operator}{border}*({x_axis[0]}){y_operator}{border}*({y_axis[0]}))",
            f"(({point[1]})+{border}{x_operator}{border}*({x_axis[1]}){y_operator}{border}*({y_axis[1]}))",
        )

    return (
        extend(top_left, -1, -1),
        extend(top_right, 1, -1),
        extend(bottom_left, -1, 1),
        extend(bottom_right, 1, 1),
    )


@dataclass(frozen=True)
class CornerPinAnimation:
    """Optional typed tracks for the four Final Cut corner offsets."""

    top_left: Optional[TimelineAnimatedVec2] = None
    top_right: Optional[TimelineAnimatedVec2] = None
    bottom_left: Optional[TimelineAnimatedVec2] = None
    bottom_right: Optional[TimelineAnimatedVec2] = None


@dataclass(frozen=True)
class CornerPinAdjustment:
    """Final Cut corner offsets in project-height percentage units.

    These values are offsets from the corresponding unpinned canvas corner,
    not absolute pixel coordinates.  Positive Final Cut Y points upward.
    """

    top_left: Point = (0.0, 0.0)
    top_right: Point = (0.0, 0.0)
    bottom_left: Point = (0.0, 0.0)
    bottom_right: Point = (0.0, 0.0)
    enabled: bool = True
    animation: Optional[CornerPinAnimation] = None

    def __post_init__(self) -> None:
        for name in ("top_left", "top_right", "bottom_left", "bottom_right"):
            object.__setattr__(self, name, _pair(getattr(self, name), name=name))

    @classmethod
    def from_attributes(
        cls,
        attributes: Mapping[str, str],
        *,
        enabled: bool = True,
    ) -> "CornerPinAdjustment":
        """Adapt preserved ``adjust-corners`` attributes without guessing.

        Main callers:
        - The compiler integration when it promotes ``PreservedAdjustment``
          into a typed intrinsic.
        """

        names = {
            "top_left": ("topLeft",),
            "top_right": ("topRight",),
            # FCPXML 1.14 calls the bottom attributes ``botLeft`` and
            # ``botRight``.  Keep the older long spellings as explicit input
            # aliases for isolated pre-v2 fixtures, but always prefer the DTD
            # spelling when both are present.
            "bottom_left": ("botLeft", "bottomLeft"),
            "bottom_right": ("botRight", "bottomRight"),
        }
        values: dict[str, Point] = {}
        for output_name, attribute_names in names.items():
            attribute_name = next(
                (name for name in attribute_names if name in attributes),
                attribute_names[0],
            )
            raw = attributes.get(attribute_name, "0 0").replace(",", " ").split()
            if len(raw) != 2:
                raise GeometryValidationError(
                    f"adjust-corners {attribute_name} must contain exactly two values"
                )
            values[output_name] = (
                _finite(raw[0], name=f"{attribute_name}[0]"),
                _finite(raw[1], name=f"{attribute_name}[1]"),
            )
        return cls(enabled=enabled, **values)


@dataclass(frozen=True)
class FilterStage:
    """One ordered geometry stage and its stock-FFmpeg fragments.

    ``camera_reference`` means the fragments describe the selected rectangle
    for exact snapshot diagnostics; they are not the final pixel operation.
    The FFmpeg builder replaces that stage with a source-coordinate camera
    transform plus its alpha window. ``direct_filters`` means the fragments
    execute as written.
    """

    name: str
    filters: tuple[str, ...]
    owner: Literal["geometry", "external"] = "geometry"
    semantics: Literal["direct_filters", "camera_reference"] = "direct_filters"


@dataclass(frozen=True)
class GeometrySnapshot:
    """Fully resolved geometry for one exact absolute project time."""

    absolute_time: Fraction
    clip_time: Fraction
    source_rect: Optional[SourceRect]
    crop_rect: Optional[PixelRect]
    camera_placement: Optional[CameraPlacement]
    transform: TransformState
    corner_quad: Quad
    transform_quad: Quad
    composed_quad: Quad
    render_surface: RenderSurface
    composed_spatial_filters: tuple[str, ...]
    stages: tuple[FilterStage, ...]

    @property
    def ffmpeg_filters(self) -> tuple[str, ...]:
        return tuple(fragment for stage in self.stages for fragment in stage.filters)

    @property
    def ffmpeg_filtergraph(self) -> str:
        return ",".join(self.ffmpeg_filters)


class GeometryPlan:
    """Validated, exactly timed geometry instructions for one render item.

    ``snapshot`` is the authoritative animated API.  It evaluates the typed
    animation kernel at an exact time and emits static FFmpeg fragments for
    that frame.  ``static_ffmpeg_filters`` is a convenience for whole clips
    only when no animated geometry is present; it fails instead of silently
    flattening keyframes.
    """

    def __init__(
        self,
        *,
        frame: FrameGeometry,
        window: GeometryWindow,
        transform: Optional[TransformAdjustment] = None,
        transform_animation: Optional[RenderTransformAnimation] = None,
        crop: Optional[CropAdjustment] = None,
        conform: str = "fit",
        corners: Optional[CornerPinAdjustment] = None,
        allow_mirrored_scale: bool = False,
    ) -> None:
        self.frame = frame
        self.window = window
        self.transform = transform
        self.transform_animation = transform_animation
        self.crop = crop
        self.conform = self._validate_conform(conform)
        self.corners = corners
        self.allow_mirrored_scale = allow_mirrored_scale
        self._validate_transform()
        self._validate_crop()

    @staticmethod
    def _validate_conform(value: str) -> ConformMode:
        compact = value.strip().lower()
        if compact not in {"fit", "fill", "none"}:
            raise GeometryValidationError(f"unknown conform mode {value!r}")
        return compact  # type: ignore[return-value]

    @property
    def has_animation(self) -> bool:
        transform = self.transform_animation
        corner_animation = self.corners.animation if self.corners else None
        return bool(
            transform
            and any((transform.position, transform.scale, transform.rotation, transform.anchor))
        ) or bool(
            corner_animation
            and any(
                (
                    corner_animation.top_left,
                    corner_animation.top_right,
                    corner_animation.bottom_left,
                    corner_animation.bottom_right,
                )
            )
        ) or bool(self.crop and self.crop.enabled and self.crop.mode.lower() == "pan")

    def _validate_transform(self) -> None:
        transform = self.transform
        if transform is not None:
            _pair(transform.position, name="transform.position")
            _pair(transform.scale, name="transform.scale")
            _finite(transform.rotation, name="transform.rotation")
            _pair(transform.anchor, name="transform.anchor")

        default_scale = transform.scale if transform and transform.enabled else (1.0, 1.0)
        tracks: tuple[tuple[float, ...], tuple[float, ...]]
        animation = self.transform_animation
        if animation and animation.scale is not None:
            points = animation.scale.source_track.control_points
            tracks = (
                tuple(point.value[0] for point in points),
                tuple(point.value[1] for point in points),
            )
        else:
            tracks = ((default_scale[0],), (default_scale[1],))
        for index, values in enumerate(tracks):
            finite_values = tuple(_finite(value, name=f"scale[{index}]") for value in values)
            if any(abs(value) < 1e-12 for value in finite_values):
                raise GeometryValidationError("zero scale would create a degenerate transform")
            signs = {value > 0 for value in finite_values}
            if len(signs) > 1:
                raise GeometryValidationError(
                    f"scale[{index}] changes sign and crosses a degenerate zero scale"
                )
            if not self.allow_mirrored_scale and any(value < 0 for value in finite_values):
                raise GeometryValidationError(
                    "negative scale requires allow_mirrored_scale=True"
                )

    def _validate_crop(self) -> None:
        crop = self.crop
        if crop is None or not crop.enabled:
            return
        mode = crop.mode.strip().lower()
        if mode not in {"crop", "trim", "pan"}:
            raise GeometryValidationError(f"unknown crop mode {crop.mode!r}")
        expected = 2 if mode == "pan" else 1
        active_rects = crop.active_rects
        if len(active_rects) != expected:
            raise GeometryValidationError(
                f"{mode} mode requires exactly {expected} matching rectangle(s); "
                f"received {len(active_rects)}"
            )
        for index, rect in enumerate(active_rects):
            values = (
                _finite(rect.left, name=f"crop.rects[{index}].left"),
                _finite(rect.top, name=f"crop.rects[{index}].top"),
                _finite(rect.right, name=f"crop.rects[{index}].right"),
                _finite(rect.bottom, name=f"crop.rects[{index}].bottom"),
            )
            if mode != "pan" and min(values) < 0:
                raise GeometryValidationError("crop edge percentages cannot be negative")
            horizontal_extent = self.frame.source_height * (values[0] + values[2]) / 100.0
            if horizontal_extent >= self.frame.source_width or values[1] + values[3] >= 100:
                raise GeometryValidationError("crop edges must leave a positive source rectangle")

    def snapshot(self, absolute_time: Fraction) -> GeometrySnapshot:
        """Resolve and compile one exact frame of the geometry plan.

        Main callers:
        - Event-frame A/B diagnostics.
        - The future FFmpeg expression/command builder.

        Why this exists:
        FFmpeg cannot express every retimed smooth curve in one ordinary
        filter expression.  The snapshot remains exact for reverse, freeze,
        and piecewise retimes while a later execution layer chooses between
        expressions, commands, or deterministic segmentation.
        """

        absolute = _exact_time(absolute_time, name="absolute_time")
        clip_time = self.window.clip_time(absolute)
        source_rect = self._source_rect_at(clip_time)
        crop_rect = self._pixel_rect(source_rect) if source_rect is not None else None
        transform = self._transform_at(clip_time)
        corner_quad = self._corner_quad_at(clip_time)
        transform_quad = self._transform_quad(transform)
        composed_quad = compose_spatial_quad(
            self.frame,
            corner_quad=corner_quad,
            transforms=(transform,),
        )
        render_surface = render_surface_for_quads(self.frame, (composed_quad,))
        composed_spatial_filters = (
            ()
            if self._quads_close(composed_quad, self._identity_quad())
            else (self._perspective(composed_quad),)
        )
        camera_placement = None
        if (
            source_rect is not None
            and self.crop is not None
            and self.crop.mode.strip().lower() in {"crop", "pan"}
        ):
            # Crop and Pan share the calibrated native fill-back camera rule
            # and alpha-window ownership. Trim remains a direct extraction
            # without a camera move.
            camera_placement = resolve_crop_camera_placement(
                self.frame,
                source_rect,
                self.conform,
                allow_outside_source=(
                    self.crop.mode.strip().lower() == "pan"
                ),
            )
        stages = self._filter_stages(crop_rect, corner_quad, transform_quad)
        return GeometrySnapshot(
            absolute_time=absolute,
            clip_time=clip_time,
            source_rect=source_rect,
            crop_rect=crop_rect,
            camera_placement=camera_placement,
            transform=transform,
            corner_quad=corner_quad,
            transform_quad=transform_quad,
            composed_quad=composed_quad,
            render_surface=render_surface,
            composed_spatial_filters=composed_spatial_filters,
            stages=stages,
        )

    def static_ffmpeg_filters(self) -> tuple[str, ...]:
        """Return a whole-clip graph only when geometry is genuinely static."""

        if self.has_animation:
            raise UnsupportedGeometryAnimationError(
                "animated geometry must be executed from the typed plan; it cannot be frozen into one filter fragment"
            )
        return self.snapshot(self.window.clip_start).ffmpeg_filters

    def _transform_at(self, clip_time: Fraction) -> TransformState:
        transform = self.transform
        enabled = transform is not None and transform.enabled
        position = transform.position if enabled else (0.0, 0.0)
        scale = transform.scale if enabled else (1.0, 1.0)
        rotation = transform.rotation if enabled else 0.0
        anchor = transform.anchor if enabled else (0.0, 0.0)
        animation = self.transform_animation if enabled else None
        if animation:
            if animation.position:
                position = self._vec_value(animation.position, clip_time)
            if animation.scale:
                scale = self._vec_value(animation.scale, clip_time)
            if animation.rotation:
                rotation = self._scalar_value(animation.rotation, clip_time)
            if animation.anchor:
                anchor = self._vec_value(animation.anchor, clip_time)
        return TransformState(
            position=_pair(position, name="position"),
            scale=_pair(scale, name="scale"),
            rotation_degrees=_finite(rotation, name="rotation"),
            anchor=_pair(anchor, name="anchor"),
        )

    def _vec_value(self, track: TimelineAnimatedVec2, clip_time: Fraction) -> Point:
        return _pair(track.value_at(clip_time), name="animated vec2")

    def _scalar_value(self, track: TimelineAnimatedScalar, clip_time: Fraction) -> float:
        return _finite(track.value_at(clip_time), name="animated scalar")

    def _source_rect_at(self, clip_time: Fraction) -> Optional[SourceRect]:
        crop = self.crop
        if crop is None or not crop.enabled:
            return None
        mode = crop.mode.strip().lower()
        if mode == "pan":
            progress = ken_burns_progress(clip_time, self.window.clip_duration)
            first, last = crop.active_rects
            rect = CropRect(
                left=first.left + (last.left - first.left) * progress,
                top=first.top + (last.top - first.top) * progress,
                right=first.right + (last.right - first.right) * progress,
                bottom=first.bottom + (last.bottom - first.bottom) * progress,
            )
        else:
            rect = crop.active_rects[0]
        unit = self.frame.source_height / 100.0
        left = unit * rect.left
        top = unit * rect.top
        right = unit * rect.right
        bottom = unit * rect.bottom
        return SourceRect(
            x=left,
            y=top,
            width=self.frame.source_width - left - right,
            height=self.frame.source_height - top - bottom,
        )

    def _pixel_rect(self, rect: SourceRect) -> PixelRect:
        """Round one diagnostic/extraction rectangle after float resolution."""

        # Pan may author negative edges.  Keep those signed coordinates in the
        # diagnostic camera-reference rectangle; source alpha supplies the
        # transparent area outside the decoded raster.
        left = _round_half_up_signed(rect.x)
        top = _round_half_up_signed(rect.y)
        right = _round_half_up_signed(self.frame.source_width - rect.x - rect.width)
        bottom = _round_half_up_signed(self.frame.source_height - rect.y - rect.height)
        active_width = self.frame.source_width - left - right
        active_height = self.frame.source_height - top - bottom
        if active_width <= 0 or active_height <= 0:
            raise GeometryValidationError("rounded crop edges leave no source pixels")
        return PixelRect(left, top, active_width, active_height)

    def _corner_quad_at(self, clip_time: Fraction) -> Quad:
        corners = self.corners
        values = {
            "top_left": corners.top_left if corners and corners.enabled else (0.0, 0.0),
            "top_right": corners.top_right if corners and corners.enabled else (0.0, 0.0),
            "bottom_left": corners.bottom_left if corners and corners.enabled else (0.0, 0.0),
            "bottom_right": corners.bottom_right if corners and corners.enabled else (0.0, 0.0),
        }
        animation = corners.animation if corners and corners.enabled else None
        if animation:
            for name in values:
                track = getattr(animation, name)
                if track is not None:
                    values[name] = self._vec_value(track, clip_time)
        unit = self.frame.spatial_unit
        width = float(self.frame.project_width)
        height = float(self.frame.project_height)

        def offset(base: Point, delta: Point) -> Point:
            return (base[0] + delta[0] * unit, base[1] - delta[1] * unit)

        return (
            offset((0.0, 0.0), values["top_left"]),
            offset((width, 0.0), values["top_right"]),
            offset((0.0, height), values["bottom_left"]),
            offset((width, height), values["bottom_right"]),
        )

    def _transform_quad(self, state: TransformState) -> Quad:
        return transform_quad(self.frame, state)

    def _filter_stages(
        self,
        crop_rect: Optional[PixelRect],
        corner_quad: Quad,
        transform_quad: Quad,
    ) -> tuple[FilterStage, ...]:
        crop_filters, working_width, working_height = self._crop_filters(crop_rect)
        conform_filters = self._conform_filters(working_width, working_height)
        # ``reference`` unpacks every decoded frame to ``rgba`` first (the
        # calibrated chain converts to 16-bit right after).  ``fast8`` keeps
        # the decoder's ``yuv420p`` until the single ``scale`` pass, which
        # emits planar ``gbrap`` directly (see ``_scale_first_conform_filters``)
        # -- one swscale pass instead of an unaccelerated full-resolution
        # ``yuv420p -> rgba`` unpack plus an RGBA resample.  A source-resolution
        # crop/pad still needs the alpha format first: ``crop`` aligns
        # subsampled offsets to the chroma grid and ``pad`` needs an alpha plane
        # for its transparent border.
        scale_first = current_render_profile().geometry_strategy == "scale_first"
        if not scale_first:
            decode_filters: tuple[str, ...] = ("format=rgba",)
        elif crop_filters:
            decode_filters = (f"format={current_render_profile().layer_pixel_format}",)
        else:
            decode_filters = ()
        identity = self._identity_quad()
        corner_filters = () if self._quads_close(corner_quad, identity) else (self._perspective(corner_quad),)
        transform_filters = () if self._quads_close(transform_quad, identity) else (self._perspective(transform_quad),)
        crop_semantics: Literal["direct_filters", "camera_reference"] = (
            "camera_reference"
            if self.crop
            and self.crop.enabled
            and self.crop.mode.strip().lower() in {"crop", "pan"}
            else "direct_filters"
        )
        return (
            FilterStage("decode_orientation", decode_filters),
            FilterStage(
                "crop_trim_pan",
                crop_filters,
                semantics=crop_semantics,
            ),
            FilterStage("conform", conform_filters),
            FilterStage("effects_and_masks", (), owner="external"),
            FilterStage("corner_pin", corner_filters),
            FilterStage("anchor_transform", transform_filters),
        )

    def _crop_filters(self, rect: Optional[PixelRect]) -> tuple[tuple[str, ...], int, int]:
        """Describe the selected window while keeping Trim directly executable.

        Crop/Pan fragments expose deterministic snapshot geometry to diagnostics.
        Their ``camera_reference`` stage metadata tells the FFmpeg builder to
        replace extraction with the calibrated camera transform and alpha
        window. Trim remains an ordinary crop-plus-transparent-pad sequence.
        """

        if rect is None:
            return (), self.frame.source_width, self.frame.source_height
        assert self.crop is not None
        mode = self.crop.mode.strip().lower()
        crop = f"crop=w={rect.width}:h={rect.height}:x={rect.x}:y={rect.y}"
        if mode == "trim":
            pad = (
                f"pad=w={self.frame.source_width}:h={self.frame.source_height}:"
                f"x={rect.x}:y={rect.y}:color=black@0"
            )
            return (crop, pad), self.frame.source_width, self.frame.source_height
        return (crop,), rect.width, rect.height

    def _conform_filters(self, width: int, height: int) -> tuple[str, ...]:
        project_width = self.frame.project_width
        project_height = self.frame.project_height
        profile = current_render_profile()
        scale_first = profile.geometry_strategy == "scale_first"
        # ``fast8`` layers must leave the conform stage in the profile's alpha
        # format even when no resample happens; ``format=<same>`` on an
        # already-matching stream is a zero-cost passthrough.
        layer_format = (f"format={profile.layer_pixel_format}",) if scale_first else ()
        if self.conform == "none":
            cropped_width = min(width, project_width)
            cropped_height = min(height, project_height)
            return (
                *layer_format,
                f"crop=w={cropped_width}:h={cropped_height}:x=(iw-ow)/2:y=(ih-oh)/2",
                f"pad=w={project_width}:h={project_height}:x=(ow-iw)/2:y=(oh-ih)/2:color=black@0",
            )
        if width == project_width and height == project_height:
            # Fit and Fill are both exact identity operations when the display
            # raster already equals the project. Avoid a lossy perspective or
            # reconstruction pass; this is also the corpus neutral-control
            # invariant.
            return layer_format
        if self.conform == "fit":
            scale = min(project_width / width, project_height / height)
        else:
            scale = max(project_width / width, project_height / height)

        if scale_first:
            return self._scale_first_conform_filters(width, height, scale)

        # Final Cut maps source pixel centers to fractional destination pixel
        # centers. Integer scale-then-pad/crop rounds that placement and moves
        # high-contrast landmarks by up to one pixel. Use one transparent
        # project-containing surface and one affine perspective sampler so the
        # exact center offset survives.
        #
        # The XYZT Fit and Fill witnesses also prove that Final Cut resamples
        # RGB in the same measured power-linear working space used by Normal
        # source-over. Code-space interpolation missed the frozen luma gate;
        # linearize -> warp -> encode clears both independent witnesses.
        border = TRANSPARENT_PERSPECTIVE_BORDER
        base_width = max(width, project_width) + 2 * border
        base_height = max(height, project_height) + 2 * border
        source_x = (base_width - width) // 2
        source_y = (base_height - height) // 2
        viewport_x = (base_width - project_width) // 2
        viewport_y = (base_height - project_height) // 2
        center_x = (project_width - scale * width) / 2.0 + (scale - 1.0) / 2.0
        center_y = (project_height - scale * height) / 2.0 + (scale - 1.0) / 2.0
        x0 = viewport_x + center_x - scale * source_x
        y0 = viewport_y + center_y - scale * source_y
        x1 = x0 + scale * base_width
        y1 = y0 + scale * base_height
        gamma = FCP_NORMAL_SOURCE_OVER_GAMMA
        inverse_gamma = 1.0 / gamma
        linearize = f"maxval*pow(val/maxval,{_number(gamma)})"
        encode = f"maxval*pow(val/maxval,{_number(inverse_gamma)})"
        perspective = (
            "setparams=range=full,"
            "perspective=sense=destination:eval=init:interpolation=linear:"
            f"x0={_number(x0)}:y0={_number(y0)}:"
            f"x1={_number(x1)}:y1={_number(y0)}:"
            f"x2={_number(x0)}:y2={_number(y1)}:"
            f"x3={_number(x1)}:y3={_number(y1)},"
            f"lutrgb=r='{encode}':g='{encode}':b='{encode}',"
            "setparams=range=full"
        )
        return (
            "format=rgba64le,"
            f"lutrgb=r='{linearize}':g='{linearize}':b='{linearize}'",
            f"pad=w={base_width}:h={base_height}:x={source_x}:y={source_y}:color=black@0",
            perspective,
            f"crop=w={project_width}:h={project_height}:x={viewport_x}:y={viewport_y}",
        )

    def _scale_first_conform_filters(
        self, width: int, height: int, scale: float
    ) -> tuple[str, ...]:
        """Fit/Fill for the ``fast8`` profile: resample once at canvas resolution.

        What it does:
        1. ``scale`` the source straight to the fitted/filled size (bicubic,
           8-bit, code space).  For Fill the scaled frame is at least the
           project; for Fit it is at most the project.
        2. Fill: ``crop`` the centered project window (a zero-copy view).
           Fit: ``pad`` onto a transparent project canvas, centered.

        Why this exists:
        The reference conform runs one linear-light ``perspective`` over a
        transparent surface at *source* resolution (a 1080p source into a
        720p canvas resamples 2.25x more pixels than necessary, in 16-bit,
        with two gamma LUT passes).  Measured on Yunah/720p that geometry was
        ~40% of all CPU work.  ``fast8`` accepts up to half a pixel of
        placement rounding in exchange.
        """

        project_width = self.frame.project_width
        project_height = self.frame.project_height
        scaled_width = max(1, int(round(scale * width)))
        scaled_height = max(1, int(round(scale * height)))
        if self.conform == "fill":
            scaled_width = max(scaled_width, project_width)
            scaled_height = max(scaled_height, project_height)
        else:
            scaled_width = min(scaled_width, project_width)
            scaled_height = min(scaled_height, project_height)
        # ``scale`` resamples and converts in one swscale pass; the trailing
        # ``format`` node pins its output to the profile's alpha format so the
        # decoder's ``yuv420p`` never reaches a compositing node.
        filters = [
            f"scale=w={scaled_width}:h={scaled_height}:flags=bicubic",
            f"format={current_render_profile().layer_pixel_format}",
        ]
        if (scaled_width, scaled_height) == (project_width, project_height):
            return tuple(filters)
        if self.conform == "fill":
            filters.append(
                f"crop=w={project_width}:h={project_height}:x=(iw-ow)/2:y=(ih-oh)/2"
            )
        else:
            filters.append(
                f"pad=w={project_width}:h={project_height}:x=(ow-iw)/2:y=(oh-ih)/2:color=black@0"
            )
        return tuple(filters)

    def _axis_aligned_rect(self, quad: Quad) -> Optional[tuple[float, float, float, float]]:
        """Return ``(x, y, width, height)`` when ``quad`` is an upright rectangle.

        Quad order is top-left, top-right, bottom-left, bottom-right in project
        pixels.  Mirrored (negative scale) and rotated quads return ``None``.
        """

        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
        tolerance = 1e-6
        if not (
            math.isclose(y0, y1, abs_tol=tolerance)
            and math.isclose(y2, y3, abs_tol=tolerance)
            and math.isclose(x0, x2, abs_tol=tolerance)
            and math.isclose(x1, x3, abs_tol=tolerance)
        ):
            return None
        rect_width = x1 - x0
        rect_height = y2 - y0
        if rect_width <= 0 or rect_height <= 0:
            return None
        return (x0, y0, rect_width, rect_height)

    def _scale_first_rect_filters(
        self, rect: tuple[float, float, float, float]
    ) -> Optional[str]:
        """Zoom/translate an upright rectangle with ``scale`` + ``crop`` + ``pad``.

        Returns ``None`` when the rectangle does not intersect the canvas (the
        caller then keeps the general perspective path, which handles empty
        coverage exactly).  Rounds the rectangle to whole pixels.
        """

        project_width = self.frame.project_width
        project_height = self.frame.project_height
        rect_x, rect_y, rect_width, rect_height = rect
        left = int(round(rect_x))
        top = int(round(rect_y))
        width = max(1, int(round(rect_width)))
        height = max(1, int(round(rect_height)))
        visible_left = max(0, left)
        visible_top = max(0, top)
        visible_right = min(project_width, left + width)
        visible_bottom = min(project_height, top + height)
        if visible_right <= visible_left or visible_bottom <= visible_top:
            return None
        filters = [f"scale=w={width}:h={height}:flags=bicubic"]
        visible_width = visible_right - visible_left
        visible_height = visible_bottom - visible_top
        if (visible_width, visible_height) != (width, height):
            filters.append(
                f"crop=w={visible_width}:h={visible_height}:"
                f"x={visible_left - left}:y={visible_top - top}"
            )
        if (visible_width, visible_height, visible_left, visible_top) != (
            project_width,
            project_height,
            0,
            0,
        ):
            filters.append(
                f"pad=w={project_width}:h={project_height}:"
                f"x={visible_left}:y={visible_top}:color=black@0"
            )
        return ",".join(filters)

    def _identity_quad(self) -> Quad:
        width = float(self.frame.project_width)
        height = float(self.frame.project_height)
        return ((0.0, 0.0), (width, 0.0), (0.0, height), (width, height))

    @staticmethod
    def _quads_close(left: Quad, right: Quad) -> bool:
        return all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
            for left_point, right_point in zip(left, right)
            for a, b in zip(left_point, right_point)
        )

    def _perspective(self, quad: Quad) -> str:
        """Map the project-sized frame onto ``quad`` (project pixels).

        ``reference``: linear-light 16-bit affine ``perspective`` over a
        transparent border, exactly as calibrated.  ``fast8``: an upright
        rectangle becomes ``scale``/``crop``/``pad`` in 8-bit code space; any
        other quad keeps the same perspective sampler but in 8-bit code space
        without the gamma LUT round trip.
        """

        width = self.frame.project_width
        height = self.frame.project_height
        border = TRANSPARENT_PERSPECTIVE_BORDER
        scale_first = current_render_profile().geometry_strategy == "scale_first"
        if scale_first:
            rect = self._axis_aligned_rect(quad)
            if rect is not None:
                rect_filters = self._scale_first_rect_filters(rect)
                if rect_filters is not None:
                    return rect_filters
        corrected = correct_quad_for_pixel_centers(
            quad,
            width=width,
            height=height,
        )
        expanded = expand_quad_for_transparent_border(
            corrected,
            width=width,
            height=height,
            border=border,
        )
        options = []
        for index, (x, y) in enumerate(expanded):
            options.extend((f"x{index}={_number(x)}", f"y{index}={_number(y)}"))
        perspective = (
            f"pad=w=iw+{2 * border}:h=ih+{2 * border}:x={border}:y={border}:"
            "color=black@0,"
            "setparams=range=full,"
            "perspective=sense=destination:eval=init:interpolation=linear:"
            + ":".join(options)
        )
        if scale_first:
            return perspective + f",crop=w={width}:h={height}:x={border}:y={border}"
        gamma = FCP_NORMAL_SOURCE_OVER_GAMMA
        inverse_gamma = 1.0 / gamma
        linearize = f"maxval*pow(val/maxval,{_number(gamma)})"
        encode = f"maxval*pow(val/maxval,{_number(inverse_gamma)})"
        return (
            "format=rgba64le,"
            f"lutrgb=r='{linearize}':g='{linearize}':b='{linearize}',"
            + perspective
            + f",lutrgb=r='{encode}':g='{encode}':b='{encode}',"
            f"setparams=range=full,crop=w={width}:h={height}:x={border}:y={border}"
        )


__all__ = [
    "CameraPlacement",
    "CanvasBounds",
    "CornerPinAdjustment",
    "CornerPinAnimation",
    "FilterStage",
    "FrameGeometry",
    "GeometryError",
    "GeometryPlan",
    "GeometrySnapshot",
    "GeometryValidationError",
    "GeometryWindow",
    "PixelRect",
    "RenderSurface",
    "SourceRect",
    "TRANSPARENT_PERSPECTIVE_BORDER",
    "TransformState",
    "UnsupportedGeometryAnimationError",
    "correct_quad_expressions_for_pixel_centers",
    "correct_quad_for_pixel_centers",
    "expand_quad_expressions_for_transparent_border",
    "expand_quad_for_transparent_border",
    "compose_spatial_quad",
    "render_surface_for_quads",
    "pan_base_conform_multiplier",
    "resolve_crop_camera_placement",
    "resolve_camera_placement",
    "transform_points",
    "transform_point_expressions",
    "transform_quad",
    "transform_quad_expressions",
]
