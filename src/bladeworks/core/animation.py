"""Typed, deterministic animation tracks for the experimental renderer IR.

Architecture map
================

``model.Keyframe`` / ``model.Parameter``
    Parser-owned strings and exact keyframe times enter through explicit
    ``from_keyframes`` and ``from_parameter`` adapters.

``ScalarControlPoint`` / ``Vec2ControlPoint``
    Validate the value shape, interpolation names, exact ``Fraction`` time,
    and preserved ``auxValue`` tangent metadata.

``AnimatedScalar`` / ``AnimatedVec2``
    Evaluate source-time animation.  Values hold outside the keyframe range;
    the interval ending at a keyframe uses that destination keyframe's
    ``interp`` and ``curve`` settings.

``TimelineAnimatedScalar`` / ``TimelineAnimatedVec2``
    Evaluate a source-time track on the output timeline by first applying a
    ``RetimeMap``.  This keeps reverse, freeze, variable-rate, and repeated
    source occurrences exact without rewriting their easing curves.

``ken_burns_progress`` / ``ken_burns_progress_expression``
    Share the measured Final Cut Pan-camera progress curve between numeric
    diagnostics and FFmpeg lowering.  All four camera edges must consume this
    one value; independent edge clocks would warp the camera rectangle.

Important invariants
--------------------

* Control-point times are exact ``fractions.Fraction`` values.  Floats are
  rejected at the timing boundary.
* Control points are strictly ordered and duplicate times are invalid.
* ``smooth`` value curves use monotone cubic interpolation independently per
  component, so a segment cannot overshoot either endpoint.
* ``auxValue`` is preserved and reported as uncalibrated.  It never silently
  changes the evaluated curve.
* Lossless retimed evaluation is the primary API.  Materializing ordinary
  timeline points is intentionally limited to one forward linear retime
  segment; reverse, freeze, and multi-segment maps fail explicitly.

Why this exists
---------------

Final Cut stores genuine animation inside ``keyframeAnimation`` and may place
those source-local keyframes under retimed clips.  The old renderer reduced
animation to ad-hoc FFmpeg expressions and could silently treat a reverse or
repeated source range as forward playback.  This module freezes a typed,
testable animation contract before the shared compiler and FFmpeg builder are
changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable, Literal, Sequence, TypeAlias

from .model import Keyframe, Parameter
from .retime import RetimeMap, RetimeSegment


Interpolation: TypeAlias = Literal["linear", "ease", "ease-in", "ease-out"]
ValueCurve: TypeAlias = Literal["linear", "smooth"]
Vec2: TypeAlias = tuple[float, float]
AnimationValue: TypeAlias = float | Vec2


# Final Cut Pan progress measured from the BT.709 1080p Stage 5 landmark oracle.
# Each point is a normalized output time and the corresponding normalized
# camera-window progress recovered from robust source-to-output affine scale.
# The same monotone curve drives all four rect edges.
# Piecewise interpolation avoids fitting an attractive but measurably wrong
# stock easing curve; private reference movies remain in the evidence tree.
KEN_BURNS_PROGRESS_CALIBRATION_ID = "xyzt-20260812-fcp-pan-trajectory-1080p-v3"
KEN_BURNS_PROGRESS_MAX_RESIDUAL = 0.0015
KEN_BURNS_PROGRESS_POINTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.050847, 0.007683),
    (0.101695, 0.027583),
    (0.152542, 0.061088),
    (0.203390, 0.107125),
    (0.254237, 0.167228),
    (0.305085, 0.233681),
    (0.355932, 0.300808),
    (0.406780, 0.367519),
    (0.457627, 0.435341),
    (0.508475, 0.501657),
    (0.559322, 0.567544),
    (0.610169, 0.634175),
    (0.661017, 0.700629),
    (0.711864, 0.767585),
    (0.762712, 0.834247),
    (0.813559, 0.893769),
    (0.864407, 0.940151),
    (0.915254, 0.974251),
    (0.966102, 0.993884),
    (1.0, 1.0),
)


class AnimationError(ValueError):
    """Base error for invalid or unsupported animation contracts."""


class AnimationValidationError(AnimationError):
    """A source track is malformed and cannot be evaluated deterministically."""


class UnsupportedAnimationMappingError(AnimationError):
    """A requested lossy timeline-point conversion would be ambiguous."""


def _exact_time(value: object, *, field_name: str) -> Fraction:
    """Accept only exact rational animation times.

    Main callers:
    - Control-point construction.
    - Public ``value_at`` methods.

    Why this exists:
    - ``Fraction(0.1)`` preserves a binary floating-point approximation, not
      the exact FCPXML rational time.  Timing floats therefore fail here.
    """

    if isinstance(value, bool):
        raise AnimationValidationError(
            f"{field_name} must be an exact Fraction, not bool"
        )
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnimationValidationError(f"{field_name} must be finite")
        raise AnimationValidationError(
            f"{field_name} must be an exact Fraction, not float"
        )
    raise AnimationValidationError(
        f"{field_name} must be an exact Fraction, got {type(value).__name__}"
    )


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise AnimationValidationError(
            f"{field_name} must be a finite number, not bool"
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnimationValidationError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(result):
        raise AnimationValidationError(f"{field_name} must be finite")
    return result


def _normalize_interpolation(raw: str | None) -> Interpolation:
    compact = (raw or "linear").strip().lower().replace("-", "").replace("_", "")
    values: dict[str, Interpolation] = {
        "linear": "linear",
        "ease": "ease",
        "easein": "ease-in",
        "easeout": "ease-out",
    }
    try:
        return values[compact]
    except KeyError as exc:
        raise AnimationValidationError(
            f"unsupported keyframe interpolation {raw!r}"
        ) from exc


def _normalize_curve(raw: str | None) -> ValueCurve:
    # FCPXML 1.14 declares ``smooth`` as the DTD default.
    compact = (raw or "smooth").strip().lower()
    if compact == "linear":
        return "linear"
    if compact == "smooth":
        return "smooth"
    raise AnimationValidationError(f"unsupported keyframe curve {raw!r}")


def _parse_components(raw: str, *, count: int, field_name: str) -> tuple[float, ...]:
    parts = raw.replace(",", " ").split()
    if len(parts) != count:
        raise AnimationValidationError(
            f"{field_name} requires exactly {count} numeric component(s), got {raw!r}"
        )
    return tuple(
        _finite_number(part, field_name=f"{field_name} component {index}")
        for index, part in enumerate(parts)
    )


@dataclass(frozen=True)
class ScalarControlPoint:
    """One validated scalar keyframe in source-local time."""

    time: Fraction
    value: float
    interpolation: Interpolation = "linear"
    curve: ValueCurve = "smooth"
    aux_value: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "time", _exact_time(self.time, field_name="control-point time")
        )
        object.__setattr__(
            self, "value", _finite_number(self.value, field_name="scalar value")
        )
        object.__setattr__(
            self, "interpolation", _normalize_interpolation(self.interpolation)
        )
        object.__setattr__(self, "curve", _normalize_curve(self.curve))
        if self.aux_value is not None and not isinstance(self.aux_value, str):
            raise AnimationValidationError("auxValue must be a string or None")

    @classmethod
    def from_keyframe(cls, keyframe: Keyframe) -> "ScalarControlPoint":
        """Parse one parser-owned keyframe as a typed scalar point."""

        if not isinstance(keyframe, Keyframe):
            raise AnimationValidationError(
                "scalar control point requires model.Keyframe"
            )
        value = _parse_components(
            keyframe.value, count=1, field_name="scalar keyframe value"
        )[0]
        return cls(
            time=keyframe.time,
            value=value,
            interpolation=_normalize_interpolation(keyframe.interp),
            curve=_normalize_curve(keyframe.curve),
            aux_value=keyframe.aux_value,
        )


@dataclass(frozen=True)
class Vec2ControlPoint:
    """One validated two-component keyframe in source-local time."""

    time: Fraction
    value: Vec2
    interpolation: Interpolation = "linear"
    curve: ValueCurve = "smooth"
    aux_value: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "time", _exact_time(self.time, field_name="control-point time")
        )
        if not isinstance(self.value, tuple) or len(self.value) != 2:
            raise AnimationValidationError(
                "vec2 value must contain exactly two components"
            )
        object.__setattr__(
            self,
            "value",
            (
                _finite_number(self.value[0], field_name="vec2 x"),
                _finite_number(self.value[1], field_name="vec2 y"),
            ),
        )
        object.__setattr__(
            self, "interpolation", _normalize_interpolation(self.interpolation)
        )
        object.__setattr__(self, "curve", _normalize_curve(self.curve))
        if self.aux_value is not None and not isinstance(self.aux_value, str):
            raise AnimationValidationError("auxValue must be a string or None")

    @classmethod
    def from_keyframe(cls, keyframe: Keyframe) -> "Vec2ControlPoint":
        """Parse one parser-owned keyframe as a typed two-component point."""

        if not isinstance(keyframe, Keyframe):
            raise AnimationValidationError("vec2 control point requires model.Keyframe")
        values = _parse_components(
            keyframe.value, count=2, field_name="vec2 keyframe value"
        )
        return cls(
            time=keyframe.time,
            value=(values[0], values[1]),
            interpolation=_normalize_interpolation(keyframe.interp),
            curve=_normalize_curve(keyframe.curve),
            aux_value=keyframe.aux_value,
        )


@dataclass(frozen=True)
class AnimationNotice:
    """A preserved animation fact that is not yet calibrated for execution."""

    code: Literal["uncalibrated_aux_value"]
    control_point_index: int
    time: Fraction
    detail: str


@dataclass(frozen=True)
class MappedControlPoint:
    """One source keyframe occurrence on the output timeline.

    A moving occurrence has equal start/end times.  A freeze occurrence spans
    an interval and deliberately cannot be flattened to an arbitrary point.
    """

    source_index: int
    source_time: Fraction
    value: AnimationValue
    interpolation: Interpolation
    curve: ValueCurve
    aux_value: str | None
    timeline_start: Fraction
    timeline_end: Fraction
    includes_start: bool
    includes_end: bool
    segment_indices: tuple[int, ...]

    @property
    def is_interval(self) -> bool:
        return self.timeline_end > self.timeline_start


def _validate_points(
    points: Sequence[ScalarControlPoint] | Sequence[Vec2ControlPoint],
    *,
    expected_type: type[ScalarControlPoint] | type[Vec2ControlPoint],
) -> None:
    if not points:
        raise AnimationValidationError(
            "animation track requires at least one control point"
        )
    previous: Fraction | None = None
    for index, point in enumerate(points):
        if not isinstance(point, expected_type):
            raise AnimationValidationError(
                f"control_points[{index}] must be {expected_type.__name__}"
            )
        if previous is not None and point.time == previous:
            raise AnimationValidationError(f"duplicate control-point time {point.time}")
        if previous is not None and point.time < previous:
            raise AnimationValidationError(
                "control-point times must be strictly increasing"
            )
        previous = point.time


def _notices(
    points: Sequence[ScalarControlPoint] | Sequence[Vec2ControlPoint],
) -> tuple[AnimationNotice, ...]:
    return tuple(
        AnimationNotice(
            code="uncalibrated_aux_value",
            control_point_index=index,
            time=point.time,
            detail=f"auxValue {point.aux_value!r} is preserved but not applied",
        )
        for index, point in enumerate(points)
        if point.aux_value is not None
    )


def _eased_fraction(value: float, interpolation: Interpolation) -> float:
    if interpolation == "linear":
        return value
    if interpolation == "ease":
        return value * value * (3.0 - 2.0 * value)
    if interpolation == "ease-in":
        return value * value * value
    if interpolation == "ease-out":
        inverse = 1.0 - value
        return 1.0 - inverse * inverse * inverse
    raise AssertionError(f"unreachable interpolation {interpolation!r}")


def ken_burns_progress(
    elapsed: Fraction,
    duration: Fraction,
) -> float:
    """Return Final Cut's calibrated, endpoint-held Pan camera progress.

    Main callers:
    - Geometry snapshots for Crop mode ``pan``.
    - Ken Burns oracle diagnostics.

    Why this exists:
    - The previous geometry snapshot used linear time while the FFmpeg path
      constructed a separate inline counter.  The shared piecewise-linear
      calibration below follows the measured Final Cut marker trajectory.
      Exact rational time is retained until the final normalized scalar is
      evaluated.
    """

    exact_elapsed = _exact_time(elapsed, field_name="Ken Burns elapsed time")
    exact_duration = _exact_time(duration, field_name="Ken Burns duration")
    if exact_duration <= 0:
        raise AnimationValidationError("Ken Burns duration must be positive")
    normalized = min(max(exact_elapsed / exact_duration, Fraction(0)), Fraction(1))
    value = float(normalized)
    for (left_time, left_value), (right_time, right_value) in zip(
        KEN_BURNS_PROGRESS_POINTS[:-1],
        KEN_BURNS_PROGRESS_POINTS[1:],
        strict=True,
    ):
        if value <= right_time:
            fraction = (value - left_time) / (right_time - left_time)
            return left_value + (right_value - left_value) * fraction
    return 1.0


def ken_burns_progress_expression(
    *,
    counter: str,
    frame_count: int,
    first_frame_index: int = 0,
) -> str:
    """Return the FFmpeg expression matching :func:`ken_burns_progress`.

    ``zoompan`` exposes the zero-based renderer-owned ``on`` counter.  The
    final visible frame is therefore ``frame_count - 1`` and must evaluate to
    exactly one.  XML text is never accepted as the counter expression.
    """

    if counter not in {"on", "N"}:
        raise AnimationValidationError(
            "Ken Burns FFmpeg counter must be the renderer-owned 'on' or 'N' variable"
        )
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise AnimationValidationError("Ken Burns frame_count must be an integer")
    if frame_count < 2:
        raise AnimationValidationError("Ken Burns frame_count must be at least two")
    if first_frame_index not in {0, 1}:
        raise AnimationValidationError(
            "Ken Burns first_frame_index must be zero or one"
        )
    elapsed = counter if first_frame_index == 0 else f"({counter}-1)"
    normalized = f"min(max({elapsed}/{frame_count - 1},0),1)"

    def number(value: float) -> str:
        return format(value, ".12g")

    expression = "1"
    for (left_time, left_value), (right_time, right_value) in reversed(
        tuple(
            zip(
                KEN_BURNS_PROGRESS_POINTS[:-1],
                KEN_BURNS_PROGRESS_POINTS[1:],
                strict=True,
            )
        )
    ):
        segment = (
            f"({number(left_value)}+({number(right_value-left_value)})*"
            f"((({normalized})-{number(left_time)})/"
            f"{number(right_time-left_time)}))"
        )
        expression = (
            f"if(lte(({normalized}),{number(right_time)}),"
            f"{segment},{expression})"
        )
    return expression


def _monotone_slopes(
    times: Sequence[Fraction], values: Sequence[float]
) -> tuple[float, ...]:
    """Return PCHIP/Fritsch-Carlson slopes without endpoint overshoot."""

    count = len(times)
    if count == 1:
        return (0.0,)
    widths = [float(times[index + 1] - times[index]) for index in range(count - 1)]
    secants = [
        (values[index + 1] - values[index]) / widths[index]
        for index in range(count - 1)
    ]
    if count == 2:
        return (secants[0], secants[0])

    slopes = [0.0] * count
    for index in range(1, count - 1):
        left = secants[index - 1]
        right = secants[index]
        if left == 0.0 or right == 0.0 or left * right <= 0.0:
            slopes[index] = 0.0
            continue
        left_weight = 2.0 * widths[index] + widths[index - 1]
        right_weight = widths[index] + 2.0 * widths[index - 1]
        slopes[index] = (left_weight + right_weight) / (
            left_weight / left + right_weight / right
        )

    slopes[0] = _endpoint_slope(widths[0], widths[1], secants[0], secants[1])
    slopes[-1] = _endpoint_slope(widths[-1], widths[-2], secants[-1], secants[-2])
    return tuple(slopes)


def _endpoint_slope(
    adjacent_width: float,
    next_width: float,
    adjacent_secant: float,
    next_secant: float,
) -> float:
    slope = (
        (2.0 * adjacent_width + next_width) * adjacent_secant
        - adjacent_width * next_secant
    ) / (adjacent_width + next_width)
    if slope * adjacent_secant <= 0.0:
        return 0.0
    if adjacent_secant * next_secant < 0.0 and abs(slope) > abs(3.0 * adjacent_secant):
        return 3.0 * adjacent_secant
    return slope


def _interpolate_component(
    *,
    index: int,
    fraction: float,
    times: Sequence[Fraction],
    values: Sequence[float],
    curve: ValueCurve,
    slopes: Sequence[float],
) -> float:
    left = values[index]
    right = values[index + 1]
    if curve == "linear":
        return left + (right - left) * fraction

    width = float(times[index + 1] - times[index])
    squared = fraction * fraction
    cubed = squared * fraction
    left_basis = 2.0 * cubed - 3.0 * squared + 1.0
    left_slope_basis = cubed - 2.0 * squared + fraction
    right_basis = -2.0 * cubed + 3.0 * squared
    right_slope_basis = cubed - squared
    result = (
        left_basis * left
        + left_slope_basis * width * slopes[index]
        + right_basis * right
        + right_slope_basis * width * slopes[index + 1]
    )
    # Floating-point roundoff must not violate the monotonicity guarantee.
    return min(max(result, min(left, right)), max(left, right))


def _segment_index(
    points: Sequence[ScalarControlPoint] | Sequence[Vec2ControlPoint],
    time: Fraction,
) -> int:
    for index in range(len(points) - 1):
        if points[index].time <= time <= points[index + 1].time:
            return index
    raise AssertionError("time inside track range did not resolve to a segment")


@dataclass(frozen=True)
class AnimatedScalar:
    """A validated scalar animation evaluated in source-local time.

    Main callers:
    - Geometry, opacity, effect, and audio compilers adapt parser parameters
      through ``from_parameter`` or ``from_keyframes``.
    - ``TimelineAnimatedScalar`` evaluates this track through a ``RetimeMap``.
    """

    control_points: tuple[ScalarControlPoint, ...]

    def __post_init__(self) -> None:
        points = tuple(self.control_points)
        object.__setattr__(self, "control_points", points)
        _validate_points(points, expected_type=ScalarControlPoint)

    @classmethod
    def from_keyframes(cls, keyframes: Iterable[Keyframe]) -> "AnimatedScalar":
        return cls(tuple(ScalarControlPoint.from_keyframe(item) for item in keyframes))

    @classmethod
    def from_parameter(cls, parameter: Parameter) -> "AnimatedScalar":
        if not isinstance(parameter, Parameter):
            raise AnimationValidationError("scalar animation requires model.Parameter")
        if not parameter.keyframes:
            raise AnimationValidationError(
                "parameter has no keyframeAnimation control points"
            )
        return cls.from_keyframes(parameter.keyframes)

    @property
    def notices(self) -> tuple[AnimationNotice, ...]:
        return _notices(self.control_points)

    def value_at(self, source_time: Fraction) -> float:
        exact_time = _exact_time(source_time, field_name="source animation time")
        points = self.control_points
        if exact_time <= points[0].time:
            return points[0].value
        if exact_time >= points[-1].time:
            return points[-1].value

        index = _segment_index(points, exact_time)
        left = points[index]
        right = points[index + 1]
        raw_fraction = float((exact_time - left.time) / (right.time - left.time))
        fraction = _eased_fraction(raw_fraction, right.interpolation)
        times = tuple(point.time for point in points)
        values = tuple(point.value for point in points)
        slopes = _monotone_slopes(times, values)
        return _interpolate_component(
            index=index,
            fraction=fraction,
            times=times,
            values=values,
            curve=right.curve,
            slopes=slopes,
        )


@dataclass(frozen=True)
class AnimatedVec2:
    """A validated two-component animation evaluated in source-local time."""

    control_points: tuple[Vec2ControlPoint, ...]

    def __post_init__(self) -> None:
        points = tuple(self.control_points)
        object.__setattr__(self, "control_points", points)
        _validate_points(points, expected_type=Vec2ControlPoint)

    @classmethod
    def from_keyframes(cls, keyframes: Iterable[Keyframe]) -> "AnimatedVec2":
        return cls(tuple(Vec2ControlPoint.from_keyframe(item) for item in keyframes))

    @classmethod
    def from_parameter(cls, parameter: Parameter) -> "AnimatedVec2":
        if not isinstance(parameter, Parameter):
            raise AnimationValidationError("vec2 animation requires model.Parameter")
        if not parameter.keyframes:
            raise AnimationValidationError(
                "parameter has no keyframeAnimation control points"
            )
        return cls.from_keyframes(parameter.keyframes)

    @property
    def notices(self) -> tuple[AnimationNotice, ...]:
        return _notices(self.control_points)

    def value_at(self, source_time: Fraction) -> Vec2:
        exact_time = _exact_time(source_time, field_name="source animation time")
        points = self.control_points
        if exact_time <= points[0].time:
            return points[0].value
        if exact_time >= points[-1].time:
            return points[-1].value

        index = _segment_index(points, exact_time)
        left = points[index]
        right = points[index + 1]
        raw_fraction = float((exact_time - left.time) / (right.time - left.time))
        fraction = _eased_fraction(raw_fraction, right.interpolation)
        times = tuple(point.time for point in points)
        components = tuple(zip(*(point.value for point in points)))
        result: list[float] = []
        for values in components:
            slopes = _monotone_slopes(times, values)
            result.append(
                _interpolate_component(
                    index=index,
                    fraction=fraction,
                    times=times,
                    values=values,
                    curve=right.curve,
                    slopes=slopes,
                )
            )
        return result[0], result[1]


def _mapped_control_points(
    control_points: Sequence[ScalarControlPoint] | Sequence[Vec2ControlPoint],
    retime_map: RetimeMap,
) -> tuple[MappedControlPoint, ...]:
    mapped: list[MappedControlPoint] = []
    for source_index, point in enumerate(control_points):
        for occurrence in retime_map.source_to_timeline_occurrences(point.time):
            mapped.append(
                MappedControlPoint(
                    source_index=source_index,
                    source_time=point.time,
                    value=point.value,
                    interpolation=point.interpolation,
                    curve=point.curve,
                    aux_value=point.aux_value,
                    timeline_start=occurrence.timeline_start,
                    timeline_end=occurrence.timeline_end,
                    includes_start=occurrence.includes_start,
                    includes_end=occurrence.includes_end,
                    segment_indices=occurrence.segment_indices,
                )
            )
    return tuple(
        sorted(
            mapped,
            key=lambda item: (
                item.timeline_start,
                item.timeline_end,
                item.source_index,
            ),
        )
    )


def _validate_timeline_wrapper(
    source_track: object, retime_map: object, expected_type: type[object]
) -> None:
    if not isinstance(source_track, expected_type):
        raise AnimationValidationError(f"source_track must be {expected_type.__name__}")
    if not isinstance(retime_map, RetimeMap):
        raise AnimationValidationError("retime_map must be RetimeMap")


def _materializable_segment(retime_map: RetimeMap) -> RetimeSegment:
    if len(retime_map.segments) != 1:
        raise UnsupportedAnimationMappingError(
            "materializing timeline control points requires one continuous retime segment"
        )
    segment = retime_map.segments[0]
    if segment.kind == "freeze":
        raise UnsupportedAnimationMappingError(
            "a freeze maps source keyframes to intervals, not unique timeline points"
        )
    if segment.kind == "reverse":
        raise UnsupportedAnimationMappingError(
            "reverse easing cannot be flattened without changing source-track semantics"
        )
    return segment


@dataclass(frozen=True)
class TimelineAnimatedScalar:
    """Losslessly evaluate a scalar source track through an exact retime map."""

    source_track: AnimatedScalar
    retime_map: RetimeMap

    def __post_init__(self) -> None:
        _validate_timeline_wrapper(self.source_track, self.retime_map, AnimatedScalar)

    @property
    def control_point_occurrences(self) -> tuple[MappedControlPoint, ...]:
        return _mapped_control_points(self.source_track.control_points, self.retime_map)

    @property
    def notices(self) -> tuple[AnimationNotice, ...]:
        return self.source_track.notices

    def value_at(self, timeline_time: Fraction) -> float:
        source_time = self.retime_map.timeline_to_source(timeline_time)
        return self.source_track.value_at(source_time)

    def materialize_timeline_track(self) -> AnimatedScalar:
        """Flatten the only unambiguous subset into ordinary timeline points.

        Main callers:
        - FFmpeg expression builders that cannot evaluate ``RetimeMap`` first.

        Why this exists:
        - Forward one-segment retimes preserve point ordering and easing.  More
          complex maps remain available through ``value_at`` but must not be
          rewritten into a misleading flat keyframe sequence.
        """

        segment = _materializable_segment(self.retime_map)
        points: list[ScalarControlPoint] = []
        for point in self.source_track.control_points:
            occurrences = self.retime_map.source_to_timeline_occurrences(point.time)
            if not occurrences:
                raise UnsupportedAnimationMappingError(
                    "materializing a partial source-track range requires synthesized boundary points"
                )
            if len(occurrences) != 1 or occurrences[0].is_interval:
                raise UnsupportedAnimationMappingError(
                    "source control point does not map to one timeline coordinate"
                )
            points.append(
                ScalarControlPoint(
                    time=occurrences[0].timeline_time,
                    value=point.value,
                    interpolation=point.interpolation,
                    curve=point.curve,
                    aux_value=point.aux_value,
                )
            )
        if not points:
            raise UnsupportedAnimationMappingError(
                "retime segment does not contain any source control points"
            )
        # Retain the local name to make the forward-only precondition obvious.
        assert segment.kind == "forward"
        return AnimatedScalar(tuple(points))


@dataclass(frozen=True)
class TimelineAnimatedVec2:
    """Losslessly evaluate a two-component source track through a retime map."""

    source_track: AnimatedVec2
    retime_map: RetimeMap

    def __post_init__(self) -> None:
        _validate_timeline_wrapper(self.source_track, self.retime_map, AnimatedVec2)

    @property
    def control_point_occurrences(self) -> tuple[MappedControlPoint, ...]:
        return _mapped_control_points(self.source_track.control_points, self.retime_map)

    @property
    def notices(self) -> tuple[AnimationNotice, ...]:
        return self.source_track.notices

    def value_at(self, timeline_time: Fraction) -> Vec2:
        source_time = self.retime_map.timeline_to_source(timeline_time)
        return self.source_track.value_at(source_time)

    def materialize_timeline_track(self) -> AnimatedVec2:
        _materializable_segment(self.retime_map)
        points: list[Vec2ControlPoint] = []
        for point in self.source_track.control_points:
            occurrences = self.retime_map.source_to_timeline_occurrences(point.time)
            if not occurrences:
                raise UnsupportedAnimationMappingError(
                    "materializing a partial source-track range requires synthesized boundary points"
                )
            if len(occurrences) != 1 or occurrences[0].is_interval:
                raise UnsupportedAnimationMappingError(
                    "source control point does not map to one timeline coordinate"
                )
            points.append(
                Vec2ControlPoint(
                    time=occurrences[0].timeline_time,
                    value=point.value,
                    interpolation=point.interpolation,
                    curve=point.curve,
                    aux_value=point.aux_value,
                )
            )
        if not points:
            raise UnsupportedAnimationMappingError(
                "retime segment does not contain any source control points"
            )
        return AnimatedVec2(tuple(points))


def map_scalar_animation(
    source_track: AnimatedScalar, retime_map: RetimeMap
) -> TimelineAnimatedScalar:
    """Create the lossless source-to-timeline scalar evaluation wrapper."""

    return TimelineAnimatedScalar(source_track=source_track, retime_map=retime_map)


def map_vec2_animation(
    source_track: AnimatedVec2, retime_map: RetimeMap
) -> TimelineAnimatedVec2:
    """Create the lossless source-to-timeline vector evaluation wrapper."""

    return TimelineAnimatedVec2(source_track=source_track, retime_map=retime_map)


__all__ = [
    "AnimatedScalar",
    "AnimatedVec2",
    "AnimationError",
    "AnimationNotice",
    "AnimationValidationError",
    "KEN_BURNS_PROGRESS_CALIBRATION_ID",
    "KEN_BURNS_PROGRESS_MAX_RESIDUAL",
    "KEN_BURNS_PROGRESS_POINTS",
    "MappedControlPoint",
    "ScalarControlPoint",
    "TimelineAnimatedScalar",
    "TimelineAnimatedVec2",
    "UnsupportedAnimationMappingError",
    "Vec2",
    "Vec2ControlPoint",
    "ken_burns_progress",
    "ken_burns_progress_expression",
    "map_scalar_animation",
    "map_vec2_animation",
]
