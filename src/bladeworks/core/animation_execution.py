"""Compile typed FCPXML animation into bounded FFmpeg execution plans.

Architecture map
================

``TimelineAnimatedScalar`` / ``TimelineAnimatedVec2``
    Exact source control points plus an exact piecewise ``RetimeMap``.

``AnimationClock``
    Defines how an FFmpeg filter's time variable relates to the clip.  A
    clip-local clock starts at the beginning of a rendered pre-roll window;
    an absolute clock uses sequence time directly.

``ClipTimeState``
    One shared temporal snapshot for every geometry component. It records
    clip-local time, retimed source time, segment ownership, and whether a
    transition handle is holding an endpoint.

``compile_scalar_expression`` / ``compile_vec2_expressions``
    Clamp pre/post-roll to endpoint holds, map every retime segment without
    averaging, then reproduce the animation kernel as deterministic FFmpeg
    expressions.

``sample_scalar_frames`` / ``sample_vec2_frames``
    Produce an exact-time, per-frame command plan for filters such as
    ``perspective`` that cannot evaluate an expression for every option.

Important invariants
--------------------

* Every time entering this boundary is an exact ``Fraction``.  Generated
  expressions retain rational times as numerator/denominator operations.
* Reverse, freeze, variable-rate, discontinuous, and repeated source ranges
  are represented segment by segment.  There is no average-speed fallback.
* Times before or after the clip hold the first or last rendered value.  This
  makes transition pre-roll and post-roll deterministic.
* Expression size, control-point count, retime-segment count, and their
  product are bounded.  Oversized plans fail explicitly instead of handing an
  unbounded expression to FFmpeg.

Central integration seam
------------------------

The central FFmpeg builder should consume the typed tracks already attached
to ``model.RenderTransformAnimation`` (and future opacity/audio records):

1. Use ``compile_*_expression`` when a filter option supports per-frame
   FFmpeg expressions.
2. Use ``sample_*_frames`` plus a renderer-owned ``sendcmd`` file when a
   filter option is command-only.
3. Surface ``AnimationExecutionError`` as a compatibility finding or compile
   error.  Never fall back to a static or endpoint-average value.

This module deliberately does not import or mutate the shared compiler or
FFmpeg builder.  It is the isolated Wave 2 contract for that later integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Literal, Sequence, TypeAlias

from .animation import (
    ScalarControlPoint,
    TimelineAnimatedScalar,
    TimelineAnimatedVec2,
    Vec2ControlPoint,
)
from .retime import RetimeMap, SegmentKind


TimeOrigin: TypeAlias = Literal["clip_local", "absolute"]
ClockPhase: TypeAlias = Literal[
    "transition_pre_roll",
    "clip",
    "transition_post_roll",
]


class AnimationExecutionError(ValueError):
    """Base error for an animation that cannot be executed faithfully."""


class AnimationExpressionLimitError(AnimationExecutionError):
    """A valid animation exceeds the explicitly configured FFmpeg bounds."""


class AnimationClockError(AnimationExecutionError):
    """The FFmpeg time origin or render window is ambiguous or invalid."""


def _exact_time(value: object, *, name: str) -> Fraction:
    """Require an exact rational time at the FFmpeg execution boundary."""

    if isinstance(value, bool):
        raise AnimationClockError(f"{name} must be an exact Fraction, not bool")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        raise AnimationClockError(f"{name} must be an exact Fraction, not float")
    raise AnimationClockError(
        f"{name} must be an exact Fraction, got {type(value).__name__}"
    )


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AnimationExpressionLimitError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class AnimationExpressionLimits:
    """Hard resource limits for generated FFmpeg expressions.

    Main callers:
    - Registry-owned renderer configuration.

    Why this exists:
    - FFmpeg's expression parser is recursive.  Explicit limits prevent a
      large untrusted project from creating excessive parser work or memory.
    """

    max_control_points: int = 64
    max_retime_segments: int = 64
    max_segment_point_product: int = 1024
    max_expression_chars: int = 65_536

    def __post_init__(self) -> None:
        for name in (
            "max_control_points",
            "max_retime_segments",
            "max_segment_point_product",
            "max_expression_chars",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )


@dataclass(frozen=True)
class ClipTimeState:
    """One exact clock/retime state shared by a clip's geometry tracks.

    ``clip_time`` and ``track_time`` are endpoint-held during transition-only
    pre/post-roll. ``source_time`` is evaluated through the same exact map as
    video playback, including reverse and freeze segments.
    """

    input_time: Fraction
    unclamped_clip_time: Fraction
    clip_time: Fraction
    track_time: Fraction
    source_time: Fraction
    phase: ClockPhase
    retime_segment_index: int
    retime_segment_kind: SegmentKind

    @property
    def is_endpoint_hold(self) -> bool:
        return self.phase != "clip"


@dataclass(frozen=True)
class AnimationClock:
    """Map an FFmpeg filter time variable onto one clip's retime domain.

    ``clip_local`` means FFmpeg time zero is the start of the rendered window.
    With two seconds of pre-roll, the clip itself therefore begins at FFmpeg
    time two.  ``absolute`` means the FFmpeg variable is sequence time and
    ``absolute_clip_start`` identifies the clip's true start.
    """

    origin: TimeOrigin = "clip_local"
    absolute_clip_start: Fraction | None = None
    pre_roll: Fraction = Fraction(0)
    post_roll: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if self.origin not in {"clip_local", "absolute"}:
            raise AnimationClockError(
                f"origin must be 'clip_local' or 'absolute', got {self.origin!r}"
            )
        pre_roll = _exact_time(self.pre_roll, name="pre_roll")
        post_roll = _exact_time(self.post_roll, name="post_roll")
        if pre_roll < 0 or post_roll < 0:
            raise AnimationClockError("pre_roll and post_roll must be non-negative")
        object.__setattr__(self, "pre_roll", pre_roll)
        object.__setattr__(self, "post_roll", post_roll)

        absolute_start = self.absolute_clip_start
        if self.origin == "absolute":
            if absolute_start is None:
                raise AnimationClockError(
                    "absolute origin requires absolute_clip_start"
                )
            object.__setattr__(
                self,
                "absolute_clip_start",
                _exact_time(absolute_start, name="absolute_clip_start"),
            )
        elif absolute_start is not None:
            raise AnimationClockError(
                "clip_local origin must not provide absolute_clip_start"
            )

    @property
    def clip_start_in_input_time(self) -> Fraction:
        """Return the FFmpeg time at which the actual clip begins."""

        if self.origin == "clip_local":
            return self.pre_roll
        assert self.absolute_clip_start is not None
        return self.absolute_clip_start

    def input_window(self, clip_duration: Fraction) -> tuple[Fraction, Fraction]:
        """Return the exact inclusive/exclusive render-window coordinates."""

        duration = _exact_time(clip_duration, name="clip_duration")
        if duration <= 0:
            raise AnimationClockError("clip_duration must be positive")
        if self.origin == "clip_local":
            return Fraction(0), self.pre_roll + duration + self.post_roll
        assert self.absolute_clip_start is not None
        return (
            self.absolute_clip_start - self.pre_roll,
            self.absolute_clip_start + duration + self.post_roll,
        )

    def clip_time(self, input_time: Fraction, clip_duration: Fraction) -> Fraction:
        """Convert one exact FFmpeg time to bounded clip-local time."""

        exact_input = _exact_time(input_time, name="input_time")
        duration = _exact_time(clip_duration, name="clip_duration")
        local = exact_input - self.clip_start_in_input_time
        return min(max(local, Fraction(0)), duration)

    def state_at(
        self,
        input_time: Fraction,
        retime_map: RetimeMap,
    ) -> ClipTimeState:
        """Resolve one shared source-time state, including transition handles.

        Main callers:
        - Expression-plan reference evaluators and per-frame command plans.
        - Geometry integration for transform, corner pin, opacity, and Pan.

        Why this exists:
        - Each component previously combined FFmpeg time, pre-roll, and retime
          offsets independently. Small differences made geometry drift during
          reverse/freeze playback and overlapping transitions.
        """

        if not isinstance(retime_map, RetimeMap):
            raise AnimationClockError("retime_map must be a RetimeMap")
        exact_input = _exact_time(input_time, name="input_time")
        duration = retime_map.timeline_duration
        unclamped = exact_input - self.clip_start_in_input_time
        if unclamped < 0:
            phase: ClockPhase = "transition_pre_roll"
        elif unclamped > duration:
            phase = "transition_post_roll"
        else:
            phase = "clip"
        clip_time = min(max(unclamped, Fraction(0)), duration)
        track_time = retime_map.timeline_start + clip_time
        sample = retime_map.sample(track_time)
        return ClipTimeState(
            input_time=exact_input,
            unclamped_clip_time=unclamped,
            clip_time=clip_time,
            track_time=track_time,
            source_time=sample.source_time,
            phase=phase,
            retime_segment_index=sample.segment_index,
            retime_segment_kind=sample.segment_kind,
        )


@dataclass(frozen=True)
class ScalarExpressionPlan:
    """One scalar FFmpeg expression plus its exact reference evaluator."""

    expression: str
    track: TimelineAnimatedScalar
    clock: AnimationClock
    time_variable: str

    @property
    def clip_duration(self) -> Fraction:
        return self.track.retime_map.timeline_duration

    @property
    def input_window_start(self) -> Fraction:
        return self.clock.input_window(self.clip_duration)[0]

    @property
    def input_window_end(self) -> Fraction:
        return self.clock.input_window(self.clip_duration)[1]

    def value_at_input_time(self, input_time: Fraction) -> float:
        """Evaluate the same clock and endpoint holds as the expression."""

        state = self.clock.state_at(input_time, self.track.retime_map)
        return self.track.source_track.value_at(state.source_time)


@dataclass(frozen=True)
class Vec2ExpressionPlan:
    """Two FFmpeg expressions sharing one exact time/retime contract."""

    x_expression: str
    y_expression: str
    track: TimelineAnimatedVec2
    clock: AnimationClock
    time_variable: str

    @property
    def clip_duration(self) -> Fraction:
        return self.track.retime_map.timeline_duration

    @property
    def input_window_start(self) -> Fraction:
        return self.clock.input_window(self.clip_duration)[0]

    @property
    def input_window_end(self) -> Fraction:
        return self.clock.input_window(self.clip_duration)[1]

    def value_at_input_time(self, input_time: Fraction) -> tuple[float, float]:
        state = self.clock.state_at(input_time, self.track.retime_map)
        return self.track.source_track.value_at(state.source_time)


@dataclass(frozen=True)
class ScalarFrameCommand:
    """One exact per-frame scalar command for a command-only filter."""

    frame_index: int
    input_time: Fraction
    clip_time: Fraction
    track_time: Fraction
    value: float


@dataclass(frozen=True)
class Vec2FrameCommand:
    """One exact per-frame two-component command for a command-only filter."""

    frame_index: int
    input_time: Fraction
    clip_time: Fraction
    track_time: Fraction
    value: tuple[float, float]


def _fraction_expression(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"({value.numerator}/{value.denominator})"


def _float_expression(value: float) -> str:
    if not math.isfinite(value):
        raise AnimationExecutionError("animation expression value must be finite")
    if abs(value) < 5e-16:
        value = 0.0
    return format(value, ".17g")


def _validate_time_variable(value: object) -> str:
    if value in {"t", "T"}:
        return str(value)
    if isinstance(value, str) and re.fullmatch(
        r"\(on-1\)\*[1-9][0-9]*/[1-9][0-9]*",
        value,
    ):
        # ``perspective=eval=frame`` exposes only its output frame number
        # (``on``), not the ordinary video-filter time variable. FFmpeg defines
        # that counter as one-based, so subtract one before multiplying by the
        # exact frame duration. Accepting only this closed renderer-owned form
        # keeps XML/user text out of FFmpeg expressions.
        return f"({value})"
    else:
        raise AnimationExecutionError(
            "time_variable must be 't' or 'T', or the renderer-owned exact "
            "perspective clock '(on-1)*N/D'"
        )


def _check_track_bounds(
    *,
    point_count: int,
    segment_count: int,
    limits: AnimationExpressionLimits,
) -> None:
    if point_count > limits.max_control_points:
        raise AnimationExpressionLimitError(
            f"animation has {point_count} control points; limit is "
            f"{limits.max_control_points}"
        )
    if segment_count > limits.max_retime_segments:
        raise AnimationExpressionLimitError(
            f"animation has {segment_count} retime segments; limit is "
            f"{limits.max_retime_segments}"
        )
    product = max(point_count - 1, 1) * segment_count
    if product > limits.max_segment_point_product:
        raise AnimationExpressionLimitError(
            f"animation expression complexity is {product}; limit is "
            f"{limits.max_segment_point_product}"
        )


def _clip_time_expression(
    *, time_variable: str, clock: AnimationClock, duration: Fraction
) -> str:
    start = _fraction_expression(clock.clip_start_in_input_time)
    return (
        f"clip(({time_variable})-({start}),0,{_fraction_expression(duration)})"
    )


def _track_time_expression(
    *, time_variable: str, clock: AnimationClock, timeline_start: Fraction,
    duration: Fraction
) -> str:
    local = _clip_time_expression(
        time_variable=time_variable,
        clock=clock,
        duration=duration,
    )
    return f"({_fraction_expression(timeline_start)}+({local}))"


def _source_time_expression(
    track: TimelineAnimatedScalar | TimelineAnimatedVec2,
    *,
    time_variable: str,
    clock: AnimationClock,
) -> str:
    """Map bounded FFmpeg time through every exact retime segment."""

    retime_map = track.retime_map
    timeline_time = _track_time_expression(
        time_variable=time_variable,
        clock=clock,
        timeline_start=retime_map.timeline_start,
        duration=retime_map.timeline_duration,
    )

    def mapped(segment_index: int) -> str:
        segment = retime_map.segments[segment_index]
        return (
            f"({_fraction_expression(segment.source_start)}+"
            f"(({timeline_time})-{_fraction_expression(segment.timeline_start)})*"
            f"{_fraction_expression(segment.rate)})"
        )

    # The final segment owns the map's final endpoint.  Every prior segment is
    # half-open, exactly matching RetimeMap.map_timeline.
    expression = mapped(len(retime_map.segments) - 1)
    for index in reversed(range(len(retime_map.segments) - 1)):
        boundary = _fraction_expression(retime_map.segments[index].timeline_end)
        expression = (
            f"if(lt(({timeline_time}),{boundary}),{mapped(index)},{expression})"
        )
    return expression


def _monotone_slopes(
    points: Sequence[ScalarControlPoint] | Sequence[Vec2ControlPoint],
    values: Sequence[float],
) -> tuple[float, ...]:
    """Mirror ``animation._monotone_slopes`` for expression coefficients.

    Why this exists:
    - The animation kernel intentionally keeps its interpolation helpers
      private.  This execution module freezes identical coefficients without
      exposing an additional central API during isolated development.
    """

    count = len(points)
    if count == 1:
        return (0.0,)
    widths = [
        float(points[index + 1].time - points[index].time)
        for index in range(count - 1)
    ]
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

    def endpoint(
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
        if (
            adjacent_secant * next_secant < 0.0
            and abs(slope) > abs(3.0 * adjacent_secant)
        ):
            return 3.0 * adjacent_secant
        return slope

    slopes[0] = endpoint(widths[0], widths[1], secants[0], secants[1])
    slopes[-1] = endpoint(widths[-1], widths[-2], secants[-1], secants[-2])
    return tuple(slopes)


def _easing_expression(raw: str, interpolation: str) -> str:
    if interpolation == "linear":
        return raw
    if interpolation == "ease":
        return f"({raw})*({raw})*(3-2*({raw}))"
    if interpolation == "ease-in":
        return f"({raw})*({raw})*({raw})"
    if interpolation == "ease-out":
        return f"1-(1-({raw}))*(1-({raw}))*(1-({raw}))"
    raise AnimationExecutionError(
        f"unsupported animation interpolation {interpolation!r}"
    )


def _component_expression(
    *,
    points: Sequence[ScalarControlPoint] | Sequence[Vec2ControlPoint],
    values: Sequence[float],
    source_time: str,
) -> str:
    if len(points) == 1:
        return _float_expression(values[0])

    slopes = _monotone_slopes(points, values)
    segments: list[str] = []
    for index, (left, right) in enumerate(zip(points, points[1:])):
        raw = (
            f"(({source_time})-{_fraction_expression(left.time)})/"
            f"{_fraction_expression(right.time-left.time)}"
        )
        eased = _easing_expression(raw, right.interpolation)
        if right.curve == "linear":
            value = (
                f"({_float_expression(values[index])}+"
                f"({_float_expression(values[index+1]-values[index])})*({eased}))"
            )
        elif right.curve == "smooth":
            squared = f"({eased})*({eased})"
            cubed = f"({squared})*({eased})"
            width = float(right.time - left.time)
            value = (
                f"((2*({cubed})-3*({squared})+1)*"
                f"{_float_expression(values[index])}+"
                f"(({cubed})-2*({squared})+({eased}))*"
                f"{_float_expression(width*slopes[index])}+"
                f"(-2*({cubed})+3*({squared}))*"
                f"{_float_expression(values[index+1])}+"
                f"(({cubed})-({squared}))*"
                f"{_float_expression(width*slopes[index+1])})"
            )
            low = _float_expression(min(values[index], values[index + 1]))
            high = _float_expression(max(values[index], values[index + 1]))
            value = f"clip({value},{low},{high})"
        else:
            raise AnimationExecutionError(
                f"unsupported animation value curve {right.curve!r}"
            )
        segments.append(value)

    # Explicit endpoint holds are based on source time, after the timeline has
    # already been clamped for transition pre/post-roll.
    expression = _float_expression(values[-1])
    for index in reversed(range(len(segments))):
        expression = (
            f"if(lte(({source_time}),{_fraction_expression(points[index+1].time)}),"
            f"{segments[index]},{expression})"
        )
    return (
        f"if(lte(({source_time}),{_fraction_expression(points[0].time)}),"
        f"{_float_expression(values[0])},{expression})"
    )


def _enforce_expression_size(
    expression: str,
    *,
    limits: AnimationExpressionLimits,
    component: str,
) -> None:
    if len(expression) > limits.max_expression_chars:
        raise AnimationExpressionLimitError(
            f"{component} expression has {len(expression)} characters; limit is "
            f"{limits.max_expression_chars}"
        )


def compile_scalar_expression(
    track: TimelineAnimatedScalar,
    *,
    clock: AnimationClock = AnimationClock(),
    time_variable: str = "t",
    limits: AnimationExpressionLimits = AnimationExpressionLimits(),
) -> ScalarExpressionPlan:
    """Compile a scalar track without flattening its retime map.

    Main callers:
    - Geometry, opacity, and audio filter builders during central integration.
    """

    if not isinstance(track, TimelineAnimatedScalar):
        raise AnimationExecutionError("track must be TimelineAnimatedScalar")
    if not isinstance(clock, AnimationClock):
        raise AnimationClockError("clock must be AnimationClock")
    if not isinstance(limits, AnimationExpressionLimits):
        raise AnimationExpressionLimitError(
            "limits must be AnimationExpressionLimits"
        )
    variable = _validate_time_variable(time_variable)
    points = track.source_track.control_points
    _check_track_bounds(
        point_count=len(points),
        segment_count=len(track.retime_map.segments),
        limits=limits,
    )
    source_time = _source_time_expression(
        track,
        time_variable=variable,
        clock=clock,
    )
    expression = _component_expression(
        points=points,
        values=tuple(point.value for point in points),
        source_time=source_time,
    )
    _enforce_expression_size(expression, limits=limits, component="scalar")
    return ScalarExpressionPlan(expression, track, clock, variable)


def compile_vec2_expressions(
    track: TimelineAnimatedVec2,
    *,
    clock: AnimationClock = AnimationClock(),
    time_variable: str = "t",
    limits: AnimationExpressionLimits = AnimationExpressionLimits(),
) -> Vec2ExpressionPlan:
    """Compile each vector component against one shared exact retime map."""

    if not isinstance(track, TimelineAnimatedVec2):
        raise AnimationExecutionError("track must be TimelineAnimatedVec2")
    if not isinstance(clock, AnimationClock):
        raise AnimationClockError("clock must be AnimationClock")
    if not isinstance(limits, AnimationExpressionLimits):
        raise AnimationExpressionLimitError(
            "limits must be AnimationExpressionLimits"
        )
    variable = _validate_time_variable(time_variable)
    points = track.source_track.control_points
    _check_track_bounds(
        point_count=len(points),
        segment_count=len(track.retime_map.segments),
        limits=limits,
    )
    source_time = _source_time_expression(
        track,
        time_variable=variable,
        clock=clock,
    )
    x_expression = _component_expression(
        points=points,
        values=tuple(point.value[0] for point in points),
        source_time=source_time,
    )
    y_expression = _component_expression(
        points=points,
        values=tuple(point.value[1] for point in points),
        source_time=source_time,
    )
    _enforce_expression_size(x_expression, limits=limits, component="vec2 x")
    _enforce_expression_size(y_expression, limits=limits, component="vec2 y")
    return Vec2ExpressionPlan(x_expression, y_expression, track, clock, variable)


def _frame_times(
    *,
    window_start: Fraction,
    frame_rate: Fraction,
    frame_count: int,
    first_frame: int,
) -> tuple[tuple[int, Fraction], ...]:
    rate = _exact_time(frame_rate, name="frame_rate")
    if rate <= 0:
        raise AnimationClockError("frame_rate must be positive")
    count = _positive_int(frame_count, name="frame_count")
    if isinstance(first_frame, bool) or not isinstance(first_frame, int):
        raise AnimationClockError("first_frame must be an integer")
    if first_frame < 0:
        raise AnimationClockError("first_frame must be non-negative")
    return tuple(
        (
            frame_index,
            window_start + Fraction(frame_index, 1) / rate,
        )
        for frame_index in range(first_frame, first_frame + count)
    )


def sample_scalar_frames(
    plan: ScalarExpressionPlan,
    *,
    frame_rate: Fraction,
    frame_count: int,
    first_frame: int = 0,
) -> tuple[ScalarFrameCommand, ...]:
    """Build deterministic scalar commands at exact output-frame times."""

    if not isinstance(plan, ScalarExpressionPlan):
        raise AnimationExecutionError("plan must be ScalarExpressionPlan")
    commands: list[ScalarFrameCommand] = []
    for frame_index, input_time in _frame_times(
        window_start=plan.input_window_start,
        frame_rate=frame_rate,
        frame_count=frame_count,
        first_frame=first_frame,
    ):
        state = plan.clock.state_at(input_time, plan.track.retime_map)
        commands.append(
            ScalarFrameCommand(
                frame_index=frame_index,
                input_time=input_time,
                clip_time=state.clip_time,
                track_time=state.track_time,
                value=plan.track.source_track.value_at(state.source_time),
            )
        )
    return tuple(commands)


def sample_vec2_frames(
    plan: Vec2ExpressionPlan,
    *,
    frame_rate: Fraction,
    frame_count: int,
    first_frame: int = 0,
) -> tuple[Vec2FrameCommand, ...]:
    """Build deterministic vector commands at exact output-frame times."""

    if not isinstance(plan, Vec2ExpressionPlan):
        raise AnimationExecutionError("plan must be Vec2ExpressionPlan")
    commands: list[Vec2FrameCommand] = []
    for frame_index, input_time in _frame_times(
        window_start=plan.input_window_start,
        frame_rate=frame_rate,
        frame_count=frame_count,
        first_frame=first_frame,
    ):
        state = plan.clock.state_at(input_time, plan.track.retime_map)
        commands.append(
            Vec2FrameCommand(
                frame_index=frame_index,
                input_time=input_time,
                clip_time=state.clip_time,
                track_time=state.track_time,
                value=plan.track.source_track.value_at(state.source_time),
            )
        )
    return tuple(commands)


__all__ = [
    "AnimationClock",
    "AnimationClockError",
    "AnimationExecutionError",
    "AnimationExpressionLimitError",
    "AnimationExpressionLimits",
    "ClipTimeState",
    "ClockPhase",
    "ScalarExpressionPlan",
    "ScalarFrameCommand",
    "TimeOrigin",
    "Vec2ExpressionPlan",
    "Vec2FrameCommand",
    "compile_scalar_expression",
    "compile_vec2_expressions",
    "sample_scalar_frames",
    "sample_vec2_frames",
]
