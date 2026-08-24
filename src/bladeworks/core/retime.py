"""Exact, piecewise-linear retiming contracts for the portable renderer.

Architecture map
================

``RetimePoint``
    One FCPXML ``timept`` expressed as exact output/source times.

``RetimeSegment``
    The exact linear mapping between two adjacent points.  A positive rate is
    forward playback, a negative rate is reverse playback, and a zero rate is
    a freeze.

``RetimeMap``
    A contiguous, ordered sequence of segments.  Several segments with
    different rates form a variable-speed ramp without converting rational
    timeline values to floating point.

``TimelineOccurrence``
    One inverse lookup result.  Moving media produces a single timeline point;
    a frozen source frame produces an interval.  Reused or reversed source
    ranges can therefore return several occurrences without losing meaning.

``RetimeSample`` / ``RetimeBoundary``
    Explain which segment owns an exact output coordinate and what happens at
    each segment boundary.  Execution tests can therefore assert freeze,
    reverse, source-jump, and terminal ownership without reconstructing the
    half-open rules themselves.

Important invariants
--------------------

* All timeline and source coordinates are ``fractions.Fraction`` values.
* Output segments are contiguous and never overlap.
* Segment boundaries use half-open ownership, except for the map's final end.
  This makes lookup deterministic and prevents duplicate boundary keyframes.
* Nonlinear interpolation is rejected explicitly.  It is never averaged into
  a misleading constant speed.

Why this exists
---------------

The old compiler reduced every FCPXML ``timeMap`` to one endpoint-average
speed.  That loses ramps, reverse playback, freezes, and repeated source-time
occurrences.  This module freezes the timing contract before parser, compiler,
and FFmpeg integration are changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
from typing import Iterable, Literal, Protocol, Sequence, runtime_checkable


SegmentKind = Literal["forward", "reverse", "freeze"]


class RetimeError(ValueError):
    """Base error for invalid or unsupported retime contracts."""


class RetimeValidationError(RetimeError):
    """The supplied points or segments cannot form a deterministic map."""


class UnsupportedRetimeMappingError(RetimeError):
    """The source requests interpolation this exact kernel cannot execute."""


class TimelineOutsideRetimeMapError(RetimeError):
    """A timeline lookup falls outside the map's closed outer domain."""


@runtime_checkable
class ParsedTimeMapPoint(Protocol):
    """Small adapter boundary matching the parser's existing ``TimeMapPoint``."""

    time: Fraction
    value: Fraction
    interp: str | None


def _exact_fraction(value: object, *, field_name: str) -> Fraction:
    """Return an exact rational coordinate or fail with a focused message.

    Main callers:
    - Dataclass validation for every public timing field.

    Why this exists:
    - ``Fraction(0.1)`` silently captures the binary float approximation of
      0.1.  Renderer timing must instead enter this boundary already exact.
    """

    if isinstance(value, bool):
        raise RetimeValidationError(f"{field_name} must be an exact Fraction, not bool")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RetimeValidationError(f"{field_name} must be finite")
        raise RetimeValidationError(
            f"{field_name} must be an exact Fraction, not float"
        )
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise RetimeValidationError(f"{field_name} must be finite")
        raise RetimeValidationError(
            f"{field_name} must be an exact Fraction, not Decimal"
        )
    raise RetimeValidationError(
        f"{field_name} must be an exact Fraction, got {type(value).__name__}"
    )


@dataclass(frozen=True)
class RetimePoint:
    """One exact output/source control point from an FCPXML time map."""

    timeline_time: Fraction
    source_time: Fraction
    interpolation: str | None = "linear"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timeline_time",
            _exact_fraction(self.timeline_time, field_name="timeline_time"),
        )
        object.__setattr__(
            self,
            "source_time",
            _exact_fraction(self.source_time, field_name="source_time"),
        )
        interpolation = self.interpolation
        if interpolation is not None and not isinstance(interpolation, str):
            raise RetimeValidationError("interpolation must be a string or None")


@dataclass(frozen=True)
class RetimeSegment:
    """One exact linear output-to-source mapping.

    ``timeline_end`` is an exclusive boundary when the segment belongs to a
    map, except for the last segment.  ``map_timeline`` itself includes both
    ends so a standalone segment remains intuitive.
    """

    timeline_start: Fraction
    timeline_end: Fraction
    source_start: Fraction
    source_end: Fraction

    def __post_init__(self) -> None:
        for field_name in (
            "timeline_start",
            "timeline_end",
            "source_start",
            "source_end",
        ):
            object.__setattr__(
                self,
                field_name,
                _exact_fraction(getattr(self, field_name), field_name=field_name),
            )
        if self.timeline_end <= self.timeline_start:
            raise RetimeValidationError(
                "retime segment timeline_end must be greater than timeline_start"
            )

    @property
    def timeline_duration(self) -> Fraction:
        return self.timeline_end - self.timeline_start

    @property
    def source_duration(self) -> Fraction:
        return self.source_end - self.source_start

    @property
    def rate(self) -> Fraction:
        """Source seconds consumed per output timeline second."""

        return self.source_duration / self.timeline_duration

    @property
    def kind(self) -> SegmentKind:
        if self.rate > 0:
            return "forward"
        if self.rate < 0:
            return "reverse"
        return "freeze"

    @classmethod
    def from_rate(
        cls,
        *,
        timeline_start: Fraction,
        timeline_end: Fraction,
        source_start: Fraction,
        rate: Fraction,
    ) -> "RetimeSegment":
        """Construct a segment from an exact playback rate."""

        exact_start = _exact_fraction(timeline_start, field_name="timeline_start")
        exact_end = _exact_fraction(timeline_end, field_name="timeline_end")
        exact_source = _exact_fraction(source_start, field_name="source_start")
        exact_rate = _exact_fraction(rate, field_name="rate")
        if exact_end <= exact_start:
            raise RetimeValidationError(
                "retime segment timeline_end must be greater than timeline_start"
            )
        return cls(
            timeline_start=exact_start,
            timeline_end=exact_end,
            source_start=exact_source,
            source_end=exact_source + (exact_end - exact_start) * exact_rate,
        )

    def map_timeline(self, timeline_time: Fraction) -> Fraction:
        """Map one timeline coordinate through this standalone segment."""

        exact_time = _exact_fraction(timeline_time, field_name="timeline_time")
        if exact_time < self.timeline_start or exact_time > self.timeline_end:
            raise TimelineOutsideRetimeMapError(
                f"timeline time {exact_time} is outside segment "
                f"[{self.timeline_start}, {self.timeline_end}]"
            )
        return self.source_start + (exact_time - self.timeline_start) * self.rate


@dataclass(frozen=True)
class TimelineOccurrence:
    """A source time's point or frozen interval on the output timeline."""

    timeline_start: Fraction
    timeline_end: Fraction
    includes_start: bool
    includes_end: bool
    segment_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timeline_start",
            _exact_fraction(self.timeline_start, field_name="timeline_start"),
        )
        object.__setattr__(
            self,
            "timeline_end",
            _exact_fraction(self.timeline_end, field_name="timeline_end"),
        )
        if self.timeline_end < self.timeline_start:
            raise RetimeValidationError(
                "timeline occurrence end must not precede its start"
            )
        if not self.segment_indices:
            raise RetimeValidationError("timeline occurrence requires a segment index")
        if self.timeline_start == self.timeline_end and not (
            self.includes_start and self.includes_end
        ):
            raise RetimeValidationError(
                "a point occurrence must include its coordinate"
            )

    @property
    def is_interval(self) -> bool:
        """Whether a frozen frame occupies more than one timeline coordinate."""

        return self.timeline_end > self.timeline_start

    @property
    def timeline_time(self) -> Fraction:
        """Return the coordinate for a point occurrence.

        Frozen occurrences deliberately require callers to inspect the interval
        instead of quietly choosing one arbitrary time.
        """

        if self.is_interval:
            raise RetimeValidationError(
                "a freeze occurrence is an interval, not one timeline time"
            )
        return self.timeline_start


@dataclass(frozen=True)
class RetimeSample:
    """One exact output-to-source lookup with explicit segment ownership."""

    timeline_time: Fraction
    source_time: Fraction
    segment_index: int
    segment_kind: SegmentKind
    at_segment_start: bool
    at_segment_end: bool


@dataclass(frozen=True)
class RetimeBoundary:
    """One internal boundary and the source coordinates on either side.

    ``source_at_boundary`` is the value owned by the incoming segment.  The
    outgoing segment's endpoint remains available as ``outgoing_source_end``
    so discontinuous source jumps cannot be mistaken for continuous ramps.
    """

    timeline_time: Fraction
    outgoing_segment_index: int
    incoming_segment_index: int
    outgoing_source_end: Fraction
    source_at_boundary: Fraction
    outgoing_kind: SegmentKind
    incoming_kind: SegmentKind

    @property
    def source_is_continuous(self) -> bool:
        return self.outgoing_source_end == self.source_at_boundary


@dataclass(frozen=True)
class RetimeMap:
    """A validated, contiguous piecewise-linear retime map.

    Main callers:
    - The future source-to-render IR compiler constructs this from a
      ``StoryNode.time_map`` via ``from_time_map_points``.
    - Animation compilation uses ``source_to_timeline_occurrences`` to project
      source-local keyframes onto every matching timeline location.
    - The FFmpeg builder consumes ``segments`` in order and renders each exact
      source trim/output duration independently.
    """

    segments: tuple[RetimeSegment, ...]

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        object.__setattr__(self, "segments", segments)
        if not segments:
            raise RetimeValidationError("retime map requires at least one segment")
        for index, segment in enumerate(segments):
            if not isinstance(segment, RetimeSegment):
                raise RetimeValidationError(
                    f"segments[{index}] must be a RetimeSegment"
                )
            if index == 0:
                continue
            previous = segments[index - 1]
            if segment.timeline_start < previous.timeline_start:
                raise RetimeValidationError(
                    f"segments[{index}] has a nonmonotonic timeline domain"
                )
            if segment.timeline_start < previous.timeline_end:
                raise RetimeValidationError(
                    f"segments[{index}] overlaps segments[{index - 1}]"
                )
            if segment.timeline_start > previous.timeline_end:
                raise RetimeValidationError(
                    f"segments[{index}] leaves a timeline gap after segments[{index - 1}]"
                )

    @classmethod
    def identity(
        cls,
        duration: Fraction,
        *,
        timeline_start: Fraction = Fraction(0),
        source_start: Fraction = Fraction(0),
    ) -> "RetimeMap":
        """Construct a 1x map for a clip without an explicit ``timeMap``."""

        exact_duration = _exact_fraction(duration, field_name="duration")
        if exact_duration <= 0:
            raise RetimeValidationError("identity duration must be positive")
        exact_timeline = _exact_fraction(timeline_start, field_name="timeline_start")
        exact_source = _exact_fraction(source_start, field_name="source_start")
        return cls(
            (
                RetimeSegment.from_rate(
                    timeline_start=exact_timeline,
                    timeline_end=exact_timeline + exact_duration,
                    source_start=exact_source,
                    rate=Fraction(1),
                ),
            )
        )

    @classmethod
    def from_points(
        cls,
        points: Iterable[RetimePoint],
        *,
        linearize_two_point_smooth2: bool = False,
    ) -> "RetimeMap":
        """Build exact linear segments from ordered explicit retime points.

        Multiple adjacent segments may have different rates; that sequence is
        the exact piecewise-linear representation of a variable-speed ramp.
        """

        frozen_points = tuple(points)
        if len(frozen_points) < 2:
            raise RetimeValidationError("retime map requires at least two points")
        linearize_smooth2 = linearize_two_point_smooth2 and len(frozen_points) == 2 and all(
            (point.interpolation or "smooth2").strip().lower() == "smooth2"
            for point in frozen_points
        )
        for index, point in enumerate(frozen_points):
            if not isinstance(point, RetimePoint):
                raise RetimeValidationError(f"points[{index}] must be a RetimePoint")
            # ``None`` is the FCPXML 1.14 DTD default ``smooth2``.  It is not
            # safe to linearize that curve until it has been calibrated.
            interpolation = (
                "smooth2"
                if point.interpolation is None
                else point.interpolation.strip().lower()
            )
            if interpolation != "linear" and not linearize_smooth2:
                raise UnsupportedRetimeMappingError(
                    f"points[{index}] requests unsupported nonlinear "
                    f"interpolation {interpolation!r}"
                )
        return cls(
            tuple(
                RetimeSegment(
                    timeline_start=left.timeline_time,
                    timeline_end=right.timeline_time,
                    source_start=left.source_time,
                    source_end=right.source_time,
                )
                for left, right in zip(frozen_points, frozen_points[1:])
            )
        )

    @classmethod
    def from_points_visible(
        cls,
        points: Iterable[RetimePoint],
        visible_duration: Fraction,
        *,
        linearize_two_point_smooth2: bool = False,
    ) -> "RetimeMap":
        """Build only the authored portion that can affect a visible item.

        Main callers:
        - Video compilation for a ``StoryNode.duration``.
        - Audio execution for the independently scheduled audio duration.

        Final Cut may append a nonlinear terminal point after the visible
        interval solely to describe media-end interpolation context. Validating
        that unreachable point first falsely rejects an otherwise executable
        clip. This routine trims raw points before capability validation. If
        the visible boundary lies inside a nonlinear segment, the synthesized
        boundary retains that interpolation and ``from_points`` still rejects
        it; no unsupported visible curve is silently linearized.
        """

        frozen_points = tuple(points)
        if len(frozen_points) < 2:
            raise RetimeValidationError("retime map requires at least two points")
        for index, point in enumerate(frozen_points):
            if not isinstance(point, RetimePoint):
                raise RetimeValidationError(
                    f"points[{index}] must be a RetimePoint"
                )
            if index and point.timeline_time <= frozen_points[index - 1].timeline_time:
                raise RetimeValidationError(
                    "retime point timeline times must be strictly increasing"
                )
        duration = _exact_fraction(
            visible_duration,
            field_name="visible_duration",
        )
        if duration <= 0:
            raise RetimeValidationError("visible_duration must be positive")
        if frozen_points[0].timeline_time != 0:
            raise RetimeValidationError(
                "visible retime map must start at timeline time 0"
            )
        if duration > frozen_points[-1].timeline_time:
            raise RetimeValidationError(
                f"visible duration {duration} is not covered by retime points"
            )

        retained = [point for point in frozen_points if point.timeline_time <= duration]
        if retained[-1].timeline_time < duration:
            right = next(
                point for point in frozen_points if point.timeline_time > duration
            )
            left = retained[-1]
            progress = (duration - left.timeline_time) / (
                right.timeline_time - left.timeline_time
            )
            retained.append(
                RetimePoint(
                    timeline_time=duration,
                    source_time=left.source_time
                    + progress * (right.source_time - left.source_time),
                    interpolation=right.interpolation,
                )
            )
        return cls.from_points(
            retained,
            linearize_two_point_smooth2=linearize_two_point_smooth2,
        )

    @classmethod
    def from_time_map_points(
        cls,
        points: Iterable[ParsedTimeMapPoint],
        *,
        linearize_two_point_smooth2: bool = False,
    ) -> "RetimeMap":
        """Adapt the parser's existing ``TimeMapPoint`` records.

        This is intentionally the only structural adapter: callers must pass
        objects exposing the parser's exact ``time``, ``value``, and ``interp``
        fields.  Dictionaries, floats, and guessed aliases are rejected.
        """

        adapted: list[RetimePoint] = []
        for index, point in enumerate(points):
            if not isinstance(point, ParsedTimeMapPoint):
                raise RetimeValidationError(
                    f"time map points[{index}] must expose time, value, and interp"
                )
            adapted.append(
                RetimePoint(
                    timeline_time=point.time,
                    source_time=point.value,
                    interpolation=point.interp,
                )
            )
        return cls.from_points(
            adapted,
            linearize_two_point_smooth2=linearize_two_point_smooth2,
        )

    @classmethod
    def from_time_map_points_visible(
        cls,
        points: Iterable[ParsedTimeMapPoint],
        visible_duration: Fraction,
        *,
        linearize_two_point_smooth2: bool = False,
    ) -> "RetimeMap":
        """Adapt parser points and validate only their visible interval."""

        adapted: list[RetimePoint] = []
        for index, point in enumerate(points):
            if not isinstance(point, ParsedTimeMapPoint):
                raise RetimeValidationError(
                    f"time map points[{index}] must expose time, value, and interp"
                )
            adapted.append(
                RetimePoint(
                    timeline_time=point.time,
                    source_time=point.value,
                    interpolation=point.interp,
                )
            )
        return cls.from_points_visible(
            adapted,
            visible_duration,
            linearize_two_point_smooth2=linearize_two_point_smooth2,
        )

    @property
    def timeline_start(self) -> Fraction:
        return self.segments[0].timeline_start

    @property
    def timeline_end(self) -> Fraction:
        return self.segments[-1].timeline_end

    @property
    def timeline_duration(self) -> Fraction:
        return self.timeline_end - self.timeline_start

    def restrict_to_visible_duration(
        self,
        visible_duration: Fraction,
    ) -> "RetimeMap":
        """Return the exact zero-based portion rendered by one visible item.

        Main callers:
        - The video compiler, using ``StoryNode.duration``.
        - The audio executor, using the independent ``RenderAudioItem.duration``
          produced by ``audioStart``/``audioDuration`` and component clipping.

        Final Cut can retain a terminal ``timept`` after the item's visible
        duration.  That later point remains interpolation context: this method
        evaluates the enclosing supported linear/freeze/reverse segment at the
        visible boundary with exact ``Fraction`` arithmetic, then discards the
        post-visible portion.  It never extrapolates beyond the authored map.
        """

        duration = _exact_fraction(
            visible_duration,
            field_name="visible_duration",
        )
        if duration <= 0:
            raise RetimeValidationError("visible_duration must be positive")
        if self.timeline_start != 0:
            raise RetimeValidationError(
                "visible retime map must start at timeline time 0"
            )
        if duration > self.timeline_end:
            raise RetimeValidationError(
                f"visible duration {duration} is not covered by retime map "
                f"ending at {self.timeline_end}"
            )
        if duration == self.timeline_end:
            return self

        retained: list[RetimeSegment] = []
        for segment in self.segments:
            if segment.timeline_end <= duration:
                retained.append(segment)
                if segment.timeline_end == duration:
                    break
                continue
            if segment.timeline_start < duration < segment.timeline_end:
                retained.append(
                    RetimeSegment.from_rate(
                        timeline_start=segment.timeline_start,
                        timeline_end=duration,
                        source_start=segment.source_start,
                        rate=segment.rate,
                    )
                )
                break
            if duration == segment.timeline_start:
                break

        if not retained or retained[-1].timeline_end != duration:
            raise RetimeValidationError(
                f"visible duration {duration} is not covered by a retime segment"
            )
        return RetimeMap(tuple(retained))

    def restrict_to_timeline_window(
        self,
        timeline_start: Fraction,
        visible_duration: Fraction,
    ) -> "RetimeMap":
        """Select and zero-base one visible window from an authored map.

        Main callers:
        - ``resolve_instance_stream_timing`` for a clip whose ``start`` selects
          a range inside the retimed media timeline.

        Why this exists:
        Final Cut writes a ``timeMap`` for the complete retimed media object,
        then uses the containing clip's ``start`` and ``duration`` to select a
        visible subrange.  Real exports therefore commonly contain a map such
        as ``0 -> 0, 10 -> 20`` on a one-second edit starting at output time
        eight.  Treating that edit as the map's first second decodes the wrong
        source frames and can turn a valid multicam selection into a gap.

        Source coordinates remain in the authored source domain.  Only the
        selected output coordinates are rebased to zero so downstream retime
        execution can keep its clip-local clock contract.
        """

        start = _exact_fraction(timeline_start, field_name="timeline_start")
        duration = _exact_fraction(
            visible_duration,
            field_name="visible_duration",
        )
        if duration <= 0:
            raise RetimeValidationError("visible_duration must be positive")
        end = start + duration
        if start < self.timeline_start or end > self.timeline_end:
            raise RetimeValidationError(
                f"visible timeline window [{start}, {end}) is not covered by "
                f"retime map [{self.timeline_start}, {self.timeline_end}]"
            )

        retained: list[RetimeSegment] = []
        for segment in self.segments:
            overlap_start = max(start, segment.timeline_start)
            overlap_end = min(end, segment.timeline_end)
            if overlap_end <= overlap_start:
                continue
            retained.append(
                RetimeSegment(
                    timeline_start=overlap_start - start,
                    timeline_end=overlap_end - start,
                    source_start=segment.map_timeline(overlap_start),
                    source_end=segment.map_timeline(overlap_end),
                )
            )

        if (
            not retained
            or retained[0].timeline_start != 0
            or retained[-1].timeline_end != duration
        ):
            raise RetimeValidationError(
                f"visible timeline window [{start}, {end}) is not completely "
                "covered by retime segments"
            )
        return RetimeMap(tuple(retained))

    @property
    def rates(self) -> tuple[Fraction, ...]:
        return tuple(segment.rate for segment in self.segments)

    @property
    def is_variable_rate(self) -> bool:
        return len(set(self.rates)) > 1

    @property
    def boundaries(self) -> tuple[RetimeBoundary, ...]:
        """Return every internal boundary with its half-open owner.

        Main callers:
        - Deterministic retime execution diagnostics and XYZT oracle checks.

        Why this exists:
        - Looking only at adjacent rates hides discontinuous source jumps.
          This record makes the exact source frame selected at a boundary
          inspectable without sampling an arbitrary epsilon on either side.
        """

        return tuple(
            RetimeBoundary(
                timeline_time=incoming.timeline_start,
                outgoing_segment_index=index,
                incoming_segment_index=index + 1,
                outgoing_source_end=outgoing.source_end,
                source_at_boundary=incoming.source_start,
                outgoing_kind=outgoing.kind,
                incoming_kind=incoming.kind,
            )
            for index, (outgoing, incoming) in enumerate(
                zip(self.segments, self.segments[1:])
            )
        )

    def sample(self, timeline_time: Fraction) -> RetimeSample:
        """Map one coordinate and report the segment that owns it exactly."""

        exact_time = _exact_fraction(timeline_time, field_name="timeline_time")
        if exact_time < self.timeline_start or exact_time > self.timeline_end:
            raise TimelineOutsideRetimeMapError(
                f"timeline time {exact_time} is outside map "
                f"[{self.timeline_start}, {self.timeline_end}]"
            )
        last_index = len(self.segments) - 1
        for index, segment in enumerate(self.segments):
            owns_end = index == last_index
            if segment.timeline_start <= exact_time and (
                exact_time < segment.timeline_end
                or (owns_end and exact_time == segment.timeline_end)
            ):
                return RetimeSample(
                    timeline_time=exact_time,
                    source_time=segment.map_timeline(exact_time),
                    segment_index=index,
                    segment_kind=segment.kind,
                    at_segment_start=exact_time == segment.timeline_start,
                    at_segment_end=exact_time == segment.timeline_end,
                )
        raise TimelineOutsideRetimeMapError(
            f"timeline time {exact_time} is not owned by a retime segment"
        )

    def require_frame_aligned_boundaries(
        self,
        frame_duration: Fraction,
    ) -> tuple[int, ...]:
        """Return output-frame indices for all boundaries or fail explicitly.

        The map start is frame zero.  Both the final endpoint and every
        internal segment boundary must land on an integer frame.  This check
        is opt-in because imported Final Cut projects can contain a retime map
        whose source positions are subframe even though the visible output is
        later normalized by a caller.
        """

        duration = _exact_fraction(frame_duration, field_name="frame_duration")
        if duration <= 0:
            raise RetimeValidationError("frame_duration must be positive")
        indices: list[int] = []
        for segment in self.segments:
            coordinate = (segment.timeline_end - self.timeline_start) / duration
            if coordinate.denominator != 1:
                raise RetimeValidationError(
                    f"retime boundary {segment.timeline_end} is not aligned to "
                    f"frame duration {duration} from map start {self.timeline_start}"
                )
            indices.append(coordinate.numerator)
        return tuple(indices)

    def with_endpoint_holds(
        self,
        *,
        pre_roll: Fraction = Fraction(0),
        post_roll: Fraction = Fraction(0),
    ) -> "RetimeMap":
        """Return a zero-based map expanded by transition-only endpoint holds.

        Authored segments keep their exact durations, rates, and source
        coordinates.  Only the output domain shifts after ``pre_roll``; extra
        freeze segments hold the nearest endpoint source frame.  This is the
        shared temporal contract for geometry and media participating in an
        overlapping transition.
        """

        before = _exact_fraction(pre_roll, field_name="pre_roll")
        after = _exact_fraction(post_roll, field_name="post_roll")
        if before < 0 or after < 0:
            raise RetimeValidationError("pre_roll and post_roll must be non-negative")

        shifted: list[RetimeSegment] = []
        if before:
            first_source = self.segments[0].source_start
            shifted.append(
                RetimeSegment(
                    timeline_start=Fraction(0),
                    timeline_end=before,
                    source_start=first_source,
                    source_end=first_source,
                )
            )
        shift = before - self.timeline_start
        shifted.extend(
            RetimeSegment(
                timeline_start=segment.timeline_start + shift,
                timeline_end=segment.timeline_end + shift,
                source_start=segment.source_start,
                source_end=segment.source_end,
            )
            for segment in self.segments
        )
        if after:
            authored_end = before + self.timeline_duration
            last_source = self.segments[-1].source_end
            shifted.append(
                RetimeSegment(
                    timeline_start=authored_end,
                    timeline_end=authored_end + after,
                    source_start=last_source,
                    source_end=last_source,
                )
            )
        return RetimeMap(tuple(shifted))

    def map_timeline(self, timeline_time: Fraction) -> Fraction:
        """Map an exact output coordinate to one exact source coordinate."""

        return self.sample(timeline_time).source_time

    timeline_to_source = map_timeline

    def source_to_timeline_occurrences(
        self,
        source_time: Fraction,
    ) -> tuple[TimelineOccurrence, ...]:
        """Return every timeline point or freeze interval for ``source_time``.

        Moving segments invert exactly because each segment is linear.  Freeze
        segments return their complete timeline span.  Candidate boundary
        points are checked against ``map_timeline`` so half-open ownership and
        discontinuous source jumps stay deterministic.
        """

        exact_source = _exact_fraction(source_time, field_name="source_time")
        occurrences: list[TimelineOccurrence] = []
        last_index = len(self.segments) - 1
        for index, segment in enumerate(self.segments):
            owns_end = index == last_index
            if segment.kind == "freeze":
                if exact_source != segment.source_start:
                    continue
                occurrences.append(
                    TimelineOccurrence(
                        timeline_start=segment.timeline_start,
                        timeline_end=segment.timeline_end,
                        includes_start=True,
                        includes_end=owns_end,
                        segment_indices=(index,),
                    )
                )
                continue

            timeline_time = (
                segment.timeline_start
                + (exact_source - segment.source_start) / segment.rate
            )
            if timeline_time < segment.timeline_start:
                continue
            if timeline_time > segment.timeline_end:
                continue
            if timeline_time == segment.timeline_end and not owns_end:
                continue
            if self.map_timeline(timeline_time) != exact_source:
                continue
            occurrences.append(
                TimelineOccurrence(
                    timeline_start=timeline_time,
                    timeline_end=timeline_time,
                    includes_start=True,
                    includes_end=True,
                    segment_indices=(index,),
                )
            )
        return _merge_occurrences(occurrences)

    source_occurrences = source_to_timeline_occurrences


def _merge_occurrences(
    occurrences: Sequence[TimelineOccurrence],
) -> tuple[TimelineOccurrence, ...]:
    """Merge duplicate boundary points and adjacent freezes deterministically."""

    if not occurrences:
        return ()
    ordered = sorted(
        occurrences,
        key=lambda item: (item.timeline_start, item.timeline_end),
    )
    merged: list[TimelineOccurrence] = [ordered[0]]
    for occurrence in ordered[1:]:
        current = merged[-1]
        if occurrence.timeline_start > current.timeline_end:
            merged.append(occurrence)
            continue
        end = max(current.timeline_end, occurrence.timeline_end)
        if occurrence.timeline_end > current.timeline_end:
            includes_end = occurrence.includes_end
        elif occurrence.timeline_end < current.timeline_end:
            includes_end = current.includes_end
        else:
            includes_end = current.includes_end or occurrence.includes_end
        merged[-1] = TimelineOccurrence(
            timeline_start=current.timeline_start,
            timeline_end=end,
            includes_start=current.includes_start,
            includes_end=includes_end,
            segment_indices=tuple(
                sorted(set(current.segment_indices + occurrence.segment_indices))
            ),
        )
    return tuple(merged)


__all__ = [
    "ParsedTimeMapPoint",
    "RetimeError",
    "RetimeMap",
    "RetimeBoundary",
    "RetimePoint",
    "RetimeSample",
    "RetimeSegment",
    "RetimeValidationError",
    "SegmentKind",
    "TimelineOccurrence",
    "TimelineOutsideRetimeMapError",
    "UnsupportedRetimeMappingError",
]
