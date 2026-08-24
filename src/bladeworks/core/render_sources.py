"""Resolve file, compound, and multicam media behind one A/V source contract.

Architecture map
================

``StoryNode`` timeline instance
    -> ``RenderableAVSource`` describes the referenced media/subgraph
    -> ``resolve_instance_stream_timing`` composes the instance's exact
       ``timeMap`` with its video or independent audio source range
    -> video/audio executors consume the same source and timing records

The source is deliberately unaware of its timeline use. A file is decoded, a
compound is recursively composed, and a multicam resolves its selected angles;
all three expose source-time-indexed video/audio. Trims, split edits, retiming,
effects, and placement belong to the ordinary clip instance above that source.

Important invariants
--------------------

* Source and timeline coordinates remain exact ``Fraction`` values.
* ``audioStart``/``audioDuration`` select a source-domain audio interval. For
  retimed clips that interval is intersected with the authored piecewise map;
  no independent clock is invented and no extrapolation is permitted.
* A stream timing plan is contiguous. A source window whose repeated
  occurrences create disjoint output islands is rejected until the instance
  executor can represent explicit silence gaps.
* This module emits no FFmpeg strings and creates no intermediate movies.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, Optional

from .model import AssetResource, SequenceFormatContext, StoryNode
from .retime import (
    RetimeMap,
    RetimeSegment,
    RetimeValidationError,
    UnsupportedRetimeMappingError,
)

if TYPE_CHECKING:
    from .story_ir import ResourceStory


SourceKind = Literal["file", "compound", "multicam"]
StreamKind = Literal["video", "audio"]


class RenderSourceError(ValueError):
    """Base class for an invalid source definition or clip-instance edit."""


class StreamTimingCoverageError(RenderSourceError):
    """Raised when an authored map cannot cover a requested stream range."""


def source_bound_tolerance(
    *,
    has_video: bool,
    frame_duration: Optional[Fraction],
    has_audio: bool,
    audio_rate: Optional[int],
) -> Fraction:
    """Return the smallest declared media unit for source-bound comparisons.

    Main callers:
    - The compiler's file and virtual video source checks.
    - The audio IR's file and virtual audio source checks.

    Why this exists:
    Final Cut sometimes serializes an asset duration on the audio sample clock
    and its clip duration on a different project timescale. The two rational
    endpoints can then differ by less than one real sample or frame. Accepting
    at most the smallest declared media unit handles that representation
    rounding while still rejecting a request for another sample or frame.
    """

    units: list[Fraction] = []
    if has_video and frame_duration is not None and frame_duration > 0:
        units.append(frame_duration)
    if has_audio and audio_rate is not None and audio_rate > 0:
        units.append(Fraction(1, audio_rate))
    return min(units, default=Fraction(0))


def source_range_within_bounds(
    *,
    requested_start: Fraction,
    requested_end: Fraction,
    source_start: Fraction,
    source_end: Fraction,
    tolerance: Fraction,
) -> bool:
    """Check one exact range, permitting only declared-unit rounding drift."""

    if requested_end < requested_start:
        return False
    if tolerance < 0:
        raise RenderSourceError("source-bound tolerance cannot be negative")
    starts_in_bounds = requested_start >= source_start or (
        tolerance > 0 and source_start - requested_start < tolerance
    )
    ends_in_bounds = requested_end <= source_end or (
        tolerance > 0 and requested_end - source_end < tolerance
    )
    return starts_in_bounds and ends_in_bounds


@dataclass(frozen=True)
class SourceWindow:
    """The smallest half-open source interval needed to execute one map."""

    start: Fraction
    end: Fraction

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise RenderSourceError("source window must have positive duration")

    @property
    def duration(self) -> Fraction:
        return self.end - self.start


@dataclass(frozen=True)
class RenderableAVSource:
    """One source-time-indexed A/V graph consumed by ordinary clip instances.

    ``video_story`` and ``audio_story`` contain the selected internal story for
    virtual media. File sources leave them empty and identify their asset with
    ``resource_id``. Multicam scope nodes preserve per-angle controls without
    turning the timeline mc-clip into a synthetic container.
    """

    id: str
    kind: SourceKind
    resource_id: str
    source_start: Fraction
    duration: Optional[Fraction]
    format_context: Optional[SequenceFormatContext]
    has_video: bool
    has_audio: bool
    video_story: tuple[StoryNode, ...] = ()
    audio_story: tuple[StoryNode, ...] = ()
    video_scope: Optional[StoryNode] = None
    audio_scope: Optional[StoryNode] = None
    video_angle_id: Optional[str] = None
    audio_angle_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.id:
            raise RenderSourceError("renderable source requires a stable id")
        if self.duration is not None and self.duration <= 0:
            raise RenderSourceError(f"source {self.id!r} duration must be positive")

    @property
    def end(self) -> Optional[Fraction]:
        if self.duration is None:
            return None
        return self.source_start + self.duration


@dataclass(frozen=True)
class InstanceStreamTiming:
    """One stream's exact source/output mapping for a clip instance."""

    stream: StreamKind
    absolute_start: Fraction
    duration: Fraction
    source_start: Fraction
    source_duration: Fraction
    retime_map: RetimeMap

    @property
    def absolute_end(self) -> Fraction:
        return self.absolute_start + self.duration

    @property
    def source_end(self) -> Fraction:
        return self.source_start + self.source_duration


@dataclass(frozen=True)
class RenderableAVInstance:
    """A normal timeline use of any file-backed or virtual A/V source."""

    path: str
    source: RenderableAVSource
    video: Optional[InstanceStreamTiming]
    audio: Optional[InstanceStreamTiming]


def resolve_file_source(
    asset: AssetResource,
    *,
    format_context: Optional[SequenceFormatContext],
) -> RenderableAVSource:
    """Expose a natural media asset through the same source contract."""

    return RenderableAVSource(
        id=f"file:{asset.id}",
        kind="file",
        resource_id=asset.id,
        source_start=asset.start,
        duration=asset.duration if asset.duration and asset.duration > 0 else None,
        format_context=format_context,
        has_video=asset.has_video,
        has_audio=asset.has_audio,
    )


def resolve_compound_source(
    resource: "ResourceStory",
    *,
    has_video: bool,
    has_audio: bool,
) -> RenderableAVSource:
    """Expose a reusable sequence without copying any outer clip edit into it."""

    return RenderableAVSource(
        id=f"compound:{resource.resource_id}",
        kind="compound",
        resource_id=resource.resource_id,
        source_start=resource.start,
        duration=resource.duration,
        format_context=resource.format_context,
        has_video=has_video,
        has_audio=has_audio,
        video_story=resource.story,
        audio_story=resource.story,
    )


def source_window_for_retime(
    retime_map: RetimeMap,
    *,
    frame_duration: Fraction,
) -> SourceWindow:
    """Find the source interval a file or composed source must expose.

    A pure freeze has equal source endpoints but still needs one decodable
    frame. Expanding that point by exactly one source frame keeps this rule
    identical for files, compounds, and multicam sources.
    """

    if frame_duration <= 0:
        raise RenderSourceError("source frame duration must be positive")
    coordinates = tuple(
        coordinate
        for segment in retime_map.segments
        for coordinate in (segment.source_start, segment.source_end)
    )
    start = min(coordinates)
    end = max(coordinates)
    if end == start:
        end += frame_duration
    return SourceWindow(start=start, end=end)


def rebase_source_retime(
    retime_map: RetimeMap,
    *,
    source_origin: Fraction,
    stream_start: Fraction,
) -> RetimeMap:
    """Map source coordinates onto one resolved source subgraph's timestamps."""

    return RetimeMap(
        tuple(
            RetimeSegment(
                timeline_start=segment.timeline_start,
                timeline_end=segment.timeline_end,
                source_start=stream_start + segment.source_start - source_origin,
                source_end=stream_start + segment.source_end - source_origin,
            )
            for segment in retime_map.segments
        )
    )


def resolve_instance_stream_timing(
    node: StoryNode,
    *,
    absolute_start: Fraction,
    stream: StreamKind,
) -> InstanceStreamTiming:
    """Compose one clip instance's stream range with its authored time map.

    Main callers:
    - The video compiler for a resolved file/compound/multicam source.
    - The audio IR builder before it schedules a completed source submix.

    Why this exists:
    ``audioStart`` is not a second retime engine. It selects a different source
    interval on the same clip instance. Intersecting that range with the
    instance map gives one executable mapping and naturally reduces to the
    ordinary J/L formula when playback is 1x.
    """

    if stream == "audio":
        requested_start = (
            node.audio_start if node.audio_start is not None else node.start
        )
        requested_duration = (
            node.audio_duration if node.audio_duration is not None else node.duration
        )
    else:
        requested_start = node.start
        requested_duration = node.duration
    if requested_duration <= 0:
        raise StreamTimingCoverageError(
            f"{node.path} {stream} duration must be positive"
        )

    if not node.time_map:
        relative_start = requested_start - node.start
        return InstanceStreamTiming(
            stream=stream,
            absolute_start=absolute_start + relative_start,
            duration=requested_duration,
            source_start=requested_start,
            source_duration=requested_duration,
            retime_map=RetimeMap.identity(
                requested_duration,
                source_start=requested_start,
            ),
        )

    try:
        authored = RetimeMap.from_time_map_points(
            node.time_map,
            linearize_two_point_smooth2=True,
        )
    except UnsupportedRetimeMappingError as error:
        raise StreamTimingCoverageError(
            f"{node.path} has an unsupported nonlinear timeMap for {stream}: {error}"
        ) from error
    except RetimeValidationError as error:
        raise StreamTimingCoverageError(
            f"{node.path} has an invalid {stream} timeMap: {error}"
        ) from error

    if stream == "video" or (
        stream == "audio"
        and node.audio_start is None
        and node.audio_duration is None
    ):
        try:
            visible = authored.restrict_to_timeline_window(
                node.start,
                node.duration,
            )
        except RetimeValidationError as error:
            raise StreamTimingCoverageError(
                f"{node.path} video timeMap does not cover its visible duration: {error}"
            ) from error
        return _timing_from_segments(
            stream,
            absolute_start,
            visible.segments,
        )

    requested_end = requested_start + requested_duration
    segments = _segments_inside_source_window(
        authored,
        source_start=requested_start,
        source_end=requested_end,
    )
    if not segments:
        raise StreamTimingCoverageError(
            f"{node.path} {stream} source range [{requested_start}, {requested_end}) "
            "does not occur in its authored timeMap"
        )
    covered_low = min(
        min(segment.source_start, segment.source_end) for segment in segments
    )
    covered_high = max(
        max(segment.source_start, segment.source_end) for segment in segments
    )
    if covered_low > requested_start or covered_high < requested_end:
        raise StreamTimingCoverageError(
            f"{node.path} {stream} source range [{requested_start}, {requested_end}) "
            f"is not fully covered by timeMap source range [{covered_low}, {covered_high}]"
        )
    return _timing_from_segments(stream, absolute_start, segments)


def _segments_inside_source_window(
    retime_map: RetimeMap,
    *,
    source_start: Fraction,
    source_end: Fraction,
) -> tuple[RetimeSegment, ...]:
    """Clip exact map segments to a half-open source interval."""

    clipped: list[RetimeSegment] = []
    for segment in retime_map.segments:
        if segment.rate == 0:
            if source_start <= segment.source_start < source_end:
                clipped.append(segment)
            continue
        low = min(segment.source_start, segment.source_end)
        high = max(segment.source_start, segment.source_end)
        overlap_low = max(low, source_start)
        overlap_high = min(high, source_end)
        if overlap_high <= overlap_low:
            continue
        if segment.rate > 0:
            clipped_source_start = overlap_low
            clipped_source_end = overlap_high
        else:
            clipped_source_start = overlap_high
            clipped_source_end = overlap_low
        timeline_start = segment.timeline_start + (
            clipped_source_start - segment.source_start
        ) / segment.rate
        timeline_end = segment.timeline_start + (
            clipped_source_end - segment.source_start
        ) / segment.rate
        clipped.append(
            RetimeSegment(
                timeline_start=timeline_start,
                timeline_end=timeline_end,
                source_start=clipped_source_start,
                source_end=clipped_source_end,
            )
        )
    return tuple(sorted(clipped, key=lambda item: item.timeline_start))


def _timing_from_segments(
    stream: StreamKind,
    clip_absolute_start: Fraction,
    segments: tuple[RetimeSegment, ...],
) -> InstanceStreamTiming:
    """Rebase a contiguous authored slice to one stream-local output clock."""

    first = segments[0]
    last = segments[-1]
    for left, right in zip(segments, segments[1:]):
        if left.timeline_end != right.timeline_start:
            raise StreamTimingCoverageError(
                f"{stream} source selection produces disjoint output intervals "
                f"[{left.timeline_end}, {right.timeline_start}); explicit gap "
                "composition is required"
            )
    output_origin = first.timeline_start
    source_low = min(
        min(segment.source_start, segment.source_end) for segment in segments
    )
    source_high = max(
        max(segment.source_start, segment.source_end) for segment in segments
    )
    rebased = RetimeMap(
        tuple(
            RetimeSegment(
                timeline_start=segment.timeline_start - output_origin,
                timeline_end=segment.timeline_end - output_origin,
                source_start=segment.source_start,
                source_end=segment.source_end,
            )
            for segment in segments
        )
    )
    return InstanceStreamTiming(
        stream=stream,
        absolute_start=clip_absolute_start + output_origin,
        duration=last.timeline_end - output_origin,
        source_start=source_low,
        source_duration=source_high - source_low,
        retime_map=rebased,
    )
