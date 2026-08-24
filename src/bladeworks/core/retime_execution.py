"""Stock-FFmpeg execution plans for exact piecewise FCPXML retiming.

Architecture map
================

``RetimeMap`` (source/render IR)
    Exact rational timeline-to-source segments from :mod:`retime`.

``build_retime_execution_plan``
    Freezes each segment into typed video and audio operations.  All timing
    remains ``Fraction`` here; no average clip speed is calculated.

``build_video_filtergraph`` / ``build_audio_filtergraph``
    Render the typed operations into stock FFmpeg filters.  Rational seconds
    become decimal strings only at this final text boundary.

``probe_stock_ffmpeg_capabilities``
    Reports whether one ordinary FFmpeg executable contains every filter used
    by a particular plan.  It does not install software or select fallbacks.

Important invariants
--------------------

* A source window is owned by exactly one segment at a timeline boundary.
* Every segment resets timestamps and is trimmed to its exact output duration
  before concat.  Concat therefore never guesses duration from neighboring
  clips.
* Reverse video trims a high-exclusive source window before reversal, so the
  authored turn frame is repeated and the nonexistent high endpoint is never
  selected.
* Each segment resolves Final Cut's default floor sampling before concat; the
  final output-rate normalization therefore cannot change source ownership.
* Audio tempo factors are exact ``Fraction`` values in FFmpeg's inclusive
  0.5--2.0 range.  No segment rate is converted to one average clip speed.
* Final Cut freeze-audio behavior is not assumed.  It is blocked by default;
  calibrated silence must carry a registry-owned calibration identifier.

Why this exists
---------------

The parser/compiler can preserve a correct ``RetimeMap`` while an FFmpeg
builder still destroys it by using one endpoint-average speed.  This module is
the isolated execution boundary: integration can consume its typed operations
without changing the existing renderer until the contract is validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import Enum
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Literal

from .retime import RetimeMap, RetimeSegment


SegmentOperation = Literal["forward", "reverse", "freeze"]
AudioSegmentOperation = Literal["media", "silence", "blocked"]
VideoSamplingDirection = Literal["forward", "reverse"]


class RetimeExecutionError(ValueError):
    """Base error for an invalid or unavailable retime execution plan."""


class RetimeExecutionValidationError(RetimeExecutionError):
    """A caller supplied an invalid exact execution option."""


class AudioFreezeBehaviorBlocked(RetimeExecutionError):
    """Audio execution reached a freeze whose Final Cut behavior is unknown."""


class MissingFFmpegRetimeCapability(RetimeExecutionError):
    """The selected FFmpeg executable lacks a required stock filter."""


@dataclass(frozen=True)
class VideoFrameOwnership:
    """One exact source frame and the edit's phase inside its playback interval.

    Architecture map
    ================

    exact source time + frame grid -> owning integer source frame
                                   -> aligned frame start
                                   -> playback-direction phase

    ``source_phase`` always measures forward in playback order.  For forward
    sampling it is the distance from the frame's low time edge.  For reverse
    sampling it is the distance from the frame's high time edge.  Keeping the
    phase separate from the aligned frame boundary lets a decoder seek to a
    real frame while a later filter preserves a non-frame-aligned edit point.

    Main callers:
    - CPU and Vulkan source schedulers in the next integration phase.
    - Retime segment planning once aligned decode windows consume this type.

    Why this exists:
    FFmpeg ``trim=start=<seconds>`` selects by decoded timestamps. Final Cut's
    default sampling selects the frame whose interval owns that time. Those
    rules differ on fine decoder time bases, so an exact source time alone is
    not a complete decoder contract.
    """

    source_time: Fraction
    frame_duration: Fraction
    frame_grid_origin: Fraction
    direction: VideoSamplingDirection
    source_start_frame: int
    source_frame_start: Fraction
    source_phase: Fraction

    def __post_init__(self) -> None:
        source_time = _require_fraction(self.source_time, field_name="source_time")
        frame_duration = _require_fraction(
            self.frame_duration, field_name="frame_duration"
        )
        frame_grid_origin = _require_fraction(
            self.frame_grid_origin, field_name="frame_grid_origin"
        )
        source_frame_start = _require_fraction(
            self.source_frame_start, field_name="source_frame_start"
        )
        source_phase = _require_fraction(self.source_phase, field_name="source_phase")
        if frame_duration <= 0:
            raise RetimeExecutionValidationError("frame_duration must be positive")
        if self.direction not in {"forward", "reverse"}:
            raise RetimeExecutionValidationError(
                "video sampling direction must be 'forward' or 'reverse'"
            )
        if isinstance(self.source_start_frame, bool) or not isinstance(
            self.source_start_frame, int
        ):
            raise RetimeExecutionValidationError(
                "source_start_frame must be an integer"
            )
        expected_frame_start = (
            frame_grid_origin + self.source_start_frame * frame_duration
        )
        if source_frame_start != expected_frame_start:
            raise RetimeExecutionValidationError(
                "source_frame_start must equal frame_grid_origin + "
                "source_start_frame * frame_duration"
            )
        if not 0 <= source_phase < frame_duration:
            raise RetimeExecutionValidationError(
                "source_phase must lie in [0, frame_duration)"
            )
        expected_source_time = (
            source_frame_start + source_phase
            if self.direction == "forward"
            else source_frame_start + frame_duration - source_phase
        )
        if source_time != expected_source_time:
            raise RetimeExecutionValidationError(
                "source_time does not match its frame boundary and playback phase"
            )
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "frame_duration", frame_duration)
        object.__setattr__(self, "frame_grid_origin", frame_grid_origin)
        object.__setattr__(self, "source_frame_start", source_frame_start)
        object.__setattr__(self, "source_phase", source_phase)


@dataclass(frozen=True)
class VideoDecodeWindow:
    """Frame-aligned decoder coverage for one exact semantic source interval.

    The semantic interval remains unchanged. Only the decoder coverage expands
    to whole source frames. ``decode_end`` is exclusive, including when the
    semantic high edge lands exactly on a frame boundary.

    Main callers:
    - Retime execution planning in phase 2.
    - Ordinary CPU/Vulkan source scheduling in phase 2.
    """

    semantic_start: Fraction
    semantic_end: Fraction
    frame_duration: Fraction
    frame_grid_origin: Fraction
    first_frame: int
    end_frame: int
    decode_start: Fraction
    decode_end: Fraction

    def __post_init__(self) -> None:
        values = {
            name: _require_fraction(getattr(self, name), field_name=name)
            for name in (
                "semantic_start",
                "semantic_end",
                "frame_duration",
                "frame_grid_origin",
                "decode_start",
                "decode_end",
            )
        }
        if values["semantic_end"] <= values["semantic_start"]:
            raise RetimeExecutionValidationError(
                "semantic source window end must be greater than its start"
            )
        if values["frame_duration"] <= 0:
            raise RetimeExecutionValidationError("frame_duration must be positive")
        for name in ("first_frame", "end_frame"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise RetimeExecutionValidationError(f"{name} must be an integer")
        if self.end_frame <= self.first_frame:
            raise RetimeExecutionValidationError(
                "decode end_frame must be greater than first_frame"
            )
        expected_start = (
            values["frame_grid_origin"]
            + self.first_frame * values["frame_duration"]
        )
        expected_end = (
            values["frame_grid_origin"]
            + self.end_frame * values["frame_duration"]
        )
        if values["decode_start"] != expected_start or values["decode_end"] != expected_end:
            raise RetimeExecutionValidationError(
                "decode window boundaries must match their exact source frame indices"
            )
        if values["decode_start"] > values["semantic_start"]:
            raise RetimeExecutionValidationError(
                "decode window must include the semantic start"
            )
        if values["decode_end"] < values["semantic_end"]:
            raise RetimeExecutionValidationError(
                "decode window must include the semantic end"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class OwnedFrameWindow:
    """One authored canvas interval resolved to half-open output frames.

    Final Cut evaluates video at frame midpoints.  A frame belongs to an
    authored half-open interval when its midpoint lies inside that interval.
    This matters for odd-length transitions centered on a cut: their authored
    edges lie half a frame off the canvas grid, while their owned output is an
    exact integer frame range.

    Main callers:
    - CPU transition-expanded source scheduling and transition composition.
    - Vulkan source/interval scheduling for the same semantic document.

    Why this exists:
    Decoder ownership and canvas ownership solve different problems.  Source
    times retain their exact fine phase through ``VideoFrameOwnership``;
    canvas intervals use this midpoint rule without rounding authored timing.
    """

    semantic_start: Fraction
    semantic_end: Fraction
    frame_duration: Fraction
    frame_grid_origin: Fraction
    first_frame: int
    end_frame: int
    start: Fraction
    end: Fraction

    def __post_init__(self) -> None:
        values = {
            name: _require_fraction(getattr(self, name), field_name=name)
            for name in (
                "semantic_start",
                "semantic_end",
                "frame_duration",
                "frame_grid_origin",
                "start",
                "end",
            )
        }
        if values["semantic_end"] <= values["semantic_start"]:
            raise RetimeExecutionValidationError(
                "semantic canvas window end must be greater than its start"
            )
        if values["frame_duration"] <= 0:
            raise RetimeExecutionValidationError("frame_duration must be positive")
        for name in ("first_frame", "end_frame"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise RetimeExecutionValidationError(f"{name} must be an integer")
        if self.end_frame <= self.first_frame:
            raise RetimeExecutionValidationError(
                "semantic canvas window owns no output frames"
            )
        expected_start = (
            values["frame_grid_origin"]
            + self.first_frame * values["frame_duration"]
        )
        expected_end = (
            values["frame_grid_origin"]
            + self.end_frame * values["frame_duration"]
        )
        if values["start"] != expected_start or values["end"] != expected_end:
            raise RetimeExecutionValidationError(
                "owned window boundaries must match their exact frame indices"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def duration(self) -> Fraction:
        return self.end - self.start

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.first_frame


def resolve_owned_frame_window(
    semantic_start: Fraction,
    semantic_end: Fraction,
    *,
    frame_duration: Fraction,
    frame_grid_origin: Fraction = Fraction(0),
) -> OwnedFrameWindow:
    """Resolve exact midpoint ownership without changing authored timing."""

    start = _require_fraction(semantic_start, field_name="semantic_start")
    end = _require_fraction(semantic_end, field_name="semantic_end")
    duration = _require_fraction(frame_duration, field_name="frame_duration")
    origin = _require_fraction(frame_grid_origin, field_name="frame_grid_origin")
    if end <= start:
        raise RetimeExecutionValidationError(
            "semantic canvas window end must be greater than its start"
        )
    if duration <= 0:
        raise RetimeExecutionValidationError("frame_duration must be positive")
    half_frame = Fraction(1, 2)
    first_frame = _ceil_fraction((start - origin) / duration - half_frame)
    end_frame = _ceil_fraction((end - origin) / duration - half_frame)
    return OwnedFrameWindow(
        semantic_start=start,
        semantic_end=end,
        frame_duration=duration,
        frame_grid_origin=origin,
        first_frame=first_frame,
        end_frame=end_frame,
        start=origin + first_frame * duration,
        end=origin + end_frame * duration,
    )


def resolve_video_frame_ownership(
    source_time: Fraction,
    *,
    frame_duration: Fraction,
    frame_grid_origin: Fraction = Fraction(0),
    direction: VideoSamplingDirection = "forward",
) -> VideoFrameOwnership:
    """Resolve Final Cut's directional owner without decoder-timebase rounding.

    Forward playback uses ordinary floor ownership. Reverse playback uses the
    left limit at an exact high boundary, preserving the established repeated
    turn-frame behavior of forward-to-reverse maps.

    Main callers:
    - Phase-2 CPU and Vulkan decoder scheduling.
    - ``align_video_decode_window`` callers that also need the initial phase.
    """

    exact_time = _require_fraction(source_time, field_name="source_time")
    duration = _require_fraction(frame_duration, field_name="frame_duration")
    origin = _require_fraction(frame_grid_origin, field_name="frame_grid_origin")
    if duration <= 0:
        raise RetimeExecutionValidationError("frame_duration must be positive")
    if direction not in {"forward", "reverse"}:
        raise RetimeExecutionValidationError(
            "video sampling direction must be 'forward' or 'reverse'"
        )
    coordinate = (exact_time - origin) / duration
    if direction == "forward":
        frame_index = _floor_fraction(coordinate)
        frame_start = origin + frame_index * duration
        phase = exact_time - frame_start
    else:
        frame_index = _ceil_fraction(coordinate) - 1
        frame_start = origin + frame_index * duration
        phase = frame_start + duration - exact_time
    return VideoFrameOwnership(
        source_time=exact_time,
        frame_duration=duration,
        frame_grid_origin=origin,
        direction=direction,
        source_start_frame=frame_index,
        source_frame_start=frame_start,
        source_phase=phase,
    )


def align_video_decode_window(
    semantic_start: Fraction,
    semantic_end: Fraction,
    *,
    frame_duration: Fraction,
    frame_grid_origin: Fraction = Fraction(0),
) -> VideoDecodeWindow:
    """Expand one semantic interval to the smallest containing frame window."""

    start = _require_fraction(semantic_start, field_name="semantic_start")
    end = _require_fraction(semantic_end, field_name="semantic_end")
    duration = _require_fraction(frame_duration, field_name="frame_duration")
    origin = _require_fraction(frame_grid_origin, field_name="frame_grid_origin")
    if end <= start:
        raise RetimeExecutionValidationError(
            "semantic source window end must be greater than its start"
        )
    if duration <= 0:
        raise RetimeExecutionValidationError("frame_duration must be positive")
    first_frame = _floor_fraction((start - origin) / duration)
    end_frame = _ceil_fraction((end - origin) / duration)
    return VideoDecodeWindow(
        semantic_start=start,
        semantic_end=end,
        frame_duration=duration,
        frame_grid_origin=origin,
        first_frame=first_frame,
        end_frame=end_frame,
        decode_start=origin + first_frame * duration,
        decode_end=origin + end_frame * duration,
    )


class AudioPitchMode(str, Enum):
    """How speed changes affect audio pitch."""

    PRESERVE = "preserve"
    CHANGE_WITH_SPEED = "change_with_speed"


class AudioFreezeMode(str, Enum):
    """The only freeze-audio states admitted by this execution contract."""

    BLOCKED = "blocked"
    CALIBRATED_SILENCE = "calibrated_silence"


@dataclass(frozen=True)
class AudioFreezePolicy:
    """Explicit Final Cut freeze-audio evidence used by the planner.

    ``BLOCKED`` is the safe default.  ``CALIBRATED_SILENCE`` is deliberately
    impossible without a stable calibration identifier, so a caller cannot
    enable an unmeasured behavior with a boolean flag.
    """

    mode: AudioFreezeMode = AudioFreezeMode.BLOCKED
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, AudioFreezeMode):
            raise RetimeExecutionValidationError(
                "audio freeze mode must be an AudioFreezeMode"
            )
        if self.mode is AudioFreezeMode.BLOCKED:
            if self.calibration_id is not None:
                raise RetimeExecutionValidationError(
                    "blocked freeze audio cannot carry a calibration identifier"
                )
            return
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise RetimeExecutionValidationError(
                "calibrated freeze silence requires a non-empty calibration identifier"
            )

    @classmethod
    def blocked(cls) -> "AudioFreezePolicy":
        return cls()

    @classmethod
    def calibrated_silence(cls, calibration_id: str) -> "AudioFreezePolicy":
        return cls(
            mode=AudioFreezeMode.CALIBRATED_SILENCE,
            calibration_id=calibration_id,
        )


@dataclass(frozen=True)
class SourceWindow:
    """One exact, forward-oriented source trim window.

    ``start`` is inclusive and ``end`` is exclusive. Reverse video uses that
    same exclusive high edge before reversal. Including the authored high
    endpoint advances the reversed sequence by one source frame.
    """

    start: Fraction
    end: Fraction

    def __post_init__(self) -> None:
        start = _require_fraction(self.start, field_name="source window start")
        end = _require_fraction(self.end, field_name="source window end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if end <= start:
            raise RetimeExecutionValidationError(
                "source window end must be greater than its start"
            )


@dataclass(frozen=True)
class VideoRetimeSegmentPlan:
    """One independently executable stock-FFmpeg video retime segment."""

    index: int
    operation: SegmentOperation
    timeline_start: Fraction
    timeline_end: Fraction
    source_start: Fraction
    source_end: Fraction
    source_window: SourceWindow
    frame_ownership: VideoFrameOwnership
    decode_window: VideoDecodeWindow
    playback_rate: Fraction

    @property
    def output_duration(self) -> Fraction:
        return self.timeline_end - self.timeline_start


@dataclass(frozen=True)
class AudioRetimeSegmentPlan:
    """One independently executable stock-FFmpeg audio retime segment."""

    index: int
    operation: AudioSegmentOperation
    timeline_start: Fraction
    timeline_end: Fraction
    source_start: Fraction
    source_end: Fraction
    source_window: SourceWindow | None
    playback_rate: Fraction
    pitch_mode: AudioPitchMode
    freeze_calibration_id: str | None = None

    @property
    def output_duration(self) -> Fraction:
        return self.timeline_end - self.timeline_start


@dataclass(frozen=True)
class RetimeExecutionPlan:
    """Complete typed video/audio plan for one exact ``RetimeMap``.

    Main callers:
    - The future central FFmpeg builder will construct this once per media
      render item and splice the returned filtergraphs into its graph.
    - Experimental tests serialize ``manifest`` to prove deterministic plans.
    """

    retime_map: RetimeMap
    video_frame_duration: Fraction
    audio_sample_rate: int
    audio_channel_layout: str
    pitch_mode: AudioPitchMode
    freeze_audio_policy: AudioFreezePolicy
    video_segments: tuple[VideoRetimeSegmentPlan, ...]
    audio_segments: tuple[AudioRetimeSegmentPlan, ...]

    @property
    def output_duration(self) -> Fraction:
        return self.retime_map.timeline_duration

    @property
    def audio_blocked(self) -> bool:
        return any(segment.operation == "blocked" for segment in self.audio_segments)

    @property
    def audio_blockers(self) -> tuple[int, ...]:
        return tuple(
            segment.index
            for segment in self.audio_segments
            if segment.operation == "blocked"
        )

    def manifest(self) -> dict[str, Any]:
        """Return a stable JSON-compatible plan with rational values as text."""

        return {
            "schema": "fcpxml_retime_execution.v1",
            "timeline": {
                "start": _fraction_text(self.retime_map.timeline_start),
                "end": _fraction_text(self.retime_map.timeline_end),
                "duration": _fraction_text(self.output_duration),
            },
            "video_frame_duration": _fraction_text(self.video_frame_duration),
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channel_layout": self.audio_channel_layout,
            "pitch_mode": self.pitch_mode.value,
            "freeze_audio": {
                "mode": self.freeze_audio_policy.mode.value,
                "calibration_id": self.freeze_audio_policy.calibration_id,
            },
            "video_segments": [
                {
                    "index": segment.index,
                    "operation": segment.operation,
                    "timeline_start": _fraction_text(segment.timeline_start),
                    "timeline_end": _fraction_text(segment.timeline_end),
                    "output_duration": _fraction_text(segment.output_duration),
                    "source_start": _fraction_text(segment.source_start),
                    "source_end": _fraction_text(segment.source_end),
                    "trim_start": _fraction_text(segment.source_window.start),
                    "trim_end": _fraction_text(segment.source_window.end),
                    "decode_start": _fraction_text(segment.decode_window.decode_start),
                    "decode_end": _fraction_text(segment.decode_window.decode_end),
                    "source_start_frame": segment.frame_ownership.source_start_frame,
                    "source_phase": _fraction_text(segment.frame_ownership.source_phase),
                    "sampling_direction": segment.frame_ownership.direction,
                    "playback_rate": _fraction_text(segment.playback_rate),
                }
                for segment in self.video_segments
            ],
            "audio_segments": [
                {
                    "index": segment.index,
                    "operation": segment.operation,
                    "timeline_start": _fraction_text(segment.timeline_start),
                    "timeline_end": _fraction_text(segment.timeline_end),
                    "output_duration": _fraction_text(segment.output_duration),
                    "source_start": _fraction_text(segment.source_start),
                    "source_end": _fraction_text(segment.source_end),
                    "trim_start": (
                        _fraction_text(segment.source_window.start)
                        if segment.source_window is not None
                        else None
                    ),
                    "trim_end": (
                        _fraction_text(segment.source_window.end)
                        if segment.source_window is not None
                        else None
                    ),
                    "playback_rate": _fraction_text(segment.playback_rate),
                    "pitch_mode": segment.pitch_mode.value,
                    "freeze_calibration_id": segment.freeze_calibration_id,
                }
                for segment in self.audio_segments
            ],
        }

    @property
    def manifest_sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FFmpegRetimeCapabilityReport:
    """Read-only result of checking one FFmpeg executable."""

    executable: str
    version_line: str
    required_filters: tuple[str, ...]
    available_filters: tuple[str, ...]
    missing_filters: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.missing_filters

    def require_supported(self) -> None:
        if self.missing_filters:
            raise MissingFFmpegRetimeCapability(
                "FFmpeg is missing retime filters: " + ", ".join(self.missing_filters)
            )


def build_retime_execution_plan(
    retime_map: RetimeMap,
    *,
    video_frame_duration: Fraction,
    include_audio: bool = True,
    audio_sample_rate: int = 48_000,
    audio_channel_layout: str = "stereo",
    preserve_audio_pitch: bool = True,
    freeze_audio_policy: AudioFreezePolicy | None = None,
) -> RetimeExecutionPlan:
    """Translate one exact map into independent segment operations.

    Main callers:
    - A future central compiler/FFmpeg integration point after ``RenderClip``
      has an exact ``retime_map``.

    Why this exists:
    - This is the only place that decides source-window ownership.  Filter
      rendering below cannot reinterpret rates or average adjacent segments.
    """

    if not isinstance(retime_map, RetimeMap):
        raise RetimeExecutionValidationError("retime_map must be a RetimeMap")
    frame_duration = _require_fraction(
        video_frame_duration, field_name="video_frame_duration"
    )
    if frame_duration <= 0:
        raise RetimeExecutionValidationError(
            "video_frame_duration must be positive"
        )
    if isinstance(audio_sample_rate, bool) or not isinstance(audio_sample_rate, int):
        raise RetimeExecutionValidationError("audio_sample_rate must be an integer")
    if audio_sample_rate <= 0:
        raise RetimeExecutionValidationError("audio_sample_rate must be positive")
    _validate_channel_layout(audio_channel_layout)
    freeze_policy = freeze_audio_policy or AudioFreezePolicy.blocked()
    if not isinstance(freeze_policy, AudioFreezePolicy):
        raise RetimeExecutionValidationError(
            "freeze_audio_policy must be an AudioFreezePolicy"
        )
    pitch_mode = (
        AudioPitchMode.PRESERVE
        if preserve_audio_pitch
        else AudioPitchMode.CHANGE_WITH_SPEED
    )

    video_segments = tuple(
        _plan_video_segment(
            index,
            segment,
            frame_duration,
            previous_segment=(
                retime_map.segments[index - 1]
                if index > 0
                else None
            ),
        )
        for index, segment in enumerate(retime_map.segments)
    )
    audio_segments: tuple[AudioRetimeSegmentPlan, ...] = ()
    if include_audio:
        audio_segments = tuple(
            _plan_audio_segment(index, segment, pitch_mode, freeze_policy)
            for index, segment in enumerate(retime_map.segments)
        )

    return RetimeExecutionPlan(
        retime_map=retime_map,
        video_frame_duration=frame_duration,
        audio_sample_rate=audio_sample_rate,
        audio_channel_layout=audio_channel_layout,
        pitch_mode=pitch_mode,
        freeze_audio_policy=freeze_policy,
        video_segments=video_segments,
        audio_segments=audio_segments,
    )


def bounded_atempo_factors(rate: Fraction) -> tuple[Fraction, ...]:
    """Factor a positive exact tempo into FFmpeg's inclusive 0.5--2 range.

    The product of returned factors is exactly ``rate``.  A 1x rate returns an
    empty tuple because no filter is needed.
    """

    remaining = _require_fraction(rate, field_name="audio tempo rate")
    if remaining <= 0:
        raise RetimeExecutionValidationError("audio tempo rate must be positive")
    factors: list[Fraction] = []
    while remaining > 2:
        factors.append(Fraction(2))
        remaining /= 2
    while remaining < Fraction(1, 2):
        factors.append(Fraction(1, 2))
        remaining /= Fraction(1, 2)
    if remaining != 1:
        factors.append(remaining)
    return tuple(factors)


def build_video_filtergraph(
    plan: RetimeExecutionPlan,
    *,
    input_label: str = "0:v:0",
    output_label: str = "retimed_video",
    label_prefix: str = "rtv",
) -> str:
    """Render the video segment plan into one stock-FFmpeg filtergraph."""

    _validate_plan(plan)
    source_label = _validate_filter_label(input_label, field_name="input_label")
    final_label = _validate_filter_label(output_label, field_name="output_label")
    prefix = _validate_filter_label(label_prefix, field_name="label_prefix")
    segment_inputs, graph_parts = _split_input_labels(
        source_label, len(plan.video_segments), prefix=prefix, audio=False
    )
    outputs: list[str] = []
    for segment, segment_input in zip(plan.video_segments, segment_inputs):
        output = f"{prefix}_segment_{segment.index}"
        filters = _video_segment_filters(segment, plan.video_frame_duration)
        graph_parts.append(
            f"[{segment_input}]" + ",".join(filters) + f"[{output}]"
        )
        outputs.append(output)
    _append_concat(
        graph_parts,
        outputs,
        output_label=final_label,
        video=True,
    )
    return ";".join(graph_parts)


def build_frame_owned_video_filters(
    *,
    ownership: VideoFrameOwnership,
    decode_window: VideoDecodeWindow,
    playback_rate: Fraction,
    output_frame_duration: Fraction,
    output_duration: Fraction,
    reverse: bool = False,
    floor_sampling_shift: bool = True,
) -> tuple[str, ...]:
    """Emit one phase-preserving source sampling chain.

    Main callers:
    - Ordinary CPU source emission after its decoder seek is fixed.
    - Ordinary Vulkan source emission while reusing that same decoder.
    - Piecewise retime segment emission.

    Why this exists:
    Seeking must happen at a real source-frame boundary, while Final Cut edit
    points may carry a finer timebase. The aligned trim supplies whole frames;
    the negative phase preserves the fine edit point. Piecewise retime callers
    additionally request the renderer's established almost-one-frame shift,
    which resolves floor ownership before a later canvas-rate normalization.
    Consequently pre-seeked and unseeked decoders produce the same sequence.
    """

    if not isinstance(ownership, VideoFrameOwnership):
        raise RetimeExecutionValidationError(
            "ownership must be a VideoFrameOwnership"
        )
    if not isinstance(decode_window, VideoDecodeWindow):
        raise RetimeExecutionValidationError(
            "decode_window must be a VideoDecodeWindow"
        )
    rate = _require_fraction(playback_rate, field_name="playback_rate")
    target_frame_duration = _require_fraction(
        output_frame_duration, field_name="output_frame_duration"
    )
    duration = _require_fraction(output_duration, field_name="output_duration")
    if rate <= 0:
        raise RetimeExecutionValidationError("playback_rate must be positive")
    if target_frame_duration <= 0:
        raise RetimeExecutionValidationError(
            "output_frame_duration must be positive"
        )
    if duration <= 0:
        raise RetimeExecutionValidationError("output_duration must be positive")
    if ownership.frame_duration != decode_window.frame_duration:
        raise RetimeExecutionValidationError(
            "ownership and decode window must use the same source frame duration"
        )
    if ownership.frame_grid_origin != decode_window.frame_grid_origin:
        raise RetimeExecutionValidationError(
            "ownership and decode window must use the same source frame grid"
        )
    if not decode_window.first_frame <= ownership.source_start_frame < decode_window.end_frame:
        raise RetimeExecutionValidationError(
            "owning source frame must lie inside the aligned decode window"
        )
    expected_direction = "reverse" if reverse else "forward"
    if ownership.direction != expected_direction:
        raise RetimeExecutionValidationError(
            f"{expected_direction} emission needs {expected_direction} frame ownership"
        )

    # Preserve the renderer's calibrated floor-selection offset. Without it,
    # FFmpeg advances a fine-timebase start to the next decoded frame; the
    # explicit negative phase alone does not change that selection rule.
    use_floor_sampling_shift = floor_sampling_shift
    floor_sample_shift = Fraction(0)
    if use_floor_sampling_shift:
        floor_sample_shift = ownership.frame_duration - Fraction(1, 1_000_000)
        if floor_sample_shift <= 0:
            raise RetimeExecutionValidationError(
                "source frame duration is too small for floor-sampling precision"
            )
    filters = [
        "trim="
        f"start={_number_text(decode_window.decode_start)}:"
        f"end={_number_text(decode_window.decode_end)}"
    ]
    if reverse:
        filters.append("reverse")
    filters.extend(
        (
            "setpts=(PTS-STARTPTS-"
            f"({_number_text(ownership.source_phase)})/TB)*"
            f"{rate.denominator}/{rate.numerator}"
            + (
                f"+({_number_text(floor_sample_shift)})/TB"
                if use_floor_sampling_shift
                else ""
            ),
            f"fps={_fraction_text(1 / target_frame_duration)}:"
            "round=up"
            + ("" if use_floor_sampling_shift else ":start_time=0"),
            f"trim=duration={_duration_text(duration)}",
            "setpts=PTS-STARTPTS",
        )
    )
    return tuple(filters)


# One structured audio filterchain: ``(in_labels, filters, out_labels)``.  This
# is the pad-labelled node/edge form the PyAV audio port (``tensor/audio_pyav``)
# consumes -- PyAV 16 has no ``filter_complex`` string parser, so the retime
# sub-graph must expose its chains structured rather than as one joined string.
AudioFilterTriple = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


def _serialize_audio_triple(triple: AudioFilterTriple) -> str:
    """Render one ``(in_labels, filters, out_labels)`` chain to stock text.

    Byte-for-byte identical to the strings the previous string builder emitted:
    bracketed inputs, comma-joined filters, bracketed outputs.
    """

    in_labels, filters, out_labels = triple
    return (
        "".join(f"[{label}]" for label in in_labels)
        + ",".join(filters)
        + "".join(f"[{label}]" for label in out_labels)
    )


def _split_input_label_triples(
    input_label: str,
    count: int,
    *,
    prefix: str,
) -> tuple[tuple[str, ...], list[AudioFilterTriple]]:
    """Structured form of ``_split_input_labels`` for the audio path (asplit)."""

    if count <= 0:
        return (), []
    if count == 1:
        return (input_label,), []
    labels = tuple(f"{prefix}_input_{index}" for index in range(count))
    return labels, [((input_label,), (f"asplit={count}",), labels)]


def _concat_triple(
    outputs: list[str],
    *,
    output_label: str,
    video: bool,
) -> AudioFilterTriple:
    """Structured form of ``_append_concat`` (returns one chain triple)."""

    if not outputs:
        raise RetimeExecutionValidationError("retime plan has no segments")
    if len(outputs) == 1:
        media_kind = "null" if video else "anull"
        return ((outputs[0],), (media_kind,), (output_label,))
    concat_options = f"n={len(outputs)}:v={1 if video else 0}:a={0 if video else 1}"
    reset_filter = "setpts=PTS-STARTPTS" if video else "asetpts=PTS-STARTPTS"
    return (
        tuple(outputs),
        (f"concat={concat_options}", reset_filter),
        (output_label,),
    )


def _audio_filtergraph_triples(
    plan: RetimeExecutionPlan,
    *,
    input_label: str,
    output_label: str,
    label_prefix: str,
) -> list[AudioFilterTriple]:
    """Build the retime audio graph as an ordered list of structured chains.

    This is the single source of truth for the audio retime graph.
    ``build_audio_filtergraph`` serialises these triples back to the calibrated
    ``filter_complex`` string (byte-for-byte); ``build_audio_filtergraph_segments``
    returns them for the node-by-node PyAV builder.
    """

    _validate_plan(plan)
    if not plan.audio_segments:
        raise RetimeExecutionValidationError("execution plan does not include audio")
    if plan.audio_blocked:
        indices = ", ".join(str(index) for index in plan.audio_blockers)
        raise AudioFreezeBehaviorBlocked(
            "freeze audio is not calibrated for segment(s): " + indices
        )
    source_label = _validate_filter_label(input_label, field_name="input_label")
    final_label = _validate_filter_label(output_label, field_name="output_label")
    prefix = _validate_filter_label(label_prefix, field_name="label_prefix")

    media_count = sum(
        segment.operation == "media" for segment in plan.audio_segments
    )
    media_inputs, triples = _split_input_label_triples(
        source_label, media_count, prefix=f"{prefix}_media"
    )
    media_input_index = 0
    outputs: list[str] = []
    for segment in plan.audio_segments:
        output = f"{prefix}_segment_{segment.index}"
        if segment.operation == "silence":
            filters = _silence_audio_filters(plan, segment)
            triples.append(((), tuple(filters), (output,)))
        else:
            segment_input = media_inputs[media_input_index]
            media_input_index += 1
            filters = _media_audio_filters(plan, segment)
            triples.append(((segment_input,), tuple(filters), (output,)))
        outputs.append(output)
    triples.append(_concat_triple(outputs, output_label=final_label, video=False))
    return triples


def build_audio_filtergraph(
    plan: RetimeExecutionPlan,
    *,
    input_label: str = "0:a:0",
    output_label: str = "retimed_audio",
    label_prefix: str = "rta",
) -> str:
    """Render the audio segment plan or fail on uncalibrated freeze audio."""

    triples = _audio_filtergraph_triples(
        plan,
        input_label=input_label,
        output_label=output_label,
        label_prefix=label_prefix,
    )
    return ";".join(_serialize_audio_triple(triple) for triple in triples)


def build_audio_filtergraph_segments(
    plan: RetimeExecutionPlan,
    *,
    input_label: str = "0:a:0",
    output_label: str = "retimed_audio",
    label_prefix: str = "rta",
) -> tuple[AudioFilterTriple, ...]:
    """Structured node/edge form of ``build_audio_filtergraph``.

    Each entry is ``(in_labels, filters, out_labels)``.  Serialising every entry
    with ``_serialize_audio_triple`` and joining on ``;`` reproduces
    ``build_audio_filtergraph`` byte-for-byte -- both derive from
    ``_audio_filtergraph_triples``.

    Main callers:
    - ``core.audio_execution`` (splices these into the delivery graph instead of
      re-splitting the joined string).
    """

    return tuple(
        _audio_filtergraph_triples(
            plan,
            input_label=input_label,
            output_label=output_label,
            label_prefix=label_prefix,
        )
    )


def required_stock_filters(plan: RetimeExecutionPlan) -> tuple[str, ...]:
    """Return the deterministic stock-filter set needed by this plan."""

    _validate_plan(plan)
    filters = {"trim", "setpts", "fps"}
    if len(plan.video_segments) > 1:
        filters.update(("split", "concat"))
    if any(segment.operation == "reverse" for segment in plan.video_segments):
        filters.add("reverse")
    if any(segment.operation == "freeze" for segment in plan.video_segments):
        filters.add("tpad")
    if plan.audio_segments:
        filters.update(("atrim", "asetpts"))
        media_count = sum(
            segment.operation == "media" for segment in plan.audio_segments
        )
        if media_count > 1:
            filters.add("asplit")
        if len(plan.audio_segments) > 1:
            filters.add("concat")
        if any(
            segment.operation == "media" and segment.playback_rate < 0
            for segment in plan.audio_segments
        ):
            filters.add("areverse")
        if any(
            segment.operation == "media"
            and abs(segment.playback_rate) != 1
            and segment.pitch_mode is AudioPitchMode.PRESERVE
            for segment in plan.audio_segments
        ):
            filters.add("atempo")
        if any(
            segment.operation == "media"
            and abs(segment.playback_rate) != 1
            and segment.pitch_mode is AudioPitchMode.CHANGE_WITH_SPEED
            for segment in plan.audio_segments
        ):
            filters.update(("asetrate", "aresample"))
        if any(segment.operation == "silence" for segment in plan.audio_segments):
            filters.add("anullsrc")
    return tuple(sorted(filters))


def probe_stock_ffmpeg_capabilities(
    plan: RetimeExecutionPlan,
    *,
    ffmpeg_path: str | Path = "ffmpeg",
) -> FFmpegRetimeCapabilityReport:
    """Inspect one installed FFmpeg without modifying the environment.

    Main callers:
    - Experimental validation and the future renderer preflight.

    The probe checks named stock filters only.  It does not treat a different
    filter as a fallback for a missing operation.
    """

    _validate_plan(plan)
    executable = str(ffmpeg_path)
    try:
        version = subprocess.run(
            [executable, "-hide_banner", "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        filters = subprocess.run(
            [executable, "-hide_banner", "-filters"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise MissingFFmpegRetimeCapability(
            f"could not inspect FFmpeg executable {executable!r}: {error}"
        ) from error
    version_line = version.stdout.splitlines()[0] if version.stdout else "unknown"
    available = _parse_ffmpeg_filter_names(filters.stdout)
    required = required_stock_filters(plan)
    missing = tuple(name for name in required if name not in available)
    return FFmpegRetimeCapabilityReport(
        executable=executable,
        version_line=version_line,
        required_filters=required,
        available_filters=tuple(sorted(available)),
        missing_filters=missing,
    )


def _plan_video_segment(
    index: int,
    segment: RetimeSegment,
    frame_duration: Fraction,
    *,
    previous_segment: RetimeSegment | None,
) -> VideoRetimeSegmentPlan:
    rate = segment.rate
    if segment.kind == "forward":
        window = SourceWindow(segment.source_start, segment.source_end)
        ownership = resolve_video_frame_ownership(
            segment.source_start,
            frame_duration=frame_duration,
            direction="forward",
        )
    elif segment.kind == "reverse":
        window = SourceWindow(segment.source_end, segment.source_start)
        ownership = resolve_video_frame_ownership(
            segment.source_start,
            frame_duration=frame_duration,
            direction="reverse",
        )
    else:
        # Final Cut treats the source value where a forward segment becomes a
        # freeze as the exclusive right edge of the preceding source frame.
        # Holding ``source_start`` itself advances the freeze by one frame even
        # though the surrounding forward and reverse segments remain correct.
        # A leading freeze has no incoming edge, and a reverse-to-freeze edge
        # approaches the value from the other direction, so those retain the
        # authored source value.
        freeze_source = segment.source_start
        if (
            previous_segment is not None
            and previous_segment.kind == "forward"
            and previous_segment.source_end == segment.source_start
        ):
            freeze_source -= frame_duration
        window = SourceWindow(
            freeze_source,
            freeze_source + frame_duration,
        )
        ownership = resolve_video_frame_ownership(
            freeze_source,
            frame_duration=frame_duration,
            direction="forward",
        )
    decode_window = align_video_decode_window(
        window.start,
        window.end,
        frame_duration=frame_duration,
    )
    return VideoRetimeSegmentPlan(
        index=index,
        operation=segment.kind,
        timeline_start=segment.timeline_start,
        timeline_end=segment.timeline_end,
        source_start=segment.source_start,
        source_end=segment.source_end,
        source_window=window,
        frame_ownership=ownership,
        decode_window=decode_window,
        playback_rate=rate,
    )


def _plan_audio_segment(
    index: int,
    segment: RetimeSegment,
    pitch_mode: AudioPitchMode,
    freeze_policy: AudioFreezePolicy,
) -> AudioRetimeSegmentPlan:
    if segment.kind == "freeze":
        operation: AudioSegmentOperation
        if freeze_policy.mode is AudioFreezeMode.BLOCKED:
            operation = "blocked"
        else:
            operation = "silence"
        return AudioRetimeSegmentPlan(
            index=index,
            operation=operation,
            timeline_start=segment.timeline_start,
            timeline_end=segment.timeline_end,
            source_start=segment.source_start,
            source_end=segment.source_end,
            source_window=None,
            playback_rate=segment.rate,
            pitch_mode=pitch_mode,
            freeze_calibration_id=freeze_policy.calibration_id,
        )
    low = min(segment.source_start, segment.source_end)
    high = max(segment.source_start, segment.source_end)
    return AudioRetimeSegmentPlan(
        index=index,
        operation="media",
        timeline_start=segment.timeline_start,
        timeline_end=segment.timeline_end,
        source_start=segment.source_start,
        source_end=segment.source_end,
        source_window=SourceWindow(low, high),
        playback_rate=segment.rate,
        pitch_mode=pitch_mode,
    )


def _video_segment_filters(
    segment: VideoRetimeSegmentPlan,
    frame_duration: Fraction,
) -> tuple[str, ...]:
    if segment.operation != "freeze":
        return build_frame_owned_video_filters(
            ownership=segment.frame_ownership,
            decode_window=segment.decode_window,
            playback_rate=abs(segment.playback_rate),
            output_frame_duration=frame_duration,
            output_duration=segment.output_duration,
            reverse=segment.operation == "reverse",
            floor_sampling_shift=True,
        )

    window = segment.decode_window
    filters = [
        "trim="
        f"start={_number_text(window.decode_start)}:end={_number_text(window.decode_end)}"
    ]
    filters.extend(
        (
            "setpts=PTS-STARTPTS",
            "trim=end_frame=1",
            "setpts=PTS-STARTPTS",
            "tpad=stop_mode=clone:"
            f"stop_duration={_duration_text(segment.output_duration)}",
            f"fps={_fraction_text(1 / frame_duration)}:round=up:start_time=0",
        )
    )
    filters.extend(
        (
            f"trim=duration={_duration_text(segment.output_duration)}",
            "setpts=PTS-STARTPTS",
        )
    )
    return tuple(filters)


def _media_audio_filters(
    plan: RetimeExecutionPlan,
    segment: AudioRetimeSegmentPlan,
) -> tuple[str, ...]:
    if segment.source_window is None:
        raise RetimeExecutionValidationError("media audio segment needs a source window")
    filters = [
        "atrim="
        f"start={_number_text(segment.source_window.start)}:"
        f"end={_number_text(segment.source_window.end)}"
    ]
    if segment.playback_rate < 0:
        filters.append("areverse")
    filters.append("asetpts=PTS-STARTPTS")
    absolute_rate = abs(segment.playback_rate)
    if absolute_rate != 1:
        if segment.pitch_mode is AudioPitchMode.PRESERVE:
            filters.extend(
                f"atempo={_number_text(factor)}"
                for factor in bounded_atempo_factors(absolute_rate)
            )
        else:
            filters.extend(
                (
                    "asetrate="
                    f"{plan.audio_sample_rate}*"
                    f"{absolute_rate.numerator}/{absolute_rate.denominator}",
                    f"aresample={plan.audio_sample_rate}",
                )
            )
    filters.extend(
        (
            f"atrim=duration={_duration_text(segment.output_duration)}",
            "asetpts=PTS-STARTPTS",
        )
    )
    return tuple(filters)


def _silence_audio_filters(
    plan: RetimeExecutionPlan,
    segment: AudioRetimeSegmentPlan,
) -> tuple[str, ...]:
    if not segment.freeze_calibration_id:
        raise RetimeExecutionValidationError(
            "silence freeze segment has no calibration identifier"
        )
    return (
        "anullsrc="
        f"r={plan.audio_sample_rate}:cl={plan.audio_channel_layout}",
        f"atrim=duration={_duration_text(segment.output_duration)}",
        "asetpts=PTS-STARTPTS",
    )


def _split_input_labels(
    input_label: str,
    count: int,
    *,
    prefix: str,
    audio: bool,
) -> tuple[tuple[str, ...], list[str]]:
    if count <= 0:
        return (), []
    if count == 1:
        return (input_label,), []
    labels = tuple(f"{prefix}_input_{index}" for index in range(count))
    filter_name = "asplit" if audio else "split"
    graph = (
        f"[{input_label}]{filter_name}={count}"
        + "".join(f"[{label}]" for label in labels)
    )
    return labels, [graph]


def _append_concat(
    graph_parts: list[str],
    outputs: list[str],
    *,
    output_label: str,
    video: bool,
) -> None:
    if not outputs:
        raise RetimeExecutionValidationError("retime plan has no segments")
    if len(outputs) == 1:
        media_kind = "null" if video else "anull"
        graph_parts.append(f"[{outputs[0]}]{media_kind}[{output_label}]")
        return
    concat_options = f"n={len(outputs)}:v={1 if video else 0}:a={0 if video else 1}"
    reset_filter = "setpts=PTS-STARTPTS" if video else "asetpts=PTS-STARTPTS"
    graph_parts.append(
        "".join(f"[{label}]" for label in outputs)
        + f"concat={concat_options},{reset_filter}[{output_label}]"
    )


def _floor_fraction(value: Fraction) -> int:
    """Return mathematical floor without converting an exact ratio to float."""

    return value.numerator // value.denominator


def _ceil_fraction(value: Fraction) -> int:
    """Return mathematical ceiling without converting an exact ratio to float."""

    return -((-value.numerator) // value.denominator)


def _require_fraction(value: object, *, field_name: str) -> Fraction:
    if isinstance(value, bool):
        raise RetimeExecutionValidationError(
            f"{field_name} must be an exact Fraction, not bool"
        )
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    raise RetimeExecutionValidationError(
        f"{field_name} must be an exact Fraction, got {type(value).__name__}"
    )


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _number_text(value: Fraction) -> str:
    """Render an exact ratio only when FFmpeg needs a numeric option value."""

    if value.denominator == 1:
        return str(value.numerator)
    with localcontext() as context:
        context.prec = 24
        rendered = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return rendered.rstrip("0").rstrip(".")


def _duration_text(value: Fraction) -> str:
    if value <= 0:
        raise RetimeExecutionValidationError("FFmpeg duration must be positive")
    return _number_text(value)


_FILTER_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")
_CHANNEL_LAYOUT_PATTERN = re.compile(r"^[A-Za-z0-9_.]+$")


def _validate_filter_label(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not _FILTER_LABEL_PATTERN.fullmatch(value):
        raise RetimeExecutionValidationError(
            f"{field_name} must contain only FFmpeg label characters"
        )
    return value


def _validate_channel_layout(value: str) -> None:
    if not isinstance(value, str) or not _CHANNEL_LAYOUT_PATTERN.fullmatch(value):
        raise RetimeExecutionValidationError(
            "audio_channel_layout must be a plain FFmpeg channel-layout name"
        )


def _validate_plan(plan: RetimeExecutionPlan) -> None:
    if not isinstance(plan, RetimeExecutionPlan):
        raise RetimeExecutionValidationError("plan must be a RetimeExecutionPlan")


def _parse_ffmpeg_filter_names(output: str) -> set[str]:
    names: set[str] = set()
    pattern = re.compile(r"^\s*[.TSCAX|]{2,6}\s+([A-Za-z0-9_]+)\s+", re.MULTILINE)
    for match in pattern.finditer(output):
        names.add(match.group(1))
    return names


__all__ = [
    "AudioFreezeBehaviorBlocked",
    "AudioFreezeMode",
    "AudioFreezePolicy",
    "AudioPitchMode",
    "AudioRetimeSegmentPlan",
    "FFmpegRetimeCapabilityReport",
    "MissingFFmpegRetimeCapability",
    "OwnedFrameWindow",
    "RetimeExecutionError",
    "RetimeExecutionPlan",
    "RetimeExecutionValidationError",
    "SourceWindow",
    "VideoDecodeWindow",
    "VideoFrameOwnership",
    "VideoRetimeSegmentPlan",
    "align_video_decode_window",
    "bounded_atempo_factors",
    "build_audio_filtergraph",
    "build_audio_filtergraph_segments",
    "build_frame_owned_video_filters",
    "build_retime_execution_plan",
    "build_video_filtergraph",
    "probe_stock_ffmpeg_capabilities",
    "resolve_video_frame_ownership",
    "resolve_owned_frame_window",
    "required_stock_filters",
]
