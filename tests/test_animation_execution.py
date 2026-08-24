"""Execution-contract tests for the isolated animation-to-FFmpeg bridge."""

from __future__ import annotations

from fractions import Fraction
import shutil
import struct
import subprocess

import pytest

from bladeworks.core.animation import (
    AnimatedScalar,
    AnimatedVec2,
    ScalarControlPoint,
    TimelineAnimatedScalar,
    TimelineAnimatedVec2,
    Vec2ControlPoint,
)
from bladeworks.core.animation_execution import (
    AnimationClock,
    AnimationClockError,
    AnimationExecutionError,
    AnimationExpressionLimitError,
    AnimationExpressionLimits,
    compile_scalar_expression,
    compile_vec2_expressions,
    sample_scalar_frames,
    sample_vec2_frames,
)
from bladeworks.core.retime import RetimeMap, RetimeSegment


FRAME_RATES = (
    Fraction(24),
    Fraction(30_000, 1_001),
    Fraction(30),
    Fraction(60_000, 1_001),
)


def _piecewise_retime() -> RetimeMap:
    """Forward, freeze, reverse, then repeat the same source range."""

    return RetimeMap(
        (
            RetimeSegment(Fraction(0), Fraction(1), Fraction(0), Fraction(1)),
            RetimeSegment(Fraction(1), Fraction(2), Fraction(1), Fraction(1)),
            RetimeSegment(Fraction(2), Fraction(3), Fraction(1), Fraction(0)),
            RetimeSegment(Fraction(3), Fraction(4), Fraction(0), Fraction(1)),
        )
    )


def _scalar_track() -> TimelineAnimatedScalar:
    source = AnimatedScalar(
        (
            ScalarControlPoint(Fraction(0), -0.4, curve="smooth"),
            ScalarControlPoint(
                Fraction(1, 4),
                0.6,
                interpolation="ease-in",
                curve="smooth",
            ),
            ScalarControlPoint(
                Fraction(1, 2),
                -0.2,
                interpolation="ease-out",
                curve="linear",
            ),
            ScalarControlPoint(
                Fraction(3, 4),
                0.7,
                interpolation="ease",
                curve="smooth",
            ),
            ScalarControlPoint(
                Fraction(1),
                0.1,
                interpolation="linear",
                curve="smooth",
            ),
        )
    )
    return TimelineAnimatedScalar(source, _piecewise_retime())


def _vec2_track() -> TimelineAnimatedVec2:
    source = AnimatedVec2(
        (
            Vec2ControlPoint(Fraction(0), (-10.0, 4.0), curve="smooth"),
            Vec2ControlPoint(
                Fraction(1, 2),
                (12.0, -8.0),
                interpolation="ease",
                curve="smooth",
            ),
            Vec2ControlPoint(
                Fraction(1),
                (2.0, 20.0),
                interpolation="ease-out",
                curve="linear",
            ),
        )
    )
    return TimelineAnimatedVec2(source, _piecewise_retime())


def _render_ffmpeg_expression(
    expression: str,
    *,
    sample_rate: int,
    duration: Fraction,
) -> tuple[float, ...]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is unavailable")
    # The small margin guarantees the final requested sample exists despite
    # duration parsing at FFmpeg's floating-point command-line boundary.
    extended_duration = float(duration + Fraction(2, sample_rate))
    command = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"aevalsrc=exprs='{expression}':s={sample_rate}:d={extended_duration:.17g}",
        "-ac",
        "1",
        "-c:a",
        "pcm_f64le",
        "-f",
        "f64le",
        "-",
    )
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
    )
    if completed.returncode != 0:
        pytest.fail(
            "FFmpeg rejected the generated animation expression:\n"
            + completed.stderr.decode("utf-8", errors="replace")
        )
    assert len(completed.stdout) % 8 == 0
    return struct.unpack(f"<{len(completed.stdout) // 8}d", completed.stdout)


@pytest.mark.parametrize("frame_rate", FRAME_RATES)
def test_real_ffmpeg_matches_python_kernel_at_every_output_frame(
    frame_rate: Fraction,
) -> None:
    track = _scalar_track()
    clock = AnimationClock(
        origin="clip_local",
        pre_roll=Fraction(1, 4),
        post_roll=Fraction(1, 4),
    )
    plan = compile_scalar_expression(track, clock=clock)
    frame_count = int((plan.input_window_end - plan.input_window_start) * frame_rate)
    commands = sample_scalar_frames(
        plan,
        frame_rate=frame_rate,
        frame_count=frame_count,
    )

    # For N/D fps, an audio sample rate of N places each frame exactly D
    # samples apart: frame_time = frame_index * D / N.
    samples = _render_ffmpeg_expression(
        plan.expression,
        sample_rate=frame_rate.numerator,
        duration=plan.input_window_end,
    )
    for command in commands:
        sample_index = command.frame_index * frame_rate.denominator
        assert samples[sample_index] == pytest.approx(command.value, abs=2e-10)


@pytest.mark.parametrize("frame_rate", FRAME_RATES)
def test_vec2_frame_plan_uses_exact_fraction_times_and_kernel_values(
    frame_rate: Fraction,
) -> None:
    track = _vec2_track()
    plan = compile_vec2_expressions(
        track,
        clock=AnimationClock(pre_roll=Fraction(1, 3), post_roll=Fraction(1, 5)),
        time_variable="T",
    )
    commands = sample_vec2_frames(plan, frame_rate=frame_rate, frame_count=180)

    assert commands[0].input_time == Fraction(0)
    assert commands[0].clip_time == Fraction(0)
    for command in commands:
        assert isinstance(command.input_time, Fraction)
        assert isinstance(command.clip_time, Fraction)
        assert isinstance(command.track_time, Fraction)
        assert command.input_time == Fraction(command.frame_index, 1) / frame_rate
        assert command.value == pytest.approx(track.value_at(command.track_time))
    assert plan.x_expression != plan.y_expression
    assert "T" in plan.x_expression


def test_local_and_absolute_clocks_are_equivalent_and_hold_endpoints() -> None:
    track = _scalar_track()
    local = compile_scalar_expression(
        track,
        clock=AnimationClock(
            pre_roll=Fraction(1, 2),
            post_roll=Fraction(1, 3),
        ),
    )
    absolute = compile_scalar_expression(
        track,
        clock=AnimationClock(
            origin="absolute",
            absolute_clip_start=Fraction(100, 3),
            pre_roll=Fraction(1, 2),
            post_roll=Fraction(1, 3),
        ),
    )

    assert local.input_window_start == 0
    assert local.input_window_end == Fraction(29, 6)
    assert absolute.input_window_start == Fraction(197, 6)
    assert absolute.input_window_end == Fraction(113, 3)
    for clip_time in (
        Fraction(-5),
        Fraction(0),
        Fraction(1, 8),
        Fraction(1),
        Fraction(5, 2),
        Fraction(4),
        Fraction(20),
    ):
        local_input = local.clock.pre_roll + clip_time
        absolute_input = Fraction(100, 3) + clip_time
        assert local.value_at_input_time(local_input) == pytest.approx(
            absolute.value_at_input_time(absolute_input)
        )

    assert local.value_at_input_time(Fraction(-10)) == pytest.approx(
        track.value_at(track.retime_map.timeline_start)
    )
    assert local.value_at_input_time(Fraction(100)) == pytest.approx(
        track.value_at(track.retime_map.timeline_end)
    )


def test_one_clock_state_drives_handles_reverse_freeze_and_repeated_source() -> None:
    mapping = _piecewise_retime()
    local = AnimationClock(pre_roll=Fraction(1, 2), post_roll=Fraction(1, 4))
    absolute = AnimationClock(
        origin="absolute",
        absolute_clip_start=Fraction(20),
        pre_roll=Fraction(1, 2),
        post_roll=Fraction(1, 4),
    )

    checks = (
        (Fraction(0), "transition_pre_roll", Fraction(0), Fraction(0), "forward"),
        (Fraction(1, 2), "clip", Fraction(0), Fraction(0), "forward"),
        (Fraction(3, 2), "clip", Fraction(1), Fraction(1), "freeze"),
        (Fraction(5, 2), "clip", Fraction(2), Fraction(1), "reverse"),
        (Fraction(7, 2), "clip", Fraction(3), Fraction(0), "forward"),
        (Fraction(9, 2), "clip", Fraction(4), Fraction(1), "forward"),
        (Fraction(19, 4), "transition_post_roll", Fraction(4), Fraction(1), "forward"),
    )
    for input_time, phase, clip_time, source_time, kind in checks:
        state = local.state_at(input_time, mapping)
        assert state.phase == phase
        assert state.clip_time == clip_time
        assert state.source_time == source_time
        assert state.retime_segment_kind == kind
        assert state.is_endpoint_hold is (phase != "clip")

        absolute_input = Fraction(20) + (input_time - local.pre_roll)
        absolute_state = absolute.state_at(absolute_input, mapping)
        assert absolute_state.clip_time == state.clip_time
        assert absolute_state.track_time == state.track_time
        assert absolute_state.source_time == state.source_time
        assert absolute_state.phase == state.phase


def test_expression_is_deterministic_rational_and_keeps_each_retime_segment() -> None:
    track = _scalar_track()
    first = compile_scalar_expression(track)
    second = compile_scalar_expression(track)

    assert first.expression == second.expression
    assert "(1/4)" in first.expression
    assert "*-1" in first.expression
    assert "*0" in first.expression
    # The repeated final forward segment is present independently rather than
    # replacing the complete map with one endpoint-average speed.
    assert first.expression.count("*1)") >= 2


def test_single_point_tracks_compile_to_constants() -> None:
    scalar = TimelineAnimatedScalar(
        AnimatedScalar((ScalarControlPoint(Fraction(7), 0.25),)),
        RetimeMap.identity(Fraction(2), source_start=Fraction(7)),
    )
    vector = TimelineAnimatedVec2(
        AnimatedVec2((Vec2ControlPoint(Fraction(7), (3.0, -4.0)),)),
        RetimeMap.identity(Fraction(2), source_start=Fraction(7)),
    )

    assert compile_scalar_expression(scalar).expression == "0.25"
    vec_plan = compile_vec2_expressions(vector)
    assert vec_plan.x_expression == "3"
    assert vec_plan.y_expression == "-4"


def test_configured_limits_fail_explicitly() -> None:
    track = _scalar_track()

    with pytest.raises(AnimationExpressionLimitError, match="control points"):
        compile_scalar_expression(
            track,
            limits=AnimationExpressionLimits(max_control_points=4),
        )
    with pytest.raises(AnimationExpressionLimitError, match="retime segments"):
        compile_scalar_expression(
            track,
            limits=AnimationExpressionLimits(max_retime_segments=3),
        )
    with pytest.raises(AnimationExpressionLimitError, match="complexity"):
        compile_scalar_expression(
            track,
            limits=AnimationExpressionLimits(max_segment_point_product=15),
        )
    with pytest.raises(AnimationExpressionLimitError, match="characters"):
        compile_scalar_expression(
            track,
            limits=AnimationExpressionLimits(max_expression_chars=100),
        )


def test_invalid_clocks_variables_and_frame_requests_fail_closed() -> None:
    track = _scalar_track()

    with pytest.raises(AnimationClockError, match="absolute_clip_start"):
        AnimationClock(origin="absolute")
    with pytest.raises(AnimationClockError, match="must not provide"):
        AnimationClock(absolute_clip_start=Fraction(1))
    with pytest.raises(AnimationClockError, match="non-negative"):
        AnimationClock(pre_roll=Fraction(-1))
    with pytest.raises(AnimationClockError, match="exact Fraction, not float"):
        AnimationClock(post_roll=0.1)  # type: ignore[arg-type]
    with pytest.raises(AnimationExecutionError, match="'t' or 'T'"):
        compile_scalar_expression(track, time_variable="pts")
    with pytest.raises(AnimationExecutionError, match=r"\(on-1\)\*N/D"):
        compile_scalar_expression(track, time_variable="on*1/30")

    perspective = compile_scalar_expression(
        track,
        time_variable="(on-1)*1/30",
    )
    assert perspective.time_variable == "((on-1)*1/30)"
    assert "(on-1)*1/30" in perspective.expression

    plan = compile_scalar_expression(track)
    with pytest.raises(AnimationClockError, match="frame_rate must be positive"):
        sample_scalar_frames(plan, frame_rate=Fraction(0), frame_count=1)
    with pytest.raises(AnimationExpressionLimitError, match="positive integer"):
        sample_scalar_frames(plan, frame_rate=Fraction(24), frame_count=0)
    with pytest.raises(AnimationClockError, match="first_frame must be non-negative"):
        sample_scalar_frames(
            plan,
            frame_rate=Fraction(24),
            frame_count=1,
            first_frame=-1,
        )
