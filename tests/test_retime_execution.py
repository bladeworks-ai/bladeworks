"""Experimental validation for the exact stock-FFmpeg retime planner."""

from __future__ import annotations

from array import array
from fractions import Fraction
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

from bladeworks.core.retime import (
    RetimeMap,
    RetimePoint,
    UnsupportedRetimeMappingError,
)
from bladeworks.core.retime_execution import (
    AudioFreezeBehaviorBlocked,
    AudioFreezeMode,
    AudioFreezePolicy,
    AudioPitchMode,
    RetimeExecutionValidationError,
    bounded_atempo_factors,
    build_audio_filtergraph,
    build_retime_execution_plan,
    build_video_filtergraph,
    probe_stock_ffmpeg_capabilities,
    required_stock_filters,
)


def _mixed_map() -> RetimeMap:
    return RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(1), Fraction(2)),
            RetimePoint(Fraction(2), Fraction(2)),
            RetimePoint(Fraction(3), Fraction(1)),
        )
    )


def _forward_reverse_map() -> RetimeMap:
    return RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(1), Fraction(1)),
            RetimePoint(Fraction(2), Fraction(0)),
        )
    )


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("stock FFmpeg is not installed")
    return executable


def _run_ffmpeg(arguments: list[str]) -> bytes:
    completed = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", *arguments],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.stderr == b""
    return completed.stdout


def test_plan_preserves_exact_piecewise_operations_and_windows() -> None:
    plan = build_retime_execution_plan(
        _mixed_map(),
        video_frame_duration=Fraction(1, 30),
    )

    assert [segment.operation for segment in plan.video_segments] == [
        "forward",
        "freeze",
        "reverse",
    ]
    assert [segment.playback_rate for segment in plan.video_segments] == [
        Fraction(2),
        Fraction(0),
        Fraction(-1),
    ]
    assert [segment.output_duration for segment in plan.video_segments] == [
        Fraction(1),
        Fraction(1),
        Fraction(1),
    ]
    assert (
        plan.video_segments[0].source_window.start,
        plan.video_segments[0].source_window.end,
    ) == (Fraction(0), Fraction(2))
    assert (
        plan.video_segments[1].source_window.start,
        plan.video_segments[1].source_window.end,
    ) == (Fraction(59, 30), Fraction(2))
    assert (
        plan.video_segments[2].source_window.start,
        plan.video_segments[2].source_window.end,
    ) == (Fraction(1), Fraction(2))
    assert [segment.operation for segment in plan.audio_segments] == [
        "media",
        "blocked",
        "media",
    ]
    assert plan.audio_blockers == (1,)


def test_nonlinear_and_inexact_inputs_fail_instead_of_being_averaged() -> None:
    with pytest.raises(UnsupportedRetimeMappingError, match="nonlinear"):
        RetimeMap.from_points(
            (
                RetimePoint(Fraction(0), Fraction(0), "smooth"),
                RetimePoint(Fraction(1), Fraction(2), "linear"),
            )
        )

    with pytest.raises(RetimeExecutionValidationError, match="exact Fraction"):
        build_retime_execution_plan(  # type: ignore[arg-type]
            RetimeMap.identity(Fraction(1)),
            video_frame_duration=0.04,
        )
    with pytest.raises(RetimeExecutionValidationError, match="must be a RetimeMap"):
        build_retime_execution_plan(  # type: ignore[arg-type]
            object(),
            video_frame_duration=Fraction(1, 25),
        )


def test_freeze_audio_is_blocked_until_named_calibration() -> None:
    blocked = build_retime_execution_plan(
        _mixed_map(),
        video_frame_duration=Fraction(1, 30),
    )
    with pytest.raises(AudioFreezeBehaviorBlocked, match=r"segment\(s\): 1"):
        build_audio_filtergraph(blocked)

    with pytest.raises(RetimeExecutionValidationError, match="identifier"):
        AudioFreezePolicy(mode=AudioFreezeMode.CALIBRATED_SILENCE)

    calibrated = build_retime_execution_plan(
        _mixed_map(),
        video_frame_duration=Fraction(1, 30),
        audio_sample_rate=48_000,
        freeze_audio_policy=AudioFreezePolicy.calibrated_silence(
            "fcp-12.3-freeze-audio-silence-v1"
        ),
    )
    graph = build_audio_filtergraph(calibrated)
    assert "anullsrc=r=48000:cl=stereo" in graph
    assert "concat=n=3:v=0:a=1" in graph
    assert not calibrated.audio_blocked
    assert (
        calibrated.audio_segments[1].freeze_calibration_id
        == "fcp-12.3-freeze-audio-silence-v1"
    )


@pytest.mark.parametrize(
    ("rate", "expected"),
    (
        (Fraction(1), ()),
        (Fraction(3, 2), (Fraction(3, 2),)),
        (Fraction(10), (Fraction(2), Fraction(2), Fraction(2), Fraction(5, 4))),
        (
            Fraction(1, 10),
            (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(4, 5)),
        ),
    ),
)
def test_atempo_factorization_is_bounded_and_exact(
    rate: Fraction,
    expected: tuple[Fraction, ...],
) -> None:
    factors = bounded_atempo_factors(rate)
    assert factors == expected
    assert all(Fraction(1, 2) <= factor <= 2 for factor in factors)
    product = Fraction(1)
    for factor in factors:
        product *= factor
    assert product == rate


def test_filtergraphs_expose_exact_segment_order_and_pitch_modes() -> None:
    preserve_map = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(1), Fraction(10)),
        )
    )
    preserve = build_retime_execution_plan(
        preserve_map,
        video_frame_duration=Fraction(1, 30),
    )
    preserve_graph = build_audio_filtergraph(preserve)
    assert preserve.pitch_mode is AudioPitchMode.PRESERVE
    assert preserve_graph.count("atempo=2") == 3
    assert "atempo=1.25" in preserve_graph
    assert "atrim=duration=1" in preserve_graph

    pitch_map = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(1), Fraction(3, 2)),
        )
    )
    changed = build_retime_execution_plan(
        pitch_map,
        video_frame_duration=Fraction(1, 30),
        preserve_audio_pitch=False,
    )
    changed_graph = build_audio_filtergraph(changed)
    assert changed.pitch_mode is AudioPitchMode.CHANGE_WITH_SPEED
    assert "asetrate=48000*3/2,aresample=48000" in changed_graph
    assert "atempo=" not in changed_graph

    reverse = build_retime_execution_plan(
        _forward_reverse_map(),
        video_frame_duration=Fraction(1, 10),
        audio_sample_rate=8_000,
    )
    video_graph = build_video_filtergraph(reverse)
    audio_graph = build_audio_filtergraph(reverse)
    assert video_graph.index("split=2") < video_graph.index("reverse")
    assert "trim=start=0:end=1,reverse" in video_graph
    assert "concat=n=2:v=1:a=0" in video_graph
    assert "atrim=start=0:end=1,areverse" in audio_graph
    assert "concat=n=2:v=0:a=1" in audio_graph


def test_manifest_and_required_filter_set_are_deterministic() -> None:
    first = build_retime_execution_plan(
        _forward_reverse_map(),
        video_frame_duration=Fraction(1001, 30_000),
        audio_sample_rate=48_000,
    )
    second = build_retime_execution_plan(
        _forward_reverse_map(),
        video_frame_duration=Fraction(1001, 30_000),
        audio_sample_rate=48_000,
    )

    assert first.manifest() == second.manifest()
    assert first.manifest_sha256 == second.manifest_sha256
    assert len(first.manifest_sha256) == 64
    assert first.manifest()["video_frame_duration"] == "1001/30000"
    assert required_stock_filters(first) == (
        "areverse",
        "asetpts",
        "asplit",
        "atrim",
        "concat",
        "fps",
        "reverse",
        "setpts",
        "split",
        "trim",
    )


def test_installed_stock_ffmpeg_has_the_required_retime_filters() -> None:
    plan = build_retime_execution_plan(
        _mixed_map(),
        video_frame_duration=Fraction(1, 30),
        freeze_audio_policy=AudioFreezePolicy.calibrated_silence(
            "test-only-freeze-silence"
        ),
    )
    report = probe_stock_ffmpeg_capabilities(plan, ffmpeg_path=_ffmpeg())

    assert report.supported
    assert report.missing_filters == ()
    assert report.version_line.startswith("ffmpeg version ")
    assert set(report.required_filters) <= set(report.available_filters)


def test_stock_ffmpeg_reverse_video_has_no_duplicate_concat_boundary() -> None:
    plan = build_retime_execution_plan(
        _forward_reverse_map(),
        video_frame_duration=Fraction(1, 10),
        include_audio=False,
    )
    graph = build_video_filtergraph(plan)
    input_args = [
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x64:rate=10:duration=1.1",
    ]
    output_args = [
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    source = _run_ffmpeg([*input_args, "-map", "0:v:0", *output_args])
    first = _run_ffmpeg(
        [
            *input_args,
            "-filter_complex",
            graph,
            "-map",
            "[retimed_video]",
            *output_args,
        ]
    )
    second = _run_ffmpeg(
        [
            *input_args,
            "-filter_complex",
            graph,
            "-map",
            "[retimed_video]",
            *output_args,
        ]
    )

    frame_size = 64 * 64 * 3 // 2
    source_frames = [
        source[index : index + frame_size]
        for index in range(0, len(source), frame_size)
    ]
    output_frames = [
        first[index : index + frame_size]
        for index in range(0, len(first), frame_size)
    ]
    assert len(source_frames) == 11
    assert len(output_frames) == 20
    assert output_frames == source_frames[:10] + list(reversed(source_frames[:10]))
    assert output_frames[9] == output_frames[10]
    assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest()


@pytest.mark.parametrize("rate", (Fraction(7, 4), Fraction(3, 2), Fraction(3, 4)))
def test_stock_ffmpeg_fractional_speed_uses_floor_source_frames(
    rate: Fraction,
) -> None:
    mapping = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(1), rate),
        )
    )
    plan = build_retime_execution_plan(
        mapping,
        video_frame_duration=Fraction(1, 30),
        include_audio=False,
    )
    graph = build_video_filtergraph(plan)
    input_args = [
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x64:rate=30:duration=2",
    ]
    output_args = [
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    source = _run_ffmpeg([*input_args, "-map", "0:v:0", *output_args])
    output = _run_ffmpeg(
        [
            *input_args,
            "-filter_complex",
            graph,
            "-map",
            "[retimed_video]",
            *output_args,
        ]
    )

    frame_size = 64 * 64 * 3 // 2
    source_frames = [
        source[index : index + frame_size]
        for index in range(0, len(source), frame_size)
    ]
    output_frames = [
        output[index : index + frame_size]
        for index in range(0, len(output), frame_size)
    ]
    expected_indices = [int(index * rate) for index in range(30)]
    assert len(output_frames) == 30
    assert output_frames == [source_frames[index] for index in expected_indices]


@pytest.mark.parametrize("rate", (Fraction(7, 4), Fraction(3, 2), Fraction(3, 4)))
def test_stock_ffmpeg_fractional_speed_preserves_fine_source_phase(
    rate: Fraction,
) -> None:
    """A Brainrot-shaped edit retains floor ownership at nonzero speed."""

    source_start = Fraction(53_053, 30_000)
    mapping = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), source_start),
            RetimePoint(Fraction(1), source_start + rate),
        )
    )
    plan = build_retime_execution_plan(
        mapping,
        video_frame_duration=Fraction(1, 30),
        include_audio=False,
    )
    graph = build_video_filtergraph(plan)
    input_args = [
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x64:rate=30:duration=4",
    ]
    output_args = [
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    source = _run_ffmpeg([*input_args, "-map", "0:v:0", *output_args])
    output = _run_ffmpeg(
        [
            *input_args,
            "-filter_complex",
            graph,
            "-map",
            "[retimed_video]",
            *output_args,
        ]
    )

    frame_size = 64 * 64 * 3 // 2
    source_frames = [
        source[index : index + frame_size]
        for index in range(0, len(source), frame_size)
    ]
    output_frames = [
        output[index : index + frame_size]
        for index in range(0, len(output), frame_size)
    ]
    expected_indices = [
        int((source_start + index * Fraction(1, 30) * rate) * 30)
        for index in range(30)
    ]
    assert len(output_frames) == 30
    assert output_frames == [source_frames[index] for index in expected_indices]


def test_stock_ffmpeg_freeze_video_holds_one_source_frame() -> None:
    freeze_map = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(1, 2)),
            RetimePoint(Fraction(1), Fraction(1, 2)),
        )
    )
    plan = build_retime_execution_plan(
        freeze_map,
        video_frame_duration=Fraction(1, 10),
        include_audio=False,
    )
    output = _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10:duration=1",
            "-filter_complex",
            build_video_filtergraph(plan),
            "-map",
            "[retimed_video]",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )

    frame_size = 64 * 64 * 3 // 2
    frames = [
        output[index : index + frame_size]
        for index in range(0, len(output), frame_size)
    ]
    assert len(frames) == 10
    assert len({hashlib.sha256(frame).digest() for frame in frames}) == 1


def test_forward_to_freeze_holds_the_source_frame_before_the_boundary() -> None:
    """Freeze owns the left-limit frame at a forward segment boundary.

    Main callers:
        This is the executable regression for Final Cut's 4s freeze boundary
        in the Stage 5 retime oracle.

    Why this exists:
        Trimming from the exact time-map value selects the following decoded
        frame. Final Cut instead holds the frame whose interval ends at that
        value, while a leading freeze still starts at its authored value.
    """

    mapping = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(1), Fraction(1)),
            RetimePoint(Fraction(2), Fraction(1)),
        )
    )
    plan = build_retime_execution_plan(
        mapping,
        video_frame_duration=Fraction(1, 10),
        include_audio=False,
    )
    source = _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10:duration=2",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    output = _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10:duration=2",
            "-filter_complex",
            build_video_filtergraph(plan),
            "-map",
            "[retimed_video]",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )

    frame_size = 64 * 64 * 3 // 2
    source_frames = [
        source[index : index + frame_size]
        for index in range(0, len(source), frame_size)
    ]
    output_frames = [
        output[index : index + frame_size]
        for index in range(0, len(output), frame_size)
    ]
    assert len(output_frames) == 20
    assert output_frames[10:] == [source_frames[9]] * 10


def test_stock_ffmpeg_reverse_audio_is_sample_exact_and_deterministic() -> None:
    plan = build_retime_execution_plan(
        _forward_reverse_map(),
        video_frame_duration=Fraction(1, 10),
        audio_sample_rate=8_000,
    )
    graph = build_audio_filtergraph(plan)
    input_args = [
        "-f",
        "lavfi",
        "-i",
        "aevalsrc=sin(2*PI*440*t):s=8000:d=1",
    ]
    output_args = ["-ac", "1", "-ar", "8000", "-f", "s16le", "pipe:1"]
    source_bytes = _run_ffmpeg([*input_args, *output_args])
    first_bytes = _run_ffmpeg(
        [
            *input_args,
            "-filter_complex",
            graph,
            "-map",
            "[retimed_audio]",
            *output_args,
        ]
    )
    second_bytes = _run_ffmpeg(
        [
            *input_args,
            "-filter_complex",
            graph,
            "-map",
            "[retimed_audio]",
            *output_args,
        ]
    )

    source = array("h")
    source.frombytes(source_bytes)
    output = array("h")
    output.frombytes(first_bytes)
    assert len(source) == 8_000
    assert len(output) == 16_000
    assert output[:8_000] == source
    assert output[8_000:] == array("h", reversed(source))
    assert hashlib.sha256(first_bytes).digest() == hashlib.sha256(second_bytes).digest()


def test_stock_ffmpeg_executes_bounded_atempo_and_pitch_change_paths() -> None:
    preserve_map = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(1), Fraction(4)),
        )
    )
    preserve = build_retime_execution_plan(
        preserve_map,
        video_frame_duration=Fraction(1, 10),
        audio_sample_rate=8_000,
    )
    preserve_output = _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=8000:duration=4",
            "-filter_complex",
            build_audio_filtergraph(preserve),
            "-map",
            "[retimed_audio]",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-f",
            "s16le",
            "pipe:1",
        ]
    )
    assert len(preserve_output) == 8_000 * 2

    pitch_map = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(1), Fraction(3, 2)),
        )
    )
    changed = build_retime_execution_plan(
        pitch_map,
        video_frame_duration=Fraction(1, 10),
        audio_sample_rate=8_000,
        preserve_audio_pitch=False,
    )
    changed_output = _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=8000:duration=1.5",
            "-filter_complex",
            build_audio_filtergraph(changed),
            "-map",
            "[retimed_audio]",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-f",
            "s16le",
            "pipe:1",
        ]
    )
    assert len(changed_output) == 8_000 * 2


def test_capability_probe_accepts_an_explicit_executable_path() -> None:
    plan = build_retime_execution_plan(
        RetimeMap.identity(Fraction(1)),
        video_frame_duration=Fraction(1, 25),
        include_audio=False,
    )
    report = probe_stock_ffmpeg_capabilities(plan, ffmpeg_path=Path(_ffmpeg()))
    assert report.executable == _ffmpeg()
