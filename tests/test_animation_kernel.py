"""Isolated contract tests for typed source and retimed animation tracks."""

from fractions import Fraction
import math

import pytest

from bladeworks.core.animation import (
    AnimatedScalar,
    AnimatedVec2,
    AnimationValidationError,
    ScalarControlPoint,
    TimelineAnimatedScalar,
    TimelineAnimatedVec2,
    UnsupportedAnimationMappingError,
    Vec2ControlPoint,
    ken_burns_progress,
    ken_burns_progress_expression,
)
from bladeworks.core.model import Keyframe, Parameter
from bladeworks.core.retime import RetimeMap, RetimePoint


def _keyframe(
    time: Fraction,
    value: str,
    *,
    interp: str = "linear",
    curve: str = "linear",
    aux_value: str | None = None,
) -> Keyframe:
    return Keyframe(
        time=time,
        value=value,
        interp=interp,
        curve=curve,
        aux_value=aux_value,
    )


def _scalar_track(
    *,
    interp: str = "linear",
    curve: str = "linear",
) -> AnimatedScalar:
    return AnimatedScalar.from_keyframes(
        (
            _keyframe(Fraction(0), "0", curve=curve),
            _keyframe(Fraction(1), "10", interp=interp, curve=curve),
        )
    )


def test_parameter_adapters_preserve_exact_time_and_report_aux_value() -> None:
    parameter = Parameter(
        name="rotation",
        key=None,
        value=None,
        keyframes=(
            _keyframe(Fraction(1, 30), "-2", aux_value="4 5"),
            _keyframe(Fraction(31, 30), "2"),
        ),
    )

    animation = AnimatedScalar.from_parameter(parameter)

    assert animation.control_points[0].time == Fraction(1, 30)
    assert animation.control_points[0].aux_value == "4 5"
    assert len(animation.notices) == 1
    assert animation.notices[0].code == "uncalibrated_aux_value"
    assert "preserved but not applied" in animation.notices[0].detail


def test_vec2_adapter_requires_two_finite_components() -> None:
    animation = AnimatedVec2.from_keyframes(
        (
            _keyframe(Fraction(0), "-20 4"),
            _keyframe(Fraction(2), "20 8"),
        )
    )

    assert animation.value_at(Fraction(1)) == pytest.approx((0.0, 6.0))

    with pytest.raises(AnimationValidationError, match="exactly 2"):
        AnimatedVec2.from_keyframes((_keyframe(Fraction(0), "10"),))
    with pytest.raises(AnimationValidationError, match="must be finite"):
        AnimatedVec2.from_keyframes((_keyframe(Fraction(0), "10 nan"),))
    with pytest.raises(AnimationValidationError, match="exactly 1"):
        AnimatedScalar.from_keyframes((_keyframe(Fraction(0), "10 20"),))
    with pytest.raises(AnimationValidationError, match="must be ScalarControlPoint"):
        AnimatedScalar(
            (Vec2ControlPoint(Fraction(0), (10.0, 20.0)),)  # type: ignore[arg-type]
        )


def test_values_hold_before_and_after_the_keyframe_range() -> None:
    animation = _scalar_track()

    assert animation.value_at(Fraction(-100)) == 0.0
    assert animation.value_at(Fraction(0)) == 0.0
    assert animation.value_at(Fraction(1)) == 10.0
    assert animation.value_at(Fraction(100)) == 10.0


@pytest.mark.parametrize(
    ("interpolation", "expected"),
    (
        ("linear", 2.5),
        ("ease", 1.5625),
        ("easeIn", 0.15625),
        ("ease-out", 5.78125),
    ),
)
def test_destination_keyframe_selects_temporal_easing(
    interpolation: str,
    expected: float,
) -> None:
    animation = _scalar_track(interp=interpolation)

    assert animation.value_at(Fraction(1, 4)) == pytest.approx(expected)


def test_middle_interpolation_controls_only_its_incoming_interval() -> None:
    animation = AnimatedScalar.from_keyframes(
        (
            _keyframe(Fraction(0), "0", curve="linear"),
            _keyframe(Fraction(1), "10", interp="easeIn", curve="linear"),
            _keyframe(Fraction(2), "20", curve="linear"),
        )
    )

    assert animation.value_at(Fraction(1, 2)) == pytest.approx(1.25)
    assert animation.value_at(Fraction(3, 2)) == pytest.approx(15.0)


def test_smooth_curve_is_monotone_and_cannot_overshoot_each_segment() -> None:
    animation = AnimatedScalar(
        (
            ScalarControlPoint(Fraction(0), 0.0, curve="smooth"),
            ScalarControlPoint(Fraction(1), 1.0, curve="smooth"),
            ScalarControlPoint(Fraction(2), 10.0, curve="smooth"),
            ScalarControlPoint(Fraction(3), 10.5, curve="smooth"),
        )
    )

    first_segment = [animation.value_at(Fraction(index, 20)) for index in range(21)]
    second_segment = [
        animation.value_at(Fraction(20 + index, 20)) for index in range(21)
    ]
    third_segment = [
        animation.value_at(Fraction(40 + index, 20)) for index in range(21)
    ]

    assert first_segment == sorted(first_segment)
    assert second_segment == sorted(second_segment)
    assert third_segment == sorted(third_segment)
    assert all(0.0 <= value <= 1.0 for value in first_segment)
    assert all(1.0 <= value <= 10.0 for value in second_segment)
    assert all(10.0 <= value <= 10.5 for value in third_segment)


def test_smooth_curve_stops_at_a_local_extremum_without_overshoot() -> None:
    animation = AnimatedScalar(
        (
            ScalarControlPoint(Fraction(0), 0.0, curve="smooth"),
            ScalarControlPoint(Fraction(1), 4.0, curve="smooth"),
            ScalarControlPoint(Fraction(2), 1.0, curve="smooth"),
        )
    )

    sampled = [animation.value_at(Fraction(index, 20)) for index in range(41)]

    assert max(sampled) == pytest.approx(4.0)
    assert min(sampled) == pytest.approx(0.0)


def test_one_point_track_is_a_constant_animation() -> None:
    animation = AnimatedScalar((ScalarControlPoint(Fraction(7), 3.5),))

    assert animation.value_at(Fraction(-10)) == 3.5
    assert animation.value_at(Fraction(100)) == 3.5


def test_malformed_tracks_fail_instead_of_being_reordered_or_coerced() -> None:
    with pytest.raises(AnimationValidationError, match="duplicate control-point time"):
        AnimatedScalar(
            (
                ScalarControlPoint(Fraction(0), 0),
                ScalarControlPoint(Fraction(0), 1),
            )
        )
    with pytest.raises(AnimationValidationError, match="strictly increasing"):
        AnimatedScalar(
            (
                ScalarControlPoint(Fraction(1), 0),
                ScalarControlPoint(Fraction(0), 1),
            )
        )
    with pytest.raises(AnimationValidationError, match="exact Fraction, not float"):
        ScalarControlPoint(0.1, 1)  # type: ignore[arg-type]
    with pytest.raises(AnimationValidationError, match="must be finite"):
        ScalarControlPoint(Fraction(0), math.inf)
    with pytest.raises(
        AnimationValidationError, match="unsupported keyframe interpolation"
    ):
        ScalarControlPoint(Fraction(0), 1, interpolation="bezier")  # type: ignore[arg-type]
    with pytest.raises(AnimationValidationError, match="unsupported keyframe curve"):
        ScalarControlPoint(Fraction(0), 1, curve="natural")  # type: ignore[arg-type]


def test_parameter_without_keyframes_is_not_invented_as_an_animation() -> None:
    parameter = Parameter(name="amount", key=None, value="5", keyframes=())

    with pytest.raises(AnimationValidationError, match="has no keyframeAnimation"):
        AnimatedScalar.from_parameter(parameter)


def test_reverse_retime_evaluates_easing_in_source_track_order() -> None:
    source = AnimatedScalar.from_keyframes(
        (
            _keyframe(Fraction(0), "0"),
            _keyframe(Fraction(2), "20", interp="easeIn"),
        )
    )
    mapping = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(2)),
            RetimePoint(Fraction(2), Fraction(0)),
        )
    )
    animation = TimelineAnimatedScalar(source, mapping)

    # timeline 1/2 maps to source 3/2, so source-order ease-in is evaluated at
    # 75%; timeline 3/2 maps to source 1/2 and evaluates it at 25%.
    assert animation.value_at(Fraction(1, 2)) == pytest.approx(20.0 * 0.75**3)
    assert animation.value_at(Fraction(3, 2)) == pytest.approx(20.0 * 0.25**3)


def test_repeated_source_ranges_expose_every_keyframe_occurrence() -> None:
    source = AnimatedScalar.from_keyframes(
        (
            _keyframe(Fraction(0), "0"),
            _keyframe(Fraction(1), "10"),
            _keyframe(Fraction(2), "20"),
        )
    )
    mapping = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(2), Fraction(2)),
            RetimePoint(Fraction(4), Fraction(0)),
        )
    )
    animation = TimelineAnimatedScalar(source, mapping)

    middle_occurrences = [
        occurrence
        for occurrence in animation.control_point_occurrences
        if occurrence.source_index == 1
    ]
    assert [item.timeline_start for item in middle_occurrences] == [
        Fraction(1),
        Fraction(3),
    ]
    assert animation.value_at(Fraction(1, 2)) == pytest.approx(5.0)
    assert animation.value_at(Fraction(7, 2)) == pytest.approx(5.0)


def test_freeze_keyframe_occurrence_remains_an_interval() -> None:
    source = _scalar_track()
    mapping = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(1), Fraction(1)),
            RetimePoint(Fraction(3), Fraction(1)),
        )
    )
    animation = TimelineAnimatedScalar(source, mapping)

    final_occurrence = [
        item for item in animation.control_point_occurrences if item.source_index == 1
    ][0]
    assert final_occurrence.is_interval is True
    assert (final_occurrence.timeline_start, final_occurrence.timeline_end) == (
        Fraction(1),
        Fraction(3),
    )
    assert animation.value_at(Fraction(2)) == 10.0
    with pytest.raises(
        UnsupportedAnimationMappingError, match="one continuous retime segment"
    ):
        animation.materialize_timeline_track()


def test_simple_forward_retime_can_materialize_exact_timeline_points() -> None:
    source = AnimatedScalar.from_keyframes(
        (
            _keyframe(Fraction(0), "0"),
            _keyframe(Fraction(1), "10"),
            _keyframe(Fraction(2), "20"),
        )
    )
    mapping = RetimeMap.from_points(
        (
            RetimePoint(Fraction(10), Fraction(0)),
            RetimePoint(Fraction(14), Fraction(2)),
        )
    )

    flattened = TimelineAnimatedScalar(source, mapping).materialize_timeline_track()

    assert [point.time for point in flattened.control_points] == [
        Fraction(10),
        Fraction(12),
        Fraction(14),
    ]
    assert flattened.value_at(Fraction(11)) == pytest.approx(5.0)


def test_reverse_and_multi_segment_maps_refuse_lossy_materialization() -> None:
    source = _scalar_track()
    reverse = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(1)),
            RetimePoint(Fraction(1), Fraction(0)),
        )
    )
    repeated = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(1), Fraction(1)),
            RetimePoint(Fraction(2), Fraction(0)),
        )
    )

    with pytest.raises(UnsupportedAnimationMappingError, match="reverse easing"):
        TimelineAnimatedScalar(source, reverse).materialize_timeline_track()
    with pytest.raises(UnsupportedAnimationMappingError, match="one continuous"):
        TimelineAnimatedScalar(source, repeated).materialize_timeline_track()


def test_partial_source_range_refuses_to_invent_boundary_control_points() -> None:
    source = AnimatedScalar.from_keyframes(
        (
            _keyframe(Fraction(-1), "-10"),
            _keyframe(Fraction(1), "10"),
            _keyframe(Fraction(3), "30"),
        )
    )
    selected_middle = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(2), Fraction(2)),
        )
    )

    with pytest.raises(
        UnsupportedAnimationMappingError, match="partial source-track range"
    ):
        TimelineAnimatedScalar(source, selected_middle).materialize_timeline_track()


def test_vec2_retime_wrapper_maps_both_components() -> None:
    source = AnimatedVec2(
        (
            Vec2ControlPoint(Fraction(0), (0.0, 10.0), curve="linear"),
            Vec2ControlPoint(Fraction(2), (20.0, -10.0), curve="linear"),
        )
    )
    mapping = RetimeMap.identity(Fraction(2), timeline_start=Fraction(5))

    assert TimelineAnimatedVec2(source, mapping).value_at(Fraction(6)) == pytest.approx(
        (10.0, 0.0)
    )


def test_ken_burns_progress_is_one_shared_calibrated_endpoint_held_curve() -> None:
    duration = Fraction(2)
    assert ken_burns_progress(Fraction(-1), duration) == 0.0
    assert ken_burns_progress(Fraction(0), duration) == 0.0
    assert ken_burns_progress(Fraction(1, 2), duration) == pytest.approx(0.1592, abs=0.006)
    assert ken_burns_progress(Fraction(1), duration) == pytest.approx(0.4863, abs=0.006)
    assert ken_burns_progress(Fraction(3, 2), duration) == pytest.approx(0.8142, abs=0.006)
    assert ken_burns_progress(Fraction(2), duration) == 1.0
    assert ken_burns_progress(Fraction(3), duration) == 1.0

    expression = ken_burns_progress_expression(counter="on", frame_count=60)
    assert expression.count("min(max(on/59,0),1)") >= 20
    perspective_expression = ken_burns_progress_expression(
        counter="on",
        frame_count=60,
        first_frame_index=1,
    )
    assert perspective_expression.count("min(max((on-1)/59,0),1)") >= 20


def test_ken_burns_progress_rejects_ambiguous_clocks_and_durations() -> None:
    with pytest.raises(AnimationValidationError, match="duration must be positive"):
        ken_burns_progress(Fraction(0), Fraction(0))
    with pytest.raises(AnimationValidationError, match="exact Fraction, not float"):
        ken_burns_progress(0.1, Fraction(1))  # type: ignore[arg-type]
    with pytest.raises(AnimationValidationError, match="renderer-owned 'on'"):
        ken_burns_progress_expression(counter="t", frame_count=60)
    with pytest.raises(AnimationValidationError, match="at least two"):
        ken_burns_progress_expression(counter="on", frame_count=1)
