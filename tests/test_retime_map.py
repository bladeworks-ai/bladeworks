"""Isolated contract tests for the experimental exact retiming kernel."""

from decimal import Decimal
from fractions import Fraction

import pytest

from bladeworks.core.model import TimeMapPoint
from bladeworks.core.retime import (
    RetimeMap,
    RetimePoint,
    RetimeSegment,
    RetimeValidationError,
    TimelineOutsideRetimeMapError,
    UnsupportedRetimeMappingError,
)


def test_constant_forward_map_stays_exact() -> None:
    mapping = RetimeMap.from_points(
        (
            RetimePoint(Fraction(1), Fraction(10)),
            RetimePoint(Fraction(3), Fraction(15)),
        )
    )

    assert mapping.rates == (Fraction(5, 2),)
    assert mapping.segments[0].kind == "forward"
    assert mapping.timeline_to_source(Fraction(3, 2)) == Fraction(45, 4)
    occurrence = mapping.source_occurrences(Fraction(45, 4))[0]
    assert occurrence.timeline_time == Fraction(3, 2)


def test_piecewise_rates_represent_a_variable_ramp() -> None:
    mapping = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(2), Fraction(4)),
            RetimePoint(Fraction(6), Fraction(6)),
        )
    )

    assert mapping.rates == (Fraction(2), Fraction(1, 2))
    assert mapping.is_variable_rate is True
    assert mapping.map_timeline(Fraction(3)) == Fraction(9, 2)
    assert mapping.source_occurrences(Fraction(5))[0].timeline_time == Fraction(4)


def test_reverse_ranges_return_every_source_occurrence() -> None:
    mapping = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(2), Fraction(2)),
            RetimePoint(Fraction(4), Fraction(0)),
        )
    )

    assert [segment.kind for segment in mapping.segments] == ["forward", "reverse"]
    assert [item.timeline_time for item in mapping.source_occurrences(Fraction(1))] == [
        Fraction(1),
        Fraction(3),
    ]
    boundary = mapping.source_occurrences(Fraction(2))
    assert len(boundary) == 1
    assert boundary[0].timeline_time == Fraction(2)
    assert boundary[0].segment_indices == (1,)


def test_freeze_returns_an_interval_and_merges_its_owned_boundary() -> None:
    mapping = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(2), Fraction(2)),
            RetimePoint(Fraction(5), Fraction(2)),
            RetimePoint(Fraction(7), Fraction(4)),
        )
    )

    occurrence = mapping.source_occurrences(Fraction(2))[0]
    assert occurrence.is_interval is True
    assert (occurrence.timeline_start, occurrence.timeline_end) == (
        Fraction(2),
        Fraction(5),
    )
    assert occurrence.includes_end is True
    assert occurrence.segment_indices == (1, 2)
    with pytest.raises(RetimeValidationError, match="freeze occurrence is an interval"):
        _ = occurrence.timeline_time


def test_final_freeze_includes_the_map_end() -> None:
    mapping = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(1), Fraction(1)),
            RetimePoint(Fraction(3), Fraction(1)),
        )
    )

    occurrence = mapping.source_occurrences(Fraction(1))[0]
    assert (occurrence.timeline_start, occurrence.timeline_end) == (
        Fraction(1),
        Fraction(3),
    )
    assert occurrence.includes_end is True


def test_half_open_boundary_ownership_handles_source_jumps() -> None:
    mapping = RetimeMap(
        (
            RetimeSegment(Fraction(0), Fraction(1), Fraction(0), Fraction(1)),
            RetimeSegment(Fraction(1), Fraction(2), Fraction(10), Fraction(11)),
        )
    )

    assert mapping.map_timeline(Fraction(1)) == Fraction(10)
    assert mapping.source_occurrences(Fraction(1)) == ()
    assert mapping.source_occurrences(Fraction(10))[0].timeline_time == Fraction(1)


def test_parser_time_map_points_have_an_explicit_adapter() -> None:
    mapping = RetimeMap.from_time_map_points(
        (
            TimeMapPoint(time=Fraction(0), value=Fraction(8), interp="linear"),
            TimeMapPoint(time=Fraction(4), value=Fraction(6), interp="linear"),
        )
    )

    assert mapping.rates == (Fraction(-1, 2),)
    assert mapping.map_timeline(Fraction(2)) == Fraction(7)


@pytest.mark.parametrize(
    ("segments", "message"),
    (
        (
            (
                RetimeSegment(Fraction(1), Fraction(2), Fraction(0), Fraction(1)),
                RetimeSegment(Fraction(0), Fraction(1), Fraction(1), Fraction(2)),
            ),
            "nonmonotonic timeline domain",
        ),
        (
            (
                RetimeSegment(Fraction(0), Fraction(2), Fraction(0), Fraction(1)),
                RetimeSegment(Fraction(1), Fraction(3), Fraction(1), Fraction(2)),
            ),
            "overlaps",
        ),
        (
            (
                RetimeSegment(Fraction(0), Fraction(1), Fraction(0), Fraction(1)),
                RetimeSegment(Fraction(2), Fraction(3), Fraction(1), Fraction(2)),
            ),
            "timeline gap",
        ),
    ),
)
def test_map_rejects_malformed_segment_domains(
    segments: tuple[RetimeSegment, ...],
    message: str,
) -> None:
    with pytest.raises(RetimeValidationError, match=message):
        RetimeMap(segments)


def test_zero_length_and_duplicate_point_domains_are_rejected() -> None:
    with pytest.raises(RetimeValidationError, match="greater than timeline_start"):
        RetimeSegment(Fraction(1), Fraction(1), Fraction(0), Fraction(1))
    with pytest.raises(RetimeValidationError, match="greater than timeline_start"):
        RetimeMap.from_points(
            (
                RetimePoint(Fraction(0), Fraction(0)),
                RetimePoint(Fraction(0), Fraction(1)),
            )
        )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), Decimal("NaN")))
def test_nonfinite_coordinates_are_rejected(value: object) -> None:
    with pytest.raises(RetimeValidationError, match="must be finite"):
        RetimePoint(value, Fraction(0))  # type: ignore[arg-type]


def test_finite_floats_are_rejected_instead_of_becoming_binary_fractions() -> None:
    with pytest.raises(RetimeValidationError, match="exact Fraction, not float"):
        RetimePoint(0.1, Fraction(0))  # type: ignore[arg-type]


def test_nonlinear_interpolation_is_explicitly_unsupported() -> None:
    with pytest.raises(
        UnsupportedRetimeMappingError,
        match="unsupported nonlinear interpolation 'smooth'",
    ):
        RetimeMap.from_points(
            (
                RetimePoint(Fraction(0), Fraction(0), "linear"),
                RetimePoint(Fraction(1), Fraction(1), "smooth"),
            )
        )


def test_visible_retime_discards_nonlinear_terminal_context_but_not_visible_curve() -> None:
    extended_reverse = (
        RetimePoint(Fraction(0), Fraction(2), "linear"),
        RetimePoint(Fraction(2), Fraction(0), "linear"),
        RetimePoint(Fraction(22), Fraction(20), "smooth2"),
    )

    visible = RetimeMap.from_points_visible(extended_reverse, Fraction(2))

    assert visible.timeline_end == 2
    assert visible.segments[0].source_start == 2
    assert visible.segments[0].source_end == 0
    with pytest.raises(UnsupportedRetimeMappingError, match="smooth2"):
        RetimeMap.from_points_visible(extended_reverse, Fraction(3))


def test_timeline_lookup_rejects_times_outside_the_map() -> None:
    mapping = RetimeMap.identity(Fraction(2))

    with pytest.raises(TimelineOutsideRetimeMapError, match="outside map"):
        mapping.map_timeline(Fraction(-1))
    with pytest.raises(TimelineOutsideRetimeMapError, match="outside map"):
        mapping.map_timeline(Fraction(3))


def test_samples_and_boundaries_expose_exact_half_open_ownership() -> None:
    mapping = RetimeMap(
        (
            RetimeSegment(Fraction(0), Fraction(1), Fraction(0), Fraction(1)),
            RetimeSegment(Fraction(1), Fraction(2), Fraction(8), Fraction(8)),
            RetimeSegment(Fraction(2), Fraction(3), Fraction(8), Fraction(7)),
        )
    )

    boundaries = mapping.boundaries
    assert len(boundaries) == 2
    assert boundaries[0].source_is_continuous is False
    assert boundaries[0].outgoing_source_end == 1
    assert boundaries[0].source_at_boundary == 8
    assert boundaries[1].source_is_continuous is True

    # At each internal boundary the incoming segment is the one exact owner.
    freeze = mapping.sample(Fraction(1))
    reverse = mapping.sample(Fraction(2))
    terminal = mapping.sample(Fraction(3))
    assert (freeze.segment_index, freeze.segment_kind, freeze.source_time) == (
        1,
        "freeze",
        Fraction(8),
    )
    assert (reverse.segment_index, reverse.segment_kind, reverse.source_time) == (
        2,
        "reverse",
        Fraction(8),
    )
    assert terminal.at_segment_end is True
    assert terminal.source_time == 7


def test_frame_boundary_assertion_returns_exact_indices_or_fails() -> None:
    aligned = RetimeMap(
        (
            RetimeSegment(Fraction(5), Fraction(6), Fraction(0), Fraction(2)),
            RetimeSegment(Fraction(6), Fraction(13, 2), Fraction(2), Fraction(2)),
        )
    )
    assert aligned.require_frame_aligned_boundaries(Fraction(1, 30)) == (30, 45)

    with pytest.raises(RetimeValidationError, match="not aligned"):
        aligned.require_frame_aligned_boundaries(Fraction(1, 29))


def test_transition_endpoint_holds_preserve_authored_segments_and_occurrences() -> None:
    authored = RetimeMap(
        (
            RetimeSegment(Fraction(4), Fraction(5), Fraction(10), Fraction(11)),
            RetimeSegment(Fraction(5), Fraction(6), Fraction(11), Fraction(10)),
        )
    )
    expanded = authored.with_endpoint_holds(
        pre_roll=Fraction(1, 2),
        post_roll=Fraction(1, 4),
    )

    assert [segment.kind for segment in expanded.segments] == [
        "freeze",
        "forward",
        "reverse",
        "freeze",
    ]
    assert expanded.timeline_start == 0
    assert expanded.timeline_end == Fraction(11, 4)
    assert expanded.map_timeline(Fraction(0)) == 10
    assert expanded.map_timeline(Fraction(1, 2)) == 10
    assert expanded.map_timeline(Fraction(3, 2)) == 11
    assert expanded.map_timeline(Fraction(5, 2)) == 10
    assert expanded.map_timeline(Fraction(11, 4)) == 10

    # Source 10 occurs in both authored directions and both handle intervals;
    # merging may join adjacent intervals but must not discard either end.
    occurrences = expanded.source_occurrences(Fraction(10))
    assert occurrences[0].timeline_start == 0
    assert occurrences[-1].timeline_end == Fraction(11, 4)


def test_transition_hold_validation_fails_on_inexact_or_negative_durations() -> None:
    mapping = RetimeMap.identity(Fraction(1))
    with pytest.raises(RetimeValidationError, match="non-negative"):
        mapping.with_endpoint_holds(pre_roll=Fraction(-1))
    with pytest.raises(RetimeValidationError, match="exact Fraction, not float"):
        mapping.with_endpoint_holds(post_roll=0.25)  # type: ignore[arg-type]
