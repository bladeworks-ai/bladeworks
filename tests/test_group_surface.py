"""Contracts for expanded intermediate XYZT group surfaces."""

from __future__ import annotations

from fractions import Fraction

import pytest

from bladeworks.core.geometry import (
    FrameGeometry,
    TransformState,
    transform_points,
)
from bladeworks.core.group_surface import (
    GroupSurfaceValidationError,
    QuadTrajectory,
    SamplingWindow,
    SpatialTrajectory,
    TransformTrajectory,
    UnboundedGroupCurveError,
    plan_group_surfaces,
    projective_map_points,
    surface_project_quad,
)


FRAME = FrameGeometry(
    source_width=160,
    source_height=90,
    project_width=160,
    project_height=90,
)
CANVAS = ((0.0, 0.0), (160.0, 0.0), (0.0, 90.0), (160.0, 90.0))
IDENTITY = TransformState(
    position=(0.0, 0.0),
    scale=(1.0, 1.0),
    rotation_degrees=0.0,
    anchor=(0.0, 0.0),
)


def _window(
    *,
    sentinels: tuple[Fraction, ...] = (),
    extrema: tuple[Fraction, ...] = (),
) -> SamplingWindow:
    return SamplingWindow(
        start=Fraction(0),
        end=Fraction(1),
        frame_duration=Fraction(1, 2),
        sentinel_times=sentinels,
        extrema_times=extrema,
    )


def test_static_off_canvas_child_is_rebased_without_clipping() -> None:
    child_quad = ((-25.25, 5.0), (80.0, 5.0), (-20.0, 75.0), (85.5, 75.0))
    plan = plan_group_surfaces(
        FRAME,
        (
            QuadTrajectory(
                id="child",
                window=_window(),
                evaluate=lambda _time: child_quad,
                bounded=True,
            ),
        ),
    )

    assert plan.sample_times == (Fraction(0), Fraction(1, 2), Fraction(1))
    assert plan.child_surface.origin_x == -28
    assert plan.child_surface.bounds.left < min(point[0] for point in child_quad)
    sample = plan.member_samples[0]
    for project_point, surface_point in zip(
        sample.project_quads[0], sample.surface_quads[0], strict=True
    ):
        assert plan.child_surface.surface_to_project(surface_point) == pytest.approx(
            project_point
        )


def test_parent_is_applied_after_child_and_maps_complete_intermediate_surface() -> None:
    child_state = TransformState(
        position=(58.0, 0.0),
        scale=(0.9, 0.9),
        rotation_degrees=18.0,
        anchor=(0.0, 0.0),
    )
    parent_state = TransformState(
        position=(-42.0, 5.0),
        scale=(0.7, 1.1),
        rotation_degrees=-13.0,
        anchor=(8.0, -4.0),
    )
    child_quad = transform_points(FRAME, child_state, CANVAS)
    plan = plan_group_surfaces(
        FRAME,
        (
            QuadTrajectory(
                id="child",
                window=_window(),
                evaluate=lambda _time: child_quad,
                bounded=True,
            ),
        ),
        owners=(
            TransformTrajectory(
                id="parent",
                window=_window(),
                evaluate=lambda _time: parent_state,
                bounded=True,
            ),
        ),
    )

    sample = plan.member_samples[0]
    expected = transform_points(FRAME, parent_state, child_quad)
    reverse_order = transform_points(
        FRAME,
        child_state,
        transform_points(FRAME, parent_state, CANVAS),
    )
    for observed, wanted in zip(sample.project_quads[1], expected, strict=True):
        assert observed == pytest.approx(wanted)
    assert any(
        observed != pytest.approx(wanted)
        for observed, wanted in zip(sample.project_quads[1], reverse_order, strict=True)
    )
    mapping = plan.surface_transforms[0][0]
    assert mapping.source_project_quad == surface_project_quad(plan.child_surface)
    expected_surface = transform_points(FRAME, parent_state, mapping.source_project_quad)
    for observed, wanted in zip(
        mapping.destination_project_quad,
        expected_surface,
        strict=True,
    ):
        assert observed == pytest.approx(wanted)
    for project_point, surface_point in zip(
        mapping.destination_project_quad,
        mapping.destination_surface_quad,
        strict=True,
    ):
        assert plan.output_surface.surface_to_project(surface_point) == pytest.approx(
            project_point
        )


def test_declared_in_between_extremum_expands_the_surface() -> None:
    extremum = Fraction(1, 4)

    def quad_at(time: Fraction):
        excursion = -90.0 if time == extremum else 0.0
        return tuple((x + excursion, y) for x, y in CANVAS)

    without_extremum = plan_group_surfaces(
        FRAME,
        (
            QuadTrajectory(
                id="child",
                window=_window(),
                evaluate=quad_at,
                bounded=True,
            ),
        ),
    )
    with_extremum = plan_group_surfaces(
        FRAME,
        (
            QuadTrajectory(
                id="child",
                window=_window(extrema=(extremum,)),
                evaluate=quad_at,
                bounded=True,
            ),
        ),
    )

    assert extremum not in without_extremum.sample_times
    assert extremum in with_extremum.sample_times
    assert without_extremum.child_surface.origin_x == -2
    assert with_extremum.child_surface.origin_x == -92


def test_owner_sentinel_is_sampled_when_a_child_is_visible() -> None:
    sentinel = Fraction(1, 4)

    def parent_at(time: Fraction) -> TransformState:
        return TransformState(
            position=(80.0 if time == sentinel else 0.0, 0.0),
            scale=(1.0, 1.0),
            rotation_degrees=0.0,
            anchor=(0.0, 0.0),
        )

    plan = plan_group_surfaces(
        FRAME,
        (
            QuadTrajectory(
                id="child",
                window=_window(),
                evaluate=lambda _time: CANVAS,
                bounded=True,
            ),
        ),
        owners=(
            TransformTrajectory(
                id="parent",
                window=_window(sentinels=(sentinel,)),
                evaluate=parent_at,
                bounded=True,
            ),
        ),
    )

    assert sentinel in plan.sample_times
    assert plan.output_surface.bounds.right > FRAME.project_width


def test_projective_owner_maps_the_expanded_surface_not_only_canvas_corners() -> None:
    destination = (
        (12.0, 9.0),
        (145.0, -4.0),
        (-8.0, 82.0),
        (171.0, 101.0),
    )
    mapped_canvas = projective_map_points(FRAME, destination, CANVAS)
    for observed, wanted in zip(mapped_canvas, destination, strict=True):
        assert observed == pytest.approx(wanted)

    child_quad = ((-35.0, -10.0), (190.0, -10.0), (-35.0, 105.0), (190.0, 105.0))
    plan = plan_group_surfaces(
        FRAME,
        (
            QuadTrajectory(
                id="child",
                window=_window(),
                evaluate=lambda _time: child_quad,
                bounded=True,
            ),
        ),
        owners=(
            SpatialTrajectory(
                id="corner-and-affine-parent",
                window=_window(),
                evaluate=lambda _time, points: projective_map_points(
                    FRAME,
                    destination,
                    points,
                ),
                bounded=True,
            ),
        ),
    )

    mapping = plan.surface_transforms[0][0]
    expected = projective_map_points(
        FRAME,
        destination,
        surface_project_quad(plan.child_surface),
    )
    for observed, wanted in zip(
        mapping.destination_project_quad,
        expected,
        strict=True,
    ):
        assert observed == pytest.approx(wanted)
    assert plan.output_surface.bounds.left <= min(point[0] for point in expected)
    assert plan.output_surface.bounds.right >= max(point[0] for point in expected)


def test_unbounded_child_or_parent_curve_fails_closed() -> None:
    child = QuadTrajectory(
        id="child",
        window=_window(),
        evaluate=lambda _time: CANVAS,
        bounded=False,
    )
    with pytest.raises(UnboundedGroupCurveError, match="finite extrema proof"):
        plan_group_surfaces(FRAME, (child,))

    bounded_child = QuadTrajectory(
        id="child",
        window=_window(),
        evaluate=lambda _time: CANVAS,
        bounded=True,
    )
    parent = TransformTrajectory(
        id="parent",
        window=_window(),
        evaluate=lambda _time: IDENTITY,
        bounded=False,
    )
    with pytest.raises(UnboundedGroupCurveError, match="parent"):
        plan_group_surfaces(FRAME, (bounded_child,), owners=(parent,))


def test_owner_must_cover_every_visible_child_sample() -> None:
    child = QuadTrajectory(
        id="child",
        window=_window(),
        evaluate=lambda _time: CANVAS,
        bounded=True,
    )
    short_owner = TransformTrajectory(
        id="parent",
        window=SamplingWindow(
            start=Fraction(0),
            end=Fraction(1, 2),
            frame_duration=Fraction(1, 2),
        ),
        evaluate=lambda _time: IDENTITY,
        bounded=True,
    )
    with pytest.raises(GroupSurfaceValidationError, match="does not cover"):
        plan_group_surfaces(FRAME, (child,), owners=(short_owner,))


def test_non_finite_and_degenerate_samples_fail_before_allocation() -> None:
    for bad_quad in (
        ((0.0, 0.0), (float("inf"), 0.0), (0.0, 90.0), (160.0, 90.0)),
        ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
    ):
        child = QuadTrajectory(
            id="child",
            window=_window(),
            evaluate=lambda _time, value=bad_quad: value,
            bounded=True,
        )
        with pytest.raises(GroupSurfaceValidationError):
            plan_group_surfaces(FRAME, (child,))


def test_sampling_is_bounded_by_an_explicit_resource_limit() -> None:
    child = QuadTrajectory(
        id="child",
        window=SamplingWindow(
            start=Fraction(0),
            end=Fraction(10),
            frame_duration=Fraction(1, 30),
        ),
        evaluate=lambda _time: CANVAS,
        bounded=True,
    )
    with pytest.raises(GroupSurfaceValidationError, match="max_samples=10"):
        plan_group_surfaces(FRAME, (child,), max_samples=10)
