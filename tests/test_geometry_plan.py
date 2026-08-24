"""Isolated executable contracts for the Wave 2 geometry planner."""

from fractions import Fraction
import shutil
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
from bladeworks.core.geometry import (
    CanvasBounds,
    CornerPinAdjustment,
    CornerPinAnimation,
    FrameGeometry,
    GeometryPlan,
    GeometryValidationError,
    GeometryWindow,
    PixelRect,
    SourceRect,
    TransformState,
    UnsupportedGeometryAnimationError,
    compose_spatial_quad,
    correct_quad_for_pixel_centers,
    render_surface_for_quads,
    resolve_camera_placement,
    resolve_crop_camera_placement,
    transform_points,
)
from bladeworks.core.model import (
    CropAdjustment,
    CropRect,
    RenderTransformAnimation,
    TransformAdjustment,
)
from bladeworks.core.retime import RetimeMap


def _window(
    *,
    clip_start: Fraction = Fraction(1),
    clip_duration: Fraction = Fraction(2),
    pre_roll: Fraction = Fraction(1),
    post_roll: Fraction = Fraction(1),
) -> GeometryWindow:
    return GeometryWindow(
        clip_start=clip_start,
        clip_duration=clip_duration,
        render_start=clip_start - pre_roll,
        render_duration=clip_duration + pre_roll + post_roll,
    )


def _frame(
    *,
    source: tuple[int, int] = (200, 100),
    project: tuple[int, int] = (200, 100),
) -> FrameGeometry:
    return FrameGeometry(
        source_width=source[0],
        source_height=source[1],
        project_width=project[0],
        project_height=project[1],
    )


def _transform(
    *,
    position: tuple[float, float] = (0.0, 0.0),
    scale: tuple[float, float] = (1.0, 1.0),
    rotation: float = 0.0,
    anchor: tuple[float, float] = (0.0, 0.0),
) -> TransformAdjustment:
    return TransformAdjustment(
        position=position,
        scale=scale,
        rotation=rotation,
        enabled=True,
        anchor=anchor,
    )


def _vec_track(start: tuple[float, float], end: tuple[float, float]) -> TimelineAnimatedVec2:
    source = AnimatedVec2(
        (
            Vec2ControlPoint(Fraction(0), start),
            Vec2ControlPoint(Fraction(2), end),
        )
    )
    return TimelineAnimatedVec2(source, RetimeMap.identity(Fraction(2)))


def _scalar_track(start: float, end: float) -> TimelineAnimatedScalar:
    source = AnimatedScalar(
        (
            ScalarControlPoint(Fraction(0), start),
            ScalarControlPoint(Fraction(2), end),
        )
    )
    return TimelineAnimatedScalar(source, RetimeMap.identity(Fraction(2)))


def test_transition_handles_hold_geometry_at_clip_endpoints() -> None:
    animation = RenderTransformAnimation(
        position=_vec_track((0.0, 0.0), (20.0, 10.0)),
        rotation=_scalar_track(0.0, 90.0),
    )
    plan = GeometryPlan(
        frame=_frame(),
        window=_window(),
        transform=_transform(),
        transform_animation=animation,
    )

    before = plan.snapshot(Fraction(0))
    middle = plan.snapshot(Fraction(2))
    after = plan.snapshot(Fraction(4))

    assert before.clip_time == 0
    assert before.transform.position == pytest.approx((0.0, 0.0))
    assert middle.transform.position == pytest.approx((10.0, 5.0))
    assert middle.transform.rotation_degrees == pytest.approx(45.0)
    assert after.clip_time == 2
    assert after.transform.position == pytest.approx((20.0, 10.0))
    assert after.transform.rotation_degrees == pytest.approx(90.0)


def test_timeline_boundary_rejects_float_time_and_unexpanded_windows() -> None:
    with pytest.raises(GeometryValidationError, match="exact Fraction"):
        GeometryWindow(0.1, Fraction(1), Fraction(0), Fraction(1))  # type: ignore[arg-type]
    trimmed = GeometryWindow(Fraction(0), Fraction(2), Fraction(1), Fraction(1))
    assert trimmed.clip_time(Fraction(1)) == 1
    with pytest.raises(GeometryValidationError, match="must overlap"):
        GeometryWindow(Fraction(0), Fraction(2), Fraction(2), Fraction(1))
    with pytest.raises(GeometryValidationError, match="outside the render interval"):
        _window().clip_time(Fraction(5))


def test_anchor_transform_uses_project_height_units_and_screen_y_boundary() -> None:
    plan = GeometryPlan(
        frame=_frame(),
        window=_window(pre_roll=Fraction(0), post_roll=Fraction(0)),
        transform=_transform(
            position=(10.0, 5.0),
            scale=(2.0, 1.0),
            anchor=(10.0, 0.0),
        ),
    )

    snapshot = plan.snapshot(Fraction(1))

    # At 200x100, one spatial unit is one pixel. Anchor moves the source origin
    # from x=100 to x=110; it does not move the output center. Position then
    # adds (+10, -5) at the pixel boundary.
    expected = ((-110.0, -5.0), (290.0, -5.0), (-110.0, 95.0), (290.0, 95.0))
    for actual_point, expected_point in zip(snapshot.transform_quad, expected):
        assert actual_point == pytest.approx(expected_point)
    assert (
        "perspective=sense=destination:eval=init:interpolation=linear"
        in snapshot.ffmpeg_filters[-1]
    )


def test_rotation_maps_anchor_source_origin_to_output_center() -> None:
    plan = GeometryPlan(
        frame=_frame(project=(100, 100), source=(100, 100)),
        window=_window(pre_roll=Fraction(0), post_roll=Fraction(0)),
        transform=_transform(rotation=90.0, anchor=(50.0, 50.0)),
    )

    quad = plan.snapshot(Fraction(1)).transform_quad

    # anchor=(50, 50) resolves to the top-right source pixel (100, 0). Final
    # Cut maps that source origin to the output center, then rotates the rest.
    assert quad[1] == pytest.approx((50.0, 50.0), abs=1e-9)
    assert quad[0] == pytest.approx((50.0, 150.0), abs=1e-9)


def test_pixel_center_correction_uses_the_full_affine_basis() -> None:
    """Correct rotation/non-uniform scale without a fixed translation guess."""

    quad = (
        (11.0, 7.0),
        (191.0, 67.0),
        (-29.0, 127.0),
        (151.0, 187.0),
    )
    corrected = correct_quad_for_pixel_centers(quad, width=100, height=80)

    # x_axis=(1.8, 0.6), y_axis=(-0.5, 1.5), so the measured half-pixel
    # correction is ((1.8 - 0.5 - 1)/2, (0.6 + 1.5 - 1)/2).
    correction = (0.15, 0.55)
    for observed, source in zip(corrected, quad):
        assert observed == pytest.approx(
            (source[0] + correction[0], source[1] + correction[1])
        )

    identity = ((0.0, 0.0), (100.0, 0.0), (0.0, 80.0), (100.0, 80.0))
    assert correct_quad_for_pixel_centers(identity, width=100, height=80) == identity


@pytest.mark.parametrize("scale", ((0.0, 1.0), (1.0, 0.0)))
def test_zero_scale_is_rejected(scale: tuple[float, float]) -> None:
    with pytest.raises(GeometryValidationError, match="zero scale"):
        GeometryPlan(frame=_frame(), window=_window(), transform=_transform(scale=scale))


def test_scale_sign_change_and_implicit_mirror_are_rejected() -> None:
    changing = RenderTransformAnimation(scale=_vec_track((1.0, 1.0), (-1.0, 1.0)))
    with pytest.raises(GeometryValidationError, match="changes sign"):
        GeometryPlan(
            frame=_frame(),
            window=_window(),
            transform=_transform(),
            transform_animation=changing,
        )
    with pytest.raises(GeometryValidationError, match="allow_mirrored_scale"):
        GeometryPlan(frame=_frame(), window=_window(), transform=_transform(scale=(-1.0, 1.0)))

    mirrored = GeometryPlan(
        frame=_frame(),
        window=_window(),
        transform=_transform(scale=(-1.0, 1.0)),
        allow_mirrored_scale=True,
    )
    assert mirrored.snapshot(Fraction(1)).transform.scale == (-1.0, 1.0)


@pytest.mark.parametrize(
    ("mode", "expected_names"),
    (
        ("crop", ("crop=w=160:h=70:x=10:y=20",)),
        (
            "trim",
            (
                "crop=w=160:h=70:x=10:y=20",
                "pad=w=200:h=100:x=10:y=20:color=black@0",
            ),
        ),
    ),
)
def test_crop_camera_window_and_trim_edges_have_distinct_plan_semantics(
    mode: str,
    expected_names: tuple[str, ...],
) -> None:
    plan = GeometryPlan(
        frame=_frame(project=(100, 100)),
        window=_window(),
        crop=CropAdjustment(
            mode=mode,
            enabled=True,
            rects=(CropRect(left=10, top=20, right=30, bottom=10),),
        ),
    )

    snapshot = plan.snapshot(Fraction(1))
    crop_stage = next(stage for stage in snapshot.stages if stage.name == "crop_trim_pan")

    assert snapshot.crop_rect == PixelRect(x=10, y=20, width=160, height=70)
    assert crop_stage.filters == expected_names
    if mode == "crop":
        # These fragments describe the selected reference window for snapshot
        # diagnostics. The FFmpeg execution seam sees this typed marker and
        # applies the camera transform plus its pre-warp alpha window.
        assert crop_stage.semantics == "camera_reference"
    else:
        assert crop_stage.semantics == "direct_filters"
        assert any(
            "perspective=sense=destination:eval=init:interpolation=linear"
            in fragment
            for fragment in snapshot.ffmpeg_filters
        )


def test_pan_interpolates_two_matching_rectangles_at_exact_clip_time() -> None:
    plan = GeometryPlan(
        frame=_frame(source=(200, 100), project=(100, 100)),
        window=_window(),
        crop=CropAdjustment(
            mode="pan",
            enabled=True,
            rects=(
                CropRect(left=0, top=0, right=50, bottom=0),
                CropRect(left=50, top=0, right=0, bottom=0),
            ),
        ),
    )

    assert plan.snapshot(Fraction(0)).crop_rect == PixelRect(0, 0, 150, 100)
    # The 1080p Final Cut calibration is effectively at 50% at the temporal
    # midpoint, so the height-unit left edge rounds to the center pixel.
    assert plan.snapshot(Fraction(2)).crop_rect == PixelRect(25, 0, 150, 100)
    assert plan.snapshot(Fraction(4)).crop_rect == PixelRect(50, 0, 150, 100)
    with pytest.raises(UnsupportedGeometryAnimationError, match="typed plan"):
        plan.static_ffmpeg_filters()


def test_crop_and_pan_keep_distinct_camera_sizing_contracts() -> None:
    crop = CropAdjustment(
        mode="crop",
        enabled=True,
        rects=(CropRect(left=18, top=6, right=7, bottom=14),),
    )
    frame = _frame(source=(240, 180), project=(320, 180))
    crop_auto = GeometryPlan(frame=frame, window=_window(), crop=crop, conform="fit")
    crop_none = GeometryPlan(frame=frame, window=_window(), crop=crop, conform="none")
    pan = GeometryPlan(
        frame=frame,
        window=_window(),
        crop=CropAdjustment(
            mode="pan",
            enabled=True,
            rects=(
                CropRect(left=2, top=1, right=2, bottom=1),
                CropRect(left=34, top=17, right=5, bottom=6),
            ),
        ),
    )

    assert crop_auto.snapshot(Fraction(1)).camera_placement.conform == "fit"
    assert crop_none.snapshot(Fraction(1)).camera_placement.conform == "fit"
    assert pan.snapshot(Fraction(1)).camera_placement.conform == "fit"
    assert crop_auto.snapshot(Fraction(1)).camera_placement.exact_scale == pytest.approx(
        1.25
    )
    assert pan.snapshot(Fraction(0)).camera_placement.exact_scale == pytest.approx(
        1 / min(232.8 / 240, 176.4 / 180)
    )
    assert pan.snapshot(Fraction(4)).camera_placement.exact_scale == pytest.approx(
        1 / min(169.8 / 240, 138.6 / 180)
    )


def test_horizontal_crop_edges_use_frame_height_and_may_exceed_one_hundred() -> None:
    plan = GeometryPlan(
        frame=_frame(source=(320, 180), project=(320, 180)),
        window=_window(),
        crop=CropAdjustment(
            mode="crop",
            enabled=True,
            rects=(CropRect(left=80, top=0, right=80, bottom=0),),
        ),
    )

    # Both 80-unit edges are legal on a 16:9 frame: each resolves from the
    # 180-pixel frame height, leaving a 32-pixel-wide active rectangle.
    assert plan.snapshot(Fraction(1)).crop_rect == PixelRect(144, 0, 32, 180)


def test_crop_mode_requires_only_its_matching_rectangle_shape() -> None:
    with pytest.raises(GeometryValidationError, match="exactly 1 matching"):
        GeometryPlan(
            frame=_frame(),
            window=_window(),
            crop=CropAdjustment(
                mode="crop",
                enabled=True,
                # This represents the old parser ambiguity: it collected a
                # crop-rect and trim-rect without retaining their element kind.
                rects=(CropRect(0, 0, 0, 0), CropRect(1, 1, 1, 1)),
            ),
        )
    with pytest.raises(GeometryValidationError, match="exactly 2 matching"):
        GeometryPlan(
            frame=_frame(),
            window=_window(),
            crop=CropAdjustment(
                mode="pan",
                enabled=True,
                rects=(CropRect(0, 0, 0, 0),),
            ),
        )


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        (
            "fit",
            (
                "format=rgba64le,lutrgb=r='maxval*pow(val/maxval,1.94)'",
                "pad=w=204:h=104:x=2:y=2:color=black@0",
                "perspective=sense=destination:eval=init:interpolation=linear:x0=50.75",
                "lutrgb=r='maxval*pow(val/maxval,0.515463917526)'",
                "crop=w=100:h=100:x=52:y=2",
            ),
        ),
        (
            "fill",
            (
                "format=rgba64le,lutrgb=r='maxval*pow(val/maxval,1.94)'",
                "pad=w=204:h=104:x=2:y=2:color=black@0",
                "perspective=sense=destination:eval=init:interpolation=linear:x0=0",
                "lutrgb=r='maxval*pow(val/maxval,0.515463917526)'",
                "crop=w=100:h=100:x=52:y=2",
            ),
        ),
        ("none", ("crop=w=100:h=100", "pad=w=100:h=100")),
    ),
)
def test_fit_fill_and_none_produce_a_project_sized_canvas(
    mode: str,
    expected: tuple[str, ...],
) -> None:
    plan = GeometryPlan(
        frame=_frame(source=(200, 100), project=(100, 100)),
        window=_window(),
        conform=mode,
    )

    filters = plan.static_ffmpeg_filters()

    for expected_fragment in expected:
        assert any(expected_fragment in fragment for fragment in filters)
    assert filters[0] == "format=rgba"


@pytest.mark.parametrize("mode", ("fit", "fill"))
def test_conform_identity_does_not_resample_matching_display_raster(mode: str) -> None:
    plan = GeometryPlan(
        frame=_frame(source=(100, 100), project=(100, 100)),
        window=_window(),
        conform=mode,
    )

    snapshot = plan.snapshot(Fraction(0))
    conform_stage = next(stage for stage in snapshot.stages if stage.name == "conform")

    assert conform_stage.filters == ()


def test_corner_offsets_compile_before_anchor_transform() -> None:
    corners = CornerPinAdjustment.from_attributes(
        {
            "topLeft": "10 -5",
            "topRight": "-10 -5",
            "botLeft": "0 5",
            "botRight": "0 5",
        }
    )
    plan = GeometryPlan(
        frame=_frame(project=(200, 100)),
        window=_window(),
        corners=corners,
        transform=_transform(position=(10, 0)),
    )

    snapshot = plan.snapshot(Fraction(1))

    expected = ((10.0, 5.0), (190.0, 5.0), (0.0, 95.0), (200.0, 95.0))
    for actual_point, expected_point in zip(snapshot.corner_quad, expected):
        assert actual_point == pytest.approx(expected_point)
    names = [stage.name for stage in snapshot.stages]
    assert names == [
        "decode_orientation",
        "crop_trim_pan",
        "conform",
        "effects_and_masks",
        "corner_pin",
        "anchor_transform",
    ]
    assert snapshot.stages[3].owner == "external"
    assert len(snapshot.stages[4].filters) == 1
    assert len(snapshot.stages[5].filters) == 1


def test_corner_pin_and_affine_resolve_to_one_unclipped_composed_quad() -> None:
    """Preserve pixels when corner pinning moves them outside before affine."""

    corners = CornerPinAdjustment(
        top_left=(-25.0, 12.0),
        top_right=(8.0, 4.0),
        bottom_left=(-18.0, -10.0),
        bottom_right=(14.0, -7.0),
    )
    transform = _transform(
        position=(17.0, -9.0),
        scale=(0.72, 0.91),
        rotation=11.0,
        anchor=(-6.0, 5.0),
    )
    plan = GeometryPlan(
        frame=_frame(project=(200, 100)),
        window=_window(),
        corners=corners,
        transform=transform,
    )

    snapshot = plan.snapshot(Fraction(1))
    expected = transform_points(plan.frame, snapshot.transform, snapshot.corner_quad)

    for observed, wanted in zip(snapshot.composed_quad, expected):
        assert observed == pytest.approx(wanted, abs=1e-9)
    assert snapshot.composed_quad != snapshot.corner_quad
    assert snapshot.composed_quad != snapshot.transform_quad
    assert len(snapshot.composed_spatial_filters) == 1
    assert snapshot.composed_spatial_filters[0].count("perspective=") == 1
    assert snapshot.render_surface.bounds.left <= min(point[0] for point in expected)
    assert snapshot.render_surface.bounds.right >= max(point[0] for point in expected)


def test_nested_affines_are_child_before_parent_and_never_preclipped() -> None:
    frame = _frame(project=(160, 90), source=(160, 90))
    child = TransformState(
        position=(58.0, 0.0),
        scale=(0.9, 0.9),
        rotation_degrees=18.0,
        anchor=(0.0, 0.0),
    )
    parent = TransformState(
        position=(-42.0, 5.0),
        scale=(0.7, 1.1),
        rotation_degrees=-13.0,
        anchor=(8.0, -4.0),
    )
    canvas = ((0.0, 0.0), (160.0, 0.0), (0.0, 90.0), (160.0, 90.0))

    composed = compose_spatial_quad(
        frame,
        corner_quad=canvas,
        transforms=(child, parent),
    )
    reverse_order = compose_spatial_quad(
        frame,
        corner_quad=canvas,
        transforms=(parent, child),
    )
    child_quad = transform_points(frame, child, canvas)
    surface = render_surface_for_quads(frame, (child_quad, composed))

    assert composed != reverse_order
    # The child leaves the right edge before its parent pulls it back.  The
    # intermediate surface must retain that excursion instead of clipping at x=160.
    assert max(point[0] for point in child_quad) > frame.project_width
    assert surface.bounds.right > frame.project_width
    for point in child_quad:
        round_trip = surface.surface_to_project(surface.project_to_surface(point))
        assert round_trip == pytest.approx(point)


def test_render_surface_unions_all_children_with_one_explicit_guard() -> None:
    frame = _frame(project=(100, 60), source=(100, 60))
    left_child = ((-20.2, 4.0), (40.0, 4.0), (-18.0, 55.0), (42.0, 55.0))
    right_child = ((70.0, -8.1), (133.7, -5.0), (72.0, 49.0), (130.0, 52.0))

    surface = render_surface_for_quads(
        frame,
        (left_child, right_child),
        guard_pixels=2,
    )

    assert surface.origin_x == -23
    assert surface.origin_y == -11
    assert surface.bounds == CanvasBounds(-23.0, -11.0, 136.0, 62.0)


def test_camera_placement_keeps_fractional_reference_in_source_coordinates() -> None:
    frame = _frame(source=(240, 160), project=(320, 180))
    reference = SourceRect(x=12.4, y=7.2, width=199.3, height=141.1)

    placement = resolve_camera_placement(frame, reference, "fit")

    assert placement.reference_rect == reference
    assert placement.reference_quad[0][0] > placement.source_quad[0][0]
    assert placement.reference_quad[0][1] > placement.source_quad[0][1]
    assert placement.reference_quad[3][0] < placement.source_quad[3][0]
    assert placement.reference_quad[3][1] < placement.source_quad[3][1]


@pytest.mark.parametrize(
    "reference",
    (
        SourceRect(x=-0.5, y=0.0, width=100.0, height=50.0),
        SourceRect(x=0.0, y=-0.5, width=100.0, height=50.0),
    ),
)
def test_camera_placement_rejects_negative_origin_by_default(
    reference: SourceRect,
) -> None:
    """Only an explicit Pan caller may request transparent off-source support."""

    with pytest.raises(GeometryValidationError, match="exceeds source"):
        resolve_camera_placement(_frame(), reference, "fit")


def test_crop_camera_rejects_negative_window_without_pan_opt_in() -> None:
    """Do not let Pan's off-source rule silently broaden static Crop."""

    with pytest.raises(GeometryValidationError, match="exceeds source"):
        resolve_crop_camera_placement(
            _frame(),
            SourceRect(x=-1.0, y=0.0, width=200.0, height=100.0),
            "fit",
        )


def test_fill_uses_square_pixel_display_width_after_sar_bake() -> None:
    # Encoded 240x180 with SAR 2:1 becomes a 480x180 square-pixel display
    # raster before geometry. Fill into 320x180 therefore stays 480x180 and
    # crops 80 pixels from both horizontal sides.
    frame = _frame(source=(480, 180), project=(320, 180))
    placement = resolve_camera_placement(
        frame,
        SourceRect(0.0, 0.0, 480.0, 180.0),
        "fill",
    )

    assert (placement.scaled_width, placement.scaled_height) == (480, 180)
    assert (placement.origin_x, placement.origin_y) == (-80, 0)
    assert placement.reference_quad == placement.source_quad


def test_dtd_corner_names_win_over_explicit_legacy_aliases() -> None:
    corners = CornerPinAdjustment.from_attributes(
        {
            "botLeft": "4 5",
            "bottomLeft": "90 90",
            "botRight": "6 7",
            "bottomRight": "80 80",
        }
    )

    assert corners.bottom_left == (4.0, 5.0)
    assert corners.bottom_right == (6.0, 7.0)


def test_animated_corner_pin_uses_typed_timeline_track() -> None:
    corners = CornerPinAdjustment(
        animation=CornerPinAnimation(top_left=_vec_track((0.0, 0.0), (20.0, -10.0)))
    )
    plan = GeometryPlan(frame=_frame(), window=_window(), corners=corners)

    assert plan.snapshot(Fraction(2)).corner_quad[0] == pytest.approx((10.0, 5.0))
    with pytest.raises(UnsupportedGeometryAnimationError):
        plan.static_ffmpeg_filters()


def test_geometry_snapshot_maps_clip_time_through_source_time() -> None:
    source = AnimatedVec2(
        (
            Vec2ControlPoint(Fraction(6), (-38.0, -19.0)),
            Vec2ControlPoint(Fraction(10), (38.0, 19.0)),
        )
    )
    position = TimelineAnimatedVec2(
        source,
        RetimeMap.identity(Fraction(4), source_start=Fraction(6)),
    )
    plan = GeometryPlan(
        frame=_frame(),
        window=_window(
            clip_start=Fraction(0),
            clip_duration=Fraction(4),
            pre_roll=Fraction(0),
            post_roll=Fraction(0),
        ),
        transform=_transform(position=(-38.0, -19.0)),
        transform_animation=RenderTransformAnimation(position=position),
    )

    assert plan.snapshot(Fraction(0)).transform.position == (-38.0, -19.0)
    assert plan.snapshot(Fraction(2)).transform.position == pytest.approx((0.0, 0.0))
    assert plan.snapshot(Fraction(4)).transform.position == (38.0, 19.0)


def _render_raw_rgba(
    source_rgb: bytes,
    *,
    width: int,
    height: int,
    filtergraph: str,
) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is unavailable")
    process = subprocess.run(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            "-vf",
            filtergraph,
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "pipe:1",
        ),
        input=source_rgb,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8", errors="replace")
    return process.stdout


def test_stock_ffmpeg_trim_keeps_canvas_and_makes_edges_transparent() -> None:
    source = bytes((255, 0, 0)) * (8 * 4)
    plan = GeometryPlan(
        frame=_frame(source=(8, 4), project=(8, 4)),
        window=_window(),
        crop=CropAdjustment(
            mode="trim",
            enabled=True,
            rects=(CropRect(left=25, top=0, right=25, bottom=0),),
        ),
        conform="none",
    )

    rendered = _render_raw_rgba(
        source,
        width=8,
        height=4,
        filtergraph=plan.snapshot(Fraction(1)).ffmpeg_filtergraph,
    )
    alpha = rendered[3::4]

    assert len(rendered) == 8 * 4 * 4
    for row in range(4):
        assert alpha[row * 8 : row * 8 + 1] == bytes((0,))
        assert alpha[row * 8 + 1 : row * 8 + 7] == bytes((255, 255, 255, 255, 255, 255))
        assert alpha[row * 8 + 7 : row * 8 + 8] == bytes((0,))


def test_pan_camera_reference_snapshots_follow_height_based_edges() -> None:
    plan = GeometryPlan(
        frame=_frame(source=(8, 4), project=(4, 4)),
        window=_window(),
        crop=CropAdjustment(
            mode="pan",
            enabled=True,
            rects=(
                CropRect(left=0, top=0, right=50, bottom=0),
                CropRect(left=50, top=0, right=0, bottom=0),
            ),
        ),
    )

    first = plan.snapshot(Fraction(0))
    last = plan.snapshot(Fraction(4))
    first_stage = next(stage for stage in first.stages if stage.name == "crop_trim_pan")
    last_stage = next(stage for stage in last.stages if stage.name == "crop_trim_pan")

    assert first.crop_rect == PixelRect(x=0, y=0, width=6, height=4)
    assert last.crop_rect == PixelRect(x=2, y=0, width=6, height=4)
    assert first_stage.semantics == "camera_reference"
    assert last_stage.semantics == "camera_reference"
