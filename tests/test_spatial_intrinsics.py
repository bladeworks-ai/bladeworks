"""Isolated executable contracts for Wave 3 spatial intrinsics."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from bladeworks.core.retime import RetimeMap, RetimePoint
from bladeworks.core.spatial_intrinsics import (
    ColorConform,
    DisplayConform,
    OpaqueCinematicLocator,
    OpaqueTrackerLocator,
    Orientation360,
    Reorientation360,
    RollingShutterAdjustment,
    SpatialIntrinsicPlan,
    SpatialValidationError,
    Stabilization,
    Stereo3DAdjustment,
    TrackerKeyframe,
    Transform360,
    build_spatial_execution_plan,
    build_tracker_animation_hook,
    classify_fcp_color_space,
    probe_stock_ffmpeg_spatial_capabilities,
)


@pytest.mark.parametrize(
    ("rotation", "rotation_filters", "output_dimensions", "required_filters"),
    (
        (0, "", (427, 180), {"null", "scale", "setsar"}),
        (90, ",transpose=clock", (180, 427), {"null", "scale", "setsar", "transpose"}),
        (180, ",hflip,vflip", (427, 180), {"null", "scale", "setsar", "hflip", "vflip"}),
        (270, ",transpose=cclock", (180, 427), {"null", "scale", "setsar", "transpose"}),
    ),
)
def test_display_rotation_bakes_pixel_aspect_into_square_pixel_raster(
    rotation: int,
    rotation_filters: str,
    output_dimensions: tuple[int, int],
    required_filters: set[str],
) -> None:
    execution = build_spatial_execution_plan(
        SpatialIntrinsicPlan(
            frame_width=320,
            frame_height=180,
            display=DisplayConform(
                rotation_degrees=rotation,
                pixel_aspect_h=8,
                pixel_aspect_v=6,
            ),
        )
    )

    display_chain = "scale=427:180:flags=lanczos,setsar=1" + rotation_filters
    assert display_chain in execution.filter_complex
    assert (execution.output_width, execution.output_height) == output_dimensions
    assert set(execution.required_filters) == required_filters


def test_display_pixel_aspect_uses_bounded_positive_half_up_rounding() -> None:
    filters, required, width, height = DisplayConform(
        pixel_aspect_h=1,
        pixel_aspect_v=2,
    ).filters(frame_width=5, frame_height=3)

    assert filters == ("scale=3:3:flags=lanczos", "setsar=1")
    assert required == ("scale", "setsar")
    assert (width, height) == (3, 3)

    with pytest.raises(SpatialValidationError, match="display width must be at most"):
        DisplayConform(pixel_aspect_h=2, pixel_aspect_v=1).filters(
            frame_width=16_384,
            frame_height=180,
        )
    with pytest.raises(SpatialValidationError, match="rounds below one pixel"):
        DisplayConform(pixel_aspect_h=1, pixel_aspect_v=4).filters(
            frame_width=1,
            frame_height=1,
        )


def test_display_metadata_is_typed_before_graph_construction() -> None:

    with pytest.raises(SpatialValidationError, match="multiple of 90"):
        DisplayConform(rotation_degrees=45)
    with pytest.raises(SpatialValidationError, match="positive integer"):
        DisplayConform(pixel_aspect_h=0)


def test_graph_labels_and_dimensions_are_bounded_before_ffmpeg() -> None:
    plan = SpatialIntrinsicPlan(frame_width=320, frame_height=180)
    with pytest.raises(SpatialValidationError, match="letters, digits"):
        build_spatial_execution_plan(plan, input_label="0:v;movie=secret")
    with pytest.raises(SpatialValidationError, match="maximum"):
        SpatialIntrinsicPlan(frame_width=1_000_000, frame_height=180)


def test_360_reorientation_and_tiny_planet_use_stock_v360() -> None:
    execution = build_spatial_execution_plan(
        SpatialIntrinsicPlan(
            frame_width=320,
            frame_height=160,
            reorientation_360=Reorientation360(
                input_projection="equirectangular",
                pan=30,
                tilt=-10,
                roll=5,
            ),
            orientation_360=Orientation360(
                input_projection="equirectangular",
                mapping="tinyPlanet",
                pan=-15,
                field_of_view=120,
                output_width=160,
                output_height=160,
            ),
        )
    )

    assert execution.filter_complex.count("v360=") == 2
    assert "output=e" in execution.filter_complex
    assert "output=sg" in execution.filter_complex
    assert "h_fov=120" in execution.filter_complex
    assert (execution.output_width, execution.output_height) == (160, 160)
    assert "v360" in execution.required_filters
    assert {finding.outcome for finding in execution.findings} == {"approximated"}

    with pytest.raises(SpatialValidationError, match="spherical source"):
        Orientation360(input_projection="none", mapping="tinyPlanet")

    convergence_only = build_spatial_execution_plan(
        SpatialIntrinsicPlan(
            frame_width=320,
            frame_height=160,
            reorientation_360=Reorientation360(
                input_projection="equirectangular", convergence=4
            ),
        )
    )
    assert "v360" not in convergence_only.required_filters
    assert convergence_only.findings[0].code == "spatial.360_convergence_unavailable"


def test_stereo_eye_swap_and_convergence_build_a_label_aware_graph() -> None:
    direct = build_spatial_execution_plan(
        SpatialIntrinsicPlan(
            frame_width=320,
            frame_height=180,
            stereo=Stereo3DAdjustment(
                input_layout="side by side", swap_eyes=True
            ),
        )
    )
    assert "stereo3d=in=sbs2l:out=sbs2r" in direct.filter_complex

    convergence = build_spatial_execution_plan(
        SpatialIntrinsicPlan(
            frame_width=320,
            frame_height=180,
            stereo=Stereo3DAdjustment(
                input_layout="side by side",
                convergence=50,
                auto_scale=False,
            ),
        )
    )
    assert "split=2" in convergence.filter_complex
    assert "hstack=inputs=2" in convergence.filter_complex
    assert "pad=w=160:h=180" in convergence.filter_complex
    assert {"crop", "hstack", "pad", "split", "stereo3d"}.issubset(
        convergence.required_filters
    )
    finding = next(
        item
        for item in convergence.findings
        if item.code == "spatial.stereo_convergence_approximation"
    )
    assert "8-pixel" in finding.detail


def test_color_conform_classification_and_graphs_fail_closed() -> None:
    assert classify_fcp_color_space("1-1-1 (Rec. 709)") == "rec709"
    assert classify_fcp_color_space("9-18-9 (Rec. 2020 HLG)") == "rec2020_hlg"
    assert classify_fcp_color_space("9-16-9 (Rec. 2020 PQ)") == "rec2020_pq"
    assert classify_fcp_color_space("mystery") is None

    with pytest.raises(SpatialValidationError, match="DTD-required"):
        ColorConform.from_attributes({}, source_color_space="1-1-1 (Rec. 709)")

    sdr = build_spatial_execution_plan(
        SpatialIntrinsicPlan(
            frame_width=64,
            frame_height=32,
            color_conform=ColorConform(
                source_color_space="6-1-6 (Rec. 601 (NTSC))"
            ),
        )
    )
    assert "colorspace=iall=smpte170m:all=bt709" in sdr.filter_complex

    hlg = build_spatial_execution_plan(
        SpatialIntrinsicPlan(
            frame_width=64,
            frame_height=32,
            color_conform=ColorConform(
                source_color_space="9-18-9 (Rec. 2020 HLG)"
            ),
        )
    )
    assert "rec2020_hlg_to_rec709_sdr_v1.cube" in hlg.filter_complex
    assert {"format", "lut3d", "setparams"}.issubset(hlg.required_filters)
    assert hlg.findings[0].outcome == "approximated"

    unsupported = build_spatial_execution_plan(
        SpatialIntrinsicPlan(
            frame_width=64,
            frame_height=32,
            color_conform=ColorConform(
                source_color_space="9-18-9 (Rec. 2020 HLG)",
                mode="conformHLGtoPQ",
            ),
        )
    )
    assert "lut3d" not in unsupported.required_filters
    assert unsupported.findings[0].outcome == "not_implemented_yet"

    unknown = build_spatial_execution_plan(
        SpatialIntrinsicPlan(
            frame_width=64,
            frame_height=32,
            color_conform=ColorConform(source_color_space="mystery"),
        )
    )
    assert unknown.findings[0].code == "spatial.color_space_unknown"


def test_frozen_hdr_lut_manifest_matches_checked_in_artifacts() -> None:
    root = Path(__file__).parents[1] / "src" / "bladeworks" / "spatial_luts"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["semantic_approximation"] is True
    for name, expected in manifest["artifacts"].items():
        data = (root / name).read_bytes()
        assert len(data) == expected["bytes"]
        assert hashlib.sha256(data).hexdigest() == expected["sha256"]


@pytest.mark.parametrize(
    ("mode", "fragment"),
    (
        ("automatic", "rx=16:ry=16:blocksize=8:contrast=125:search=less"),
        ("inertiaCam", "rx=32:ry=32:blocksize=16:contrast=100:search=exhaustive"),
        ("smoothCam", "rx=48:ry=48:blocksize=8:contrast=100:search=less"),
    ),
)
def test_stabilization_modes_use_distinct_bounded_deshake_presets(
    mode: str, fragment: str
) -> None:
    execution = build_spatial_execution_plan(
        SpatialIntrinsicPlan(
            frame_width=64,
            frame_height=32,
            stabilization=Stabilization(mode=mode),  # type: ignore[arg-type]
        )
    )
    assert fragment in execution.filter_complex
    assert "filename=" not in execution.filter_complex
    assert execution.findings[0].outcome == "approximated"


def test_opaque_tracker_cinematic_and_rolling_shutter_are_never_silent() -> None:
    execution = build_spatial_execution_plan(
        SpatialIntrinsicPlan(
            frame_width=64,
            frame_height=32,
            transform_360=Transform360.from_attributes(
                {"coordinates": "spherical", "longitude": "20"}
            ),
            rolling_shutter=RollingShutterAdjustment(amount="high"),
            cinematic=OpaqueCinematicLocator(data_locator="depth-1"),
            opaque_trackers=(
                OpaqueTrackerLocator(
                    tracker_id="tracker-1", data_locator="tracker-data-1"
                ),
            ),
        )
    )
    assert {finding.code for finding in execution.findings} == {
        "spatial.rolling_shutter_unavailable",
        "spatial.360_content_transform_unavailable",
        "spatial.cinematic_locator_opaque",
        "spatial.tracker_locator_opaque",
    }
    assert {finding.outcome for finding in execution.findings} == {
        "not_implemented_yet"
    }
    assert execution.manifest()["custom_ffmpeg_required"] is False
    assert execution.manifest()["vulkan_required"] is False


def test_readable_tracker_keyframes_share_exact_reverse_and_freeze_retime() -> None:
    retime = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(2)),
            RetimePoint(Fraction(1), Fraction(1)),
            RetimePoint(Fraction(2), Fraction(1)),
        )
    )
    hook = build_tracker_animation_hook(
        "tracker-1",
        (
            TrackerKeyframe(Fraction(1), (10.0, 20.0), rotation=15),
            TrackerKeyframe(Fraction(2), (30.0, 40.0), rotation=45),
        ),
        retime,
    )

    assert hook.position.value_at(Fraction(0)) == pytest.approx((30.0, 40.0))
    assert hook.position.value_at(Fraction(1)) == pytest.approx((10.0, 20.0))
    assert hook.position.value_at(Fraction(3, 2)) == pytest.approx((10.0, 20.0))
    assert hook.rotation.value_at(Fraction(3, 2)) == pytest.approx(15.0)

    with pytest.raises(SpatialValidationError, match="exact Fraction"):
        TrackerKeyframe(0.1, (0.0, 0.0))  # type: ignore[arg-type]


def _ffmpeg() -> Path:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("ffmpeg is unavailable")
    return Path(executable)


def _make_test_video(path: Path, *, size: str = "64x32", frames: int = 5) -> None:
    subprocess.run(
        (
            str(_ffmpeg()),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={size}:rate=10",
            "-frames:v",
            str(frames),
            "-c:v",
            "ffv1",
            str(path),
        ),
        check=True,
        timeout=20,
    )


@pytest.mark.parametrize(
    "spatial_plan",
    (
        SpatialIntrinsicPlan(
            frame_width=64,
            frame_height=32,
            display=DisplayConform(rotation_degrees=180, pixel_aspect_h=4, pixel_aspect_v=3),
            stabilization=Stabilization(mode="automatic"),
        ),
        SpatialIntrinsicPlan(
            frame_width=64,
            frame_height=32,
            orientation_360=Orientation360(
                input_projection="equirectangular",
                mapping="tinyPlanet",
                output_width=32,
                output_height=32,
            ),
        ),
        SpatialIntrinsicPlan(
            frame_width=64,
            frame_height=32,
            stereo=Stereo3DAdjustment(
                input_layout="side by side", convergence=35, auto_scale=True
            ),
        ),
        SpatialIntrinsicPlan(
            frame_width=64,
            frame_height=32,
            color_conform=ColorConform(
                source_color_space="9-16-9 (Rec. 2020 PQ)",
                mode="conformPQtoSDR",
            ),
        ),
    ),
)
def test_small_real_stock_ffmpeg_spatial_graphs_execute(
    tmp_path: Path, spatial_plan: SpatialIntrinsicPlan
) -> None:
    source = tmp_path / "source.mkv"
    output = tmp_path / "output.nut"
    _make_test_video(source)
    execution = build_spatial_execution_plan(spatial_plan)
    report = probe_stock_ffmpeg_spatial_capabilities(_ffmpeg(), execution)
    report.require_supported()

    subprocess.run(
        (
            *execution.command(
                ffmpeg=_ffmpeg(), input_path=source, output_path=output, frames=3
            )[:-2],
            "-c:v",
            "ffv1",
            "-an",
            str(output),
        ),
        check=True,
        capture_output=True,
        timeout=30,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
