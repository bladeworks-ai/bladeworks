"""Tensor sampler: the fused homography lands where the reference ``perspective`` lands.

For canned static geometry (transform with anchor / rotation / non-uniform
scale / position; corner pin + scale; identity) the same ``GeometryPlan``
snapshot is executed two ways on one synthetic RGBA plate:

* reference: the snapshot's own FFmpeg fragments (``reference`` profile:
  linear-light 16-bit ``perspective`` for conform, then the composed corner
  pin + affine ``perspective``) run through the ffmpeg CLI;
* tensor: ``sampler.layer_homography`` -> ``grid_sample``.

Gates: the centroid of a bright marker lands within 0.5 px in both outputs
(the corner-error test), the visible-pixel footprint agrees, and the mean
absolute difference over the frame is small.  Skips without torch / ffmpeg.
"""

from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from bladeworks.core.geometry import (  # noqa: E402
    CornerPinAdjustment,
    FrameGeometry,
    GeometryPlan,
    GeometryWindow,
)
from bladeworks.core.model import TransformAdjustment  # noqa: E402
from bladeworks.tensor import sampler  # noqa: E402
from bladeworks.tensor.color import encode, linearize  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg")

SOURCE = (480, 320)
PROJECT = (640, 360)
MARKER = (60, 44, 8)  # x, y, size of a white marker on a mid-grey opaque plate


def _plate() -> np.ndarray:
    width, height = SOURCE
    plate = np.zeros((height, width, 4), dtype=np.uint8)
    plate[..., :3] = 96
    plate[..., 3] = 255
    yy, xx = np.mgrid[0:height, 0:width]
    plate[((xx // 40 + yy // 40) % 2 == 0), :3] = 128
    x, y, size = MARKER
    plate[y:y + size, x:x + size, :3] = 255
    return plate


def _plan(*, transform: TransformAdjustment | None, corners: CornerPinAdjustment | None, conform: str = "fit") -> GeometryPlan:
    return GeometryPlan(
        frame=FrameGeometry(source_width=SOURCE[0], source_height=SOURCE[1], project_width=PROJECT[0], project_height=PROJECT[1]),
        window=GeometryWindow(clip_start=Fraction(0), clip_duration=Fraction(2), render_start=Fraction(0), render_duration=Fraction(2)),
        transform=transform,
        transform_animation=None,
        crop=None,
        conform=conform,
        corners=corners,
    )


def _reference(tmp_path: Path, plate: np.ndarray, plan: GeometryPlan) -> np.ndarray:
    snapshot = plan.snapshot(Fraction(0))
    stage = {stage.name: stage.filters for stage in snapshot.stages}
    filters = (
        *stage["decode_orientation"],
        *stage["conform"],
        *snapshot.composed_spatial_filters,
        "format=rgba",
    )
    source = tmp_path / "plate.png"
    Image.fromarray(plate, "RGBA").save(source)
    output = tmp_path / "reference.png"
    subprocess.run(
        (FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-vf", ",".join(filters),
         "-frames:v", "1", str(output)),
        check=True,
    )
    return np.asarray(Image.open(output).convert("RGBA"))


def _tensor(plate: np.ndarray, plan: GeometryPlan) -> np.ndarray:
    snapshot = plan.snapshot(Fraction(0))
    frame = plan.frame
    rgba = torch.from_numpy(plate).permute(2, 0, 1).float() / 255.0
    straight = torch.cat((linearize(rgba[:3]), rgba[3:4]), dim=0)
    homography = sampler.layer_homography(
        snapshot, frame=frame, conform=plan.conform, canvas_to_project=sampler.identity_matrix()
    )
    grid = sampler.sampling_grid(
        homography,
        centres=sampler.GridCache().centres(PROJECT[1], PROJECT[0], torch.device("cpu")),
        out_height=PROJECT[1], out_width=PROJECT[0], source_height=SOURCE[1], source_width=SOURCE[0],
    )
    warped = sampler.warp(straight, grid)
    out = torch.cat((encode(warped[:3]), warped[3:4]), dim=0)
    return (out * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()


def _centroid(image: np.ndarray) -> tuple[float, float]:
    luma = image[..., :3].astype(np.float64).mean(axis=-1) * (image[..., 3] / 255.0)
    mask = luma > 200.0
    assert mask.any(), "marker not visible"
    ys, xs = np.nonzero(mask)
    weights = luma[mask]
    return float((xs * weights).sum() / weights.sum()), float((ys * weights).sum() / weights.sum())


CASES = {
    "transform_anchor_rotation": _plan(
        transform=TransformAdjustment(position=(19.0, -13.0), scale=(0.84, 1.16), rotation=27.0, enabled=True, anchor=(12.0, -8.0)),
        corners=None,
    ),
    "corner_pin_scaled": _plan(
        transform=TransformAdjustment(position=(0.0, 0.0), scale=(0.9, 0.9), rotation=0.0, enabled=True, anchor=(0.0, 0.0)),
        corners=CornerPinAdjustment(top_left=(-13.0, 9.0), top_right=(0.0, 0.0), bottom_left=(0.0, 0.0), bottom_right=(15.0, -7.0)),
    ),
    "fill_position": _plan(
        transform=TransformAdjustment(position=(-12.0, -11.0), scale=(1.0, 1.0), rotation=0.0, enabled=True, anchor=(0.0, 0.0)),
        corners=None,
        conform="fill",
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_marker_lands_within_half_a_pixel(tmp_path: Path, name: str) -> None:
    plan = CASES[name]
    plate = _plate()
    reference = _reference(tmp_path, plate, plan)
    result = _tensor(plate, plan)
    assert reference.shape == result.shape == (PROJECT[1], PROJECT[0], 4)
    ref_centroid = _centroid(reference)
    out_centroid = _centroid(result)
    distance = float(np.hypot(ref_centroid[0] - out_centroid[0], ref_centroid[1] - out_centroid[1]))
    visible_ref = (reference[..., 3] > 127).sum()
    visible_out = (result[..., 3] > 127).sum()
    diff = np.abs(reference[..., :3].astype(np.int16) - result[..., :3].astype(np.int16)).mean()
    print(f"{name}: centroid distance {distance:.3f}px, footprint {visible_ref} vs {visible_out}, mean abs diff {diff:.2f}")
    assert distance <= 0.5, f"{name}: marker centroid off by {distance:.3f}px ({ref_centroid} vs {out_centroid})"
    if plan.conform == "fill":
        # The snapshot's staged fragments conform-then-transform, so Fill
        # overscan is cropped before the affine; the builder's fused
        # ``_static_conform_affine_filter`` (and this sampler) keep the overscan
        # and let the affine move it back into view.  Only placement is compared.
        return
    assert abs(visible_ref - visible_out) <= 0.01 * max(visible_ref, 1), f"{name}: footprint {visible_ref} vs {visible_out}"
    # Reference resamples twice (conform, then composed warp) with a 16-bit
    # round trip; the tensor path resamples once.  Content differs by resample
    # blur only, so the frame-mean difference stays a few codes.
    assert diff <= 4.0, f"{name}: mean abs diff {diff:.2f} codes"


def test_identity_conform_is_exact() -> None:
    """Equal source/project size with Fit and no transform must copy pixels exactly."""

    plate = _plate()
    plan = GeometryPlan(
        frame=FrameGeometry(source_width=SOURCE[0], source_height=SOURCE[1], project_width=SOURCE[0], project_height=SOURCE[1]),
        window=GeometryWindow(clip_start=Fraction(0), clip_duration=Fraction(2), render_start=Fraction(0), render_duration=Fraction(2)),
        transform=None, transform_animation=None, crop=None, conform="fit", corners=None,
    )
    snapshot = plan.snapshot(Fraction(0))
    homography = sampler.layer_homography(snapshot, frame=plan.frame, conform="fit", canvas_to_project=sampler.identity_matrix())
    assert sampler.is_identity(homography)
    rgba = torch.from_numpy(plate).permute(2, 0, 1).float() / 255.0
    grid = sampler.sampling_grid(
        homography, centres=sampler.GridCache().centres(SOURCE[1], SOURCE[0], torch.device("cpu")),
        out_height=SOURCE[1], out_width=SOURCE[0], source_height=SOURCE[1], source_width=SOURCE[0],
    )
    out = (sampler.warp(rgba, grid) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    assert np.array_equal(out, plate)


def test_alpha_windows_follow_the_reference_index_rule() -> None:
    frame = FrameGeometry(source_width=100, source_height=50, project_width=100, project_height=50)
    plan = GeometryPlan(
        frame=frame,
        window=GeometryWindow(clip_start=Fraction(0), clip_duration=Fraction(1), render_start=Fraction(0), render_duration=Fraction(1)),
        transform=None, transform_animation=None, crop=None, conform="none", corners=None,
    )
    snapshot = plan.snapshot(Fraction(0))
    # No crop and a source that fits: no window at all.
    assert sampler.source_alpha_window(snapshot, frame=frame, conform="none", crop_mode=None) is None
    # Conform none with a wider source: crop to the project, centred by truncation.
    wide = FrameGeometry(source_width=131, source_height=50, project_width=100, project_height=50)
    assert sampler.source_alpha_window(snapshot, frame=wide, conform="none", crop_mode=None) == (15, 115, 0, 50)
    assert sampler.conform_matrix(snapshot, wide, "none")[0, 2] == -15.0


def test_exact_aspect_opaque_minification_matches_swscale_bilinear() -> None:
    """Sharp encoded content matches the installed legacy swscale oracle within one code."""

    height, width = 72, 128
    yy, xx = np.mgrid[0:height, 0:width]
    # Deliberately stress phase, support, chroma pair reduction, coefficient
    # quantisation, and border folding. Generic antialiased interpolation misses
    # this witness by many codes even though it passes a smooth-ramp comparison.
    rgb = np.stack(
        (
            ((xx * 37 + yy * 17) ^ ((xx // 2) * 91)) & 255,
            ((xx + yy) & 1) * 255,
            ((xx * 73) ^ (yy * 151)) & 255,
        ),
        axis=-1,
    ).astype(np.uint8)
    rgba = np.concatenate((rgb, np.full((height, width, 1), 255, dtype=np.uint8)), axis=-1)
    process = subprocess.run(
        (
            FFMPEG,
            "-v", "error",
            "-f", "rawvideo",
            "-pixel_format", "rgba",
            "-video_size", f"{width}x{height}",
            "-i", "pipe:0",
            "-vf", "scale=32:18:flags=bilinear",
            "-frames:v", "1",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "pipe:1",
        ),
        input=rgba.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    reference = np.frombuffer(process.stdout, dtype=np.uint8).reshape(18, 32, 4)[..., :3]
    source = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    observed = (
        sampler.resize_exact_aspect_opaque(source, height=18, width=32)
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    difference = np.abs(observed.astype(np.int16) - reference.astype(np.int16))
    print(
        "legacy swscale bilinear parity: "
        f"max={difference.max()} mean={difference.mean():.6f} "
        f"p99={np.percentile(difference, 99):.3f} over_one={(difference > 1).sum()}"
    )
    assert difference.max() <= 1


def test_exact_aspect_minification_selection_is_strict() -> None:
    frame = FrameGeometry(
        source_width=1920,
        source_height=1080,
        project_width=640,
        project_height=360,
    )
    plan = GeometryPlan(
        frame=frame,
        window=GeometryWindow(
            clip_start=Fraction(0),
            clip_duration=Fraction(1),
            render_start=Fraction(0),
            render_duration=Fraction(1),
        ),
        transform=None,
        transform_animation=None,
        crop=None,
        conform="fit",
        corners=None,
    )
    snapshot = plan.snapshot(Fraction(0))
    assert sampler.uses_exact_aspect_minification(
        snapshot,
        frame=frame,
        conform="fit",
        crop_mode=None,
        source_is_opaque=True,
    )
    assert not sampler.uses_exact_aspect_minification(
        snapshot,
        frame=frame,
        conform="fit",
        crop_mode=None,
        source_is_opaque=False,
    )
    mismatched = FrameGeometry(
        source_width=1920,
        source_height=1080,
        project_width=360,
        project_height=640,
    )
    assert not sampler.uses_exact_aspect_minification(
        snapshot,
        frame=mismatched,
        conform="fit",
        crop_mode=None,
        source_is_opaque=True,
    )
    tiny = FrameGeometry(
        source_width=1920,
        source_height=1920,
        project_width=2,
        project_height=2,
    )
    assert not sampler.uses_exact_aspect_minification(
        snapshot,
        frame=tiny,
        conform="fit",
        crop_mode=None,
        source_is_opaque=True,
    )
