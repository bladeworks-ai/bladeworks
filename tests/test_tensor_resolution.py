"""Focused product-policy tests for TensorFCP output resolutions."""

from __future__ import annotations

import pytest

from bladeworks.tensor.resolution import (
    RenderMode,
    ResolutionProfile,
    profile_for_mode,
    resolve_output_resolution,
)


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("1080p", (1920, 1080)),
        ("720p", (1280, 720)),
        ("540p", (960, 540)),
        # 854x480 is the envelope. Exact 16:9 content resolves to 852x480
        # because stretching to the envelope would change its aspect ratio.
        ("480p", (852, 480)),
    ],
)
def test_landscape_profiles_fit_their_named_envelopes(
    profile: str, expected: tuple[int, int]
) -> None:
    resolved = resolve_output_resolution(3840, 2160, profile)
    assert (resolved.width, resolved.height) == expected
    assert resolved.was_downscaled


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("1080p", (1080, 1920)),
        ("720p", (720, 1280)),
        ("540p", (540, 960)),
        ("480p", (480, 852)),
    ],
)
def test_portrait_profiles_transpose_the_envelope(
    profile: str, expected: tuple[int, int]
) -> None:
    resolved = resolve_output_resolution(2160, 3840, profile)
    assert (resolved.width, resolved.height) == expected


def test_unusual_aspect_ratio_fits_inside_envelope_without_stretching() -> None:
    resolved = resolve_output_resolution(3840, 1080, "720p")
    assert (resolved.width, resolved.height) == (1280, 360)
    assert resolved.scale_x == pytest.approx(resolved.scale_y)


def test_small_project_is_not_upscaled() -> None:
    resolved = resolve_output_resolution(640, 360, "1080p")
    assert (resolved.width, resolved.height) == (640, 360)
    assert resolved.scale_x == 1.0
    assert resolved.scale_y == 1.0
    assert not resolved.was_downscaled


def test_odd_dimensions_are_floored_for_yuv_without_exceeding_source() -> None:
    resolved = resolve_output_resolution(853, 479, "1080p")
    assert (resolved.width, resolved.height) == (852, 478)
    assert resolved.width % 2 == resolved.height % 2 == 0
    assert resolved.width <= resolved.source_width
    assert resolved.height <= resolved.source_height
    assert resolved.scale_x == pytest.approx(852 / 853)
    assert resolved.scale_y == pytest.approx(478 / 479)


def test_defaults_and_supported_mode_boundaries_are_explicit() -> None:
    assert profile_for_mode(RenderMode.RENDER) is ResolutionProfile.P1080
    assert profile_for_mode(RenderMode.SEEK) is ResolutionProfile.P720
    assert profile_for_mode(RenderMode.SCAN) is ResolutionProfile.P720
    assert profile_for_mode("render", "480p") is ResolutionProfile.P480
    with pytest.raises(ValueError, match="1080p is not supported for seek"):
        profile_for_mode("seek", "1080p")


@pytest.mark.parametrize("dimensions", [(0, 1080), (1920, 1), (-2, 100)])
def test_invalid_source_dimensions_fail_loudly(dimensions: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="at least 2 pixels"):
        resolve_output_resolution(*dimensions, "720p")


def test_profile_rejects_an_aspect_ratio_too_thin_for_even_pixel_output() -> None:
    with pytest.raises(ValueError, match="cannot fit"):
        resolve_output_resolution(2, 10_000, "480p")


def test_non_integer_and_boolean_dimensions_fail_loudly() -> None:
    with pytest.raises(TypeError, match="integers"):
        resolve_output_resolution(1920.0, 1080, "720p")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="not bool"):
        resolve_output_resolution(True, 1080, "720p")  # type: ignore[arg-type]
