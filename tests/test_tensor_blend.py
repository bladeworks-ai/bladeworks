"""Focused mathematical coverage for the tensor FCPXML blend contract."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bladeworks.tensor.blend import composite_layers
from bladeworks.tensor.color import linearize


def _canvas(rgb_code: tuple[float, float, float], alpha: float) -> torch.Tensor:
    rgb = torch.tensor(rgb_code, dtype=torch.float32).view(3, 1, 1)
    alpha_tensor = torch.full((1, 1, 1), alpha, dtype=torch.float32)
    return torch.cat((linearize(rgb) * alpha_tensor, alpha_tensor), dim=0)


def _straight(canvas: torch.Tensor) -> torch.Tensor:
    alpha = canvas[3:4]
    return torch.cat((canvas[:3] / alpha.clamp_min(1.0e-8), alpha), dim=0)


def test_multiply_is_encoded_space_and_alpha_aware() -> None:
    lower = _canvas((100 / 255, 200 / 255, 50 / 255), 128 / 255)
    upper = _canvas((200 / 255, 100 / 255, 250 / 255), 128 / 255)

    result = composite_layers(lower, upper, "Multiply")
    lower_straight = _straight(lower)
    upper_straight = _straight(upper)
    lower_alpha, upper_alpha = lower[3:4], upper[3:4]
    upper_code = upper_straight[:3].pow(1.0 / 1.94)
    lower_code = lower_straight[:3].pow(1.0 / 1.94)
    blended_code = upper_code * lower_code
    selected_code = (
        upper_code * (1.0 - lower_alpha)
        + blended_code * lower_alpha
    )
    output_alpha = upper_alpha + lower_alpha * (1.0 - upper_alpha)
    expected_rgb = (
        selected_code.pow(1.94) * upper_alpha
        + lower[:3] * (1.0 - upper_alpha)
    )
    expected = torch.cat((expected_rgb, output_alpha), dim=0)
    torch.testing.assert_close(result, expected, atol=2.0e-6, rtol=0.0)


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("Subtract", (0.55, 0.0, 0.0)),
        ("Divide", (1.0, 0.5, 0.625)),
        ("Overlay", (0.79, 0.16, 0.8)),
        ("Hard Light", (0.51, 0.16, 0.8)),
        ("Soft Light", (0.799, 0.168, 0.65)),
        ("Pin Light", (0.6, 0.2, 0.6)),
    ),
)
def test_non_symmetric_modes_follow_stock_ffmpeg_argument_order(
    mode: str, expected: tuple[float, float, float]
) -> None:
    lower = _canvas((0.3, 0.4, 0.8), 1.0)
    upper = _canvas((0.85, 0.2, 0.5), 1.0)

    result = composite_layers(lower, upper, mode)
    result_code = result[:3].clamp_min(0.0).pow(1.0 / 1.94)
    torch.testing.assert_close(
        result_code,
        torch.tensor(expected, dtype=torch.float32).view(3, 1, 1),
        atol=2.0e-6,
        rtol=0.0,
    )


def test_rgb_mode_reveals_unblended_foreground_over_transparent_lower() -> None:
    lower = _canvas((0.0, 1.0, 0.0), 0.0)
    upper = _canvas((1.0, 0.25, 0.0), 0.8)

    result = composite_layers(lower, upper, "Multiply")
    torch.testing.assert_close(result, upper, atol=2.0e-6, rtol=0.0)


@pytest.mark.parametrize(
    "mode",
    (
        "Add", "Subtract", "Darken", "Lighten", "Multiply", "Screen",
        "Overlay", "Soft Light", "Hard Light", "Difference", "Exclusion",
        "Color Burn", "Color Dodge", "Divide", "Linear Light", "Pin Light",
        "Hard Mix",
    ),
)
def test_every_reviewed_rgb_mode_is_finite_and_preserves_alpha_contract(mode: str) -> None:
    lower = _canvas((0.17, 0.63, 0.91), 0.37)
    upper = _canvas((0.82, 0.29, 0.44), 0.61)

    result = composite_layers(lower, upper, mode)
    expected_alpha = 0.61 + 0.37 * (1.0 - 0.61)
    assert torch.isfinite(result).all()
    assert result[:3].min() >= 0.0 and result[:3].max() <= 1.0
    assert result[3, 0, 0].item() == pytest.approx(expected_alpha)


@pytest.mark.parametrize(
    ("mode", "expected_alpha"),
    (("Stencil Alpha", 0.37 * 0.61), ("Silhouette Alpha", 0.37 * (1.0 - 0.61))),
)
def test_alpha_mattes_change_only_the_lower_coverage(mode: str, expected_alpha: float) -> None:
    lower = _canvas((0.12, 0.42, 0.85), 0.37)
    upper = _canvas((0.82, 0.29, 0.44), 0.61)

    result = composite_layers(lower, upper, mode)
    expected_rgb = _straight(lower)[:3] * expected_alpha
    torch.testing.assert_close(result[:3], expected_rgb, atol=2.0e-6, rtol=0.0)
    assert result[3, 0, 0].item() == pytest.approx(expected_alpha)


@pytest.mark.parametrize("mode,invert", (("Stencil Luma", False), ("Silhouette Luma", True)))
def test_luma_mattes_use_foreground_luma_and_opacity(mode: str, invert: bool) -> None:
    lower = _canvas((0.12, 0.42, 0.85), 0.37)
    upper = _canvas((0.82, 0.29, 0.44), 0.61)

    result = composite_layers(lower, upper, mode)
    luma = 0.299 * 0.82 + 0.587 * 0.29 + 0.114 * 0.44
    matte = luma * 0.61
    if invert:
        matte = 1.0 - matte
    expected_alpha = 0.37 * matte
    torch.testing.assert_close(
        result[:3],
        _straight(lower)[:3] * expected_alpha,
        atol=2.0e-6,
        rtol=0.0,
    )
    assert result[3, 0, 0].item() == pytest.approx(expected_alpha)


def test_behind_places_the_new_layer_below_the_existing_canvas() -> None:
    lower = _canvas((0.12, 0.42, 0.85), 0.37)
    upper = _canvas((0.82, 0.29, 0.44), 0.61)

    result = composite_layers(lower, upper, "Behind")
    expected = composite_layers(upper, lower, "Normal")
    torch.testing.assert_close(result, expected, atol=2.0e-6, rtol=0.0)


def test_group_surface_uses_the_same_blend_operation_as_a_layer() -> None:
    lower = _canvas((0.2, 0.4, 0.8), 1.0)
    child = _canvas((0.9, 0.1, 0.2), 0.6)

    # This is the renderer's group shape: compose children on a transparent
    # scope surface, then place that finished surface on the parent with the
    # scope's own blend mode.
    group_surface = composite_layers(torch.zeros_like(lower), child, "Normal")
    grouped = composite_layers(lower, group_surface, "Screen")
    direct = composite_layers(lower, child, "Screen")
    torch.testing.assert_close(grouped, direct, atol=2.0e-6, rtol=0.0)


def test_uncalibrated_cross_channel_modes_remain_loud_rejects() -> None:
    lower = _canvas((0.2, 0.4, 0.8), 1.0)
    upper = _canvas((0.9, 0.1, 0.2), 1.0)

    with pytest.raises(ValueError, match="known but not implemented"):
        composite_layers(lower, upper, "Hue")
