"""Calibrated transfer helpers (Final Cut's measured power-law working space)."""

from __future__ import annotations

import torch

from ..core.compositor import FCP_NORMAL_SOURCE_OVER_GAMMA

GAMMA = float(FCP_NORMAL_SOURCE_OVER_GAMMA)


def linearize(code: torch.Tensor) -> torch.Tensor:
    """Code space (0..1) -> calibrated linear light."""

    return code.clamp(0.0, 1.0).pow(GAMMA)


def encode(linear: torch.Tensor) -> torch.Tensor:
    """Calibrated linear light -> code space (0..1)."""

    return linear.clamp(0.0, 1.0).pow(1.0 / GAMMA)


def unpremultiply(premultiplied: torch.Tensor) -> torch.Tensor:
    """Premultiplied RGBA -> straight RGBA (colour 0 where alpha is 0)."""

    alpha = premultiplied[3:4]
    straight_rgb = torch.where(
        alpha > 0, premultiplied[:3] / alpha.clamp(min=1e-12), torch.zeros_like(premultiplied[:3])
    )
    return torch.cat((straight_rgb, alpha), dim=0)


def premultiplied_to_code(premultiplied_linear: torch.Tensor) -> torch.Tensor:
    """Premultiplied linear RGBA -> straight 0..255 encoded code values.

    This is the ``format=rgba`` / ``gbrap`` domain the CPU reference feeds its 8-bit
    code-space filters and transition modules (``ffmpeg._adapt_transition_side_to_encoded``);
    effect / transition ports round-trip through it.
    """

    straight = unpremultiply(premultiplied_linear)
    rgb_code = encode(straight[:3]) * 255.0
    alpha_code = straight[3:4].clamp(0.0, 1.0) * 255.0
    return torch.cat((rgb_code, alpha_code), dim=0)


def code_to_premultiplied(code_rgba: torch.Tensor) -> torch.Tensor:
    """Straight 0..255 code RGBA -> premultiplied linear RGBA (inverse of ``premultiplied_to_code``)."""

    alpha = code_rgba[3:4] / 255.0
    rgb_linear = linearize(code_rgba[:3] / 255.0)
    return torch.cat((rgb_linear * alpha, alpha), dim=0)
