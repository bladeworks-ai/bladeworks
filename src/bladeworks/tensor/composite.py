"""Calibrated source-over on premultiplied linear RGBA tensors."""

from __future__ import annotations

import torch


def over(background: torch.Tensor, foreground: torch.Tensor) -> torch.Tensor:
    """Premultiplied source-over: fg + bg * (1 - fg.alpha).  Both [4, H, W]."""

    alpha = foreground[3:4]
    return foreground + background * (1.0 - alpha)


def opaque_black(height: int, width: int, *, like: torch.Tensor) -> torch.Tensor:
    canvas = like.new_zeros((4, height, width))
    canvas[3] = 1.0
    return canvas


def unpremultiply(canvas: torch.Tensor) -> torch.Tensor:
    """Premultiplied -> straight RGBA (colour is 0 where alpha is 0).  ``[4, H, W]``.

    Why this exists: the reference re-samples a finished group surface as
    *straight* RGBA (``_group_video_chain``: the composed pad is adapted to
    encoded straight 8-bit before ``format=rgba`` and the linear-light
    ``perspective``), so a rendered scope is unpremultiplied before its
    placement warp and premultiplied again after it, like a decoded leaf.
    """

    alpha = canvas[3:4]
    safe = torch.where(alpha > 0.0, alpha, torch.ones_like(alpha))
    return torch.cat((canvas[:3] / safe, alpha), dim=0)
