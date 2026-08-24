"""Portable tensor implementation of the documented Green Screen Keyer.

Architecture map
================

    compiler ``GreenScreenKeyerSettings.as_data``
        -> ``lower_green_screen_keyer``: strict typed payload
        -> ``green_screen_key``: encoded straight RGB colorkey + despill
        -> premultiplied linear RGBA returned to the effect stack

This is deliberately the same bounded approximation described by
``GREEN_SCREEN_KEYER.md`` and the CPU FFmpeg path: a Euclidean RGB colorkey,
green/blue despill, and an alpha mix.  It is not Apple's private Keyer math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

from ..core.model import ResolvedEffect
from .color import code_to_premultiplied, premultiplied_to_code
from .support import reject


@dataclass(frozen=True)
class GreenScreenKeyerPayload:
    key_color: tuple[float, float, float]
    softness: float
    strength: float
    spill_level: float
    chroma_rolloff: float
    luma_rolloff: float
    green_chroma: float
    blue_chroma: float
    min_green: float
    max_green: float
    min_blue: float
    max_blue: float
    mix: float


def lower_green_screen_keyer(effect: ResolvedEffect, _ctx: object) -> GreenScreenKeyerPayload:
    """Freeze every keyer scalar from compiler-owned data and reject omissions.

    Main callers: ``tensor.effects.lower_effect``.

    Why this exists: the compiler has decoded opaque FCPXML payloads into
    explicit strings.  A tensor plan must still verify that boundary, because
    a hand-built or corrupted ``ResolvedEffect`` must not silently become an
    identity effect.
    """

    try:
        color = _numbers(effect.data, "key_color", 3, 0.0, 1.0)
        values = {
            name: _scalar(effect.data, name, minimum, maximum)
            for name, minimum, maximum in (
                ("softness", 0.0, 20.0), ("strength", 0.0, 2.0),
                ("spill_level", 0.0, 1.0), ("chroma_rolloff", 0.0, 1.0),
                ("luma_rolloff", 0.0, 1.0), ("green_chroma", -10.0, 10.0),
                ("blue_chroma", -10.0, 10.0), ("min_green", -10.0, 10.0),
                ("max_green", -10.0, 10.0), ("min_blue", -10.0, 10.0),
                ("max_blue", -10.0, 10.0), ("mix", 0.0, 1.0),
            )
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise reject("effect (unsupported parameters)", f"{effect.path}: invalid Green Screen Keyer data: {exc}") from exc
    return GreenScreenKeyerPayload(key_color=color, **values)


def green_screen_key(canvas: torch.Tensor, payload: GreenScreenKeyerPayload) -> torch.Tensor:
    """Apply the bounded colorkey/despill approximation to premultiplied RGBA.

    The CPU filters run in straight encoded RGBA.  We therefore unassociate and
    encode before measuring color distance, keep the source alpha in the
    colorkey result, then convert the edited straight image back to the tensor
    renderer's premultiplied linear working space.
    """

    code = premultiplied_to_code(canvas)
    if payload.mix <= 0.0:
        return canvas
    rgb = code[:3] / 255.0
    key = canvas.new_tensor(payload.key_color).reshape(3, 1, 1)
    graph_width = min(
        20.0,
        abs(payload.max_green - payload.min_green)
        + abs(payload.max_blue - payload.min_blue)
        + abs(payload.green_chroma)
        + abs(payload.blue_chroma),
    )
    similarity = max(0.00001, min(1.0, 0.025 + 0.08 * (payload.strength / 2.0) + 0.06 * payload.chroma_rolloff + 0.002 * graph_width))
    blend = max(0.0, min(1.0, 0.005 + 0.22 * (payload.softness / 20.0) + 0.08 * payload.luma_rolloff))
    # FFmpeg colorkey defines similarity against normalized RMS distance
    # across the three encoded RGB channels, not raw Euclidean distance.
    distance = ((rgb - key).square().sum(dim=0) / 3.0).sqrt()
    if blend <= 0.0:
        key_alpha = (distance > similarity).to(canvas.dtype)
    else:
        key_alpha = ((distance - similarity) / blend).clamp(0.0, 1.0)
    source_alpha = code[3] / 255.0
    screen_is_green = payload.key_color[1] >= payload.key_color[2]
    if payload.spill_level > 0.0:
        rgb = _despill(rgb, screen_channel=1 if screen_is_green else 2, mix=payload.spill_level)
    # Keep pre-existing transparency transparent while mixing between the
    # original and keyed alpha. This is the premultiplied tensor equivalent
    # of applying the key at this ordered effect boundary.
    alpha = source_alpha * ((1.0 - payload.mix) + key_alpha * payload.mix)
    return code_to_premultiplied(torch.cat((rgb * 255.0, alpha.unsqueeze(0) * 255.0), dim=0))


def _despill(rgb: torch.Tensor, *, screen_channel: int, mix: float) -> torch.Tensor:
    # Port FFmpeg despill's defaults: expand=0, screen scale=-1, the other
    # channel scales and brightness=0. The mix controls how red versus the
    # third channel contributes to the spill map.
    red = rgb[0]
    other = rgb[2] if screen_channel == 1 else rgb[1]
    spill = (rgb[screen_channel] - (red * mix + other * (1.0 - mix))).clamp_min(0.0)
    result = rgb.clone()
    result[screen_channel] = (rgb[screen_channel] - spill).clamp_min(0.0)
    return result.clamp(0.0, 1.0)


def _scalar(data: Mapping[str, str], key: str, minimum: float, maximum: float) -> float:
    raw = data[key]
    value = float(raw)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{key}={raw!r} outside [{minimum}, {maximum}]")
    return value


def _numbers(data: Mapping[str, str], key: str, count: int, minimum: float, maximum: float) -> tuple[float, ...]:
    values = tuple(float(piece) for piece in data[key].replace(",", " ").split())
    if len(values) != count or not all(math.isfinite(value) and minimum <= value <= maximum for value in values):
        raise ValueError(f"{key} is not {count} bounded numeric components")
    return values
