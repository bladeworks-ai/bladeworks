"""Tensor mattes for the reviewed portable FCPXML mask subset.

Architecture map
================

    ResolvedMaskGroup
        -> ``matte_for_group``: shape / draw / explicit color / luma planes
        -> ``apply_masked_effect``: source alpha * matte, then inside over outside

The compiler is the trust boundary.  It has already rejected tracking payloads,
Apple keyed archives, non-convex polygons, and unbounded numeric values.  This
module only evaluates the typed values that remain in ``ResolvedMask`` and
never treats missing data as an implicit mask.

Working-space invariant
-----------------------

The renderer carries premultiplied linear RGBA.  Shape and draw mattes are
geometric alpha planes.  Color and luma mattes use straight encoded RGB, like
the CPU ``format=rgba`` mask path, and the resulting matte is applied to the
premultiplied branch.  This preserves correct fractional alpha at the mask
edge and keeps group/leaf composition on the existing ``over`` operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Optional

import torch

from ..core.model import Parameter, ResolvedMask, ResolvedMaskGroup
from ..core.retime import RetimeMap
from .color import premultiplied_to_code
from .composite import over


@dataclass(frozen=True)
class MaskEffectPayload:
    """A lowered masked effect and its optional outside branch.

    ``inside`` and ``outside`` are tensor ``EffectSpec`` objects, kept as
    ``Any`` here to avoid a registry import cycle.  The owning effects module
    constructs and applies them through its normal port dispatch.
    """

    group: ResolvedMaskGroup
    inside: Any
    outside: Optional[Any]
    source_start: Fraction = Fraction(0)
    playback_rate: Fraction = Fraction(1)
    retime_map: Optional[RetimeMap] = None
    coordinate_scale_x: float = 1.0
    coordinate_scale_y: float = 1.0


def apply_masked_effect(
    payload: MaskEffectPayload,
    canvas: torch.Tensor,
    *,
    frame: int,
    seconds: float = 0.0,
    apply_effect: Callable[[Any, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """Apply a masked effect in the same order as the CPU reference.

    Pythonese:
    1. Build the combined matte from the unmodified input.  The mask belongs
       to the effect boundary, so it is not generated from the already
       effected inside branch.
    2. Run the inside effect and multiply every premultiplied channel,
       including alpha, by the matte.
    3. Run the optional outside effect on the original input.  With no outside
       filter, the outside branch is the original input.
    4. Source-over the masked inside branch over the outside branch.

    Main callers: ``tensor.effects.apply_effects``.

    Why this exists: a masked filter is not a crop.  Both branches must retain
    their full canvas so pixels outside the matte can receive the optional
    outside correction and transparent output remains transparent.
    """

    if payload.retime_map is None:
        source_seconds = float(payload.source_start) + seconds * float(payload.playback_rate)
    else:
        # Effect runtime time is local output time. The exact map preserves
        # reverse, freeze, and variable-rate segments instead of reducing them
        # to one average playback rate.
        local_time = Fraction(str(seconds)).limit_denominator(1_000_000_000)
        source_seconds = float(payload.retime_map.map_timeline(local_time))
    matte = matte_for_group(
        payload.group,
        canvas,
        seconds=source_seconds,
        coordinate_scale_x=payload.coordinate_scale_x,
        coordinate_scale_y=payload.coordinate_scale_y,
    )
    inside = apply_effect(payload.inside, canvas, frame)
    masked_inside = inside * matte.unsqueeze(0)
    outside = canvas if payload.outside is None else apply_effect(payload.outside, canvas, frame)
    return over(outside, masked_inside)


def matte_for_group(
    group: ResolvedMaskGroup,
    canvas: torch.Tensor,
    *,
    seconds: float = 0.0,
    coordinate_scale_x: float = 1.0,
    coordinate_scale_y: float = 1.0,
) -> torch.Tensor:
    """Return one validated combined matte as ``[H, W]`` float alpha.

    Main callers:
    - ``apply_masked_effect``.
    - Focused tensor kernel tests.

    ``add`` is maximum, ``subtract`` removes the next matte from the current
    matte, and ``multiply`` intersects the two.  Inversion happens after the
    full ordered combination, matching the FCPXML container flag.
    """

    if canvas.ndim != 3 or canvas.shape[0] != 4:
        raise ValueError(f"mask input must be [4,H,W] RGBA, got {tuple(canvas.shape)}")
    if not group.masks:
        raise ValueError("mask group must contain at least one resolved mask")
    result: Optional[torch.Tensor] = None
    for mask in group.masks:
        current = _mask_plane(
            mask,
            canvas,
            seconds=seconds,
            coordinate_scale_x=coordinate_scale_x,
            coordinate_scale_y=coordinate_scale_y,
        )
        if result is None:
            result = current
        elif mask.blend_mode == "add":
            result = torch.maximum(result, current)
        elif mask.blend_mode == "subtract":
            result = result * (1.0 - current)
        elif mask.blend_mode == "multiply":
            result = result * current
        else:
            # The compiler currently guarantees this set.  Keeping this error
            # makes a corrupted plan loud instead of silently changing alpha.
            raise ValueError(f"unsupported resolved mask blend mode {mask.blend_mode!r}")
    assert result is not None
    return 1.0 - result if group.inverted else result


def _mask_plane(
    mask: ResolvedMask,
    canvas: torch.Tensor,
    *,
    seconds: float,
    coordinate_scale_x: float,
    coordinate_scale_y: float,
) -> torch.Tensor:
    height, width = int(canvas.shape[1]), int(canvas.shape[2])
    if mask.kind == "shape":
        return _shape_matte(
            mask,
            height,
            width,
            canvas,
            seconds=seconds,
            coordinate_scale_x=coordinate_scale_x,
            coordinate_scale_y=coordinate_scale_y,
        )
    if mask.kind == "draw":
        return _draw_matte(
            mask,
            height,
            width,
            canvas,
            seconds=seconds,
            coordinate_scale_x=coordinate_scale_x,
            coordinate_scale_y=coordinate_scale_y,
        )
    if mask.kind == "color":
        return _color_matte(mask, canvas)
    if mask.kind == "range":
        return _range_matte(mask, canvas)
    raise ValueError(f"unsupported resolved mask kind {mask.kind!r}")


def _shape_matte(
    mask: ResolvedMask,
    height: int,
    width: int,
    canvas: torch.Tensor,
    *,
    seconds: float,
    coordinate_scale_x: float,
    coordinate_scale_y: float,
) -> torch.Tensor:
    radius_x = _parameter(mask, ("160", "radius"), 0, width * 0.25 / coordinate_scale_x, seconds=seconds) * coordinate_scale_x
    radius_y = _parameter(mask, ("160", "radius"), 1, height * 0.25 / coordinate_scale_y, seconds=seconds) * coordinate_scale_y
    position_x = _parameter(mask, ("201", "position"), 0, 0.0, seconds=seconds) * coordinate_scale_x
    position_y = _parameter(mask, ("201", "position"), 1, 0.0, seconds=seconds) * coordinate_scale_y
    rotation = _parameter(mask, ("202", "rotation"), 0, 0.0, seconds=seconds)
    curvature = _parameter(mask, ("159", "curvature"), 0, 1.0, seconds=seconds)
    feather = _parameter(mask, ("102", "feather"), 0, 0.0, seconds=seconds) * min(coordinate_scale_x, coordinate_scale_y)
    opacity = _parameter(mask, ("103", "opacity"), 0, 1.0, seconds=seconds)
    falloff = _parameter(mask, ("104", "falloff"), 0, 1.0, seconds=seconds)

    # The CPU expression uses integer X/Y samples with the origin at the frame
    # center.  Keep that convention here for calibration parity.
    y, x = torch.meshgrid(
        torch.arange(height, device=canvas.device, dtype=canvas.dtype),
        torch.arange(width, device=canvas.device, dtype=canvas.dtype),
        indexing="ij",
    )
    dx = x - width / 2.0 - position_x
    dy = y - height / 2.0 + position_y
    angle = math.radians(rotation)
    rotated_x = dx * math.cos(angle) + dy * math.sin(angle)
    rotated_y = -dx * math.sin(angle) + dy * math.cos(angle)
    exponent = 2.0 + 6.0 * (1.0 - curvature)
    rx = max(1.0, radius_x)
    ry = max(1.0, radius_y)
    distance = (torch.abs(rotated_x / rx).pow(exponent) + torch.abs(rotated_y / ry).pow(exponent)).pow(1.0 / exponent)
    edge = (1.0 - distance) * min(rx, ry)
    if feather <= 0.000001:
        ramp = (distance <= 1.0).to(canvas.dtype)
    else:
        ramp = ((edge / feather) + 0.5).clamp(0.0, 1.0)
    return opacity * ramp.clamp_min(0.0).pow(falloff)


def _draw_matte(
    mask: ResolvedMask,
    height: int,
    width: int,
    canvas: torch.Tensor,
    *,
    seconds: float,
    coordinate_scale_x: float,
    coordinate_scale_y: float,
) -> torch.Tensor:
    raw = mask.data.get("points")
    if not raw:
        raise ValueError("resolved Draw Mask has no points")
    points = [
        (
            float(token.split(",")[0]) * coordinate_scale_x,
            float(token.split(",")[1]) * coordinate_scale_y,
        )
        for token in raw.split(";")
    ]
    y, x = torch.meshgrid(
        torch.arange(height, device=canvas.device, dtype=canvas.dtype),
        torch.arange(width, device=canvas.device, dtype=canvas.dtype),
        indexing="ij",
    )
    local_x = x - width / 2.0
    local_y = height / 2.0 - y
    area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, (*points[1:], points[0])))
    sign = 1.0 if area >= 0.0 else -1.0
    inside = torch.ones_like(local_x, dtype=torch.bool)
    for (x0, y0), (x1, y1) in zip(points, (*points[1:], points[0])):
        cross = (local_x - x0) * (y1 - y0) - (local_y - y0) * (x1 - x0)
        # ``cross`` is point x edge (the reverse of the usual edge x point),
        # so its sign is opposite the shoelace orientation.
        inside &= -sign * cross >= 0.0
    opacity = _parameter(mask, ("103", "opacity"), 0, 1.0, seconds=seconds)
    return opacity * inside.to(canvas.dtype)


def _color_matte(mask: ResolvedMask, canvas: torch.Tensor) -> torch.Tensor:
    color = _components(mask.data, "color", 3)
    tolerance = float(mask.data["tolerance"])
    softness = float(mask.data["softness"])
    opacity = float(mask.data["opacity"])
    code = premultiplied_to_code(canvas)[:3] / 255.0
    key = canvas.new_tensor(color).reshape(3, 1, 1)
    distance = ((code - key).square().sum(dim=0) / 3.0).sqrt()
    keyed = _soft_key(distance, tolerance, softness)
    return opacity * (1.0 - keyed)


def _range_matte(mask: ResolvedMask, canvas: torch.Tensor) -> torch.Tensor:
    low = float(mask.data["luma_min"])
    high = float(mask.data["luma_max"])
    softness = float(mask.data["softness"])
    opacity = float(mask.data["opacity"])
    code = premultiplied_to_code(canvas)[:3] / 255.0
    # ``format=gray`` in the CPU mask graph uses the default RGB-to-gray
    # coefficients, not the Rec.709 delivery coefficients used by the video
    # encoder exit.
    luma = 0.299 * code[0] + 0.587 * code[1] + 0.114 * code[2]
    if softness <= 0.0:
        return opacity * ((luma >= low) & (luma <= high)).to(canvas.dtype)
    return opacity * torch.minimum(
        ((luma - low) / softness).clamp(0.0, 1.0),
        ((high - luma) / softness).clamp(0.0, 1.0),
    )


def _soft_key(distance: torch.Tensor, tolerance: float, softness: float) -> torch.Tensor:
    """Match the CPU colorkey contract: zero at tolerance, one after the ramp."""

    if softness <= 0.0:
        return (distance <= tolerance).to(distance.dtype)
    return ((distance - tolerance) / softness).clamp(0.0, 1.0)


def _components(data: dict[str, str], key: str, count: int) -> tuple[float, ...]:
    raw = data.get(key)
    if raw is None:
        raise ValueError(f"resolved mask data is missing {key!r}")
    values = tuple(float(value) for value in raw.replace(",", " ").split())
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise ValueError(f"resolved mask data {key!r} has invalid components")
    return values


def _parameter(mask: ResolvedMask, keys: tuple[str, ...], component: int, default: float, *, seconds: float) -> float:
    wanted = {key.casefold() for key in keys}
    parameter = next(
        (item for item in mask.params if (item.key and item.key.casefold() in wanted) or (item.name and item.name.casefold() in wanted)),
        None,
    )
    if parameter is None:
        return default
    return _sample_parameter(parameter, component, seconds, default)


def _sample_parameter(parameter: Parameter, component: int, seconds: float, default: float) -> float:
    def value(raw: str) -> float:
        pieces = raw.replace(",", " ").split()
        if not pieces:
            raise ValueError(f"mask parameter {parameter.name or parameter.key!r} has no value")
        return float(pieces[component if len(pieces) > 1 else 0])

    base = value(parameter.value) if parameter.value is not None else default
    frames = parameter.keyframes
    if not frames:
        return base
    if seconds <= float(frames[0].time):
        return value(frames[0].value)
    for first, second in zip(frames, frames[1:]):
        t0, t1 = float(first.time), float(second.time)
        if seconds < t1:
            a, b = value(first.value), value(second.value)
            fraction = (seconds - t0) / (t1 - t0)
            return a + (b - a) * fraction
    return value(frames[-1].value)
