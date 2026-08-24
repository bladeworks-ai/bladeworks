"""Bounded SDR mappings for Final Cut's built-in color corrections.

Architecture map
================

``filter-video`` parameters -> registry-owned ranges/curves -> one ordered
sequence of stock FFmpeg filters.

The module only interprets scalar or two-component values that Final Cut writes
directly into FCPXML.  It deliberately does not decode the proprietary channel
blobs used by Color Wheels, Color Curves, and Hue/Saturation Curves.  Those
values remain source-preserved and are reported by the compiler as unsupported.

Important invariants:

* All mappings assume an SDR Rec. 709 working image.
* A value outside the registry contract is clamped, after the compiler has
  emitted a compatibility finding.
* Filter order is stable and follows the FCPXML filter order supplied by the
  graph planner.
"""

from __future__ import annotations

import colorsys
import math
from typing import Mapping, Optional

from .model import Parameter, ResolvedEffect


COLOR_HANDLERS = frozenset({"color_adjustments", "color_board", "color_wheels"})


def unsupported_color_reason(
    handler: str,
    params: tuple[Parameter, ...],
    *,
    sequence_color_space: Optional[str],
) -> Optional[str]:
    """Return why a color correction must not execute, if anything.

    Main callers:
    - ``compiler._resolve_filter_instance`` before creating renderer IR.

    Why this exists:
    The numeric mappings below are calibrated only for SDR Rec. 709. Applying
    them to an HDR control range would be a plausible-looking but materially
    wrong silent fallback.
    """

    if handler not in COLOR_HANDLERS:
        return None
    if sequence_color_space and "709" not in sequence_color_space:
        return f"portable {handler.replace('_', ' ')} is calibrated only for SDR Rec. 709"
    if handler == "color_adjustments":
        values = _parameter_values(params)
        control_range = values.get("19", values.get("Control Range"))
        if control_range is not None and control_range != "0 (SDR)":
            return f"Color Adjustments control range {control_range!r} is not the supported SDR range"
    return None


def color_effect_filters(effect: ResolvedEffect) -> list[str]:
    """Translate one registry-resolved color correction to stock filters.

    Main callers:
    - ``ffmpeg._effect_filters`` while preserving FCPXML effect order.
    """

    if effect.handler == "color_adjustments":
        return _color_adjustments_filters(effect)
    if effect.handler == "color_board":
        return _color_board_filters(effect)
    if effect.handler == "color_wheels":
        return _color_wheels_filters(effect)
    return []


def _color_adjustments_filters(effect: ResolvedEffect) -> list[str]:
    values = _parameter_values(effect.params)

    def adjusted(key: str, name: str) -> float:
        return _scalar(effect, values, key, name, 0.0) * _calibration(effect, key, "ffmpeg_scale", 0.0)

    brightness = adjusted("2", "Brightness")
    exposure = adjusted("3", "Exposure")
    contrast = _clamp(1.0 + adjusted("17", "Contrast"), 0.05, 3.0)
    saturation = _clamp(1.0 + adjusted("16", "Saturation"), 0.0, 3.0)
    shadows = adjusted("4", "Shadows")
    highlights = adjusted("7", "Highlights")
    black_point = _clamp(-adjusted("1", "Black Point"), 0.0, 0.2)
    warmth = (
        adjusted("14", "Shadows Warmth"),
        adjusted("12", "Midtones Warmth"),
        adjusted("10", "Highlights Warmth"),
    )
    tint = (
        adjusted("15", "Shadows Tint"),
        adjusted("13", "Midtones Tint"),
        adjusted("11", "Highlights Tint"),
    )
    output = [
        "eq="
        f"brightness={_number(brightness + exposure)}:"
        f"contrast={_number(contrast)}:"
        f"saturation={_number(saturation)}:"
        f"gamma={_number(max(0.2, 1.0 - shadows + highlights))}"
    ]
    if black_point > 0:
        output.append(
            f"colorlevels=rimin={_number(black_point)}:gimin={_number(black_point)}:bimin={_number(black_point)}"
        )
    if any(abs(value) > 1e-6 for value in (*warmth, *tint)):
        output.append(
            f"colorbalance=rs={_number(warmth[0])}:bs={_number(-warmth[0])}:"
            f"rm={_number(warmth[1])}:bm={_number(-warmth[1])}:"
            f"rh={_number(warmth[2])}:bh={_number(-warmth[2])}:"
            f"gs={_number(tint[0])}:gm={_number(tint[1])}:gh={_number(tint[2])}"
        )
    return output


def _color_board_filters(effect: ResolvedEffect) -> list[str]:
    """Approximate the legacy Color Board's explicit 2000-2011 controls.

    Color pucks are exported as ``hue amount`` pairs in normalized board
    coordinates; saturation and exposure pucks are scalar values with 0.5 as
    neutral.  FFmpeg has no direct four-zone saturation filter, so the bounded
    approximation preserves zone color/exposure and combines zone saturation
    into a conservative global saturation adjustment.
    """

    values = _parameter_values(effect.params)
    color_vectors = [
        _board_color_vector(effect, values, "2000", "Color Global"),
        _board_color_vector(effect, values, "2003", "Color Shadows"),
        _board_color_vector(effect, values, "2002", "Color Midtones"),
        _board_color_vector(effect, values, "2001", "Color Highlights"),
    ]
    global_rgb, shadow_rgb, midtone_rgb, highlight_rgb = color_vectors
    color_strength = _calibration(effect, "2000", "ffmpeg_balance_scale", 0.35)
    zones = []
    for global_component, shadow_component, midtone_component, highlight_component in zip(
        global_rgb,
        shadow_rgb,
        midtone_rgb,
        highlight_rgb,
    ):
        zones.append(
            (
                _clamp((global_component + shadow_component) * color_strength, -1.0, 1.0),
                _clamp((global_component + midtone_component) * color_strength, -1.0, 1.0),
                _clamp((global_component + highlight_component) * color_strength, -1.0, 1.0),
            )
        )

    saturation_global = _board_delta(effect, values, "2004", "Saturation Global")
    saturation_zones = (
        _board_delta(effect, values, "2007", "Saturation Shadows"),
        _board_delta(effect, values, "2006", "Saturation Midtones"),
        _board_delta(effect, values, "2005", "Saturation Highlights"),
    )
    zone_weight = _calibration(effect, "2004", "zone_weight", 0.25)
    saturation = _clamp(
        1.0 + saturation_global + zone_weight * sum(saturation_zones),
        0.0,
        3.0,
    )
    output: list[str] = []
    if abs(saturation - 1.0) > 1e-6:
        output.append(f"eq=saturation={_number(saturation)}")

    curve = _board_exposure_curve(effect, values)
    if curve is not None:
        output.append(f"curves=master='{curve}'")

    if any(abs(component) > 1e-6 for zone in zones for component in zone):
        red, green, blue = zones
        output.append(
            f"colorbalance=rs={_number(red[0])}:rm={_number(red[1])}:rh={_number(red[2])}:"
            f"gs={_number(green[0])}:gm={_number(green[1])}:gh={_number(green[2])}:"
            f"bs={_number(blue[0])}:bm={_number(blue[1])}:bh={_number(blue[2])}"
        )
    return output


def _color_wheels_filters(effect: ResolvedEffect) -> list[str]:
    """Apply only Color Wheels controls serialized as ordinary scalars.

    Wheel channels 1-4 remain proprietary base64 Motion parameter blobs. The
    compiler reports them individually and they never reach this mapping.
    """

    values = _parameter_values(effect.params)
    temperature = _scalar(effect, values, "8890", "Temperature", 5000.0)
    tint = _scalar(effect, values, "8891", "Tint", 0.0)
    hue = _scalar(effect, values, "8892", "Hue", 0.0)
    temperature_delta = (temperature - 5000.0) * _calibration(
        effect,
        "8890",
        "ffmpeg_scale",
        0.00006,
    )
    tint_delta = tint * _calibration(effect, "8891", "ffmpeg_scale", 0.006)
    output: list[str] = []
    if abs(hue) > 1e-6:
        output.append(f"hue=h={_number(hue)}")
    if abs(temperature_delta) > 1e-6 or abs(tint_delta) > 1e-6:
        red = _clamp(temperature_delta, -1.0, 1.0)
        green = _clamp(tint_delta, -1.0, 1.0)
        blue = _clamp(-temperature_delta, -1.0, 1.0)
        output.append(
            f"colorbalance=rs={_number(red)}:rm={_number(red)}:rh={_number(red)}:"
            f"gs={_number(green)}:gm={_number(green)}:gh={_number(green)}:"
            f"bs={_number(blue)}:bm={_number(blue)}:bh={_number(blue)}"
        )
    return output


def _board_color_vector(
    effect: ResolvedEffect,
    values: Mapping[str, str],
    key: str,
    name: str,
) -> tuple[float, float, float]:
    raw = values.get(key, values.get(name, "0.5 0.5"))
    pieces = raw.replace(",", " ").split()
    try:
        hue = _clamp(float(pieces[0]), 0.0, 1.0)
        amount = _clamp(float(pieces[1]), 0.0, 1.0) - 0.5
    except (ValueError, IndexError):
        return (0.0, 0.0, 0.0)
    red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    mean = (red + green + blue) / 3.0
    return (amount * (red - mean), amount * (green - mean), amount * (blue - mean))


def _board_delta(effect: ResolvedEffect, values: Mapping[str, str], key: str, name: str) -> float:
    neutral = _calibration(effect, key, "neutral", 0.5)
    scale = _calibration(effect, key, "ffmpeg_scale", 2.0)
    return (_scalar(effect, values, key, name, neutral) - neutral) * scale


def _board_exposure_curve(effect: ResolvedEffect, values: Mapping[str, str]) -> Optional[str]:
    global_delta = _board_delta(effect, values, "2008", "Exposure Global")
    highlight_delta = _board_delta(effect, values, "2009", "Exposure Highlights")
    midtone_delta = _board_delta(effect, values, "2010", "Exposure Midtones")
    shadow_delta = _board_delta(effect, values, "2011", "Exposure Shadows")
    if all(abs(value) <= 1e-6 for value in (global_delta, highlight_delta, midtone_delta, shadow_delta)):
        return None
    global_weight = _calibration(effect, "2008", "curve_weight", 0.30)
    zone_weight = _calibration(effect, "2009", "curve_weight", 0.22)
    points = [
        (0.0, 0.0),
        (0.2, 0.2 + global_delta * global_weight + shadow_delta * zone_weight),
        (0.5, 0.5 + global_delta * global_weight + midtone_delta * zone_weight),
        (0.8, 0.8 + global_delta * global_weight + highlight_delta * zone_weight),
        (1.0, 1.0),
    ]
    monotonic: list[tuple[float, float]] = []
    previous = 0.0
    for index, (x, y) in enumerate(points):
        if index == 0:
            bounded = 0.0
        elif index == len(points) - 1:
            bounded = 1.0
        else:
            bounded = _clamp(y, previous + 0.0001, 0.9999)
        monotonic.append((x, bounded))
        previous = bounded
    return " ".join(f"{_number(x)}/{_number(y)}" for x, y in monotonic)


def _parameter_values(params: tuple[Parameter, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for param in params:
        if param.value is None:
            continue
        if param.key:
            values[param.key] = param.value
        if param.name:
            values[param.name] = param.value
    return values


def _scalar(
    effect: ResolvedEffect,
    values: Mapping[str, str],
    key: str,
    name: str,
    default: float,
) -> float:
    raw = values.get(key, values.get(name))
    try:
        value = float(raw.split()[0]) if raw is not None else default
    except (ValueError, IndexError):
        value = default
    definition = effect.calibration.get(key, {})
    if not isinstance(definition, Mapping):
        return value
    minimum = float(definition.get("minimum", value))
    maximum = float(definition.get("maximum", value))
    return _clamp(value, minimum, maximum)


def _calibration(effect: ResolvedEffect, key: str, name: str, default: float) -> float:
    definition = effect.calibration.get(key, {})
    if not isinstance(definition, Mapping):
        return default
    try:
        return float(definition.get(name, default))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _number(value: float) -> str:
    if not math.isfinite(value):
        return "0"
    return format(value, ".12g")
