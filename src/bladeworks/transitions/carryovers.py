"""Calibrated parameter mappings for the four non-flat carryover transitions.

Architecture map
================

genuine Final Cut parameter key/value
    -> capability-registry type/range validation
    -> this module's renderer-owned semantic mapping
    -> radial-zoom strengths/centers or a fixed Black Hole expression

This module deliberately knows nothing about XML parsing or FFmpeg graph
assembly.  It converts already validated values into a small immutable plan.
``transitions.stock`` attaches that plan to the generic stock-transition plan,
and ``ffmpeg`` performs the final project-size center conversion.

Product invariants
------------------

* Final Cut text never becomes an FFmpeg expression fragment.
* Zoom and Cross Zoom preserve their previously calibrated defaults exactly.
* A Motion point is converted to a crop center only after project dimensions
  are known, and an off-canvas result is rejected rather than clamped.
* Black Hole's hidden structural fields are accepted only at their proved
  values; its pixel-inert visible nondefaults are rejected explicitly.

Why this exists
---------------

The carryovers do not share the flat-transition geometry and are not 360°
panorama operations.  Keeping their calibration here prevents ``stock.py``
from turning into another parameter-specific conditional ladder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ..core.errors import FCPXMLCompileError


# Genuine Final Cut 12.3 roundtrip keys.  Keeping these beside the mappings
# makes it obvious which serialized contract each formula consumes.
ZOOM_AMOUNT_KEY = "9999/988471386/3/999166426/1"
ZOOM_CENTER_KEY = "9999/988471386/3/999166426/2"

CROSS_ZOOM_AMOUNT_KEY = "3"
CROSS_ZOOM_START_POINT_KEY = "1"
CROSS_ZOOM_END_POINT_KEY = "2"

BLACK_HOLE_DURATION_KEY = "9999/10059/10065/3/989648928/1"
BLACK_HOLE_TRAIL_OPACITY_KEY = "9999/10059/10065/3/989648928/10001"
BLACK_HOLE_FLIP_KEY = "9999/10059/10065/3/989648928/10002"
BLACK_HOLE_INPUT_POINTS_KEY = "9999/10059/10065/3/989648928/10003"
BLACK_HOLE_ECHOES_KEY = "9999/10059/10065/3/989648928/2"
BLACK_HOLE_DECAY_KEY = "9999/10059/10065/3/989648928/3"
BLACK_HOLE_TRAIL_ON_KEY = "9999/10059/10065/3/989648928/4"


CenterSpace = Literal["normalized", "motion-project-height"]


@dataclass(frozen=True)
class RadialZoomCenter:
    """One bounded Final Cut center plus its explicit coordinate space."""

    value: tuple[float, float]
    space: CenterSpace


@dataclass(frozen=True)
class CarryoverRadialPlan:
    """The calibrated radial parameters consumed by the stock FFmpeg graph."""

    strength: float
    spread: float
    outgoing_center: RadialZoomCenter
    incoming_center: RadialZoomCenter


def build_radial_plan(
    implementation_id: str,
    values: Mapping[str, Any],
) -> CarryoverRadialPlan:
    """Map Zoom-family controls while preserving both old default renders.

    Main callers:
    - ``transitions.stock.build_stock_transition_plan`` for Zoom and Cross
      Zoom after the generic registry contract has validated every value.

    Final Cut Zoom publishes Amount with a 100 default.  Cross Zoom publishes
    a different FxPlug Amount whose default is 50, plus normalized start/end
    points.  The two amount normalizations below therefore produce the exact
    historical ``strength`` and ``spread`` values at their respective defaults.
    """

    if implementation_id == "zoom_blur_default":
        amount = _scalar(values, ZOOM_AMOUNT_KEY, default=100.0)
        center = RadialZoomCenter(
            _vector2(values, ZOOM_CENTER_KEY, default=(0.5, 0.5)),
            "motion-project-height",
        )
        factor = amount / 100.0
        return CarryoverRadialPlan(
            strength=1.70 * factor,
            spread=0.52 * factor,
            outgoing_center=center,
            incoming_center=center,
        )

    if implementation_id == "cross_zoom_default":
        amount = _scalar(values, CROSS_ZOOM_AMOUNT_KEY, default=50.0)
        factor = amount / 50.0
        return CarryoverRadialPlan(
            strength=2.00 * factor,
            spread=0.62 * factor,
            outgoing_center=RadialZoomCenter(
                _vector2(values, CROSS_ZOOM_START_POINT_KEY, default=(0.75, 0.5)),
                "normalized",
            ),
            incoming_center=RadialZoomCenter(
                _vector2(values, CROSS_ZOOM_END_POINT_KEY, default=(0.25, 0.5)),
                "normalized",
            ),
        )

    raise FCPXMLCompileError(
        f"carryover radial mapper does not own implementation {implementation_id!r}"
    )


def resolve_radial_center(
    center: RadialZoomCenter,
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Convert a proved Final Cut center to a normalized crop anchor.

    Main callers:
    - The CPU radial-zoom graph builder, after the output dimensions are known.

    Why this exists:
    Cross Zoom serializes ordinary normalized points.  Motion's Zoom template
    instead serializes offsets in a square coordinate system whose unit scale
    is the project height and whose neutral center is ``0.5 0.5``.  A genuine
    540x960 UI roundtrip (75 px, -120 px -> ``133.833 -119.5``) proves this
    conversion.  Off-canvas anchors need a different sampling primitive, so
    this stock crop implementation rejects them explicitly.
    """

    if width <= 0 or height <= 0:
        raise FCPXMLCompileError("radial Zoom needs positive project dimensions")
    x, y = center.value
    if center.space == "normalized":
        resolved = (x, y)
    elif center.space == "motion-project-height":
        resolved = (
            0.5 + (x - 0.5) / float(height),
            0.5 + (y - 0.5) / float(height),
        )
    else:  # pragma: no cover - the Literal plus frozen constructors own this.
        raise FCPXMLCompileError(f"unknown radial Zoom center space {center.space!r}")
    if any(not math.isfinite(component) for component in resolved):
        raise FCPXMLCompileError("radial Zoom center must be finite")
    if any(component < 0.0 or component > 1.0 for component in resolved):
        raise FCPXMLCompileError(
            "radial Zoom center resolves outside the project canvas; "
            "the stock CPU crop supports normalized centers in [0, 1]"
        )
    return resolved


def _plane_sample(source: str, x: str, y: str) -> str:
    """Return one color-preserving xfade sample from ``a`` or ``b``."""

    return (
        f"if(eq(PLANE,0),{source}0({x},{y}),"
        f"if(eq(PLANE,1),{source}1({x},{y}),"
        f"if(eq(PLANE,2),{source}2({x},{y}),"
        f"{source}3({x},{y}))))"
    )


# The default expression is byte-for-byte the previously reviewed Black Hole
# implementation.  ``build_black_hole_expression`` returns it unchanged while
# Trails is disabled, preserving existing default semantics and snapshots.
_ELAPSED = "(1-P)"
_REMAINING = f"pow(max(0,1-clip(({_ELAPSED}-0.18)/0.55,0,1)),0.6)"
_VERTICAL_SCALE = f"(1-clip(({_ELAPSED}-0.48)/0.26,0,1))"
_HALF_HEIGHT = f"(H/2*{_VERTICAL_SCALE})"
_Y = f"clip((Y-H/2)/{_VERTICAL_SCALE}+H/2,0,H-1)"
_VERTICAL_POSITION = f"abs((Y-H/2)/max(1,{_HALF_HEIGHT}))"
_PROFILE = f"(0.04+0.96*pow({_VERTICAL_POSITION},0.55))"
_PINCH_PROGRESS = "clip(((1-P)-0.05)/0.46,0,1)"
_ACTIVE_PROFILE = f"(1-{_PINCH_PROGRESS}*(1-{_PROFILE}))"
_SCALE = f"({_REMAINING}*{_ACTIVE_PROFILE})"
_HALF_WIDTH = f"(W/2*{_SCALE})"
_X = f"clip((X-W/2)/max(0.001,{_SCALE})+W/2,0,W-1)"
_SAMPLE = _plane_sample("a", _X, _Y)
_GHOST_SCALE = f"(min(1,{_SCALE}+0.16*{_REMAINING}))"
_GHOST_WIDTH = f"(W/2*{_GHOST_SCALE})"
_GHOST_VERTICAL_SCALE = f"min(1,1.08*{_VERTICAL_SCALE})"
_GHOST_HALF_HEIGHT = f"(H/2*{_GHOST_VERTICAL_SCALE})"
_GHOST_Y = f"clip((Y-H/2)/max(0.001,{_GHOST_VERTICAL_SCALE})+H/2,0,H-1)"
_GHOST_X = f"clip((X-W/2)/max(0.001,{_GHOST_SCALE})+W/2,0,W-1)"
_GHOST_SAMPLE = _plane_sample("a", _GHOST_X, _GHOST_Y)
_OPACITY_RAW = "clip((P-0.2)/0.4,0,1)"
_OPACITY = f"({_OPACITY_RAW}*{_OPACITY_RAW}*(3-2*{_OPACITY_RAW}))"
_ACTIVE_SAMPLE = f"({_OPACITY}*{_SAMPLE}+(1-{_OPACITY})*B)"
_GHOST_OPACITY = f"(0.16*{_OPACITY})"
_ACTIVE_GHOST = f"({_GHOST_OPACITY}*{_GHOST_SAMPLE}+(1-{_GHOST_OPACITY})*B)"
BLACK_HOLE_DEFAULT_EXPRESSION = (
    "if(gte(P,1),A,if(lte(P,0),B,"
    f"if(lte(abs(Y-H/2),{_HALF_HEIGHT})*"
    f"lte(abs(X-W/2),{_HALF_WIDTH}),{_ACTIVE_SAMPLE},"
    f"if(lte(abs(Y-H/2),{_GHOST_HALF_HEIGHT})*"
    f"lte(abs(X-W/2),{_GHOST_WIDTH}),{_ACTIVE_GHOST},B))))"
)


def build_black_hole_expression(values: Mapping[str, Any]) -> str:
    """Preserve Black Hole's default and reject its inert serialized controls.

    Main callers:
    - ``transitions.stock.build_stock_transition_plan``.

    Final Cut serializes the Trails object and preserves every authored value,
    but genuine renders for all fourteen nondefault settings are byte-identical
    to its default movie.  Rendering a portable-only response would invent a
    semantic contract.  The registry still types all seven published keys so
    malformed or unknown values fail normally; this boundary accepts only the
    exact visible defaults plus the three fixed structural values.
    """

    duration = _scalar(values, BLACK_HOLE_DURATION_KEY, default=0.0)
    _required_boolean(values, BLACK_HOLE_FLIP_KEY, expected=False)
    _required_boolean(values, BLACK_HOLE_INPUT_POINTS_KEY, expected=True)
    _required_scalar(values, BLACK_HOLE_DECAY_KEY, expected=1.0)
    echoes_raw = _scalar(values, BLACK_HOLE_ECHOES_KEY, default=4.0)
    if not echoes_raw.is_integer():
        raise FCPXMLCompileError("Black Hole Echoes must be an integer")
    echoes = int(echoes_raw)
    opacity = _scalar(values, BLACK_HOLE_TRAIL_OPACITY_KEY, default=1.0)
    trail_on_raw = _scalar(values, BLACK_HOLE_TRAIL_ON_KEY, default=0.0)
    if not trail_on_raw.is_integer() or int(trail_on_raw) not in {0, 1}:
        raise FCPXMLCompileError("Black Hole Trail On must be Light (0) or Dark (1)")
    trail_on = int(trail_on_raw)
    if (duration, opacity, echoes, trail_on) != (0.0, 1.0, 4, 0):
        raise FCPXMLCompileError(
            "Black Hole Trails controls preserve XML but produce no Final Cut "
            "pixel response; nondefault values are explicitly unsupported"
        )
    return BLACK_HOLE_DEFAULT_EXPRESSION


def _scalar(values: Mapping[str, Any], key: str, *, default: float) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FCPXMLCompileError(f"validated carryover parameter {key!r} must be scalar")
    number = float(value)
    if not math.isfinite(number):
        raise FCPXMLCompileError(f"validated carryover parameter {key!r} must be finite")
    return number


def _required_scalar(values: Mapping[str, Any], key: str, *, expected: float) -> None:
    value = _scalar(values, key, default=expected)
    if value != expected:
        raise FCPXMLCompileError(
            f"Black Hole structural parameter {key!r} must remain {expected:g}"
        )


def _required_boolean(values: Mapping[str, Any], key: str, *, expected: bool) -> None:
    """Validate a registry-typed hidden boolean at its one structural value."""

    value = values.get(key, expected)
    if not isinstance(value, bool) or value is not expected:
        raise FCPXMLCompileError(
            f"Black Hole structural parameter {key!r} must remain "
            f"{'true' if expected else 'false'}"
        )


def _vector2(
    values: Mapping[str, Any],
    key: str,
    *,
    default: tuple[float, float],
) -> tuple[float, float]:
    value = values.get(key, default)
    if not isinstance(value, tuple) or len(value) != 2:
        raise FCPXMLCompileError(f"validated carryover parameter {key!r} must be vec2")
    resolved = (float(value[0]), float(value[1]))
    if any(not math.isfinite(component) for component in resolved):
        raise FCPXMLCompileError(f"validated carryover parameter {key!r} must be finite")
    return resolved
