"""Bounded upstream-FFmpeg implementations for two-input transitions.

Architecture map
================

Capability-registry UID -> registry-owned implementation ID -> native ``xfade``
mode, fixed ``xfade=custom`` expression, or fixed side prefilter plus ``xfade``
-> recursively composed outgoing and incoming full-canvas streams.

FCPXML never supplies an FFmpeg mode, expression, path, or shader. It can only
select a capability UID and the bounded parameters declared for that UID. Most
initial approximations intentionally expose no source parameters because the
available genuine Final Cut exports prove only their UIDs and default modes.

Why this exists
---------------
The custom Vulkan prototype remains a useful experimental escape hatch, but a
portable MVP should run on an unmodified upstream FFmpeg. Keeping every stock
implementation in this small allow-list makes that portability boundary easy
to audit and prevents XML text from becoming executable filter syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..core.errors import FCPXMLCompileError
from .cohort import (
    IMPLEMENTATION_IDS as COHORT_IMPLEMENTATION_IDS,
    build_cohort_transition_plan,
)
from .carryovers import (
    RadialZoomCenter,
    build_black_hole_expression,
    build_radial_plan,
)


@dataclass(frozen=True)
class StockTransitionPlan:
    """One fully registry-owned ``xfade`` configuration."""

    mode: str
    expression: str | None = None
    prefilter: str | None = None
    strength: float = 0.0
    spread: float = 0.0
    outgoing_center: RadialZoomCenter | None = None
    incoming_center: RadialZoomCenter | None = None


NATIVE_IMPLEMENTATIONS: Mapping[str, str] = {
    # Final Cut's default Directional angle is horizontal in the inspected
    # Motion template. FFmpeg's native horizontal blur is a bounded, fast
    # approximation while the FCPXML key for Angle remains unvalidated.
    "directional_blur_default": "hblur",
    "cross_blur_default": "hblur",
}


# ``P`` is FFmpeg xfade's remaining-outgoing progress: 1 at the transition
# start and 0 at its end. ``a0``/``a1``/etc sample one specific color plane;
# selecting by ``PLANE`` is required to preserve color. Explicit endpoint
# guards keep exact source pixels at both boundaries.


def _plane_sample(source: str, x: str, y: str) -> str:
    """Return one color-preserving xfade sample from source ``a`` or ``b``."""

    return (
        f"if(eq(PLANE,0),{source}0({x},{y}),"
        f"if(eq(PLANE,1),{source}1({x},{y}),"
        f"if(eq(PLANE,2),{source}2({x},{y}),"
        f"{source}3({x},{y}))))"
    )


# Final Cut holds the outgoing panel nearly still through the first third,
# introduces a small hinge rotation around the midpoint, and does most of the
# downward travel in the final quarter.  A linear drop exposed the incoming
# canvas much too early.  The cubic travel curve keeps the card readable at the
# reviewed 73%-progress frame while still moving it completely below the canvas
# before the exact incoming endpoint.
_FALL_RAW_PROGRESS = "(1-P)"
_FALL_PROGRESS = f"(clip(({_FALL_RAW_PROGRESS}-0.42)/0.57,0,1))"
_FALL_ANGLE = f"(-0.09*sin(PI*{_FALL_PROGRESS}))"
_FALL_SCALE = f"(1-0.06*{_FALL_PROGRESS})"
_FALL_CENTER_X = f"(W/2+0.03*W*{_FALL_PROGRESS})"
_FALL_CENTER_Y = (
    f"(H/2+1.02*H*pow({_FALL_PROGRESS},3.4))"
)
_FALL_DX = f"(X-{_FALL_CENTER_X})"
_FALL_DY = f"(Y-{_FALL_CENTER_Y})"
_FALL_X = (
    f"(cos({_FALL_ANGLE})*{_FALL_DX}+sin({_FALL_ANGLE})*{_FALL_DY})/"
    f"{_FALL_SCALE}+W/2"
)
_FALL_Y = (
    f"(-sin({_FALL_ANGLE})*{_FALL_DX}+cos({_FALL_ANGLE})*{_FALL_DY})/"
    f"{_FALL_SCALE}+H/2"
)
_FALL_SAMPLE = _plane_sample("a", _FALL_X, _FALL_Y)
FALL_EXPRESSION = (
    "if(gte(P,1),A,if(lte(P,0),B,"
    f"if(between({_FALL_X},0,W-1)*between({_FALL_Y},0,H-1),"
    f"{_FALL_SAMPLE},B)))"
)


# Final Cut's Squares transition reveals whole square tiles at different
# moments. It does not blur or enlarge the source pixels. The tile side is one
# third of the shorter canvas edge, which preserves square geometry in both
# landscape and portrait projects. Each tile receives a deterministic reveal
# threshold; endpoint guards still return the exact source frames.
_SQUARE_TILE_SIDE = "(min(W,H)/3)"
_SQUARE_TILE_X = f"floor(X/{_SQUARE_TILE_SIDE})"
_SQUARE_TILE_Y = f"floor(Y/{_SQUARE_TILE_SIDE})"
_SQUARE_TILE_THRESHOLD = (
    f"((mod({_SQUARE_TILE_X}*37+{_SQUARE_TILE_Y}*17,13)+1)/14)"
)
SQUARES_TILE_REVEAL_EXPRESSION = (
    "if(gte(P,1),A,if(lte(P,0),B,"
    f"if(gte(1-P,{_SQUARE_TILE_THRESHOLD}),B,A)))"
)


CUSTOM_IMPLEMENTATIONS: Mapping[str, str] = {
    "fall_default": FALL_EXPRESSION,
    "squares_tile_reveal_default": SQUARES_TILE_REVEAL_EXPRESSION,
}


IMPLEMENTATION_IDS = frozenset(
    {
        *NATIVE_IMPLEMENTATIONS,
        *CUSTOM_IMPLEMENTATIONS,
        *COHORT_IMPLEMENTATION_IDS,
        "black_hole_default",
        "circle_default",
        "cross_dissolve_default",
        "zoom_blur_default",
        "cross_zoom_default",
    }
)


_CIRCLE_RADIUS = "hypot(W,H)/2"
_CIRCLE_OPEN_EXPRESSION = (
    "if(gte(P,1),A,if(lte(P,0),B,"
    f"if(lte(hypot(X-W/2,Y-H/2),(1-P)*{_CIRCLE_RADIUS}),B,A)))"
)
_CIRCLE_CLOSE_EXPRESSION = (
    "if(gte(P,1),A,if(lte(P,0),B,"
    f"if(lte(hypot(X-W/2,Y-H/2),P*{_CIRCLE_RADIUS}),A,B)))"
)


def build_stock_transition_plan(
    implementation_id: str,
    parameter_values: Mapping[str, Any],
) -> StockTransitionPlan:
    """Resolve one validated implementation ID to fixed FFmpeg syntax.

    Main callers:
    - ``ffmpeg._build_stock_transition_groups`` after the compiler has already
      validated the selected capability and its bounded parameters.
    """

    if implementation_id == "cross_dissolve_default":
        # Both sides have already been fully composed on project-sized
        # transparent canvases.  The stock fade is therefore the exact
        # two-image ownership operation; per-clip opacity fades would apply
        # alpha twice and reveal the lower timeline during the transition.
        return StockTransitionPlan(mode="fade")
    native = NATIVE_IMPLEMENTATIONS.get(implementation_id)
    if native is not None:
        return StockTransitionPlan(mode=native)
    if implementation_id == "zoom_blur_default":
        radial = build_radial_plan(implementation_id, parameter_values)
        return StockTransitionPlan(
            mode="fade",
            prefilter="radial_zoom",
            strength=radial.strength,
            spread=radial.spread,
            outgoing_center=radial.outgoing_center,
            incoming_center=radial.incoming_center,
        )
    if implementation_id == "cross_zoom_default":
        radial = build_radial_plan(implementation_id, parameter_values)
        return StockTransitionPlan(
            mode="fade",
            prefilter="radial_zoom",
            strength=radial.strength,
            spread=radial.spread,
            outgoing_center=radial.outgoing_center,
            incoming_center=radial.incoming_center,
        )
    if implementation_id == "black_hole_default":
        return StockTransitionPlan(
            mode="custom",
            expression=build_black_hole_expression(parameter_values),
        )
    expression = CUSTOM_IMPLEMENTATIONS.get(implementation_id)
    if expression is not None:
        return StockTransitionPlan(mode="custom", expression=expression)
    if implementation_id == "circle_default":
        mode = parameter_values.get("7", False)
        if not isinstance(mode, bool):
            raise FCPXMLCompileError("validated Circle mode must be a boolean")
        # Native xfade circleopen/circleclose blend the two sources through a
        # soft disc. Final Cut uses a hard circular matte, so keep the stock
        # FFmpeg runtime but provide the exact bounded geometry ourselves.
        expression = _CIRCLE_CLOSE_EXPRESSION if mode else _CIRCLE_OPEN_EXPRESSION
        return StockTransitionPlan(mode="custom", expression=expression)
    cohort = build_cohort_transition_plan(implementation_id, parameter_values)
    if cohort is not None:
        return StockTransitionPlan(
            mode=cohort.mode,
            expression=cohort.expression,
            prefilter=cohort.prefilter,
            strength=cohort.strength,
            spread=cohort.spread,
        )
    raise FCPXMLCompileError(
        f"registry selected unknown stock transition implementation {implementation_id!r}"
    )


def build_fade_color_plan(color: tuple[int, int, int]) -> StockTransitionPlan:
    """Build Final Cut's source-to-color-to-destination fade profile.

    Main callers:
    - ``ffmpeg._build_stock_transition_groups`` for the bounded Fade to Color
      handler.

    Why this exists:
    FFmpeg's native ``fadeblack``/``fadewhite`` modes spend too little time at
    the solid midpoint. A genuine Final Cut 12.3 render reaches the color at
    11/30 progress, holds it through 19/30, then reveals the incoming image.
    The expression keeps that timing while accepting only the registry's
    bounded RGB triplet.
    """

    red, green, blue = color
    if any(component < 0 or component > 255 for component in color):
        raise FCPXMLCompileError("validated Fade to Color components must be in [0, 255]")
    # The custom-xfade graph runs in ``gbrap``, so ``PLANE`` is *not* an RGB index:
    # plane 0 is G, plane 1 is B, plane 2 is R, plane 3 is A. The registry hands us a
    # plain RGB triplet, so it has to be permuted into plane order here. Getting this
    # wrong is silent and total -- the literal still renders, just in the wrong hue
    # (an authored red-orange came out green until this was fixed). The hand-authored
    # colour literals elsewhere in this package (``_LEAF_GREEN`` in organic_light,
    # ``_DECO_GOLD`` in light_deco) are already written directly in plane order, which
    # is why only this one, fed from outside, needed the permutation.
    solid = (
        f"if(eq(PLANE,0),{green},"
        f"if(eq(PLANE,1),{blue},"
        f"if(eq(PLANE,2),{red},255)))"
    )
    first_phase = "0.3666666667"
    second_phase = "0.6333333333"
    expression = (
        "if(gte(P,1),A,if(lte(P,0),B,"
        f"if(gt(P,{second_phase}),"
        f"{solid}+(A-{solid})*(P-{second_phase})/{first_phase},"
        f"if(gte(P,{first_phase}),{solid},"
        f"{solid}+(B-{solid})*({first_phase}-P)/{first_phase}))))"
    )
    return StockTransitionPlan(mode="custom", expression=expression)
