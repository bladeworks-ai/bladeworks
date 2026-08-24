"""Portable primitives for the flat and object transition cohort.

Architecture map
================

Final Cut transition UID
    -> capability-registry implementation ID
    -> one fixed native ``xfade`` mode, bounded custom expression, or existing
       radial sampling prefilter
    -> exact outgoing/incoming endpoint guards.

The expressions in this module are renderer-owned constants.  FCPXML can only
select a registered UID; it cannot provide an FFmpeg expression, filter name,
path, or shader body.  Parameterless defaults are deliberate where the real
exports prove a UID but do not expose a trustworthy numeric parameter range.

Why this exists
---------------
The cohort contains many names but only a handful of useful mechanisms:
geometric mattes, inverse affine sampling, deterministic procedural mattes,
light/noise pulses, and stock blur/wipe modes.  Keeping those mechanisms here
avoids a large dispatch ladder in ``ffmpeg.py`` and makes every expression
independently auditable and render-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .light_deco import CUSTOM_IMPLEMENTATIONS as LIGHT_DECO_IMPLEMENTATIONS
from .panel_motion import CUSTOM_IMPLEMENTATIONS as PANEL_MOTION_IMPLEMENTATIONS
from .panel_motion import build_panel_motion_expression


@dataclass(frozen=True)
class CohortTransitionPlan:
    """One trusted stock-FFmpeg plan returned to ``transitions.stock``."""

    mode: str
    expression: str | None = None
    prefilter: str | None = None
    strength: float = 0.0
    spread: float = 0.0


def _sample(source: str, x: str = "X", y: str = "Y") -> str:
    """Sample one source without allowing a color plane to leak into another."""

    safe_x = f"clip({x},0,W-1)"
    safe_y = f"clip({y},0,H-1)"
    return (
        f"if(eq(PLANE,0),{source}0({safe_x},{safe_y}),"
        f"if(eq(PLANE,1),{source}1({safe_x},{safe_y}),"
        f"if(eq(PLANE,2),{source}2({safe_x},{safe_y}),"
        f"{source}3({safe_x},{safe_y}))))"
    )


def _endpoint(body: str) -> str:
    """Keep both boundary frames equal to their recursively composed sources."""

    return f"if(gte(P,1),A,if(lte(P,0),B,{body}))"


def _mix(first: str, second: str, first_weight: str = "P") -> str:
    return f"(({first_weight})*({first})+(1-({first_weight}))*({second}))"


def _smoothstep(value: str) -> str:
    """Return a bounded cubic ease without relying on host-specific helpers."""

    bounded = f"clip({value},0,1)"
    return f"(({bounded})*({bounded})*(3-2*({bounded})))"


def _number(value: float) -> str:
    """Serialize one already-bounded registry number for FFmpeg syntax."""

    return f"{value:.9f}".rstrip("0").rstrip(".") or "0"


def _affine_coordinates(
    *,
    angle: str,
    scale: str,
    center_x: str = "W/2",
    center_y: str = "H/2",
) -> tuple[str, str]:
    """Return inverse coordinates for a panel rotated/scaled around a pivot."""

    dx = f"(X-{center_x})"
    dy = f"(Y-{center_y})"
    x = f"((cos({angle})*{dx}+sin({angle})*{dy})/max(0.01,{scale})+W/2)"
    y = f"((-sin({angle})*{dx}+cos({angle})*{dy})/max(0.01,{scale})+H/2)"
    return x, y


_PROGRESS = "(1-P)"
_MID_ACTIVITY = f"(4*{_PROGRESS}*(1-{_PROGRESS}))"
_PLAIN_MIX = _endpoint(_mix("A", "B"))

_CENTER_PROGRESS = _smoothstep(_PROGRESS)
_CENTER_DISTANCE = f"(W/2*{_CENTER_PROGRESS}-abs(X-W/2))"
_CENTER_WEIGHT = f"clip(0.5+{_CENTER_DISTANCE}/(0.035*W),0,1)"
CENTER_EXPRESSION = _endpoint(
    f"({_CENTER_WEIGHT}*B+(1-{_CENTER_WEIGHT})*A)"
)


def _center_expression(
    *,
    angle_degrees: float,
    direction: int,
    border: float,
    feather: bool,
) -> str:
    """Build bounded Center geometry from the native FxPlug controls."""

    if (angle_degrees, direction, border, feather) == (0.0, 2, 15.0, True):
        return CENTER_EXPRESSION
    angle = f"({_number(angle_degrees)}*PI/180)"
    coordinate = f"(cos({angle})*(X-W/2)+sin({angle})*(Y-H/2))"
    extent = f"(abs(cos({angle}))*W/2+abs(sin({angle}))*H/2)"
    progress = _CENTER_PROGRESS
    if direction == 1:
        progress = f"(1-{progress})"
    distance = f"({extent}*{progress}-abs({coordinate}))"
    softness = f"max(1,{_number(border)}*min(W,H)/100)" if feather else "1"
    weight = f"clip(0.5+{distance}/{softness},0,1)"
    if direction == 1:
        return _endpoint(f"({weight}*A+(1-{weight})*B)")
    return _endpoint(f"({weight}*B+(1-{weight})*A)")

def _rotating_panel_expression(*, turns: float, shrink: float) -> str:
    angle = f"({turns:.6f}*PI*{_PROGRESS})"
    scale = f"(1-{shrink:.6f}*{_PROGRESS})"
    x, y = _affine_coordinates(angle=angle, scale=scale)
    inside = f"between({x},0,W-1)*between({y},0,H-1)"
    panel = _sample("a", x, y)
    return _endpoint(f"if({inside},{panel},B)")


def _dual_rotating_panel_expression() -> str:
    """Keep two large opposing cards over black through the rotate midpoint."""

    a_x, a_y = _affine_coordinates(
        angle=f"(0.50*PI*{_PROGRESS})",
        scale=f"(1-0.45*{_PROGRESS})",
    )
    b_x, b_y = _affine_coordinates(
        angle=f"(-0.50*PI*(1-{_PROGRESS}))",
        scale=f"(0.55+0.45*{_PROGRESS})",
    )
    a_inside = (
        f"lt({_PROGRESS},0.80)*between({a_x},0,W-1)*between({a_y},0,H-1)"
    )
    b_inside = (
        f"gt({_PROGRESS},0.20)*between({b_x},0,W-1)*between({b_y},0,H-1)"
    )
    a_panel = _sample("a", a_x, a_y)
    b_panel = _sample("b", b_x, b_y)
    return _endpoint(
        f"if(lt({_PROGRESS},0.5),"
        f"if({a_inside},{a_panel},if({b_inside},{b_panel},0)),"
        f"if({b_inside},{b_panel},if({a_inside},{a_panel},0)))"
    )


def _parameterized_rotate_expression(
    *,
    counterclockwise: bool,
    black_background: bool,
) -> str:
    """Return the non-default rotation/background combinations."""

    if not counterclockwise and not black_background:
        return ROTATE_EXPRESSION
    sign = -1 if counterclockwise else 1
    background_scale = 0.96 if black_background else 1.0
    a_x, a_y = _affine_coordinates(
        angle=f"({sign}*0.50*PI*{_PROGRESS})",
        scale=f"({background_scale:.2f}*(1-0.45*{_PROGRESS}))",
    )
    b_x, b_y = _affine_coordinates(
        angle=f"({-sign}*0.50*PI*(1-{_PROGRESS}))",
        scale=f"({background_scale:.2f}*(0.55+0.45*{_PROGRESS}))",
    )
    a_inside = f"lt({_PROGRESS},0.80)*between({a_x},0,W-1)*between({a_y},0,H-1)"
    b_inside = f"gt({_PROGRESS},0.20)*between({b_x},0,W-1)*between({b_y},0,H-1)"
    a_panel = _sample("a", a_x, a_y)
    b_panel = _sample("b", b_x, b_y)
    background = "0" if black_background else f"if(lt({_PROGRESS},0.5),A,B)"
    return _endpoint(
        f"if(lt({_PROGRESS},0.5),if({a_inside},{a_panel},"
        f"if({b_inside},{b_panel},{background})),"
        f"if({b_inside},{b_panel},if({a_inside},{a_panel},{background})))"
    )


def _swing_expression(*, pivot_x: str, sign: int) -> str:
    angle = f"({sign}*0.72*sin(PI*{_PROGRESS}))"
    x, y = _affine_coordinates(
        angle=angle,
        scale="1",
        center_x=pivot_x,
        center_y="0",
    )
    inside = f"between({x},0,W-1)*between({y},0,H-1)"
    return _endpoint(f"if({inside},{_sample('a', x, y)},B)")


def _top_hinged_swing_expression() -> str:
    """Rotate one source-switched card around its top edge over black.

    Final Cut's default swings the outgoing card toward screen-left before the
    incoming face settles from the same side.  The first implementation had
    the right motion envelope but negated both angles, so the whole card leaned
    toward screen-right.  Keep the reviewed timing and scale exactly as they
    were while correcting only that rejected direction.
    """

    incoming_ease = _smoothstep(f"(({_PROGRESS}-0.38)/0.38)")
    outgoing_ease = _smoothstep(f"({_PROGRESS}/0.38)")
    # The incoming side passes very close to Final Cut's virtual camera at the
    # midpoint.  A milder rotation plus a larger scale keeps the card covering
    # the canvas instead of exposing a diagonal black wedge.
    angle = (
        f"if(lt({_PROGRESS},0.38),0.18*{outgoing_ease},"
        f"0.28*(1-{incoming_ease}))"
    )
    scale = f"if(lt({_PROGRESS},0.38),1,1+3.20*(1-{incoming_ease}))"
    dx = "(X-W/2)"
    y0 = "(Y-H/2)"
    x = f"((cos({angle})*{dx}+sin({angle})*{y0})/{scale}+W/2)"
    y = f"((-sin({angle})*{dx}+cos({angle})*{y0})/{scale}+H/2)"
    inside = f"between({x},0,W-1)*between({y},0,H-1)"
    panel = f"if(lt({_PROGRESS},0.38),{_sample('a', x, y)},{_sample('b', x, y)})"
    return _endpoint(f"if({inside},{panel},0)")


def _parameterized_swing_expression(
    *,
    anchor: int,
    towards: bool,
    black_background: bool,
) -> str:
    """Build Right/Left/Top/Bottom and Towards/Away Swing variants."""

    if anchor == 2 and not towards and not black_background:
        return SWING_EXPRESSION
    activity = f"sin(PI*{_PROGRESS})"
    signed_angle = f"({1 if towards else -1}*0.72*{activity})"
    if anchor in {0, 1}:
        pivot_x = "W" if anchor == 0 else "0"
        angle = f"(-1*{signed_angle})" if anchor == 0 else signed_angle
        x, y = _affine_coordinates(
            angle=angle,
            scale="1",
            center_x=pivot_x,
            center_y="H/2",
        )
    else:
        pivot_y = "0" if anchor == 2 else "H"
        angle = f"(-1*{signed_angle})" if anchor == 3 else signed_angle
        x, y = _affine_coordinates(
            angle=angle,
            scale="1",
            center_x="W/2",
            center_y=pivot_y,
        )
    inside = f"between({x},0,W-1)*between({y},0,H-1)"
    panel = f"if(lt({_PROGRESS},0.5),{_sample('a', x, y)},{_sample('b', x, y)})"
    background = "0" if black_background else f"if(lt({_PROGRESS},0.5),B,A)"
    return _endpoint(f"if({inside},{panel},{background})")


# Geometric mattes ---------------------------------------------------------

_CLOCK_PHASE = "mod(PI/2+atan2(Y-H/2,X-W/2)+2*PI,2*PI)"
_CLOCK_DISTANCE = f"(2*PI*{_PROGRESS}-{_CLOCK_PHASE})"
_CLOCK_WEIGHT = f"clip(0.5+{_CLOCK_DISTANCE}/(0.30*PI),0,1)"
CLOCK_EXPRESSION = _endpoint(
    f"({_CLOCK_WEIGHT}*B+(1-{_CLOCK_WEIGHT})*A)"
)


def _clock_expression(
    *,
    angle_degrees: float,
    counterclockwise: bool,
    border: float,
    feather: bool,
) -> str:
    """Build native Clock angle, direction, border, and edge choices."""

    if (angle_degrees, counterclockwise, border, feather) == (0.0, False, 15.0, True):
        return CLOCK_EXPRESSION
    phase = (
        f"mod(PI/2+({_number(angle_degrees)}*PI/180)+"
        f"atan2(Y-H/2,X-W/2)+2*PI,2*PI)"
    )
    if counterclockwise:
        phase = f"mod(2*PI-({phase}),2*PI)"
    distance = f"(2*PI*{_PROGRESS}-{phase})"
    softness = f"max(0.001,{_number(border)}*PI/100)" if feather else "0.001"
    weight = f"clip(0.5+{distance}/{softness},0,1)"
    return _endpoint(f"({weight}*B+(1-{weight})*A)")

_PAGE_EDGE = f"({_PROGRESS}*(W+H+0.24*W)-0.24*W)"
_PAGE_CURVE = "(0.30*pow(Y-H/2,2)/H)"
_PAGE_DISTANCE = f"(X+Y+{_PAGE_CURVE}-{_PAGE_EDGE})"
_PAGE_WIDTH = "(0.075*(W+H))"
_PAGE_SHADOW = "(0.035*(W+H))"
_PAGE_REFLECT_X = f"(X-0.72*clip({_PAGE_DISTANCE},0,{_PAGE_WIDTH}))"
_PAGE_REFLECT_Y = f"(Y-0.72*clip({_PAGE_DISTANCE},0,{_PAGE_WIDTH}))"
_PAGE_FRONT = _sample("a", _PAGE_REFLECT_X, _PAGE_REFLECT_Y)
_PAGE_BACK = (
    f"min(255,0.26*{_PAGE_FRONT}+154+"
    f"58*sin(PI*clip({_PAGE_DISTANCE}/{_PAGE_WIDTH},0,1)))"
)
_PAGE_SHADOW_WEIGHT = (
    f"clip(({_PAGE_DISTANCE}-{_PAGE_WIDTH})/{_PAGE_SHADOW},0,1)"
)
PAGE_CURL_EXPRESSION = _endpoint(
    f"if(lte({_PAGE_DISTANCE},0),B,"
    f"if(lt({_PAGE_DISTANCE},{_PAGE_WIDTH}),{_PAGE_BACK},"
    f"if(lt({_PAGE_DISTANCE},{_PAGE_WIDTH}+{_PAGE_SHADOW}),"
    f"A*(0.52+0.48*{_PAGE_SHADOW_WEIGHT}),A)))"
)


def _page_curl_expression(preset: int, direction: int) -> str:
    """Build the two presets and Open/Close/Automatic ownership choices."""

    if preset == 0 and direction == 2:
        return PAGE_CURL_EXPRESSION
    # Explicit Open uses Final Cut's eased inspector path. Automatic preserves
    # the already reviewed linear default, so the two published menu choices
    # remain measurably distinct without changing default output.
    travel_progress = _smoothstep(_PROGRESS) if direction == 0 else _PROGRESS
    edge = f"({travel_progress}*(W+H+0.24*W)-0.24*W)"
    curve = "(0.30*pow(Y-H/2,2)/H)"
    distance = (
        f"((W-X)+Y+{curve}-{edge})"
        if preset == 1
        else f"(X+Y+{curve}-{edge})"
    )
    width = "(0.075*(W+H))"
    shadow = "(0.035*(W+H))"
    reflected_x = (
        f"(X+0.72*clip({distance},0,{width}))"
        if preset == 1
        else f"(X-0.72*clip({distance},0,{width}))"
    )
    reflected_y = f"(Y-0.72*clip({distance},0,{width}))"
    front = _sample("a", reflected_x, reflected_y)
    back = f"min(255,0.26*{front}+154+58*sin(PI*clip({distance}/{width},0,1)))"
    shadow_weight = f"clip(({distance}-{width})/{shadow},0,1)"
    opened = _endpoint(
        f"if(lte({distance},0),B,if(lt({distance},{width}),{back},"
        f"if(lt({distance},{width}+{shadow}),A*(0.52+0.48*{shadow_weight}),A)))"
    )
    if direction in {0, 2}:
        return opened
    # Close is the source-opposed fold.  Keep the same reviewed geometry while
    # reversing the interior source ownership and fold travel; endpoint guards
    # still return the exact outgoing/incoming frames.
    close_progress = f"(1-{_PROGRESS})"
    close_edge = f"({close_progress}*(W+H+0.24*W)-0.24*W)"
    close_distance = (
        f"((W-X)+Y+{curve}-{close_edge})"
        if preset == 1
        else f"(X+Y+{curve}-{close_edge})"
    )
    close_reflected_x = (
        f"(X+0.72*clip({close_distance},0,{width}))"
        if preset == 1
        else f"(X-0.72*clip({close_distance},0,{width}))"
    )
    close_reflected_y = f"(Y-0.72*clip({close_distance},0,{width}))"
    close_front = _sample("b", close_reflected_x, close_reflected_y)
    close_back = (
        f"min(255,0.26*{close_front}+154+"
        f"58*sin(PI*clip({close_distance}/{width},0,1)))"
    )
    close_shadow_weight = f"clip(({close_distance}-{width})/{shadow},0,1)"
    return _endpoint(
        f"if(lte({close_distance},0),A,if(lt({close_distance},{width}),{close_back},"
        f"if(lt({close_distance},{width}+{shadow}),"
        f"B*(0.52+0.48*{close_shadow_weight}),B)))"
    )

ROTATE_EXPRESSION = _dual_rotating_panel_expression()
SWING_EXPRESSION = _top_hinged_swing_expression()

_DROP_PROGRESS = _smoothstep(f"({_PROGRESS}/0.55)")
_DROP_OFFSET_Y = f"(H*(1-{_DROP_PROGRESS}))"
_DROP_PANEL_Y = f"(Y-{_DROP_OFFSET_Y})"
_DROP_BLUR = f"(0.07*H*(1-{_DROP_PROGRESS}))"
_DROP_PANEL = (
    f"(0.52*{_sample('b', 'X', _DROP_PANEL_Y)}+"
    f"0.22*{_sample('b', 'X', f'({_DROP_PANEL_Y}-0.30*{_DROP_BLUR})')}+"
    f"0.16*{_sample('b', 'X', f'({_DROP_PANEL_Y}-0.65*{_DROP_BLUR})')}+"
    f"0.10*{_sample('b', 'X', f'({_DROP_PANEL_Y}-{_DROP_BLUR})')})"
)
_DROP_SHADOW = (
    f"0.32*A*clip(1+{_DROP_PANEL_Y}/(0.07*H),0,1)"
)
DROP_IN_EXPRESSION = _endpoint(
    f"if(between({_DROP_PANEL_Y},0,H-1),{_DROP_PANEL},"
    f"if(between({_DROP_PANEL_Y},-0.07*H,0),{_DROP_SHADOW},A))"
)

_SWITCH_SEPARATION = f"sin(PI*{_PROGRESS})"
_SWITCH_SCALE = f"(0.84+0.16*abs(2*{_PROGRESS}-1))"
_SWITCH_A_CENTER_X = f"(W/2-0.45*W*{_SWITCH_SEPARATION})"
_SWITCH_A_CENTER_Y = f"(H/2-0.55*H*{_SWITCH_SEPARATION})"
_SWITCH_B_CENTER_X = f"(W/2+0.45*W*{_SWITCH_SEPARATION})"
_SWITCH_B_CENTER_Y = f"(H/2+0.55*H*{_SWITCH_SEPARATION})"
_SWITCH_A_ANGLE = f"(-0.45*{_SWITCH_SEPARATION})"
_SWITCH_B_ANGLE = f"(0.45*{_SWITCH_SEPARATION})"
_SWITCH_A_DX = f"(X-{_SWITCH_A_CENTER_X})"
_SWITCH_A_DY = f"(Y-{_SWITCH_A_CENTER_Y})"
_SWITCH_B_DX = f"(X-{_SWITCH_B_CENTER_X})"
_SWITCH_B_DY = f"(Y-{_SWITCH_B_CENTER_Y})"
_SWITCH_A_X = f"((cos({_SWITCH_A_ANGLE})*{_SWITCH_A_DX}+sin({_SWITCH_A_ANGLE})*{_SWITCH_A_DY})/{_SWITCH_SCALE}+W/2)"
_SWITCH_A_Y = f"((-sin({_SWITCH_A_ANGLE})*{_SWITCH_A_DX}+cos({_SWITCH_A_ANGLE})*{_SWITCH_A_DY})/{_SWITCH_SCALE}+H/2)"
_SWITCH_B_X = f"((cos({_SWITCH_B_ANGLE})*{_SWITCH_B_DX}+sin({_SWITCH_B_ANGLE})*{_SWITCH_B_DY})/{_SWITCH_SCALE}+W/2)"
_SWITCH_B_Y = f"((-sin({_SWITCH_B_ANGLE})*{_SWITCH_B_DX}+cos({_SWITCH_B_ANGLE})*{_SWITCH_B_DY})/{_SWITCH_SCALE}+H/2)"
_SWITCH_A_INSIDE = f"between({_SWITCH_A_X},0,W-1)*between({_SWITCH_A_Y},0,H-1)"
_SWITCH_B_INSIDE = f"between({_SWITCH_B_X},0,W-1)*between({_SWITCH_B_Y},0,H-1)"
_SWITCH_A_PANEL = _sample("a", _SWITCH_A_X, _SWITCH_A_Y)
_SWITCH_B_PANEL = _sample("b", _SWITCH_B_X, _SWITCH_B_Y)
SWITCH_EXPRESSION = _endpoint(
    f"if(lt({_PROGRESS},0.5),"
    f"if({_SWITCH_A_INSIDE},{_SWITCH_A_PANEL},if({_SWITCH_B_INSIDE},{_SWITCH_B_PANEL},0)),"
    f"if({_SWITCH_B_INSIDE},{_SWITCH_B_PANEL},if({_SWITCH_A_INSIDE},{_SWITCH_A_PANEL},0)))"
)


def _switch_expression(direction: int) -> str:
    """Return From Left (1) or the horizontally opposed From Right (2)."""

    if direction == 1:
        return SWITCH_EXPRESSION
    separation = f"sin(PI*{_PROGRESS})"
    scale = f"(0.84+0.16*abs(2*{_PROGRESS}-1))"
    a_center_x = f"(W/2+0.45*W*{separation})"
    b_center_x = f"(W/2-0.45*W*{separation})"
    a_center_y = f"(H/2-0.55*H*{separation})"
    b_center_y = f"(H/2+0.55*H*{separation})"
    a_angle = f"(0.45*{separation})"
    b_angle = f"(-0.45*{separation})"
    a_x, a_y = _inverse_card_coordinates(a_center_x, a_center_y, a_angle, scale)
    b_x, b_y = _inverse_card_coordinates(b_center_x, b_center_y, b_angle, scale)
    a_inside = f"between({a_x},0,W-1)*between({a_y},0,H-1)"
    b_inside = f"between({b_x},0,W-1)*between({b_y},0,H-1)"
    a_panel = _sample("a", a_x, a_y)
    b_panel = _sample("b", b_x, b_y)
    return _endpoint(
        f"if(lt({_PROGRESS},0.5),if({a_inside},{a_panel},"
        f"if({b_inside},{b_panel},0)),if({b_inside},{b_panel},"
        f"if({a_inside},{a_panel},0)))"
    )


def _inverse_card_coordinates(
    center_x: str,
    center_y: str,
    angle: str,
    scale: str,
) -> tuple[str, str]:
    """Return inverse card coordinates for translated panel centers."""

    dx = f"(X-{center_x})"
    dy = f"(Y-{center_y})"
    x = f"((cos({angle})*{dx}+sin({angle})*{dy})/{scale}+W/2)"
    y = f"((-sin({angle})*{dx}+cos({angle})*{dy})/{scale}+H/2)"
    return x, y

_SWAP_ACTIVITY = f"sin(PI*{_PROGRESS})"
_SWAP_SCALE_X = f"(1-0.10*{_SWAP_ACTIVITY})"
_SWAP_SCALE_Y = f"(1-0.04*sin(PI*{_PROGRESS}))"
_SWAP_A_CENTER_X = f"(W/2+W*{_PROGRESS})"
_SWAP_B_CENTER_X = f"(-W/2+W*{_PROGRESS})"
_SWAP_A_PERSPECTIVE = f"(1+0.22*{_SWAP_ACTIVITY}*(Y/H-0.5))"
_SWAP_B_PERSPECTIVE = f"(1-0.22*{_SWAP_ACTIVITY}*(Y/H-0.5))"
_SWAP_A_X = f"((X-{_SWAP_A_CENTER_X})/({_SWAP_SCALE_X}*{_SWAP_A_PERSPECTIVE})+W/2)"
_SWAP_B_X = f"((X-{_SWAP_B_CENTER_X})/({_SWAP_SCALE_X}*{_SWAP_B_PERSPECTIVE})+W/2)"
_SWAP_Y = f"((Y-H/2)/{_SWAP_SCALE_Y}+H/2)"
_SWAP_A_INSIDE = f"between({_SWAP_A_X},0,W-1)*between({_SWAP_Y},0,H-1)"
_SWAP_B_INSIDE = f"between({_SWAP_B_X},0,W-1)*between({_SWAP_Y},0,H-1)"
_SWAP_A_PANEL = _sample("a", _SWAP_A_X, _SWAP_Y)
_SWAP_B_PANEL = _sample("b", _SWAP_B_X, _SWAP_Y)
SWAP_EXPRESSION = _endpoint(
    f"if(lt({_PROGRESS},0.5),"
    f"if({_SWAP_A_INSIDE},{_SWAP_A_PANEL},if({_SWAP_B_INSIDE},{_SWAP_B_PANEL},0)),"
    f"if({_SWAP_B_INSIDE},{_SWAP_B_PANEL},if({_SWAP_A_INSIDE},{_SWAP_A_PANEL},0)))"
)


def _swap_expression(direction: int) -> str:
    """Return the reviewed Right path or its horizontally opposed Left path."""

    if direction == 1:
        return SWAP_EXPRESSION
    activity = f"sin(PI*{_PROGRESS})"
    scale_x = f"(1-0.10*{activity})"
    scale_y = f"(1-0.04*{activity})"
    a_center_x = f"(3*W/2-W*{_PROGRESS})"
    b_center_x = f"(W/2+W*{_PROGRESS})"
    a_perspective = f"(1-0.22*{activity}*(Y/H-0.5))"
    b_perspective = f"(1+0.22*{activity}*(Y/H-0.5))"
    a_x = f"((X-{a_center_x})/({scale_x}*{a_perspective})+W/2)"
    b_x = f"((X-{b_center_x})/({scale_x}*{b_perspective})+W/2)"
    source_y = f"((Y-H/2)/{scale_y}+H/2)"
    a_inside = f"between({a_x},0,W-1)*between({source_y},0,H-1)"
    b_inside = f"between({b_x},0,W-1)*between({source_y},0,H-1)"
    a_panel = _sample("a", a_x, source_y)
    b_panel = _sample("b", b_x, source_y)
    return _endpoint(
        f"if(lt({_PROGRESS},0.5),if({a_inside},{a_panel},"
        f"if({b_inside},{b_panel},0)),if({b_inside},{b_panel},"
        f"if({a_inside},{a_panel},0)))"
    )


# Color, light, blur, and deterministic disturbance -----------------------

_HASH = (
    f"abs(mod(sin(X*12.9898+Y*78.233+floor({_PROGRESS}*30)*37.719)"
    f"*43758.5453,1))"
)
_STATIC_ACTIVITY = f"pow(sin(PI*{_PROGRESS}),1.50)"
_STATIC_BASE = f"if(lt({_PROGRESS},0.54),A,B)"
_STATIC_SNOW = (
    f"255*clip(0.93*{_HASH}+0.07*(0.5+0.5*sin(Y*0.035+"
    f"floor({_PROGRESS}*30)*2.3)),0,1)"
)
STATIC_EXPRESSION = _endpoint(
    f"((1-{_STATIC_ACTIVITY})*{_STATIC_BASE}+"
    f"{_STATIC_ACTIVITY}*{_STATIC_SNOW})"
)


def _static_expression(style: int) -> str:
    """Return Final Cut's two published static texture styles."""

    if style == 0:
        return STATIC_EXPRESSION
    activity = f"pow(sin(PI*{_PROGRESS}),1.25)"
    base = f"if(lt({_PROGRESS},0.50),A,B)"
    coarse_hash = (
        f"abs(mod(sin(floor(X/3)*91.713+floor(Y/2)*17.137+"
        f"floor({_PROGRESS}*24)*41.271)*43758.5453,1))"
    )
    scan = f"(0.5+0.5*sin(Y*0.70+floor({_PROGRESS}*24)*1.9))"
    snow = f"255*clip(0.72*{coarse_hash}+0.28*{scan},0,1)"
    return _endpoint(f"((1-{activity})*{base}+{activity}*{snow})")

_QUAKE_X = f"(X+18*sin(19*PI*{_PROGRESS})*{_MID_ACTIVITY})"
_QUAKE_Y = f"(Y+10*sin(27*PI*{_PROGRESS})*{_MID_ACTIVITY})"
_QUAKE_ECHO_X = f"({_QUAKE_X}+12*{_MID_ACTIVITY})"
_QUAKE_ECHO_Y = f"({_QUAKE_Y}-7*{_MID_ACTIVITY})"
_QUAKE_A = f"(0.78*{_sample('a', _QUAKE_X, _QUAKE_Y)}+0.22*{_sample('a', _QUAKE_ECHO_X, _QUAKE_ECHO_Y)})"
_QUAKE_B = f"(0.78*{_sample('b', _QUAKE_X, _QUAKE_Y)}+0.22*{_sample('b', _QUAKE_ECHO_X, _QUAKE_ECHO_Y)})"
_QUAKE_TRANSITION = _smoothstep(f"(({_PROGRESS}-0.52)/0.16)")
_QUAKE_SMOKE = (
    f"(80*clip(({_PROGRESS}-0.42)/0.30,0,1)*sin(PI*{_PROGRESS})*"
    f"exp(-(H-Y)/max(1,0.22*H)))"
)
EARTHQUAKE_EXPRESSION = _endpoint(
    f"min(255,{_mix(_QUAKE_A, _QUAKE_B, f'1-{_QUAKE_TRANSITION}')}+{_QUAKE_SMOKE})"
)

_FLASHBACK_ACTIVITY = f"pow(sin(PI*{_PROGRESS}),0.75)"
_FLASHBACK_WAVE_X = (
    f"(0.055*W*{_FLASHBACK_ACTIVITY}*(0.65*sin(0.045*Y+"
    f"8*PI*{_PROGRESS})+0.35*sin(0.025*(X+Y)-6*PI*{_PROGRESS})))"
)
_FLASHBACK_WAVE_Y = (
    f"(0.045*H*{_FLASHBACK_ACTIVITY}*(0.62*sin(0.042*X-"
    f"9*PI*{_PROGRESS})+0.38*sin(0.022*(X-Y)+7*PI*{_PROGRESS})))"
)
_FLASHBACK_X = f"(X+{_FLASHBACK_WAVE_X})"
_FLASHBACK_Y = f"(Y+{_FLASHBACK_WAVE_Y})"
_FLASHBACK_A = _sample("a", _FLASHBACK_X, _FLASHBACK_Y)
_FLASHBACK_B = _sample("b", _FLASHBACK_X, _FLASHBACK_Y)
_FLASHBACK_HANDOFF = _smoothstep(f"(({_PROGRESS}-0.40)/0.22)")
_FLASHBACK_BASE = _mix("A", "B", f"1-{_FLASHBACK_HANDOFF}")
_FLASHBACK_WARP = _mix(
    _FLASHBACK_A, _FLASHBACK_B, f"1-{_FLASHBACK_HANDOFF}"
)
_FLASHBACK_ECHO_X = f"((X-W/2)/1.035+W/2-0.52*{_FLASHBACK_WAVE_X})"
_FLASHBACK_ECHO_Y = f"((Y-H/2)/1.035+H/2-0.52*{_FLASHBACK_WAVE_Y})"
_FLASHBACK_ECHO_A = _sample("a", _FLASHBACK_ECHO_X, _FLASHBACK_ECHO_Y)
_FLASHBACK_ECHO_B = _sample("b", _FLASHBACK_ECHO_X, _FLASHBACK_ECHO_Y)
_FLASHBACK_ECHO = _mix(
    _FLASHBACK_ECHO_A, _FLASHBACK_ECHO_B, f"1-{_FLASHBACK_HANDOFF}"
)
_FLASHBACK_BURN_X = f"(W*(0.32+0.25*{_PROGRESS}))"
_FLASHBACK_BURN_Y = f"(H*(0.58-0.20*{_PROGRESS}))"
_FLASHBACK_RADIUS = (
    f"hypot(X-{_FLASHBACK_BURN_X},Y-{_FLASHBACK_BURN_Y})"
)
_FLASHBACK_RING_RADIUS = f"((0.08+0.62*{_PROGRESS})*min(W,H))"
_FLASHBACK_RING = (
    f"(82*exp(-abs({_FLASHBACK_RADIUS}-{_FLASHBACK_RING_RADIUS})/"
    f"max(1,0.055*min(W,H)))*{_FLASHBACK_ACTIVITY})"
)
FLASHBACK_EXPRESSION = _endpoint(
    f"min(255,0.12*{_FLASHBACK_BASE}+0.70*{_FLASHBACK_WARP}+"
    f"0.18*{_FLASHBACK_ECHO}+34*pow(sin(PI*{_PROGRESS}),4)+"
    f"{_FLASHBACK_RING})"
)

_COLOR_PHASE = "if(eq(PLANE,0),-1,if(eq(PLANE,1),0,1))"
_COLOR_COLLAPSE = _smoothstep(f"(({_PROGRESS}-0.30)/0.20)")
_COLOR_EXPAND = _smoothstep(f"(({_PROGRESS}-0.50)/0.20)")
_COLOR_CARD_SCALE = (
    f"if(lt({_PROGRESS},0.50),0.91-0.88*{_COLOR_COLLAPSE},"
    f"0.03+0.97*{_COLOR_EXPAND})"
)
_COLOR_SCALE = f"max(0.03,{_COLOR_CARD_SCALE})"
_COLOR_SETTLE = _smoothstep(f"(({_PROGRESS}-0.50)/0.20)")
_COLOR_OFFSET_ACTIVITY = f"(1-{_smoothstep(f'(({_PROGRESS}-0.70)/0.20)')})"
_COLOR_TRAVEL_CENTER = (
    f"(W*(-0.10+1.20*{_PROGRESS})+0.16*W*{_COLOR_PHASE})"
)
_COLOR_SETTLED_CENTER = (
    f"(W/2+0.035*W*{_COLOR_PHASE}*{_COLOR_OFFSET_ACTIVITY})"
)
_COLOR_CENTER = (
    f"((1-{_COLOR_SETTLE})*{_COLOR_TRAVEL_CENTER}+"
    f"{_COLOR_SETTLE}*{_COLOR_SETTLED_CENTER})"
)
_COLOR_ANGLE = (
    f"((0.22+0.16*{_COLOR_PHASE})*sin(PI*{_PROGRESS})*"
    f"(1-{_COLOR_SETTLE}))"
)
_COLOR_DX = f"(X-{_COLOR_CENTER})"
_COLOR_DY = "(Y-H/2)"
_COLOR_PERSPECTIVE = (
    f"(1+0.18*{_COLOR_PHASE}*sin(PI*{_PROGRESS})*"
    f"(1-{_COLOR_SETTLE})*(2*Y/H-1))"
)
_COLOR_X = (
    f"((cos({_COLOR_ANGLE})*{_COLOR_DX}+sin({_COLOR_ANGLE})*{_COLOR_DY})/"
    f"({_COLOR_SCALE}*{_COLOR_PERSPECTIVE})+W/2)"
)
_COLOR_Y = (
    f"((-sin({_COLOR_ANGLE})*{_COLOR_DX}+cos({_COLOR_ANGLE})*{_COLOR_DY})/"
    f"(0.94+0.06*{_COLOR_SETTLE})+H/2)"
)
_COLOR_INSIDE = f"between({_COLOR_X},0,W-1)*between({_COLOR_Y},0,H-1)"
_COLOR_SAMPLE_A = _sample("a", _COLOR_X, _COLOR_Y)
_COLOR_SAMPLE_B = _sample("b", _COLOR_X, _COLOR_Y)
_COLOR_SLIT = f"between({_PROGRESS},0.42,0.54)"
COLOR_PLANES_EXPRESSION = _endpoint(
    f"if({_COLOR_INSIDE},if(lt({_PROGRESS},0.48),"
    f"if({_COLOR_SLIT}*gt(PLANE,0),0,{_COLOR_SAMPLE_A}),"
    f"if({_COLOR_SLIT}*gt(PLANE,0),0,{_COLOR_SAMPLE_B})),0)"
)

_SMEAR_LOCAL_PROGRESS = f"clip({_PROGRESS}/0.23,0,1)"
_SMEAR_ACTIVITY = f"pow(sin(PI*{_SMEAR_LOCAL_PROGRESS}),1.15)"
_SMEAR_X1 = f"((X-W/2)/(1+1.5*{_SMEAR_ACTIVITY})+W/2)"
_SMEAR_Y1 = f"((Y-H/2)/(1+3.8*{_SMEAR_ACTIVITY})+H/2)"
_SMEAR_X2 = f"((X-W/2)/(1+0.75*{_SMEAR_ACTIVITY})+W/2)"
_SMEAR_Y2 = f"((Y-H/2)/(1+1.9*{_SMEAR_ACTIVITY})+H/2)"
_SMEAR_BASE_X = f"((X-W/2)/(1+0.30*{_SMEAR_ACTIVITY})+W/2)"
_SMEAR_BASE_Y = f"((Y-H/2)/(1+0.70*{_SMEAR_ACTIVITY})+H/2)"
_SMEAR_A = (
    f"(0.46*{_sample('a', _SMEAR_X1, _SMEAR_Y1)}+"
    f"0.31*{_sample('a', _SMEAR_X2, _SMEAR_Y2)}+"
    f"0.23*{_sample('a', _SMEAR_BASE_X, _SMEAR_BASE_Y)})"
)
_SMEAR_B = (
    f"(0.46*{_sample('b', _SMEAR_X1, _SMEAR_Y1)}+"
    f"0.31*{_sample('b', _SMEAR_X2, _SMEAR_Y2)}+"
    f"0.23*{_sample('b', _SMEAR_BASE_X, _SMEAR_BASE_Y)})"
)
_SMEAR_HANDOFF = _smoothstep(f"(({_PROGRESS}-0.14)/0.09)")
SMEAR_EXPRESSION = _endpoint(
    f"if(gte({_PROGRESS},0.23),B,"
    f"((1-{_SMEAR_HANDOFF})*{_SMEAR_A}+{_SMEAR_HANDOFF}*B))"
)


# Replication and object-shaped procedural mattes -------------------------

_ARROW_CELL_W = "(W/3)"
_ARROW_CELL_H = "(H/4)"
_ARROW_LOCAL_X = f"(mod(X,{_ARROW_CELL_W})/{_ARROW_CELL_W}-0.5)"
_ARROW_LOCAL_Y = f"(mod(Y,{_ARROW_CELL_H})/{_ARROW_CELL_H}-0.5)"
_ARROW_GROW = f"clip({_PROGRESS}/0.40,0,1)"
_ARROW_SHRINK = f"clip(({_PROGRESS}-0.57)/0.28,0,1)"
_ARROW_SCALE = (
    f"max(0.02,if(lt({_PROGRESS},0.57),{_ARROW_GROW},1-{_ARROW_SHRINK}))"
)
_ARROW_CENTER = (
    f"if(lt({_PROGRESS},0.57),-0.82*(1-{_ARROW_GROW}),"
    f"0.82*{_ARROW_SHRINK})"
)
_ARROW_U = (
    f"({_ARROW_LOCAL_Y}-{_ARROW_CENTER})"
)
_ARROW_V = f"(-{_ARROW_LOCAL_X})"
_ARROW_EXTENT = f"(0.62*{_ARROW_SCALE})"
_ARROW_HEAD = (
    f"gt({_ARROW_U},-0.08*{_ARROW_SCALE})*"
    f"lt({_ARROW_U},{_ARROW_EXTENT})*"
    f"lt(abs({_ARROW_V}),0.62*({_ARROW_EXTENT}-{_ARROW_U}))"
)
_ARROW_SHAFT = (
    f"gt({_ARROW_U},-{_ARROW_EXTENT})*"
    f"lte({_ARROW_U},0.08*{_ARROW_SCALE})*"
    f"lt(abs({_ARROW_V}),0.18*{_ARROW_SCALE})"
)
_ARROW_TRIANGLE = (
    f"({_ARROW_HEAD}+{_ARROW_SHAFT})"
)
_ARROW_BASE = f"if(lt({_PROGRESS},0.57),A,B)"
_ARROW_SHARD = f"if(lt({_PROGRESS},0.57),B,A)"
ARROWS_EXPRESSION = _endpoint(
    f"if({_ARROW_TRIANGLE},{_ARROW_SHARD},{_ARROW_BASE})"
)


def _arrows_expression(end_cap: int, motion_blur: bool) -> str:
    """Build the five published end-cap silhouettes and optional trail."""

    if end_cap == 0 and not motion_blur:
        return ARROWS_EXPRESSION
    extent = _ARROW_EXTENT
    round_extent = f"(0.70*{extent})"
    square_extent = f"(0.48*{extent})"
    bevel_extent = f"(1.08*{extent})"
    shapes = {
        0: _ARROW_TRIANGLE,
        3: f"lt(hypot({_ARROW_U},{_ARROW_V}),{round_extent})",
        4: (
            f"lt(abs({_ARROW_U}),{square_extent})*"
            f"lt(abs({_ARROW_V}),{square_extent})"
        ),
        5: (
            f"gt({_ARROW_U},-{extent})*lt({_ARROW_U},{extent})*"
            f"lt(abs({_ARROW_V}),0.38*{_ARROW_SCALE})"
        ),
        6: (
            f"lt(abs({_ARROW_U}),{bevel_extent})*"
            f"lt(abs({_ARROW_V}),({bevel_extent}-0.35*abs({_ARROW_U})))"
        ),
    }
    shape = shapes[end_cap]
    if motion_blur:
        trail_u = f"({_ARROW_U}+0.22*{_ARROW_SCALE})"
        trail = (
            f"gt({trail_u},-{extent})*lt({trail_u},{extent})*"
            f"lt(abs({_ARROW_V}),0.35*({extent}-{trail_u}))"
        )
        shard = f"if({shape},{_ARROW_SHARD},if({trail},0.42*{_ARROW_SHARD},{_ARROW_BASE}))"
        return _endpoint(shard)
    return _endpoint(f"if({shape},min(255,0.88*{_ARROW_SHARD}+28),{_ARROW_BASE})")

_VEIL_BASE = f"if(lt({_PROGRESS},0.40),A,B)"
_VEIL_CENTER_Y = (
    f"(H*(-0.15+1.40*{_PROGRESS})+0.10*H*"
    f"sin(2*PI*X/W+3*PI*{_PROGRESS}))"
)
_VEIL_HALF_WIDTH = f"(H*(0.075+0.16*sin(PI*{_PROGRESS})))"
_VEIL_DISTANCE = f"abs(Y-{_VEIL_CENTER_Y})"
_VEIL_OPACITY = f"(0.56*pow(clip(1-{_VEIL_DISTANCE}/{_VEIL_HALF_WIDTH},0,1),0.55))"
_VEIL_WHITE = (
    f"(216+24*(0.5+0.5*sin(12*(Y-{_VEIL_CENTER_Y})/"
    f"max(1,{_VEIL_HALF_WIDTH})+4*PI*{_PROGRESS})))"
)
VEIL_EXPRESSION = _endpoint(
    f"clip((1-{_VEIL_OPACITY})*({_VEIL_BASE})+{_VEIL_OPACITY}*{_VEIL_WHITE},0,255)"
)

_CURTAIN_CLOSE = _smoothstep(f"({_PROGRESS}/0.24)")
_CURTAIN_REOPEN = _smoothstep(f"(({_PROGRESS}-0.68)/0.22)")
_CURTAIN_GAP_FACTOR = (
    f"if(lt({_PROGRESS},0.34),1-{_CURTAIN_CLOSE},{_CURTAIN_REOPEN})"
)
_CURTAIN_GAP = f"(W/2*{_CURTAIN_GAP_FACTOR})"
_CURTAIN_FOLD = "(0.58+0.42*(0.5+0.5*cos(0.16*X)))"
_CURTAIN_RED = (
    f"if(eq(PLANE,0),12*{_CURTAIN_FOLD},"
    f"if(eq(PLANE,1),8*{_CURTAIN_FOLD},"
    f"if(eq(PLANE,2),150*{_CURTAIN_FOLD},255)))"
)
CURTAINS_EXPRESSION = _endpoint(
    f"if(lt(abs(X-W/2),{_CURTAIN_GAP}),"
    f"if(lt({_PROGRESS},0.5),A,B),{_CURTAIN_RED})"
)


def _curtains_expression(animation: int) -> str:
    """Return Open & Close, Open Only, or Close Only curtain timing."""

    if animation == 0:
        return CURTAINS_EXPRESSION
    if animation == 1:
        open_amount = _smoothstep(_PROGRESS)
        gap = f"(W/2*{open_amount})"
        return _endpoint(f"if(lt(abs(X-W/2),{gap}),B,{_CURTAIN_RED})")
    close_amount = _smoothstep(f"(1-{_PROGRESS})")
    gap = f"(W/2*{close_amount})"
    base = f"if(lt({_PROGRESS},0.5),A,B)"
    return _endpoint(f"if(lt(abs(X-W/2),{gap}),{base},{_CURTAIN_RED})")


NATIVE_IMPLEMENTATIONS: Mapping[str, str] = {}

CUSTOM_IMPLEMENTATIONS: Mapping[str, str] = {
    **PANEL_MOTION_IMPLEMENTATIONS,
    **LIGHT_DECO_IMPLEMENTATIONS,
    "cohort_center_default": CENTER_EXPRESSION,
    "cohort_clock_default": CLOCK_EXPRESSION,
    "cohort_page_curl_default": PAGE_CURL_EXPRESSION,
    "cohort_swap_default": SWAP_EXPRESSION,
    "cohort_static_default": STATIC_EXPRESSION,
    "cohort_rotate_default": ROTATE_EXPRESSION,
    "cohort_swing_default": SWING_EXPRESSION,
    "cohort_switch_default": SWITCH_EXPRESSION,
    "cohort_arrows_default": ARROWS_EXPRESSION,
    "cohort_curtains_default": CURTAINS_EXPRESSION,
    "cohort_veil_default": VEIL_EXPRESSION,
}

PREFILTER_IMPLEMENTATIONS: Mapping[str, tuple[str, float, float]] = {
    "cohort_gaussian_default": ("gaussian_blur", 0.10, 0.0),
    "cohort_radial_default": ("radial_spin", 0.58, 0.0),
    "cohort_earthquake_default": ("earthquake_shake", 0.028, 0.018),
    "cohort_drop_in_default": ("drop_in_panel", 0.45, 0.0),
    "cohort_color_planes_default": ("color_planes_cards", 0.0, 0.0),
    "cohort_light_noise_default": ("light_noise_pulses", 0.0, 0.0),
    "cohort_leaves_default": ("leaves_sprite", 0.0, 0.0),
    "cohort_flashback_default": ("liquid_ripple", 0.055, 0.045),
    "cohort_smear_default": ("smear_streak", 0.23, 0.0),
}

IMPLEMENTATION_IDS = frozenset(
    {*NATIVE_IMPLEMENTATIONS, *CUSTOM_IMPLEMENTATIONS, *PREFILTER_IMPLEMENTATIONS}
)


# Exact Final Cut serialization keys.  These constants intentionally live next
# to the dispatch boundary: the compiler validates each value against the
# registry before this module selects any renderer-owned expression or filter.
CENTER_ANGLE_KEY = "3"
CENTER_DIRECTION_KEY = "7"
CENTER_BORDER_KEY = "16/1"
CENTER_EDGE_TYPE_KEY = "16/15"
CLOCK_ANGLE_KEY = "3"
CLOCK_DIRECTION_KEY = "6"
CLOCK_BORDER_KEY = "16/1"
CLOCK_EDGE_TYPE_KEY = "16/15"
PAGE_CURL_PRESET_KEY = "1"
PAGE_CURL_DIRECTION_KEY = "9"
SWAP_DIRECTION_KEY = "3"
STATIC_STYLE_KEY = "9999/987200168/100/987200169/2/100"
DROP_IN_SMOKE_KEY = "9999/999120132/100/999120133/2/100"
EARTHQUAKE_SMOKE_KEY = "9999/980604875/100/980604876/2/100"
ROTATE_DIRECTION_KEY = "9999/987639695/100/987639696/2/100"
ROTATE_BACKGROUND_KEY = "9999/987639695/100/1978700478/2/100"
SMEAR_DIRECTION_KEY = "9999/989648291/100/989648292/2/100"
SWING_ANCHOR_KEY = "9999/987619294/100/987619295/2/100"
SWING_DIRECTION_KEY = "9999/987619294/100/987619296/2/100"
SWING_BACKGROUND_KEY = "9999/987619294/100/1978700512/2/100"
SWITCH_DIRECTION_KEY = "9999/1999871391/100/1999871392/2/100"
ARROWS_END_CAP_KEY = "9999/1979006420/100/1979006421/2/100"
ARROWS_MOTION_BLUR_KEY = "9999/1979006420/100/1979006503/2/100"
CURTAINS_ANIMATION_KEY = "9999/989454316/100/989454317/2/100"
LEAVES_SEASON_KEY = "9999/987200026/100/987200027/2/100"


def build_cohort_transition_plan(
    implementation_id: str,
    parameter_values: Mapping[str, Any] | None = None,
) -> CohortTransitionPlan | None:
    """Resolve a cohort implementation ID without accepting source syntax.

    Main callers:
    - ``transitions.stock.build_stock_transition_plan`` after registry and
      compiler validation.
    """

    values = parameter_values or {}
    native = NATIVE_IMPLEMENTATIONS.get(implementation_id)
    if native is not None:
        return CohortTransitionPlan(mode=native)

    panel_expression = build_panel_motion_expression(implementation_id, values)
    if panel_expression is not None:
        return CohortTransitionPlan(mode="custom", expression=panel_expression)

    dynamic_expressions = {
        "cohort_center_default": lambda: _center_expression(
            angle_degrees=0.0,
            direction=int(values.get(CENTER_DIRECTION_KEY, 0)),
            border=15.0,
            feather=int(values.get(CENTER_EDGE_TYPE_KEY, 1)) == 1,
        ),
        "cohort_clock_default": lambda: _clock_expression(
            angle_degrees=0.0,
            counterclockwise=int(values.get(CLOCK_DIRECTION_KEY, 0)) == 1,
            border=15.0,
            feather=int(values.get(CLOCK_EDGE_TYPE_KEY, 1)) == 1,
        ),
        "cohort_page_curl_default": lambda: _page_curl_expression(
            int(values.get(PAGE_CURL_PRESET_KEY, 0)),
            int(values.get(PAGE_CURL_DIRECTION_KEY, 0)),
        ),
        "cohort_swap_default": lambda: _swap_expression(
            int(values.get(SWAP_DIRECTION_KEY, 1))
        ),
        "cohort_static_default": lambda: _static_expression(
            int(values.get(STATIC_STYLE_KEY, 0))
        ),
        "cohort_rotate_default": lambda: _parameterized_rotate_expression(
            counterclockwise=int(values.get(ROTATE_DIRECTION_KEY, 0)) == 1,
            black_background=bool(values.get(ROTATE_BACKGROUND_KEY, False)),
        ),
        "cohort_swing_default": lambda: _parameterized_swing_expression(
            anchor=int(values.get(SWING_ANCHOR_KEY, 2)),
            towards=int(values.get(SWING_DIRECTION_KEY, 1)) == 0,
            black_background=bool(values.get(SWING_BACKGROUND_KEY, False)),
        ),
        "cohort_switch_default": lambda: _switch_expression(
            int(values.get(SWITCH_DIRECTION_KEY, 1))
        ),
        "cohort_arrows_default": lambda: _arrows_expression(
            int(values.get(ARROWS_END_CAP_KEY, 0)),
            bool(values.get(ARROWS_MOTION_BLUR_KEY, False)),
        ),
        "cohort_curtains_default": lambda: _curtains_expression(
            int(values.get(CURTAINS_ANIMATION_KEY, 0))
        ),
    }
    dynamic = dynamic_expressions.get(implementation_id)
    if dynamic is not None:
        return CohortTransitionPlan(mode="custom", expression=dynamic())

    expression = CUSTOM_IMPLEMENTATIONS.get(implementation_id)
    if expression is not None:
        return CohortTransitionPlan(mode="custom", expression=expression)

    prefilter = PREFILTER_IMPLEMENTATIONS.get(implementation_id)
    if prefilter is not None:
        name, strength, spread = prefilter
        if implementation_id == "cohort_drop_in_default" and not bool(
            values.get(DROP_IN_SMOKE_KEY, True)
        ):
            name = "drop_in_panel_no_smoke"
        elif implementation_id == "cohort_earthquake_default" and not bool(
            values.get(EARTHQUAKE_SMOKE_KEY, True)
        ):
            name = "earthquake_shake_no_smoke"
        elif implementation_id == "cohort_smear_default" and int(
            values.get(SMEAR_DIRECTION_KEY, 1)
        ) == 0:
            name = "smear_streak_left"
        elif implementation_id == "cohort_leaves_default":
            season = int(values.get(LEAVES_SEASON_KEY, 0))
            name = {
                0: "leaves_sprite",
                1: "leaves_sprite_summer",
                2: "leaves_sprite_fall",
                3: "leaves_sprite_winter",
            }[season]
        return CohortTransitionPlan(
            mode="fade",
            prefilter=name,
            strength=strength,
            spread=spread,
        )
    return None
