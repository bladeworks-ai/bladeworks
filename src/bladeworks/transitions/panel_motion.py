"""Bounded stock-FFmpeg approximations for panel-motion transitions.

Architecture map
================

trusted capability-registry implementation ID
    -> one renderer-owned expression in ``P``, ``X``, and ``Y``
    -> FFmpeg's stock ``xfade=custom`` evaluator
    -> exact recursively composed outgoing/incoming endpoint frames.

The expressions approximate Final Cut's private 2D/3D panel animations with
inverse affine sampling, geometric mattes, and deterministic source switches.
They never consume shader text, filter syntax, or file paths from FCPXML.

Why this exists
---------------
These transitions share a useful tuning boundary: their recognizable behavior
comes from moving or folding whole image panels.  Keeping them together makes
their motion curves independently testable without growing the mixed flat,
light, and procedural cohort module.
"""

from __future__ import annotations

from typing import Any, Mapping


def _sample(source: str, x: str = "X", y: str = "Y") -> str:
    """Sample one source while keeping FFmpeg's four planes independent."""

    safe_x = f"clip({x},0,W-1)"
    safe_y = f"clip({y},0,H-1)"
    return (
        f"if(eq(PLANE,0),{source}0({safe_x},{safe_y}),"
        f"if(eq(PLANE,1),{source}1({safe_x},{safe_y}),"
        f"if(eq(PLANE,2),{source}2({safe_x},{safe_y}),"
        f"{source}3({safe_x},{safe_y}))))"
    )


def _endpoint(body: str) -> str:
    """Return exact source frames at both boundaries of FFmpeg's transition."""

    return f"if(gte(P,1),A,if(lte(P,0),B,{body}))"


def _smoothstep(value: str) -> str:
    """Return a bounded cubic ease using stock expression primitives."""

    bounded = f"clip({value},0,1)"
    return f"(({bounded})*({bounded})*(3-2*({bounded})))"


def _affine_coordinates(
    *,
    angle: str,
    scale: str,
    center_x: str = "W/2",
    center_y: str = "H/2",
) -> tuple[str, str]:
    """Return inverse coordinates for a card rotated/scaled about a pivot."""

    dx = f"(X-{center_x})"
    dy = f"(Y-{center_y})"
    x = f"((cos({angle})*{dx}+sin({angle})*{dy})/max(0.01,{scale})+W/2)"
    y = f"((-sin({angle})*{dx}+cos({angle})*{dy})/max(0.01,{scale})+H/2)"
    return x, y


_PROGRESS = "(1-P)"


# Divide ------------------------------------------------------------------

_DIVIDE_COLUMN = "min(2,floor(3*X/W))"
_DIVIDE_PHASE = (
    f"(if(eq({_DIVIDE_COLUMN},0),0,if(eq({_DIVIDE_COLUMN},1),0.24,0.12)))"
)
_DIVIDE_PROGRESS = (
    f"clip(({_PROGRESS}-{_DIVIDE_PHASE})/(0.84-{_DIVIDE_PHASE}),0,1)"
)
_DIVIDE_SOURCE_Y = (
    f"if(eq({_DIVIDE_COLUMN},1),Y-H*(1-{_DIVIDE_PROGRESS}),"
    f"Y+H*(1-{_DIVIDE_PROGRESS}))"
)
_DIVIDE_INSIDE = f"between({_DIVIDE_SOURCE_Y},0,H-1)"
_DIVIDE_LOCAL_X = "mod(X,W/3)"
_DIVIDE_BORDER = (
    f"lt({_DIVIDE_LOCAL_X},0.012*W)+gt({_DIVIDE_LOCAL_X},W/3-0.012*W)"
)
_DIVIDE_SETTLED = f"gte({_DIVIDE_PROGRESS},0.985)"
DIVIDE_EXPRESSION = _endpoint(
    f"if({_DIVIDE_INSIDE},if(({_DIVIDE_BORDER})*(1-{_DIVIDE_SETTLED}),0,"
    f"{_sample('b', 'X', _DIVIDE_SOURCE_Y)}),A)"
)


def _divide_expression(section_value: int) -> str:
    """Build the exact published four-, three-, or two-section variant.

    Final Cut serializes the menu labels ``4``, ``3``, and ``2`` as values
    ``0``, ``1``, and ``2``. The three-section choice has the strongest
    stagger in Final Cut's real-clip sweep; the two-section choice moves as an
    opposing pair. Keeping those phase profiles separate preserves the
    observed response ordering without changing the labeled panel count.
    """

    section_count = {0: 4, 1: 3, 2: 2}[section_value]
    column = f"min({section_count - 1},floor({section_count}*X/W))"
    phase_step = {0: 0.10, 1: 0.20, 2: 0.06}[section_value]
    phase = f"({phase_step:.2f}*mod(2*{column},{section_count}))"
    progress = f"clip(({_PROGRESS}-{phase})/(0.84-{phase}),0,1)"
    source_y = (
        f"if(eq(mod({column},2),1),Y-H*(1-{progress}),"
        f"Y+H*(1-{progress}))"
    )
    inside = f"between({source_y},0,H-1)"
    local_x = f"mod(X,W/{section_count})"
    border = (
        f"lt({local_x},0.012*W)+"
        f"gt({local_x},W/{section_count}-0.012*W)"
    )
    settled = f"gte({progress},0.985)"
    return _endpoint(
        f"if({inside},if(({border})*(1-{settled}),0,"
        f"{_sample('b', 'X', source_y)}),A)"
    )


# Spin and Clothesline ----------------------------------------------------


def _incoming_spinning_panel_expression() -> str:
    """Expand one rotating incoming card over the outgoing full canvas."""

    motion_progress = _smoothstep(f"({_PROGRESS}/0.85)")
    early_spin = f"pow(clip((0.48-{_PROGRESS})/0.48,0,1),2.2)"
    late_settle = f"clip(({_PROGRESS}-0.33)/0.60,0,1)"
    angle = f"(2*PI*{early_spin}-0.24*sin(PI*{late_settle}))"
    scale = f"max(0.01,pow({motion_progress},1.5))"
    x, y = _affine_coordinates(angle=angle, scale=scale)
    inside = f"between({x},0,W-1)*between({y},0,H-1)"
    panel = _sample("b", x, y)
    return _endpoint(f"if({inside},{panel},A)")


def _outgoing_spinning_panel_expression() -> str:
    """Shrink and rotate the outgoing card to reveal the incoming side."""

    motion_progress = _smoothstep(f"((1-{_PROGRESS})/0.85)")
    early_spin = f"pow(clip({_PROGRESS}/0.48,0,1),2.2)"
    late_settle = f"clip((1-{_PROGRESS}-0.33)/0.60,0,1)"
    angle = f"(-2*PI*{early_spin}+0.24*sin(PI*{late_settle}))"
    scale = f"max(0.01,pow({motion_progress},1.5))"
    x, y = _affine_coordinates(angle=angle, scale=scale)
    inside = f"between({x},0,W-1)*between({y},0,H-1)"
    panel = _sample("a", x, y)
    return _endpoint(f"if({inside},{panel},B)")


def _spin_in_expression() -> str:
    """Expand the incoming card with the explicit, shorter Spin In path.

    ``Automatic`` keeps the already reviewed two-turn default.  Final Cut
    publishes ``In`` as a separate user choice, so it must not collapse to the
    same portable output.  This bounded variant keeps incoming ownership while
    using one shorter turn and an earlier settle.
    """

    motion_progress = _smoothstep(f"({_PROGRESS}/0.72)")
    turn = f"pow(clip((0.42-{_PROGRESS})/0.42,0,1),2.0)"
    settle = f"clip(({_PROGRESS}-0.28)/0.55,0,1)"
    angle = f"(1.30*PI*{turn}-0.16*sin(PI*{settle}))"
    scale = f"max(0.01,pow({motion_progress},1.35))"
    x, y = _affine_coordinates(angle=angle, scale=scale)
    inside = f"between({x},0,W-1)*between({y},0,H-1)"
    return _endpoint(f"if({inside},{_sample('b', x, y)},A)")


def _clothesline_expression() -> str:
    """Swing the outgoing card away, then bring the incoming card from right.

    The genuine default is a two-card handoff with a short black interval.  It
    is not an incoming-only horizontal expansion: the outgoing face collapses
    toward screen-left first, and the incoming face then opens from the right
    edge.  The Y-dependent edge gives both cards the pronounced diagonal hinge
    visible in Final Cut without mirroring either source image.
    """

    close_progress = _smoothstep(f"({_PROGRESS}/0.15)")
    close_slant = f"(0.62*W*sin(PI*{close_progress})*(Y/H-0.5))"
    close_edge = f"clip(W*(1-{close_progress})+{close_slant},0,W)"
    close_x = f"(X/max(1,{close_edge})*W)"
    closing_panel = f"if(lt(X,{close_edge}),{_sample('a', close_x, 'Y')},0)"

    open_progress = _smoothstep(f"(({_PROGRESS}-0.18)/0.27)")
    open_slant = f"(0.46*W*sin(PI*{open_progress})*(Y/H-0.5))"
    open_edge = f"clip(W*(1-{open_progress})+{open_slant},0,W)"
    open_x = f"((X-{open_edge})/max(1,W-{open_edge})*W)"
    opening_panel = f"if(gt(X,{open_edge}),{_sample('b', open_x, 'Y')},0)"
    return _endpoint(
        f"if(lt({_PROGRESS},0.18),{closing_panel},{opening_panel})"
    )


def _reverse_clothesline_expression() -> str:
    """Mirror the two-card handoff without mirroring source pixels."""

    close_progress = _smoothstep(f"({_PROGRESS}/0.15)")
    close_slant = f"(-0.62*W*sin(PI*{close_progress})*(Y/H-0.5))"
    close_width = f"clip(W*(1-{close_progress})+{close_slant},0,W)"
    close_edge = f"(W-{close_width})"
    close_x = f"((X-{close_edge})/max(1,{close_width})*W)"
    closing_panel = f"if(gt(X,{close_edge}),{_sample('a', close_x, 'Y')},0)"

    open_progress = _smoothstep(f"(({_PROGRESS}-0.18)/0.27)")
    open_slant = f"(-0.46*W*sin(PI*{open_progress})*(Y/H-0.5))"
    open_width = f"clip(W*{open_progress}+{open_slant},0,W)"
    open_x = f"(X/max(1,{open_width})*W)"
    opening_panel = f"if(lt(X,{open_width}),{_sample('b', open_x, 'Y')},0)"
    return _endpoint(
        f"if(lt({_PROGRESS},0.18),{closing_panel},{opening_panel})"
    )


SPIN_EXPRESSION = _incoming_spinning_panel_expression()
CLOTHESLINE_EXPRESSION = _clothesline_expression()


# Flip, Scale, and Multi-flip ---------------------------------------------

_FLIP_SCALE = f"max(0.01,pow(abs(cos(PI*{_PROGRESS})),0.58))"
_FLIP_ACTIVITY = f"sin(PI*{_PROGRESS})"
_FLIP_PERSPECTIVE = f"(1+0.16*{_FLIP_ACTIVITY}*(2*Y/H-1))"
_FLIP_CENTER_X = f"(W/2+0.07*W*{_FLIP_ACTIVITY}*(2*Y/H-1))"
_FLIP_X = (
    f"((X-{_FLIP_CENTER_X})/({_FLIP_SCALE}*{_FLIP_PERSPECTIVE})+W/2)"
)
_FLIP_INSIDE = f"between({_FLIP_X},0,W-1)"
FLIP_EXPRESSION = _endpoint(
    f"if({_FLIP_INSIDE},if(lt({_PROGRESS},0.5),"
    f"{_sample('a', _FLIP_X, 'Y')},{_sample('b', _FLIP_X, 'Y')}),0)"
)


def _flip_expression(direction: int) -> str:
    """Return all four published whole-card flip directions."""

    if direction == 0:
        return FLIP_EXPRESSION
    scale_exponent = 3.00 if direction == 1 else 0.58
    scale = (
        f"max(0.01,pow(abs(cos(PI*{_PROGRESS})),{scale_exponent:.2f}))"
    )
    activity = f"sin(PI*{_PROGRESS})"
    if direction in {0, 1}:
        sign = -1 if direction == 1 else 1
        # Final Cut's Left choice produces a much stronger opposite hinge than
        # the nearly symmetric Right/Left draft did on real asymmetric clips.
        perspective_strength = 0.38 if direction == 1 else 0.16
        center_strength = 0.18 if direction == 1 else 0.07
        perspective = (
            f"(1+{sign}*{perspective_strength:.2f}*{activity}*(2*Y/H-1))"
        )
        center = (
            f"(W/2+{sign}*{center_strength:.2f}*W*{activity}*(2*Y/H-1))"
        )
        source_x = f"((X-{center})/({scale}*{perspective})+W/2)"
        inside = f"between({source_x},0,W-1)"
        outgoing = _sample("a", source_x, "Y")
        incoming = _sample("b", source_x, "Y")
    else:
        sign = 1 if direction == 2 else -1
        perspective = f"(1+{sign}*0.16*{activity}*(2*X/W-1))"
        center = f"(H/2+{sign}*0.07*H*{activity}*(2*X/W-1))"
        source_y = f"((Y-{center})/({scale}*{perspective})+H/2)"
        inside = f"between({source_y},0,H-1)"
        outgoing = _sample("a", "X", source_y)
        incoming = _sample("b", "X", source_y)
    return _endpoint(
        f"if({inside},if(lt({_PROGRESS},0.5),{outgoing},{incoming}),0)"
    )

_SCALE_INCOMING = f"max(0.01,pow({_PROGRESS},0.24))"
_SCALE_INCOMING_X = f"((X-W/2)/{_SCALE_INCOMING}+W/2)"
_SCALE_INCOMING_Y = f"((Y-H/2)/{_SCALE_INCOMING}+H/2)"
_SCALE_INCOMING_INSIDE = (
    f"between({_SCALE_INCOMING_X},0,W-1)*"
    f"between({_SCALE_INCOMING_Y},0,H-1)"
)
_SCALE_OPACITY = f"pow({_PROGRESS},0.80)"
SCALE_EXPRESSION = _endpoint(
    f"if({_SCALE_INCOMING_INSIDE},"
    f"{_SCALE_OPACITY}*{_sample('b', _SCALE_INCOMING_X, _SCALE_INCOMING_Y)}+"
    f"(1-{_SCALE_OPACITY})*A,A)"
)


def _scale_expression(direction: int) -> str:
    """Return the four published Scale motion choices.

    ``Up`` keeps the reviewed default. ``Down`` shrinks the outgoing card;
    ``In`` settles an oversized incoming card; ``Out`` enlarges the outgoing
    card while revealing the incoming source.
    """

    if direction == 0:
        return SCALE_EXPRESSION
    if direction == 1:
        scale = f"max(0.01,pow(1-{_PROGRESS},0.24))"
        source_x = f"((X-W/2)/{scale}+W/2)"
        source_y = f"((Y-H/2)/{scale}+H/2)"
        inside = f"between({source_x},0,W-1)*between({source_y},0,H-1)"
        opacity = f"pow(1-{_PROGRESS},0.80)"
        return _endpoint(
            f"if({inside},{opacity}*{_sample('a', source_x, source_y)}+"
            f"(1-{opacity})*B,B)"
        )
    if direction == 2:
        scale = f"(1.75-0.75*{_smoothstep(_PROGRESS)})"
        source_x = f"((X-W/2)/{scale}+W/2)"
        source_y = f"((Y-H/2)/{scale}+H/2)"
        opacity = _smoothstep(_PROGRESS)
        return _endpoint(
            f"{opacity}*{_sample('b', source_x, source_y)}+(1-{opacity})*A"
        )
    scale = f"(1+0.75*{_smoothstep(_PROGRESS)})"
    source_x = f"((X-W/2)/{scale}+W/2)"
    source_y = f"((Y-H/2)/{scale}+H/2)"
    opacity = _smoothstep(_PROGRESS)
    return _endpoint(
        f"(1-{opacity})*{_sample('a', source_x, source_y)}+{opacity}*B"
    )

_MULTI_FLIP_SCALE = f"(0.22+0.78*abs(cos(2*PI*{_PROGRESS})))"
_MULTI_FLIP_ANGLE = (
    f"(-0.85*sin(PI*clip(({_PROGRESS}-0.20)/0.80,0,1)))"
)
_MULTI_FLIP_DX = "(X-W/2)"
_MULTI_FLIP_DY = "(Y-H/2)"
_MULTI_FLIP_X = (
    f"(cos({_MULTI_FLIP_ANGLE})*{_MULTI_FLIP_DX}+"
    f"sin({_MULTI_FLIP_ANGLE})*{_MULTI_FLIP_DY}+W/2)"
)
_MULTI_FLIP_Y = (
    f"((-sin({_MULTI_FLIP_ANGLE})*{_MULTI_FLIP_DX}+"
    f"cos({_MULTI_FLIP_ANGLE})*{_MULTI_FLIP_DY})/{_MULTI_FLIP_SCALE}+H/2)"
)
_MULTI_FLIP_INSIDE = (
    f"between({_MULTI_FLIP_X},0,W-1)*between({_MULTI_FLIP_Y},0,H-1)"
)
_MULTI_FLIP_FACE = f"mod(floor(4*{_PROGRESS}),2)"
MULTI_FLIP_EXPRESSION = _endpoint(
    f"if({_MULTI_FLIP_INSIDE},if(eq({_MULTI_FLIP_FACE},0),"
    f"{_sample('a', _MULTI_FLIP_X, _MULTI_FLIP_Y)},"
    f"{_sample('b', _MULTI_FLIP_X, _MULTI_FLIP_Y)}),0)"
)


# Pinwheel ----------------------------------------------------------------

_PINWHEEL_ANGLE = f"(atan2(Y-H/2,X-W/2)+1.10*{_PROGRESS})"
_PINWHEEL_DISTANCE = f"abs(mod({_PINWHEEL_ANGLE}+PI/4,PI/2)-PI/4)"
_PINWHEEL_WIDTH = (
    f"if(lt({_PROGRESS},0.5),0.48*pow(2*{_PROGRESS},2.0),"
    f"0.48*pow(2*(1-{_PROGRESS}),2.2))"
)
_PINWHEEL_SOURCE = f"if(lt({_PROGRESS},0.55),A,B)"
PINWHEEL_EXPRESSION = _endpoint(
    f"if(gte({_PROGRESS},0.88),B,"
    f"if(lte({_PINWHEEL_DISTANCE},{_PINWHEEL_WIDTH}),0,{_PINWHEEL_SOURCE}))"
)


def _pinwheel_expression(black_background: bool) -> str:
    if not black_background:
        return PINWHEEL_EXPRESSION
    black_width = f"min(PI/4,{_PINWHEEL_WIDTH}+0.14*sin(PI*{_PROGRESS}))"
    return _endpoint(
        f"if(gte({_PROGRESS},0.88),B,"
        f"if(lte({_PINWHEEL_DISTANCE},{black_width}),0,{_PINWHEEL_SOURCE}))"
    )


# Reflection --------------------------------------------------------------

_REFLECTION_FLOOR = "(0.80*H)"
_REFLECTION_FIRST_TURN = _smoothstep(f"({_PROGRESS}/0.36)")
_REFLECTION_SECOND_TURN = _smoothstep(f"(({_PROGRESS}-0.66)/0.30)")
_REFLECTION_PROGRESS = (
    f"(0.5*{_REFLECTION_FIRST_TURN}+0.5*{_REFLECTION_SECOND_TURN})"
)
_REFLECTION_SPLIT = (
    f"clip(W*(0.01+0.98*{_REFLECTION_PROGRESS}+"
    f"0.10*sin(PI*{_REFLECTION_PROGRESS})*(Y/H-0.35)),0.005*W,0.995*W)"
)
_REFLECTION_PANEL = (
    f"if(lt(X,{_REFLECTION_SPLIT}),B,A)"
)
_REFLECTION_MIRROR = (
    f"if(lt(W-X,{_REFLECTION_SPLIT}),B,A)"
)
_REFLECTION_SEAM = f"lt(abs(X-{_REFLECTION_SPLIT}),0.012*W)"
_REFLECTION_TOP_INSIDE = "gte(Y,0.10*abs(X-W/2))"
REFLECTION_EXPRESSION = _endpoint(
    f"if(gte({_PROGRESS},0.96),B,if(lt(Y,{_REFLECTION_FLOOR}),"
    f"if({_REFLECTION_TOP_INSIDE},if({_REFLECTION_SEAM},0,{_REFLECTION_PANEL}),0),"
    f"if({_REFLECTION_SEAM},0,0.28*{_REFLECTION_MIRROR})))"
)


def _reflection_expression(direction: int) -> str:
    """Mirror panel travel for Final Cut's From Right menu value.

    Reflection deliberately uses the current A/B pixels instead of arbitrary
    coordinate resampling.  The earlier four-sampler expression took more
    than ten minutes for a three-second 540x960 review clip.  The moving,
    tilted ownership seam and dark floor still communicate the two-panel turn,
    while this bounded matte executes at ordinary cohort speed.
    """

    if direction == 0:
        return REFLECTION_EXPRESSION
    split = (
        f"clip(W*(0.99-0.98*{_REFLECTION_PROGRESS}-"
        f"0.10*sin(PI*{_REFLECTION_PROGRESS})*(Y/H-0.35)),0.005*W,0.995*W)"
    )
    panel = f"if(lt(X,{split}),A,B)"
    mirror = f"if(lt(W-X,{split}),A,B)"
    seam = f"lt(abs(X-{split}),0.012*W)"
    return _endpoint(
        f"if(gte({_PROGRESS},0.96),B,if(lt(Y,{_REFLECTION_FLOOR}),"
        f"if({_REFLECTION_TOP_INSIDE},if({seam},0,{panel}),0),"
        f"if({seam},0,0.28*{mirror})))"
    )


CUSTOM_IMPLEMENTATIONS: Mapping[str, str] = {
    "cohort_divide_default": DIVIDE_EXPRESSION,
    "cohort_spin_default": SPIN_EXPRESSION,
    "cohort_clothesline_default": CLOTHESLINE_EXPRESSION,
    "cohort_flip_default": FLIP_EXPRESSION,
    "cohort_scale_default": SCALE_EXPRESSION,
    "cohort_multi_flip_default": MULTI_FLIP_EXPRESSION,
    "cohort_pinwheel_default": PINWHEEL_EXPRESSION,
    "cohort_reflection_default": REFLECTION_EXPRESSION,
}

IMPLEMENTATION_IDS = frozenset(CUSTOM_IMPLEMENTATIONS)


DIVIDE_SECTIONS_KEY = "9999/1899850079/100/1899850080/2/100"
CLOTHESLINE_DIRECTION_KEY = "9999/1999869149/100/1999869150/2/100"
FLIP_DIRECTION_KEY = "9999/987260515/100/987260516/2/100"
PINWHEEL_BACKGROUND_KEY = "9999/1978700434/100/1978700435/2/100"
REFLECTION_DIRECTION_KEY = "9999/1999870985/100/1999870986/2/100"
SCALE_DIRECTION_KEY = "9999/1999870157/100/1999870158/2/100"
SPIN_DIRECTION_KEY = "1"


def build_panel_motion_expression(
    implementation_id: str,
    parameter_values: Mapping[str, Any],
) -> str | None:
    """Resolve parameterized panel variants after compiler validation.

    Main callers:
    - :func:`transitions.cohort.build_cohort_transition_plan`.

    Default values return the pre-existing reviewed constants byte-for-byte.
    The registry/contract has already rejected missing, duplicate, or invalid
    source values before this routine sees them.
    """

    if implementation_id == "cohort_divide_default":
        return _divide_expression(int(parameter_values.get(DIVIDE_SECTIONS_KEY, 0)))
    if implementation_id == "cohort_clothesline_default":
        direction = int(parameter_values.get(CLOTHESLINE_DIRECTION_KEY, 0))
        return CLOTHESLINE_EXPRESSION if direction == 0 else _reverse_clothesline_expression()
    if implementation_id == "cohort_spin_default":
        direction = int(parameter_values.get(SPIN_DIRECTION_KEY, 1))
        if direction == 0:
            return SPIN_EXPRESSION
        if direction == 1:
            return _spin_in_expression()
        return _outgoing_spinning_panel_expression()
    if implementation_id == "cohort_flip_default":
        return _flip_expression(int(parameter_values.get(FLIP_DIRECTION_KEY, 0)))
    if implementation_id == "cohort_pinwheel_default":
        return _pinwheel_expression(bool(parameter_values.get(PINWHEEL_BACKGROUND_KEY, False)))
    if implementation_id == "cohort_reflection_default":
        return _reflection_expression(int(parameter_values.get(REFLECTION_DIRECTION_KEY, 0)))
    if implementation_id == "cohort_scale_default":
        return _scale_expression(int(parameter_values.get(SCALE_DIRECTION_KEY, 0)))
    return CUSTOM_IMPLEMENTATIONS.get(implementation_id)
