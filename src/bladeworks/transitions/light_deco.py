"""Bounded stock-FFmpeg approximations for four light/deco transitions.

Architecture map
================

registry-owned implementation ID
    -> fixed expression from ``CUSTOM_IMPLEMENTATIONS``
    -> FFmpeg ``xfade=custom`` over the recursively composed storylines
    -> literal outgoing/incoming endpoint guards.

The four expressions are deliberately self-contained and parameterless.  An
FCPXML document may select one of their registered UIDs, but it cannot supply
expression text, file paths, or unbounded sampling controls.

Why this exists
---------------
Deco, Bloom, Flash, and Lens Flare form one small calibration family: each
depends on an authored time envelope and, for Deco/Lens Flare, deterministic
screen-space geometry.  Keeping the accepted expressions together makes it
possible to tune and review them without changing unrelated cohort mappings.
"""

from __future__ import annotations

from typing import Mapping


def _endpoint(body: str) -> str:
    """Keep both boundary frames equal to their composed source frames."""

    return f"if(gte(P,1),A,if(lte(P,0),B,{body}))"


def _mix(first: str, second: str, first_weight: str = "P") -> str:
    """Return a scalar source mix suitable for one xfade color plane."""

    return f"(({first_weight})*({first})+(1-({first_weight}))*({second}))"


def _smoothstep(value: str) -> str:
    """Return a bounded cubic ease using stock FFmpeg expression syntax."""

    bounded = f"clip({value},0,1)"
    return f"(({bounded})*({bounded})*(3-2*({bounded})))"


_PROGRESS = "(1-P)"


# Bloom --------------------------------------------------------------------

_BLOOM_HANDOFF = _smoothstep(f"(({_PROGRESS}-0.38)/0.24)")
_BLOOM_BASE = _mix("A", "B", f"1-{_BLOOM_HANDOFF}")
_BLOOM_ACTIVITY = f"sin(PI*{_PROGRESS})"
BLOOM_EXPRESSION = _endpoint(
    f"min(255,{_BLOOM_BASE}+"
    # Final Cut clips bright surfaces several frames before it lifts the green
    # bowl and plants.  A broad cubic white term made the whole portable frame
    # pale too early.  Keep the highlight bloom broad, but reserve the neutral
    # white wash for the narrow midpoint plateau proved by the A/B movie.
    f"700*pow({_BLOOM_ACTIVITY},2)*pow({_BLOOM_BASE}/255,3)+"
    # The outgoing side keeps its green subject through the approach to white;
    # the incoming side carries a longer luminous decay around the plants.
    f"if(lt({_PROGRESS},0.5),255*pow({_BLOOM_ACTIVITY},10),"
    f"180*pow({_BLOOM_ACTIVITY},4)))"
)


# Flash --------------------------------------------------------------------

_FLASH_IN_WEIGHT = f"pow(clip({_PROGRESS}/0.40,0,1),1.40)"
_FLASH_OUT_WEIGHT = f"pow(clip((0.76-{_PROGRESS})/0.13,0,1),1.40)"
FLASH_EXPRESSION = _endpoint(
    f"if(lt({_PROGRESS},0.40),min(255,A*(1+2.2*{_FLASH_IN_WEIGHT})),"
    f"if(lt({_PROGRESS},0.58),255,"
    f"min(255,B*(1+2.2*{_FLASH_OUT_WEIGHT}))))"
)


# Lens Flare ---------------------------------------------------------------

_FLARE_X = f"(W*(-0.15+1.30*{_PROGRESS}))"
_FLARE_Y = f"(H*(0.72-0.44*{_PROGRESS}))"
_FLARE = (
    f"(55*exp(-hypot(X-{_FLARE_X},Y-{_FLARE_Y})/"
    f"max(1,0.42*min(W,H)))*pow(sin(PI*{_PROGRESS}),2))"
)
_FLARE_GHOST_X = f"(W*(1.10-1.20*{_PROGRESS}))"
_FLARE_GHOST_Y = f"(H*(0.30+0.38*{_PROGRESS}))"
_FLARE_GHOST_RADIUS = "(0.10*min(W,H))"
_FLARE_GHOST_DISTANCE = (
    f"abs(hypot(X-{_FLARE_GHOST_X},Y-{_FLARE_GHOST_Y})-"
    f"{_FLARE_GHOST_RADIUS})"
)
_FLARE_GHOST = (
    f"(48*exp(-{_FLARE_GHOST_DISTANCE}/max(1,0.025*min(W,H)))*"
    f"pow(sin(PI*{_PROGRESS}),2))"
)
_FLARE_DECAY = f"(1-0.52*{_smoothstep(f'(({_PROGRESS}-0.55)/0.25)')})"
_FLARE_WASH = f"(82*pow(sin(PI*{_PROGRESS}),2.3)*{_FLARE_DECAY})"
_FLARE_HANDOFF = _smoothstep(f"(({_PROGRESS}-0.28)/0.24)")
_FLARE_BASE = _mix("A", "B", f"1-{_FLARE_HANDOFF}")
LENS_FLARE_EXPRESSION = _endpoint(
    f"min(255,{_FLARE_BASE}+{_FLARE_WASH}+{_FLARE}+{_FLARE_GHOST})"
)


# Deco ---------------------------------------------------------------------

_DECO_DIAMOND = "(abs(X-W/2)/(0.58*W)+abs(Y-H/2)/(0.28*H))"
# ``xfade=custom`` receives GBR planes after the renderer's RGBA conversion.
_DECO_GOLD = "if(eq(PLANE,0),158,if(eq(PLANE,1),52,if(eq(PLANE,2),218,255)))"
_DECO_RINGS = f"lt(mod({_DECO_DIAMOND},0.115),0.020)"
_DECO_GRAPHIC = (
    f"if(lte({_DECO_DIAMOND},1.08)*{_DECO_RINGS},{_DECO_GOLD},0)"
)
_DECO_GRAPHIC_WEIGHT = f"clip({_PROGRESS}/0.12,0,1)"
_DECO_INTRO = (
    f"((1-{_DECO_GRAPHIC_WEIGHT})*A+{_DECO_GRAPHIC_WEIGHT}*({_DECO_GRAPHIC}))"
)
_DECO_REVEAL_PROGRESS = _smoothstep(f"(({_PROGRESS}-0.24)/0.32)")
_DECO_BOUNDARY = f"(0.10+3.10*{_DECO_REVEAL_PROGRESS})"
_DECO_EDGE = f"lt(abs({_DECO_DIAMOND}-{_DECO_BOUNDARY}),0.026)"
_DECO_REVEAL = (
    f"if({_DECO_EDGE},{_DECO_GOLD},if(lte({_DECO_DIAMOND},{_DECO_BOUNDARY}),B,0))"
)
_DECO_REVEAL_WEIGHT = f"clip(({_PROGRESS}-0.24)/0.10,0,1)"
DECO_EXPRESSION = _endpoint(
    f"((1-{_DECO_REVEAL_WEIGHT})*({_DECO_INTRO})+"
    f"{_DECO_REVEAL_WEIGHT}*({_DECO_REVEAL}))"
)


CUSTOM_IMPLEMENTATIONS: Mapping[str, str] = {
    "cohort_deco_default": DECO_EXPRESSION,
    "cohort_bloom_default": BLOOM_EXPRESSION,
    "cohort_flash_default": FLASH_EXPRESSION,
    "cohort_lens_flare_default": LENS_FLARE_EXPRESSION,
}

IMPLEMENTATION_IDS = frozenset(CUSTOM_IMPLEMENTATIONS)
