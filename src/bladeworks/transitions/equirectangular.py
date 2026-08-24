"""Bounded, seam-aware FFmpeg primitives for Final Cut's 360° transitions.

Architecture map
================

Final Cut Motion UID
    -> registry-owned equirectangular implementation ID
    -> strictly typed numeric parameters
    -> fixed ``xfade=custom`` expression
    -> horizontal samples wrapped with ``mod(..., W)``

Every expression keeps the latitude coordinate bounded but treats longitude as
periodic.  This is the essential difference from a flat transition: a sample
that crosses the left or right edge re-enters on the opposite side instead of
being clamped to an edge pixel.  Exact endpoint guards return the unmodified
outgoing and incoming frames.

Product and security invariants
-------------------------------

* FCPXML selects a known implementation ID and bounded numeric values only.
* FCPXML never contributes FFmpeg expressions, filter names, or file paths.
* The expressions use only stock FFmpeg's documented ``xfade=custom``
  expression environment, so they do not require the custom Vulkan runtime.
* Published Final Cut controls that are not modeled are rejected explicitly;
  no accepted parameter is silently ignored.

Why this exists
---------------

Flat wipes and pushes create a discontinuity when an equirectangular panorama
is sampled across the +/-180° longitude boundary.  Keeping the coordinate
math in one module makes the wrap rule auditable and reusable across all eight
360° operations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from ..core.errors import FCPXMLCompileError
from ..core.model import Parameter


BlurActivityProfile = Literal["symmetric", "bloom_plateau", "gaussian_skew"]
OwnershipProfile = Literal["linear", "bloom_hold", "gaussian_lead"]


@dataclass(frozen=True)
class EquirectangularTransitionPlan:
    """One registry-owned stock-FFmpeg transition expression."""

    mode: str
    expression: str
    prefilter: str | None = None
    strength: float = 0.0
    spread: float = 0.0
    activity_profile: BlurActivityProfile = "symmetric"
    ownership_profile: OwnershipProfile = "linear"
    bloom_lift: float = 0.0
    bloom_gain: float = 1.0


@dataclass(frozen=True)
class EquirectangularParameterSpec:
    """One numeric control accepted from a genuine Motion template export."""

    key: str
    name: str
    type: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    allowed: tuple[float, ...] = ()

    @property
    def components(self) -> int:
        return {"scalar": 1, "bool": 1, "vec2": 2, "vec3": 3, "vec4": 4}[self.type]


IMPLEMENTATION_IDS = frozenset(
    {
        "equirect_bloom_default",
        "equirect_circle_wipe",
        "equirect_divide",
        "equirect_gaussian_blur",
        "equirect_push",
        "equirect_reveal_wipe",
        "equirect_slide",
        "equirect_wipe",
    }
)


# Only the Rig-widget keys below have survived genuine Final Cut import/export.
# Direct Motion targets (points, longitudes, Gaussian axes, colors, and HDR)
# remain intentionally absent from the registry and are therefore rejected at
# compilation.  The expression builders continue to receive semantic names so
# the geometry stays readable; this table is the single exact-key boundary.
EQUIRECTANGULAR_PARAMETER_KEYS: Mapping[str, Mapping[str, str]] = {
    "equirect_circle_wipe": {
        "speed": "9999/10885/100/1999894671/2/100",
        "border": "9999/10885/100/1999894241/2/100",
        "border_width": "9999/10885/100/1999894240/2/100",
    },
    "equirect_divide": {
        "direction": "9999/10885/100/999552800/2/100",
        "speed": "9999/10885/100/999552899/2/100",
        "soften_edges": "9999/10885/100/10887/2/100",
        "slices": "9999/10885/100/999166301/2/100",
        "spacing": "9999/10885/100/999552867/2/100",
    },
    "equirect_gaussian_blur": {
        "blur_amount": "9999/999166152/100/999166153/2/100",
    },
    "equirect_push": {
        "direction": "9999/1999892560/100/1999892562/2/100",
        "speed": "9999/1999892560/100/1999895804/2/100",
        "soften_edges": "9999/1999892560/100/1999893684/2/100",
    },
    "equirect_reveal_wipe": {
        "speed": "9999/10885/100/1999896191/2/100",
        "soften_edges": "9999/10885/100/10887/2/100",
        "border": "9999/10885/100/1999893489/2/100",
        "border_width": "9999/10885/100/1999893490/2/100",
    },
    "equirect_slide": {
        "direction": "9999/1999892560/100/1999892562/2/100",
        "speed": "9999/1999892560/100/1999895804/2/100",
        "soften_edges": "9999/1999892560/100/1999893684/2/100",
    },
    "equirect_wipe": {
        "direction": "9999/10885/100/1999892653/2/100",
        "speed": "9999/10885/100/1999892878/2/100",
        "soften_edges": "9999/10885/100/10887/2/100",
        "border": "9999/10885/100/1999892918/2/100",
        "border_width": "9999/10885/100/1999892986/2/100",
    },
}


_SEMANTIC_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "equirect_circle_wipe": {
        "start_position": (0.5, 0.5),
        "speed": 0.0,
        "soften_edges": 0.0,
        "border": 0.0,
        "border_width": 0.49494949494949497,
    },
    "equirect_divide": {
        "start_longitude": 0.0,
        "direction": 2.0,
        "speed": 0.0,
        "soften_edges": 0.0,
        "slices": 0.25,
        "spacing": 0.75,
    },
    "equirect_gaussian_blur": {
        "blur_amount": 0.4,
        "horizontal": 100.0,
        "vertical": 100.0,
    },
    "equirect_push": {
        "start_longitude": 0.0,
        "direction": 0.0,
        "speed": 0.0,
        "soften_edges": 0.0,
    },
    "equirect_reveal_wipe": {
        "start_longitude": 0.0,
        "speed": 0.0,
        "soften_edges": 0.0,
        "border": 0.0,
        "border_width": 0.48979591836734693,
    },
    "equirect_slide": {
        "start_longitude": 0.0,
        "direction": 0.0,
        "speed": 0.0,
        "soften_edges": 0.0,
    },
    "equirect_wipe": {
        "start_longitude": 0.0,
        "direction": 0.0,
        "speed": 0.0,
        "soften_edges": 0.0,
        "border": 0.0,
        "border_width": 0.5,
    },
}


def parse_equirectangular_parameter_specs(
    raw: Mapping[str, Any],
) -> tuple[EquirectangularParameterSpec, ...]:
    """Validate the registry's numeric 360° parameter allow-list.

    Main callers:
    - ``CapabilityRegistry`` while loading the static capability registry.
    - ``compiler._compile_transition`` before resolving FCPXML parameter text.

    Why this exists:
    The Vulkan transition ABI is intentionally limited to four push-constant
    slots.  Final Cut's 360° Divide exposes six numeric controls, so reusing
    that ABI validator would impose an unrelated GPU limit on a CPU expression.
    """

    specs: list[EquirectangularParameterSpec] = []
    for key, definition in raw.items():
        if not isinstance(definition, Mapping):
            raise FCPXMLCompileError(f"360 transition parameter {key!r} must be an object")
        parameter_type = str(definition.get("type", ""))
        if parameter_type not in {"scalar", "bool", "vec2", "vec3", "vec4"}:
            raise FCPXMLCompileError(
                f"360 transition parameter {key!r} has unsupported type {parameter_type!r}"
            )
        if "default" not in definition:
            raise FCPXMLCompileError(f"360 transition parameter {key!r} requires a default")
        allowed_raw = definition.get("allowed", ())
        if not isinstance(allowed_raw, (list, tuple)):
            raise FCPXMLCompileError(f"360 transition parameter {key!r} allowed must be a list")
        try:
            allowed = tuple(float(value) for value in allowed_raw)
        except (TypeError, ValueError) as exc:
            raise FCPXMLCompileError(
                f"360 transition parameter {key!r} allowed values must be numeric"
            ) from exc
        spec = EquirectangularParameterSpec(
            key=str(key),
            name=str(definition.get("name", key)),
            type=parameter_type,
            default=definition["default"],
            minimum=float(definition["minimum"]) if "minimum" in definition else None,
            maximum=float(definition["maximum"]) if "maximum" in definition else None,
            allowed=allowed,
        )
        _coerce_value(spec, spec.default)
        specs.append(spec)
    return tuple(specs)


def resolve_equirectangular_parameter_values(
    specs: Sequence[EquirectangularParameterSpec],
    supplied: tuple[Parameter, ...],
) -> dict[str, bool | float | tuple[float, ...]]:
    """Resolve only registry-declared numeric values from FCPXML.

    Unknown names/keys, duplicate controls, malformed values, and values outside
    their reviewed range fail compilation.  This is intentionally strict: an
    unmodeled Motion control must never turn into executable filter text or a
    silently ignored UI setting.
    """

    by_key = {spec.key: spec for spec in specs}
    by_name = {spec.name.casefold(): spec for spec in specs}
    values: dict[str, bool | float | tuple[float, ...]] = {
        spec.key: _coerce_value(spec, spec.default) for spec in specs
    }
    seen: set[str] = set()
    for item in supplied:
        if item.keyframes:
            raise FCPXMLCompileError(
                f"360 transition parameter {item.name or item.key or 'unnamed'!r} "
                "uses keyframes, which the stock transition runtime does not support"
            )
        if item.value is None:
            continue
        spec = by_key.get(item.key or "")
        if spec is None and item.name:
            spec = by_name.get(item.name.casefold())
        if spec is None:
            raise FCPXMLCompileError(
                f"arbitrary 360 transition parameter {item.name or item.key or 'unnamed'!r} "
                "is not declared"
            )
        if spec.key in seen:
            raise FCPXMLCompileError(
                f"arbitrary 360 transition parameter {spec.key!r} is supplied more than once"
            )
        seen.add(spec.key)
        values[spec.key] = _coerce_value(spec, item.value)
    return values


def build_equirectangular_transition_plan(
    implementation_id: str,
    parameter_values: Mapping[str, Any],
) -> EquirectangularTransitionPlan:
    """Build one fixed wrap-aware expression from validated numeric values.

    Main callers:
    - ``ffmpeg._build_stock_transition_groups`` after compilation.

    All parameter values are inserted as finite decimal constants.  The
    expression shape and every FFmpeg function name remain source-controlled.
    """

    if implementation_id not in IMPLEMENTATION_IDS:
        raise FCPXMLCompileError(
            f"registry selected unknown 360 transition implementation {implementation_id!r}"
        )
    parameter_values = semantic_parameter_values(implementation_id, parameter_values)
    if implementation_id == "equirect_bloom_default":
        expression = _blur_expression(
            blur_amount=0.02,
            horizontal=1.0,
            vertical=1.0,
            bloom=1.85,
        )
        return EquirectangularTransitionPlan(
            mode="custom",
            expression=expression,
            prefilter="equirectangular_bloom",
            strength=0.029166666667,
            spread=0.029166666667,
            activity_profile="bloom_plateau",
            ownership_profile="bloom_hold",
            bloom_lift=0.18,
            bloom_gain=3.571428571429,
        )
    elif implementation_id == "equirect_circle_wipe":
        expression = _circle_expression(parameter_values)
    elif implementation_id == "equirect_divide":
        expression = _divide_expression(parameter_values)
    elif implementation_id == "equirect_gaussian_blur":
        blur_amount = _scalar(parameter_values, "blur_amount")
        horizontal = _scalar(parameter_values, "horizontal") / 100.0
        vertical = _scalar(parameter_values, "vertical") / 100.0
        tuned_blur_amount = 0.1375 * blur_amount
        expression = _blur_expression(
            blur_amount=tuned_blur_amount,
            horizontal=horizontal,
            vertical=vertical,
            bloom=0.0,
        )
        return EquirectangularTransitionPlan(
            mode="custom",
            expression=expression,
            prefilter="equirectangular_gaussian",
            strength=tuned_blur_amount * horizontal,
            spread=tuned_blur_amount * vertical,
            activity_profile="gaussian_skew",
            ownership_profile="gaussian_lead",
        )
    elif implementation_id == "equirect_push":
        expression = _longitude_motion_expression(parameter_values, mode="push")
    elif implementation_id == "equirect_reveal_wipe":
        expression = _longitude_motion_expression(parameter_values, mode="reveal")
    elif implementation_id == "equirect_slide":
        expression = _longitude_motion_expression(parameter_values, mode="slide")
    else:
        expression = _wipe_expression(parameter_values)
    return EquirectangularTransitionPlan(mode="custom", expression=expression)


def semantic_parameter_values(
    implementation_id: str,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate exact FCPXML keys once, then keep geometry names readable.

    Main callers:
    - ``build_equirectangular_transition_plan``.

    Semantic keys are also accepted by this private runtime boundary so the
    geometry unit tests can exercise expression builders without fabricating
    XML.  They are not capability-registry keys, so FCPXML containing them is
    still rejected by the compiler.
    """

    defaults = _SEMANTIC_DEFAULTS.get(implementation_id)
    if defaults is None:
        return dict(values)
    output = dict(defaults)
    exact_keys = EQUIRECTANGULAR_PARAMETER_KEYS.get(implementation_id, {})
    for semantic in defaults:
        exact = exact_keys.get(semantic)
        exact_present = exact is not None and exact in values
        semantic_present = semantic in values
        if exact_present and semantic_present:
            raise FCPXMLCompileError(
                f"360 transition parameter {semantic!r} was supplied by exact and semantic key"
            )
        if exact_present:
            assert exact is not None
            output[semantic] = values[exact]
        elif semantic_present:
            output[semantic] = values[semantic]
    unknown = set(values) - set(exact_keys.values()) - set(defaults)
    if unknown:
        raise FCPXMLCompileError(
            f"360 transition runtime received unknown validated keys {sorted(unknown)!r}"
        )
    return output


def _coerce_value(
    spec: EquirectangularParameterSpec,
    raw: Any,
) -> bool | float | tuple[float, ...]:
    if spec.type == "bool":
        if isinstance(raw, bool):
            return raw
        normalized = str(raw).strip().casefold()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
        raise FCPXMLCompileError(f"360 transition parameter {spec.key!r} expects a boolean")

    if isinstance(raw, (list, tuple)):
        pieces = list(raw)
    else:
        pieces = str(raw).replace(",", " ").split()
    if len(pieces) != spec.components:
        raise FCPXMLCompileError(
            f"360 transition parameter {spec.key!r} expects {spec.components} numeric component(s)"
        )
    try:
        numbers = tuple(float(value) for value in pieces)
    except (TypeError, ValueError) as exc:
        raise FCPXMLCompileError(
            f"360 transition parameter {spec.key!r} is not numeric"
        ) from exc
    if any(not math.isfinite(value) for value in numbers):
        raise FCPXMLCompileError(
            f"360 transition parameter {spec.key!r} must be finite"
        )
    if spec.minimum is not None and any(value < spec.minimum for value in numbers):
        raise FCPXMLCompileError(
            f"360 transition parameter {spec.key!r} is below minimum {spec.minimum}"
        )
    if spec.maximum is not None and any(value > spec.maximum for value in numbers):
        raise FCPXMLCompileError(
            f"360 transition parameter {spec.key!r} is above maximum {spec.maximum}"
        )
    if spec.allowed and any(value not in spec.allowed for value in numbers):
        raise FCPXMLCompileError(
            f"360 transition parameter {spec.key!r} is not one of {spec.allowed}"
        )
    if spec.type == "scalar":
        return numbers[0]
    return numbers


def _scalar(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FCPXMLCompileError(f"validated 360 transition parameter {key!r} must be scalar")
    if not math.isfinite(float(value)):
        raise FCPXMLCompileError(f"validated 360 transition parameter {key!r} must be finite")
    return float(value)


def _vector(values: Mapping[str, Any], key: str, components: int) -> tuple[float, ...]:
    value = values.get(key)
    if not isinstance(value, tuple) or len(value) != components:
        raise FCPXMLCompileError(
            f"validated 360 transition parameter {key!r} must have {components} components"
        )
    return tuple(float(component) for component in value)


def _number(value: float) -> str:
    return format(value, ".12g")


def _wrap_x(expression: str) -> str:
    """Wrap an x coordinate into ``[0, W)`` even when it is negative."""

    return f"mod(mod(({expression}),W)+W,W)"


def _clamp_y(expression: str) -> str:
    """Clamp latitude at the poles; latitude is not periodic."""

    return f"clip(({expression}),0,H-1)"


def _plane_sample(source: str, x: str, y: str) -> str:
    """Sample the current RGBA plane without collapsing the frame to luma."""

    return (
        f"if(eq(PLANE,0),{source}0({x},{y}),"
        f"if(eq(PLANE,1),{source}1({x},{y}),"
        f"if(eq(PLANE,2),{source}2({x},{y}),"
        f"{source}3({x},{y}))))"
    )


def _progress(values: Mapping[str, Any], *, divide: bool = False) -> str:
    """Return the Motion template's published, renderer-owned timing curve.

    The installed templates expose ``Speed`` as a menu, not as a continuous
    exponent.  Tags 0..5 are Constant, Ease In, Ease Out, Ease Both,
    Accelerate, and Decelerate.  Divide publishes only tags 0 and 5, where tag
    5 is labelled Ease In/Out.  Keeping these formulas source-controlled makes
    every accepted menu value deterministic and prevents arbitrary FCPXML text
    from becoming an expression.

    Main callers:
    - the circle, longitude-motion, and Divide expression builders below.
    """

    raw_speed = _scalar(values, "speed") if "speed" in values else 0.0
    if not raw_speed.is_integer():
        raise FCPXMLCompileError("360 transition Speed must be an integer Motion menu tag")
    speed = int(raw_speed)
    q = "(1-P)"
    if speed == 0:
        return q
    if divide:
        if speed != 5:
            raise FCPXMLCompileError(
                "360 Divide Speed must be Constant (0) or Ease In/Out (5)"
            )
        return f"(({q})*({q})*(3-2*({q})))"
    if speed == 1:
        return f"(({q})*({q}))"
    if speed == 2:
        return "(1-P*P)"
    if speed == 3:
        return f"(({q})*({q})*(3-2*({q})))"
    if speed == 4:
        return f"(({q})*({q})*({q}))"
    if speed == 5:
        return "(1-P*P*P)"
    raise FCPXMLCompileError("360 transition Speed must be one of Motion tags 0..5")


def _longitude_origin(values: Mapping[str, Any]) -> str:
    longitude = _scalar(values, "start_longitude") if "start_longitude" in values else 0.0
    return _number(0.5 + longitude / (2.0 * math.pi))


def _soft_mix(mask_distance: str, softness: float) -> str:
    if softness <= 0:
        # Strict comparison keeps the exact transition-start frame clean.  A
        # non-strict comparison exposes a one-pixel incoming meridian wherever
        # a zero-width mask lands exactly on a pixel center.
        return f"gt(({mask_distance}),0)"
    width = max(1.0e-6, 0.08 * softness)
    return f"clip(({mask_distance})/{_number(width)}+0.5,0,1)"


def _bordered_result(
    result: str,
    *,
    boundary_distance: str,
    enabled: float,
    width: float,
    base_width: float = 0.0025,
    width_scale: float = 0.06,
    visibility: str = "1",
    zero_is_disabled: bool = False,
) -> str:
    """Overlay the installed template's fixed red SDR border at one matte edge.

    Border Color and Graphics HDR Level remain rejected because their direct
    parameter serialization is unproved.  The Rig-owned Border and Border
    Width controls are roundtrip-exact, so enabling them selects the template's
    installed red default through this renderer-owned expression.
    """

    if enabled == 0.0:
        return result
    if enabled != 1.0:
        raise FCPXMLCompileError("360 transition Border must be 0 or 1")
    if zero_is_disabled and width == 0.0:
        return result
    border_width = base_width + width_scale * width
    # FFmpeg's custom xfade evaluates planar GBR even though the composed group
    # is converted to RGBA afterward: plane 0=G, 1=B, 2=R, 3=A. Motion's
    # installed default is exact RGB red (255, 0, 0), so only planes R and A
    # are full scale. Direct response-sheet review caught both the original
    # plane-0 green and an intermediate mistaken YUV interpretation.
    color = "if(eq(PLANE,2),255,if(eq(PLANE,3),255,0))"
    # Motion rasterizes this geometric edge with fractional pixel coverage.
    # A binary comparison made nearby published widths (including 0.489795 and
    # 0.5 at 480p) collapse to the same portable frame even though Final Cut's
    # genuine movies differ.  Keep one output pixel of outward coverage so
    # every bounded width has a deterministic sub-pixel response.
    pixel_scale = "min(W,H)"
    coverage = (
        f"({visibility})*clip((({_number(border_width)}+1/{pixel_scale})"
        f"-abs({boundary_distance}))*{pixel_scale},0,1)"
    )
    return f"(({coverage})*({color})+(1-({coverage}))*({result}))"


def _longitude_motion_expression(values: Mapping[str, Any], *, mode: str) -> str:
    """Build a cyclic wipe, slide, reveal, or two-sided push.

    ``phase`` measures distance around the panorama from the selected starting
    longitude.  A direction change reverses that cyclic distance.  Every
    moving sample uses ``_wrap_x``; no longitude sample can clamp at an edge.
    """

    progress = _progress(values)
    origin = _longitude_origin(values)
    direction = int(_scalar(values, "direction")) if "direction" in values else 0
    if direction not in {0, 1}:
        raise FCPXMLCompileError("360 transition Direction must be East (0) or West (1)")
    softness = _scalar(values, "soften_edges") if "soften_edges" in values else 0.0
    if direction == 0:
        phase = f"mod(mod((X/W-{origin}),1)+1,1)"
        travel = -1
    else:
        phase = f"mod(mod(({origin}-X/W),1)+1,1)"
        travel = 1

    if mode == "reveal":
        # Final Cut's Reveal Wipe opens equally east and west from the chosen
        # longitude.  Neither panorama rotates; two opposed rectangles in the
        # installed Motion template form one expanding cyclic band.
        shortest = f"abs(mod(mod((X/W-{origin}+0.5),1)+1,1)-0.5)"
        mix = _soft_mix(f"0.5*({progress})-({shortest})", softness)
        result = f"A*(1-({mix}))+B*({mix})"
        result = _bordered_result(
            result,
            boundary_distance=f"0.5*({progress})-({shortest})",
            enabled=_scalar(values, "border"),
            width=_scalar(values, "border_width"),
        )
        return f"if(gte(P,1),A,if(lte(P,0),B,{result}))"

    mix = _soft_mix(f"{progress}-{phase}", softness)

    # Motion's East trajectory moves the visible panorama west: output pixels
    # sample decreasing outgoing longitudes and increasing incoming longitudes.
    # West reverses both trajectories.  This sign convention is visible in the
    # genuine Final Cut Push and Slide checkpoints.
    outgoing_x = _wrap_x(f"X+({travel})*{progress}*W")
    incoming_x = _wrap_x(f"X-({travel})*(1-{progress})*W")
    outgoing = _plane_sample("a", outgoing_x, "Y")
    incoming = _plane_sample("b", incoming_x, "Y")
    if mode == "slide":
        outgoing = "A"
    result = f"({outgoing})*(1-({mix}))+({incoming})*({mix})"
    return f"if(gte(P,1),A,if(lte(P,0),B,{result}))"


def _zoomed_outgoing(origin: str, zoom: str = "3.4") -> str:
    """Sample Final Cut's enlarged outgoing 360° environment.

    Longitude wraps through the joined edge while latitude stops at the poles.
    Divide and Wipe share this installed-template projection, but own different
    reveal masks and timing profiles.

    Main callers:
    - ``_divide_expression`` and ``_wipe_expression`` below.
    """

    outgoing_x = _wrap_x(f"({origin})*W+(X-({origin})*W)/{zoom}")
    outgoing_y = _clamp_y(f"H/2+(Y-H/2)/{zoom}")
    return _plane_sample("a", outgoing_x, outgoing_y)


def _wipe_expression(values: Mapping[str, Any]) -> str:
    """Hold then collapse one directed half of the enlarged panorama.

    The genuine checkpoints do not show a flat wipe edge traversing the whole
    frame.  The installed template enters its 3.4x 360° environment, holds the
    outgoing west/east hemisphere through the middle of the overlap, and then
    collapses that directed band into the chosen start longitude.  This fixed
    piecewise width follows those visible lifecycle stages while every sample
    still obeys the periodic-longitude rule.
    """

    progress = _progress(values)
    origin = _longitude_origin(values)
    direction = int(_scalar(values, "direction"))
    if direction not in {0, 1}:
        raise FCPXMLCompileError("360 Wipe Direction must be East (0) or West (1)")
    softness = _scalar(values, "soften_edges")
    # Enter the half-sphere quickly, hold it through the reviewed middle
    # checkpoints, then leave a narrowing remnant immediately beside origin.
    width = (
        f"if(lt(({progress}),0.2),1-2.5*({progress}),"
        f"if(lt(({progress}),0.833333333333),0.5,3*(1-({progress}))))"
    )
    if direction == 0:
        directed_distance = f"mod(mod(({origin}-X/W),1)+1,1)"
    else:
        directed_distance = f"mod(mod((X/W-{origin}),1)+1,1)"
    outgoing_weight = _soft_mix(f"({width})-({directed_distance})", softness)
    outgoing = _zoomed_outgoing(origin)
    result = f"({outgoing})*({outgoing_weight})+B*(1-({outgoing_weight}))"
    result = _bordered_result(
        result,
        boundary_distance=f"({width})-({directed_distance})",
        enabled=_scalar(values, "border"),
        width=_scalar(values, "border_width"),
        # Genuine Wipe movies expose the red boundary only while the enlarged
        # hemisphere enters or leaves the viewport. Values 0.75 and 1.0 still
        # affect that near-endpoint boundary, but no border is visible at the
        # p=.25/.5/.75 checkpoints. The 23/480 scale reproduces the measured
        # 13/25/36/48-pixel runs for values .25/.5/.75/1 at 480px.
        base_width=0.0,
        width_scale=23.0 / 480.0,
        visibility=(
            f"if(lt(({progress}),0.166666666667),1,"
            f"gt(({progress}),0.833333333333))"
        ),
        zero_is_disabled=True,
    )
    # P=1 deliberately returns the enlarged outgoing environment because that
    # discontinuity is visible in Final Cut.  P=0 remains the exact incoming.
    return f"if(gte(P,1),{outgoing},if(lte(P,0),B,{result}))"


def _circle_expression(values: Mapping[str, Any]) -> str:
    """Reveal the incoming sphere using great-circle angular distance."""

    center_x, center_y = _vector(values, "start_position", 2)
    progress = _progress(values)
    softness = _scalar(values, "soften_edges")
    center_latitude = (0.5 - center_y) * math.pi
    latitude = "((0.5-Y/H)*PI)"
    # The explicit modulo chooses the shortest longitude delta at +/-180°.
    longitude_delta = (
        f"(mod(mod(((X/W-{_number(center_x)})*2*PI+PI),2*PI)+2*PI,2*PI)-PI)"
    )
    cosine_distance = (
        f"clip(sin({latitude})*{_number(math.sin(center_latitude))}+"
        f"cos({latitude})*{_number(math.cos(center_latitude))}*cos({longitude_delta}),-1,1)"
    )
    distance = f"acos({cosine_distance})"
    if softness <= 0:
        mix = f"lte({distance},{progress}*PI)"
    else:
        angular_width = max(1.0e-6, softness * 0.12 * math.pi)
        mix = f"clip(({progress}*PI-{distance})/{_number(angular_width)}+0.5,0,1)"
    result = f"A*(1-({mix}))+B*({mix})"
    result = _bordered_result(
        result,
        boundary_distance=f"({progress})-({distance})/PI",
        enabled=_scalar(values, "border"),
        width=_scalar(values, "border_width"),
    )
    return f"if(gte(P,1),A,if(lte(P,0),B,{result}))"


def _divide_expression(values: Mapping[str, Any]) -> str:
    """Reveal repeated longitude slices over a wrapped outgoing zoom.

    Final Cut expands the outgoing panorama before it exposes the alternating
    incoming slices.  Scaling longitude around the selected origin and wrapping
    that sample preserves the repeated view at +/-180 degrees; scaling latitude
    remains pole-clamped.  A deliberately front-loaded reveal matches the
    installed template's early stripe ownership.  The template intentionally
    enters its enlarged 360° environment on the first transition frame; the
    ordinary timeline still supplies exact clean frames before and after it.
    """

    timing = _progress(values, divide=True)
    # Motion's width Ramp is deliberately front-loaded.  The menu controls the
    # traversal timing; this reviewed 0.3 exponent matches the installed
    # default's early stripe ownership without accepting an arbitrary curve.
    progress = f"pow(({timing}),0.3)"
    origin = _longitude_origin(values)
    direction = int(_scalar(values, "direction"))
    if direction not in {0, 1, 2}:
        raise FCPXMLCompileError(
            "360 Divide Direction must be East (0), West (1), or East & West (2)"
        )
    softness = _scalar(values, "soften_edges")
    # Motion's default rig owns ten replicator points, but the projected 360°
    # environment exposes three longitude bands in the genuine reference.  This
    # bounded mapping follows the visible result rather than leaking the
    # template's pre-projection implementation detail into a flat sampler.
    slices = max(2, min(6, int(round(2 + 4 * _scalar(values, "slices")))))
    # Genuine full-movie hashes prove Rig values 0.75 and 1.0 are the same
    # rendered state.  Preserve that published saturation instead of letting
    # the portable approximation keep changing above the Final Cut plateau.
    spacing = min(0.75, _scalar(values, "spacing"))
    phase = f"mod(mod((X/W-{origin}),1)+1,1)"
    slice_position = f"mod(({phase})*{slices},1)"
    if direction == 0:
        band_distance = slice_position
        full_width = f"min(0.98,2*({progress})*(0.5-0.025*{_number(spacing)}))"
        distance = f"({full_width})-({band_distance})"
    elif direction == 1:
        band_distance = f"1-({slice_position})"
        full_width = f"min(0.98,2*({progress})*(0.5-0.025*{_number(spacing)}))"
        distance = f"({full_width})-({band_distance})"
    else:
        # Final Cut's projected default is visibly nonuniform: two bands sit
        # near the wrapped edges and one straddles the central longitude.  A
        # flat three-period saw puts them at 1/6, 1/2, and 5/6 instead, which
        # was the remaining tuned4 phase error.  Keep two-slice variants on the
        # generic periodic path, but encode the genuine three-band projection
        # as reviewed centers and slightly nonuniform duties.
        if slices == 3:
            finish = f"clip((({timing})-0.55)/0.383333333333,0,1)"
            finish = f"(({finish})*({finish})*(3-2*({finish})))"
            remaining = f"(1-({finish}))"
            spacing_scale = _number(1.0 - 0.25 * (spacing - 0.75))
            shifted_x = f"mod(mod((X/W-({origin})+0.5),1)+1,1)"
            outgoing_weights: list[str] = []
            # At the installed default, the genuine 960px reference projects
            # these bands to widths 110/66/111px.  Their centers already line
            # up at x=75/481/886; keep that phase and encode the non-equal duty
            # rather than returning to three equal periodic slices.
            for center, half_width in (
                (0.077, 0.0573),
                (0.5, 0.0344),
                (0.923, 0.0573),
            ):
                shortest = (
                    f"abs(mod(mod((({shifted_x})-{_number(center)}+0.5),1)+1,1)-0.5)"
                )
                width = f"{_number(half_width)}*{spacing_scale}*{remaining}"
                outgoing_weights.append(_soft_mix(f"({width})-({shortest})", softness))
            outgoing_weight = f"max({outgoing_weights[0]},max({outgoing_weights[1]},{outgoing_weights[2]}))"
            mix = f"1-({outgoing_weight})"
        else:
            rise = f"clip(({timing})/0.233333333333,0,1)"
            finish = f"clip((({timing})-0.55)/0.383333333333,0,1)"
            finish = f"(({finish})*({finish})*(3-2*({finish})))"
            spacing_scale = _number(1.0 - 0.25 * (spacing - 0.75))
            half_width = (
                f"min(0.49,(0.35*({rise})+(0.5-0.35)*({finish}))*{spacing_scale})"
            )
            distance = f"({half_width})-abs(({slice_position})-0.5)"
            mix = _soft_mix(distance, softness)
    if direction != 2:
        mix = _soft_mix(distance, softness)
    # Both Divide and Wipe enter their 360° environment at a 3.4x panorama
    # scale on the first transition frame.  It is a real boundary discontinuity
    # in the installed template, not a clean endpoint that should be hidden.
    outgoing = _zoomed_outgoing(origin)
    result = f"({outgoing})*(1-({mix}))+B*({mix})"
    return f"if(gte(P,1),{outgoing},if(lte(P,0),B,{result}))"


def _blurred_sample(source: str, horizontal_offset: str, vertical_offset: str) -> str:
    """Return a five-tap cross whose longitude taps wrap at the seam."""

    center = _plane_sample(source, _wrap_x("X"), _clamp_y("Y"))
    left = _plane_sample(source, _wrap_x(f"X-({horizontal_offset})"), _clamp_y("Y"))
    right = _plane_sample(source, _wrap_x(f"X+({horizontal_offset})"), _clamp_y("Y"))
    above = _plane_sample(source, _wrap_x("X"), _clamp_y(f"Y-({vertical_offset})"))
    below = _plane_sample(source, _wrap_x("X"), _clamp_y(f"Y+({vertical_offset})"))
    return f"0.4*({center})+0.15*(({left})+({right})+({above})+({below}))"


def _blur_expression(
    *,
    blur_amount: float,
    horizontal: float,
    vertical: float,
    bloom: float,
) -> str:
    """Crossfade two wrap-aware five-tap blurs, optionally brightening RGB."""

    midpoint = "sin(PI*(1-P))"
    horizontal_offset = f"{_number(blur_amount * horizontal)}*min(W,H)*{midpoint}"
    vertical_offset = f"{_number(blur_amount * vertical)}*min(W,H)*{midpoint}"
    outgoing = _blurred_sample("a", horizontal_offset, vertical_offset)
    incoming = _blurred_sample("b", horizontal_offset, vertical_offset)
    mixed = f"P*({outgoing})+(1-P)*({incoming})"
    if bloom > 0:
        brightened = f"min(255,({mixed})*(1+{_number(bloom)}*{midpoint}))"
        mixed = f"if(eq(PLANE,3),{mixed},{brightened})"
    return f"if(gte(P,1),A,if(lte(P,0),B,{mixed}))"


def equirectangular_blur_activity(profile: str, duration: str) -> str:
    """Return one fixed, endpoint-guarded Motion-derived blur envelope.

    ``T`` is local transition time.  Bloom uses the installed template's
    46.7%..60% peak plateau.  Gaussian reaches its maximum slightly before the
    midpoint and decays more quickly than the old broad sine envelope.

    Main callers:
    - ``legacy_ffmpeg.ffmpeg._equirectangular_blur_side`` for both sides of one transition.
    - ``tensor.tr_equirect``, which evaluates this very string on the GPU.

    Why this lives here rather than in the emitter: it is a registry-owned calibrated
    envelope that BOTH backends must agree on, exactly like the transition expressions
    beside it.  It sat in ``ffmpeg.py`` only for historical reasons, which forced the
    tensor port to import an emitter private.
    """

    q = f"(T/{duration})"
    if profile == "symmetric":
        interior = f"pow(max(0,sin(PI*({q}))),0.65)"
    elif profile == "bloom_plateau":
        rise = f"clip(({q})/0.466666666667,0,1)"
        rise = f"(({rise})*({rise})*(3-2*({rise})))"
        fall = f"clip((1-({q}))/0.4,0,1)"
        fall = f"(({fall})*({fall})*(3-2*({fall})))"
        interior = f"if(lt(({q}),0.466666666667),{rise},if(lte(({q}),0.6),1,{fall}))"
    elif profile == "gaussian_skew":
        # Final Cut is already strongly blurred at q=0.23 and still visibly
        # blurred at q=0.73.  The lower exponents broaden both shoulders while
        # retaining the slightly-early 47% peak and exact endpoint guards.
        rise = f"pow(max(0,sin(PI*0.5*({q})/0.47)),0.65)"
        fall = f"pow(max(0,sin(PI*0.5*(1-({q}))/0.53)),2)"
        interior = f"if(lt(({q}),0.47),{rise},{fall})"
    else:
        raise FCPXMLCompileError(
            f"unknown equirectangular blur activity profile {profile!r}"
        )
    return f"if(lte(T,0),0,if(gte(T,{duration}),0,{interior}))"


_RADIAL_ZOOM_BLEND_EXPRESSION = (
    "if(gte(P,1),A,if(lte(P,0),B,"
    "(A*P*P*P+B*(1-P)*(1-P)*(1-P))/"
    "(P*P*P+(1-P)*(1-P)*(1-P))))"
)
