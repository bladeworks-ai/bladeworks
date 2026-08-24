"""Validate the bounded portable subset of Final Cut masked filters.

Architecture map
================

``filter-video-mask`` source nodes -> structural/security validation ->
``ResolvedMaskGroup`` -> FFmpeg mask planes built by :mod:`ffmpeg`.

The FCPXML document may select geometry and bounded numeric parameters. It may
not provide an FFmpeg expression, a file path, executable data, or a decoder
choice. Apple stores color-isolation samples and Draw Mask splines in opaque
payloads; those payloads are preserved by the parser but rejected here until a
reviewed decoder exists.

Coordinate contract
-------------------

Shape position, radius, and feather use Final Cut image-plane pixels at the
project's output size. The origin is frame center, +x is right, and +y is up.
This was checked with a headless Final Cut render; treating the values as a
1080-normalized coordinate space made a 320-pixel fixture visibly too small.
Alpha is straight, 0 outside and 1 inside. Feather is a symmetric linear edge
ramp; falloff raises that ramp to a bounded exponent. Mask opacity multiplies
the resulting alpha.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable, Optional

from .model import MaskSource, Parameter, ResolvedMask, ResolvedMaskGroup


MAX_MASKS = 32
MAX_DRAW_POINTS = 64
MASK_BLEND_MODES = ("add", "subtract", "multiply")

# One source of truth for the static and animated Shape Mask execution bounds.
_SHAPE_PARAMETER_BOUNDS = (
    (("160", "radius"), 2, 0.0, 32768.0),
    (("159", "curvature"), 1, 0.0, 1.0),
    (("102", "feather"), 1, 0.0, 8192.0),
    (("201", "position"), 2, -32768.0, 32768.0),
    (("202", "rotation"), 1, -3600.0, 3600.0),
    (("opacity", "103"), 1, 0.0, 1.0),
    (("falloff", "104"), 1, 0.1, 8.0),
)

# Machine-readable authoring surface shared with Bladeworks Studio.  The
# resolver below remains the execution gate; this table publishes the same
# exact keys, coordinate units, bounds, and animation policy without requiring
# the browser to duplicate Python constants.
MASK_AUTHORING_SOURCES: tuple[dict[str, object], ...] = (
    {
        "id": "shape",
        "fcpxmlKind": "mask-shape",
        "support": "approximate",
        "parameters": (
            {"key": "160", "name": "Radius", "type": "point", "components": ("x", "y"), "minimum": 0.0, "maximum": 32768.0, "units": "image_plane_pixels", "animatable": True},
            {"key": "159", "name": "Curvature", "type": "number", "minimum": 0.0, "maximum": 1.0, "units": "normalized", "default": 1.0, "animatable": True},
            {"key": "102", "name": "Feather", "type": "number", "minimum": 0.0, "maximum": 8192.0, "units": "image_plane_pixels", "default": 0.0, "animatable": True},
            {"key": "201", "name": "Position", "type": "point", "components": ("x", "y"), "minimum": -32768.0, "maximum": 32768.0, "units": "image_plane_pixels", "animatable": True},
            {"key": "202", "name": "Rotation", "type": "number", "minimum": -3600.0, "maximum": 3600.0, "units": "degrees", "default": 0.0, "animatable": True},
            {"key": "103", "name": "Opacity", "type": "number", "minimum": 0.0, "maximum": 1.0, "units": "normalized", "default": 1.0, "animatable": True},
            {"key": "104", "name": "Falloff", "type": "number", "minimum": 0.1, "maximum": 8.0, "units": "exponent", "default": 1.0, "animatable": True},
        ),
        "notes": "Curvature and feather use a portable superellipse and linear-ramp approximation.",
    },
    {
        "id": "draw",
        "fcpxmlKind": "mask-shape",
        "support": "approximate",
        "parameters": (
            {"key": "points", "name": "Points", "type": "point_list", "components": ("x", "y"), "minimum": -32768.0, "maximum": 32768.0, "units": "image_plane_pixels", "minimumItems": 3, "maximumItems": MAX_DRAW_POINTS, "convex": True, "animatable": False},
            {"key": "opacity", "name": "Opacity", "type": "number", "minimum": 0.0, "maximum": 1.0, "units": "normalized", "default": 1.0, "animatable": True},
        ),
        "notes": "Polygon edges are linear; Final Cut Bezier handles are not reproduced.",
    },
    {
        "id": "color",
        "fcpxmlKind": "mask-isolation",
        "dataAbi": "spell-mask-isolation-v1",
        "support": "approximate",
        "parameters": (
            {"key": "color", "name": "Color", "type": "color", "components": ("red", "green", "blue"), "minimum": 0.0, "maximum": 1.0, "units": "normalized", "animatable": False},
            {"key": "tolerance", "name": "Tolerance", "type": "number", "minimum": 0.00001, "maximum": 1.0, "units": "normalized", "default": 0.12, "animatable": False},
            {"key": "softness", "name": "Softness", "type": "number", "minimum": 0.0, "maximum": 1.0, "units": "normalized", "default": 0.05, "animatable": False},
            {"key": "opacity", "name": "Opacity", "type": "number", "minimum": 0.0, "maximum": 1.0, "units": "normalized", "default": 1.0, "animatable": False},
        ),
        "notes": "Color isolation uses a portable RGB-distance approximation.",
    },
    {
        "id": "luma",
        "fcpxmlKind": "mask-isolation",
        "dataAbi": "spell-mask-isolation-v1",
        "support": "approximate",
        "parameters": (
            {"key": "luma_min", "name": "Luma Minimum", "type": "number", "minimum": 0.0, "maximum": 1.0, "units": "normalized", "default": 0.0, "animatable": False},
            {"key": "luma_max", "name": "Luma Maximum", "type": "number", "minimum": 0.0, "maximum": 1.0, "units": "normalized", "default": 1.0, "animatable": False},
            {"key": "softness", "name": "Softness", "type": "number", "minimum": 0.0, "maximum": 1.0, "units": "normalized", "default": 0.05, "animatable": False},
            {"key": "opacity", "name": "Opacity", "type": "number", "minimum": 0.0, "maximum": 1.0, "units": "normalized", "default": 1.0, "animatable": False},
        ),
        "notes": "Luma isolation uses a portable bounded luma ramp.",
    },
)


class MaskResolutionError(ValueError):
    """Raised when a source mask cannot enter the portable render IR."""


@dataclass(frozen=True)
class MaskResolution:
    group: ResolvedMaskGroup
    approximations: tuple[str, ...]


def resolve_mask_group(masks: tuple[MaskSource, ...], *, inverted: bool) -> MaskResolution:
    """Resolve enabled, non-tracked masks without interpreting opaque blobs.

    Main callers:
    - ``compiler._Compiler._resolve_masked_effect``.

    Why this exists:
    Parser preservation and render authorization are separate stages. A valid
    FCPXML mask can still contain Apple-private data that portable rendering
    must not guess at.
    """

    enabled = tuple(mask for mask in masks if mask.enabled)
    if not enabled:
        raise MaskResolutionError("masked filter has no enabled masks")
    if len(enabled) > MAX_MASKS:
        raise MaskResolutionError(f"masked filter exceeds the {MAX_MASKS}-mask portable limit")

    output: list[ResolvedMask] = []
    approximations: list[str] = []
    for source in enabled:
        if source.blend_mode not in MASK_BLEND_MODES:
            raise MaskResolutionError(f"mask blend mode {source.blend_mode!r} is not supported")
        normalized_name = (source.name or source.kind).casefold()
        if "auto mask" in normalized_name or "magnetic mask" in normalized_name:
            raise MaskResolutionError(f"{source.name or source.kind} is intentionally outside the portable mask scope")
        if source.tracking:
            raise MaskResolutionError(
                f"{source.name or source.kind} references tracking data {source.tracking!r}; tracked masks are unsupported"
            )
        if source.kind == "mask-shape":
            resolved, notes = _resolve_shape(source)
        elif source.kind == "mask-isolation":
            resolved, notes = _resolve_isolation(source)
        else:
            raise MaskResolutionError(f"unknown mask kind {source.kind!r}")
        output.append(resolved)
        approximations.extend(notes)
    return MaskResolution(
        group=ResolvedMaskGroup(masks=tuple(output), inverted=inverted),
        approximations=tuple(approximations),
    )


def _resolve_shape(source: MaskSource) -> tuple[ResolvedMask, tuple[str, ...]]:
    values = _parameter_index(source.params)
    points = _parameter(values, keys=("points", "vertices", "path", "300"))
    if points is not None:
        parsed = _draw_points(points.value)
        data = {"points": ";".join(f"{x:g},{y:g}" for x, y in parsed)}
        _bounded_vector(values, ("opacity", "103"), components=1, minimum=0.0, maximum=1.0, optional=True)
        _validate_animated_parameters(source.params, allowed_components={"opacity": 1, "103": 1})
        _validate_animated_bounds(
            values,
            keys=("opacity", "103"),
            components=1,
            minimum=0.0,
            maximum=1.0,
        )
        return (
            ResolvedMask(
                kind="draw",
                name=source.name or "Draw Mask",
                blend_mode=source.blend_mode,
                params=source.params,
                data=data,
            ),
            ("Draw Mask polygon edges use linear segments; Final Cut Bezier handles are not reproduced",),
        )

    for keys, components, minimum, maximum in _SHAPE_PARAMETER_BOUNDS:
        _bounded_vector(
            values,
            keys,
            components=components,
            minimum=minimum,
            maximum=maximum,
            optional=True,
        )
    _validate_animated_parameters(
        source.params,
        allowed_components={
            "160": 2,
            "radius": 2,
            "159": 1,
            "curvature": 1,
            "102": 1,
            "feather": 1,
            "201": 2,
            "position": 2,
            "202": 1,
            "rotation": 1,
            "opacity": 1,
            "103": 1,
            "falloff": 1,
            "104": 1,
        },
    )
    for keys, components, minimum, maximum in _SHAPE_PARAMETER_BOUNDS:
        _validate_animated_bounds(
            values,
            keys=keys,
            components=components,
            minimum=minimum,
            maximum=maximum,
        )
    return (
        ResolvedMask(
            kind="shape",
            name=source.name or "Shape Mask",
            blend_mode=source.blend_mode,
            params=source.params,
        ),
        ("shape curvature and feather use a portable superellipse/linear-ramp approximation",),
    )


def _resolve_isolation(source: MaskSource) -> tuple[ResolvedMask, tuple[str, ...]]:
    """Accept an explicit reviewed JSON/param contract, never Apple's archive."""

    explicit: dict[str, object] = {}
    if source.data:
        try:
            decoded = json.loads(source.data)
        except json.JSONDecodeError as exc:
            raise MaskResolutionError(
                f"{source.name or 'Color Mask'} uses opaque Final Cut isolation data; no reviewed decoder is available"
            ) from exc
        if not isinstance(decoded, dict) or decoded.get("abi") != "spell-mask-isolation-v1":
            raise MaskResolutionError("color/range mask data must declare abi='spell-mask-isolation-v1'")
        allowed = {"abi", "color", "tolerance", "softness", "luma_min", "luma_max", "opacity"}
        unknown = set(decoded) - allowed
        if unknown:
            raise MaskResolutionError(f"color/range mask data contains unsupported keys: {', '.join(sorted(unknown))}")
        explicit = decoded

    values = _parameter_index(source.params)
    color = explicit.get("color")
    color_parameter = _parameter(values, keys=("color", "sample color", "bladeworks/color"))
    if color is None and color_parameter is not None:
        color = _numbers(color_parameter.value, 3, "Color")
    luma_min = explicit.get("luma_min", _optional_scalar(values, ("luma min", "luminance min")))
    luma_max = explicit.get("luma_max", _optional_scalar(values, ("luma max", "luminance max")))
    if color is None and luma_min is None and luma_max is None:
        raise MaskResolutionError("color/range mask exposes neither a bounded color sample nor a luma range")

    data: dict[str, str] = {}
    if color is not None:
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            raise MaskResolutionError("color sample must have exactly three components")
        components = tuple(_finite_float(value, "color component") for value in color)
        if any(value < 0 or value > 1 for value in components):
            raise MaskResolutionError("color components must be in [0, 1]")
        data["color"] = " ".join(f"{value:.17g}" for value in components)
    tolerance = _explicit_scalar(explicit, values, "tolerance", ("tolerance",), 0.12, 0.00001, 1.0)
    softness = _explicit_scalar(explicit, values, "softness", ("softness", "feather"), 0.05, 0.0, 1.0)
    opacity = _explicit_scalar(explicit, values, "opacity", ("opacity",), 1.0, 0.0, 1.0)
    low = 0.0 if luma_min is None else _finite_float(luma_min, "luma minimum")
    high = 1.0 if luma_max is None else _finite_float(luma_max, "luma maximum")
    if not 0 <= low <= high <= 1:
        raise MaskResolutionError("luma range must satisfy 0 <= minimum <= maximum <= 1")
    data.update(
        tolerance=f"{tolerance:.17g}",
        softness=f"{softness:.17g}",
        opacity=f"{opacity:.17g}",
        luma_min=f"{low:.17g}",
        luma_max=f"{high:.17g}",
    )
    kind = "color" if "color" in data else "range"
    return (
        ResolvedMask(
            kind=kind,
            name=source.name or ("Color Mask" if kind == "color" else "Range Mask"),
            blend_mode=source.blend_mode,
            params=source.params,
            data=data,
        ),
        ("color/range isolation uses portable RGB-distance and luma ramps",),
    )


def _parameter_index(params: tuple[Parameter, ...]) -> dict[str, Parameter]:
    output: dict[str, Parameter] = {}
    for parameter in params:
        if parameter.key:
            output[parameter.key.casefold()] = parameter
        if parameter.name:
            output[parameter.name.casefold()] = parameter
    return output


def _parameter(values: dict[str, Parameter], *, keys: Iterable[str]) -> Optional[Parameter]:
    return next((values[key.casefold()] for key in keys if key.casefold() in values), None)


def _bounded_vector(
    values: dict[str, Parameter],
    keys: tuple[str, ...],
    *,
    components: int,
    minimum: float,
    maximum: float,
    optional: bool,
) -> Optional[tuple[float, ...]]:
    parameter = _parameter(values, keys=keys)
    if parameter is None or parameter.value is None:
        if optional:
            return None
        raise MaskResolutionError(f"missing mask parameter {keys[0]!r}")
    numbers = _numbers(parameter.value, components, parameter.name or parameter.key or keys[0])
    if any(number < minimum or number > maximum for number in numbers):
        raise MaskResolutionError(
            f"mask parameter {parameter.name or parameter.key!r} is outside [{minimum}, {maximum}]"
        )
    return numbers


def _numbers(raw: Optional[str], components: int, label: str) -> tuple[float, ...]:
    if raw is None:
        raise MaskResolutionError(f"mask parameter {label!r} has no value")
    pieces = raw.replace(",", " ").split()
    if len(pieces) != components:
        raise MaskResolutionError(f"mask parameter {label!r} requires {components} numeric components")
    return tuple(_finite_float(piece, label) for piece in pieces)


def _finite_float(raw: object, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise MaskResolutionError(f"{label} must be numeric") from exc
    if not math.isfinite(value):
        raise MaskResolutionError(f"{label} must be finite")
    return value


def _draw_points(raw: Optional[str]) -> tuple[tuple[float, float], ...]:
    if not raw:
        raise MaskResolutionError("Draw Mask has no points")
    points: list[tuple[float, float]] = []
    for token in raw.replace("|", ";").split(";"):
        if not token.strip():
            continue
        pieces = token.replace(",", " ").split()
        if len(pieces) != 2:
            raise MaskResolutionError("Draw Mask points must use 'x,y;x,y' syntax")
        points.append((_finite_float(pieces[0], "Draw Mask x"), _finite_float(pieces[1], "Draw Mask y")))
    if len(points) < 3 or len(points) > MAX_DRAW_POINTS:
        raise MaskResolutionError(f"Draw Mask requires 3..{MAX_DRAW_POINTS} points")
    if any(abs(value) > 32768 for point in points for value in point):
        raise MaskResolutionError("Draw Mask point is outside the portable image-plane range")
    cross_products: list[float] = []
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        after = points[(index + 2) % len(points)]
        cross = (following[0] - current[0]) * (after[1] - following[1]) - (
            following[1] - current[1]
        ) * (after[0] - following[0])
        if abs(cross) > 1e-9:
            cross_products.append(cross)
    if not cross_products or any(value * cross_products[0] < 0 for value in cross_products[1:]):
        raise MaskResolutionError("Draw Mask points must form a non-degenerate convex polygon")
    return tuple(points)


def _validate_animated_parameters(params: tuple[Parameter, ...], *, allowed_components: dict[str, int]) -> None:
    for parameter in params:
        if not parameter.keyframes:
            continue
        identity = (parameter.key or parameter.name or "").casefold()
        components = allowed_components.get(identity)
        if components is None:
            raise MaskResolutionError(f"animated mask parameter {parameter.name or parameter.key!r} is unsupported")
        previous = None
        for frame in parameter.keyframes:
            if previous is not None and frame.time <= previous:
                raise MaskResolutionError(f"mask keyframes for {parameter.name or parameter.key!r} are not increasing")
            _numbers(frame.value, components, parameter.name or parameter.key or "mask keyframe")
            previous = frame.time


def _validate_animated_bounds(
    values: dict[str, Parameter],
    *,
    keys: tuple[str, ...],
    components: int,
    minimum: float,
    maximum: float,
) -> None:
    """Validate every authored sample of one bounded animated parameter.

    Main callers: ``_resolve_shape`` for bounded Draw and Shape Mask parameters.

    Why this exists: ``_bounded_vector`` checks the static value, while an
    animation can introduce different values at each source-clock keyframe.
    The renderer must never receive an out-of-range matte multiplier through
    that second path.
    """

    parameter = _parameter(values, keys=keys)
    if parameter is None:
        return
    for frame in parameter.keyframes:
        numbers = _numbers(frame.value, components, parameter.name or parameter.key or keys[0])
        if any(number < minimum or number > maximum for number in numbers):
            raise MaskResolutionError(
                f"mask keyframe for {parameter.name or parameter.key!r} is outside [{minimum}, {maximum}]"
            )


def _optional_scalar(values: dict[str, Parameter], keys: tuple[str, ...]) -> Optional[float]:
    parameter = _parameter(values, keys=keys)
    if parameter is None:
        return None
    return _numbers(parameter.value, 1, parameter.name or parameter.key or keys[0])[0]


def _explicit_scalar(
    explicit: dict[str, object],
    values: dict[str, Parameter],
    field: str,
    aliases: tuple[str, ...],
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = explicit.get(field)
    if raw is None:
        raw = _optional_scalar(values, aliases)
    value = default if raw is None else _finite_float(raw, field)
    if value < minimum or value > maximum:
        raise MaskResolutionError(f"{field} is outside [{minimum}, {maximum}]")
    return value
