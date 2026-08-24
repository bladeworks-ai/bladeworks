"""The narrow ``spell-transition-v1`` shader and parameter contract.

Architecture map
================

Registry parameter schema -> strict typed values -> one push-constant ``vec4``
slot
per parameter -> immutable hexadecimal parameter blob consumed by the pinned
FFmpeg filter.

The shader side receives exactly two sampled images, one storage output image,
normalized top-left-origin UV coordinates, exact output dimensions/aspect, and
progress clamped to ``[0, 1]``. The generated wrapper enforces the endpoints.

Why this exists
---------------
GL Transitions provides a useful creative shape, but WebGL shaders are not
Vulkan compute modules. This product-owned ABI documents the smaller subset we
can compile, validate, reproduce, and safely select from FCPXML.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..core.errors import FCPXMLCompileError
from ..core.model import Parameter


ABI_VERSION = "spell-transition-v1"
PARAMETER_LAYOUT = "push-constant-vec4-slot-v1"
MAX_PARAMETER_SLOTS = 4
EXPECTED_DESCRIPTOR_BINDINGS = {
    0: "outgoing combined image sampler",
    1: "incoming combined image sampler",
    2: "RGBA16 output storage image",
}


@dataclass(frozen=True)
class WorkingImageContract:
    color_primaries: str = "ITU-R BT.709"
    transfer: str = "linear-light"
    pixel_representation: str = "RGBA 16-bit UNORM"
    alpha: str = "straight"
    uv_origin: str = "top-left"


WORKING_IMAGE = WorkingImageContract()


@dataclass(frozen=True)
class AllowedParameterContext:
    """One exact semantic-key context for a materialized no-op value."""

    when: tuple[tuple[str, float], ...]
    allowed: tuple[float, ...]


@dataclass(frozen=True)
class SemanticParameterAlias:
    """One exact published value mapped to a proved portable baseline."""

    source: float
    target: float


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    name: str
    type: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    allowed: tuple[float, ...] | None = None
    transport: str = "semantic"
    allowed_contexts: tuple[AllowedParameterContext, ...] = ()
    semantic_aliases: tuple[SemanticParameterAlias, ...] = ()

    @property
    def components(self) -> int:
        return {"scalar": 1, "bool": 1, "vec2": 2, "vec3": 3, "vec4": 4, "color": 4}[self.type]


def parse_parameter_specs(
    raw: Mapping[str, Any],
    *,
    max_slots: int | None = MAX_PARAMETER_SLOTS,
) -> tuple[ParameterSpec, ...]:
    """Validate registry declarations and freeze their serialization order.

    Main callers:
    - Capability-registry validation.
    - The compiler when it resolves FCPXML values.
    - The offline wrapper builder.
    """

    specs: list[ParameterSpec] = []
    for key, definition in raw.items():
        if not isinstance(definition, Mapping):
            raise FCPXMLCompileError(f"transition parameter {key!r} must be an object")
        parameter_type = str(definition.get("type", ""))
        if parameter_type not in {"scalar", "vec2", "vec3", "vec4", "bool", "color"}:
            raise FCPXMLCompileError(f"transition parameter {key!r} has unsupported type {parameter_type!r}")
        if "default" not in definition:
            raise FCPXMLCompileError(f"transition parameter {key!r} requires a default")
        transport = str(definition.get("transport", "semantic"))
        if transport not in {"semantic", "ignored"}:
            raise FCPXMLCompileError(
                f"transition parameter {key!r} has unsupported transport {transport!r}"
            )
        raw_contexts = definition.get("allowed_contexts", [])
        if not isinstance(raw_contexts, list):
            raise FCPXMLCompileError(
                f"transition parameter {key!r} allowed_contexts must be a list"
            )
        contexts: list[AllowedParameterContext] = []
        for index, raw_context in enumerate(raw_contexts):
            if not isinstance(raw_context, Mapping):
                raise FCPXMLCompileError(
                    f"transition parameter {key!r} context {index} must be an object"
                )
            when = raw_context.get("when")
            allowed = raw_context.get("allowed")
            if not isinstance(when, Mapping) or not when:
                raise FCPXMLCompileError(
                    f"transition parameter {key!r} context {index} requires non-empty when"
                )
            if not isinstance(allowed, list) or not allowed:
                raise FCPXMLCompileError(
                    f"transition parameter {key!r} context {index} requires allowed values"
                )
            try:
                frozen_when = tuple((str(item), float(value)) for item, value in when.items())
                frozen_allowed = tuple(float(value) for value in allowed)
            except (TypeError, ValueError) as exc:
                raise FCPXMLCompileError(
                    f"transition parameter {key!r} context {index} must be numeric"
                ) from exc
            if any(not math.isfinite(value) for _, value in frozen_when) or any(
                not math.isfinite(value) for value in frozen_allowed
            ):
                raise FCPXMLCompileError(
                    f"transition parameter {key!r} context {index} must be finite"
                )
            contexts.append(
                AllowedParameterContext(when=frozen_when, allowed=frozen_allowed)
            )
        raw_aliases = definition.get("semantic_aliases", {})
        if not isinstance(raw_aliases, Mapping):
            raise FCPXMLCompileError(
                f"transition parameter {key!r} semantic_aliases must be an object"
            )
        aliases: list[SemanticParameterAlias] = []
        try:
            for source, target in raw_aliases.items():
                aliases.append(
                    SemanticParameterAlias(source=float(source), target=float(target))
                )
        except (TypeError, ValueError) as exc:
            raise FCPXMLCompileError(
                f"transition parameter {key!r} semantic aliases must be numeric"
            ) from exc
        if any(
            not math.isfinite(item.source) or not math.isfinite(item.target)
            for item in aliases
        ):
            raise FCPXMLCompileError(
                f"transition parameter {key!r} semantic aliases must be finite"
            )
        spec = ParameterSpec(
            key=str(key),
            name=str(definition.get("name", key)),
            type=parameter_type,
            default=definition["default"],
            minimum=float(definition["minimum"]) if "minimum" in definition else None,
            maximum=float(definition["maximum"]) if "maximum" in definition else None,
            allowed=(
                tuple(float(value) for value in definition["allowed"])
                if "allowed" in definition
                else None
            ),
            transport=transport,
            allowed_contexts=tuple(contexts),
            semantic_aliases=tuple(aliases),
        )
        if spec.allowed is not None and not spec.allowed:
            raise FCPXMLCompileError(
                f"transition parameter {key!r} requires at least one allowed value"
            )
        if spec.transport == "ignored" and spec.allowed is None:
            raise FCPXMLCompileError(
                f"ignored transition parameter {key!r} requires exact allowed values"
            )
        if spec.allowed_contexts and spec.transport != "ignored":
            raise FCPXMLCompileError(
                f"transition parameter {key!r} contexts are reserved for ignored inputs"
            )
        if spec.semantic_aliases and (
            spec.transport != "semantic" or spec.type != "scalar"
        ):
            raise FCPXMLCompileError(
                f"transition parameter {key!r} semantic aliases require a semantic scalar"
            )
        alias_sources = {item.source for item in spec.semantic_aliases}
        if len(alias_sources) != len(spec.semantic_aliases):
            raise FCPXMLCompileError(
                f"transition parameter {key!r} repeats a semantic alias source"
            )
        if any(
            item.source == item.target
            or item.source not in (spec.allowed or ())
            or item.target not in (spec.allowed or ())
            or item.target in alias_sources
            for item in spec.semantic_aliases
        ):
            raise FCPXMLCompileError(
                f"transition parameter {key!r} has an invalid bounded semantic alias"
            )
        if any(
            value not in (spec.allowed or ())
            for context in spec.allowed_contexts
            for value in context.allowed
        ):
            raise FCPXMLCompileError(
                f"transition parameter {key!r} context exceeds its allowed values"
            )
        _coerce_value(spec, spec.default)
        specs.append(spec)
    by_key = {spec.key: spec for spec in specs}
    for spec in specs:
        for context in spec.allowed_contexts:
            for context_key, _ in context.when:
                referenced = by_key.get(context_key)
                if referenced is None:
                    raise FCPXMLCompileError(
                        f"transition parameter {spec.key!r} context references unknown key {context_key!r}"
                    )
                if referenced.components != 1 or referenced.transport != "semantic":
                    raise FCPXMLCompileError(
                        f"transition parameter {spec.key!r} context key {context_key!r} must be semantic scalar"
                    )
    if max_slots is not None and len(specs) > max_slots:
        raise FCPXMLCompileError(
            f"spell-transition-v1 permits at most {max_slots} parameter slots"
        )
    return tuple(specs)


def resolve_parameter_values(
    specs: Sequence[ParameterSpec],
    supplied: tuple[Parameter, ...],
) -> dict[str, bool | float | tuple[float, ...]]:
    """Resolve only declared FCPXML parameters and reject invalid values.

    FCPXML may choose bounded values. It cannot add uniforms or influence the
    binary layout. Unknown, ambiguous, malformed, non-finite, and out-of-range
    values fail compilation instead of being silently ignored or clamped.
    """

    by_key = {spec.key: spec for spec in specs}
    by_name = {spec.name.casefold(): spec for spec in specs}
    values: dict[str, bool | float | tuple[float, ...]] = {
        spec.key: _apply_semantic_alias(spec, _coerce_value(spec, spec.default))
        for spec in specs
    }
    seen: set[str] = set()
    for item in supplied:
        if item.keyframes:
            raise FCPXMLCompileError(
                f"transition parameter {item.name or item.key or 'unnamed'!r} "
                "uses keyframes, which the stock transition runtime does not support"
            )
        if item.value is None:
            continue
        spec = by_key.get(item.key or "")
        if spec is None and item.name:
            spec = by_name.get(item.name.casefold())
        if spec is None:
            raise FCPXMLCompileError(
                f"arbitrary transition parameter {item.name or item.key or 'unnamed'!r} is not declared"
            )
        if spec.key in seen:
            raise FCPXMLCompileError(f"arbitrary transition parameter {spec.key!r} is supplied more than once")
        seen.add(spec.key)
        values[spec.key] = _apply_semantic_alias(
            spec, _coerce_value(spec, item.value)
        )
    _validate_ignored_contexts(specs, values, seen)
    return values


def semantic_parameter_values(
    specs: Sequence[ParameterSpec],
    values: Mapping[str, bool | float | tuple[float, ...]],
) -> dict[str, bool | float | tuple[float, ...]]:
    """Strip exact serialization-only inputs before semantic plan building."""

    return {
        spec.key: values[spec.key]
        for spec in specs
        if spec.transport == "semantic"
    }


def supplied_ignored_parameter_values(
    specs: Sequence[ParameterSpec],
    supplied: Sequence[Parameter],
    values: Mapping[str, bool | float | tuple[float, ...]],
) -> tuple[tuple[ParameterSpec, bool | float | tuple[float, ...]], ...]:
    """Return only explicitly materialized, validated no-op inputs."""

    by_key = {spec.key: spec for spec in specs}
    by_name = {spec.name.casefold(): spec for spec in specs}
    found: list[tuple[ParameterSpec, bool | float | tuple[float, ...]]] = []
    for item in supplied:
        if item.value is None:
            continue
        spec = by_key.get(item.key or "")
        if spec is None and item.name:
            spec = by_name.get(item.name.casefold())
        if spec is not None and spec.transport == "ignored":
            found.append((spec, values[spec.key]))
    return tuple(found)


def applied_semantic_parameter_aliases(
    specs: Sequence[ParameterSpec],
    supplied: Sequence[Parameter],
) -> tuple[tuple[ParameterSpec, float, float], ...]:
    """Return effective values that use a bounded semantic alias.

    Main callers:
    - The compiler compatibility report, after normal strict resolution.

    Why this exists:
    Final Cut's adaptive ``Automatic`` direction is a published value, but its
    broader context-selection rule is not reproduced. Reporting the exact
    proved alias keeps that approximation visible instead of silently treating
    Automatic as a fully calibrated independent direction.
    """

    by_key = {spec.key: spec for spec in specs}
    by_name = {spec.name.casefold(): spec for spec in specs}
    raw_values: dict[str, bool | float | tuple[float, ...]] = {
        spec.key: _coerce_value(spec, spec.default) for spec in specs
    }
    for item in supplied:
        if item.value is None:
            continue
        spec = by_key.get(item.key or "")
        if spec is None and item.name:
            spec = by_name.get(item.name.casefold())
        if spec is None:
            continue
        raw_values[spec.key] = _coerce_value(spec, item.value)
    found: list[tuple[ParameterSpec, float, float]] = []
    for spec in specs:
        if not spec.semantic_aliases:
            continue
        raw = raw_values[spec.key]
        if isinstance(raw, (bool, tuple)):
            continue
        target = {
            alias.source: alias.target for alias in spec.semantic_aliases
        }.get(raw)
        if target is not None:
            found.append((spec, raw, target))
    return tuple(found)


def _validate_ignored_contexts(
    specs: Sequence[ParameterSpec],
    values: Mapping[str, bool | float | tuple[float, ...]],
    supplied_keys: set[str],
) -> None:
    """Reject a materialized no-op outside its exact FCP-owned context."""

    for spec in specs:
        if spec.key not in supplied_keys or not spec.allowed_contexts:
            continue
        value = values[spec.key]
        if isinstance(value, (bool, tuple)):
            raise FCPXMLCompileError(
                f"ignored transition parameter {spec.key!r} must resolve to one scalar"
            )
        matching = [
            context
            for context in spec.allowed_contexts
            if all(values[key] == expected for key, expected in context.when)
        ]
        if not matching or all(value not in context.allowed for context in matching):
            rendered = {key: values[key] for context in spec.allowed_contexts for key, _ in context.when}
            raise FCPXMLCompileError(
                f"ignored transition parameter {spec.key!r} value {value} is invalid "
                f"for semantic context {rendered}"
            )


def _apply_semantic_alias(
    spec: ParameterSpec,
    value: bool | float | tuple[float, ...],
) -> bool | float | tuple[float, ...]:
    """Map one exact published alias after normal type/bounds validation."""

    if isinstance(value, (bool, tuple)):
        return value
    return {
        alias.source: alias.target for alias in spec.semantic_aliases
    }.get(value, value)


def pack_parameter_blob(
    specs: Sequence[ParameterSpec],
    values: Mapping[str, bool | float | tuple[float, ...]],
) -> bytes:
    """Pack one little-endian 16-byte push-constant slot per parameter."""

    output = bytearray()
    for spec in specs:
        raw = values[spec.key]
        if isinstance(raw, bool):
            components = [1.0 if raw else 0.0]
        elif isinstance(raw, tuple):
            components = list(raw)
        else:
            components = [float(raw)]
        components.extend([0.0] * (4 - len(components)))
        output.extend(struct.pack("<4f", *components))
    return bytes(output)


def _coerce_value(spec: ParameterSpec, raw: Any) -> bool | float | tuple[float, ...]:
    if spec.type == "bool":
        if isinstance(raw, bool):
            return raw
        normalized = str(raw).strip().casefold()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
        raise FCPXMLCompileError(f"transition parameter {spec.key!r} expects a boolean")

    if isinstance(raw, (list, tuple)):
        pieces = list(raw)
    else:
        pieces = str(raw).replace(",", " ").split()
        if spec.allowed is not None and spec.components == 1 and len(pieces) > 1:
            # Final Cut normalizes menu values to strings such as
            # ``1 (CounterClockwise)``.  The label is localized display text;
            # only the leading numeric value is the serialization contract.
            # Accept exactly that one parenthesized suffix shape, then still
            # validate the numeric value against the registry allow-list.
            rendered = str(raw).strip()
            numeric = rendered.split(maxsplit=1)[0]
            suffix = rendered[len(numeric) :].strip()
            if not (suffix.startswith("(") and suffix.endswith(")")):
                raise FCPXMLCompileError(
                    f"transition parameter {spec.key!r} has an invalid enum label shape"
                )
            pieces = [numeric]
    if len(pieces) != spec.components:
        raise FCPXMLCompileError(
            f"transition parameter {spec.key!r} expects {spec.components} numeric component(s)"
        )
    try:
        numbers = tuple(float(value) for value in pieces)
    except (TypeError, ValueError) as exc:
        raise FCPXMLCompileError(f"transition parameter {spec.key!r} is not numeric") from exc
    if any(not math.isfinite(value) for value in numbers):
        raise FCPXMLCompileError(f"transition parameter {spec.key!r} must be finite")
    if spec.minimum is not None and any(value < spec.minimum for value in numbers):
        raise FCPXMLCompileError(
            f"transition parameter {spec.key!r} is below minimum {spec.minimum}"
        )
    if spec.maximum is not None and any(value > spec.maximum for value in numbers):
        raise FCPXMLCompileError(
            f"transition parameter {spec.key!r} is above maximum {spec.maximum}"
        )
    if spec.allowed is not None and any(value not in spec.allowed for value in numbers):
        raise FCPXMLCompileError(
            f"transition parameter {spec.key!r} is outside allowed values {spec.allowed}"
        )
    if spec.type == "scalar":
        return numbers[0]
    return numbers
