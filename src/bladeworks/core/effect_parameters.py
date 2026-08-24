"""Validate the closed parameter ABI used by portable cohort effects.

Architecture map
================

``FCPXML <param>``
    -> require the exact published key from the capability registry
    -> reject animation, duplicates, malformed component counts, and non-finite values
    -> enforce the declared scalar/vector type and inclusive bounds
    -> allow the compiler to construct a ``ResolvedEffect``

The capability registry is trusted renderer configuration. FCPXML is not. This
module therefore validates XML values without normalizing, clamping, guessing a
key from a display name, or substituting a default. Defaults are used later only
when Final Cut omitted the complete ``<param>`` node.

Why this exists
---------------

Effect graph builders need ordinary floats, while the compiler needs a useful
reason when untrusted XML is rejected. Keeping the closed-boundary checks here
lets every effect share the same fail-closed behavior without mixing validation
branches into the visual mapping code.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional, Sequence

from .model import Parameter


_NUMERIC_TYPES = frozenset(
    {"scalar", "integer", "vector", "position", "angle", "color"}
)
_KNOWN_TYPES = _NUMERIC_TYPES | {"boolean", "enum"}


def unsupported_parameter_reason(
    params: tuple[Parameter, ...],
    definitions: Mapping[str, Any],
) -> Optional[str]:
    """Return the first reason an effect's parameters violate its typed ABI.

    Main callers:
    - ``cohort_effects.unsupported_cohort_effect_reason`` before render-IR
      construction.

    Registry keys are the serialized Final Cut identifiers. Display names are
    deliberately diagnostic only: two Motion templates can publish the same
    label while using unrelated internal controls.
    """

    typed_definitions: dict[str, Mapping[str, Any]] = {}
    for key, raw in definitions.items():
        if not isinstance(raw, Mapping):
            return f"registry control {key!r} does not have a typed definition"
        typed_definitions[str(key)] = raw

    seen_keys: set[str] = set()
    for parameter in params:
        label = parameter.name or parameter.key or "unnamed control"
        if not parameter.key:
            return f"control {label!r} is missing its exact Final Cut key"
        if parameter.key in seen_keys:
            return f"control key {parameter.key!r} is duplicated"
        seen_keys.add(parameter.key)

        definition = typed_definitions.get(parameter.key)
        if definition is None:
            return f"control {label!r} is outside the bounded handler contract"
        if parameter.keyframes:
            return f"animated control {label!r} is not implemented"
        if parameter.value is None:
            return f"control {label!r} has no value"

        reason = _value_reason(parameter.value, definition, label=label)
        if reason is not None:
            return reason
    return None


def _value_reason(
    raw_value: str,
    definition: Mapping[str, Any],
    *,
    label: str,
) -> Optional[str]:
    """Validate one static serialized value without changing its meaning."""

    parameter_type = str(definition.get("type", ""))
    if parameter_type not in _KNOWN_TYPES:
        return f"control {label!r} has an invalid registry type {parameter_type!r}"

    if parameter_type == "enum":
        allowed = definition.get("allowed")
        if not isinstance(allowed, Sequence) or isinstance(allowed, (str, bytes)):
            return f"control {label!r} has no closed enum choices in the registry"
        choices = {str(value) for value in allowed}
        if raw_value not in choices:
            return f"control {label!r} has unsupported value {raw_value!r}"
        return None

    if parameter_type == "boolean":
        allowed = definition.get("allowed", ["0", "1"])
        if not isinstance(allowed, Sequence) or isinstance(allowed, (str, bytes)):
            return f"control {label!r} has invalid boolean choices in the registry"
        choices = {str(value) for value in allowed}
        if raw_value not in choices:
            return f"control {label!r} must be one of {sorted(choices)!r}"
        return None

    expected_components = _component_count(parameter_type, definition)
    if expected_components is None:
        return f"control {label!r} has an invalid component count in the registry"
    pieces = raw_value.replace(",", " ").split()
    if len(pieces) != expected_components:
        return f"control {label!r} requires {expected_components} numeric values"

    values: list[Decimal] = []
    for piece in pieces:
        try:
            value = Decimal(piece)
        except InvalidOperation:
            return f"control {label!r} is not numeric"
        if not value.is_finite():
            return f"control {label!r} must be finite"
        values.append(value)

    if parameter_type == "integer" and any(
        value != value.to_integral_value() for value in values
    ):
        return f"control {label!r} requires an integer value"

    minimum = _bounds(definition.get("minimum"), expected_components)
    maximum = _bounds(definition.get("maximum"), expected_components)
    if minimum is None or maximum is None:
        return f"control {label!r} has incomplete numeric bounds in the registry"
    for index, (value, lower, upper) in enumerate(zip(values, minimum, maximum)):
        if lower > upper:
            return f"control {label!r} has inverted registry bounds"
        if value < lower or value > upper:
            component = f" component {index + 1}" if expected_components > 1 else ""
            return (
                f"control {label!r}{component} value {str(value)!r} is outside "
                f"the inclusive range [{str(lower)}, {str(upper)}]"
            )
    return None


def _component_count(
    parameter_type: str,
    definition: Mapping[str, Any],
) -> Optional[int]:
    default = (
        2
        if parameter_type in {"vector", "position"}
        else 4
        if parameter_type == "color"
        else 1
    )
    raw = definition.get("components", default)
    if isinstance(raw, bool):
        return None
    try:
        count = int(raw)
    except (TypeError, ValueError):
        return None
    if count <= 0 or str(count) != str(raw):
        return None
    return count


def _bounds(raw: Any, components: int) -> Optional[tuple[Decimal, ...]]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) != components:
            return None
        candidates = tuple(raw)
    else:
        candidates = (raw,) * components
    output: list[Decimal] = []
    for candidate in candidates:
        if candidate is None or isinstance(candidate, bool):
            return None
        try:
            value = Decimal(str(candidate))
        except InvalidOperation:
            return None
        if not value.is_finite():
            return None
        output.append(value)
    return tuple(output)
