"""The bounded ``spell-effect-v1`` single-image compute contract.

Architecture map
================

Registry parameter schema -> strict typed values -> fixed vec4 slots -> an
offline-generated GLSL 450 wrapper -> reviewed SPIR-V. At render time the
filter supplies clip-local progress, time and frame index; FCPXML can only
select the registered effect and bounded parameter values.

Why this exists
---------------
Single-input effects do not fit the two-image transition ABI. Keeping a
separate ABI prevents an apparently harmless effect addition from changing the
descriptor or push-constant layout of ``spell-transition-v1``.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..core.errors import FCPXMLCompileError
from ..core.model import Parameter


ABI_VERSION = "spell-effect-v1"
PARAMETER_LAYOUT = "effect-frame-32-plus-vec4-slot-v1"
MAX_PARAMETER_SLOTS = 4
MAX_DETERMINISTIC_SAMPLES = 32
EXPECTED_DESCRIPTOR_BINDINGS = {
    0: "input combined image sampler",
    1: "RGBA16 output storage image",
}


@dataclass(frozen=True)
class WorkingImageContract:
    color_primaries: str = "ITU-R BT.709"
    transfer: str = "linear-light"
    pixel_representation: str = "RGBA 16-bit UNORM"
    alpha: str = "straight"
    uv_origin: str = "top-left"


WORKING_IMAGE = WorkingImageContract()

# Color Adjustments is intentionally different from the ordinary spell-effect
# island.  The shared CompositionPlan hands effects an ``fcp_encoded`` image
# contract, and this artifact implements the CPU color filter sequence in that
# encoded domain.  It is promoted to RGBA16 only for the storage-image ABI;
# that promotion is not a transfer to linear light.
FCP_ENCODED_WORKING_IMAGE = WorkingImageContract(transfer="fcp_encoded")


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    name: str
    type: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None

    @property
    def components(self) -> int:
        return {"scalar": 1, "bool": 1, "vec2": 2, "vec3": 3, "vec4": 4, "color": 4}[self.type]


def parse_parameter_specs(raw: Mapping[str, Any]) -> tuple[ParameterSpec, ...]:
    """Validate and freeze registry parameter order for one effect artifact."""

    if len(raw) > MAX_PARAMETER_SLOTS:
        raise FCPXMLCompileError(f"spell-effect-v1 permits at most {MAX_PARAMETER_SLOTS} parameter slots")
    specs: list[ParameterSpec] = []
    for key, definition in raw.items():
        if not isinstance(definition, Mapping):
            raise FCPXMLCompileError(f"effect parameter {key!r} must be an object")
        parameter_type = str(definition.get("type", ""))
        if parameter_type not in {"scalar", "vec2", "vec3", "vec4", "bool", "color"}:
            raise FCPXMLCompileError(f"effect parameter {key!r} has unsupported type {parameter_type!r}")
        if "default" not in definition:
            raise FCPXMLCompileError(f"effect parameter {key!r} requires a default")
        spec = ParameterSpec(
            key=str(key),
            name=str(definition.get("name", key)),
            type=parameter_type,
            default=definition["default"],
            minimum=float(definition["minimum"]) if "minimum" in definition else None,
            maximum=float(definition["maximum"]) if "maximum" in definition else None,
        )
        _coerce_value(spec, spec.default)
        specs.append(spec)
    return tuple(specs)


def resolve_parameter_values(
    specs: Sequence[ParameterSpec], supplied: tuple[Parameter, ...]
) -> dict[str, bool | float | tuple[float, ...]]:
    """Accept only declared finite values; shader/path-like extras are errors."""

    by_key = {spec.key: spec for spec in specs}
    by_name = {spec.name.casefold(): spec for spec in specs}
    values = {spec.key: _coerce_value(spec, spec.default) for spec in specs}
    seen: set[str] = set()
    for item in supplied:
        if item.value is None:
            continue
        spec = by_key.get(item.key or "")
        if spec is None and item.name:
            spec = by_name.get(item.name.casefold())
        if spec is None:
            raise FCPXMLCompileError(
                f"arbitrary effect parameter {item.name or item.key or 'unnamed'!r} is not declared"
            )
        if spec.key in seen:
            raise FCPXMLCompileError(f"arbitrary effect parameter {spec.key!r} is supplied more than once")
        seen.add(spec.key)
        values[spec.key] = _coerce_value(spec, item.value)
    return values


def pack_parameter_blob(
    specs: Sequence[ParameterSpec], values: Mapping[str, bool | float | tuple[float, ...]]
) -> bytes:
    """Pack one little-endian 16-byte push-constant slot per parameter."""

    output = bytearray()
    for spec in specs:
        raw = values[spec.key]
        components = [1.0 if raw else 0.0] if isinstance(raw, bool) else list(raw) if isinstance(raw, tuple) else [float(raw)]
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
        raise FCPXMLCompileError(f"effect parameter {spec.key!r} expects a boolean")
    pieces = list(raw) if isinstance(raw, (list, tuple)) else str(raw).replace(",", " ").split()
    if len(pieces) != spec.components:
        raise FCPXMLCompileError(f"effect parameter {spec.key!r} expects {spec.components} numeric component(s)")
    try:
        numbers = tuple(float(value) for value in pieces)
    except (TypeError, ValueError) as exc:
        raise FCPXMLCompileError(f"effect parameter {spec.key!r} is not numeric") from exc
    if any(not math.isfinite(value) for value in numbers):
        raise FCPXMLCompileError(f"effect parameter {spec.key!r} must be finite")
    if spec.minimum is not None and any(value < spec.minimum for value in numbers):
        raise FCPXMLCompileError(f"effect parameter {spec.key!r} is below minimum {spec.minimum}")
    if spec.maximum is not None and any(value > spec.maximum for value in numbers):
        raise FCPXMLCompileError(f"effect parameter {spec.key!r} is above maximum {spec.maximum}")
    return numbers[0] if spec.type == "scalar" else numbers
