"""Bounded decoder and portable contract for Final Cut's Green Screen Keyer.

Architecture map
================

``filter-video/data`` -> bounded base64 decoding -> safe ozml parsing ->
``GreenScreenKeyerSettings`` -> FFmpeg ``colorkey``/``despill`` filters.

The source model keeps the original opaque blobs for diagnostics and round
tripping.  This module exposes only the small, typed subset the portable
renderer understands.  It never executes code from ``effectData`` and never
uses values from unrecognized XML nodes as fallbacks.

Important invariants:
- Input/output pixels are SDR Rec.709 RGBA with straight (unpremultiplied)
  alpha.  The surrounding FFmpeg graph performs that normalization.
- The decoded payload is bounded to 4 MiB and may not contain DTDs/entities.
- A malformed or unsupported payload is rejected; the caller omits the keyer
  and records a compatibility finding rather than guessing.
"""

from __future__ import annotations

import base64
import binascii
import plistlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Mapping

from .model import Parameter


MAX_EFFECT_CONFIG_BYTES = 64 * 1024
MAX_EFFECT_DATA_BYTES = 4 * 1024 * 1024


class GreenScreenKeyerDataError(ValueError):
    """Raised when an FCP keyer payload cannot enter the portable contract."""


@dataclass(frozen=True)
class GreenScreenKeyerSettings:
    """Validated settings consumed by the portable FFmpeg keyer handler.

    Main callers:
    - ``compiler._Compiler._resolve_effects`` validates and freezes settings.
    - ``ffmpeg._effect_filters`` translates the frozen values to AVOptions.

    Why this exists:
    Final Cut stores these controls inside opaque ``effectData`` rather than as
    ordinary top-level FCPXML parameters.  Freezing a typed contract here keeps
    the graph builder independent of Apple's serialized XML shape.
    """

    key_color: tuple[float, float, float]
    softness: float
    strength: float
    spill_level: float
    chroma_rolloff: float
    luma_rolloff: float
    green_chroma: float
    blue_chroma: float
    min_green: float
    max_green: float
    min_blue: float
    max_blue: float
    mix: float

    def as_data(self) -> dict[str, str]:
        """Serialize deterministic internal values for ``ResolvedEffect``."""

        values = {
            "key_color": " ".join(_number(component) for component in self.key_color),
            "softness": _number(self.softness),
            "strength": _number(self.strength),
            "spill_level": _number(self.spill_level),
            "chroma_rolloff": _number(self.chroma_rolloff),
            "luma_rolloff": _number(self.luma_rolloff),
            "green_chroma": _number(self.green_chroma),
            "blue_chroma": _number(self.blue_chroma),
            "min_green": _number(self.min_green),
            "max_green": _number(self.max_green),
            "min_blue": _number(self.min_blue),
            "max_blue": _number(self.max_blue),
            "mix": _number(self.mix),
        }
        return values


@dataclass(frozen=True)
class GreenScreenKeyerResolution:
    settings: GreenScreenKeyerSettings
    ignored_control_names: tuple[str, ...]
    used_default_key_color: bool


_SCALARS: dict[str, tuple[str, float, float, float]] = {
    "defaultsoftness": ("softness", 9.0, 0.0, 20.0),
    "strength": ("strength", 1.0, 0.0, 2.0),
    "spilllevel": ("spill_level", 0.46, 0.0, 1.0),
    "chromarolloff": ("chroma_rolloff", 0.1, 0.0, 1.0),
    "lumarolloff": ("luma_rolloff", 0.1, 0.0, 1.0),
    "greenchroma": ("green_chroma", 0.09, -10.0, 10.0),
    "bluechroma": ("blue_chroma", 0.09, -10.0, 10.0),
    "mingreen": ("min_green", -3.0, -10.0, 10.0),
    "maxgreen": ("max_green", -1.7, -10.0, 10.0),
    "minblue": ("min_blue", -1.25, -10.0, 10.0),
    "maxblue": ("max_blue", 0.125, -10.0, 10.0),
    "mix": ("mix", 1.0, 0.0, 1.0),
}

_COLOR_NAMES = ("keycolor", "samplecolor", "screencolor")
_KNOWN_OPAQUE_GROUPS = {
    "colorselection",
    "keyerautokey",
    "mattetools",
    "spillsuppression",
    "lightwrap",
}


def resolve_green_screen_keyer(
    data: Mapping[str, str],
    top_level_params: tuple[Parameter, ...],
) -> GreenScreenKeyerResolution:
    """Decode and validate the bounded portable subset of ``effectData``.

    ``effectConfig`` must be a small base64 NSKeyedArchiver plist and
    ``effectData`` must decode directly to an ``ozml`` XML document.  Top-level
    params may override supported values when present, but cannot replace a
    missing/invalid Final Cut payload.

    Main callers:
    - ``compiler._Compiler._resolve_effects`` for the documented Keyer UID.
    """

    config_bytes = _decode_bounded_base64(
        data.get("effectConfig"),
        label="effectConfig",
        maximum=MAX_EFFECT_CONFIG_BYTES,
    )
    if not config_bytes.startswith(b"bplist00"):
        raise GreenScreenKeyerDataError("effectConfig is not an NSKeyedArchiver binary plist")
    try:
        config = plistlib.loads(config_bytes)
    except plistlib.InvalidFileException as exc:
        raise GreenScreenKeyerDataError("effectConfig is not a valid binary plist") from exc
    if not isinstance(config, dict) or "$archiver" not in config:
        raise GreenScreenKeyerDataError("effectConfig is not an NSKeyedArchiver dictionary")

    xml_bytes = _decode_bounded_base64(
        data.get("effectData"),
        label="effectData",
        maximum=MAX_EFFECT_DATA_BYTES,
    )
    if re.search(br"<!\s*(?:DOCTYPE|ENTITY)\b", xml_bytes, flags=re.IGNORECASE):
        raise GreenScreenKeyerDataError("effectData XML declarations are not allowed")
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise GreenScreenKeyerDataError(f"effectData is not direct ozml XML: {exc}") from exc
    if _tag(root) != "ozml":
        raise GreenScreenKeyerDataError("effectData root must be <ozml>")

    raw_values: dict[str, str] = {}
    ignored_names: set[str] = set()
    for element in root.iter():
        if _tag(element) != "parameter":
            continue
        name = element.get("name")
        if not name:
            continue
        normalized = _normalize(name)
        if normalized in _SCALARS or normalized in _COLOR_NAMES:
            if "value" not in element.attrib:
                raise GreenScreenKeyerDataError(f"supported keyer control {name!r} has no value")
            raw_values[normalized] = element.attrib["value"]
        elif normalized in _KNOWN_OPAQUE_GROUPS:
            ignored_names.add(name)

    # Real FCP payloads include Strength.  Requiring one recognized scalar is a
    # deliberate guard against accepting an unrelated ozml document.
    if not any(key in raw_values for key in _SCALARS):
        raise GreenScreenKeyerDataError("effectData contains no supported Green Screen Keyer controls")

    for param in top_level_params:
        if param.value is None:
            continue
        normalized = _normalize(param.key or "")
        if normalized.startswith("bladeworks"):
            normalized = normalized.removeprefix("bladeworks")
        if normalized not in _SCALARS and normalized not in _COLOR_NAMES:
            normalized = _normalize(param.name or "")
        if normalized in _SCALARS or normalized in _COLOR_NAMES:
            raw_values[normalized] = param.value

    scalar_values: dict[str, float] = {}
    for normalized, (field, default, minimum, maximum) in _SCALARS.items():
        scalar_values[field] = _bounded_scalar(
            raw_values.get(normalized),
            label=field,
            default=default,
            minimum=minimum,
            maximum=maximum,
        )
    raw_key_color = next((raw_values[name] for name in _COLOR_NAMES if name in raw_values), None)
    key_color = _bounded_color(raw_key_color)
    settings = GreenScreenKeyerSettings(key_color=key_color, **scalar_values)
    return GreenScreenKeyerResolution(
        settings=settings,
        ignored_control_names=tuple(sorted(ignored_names, key=str.casefold)),
        used_default_key_color=raw_key_color is None,
    )


def _decode_bounded_base64(raw: str | None, *, label: str, maximum: int) -> bytes:
    if raw is None or not raw.strip():
        raise GreenScreenKeyerDataError(f"missing required {label}")
    compact = "".join(raw.split())
    if len(compact) > ((maximum + 2) // 3) * 4:
        raise GreenScreenKeyerDataError(f"{label} exceeds the {maximum}-byte portable limit")
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise GreenScreenKeyerDataError(f"{label} is not valid base64") from exc
    if len(decoded) > maximum:
        raise GreenScreenKeyerDataError(f"{label} exceeds the {maximum}-byte portable limit")
    return decoded


def _bounded_scalar(
    raw: str | None,
    *,
    label: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise GreenScreenKeyerDataError(f"keyer {label} must be numeric") from exc
    if not minimum <= value <= maximum:
        raise GreenScreenKeyerDataError(
            f"keyer {label} value {value!r} is outside [{minimum}, {maximum}]"
        )
    return value


def _bounded_color(raw: str | None) -> tuple[float, float, float]:
    if raw is None:
        return (0.0, 1.0, 0.0)
    pieces = raw.replace(",", " ").split()
    if len(pieces) not in {3, 4}:
        raise GreenScreenKeyerDataError("keyer key_color must have three RGB components")
    try:
        color = tuple(float(piece) for piece in pieces[:3])
    except ValueError as exc:
        raise GreenScreenKeyerDataError("keyer key_color must be numeric") from exc
    if any(component < 0 or component > 1 for component in color):
        raise GreenScreenKeyerDataError("keyer key_color components must be in [0, 1]")
    return color  # type: ignore[return-value]


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _normalize(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.9f}".rstrip("0").rstrip(".")
