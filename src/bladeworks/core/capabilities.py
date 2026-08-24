"""Load and query the renderer's machine-readable FCPXML capability registry."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Optional

from .errors import FCPXMLCompileError


DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[1] / "data" / "FCPXML_RENDER_CAPABILITIES.yaml"
)


@dataclass(frozen=True)
class Capability:
    id: str
    kind: str
    uid: Optional[str]
    uid_glob: Optional[str]
    aliases: tuple[str, ...]
    authoring_status: str
    portable_status: str
    handler: Optional[str]
    parameters: Mapping[str, Any]
    calibration_fixtures: tuple[str, ...]
    omission: str
    approximation: str = ""
    xfade: Optional[Mapping[str, Any]] = None
    transition_artifact: Optional[Mapping[str, Any]] = None
    effect_artifact: Optional[Mapping[str, Any]] = None


class CapabilityRegistry:
    def __init__(self, entries: tuple[Capability, ...], *, schema_version: int):
        if schema_version != 1:
            raise FCPXMLCompileError(f"unsupported capability registry schema {schema_version}")
        valid_authoring = {"authorable", "preserve_only", "out_of_scope"}
        valid_portable = {"exact_portable", "calibrated_portable", "apple_only", "unsupported"}
        seen_ids: set[str] = set()
        seen_uids: set[tuple[str, str]] = set()
        for entry in entries:
            if entry.id in seen_ids:
                raise FCPXMLCompileError(f"duplicate capability id {entry.id!r}")
            seen_ids.add(entry.id)
            if entry.authoring_status not in valid_authoring:
                raise FCPXMLCompileError(f"capability {entry.id} has invalid authoring status")
            if entry.portable_status not in valid_portable:
                raise FCPXMLCompileError(f"capability {entry.id} has invalid portable status")
            if entry.uid is not None:
                identity = (entry.kind, entry.uid)
                if identity in seen_uids:
                    raise FCPXMLCompileError(f"duplicate capability UID {entry.uid!r} for {entry.kind}")
                seen_uids.add(identity)
            for key, raw in entry.parameters.items():
                if not isinstance(raw, dict):
                    raise FCPXMLCompileError(f"capability {entry.id} parameter {key!r} must be an object")
                if "minimum" in raw and "maximum" in raw and raw["minimum"] > raw["maximum"]:
                    raise FCPXMLCompileError(f"capability {entry.id} parameter {key!r} has an invalid range")
            if entry.transition_artifact is not None:
                from ..transitions.contract import ABI_VERSION, parse_parameter_specs

                artifact = entry.transition_artifact
                required = {"abi", "id", "version", "source", "license"}
                if set(artifact) != required or artifact.get("abi") != ABI_VERSION:
                    raise FCPXMLCompileError(
                        f"capability {entry.id} has an invalid spell-transition-v1 artifact declaration"
                    )
            if entry.handler == "spell_transition_vulkan":
                from ..transitions.contract import parse_parameter_specs

                parse_parameter_specs(entry.parameters)
                if entry.transition_artifact is None:
                    raise FCPXMLCompileError(f"capability {entry.id} requires a transition_artifact")
            if entry.handler == "xfade":
                from ..transitions.contract import parse_parameter_specs
                from ..transitions.stock import IMPLEMENTATION_IDS

                # Stock-FFmpeg handlers store validated values in the render
                # plan rather than Vulkan push constants, so they are not
                # subject to the shader ABI's four-slot limit.
                parse_parameter_specs(entry.parameters, max_slots=None)
                implementation = entry.xfade or {}
                if set(implementation) != {"id"} or implementation.get("id") not in IMPLEMENTATION_IDS:
                    raise FCPXMLCompileError(
                        f"capability {entry.id} has an invalid bounded xfade implementation"
                    )
            if entry.handler == "equirectangular":
                from ..transitions.equirectangular import (
                    IMPLEMENTATION_IDS,
                    parse_equirectangular_parameter_specs,
                )

                parse_equirectangular_parameter_specs(entry.parameters)
                implementation = entry.xfade or {}
                if set(implementation) != {"id"} or implementation.get("id") not in IMPLEMENTATION_IDS:
                    raise FCPXMLCompileError(
                        f"capability {entry.id} has an invalid bounded 360 transition implementation"
                    )
            if entry.handler == "spell_effect_vulkan" or entry.effect_artifact is not None:
                from ..effects.contract import ABI_VERSION, parse_parameter_specs

                parse_parameter_specs(entry.parameters)
                artifact = entry.effect_artifact
                if not artifact:
                    raise FCPXMLCompileError(f"capability {entry.id} requires an effect_artifact")
                required = {"abi", "id", "version", "source", "license"}
                if set(artifact) != required or artifact.get("abi") != ABI_VERSION:
                    raise FCPXMLCompileError(
                        f"capability {entry.id} has an invalid spell-effect-v1 artifact declaration"
                    )
        self.schema_version = schema_version
        self.entries = entries

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CapabilityRegistry":
        registry_path = path or DEFAULT_REGISTRY
        try:
            text = registry_path.read_text(encoding="utf-8")
            # The committed .yaml uses JSON-compatible YAML. This keeps the CLI
            # usable in the repository's minimal Python environment; PyYAML is
            # not a hidden runtime requirement for a static registry.
            payload = json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            raise FCPXMLCompileError(f"could not load capability registry {registry_path}: {exc}") from exc
        entries = [
            Capability(
                id=str(raw["id"]),
                kind=str(raw["kind"]),
                uid=raw.get("uid"),
                uid_glob=raw.get("uid_glob"),
                aliases=tuple(raw.get("aliases", [])),
                authoring_status=str(raw["authoring_status"]),
                portable_status=str(raw["portable_status"]),
                handler=raw.get("handler"),
                parameters=raw.get("parameters", {}),
                calibration_fixtures=tuple(raw.get("calibration_fixtures", [])),
                omission=str(raw.get("omission", "omit and report")),
                approximation=str(raw.get("approximation", "")),
                xfade=raw.get("xfade"),
                transition_artifact=raw.get("transition_artifact"),
                effect_artifact=raw.get("effect_artifact"),
            )
            for raw in payload.get("capabilities", [])
        ]
        for index, uid in enumerate(payload.get("documented_motion_transition_uids", []), start=1):
            if any(entry.kind == "transition" and entry.uid == str(uid) for entry in entries):
                continue
            entries.append(
                Capability(
                    id=f"documented-motion-transition-{index}",
                    kind="transition",
                    uid=str(uid),
                    uid_glob=None,
                    aliases=(),
                    authoring_status="authorable",
                    portable_status="unsupported",
                    handler=None,
                    parameters={},
                    calibration_fixtures=(),
                    omission="documented Motion transition becomes a hard cut in the MVP",
                    transition_artifact=None,
                    effect_artifact=None,
                )
            )
        for index, uid in enumerate(payload.get("documented_title_uids", []), start=1):
            entries.append(
                Capability(
                    id=f"documented-title-{index}",
                    kind="title",
                    uid=str(uid),
                    uid_glob=None,
                    aliases=(),
                    authoring_status="authorable",
                    portable_status="apple_only",
                    handler=None,
                    parameters={},
                    calibration_fixtures=(),
                    omission="non-Basic Motion title template omitted in the MVP",
                    transition_artifact=None,
                    effect_artifact=None,
                )
            )
        for index, raw in enumerate(payload.get("documented_unsupported_transitions", []), start=1):
            if any(entry.kind == "transition" and entry.uid == str(raw["uid"]) for entry in entries):
                continue
            entries.append(
                Capability(
                    id=f"documented-unsupported-transition-{index}",
                    kind="transition",
                    uid=str(raw["uid"]),
                    uid_glob=None,
                    aliases=tuple(raw.get("aliases", [])),
                    authoring_status="authorable",
                    portable_status="unsupported",
                    handler=None,
                    parameters=raw.get("parameters", {}),
                    calibration_fixtures=tuple(raw.get("calibration_fixtures", [])),
                    omission="documented transition becomes a hard cut in the MVP",
                    approximation="",
                    xfade=None,
                    transition_artifact=None,
                    effect_artifact=None,
                )
            )
        for index, raw in enumerate(payload.get("documented_unsupported_effects", []), start=1):
            if any(entry.kind == "video_filter" and entry.uid == str(raw["uid"]) for entry in entries):
                continue
            entries.append(
                Capability(
                    id=f"documented-unsupported-effect-{index}",
                    kind="video_filter",
                    uid=str(raw["uid"]),
                    uid_glob=None,
                    aliases=tuple(raw.get("aliases", [])),
                    authoring_status="authorable",
                    portable_status="unsupported",
                    handler=None,
                    parameters=raw.get("parameters", {}),
                    calibration_fixtures=tuple(raw.get("calibration_fixtures", [])),
                    omission="documented effect omitted; underlying clip remains",
                    approximation="",
                    xfade=None,
                    transition_artifact=None,
                    effect_artifact=None,
                )
            )
        return cls(tuple(entries), schema_version=int(payload.get("schema_version", 0)))

    def match(self, *, kind: str, uid: Optional[str], name: Optional[str] = None) -> Optional[Capability]:
        # Exact UIDs always win. Explicit aliases are only considered after UID
        # matching and are stored in the registry, never guessed from display text.
        for entry in self.entries:
            if entry.kind == kind and uid is not None and entry.uid == uid:
                return entry
        for entry in self.entries:
            if entry.kind == kind and uid is not None and entry.uid_glob and fnmatch.fnmatchcase(uid, entry.uid_glob):
                return entry
        if name and uid is None:
            normalized = _normalize(name)
            for entry in self.entries:
                if (
                    entry.kind == kind
                    and entry.uid is None
                    and normalized in {_normalize(alias) for alias in entry.aliases}
                ):
                    return entry
        return None


def _normalize(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())
