"""Public, renderer-owned capability catalog for Bladeworks Studio.

Architecture map
================

    tensor effect and transition registries
        + FCPXML capability registry (UIDs and typed parameter contracts)
        + core renderer policy (mechanics, media, audio, export)
        -> ``bladeworks_capabilities`` validates registry coverage
        -> authenticated ``GET /api/editor/capabilities``
        -> Studio builds only controls Bladeworks can execute

The renderer is the rate limiter.  This module does not describe what the
browser happens to implement today.  A missing browser control for an item in
this response is a Studio bug.  Conversely, unsupported entries are explicit
so the browser cannot accidentally expose controls that Bladeworks rejects.

Resource identity is an invariant: every authorable effect and transition has
the exact FCPXML resource UID from ``FCPXML_RENDER_CAPABILITIES.yaml``.  The
catalog fails loudly when the renderer registry grows without a matching
resource declaration, preventing a silent UI/backend drift.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from fastapi import APIRouter

from ..core.capabilities import Capability, CapabilityRegistry
from ..core.compositor import _MODE_SPECS  # renderer-owned canonical map
from ..core.masks import MASK_AUTHORING_SOURCES, MASK_BLEND_MODES, MAX_MASKS
from ..tensor.effects import EFFECT_PORTS
from ..tensor.tr_equirect import ADMITTED_EQUIRECT_IDS
from ..tensor.tr_phase5 import PHASE5_IDS
from ..tensor.resolution import DEFAULT_PROFILE, SUPPORTED_PROFILES, RenderMode
from ..tensor.transitions import ADMITTED_XFADE_IDS, HANDLERS
from .render_jobs import STUDIO_EXPORT_PROFILES

SCHEMA_VERSION = 1

_IDENTITY_EFFECTS = frozenset(
    {"cohort_color_curves", "cohort_hue_saturation_curves"}
)
_DEFAULT_ONLY_EFFECTS = frozenset(
    {
        "negative",
        "threshold",
        "mirror",
        "colorize",
        "tint",
        "flipped",
        "add_noise",
        "pixellate_default",
        "cohort_drop_shadow",
        "cohort_callout",
        "cohort_directional_blur",
        "cohort_fisheye",
        "cohort_vignette_mask",
        "cohort_kaleidoscope",
        "cohort_perspective_tile",
    }
)
_PARTIAL_EFFECTS = frozenset(
    {"cohort_cartoon", "cohort_camcorder", "cohort_focus_blur", "cohort_earthquake"}
)
_EXACT_EFFECTS = frozenset({"color_adjustments"})
_NATIVE_TRANSITION_HANDLERS = frozenset(
    {"cross_dissolve", "fade_color", "wipe", "slide_push"}
)


def _parameter(
    key: str,
    raw: Mapping[str, Any],
    *,
    animatable: bool,
) -> dict[str, Any]:
    """Translate one renderer parameter declaration to the stable HTTP shape.

    Main callers:
    - ``_effect`` and ``_transition`` while materializing the catalog.

    Why this exists:
    The internal registry contains calibration constants alongside its public
    type contract.  Studio needs the exact serialized key and editing bounds,
    but must not depend on private FFmpeg scale factors.
    """

    semantic_type = raw.get("semantic_type")
    raw_type = raw.get("type")
    components = int(raw.get("components", 1))
    if raw_type in {"bool", "boolean"} or semantic_type == "boolean":
        parameter_type = "boolean"
    elif semantic_type == "enum" or "allowed" in raw:
        parameter_type = "enum"
    elif raw_type in {"integer", "color", "position", "vector"}:
        parameter_type = {
            "position": "point",
            "vector": "point",
        }.get(str(raw_type), str(raw_type))
    elif components in {3, 4} and "color" in str(raw.get("name", "")).lower():
        parameter_type = "color"
    else:
        parameter_type = "number"

    result: dict[str, Any] = {
        "key": key,
        "name": str(raw.get("name", key)),
        "type": parameter_type,
        "animatable": animatable,
    }
    if raw.get("required") is True:
        result["required"] = True
    if "default" in raw:
        if parameter_type == "boolean":
            value = raw["default"]
            result["default"] = value if isinstance(value, bool) else str(value).strip().casefold() in {"1", "true"}
        else:
            result["default"] = (
                str(raw["default"]) if parameter_type == "enum" else raw["default"]
            )
    if "minimum" in raw:
        result["min"] = raw["minimum"]
    if "maximum" in raw:
        result["max"] = raw["maximum"]
    if "allowed" in raw and parameter_type != "boolean":
        result["choices"] = [str(value) for value in raw["allowed"]]
    if components != 1:
        if parameter_type == "color":
            component_names = ["red", "green", "blue", "alpha"][:components]
        elif parameter_type == "point":
            component_names = ["x", "y"][:components]
        else:
            component_names = [f"component{index + 1}" for index in range(components)]
        result["components"] = component_names
    return result


def _display_name(entry: Capability) -> str:
    return entry.aliases[0] if entry.aliases else entry.id


def _effect(entry: Capability) -> dict[str, Any]:
    handler = entry.handler or ""
    if handler in _DEFAULT_ONLY_EFFECTS:
        support = "default_only"
        notes = "Bladeworks renders the calibrated default and rejects authored parameters."
    elif handler in _PARTIAL_EFFECTS:
        support = "partial"
        notes = "Bladeworks honors the listed controls; other template controls are unavailable."
    elif handler in _EXACT_EFFECTS:
        support = "exact"
        notes = "Bladeworks honors the complete listed parameter contract."
    else:
        support = "approximate"
        notes = entry.approximation or "Bladeworks renders a calibrated approximation."
    assert entry.uid is not None, f"authorable effect {entry.id} has no FCPXML UID"
    return {
        "id": entry.id,
        "name": _display_name(entry),
        "handler": handler,
        "resource": {"uid": entry.uid, "xfadeId": None},
        "authorable": True,
        "support": support,
        # Ordinary effect parameter animation is rejected or not executed by
        # Tensor. Mask mattes publish their separate, genuinely animated ABI.
        "parameters": [
            _parameter(key, raw, animatable=False)
            for key, raw in entry.parameters.items()
        ],
        "notes": [notes],
    }


def _transition(entry: Capability) -> dict[str, Any]:
    xfade_id = str(entry.xfade["id"]) if entry.xfade else None
    assert entry.uid is not None, f"authorable transition {entry.id} has no FCPXML UID"
    support = "exact" if entry.handler == "cross_dissolve" else "approximate"
    parameters = entry.parameters
    if entry.handler == "cross_dissolve":
        # Tensor implements Final Cut's fixed default dissolve. The template's
        # Look/Ease metadata never changes the kernel and must not become fake
        # Studio controls.
        support = "default_only"
        parameters = {}
    if not entry.parameters:
        support = "default_only" if entry.handler == "xfade" else support
    return {
        "id": entry.id,
        "name": _display_name(entry),
        "handler": entry.handler,
        "resource": {"uid": entry.uid, "xfadeId": xfade_id},
        "authorable": True,
        "support": support,
        "parameters": [
            _parameter(key, raw, animatable=False)
            for key, raw in parameters.items()
        ],
        "notes": [
            entry.approximation
            or (
                "Exact premultiplied linear-light dissolve."
                if support == "exact"
                else "Calibrated Bladeworks transition."
            )
        ],
    }


def _slide_push_transitions(entry: Capability) -> list[dict[str, Any]]:
    """Publish the two real modes behind Final Cut's shared resource UID.

    Main callers:
    - ``_registry_surfaces`` while expanding the native transition registry.

    Final Cut uses one resource for Slide and Push and distinguishes them with
    parameter key ``5``. Separate catalog tiles remain executable because each
    carries a one-choice fixed mode parameter that Studio serializes normally.
    """

    direction = entry.parameters["4"]
    result: list[dict[str, Any]] = []
    for identifier, name, mode in (
        ("transition-slide", "Slide", "0"),
        ("transition-push", "Push", "2"),
    ):
        public = _transition(entry)
        public["id"] = identifier
        public["name"] = name
        public["parameters"] = [
            _parameter("4", direction, animatable=False),
            _parameter(
                "5",
                {"name": "Mode", "type": "scalar", "default": mode, "allowed": [mode], "semantic_type": "enum"},
                animatable=False,
            ),
        ]
        public["notes"] = [
            f"Calibrated Final Cut {name} using shared resource mode {mode}."
        ]
        result.append(public)
    return result


def _registry_surfaces() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve public effects/transitions and reject any registry drift.

    Main callers:
    - ``bladeworks_capabilities`` on each HTTP request and in drift tests.

    Keeping this check on the serving path matters: adding a tensor port without
    a browser-safe FCPXML identity is a broken release, not an optional catalog
    omission.
    """

    registry = CapabilityRegistry.load()
    effect_entries = [
        entry
        for entry in registry.entries
        if entry.kind == "video_filter" and entry.handler in EFFECT_PORTS
    ]
    registered_effects = set(EFFECT_PORTS)
    catalog_effects = {entry.handler for entry in effect_entries if entry.handler}
    if catalog_effects != registered_effects:
        raise RuntimeError(
            "Tensor effect registry and FCPXML resource registry disagree: "
            f"ports_only={sorted(registered_effects - catalog_effects)!r}, "
            f"resources_only={sorted(catalog_effects - registered_effects)!r}"
        )

    admitted_xfade_ids = set(ADMITTED_XFADE_IDS) | set(PHASE5_IDS) | set(ADMITTED_EQUIRECT_IDS)
    transition_entries = [
        entry
        for entry in registry.entries
        if entry.kind == "transition"
        and (
            entry.handler in _NATIVE_TRANSITION_HANDLERS
            or (entry.xfade is not None and entry.xfade.get("id") in admitted_xfade_ids)
        )
    ]
    catalog_xfade_ids = {
        str(entry.xfade["id"]) for entry in transition_entries if entry.xfade is not None
    }
    if catalog_xfade_ids != admitted_xfade_ids:
        raise RuntimeError(
            "Tensor transition admission and FCPXML resource registry disagree: "
            f"ports_only={sorted(admitted_xfade_ids - catalog_xfade_ids)!r}, "
            f"resources_only={sorted(catalog_xfade_ids - admitted_xfade_ids)!r}"
        )
    native_handlers = {entry.handler for entry in transition_entries} & _NATIVE_TRANSITION_HANDLERS
    if native_handlers != _NATIVE_TRANSITION_HANDLERS or not _NATIVE_TRANSITION_HANDLERS <= set(HANDLERS):
        raise RuntimeError("Native tensor transition handlers and capability resources disagree")

    effects = [
        _effect(entry)
        for entry in effect_entries
        if entry.handler not in _IDENTITY_EFFECTS
    ]
    transitions = [
        public
        for entry in transition_entries
        for public in (
            _slide_push_transitions(entry)
            if entry.handler == "slide_push"
            else [_transition(entry)]
        )
    ]
    return effects, transitions


def bladeworks_capabilities() -> dict[str, Any]:
    """Return the complete public authoring and delivery contract."""

    effects, transitions = _registry_surfaces()
    blend_modes_by_name = {spec.canonical_name: spec for spec in _MODE_SPECS.values()}
    blend_modes = [
        {
            "id": "".join(character.lower() for character in name if character.isalnum()),
            "name": name,
            "fcpxmlValue": re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
            "support": "exact"
            if spec.semantic_status == "exact_alpha"
            else "approximate",
            "authorable": True,
        }
        for name, spec in sorted(blend_modes_by_name.items())
    ]
    return {
        "schemaVersion": SCHEMA_VERSION,
        "renderer": "tensor",
        "mechanics": [
            {"id": "timeline", "name": "Spine, connected clips, lanes and gaps", "support": "exact", "authorable": True},
            {"id": "transform", "name": "Position, scale, rotation and anchor", "support": "exact", "authorable": True, "animatable": True},
            {"id": "crop", "name": "Trim, Crop and Ken Burns", "support": "approximate", "authorable": True, "animatable": True, "modes": ["trim", "crop", "ken_burns"]},
            {"id": "distort", "name": "Four-corner Distort", "support": "exact", "authorable": True, "animatable": True},
            {"id": "spatialConform", "name": "Spatial conform", "support": "exact", "authorable": True, "modes": ["fit", "fill", "none"]},
            {"id": "opacity", "name": "Opacity", "support": "exact", "authorable": True, "animatable": True},
            {
                "id": "masks",
                "name": "Numeric shape, draw, color and luma masks",
                "support": "approximate",
                "authorable": True,
                "animatable": True,
                "maximumMasks": MAX_MASKS,
                "blendModes": list(MASK_BLEND_MODES),
                "invert": True,
                "sourceKinds": [
                    {
                        **{
                            key: list(value) if isinstance(value, tuple) else value
                            for key, value in source.items()
                            if key != "parameters"
                        },
                        "parameters": [
                            {
                                "key": parameter["key"],
                                "name": parameter["name"],
                                "type": parameter["type"],
                                "min": parameter.get("minimum"),
                                "max": parameter.get("maximum"),
                                "default": parameter.get("default"),
                                "components": list(parameter.get("components", ())),
                                "units": parameter["units"],
                                "minimumItems": parameter.get("minimumItems"),
                                "maximumItems": parameter.get("maximumItems"),
                                "convex": parameter.get("convex"),
                                "animatable": parameter["animatable"],
                            }
                            for parameter in source["parameters"]
                        ],
                    }
                    for source in MASK_AUTHORING_SOURCES
                ],
            },
            {"id": "titles", "name": "Titles and captions", "support": "partial", "authorable": True, "notes": "Supported title text/styles rasterize in the preview service; opaque Motion rigs reject."},
            {"id": "customSolid", "name": "Custom Solid generator", "support": "exact", "authorable": True},
            {"id": "nestedScopes", "name": "Compound, multicam, sync and audition scopes", "support": "exact", "authorable": True},
            {"id": "preview", "name": "Seek and scan preview", "support": "exact", "authorable": False, "qualities": [720, 540, 480]},
        ],
        "blendModes": blend_modes,
        "retime": {
            "support": "exact",
            "authorable": True,
            "modes": ["constant", "reverse", "freeze", "piecewise_linear"],
            "frameSampling": ["floor"],
            "preservePitch": True,
            "notes": "Matches Final Cut's constant speed, Reverse, Hold and segmented retime editor model. Smooth speed transitions are unsupported.",
        },
        "effects": effects,
        "transitions": transitions,
        "audio": {
            "support": "exact",
            "authorable": True,
            "controls": ["gain", "mute", "fade_in", "fade_out", "pan", "roles", "j_l_edits", "resampling"],
            "outputLayouts": ["mono", "stereo"],
        },
        "media": {
            "support": "exact",
            "decodedPixelFormats": ["yuv420p", "yuv422p", "yuv444p", "yuvj420p", "yuvj422p", "yuvj444p", "yuv420p10le", "yuv422p10le", "yuv444p10le", "yuv444p12le"],
            "colorMatrices": ["bt709", "bt601", "smpte170m", "bt2020nc"],
            "hdrInputTransfers": ["arib-std-b67", "smpte2084"],
            "missingMedia": "placeholder_video_and_silent_audio",
        },
        "export": {
            "support": "exact",
            "supportedResolutions": [
                int(profile.value.removesuffix("p"))
                for profile in SUPPORTED_PROFILES[RenderMode.RENDER]
            ],
            "defaultResolution": int(
                DEFAULT_PROFILE[RenderMode.RENDER].value.removesuffix("p")
            ),
            "profiles": [
                {
                    "id": profile.id,
                    "container": profile.container,
                    "video": profile.video_codec,
                    "audio": profile.audio_codec,
                    "alpha": profile.alpha,
                    "defaultResolution": 1080,
                }
                for profile in STUDIO_EXPORT_PROFILES.values()
            ],
        },
        "unsupported": [
            {"id": "stabilization", "category": "mechanic", "reason": "Bladeworks rejects stabilization metadata."},
            {"id": "rollingShutter", "category": "mechanic", "reason": "Bladeworks rejects rolling-shutter correction."},
            {"id": "tracking", "category": "mask", "reason": "Opaque, tracked, Magnetic, Auto and ML masks reject."},
            {"id": "smoothRetime", "category": "retime", "reason": "Smooth/eased time-map interpolation rejects."},
            {"id": "frameBlending", "category": "retime", "reason": "Frame Blending is not implemented."},
            {"id": "opticalFlow", "category": "retime", "reason": "Optical Flow and ML retiming are not implemented."},
            {"id": "colorCurves", "category": "effect", "reason": "Authored curve data rejects; the identity port is not authorable."},
            {"id": "hueSaturationCurves", "category": "effect", "reason": "Authored curve data rejects; the identity port is not authorable."},
            {"id": "customLuts", "category": "effect", "reason": "Custom LUTs reject."},
            {"id": "motionTitleRigs", "category": "title", "reason": "Opaque Motion title rigs reject."},
            {"id": "nonCustomSolidGenerators", "category": "generator", "reason": "Only Custom Solid generators render."},
            {"id": "surroundAudio", "category": "audio", "reason": "5.1 output rejects before delivery planning."},
            {"id": "alphaSourceMedia", "category": "media", "reason": "Alpha-carrying decoded source formats reject."},
            {"id": "hdrDelivery", "category": "export", "reason": "HDR input is tone-mapped to SDR; HDR output is not offered."},
            {"id": "oracleMezzanine", "category": "export", "reason": "The tensor encoder has no ProRes 4:2:2 10-bit exit; the legacy CPU-only oracle profile is not a Studio capability."},
            {"id": "crossChannelBlendModes", "category": "blend", "reason": "Hue, Saturation, Color and Luminosity blend modes reject."},
        ],
    }


def create_capability_router() -> APIRouter:
    """Build the narrow router mounted behind the server bearer dependency."""

    router = APIRouter(prefix="/api/editor", tags=["capabilities"])

    @router.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        return bladeworks_capabilities()

    return router
