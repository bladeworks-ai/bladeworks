"""Contract and drift tests for the public Bladeworks capability catalog."""

from __future__ import annotations

from fractions import Fraction
from importlib.resources import files

from fastapi.testclient import TestClient

from bladeworks.core.capabilities import CapabilityRegistry
from bladeworks.core.compositor import resolve_blend_mode
from bladeworks.core.model import Parameter, RenderTransition, ResolvedEffect
from bladeworks.preview.capabilities import (
    bladeworks_capabilities,
    create_capability_router,
)
from bladeworks.preview.routes import create_app
from bladeworks.tensor.effects import EFFECT_PORTS
from bladeworks.tensor.effects import LowerContext as EffectLowerContext
from bladeworks.tensor.effects import lower_effect
from bladeworks.tensor.tr_equirect import ADMITTED_EQUIRECT_IDS
from bladeworks.tensor.tr_phase5 import PHASE5_IDS
from bladeworks.tensor.transitions import ADMITTED_XFADE_IDS
from bladeworks.tensor.transitions import LowerContext as TransitionLowerContext
from bladeworks.tensor.transitions import lower_transition


class _Service:
    def shutdown(self) -> None:
        pass


def test_catalog_covers_renderer_registries_and_has_resource_identity() -> None:
    payload = bladeworks_capabilities()

    assert payload["schemaVersion"] == 1
    assert payload["renderer"] == "tensor"
    effects = payload["effects"]
    transitions = payload["transitions"]
    assert {effect["handler"] for effect in effects} == set(EFFECT_PORTS) - {
        "cohort_color_curves",
        "cohort_hue_saturation_curves",
    }
    assert all(effect["resource"]["uid"] for effect in effects)
    assert all(transition["resource"]["uid"] for transition in transitions)
    assert {
        transition["resource"]["xfadeId"]
        for transition in transitions
        if transition["resource"]["xfadeId"] is not None
    } == set(ADMITTED_XFADE_IDS) | set(PHASE5_IDS) | set(ADMITTED_EQUIRECT_IDS)


def test_catalog_parameter_keys_and_bounds_come_from_capability_registry() -> None:
    payload = bladeworks_capabilities()
    registry = CapabilityRegistry.load()
    gaussian_source = next(entry for entry in registry.entries if entry.id == "effect-gaussian")
    gaussian_public = next(effect for effect in payload["effects"] if effect["id"] == "effect-gaussian")

    assert {parameter["key"] for parameter in gaussian_public["parameters"]} == set(
        gaussian_source.parameters
    )
    amount = gaussian_public["parameters"][0]
    assert amount["name"] == "Amount"
    assert amount["min"] == 0
    assert amount["max"] == 1
    assert amount["animatable"] is False

    # The YAML records that Final Cut can keyframe Vibrancy, but Tensor's
    # cohort effect validator rejects animated controls. Public authoring must
    # describe executable behavior, not the source template's broader syntax.
    vibrancy_source = next(
        entry for entry in registry.entries if entry.id == "effect-vibrancy-cohort"
    )
    assert next(iter(vibrancy_source.parameters.values()))["keyframes"] is True
    vibrancy_public = next(
        effect for effect in payload["effects"] if effect["id"] == "effect-vibrancy-cohort"
    )
    assert vibrancy_public["parameters"][0]["animatable"] is False


def test_mask_catalog_publishes_the_executable_animated_contract() -> None:
    payload = bladeworks_capabilities()
    masks = next(item for item in payload["mechanics"] if item["id"] == "masks")

    assert masks["blendModes"] == ["add", "subtract", "multiply"]
    assert masks["invert"] is True
    assert masks["maximumMasks"] == 32
    sources = {source["id"]: source for source in masks["sourceKinds"]}
    assert set(sources) == {"shape", "draw", "color", "luma"}
    shape = {item["key"]: item for item in sources["shape"]["parameters"]}
    assert shape["160"] == {
        "key": "160",
        "name": "Radius",
        "type": "point",
        "min": 0.0,
        "max": 32768.0,
        "default": None,
        "components": ["x", "y"],
        "units": "image_plane_pixels",
        "minimumItems": None,
        "maximumItems": None,
        "convex": None,
        "animatable": True,
    }
    assert shape["104"]["min"] == 0.1
    assert shape["104"]["max"] == 8.0
    draw = {item["key"]: item for item in sources["draw"]["parameters"]}
    assert draw["points"]["type"] == "point_list"
    assert draw["points"]["minimumItems"] == 3
    assert draw["points"]["maximumItems"] == 64
    assert draw["points"]["convex"] is True
    assert draw["points"]["animatable"] is False
    assert draw["opacity"] == {
        "key": "opacity",
        "name": "Opacity",
        "type": "number",
        "min": 0.0,
        "max": 1.0,
        "default": 1.0,
        "components": [],
        "units": "normalized",
        "minimumItems": None,
        "maximumItems": None,
        "convex": None,
        "animatable": True,
    }
    assert set(draw) == {"points", "opacity"}
    assert all(
        parameter["animatable"] is False
        for source_id in ("color", "luma")
        for parameter in sources[source_id]["parameters"]
    )


def test_canonical_registry_is_an_importable_package_resource() -> None:
    resource = files("bladeworks.data").joinpath(
        "FCPXML_RENDER_CAPABILITIES.yaml"
    )

    assert resource.is_file()
    assert CapabilityRegistry.load().entries


def test_catalog_exposes_supported_and_explicitly_rejected_editor_surfaces() -> None:
    payload = bladeworks_capabilities()

    assert payload["retime"]["modes"] == [
        "constant",
        "reverse",
        "freeze",
        "piecewise_linear",
    ]
    assert payload["retime"]["frameSampling"] == ["floor"]
    assert {item["name"] for item in payload["blendModes"]} >= {
        "Normal",
        "Behind",
        "Multiply",
        "Stencil Alpha",
        "Silhouette Luma",
    }
    unsupported = {item["id"] for item in payload["unsupported"]}
    assert {
        "stabilization",
        "rollingShutter",
        "tracking",
        "smoothRetime",
        "frameBlending",
        "opticalFlow",
        "colorCurves",
        "hueSaturationCurves",
        "surroundAudio",
    } <= unsupported
    preview = next(item for item in payload["mechanics"] if item["id"] == "preview")
    assert preview["qualities"] == [720, 540, 480]
    assert payload["export"]["supportedResolutions"] == [1080, 720, 540, 480]
    assert payload["export"]["defaultResolution"] == 1080
    assert all(
        profile["defaultResolution"] == 1080
        for profile in payload["export"]["profiles"]
    )


def test_native_transition_catalog_only_exposes_executable_controls() -> None:
    transitions = {item["id"]: item for item in bladeworks_capabilities()["transitions"]}

    dissolve = transitions["transition-cross-dissolve"]
    assert dissolve["support"] == "default_only"
    assert dissolve["parameters"] == []

    slide = transitions["transition-slide"]
    push = transitions["transition-push"]
    assert slide["resource"] == push["resource"]
    assert slide["name"] == "Slide"
    assert push["name"] == "Push"
    for transition, mode in ((slide, "0"), (push, "2")):
        parameters = {item["key"]: item for item in transition["parameters"]}
        assert parameters["4"]["choices"] == ["0", "1", "2", "3"]
        assert parameters["5"]["default"] == mode
        assert parameters["5"]["choices"] == [mode]


def test_boolean_registry_parameters_publish_real_booleans() -> None:
    payload = bladeworks_capabilities()
    parameters = [
        parameter
        for transition in payload["transitions"]
        for parameter in transition["parameters"]
    ]
    boolean_parameters = [item for item in parameters if item["type"] == "boolean"]

    assert boolean_parameters
    assert all(isinstance(item.get("default"), bool) for item in boolean_parameters if "default" in item)
    assert all("choices" not in item for item in boolean_parameters)


def test_catalog_never_invents_missing_registry_defaults() -> None:
    """Every published default must be an explicitly registered value.

    This generated check covers every effect and transition. A missing default
    deliberately remains absent so Studio can let Final Cut and Tensor apply
    the resource's native default rather than synthesizing a minimum or zero.
    """

    payload = bladeworks_capabilities()
    registry = CapabilityRegistry.load()
    source_by_id = {
        entry.id: entry
        for entry in registry.entries
        if entry.handler is not None
    }
    for capability in [*payload["effects"], *payload["transitions"]]:
        if capability["handler"] == "slide_push":
            continue  # Public Slide/Push modes are the intentional expansion above.
        source = source_by_id[capability["id"]]
        for parameter in capability["parameters"]:
            raw = source.parameters[parameter["key"]]
            assert ("default" in parameter) == ("default" in raw)


def _parameter_default(parameter: dict[str, object]) -> Parameter:
    value = parameter["default"]
    # This is the exact FCPXML scalar spelling used by Studio's serializer.
    serialized = ("1" if value else "0") if isinstance(value, bool) else str(value)
    return Parameter(
        name=str(parameter["name"]),
        key=str(parameter["key"]),
        value=serialized,
    )


def test_every_advertised_effect_default_payload_lowers_or_requires_input() -> None:
    """Exercise every public effect at the real Tensor port boundary."""

    registry = CapabilityRegistry.load()
    source_by_id = {entry.id: entry for entry in registry.entries}
    context = EffectLowerContext(
        clip_path="catalog-default-audit",
        width=64,
        height=36,
        frame_duration=Fraction(1, 30),
        clip_duration=Fraction(1),
        source_colorspace="bt709",
        source_color_range="tv",
        reference_effect_link="rgba:bt709:tv",
    )
    required_inputs: set[str] = set()
    for public in bladeworks_capabilities()["effects"]:
        source = source_by_id[public["id"]]
        required = [item for item in public["parameters"] if item.get("required")]
        if required:
            required_inputs.add(public["id"])
            continue
        params = tuple(
            _parameter_default(item)
            for item in public["parameters"]
            if "default" in item
        )
        effect = ResolvedEffect(
            kind="video_filter",
            uid=source.uid,
            name=public["name"],
            handler=public["handler"],
            portable_status=source.portable_status,
            params=params,
            calibration=source.parameters,
            data={},
            path="catalog-default-audit",
            capability_id=source.id,
        )
        assert lower_effect(effect, context, frame_origin=0).handler == public["handler"]

    assert required_inputs == {"effect-green-screen-keyer"}


def test_every_advertised_transition_default_payload_lowers() -> None:
    """Resolve registry defaults exactly as the compiler does, then lower them."""

    registry = CapabilityRegistry.load()
    source_by_id = {entry.id: entry for entry in registry.entries}
    context = TransitionLowerContext(
        width=64,
        height=36,
        frame_duration=Fraction(1, 30),
        frame_count=30,
    )
    for public in bladeworks_capabilities()["transitions"]:
        source = (
            source_by_id["transition-slide-push"]
            if public["handler"] == "slide_push"
            else source_by_id[public["id"]]
        )
        params = tuple(
            _parameter_default(item)
            for item in public["parameters"]
            if "default" in item
        )
        parameter_values: dict[str, object] = {}
        if source.handler == "xfade":
            from bladeworks.transitions.contract import (
                parse_parameter_specs,
                resolve_parameter_values,
                semantic_parameter_values,
            )

            specs = parse_parameter_specs(source.parameters, max_slots=None)
            parameter_values = semantic_parameter_values(
                specs, resolve_parameter_values(specs, params)
            )
        elif source.handler == "equirectangular":
            from bladeworks.transitions.equirectangular import (
                parse_equirectangular_parameter_specs,
                resolve_equirectangular_parameter_values,
                semantic_parameter_values,
            )

            specs = parse_equirectangular_parameter_specs(source.parameters)
            resolved = resolve_equirectangular_parameter_values(specs, params)
            assert source.xfade is not None
            parameter_values = semantic_parameter_values(str(source.xfade["id"]), resolved)
        transition = RenderTransition(
            path="catalog-default-audit",
            absolute_start=Fraction(0),
            duration=Fraction(1),
            uid=source.uid,
            name=public["name"],
            handler=public["handler"],
            params=params,
            capability_id=public["id"],
            portable_status=source.portable_status,
            xfade_id=public["resource"]["xfadeId"],
            parameter_values=parameter_values,
        )
        assert lower_transition(transition, context).kind


def test_all_blend_modes_publish_the_exact_fcpxml_serializer_value() -> None:
    modes = bladeworks_capabilities()["blendModes"]

    assert len(modes) == 23
    assert len({mode["fcpxmlValue"] for mode in modes}) == 23
    assert {mode["fcpxmlValue"] for mode in modes} >= {
        "normal",
        "soft-light",
        "color-burn",
        "linear-light",
        "stencil-alpha",
        "silhouette-luma",
    }
    for mode in modes:
        assert resolve_blend_mode(mode["fcpxmlValue"]).canonical_name == mode["name"]


def test_capability_route_is_bearer_authenticated() -> None:
    app = create_app(
        _Service(),  # type: ignore[arg-type]
        auth_token="secret-token",
        protected_routers=(create_capability_router(),),
    )

    with TestClient(app) as client:
        unauthorized = client.get("/api/editor/capabilities")
        authorized = client.get(
            "/api/editor/capabilities",
            headers={"Authorization": "Bearer secret-token"},
        )

    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "unauthorized"
    assert authorized.status_code == 200
    assert authorized.json()["renderer"] == "tensor"
