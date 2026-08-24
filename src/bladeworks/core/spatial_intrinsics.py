"""Plan advanced Final Cut spatial intrinsics with stock FFmpeg filters.

Architecture map
================

``SpatialIntrinsicPlan``
    Typed, validated display, color, stereo, stabilization, and 360 controls.

``build_spatial_execution_plan``
    Applies the renderer's canonical early-video order and emits a label-aware
    FFmpeg graph::

        pixel-aspect display raster / display rotation
        -> SDR or HDR-to-SDR color conform
        -> stereo layout / eye order / convergence
        -> bounded semantic stabilization
        -> 360 reorientation
        -> 360 viewer orientation or Tiny Planet

``TrackerKeyframe`` / ``build_tracker_animation_hook``
    Convert genuinely readable tracker samples into the shared exact retime
    and animation kernel.  Locator-only tracker data never enters this path.

``probe_stock_ffmpeg_spatial_capabilities``
    Verifies the exact filter set used by a plan.  It only inspects an existing
    FFmpeg executable; no Vulkan runtime or custom FFmpeg build is involved.

Important invariants
--------------------

* Every numeric value is finite and bounded before it reaches a filter string.
* Graph labels are restricted to alphanumerics and underscores.  User strings
  never become filter names, file paths, or free-form FFmpeg options.
* HDR conform uses renderer-owned, checked-in 3D LUTs.  A render cannot name
  an arbitrary LUT from FCPXML.
* Stabilization presets expose no log filename or caller-supplied deshake
  option.  This prevents unexpected filesystem writes and unbounded searches.
* Semantic approximations and unimplemented active controls are returned as
  typed findings.  A missing implementation never turns into a silent no-op.
* Source pixel aspect is baked into a square-pixel raster before rotation.
  Downstream crop and conform may normalize SAR metadata without changing the
  authored display geometry.

Central integration seam
------------------------

The parser already preserves these elements in
``StoryNode.preserved_adjustments`` and format metadata in ``FormatResource``.
The root integrator should:

1. Promote the preserved attributes into the records in this module.
2. Attach one ``SpatialIntrinsicPlan`` to each render-media node.
3. Insert ``SpatialExecutionPlan.filter_complex`` before crop/effects in the
   shared video graph and merge ``required_filters`` into host preflight.
4. Convert every ``SpatialFinding`` into the compatibility report at the
   owning node's FCPXML path.
5. Feed readable tracker samples through ``build_tracker_animation_hook`` and
   connect its tracks to transform/mask animation.  Locator-only records use
   ``OpaqueTrackerLocator`` instead.

Why this exists:
The source parser was already preserving these controls, but the portable
backend could only report them as omitted.  This module freezes an executable,
bounded stock-FFmpeg contract without editing the shared parser/compiler while
the experimental renderer is still being validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Literal, Mapping, Sequence, TypeAlias

from .animation import (
    AnimatedScalar,
    AnimatedVec2,
    ScalarControlPoint,
    TimelineAnimatedScalar,
    TimelineAnimatedVec2,
    Vec2ControlPoint,
)
from .retime import RetimeMap


Projection: TypeAlias = Literal[
    "none", "equirectangular", "fisheye", "back-to-back fisheye", "cubic"
]
StereoLayout: TypeAlias = Literal["mono", "side by side", "over under"]
HeroEye: TypeAlias = Literal["left", "right"]
OrientationMapping: TypeAlias = Literal["normal", "tinyPlanet"]
StabilizationMode: TypeAlias = Literal["automatic", "inertiaCam", "smoothCam"]
ColorConformMode: TypeAlias = Literal[
    "conformNone",
    "conformAuto",
    "conformHLGtoSDR",
    "conformPQtoSDR",
    "conformHLGtoPQ",
    "conformPQtoHLG",
    "conformSDRtoHLG75",
    "conformSDRtoHLG100",
    "conformSDRtoPQ",
]
FindingOutcome: TypeAlias = Literal["approximated", "not_implemented_yet"]


class SpatialIntrinsicError(ValueError):
    """Base error for unsafe, malformed, or unavailable spatial execution."""


class SpatialValidationError(SpatialIntrinsicError):
    """A spatial record cannot be interpreted without guessing."""


class MissingFFmpegSpatialCapability(SpatialIntrinsicError):
    """The selected FFmpeg executable lacks a filter required by the plan."""


_FILTER_LABEL = re.compile(r"^[A-Za-z0-9_]+$")
_FILTER_ROW = re.compile(r"^\s*[TSC.]{2,3}\s+([A-Za-z0-9_]+)\s", re.MULTILINE)
_MAX_FRAME_DIMENSION = 16_384
_MAX_TRACKER_POINTS = 1_024
_LUT_DIRECTORY = Path(__file__).resolve().parents[1] / "spatial_luts"
_HDR_LUTS = {
    "conformHLGtoSDR": _LUT_DIRECTORY / "rec2020_hlg_to_rec709_sdr_v1.cube",
    "conformPQtoSDR": _LUT_DIRECTORY / "rec2020_pq_to_rec709_sdr_v1.cube",
}


def _finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise SpatialValidationError(f"{name} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SpatialValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise SpatialValidationError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise SpatialValidationError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise SpatialValidationError(f"{name} must be at most {maximum}")
    return result


def _positive_int(value: object, *, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpatialValidationError(f"{name} must be a positive integer")
    if value > maximum:
        raise SpatialValidationError(f"{name} exceeds the maximum {maximum}")
    return value


def _exact_time(value: object, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise SpatialValidationError(f"{name} must be an exact Fraction, not bool")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    raise SpatialValidationError(
        f"{name} must be an exact Fraction, got {type(value).__name__}"
    )


def _number(value: float | int) -> str:
    result = _finite(value, name="FFmpeg number")
    if abs(result) < 5e-13:
        result = 0.0
    return format(result, ".12g")


def _parse_bool(raw: str | None, *, name: str, default: bool) -> bool:
    if raw is None:
        return default
    if raw == "1":
        return True
    if raw == "0":
        return False
    raise SpatialValidationError(f"{name} must be '0' or '1'")


def _parse_float(
    attributes: Mapping[str, str],
    name: str,
    default: float = 0.0,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    return _finite(
        attributes.get(name, default),
        name=name,
        minimum=minimum,
        maximum=maximum,
    )


def _parse_pair(raw: str | None, *, name: str, default: tuple[float, float]) -> tuple[float, float]:
    if raw is None:
        return default
    parts = raw.replace(",", " ").split()
    if len(parts) != 2:
        raise SpatialValidationError(f"{name} must contain exactly two numbers")
    return (
        _finite(parts[0], name=f"{name} x"),
        _finite(parts[1], name=f"{name} y"),
    )


def _normalize_degrees(value: float) -> float:
    normalized = (value + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 and value > 0 else normalized


@dataclass(frozen=True)
class SpatialFinding:
    """One explicit compatibility decision made by the spatial planner."""

    code: str
    construct: str
    outcome: FindingOutcome
    detail: str

    def manifest(self) -> dict[str, str]:
        return {
            "code": self.code,
            "construct": self.construct,
            "outcome": self.outcome,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DisplayConform:
    """Build a square-pixel display raster, then apply display rotation.

    Final Cut's ``paspH/paspV`` describes the width of one encoded pixel. The
    portable graph turns that metadata into real pixels before geometry:
    encoded width ``W`` becomes ``round_half_up(W * paspH / paspV)`` while
    height stays unchanged. A 90/270-degree rotation then swaps the finished
    square-pixel raster dimensions.

    Why this exists:
    Crop and conform deliberately end in ``setsar=1``. Leaving pixel aspect as
    metadata would let those later stages silently squash or stretch it.
    """

    rotation_degrees: int = 0
    pixel_aspect_h: int = 1
    pixel_aspect_v: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.rotation_degrees, bool) or not isinstance(
            self.rotation_degrees, int
        ):
            raise SpatialValidationError("display rotation must be an integer")
        normalized = self.rotation_degrees % 360
        if normalized not in {0, 90, 180, 270}:
            raise SpatialValidationError(
                "display rotation must be a multiple of 90 degrees"
            )
        object.__setattr__(self, "rotation_degrees", normalized)
        horizontal = _positive_int(
            self.pixel_aspect_h, name="pixel_aspect_h", maximum=32_767
        )
        vertical = _positive_int(
            self.pixel_aspect_v, name="pixel_aspect_v", maximum=32_767
        )
        divisor = math.gcd(horizontal, vertical)
        object.__setattr__(self, "pixel_aspect_h", horizontal // divisor)
        object.__setattr__(self, "pixel_aspect_v", vertical // divisor)

    def filters(
        self,
        *,
        frame_width: int,
        frame_height: int,
    ) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
        """Return filters and their square-pixel output dimensions.

        Main callers:
        - ``build_spatial_execution_plan``, before color, stereo, and geometry.

        Positive half-up rounding is explicit so Python and FFmpeg never pick
        different raster widths at an exact half-pixel boundary.
        """

        width = _positive_int(
            frame_width, name="frame_width", maximum=_MAX_FRAME_DIMENSION
        )
        height = _positive_int(
            frame_height, name="frame_height", maximum=_MAX_FRAME_DIMENSION
        )
        filters: list[str] = []
        required: list[str] = []
        if (self.pixel_aspect_h, self.pixel_aspect_v) != (1, 1):
            numerator = width * self.pixel_aspect_h
            display_width = (2 * numerator + self.pixel_aspect_v) // (
                2 * self.pixel_aspect_v
            )
            if display_width < 1:
                raise SpatialValidationError(
                    "pixel-aspect display width rounds below one pixel"
                )
            if display_width > _MAX_FRAME_DIMENSION:
                raise SpatialValidationError(
                    "pixel-aspect display width must be at most "
                    f"{_MAX_FRAME_DIMENSION}, got {display_width}"
                )
            filters.extend(
                (f"scale={display_width}:{height}:flags=lanczos", "setsar=1")
            )
            required.extend(("scale", "setsar"))
            width = display_width

        if self.rotation_degrees == 90:
            filters.append("transpose=clock")
            required.append("transpose")
            width, height = height, width
        elif self.rotation_degrees == 180:
            filters.extend(("hflip", "vflip"))
            required.extend(("hflip", "vflip"))
        elif self.rotation_degrees == 270:
            filters.append("transpose=cclock")
            required.append("transpose")
            width, height = height, width
        return tuple(filters), tuple(required), width, height


_PROJECTION_FILTER_NAMES: Mapping[Projection, str] = {
    "none": "flat",
    "equirectangular": "e",
    "fisheye": "fisheye",
    "back-to-back fisheye": "dfisheye",
    "cubic": "c3x2",
}


def _projection(raw: str) -> Projection:
    compact = raw.strip().casefold()
    aliases = {
        "none": "none",
        "flat": "none",
        "equirectangular": "equirectangular",
        "fisheye": "fisheye",
        "back-to-back fisheye": "back-to-back fisheye",
        "back to back fisheye": "back-to-back fisheye",
        "cubic": "cubic",
    }
    try:
        return aliases[compact]  # type: ignore[return-value]
    except KeyError as exc:
        raise SpatialValidationError(f"unsupported 360 projection {raw!r}") from exc


@dataclass(frozen=True)
class Reorientation360:
    """Final Cut ``adjust-reorient`` mapped to a stock ``v360`` rotation."""

    input_projection: Projection
    enabled: bool = True
    tilt: float = 0.0
    pan: float = 0.0
    roll: float = 0.0
    convergence: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_projection", _projection(self.input_projection))
        for name in ("tilt", "pan", "roll"):
            value = _finite(
                getattr(self, name), name=f"360 reorientation {name}", minimum=-36_000, maximum=36_000
            )
            object.__setattr__(self, name, _normalize_degrees(value))
        object.__setattr__(
            self,
            "convergence",
            _finite(self.convergence, name="360 reorientation convergence", minimum=-100, maximum=100),
        )

    @classmethod
    def from_attributes(
        cls, attributes: Mapping[str, str], *, input_projection: str
    ) -> "Reorientation360":
        return cls(
            input_projection=_projection(input_projection),
            enabled=_parse_bool(
                attributes.get("enabled"), name="adjust-reorient enabled", default=True
            ),
            tilt=_parse_float(attributes, "tilt"),
            pan=_parse_float(attributes, "pan"),
            roll=_parse_float(attributes, "roll"),
            convergence=_parse_float(
                attributes, "convergence", minimum=-100, maximum=100
            ),
        )

    @property
    def active(self) -> bool:
        return self.enabled and any(
            abs(value) > 1e-12
            for value in (self.tilt, self.pan, self.roll, self.convergence)
        )

    @property
    def rotation_active(self) -> bool:
        return self.enabled and any(
            abs(value) > 1e-12 for value in (self.tilt, self.pan, self.roll)
        )


@dataclass(frozen=True)
class Transform360:
    """Preserve Final Cut's 3D content transform without inventing a camera map.

    ``adjust-360-transform`` is distinct from viewer reorientation.  Its
    spherical/cartesian position, automatic facing, interaxial distance, and
    stereo convergence jointly change how content sits inside the sphere.
    Stock ``v360`` only exposes a viewer projection.  Until Final Cut oracle
    evidence defines that joint mapping, active values remain a typed finding.
    """

    coordinates: Literal["spherical", "cartesian"]
    enabled: bool = True
    latitude: float = 0.0
    longitude: float = 0.0
    distance: float | None = None
    x_position: float = 0.0
    y_position: float = 0.0
    z_position: float | None = None
    x_orientation: float = 0.0
    y_orientation: float = 0.0
    z_orientation: float = 0.0
    auto_orient: bool = True
    convergence: float = 0.0
    interaxial: float | None = None
    scale: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if self.coordinates not in {"spherical", "cartesian"}:
            raise SpatialValidationError(
                f"adjust-360-transform coordinates are invalid: {self.coordinates!r}"
            )
        for name in (
            "latitude",
            "longitude",
            "x_position",
            "y_position",
            "x_orientation",
            "y_orientation",
            "z_orientation",
            "convergence",
        ):
            object.__setattr__(
                self,
                name,
                _finite(
                    getattr(self, name),
                    name=f"adjust-360-transform {name}",
                    minimum=-100_000,
                    maximum=100_000,
                ),
            )
        for name in ("distance", "z_position", "interaxial"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite(
                        value,
                        name=f"adjust-360-transform {name}",
                        minimum=-100_000,
                        maximum=100_000,
                    ),
                )
        if not isinstance(self.scale, tuple) or len(self.scale) != 2:
            raise SpatialValidationError("adjust-360-transform scale must contain two numbers")
        object.__setattr__(
            self,
            "scale",
            (
                _finite(self.scale[0], name="adjust-360-transform scale x"),
                _finite(self.scale[1], name="adjust-360-transform scale y"),
            ),
        )

    @classmethod
    def from_attributes(cls, attributes: Mapping[str, str]) -> "Transform360":
        coordinates = attributes.get("coordinates")
        if coordinates is None:
            raise SpatialValidationError(
                "adjust-360-transform is missing DTD-required coordinates"
            )
        return cls(
            coordinates=coordinates,  # type: ignore[arg-type]
            enabled=_parse_bool(
                attributes.get("enabled"),
                name="adjust-360-transform enabled",
                default=True,
            ),
            latitude=_parse_float(attributes, "latitude"),
            longitude=_parse_float(attributes, "longitude"),
            distance=(
                _parse_float(attributes, "distance")
                if "distance" in attributes
                else None
            ),
            x_position=_parse_float(attributes, "xPosition"),
            y_position=_parse_float(attributes, "yPosition"),
            z_position=(
                _parse_float(attributes, "zPosition")
                if "zPosition" in attributes
                else None
            ),
            x_orientation=_parse_float(attributes, "xOrientation"),
            y_orientation=_parse_float(attributes, "yOrientation"),
            z_orientation=_parse_float(attributes, "zOrientation"),
            auto_orient=_parse_bool(
                attributes.get("autoOrient"),
                name="adjust-360-transform autoOrient",
                default=True,
            ),
            convergence=_parse_float(attributes, "convergence"),
            interaxial=(
                _parse_float(attributes, "interaxial")
                if "interaxial" in attributes
                else None
            ),
            scale=_parse_pair(
                attributes.get("scale"),
                name="adjust-360-transform scale",
                default=(1.0, 1.0),
            ),
        )

    @property
    def active(self) -> bool:
        if not self.enabled:
            return False
        scalars = (
            self.latitude,
            self.longitude,
            self.distance or 0.0,
            self.x_position,
            self.y_position,
            self.z_position or 0.0,
            self.x_orientation,
            self.y_orientation,
            self.z_orientation,
            self.convergence,
            self.interaxial or 0.0,
        )
        return any(abs(value) > 1e-12 for value in scalars) or self.scale != (1.0, 1.0)

    def finding(self) -> SpatialFinding | None:
        if not self.active:
            return None
        return SpatialFinding(
            code="spatial.360_content_transform_unavailable",
            construct="adjust-360-transform",
            outcome="not_implemented_yet",
            detail=(
                f"active {self.coordinates} 360 content transform is preserved; "
                "viewer-only v360 rotation would not reproduce its 3D/stereo semantics"
            ),
        )


@dataclass(frozen=True)
class Orientation360:
    """Final Cut 360 viewer orientation, including Tiny Planet projection."""

    input_projection: Projection
    mapping: OrientationMapping = "normal"
    enabled: bool = True
    tilt: float = 0.0
    pan: float = 0.0
    roll: float = 0.0
    field_of_view: float | None = None
    output_width: int | None = None
    output_height: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_projection", _projection(self.input_projection))
        if self.mapping not in {"normal", "tinyPlanet"}:
            raise SpatialValidationError(f"unsupported 360 mapping {self.mapping!r}")
        for name in ("tilt", "pan", "roll"):
            value = _finite(
                getattr(self, name), name=f"360 orientation {name}", minimum=-36_000, maximum=36_000
            )
            object.__setattr__(self, name, _normalize_degrees(value))
        if self.field_of_view is not None:
            object.__setattr__(
                self,
                "field_of_view",
                _finite(
                    self.field_of_view,
                    name="360 field_of_view",
                    minimum=1,
                    maximum=179.9,
                ),
            )
        for name in ("output_width", "output_height"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _positive_int(value, name=name, maximum=_MAX_FRAME_DIMENSION),
                )
        if (self.output_width is None) != (self.output_height is None):
            raise SpatialValidationError(
                "360 output_width and output_height must be provided together"
            )
        if self.mapping == "tinyPlanet" and self.input_projection == "none":
            raise SpatialValidationError(
                "Tiny Planet requires spherical source projection metadata"
            )

    @classmethod
    def from_attributes(
        cls,
        attributes: Mapping[str, str],
        *,
        input_projection: str,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> "Orientation360":
        mapping = attributes.get("mapping", "normal")
        field_of_view = attributes.get("fieldOfView")
        return cls(
            input_projection=_projection(input_projection),
            mapping=mapping,  # type: ignore[arg-type]
            enabled=_parse_bool(
                attributes.get("enabled"), name="adjust-orientation enabled", default=True
            ),
            tilt=_parse_float(attributes, "tilt"),
            pan=_parse_float(attributes, "pan"),
            roll=_parse_float(attributes, "roll"),
            field_of_view=(
                _finite(field_of_view, name="fieldOfView", minimum=1, maximum=179.9)
                if field_of_view is not None
                else None
            ),
            output_width=output_width,
            output_height=output_height,
        )

    @property
    def active(self) -> bool:
        if not self.enabled:
            return False
        if self.input_projection != "none":
            return True
        return self.mapping == "tinyPlanet" or any(
            abs(value) > 1e-12 for value in (self.tilt, self.pan, self.roll)
        )


def _v360_filter(
    *,
    input_projection: Projection,
    output_projection: str,
    pan: float,
    tilt: float,
    roll: float,
    field_of_view: float | None,
    width: int,
    height: int,
) -> str:
    options = [
        f"input={_PROJECTION_FILTER_NAMES[input_projection]}",
        f"output={output_projection}",
        "interp=lanczos",
        f"yaw={_number(pan)}",
        f"pitch={_number(tilt)}",
        f"roll={_number(roll)}",
        f"w={width}",
        f"h={height}",
    ]
    if field_of_view is not None:
        options.append(f"h_fov={_number(field_of_view)}")
    return "v360=" + ":".join(options)


_STEREO_INPUT_CODES: Mapping[tuple[StereoLayout, HeroEye], str] = {
    ("side by side", "left"): "sbs2l",
    ("side by side", "right"): "sbs2r",
    ("over under", "left"): "ab2l",
    ("over under", "right"): "ab2r",
}


@dataclass(frozen=True)
class Stereo3DAdjustment:
    """Layout, eye-order, and bounded convergence for packed stereo media.

    ``convergence`` is Final Cut's exported value, bounded to ``[-100, 100]``.
    Until an oracle calibration exists, ten percent of one eye width is the
    maximum shift.  The plan therefore reports convergence as a semantic
    approximation while still producing the intended direction and magnitude.
    """

    input_layout: StereoLayout
    hero_eye: HeroEye = "left"
    output_layout: StereoLayout | None = None
    enabled: bool = True
    convergence: float = 0.0
    auto_scale: bool = True
    swap_eyes: bool = False
    depth: float = 0.0

    def __post_init__(self) -> None:
        if self.input_layout not in {"mono", "side by side", "over under"}:
            raise SpatialValidationError(
                f"unsupported stereo input layout {self.input_layout!r}"
            )
        output = self.input_layout if self.output_layout is None else self.output_layout
        if output not in {"mono", "side by side", "over under"}:
            raise SpatialValidationError(f"unsupported stereo output layout {output!r}")
        object.__setattr__(self, "output_layout", output)
        if self.hero_eye not in {"left", "right"}:
            raise SpatialValidationError(f"unsupported hero eye {self.hero_eye!r}")
        object.__setattr__(
            self,
            "convergence",
            _finite(self.convergence, name="stereo convergence", minimum=-100, maximum=100),
        )
        object.__setattr__(
            self, "depth", _finite(self.depth, name="stereo depth", minimum=-100, maximum=100)
        )

    @classmethod
    def from_attributes(
        cls,
        attributes: Mapping[str, str],
        *,
        input_layout: str,
        hero_eye: str | None,
    ) -> "Stereo3DAdjustment":
        return cls(
            input_layout=input_layout,  # type: ignore[arg-type]
            hero_eye=(hero_eye or "left"),  # type: ignore[arg-type]
            enabled=_parse_bool(
                attributes.get("enabled"), name="adjust-stereo-3D enabled", default=True
            ),
            convergence=_parse_float(
                attributes, "convergence", minimum=-100, maximum=100
            ),
            auto_scale=_parse_bool(
                attributes.get("autoScale"), name="adjust-stereo-3D autoScale", default=True
            ),
            swap_eyes=_parse_bool(
                attributes.get("swapEyes"), name="adjust-stereo-3D swapEyes", default=False
            ),
            depth=_parse_float(attributes, "depth", minimum=-100, maximum=100),
        )


_SDR_COLORSPACE_NAMES = {
    "rec601_ntsc": "smpte170m",
    "rec601_pal": "bt470bg",
    "rec709": "bt709",
    "rec2020": "bt2020",
}


def classify_fcp_color_space(raw: str | None) -> str | None:
    """Classify Final Cut's numeric/name color-space string without guessing."""

    if raw is None or not raw.strip():
        return None
    compact = raw.casefold()
    numeric = raw.strip().split(maxsplit=1)[0]
    if "hlg" in compact or numeric == "9-18-9":
        return "rec2020_hlg"
    if "pq" in compact or numeric == "9-16-9":
        return "rec2020_pq"
    if "rec. 2020" in compact or numeric in {"9-1-9", "9-14-9"}:
        return "rec2020"
    if "601 (ntsc)" in compact or numeric in {"6-1-6", "6-6-6"}:
        return "rec601_ntsc"
    if "601 (pal)" in compact or numeric in {"5-1-6", "5-6-6"}:
        return "rec601_pal"
    if "rec. 709" in compact or numeric == "1-1-1":
        return "rec709"
    return None


@dataclass(frozen=True)
class ColorConform:
    """One Final Cut color-conform request targeting ordinary Rec.709 SDR."""

    source_color_space: str | None
    mode: ColorConformMode = "conformAuto"
    enabled: bool = True
    peak_nits_of_pq_source: float = 1_000.0
    peak_nits_of_sdr_to_pq_source: float = 100.0

    def __post_init__(self) -> None:
        if self.mode not in {
            "conformNone",
            "conformAuto",
            "conformHLGtoSDR",
            "conformPQtoSDR",
            "conformHLGtoPQ",
            "conformPQtoHLG",
            "conformSDRtoHLG75",
            "conformSDRtoHLG100",
            "conformSDRtoPQ",
        }:
            raise SpatialValidationError(f"unsupported color conform mode {self.mode!r}")
        object.__setattr__(
            self,
            "peak_nits_of_pq_source",
            _finite(
                self.peak_nits_of_pq_source,
                name="peakNitsOfPQSource",
                minimum=100,
                maximum=10_000,
            ),
        )
        object.__setattr__(
            self,
            "peak_nits_of_sdr_to_pq_source",
            _finite(
                self.peak_nits_of_sdr_to_pq_source,
                name="peakNitsOfSDRToPQSource",
                minimum=48,
                maximum=1_000,
            ),
        )

    @classmethod
    def from_attributes(
        cls, attributes: Mapping[str, str], *, source_color_space: str | None
    ) -> "ColorConform":
        missing = tuple(
            name
            for name in ("peakNitsOfPQSource", "peakNitsOfSDRToPQSource")
            if name not in attributes
        )
        if missing:
            raise SpatialValidationError(
                "adjust-colorConform is missing DTD-required attribute(s): "
                + ", ".join(missing)
            )
        return cls(
            source_color_space=source_color_space,
            mode=attributes.get("conformType", "conformNone"),  # type: ignore[arg-type]
            enabled=_parse_bool(
                attributes.get("enabled"), name="adjust-colorConform enabled", default=True
            ),
            peak_nits_of_pq_source=_parse_float(
                attributes,
                "peakNitsOfPQSource",
                minimum=100,
                maximum=10_000,
            ),
            peak_nits_of_sdr_to_pq_source=_parse_float(
                attributes,
                "peakNitsOfSDRToPQSource",
                minimum=48,
                maximum=1_000,
            ),
        )

    def resolved_mode(self) -> ColorConformMode:
        if not self.enabled or self.mode == "conformNone":
            return "conformNone"
        if self.mode != "conformAuto":
            return self.mode
        classified = classify_fcp_color_space(self.source_color_space)
        if classified == "rec2020_hlg":
            return "conformHLGtoSDR"
        if classified == "rec2020_pq":
            return "conformPQtoSDR"
        return "conformAuto"


@dataclass(frozen=True)
class Stabilization:
    """Bounded semantic approximation of Final Cut stabilization modes."""

    mode: StabilizationMode = "automatic"
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"automatic", "inertiaCam", "smoothCam"}:
            raise SpatialValidationError(f"unsupported stabilization mode {self.mode!r}")

    @classmethod
    def from_attributes(cls, attributes: Mapping[str, str]) -> "Stabilization":
        return cls(
            mode=attributes.get("type", "automatic"),  # type: ignore[arg-type]
            enabled=_parse_bool(
                attributes.get("enabled"), name="adjust-stabilization enabled", default=True
            ),
        )


_DESHAKE_PRESETS: Mapping[StabilizationMode, str] = {
    "automatic": "deshake=rx=16:ry=16:blocksize=8:contrast=125:search=less:edge=mirror",
    "inertiaCam": "deshake=rx=32:ry=32:blocksize=16:contrast=100:search=exhaustive:edge=mirror",
    # FFmpeg requires rx/ry to be multiples of 16. A wider, low-search window
    # preserves SmoothCam's gentler semantic profile without emitting an
    # invalid graph.
    "smoothCam": "deshake=rx=48:ry=48:blocksize=8:contrast=100:search=less:edge=mirror",
}


@dataclass(frozen=True)
class RollingShutterAdjustment:
    """An explicitly unsupported active rolling-shutter control."""

    amount: str = "none"
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.amount not in {"none", "low", "medium", "high", "extraHigh"}:
            raise SpatialValidationError(f"unsupported rolling-shutter amount {self.amount!r}")

    @classmethod
    def from_attributes(cls, attributes: Mapping[str, str]) -> "RollingShutterAdjustment":
        return cls(
            amount=attributes.get("amount", "none"),
            enabled=_parse_bool(
                attributes.get("enabled"), name="adjust-rollingShutter enabled", default=True
            ),
        )

    def finding(self) -> SpatialFinding | None:
        if not self.enabled or self.amount == "none":
            return None
        return SpatialFinding(
            code="spatial.rolling_shutter_unavailable",
            construct="adjust-rollingShutter",
            outcome="not_implemented_yet",
            detail=(
                f"rolling-shutter correction amount {self.amount!r} has no defensible "
                "stock-FFmpeg equivalent"
            ),
        )


@dataclass(frozen=True)
class OpaqueCinematicLocator:
    """Cinematic focus metadata that cannot be decoded from exported XML."""

    data_locator: str | None
    enabled: bool = True

    def finding(self) -> SpatialFinding | None:
        if not self.enabled:
            return None
        return SpatialFinding(
            code="spatial.cinematic_locator_opaque",
            construct="adjust-cinematic",
            outcome="not_implemented_yet",
            detail=(
                "Cinematic focus needs readable depth/focus samples; the exported "
                f"data locator {self.data_locator!r} is opaque"
            ),
        )


@dataclass(frozen=True)
class OpaqueTrackerLocator:
    """A tracker shape whose only motion data is an opaque data locator."""

    tracker_id: str
    data_locator: str | None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.tracker_id, str) or not self.tracker_id.strip():
            raise SpatialValidationError("opaque tracker requires a non-empty tracker_id")

    def finding(self) -> SpatialFinding | None:
        if not self.enabled:
            return None
        return SpatialFinding(
            code="spatial.tracker_locator_opaque",
            construct="object-tracker",
            outcome="not_implemented_yet",
            detail=(
                f"tracker {self.tracker_id!r} has no readable keyframes; "
                f"data locator {self.data_locator!r} is opaque"
            ),
        )


@dataclass(frozen=True)
class TrackerKeyframe:
    """One readable tracker sample in exact source-local time."""

    time: Fraction
    center: tuple[float, float]
    scale: tuple[float, float] = (1.0, 1.0)
    rotation: float = 0.0
    interpolation: str = "linear"
    curve: str = "smooth"

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _exact_time(self.time, name="tracker keyframe time"))
        for name in ("center", "scale"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(value) != 2:
                raise SpatialValidationError(f"tracker {name} must contain two values")
            typed = (
                _finite(value[0], name=f"tracker {name} x"),
                _finite(value[1], name=f"tracker {name} y"),
            )
            object.__setattr__(self, name, typed)
        if self.scale[0] == 0 or self.scale[1] == 0:
            raise SpatialValidationError("tracker scale cannot contain zero")
        object.__setattr__(
            self,
            "rotation",
            _finite(self.rotation, name="tracker rotation", minimum=-36_000, maximum=36_000),
        )


@dataclass(frozen=True)
class ObjectTrackerAnimationHook:
    """Exact retimed tracks ready for transform or mask animation consumers."""

    tracker_id: str
    position: TimelineAnimatedVec2
    scale: TimelineAnimatedVec2
    rotation: TimelineAnimatedScalar


def build_tracker_animation_hook(
    tracker_id: str,
    keyframes: Sequence[TrackerKeyframe],
    retime_map: RetimeMap,
) -> ObjectTrackerAnimationHook:
    """Map readable tracker samples to the shared lossless animation kernel.

    Main callers:
    - The compiler integration after a future readable tracker decoder.

    Why this exists:
    - Tracker animation should share reverse/freeze/variable-rate semantics
      with transform keyframes.  Re-sampling into guessed timeline points
      would lose repeated and reverse source occurrences.
    """

    if not isinstance(tracker_id, str) or not tracker_id.strip():
        raise SpatialValidationError("readable tracker requires a non-empty tracker_id")
    points = tuple(keyframes)
    if not points:
        raise SpatialValidationError("readable tracker requires at least one keyframe")
    if len(points) > _MAX_TRACKER_POINTS:
        raise SpatialValidationError(
            f"readable tracker exceeds {_MAX_TRACKER_POINTS} keyframes"
        )
    if not isinstance(retime_map, RetimeMap):
        raise SpatialValidationError("tracker retime_map must be RetimeMap")
    for index, point in enumerate(points):
        if not isinstance(point, TrackerKeyframe):
            raise SpatialValidationError(
                f"tracker keyframes[{index}] must be TrackerKeyframe"
            )

    positions = AnimatedVec2(
        tuple(
            Vec2ControlPoint(
                point.time,
                point.center,
                interpolation=point.interpolation,  # type: ignore[arg-type]
                curve=point.curve,  # type: ignore[arg-type]
            )
            for point in points
        )
    )
    scales = AnimatedVec2(
        tuple(
            Vec2ControlPoint(
                point.time,
                point.scale,
                interpolation=point.interpolation,  # type: ignore[arg-type]
                curve=point.curve,  # type: ignore[arg-type]
            )
            for point in points
        )
    )
    rotations = AnimatedScalar(
        tuple(
            ScalarControlPoint(
                point.time,
                point.rotation,
                interpolation=point.interpolation,  # type: ignore[arg-type]
                curve=point.curve,  # type: ignore[arg-type]
            )
            for point in points
        )
    )
    return ObjectTrackerAnimationHook(
        tracker_id=tracker_id,
        position=TimelineAnimatedVec2(positions, retime_map),
        scale=TimelineAnimatedVec2(scales, retime_map),
        rotation=TimelineAnimatedScalar(rotations, retime_map),
    )


@dataclass(frozen=True)
class SpatialIntrinsicPlan:
    """All early-video spatial intrinsics for one resolved media node."""

    frame_width: int
    frame_height: int
    display: DisplayConform | None = None
    color_conform: ColorConform | None = None
    stereo: Stereo3DAdjustment | None = None
    stabilization: Stabilization | None = None
    transform_360: Transform360 | None = None
    reorientation_360: Reorientation360 | None = None
    orientation_360: Orientation360 | None = None
    rolling_shutter: RollingShutterAdjustment | None = None
    cinematic: OpaqueCinematicLocator | None = None
    opaque_trackers: tuple[OpaqueTrackerLocator, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_width",
            _positive_int(
                self.frame_width, name="frame_width", maximum=_MAX_FRAME_DIMENSION
            ),
        )
        object.__setattr__(
            self,
            "frame_height",
            _positive_int(
                self.frame_height, name="frame_height", maximum=_MAX_FRAME_DIMENSION
            ),
        )
        object.__setattr__(self, "opaque_trackers", tuple(self.opaque_trackers))
        for index, tracker in enumerate(self.opaque_trackers):
            if not isinstance(tracker, OpaqueTrackerLocator):
                raise SpatialValidationError(
                    f"opaque_trackers[{index}] must be OpaqueTrackerLocator"
                )


@dataclass(frozen=True)
class SpatialExecutionPlan:
    """A complete stock-FFmpeg fragment plus explicit compatibility findings."""

    filter_complex: str
    input_label: str
    output_label: str
    output_width: int
    output_height: int
    required_filters: tuple[str, ...]
    findings: tuple[SpatialFinding, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "backend": "stock_ffmpeg",
            "custom_ffmpeg_required": False,
            "vulkan_required": False,
            "input_label": self.input_label,
            "output_label": self.output_label,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "required_filters": list(self.required_filters),
            "findings": [finding.manifest() for finding in self.findings],
        }

    def command(
        self,
        *,
        ffmpeg: Path,
        input_path: Path,
        output_path: Path,
        frames: int | None = None,
    ) -> tuple[str, ...]:
        """Return a shell-free FFmpeg argv for isolated executable validation."""

        argv = [
            str(Path(ffmpeg)),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(Path(input_path)),
            "-filter_complex",
            f"[0:v]null[{self.input_label}];{self.filter_complex}",
            "-map",
            f"[{self.output_label}]",
        ]
        if frames is not None:
            count = _positive_int(frames, name="frames", maximum=10_000_000)
            argv.extend(("-frames:v", str(count)))
        argv.extend(("-an", str(Path(output_path))))
        return tuple(argv)


class _GraphBuilder:
    def __init__(self, input_label: str) -> None:
        self.current = input_label
        self.lines: list[str] = []
        self.index = 0

    def label(self, stem: str) -> str:
        result = f"spatial_{stem}_{self.index}"
        self.index += 1
        return result

    def chain(self, filters: Sequence[str], *, stem: str) -> None:
        typed = tuple(filters)
        if not typed:
            return
        output = self.label(stem)
        self.lines.append(f"[{self.current}]" + ",".join(typed) + f"[{output}]")
        self.current = output


def _validate_label(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _FILTER_LABEL.fullmatch(value):
        raise SpatialValidationError(
            f"{name} must contain only letters, digits, and underscores"
        )
    return value


def _append_stereo(
    builder: _GraphBuilder,
    adjustment: Stereo3DAdjustment,
    *,
    width: int,
    height: int,
    required: set[str],
    findings: list[SpatialFinding],
) -> None:
    if not adjustment.enabled:
        return
    if adjustment.input_layout == "mono":
        if adjustment.swap_eyes or adjustment.convergence or adjustment.depth:
            findings.append(
                SpatialFinding(
                    code="spatial.stereo_controls_on_mono",
                    construct="adjust-stereo-3D",
                    outcome="not_implemented_yet",
                    detail="stereo eye controls cannot operate on a mono source",
                )
            )
        return
    assert adjustment.output_layout is not None
    if adjustment.output_layout == "mono":
        output_code = "mr" if adjustment.swap_eyes else "ml"
    else:
        output_eye: HeroEye = (
            "right" if adjustment.hero_eye == "left" else "left"
        ) if adjustment.swap_eyes else adjustment.hero_eye
        output_code = _STEREO_INPUT_CODES[(adjustment.output_layout, output_eye)]
    input_code = _STEREO_INPUT_CODES[(adjustment.input_layout, adjustment.hero_eye)]

    if adjustment.depth:
        findings.append(
            SpatialFinding(
                code="spatial.stereo_depth_unavailable",
                construct="adjust-stereo-3D@depth",
                outcome="not_implemented_yet",
                detail=(
                    f"stereo depth {adjustment.depth:g} is preserved; its Final Cut "
                    "cross-channel semantics are not calibrated"
                ),
            )
        )

    convergence = adjustment.convergence
    if abs(convergence) < 1e-12:
        if input_code != output_code:
            builder.chain(
                (f"stereo3d=in={input_code}:out={output_code}",), stem="stereo_layout"
            )
            required.add("stereo3d")
        return
    if width % 2:
        raise SpatialValidationError(
            "packed stereo convergence requires an even frame width"
        )

    eye_width = width // 2
    shift = max(1, round(abs(convergence) / 100.0 * eye_width * 0.10))
    shift = min(shift, max(1, eye_width // 8))
    normalized = builder.label("stereo_normalized")
    builder.lines.append(
        f"[{builder.current}]stereo3d=in={input_code}:out=sbsl[{normalized}]"
    )
    left_source = builder.label("stereo_left_source")
    right_source = builder.label("stereo_right_source")
    builder.lines.append(
        f"[{normalized}]split=2[{left_source}][{right_source}]"
    )
    left_eye = builder.label("stereo_left_eye")
    right_eye = builder.label("stereo_right_eye")
    builder.lines.append(
        f"[{left_source}]crop=w={eye_width}:h={height}:x=0:y=0[{left_eye}]"
    )
    builder.lines.append(
        f"[{right_source}]crop=w={eye_width}:h={height}:x={eye_width}:y=0[{right_eye}]"
    )

    positive = convergence > 0
    left_shifted = builder.label("stereo_left_shifted")
    right_shifted = builder.label("stereo_right_shifted")
    if adjustment.auto_scale:
        scale_width = eye_width + 2 * shift
        scale_height = max(height, round(height * scale_width / eye_width))
        if scale_height % 2:
            scale_height += 1
        vertical = (scale_height - height) // 2
        left_x = 0 if positive else 2 * shift
        right_x = 2 * shift if positive else 0
        builder.lines.append(
            f"[{left_eye}]scale=w={scale_width}:h={scale_height},"
            f"crop=w={eye_width}:h={height}:x={left_x}:y={vertical}[{left_shifted}]"
        )
        builder.lines.append(
            f"[{right_eye}]scale=w={scale_width}:h={scale_height},"
            f"crop=w={eye_width}:h={height}:x={right_x}:y={vertical}[{right_shifted}]"
        )
        required.add("scale")
    else:
        remaining = eye_width - shift
        left_crop_x = 0 if positive else shift
        right_crop_x = shift if positive else 0
        left_pad_x = shift if positive else 0
        right_pad_x = 0 if positive else shift
        builder.lines.append(
            f"[{left_eye}]crop=w={remaining}:h={height}:x={left_crop_x}:y=0,"
            f"pad=w={eye_width}:h={height}:x={left_pad_x}:y=0:color=black[{left_shifted}]"
        )
        builder.lines.append(
            f"[{right_eye}]crop=w={remaining}:h={height}:x={right_crop_x}:y=0,"
            f"pad=w={eye_width}:h={height}:x={right_pad_x}:y=0:color=black[{right_shifted}]"
        )
        required.add("pad")
    shifted = builder.label("stereo_shifted")
    builder.lines.append(
        f"[{left_shifted}][{right_shifted}]hstack=inputs=2[{shifted}]"
    )
    output = builder.label("stereo_output")
    builder.lines.append(
        f"[{shifted}]stereo3d=in=sbsl:out={output_code}[{output}]"
    )
    builder.current = output
    required.update({"stereo3d", "split", "crop", "hstack"})
    findings.append(
        SpatialFinding(
            code="spatial.stereo_convergence_approximation",
            construct="adjust-stereo-3D@convergence",
            outcome="approximated",
            detail=(
                f"convergence {convergence:g} uses a bounded {shift}-pixel opposite-eye "
                f"shift; autoScale={int(adjustment.auto_scale)}"
            ),
        )
    )


def _append_color(
    builder: _GraphBuilder,
    conform: ColorConform,
    *,
    required: set[str],
    findings: list[SpatialFinding],
) -> None:
    mode = conform.resolved_mode()
    source = classify_fcp_color_space(conform.source_color_space)
    if mode == "conformNone":
        return
    if mode == "conformAuto":
        if source is None:
            findings.append(
                SpatialFinding(
                    code="spatial.color_space_unknown",
                    construct="adjust-colorConform",
                    outcome="not_implemented_yet",
                    detail=(
                        f"automatic color conform cannot classify source color space "
                        f"{conform.source_color_space!r}"
                    ),
                )
            )
            return
        if source == "rec709":
            return
        if source in _SDR_COLORSPACE_NAMES:
            builder.chain(
                (
                    "colorspace="
                    f"iall={_SDR_COLORSPACE_NAMES[source]}:all=bt709:"
                    "fast=0:dither=fsb:format=yuv420p",
                ),
                stem="sdr_color_conform",
            )
            required.add("colorspace")
            return
        raise AssertionError(f"unreachable classified color space {source!r}")
    if mode in _HDR_LUTS:
        lut_path = _HDR_LUTS[mode]
        if not lut_path.is_file():
            raise SpatialValidationError(
                f"renderer-owned HDR conform LUT is missing: {lut_path.name}"
            )
        builder.chain(
            (
                "format=gbrpf32le",
                f"lut3d=file='{lut_path.as_posix()}':interp=tetrahedral",
                "format=yuv420p",
                "setparams=range=limited:color_primaries=bt709:"
                "color_trc=bt709:colorspace=bt709",
            ),
            stem="hdr_color_conform",
        )
        required.update({"format", "lut3d", "setparams"})
        detail = (
            "renderer-owned Rec.2020 HLG-to-Rec.709 SDR LUT"
            if mode == "conformHLGtoSDR"
            else "renderer-owned Rec.2020 PQ-to-Rec.709 SDR LUT calibrated for a 1000-nit source"
        )
        if mode == "conformPQtoSDR" and abs(conform.peak_nits_of_pq_source - 1_000) > 1e-9:
            detail += (
                f"; exported peak {conform.peak_nits_of_pq_source:g} nits is preserved "
                "but the frozen LUT remains a semantic approximation"
            )
        findings.append(
            SpatialFinding(
                code="spatial.hdr_to_sdr_lut_approximation",
                construct="adjust-colorConform",
                outcome="approximated",
                detail=detail,
            )
        )
        return
    findings.append(
        SpatialFinding(
            code="spatial.color_conform_direction_unavailable",
            construct="adjust-colorConform",
            outcome="not_implemented_yet",
            detail=(
                f"color conform mode {mode!r} targets HDR or another HDR transfer; "
                "only SDR output and HLG/PQ-to-SDR are implemented"
            ),
        )
    )


def build_spatial_execution_plan(
    plan: SpatialIntrinsicPlan,
    *,
    input_label: str = "spatial_in",
    output_label: str = "spatial_out",
) -> SpatialExecutionPlan:
    """Compile one spatial plan into a deterministic stock-FFmpeg graph.

    Main callers:
    - Experimental core tests and the future central FFmpeg integration.

    Why this exists:
    - A label-aware graph is needed because stereo convergence branches into
      two eye streams.  Returning a comma-only filter chain would make that
      operation impossible to compose safely.
    """

    if not isinstance(plan, SpatialIntrinsicPlan):
        raise SpatialValidationError("plan must be SpatialIntrinsicPlan")
    input_label = _validate_label(input_label, name="input_label")
    output_label = _validate_label(output_label, name="output_label")
    if input_label == output_label:
        raise SpatialValidationError("input_label and output_label must differ")

    builder = _GraphBuilder(input_label)
    required: set[str] = set()
    findings: list[SpatialFinding] = []
    width = plan.frame_width
    height = plan.frame_height

    if plan.display is not None:
        filters, display_required, width, height = plan.display.filters(
            frame_width=width,
            frame_height=height,
        )
        builder.chain(filters, stem="display")
        required.update(display_required)

    if plan.color_conform is not None:
        _append_color(
            builder, plan.color_conform, required=required, findings=findings
        )

    if plan.stereo is not None:
        _append_stereo(
            builder,
            plan.stereo,
            width=width,
            height=height,
            required=required,
            findings=findings,
        )
        if (
            plan.stereo.enabled
            and plan.stereo.input_layout != "mono"
            and plan.stereo.output_layout == "mono"
        ):
            if plan.stereo.input_layout == "side by side":
                width //= 2
            else:
                height //= 2

    if plan.stabilization is not None and plan.stabilization.enabled:
        builder.chain((_DESHAKE_PRESETS[plan.stabilization.mode],), stem="stabilize")
        required.add("deshake")
        findings.append(
            SpatialFinding(
                code="spatial.stabilization_semantic_approximation",
                construct="adjust-stabilization",
                outcome="approximated",
                detail=(
                    f"Final Cut {plan.stabilization.mode!r} stabilization uses a "
                    "bounded renderer-owned deshake preset"
                ),
            )
        )

    if plan.transform_360 is not None:
        finding = plan.transform_360.finding()
        if finding is not None:
            findings.append(finding)

    if plan.reorientation_360 is not None and plan.reorientation_360.active:
        reorientation = plan.reorientation_360
        if reorientation.rotation_active and reorientation.input_projection == "none":
            raise SpatialValidationError(
                "active 360 reorientation requires spherical projection metadata"
            )
        if reorientation.rotation_active:
            builder.chain(
                (
                    _v360_filter(
                        input_projection=reorientation.input_projection,
                        output_projection=_PROJECTION_FILTER_NAMES[
                            reorientation.input_projection
                        ],
                        pan=reorientation.pan,
                        tilt=reorientation.tilt,
                        roll=reorientation.roll,
                        field_of_view=None,
                        width=width,
                        height=height,
                    ),
                ),
                stem="reorient_360",
            )
            required.add("v360")
            findings.append(
                SpatialFinding(
                    code="spatial.360_axis_calibration_pending",
                    construct="adjust-reorient",
                    outcome="approximated",
                    detail="pan/tilt/roll execute through v360; Final Cut axis calibration remains pending",
                )
            )
        if reorientation.convergence:
            findings.append(
                SpatialFinding(
                    code="spatial.360_convergence_unavailable",
                    construct="adjust-reorient@convergence",
                    outcome="not_implemented_yet",
                    detail=(
                        f"360 convergence {reorientation.convergence:g} is preserved; "
                        "v360 has no calibrated Final Cut convergence mapping"
                    ),
                )
            )

    if plan.orientation_360 is not None and plan.orientation_360.active:
        orientation = plan.orientation_360
        output_projection = "flat" if orientation.mapping == "normal" else "sg"
        output_width = orientation.output_width or width
        output_height = orientation.output_height or height
        builder.chain(
            (
                _v360_filter(
                    input_projection=orientation.input_projection,
                    output_projection=output_projection,
                    pan=orientation.pan,
                    tilt=orientation.tilt,
                    roll=orientation.roll,
                    field_of_view=orientation.field_of_view,
                    width=output_width,
                    height=output_height,
                ),
            ),
            stem="orientation_360",
        )
        width, height = output_width, output_height
        required.add("v360")
        findings.append(
            SpatialFinding(
                code=(
                    "spatial.tiny_planet_semantic_approximation"
                    if orientation.mapping == "tinyPlanet"
                    else "spatial.360_view_semantic_approximation"
                ),
                construct="adjust-orientation",
                outcome="approximated",
                detail=(
                    "Tiny Planet uses v360 stereographic projection"
                    if orientation.mapping == "tinyPlanet"
                    else "360 viewer orientation uses v360 rectilinear projection"
                ),
            )
        )

    if plan.rolling_shutter is not None:
        finding = plan.rolling_shutter.finding()
        if finding is not None:
            findings.append(finding)
    if plan.cinematic is not None:
        finding = plan.cinematic.finding()
        if finding is not None:
            findings.append(finding)
    for tracker in plan.opaque_trackers:
        finding = tracker.finding()
        if finding is not None:
            findings.append(finding)

    builder.lines.append(f"[{builder.current}]null[{output_label}]")
    required.add("null")
    return SpatialExecutionPlan(
        filter_complex=";".join(builder.lines),
        input_label=input_label,
        output_label=output_label,
        output_width=width,
        output_height=height,
        required_filters=tuple(sorted(required)),
        findings=tuple(findings),
    )


@dataclass(frozen=True)
class FFmpegSpatialCapabilityReport:
    executable: str
    version_line: str
    required_filters: tuple[str, ...]
    available_filters: tuple[str, ...]
    missing_filters: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.missing_filters

    def require_supported(self) -> None:
        if self.missing_filters:
            raise MissingFFmpegSpatialCapability(
                "FFmpeg is missing spatial filters: " + ", ".join(self.missing_filters)
            )


def probe_stock_ffmpeg_spatial_capabilities(
    ffmpeg: Path,
    plan: SpatialExecutionPlan,
    *,
    timeout_seconds: float = 10.0,
) -> FFmpegSpatialCapabilityReport:
    """Inspect one existing FFmpeg binary for exactly the plan's filters."""

    if not isinstance(plan, SpatialExecutionPlan):
        raise SpatialValidationError("capability probe requires SpatialExecutionPlan")
    timeout = _finite(
        timeout_seconds, name="timeout_seconds", minimum=0.1, maximum=60.0
    )
    executable = str(Path(ffmpeg))
    try:
        version = subprocess.run(
            (executable, "-version"),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        filters = subprocess.run(
            (executable, "-hide_banner", "-filters"),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MissingFFmpegSpatialCapability(
            f"could not inspect FFmpeg executable {executable!r}: {exc}"
        ) from exc
    available = tuple(sorted(set(_FILTER_ROW.findall(filters.stdout + filters.stderr))))
    missing = tuple(
        name for name in plan.required_filters if name not in set(available)
    )
    version_line = (version.stdout or version.stderr).splitlines()[0]
    return FFmpegSpatialCapabilityReport(
        executable=executable,
        version_line=version_line,
        required_filters=plan.required_filters,
        available_filters=available,
        missing_filters=missing,
    )


__all__ = [
    "ColorConform",
    "DisplayConform",
    "FFmpegSpatialCapabilityReport",
    "MissingFFmpegSpatialCapability",
    "ObjectTrackerAnimationHook",
    "OpaqueCinematicLocator",
    "OpaqueTrackerLocator",
    "Orientation360",
    "Reorientation360",
    "RollingShutterAdjustment",
    "SpatialExecutionPlan",
    "SpatialFinding",
    "SpatialIntrinsicError",
    "SpatialIntrinsicPlan",
    "SpatialValidationError",
    "Stabilization",
    "Stereo3DAdjustment",
    "TrackerKeyframe",
    "Transform360",
    "build_spatial_execution_plan",
    "build_tracker_animation_hook",
    "classify_fcp_color_space",
    "probe_stock_ffmpeg_spatial_capabilities",
]
