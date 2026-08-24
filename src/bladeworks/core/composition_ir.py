"""Compile one backend-neutral FCPXML video composition plan.

Architecture map
================

``RenderDocument`` plus trusted runtime source resolution
    -> ``DecoderSourcePlan`` / ``RasterSourcePlan``
    -> crop and conform (``RasterPlacementPlan``)
    -> ordered effects and masks (``EffectStackPlan``)
    -> corner pin and affine transform (``SpatialTransformPlan``)
    -> nested ``CompositionScopePlan`` modules
    -> transition replacement and canonical stable intervals
    -> immutable ``CompositionPlan``

The CPU and Vulkan lowerers consume the same complete plan.  Backend-specific
decoder seeks, FFmpeg input numbers, generated labels, shader artifacts,
working-format fusion, uploads, and downloads are separate execution plans.

Important invariants
--------------------

* Semantic source ownership uses validated ``VideoFrameOwnership`` and
  ``VideoDecodeWindow`` values.  Pre-seeked and unseeked execution bindings
  therefore describe the same semantic source.
* Every canvas lifetime is one exact midpoint-owned ``OwnedFrameWindow``.
  Backends never reconstruct active layers from floating-point timestamps.
* Each nested scope is a proper semantic module and appears once in its
  parent's stack.  This does not force the module to leave GPU memory.
* Transition intervals replace their complete recursively composed sides;
  participant layers cannot also remain in the ordinary stack.
* Geometry preserves the Final Cut stage boundary:
  crop/conform -> effects/masks -> corner/affine -> opacity/blend.
* Crop/Pan sampling carries resolved source alpha, support, camera quads,
  interpolation, border, clipping, and the complete output-frame schedule.
* Pixel contracts declare surface origin, transfer, range, alpha association,
  precision, and opaque/transparent coverage before choosing a backend.
* Fusion legality is derived from contracts and boundaries.  It is never a
  caller-provided assertion.

Why this exists
---------------

The initial Vulkan prototype independently rediscovered source windows,
retimes, groups, active layers, z-order, transition participants, geometry,
and alpha behavior.  That made it a second semantic compiler.  This module is
the single semantic handoff so CPU and Vulkan differ only in pixel execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias

from .animation import TimelineAnimatedScalar
from .compositor import BlendModeSpec, CompositorWindow, OpacityPlan, resolve_blend_mode
from .geometry import (
    TRANSPARENT_PERSPECTIVE_BORDER,
    CornerPinAdjustment,
    RenderSurface,
    correct_quad_for_pixel_centers,
    expand_quad_for_transparent_border,
)
from .model import (
    CropAdjustment,
    FadeEnvelope,
    Parameter,
    RenderTransformAnimation,
    TransformAdjustment,
    VideoDecoderOrigin,
)
from .pixel_domains import FrameClock, PixelDomain
from .retime_execution import (
    OwnedFrameWindow,
    RetimeExecutionPlan,
    VideoDecodeWindow,
    VideoFrameOwnership,
    resolve_owned_frame_window,
)
from .spatial_intrinsics import SpatialIntrinsicPlan


PlanId: TypeAlias = str
SourceKind: TypeAlias = Literal[
    "decoder", "still", "runtime_raster", "module", "transparent"
]
ColorTransfer: TypeAlias = Literal["fcp_encoded", "fcp_linear"]
ColorRange: TypeAlias = Literal["full", "tv"]
AlphaAssociation: TypeAlias = Literal["opaque", "straight", "premultiplied"]
PixelCoverage: TypeAlias = Literal[
    "full_opaque", "binary_canvas", "arbitrary_alpha"
]
TransitionExtension: TypeAlias = Literal["none", "hold_first", "hold_last"]
StackItemKind: TypeAlias = Literal["layer", "transition"]
LayerExecution: TypeAlias = Literal[
    "composite", "omit_transparent", "authored_disabled"
]
VideoReplacement: TypeAlias = Literal["transparent", "identity", "hard_cut"]


# CompositionPlan is a trust boundary shared by every renderer.  Keep its
# generic resource ceilings here so individual backends cannot silently accept
# different project sizes.  These limits are intentionally above the reviewed
# corpus (8K-class rasters, sub-thousand-frame clips, and ~2 MB pretty-printed
# manifests) while still bounding allocation and serialization work.
MAX_COMPOSITION_SURFACE_DIMENSION = 32_768
MAX_COMPOSITION_SURFACE_PIXELS = 67_108_864
MAX_COMPOSITION_FRAME_COUNT = 1_000_000
MAX_RASTER_CAMERA_SAMPLES = 4_096
MAX_COMPOSITION_CAMERA_SAMPLES = 16_384
MAX_COMPOSITION_MANIFEST_BYTES = 8 * 1024 * 1024


class CompositionPlanError(ValueError):
    """A shared composition contract is incomplete or contradictory."""


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompositionPlanError(f"{field_name} must be a non-empty string")
    return value.strip()


def _sha256(value: object, *, field_name: str) -> str:
    text = _identifier(value, field_name=field_name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CompositionPlanError(f"{field_name} must be lowercase SHA-256 hex")
    return text


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _window_manifest(window: OwnedFrameWindow) -> dict[str, object]:
    return {
        "semantic_start": _fraction_text(window.semantic_start),
        "semantic_end": _fraction_text(window.semantic_end),
        "frame_duration": _fraction_text(window.frame_duration),
        "frame_grid_origin": _fraction_text(window.frame_grid_origin),
        "first_frame": window.first_frame,
        "end_frame": window.end_frame,
        "start": _fraction_text(window.start),
        "end": _fraction_text(window.end),
    }


def _validate_contract_window(
    contract: "FrameContract",
    window: OwnedFrameWindow,
    *,
    owner: str,
) -> None:
    """Require the current absolute-clock convention at every stream edge."""

    if contract.clock.frame_duration != window.frame_duration:
        raise CompositionPlanError(
            f"{owner} frame contract and window use different frame durations"
        )
    if contract.clock.duration != window.duration:
        raise CompositionPlanError(
            f"{owner} frame contract must cover exactly its owned window"
        )
    if contract.clock.pts_origin != window.start:
        raise CompositionPlanError(
            f"{owner} frame contract PTS origin must equal owned window start"
        )


def _validate_contract_extent(
    contract: "FrameContract",
    window: OwnedFrameWindow,
    *,
    owner: str,
) -> None:
    """Require exact duration and placement while allowing an internal cadence."""

    if contract.clock.duration != window.duration:
        raise CompositionPlanError(
            f"{owner} frame contract must cover exactly its owned window"
        )
    if contract.clock.pts_origin != window.start:
        raise CompositionPlanError(
            f"{owner} frame contract PTS origin must equal owned window start"
        )


def _same_frame_representation(left: "FrameContract", right: "FrameContract") -> bool:
    """Return whether two contracts differ, if at all, only in their clock."""

    return (
        left.surface,
        left.transfer,
        left.range,
        left.alpha,
        left.precision,
        left.coverage,
        left.primaries,
        left.matrix,
    ) == (
        right.surface,
        right.transfer,
        right.range,
        right.alpha,
        right.precision,
        right.coverage,
        right.primaries,
        right.matrix,
    )


def _semantic_value(value: Any) -> Any:
    """Return a canonical JSON-safe value for immutable semantic records."""

    if isinstance(value, Fraction):
        return _fraction_text(value)
    if isinstance(value, Enum):
        return _semantic_value(value.value)
    if isinstance(value, Path):
        raise CompositionPlanError(
            "machine-local paths cannot participate in a semantic manifest"
        )
    if is_dataclass(value):
        return {
            item.name: _semantic_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return [_semantic_value(item) for item in value]
    if isinstance(value, list):
        return [_semantic_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise CompositionPlanError(
        f"unsupported semantic manifest value {type(value).__name__}"
    )


@dataclass(frozen=True)
class SurfaceSpec:
    """One raster plus its origin in the owning project's coordinates."""

    width: int
    height: int
    origin_x: int = 0
    origin_y: int = 0

    def __post_init__(self) -> None:
        for field_name in ("width", "height", "origin_x", "origin_y"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise CompositionPlanError(f"{field_name} must be an integer")
        if min(self.width, self.height) <= 0:
            raise CompositionPlanError("surface dimensions must be positive")
        if max(self.width, self.height) > MAX_COMPOSITION_SURFACE_DIMENSION:
            raise CompositionPlanError(
                "surface dimension exceeds the composition plan limit of "
                f"{MAX_COMPOSITION_SURFACE_DIMENSION} pixels"
            )
        if self.width * self.height > MAX_COMPOSITION_SURFACE_PIXELS:
            raise CompositionPlanError(
                "surface area exceeds the composition plan limit of "
                f"{MAX_COMPOSITION_SURFACE_PIXELS} pixels"
            )

    @classmethod
    def from_render_surface(cls, surface: RenderSurface) -> "SurfaceSpec":
        return cls(
            width=surface.width,
            height=surface.height,
            origin_x=surface.origin_x,
            origin_y=surface.origin_y,
        )

    def to_render_surface(self) -> RenderSurface:
        return RenderSurface(self.origin_x, self.origin_y, self.width, self.height)

    def manifest(self) -> dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
        }


@dataclass(frozen=True)
class FrameContract:
    """Complete semantic meaning of frames crossing one module edge."""

    surface: SurfaceSpec
    clock: FrameClock
    transfer: ColorTransfer
    range: ColorRange
    alpha: AlphaAssociation
    precision: Literal[8, 16]
    coverage: PixelCoverage
    primaries: Literal["bt709"] = "bt709"
    matrix: Literal["bt709"] = "bt709"

    def __post_init__(self) -> None:
        if not isinstance(self.surface, SurfaceSpec):
            raise CompositionPlanError("frame surface must be SurfaceSpec")
        if not isinstance(self.clock, FrameClock):
            raise CompositionPlanError("frame clock must be FrameClock")
        if self.clock.frame_count > MAX_COMPOSITION_FRAME_COUNT:
            raise CompositionPlanError(
                "frame clock exceeds the composition plan limit of "
                f"{MAX_COMPOSITION_FRAME_COUNT} frames"
            )
        if self.transfer not in {"fcp_encoded", "fcp_linear"}:
            raise CompositionPlanError(f"unknown transfer {self.transfer!r}")
        if self.range not in {"full", "tv"}:
            raise CompositionPlanError(f"unknown color range {self.range!r}")
        if self.alpha not in {"opaque", "straight", "premultiplied"}:
            raise CompositionPlanError(f"unknown alpha association {self.alpha!r}")
        if self.precision not in {8, 16}:
            raise CompositionPlanError("frame precision must be 8 or 16")
        if self.coverage not in {
            "full_opaque",
            "binary_canvas",
            "arbitrary_alpha",
        }:
            raise CompositionPlanError(f"unknown pixel coverage {self.coverage!r}")
        if (self.alpha == "opaque") != (self.coverage == "full_opaque"):
            raise CompositionPlanError(
                "opaque alpha and full_opaque coverage must be declared together"
            )
        if self.primaries != "bt709" or self.matrix != "bt709":
            raise CompositionPlanError("the current renderer contract is Rec.709 only")

    @classmethod
    def from_pixel_domain(
        cls,
        domain: PixelDomain,
        *,
        surface: SurfaceSpec,
        range: ColorRange,
        coverage: PixelCoverage,
    ) -> "FrameContract":
        """Lift the existing typed pixel domain without losing surface origin."""

        if (domain.width, domain.height) != (surface.width, surface.height):
            raise CompositionPlanError(
                "pixel-domain dimensions do not match the declared surface"
            )
        return cls(
            surface=surface,
            clock=domain.clock,
            transfer=domain.transfer,
            range=range,
            alpha="opaque" if coverage == "full_opaque" else domain.alpha,
            precision=domain.precision,
            coverage=coverage,
        )

    def to_pixel_domain(self) -> PixelDomain:
        """Return the current CPU compiler's narrower pixel-domain view."""

        return PixelDomain(
            transfer=self.transfer,
            alpha="straight" if self.alpha == "opaque" else self.alpha,
            precision=self.precision,
            width=self.surface.width,
            height=self.surface.height,
            clock=self.clock,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "surface": self.surface.manifest(),
            "clock": self.clock.manifest(),
            "transfer": self.transfer,
            "range": self.range,
            "primaries": self.primaries,
            "matrix": self.matrix,
            "alpha": self.alpha,
            "precision": self.precision,
            "coverage": self.coverage,
        }


@dataclass(frozen=True)
class SourceIdentity:
    """Stable source identity independent of paths and decoder input slots."""

    source_id: PlanId
    clip_id: str
    resource_id: str | None = None
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_id", _identifier(self.source_id, field_name="source_id")
        )
        object.__setattr__(self, "clip_id", _identifier(self.clip_id, field_name="clip_id"))
        if self.resource_id is None and self.content_sha256 is None:
            raise CompositionPlanError(
                "source identity requires resource_id or content_sha256"
            )
        if self.resource_id is not None:
            _identifier(self.resource_id, field_name="resource_id")
        if self.content_sha256 is not None:
            _sha256(self.content_sha256, field_name="content_sha256")


@dataclass(frozen=True)
class NominalSamplingContract:
    """A finite sampled sequence whose semantic end may split a frame.

    Main callers:
    - ``DecoderCadenceNormalizationPlan`` after decoder-local retiming.

    Why this exists:
    A retime can produce frames at a nominal source cadence while its semantic
    duration is not an integral number of those frames.  The terminal carrier
    frame is still required so round-up ownership can sample the final project
    frame.  ``FrameClock`` deliberately rejects that shape, so this contract
    records cadence and semantic extent without claiming an exact frame count.
    """

    frame_duration: Fraction
    semantic_duration: Fraction
    pts_origin: Fraction
    endpoint_policy: Literal["cover_semantic_end"] = "cover_semantic_end"

    def __post_init__(self) -> None:
        for field_name in ("frame_duration", "semantic_duration", "pts_origin"):
            value = getattr(self, field_name)
            if not isinstance(value, Fraction):
                raise CompositionPlanError(
                    f"nominal sampling {field_name} must be Fraction"
                )
        if self.frame_duration <= 0 or self.semantic_duration <= 0:
            raise CompositionPlanError(
                "nominal sampling cadence and semantic duration must be positive"
            )
        if self.endpoint_policy != "cover_semantic_end":
            raise CompositionPlanError(
                "nominal sampling must retain a carrier through the semantic end"
            )


@dataclass(frozen=True)
class DecoderCadenceNormalizationPlan:
    """Sample decoder-retimed frames onto the exact layer-facing frame grid.

    This stage executes after ``DecoderSourcePlan.retime`` and before source
    spatial normalization.  Its input is nominal rather than a ``FrameClock``
    because a semantic endpoint can fall inside the terminal input frame.
    """

    input_sampling: NominalSamplingContract
    output_window: OwnedFrameWindow
    output_contract: FrameContract
    frame_grid_origin: Fraction
    rounding: Literal["up"] = "up"

    def __post_init__(self) -> None:
        if not isinstance(self.input_sampling, NominalSamplingContract):
            raise CompositionPlanError(
                "decoder cadence normalization input must be nominal sampling"
            )
        if not isinstance(self.output_window, OwnedFrameWindow):
            raise CompositionPlanError(
                "decoder cadence normalization requires an owned output window"
            )
        if not isinstance(self.output_contract, FrameContract):
            raise CompositionPlanError(
                "decoder cadence normalization output must be FrameContract"
            )
        if not isinstance(self.frame_grid_origin, Fraction):
            raise CompositionPlanError(
                "decoder cadence normalization grid origin must be Fraction"
            )
        if self.rounding != "up":
            raise CompositionPlanError(
                "decoder cadence normalization requires round-up ownership"
            )
        _validate_contract_window(
            self.output_contract,
            self.output_window,
            owner="decoder cadence normalization output",
        )
        if self.input_sampling.semantic_duration != self.output_window.duration:
            raise CompositionPlanError(
                "decoder nominal sampling must cover the owned output duration"
            )
        if self.input_sampling.pts_origin != self.output_window.start:
            raise CompositionPlanError(
                "decoder nominal sampling origin must equal owned output start"
            )
        if self.frame_grid_origin != self.output_window.frame_grid_origin:
            raise CompositionPlanError(
                "decoder cadence normalization must use the owned output grid"
            )
        if (
            self.input_sampling.frame_duration
            == self.output_contract.clock.frame_duration
        ):
            raise CompositionPlanError(
                "decoder cadence normalization requires different cadences"
            )


@dataclass(frozen=True)
class DecoderSourcePlan:
    """File-local sampling semantics shared by every decoder backend.

    Main callers:
    - The shared composition compiler after source bounds are validated.
    - Backend decoder binders, which choose pre-seek policy separately.

    Why this exists:
    Equivalent pre-seeked and unseeked FFmpeg inputs must have identical
    semantic plans.  Only validated frame ownership and complete aligned
    decode coverage belong here. ``retime`` is decoder-local source sampling;
    it is fully consumed before ``frame_contract`` reaches a composition
    layer. The frame contract therefore records the final layer-facing output
    cadence and may intentionally differ from the decoder/source cadence. A
    backend must normalize that difference inside source execution rather than
    attaching the same retime to the consuming layer.
    """

    identity: SourceIdentity
    encoded_probe_surface: SurfaceSpec
    display_surface: SurfaceSpec
    spatial_intrinsics: SpatialIntrinsicPlan | None
    output_window: OwnedFrameWindow
    decode_window: VideoDecodeWindow
    first_sample: VideoFrameOwnership
    frame_contract: FrameContract
    retime: RetimeExecutionPlan | None = None
    cadence_normalization: DecoderCadenceNormalizationPlan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SourceIdentity):
            raise CompositionPlanError("decoder identity must be SourceIdentity")
        if not isinstance(self.encoded_probe_surface, SurfaceSpec):
            raise CompositionPlanError("encoded_probe_surface must be SurfaceSpec")
        if not isinstance(self.display_surface, SurfaceSpec):
            raise CompositionPlanError("display_surface must be SurfaceSpec")
        if self.spatial_intrinsics is not None and not isinstance(
            self.spatial_intrinsics,
            SpatialIntrinsicPlan,
        ):
            raise CompositionPlanError(
                "spatial_intrinsics must be SpatialIntrinsicPlan or None"
            )
        if self.spatial_intrinsics is not None and (
            self.spatial_intrinsics.frame_width,
            self.spatial_intrinsics.frame_height,
        ) != (
            self.encoded_probe_surface.width,
            self.encoded_probe_surface.height,
        ):
            raise CompositionPlanError(
                "source normalization input dimensions must match the encoded "
                "probe surface"
            )
        if (
            self.spatial_intrinsics is None
            and self.encoded_probe_surface != self.display_surface
        ):
            raise CompositionPlanError(
                "encoded and display surfaces may differ only when source "
                "normalization is explicit"
            )
        if not isinstance(self.output_window, OwnedFrameWindow):
            raise CompositionPlanError("output_window must be OwnedFrameWindow")
        if not isinstance(self.decode_window, VideoDecodeWindow):
            raise CompositionPlanError("decode_window must be VideoDecodeWindow")
        if not isinstance(self.first_sample, VideoFrameOwnership):
            raise CompositionPlanError("first_sample must be VideoFrameOwnership")
        if not isinstance(self.frame_contract, FrameContract):
            raise CompositionPlanError("frame_contract must be FrameContract")
        if self.first_sample.frame_duration != self.decode_window.frame_duration:
            raise CompositionPlanError(
                "first sample and decode window must share source frame duration"
            )
        if self.first_sample.frame_grid_origin != self.decode_window.frame_grid_origin:
            raise CompositionPlanError(
                "first sample and decode window must share source frame grid"
            )
        if not (
            self.decode_window.decode_start
            <= self.first_sample.source_frame_start
            < self.decode_window.decode_end
        ):
            raise CompositionPlanError("first sample lies outside decoder coverage")
        _validate_contract_window(
            self.frame_contract,
            self.output_window,
            owner="decoder source",
        )
        if self.frame_contract.surface != self.display_surface:
            raise CompositionPlanError(
                "decoder frame contract must expose the declared display surface"
            )
        if self.retime is not None and not isinstance(self.retime, RetimeExecutionPlan):
            raise CompositionPlanError("retime must be RetimeExecutionPlan or None")
        if self.retime is None:
            if self.cadence_normalization is not None:
                raise CompositionPlanError(
                    "ordinary decoder cannot declare cadence normalization"
                )
            if self.first_sample.source_time != self.decode_window.semantic_start:
                raise CompositionPlanError(
                    "ordinary decoder first sample must own semantic source start"
                )
        else:
            if self.retime.output_duration != self.output_window.duration:
                raise CompositionPlanError(
                    "retime output duration must equal the owned output window"
                )
            if not self.retime.video_segments:
                raise CompositionPlanError("retime plan has no video segments")
            first_segment = self.retime.video_segments[0]
            if self.first_sample != first_segment.frame_ownership:
                raise CompositionPlanError(
                    "decoder first sample must match the first retime segment"
                )
            for segment in self.retime.video_segments:
                if (
                    segment.decode_window.frame_duration
                    != self.decode_window.frame_duration
                    or segment.decode_window.frame_grid_origin
                    != self.decode_window.frame_grid_origin
                ):
                    raise CompositionPlanError(
                        "retime segments and decoder coverage must share one source grid"
                    )
                if (
                    segment.decode_window.decode_start
                    < self.decode_window.decode_start
                    or segment.decode_window.decode_end
                    > self.decode_window.decode_end
                ):
                    raise CompositionPlanError(
                        "decoder coverage does not include every retime segment"
                    )
            cadence_differs = (
                self.retime.video_frame_duration
                != self.frame_contract.clock.frame_duration
            )
            if cadence_differs and self.cadence_normalization is None:
                raise CompositionPlanError(
                    "decoder retime cadence differs from its layer-facing cadence "
                    "without normalization"
                )
            if not cadence_differs and self.cadence_normalization is not None:
                raise CompositionPlanError(
                    "same-cadence decoder retime cannot declare normalization"
                )
            if self.cadence_normalization is not None:
                normalization = self.cadence_normalization
                if not isinstance(normalization, DecoderCadenceNormalizationPlan):
                    raise CompositionPlanError(
                        "cadence_normalization must be DecoderCadenceNormalizationPlan"
                    )
                if (
                    normalization.input_sampling.frame_duration
                    != self.retime.video_frame_duration
                ):
                    raise CompositionPlanError(
                        "decoder normalization input cadence must match retime output"
                    )
                if (
                    normalization.input_sampling.semantic_duration
                    != self.retime.output_duration
                ):
                    raise CompositionPlanError(
                        "decoder normalization duration must match retime output"
                    )
                if normalization.output_window != self.output_window:
                    raise CompositionPlanError(
                        "decoder normalization output window must match its source"
                    )
                if normalization.output_contract != self.frame_contract:
                    raise CompositionPlanError(
                        "decoder normalization output contract must be layer-facing"
                    )


@dataclass(frozen=True)
class RasterSourcePlan:
    """A still, title, caption, or generator raster resolved before lowering."""

    identity: SourceIdentity
    kind: Literal["still", "runtime_raster"]
    output_window: OwnedFrameWindow
    frame_contract: FrameContract

    def __post_init__(self) -> None:
        if self.kind not in {"still", "runtime_raster"}:
            raise CompositionPlanError(f"unknown raster source kind {self.kind!r}")
        if not isinstance(self.identity, SourceIdentity):
            raise CompositionPlanError("raster identity must be SourceIdentity")
        if self.kind == "runtime_raster" and self.identity.content_sha256 is None:
            raise CompositionPlanError(
                "runtime raster source requires a content hash"
            )
        _validate_contract_window(
            self.frame_contract,
            self.output_window,
            owner="raster source",
        )


@dataclass(frozen=True)
class RuntimeSourceBinding:
    """Machine-local path/probe data deliberately excluded from semantics."""

    source_id: PlanId
    resolved_path: Path
    width: int
    height: int
    pixel_format: str | None
    sample_aspect_ratio: str | None
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, field_name="source_id"))
        if not isinstance(self.resolved_path, Path) or not self.resolved_path.is_absolute():
            raise CompositionPlanError("resolved_path must be an absolute Path")
        for field_name in ("width", "height"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CompositionPlanError(f"{field_name} must be a positive integer")
        if self.content_sha256 is not None:
            _sha256(self.content_sha256, field_name="content_sha256")

    def validate_against(
        self,
        source: DecoderSourcePlan | RasterSourcePlan,
    ) -> None:
        """Prove that runtime resolution still names the frozen source raster."""

        if self.source_id != source.identity.source_id:
            raise CompositionPlanError("runtime binding references another source")
        surface = (
            source.encoded_probe_surface
            if isinstance(source, DecoderSourcePlan)
            else source.frame_contract.surface
        )
        if (self.width, self.height) != (surface.width, surface.height):
            raise CompositionPlanError(
                "runtime binding dimensions differ from the semantic source"
            )
        expected_hash = source.identity.content_sha256
        if expected_hash is not None and self.content_sha256 != expected_hash:
            raise CompositionPlanError(
                "runtime binding content hash differs from the semantic source"
            )


@dataclass(frozen=True)
class DecoderBinding:
    """One backend decoder optimization bound to shared source semantics."""

    decoder_id: PlanId
    input_index: int
    decoder_seek: Fraction
    decoder_start_frame: int
    filter_start_frame: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "decoder_id", _identifier(self.decoder_id, field_name="decoder_id")
        )
        if isinstance(self.input_index, bool) or not isinstance(self.input_index, int):
            raise CompositionPlanError("input_index must be an integer")
        if self.input_index < 0:
            raise CompositionPlanError("input_index cannot be negative")
        if not isinstance(self.decoder_seek, Fraction) or self.decoder_seek < 0:
            raise CompositionPlanError("decoder_seek must be a non-negative Fraction")
        for field_name in ("decoder_start_frame", "filter_start_frame"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise CompositionPlanError(f"{field_name} must be an integer")

    @classmethod
    def from_origin(cls, decoder_id: PlanId, origin: VideoDecoderOrigin) -> "DecoderBinding":
        complete = origin.require_frame_ownership()
        assert complete.decoder_start_frame is not None
        assert complete.filter_start_frame is not None
        return cls(
            decoder_id=decoder_id,
            input_index=complete.input_index,
            decoder_seek=complete.decoder_seek,
            decoder_start_frame=complete.decoder_start_frame,
            filter_start_frame=complete.filter_start_frame,
        )

    def validate_against(
        self,
        source: DecoderSourcePlan,
        *,
        clip_id: str,
    ) -> VideoDecoderOrigin:
        """Reconstruct and validate the existing complete backend origin."""

        if self.decoder_id != source.identity.source_id:
            raise CompositionPlanError("decoder binding references another source")
        ownership = source.first_sample
        expected_seek = (
            ownership.frame_grid_origin
            + self.decoder_start_frame * ownership.frame_duration
        )
        if self.decoder_seek != expected_seek:
            raise CompositionPlanError(
                "decoder seek must equal its exact aligned decoder frame"
            )
        if (
            ownership.source_start_frame
            != self.decoder_start_frame + self.filter_start_frame
        ):
            raise CompositionPlanError(
                "source frame must equal decoder frame plus filter-local frame"
            )
        filter_source_start = ownership.source_time - self.decoder_seek
        return VideoDecoderOrigin(
            clip_id=clip_id,
            input_index=self.input_index,
            file_source_start=ownership.source_time,
            decoder_seek=self.decoder_seek,
            filter_source_start=filter_source_start,
            source_frame_duration=ownership.frame_duration,
            source_start_frame=ownership.source_start_frame,
            source_phase=ownership.source_phase,
            sampling_direction=ownership.direction,
            decoder_start_frame=self.decoder_start_frame,
            filter_start_frame=self.filter_start_frame,
        ).require_frame_ownership()


@dataclass(frozen=True)
class SourceRef:
    """A layer's decoder, raster, nested module, or explicit transparency."""

    kind: SourceKind
    ref: PlanId | None
    omission_reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "decoder",
            "still",
            "runtime_raster",
            "module",
            "transparent",
        }:
            raise CompositionPlanError(f"unknown source reference kind {self.kind!r}")
        if self.kind == "transparent":
            if self.ref is not None:
                raise CompositionPlanError("transparent source must not carry a reference")
            _identifier(self.omission_reason, field_name="transparent source reason")
        else:
            _identifier(self.ref, field_name="source reference")
            if self.omission_reason is not None:
                raise CompositionPlanError(
                    "non-transparent source cannot carry an omission reason"
                )


def _finite_sampling_value(value: object, *, field_name: str) -> float:
    """Return one finite sampler coordinate without accepting bool as a number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompositionPlanError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CompositionPlanError(f"{field_name} must be a finite number")
    return result


@dataclass(frozen=True)
class SamplingRect:
    """One resolved source-space rectangle in square display pixels."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        for field_name in ("left", "top", "right", "bottom"):
            _finite_sampling_value(
                getattr(self, field_name),
                field_name=f"sampling rectangle {field_name}",
            )
        if self.right <= self.left or self.bottom <= self.top:
            raise CompositionPlanError(
                "sampling rectangle must have positive resolved extent"
            )


@dataclass(frozen=True)
class SourceAlphaWindowPlan:
    """Describe how Crop/Pan owns source alpha before camera sampling.

    ``preserve_full_source`` is the no-filter identity case. Crop and Pan use
    ``multiply_inside_window``: existing source alpha survives when the source
    FFmpeg sample index ``(X, Y)`` lies inside the inclusive window and becomes
    transparent outside it before interpolation. This records the CPU
    renderer's existing GEQ ownership rule; it does not reinterpret authored
    boundaries as pixel-center coordinates.
    """

    rect: SamplingRect
    behavior: Literal["preserve_full_source", "multiply_inside_window"]
    edge_rule: Literal["inclusive_integer_sample_indices"] = (
        "inclusive_integer_sample_indices"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.rect, SamplingRect):
            raise CompositionPlanError("source alpha window must use SamplingRect")
        if self.behavior not in {
            "preserve_full_source",
            "multiply_inside_window",
        }:
            raise CompositionPlanError(
                f"unknown source alpha behavior {self.behavior!r}"
            )
        if self.edge_rule != "inclusive_integer_sample_indices":
            raise CompositionPlanError(
                "source alpha window must use inclusive integer sample indices"
            )


SamplingPoint: TypeAlias = tuple[float, float]
SamplingQuad: TypeAlias = tuple[
    SamplingPoint,
    SamplingPoint,
    SamplingPoint,
    SamplingPoint,
]


def _validate_sampling_quad(value: object, *, field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 4:
        raise CompositionPlanError(f"{field_name} must contain four points")
    for point_index, point in enumerate(value):
        if not isinstance(point, tuple) or len(point) != 2:
            raise CompositionPlanError(
                f"{field_name}[{point_index}] must be an (x, y) pair"
            )
        for axis, coordinate in zip(("x", "y"), point, strict=True):
            _finite_sampling_value(
                coordinate,
                field_name=f"{field_name}[{point_index}].{axis}",
            )


def _sampling_values_close(left: object, right: object) -> bool:
    """Compare nested sampler coordinates at the float boundary they own."""

    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _sampling_values_close(a, b)
            for a, b in zip(left, right, strict=True)
        )
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


@dataclass(frozen=True)
class ResolvedCameraSample:
    """One fully resolved camera sample on the output-owned frame schedule."""

    frame_index: int
    source_alpha_window: SourceAlphaWindowPlan
    scale: float
    origin_x: float
    origin_y: float
    camera_quad: SamplingQuad
    pixel_center_quad: SamplingQuad
    transparent_border_quad: SamplingQuad

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise CompositionPlanError("camera sample frame_index must be an integer")
        if self.frame_index < 0:
            raise CompositionPlanError("camera sample frame_index cannot be negative")
        if not isinstance(self.source_alpha_window, SourceAlphaWindowPlan):
            raise CompositionPlanError(
                "camera sample requires SourceAlphaWindowPlan"
            )
        scale = _finite_sampling_value(self.scale, field_name="camera scale")
        if scale <= 0:
            raise CompositionPlanError("camera scale must be positive")
        _finite_sampling_value(self.origin_x, field_name="camera origin_x")
        _finite_sampling_value(self.origin_y, field_name="camera origin_y")
        for field_name in (
            "camera_quad",
            "pixel_center_quad",
            "transparent_border_quad",
        ):
            _validate_sampling_quad(getattr(self, field_name), field_name=field_name)


@dataclass(frozen=True)
class RasterCameraSamplingPlan:
    """Backend-neutral reconstruction contract for one raster camera.

    Architecture map
    ================

    resolved source alpha window
        -> optional two-pixel transparent support border
        -> power-linear RGB / linear-alpha bilinear sampling
        -> Final Cut pixel-center corrected destination quad
        -> exact output-canvas clip

    ``samples`` is already on the layer's output frame schedule. Static Crop
    and identity Fit/Fill carry one sample; two-rectangle Pan carries every
    output frame. A lowerer therefore does not reopen FCPXML, rediscover frame
    count, or rebuild Ken Burns timing.
    """

    operation: Literal[
        "identity_fit", "identity_fill", "static_crop", "two_rect_pan"
    ]
    frame_count: int
    support_surface: SurfaceSpec
    padded_support_surface: SurfaceSpec
    output_clip: SurfaceSpec
    pixel_center_convention: Literal[
        "identity", "half_transformed_pixel_diagonal"
    ]
    interpolation_kernel: Literal["identity", "bilinear"]
    rgb_interpolation_space: Literal["identity", "power_linear_1_94"]
    alpha_interpolation: Literal["identity", "linear"]
    transparent_border_behavior: Literal[
        "none", "pad_expand_quad_then_clip"
    ]
    transparent_border_pixels: int
    samples: tuple[ResolvedCameraSample, ...]

    def __post_init__(self) -> None:
        if self.operation not in {
            "identity_fit",
            "identity_fill",
            "static_crop",
            "two_rect_pan",
        }:
            raise CompositionPlanError(
                f"unknown raster camera operation {self.operation!r}"
            )
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int):
            raise CompositionPlanError("camera frame_count must be an integer")
        if self.frame_count <= 0:
            raise CompositionPlanError("camera frame_count must be positive")
        if self.frame_count > MAX_COMPOSITION_FRAME_COUNT:
            raise CompositionPlanError(
                "camera frame_count exceeds the composition plan limit of "
                f"{MAX_COMPOSITION_FRAME_COUNT} frames"
            )
        if len(self.samples) > MAX_RASTER_CAMERA_SAMPLES:
            raise CompositionPlanError(
                "raster camera sample count exceeds the composition plan limit of "
                f"{MAX_RASTER_CAMERA_SAMPLES} samples"
            )
        for field_name in (
            "support_surface",
            "padded_support_surface",
            "output_clip",
        ):
            if not isinstance(getattr(self, field_name), SurfaceSpec):
                raise CompositionPlanError(
                    f"camera {field_name} must be SurfaceSpec"
                )
        if self.output_clip.origin_x != 0 or self.output_clip.origin_y != 0:
            raise CompositionPlanError(
                "camera output clip must use top-left output coordinates"
            )
        if (
            isinstance(self.transparent_border_pixels, bool)
            or not isinstance(self.transparent_border_pixels, int)
            or self.transparent_border_pixels < 0
        ):
            raise CompositionPlanError(
                "transparent camera border must be a non-negative integer"
            )
        is_identity = self.operation in {"identity_fit", "identity_fill"}
        expected_convention = (
            "identity" if is_identity else "half_transformed_pixel_diagonal"
        )
        expected_kernel = "identity" if is_identity else "bilinear"
        expected_rgb = "identity" if is_identity else "power_linear_1_94"
        expected_alpha = "identity" if is_identity else "linear"
        expected_border_behavior = (
            "none" if is_identity else "pad_expand_quad_then_clip"
        )
        expected_border = 0 if is_identity else TRANSPARENT_PERSPECTIVE_BORDER
        actual = (
            self.pixel_center_convention,
            self.interpolation_kernel,
            self.rgb_interpolation_space,
            self.alpha_interpolation,
            self.transparent_border_behavior,
            self.transparent_border_pixels,
        )
        expected = (
            expected_convention,
            expected_kernel,
            expected_rgb,
            expected_alpha,
            expected_border_behavior,
            expected_border,
        )
        if actual != expected:
            raise CompositionPlanError(
                "camera sampler kernel, pixel-center, or transparent-border "
                "contract does not match its operation"
            )
        border = self.transparent_border_pixels
        expected_padded = SurfaceSpec(
            self.support_surface.width + 2 * border,
            self.support_surface.height + 2 * border,
            self.support_surface.origin_x - border,
            self.support_surface.origin_y - border,
        )
        if self.padded_support_surface != expected_padded:
            raise CompositionPlanError(
                "camera padded support does not match its transparent border"
            )
        expected_samples = 1 if self.operation != "two_rect_pan" else self.frame_count
        if len(self.samples) != expected_samples:
            raise CompositionPlanError(
                f"{self.operation} camera requires exactly {expected_samples} sample(s)"
            )
        expected_indices = (
            (0,)
            if self.operation != "two_rect_pan"
            else tuple(range(self.frame_count))
        )
        if tuple(sample.frame_index for sample in self.samples) != expected_indices:
            raise CompositionPlanError(
                "camera samples must exactly cover their canonical frame indices"
            )
        for sample in self.samples:
            width = self.support_surface.width
            height = self.support_surface.height
            expected_camera: SamplingQuad = (
                (sample.origin_x, sample.origin_y),
                (sample.origin_x + width * sample.scale, sample.origin_y),
                (sample.origin_x, sample.origin_y + height * sample.scale),
                (
                    sample.origin_x + width * sample.scale,
                    sample.origin_y + height * sample.scale,
                ),
            )
            if not _sampling_values_close(sample.camera_quad, expected_camera):
                raise CompositionPlanError(
                    "camera quad does not match resolved scale/origin/support"
                )
            if is_identity:
                if not _sampling_values_close(
                    sample.pixel_center_quad, sample.camera_quad
                ) or not _sampling_values_close(
                    sample.transparent_border_quad, sample.camera_quad
                ):
                    raise CompositionPlanError(
                        "identity camera cannot alter its destination quad"
                    )
            else:
                corrected = correct_quad_for_pixel_centers(
                    sample.camera_quad,
                    width=width,
                    height=height,
                )
                expanded = expand_quad_for_transparent_border(
                    corrected,
                    width=width,
                    height=height,
                    border=border,
                )
                if not _sampling_values_close(sample.pixel_center_quad, corrected):
                    raise CompositionPlanError(
                        "camera pixel-center quad is not the canonical correction"
                    )
                if not _sampling_values_close(
                    sample.transparent_border_quad, expanded
                ):
                    raise CompositionPlanError(
                        "camera border quad is not the canonical transparent expansion"
                    )
            expected_alpha_behavior = (
                "preserve_full_source" if is_identity else "multiply_inside_window"
            )
            if sample.source_alpha_window.behavior != expected_alpha_behavior:
                raise CompositionPlanError(
                    "camera source alpha behavior does not match its operation"
                )


@dataclass(frozen=True)
class RasterPlacementPlan:
    """Crop and conform semantics before effects and spatial transforms."""

    source_surface: SurfaceSpec
    output_surface: SurfaceSpec
    conform: Literal["fit", "fill", "none"]
    crop: CropAdjustment | None
    requires_transparent_plate: bool
    input_contract: FrameContract
    output_contract: FrameContract
    camera_sampling: RasterCameraSamplingPlan | None = None

    def __post_init__(self) -> None:
        if self.conform not in {"fit", "fill", "none"}:
            raise CompositionPlanError(f"unknown conform mode {self.conform!r}")
        if self.crop is not None and not isinstance(self.crop, CropAdjustment):
            raise CompositionPlanError("crop must be CropAdjustment or None")
        if not isinstance(self.requires_transparent_plate, bool):
            raise CompositionPlanError("requires_transparent_plate must be bool")
        if self.input_contract.surface != self.source_surface:
            raise CompositionPlanError(
                "raster input contract must match its source surface"
            )
        if self.output_contract.surface != self.output_surface:
            raise CompositionPlanError(
                "raster output contract must match its output surface"
            )
        if self.input_contract.clock != self.output_contract.clock:
            raise CompositionPlanError(
                "raster placement cannot change the semantic frame clock"
            )
        crop_mode = (
            self.crop.mode.strip().casefold()
            if self.crop is not None and self.crop.enabled
            else None
        )
        sampling = self.camera_sampling
        if sampling is None:
            if crop_mode in {"crop", "pan"}:
                raise CompositionPlanError(
                    "active Crop/Pan raster placement requires captured camera sampling"
                )
            return
        if not isinstance(sampling, RasterCameraSamplingPlan):
            raise CompositionPlanError(
                "raster camera_sampling must be RasterCameraSamplingPlan or None"
            )
        if sampling.frame_count != self.input_contract.clock.frame_count:
            raise CompositionPlanError(
                "raster camera schedule must cover the exact input frame count"
            )
        if sampling.output_clip != self.output_surface:
            raise CompositionPlanError(
                "raster camera output clip must match its output surface"
            )
        if (
            sampling.support_surface.origin_x != 0
            or sampling.support_surface.origin_y != 0
            or sampling.support_surface.width
            != max(self.source_surface.width, self.output_surface.width)
            or sampling.support_surface.height
            != max(self.source_surface.height, self.output_surface.height)
        ):
            raise CompositionPlanError(
                "raster camera support must be the exact source/output union at origin zero"
            )
        expected_operation = {
            "crop": "static_crop",
            "pan": "two_rect_pan",
        }.get(crop_mode)
        if expected_operation is None and self.source_surface == self.output_surface:
            expected_operation = {
                "fit": "identity_fit",
                "fill": "identity_fill",
            }.get(self.conform)
        if expected_operation != sampling.operation:
            raise CompositionPlanError(
                "raster camera operation does not match Crop/Pan/Fit/Fill semantics"
            )
        source_bounds = SamplingRect(
            0.0,
            0.0,
            float(self.source_surface.width),
            float(self.source_surface.height),
        )
        if sampling.operation in {"identity_fit", "identity_fill"}:
            sample = sampling.samples[0]
            if (
                sample.source_alpha_window.rect != source_bounds
                or not _sampling_values_close(sample.scale, 1.0)
                or not _sampling_values_close(sample.origin_x, 0.0)
                or not _sampling_values_close(sample.origin_y, 0.0)
            ):
                raise CompositionPlanError(
                    "identity Fit/Fill must preserve the full source at scale one"
                )
            return
        assert self.crop is not None
        active_rects = self.crop.active_rects
        authored_rects = (
            (active_rects[0],)
            if sampling.operation == "static_crop"
            else (active_rects[0], active_rects[-1])
        )
        sampled_endpoints = (
            (sampling.samples[0],)
            if sampling.operation == "static_crop"
            else (sampling.samples[0], sampling.samples[-1])
        )
        unit = self.source_surface.height / 100.0
        for authored, sample in zip(
            authored_rects,
            sampled_endpoints,
            strict=True,
        ):
            expected_window = SamplingRect(
                authored.left * unit,
                authored.top * unit,
                self.source_surface.width - authored.right * unit,
                self.source_surface.height - authored.bottom * unit,
            )
            if not _sampling_values_close(
                (
                    sample.source_alpha_window.rect.left,
                    sample.source_alpha_window.rect.top,
                    sample.source_alpha_window.rect.right,
                    sample.source_alpha_window.rect.bottom,
                ),
                (
                    expected_window.left,
                    expected_window.top,
                    expected_window.right,
                    expected_window.bottom,
                ),
            ):
                raise CompositionPlanError(
                    "camera source alpha endpoints do not match authored Crop/Pan rectangles"
                )


def _raster_placement_manifest(
    placement: RasterPlacementPlan,
) -> dict[str, object]:
    """Add sampler semantics without changing hashes for unrelated plans."""

    result = _semantic_value(placement)
    if not isinstance(result, dict):
        raise CompositionPlanError("raster placement manifest must be a mapping")
    if placement.camera_sampling is None:
        result.pop("camera_sampling", None)
    return result


@dataclass(frozen=True)
class RasterSpatialBoundaryPlan:
    """Control what a backend may materialize between raster and spatial stages.

    ``preserve_raster_support`` keeps the semantic overscan requirement.  Each
    backend may fuse both stages or materialize an expanded intermediate, but
    neither choice may crop to the logical canvas before the following spatial
    transform samples overscan.  The physical choice belongs to that backend's
    execution plan and never participates in this semantic record.
    """

    materialization: Literal[
        "logical_canvas_allowed", "preserve_raster_support"
    ] = "logical_canvas_allowed"
    reason: Literal["conform_overscan"] | None = None

    def __post_init__(self) -> None:
        if self.materialization not in {
            "logical_canvas_allowed",
            "preserve_raster_support",
        }:
            raise CompositionPlanError(
                f"unknown raster/spatial materialization {self.materialization!r}"
            )
        if self.materialization == "preserve_raster_support":
            if self.reason != "conform_overscan":
                raise CompositionPlanError(
                    "preserve_raster_support requires reason='conform_overscan'"
                )
        elif self.reason is not None:
            raise CompositionPlanError(
                "logical_canvas_allowed cannot carry a preservation reason"
            )


@dataclass(frozen=True)
class ContainerConformPlan:
    """Fit/fill a completed child container into its parent's coordinates.

    This conform is intentionally not source normalization or ordinary raster
    placement. Final Cut applies it after the nested group's effects, at the
    same boundary as the owning container's affine transform. The actual
    spatial input may be an expanded child surface, so the two camera canvases
    are recorded independently from that raster's bounds.
    """

    source_canvas: SurfaceSpec
    parent_canvas: SurfaceSpec
    conform: Literal["fit", "fill", "none"]

    def __post_init__(self) -> None:
        if not isinstance(self.source_canvas, SurfaceSpec):
            raise CompositionPlanError(
                "container source canvas must be SurfaceSpec"
            )
        if not isinstance(self.parent_canvas, SurfaceSpec):
            raise CompositionPlanError(
                "container parent canvas must be SurfaceSpec"
            )
        if self.conform not in {"fit", "fill", "none"}:
            raise CompositionPlanError(
                f"unknown container conform mode {self.conform!r}"
            )


@dataclass(frozen=True)
class SpatialTransformPlan:
    """Corner-pin then affine semantics after the complete effect stack."""

    input_surface: SurfaceSpec
    output_surface: SurfaceSpec
    transform: TransformAdjustment | None
    animation: RenderTransformAnimation | None
    corner_pin: CornerPinAdjustment | None
    requires_transparent_intermediate: bool
    input_contract: FrameContract
    output_contract: FrameContract
    container_conform: ContainerConformPlan | None = None

    def __post_init__(self) -> None:
        if self.transform is not None and not isinstance(self.transform, TransformAdjustment):
            raise CompositionPlanError("transform must be TransformAdjustment or None")
        if self.animation is not None and not isinstance(
            self.animation, RenderTransformAnimation
        ):
            raise CompositionPlanError(
                "animation must be RenderTransformAnimation or None"
            )
        if self.corner_pin is not None and not isinstance(
            self.corner_pin, CornerPinAdjustment
        ):
            raise CompositionPlanError(
                "corner_pin must be CornerPinAdjustment or None"
            )
        if not isinstance(self.requires_transparent_intermediate, bool):
            raise CompositionPlanError(
                "requires_transparent_intermediate must be bool"
            )
        if self.container_conform is not None and not isinstance(
            self.container_conform, ContainerConformPlan
        ):
            raise CompositionPlanError(
                "container_conform must be ContainerConformPlan or None"
            )
        if self.input_contract.surface != self.input_surface:
            raise CompositionPlanError(
                "spatial input contract must match its input surface"
            )
        if self.output_contract.surface != self.output_surface:
            raise CompositionPlanError(
                "spatial output contract must match its output surface"
            )
        if self.input_contract.clock != self.output_contract.clock:
            raise CompositionPlanError(
                "spatial transform cannot change the semantic frame clock"
            )


@dataclass(frozen=True)
class FrameRateNormalizationPlan:
    """Normalize a completed child raster onto its parent's video cadence.

    Main callers:
    - The shared composition compiler at a nested-container module boundary.

    Why this exists:
    A child group may execute effects, geometry, and a source-instance retime
    on its own exact frame grid. Only after those stages does Final Cut sample
    the completed child onto the parent grid. Keeping this as a typed stage
    prevents a backend from silently treating cadence conversion as retiming.
    """

    input_contract: FrameContract
    output_contract: FrameContract
    frame_grid_origin: Fraction = Fraction(0)
    rounding: Literal["up"] = "up"

    def __post_init__(self) -> None:
        if not isinstance(self.input_contract, FrameContract) or not isinstance(
            self.output_contract, FrameContract
        ):
            raise CompositionPlanError(
                "frame-rate normalization contracts must be FrameContract"
            )
        if not isinstance(self.frame_grid_origin, Fraction):
            raise CompositionPlanError(
                "frame-rate normalization grid origin must be Fraction"
            )
        if self.rounding != "up":
            raise CompositionPlanError(
                "frame-rate normalization currently requires round-up ownership"
            )
        if not _same_frame_representation(
            self.input_contract, self.output_contract
        ):
            raise CompositionPlanError(
                "frame-rate normalization cannot change surface or pixel representation"
            )
        if (
            self.input_contract.clock.duration
            != self.output_contract.clock.duration
            or self.input_contract.clock.pts_origin
            != self.output_contract.clock.pts_origin
        ):
            raise CompositionPlanError(
                "frame-rate normalization must preserve exact duration and PTS origin"
            )
        if (
            self.input_contract.clock.frame_duration
            == self.output_contract.clock.frame_duration
        ):
            raise CompositionPlanError(
                "frame-rate normalization requires different input and output cadences"
            )


@dataclass(frozen=True)
class OpacityEnvelopePlan:
    """Static/animated opacity and fade independent of a CPU expression."""

    window: CompositorWindow
    static_opacity: float
    animation: TimelineAnimatedScalar | None
    fade: FadeEnvelope | None
    expression_time_origin: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        try:
            self.to_opacity_plan()
        except ValueError as error:
            raise CompositionPlanError(f"invalid opacity envelope: {error}") from error

    def to_opacity_plan(self) -> OpacityPlan:
        """Adapt the semantic envelope to the existing CPU compositor."""

        return OpacityPlan(
            window=self.window,
            static_opacity=self.static_opacity,
            animation=self.animation,
            fade=self.fade,
            expression_time_origin=self.expression_time_origin,
        )

    @property
    def is_constant_one(self) -> bool:
        return self.to_opacity_plan().is_constant_one


@dataclass(frozen=True)
class MaskOp:
    """One normalized mask primitive without mutable mapping state."""

    kind: str
    name: str
    blend_mode: str
    params: tuple[Parameter, ...]
    data: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.kind, field_name="mask kind")
        _identifier(self.name, field_name="mask name")
        _identifier(self.blend_mode, field_name="mask blend mode")
        if tuple(sorted(self.data)) != self.data:
            raise CompositionPlanError("mask data must be sorted canonical pairs")


@dataclass(frozen=True)
class EffectOp:
    """One reviewed executable semantic effect.

    ``data`` preserves opaque authored sidecar fields as sorted immutable
    pairs.  A backend may understand an exact sidecar value or reject it, but
    it may never make an effect appear simpler by dropping that value between
    CPU lowering and the shared plan.
    """

    effect_id: PlanId
    path: str
    handler: str
    params: tuple[Parameter, ...]
    calibration_id: str | None
    input_contract: FrameContract
    output_contract: FrameContract
    data: tuple[tuple[str, str], ...] = ()
    temporal_radius: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        _identifier(self.effect_id, field_name="effect_id")
        _identifier(self.path, field_name="effect path")
        _identifier(self.handler, field_name="effect handler")
        if tuple(sorted(self.data)) != self.data:
            raise CompositionPlanError("effect data must be sorted canonical pairs")
        if len({key for key, _value in self.data}) != len(self.data):
            raise CompositionPlanError("effect data keys must be unique")
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str)
            for key, value in self.data
        ):
            raise CompositionPlanError(
                "effect data must contain non-empty string keys and string values"
            )
        if not isinstance(self.temporal_radius, Fraction) or self.temporal_radius < 0:
            raise CompositionPlanError(
                "effect temporal_radius must be a non-negative Fraction"
            )
        if self.input_contract.clock != self.output_contract.clock:
            raise CompositionPlanError(
                "non-retime effect cannot change the semantic frame clock"
            )


@dataclass(frozen=True)
class MaskedEffectGroup:
    """Mask extraction plus ordered inside/outside effect chains and merge."""

    effect_id: PlanId
    path: str
    masks: tuple[MaskOp, ...]
    inverted: bool
    inside: tuple[EffectOp, ...]
    outside: tuple[EffectOp, ...]
    input_contract: FrameContract
    output_contract: FrameContract

    def __post_init__(self) -> None:
        _identifier(self.effect_id, field_name="masked effect id")
        _identifier(self.path, field_name="masked effect path")
        if not self.masks:
            raise CompositionPlanError("masked effect group requires at least one mask")
        for name, chain in (("inside", self.inside), ("outside", self.outside)):
            previous = self.input_contract
            for stage in chain:
                if stage.input_contract != previous:
                    raise CompositionPlanError(
                        f"masked {name} chain has an incompatible input contract"
                    )
                previous = stage.output_contract
            if previous != self.output_contract:
                raise CompositionPlanError(
                    f"masked {name} chain does not reach the group output contract"
                )

    @property
    def temporal_radius(self) -> Fraction:
        return max(
            (
                stage.temporal_radius
                for stage in self.inside + self.outside
            ),
            default=Fraction(0),
        )


@dataclass(frozen=True)
class IgnoredEffectOp:
    """One registry-authorized identity operation with an explicit warning."""

    effect_id: PlanId
    path: str
    handler: str
    reason: str
    input_contract: FrameContract
    output_contract: FrameContract

    def __post_init__(self) -> None:
        _identifier(self.effect_id, field_name="ignored effect id")
        _identifier(self.path, field_name="ignored effect path")
        _identifier(self.handler, field_name="ignored effect handler")
        _identifier(self.reason, field_name="ignored effect reason")
        if self.input_contract != self.output_contract:
            raise CompositionPlanError(
                "warn-and-ignore effect must be an identity frame contract"
            )

    @property
    def temporal_radius(self) -> Fraction:
        return Fraction(0)


EffectStage: TypeAlias = EffectOp | MaskedEffectGroup | IgnoredEffectOp


@dataclass(frozen=True)
class EffectStackPlan:
    """Authored executable, masked, and ignored stages in exact order."""

    stages: tuple[EffectStage, ...] = ()

    def __post_init__(self) -> None:
        ids = tuple(stage.effect_id for stage in self.stages)
        if len(ids) != len(set(ids)):
            raise CompositionPlanError("effect stage IDs must be unique")
        for left, right in zip(self.stages, self.stages[1:]):
            if left.output_contract != right.input_contract:
                raise CompositionPlanError(
                    "adjacent effect stages have incompatible frame contracts"
                )

    @property
    def temporal_radius(self) -> Fraction:
        return max(
            (stage.temporal_radius for stage in self.stages),
            default=Fraction(0),
        )

    @property
    def contains_mask_boundary(self) -> bool:
        return any(isinstance(stage, MaskedEffectGroup) for stage in self.stages)

    @property
    def ignored_stages(self) -> tuple[IgnoredEffectOp, ...]:
        return tuple(
            stage for stage in self.stages if isinstance(stage, IgnoredEffectOp)
        )


@dataclass(frozen=True, order=True)
class ZOrderKey:
    lane: int
    document_order: int

    def __post_init__(self) -> None:
        for field_name in ("lane", "document_order"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise CompositionPlanError(f"{field_name} must be an integer")


@dataclass(frozen=True)
class LayerPlan:
    """One complete source-to-scope semantic pixel operation."""

    layer_id: PlanId
    path: str
    source: SourceRef
    window: OwnedFrameWindow
    z_order: ZOrderKey
    raster: RasterPlacementPlan
    effects: EffectStackPlan
    spatial: SpatialTransformPlan
    opacity: OpacityEnvelopePlan
    blend: BlendModeSpec
    source_contract: FrameContract
    source_retime: RetimeExecutionPlan | None
    input_contract: FrameContract
    output_contract: FrameContract
    execution: LayerExecution = "composite"
    frame_rate_normalization: FrameRateNormalizationPlan | None = None
    raster_spatial_boundary: RasterSpatialBoundaryPlan = field(
        default_factory=RasterSpatialBoundaryPlan
    )

    def __post_init__(self) -> None:
        _identifier(self.layer_id, field_name="layer_id")
        _identifier(self.path, field_name="layer path")
        if not isinstance(self.source, SourceRef):
            raise CompositionPlanError("layer source must be SourceRef")
        if not isinstance(self.window, OwnedFrameWindow):
            raise CompositionPlanError("layer window must be OwnedFrameWindow")
        if not isinstance(self.z_order, ZOrderKey):
            raise CompositionPlanError("layer z_order must be ZOrderKey")
        if not isinstance(self.blend, BlendModeSpec):
            raise CompositionPlanError("layer blend must be BlendModeSpec")
        if self.execution not in {
            "composite",
            "omit_transparent",
            "authored_disabled",
        }:
            raise CompositionPlanError(
                f"unknown layer execution {self.execution!r}"
            )
        if not isinstance(self.raster_spatial_boundary, RasterSpatialBoundaryPlan):
            raise CompositionPlanError(
                "layer raster_spatial_boundary must be RasterSpatialBoundaryPlan"
            )
        if (
            self.execution == "omit_transparent"
            and self.source.kind != "transparent"
        ):
            raise CompositionPlanError(
                "omit_transparent layer must use an explicit transparent source"
            )
        if (
            self.execution == "authored_disabled"
            and self.source.kind not in {"module", "transparent"}
        ):
            raise CompositionPlanError(
                "authored_disabled layer must reference its module or explicit transparency"
            )
        _validate_contract_extent(
            self.input_contract,
            self.window,
            owner=f"layer {self.layer_id} input",
        )
        _validate_contract_window(
            self.output_contract,
            self.window,
            owner=f"layer {self.layer_id} output",
        )
        if self.opacity.window.render_duration != self.window.duration:
            raise CompositionPlanError(
                "layer opacity window must equal the owned layer duration"
            )
        if self.opacity.expression_time_origin != self.window.start:
            raise CompositionPlanError(
                "layer opacity origin must equal the owned layer start"
            )
        if self.source_retime is None:
            if self.source_contract != self.input_contract:
                raise CompositionPlanError(
                    "unretimed layer source and input contracts must match"
                )
        else:
            if self.source_retime.output_duration != self.window.duration:
                raise CompositionPlanError(
                    "layer source retime must produce the owned layer duration"
                )
            if not _same_frame_representation(
                self.source_contract, self.input_contract
            ):
                raise CompositionPlanError(
                    "source retime cannot change surface or pixel representation"
                )
            if (
                self.source_retime.video_frame_duration
                != self.source_contract.clock.frame_duration
                or self.source_retime.video_frame_duration
                != self.input_contract.clock.frame_duration
            ):
                raise CompositionPlanError(
                    "source retime must execute on the source and layer input cadence"
                )
        if self.input_contract != self.raster.input_contract:
            raise CompositionPlanError(
                "layer input contract must match raster input contract"
            )
        if self.frame_rate_normalization is None:
            if self.output_contract != self.spatial.output_contract:
                raise CompositionPlanError(
                    "layer output contract must match spatial output contract"
                )
        else:
            if not isinstance(
                self.frame_rate_normalization, FrameRateNormalizationPlan
            ):
                raise CompositionPlanError(
                    "frame_rate_normalization must be FrameRateNormalizationPlan or None"
                )
            if (
                self.frame_rate_normalization.input_contract
                != self.spatial.output_contract
            ):
                raise CompositionPlanError(
                    "frame-rate normalization input must match spatial output"
                )
            if (
                self.frame_rate_normalization.output_contract
                != self.output_contract
            ):
                raise CompositionPlanError(
                    "layer output must match frame-rate normalization output"
                )
            if (
                self.frame_rate_normalization.frame_grid_origin
                != self.window.frame_grid_origin
            ):
                raise CompositionPlanError(
                    "frame-rate normalization must use the owned output frame grid"
                )
        expected_effect_input = self.raster.output_contract
        if self.effects.stages:
            if self.effects.stages[0].input_contract != expected_effect_input:
                raise CompositionPlanError(
                    "effect stack input does not match raster output contract"
                )
            expected_spatial_input = self.effects.stages[-1].output_contract
        else:
            expected_spatial_input = expected_effect_input
        if self.spatial.input_contract != expected_spatial_input:
            raise CompositionPlanError(
                "effect stack output does not match spatial input contract"
            )
        if self.raster_spatial_boundary.materialization == "preserve_raster_support":
            crop_is_camera = bool(
                self.raster.crop is not None
                and self.raster.crop.enabled
                and self.raster.crop.mode.casefold() in {"crop", "pan"}
            )
            has_conform_camera = self.raster.conform in {"fit", "fill"} or crop_is_camera
            transform = self.spatial.transform
            has_active_spatial = bool(
                (transform is not None and transform.enabled)
                or self.spatial.animation is not None
                or (
                    self.spatial.corner_pin is not None
                    and self.spatial.corner_pin.enabled
                )
            )
            if self.execution != "composite":
                raise CompositionPlanError(
                    "raster support preservation requires composite execution"
                )
            if not has_conform_camera:
                raise CompositionPlanError(
                    "raster support preservation requires conform or camera placement"
                )
            if not has_active_spatial:
                raise CompositionPlanError(
                    "raster support preservation requires an active spatial stage"
                )

    @classmethod
    def resolve_blend(cls, value: str | None) -> BlendModeSpec:
        return resolve_blend_mode(value)


@dataclass(frozen=True)
class TransitionSidePlan:
    """One recursively composed transition side and its handle behavior."""

    composed_sources: tuple[PlanId, ...]
    semantic_handle: OwnedFrameWindow
    source_extension: TransitionExtension

    def __post_init__(self) -> None:
        if not self.composed_sources:
            raise CompositionPlanError("transition side requires composed sources")
        for source in self.composed_sources:
            _identifier(source, field_name="transition side source")
        if len(set(self.composed_sources)) != len(self.composed_sources):
            raise CompositionPlanError("transition side sources must be unique")
        if self.source_extension not in {"none", "hold_first", "hold_last"}:
            raise CompositionPlanError(
                f"unknown transition source extension {self.source_extension!r}"
            )


@dataclass(frozen=True)
class TransitionPlan:
    """One exact replacement of outgoing and incoming composed sides."""

    transition_id: PlanId
    path: str
    window: OwnedFrameWindow
    outgoing: TransitionSidePlan
    incoming: TransitionSidePlan
    z_order: ZOrderKey
    handler: str
    parameters: tuple[tuple[str, str], ...]
    input_contract: FrameContract
    output_contract: FrameContract
    artifact_semantic_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.transition_id, field_name="transition_id")
        _identifier(self.path, field_name="transition path")
        _identifier(self.handler, field_name="transition handler")
        if set(self.outgoing.composed_sources) & set(self.incoming.composed_sources):
            raise CompositionPlanError("transition sides must be disjoint")
        if tuple(sorted(self.parameters)) != self.parameters:
            raise CompositionPlanError(
                "transition parameters must be sorted canonical pairs"
            )
        if self.input_contract.surface != self.output_contract.surface:
            raise CompositionPlanError(
                "transition input and output surfaces must match"
            )
        _validate_contract_window(
            self.input_contract,
            self.window,
            owner=f"transition {self.transition_id} input",
        )
        _validate_contract_window(
            self.output_contract,
            self.window,
            owner=f"transition {self.transition_id} output",
        )
        for name, side in (("outgoing", self.outgoing), ("incoming", self.incoming)):
            handle = side.semantic_handle
            if (
                handle.frame_duration != self.window.frame_duration
                or handle.frame_grid_origin != self.window.frame_grid_origin
            ):
                raise CompositionPlanError(
                    f"{name} transition handle uses another frame grid"
                )
            if (
                handle.first_frame > self.window.first_frame
                or handle.end_frame < self.window.end_frame
            ):
                raise CompositionPlanError(
                    f"{name} transition handle does not cover the transition window"
                )


@dataclass(frozen=True)
class HardCutPlan:
    """One authored transition whose explicit CPU replacement is a hard cut.

    A hard cut is audit metadata, not a composition stack item.  Ordinary
    layer ownership determines the pixels, so this marker never expands
    handles, excludes participants, or introduces interval boundaries. Exact
    neighboring story IDs retain the compiler's local topology without asking
    a backend to search for the nearest visible layer.
    """

    transition_id: PlanId
    path: str
    window: OwnedFrameWindow
    z_order: ZOrderKey
    parameters: tuple[tuple[str, str], ...]
    finding_id: PlanId
    previous_story_id: PlanId | None = None
    next_story_id: PlanId | None = None

    def __post_init__(self) -> None:
        _identifier(self.transition_id, field_name="hard-cut transition_id")
        _identifier(self.path, field_name="hard-cut transition path")
        _identifier(self.finding_id, field_name="hard-cut finding_id")
        if self.previous_story_id is not None:
            _identifier(
                self.previous_story_id,
                field_name="hard-cut previous_story_id",
            )
        if self.next_story_id is not None:
            _identifier(
                self.next_story_id,
                field_name="hard-cut next_story_id",
            )
        if not isinstance(self.window, OwnedFrameWindow):
            raise CompositionPlanError(
                "hard-cut transition window must be OwnedFrameWindow"
            )
        if not isinstance(self.z_order, ZOrderKey):
            raise CompositionPlanError(
                "hard-cut transition z_order must be ZOrderKey"
            )
        if tuple(sorted(self.parameters)) != self.parameters:
            raise CompositionPlanError(
                "hard-cut transition parameters must be sorted canonical pairs"
            )


@dataclass(frozen=True)
class StackItem:
    """One lower-to-upper item in a stable composition interval."""

    kind: StackItemKind
    ref: PlanId
    z_order: ZOrderKey

    def __post_init__(self) -> None:
        if self.kind not in {"layer", "transition"}:
            raise CompositionPlanError(f"unknown stack item kind {self.kind!r}")
        _identifier(self.ref, field_name="stack item reference")


@dataclass(frozen=True)
class CompositionInterval:
    """One exact interval with a stable lower-to-upper item stack."""

    window: OwnedFrameWindow
    stack: tuple[StackItem, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(self.stack, key=lambda item: item.z_order)) != self.stack:
            raise CompositionPlanError("composition stack is not in stable z-order")
        refs = tuple(item.ref for item in self.stack)
        if len(refs) != len(set(refs)):
            raise CompositionPlanError("composition interval contains duplicate items")


def derive_composition_intervals(
    *,
    scope_window: OwnedFrameWindow,
    layers: tuple[LayerPlan, ...],
    transitions: tuple[TransitionPlan, ...],
) -> tuple[CompositionInterval, ...]:
    """Derive the canonical active stack from exact owned windows.

    Main callers:
    - ``CompositionScopePlan`` validation.
    - CPU and Vulkan interval lowerers.

    Why this exists:
    Active layers and transition replacement are compiler semantics.  A
    backend must not independently rebuild them from raw clip timestamps.
    """

    frame_duration = scope_window.frame_duration
    frame_grid_origin = scope_window.frame_grid_origin
    boundaries = {scope_window.first_frame, scope_window.end_frame}
    for layer in layers:
        if (
            layer.window.frame_duration != frame_duration
            or layer.window.frame_grid_origin != frame_grid_origin
        ):
            raise CompositionPlanError("scope layers must share one exact frame grid")
        if (
            layer.window.first_frame < scope_window.first_frame
            or layer.window.end_frame > scope_window.end_frame
        ):
            raise CompositionPlanError("layer window lies outside its scope")
        if layer.execution == "composite":
            boundaries.update((layer.window.first_frame, layer.window.end_frame))
    layer_ids = {layer.layer_id for layer in layers}
    for transition in transitions:
        if (
            transition.window.frame_duration != frame_duration
            or transition.window.frame_grid_origin != frame_grid_origin
        ):
            raise CompositionPlanError(
                "scope transitions must share one exact frame grid"
            )
        participants = (
            transition.outgoing.composed_sources
            + transition.incoming.composed_sources
        )
        if any(participant not in layer_ids for participant in participants):
            raise CompositionPlanError(
                "transition sides must reference layers in the owning scope"
            )
        if any(
            layer.layer_id in participants and layer.execution != "composite"
            for layer in layers
        ):
            raise CompositionPlanError(
                "transition sides cannot reference suppressed layers"
            )
        if (
            transition.window.first_frame < scope_window.first_frame
            or transition.window.end_frame > scope_window.end_frame
        ):
            raise CompositionPlanError("transition window lies outside its scope")
        boundaries.update((transition.window.first_frame, transition.window.end_frame))
    ordered = sorted(boundaries)
    result: list[CompositionInterval] = []
    for first_frame, end_frame in zip(ordered, ordered[1:]):
        if end_frame <= first_frame:
            continue
        active: dict[str, StackItem] = {
            layer.layer_id: StackItem("layer", layer.layer_id, layer.z_order)
            for layer in layers
            if layer.execution == "composite"
            if layer.window.first_frame <= first_frame < layer.window.end_frame
        }
        for transition in transitions:
            if not (
                transition.window.first_frame
                <= first_frame
                < transition.window.end_frame
            ):
                continue
            for participant in (
                transition.outgoing.composed_sources
                + transition.incoming.composed_sources
            ):
                active.pop(participant, None)
            active[transition.transition_id] = StackItem(
                "transition",
                transition.transition_id,
                transition.z_order,
            )
        start = frame_grid_origin + first_frame * frame_duration
        end = frame_grid_origin + end_frame * frame_duration
        window = resolve_owned_frame_window(
            start,
            end,
            frame_duration=frame_duration,
            frame_grid_origin=frame_grid_origin,
        )
        result.append(
            CompositionInterval(
                window=window,
                stack=tuple(sorted(active.values(), key=lambda item: item.z_order)),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class CompositionScopePlan:
    """One nested semantic module with exhaustive canonical intervals."""

    scope_id: PlanId
    path: str
    parent_scope_id: PlanId | None
    canvas: SurfaceSpec
    window: OwnedFrameWindow
    layers: tuple[LayerPlan, ...]
    transitions: tuple[TransitionPlan, ...]
    intervals: tuple[CompositionInterval, ...]
    output_contract: FrameContract
    requires_transparent_intermediate: bool
    hard_cuts: tuple[HardCutPlan, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        _identifier(self.scope_id, field_name="scope_id")
        _identifier(self.path, field_name="scope path")
        if self.parent_scope_id is not None:
            _identifier(self.parent_scope_id, field_name="parent_scope_id")
        layer_ids = tuple(layer.layer_id for layer in self.layers)
        transition_ids = tuple(item.transition_id for item in self.transitions)
        hard_cut_ids = tuple(item.transition_id for item in self.hard_cuts)
        if len(layer_ids) != len(set(layer_ids)):
            raise CompositionPlanError("scope layer IDs must be unique")
        if len(transition_ids) != len(set(transition_ids)):
            raise CompositionPlanError("scope transition IDs must be unique")
        if len(hard_cut_ids) != len(set(hard_cut_ids)):
            raise CompositionPlanError("scope hard-cut IDs must be unique")
        if (
            set(layer_ids) & set(transition_ids)
            or set(layer_ids) & set(hard_cut_ids)
            or set(transition_ids) & set(hard_cut_ids)
        ):
            raise CompositionPlanError(
                "scope layer, transition, and hard-cut IDs must be disjoint"
            )
        if not isinstance(self.enabled, bool):
            raise CompositionPlanError("scope enabled must be bool")
        for hard_cut in self.hard_cuts:
            if (
                hard_cut.window.frame_duration != self.window.frame_duration
                or hard_cut.window.frame_grid_origin
                != self.window.frame_grid_origin
            ):
                raise CompositionPlanError(
                    "scope hard cuts must share one exact frame grid"
                )
            if (
                hard_cut.window.first_frame < self.window.first_frame
                or hard_cut.window.end_frame > self.window.end_frame
            ):
                raise CompositionPlanError(
                    "hard-cut transition window lies outside its scope"
                )
        for index, left in enumerate(self.transitions):
            left_participants = set(
                left.outgoing.composed_sources + left.incoming.composed_sources
            )
            for right in self.transitions[index + 1 :]:
                overlaps_in_time = (
                    left.window.first_frame < right.window.end_frame
                    and right.window.first_frame < left.window.end_frame
                )
                right_participants = set(
                    right.outgoing.composed_sources
                    + right.incoming.composed_sources
                )
                if overlaps_in_time and left_participants & right_participants:
                    raise CompositionPlanError(
                        "overlapping transitions cannot share a participant"
                    )
        expected = derive_composition_intervals(
            scope_window=self.window,
            layers=self.layers,
            transitions=self.transitions,
        )
        if self.intervals != expected:
            raise CompositionPlanError(
                "scope intervals do not match canonical active-layer scheduling"
            )
        if self.output_contract.surface != self.canvas:
            raise CompositionPlanError(
                "scope output contract must match its declared canvas"
            )
        _validate_contract_window(
            self.output_contract,
            self.window,
            owner=f"scope {self.scope_id} output",
        )
        if self.requires_transparent_intermediate and (
            self.output_contract.coverage == "full_opaque"
        ):
            raise CompositionPlanError(
                "transparent scope intermediate cannot claim full opaque coverage"
            )


def build_composition_scope_plan(
    *,
    scope_id: PlanId,
    path: str,
    parent_scope_id: PlanId | None,
    canvas: SurfaceSpec,
    window: OwnedFrameWindow,
    layers: tuple[LayerPlan, ...],
    transitions: tuple[TransitionPlan, ...],
    output_contract: FrameContract,
    requires_transparent_intermediate: bool,
    hard_cuts: tuple[HardCutPlan, ...] = (),
    enabled: bool = True,
) -> CompositionScopePlan:
    """Build a scope with the one canonical interval schedule.

    Main callers:
    - The shadow CPU-plan compiler during the migration.
    - The final shared compiler once CPU emission consumes this IR directly.

    Why this exists:
    Callers may supply semantic layers and transitions, but they cannot supply
    an independently reconstructed active-layer schedule.  That schedule is a
    single shared compiler decision.
    """

    intervals = derive_composition_intervals(
        scope_window=window,
        layers=layers,
        transitions=transitions,
    )
    return CompositionScopePlan(
        scope_id=scope_id,
        path=path,
        parent_scope_id=parent_scope_id,
        canvas=canvas,
        window=window,
        layers=layers,
        transitions=transitions,
        intervals=intervals,
        output_contract=output_contract,
        requires_transparent_intermediate=requires_transparent_intermediate,
        hard_cuts=hard_cuts,
        enabled=enabled,
    )


@dataclass(frozen=True)
class FusionDecision:
    """A derived answer about representation fusion across one module edge."""

    legal: bool
    reasons: tuple[str, ...]


def _inert_module_fusion_reasons(
    owner: LayerPlan,
    module_scope: CompositionScopePlan | None,
) -> tuple[str, ...]:
    """Derive whether one nested module boundary is semantically inert.

    A module is not automatically a fusion barrier, but omitting that boundary
    is legal only when the referenced child scope and every owner stage prove
    identity. This deliberately consumes the child plan rather than accepting
    a Boolean assertion from a backend.
    """

    reasons: list[str] = []
    if module_scope is None or owner.source.ref != module_scope.scope_id:
        return ("nested module scope is unavailable for an inertness proof",)
    if not module_scope.enabled:
        reasons.append("nested module scope is authored disabled")
    active_children = tuple(
        layer for layer in module_scope.layers if layer.execution == "composite"
    )
    if len(active_children) != 1:
        reasons.append("nested module does not contain exactly one active child")
    if module_scope.transitions:
        reasons.append("nested module contains a transition")
    if module_scope.output_contract != owner.source_contract:
        reasons.append(
            "nested module output does not match its owner source contract"
        )
    if any(
        child.output_contract.coverage == "arbitrary_alpha"
        for child in active_children
    ):
        reasons.append("nested module child has arbitrary alpha coverage")
    if owner.source_retime is not None:
        reasons.append("nested module owner has a source-instance retime")
    if owner.frame_rate_normalization is not None:
        reasons.append("nested module owner normalizes frame cadence")
    if owner.source_contract != owner.input_contract:
        reasons.append("nested module owner changes its source contract")
    if (
        owner.raster.crop is not None
        or owner.raster.source_surface != owner.raster.output_surface
        or owner.raster.input_contract != owner.raster.output_contract
    ):
        reasons.append("nested module owner has non-identity raster placement")
    if any(not isinstance(stage, IgnoredEffectOp) for stage in owner.effects.stages):
        reasons.append("nested module owner has an executable effect")
    if (
        owner.spatial.transform is not None
        or owner.spatial.animation is not None
        or owner.spatial.corner_pin is not None
        or owner.spatial.container_conform is not None
        or owner.spatial.input_contract != owner.spatial.output_contract
    ):
        reasons.append("nested module owner has non-identity spatial placement")
    if not owner.opacity.is_constant_one:
        reasons.append("nested module owner has non-identity opacity")
    if owner.blend.family != "normal":
        reasons.append("nested module owner has non-Normal blending")
    return tuple(reasons)


def decide_fusion(
    source: FrameContract,
    target: FrameContract,
    *,
    stable_active_set: bool,
    boundary: LayerPlan | TransitionPlan | CompositionScopePlan | None,
    reviewed_conversion_id: str | None = None,
    module_scope: CompositionScopePlan | None = None,
) -> FusionDecision:
    """Return whether two modules may share one physical execution island."""

    reasons: list[str] = []
    if not stable_active_set:
        reasons.append("active layer set changes inside the proposed fusion")
    if isinstance(boundary, TransitionPlan):
        reasons.append("transition side/result is an initial ownership barrier")
    elif isinstance(boundary, CompositionScopePlan):
        reasons.append("nested composition output is an initial ownership barrier")
    elif isinstance(boundary, LayerPlan):
        if boundary.source.kind == "module":
            reasons.extend(_inert_module_fusion_reasons(boundary, module_scope))
        elif module_scope is not None:
            reasons.append("module scope was supplied for a non-module layer")
        if boundary.raster.requires_transparent_plate:
            reasons.append("crop/conform requires a preserved transparent plate")
        if boundary.spatial.requires_transparent_intermediate:
            reasons.append("spatial transform requires a transparent intermediate")
        if boundary.effects.contains_mask_boundary:
            reasons.append("mask extraction/merge is an initial ownership barrier")
        if boundary.effects.temporal_radius != 0:
            reasons.append("temporal/history operation is an initial fusion barrier")

    # Surface geometry and time ownership are semantic stages, never mere
    # representation conversions. A reviewed transfer shader cannot waive
    # them by name.
    if source.surface != target.surface:
        reasons.append("surface dimensions or origin differ")
    if source.clock != target.clock:
        reasons.append("frame clock or PTS origin differs")
    if source.primaries != target.primaries or source.matrix != target.matrix:
        reasons.append("color primaries or matrix differ")
    representation_matches = (
        source.transfer,
        source.range,
        source.alpha,
        source.precision,
        source.coverage,
    ) == (
        target.transfer,
        target.range,
        target.alpha,
        target.precision,
        target.coverage,
    )
    if not representation_matches and reviewed_conversion_id is None:
        reasons.append(
            "pixel representation differs without a reviewed fused conversion"
        )
    if reviewed_conversion_id is not None:
        _identifier(reviewed_conversion_id, field_name="reviewed_conversion_id")
    return FusionDecision(not reasons, tuple(reasons))


@dataclass(frozen=True)
class VideoDispositionFinding:
    """One explicit degraded video semantic and its exact replacement.

    Main callers:
    - The shared compiler when an authored video construct cannot execute.
    - Compatibility-report and manifest writers.

    Why this exists:
    A missing raster, ignored effect, or unsupported transition must remain a
    typed part of the semantic plan.  The replacement says what pixels the
    backend must produce; the remaining fields preserve the user-facing
    compatibility evidence without relying on free-form FFmpeg text.
    """

    finding_id: PlanId
    target_id: PlanId
    path: str
    construct: str
    replacement: VideoReplacement
    reason: str
    portable_status: str = "unsupported"
    outcome: Literal["omitted"] = "omitted"
    uid: str | None = None
    authored_start: Fraction | None = None
    authored_duration: Fraction | None = None

    def __post_init__(self) -> None:
        _identifier(self.finding_id, field_name="video finding_id")
        _identifier(self.target_id, field_name="video finding target_id")
        _identifier(self.path, field_name="video finding path")
        _identifier(self.construct, field_name="video finding construct")
        _identifier(self.reason, field_name="video finding reason")
        _identifier(self.portable_status, field_name="video finding portable_status")
        if self.replacement not in {"transparent", "identity", "hard_cut"}:
            raise CompositionPlanError(
                f"unknown video replacement {self.replacement!r}"
            )
        if self.outcome != "omitted":
            raise CompositionPlanError(
                "video disposition finding outcome must be omitted"
            )
        if self.uid is not None:
            _identifier(self.uid, field_name="video finding uid")
        if self.authored_start is not None and not isinstance(
            self.authored_start, Fraction
        ):
            raise CompositionPlanError(
                "video finding authored_start must be Fraction or None"
            )
        if self.authored_duration is not None:
            if not isinstance(self.authored_duration, Fraction):
                raise CompositionPlanError(
                    "video finding authored_duration must be Fraction or None"
                )
            if self.authored_duration < 0:
                raise CompositionPlanError(
                    "video finding authored_duration cannot be negative"
                )


@dataclass(frozen=True)
class IgnoredEffectFinding:
    """Legacy three-field identity finding accepted during shadow migration."""

    path: str
    handler: str
    reason: str

    def __post_init__(self) -> None:
        _identifier(self.path, field_name="ignored effect path")
        _identifier(self.handler, field_name="ignored effect handler")
        _identifier(self.reason, field_name="ignored effect reason")


@dataclass(frozen=True)
class CompositionPlan:
    """Complete, acyclic, machine-independent video semantics for one render."""

    schema_version: int
    document_source_sha256: str
    project_canvas: SurfaceSpec
    project_clock: FrameClock
    decoders: tuple[DecoderSourcePlan, ...]
    rasters: tuple[RasterSourcePlan, ...]
    scopes: tuple[CompositionScopePlan, ...]
    root_scope_id: PlanId
    video_findings: tuple[VideoDispositionFinding, ...] = ()
    ignored_findings: tuple[IgnoredEffectFinding, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise CompositionPlanError("unsupported composition plan schema version")
        _sha256(self.document_source_sha256, field_name="document_source_sha256")
        _identifier(self.root_scope_id, field_name="root_scope_id")
        decoder_ids = tuple(item.identity.source_id for item in self.decoders)
        raster_ids = tuple(item.identity.source_id for item in self.rasters)
        scope_ids = tuple(item.scope_id for item in self.scopes)
        all_ids = decoder_ids + raster_ids + scope_ids
        if len(all_ids) != len(set(all_ids)):
            raise CompositionPlanError("composition plan IDs must be globally unique")
        if self.root_scope_id not in set(scope_ids):
            raise CompositionPlanError("root_scope_id does not identify a scope")
        scope_by_id = {scope.scope_id: scope for scope in self.scopes}
        root = scope_by_id[self.root_scope_id]
        if root.parent_scope_id is not None:
            raise CompositionPlanError("root scope cannot have a parent")
        if root.canvas != self.project_canvas:
            raise CompositionPlanError("root scope canvas must match project canvas")
        if root.output_contract.clock != self.project_clock:
            raise CompositionPlanError("root output clock must match project clock")
        decoder_by_id = {
            source.identity.source_id: source for source in self.decoders
        }
        raster_by_id = {
            source.identity.source_id: source for source in self.rasters
        }
        scope_position = {scope.scope_id: index for index, scope in enumerate(self.scopes)}
        child_owner_count = {scope.scope_id: 0 for scope in self.scopes}
        ignored_stages: dict[str, IgnoredEffectOp] = {}
        expected_dispositions: dict[str, tuple[VideoReplacement, str, str | None]] = {}
        expected_finding_ids: dict[str, str] = {}

        def register_disposition(
            target_id: str,
            replacement: VideoReplacement,
            path: str,
            reason: str | None,
            *,
            finding_id: str | None = None,
        ) -> None:
            if target_id in expected_dispositions:
                raise CompositionPlanError(
                    f"video disposition target {target_id!r} is duplicated"
                )
            expected_dispositions[target_id] = (replacement, path, reason)
            if finding_id is not None:
                expected_finding_ids[target_id] = finding_id

        for scope in self.scopes:
            if scope.parent_scope_id is not None:
                if scope.parent_scope_id not in scope_by_id:
                    raise CompositionPlanError(
                        f"scope {scope.scope_id} has an unknown parent"
                    )
                if scope_position[scope.scope_id] >= scope_position[scope.parent_scope_id]:
                    raise CompositionPlanError(
                        "nested scopes must be ordered children before parents"
                    )
            for layer in scope.layers:
                if layer.source.kind == "decoder":
                    if layer.source.ref not in decoder_by_id:
                        raise CompositionPlanError(
                            f"layer {layer.layer_id} references an unknown decoder"
                        )
                    assert layer.source.ref is not None
                    decoder_source = decoder_by_id[layer.source.ref]
                    if layer.source_contract != decoder_source.frame_contract:
                        raise CompositionPlanError(
                            "decoder source contract does not match layer source edge"
                        )
                    if layer.window != decoder_source.output_window:
                        raise CompositionPlanError(
                            "decoder instance output window does not match its layer"
                        )
                elif layer.source.kind in {"still", "runtime_raster"}:
                    if layer.source.ref not in raster_by_id:
                        raise CompositionPlanError(
                            f"layer {layer.layer_id} references an unknown raster"
                        )
                    assert layer.source.ref is not None
                    raster_source = raster_by_id[layer.source.ref]
                    if raster_source.kind != layer.source.kind:
                        raise CompositionPlanError(
                            "raster source kind does not match its layer reference"
                        )
                    if layer.source_contract != raster_source.frame_contract:
                        raise CompositionPlanError(
                            "raster source contract does not match layer source edge"
                        )
                    if layer.window != raster_source.output_window:
                        raise CompositionPlanError(
                            "raster instance output window does not match its layer"
                        )
                elif layer.source.kind == "module":
                    assert layer.source.ref is not None
                    if layer.source.ref not in scope_by_id:
                        raise CompositionPlanError(
                            f"layer {layer.layer_id} references an unknown module"
                        )
                    child = scope_by_id[layer.source.ref]
                    if child.parent_scope_id != scope.scope_id:
                        raise CompositionPlanError(
                            "module source and child parent references are not reciprocal"
                        )
                    child_owner_count[child.scope_id] += 1
                    if scope_position[child.scope_id] >= scope_position[scope.scope_id]:
                        raise CompositionPlanError(
                            "module source must reference an earlier child scope"
                        )
                    if layer.source_contract != child.output_contract:
                        raise CompositionPlanError(
                            "child scope output contract does not match module source edge"
                        )
                    expected_execution = (
                        "composite" if child.enabled else "authored_disabled"
                    )
                    if layer.execution != expected_execution:
                        raise CompositionPlanError(
                            "module layer execution must match its child scope enabled state"
                        )
                elif layer.source.kind == "transparent":
                    if layer.source_contract.coverage == "full_opaque":
                        raise CompositionPlanError(
                            "transparent source cannot claim full opaque coverage"
                        )
                if layer.execution == "omit_transparent":
                    assert layer.source.omission_reason is not None
                    register_disposition(
                        layer.layer_id,
                        "transparent",
                        layer.path,
                        layer.source.omission_reason,
                    )
                for stage in layer.effects.ignored_stages:
                    if stage.effect_id in ignored_stages:
                        raise CompositionPlanError(
                            f"ignored effect target {stage.effect_id!r} is duplicated"
                        )
                    ignored_stages[stage.effect_id] = stage
                    register_disposition(
                        stage.effect_id,
                        "identity",
                        stage.path,
                        stage.reason,
                    )
            for hard_cut in scope.hard_cuts:
                register_disposition(
                    hard_cut.transition_id,
                    "hard_cut",
                    hard_cut.path,
                    None,
                    finding_id=hard_cut.finding_id,
                )
        for scope_id, count in child_owner_count.items():
            expected = 0 if scope_id == self.root_scope_id else 1
            if count != expected:
                raise CompositionPlanError(
                    f"scope {scope_id} must appear in exactly {expected} parent layer(s)"
                )
        finding_ids = tuple(finding.finding_id for finding in self.video_findings)
        finding_targets = tuple(finding.target_id for finding in self.video_findings)
        if len(finding_ids) != len(set(finding_ids)):
            raise CompositionPlanError("video finding IDs must be unique")
        if len(finding_targets) != len(set(finding_targets)):
            raise CompositionPlanError("video finding targets must be unique")
        finding_by_target = {
            finding.target_id: finding for finding in self.video_findings
        }
        legacy_finding_pairs = {
            (finding.path, finding.handler, finding.reason)
            for finding in self.ignored_findings
        }
        legacy_covered_targets = {
            effect_id
            for effect_id, stage in ignored_stages.items()
            if (stage.path, stage.handler, stage.reason) in legacy_finding_pairs
        }
        if legacy_covered_targets & set(finding_by_target):
            raise CompositionPlanError(
                "identity stage cannot have both legacy and general findings"
            )
        expected_general_targets = set(expected_dispositions) - legacy_covered_targets
        if set(finding_by_target) != expected_general_targets:
            missing = sorted(expected_general_targets - set(finding_by_target))
            orphaned = sorted(set(finding_by_target) - expected_general_targets)
            detail: list[str] = []
            if missing:
                detail.append("missing targets: " + ", ".join(missing))
            if orphaned:
                detail.append("orphan targets: " + ", ".join(orphaned))
            raise CompositionPlanError(
                "video findings do not exactly match degraded semantic targets ("
                + "; ".join(detail)
                + ")"
            )
        for target_id in expected_general_targets:
            replacement, path, reason = expected_dispositions[target_id]
            finding = finding_by_target[target_id]
            if finding.replacement != replacement or finding.path != path:
                raise CompositionPlanError(
                    f"video finding for {target_id} does not match its semantic replacement"
                )
            if reason is not None and finding.reason != reason:
                raise CompositionPlanError(
                    f"video finding for {target_id} does not match its semantic reason"
                )
            expected_finding_id = expected_finding_ids.get(target_id)
            if (
                expected_finding_id is not None
                and finding.finding_id != expected_finding_id
            ):
                raise CompositionPlanError(
                    f"video finding for {target_id} does not match its finding_id reference"
                )

        new_identity_targets = {
            finding.target_id
            for finding in self.video_findings
            if finding.replacement == "identity"
        }
        legacy_expected_pairs = {
            (stage.path, stage.handler, stage.reason)
            for effect_id, stage in ignored_stages.items()
            if effect_id not in new_identity_targets
        }
        if legacy_expected_pairs != legacy_finding_pairs:
            raise CompositionPlanError(
                "legacy warn-and-ignore findings do not match uncovered identity stages"
            )

        camera_sample_count = sum(
            len(layer.raster.camera_sampling.samples)
            for scope in self.scopes
            for layer in scope.layers
            if layer.raster.camera_sampling is not None
        )
        if camera_sample_count > MAX_COMPOSITION_CAMERA_SAMPLES:
            raise CompositionPlanError(
                "composition camera sample count exceeds the plan limit of "
                f"{MAX_COMPOSITION_CAMERA_SAMPLES} samples"
            )

    def _manifest_unchecked(self) -> dict[str, object]:
        """Build stable semantics before the one canonical size check."""

        return {
            "schema": "bladeworks.composition-plan.v1",
            "document_source_sha256": self.document_source_sha256,
            "project_canvas": self.project_canvas.manifest(),
            "project_clock": self.project_clock.manifest(),
            "root_scope_id": self.root_scope_id,
            "decoders": [
                {
                    "source_id": source.identity.source_id,
                    "clip_id": source.identity.clip_id,
                    "resource_id": source.identity.resource_id,
                    "content_sha256": source.identity.content_sha256,
                    "encoded_probe_surface": source.encoded_probe_surface.manifest(),
                    "display_surface": source.display_surface.manifest(),
                    "spatial_intrinsics": _semantic_value(
                        source.spatial_intrinsics
                    ),
                    "output_window": _window_manifest(source.output_window),
                    "decode_window": {
                        "semantic_start": _fraction_text(
                            source.decode_window.semantic_start
                        ),
                        "semantic_end": _fraction_text(
                            source.decode_window.semantic_end
                        ),
                        "decode_start": _fraction_text(
                            source.decode_window.decode_start
                        ),
                        "decode_end": _fraction_text(source.decode_window.decode_end),
                        "frame_duration": _fraction_text(
                            source.decode_window.frame_duration
                        ),
                        "frame_grid_origin": _fraction_text(
                            source.decode_window.frame_grid_origin
                        ),
                        "first_frame": source.decode_window.first_frame,
                        "end_frame": source.decode_window.end_frame,
                    },
                    "first_sample": {
                        "source_time": _fraction_text(source.first_sample.source_time),
                        "frame_duration": _fraction_text(
                            source.first_sample.frame_duration
                        ),
                        "frame_grid_origin": _fraction_text(
                            source.first_sample.frame_grid_origin
                        ),
                        "source_start_frame": source.first_sample.source_start_frame,
                        "source_frame_start": _fraction_text(
                            source.first_sample.source_frame_start
                        ),
                        "source_phase": _fraction_text(
                            source.first_sample.source_phase
                        ),
                        "direction": source.first_sample.direction,
                    },
                    "retime_manifest_sha256": (
                        source.retime.manifest_sha256
                        if source.retime is not None
                        else None
                    ),
                    **(
                        {
                            "cadence_normalization": _semantic_value(
                                source.cadence_normalization
                            )
                        }
                        if source.cadence_normalization is not None
                        else {}
                    ),
                    "frame_contract": source.frame_contract.manifest(),
                }
                for source in self.decoders
            ],
            "rasters": [
                {
                    "source_id": source.identity.source_id,
                    "clip_id": source.identity.clip_id,
                    "kind": source.kind,
                    "resource_id": source.identity.resource_id,
                    "content_sha256": source.identity.content_sha256,
                    "output_window": _window_manifest(source.output_window),
                    "frame_contract": source.frame_contract.manifest(),
                }
                for source in self.rasters
            ],
            "scopes": [
                {
                    "scope_id": scope.scope_id,
                    "path": scope.path,
                    "parent_scope_id": scope.parent_scope_id,
                    "enabled": scope.enabled,
                    "canvas": scope.canvas.manifest(),
                    "window": _window_manifest(scope.window),
                    "requires_transparent_intermediate": (
                        scope.requires_transparent_intermediate
                    ),
                    "layers": [
                        {
                            "layer_id": layer.layer_id,
                            "path": layer.path,
                            "execution": layer.execution,
                            "source": _semantic_value(layer.source),
                            "window": _window_manifest(layer.window),
                            "z_order": [
                                layer.z_order.lane,
                                layer.z_order.document_order,
                            ],
                            "raster": _raster_placement_manifest(layer.raster),
                            "effects": _semantic_value(layer.effects),
                            "spatial": _semantic_value(layer.spatial),
                            "opacity": _semantic_value(layer.opacity),
                            "blend": _semantic_value(layer.blend),
                            "source_contract": layer.source_contract.manifest(),
                            "source_retime_manifest_sha256": (
                                layer.source_retime.manifest_sha256
                                if layer.source_retime is not None
                                else None
                            ),
                            "frame_rate_normalization": _semantic_value(
                                layer.frame_rate_normalization
                            ),
                            **(
                                {
                                    "raster_spatial_boundary": _semantic_value(
                                        layer.raster_spatial_boundary
                                    )
                                }
                                if layer.raster_spatial_boundary.materialization
                                == "preserve_raster_support"
                                else {}
                            ),
                            "input_contract": layer.input_contract.manifest(),
                            "output_contract": layer.output_contract.manifest(),
                        }
                        for layer in scope.layers
                    ],
                    "transitions": [
                        {
                            "transition_id": transition.transition_id,
                            "path": transition.path,
                            "window": _window_manifest(transition.window),
                            "outgoing": {
                                "composed_sources": list(
                                    transition.outgoing.composed_sources
                                ),
                                "semantic_handle": _window_manifest(
                                    transition.outgoing.semantic_handle
                                ),
                                "source_extension": (
                                    transition.outgoing.source_extension
                                ),
                            },
                            "incoming": {
                                "composed_sources": list(
                                    transition.incoming.composed_sources
                                ),
                                "semantic_handle": _window_manifest(
                                    transition.incoming.semantic_handle
                                ),
                                "source_extension": (
                                    transition.incoming.source_extension
                                ),
                            },
                            "z_order": [
                                transition.z_order.lane,
                                transition.z_order.document_order,
                            ],
                            "handler": transition.handler,
                            "parameters": [list(item) for item in transition.parameters],
                            "artifact_semantic_id": transition.artifact_semantic_id,
                            "input_contract": transition.input_contract.manifest(),
                            "output_contract": transition.output_contract.manifest(),
                        }
                        for transition in scope.transitions
                    ],
                    "hard_cuts": [
                        {
                            "transition_id": hard_cut.transition_id,
                            "path": hard_cut.path,
                            "window": _window_manifest(hard_cut.window),
                            "z_order": [
                                hard_cut.z_order.lane,
                                hard_cut.z_order.document_order,
                            ],
                            "parameters": [
                                list(item) for item in hard_cut.parameters
                            ],
                            "finding_id": hard_cut.finding_id,
                            "previous_story_id": hard_cut.previous_story_id,
                            "next_story_id": hard_cut.next_story_id,
                        }
                        for hard_cut in scope.hard_cuts
                    ],
                    "intervals": [
                        {
                            "window": _window_manifest(interval.window),
                            "stack": [
                                {
                                    "kind": item.kind,
                                    "ref": item.ref,
                                    "z": [
                                        item.z_order.lane,
                                        item.z_order.document_order,
                                    ],
                                }
                                for item in interval.stack
                            ],
                        }
                        for interval in scope.intervals
                    ],
                    "output_contract": scope.output_contract.manifest(),
                }
                for scope in self.scopes
            ],
            "video_findings": [
                {
                    "finding_id": finding.finding_id,
                    "target_id": finding.target_id,
                    "path": finding.path,
                    "construct": finding.construct,
                    "replacement": finding.replacement,
                    "reason": finding.reason,
                    "portable_status": finding.portable_status,
                    "outcome": finding.outcome,
                    "uid": finding.uid,
                    "authored_start": (
                        _fraction_text(finding.authored_start)
                        if finding.authored_start is not None
                        else None
                    ),
                    "authored_duration": (
                        _fraction_text(finding.authored_duration)
                        if finding.authored_duration is not None
                        else None
                    ),
                }
                for finding in self.video_findings
            ],
            "ignored_findings": [
                {
                    "path": finding.path,
                    "handler": finding.handler,
                    "reason": finding.reason,
                }
                for finding in self.ignored_findings
            ],
        }

    def _manifest_payload(self) -> tuple[dict[str, object], bytes]:
        """Return the canonical bounded manifest and its encoded bytes.

        Main callers:
        - :meth:`manifest` before any caller receives JSON-safe plan data.
        - :attr:`manifest_sha256` so hashing and external serialization enforce
          the exact same byte ceiling.

        Why this exists:
        Per-record limits bound the common allocation paths, while identifiers
        and the number of ordinary semantic records can still grow.  Measuring
        the canonical compact JSON closes that aggregate resource boundary
        before an executor writes a manifest to disk.
        """

        manifest = self._manifest_unchecked()
        payload = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > MAX_COMPOSITION_MANIFEST_BYTES:
            raise CompositionPlanError(
                "composition manifest exceeds the canonical JSON limit of "
                f"{MAX_COMPOSITION_MANIFEST_BYTES} bytes"
            )
        return manifest, payload

    def manifest(self) -> dict[str, object]:
        """Return bounded stable semantics without runtime execution state."""

        manifest, _payload = self._manifest_payload()
        return manifest

    @property
    def manifest_sha256(self) -> str:
        _manifest, payload = self._manifest_payload()
        return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ContainerConformPlan",
    "CompositionInterval",
    "CompositionPlan",
    "CompositionPlanError",
    "CompositionScopePlan",
    "DecoderCadenceNormalizationPlan",
    "DecoderBinding",
    "DecoderSourcePlan",
    "EffectOp",
    "EffectStackPlan",
    "FrameContract",
    "FrameRateNormalizationPlan",
    "FusionDecision",
    "HardCutPlan",
    "IgnoredEffectOp",
    "IgnoredEffectFinding",
    "LayerPlan",
    "MAX_COMPOSITION_CAMERA_SAMPLES",
    "MAX_COMPOSITION_FRAME_COUNT",
    "MAX_COMPOSITION_MANIFEST_BYTES",
    "MAX_COMPOSITION_SURFACE_DIMENSION",
    "MAX_COMPOSITION_SURFACE_PIXELS",
    "MAX_RASTER_CAMERA_SAMPLES",
    "MaskOp",
    "MaskedEffectGroup",
    "NominalSamplingContract",
    "OpacityEnvelopePlan",
    "RasterPlacementPlan",
    "RasterCameraSamplingPlan",
    "RasterSpatialBoundaryPlan",
    "ResolvedCameraSample",
    "RasterSourcePlan",
    "RuntimeSourceBinding",
    "SourceIdentity",
    "SourceAlphaWindowPlan",
    "SourceRef",
    "SamplingRect",
    "SpatialTransformPlan",
    "StackItem",
    "SurfaceSpec",
    "TransitionPlan",
    "TransitionSidePlan",
    "VideoDispositionFinding",
    "ZOrderKey",
    "build_composition_scope_plan",
    "decide_fusion",
    "derive_composition_intervals",
]
