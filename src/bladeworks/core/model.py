"""Models shared by the FCPXML parser, compiler, and FFmpeg executor.

Architecture map
================

``SourceDocument`` is the parse result. It mirrors resources, synchronized
multicam angle storylines, and nested project story nodes closely enough to
diagnose unsupported Final Cut constructs.

``RenderDocument`` is the resolved timeline. It contains only renderer-facing
values: absolute rational time, concrete media paths, stack order, and selected
portable handler IDs.

Important invariant: every timeline value remains ``fractions.Fraction`` until
it is formatted for FFmpeg. Geometry and effect scalars may be floats because
they are not timeline coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Optional

if TYPE_CHECKING:
    from .animation import AnimationNotice, TimelineAnimatedScalar, TimelineAnimatedVec2
    from .audio_ir import AudioRenderPlan
    from .composition_ir import CompositionPlan
    from .geometry import CornerPinAdjustment
    from .retime import RetimeMap
    from .spatial_intrinsics import SpatialIntrinsicPlan
    from .story_ir import RenderStoryPlan
    from .text_templates import GeneratorRenderPlan, TextRenderPlan


SCHEMA_VERSION = 2


def parse_time(raw: Optional[str], *, required: bool = False, field_name: str = "time") -> Optional[Fraction]:
    """Parse an FCPXML seconds value without passing through floating point."""

    if raw is None or not raw.strip():
        if required:
            raise ValueError(f"missing required {field_name}")
        return None
    text = raw.strip()
    if not text.endswith("s"):
        raise ValueError(f"invalid {field_name} {raw!r}: expected FCPXML seconds syntax")
    body = text[:-1]
    try:
        return Fraction(body)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid {field_name} {raw!r}") from exc


def fraction_text(value: Fraction) -> str:
    """Return a deterministic rational-seconds string accepted by FFmpeg."""

    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def fraction_json(value: Optional[Fraction]) -> Optional[dict[str, int]]:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


def pair(raw: Optional[str], default: tuple[float, float]) -> tuple[float, float]:
    if raw is None:
        return default
    parts = raw.replace(",", " ").split()
    if len(parts) < 2:
        raise ValueError(f"expected two numeric components, got {raw!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"expected two numeric components, got {raw!r}") from exc


@dataclass(frozen=True)
class FormatResource:
    id: str
    name: Optional[str]
    frame_duration: Optional[Fraction]
    width: Optional[int]
    height: Optional[int]
    color_space: Optional[str]
    field_order: Optional[str] = None
    pixel_aspect_h: Optional[int] = None
    pixel_aspect_v: Optional[int] = None
    projection: Optional[str] = None
    stereoscopic: Optional[str] = None
    hero_eye: Optional[str] = None


@dataclass(frozen=True)
class SequenceFormatContext:
    """One sequence's complete local raster and timing coordinate system.

    Main callers:
    - Compound-resource parsing records this on ``ResourceStory``.
    - The compiler assigns it to descendants and their completed group scope.

    Why this exists:
    A reusable sequence can be vertical, anamorphic, or use a different frame
    cadence from its parent. Width and height alone cannot identify the local
    coordinate system that owns its internal geometry.
    """

    format_id: str
    frame_duration: Fraction
    width: int
    height: int
    color_space: Optional[str]
    field_order: Optional[str] = None
    pixel_aspect_h: Optional[int] = None
    pixel_aspect_v: Optional[int] = None
    projection: Optional[str] = None
    stereoscopic: Optional[str] = None
    hero_eye: Optional[str] = None

    @classmethod
    def from_resource(cls, resource: FormatResource) -> "SequenceFormatContext":
        if resource.frame_duration is None or not resource.width or not resource.height:
            raise ValueError(f"format {resource.id!r} is incomplete")
        return cls(
            format_id=resource.id,
            frame_duration=resource.frame_duration,
            width=resource.width,
            height=resource.height,
            color_space=resource.color_space,
            field_order=resource.field_order,
            pixel_aspect_h=resource.pixel_aspect_h,
            pixel_aspect_v=resource.pixel_aspect_v,
            projection=resource.projection,
            stereoscopic=resource.stereoscopic,
            hero_eye=resource.hero_eye,
        )


@dataclass(frozen=True)
class MediaRepresentation:
    kind: Optional[str]
    src: Optional[str]


@dataclass(frozen=True)
class AssetResource:
    id: str
    name: Optional[str]
    uid: Optional[str]
    start: Fraction
    duration: Optional[Fraction]
    has_video: bool
    has_audio: bool
    format_id: Optional[str]
    media_representations: tuple[MediaRepresentation, ...]
    raw_xml: str
    video_sources: Optional[int] = None
    audio_sources: Optional[int] = None
    audio_channels: Optional[int] = None
    audio_rate: Optional[int] = None
    custom_lut_override: Optional[str] = None
    color_space_override: Optional[str] = None
    projection_override: Optional[str] = None
    stereoscopic_override: Optional[str] = None
    hero_eye_override: Optional[str] = None


@dataclass(frozen=True)
class EffectResource:
    id: str
    name: Optional[str]
    uid: Optional[str]
    src: Optional[str]
    raw_xml: str


@dataclass(frozen=True)
class OtherResource:
    id: Optional[str]
    kind: str
    name: Optional[str]
    uid: Optional[str]
    raw_xml: str


@dataclass(frozen=True)
class MulticamSource:
    """One timeline selection inside an ``mc-clip``.

    ``child_kinds`` preserves the presence of per-angle adjustments that this
    bounded prototype cannot yet translate.  The compiler reports those
    constructs instead of silently dropping them.
    """

    angle_id: str
    src_enable: str
    child_kinds: tuple[str, ...]
    raw_xml: str


@dataclass(frozen=True)
class MulticamAngle:
    name: Optional[str]
    angle_id: str
    story: tuple["StoryNode", ...]
    raw_xml: str


@dataclass(frozen=True)
class MulticamResource:
    """A preserved Final Cut multicam resource resolved for portable playback."""

    id: str
    name: Optional[str]
    uid: Optional[str]
    format_id: str
    tc_start: Fraction
    duration: Optional[Fraction]
    angles: tuple[MulticamAngle, ...]
    raw_xml: str


@dataclass(frozen=True)
class Parameter:
    name: Optional[str]
    key: Optional[str]
    value: Optional[str]
    keyframes: tuple["Keyframe", ...] = ()


@dataclass(frozen=True)
class FilterInstance:
    kind: str
    ref: Optional[str]
    name: Optional[str]
    enabled: bool
    params: tuple[Parameter, ...]
    data: Mapping[str, str]
    raw_xml: str


@dataclass(frozen=True)
class MaskSource:
    """One FCPXML mask primitive preserved before portable classification."""

    kind: str
    name: Optional[str]
    enabled: bool
    blend_mode: str
    mask_type: Optional[str]
    tracking: Optional[str]
    params: tuple[Parameter, ...]
    data: Optional[str]
    raw_xml: str


@dataclass(frozen=True)
class MaskedFilterInstance:
    """A ``filter-video-mask`` and its ordered inside/outside filters."""

    kind: str
    enabled: bool
    inverted: bool
    masks: tuple[MaskSource, ...]
    filters: tuple[FilterInstance, ...]
    raw_xml: str


@dataclass(frozen=True)
class PreservedAdjustment:
    """One render-affecting child preserved before backend support is decided.

    The parser keeps intrinsic and component-level adjustments even when no
    portable handler exists.  The compiler can then emit a timeline-scoped
    compatibility finding instead of silently losing active Final Cut
    behavior.
    """

    kind: str
    enabled: bool
    attributes: Mapping[str, str]
    params: tuple["Parameter", ...]
    raw_xml: str


@dataclass(frozen=True)
class Keyframe:
    time: Fraction
    value: str
    interp: Optional[str]
    curve: Optional[str]
    aux_value: Optional[str] = None


@dataclass(frozen=True)
class TransformAdjustment:
    position: tuple[float, float]
    scale: tuple[float, float]
    rotation: float
    enabled: bool
    anchor: tuple[float, float] = (0.0, 0.0)
    tracking_ref: Optional[str] = None
    position_keyframes: tuple[Keyframe, ...] = ()
    scale_keyframes: tuple[Keyframe, ...] = ()
    rotation_keyframes: tuple[Keyframe, ...] = ()
    anchor_keyframes: tuple[Keyframe, ...] = ()


@dataclass(frozen=True)
class CropRect:
    left: float
    top: float
    right: float
    bottom: float
    kind: str = "active"


@dataclass(frozen=True)
class CropAdjustment:
    mode: str
    enabled: bool
    rects: tuple[CropRect, ...]

    @property
    def active_rects(self) -> tuple[CropRect, ...]:
        """Return only rectangle elements belonging to the active crop mode.

        Final Cut may serialize crop, trim, and both pan rectangles together.
        Parser-owned records retain those element names so selection never
        depends on document order. ``active`` is reserved for isolated plans
        that construct only the already-selected rectangles.
        """

        expected = f"{self.mode}-rect"
        typed = tuple(rect for rect in self.rects if rect.kind == expected)
        synthetic = tuple(rect for rect in self.rects if rect.kind == "active")
        if typed and synthetic:
            raise ValueError("crop adjustment mixes typed and synthetic rectangles")
        if typed:
            return typed
        known = {"crop-rect", "trim-rect", "pan-rect"}
        if any(rect.kind in known for rect in self.rects):
            return ()
        return synthetic


@dataclass(frozen=True)
class FadeEnvelope:
    fade_in: Optional[Fraction]
    fade_in_type: Optional[str]
    fade_out: Optional[Fraction]
    fade_out_type: Optional[str]


@dataclass(frozen=True)
class TimeMapPoint:
    time: Fraction
    value: Fraction
    interp: Optional[str]


@dataclass(frozen=True)
class TextStyle:
    id: Optional[str]
    font: Optional[str]
    font_face: Optional[str]
    font_size: Optional[float]
    font_color: Optional[str]
    alignment: Optional[str]
    stroke_color: Optional[str]
    stroke_width: Optional[float]
    tracking: Optional[float]
    bold: bool
    italic: bool


@dataclass(frozen=True)
class TextRun:
    text: str
    style_ref: Optional[str]
    inline_style: Optional[TextStyle]


@dataclass(frozen=True)
class StoryNode:
    kind: str
    path: str
    name: Optional[str]
    ref: Optional[str]
    lane: int
    offset: Optional[Fraction]
    start: Fraction
    duration: Fraction
    enabled: bool
    src_enable: Optional[str]
    audio_start: Optional[Fraction]
    audio_duration: Optional[Fraction]
    role: Optional[str]
    video_role: Optional[str]
    audio_role: Optional[str]
    conform_type: str
    transform: Optional[TransformAdjustment]
    crop: Optional[CropAdjustment]
    blend_opacity: float
    blend_mode: Optional[str]
    opacity_fade: Optional[FadeEnvelope]
    volume_db: Optional[float]
    audio_fade: Optional[FadeEnvelope]
    time_map: tuple[TimeMapPoint, ...]
    time_map_preserves_pitch: bool
    time_map_frame_sampling: Optional[str]
    filters: tuple[FilterInstance | MaskedFilterInstance, ...]
    params: tuple[Parameter, ...]
    text_runs: tuple[TextRun, ...]
    text_styles: Mapping[str, TextStyle]
    multicam_sources: tuple[MulticamSource, ...]
    children: tuple["StoryNode", ...]
    raw_xml: str
    blend_keyframes: tuple[Keyframe, ...] = ()
    preserved_adjustments: tuple[PreservedAdjustment, ...] = ()


@dataclass(frozen=True)
class SourceDocument:
    schema_version: int
    source_path: Path
    source_sha256: str
    fcpxml_version: str
    project_name: str
    event_name: Optional[str]
    sequence_format_id: str
    sequence_duration: Fraction
    sequence_tc_start: Fraction
    formats: Mapping[str, FormatResource]
    assets: Mapping[str, AssetResource]
    effects: Mapping[str, EffectResource]
    multicams: Mapping[str, MulticamResource]
    other_resources: tuple[OtherResource, ...]
    spine: tuple[StoryNode, ...]
    sequence_audio_layout: str = "stereo"
    sequence_audio_rate: int = 48_000
    sequence_render_format: Optional[str] = None
    # Directory that BUNDLE-RELATIVE media-rep ``src`` values are resolved
    # against. For a plain ``.fcpxml`` file this is the file's parent
    # directory; for a canonical ``.fcpxmld`` bundle it is the bundle root
    # (the directory holding ``Info.fcpxml`` and the ``Media/`` subfolder).
    # ``None`` only in synthetic fixtures that never resolve relative media.
    media_base_dir: Optional[Path] = None


@dataclass(frozen=True)
class FontBinding:
    name: str
    path: Path
    index: int = 0


@dataclass(frozen=True)
class AssetBinding:
    resource_id: Optional[str]
    uid: Optional[str]
    path: Path


@dataclass(frozen=True)
class Bindings:
    assets: tuple[AssetBinding, ...] = ()
    fonts: tuple[FontBinding, ...] = ()


@dataclass(frozen=True)
class ResolvedEffect:
    kind: str
    uid: Optional[str]
    name: Optional[str]
    handler: Optional[str]
    portable_status: str
    params: tuple[Parameter, ...]
    calibration: Mapping[str, Any]
    data: Mapping[str, str]
    mask: Optional["ResolvedMaskGroup"] = None
    outside_effect: Optional["ResolvedEffect"] = None
    path: str = ""
    artifact_id: Optional[str] = None
    artifact_version: Optional[int] = None
    parameter_values: Mapping[str, Any] = field(default_factory=dict)
    # ``identity`` preserves an authored effect in the semantic sequence while
    # retaining the CPU renderer's explicit warn-and-ignore behavior. Applied
    # effects continue through the existing execution tuple unchanged.
    execution: str = "apply"
    capability_id: Optional[str] = None
    omission_reason: Optional[str] = None


@dataclass(frozen=True)
class ResolvedMask:
    """A validated mask whose parameters are safe to turn into FFmpeg expressions."""

    kind: str
    name: str
    blend_mode: str
    params: tuple[Parameter, ...]
    data: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedMaskGroup:
    masks: tuple[ResolvedMask, ...]
    inverted: bool


@dataclass(frozen=True)
class RenderVideoDisposition:
    """The compiler's explicit pixel replacement for one video story item.

    ``composite`` means a backend must execute the layer. ``omit_transparent``
    retains an authored video item whose portable result is transparency.
    ``authored_disabled`` retains a user-disabled item without treating it as
    degradation. ``not_applicable`` identifies audio-only/non-video story
    items that do not belong in a video CompositionPlan.

    Main callers:
    - ``compiler._compile_clip`` after source/title resolution.
    - The shared composition snapshot, which must not infer an omission from a
      missing path or absent runtime raster.

    Why this exists:
    A compatibility report explains a decision to the user, but it is not an
    executable semantic IR. Keeping the replacement on the compiled item
    prevents CPU and Vulkan planners from independently guessing why a layer
    disappeared.
    """

    execution: Literal[
        "composite",
        "omit_transparent",
        "authored_disabled",
        "not_applicable",
    ]
    reason: Optional[str] = None
    portable_status: Optional[str] = None
    construct: Optional[str] = None
    uid: Optional[str] = None

    def __post_init__(self) -> None:
        if self.execution not in {
            "composite",
            "omit_transparent",
            "authored_disabled",
            "not_applicable",
        }:
            raise ValueError(f"unknown video disposition {self.execution!r}")
        if self.execution == "omit_transparent":
            for name in ("reason", "portable_status", "construct"):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"omit_transparent video disposition requires {name}"
                    )
        elif any(
            value is not None
            for value in (self.reason, self.portable_status, self.construct, self.uid)
        ):
            raise ValueError(
                f"{self.execution} video disposition cannot carry omission metadata"
            )


@dataclass(frozen=True)
class RenderTransformAnimation:
    """Typed v2 transform tracks evaluated through one exact retime map."""

    position: Optional["TimelineAnimatedVec2"] = None
    scale: Optional["TimelineAnimatedVec2"] = None
    rotation: Optional["TimelineAnimatedScalar"] = None
    anchor: Optional["TimelineAnimatedVec2"] = None
    notices: tuple["AnimationNotice", ...] = ()


@dataclass
class RenderClip:
    id: str
    ancestor_clip_ids: tuple[str, ...]
    kind: str
    path: str
    name: Optional[str]
    absolute_start: Fraction
    duration: Fraction
    source_start: Fraction
    lane: int
    document_order: int
    media_path: Optional[Path]
    asset_id: Optional[str]
    asset_uid: Optional[str]
    has_video: bool
    has_audio: bool
    is_still: bool
    enabled: bool
    src_enable: Optional[str]
    conform_type: str
    transform: Optional[TransformAdjustment]
    crop: Optional[CropAdjustment]
    blend_opacity: float
    opacity_fade: Optional[FadeEnvelope]
    volume_db: Optional[float]
    audio_fade: Optional[FadeEnvelope]
    speed: Fraction
    effects: tuple[ResolvedEffect, ...]
    params: tuple[Parameter, ...]
    text_runs: tuple[TextRun, ...]
    text_styles: Mapping[str, TextStyle]
    retime_map: Optional["RetimeMap"] = None
    transform_animation: Optional[RenderTransformAnimation] = None
    opacity_animation: Optional["TimelineAnimatedScalar"] = None
    corner_pin: Optional["CornerPinAdjustment"] = None
    spatial_intrinsics: Optional["SpatialIntrinsicPlan"] = None
    text_plan: Optional["TextRenderPlan"] = None
    generator_plan: Optional["GeneratorRenderPlan"] = None
    blend_mode: Optional[str] = None
    transition_in: Optional["RenderTransition"] = None
    transition_out: Optional["RenderTransition"] = None
    # File-local timing is distinct from the FCPXML asset timecode domain.
    # ``source_start`` is always file-local; the origin remains recorded for
    # diagnostics and exact bounds reasoning.
    asset_source_origin: Fraction = Fraction(0)
    asset_source_duration: Optional[Fraction] = None
    source_frame_duration: Optional[Fraction] = None
    # Target canvas for this clip's own geometry. Compound descendants use the
    # referenced sequence canvas; root clips leave these unset and use Project.
    canvas_context: Optional[SequenceFormatContext] = None
    # A container scope additionally owns the completed child raster size.
    container_context: Optional[SequenceFormatContext] = None
    # A recursively composed source exposes this source-time window before the
    # ordinary clip-instance retime. File inputs use their direct decode trim.
    source_window_origin: Optional[Fraction] = None
    source_window_duration: Optional[Fraction] = None
    # Stable identity of the file/compound/multicam source below this clip.
    render_source_id: Optional[str] = None
    render_source_kind: Optional[str] = None
    # Complete authored effect order, including registry-authorized identity
    # omissions. ``effects`` remains the executable CPU subset so preserving
    # these records cannot change the reference filter graph.
    semantic_effects: tuple[ResolvedEffect, ...] = ()
    video_disposition: Optional[RenderVideoDisposition] = None
    # Exact authored or explicitly bound locations that could not be opened.
    # A non-empty tuple means video uses Bladeworks's visible placeholder and
    # audio contributes silence. The locations remain data, not guessed paths.
    missing_media_locators: tuple[str, ...] = ()

    @property
    def canvas_width(self) -> Optional[int]:
        return self.canvas_context.width if self.canvas_context else None

    @property
    def canvas_height(self) -> Optional[int]:
        return self.canvas_context.height if self.canvas_context else None

    @property
    def canvas_frame_duration(self) -> Optional[Fraction]:
        return self.canvas_context.frame_duration if self.canvas_context else None

    @property
    def container_width(self) -> Optional[int]:
        return self.container_context.width if self.container_context else None

    @property
    def container_height(self) -> Optional[int]:
        return self.container_context.height if self.container_context else None

    @property
    def container_frame_duration(self) -> Optional[Fraction]:
        return self.container_context.frame_duration if self.container_context else None

    @property
    def end(self) -> Fraction:
        return self.absolute_start + self.duration


@dataclass(frozen=True)
class RenderTransition:
    path: str
    absolute_start: Fraction
    duration: Fraction
    uid: Optional[str]
    name: Optional[str]
    handler: Optional[str]
    params: tuple[Parameter, ...]
    # Registry identity and an explicit hard-cut disposition survive even
    # when no portable pixel handler exists.  The shared CompositionPlan can
    # therefore retain the authored transition without reconstructing a
    # warning from report text or inventing participant handles.
    capability_id: Optional[str] = None
    portable_status: str = "unsupported"
    omission_reason: Optional[str] = None
    xfade_id: Optional[str] = None
    artifact_id: Optional[str] = None
    artifact_version: Optional[int] = None
    parameter_values: Mapping[str, Any] = field(default_factory=dict)
    ancestor_group_ids: tuple[str, ...] = ()
    # Local storyline topology is preserved even when this transition has no
    # portable handler and therefore renders as an explicit hard cut. These
    # IDs are compiler facts, never a later nearest-layer guess.
    previous_story_id: Optional[str] = None
    next_story_id: Optional[str] = None

    @property
    def end(self) -> Fraction:
        return self.absolute_start + self.duration


@dataclass(frozen=True)
class MissingMediaReference:
    """One timeline use of an exact local-media locator that is offline."""

    locator: str
    fcpxml_path: str
    timeline_start: Fraction
    timeline_duration: Fraction
    has_video: bool
    has_audio: bool


@dataclass(frozen=True)
class RenderDocument:
    schema_version: int
    source_sha256: str
    source_path: Path
    project_name: str
    width: int
    height: int
    frame_duration: Fraction
    duration: Fraction
    tc_start: Fraction
    clips: tuple[RenderClip, ...]
    transitions: tuple[RenderTransition, ...]
    asset_bindings: tuple[AssetBinding, ...]
    font_bindings: tuple[FontBinding, ...]
    story: Optional["RenderStoryPlan"] = None
    audio: Optional["AudioRenderPlan"] = None
    group_scopes: tuple[RenderClip, ...] = ()
    missing_media_references: tuple[MissingMediaReference, ...] = ()

    @property
    def fps(self) -> Fraction:
        return 1 / self.frame_duration

    @property
    def frame_count(self) -> int:
        frames = self.duration / self.frame_duration
        if frames.denominator == 1:
            return frames.numerator
        return (frames.numerator + frames.denominator - 1) // frames.denominator


@dataclass(frozen=True)
class VideoDecoderOrigin:
    """Bind one FFmpeg input to its file and decoder-local time origins.

    ``file_source_start`` is the first source instant needed by the ordinary
    semantic layer. ``decoder_seek`` is the input-level ``-ss`` applied before
    that decoder. ``filter_source_start`` is therefore the same instant in the
    timestamps visible to filters. Keeping all three values makes a backend
    replacement auditable and prevents it from assuming every retained input
    either starts at the file beginning or was pre-seeked.

    The optional frame-ownership fields are one explicit migration boundary.
    Legacy internal constructors may omit all of them until the phase-2 emitter
    integration lands. New scheduling code must populate the complete set and
    call ``require_frame_ownership``; a partial contract always fails. No field
    is inferred from decoder timestamp rounding.

    Main callers:
    - The CPU invocation builder, which owns decoder scheduling.
    - The Vulkan lowerer, which reuses those exact inputs but replaces only the
      pixel graph.
    """

    clip_id: str
    input_index: int
    file_source_start: Fraction
    decoder_seek: Fraction
    filter_source_start: Fraction
    source_frame_duration: Optional[Fraction] = None
    source_start_frame: Optional[int] = None
    source_phase: Optional[Fraction] = None
    sampling_direction: Optional[str] = None
    decoder_start_frame: Optional[int] = None
    filter_start_frame: Optional[int] = None

    def __post_init__(self) -> None:
        if self.input_index < 0:
            raise ValueError("video decoder input_index cannot be negative")
        for name in ("file_source_start", "decoder_seek", "filter_source_start"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Fraction):
                raise ValueError(f"video decoder {name} must be an exact Fraction")
        if self.decoder_seek < 0:
            raise ValueError("video decoder seek cannot be negative")
        if self.file_source_start != self.decoder_seek + self.filter_source_start:
            raise ValueError(
                "video decoder origin must satisfy "
                "file_source_start = decoder_seek + filter_source_start"
            )
        frame_fields = {
            "source_frame_duration": self.source_frame_duration,
            "source_start_frame": self.source_start_frame,
            "source_phase": self.source_phase,
            "sampling_direction": self.sampling_direction,
            "decoder_start_frame": self.decoder_start_frame,
            "filter_start_frame": self.filter_start_frame,
        }
        present = {name for name, value in frame_fields.items() if value is not None}
        if present and len(present) != len(frame_fields):
            missing = sorted(set(frame_fields) - present)
            raise ValueError(
                "video decoder frame ownership is partial; missing "
                + ", ".join(missing)
            )
        if not present:
            return

        frame_duration = self.source_frame_duration
        source_frame = self.source_start_frame
        source_phase = self.source_phase
        decoder_frame = self.decoder_start_frame
        filter_frame = self.filter_start_frame
        direction = self.sampling_direction
        assert frame_duration is not None
        assert source_frame is not None
        assert source_phase is not None
        assert decoder_frame is not None
        assert filter_frame is not None
        assert direction is not None
        if isinstance(frame_duration, bool) or not isinstance(
            frame_duration, Fraction
        ):
            raise ValueError(
                "video decoder source_frame_duration must be an exact Fraction"
            )
        if frame_duration <= 0:
            raise ValueError("video decoder source_frame_duration must be positive")
        if isinstance(source_phase, bool) or not isinstance(source_phase, Fraction):
            raise ValueError("video decoder source_phase must be an exact Fraction")
        if not 0 <= source_phase < frame_duration:
            raise ValueError(
                "video decoder source_phase must lie in [0, source_frame_duration)"
            )
        for name, value in (
            ("source_start_frame", source_frame),
            ("decoder_start_frame", decoder_frame),
            ("filter_start_frame", filter_frame),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"video decoder {name} must be an integer")
            if value < 0:
                raise ValueError(f"video decoder {name} cannot be negative")
        if direction not in {"forward", "reverse"}:
            raise ValueError(
                "video decoder sampling_direction must be 'forward' or 'reverse'"
            )
        if source_frame != decoder_frame + filter_frame:
            raise ValueError(
                "video decoder frame origin must satisfy source_start_frame = "
                "decoder_start_frame + filter_start_frame"
            )
        if self.decoder_seek != decoder_frame * frame_duration:
            raise ValueError(
                "video decoder seek must equal decoder_start_frame * "
                "source_frame_duration"
            )

        if direction == "forward":
            expected_file_start = source_frame * frame_duration + source_phase
            expected_filter_start = filter_frame * frame_duration + source_phase
        else:
            expected_file_start = (source_frame + 1) * frame_duration - source_phase
            expected_filter_start = (filter_frame + 1) * frame_duration - source_phase
        if self.file_source_start != expected_file_start:
            raise ValueError(
                "video decoder file_source_start does not match its owning frame "
                "and playback phase"
            )
        if self.filter_source_start != expected_filter_start:
            raise ValueError(
                "video decoder filter_source_start does not match its decoder-local "
                "frame and playback phase"
            )

    @property
    def has_frame_ownership(self) -> bool:
        """Return whether this origin carries the complete phase-2 contract."""

        return self.source_frame_duration is not None

    def require_frame_ownership(self) -> "VideoDecoderOrigin":
        """Reject a legacy time-only origin at a backend integration boundary.

        Main callers:
        - CPU and Vulkan emitters in phase 2, before generating source filters.
        """

        if not self.has_frame_ownership:
            raise ValueError(
                f"video decoder origin for {self.clip_id} has no frame ownership contract"
            )
        return self


@dataclass(frozen=True)
class FFmpegInvocation:
    """A shell-free FFmpeg execution plan suitable for JSON diagnostics."""

    argv: tuple[str, ...]
    filter_script: str
    expected_frame_count: int
    output_path: Path
    input_paths: tuple[Path, ...]
    video_decoder_origins: tuple[VideoDecoderOrigin, ...] = ()
    gpu_islands: tuple["GPUIslandPlan", ...] = ()
    audio_execution_manifest: Optional[Mapping[str, Any]] = None
    spatial_execution_manifests: tuple[Mapping[str, Any], ...] = ()
    pixel_domain_manifest: Optional[Mapping[str, Any]] = None
    # The composition plan is attached during the CPU-shadow migration for
    # diagnostics and backend lowering. It deliberately does not participate
    # in invocation equality or repr: the executable CPU command remains the
    # no-change reference while this semantic plan is validated beside it.
    composition_plan: Optional["CompositionPlan"] = field(
        default=None,
        compare=False,
        repr=False,
    )
    requested_backend: str = "cpu"
    selected_backend: str = "cpu"
    backend_manifest: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class GPUNodePlan:
    """One immutable registry artifact dispatched within a GPU island."""

    kind: str
    fcpxml_path: str
    artifact_id: str
    spirv_sha256: str
    pipeline_cache_key: str


@dataclass(frozen=True)
class GPUIslandPlan:
    """A maximal practical run of Vulkan nodes with one upload/download pair."""

    id: str
    scope_path: str
    nodes: tuple[GPUNodePlan, ...]
    upload_count: int = 1
    download_count: int = 1


def dataclass_json(value: Any) -> Any:
    """Convert renderer dataclasses into stable JSON-compatible values."""

    from dataclasses import fields, is_dataclass

    if isinstance(value, Fraction):
        return fraction_json(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {item.name: dataclass_json(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): dataclass_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [dataclass_json(item) for item in value]
    return value


def walk_story(nodes: Iterable[StoryNode]) -> Iterable[StoryNode]:
    for node in nodes:
        yield node
        yield from walk_story(node.children)
