"""Compile the source-preserving FCPXML graph into a portable render document.

Architecture map
================

1. Validate sequence format and exact frame timing.
2. Resolve nested FCPXML offsets with the parent anchor identity.
3. Bind file-backed assets without guessing names.
4. Resolve explicit multicam angle selections back to synchronized assets.
5. Classify titles, filters, and transitions through the shared registry.
6. Associate transitions with adjacent clips inside each storyline.

Main callers:
- The CLI ``inspect`` and ``render`` commands.
- Future backend adapters should call this function rather than parsing XML.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

from .bindings import load_bindings, resolve_asset, unresolved_asset_locators
from .audio_ir import AudioIRCompileError, AudioIRFinding
from .animation import (
    AnimatedScalar,
    AnimatedVec2,
    AnimationValidationError,
    map_scalar_animation,
    map_vec2_animation,
)
from .basic_effects import unsupported_basic_effect_reason
from .capabilities import Capability, CapabilityRegistry
from .color import unsupported_color_reason
from .cohort_effects import unsupported_cohort_effect_reason
from .compositor import (
    UnknownBlendModeError,
    UnsupportedBlendModeError,
    resolve_blend_mode,
)
from .errors import FCPXMLCompileError
from .geometry import CornerPinAdjustment, CornerPinAnimation, GeometryValidationError
from .keyer import GreenScreenKeyerDataError, resolve_green_screen_keyer
from .masks import MaskResolutionError, resolve_mask_group
from .multicam_execution import (
    MulticamExecutionError,
    MulticamExecutionFinding,
    build_multicam_execution_plan,
)
from .render_sources import (
    RenderableAVSource,
    RenderSourceError,
    resolve_file_source,
    resolve_instance_stream_timing,
    source_bound_tolerance,
    source_range_within_bounds,
    source_window_for_retime,
)
from .model import (
    SCHEMA_VERSION,
    Bindings,
    FilterInstance,
    MaskedFilterInstance,
    MissingMediaReference,
    Parameter,
    RenderClip,
    RenderDocument,
    RenderTransformAnimation,
    RenderTransition,
    RenderVideoDisposition,
    ResolvedEffect,
    SequenceFormatContext,
    SourceDocument,
    StoryNode,
    walk_story,
)
from .parser import parse_fcpxml
from .report import CompatibilityReport
from .retime import (
    RetimeMap,
    RetimeSegment,
    RetimeValidationError,
    UnsupportedRetimeMappingError,
)
from .story_ir import StoryIRFinding
from .story_containers import (
    StoryContainerError,
    build_story_container_plan,
)
from .spatial_intrinsics import (
    ColorConform,
    DisplayConform,
    OpaqueCinematicLocator,
    OpaqueTrackerLocator,
    Orientation360,
    Reorientation360,
    RollingShutterAdjustment,
    SpatialIntrinsicError,
    SpatialIntrinsicPlan,
    Stabilization,
    Stereo3DAdjustment,
    Transform360,
    classify_fcp_color_space,
)
from .text_templates import (
    GeneratorRenderPlan,
    TextFinding,
    TextPlanError,
    TextRenderPlan,
    build_generator_render_plan,
    build_text_render_plan,
)


@dataclass(frozen=True)
class CompileResult:
    source: SourceDocument
    render: RenderDocument
    report: CompatibilityReport


@dataclass(frozen=True)
class _ResolvedEffectSet:
    """Keep executable CPU effects and the complete semantic order separate.

    Main callers:
    - ``_Compiler._resolve_effects`` while compiling a clip or group scope.

    Why this exists:
    Registry-authorized warn-and-ignore effects must remain visible to the
    shared CompositionPlan, but placing them in ``RenderClip.effects`` would
    change existing CPU fusion decisions that intentionally operate on the
    executable subset only.
    """

    applied: tuple[ResolvedEffect, ...]
    semantic: tuple[ResolvedEffect, ...]


def compile_fcpxml(
    path: Path,
    *,
    project: Optional[str] = None,
    bindings_path: Optional[Path] = None,
    bindings: Optional[Bindings] = None,
    capability_registry: Optional[CapabilityRegistry] = None,
) -> CompileResult:
    """Compile one FCPXML document without invoking FFmpeg or touching outputs."""

    source = parse_fcpxml(path, project=project)
    _validate_references(source)
    resolved_bindings = bindings if bindings is not None else load_bindings(bindings_path)
    registry = capability_registry or CapabilityRegistry.load()
    report = CompatibilityReport(
        source_path=str(source.source_path),
        source_sha256=source.source_sha256,
        project_name=source.project_name,
        timeline_start=source.sequence_tc_start,
        timeline_duration=source.sequence_duration,
    )
    compiler = _Compiler(source=source, bindings=resolved_bindings, registry=registry, report=report)
    render = compiler.compile()
    return CompileResult(source=source, render=render, report=report)


class _Compiler:
    def __init__(
        self,
        *,
        source: SourceDocument,
        bindings: Bindings,
        registry: CapabilityRegistry,
        report: CompatibilityReport,
    ):
        self.source = source
        self.bindings = bindings
        self.registry = registry
        self.report = report
        self.clips: list[RenderClip] = []
        self.group_scopes: list[RenderClip] = []
        self.transitions: list[RenderTransition] = []
        self.used_asset_bindings = []
        self.missing_media_references: list[MissingMediaReference] = []
        self.document_order = 0
        self._render_sources: dict[str, RenderableAVSource] = {}
        self._canvas_context: SequenceFormatContext | None = None

    def compile(self) -> RenderDocument:
        sequence_format = self.source.formats[self.source.sequence_format_id]
        if sequence_format.frame_duration is None or sequence_format.frame_duration <= 0:
            raise FCPXMLCompileError("sequence format requires a positive frameDuration")
        if not sequence_format.width or not sequence_format.height:
            raise FCPXMLCompileError("sequence format requires positive width and height")
        self._canvas_context = SequenceFormatContext.from_resource(sequence_format)
        if self.source.sequence_duration / sequence_format.frame_duration != int(
            self.source.sequence_duration / sequence_format.frame_duration
        ):
            raise FCPXMLCompileError("sequence duration is not aligned to its frameDuration")

        if self.source.fcpxml_version != "1.14":
            self.report.add(
                outcome="info",
                portable_status="calibrated_portable",
                fcpxml_path="fcpxml",
                construct=f"FCPXML {self.source.fcpxml_version}",
                disposition="parsed with the documented 1.14 subset; version-specific extensions remain unsupported",
            )
        color_space = sequence_format.color_space or ""
        if color_space and "709" not in color_space:
            self.report.add(
                outcome="approximated",
                portable_status="calibrated_portable",
                fcpxml_path="sequence",
                construct=f"sequence color space {color_space}",
                disposition="normalized to the MVP SDR Rec.709 output",
            )
        self._report_sequence_metadata(sequence_format)

        for resource in self.source.other_resources:
            self.report.add(
                outcome="info",
                portable_status="apple_only" if resource.kind == "media" else "unsupported",
                fcpxml_path=f"resources/{resource.kind}[@id='{resource.id or ''}']",
                construct=resource.kind,
                uid=resource.uid,
                disposition="preserved in the source model; only a referencing story node can affect the render",
            )
        for resource in self.source.multicams.values():
            self.report.add(
                outcome="info",
                portable_status="exact_portable",
                fcpxml_path=f"resources/media[@id='{resource.id}']/multicam",
                construct=f"multicam resource {resource.name or resource.id}",
                uid=resource.uid,
                disposition="existing Final Cut angle synchronization preserved and available for explicit angle selection",
            )

        try:
            container_plan = build_story_container_plan(self.source)
        except StoryContainerError as error:
            raise FCPXMLCompileError(f"invalid story container graph: {error}") from error
        try:
            multicam = build_multicam_execution_plan(
                self.source,
                container_plan=container_plan,
            )
        except (MulticamExecutionError, AudioIRCompileError) as error:
            raise FCPXMLCompileError(
                f"invalid or ambiguous multicam/audio graph: {error}"
            ) from error
        story = multicam.story
        audio = multicam.audio
        self._render_sources = dict(multicam.sources)
        self._report_story_findings(story.findings)
        self._report_multicam_findings(multicam.findings)
        self._report_audio_findings(audio.findings)
        audio = self._bind_audio_assets(audio)

        self._resolve_storyline(
            self.source.spine,
            parent_absolute=Fraction(0),
            parent_source=self.source.sequence_tc_start,
            inherited_lane=0,
            ancestor_clip_ids=(),
            sequential=True,
        )
        clips = tuple(sorted(self.clips, key=lambda clip: (clip.lane, clip.document_order)))
        for clip in clips:
            for locator in clip.missing_media_locators:
                self.missing_media_references.append(
                    MissingMediaReference(
                        locator=locator,
                        fcpxml_path=clip.path,
                        timeline_start=clip.absolute_start,
                        timeline_duration=clip.duration,
                        has_video=clip.has_video,
                        has_audio=clip.has_audio,
                    )
                )
        return RenderDocument(
            schema_version=SCHEMA_VERSION,
            source_sha256=self.source.source_sha256,
            source_path=self.source.source_path,
            project_name=self.source.project_name,
            width=sequence_format.width,
            height=sequence_format.height,
            frame_duration=sequence_format.frame_duration,
            duration=self.source.sequence_duration,
            tc_start=self.source.sequence_tc_start,
            clips=clips,
            transitions=tuple(self.transitions),
            asset_bindings=tuple(self.used_asset_bindings),
            font_bindings=self.bindings.fonts,
            story=story,
            audio=audio,
            group_scopes=tuple(
                sorted(
                    self.group_scopes,
                    key=lambda group: (len(group.ancestor_clip_ids), group.document_order),
                )
            ),
            missing_media_references=tuple(self.missing_media_references),
        )

    def _report_audio_findings(
        self,
        findings: tuple[AudioIRFinding, ...],
    ) -> None:
        """Expose audio-IR decisions before the stock engine executes.

        Main callers:
        - ``compile`` immediately after independent audio scheduling.

        Preserved and inactive records are informational. A deferred active
        operation remains an explicit omission until AUDIO-2 consumes it.
        """

        for finding in findings:
            if finding.disposition == "not_implemented_yet":
                outcome = "omitted"
                status = "unsupported"
            else:
                outcome = "info"
                status = "calibrated_portable" if finding.disposition == "preserved" else "unsupported"
            self.report.add(
                outcome=outcome,
                portable_status=status,
                fcpxml_path=finding.path,
                construct=finding.code,
                disposition=finding.detail,
            )

    def _report_multicam_findings(
        self,
        findings: tuple[MulticamExecutionFinding, ...],
    ) -> None:
        """Expose inactive selected-angle decisions without guessing angles."""

        for finding in findings:
            self.report.add(
                outcome="info" if finding.disposition == "inactive" else "omitted",
                portable_status=(
                    "exact_portable"
                    if finding.disposition == "inactive"
                    else "unsupported"
                ),
                fcpxml_path=finding.path,
                construct=finding.code,
                disposition=finding.detail,
            )

    def _bind_audio_assets(self, audio):
        """Resolve every independently scheduled audio asset by exact identity.

        Main callers:
        - ``compile`` after compound audio expansion.

        Why this exists:
        Compound resources can contain audio assets that never appear in the
        legacy flat video clip list.  The v2 audio executor must still receive
        their verified local bindings rather than guessing from filenames.
        """

        known = {
            binding.resource_id
            for binding in self.used_asset_bindings
            if binding.resource_id is not None
        }
        missing_asset_ids: set[str] = set()
        for item in audio.items:
            if item.asset_id in known:
                continue
            asset = self.source.assets.get(item.asset_id)
            if asset is None:
                raise FCPXMLCompileError(
                    f"audio item {item.path} references unknown asset {item.asset_id!r}"
                )
            media_path, binding = resolve_asset(asset, self.bindings, self.source.media_base_dir)
            if media_path is None or binding is None or not media_path.is_file():
                missing_asset_ids.add(item.asset_id)
                locators = unresolved_asset_locators(
                    asset,
                    self.bindings,
                    self.source.media_base_dir,
                )
                self.report.add(
                    outcome="info",
                    portable_status="degraded",
                    fcpxml_path=item.path,
                    construct=f"missing audio media {asset.name or asset.id}",
                    timeline_start=item.absolute_start,
                    timeline_duration=item.duration,
                    disposition=(
                        "missing source contributes silence: " + ", ".join(locators)
                    ),
                )
                for locator in locators:
                    self.missing_media_references.append(
                        MissingMediaReference(
                            locator=locator,
                            fcpxml_path=item.path,
                            timeline_start=item.absolute_start,
                            timeline_duration=item.duration,
                            has_video=False,
                            has_audio=True,
                        )
                    )
                continue
            self.used_asset_bindings.append(binding)
            known.add(item.asset_id)
        if not missing_asset_ids:
            return audio
        return replace(
            audio,
            items=tuple(
                replace(item, active=False)
                if item.asset_id in missing_asset_ids
                else item
                for item in audio.items
            ),
        )

    def _report_story_findings(
        self,
        findings: tuple[StoryIRFinding, ...],
    ) -> None:
        """Expose every hierarchy-resolution decision in compatibility output.

        Main callers:
        - ``compile`` immediately after constructing the v2 story graph.

        Inactive records are informational. Unresolved groups remain explicit
        omissions or invalid-source rejections.
        """

        for finding in findings:
            if finding.disposition == "inactive":
                outcome = "info"
                status = "unsupported"
                detail = finding.detail
            elif finding.disposition == "invalid":
                outcome = "failed"
                status = "unsupported"
                detail = finding.detail
            else:
                outcome = "omitted"
                status = "unsupported"
                detail = finding.detail
            self.report.add(
                outcome=outcome,
                portable_status=status,
                fcpxml_path=finding.path,
                construct=finding.code,
                disposition=detail,
            )

    def _resolve_storyline(
        self,
        nodes: tuple[StoryNode, ...],
        *,
        parent_absolute: Fraction,
        parent_source: Fraction,
        inherited_lane: int,
        ancestor_clip_ids: tuple[str, ...],
        sequential: bool,
    ) -> None:
        cursor = parent_source
        local_items: list[tuple[str, object]] = []
        for node in nodes:
            offset = node.offset if node.offset is not None else cursor if sequential else parent_source
            absolute = parent_absolute + offset - parent_source
            effective_lane = inherited_lane + node.lane
            self.document_order += 1
            compiled_clip: Optional[RenderClip] = None

            if node.kind == "transition":
                transition = self._compile_transition(
                    node,
                    absolute,
                    ancestor_group_ids=ancestor_clip_ids,
                )
                local_items.append(("transition", transition))
            elif node.kind == "spine":
                self._resolve_storyline(
                    node.children,
                    parent_absolute=absolute,
                    parent_source=node.start,
                    inherited_lane=effective_lane,
                    ancestor_clip_ids=ancestor_clip_ids,
                    sequential=True,
                )
            elif node.kind == "gap":
                local_items.append(("gap", node))
            elif node.kind in {"asset-clip", "video", "audio", "title", "caption"}:
                clip = self._compile_clip(
                    node,
                    absolute,
                    effective_lane,
                    self.document_order,
                    ancestor_clip_ids,
                )
                compiled_clip = clip
                self.clips.append(clip)
                local_items.append(("clip", clip))
            elif node.kind == "mc-clip":
                resolved = self._render_sources.get(node.path)
                if resolved is None:
                    raise FCPXMLCompileError(
                        f"{node.path} has no resolved multicam source"
                    )
                if resolved.has_video:
                    group = self._resolve_virtual_video_source(
                        node,
                        resolved,
                        absolute=absolute,
                        lane=effective_lane,
                        order=self.document_order,
                        ancestor_clip_ids=ancestor_clip_ids,
                    )
                    local_items.append(("clip", group))
                if node.children:
                    # Anchored timeline items are outside the referenced
                    # multicam source, just as they are for a natural clip.
                    self._resolve_storyline(
                        node.children,
                        parent_absolute=absolute,
                        parent_source=node.start,
                        inherited_lane=effective_lane,
                        ancestor_clip_ids=ancestor_clip_ids,
                        sequential=False,
                    )
            elif node.kind in {"clip", "sync-clip", "audition"}:
                group = self._compile_group_scope(
                    node,
                    absolute,
                    effective_lane,
                    self.document_order,
                    ancestor_clip_ids,
                )
                self.group_scopes.append(group)
                local_items.append(("clip", group))
                if node.kind == "audition":
                    active_children = node.children[:1]
                    for inactive in node.children[1:]:
                        self.report.add(
                            outcome="info",
                            portable_status="unsupported",
                            fcpxml_path=inactive.path,
                            construct="inactive audition choice",
                            timeline_start=absolute,
                            timeline_duration=inactive.duration,
                            disposition="preserved but not rendered because the first audition choice is active",
                        )
                    inner_children = active_children
                    connected_children = ()
                else:
                    inner_children = tuple(child for child in node.children if child.lane == 0)
                    connected_children = tuple(child for child in node.children if child.lane != 0)
                if inner_children:
                    group_source_origin = (
                        group.source_window_origin
                        if group.source_window_origin is not None
                        else node.start
                    )
                    self._resolve_storyline(
                        inner_children,
                        parent_absolute=absolute,
                        parent_source=group_source_origin,
                        inherited_lane=effective_lane,
                        ancestor_clip_ids=(*ancestor_clip_ids, group.id),
                        sequential=False,
                    )
                if connected_children:
                    self._resolve_storyline(
                        connected_children,
                        parent_absolute=absolute,
                        parent_source=node.start,
                        inherited_lane=effective_lane,
                        ancestor_clip_ids=ancestor_clip_ids,
                        sequential=False,
                    )
            elif node.kind == "ref-clip":
                resolved = self._render_sources.get(node.path)
                if resolved is None:
                    raise FCPXMLCompileError(
                        f"{node.path} has no resolved compound source"
                    )
                group = self._resolve_virtual_video_source(
                    node,
                    resolved,
                    absolute=absolute,
                    lane=effective_lane,
                    order=self.document_order,
                    ancestor_clip_ids=ancestor_clip_ids,
                )
                local_items.append(("clip", group))
                if node.children:
                    # Connected timeline children belong to the instance's
                    # parent timeline, never to the referenced source graph.
                    self._resolve_storyline(
                        node.children,
                        parent_absolute=absolute,
                        parent_source=node.start,
                        inherited_lane=effective_lane,
                        ancestor_clip_ids=ancestor_clip_ids,
                        sequential=False,
                    )
            else:
                self.report.add(
                    outcome="omitted",
                    portable_status="apple_only" if node.kind in {"mc-clip", "ref-clip", "sync-clip", "audition"} else "unsupported",
                    fcpxml_path=node.path,
                    construct=node.kind,
                    timeline_start=absolute,
                    timeline_duration=node.duration,
                    disposition="story content omitted; its timeline interval and anchored children are preserved",
                )

            # Connected clips use their parent's source domain, not the current
            # storyline cursor. A nested spine, however, owns a sequential cursor.
            if node.kind not in {
                "spine",
                "clip",
                "sync-clip",
                "audition",
                "ref-clip",
                "mc-clip",
            } and node.children:
                child_ancestors = ancestor_clip_ids
                if compiled_clip is not None:
                    child_ancestors = (*ancestor_clip_ids, compiled_clip.id)
                self._resolve_storyline(
                    node.children,
                    parent_absolute=absolute,
                    parent_source=node.start,
                    inherited_lane=effective_lane,
                    ancestor_clip_ids=child_ancestors,
                    sequential=False,
                )
            if sequential:
                cursor = offset + node.duration

        self._associate_transitions(local_items)

    def _compile_group_scope(
        self,
        node: StoryNode,
        absolute: Fraction,
        lane: int,
        order: int,
        ancestor_group_ids: tuple[str, ...],
    ) -> RenderClip:
        """Compile one container's post-composition video scope.

        Main callers:
        - ``_resolve_storyline`` for clip, sync-clip, audition, and ref-clip.

        Why this exists:
        Child media must be composed first.  Representing the parent scope as
        a media-less RenderClip lets the FFmpeg group compositor reuse the
        exact geometry, effect, opacity, and blend contracts without applying
        them separately to every descendant.
        """

        retime_map = self._retime_map(node)
        frame_duration = (
            self._canvas_context.frame_duration
            if self._canvas_context is not None
            else self.source.formats[self.source.sequence_format_id].frame_duration
        )
        if frame_duration is None:
            raise FCPXMLCompileError(
                f"{node.path} container has no source frame duration"
            )
        try:
            source_window = source_window_for_retime(
                retime_map,
                frame_duration=frame_duration,
            )
        except ValueError as error:
            raise FCPXMLCompileError(
                f"{node.path} has an invalid source-instance window: {error}"
            ) from error
        animation_time_map = self._visual_animation_time_map(node)
        transform_animation = self._transform_animation(
            node, animation_time_map, absolute
        )
        opacity_animation = self._opacity_animation(node, animation_time_map, absolute)
        corner_pin = self._corner_pin(node, animation_time_map, absolute)
        blend_mode = self._blend_mode(node, absolute)
        self._report_normal_opacity_composition(node, absolute, blend_mode)
        self._report_preserved_adjustments(node, absolute)
        self._report_curve_approximations(node, absolute)
        resolved_effects = self._resolve_effects(node, absolute)
        return RenderClip(
            id=f"group-{order}",
            ancestor_clip_ids=ancestor_group_ids,
            kind=f"group:{node.kind}",
            path=node.path,
            name=node.name,
            absolute_start=absolute,
            duration=node.duration,
            source_start=node.start,
            lane=lane,
            document_order=order,
            media_path=None,
            asset_id=None,
            asset_uid=None,
            has_video=True,
            has_audio=False,
            is_still=False,
            enabled=node.enabled,
            src_enable=node.src_enable,
            conform_type=node.conform_type,
            transform=node.transform,
            crop=node.crop,
            blend_opacity=max(0.0, min(node.blend_opacity, 1.0)),
            opacity_fade=node.opacity_fade,
            volume_db=node.volume_db,
            audio_fade=node.audio_fade,
            speed=self._constant_speed(node, absolute),
            effects=resolved_effects.applied,
            params=node.params,
            text_runs=(),
            text_styles={},
            retime_map=retime_map,
            transform_animation=transform_animation,
            opacity_animation=opacity_animation,
            corner_pin=corner_pin,
            spatial_intrinsics=None,
            blend_mode=blend_mode,
            canvas_context=self._canvas_context,
            source_window_origin=source_window.start,
            source_window_duration=source_window.duration,
            semantic_effects=resolved_effects.semantic,
            video_disposition=RenderVideoDisposition(
                execution="composite" if node.enabled else "authored_disabled"
            ),
        )

    def _resolve_virtual_video_source(
        self,
        node: StoryNode,
        source: RenderableAVSource,
        *,
        absolute: Fraction,
        lane: int,
        order: int,
        ancestor_clip_ids: tuple[str, ...],
    ) -> RenderClip:
        """Compile a compound/multicam source through the normal clip instance.

        Main callers:
        - ``_resolve_storyline`` for a resolved multicam instance.

        The source story remains on its local source clock. Its completed group
        is then consumed by the outer clip scope, whose ordinary retime,
        effects, geometry, and timeline placement are identical to a file clip.
        No synthetic timeline ``clip/spine`` is created.
        """

        try:
            timing = resolve_instance_stream_timing(
                node,
                absolute_start=absolute,
                stream="video",
            )
        except RenderSourceError as error:
            raise FCPXMLCompileError(str(error)) from error
        source_end = source.end
        if source_end is None:
            raise FCPXMLCompileError(
                f"{node.path} {source.kind} source has no finite duration"
            )
        source_tolerance = source_bound_tolerance(
            has_video=source.has_video,
            frame_duration=(
                source.format_context.frame_duration
                if source.format_context is not None
                else None
            ),
            has_audio=False,
            audio_rate=None,
        )
        if not source_range_within_bounds(
            requested_start=timing.source_start,
            requested_end=timing.source_end,
            source_start=source.source_start,
            source_end=source_end,
            tolerance=source_tolerance,
        ):
            raise FCPXMLCompileError(
                f"{node.path} video source range [{timing.source_start}, "
                f"{timing.source_end}) is outside {source.kind} source "
                f"[{source.source_start}, {source_end})"
            )
        if source.format_context is None:
            raise FCPXMLCompileError(
                f"{node.path} {source.kind} source has no complete local format"
            )
        self._validate_compound_format_context(node.path, source.format_context)
        source_window = source_window_for_retime(
            timing.retime_map,
            frame_duration=source.format_context.frame_duration,
        )
        if not source_range_within_bounds(
            requested_start=source_window.start,
            requested_end=source_window.end,
            source_start=source.source_start,
            source_end=source_end,
            tolerance=source_tolerance,
        ):
            raise FCPXMLCompileError(
                f"{node.path} executable video source window "
                f"[{source_window.start}, {source_window.end}) is outside "
                f"{source.kind} source [{source.source_start}, {source_end})"
            )
        outer = self._compile_group_scope(
            node,
            absolute,
            lane,
            order,
            ancestor_clip_ids,
        )
        outer.container_context = source.format_context
        outer.source_window_origin = source_window.start
        outer.source_window_duration = source_window.duration
        outer.render_source_id = source.id
        outer.render_source_kind = source.kind
        self.group_scopes.append(outer)

        source_ancestors = (*ancestor_clip_ids, outer.id)
        if source.video_scope is not None:
            self.document_order += 1
            source_scope = replace(
                source.video_scope,
                offset=source_window.start,
                start=source_window.start,
                duration=source_window.duration,
            )
            inner = self._compile_group_scope(
                source_scope,
                absolute,
                lane,
                self.document_order,
                source_ancestors,
            )
            inner.container_context = source.format_context
            inner.source_window_origin = source_window.start
            inner.source_window_duration = source_window.duration
            self.group_scopes.append(inner)
            source_ancestors = (*source_ancestors, inner.id)

        outer_canvas = self._canvas_context
        if source.format_context is not None:
            self._canvas_context = source.format_context
        clip_count_before = len(self.clips)
        try:
            self._resolve_compound_story(
                source.video_story,
                parent_absolute=absolute,
                parent_source=source_window.start,
                inherited_lane=lane,
                ancestor_group_ids=source_ancestors,
                visible_start=absolute,
                visible_end=absolute + source_window.duration,
            )
        finally:
            self._canvas_context = outer_canvas
        if (
            len(self.clips) == clip_count_before
            and not self._story_interval_is_explicit_gap(
                source.video_story,
                source_window.start,
                source_window.end,
                source_start=source.source_start,
            )
        ):
            raise FCPXMLCompileError(
                f"{node.path} resolved video source {source.id!r} has no "
                "video stream in its requested range"
            )
        return outer

    @staticmethod
    def _story_interval_is_explicit_gap(
        story: tuple[StoryNode, ...],
        requested_start: Fraction,
        requested_end: Fraction,
        *,
        source_start: Fraction,
    ) -> bool:
        """Prove that a virtual video interval intentionally contains no image.

        Main callers:
        - ``_resolve_virtual_video_source`` when selected multicam execution
          emits no inner video clips.

        Why this exists:
        - Final Cut multicam angles can begin with explicit gaps. A clip wholly
          inside such a gap is transparent by construction; only uncovered or
          partially covered empty ranges remain errors.
        """

        cursor = source_start
        gaps: list[tuple[Fraction, Fraction]] = []
        for child in story:
            offset = child.offset if child.offset is not None else cursor
            end = offset + child.duration
            if child.kind == "gap" and child.enabled:
                gaps.append((offset, end))
            cursor = end
        covered_until = requested_start
        for gap_start, gap_end in sorted(gaps):
            if gap_end <= covered_until:
                continue
            if gap_start > covered_until:
                return False
            covered_until = max(covered_until, gap_end)
            if covered_until >= requested_end:
                return True
        return False

    def _resolve_compound_story(
        self,
        nodes: tuple[StoryNode, ...],
        *,
        parent_absolute: Fraction,
        parent_source: Fraction,
        inherited_lane: int,
        ancestor_group_ids: tuple[str, ...],
        visible_start: Fraction,
        visible_end: Fraction,
    ) -> None:
        """Clip one reusable sequence to its ref range, then compile it.

        The source node copies are build-local.  Their offsets remain in the
        resource source domain while leading trims advance ``start`` by the
        same exact amount, preserving file-media source selection.
        """

        clipped = self._clip_story_nodes(
            nodes,
            parent_absolute=parent_absolute,
            parent_source=parent_source,
            visible_start=visible_start,
            visible_end=visible_end,
            sequential=True,
        )
        self._resolve_storyline(
            clipped,
            parent_absolute=parent_absolute,
            parent_source=parent_source,
            inherited_lane=inherited_lane,
            ancestor_clip_ids=ancestor_group_ids,
            sequential=True,
        )

    def _clip_story_nodes(
        self,
        nodes: tuple[StoryNode, ...],
        *,
        parent_absolute: Fraction,
        parent_source: Fraction,
        visible_start: Fraction,
        visible_end: Fraction,
        sequential: bool,
    ) -> tuple[StoryNode, ...]:
        cursor = parent_source
        result = []
        for node in nodes:
            offset = (
                node.offset
                if node.offset is not None
                else cursor if sequential else parent_source
            )
            raw_absolute = parent_absolute + offset - parent_source
            raw_end = raw_absolute + node.duration
            start = max(raw_absolute, visible_start)
            end = min(raw_end, visible_end)
            if end > start:
                leading = start - raw_absolute
                children = self._clip_story_nodes(
                    node.children,
                    parent_absolute=raw_absolute,
                    parent_source=node.start,
                    visible_start=start,
                    visible_end=end,
                    sequential=node.kind == "spine",
                ) if node.children else ()
                result.append(
                    replace(
                        node,
                        offset=parent_source + (start - parent_absolute),
                        start=node.start + leading,
                        duration=end - start,
                        children=children,
                    )
                )
            if sequential:
                cursor = offset + node.duration
        return tuple(result)

    def _compile_clip(
        self,
        node: StoryNode,
        absolute: Fraction,
        lane: int,
        order: int,
        ancestor_clip_ids: tuple[str, ...],
    ) -> RenderClip:
        asset = None
        media_path = None
        used_binding = None
        has_video = node.kind in {"title", "caption"}
        has_audio = False
        is_still = False
        crop = node.crop
        asset_id = None
        asset_uid = None
        text_plan: Optional[TextRenderPlan] = None
        generator_plan: Optional[GeneratorRenderPlan] = None
        render_source: Optional[RenderableAVSource] = None
        video_omission: Optional[RenderVideoDisposition] = None
        missing_media_locators: tuple[str, ...] = ()

        if node.kind in {"asset-clip", "video", "audio"}:
            if not node.ref:
                raise FCPXMLCompileError(f"{node.path} requires a resource ref")
            asset = self.source.assets.get(node.ref)
            if asset is None:
                if node.ref in self.source.effects and node.kind == "video":
                    resource = self.source.effects[node.ref]
                    if not resource.uid:
                        raise FCPXMLCompileError(
                            f"{node.path} generator resource requires an exact UID"
                        )
                    try:
                        generator_plan = build_generator_render_plan(
                            ET.fromstring(node.raw_xml),
                            template_uid=resource.uid,
                            timeline_start=absolute,
                        )
                    except (ET.ParseError, TextPlanError) as error:
                        raise FCPXMLCompileError(
                            f"{node.path} has invalid generator controls: {error}"
                        ) from error
                    self._report_text_findings(
                        generator_plan.findings,
                        path=node.path,
                        absolute=absolute,
                        duration=node.duration,
                        uid=resource.uid,
                    )
                    if generator_plan.execution == "solid_color":
                        has_video = True
                        is_still = True
                        self.report.add(
                            outcome="approximated",
                            portable_status="calibrated_portable",
                            fcpxml_path=node.path,
                            construct=f"generator {resource.name or node.ref}",
                            uid=resource.uid,
                            timeline_start=absolute,
                            timeline_duration=node.duration,
                            disposition=(
                                "bounded exported Color renders as a stock "
                                "full-canvas solid source"
                            ),
                        )
                    else:
                        reason = (
                            generator_plan.findings[0].detail
                            if generator_plan.findings
                            else "generator has no calibrated portable adapter"
                        )
                        video_omission = RenderVideoDisposition(
                            execution="omit_transparent",
                            reason=reason,
                            portable_status="unsupported",
                            construct=f"generator {resource.name or node.ref}",
                            uid=resource.uid,
                        )
                elif any(resource.id == node.ref for resource in self.source.other_resources):
                    reason = "referenced Final Cut media container cannot be decoded portably"
                    construct = f"{node.kind} {node.name or node.ref}"
                    self.report.add(
                        outcome="omitted",
                        portable_status="apple_only",
                        fcpxml_path=node.path,
                        construct=construct,
                        timeline_start=absolute,
                        timeline_duration=node.duration,
                        disposition=reason,
                    )
                    if node.kind != "audio" and node.src_enable != "audio":
                        video_omission = RenderVideoDisposition(
                            execution="omit_transparent",
                            reason=reason,
                            portable_status="apple_only",
                            construct=construct,
                        )
                else:
                    raise FCPXMLCompileError(f"{node.path} references unknown resource {node.ref!r}")
            else:
                asset_id = asset.id
                asset_uid = asset.uid
                media_path, used_binding = resolve_asset(asset, self.bindings, self.source.media_base_dir)
                has_video = asset.has_video and node.kind != "audio" and node.src_enable != "audio"
                has_audio = asset.has_audio and node.kind != "video" and node.src_enable != "video"
                is_still = asset.has_video and (asset.duration is None or asset.duration <= 0)
                if media_path is None:
                    missing_media_locators = unresolved_asset_locators(
                        asset,
                        self.bindings,
                        self.source.media_base_dir,
                    )
                    reason = "missing source uses visible placeholder and silence: " + ", ".join(
                        missing_media_locators
                    )
                    construct = f"missing media {asset.name or asset.id}"
                    self.report.add(
                        outcome="info",
                        portable_status="degraded",
                        fcpxml_path=node.path,
                        construct=construct,
                        timeline_start=absolute,
                        timeline_duration=node.duration,
                        disposition=reason,
                    )
                elif not media_path.is_file():
                    missing_media_locators = unresolved_asset_locators(
                        asset,
                        self.bindings,
                        self.source.media_base_dir,
                    )
                    reason = "missing source uses visible placeholder and silence: " + ", ".join(
                        missing_media_locators
                    )
                    construct = f"missing media {asset.name or asset.id}"
                    self.report.add(
                        outcome="info",
                        portable_status="degraded",
                        fcpxml_path=node.path,
                        construct=construct,
                        timeline_start=absolute,
                        timeline_duration=node.duration,
                        disposition=reason,
                    )
                    media_path = None
                elif used_binding is not None and used_binding not in self.used_asset_bindings:
                    self.used_asset_bindings.append(used_binding)
                self._report_asset_metadata(node, asset, absolute)

        if crop and crop.enabled and crop.mode == "pan":
            asset_format = self.source.formats.get(asset.format_id) if asset and asset.format_id else None
            if not (
                asset_format
                and asset_format.width
                and asset_format.height
                and len(crop.active_rects) >= 2
            ):
                self.report.add(
                    outcome="omitted",
                    portable_status="unsupported",
                    fcpxml_path=f"{node.path}/adjust-crop",
                    construct="Pan/Ken Burns crop",
                    timeline_start=absolute,
                    timeline_duration=node.duration,
                    disposition=(
                        "Pan requires known source geometry and distinct start/end rectangles"
                    ),
                )
                crop = None
            else:
                self.report.add(
                    outcome="approximated",
                    portable_status="calibrated_portable",
                    fcpxml_path=f"{node.path}/adjust-crop",
                    construct="Pan/Ken Burns crop",
                    timeline_start=absolute,
                    timeline_duration=node.duration,
                    disposition=(
                        "first and last rectangles are interpolated once per output frame; "
                        "the source window owns alpha and static geometry uses one "
                        "calibrated subpixel camera warp"
                    ),
                )

        if node.kind == "title":
            if not node.ref or node.ref not in self.source.effects:
                raise FCPXMLCompileError(f"{node.path} title references unknown effect {node.ref!r}")
            title_effect = self.source.effects[node.ref]
            try:
                text_plan = build_text_render_plan(
                    ET.fromstring(node.raw_xml),
                    template_uid=title_effect.uid,
                    timeline_start=absolute,
                )
            except (ET.ParseError, TextPlanError) as error:
                raise FCPXMLCompileError(
                    f"{node.path} has invalid title text/controls: {error}"
                ) from error
            self._report_text_findings(
                text_plan.findings,
                path=node.path,
                absolute=absolute,
                duration=node.duration,
                uid=title_effect.uid,
            )
            capability = self.registry.match(kind="title", uid=title_effect.uid, name=title_effect.name)
            if text_plan.execution is None:
                reason = "opaque Motion title has no calibrated portable adapter"
                construct = title_effect.name or "title"
                self.report.add(
                    outcome="omitted",
                    portable_status="unsupported",
                    fcpxml_path=node.path,
                    construct=construct,
                    uid=title_effect.uid,
                    timeline_start=absolute,
                    timeline_duration=node.duration,
                    disposition=reason,
                )
                has_video = False
                video_omission = RenderVideoDisposition(
                    execution="omit_transparent",
                    reason=reason,
                    portable_status="unsupported",
                    construct=construct,
                    uid=title_effect.uid,
                )
            else:
                if capability is None or capability.handler != "basic_title":
                    raise FCPXMLCompileError(
                        f"{node.path} executable title adapter is missing its registry capability"
                    )
                self._report_parameter_coverage(
                    node.params,
                    capability=capability,
                    path=node.path,
                    absolute=absolute,
                    duration=node.duration,
                )
                self.report.add(
                    outcome="approximated",
                    portable_status=capability.portable_status,
                    fcpxml_path=node.path,
                    construct=title_effect.name or "Basic Title",
                    uid=title_effect.uid,
                    timeline_start=absolute,
                    timeline_duration=node.duration,
                    disposition="rasterized with portable FreeType/HarfBuzz text metrics",
                )
        elif node.kind == "caption":
            try:
                text_plan = build_text_render_plan(
                    ET.fromstring(node.raw_xml),
                    template_uid=None,
                    timeline_start=absolute,
                )
            except (ET.ParseError, TextPlanError) as error:
                raise FCPXMLCompileError(
                    f"{node.path} has invalid caption text/metadata: {error}"
                ) from error
            self._report_text_findings(
                text_plan.findings,
                path=node.path,
                absolute=absolute,
                duration=node.duration,
                uid=None,
            )
            self.report.add(
                outcome="approximated",
                portable_status="calibrated_portable",
                fcpxml_path=node.path,
                construct="native caption",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition="rendered visually as a portable text overlay; caption interchange metadata is not embedded",
            )

        blend_mode = self._blend_mode(node, absolute)
        self._report_normal_opacity_composition(node, absolute, blend_mode)
        self._report_preserved_adjustments(node, absolute)
        self._report_curve_approximations(node, absolute)
        source_format = (
            self.source.formats.get(asset.format_id)
            if asset is not None and asset.format_id is not None
            else None
        )
        if asset is not None:
            format_context = None
            if (
                source_format is not None
                and source_format.frame_duration is not None
                and source_format.width
                and source_format.height
            ):
                format_context = SequenceFormatContext.from_resource(source_format)
            render_source = resolve_file_source(
                asset,
                format_context=format_context,
            )
            try:
                retime_map = resolve_instance_stream_timing(
                    node,
                    absolute_start=absolute,
                    stream="video",
                ).retime_map
            except RenderSourceError as error:
                raise FCPXMLCompileError(str(error)) from error
        else:
            retime_map = self._retime_map(node)
        animation_time_map = self._visual_animation_time_map(node)
        speed = self._constant_speed(node, absolute)
        transform_animation = self._transform_animation(
            node, animation_time_map, absolute
        )
        opacity_animation = self._opacity_animation(node, animation_time_map, absolute)
        corner_pin = self._corner_pin(node, animation_time_map, absolute)
        spatial_intrinsics = self._spatial_intrinsics(node, asset, absolute)
        source_domain_start = retime_map.segments[0].source_start
        source_origin = asset.start if asset is not None else Fraction(0)
        if asset is not None:
            requested_source_points = tuple(
                value
                for segment in retime_map.segments
                for value in (segment.source_start, segment.source_end)
            )
            source_domain_low = min(requested_source_points)
            source_domain_end = max(requested_source_points)
            asset_end = (
                asset.start + asset.duration
                if asset.duration is not None and asset.duration > 0
                else None
            )
            source_tolerance = source_bound_tolerance(
                has_video=asset.has_video,
                frame_duration=(
                    source_format.frame_duration if source_format is not None else None
                ),
                has_audio=asset.has_audio,
                audio_rate=asset.audio_rate,
            )
            if not source_range_within_bounds(
                requested_start=source_domain_low,
                requested_end=source_domain_end,
                source_start=asset.start,
                source_end=asset_end if asset_end is not None else source_domain_end,
                tolerance=source_tolerance,
            ):
                if source_domain_low < asset.start:
                    raise FCPXMLCompileError(
                        f"{node.path} starts at {source_domain_low}, before asset "
                        f"{asset.id!r} source origin {asset.start}"
                    )
                assert asset_end is not None
                raise FCPXMLCompileError(
                    f"{node.path} source range ends at {source_domain_end}, after "
                    f"asset {asset.id!r} end {asset_end}"
                )
        source_start = source_domain_start - source_origin
        if source_origin:
            retime_map = RetimeMap(
                tuple(
                    RetimeSegment(
                        timeline_start=segment.timeline_start,
                        timeline_end=segment.timeline_end,
                        source_start=segment.source_start - source_origin,
                        source_end=segment.source_end - source_origin,
                    )
                    for segment in retime_map.segments
                )
            )
        resolved_effects = self._resolve_effects(node, absolute)
        if not node.enabled and (
            has_video
            or video_omission is not None
            or node.kind in {"title", "caption", "video"}
        ):
            video_disposition = RenderVideoDisposition(
                execution="authored_disabled"
            )
        elif video_omission is not None:
            video_disposition = video_omission
        elif has_video:
            video_disposition = RenderVideoDisposition(execution="composite")
        else:
            video_disposition = RenderVideoDisposition(execution="not_applicable")
        return RenderClip(
            id=f"clip-{order}",
            ancestor_clip_ids=ancestor_clip_ids,
            kind=node.kind,
            path=node.path,
            name=node.name,
            absolute_start=absolute,
            duration=node.duration,
            source_start=source_start,
            lane=lane,
            document_order=order,
            media_path=media_path,
            asset_id=asset_id,
            asset_uid=asset_uid,
            has_video=has_video,
            has_audio=has_audio,
            is_still=is_still,
            enabled=node.enabled,
            src_enable=node.src_enable,
            conform_type=node.conform_type,
            transform=node.transform,
            crop=crop,
            blend_opacity=max(0.0, min(node.blend_opacity, 1.0)),
            opacity_fade=node.opacity_fade,
            volume_db=node.volume_db,
            audio_fade=node.audio_fade,
            speed=speed,
            effects=resolved_effects.applied,
            params=node.params,
            text_runs=node.text_runs,
            text_styles=node.text_styles,
            retime_map=retime_map,
            transform_animation=transform_animation,
            opacity_animation=opacity_animation,
            corner_pin=corner_pin,
            spatial_intrinsics=spatial_intrinsics,
            text_plan=text_plan,
            generator_plan=generator_plan,
            blend_mode=blend_mode,
            asset_source_origin=source_origin,
            asset_source_duration=(asset.duration if asset is not None else None),
            source_frame_duration=(
                source_format.frame_duration if source_format is not None else None
            ),
            canvas_context=self._canvas_context,
            render_source_id=(render_source.id if render_source else None),
            render_source_kind=(render_source.kind if render_source else None),
            semantic_effects=resolved_effects.semantic,
            video_disposition=video_disposition,
            missing_media_locators=missing_media_locators,
        )

    def _report_text_findings(
        self,
        findings: tuple[TextFinding, ...],
        *,
        path: str,
        absolute: Fraction,
        duration: Fraction,
        uid: Optional[str],
    ) -> None:
        """Copy typed text/template decisions into the compatibility report."""

        for finding in findings:
            omitted = finding.disposition == "not_implemented_yet"
            self.report.add(
                outcome="omitted" if omitted else "approximated",
                portable_status="unsupported" if omitted else "calibrated_portable",
                fcpxml_path=path,
                construct=finding.construct,
                uid=uid,
                timeline_start=absolute,
                timeline_duration=duration,
                disposition=finding.detail,
            )

    def _spatial_intrinsics(
        self,
        node: StoryNode,
        asset,
        absolute: Fraction,
    ) -> Optional[SpatialIntrinsicPlan]:
        """Promote preserved spatial controls into one typed early-video plan.

        Main callers:
        - ``_compile_clip`` after exact source metadata resolution.

        Asset overrides take precedence over the asset format.  XML never
        supplies a LUT path, deshake option, or free-form FFmpeg expression;
        the spatial module owns every executable constant.
        """

        if asset is None or not asset.has_video:
            return None
        source_format = (
            self.source.formats.get(asset.format_id)
            if asset.format_id is not None
            else None
        )
        sequence_format = self.source.formats[self.source.sequence_format_id]
        frame_width = (
            source_format.width
            if source_format is not None and source_format.width is not None
            else sequence_format.width
        )
        frame_height = (
            source_format.height
            if source_format is not None and source_format.height is not None
            else sequence_format.height
        )
        if frame_width is None or frame_height is None:
            raise FCPXMLCompileError(
                f"{node.path} has no source dimensions for spatial execution"
            )
        color_space = asset.color_space_override or (
            source_format.color_space if source_format is not None else None
        )
        projection = asset.projection_override or (
            source_format.projection if source_format is not None else None
        ) or "none"
        stereo_layout = asset.stereoscopic_override or (
            source_format.stereoscopic if source_format is not None else None
        ) or "mono"
        hero_eye = asset.hero_eye_override or (
            source_format.hero_eye if source_format is not None else None
        )

        by_kind = {}
        trackers = []
        for adjustment in node.preserved_adjustments:
            if not adjustment.enabled:
                continue
            if adjustment.kind in {
                "adjust-volume",
                "adjust-panner",
                "adjust-loudness",
                "adjust-noiseReduction",
                "adjust-humReduction",
                "adjust-EQ",
                "adjust-matchEQ",
                "adjust-voiceIsolation",
                "audio-channel-source",
                "audio-role-source",
                "sync-source",
            }:
                # Audio components are intentionally repeatable. The
                # independent audio IR compiles each source/component and its
                # controls; the video-spatial uniqueness map must not collapse
                # or reject them as duplicate intrinsic adjustments.
                continue
            if adjustment.kind == "object-tracker":
                tracker_id = (
                    adjustment.attributes.get("id")
                    or adjustment.attributes.get("name")
                    or f"tracker-{len(trackers) + 1}"
                )
                trackers.append(
                    OpaqueTrackerLocator(
                        tracker_id=tracker_id,
                        data_locator=(
                            adjustment.attributes.get("dataLocator")
                            or adjustment.attributes.get("data-locator")
                        ),
                    )
                )
                continue
            if adjustment.kind in by_kind:
                raise FCPXMLCompileError(
                    f"{node.path} has duplicate {adjustment.kind} adjustments"
                )
            by_kind[adjustment.kind] = adjustment

        try:
            display = DisplayConform(
                pixel_aspect_h=(
                    source_format.pixel_aspect_h
                    if source_format and source_format.pixel_aspect_h
                    else 1
                ),
                pixel_aspect_v=(
                    source_format.pixel_aspect_v
                    if source_format and source_format.pixel_aspect_v
                    else 1
                ),
            )
            color_adjustment = by_kind.get("adjust-colorConform")
            color_conform = (
                ColorConform.from_attributes(
                    color_adjustment.attributes,
                    source_color_space=color_space,
                )
                if color_adjustment is not None
                else (
                    ColorConform(
                        source_color_space=color_space,
                        mode="conformAuto",
                    )
                    if classify_fcp_color_space(color_space)
                    not in {None, "rec709"}
                    else None
                )
            )
            stereo_adjustment = by_kind.get("adjust-stereo-3D")
            stereo = (
                Stereo3DAdjustment.from_attributes(
                    stereo_adjustment.attributes,
                    input_layout=stereo_layout,
                    hero_eye=hero_eye,
                )
                if stereo_adjustment is not None
                else None
            )
            stabilization_adjustment = by_kind.get("adjust-stabilization")
            stabilization = (
                Stabilization.from_attributes(stabilization_adjustment.attributes)
                if stabilization_adjustment is not None
                else None
            )
            transform_360_adjustment = by_kind.get("adjust-360-transform")
            transform_360 = (
                Transform360.from_attributes(transform_360_adjustment.attributes)
                if transform_360_adjustment is not None
                else None
            )
            reorient_adjustment = by_kind.get("adjust-reorient")
            reorientation = (
                Reorientation360.from_attributes(
                    reorient_adjustment.attributes,
                    input_projection=projection,
                )
                if reorient_adjustment is not None
                else None
            )
            orientation_adjustment = by_kind.get("adjust-orientation")
            orientation = (
                Orientation360.from_attributes(
                    orientation_adjustment.attributes,
                    input_projection=projection,
                    output_width=sequence_format.width,
                    output_height=sequence_format.height,
                )
                if orientation_adjustment is not None
                else None
            )
            rolling_adjustment = by_kind.get("adjust-rollingShutter")
            rolling = (
                RollingShutterAdjustment.from_attributes(
                    rolling_adjustment.attributes
                )
                if rolling_adjustment is not None
                else None
            )
            cinematic_adjustment = by_kind.get("adjust-cinematic")
            cinematic = (
                OpaqueCinematicLocator(
                    data_locator=(
                        cinematic_adjustment.attributes.get("dataLocator")
                        or cinematic_adjustment.attributes.get("data-locator")
                    )
                )
                if cinematic_adjustment is not None
                else None
            )
            plan = SpatialIntrinsicPlan(
                frame_width=frame_width,
                frame_height=frame_height,
                display=display,
                color_conform=color_conform,
                stereo=stereo,
                stabilization=stabilization,
                transform_360=transform_360,
                reorientation_360=reorientation,
                orientation_360=orientation,
                rolling_shutter=rolling,
                cinematic=cinematic,
                opaque_trackers=tuple(trackers),
            )
        except SpatialIntrinsicError as error:
            raise FCPXMLCompileError(
                f"{node.path} has invalid spatial intrinsic metadata: {error}"
            ) from error
        active = any(
            (
                display.rotation_degrees,
                (display.pixel_aspect_h, display.pixel_aspect_v) != (1, 1),
                color_conform,
                stereo,
                stabilization,
                transform_360,
                reorientation,
                orientation,
                rolling,
                cinematic,
                trackers,
            )
        )
        return plan if active else None

    def _blend_mode(self, node: StoryNode, absolute: Fraction) -> Optional[str]:
        """Resolve one blend mode without an unknown-to-Normal fallback."""

        try:
            specification = resolve_blend_mode(node.blend_mode)
        except UnknownBlendModeError as error:
            raise FCPXMLCompileError(f"{node.path}: {error}") from error
        except UnsupportedBlendModeError as error:
            self.report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path=f"{node.path}/adjust-blend",
                construct=f"blend mode {node.blend_mode}",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=f"{error}; Normal alpha composition is used and explicitly reported",
            )
            return None
        if specification.semantic_status == "stock_ffmpeg_semantic_approximation":
            self.report.add(
                outcome="approximated",
                portable_status="calibrated_portable",
                fcpxml_path=f"{node.path}/adjust-blend",
                construct=f"blend mode {specification.canonical_name}",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition="executed with the reviewed stock-FFmpeg RGB/luma mapping",
            )
        return specification.canonical_name

    def _report_normal_opacity_composition(
        self,
        node: StoryNode,
        absolute: Fraction,
        blend_mode: Optional[str],
    ) -> None:
        """Report the calibrated working space for active Normal opacity.

        Main callers:
        - ``_compile_group_scope`` for container opacity.
        - ``_compile_clip`` for ordinary layer opacity.

        Why this exists:
        Final Cut's Normal blend does not multiply display-coded RGB directly.
        A serialized Final Cut 12.3 four-patch Rec.709 oracle measured a simple
        power-linear source-over response.  The portable exponent is tightly
        calibrated but remains an empirical semantic approximation, so every
        active use must remain visible in the compatibility report.  The
        sanitized measurements and private-artifact hashes are frozen in
        ``evidence/normal_opacity_calibration.v1.json``.
        """

        # ``None`` is the FCPXML default Normal mode.  An explicitly named mode
        # that resolved to ``None`` is an unsupported cross-channel fallback;
        # ``_blend_mode`` already reports that separate omission.
        if blend_mode != "Normal" and node.blend_mode is not None:
            return
        if (
            node.blend_opacity == 1.0
            and not node.blend_keyframes
            and node.opacity_fade is None
        ):
            return
        self.report.add(
            outcome="approximated",
            portable_status="calibrated_portable",
            fcpxml_path=f"{node.path}/adjust-blend",
            construct="Normal opacity working space",
            timeline_start=absolute,
            timeline_duration=node.duration,
            disposition=(
                "executed with power-linear Normal source-over (gamma 1.94; "
                "continuous oracle fit 1.9315 with measured FFmpeg LUT compensation), "
                "calibrated against a serialized Final Cut 12.3 standard-Rec.709 "
                "four-patch oracle within two decoded 8-bit levels per channel; "
                "this is an empirical semantic approximation, not a claim that "
                "Final Cut uses a published BT.709 transfer equation"
            ),
        )

    def _corner_pin(
        self,
        node: StoryNode,
        retime_map: RetimeMap,
        absolute: Fraction,
    ) -> Optional[CornerPinAdjustment]:
        """Promote preserved corner offsets into typed geometry IR."""

        adjustment = next(
            (
                item
                for item in node.preserved_adjustments
                if item.kind == "adjust-corners" and item.enabled
            ),
            None,
        )
        if adjustment is None:
            return None
        try:
            static = CornerPinAdjustment.from_attributes(adjustment.attributes)
            aliases = {
                "topleft": "top_left",
                "topright": "top_right",
                "botleft": "bottom_left",
                "bottomleft": "bottom_left",
                "botright": "bottom_right",
                "bottomright": "bottom_right",
            }
            tracks = {}
            notices = []
            for parameter in adjustment.params:
                target = aliases.get((parameter.name or "").casefold().replace(" ", ""))
                if target is None or not parameter.keyframes:
                    continue
                if target in tracks:
                    raise FCPXMLCompileError(
                        f"{node.path}/adjust-corners has duplicate {parameter.name!r} animation"
                    )
                authored = AnimatedVec2.from_keyframes(parameter.keyframes)
                # Final Cut 12.3's canonical corner controls use the DTD
                # default ``curve=smooth`` spelling, but measured vertex
                # motion is segment-linear. Applying the general transform
                # PCHIP kernel bends the corner by several pixels. Keep
                # explicit interpolation timing (ease/ease-in/ease-out), but
                # use the calibrated linear value path for corner vertices.
                corner_track = AnimatedVec2(
                    tuple(
                        replace(point, curve="linear")
                        for point in authored.control_points
                    )
                )
                track = map_vec2_animation(corner_track, retime_map)
                tracks[target] = track
                notices.extend(track.notices)
            animation = CornerPinAnimation(**tracks) if tracks else None
        except (AnimationValidationError, GeometryValidationError) as error:
            raise FCPXMLCompileError(
                f"{node.path} has invalid corner-pin geometry: {error}"
            ) from error
        for notice in notices:
            self.report.add(
                outcome="info",
                portable_status="unsupported",
                fcpxml_path=f"{node.path}/adjust-corners",
                construct="corner keyframe auxValue",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=notice.detail,
            )
        self.report.add(
            outcome="approximated",
            portable_status="calibrated_portable",
            fcpxml_path=f"{node.path}/adjust-corners",
            construct="corner pin",
            timeline_start=absolute,
            timeline_duration=node.duration,
            disposition=(
                "four corner offsets and genuine nested keyframes execute through "
                "stock perspective on the exact project-frame clock; canonical "
                "default corner curves use the measured segment-linear value path"
            ),
        )
        return replace(static, animation=animation)

    def _opacity_animation(
        self,
        node: StoryNode,
        retime_map: RetimeMap,
        absolute: Fraction,
    ):
        """Map genuine opacity automation with Final Cut's segment ownership.

        Final Cut attaches ``interp`` on an opacity point to the segment that
        leaves that point. The shared animation kernel consumes interpolation
        from the destination point, so this boundary shifts each authored
        value forward by one control point. Canonical missing value curves are
        component-linear for this parameter, matching the measured XYZT
        opacity trajectory; explicitly authored curves remain preserved.
        """

        if not node.blend_keyframes:
            return None
        keyframes = tuple(node.blend_keyframes)
        canonical_keyframes = tuple(
            replace(
                keyframe,
                interp=(keyframes[index - 1].interp if index else keyframe.interp),
                curve="linear" if keyframe.curve is None else keyframe.curve,
            )
            for index, keyframe in enumerate(keyframes)
        )
        try:
            track = map_scalar_animation(
                AnimatedScalar.from_keyframes(canonical_keyframes),
                retime_map,
            )
        except AnimationValidationError as error:
            raise FCPXMLCompileError(
                f"{node.path} has invalid opacity animation: {error}"
            ) from error
        for notice in track.notices:
            self.report.add(
                outcome="info",
                portable_status="unsupported",
                fcpxml_path=f"{node.path}/adjust-blend",
                construct="opacity keyframe auxValue",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=notice.detail,
            )
        return track

    def _transform_animation(
        self,
        node: StoryNode,
        retime_map: RetimeMap,
        absolute: Fraction,
    ) -> Optional[RenderTransformAnimation]:
        """Compile genuine nested transform tracks into the typed v2 kernel.

        Final Cut's canonical transform keyframes often omit ``curve``.  The
        DTD calls that default ``smooth``, but the measured XYZT oracle moves
        position, scale, rotation, and anchor component-linearly between those
        points.  Preserve an explicitly authored smooth curve; only normalize
        the canonical omission at this transform-specific boundary.
        """

        transform = node.transform
        if transform is None or not transform.enabled:
            return None
        notices = []

        def canonical_transform_keyframes(keyframes):
            return tuple(
                replace(keyframe, curve="linear")
                if keyframe.curve is None
                else keyframe
                for keyframe in keyframes
            )

        try:
            position = (
                map_vec2_animation(
                    AnimatedVec2.from_keyframes(
                        canonical_transform_keyframes(transform.position_keyframes)
                    ),
                    retime_map,
                )
                if transform.position_keyframes
                else None
            )
            scale = (
                map_vec2_animation(
                    AnimatedVec2.from_keyframes(
                        canonical_transform_keyframes(transform.scale_keyframes)
                    ),
                    retime_map,
                )
                if transform.scale_keyframes
                else None
            )
            rotation = (
                map_scalar_animation(
                    AnimatedScalar.from_keyframes(
                        canonical_transform_keyframes(transform.rotation_keyframes)
                    ),
                    retime_map,
                )
                if transform.rotation_keyframes
                else None
            )
            anchor = (
                map_vec2_animation(
                    AnimatedVec2.from_keyframes(
                        canonical_transform_keyframes(transform.anchor_keyframes)
                    ),
                    retime_map,
                )
                if transform.anchor_keyframes
                else None
            )
        except AnimationValidationError as error:
            raise FCPXMLCompileError(
                f"{node.path} has invalid transform animation: {error}"
            ) from error
        for track in (position, scale, rotation, anchor):
            if track is not None:
                notices.extend(track.notices)
        for notice in notices:
            self.report.add(
                outcome="info",
                portable_status="unsupported",
                fcpxml_path=f"{node.path}/adjust-transform",
                construct="transform keyframe auxValue",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=notice.detail,
            )
        if not any((position, scale, rotation, anchor)):
            return None
        return RenderTransformAnimation(
            position=position,
            scale=scale,
            rotation=rotation,
            anchor=anchor,
            notices=tuple(notices),
        )

    def _retime_map(self, node: StoryNode) -> RetimeMap:
        """Build the exact v2 clip-local output-to-source mapping.

        Main callers:
        - ``_compile_clip`` for every ordinary media/title/caption item.

        The legacy ``speed`` field remains temporarily for the v1 FFmpeg
        bridge. This exact map is authoritative for Wave 2 segment execution
        and never reduces ramps, reverse sections, or freezes to an average.
        """

        if not node.time_map:
            return RetimeMap.identity(
                node.duration,
                source_start=node.start,
            )
        try:
            return RetimeMap.from_time_map_points(
                node.time_map,
                linearize_two_point_smooth2=True,
            ).restrict_to_timeline_window(node.start, node.duration)
        except UnsupportedRetimeMappingError as error:
            raise FCPXMLCompileError(
                f"{node.path} requests a nonlinear timeMap that the exact v2 retime kernel cannot execute: {error}"
            ) from error
        except RetimeValidationError as error:
            raise FCPXMLCompileError(f"{node.path} has an invalid timeMap: {error}") from error

    @staticmethod
    def _visual_animation_time_map(node: StoryNode) -> RetimeMap:
        """Build Final Cut's clip-output clock for visual parameters.

        Main callers:
        - ``_compile_clip`` and ``_compile_group_scope`` before transform,
          opacity, and corner-pin tracks are promoted into render IR.

        Why this exists:
        A media ``timeMap`` changes which source frame is visible, but measured
        Final Cut projects keep spatial and opacity keyframes moving across the
        clip's visible output duration.  Reusing the media map here freezes,
        reverses, or accelerates geometry together with the footage.  The
        identity map retains the node's source-domain keyframe origin while
        deliberately separating the visual clock from media retiming.
        """

        return RetimeMap.identity(node.duration, source_start=node.start)

    def _report_sequence_metadata(self, sequence_format) -> None:
        """Report output-format controls that the current graph normalizes.

        Main callers:
        - ``compile`` after the project format has passed basic validation.

        Why this exists:
        FFmpeg currently emits square-pixel progressive Rec.709 video and
        stereo 48 kHz audio.  A non-default Final Cut project format must be
        visible as a compatibility decision until the v2 output contract owns
        those values.
        """

        unsupported: list[str] = []
        if sequence_format.field_order:
            unsupported.append(f"fieldOrder={sequence_format.field_order}")
        if (sequence_format.pixel_aspect_h, sequence_format.pixel_aspect_v) not in {
            (None, None),
            (1, 1),
        }:
            unsupported.append(
                f"pixelAspect={sequence_format.pixel_aspect_h}:{sequence_format.pixel_aspect_v}"
            )
        if sequence_format.projection:
            unsupported.append(f"projection={sequence_format.projection}")
        if sequence_format.stereoscopic:
            unsupported.append(f"stereoscopic={sequence_format.stereoscopic}")
        if sequence_format.hero_eye:
            unsupported.append(f"heroEye={sequence_format.hero_eye}")
        if unsupported:
            self.report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path="resources/format",
                construct="sequence spatial format",
                disposition="current output normalizes these format controls: " + ", ".join(unsupported),
            )

        if self.source.sequence_render_format:
            self.report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path="sequence",
                construct="sequence renderFormat",
                disposition=(
                    f"Final Cut render format {self.source.sequence_render_format!r} is preserved; "
                    "the portable encoder profile is selected independently"
                ),
            )

    def _validate_compound_format_context(
        self,
        path: str,
        context: SequenceFormatContext,
    ) -> None:
        """Fail before rendering local sequence controls we cannot yet own.

        Main callers:
        - ``_resolve_storyline`` before descending through a ``ref-clip``.

        The P0 renderer owns a compound's raster dimensions and cadence. It
        does not yet own a compound-local anamorphic, interlaced, projected,
        stereoscopic, or distinct color-processing surface. Rejecting those
        controls prevents an outer-project default from silently replacing
        the reusable sequence's declared format.
        """

        unsupported: list[str] = []
        if context.field_order:
            unsupported.append(f"fieldOrder={context.field_order}")
        if (context.pixel_aspect_h, context.pixel_aspect_v) not in {
            (None, None),
            (1, 1),
        }:
            unsupported.append(
                f"pixelAspect={context.pixel_aspect_h}:{context.pixel_aspect_v}"
            )
        if context.projection:
            unsupported.append(f"projection={context.projection}")
        if context.stereoscopic:
            unsupported.append(f"stereoscopic={context.stereoscopic}")
        if context.hero_eye:
            unsupported.append(f"heroEye={context.hero_eye}")
        outer = self._canvas_context
        if (
            outer is not None
            and context.color_space
            and outer.color_space
            and context.color_space != outer.color_space
        ):
            unsupported.append(
                f"local colorSpace={context.color_space!r} differs from parent "
                f"{outer.color_space!r}"
            )
        if unsupported:
            raise FCPXMLCompileError(
                f"{path} compound local format is explicitly unsupported: "
                + ", ".join(unsupported)
            )

    def _report_asset_metadata(self, node: StoryNode, asset, absolute: Fraction) -> None:
        """Report source-stream and override metadata not represented by RenderClip."""

        stream_controls: list[str] = []
        if asset.video_sources not in {None, 0, 1}:
            stream_controls.append(f"videoSources={asset.video_sources}")
        if stream_controls:
            self.report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path=node.path,
                construct="asset stream topology",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=(
                    "the first decodable video stream is used: "
                    + ", ".join(stream_controls)
                ),
            )

        if asset.custom_lut_override:
            self.report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path=node.path,
                construct="asset custom LUT override",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=(
                    f"customLUTOverride={asset.custom_lut_override} is preserved; "
                    "opaque or external LUT data is not applied"
                ),
            )

        typed_overrides = {
            "colorSpaceOverride": asset.color_space_override,
            "projectionOverride": asset.projection_override,
            "stereoscopicOverride": asset.stereoscopic_override,
            "heroEyeOverride": asset.hero_eye_override,
        }
        active_overrides = [
            f"{name}={value}" for name, value in typed_overrides.items() if value
        ]
        if active_overrides:
            self.report.add(
                outcome="info",
                portable_status="exact_portable",
                fcpxml_path=node.path,
                construct="asset spatial/color overrides",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=(
                    "consumed as typed input metadata by the spatial plan: "
                    + ", ".join(active_overrides)
                ),
            )

    def _report_preserved_adjustments(self, node: StoryNode, absolute: Fraction) -> None:
        """Report active intrinsic behavior not consumed by the current graph.

        Main callers:
        - ``_compile_clip`` after the typed static adjustment fields are
          classified.

        Why this exists:
        FCPXML permits many render-affecting intrinsic children.  Previously
        the parser silently discarded most of them, so a successful render
        could look valid while losing audio routing, corner geometry, or
        stabilization.  Preserved source records make that impossible.
        """

        for adjustment in node.preserved_adjustments:
            if not adjustment.enabled or _preserved_adjustment_is_noop(adjustment):
                continue
            kind = adjustment.kind
            if kind in {
                "adjust-volume",
                "adjust-panner",
                "adjust-loudness",
                "adjust-noiseReduction",
                "adjust-humReduction",
                "adjust-EQ",
                "adjust-matchEQ",
                "adjust-voiceIsolation",
                "audio-channel-source",
                "audio-role-source",
                "sync-source",
                "object-tracker",
                "adjust-stabilization",
                "adjust-rollingShutter",
                "adjust-reorient",
                "adjust-orientation",
                "adjust-cinematic",
                "adjust-colorConform",
                "adjust-stereo-3D",
                "adjust-360-transform",
            }:
                # The independent audio and spatial plans own execution and
                # typed findings for these surfaces. Reporting them here as a
                # generic omission would contradict the later graph decision.
                continue
            if kind in {"adjust-conform"}:
                continue
            if kind == "adjust-transform":
                has_tracking = bool(adjustment.attributes.get("tracking"))
                if not has_tracking:
                    continue
                detail = "opaque tracking reference execution is not implemented yet"
            elif kind == "adjust-corners":
                # ``_corner_pin`` owns static and animated perspective
                # execution plus its compatibility finding.
                continue
            elif kind == "adjust-crop":
                if not any(param.keyframes for param in adjustment.params):
                    continue
                detail = "animated crop/trim rectangle parameters are not applied"
            elif kind == "adjust-blend":
                animated = [param for param in adjustment.params if param.keyframes]
                if not animated or all(
                    (param.name or "").casefold() == "amount" for param in animated
                ):
                    continue
                detail = "an unknown non-opacity blend parameter animation is not applied"
            else:
                detail = "this active FCPXML intrinsic adjustment is not implemented yet"
            self.report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path=f"{node.path}/{kind}",
                construct=kind,
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=detail,
            )

    def _resolve_effects(
        self,
        node: StoryNode,
        absolute: Fraction,
    ) -> _ResolvedEffectSet:
        """Resolve CPU effects while retaining explicit identity omissions.

        Main callers:
        - File-backed clips and group-scope construction.

        Why this exists:
        ``effects`` remains the unchanged executable CPU subset. ``semantic``
        preserves every enabled authored position, including Magnetic Mask and
        other registry-declared identity omissions, for CompositionPlan.
        """

        applied: list[ResolvedEffect] = []
        semantic: list[ResolvedEffect] = []
        kind_counts: dict[str, int] = {}
        for instance in node.filters:
            if not instance.enabled:
                continue
            kind_counts[instance.kind] = kind_counts.get(instance.kind, 0) + 1
            if isinstance(instance, MaskedFilterInstance):
                masked = self._resolve_masked_effect(node, instance, absolute)
                if masked is not None:
                    semantic.append(masked)
                    if masked.execution == "apply":
                        applied.append(masked)
                continue
            resolved = self._resolve_filter_instance(
                node,
                instance,
                absolute,
                path=f"{node.path}/{instance.kind}[{kind_counts[instance.kind]}]",
            )
            if resolved is not None:
                semantic.append(resolved)
                if resolved.execution == "apply":
                    applied.append(resolved)
        return _ResolvedEffectSet(tuple(applied), tuple(semantic))

    def _resolve_masked_effect(
        self,
        node: StoryNode,
        instance: MaskedFilterInstance,
        absolute: Fraction,
    ) -> Optional[ResolvedEffect]:
        """Compile one standard masked-filter group without guessing opaque data.

        Main callers:
        - ``_resolve_effects`` in document order.

        Why this exists:
        The optional second child filter represents the outside correction; it
        is not another serial effect. Keeping the group intact lets FFmpeg blend
        inside and outside results using one validated mask plane.
        """

        path = f"{node.path}/filter-video-mask"
        if not 1 <= len(instance.filters) <= 2:
            raise FCPXMLCompileError(f"{path} requires one inside filter and at most one outside filter")
        try:
            resolution = resolve_mask_group(instance.masks, inverted=instance.inverted)
        except MaskResolutionError as exc:
            reason = (
                f"portable mask rejected: {exc}; underlying clip remains unchanged"
            )
            self.report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path=path,
                construct="masked video filter",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=reason,
            )
            return ResolvedEffect(
                kind="masked_video_filter",
                uid=None,
                name="masked video filter",
                handler=None,
                portable_status="unsupported",
                params=tuple(
                    parameter
                    for child in instance.filters
                    for parameter in child.params
                ),
                calibration={},
                data={},
                path=path,
                execution="identity",
                omission_reason=reason,
            )

        inside = self._resolve_filter_instance(
            node,
            instance.filters[0],
            absolute,
            path=f"{path}/filter-video[1]",
        )
        if inside is None or inside.execution != "apply":
            reason = (
                "inside effect has no portable handler; mask group was not applied"
            )
            self.report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path=path,
                construct="masked video filter",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=reason,
            )
            return ResolvedEffect(
                kind="masked_video_filter",
                uid=inside.uid if inside is not None else None,
                name="masked video filter",
                handler=None,
                portable_status=(
                    inside.portable_status if inside is not None else "unsupported"
                ),
                params=inside.params if inside is not None else (),
                calibration={},
                data={},
                path=path,
                execution="identity",
                capability_id=(
                    inside.capability_id if inside is not None else None
                ),
                omission_reason=reason,
            )
        outside = None
        if len(instance.filters) == 2 and instance.filters[1].enabled:
            outside = self._resolve_filter_instance(
                node,
                instance.filters[1],
                absolute,
                path=f"{path}/filter-video[2]",
            )
        for note in resolution.approximations:
            self.report.add(
                outcome="approximated",
                portable_status="calibrated_portable",
                fcpxml_path=path,
                construct="portable mask geometry",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=note,
            )
        return replace(inside, mask=resolution.group, outside_effect=outside)

    def _resolve_filter_instance(
        self,
        node: StoryNode,
        instance: FilterInstance,
        absolute: Fraction,
        *,
        path: str,
    ) -> Optional[ResolvedEffect]:
        """Resolve an ordinary child filter for serial or masked application."""

        if not instance.enabled:
            return None
        if not instance.ref or instance.ref not in self.source.effects:
            raise FCPXMLCompileError(f"{node.path} filter references unknown effect {instance.ref!r}")
        resource = self.source.effects[instance.ref]
        kind = "video_filter" if instance.kind == "filter-video" else "audio_filter"
        capability = self.registry.match(kind=kind, uid=resource.uid, name=resource.name)

        def ignored_effect(
            reason: str,
            *,
            portable_status: str,
        ) -> ResolvedEffect:
            """Preserve this exact authored slot as a semantic identity op."""

            return ResolvedEffect(
                kind=kind,
                uid=resource.uid,
                name=resource.name or instance.name,
                handler=None,
                portable_status=portable_status,
                params=instance.params,
                calibration=(capability.parameters if capability else {}),
                data=instance.data,
                path=path,
                execution="identity",
                capability_id=(capability.id if capability else None),
                omission_reason=reason,
            )

        if capability is None or capability.handler is None:
            omission = (
                capability.omission
                if capability
                else "unknown filter omitted; underlying clip remains"
            )
            self.report.add(
                    outcome="omitted",
                    portable_status=capability.portable_status if capability else "unsupported",
                    fcpxml_path=path,
                    construct=resource.name or instance.name or instance.kind,
                    uid=resource.uid,
                    timeline_start=absolute,
                    timeline_duration=node.duration,
                    disposition=omission,
            )
            return ignored_effect(
                omission,
                portable_status=(
                    capability.portable_status if capability else "unsupported"
                ),
            )
        basic_reason = unsupported_basic_effect_reason(capability.handler, instance.params)
        if basic_reason is not None:
            omission = f"{basic_reason}; underlying clip remains unchanged"
            self.report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path=path,
                construct=resource.name or instance.name or instance.kind,
                uid=resource.uid,
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=omission,
            )
            return ignored_effect(omission, portable_status="unsupported")
        cohort_reason = unsupported_cohort_effect_reason(
            capability.handler,
            instance.params,
            capability.parameters,
            instance.data,
        )
        if cohort_reason is not None:
            omission = (
                f"portable cohort effect rejected input: {cohort_reason}; "
                "underlying clip remains unchanged"
            )
            self.report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path=path,
                construct=resource.name or instance.name or instance.kind,
                uid=resource.uid,
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=omission,
            )
            return ignored_effect(omission, portable_status="unsupported")
        self._report_parameter_coverage(
                instance.params,
                capability=capability,
                path=path,
                absolute=absolute,
                duration=node.duration,
        )
        color_reason = unsupported_color_reason(
            capability.handler,
            instance.params,
            sequence_color_space=self.source.formats[self.source.sequence_format_id].color_space,
        )
        if color_reason is not None:
            omission = f"{color_reason}; underlying clip remains unchanged"
            self.report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path=path,
                construct=resource.name or instance.name or instance.kind,
                uid=resource.uid,
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=omission,
            )
            return ignored_effect(omission, portable_status="unsupported")
        resolved_data = instance.data
        artifact_id = None
        artifact_version = None
        parameter_values = {}
        if capability.handler == "spell_effect_vulkan":
            from ..effects.contract import parse_parameter_specs, resolve_parameter_values

            artifact = capability.effect_artifact
            assert artifact is not None
            specs = parse_parameter_specs(capability.parameters)
            parameter_values = resolve_parameter_values(specs, instance.params)
            artifact_id = str(artifact["id"])
            artifact_version = int(artifact["version"])
        if capability.handler == "green_screen_keyer":
            try:
                keyer = resolve_green_screen_keyer(instance.data, instance.params)
            except GreenScreenKeyerDataError as exc:
                omission = (
                    f"portable keyer rejected effectData: {exc}; "
                    "underlying clip remains opaque"
                )
                self.report.add(
                        outcome="omitted",
                        portable_status="unsupported",
                        fcpxml_path=path,
                        construct=resource.name or "Green Screen Keyer",
                        uid=resource.uid,
                        timeline_start=absolute,
                        timeline_duration=node.duration,
                        disposition=omission,
                )
                return ignored_effect(omission, portable_status="unsupported")
            resolved_data = keyer.settings.as_data()
            if keyer.used_default_key_color:
                self.report.add(
                        outcome="approximated",
                        portable_status="calibrated_portable",
                        fcpxml_path=f"{path}/data[@key='effectData']",
                        construct="Green Screen Keyer sample color",
                        uid=resource.uid,
                        timeline_start=absolute,
                        timeline_duration=node.duration,
                        disposition=(
                            "effectData exposed no bounded sample color; portable keyer used explicit green RGB (0, 1, 0)"
                        ),
                )
            if keyer.ignored_control_names:
                self.report.add(
                        outcome="omitted",
                        portable_status="unsupported",
                        fcpxml_path=f"{path}/data[@key='effectData']",
                        construct="Green Screen Keyer opaque controls",
                        uid=resource.uid,
                        timeline_start=absolute,
                        timeline_duration=node.duration,
                        disposition=(
                            "portable keyer ignored unsupported groups: "
                            + ", ".join(keyer.ignored_control_names)
                        ),
                )
        outcome = "exact" if capability.portable_status == "exact_portable" else "approximated"
        if outcome == "approximated":
            self.report.add(
                    outcome=outcome,
                    portable_status=capability.portable_status,
                    fcpxml_path=path,
                    construct=resource.name or instance.name or instance.kind,
                    uid=resource.uid,
                    timeline_start=absolute,
                    timeline_duration=node.duration,
                    disposition=(
                        capability.approximation
                        or f"mapped through portable handler {capability.handler}"
                    ),
            )
        return ResolvedEffect(
            kind=kind,
            uid=resource.uid,
            name=resource.name or instance.name,
            handler=capability.handler,
            portable_status=capability.portable_status,
            params=instance.params,
            calibration=capability.parameters,
            data=resolved_data,
            path=path,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            parameter_values=parameter_values,
            execution="apply",
            capability_id=capability.id,
            )

    def _compile_transition(
        self,
        node: StoryNode,
        absolute: Fraction,
        *,
        ancestor_group_ids: tuple[str, ...] = (),
    ) -> RenderTransition:
        video_filter = next((item for item in node.filters if item.kind == "filter-video" and item.enabled), None)
        resource = self.source.effects.get(video_filter.ref) if video_filter and video_filter.ref else None
        if video_filter and resource is None:
            raise FCPXMLCompileError(f"{node.path} transition references unknown effect {video_filter.ref!r}")
        capability = self.registry.match(
            kind="transition",
            uid=resource.uid if resource else None,
            name=(resource.name if resource else node.name),
        )
        handler = capability.handler if capability else None
        portable_status = capability.portable_status if capability else "unsupported"
        omission_reason = (
            capability.omission
            if capability is not None and handler is None
            else "unknown transition becomes a hard cut"
            if handler is None
            else None
        )
        artifact_id = None
        artifact_version = None
        xfade_id = None
        parameter_values = {}
        if handler is None:
            self.report.add(
                outcome="omitted",
                portable_status=portable_status,
                fcpxml_path=node.path,
                construct=(resource.name if resource else node.name) or "transition",
                uid=resource.uid if resource else None,
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=omission_reason,
            )
        else:
            assert capability is not None
            if handler == "spell_transition_vulkan":
                from ..transitions.contract import parse_parameter_specs, resolve_parameter_values

                artifact = capability.transition_artifact
                assert artifact is not None
                specs = parse_parameter_specs(capability.parameters)
                parameter_values = resolve_parameter_values(
                    specs,
                    video_filter.params if video_filter else (),
                )
                artifact_id = str(artifact["id"])
                artifact_version = int(artifact["version"])
            elif handler == "xfade":
                from ..transitions.contract import (
                    parse_parameter_specs,
                    resolve_parameter_values,
                    semantic_parameter_values,
                    supplied_ignored_parameter_values,
                    applied_semantic_parameter_aliases,
                )

                assert capability.xfade is not None
                specs = parse_parameter_specs(capability.parameters, max_slots=None)
                supplied_parameters = video_filter.params if video_filter else ()
                resolved_values = resolve_parameter_values(
                    specs, supplied_parameters,
                )
                for ignored_spec, ignored_value in supplied_ignored_parameter_values(
                    specs, supplied_parameters, resolved_values
                ):
                    self.report.add(
                        outcome="info",
                        portable_status=portable_status,
                        fcpxml_path=node.path,
                        construct=f"{(resource.name if resource else node.name) or 'transition'} · {ignored_spec.name}",
                        uid=resource.uid if resource else None,
                        timeline_start=absolute,
                        timeline_duration=node.duration,
                        disposition=(
                            f"validated exact default-only FCPXML key {ignored_spec.key!r} "
                            f"at {ignored_value!r}; ignored by the semantic renderer"
                        ),
                    )
                for alias_spec, alias_source, alias_target in applied_semantic_parameter_aliases(
                    specs, supplied_parameters
                ):
                    self.report.add(
                        outcome="approximated",
                        portable_status=portable_status,
                        fcpxml_path=node.path,
                        construct=f"{(resource.name if resource else node.name) or 'transition'} · {alias_spec.name}",
                        uid=resource.uid if resource else None,
                        timeline_start=absolute,
                        timeline_duration=node.duration,
                        disposition=(
                            f"validated bounded semantic alias key {alias_spec.key!r} "
                            f"at {alias_source!r} → {alias_target!r}; adaptive "
                            "Automatic semantics are not implemented"
                        ),
                    )
                parameter_values = semantic_parameter_values(specs, resolved_values)
                xfade_id = str(capability.xfade["id"])
            elif handler == "equirectangular":
                from ..transitions.equirectangular import (
                    parse_equirectangular_parameter_specs,
                    resolve_equirectangular_parameter_values,
                    semantic_parameter_values,
                )

                assert capability.xfade is not None
                xfade_id = str(capability.xfade["id"])
                specs = parse_equirectangular_parameter_specs(capability.parameters)
                parameter_values = resolve_equirectangular_parameter_values(
                    specs,
                    video_filter.params if video_filter else (),
                )
                parameter_values = semantic_parameter_values(
                    xfade_id,
                    parameter_values,
                )
            else:
                self._report_parameter_coverage(
                    video_filter.params if video_filter else (),
                    capability=capability,
                    path=node.path,
                    absolute=absolute,
                    duration=node.duration,
                )
            self.report.add(
                outcome="approximated",
                portable_status=portable_status,
                fcpxml_path=node.path,
                construct=(resource.name if resource else node.name) or "transition",
                uid=resource.uid if resource else None,
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=(
                    capability.approximation
                    or f"mapped through portable transition handler {handler}"
                ),
            )
        transition = RenderTransition(
            path=node.path,
            absolute_start=absolute,
            duration=node.duration,
            uid=resource.uid if resource else None,
            name=(resource.name if resource else node.name),
            handler=handler,
            params=video_filter.params if video_filter else (),
            capability_id=(capability.id if capability else None),
            portable_status=portable_status,
            omission_reason=omission_reason,
            xfade_id=xfade_id,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            parameter_values=parameter_values,
            ancestor_group_ids=ancestor_group_ids,
        )
        self.transitions.append(transition)
        return transition

    def _associate_transitions(self, items: list[tuple[str, object]]) -> None:
        for index, (kind, item) in enumerate(items):
            if kind != "transition" or not isinstance(item, RenderTransition):
                continue
            # Adjacency is topological, not a search for the nearest leaf.
            # Searching skipped gaps and container boundaries, which could bind
            # a transition to a more distant clip. Completed groups are entered
            # into ``items`` as one participant; unsupported gap sides therefore
            # fail visibly instead of transitioning unrelated media.
            previous = items[index - 1][1] if index > 0 and items[index - 1][0] == "clip" else None
            following = (
                items[index + 1][1]
                if index + 1 < len(items) and items[index + 1][0] == "clip"
                else None
            )
            updated = replace(
                item,
                previous_story_id=(
                    previous.id if isinstance(previous, RenderClip) else None
                ),
                next_story_id=(
                    following.id if isinstance(following, RenderClip) else None
                ),
            )
            items[index] = (kind, updated)
            for transition_index, transition in enumerate(self.transitions):
                if transition is item:
                    self.transitions[transition_index] = updated
                    break
            item = updated
            if item.handler is None:
                continue
            if isinstance(previous, RenderClip) and isinstance(following, RenderClip):
                self._apply_group_transition(previous, item, incoming=False)
                self._apply_group_transition(following, item, incoming=True)
            else:
                self.report.add(
                    outcome="omitted",
                    portable_status="unsupported",
                    fcpxml_path=item.path,
                    construct=item.name or "transition",
                    timeline_start=item.absolute_start,
                    timeline_duration=item.duration,
                    disposition="transition does not have adjacent renderable clips in its storyline",
                    uid=item.uid,
                )

    def _apply_group_transition(
        self,
        root: RenderClip,
        transition: RenderTransition,
        *,
        incoming: bool,
    ) -> None:
        """Apply a storyline transition to the root and overlapping descendants.

        Final Cut transitions the recursively composed storyline item. The MVP
        keeps a flat FFmpeg graph, so assigning the same calibrated envelope to
        every overlapping connected layer produces the equivalent alpha/motion
        behavior without requiring intermediate movie renders.

        Main callers:
        - ``_associate_transitions`` after one storyline has been resolved.
        """

        for clip in self.clips:
            if clip.id != root.id and root.id not in clip.ancestor_clip_ids:
                continue
            if clip.end <= transition.absolute_start or clip.absolute_start >= transition.end:
                continue
            if incoming:
                clip.transition_in = transition
            else:
                clip.transition_out = transition

    def _constant_speed(self, node: StoryNode, absolute: Fraction) -> Fraction:
        if len(node.time_map) < 2:
            return Fraction(1)
        retime_map = self._retime_map(node)
        if len(node.time_map) == 2 and all(
            (point.interp or "smooth2").strip().lower() == "smooth2"
            for point in node.time_map
        ):
            self.report.add(
                outcome="approximated",
                portable_status="calibrated_portable",
                fcpxml_path=f"{node.path}/timeMap",
                construct="two-point smooth2 retime",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=(
                    "the two authored endpoints are preserved exactly and the "
                    "unavailable Final Cut speed curve is linearized between them"
                ),
            )
        segments = retime_map.segments
        speed = (
            segments[0].rate
            if len(segments) == 1 and segments[0].kind == "forward"
            else Fraction(1)
        )
        if not node.time_map_preserves_pitch:
            self.report.add(
                outcome="approximated",
                portable_status="unsupported",
                fcpxml_path=f"{node.path}/timeMap",
                construct="pitch-changing retime",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=(
                    "exact video retiming is preserved; independent pitch-changing "
                    "audio execution is pending the v2 audio engine"
                ),
            )
        if node.time_map_frame_sampling not in {None, "floor"}:
            self.report.add(
                outcome="approximated",
                portable_status="calibrated_portable",
                fcpxml_path=f"{node.path}/timeMap",
                construct=f"frame sampling {node.time_map_frame_sampling}",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition="normalized output uses nearest-frame FPS sampling",
            )
        if len(segments) > 1 or any(segment.kind != "forward" for segment in segments):
            self.report.add(
                outcome="info",
                portable_status="unsupported",
                fcpxml_path=f"{node.path}/timeMap",
                construct="piecewise retime",
                timeline_start=absolute,
                timeline_duration=node.duration,
                disposition=(
                    "v2 video execution uses every exact forward, reverse, and freeze "
                    "segment; audio execution remains independently classified"
                ),
            )
        return speed

    def _report_curve_approximations(self, node: StoryNode, absolute: Fraction) -> None:
        if node.transform:
            frames = (
                *node.transform.position_keyframes,
                *node.transform.scale_keyframes,
                *node.transform.rotation_keyframes,
                *node.transform.anchor_keyframes,
            )
            if any(frame.interp not in {None, "linear"} or frame.curve not in {None, "linear"} for frame in frames):
                self.report.add(
                    outcome="approximated",
                    portable_status="calibrated_portable",
                    fcpxml_path=f"{node.path}/adjust-transform",
                    construct="nonlinear transform keyframes",
                    timeline_start=absolute,
                    timeline_duration=node.duration,
                    disposition=(
                        "Final Cut interpolation labels execute through bounded, "
                        "retime-aware FFmpeg expressions; smooth curves use "
                        "overshoot-free monotone cubic interpolation"
                    ),
                )
        for construct, envelope in (("opacity fade", node.opacity_fade), ("audio fade", node.audio_fade)):
            if envelope is None:
                continue
            kinds = {value for value in (envelope.fade_in_type, envelope.fade_out_type) if value}
            if any(value != "linear" for value in kinds):
                self.report.add(
                    outcome="approximated",
                    portable_status="calibrated_portable",
                    fcpxml_path=node.path,
                    construct=construct,
                    timeline_start=absolute,
                    timeline_duration=node.duration,
                    disposition="non-linear Final Cut fade curve rendered as a linear envelope",
                )

    def _report_parameter_coverage(
        self,
        params: tuple[Parameter, ...],
        *,
        capability: Capability,
        path: str,
        absolute: Fraction,
        duration: Fraction,
    ) -> None:
        """Report handler parameters that are ignored or outside calibration."""

        definitions = capability.parameters
        for param in params:
            if param.value is None:
                continue
            definition = definitions.get(param.key) if param.key else None
            if definition is None and param.name:
                definition = next(
                    (
                        raw
                        for raw in definitions.values()
                        if isinstance(raw, dict)
                        and str(raw.get("name", "")).casefold() == param.name.casefold()
                    ),
                    None,
                )
            label = param.name or param.key or "unnamed"
            if definition is None:
                self.report.add(
                    outcome="omitted",
                    portable_status="unsupported",
                    fcpxml_path=path,
                    construct=f"{capability.id} parameter {label}",
                    timeline_start=absolute,
                    timeline_duration=duration,
                    disposition="parameter is outside the portable handler contract and was ignored",
                )
                continue
            allowed = definition.get("allowed")
            if allowed is not None and param.value not in {str(value) for value in allowed}:
                self.report.add(
                    outcome="omitted",
                    portable_status="unsupported",
                    fcpxml_path=path,
                    construct=f"{capability.id} parameter {label}",
                    timeline_start=absolute,
                    timeline_duration=duration,
                    disposition=f"value {param.value!r} is outside the allowed portable values",
                )
                continue
            if "minimum" not in definition and "maximum" not in definition:
                component_count = int(definition.get("components", 0))
                if component_count == 0:
                    continue
            else:
                component_count = int(definition.get("components", 1))
            try:
                pieces = param.value.replace(",", " ").split()
                if len(pieces) < component_count:
                    raise ValueError
                scalars = [float(piece) for piece in pieces[:component_count]]
            except (ValueError, IndexError):
                self.report.add(
                    outcome="omitted",
                    portable_status="unsupported",
                    fcpxml_path=path,
                    construct=f"{capability.id} parameter {label}",
                    timeline_start=absolute,
                    timeline_duration=duration,
                    disposition=f"value {param.value!r} is not the expected numeric shape",
                )
                continue
            if "minimum" not in definition and "maximum" not in definition:
                continue
            minimum = float(definition.get("minimum", min(scalars)))
            maximum = float(definition.get("maximum", max(scalars)))
            if any(scalar < minimum or scalar > maximum for scalar in scalars):
                self.report.add(
                    outcome="approximated",
                    portable_status="calibrated_portable",
                    fcpxml_path=path,
                    construct=f"{capability.id} parameter {label}",
                    timeline_start=absolute,
                    timeline_duration=duration,
                    disposition=f"value {param.value!r} clamped to calibrated range [{minimum}, {maximum}]",
                )


def _validate_references(source: SourceDocument) -> None:
    """Fail compilation when a renderer-visible IDREF is unresolved.

    Main callers:
    - ``compile_fcpxml`` immediately after secure parsing.

    Why this exists:
    Unsupported story containers are preserved instead of compiled, but their
    references still must not be silently accepted as valid FCPXML.
    """

    resource_ids = set(source.formats) | set(source.assets) | set(source.effects) | set(source.multicams)
    resource_ids.update(resource.id for resource in source.other_resources if resource.id)
    for asset in source.assets.values():
        if asset.format_id is not None and asset.format_id not in source.formats:
            raise FCPXMLCompileError(f"asset {asset.id!r} references unknown format {asset.format_id!r}")
    for multicam in source.multicams.values():
        if multicam.format_id not in source.formats:
            raise FCPXMLCompileError(
                f"multicam resource {multicam.id!r} references unknown format {multicam.format_id!r}"
            )
        for angle in multicam.angles:
            for angle_node in walk_story(angle.story):
                if angle_node.ref is not None and angle_node.ref not in resource_ids:
                    raise FCPXMLCompileError(
                        f"{angle_node.path} references unknown resource {angle_node.ref!r}"
                    )
                for instance in angle_node.filters:
                    _validate_filter_references(instance, source, angle_node.path)
    for node in walk_story(source.spine):
        if node.ref is not None and node.ref not in resource_ids:
            raise FCPXMLCompileError(f"{node.path} references unknown resource {node.ref!r}")
        for instance in node.filters:
            _validate_filter_references(instance, source, node.path)


def _validate_filter_references(
    instance: FilterInstance | MaskedFilterInstance,
    source: SourceDocument,
    node_path: str,
) -> None:
    """Validate both ordinary and nested masked-filter IDREFs."""

    filters = instance.filters if isinstance(instance, MaskedFilterInstance) else (instance,)
    for child in filters:
        if child.ref is not None and child.ref not in source.effects:
            raise FCPXMLCompileError(
                f"{node_path}/{instance.kind} references unknown effect {child.ref!r}"
            )


def _preserved_adjustment_is_noop(adjustment) -> bool:
    """Recognize only documented, unambiguous intrinsic no-op states."""

    attributes = adjustment.attributes
    if adjustment.kind == "adjust-rollingShutter":
        return attributes.get("amount", "none") == "none"
    if adjustment.kind == "adjust-colorConform":
        return attributes.get("conformType", "conformNone") == "conformNone"
    if adjustment.kind == "adjust-panner":
        return attributes.get("amount", "0") in {"0", "0.0"} and not adjustment.params
    if adjustment.kind == "conform-rate":
        # Final Cut canonically emits this marker for mixed-rate clips even
        # when it is only sampling native frames on the project cadence. A
        # true rate conform remains active and is reported unsupported.
        return attributes.get("scaleEnabled", "0") == "0"
    return False
