"""Compile parsed FCPXML into a video-independent audio render plan.

Architecture map
================

``SourceDocument`` plus resolved ``RenderableAVSource`` records
    -> ordinary clip-instance stream timing (including J/L edits and retime)
    -> recursively composed file/compound/multicam source pads
    -> asset stream and component validation
    -> channel/role selection
    -> ordered gain, pan, mute, enhancement, and retime controls
    -> immutable ``AudioRenderPlan`` for the stock-FFmpeg audio engine

Important invariants
--------------------

* Audio intervals are selected independently in the shared source-time map.
  ``audioStart`` and ``audioDuration`` never create a second clock.
* Every timeline and source-time coordinate remains a ``Fraction``.  Only
  non-time control values use floats.
* A missing stream choice on a multi-stream asset is an error.  Invalid source
  channels, output channels, and role selectors are errors too; the compiler
  never guesses another component.
* Disabled clips and inactive components remain in the plan with
  ``audible == False``.  The executor can create deterministic silence without
  losing the reason that the component is silent.
* Controls stay in ordered layers.  This preserves the distinction between an
  angle component adjustment, a multicam role adjustment, and a clip-level
  adjustment instead of prematurely combining their automation curves.

Main callers:
- The central compiler after source resolution.
- The stock-FFmpeg audio graph builder.
- Experimental core tests and the editorial A/B manifest generator.

Why this exists:
The legacy video-owned audio path cannot represent split edits, multiple
streams, role-based multicam audio, or per-component controls. This IR keeps
those decisions exact while sharing source timing with video.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, Mapping, Optional

from .model import AssetResource, SourceDocument, StoryNode, parse_time
from .render_sources import (
    InstanceStreamTiming,
    RenderableAVSource,
    RenderSourceError,
    resolve_compound_source,
    resolve_instance_stream_timing,
    source_bound_tolerance,
    source_range_within_bounds,
    source_window_for_retime,
)

if TYPE_CHECKING:
    from .story_ir import ResourceStory


AudioLayout = Literal["mono", "stereo", "surround"]
FindingDisposition = Literal["preserved", "inactive", "not_implemented_yet"]

_OUTPUT_CHANNELS: Mapping[AudioLayout, tuple[str, ...]] = {
    "mono": ("C",),
    "stereo": ("L", "R"),
    "surround": ("L", "R", "C", "LFE", "Ls", "Rs"),
}
_ENHANCEMENT_KINDS = {
    "adjust-loudness",
    "adjust-noiseReduction",
    "adjust-humReduction",
    "adjust-EQ",
    "adjust-matchEQ",
    "adjust-voiceIsolation",
}


class AudioIRCompileError(ValueError):
    """Base class for invalid or ambiguous audio plans."""


class AudioIRAmbiguityError(AudioIRCompileError):
    """Raised when FCPXML does not identify one source stream/component."""


class AudioIRReferenceError(AudioIRCompileError):
    """Raised for an unknown asset, channel, output, role, or multicam angle."""


class AudioIRValidationError(AudioIRCompileError):
    """Raised when a preserved audio value is malformed or non-finite."""


@dataclass(frozen=True)
class AudioRole:
    """A Final Cut primary role and optional subrole."""

    primary: str
    subrole: Optional[str] = None

    @property
    def qualified(self) -> str:
        return (
            self.primary if self.subrole is None else f"{self.primary}.{self.subrole}"
        )


@dataclass(frozen=True)
class AudioAutomationPoint:
    """One typed scalar control point in source-media time."""

    time: Fraction
    value: float
    interp: str
    curve: str
    aux_value: Optional[str] = None


@dataclass(frozen=True)
class AudioFade:
    kind: Literal["in", "out"]
    duration: Fraction
    curve: str


@dataclass(frozen=True)
class AnimatedAudioScalar:
    """A scalar plus exact-time automation and Final Cut fade handles."""

    initial: float
    unit: Literal["dB", "normalized", "raw"]
    keyframes: tuple[AudioAutomationPoint, ...] = ()
    fades: tuple[AudioFade, ...] = ()


@dataclass(frozen=True)
class AudioPanner:
    mode: Optional[str]
    amount: AnimatedAudioScalar
    parameters: Mapping[str, float | str]


@dataclass(frozen=True)
class AudioMuteRange:
    """A mute in source-media time; omitted bounds retain Final Cut defaults."""

    source_start: Optional[Fraction]
    duration: Optional[Fraction]
    fades: tuple[AudioFade, ...] = ()


@dataclass(frozen=True)
class AudioEnhancement:
    kind: str
    attributes: Mapping[str, str]
    parameters: Mapping[str, str]
    backend_status: Literal["pending_audio_3", "not_implemented_yet"]
    opaque_data: Optional[str] = None


@dataclass(frozen=True)
class AudioControlLayer:
    """Controls applied together at one FCPXML scope, in document order."""

    path: str
    gain: Optional[AnimatedAudioScalar] = None
    panner: Optional[AudioPanner] = None
    mutes: tuple[AudioMuteRange, ...] = ()
    enhancements: tuple[AudioEnhancement, ...] = ()
    role_selector: Optional[AudioRole] = None
    source_start: Optional[Fraction] = None
    source_duration: Optional[Fraction] = None
    enabled: bool = True
    active: bool = True

    @property
    def is_empty(self) -> bool:
        return (
            self.gain is None
            and self.panner is None
            and not self.mutes
            and not self.enhancements
            and self.role_selector is None
            and self.source_start is None
            and self.source_duration is None
            and self.enabled
            and self.active
        )


@dataclass(frozen=True)
class AudioRetimePoint:
    output_time: Fraction
    source_time: Fraction
    interp: Optional[str]


@dataclass(frozen=True)
class RenderAudioItem:
    """One independently scheduled source component.

    ``source_channels`` are one-based channel indices in ``source_stream_id``.
    An empty tuple means the document omitted ``audioChannels`` and did not
    request explicit routing, so AUDIO-2 must resolve the stream's complete
    channel set from the media probe before graph construction.
    ``output_channels is None`` means Final Cut's layout-dependent default is
    intentionally still unspecified; an explicit ``outCh`` is never lost.
    """

    id: str
    path: str
    name: Optional[str]
    absolute_start: Fraction
    duration: Fraction
    source_start: Fraction
    source_duration: Fraction
    asset_id: str
    asset_uid: Optional[str]
    source_stream_id: str
    source_sample_rate: Optional[int]
    source_channels: tuple[int, ...]
    output_channels: Optional[tuple[str, ...]]
    role: Optional[AudioRole]
    enabled: bool
    active: bool
    control_layers: tuple[AudioControlLayer, ...]
    retime: tuple[AudioRetimePoint, ...]
    preserves_pitch: bool
    ancestor_paths: tuple[str, ...] = ()

    @property
    def end(self) -> Fraction:
        return self.absolute_start + self.duration

    @property
    def audible(self) -> bool:
        return self.enabled and self.active


@dataclass(frozen=True)
class AudioSourceInstance:
    """One ordinary clip instance consuming a file or composed audio source."""

    path: str
    source_id: str
    ancestor_paths: tuple[str, ...]
    timing: InstanceStreamTiming
    preserves_pitch: bool
    controls: tuple[AudioControlLayer, ...] = ()


@dataclass(frozen=True)
class AudioIRFinding:
    code: str
    path: str
    disposition: FindingDisposition
    detail: str


@dataclass(frozen=True)
class AudioRenderPlan:
    schema_version: int
    source_sha256: str
    sequence_duration: Fraction
    sample_rate: int
    layout: AudioLayout
    output_channels: tuple[str, ...]
    items: tuple[RenderAudioItem, ...]
    findings: tuple[AudioIRFinding, ...]
    source_instances: tuple[AudioSourceInstance, ...] = ()


@dataclass(frozen=True)
class _Component:
    path: str
    source_stream_id: Optional[str]
    source_channels: Optional[tuple[int, ...]]
    output_channels: Optional[tuple[str, ...]]
    role: Optional[AudioRole]
    source_start: Optional[Fraction]
    duration: Optional[Fraction]
    enabled: bool
    active: bool
    controls: AudioControlLayer


@dataclass(frozen=True)
class _RoleControl:
    selector: AudioRole
    controls: AudioControlLayer
    path: str
    active: bool


@dataclass(frozen=True)
class _AudioScope:
    """One nested audio scope, stored outer-to-inner while traversing."""

    path: str
    controls: AudioControlLayer
    role_controls: tuple[_RoleControl, ...] = ()


class _AudioIRBuilder:
    """Stateful exact-time compiler used by ``compile_audio_ir``.

    Main callers:
    - ``compile_audio_ir`` once per parsed document.

    Why this exists:
    Item IDs, explicit findings, and role-selector match counts are build-time
    state.  Keeping them outside the frozen result makes repeated compilation
    deterministic.
    """

    def __init__(
        self,
        source: SourceDocument,
        *,
        resource_stories: Mapping[str, "ResourceStory"],
        render_sources: Mapping[str, RenderableAVSource],
    ) -> None:
        self.source = source
        self.resource_stories = resource_stories
        self.render_sources = render_sources
        self.items: list[RenderAudioItem] = []
        self.findings: list[AudioIRFinding] = []
        self.source_instances: list[AudioSourceInstance] = []
        self._item_counter = 0
        self._role_match_counts: dict[str, int] = {}

    def build(self) -> AudioRenderPlan:
        layout = self.source.sequence_audio_layout
        if layout not in _OUTPUT_CHANNELS:
            raise AudioIRValidationError(
                f"sequence has invalid audio layout {layout!r}"
            )
        if self.source.sequence_audio_rate <= 0:
            raise AudioIRValidationError("sequence audio sample rate must be positive")
        self._walk_storyline(
            self.source.spine,
            parent_absolute=Fraction(0),
            parent_source=self.source.sequence_tc_start,
            sequential=True,
            inherited_enabled=True,
            scopes=(),
            component_overrides=(),
            audio_window=None,
            ancestor_paths=("sequence",),
        )
        for instance in self.source_instances:
            if not any(
                instance.path in item.ancestor_paths for item in self.items
            ) and not self._instance_is_explicit_silence(instance):
                raise AudioIRReferenceError(
                    f"{instance.path} resolved audio source {instance.source_id!r} "
                    "has no audio stream in its requested range"
                )
        for selector_path, count in self._role_match_counts.items():
            if count == 0 and not any(
                selector_path.startswith(f"{instance.path}/")
                and self._instance_is_explicit_silence(instance)
                for instance in self.source_instances
            ):
                raise AudioIRReferenceError(
                    f"{selector_path} does not match any emitted audio component"
                )
        items = tuple(
            sorted(self.items, key=lambda item: (item.absolute_start, item.id))
        )
        return AudioRenderPlan(
            schema_version=2,
            source_sha256=self.source.source_sha256,
            sequence_duration=self.source.sequence_duration,
            sample_rate=self.source.sequence_audio_rate,
            layout=layout,
            output_channels=_OUTPUT_CHANNELS[layout],
            items=items,
            findings=tuple(self.findings),
            source_instances=tuple(self.source_instances),
        )

    def _instance_is_explicit_silence(
        self,
        instance: AudioSourceInstance,
    ) -> bool:
        """Return whether an empty virtual-source interval is authored as gaps.

        Main callers:
        - ``build`` before treating a source instance with no emitted audio as
          a broken reference.

        Why this exists:
        - A valid multicam angle can begin with an explicit ``gap``. Selecting
          only that interval means intentional silence, not a missing stream.
          The interval must be completely covered by finite direct gaps; an
          unknown or partially covered interval still fails loudly.
        """

        source = next(
            (
                candidate
                for candidate in self.render_sources.values()
                if candidate.id == instance.source_id
            ),
            None,
        )
        if source is None:
            return False
        requested_start = instance.timing.source_start
        requested_end = requested_start + instance.timing.source_duration
        cursor = source.source_start
        gaps: list[tuple[Fraction, Fraction]] = []
        for node in source.audio_story:
            offset = node.offset if node.offset is not None else cursor
            end = offset + node.duration
            if node.kind == "gap" and node.enabled:
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

    def _walk_storyline(
        self,
        nodes: tuple[StoryNode, ...],
        *,
        parent_absolute: Fraction,
        parent_source: Fraction,
        sequential: bool,
        inherited_enabled: bool,
        scopes: tuple[_AudioScope, ...],
        component_overrides: tuple[_Component, ...],
        audio_window: Optional[tuple[Fraction, Fraction]],
        ancestor_paths: tuple[str, ...],
    ) -> None:
        """Schedule siblings using the same exact source-domain rule as FCPXML."""

        cursor = parent_source
        for node in nodes:
            offset = (
                node.offset
                if node.offset is not None
                else (cursor if sequential else parent_source)
            )
            absolute = parent_absolute + offset - parent_source
            enabled = inherited_enabled and node.enabled
            element = _xml_element(node.raw_xml, node.path)
            direct_layer = _control_layer(element, node.path)
            own_role_controls = _role_controls(element, node.path)
            self._register_role_controls(own_role_controls)
            own_scope = _AudioScope(node.path, direct_layer, own_role_controls)
            local_scopes = scopes + (
                () if direct_layer.is_empty and not own_role_controls else (own_scope,)
            )

            if node.kind in {"asset-clip", "audio"}:
                asset = self.source.assets.get(node.ref or "")
                emits_audio = (
                    asset is not None
                    and asset.has_audio
                    and node.kind != "video"
                    and node.src_enable != "video"
                )
                if node.time_map and emits_audio:
                    self._compile_file_source_instance(
                        node,
                        element,
                        absolute=absolute,
                        enabled=enabled,
                        scopes=scopes,
                        direct_layer=direct_layer,
                        role_controls=own_role_controls,
                        component_overrides=component_overrides,
                        audio_window=audio_window,
                        ancestor_paths=ancestor_paths,
                    )
                else:
                    self._compile_asset_node(
                        node,
                        element,
                        absolute=absolute,
                        enabled=enabled,
                        scopes=local_scopes,
                        component_overrides=component_overrides,
                        audio_window=audio_window,
                        ancestor_paths=ancestor_paths,
                    )
                # Connected items are siblings of the media's own audio; the
                # media's gain/pan scope does not implicitly own them.
                if node.children:
                    self._walk_storyline(
                        node.children,
                        parent_absolute=absolute,
                        parent_source=node.start,
                        sequential=False,
                        inherited_enabled=enabled,
                        scopes=scopes,
                        component_overrides=(),
                        audio_window=None,
                        # Connected items are timeline siblings of this file
                        # source.  They must stay inside any enclosing source
                        # module, but must not become members of this source's
                        # completed pad or inherit its post-mix controls.
                        ancestor_paths=ancestor_paths,
                    )
            elif node.kind == "mc-clip":
                resolved = self.render_sources.get(node.path)
                if resolved is None:
                    raise AudioIRReferenceError(
                        f"{node.path} has no resolved multicam A/V source"
                    )
                self._compile_virtual_source(
                    node,
                    element,
                    resolved,
                    absolute=absolute,
                    enabled=enabled,
                    scopes=scopes,
                    direct_layer=direct_layer,
                    role_controls=own_role_controls,
                    audio_window=audio_window,
                    ancestor_paths=ancestor_paths,
                )
                if node.children:
                    self._walk_storyline(
                        node.children,
                        parent_absolute=absolute,
                        parent_source=node.start,
                        sequential=False,
                        inherited_enabled=enabled,
                        scopes=scopes,
                        component_overrides=(),
                        audio_window=None,
                        # A connected role/component is composed beside the
                        # multicam result.  Only the selected angle story is
                        # folded into the multicam source instance below.
                        ancestor_paths=ancestor_paths,
                    )
            elif node.kind in {"clip", "sync-clip", "spine", "gap"}:
                primary = tuple(
                    child
                    for child in node.children
                    if node.kind == "spine" or child.lane == 0
                )
                connected = tuple(
                    child
                    for child in node.children
                    if node.kind != "spine" and child.lane != 0
                )
                own_components = _channel_components(
                    element,
                    node,
                    self.source.sequence_audio_layout,
                )
                primary_components = own_components or component_overrides
                primary_scopes = local_scopes
                connected_scopes = scopes
                primary_window = audio_window
                primary_parent_source = node.start
                if node.kind == "clip" and node.time_map:
                    try:
                        timing = resolve_instance_stream_timing(
                            node,
                            absolute_start=absolute,
                            stream="audio",
                        )
                    except RenderSourceError as error:
                        raise AudioIRValidationError(str(error)) from error
                    virtual = RenderableAVSource(
                        id=f"inline:{node.path}",
                        kind="compound",
                        resource_id=node.path,
                        source_start=timing.source_start,
                        duration=timing.source_duration,
                        format_context=None,
                        has_video=True,
                        has_audio=True,
                        video_story=primary,
                        audio_story=primary,
                    )
                    self._compile_virtual_source(
                        node,
                        element,
                        virtual,
                        absolute=absolute,
                        enabled=enabled,
                        scopes=scopes,
                        direct_layer=direct_layer,
                        role_controls=own_role_controls,
                        audio_window=audio_window,
                        ancestor_paths=ancestor_paths,
                    )
                    primary = ()
                elif node.audio_start is not None or node.audio_duration is not None:
                    range_start = (
                        node.audio_start if node.audio_start is not None else node.start
                    )
                    range_duration = (
                        node.audio_duration
                        if node.audio_duration is not None
                        else node.duration
                    )
                    if range_duration <= 0:
                        raise AudioIRValidationError(
                            f"{node.path} audio duration must be positive"
                        )
                    own_window_start = absolute + range_start - node.start
                    primary_window = _intersect_windows(
                        primary_window,
                        (own_window_start, own_window_start + range_duration),
                    )
                if node.kind == "sync-clip":
                    storyline_roles = _sync_role_controls(
                        element, node.path, "storyline"
                    )
                    connected_source_roles = _sync_role_controls(
                        element, node.path, "connected"
                    )
                    self._register_role_controls(storyline_roles)
                    self._register_role_controls(connected_source_roles)
                    if storyline_roles:
                        primary_scopes = (
                            *primary_scopes,
                            _AudioScope(
                                f"{node.path}/sync-source[@sourceID='storyline']",
                                AudioControlLayer(
                                    path=f"{node.path}/sync-source[@sourceID='storyline']"
                                ),
                                storyline_roles,
                            ),
                        )
                    if connected_source_roles:
                        connected_scopes = (
                            *connected_scopes,
                            _AudioScope(
                                f"{node.path}/sync-source[@sourceID='connected']",
                                AudioControlLayer(
                                    path=f"{node.path}/sync-source[@sourceID='connected']"
                                ),
                                connected_source_roles,
                            ),
                        )
                if primary:
                    self._walk_storyline(
                        primary,
                        parent_absolute=absolute,
                        parent_source=primary_parent_source,
                        sequential=node.kind == "spine",
                        inherited_enabled=enabled,
                        scopes=primary_scopes,
                        component_overrides=primary_components,
                        audio_window=primary_window,
                        ancestor_paths=(*ancestor_paths, node.path),
                    )
                if connected:
                    self._walk_storyline(
                        connected,
                        parent_absolute=absolute,
                        parent_source=node.start,
                        sequential=False,
                        inherited_enabled=enabled,
                        scopes=connected_scopes,
                        component_overrides=(),
                        audio_window=None,
                        # Connected children do not cross the containing
                        # clip's source-instance boundary.  Preserve outer
                        # module membership while excluding this clip only.
                        ancestor_paths=ancestor_paths,
                    )
            elif node.kind == "audition":
                if node.children:
                    self._walk_storyline(
                        node.children[:1],
                        parent_absolute=absolute,
                        parent_source=node.start,
                        sequential=False,
                        inherited_enabled=enabled,
                        scopes=local_scopes,
                        component_overrides=component_overrides,
                        audio_window=audio_window,
                        ancestor_paths=(*ancestor_paths, node.path),
                    )
                for choice in node.children[1:]:
                    self.findings.append(
                        AudioIRFinding(
                            code="audition_audio_inactive",
                            path=choice.path,
                            disposition="inactive",
                            detail="alternative audition audio is preserved but not scheduled",
                        )
                    )
            elif node.kind == "ref-clip":
                resolved = self.render_sources.get(node.path)
                if resolved is None:
                    resource = self.resource_stories.get(node.ref or "")
                    if resource is None:
                        raise AudioIRReferenceError(
                            f"{node.path} references unknown compound source {node.ref!r}"
                        )
                    has_video, has_audio = _resource_stream_capabilities(
                        self.source,
                        resource,
                        self.resource_stories,
                        resource_chain=(resource.resource_id,),
                    )
                    resolved = resolve_compound_source(
                        resource,
                        has_video=has_video,
                        has_audio=has_audio,
                    )
                self._compile_virtual_source(
                    node,
                    element,
                    resolved,
                    absolute=absolute,
                    enabled=enabled,
                    scopes=scopes,
                    direct_layer=direct_layer,
                    role_controls=own_role_controls,
                    audio_window=audio_window,
                    ancestor_paths=ancestor_paths,
                )
                if node.children:
                    self._walk_storyline(
                        node.children,
                        parent_absolute=absolute,
                        parent_source=node.start,
                        sequential=False,
                        inherited_enabled=enabled,
                        scopes=scopes,
                        component_overrides=(),
                        audio_window=None,
                        # A connected item is outside the referenced
                        # composition's completed audio pad and its controls.
                        ancestor_paths=ancestor_paths,
                    )
            elif node.kind not in {"gap", "transition", "title", "caption", "video"}:
                self.findings.append(
                    AudioIRFinding(
                        code="audio_story_kind_deferred",
                        path=node.path,
                        disposition="not_implemented_yet",
                        detail=f"audio scheduling for story kind {node.kind!r} is not implemented yet",
                    )
                )

            if sequential:
                cursor = offset + node.duration

    def _compile_virtual_source(
        self,
        node: StoryNode,
        element: ET.Element,
        source: RenderableAVSource,
        *,
        absolute: Fraction,
        enabled: bool,
        scopes: tuple[_AudioScope, ...],
        direct_layer: AudioControlLayer,
        role_controls: tuple[_RoleControl, ...],
        audio_window: Optional[tuple[Fraction, Fraction]],
        ancestor_paths: tuple[str, ...],
    ) -> None:
        """Schedule one resolved compound/multicam like an ordinary A/V source.

        The source story is composed on its own source clock. The resulting
        pad receives the instance timing and direct controls once; role-based
        selectors stay inside the source so they can address real components.
        """

        if not source.has_audio:
            return
        try:
            timing = resolve_instance_stream_timing(
                node,
                absolute_start=absolute,
                stream="audio",
            )
        except RenderSourceError as error:
            raise AudioIRValidationError(str(error)) from error
        source_frame_duration = (
            source.format_context.frame_duration
            if source.format_context is not None
            else self.source.formats[
                self.source.sequence_format_id
            ].frame_duration
        )
        if source_frame_duration is None:
            raise AudioIRValidationError(
                f"{node.path} source has no frame cadence for stream timing"
            )
        try:
            required_window = source_window_for_retime(
                timing.retime_map,
                frame_duration=source_frame_duration,
            )
        except RenderSourceError as error:
            raise AudioIRValidationError(str(error)) from error
        source_end = source.end
        if source_end is None:
            raise AudioIRReferenceError(
                f"{node.path} {source.kind} source has no finite duration"
            )
        source_tolerance = source_bound_tolerance(
            has_video=False,
            frame_duration=None,
            has_audio=source.has_audio,
            audio_rate=self.source.sequence_audio_rate,
        )
        if not source_range_within_bounds(
            requested_start=required_window.start,
            requested_end=required_window.end,
            source_start=source.source_start,
            source_end=source_end,
            tolerance=source_tolerance,
        ):
            raise AudioIRReferenceError(
                f"{node.path} audio source range [{required_window.start}, "
                f"{required_window.end}) is outside {source.kind} source "
                f"[{source.source_start}, {source_end})"
            )
        timing = replace(
            timing,
            source_start=required_window.start,
            source_duration=required_window.duration,
        )
        if audio_window is not None:
            requested = (timing.absolute_start, timing.absolute_end)
            intersection = _intersect_windows(audio_window, requested)
            if intersection != requested:
                raise AudioIRValidationError(
                    f"{node.path} virtual source is partially clipped by an outer "
                    "audio window; nested stream-map slicing is required"
                )
        self.source_instances.append(
            AudioSourceInstance(
                path=node.path,
                source_id=source.id,
                ancestor_paths=ancestor_paths,
                timing=timing,
                preserves_pitch=node.time_map_preserves_pitch,
                controls=(() if direct_layer.is_empty else (direct_layer,)),
            )
        )
        source_scopes = scopes
        if role_controls:
            source_scopes = (
                *source_scopes,
                _AudioScope(
                    node.path,
                    AudioControlLayer(path=node.path),
                    role_controls,
                ),
            )
        if source.audio_scope is not None:
            scope_element = _xml_element(
                source.audio_scope.raw_xml,
                source.audio_scope.path,
            )
            scope_layer = _control_layer(scope_element, source.audio_scope.path)
            scope_roles = _role_controls(scope_element, source.audio_scope.path)
            self._register_role_controls(scope_roles)
            if not scope_layer.is_empty or scope_roles:
                source_scopes = (
                    *source_scopes,
                    _AudioScope(source.audio_scope.path, scope_layer, scope_roles),
                )
        local_window = (
            timing.absolute_start,
            timing.absolute_start + required_window.duration,
        )
        self._walk_storyline(
            source.audio_story,
            parent_absolute=timing.absolute_start,
            parent_source=required_window.start,
            sequential=True,
            inherited_enabled=enabled,
            scopes=source_scopes,
            component_overrides=(),
            audio_window=local_window,
            ancestor_paths=(*ancestor_paths, node.path),
        )

    def _compile_file_source_instance(
        self,
        node: StoryNode,
        element: ET.Element,
        *,
        absolute: Fraction,
        enabled: bool,
        scopes: tuple[_AudioScope, ...],
        direct_layer: AudioControlLayer,
        role_controls: tuple[_RoleControl, ...],
        component_overrides: tuple[_Component, ...],
        audio_window: Optional[tuple[Fraction, Fraction]],
        ancestor_paths: tuple[str, ...],
    ) -> None:
        """Apply the common source-instance timing boundary to a file asset.

        The raw file interval is scheduled as one descendant pad. Its clip
        timeMap then runs on that completed pad, followed by direct gain,
        panner, fades, and enhancements exactly once—the same order used for
        compound and multicam sources.
        """

        if not node.ref or node.ref not in self.source.assets:
            raise AudioIRReferenceError(
                f"{node.path} references unknown file source {node.ref!r}"
            )
        asset = self.source.assets[node.ref]
        try:
            timing = resolve_instance_stream_timing(
                node,
                absolute_start=absolute,
                stream="audio",
            )
        except RenderSourceError as error:
            raise AudioIRValidationError(str(error)) from error
        source_format = (
            self.source.formats.get(asset.format_id)
            if asset.format_id is not None
            else None
        )
        frame_duration = (
            source_format.frame_duration
            if source_format is not None
            else self.source.formats[
                self.source.sequence_format_id
            ].frame_duration
        )
        if frame_duration is None:
            raise AudioIRValidationError(
                f"{node.path} file source has no frame cadence"
            )
        required_window = source_window_for_retime(
            timing.retime_map,
            frame_duration=frame_duration,
        )
        asset_end = (
            asset.start + asset.duration
            if asset.duration is not None and asset.duration > 0
            else None
        )
        source_tolerance = source_bound_tolerance(
            has_video=asset.has_video,
            frame_duration=frame_duration,
            has_audio=asset.has_audio,
            audio_rate=asset.audio_rate,
        )
        if not source_range_within_bounds(
            requested_start=required_window.start,
            requested_end=required_window.end,
            source_start=asset.start,
            source_end=asset_end if asset_end is not None else required_window.end,
            tolerance=source_tolerance,
        ):
            raise AudioIRReferenceError(
                f"{node.path} audio source range [{required_window.start}, "
                f"{required_window.end}) is outside file source range"
            )
        timing = replace(
            timing,
            source_start=required_window.start,
            source_duration=required_window.duration,
        )
        self.source_instances.append(
            AudioSourceInstance(
                path=node.path,
                source_id=f"file:{asset.id}",
                ancestor_paths=ancestor_paths,
                timing=timing,
                preserves_pitch=node.time_map_preserves_pitch,
                controls=(() if direct_layer.is_empty else (direct_layer,)),
            )
        )
        source_scopes = scopes
        if role_controls:
            source_scopes = (
                *source_scopes,
                _AudioScope(
                    node.path,
                    AudioControlLayer(path=node.path),
                    role_controls,
                ),
            )
        raw_node = replace(
            node,
            time_map=(),
            audio_start=None,
            audio_duration=None,
        )
        self._compile_asset_node(
            raw_node,
            element,
            absolute=timing.absolute_start,
            enabled=enabled,
            scopes=source_scopes,
            component_overrides=component_overrides,
            audio_window=audio_window,
            ancestor_paths=(*ancestor_paths, node.path),
            forced_source_start=required_window.start,
            forced_duration=required_window.duration,
        )

    def _compile_asset_node(
        self,
        node: StoryNode,
        element: ET.Element,
        *,
        absolute: Fraction,
        enabled: bool,
        scopes: tuple[_AudioScope, ...],
        component_overrides: tuple[_Component, ...],
        audio_window: Optional[tuple[Fraction, Fraction]],
        ancestor_paths: tuple[str, ...],
        forced_source_start: Optional[Fraction] = None,
        forced_duration: Optional[Fraction] = None,
    ) -> None:
        if node.kind == "asset-clip" and node.src_enable == "video":
            return
        if not node.ref:
            raise AudioIRReferenceError(f"{node.path} has no asset ref")
        asset = self.source.assets.get(node.ref)
        if asset is None:
            raise AudioIRReferenceError(
                f"{node.path} references unknown audio asset {node.ref!r}"
            )
        if not asset.has_audio:
            if node.kind == "audio" or node.src_enable == "audio":
                raise AudioIRReferenceError(
                    f"{node.path} requests audio from asset {asset.id!r} without audio"
                )
            return
        components = component_overrides or _components(
            element,
            node,
            asset,
            self.source.sequence_audio_layout,
        )
        base_start = (
            forced_source_start
            if forced_source_start is not None
            else (node.audio_start if node.audio_start is not None else node.start)
        )
        base_duration = (
            forced_duration
            if forced_duration is not None
            else (
                node.audio_duration
                if node.audio_duration is not None
                else node.duration
            )
        )
        base_absolute = (
            absolute
            if forced_source_start is not None
            else absolute + base_start - node.start
        )
        if base_duration <= 0:
            raise AudioIRValidationError(f"{node.path} audio duration must be positive")

        for component in components:
            stream_id = _resolve_stream_id(component, asset, node.path)
            source_channels = _resolve_source_channels(component, asset, node.path)
            if not source_channels and asset.audio_channels is None:
                self.findings.append(
                    AudioIRFinding(
                        code="audio_channels_probe_required",
                        path=component.path,
                        disposition="preserved",
                        detail=(
                            "FCPXML omitted explicit srcCh; the complete source "
                            "channel set is resolved from bound media metadata"
                        ),
                    )
                )
            source_start, duration, item_absolute = _component_interval(
                component,
                base_start=base_start,
                base_duration=base_duration,
                base_absolute=base_absolute,
            )
            if audio_window is not None:
                windowed = _clip_to_window(
                    item_absolute,
                    source_start,
                    duration,
                    audio_window,
                )
                if windowed is None:
                    continue
                item_absolute, source_start, duration = windowed
            schedule_limit = self.source.sequence_duration
            for instance in self.source_instances:
                if instance.path in ancestor_paths:
                    schedule_limit = max(
                        schedule_limit,
                        instance.timing.absolute_start
                        + instance.timing.source_duration,
                    )
            clipped = _clip_to_sequence(
                item_absolute,
                source_start,
                duration,
                schedule_limit,
            )
            if clipped is None:
                continue
            item_absolute, source_start, duration = clipped
            source_format = (
                self.source.formats.get(asset.format_id)
                if asset.format_id is not None
                else None
            )
            _validate_asset_range(
                asset,
                source_start,
                duration,
                component.path,
                frame_duration=(
                    source_format.frame_duration if source_format is not None else None
                ),
            )
            matched_controls: list[_RoleControl] = []
            component_layers = (
                () if component.controls.is_empty else (component.controls,)
            )
            ordered_scope_layers: list[AudioControlLayer] = []
            for scope in reversed(scopes):
                for role_control in scope.role_controls:
                    if _role_matches(role_control.selector, component.role):
                        self._role_match_counts[role_control.path] += 1
                        matched_controls.append(role_control)
                        if not role_control.controls.is_empty:
                            ordered_scope_layers.append(role_control.controls)
                if not scope.controls.is_empty:
                    ordered_scope_layers.append(scope.controls)
            control_layers = (*component_layers, *ordered_scope_layers)
            self._append_item(
                path=component.path,
                name=node.name or asset.name,
                absolute_start=item_absolute,
                duration=duration,
                source_start=source_start,
                asset=asset,
                source_stream_id=stream_id,
                source_channels=source_channels,
                output_channels=component.output_channels,
                role=component.role,
                enabled=enabled and component.enabled,
                active=component.active
                and all(control.active for control in matched_controls),
                control_layers=control_layers,
                node=node,
                ancestor_paths=ancestor_paths,
            )

    def _register_role_controls(self, controls: tuple[_RoleControl, ...]) -> None:
        for control in controls:
            if control.path in self._role_match_counts:
                raise AudioIRValidationError(
                    f"duplicate role-control path {control.path!r}"
                )
            self._role_match_counts[control.path] = 0

    def _append_item(
        self,
        *,
        path: str,
        name: Optional[str],
        absolute_start: Fraction,
        duration: Fraction,
        source_start: Fraction,
        asset: AssetResource,
        source_stream_id: str,
        source_channels: tuple[int, ...],
        output_channels: Optional[tuple[str, ...]],
        role: Optional[AudioRole],
        enabled: bool,
        active: bool,
        control_layers: tuple[AudioControlLayer, ...],
        node: StoryNode,
        ancestor_paths: tuple[str, ...],
    ) -> None:
        self._item_counter += 1
        source_origin = asset.start
        retime = tuple(
            AudioRetimePoint(
                point.time,
                point.value - source_origin,
                point.interp,
            )
            for point in node.time_map
        )
        local_source_start = source_start - source_origin
        source_duration = duration
        if retime:
            source_duration = max(
                source_duration,
                max(point.source_time for point in retime) - local_source_start,
            )
        item = RenderAudioItem(
            id=f"audio-{self._item_counter}",
            path=path,
            name=name,
            absolute_start=absolute_start,
            duration=duration,
            source_start=local_source_start,
            source_duration=source_duration,
            asset_id=asset.id,
            asset_uid=asset.uid,
            source_stream_id=source_stream_id,
            source_sample_rate=asset.audio_rate,
            source_channels=source_channels,
            output_channels=output_channels,
            role=role,
            enabled=enabled,
            active=active,
            control_layers=control_layers,
            retime=retime,
            preserves_pitch=node.time_map_preserves_pitch,
            ancestor_paths=ancestor_paths,
        )
        self.items.append(item)
        if not item.audible:
            self.findings.append(
                AudioIRFinding(
                    code="audio_component_silent",
                    path=path,
                    disposition="inactive",
                    detail="disabled clip or inactive component produces silence",
                )
            )
        for layer in control_layers:
            for enhancement in layer.enhancements:
                if enhancement.backend_status == "not_implemented_yet":
                    self.findings.append(
                        AudioIRFinding(
                            code="audio_enhancement_not_implemented",
                            path=layer.path,
                            disposition="not_implemented_yet",
                            detail=f"{enhancement.kind} is preserved but has no portable backend",
                        )
                    )


def _resource_stream_capabilities(
    source: SourceDocument,
    resource: "ResourceStory",
    resources: Mapping[str, "ResourceStory"],
    *,
    resource_chain: tuple[str, ...],
) -> tuple[bool, bool]:
    """Resolve compound capabilities for standalone audio-IR callers."""

    def visit(
        nodes: tuple[StoryNode, ...],
        chain: tuple[str, ...],
    ) -> tuple[bool, bool]:
        has_video = False
        has_audio = False
        for node in nodes:
            if not node.enabled or node.src_enable == "none":
                continue
            if node.kind in {"asset-clip", "video", "audio"} and node.ref:
                asset = source.assets.get(node.ref)
                if asset is not None:
                    has_video = has_video or (
                        asset.has_video
                        and node.kind != "audio"
                        and node.src_enable != "audio"
                    )
                    has_audio = has_audio or (
                        asset.has_audio
                        and node.kind != "video"
                        and node.src_enable != "video"
                    )
            elif node.kind in {"title", "caption"}:
                has_video = True
            elif node.kind == "ref-clip":
                child = resources.get(node.ref or "")
                if child is None:
                    raise AudioIRReferenceError(
                        f"{node.path} references unknown compound source {node.ref!r}"
                    )
                if child.resource_id in chain:
                    readable = " -> ".join((*chain, child.resource_id))
                    raise AudioIRReferenceError(
                        f"recursive render source reference: {readable}"
                    )
                child_video, child_audio = visit(
                    child.story,
                    (*chain, child.resource_id),
                )
                has_video = has_video or child_video
                has_audio = has_audio or child_audio
            if node.children:
                child_video, child_audio = visit(node.children, chain)
                has_video = has_video or child_video
                has_audio = has_audio or child_audio
        return has_video, has_audio

    return visit(resource.story, resource_chain)


def compile_audio_ir(
    source: SourceDocument,
    *,
    resource_stories: Optional[Mapping[str, "ResourceStory"]] = None,
    render_sources: Optional[Mapping[str, RenderableAVSource]] = None,
) -> AudioRenderPlan:
    """Compile the parsed source model into independent audio IR v2.

    Main callers:
    - The future ``AUDIO-2`` FFmpeg audio engine.
    - Experimental source/IR audits.

    This routine performs no media I/O and changes no shared renderer state.
    """

    return _AudioIRBuilder(
        source,
        resource_stories=resource_stories or {},
        render_sources=render_sources or {},
    ).build()


def _xml_element(raw_xml: str, path: str) -> ET.Element:
    try:
        return ET.fromstring(raw_xml)
    except ET.ParseError as exc:
        raise AudioIRValidationError(
            f"{path} has invalid preserved XML: {exc}"
        ) from exc


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _direct_children(element: ET.Element, *kinds: str) -> tuple[ET.Element, ...]:
    accepted = set(kinds)
    return tuple(child for child in element if _tag(child) in accepted)


def _bool_attribute(element: ET.Element, name: str, *, default: bool) -> bool:
    raw = element.get(name)
    if raw is None:
        return default
    if raw not in {"0", "1"}:
        raise AudioIRValidationError(f"<{_tag(element)}> has invalid {name}={raw!r}")
    return raw == "1"


def _time_attribute(element: ET.Element, name: str, path: str) -> Optional[Fraction]:
    try:
        value = parse_time(element.get(name), field_name=f"{path} {name}")
    except ValueError as exc:
        raise AudioIRValidationError(str(exc)) from exc
    return value


def _float_value(
    raw: Optional[str], *, path: str, default: float, suffix: Optional[str] = None
) -> float:
    if raw is None:
        return default
    text = raw.strip()
    if suffix and text.endswith(suffix):
        text = text[: -len(suffix)]
    try:
        value = float(text)
    except ValueError as exc:
        raise AudioIRValidationError(
            f"{path} has invalid numeric value {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise AudioIRValidationError(f"{path} has non-finite numeric value {raw!r}")
    return value


def _role(raw: Optional[str], path: str) -> Optional[AudioRole]:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        raise AudioIRValidationError(f"{path} has an empty role")
    primary, separator, subrole = text.partition(".")
    if not primary.strip() or (separator and not subrole.strip()):
        raise AudioIRValidationError(f"{path} has invalid role {raw!r}")
    return AudioRole(primary.strip(), subrole.strip() if separator else None)


def _role_matches(selector: AudioRole, role: Optional[AudioRole]) -> bool:
    if role is None or selector.primary != role.primary:
        return False
    if selector.subrole is None or selector.subrole == role.subrole:
        return True
    return (
        role.subrole is None
        and selector.subrole == f"{role.primary}-1"
    )


def _channel_numbers(raw: str, path: str) -> tuple[int, ...]:
    values: list[int] = []
    for piece in raw.split(","):
        try:
            value = int(piece.strip())
        except ValueError as exc:
            raise AudioIRValidationError(
                f"{path} has invalid source channel list {raw!r}"
            ) from exc
        if value < 1:
            raise AudioIRValidationError(
                f"{path} source channels are one-based, got {value}"
            )
        values.append(value)
    if not values or len(set(values)) != len(values):
        raise AudioIRValidationError(
            f"{path} has empty or duplicate source channels {raw!r}"
        )
    return tuple(values)


def _output_channel_names(raw: str, layout: AudioLayout, path: str) -> tuple[str, ...]:
    values = tuple(piece.strip() for piece in raw.split(",") if piece.strip())
    allowed = {*_OUTPUT_CHANNELS[layout], "X"}
    if not values or any(value not in allowed for value in values):
        raise AudioIRReferenceError(
            f"{path} has output channels {raw!r} outside {layout} layout {sorted(allowed)}"
        )
    return values


def _fades(parent: ET.Element, path: str) -> tuple[AudioFade, ...]:
    fades: list[AudioFade] = []
    containers = [parent]
    for parameter in _direct_children(parent, "param"):
        if (parameter.get("name") or "").casefold() != "amount":
            continue
        if not _bool_attribute(parameter, "enabled", default=True):
            continue
        containers.append(parameter)
    for container in containers:
        for child in container:
            kind = _tag(child)
            if kind not in {"fadeIn", "fadeOut"}:
                continue
            duration = _time_attribute(child, "duration", f"{path}/{kind}")
            if duration is None or duration < 0:
                raise AudioIRValidationError(
                    f"{path}/{kind} requires a non-negative duration"
                )
            default_curve = "easeIn" if kind == "fadeIn" else "easeOut"
            fades.append(
                AudioFade(
                    "in" if kind == "fadeIn" else "out",
                    duration,
                    child.get("type") or default_curve,
                )
            )
    return tuple(fades)


def _automation(
    parent: ET.Element,
    *,
    parameter_name: str,
    path: str,
    initial: float,
    unit: Literal["dB", "normalized", "raw"],
    suffix: Optional[str] = None,
) -> AnimatedAudioScalar:
    points: list[AudioAutomationPoint] = []
    for parameter in _direct_children(parent, "param"):
        if (parameter.get("name") or "").casefold() != parameter_name.casefold():
            continue
        if not _bool_attribute(parameter, "enabled", default=True):
            continue
        for animation in _direct_children(parameter, "keyframeAnimation"):
            for keyframe in _direct_children(animation, "keyframe"):
                time = _time_attribute(keyframe, "time", f"{path}/keyframe")
                if time is None:
                    raise AudioIRValidationError(f"{path}/keyframe is missing time")
                value = _float_value(
                    keyframe.get("value"),
                    path=f"{path}/keyframe@value",
                    default=initial,
                    suffix=suffix,
                )
                points.append(
                    AudioAutomationPoint(
                        time=time,
                        value=value,
                        interp=keyframe.get("interp") or "linear",
                        curve=keyframe.get("curve") or "smooth",
                        aux_value=keyframe.get("auxValue"),
                    )
                )
    points.sort(key=lambda point: point.time)
    if len({point.time for point in points}) != len(points):
        raise AudioIRValidationError(f"{path} has duplicate automation times")
    return AnimatedAudioScalar(
        initial=initial, unit=unit, keyframes=tuple(points), fades=_fades(parent, path)
    )


def _control_layer(element: ET.Element, path: str) -> AudioControlLayer:
    gain: Optional[AnimatedAudioScalar] = None
    panner: Optional[AudioPanner] = None
    mutes: list[AudioMuteRange] = []
    enhancements: list[AudioEnhancement] = []
    for child in element:
        kind = _tag(child)
        child_path = f"{path}/{kind}"
        if kind == "adjust-volume":
            if gain is not None:
                raise AudioIRAmbiguityError(
                    f"{path} has more than one direct adjust-volume"
                )
            initial = _float_value(
                child.get("amount"),
                path=f"{child_path}@amount",
                default=0.0,
                suffix="dB",
            )
            gain = _automation(
                child,
                parameter_name="amount",
                path=child_path,
                initial=initial,
                unit="dB",
                suffix="dB",
            )
        elif kind == "adjust-panner":
            if panner is not None:
                raise AudioIRAmbiguityError(
                    f"{path} has more than one direct adjust-panner"
                )
            raw_mode = child.get("mode")
            mode = raw_mode
            amount_scale = 1.0
            if raw_mode and raw_mode.strip() in {"1", "1 (Stereo Left/Right)"}:
                # Final Cut's canonical stereo token is
                # ``1 (Stereo Left/Right)`` and its amount is a percentage in
                # [-100, 100]. Normalize it once at the source/IR boundary so
                # execution continues to consume the typed [-1, 1] contract.
                mode = "stereo"
                amount_scale = 0.01
            amount = _float_value(
                child.get("amount"), path=f"{child_path}@amount", default=0.0
            ) * amount_scale
            parameters: dict[str, float | str] = {}
            for name, raw in child.attrib.items():
                if name in {"mode", "amount"}:
                    continue
                try:
                    parameters[name] = _float_value(
                        raw, path=f"{child_path}@{name}", default=0.0
                    )
                except AudioIRValidationError:
                    parameters[name] = raw
            automation = _automation(
                    child,
                    parameter_name="amount",
                    path=child_path,
                    initial=amount / amount_scale if amount_scale != 1.0 else amount,
                    unit="normalized",
                )
            if amount_scale != 1.0:
                automation = AnimatedAudioScalar(
                    initial=automation.initial * amount_scale,
                    unit=automation.unit,
                    keyframes=tuple(
                        AudioAutomationPoint(
                            time=point.time,
                            value=point.value * amount_scale,
                            interp=point.interp,
                            curve=point.curve,
                            aux_value=point.aux_value,
                        )
                        for point in automation.keyframes
                    ),
                    fades=automation.fades,
                )
            panner = AudioPanner(
                mode=mode,
                amount=automation,
                parameters=parameters,
            )
        elif kind == "mute":
            duration = _time_attribute(child, "duration", child_path)
            if duration is not None and duration < 0:
                raise AudioIRValidationError(
                    f"{child_path} duration must not be negative"
                )
            mutes.append(
                AudioMuteRange(
                    source_start=_time_attribute(child, "start", child_path),
                    duration=duration,
                    fades=_fades(child, child_path),
                )
            )
        elif kind in _ENHANCEMENT_KINDS:
            parameters = {
                (param.get("key") or param.get("name") or ""): param.get("value") or ""
                for param in _direct_children(child, "param")
            }
            data_element = next(
                (entry for entry in child if _tag(entry) == "data"), None
            )
            enhancements.append(
                AudioEnhancement(
                    kind=kind,
                    attributes=dict(child.attrib),
                    parameters=parameters,
                    backend_status="not_implemented_yet"
                    if kind == "adjust-matchEQ"
                    else "pending_audio_3",
                    opaque_data=(data_element.text or "")
                    if data_element is not None
                    else None,
                )
            )
        elif kind == "filter-audio":
            enhancements.append(
                AudioEnhancement(
                    kind=kind,
                    attributes=dict(child.attrib),
                    parameters={},
                    backend_status="not_implemented_yet",
                )
            )
    return AudioControlLayer(
        path=path,
        gain=gain,
        panner=panner,
        mutes=tuple(mutes),
        enhancements=tuple(enhancements),
    )


def _role_controls(element: ET.Element, path: str) -> tuple[_RoleControl, ...]:
    controls: list[_RoleControl] = []
    for index, child in enumerate(
        _direct_children(element, "audio-role-source"), start=1
    ):
        child_path = f"{path}/audio-role-source[{index}]"
        selector = _role(child.get("role"), child_path)
        if selector is None:
            raise AudioIRValidationError(f"{child_path} is missing role")
        enabled = _bool_attribute(child, "enabled", default=True)
        active = _bool_attribute(child, "active", default=True)
        parsed_layer = _control_layer(child, child_path)
        source_duration = _time_attribute(child, "duration", child_path)
        if source_duration is not None and source_duration <= 0:
            raise AudioIRValidationError(f"{child_path} duration must be positive")
        layer = AudioControlLayer(
            path=parsed_layer.path,
            gain=parsed_layer.gain,
            panner=parsed_layer.panner,
            mutes=parsed_layer.mutes,
            enhancements=parsed_layer.enhancements,
            role_selector=selector,
            source_start=_time_attribute(child, "start", child_path),
            source_duration=source_duration,
            enabled=enabled,
            active=active,
        )
        controls.append(
            _RoleControl(
                selector=selector,
                controls=layer,
                path=child_path,
                active=enabled and active,
            )
        )
    return tuple(controls)


def _sync_role_controls(
    element: ET.Element,
    path: str,
    source_id: Literal["storyline", "connected"],
) -> tuple[_RoleControl, ...]:
    """Parse role selectors for one synchronized-clip source domain."""

    controls: list[_RoleControl] = []
    for index, source in enumerate(_direct_children(element, "sync-source"), start=1):
        source_path = f"{path}/sync-source[{index}]"
        raw_source_id = source.get("sourceID")
        if raw_source_id not in {"storyline", "connected"}:
            raise AudioIRValidationError(
                f"{source_path} has invalid sourceID {raw_source_id!r}"
            )
        if raw_source_id == source_id:
            controls.extend(_role_controls(source, source_path))
    return tuple(controls)


def _channel_components(
    element: ET.Element,
    node: StoryNode,
    layout: AudioLayout,
) -> tuple[_Component, ...]:
    """Parse explicit channel components owned by a clip or asset item."""

    channel_sources = _direct_children(element, "audio-channel-source")
    components: list[_Component] = []
    for index, child in enumerate(channel_sources, start=1):
        path = f"{node.path}/audio-channel-source[{index}]"
        source_channels = _channel_numbers(child.get("srcCh") or "", path)
        output_channels = None
        if child.get("outCh") is not None:
            output_channels = _output_channel_names(
                child.get("outCh") or "", layout, path
            )
            if len(output_channels) != len(source_channels):
                raise AudioIRValidationError(
                    f"{path} maps {len(source_channels)} source channels to {len(output_channels)} outputs"
                )
        components.append(
            _Component(
                path=path,
                source_stream_id=None,
                source_channels=source_channels,
                output_channels=output_channels,
                role=_role(child.get("role") or node.audio_role or node.role, path),
                source_start=_time_attribute(child, "start", path),
                duration=_time_attribute(child, "duration", path),
                enabled=_bool_attribute(child, "enabled", default=True),
                active=_bool_attribute(child, "active", default=True),
                controls=_control_layer(child, path),
            )
        )
    return tuple(components)


def _components(
    element: ET.Element,
    node: StoryNode,
    asset: AssetResource,
    layout: AudioLayout,
) -> tuple[_Component, ...]:
    channel_components = _channel_components(element, node, layout)
    if channel_components:
        return channel_components

    source_channels = None
    output_channels = None
    if element.get("srcCh") is not None:
        source_channels = _channel_numbers(element.get("srcCh") or "", node.path)
    if element.get("outCh") is not None:
        output_channels = _output_channel_names(
            element.get("outCh") or "", layout, node.path
        )
        if source_channels is None:
            raise AudioIRValidationError(f"{node.path} has outCh without srcCh")
        if len(output_channels) != len(source_channels):
            raise AudioIRValidationError(
                f"{node.path} maps {len(source_channels)} source channels to {len(output_channels)} outputs"
            )
    return (
        _Component(
            path=node.path,
            source_stream_id=element.get("srcID"),
            source_channels=source_channels,
            output_channels=output_channels,
            role=_role(element.get("role") or node.audio_role or node.role, node.path),
            source_start=None,
            duration=None,
            enabled=True,
            active=True,
            controls=AudioControlLayer(path=node.path),
        ),
    )


def _resolve_stream_id(component: _Component, asset: AssetResource, path: str) -> str:
    count = asset.audio_sources
    if count is not None and count <= 0:
        raise AudioIRReferenceError(
            f"{path} requests audio from asset {asset.id!r} with audioSources={count}"
        )
    raw = component.source_stream_id
    if raw is None:
        if count is not None and count > 1:
            raise AudioIRAmbiguityError(
                f"{path} references asset {asset.id!r} with {count} audio streams but has no srcID"
            )
        return "1"
    stream_id = raw.strip()
    if not stream_id:
        raise AudioIRValidationError(f"{path} has an empty srcID")
    if count is not None:
        try:
            numeric_id = int(stream_id)
        except ValueError:
            numeric_id = None
        if numeric_id is not None and not (1 <= numeric_id <= count):
            raise AudioIRReferenceError(
                f"{path} selects srcID {stream_id!r}, but asset {asset.id!r} has {count} audio streams"
            )
    return stream_id


def _resolve_source_channels(
    component: _Component, asset: AssetResource, path: str
) -> tuple[int, ...]:
    count = asset.audio_channels
    if component.source_channels is None:
        if count is not None and count <= 0:
            raise AudioIRReferenceError(
                f"{path} requests audio from asset {asset.id!r} with audioChannels={count}"
            )
        return ()
    if count is None or count <= 0:
        raise AudioIRAmbiguityError(
            f"{path} cannot validate srcCh because asset {asset.id!r} has no positive audioChannels"
        )
    invalid = tuple(channel for channel in component.source_channels if channel > count)
    if invalid:
        raise AudioIRReferenceError(
            f"{path} selects channels {invalid}, but asset {asset.id!r} has {count} channels"
        )
    return component.source_channels


def _component_interval(
    component: _Component,
    *,
    base_start: Fraction,
    base_duration: Fraction,
    base_absolute: Fraction,
) -> tuple[Fraction, Fraction, Fraction]:
    base_end = base_start + base_duration
    component_start = (
        component.source_start if component.source_start is not None else base_start
    )
    component_end = component_start + (
        component.duration if component.duration is not None else base_duration
    )
    source_start = max(base_start, component_start)
    source_end = min(base_end, component_end)
    if source_end <= source_start:
        raise AudioIRReferenceError(
            f"{component.path} component range does not intersect its clip audio range"
        )
    return (
        source_start,
        source_end - source_start,
        base_absolute + source_start - base_start,
    )


def _intersect_windows(
    first: Optional[tuple[Fraction, Fraction]],
    second: tuple[Fraction, Fraction],
) -> tuple[Fraction, Fraction]:
    """Intersect absolute half-open audio windows without using floats."""

    if first is None:
        return second
    return max(first[0], second[0]), min(first[1], second[1])


def _clip_to_window(
    absolute_start: Fraction,
    source_start: Fraction,
    duration: Fraction,
    window: tuple[Fraction, Fraction],
) -> Optional[tuple[Fraction, Fraction, Fraction]]:
    """Clip one linear item to an ancestor's independent audio interval."""

    item_end = absolute_start + duration
    overlap_start = max(absolute_start, window[0])
    overlap_end = min(item_end, window[1])
    if overlap_end <= overlap_start:
        return None
    source_start += overlap_start - absolute_start
    return overlap_start, source_start, overlap_end - overlap_start


def _clip_to_sequence(
    absolute_start: Fraction,
    source_start: Fraction,
    duration: Fraction,
    sequence_duration: Fraction,
) -> Optional[tuple[Fraction, Fraction, Fraction]]:
    if absolute_start < 0:
        trim = -absolute_start
        source_start += trim
        duration -= trim
        absolute_start = Fraction(0)
    if absolute_start + duration > sequence_duration:
        duration = sequence_duration - absolute_start
    if duration <= 0:
        return None
    return absolute_start, source_start, duration


def _validate_asset_range(
    asset: AssetResource,
    source_start: Fraction,
    duration: Fraction,
    path: str,
    *,
    frame_duration: Optional[Fraction],
) -> None:
    source_end = (
        asset.start + asset.duration
        if asset.duration is not None
        else source_start + duration
    )
    tolerance = source_bound_tolerance(
        has_video=asset.has_video,
        frame_duration=frame_duration,
        has_audio=asset.has_audio,
        audio_rate=asset.audio_rate,
    )
    if not source_range_within_bounds(
        requested_start=source_start,
        requested_end=source_start + duration,
        source_start=asset.start,
        source_end=source_end,
        tolerance=tolerance,
    ):
        if source_start < asset.start:
            raise AudioIRReferenceError(
                f"{path} source starts at {source_start}, before asset {asset.id!r} start {asset.start}"
            )
        raise AudioIRReferenceError(
            f"{path} source range ends at {source_start + duration}, after asset {asset.id!r} "
            f"end {source_end}"
        )


__all__ = [
    "AnimatedAudioScalar",
    "AudioAutomationPoint",
    "AudioControlLayer",
    "AudioEnhancement",
    "AudioFade",
    "AudioIRAmbiguityError",
    "AudioIRCompileError",
    "AudioIRFinding",
    "AudioIRReferenceError",
    "AudioIRValidationError",
    "AudioMuteRange",
    "AudioPanner",
    "AudioRenderPlan",
    "AudioRetimePoint",
    "AudioRole",
    "RenderAudioItem",
    "compile_audio_ir",
]
