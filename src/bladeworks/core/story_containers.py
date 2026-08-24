"""Resolve Final Cut compound, synchronized, clip, and audition containers.

Architecture map
================

``SourceDocument.other_resources`` (preserved ``<media><sequence>`` XML)
    -> strict reusable-resource catalog
    -> reference/range/cycle validation
    -> existing hierarchical ``story_ir`` builder
    -> compiler-facing ``StoryContainerPlan``

The catalog is consumed directly by the shared render-source resolver. It does
not rewrite ``ref-clip`` elements into synthetic clips or spines.

Important invariants
--------------------

* A reference is resolved only to a ``media`` resource containing one
  ``sequence``.  Missing or malformed targets are compile errors.
* All timing remains ``Fraction``.  A ref selection must fit completely inside
  the reusable sequence before either video or audio expansion can proceed.
* The ``ref-clip`` scope remains outside its internal composition.  Ref-level
  transform/opacity/filter controls therefore apply after video composition,
  while gain/role/J-L controls apply around internal audio scheduling.
* The first audition child is active.  Alternatives remain in ``story_ir`` as
  inactive nodes and are never scheduled by the audio engine.
* ``live-drawing`` is a known explicit ``not_implemented_yet`` finding.
  Any other parser-preserved unknown story element is rejected.
* Recursion is bounded and resource cycles include their full readable chain.

Main callers:
- The root compiler immediately after ``parse_fcpxml``.
- The shared A/V source resolver before clip-instance compilation.

Why this exists:
The secure parser intentionally preserves non-multicam ``media`` resources as
raw XML.  Keeping this adapter isolated makes compound execution available now
without teaching the parser, video compiler, and audio compiler separate and
potentially inconsistent interpretations of the same resource.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import xml.etree.ElementTree as ET
from typing import Literal, Mapping, Optional

from .errors import FCPXMLParseError
from .model import SequenceFormatContext, SourceDocument, StoryNode, parse_time
from .parser import _parse_story_children
from .story_ir import (
    RenderStoryPlan,
    ResourceStory,
    StoryIRCycleError,
    StoryIRDepthError,
    StoryIRFinding,
    build_render_story,
)


class StoryContainerError(ValueError):
    """Base class for resource/container adapter failures."""


class StoryContainerResourceError(StoryContainerError):
    """Raised when a preserved reusable media resource is malformed."""


class StoryContainerReferenceError(StoryContainerError):
    """Raised when an active ref-clip cannot be resolved exactly."""


@dataclass(frozen=True)
class AudioRoleSelectorHook:
    """One role selector nested below ``sync-source``.

    This is an integration/audit record, not a second audio implementation.
    The existing ``audio_ir`` module remains responsible for applying it.
    """

    path: str
    role: str
    enabled: bool
    active: bool
    source_start: Optional[Fraction]
    source_duration: Optional[Fraction]


@dataclass(frozen=True)
class SyncAudioSourceHook:
    """Role-based audio routing for one synchronized-clip source domain."""

    path: str
    source_id: Literal["storyline", "connected"]
    role_selectors: tuple[AudioRoleSelectorHook, ...]


@dataclass(frozen=True)
class CompoundAudioReferenceHook:
    """Exact source selection that the audio resource expansion must honor."""

    path: str
    resource_id: str
    selection_start: Fraction
    selection_duration: Fraction
    audio_start: Fraction
    audio_duration: Fraction
    use_audio_subroles: bool


@dataclass(frozen=True)
class ContainerAudioHooks:
    """Frozen bridge records consumed by audio integration and audits."""

    synchronized_sources: tuple[SyncAudioSourceHook, ...]
    compound_references: tuple[CompoundAudioReferenceHook, ...]


@dataclass(frozen=True)
class CompoundResourceCatalog:
    """Reusable sequence stories parsed from preserved media resources."""

    stories: Mapping[str, ResourceStory]


@dataclass(frozen=True)
class StoryContainerPlan:
    """Compiler-facing result for all Wave 3 story containers."""

    resources: CompoundResourceCatalog
    story: RenderStoryPlan
    audio_hooks: ContainerAudioHooks


def parse_compound_resource_stories(source: SourceDocument) -> CompoundResourceCatalog:
    """Parse every preserved ``media/sequence`` into one reusable story.

    Main callers:
    - ``build_story_container_plan``.

    Why this exists:
    ``parser._parse_resources`` reserves typed handling for multicam and stores
    compound media as ``OtherResource.raw_xml``.  This routine deliberately
    reuses the parser's one story-node routine, so timeline and resource
    stories cannot disagree about intrinsic adjustments or exact time syntax.
    """

    stories: dict[str, ResourceStory] = {}
    for preserved in source.other_resources:
        if preserved.kind != "media":
            continue
        if not preserved.id:
            raise StoryContainerResourceError("compound media resource is missing id")
        try:
            media = ET.fromstring(preserved.raw_xml)
        except ET.ParseError as exc:
            raise StoryContainerResourceError(
                f"media resource {preserved.id!r} contains malformed preserved XML"
            ) from exc
        if _tag(media) != "media":
            raise StoryContainerResourceError(
                f"resource {preserved.id!r} was recorded as media but contains <{_tag(media)}>"
            )
        sequences = tuple(child for child in media if _tag(child) == "sequence")
        if not sequences:
            # A media object may be a project reference with no inline body.
            # If it becomes active, reference validation below rejects it.
            continue
        if len(sequences) != 1:
            raise StoryContainerResourceError(
                f"media resource {preserved.id!r} must contain exactly one sequence"
            )
        sequence = sequences[0]
        spines = tuple(child for child in sequence if _tag(child) == "spine")
        if len(spines) != 1:
            raise StoryContainerResourceError(
                f"media resource {preserved.id!r} sequence must contain exactly one spine"
            )
        path = f"resources/media[@id='{preserved.id}']/sequence"
        try:
            start = parse_time(
                sequence.get("tcStart"),
                field_name=f"{path} tcStart",
            ) or Fraction(0)
            story = tuple(_parse_story_children(spines[0], f"{path}/spine"))
            duration = parse_time(
                sequence.get("duration"),
                field_name=f"{path} duration",
            )
        except (FCPXMLParseError, ValueError) as exc:
            raise StoryContainerResourceError(str(exc)) from exc
        if duration is None:
            duration = _storyline_extent(story, start)
        if duration <= 0:
            raise StoryContainerResourceError(
                f"media resource {preserved.id!r} sequence duration must be positive"
            )
        if preserved.id in stories:
            raise StoryContainerResourceError(
                f"duplicate compound media resource {preserved.id!r}"
            )
        format_id = sequence.get("format")
        sequence_format = source.formats.get(format_id or "")
        if sequence_format is None:
            raise StoryContainerResourceError(
                f"media resource {preserved.id!r} sequence references unknown format {format_id!r}"
            )
        if (
            sequence_format.frame_duration is None
            or not sequence_format.width
            or not sequence_format.height
        ):
            raise StoryContainerResourceError(
                f"media resource {preserved.id!r} sequence format is incomplete"
            )
        stories[preserved.id] = ResourceStory(
            resource_id=preserved.id,
            path=path,
            start=start,
            duration=duration,
            story=story,
            format_context=SequenceFormatContext.from_resource(sequence_format),
        )
    return CompoundResourceCatalog(stories=stories)


def build_story_container_plan(
    source: SourceDocument,
    *,
    max_depth: int = 32,
) -> StoryContainerPlan:
    """Resolve all supported story containers and preserve explicit findings.

    Unknown story elements fail before ``story_ir`` construction.  The one
    known exception, live drawing/PKDrawing, is retained as an unresolved
    group with a specific compatibility finding rather than a fake visual.

    Main callers:
    - The root compiler's render-IR v2 integration seam.
    """

    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    resources = parse_compound_resource_stories(source)
    live_paths = _validate_known_story_surface(source, resources.stories)
    _validate_references(
        source.spine,
        resources.stories,
        max_depth=max_depth,
        depth=1,
        resource_chain=(),
    )
    story = build_render_story(
        source,
        resource_stories=resources.stories,
        max_depth=max_depth,
        unresolved_policy="report",
    )
    story = replace(
        story,
        findings=tuple(
            _rewrite_live_drawing_finding(finding, live_paths)
            for finding in story.findings
        ),
    )
    return StoryContainerPlan(
        resources=resources,
        story=story,
        audio_hooks=collect_container_audio_hooks(source, resources.stories),
    )


def collect_container_audio_hooks(
    source: SourceDocument,
    resource_stories: Mapping[str, ResourceStory],
) -> ContainerAudioHooks:
    """Extract synchronized-source and compound-reference audio seams.

    Main callers:
    - ``build_story_container_plan`` for compatibility/audit evidence.

    Why this exists:
    These controls live in raw XML rather than typed ``StoryNode`` fields.
    Freezing them here proves their presence before the expanded source view is
    passed to ``audio_ir``, which performs the actual routing and role match.
    """

    synchronized: list[SyncAudioSourceHook] = []
    references: list[CompoundAudioReferenceHook] = []

    def visit(nodes: tuple[StoryNode, ...], resource_chain: tuple[str, ...]) -> None:
        for node in nodes:
            try:
                element = ET.fromstring(node.raw_xml)
            except ET.ParseError as exc:
                raise StoryContainerResourceError(
                    f"{node.path} contains malformed preserved XML"
                ) from exc
            if node.kind == "sync-clip":
                sync_index = 0
                for child in element:
                    if _tag(child) != "sync-source":
                        continue
                    sync_index += 1
                    source_id = child.get("sourceID")
                    hook_path = f"{node.path}/sync-source[{sync_index}]"
                    if source_id not in {"storyline", "connected"}:
                        raise StoryContainerResourceError(
                            f"{hook_path} has invalid sourceID {source_id!r}"
                        )
                    roles: list[AudioRoleSelectorHook] = []
                    role_index = 0
                    for role_source in child:
                        if _tag(role_source) != "audio-role-source":
                            raise StoryContainerResourceError(
                                f"{hook_path} contains unsupported <{_tag(role_source)}>"
                            )
                        role_index += 1
                        role_path = f"{hook_path}/audio-role-source[{role_index}]"
                        role = role_source.get("role")
                        if not role:
                            raise StoryContainerResourceError(f"{role_path} is missing role")
                        roles.append(
                            AudioRoleSelectorHook(
                                path=role_path,
                                role=role,
                                enabled=_bool_attr(role_source, "enabled", default=True, path=role_path),
                                active=_bool_attr(role_source, "active", default=True, path=role_path),
                                source_start=_time_attr(role_source, "start", role_path),
                                source_duration=_positive_optional_duration(role_source, role_path),
                            )
                        )
                    synchronized.append(
                        SyncAudioSourceHook(
                            path=hook_path,
                            source_id=source_id,
                            role_selectors=tuple(roles),
                        )
                    )
            if node.kind == "ref-clip":
                resource = resource_stories.get(node.ref or "")
                if resource is None:
                    raise StoryContainerReferenceError(
                        f"{node.path} references media resource {node.ref!r} without an inline sequence"
                    )
                audio_start = node.audio_start if node.audio_start is not None else node.start
                audio_duration = node.audio_duration if node.audio_duration is not None else node.duration
                references.append(
                    CompoundAudioReferenceHook(
                        path=node.path,
                        resource_id=resource.resource_id,
                        selection_start=node.start,
                        selection_duration=node.duration,
                        audio_start=audio_start,
                        audio_duration=audio_duration,
                        use_audio_subroles=_bool_attr(
                            element,
                            "useAudioSubroles",
                            default=False,
                            path=node.path,
                        ),
                    )
                )
                if resource.resource_id not in resource_chain:
                    visit(resource.story, (*resource_chain, resource.resource_id))
            visit(node.children, resource_chain)

    visit(source.spine, ())
    return ContainerAudioHooks(
        synchronized_sources=tuple(synchronized),
        compound_references=tuple(references),
    )


def _validate_known_story_surface(
    source: SourceDocument,
    resource_stories: Mapping[str, ResourceStory],
) -> frozenset[str]:
    live_paths: set[str] = set()

    def visit(nodes: tuple[StoryNode, ...], visited_resources: frozenset[str]) -> None:
        for node in nodes:
            if node.kind in {"live-drawing", "unknown:live-drawing"}:
                live_paths.add(node.path)
            elif node.kind.startswith("unknown:"):
                raise StoryContainerResourceError(
                    f"{node.path} has unsupported story element {node.kind.removeprefix('unknown:')!r}"
                )
            visit(node.children, visited_resources)
            if node.kind == "ref-clip" and node.ref in resource_stories and node.ref not in visited_resources:
                visit(resource_stories[node.ref].story, visited_resources | {node.ref})

    visit(source.spine, frozenset())
    return frozenset(live_paths)


def _validate_references(
    nodes: tuple[StoryNode, ...],
    resource_stories: Mapping[str, ResourceStory],
    *,
    max_depth: int,
    depth: int,
    resource_chain: tuple[str, ...],
) -> None:
    if not nodes:
        return
    if depth > max_depth:
        raise StoryIRDepthError(
            f"story container validation exceeds max_depth={max_depth}"
        )
    for node in nodes:
        if node.kind == "ref-clip":
            resource = resource_stories.get(node.ref or "")
            if resource is None:
                raise StoryContainerReferenceError(
                    f"{node.path} references media resource {node.ref!r} without an inline sequence"
                )
            if resource.resource_id in resource_chain:
                chain = " -> ".join((*resource_chain, resource.resource_id))
                raise StoryIRCycleError(
                    f"compound resource cycle at {node.path}: {chain}"
                )
            _validate_selection(node, resource)
            _validate_references(
                resource.story,
                resource_stories,
                max_depth=max_depth,
                depth=depth + 1,
                resource_chain=(*resource_chain, resource.resource_id),
            )
        _validate_references(
            node.children,
            resource_stories,
            max_depth=max_depth,
            depth=depth + 1,
            resource_chain=resource_chain,
        )


def _validate_selection(node: StoryNode, resource: ResourceStory) -> None:
    selection_end = node.start + node.duration
    resource_end = resource.start + resource.duration
    if node.start < resource.start or selection_end > resource_end:
        raise StoryContainerReferenceError(
            f"{node.path} source range [{node.start}, {selection_end}) exceeds "
            f"resource {resource.resource_id!r} range [{resource.start}, {resource_end})"
        )
    audio_start = node.audio_start if node.audio_start is not None else node.start
    audio_duration = node.audio_duration if node.audio_duration is not None else node.duration
    if audio_duration <= 0:
        raise StoryContainerReferenceError(
            f"{node.path} audio duration must be positive"
        )
    if audio_start < resource.start or audio_start + audio_duration > resource_end:
        raise StoryContainerReferenceError(
            f"{node.path} audio source range [{audio_start}, {audio_start + audio_duration}) "
            f"exceeds resource {resource.resource_id!r} range "
            f"[{resource.start}, {resource_end})"
        )


def _rewrite_live_drawing_finding(
    finding: StoryIRFinding,
    live_paths: frozenset[str],
) -> StoryIRFinding:
    if finding.path not in live_paths or finding.code != "unsupported_story_kind":
        return finding
    return StoryIRFinding(
        code="live_drawing_not_implemented",
        path=finding.path,
        detail=(
            "Final Cut live drawing/PKDrawing data is preserved, but stock FFmpeg "
            "cannot decode the serialized drawing payload"
        ),
        disposition="not_implemented_yet",
    )


def _storyline_extent(nodes: tuple[StoryNode, ...], start: Fraction) -> Fraction:
    cursor = start
    maximum_end = start
    for node in nodes:
        offset = node.offset if node.offset is not None else cursor
        maximum_end = max(maximum_end, offset + node.duration)
        cursor = offset + node.duration
    return maximum_end - start


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _bool_attr(
    element: ET.Element,
    name: str,
    *,
    default: bool,
    path: str,
) -> bool:
    raw = element.get(name)
    if raw is None:
        return default
    if raw == "1":
        return True
    if raw == "0":
        return False
    raise StoryContainerResourceError(
        f"{path} has invalid {name} value {raw!r}; expected '0' or '1'"
    )


def _time_attr(element: ET.Element, name: str, path: str) -> Optional[Fraction]:
    try:
        return parse_time(element.get(name), field_name=f"{path} {name}")
    except ValueError as exc:
        raise StoryContainerResourceError(str(exc)) from exc


def _positive_optional_duration(element: ET.Element, path: str) -> Optional[Fraction]:
    duration = _time_attr(element, "duration", path)
    if duration is not None and duration <= 0:
        raise StoryContainerResourceError(f"{path} duration must be positive")
    return duration
