"""Secure, hierarchy-preserving parser for the documented FCPXML surface.

Architecture map
================

1. Reject entity-bearing or external DTD input before XML parsing.
2. Parse resources without resolving media paths.
3. Select one project by name or UID across the complete library hierarchy.
4. Preserve story nesting, raw unknown subtrees, and exact rational timing.

Main callers:
- ``compile_fcpxml`` and the ``inspect`` CLI command.

Why this exists:
The existing Swift importer resolves much of the timeline into ``Double`` and
maps into SPLYML. This parser instead preserves the source information needed
to explain portable approximations and omissions.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, NoReturn, Optional

from .errors import FCPXMLParseError
from .model import (
    SCHEMA_VERSION,
    AlphaHandling,
    AssetResource,
    CropAdjustment,
    CropRect,
    EffectResource,
    FadeEnvelope,
    FilterInstance,
    FormatResource,
    Keyframe,
    MediaRepresentation,
    MaskedFilterInstance,
    MaskSource,
    MulticamAngle,
    MulticamResource,
    MulticamSource,
    OtherResource,
    Parameter,
    PreservedAdjustment,
    SourceDocument,
    StoryNode,
    TextRun,
    TextStyle,
    TimeMapPoint,
    TransformAdjustment,
    pair,
    parse_time,
)

_ALPHA_HANDLING_KEY = "com.apple.proapps.studio.alphaHandling"
_ALPHA_HANDLING_VALUES: dict[int, AlphaHandling] = {
    0: "premultiplied",
    1: "straight",
    2: "ignore",
}


def _asset_alpha_handling(element: ET.Element) -> Optional[AlphaHandling]:
    """Parse Final Cut's asset-level alpha interpretation, rejecting ambiguity.

    Main callers:
    - ``_parse_resources`` while constructing an ``AssetResource``.

    Why this exists:
    Alpha association cannot be inferred reliably from a decoded YUVA plane.
    Final Cut stores the user's authoritative choice as metadata, so it must
    become typed input rather than remain buried in ``raw_xml``.
    """

    values: list[AlphaHandling] = []
    for metadata in element:
        if _tag(metadata) != "metadata":
            continue
        for item in metadata:
            if _tag(item) == "md" and item.get("key") == _ALPHA_HANDLING_KEY:
                raw = item.get("value")
                match = None if raw is None else re.fullmatch(r"\s*([012])(?:\s+\([^)]*\))?\s*", raw)
                if match is None:
                    raise FCPXMLParseError(
                        f"asset {element.get('id')!r} has malformed {_ALPHA_HANDLING_KEY} value {raw!r}"
                    )
                values.append(_ALPHA_HANDLING_VALUES[int(match.group(1))])
    if len(set(values)) > 1:
        raise FCPXMLParseError(
            f"asset {element.get('id')!r} has conflicting {_ALPHA_HANDLING_KEY} values {values}"
        )
    return values[0] if values else None


_STORY_ELEMENTS = {
    "asset-clip",
    "video",
    "audio",
    "title",
    "caption",
    "gap",
    "transition",
    "ref-clip",
    "mc-clip",
    "sync-clip",
    "audition",
    "clip",
    "spine",
}

# Range metadata may carry the same ``start``/``duration`` attributes as a
# timeline item, but it does not produce pixels, audio, or anchored story
# children. Keep it in its owning node's ``raw_xml`` without compiling it as an
# unknown story node. Final Cut commonly embeds these markers in multicam angle
# asset clips, where treating them as story content used to create bogus group
# scopes and duplicate audio-omission findings.
_STORY_RANGE_METADATA = {
    "analysis-marker",
    "chapter-marker",
    "keyword",
    "marker",
    "rating",
    "todo-marker",
}

# These children affect rendered pixels, samples, or source-time mapping but
# are not themselves storyline items.  Preserve them independently from the
# small typed MVP surface so unsupported active behavior is always reportable.
_PRESERVED_ADJUSTMENT_ELEMENTS = {
    "object-tracker",
    "conform-rate",
    "adjust-crop",
    "adjust-corners",
    "adjust-conform",
    "adjust-transform",
    "adjust-blend",
    "adjust-stabilization",
    "adjust-rollingShutter",
    "adjust-360-transform",
    "adjust-reorient",
    "adjust-orientation",
    "adjust-cinematic",
    "adjust-colorConform",
    "adjust-stereo-3D",
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
}


_PROJECT_REF_PATTERN = re.compile(
    r"^library\[([1-9][0-9]*)\]/event\[([1-9][0-9]*)\]/project\[([1-9][0-9]*)\]$"
)


@dataclass(frozen=True)
class LibraryProject:
    """One addressable Project in the complete FCPXML hierarchy.

    ``project_ref`` is structural and version-scoped. It deliberately does
    not use names or UIDs because both may be absent or duplicated.
    """

    project_ref: str
    library: ET.Element
    event: ET.Element
    project: ET.Element


def parse_fcpxml(path: Path, *, project: Optional[str] = None) -> SourceDocument:
    """Parse one project from an FCPXML file and fail loudly on ambiguity.

    Main callers:
    - ``compile_fcpxml`` for the CLI and programmatic render path.
    - Tests and research tools that inspect the source-preserving model.

    Why this exists:
    A Final Cut library export can contain many events and projects. When
    ``project`` is supplied, search the complete library/event hierarchy and
    match an exact project name or UID. Without it, preserve the convenient
    single-project behavior but refuse to guess when the library is ambiguous.
    """

    source_path, media_base_dir = _resolve_source_and_base(path)
    data = _read_fcpxml_bytes(source_path)
    root = _parse_xml_root(data)
    version = root.get("version")
    if not version:
        raise FCPXMLParseError("fcpxml/@version is required")
    formats, assets, effects, multicams, other_resources = _parse_resource_tree(root)

    event, project_element = _select_project(root, project)
    sequences = [child for child in project_element if _tag(child) == "sequence"]
    if len(sequences) != 1:
        raise FCPXMLParseError("project must contain exactly one sequence")
    sequence = sequences[0]
    spines = [child for child in sequence if _tag(child) == "spine"]
    if len(spines) != 1:
        raise FCPXMLParseError("sequence must contain exactly one primary spine")

    format_id = sequence.get("format")
    if not format_id or format_id not in formats:
        raise FCPXMLParseError(f"sequence references unknown format {format_id!r}")
    try:
        duration = parse_time(sequence.get("duration"), required=True, field_name="sequence duration")
        tc_start = parse_time(sequence.get("tcStart"), field_name="sequence tcStart") or Fraction(0)
    except ValueError as exc:
        raise FCPXMLParseError(str(exc)) from exc
    assert duration is not None
    if duration < 0:
        raise FCPXMLParseError("sequence duration must be non-negative")
    # 0s is a valid empty Project. Clip durations stay strictly positive.

    spine = tuple(_parse_story_children(spines[0], "spine"))
    if duration == 0 and spine:
        raise FCPXMLParseError("zero-duration sequence must have an empty spine")
    return SourceDocument(
        schema_version=SCHEMA_VERSION,
        source_path=source_path,
        source_sha256=hashlib.sha256(data).hexdigest(),
        fcpxml_version=version,
        project_name=project_element.get("name") or "Untitled Project",
        event_name=event.get("name") if event is not None else None,
        sequence_format_id=format_id,
        sequence_duration=duration,
        sequence_tc_start=tc_start,
        formats=formats,
        assets=assets,
        effects=effects,
        multicams=multicams,
        other_resources=tuple(other_resources),
        spine=spine,
        sequence_audio_layout=_audio_layout(sequence.get("audioLayout")),
        sequence_audio_rate=_audio_rate(sequence.get("audioRate"), default=48_000),
        sequence_render_format=sequence.get("renderFormat"),
        media_base_dir=media_base_dir,
    )


@dataclass(frozen=True)
class ResourceDocument:
    """A securely read FCPXML whose ``<resources>`` are parsed but NO Project is selected.

    - ``source_path`` / ``media_base_dir``: the file actually read and the
      directory bundle-relative media resolves against -- determined exactly
      like ``parse_fcpxml`` does (``.fcpxmld`` bundle -> ``Info.fcpxml``).
    - ``data``: the raw bytes read (callers preserving the ``<?xml?>`` /
      ``<!DOCTYPE>`` prolog on write-back split them themselves).
    - ``root``: the parsed, hardened ``<fcpxml>`` element. It is a plain
      mutable ElementTree; ``assets`` only READ it, so a caller may edit it and
      re-serialize.
    - ``assets``: every ``<asset>`` as an ``AssetResource`` (``has_video``,
      ``duration``, media reps, ...).
    """

    source_path: Path
    media_base_dir: Path
    data: bytes
    root: ET.Element
    assets: dict[str, AssetResource]


def parse_fcpxml_resources(path: Path) -> ResourceDocument:
    """Load a document's ``<resources>`` WITHOUT selecting a Project.

    Main callers:
    - ``core/proxy_media.generate_proxies`` (the ``bladeworks proxy`` command).

    Why this exists: ``parse_fcpxml`` refuses a library holding several
    Projects unless one is named, because a compile needs exactly one
    timeline. A pass that is document-wide over ``<resources>`` (proxy
    generation) has no timeline to pick, so forcing ``--project`` there would
    be a spurious failure. This shares the SAME bundle resolution, XML
    hardening, and resources validation as ``parse_fcpxml`` and simply stops
    before the Project stage.
    """

    source_path, media_base_dir = _resolve_source_and_base(path)
    data = _read_fcpxml_bytes(source_path)
    root = _parse_xml_root(data)
    _, assets, _, _, _ = _parse_resource_tree(root)
    return ResourceDocument(
        source_path=source_path,
        media_base_dir=media_base_dir,
        data=data,
        root=root,
        assets=assets,
    )


def _read_fcpxml_bytes(source_path: Path) -> bytes:
    """Read the resolved FCPXML file, turning an OS failure into a parse error."""

    try:
        return source_path.read_bytes()
    except OSError as exc:
        raise FCPXMLParseError(f"could not read FCPXML {source_path}: {exc}") from exc


def _parse_resource_tree(
    root: ET.Element,
) -> tuple[
    dict[str, FormatResource],
    dict[str, AssetResource],
    dict[str, EffectResource],
    dict[str, MulticamResource],
    list[OtherResource],
]:
    """Validate that ``root`` holds exactly one ``<resources>`` and parse it.

    Shared by ``parse_fcpxml`` (full compile) and ``parse_fcpxml_resources``
    (Project-free resource pass) so both apply the identical structural check.
    """

    resources_elements = [child for child in root if _tag(child) == "resources"]
    if len(resources_elements) != 1:
        raise FCPXMLParseError("document must contain exactly one top-level <resources>")
    return _parse_resources(resources_elements[0])


def _resolve_source_and_base(path: Path) -> tuple[Path, Path]:
    """Resolve the FCPXML file to read and the base dir for relative media.

    Two accepted input forms:

    - A plain ``.fcpxml`` FILE. The document is read directly and
      BUNDLE-RELATIVE media ``src`` values resolve against the file's parent
      directory.
    - A canonical ``.fcpxmld`` BUNDLE, i.e. a DIRECTORY (conventionally with a
      ``.fcpxmld`` suffix) that holds ``Info.fcpxml`` at its root plus its
      media (a ``Media/`` subfolder by FCP convention). The document read is
      ``Info.fcpxml`` and the base dir is the bundle root, so a media ``src``
      like ``Media/clip.mp4`` resolves inside the self-contained bundle.

    A directory with NO ``Info.fcpxml`` at its root is an error named loudly
    (no silent fallback): the caller passed something that is not a valid
    ``.fcpxmld`` bundle.

    Returns a ``(source_path, media_base_dir)`` pair where ``source_path`` is
    the file actually read (used for sha and reporting) and ``media_base_dir``
    is the directory relative media is resolved against.
    """

    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        info = resolved / "Info.fcpxml"
        if not info.is_file():
            raise FCPXMLParseError(
                f"FCPXML bundle {resolved} has no Info.fcpxml at its root; "
                "a canonical .fcpxmld bundle must contain Info.fcpxml"
            )
        return info, resolved
    return resolved, resolved.parent


def _parse_xml_root(data: bytes) -> ET.Element:
    """Security-check raw bytes and return the parsed ``<fcpxml>`` root element.

    Shared by ``parse_fcpxml`` (the full compile path) and ``read_fcpxml_root``
    (the lightweight browse path). Rejects entity/DTD tricks via
    ``_assert_safe_xml`` before handing bytes to ElementTree, so every entry
    point that reads FCPXML gets the same hardening and the same loud errors.
    """

    _assert_safe_xml(data)
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise FCPXMLParseError(f"malformed FCPXML: {exc}") from exc
    if _tag(root) != "fcpxml":
        raise FCPXMLParseError("root element must be <fcpxml>")
    return root


def read_fcpxml_root(path: Path) -> ET.Element:
    """Resolve a file/bundle, securely read it, and return its ``<fcpxml>`` root.

    Main callers:
    - the ``bladeworks projects`` CLI command, which only needs the
      library/event/project tree and not a full project compile.

    Why this exists: browsing the projects in a document is far cheaper than
    parsing one out. This reuses ``_resolve_source_and_base`` (so a
    ``.fcpxmld`` bundle directory resolves its ``Info.fcpxml``) and the shared
    ``_parse_xml_root`` hardening, then stops -- it does NOT parse resources,
    select a project, or build a ``SourceDocument``.
    """

    source_path, _ = _resolve_source_and_base(path)
    return _parse_xml_root(_read_fcpxml_bytes(source_path))


def enumerate_library_projects(root: ET.Element) -> list[LibraryProject]:
    """Return every Project with its deterministic structural address.

    Indexes are one-based and count only matching child element names. This
    makes metadata siblings irrelevant while keeping the reference grammar
    identical in Python and the browser codec.
    """

    result: list[LibraryProject] = []
    libraries = [child for child in root if _tag(child) == "library"]
    for library_index, library in enumerate(libraries, start=1):
        events = [child for child in library if _tag(child) == "event"]
        for event_index, event in enumerate(events, start=1):
            projects = [child for child in event if _tag(child) == "project"]
            for project_index, project in enumerate(projects, start=1):
                result.append(
                    LibraryProject(
                        project_ref=(
                            f"library[{library_index}]/event[{event_index}]/"
                            f"project[{project_index}]"
                        ),
                        library=library,
                        event=event,
                        project=project,
                    )
                )
    return result


def validate_project_ref(project_ref: str) -> tuple[int, int, int]:
    """Parse one public structural Project reference or fail explicitly."""

    match = _PROJECT_REF_PATTERN.fullmatch(project_ref)
    if match is None:
        raise FCPXMLParseError(
            "projectRef must match library[N]/event[N]/project[N] with one-based indexes"
        )
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def list_library_projects(root: ET.Element) -> list[tuple[ET.Element, ET.Element]]:
    """Enumerate every ``(event, project)`` under all top-level libraries.

    Walks ``fcpxml -> library -> event -> project`` in document order and
    returns one ``(event, project)`` pair per project found. This is the single
    source of truth for "which projects does this document hold", shared by two
    callers:

    - ``_select_project`` -- picks one project (by name/UID) from this list.
    - the ``bladeworks projects`` CLI command -- prints this list for browsing.

    Why this exists: the candidate enumeration used to live inline inside
    ``_select_project``. Extracting it keeps the browse command and the
    selection logic from drifting apart (e.g. a future nesting change would be
    reflected in both). The list may be empty; callers decide how to treat an
    empty document (both currently fail loudly with the same message).
    """

    return [(item.event, item.project) for item in enumerate_library_projects(root)]


def _select_project(root: ET.Element, selector: Optional[str]) -> tuple[ET.Element, ET.Element]:
    """Return the event and project selected from all top-level libraries.

    Main callers:
    - ``parse_fcpxml`` after resources have been parsed.

    Project names are the human-facing CLI contract. UIDs are accepted too so
    automation can remain stable when editors reuse a name in multiple events.
    Exact matching keeps selection deterministic and avoids surprising partial
    matches in large production libraries.
    """

    addressed = enumerate_library_projects(root)
    candidates = [(item.event, item.project) for item in addressed]
    if not candidates:
        raise FCPXMLParseError("document does not contain a project inside a library event")

    if selector is None:
        if len(candidates) == 1:
            return candidates[0]
        raise FCPXMLParseError(
            f"document contains {len(candidates)} projects; select one with --project NAME_OR_UID "
            f"(available: {_project_choices(candidates)})"
        )

    if selector is not None and _PROJECT_REF_PATTERN.fullmatch(selector):
        for item in addressed:
            if item.project_ref == selector:
                return item.event, item.project
        raise FCPXMLParseError(f"project reference {selector!r} was not found")

    matches = [
        candidate
        for candidate in candidates
        if selector in {candidate[1].get("name"), candidate[1].get("uid")}
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FCPXMLParseError(
            f"project {selector!r} was not found; available projects: {_project_choices(candidates)}"
        )
    raise FCPXMLParseError(
        f"project selector {selector!r} matched {len(matches)} projects; use a project UID instead"
    )


def _project_choices(candidates: list[tuple[ET.Element, ET.Element]]) -> str:
    """Format a bounded project list for actionable parser errors."""

    limit = 20
    choices = [
        f"{project.get('name') or 'Untitled Project'} [{project.get('uid') or 'no uid'}]"
        for _, project in candidates[:limit]
    ]
    if len(candidates) > limit:
        choices.append(f"... and {len(candidates) - limit} more")
    return ", ".join(choices)


def _assert_safe_xml(data: bytes) -> None:
    if len(data) > 128 * 1024 * 1024:
        raise FCPXMLParseError("FCPXML exceeds the 128 MiB parser limit")
    # Inspect the complete bounded document. Limiting this check to a prefix
    # would let a deliberately padded declaration reach ElementTree.
    source = data.decode("utf-8", errors="replace")
    if re.search(r"<!ENTITY\b", source, flags=re.IGNORECASE):
        raise FCPXMLParseError("XML entity declarations are not allowed")
    for declaration in re.findall(r"<!DOCTYPE[\s\S]*?>", source, flags=re.IGNORECASE):
        if not re.fullmatch(r"<!DOCTYPE\s+fcpxml\s*>", declaration, flags=re.IGNORECASE):
            raise FCPXMLParseError("only the literal <!DOCTYPE fcpxml> declaration is allowed")


def _parse_resources(
    resources: ET.Element,
) -> tuple[
    dict[str, FormatResource],
    dict[str, AssetResource],
    dict[str, EffectResource],
    dict[str, MulticamResource],
    list[OtherResource],
]:
    formats: dict[str, FormatResource] = {}
    assets: dict[str, AssetResource] = {}
    effects: dict[str, EffectResource] = {}
    multicams: dict[str, MulticamResource] = {}
    other: list[OtherResource] = []
    seen_ids: set[str] = set()
    for element in resources:
        kind = _tag(element)
        resource_id = element.get("id")
        if resource_id:
            if resource_id in seen_ids:
                raise FCPXMLParseError(f"duplicate resource id {resource_id!r}")
            seen_ids.add(resource_id)
        if kind == "format":
            if not resource_id:
                raise FCPXMLParseError("format resource is missing id")
            try:
                frame_duration = parse_time(element.get("frameDuration"), field_name=f"format {resource_id} frameDuration")
            except ValueError as exc:
                raise FCPXMLParseError(str(exc)) from exc
            formats[resource_id] = FormatResource(
                id=resource_id,
                name=element.get("name"),
                frame_duration=frame_duration,
                width=_int_attr(element, "width"),
                height=_int_attr(element, "height"),
                color_space=element.get("colorSpace"),
                field_order=element.get("fieldOrder"),
                pixel_aspect_h=_int_attr(element, "paspH"),
                pixel_aspect_v=_int_attr(element, "paspV"),
                projection=element.get("projection"),
                stereoscopic=element.get("stereoscopic"),
                hero_eye=element.get("heroEye"),
            )
        elif kind == "asset":
            if not resource_id:
                raise FCPXMLParseError("asset resource is missing id")
            try:
                start = parse_time(element.get("start"), field_name=f"asset {resource_id} start") or Fraction(0)
                duration = parse_time(element.get("duration"), field_name=f"asset {resource_id} duration")
            except ValueError as exc:
                raise FCPXMLParseError(str(exc)) from exc
            reps = tuple(
                MediaRepresentation(kind=rep.get("kind"), src=rep.get("src"))
                for rep in element
                if _tag(rep) == "media-rep"
            )
            assets[resource_id] = AssetResource(
                id=resource_id,
                name=element.get("name"),
                uid=element.get("uid"),
                start=start,
                duration=duration,
                has_video=_bool_attr(element, "hasVideo", default=False),
                has_audio=_bool_attr(element, "hasAudio", default=False),
                format_id=element.get("format"),
                media_representations=reps,
                raw_xml=ET.tostring(element, encoding="unicode"),
                video_sources=_int_attr(element, "videoSources"),
                audio_sources=_int_attr(element, "audioSources"),
                audio_channels=_int_attr(element, "audioChannels"),
                audio_rate=_audio_rate(element.get("audioRate"), default=None),
                custom_lut_override=element.get("customLUTOverride"),
                color_space_override=element.get("colorSpaceOverride"),
                projection_override=element.get("projectionOverride"),
                stereoscopic_override=element.get("stereoscopicOverride"),
                hero_eye_override=element.get("heroEyeOverride"),
                alpha_handling=_asset_alpha_handling(element),
            )
        elif kind == "effect":
            if not resource_id:
                raise FCPXMLParseError("effect resource is missing id")
            effects[resource_id] = EffectResource(
                id=resource_id,
                name=element.get("name"),
                uid=element.get("uid"),
                src=element.get("src"),
                raw_xml=ET.tostring(element, encoding="unicode"),
            )
        elif kind == "media" and _first(element, "multicam") is not None:
            if not resource_id:
                raise FCPXMLParseError("multicam media resource is missing id")
            multicams[resource_id] = _parse_multicam_resource(element, resource_id)
        else:
            other.append(
                OtherResource(
                    id=resource_id,
                    kind=kind,
                    name=element.get("name"),
                    uid=element.get("uid"),
                    raw_xml=ET.tostring(element, encoding="unicode"),
                )
            )
    return formats, assets, effects, multicams, other


def _parse_multicam_resource(element: ET.Element, resource_id: str) -> MulticamResource:
    """Parse existing angle structure without attempting media synchronization.

    Main callers:
    - ``_parse_resources`` for a ``media`` resource containing ``multicam``.

    Why this exists:
    Final Cut has already synchronized these angle storylines.  The portable
    renderer only needs their exact offsets and source trims to execute an
    explicit timeline angle choice.
    """

    multicam = _first(element, "multicam")
    assert multicam is not None
    format_id = multicam.get("format")
    if not format_id:
        raise FCPXMLParseError(f"multicam resource {resource_id!r} is missing format")
    try:
        tc_start = parse_time(
            multicam.get("tcStart"),
            field_name=f"multicam resource {resource_id} tcStart",
        ) or Fraction(0)
        duration = parse_time(
            multicam.get("duration"),
            field_name=f"multicam resource {resource_id} duration",
        )
    except ValueError as exc:
        raise FCPXMLParseError(str(exc)) from exc

    angles: list[MulticamAngle] = []
    seen_angle_ids: set[str] = set()
    for angle_index, angle in enumerate(
        (child for child in multicam if _tag(child) == "mc-angle"),
        start=1,
    ):
        angle_id = angle.get("angleID")
        if not angle_id:
            raise FCPXMLParseError(
                f"resources/media[@id='{resource_id}']/multicam/mc-angle[{angle_index}] is missing angleID"
            )
        if angle_id in seen_angle_ids:
            raise FCPXMLParseError(f"multicam resource {resource_id!r} has duplicate angleID {angle_id!r}")
        seen_angle_ids.add(angle_id)
        path = f"resources/media[@id='{resource_id}']/multicam/mc-angle[{angle_index}]"
        angles.append(
            MulticamAngle(
                name=angle.get("name"),
                angle_id=angle_id,
                story=tuple(_parse_story_children(angle, path)),
                raw_xml=ET.tostring(angle, encoding="unicode"),
            )
        )
    if duration is None:
        duration = _inferred_multicam_duration(tuple(angles), tc_start)
    return MulticamResource(
        id=resource_id,
        name=element.get("name"),
        uid=element.get("uid"),
        format_id=format_id,
        tc_start=tc_start,
        duration=duration,
        angles=tuple(angles),
        raw_xml=ET.tostring(element, encoding="unicode"),
    )


def _inferred_multicam_duration(
    angles: tuple[MulticamAngle, ...],
    tc_start: Fraction,
) -> Optional[Fraction]:
    """Derive an omitted multicam duration from its synchronized angle stories.

    Main callers:
    - ``_parse_multicam_resource`` for real Final Cut exports that omit
      ``multicam/@duration``.

    Why this exists:
    - Final Cut commonly stores the finite multicam extent only in the child
      angle storylines.  Their offsets already share the multicam source clock,
      so the longest finite angle extent is the resource duration.  An empty
      resource remains unknown instead of acquiring a guessed duration.
    """

    extents = tuple(
        _inferred_storyline_duration(angle.story, tc_start)
        for angle in angles
        if angle.story
    )
    if not extents:
        return None
    duration = max(extents)
    return duration if duration > 0 else None


def _parse_story_children(parent: ET.Element, parent_path: str) -> Iterable[StoryNode]:
    counts: dict[str, int] = {}
    for child in parent:
        kind = _tag(child)
        if kind in _STORY_RANGE_METADATA:
            continue
        if kind not in _STORY_ELEMENTS and child.get("duration") is None:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        yield _parse_story_node(child, f"{parent_path}/{kind}[{counts[kind]}]")


def _parse_story_node(element: ET.Element, path: str) -> StoryNode:
    kind = _tag(element)
    try:
        offset = parse_time(element.get("offset"), field_name=f"{path} offset")
        start = parse_time(element.get("start"), field_name=f"{path} start") or Fraction(0)
        audio_start = parse_time(element.get("audioStart"), field_name=f"{path} audioStart")
        audio_duration = parse_time(element.get("audioDuration"), field_name=f"{path} audioDuration")
        duration = parse_time(
            element.get("duration"),
            required=kind not in {"spine", "audition"},
            field_name=f"{path} duration",
        )
    except ValueError as exc:
        raise FCPXMLParseError(str(exc)) from exc

    transform = _parse_transform(_first(element, "adjust-transform"), path)
    crop = _parse_crop(_first(element, "adjust-crop"))
    blend_element = _first(element, "adjust-blend")
    blend_keyframes = _named_parameter_keyframes(blend_element, "amount")
    volume_element = _first(element, "adjust-volume")
    filters = tuple(
        _parse_filter(child) if _tag(child) in {"filter-video", "filter-audio"} else _parse_masked_filter(child)
        for child in element
        if _tag(child) in {"filter-video", "filter-audio", "filter-video-mask"}
    )
    params = tuple(_parse_params(element))
    styles = _parse_text_styles(element)
    text_runs = _parse_text_runs(element, styles)
    multicam_sources = tuple(_parse_multicam_source(child, path) for child in element if _tag(child) == "mc-source")
    children = tuple(_parse_story_children(element, path))
    if duration is None:
        if kind == "audition":
            active_choice = next((child for child in children if child.enabled), None)
            if active_choice is None:
                raise FCPXMLParseError(
                    f"{path} audition requires at least one enabled choice"
                )
            duration = active_choice.duration
        else:
            duration = _inferred_storyline_duration(children, start)
    if duration <= 0:
        raise FCPXMLParseError(f"{path} duration must be positive")

    conform = _first(element, "adjust-conform")
    time_map_element = _first(element, "timeMap")
    return StoryNode(
        kind=kind if kind in _STORY_ELEMENTS else f"unknown:{kind}",
        path=path,
        name=element.get("name"),
        ref=element.get("ref"),
        lane=_int_attr(element, "lane") or 0,
        offset=offset,
        start=start,
        duration=duration,
        enabled=_bool_attr(element, "enabled", default=True),
        src_enable=element.get("srcEnable"),
        audio_start=audio_start,
        audio_duration=audio_duration,
        role=element.get("role"),
        video_role=element.get("videoRole"),
        audio_role=element.get("audioRole"),
        conform_type=(conform.get("type") if conform is not None else None) or "fit",
        transform=transform,
        crop=crop,
        blend_opacity=_float_attr(blend_element, "amount", 1.0),
        blend_keyframes=blend_keyframes,
        blend_mode=blend_element.get("mode") if blend_element is not None else None,
        opacity_fade=_parse_fade(blend_element),
        volume_db=_parse_db(volume_element.get("amount")) if volume_element is not None else None,
        audio_fade=_parse_fade(volume_element),
        time_map=_parse_time_map(time_map_element, path),
        time_map_preserves_pitch=_bool_attr(time_map_element, "preservesPitch", default=True) if time_map_element is not None else True,
        time_map_frame_sampling=time_map_element.get("frameSampling") if time_map_element is not None else None,
        filters=filters,
        params=params,
        text_runs=text_runs,
        text_styles=styles,
        multicam_sources=multicam_sources,
        children=children,
        raw_xml=ET.tostring(element, encoding="unicode"),
        preserved_adjustments=_parse_preserved_adjustments(element),
    )


def _parse_preserved_adjustments(element: ET.Element) -> tuple[PreservedAdjustment, ...]:
    """Preserve every direct render-affecting non-story child.

    Main callers:
    - ``_parse_story_node`` after its typed MVP fields are parsed.

    Why this exists:
    ElementTree otherwise discards unmodeled intrinsic adjustments at the
    source-model boundary.  Keeping the raw bounded subtree lets the compiler
    prove that each active construct was executed or explicitly reported.
    """

    adjustments: list[PreservedAdjustment] = []
    for child in element:
        kind = _tag(child)
        if kind not in _PRESERVED_ADJUSTMENT_ELEMENTS and not kind.startswith("adjust-"):
            continue
        adjustments.append(
            PreservedAdjustment(
                kind=kind,
                enabled=_bool_attr(child, "enabled", default=True),
                attributes=dict(child.attrib),
                params=tuple(_parse_params(child)),
                raw_xml=ET.tostring(child, encoding="unicode"),
            )
        )
    return tuple(adjustments)


def _named_parameter_keyframes(
    element: Optional[ET.Element],
    name: str,
) -> tuple[Keyframe, ...]:
    """Return one intrinsic parameter's genuine nested animation track.

    Main callers:
    - ``_parse_story_node`` for intrinsic scalar controls such as opacity.

    Why this exists:
    Intrinsic attributes hold their static value, while Final Cut serializes
    arbitrary automation in a same-named ``param/keyframeAnimation`` child.
    Keeping that lookup centralized prevents another synthetic direct-
    keyframe parser from drifting away from the FCPXML 1.14 shape.
    """

    if element is None:
        return ()
    matches = [
        _parse_param(child)
        for child in element
        if _tag(child) == "param" and (child.get("name") or "").casefold() == name.casefold()
    ]
    if len(matches) > 1:
        raise FCPXMLParseError(
            f"{_tag(element)} has duplicate {name!r} parameter tracks"
        )
    return matches[0].keyframes if matches else ()


def _parse_multicam_source(element: ET.Element, path: str) -> MulticamSource:
    angle_id = element.get("angleID")
    if not angle_id:
        raise FCPXMLParseError(f"{path}/mc-source is missing angleID")
    src_enable = element.get("srcEnable") or "all"
    if src_enable not in {"all", "audio", "video", "none"}:
        raise FCPXMLParseError(f"{path}/mc-source has invalid srcEnable {src_enable!r}")
    return MulticamSource(
        angle_id=angle_id,
        src_enable=src_enable,
        child_kinds=tuple(_tag(child) for child in element),
        raw_xml=ET.tostring(element, encoding="unicode"),
    )


def _inferred_storyline_duration(children: tuple[StoryNode, ...], start: Fraction) -> Fraction:
    """Infer a secondary spine extent from its sequential children.

    FCPXML secondary ``spine`` elements normally omit ``duration``. Their
    extent is the maximum child end in the spine's own source-time domain.

    Main callers:
    - ``_parse_story_node`` for nested secondary storylines.
    """

    cursor = start
    maximum_end = start
    for child in children:
        offset = child.offset if child.offset is not None else cursor
        maximum_end = max(maximum_end, offset + child.duration)
        cursor = offset + child.duration
    return maximum_end - start


# Final Cut's Transform inspector exposes only position, rotation, non-uniform
# scale, and anchor -- there is no shear/skew degree of freedom.  These names are
# the ones a hand-authored or third-party FCPXML might use to request a shear warp
# under ``adjust-transform``; we recognize them purely to emit a precise capability
# error.  Genuine shear-like distortion is expressed through the four-corner Distort
# effect (``adjust-corners``), which the renderer already supports as an 8-DOF
# projective corner-pin.
_SHEAR_SKEW_TRANSFORM_PARAMS = frozenset(
    {"shear", "skew", "shearx", "sheary", "skewx", "skewy"}
)


def _reject_unsupported_transform_param(name: str, path: str) -> NoReturn:
    """Fail loudly on an adjust-transform param outside the four FCP channels.

    What it does:
    Always raises ``FCPXMLParseError`` naming the offending parameter.  Shear/
    skew names get a capability-specific message that points at the supported
    Distort path; every other unrecognized name gets a generic explicit reject.

    Main callers:
    - ``_parse_transform`` when a direct ``param`` child of ``adjust-transform``
      is not one of position/scale/rotation/anchor.

    Why this exists:
    The transform parser previously ``continue``-d past unrecognized params,
    silently discarding shear/skew requests.  A silent drop renders the wrong
    picture (an identity transform) with no signal, which violates the repo's
    no-silent-failure rule; surfacing it keeps unsupported geometry from passing
    as success.
    """
    if name in _SHEAR_SKEW_TRANSFORM_PARAMS:
        raise FCPXMLParseError(
            f"{path} adjust-transform: shear/skew parameter {name!r} is not "
            "supported. Final Cut's Transform has no shear control; express "
            "shear-like distortion through the four-corner Distort effect "
            "(adjust-corners), which the renderer supports."
        )
    raise FCPXMLParseError(
        f"{path} adjust-transform: unsupported parameter {name!r} "
        "(adjust-transform accepts only position, scale, rotation, and anchor)"
    )


def _parse_transform(element: Optional[ET.Element], path: str) -> Optional[TransformAdjustment]:
    if element is None:
        return None
    position_frames: list[Keyframe] = []
    scale_frames: list[Keyframe] = []
    rotation_frames: list[Keyframe] = []
    anchor_frames: list[Keyframe] = []
    # Direct ``param`` children of ``adjust-transform`` may only publish the four
    # Final Cut Transform channels below (each mapped to its keyframe accumulator).
    # A dict lookup keeps this loop flat instead of an if/else ladder and turns the
    # "unknown param" case into one explicit fail-closed call.
    channel_frames = {
        "position": position_frames,
        "scale": scale_frames,
        "rotation": rotation_frames,
        "anchor": anchor_frames,
    }
    for param in element:
        if _tag(param) != "param":
            continue
        target = (param.get("name") or "").lower()
        output = channel_frames.get(target)
        if output is None:
            # No silent skip: shear/skew (or any other non-Transform param) must
            # surface loudly instead of vanishing into an identity render.
            _reject_unsupported_transform_param(target, path)
        # FCPXML 1.14 requires keyframes to be nested inside one
        # keyframeAnimation element.  Earlier synthetic fixtures put keyframe
        # elements directly under param, which made the renderer appear to
        # support transform animation while genuine Final Cut exports parsed
        # as static.  Reuse the ordinary parameter parser so every animated
        # published value follows the same documented shape.
        output.extend(_parse_param(param).keyframes)
    try:
        return TransformAdjustment(
            position=pair(element.get("position"), (0.0, 0.0)),
            scale=pair(element.get("scale"), (1.0, 1.0)),
            rotation=_float_attr(element, "rotation", 0.0),
            enabled=_bool_attr(element, "enabled", default=True),
            anchor=pair(element.get("anchor"), (0.0, 0.0)),
            tracking_ref=element.get("tracking"),
            position_keyframes=tuple(position_frames),
            scale_keyframes=tuple(scale_frames),
            rotation_keyframes=tuple(rotation_frames),
            anchor_keyframes=tuple(anchor_frames),
        )
    except ValueError as exc:
        raise FCPXMLParseError(f"{path} adjust-transform: {exc}") from exc


def _parse_crop(element: Optional[ET.Element]) -> Optional[CropAdjustment]:
    if element is None:
        return None
    rects: list[CropRect] = []
    for child in element:
        if _tag(child) in {"crop-rect", "trim-rect", "pan-rect"}:
            rects.append(
                CropRect(
                    left=_float_attr(child, "left", 0.0),
                    top=_float_attr(child, "top", 0.0),
                    right=_float_attr(child, "right", 0.0),
                    bottom=_float_attr(child, "bottom", 0.0),
                    kind=_tag(child),
                )
            )
    adjustment = CropAdjustment(
        mode=element.get("mode") or "trim",
        enabled=_bool_attr(element, "enabled", default=True),
        rects=tuple(rects),
    )
    # Apple's FCPXML DTD deliberately makes every adjustment rectangle
    # optional. Final Cut and authored validation exports may therefore keep
    # an enabled ``adjust-crop`` shell while the selected mode has no matching
    # rectangle. That shell has no pixel effect; treating it as an active crop
    # makes the geometry validator reject valid FCPXML. Preserve the authored
    # mode and any inactive-mode rectangles, but normalize the zero-active-
    # rectangle case to the renderer's disabled identity contract. A partial
    # Pan pair still remains enabled and fails strict geometry validation.
    normalized_mode = adjustment.mode.strip().lower()
    if (
        adjustment.enabled
        and normalized_mode in {"crop", "trim", "pan"}
        and not adjustment.active_rects
    ):
        active_kind = f"{normalized_mode}-rect"
        has_selected_mode_rect = any(
            rect.kind == active_kind for rect in adjustment.rects
        )
        if not has_selected_mode_rect:
            return CropAdjustment(
                mode=adjustment.mode,
                enabled=False,
                rects=adjustment.rects,
            )
    return adjustment


def _parse_fade(element: Optional[ET.Element]) -> Optional[FadeEnvelope]:
    if element is None:
        return None
    roots = [element, *[child for child in element if _tag(child) == "param"]]
    fade_in = next((_first(root, "fadeIn") for root in roots if _first(root, "fadeIn") is not None), None)
    fade_out = next((_first(root, "fadeOut") for root in roots if _first(root, "fadeOut") is not None), None)
    if fade_in is None and fade_out is None:
        return None
    try:
        return FadeEnvelope(
            fade_in=parse_time(fade_in.get("duration"), field_name="fadeIn duration") if fade_in is not None else None,
            fade_in_type=fade_in.get("type") if fade_in is not None else None,
            fade_out=parse_time(fade_out.get("duration"), field_name="fadeOut duration") if fade_out is not None else None,
            fade_out_type=fade_out.get("type") if fade_out is not None else None,
        )
    except ValueError as exc:
        raise FCPXMLParseError(str(exc)) from exc


def _parse_time_map(element: Optional[ET.Element], path: str) -> tuple[TimeMapPoint, ...]:
    if element is None:
        return ()
    points: list[TimeMapPoint] = []
    for child in element:
        if _tag(child) != "timept":
            continue
        try:
            time = parse_time(child.get("time"), required=True, field_name=f"{path} timeMap time")
            value = parse_time(child.get("value"), required=True, field_name=f"{path} timeMap value")
        except ValueError as exc:
            raise FCPXMLParseError(str(exc)) from exc
        assert time is not None and value is not None
        # FCPXML 1.14 declares ``smooth2`` as the DTD default.  Preserve that
        # semantic value explicitly so the linear-only portable retime kernel
        # fails closed instead of silently treating an omitted attribute as
        # linear interpolation.
        points.append(
            TimeMapPoint(
                time=time,
                value=value,
                interp=child.get("interp") or "smooth2",
            )
        )
    if len(points) < 2:
        raise FCPXMLParseError(
            f"{path} timeMap requires at least two timept elements"
        )
    return tuple(points)


def _parse_filter(element: ET.Element) -> FilterInstance:
    params = tuple(_parse_params(element))
    data = {
        child.get("key") or "": "".join(child.itertext()).strip()
        for child in element
        if _tag(child) == "data"
    }
    return FilterInstance(
        kind=_tag(element),
        ref=element.get("ref"),
        name=element.get("name"),
        enabled=_bool_attr(element, "enabled", default=True),
        params=params,
        data=data,
        raw_xml=ET.tostring(element, encoding="unicode"),
    )


def _parse_masked_filter(element: ET.Element) -> MaskedFilterInstance:
    """Preserve the standard FCPXML masked-filter container in document order.

    Main callers:
    - ``_parse_story_node`` for direct video-filter children.

    Why this exists:
    A masked filter is not equivalent to an ordinary filter plus a clip crop.
    The referenced effect applies inside the combined mask, and an optional
    second filter applies outside it.
    """

    masks: list[MaskSource] = []
    filters: list[FilterInstance] = []
    for child in element:
        kind = _tag(child)
        if kind in {"mask-shape", "mask-isolation"}:
            data_element = _first(child, "data")
            masks.append(
                MaskSource(
                    kind=kind,
                    name=child.get("name"),
                    enabled=_bool_attr(child, "enabled", default=True),
                    blend_mode=child.get("blendMode") or ("multiply" if kind == "mask-isolation" else "add"),
                    mask_type=child.get("type"),
                    tracking=child.get("tracking"),
                    params=tuple(_parse_params(child)),
                    data="".join(data_element.itertext()).strip() if data_element is not None else None,
                    raw_xml=ET.tostring(child, encoding="unicode"),
                )
            )
        elif kind == "filter-video":
            filters.append(_parse_filter(child))
    return MaskedFilterInstance(
        kind="filter-video-mask",
        enabled=_bool_attr(element, "enabled", default=True),
        inverted=_bool_attr(element, "inverted", default=False),
        masks=tuple(masks),
        filters=tuple(filters),
        raw_xml=ET.tostring(element, encoding="unicode"),
    )


def _parse_param(element: ET.Element) -> Parameter:
    frames: list[Keyframe] = []
    if any(_tag(child) == "keyframe" for child in element):
        raise FCPXMLParseError(
            "param keyframes must be nested inside keyframeAnimation in FCPXML 1.14"
        )
    animation = _first(element, "keyframeAnimation")
    if animation is not None:
        for keyframe in animation:
            if _tag(keyframe) != "keyframe":
                continue
            try:
                time = parse_time(keyframe.get("time"), required=True, field_name="parameter keyframe time")
            except ValueError as exc:
                raise FCPXMLParseError(str(exc)) from exc
            value = keyframe.get("value")
            if time is None or value is None:
                raise FCPXMLParseError("parameter keyframe requires time and value")
            frames.append(
                Keyframe(
                    time=time,
                    value=value,
                    interp=keyframe.get("interp"),
                    curve=keyframe.get("curve"),
                    aux_value=keyframe.get("auxValue"),
                )
            )
    return Parameter(
        name=element.get("name"),
        key=element.get("key"),
        value=element.get("value"),
        keyframes=tuple(frames),
    )


def _parse_params(element: ET.Element) -> Iterable[Parameter]:
    """Flatten nested published parameters while preserving document order."""

    for child in element:
        if _tag(child) != "param":
            continue
        yield _parse_param(child)
        yield from _parse_params(child)


def _parse_text_styles(element: ET.Element) -> dict[str, TextStyle]:
    styles: dict[str, TextStyle] = {}
    for definition in element:
        if _tag(definition) != "text-style-def":
            continue
        style_id = definition.get("id")
        style_element = _first(definition, "text-style")
        if style_id and style_element is not None:
            styles[style_id] = _text_style(style_element, style_id)
    return styles


def _parse_text_runs(element: ET.Element, styles: dict[str, TextStyle]) -> tuple[TextRun, ...]:
    text_elements = [child for child in element if _tag(child) == "text"]
    runs: list[TextRun] = []
    for text_element in text_elements:
        styled_children = [child for child in text_element if _tag(child) == "text-style"]
        if not styled_children:
            text = "".join(text_element.itertext())
            if text:
                runs.append(TextRun(text=text, style_ref=None, inline_style=None))
            continue
        if text_element.text and text_element.text.strip():
            runs.append(TextRun(text=text_element.text, style_ref=None, inline_style=None))
        for child in styled_children:
            text = "".join(child.itertext())
            inline = _text_style(child, None) if any(key in child.attrib for key in ("font", "fontSize", "fontColor")) else None
            runs.append(TextRun(text=text, style_ref=child.get("ref"), inline_style=inline))
            if child.tail and child.tail.strip():
                runs.append(TextRun(text=child.tail, style_ref=None, inline_style=None))
    return tuple(run for run in runs if run.text)


def _text_style(element: ET.Element, style_id: Optional[str]) -> TextStyle:
    font_face = element.get("fontFace")
    normalized_face = (font_face or "").lower()
    return TextStyle(
        id=style_id,
        font=element.get("font"),
        font_face=font_face,
        font_size=_optional_float(element.get("fontSize")),
        font_color=_validated_color(element, "fontColor"),
        alignment=element.get("alignment"),
        stroke_color=_validated_color(element, "strokeColor"),
        stroke_width=_optional_float(element.get("strokeWidth")),
        tracking=_optional_float(element.get("tracking")),
        bold=element.get("bold") == "1" or "bold" in normalized_face,
        italic=element.get("italic") == "1" or "italic" in normalized_face or "oblique" in normalized_face,
    )


def _validated_color(element: ET.Element, name: str) -> Optional[str]:
    raw = element.get(name)
    if raw is None:
        return None
    try:
        components = [float(piece) for piece in raw.replace(",", " ").split()]
    except ValueError as exc:
        raise FCPXMLParseError(f"text-style/@{name} must contain numeric color components") from exc
    if len(components) not in {3, 4}:
        raise FCPXMLParseError(f"text-style/@{name} must contain three or four color components")
    return raw


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _first(element: ET.Element, name: str) -> Optional[ET.Element]:
    return next((child for child in element if _tag(child) == name), None)


def _bool_attr(element: ET.Element, name: str, *, default: bool) -> bool:
    raw = element.get(name)
    if raw is None:
        return default
    return raw not in {"0", "false", "False"}


def _int_attr(element: ET.Element, name: str) -> Optional[int]:
    raw = element.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise FCPXMLParseError(f"{_tag(element)}/@{name} must be an integer") from exc


def _audio_rate(raw: Optional[str], *, default: Optional[int]) -> Optional[int]:
    """Parse documented sequence-kHz or asset-Hz syntax into integer hertz.

    FCPXML 1.14 declares ``sequence/@audioRate`` as values such as ``48k``,
    while ``asset/@audioRate`` is ordinary CDATA and Final Cut exports values
    such as ``48000``.  Both cross this one typed boundary.
    """

    if raw is None:
        return default
    text = raw.strip().casefold()
    try:
        rate = float(text[:-1]) * 1_000 if text.endswith("k") else int(text)
    except ValueError as exc:
        raise FCPXMLParseError(f"invalid FCPXML audio rate {raw!r}") from exc
    if isinstance(rate, float) and not rate.is_integer():
        raise FCPXMLParseError(f"invalid FCPXML audio rate {raw!r}")
    if rate <= 0:
        raise FCPXMLParseError(f"invalid FCPXML audio rate {raw!r}")
    return int(rate)


def _audio_layout(raw: Optional[str]) -> str:
    layout = raw or "stereo"
    if layout not in {"mono", "stereo", "surround"}:
        raise FCPXMLParseError(f"unsupported FCPXML sequence audio layout {layout!r}")
    return layout


def _float_attr(element: Optional[ET.Element], name: str, default: float) -> float:
    if element is None or element.get(name) is None:
        return default
    try:
        return float(element.get(name, str(default)))
    except ValueError as exc:
        raise FCPXMLParseError(f"{_tag(element)}/@{name} must be numeric") from exc


def _optional_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise FCPXMLParseError(f"expected numeric value, got {raw!r}") from exc


def _parse_db(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    text = raw.strip()
    if text.lower().endswith("db"):
        text = text[:-2]
    return _optional_float(text)
