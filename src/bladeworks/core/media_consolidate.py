"""Consolidate a ``.fcpxmld`` bundle's referenced media into its ``Media/`` folder.

Architecture map
----------------
This module implements the ``bladeworks ... --symlink-media`` behavior: given a
``.fcpxmld`` bundle (a directory holding ``Info.fcpxml`` plus a ``Media/``
subfolder by FCP convention), it makes the bundle *self-contained by reference*.

For every ``<media-rep src=...>`` inside every ``<asset>`` that today points at
an EXTERNAL file elsewhere on disk, it:

1. symlinks that file into ``<bundle>/Media/<basename>`` (the on-disk link
   keeps the RAW filename), and
2. rewrites the ``src`` in ``Info.fcpxml`` to the bundle-relative
   ``Media/<percent-encoded basename>``.

The ``src`` is a URL locator, not a filesystem path: ``core/bindings.py``
urlparses + unquotes it when resolving, so a name like ``take #1.mov`` must be
written as ``Media/take%20%231.mov`` (see ``encode_src_segment``) or the
fragment/query delimiters would truncate it and the asset would go offline.

The result is a bundle whose ``Info.fcpxml`` resolves entirely against its own
``Media/`` folder, while the actual bytes still live at their originals (the
links are absolute, so they survive moving the bundle as long as the originals
stay put).

Three-stage design: PLAN -> VALIDATE -> APPLY
---------------------------------------------
The work is split so the on-disk mutation only happens once the whole operation
is known to be safe:

- ``_plan_consolidation`` walks the parsed document and classifies every
  media-rep: external-and-resolvable (a link to make), already-inside-Media
  (nothing to do), or unresolved/offline (a warning, never a silent drop).
- validation rejects the *one* unrecoverable case up front -- two DISTINCT
  source files that would collide on the same ``Media/<name>`` link -- so we
  never leave a bundle half-consolidated in an ambiguous state. "Same name"
  is judged with the DESTINATION filesystem's semantics: on a case-insensitive
  volume (the macOS default) ``Clip.mov`` and ``clip.mov`` are one link slot,
  so they collide too (``_filesystem_is_case_insensitive`` probes the bundle).
- ``_apply_plan`` creates the links and rewrites the attributes. When a link
  slot is already occupied it VERIFIES the occupant resolves to the planned
  source and raises otherwise -- it never assumes a pre-existing entry is
  correct, which is the second line of defense against silently repointing a
  clip at the wrong file.
- The caller then writes ``Info.fcpxml`` back atomically
  (``write_fcpxml_document``: temp file + fsync + ``os.replace``), so a crash
  mid-write can never leave a truncated document behind.

Why this exists
---------------
``resolve_asset`` (see ``core/bindings.py``) already RESOLVES an absolute
``file://`` src at compile time, so a render works without this step. This
module is about MAKING THE BUNDLE PORTABLE: after it runs, the document no
longer depends on absolute paths baked into the FCPXML.

No silent failures: a hard problem raises ``MediaConsolidationError``; offline
media is surfaced as a warning on the returned result; nothing is ever
overwritten in place.

Main callers: ``cli.main`` for the ``--symlink-media`` flag on the ``render``,
``server run``, and ``studio`` commands.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote, urlparse

from .bindings import file_url_path
from .errors import FCPXMLParseError, FCPXMLWriteConflictError, MediaConsolidationError
from .parser import read_fcpxml_root


def encode_src_segment(name: str) -> str:
    """Percent-encode ONE path segment for use inside a ``media-rep src``.

    ``src`` locators are parsed with ``urllib.parse.urlparse`` + ``unquote`` (see
    ``core/bindings.py:relative_src_path``), so a raw filename containing a URL
    delimiter (``#``, ``?``, ``%``) would be split into path + query/fragment and
    never resolve. Encoding every segment we WRITE keeps the round-trip exact:
    ``clip#1.mov`` -> ``clip%231.mov`` -> unquote -> ``clip#1.mov``.

    ``safe=""`` also encodes ``/``, which is correct for a single segment.

    Main callers: ``_apply_plan`` (consolidated ``Media/<name>``) and
    ``core/proxy_media._proxy_src`` (generated proxy names).
    """

    return quote(name, safe="")


def write_fcpxml_document(
    path: Path, prolog: str, root: ET.Element, *, expected_bytes: bytes
) -> None:
    """Atomically replace ``path`` with ``prolog`` + the serialized ``root``.

    Procedurally:
    1. Serialize the tree to a sibling temp file, flush + fsync it.
    2. Re-read ``path`` and require it to still equal ``expected_bytes`` -- the
       exact bytes the caller parsed ``root`` from. If anything (Final Cut,
       Studio, another ``bladeworks`` command) rewrote the document in the
       meantime, raise ``FCPXMLWriteConflictError`` and leave the file alone:
       publishing the stale tree would silently delete those edits, and the
       proxy path in particular can spend minutes encoding between read and
       write.
    3. ``os.replace`` the temp file over the target.

    Steps 1 + 3 make the write all-or-nothing (an interruption or a disk-full
    error leaves either the complete old file or the complete new file -- never
    a truncated ``Info.fcpxml``); step 2 is a compare-and-replace that closes the
    read-mutate-write window. It is a check, not a lock: a writer landing in the
    microseconds between the comparison and the replace is still not detected.

    Main callers: ``consolidate_bundle_media`` and
    ``core/proxy_media.generate_proxies`` after they mutate the tree.
    """

    serialized = prolog + ET.tostring(root, encoding="unicode") + "\n"
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if path.read_bytes() != expected_bytes:
            raise FCPXMLWriteConflictError(
                f"{path} changed on disk after it was read; refusing to overwrite "
                f"the newer document (re-run the command to pick up those changes)"
            )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

MEDIA_DIR_NAME = "Media"


@dataclass
class MediaConsolidationResult:
    """What a consolidation pass did, for a concise human/JSON report.

    - ``bundle`` / ``info_path`` / ``media_dir``: the resolved locations.
    - ``linked``: newly created ``(target_name, source_path)`` symlinks.
    - ``already_linked``: links that already pointed at the right source
      (idempotent re-runs land here, not in ``linked``).
    - ``already_inside``: media-reps whose src already lived inside ``Media/``.
    - ``rewritten_src``: how many ``src`` attributes were repointed.
    - ``warnings``: offline / remote / unresolvable srcs we could not link.
    - ``changed``: whether ``Info.fcpxml`` needs to be written back.
    """

    bundle: Path
    info_path: Path
    media_dir: Path
    linked: list[tuple[str, Path]] = field(default_factory=list)
    already_linked: list[tuple[str, Path]] = field(default_factory=list)
    already_inside: int = 0
    rewritten_src: int = 0
    warnings: list[str] = field(default_factory=list)
    changed: bool = False

    def summary_lines(self) -> list[str]:
        """Render a short, deterministic multi-line report for stderr/stdout."""

        lines = [f"symlink-media: {self.bundle}"]
        for name, source in self.linked:
            lines.append(f"  linked   Media/{name} -> {source}")
        for name, source in self.already_linked:
            lines.append(f"  exists   Media/{name} -> {source}")
        if self.already_inside:
            lines.append(f"  in-media {self.already_inside} media-rep(s) already inside Media/")
        lines.append(
            f"  rewrote {self.rewritten_src} src attribute(s); "
            f"{'wrote' if self.changed else 'no change to'} Info.fcpxml"
        )
        for warning in self.warnings:
            lines.append(f"  warning  {warning}")
        return lines


@dataclass
class _PlannedLink:
    """One media-rep that references an external file we will link + rewrite."""

    element: ET.Element
    old_src: str
    source: Path  # resolved, absolute, existing external file
    target_name: str  # basename used under Media/


def _local_tag(element: ET.Element) -> str:
    """Local element name, stripping any XML namespace (mirrors parser._tag)."""

    return element.tag.rsplit("}", 1)[-1]


def _resolve_bundle(input_path: Path) -> tuple[Path, Path]:
    """Return ``(bundle_dir, info_path)`` or fail loudly.

    ``--symlink-media`` consolidates INTO a bundle's ``Media/`` folder, which
    only a ``.fcpxmld`` bundle has. A plain ``.fcpxml`` file is rejected by name
    rather than silently inventing a ``Media/`` next to it.
    """

    resolved = Path(input_path).expanduser().resolve()
    if not resolved.is_dir():
        raise MediaConsolidationError(
            f"--symlink-media requires a .fcpxmld bundle directory (holding "
            f"Info.fcpxml + Media/); got {resolved}, which is not a directory"
        )
    info = resolved / "Info.fcpxml"
    if not info.is_file():
        raise MediaConsolidationError(
            f"--symlink-media bundle {resolved} has no Info.fcpxml at its root"
        )
    return resolved, info


def _resolve_source_file(raw_src: Optional[str], bundle_dir: Path) -> Optional[Path]:
    """Resolve one media-rep ``src`` to an EXISTING local file, or ``None``.

    Accepts, in order: an absolute ``file://`` URI, a bare absolute path, and a
    bundle-relative path (resolved against the bundle root). A real remote
    scheme (``http``/``https``) or a non-existent path returns ``None`` so the
    caller can warn instead of linking a phantom.
    """

    if not raw_src:
        return None
    absolute = file_url_path(raw_src)
    if absolute is None:
        parsed = urlparse(raw_src)
        if parsed.scheme not in ("", "file"):
            return None  # a genuine remote scheme is never a local bundle file
        candidate = Path(unquote(parsed.path))
        absolute = candidate if candidate.is_absolute() else (bundle_dir / candidate)
    resolved = absolute.expanduser().resolve()
    return resolved if resolved.is_file() else None


def _is_inside(path: Path, directory: Path) -> bool:
    """True when ``path`` (already resolved) lives at/under ``directory``."""

    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _plan_consolidation(
    root: ET.Element,
    bundle_dir: Path,
    media_dir: Path,
    result: MediaConsolidationResult,
) -> list[_PlannedLink]:
    """Classify every media-rep; return the external ones to link + rewrite.

    Populates ``result.already_inside`` and ``result.warnings`` as a side
    effect so the caller reports offline media loudly without failing.
    """

    media_dir_resolved = media_dir.resolve()
    planned: list[_PlannedLink] = []
    for asset in (el for el in root.iter() if _local_tag(el) == "asset"):
        for rep in (child for child in asset if _local_tag(child) == "media-rep"):
            raw_src = rep.get("src")
            source = _resolve_source_file(raw_src, bundle_dir)
            if source is None:
                asset_id = asset.get("id") or "?"
                result.warnings.append(
                    f"asset {asset_id!r} media-rep src {raw_src!r} does not resolve "
                    f"to an existing local file; left unchanged"
                )
                continue
            if _is_inside(source, media_dir_resolved):
                result.already_inside += 1
                continue
            planned.append(
                _PlannedLink(
                    element=rep,
                    old_src=raw_src or "",
                    source=source,
                    target_name=source.name,
                )
            )
    return planned


def _filesystem_is_case_insensitive(directory: Path) -> bool:
    """Probe whether names in ``directory`` are matched case-insensitively.

    Why this exists: ``_validate_no_collisions`` must decide whether
    ``Clip.mov`` and ``clip.mov`` would land on the SAME ``Media/`` entry. That
    depends on the destination volume (APFS/HFS+ default to case-insensitive on
    macOS; Linux ext4 and case-sensitive APFS are not), not on the platform, so
    we ask the filesystem directly instead of guessing from ``sys.platform``.

    Procedurally: create an empty temp file in ``directory`` whose name is
    guaranteed to contain letters, then check whether the SAME name with every
    letter's case swapped also ``exists()``. Only a case-insensitive volume
    answers yes. The probe file is always removed.

    ``directory`` should be the bundle root (which always exists) rather than
    ``Media/``, which may not have been created yet and must stay absent when
    validation aborts.

    Main callers: ``consolidate_bundle_media`` (result passed into
    ``_validate_no_collisions``).
    """

    fd, probe_name = tempfile.mkstemp(
        prefix=".bladeworks-case-probe-", suffix=".CaseProbe", dir=str(directory)
    )
    os.close(fd)
    probe = Path(probe_name)
    try:
        swapped = probe.with_name(probe.name.swapcase())
        if swapped.name == probe.name:  # cannot happen given the suffix; be loud
            raise MediaConsolidationError(
                f"case-sensitivity probe name {probe.name!r} has no letters to swap"
            )
        return swapped.exists()
    finally:
        probe.unlink()


def _collision_key(name: str, case_insensitive: bool) -> str:
    """Normalize a ``Media/`` basename to the identity the destination FS uses."""

    return name.casefold() if case_insensitive else name


def _validate_no_collisions(
    planned: list[_PlannedLink], media_dir: Path, case_insensitive: bool
) -> None:
    """Reject two DISTINCT sources that would claim the same ``Media/<name>``.

    Same source referenced by several media-reps is fine (one link, many
    rewrites). A basename already occupied in ``Media/`` by a different real
    file is also a hard conflict. Everything is checked BEFORE any link is made,
    so a conflict aborts with zero on-disk changes.

    ``case_insensitive`` selects the destination filesystem's notion of "same
    name" (see ``_filesystem_is_case_insensitive``): when true, ``Clip.mov`` and
    ``clip.mov`` are one slot and two different sources with those names are a
    conflict, because the second symlink would silently alias the first.
    """

    claimed: dict[str, _PlannedLink] = {}
    conflicts: list[str] = []
    for link in planned:
        key = _collision_key(link.target_name, case_insensitive)
        prior = claimed.get(key)
        if prior is None:
            claimed[key] = link
            continue
        if prior.source == link.source:
            continue
        detail = ""
        if prior.target_name != link.target_name:
            detail = (
                f" ({prior.target_name!r} and {link.target_name!r} differ only by "
                f"case, which the destination filesystem does not distinguish)"
            )
        conflicts.append(
            f"Media/{link.target_name} is claimed by two different files: "
            f"{prior.source} and {link.source}{detail}"
        )

    for link in claimed.values():
        existing = media_dir / link.target_name
        if not existing.exists() and not existing.is_symlink():
            continue
        # Something already sits at the target. It is only OK when it already
        # points at (or is) the same source we intend to link.
        try:
            if existing.resolve() == link.source.resolve():
                continue
        except OSError:
            pass
        conflicts.append(
            f"Media/{link.target_name} already exists and points elsewhere than "
            f"{link.source}; refusing to overwrite"
        )

    if conflicts:
        raise MediaConsolidationError(
            "cannot consolidate media without overwriting or ambiguity:\n  - "
            + "\n  - ".join(sorted(conflicts))
        )


def _verify_existing_link(target: Path, link: _PlannedLink) -> None:
    """Raise unless the entry already at ``target`` resolves to ``link.source``.

    Why this exists: ``_apply_plan`` treats an occupied ``Media/<name>`` as an
    idempotent re-run. That is only safe if the occupant really is our link;
    on a case-insensitive volume ``Media/clip.mov`` may actually be the
    ``Media/Clip.mov`` created a moment ago for a DIFFERENT file, and a dangling
    or foreign symlink is possible too. Checking here (in addition to
    validation) means the apply stage can never quietly alias one clip to
    another.
    """

    try:
        actual = target.resolve()
    except OSError as exc:
        raise MediaConsolidationError(
            f"Media/{target.name} already exists but cannot be resolved "
            f"({exc}); refusing to treat it as the link for {link.source}"
        ) from exc
    if actual != link.source:
        raise MediaConsolidationError(
            f"Media/{target.name} already exists and resolves to {actual}, not "
            f"{link.source}; refusing to repoint the media-rep at the wrong file"
        )


def _apply_plan(
    planned: list[_PlannedLink],
    media_dir: Path,
    result: MediaConsolidationResult,
) -> None:
    """Create the symlinks and repoint each media-rep ``src`` to Media/<name>.

    Validation already guaranteed no destructive collision, so link creation
    here is either "make a new absolute symlink" or "a correct link is already
    present" (idempotent re-run). "Correct" is VERIFIED, not assumed: an
    occupant that resolves anywhere other than ``link.source`` raises
    ``MediaConsolidationError``, so a case-variant alias or a stale link can
    never silently repoint a clip at the wrong file.

    The ``src`` rewrite is a relative ``Media/<encoded name>`` URL locator; the
    segment is percent-encoded (``encode_src_segment``) because the parser
    unquotes it, while the on-disk symlink keeps the raw filename.
    """

    if planned:
        media_dir.mkdir(parents=True, exist_ok=True)

    linked_names: set[str] = set()
    for link in planned:
        target = media_dir / link.target_name
        if link.target_name not in linked_names:
            if target.is_symlink() or target.exists():
                _verify_existing_link(target, link)
                result.already_linked.append((link.target_name, link.source))
            else:
                os.symlink(link.source, target)
                result.linked.append((link.target_name, link.source))
            linked_names.add(link.target_name)

        new_src = f"{MEDIA_DIR_NAME}/{encode_src_segment(link.target_name)}"
        if link.element.get("src") != new_src:
            link.element.set("src", new_src)
            result.rewritten_src += 1
            result.changed = True


def _split_prolog(raw: bytes) -> tuple[str, str]:
    """Split raw FCPXML into (prolog, root_onward) at the first ``<fcpxml``.

    ElementTree drops the ``<?xml ...?>`` declaration and the ``<!DOCTYPE
    fcpxml>`` line on re-serialization, so we preserve the exact bytes before
    the root element and re-attach them ourselves when writing back.
    """

    text = raw.decode("utf-8")
    index = text.find("<fcpxml")
    if index < 0:
        raise FCPXMLParseError("Info.fcpxml has no <fcpxml> root element")
    return text[:index], text[index:]


def consolidate_bundle_media(input_path: Path) -> MediaConsolidationResult:
    """Symlink external media into ``Media/`` and repoint ``Info.fcpxml``.

    Procedurally:
    1. Resolve the input to a ``.fcpxmld`` bundle (fail loudly otherwise).
    2. Securely read + parse ``Info.fcpxml`` (reusing the renderer's XML
       hardening), keeping the original prolog so the DOCTYPE round-trips.
    3. PLAN: classify every ``<asset><media-rep src>`` -- external files become
       planned links; already-inside-Media reps and offline srcs are recorded.
    4. VALIDATE: abort before touching disk if two distinct files collide on one
       ``Media/<name>`` (case-insensitively when the bundle's volume is).
    5. APPLY: create the absolute symlinks (verifying any pre-existing one) and
       rewrite each src to ``Media/<percent-encoded name>``.
    6. Atomically write ``Info.fcpxml`` back only when something changed,
       refusing if the file changed on disk since step 2.

    Returns a ``MediaConsolidationResult`` describing exactly what happened.

    Main callers: ``cli.main`` for ``--symlink-media``.
    """

    bundle_dir, info_path = _resolve_bundle(input_path)
    media_dir = bundle_dir / MEDIA_DIR_NAME

    raw = info_path.read_bytes()
    # read_fcpxml_root applies the shared entity/DOCTYPE hardening. We re-parse
    # the same bytes into a tree we can mutate + re-serialize.
    read_fcpxml_root(bundle_dir)  # validates safety + shape; raises on trouble
    prolog, _ = _split_prolog(raw)
    root = ET.fromstring(raw)

    # Preserve a default namespace if the document declares one, so re-serialize
    # does not sprout ``ns0:`` prefixes. FCPXML is namespace-less in practice.
    if root.tag.startswith("{"):
        uri = root.tag[1:].split("}", 1)[0]
        ET.register_namespace("", uri)

    result = MediaConsolidationResult(
        bundle=bundle_dir, info_path=info_path, media_dir=media_dir
    )
    planned = _plan_consolidation(root, bundle_dir, media_dir, result)
    _validate_no_collisions(
        planned, media_dir, case_insensitive=_filesystem_is_case_insensitive(bundle_dir)
    )
    _apply_plan(planned, media_dir, result)

    if result.changed:
        write_fcpxml_document(info_path, prolog, root, expected_bytes=raw)

    return result
