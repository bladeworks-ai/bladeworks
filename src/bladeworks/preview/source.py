"""Opened ``.fcpxmld`` source, content versions, and atomic replacement.

Architecture map
================

    command boundary
        -> read exact ``Info.fcpxml`` bytes and compute SHA-256
        -> compile a changed disk version, without a watcher
        -> retain the last successful compile if new disk bytes are invalid

    managed PUT
        -> compare the caller's content version
        -> compile a hidden sibling candidate under the configured policy
        -> atomically replace ``Info.fcpxml``
        -> advance immutable session history

The bundle is the durable source of truth. Compiled documents and history are
process-local accelerators and can be discarded when the server exits.
"""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
import threading
import weakref
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

from ..core.compiler import CompileResult, compile_fcpxml
from ..core.errors import FCPXMLParseError, FCPXMLRenderError
from ..core.model import RenderDocument
from ..core.parser import enumerate_library_projects, read_fcpxml_root, validate_project_ref
from ..core.report import CompatibilityReport
from .contracts import PreviewAPIError
from .history import HistoryBoundaryError, SessionHistory


EDITOR_PROFILE = "bladeworks-fcpxml-v1"


def source_version(xml: bytes) -> str:
    """Return the stable public content version for exact FCPXML bytes."""

    return f"sha256:{hashlib.sha256(xml).hexdigest()}"


@dataclass(frozen=True)
class LoadedProject:
    """One Project compile pinned by preview or render work."""

    version: str
    project_ref: str
    document: RenderDocument
    report: CompatibilityReport


@dataclass(frozen=True)
class ProjectCatalogEntry:
    project_ref: str
    library_name: str | None
    event_name: str | None
    project_name: str | None
    uid: str | None
    degraded: bool
    # When a single Project fails to compile (an unsupported construct, a bad
    # reference), it is still listed here so the whole library stays openable --
    # ``error`` carries the reason and marks the entry unopenable. ``None`` means
    # the Project compiled and can be selected.
    error: str | None = None

    def payload(self) -> dict[str, object]:
        return {
            "projectRef": self.project_ref,
            "libraryName": self.library_name,
            "eventName": self.event_name,
            "projectName": self.project_name,
            "uid": self.uid,
            "degraded": self.degraded,
            "openable": self.error is None,
            "error": self.error,
        }


def _canonical_project_xml(project: ET.Element) -> str:
    """Content identity of one ``<project>`` subtree, independent of formatting.

    C14N (``ET.canonicalize``) normalizes attribute order, quoting, and
    insignificant whitespace, so a Project that survived a Studio round-trip
    untouched (the browser patches the parsed DOM in place and re-serializes
    it) compares equal to the bytes it was loaded from, while any real edit
    inside the Project changes the string.

    Main callers: ``OpenedSourceStore._compile_path`` -- both to RECORD the
    identity of a Project that failed to compile and to CHECK a later failing
    Project against those records (see ``LoadedLibrary.frozen_projects``).
    """

    return ET.canonicalize(ET.tostring(project, encoding="unicode"), strip_text=True)


@dataclass(frozen=True)
class LoadedLibrary:
    """Every Project compiled from one exact complete-library byte string.

    ``frozen_projects`` holds the canonical XML (``_canonical_project_xml``) of
    every Project that did NOT compile and was tolerated as an unopenable
    catalog entry -- in THIS library version or any earlier one this session.
    Those Projects are READ-ONLY for the rest of the session: a later source
    replacement may keep them verbatim (still tolerated) but any change to
    their XML is rejected, because there is no compile to validate the edit
    against. Identity is by content, not by position or uid, so reordering or
    renaming sibling Projects never spoofs or breaks the rule.

    The set only ever grows: an identity tolerated once stays tolerated, so
    that after an edit FIXES the bad Project, undoing that edit (which restores
    the exact tolerated XML) is still accepted rather than rejected as a new
    ``source_invalid`` Project.
    """

    version: str
    projects: tuple[LoadedProject, ...]
    catalog: tuple[ProjectCatalogEntry, ...]
    frozen_projects: frozenset[str] = frozenset()

    @property
    def degraded(self) -> bool:
        return any(project.report.degraded for project in self.projects)

    def project(self, project_ref: str) -> LoadedProject | None:
        return next(
            (project for project in self.projects if project.project_ref == project_ref),
            None,
        )


@dataclass(frozen=True)
class SourceStatus:
    """Separate the latest disk bytes from the last usable compiled runtime."""

    disk_version: str
    loaded_version: str | None
    compile_status: str
    degraded: bool
    history_index: int
    history_length: int
    error: str | None
    projects: tuple[ProjectCatalogEntry, ...] = ()

    def payload(self) -> dict[str, object]:
        result: dict[str, object] = {
            "diskVersion": self.disk_version,
            "loadedVersion": self.loaded_version,
            "compileStatus": self.compile_status,
            "degraded": self.degraded,
            "historyIndex": self.history_index,
            "historyLength": self.history_length,
            "editorProfile": EDITOR_PROFILE,
            "projects": [project.payload() for project in self.projects],
        }
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass(frozen=True)
class SourceRead:
    xml: bytes
    status: SourceStatus


class OpenedSourceStore:
    """Serialize access to one bundle and its disposable session history.

    Main callers:
    - source HTTP routes for complete-document reads and writes.
    - preview and render services before they pin a requested source version.

    Why this exists:
    The renderer should only see immutable ``RenderDocument`` values. This
    boundary owns all mutable filesystem behavior and makes stale or invalid
    disk contents explicit instead of silently rendering an older document.
    """

    def __init__(
        self,
        *,
        bundle_path: Path,
        source_path: Path,
        history: SessionHistory,
        strict: bool,
        initial_xml: bytes,
        initial_loaded: LoadedLibrary,
    ) -> None:
        self.bundle_path = bundle_path
        self.source_path = source_path
        self.history = history
        self.strict = strict
        self._lock = threading.RLock()
        self._disk_xml = initial_xml
        self._disk_version = initial_loaded.version
        self._loaded = initial_loaded
        self._compile_error: str | None = None
        self._reports_by_document: dict[
            int,
            tuple[weakref.ReferenceType[RenderDocument], CompatibilityReport],
        ] = {}
        for project in initial_loaded.projects:
            self._remember_report_locked(project.document, project.report)

    @classmethod
    def open(
        cls,
        bundle_path: Path,
        *,
        history_directory: Path,
        history_limit: int = 50,
        strict: bool = False,
    ) -> "OpenedSourceStore":
        """Validate, compile, and snapshot one canonical bundle at startup."""

        bundle = Path(bundle_path).expanduser().resolve()
        if not bundle.is_dir() or bundle.suffix.lower() != ".fcpxmld":
            raise PreviewAPIError(
                "source_not_found",
                f"Opened source must be an existing .fcpxmld directory: {bundle}",
                status=404,
            )
        source_path = bundle / "Info.fcpxml"
        try:
            xml = source_path.read_bytes()
        except OSError as error:
            raise PreviewAPIError(
                "source_not_found",
                f"Could not read {source_path}: {error}",
                status=404,
            ) from error
        loaded = cls._compile_path(
            candidate_path=source_path,
            canonical_path=source_path,
            strict=strict,
            # Initial open tolerates any bad Project so the library still opens;
            # from then on those Projects are read-only and every other Project
            # must compile on each replacement (see _compile_path).
            initial_open=True,
        )
        history = SessionHistory(history_directory, limit=history_limit)
        history.initialize(xml, version=loaded.version)
        return cls(
            bundle_path=bundle,
            source_path=source_path,
            history=history,
            strict=strict,
            initial_xml=xml,
            initial_loaded=loaded,
        )

    def read_source(self) -> SourceRead:
        """Return exact latest disk bytes, including malformed external bytes."""

        with self._lock:
            self._freshen_locked()
            return SourceRead(xml=self._disk_xml, status=self._status_locked())

    def reload(self) -> SourceStatus:
        """Adopt the latest valid disk version without appending history."""

        with self._lock:
            self._freshen_locked()
            self._raise_if_invalid_locked()
            return self._status_locked()

    def replace(self, xml: bytes, *, expected_version: str) -> SourceStatus:
        """Validate and atomically install one complete FCPXML document.

        Main callers:
        - ``PUT /api/editor/source``.

        The expected version is compared against disk, not the loaded compile,
        so an external edit cannot be accidentally overwritten.
        """

        if not isinstance(xml, bytes):
            raise PreviewAPIError("source_invalid", "FCPXML body must be bytes.", status=400)
        with self._lock:
            self._freshen_locked()
            self._require_expected_locked(expected_version)
            if xml == self._disk_xml:
                self._raise_if_invalid_locked()
                return self._status_locked()

            candidate_path = self._write_candidate(xml)
            replaced_candidate = False
            history_checkpoint = self.history.checkpoint()
            try:
                loaded = self._compile_path(
                    candidate_path=candidate_path,
                    canonical_path=self.source_path,
                    strict=self.strict,
                    previous=self._loaded,
                )
                latest_xml = self._read_live_bytes()
                latest_version = source_version(latest_xml)
                if latest_version != expected_version:
                    self._record_disk_observation_locked(latest_xml)
                    raise self._version_conflict(expected_version, latest_version)

                # A valid direct edit becomes the departure point before the
                # managed branch is appended. Invalid external bytes are never
                # made undoable because restoring them would make the runtime
                # unusable again.
                if (
                    self._compile_error is None
                    and self._disk_version != self.history.selected_version
                ):
                    self.history.append(self._disk_xml, version=self._disk_version)

                # Persist the undo snapshot before committing the live source.
                # A history I/O failure must leave Info.fcpxml untouched.
                self.history.append(xml, version=loaded.version)
                os.replace(candidate_path, self.source_path)
                replaced_candidate = True
                self._fsync_directory(self.source_path.parent)
                self._install_loaded_locked(xml, loaded)
                self.history.commit(history_checkpoint)
                return self._status_locked()
            except BaseException:
                if not replaced_candidate:
                    self.history.restore(history_checkpoint)
                raise
            finally:
                if not replaced_candidate:
                    candidate_path.unlink(missing_ok=True)

    def undo(self) -> SourceStatus:
        """Restore the previous managed state, accounting for direct edits."""

        with self._lock:
            self._freshen_locked()
            if self._disk_version != self.history.selected_version:
                xml = self.history.selected_bytes()
                return self._materialize_history_locked(xml)
            moved = False
            try:
                _position, xml = self.history.undo()
                moved = True
                return self._materialize_history_locked(xml)
            except HistoryBoundaryError as error:
                raise self._history_error(error) from error
            except BaseException:
                if moved:
                    self.history.redo()
                raise

    def redo(self) -> SourceStatus:
        with self._lock:
            self._freshen_locked()
            moved = False
            try:
                _position, xml = self.history.redo()
                moved = True
                return self._materialize_history_locked(xml)
            except HistoryBoundaryError as error:
                raise self._history_error(error) from error
            except BaseException:
                if moved:
                    self.history.undo()
                raise

    def require_current(
        self,
        source_version_value: str,
        project_ref: str,
    ) -> LoadedProject:
        """Return one Project from the exact current complete-library compile.

        The version check happens before Project selection. A reference is
        meaningful only inside the library bytes that produced it.
        """

        with self._lock:
            self._freshen_locked()
            self._raise_if_invalid_locked()
            if source_version_value != self._disk_version:
                raise self._version_conflict(source_version_value, self._disk_version)
            assert self._loaded is not None
            try:
                validate_project_ref(project_ref)
            except FCPXMLParseError as error:
                raise PreviewAPIError("invalid_project_ref", str(error), status=400) from error
            project = self._loaded.project(project_ref)
            if project is None:
                raise PreviewAPIError(
                    "project_not_found",
                    f"Project {project_ref!r} does not exist in source {source_version_value}.",
                    status=404,
                )
            return project

    def compatibility(self, source_version_value: str, project_ref: str) -> dict[str, object]:
        """Return the selected Project's compatibility report as public JSON."""

        loaded = self.require_current(source_version_value, project_ref)
        return {
            "sourceVersion": loaded.version,
            "projectRef": loaded.project_ref,
            "degraded": loaded.report.degraded,
            "compatibility": loaded.report.to_json(),
        }

    def report_for(self, document: RenderDocument) -> CompatibilityReport:
        with self._lock:
            owned = self._reports_by_document.get(id(document))
            if owned is None or owned[0]() is not document:
                raise PreviewAPIError(
                    "source_version_conflict",
                    "The render document is not owned by this opened source.",
                    status=409,
                )
            return copy.deepcopy(owned[1])

    def status(self) -> SourceStatus:
        with self._lock:
            self._freshen_locked()
            return self._status_locked()

    def _freshen_locked(self) -> None:
        xml = self._read_live_bytes()
        version = source_version(xml)
        if version == self._disk_version:
            return
        self._record_disk_observation_locked(xml)

    def _record_disk_observation_locked(self, xml: bytes) -> None:
        version = source_version(xml)
        self._disk_xml = xml
        self._disk_version = version
        candidate_path = self._write_candidate(xml)
        try:
            # An external write (Final Cut re-export, a hand edit) is the
            # source of truth, exactly like the bytes at initial open: a
            # Project it leaves uncompilable is tolerated as an unopenable,
            # read-only catalog entry rather than wedging every later save
            # behind ``source_invalid`` until Studio is restarted on the very
            # same bytes.
            loaded = self._compile_path(
                candidate_path=candidate_path,
                canonical_path=self.source_path,
                strict=self.strict,
                initial_open=True,
                previous=self._loaded,
            )
        except PreviewAPIError as error:
            self._compile_error = error.message
            return
        finally:
            candidate_path.unlink(missing_ok=True)
        self._install_loaded_locked(xml, loaded)

    def _install_loaded_locked(self, xml: bytes, loaded: LoadedLibrary) -> None:
        self._disk_xml = xml
        self._disk_version = loaded.version
        self._loaded = loaded
        self._compile_error = None
        for project in loaded.projects:
            self._remember_report_locked(project.document, project.report)

    def _remember_report_locked(
        self,
        document: RenderDocument,
        report: CompatibilityReport,
    ) -> None:
        """Retain report metadata only while a current or pinned document lives."""

        document_id = id(document)

        def discard(reference: weakref.ReferenceType[RenderDocument]) -> None:
            with self._lock:
                current = self._reports_by_document.get(document_id)
                if current is not None and current[0] is reference:
                    self._reports_by_document.pop(document_id, None)

        reference = weakref.ref(document, discard)
        self._reports_by_document[document_id] = (reference, copy.deepcopy(report))

    def _materialize_history_locked(self, xml: bytes) -> SourceStatus:
        candidate_path = self._write_candidate(xml)
        replaced_candidate = False
        try:
            loaded = self._compile_path(
                candidate_path=candidate_path,
                canonical_path=self.source_path,
                strict=self.strict,
                previous=self._loaded,
            )
            os.replace(candidate_path, self.source_path)
            replaced_candidate = True
            self._fsync_directory(self.source_path.parent)
            self._install_loaded_locked(xml, loaded)
            return self._status_locked()
        finally:
            if not replaced_candidate:
                candidate_path.unlink(missing_ok=True)

    def _write_candidate(self, xml: bytes) -> Path:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".Info.fcpxml.",
            suffix=".tmp",
            dir=self.source_path.parent,
        )
        path = Path(raw_path)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(xml)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path

    def _read_live_bytes(self) -> bytes:
        try:
            return self.source_path.read_bytes()
        except OSError as error:
            raise PreviewAPIError(
                "source_not_found",
                f"Could not read {self.source_path}: {error}",
                status=404,
            ) from error

    def _status_locked(self) -> SourceStatus:
        position = self.history.position
        loaded_version = self._loaded.version if self._loaded is not None else None
        degraded = self._loaded.degraded if self._loaded is not None else False
        return SourceStatus(
            disk_version=self._disk_version,
            loaded_version=loaded_version,
            compile_status="source_invalid" if self._compile_error else "ready",
            degraded=degraded,
            history_index=position.index,
            history_length=position.length,
            error=self._compile_error,
            projects=self._loaded.catalog if self._loaded is not None else (),
        )

    def _raise_if_invalid_locked(self) -> None:
        if self._compile_error is not None:
            raise PreviewAPIError("source_invalid", self._compile_error, status=422)

    def _require_expected_locked(self, expected_version: str) -> None:
        if expected_version != self._disk_version:
            raise self._version_conflict(expected_version, self._disk_version)

    @staticmethod
    def _version_conflict(expected: str, actual: str) -> PreviewAPIError:
        return PreviewAPIError(
            "source_version_conflict",
            f"Info.fcpxml changed after the editor loaded it (expected {expected}, found {actual}).",
            status=409,
            retryable=True,
        )

    @staticmethod
    def _history_error(error: HistoryBoundaryError) -> PreviewAPIError:
        code = "history_at_start" if error.boundary == "start" else "history_at_end"
        return PreviewAPIError(code, str(error), status=409)

    @staticmethod
    def _compile_path(
        *,
        candidate_path: Path,
        canonical_path: Path,
        strict: bool,
        initial_open: bool = False,
        previous: LoadedLibrary | None = None,
    ) -> LoadedLibrary:
        """Compile every Project before one library version is accepted.

        A Project that will not compile is handled by lifecycle stage:

        - INITIAL OPEN and an OBSERVED EXTERNAL EDIT (``initial_open=True``):
          the bytes are the source of truth (nobody in Studio authored them),
          so a Project that will not compile is recorded as an unopenable
          catalog entry and the rest still load -- opening a real library, or
          picking up a Final Cut re-export, is never blocked by one
          unsupported Project. Its canonical XML is kept in
          ``LoadedLibrary.frozen_projects``.
        - Every STUDIO-AUTHORED compile (a source replacement, undo/redo)
          passes the currently loaded library as ``previous`` without
          ``initial_open``.
          A failing Project is then tolerated ONLY if its canonical XML is one
          of ``previous.frozen_projects`` -- i.e. it is a known-unopenable
          Project carried through verbatim. That makes exactly those Projects
          read-only: editing one is rejected with ``project_read_only`` (there
          is no compile to validate the edit against), while any OTHER Project
          that stops compiling is a bad edit and is rejected atomically with
          ``source_invalid`` as before. A Project that an edit FIXES simply
          compiles and becomes openable; its frozen identity is still carried
          forward so an undo back to the tolerated XML is accepted.

        ``strict`` is orthogonal and always fatal on the first defect.

        Main callers: ``open`` (initial) and ``_record_disk_observation_locked``
        (external edits) with ``initial_open=True``; ``replace`` and
        ``_materialize_history_locked`` (undo/redo) without.
        """

        tolerated = frozenset() if previous is None else previous.frozen_projects
        previously_unopenable = {
            entry.project_ref: entry
            for entry in (previous.catalog if previous is not None else ())
            if entry.error is not None
        }

        try:
            root = read_fcpxml_root(candidate_path)
            addressed = enumerate_library_projects(root)
        except (FCPXMLRenderError, OSError) as error:
            raise PreviewAPIError("source_invalid", str(error), status=422) from error
        if not addressed:
            raise PreviewAPIError(
                "source_invalid",
                "document does not contain a project inside a library event",
                status=422,
            )
        canonical = canonical_path.resolve()
        projects: list[LoadedProject] = []
        catalog: list[ProjectCatalogEntry] = []
        # Identities tolerated in ANY earlier version stay tolerated (see
        # ``LoadedLibrary.frozen_projects``): a history snapshot that
        # re-introduces a once-tolerated Project verbatim must compile.
        frozen: set[str] = set(tolerated)
        version: str | None = None
        for address in addressed:
            try:
                compiled = compile_fcpxml(candidate_path, project=address.project_ref)
            except (FCPXMLRenderError, OSError) as error:
                # A Project that will not compile is tolerated (recorded as an
                # unopenable catalog entry, the rest keep loading) at initial
                # open, or later only when it is a known-unopenable Project
                # carried through verbatim. ``strict`` is always fatal. Its
                # name/uid come from the document address, not a compile, so
                # the listing is still complete either way.
                canonical_xml = _canonical_project_xml(address.project)
                tolerable = not strict and (initial_open or canonical_xml in tolerated)
                if not tolerable:
                    was_unopenable = previously_unopenable.get(address.project_ref)
                    if was_unopenable is not None and not strict:
                        # The Project was already unopenable and its XML changed:
                        # nothing can validate that edit, so it is read-only.
                        raise PreviewAPIError(
                            "project_read_only",
                            f"{address.project_ref} "
                            f"({was_unopenable.project_name or 'unnamed'}) cannot be "
                            f"opened ({was_unopenable.error}) and is read-only; its "
                            f"XML was modified. Revert the change to that Project.",
                            status=422,
                        ) from error
                    raise PreviewAPIError(
                        "source_invalid",
                        f"{address.project_ref}: {error}",
                        status=422,
                    ) from error
                frozen.add(canonical_xml)
                catalog.append(
                    ProjectCatalogEntry(
                        project_ref=address.project_ref,
                        library_name=address.library.get("name"),
                        event_name=address.event.get("name"),
                        project_name=address.project.get("name"),
                        uid=address.project.get("uid"),
                        degraded=True,
                        error=str(error),
                    )
                )
                continue
            if strict and compiled.report.has_strict_failures:
                failures = [
                    f"{item.fcpxml_path}: {item.disposition}"
                    for item in compiled.report.findings
                    if item.outcome in {"approximated", "omitted", "failed"}
                ]
                message = (
                    f"Unsupported FCPXML constructs in {address.project_ref}: "
                    + "; ".join(failures)
                )
                raise PreviewAPIError("unsupported_construct", message, status=422)

            source = replace(compiled.source, source_path=canonical)
            render = replace(compiled.render, source_path=canonical)
            report = copy.deepcopy(compiled.report)
            report.source_path = str(canonical)
            normalized = CompileResult(source=source, render=render, report=report)
            project_version = f"sha256:{normalized.render.source_sha256}"
            if version is None:
                version = project_version
            elif project_version != version:
                raise PreviewAPIError(
                    "source_invalid",
                    "Project compiles did not retain one complete-library source hash.",
                    status=422,
                )
            projects.append(
                LoadedProject(
                    version=project_version,
                    project_ref=address.project_ref,
                    document=normalized.render,
                    report=normalized.report,
                )
            )
            catalog.append(
                ProjectCatalogEntry(
                    project_ref=address.project_ref,
                    library_name=address.library.get("name"),
                    event_name=address.event.get("name"),
                    project_name=address.project.get("name"),
                    uid=address.project.get("uid"),
                    degraded=normalized.report.degraded,
                )
            )
        if version is None:
            # Every Project failed to compile: there is nothing to open, so this
            # is a genuine whole-library failure rather than a tolerable gap.
            details = "; ".join(
                f"{entry.project_ref}: {entry.error}" for entry in catalog if entry.error
            )
            raise PreviewAPIError(
                "source_invalid",
                f"No Project in the library could be compiled ({details})",
                status=422,
            )
        return LoadedLibrary(
            version=version,
            projects=tuple(projects),
            catalog=tuple(catalog),
            frozen_projects=frozenset(frozen),
        )

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
