"""Small exact-source provider used by focused preview and export tests.

Production uses ``OpenedSourceStore``. This registry keeps tests independent
from filesystem mutation while honoring the same content-version contract.
"""

from __future__ import annotations

import copy
import threading
from ..core.compiler import CompileResult
from ..core.model import RenderDocument
from ..core.report import CompatibilityReport
from .contracts import PreviewAPIError
from .source import LoadedProject


DEFAULT_REGISTERED_PROJECT_REF = "library[1]/event[1]/project[1]"


class RegisteredSourceProvider:
    """Thread-safe registry populated with immutable compiled source hashes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sources: dict[tuple[str, str], LoadedProject] = {}
        self._reports_by_document: dict[int, CompatibilityReport] = {}

    def register(
        self,
        source_version: str,
        compiled: CompileResult,
        *,
        project_ref: str = DEFAULT_REGISTERED_PROJECT_REF,
    ) -> None:
        value = LoadedProject(
            source_version,
            project_ref,
            compiled.render,
            copy.deepcopy(compiled.report),
        )
        with self._lock:
            key = (source_version, project_ref)
            previous = self._sources.get(key)
            if previous is not None:
                self._reports_by_document.pop(id(previous.document), None)
            self._sources[key] = value
            self._reports_by_document[id(compiled.render)] = value.report

    def require_current(
        self,
        source_version: str,
        project_ref: str,
    ) -> LoadedProject:
        with self._lock:
            value = self._sources.get((source_version, project_ref))
        if value is None:
            raise PreviewAPIError(
                "source_version_conflict",
                f"Source version and Project {(source_version, project_ref)!r} are not registered.",
                status=409,
            )
        return value

    def report_for(self, document: RenderDocument) -> CompatibilityReport:
        with self._lock:
            report = self._reports_by_document.get(id(document))
        if report is None:
            raise PreviewAPIError(
                "source_version_conflict",
                "The render document is not owned by a registered source version.",
                status=409,
            )
        return copy.deepcopy(report)
