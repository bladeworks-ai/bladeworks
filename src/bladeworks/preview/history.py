"""Immutable, process-local FCPXML history for one opened bundle.

Architecture map
================

    managed source save
        -> truncate snapshots after the cursor
        -> write one immutable, full-document snapshot
        -> move the in-memory cursor to that snapshot

    undo / redo
        -> move the cursor across existing snapshots
        -> let OpenedSourceStore materialize the selected bytes atomically

History files are disposable session state. They are complete FCPXML documents,
not patches, and this module never reads or writes the live ``Info.fcpxml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HistoryPosition:
    """Public cursor metadata returned after a history operation."""

    index: int
    length: int
    version: str


@dataclass(frozen=True)
class HistorySnapshot:
    """One immutable full-document snapshot."""

    sequence: int
    version: str
    path: Path


@dataclass(frozen=True)
class HistoryCheckpoint:
    """Recoverable copy of history bookkeeping before a live-file commit.

    Why this exists: source replacement prepares history before replacing
    ``Info.fcpxml``. If that final atomic replacement fails, the rejected edit
    must not remain selected or become available through Undo.
    """

    snapshots: tuple[HistorySnapshot, ...]
    cursor: int
    next_sequence: int


class HistoryBoundaryError(RuntimeError):
    """Raised when undo or redo cannot move in the requested direction."""

    def __init__(self, boundary: str) -> None:
        super().__init__(f"history is already at the {boundary}")
        self.boundary = boundary


class SessionHistory:
    """Own immutable snapshots and one in-memory cursor.

    Main callers:
    - ``OpenedSourceStore`` during startup, managed saves, undo, and redo.

    Why this exists:
    Keeping cursor mechanics separate from live-file replacement makes it
    impossible for an undo bookkeeping operation to mutate the project bundle
    by accident.
    """

    def __init__(self, directory: Path, *, limit: int) -> None:
        if limit < 1:
            raise ValueError("history limit must be at least 1")
        self.directory = Path(directory).resolve()
        self.limit = limit
        self.directory.mkdir(parents=True, exist_ok=True)
        self._snapshots: list[HistorySnapshot] = []
        self._cursor = -1
        self._next_sequence = 0

    @property
    def position(self) -> HistoryPosition:
        if self._cursor < 0:
            raise RuntimeError("history has not been initialized")
        selected = self._snapshots[self._cursor]
        return HistoryPosition(
            index=self._cursor,
            length=len(self._snapshots),
            version=selected.version,
        )

    @property
    def selected_version(self) -> str:
        return self.position.version

    def selected_bytes(self) -> bytes:
        if self._cursor < 0:
            raise RuntimeError("history has not been initialized")
        return self._snapshots[self._cursor].path.read_bytes()

    def initialize(self, xml: bytes, *, version: str) -> HistoryPosition:
        """Create the sole initial snapshot.

        Main callers:
        - ``OpenedSourceStore.open`` after the source compiles successfully.
        """

        if self._snapshots:
            raise RuntimeError("history is already initialized")
        self._append_file(xml, version=version)
        self._cursor = 0
        return self.position

    def append(self, xml: bytes, *, version: str) -> HistoryPosition:
        """Append a new branch head, truncating an incompatible redo tail."""

        if self._cursor < 0:
            return self.initialize(xml, version=version)
        if version == self.selected_version and xml == self.selected_bytes():
            return self.position
        self._truncate_after_cursor()
        self._append_file(xml, version=version)
        self._cursor = len(self._snapshots) - 1
        self._trim_to_limit()
        return self.position

    def checkpoint(self) -> HistoryCheckpoint:
        """Capture enough state to undo a prepared, uncommitted append."""

        return HistoryCheckpoint(
            snapshots=tuple(self._snapshots),
            cursor=self._cursor,
            next_sequence=self._next_sequence,
        )

    def restore(self, checkpoint: HistoryCheckpoint) -> HistoryPosition:
        """Restore a checkpoint after the corresponding live save failed."""

        retained = {snapshot.sequence for snapshot in checkpoint.snapshots}
        for snapshot in self._snapshots:
            if snapshot.sequence not in retained:
                snapshot.path.unlink(missing_ok=True)
        self._snapshots = list(checkpoint.snapshots)
        self._cursor = checkpoint.cursor
        self._next_sequence = checkpoint.next_sequence
        return self.position

    def commit(self, checkpoint: HistoryCheckpoint) -> HistoryPosition:
        """Delete snapshots superseded by a successfully installed branch."""

        retained = {snapshot.sequence for snapshot in self._snapshots}
        for snapshot in checkpoint.snapshots:
            if snapshot.sequence not in retained:
                snapshot.path.unlink(missing_ok=True)
        return self.position

    def undo(self) -> tuple[HistoryPosition, bytes]:
        if self._cursor <= 0:
            raise HistoryBoundaryError("start")
        self._cursor -= 1
        return self.position, self.selected_bytes()

    def redo(self) -> tuple[HistoryPosition, bytes]:
        if self._cursor < 0 or self._cursor >= len(self._snapshots) - 1:
            raise HistoryBoundaryError("end")
        self._cursor += 1
        return self.position, self.selected_bytes()

    def _append_file(self, xml: bytes, *, version: str) -> None:
        path = self.directory / f"{self._next_sequence:06d}.fcpxml"
        self._next_sequence += 1
        with path.open("xb") as handle:
            handle.write(xml)
            handle.flush()
            os.fsync(handle.fileno())
        self._snapshots.append(
            HistorySnapshot(
                sequence=self._next_sequence - 1,
                version=version,
                path=path,
            )
        )

    def _truncate_after_cursor(self) -> None:
        self._snapshots = self._snapshots[: self._cursor + 1]

    def _trim_to_limit(self) -> None:
        while len(self._snapshots) > self.limit:
            self._snapshots.pop(0)
            self._cursor -= 1
