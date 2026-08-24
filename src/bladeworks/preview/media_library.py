"""File-oriented inventory and staged import for one opened FCPXML bundle.

Architecture map
================

``<bundle>/Media``
    -> explicit recursive scan
    -> per-file PyAV stream probe (one failure does not abort the scan)
    -> immutable, path-sorted inventory snapshot plus explicit probe failures

An import copies to a hidden ``.partial`` sibling, flushes it, and publishes it
with an atomic no-overwrite hard link. Media changes never edit ``Info.fcpxml``
and never participate in FCPXML undo or redo.
"""

from __future__ import annotations

import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av


class MediaLibraryError(RuntimeError):
    """Stable media-library failure translated by the HTTP adapter."""

    def __init__(self, code: str, message: str, *, status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass(frozen=True)
class MediaRecord:
    relative_path: str
    filename: str
    size_bytes: int
    duration: Fraction | None
    width: int | None
    height: int | None
    frame_duration: Fraction | None
    has_video: bool
    has_audio: bool

    def to_json(self) -> dict[str, object]:
        return {
            "relativePath": self.relative_path,
            "filename": self.filename,
            "sizeBytes": self.size_bytes,
            "duration": float(self.duration) if self.duration is not None else None,
            "width": self.width,
            "height": self.height,
            "frameDuration": (
                f"{self.frame_duration.numerator}/{self.frame_duration.denominator}s"
                if self.frame_duration is not None
                else None
            ),
            "hasVideo": self.has_video,
            "hasAudio": self.has_audio,
        }


@dataclass(frozen=True)
class MediaProbeFailure:
    """One Media file the scan could not open as video or audio."""

    relative_path: str
    message: str

    def to_json(self) -> dict[str, object]:
        return {"relativePath": self.relative_path, "message": self.message}


@dataclass(frozen=True)
class MediaInventory:
    items: tuple[MediaRecord, ...]
    failures: tuple[MediaProbeFailure, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "items": [item.to_json() for item in self.items],
            "failures": [failure.to_json() for failure in self.failures],
        }


class MediaLibrary:
    """Manage only physical files below one bundle's media directory."""

    def __init__(self, bundle_path: Path, *, media_dir_name: str = "Media") -> None:
        bundle = bundle_path.expanduser().resolve()
        if not bundle.is_dir():
            raise MediaLibraryError(
                "source_not_found",
                f"FCPXML bundle does not exist: {bundle}",
                status=404,
            )
        if Path(media_dir_name).name != media_dir_name or media_dir_name in {"", ".", ".."}:
            raise ValueError("media_dir_name must be one directory name")
        self.bundle_path = bundle
        self.media_directory = bundle / media_dir_name
        self._lock = threading.RLock()
        self._inventory: MediaInventory | None = None

    def inventory(self) -> MediaInventory:
        """Return the current explicit snapshot, scanning once on first use."""

        with self._lock:
            if self._inventory is None:
                self._inventory = self._scan()
            return self._inventory

    def refresh(self) -> MediaInventory:
        """Replace the complete inventory snapshot from current disk state."""

        with self._lock:
            inventory = self._scan()
            self._inventory = inventory
            return inventory

    def resolve_media_path(self, relative_path: str) -> Path:
        """Resolve one browser-visible inventory path without escaping Media.

        Main callers:
        - the transient filmstrip and waveform sampler.

        The sampler accepts the same ``relativePath`` returned by inventory.
        It never accepts an arbitrary host path because the local API bearer
        token should not become a general-purpose file reader.
        """

        if not relative_path or Path(relative_path).is_absolute():
            raise MediaLibraryError(
                "invalid_media_path",
                "relativePath must identify one file below the opened bundle's Media directory.",
                status=400,
            )
        candidate = (self.bundle_path / relative_path).resolve()
        media_root = self.media_directory.resolve()
        try:
            candidate.relative_to(media_root)
        except ValueError as error:
            raise MediaLibraryError(
                "invalid_media_path",
                "relativePath must identify one file below the opened bundle's Media directory.",
                status=400,
            ) from error
        if not candidate.is_file() or self._ignored(candidate):
            raise MediaLibraryError(
                "media_not_found",
                f"Media file is missing: {relative_path}",
                status=404,
            )
        return candidate

    def import_file(self, source_path: Path, destination_name: str) -> MediaRecord:
        """Stage, publish without overwrite, probe, and inventory one local file.

        Main callers:
        - ``POST /api/editor/media/import``.

        The target is published before probing so direct filesystem readers see
        either the complete bytes or no file. A probe failure removes the newly
        imported target because the API never reports an unusable import as a
        success.
        """

        source = source_path.expanduser().resolve()
        if not source.is_file():
            raise MediaLibraryError(
                "media_source_not_found",
                f"Media import source is not a file: {source}",
                status=404,
            )
        self._validate_destination_name(destination_name)
        with self._lock:
            self.media_directory.mkdir(parents=True, exist_ok=True)
            target = self.media_directory / destination_name
            if target.exists():
                raise self._collision(target)
            stage = self.media_directory / f".{destination_name}.{uuid.uuid4().hex}.partial"
            published = False
            try:
                with source.open("rb") as incoming, stage.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                try:
                    os.link(stage, target)
                except FileExistsError as error:
                    raise self._collision(target) from error
                published = True
                stage.unlink()
                self._fsync_directory()
                record = self._probe(target)
            except Exception:
                stage.unlink(missing_ok=True)
                if published:
                    target.unlink(missing_ok=True)
                    self._fsync_directory()
                raise
            self._inventory = self._scan()
            return record

    def create_upload_stage(self, destination_name: str) -> Path:
        """Reserve a hidden stage path for an HTTP request body.

        The route streams directly into this file. Publication remains a
        separate operation so a disconnected or invalid request can only
        leave a hidden partial file, which the route removes in ``finally``.
        """

        self._validate_destination_name(destination_name)
        with self._lock:
            self.media_directory.mkdir(parents=True, exist_ok=True)
            target = self.media_directory / destination_name
            if target.exists():
                raise self._collision(target)
            return self.media_directory / f".{destination_name}.{uuid.uuid4().hex}.partial"

    def publish_upload(self, stage: Path, destination_name: str) -> MediaRecord:
        """Atomically publish and probe a completely written upload stage."""

        self._validate_destination_name(destination_name)
        stage = stage.resolve()
        if stage.parent != self.media_directory.resolve() or not stage.name.endswith(".partial"):
            raise MediaLibraryError(
                "invalid_request",
                "Upload stage is not owned by this media library.",
                status=400,
            )
        with self._lock:
            target = self.media_directory / destination_name
            if target.exists():
                raise self._collision(target)
            published = False
            try:
                try:
                    os.link(stage, target)
                except FileExistsError as error:
                    raise self._collision(target) from error
                published = True
                stage.unlink()
                self._fsync_directory()
                record = self._probe(target)
            except Exception:
                stage.unlink(missing_ok=True)
                if published:
                    target.unlink(missing_ok=True)
                    self._fsync_directory()
                raise
            self._inventory = self._scan()
            return record

    def _scan(self) -> MediaInventory:
        """Probe every visible Media file. Keep readable items; record the rest.

        Why this exists: a sidecar, damaged movie, or non-media file in Media
        must not prevent Studio from opening the FCPXML Project. Import of a
        single new file still raises through ``_probe`` so a failed upload is
        never reported as success.
        """

        if not self.media_directory.is_dir():
            return MediaInventory(items=())
        paths = sorted(
            (
                path
                for path in self.media_directory.rglob("*")
                if path.is_file() and not self._ignored(path)
            ),
            key=lambda path: path.relative_to(self.bundle_path).as_posix(),
        )
        records: list[MediaRecord] = []
        failures: list[MediaProbeFailure] = []
        for path in paths:
            relative = path.relative_to(self.bundle_path).as_posix()
            try:
                records.append(self._probe(path))
            except MediaLibraryError as error:
                failures.append(
                    MediaProbeFailure(
                        relative_path=relative,
                        message=error.message,
                    )
                )
        return MediaInventory(items=tuple(records), failures=tuple(failures))

    def _ignored(self, path: Path) -> bool:
        try:
            relative = path.resolve().relative_to(self.media_directory.resolve())
        except ValueError:
            # A file-level symlink may point outside the allowed Media root.
            # Keep it out of inventory just as resolve_media_path does.
            return True
        return any(part.startswith(".") for part in relative.parts) or path.name.endswith(".partial")

    def _probe(self, path: Path) -> MediaRecord:
        try:
            with av.open(str(path), mode="r") as container:
                video_streams = tuple(container.streams.video)
                audio_streams = tuple(container.streams.audio)
                width = video_streams[0].width if video_streams else None
                height = video_streams[0].height if video_streams else None
                frame_duration = _video_frame_duration(video_streams[0]) if video_streams else None
                duration = _container_duration(container)
        except Exception as error:
            raise MediaLibraryError(
                "media_probe_failed",
                f"Could not probe media file {path}: {error}",
                status=422,
            ) from error
        if not video_streams and not audio_streams:
            raise MediaLibraryError(
                "media_probe_failed",
                f"Media file has no audio or video streams: {path}",
                status=422,
            )
        return MediaRecord(
            relative_path=path.relative_to(self.bundle_path).as_posix(),
            filename=path.name,
            size_bytes=path.stat().st_size,
            duration=duration,
            width=width,
            height=height,
            frame_duration=frame_duration,
            has_video=bool(video_streams),
            has_audio=bool(audio_streams),
        )

    def _validate_destination_name(self, name: str) -> None:
        if (
            not name
            or Path(name).name != name
            or name in {".", ".."}
            or name.startswith(".")
            or name.endswith(".partial")
        ):
            raise MediaLibraryError(
                "invalid_request",
                "destinationName must be one visible filename and cannot end in .partial",
                status=400,
            )

    def _collision(self, target: Path) -> MediaLibraryError:
        return MediaLibraryError(
            "media_import_conflict",
            f"Media destination already exists: {target.name}",
            status=409,
        )

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.media_directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _video_frame_duration(stream: Any) -> Fraction | None:
    rate = stream.average_rate or stream.base_rate
    if rate is None or rate == 0:
        return None
    return 1 / Fraction(rate)


def _container_duration(container: Any) -> Fraction | None:
    if container.duration is not None:
        return Fraction(container.duration, av.time_base)
    durations = [
        Fraction(stream.duration * stream.time_base)
        for stream in container.streams
        if stream.duration is not None and stream.time_base is not None
    ]
    return max(durations, default=None)
