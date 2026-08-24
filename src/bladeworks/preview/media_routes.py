"""Standalone FastAPI router for explicit bundle-media operations.

The main server mounts this router after it opens one ``.fcpxmld``. Keeping the
router separate lets source/history and media lifecycles remain independent.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import anyio
from fastapi import APIRouter, Header, Request

from .contracts import PreviewAPIError
from .media_library import MediaLibrary, MediaLibraryError
from .media_visuals import MediaVisualRequest, sample_media


def create_media_router(library: MediaLibrary) -> APIRouter:
    router = APIRouter()

    @router.get("/api/editor/media")
    async def inventory() -> dict[str, object]:
        return await _call(library.inventory)

    @router.post("/api/editor/media/refresh")
    async def refresh() -> dict[str, object]:
        return await _call(library.refresh)

    @router.post("/api/editor/media/import", status_code=201)
    async def import_media(payload: dict[str, Any]) -> dict[str, object]:
        source_path = payload.get("sourcePath")
        destination_name = payload.get("destinationName")
        if not isinstance(source_path, str) or not source_path:
            raise PreviewAPIError(
                "invalid_request",
                "sourcePath is required and must be a non-empty string",
                status=400,
            )
        if not isinstance(destination_name, str) or not destination_name:
            raise PreviewAPIError(
                "invalid_request",
                "destinationName is required and must be a non-empty string",
                status=400,
            )
        try:
            record = await asyncio.to_thread(
                library.import_file,
                Path(source_path),
                destination_name,
            )
        except MediaLibraryError as error:
            raise _public_error(error) from error
        return {"media": record.to_json()}

    @router.post("/api/editor/media/upload", status_code=201)
    async def upload_media(
        request: Request,
        filename: str | None = Header(default=None, alias="X-Bladeworks-Filename"),
    ) -> dict[str, object]:
        """Stream a browser upload into a hidden stage before publication."""

        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/octet-stream":
            raise PreviewAPIError(
                "invalid_request",
                "Media upload requires Content-Type application/octet-stream.",
                status=400,
            )
        if not isinstance(filename, str) or not filename:
            raise PreviewAPIError(
                "invalid_request",
                "X-Bladeworks-Filename is required.",
                status=400,
            )
        filename = unquote(filename)
        try:
            stage = library.create_upload_stage(filename)
        except MediaLibraryError as error:
            raise _public_error(error) from error
        try:
            async with await anyio.open_file(stage, "xb") as output:
                async for chunk in request.stream():
                    if chunk:
                        await output.write(chunk)
                await output.flush()
            await asyncio.to_thread(_fsync_file, stage)
            record = await asyncio.to_thread(library.publish_upload, stage, filename)
        except MediaLibraryError as error:
            raise _public_error(error) from error
        finally:
            stage.unlink(missing_ok=True)
        return {"media": record.to_json()}

    @router.post("/api/editor/media/visuals")
    async def media_visuals(payload: dict[str, Any]) -> dict[str, object]:
        """Return transient thumbnails and audio peaks for one visible clip."""

        relative_path = payload.get("relativePath")
        if not isinstance(relative_path, str):
            raise PreviewAPIError("invalid_request", "relativePath is required.", status=400)
        try:
            request = MediaVisualRequest(
                start=_number(payload, "start", default=0.0),
                duration=_number(payload, "duration"),
                thumbnail_count=_integer(payload, "thumbnailCount", default=0),
                thumbnail_width=_integer(payload, "thumbnailWidth", default=96),
                audio_bands=_integer(payload, "audioBands", default=0),
            )
            path = library.resolve_media_path(relative_path)
            result = await asyncio.to_thread(sample_media, path, request)
            # Preserve the inventory identity supplied by the caller. ``path``
            # may be resolved through a symlink and its basename is not enough
            # to distinguish nested files with the same name.
            result["relativePath"] = relative_path
            return result
        except MediaLibraryError as error:
            raise _public_error(error) from error

    return router


async def _call(operation) -> dict[str, object]:
    try:
        value = await asyncio.to_thread(operation)
    except MediaLibraryError as error:
        raise _public_error(error) from error
    return value.to_json()


def _public_error(error: MediaLibraryError) -> PreviewAPIError:
    return PreviewAPIError(error.code, error.message, status=error.status)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        import os

        os.fsync(handle.fileno())


def _number(payload: dict[str, Any], name: str, *, default: float | None = None) -> float:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreviewAPIError("invalid_request", f"{name} must be a number.", status=400)
    return float(value)


def _integer(payload: dict[str, Any], name: str, *, default: int) -> int:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreviewAPIError("invalid_request", f"{name} must be an integer.", status=400)
    return value
