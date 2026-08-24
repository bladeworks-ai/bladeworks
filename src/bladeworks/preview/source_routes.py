"""FastAPI routes for complete-document source mutation and session history.

Architecture map
================

    HTTP request
        -> parse transport headers/body only
        -> call ``OpenedSourceStore`` under its source lock
        -> expose exact disk and loaded versions in every response

This router is separate from preview and render routes so the filesystem
boundary can be implemented and tested without touching transport sessions.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, Response

from .contracts import PreviewAPIError
from .source import EDITOR_PROFILE, OpenedSourceStore, SourceStatus


def _status_payload(status: SourceStatus) -> dict[str, object]:
    payload = status.payload()
    payload["version"] = status.disk_version
    return payload


def _status_headers(status: SourceStatus) -> dict[str, str]:
    headers = {
        "X-Bladeworks-Editor-Profile": EDITOR_PROFILE,
        "X-Bladeworks-Disk-Version": status.disk_version,
        "X-Bladeworks-Compile-Status": status.compile_status,
        "X-Bladeworks-History-Index": str(status.history_index),
        "X-Bladeworks-History-Length": str(status.history_length),
    }
    if status.loaded_version is not None:
        headers["X-Bladeworks-Loaded-Version"] = status.loaded_version
    return headers


def _if_match(value: str | None) -> str:
    if value is None:
        raise PreviewAPIError(
            "source_version_conflict",
            "If-Match is required for source replacement.",
            status=409,
            retryable=True,
        )
    candidate = value.strip()
    if len(candidate) < 2 or candidate[0] != '"' or candidate[-1] != '"':
        raise PreviewAPIError(
            "source_version_conflict",
            "If-Match must contain one quoted Bladeworks source version.",
            status=409,
            retryable=True,
        )
    version = candidate[1:-1]
    if not version.startswith("sha256:") or len(version) != 71:
        raise PreviewAPIError(
            "source_version_conflict",
            "If-Match does not contain a valid Bladeworks source version.",
            status=409,
            retryable=True,
        )
    return version


def build_source_router(store: OpenedSourceStore) -> APIRouter:
    """Build the five single-source routes around one opened store.

    Main callers:
    - the production application factory, which includes this router once.
    """

    router = APIRouter(prefix="/api/editor/source", tags=["source"])

    @router.get("")
    async def get_source() -> Response:
        result = store.read_source()
        return Response(
            content=result.xml,
            media_type="application/xml",
            headers={
                "ETag": f'"{result.status.disk_version}"',
                **_status_headers(result.status),
            },
        )

    @router.put("")
    async def put_source(
        request: Request,
        if_match: str | None = Header(default=None, alias="If-Match"),
    ) -> JSONResponse:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"application/xml", "text/xml"}:
            raise PreviewAPIError(
                "source_invalid",
                "Source replacement requires Content-Type application/xml.",
                status=400,
            )
        status = await asyncio.to_thread(
            store.replace,
            await request.body(),
            expected_version=_if_match(if_match),
        )
        return JSONResponse(
            content=_status_payload(status),
            headers={"ETag": f'"{status.disk_version}"', **_status_headers(status)},
        )

    @router.post("/reload")
    async def reload_source() -> JSONResponse:
        status = store.reload()
        return JSONResponse(
            content=_status_payload(status),
            headers={"ETag": f'"{status.disk_version}"', **_status_headers(status)},
        )

    @router.post("/undo")
    async def undo_source() -> JSONResponse:
        status = store.undo()
        return JSONResponse(
            content=_status_payload(status),
            headers={"ETag": f'"{status.disk_version}"', **_status_headers(status)},
        )

    @router.post("/redo")
    async def redo_source() -> JSONResponse:
        status = store.redo()
        return JSONResponse(
            content=_status_payload(status),
            headers={"ETag": f'"{status.disk_version}"', **_status_headers(status)},
        )

    return router


def build_compatibility_router(store: OpenedSourceStore) -> APIRouter:
    """Expose compatibility for one Project in one exact library version."""

    router = APIRouter(tags=["compatibility"])

    @router.get("/api/editor/compatibility")
    async def get_compatibility(sourceVersion: str, projectRef: str) -> dict[str, object]:
        return store.compatibility(sourceVersion, projectRef)

    return router
