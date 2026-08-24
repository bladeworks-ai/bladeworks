"""Thin HTTP contract tests for the complete FCPXML source router."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from bladeworks.preview.contracts import PreviewAPIError
from bladeworks.preview.source import EDITOR_PROFILE, OpenedSourceStore, source_version
from bladeworks.preview.source_routes import (
    build_compatibility_router,
    build_source_router,
)
from bladeworks.preview import source_routes


def _xml(name: str) -> bytes:
    return f'''<?xml version="1.0"?><fcpxml version="1.14">
<resources><format id="fmt" frameDuration="1/30s" width="160" height="90" colorSpace="1-1-1 (Rec. 709)"/></resources>
<library><event name="Event"><project name="{name}"><sequence format="fmt" duration="1s"><spine>
<gap offset="0s" duration="1s"/>
</spine></sequence></project></event></library></fcpxml>'''.encode()


def _client(tmp_path: Path) -> tuple[TestClient, OpenedSourceStore, bytes]:
    initial = _xml("initial")
    bundle = tmp_path / "Project.fcpxmld"
    bundle.mkdir()
    (bundle / "Info.fcpxml").write_bytes(initial)
    store = OpenedSourceStore.open(bundle, history_directory=tmp_path / "history")
    app = FastAPI()

    @app.exception_handler(PreviewAPIError)
    async def public_error(_request: Request, error: PreviewAPIError) -> JSONResponse:
        return JSONResponse(status_code=error.status, content=error.body())

    app.include_router(build_source_router(store))
    app.include_router(build_compatibility_router(store))
    return TestClient(app), store, initial


def test_get_returns_exact_xml_etag_profile_and_source_versions(tmp_path: Path) -> None:
    client, _store, initial = _client(tmp_path)

    response = client.get("/api/editor/source")

    assert response.status_code == 200
    assert response.content == initial
    assert response.headers["content-type"].startswith("application/xml")
    assert response.headers["etag"] == f'"{source_version(initial)}"'
    assert response.headers["x-bladeworks-editor-profile"] == EDITOR_PROFILE
    assert response.headers["x-bladeworks-disk-version"] == source_version(initial)
    assert response.headers["x-bladeworks-loaded-version"] == source_version(initial)


def test_put_requires_xml_and_quoted_if_match_then_returns_status(tmp_path: Path) -> None:
    client, store, initial = _client(tmp_path)
    updated = _xml("updated")

    missing = client.put(
        "/api/editor/source",
        content=updated,
        headers={"Content-Type": "application/xml"},
    )
    assert missing.status_code == 409
    assert missing.json()["error"]["code"] == "source_version_conflict"

    unquoted = client.put(
        "/api/editor/source",
        content=updated,
        headers={"Content-Type": "application/xml", "If-Match": source_version(initial)},
    )
    assert unquoted.status_code == 409

    wrong_type = client.put(
        "/api/editor/source",
        content=updated,
        headers={"Content-Type": "application/json", "If-Match": f'"{source_version(initial)}"'},
    )
    assert wrong_type.status_code == 400
    assert store.source_path.read_bytes() == initial

    response = client.put(
        "/api/editor/source",
        content=updated,
        headers={"Content-Type": "application/xml", "If-Match": f'"{source_version(initial)}"'},
    )
    assert response.status_code == 200
    assert response.json()["version"] == source_version(updated)
    assert response.json()["diskVersion"] == source_version(updated)
    assert response.json()["loadedVersion"] == source_version(updated)
    assert response.json()["historyIndex"] == 1
    assert response.headers["etag"] == f'"{source_version(updated)}"'


def test_put_runs_source_compilation_off_the_async_event_loop(tmp_path: Path, monkeypatch) -> None:
    client, _store, initial = _client(tmp_path)
    updated = _xml("threaded")
    calls: list[str] = []

    async def run_in_test_thread(function, *args, **kwargs):
        calls.append(function.__name__)
        return function(*args, **kwargs)

    monkeypatch.setattr(source_routes.asyncio, "to_thread", run_in_test_thread)
    response = client.put(
        "/api/editor/source",
        content=updated,
        headers={"Content-Type": "application/xml", "If-Match": f'"{source_version(initial)}"'},
    )

    assert response.status_code == 200
    assert calls == ["replace"]


def test_reload_undo_and_redo_routes_use_latest_disk_bytes(tmp_path: Path) -> None:
    client, store, initial = _client(tmp_path)
    updated = _xml("updated")
    external = _xml("external")
    put = client.put(
        "/api/editor/source",
        content=updated,
        headers={"Content-Type": "application/xml", "If-Match": f'"{source_version(initial)}"'},
    )
    assert put.status_code == 200

    store.source_path.write_bytes(external)
    reload_response = client.post("/api/editor/source/reload")
    assert reload_response.status_code == 200
    assert reload_response.json()["version"] == source_version(external)
    assert reload_response.json()["historyLength"] == 2

    undo = client.post("/api/editor/source/undo")
    assert undo.status_code == 200
    assert undo.json()["version"] == source_version(updated)
    assert undo.json()["historyIndex"] == 1

    undo_again = client.post("/api/editor/source/undo")
    assert undo_again.status_code == 200
    assert undo_again.json()["version"] == source_version(initial)
    redo = client.post("/api/editor/source/redo")
    assert redo.status_code == 200
    assert redo.json()["version"] == source_version(updated)


def test_get_exposes_invalid_disk_and_last_loaded_versions_without_hiding_bytes(tmp_path: Path) -> None:
    client, store, initial = _client(tmp_path)
    malformed = b"<fcpxml>"
    store.source_path.write_bytes(malformed)

    response = client.get("/api/editor/source")

    assert response.status_code == 200
    assert response.content == malformed
    assert response.headers["x-bladeworks-disk-version"] == source_version(malformed)
    assert response.headers["x-bladeworks-loaded-version"] == source_version(initial)
    assert response.headers["x-bladeworks-compile-status"] == "source_invalid"
    reload_response = client.post("/api/editor/source/reload")
    assert reload_response.status_code == 422
    assert reload_response.json()["error"]["code"] == "source_invalid"


def test_history_boundary_uses_stable_public_error(tmp_path: Path) -> None:
    client, _store, _initial = _client(tmp_path)

    undo = client.post("/api/editor/source/undo")
    redo = client.post("/api/editor/source/redo")

    assert undo.status_code == 409
    assert undo.json()["error"]["code"] == "history_at_start"
    assert redo.status_code == 409
    assert redo.json()["error"]["code"] == "history_at_end"


def test_compatibility_route_is_scoped_to_source_and_project(tmp_path: Path) -> None:
    client, _store, initial = _client(tmp_path)
    project_ref = "library[1]/event[1]/project[1]"

    response = client.get(
        "/api/editor/compatibility",
        params={"sourceVersion": source_version(initial), "projectRef": project_ref},
    )

    assert response.status_code == 200
    assert response.json()["sourceVersion"] == source_version(initial)
    assert response.json()["projectRef"] == project_ref
    assert response.json()["compatibility"]["project_name"] == "initial"
