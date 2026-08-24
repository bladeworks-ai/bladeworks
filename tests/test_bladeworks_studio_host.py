"""Studio static hosting and the API-only/headless separation contract."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bladeworks.preview.studio import mount_studio, studio_asset_directory


def test_packaged_studio_assets_resolve_without_repository_relative_paths() -> None:
    assets = studio_asset_directory()

    assert assets.name == "studio_static"
    assert (assets / "index.html").is_file()
    assert (assets / "app.js").is_file()


def test_studio_mount_preserves_api_precedence_and_applies_security_headers() -> None:
    app = FastAPI()

    @app.get("/healthz")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    mount_studio(app)
    client = TestClient(app)

    health_response = client.get("/healthz")
    index = client.get("/")
    traversal = client.get("/%2e%2e/pyproject.toml")

    assert health_response.status_code == 200 and health_response.json() == {"ok": True}
    assert index.status_code == 200
    assert "Bladeworks Studio" in index.text
    assert traversal.status_code == 404
    for response in (health_response, index, traversal):
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["cross-origin-opener-policy"] == "same-origin"
        assert response.headers["cross-origin-embedder-policy"] == "require-corp"
        assert response.headers["cross-origin-resource-policy"] == "same-origin"
        assert "media-src 'self' blob:" in response.headers["content-security-policy"]
        assert "script-src 'self';" in response.headers["content-security-policy"]
        assert "style-src 'self' 'unsafe-inline'" in response.headers["content-security-policy"]


def test_api_only_app_has_no_studio_root_route() -> None:
    app = FastAPI()

    @app.get("/healthz")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 404
