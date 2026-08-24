"""Package-resource hosting for the optional Bladeworks Studio UI.

Architecture map
================

    ``bladeworks studio`` only
        -> resolve committed frontend assets with ``importlib.resources``
        -> mount them after every API and health route
        -> add browser-isolation and no-cache response headers

The API-only ``server run`` path never imports or calls this module. That is a
hard headless boundary, not a runtime option inferred from browser state.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles


STUDIO_PACKAGE = "bladeworks.studio_static"
_CSP = (
    "default-src 'self'; "
    "base-uri 'none'; "
    "connect-src 'self'; "
    "font-src 'self'; "
    "frame-ancestors 'none'; "
    "img-src 'self' data: blob:; "
    "media-src 'self' blob:; "
    "object-src 'none'; "
    "script-src 'self'; "
    # The editor positions timeline clips and viewer overlays with generated
    # ``style`` attributes. Keep scripts restricted to packaged files, while
    # permitting the inline CSS required for those calculated dimensions.
    "style-src 'self' 'unsafe-inline'"
)


def studio_asset_directory() -> Path:
    """Return the installed Studio asset tree or fail with an actionable error."""

    try:
        root = resources.files(STUDIO_PACKAGE)
    except (ModuleNotFoundError, TypeError) as error:
        raise RuntimeError(
            "Bladeworks Studio assets are not installed. Reinstall a wheel that includes studio_static."
        ) from error
    path = Path(str(root)).resolve()
    if not path.is_dir() or not (path / "index.html").is_file():
        raise RuntimeError(
            "Bladeworks Studio assets are incomplete: packaged index.html is missing."
        )
    return path


def mount_studio(app: FastAPI) -> None:
    """Mount the packaged SPA after the app's concrete API routes."""

    assets = studio_asset_directory()

    @app.middleware("http")
    async def studio_security_headers(request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = _CSP
        return response

    # Mount last. Starlette matches routes in declaration order, so concrete
    # API/health routes retain precedence over this root catch-all.
    app.mount("/", StaticFiles(directory=assets, html=True), name="studio")
