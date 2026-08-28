"""Foreground Uvicorn lifecycle for one opened ``.fcpxmld`` bundle.

Architecture map
================

    fcpxml server run PATH
        -> create disposable instance directory
        -> open, hash, and compile Info.fcpxml
        -> bind one loopback socket, including port 0 selection
        -> print one machine-readable ready record
        -> Uvicorn serves until SIGINT/SIGTERM
        -> app lifespan cancels preview/render work
        -> remove disposable instance state

The runner owns process lifecycle only. Source and media routers are mounted
by the final application assembly, keeping their filesystem behavior in their
dedicated modules.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .application import create_local_preview_app
from .capabilities import create_capability_router
from .contracts import PreviewAPIError
from .media_library import MediaLibrary
from .media_routes import create_media_router
from .security import new_auth_token, validate_loopback_origin
from .source import EDITOR_PROFILE, OpenedSourceStore
from .source_routes import build_compatibility_router, build_source_router


@dataclass(frozen=True)
class ServerConfig:
    source: Path
    host: str = "127.0.0.1"
    port: int = 8765
    device: str = "auto"
    decoder_threads: int = 2
    history_limit: int = 50
    render_directory: Path | None = None
    log_level: str = "info"
    strict: bool = False
    allowed_origins: tuple[str, ...] = ()


def _selected_device(requested: str) -> str:
    if requested != "auto":
        from ..tensor.renderer import require_torch_device

        require_torch_device(requested)
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _listener(host: str, port: int) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((host, port))
        listener.listen(2048)
    except BaseException:
        listener.close()
        raise
    return listener


def run_server(config: ServerConfig) -> int:
    """Run the API-only server without importing or launching a browser.

    This is the hard headless entry point used by ``fcpxml server run``.
    Studio hosting and browser launch are reachable only through
    :func:`run_studio`.
    """

    return _run_server(config, mode="server", open_browser=False)


def run_studio(config: ServerConfig, *, open_browser: bool = True) -> int:
    """Run the API and packaged editor in one foreground Uvicorn process."""

    return _run_server(config, mode="studio", open_browser=open_browser)


def _run_server(config: ServerConfig, *, mode: str, open_browser: bool) -> int:
    """Run one local process and clean its disposable state.

    Main callers:
    - ``run_server`` for API-only/headless use.
    - ``run_studio`` for same-origin packaged editor use.

    Why this exists:
    Uvicorn's convenience function does not expose the selected port when
    binding port zero. Owning the socket lets the launcher receive the exact
    URL without scraping human log output.
    """

    if mode not in {"server", "studio"}:
        raise ValueError(f"Unknown Bladeworks server mode: {mode}")
    if config.host != "127.0.0.1":
        raise ValueError("Bladeworks server initially binds to 127.0.0.1 only.")
    if not 0 <= config.port <= 65535:
        raise ValueError("Server port must be between 0 and 65535.")
    origins = tuple(validate_loopback_origin(value) for value in config.allowed_origins)
    instance_root = Path(tempfile.gettempdir()) / "bladeworks"
    instance_root.mkdir(parents=True, exist_ok=True)
    instance_directory = Path(tempfile.mkdtemp(prefix="instance-", dir=instance_root))
    instance_id = instance_directory.name.removeprefix("instance-")
    render_directory = (
        config.render_directory.expanduser().resolve()
        if config.render_directory is not None
        else instance_directory / "renders"
    )
    listener: socket.socket | None = None
    service = None
    renders = None
    try:
        store = OpenedSourceStore.open(
            config.source,
            history_directory=instance_directory / "history",
            history_limit=config.history_limit,
            strict=config.strict,
        )
        device = _selected_device(config.device)
        token = new_auth_token()
        app_holder: dict[str, object] = {}

        def readiness() -> dict[str, object]:
            service = app_holder.get("service")
            renders = app_holder.get("renders")
            try:
                status = store.status()
                source_payload = status.payload()
                ready = status.compile_status == "ready"
            except PreviewAPIError as error:
                source_payload = {
                    "diskVersion": None,
                    "loadedVersion": None,
                    "compileStatus": error.code,
                    "degraded": False,
                    "editorProfile": EDITOR_PROFILE,
                    "error": error.message,
                }
                ready = False
            return {
                "ready": ready,
                "mode": mode,
                "instanceId": instance_id,
                "sourcePath": str(store.bundle_path),
                **source_payload,
                "device": device,
                "activeSessions": getattr(service, "active_count", 0),
                "activeRenders": getattr(renders, "active_count", 0),
            }

        app = create_local_preview_app(
            documents=store,
            report_for=store.report_for,
            artifact_directory=render_directory,
            device=device,
            decoder_threads=config.decoder_threads,
            encoder_preset="veryfast",
            auth_token=token,
            allowed_origins=origins,
            readiness=readiness,
            protected_routers=(
                create_capability_router(),
                build_source_router(store),
                build_compatibility_router(store),
                create_media_router(MediaLibrary(store.bundle_path)),
            ),
        )
        if mode == "studio":
            from .studio import mount_studio

            mount_studio(app)
        service = app.state.preview_service
        renders = app.state.render_service
        app_holder["service"] = service
        app_holder["renders"] = renders

        listener = _listener(config.host, config.port)
        selected_port = listener.getsockname()[1]
        url = f"http://{config.host}:{selected_port}"
        status = store.status()
        ready_payload: dict[str, object] = {
            "event": "ready",
            "instanceId": instance_id,
            "url": url,
            "sourcePath": str(store.bundle_path),
            "version": status.disk_version,
            "editorProfile": EDITOR_PROFILE,
            "authToken": token,
        }
        studio_url: str | None = None
        if mode == "studio":
            studio_url = f"{url}/?runtime=localhost#runtimeToken={token}"
            ready_payload["mode"] = "studio"
            ready_payload["studioUrl"] = studio_url
        print(
            json.dumps(ready_payload, separators=(",", ":")),
            flush=True,
        )
        if studio_url is not None and open_browser:
            _launch_browser(studio_url)

        import uvicorn

        shutdown_started = threading.Event()

        class ForegroundServer(uvicorn.Server):
            """Wake preview streams as soon as a foreground signal arrives.

            Uvicorn waits for open HTTP responses before running application
            lifespan cleanup. An SSE response is intentionally unbounded, so
            waiting for lifespan first creates a shutdown cycle. Begin the
            idempotent preview cleanup on a helper thread, which closes every
            session event stream and lets Uvicorn proceed normally.
            """

            def handle_exit(self, sig, frame) -> None:
                if not self.should_exit and not shutdown_started.is_set():
                    shutdown_started.set()
                    threading.Thread(
                        target=service.shutdown,
                        name="bladeworks-preview-shutdown",
                        daemon=True,
                    ).start()
                super().handle_exit(sig, frame)

        server = ForegroundServer(
            uvicorn.Config(
                app,
                host=config.host,
                port=selected_port,
                log_level=config.log_level,
                workers=1,
                reload=False,
                # Infinite SSE responses must not prevent a foreground signal
                # from reaching lifespan cleanup. Uvicorn cancels any client
                # connection still open after this bounded grace period, then
                # the application closes its preview sessions and WebRTC peers.
                timeout_graceful_shutdown=5,
            )
        )
        try:
            server.run(sockets=[listener])
        except KeyboardInterrupt:
            # Some event-loop implementations re-raise after Uvicorn has
            # completed lifespan shutdown. A foreground Ctrl-C is a normal
            # stop, not a Bladeworks error or traceback.
            pass
        return 0
    finally:
        # Lifespan normally owns this cleanup. Repeat it here because a second
        # signal or event-loop cancellation can interrupt ASGI shutdown. Both
        # services are idempotent, so the fallback cannot double-close work.
        if service is not None:
            service.shutdown()
        if renders is not None:
            renders.shutdown()
        if listener is not None:
            listener.close()
        shutil.rmtree(instance_directory, ignore_errors=True)


_CHROME_APP_BINARIES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
)


def _launch_chrome_app_window(url: str) -> bool:
    """Open Studio as a Chromium app window so Cmd-N is not a New Window chord.

    A normal Chrome tab never delivers a cancelable Cmd-N to the page. App
    windows only reserve Close and Exit, so Studio's capture-phase keydown
    can own New Project.

    Main callers: ``_launch_browser`` when ``fcpxml studio`` opens a UI.
    """

    for binary in _CHROME_APP_BINARIES:
        if not binary.is_file():
            continue
        try:
            subprocess.Popen(
                [str(binary), f"--app={url}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except OSError:
            continue
    return False


def _launch_browser(url: str) -> None:
    """Open Studio without putting the runtime token in a warning line."""

    if _launch_chrome_app_window(url):
        return
    import webbrowser

    if not webbrowser.open(url):
        print(
            "warning: could not open a browser automatically; use studioUrl from the ready record",
            file=sys.stderr,
        )


def check_health(url: str, *, timeout: float = 2.0) -> int:
    """Check liveness and readiness, printing readiness JSON on success."""

    base = url.rstrip("/")
    try:
        with urlopen(Request(f"{base}/healthz", method="GET"), timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"healthz returned HTTP {response.status}")
        with urlopen(Request(f"{base}/readyz", method="GET"), timeout=timeout) as response:
            payload = json.loads(response.read())
            if response.status != 200 or payload.get("ready") is not True:
                raise RuntimeError("server is not ready")
    except (HTTPError, URLError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: Bladeworks server health check failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
