"""FastAPI adapter for source-version preview commands, renders, and SSE.

Renderer work runs in worker threads so seek, scan shutdown, and export setup
cannot freeze the HTTP event loop. Source and media routers are mounted by the
production assembly and remain separate from this transport module.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from typing import Any, Callable, Mapping, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from ..tensor.errors import TensorRenderError, TensorRenderUnsupported
from .contracts import PreviewAPIError, SessionDescription, SessionQuality, fraction_from_seconds
from .rawframe import CLOSED, EMPTY, RawFrameMediaSink, RawFramePayload
from .render_jobs import RenderJobService
from .security import BearerTokenAuth
from .service import PreviewService, validate_quality


# The browser cannot set an Authorization header on a WebSocket handshake, so
# the bearer token rides in the ``Sec-WebSocket-Protocol`` list alongside this
# marker subprotocol. token_urlsafe values are valid subprotocol tokens.
_WS_SUBPROTOCOL = "bladeworks-preview"


def _required(payload: dict[str, Any], name: str, expected_type: type) -> Any:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, expected_type):
        raise PreviewAPIError("invalid_request", f"{name} is required and must be {expected_type.__name__}", status=400)
    return value


def _offer(payload: dict[str, Any]) -> SessionDescription:
    value = payload.get("offer")
    if not isinstance(value, dict):
        raise PreviewAPIError("invalid_request", "offer is required", status=400)
    offer_type = _required(value, "type", str)
    sdp = _required(value, "sdp", str)
    if offer_type != "offer":
        raise PreviewAPIError("invalid_request", "offer.type must be 'offer'", status=400)
    return SessionDescription(type=offer_type, sdp=sdp)


def _quality(payload: dict[str, Any]) -> SessionQuality:
    """Parse one fixed preview quality without accepting obsolete fallback fields."""

    value = payload.get("quality")
    if value is not None and not isinstance(value, dict):
        raise PreviewAPIError("invalid_request", "quality must be an object", status=400)
    value = value or {}
    unknown = sorted(set(value) - {"resolution"})
    if unknown:
        raise PreviewAPIError(
            "invalid_request",
            f"Unsupported quality fields: {', '.join(unknown)}",
            status=400,
        )
    resolution = value.get("resolution")
    if resolution is not None and not isinstance(resolution, str):
        raise PreviewAPIError("invalid_resolution", "quality.resolution must be a string", status=400)
    return validate_quality(resolution)


def create_app(
    service: PreviewService,
    *,
    renders: RenderJobService | None = None,
    auth_token: str | None = None,
    allowed_origins: tuple[str, ...] = (),
    readiness: Callable[[], Mapping[str, object]] | None = None,
    protected_routers: tuple[APIRouter, ...] = (),
) -> FastAPI:
    """Create the transport app around already-constructed service objects.

    Main callers:
    - ``application.create_local_preview_app`` for the production server.
    - focused tests that inject fake render, media, and source adapters.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await asyncio.to_thread(service.shutdown)
        if renders is not None:
            await asyncio.to_thread(renders.shutdown)

    app = FastAPI(title="Bladeworks Local FCPXML API", version="1", lifespan=lifespan)
    app.state.preview_service = service
    app.state.render_service = renders
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=[
                "Accept",
                "Authorization",
                "Content-Type",
                "If-Match",
                "Last-Event-ID",
                "X-Bladeworks-Filename",
            ],
            expose_headers=[
                "ETag",
                "X-Bladeworks-Compile-Status",
                "X-Bladeworks-Disk-Version",
                "X-Bladeworks-Editor-Profile",
                "X-Bladeworks-History-Index",
                "X-Bladeworks-History-Length",
                "X-Bladeworks-Loaded-Version",
            ],
        )

    auth = BearerTokenAuth(auth_token) if auth_token is not None else None

    def protected() -> tuple[Any, ...]:
        return (Depends(auth),) if auth is not None else ()

    for router in protected_routers:
        app.include_router(router, dependencies=list(protected()))

    @app.exception_handler(PreviewAPIError)
    async def preview_error(_request: Request, error: PreviewAPIError) -> JSONResponse:
        return JSONResponse(status_code=error.status, content=error.body())

    @app.exception_handler(RequestValidationError)
    async def request_error(_request: Request, _error: RequestValidationError) -> JSONResponse:
        public = PreviewAPIError("invalid_request", "Request JSON does not match the API contract.", status=400)
        return JSONResponse(status_code=public.status, content=public.body())

    @app.exception_handler(TensorRenderUnsupported)
    async def unsupported_error(_request: Request, error: TensorRenderUnsupported) -> JSONResponse:
        public = PreviewAPIError("unsupported_construct", str(error), status=422)
        return JSONResponse(status_code=public.status, content=public.body())

    @app.exception_handler(TensorRenderError)
    async def renderer_error(_request: Request, error: TensorRenderError) -> JSONResponse:
        public = PreviewAPIError("preview_failed", str(error), status=500)
        return JSONResponse(status_code=public.status, content=public.body())

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, _error: Exception) -> JSONResponse:
        public = PreviewAPIError("preview_failed", "The local preview service encountered an unexpected error.", status=500)
        return JSONResponse(status_code=public.status, content=public.body())

    @app.get("/healthz")
    async def health() -> dict[str, object]:
        return {"ok": True}

    @app.get("/readyz")
    async def ready() -> JSONResponse:
        payload = dict(readiness()) if readiness is not None else {"ready": True}
        return JSONResponse(status_code=200 if payload.get("ready") is True else 503, content=payload)

    @app.post("/api/editor/preview/sessions", status_code=201, dependencies=list(protected()))
    async def create_session(payload: dict[str, Any]) -> dict[str, object]:
        source_version = _required(payload, "sourceVersion", str)
        project_ref = _required(payload, "projectRef", str)
        playhead = fraction_from_seconds(payload.get("playhead", 0), field="playhead")
        session, answer = await asyncio.to_thread(
            service.create_session,
            source_version=source_version,
            project_ref=project_ref,
            playhead=playhead,
            offer=_offer(payload),
            quality=_quality(payload),
        )
        return {
            "sessionId": session.session_id,
            "answer": {"type": answer.type, "sdp": answer.sdp},
            "eventsUrl": f"/api/editor/preview/sessions/{session.session_id}/events",
            **session.state_payload(),
        }

    @app.post("/api/editor/preview/sessions/raw", status_code=201, dependencies=list(protected()))
    async def create_raw_session(payload: dict[str, Any]) -> dict[str, object]:
        """Create a raw-frame preview session (the main preview transport).

        Returns the session id plus the WebSocket URL the browser opens to
        receive uncompressed frames. There is no SDP answer — see
        ``PreviewService.create_raw_session``.
        """

        source_version = _required(payload, "sourceVersion", str)
        project_ref = _required(payload, "projectRef", str)
        playhead = fraction_from_seconds(payload.get("playhead", 0), field="playhead")
        session = await asyncio.to_thread(
            service.create_raw_session,
            source_version=source_version,
            project_ref=project_ref,
            playhead=playhead,
            quality=_quality(payload),
        )
        return {
            "sessionId": session.session_id,
            "streamUrl": f"/api/editor/preview/sessions/{session.session_id}/stream",
            "eventsUrl": f"/api/editor/preview/sessions/{session.session_id}/events",
            **session.state_payload(),
        }

    @app.websocket("/api/editor/preview/sessions/{session_id}/stream")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        """Drain one raw-frame session's queue to the browser.

        Two cooperating tasks: the sender pulls frames off the sink's
        thread-safe queue (through a worker thread so the blocking get never
        stalls the event loop) and writes them as binary WebSocket messages;
        the receiver just watches for the client hanging up. Either one ending
        tears the pair down.
        """

        offered = list(websocket.scope.get("subprotocols", ()))
        if auth is not None and (auth_token is None or auth_token not in offered):
            await websocket.close(code=1008)
            return
        try:
            session = service.session(session_id)
        except PreviewAPIError:
            await websocket.close(code=1011)
            return
        sink = session.media_sink
        if not isinstance(sink, RawFrameMediaSink):
            # Session exists but was created for the WebRTC transport.
            await websocket.close(code=1011)
            return

        accept_kwargs = {"subprotocol": _WS_SUBPROTOCOL} if auth is not None else {}
        await websocket.accept(**accept_kwargs)
        await websocket.send_json(
            {**session.state_payload(), "type": "meta"}
        )

        stop = asyncio.Event()

        async def receiver() -> None:
            try:
                while not stop.is_set():
                    await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                stop.set()

        async def sender() -> None:
            try:
                while not stop.is_set():
                    payload = await asyncio.to_thread(sink.get, 0.25)
                    if payload is CLOSED:
                        break
                    if payload is EMPTY:
                        continue
                    if not isinstance(payload, RawFramePayload) or not sink.is_current(payload):
                        continue
                    await websocket.send_bytes(payload.data)
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                stop.set()

        receive_task = asyncio.create_task(receiver())
        send_task = asyncio.create_task(sender())
        try:
            await asyncio.wait({receive_task, send_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop.set()
            for task in (receive_task, send_task):
                task.cancel()
            await asyncio.gather(receive_task, send_task, return_exceptions=True)
            # The raw-frame transport binds one session to exactly one WebSocket
            # and has no reconnect path: once this socket is gone the session can
            # never be resumed. Tear it down here so a crashed or closed tab does
            # not strand its producer, decoder, and scan thread until the whole
            # server shuts down. ``service.close`` is idempotent (a later DELETE
            # is a no-op) and joins the scan thread, so it runs off the event
            # loop via a worker thread.
            await asyncio.to_thread(service.close, session_id)

    @app.post("/api/editor/preview/sessions/{session_id}/sync", dependencies=list(protected()))
    async def sync(session_id: str, payload: dict[str, Any]) -> dict[str, object]:
        return await asyncio.to_thread(
            service.sync,
            session_id,
            source_version=_required(payload, "sourceVersion", str),
            project_ref=_required(payload, "projectRef", str),
            playhead=fraction_from_seconds(payload.get("playhead", 0), field="playhead"),
            quality=_quality(payload) if "quality" in payload else None,
        )

    @app.post("/api/editor/preview/sessions/{session_id}/seek", dependencies=list(protected()))
    async def seek(session_id: str, payload: dict[str, Any]) -> dict[str, object]:
        return await asyncio.to_thread(
            service.session(session_id).seek,
            requested_time=fraction_from_seconds(payload.get("time"), field="time"),
            request_id=_required(payload, "requestId", str),
        )

    @app.post("/api/editor/preview/sessions/{session_id}/play", status_code=202, dependencies=list(protected()))
    async def play(session_id: str, payload: Optional[dict[str, Any]] = None) -> dict[str, object]:
        payload = payload or {}
        requested = fraction_from_seconds(payload["time"], field="time") if "time" in payload else None
        return await asyncio.to_thread(service.session(session_id).play, requested_time=requested)

    @app.post("/api/editor/preview/sessions/{session_id}/pause", dependencies=list(protected()))
    async def pause(session_id: str) -> dict[str, object]:
        return await asyncio.to_thread(service.session(session_id).pause)

    @app.get("/api/editor/preview/sessions/{session_id}/events", dependencies=list(protected()))
    async def events(
        session_id: str,
        last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        try:
            after_id = int(last_event_id) if last_event_id is not None else 0
        except ValueError as error:
            raise PreviewAPIError("invalid_request", "Last-Event-ID must be an integer", status=400) from error
        session = service.session(session_id)

        def body():
            for record in session.events.subscribe(after_id=after_id):
                data = json.dumps(record.data, separators=(",", ":"))
                yield f"id: {record.id}\nevent: {record.event}\ndata: {data}\n\n"

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/api/editor/preview/sessions/{session_id}", status_code=204, dependencies=list(protected()))
    async def close(session_id: str) -> Response:
        await asyncio.to_thread(service.close, session_id)
        return Response(status_code=204)

    if renders is not None:

        @app.post("/api/editor/render", status_code=202, dependencies=list(protected()))
        async def start_render(payload: dict[str, Any]) -> dict[str, object]:
            job = await asyncio.to_thread(
                renders.start,
                source_version=_required(payload, "sourceVersion", str),
                project_ref=_required(payload, "projectRef", str),
                profile=payload.get("resolution"),
                export_profile=payload.get("profile"),
            )
            return job.payload(status_override="queued")

        @app.get("/api/editor/renders/{job_id}", dependencies=list(protected()))
        async def get_render(job_id: str) -> dict[str, object]:
            return renders.get(job_id).payload()

        @app.delete("/api/editor/renders/{job_id}", status_code=202, dependencies=list(protected()))
        async def cancel_render(job_id: str) -> dict[str, object]:
            return renders.cancel(job_id).payload()

        @app.get("/api/editor/renders/{job_id}/artifact")
        async def render_artifact(
            request: Request, job_id: str, token: str | None = None
        ) -> FileResponse:
            bearer_authorized = False
            if auth is not None:
                supplied = request.headers.get("Authorization") or ""
                bearer_authorized = hmac.compare_digest(
                    supplied, f"Bearer {auth.token}"
                )
            job = renders.get_artifact(
                job_id, token, bearer_authorized=bearer_authorized
            )
            if job.status != "completed":
                raise PreviewAPIError("render_failed", "Render artifact is not complete.", status=409)
            return FileResponse(
                job.output_path,
                media_type=job.export_profile.content_type,
                filename=job.output_path.name,
            )

    return app
