"""Compose the production local preview application from narrow adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping

from fastapi import APIRouter, FastAPI

from ..core.model import RenderDocument
from ..core.report import CompatibilityReport
from .audio import FFmpegPreviewAudioFactory
from .contracts import SourceDocumentProvider
from .export import TensorExecutorExportRunner
from .producer import TensorFrameProducerFactory
from .rawframe import RawFrameMediaFactory
from .render_jobs import RenderJobService
from .routes import create_app
from .service import PreviewService


def _webrtc_enabled() -> bool:
    """Whether to build the quarantined WebRTC transport alongside raw frames.

    The raw-frame WebSocket path is the default preview transport. WebRTC is
    kept for regression comparison but only constructed when explicitly asked,
    so the common path never imports aiortc.
    """

    return os.environ.get("BLADEFRAME_PREVIEW_WEBRTC", "").strip().lower() in {"1", "true", "yes", "on"}


def create_local_preview_app(
    *,
    documents: SourceDocumentProvider,
    report_for: Callable[[RenderDocument], CompatibilityReport],
    artifact_directory: Path,
    device: str | None = None,
    decoder_threads: int = 2,
    encoder_preset: str | None = None,
    auth_token: str | None = None,
    allowed_origins: tuple[str, ...] = (),
    readiness: Callable[[], Mapping[str, object]] | None = None,
    protected_routers: tuple[APIRouter, ...] = (),
) -> FastAPI:
    """Return the local API around one opened FCPXML source.

    Main callers:
    - ``preview.runner`` after it opens and compiles one ``.fcpxmld`` bundle.
    """

    webrtc_media = None
    if _webrtc_enabled():
        from .webrtc import AiortcMediaFactory

        webrtc_media = AiortcMediaFactory()
    service = PreviewService(
        documents=documents,
        producers=TensorFrameProducerFactory(
            device=device,
            decoder_threads=decoder_threads,
        ),
        raw_media=RawFrameMediaFactory(),
        media=webrtc_media,
        audio=FFmpegPreviewAudioFactory(report_for=report_for),
    )
    renders = RenderJobService(
        documents=documents,
        runner=TensorExecutorExportRunner(
            report_for=report_for,
            encoder_preset=encoder_preset,
            device=device,
        ),
        artifact_directory=artifact_directory,
    )
    return create_app(
        service,
        renders=renders,
        auth_token=auth_token,
        allowed_origins=allowed_origins,
        readiness=readiness,
        protected_routers=protected_routers,
    )
