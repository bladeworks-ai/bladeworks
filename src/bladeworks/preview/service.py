"""Session registry and command orchestration for local Bladeworks preview."""

from __future__ import annotations

import threading
import uuid
from fractions import Fraction
from ..tensor.resolution import RenderMode, profile_for_mode
from .contracts import (
    FrameProducerFactory,
    PreviewAPIError,
    PreviewMediaFactory,
    PreviewAudioProducerFactory,
    SourceDocumentProvider,
    SessionDescription,
    SessionQuality,
)
from .media_reporting import missing_media_event_data
from .session import PreviewSession


class PreviewService:
    """Own preview sessions but not projects, Final Cut libraries, or HTTP."""

    def __init__(
        self,
        *,
        documents: SourceDocumentProvider,
        producers: FrameProducerFactory,
        raw_media=None,
        media: PreviewMediaFactory | None = None,
        audio: PreviewAudioProducerFactory | None = None,
    ) -> None:
        self.documents = documents
        self.producers = producers
        # ``raw_media`` is the main preview transport (raw frames over a
        # WebSocket). ``media`` is the quarantined WebRTC factory, kept for the
        # legacy ``/sessions`` route but no longer the default path.
        self.raw_media = raw_media
        self.media = media
        self.audio = audio
        self._lock = threading.Lock()
        self._sessions: dict[str, PreviewSession] = {}

    def _require_audio_support(self, document) -> None:
        has_audible_audio = (
            document.audio is not None
            and any(item.audible for item in document.audio.items)
        )
        if has_audible_audio and self.audio is None:
            raise PreviewAPIError(
                "preview_audio_unavailable",
                "This project has audible timeline audio, but no live audio producer is configured.",
                status=503,
            )

    def create_raw_session(
        self,
        *,
        source_version: str,
        project_ref: str,
        playhead: Fraction,
        quality: SessionQuality,
    ) -> PreviewSession:
        """Create a session whose media flows as raw frames over a WebSocket.

        Why this exists:
        The raw-frame transport has no SDP offer/answer handshake — the browser
        just opens a WebSocket after this returns. So unlike ``create_session``
        there is nothing to negotiate and nothing to return but the session
        itself. All session behaviour (seek, paced scan, generation guarding)
        is identical; only the sink differs.

        Main callers:
        - the ``POST /api/editor/preview/sessions/raw`` route.
        """

        if self.raw_media is None:
            raise PreviewAPIError(
                "unsupported_construct",
                "The raw-frame preview transport is not enabled on this server.",
                status=422,
            )
        loaded = self.documents.require_current(source_version, project_ref)
        document = loaded.document
        self._require_audio_support(document)
        sink = self.raw_media.open()
        session_id = f"preview-{uuid.uuid4().hex}"
        try:
            session = PreviewSession(
                session_id=session_id,
                source_version=loaded.version,
                project_ref=loaded.project_ref,
                document=document,
                quality=quality,
                producer_factory=self.producers,
                media_sink=sink,
                audio_factory=self.audio,
                playhead=playhead,
            )
        except BaseException:
            sink.close()
            raise
        with self._lock:
            self._sessions[session_id] = session
        self._publish_missing_media(session)
        return session

    def create_session(
        self,
        *,
        source_version: str,
        project_ref: str,
        playhead: Fraction,
        offer: SessionDescription,
        quality: SessionQuality,
    ) -> tuple[PreviewSession, SessionDescription]:
        if self.media is None:
            raise PreviewAPIError(
                "unsupported_construct",
                "The WebRTC preview transport is not enabled on this server.",
                status=422,
            )
        loaded = self.documents.require_current(source_version, project_ref)
        document = loaded.document
        self._require_audio_support(document)
        answer, sink = self.media.negotiate(offer)
        session_id = f"preview-{uuid.uuid4().hex}"
        try:
            session = PreviewSession(
                session_id=session_id,
                source_version=loaded.version,
                project_ref=loaded.project_ref,
                document=document,
                quality=quality,
                producer_factory=self.producers,
                media_sink=sink,
                audio_factory=self.audio,
                playhead=playhead,
            )
        except BaseException:
            sink.close()
            raise
        with self._lock:
            self._sessions[session_id] = session
        self._publish_missing_media(session)
        return session, answer

    def session(self, session_id: str) -> PreviewSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise PreviewAPIError(
                "preview_not_found",
                f"Preview session {session_id!r} does not exist.",
                status=404,
            )
        return session

    def sync(
        self,
        session_id: str,
        *,
        source_version: str,
        project_ref: str,
        playhead: Fraction,
        quality: SessionQuality | None = None,
    ) -> dict[str, object]:
        loaded = self.documents.require_current(source_version, project_ref)
        document = loaded.document
        self._require_audio_support(document)
        session = self.session(session_id)
        result = session.sync(
            source_version=loaded.version,
            project_ref=loaded.project_ref,
            document=document,
            playhead=playhead,
            quality=quality or session.quality,
        )
        self._publish_missing_media(session)
        return result

    @staticmethod
    def _publish_missing_media(session: PreviewSession) -> None:
        """Publish degraded media state after a source is pinned.

        Main callers:
        - preview creation, after the session owns its initial document.
        - source sync, after replacement succeeds transactionally.
        """

        payload = missing_media_event_data(
            session.document,
            source_version=session.source_version,
        )
        if payload is not None:
            payload["projectRef"] = session.project_ref
            session.events.publish("missing_media", payload)

    def close(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def shutdown(self) -> None:
        """Close every session and refuse no cleanup work during app shutdown."""

        with self._lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()


def validate_quality(resolution: str | None, fallbacks: list[str] | None = None) -> SessionQuality:
    """Validate the one fixed preview tier selected by the user."""

    if fallbacks:
        raise PreviewAPIError(
            "invalid_resolution",
            "Automatic preview fallback is disabled. Select one fixed resolution.",
            status=400,
        )
    try:
        selected = profile_for_mode(RenderMode.SCAN, resolution)
    except ValueError as error:
        raise PreviewAPIError("invalid_resolution", str(error), status=400) from error
    return SessionQuality(selected)
