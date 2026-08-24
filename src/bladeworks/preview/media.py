"""Live-media boundaries and the explicit unavailable default.

Why this exists
---------------
The API contract requires WebRTC. Returning a fabricated SDP answer would make
HTTP tests pass while the browser receives no pixels. The production default
therefore fails loudly until a real adapter is configured; tests may inject a
recording sink and factory.
"""

from __future__ import annotations

from .contracts import PreviewAPIError, PreviewMediaSink, SessionDescription


class UnavailableMediaFactory:
    def negotiate(self, offer: SessionDescription) -> tuple[SessionDescription, PreviewMediaSink]:
        raise PreviewAPIError(
            "preview_media_unavailable",
            "No WebRTC media adapter is configured for the local preview service.",
            status=503,
            retryable=False,
        )
