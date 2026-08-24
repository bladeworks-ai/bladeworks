"""Local HTTP/SSE/WebRTC orchestration around the Bladeworks renderer.

The package deliberately keeps browser transports outside Bladeworks. Service
objects depend on small document, frame, and media protocols; ``routes`` is
only the FastAPI adapter.
"""

from .contracts import PreviewAPIError
from .service import PreviewService

__all__ = ["PreviewAPIError", "PreviewService"]
