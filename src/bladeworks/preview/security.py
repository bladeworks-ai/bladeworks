"""Local API authentication and loopback CORS validation.

Architecture map
================

    runner creates random token
        -> stdout ready record gives it to the launcher
        -> BearerTokenAuth protects preview, render, and mutation routers

Cross-origin browser access is opt-in and limited to explicit loopback
origins. Command-line tools do not send an Origin header and are unaffected.
"""

from __future__ import annotations

import hmac
import secrets
from dataclasses import dataclass
from urllib.parse import urlsplit

from fastapi import Request

from .contracts import PreviewAPIError


def new_auth_token() -> str:
    return secrets.token_urlsafe(32)


def validate_loopback_origin(value: str) -> str:
    """Return a normalized explicit HTTP origin or fail loudly."""

    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"CORS origin must be an explicit HTTP loopback origin: {value!r}")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"CORS origin has an invalid port: {value!r}") from error
    host = f"[{parsed.hostname}]" if parsed.hostname == "::1" else parsed.hostname
    suffix = f":{port}" if port is not None else ""
    return f"{parsed.scheme}://{host}{suffix}"


@dataclass(frozen=True)
class BearerTokenAuth:
    token: str

    def __call__(self, request: Request) -> None:
        supplied = request.headers.get("Authorization", "")
        expected = f"Bearer {self.token}"
        if not hmac.compare_digest(supplied, expected):
            raise PreviewAPIError("unauthorized", "A valid Bladeworks bearer token is required.", status=401)
