"""Trusted arbitrary-transition support for the portable FCPXML renderer.

This package intentionally separates offline shader construction from runtime
artifact loading. Customer renders import ``runtime`` only; no compiler binary
or shader-source entry point is reachable from the render path.
"""

from .contract import ABI_VERSION, WorkingImageContract

__all__ = ["ABI_VERSION", "WorkingImageContract"]
