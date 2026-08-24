"""Portable FCPXML compiler and renderer.

Architecture map
================

``parse_fcpxml`` keeps source hierarchy and rational time intact.
``compile_fcpxml`` resolves that source graph into renderer-facing clips.
``execute_render`` turns the render document into a video.

The package deliberately does not import the existing Swift/SPLYML converter:
that converter targets edit interchange and is allowed to lose Final Cut-only
details, while this package needs those details to explain every render choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core.compiler import CompileResult
    from .executor import RenderResult

__all__ = [
    "CompileResult",
    "RenderResult",
    "compile_fcpxml",
    "execute_render",
    "parse_fcpxml",
]


def __getattr__(name: str) -> Any:
    """Load renderer APIs only when a caller asks for one.

    Main callers:
        Users importing a symbol from the package root.

    Why this exists:
        Python imports the package before running ``python -m bladeworks``.
        Keeping the root lightweight lets help and package introspection work
        without initializing PyTorch and the video runtime.
    """

    if name in {"CompileResult", "compile_fcpxml"}:
        from .core import compiler

        return getattr(compiler, name)
    if name in {"RenderResult", "execute_render"}:
        from . import executor

        return getattr(executor, name)
    if name == "parse_fcpxml":
        from .core.parser import parse_fcpxml

        return parse_fcpxml
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
