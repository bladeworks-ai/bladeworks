"""Production export runner backed by Bladeworks's existing audio mux executor.

The runner does not compile projects or own source versions. Its caller supplies the
already-compiled ``RenderDocument`` and a callable that returns the matching
fresh compatibility report. Bladeworks renders video at the requested fixed
resolution; the existing executor builds and muxes the calibrated audio graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..core.model import RenderDocument
from ..core.report import CompatibilityReport
from ..executor import execute_render
from ..tensor.resolution import OutputResolution


class TensorExecutorExportRunner:
    """Bridge ``RenderJobService`` to one validated tensor delivery profile."""

    def __init__(
        self,
        *,
        report_for: Callable[[RenderDocument], CompatibilityReport],
        encoder_preset: str | None = None,
        device: str | None = None,
    ) -> None:
        self.report_for = report_for
        self.encoder_preset = encoder_preset
        self.device = device

    def run(
        self,
        document: RenderDocument,
        *,
        output_path: Path,
        output_resolution: OutputResolution,
        output_profile: str,
        video_only: bool,
        progress,
        is_cancelled,
    ) -> None:
        execute_render(
            document,
            self.report_for(document),
            output_path=output_path,
            backend="tensor",
            output_profile=output_profile,
            video_only=video_only,
            output_resolution=output_resolution,
            encoder_preset=self.encoder_preset,
            progress=progress,
            is_cancelled=is_cancelled,
            device=None if self.device in (None, "auto") else self.device,
        )
