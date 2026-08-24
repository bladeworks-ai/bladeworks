"""Bladeworks implementation of the preview frame-production contracts.

Seek and scan (this producer) build their plans with
``DecodePolicy.VISIBLE``: ordinary Fit / Fill / static crop / static zoom
leaves decode near their visible output contribution instead of at native
resolution (``tensor/decode_policy.py`` documents the fallback rules).  Export
(``preview/export.py`` -> ``execute_render``) never uses this producer and keeps
native decoding. Because the decode raster is a property of the plan, changing
the user's fixed 720p / 540p / 480p preview choice recreates the producer and
every decoder with rasters sized for that tier.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from ..core.missing_media import resolve_missing_media_raster
from ..core.model import RenderClip, RenderDocument
from ..core.report import CompatibilityReport
from ..core.text import (
    FontResolver,
    RuntimeRasterResolution,
    resolve_generator_clip_raster,
    resolve_text_clip_raster,
)
from ..tensor.decode_policy import DecodePolicy
from ..tensor.plan import build_tensor_plan
from ..tensor.renderer import TensorRenderSession
from ..tensor.resolution import OutputResolution
from .contracts import PreviewAPIError

# The interactive contract: seek and scan may downscale at the decoder.
PREVIEW_DECODE_POLICY = DecodePolicy.VISIBLE


class TensorFrameProducer(TensorRenderSession):
    def __init__(
        self,
        document: RenderDocument,
        *,
        output_resolution: OutputResolution,
        device: str | None,
        decoder_threads: int,
    ) -> None:
        self._runtime_raster_directory: TemporaryDirectory[str] | None = None
        raster_clips = _runtime_raster_clips(document)
        if not raster_clips:
            super().__init__(
                document,
                output_resolution=output_resolution,
                device=device,
                decoder_threads=decoder_threads,
                decode_policy=PREVIEW_DECODE_POLICY,
            )
            return

        temporary = TemporaryDirectory(prefix="tensor-preview-runtime-rasters-")
        self._runtime_raster_directory = temporary
        work_dir = Path(temporary.name)
        try:
            prepared_document, rasters = _prepare_runtime_rasters(
                document,
                raster_clips,
                work_dir=work_dir,
                reject_omissions=True,
            )
            plan = build_tensor_plan(
                prepared_document,
                rasters=rasters,
                output_resolution=output_resolution,
                decode_policy=PREVIEW_DECODE_POLICY,
            )
            super().__init__(
                prepared_document,
                plan=plan,
                device=device,
                decoder_threads=decoder_threads,
            )
        except Exception:
            temporary.cleanup()
            self._runtime_raster_directory = None
            raise

    @property
    def frame_duration(self) -> Fraction:
        return self.plan.frame_duration

    @property
    def frame_count(self) -> int:
        return self.plan.frame_count

    def close(self) -> None:
        try:
            super().close()
        finally:
            temporary = self._runtime_raster_directory
            self._runtime_raster_directory = None
            if temporary is not None:
                temporary.cleanup()


def _runtime_raster_clips(document: RenderDocument) -> tuple[RenderClip, ...]:
    """Return every live-preview clip whose pixels must be prepared locally."""

    return tuple(
        clip
        for clip in document.clips
        if clip.enabled
        and clip.video_disposition is not None
        and clip.video_disposition.execution == "composite"
        and (
            bool(clip.missing_media_locators)
            or clip.kind in {"title", "caption"}
            or (
                clip.generator_plan is not None
                and clip.generator_plan.execution == "solid_color"
            )
        )
    )


def _prepare_runtime_rasters(
    document: RenderDocument,
    clips: tuple[RenderClip, ...],
    *,
    work_dir: Path,
    reject_omissions: bool = False,
) -> tuple[RenderDocument, dict[str, Path]]:
    """Prepare the same title, generator, and missing-media pixels as export.

    Main callers:
    - ``TensorFrameProducer.__init__`` before it builds the live seek/scan plan.

    Why this exists:
    Preview bypasses ``executor.execute_render``, which normally materializes
    these sources for export. Preparing them here keeps seek and scan on the
    same compiled document semantics without routing preview through the batch
    export lifecycle.
    """

    report = CompatibilityReport(project_name=document.project_name)
    resolver = FontResolver(document.font_bindings)
    resolutions: list[RuntimeRasterResolution] = []
    for clip in clips:
        if clip.missing_media_locators:
            resolution = resolve_missing_media_raster(clip, document, work_dir=work_dir)
        elif clip.kind in {"title", "caption"}:
            resolution = resolve_text_clip_raster(
                clip,
                document,
                work_dir=work_dir,
                resolver=resolver,
                report=report,
            )
        else:
            resolution = resolve_generator_clip_raster(
                clip,
                document,
                work_dir=work_dir,
                report=report,
            )
        resolutions.append(resolution)

    unavailable = [
        finding
        for finding in report.findings
        if finding.outcome in {"omitted", "failed"}
    ]
    if reject_omissions and unavailable:
        details = "; ".join(
            f"{finding.construct}: {finding.disposition}" for finding in unavailable
        )
        raise PreviewAPIError(
            "preview_raster_unavailable",
            f"Preview cannot display one or more titles or generators. {details}",
            status=422,
        )

    by_id = {resolution.clip_id: resolution for resolution in resolutions}
    prepared = replace(
        document,
        clips=tuple(
            replace(clip, video_disposition=by_id[clip.id].video_disposition)
            if clip.id in by_id
            else clip
            for clip in document.clips
        ),
    )
    rasters = {
        resolution.clip_id: resolution.image_path
        for resolution in resolutions
        if resolution.image_path is not None
    }
    return prepared, rasters


class TensorFrameProducerFactory:
    def __init__(self, *, device: str | None = None, decoder_threads: int = 2) -> None:
        self.device = device
        self.decoder_threads = decoder_threads

    def create(self, document: RenderDocument, resolution: OutputResolution) -> TensorFrameProducer:
        return TensorFrameProducer(
            document,
            output_resolution=resolution,
            device=self.device,
            decoder_threads=self.decoder_threads,
        )
