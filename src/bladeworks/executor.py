"""Tensor-only execution boundary for the standalone Bladeworks package.

Architecture map:

    compiled RenderDocument + CompatibilityReport
        -> resolve PyAV audio delivery
        -> render video and audio through the tensor renderer
        -> write compatibility and execution manifests

The internal Bladeworks executor also dispatches deprecated CPU and Vulkan
backends. The public package deliberately uses this smaller boundary so those
implementations are neither imported nor distributed.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from .core.errors import FCPXMLCompileError, RenderCapabilityError
from .core.missing_media import resolve_missing_media_raster
from .core.model import FFmpegInvocation, RenderClip, RenderDocument, dataclass_json
from .core.report import CompatibilityReport
from .core.text import (
    FontResolver,
    RuntimeRasterResolution,
    resolve_generator_clip_raster,
    resolve_text_clip_raster,
)
from .tensor import audio_pyav
from .tensor.audio_delivery import (
    audio_delivery_layout,
    resolve_audio_delivery,
    video_only_silence_resolution,
)
from .tensor.encode import EncoderAudio
from .tensor.renderer import render_document
from .tensor.plan import build_tensor_plan


@dataclass(frozen=True)
class RenderResult:
    output_path: Path
    report_path: Path
    manifest_path: Path
    invocation: FFmpegInvocation
    degraded: bool
    requested_backend: str = "tensor"
    selected_backend: str = "tensor"


def _runtime_raster_clips(document: RenderDocument) -> tuple[RenderClip, ...]:
    """Return composite clips that require a locally generated image."""

    result = []
    for clip in document.clips:
        requires_raster = clip.enabled and (
            bool(clip.missing_media_locators)
            or clip.kind in {"title", "caption"}
            or (clip.generator_plan is not None and clip.generator_plan.execution == "solid_color")
        )
        if requires_raster and clip.video_disposition is not None and clip.video_disposition.execution == "composite":
            result.append(clip)
    return tuple(result)


def _prepare_runtime_rasters(
    document: RenderDocument,
    report: CompatibilityReport,
    work_dir: Path,
) -> tuple[RenderDocument, dict[str, Path]]:
    """Rasterize supported runtime surfaces and return their tensor inputs."""

    clips = _runtime_raster_clips(document)
    resolver = FontResolver(document.font_bindings)
    resolutions: list[RuntimeRasterResolution] = []
    for clip in clips:
        if clip.missing_media_locators:
            resolutions.append(resolve_missing_media_raster(clip, document, work_dir=work_dir))
        elif clip.kind in {"title", "caption"}:
            resolutions.append(
                resolve_text_clip_raster(clip, document, work_dir=work_dir, resolver=resolver, report=report)
            )
        elif clip.generator_plan is not None:
            resolutions.append(
                resolve_generator_clip_raster(clip, document, work_dir=work_dir, report=report)
            )

    by_id = {resolution.clip_id: resolution for resolution in resolutions}
    if set(by_id) != {clip.id for clip in clips}:
        raise FCPXMLCompileError("runtime raster results do not cover every raster clip")
    updated = tuple(
        replace(clip, video_disposition=by_id[clip.id].video_disposition)
        if clip.id in by_id
        else clip
        for clip in document.clips
    )
    rasters = {
        resolution.clip_id: resolution.image_path
        for resolution in resolutions
        if resolution.image_path is not None
    }
    return replace(document, clips=updated), rasters


def _write_failure_manifest(path: Path, output_path: Path, error: BaseException, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "engine": "bladeworks",
                "status": status,
                "error": str(error),
                "render_backend": {"requested": "tensor", "selected": "tensor"},
                "output": {"path": str(output_path), "size_bytes": None},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _encoder_audio(document: RenderDocument, resolution: Any) -> EncoderAudio:
    if resolution.mode == "render" and resolution.execution is not None:
        execution = resolution.execution
        return EncoderAudio(
            audio_pyav.render_execution_frames(execution),
            execution.sample_rate,
            execution.ffmpeg_layout,
        )
    sample_rate = document.audio.sample_rate if document.audio is not None else 48_000
    layout = audio_delivery_layout(document)
    duration = document.audio.sequence_duration if document.audio is not None else document.duration
    return EncoderAudio(
        audio_pyav.render_silence_frames(
            sample_rate=sample_rate,
            ffmpeg_layout=layout,
            duration=duration,
        ),
        sample_rate,
        layout,
    )


def execute_render(
    document: RenderDocument,
    report: CompatibilityReport,
    *,
    output_path: Path,
    report_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    emit_plan_path: Optional[Path] = None,
    strict: bool = False,
    video_only: bool = False,
    output_profile: str = "delivery",
    backend: str = "tensor",
    cpu_segmentation: Any = None,
    shared_planning: Any = None,
    render_profile: str = "reference",
    encoder_preset: Optional[str] = None,
    show_progress: bool = False,
    output_resolution: Any = None,
    progress: Optional[Callable[[int, int], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    device: Optional[str] = None,
    **_: Any,
) -> RenderResult:
    """Render with the standalone package's single supported backend."""

    output_path = Path(output_path).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve() if report_path else output_path.with_suffix(".compatibility.json")
    manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path else output_path.with_suffix(".manifest.json")
    bar = None
    try:
        if backend != "tensor":
            raise RenderCapabilityError("standalone Bladeworks supports only the tensor backend")
        if cpu_segmentation is not None or shared_planning is not None:
            raise RenderCapabilityError("CPU segmentation and Vulkan planning are not part of Bladeworks")
        if render_profile != "reference":
            raise RenderCapabilityError(
                f"standalone Bladeworks supports only the reference render profile, not {render_profile!r}"
            )
        if output_profile not in {"delivery", "delivery_alpha"}:
            raise RenderCapabilityError(f"unsupported Bladeworks output profile {output_profile!r}")

        report.write(report_path)
        if strict and report.has_strict_failures:
            raise FCPXMLCompileError("--strict rejected approximated or omitted constructs; see compatibility report")

        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            raise RenderCapabilityError("ffprobe is required to resolve source audio")
        audio_resolution = (
            video_only_silence_resolution(document)
            if video_only
            else resolve_audio_delivery(document, ffprobe=ffprobe, report=report)
        )
        if video_only and document.audio is not None:
            report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path="fcpxml/project/sequence",
                construct="source audio in video-only render",
                timeline_start=document.tc_start,
                timeline_duration=document.duration,
                disposition="the caller requested video-only output; source audio was omitted",
            )
        report.write(report_path)
        if strict and report.has_strict_failures:
            raise FCPXMLCompileError("--strict rejected approximated or omitted constructs; see compatibility report")

        if emit_plan_path is not None:
            plan_path = Path(emit_plan_path).expanduser().resolve()
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(dataclass_json(document), indent=2, sort_keys=True) + "\n", encoding="utf-8")

        codec = "prores_ks" if output_profile == "delivery_alpha" else "libx264"
        pixel_policy = "alpha" if output_profile == "delivery_alpha" else "opaque"
        preset = encoder_preset or "medium"
        callback = progress
        if show_progress and callback is None:
            from tqdm.auto import tqdm

            bar = tqdm(total=document.frame_count, unit="frame")
            callback = lambda completed, total: bar.update(completed - bar.n)

        with tempfile.TemporaryDirectory(prefix="bladeworks-raster-") as temporary:
            document, rasters = _prepare_runtime_rasters(document, report, Path(temporary))
            report.write(report_path)
            if strict and report.has_strict_failures:
                raise FCPXMLCompileError(
                    "--strict rejected approximated or omitted constructs; see compatibility report"
                )
            plan = build_tensor_plan(
                document,
                rasters=rasters,
                output_resolution=output_resolution,
            )
            for layer in plan.layers:
                if layer.source_has_alpha and layer.alpha_handling is None:
                    report.add(
                        outcome="info",
                        portable_status="exact_portable",
                        fcpxml_path=layer.path,
                        construct="alpha source interpretation",
                        disposition=(
                            "source contains alpha and carries no Final Cut alphaHandling "
                            "override; interpreted as straight alpha"
                        ),
                    )
            report.write(report_path)
            stats = render_document(
                document,
                output_path=output_path,
                codec=codec,
                preset=preset,
                pixel_policy=pixel_policy,
                plan=plan,
                progress=callback,
                is_cancelled=is_cancelled,
                device=device,
                audio=_encoder_audio(document, audio_resolution),
            )
    except KeyboardInterrupt as error:
        output_path.unlink(missing_ok=True)
        _write_failure_manifest(manifest_path, output_path, error, "cancelled")
        raise
    except Exception as error:
        output_path.unlink(missing_ok=True)
        report.add(
            outcome="failed",
            portable_status="unsupported",
            fcpxml_path="sequence",
            construct="portable render",
            disposition=str(error),
        )
        report.write(report_path)
        _write_failure_manifest(manifest_path, output_path, error, "failed")
        raise
    finally:
        if bar is not None:
            bar.close()

    invocation = FFmpegInvocation(
        argv=(),
        filter_script="",
        expected_frame_count=document.frame_count,
        output_path=output_path,
        input_paths=(),
        requested_backend="tensor",
        selected_backend="tensor",
    )
    alpha_sources = [
        {
            "path": layer.path,
            "handling": layer.alpha_handling or "straight",
            "authored_override": layer.alpha_handling is not None,
        }
        for layer in plan.layers
        if layer.source_has_alpha
    ]
    render_backend: dict[str, Any] = {"requested": "tensor", "selected": "tensor"}
    if alpha_sources:
        render_backend["alpha_sources"] = alpha_sources
    manifest = {
        "schema_version": 1,
        "engine": "bladeworks",
        "status": "succeeded",
        "render_backend": render_backend,
        "output": {"path": str(output_path)},
        "stats": dataclass_json(stats),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RenderResult(output_path, report_path, manifest_path, invocation, report.degraded)
