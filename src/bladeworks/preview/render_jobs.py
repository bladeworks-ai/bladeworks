"""Asynchronous export-job registry for the local editor API.

Architecture map
================

    POST render
        -> exact source hash from the opened FCPXML store
        -> fixed OutputResolution and validated export profile for the whole job
        -> injected ExportRunner on one worker thread
        -> polled job state and a typed MP4 or MOV artifact route

The registry never changes export quality automatically. Cancellation is a
token passed into the runner and partial artifacts are removed when the runner
returns or raises after cancellation.
"""

from __future__ import annotations

import secrets
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

from ..core.model import RenderDocument
from ..tensor.resolution import OutputResolution, RenderMode, ResolutionProfile, profile_for_mode, resolve_output_resolution
from .contracts import PreviewAPIError, SourceDocumentProvider


@dataclass(frozen=True)
class StudioExportProfile:
    """One complete public export contract the tensor executor can honor."""

    id: str
    output_profile: str
    video_only: bool
    suffix: str
    content_type: str
    container: str
    video_codec: str
    audio_codec: str | None
    alpha: bool


STUDIO_EXPORT_PROFILES: dict[str, StudioExportProfile] = {
    "delivery": StudioExportProfile(
        id="delivery",
        output_profile="delivery",
        video_only=False,
        suffix=".mp4",
        content_type="video/mp4",
        container="mp4",
        video_codec="h264",
        audio_codec="aac",
        alpha=False,
    ),
    "delivery_alpha": StudioExportProfile(
        id="delivery_alpha",
        output_profile="delivery_alpha",
        video_only=False,
        suffix=".mov",
        content_type="video/quicktime",
        container="mov",
        video_codec="prores4444",
        audio_codec="aac",
        alpha=True,
    ),
}


class ExportRunner(Protocol):
    def run(
        self,
        document: RenderDocument,
        *,
        output_path: Path,
        output_resolution: OutputResolution,
        output_profile: str,
        video_only: bool,
        progress: Callable[[int, int], None],
        is_cancelled: Callable[[], bool],
    ) -> None: ...


@dataclass
class RenderJob:
    job_id: str
    source_version: str
    project_ref: str
    export_profile: StudioExportProfile
    profile: ResolutionProfile
    resolution: OutputResolution
    output_path: Path
    total_frames: int
    download_token: str
    status: str = "queued"
    completed_frames: int = 0
    error: Optional[str] = None
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: Optional[threading.Thread] = field(default=None, repr=False)

    def payload(self, *, status_override: str | None = None) -> dict[str, object]:
        status = status_override or self.status
        result: dict[str, object] = {
            "jobId": self.job_id,
            "sourceVersion": self.source_version,
            "projectRef": self.project_ref,
            "profile": self.export_profile.id,
            "status": status,
            "resolution": self.profile.value,
            "width": self.resolution.width,
            "height": self.resolution.height,
            "completedFrames": self.completed_frames,
            "totalFrames": self.total_frames,
        }
        if status == "completed":
            result["artifact"] = {
                "contentType": self.export_profile.content_type,
                "fileName": self.output_path.name,
                "url": f"/api/editor/renders/{self.job_id}/artifact?token={self.download_token}",
            }
        if self.error is not None:
            result["error"] = {
                "code": "render_failed",
                "message": self.error,
                "retryable": False,
            }
        return result


class RenderJobService:
    def __init__(
        self,
        *,
        documents: SourceDocumentProvider,
        runner: ExportRunner,
        artifact_directory: Path,
    ) -> None:
        self.documents = documents
        self.runner = runner
        self.artifact_directory = Path(artifact_directory).resolve()
        self._lock = threading.Lock()
        self._jobs: dict[str, RenderJob] = {}
        self._accepting = True

    def start(
        self,
        *,
        source_version: str,
        project_ref: str,
        profile: str | None,
        export_profile: str | None = None,
    ) -> RenderJob:
        export_profile_id = "delivery" if export_profile is None else export_profile
        try:
            export_spec = STUDIO_EXPORT_PROFILES[export_profile_id]
        except (KeyError, TypeError) as error:
            supported = ", ".join(STUDIO_EXPORT_PROFILES)
            raise PreviewAPIError(
                "invalid_render_profile",
                f"Render profile must be one of: {supported}.",
                status=400,
            ) from error
        try:
            selected = profile_for_mode(RenderMode.RENDER, profile)
        except ValueError as error:
            raise PreviewAPIError("invalid_resolution", str(error), status=400) from error
        with self._lock:
            if not self._accepting:
                raise PreviewAPIError("render_failed", "The render service is shutting down.", status=503)
        loaded = self.documents.require_current(source_version, project_ref)
        document = loaded.document
        if document.frame_count == 0:
            raise PreviewAPIError(
                "empty_timeline",
                "The selected Project has no timeline frames to export.",
                status=422,
            )
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        resolution = resolve_output_resolution(document.width, document.height, selected)
        job_id = f"render-{uuid.uuid4().hex}"
        job = RenderJob(
            job_id=job_id,
            source_version=loaded.version,
            project_ref=loaded.project_ref,
            export_profile=export_spec,
            profile=selected,
            resolution=resolution,
            output_path=self.artifact_directory / f"{job_id}{export_spec.suffix}",
            total_frames=document.frame_count,
            download_token=secrets.token_urlsafe(32),
        )
        thread = threading.Thread(
            target=self._run,
            args=(job, document),
            name=f"tensor-export:{job_id}",
            daemon=True,
        )
        job.thread = thread
        with self._lock:
            if not self._accepting:
                raise PreviewAPIError("render_failed", "The render service is shutting down.", status=503)
            if any(
                existing.status in {"queued", "running", "cancelling"}
                for existing in self._jobs.values()
            ):
                raise PreviewAPIError("render_busy", "An export is already in progress.", status=409)
            self._jobs[job_id] = job
            thread.start()
        return job

    def _run(self, job: RenderJob, document: RenderDocument) -> None:
        with self._lock:
            if job.cancel.is_set():
                job.status = "cancelled"
                return
            job.status = "running"

        def progress(completed: int, total: int) -> None:
            with self._lock:
                job.completed_frames = completed
                job.total_frames = total

        try:
            self.runner.run(
                document,
                output_path=job.output_path,
                output_resolution=job.resolution,
                output_profile=job.export_profile.output_profile,
                video_only=job.export_profile.video_only,
                progress=progress,
                is_cancelled=job.cancel.is_set,
            )
            with self._lock:
                # Returning from the runner is the export commit boundary. A
                # cancellation that arrives after that boundary must not
                # discard an already completed artifact.
                if not job.output_path.is_file() or job.output_path.stat().st_size <= 0:
                    if job.cancel.is_set():
                        job.status = "cancelled"
                        return
                    job.status = "failed"
                    job.error = "Export runner returned without a non-empty artifact."
                else:
                    job.status = "completed"
                    job.completed_frames = job.total_frames
        except Exception as error:  # noqa: BLE001 - retained as polled job state
            with self._lock:
                if job.cancel.is_set():
                    job.status = "cancelled"
                else:
                    job.status = "failed"
                    job.error = str(error)
        finally:
            if job.status == "cancelled":
                job.output_path.unlink(missing_ok=True)
                job.output_path.with_suffix(".compatibility.json").unlink(missing_ok=True)
                job.output_path.with_suffix(".manifest.json").unlink(missing_ok=True)

    def get(self, job_id: str) -> RenderJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise PreviewAPIError("render_not_found", f"Render job {job_id!r} does not exist.", status=404)
        return job

    def get_artifact(
        self, job_id: str, token: str | None, *, bearer_authorized: bool = False
    ) -> RenderJob:
        job = self.get(job_id)
        if bearer_authorized:
            return job
        if not secrets.compare_digest(token or "", job.download_token):
            raise PreviewAPIError("render_not_found", "Render artifact token is invalid.", status=404)
        return job

    def cancel(self, job_id: str) -> RenderJob:
        job = self.get(job_id)
        job.cancel.set()
        with self._lock:
            if job.status in {"queued", "running"}:
                job.status = "cancelling"
        return job

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(job.status in {"queued", "running", "cancelling"} for job in self._jobs.values())

    def shutdown(self) -> None:
        """Cancel active jobs, wait for workers, and remove partial artifacts."""

        with self._lock:
            self._accepting = False
            jobs = tuple(self._jobs.values())
            for job in jobs:
                if job.status in {"queued", "running", "cancelling"}:
                    job.cancel.set()
        for job in jobs:
            if job.thread is not None and job.thread is not threading.current_thread():
                job.thread.join()
            if job.status != "completed":
                job.output_path.unlink(missing_ok=True)
                job.output_path.with_suffix(".compatibility.json").unlink(missing_ok=True)
                job.output_path.with_suffix(".manifest.json").unlink(missing_ok=True)


class UnavailableExportRunner:
    def run(self, document: RenderDocument, **_kwargs) -> None:
        raise PreviewAPIError(
            "render_failed",
            "No Bladeworks export runner is configured.",
            status=503,
        )
