"""Focused Studio export-profile, job, and artifact contract tests."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from bladeworks.preview.export import TensorExecutorExportRunner
from bladeworks.preview.capabilities import bladeworks_capabilities
from bladeworks.preview.render_jobs import (
    STUDIO_EXPORT_PROFILES,
    RenderJobService,
)
from bladeworks.preview.routes import create_app


class _Documents:
    def __init__(self) -> None:
        self.document = SimpleNamespace(width=1920, height=1080, frame_count=2)

    def require_current(self, source_version: str, project_ref: str):
        assert source_version == "sha256:test"
        assert project_ref == "library[1]/event[1]/project[1]"
        return SimpleNamespace(
            version=source_version,
            project_ref=project_ref,
            document=self.document,
        )


class _PreviewService:
    def shutdown(self) -> None:
        pass


class _RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, document, **kwargs) -> None:
        self.calls.append(dict(kwargs))
        kwargs["progress"](document.frame_count, document.frame_count)
        kwargs["output_path"].write_bytes(b"rendered-artifact")


def _client(tmp_path: Path) -> tuple[TestClient, _RecordingRunner, RenderJobService]:
    runner = _RecordingRunner()
    jobs = RenderJobService(
        documents=_Documents(),  # type: ignore[arg-type]
        runner=runner,
        artifact_directory=tmp_path / "renders",
    )
    return TestClient(create_app(_PreviewService(), renders=jobs)), runner, jobs  # type: ignore[arg-type]


def _wait(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        response = client.get(f"/api/editor/renders/{job_id}")
        payload = response.json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.005)
    raise AssertionError(f"render {job_id} did not finish")


@pytest.mark.parametrize(
    ("profile", "suffix", "content_type", "executor_profile", "video_only"),
    (
        ("delivery", ".mp4", "video/mp4", "delivery", False),
        ("delivery_alpha", ".mov", "video/quicktime", "delivery_alpha", False),
    ),
)
def test_render_profiles_pin_executor_and_artifact_contract(
    tmp_path: Path,
    profile: str,
    suffix: str,
    content_type: str,
    executor_profile: str,
    video_only: bool,
) -> None:
    client, runner, jobs = _client(tmp_path)
    with client:
        started = client.post(
            "/api/editor/render",
            json={
                "sourceVersion": "sha256:test",
                "projectRef": "library[1]/event[1]/project[1]",
                "profile": profile,
            },
        )
        assert started.status_code == 202
        queued = started.json()
        assert queued["profile"] == profile
        assert queued["resolution"] == "1080p"

        completed = _wait(client, queued["jobId"])
        artifact_contract = completed["artifact"]
        assert artifact_contract["contentType"] == content_type
        assert artifact_contract["fileName"].endswith(suffix)
        without_token = client.get(artifact_contract["url"].split("?", 1)[0])
        artifact = client.get(artifact_contract["url"])

    assert without_token.status_code == 404
    assert artifact.status_code == 200
    assert artifact.content == b"rendered-artifact"
    assert artifact.headers["content-type"].startswith(content_type)
    assert f'filename="{artifact_contract["fileName"]}"' in artifact.headers[
        "content-disposition"
    ]
    assert len(runner.calls) == 1
    assert runner.calls[0]["output_profile"] == executor_profile
    assert runner.calls[0]["video_only"] is video_only
    assert jobs.get(queued["jobId"]).output_path.suffix == suffix


def test_render_defaults_to_1080p_delivery_and_rejects_unknown_profiles(tmp_path: Path) -> None:
    client, runner, _jobs = _client(tmp_path)
    with client:
        started = client.post(
            "/api/editor/render",
            json={
                "sourceVersion": "sha256:test",
                "projectRef": "library[1]/event[1]/project[1]",
            },
        )
        assert started.status_code == 202
        assert started.json()["profile"] == "delivery"
        assert started.json()["resolution"] == "1080p"

        for invalid in ("video_only", "oracle_mezzanine", "", 7, False):
            response = client.post(
                "/api/editor/render",
                json={
                    "sourceVersion": "sha256:test",
                    "projectRef": "library[1]/event[1]/project[1]",
                    "profile": invalid,
                },
            )
            assert response.status_code == 400
            assert response.json()["error"]["code"] == "invalid_render_profile"

    assert runner.calls


def test_capability_catalog_and_render_service_share_one_profile_registry() -> None:
    public = bladeworks_capabilities()["export"]["profiles"]

    assert [profile["id"] for profile in public] == list(STUDIO_EXPORT_PROFILES)
    assert {profile["id"]: profile["container"] for profile in public} == {
        "delivery": "mp4",
        "delivery_alpha": "mov",
    }


def test_tensor_executor_bridge_forwards_profile_and_video_only(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_execute_render(document, report, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("bladeworks.preview.export.execute_render", fake_execute_render)
    runner = TensorExecutorExportRunner(report_for=lambda _document: object())  # type: ignore[arg-type]
    output = tmp_path / "alpha.mov"

    runner.run(
        object(),  # type: ignore[arg-type]
        output_path=output,
        output_resolution=object(),  # type: ignore[arg-type]
        output_profile="delivery_alpha",
        video_only=True,
        progress=lambda _completed, _total: None,
        is_cancelled=lambda: False,
    )

    assert captured["backend"] == "tensor"
    assert captured["output_profile"] == "delivery_alpha"
    assert captured["video_only"] is True
    assert captured["output_path"] == output
