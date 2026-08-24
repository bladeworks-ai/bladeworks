"""``bladeworks`` CLI first-pass surface: examples / render defaults / doctor.

What this covers
----------------
The ``bladeworks`` CLI (``bladeworks.cli:main``) grew four
user-facing behaviors this test locks down:

1. ``examples ls`` lists all packaged sample projects (one line each).
2. ``examples cp <name> [DEST]`` copies a renderable ``.fcpxmld`` bundle out,
   and fails LOUDLY (non-zero, no silent overwrite) on an unknown name or a
   destination that already exists. The copied bundle then renders end to end.
3. ``render <bundle>`` with NO ``-o`` writes ``<stem>.mp4`` next to the input
   and defaults to the ``tensor`` backend; ``--alpha`` writes ``<stem>.mov``.
4. ``doctor`` exits 0 when ``ffprobe`` is on PATH.
5. Backwards path: ``python -m bladeworks render ...`` still works.

Skips when ffmpeg / ffprobe / torch / PyAV are unavailable (rendering needs
all four; the pure ``examples ls`` / ``doctor`` checks do not and run anyway).
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from bladeworks.cli import main
from bladeworks.examples import EXAMPLES
from bladeworks.preview import runner as preview_runner


# Repo root: the parent of the top-level ``backend`` package. Put it on the
# subprocess PYTHONPATH so ``python -m bladeworks`` resolves even
# without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_render_tools() -> tuple[str, str]:
    """Skip unless the full render toolchain (ffmpeg/ffprobe/torch/av) is present."""

    pytest.importorskip("torch")
    pytest.importorskip("av")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("needs ffmpeg/ffprobe")
    return ffmpeg, ffprobe


def _stream_codecs(ffprobe: str, path: Path, kind: str) -> list[str]:
    """Return the codec names of every stream of ``kind`` ('v' or 'a')."""

    result = subprocess.run(
        (ffprobe, "-v", "error", "-select_streams", kind,
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)),
        capture_output=True, text=True, check=True,
    )
    return [line for line in result.stdout.split() if line]


def test_examples_ls_lists_all(capsys: pytest.CaptureFixture[str]) -> None:
    """``examples ls`` prints every packaged sample name."""

    rc = main(["examples", "ls"])
    assert rc == 0
    out = capsys.readouterr().out
    for name in EXAMPLES:
        assert name in out, f"{name} missing from `examples ls` output"


def test_inspect_cli_emits_exact_reverse_freeze_and_variable_retime_plan(
    tmp_path: Path,
) -> None:
    """A generated FCPXML fixture keeps every retime segment through the public CLI."""

    source = tmp_path / "retime.fcpxml"
    source.write_text(
        '''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
<resources>
  <format id="fmt" frameDuration="1/30s" width="64" height="64" colorSpace="1-1-1 (Rec. 709)"/>
  <asset id="src" start="0s" duration="20s" hasVideo="1" hasAudio="0" format="fmt">
    <media-rep kind="original-media" src="file:///renderer-test-mock.mov"/>
  </asset>
</resources>
<library><event name="Tests"><project name="Retime Clock"><sequence format="fmt" duration="4s"><spine>
  <asset-clip ref="src" offset="0s" start="0s" duration="4s" format="fmt">
    <timeMap>
      <timept time="0s" value="4s" interp="linear"/>
      <timept time="1s" value="2s" interp="linear"/>
      <timept time="2s" value="2s" interp="linear"/>
      <timept time="4s" value="5s" interp="linear"/>
    </timeMap>
  </asset-clip>
</spine></sequence></project></event></library></fcpxml>''',
        encoding="utf-8",
    )
    plan = tmp_path / "plan.json"

    assert main(["inspect", str(source), "--emit-plan", str(plan)]) == 0

    payload = json.loads(plan.read_text(encoding="utf-8"))
    segments = payload["clips"][0]["retime_map"]["segments"]
    source_ranges = [
        (segment["source_start"]["numerator"], segment["source_end"]["numerator"])
        for segment in segments
    ]
    assert source_ranges == [(4, 2), (2, 2), (2, 5)]
    assert segments[2]["source_end"] == {"denominator": 1, "numerator": 5}


def test_examples_cp_copies_and_renders(tmp_path: Path) -> None:
    """``examples cp single_clip <dest>`` copies a renderable bundle; it renders."""

    _, ffprobe = _require_render_tools()

    dest = tmp_path / "out"
    dest.mkdir()
    rc = main(["examples", "cp", "single_clip", str(dest)])
    assert rc == 0

    bundle = dest / "single_clip.fcpxmld"
    assert (bundle / "Info.fcpxml").is_file()
    assert (bundle / "Media" / "a.mp4").is_file()

    # The copied bundle renders end to end (no -o -> single_clip.mp4 beside it).
    rc = main(["render", str(bundle), "--no-progress"])
    assert rc == 0
    output = dest / "single_clip.mp4"
    assert output.exists()
    assert _stream_codecs(ffprobe, output, "v"), "copied bundle produced no video"
    assert "aac" in _stream_codecs(ffprobe, output, "a")


def test_examples_cp_unknown_name_fails_loudly(tmp_path: Path) -> None:
    """An unknown sample name is a loud non-zero error, not a silent miss."""

    rc = main(["examples", "cp", "does_not_exist", str(tmp_path)])
    assert rc == 1


def test_examples_cp_existing_dest_fails_loudly(tmp_path: Path) -> None:
    """A target bundle that already exists is refused (no silent overwrite)."""

    rc = main(["examples", "cp", "single_clip", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "single_clip.fcpxmld").is_dir()

    # Second copy to the same dest must refuse rather than clobber.
    rc = main(["examples", "cp", "single_clip", str(tmp_path)])
    assert rc == 1


def test_render_without_output_writes_mp4(tmp_path: Path) -> None:
    """``render <bundle>`` with NO -o writes <stem>.mp4 with A + V, tensor default."""

    _, ffprobe = _require_render_tools()

    dest = tmp_path / "proj"
    dest.mkdir()
    assert main(["examples", "cp", "single_clip", str(dest)]) == 0
    bundle = dest / "single_clip.fcpxmld"

    rc = main(["render", str(bundle), "--no-progress"])
    assert rc == 0
    output = dest / "single_clip.mp4"
    assert output.exists(), "default output should be <stem>.mp4 beside the input"
    assert _stream_codecs(ffprobe, output, "v"), "no video stream"
    assert "aac" in _stream_codecs(ffprobe, output, "a"), "no AAC audio stream"


def test_render_alpha_without_output_writes_mov(tmp_path: Path) -> None:
    """``render <bundle> --alpha`` with NO -o writes <stem>.mov (ProRes 4444)."""

    _, ffprobe = _require_render_tools()

    dest = tmp_path / "proj"
    dest.mkdir()
    assert main(["examples", "cp", "single_clip", str(dest)]) == 0
    bundle = dest / "single_clip.fcpxmld"

    rc = main(["render", str(bundle), "--alpha", "--no-progress"])
    assert rc == 0
    output = dest / "single_clip.mov"
    assert output.exists(), "--alpha default output should be <stem>.mov"
    assert "prores" in _stream_codecs(ffprobe, output, "v"), "--alpha should be ProRes"


def test_doctor_exits_zero_when_ffprobe_present() -> None:
    """``doctor`` returns 0 when ffprobe is on PATH."""

    if not shutil.which("ffprobe"):
        pytest.skip("needs ffprobe")
    assert main(["doctor"]) == 0


def test_server_run_parses_the_foreground_lifecycle_flags(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "project.fcpxmld"
    bundle.mkdir()
    captured = []
    monkeypatch.setattr(preview_runner, "run_server", lambda config: captured.append(config) or 0)

    result = main(
        [
            "server",
            "run",
            str(bundle),
            "--port",
            "0",
            "--device",
            "cpu",
            "--decoder-threads",
            "3",
            "--history-limit",
            "12",
            "--strict",
            "--allow-origin",
            "http://localhost:3000",
        ]
    )

    assert result == 0
    assert captured[0].source == bundle
    assert captured[0].port == 0
    assert captured[0].device == "cpu"
    assert captured[0].decoder_threads == 3
    assert captured[0].history_limit == 12
    assert captured[0].strict is True
    assert captured[0].allowed_origins == ("http://localhost:3000",)


def test_server_health_delegates_to_the_machine_endpoint(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        preview_runner,
        "check_health",
        lambda url, *, timeout: calls.append((url, timeout)) or 0,
    )

    assert main(["server", "health", "--url", "http://127.0.0.1:8765"]) == 0
    assert calls == [("http://127.0.0.1:8765", 2.0)]


def test_studio_defaults_to_dynamic_port_and_can_remain_headless(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "library.fcpxmld"
    bundle.mkdir()
    captured = []
    monkeypatch.setattr(
        preview_runner,
        "run_studio",
        lambda config, *, open_browser: captured.append((config, open_browser)) or 0,
    )

    assert main(["studio", str(bundle), "--no-open", "--device", "cpu"]) == 0

    config, open_browser = captured[0]
    assert config.source == bundle
    assert config.port == 0
    assert config.device == "cpu"
    assert open_browser is False


def test_runner_entrypoints_keep_api_server_browser_free(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        preview_runner,
        "_run_server",
        lambda config, *, mode, open_browser: calls.append((mode, open_browser)) or 0,
    )
    config = preview_runner.ServerConfig(source=Path("Library.fcpxmld"))

    assert preview_runner.run_server(config) == 0
    assert preview_runner.run_studio(config, open_browser=False) == 0
    assert preview_runner.run_studio(config) == 0
    assert calls == [("server", False), ("studio", False), ("studio", True)]


def test_failed_browser_launch_does_not_print_the_token_url(monkeypatch, capsys) -> None:
    import webbrowser

    monkeypatch.setattr(preview_runner, "_launch_chrome_app_window", lambda _url: False)
    monkeypatch.setattr(webbrowser, "open", lambda _url: False)
    preview_runner._launch_browser("http://127.0.0.1/#runtimeToken=top-secret")

    error = capsys.readouterr().err
    assert "could not open" in error
    assert "top-secret" not in error


def test_module_entry_point_render_still_works(tmp_path: Path) -> None:
    """Compat: ``python -m bladeworks render ...`` still renders."""

    _, ffprobe = _require_render_tools()

    dest = tmp_path / "proj"
    dest.mkdir()
    assert main(["examples", "cp", "single_clip", str(dest)]) == 0
    bundle = dest / "single_clip.fcpxmld"
    output = dest / "via_module.mp4"

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # Run from the repo root so ``python -m`` resolves ``backend`` from THIS
    # checkout (its cwd goes on sys.path[0]); otherwise a sibling worktree that
    # also has a ``backend/`` package could shadow it.
    result = subprocess.run(
        (sys.executable, "-m", "bladeworks", "render",
         str(bundle), "-o", str(output), "--no-progress"),
        capture_output=True, text=True, env=env, cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert "aac" in _stream_codecs(ffprobe, output, "a")
