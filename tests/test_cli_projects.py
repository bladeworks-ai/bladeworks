"""``bladeworks projects`` -- browse the projects a library/bundle contains.

What this covers
----------------
``render``/``inspect`` take ``--project NAME_OR_UID``, but the only way to see
the available names/UIDs used to be triggering the parser's ambiguity error.
The ``projects`` subcommand (``bladeworks.cli:_run_projects``) is the
non-error browse view. This locks down:

1. A multi-event / multi-project library lists EVERY project with its name and
   UID, grouped by library -> event, and exits 0.
2. The packaged single-project ``single_clip.fcpxmld`` bundle resolves its
   ``Info.fcpxml`` and lists its one project, exit 0.
3. A document with a library/event but NO project fails LOUDLY (non-zero, the
   parser's exact message) -- no silent fallback.
4. ``--json`` emits a machine-readable ``[{event, name, uid}, ...]`` array.

These tests are pure XML parsing (no ffmpeg/torch), so they always run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bladeworks.cli import main

# The packaged single-project bundle lives beside the fcpxml package. This file
# is at backend/render/fcpxml/experimental_tests/core/, so parents[2] is the
# fcpxml package root.
_FCPXML_PKG = Path(__file__).resolve().parents[1] / "src" / "bladeworks"
_SINGLE_CLIP_BUNDLE = _FCPXML_PKG / "examples" / "single_clip.fcpxmld"


def _multi_project_library() -> str:
    """A library with two events and three projects (one name-less, for the
    'Untitled Project' fallback)."""

    return '''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
      <resources>
        <format id="r1" frameDuration="1/30s" width="320" height="180" colorSpace="1-1-1 (Rec. 709)"/>
        <asset id="a1" start="0s" duration="1s" hasVideo="1" format="r1">
          <media-rep kind="original-media" src="file:///tmp/a.mov"/>
        </asset>
      </resources>
      <library name="Complete Library">
        <event name="First Event">
          <project name="Alpha" uid="UID-ALPHA"><sequence format="r1" duration="1s"><spine>
            <asset-clip ref="a1" offset="0s" start="0s" duration="1s"/>
          </spine></sequence></project>
        </event>
        <event name="Second Event">
          <project name="Beta" uid="UID-BETA"><sequence format="r1" duration="1s"><spine>
            <asset-clip ref="a1" offset="0s" start="0s" duration="1s"/>
          </spine></sequence></project>
          <project uid="UID-GAMMA"><sequence format="r1" duration="1s"><spine>
            <asset-clip ref="a1" offset="0s" start="0s" duration="1s"/>
          </spine></sequence></project>
        </event>
      </library>
    </fcpxml>'''


def _empty_library() -> str:
    """A valid document whose library event holds NO project."""

    return '''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
      <resources>
        <format id="r1" frameDuration="1/30s" width="320" height="180" colorSpace="1-1-1 (Rec. 709)"/>
      </resources>
      <library name="Empty Library"><event name="Empty Event"/></library>
    </fcpxml>'''


def test_projects_lists_every_project_in_a_multi_project_library(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """All three projects list with name + UID, grouped by library -> event."""

    path = tmp_path / "multi.fcpxml"
    path.write_text(_multi_project_library(), encoding="utf-8")

    rc = main(["projects", str(path)])
    assert rc == 0

    out = capsys.readouterr().out
    # Every project (name + uid) appears; the name-less one uses the fallback.
    assert "Alpha [UID-ALPHA]" in out
    assert "Beta [UID-BETA]" in out
    assert "Untitled Project [UID-GAMMA]" in out
    # Grouped by library -> event.
    assert "library: Complete Library" in out
    assert "event: First Event" in out
    assert "event: Second Event" in out
    # Ends with the copy-pasteable render hint.
    assert "bladeworks render" in out
    assert "--project NAME_OR_UID" in out


def test_projects_json_emits_event_name_uid_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--json`` yields one {event, name, uid} record per project."""

    path = tmp_path / "multi.fcpxml"
    path.write_text(_multi_project_library(), encoding="utf-8")

    rc = main(["projects", str(path), "--json"])
    assert rc == 0

    records = json.loads(capsys.readouterr().out)
    assert records == [
        {"event": "First Event", "name": "Alpha", "uid": "UID-ALPHA"},
        {"event": "Second Event", "name": "Beta", "uid": "UID-BETA"},
        {"event": "Second Event", "name": None, "uid": "UID-GAMMA"},
    ]


def test_projects_lists_single_clip_bundle(capsys: pytest.CaptureFixture[str]) -> None:
    """A ``.fcpxmld`` bundle resolves its Info.fcpxml and lists its one project."""

    assert (_SINGLE_CLIP_BUNDLE / "Info.fcpxml").is_file(), "fixture bundle missing"

    rc = main(["projects", str(_SINGLE_CLIP_BUNDLE)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "single_clip" in out
    assert "bladeworks render" in out


def test_projects_empty_document_fails_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No project inside a library event -> non-zero + the parser's message."""

    path = tmp_path / "empty.fcpxml"
    path.write_text(_empty_library(), encoding="utf-8")

    rc = main(["projects", str(path)])
    assert rc == 1

    err = capsys.readouterr().err
    assert "does not contain a project inside a library event" in err
