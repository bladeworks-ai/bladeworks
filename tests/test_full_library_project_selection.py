"""Project-selection contracts for complete Final Cut library exports.

Architecture map
================

1. Build a library with projects spread across multiple events.
2. Verify exact name and UID selection at the parser and compiler boundaries.
3. Verify ambiguous or missing selectors fail with actionable errors.

These tests keep project selection at ingest time. Unselected timelines never
enter the source model or compiler, while the shared resource table remains
available to the chosen project.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bladeworks.cli import build_parser
from bladeworks.core.compiler import compile_fcpxml
from bladeworks.core.errors import FCPXMLParseError
from bladeworks.core.parser import parse_fcpxml


def _project(name: str, uid: str, asset_id: str) -> str:
    return f'''<project name="{name}" uid="{uid}">
      <sequence format="r1" duration="1s"><spine>
        <asset-clip ref="{asset_id}" offset="0s" start="0s" duration="1s"/>
      </spine></sequence>
    </project>'''


def _full_library() -> str:
    return f'''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
      <resources>
        <format id="r1" frameDuration="1/30s" width="320" height="180" colorSpace="1-1-1 (Rec. 709)"/>
        <asset id="a1" start="0s" duration="1s" hasVideo="1" format="r1">
          <media-rep kind="original-media" src="file:///tmp/alpha.mov"/>
        </asset>
        <asset id="a2" start="0s" duration="1s" hasVideo="1" format="r1">
          <media-rep kind="original-media" src="file:///tmp/beta.mov"/>
        </asset>
      </resources>
      <library name="Complete Library">
        <event name="First Event">{_project("Shared Name", "UID-ALPHA", "a1")}</event>
        <event name="Second Event">
          {_project("Chosen Project", "UID-BETA", "a2")}
          {_project("Shared Name", "UID-GAMMA", "a2")}
        </event>
      </library>
    </fcpxml>'''


@pytest.fixture
def full_library(tmp_path: Path) -> Path:
    path = tmp_path / "full-library.fcpxml"
    path.write_text(_full_library(), encoding="utf-8")
    return path


def test_parser_selects_project_by_exact_name_across_events(full_library: Path) -> None:
    source = parse_fcpxml(full_library, project="Chosen Project")

    assert source.project_name == "Chosen Project"
    assert source.event_name == "Second Event"
    assert source.spine[0].ref == "a2"


def test_compiler_accepts_stable_project_uid(full_library: Path) -> None:
    compiled = compile_fcpxml(full_library, project="UID-BETA")

    assert compiled.render.project_name == "Chosen Project"
    assert compiled.source.event_name == "Second Event"


def test_multiple_projects_require_an_explicit_selector(full_library: Path) -> None:
    with pytest.raises(FCPXMLParseError, match=r"contains 3 projects; select one with --project"):
        parse_fcpxml(full_library)


def test_duplicate_project_name_requires_uid(full_library: Path) -> None:
    with pytest.raises(FCPXMLParseError, match=r"matched 2 projects; use a project UID"):
        parse_fcpxml(full_library, project="Shared Name")


def test_missing_project_lists_available_choices(full_library: Path) -> None:
    with pytest.raises(FCPXMLParseError, match=r"'Missing' was not found; available projects"):
        parse_fcpxml(full_library, project="Missing")


@pytest.mark.parametrize("command", ["inspect", "render"])
def test_cli_exposes_project_selector(command: str) -> None:
    parser = build_parser()
    extra = ["--output", "output.mp4"] if command == "render" else []

    args = parser.parse_args([command, "library.fcpxml", "--project", "Chosen Project", *extra])

    assert args.project == "Chosen Project"
