"""Sequence-duration consistency checks at the FCPXML parser boundary."""

from pathlib import Path

import pytest

from bladeworks.core.errors import FCPXMLParseError
from bladeworks.core.parser import parse_fcpxml


def _write_project(path: Path, spine: str) -> Path:
    path.write_text(
        f'''<fcpxml version="1.14"><resources>
        <format id="fmt" frameDuration="1/30s" width="160" height="90"/>
        </resources><library><event name="Event"><project name="Project">
        <sequence format="fmt" duration="0s"><spine>{spine}</spine></sequence>
        </project></event></library></fcpxml>''',
        encoding="utf-8",
    )
    return path


def test_zero_duration_sequence_accepts_an_empty_spine(tmp_path: Path) -> None:
    document = parse_fcpxml(_write_project(tmp_path / "empty.fcpxml", ""))
    assert document.sequence_duration == 0
    assert document.spine == ()


def test_zero_duration_sequence_rejects_timeline_items(tmp_path: Path) -> None:
    source = _write_project(tmp_path / "inconsistent.fcpxml", '<gap offset="0s" duration="1s"/>')
    with pytest.raises(FCPXMLParseError, match="zero-duration sequence must have an empty spine"):
        parse_fcpxml(source)
