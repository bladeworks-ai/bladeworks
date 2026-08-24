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


def _write_alpha_metadata_project(path: Path, metadata: str) -> Path:
    path.write_text(
        f'''<fcpxml version="1.14"><resources>
        <format id="fmt" frameDuration="1/30s" width="160" height="90"/>
        <asset id="asset" hasVideo="1" hasAudio="0" format="fmt">
          <media-rep kind="original-media" src="file:///tmp/alpha.mov"/>
          <metadata>{metadata}</metadata>
        </asset>
        </resources><library><event name="Event"><project name="Project">
        <sequence format="fmt" duration="0s"><spine/></sequence>
        </project></event></library></fcpxml>''',
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", "premultiplied"),
        ("0 (Premultiply)", "premultiplied"),
        ("1", "straight"),
        ("1 (Straight)", "straight"),
        ("2", "ignore"),
        ("2 (None/Ignore Alpha)", "ignore"),
    ],
)
def test_asset_alpha_handling_metadata_is_typed(
    tmp_path: Path,
    value: str,
    expected: str,
) -> None:
    source = _write_alpha_metadata_project(
        tmp_path / "alpha-metadata.fcpxml",
        f'<md key="com.apple.proapps.studio.alphaHandling" value="{value}"/>',
    )
    assert parse_fcpxml(source).assets["asset"].alpha_handling == expected


def test_asset_without_alpha_metadata_preserves_unset_state(tmp_path: Path) -> None:
    source = _write_alpha_metadata_project(tmp_path / "alpha-unset.fcpxml", "")
    assert parse_fcpxml(source).assets["asset"].alpha_handling is None


@pytest.mark.parametrize("value", ["", "3", "straight", "1 trailing"])
def test_malformed_asset_alpha_handling_rejects(tmp_path: Path, value: str) -> None:
    source = _write_alpha_metadata_project(
        tmp_path / "alpha-malformed.fcpxml",
        f'<md key="com.apple.proapps.studio.alphaHandling" value="{value}"/>',
    )
    with pytest.raises(FCPXMLParseError, match="malformed.*alphaHandling"):
        parse_fcpxml(source)


def test_conflicting_asset_alpha_handling_rejects(tmp_path: Path) -> None:
    key = "com.apple.proapps.studio.alphaHandling"
    source = _write_alpha_metadata_project(
        tmp_path / "alpha-conflict.fcpxml",
        f'<md key="{key}" value="0"/><md key="{key}" value="1"/>',
    )
    with pytest.raises(FCPXMLParseError, match="conflicting.*alphaHandling"):
        parse_fcpxml(source)
