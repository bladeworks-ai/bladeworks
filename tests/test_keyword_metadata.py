"""FCPXML range metadata must not become drawable or audible story content.

Architecture map
================

asset-clip with a timed ``keyword`` marker
    -> marker retained in the owner's raw XML
    -> no child ``StoryNode``, group scope, audio item, or omission finding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bladeworks.core.compiler import compile_fcpxml
from bladeworks.core.parser import parse_fcpxml


FIXTURE = Path(__file__).with_name("fixtures") / "keyword_metadata_1_14.fcpxml"


@pytest.mark.parametrize(
    "tag, raw_fragment",
    [
        ("keyword", '<keyword start="1/2s" duration="1s" value="Favorite"'),
        ("marker", '<marker start="1/2s" duration="1/30s" value="Beat"'),
        ("chapter-marker", '<chapter-marker start="1s" duration="1/30s" value="Chapter"'),
        ("todo-marker", '<todo-marker start="3/2s" duration="1/30s" value="Review" completed="0"'),
        ("analysis-marker", '<analysis-marker start="0s" duration="1s"'),
        ("rating", '<rating start="0s" duration="2s" value="favorite"'),
    ],
)
def test_range_metadata_is_preserved_but_not_parsed_as_a_story_child(
    tag: str,
    raw_fragment: str,
) -> None:
    source = parse_fcpxml(FIXTURE)
    clip = source.spine[0]

    assert clip.children == ()
    assert raw_fragment in clip.raw_xml, tag


def test_keyword_range_creates_no_render_scope_audio_item_or_omission() -> None:
    compiled = compile_fcpxml(FIXTURE)

    assert len(compiled.render.clips) == 1
    assert compiled.render.group_scopes == ()
    assert compiled.render.audio is not None
    assert [item.path for item in compiled.render.audio.items] == ["spine/asset-clip[1]"]
    for metadata_tag in (
        "keyword",
        "marker",
        "chapter-marker",
        "todo-marker",
        "analysis-marker",
        "rating",
    ):
        assert not any(
            metadata_tag in finding.construct.casefold()
            or metadata_tag in finding.fcpxml_path.casefold()
            for finding in compiled.report.findings
        )
