"""Contracts for the common file/compound/multicam clip-instance clock."""

from dataclasses import replace
from fractions import Fraction

import pytest

from bladeworks.core.model import StoryNode, TimeMapPoint
from bladeworks.core.render_sources import (
    RenderableAVInstance,
    RenderableAVSource,
    StreamTimingCoverageError,
    resolve_instance_stream_timing,
)


def _node(**changes) -> StoryNode:
    node = StoryNode(
        kind="mc-clip",
        path="spine/mc-clip[1]",
        name="Source instance",
        ref="mc",
        lane=0,
        offset=Fraction(5),
        start=Fraction(10),
        duration=Fraction(2),
        enabled=True,
        src_enable="all",
        audio_start=None,
        audio_duration=None,
        role=None,
        video_role=None,
        audio_role=None,
        conform_type="fit",
        transform=None,
        crop=None,
        blend_opacity=1.0,
        blend_mode=None,
        opacity_fade=None,
        volume_db=None,
        audio_fade=None,
        time_map=(),
        time_map_preserves_pitch=True,
        time_map_frame_sampling=None,
        filters=(),
        params=(),
        text_runs=(),
        text_styles={},
        multicam_sources=(),
        children=(),
        raw_xml='<mc-clip ref="mc"/>',
    )
    return replace(node, **changes)


def _point(time: int, value: int) -> TimeMapPoint:
    return TimeMapPoint(Fraction(time), Fraction(value), "linear")


def test_one_x_split_audio_reduces_to_existing_j_l_mapping() -> None:
    timing = resolve_instance_stream_timing(
        _node(audio_start=Fraction(9), audio_duration=Fraction(4)),
        absolute_start=Fraction(5),
        stream="audio",
    )
    assert timing.absolute_start == 4
    assert timing.duration == 4
    assert timing.source_start == 9
    assert timing.source_duration == 4


def test_retimed_audio_range_is_sliced_from_same_instance_map() -> None:
    node = _node(
        time_map=(_point(0, 8), _point(1, 10), _point(3, 14)),
        audio_start=Fraction(9),
        audio_duration=Fraction(4),
    )
    timing = resolve_instance_stream_timing(
        node,
        absolute_start=Fraction(5),
        stream="audio",
    )
    assert timing.absolute_start == Fraction(11, 2)
    assert timing.duration == 2
    assert timing.source_start == 9
    assert timing.source_duration == 4
    assert [segment.rate for segment in timing.retime_map.segments] == [2, 2]


def test_reverse_and_freeze_segments_survive_audio_range_composition() -> None:
    node = _node(
        duration=Fraction(3),
        time_map=(
            _point(0, 12),
            _point(1, 10),
            _point(2, 10),
            _point(3, 8),
        ),
        audio_start=Fraction(8),
        audio_duration=Fraction(4),
    )
    timing = resolve_instance_stream_timing(
        node,
        absolute_start=Fraction(0),
        stream="audio",
    )
    assert [segment.kind for segment in timing.retime_map.segments] == [
        "reverse",
        "freeze",
        "reverse",
    ]


def test_uncovered_split_audio_range_fails_without_extrapolation() -> None:
    node = _node(
        time_map=(_point(0, 10), _point(2, 14)),
        audio_start=Fraction(9),
        audio_duration=Fraction(6),
    )
    with pytest.raises(StreamTimingCoverageError, match="not fully covered"):
        resolve_instance_stream_timing(
            node,
            absolute_start=Fraction(0),
            stream="audio",
        )


@pytest.mark.parametrize(
    ("points", "expected_kinds", "expected_rates"),
    [
        ((10, 10, 12, 12), ["forward"], [Fraction(1)]),
        ((10, 10, 12, 14), ["forward"], [Fraction(2)]),
        ((10, 10, 12, 11), ["forward"], [Fraction(1, 2)]),
        ((10, 12, 12, 10), ["reverse"], [Fraction(-1)]),
        ((10, 11, 12, 11), ["freeze"], [Fraction(0)]),
        (
            (10, 10, 11, 11, 12, 14),
            ["forward", "forward"],
            [Fraction(1), Fraction(3)],
        ),
    ],
)
def test_file_compound_and_multicam_instances_share_every_retime_shape(
    points: tuple[int, ...],
    expected_kinds: list[str],
    expected_rates: list[Fraction],
) -> None:
    time_map = tuple(
        _point(points[index], points[index + 1])
        for index in range(0, len(points), 2)
    )
    node = _node(start=Fraction(10), time_map=time_map)
    sources = tuple(
        RenderableAVSource(
            id=f"{kind}:source",
            kind=kind,
            resource_id="source",
            source_start=Fraction(0),
            duration=Fraction(20),
            format_context=None,
            has_video=True,
            has_audio=True,
        )
        for kind in ("file", "compound", "multicam")
    )
    instances = tuple(
        RenderableAVInstance(
            path=f"spine/{source.kind}[1]",
            source=source,
            video=resolve_instance_stream_timing(
                node,
                absolute_start=Fraction(5),
                stream="video",
            ),
            audio=resolve_instance_stream_timing(
                node,
                absolute_start=Fraction(5),
                stream="audio",
            ),
        )
        for source in sources
    )
    for instance in instances:
        assert instance.video is not None
        assert instance.audio is not None
        assert instance.video.absolute_start == 5
        assert instance.video.retime_map == instance.audio.retime_map
        assert [segment.kind for segment in instance.video.retime_map.segments] == expected_kinds
        assert [segment.rate for segment in instance.video.retime_map.segments] == expected_rates
