"""Focused tensor kernels and FCPXML integration for masks and keying."""

from __future__ import annotations

import base64
import plistlib
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from bladeworks.core.compiler import compile_fcpxml  # noqa: E402
from bladeworks.core.masks import MaskResolutionError, resolve_mask_group  # noqa: E402
from bladeworks.core.model import Keyframe, MaskSource, Parameter, ResolvedMask, ResolvedMaskGroup  # noqa: E402
from bladeworks.core.retime import RetimeMap, RetimePoint  # noqa: E402
from bladeworks.tensor.fx_keyer import GreenScreenKeyerPayload, green_screen_key  # noqa: E402
from bladeworks.tensor.fx_mask import MaskEffectPayload, apply_masked_effect, matte_for_group  # noqa: E402
from bladeworks.tensor.renderer import render_document  # noqa: E402


def _canvas(width: int = 8, height: int = 8) -> torch.Tensor:
    canvas = torch.ones(4, height, width, dtype=torch.float32)
    canvas[3] = 1.0
    return canvas


def _shape(*params: Parameter, blend: str = "add") -> ResolvedMask:
    return ResolvedMask(kind="shape", name="Shape", blend_mode=blend, params=params)


def _draw(*params: Parameter) -> ResolvedMask:
    return ResolvedMask(
        kind="draw",
        name="Draw",
        blend_mode="add",
        params=params,
        data={"points": "-2,-2;2,-2;2,2;-2,2"},
    )


def test_shape_mask_inversion_feather_and_opacity() -> None:
    mask = _shape(
        Parameter("Radius", "160", "2 2"),
        Parameter("Feather", "102", "2"),
        Parameter("Opacity", "103", "0.5"),
    )
    normal = matte_for_group(ResolvedMaskGroup((mask,), inverted=False), _canvas())
    inverted = matte_for_group(ResolvedMaskGroup((mask,), inverted=True), _canvas())
    assert 0.0 < float(normal[4, 6]) < 0.5  # feathered edge, then opacity
    assert float(normal[4, 4]) == pytest.approx(0.5)
    assert float(inverted[4, 4]) == pytest.approx(0.5)
    assert float(inverted[0, 0]) == pytest.approx(1.0)


def test_mask_combine_add_subtract_and_multiply() -> None:
    left = _shape(Parameter("Radius", "160", "3 3"))
    right = _shape(Parameter("Radius", "160", "1 1"), Parameter("Position", "201", "2 0"))
    canvas = _canvas()
    add = matte_for_group(ResolvedMaskGroup((left, right), False), canvas)
    subtract = matte_for_group(ResolvedMaskGroup((left, _shape(*right.params, blend="subtract")), False), canvas)
    multiply = matte_for_group(ResolvedMaskGroup((left, _shape(*right.params, blend="multiply")), False), canvas)
    assert float(add[4, 6]) == pytest.approx(1.0)
    assert float(subtract[4, 6]) == pytest.approx(0.0)
    assert float(multiply[4, 6]) == pytest.approx(1.0)
    assert float(multiply[0, 0]) == pytest.approx(0.0)


def test_shape_mask_samples_animation_on_the_local_effect_clock() -> None:
    position = Parameter(
        "Position",
        "201",
        "-2 0",
        keyframes=(
            Keyframe(time=0, value="-2 0", interp="linear", curve=None),
            Keyframe(time=2, value="2 0", interp="linear", curve=None),
        ),
    )
    mask = _shape(Parameter("Radius", "160", "1 1"), position)
    group = ResolvedMaskGroup((mask,), False)
    canvas = _canvas()
    first = matte_for_group(group, canvas, seconds=0.0)
    middle = matte_for_group(group, canvas, seconds=1.0)
    last = matte_for_group(group, canvas, seconds=2.0)
    assert float(first[4, 2]) == 1.0
    assert float(middle[4, 4]) == 1.0
    assert float(last[4, 6]) == 1.0


def test_mask_pixel_coordinates_scale_with_preview_canvas() -> None:
    mask = _shape(
        Parameter("Radius", "160", "4 4"),
        Parameter("Position", "201", "4 0"),
        Parameter("Feather", "102", "2"),
    )
    group = ResolvedMaskGroup((mask,), False)
    preview = matte_for_group(
        group,
        _canvas(8, 8),
        coordinate_scale_x=0.5,
        coordinate_scale_y=0.5,
    )

    # Native center x=8 plus authored position 4 becomes preview center x=4
    # plus position 2. Radius and feather shrink by the same target ratio.
    assert float(preview[4, 6]) == pytest.approx(1.0)
    assert 0.0 < float(preview[4, 4]) < 1.0
    assert float(preview[4, 2]) == pytest.approx(0.0)


def test_masked_effect_maps_local_time_to_authored_source_keyframes() -> None:
    position = Parameter(
        "Position",
        "201",
        "-2 0",
        keyframes=(
            Keyframe(time=10, value="-2 0", interp="linear", curve=None),
            Keyframe(time=12, value="2 0", interp="linear", curve=None),
        ),
    )
    payload = MaskEffectPayload(
        group=ResolvedMaskGroup((_shape(Parameter("Radius", "160", "1 1"), position),), False),
        inside="inside",
        outside=None,
        source_start=10,
        playback_rate=2,
    )
    canvas = _canvas()
    result = apply_masked_effect(
        payload,
        canvas,
        frame=0,
        seconds=0.5,
        apply_effect=lambda _spec, pixels, _frame: torch.cat(
            (torch.zeros_like(pixels[:3]), pixels[3:4]), dim=0
        ),
    )
    # local 0.5 s at 2x maps to source time 11 s, where the animated mask is centered.
    assert float(result[0, 4, 4]) == pytest.approx(0.0)
    assert float(result[0, 4, 2]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("points", "local_seconds", "expected_source_seconds"),
    (
        (((0, 12), (2, 10)), 0.5, 11.5),
        (((0, 10), (2, 10)), 1.5, 10.0),
        (((0, 10), (1, 12), (3, 13)), 2.0, 12.5),
    ),
    ids=("reverse", "freeze", "variable-rate"),
)
def test_masked_effect_uses_exact_retime_map_for_keyframe_clock(
    points: tuple[tuple[int, int], ...],
    local_seconds: float,
    expected_source_seconds: float,
) -> None:
    opacity = Parameter(
        "Opacity",
        "opacity",
        "0",
        keyframes=(
            Keyframe(time=10, value="0", interp="linear", curve=None),
            Keyframe(time=13, value="1", interp="linear", curve=None),
        ),
    )
    retime_map = RetimeMap.from_points(
        tuple(RetimePoint(Fraction(timeline), Fraction(source)) for timeline, source in points)
    )
    payload = MaskEffectPayload(
        group=ResolvedMaskGroup((_draw(opacity),), False),
        inside="inside",
        outside=None,
        retime_map=retime_map,
    )

    result = apply_masked_effect(
        payload,
        _canvas(),
        frame=0,
        seconds=local_seconds,
        apply_effect=lambda _spec, pixels, _frame: torch.cat(
            (torch.zeros_like(pixels[:3]), pixels[3:4]), dim=0
        ),
    )

    expected_opacity = (expected_source_seconds - 10.0) / 3.0
    assert float(result[0, 4, 4]) == pytest.approx(1.0 - expected_opacity)


def test_draw_color_and_luma_masks_use_explicit_payloads() -> None:
    canvas = _canvas(8, 8)
    canvas[:3, 4, 4] = torch.tensor((0.0, 1.0, 0.0))
    canvas[:3, 4, 5] = torch.tensor((0.0, 0.0, 1.0))
    draw = ResolvedMask(
        kind="draw", name="Draw", blend_mode="add", params=(),
        data={"points": "-2,-2;2,-2;2,2;-2,2"},
    )
    color = ResolvedMask(
        kind="color", name="Color", blend_mode="add", params=(),
        data={"color": "0 1 0", "tolerance": "0.01", "softness": "0", "opacity": "0.8", "luma_min": "0", "luma_max": "1"},
    )
    range_mask = ResolvedMask(
        kind="range", name="Range", blend_mode="add", params=(),
        data={"luma_min": "0.05", "luma_max": "0.2", "softness": "0", "opacity": "1"},
    )
    draw_matte = matte_for_group(ResolvedMaskGroup((draw,), False), canvas)
    color_matte = matte_for_group(ResolvedMaskGroup((color,), False), canvas)
    range_matte = matte_for_group(ResolvedMaskGroup((range_mask,), False), canvas)
    assert float(draw_matte[4, 4]) == 1.0 and float(draw_matte[0, 0]) == 0.0
    assert float(color_matte[4, 4]) == pytest.approx(0.0)
    assert float(color_matte[4, 5]) == pytest.approx(0.8)
    assert float(range_matte[4, 4]) == 0.0 and float(range_matte[4, 5]) == 1.0


def test_draw_mask_static_opacity_changes_rendered_matte_pixels() -> None:
    canvas = _canvas()
    opaque = matte_for_group(ResolvedMaskGroup((_draw(Parameter("Opacity", "opacity", "1")),), False), canvas)
    quarter = matte_for_group(ResolvedMaskGroup((_draw(Parameter("Opacity", "opacity", "0.25")),), False), canvas)

    assert float(opaque[4, 4]) == pytest.approx(1.0)
    assert float(quarter[4, 4]) == pytest.approx(0.25)
    assert float(quarter[0, 0]) == pytest.approx(0.0)


def test_draw_mask_opacity_is_piecewise_linear_on_the_source_clock() -> None:
    opacity = Parameter(
        "Opacity",
        "opacity",
        "0",
        keyframes=(
            Keyframe(time=10, value="0", interp="linear", curve=None),
            Keyframe(time=12, value="1", interp="linear", curve=None),
        ),
    )
    payload = MaskEffectPayload(
        group=ResolvedMaskGroup((_draw(opacity),), False),
        inside="inside",
        outside=None,
        source_start=10,
        playback_rate=2,
    )
    canvas = _canvas()
    result = apply_masked_effect(
        payload,
        canvas,
        frame=0,
        seconds=0.5,
        apply_effect=lambda _spec, pixels, _frame: torch.cat(
            (torch.zeros_like(pixels[:3]), pixels[3:4]), dim=0
        ),
    )

    # Local 0.5 s at 2x is source time 11 s, halfway between opacity 0 and 1.
    assert float(result[0, 4, 4]) == pytest.approx(0.5)
    assert float(result[0, 0, 0]) == pytest.approx(1.0)


@pytest.mark.parametrize("value", ("-0.01", "1.01"))
def test_draw_mask_rejects_out_of_range_static_and_animated_opacity(value: str) -> None:
    for opacity in (
        Parameter("Opacity", "opacity", value),
        Parameter(
            "Opacity",
            "opacity",
            "0.5",
            keyframes=(
                Keyframe(time=0, value="0.5", interp="linear", curve=None),
                Keyframe(time=1, value=value, interp="linear", curve=None),
            ),
        ),
    ):
        source = MaskSource(
            kind="mask-shape",
            name="Draw Mask",
            enabled=True,
            blend_mode="add",
            mask_type=None,
            tracking=None,
            params=(Parameter("Points", "points", "-2,-2;2,-2;2,2;-2,2"), opacity),
            data=None,
            raw_xml="",
        )
        with pytest.raises(MaskResolutionError, match=r"outside \[0\.0, 1\.0\]"):
            resolve_mask_group((source,), inverted=False)


@pytest.mark.parametrize(
    ("name", "key", "valid", "invalid"),
    (
        ("Radius", "160", "16 16", "-1 16"),
        ("Curvature", "159", "0.5", "1.01"),
        ("Feather", "102", "4", "-0.01"),
        ("Position", "201", "0 0", "32769 0"),
        ("Rotation", "202", "0", "3601"),
        ("Opacity", "103", "0.5", "1.01"),
        ("Falloff", "104", "1", "0.09"),
    ),
)
def test_shape_mask_rejects_out_of_range_animated_values(
    name: str,
    key: str,
    valid: str,
    invalid: str,
) -> None:
    parameter = Parameter(
        name,
        key,
        valid,
        keyframes=(
            Keyframe(time=0, value=valid, interp="linear", curve=None),
            Keyframe(time=1, value=invalid, interp="linear", curve=None),
        ),
    )
    source = MaskSource(
        kind="mask-shape",
        name="Shape Mask",
        enabled=True,
        blend_mode="add",
        mask_type=None,
        tracking=None,
        params=(parameter,),
        data=None,
        raw_xml="",
    )

    with pytest.raises(MaskResolutionError, match="mask keyframe.*outside"):
        resolve_mask_group((source,), inverted=False)


def test_keyer_has_soft_edge_green_spill_and_transparent_green() -> None:
    payload = GreenScreenKeyerPayload(
        key_color=(0.0, 1.0, 0.0), softness=4.0, strength=1.0,
        spill_level=1.0, chroma_rolloff=0.05, luma_rolloff=0.05,
        green_chroma=0.09, blue_chroma=0.09, min_green=-3.0,
        max_green=-1.7, min_blue=-1.25, max_blue=0.125, mix=1.0,
    )
    canvas = _canvas(4, 1)
    canvas[:3, 0, 0] = torch.tensor((0.0, 1.0, 0.0))
    canvas[:3, 0, 1] = torch.tensor((0.0, 0.7, 0.0))
    canvas[:3, 0, 2] = torch.tensor((0.0, 0.7, 0.2))
    canvas[:3, 0, 3] = torch.tensor((0.0, 0.0, 1.0))
    keyed = green_screen_key(canvas, payload)
    assert float(keyed[3, 0, 0]) == pytest.approx(0.0)
    assert 0.0 < float(keyed[3, 0, 1]) < 1.0
    assert float(keyed[1, 0, 2]) < float(canvas[1, 0, 2])
    assert float(keyed[2, 0, 3]) > 0.9 and float(keyed[3, 0, 3]) > 0.9


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg/ffprobe")
@pytest.mark.timeout(60)
def test_tensor_keyer_end_to_end_preserves_transparent_output(tmp_path: Path) -> None:
    pytest.importorskip("av")
    source = tmp_path / "green.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i",
        "color=c=0x00FF00:s=64x64:r=2:d=1,drawbox=x=20:y=20:w=24:h=24:color=blue:t=fill",
        "-pix_fmt", "yuv420p", str(source),
    )
    # The compiler's keyer decoder needs a bounded binary plist with the
    # NSKeyedArchiver marker.  The tensor test does not depend on Apple's
    # private archive contents beyond that reviewed contract.
    config = base64.b64encode(
        plistlib.dumps({"$archiver": "NSKeyedArchiver"}, fmt=plistlib.FMT_BINARY)
    ).decode("ascii")
    effect_data = base64.b64encode(
        b'<ozml><parameter name="Strength" value="1"/><parameter name="Key Color" value="0 1 0"/></ozml>'
    ).decode("ascii")
    fixture = tmp_path / "keyer.fcpxml"
    fixture.write_text(_fcpxml(source, config, effect_data), encoding="utf-8")
    document = compile_fcpxml(fixture).render
    output = tmp_path / "keyed.mov"
    stats = render_document(
        document, output_path=output, device="cpu", pipelined=False,
        pixel_policy="alpha", codec="prores_ks",
    )
    assert stats.frames == 2
    alpha = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(output), "-vf", "format=rgba", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
        check=True, capture_output=True,
    ).stdout[3::4]
    assert min(alpha) == 0
    assert max(alpha) > 200


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg/ffprobe")
@pytest.mark.timeout(60)
def test_tensor_shape_mask_end_to_end_keeps_outside_branch(tmp_path: Path) -> None:
    pytest.importorskip("av")
    source = tmp_path / "mask.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i",
        "color=c=red:s=64x64:r=2:d=1,drawbox=x=24:y=24:w=16:h=16:color=blue:t=fill",
        "-pix_fmt", "yuv420p", str(source),
    )
    fixture = tmp_path / "mask.fcpxml"
    fixture.write_text(_mask_fcpxml(source), encoding="utf-8")
    document = compile_fcpxml(fixture).render
    output = tmp_path / "masked.mp4"
    stats = render_document(document, output_path=output, device="cpu", pipelined=False)
    assert stats.frames == 2
    pixels = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(output), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    center = _pixel(pixels, 32, 32, 64)
    corner = _pixel(pixels, 4, 4, 64)
    assert center[0] > 150 and center[2] < 100
    assert corner[0] > 150 and corner[1] < 100 and corner[2] < 100


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _fcpxml(source: Path, config: str, effect_data: str) -> str:
    return f'''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
<resources><format id="f" frameDuration="1/2s" width="64" height="64" colorSpace="1-1-1 (Rec. 709)"/>
<asset id="a" start="0s" duration="1s" hasVideo="1" hasAudio="0" format="f"><media-rep kind="original-media" src="{source.as_uri()}"/></asset>
<effect id="k" name="Green Screen Keyer" uid="FxPlug:41122549-B8A6-470E-94DA-211294D20B62"/></resources>
<library><event name="e"><project name="p"><sequence format="f" duration="1s"><spine><asset-clip ref="a" offset="0s" start="0s" duration="1s">
<filter-video ref="k"><data key="effectConfig">{config}</data><data key="effectData">{effect_data}</data></filter-video>
</asset-clip></spine></sequence></project></event></library></fcpxml>'''


def _mask_fcpxml(source: Path) -> str:
    return f'''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
<resources><format id="f" frameDuration="1/2s" width="64" height="64" colorSpace="1-1-1 (Rec. 709)"/>
<asset id="a" start="0s" duration="1s" hasVideo="1" hasAudio="0" format="f"><media-rep kind="original-media" src="{source.as_uri()}"/></asset>
<effect id="n" name="Negative" uid=".../Effects.localized/Basics.localized/Negative.localized/Negative.moef"/></resources>
<library><event name="e"><project name="p"><sequence format="f" duration="1s"><spine><asset-clip ref="a" offset="0s" start="0s" duration="1s">
<filter-video-mask inverted="0"><mask-shape name="Shape Mask" blendMode="add"><param name="Radius" key="160" value="16 16"/><param name="Feather" key="102" value="0"/></mask-shape><filter-video ref="n"/></filter-video-mask>
</asset-clip></spine></sequence></project></event></library></fcpxml>'''


def _pixel(data: bytes, x: int, y: int, width: int) -> tuple[int, int, int]:
    offset = (y * width + x) * 3
    return data[offset], data[offset + 1], data[offset + 2]
