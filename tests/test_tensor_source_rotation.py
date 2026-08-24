"""Tensor source display-rotation fixtures.

Architecture map
================

    asymmetric encoded movie
        -> FFmpeg display matrix 0 / 90 / 180 / 270
        -> ffprobe / tensor plan records encoded and displayed dimensions
        -> tensor ``apply_display_rotation`` before crop, conform, and transform
        -> ordinary geometry sees the displayed raster

The quarter-turn fixture first compares the tensor orientation operation with
FFmpeg autorotate byte-for-byte. The integration fixture then renders a rotated
source with crop plus authored transform through both complete backends. This
separates source metadata from FCPXML ``adjust-transform`` rotation and catches
double application.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("av")

from bladeworks.core.compiler import compile_fcpxml  # noqa: E402
from bladeworks.tensor import build_tensor_plan, render_document  # noqa: E402
from bladeworks.tensor.decode import probe_video  # noqa: E402
from bladeworks.tensor.sampler import apply_display_rotation  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg")


def _base_movie(path: Path) -> Path:
    subprocess.run(
        (
            FFMPEG, "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=s=160x96:r=30:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ),
        check=True,
    )
    return path


def _with_rotation(base: Path, output: Path, degrees: int) -> Path:
    if degrees == 0:
        shutil.copyfile(base, output)
        return output
    subprocess.run(
        (
            FFMPEG, "-v", "error", "-y", "-display_rotation", str(degrees),
            "-i", str(base), "-c", "copy", str(output),
        ),
        check=True,
    )
    return output


def _raw_frame(path: Path, *, autorotate: bool, width: int, height: int) -> np.ndarray:
    command = [FFMPEG, "-v", "error"]
    if not autorotate:
        command.append("-noautorotate")
    command.extend(("-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"))
    result = subprocess.run(command, check=True, capture_output=True)
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3).copy()


@pytest.mark.parametrize("degrees", (0, 90, 180, 270))
def test_quarter_turn_matches_ffmpeg_autorotate(tmp_path: Path, degrees: int) -> None:
    base = _base_movie(tmp_path / "base.mov")
    media = _with_rotation(base, tmp_path / f"r{degrees}.mov", degrees)
    probe = probe_video(media)
    assert probe.rotation_degrees == degrees
    displayed_width, displayed_height = ((96, 160) if degrees in {90, 270} else (160, 96))
    raw = _raw_frame(media, autorotate=False, width=160, height=96)
    reference = _raw_frame(
        media,
        autorotate=True,
        width=displayed_width,
        height=displayed_height,
    )
    source = torch.from_numpy(raw).permute(2, 0, 1)
    observed = apply_display_rotation(source, degrees).permute(1, 2, 0).numpy()
    assert np.array_equal(observed, reference)


def _project(
    path: Path,
    media: Path,
    *,
    inner: str = "",
    spine: str | None = None,
    extra_resources: str = "",
) -> Path:
    if spine is None:
        spine = f'<asset-clip ref="src" offset="0s" start="0s" duration="1s" format="srcfmt">{inner}</asset-clip>'
    path.write_text(
        f'''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
<resources>
  <format id="fmt" frameDuration="1/30s" width="320" height="180" colorSpace="1-1-1 (Rec. 709)"/>
  <format id="srcfmt" frameDuration="1/30s" width="160" height="96" colorSpace="1-1-1 (Rec. 709)"/>
  <asset id="src" start="0s" duration="1s" hasVideo="1" hasAudio="0" format="srcfmt"><media-rep kind="original-media" src="{media.as_uri()}"/></asset>
  {extra_resources}
</resources>
<library><event name="e"><project name="rotation"><sequence format="fmt" duration="1s"><spine>
  {spine}
</spine></sequence></project></event></library></fcpxml>''',
        encoding="utf-8",
    )
    return path


def _ssim_score(observed: Path, reference: Path) -> float:
    comparison = subprocess.run(
        (
            FFMPEG, "-v", "info", "-i", str(observed), "-i", str(reference),
            "-lavfi", "ssim", "-f", "null", "-",
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    return float(comparison.stderr.rsplit("All:", 1)[1].split()[0])


def test_rotated_source_stays_independent_of_authored_geometry(tmp_path: Path) -> None:
    media = _with_rotation(_base_movie(tmp_path / "base.mov"), tmp_path / "rotated.mov", 90)
    authored = (
        '<adjust-crop mode="trim"><trim-rect left="3" top="4" right="5" bottom="6"/></adjust-crop>'
        '<adjust-transform position="12 -7" scale="0.82 0.82" rotation="17"/>'
    )
    compiled = compile_fcpxml(_project(tmp_path / "rotation.fcpxml", media, inner=authored))
    plan = build_tensor_plan(compiled.render)
    layer = plan.layers[0]
    assert layer.source_rotation_degrees == 90
    assert (layer.frame.source_width, layer.frame.source_height) == (96, 160)
    # The authored transform remains a separate GeometryPlan stage after source
    # orientation; rendering it proves crop dimensions are based on 96x160.
    authored_output = tmp_path / "authored.tensor.mp4"
    assert render_document(
        compiled.render,
        output_path=authored_output,
        plan=plan,
        device="cpu",
    ).frames == 30

    # The legacy graph does not yet own display metadata, so use an independent
    # FFmpeg autorotate + Fit oracle for the pixel comparison.
    plain = compile_fcpxml(_project(tmp_path / "rotation-fit.fcpxml", media))
    output = tmp_path / "fit.tensor.mp4"
    stats = render_document(plain.render, output_path=output, device="cpu")
    assert stats.frames == 30
    reference = tmp_path / "fit.reference.mp4"
    subprocess.run(
        (
            FFMPEG, "-v", "error", "-y", "-i", str(media),
            "-vf", "scale=108:180:flags=bilinear,pad=320:180:106:0:black",
            "-frames:v", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(reference),
        ),
        check=True,
    )

    assert _ssim_score(output, reference) >= 0.98


def test_rotated_source_composes_with_fill(tmp_path: Path) -> None:
    media = _with_rotation(_base_movie(tmp_path / "base.mov"), tmp_path / "rotated.mov", 90)
    compiled = compile_fcpxml(
        _project(
            tmp_path / "rotation-fill.fcpxml",
            media,
            inner='<adjust-conform type="fill"/>',
        )
    )
    plan = build_tensor_plan(compiled.render)
    (layer,) = plan.layers
    assert layer.source_rotation_degrees == 90
    assert (layer.frame.source_width, layer.frame.source_height) == (96, 160)
    assert layer.conform == "fill"

    output = tmp_path / "fill.tensor.mp4"
    assert render_document(compiled.render, output_path=output, plan=plan, device="cpu").frames == 30
    reference = tmp_path / "fill.reference.mp4"
    subprocess.run(
        (
            FFMPEG, "-v", "error", "-y", "-i", str(media),
            "-vf", "scale=320:534:flags=bilinear,crop=320:180:0:177",
            "-frames:v", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(reference),
        ),
        check=True,
    )
    # This oracle uses FFmpeg's encoded-domain scaler while tensor geometry
    # resamples in linear light. The gate is intentionally below the Fit case,
    # but still catches a missed quarter-turn or Fit mistakenly used as Fill.
    assert _ssim_score(output, reference) >= 0.94


def test_rotated_source_composes_with_animated_authored_transform(tmp_path: Path) -> None:
    media = _with_rotation(_base_movie(tmp_path / "base.mov"), tmp_path / "rotated.mov", 270)
    animated = '''<adjust-transform position="0 0" scale="1 1" rotation="0" anchor="0 0">
      <param name="position"><keyframeAnimation>
        <keyframe time="0s" value="-18 9" curve="linear"/>
        <keyframe time="1/2s" value="12 -7" curve="linear"/>
        <keyframe time="29/30s" value="20 11" curve="linear"/>
      </keyframeAnimation></param>
      <param name="scale"><keyframeAnimation>
        <keyframe time="0s" value="0.8 0.8" curve="linear"/>
        <keyframe time="1/2s" value="1.05 0.9" curve="linear"/>
        <keyframe time="29/30s" value="0.9 1.1" curve="linear"/>
      </keyframeAnimation></param>
      <param name="rotation"><keyframeAnimation>
        <keyframe time="0s" value="-12"/>
        <keyframe time="1/2s" value="8"/>
        <keyframe time="29/30s" value="19"/>
      </keyframeAnimation></param>
    </adjust-transform>'''
    compiled = compile_fcpxml(_project(tmp_path / "rotation-animated.fcpxml", media, inner=animated))
    plan = build_tensor_plan(compiled.render)
    (layer,) = plan.layers
    assert layer.source_rotation_degrees == 270
    assert (layer.frame.source_width, layer.frame.source_height) == (96, 160)
    assert not layer.is_static
    first = layer.geometry_at(0, plan.frame_duration).transform
    middle = layer.geometry_at(15, plan.frame_duration).transform
    last = layer.geometry_at(29, plan.frame_duration).transform
    assert first != middle
    assert middle != last
    assert render_document(
        compiled.render,
        output_path=tmp_path / "animated.tensor.mp4",
        plan=plan,
        device="cpu",
    ).frames == 30


def test_rotated_source_renders_inside_nested_compound(tmp_path: Path) -> None:
    media = _with_rotation(_base_movie(tmp_path / "base.mov"), tmp_path / "rotated.mov", 90)
    compound = '''<media id="cmp" name="rotated compound"><sequence format="fmt" duration="1s"><spine>
      <asset-clip ref="src" name="rotated leaf" offset="0s" start="0s" duration="1s" format="srcfmt">
        <adjust-conform type="fill"/>
      </asset-clip>
    </spine></sequence></media>'''
    spine = '''<ref-clip ref="cmp" name="compound" offset="0s" start="0s" duration="1s">
      <adjust-transform position="7 -4" scale="0.92 0.92" rotation="5"/>
    </ref-clip>'''
    compiled = compile_fcpxml(
        _project(
            tmp_path / "rotation-compound.fcpxml",
            media,
            spine=spine,
            extra_resources=compound,
        )
    )
    plan = build_tensor_plan(compiled.render)
    (layer,) = plan.layers
    assert layer.source_rotation_degrees == 90
    assert (layer.frame.source_width, layer.frame.source_height) == (96, 160)
    assert layer.nearest_scope_id is not None
    assert len(plan.scopes) == 1
    assert render_document(
        compiled.render,
        output_path=tmp_path / "compound.tensor.mp4",
        plan=plan,
        device="cpu",
    ).frames == 30
