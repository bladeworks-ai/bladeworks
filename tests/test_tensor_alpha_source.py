"""Portable alpha-source acceptance tests shipped with standalone Bladeworks.

Architecture map
----------------
    generated ProRes 4444 fixture
        -> probe and ClipDecoder: planar YUVA becomes straight RGBA
        -> FCPXML compiler: alphaHandling becomes a typed RenderClip field
        -> tensor plan: records alpha coverage and hands the mode to the decoder

These tests deliberately use only public compiler/tensor modules and local
FFmpeg fixtures so the release exporter can run them outside Spellshot.
"""

from __future__ import annotations

import shutil
import subprocess
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("av")

from bladeworks.core.compiler import compile_fcpxml  # noqa: E402
from bladeworks.cli import main as cli_main  # noqa: E402
from bladeworks.tensor import build_tensor_plan  # noqa: E402
from bladeworks.tensor.decode import ClipDecoder, check_source_color, probe_video  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="needs ffmpeg/ffprobe")

WIDTH, HEIGHT = 96, 54


def _alpha_media(directory: Path) -> Path:
    media = directory / "source-alpha.mov"
    graph = (
        f"nullsrc=s={WIDTH}x{HEIGHT}:r=30:d=0.2,format=yuva444p10le,"
        "geq=lum='384+X*256/W':cb='448':cr='576':a='1023*X/(W-1)'"
    )
    subprocess.run(
        (
            FFMPEG,
            "-v", "error", "-y", "-f", "lavfi", "-i", graph,
            "-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            str(media),
        ),
        check=True,
    )
    return media


def _project(directory: Path, media: Path, alpha_value: str | None) -> Path:
    metadata = (
        ""
        if alpha_value is None
        else (
            '<metadata><md key="com.apple.proapps.studio.alphaHandling" '
            f'value="{alpha_value}"/></metadata>'
        )
    )
    source = directory / f"alpha-{alpha_value or 'default'}.fcpxml"
    source.write_text(
        f'''<fcpxml version="1.14"><resources>
        <format id="fmt" frameDuration="1/30s" width="{WIDTH}" height="{HEIGHT}" colorSpace="1-1-1 (Rec. 709)"/>
        <asset id="asset" start="0s" duration="1/5s" hasVideo="1" hasAudio="0" format="fmt">
          <media-rep kind="original-media" src="{media.as_uri()}"/>{metadata}
        </asset></resources><library><event name="Alpha"><project name="Alpha">
        <sequence format="fmt" duration="1/5s"><spine>
          <asset-clip ref="asset" offset="0s" start="0s" duration="1/5s"/>
        </spine></sequence></project></event></library></fcpxml>''',
        encoding="utf-8",
    )
    return source


def test_prores4444_decodes_as_rgba_with_full_range_alpha(tmp_path: Path) -> None:
    media = _alpha_media(tmp_path)
    color = check_source_color(probe_video(media), subject="ProRes 4444 fixture")
    assert color.pixel_format == "yuva444p12le"
    assert color.has_alpha

    decoder = ClipDecoder(media, device=torch.device("cpu"))
    try:
        frame = decoder.frame_at(Fraction(0))
    finally:
        decoder.close()
    assert frame.shape == (4, HEIGHT, WIDTH)
    assert float(frame[3].min()) == pytest.approx(0.0)
    assert float(frame[3].max()) == pytest.approx(1.0)
    assert np.all(np.diff(frame[3, HEIGHT // 2].numpy()) >= 0)


@pytest.mark.parametrize(
    ("authored", "expected", "channels"),
    [
        (None, None, 4),
        ("0", "premultiplied", 4),
        ("1", "straight", 4),
        ("2", "ignore", 3),
    ],
)
def test_fcpxml_alpha_handling_controls_tensor_decode(
    tmp_path: Path,
    authored: str | None,
    expected: str | None,
    channels: int,
) -> None:
    compiled = compile_fcpxml(_project(tmp_path, _alpha_media(tmp_path), authored))
    plan = build_tensor_plan(compiled.render)
    layer = plan.layers[0]
    assert layer.source_has_alpha
    assert layer.alpha_handling == expected

    decoder = ClipDecoder(
        layer.media_path,
        device=torch.device("cpu"),
        alpha_handling=layer.alpha_handling,
    )
    try:
        frame = decoder.frame_at(Fraction(0))
    finally:
        decoder.close()
    assert frame.shape == (channels, HEIGHT, WIDTH)


def test_preview_downscale_keeps_alpha_plane(tmp_path: Path) -> None:
    decoder = ClipDecoder(
        _alpha_media(tmp_path),
        device=torch.device("cpu"),
        decode_size=(WIDTH // 2, HEIGHT // 2),
    )
    try:
        frame = decoder.frame_at(Fraction(0))
    finally:
        decoder.close()
    assert frame.shape == (4, HEIGHT // 2, WIDTH // 2)
    assert float(frame[3].min()) < 0.02
    assert float(frame[3].max()) > 0.98


def test_cli_renders_alpha_source_and_records_default_interpretation(tmp_path: Path) -> None:
    source = _project(tmp_path, _alpha_media(tmp_path), None)
    output = tmp_path / "alpha-source.mp4"
    assert cli_main(
        [
            "render",
            str(source),
            "--output",
            str(output),
            "--backend",
            "tensor",
            "--no-progress",
        ]
    ) == 0
    assert output.is_file()
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    alpha_sources = manifest["render_backend"]["alpha_sources"]
    assert alpha_sources == [
        {
            "path": "spine/asset-clip[1]",
            "handling": "straight",
            "authored_override": False,
        }
    ]
