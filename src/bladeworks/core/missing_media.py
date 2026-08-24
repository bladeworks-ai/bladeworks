"""Create the visible source raster used when authored media is offline.

Architecture map
================

``RenderClip.missing_media_locators``
    -> deterministic full-canvas RGBA placeholder
    -> existing runtime-raster map
    -> ordinary conform, crop, transform, effects, transitions, and compositing

The placeholder is source media, not a final-frame overlay. This preserves the
authored clip geometry and timing across seek, scan, and export.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image, ImageDraw, ImageFont

from .model import RenderClip, RenderDocument, RenderVideoDisposition
from .text import RuntimeRasterResolution


def missing_media_basename(locator: str) -> str:
    """Return a safe user-facing basename without changing the locator."""

    parsed = urlparse(locator)
    candidate = unquote(parsed.path) if parsed.path else locator
    name = Path(candidate).name
    return name or locator


def resolve_missing_media_raster(
    clip: RenderClip,
    document: RenderDocument,
    *,
    work_dir: Path,
) -> RuntimeRasterResolution:
    """Materialize one deterministic placeholder for an offline video clip.

    Main callers:
    - ``executor.execute_render`` beside title and generator rasterization.

    Why this exists:
    Transparent omission hides both the editorial timing and the reason pixels
    are absent. A normal raster source keeps downstream renderer behavior
    unchanged while making the broken reference obvious in every output mode.
    """

    if not clip.missing_media_locators:
        raise ValueError(f"{clip.path} is not marked as missing media")
    if document.width <= 0 or document.height <= 0:
        raise ValueError("missing-media placeholder requires a positive canvas")

    width = document.width
    height = document.height
    image = Image.new("RGBA", (width, height), (75, 8, 14, 255))
    draw = ImageDraw.Draw(image)

    tile = max(18, min(width, height) // 24)
    alternate = (96, 13, 21, 255)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if ((x // tile) + (y // tile)) % 2:
                draw.rectangle((x, y, min(x + tile, width), min(y + tile, height)), fill=alternate)

    center_x = width // 2
    triangle_size = max(32, min(width, height) // 7)
    triangle_top = max(16, height // 2 - triangle_size)
    triangle = (
        (center_x, triangle_top),
        (center_x - triangle_size, triangle_top + triangle_size * 2),
        (center_x + triangle_size, triangle_top + triangle_size * 2),
    )
    draw.polygon(triangle, fill=(246, 196, 43, 255), outline=(30, 24, 8, 255), width=max(2, tile // 8))
    mark_font = _font(max(24, triangle_size))
    _centered_text(draw, "!", center_x, triangle_top + triangle_size, mark_font, fill=(30, 24, 8, 255))

    title_font = _font(max(24, min(width, height) // 18))
    detail_font = _font(max(16, min(width, height) // 30))
    title_y = min(height - 56, triangle_top + triangle_size * 2 + tile)
    _centered_text(draw, "Missing Media", center_x, title_y, title_font, fill=(255, 244, 218, 255))
    basename = missing_media_basename(clip.missing_media_locators[0])
    _centered_text(
        draw,
        basename,
        center_x,
        min(height - 24, title_y + max(30, min(width, height) // 14)),
        detail_font,
        fill=(246, 196, 43, 255),
    )

    path = work_dir / f"{clip.id}-missing-media.png"
    image.save(path)
    return RuntimeRasterResolution(
        clip_id=clip.id,
        image_path=path,
        video_disposition=RenderVideoDisposition(execution="composite"),
    )


def _font(size: int) -> ImageFont.ImageFont:
    """Load Pillow's bundled sans face, with an explicit minimal fallback."""

    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    center_y: int,
    font: ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int, int],
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (center_x - width / 2 - bounds[0], center_y - height / 2 - bounds[1]),
        text,
        font=font,
        fill=fill,
    )
