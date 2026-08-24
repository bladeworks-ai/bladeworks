"""Portable Basic Title and native-caption rasterization.

Architecture map
================

1. Resolve an exact font face from explicit bindings or local font files.
2. Shape each text run through Pillow's Raqm engine (FreeType + HarfBuzz).
3. Place Basic Title's first baseline in its 1920x1080 template coordinates.
4. Save one full-canvas transparent PNG for FFmpeg to transform/composite.

This module never calls Core Text. Missing or ambiguous fonts produce an
omission finding; a visually different fallback font is never selected.
"""

from __future__ import annotations

import os
import logging
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

from .errors import RenderCapabilityError
from .model import (
    FontBinding,
    Parameter,
    RenderClip,
    RenderDocument,
    RenderVideoDisposition,
    TextRun,
    TextStyle,
)
from .report import CompatibilityReport
from .text_templates import GeneratorRenderPlan, RGBA, TextStylePlan


@dataclass(frozen=True)
class FontFace:
    path: Path
    index: int
    names: tuple[str, ...]
    # OpenType has both legacy family/subfamily records (IDs 1/2) and
    # preferred typographic records (IDs 16/17).  A collection may label
    # several distinct faces as legacy ``Regular`` while the preferred
    # subfamily still says ``Medium``, ``Heavy``, and so on.  Preserve the
    # authoritative pair instead of flattening every name into one ambiguous
    # lookup bucket.
    family_subfamilies: tuple[tuple[str, str], ...] = ()
    weight_class: Optional[int] = None
    italic: Optional[bool] = None


@dataclass(frozen=True)
class _LocalFontIndex:
    """Reusable, read-only view of the installed OpenType faces."""

    faces: dict[str, tuple[FontFace, ...]]
    family_faces: dict[tuple[str, str], tuple[FontFace, ...]]
    families: dict[str, tuple[FontFace, ...]]


_LOCAL_FONT_INDEXES: dict[tuple[Path, ...], _LocalFontIndex] = {}
_LOCAL_FONT_INDEX_LOCK = threading.Lock()


class _IgnoreBrokenFontTimestamp(logging.Filter):
    """Hide fontTools repairs for known-bad, unused font date metadata."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            message.startswith("'created' timestamp seems very low")
            or message.startswith("'modified' timestamp seems very low")
            or message.startswith("'created' timestamp out of range")
            or message.startswith("'modified' timestamp out of range")
        )


@dataclass(frozen=True)
class RuntimeRasterResolution:
    """One exact runtime raster result and its executable video disposition.

    Architecture map:
    authored text/generator clip -> portable rasterizer -> this record ->
    executor-owned runtime ``RenderDocument`` replacement.

    A missing image is never meaningful by itself. Failure records carry the
    same typed reason, construct, and portability status written to the report;
    successful records explicitly retain ``composite`` execution.
    """

    clip_id: str
    image_path: Optional[Path]
    video_disposition: RenderVideoDisposition

    def __post_init__(self) -> None:
        if not self.clip_id.strip():
            raise ValueError("runtime raster resolution requires a clip ID")
        if self.image_path is None:
            if self.video_disposition.execution != "omit_transparent":
                raise ValueError(
                    "missing runtime raster requires omit_transparent disposition"
                )
        elif self.video_disposition.execution != "composite":
            raise ValueError(
                "resolved runtime raster requires composite disposition"
            )


def _runtime_raster_omission(
    clip: RenderClip,
    report: CompatibilityReport,
    *,
    portable_status: str,
    construct: str,
    reason: str,
    uid: Optional[str] = None,
) -> RuntimeRasterResolution:
    """Write and return one omission from the same typed runtime facts."""

    report.add(
        outcome="omitted",
        portable_status=portable_status,
        fcpxml_path=clip.path,
        construct=construct,
        uid=uid,
        timeline_start=clip.absolute_start,
        timeline_duration=clip.duration,
        disposition=reason,
    )
    return RuntimeRasterResolution(
        clip_id=clip.id,
        image_path=None,
        video_disposition=RenderVideoDisposition(
            execution="omit_transparent",
            reason=reason,
            portable_status=portable_status,
            construct=construct,
            uid=uid,
        ),
    )


def verify_text_runtime() -> None:
    if not features.check("freetype2"):
        raise RenderCapabilityError("Pillow was built without FreeType support")
    if not features.check("raqm"):
        raise RenderCapabilityError("Pillow was built without Raqm/HarfBuzz text layout support")


def _local_font_roots() -> tuple[Path, ...]:
    """Return the configured and standard font roots for this process."""

    configured = tuple(
        Path(item).expanduser()
        for item in os.environ.get("FCPXML_RENDER_FONT_DIRS", "").split(os.pathsep)
        if item
    )
    return configured + (
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
        Path.home() / "Library" / "Fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".local" / "share" / "fonts",
    )


def _build_local_font_index(roots: tuple[Path, ...]) -> _LocalFontIndex:
    """Scan installed fonts once and return their exact names and traits.

    Main callers:
    - ``_cached_local_font_index`` on the first title render for a root set.

    Why this exists:
    Preview synchronization rebuilds its frame producer after every accepted
    edit. Repeating a complete macOS font scan there wastes time and repeats
    fontTools warnings about irrelevant creation-date metadata.
    """

    try:
        from fontTools.ttLib import TTCollection, TTFont
    except ImportError as exc:
        raise RenderCapabilityError("fontTools is required to resolve FCPXML font names") from exc

    faces: dict[str, list[FontFace]] = {}
    family_faces: dict[tuple[str, str], list[FontFace]] = {}
    families: dict[str, list[FontFace]] = {}
    seen: set[Path] = set()
    timestamp_logger = logging.getLogger("fontTools.ttLib.tables._h_e_a_d")
    timestamp_filter = _IgnoreBrokenFontTimestamp()
    timestamp_logger.addFilter(timestamp_filter)
    try:
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if path.suffix.lower() not in {".ttf", ".otf", ".ttc", ".otc"} or path in seen:
                    continue
                seen.add(path)
                try:
                    if path.suffix.lower() in {".ttc", ".otc"}:
                        collection = TTCollection(path, lazy=True)
                        count = len(collection.fonts)
                        collection.close()
                    else:
                        count = 1
                    for index in range(count):
                        font = TTFont(path, fontNumber=index, lazy=True)
                        names, family_subfamilies = _font_metadata(font)
                        weight_class, italic = _font_style_traits(font)
                        font.close()
                        face = FontFace(
                            path=path.resolve(),
                            index=index,
                            names=tuple(names),
                            family_subfamilies=tuple(sorted(family_subfamilies)),
                            weight_class=weight_class,
                            italic=italic,
                        )
                        for name in names:
                            faces.setdefault(_normalize(name), []).append(face)
                        for family, subfamily in family_subfamilies:
                            key = (_normalize(family), _normalize(subfamily))
                            family_faces.setdefault(key, []).append(face)
                            families.setdefault(_normalize(family), []).append(face)
                except Exception:
                    # Broken or unsupported font files are not candidates. A
                    # requested name still fails explicitly if nothing valid remains.
                    continue
    finally:
        timestamp_logger.removeFilter(timestamp_filter)

    return _LocalFontIndex(
        faces={key: tuple(value) for key, value in faces.items()},
        family_faces={key: tuple(value) for key, value in family_faces.items()},
        families={key: tuple(value) for key, value in families.items()},
    )


def _cached_local_font_index(roots: tuple[Path, ...]) -> _LocalFontIndex:
    """Return one process-wide font index for an exact set of roots."""

    with _LOCAL_FONT_INDEX_LOCK:
        cached = _LOCAL_FONT_INDEXES.get(roots)
        if cached is None:
            cached = _build_local_font_index(roots)
            _LOCAL_FONT_INDEXES[roots] = cached
        return cached


class FontResolver:
    """Resolve exact font names from bindings and portable font name tables."""

    def __init__(self, bindings: tuple[FontBinding, ...]):
        self.bindings = bindings
        self._scanned = False
        self._binding_faces: dict[str, list[FontFace]] = {}
        self._faces: dict[str, list[FontFace]] = {}
        self._family_faces: dict[tuple[str, str], list[FontFace]] = {}
        self._families: dict[str, list[FontFace]] = {}
        for binding in bindings:
            face = FontFace(path=binding.path, index=binding.index, names=(binding.name,))
            key = _normalize(binding.name)
            self._binding_faces.setdefault(key, []).append(face)
            self._faces.setdefault(key, []).append(face)

    def resolve(
        self,
        name: str,
        *,
        font_face: Optional[str] = None,
        bold: bool,
        italic: bool,
    ) -> Optional[FontFace]:
        face_name = (font_face or "").strip()
        normalized_face = face_name.casefold()
        if not face_name:
            pieces = []
            if bold:
                pieces.append("Bold")
            if italic:
                pieces.append("Italic")
            face_name = " ".join(pieces) or "Regular"
        styled_name = name if normalized_face and normalized_face in name.casefold() else f"{name} {face_name}"
        # A requested bold/italic face must not fall back to the regular face.
        requested = [styled_name]
        if not bold and not italic and normalized_face in {"", "regular", "normal", "roman"}:
            requested.append(name)

        # Explicit bindings are the user's exact machine-independent choice.
        # A duplicate binding is an ambiguity, not permission to scan the host
        # and silently substitute a different local face.
        if any(
            self._binding_faces.get(_normalize(candidate))
            for candidate in requested
        ):
            return self._unique(requested, faces=self._binding_faces)

        self._scan_local_fonts()
        if not normalized_face:
            complete_traits, styled = self._unique_family_style(
                name,
                weight_class=700 if bold else 400,
                italic=italic,
            )
            if complete_traits:
                return styled
        family_key = (_normalize(name), _normalize(face_name))
        if self._family_faces.get(family_key):
            # Presence plus no unique result means the exact pair is
            # ambiguous.  Do not escape through a coincidentally unique full
            # or legacy name below.
            return self._unique_family_face(name, face_name)
        return self._unique(requested)

    def _unique_family_style(
        self,
        family: str,
        *,
        weight_class: int,
        italic: bool,
    ) -> tuple[bool, Optional[FontFace]]:
        """Select an omitted ``fontFace`` from exact OpenType traits.

        Main callers:
        - ``resolve`` when FCPXML authored only a family plus bold/italic flags.

        Why this exists:
        OpenType permits a normal face to call its subfamily ``Roman`` rather
        than ``Regular``. Looking up the invented name ``Family Regular`` can
        therefore miss the intended face or collide with a heavy face whose
        legacy name table also says ``Regular``. The preferred subfamily is
        authoritative for the role inside its preferred family: for example,
        ``Arial Black Regular`` is the normal face of the face-specific
        ``Arial Black`` family even though its global weight class is 900.
        Exact slant metadata still has to agree with the authored italic flag.

        The boolean is false when the installed family lacks complete traits,
        so the older exact-name lookup may still decide. A true boolean plus
        ``None`` means complete traits proved the authored request absent or
        ambiguous and must not fall through to an alias.
        """

        unique = {
            (candidate.path, candidate.index): candidate
            for candidate in self._families.get(_normalize(family), ())
        }
        if not unique:
            return False, None
        candidates = tuple(unique.values())
        if any(
            candidate.weight_class is None or candidate.italic is None
            for candidate in candidates
        ):
            return False, None
        canonical = tuple(
            candidate
            for candidate in candidates
            if (
                candidate.italic is italic
                and _subfamily_matches_style(
                    candidate,
                    family=family,
                    weight_class=weight_class,
                    italic=italic,
                )
            )
        )
        if len(canonical) == 1:
            return True, canonical[0]
        # A complete preferred-family record is authoritative. Do not let a
        # sole same-weight design variant (for example Titling) or a heavy
        # noncanonical alias escape through the legacy exact-name lookup.
        return True, None

    def _unique_family_face(self, family: str, subfamily: str) -> Optional[FontFace]:
        """Resolve one exact typographic family/subfamily pair.

        Main callers:
        - ``resolve`` after local OpenType name tables have been scanned.

        Why this exists:
        Variable and collection fonts often reuse the legacy subfamily name
        ``Regular`` for several faces.  OpenType IDs 16/17 carry the actual
        typographic pair and prevent Avenir Next Medium or Heavy from making
        Avenir Next Regular appear ambiguous.
        """

        candidates = self._family_faces.get(
            (_normalize(family), _normalize(subfamily)),
            [],
        )
        unique = {
            (candidate.path, candidate.index): candidate
            for candidate in candidates
        }
        if len(unique) == 1:
            return next(iter(unique.values()))
        return None

    def _unique(
        self,
        names: Iterable[str],
        *,
        faces: Optional[dict[str, list[FontFace]]] = None,
    ) -> Optional[FontFace]:
        candidates_by_name = self._faces if faces is None else faces
        for name in names:
            candidates = candidates_by_name.get(_normalize(name), [])
            unique = {(candidate.path, candidate.index): candidate for candidate in candidates}
            if len(unique) == 1:
                return next(iter(unique.values()))
            if len(unique) > 1:
                return None
        return None

    def _scan_local_fonts(self) -> None:
        if self._scanned:
            return
        self._scanned = True
        index = _cached_local_font_index(_local_font_roots())
        for name, faces in index.faces.items():
            self._faces.setdefault(name, []).extend(faces)
        for family, faces in index.family_faces.items():
            self._family_faces.setdefault(family, []).extend(faces)
        for family, faces in index.families.items():
            self._families.setdefault(family, []).extend(faces)


def resolve_text_clip_raster(
    clip: RenderClip,
    document: RenderDocument,
    *,
    work_dir: Path,
    resolver: FontResolver,
    report: CompatibilityReport,
) -> RuntimeRasterResolution:
    """Rasterize text and return an explicit composite-or-transparent result."""

    try:
        verify_text_runtime()
    except RenderCapabilityError as error:
        return _runtime_raster_omission(
            clip,
            report,
            portable_status="unsupported",
            construct=f"{clip.kind} text rasterizer",
            reason=str(error),
            uid=(clip.text_plan.template_uid if clip.text_plan is not None else None),
        )
    if clip.text_plan is not None and clip.text_plan.execution is None:
        return _runtime_raster_omission(
            clip,
            report,
            portable_status="unsupported",
            construct=clip.kind,
            reason="opaque text template has no calibrated execution adapter",
            uid=clip.text_plan.template_uid,
        )
    if not clip.text_runs:
        return _runtime_raster_omission(
            clip,
            report,
            portable_status="unsupported",
            construct=clip.kind,
            reason="text item has no renderable text runs",
        )

    default_style = next(iter(clip.text_styles.values()), None) or TextStyle(
        id=None,
        font="Helvetica",
        font_face=None,
        font_size=50.0,
        font_color="1 1 1 1",
        alignment="center",
        stroke_color=None,
        stroke_width=None,
        tracking=None,
        bold=False,
        italic=False,
    )
    published_size = _parameter_float(clip.params, "Size", 0.0)
    published_scale = _parameter_pair(clip.params, "Scale", (1.0, 1.0))
    uniform_scale = max(0.01, (abs(published_scale[0]) + abs(published_scale[1])) / 2.0)
    styled_runs: list[tuple[TextRun, TextStyle, ImageFont.FreeTypeFont]] = []
    for run in clip.text_runs:
        style = run.inline_style or (clip.text_styles.get(run.style_ref) if run.style_ref else None) or default_style
        font_name = style.font or default_style.font or "Helvetica"
        try:
            face = resolver.resolve(
                font_name,
                font_face=style.font_face,
                bold=style.bold,
                italic=style.italic,
            )
        except RenderCapabilityError as error:
            return _runtime_raster_omission(
                clip,
                report,
                portable_status="unsupported",
                construct=f"font {font_name}",
                reason=str(error),
            )
        if face is None or not face.path.is_file():
            return _runtime_raster_omission(
                clip,
                report,
                portable_status="unsupported",
                construct=f"font {font_name}",
                reason=(
                    "font name did not resolve uniquely to a readable local "
                    "font file; title/caption omitted"
                ),
            )
        point_size = published_size if published_size > 0 else (style.font_size or 50.0)
        # Some FreeType/Raqm builds divide by zero for extremely small faces.
        # Eight pixels stays within the MVP text-bounds tolerance on tiny test
        # canvases and avoids turning one caption into a whole-render failure.
        size = max(8, round(point_size * uniform_scale * document.height / 1080.0))
        try:
            font = ImageFont.truetype(
                str(face.path),
                size=size,
                index=face.index,
                layout_engine=ImageFont.Layout.RAQM,
            )
        except OSError as exc:
            return _runtime_raster_omission(
                clip,
                report,
                portable_status="unsupported",
                construct=f"font {font_name}",
                reason=f"font could not be loaded: {exc}",
            )
        styled_runs.append((run, style, font))

    image = Image.new("RGBA", (document.width, document.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    tracking = _parameter_float(clip.params, "Tracking", _parameter_float(clip.params, "motionTextTracking", default_style.tracking or 0.0))
    position = _parameter_pair(clip.params, "Position", (0.0, 0.0))
    alignment = (styled_runs[0][1].alignment or "center").lower()
    if alignment not in {"left", "right", "center"}:
        report.add(
            outcome="approximated",
            portable_status="calibrated_portable",
            fcpxml_path=clip.path,
            construct=f"text alignment {alignment}",
            timeline_start=clip.absolute_start,
            timeline_duration=clip.duration,
            disposition="unknown alignment rendered as center",
        )
        alignment = "center"
    anchor_x = document.width / 2 + position[0] * document.height / 1080.0
    if clip.kind == "caption" and position == (0.0, 0.0):
        first_baseline = document.height * 0.88
    else:
        first_baseline = document.height / 2 - position[1] * document.height / 1080.0

    lines = _split_lines(styled_runs)
    baseline = first_baseline
    for line in lines:
        widths = [_text_width(draw, run.text, font, tracking) for run, _, font in line]
        line_width = sum(widths)
        if alignment == "left":
            x = anchor_x
        elif alignment == "right":
            x = anchor_x - line_width
        else:
            x = anchor_x - line_width / 2
        ascent = max((font.getmetrics()[0] for _, _, font in line), default=0)
        descent = max((font.getmetrics()[1] for _, _, font in line), default=0)
        for (run, style, font), width in zip(line, widths):
            fill, stroke_fill, stroke_width = _text_paint(style, document.height)
            planned_style = _planned_style(clip, run.style_ref)
            top = baseline - font.getmetrics()[0]
            bottom = baseline + font.getmetrics()[1]
            if planned_style and planned_style.background_color is not None:
                draw.rectangle(
                    (x, top, x + width, bottom),
                    fill=_plan_rgba(planned_style.background_color),
                )
            if planned_style and planned_style.shadow_color is not None:
                offset_x, offset_y = planned_style.shadow_offset or (0.0, 0.0)
                scale = document.height / 1080.0
                shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
                shadow_draw = ImageDraw.Draw(shadow)
                _draw_text(
                    shadow_draw,
                    (x + offset_x * scale, top - offset_y * scale),
                    run.text,
                    font=font,
                    fill=_plan_rgba(planned_style.shadow_color),
                    stroke_fill=(0, 0, 0, 0),
                    stroke_width=0,
                    tracking=tracking,
                )
                radius = max(
                    0.0,
                    (planned_style.shadow_blur_radius or 0.0) * scale,
                )
                if radius:
                    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=radius))
                image.alpha_composite(shadow)
                draw = ImageDraw.Draw(image)
            _draw_text(
                draw,
                (x, top + ((planned_style.baseline_offset or 0.0) if planned_style else 0.0)),
                run.text,
                font=font,
                fill=fill,
                stroke_fill=stroke_fill,
                stroke_width=stroke_width,
                tracking=tracking,
            )
            if planned_style and planned_style.underline:
                underline_y = baseline + max(1, round(document.height / 540.0))
                draw.line(
                    (x, underline_y, x + width, underline_y),
                    fill=fill,
                    width=max(1, round(document.height / 540.0)),
                )
            x += width
        planned_spacing = max(
            (
                _planned_style(clip, run.style_ref).line_spacing or 0.0
                for run, _, _ in line
                if _planned_style(clip, run.style_ref) is not None
            ),
            default=0.0,
        )
        baseline += max(1, ascent + descent + planned_spacing)

    path = work_dir / f"{clip.id}-text.png"
    image.save(path)
    return RuntimeRasterResolution(
        clip_id=clip.id,
        image_path=path,
        video_disposition=RenderVideoDisposition(execution="composite"),
    )


def render_text_clip(
    clip: RenderClip,
    document: RenderDocument,
    *,
    work_dir: Path,
    resolver: FontResolver,
    report: CompatibilityReport,
) -> Optional[Path]:
    """Compatibility wrapper returning the historical optional image path."""

    verify_text_runtime()
    return resolve_text_clip_raster(
        clip,
        document,
        work_dir=work_dir,
        resolver=resolver,
        report=report,
    ).image_path


def resolve_generator_clip_raster(
    clip: RenderClip,
    document: RenderDocument,
    *,
    work_dir: Path,
    report: CompatibilityReport,
) -> RuntimeRasterResolution:
    """Rasterize one evidence-backed generator into a frozen RGBA source.

    Main callers:
    - ``executor.execute_render`` before the shared FFmpeg graph is built.

    Only the Custom Solid adapter reaches this function. Unknown Motion
    generators have ``execution=not_implemented_yet`` and remain explicit
    compatibility findings instead of becoming arbitrary colored frames.
    """

    plan: Optional[GeneratorRenderPlan] = clip.generator_plan
    if plan is None or plan.execution != "solid_color":
        reason = "generator has no executable portable raster adapter"
        construct = "generator"
        uid = plan.template_uid if plan is not None else None
        if clip.video_disposition is not None and (
            clip.video_disposition.execution == "omit_transparent"
        ):
            reason = clip.video_disposition.reason or reason
            construct = clip.video_disposition.construct or construct
            uid = clip.video_disposition.uid
        return _runtime_raster_omission(
            clip,
            report,
            portable_status="unsupported",
            construct=construct,
            reason=reason,
            uid=uid,
        )
    color = next(
        (
            control.value
            for control in plan.controls
            if control.name == "Color" and isinstance(control.value, RGBA)
        ),
        None,
    )
    if color is None:
        return _runtime_raster_omission(
            clip,
            report,
            portable_status="unsupported",
            construct="Custom Solid Color",
            reason="generator adapter did not contain its required typed Color",
            uid=plan.template_uid,
        )
    image = Image.new(
        "RGBA",
        (document.width, document.height),
        _plan_rgba(color),
    )
    path = work_dir / f"{clip.id}-generator.png"
    image.save(path)
    return RuntimeRasterResolution(
        clip_id=clip.id,
        image_path=path,
        video_disposition=RenderVideoDisposition(execution="composite"),
    )


def render_generator_clip(
    clip: RenderClip,
    document: RenderDocument,
    *,
    work_dir: Path,
    report: CompatibilityReport,
) -> Optional[Path]:
    """Compatibility wrapper returning the historical optional image path."""

    if clip.generator_plan is None or (
        clip.generator_plan.execution != "solid_color"
    ):
        return None
    return resolve_generator_clip_raster(
        clip,
        document,
        work_dir=work_dir,
        report=report,
    ).image_path


def _planned_style(clip: RenderClip, style_ref: Optional[str]) -> Optional[TextStylePlan]:
    if clip.text_plan is None or style_ref is None:
        return None
    return clip.text_plan.styles.get(style_ref)


def _plan_rgba(color: RGBA) -> tuple[int, int, int, int]:
    return tuple(
        max(0, min(255, round(component * 255)))
        for component in (color.red, color.green, color.blue, color.alpha)
    )  # type: ignore[return-value]


def _text_paint(
    style: TextStyle,
    project_height: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int]:
    """Translate Final Cut's signed stroke convention into Pillow paint.

    Final Cut uses a positive ``strokeWidth`` for outline-only glyphs and a
    negative value for filled glyphs with an outside stroke. Pillow only takes
    a non-negative pixel width, so preserve the sign by changing the face fill
    before passing the absolute width to Pillow.

    Main callers: ``render_text_clip`` while rasterizing each styled text run.
    Why this exists: taking ``abs(strokeWidth)`` alone made positive and
    negative FCPXML styles look identical and produced visibly wrong title
    fills in real Final Cut comparisons.
    """

    fill = _rgba(style.font_color, (255, 255, 255, 255))
    stroke_fill = _rgba(style.stroke_color, (0, 0, 0, 255))
    signed_width = style.stroke_width or 0.0
    stroke_width = max(0, round(abs(signed_width) * project_height / 1080.0))
    if signed_width > 0:
        fill = (fill[0], fill[1], fill[2], 0)
    return fill, stroke_fill, stroke_width


def _split_lines(
    styled_runs: list[tuple[TextRun, TextStyle, ImageFont.FreeTypeFont]],
) -> list[list[tuple[TextRun, TextStyle, ImageFont.FreeTypeFont]]]:
    lines: list[list[tuple[TextRun, TextStyle, ImageFont.FreeTypeFont]]] = [[]]
    for run, style, font in styled_runs:
        pieces = run.text.split("\n")
        for index, piece in enumerate(pieces):
            if piece:
                lines[-1].append((TextRun(text=piece, style_ref=run.style_ref, inline_style=run.inline_style), style, font))
            if index < len(pieces) - 1:
                lines.append([])
    return lines


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, tracking: float) -> float:
    base = draw.textlength(text, font=font, embedded_color=False)
    return float(base + max(len(text) - 1, 0) * tracking * font.size / 1000.0)


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    stroke_fill: tuple[int, int, int, int],
    stroke_width: int,
    tracking: float,
) -> None:
    spacing = tracking * font.size / 1000.0
    if abs(spacing) < 0.01:
        draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
        return
    x, y = xy
    for character in text:
        draw.text((x, y), character, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
        x += draw.textlength(character, font=font) + spacing


def _parameter_float(params: tuple[Parameter, ...], name: str, default: float) -> float:
    for param in params:
        if (param.name or "").lower() == name.lower() and param.value is not None:
            try:
                return float(param.value.split()[0])
            except ValueError:
                return default
    return default


def _parameter_pair(params: tuple[Parameter, ...], name: str, default: tuple[float, float]) -> tuple[float, float]:
    for param in params:
        if (param.name or "").lower() == name.lower() and param.value:
            pieces = param.value.replace(",", " ").split()
            if len(pieces) >= 2:
                try:
                    return float(pieces[0]), float(pieces[1])
                except ValueError:
                    return default
    return default


def _rgba(raw: Optional[str], default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if not raw:
        return default
    try:
        values = [float(piece) for piece in raw.replace(",", " ").split()]
    except ValueError:
        return default
    if len(values) < 3:
        return default
    if len(values) == 3:
        values.append(1.0)
    return tuple(max(0, min(255, round(value * 255))) for value in values[:4])  # type: ignore[return-value]


def _font_metadata(font: object) -> tuple[set[str], set[tuple[str, str]]]:
    """Return searchable names plus exact typographic face pairs.

    Preferred OpenType family/subfamily IDs 16/17 take precedence as a pair.
    Legacy IDs 1/2 are used only when no complete preferred pair exists.
    Full and PostScript names remain searchable, but they cannot override an
    exact family/subfamily match.
    """

    output: set[str] = set()
    values_by_locale: dict[tuple[int, int, int], dict[int, set[str]]] = {}
    name_table = font["name"]  # type: ignore[index]
    for record in name_table.names:
        if record.nameID not in {1, 2, 4, 6, 16, 17}:
            continue
        try:
            value = record.toUnicode().strip()
        except Exception:
            continue
        if value:
            output.add(value)
            locale = (
                getattr(record, "platformID", 0),
                getattr(record, "platEncID", 0),
                getattr(record, "langID", 0),
            )
            values_by_locale.setdefault(locale, {}).setdefault(
                record.nameID,
                set(),
            ).add(value)

    # Pair records only inside one platform/encoding/language tuple.  Mixing
    # an English family with a French subfamily invents a name that the font
    # never declared and can silently select the wrong face.
    preferred_pairs = {
        (family, subfamily)
        for values in values_by_locale.values()
        for family in values.get(16, ())
        for subfamily in values.get(17, ())
    }
    if preferred_pairs:
        family_subfamilies = preferred_pairs
    else:
        family_subfamilies = {
            (family, subfamily)
            for values in values_by_locale.values()
            for family in values.get(1, ())
            for subfamily in values.get(2, ())
        }
    for family, subfamily in family_subfamilies:
        output.add(f"{family} {subfamily}")
    return output, family_subfamilies


def _font_style_traits(font: object) -> tuple[Optional[int], Optional[bool]]:
    """Read exact weight/italic traits used when ``fontFace`` is absent.

    Main callers:
    - ``FontResolver._scan_local_fonts`` for every installed collection face.

    OpenType's OS/2 table is authoritative for weight. Slant metadata is
    distributed across OS/2 ITALIC/OBLIQUE, head macStyle, and post
    italicAngle in real fonts, so any positive declaration establishes the
    italic class. An unknown weight remains unknown and cannot win
    trait-based resolution.
    """

    weight_class: Optional[int] = None
    italic_signals: list[bool] = []
    try:
        os2 = font["OS/2"]  # type: ignore[index]
    except Exception:
        os2 = None
    if os2 is not None:
        raw_weight = getattr(os2, "usWeightClass", None)
        if isinstance(raw_weight, int) and 1 <= raw_weight <= 1000:
            weight_class = raw_weight
        selection = getattr(os2, "fsSelection", None)
        if isinstance(selection, int):
            # OpenType defines distinct ITALIC and OBLIQUE bits. Both request
            # the slanted face class represented by FCPXML's italic flag.
            italic_signals.append(bool(selection & ((1 << 0) | (1 << 9))))

    try:
        head = font["head"]  # type: ignore[index]
    except Exception:
        head = None
    mac_style = getattr(head, "macStyle", None)
    if isinstance(mac_style, int):
        italic_signals.append(bool(mac_style & 0x02))

    italic_angle = _raw_post_italic_angle(font)
    if isinstance(italic_angle, (int, float)):
        italic_signals.append(bool(italic_angle))
    italic = any(italic_signals) if italic_signals else None
    return weight_class, italic


def _raw_post_italic_angle(font: object) -> Optional[float]:
    """Read the small fixed header without decoding the ``post`` glyph map.

    Main callers:
    - ``_font_style_traits`` while indexing local fonts for title rendering.

    Why this exists:
    Some system collections have enormous format-2 ``post`` name maps. Asking
    fontTools to materialize that whole table can stall the first live preview
    for over a minute even though italic detection needs only bytes 4 through
    7 of the fixed header.
    """

    reader = getattr(font, "reader", None)
    if reader is not None:
        try:
            data = reader["post"]
            if len(data) >= 8:
                return struct.unpack(">l", data[4:8])[0] / 65536.0
        except Exception:
            return None
    try:
        post = font["post"]  # type: ignore[index]
    except Exception:
        return None
    value = getattr(post, "italicAngle", None)
    return float(value) if isinstance(value, (int, float)) else None


def _subfamily_matches_style(
    face: FontFace,
    *,
    family: str,
    weight_class: int,
    italic: bool,
) -> bool:
    """Distinguish the canonical face from same-weight design variants.

    A collection can contain both ``Roman`` and ``Titling`` at weight 400.
    Both are upright, but only Roman/Regular/Normal/Book declares the ordinary
    face requested when FCPXML omits ``fontFace``.  The equivalent canonical
    labels for bold and italic requests are matched the same way.  Unknown or
    multiple canonical declarations remain ambiguous.
    """

    if weight_class == 400 and not italic:
        names = {"regular", "roman", "normal", "book"}
    elif weight_class == 700 and not italic:
        names = {"bold"}
    elif weight_class == 400 and italic:
        names = {"italic", "oblique", "regularitalic", "regularoblique"}
    elif weight_class == 700 and italic:
        names = {"bolditalic", "boldoblique"}
    else:
        return False
    normalized_family = _normalize(family)
    return any(
        _normalize(candidate_family) == normalized_family
        and _normalize(subfamily) in names
        for candidate_family, subfamily in face.family_subfamilies
    )


def _normalize(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())
