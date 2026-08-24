"""Typed portable plans for Final Cut titles, captions, and generators.

Architecture map
================

FCPXML ``title`` / ``caption`` element
    -> exact text blocks and styled runs
    -> bounded exported template controls
    -> connected transform and opacity records
    -> RGBA raster source plus stock-FFmpeg composition contract

FCPXML generator ``video`` element
    -> evidence-owned adapter
    -> bounded stock-FFmpeg source contract, or an explicit unsupported result

The existing renderer shapes text with Pillow/Raqm because a sequence of
``drawtext`` filters cannot preserve mixed-run shaping reliably.  The shaped
full-canvas RGBA image is then handled only by stock FFmpeg filters.  This
module makes that two-stage contract explicit; it does not silently replace a
Motion template with generic text.

Important invariants
--------------------

* Timing stays as ``Fraction`` values.
* Unknown style references, invalid colors, malformed numeric controls, and
  keyframes without ``keyframeAnimation`` fail explicitly.
* Caption role and interchange-only text attributes survive even though the
  current output is a burned visual approximation.
* Only controls owned by an evidence-backed adapter are executable.  Opaque
  Motion rigs, image wells, and unpublished controls remain findings.

Central integration seam
------------------------

The parser should eventually attach ``TextRenderPlan`` to ``RenderClip`` and
the executor should pass it to ``text.render_text_clip``.  The central FFmpeg
builder should then apply the plan's connected motion using the shared geometry
and compositor modules.  This isolated module deliberately does not edit those
shared files.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable, Literal, Mapping, Optional
import xml.etree.ElementTree as ET

from .model import parse_time


BASIC_TITLE_UID = (
    ".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/"
    "Basic Title.moti"
)
CUSTOM_SOLID_GENERATOR_UID = (
    ".../Generators.localized/Solids.localized/Custom.localized/Custom.motn"
)

TemplateKind = Literal["title", "generator"]
ControlKind = Literal["scalar", "vec2", "rgba", "enum"]
ExecutionKind = Literal["text_rgba", "solid_color", "not_implemented_yet"]


class TextPlanError(ValueError):
    """The text/template XML cannot be represented without guessing."""


@dataclass(frozen=True)
class RGBA:
    red: float
    green: float
    blue: float
    alpha: float = 1.0

    def __post_init__(self) -> None:
        for name in ("red", "green", "blue", "alpha"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise TextPlanError(f"{name} must be finite and between 0 and 1")


@dataclass(frozen=True)
class TextStylePlan:
    """Every FCPXML 1.14 text-style field, split into typed and preserved data."""

    id: Optional[str]
    name: Optional[str]
    font: Optional[str]
    font_face: Optional[str]
    font_size: Optional[float]
    font_color: Optional[RGBA]
    background_color: Optional[RGBA]
    bold: bool
    italic: bool
    stroke_color: Optional[RGBA]
    stroke_width: Optional[float]
    tracking: Optional[float]
    alignment: Optional[str]
    baseline: Optional[float]
    line_spacing: Optional[float]
    baseline_offset: Optional[float]
    underline: Optional[bool]
    shadow_color: Optional[RGBA]
    shadow_offset: Optional[tuple[float, float]]
    shadow_blur_radius: Optional[float]
    tab_stops: Optional[str]
    preserved_params: tuple["PreservedTextParameter", ...]


@dataclass(frozen=True)
class PreservedTextParameter:
    """A Motion-owned style control that must not disappear at parse time."""

    name: str
    key: Optional[str]
    value: Optional[str]
    keyframed: bool


@dataclass(frozen=True)
class TextRunPlan:
    text: str
    style_ref: Optional[str]
    style: TextStylePlan


@dataclass(frozen=True)
class CaptionBlockMetadata:
    """Interchange semantics that a burned caption cannot embed in pixels."""

    display_style: Optional[str]
    roll_up_height: Optional[str]
    position: Optional[tuple[float, float]]
    placement: Optional[str]
    alignment: Optional[str]


@dataclass(frozen=True)
class TextBlockPlan:
    runs: tuple[TextRunPlan, ...]
    caption: CaptionBlockMetadata

    @property
    def text(self) -> str:
        return "".join(run.text for run in self.runs)


@dataclass(frozen=True)
class TemplateControlSpec:
    key: str
    name: str
    kind: ControlKind
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    components: Optional[int] = None
    allowed: tuple[str, ...] = ()


@dataclass(frozen=True)
class TemplateAdapter:
    """A frozen, evidence-owned semantic approximation boundary."""

    uid: str
    name: str
    kind: TemplateKind
    execution: ExecutionKind
    controls: tuple[TemplateControlSpec, ...]
    approximation: str
    evidence: str


@dataclass(frozen=True)
class TemplateControlValue:
    key: str
    name: str
    value: float | tuple[float, float] | RGBA | str


@dataclass(frozen=True)
class TextKeyframe:
    time: Fraction
    value: float | tuple[float, float]
    interp: Optional[str]
    curve: Optional[str]
    aux_value: Optional[str]


@dataclass(frozen=True)
class AnimatedTextControl:
    name: str
    keyframes: tuple[TextKeyframe, ...]


@dataclass(frozen=True)
class ConnectedTextMotion:
    """Intrinsic motion evaluated after the full-canvas text raster exists."""

    position: tuple[float, float]
    scale: tuple[float, float]
    rotation: float
    anchor: tuple[float, float]
    opacity: float
    animations: tuple[AnimatedTextControl, ...]


@dataclass(frozen=True)
class TextExecutionContract:
    """Portable execution stages without any custom FFmpeg build."""

    raster_backend: str = "pillow_raqm_rgba"
    ffmpeg_filters: tuple[str, ...] = (
        "format",
        "scale",
        "rotate",
        "overlay",
        "colorchannelmixer",
    )
    stage_order: tuple[str, ...] = (
        "shape_styled_runs",
        "rasterize_transparent_project_canvas",
        "apply_connected_transform",
        "apply_opacity",
        "alpha_composite",
    )


@dataclass(frozen=True)
class TextFinding:
    construct: str
    disposition: Literal["approximated", "not_implemented_yet"]
    detail: str


@dataclass(frozen=True)
class TextRenderPlan:
    kind: Literal["title", "caption"]
    template_uid: Optional[str]
    role: Optional[str]
    timeline_start: Fraction
    duration: Fraction
    blocks: tuple[TextBlockPlan, ...]
    styles: Mapping[str, TextStylePlan]
    controls: tuple[TemplateControlValue, ...]
    motion: ConnectedTextMotion
    execution: Optional[TextExecutionContract]
    visual_disposition: Literal["semantic_approximation", "not_implemented_yet"]
    findings: tuple[TextFinding, ...]


@dataclass(frozen=True)
class GeneratorRenderPlan:
    template_uid: str
    timeline_start: Fraction
    duration: Fraction
    execution: ExecutionKind
    controls: tuple[TemplateControlValue, ...]
    stock_source: Optional[str]
    findings: tuple[TextFinding, ...]


_BASIC_CONTROLS = (
    TemplateControlSpec(
        key="9999/999166631/999166633/1/100/101",
        name="Position",
        kind="vec2",
        components=2,
    ),
    TemplateControlSpec(
        key="Scale",
        name="Scale",
        kind="vec2",
        minimum=0.01,
        maximum=10.0,
        components=2,
    ),
    TemplateControlSpec(
        key="9999/999166631/999166633/5/999166635/3",
        name="Size",
        kind="scalar",
        minimum=1.0,
        maximum=1000.0,
    ),
    TemplateControlSpec(
        key="9999/999166631/999166633/5/999166635/81/79",
        name="Tracking",
        kind="scalar",
        minimum=-100.0,
        maximum=100.0,
    ),
    TemplateControlSpec(
        key="tracking",
        name="motionTextTracking",
        kind="scalar",
        minimum=-100.0,
        maximum=100.0,
    ),
    TemplateControlSpec(
        key="9999/999166631/999166633/2/354/999169573/401",
        name="Alignment",
        kind="enum",
        allowed=("1 (Center)",),
    ),
)

TEMPLATE_ADAPTERS: Mapping[tuple[TemplateKind, str], TemplateAdapter] = {
    ("title", BASIC_TITLE_UID): TemplateAdapter(
        uid=BASIC_TITLE_UID,
        name="Basic Title",
        kind="title",
        execution="text_rgba",
        controls=_BASIC_CONTROLS,
        approximation=(
            "styled runs are shaped into RGBA, then connected transforms and "
            "opacity are composed with stock FFmpeg"
        ),
        evidence="existing Basic Title sizing calibration and portable fixtures",
    ),
    ("generator", CUSTOM_SOLID_GENERATOR_UID): TemplateAdapter(
        uid=CUSTOM_SOLID_GENERATOR_UID,
        name="Custom",
        kind="generator",
        execution="solid_color",
        controls=(
            TemplateControlSpec(
                key="9999/10008/10006/2/1/1",
                name="Color",
                kind="rgba",
                components=4,
            ),
        ),
        approximation="Custom Solid maps to FFmpeg's bounded color source",
        evidence="independent corpus occurrence exporting the Color parameter",
    ),
}


def template_adapter(kind: TemplateKind, uid: str) -> TemplateAdapter:
    """Return an executable adapter or an explicit opaque-Motion result.

    Main callers:
    - The future central compiler when it resolves an ``effect`` resource.
    - Unit tests and the frozen target report.

    Why this exists:
    - A template name is not enough evidence to synthesize its Motion rig.
    """

    adapter = TEMPLATE_ADAPTERS.get((kind, uid))
    if adapter is not None:
        return adapter
    return TemplateAdapter(
        uid=uid,
        name=_display_name(uid),
        kind=kind,
        execution="not_implemented_yet",
        controls=(),
        approximation="",
        evidence=(
            "opaque Motion rig, image wells, or unpublished controls have no "
            "calibrated portable contract"
        ),
    )


def build_text_render_plan(
    element: ET.Element,
    *,
    template_uid: Optional[str],
    timeline_start: Fraction,
) -> TextRenderPlan:
    """Compile a DTD-shaped title or caption without discarding metadata.

    Main callers:
    - The future central parser/compiler bridge.
    - Experimental DTD-valid fixtures.

    The caller supplies the absolute timeline start because FCPXML ``offset``
    is container-relative.  This prevents the isolated text planner from
    inventing container timing rules.
    """

    tag = _tag(element)
    if tag not in {"title", "caption"}:
        raise TextPlanError(f"expected title or caption, got {tag!r}")
    if not isinstance(timeline_start, Fraction):
        raise TextPlanError("timeline_start must be an exact Fraction")
    duration = _required_time(element.get("duration"), "text duration")
    if duration <= 0:
        raise TextPlanError("text duration must be positive")

    if tag == "title":
        if not template_uid:
            raise TextPlanError("title requires a resolved template UID")
        adapter = template_adapter("title", template_uid)
    else:
        adapter = None

    styles = _parse_style_definitions(element)
    blocks = tuple(_parse_text_block(child, styles) for child in element if _tag(child) == "text")
    if not blocks or not any(block.text for block in blocks):
        raise TextPlanError(f"{tag} has no non-empty text")

    params = tuple(_iter_params(element))
    controls: tuple[TemplateControlValue, ...] = ()
    findings: list[TextFinding] = []
    for style in styles.values():
        for parameter in style.preserved_params:
            findings.append(
                TextFinding(
                    construct=f"text style control {parameter.name}",
                    disposition="not_implemented_yet",
                    detail="Motion-owned text-style param is preserved but has no calibrated stock-FFmpeg mapping",
                )
            )
    if adapter is not None:
        if adapter.execution == "not_implemented_yet":
            findings.append(
                TextFinding(
                    construct=f"template {adapter.name}",
                    disposition="not_implemented_yet",
                    detail=adapter.evidence,
                )
            )
        else:
            controls, control_findings = _compile_controls(params, adapter)
            findings.extend(control_findings)
    elif params:
        findings.append(
            TextFinding(
                construct="caption published parameters",
                disposition="not_implemented_yet",
                detail="native caption parameters are preserved but have no template adapter",
            )
        )

    motion = _parse_connected_motion(element)
    if tag == "caption":
        findings.append(
            TextFinding(
                construct="native caption interchange metadata",
                disposition="approximated",
                detail=(
                    "role, timing, placement, display style, and roll-up fields are "
                    "preserved while the output is burned into video"
                ),
            )
        )

    executable = adapter is None or adapter.execution != "not_implemented_yet"
    return TextRenderPlan(
        kind=tag,
        template_uid=template_uid,
        role=element.get("role"),
        timeline_start=timeline_start,
        duration=duration,
        blocks=blocks,
        styles=styles,
        controls=controls,
        motion=motion,
        execution=TextExecutionContract() if executable else None,
        visual_disposition=(
            "semantic_approximation" if executable else "not_implemented_yet"
        ),
        findings=tuple(findings),
    )


def build_generator_render_plan(
    element: ET.Element,
    *,
    template_uid: str,
    timeline_start: Fraction,
) -> GeneratorRenderPlan:
    """Compile an evidence-backed generator or preserve an explicit finding."""

    if _tag(element) not in {"video", "asset-clip"}:
        raise TextPlanError("generator must be represented by a video-like story item")
    if not isinstance(timeline_start, Fraction):
        raise TextPlanError("timeline_start must be an exact Fraction")
    duration = _required_time(element.get("duration"), "generator duration")
    adapter = template_adapter("generator", template_uid)
    params = tuple(_iter_params(element))
    if adapter.execution == "not_implemented_yet":
        return GeneratorRenderPlan(
            template_uid=template_uid,
            timeline_start=timeline_start,
            duration=duration,
            execution=adapter.execution,
            controls=(),
            stock_source=None,
            findings=(
                TextFinding(
                    construct=f"generator {adapter.name}",
                    disposition="not_implemented_yet",
                    detail=adapter.evidence,
                ),
            ),
        )
    controls, findings = _compile_controls(params, adapter)
    return GeneratorRenderPlan(
        template_uid=template_uid,
        timeline_start=timeline_start,
        duration=duration,
        execution=adapter.execution,
        controls=controls,
        stock_source="color",
        findings=findings,
    )


def _parse_style_definitions(element: ET.Element) -> dict[str, TextStylePlan]:
    styles: dict[str, TextStylePlan] = {}
    for definition in element:
        if _tag(definition) != "text-style-def":
            continue
        style_id = definition.get("id")
        if not style_id:
            raise TextPlanError("text-style-def requires id")
        style_elements = [child for child in definition if _tag(child) == "text-style"]
        if len(style_elements) != 1:
            raise TextPlanError(f"text-style-def {style_id!r} requires exactly one text-style")
        if style_id in styles:
            raise TextPlanError(f"duplicate text style id {style_id!r}")
        styles[style_id] = _parse_style(
            style_elements[0], style_id=style_id, name=definition.get("name")
        )
    return styles


def _parse_text_block(
    element: ET.Element,
    styles: Mapping[str, TextStylePlan],
) -> TextBlockPlan:
    default = _default_style()
    runs: list[TextRunPlan] = []
    if element.text:
        runs.append(TextRunPlan(element.text, None, default))
    for child in element:
        if _tag(child) != "text-style":
            raise TextPlanError(f"text contains unsupported child {_tag(child)!r}")
        ref = child.get("ref")
        inline_fields = set(child.attrib) - {"ref"}
        if ref and ref not in styles:
            raise TextPlanError(f"text-style references unknown style {ref!r}")
        if inline_fields:
            style = _parse_style(child, style_id=None, name=None, base=styles.get(ref, default))
        else:
            style = styles.get(ref, default)
        runs.append(TextRunPlan("".join(child.itertext()), ref, style))
        if child.tail:
            runs.append(TextRunPlan(child.tail, None, default))
    return TextBlockPlan(
        runs=tuple(run for run in runs if run.text),
        caption=CaptionBlockMetadata(
            display_style=element.get("display-style"),
            roll_up_height=element.get("roll-up-height"),
            position=_optional_pair(element.get("position"), "caption text position"),
            placement=element.get("placement"),
            alignment=element.get("alignment"),
        ),
    )


def _parse_style(
    element: ET.Element,
    *,
    style_id: Optional[str],
    name: Optional[str],
    base: Optional[TextStylePlan] = None,
) -> TextStylePlan:
    base = base or _empty_style()
    kerning = _optional_float(element.get("kerning"), "text kerning")
    return TextStylePlan(
        id=style_id,
        name=name,
        font=element.get("font", base.font),
        font_face=element.get("fontFace", base.font_face),
        font_size=_optional_float(element.get("fontSize"), "font size", base.font_size),
        font_color=_optional_rgba(element.get("fontColor"), "font color", base.font_color),
        background_color=_optional_rgba(
            element.get("backgroundColor"), "background color", base.background_color
        ),
        bold=_optional_bool(element.get("bold"), "bold", base.bold),
        italic=_optional_bool(element.get("italic"), "italic", base.italic),
        stroke_color=_optional_rgba(
            element.get("strokeColor"), "stroke color", base.stroke_color
        ),
        stroke_width=_optional_float(
            element.get("strokeWidth"), "stroke width", base.stroke_width
        ),
        tracking=kerning if kerning is not None else base.tracking,
        alignment=element.get("alignment", base.alignment),
        baseline=_optional_float(element.get("baseline"), "baseline", base.baseline),
        line_spacing=_optional_float(
            element.get("lineSpacing"), "line spacing", base.line_spacing
        ),
        baseline_offset=_optional_float(
            element.get("baselineOffset"), "baseline offset", base.baseline_offset
        ),
        underline=_optional_bool(element.get("underline"), "underline", base.underline),
        shadow_color=_optional_rgba(
            element.get("shadowColor"), "shadow color", base.shadow_color
        ),
        shadow_offset=_optional_pair(
            element.get("shadowOffset"), "shadow offset", base.shadow_offset
        ),
        shadow_blur_radius=_optional_float(
            element.get("shadowBlurRadius"),
            "shadow blur radius",
            base.shadow_blur_radius,
        ),
        tab_stops=element.get("tabStops", base.tab_stops),
        preserved_params=tuple(
            PreservedTextParameter(
                name=child.get("name") or "<unnamed>",
                key=child.get("key"),
                value=child.get("value"),
                keyframed=any(_tag(grandchild) == "keyframeAnimation" for grandchild in child),
            )
            for child in element
            if _tag(child) == "param"
        )
        or base.preserved_params,
    )


def _default_style() -> TextStylePlan:
    return TextStylePlan(
        id=None,
        name=None,
        font="Helvetica",
        font_face=None,
        font_size=50.0,
        font_color=RGBA(1.0, 1.0, 1.0, 1.0),
        background_color=None,
        bold=False,
        italic=False,
        stroke_color=None,
        stroke_width=None,
        tracking=0.0,
        alignment="center",
        baseline=None,
        line_spacing=None,
        baseline_offset=None,
        underline=None,
        shadow_color=None,
        shadow_offset=None,
        shadow_blur_radius=None,
        tab_stops=None,
        preserved_params=(),
    )


def _empty_style() -> TextStylePlan:
    value = _default_style()
    return TextStylePlan(
        **{
            **value.__dict__,
            "font": None,
            "font_size": None,
            "font_color": None,
            "tracking": None,
            "alignment": None,
        }
    )


def _iter_params(element: ET.Element) -> Iterable[ET.Element]:
    for child in element:
        if _tag(child) != "param":
            continue
        yield child
        yield from _iter_params(child)


def _compile_controls(
    params: tuple[ET.Element, ...],
    adapter: TemplateAdapter,
) -> tuple[tuple[TemplateControlValue, ...], tuple[TextFinding, ...]]:
    by_key = {spec.key: spec for spec in adapter.controls}
    by_name = {spec.name.casefold(): spec for spec in adapter.controls}
    values: list[TemplateControlValue] = []
    findings: list[TextFinding] = []
    seen: dict[str, TemplateControlValue] = {}
    for param in params:
        key = param.get("key") or ""
        name = param.get("name") or ""
        spec = by_key.get(key) or by_name.get(name.casefold())
        if spec is None:
            findings.append(
                TextFinding(
                    construct=f"published control {name or key or '<unnamed>'}",
                    disposition="not_implemented_yet",
                    detail="control is preserved but absent from the evidence-owned adapter",
                )
            )
            continue
        if any(_tag(child) == "keyframeAnimation" for child in param):
            findings.append(
                TextFinding(
                    construct=f"published control animation {spec.name}",
                    disposition="not_implemented_yet",
                    detail="template-control animation is preserved; connected intrinsic animation executes separately",
                )
            )
        raw = param.get("value")
        if raw is None:
            continue
        value = TemplateControlValue(
            key=spec.key,
            name=spec.name,
            value=_parse_control_value(raw, spec),
        )
        previous = seen.get(spec.key)
        if previous is not None:
            if previous.value != value.value:
                raise TextPlanError(
                    f"conflicting duplicate published control {spec.name!r}: "
                    f"{previous.value!r} != {value.value!r}"
                )
            continue
        seen[spec.key] = value
        values.append(value)
    return tuple(values), tuple(findings)


def _parse_control_value(
    raw: str,
    spec: TemplateControlSpec,
) -> float | tuple[float, float] | RGBA | str:
    if spec.kind == "enum":
        if raw not in spec.allowed:
            raise TextPlanError(f"{spec.name} value {raw!r} is outside {spec.allowed!r}")
        return raw
    if spec.kind == "rgba":
        return _rgba(raw, spec.name)
    numbers = _numbers(raw, spec.name)
    if spec.kind == "scalar":
        if len(numbers) != 1:
            raise TextPlanError(f"{spec.name} requires one numeric component")
        _check_range(numbers, spec)
        return numbers[0]
    if len(numbers) != 2:
        raise TextPlanError(f"{spec.name} requires two numeric components")
    _check_range(numbers, spec)
    return numbers[0], numbers[1]


def _check_range(numbers: Iterable[float], spec: TemplateControlSpec) -> None:
    for number in numbers:
        if spec.minimum is not None and number < spec.minimum:
            raise TextPlanError(f"{spec.name} is below {spec.minimum}")
        if spec.maximum is not None and number > spec.maximum:
            raise TextPlanError(f"{spec.name} is above {spec.maximum}")


def _parse_connected_motion(element: ET.Element) -> ConnectedTextMotion:
    transform = next((child for child in element if _tag(child) == "adjust-transform"), None)
    blend = next((child for child in element if _tag(child) == "adjust-blend"), None)
    animations: list[AnimatedTextControl] = []
    if transform is not None:
        for param in transform:
            if _tag(param) == "param":
                track = _parse_animation(param)
                if track is not None:
                    animations.append(track)
    if blend is not None:
        for param in blend:
            if _tag(param) == "param":
                track = _parse_animation(param)
                if track is not None:
                    animations.append(track)
    return ConnectedTextMotion(
        position=_optional_pair(
            transform.get("position") if transform is not None else None,
            "transform position",
            (0.0, 0.0),
        )
        or (0.0, 0.0),
        scale=_optional_pair(
            transform.get("scale") if transform is not None else None,
            "transform scale",
            (1.0, 1.0),
        )
        or (1.0, 1.0),
        rotation=_optional_float(
            transform.get("rotation") if transform is not None else None,
            "transform rotation",
            0.0,
        )
        or 0.0,
        anchor=_optional_pair(
            transform.get("anchor") if transform is not None else None,
            "transform anchor",
            (0.0, 0.0),
        )
        or (0.0, 0.0),
        opacity=_bounded_opacity(blend.get("amount") if blend is not None else None),
        animations=tuple(animations),
    )


def _parse_animation(param: ET.Element) -> Optional[AnimatedTextControl]:
    direct_keyframes = [child for child in param if _tag(child) == "keyframe"]
    if direct_keyframes:
        raise TextPlanError("keyframes must be nested inside keyframeAnimation")
    containers = [child for child in param if _tag(child) == "keyframeAnimation"]
    if not containers:
        return None
    if len(containers) != 1:
        raise TextPlanError("param must not contain multiple keyframeAnimation elements")
    frames: list[TextKeyframe] = []
    for element in containers[0]:
        if _tag(element) != "keyframe":
            continue
        time = _required_time(element.get("time"), "text animation keyframe time")
        raw = element.get("value")
        if raw is None:
            raise TextPlanError("text animation keyframe requires value")
        numbers = _numbers(raw, "text animation keyframe value")
        if len(numbers) not in {1, 2}:
            raise TextPlanError("text animation keyframe value must be scalar or vec2")
        frames.append(
            TextKeyframe(
                time=time,
                value=numbers[0] if len(numbers) == 1 else (numbers[0], numbers[1]),
                interp=element.get("interp"),
                curve=element.get("curve"),
                aux_value=element.get("auxValue"),
            )
        )
    if not frames:
        raise TextPlanError("keyframeAnimation must contain keyframes")
    if any(right.time <= left.time for left, right in zip(frames, frames[1:])):
        raise TextPlanError("text animation keyframes must have strictly increasing times")
    first_is_vector = isinstance(frames[0].value, tuple)
    if any(isinstance(frame.value, tuple) != first_is_vector for frame in frames[1:]):
        raise TextPlanError("text animation cannot mix scalar and vec2 keyframes")
    return AnimatedTextControl(
        name=param.get("name") or param.get("key") or "<unnamed>",
        keyframes=tuple(frames),
    )


def _bounded_opacity(raw: Optional[str]) -> float:
    if raw is None:
        return 1.0
    value = _optional_float(raw, "opacity")
    if value is None or not 0.0 <= value <= 1.0:
        raise TextPlanError("opacity must be between 0 and 1")
    return value


def _required_time(raw: Optional[str], name: str) -> Fraction:
    try:
        value = parse_time(raw, required=True, field_name=name)
    except ValueError as error:
        raise TextPlanError(str(error)) from error
    if value is None:
        raise TextPlanError(f"{name} is required")
    return value


def _numbers(raw: str, name: str) -> tuple[float, ...]:
    try:
        numbers = tuple(float(piece) for piece in raw.replace(",", " ").split())
    except ValueError as error:
        raise TextPlanError(f"{name} must contain numeric components") from error
    if not numbers or any(not math.isfinite(value) for value in numbers):
        raise TextPlanError(f"{name} must contain finite numeric components")
    return numbers


def _rgba(raw: str, name: str) -> RGBA:
    numbers = _numbers(raw, name)
    if len(numbers) == 3:
        numbers = (*numbers, 1.0)
    if len(numbers) != 4:
        raise TextPlanError(f"{name} requires three or four components")
    return RGBA(*numbers)


def _optional_rgba(
    raw: Optional[str],
    name: str,
    default: Optional[RGBA] = None,
) -> Optional[RGBA]:
    return default if raw is None else _rgba(raw, name)


def _optional_float(
    raw: Optional[str],
    name: str,
    default: Optional[float] = None,
) -> Optional[float]:
    if raw is None:
        return default
    numbers = _numbers(raw, name)
    if len(numbers) != 1:
        raise TextPlanError(f"{name} requires one component")
    return numbers[0]


def _optional_pair(
    raw: Optional[str],
    name: str,
    default: Optional[tuple[float, float]] = None,
) -> Optional[tuple[float, float]]:
    if raw is None:
        return default
    numbers = _numbers(raw, name)
    if len(numbers) != 2:
        raise TextPlanError(f"{name} requires two components")
    return numbers[0], numbers[1]


def _optional_bool(raw: Optional[str], name: str, default: Optional[bool]) -> Optional[bool]:
    if raw is None:
        return default
    if raw not in {"0", "1"}:
        raise TextPlanError(f"{name} must be 0 or 1")
    return raw == "1"


def _display_name(uid: str) -> str:
    leaf = uid.rsplit("/", 1)[-1]
    return leaf.rsplit(".", 1)[0] or uid


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]
