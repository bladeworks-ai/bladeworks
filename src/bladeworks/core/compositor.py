"""Plan deterministic alpha-correct Final Cut layer composition with FFmpeg.

Architecture map
================

``resolve_blend_mode``
    -> normalizes one Final Cut mode name
    -> returns a typed stock-FFmpeg mapping
    -> rejects unknown and deliberately unsupported cross-channel modes

``OpacityPlan``
    -> evaluates static or retime-aware animated opacity
    -> multiplies explicit Final Cut fade envelopes
    -> emits the same bounded expression for FFmpeg's ``geq`` alpha plane

``CompositorPlan.build_filter_graph``
    -> normalizes both project-sized canvases to planar RGBA
    -> applies opacity to the foreground alpha
    -> performs blend/matte math
    -> finishes with straight-alpha source-over composition

The plan accepts an ownership label (ordinary layer, group, mask, or
transition side), but intentionally uses the same pixel pipeline for all four.
This is the seam that prevents those call sites from drifting into subtly
different alpha behavior.

Important invariants
--------------------

* The first input is always the already-composited lower canvas.  The second
  input is always the new foreground canvas.
* Opacity is applied before blend and composite operations.
* Non-normal RGB modes account for the lower canvas alpha before overlaying.
  Transparent lower pixels therefore reveal the unblended foreground color.
* Hue, Saturation, Color, and Luminosity are explicit unsupported findings.
  Unknown strings are errors and never become Normal.
* All timeline coordinates are exact ``Fraction`` values.  Generated FFmpeg
  text is deterministic and contains no machine-specific state.
* Final Cut's Normal source-over happens in an approximately power-linear
  working space.  A genuine BT.709 four-patch oracle calibrated the bounded
  exponent below.  Its sanitized samples, environment, and private-artifact
  hashes are frozen in ``evidence/normal_opacity_calibration.v1.json``;
  non-Normal blend modes retain their separately reviewed stock-FFmpeg
  mappings.

Why this exists
---------------

FFmpeg's ``blend`` filter computes channel colors but does not by itself
perform Final Cut's layer composition.  Feeding its result directly to the
timeline loses alpha correctness for transparent groups.  This module keeps
blend color selection and source-over composition as separate, testable
stages while remaining entirely within stock FFmpeg.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Literal, TypeAlias

from .animation import ScalarControlPoint, TimelineAnimatedScalar
from .model import FadeEnvelope
from .pixel_domains import CompositionModuleContract


CompositionOwner: TypeAlias = Literal[
    "layer", "group", "mask", "transition_side"
]
BlendFamily: TypeAlias = Literal["normal", "rgb", "behind", "matte"]
MatteKind: TypeAlias = Literal[
    "stencil_alpha", "stencil_luma", "silhouette_alpha", "silhouette_luma"
]
SemanticStatus: TypeAlias = Literal[
    "exact_alpha", "stock_ffmpeg_semantic_approximation"
]
ForegroundAlphaContract: TypeAlias = Literal[
    "arbitrary", "binary_canvas", "binary_full_canvas"
]


# Final Cut 12.3, standard Rec.709 library, Normal blend amount 0.18.
#
# The serialized oracle used four opaque upper/lower color pairs.  Across its
# 12 decoded output channels a power-linear source-over with this exponent was
# within two 8-bit levels per channel; code-space alpha missed by as much as 58.
# This is intentionally not called the BT.709 transfer function: Final Cut's
# measured dark-channel response fit a simple power law more closely than the
# published piecewise transfer curve.
# The continuous best fit is 1.9315.  FFmpeg's 16-bit ``lutrgb`` quantization
# lands closest to the decoded colored-lower oracle at 1.94 (mean error 0.58
# and maximum error two across the twelve measured RGB channels), so the executable
# constant includes that measured backend compensation.
FCP_NORMAL_SOURCE_OVER_GAMMA = 1.94


class CompositorError(ValueError):
    """Base error for an invalid or unavailable composition contract."""


class CompositorValidationError(CompositorError):
    """A typed request contains invalid timing, opacity, or graph labels."""


class UnknownBlendModeError(CompositorError):
    """The requested blend string is not a known Final Cut mode."""


class UnsupportedBlendModeError(CompositorError):
    """The mode is known but lacks a defensible stock-FFmpeg implementation."""


@dataclass(frozen=True)
class BlendModeSpec:
    """One reviewed Final Cut-to-FFmpeg blend mapping."""

    canonical_name: str
    family: BlendFamily
    ffmpeg_mode: str | None
    matte_kind: MatteKind | None
    semantic_status: SemanticStatus


_RGB_MODES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Add", "addition", ("add", "addition")),
    ("Subtract", "subtract", ("subtract",)),
    ("Darken", "darken", ("darken",)),
    ("Lighten", "lighten", ("lighten",)),
    ("Multiply", "multiply", ("multiply",)),
    ("Screen", "screen", ("screen",)),
    ("Overlay", "overlay", ("overlay",)),
    ("Soft Light", "softlight", ("softlight",)),
    ("Hard Light", "hardlight", ("hardlight",)),
    ("Difference", "difference", ("difference",)),
    ("Exclusion", "exclusion", ("exclusion",)),
    ("Color Burn", "burn", ("burn", "colorburn")),
    ("Color Dodge", "dodge", ("dodge", "colordodge")),
    ("Divide", "divide", ("divide",)),
    ("Linear Light", "linearlight", ("linearlight",)),
    ("Pin Light", "pinlight", ("pinlight",)),
    ("Hard Mix", "hardmix", ("hardmix",)),
)


def _compact_mode(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UnknownBlendModeError("blend mode must be a non-empty string")
    return re.sub(r"[^a-z0-9]+", "", value.lower())


_MODE_SPECS: dict[str, BlendModeSpec] = {
    "normal": BlendModeSpec("Normal", "normal", None, None, "exact_alpha"),
    "behind": BlendModeSpec("Behind", "behind", None, None, "exact_alpha"),
}
for _canonical, _ffmpeg, _aliases in _RGB_MODES:
    _spec = BlendModeSpec(
        _canonical,
        "rgb",
        _ffmpeg,
        None,
        "stock_ffmpeg_semantic_approximation",
    )
    for _alias in _aliases:
        _MODE_SPECS[_alias] = _spec
for _canonical, _kind in (
    ("Stencil Alpha", "stencil_alpha"),
    ("Stencil Luma", "stencil_luma"),
    ("Silhouette Alpha", "silhouette_alpha"),
    ("Silhouette Luma", "silhouette_luma"),
):
    _MODE_SPECS[_compact_mode(_canonical)] = BlendModeSpec(
        _canonical,
        "matte",
        None,
        _kind,  # type: ignore[arg-type]
        "exact_alpha" if _kind.endswith("alpha") else "stock_ffmpeg_semantic_approximation",
    )

_UNSUPPORTED_CROSS_CHANNEL = {
    "hue": "Hue",
    "saturation": "Saturation",
    "color": "Color",
    "luminosity": "Luminosity",
}


def resolve_blend_mode(value: str | None) -> BlendModeSpec:
    """Resolve one Final Cut blend string without a silent Normal fallback.

    Main callers:
    - ``CompositorPlan`` validation.
    - Compiler compatibility reporting before central graph integration.

    ``None`` means the FCPXML omitted ``adjust-blend/@mode``, whose declared
    behavior is Normal.  An explicit but unknown string is always an error.
    """

    if value is None:
        return _MODE_SPECS["normal"]
    key = _compact_mode(value)
    if key in _UNSUPPORTED_CROSS_CHANNEL:
        raise UnsupportedBlendModeError(
            f"Final Cut {_UNSUPPORTED_CROSS_CHANNEL[key]} blend is known but not "
            "implemented: it requires calibrated cross-channel color semantics"
        )
    try:
        return _MODE_SPECS[key]
    except KeyError as exc:
        raise UnknownBlendModeError(f"unknown Final Cut blend mode {value!r}") from exc


def _exact_time(value: object, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise CompositorValidationError(f"{name} must be an exact Fraction, not bool")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        raise CompositorValidationError(f"{name} must be an exact Fraction, not float")
    raise CompositorValidationError(
        f"{name} must be an exact Fraction, got {type(value).__name__}"
    )


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise CompositorValidationError(f"{name} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CompositorValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise CompositorValidationError(f"{name} must be finite")
    return result


def _opacity(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0 or result > 1.0:
        raise CompositorValidationError(f"{name} must be between 0 and 1")
    return result


def _number(value: float) -> str:
    result = _finite(value, name="FFmpeg expression value")
    if abs(result) < 5e-13:
        result = 0.0
    return format(result, ".12g")


def _power_transfer_expression(exponent: float) -> str:
    """Return one bounded RGB-only FFmpeg LUT expression.

    Main callers:
    - ``CompositorPlan.build_filter_graph`` immediately before and after
      Final Cut Normal source-over.

    Why this exists:
    ``lutrgb`` exposes the current component as ``val`` and that component's
    full-scale value as ``maxval``.  Keeping the normalization inside this
    renderer-owned helper makes the same calibrated exponent work for the
    16-bit composition canvas without accepting expression text from XML.
    Alpha is deliberately omitted and therefore passes through unchanged.
    """

    value = _finite(exponent, name="power-transfer exponent")
    if value <= 0:
        raise CompositorValidationError("power-transfer exponent must be positive")
    return f"maxval*pow(val/maxval,{_number(value)})"


def _ratio(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"({value.numerator}/{value.denominator})"


def _validate_label(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise CompositorValidationError(
            f"{name} must contain only ASCII letters, digits, or underscores"
        )
    return value


@dataclass(frozen=True)
class CompositorWindow:
    """One local layer stream including optional transition handles."""

    clip_duration: Fraction
    transition_pre_roll: Fraction = Fraction(0)
    transition_post_roll: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        for name in (
            "clip_duration",
            "transition_pre_roll",
            "transition_post_roll",
        ):
            object.__setattr__(self, name, _exact_time(getattr(self, name), name=name))
        if self.clip_duration <= 0:
            raise CompositorValidationError("clip_duration must be positive")
        if self.transition_pre_roll < 0 or self.transition_post_roll < 0:
            raise CompositorValidationError("transition handles cannot be negative")

    @property
    def render_duration(self) -> Fraction:
        return self.transition_pre_roll + self.clip_duration + self.transition_post_roll

    def clip_time(self, render_time: Fraction) -> Fraction:
        """Clamp transition handles to the first/last clip opacity state."""

        exact = _exact_time(render_time, name="render_time")
        if exact < 0 or exact > self.render_duration:
            raise CompositorValidationError("render_time is outside the layer window")
        return min(
            max(exact - self.transition_pre_roll, Fraction(0)),
            self.clip_duration,
        )


_FADE_ALIASES = {
    "linear": "linear",
    "easein": "ease_in",
    "easeout": "ease_out",
    "easeinout": "ease_in_out",
    "ease": "ease_in_out",
    "smooth": "ease_in_out",
}


def _fade_curve_name(raw: str | None, *, direction: Literal["in", "out"]) -> str:
    # Final Cut's omitted video-fade curve holds the corresponding one-sided
    # ease.  These defaults are frozen here instead of inheriting FFmpeg's
    # unrelated linear ``fade`` default.
    if raw is None:
        return "ease_in" if direction == "in" else "ease_out"
    key = re.sub(r"[^a-z0-9]+", "", raw.lower())
    try:
        return _FADE_ALIASES[key]
    except KeyError as exc:
        raise CompositorValidationError(f"unsupported fade curve {raw!r}") from exc


def _curve_value(progress: float, curve: str) -> float:
    value = min(max(progress, 0.0), 1.0)
    if curve == "linear":
        return value
    if curve == "ease_in":
        return value * value * value
    if curve == "ease_out":
        inverse = 1.0 - value
        return 1.0 - inverse * inverse * inverse
    if curve == "ease_in_out":
        return value * value * (3.0 - 2.0 * value)
    raise AssertionError(f"unreachable fade curve {curve!r}")


def _curve_expression(progress: str, curve: str) -> str:
    if curve == "linear":
        return f"({progress})"
    if curve == "ease_in":
        return f"({progress})*({progress})*({progress})"
    if curve == "ease_out":
        return f"1-(1-({progress}))*(1-({progress}))*(1-({progress}))"
    if curve == "ease_in_out":
        return f"({progress})*({progress})*(3-2*({progress}))"
    raise AssertionError(f"unreachable fade curve {curve!r}")


def _monotone_slopes(points: tuple[ScalarControlPoint, ...]) -> tuple[float, ...]:
    """Match the animation kernel's bounded smooth interpolation slopes."""

    count = len(points)
    if count == 1:
        return (0.0,)
    widths = [float(points[index + 1].time - points[index].time) for index in range(count - 1)]
    secants = [
        (points[index + 1].value - points[index].value) / widths[index]
        for index in range(count - 1)
    ]
    if count == 2:
        return (secants[0], secants[0])
    slopes = [0.0] * count
    for index in range(1, count - 1):
        left = secants[index - 1]
        right = secants[index]
        if left == 0.0 or right == 0.0 or left * right <= 0.0:
            slopes[index] = 0.0
            continue
        left_weight = 2.0 * widths[index] + widths[index - 1]
        right_weight = widths[index] + 2.0 * widths[index - 1]
        slopes[index] = (left_weight + right_weight) / (
            left_weight / left + right_weight / right
        )

    def endpoint(width0: float, width1: float, secant0: float, secant1: float) -> float:
        slope = ((2.0 * width0 + width1) * secant0 - width0 * secant1) / (width0 + width1)
        if slope * secant0 <= 0.0:
            return 0.0
        if secant0 * secant1 < 0.0 and abs(slope) > abs(3.0 * secant0):
            return 3.0 * secant0
        return slope

    slopes[0] = endpoint(widths[0], widths[1], secants[0], secants[1])
    slopes[-1] = endpoint(widths[-1], widths[-2], secants[-1], secants[-2])
    return tuple(slopes)


def _source_time_expression(track: TimelineAnimatedScalar, clip_time: str) -> str:
    segments = track.retime_map.segments
    expression = _ratio(segments[-1].source_end)
    for index in reversed(range(len(segments))):
        segment = segments[index]
        mapped = (
            f"({_ratio(segment.source_start)}+"
            f"(({clip_time})-{_ratio(segment.timeline_start)})*{_ratio(segment.rate)})"
        )
        # Match RetimeMap's half-open segment ownership.  A discontinuous map
        # boundary belongs to the following segment; only the map's final end
        # is inclusive.
        comparison = "lte" if index == len(segments) - 1 else "lt"
        expression = (
            f"if({comparison}(({clip_time}),{_ratio(segment.timeline_end)}),"
            f"{mapped},{expression})"
        )
    return expression


def _animated_opacity_expression(track: TimelineAnimatedScalar, clip_time: str) -> str:
    points = track.source_track.control_points
    source_time = _source_time_expression(track, clip_time)
    if len(points) == 1:
        return _number(points[0].value)
    slopes = _monotone_slopes(points)
    segment_expressions: list[str] = []
    for index, (left, right) in enumerate(zip(points, points[1:])):
        raw = f"(({source_time})-{_ratio(left.time)})/{_ratio(right.time-left.time)}"
        if right.interpolation == "linear":
            eased = raw
        elif right.interpolation == "ease":
            eased = f"({raw})*({raw})*(3-2*({raw}))"
        elif right.interpolation == "ease-in":
            eased = f"({raw})*({raw})*({raw})"
        elif right.interpolation == "ease-out":
            eased = f"1-(1-({raw}))*(1-({raw}))*(1-({raw}))"
        else:
            raise CompositorValidationError(
                f"unsupported opacity interpolation {right.interpolation!r}"
            )
        if right.curve == "linear":
            value = f"({_number(left.value)}+({_number(right.value-left.value)})*({eased}))"
        else:
            width = float(right.time - left.time)
            u2 = f"({eased})*({eased})"
            u3 = f"({u2})*({eased})"
            value = (
                f"((2*({u3})-3*({u2})+1)*{_number(left.value)}+"
                f"(({u3})-2*({u2})+({eased}))*{_number(width*slopes[index])}+"
                f"(-2*({u3})+3*({u2}))*{_number(right.value)}+"
                f"(({u3})-({u2}))*{_number(width*slopes[index+1])})"
            )
            value = f"clip({value},{_number(min(left.value,right.value))},{_number(max(left.value,right.value))})"
        segment_expressions.append(value)

    expression = _number(points[-1].value)
    for index in reversed(range(len(segment_expressions))):
        expression = (
            f"if(lte(({source_time}),{_ratio(points[index+1].time)}),"
            f"{segment_expressions[index]},{expression})"
        )
    return (
        f"if(lte(({source_time}),{_ratio(points[0].time)}),{_number(points[0].value)},"
        f"{expression})"
    )


@dataclass(frozen=True)
class OpacitySnapshot:
    """One inspectable opacity state at an exact render-local time."""

    render_time: Fraction
    clip_time: Fraction
    base_opacity: float
    fade_factor: float
    result: float


@dataclass(frozen=True)
class OpacityPlan:
    """Static/animated opacity plus explicit Final Cut fade envelopes."""

    window: CompositorWindow
    static_opacity: float = 1.0
    animation: TimelineAnimatedScalar | None = None
    fade: FadeEnvelope | None = None
    expression_time_origin: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if not isinstance(self.window, CompositorWindow):
            raise CompositorValidationError("window must be CompositorWindow")
        object.__setattr__(
            self,
            "static_opacity",
            _opacity(self.static_opacity, name="static_opacity"),
        )
        object.__setattr__(
            self,
            "expression_time_origin",
            _exact_time(self.expression_time_origin, name="expression_time_origin"),
        )
        if self.animation is not None:
            if not isinstance(self.animation, TimelineAnimatedScalar):
                raise CompositorValidationError(
                    "animation must be TimelineAnimatedScalar or None"
                )
            if self.animation.retime_map.timeline_start != 0:
                raise CompositorValidationError(
                    "opacity animation retime map must start at clip-local time zero"
                )
            if self.animation.retime_map.timeline_end != self.window.clip_duration:
                raise CompositorValidationError(
                    "opacity animation retime map must cover the complete clip duration"
                )
            for index, point in enumerate(self.animation.source_track.control_points):
                _opacity(point.value, name=f"opacity control point {index}")
        if self.fade is not None:
            if not isinstance(self.fade, FadeEnvelope):
                raise CompositorValidationError("fade must be FadeEnvelope or None")
            for name in ("fade_in", "fade_out"):
                value = getattr(self.fade, name)
                if value is None:
                    continue
                exact = _exact_time(value, name=name)
                if exact <= 0 or exact > self.window.clip_duration:
                    raise CompositorValidationError(
                        f"{name} must be positive and no longer than the clip"
                    )
            if self.fade.fade_in is not None:
                _fade_curve_name(self.fade.fade_in_type, direction="in")
            if self.fade.fade_out is not None:
                _fade_curve_name(self.fade.fade_out_type, direction="out")

    def snapshot(self, render_time: Fraction) -> OpacitySnapshot:
        """Evaluate one state through the exact animation kernel.

        Main callers:
        - A/B event-frame diagnostics.
        - Unit tests that compare the typed state with FFmpeg output frames.
        """

        clip_time = self.window.clip_time(render_time)
        base = (
            self.animation.value_at(clip_time)
            if self.animation is not None
            else self.static_opacity
        )
        factor = 1.0
        if self.fade is not None and self.fade.fade_in is not None and clip_time < self.fade.fade_in:
            progress = float(clip_time / self.fade.fade_in)
            factor *= _curve_value(
                progress,
                _fade_curve_name(self.fade.fade_in_type, direction="in"),
            )
        if self.fade is not None and self.fade.fade_out is not None:
            start = self.window.clip_duration - self.fade.fade_out
            if clip_time > start:
                progress = float((clip_time - start) / self.fade.fade_out)
                factor *= 1.0 - _curve_value(
                    progress,
                    _fade_curve_name(self.fade.fade_out_type, direction="out"),
                )
        result = _opacity(base * factor, name="evaluated opacity")
        return OpacitySnapshot(render_time, clip_time, base, factor, result)

    @property
    def is_constant_one(self) -> bool:
        """Return whether opacity is provably one for the complete render window.

        Main callers:
        - ``CompositorPlan`` before its opaque Normal fast path.
        - The FFmpeg integration proof that selects that path.

        Why this exists:
        Endpoint sampling cannot prove that an animated or faded envelope is
        one between samples.  This structural check accepts only the static,
        animation-free, fade-free representation.
        """

        return (
            self.static_opacity == 1.0
            and self.animation is None
            and self.fade is None
        )

    def ffmpeg_expression(self, *, time_variable: str = "T") -> str:
        """Emit the alpha multiplier evaluated in one layer-local FFmpeg stream."""

        if time_variable not in {"T", "t"}:
            raise CompositorValidationError("time_variable must be FFmpeg T or t")
        pre = _ratio(self.window.transition_pre_roll)
        duration = _ratio(self.window.clip_duration)
        origin = _ratio(self.expression_time_origin)
        clip_time = f"clip((({time_variable})-{origin})-{pre},0,{duration})"
        base = (
            _animated_opacity_expression(self.animation, clip_time)
            if self.animation is not None
            else _number(self.static_opacity)
        )
        factors = [f"({base})"]
        if self.fade is not None and self.fade.fade_in is not None:
            progress = f"clip(({clip_time})/{_ratio(self.fade.fade_in)},0,1)"
            factors.append(
                f"({_curve_expression(progress, _fade_curve_name(self.fade.fade_in_type, direction='in'))})"
            )
        if self.fade is not None and self.fade.fade_out is not None:
            start = self.window.clip_duration - self.fade.fade_out
            progress = f"clip((({clip_time})-{_ratio(start)})/{_ratio(self.fade.fade_out)},0,1)"
            curve = _curve_expression(
                progress,
                _fade_curve_name(self.fade.fade_out_type, direction="out"),
            )
            factors.append(f"(1-({curve}))")
        return f"clip({'*'.join(factors)},0,1)"


@dataclass(frozen=True)
class CompositorGraph:
    """A deterministic filter graph fragment and its audited requirements."""

    filter_graph: str
    output_label: str
    mode: BlendModeSpec
    opacity_expression: str
    required_filters: tuple[str, ...]
    operation_order: tuple[str, ...]
    pixel_contract: CompositionModuleContract | None = None


@dataclass(frozen=True)
class OpacityMaskGeometry:
    """Project clock/canvas used to evaluate opacity once per frame.

    The fallback graph evaluates opacity inside a full-frame ``geq``.  The
    central renderer supplies this geometry so the same expression is instead
    evaluated on a one-pixel mask and expanded across the canvas. This keeps
    arbitrary easing exact without repeating a large expression per pixel.
    """

    width: int
    height: int
    fps: Fraction
    duration: Fraction

    def __post_init__(self) -> None:
        if min(self.width, self.height) <= 0:
            raise CompositorValidationError("opacity mask dimensions must be positive")
        if self.fps <= 0 or self.duration <= 0:
            raise CompositorValidationError("opacity mask clock must be positive")


@dataclass(frozen=True)
class CompositorPlan:
    """Build one reusable two-canvas composition operation.

    Main callers:
    - Ordinary storyline layer folding.
    - ``RenderGroup`` flattening.
    - Mask result reinsertion.
    - Transition-side composition before a two-input transition.

    Why this exists:
    - All four callers own the same operation but have different lifetimes.
      The ``owner`` field records that scope for reports without changing the
      pixel math.
    """

    owner: CompositionOwner
    blend_mode: str | None
    opacity: OpacityPlan

    def __post_init__(self) -> None:
        if self.owner not in {"layer", "group", "mask", "transition_side"}:
            raise CompositorValidationError(f"unknown composition owner {self.owner!r}")
        if not isinstance(self.opacity, OpacityPlan):
            raise CompositorValidationError("opacity must be OpacityPlan")
        resolve_blend_mode(self.blend_mode)

    @property
    def mode(self) -> BlendModeSpec:
        return resolve_blend_mode(self.blend_mode)

    def build_filter_graph(
        self,
        *,
        lower_label: str,
        foreground_label: str,
        output_label: str,
        prefix: str,
        opacity_mask_geometry: OpacityMaskGeometry | None = None,
        foreground_alpha_contract: ForegroundAlphaContract = "arbitrary",
        active_frame_range: tuple[int, int] | None = None,
        semantic_enable: str | None = None,
    ) -> CompositorGraph:
        """Return a stock-FFmpeg graph fragment for two RGBA canvases.

        ``binary_full_canvas`` is a proof supplied by the integration layer,
        not a pixel guess: while enabled, the foreground alpha is 255 over the
        whole project canvas, and outside its interval it is zero.  Normal
        source-over at constant opacity one is therefore an identity selection
        and must not round-trip already-final RGB through calibration LUTs.

        ``active_frame_range`` limits the calibrated Normal operations to the
        foreground's authored frame window.  FFmpeg then streams the lower
        canvas through unchanged outside that window instead of splitting and
        concatenating three full-resolution timeline branches.

        ``semantic_enable`` is the integration layer's complete ownership
        expression.  Unlike ``active_frame_range``, it can exclude intervals
        owned by transitions.  The binary-alpha selection path must apply this
        expression itself: a transparent placement canvas is not a substitute
        for disabling the final framesync operation.  Its explicit planar RGB
        format also prevents FFmpeg from converting a disabled lower frame
        through YUV during automatic format negotiation.
        """

        lower = _validate_label(lower_label, name="lower_label")
        foreground = _validate_label(foreground_label, name="foreground_label")
        output = _validate_label(output_label, name="output_label")
        stem = _validate_label(prefix, name="prefix")
        mode = self.mode
        if foreground_alpha_contract not in {
            "arbitrary",
            "binary_canvas",
            "binary_full_canvas",
        }:
            raise CompositorValidationError(
                "foreground_alpha_contract must be arbitrary, binary_canvas, "
                "or binary_full_canvas"
            )
        if foreground_alpha_contract != "arbitrary" and (
            mode.family != "normal" or not self.opacity.is_constant_one
        ):
            raise CompositorValidationError(
                "binary alpha contracts require Normal blend and constant opacity one"
            )
        if active_frame_range is not None:
            if (
                not isinstance(active_frame_range, tuple)
                or len(active_frame_range) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in active_frame_range
                )
                or active_frame_range[0] < 0
                or active_frame_range[1] <= active_frame_range[0]
            ):
                raise CompositorValidationError(
                    "active_frame_range must be a non-empty half-open integer range"
                )
        if semantic_enable is not None and (
            not isinstance(semantic_enable, str) or not semantic_enable.strip()
        ):
            raise CompositorValidationError(
                "semantic_enable must be a non-empty FFmpeg expression"
            )
        opacity = self.opacity.ffmpeg_expression(time_variable="T")
        lower_rgba = f"{stem}_lower_rgba"
        foreground_rgba = f"{stem}_foreground_rgba"
        lines = [
            f"[{lower}]format=gbrap[{lower_rgba}]",
            f"[{foreground}]format=gbrap[{stem}_fg_pre]",
        ]
        if foreground_alpha_contract != "arbitrary":
            binary_enable = ""
            if semantic_enable is not None:
                binary_enable = f":enable='{semantic_enable}'"
            elif active_frame_range is not None:
                start_frame, end_frame = active_frame_range
                binary_enable = (
                    f":enable='gte(n,{start_frame})*lt(n,{end_frame})'"
                )
            lines.append(
                f"[{lower_rgba}][{stem}_fg_pre]"
                f"overlay=x=0:y=0:eof_action=pass:repeatlast=0:"
                f"format=gbrp:alpha=straight{binary_enable}[{output}]"
            )
            return CompositorGraph(
                filter_graph=";".join(lines),
                output_label=output,
                mode=mode,
                opacity_expression=opacity,
                required_filters=("format", "overlay"),
                operation_order=(
                    "normalize_rgba",
                    "opaque_normal_source_over",
                ),
            )
        if opacity_mask_geometry is None:
            lines.append(
                f"[{stem}_fg_pre]geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
                f"a='alpha(X,Y)*({opacity})'[{foreground_rgba}]"
            )
            opacity_filters = ("geq",)
        else:
            geometry = opacity_mask_geometry
            fps = f"{geometry.fps.numerator}/{geometry.fps.denominator}"
            duration = format(float(geometry.duration), ".12g")
            lines.extend(
                (
                    f"color=c=white:s=1x1:r={fps}:d={duration},format=gray,"
                    f"geq=lum='255*({opacity})',"
                    f"scale=w={geometry.width}:h={geometry.height}:flags=neighbor"
                    f"[{stem}_opacity_mask]",
                    f"[{stem}_fg_pre]split=2[{stem}_fg_color][{stem}_fg_alpha_src]",
                    f"[{stem}_fg_alpha_src]alphaextract[{stem}_fg_alpha]",
                    f"[{stem}_fg_alpha][{stem}_opacity_mask]"
                    f"blend=all_expr='A*B/255':shortest=1:repeatlast=0"
                    f"[{stem}_scaled_alpha]",
                    f"[{stem}_fg_color][{stem}_scaled_alpha]"
                    f"alphamerge[{foreground_rgba}]",
                )
            )
            opacity_filters = (
                "color",
                "format",
                "geq",
                "scale",
                "split",
                "alphaextract",
                "blend",
                "alphamerge",
            )

        if mode.family == "normal":
            linearize = _power_transfer_expression(FCP_NORMAL_SOURCE_OVER_GAMMA)
            encode = _power_transfer_expression(1.0 / FCP_NORMAL_SOURCE_OVER_GAMMA)
            enable = ""
            if active_frame_range is not None:
                start_frame, end_frame = active_frame_range
                enable = f":enable='gte(n,{start_frame})*lt(n,{end_frame})'"
            lines.extend(
                (
                    f"[{lower_rgba}]format=rgba64le,"
                    f"lutrgb=r='{linearize}':g='{linearize}':b='{linearize}'"
                    f"{enable}"
                    f"[{stem}_lower_linear]",
                    f"[{foreground_rgba}]format=rgba64le,"
                    f"lutrgb=r='{linearize}':g='{linearize}':b='{linearize}'"
                    f"{enable}"
                    f"[{stem}_fg_linear]",
                    f"[{stem}_lower_linear][{stem}_fg_linear]"
                    f"overlay=x=0:y=0:eof_action=pass:repeatlast=0:"
                    f"format=auto:alpha=straight{enable}"
                    f"[{stem}_composited_linear]",
                    f"[{stem}_composited_linear]"
                    f"lutrgb=r='{encode}':g='{encode}':b='{encode}'{enable},"
                    f"format=gbrap[{output}]",
                )
            )
            required = ("format", *opacity_filters, "lutrgb", "overlay")
            order = (
                "normalize_rgba",
                "foreground_opacity",
                "power_linearize",
                "source_over",
                "power_encode",
            )
        elif mode.family == "behind":
            lines.extend(
                (
                    f"[{lower_rgba}]split=2[{stem}_lower_out][{stem}_lower_clock]",
                    f"[{stem}_lower_clock]colorchannelmixer=aa=0[{stem}_transparent_clock]",
                    f"[{stem}_transparent_clock][{foreground_rgba}]overlay=x=0:y=0:eof_action=pass:"
                    f"repeatlast=0:format=auto:alpha=straight[{stem}_padded_foreground]",
                    f"[{stem}_padded_foreground][{stem}_lower_out]overlay=x=0:y=0:eof_action=pass:"
                    f"repeatlast=0:format=auto:alpha=straight[{output}]",
                )
            )
            required = (
                "format",
                *opacity_filters,
                "split",
                "colorchannelmixer",
                "overlay",
            )
            order = (
                "normalize_rgba",
                "foreground_opacity",
                "pad_foreground_to_lower_clock",
                "source_behind",
            )
        elif mode.family == "rgb":
            assert mode.ffmpeg_mode is not None
            lines.extend(
                (
                    f"[{lower_rgba}]split=3[{stem}_lower_out][{stem}_lower_mode][{stem}_lower_alpha_src]",
                    f"[{foreground_rgba}]split=2[{stem}_fg_out][{stem}_fg_mode]",
                    f"[{stem}_lower_alpha_src]alphaextract[{stem}_lower_alpha]",
                    f"[{stem}_fg_mode][{stem}_lower_mode]blend=all_mode={mode.ffmpeg_mode}:"
                    f"shortest=1:repeatlast=0[{stem}_mode_rgb]",
                    f"[{stem}_fg_out][{stem}_mode_rgb][{stem}_lower_alpha]"
                    f"maskedmerge=planes=7[{stem}_alpha_aware_fg]",
                    f"[{stem}_lower_out][{stem}_alpha_aware_fg]overlay=x=0:y=0:eof_action=pass:"
                    f"repeatlast=0:format=auto:alpha=straight[{output}]",
                )
            )
            required = (
                "format",
                *opacity_filters,
                "split",
                "alphaextract",
                "blend",
                "maskedmerge",
                "overlay",
            )
            order = (
                "normalize_rgba",
                "foreground_opacity",
                "blend_rgb",
                "respect_lower_alpha",
                "source_over",
            )
        else:
            assert mode.matte_kind is not None
            lines.extend(
                (
                    f"[{lower_rgba}]split=2[{stem}_lower_color][{stem}_lower_alpha_src]",
                    f"[{stem}_lower_alpha_src]alphaextract[{stem}_lower_alpha]",
                )
            )
            if mode.matte_kind.endswith("luma"):
                lines.extend(
                    (
                        f"[{foreground_rgba}]split=2[{stem}_matte_color][{stem}_matte_alpha_src]",
                        f"[{stem}_matte_alpha_src]alphaextract[{stem}_matte_alpha]",
                        f"[{stem}_matte_color]format=gray[{stem}_matte_luma]",
                        f"[{stem}_matte_luma][{stem}_matte_alpha]"
                        f"blend=all_expr='A*B/255':shortest=1:repeatlast=0[{stem}_matte]",
                    )
                )
            else:
                lines.append(f"[{foreground_rgba}]alphaextract[{stem}_matte]")
            mask = f"{stem}_matte"
            if mode.matte_kind.startswith("silhouette"):
                lines.append(f"[{mask}]negate[{stem}_inverse_matte]")
                mask = f"{stem}_inverse_matte"
            matte_expression = "A*B/255"
            if active_frame_range is not None:
                start_frame, end_frame = active_frame_range
                matte_expression = (
                    f"if(between(N,{start_frame + 1},{end_frame}),A*B/255,A)"
                )
            lines.extend(
                (
                    f"[{stem}_lower_alpha][{mask}]blend=all_expr="
                    f"'{matte_expression}':"
                    f"shortest=1:repeatlast=0[{stem}_output_alpha]",
                    f"[{stem}_lower_color][{stem}_output_alpha]alphamerge[{output}]",
                )
            )
            required = (
                "format",
                *opacity_filters,
                "split",
                "alphaextract",
                "blend",
                "alphamerge",
            )
            if mode.matte_kind.endswith("luma"):
                required += ("format",)
            if mode.matte_kind.startswith("silhouette"):
                required += ("negate",)
            order = (
                "normalize_rgba",
                "foreground_opacity",
                "derive_matte",
                "multiply_lower_alpha",
                "restore_lower_color",
            )

        return CompositorGraph(
            filter_graph=";".join(lines),
            output_label=output,
            mode=mode,
            opacity_expression=opacity,
            required_filters=tuple(dict.fromkeys(required)),
            operation_order=order,
        )

    def build_working_filter_graph(
        self,
        *,
        lower_label: str,
        foreground_label: str,
        output_label: str,
        prefix: str,
        pixel_contract: CompositionModuleContract,
        active_frame_range: tuple[int, int] | None = None,
    ) -> CompositorGraph:
        """Build one Normal semantic module entirely in the shared working domain.

        Main callers:
        - The contract-aware central renderer after ``PixelDomainCompiler`` has
          adapted both incoming links.

        Why this exists:
        ``build_filter_graph`` remains the standalone encoded-input contract
        used by isolated pixel oracles.  A real timeline, however, connects
        many such modules.  This method performs only this module's opacity and
        source-over semantics; transfer conversion belongs to the graph-level
        pixel-domain compiler and can therefore be fused across module edges.
        """

        lower = _validate_label(lower_label, name="lower_label")
        foreground = _validate_label(foreground_label, name="foreground_label")
        output = _validate_label(output_label, name="output_label")
        stem = _validate_label(prefix, name="prefix")
        if not isinstance(pixel_contract, CompositionModuleContract):
            raise CompositorValidationError(
                "pixel_contract must be CompositionModuleContract"
            )
        working_domains = (
            pixel_contract.lower,
            pixel_contract.foreground,
            pixel_contract.output,
        )
        # The working domain is a render-profile decision: ``reference``
        # composites in 16-bit linear light, ``fast8`` in 8-bit encoded
        # pixels.  Either way all three links must share one straight-alpha
        # domain so this module performs only opacity and source-over.
        working_pixels = {
            (domain.transfer, domain.alpha, domain.precision)
            for domain in working_domains
        }
        if len(working_pixels) != 1 or working_pixels - {
            ("fcp_linear", "straight", 16),
            ("fcp_encoded", "straight", 8),
        }:
            raise CompositorValidationError(
                "working Normal composition requires one shared straight-alpha "
                "working domain (linear 16-bit or encoded 8-bit)"
            )
        working_format = (
            "rgba64le" if pixel_contract.output.precision == 16 else "gbrap"
        )
        if self.mode.family != "normal":
            raise CompositorValidationError(
                "only Normal composition is defined in the calibrated working domain"
            )
        if active_frame_range is not None and (
            not isinstance(active_frame_range, tuple)
            or len(active_frame_range) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in active_frame_range
            )
            or active_frame_range[0] < 0
            or active_frame_range[1] <= active_frame_range[0]
        ):
            raise CompositorValidationError(
                "active_frame_range must be a non-empty half-open integer range"
            )

        enable = ""
        if active_frame_range is not None:
            start_frame, end_frame = active_frame_range
            enable = f":enable='gte(n,{start_frame})*lt(n,{end_frame})'"
        opacity = self.opacity.ffmpeg_expression(time_variable="T")
        foreground_ready = foreground
        lines: list[str] = []
        required: tuple[str, ...] = ("overlay", "format")
        order: tuple[str, ...]
        if not self.opacity.is_constant_one:
            foreground_ready = f"{stem}_foreground_opacity"
            lines.append(
                f"[{foreground}]geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
                f"a='alpha(X,Y)*({opacity})'{enable}[{foreground_ready}]"
            )
            required = ("geq", *required)
            order = ("foreground_opacity", "source_over")
        else:
            order = ("source_over",)
        # ``overlay`` has no 16-bit RGB path.  With ``format=auto`` and
        # rgba64le inputs FFmpeg negotiates yuva444p10le, i.e. a limited-range
        # YUV round trip per composite (~2% loss, and a wrong result whenever
        # the lower layer is transparent, as in transition sides and nested
        # scopes).  ``format=gbrp`` composites in planar RGB with alpha, which
        # is exact for the straight source-over this module owns.
        if active_frame_range is None:
            lines.append(
                f"[{lower}][{foreground_ready}]"
                "overlay=x=0:y=0:eof_action=pass:repeatlast=0:"
                f"format=gbrp:alpha=straight{enable},format={working_format}[{output}]"
            )
        else:
            # FFmpeg's overlay frame synchronizer still negotiates and emits
            # its chosen pixel format on disabled frames.  In a long chain,
            # that physical round trip is not an identity operation even when
            # the semantic foreground is inactive.  Keep the lower canvas on
            # a direct branch for the inactive intervals and run overlay only
            # on the authored window.  concat joins the branches by frame
            # order while preserving the existing zero-origin timeline clock.
            start_frame, end_frame = active_frame_range
            branch_labels = [f"{stem}_active_lower"]
            if start_frame > 0:
                branch_labels.insert(0, f"{stem}_prefix")
            branch_labels.append(f"{stem}_suffix")
            lines.append(
                f"[{lower}]split={len(branch_labels)}"
                + "".join(f"[{label}]" for label in branch_labels)
            )
            concat_inputs: list[str] = []
            branch_index = 0
            if start_frame > 0:
                prefix = branch_labels[branch_index]
                branch_index += 1
                prefix_trimmed = f"{stem}_prefix_trimmed"
                lines.append(
                    f"[{prefix}]trim=start_frame=0:end_frame={start_frame},"
                    f"setpts=PTS-STARTPTS[{prefix_trimmed}]"
                )
                concat_inputs.append(f"[{prefix_trimmed}]")

            active_lower = branch_labels[branch_index]
            branch_index += 1
            active_lower_trimmed = f"{stem}_active_lower_trimmed"
            active_foreground_trimmed = f"{stem}_active_foreground_trimmed"
            active_output = f"{stem}_active_output"
            lines.extend(
                (
                    f"[{active_lower}]trim=start_frame={start_frame}:"
                    f"end_frame={end_frame},setpts=PTS-STARTPTS"
                    f"[{active_lower_trimmed}]",
                    f"[{foreground_ready}]trim=start_frame={start_frame}:"
                    f"end_frame={end_frame},setpts=PTS-STARTPTS"
                    f"[{active_foreground_trimmed}]",
                    f"[{active_lower_trimmed}][{active_foreground_trimmed}]"
                    "overlay=x=0:y=0:eof_action=pass:repeatlast=0:"
                    f"format=gbrp:alpha=straight,format={working_format}"
                    f"[{active_output}]",
                )
            )
            concat_inputs.append(f"[{active_output}]")

            suffix = branch_labels[branch_index]
            suffix_trimmed = f"{stem}_suffix_trimmed"
            lines.append(
                f"[{suffix}]trim=start_frame={end_frame},"
                f"setpts=PTS-STARTPTS[{suffix_trimmed}]"
            )
            concat_inputs.append(f"[{suffix_trimmed}]")
            lines.append(
                "".join(concat_inputs)
                + f"concat=n={len(concat_inputs)}:v=1:a=0[{output}]"
            )
            required = (*required, "split", "trim", "setpts", "concat")
            order = (*order[:-1], "window_bypass", "source_over")
        return CompositorGraph(
            filter_graph=";".join(lines),
            output_label=output,
            mode=self.mode,
            opacity_expression=opacity,
            required_filters=required,
            operation_order=order,
            pixel_contract=pixel_contract,
        )


__all__ = [
    "BlendModeSpec",
    "CompositorError",
    "CompositorGraph",
    "CompositorPlan",
    "CompositorValidationError",
    "CompositorWindow",
    "OpacityPlan",
    "OpacityMaskGeometry",
    "OpacitySnapshot",
    "UnknownBlendModeError",
    "UnsupportedBlendModeError",
    "resolve_blend_mode",
]
