"""Bounded stock-FFmpeg approximations for the long-running effect cohort.

Architecture map
================

``Final Cut filter-video``
    -> registry-selected ``cohort_*`` handler
    -> closed numeric parameter validation
    -> reusable color / sampling / distortion / matte primitive
    -> ordinary FFmpeg filter chain, or one renderer-owned branch graph

The published parameter keys and defaults in this module were inspected from
the Final Cut 12.3 Motion templates installed in ``PETemplates.localized``.
Those templates are evidence for the public controls, not pixel-equivalence
evidence.  The broad mechanisms below are intentionally marked as needing A/B
tuning until a genuine Final Cut render has been reviewed.

Security and product invariants
-------------------------------

* XML can select only a named registry handler and bounded numeric controls.
* XML never supplies an FFmpeg filter, expression, shader, path, or decoder.
* Opaque FxPlug archives and unknown parameters reject the whole effect; they
  never fall through to a plausible-looking default.
* Sampling expressions are renderer-owned constants and clamp their source
  coordinates to the current frame.
* Alpha is copied unless an operation is specifically an alpha matte.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from .effect_parameters import unsupported_parameter_reason
from .model import Parameter, ResolvedEffect


COHORT_EFFECT_HANDLERS = frozenset(
    {
        "cohort_vibrancy",
        "cohort_directional_blur",
        "cohort_focus_blur",
        "cohort_radial_blur",
        "cohort_color_curves",
        "cohort_hue_saturation_curves",
        "cohort_drop_shadow",
        "cohort_crop_feather",
        "cohort_droplet",
        "cohort_earthquake",
        "cohort_fisheye",
        "cohort_vignette_mask",
        "cohort_callout",
        "cohort_camcorder",
        "cohort_cartoon",
        "cohort_kaleidoscope",
        "cohort_perspective_tile",
    }
)

BRANCHED_COHORT_HANDLERS = frozenset(
    {"cohort_drop_shadow", "cohort_focus_blur", "cohort_callout"}
)


def unsupported_cohort_effect_reason(
    handler: str,
    params: tuple[Parameter, ...],
    calibration: Mapping[str, Any],
    data: Mapping[str, str],
) -> Optional[str]:
    """Return why a cohort effect must not enter the render IR.

    Main callers:
    - ``compiler._Compiler._resolve_filter_instance``.

    Why this exists:
    Compiler coverage findings are useful diagnostics, but they do not stop a
    filter.  These handlers must fail closed so an opaque archive or malformed
    value cannot silently turn into the handler's default look.
    """

    if handler not in COHORT_EFFECT_HANDLERS:
        return None
    if data:
        return "portable cohort effects do not decode opaque filter data"

    return unsupported_parameter_reason(params, calibration)


def cohort_effect_filters(effect: ResolvedEffect) -> list[str]:
    """Build one validated cohort effect from renderer-owned primitives.

    Main callers:
    - ``ffmpeg._effect_filters`` while preserving FCPXML filter order.
    """

    handler = effect.handler
    if handler == "cohort_vibrancy":
        amount = _scalar(
            effect, "9999/987152514/100/987152515/2/100", "Amount", 0.4976076555
        )
        protect = _scalar(
            effect, "9999/987152514/100/987175526/2/100", "Protect Skin", 0.0
        )
        filters: list[str] = []
        brightness = max(0.0, min(0.2, (0.5 - amount) * 0.4))
        contrast = 1.0 + max(0.0, min(0.4, (0.5 - amount) * 0.8))
        if brightness > 0.0:
            filters.append(
                f"eq=brightness={_number(brightness)}:contrast={_number(contrast)}"
            )
        filters.append(
            "vibrance="
            f"intensity={_number((amount - 0.5) * (4.0 / 3.0))}:"
            f"rbal={_number(1.0 - protect * 0.15)}:gbal=1:bbal=1"
        )
        return filters
    if handler == "cohort_directional_blur":
        return [_directional_blur_filter(effect)]
    if handler == "cohort_focus_blur":
        # The branched graph retains a sharp source while applying a true
        # Gaussian field outside the bounded focus region.
        return []
    if handler == "cohort_radial_blur":
        return [_radial_blur_filter(effect)]
    if handler == "cohort_color_curves":
        # Final Cut's edited curve points live in an opaque FxPlug archive.
        # Its archive-free default is neutral, so keep a named no-op curve while
        # rejecting arbitrary archive bytes before they reach this handler.
        return ["curves=master='0/0 1/1'"]
    if handler == "cohort_hue_saturation_curves":
        return ["huesaturation=colors=r+y+m:saturation=0:strength=0:lightness=0"]
    if handler == "cohort_crop_feather":
        return [_crop_feather_filter(effect)]
    if handler == "cohort_droplet":
        return [_droplet_filter(effect)]
    if handler == "cohort_earthquake":
        return [_earthquake_filter(effect)]
    if handler == "cohort_fisheye":
        return [_fisheye_filter(effect)]
    if handler == "cohort_vignette_mask":
        return [_vignette_mask_filter(effect)]
    if handler == "cohort_callout":
        return []
    if handler == "cohort_camcorder":
        return _camcorder_filters(effect)
    if handler == "cohort_cartoon":
        return _cartoon_filters(effect)
    if handler == "cohort_kaleidoscope":
        return [_kaleidoscope_filter(effect)]
    if handler == "cohort_perspective_tile":
        return [_perspective_tile_filter(effect)]
    return []


def cohort_effect_graph_lines(
    input_label: str,
    output_label: str,
    effect: ResolvedEffect,
    *,
    prefix: str,
) -> list[str]:
    """Build a complex graph for a cohort effect that needs frame branches.

    Main callers:
    - ``ffmpeg._branched_effect_graph`` for Focus Blur and Drop Shadow.

    Why this exists:
    Both effects must retain the original frame while separately processing a
    second branch. A serial filter list cannot represent that ownership.
    """

    if effect.handler == "cohort_focus_blur":
        return _focus_blur_graph_lines(input_label, output_label, effect, prefix=prefix)
    if effect.handler == "cohort_callout":
        return _callout_graph_lines(input_label, output_label, effect, prefix=prefix)
    if effect.handler != "cohort_drop_shadow":
        return []
    opacity = _scalar(effect, "2", "Opacity", 0.75)
    # Final Cut's exported Classic preset reports Blur=30 but its visible
    # 160x90 Gaussian spread is about three pixels. Its derived Position is not
    # exported; the revealing two-source layered A/B measures a 4 px diagonal
    # offset. These are preset defaults, not accepted XML controls.
    position = _vector(effect, "3", "Position", (4.0, -4.0))
    blur = _scalar(effect, "4", "Blur", 30.0)
    falloff = round(_scalar(effect, "5", "Blur Falloff", 1.0))
    perspective = _scalar(effect, "7", "Perspective Amount", 0.0)
    sigma = max(
        0.01,
        blur * (0.06 + 0.04 * falloff) * (1.0 + 0.25 * abs(perspective)),
    )
    # FCP's image plane has +y upward; overlay coordinates have +y downward.
    offset_x = position[0]
    offset_y = -position[1] * (1.0 + 0.5 * perspective)
    canvas = f"{prefix}canvas"
    base = f"{prefix}base"
    shadow = f"{prefix}shadow"
    clear = f"{prefix}clear"
    blurred = f"{prefix}blurred"
    with_shadow = f"{prefix}withshadow"
    return [
        f"[{input_label}]split=3[{canvas}][{base}][{shadow}]",
        f"[{canvas}]colorchannelmixer=rr=0:gg=0:bb=0:aa=0[{clear}]",
        f"[{shadow}]colorchannelmixer=rr=0:gg=0:bb=0:aa={_number(opacity)},"
        f"gblur=sigma={_number(sigma)}:steps=2[{blurred}]",
        f"[{clear}][{blurred}]overlay=x='{_number(offset_x)}':y='{_number(offset_y)}':"
        f"eval=frame:eof_action=pass:repeatlast=0:format=auto[{with_shadow}]",
        f"[{with_shadow}][{base}]overlay=x=0:y=0:eof_action=pass:repeatlast=0:format=auto[{output_label}]",
    ]


def _focus_blur_graph_lines(
    input_label: str,
    output_label: str,
    effect: ResolvedEffect,
    *,
    prefix: str,
) -> list[str]:
    """Blend a true Gaussian field outside a bounded elliptical focus region."""

    amount = _scalar(effect, "9999/11249/100/11250/2/100", "Amount", 0.3)
    softness = _scalar(effect, "9999/11249/100/999242278/2/100", "Softness", 0.57862)
    emphasis = _scalar(effect, "9999/11249/100/999234268/2/100", "Emphasis", 0.5)
    width = _scalar(effect, "9999/11249/100/1978911431/2/100", "Width", 0.5)
    height = _scalar(effect, "9999/11249/100/1978911462/2/100", "Height", 0.25)
    center = _vector(effect, "9999/999241996/100/999241915/2", "Center", (0.5, 0.5))
    sigma = max(0.01, amount * (3.0 + 3.5 * emphasis))
    cx = f"W*{_number(center[0])}"
    cy = f"H*{_number(1.0 - center[1])}"
    distance = (
        "sqrt("
        f"pow((X-({cx}))/max(W*{_number(width * 0.84)},1),2)+"
        f"pow((Y-({cy}))/max(H*{_number(height * 0.93)},1),2))"
    )
    outside = f"clip((({distance})-1)/{_number(max(softness * 1.4, 0.001))},0,1)"
    base = f"{prefix}base"
    blurred = f"{prefix}blurred"
    weighted = f"{prefix}weighted"
    alpha = f"alpha(X,Y)*({outside})"
    return [
        f"[{input_label}]format=rgba,split=2[{base}][{blurred}]",
        f"[{blurred}]gblur=sigma={_number(sigma)}:steps=2:planes=0x7,"
        f"{_identity_rgb_with_alpha(alpha)}[{weighted}]",
        f"[{base}][{weighted}]overlay=x=0:y=0:eof_action=pass:repeatlast=0:"
        f"format=auto[{output_label}]",
    ]


def _callout_graph_lines(
    input_label: str,
    output_label: str,
    effect: ResolvedEffect,
    *,
    prefix: str,
) -> list[str]:
    """Approximate Reframe/Callout's dimmed field and magnified inset.

    Main callers:
    - ``cohort_effect_graph_lines`` for the genuine installed Reframe UID.

    The default Motion template dims and softens the full source, then grows a
    crop while moving it from the center toward the lower-right.  The crop and
    final placement below were measured from the genuine 30-frame reference;
    keeping the motion in the renderer graph avoids freezing the cut-out at its
    final geometry. No XML controls enter this default-only graph.
    """

    original = f"{prefix}original"
    field_source = f"{prefix}fieldsource"
    field = f"{prefix}field"
    inset = f"{prefix}inset"
    built_in = f"{prefix}builtin"
    return [
        f"[{input_label}]format=rgba,split=3[{original}][{field_source}][{inset}]",
        f"[{field_source}]gblur=sigma=4:steps=2:planes=0x7,"
        f"eq=brightness=-0.28:saturation=0.6[{field}]",
        f"[{original}][{field}]blend=all_expr='A*(1-(3*pow(clip(T/0.6,0,1),2)-"
        f"2*pow(clip(T/0.6,0,1),3)))+B*(3*pow(clip(T/0.6,0,1),2)-"
        f"2*pow(clip(T/0.6,0,1),3))'[{built_in}]",
        f"[{inset}]crop=w='0.24*iw':h='0.46*ih':x='0.39*iw':y='0.18*ih',"
        "drawbox=x=0:y=0:w=iw:h=ih:c=white@0.65:t=1,"
        "scale=w='iw*(0.68+0.68*clip(t/0.8,0,1))':"
        "h='ih*(0.775+0.775*clip(t/0.8,0,1))':eval=frame,"
        "fade=t=in:st=0:d=0.3:alpha=1["
        f"{prefix}window]",
        f"[{built_in}][{prefix}window]overlay="
        "x='W*(0.42+0.18*(3*pow(clip(t/0.8,0,1),2)-"
        "2*pow(clip(t/0.8,0,1),3)))':y='0.23*H':"
        f"eof_action=pass:repeatlast=0:format=auto[{output_label}]",
    ]


def _directional_blur_filter(effect: ResolvedEffect) -> str:
    """Sample a bounded line without rotate/crop corner artifacts."""

    # The Motion template publishes 25, but a genuine import/roundtrip without
    # an accepted Amount channel renders near-neutral in Final Cut 12.3.
    amount = _scalar(effect, "9999/986883358/100/986884620/1", "Amount", 0.0)
    angle = _scalar(effect, "9999/986883358/100/986884620/2", "Angle", 0.0)
    # Motion/FCPXML serializes angle channels in radians even though Final Cut
    # presents them as degrees in the inspector.
    radians = angle
    span_x = amount * 0.5 * math.cos(radians)
    span_y = -amount * 0.5 * math.sin(radians)
    taps = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)

    def channel_expression(channel: str) -> str:
        samples = [
            f"{channel}(clip(X+{_number(span_x * tap)},0,W-1),"
            f"clip(Y+{_number(span_y * tap)},0,H-1))"
            for tap in taps
        ]
        return f"({'+'.join(samples)})/{len(samples)}"

    return _sample_filter(channel_expression)


def _radial_blur_filter(effect: ResolvedEffect) -> str:
    amount = _scalar(
        effect, "9999/986883370/100/986883376/2/100", "Amount", 0.2503840246
    )
    center = _vector(effect, "9999/986883358/100/986883365/1", "Center", (0.5, 0.5))
    cx = f"W*{_number(center[0])}"
    cy = f"H*{_number(1.0 - center[1])}"
    taps = tuple(index / 8.0 for index in range(-8, 9))

    def channel_expression(channel: str) -> str:
        samples = []
        for tap in taps:
            radians = amount * 1.7 * tap
            cosine = _number(math.cos(radians))
            sine = _number(math.sin(radians))
            x = f"clip(({cx})+(X-({cx}))*{cosine}-(Y-({cy}))*{sine},0,W-1)"
            y = f"clip(({cy})+(X-({cx}))*{sine}+(Y-({cy}))*{cosine},0,H-1)"
            samples.append(f"{channel}({x},{y})")
        return f"({'+'.join(samples)})/{len(samples)}"

    return _sample_filter(channel_expression)


def _fisheye_filter(effect: ResolvedEffect) -> str:
    """Apply a bounded radial bulge without lenscorrection's black borders.

    Final Cut's positive Amount magnifies the center of the selected circular
    region. Mapping each output pixel toward the effect center creates that
    bulge; clamping source coordinates keeps every sample inside the frame.
    """

    amount = _scalar(effect, "9999/10193/100/10194/2/100", "Amount", 0.575)
    radius = _scalar(effect, "9999/10153/100/10158/1", "Radius", 2.58)
    center = _vector(effect, "9999/10153/100/10158/3", "Center", (0.5, 0.5))
    cx = f"W*{_number(center[0])}"
    cy = f"H*{_number(1.0 - center[1])}"
    normalized_radius_squared = (
        f"pow((X-({cx}))/max(W*0.5*{_number(radius)},1),2)+"
        f"pow((Y-({cy}))/max(H*0.5*{_number(radius)},1),2)"
    )
    strength = max(-5.0, min(0.95, (amount - 0.564) * 9.406))
    scale = f"1-{_number(strength)}*pow(clip(1-({normalized_radius_squared}),0,1),2.95)"
    source_x = f"({cx})+(X-({cx}))*({scale})"
    source_y = f"({cy})+(Y-({cy}))*({scale})"
    in_bounds = f"between(({source_x}),0,W-1)*between(({source_y}),0,H-1)"
    return _sample_filter(
        lambda channel: f"if({in_bounds},{channel}({source_x},{source_y}),0)"
    )


def _crop_feather_filter(effect: ResolvedEffect) -> str:
    width = _scalar(effect, "9999/989379745/100/989379746/2/100", "Width", 0.25)
    height = _scalar(effect, "9999/989379745/100/989379838/2/100", "Height", 0.25)
    roundness = _scalar(
        effect, "9999/988494964/100/988494966/2/353/144", "Roundness", 0.0
    )
    feather = _scalar(effect, "9999/989379745/100/989379995/2/100", "Feather", 0.5)
    position = _vector(
        effect, "9999/988494964/100/988494966/1/100/101", "Position", (0.0, 0.0)
    )
    cx = f"W*(0.5+{_number(position[0])})"
    cy = f"H*(0.5-{_number(position[1])})"
    half_width = f"max(0.001,W*{_number(min(0.5, width * 1.607))})"
    half_height = f"max(0.001,H*{_number(min(0.5, height * 1.533))})"
    # Final Cut's zero-Roundness default is a rectangle; increasing Roundness
    # moves toward an ellipse. A superellipse therefore runs high-to-low here.
    exponent = 32.0 - 30.0 * roundness
    distance = (
        f"pow(pow(abs((X-({cx}))/({half_width})),{_number(exponent)})+"
        f"pow(abs((Y-({cy}))/({half_height})),{_number(exponent)}),"
        f"1/{_number(exponent)})"
    )
    # The published slider is inverted relative to alpha-ramp width: Final Cut
    # renders value 0 with the broadest feather and value 1 with a hard edge.
    # The genuine five-point response sweep established this direction.
    edge = max(0.001, (1.0 - feather) * 0.566)
    alpha = f"alpha(X,Y)*clip((1+{_number(edge)}-({distance}))/{_number(edge)},0,1)"
    return _identity_rgb_with_alpha(alpha)


def _droplet_filter(effect: ResolvedEffect) -> str:
    center = _vector(effect, "9999/10006/100/10011/1", "Center", (0.5, 0.5))
    intensity = _scalar(effect, "9999/10012/100/10013/2/100", "Intensity", 0.7618)
    radius = _scalar(effect, "9999/10006/100/10011/2", "Radius", 324.0)
    thickness = _scalar(effect, "9999/10006/100/10011/4", "Thickness", 43.0)
    cx = f"W*{_number(center[0])}"
    cy = f"H*{_number(1.0 - center[1])}"
    distance = f"sqrt(pow(X-({cx}),2)+pow(Y-({cy}),2))"
    # Motion's published pixel units use its template canvas, not the current
    # output raster. Normalize them by height so the ring survives at 90p and
    # scales proportionally at 1080p/4K.
    radius_pixels = f"H*{_number(radius / 833.0)}"
    thickness_pixels = f"max(H*{_number(thickness / 2345.0)},0.5)"
    displacement = (
        f"{_number(intensity * 2.55)}*({thickness_pixels})*"
        f"exp(-pow((({distance})-({radius_pixels}))/({thickness_pixels}),2))"
    )
    source_x = f"clip(({cx})+(X-({cx}))*(1-({displacement})/max(({distance}),1)),0,W-1)"
    source_y = f"clip(({cy})+(Y-({cy}))*(1-({displacement})/max(({distance}),1)),0,H-1)"
    return _coordinate_filter(source_x, source_y)


def _earthquake_filter(effect: ResolvedEffect) -> str:
    amount = _scalar(effect, "9999/10062/100/10063/2/100", "Amount", 0.0979)
    layers = _scalar(effect, "9999/10039/100/10044/4", "Layers", 3.0)
    epicenter = _vector(effect, "9999/10039/100/10044/5", "Epicenter", (0.5, 0.5))
    # Genuine low/high A/B shows sub-pixel translation at Amount=0.35. The
    # earlier coefficient moved content by more than two pixels at 160x90 and
    # an unconditional five-frame tmix blurred even Amount=0.  Keep the
    # deterministic two-frequency path, but match Final Cut's measured scale.
    amplitude = amount * (0.0034 + 0.0003 * layers)
    phase_x = epicenter[0] * math.pi
    phase_y = epicenter[1] * math.pi
    source_x = (
        f"clip(X+W*{_number(amplitude)}*(sin(N*1.71+{_number(phase_x)})+"
        f"0.35*sin(N*4.13)),0,W-1)"
    )
    source_y = (
        f"clip(Y+H*{_number(amplitude)}*(cos(N*1.37+{_number(phase_y)})+"
        f"0.35*sin(N*3.29)),0,H-1)"
    )
    return _coordinate_filter(source_x, source_y)


def _vignette_mask_filter(effect: ResolvedEffect) -> str:
    size = _scalar(effect, "9999/999156327/100/999156368/1", "Size:", 0.777)
    falloff = _scalar(effect, "9999/999156327/100/999156368/5", "Falloff:", 0.193)
    center = _vector(effect, "9999/999156327/100/999156368/6", "Center:", (0.5, 0.5))
    cx = f"W*{_number(center[0])}"
    cy = f"H*{_number(1.0 - center[1])}"
    distance = (
        "sqrt("
        f"pow((X-({cx}))/max(W*0.5*{_number(size)},1),2)+"
        f"pow((Y-({cy}))/max(H*0.5*{_number(size)},1),2))"
    )
    edge = max(falloff, 0.001)
    alpha = f"alpha(X,Y)*clip((1+{_number(edge)}-({distance}))/{_number(edge)},0,1)"
    return _identity_rgb_with_alpha(alpha)


def _camcorder_filters(effect: ResolvedEffect) -> list[str]:
    amount = _scalar(effect, "9999/999213243/100/999214138/2/100", "Amount", 1.0)
    recording = _scalar(effect, "9999/999213243/100/999213300/2/100", "Recording", 1.0)
    size = _scalar(effect, "9999/999213243/100/999214036/2/100", "Size", 0.1489361702)
    battery = _scalar(
        effect, "9999/999213243/100/999213244/2/100", "Battery Level", 1.0
    )
    output = [
        "eq="
        f"contrast={_number(1.0 + 0.01 * amount)}:"
        f"brightness={_number(-0.005 * amount)}:"
        f"saturation={_number(1.0 - 0.03 * amount)}"
    ]
    if recording >= 0.5:
        thickness = max(1, round(size * 4))
        primary_alpha = _number(0.82 * amount)
        secondary_alpha = _number(0.65 * amount)
        guide_alpha = _number(0.28 * amount)
        battery_alpha = _number(0.75 * amount)
        battery_fill_width = _number(0.085 * battery)
        output.extend(
            [
                # The default HUD is a renderer-owned vector reconstruction.
                # A red recording dot and compact R/E/C strokes replace the
                # old oversized white block glyph that read as "PEC".
                f"drawbox=x=.037*iw:y=.067*ih:w=.025*iw:h=.045*ih:c=red@{primary_alpha}:t=fill",
                f"drawbox=x=.066*iw:y=.064*ih:w=.008*iw:h=.06*ih:c=white@{primary_alpha}:t=fill",
                f"drawbox=x=.066*iw:y=.064*ih:w=.020*iw:h=.012*ih:c=white@{primary_alpha}:t=fill",
                f"drawbox=x=.066*iw:y=.088*ih:w=.020*iw:h=.012*ih:c=white@{primary_alpha}:t=fill",
                f"drawbox=x=.081*iw:y=.064*ih:w=.008*iw:h=.032*ih:c=white@{primary_alpha}:t=fill",
                f"drawbox=x=.079*iw:y=.099*ih:w=.008*iw:h=.025*ih:c=white@{primary_alpha}:t=fill",
                f"drawbox=x=.094*iw:y=.064*ih:w=.008*iw:h=.06*ih:c=white@{primary_alpha}:t=fill",
                f"drawbox=x=.094*iw:y=.064*ih:w=.023*iw:h=.012*ih:c=white@{primary_alpha}:t=fill",
                f"drawbox=x=.094*iw:y=.088*ih:w=.020*iw:h=.012*ih:c=white@{primary_alpha}:t=fill",
                f"drawbox=x=.094*iw:y=.112*ih:w=.023*iw:h=.012*ih:c=white@{primary_alpha}:t=fill",
                f"drawbox=x=.123*iw:y=.064*ih:w=.024*iw:h=.012*ih:c=white@{primary_alpha}:t=fill",
                f"drawbox=x=.123*iw:y=.064*ih:w=.008*iw:h=.06*ih:c=white@{primary_alpha}:t=fill",
                f"drawbox=x=.123*iw:y=.112*ih:w=.024*iw:h=.012*ih:c=white@{primary_alpha}:t=fill",
                # Battery outline, terminal, and an independent fill whose
                # width (not the outline width) follows Battery Level.
                f"drawbox=x=.815*iw:y=.065*ih:w=.12*iw:h=.052*ih:c=white@{battery_alpha}:t={thickness}",
                f"drawbox=x=.94*iw:y=.079*ih:w=.012*iw:h=.024*ih:c=white@{battery_alpha}:t=fill",
                f"drawbox=x=.824*iw:y=.076*ih:w={battery_fill_width}*iw:h=.03*ih:c=white@{_number(0.35 * amount)}:t=fill",
                # Final Cut's lower framing guides are part of the default HUD.
                f"drawbox=x=.10*iw:y=.48*ih:w={thickness}:h=.38*ih:c=white@{guide_alpha}:t=fill",
                f"drawbox=x=.10*iw:y=.84*ih:w=.10*iw:h={thickness}:c=white@{guide_alpha}:t=fill",
                f"drawbox=x=.90*iw:y=.48*ih:w={thickness}:h=.38*ih:c=white@{guide_alpha}:t=fill",
                f"drawbox=x=.80*iw:y=.84*ih:w=.10*iw:h={thickness}:c=white@{guide_alpha}:t=fill",
                f"drawbox=x=.655*iw:y=.78*ih:w=.055*iw:h=.075*ih:c=white@{secondary_alpha}:t={thickness}",
                f"drawbox=x=.67*iw:y=.758*ih:w=.025*iw:h=.022*ih:c=white@{secondary_alpha}:t=fill",
            ]
        )
    return output


def _cartoon_filters(effect: ResolvedEffect) -> list[str]:
    """Scale the fixed Cartoon approximation while preserving its old default."""

    amount = _scalar(effect, "9999/100309/100/100310/2/100", "Amount", 1.0)
    if amount <= 0.0:
        return []
    poster_step = 1 + round(3.0 * amount)
    return [
        f"gblur=sigma={_number(0.35 * amount)}:steps=1:planes=0x7",
        "lutrgb="
        f"r='floor(val/{poster_step})*{poster_step}':"
        f"g='floor(val/{poster_step})*{poster_step}':"
        f"b='floor(val/{poster_step})*{poster_step}'",
        f"unsharp=5:5:{_number(0.2 * amount)}:5:5:0",
    ]


def _kaleidoscope_filter(effect: ResolvedEffect) -> str:
    center = _vector(effect, "9999/986883879/100/986883884/1", "Center", (0.5, 0.5))
    offset = _scalar(
        effect, "9999/986883879/100/986883884/3", "Offset Angle", 0.0137077839
    )
    segment = _scalar(
        effect, "9999/986883879/100/986883884/2", "Segment Angle", math.pi / 8.0
    )
    mix = _scalar(effect, "9999/986883879/100/986883884/10001", "Mix", 1.0)
    cx = f"W*{_number(center[0])}"
    cy = f"H*{_number(1.0 - center[1])}"
    radius = f"sqrt(pow(X-({cx}),2)+pow(Y-({cy}),2))"
    # FCP's published Offset Angle is measured from the middle of a segment;
    # geq's polar angle begins at its edge, hence the half-segment correction.
    theta = f"atan2(Y-({cy}),X-({cx}))+{_number(offset - segment / 2.0)}"
    folded = f"abs(mod(({theta})+100*{_number(segment)},{_number(segment)})-{_number(segment / 2.0)})"
    source_x = f"clip(({cx})+({radius})*cos({folded}),0,W-1)"
    source_y = f"clip(({cy})+({radius})*sin({folded}),0,H-1)"

    def channel_expression(channel: str) -> str:
        return (
            f"{_number(1.0 - mix)}*{channel}(X,Y)+"
            f"{_number(mix)}*{channel}({source_x},{source_y})"
        )

    return _sample_filter(channel_expression)


def _perspective_tile_filter(effect: ResolvedEffect) -> str:
    amount = _scalar(effect, "9999/1919299884/100/1919299885/2/100", "Amount", 1.0)
    top_left = _vector(
        effect, "9999/1919300109/100/986883922/1", "Top Left", (0.30977, 0.76747)
    )
    top_right = _vector(
        effect, "9999/1919300109/100/986883922/2", "Top Right", (0.83315, 0.75799)
    )
    bottom_right = _vector(
        effect, "9999/1919300109/100/986883922/3", "Bottom Right", (0.77646, 0.12456)
    )
    bottom_left = _vector(
        effect, "9999/1919300109/100/986883922/4", "Bottom Left", (0.29539, 0.10524)
    )
    mix = _scalar(effect, "9999/1919299884/100/1919300128/2/100", "Mix", 1.0)

    weight = amount * mix
    default_top_left = (0.30977, 0.76747)
    default_top_right = (0.83315, 0.75799)
    default_bottom_right = (0.77646, 0.12456)
    default_bottom_left = (0.29539, 0.10524)

    def quadrilateral_terms(
        tl: tuple[float, float],
        tr: tuple[float, float],
        br: tuple[float, float],
        bl: tuple[float, float],
    ) -> tuple[float, float, float, float, float, float]:
        return (
            sum(point[0] for point in (tl, tr, br, bl)) / 4.0,
            sum(point[1] for point in (tl, tr, br, bl)) / 4.0,
            ((tr[0] - tl[0]) + (br[0] - bl[0])) / 2.0,
            ((tl[1] - bl[1]) + (tr[1] - br[1])) / 2.0,
            ((tl[0] - bl[0]) + (tr[0] - br[0])) / 2.0,
            ((tr[1] - tl[1]) + (br[1] - bl[1])) / 2.0,
        )

    reviewed = quadrilateral_terms(
        default_top_left,
        default_top_right,
        default_bottom_right,
        default_bottom_left,
    )
    authored = quadrilateral_terms(top_left, top_right, bottom_right, bottom_left)
    center_x, center_y, span_x, span_y, shear_x, shear_y = (
        authored[index] - reviewed[index] for index in range(6)
    )
    # The genuine default is a repeated affine lattice: each tile is a skewed
    # parallelogram, not an axis-aligned thumbnail. Constants below were fit
    # across three real-video frames at the reviewed Amount/Mix/corner state.
    # Corner deltas adjust the same bounded lattice coefficients. Every delta
    # is zero at the historical defaults, preserving the existing render.
    tile_scale_x = 1.0 + (0.642 + 1.5 * span_x) * weight
    tile_scale_y = 1.0 + (0.609 + 1.5 * span_y) * weight
    cross_x = (0.969 + 2.0 * shear_x) * weight
    cross_y = (-0.172 + 2.0 * shear_y) * weight
    phase_x = (1.033 + 2.0 * center_x) * weight
    phase_y = (1.097 - 2.0 * center_y) * weight
    denominator = (
        f"1+{_number((0.00972 + 0.2 * shear_x) * weight)}*X/W+"
        f"{_number((0.09197 + 0.2 * shear_y) * weight)}*Y/H"
    )
    source_x = (
        f"mod((X*{_number(tile_scale_x)}+({_number(cross_x)})*(Y-H/2)+"
        f"W*{_number(phase_x)})/({denominator})+100*W,W)"
    )
    source_y = (
        f"mod((Y*{_number(tile_scale_y)}+({_number(cross_y)})*(X-W/2)+"
        f"H*{_number(phase_y)})/({denominator})+100*H,H)"
    )
    return _coordinate_filter(source_x, source_y)


def _coordinate_filter(source_x: str, source_y: str) -> str:
    return _sample_filter(lambda channel: f"{channel}({source_x},{source_y})")


def _sample_filter(channel_expression: Any) -> str:
    return (
        "geq="
        f"r='{channel_expression('r')}':"
        f"g='{channel_expression('g')}':"
        f"b='{channel_expression('b')}':"
        "a='alpha(X,Y)'"
    )


def _identity_rgb_with_alpha(alpha: str) -> str:
    return f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='{alpha}'"


def _scalar(effect: ResolvedEffect, key: str, name: str, default: float) -> float:
    """Read a compiler-validated scalar, or use the omitted-control default."""

    raw = _raw_value(effect, key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - compiler validation owns this edge.
        raise ValueError(f"validated control {name!r} became nonnumeric") from exc


def _vector(
    effect: ResolvedEffect,
    key: str,
    name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    """Read a compiler-validated two-component value without normalizing it."""

    raw = _raw_value(effect, key)
    if raw is None:
        return default
    pieces = raw.replace(",", " ").split()
    try:
        return float(pieces[0]), float(pieces[1])
    except (
        ValueError,
        IndexError,
    ) as exc:  # pragma: no cover - compiler owns this edge.
        raise ValueError(f"validated control {name!r} became malformed") from exc


def _raw_value(effect: ResolvedEffect, key: str) -> Optional[str]:
    """Return the exact-key static value after compiler validation.

    Main callers:
    - ``_scalar`` and ``_vector`` in this module.

    Display-name lookup is intentionally absent. A label such as ``Amount`` is
    not a stable identifier across Motion templates.
    """

    for parameter in effect.params:
        if parameter.key == key:
            return parameter.value
    return None


def _number(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.10f}".rstrip("0").rstrip(".")
