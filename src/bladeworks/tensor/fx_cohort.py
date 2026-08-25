"""Tensor ports for the approved Cartoon, Camcorder, Drop Shadow, and Focus Blur cohort.

Architecture map
================

``ResolvedEffect``
    -> validate the capability registry's closed static parameter contract
    -> freeze the CPU cohort builder's numeric constants in a payload
    -> convert the premultiplied-linear canvas to straight RGBA8 code values
    -> run reusable Gaussian, YUV colour, vector-drawing, or branch-composite stages
    -> convert the resulting straight RGBA8 frame back to premultiplied linear light

The CPU implementation in ``core.cohort_effects`` remains the semantic owner. This
module ports those formulas without widening their accepted parameter surface. In
particular, Drop Shadow remains default-only because its explicit controls are not
present in the reviewed capability contract.

Main callers:
- ``tensor.effects.lower_effect`` through the effect-port registry.
- ``tensor.effects.apply_effects`` for each rendered layer frame.

Why this exists:
These four effects use split/process/recombine graphs or deterministic HUD drawing,
so they do not fit the one-filter ports in ``fx_basic`` and ``fx_warp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from ..core.effect_parameters import unsupported_parameter_reason
from ..core.model import ResolvedEffect
from .color import code_to_premultiplied, premultiplied_to_code
from .effects import OVERSCAN_SPLICE, ApplyContext, EffectPort, LowerContext, register
from .fx_basic import unsharp_luma
from .fx_branched import overlay_yuva444p
from .fx_color import BridgeLink, eq_process_lut, rgba8_to_yuva444p8, yuva444p8_to_rgba8
from .support import reject
from .tr_equirect import gblur_gbrap


CARTOON_HANDLER = "cohort_cartoon"
CAMCORDER_HANDLER = "cohort_camcorder"
DROP_SHADOW_HANDLER = "cohort_drop_shadow"
FOCUS_BLUR_HANDLER = "cohort_focus_blur"

CARTOON_AMOUNT = "9999/100309/100/100310/2/100"
CAMCORDER_AMOUNT = "9999/999213243/100/999214138/2/100"
CAMCORDER_RECORDING = "9999/999213243/100/999213300/2/100"
CAMCORDER_SIZE = "9999/999213243/100/999214036/2/100"
CAMCORDER_BATTERY = "9999/999213243/100/999213244/2/100"
FOCUS_AMOUNT = "9999/11249/100/11250/2/100"
FOCUS_SOFTNESS = "9999/11249/100/999242278/2/100"
FOCUS_EMPHASIS = "9999/11249/100/999234268/2/100"
FOCUS_WIDTH = "9999/11249/100/1978911431/2/100"
FOCUS_HEIGHT = "9999/11249/100/1978911462/2/100"


def _validate(effect: ResolvedEffect) -> None:
    """Reject any value the compiler's reviewed static parameter ABI would reject.

    Main callers:
    - Every lowerer in this module.

    The compiler normally performs this check first. Repeating it at the tensor port
    boundary keeps direct API use fail-closed and makes the port independently testable.
    """

    if effect.data:
        raise reject(
            "effect (unsupported parameters)",
            f"{effect.path}: {effect.name or effect.handler} contains opaque filter data",
        )
    reason = unsupported_parameter_reason(effect.params, effect.calibration)
    if reason is not None:
        raise reject(
            "effect (unsupported parameters)",
            f"{effect.path}: {effect.name or effect.handler}: {reason}",
        )


def _raw(effect: ResolvedEffect, key: str) -> str | None:
    return next((parameter.value for parameter in effect.params if parameter.key == key), None)


def _scalar(effect: ResolvedEffect, key: str, default: float) -> float:
    value = _raw(effect, key)
    if value is not None:
        return float(value)
    definition = effect.calibration.get(key) if isinstance(effect.calibration, Mapping) else None
    if isinstance(definition, Mapping) and "default" in definition:
        return float(definition["default"])
    return default


def _rgba8(canvas: torch.Tensor) -> torch.Tensor:
    return premultiplied_to_code(canvas).round().clamp(0.0, 255.0).to(torch.int32)


def _canvas(rgba8: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return code_to_premultiplied(rgba8.to(dtype))


def _option_float(value: float) -> float:
    """Mirror an FFmpeg AVOption stored as float32."""

    return float(np.float32(value))


def _bridge(effect: ResolvedEffect, ctx: LowerContext) -> BridgeLink:
    text = ctx.reference_effect_link
    if text is None:
        raise reject(
            "effect (unsupported parameters)",
            f"{effect.path}: no resolved reference pixel link for {effect.name or effect.handler}",
        )
    try:
        parsed = BridgeLink.parse(text)
        return BridgeLink("rgba", parsed.colorspace, parsed.color_range)
    except ValueError as error:
        raise reject(
            "effect (unsupported parameters)",
            f"{effect.path}: {effect.name or effect.handler} on {text}: {error}",
        ) from error


def _over(main: torch.Tensor, overlay: torch.Tensor) -> torch.Tensor:
    """Composite straight RGBA8 ``overlay`` over ``main`` using premultiplied math.

    Main callers:
    - Focus Blur's weighted blur branch.
    - Drop Shadow's shadow and source branches.

    Both inputs are full-frame float code values. The conversion to premultiplied
    values prevents dark or bright fringes around partial-alpha source pixels.
    """

    main_alpha = main[3:4] / 255.0
    over_alpha = overlay[3:4] / 255.0
    out_alpha = over_alpha + main_alpha * (1.0 - over_alpha)
    premul = overlay[:3] * over_alpha + main[:3] * main_alpha * (1.0 - over_alpha)
    rgb = torch.where(out_alpha > 0.0, premul / out_alpha.clamp_min(1e-12), torch.zeros_like(premul))
    return torch.cat((rgb, out_alpha * 255.0), dim=0).round().clamp(0.0, 255.0)


# Cartoon -------------------------------------------------------------------


@dataclass(frozen=True)
class CartoonPayload:
    amount: float
    poster_step: int
    link: BridgeLink


def _lower_cartoon(effect: ResolvedEffect, ctx: LowerContext) -> CartoonPayload:
    _validate(effect)
    amount = _scalar(effect, CARTOON_AMOUNT, 1.0)
    return CartoonPayload(
        amount=amount,
        poster_step=1 + round(3.0 * amount),
        link=_bridge(effect, ctx),
    )


def _apply_cartoon(payload: CartoonPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    if payload.amount <= 0.0:
        return canvas
    code = _rgba8(canvas).to(canvas.dtype)
    sigma = _option_float(0.35 * payload.amount)
    rgb = gblur_gbrap(code[:3], sigma=sigma, sigma_v=sigma, steps=1)
    rgb = torch.floor(rgb / payload.poster_step) * payload.poster_step
    rgba = torch.cat((rgb, code[3:4]), dim=0).round().to(torch.int32)
    yuva = rgba8_to_yuva444p8(rgba, payload.link).long()
    amount = int(_option_float(0.2 * payload.amount) * 65536.0)
    yuva[0] = unsharp_luma(yuva[0], amount)
    return _canvas(yuva444p8_to_rgba8(yuva.to(torch.int32), payload.link), canvas.dtype)


# Camcorder -----------------------------------------------------------------


@dataclass(frozen=True)
class CamcorderPayload:
    amount: float
    recording: bool
    size: float
    battery: float
    link: BridgeLink
    y_lut: tuple[int, ...]
    c_lut: tuple[int, ...]


def _lower_camcorder(effect: ResolvedEffect, ctx: LowerContext) -> CamcorderPayload:
    _validate(effect)
    amount = _scalar(effect, CAMCORDER_AMOUNT, 1.0)
    contrast = _option_float(1.0 + 0.01 * amount)
    brightness = _option_float(-0.005 * amount)
    saturation = _option_float(1.0 - 0.03 * amount)
    return CamcorderPayload(
        amount=amount,
        recording=_scalar(effect, CAMCORDER_RECORDING, 1.0) >= 0.5,
        size=_scalar(effect, CAMCORDER_SIZE, 0.1489361702),
        battery=_scalar(effect, CAMCORDER_BATTERY, 1.0),
        link=_bridge(effect, ctx),
        y_lut=eq_process_lut(contrast, brightness),
        c_lut=eq_process_lut(saturation, 0.0),
    )


def _drawbox(
    rgba: torch.Tensor,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    color: tuple[int, int, int],
    alpha: float,
    thickness: int | None,
    canvas: tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    """Draw one CPU-reference HUD rectangle on a straight RGBA8 frame.

    Coordinates and sizes use the same truncating normalized-frame expressions as
    ``core.cohort_effects._camcorder_filters``. ``thickness=None`` means ``t=fill``.
    RGB is alpha-mixed in float32 and truncated exactly like packed-RGBA drawbox;
    the source alpha channel is unchanged.

    ``canvas = (width, height, origin_x, origin_y)`` places the CLIP CANVAS when
    ``rgba`` is an overscan surface: the normalized coordinates are relative to the
    canvas and the box is clipped to the canvas (drawbox clips to its frame), so the
    HUD never leaks into the overscan.  ``None`` means the frame is the canvas.
    """

    _, surface_h, surface_w = rgba.shape
    frame_w, frame_h, origin_x, origin_y = canvas if canvas is not None else (surface_w, surface_h, 0, 0)
    canvas_left, canvas_top = -origin_x, -origin_y
    left, top = int(x * frame_w) + canvas_left, int(y * frame_h) + canvas_top
    box_w, box_h = int(width * frame_w), int(height * frame_h)
    # No silent failures: a strictly-positive requested extent must never floor to
    # nothing. The normalized guide widths (thickness / max(ctx.width, 1)) round-trip
    # through int(width * frame_w); at odd canvas widths (e.g. 853) that product can
    # truncate to 0, silently dropping a guide line the caller explicitly asked for.
    # So when the caller requested a positive extent, clamp its pixel size up to 1.
    # A zero requested extent still draws nothing (we do NOT invent a pixel there).
    if width > 0:
        box_w = max(1, box_w)
    if height > 0:
        box_h = max(1, box_h)
    right = min(canvas_left + frame_w, left + max(0, box_w))
    bottom = min(canvas_top + frame_h, top + max(0, box_h))
    left, top = max(canvas_left, left), max(canvas_top, top)
    if right <= left or bottom <= top or alpha <= 0.0:
        return rgba
    ys = torch.arange(surface_h, device=rgba.device).view(-1, 1)
    xs = torch.arange(surface_w, device=rgba.device).view(1, -1)
    inside = (xs >= left) & (xs < right) & (ys >= top) & (ys < bottom)
    if thickness is not None:
        edge = max(1, thickness)
        inner = (xs >= left + edge) & (xs < right - edge) & (ys >= top + edge) & (ys < bottom - edge)
        inside &= ~inner
    alpha32 = torch.tensor(alpha, device=rgba.device, dtype=torch.float32)
    source = rgba[:3].to(torch.float32)
    ink = torch.tensor(color, device=rgba.device, dtype=torch.float32).view(3, 1, 1)
    mixed = torch.trunc(source * (1.0 - alpha32) + ink * alpha32).clamp(0.0, 255.0)
    output = rgba.clone()
    output[:3] = torch.where(inside.unsqueeze(0), mixed.to(rgba.dtype), rgba[:3])
    return output


def _apply_camcorder(payload: CamcorderPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    code = _rgba8(canvas)
    yuva = rgba8_to_yuva444p8(code, payload.link).long()
    y_lut = torch.tensor(payload.y_lut, device=code.device, dtype=torch.long)
    c_lut = torch.tensor(payload.c_lut, device=code.device, dtype=torch.long)
    yuva[0] = y_lut[yuva[0]]
    yuva[1] = c_lut[yuva[1]]
    yuva[2] = c_lut[yuva[2]]
    rgba = yuva444p8_to_rgba8(yuva.to(torch.int32), payload.link).to(canvas.dtype)
    if not payload.recording:
        return _canvas(rgba.round().to(torch.int32), canvas.dtype)

    thickness = max(1, round(payload.size * 4.0))
    a, secondary, guide, battery_a = (
        0.82 * payload.amount,
        0.65 * payload.amount,
        0.28 * payload.amount,
        0.75 * payload.amount,
    )
    boxes = (
        (.037, .067, .025, .045, (255, 0, 0), a, None),
        (.066, .064, .008, .060, (255, 255, 255), a, None),
        (.066, .064, .020, .012, (255, 255, 255), a, None),
        (.066, .088, .020, .012, (255, 255, 255), a, None),
        (.081, .064, .008, .032, (255, 255, 255), a, None),
        (.079, .099, .008, .025, (255, 255, 255), a, None),
        (.094, .064, .008, .060, (255, 255, 255), a, None),
        (.094, .064, .023, .012, (255, 255, 255), a, None),
        (.094, .088, .020, .012, (255, 255, 255), a, None),
        (.094, .112, .023, .012, (255, 255, 255), a, None),
        (.123, .064, .024, .012, (255, 255, 255), a, None),
        (.123, .064, .008, .060, (255, 255, 255), a, None),
        (.123, .112, .024, .012, (255, 255, 255), a, None),
        (.815, .065, .120, .052, (255, 255, 255), battery_a, thickness),
        (.940, .079, .012, .024, (255, 255, 255), battery_a, None),
        (.824, .076, .085 * payload.battery, .030, (255, 255, 255), .35 * payload.amount, None),
        (.100, .480, thickness / max(ctx.width, 1), .380, (255, 255, 255), guide, None),
        (.100, .840, .100, thickness / max(ctx.height, 1), (255, 255, 255), guide, None),
        (.900, .480, thickness / max(ctx.width, 1), .380, (255, 255, 255), guide, None),
        (.800, .840, .100, thickness / max(ctx.height, 1), (255, 255, 255), guide, None),
        (.655, .780, .055, .075, (255, 255, 255), secondary, thickness),
        (.670, .758, .025, .022, (255, 255, 255), secondary, None),
    )
    placement = (ctx.width, ctx.height, ctx.origin_x, ctx.origin_y)
    for x, y, width, height, color, alpha, box_thickness in boxes:
        rgba = _drawbox(
            rgba,
            x=x,
            y=y,
            width=width,
            height=height,
            color=color,
            alpha=alpha,
            thickness=box_thickness,
            canvas=placement,
        )
    return _canvas(rgba.round().to(torch.int32), canvas.dtype)


# Focus Blur ----------------------------------------------------------------


@dataclass(frozen=True)
class FocusBlurPayload:
    sigma: float
    softness: float
    emphasis: float
    width: float
    height: float


def _lower_focus_blur(effect: ResolvedEffect, ctx: LowerContext) -> FocusBlurPayload:
    _validate(effect)
    amount = _scalar(effect, FOCUS_AMOUNT, 0.3)
    emphasis = _scalar(effect, FOCUS_EMPHASIS, 0.5)
    return FocusBlurPayload(
        sigma=_option_float(max(0.01, amount * (3.0 + 3.5 * emphasis))),
        softness=_scalar(effect, FOCUS_SOFTNESS, 0.57862),
        emphasis=emphasis,
        width=_scalar(effect, FOCUS_WIDTH, 0.5),
        height=_scalar(effect, FOCUS_HEIGHT, 0.25),
    )


def _apply_focus_blur(payload: FocusBlurPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    code = _rgba8(canvas)
    blurred_rgb = gblur_gbrap(code[:3], sigma=payload.sigma, sigma_v=payload.sigma, steps=2)
    # The focus ellipse is centred on and sized by the CLIP CANVAS; ``pixel_axes`` spans the
    # whole surface in canvas coordinates so the ellipse continues into the overscan.
    xs, ys = ctx.pixel_axes(canvas)
    radius_x = max(ctx.width * payload.width * 0.84, 1.0)
    radius_y = max(ctx.height * payload.height * 0.93, 1.0)
    distance = torch.sqrt(((xs - ctx.width * 0.5) / radius_x) ** 2 + ((ys - ctx.height * 0.5) / radius_y) ** 2)
    outside = ((distance - 1.0) / max(payload.softness * 1.4, 0.001)).clamp(0.0, 1.0)
    # ``geq`` truncates its alpha expression to uint8. The following overlay uses
    # libavfilter's integer straight-alpha path, shared with the calibrated Callout
    # graph, rather than a generic floating-point source-over approximation.
    weighted_alpha = torch.trunc(code[3:4].to(canvas.dtype) * outside.unsqueeze(0)).to(torch.int32)
    weighted = torch.cat((blurred_rgb.to(torch.int32), weighted_alpha), dim=0)
    return _canvas(overlay_yuva444p(code, weighted, 0, 0), canvas.dtype)


# Drop Shadow ---------------------------------------------------------------


@dataclass(frozen=True)
class DropShadowPayload:
    opacity: float = 0.75
    sigma: float = 3.0
    offset_x: int = 4
    offset_y: int = 4


def _lower_drop_shadow(effect: ResolvedEffect, ctx: LowerContext) -> DropShadowPayload:
    _validate(effect)
    return DropShadowPayload()


def _shift(frame: torch.Tensor, x: int, y: int) -> torch.Tensor:
    output = torch.zeros_like(frame)
    _, height, width = frame.shape
    src_x0, src_x1 = max(0, -x), min(width, width - x)
    src_y0, src_y1 = max(0, -y), min(height, height - y)
    if src_x1 > src_x0 and src_y1 > src_y0:
        output[:, src_y0 + y:src_y1 + y, src_x0 + x:src_x1 + x] = frame[:, src_y0:src_y1, src_x0:src_x1]
    return output


def _apply_drop_shadow(payload: DropShadowPayload, canvas: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    code = _rgba8(canvas).to(canvas.dtype)
    shadow = torch.zeros_like(code)
    shadow[3] = code[3] * payload.opacity
    shadow = gblur_gbrap(shadow, sigma=payload.sigma, sigma_v=payload.sigma, steps=2)
    shadow = _shift(shadow, payload.offset_x, payload.offset_y)
    return _canvas(_over(shadow, code).to(torch.int32), canvas.dtype)


# All four blur (gblur / unsharp: frame-boundary dependent) or shift content, so the canvas
# region comes from the crop run and the surface run supplies the overscan (``"splice"``).
# Camcorder's HUD and Focus Blur's ellipse are drawn in clip-canvas coordinates on that
# surface run, so nothing canvas-relative leaks into the overscan.
register(EffectPort(handler=CARTOON_HANDLER, lower=_lower_cartoon, apply=_apply_cartoon, overscan=OVERSCAN_SPLICE))
register(EffectPort(handler=CAMCORDER_HANDLER, lower=_lower_camcorder, apply=_apply_camcorder, overscan=OVERSCAN_SPLICE))
register(EffectPort(handler=DROP_SHADOW_HANDLER, lower=_lower_drop_shadow, apply=_apply_drop_shadow, overscan=OVERSCAN_SPLICE))
register(EffectPort(handler=FOCUS_BLUR_HANDLER, lower=_lower_focus_blur, apply=_apply_focus_blur, overscan=OVERSCAN_SPLICE))
