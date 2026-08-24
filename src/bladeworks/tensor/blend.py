"""Tensor implementation of the reviewed FCPXML blend contract.

Architecture map
================

``composite_layers(background, foreground, mode)``
    -> resolves the shared core mode name without a silent fallback
    -> keeps alpha in premultiplied linear working space
    -> evaluates reviewed RGB modes in straight encoded colour, matching the
       stock-FFmpeg barrier used by ``core/compositor.py``
    -> applies alpha-aware source-over, Behind, or stencil/silhouette matte
       semantics and returns premultiplied linear RGBA

Important invariants
--------------------

* Inputs and output are channels-first ``[4, H, W]`` premultiplied linear
  tensors. RGB values are never divided by alpha without a zero-alpha guard.
* Opacity is already part of the foreground tensor when this function runs.
  It therefore affects both ordinary blend coverage and matte coverage once,
  at the same stage as the shared FFmpeg compositor.
* The reviewed RGB modes are encoded-space operations. The tensor boundary is
  explicit: unpremultiply -> calibrated encode -> mode formula -> linearize ->
  premultiply. This is why a tensor render does not accidentally change the
  CPU/reference meaning merely because its working canvas is linear.
* Hue, Saturation, Color, and Luminosity never enter this module: the shared
  resolver rejects them as uncalibrated cross-channel modes.

Why this exists
---------------

``tensor/composite.py`` owns only Normal source-over and surface conversion.
Putting the mode formulas here keeps the renderer's stack fold small and gives
the mathematical rules one focused test surface. Layer placement and rendered
group placement both call this same function, so group blend modes cannot drift
from ordinary layer blend modes.
"""

from __future__ import annotations

import torch

from ..core.compositor import BlendModeSpec, resolve_blend_mode
from .color import encode, linearize
from .composite import over


# ``format=gray`` in the shared FFmpeg graph receives an untagged RGBA link.
# FFmpeg's default RGB-to-gray conversion is the BT.601 luma table, which is
# also the table recorded by the tensor group-effect contract.
_BT601_LUMA = (0.299, 0.587, 0.114)
_EPSILON = 1.0e-8


def _validate_canvas_pair(background: torch.Tensor, foreground: torch.Tensor) -> None:
    if background.ndim != 3 or foreground.ndim != 3:
        raise ValueError("blend canvases must have shape [4, H, W]")
    if background.shape != foreground.shape or background.shape[0] != 4:
        raise ValueError(
            "blend canvases must have matching shape [4, H, W], got "
            f"{tuple(background.shape)} and {tuple(foreground.shape)}"
        )


def _unpremultiply(canvas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = canvas[3:4].clamp(0.0, 1.0)
    safe_alpha = alpha.clamp_min(_EPSILON)
    rgb = torch.where(alpha > 0.0, canvas[:3] / safe_alpha, torch.zeros_like(canvas[:3]))
    return rgb.clamp(0.0, 1.0), alpha


def _rgb_mode(top: torch.Tensor, bottom: torch.Tensor, mode: str) -> torch.Tensor:
    """Evaluate one stock-FFmpeg separable RGB formula on encoded colours.

    ``top`` is the foreground and ``bottom`` is the lower canvas, matching
    ``[fg][lower]blend=all_mode=...`` in ``core/compositor.py``. FFmpeg's
    formulas use different samples for their branches: Overlay uses ``top``,
    while Hard Light and Pin Light use ``bottom``. The argument order is
    deliberately visible here because the formulas are not all symmetric.

    Main callers: ``composite_layers`` for RGB blend modes.
    """

    if mode == "addition":
        result = top + bottom
    elif mode == "subtract":
        result = top - bottom
    elif mode == "darken":
        result = torch.minimum(top, bottom)
    elif mode == "lighten":
        result = torch.maximum(top, bottom)
    elif mode == "multiply":
        result = top * bottom
    elif mode == "screen":
        result = 1.0 - (1.0 - top) * (1.0 - bottom)
    elif mode == "overlay":
        result = torch.where(
            top < 0.5,
            2.0 * top * bottom,
            1.0 - 2.0 * (1.0 - top) * (1.0 - bottom),
        )
    elif mode == "softlight":
        # This is FFmpeg's reviewed softlight implementation, not the more
        # common Photoshop/W3C square-root variant.
        result = top * top + 2.0 * bottom * top * (1.0 - top)
    elif mode == "hardlight":
        result = torch.where(
            bottom < 0.5,
            2.0 * top * bottom,
            1.0 - 2.0 * (1.0 - top) * (1.0 - bottom),
        )
    elif mode == "difference":
        result = torch.abs(bottom - top)
    elif mode == "exclusion":
        result = top + bottom - 2.0 * top * bottom
    elif mode == "burn":
        result = 1.0 - (1.0 - bottom) / top.clamp_min(_EPSILON)
        result = torch.where(top <= 0.0, torch.zeros_like(result), result)
    elif mode == "dodge":
        result = bottom / (1.0 - top).clamp_min(_EPSILON)
        result = torch.where(top >= 1.0, torch.ones_like(result), result)
    elif mode == "divide":
        result = top / bottom.clamp_min(_EPSILON)
        result = torch.where(bottom <= 0.0, torch.ones_like(result), result)
    elif mode == "linearlight":
        result = bottom + 2.0 * top - 1.0
    elif mode == "pinlight":
        result = torch.where(
            bottom < 0.5,
            torch.minimum(top, 2.0 * bottom),
            torch.maximum(top, 2.0 * bottom - 1.0),
        )
    elif mode == "hardmix":
        result = torch.where(top + bottom < 1.0, torch.zeros_like(top), torch.ones_like(top))
    else:
        raise ValueError(f"unhandled reviewed RGB blend mode {mode!r}")
    return result.clamp(0.0, 1.0)


def _rgb_composite(
    background: torch.Tensor,
    foreground: torch.Tensor,
    spec: BlendModeSpec,
) -> torch.Tensor:
    lower_rgb, lower_alpha = _unpremultiply(background)
    upper_rgb, upper_alpha = _unpremultiply(foreground)

    # The core graph evaluates non-Normal RGB formulas in encoded RGBA. The
    # lower-alpha interpolation is done on the encoded result before returning
    # to the tensor renderer's linear working domain. With a transparent lower
    # pixel this selects the untouched foreground colour exactly.
    lower_code = encode(lower_rgb)
    upper_code = encode(upper_rgb)
    assert spec.ffmpeg_mode is not None
    blended_code = _rgb_mode(upper_code, lower_code, spec.ffmpeg_mode)
    selected_code = upper_code * (1.0 - lower_alpha) + blended_code * lower_alpha
    selected_linear = linearize(selected_code)

    output_alpha = (upper_alpha + lower_alpha * (1.0 - upper_alpha)).clamp(0.0, 1.0)
    output_rgb = selected_linear * upper_alpha + background[:3] * (1.0 - upper_alpha)
    return torch.cat((output_rgb.clamp(0.0, 1.0), output_alpha), dim=0)


def _matte_composite(
    background: torch.Tensor,
    foreground: torch.Tensor,
    spec: BlendModeSpec,
) -> torch.Tensor:
    lower_rgb, lower_alpha = _unpremultiply(background)
    upper_rgb, upper_alpha = _unpremultiply(foreground)
    assert spec.matte_kind is not None

    if spec.matte_kind.endswith("luma"):
        upper_code = encode(upper_rgb)
        matte = sum(weight * channel for weight, channel in zip(_BT601_LUMA, upper_code))
        matte = matte.unsqueeze(0) if matte.ndim == 2 else matte
        matte = matte * upper_alpha
    else:
        matte = upper_alpha
    if spec.matte_kind.startswith("silhouette"):
        matte = 1.0 - matte

    output_alpha = (lower_alpha * matte).clamp(0.0, 1.0)
    return torch.cat((lower_rgb * output_alpha, output_alpha), dim=0)


def composite_layers(
    background: torch.Tensor,
    foreground: torch.Tensor,
    mode: str | None = None,
) -> torch.Tensor:
    """Composite one placed foreground onto a premultiplied linear canvas.

    Main callers:
    - ``renderer._FrameComposer.render_scope`` for ordinary layers and groups.
    - ``renderer._FrameComposer.side`` for transition participants.

    ``mode`` is resolved through the shared core contract. Opacity must already
    be applied to ``foreground`` by the placement stage, matching the CPU
    compositor's order. A known but uncalibrated mode therefore raises instead
    of silently becoming Normal.
    """

    _validate_canvas_pair(background, foreground)
    spec = resolve_blend_mode(mode)
    if spec.family == "normal":
        return over(background, foreground)
    if spec.family == "behind":
        return over(foreground, background)
    if spec.family == "rgb":
        return _rgb_composite(background, foreground, spec)
    if spec.family == "matte":
        return _matte_composite(background, foreground, spec)
    raise ValueError(f"unhandled blend family {spec.family!r}")


__all__ = ["composite_layers"]
