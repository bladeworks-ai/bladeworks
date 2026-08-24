"""Default-only mappings for simple built-in Final Cut video effects.

Architecture map
================

``filter-video`` selected by the capability registry
    -> reject every undocumented parameter
    -> emit a short, ordered stock-FFmpeg filter sequence
    -> preserve source alpha

These eight handlers cover only the no-parameter defaults that Final Cut 12.3
accepted, re-exported, and rendered in the ``portable-basic-effects-defaults``
fixture.  They do not infer controls from a template display name.  If a future
FCPXML contains parameters for one of these effects, compilation omits the
whole effect and reports why instead of applying a plausible-looking default.

The mappings use the same visible operation as the Final Cut reference; merely
sharing an effect name is not sufficient. They are still approximations, not
Motion-template emulation, and the current evidence is one synthetic default
fixture rather than real-project calibration.
The renderer-owned ``spell-effect-v1`` Vulkan fixtures remain available as
experimental escape hatches. The genuine no-parameter Final Cut Pixellate UID
uses a stock handler here. Radial remains omitted because the bounded CPU
prototype failed the documented performance gate.
"""

from __future__ import annotations

from typing import Optional

from .model import Parameter, ResolvedEffect


BASIC_EFFECT_HANDLERS = frozenset(
    {
        "negative",
        "threshold",
        "colorize",
        "tint",
        "flipped",
        "mirror",
        "add_noise",
        "pixellate_default",
    }
)


def unsupported_basic_effect_reason(
    handler: str,
    params: tuple[Parameter, ...],
) -> Optional[str]:
    """Return why a default-only mapping must not execute, if anything.

    Main callers:
    - ``compiler._resolve_filter_instance`` before creating renderer IR.

    Why this exists:
    Final Cut's template controls have not appeared in the available real
    corpus.  Applying the default while ignoring an unknown control would turn
    a visible compatibility problem into a silent wrong render.
    """

    if handler in BASIC_EFFECT_HANDLERS and params:
        names = sorted(
            {
                parameter.name or parameter.key or "unnamed parameter"
                for parameter in params
            }
        )
        return (
            f"portable {handler.replace('_', ' ')} supports only Final Cut's "
            f"round-tripped no-parameter default; unsupported controls: {', '.join(names)}"
        )
    return None


def basic_effect_filters(effect: ResolvedEffect) -> list[str]:
    """Translate one validated default effect to stock FFmpeg filters.

    Main callers:
    - ``ffmpeg._effect_filters`` while preserving FCPXML filter order.
    """

    handler = effect.handler
    if handler == "negative":
        # The installed Motion template is Gamma(0.45) -> Invert ->
        # Gamma(2.2). Motion's gamma convention is the reciprocal of the
        # exponent name, producing this bounded channel curve.
        curve = "pow(clip(1-pow(clip(val/maxval,0,1),2.2),0,1),0.45)*maxval"
        return [f"lutrgb=r='{curve}':g='{curve}':b='{curve}'"]
    if handler == "flipped":
        return ["hflip"]
    if handler == "mirror":
        return [_mirror_filter()]
    if handler == "threshold":
        return [_threshold_filter()]
    if handler == "colorize":
        return [
            "colorchannelmixer="
            "rr=0.69:rg=0.27:rb=0.20:"
            "gr=0.087:gg=0.47:gb=0.08:"
            "br=0:bg=0:bb=0.51"
        ]
    if handler == "tint":
        # Exact default color published by Final Cut's installed Tint.moef.
        # FFmpeg colorize retains source lightness while applying its hue and
        # saturation, matching the template's PAETint mechanism much more
        # closely than the former nearly-gray three-filter chain.
        return [
            "colorize=hue=245.54457:saturation=0.324662:lightness=0.270485:mix=1",
            # Motion's PAETint also compresses source luminance around its dark
            # default swatch. A Final Cut 12.3 real-video sweep resolves that
            # processing-space difference to this bounded gamma-RGB offset.
            "eq=brightness=-0.15",
        ]
    if handler == "add_noise":
        # Explicit per-plane seeds make repeated renders byte-stable. Alpha is
        # left untouched; FFmpeg may internally reorder the three color planes,
        # but they intentionally use the same strength.
        return [
            "noise="
            "c0s=6:c1s=6:c2s=6:c3s=0:"
            "c0_seed=424242:c1_seed=424243:c2_seed=424244"
        ]
    if handler == "pixellate_default":
        # Process RGB only. Alpha stays byte-for-byte aligned with the input,
        # which matters when this effect runs inside a connected layer.
        return ["pixelize=width=4:height=4:mode=avg:planes=0x7"]
    return []


def _threshold_filter() -> str:
    """Approximate Final Cut's default two-band tonal threshold.

    Final Cut's installed template publishes threshold 0.22, smoothness 0.15,
    and mix 0.5838. Its processing space is not FFmpeg's gamma-coded RGB, so a
    genuine Final Cut 12.3 A/B sweep resolves the equivalent portable boundary
    to 0.40 with width 0.25 and mix 0.625. The original image supplies the
    remaining 0.375, preserving color in both tonal bands.
    """

    luma = "0.2126*r(X,Y)+0.7152*g(X,Y)+0.0722*b(X,Y)"
    progress = f"clip((({luma})-70.125)/63.75,0,1)"
    smooth = f"pow({progress},2)*(3-2*{progress})"

    def channel(name: str) -> str:
        return f"{name}(X,Y)*0.375+255*0.625*({smooth})"

    return (
        "geq="
        f"r='{channel('r')}':"
        f"g='{channel('g')}':"
        f"b='{channel('b')}':"
        "a='alpha(X,Y)'"
    )


def _mirror_filter() -> str:
    """Mirror the source's right half across the vertical center seam."""

    source_x = "min(W-1,W/2+abs(X-W/2))"
    return (
        "geq="
        f"r='r({source_x},Y)':"
        f"g='g({source_x},Y)':"
        f"b='b({source_x},Y)':"
        f"a='alpha({source_x},Y)'"
    )
