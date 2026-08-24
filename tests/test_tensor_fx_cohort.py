"""Focused synthetic goldens for the Phase 4 tensor effect cohort.

Architecture map
================

deterministic RGBA8 plate
    -> CPU cohort builder -> local FFmpeg reference frame
    -> tensor effect registry -> tensor frame
    -> per-effect SSIM gate and deterministic output assertion

The lowering tests separately exercise default controls, reviewed non-default
controls, and strict rejection of unknown, animated, opaque, or default-only
parameters. This keeps visual parity and the accepted API surface independently
reviewable.
"""

from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bladeworks.core.capabilities import CapabilityRegistry  # noqa: E402
from bladeworks.core.cohort_effects import (  # noqa: E402
    cohort_effect_filters,
    cohort_effect_graph_lines,
)
from bladeworks.core.model import Parameter, ResolvedEffect  # noqa: E402
from bladeworks.tensor import TensorRenderUnsupported  # noqa: E402
from bladeworks.tensor.color import (  # noqa: E402
    code_to_premultiplied,
    premultiplied_to_code,
)
from bladeworks.tensor.effects import (  # noqa: E402
    EFFECT_PORTS,
    ApplyContext,
    LowerContext,
)
from bladeworks.tensor.fx_cohort import (  # noqa: E402
    CAMCORDER_AMOUNT,
    CAMCORDER_BATTERY,
    CAMCORDER_RECORDING,
    CAMCORDER_SIZE,
    CARTOON_AMOUNT,
    FOCUS_AMOUNT,
    FOCUS_EMPHASIS,
    FOCUS_HEIGHT,
    FOCUS_SOFTNESS,
    FOCUS_WIDTH,
)


FFMPEG = shutil.which("ffmpeg")
WIDTH, HEIGHT = 192, 112
HANDLERS = (
    "cohort_cartoon",
    "cohort_camcorder",
    "cohort_drop_shadow",
    "cohort_focus_blur",
)
CAPABILITY_IDS = {
    "cohort_cartoon": "effect-cartoon-cohort",
    "cohort_camcorder": "effect-camcorder-cohort",
    "cohort_drop_shadow": "effect-drop-shadow-cohort",
    "cohort_focus_blur": "effect-focus-blur-cohort",
}


def _plate() -> np.ndarray:
    """Return an edge-rich RGBA plate with visible transparent shadow margins."""

    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    red = (xx * 7 + yy * 3) % 256
    green = (yy * 11 + xx * 2) % 256
    blue = (255 - xx * 5 + yy * 7) % 256
    alpha = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    alpha[12:-16, 18:-24] = 255
    alpha[32:78, 55:142] = 144
    checker = ((xx // 8 + yy // 8) % 2) == 0
    red = np.where(checker, red, 255 - red)
    blue = np.where(checker, 255 - blue, blue)
    return np.stack((red, green, blue, alpha), axis=-1).astype(np.uint8)


def _capability(handler: str):
    capability_id = CAPABILITY_IDS[handler]
    return next(item for item in CapabilityRegistry.load().entries if item.id == capability_id)


def _effect(handler: str, params: tuple[Parameter, ...] = (), *, data=None) -> ResolvedEffect:
    capability = _capability(handler)
    return ResolvedEffect(
        kind="video_filter",
        uid=capability.uid,
        name=capability.aliases[0],
        handler=handler,
        portable_status=capability.portable_status,
        params=params,
        calibration=capability.parameters,
        data={} if data is None else data,
        path=f"fixture/{handler}",
    )


def _context() -> LowerContext:
    return LowerContext(
        clip_path="fixture",
        width=WIDTH,
        height=HEIGHT,
        frame_duration=Fraction(1, 30),
        clip_duration=Fraction(1),
        source_colorspace="bt709",
        source_color_range="tv",
        reference_effect_link="rgba:bt709:tv",
    )


def _tensor_frame(effect: ResolvedEffect) -> np.ndarray:
    port = EFFECT_PORTS[effect.handler]
    payload = port.lower(effect, _context())
    code = torch.from_numpy(_plate().transpose(2, 0, 1).copy()).to(torch.float64)
    canvas = code_to_premultiplied(code)
    context = ApplyContext(frame=0, seconds=0.0, width=WIDTH, height=HEIGHT)
    first = port.apply(payload, canvas, context)
    second = port.apply(payload, canvas, context)
    assert torch.equal(first, second), f"{effect.handler} output is not deterministic"
    return (
        premultiplied_to_code(first)
        .round()
        .clamp(0.0, 255.0)
        .to(torch.uint8)
        .numpy()
        .transpose(1, 2, 0)
    )


def _reference_frame(effect: ResolvedEffect, tmp_path: Path) -> np.ndarray:
    if FFMPEG is None:
        pytest.skip("FFmpeg is required for CPU cohort goldens")
    source = tmp_path / "plate.rgba"
    output = tmp_path / f"{effect.handler}.rgba"
    _plate().tofile(source)
    filters = cohort_effect_filters(effect)
    graph_lines = cohort_effect_graph_lines("pre", "out", effect, prefix="golden")
    if graph_lines:
        graph = ";".join(
            ["[0:v]setparams=colorspace=bt709:range=tv,format=rgba[pre]", *graph_lines]
        )
    else:
        chain = ",".join(["setparams=colorspace=bt709:range=tv", "format=rgba", *filters, "format=rgba"])
        graph = f"[0:v]{chain}[out]"
    result = subprocess.run(
        (
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-i",
            str(source),
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgba",
            "-f",
            "rawvideo",
            str(output),
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return np.fromfile(output, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)


def _ssim(actual: np.ndarray, expected: np.ndarray) -> float:
    """Global RGB SSIM after both straight-alpha frames are shown over black.

    RGB values underneath alpha zero are intentionally excluded: they are not visible,
    and the tensor working space correctly discards them during premultiplication while
    packed-RGBA FFmpeg filters are free to retain arbitrary hidden channel values.
    """

    actual_rgb = actual[..., :3].astype(np.float64) * (actual[..., 3:4] / 255.0)
    expected_rgb = expected[..., :3].astype(np.float64) * (expected[..., 3:4] / 255.0)
    scores = []
    for channel in range(3):
        left = actual_rgb[..., channel]
        right = expected_rgb[..., channel]
        mean_left, mean_right = left.mean(), right.mean()
        variance_left, variance_right = left.var(), right.var()
        covariance = ((left - mean_left) * (right - mean_right)).mean()
        c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
        scores.append(
            ((2 * mean_left * mean_right + c1) * (2 * covariance + c2))
            / ((mean_left**2 + mean_right**2 + c1) * (variance_left + variance_right + c2))
        )
    return float(np.mean(scores))


NON_DEFAULTS = {
    "cohort_cartoon": (Parameter("Amount", CARTOON_AMOUNT, "0.5"),),
    "cohort_camcorder": (
        Parameter("Amount", CAMCORDER_AMOUNT, "0.55"),
        Parameter("Size", CAMCORDER_SIZE, "0.3"),
        Parameter("Battery Level", CAMCORDER_BATTERY, "0.35"),
        Parameter("Recording", CAMCORDER_RECORDING, "1"),
    ),
    # The reviewed Drop Shadow contract is intentionally default-only.
    "cohort_drop_shadow": (),
    "cohort_focus_blur": (
        Parameter("Amount", FOCUS_AMOUNT, "0.8"),
        Parameter("Softness", FOCUS_SOFTNESS, "0.2"),
        Parameter("Emphasis", FOCUS_EMPHASIS, "0.75"),
        Parameter("Width", FOCUS_WIDTH, "0.35"),
        Parameter("Height", FOCUS_HEIGHT, "0.4"),
    ),
}


@pytest.mark.parametrize("handler", HANDLERS)
@pytest.mark.parametrize("non_default", (False, True), ids=("default", "non-default"))
def test_phase4_effect_matches_cpu_cohort_golden(
    handler: str,
    non_default: bool,
    tmp_path: Path,
) -> None:
    if handler == "cohort_drop_shadow" and non_default:
        pytest.skip("Drop Shadow is a reviewed default-only effect")
    effect = _effect(handler, NON_DEFAULTS[handler] if non_default else ())
    actual = _tensor_frame(effect)
    expected = _reference_frame(effect, tmp_path)
    score = _ssim(actual, expected)
    print(f"{handler} {'non-default' if non_default else 'default'} SSIM={score:.6f}")
    assert score >= 0.98


def test_reviewed_non_default_parameters_change_pixels() -> None:
    for handler in ("cohort_cartoon", "cohort_camcorder", "cohort_focus_blur"):
        default = _tensor_frame(_effect(handler))
        changed = _tensor_frame(_effect(handler, NON_DEFAULTS[handler]))
        assert not np.array_equal(default, changed), f"{handler} ignored its reviewed controls"


def test_camcorder_recording_zero_removes_the_hud() -> None:
    off = _effect(
        "cohort_camcorder",
        (Parameter("Recording", CAMCORDER_RECORDING, "0"),),
    )
    default = _tensor_frame(_effect("cohort_camcorder"))
    no_hud = _tensor_frame(off)
    # The colour treatment remains, but the red REC marker and white guides disappear.
    assert not np.array_equal(default, no_hud)
    assert not np.array_equal(no_hud[54, 19, :3], default[54, 19, :3])


@pytest.mark.parametrize("handler", HANDLERS)
def test_unknown_parameter_rejects_loudly(handler: str) -> None:
    effect = _effect(handler, (Parameter("Shader", "untrusted/path", "1"),))
    with pytest.raises(TensorRenderUnsupported, match="unsupported parameters"):
        EFFECT_PORTS[handler].lower(effect, _context())


@pytest.mark.parametrize("handler", HANDLERS)
def test_opaque_data_rejects_loudly(handler: str) -> None:
    with pytest.raises(TensorRenderUnsupported, match="opaque filter data"):
        EFFECT_PORTS[handler].lower(_effect(handler, data={"archive": "bytes"}), _context())


def test_animated_and_malformed_reviewed_parameters_reject_loudly() -> None:
    animated = Parameter("Amount", CARTOON_AMOUNT, "0.5", keyframes=("frame",))
    malformed = Parameter("Amount", FOCUS_AMOUNT, "nan")
    with pytest.raises(TensorRenderUnsupported, match="animated control"):
        EFFECT_PORTS["cohort_cartoon"].lower(_effect("cohort_cartoon", (animated,)), _context())
    with pytest.raises(TensorRenderUnsupported, match="finite"):
        EFFECT_PORTS["cohort_focus_blur"].lower(_effect("cohort_focus_blur", (malformed,)), _context())


def test_drop_shadow_keeps_explicit_controls_rejected() -> None:
    opacity = Parameter("Opacity", "2", "0.5")
    with pytest.raises(TensorRenderUnsupported, match="outside the bounded handler contract"):
        EFFECT_PORTS["cohort_drop_shadow"].lower(
            _effect("cohort_drop_shadow", (opacity,)),
            _context(),
        )
