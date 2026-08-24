"""Stage 0 gate: perceptual audio-parity harness for the PyAV port.

Architecture map
================

This is the *perceptual* half of the PyAV audio-delivery unification (plan:
``pyav-audio-delivery-unification.md``).  Once chunk 2 lands, it will compare the
in-process PyAV audio render against the stock-FFmpeg reference
(``run_audio_execution_plan``) within a perceptual dBFS ceiling.

CHUNK 1 SCOPE: the PyAV renderer does not exist yet, so this file validates only
the *comparator* -- the thing every later assertion depends on:

1. ``test_comparator_is_stable_on_identical_renders`` -- two independent CLI
   renders of the SAME plan must agree to ~zero error (the comparator does not
   invent differences).
2. ``test_comparator_catches_a_structural_miswire`` -- a plan with a dropped
   mute must DIFFER from the correct plan by far more than the ceiling (the gate
   actually bites; it is not a rubber stamp).

The real PyAV-vs-reference assertion (``test_pyav_candidate_matches_reference``)
is present but skipped until chunk 2 wires the renderer.

The ceiling
-----------
The standard is "sounds the same to a human", not bit-exact.  The ceiling sits
in the band BETWEEN encoder jitter and an audible mistake: resampler/dither
noise lives near or below -90 dBFS, while a structural error (dropped mute, wrong
gain, mis-panned channel, off-by-a-frame trim) is tens of dB louder.  ``-72
dBFS`` peak is comfortably above the jitter floor and far below any audible
change.  The measured error is PRINTED every run so drift is visible.

Main callers: pytest (experimental renderer job).
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from bladeworks.core.audio_execution import (
    AudioAssetBinding,
    build_audio_execution_plan,
    probe_audio_asset,
    run_audio_execution_plan,
)
from bladeworks.core.audio_ir import (
    AudioControlLayer,
    AudioMuteRange,
)
from tests._audio_delivery_corpus import (
    CAPABILITY_REJECTED_CASES,
    generate_tone,
    item,
    plan,
    rebind_on_real_media,
    structural_cases,
)


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
_SKIP = pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="stock FFmpeg is unavailable")

# Perceptual pass ceiling; see the module docstring for the derivation.
PEAK_ERROR_CEILING_DBFS = -72.0


def _decode_f32le(path: Path) -> np.ndarray:
    """Decode any rendered audio file to a flat interleaved float32 array.

    Full-scale is +/-1.0, so a peak absolute difference maps directly to dBFS.
    Decoding through FFmpeg keeps precision far below the ceiling (unlike a
    16-bit wav read) so the comparator's own floor never masks jitter.
    """

    assert FFMPEG is not None
    completed = subprocess.run(
        (FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-"),
        check=True,
        capture_output=True,
    )
    return np.frombuffer(completed.stdout, dtype="<f4")


def _peak_error_dbfs(a: np.ndarray, b: np.ndarray) -> float:
    """Peak absolute sample difference in dBFS.

    A length mismatch is itself a structural failure (different sample count or
    layout), reported as ``+inf`` so it always breaches any finite ceiling --
    never silently truncated to a comparable prefix.
    """

    if a.shape != b.shape:
        return math.inf
    if a.size == 0:
        return -math.inf
    peak = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    if peak == 0.0:
        return -math.inf
    return 20.0 * math.log10(peak)


def _render_pcm(plan_obj, bindings, out: Path) -> Path:
    run_audio_execution_plan(
        build_audio_execution_plan(plan_obj, bindings),
        out,
        ffmpeg_path=FFMPEG,
    )
    return out


def _tone_binding(tmp_path: Path, *, name: str = "asset-a") -> dict[str, AudioAssetBinding]:
    media = tmp_path / f"{name}.wav"
    generate_tone(media, ffmpeg=FFMPEG)
    return {name: probe_audio_asset(name, media, ffprobe_path=FFPROBE)}


@_SKIP
def test_comparator_is_stable_on_identical_renders(tmp_path: Path) -> None:
    """Same plan rendered twice -> the comparator sees ~zero error."""

    bindings = _tone_binding(tmp_path)
    only = plan(item())
    a = _decode_f32le(_render_pcm(only, bindings, tmp_path / "a.wav"))
    b = _decode_f32le(_render_pcm(only, bindings, tmp_path / "b.wav"))
    error = _peak_error_dbfs(a, b)
    print(f"[parity] identical-render peak error = {error:.1f} dBFS (ceiling {PEAK_ERROR_CEILING_DBFS})")
    assert error <= PEAK_ERROR_CEILING_DBFS


@_SKIP
def test_comparator_catches_a_structural_miswire(tmp_path: Path) -> None:
    """A dropped mute must breach the ceiling -- proof the gate bites."""

    bindings = _tone_binding(tmp_path)
    muted = plan(
        item(
            control_layers=(
                AudioControlLayer(
                    path="spine/audio[1]",
                    mutes=(AudioMuteRange(Fraction(1, 4), Fraction(1, 4)),),
                ),
            )
        )
    )
    unmuted = plan(item())  # the mis-wire: the [0.25s, 0.5s] mute is gone
    correct = _decode_f32le(_render_pcm(muted, bindings, tmp_path / "muted.wav"))
    miswired = _decode_f32le(_render_pcm(unmuted, bindings, tmp_path / "unmuted.wav"))
    error = _peak_error_dbfs(correct, miswired)
    print(f"[parity] dropped-mute peak error = {error:.1f} dBFS (ceiling {PEAK_ERROR_CEILING_DBFS})")
    assert error > PEAK_ERROR_CEILING_DBFS, (
        "a dropped mute did not breach the ceiling -- the gate would rubber-stamp "
        "structural errors"
    )


# One corpus shape is a CAPABILITY REJECT, not a parity case: ``surround_layout``
# asks for 5.1 output, which the renderer does not support (its 5.1 ``pan`` upmix
# names side channels ``SL``/``SR`` while FFmpeg 8's ``5.1`` layout uses
# ``BL``/``BR`` -- "Channel SL does not exist in the chosen layout").  Rather than
# letting that surface as a cryptic libav ``Errno 22`` (exit 234) deep in the
# audio graph, ``build_audio_execution_plan`` now rejects surround output UP FRONT
# with a clear ``RenderCapabilityError`` (see ``reject_unsupported_output_layout``).
# So surround is excluded from the perceptual gate and instead asserted to reject
# cleanly by ``test_surround_layout_rejects_before_graph_build`` below.
#
# ``freeze_silence`` used to be excluded here too (its freeze path built a media
# ``pan`` route it then dropped for calibrated silence, leaving that pad
# unconnected).  That plumbing is now fixed in ``_build_item_graph``: a fully
# frozen item skips the media route and emits pure ``anullsrc`` silence, which
# both FFmpeg and the PyAV graph render, so it is a full parity case again.
_RENDERABLE_CASES = [
    case for case in structural_cases() if case.name not in CAPABILITY_REJECTED_CASES
]


def test_surround_layout_rejects_before_graph_build() -> None:
    """The ``surround`` output shape must reject cleanly, not crash in libav.

    ``surround_layout`` requests 5.1 output. The renderer does not implement the
    5.1 channel mapping, so ``build_audio_execution_plan`` must raise a
    ``RenderCapabilityError`` naming the construct BEFORE it builds any graph node
    -- never let it fall through to the cryptic libav ``Errno 22`` (ffmpeg exit
    234) the 5.1 ``pan`` upmix would otherwise trigger. The reject fires ahead of
    binding resolution, so no real media (empty bindings) is required here.
    """

    from bladeworks.core.audio_execution import (
        build_audio_execution_plan,
    )
    from bladeworks.core.errors import RenderCapabilityError

    surround = next(
        case for case in structural_cases() if case.name == "surround_layout"
    )
    with pytest.raises(RenderCapabilityError, match="surround"):
        build_audio_execution_plan(surround.plan, {})


@_SKIP
@pytest.mark.parametrize("case", _RENDERABLE_CASES, ids=lambda case: case.name)
def test_pyav_candidate_matches_reference(case, tmp_path: Path) -> None:
    """The real Stage 0 gate: PyAV render vs stock-FFmpeg reference.

    For each renderable corpus case: render the plan via the PyAV
    ``av.filter.Graph`` path (``audio_pyav``) AND via ``run_audio_execution_plan``
    (the stock-FFmpeg reference), decode both with ``_decode_f32le``, and assert
    the peak error is within the perceptual ceiling.  The interior graph is proven
    byte-identical by the Stage 2a gate, so any breach here is a boundary
    (source/sink negotiation) defect, not a calibrated-semantics one.
    """

    from bladeworks.core.audio_execution import build_audio_execution_plan
    from bladeworks.tensor import audio_pyav

    bindings = rebind_on_real_media(case, tmp_path, ffmpeg=FFMPEG, ffprobe=FFPROBE)
    execution = build_audio_execution_plan(case.plan, bindings)

    reference = _decode_f32le(
        _render_pcm_from_execution(execution, tmp_path / "reference.wav")
    )
    candidate = _decode_f32le(
        audio_pyav.render_execution_to_wav(execution, tmp_path / "candidate.wav")
    )
    error = _peak_error_dbfs(reference, candidate)
    print(
        f"[parity] {case.name:22s} peak error = {error:6.1f} dBFS "
        f"(ceiling {PEAK_ERROR_CEILING_DBFS})"
    )
    assert error <= PEAK_ERROR_CEILING_DBFS, (
        f"{case.name}: PyAV audio diverged from the stock-FFmpeg reference by "
        f"{error:.1f} dBFS (ceiling {PEAK_ERROR_CEILING_DBFS})"
    )


def _render_pcm_from_execution(execution, out: Path) -> Path:
    run_audio_execution_plan(execution, out, ffmpeg_path=FFMPEG)
    return out
