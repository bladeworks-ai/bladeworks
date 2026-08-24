"""Shared audio-delivery test corpus for the PyAV unification gates.

Architecture map
================

This module is the single source of graph-shape fixtures used by BOTH audio
delivery gates that guard the PyAV port (plan:
``pyav-audio-delivery-unification.md``):

- ``test_audio_graph_serialize_golden`` (Stage 2a) — proves the refactored
  *structured* emitter serialises byte-for-byte to the calibrated
  ``filter_complex`` strings captured from HEAD.  Needs only plan/binding
  shapes; media paths are never opened.
- ``test_tensor_audio_parity`` (Stage 0) — proves the comparator (and later the
  PyAV renderer) matches the stock-FFmpeg reference within a perceptual dBFS
  ceiling.  Needs REAL rendered PCM, so it uses the render helpers here.

Why this exists
---------------
Both gates must exercise the *same* feature surface (multi-source ``amix`` /
``asplit``, constant-power ``pan``, ``afade`` in+out, retime ``atempo``, freeze
silence, mute automation, the silence-bed no-audible case, and all three output
layouts).  Keeping one corpus prevents the two gates from drifting apart.

The structural cases deliberately use fake ``/tmp`` binding paths: the graph
string is a pure function of the IR plus the binding channel/stream metadata,
so no media is decoded to build it.  The render helpers synthesise real tones
only where a gate actually decodes audio.

Main callers:
- ``test_audio_graph_serialize_golden.py``
- ``test_tensor_audio_parity.py``
- ``tools`` that capture golden fixtures (see the module ``__main__`` block).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import subprocess
from typing import Callable

from bladeworks.core.audio_execution import (
    AudioAssetBinding,
    AudioStreamBinding,
    probe_audio_asset,
)
from bladeworks.core.audio_ir import (
    AnimatedAudioScalar,
    AudioAutomationPoint,
    AudioControlLayer,
    AudioEnhancement,
    AudioFade,
    AudioMuteRange,
    AudioPanner,
    AudioRenderPlan,
    AudioRetimePoint,
    RenderAudioItem,
)


# --------------------------------------------------------------------------
# IR construction helpers (mirror the proven shapes in test_audio_execution.py)
# --------------------------------------------------------------------------

_LAYOUT_CHANNELS = {
    "mono": ("C",),
    "stereo": ("L", "R"),
    "surround": ("L", "R", "C", "LFE", "Ls", "Rs"),
}


def item(**overrides: object) -> RenderAudioItem:
    """Build a ``RenderAudioItem`` with sensible one-second-tone defaults."""

    values: dict[str, object] = {
        "id": "audio-1",
        "path": "spine/audio[1]",
        "name": "tone",
        "absolute_start": Fraction(0),
        "duration": Fraction(1),
        "source_start": Fraction(0),
        "source_duration": Fraction(1),
        "asset_id": "asset-a",
        "asset_uid": None,
        "source_stream_id": "1",
        "source_sample_rate": 48_000,
        "source_channels": (),
        "output_channels": None,
        "role": None,
        "enabled": True,
        "active": True,
        "control_layers": (),
        "retime": (),
        "preserves_pitch": True,
    }
    values.update(overrides)
    return RenderAudioItem(**values)  # type: ignore[arg-type]


def plan(
    *items: RenderAudioItem,
    duration: Fraction = Fraction(2),
    layout: str = "stereo",
) -> AudioRenderPlan:
    """Wrap items in an ``AudioRenderPlan`` for the requested output layout."""

    return AudioRenderPlan(
        schema_version=2,
        source_sha256="corpus-sha",
        sequence_duration=duration,
        sample_rate=48_000,
        layout=layout,  # type: ignore[arg-type]
        output_channels=_LAYOUT_CHANNELS[layout],
        items=tuple(items),
        findings=(),
    )


def binding(
    *,
    asset_id: str = "asset-a",
    channels: int = 1,
    layout: str = "mono",
    streams: int = 1,
    path: Path = Path("/tmp/audio-fixture.wav"),
) -> AudioAssetBinding:
    """Build a synthetic asset binding; ``path`` is metadata only for structural cases."""

    return AudioAssetBinding(
        asset_id=asset_id,
        path=path,
        streams=tuple(
            AudioStreamBinding(str(index + 1), index, channels, 48_000, layout)
            for index in range(streams)
        ),
    )


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One named graph-shape fixture.

    ``bindings_for`` lets a render gate rebind the same plan onto real probed
    media (keyed by asset id) while the structural gate uses the fake bindings.
    """

    name: str
    plan: AudioRenderPlan
    bindings: dict[str, AudioAssetBinding]


def _gain(initial: float = -6.0, *, fades: tuple[AudioFade, ...] = ()) -> AnimatedAudioScalar:
    return AnimatedAudioScalar(
        initial=initial,
        unit="dB",
        keyframes=(
            AudioAutomationPoint(Fraction(0), initial, "linear", "linear"),
            AudioAutomationPoint(Fraction(1), 0, "linear", "linear"),
        ),
        fades=fades,
    )


# Corpus cases whose OUTPUT layout the renderer deliberately rejects as an
# unsupported capability, rather than rendering. ``surround_layout`` requests 5.1
# output, which ``build_audio_execution_plan`` now rejects loudly at plan time
# (its 5.1 ``pan`` upmix names side channels FFmpeg 8's ``5.1`` layout does not
# accept). Such a case has NO renderable/serialisable graph, so build-based gates
# (perceptual parity, golden serialisation) must skip it and assert the clean
# ``RenderCapabilityError`` separately instead.
CAPABILITY_REJECTED_CASES = frozenset({"surround_layout"})


def structural_cases() -> list[Case]:
    """Return every graph-shape fixture with fake (unopened) binding paths.

    These drive the Stage 2a byte-identity gate.  Each case is a distinct
    calibrated graph shape; see the module docstring for the covered surface.
    Cases named in ``CAPABILITY_REJECTED_CASES`` are still returned (a rejection
    gate asserts on them), but do NOT build into a graph.
    """

    cases: list[Case] = []

    # 1. Single stereo item — the baseline atrim/pan/aresample/adelay chain.
    cases.append(Case("single_stereo", plan(item(absolute_start=Fraction(1, 10))), {"asset-a": binding()}))

    # 2. Multi-source: two items reuse one stream -> asplit=2 then multi-input amix.
    a1 = item(id="z", asset_id="z-asset")
    a2 = item(id="a", asset_id="a-asset", absolute_start=Fraction(1))
    a3 = item(id="reuse", asset_id="a-asset", absolute_start=Fraction(1, 2))
    cases.append(
        Case(
            "multi_source_asplit",
            plan(a1, a2, a3),
            {"z-asset": binding(asset_id="z-asset"), "a-asset": binding(asset_id="a-asset")},
        )
    )

    # 3. Gain + mute + constant-power stereo panner (channelsplit/join path).
    controlled = item(
        control_layers=(
            AudioControlLayer(
                path="spine/audio[1]",
                gain=_gain(),
                mutes=(AudioMuteRange(Fraction(1, 4), Fraction(1, 4)),),
                panner=AudioPanner(
                    mode="stereo",
                    amount=AnimatedAudioScalar(initial=-1, unit="normalized"),
                    parameters={},
                ),
            ),
        )
    )
    cases.append(Case("gain_mute_pan", plan(controlled), {"asset-a": binding()}))

    # 4. Gain automation with FCP fade handles (afade in + afade out).
    faded = item(
        control_layers=(
            AudioControlLayer(
                path="spine/audio[1]",
                gain=_gain(
                    fades=(
                        AudioFade("in", Fraction(1, 4), "easeIn"),
                        AudioFade("out", Fraction(1, 4), "linear"),
                    ),
                ),
            ),
        )
    )
    cases.append(Case("gain_fades", plan(faded), {"asset-a": binding()}))

    # 5. Linear retime 1s->2s source -> atempo=2 piecewise graph.
    retimed = item(
        duration=Fraction(1),
        source_duration=Fraction(2),
        retime=(
            AudioRetimePoint(Fraction(0), Fraction(0), "linear"),
            AudioRetimePoint(Fraction(1), Fraction(2), "linear"),
        ),
    )
    cases.append(Case("retime_atempo", plan(retimed), {"asset-a": binding()}))

    # 6. Freeze retime -> calibrated anullsrc silence.
    freeze = item(
        duration=Fraction(1),
        source_duration=Fraction(1),
        retime=(
            AudioRetimePoint(Fraction(0), Fraction(0), "linear"),
            AudioRetimePoint(Fraction(1), Fraction(0), "linear"),
        ),
    )
    cases.append(Case("freeze_silence", plan(freeze), {"asset-a": binding()}))

    # 7. adjust-loudness enhancement -> dynaudnorm.
    enhanced = item(
        control_layers=(
            AudioControlLayer(
                path="spine/audio[1]",
                enhancements=(
                    AudioEnhancement(
                        kind="adjust-loudness",
                        attributes={"amount": "35", "uniformity": "50"},
                        parameters={},
                        backend_status="pending_audio_3",
                    ),
                ),
            ),
        )
    )
    cases.append(Case("dynaudnorm", plan(enhanced), {"asset-a": binding()}))

    # 8-9. Mono and surround (5.1) output layouts, stereo source downmix/upmix.
    cases.append(
        Case(
            "mono_layout",
            plan(item(source_channels=(1, 2)), layout="mono"),
            {"asset-a": binding(channels=2, layout="stereo")},
        )
    )
    cases.append(
        Case(
            "surround_layout",
            plan(item(source_channels=(1, 2)), layout="surround"),
            {"asset-a": binding(channels=2, layout="stereo")},
        )
    )

    # 10. Silence-bed only: an inactive item leaves the sequence with no audible
    #     source, exercising the anullsrc bed + single-input amix.
    cases.append(Case("silence_bed_only", plan(item(active=False)), {"asset-a": binding()}))

    return cases


# --------------------------------------------------------------------------
# Render helpers (real media) for the Stage 0 perceptual gate
# --------------------------------------------------------------------------


def generate_tone(
    path: Path,
    *,
    frequency: int = 997,
    duration: float = 2.0,
    channels: int = 1,
    ffmpeg: str,
) -> None:
    """Synthesise a deterministic sine tone for probing/rendering.

    ``channels`` upmixes the mono sine to that channel count (``-ac``) so a case
    whose plan reads ``source_channels=(1, 2)`` can be re-bound onto real stereo
    media in the perceptual gate.
    """

    upmix = ("-ac", str(channels)) if channels != 1 else ()
    subprocess.run(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:sample_rate=48000:duration={duration}",
            *upmix,
            "-c:a",
            "pcm_s16le",
            str(path),
        ),
        check=True,
    )


def rebind_on_real_media(
    case: Case,
    tmp_dir: Path,
    *,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, AudioAssetBinding]:
    """Re-probe each asset in ``case`` onto a freshly generated real tone.

    Returns bindings keyed by asset id so a render gate can decode actual audio
    while keeping the exact plan shape from the structural corpus.  Distinct
    frequencies per asset keep multi-source mixes non-degenerate.
    """

    bindings: dict[str, AudioAssetBinding] = {}
    for offset, asset_id in enumerate(sorted(case.bindings)):
        media = tmp_dir / f"{asset_id}.wav"
        # Honour the case's declared channel count so a plan that reads two
        # source channels resolves against real stereo media.
        channels = case.bindings[asset_id].streams[0].channels
        generate_tone(
            media,
            frequency=440 + 220 * offset,
            channels=channels,
            ffmpeg=ffmpeg,
        )
        bindings[asset_id] = probe_audio_asset(asset_id, media, ffprobe_path=ffprobe)
    return bindings
