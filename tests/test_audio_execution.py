"""Isolated stock-FFmpeg execution tests for audio IR v2.

Architecture map
================

Typed ``AudioRenderPlan`` fixtures
    -> explicit/probed local media bindings
    -> ``build_audio_execution_plan``
    -> graph/manifest fail-closed checks
    -> small real PCM renders for sample timing, routing, gain, pan, and hashes

These tests live under the experimental renderer runner.  They do not add the
unfinished audio engine to the production backend test job.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import shutil
import struct
import subprocess
import wave

import pytest

from bladeworks.core.audio_execution import (
    AudioAssetBinding,
    AudioAssetBindingError,
    AudioStreamBinding,
    AudioStreamResolutionError,
    UnsupportedAudioControlError,
    build_audio_execution_plan,
    probe_audio_asset,
    probe_stock_ffmpeg_audio_capabilities,
    run_audio_execution_plan,
)
from bladeworks.core.errors import RenderCapabilityError
from bladeworks.core.audio_ir import (
    AnimatedAudioScalar,
    AudioAutomationPoint,
    AudioControlLayer,
    AudioEnhancement,
    AudioMuteRange,
    AudioPanner,
    AudioRenderPlan,
    AudioRetimePoint,
    AudioSourceInstance,
    RenderAudioItem,
)
from bladeworks.core.render_sources import InstanceStreamTiming
from bladeworks.core.retime import RetimeMap, RetimePoint


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _item(**overrides: object) -> RenderAudioItem:
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


def _plan(
    *items: RenderAudioItem,
    duration: Fraction = Fraction(2),
    layout: str = "stereo",
) -> AudioRenderPlan:
    channels = {
        "mono": ("C",),
        "stereo": ("L", "R"),
        "surround": ("L", "R", "C", "LFE", "Ls", "Rs"),
    }[layout]
    return AudioRenderPlan(
        schema_version=2,
        source_sha256="fixture-sha",
        sequence_duration=duration,
        sample_rate=48_000,
        layout=layout,  # type: ignore[arg-type]
        output_channels=channels,
        items=tuple(items),
        findings=(),
    )


def _binding(
    *,
    asset_id: str = "asset-a",
    channels: int = 1,
    layout: str = "mono",
    streams: int = 1,
    path: Path = Path("/tmp/audio-fixture.wav"),
) -> AudioAssetBinding:
    return AudioAssetBinding(
        asset_id=asset_id,
        path=path,
        streams=tuple(
            AudioStreamBinding(str(index + 1), index, channels, 48_000, layout)
            for index in range(streams)
        ),
    )


def _generate_mono_tone(path: Path, *, duration: float = 1.0) -> None:
    assert FFMPEG is not None
    subprocess.run(
        (
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=997:sample_rate=48000:duration={duration}",
            "-c:a",
            "pcm_s16le",
            str(path),
        ),
        check=True,
    )


def _read_pcm16(path: Path) -> tuple[int, int, tuple[tuple[int, ...], ...]]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        frames = audio.readframes(audio.getnframes())
    values = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    per_channel = tuple(tuple(values[index::channels]) for index in range(channels))
    return sample_rate, channels, per_channel


def _rms(values: tuple[int, ...]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def test_plan_resolves_defaults_orders_graph_and_never_adds_transition_audio() -> None:
    item = _item(absolute_start=Fraction(1, 10))
    execution = build_audio_execution_plan(
        _plan(item), {"asset-a": _binding()}
    )

    graph = execution.filter_complex
    assert graph.index("atrim=start=0:end=1") < graph.index("pan=stereo")
    assert graph.index("pan=stereo") < graph.index("aresample=48000")
    assert "adelay=delays=4800S" in graph
    assert "amix=inputs=2:duration=first:dropout_transition=0:normalize=0" in graph
    assert "acrossfade" not in graph
    assert "loudnorm" not in graph
    assert "alimiter" not in graph
    assert execution.items[0].source_channels == (1,)
    assert execution.items[0].output_routes == (
        (1, "FL", Fraction(1)),
        (1, "FR", Fraction(1)),
    )
    assert execution.manifest()["mix"] == {
        "filter": "amix",
        "normalize": False,
        "hidden_limiter": False,
        "hidden_loudness_normalization": False,
        "video_transition_audio": False,
    }


def test_input_and_source_split_order_is_deterministic() -> None:
    a1 = _item(id="z", asset_id="z-asset")
    a2 = _item(id="a", asset_id="a-asset", absolute_start=Fraction(1))
    a3 = _item(id="reuse", asset_id="a-asset", absolute_start=Fraction(1, 2))
    bindings = {
        "z-asset": _binding(asset_id="z-asset"),
        "a-asset": _binding(asset_id="a-asset"),
    }

    first = build_audio_execution_plan(_plan(a1, a2, a3), bindings)
    second = build_audio_execution_plan(_plan(a1, a2, a3), bindings)

    assert first.input_asset_ids == ("a-asset", "z-asset")
    assert "[0:a:0]asplit=2" in first.filter_complex
    assert first.filter_complex == second.filter_complex
    assert first.manifest_sha256 == second.manifest_sha256


def test_explicit_channel_routes_and_supported_output_layouts() -> None:
    explicit = _item(source_channels=(2,), output_channels=("R",))
    stereo = build_audio_execution_plan(
        _plan(explicit), {"asset-a": _binding(channels=2, layout="stereo")}
    )
    assert stereo.items[0].output_routes == ((2, "FR", Fraction(1)),)
    assert "pan=stereo|FL=0*c0|FR=c1" in stereo.filter_complex

    mono = build_audio_execution_plan(
        _plan(_item(source_channels=(1, 2)), layout="mono"),
        {"asset-a": _binding(channels=2, layout="stereo")},
    )
    assert mono.ffmpeg_layout == "mono"
    assert mono.items[0].output_routes == (
        (1, "FC", Fraction(1, 2)),
        (2, "FC", Fraction(1, 2)),
    )


def test_surround_output_layout_is_rejected_as_a_capability_gap() -> None:
    """Surround (5.1) output rejects loudly before any graph node is built.

    The 5.1 ``pan`` upmix names side channels ``SL``/``SR`` while FFmpeg 8's
    ``5.1`` layout uses ``BL``/``BR``, which libav rejects with a cryptic
    ``Errno 22`` deep in execution. ``build_audio_execution_plan`` instead raises
    a clear ``RenderCapabilityError`` naming the construct at plan time. The reject
    fires ahead of binding resolution, so empty bindings suffice.
    """

    with pytest.raises(RenderCapabilityError, match="surround"):
        build_audio_execution_plan(
            _plan(_item(source_channels=(1, 2)), layout="surround"), {}
        )


def test_missing_assets_streams_channels_and_unknown_layout_fail_closed() -> None:
    with pytest.raises(AudioAssetBindingError, match="unresolved asset"):
        build_audio_execution_plan(_plan(_item()), {})
    with pytest.raises(AudioStreamResolutionError, match="known streams"):
        build_audio_execution_plan(
            _plan(_item(source_stream_id="2")), {"asset-a": _binding()}
        )
    with pytest.raises(AudioStreamResolutionError, match="source channels"):
        build_audio_execution_plan(
            _plan(_item(source_channels=(2,))), {"asset-a": _binding()}
        )
    with pytest.raises(AudioStreamResolutionError, match="recognized channel layout"):
        build_audio_execution_plan(
            _plan(_item(source_channels=(1, 2, 3))),
            {"asset-a": _binding(channels=3, layout="unknown")},
        )


def test_controls_are_ordered_and_unsupported_enhancements_or_aux_fail() -> None:
    gain = AnimatedAudioScalar(
        initial=-6,
        unit="dB",
        keyframes=(
            AudioAutomationPoint(Fraction(0), -6, "linear", "linear"),
            AudioAutomationPoint(Fraction(1), 0, "linear", "linear"),
        ),
    )
    mute = AudioMuteRange(Fraction(1, 4), Fraction(1, 4))
    panner = AudioPanner(
        mode="stereo",
        amount=AnimatedAudioScalar(initial=-1, unit="normalized"),
        parameters={},
    )
    controlled = _item(
        control_layers=(
            AudioControlLayer(
                path="spine/audio[1]", gain=gain, mutes=(mute,), panner=panner
            ),
        )
    )
    execution = build_audio_execution_plan(
        _plan(controlled), {"asset-a": _binding()}
    )
    graph = execution.filter_complex
    assert graph.index("_gain_") < graph.index("_mute_") < graph.index("channelsplit")
    assert {"volume", "channelsplit", "join"}.issubset(execution.required_filters)
    assert "cos((" in graph
    assert "+1)*PI/4)" in graph
    assert "sin((" in graph

    enhancement = AudioEnhancement(
        kind="adjust-loudness",
        attributes={},
        parameters={},
        backend_status="pending_audio_3",
    )
    unsupported = replace(
        controlled,
        control_layers=(
            AudioControlLayer(path="spine/audio[1]", enhancements=(enhancement,)),
        ),
    )
    with pytest.raises(
        UnsupportedAudioControlError,
        match="unexecutable audio enhancement.*missing attributes",
    ):
        build_audio_execution_plan(
            _plan(unsupported), {"asset-a": _binding()}
        )

    calibrated = replace(
        enhancement,
        attributes={"amount": "35", "uniformity": "50"},
    )
    enhanced = replace(
        controlled,
        control_layers=(
            AudioControlLayer(path="spine/audio[1]", enhancements=(calibrated,)),
        ),
    )
    enhanced_plan = build_audio_execution_plan(
        _plan(enhanced), {"asset-a": _binding()}
    )
    assert "dynaudnorm=" in enhanced_plan.filter_complex
    assert "dynaudnorm" in enhanced_plan.required_filters
    assert enhanced_plan.items[0].enhancement_plans[0]["executable"] is True

    aux_gain = replace(
        gain,
        keyframes=(
            AudioAutomationPoint(Fraction(0), -6, "linear", "linear", "2"),
            gain.keyframes[1],
        ),
    )
    aux = replace(
        controlled,
        control_layers=(AudioControlLayer(path="x", gain=aux_gain),),
    )
    with pytest.raises(UnsupportedAudioControlError, match="auxValue"):
        build_audio_execution_plan(_plan(aux), {"asset-a": _binding()})


def test_retime_uses_piecewise_stock_filters_and_freeze_is_calibrated_silence() -> None:
    retimed = _item(
        duration=Fraction(1),
        source_duration=Fraction(2),
        retime=(
            AudioRetimePoint(Fraction(0), Fraction(0), "linear"),
            AudioRetimePoint(Fraction(1), Fraction(2), "linear"),
        ),
    )
    execution = build_audio_execution_plan(
        _plan(retimed), {"asset-a": _binding()}
    )
    assert execution.items[0].retimed is True
    assert "atempo=2" in execution.filter_complex
    assert "atempo" in execution.required_filters

    freeze = replace(
        retimed,
        source_duration=Fraction(1),
        retime=(
            AudioRetimePoint(Fraction(0), Fraction(0), "linear"),
            AudioRetimePoint(Fraction(1), Fraction(0), "linear"),
        ),
    )
    frozen = build_audio_execution_plan(_plan(freeze), {"asset-a": _binding()})
    # A fully frozen item is pure calibrated silence: it reads no source media, so
    # it emits an ``anullsrc`` bed trimmed to the freeze duration and NO trim/pan
    # route (that route's ``pan`` output would otherwise be left unconnected).
    assert "anullsrc=" in frozen.filter_complex
    assert "atrim=duration=1" in frozen.filter_complex
    assert "pan=" not in frozen.filter_complex
    assert "pan" not in frozen.required_filters
    assert "anullsrc" in frozen.required_filters


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="stock FFmpeg is unavailable")
def test_fully_frozen_virtual_source_terminates_unused_descendant_mix(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    output = tmp_path / "frozen-source.wav"
    _generate_mono_tone(source)
    binding = probe_audio_asset("asset-a", source, ffprobe_path=FFPROBE)
    instance_path = "spine/ref-clip[1]"
    freeze_map = RetimeMap.from_points(
        (
            RetimePoint(Fraction(0), Fraction(0)),
            RetimePoint(Fraction(1), Fraction(0)),
        )
    )
    instance = AudioSourceInstance(
        path=instance_path,
        source_id="compound-a",
        ancestor_paths=(),
        timing=InstanceStreamTiming(
            stream="audio",
            absolute_start=Fraction(0),
            duration=Fraction(1),
            source_start=Fraction(0),
            source_duration=Fraction(1),
            retime_map=freeze_map,
        ),
        preserves_pitch=True,
    )
    item = _item(path=f"{instance_path}/asset-clip[1]", ancestor_paths=(instance_path,))
    plan = replace(_plan(item), source_instances=(instance,))

    execution = build_audio_execution_plan(plan, {"asset-a": binding})
    run_audio_execution_plan(execution, output, ffmpeg_path=FFMPEG)

    assert "anullsink" in execution.required_filters
    assert "]anullsink" in execution.filter_complex
    assert output.stat().st_size > 0


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="stock FFmpeg is unavailable")
def test_ffprobe_enumerates_multiple_audio_streams(tmp_path: Path) -> None:
    media = tmp_path / "two-streams.mka"
    subprocess.run(
        (
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=48000:duration=0.1",
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-c:a",
            "pcm_s16le",
            str(media),
        ),
        check=True,
    )

    binding = probe_audio_asset("asset-a", media, ffprobe_path=FFPROBE)
    assert [(stream.source_stream_id, stream.audio_ordinal) for stream in binding.streams] == [
        ("1", 0),
        ("2", 1),
    ]
    assert all(stream.channels == 1 for stream in binding.streams)


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="stock FFmpeg is unavailable")
def test_real_pcm_render_has_sample_onset_no_leakage_gain_pan_and_determinism(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tone.wav"
    _generate_mono_tone(source)
    binding = probe_audio_asset("asset-a", source, ffprobe_path=FFPROBE)
    controls = AudioControlLayer(
        path="spine/audio[1]",
        gain=AnimatedAudioScalar(initial=-6.0, unit="dB"),
        panner=AudioPanner(
            mode="stereo",
            amount=AnimatedAudioScalar(initial=-1.0, unit="normalized"),
            parameters={},
        ),
    )
    item = _item(
        absolute_start=Fraction(1, 10),
        control_layers=(controls,),
    )
    execution = build_audio_execution_plan(_plan(item), {"asset-a": binding})
    capability = probe_stock_ffmpeg_audio_capabilities(execution, ffmpeg_path=FFMPEG)
    capability.require_supported()
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    run_audio_execution_plan(
        execution, first, ffmpeg_path=FFMPEG, codec="pcm_s16le"
    )
    run_audio_execution_plan(
        execution, second, ffmpeg_path=FFMPEG, codec="pcm_s16le"
    )

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    source_rate, source_channels, source_pcm = _read_pcm16(source)
    output_rate, output_channels, output_pcm = _read_pcm16(first)
    assert (source_rate, source_channels) == (48_000, 1)
    assert (output_rate, output_channels) == (48_000, 2)
    assert len(output_pcm[0]) == 96_000
    onset = next(index for index, value in enumerate(output_pcm[0]) if abs(value) > 100)
    assert abs(onset - 4_801) <= 1
    assert max(abs(value) for value in output_pcm[1]) == 0
    source_rms = _rms(source_pcm[0][1_000:40_000])
    output_rms = _rms(output_pcm[0][5_800:44_800])
    assert output_rms / source_rms == pytest.approx(10 ** (-6 / 20), abs=0.015)


def test_inactive_items_need_no_asset_and_produce_sequence_silence() -> None:
    inactive = _item(enabled=False)
    execution = build_audio_execution_plan(_plan(inactive), {})
    assert execution.inputs == ()
    assert execution.items == ()
    assert execution.filter_complex.startswith("anullsrc=")
    assert "amix=inputs=1" in execution.filter_complex
