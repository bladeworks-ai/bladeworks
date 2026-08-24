"""Experimental tests for bounded intrinsic audio-enhancement planning.

Architecture map
================

Typed ``AudioEnhancement`` fixtures
    -> strict standalone planner validation
    -> deterministic filters/manifests and explicit blocked findings
    -> small real PCM renders through the installed stock FFmpeg

These tests remain under the isolated renderer runner and do not enter the
production backend test job.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import wave

import pytest

from bladeworks.core.audio_enhancements import (
    AudioEnhancementValidationError,
    UnsupportedAudioEnhancementError,
    VoiceIsolationModelError,
    build_audio_enhancement_plan,
    load_frozen_voice_isolation_model,
    required_stock_filters,
)
from bladeworks.core.audio_ir import AudioEnhancement


FFMPEG = shutil.which("ffmpeg")


def _enhancement(
    kind: str,
    attributes: dict[str, str],
    *,
    parameters: dict[str, str] | None = None,
    opaque_data: str | None = None,
) -> AudioEnhancement:
    return AudioEnhancement(
        kind=kind,
        attributes=attributes,
        parameters=parameters or {},
        backend_status=(
            "not_implemented_yet"
            if kind == "adjust-matchEQ"
            else "pending_audio_3"
        ),
        opaque_data=opaque_data,
    )


def test_supported_adjustments_preserve_order_and_exact_filter_requirements() -> None:
    enhancements = (
        _enhancement("adjust-loudness", {"amount": "59", "uniformity": "3"}),
        _enhancement("adjust-noiseReduction", {"amount": "50"}),
        _enhancement("adjust-humReduction", {"frequency": "50"}),
        _enhancement("adjust-EQ", {"mode": "voice_enhance"}),
    )

    first = build_audio_enhancement_plan(enhancements)
    second = build_audio_enhancement_plan(enhancements)

    assert first.executable is True
    assert [step.kind for step in first.steps] == [
        "adjust-loudness",
        "adjust-noiseReduction",
        "adjust-humReduction",
        "adjust-EQ",
    ]
    assert first.required_filters == (
        "afftdn",
        "bandreject",
        "dynaudnorm",
        "equalizer",
        "highpass",
        "lowpass",
    )
    assert required_stock_filters(enhancements) == first.required_filters
    assert required_stock_filters(first) == first.required_filters
    assert first.manifest() == second.manifest()
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest()["operation_position"] == "after_route_gain_pan_retime"
    assert all(
        finding.disposition == "semantic_approximation"
        for finding in first.findings
    )


@pytest.mark.parametrize(
    ("mode", "expected_filters"),
    (
        ("flat", ()),
        ("voice_enhance", ("highpass", "equalizer", "lowpass")),
        ("music_enhance", ("equalizer", "equalizer")),
        ("loudness", ("equalizer", "equalizer")),
        ("hum_reduction", ("bandreject",) * 4),
        ("bass_boost", ("equalizer",)),
        ("bass_reduce", ("equalizer",)),
        ("treble_boost", ("equalizer",)),
        ("treble_reduce", ("equalizer",)),
    ),
)
def test_every_dtd_named_eq_mode_has_one_frozen_preset(
    mode: str, expected_filters: tuple[str, ...]
) -> None:
    plan = build_audio_enhancement_plan(
        (_enhancement("adjust-EQ", {"mode": mode}),)
    )
    assert tuple(value.split("=", 1)[0] for value in plan.filters) == expected_filters


def test_zero_amounts_are_explicit_noops_not_dropped_records() -> None:
    plan = build_audio_enhancement_plan(
        (
            _enhancement(
                "adjust-loudness", {"amount": "0", "uniformity": "100"}
            ),
            _enhancement("adjust-noiseReduction", {"amount": "0"}),
            _enhancement("adjust-voiceIsolation", {"amount": "0"}),
        )
    )
    assert len(plan.steps) == 3
    assert plan.filters == ()
    assert plan.executable is True
    assert all(step.disposition == "semantic_approximation" for step in plan.steps)


def test_match_eq_and_missing_voice_model_are_typed_blocking_findings() -> None:
    match_eq = _enhancement(
        "adjust-matchEQ", {}, opaque_data="opaque-apple-archive"
    )
    voice = _enhancement("adjust-voiceIsolation", {"amount": "75"})
    plan = build_audio_enhancement_plan((match_eq, voice))

    assert plan.executable is False
    assert [step.disposition for step in plan.steps] == [
        "not_implemented_yet",
        "not_implemented_yet",
    ]
    assert [finding.code for finding in plan.findings] == [
        "audio_match_eq_opaque_not_implemented",
        "audio_voice_isolation_model_not_available",
    ]
    with pytest.raises(UnsupportedAudioEnhancementError, match="opaque Apple archive"):
        plan.require_executable()


@pytest.mark.parametrize("raw", ("-0.1", "100.1", "nan", "inf", " 50", "50%", ""))
def test_amounts_are_finite_plain_numbers_in_the_frozen_range(raw: str) -> None:
    with pytest.raises(AudioEnhancementValidationError):
        build_audio_enhancement_plan(
            (_enhancement("adjust-noiseReduction", {"amount": raw}),)
        )


def test_unknown_missing_extra_duplicate_and_unpublished_controls_fail() -> None:
    invalid_cases = (
        (_enhancement("custom-audio-magic", {}),),
        (_enhancement("adjust-loudness", {"amount": "50"}),),
        (_enhancement("adjust-noiseReduction", {"amount": "50", "x": "1"}),),
        (
            _enhancement("adjust-humReduction", {"frequency": "55"}),
        ),
        (_enhancement("adjust-EQ", {"mode": "cinema"}),),
        (
            _enhancement(
                "adjust-EQ", {"mode": "flat"}, parameters={"secret": "1"}
            ),
        ),
        (
            _enhancement("adjust-noiseReduction", {"amount": "10"}),
            _enhancement("adjust-noiseReduction", {"amount": "20"}),
        ),
        (
            _enhancement("adjust-EQ", {"mode": "flat"}),
            _enhancement("adjust-matchEQ", {}, opaque_data="x"),
        ),
    )
    for enhancements in invalid_cases:
        with pytest.raises(AudioEnhancementValidationError):
            build_audio_enhancement_plan(enhancements)
    with pytest.raises(AudioEnhancementValidationError, match="sequence"):
        build_audio_enhancement_plan("not-a-sequence")  # type: ignore[arg-type]


def test_voice_model_load_is_local_checksum_pinned_and_registry_owned(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    assert load_frozen_voice_isolation_model(registry) is None

    model = registry / "models" / "speech.rnnn"
    model.parent.mkdir()
    model.write_bytes(b"frozen-test-model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = registry / "voice_isolation.v1.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "fcpxml_voice_isolation_model.v1",
                "model_id": "speech-v1",
                "path": "models/speech.rnnn",
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    frozen = load_frozen_voice_isolation_model(registry)
    assert frozen is not None
    plan = build_audio_enhancement_plan(
        (_enhancement("adjust-voiceIsolation", {"amount": "60"}),),
        voice_model=frozen,
    )
    assert plan.executable is True
    assert plan.required_filters == ("arnndn",)
    assert frozen.sha256 in json.dumps(plan.manifest())
    assert str(model.resolve()) in plan.filters[0]

    model.write_bytes(b"changed")
    with pytest.raises(VoiceIsolationModelError, match="checksum mismatch"):
        load_frozen_voice_isolation_model(registry)


def test_voice_model_manifest_cannot_escape_registry(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    outside = tmp_path / "outside.rnnn"
    outside.write_bytes(b"outside")
    (registry / "voice_isolation.v1.json").write_text(
        json.dumps(
            {
                "schema": "fcpxml_voice_isolation_model.v1",
                "model_id": "bad",
                "path": "../outside.rnnn",
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(VoiceIsolationModelError, match="escapes"):
        load_frozen_voice_isolation_model(registry)


def _write_two_tone(path: Path, *, duration: float = 1.0) -> None:
    sample_rate = 48_000
    samples = []
    for index in range(round(duration * sample_rate)):
        time = index / sample_rate
        value = 0.30 * math.sin(2 * math.pi * 60 * time)
        value += 0.20 * math.sin(2 * math.pi * 997 * time)
        samples.append(round(max(-1.0, min(1.0, value)) * 32767))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _read_pcm16(path: Path) -> tuple[int, tuple[int, ...]]:
    with wave.open(str(path), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        frames = audio.readframes(audio.getnframes())
        values = struct.unpack(f"<{len(frames) // 2}h", frames)
        return audio.getframerate(), values


def _tone_amplitude(samples: tuple[int, ...], sample_rate: int, hz: float) -> float:
    # Ignore the first 100 ms so the causal notch filters can settle.
    values = samples[sample_rate // 10 :]
    cosine = 0.0
    sine = 0.0
    for index, value in enumerate(values):
        phase = 2 * math.pi * hz * index / sample_rate
        cosine += value * math.cos(phase)
        sine += value * math.sin(phase)
    return 2 * math.hypot(cosine, sine) / len(values)


@pytest.mark.skipif(not FFMPEG, reason="stock FFmpeg is unavailable")
def test_real_pcm_hum_reduction_is_effective_selective_and_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_two_tone(source)
    plan = build_audio_enhancement_plan(
        (_enhancement("adjust-humReduction", {"frequency": "60"}),)
    )
    for output in (first, second):
        subprocess.run(
            (
                FFMPEG,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-af",
                ",".join(plan.filters),
                "-c:a",
                "pcm_s16le",
                str(output),
            ),
            check=True,
        )
    assert first.read_bytes() == second.read_bytes()
    source_rate, source_pcm = _read_pcm16(source)
    output_rate, output_pcm = _read_pcm16(first)
    assert source_rate == output_rate == 48_000
    assert len(source_pcm) == len(output_pcm) == 48_000
    assert _tone_amplitude(output_pcm, output_rate, 60) < (
        0.25 * _tone_amplitude(source_pcm, source_rate, 60)
    )
    assert _tone_amplitude(output_pcm, output_rate, 997) > (
        0.90 * _tone_amplitude(source_pcm, source_rate, 997)
    )


@pytest.mark.skipif(not FFMPEG, reason="stock FFmpeg is unavailable")
@pytest.mark.parametrize(
    "enhancement",
    (
        _enhancement("adjust-noiseReduction", {"amount": "50"}),
        _enhancement("adjust-loudness", {"amount": "40", "uniformity": "25"}),
        _enhancement("adjust-EQ", {"mode": "music_enhance"}),
    ),
)
def test_real_pcm_stock_plans_render_without_changing_sample_count(
    tmp_path: Path, enhancement: AudioEnhancement
) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "output.wav"
    _write_two_tone(source, duration=0.5)
    plan = build_audio_enhancement_plan((enhancement,))
    plan.require_executable()
    subprocess.run(
        (
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            ",".join(plan.filters),
            "-c:a",
            "pcm_s16le",
            str(output),
        ),
        check=True,
    )
    sample_rate, pcm = _read_pcm16(output)
    assert sample_rate == 48_000
    assert len(pcm) == 24_000
