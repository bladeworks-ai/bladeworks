"""Plan bounded Final Cut audio-enhancement approximations for stock FFmpeg.

Architecture map
================

``AudioEnhancement`` records from :mod:`audio_ir`
    -> strict FCP value and combination validation
    -> one ordered ``AudioEnhancementStep`` per XML adjustment
    -> stock-FFmpeg filter fragments plus an audit manifest

The eventual audio-engine seam is intentionally narrow::

    source routing -> gain/mute/pan -> retime
        -> build_audio_enhancement_plan(...).filters
        -> resample/timeline delay -> mix

Important invariants
--------------------

* These filters are semantic approximations, not claims of Apple's private
  DSP.  Their constants are deliberately bounded and frozen here for later
  Final Cut A/B calibration.
* Opaque Match EQ data is never guessed.  Voice isolation is executable only
  with a checksum-pinned model named by a local registry manifest.  This
  module never searches for or downloads a model.
* Unknown controls, malformed values, duplicate adjustment kinds, and
  unpublished ``param`` children fail before any FFmpeg graph is emitted.
* XML order is preserved.  The manifest and its digest are deterministic.

Main callers:
- The experimental audio-engine adapter after its retime stage.
- Isolated renderer tests and future compatibility-report generation.

Why this exists:
Final Cut's intrinsic audio enhancements are meaningful editorial controls,
but their implementations are private.  Keeping the approximation policy in
one typed module prevents the general audio engine from silently ignoring an
adjustment or scattering uncalibrated constants across its graph builder.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .audio_ir import AudioEnhancement


EnhancementDisposition = Literal["semantic_approximation", "not_implemented_yet"]

_KNOWN_KINDS = {
    "adjust-loudness",
    "adjust-noiseReduction",
    "adjust-humReduction",
    "adjust-EQ",
    "adjust-matchEQ",
    "adjust-voiceIsolation",
}
_EQ_MODES = {
    "flat",
    "voice_enhance",
    "music_enhance",
    "loudness",
    "hum_reduction",
    "bass_boost",
    "bass_reduce",
    "treble_boost",
    "treble_reduce",
}
_VOICE_MODEL_MANIFEST = "voice_isolation.v1.json"


class AudioEnhancementError(ValueError):
    """Base error for an invalid or non-executable enhancement plan."""


class AudioEnhancementValidationError(AudioEnhancementError):
    """An FCP enhancement record is malformed, ambiguous, or out of range."""


class UnsupportedAudioEnhancementError(AudioEnhancementError):
    """A preserved adjustment has no honest executable approximation."""


class VoiceIsolationModelError(AudioEnhancementError):
    """A local voice model or its frozen registry manifest is invalid."""


@dataclass(frozen=True)
class FrozenVoiceIsolationModel:
    """One locally available, checksum-pinned ``arnndn`` model.

    Instances should normally come from ``load_frozen_voice_isolation_model``.
    The absolute path is kept in the plan so execution never depends on the
    process working directory.
    """

    model_id: str
    path: Path
    sha256: str
    registry_manifest: Path

    def manifest(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "registry_manifest": str(self.registry_manifest),
        }


@dataclass(frozen=True)
class AudioEnhancementFinding:
    """One explicit compatibility result for a preserved adjustment."""

    code: str
    disposition: EnhancementDisposition
    kind: str
    detail: str

    def manifest(self) -> dict[str, str]:
        return {
            "code": self.code,
            "disposition": self.disposition,
            "kind": self.kind,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AudioEnhancementStep:
    """The frozen FFmpeg approximation for one FCPXML adjustment."""

    index: int
    kind: str
    disposition: EnhancementDisposition
    normalized_controls: Mapping[str, str]
    filters: tuple[str, ...]
    required_filters: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "disposition": self.disposition,
            "normalized_controls": dict(sorted(self.normalized_controls.items())),
            "filters": list(self.filters),
            "required_filters": list(self.required_filters),
        }


@dataclass(frozen=True)
class AudioEnhancementPlan:
    """An ordered enhancement chain suitable for insertion into an audio graph.

    Main callers:
    - The future AUDIO-2 integration seam appends ``filters`` after retiming.
    - Compatibility reporting consumes ``findings`` and ``manifest``.

    ``require_executable`` must be called before graph execution.  This makes
    Match EQ and missing voice models visible without pretending that a
    partially constructed graph implements them.
    """

    schema_version: int
    steps: tuple[AudioEnhancementStep, ...]
    filters: tuple[str, ...]
    required_filters: tuple[str, ...]
    findings: tuple[AudioEnhancementFinding, ...]
    voice_model: FrozenVoiceIsolationModel | None = None

    @property
    def executable(self) -> bool:
        return all(
            finding.disposition != "not_implemented_yet"
            for finding in self.findings
        )

    def require_executable(self) -> None:
        """Reject a graph that would otherwise omit one preserved adjustment."""

        blocked = [
            finding for finding in self.findings
            if finding.disposition == "not_implemented_yet"
        ]
        if blocked:
            summary = "; ".join(
                f"{finding.kind}: {finding.detail}" for finding in blocked
            )
            raise UnsupportedAudioEnhancementError(summary)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "fcpxml_audio_enhancements.v1",
            "operation_position": "after_route_gain_pan_retime",
            "approximation_policy": "bounded_stock_ffmpeg",
            "executable": self.executable,
            "required_filters": list(self.required_filters),
            "filters": list(self.filters),
            "steps": [step.manifest() for step in self.steps],
            "findings": [finding.manifest() for finding in self.findings],
            "voice_model": (
                self.voice_model.manifest() if self.voice_model is not None else None
            ),
        }

    @property
    def manifest_sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_frozen_voice_isolation_model(
    registry_root: str | Path,
) -> FrozenVoiceIsolationModel | None:
    """Load one local registry-owned model, returning ``None`` when absent.

    The only recognized registry entry is ``voice_isolation.v1.json`` under
    the caller-supplied root.  Its model path must remain inside that root and
    its bytes must match the declared SHA-256.  No directory search, network
    access, package installation, or fallback location is attempted.

    Main callers:
    - Startup/integration code that already owns a renderer asset registry.

    Why this exists:
    A random model found on a workstation is neither reproducible nor safe to
    treat as part of the renderer.  The manifest makes model ownership and
    exact bytes explicit.
    """

    root = Path(registry_root).expanduser().resolve()
    manifest_path = root / _VOICE_MODEL_MANIFEST
    if not manifest_path.exists():
        return None
    if not manifest_path.is_file():
        raise VoiceIsolationModelError(
            f"voice model manifest is not a file: {manifest_path}"
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VoiceIsolationModelError(
            f"could not read voice model manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise VoiceIsolationModelError("voice model manifest must be a JSON object")
    allowed_keys = {"schema", "model_id", "path", "sha256"}
    extra_keys = set(raw) - allowed_keys
    if extra_keys:
        raise VoiceIsolationModelError(
            "voice model manifest has unknown keys: " + ", ".join(sorted(extra_keys))
        )
    if raw.get("schema") != "fcpxml_voice_isolation_model.v1":
        raise VoiceIsolationModelError("unsupported voice model manifest schema")
    model_id = _required_nonempty_string(raw.get("model_id"), "model_id")
    relative_path = _required_nonempty_string(raw.get("path"), "path")
    declared_sha256 = _required_sha256(raw.get("sha256"))
    path_value = Path(relative_path)
    if path_value.is_absolute():
        raise VoiceIsolationModelError("voice model path must be registry-relative")
    model_path = (root / path_value).resolve()
    try:
        model_path.relative_to(root)
    except ValueError as error:
        raise VoiceIsolationModelError(
            "voice model path escapes the registry root"
        ) from error
    if not model_path.is_file():
        raise VoiceIsolationModelError(f"voice model is missing: {model_path}")
    actual_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_sha256 != declared_sha256:
        raise VoiceIsolationModelError(
            f"voice model checksum mismatch for {model_path}"
        )
    return FrozenVoiceIsolationModel(
        model_id=model_id,
        path=model_path,
        sha256=actual_sha256,
        registry_manifest=manifest_path.resolve(),
    )


def build_audio_enhancement_plan(
    enhancements: Sequence[AudioEnhancement],
    *,
    voice_model: FrozenVoiceIsolationModel | None = None,
) -> AudioEnhancementPlan:
    """Validate and compile ordered FCP enhancements into filter fragments.

    Values are accepted only inside FCP's meaningful normalized/percentage
    range of 0 through 100.  Zero-valued amount controls intentionally emit no
    filter, but remain represented in the manifest.  Named EQ modes are fixed
    presets; unpublished ``param`` children are rejected instead of guessed.

    Main callers:
    - The future audio engine, once per ordered ``AudioControlLayer``.
    """

    if isinstance(enhancements, (str, bytes)) or not isinstance(
        enhancements, Sequence
    ):
        raise AudioEnhancementValidationError(
            "enhancements must be a sequence of AudioEnhancement values"
        )
    seen: set[str] = set()
    steps: list[AudioEnhancementStep] = []
    findings: list[AudioEnhancementFinding] = []
    all_filters: list[str] = []
    required: set[str] = set()
    for index, enhancement in enumerate(enhancements):
        if not isinstance(enhancement, AudioEnhancement):
            raise AudioEnhancementValidationError(
                f"enhancement {index} must be an AudioEnhancement"
            )
        kind = enhancement.kind
        if kind not in _KNOWN_KINDS:
            raise AudioEnhancementValidationError(
                f"unknown audio enhancement kind {kind!r}"
            )
        if kind in seen:
            raise AudioEnhancementValidationError(
                f"duplicate audio enhancement kind {kind!r}"
            )
        if kind in {"adjust-EQ", "adjust-matchEQ"} and seen.intersection(
            {"adjust-EQ", "adjust-matchEQ"}
        ):
            raise AudioEnhancementValidationError(
                "adjust-EQ and adjust-matchEQ are mutually exclusive"
            )
        seen.add(kind)
        step, step_findings = _build_step(
            index,
            enhancement,
            voice_model=voice_model,
        )
        steps.append(step)
        findings.extend(step_findings)
        all_filters.extend(step.filters)
        required.update(step.required_filters)
    return AudioEnhancementPlan(
        schema_version=1,
        steps=tuple(steps),
        filters=tuple(all_filters),
        required_filters=tuple(sorted(required)),
        findings=tuple(findings),
        voice_model=voice_model if "adjust-voiceIsolation" in seen else None,
    )


def required_stock_filters(
    enhancements_or_plan: Sequence[AudioEnhancement] | AudioEnhancementPlan,
    *,
    voice_model: FrozenVoiceIsolationModel | None = None,
) -> tuple[str, ...]:
    """Return the exact deterministic stock-filter requirement set."""

    if isinstance(enhancements_or_plan, AudioEnhancementPlan):
        return enhancements_or_plan.required_filters
    return build_audio_enhancement_plan(
        enhancements_or_plan, voice_model=voice_model
    ).required_filters


def _build_step(
    index: int,
    enhancement: AudioEnhancement,
    *,
    voice_model: FrozenVoiceIsolationModel | None,
) -> tuple[AudioEnhancementStep, tuple[AudioEnhancementFinding, ...]]:
    kind = enhancement.kind
    if enhancement.opaque_data and kind != "adjust-matchEQ":
        raise AudioEnhancementValidationError(
            f"{kind} contains unexpected opaque data"
        )
    if kind != "adjust-EQ" and enhancement.parameters:
        raise AudioEnhancementValidationError(
            f"{kind} contains unsupported parameter children"
        )
    if kind == "adjust-loudness":
        _require_exact_attributes(enhancement, {"amount", "uniformity"})
        amount = _percentage(enhancement.attributes["amount"], f"{kind}@amount")
        uniformity = _percentage(
            enhancement.attributes["uniformity"], f"{kind}@uniformity"
        )
        filters = _loudness_filters(amount, uniformity)
        controls = {
            "amount": _number_text(amount),
            "uniformity": _number_text(uniformity),
        }
        return _supported_step(index, kind, controls, filters)
    if kind == "adjust-noiseReduction":
        _require_exact_attributes(enhancement, {"amount"})
        amount = _percentage(enhancement.attributes["amount"], f"{kind}@amount")
        filters = _noise_reduction_filters(amount)
        return _supported_step(
            index, kind, {"amount": _number_text(amount)}, filters
        )
    if kind == "adjust-humReduction":
        _require_exact_attributes(enhancement, {"frequency"})
        frequency_text = enhancement.attributes["frequency"]
        if frequency_text not in {"50", "60"}:
            raise AudioEnhancementValidationError(
                f"{kind}@frequency must be exactly 50 or 60"
            )
        frequency = int(frequency_text)
        filters = _hum_filters(frequency)
        return _supported_step(
            index, kind, {"frequency_hz": str(frequency)}, filters
        )
    if kind == "adjust-EQ":
        _require_exact_attributes(enhancement, {"mode"})
        if enhancement.parameters:
            raise AudioEnhancementValidationError(
                "adjust-EQ param children are unpublished and not calibrated"
            )
        mode = enhancement.attributes["mode"]
        if mode not in _EQ_MODES:
            raise AudioEnhancementValidationError(
                f"adjust-EQ@mode is unknown: {mode!r}"
            )
        filters = _eq_filters(mode)
        return _supported_step(index, kind, {"mode": mode}, filters)
    if kind == "adjust-matchEQ":
        _require_exact_attributes(enhancement, set())
        if enhancement.parameters:
            raise AudioEnhancementValidationError(
                "adjust-matchEQ cannot contain param children"
            )
        finding = AudioEnhancementFinding(
            code="audio_match_eq_opaque_not_implemented",
            disposition="not_implemented_yet",
            kind=kind,
            detail="Match EQ contains an opaque Apple archive with no defensible stock-FFmpeg mapping",
        )
        step = AudioEnhancementStep(
            index=index,
            kind=kind,
            disposition="not_implemented_yet",
            normalized_controls={
                "opaque_data_present": "true" if enhancement.opaque_data else "false"
            },
            filters=(),
            required_filters=(),
        )
        return step, (finding,)
    if kind == "adjust-voiceIsolation":
        _require_exact_attributes(enhancement, {"amount"})
        amount = _percentage(enhancement.attributes["amount"], f"{kind}@amount")
        controls = {"amount": _number_text(amount)}
        if amount == 0:
            return _supported_step(index, kind, controls, ())
        if voice_model is None:
            finding = AudioEnhancementFinding(
                code="audio_voice_isolation_model_not_available",
                disposition="not_implemented_yet",
                kind=kind,
                detail="no checksum-pinned voice model exists in the renderer-owned local registry",
            )
            step = AudioEnhancementStep(
                index=index,
                kind=kind,
                disposition="not_implemented_yet",
                normalized_controls=controls,
                filters=(),
                required_filters=(),
            )
            return step, (finding,)
        controls = {
            **controls,
            "model_id": voice_model.model_id,
            "model_sha256": voice_model.sha256,
        }
        filters = (
            "arnndn="
            f"model='{_escape_filter_value(voice_model.path)}':"
            f"mix={_number_text(amount / 100.0)}",
        )
        return _supported_step(index, kind, controls, filters)
    raise AssertionError(f"unhandled audio enhancement kind {kind}")


def _supported_step(
    index: int,
    kind: str,
    controls: Mapping[str, str],
    filters: tuple[str, ...],
) -> tuple[AudioEnhancementStep, tuple[AudioEnhancementFinding, ...]]:
    required = tuple(sorted({_filter_name(value) for value in filters}))
    step = AudioEnhancementStep(
        index=index,
        kind=kind,
        disposition="semantic_approximation",
        normalized_controls=controls,
        filters=filters,
        required_filters=required,
    )
    finding = AudioEnhancementFinding(
        code="audio_enhancement_semantic_approximation",
        disposition="semantic_approximation",
        kind=kind,
        detail="uses a bounded stock-FFmpeg approximation pending Final Cut A/B calibration",
    )
    return step, (finding,)


def _loudness_filters(amount: float, uniformity: float) -> tuple[str, ...]:
    if amount == 0:
        return ()
    strength = amount / 100.0
    uniformity_ratio = uniformity / 100.0
    max_gain = 1.0 + 11.0 * strength
    target_rms = 0.05 + 0.13 * strength
    compression = 12.0 * uniformity_ratio
    return (
        "dynaudnorm="
        "framelen=500:gausssize=31:peak=0.95:"
        f"maxgain={_number_text(max_gain)}:"
        f"targetrms={_number_text(target_rms)}:"
        f"compress={_number_text(compression)}:"
        "coupling=1:correctdc=1:overlap=0.5",
    )


def _noise_reduction_filters(amount: float) -> tuple[str, ...]:
    if amount == 0:
        return ()
    strength = amount / 100.0
    reduction_db = 6.0 + 18.0 * strength
    adaptivity = 0.8 - 0.5 * strength
    smoothing = 1 + round(5 * strength)
    return (
        "afftdn="
        f"noise_reduction={_number_text(reduction_db)}:"
        "noise_floor=-50:track_noise=1:"
        f"adaptivity={_number_text(adaptivity)}:"
        f"gain_smooth={smoothing}:output_mode=output",
    )


def _hum_filters(frequency: int) -> tuple[str, ...]:
    return tuple(
        f"bandreject=frequency={frequency * harmonic}:width_type=q:width=20:mix=1"
        for harmonic in range(1, 5)
    )


def _eq_filters(mode: str) -> tuple[str, ...]:
    presets: Mapping[str, tuple[str, ...]] = {
        "flat": (),
        "voice_enhance": (
            "highpass=frequency=80",
            "equalizer=frequency=3000:width_type=q:width=1:gain=3",
            "lowpass=frequency=14000",
        ),
        "music_enhance": (
            "equalizer=frequency=100:width_type=q:width=0.8:gain=2",
            "equalizer=frequency=4000:width_type=q:width=0.8:gain=2",
        ),
        "loudness": (
            "equalizer=frequency=90:width_type=q:width=0.7:gain=3",
            "equalizer=frequency=6500:width_type=q:width=0.7:gain=2",
        ),
        "hum_reduction": _hum_filters(60),
        "bass_boost": (
            "equalizer=frequency=100:width_type=q:width=0.7:gain=6",
        ),
        "bass_reduce": (
            "equalizer=frequency=100:width_type=q:width=0.7:gain=-6",
        ),
        "treble_boost": (
            "equalizer=frequency=7000:width_type=q:width=0.7:gain=6",
        ),
        "treble_reduce": (
            "equalizer=frequency=7000:width_type=q:width=0.7:gain=-6",
        ),
    }
    return presets[mode]


def _require_exact_attributes(
    enhancement: AudioEnhancement, expected: set[str]
) -> None:
    actual = set(enhancement.attributes)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise AudioEnhancementValidationError(
            f"{enhancement.kind} is missing attributes: {', '.join(sorted(missing))}"
        )
    if extra:
        raise AudioEnhancementValidationError(
            f"{enhancement.kind} has unknown attributes: {', '.join(sorted(extra))}"
        )


def _percentage(raw: str, path: str) -> float:
    if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
        raise AudioEnhancementValidationError(
            f"{path} must be a plain finite number from 0 through 100"
        )
    try:
        value = float(raw)
    except ValueError as error:
        raise AudioEnhancementValidationError(
            f"{path} must be a plain finite number from 0 through 100"
        ) from error
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise AudioEnhancementValidationError(
            f"{path} must be between 0 and 100 inclusive"
        )
    return value


def _required_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise VoiceIsolationModelError(f"voice model {name} must be a non-empty string")
    return value


def _required_sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise VoiceIsolationModelError(
            "voice model sha256 must be 64 lowercase hexadecimal characters"
        )
    return value


def _escape_filter_value(path: Path) -> str:
    rendered = str(path)
    for character in ("\\", "'", ":", ",", ";", "[", "]"):
        rendered = rendered.replace(character, "\\" + character)
    return rendered


def _filter_name(filter_text: str) -> str:
    name = filter_text.split("=", 1)[0]
    if not name or not name.replace("_", "").isalnum():
        raise AudioEnhancementValidationError(
            f"invalid generated FFmpeg filter name {name!r}"
        )
    return name


def _number_text(value: float) -> str:
    if not math.isfinite(value):
        raise AudioEnhancementValidationError("generated FFmpeg value must be finite")
    return format(value, ".12g")


__all__ = [
    "AudioEnhancementError",
    "AudioEnhancementFinding",
    "AudioEnhancementPlan",
    "AudioEnhancementStep",
    "AudioEnhancementValidationError",
    "FrozenVoiceIsolationModel",
    "UnsupportedAudioEnhancementError",
    "VoiceIsolationModelError",
    "build_audio_enhancement_plan",
    "load_frozen_voice_isolation_model",
    "required_stock_filters",
]
