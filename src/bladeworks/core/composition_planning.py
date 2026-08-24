"""Freeze semantic composition together with machine-local render bindings.

Architecture map
================

``CompositionPlan``
    Machine-independent video meaning and the sole semantic hash.

CPU planning decisions
    -> exact local source paths and probes
    -> every physical primary/transition/raster input
    -> typed audio plan and probed audio assets
    -> exact video/audio output contract
    -> ``CompositionPlanningResult``.

Backend preparation
    Consumes the frozen result without rediscovering source-to-input mappings,
    parsing an FFmpeg graph, or changing the CompositionPlan semantic hash.

Important invariants
--------------------

* Every semantic decoder/raster has exactly one runtime source record.
* A decoder has exactly one primary input and may have explicit physical
  transition branches. A raster has exactly one raster input and no decoder.
* Every transition branch names a real transition and a source reachable from
  one of that transition's composed participant layers.
* Physical input IDs and indices are globally unique.
* Audio remains an independent typed plan. Silence is explicit and never
  inferred from a missing asset binding.
* ``semantic_sha256`` is exactly ``CompositionPlan.manifest_sha256``. Paths,
  input indices, decoder seeks, audio bindings, and output packaging affect a
  separate ``runtime_fingerprint`` only.

Why this exists
---------------

The CPU source loop already knows the exact source path, decoder seek, and
transition branch for every FFmpeg input, but those facts do not belong in the
machine-independent CompositionPlan. This module preserves them beside the
plan so CPU and Vulkan emitters can consume one authoritative planning result
without positional assumptions or semantic duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

from .audio_execution import AudioAssetBinding
from .audio_ir import AudioLayout, AudioRenderPlan
from .composition_ir import (
    CompositionPlan,
    DecoderBinding,
    DecoderSourcePlan,
    FrameContract,
    RasterSourcePlan,
    RuntimeSourceBinding,
)
from .retime_execution import VideoFrameOwnership


VideoInputRole = Literal["primary", "raster", "transition_branch"]
AudioRuntimeMode = Literal["render", "silence"]


class CompositionPlanningError(ValueError):
    """A semantic plan and its physical runtime bindings disagree."""


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompositionPlanningError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CompositionPlanningError(f"{field_name} must be a positive integer")
    return value


def _canonical_value(value: Any) -> Any:
    """Return a deterministic JSON value for the runtime-only fingerprint.

    Main callers:
    - ``CompositionPlanningResult.runtime_manifest``.

    Why this exists:
    The audio and source-binding records are already frozen dataclasses, but
    their nested Fractions, Paths, enums, and mappings are not directly JSON
    serializable. Keeping this conversion local avoids importing model-level
    serialization and therefore keeps the planning boundary acyclic.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if is_dataclass(value):
        return {
            item.name: _canonical_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonical_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    raise CompositionPlanningError(
        f"runtime fingerprint cannot serialize {type(value).__name__}"
    )


@dataclass(frozen=True)
class VideoInputBinding:
    """One physical video input selected by the authoritative source loop.

    ``primary`` bindings always carry decoder execution. A decoder transition
    branch additionally owns its exact first sampled source frame because it
    may begin at a different transition handle. Raster sources use one
    ``raster`` primary and may have physical ``transition_branch`` duplicates;
    both raster shapes carry neither decoder state nor source-frame ownership.

    Main callers:
    - The future CPU source-loop integration.
    - Backend emitters allocating their final argv inputs.
    """

    binding_id: str
    source_id: str
    role: VideoInputRole
    transition_id: str | None
    input_index: int
    decoder: DecoderBinding | None
    branch_ownership: VideoFrameOwnership | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "binding_id", _identifier(self.binding_id, field_name="binding_id")
        )
        object.__setattr__(
            self, "source_id", _identifier(self.source_id, field_name="source_id")
        )
        if self.role not in {"primary", "raster", "transition_branch"}:
            raise CompositionPlanningError(f"unknown video input role {self.role!r}")
        if isinstance(self.input_index, bool) or not isinstance(self.input_index, int):
            raise CompositionPlanningError("input_index must be an integer")
        if self.input_index < 0:
            raise CompositionPlanningError("input_index cannot be negative")
        if self.role == "raster":
            if self.transition_id is not None:
                raise CompositionPlanningError(
                    "raster input cannot reference a transition"
                )
            if self.decoder is not None or self.branch_ownership is not None:
                raise CompositionPlanningError(
                    "raster input cannot carry decoder or frame ownership"
                )
            return
        if self.role == "primary":
            if not isinstance(self.decoder, DecoderBinding):
                raise CompositionPlanningError(
                    "primary input requires a DecoderBinding"
                )
            if self.decoder.decoder_id != self.source_id:
                raise CompositionPlanningError(
                    "video input decoder references another semantic source"
                )
            if self.decoder.input_index != self.input_index:
                raise CompositionPlanningError(
                    "video input index differs from its decoder binding"
                )
            if self.transition_id is not None or self.branch_ownership is not None:
                raise CompositionPlanningError(
                    "primary input cannot carry transition branch state"
                )
            return
        object.__setattr__(
            self,
            "transition_id",
            _identifier(self.transition_id, field_name="transition_id"),
        )
        if self.decoder is not None and not isinstance(self.decoder, DecoderBinding):
            raise CompositionPlanningError(
                "transition branch decoder must be DecoderBinding or None"
            )
        decoder_state = isinstance(self.decoder, DecoderBinding)
        ownership_state = self.branch_ownership is not None
        if decoder_state != ownership_state:
            raise CompositionPlanningError(
                "transition branch decoder and ownership must be present together"
            )
        if self.decoder is not None:
            if self.decoder.decoder_id != self.source_id:
                raise CompositionPlanningError(
                    "video input decoder references another semantic source"
                )
            if self.decoder.input_index != self.input_index:
                raise CompositionPlanningError(
                    "video input index differs from its decoder binding"
                )
            if not isinstance(self.branch_ownership, VideoFrameOwnership):
                raise CompositionPlanningError(
                    "decoder transition branch requires exact VideoFrameOwnership"
                )


@dataclass(frozen=True)
class CompositionSourceRuntime:
    """One semantic source plus all physical inputs opened for that source."""

    source: RuntimeSourceBinding
    bindings: tuple[VideoInputBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, RuntimeSourceBinding):
            raise CompositionPlanningError(
                "source runtime requires RuntimeSourceBinding"
            )
        object.__setattr__(self, "bindings", tuple(self.bindings))
        if not self.bindings:
            raise CompositionPlanningError("source runtime requires a physical input")
        if any(not isinstance(item, VideoInputBinding) for item in self.bindings):
            raise CompositionPlanningError(
                "source runtime bindings must be VideoInputBinding records"
            )
        if any(item.source_id != self.source.source_id for item in self.bindings):
            raise CompositionPlanningError(
                "source runtime contains a binding for another semantic source"
            )
        binding_ids = tuple(item.binding_id for item in self.bindings)
        if len(binding_ids) != len(set(binding_ids)):
            raise CompositionPlanningError(
                "source runtime binding IDs must be unique"
            )


@dataclass(frozen=True)
class CompositionAudioRuntime:
    """Independent typed audio semantics, probes, and exact output meaning.

    ``render`` requires one complete AudioRenderPlan and exact bindings for
    every audible asset. ``silence`` is an explicit output decision; it may
    retain a plan containing only inactive items, but never accepts an audible
    item or a machine-local asset binding.
    """

    plan: AudioRenderPlan | None
    asset_bindings: tuple[AudioAssetBinding, ...]
    mode: AudioRuntimeMode
    output_sample_rate: int
    output_layout: AudioLayout
    output_duration: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_bindings", tuple(self.asset_bindings))
        if self.mode not in {"render", "silence"}:
            raise CompositionPlanningError(f"unknown audio mode {self.mode!r}")
        _positive_integer(self.output_sample_rate, field_name="audio sample rate")
        if self.output_layout not in {"mono", "stereo", "surround"}:
            raise CompositionPlanningError(
                f"unknown audio output layout {self.output_layout!r}"
            )
        if not isinstance(self.output_duration, Fraction) or self.output_duration <= 0:
            raise CompositionPlanningError(
                "audio output duration must be a positive Fraction"
            )
        if any(not isinstance(item, AudioAssetBinding) for item in self.asset_bindings):
            raise CompositionPlanningError(
                "audio asset bindings must be AudioAssetBinding records"
            )
        asset_ids = tuple(item.asset_id for item in self.asset_bindings)
        if len(asset_ids) != len(set(asset_ids)):
            raise CompositionPlanningError("audio asset IDs must be unique")
        if any(not item.path.is_absolute() for item in self.asset_bindings):
            raise CompositionPlanningError(
                "audio asset binding paths must be absolute"
            )

        if self.mode == "render" and not isinstance(self.plan, AudioRenderPlan):
            raise CompositionPlanningError("render audio mode requires AudioRenderPlan")
        if self.plan is not None and not isinstance(self.plan, AudioRenderPlan):
            raise CompositionPlanningError(
                "audio plan must be AudioRenderPlan or None"
            )
        audible_asset_ids = (
            {item.asset_id for item in self.plan.items if item.audible}
            if self.plan is not None
            else set()
        )
        if self.mode == "silence":
            if audible_asset_ids:
                raise CompositionPlanningError(
                    "silence mode cannot contain audible audio items"
                )
            if self.asset_bindings:
                raise CompositionPlanningError(
                    "silence mode cannot carry audio asset bindings"
                )
        elif set(asset_ids) != audible_asset_ids:
            raise CompositionPlanningError(
                "audio bindings do not exactly cover audible asset IDs"
            )
        if self.plan is not None:
            if self.plan.sample_rate != self.output_sample_rate:
                raise CompositionPlanningError(
                    "audio plan and output sample rates differ"
                )
            if self.plan.layout != self.output_layout:
                raise CompositionPlanningError(
                    "audio plan and output layouts differ"
                )
            if self.plan.sequence_duration != self.output_duration:
                raise CompositionPlanningError(
                    "audio plan and output durations differ"
                )


@dataclass(frozen=True)
class CompositionOutputContract:
    """Exact final video frame ownership and independent audio output shape."""

    root_frame_contract: FrameContract
    frame_count: int
    audio_sample_rate: int
    audio_layout: AudioLayout
    audio_duration: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.root_frame_contract, FrameContract):
            raise CompositionPlanningError(
                "output root_frame_contract must be FrameContract"
            )
        _positive_integer(self.frame_count, field_name="output frame_count")
        _positive_integer(
            self.audio_sample_rate, field_name="output audio_sample_rate"
        )
        if self.audio_layout not in {"mono", "stereo", "surround"}:
            raise CompositionPlanningError(
                f"unknown output audio layout {self.audio_layout!r}"
            )
        if not isinstance(self.audio_duration, Fraction) or self.audio_duration <= 0:
            raise CompositionPlanningError(
                "output audio_duration must be a positive Fraction"
            )


@dataclass(frozen=True)
class CompositionPlanningResult:
    """Complete semantic and physical handoff ready for backend preparation.

    Main callers:
    - The future CPU planner after its authoritative source/audio loops.
    - Strict Vulkan runtime preparation and the unchanged CPU emitter.

    Why this exists:
    CompositionPlan must remain portable and hash-stable, while an executable
    render still needs local files, decoder optimization choices, duplicate
    transition inputs, probed audio assets, and final packaging contracts.
    This object binds those two lifetimes without merging their identities.
    """

    composition_plan: CompositionPlan
    sources: tuple[CompositionSourceRuntime, ...]
    audio: CompositionAudioRuntime
    output: CompositionOutputContract

    def __post_init__(self) -> None:
        if not isinstance(self.composition_plan, CompositionPlan):
            raise CompositionPlanningError(
                "planning result requires CompositionPlan"
            )
        object.__setattr__(self, "sources", tuple(self.sources))
        if any(not isinstance(item, CompositionSourceRuntime) for item in self.sources):
            raise CompositionPlanningError(
                "planning sources must be CompositionSourceRuntime records"
            )
        if not isinstance(self.audio, CompositionAudioRuntime):
            raise CompositionPlanningError(
                "planning audio must be CompositionAudioRuntime"
            )
        if not isinstance(self.output, CompositionOutputContract):
            raise CompositionPlanningError(
                "planning output must be CompositionOutputContract"
            )
        self._validate_sources()
        self._validate_output()

    @property
    def semantic_sha256(self) -> str:
        """Return the unchanged machine-independent CompositionPlan hash."""

        return self.composition_plan.manifest_sha256

    def runtime_manifest(self) -> dict[str, Any]:
        """Return canonical machine-local execution facts for diagnostics."""

        source_records = sorted(
            self.sources, key=lambda item: item.source.source_id
        )
        return {
            "schema": "bladeworks.composition-planning-runtime.v1",
            "composition_plan_sha256": self.semantic_sha256,
            "sources": [
                {
                    "source": _canonical_value(item.source),
                    "bindings": [
                        _canonical_value(binding)
                        for binding in sorted(
                            item.bindings,
                            key=lambda binding: (
                                binding.input_index,
                                binding.binding_id,
                            ),
                        )
                    ],
                }
                for item in source_records
            ],
            "audio": {
                "plan": _canonical_value(self.audio.plan),
                "asset_bindings": [
                    _canonical_value(binding)
                    for binding in sorted(
                        self.audio.asset_bindings,
                        key=lambda binding: binding.asset_id,
                    )
                ],
                "mode": self.audio.mode,
                "output_sample_rate": self.audio.output_sample_rate,
                "output_layout": self.audio.output_layout,
                "output_duration": _canonical_value(
                    self.audio.output_duration
                ),
            },
            "output": _canonical_value(self.output),
        }

    @property
    def runtime_fingerprint(self) -> str:
        """Hash paths, indices, decoder choices, audio, and output packaging."""

        payload = json.dumps(
            self.runtime_manifest(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _validate_sources(self) -> None:
        """Prove exact plan coverage and every physical decoder/raster shape."""

        plan = self.composition_plan
        source_plans = {
            source.identity.source_id: source
            for source in plan.decoders + plan.rasters
        }
        runtime_ids = tuple(item.source.source_id for item in self.sources)
        if len(runtime_ids) != len(set(runtime_ids)):
            raise CompositionPlanningError("runtime source IDs must be unique")
        if set(runtime_ids) != set(source_plans):
            raise CompositionPlanningError(
                "runtime sources do not exactly cover CompositionPlan sources"
            )
        binding_ids = tuple(
            binding.binding_id
            for source in self.sources
            for binding in source.bindings
        )
        if len(binding_ids) != len(set(binding_ids)):
            raise CompositionPlanningError(
                "video binding IDs must be globally unique"
            )
        input_indices = tuple(
            binding.input_index
            for source in self.sources
            for binding in source.bindings
        )
        if len(input_indices) != len(set(input_indices)):
            raise CompositionPlanningError(
                "video input indices must be globally unique"
            )

        transition_sources = self._transition_source_ids()
        for runtime in self.sources:
            source_plan = source_plans[runtime.source.source_id]
            try:
                runtime.source.validate_against(source_plan)
            except (TypeError, ValueError) as error:
                raise CompositionPlanningError(str(error)) from error
            roles = tuple(binding.role for binding in runtime.bindings)
            if isinstance(source_plan, RasterSourcePlan):
                if roles.count("raster") != 1 or "primary" in roles:
                    raise CompositionPlanningError(
                        "raster source requires exactly one raster primary and no "
                        "decoder primary"
                    )
                for binding in runtime.bindings:
                    if binding.decoder is not None or binding.branch_ownership is not None:
                        raise CompositionPlanningError(
                            "raster source inputs cannot carry decoder frame state"
                        )
                    if binding.role == "transition_branch":
                        assert binding.transition_id is not None
                        if binding.transition_id not in transition_sources:
                            raise CompositionPlanningError(
                                "transition branch references an unknown transition"
                            )
                        if source_plan.identity.source_id not in transition_sources[
                            binding.transition_id
                        ]:
                            raise CompositionPlanningError(
                                "transition branch source is not a transition participant"
                            )
                continue
            if not isinstance(source_plan, DecoderSourcePlan):
                raise CompositionPlanningError("unknown CompositionPlan source type")
            if roles.count("primary") != 1 or "raster" in roles:
                raise CompositionPlanningError(
                    "decoder source requires exactly one primary and no raster input"
                )
            for binding in runtime.bindings:
                if binding.role == "primary":
                    assert binding.decoder is not None
                    try:
                        binding.decoder.validate_against(
                            source_plan,
                            clip_id=source_plan.identity.clip_id,
                        )
                    except (TypeError, ValueError) as error:
                        raise CompositionPlanningError(str(error)) from error
                    continue
                if binding.decoder is None or binding.branch_ownership is None:
                    raise CompositionPlanningError(
                        "decoder transition branch requires decoder frame state"
                    )
                assert binding.transition_id is not None
                if binding.transition_id not in transition_sources:
                    raise CompositionPlanningError(
                        "transition branch references an unknown transition"
                    )
                if source_plan.identity.source_id not in transition_sources[
                    binding.transition_id
                ]:
                    raise CompositionPlanningError(
                        "transition branch source is not a transition participant"
                    )
                assert binding.branch_ownership is not None
                self._validate_branch(source_plan, binding)

    def _transition_source_ids(self) -> dict[str, frozenset[str]]:
        """Resolve transition participant layers to descendant semantic sources."""

        plan = self.composition_plan
        scope_by_id = {scope.scope_id: scope for scope in plan.scopes}
        layer_by_id = {
            layer.layer_id: layer for scope in plan.scopes for layer in scope.layers
        }
        cached_scope_sources: dict[str, frozenset[str]] = {}

        def source_ids_for_scope(scope_id: str) -> frozenset[str]:
            cached = cached_scope_sources.get(scope_id)
            if cached is not None:
                return cached
            source_ids: set[str] = set()
            for layer in scope_by_id[scope_id].layers:
                if layer.source.kind in {"decoder", "still", "runtime_raster"}:
                    assert layer.source.ref is not None
                    source_ids.add(layer.source.ref)
                elif layer.source.kind == "module":
                    assert layer.source.ref is not None
                    source_ids.update(source_ids_for_scope(layer.source.ref))
            result = frozenset(source_ids)
            cached_scope_sources[scope_id] = result
            return result

        def source_ids_for_layer(layer_id: str) -> frozenset[str]:
            layer = layer_by_id[layer_id]
            if layer.source.kind in {"decoder", "still", "runtime_raster"}:
                assert layer.source.ref is not None
                return frozenset({layer.source.ref})
            if layer.source.kind == "module":
                assert layer.source.ref is not None
                return source_ids_for_scope(layer.source.ref)
            return frozenset()

        result: dict[str, frozenset[str]] = {}
        for scope in plan.scopes:
            for transition in scope.transitions:
                participant_ids = (
                    transition.outgoing.composed_sources
                    + transition.incoming.composed_sources
                )
                result[transition.transition_id] = frozenset(
                    source_id
                    for layer_id in participant_ids
                    for source_id in source_ids_for_layer(layer_id)
                )
        return result

    @staticmethod
    def _validate_branch(
        source: DecoderSourcePlan,
        binding: VideoInputBinding,
    ) -> None:
        """Validate a branch against its own ownership, not the primary sample."""

        ownership = binding.branch_ownership
        decoder = binding.decoder
        assert ownership is not None and decoder is not None
        decode_window = source.decode_window
        if (
            ownership.frame_duration != decode_window.frame_duration
            or ownership.frame_grid_origin != decode_window.frame_grid_origin
        ):
            raise CompositionPlanningError(
                "transition branch and decoder source use different frame grids"
            )
        if not (
            decode_window.decode_start
            <= ownership.source_frame_start
            < decode_window.decode_end
        ):
            raise CompositionPlanningError(
                "transition branch ownership lies outside decoder coverage"
            )
        expected_seek = (
            ownership.frame_grid_origin
            + decoder.decoder_start_frame * ownership.frame_duration
        )
        if decoder.decoder_seek != expected_seek:
            raise CompositionPlanningError(
                "transition branch decoder seek is not frame aligned"
            )
        if (
            ownership.source_start_frame
            != decoder.decoder_start_frame + decoder.filter_start_frame
        ):
            raise CompositionPlanningError(
                "transition branch frame differs from decoder plus filter origins"
            )

    def _validate_output(self) -> None:
        """Prove root video and independent audio output contracts are identical."""

        plan = self.composition_plan
        root = next(scope for scope in plan.scopes if scope.scope_id == plan.root_scope_id)
        if self.output.root_frame_contract != root.output_contract:
            raise CompositionPlanningError(
                "planning output root frame contract differs from CompositionPlan"
            )
        if self.output.frame_count != root.output_contract.clock.frame_count:
            raise CompositionPlanningError(
                "planning output frame count differs from the root frame clock"
            )
        if self.output.audio_duration != plan.project_clock.duration:
            raise CompositionPlanningError(
                "audio output duration differs from the project duration"
            )
        audio_values = (
            self.audio.output_sample_rate,
            self.audio.output_layout,
            self.audio.output_duration,
        )
        output_values = (
            self.output.audio_sample_rate,
            self.output.audio_layout,
            self.output.audio_duration,
        )
        if audio_values != output_values:
            raise CompositionPlanningError(
                "audio runtime and final output contracts differ"
            )
        if self.audio.plan is not None:
            if self.audio.plan.source_sha256 != plan.document_source_sha256:
                raise CompositionPlanningError(
                    "audio and composition plans belong to different documents"
                )


__all__ = [
    "AudioRuntimeMode",
    "CompositionAudioRuntime",
    "CompositionOutputContract",
    "CompositionPlanningError",
    "CompositionPlanningResult",
    "CompositionSourceRuntime",
    "VideoInputBinding",
    "VideoInputRole",
]
