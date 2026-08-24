"""Build and execute a fail-closed stock-FFmpeg audio graph.

Architecture map
================

``AudioRenderPlan``
    Independent timeline items produced by :mod:`audio_ir`.

``probe_audio_asset`` / caller-owned bindings
    Resolve an FCP asset ID to one local file and enumerate every audio stream.

``build_audio_execution_plan``
    Resolve streams and channels, then emit this fixed operation order for
    every audible item::

        source trim -> channel route -> gain/keyframes/fades -> mutes -> pan
        -> piecewise retime -> resample -> sample-exact timeline delay

    Descendants of a file, compound, or multicam source instance are mixed
    into one local pad first when needed. The ordinary retime kernel changes
    that pad once, clip controls run once, and the result returns to its
    parent/root mix.

    The item outputs are mixed with ``amix normalize=0`` over an explicit
    sequence-length silence bed.  No limiter or loudness normalizer is added.

``AudioExecutionPlan.command``
    Produces an ordinary FFmpeg invocation for isolated validation or future
    central-executor integration.

Important invariants
--------------------

* Source stream IDs are resolved against a complete, deterministic probe.
  Missing assets, streams, and channels are errors.
* An omitted ``srcCh`` is resolved late to every channel in the chosen stream.
  An omitted ``outCh`` follows the documented default routing matrix below.
* Control layers remain ordered.  They are never collapsed into one guessed
  value, and opaque/unimplemented enhancements stop graph construction.
* Audio retiming reuses :mod:`retime_execution`; reverse and pitch behavior are
  explicit, and freeze segments emit calibrated silence matching Final Cut.
* A clip-instance timeMap owns its completed source. It is never copied onto each
  descendant source, which would retime angle switches and mixes separately.
* Video transitions are absent from this API.  They cannot accidentally add
  an audio crossfade.

Main callers:
- The central FFmpeg invocation builder.
- Experimental core renderer tests.

Why this exists:
The legacy FFmpeg builder schedules audio from video clips and assumes the
first stream.  That makes J/L edits, split multicam audio, channel routing, and
role controls impossible to represent faithfully.  This module validates the
replacement engine without putting experimental behavior in production CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence, Union

from .audio_enhancements import (
    AudioEnhancementError,
    build_audio_enhancement_plan,
)
from .audio_ir import (
    AnimatedAudioScalar,
    AudioControlLayer,
    AudioFade,
    AudioMuteRange,
    AudioPanner,
    AudioRenderPlan,
    AudioSourceInstance,
    RenderAudioItem,
)
from .errors import RenderCapabilityError
from .render_sources import rebase_source_retime
from .retime import RetimeMap, RetimePoint
from .retime_execution import (
    AudioFreezeBehaviorBlocked,
    AudioFreezePolicy,
    build_audio_filtergraph_segments,
    build_retime_execution_plan,
    required_stock_filters as required_retime_filters,
)


class AudioExecutionError(ValueError):
    """Base error for an invalid or unavailable audio execution plan."""


class AudioAssetBindingError(AudioExecutionError):
    """An asset has no binding or its binding is ambiguous/invalid."""


class AudioStreamResolutionError(AudioExecutionError):
    """An item requests a stream or channel absent from the media probe."""


class UnsupportedAudioControlError(AudioExecutionError):
    """A preserved control has no honest stock-FFmpeg implementation yet."""


class AudioFreezeCalibrationError(AudioExecutionError):
    """A retime contains freeze audio whose Final Cut behavior is uncalibrated."""


_FCP_FREEZE_SILENCE_CALIBRATION = "fcp-12.3-ui-hold-silence-20260812"


class MissingFFmpegAudioCapability(AudioExecutionError):
    """The selected FFmpeg executable lacks a filter required by the plan."""


_OUTPUT_LAYOUTS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "mono": ("mono", ("FC",)),
    "stereo": ("stereo", ("FL", "FR")),
    "surround": ("5.1", ("FL", "FR", "FC", "LFE", "SL", "SR")),
}
_FCP_OUTPUT_TO_FFMPEG = {
    "L": "FL",
    "R": "FR",
    "C": "FC",
    "LFE": "LFE",
    "Ls": "SL",
    "Rs": "SR",
}
_KNOWN_SOURCE_LAYOUTS: Mapping[str, tuple[str, ...]] = {
    "mono": ("FC",),
    "stereo": ("FL", "FR"),
    "2.1": ("FL", "FR", "LFE"),
    "3.0": ("FL", "FR", "FC"),
    "4.0": ("FL", "FR", "FC", "BC"),
    "5.0": ("FL", "FR", "FC", "BL", "BR"),
    "5.0(side)": ("FL", "FR", "FC", "SL", "SR"),
    "5.1": ("FL", "FR", "FC", "LFE", "BL", "BR"),
    "5.1(side)": ("FL", "FR", "FC", "LFE", "SL", "SR"),
}
_FILTER_LABEL_RE = re.compile(r"^[A-Za-z0-9_]+$")


def reject_unsupported_output_layout(layout: str) -> None:
    """Loudly reject a sequence audio output layout the renderer cannot deliver.

    What it does:
    If ``layout`` is ``"surround"`` (Final Cut's 5.1 output request) raise a
    ``RenderCapabilityError`` naming the construct. ``"mono"`` and ``"stereo"``
    pass through untouched; any other value is left for the caller's own layout
    validation so this guard only ever speaks about the one unsupported case.

    Why this exists:
    Final Cut's ``sequence/@audioLayout="surround"`` asks for a 5.1 delivery. The
    audio graph would build a stereo->5.1 ``pan`` upmix whose surround channels
    are named ``Ls``/``Rs`` -> ``SL``/``SR``, but FFmpeg 8's ``5.1`` channel
    layout uses ``BL``/``BR`` -- so libav rejects the graph DEEP in execution with
    a cryptic ``ValueError [Errno 22]`` (ffmpeg exit 234, "Channel SL does not
    exist in the chosen layout"). We do NOT implement the 5.1 channel mapping;
    instead we reject surround output UP FRONT, at plan time, as an explicit
    capability gap -- the same loud-reject discipline used for other unsupported
    constructs (e.g. the shear/skew transform reject). This is a clean capability
    rejection, not a crash, and it NEVER silently downmixes to stereo.

    Main callers:
    - ``build_audio_execution_plan`` (before any audio graph node is built).
    - ``tensor.audio_delivery`` resolvers (before any frame is rendered).
    """
    if layout == "surround":
        raise RenderCapabilityError(
            "surround (5.1) audio output is not supported: the renderer delivers "
            "only mono and stereo audio layouts. Set the sequence audioLayout to "
            "stereo (or mono) to render this project."
        )


@dataclass(frozen=True)
class AudioStreamBinding:
    """One probed audio stream addressable by FCP's one-based ``srcID``."""

    source_stream_id: str
    audio_ordinal: int
    channels: int
    sample_rate: int
    channel_layout: str | None = None

    def __post_init__(self) -> None:
        if not self.source_stream_id or not self.source_stream_id.strip():
            raise AudioAssetBindingError("audio stream ID must not be empty")
        if isinstance(self.audio_ordinal, bool) or self.audio_ordinal < 0:
            raise AudioAssetBindingError("audio stream ordinal must be non-negative")
        if isinstance(self.channels, bool) or self.channels <= 0:
            raise AudioAssetBindingError("audio stream channel count must be positive")
        if isinstance(self.sample_rate, bool) or self.sample_rate <= 0:
            raise AudioAssetBindingError("audio stream sample rate must be positive")


@dataclass(frozen=True)
class AudioAssetBinding:
    """One local asset file plus the complete audio-stream probe."""

    asset_id: str
    path: Path
    streams: tuple[AudioStreamBinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "streams", tuple(self.streams))
        if not self.asset_id:
            raise AudioAssetBindingError("asset binding requires an asset ID")
        if not self.streams:
            raise AudioAssetBindingError(
                f"asset {self.asset_id!r} has no probed audio streams"
            )
        ids = [stream.source_stream_id for stream in self.streams]
        ordinals = [stream.audio_ordinal for stream in self.streams]
        if len(set(ids)) != len(ids):
            raise AudioAssetBindingError(
                f"asset {self.asset_id!r} has duplicate source stream IDs"
            )
        if len(set(ordinals)) != len(ordinals):
            raise AudioAssetBindingError(
                f"asset {self.asset_id!r} has duplicate audio ordinals"
            )

    def stream(self, source_stream_id: str) -> AudioStreamBinding:
        matches = tuple(
            stream
            for stream in self.streams
            if stream.source_stream_id == source_stream_id
        )
        if len(matches) != 1:
            known = ", ".join(stream.source_stream_id for stream in self.streams)
            raise AudioStreamResolutionError(
                f"asset {self.asset_id!r} has no unique audio stream "
                f"{source_stream_id!r}; known streams: {known}"
            )
        return matches[0]


@dataclass(frozen=True)
class AudioItemExecution:
    """Resolved, deterministic execution details for one audible item."""

    item_id: str
    asset_id: str
    input_index: int
    audio_ordinal: int
    source_channels: tuple[int, ...]
    output_routes: tuple[tuple[int, str, Fraction], ...]
    start_sample: int
    output_label: str
    retimed: bool
    enhancement_plans: tuple[Mapping[str, Any], ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "asset_id": self.asset_id,
            "input_index": self.input_index,
            "audio_ordinal": self.audio_ordinal,
            "source_channels": list(self.source_channels),
            "output_routes": [
                {
                    "source_channel": source,
                    "output_channel": output,
                    "coefficient": _fraction_text(coefficient),
                }
                for source, output, coefficient in self.output_routes
            ],
            "start_sample": self.start_sample,
            "output_label": self.output_label,
            "retimed": self.retimed,
            "enhancements": [dict(plan) for plan in self.enhancement_plans],
        }


@dataclass(frozen=True)
class AudioSourceExecution:
    """Audit record for one virtual source consumed by a clip instance."""

    path: str
    source_id: str
    input_count: int
    output_label: str
    start_sample: int
    source_origin: Fraction
    source_duration: Fraction

    def manifest(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source_id": self.source_id,
            "input_count": self.input_count,
            "output_label": self.output_label,
            "start_sample": self.start_sample,
            "source_origin": _fraction_text(self.source_origin),
            "source_duration": _fraction_text(self.source_duration),
        }


@dataclass(frozen=True)
class FFmpegAudioCapabilityReport:
    executable: str
    version_line: str
    required_filters: tuple[str, ...]
    available_filters: tuple[str, ...]
    missing_filters: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.missing_filters

    def require_supported(self) -> None:
        if self.missing_filters:
            raise MissingFFmpegAudioCapability(
                "FFmpeg is missing audio filters: " + ", ".join(self.missing_filters)
            )


@dataclass(frozen=True)
class AudioExecutionPlan:
    """A complete stock-FFmpeg graph and audit manifest.

    Main callers:
    - ``build_audio_execution_plan`` returns this to the isolated runner.
    - The future central executor can merge ``inputs`` and ``filter_complex``
      into the combined video/audio invocation.
    """

    schema_version: int
    source_sha256: str
    sample_rate: int
    layout: str
    ffmpeg_layout: str
    sequence_duration: Fraction
    inputs: tuple[Path, ...]
    input_asset_ids: tuple[str, ...]
    filter_complex: str
    output_label: str
    items: tuple[AudioItemExecution, ...]
    required_filters: tuple[str, ...]
    source_instances: tuple[AudioSourceExecution, ...] = ()
    # Structured node/edge form of ``filter_complex`` (serialises back to it
    # byte-for-byte).  The PyAV port (plan chunk 2) builds the libav graph from
    # this; the golden gate proves ``serialize_segments(graph_segments) ==
    # filter_complex``.  Defaulted so existing direct constructors still work.
    graph_segments: tuple[AudioGraphSegmentLike, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "fcpxml_audio_execution.v1",
            "source_sha256": self.source_sha256,
            "sample_rate": self.sample_rate,
            "layout": self.layout,
            "ffmpeg_layout": self.ffmpeg_layout,
            "sequence_duration": _fraction_text(self.sequence_duration),
            "inputs": [
                {"asset_id": asset_id, "path": str(path)}
                for asset_id, path in zip(self.input_asset_ids, self.inputs)
            ],
            "output_label": self.output_label,
            "required_filters": list(self.required_filters),
            "items": [item.manifest() for item in self.items],
            "source_instances": [
                instance.manifest() for instance in self.source_instances
            ],
            "mix": {
                "filter": "amix",
                "normalize": False,
                "hidden_limiter": False,
                "hidden_loudness_normalization": False,
                "video_transition_audio": False,
            },
        }

    @property
    def manifest_sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def command(
        self,
        output_path: str | Path,
        *,
        ffmpeg_path: str | Path = "ffmpeg",
        codec: str = "pcm_s24le",
        overwrite: bool = True,
    ) -> tuple[str, ...]:
        """Return an argv tuple; no shell parsing or caller-provided filters."""

        command = [str(ffmpeg_path), "-hide_banner", "-nostdin"]
        if overwrite:
            command.append("-y")
        else:
            command.append("-n")
        for path in self.inputs:
            command.extend(("-i", str(path)))
        command.extend(
            (
                "-filter_complex",
                self.filter_complex,
                "-map",
                f"[{self.output_label}]",
                "-vn",
                "-c:a",
                codec,
                "-ar",
                str(self.sample_rate),
                "-channel_layout",
                self.ffmpeg_layout,
                str(output_path),
            )
        )
        return tuple(command)


def probe_audio_asset(
    asset_id: str,
    path: str | Path,
    *,
    ffprobe_path: str | Path = "ffprobe",
) -> AudioAssetBinding:
    """Enumerate every audio stream in one asset with stock ``ffprobe``.

    Stream IDs are deliberately one-based audio ordinals (``"1"``, ``"2"``)
    because that is the stable form consumed by ``AudioRenderPlan``.  The
    container's global stream index is not used as an FCP stream ID.

    Main callers:
    - Asset-binding setup before ``build_audio_execution_plan``.
    """

    media_path = Path(path)
    try:
        completed = subprocess.run(
            (
                str(ffprobe_path),
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index,channels,sample_rate,channel_layout",
                "-of",
                "json",
                str(media_path),
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise AudioAssetBindingError(
            f"could not probe audio asset {asset_id!r} at {media_path}: {error}"
        ) from error
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list) or not raw_streams:
        raise AudioAssetBindingError(f"asset {asset_id!r} has no audio streams")
    streams: list[AudioStreamBinding] = []
    for ordinal, raw in enumerate(raw_streams):
        if not isinstance(raw, dict):
            raise AudioAssetBindingError(
                f"asset {asset_id!r} returned malformed audio stream metadata"
            )
        try:
            channels = int(raw["channels"])
            sample_rate = int(raw["sample_rate"])
        except (KeyError, TypeError, ValueError) as error:
            raise AudioAssetBindingError(
                f"asset {asset_id!r} stream {ordinal + 1} lacks channel/rate metadata"
            ) from error
        layout = raw.get("channel_layout")
        streams.append(
            AudioStreamBinding(
                source_stream_id=str(ordinal + 1),
                audio_ordinal=ordinal,
                channels=channels,
                sample_rate=sample_rate,
                channel_layout=str(layout) if layout else None,
            )
        )
    return AudioAssetBinding(asset_id=asset_id, path=media_path, streams=tuple(streams))


@dataclass(frozen=True)
class AudioFilterNode:
    """One FFmpeg filter as its raw ``name`` or ``name=args`` text.

    Serialisation is the identity on ``raw`` so a structured graph reproduces
    the calibrated ``filter_complex`` byte-for-byte.  ``name``/``args`` are
    derived read-only by splitting on the FIRST ``=`` only -- filter names are
    ``[a-z0-9_]+`` and never contain ``=``, so this never corrupts an argument
    that itself contains ``=`` (e.g. ``anullsrc=r=48000:cl=stereo``).  The PyAV
    port (plan chunk 2) consumes ``name``/``args`` to build one libav filter per
    node; this file only ever serialises ``raw``.
    """

    raw: str

    @property
    def name(self) -> str:
        return self.raw.split("=", 1)[0]

    @property
    def args(self) -> str:
        parts = self.raw.split("=", 1)
        return parts[1] if len(parts) == 2 else ""


@dataclass(frozen=True)
class AudioGraphSegment:
    """One FFmpeg filterchain: ``[in..] f1,f2,.. [out..]``.

    Multi-input (``amix``, ``join``) and multi-output (``asplit``,
    ``channelsplit``) chains are expressed by the label tuples; a source-only
    chain (``anullsrc``) has empty ``in_labels``.
    """

    in_labels: tuple[str, ...]
    filters: tuple[AudioFilterNode, ...]
    out_labels: tuple[str, ...]

    def serialize(self) -> str:
        return (
            "".join(f"[{label}]" for label in self.in_labels)
            + ",".join(node.raw for node in self.filters)
            + "".join(f"[{label}]" for label in self.out_labels)
        )


@dataclass(frozen=True)
class AudioRawSegment:
    """An already-serialised filtergraph segment spliced verbatim.

    Used only for retime sub-graphs produced by ``build_audio_filtergraph``,
    which are ``;``-joined and re-split here.  Chunk 2 must give that helper a
    structured accessor; until then this preserves byte-identity without parsing
    FFmpeg filterchain syntax.
    """

    text: str

    def serialize(self) -> str:
        return self.text


AudioGraphSegmentLike = Union[AudioGraphSegment, AudioRawSegment]


def serialize_segments(segments: Sequence[AudioGraphSegmentLike]) -> str:
    """Join structured segments into a stock ``filter_complex`` string.

    Main callers:
    - ``AudioExecutionPlan.filter_complex`` construction (this file).
    - ``test_audio_graph_serialize_golden`` (Stage 2a byte-identity gate).
    """

    return ";".join(segment.serialize() for segment in segments)


def _filter_nodes(filters: Sequence[str]) -> tuple[AudioFilterNode, ...]:
    return tuple(AudioFilterNode(spec) for spec in filters)


class _GraphBuilder:
    """Structured audio filtergraph accumulator + graph-pad label allocator.

    Holds an ordered list of typed ``segments`` instead of raw strings, so the
    same graph can serialise to a stock ``filter_complex`` (this file) or be
    rebuilt node-by-node in PyAV (plan chunk 2).  ``label`` still prevents
    accidental pad reuse.

    Why this exists:
    PyAV 16 exposes no ``filter_complex`` string parser, so the graph must be
    consumed as structured nodes/edges.  Building it structured at the source --
    where filter name/args and pad labels are already separate -- avoids ever
    parsing FFmpeg filterchain syntax back out of a string.
    """

    def __init__(self) -> None:
        self.segments: list[AudioGraphSegmentLike] = []
        self._counter = 0

    def label(self, stem: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_]", "_", stem)
        self._counter += 1
        return f"a_{safe}_{self._counter}"

    def add(
        self,
        in_labels: Sequence[str],
        filters: Sequence[str],
        out_labels: Sequence[str],
    ) -> None:
        """Append one structured filterchain of any input/output arity."""

        self.segments.append(
            AudioGraphSegment(
                tuple(in_labels), _filter_nodes(filters), tuple(out_labels)
            )
        )

    def add_raw(self, text: str) -> None:
        """Splice an already-serialised sub-graph segment verbatim."""

        self.segments.append(AudioRawSegment(text))

    def chain(self, source: str, filters: Sequence[str], *, stem: str) -> str:
        """Single-input/single-output labelled chain; returns the output label."""

        output = self.label(stem)
        self.add((source,), filters, (output,))
        return output

    def serialize(self) -> str:
        return serialize_segments(self.segments)


@dataclass(frozen=True)
class _AudioPad:
    """One scheduled audio pad plus the unresolved source paths owning it."""

    label: str
    ancestor_paths: tuple[str, ...]


def build_audio_execution_plan(
    audio_plan: AudioRenderPlan,
    bindings: Mapping[str, AudioAssetBinding],
    *,
    output_label: str = "audio_out",
    freeze_audio_policy: AudioFreezePolicy | None = None,
    input_index_offset: int = 0,
) -> AudioExecutionPlan:
    """Resolve an audio IR plan into one deterministic stock-FFmpeg graph.

    Main callers:
    - Experimental AUDIO-2 tests.
    - Future central executor integration.

    Why this exists:
    This is the sole point that joins logical FCP source IDs to physical media
    streams.  Keeping that decision here makes missing bindings and invalid
    channels visible before FFmpeg starts.
    """

    if not isinstance(audio_plan, AudioRenderPlan):
        raise AudioExecutionError("audio_plan must be an AudioRenderPlan")
    if audio_plan.schema_version != 2:
        raise AudioExecutionError(
            f"unsupported audio plan schema {audio_plan.schema_version!r}"
        )
    if audio_plan.layout not in _OUTPUT_LAYOUTS:
        raise AudioExecutionError(f"unsupported output layout {audio_plan.layout!r}")
    # Reject surround (5.1) output loudly here, before a single graph node is
    # built, so the render fails with a clear capability message instead of the
    # cryptic libav ``Errno 22`` this plan's 5.1 ``pan`` upmix would trigger deep
    # in execution. See ``reject_unsupported_output_layout``.
    reject_unsupported_output_layout(audio_plan.layout)
    if audio_plan.sample_rate <= 0 or audio_plan.sequence_duration <= 0:
        raise AudioExecutionError("audio sample rate and sequence duration must be positive")
    if not _FILTER_LABEL_RE.fullmatch(output_label):
        raise AudioExecutionError("output label must contain only letters, digits, or underscore")
    if (
        isinstance(input_index_offset, bool)
        or not isinstance(input_index_offset, int)
        or input_index_offset < 0
    ):
        raise AudioExecutionError("input_index_offset must be a non-negative integer")

    ffmpeg_layout, output_positions = _OUTPUT_LAYOUTS[audio_plan.layout]
    audible_items = tuple(item for item in audio_plan.items if item.audible)
    used_asset_ids = tuple(sorted({item.asset_id for item in audible_items}))
    for asset_id in used_asset_ids:
        if asset_id not in bindings:
            raise AudioAssetBindingError(
                f"audible audio item references unresolved asset {asset_id!r}"
            )
        binding = bindings[asset_id]
        if binding.asset_id != asset_id:
            raise AudioAssetBindingError(
                f"binding key {asset_id!r} contains asset {binding.asset_id!r}"
            )
    inputs = tuple(bindings[asset_id].path for asset_id in used_asset_ids)
    input_indices = {
        asset_id: input_index_offset + index
        for index, asset_id in enumerate(used_asset_ids)
    }

    resolved: list[tuple[RenderAudioItem, AudioAssetBinding, AudioStreamBinding, tuple[int, ...], tuple[tuple[int, str, Fraction], ...]]] = []
    source_use_counts: dict[tuple[int, int], int] = {}
    for item in audible_items:
        binding = bindings[item.asset_id]
        stream = binding.stream(item.source_stream_id)
        source_channels = item.source_channels or tuple(range(1, stream.channels + 1))
        invalid = tuple(channel for channel in source_channels if channel > stream.channels)
        if invalid:
            raise AudioStreamResolutionError(
                f"{item.path} requests source channels {invalid} from "
                f"{stream.channels}-channel stream {item.source_stream_id!r}"
            )
        routes = _resolve_output_routes(
            item,
            stream,
            source_channels,
            output_positions,
        )
        input_index = input_indices[item.asset_id]
        key = (input_index, stream.audio_ordinal)
        source_use_counts[key] = source_use_counts.get(key, 0) + 1
        resolved.append((item, binding, stream, source_channels, routes))

    builder = _GraphBuilder()
    source_labels: dict[tuple[int, int], list[str]] = {}
    for key in sorted(source_use_counts):
        count = source_use_counts[key]
        source = f"{key[0]}:a:{key[1]}"
        if count == 1:
            source_labels[key] = [source]
            continue
        labels = [builder.label(f"source_{key[0]}_{key[1]}") for _ in range(count)]
        builder.add((source,), (f"asplit={count}",), labels)
        source_labels[key] = labels

    source_offsets = {key: 0 for key in source_labels}
    pending_pads: list[_AudioPad] = []
    item_execution: list[AudioItemExecution] = []
    required = {"anullsrc", "atrim", "asetpts", "aresample", "amix"}
    if any(count > 1 for count in source_use_counts.values()):
        required.add("asplit")
    for item, binding, stream, source_channels, routes in resolved:
        key = (input_indices[item.asset_id], stream.audio_ordinal)
        source = source_labels[key][source_offsets[key]]
        source_offsets[key] += 1
        item_label, item_required, retimed, enhancement_manifests = _build_item_graph(
            builder,
            item,
            source,
            routes=routes,
            output_layout=audio_plan.layout,
            ffmpeg_layout=ffmpeg_layout,
            sample_rate=audio_plan.sample_rate,
            freeze_audio_policy=freeze_audio_policy
            or AudioFreezePolicy.calibrated_silence(
                _FCP_FREEZE_SILENCE_CALIBRATION
            ),
        )
        required.update(item_required)
        pending_pads.append(
            _AudioPad(label=item_label, ancestor_paths=item.ancestor_paths)
        )
        item_execution.append(
            AudioItemExecution(
                item_id=item.id,
                asset_id=item.asset_id,
                input_index=key[0],
                audio_ordinal=key[1],
                source_channels=source_channels,
                output_routes=routes,
                start_sample=_time_to_sample(item.absolute_start, audio_plan.sample_rate),
                output_label=item_label,
                retimed=retimed,
                enhancement_plans=enhancement_manifests,
            )
        )

    pending_pads, source_execution, source_required = (
        _apply_audio_source_instances(
            builder,
            pending_pads,
            audio_plan.source_instances,
            sequence_duration=audio_plan.sequence_duration,
            sample_rate=audio_plan.sample_rate,
            ffmpeg_layout=ffmpeg_layout,
            output_layout=audio_plan.layout,
            freeze_audio_policy=freeze_audio_policy
            or AudioFreezePolicy.calibrated_silence(
                _FCP_FREEZE_SILENCE_CALIBRATION
            ),
        )
    )
    required.update(source_required)
    duration = _number_text(audio_plan.sequence_duration)
    silence = builder.label("sequence_silence")
    builder.add(
        (),
        (
            f"anullsrc=r={audio_plan.sample_rate}:cl={ffmpeg_layout}",
            f"atrim=duration={duration}",
            "asetpts=PTS-STARTPTS",
        ),
        (silence,),
    )
    mix_inputs = [silence, *(pad.label for pad in pending_pads)]
    builder.add(
        mix_inputs,
        (
            f"amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0:normalize=0",
            f"atrim=duration={duration}",
            "asetpts=PTS-STARTPTS",
        ),
        (output_label,),
    )
    if _terminate_unconsumed_outputs(builder, output_label=output_label):
        required.add("anullsink")

    return AudioExecutionPlan(
        schema_version=1,
        source_sha256=audio_plan.source_sha256,
        sample_rate=audio_plan.sample_rate,
        layout=audio_plan.layout,
        ffmpeg_layout=ffmpeg_layout,
        sequence_duration=audio_plan.sequence_duration,
        inputs=inputs,
        input_asset_ids=used_asset_ids,
        filter_complex=builder.serialize(),
        graph_segments=tuple(builder.segments),
        output_label=output_label,
        items=tuple(item_execution),
        required_filters=tuple(sorted(required)),
        source_instances=source_execution,
    )


def _terminate_unconsumed_outputs(
    builder: _GraphBuilder,
    *,
    output_label: str,
) -> int:
    """Connect abandoned audio branches to explicit sinks.

    A fully frozen virtual source produces calibrated silence and therefore
    does not read the descendant media mix that was already assembled for it.
    FFmpeg and PyAV both reject that otherwise harmless, unconnected branch.
    Ending every unconsumed pad at ``anullsink`` keeps the structured graph
    valid without changing the final mix.

    Main callers:
    - ``build_audio_execution_plan`` after the root output has been assembled.
    """

    original_segments = tuple(builder.segments)
    consumed = {
        label
        for segment in original_segments
        for label in segment.in_labels
    }
    consumed.add(output_label)
    dangling = tuple(
        label
        for segment in original_segments
        for label in segment.out_labels
        if label not in consumed
    )
    for label in dangling:
        builder.add((label,), ("anullsink",), ())
    return len(dangling)


def _apply_audio_source_instances(
    builder: _GraphBuilder,
    pads: list[_AudioPad],
    instances: tuple[AudioSourceInstance, ...],
    *,
    sequence_duration: Fraction,
    sample_rate: int,
    ffmpeg_layout: str,
    output_layout: str,
    freeze_audio_policy: AudioFreezePolicy,
) -> tuple[list[_AudioPad], tuple[AudioSourceExecution, ...], set[str]]:
    """Mix each virtual source locally, then execute its ordinary clip timing.

    Main callers:
    - ``build_audio_execution_plan`` after ordinary descendant item graphs are
      complete and before the sequence root mix.

    Sources are processed deepest-first. An inner completed pad therefore
    becomes one input to its parent's submix, matching the video group fold and
    preventing an outer timeMap from being copied into every source item.
    """

    remaining = list(pads)
    executions: list[AudioSourceExecution] = []
    required: set[str] = set()
    seen_paths: set[str] = set()
    ordered = sorted(
        instances,
        key=lambda instance: (len(instance.ancestor_paths), instance.path),
        reverse=True,
    )
    for index, instance in enumerate(ordered):
        timing = instance.timing
        if instance.path in seen_paths:
            raise AudioExecutionError(
                f"duplicate audio source instance path {instance.path!r}"
            )
        seen_paths.add(instance.path)
        members = tuple(
            pad for pad in remaining if instance.path in pad.ancestor_paths
        )
        remaining = [
            pad for pad in remaining if instance.path not in pad.ancestor_paths
        ]
        if not members:
            # Audio IR only permits this shape when the requested source
            # interval is completely covered by explicit gaps.
            continue
        composition_duration = max(
            sequence_duration,
            timing.absolute_start + timing.source_duration,
        )
        base = builder.label(f"source_instance_{index}_silence")
        builder.add(
            (),
            (
                f"anullsrc=r={sample_rate}:cl={ffmpeg_layout}",
                f"atrim=duration={_number_text(composition_duration)}",
                "asetpts=PTS-STARTPTS",
            ),
            (base,),
        )
        mixed = builder.label(f"source_instance_{index}_mixed")
        inputs = (base, *(member.label for member in members))
        builder.add(
            inputs,
            (
                f"amix=inputs={len(inputs)}:duration=first:dropout_transition=0:normalize=0",
                f"atrim=duration={_number_text(composition_duration)}",
                "asetpts=PTS-STARTPTS",
            ),
            (mixed,),
        )
        try:
            rebased = rebase_source_retime(
                timing.retime_map,
                source_origin=timing.source_start,
                stream_start=timing.absolute_start,
            )
            execution = build_retime_execution_plan(
                rebased,
                video_frame_duration=Fraction(1, 30),
                include_audio=True,
                audio_sample_rate=sample_rate,
                audio_channel_layout=ffmpeg_layout,
                preserve_audio_pitch=instance.preserves_pitch,
                freeze_audio_policy=freeze_audio_policy,
            )
            retimed = builder.label(f"source_instance_{index}_retimed")
            prefix = builder.label(f"source_instance_{index}_retime_prefix")
            retime_segments = build_audio_filtergraph_segments(
                execution,
                input_label=mixed,
                output_label=retimed,
                label_prefix=prefix,
            )
        except (ValueError, AudioFreezeBehaviorBlocked) as error:
            raise AudioExecutionError(
                f"{instance.path} has an unexecutable virtual-source audio retime: {error}"
            ) from error
        # Splice each retime chain as a STRUCTURED segment (not add_raw): the
        # node/edge form serialises back to the exact same string, so the byte
        # gate holds, while the PyAV port gets real filter nodes to build.
        for in_labels, filters, out_labels in retime_segments:
            builder.add(in_labels, filters, out_labels)
        audio_retime_filters = {
            "anullsrc",
            "areverse",
            "aresample",
            "asetpts",
            "asetrate",
            "asplit",
            "atempo",
            "atrim",
            "concat",
        }
        required.update(
            name
            for name in required_retime_filters(execution)
            if name in audio_retime_filters
        )
        retimed, control_required = _apply_source_instance_controls(
            builder,
            retimed,
            instance,
            output_layout=output_layout,
            ffmpeg_layout=ffmpeg_layout,
        )
        required.update(control_required)
        delayed = retimed
        delay_samples = _time_to_sample(timing.absolute_start, sample_rate)
        if delay_samples:
            delayed = builder.chain(
                retimed,
                (f"adelay=delays={delay_samples}S:all=1",),
                stem=f"source_instance_{index}_timeline",
            )
            required.add("adelay")
        remaining.append(
            _AudioPad(
                label=delayed,
                ancestor_paths=instance.ancestor_paths,
            )
        )
        executions.append(
            AudioSourceExecution(
                path=instance.path,
                source_id=instance.source_id,
                input_count=len(members),
                output_label=delayed,
                start_sample=delay_samples,
                source_origin=timing.source_start,
                source_duration=timing.source_duration,
            )
        )
    return remaining, tuple(executions), required


def _apply_source_instance_controls(
    builder: _GraphBuilder,
    source: str,
    instance: AudioSourceInstance,
    *,
    output_layout: str,
    ffmpeg_layout: str,
) -> tuple[str, set[str]]:
    """Apply ordinary clip audio controls after its source timing.

    Static/animated gain, fades, stereo panning, and calibrated enhancements
    reuse the ordinary item routines. Source-window role controls and mute
    ranges remain invalid here because they require a second source-domain
    clock rather than the completed output clock.
    """

    timing = instance.timing
    if not instance.controls:
        return source, set()
    proxy = RenderAudioItem(
        id="source_instance",
        path=instance.path,
        name=None,
        absolute_start=Fraction(0),
        duration=timing.duration,
        source_start=Fraction(0),
        source_duration=timing.duration,
        asset_id="<renderable-source>",
        asset_uid=None,
        source_stream_id="1",
        source_sample_rate=None,
        source_channels=(),
        output_channels=None,
        role=None,
        enabled=True,
        active=True,
        control_layers=instance.controls,
        retime=(),
        preserves_pitch=instance.preserves_pitch,
    )
    current = source
    required: set[str] = set()
    for layer_index, layer in enumerate(instance.controls):
        _validate_control_layer(layer, proxy)
        if layer.mutes or layer.source_start is not None or layer.source_duration is not None:
            raise UnsupportedAudioControlError(
                f"{layer.path} uses a source-window audio control on a retimed "
                "renderable source"
            )
        if layer.gain is not None:
            expression = _gain_amplitude_expression(layer.gain, proxy)
            current = builder.chain(
                current,
                (f"volume='{expression}':eval=frame",),
                stem=f"source_instance_gain_{layer_index}",
            )
            required.add("volume")
            current = _apply_fades(
                builder,
                current,
                layer.gain.fades,
                timing.duration,
                stem=f"source_instance_gainfade_{layer_index}",
            )
            if layer.gain.fades:
                required.add("afade")
        if layer.panner is not None:
            current = _apply_panner(
                builder,
                current,
                layer.panner,
                proxy,
                output_layout=output_layout,
                ffmpeg_layout=ffmpeg_layout,
                stem=f"source_instance_panner_{layer_index}",
            )
            required.update(("channelsplit", "volume", "join"))
        if layer.enhancements:
            try:
                plan = build_audio_enhancement_plan(layer.enhancements)
                plan.require_executable()
            except AudioEnhancementError as error:
                raise UnsupportedAudioControlError(
                    f"{layer.path} has an unexecutable source-instance "
                    f"audio enhancement: {error}"
                ) from error
            if plan.filters:
                current = builder.chain(
                    current,
                    plan.filters,
                    stem=f"source_instance_enhancement_{layer_index}",
                )
            required.update(plan.required_filters)
    return current, required


def _resolve_output_routes(
    item: RenderAudioItem,
    stream: AudioStreamBinding,
    source_channels: tuple[int, ...],
    output_positions: tuple[str, ...],
) -> tuple[tuple[int, str, Fraction], ...]:
    if item.output_channels is not None:
        if len(item.output_channels) != len(source_channels):
            raise AudioStreamResolutionError(
                f"{item.path} maps {len(source_channels)} source channels to "
                f"{len(item.output_channels)} output channels"
            )
        routes = []
        for source, raw_output in zip(source_channels, item.output_channels):
            if raw_output == "X":
                continue
            output = _FCP_OUTPUT_TO_FFMPEG.get(raw_output)
            if output is None or output not in output_positions:
                raise AudioStreamResolutionError(
                    f"{item.path} routes to unavailable output channel {raw_output!r}"
                )
            routes.append((source, output, Fraction(1)))
        return tuple(routes)

    count = len(source_channels)
    if output_positions == ("FC",):
        coefficient = Fraction(1, count)
        return tuple((source, "FC", coefficient) for source in source_channels)
    if output_positions == ("FL", "FR"):
        if count == 1:
            return (
                (source_channels[0], "FL", Fraction(1)),
                (source_channels[0], "FR", Fraction(1)),
            )
        semantic = _source_channel_positions(stream)
        selected = tuple((source, semantic[source - 1]) for source in source_channels)
        routes: list[tuple[int, str, Fraction]] = []
        for source, position in selected:
            if position == "FL":
                routes.append((source, "FL", Fraction(1)))
            elif position == "FR":
                routes.append((source, "FR", Fraction(1)))
            elif position == "FC":
                routes.extend(
                    ((source, "FL", Fraction(707, 1000)), (source, "FR", Fraction(707, 1000)))
                )
            elif position in {"SL", "BL", "BC"}:
                routes.append((source, "FL", Fraction(707, 1000)))
            elif position in {"SR", "BR"}:
                routes.append((source, "FR", Fraction(707, 1000)))
            elif position == "LFE":
                # Final Cut's default LFE downmix is not guessed.
                continue
            else:
                raise AudioStreamResolutionError(
                    f"{item.path} has unsupported source position {position!r}"
                )
        return tuple(routes)
    if output_positions == ("FL", "FR", "FC", "LFE", "SL", "SR"):
        if count == 1:
            return ((source_channels[0], "FC", Fraction(1)),)
        semantic = _source_channel_positions(stream)
        routes = []
        aliases = {"BL": "SL", "BR": "SR", "BC": "FC"}
        for source in source_channels:
            position = aliases.get(semantic[source - 1], semantic[source - 1])
            if position not in output_positions:
                raise AudioStreamResolutionError(
                    f"{item.path} cannot default-route source position {position!r} to 5.1"
                )
            routes.append((source, position, Fraction(1)))
        return tuple(routes)
    raise AudioExecutionError("unreachable output layout")


def _source_channel_positions(stream: AudioStreamBinding) -> tuple[str, ...]:
    if stream.channel_layout in _KNOWN_SOURCE_LAYOUTS:
        positions = _KNOWN_SOURCE_LAYOUTS[stream.channel_layout]
        if len(positions) == stream.channels:
            return positions
    if stream.channels == 1:
        return ("FC",)
    if stream.channels == 2:
        return ("FL", "FR")
    raise AudioStreamResolutionError(
        f"stream {stream.source_stream_id!r} has {stream.channels} channels but no "
        "recognized channel layout; explicit outCh routing is required"
    )


def _build_item_graph(
    builder: _GraphBuilder,
    item: RenderAudioItem,
    source_label: str,
    *,
    routes: tuple[tuple[int, str, Fraction], ...],
    output_layout: str,
    ffmpeg_layout: str,
    sample_rate: int,
    freeze_audio_policy: AudioFreezePolicy,
) -> tuple[str, set[str], bool, tuple[Mapping[str, Any], ...]]:
    if item.duration <= 0 or item.source_duration <= 0 or item.source_start < 0:
        raise AudioExecutionError(f"{item.path} has an invalid audio interval")
    for layer in item.control_layers:
        _validate_control_layer(layer, item)

    required = {"atrim", "asetpts", "aresample"}
    enhancement_plans = []

    # Decide up front whether this item reads ANY source media.  A fully frozen
    # item -- every retime segment is a freeze that Final Cut renders as
    # calibrated silence (``_FCP_FREEZE_SILENCE_CALIBRATION``) -- produces pure
    # ``anullsrc`` silence and consumes nothing from the source.  Building the
    # source ``atrim`` + ``pan`` route in that case leaves the ``pan`` output pad
    # unconnected, which FFmpeg (exit 234) and the in-process PyAV graph (Errno
    # 22) both reject.  So skip the media route entirely for a pure-silence item:
    # silence through any gain/mute/panner control is still silence, so nothing is
    # lost, and only fully-frozen items take this branch (every item with >= 1
    # media segment, including play-then-freeze, keeps its exact media route byte
    # for byte).
    #
    # NOTE: "freeze audio == silence" is the current MVP assumption; it is not yet
    # verified against real Final Cut and may be revisited post-MVP.
    retimed = bool(item.retime)
    retime_execution = (
        _build_item_retime_execution(
            item,
            sample_rate=sample_rate,
            ffmpeg_layout=ffmpeg_layout,
            freeze_audio_policy=freeze_audio_policy,
        )
        if retimed
        else None
    )
    consumes_media = retime_execution is None or any(
        segment.operation == "media" for segment in retime_execution.audio_segments
    )

    current = None
    if consumes_media:
        required.add("pan")
        source_end = item.source_start + item.source_duration
        current = builder.chain(
            source_label,
            (
                f"atrim=start={_number_text(item.source_start)}:end={_number_text(source_end)}",
                "asetpts=PTS-STARTPTS",
            ),
            stem=f"{item.id}_trim",
        )
        current = builder.chain(
            current,
            (_pan_filter(routes, ffmpeg_layout),),
            stem=f"{item.id}_route",
        )

        for layer_index, layer in enumerate(item.control_layers):
            if layer.gain is not None:
                expression = _gain_amplitude_expression(layer.gain, item)
                current = builder.chain(
                    current,
                    (f"volume='{expression}':eval=frame",),
                    stem=f"{item.id}_gain_{layer_index}",
                )
                required.add("volume")
                current = _apply_fades(
                    builder,
                    current,
                    layer.gain.fades,
                    item.source_duration,
                    stem=f"{item.id}_gainfade_{layer_index}",
                )
                if layer.gain.fades:
                    required.add("afade")
            for mute_index, mute in enumerate(layer.mutes):
                expression = _mute_amplitude_expression(mute, item)
                current = builder.chain(
                    current,
                    (f"volume='{expression}':eval=frame",),
                    stem=f"{item.id}_mute_{layer_index}_{mute_index}",
                )
                required.add("volume")
            if layer.panner is not None:
                current = _apply_panner(
                    builder,
                    current,
                    layer.panner,
                    item,
                    output_layout=output_layout,
                    ffmpeg_layout=ffmpeg_layout,
                    stem=f"{item.id}_panner_{layer_index}",
                )
                required.update(("channelsplit", "volume", "join"))
            if layer.source_start is not None or layer.source_duration is not None:
                expression = _role_window_expression(layer, item)
                current = builder.chain(
                    current,
                    (f"volume='{expression}':eval=frame",),
                    stem=f"{item.id}_role_window_{layer_index}",
                )
                required.add("volume")

    if retimed:
        current, retime_required = _apply_retime(
            builder,
            current if current is not None else source_label,
            item,
            retime_execution,
        )
        required.update(retime_required)
    else:
        current = builder.chain(
            current,
            (
                f"atrim=duration={_number_text(item.duration)}",
                "asetpts=PTS-STARTPTS",
            ),
            stem=f"{item.id}_duration",
        )

    # Final Cut applies these intrinsic cleanup controls to the already
    # routed/editorially retimed signal. Keep their order across control
    # scopes and fail the whole item if one active adjustment is opaque.
    for layer_index, layer in enumerate(item.control_layers):
        if not layer.enhancements:
            continue
        try:
            enhancement_plan = build_audio_enhancement_plan(layer.enhancements)
            enhancement_plan.require_executable()
        except AudioEnhancementError as error:
            raise UnsupportedAudioControlError(
                f"{layer.path} has an unexecutable audio enhancement: {error}"
            ) from error
        if enhancement_plan.filters:
            current = builder.chain(
                current,
                enhancement_plan.filters,
                stem=f"{item.id}_enhancements_{layer_index}",
            )
        required.update(enhancement_plan.required_filters)
        enhancement_plans.append(enhancement_plan.manifest())

    delay_samples = _time_to_sample(item.absolute_start, sample_rate)
    filters = [f"aresample={sample_rate}"]
    if delay_samples:
        filters.append(f"adelay=delays={delay_samples}S:all=1")
        required.add("adelay")
    current = builder.chain(
        current,
        tuple(filters),
        stem=f"{item.id}_timeline",
    )
    return current, required, retimed, tuple(enhancement_plans)


def _validate_control_layer(layer: AudioControlLayer, item: RenderAudioItem) -> None:
    if not layer.enabled or not layer.active:
        raise AudioExecutionError(
            f"audible item {item.path} contains inactive control layer {layer.path}"
        )
    if layer.gain is not None and layer.gain.unit != "dB":
        raise UnsupportedAudioControlError(
            f"{layer.path} gain unit {layer.gain.unit!r} is unsupported"
        )
    if layer.panner is not None and layer.panner.amount.unit != "normalized":
        raise UnsupportedAudioControlError(
            f"{layer.path} panner unit {layer.panner.amount.unit!r} is unsupported"
        )


def _pan_filter(
    routes: tuple[tuple[int, str, Fraction], ...], ffmpeg_layout: str
) -> str:
    terms: dict[str, list[str]] = {}
    for source_channel, output_channel, coefficient in routes:
        if coefficient == 0:
            continue
        rendered = f"c{source_channel - 1}"
        if coefficient != 1:
            rendered = f"{_number_text(coefficient)}*{rendered}"
        terms.setdefault(output_channel, []).append(rendered)
    assignments = []
    for output in _OUTPUT_LAYOUTS[
        "mono" if ffmpeg_layout == "mono" else "stereo" if ffmpeg_layout == "stereo" else "surround"
    ][1]:
        expression = "+".join(terms.get(output, ())) or "0*c0"
        assignments.append(f"{output}={expression}")
    return f"pan={ffmpeg_layout}|" + "|".join(assignments)


def _gain_amplitude_expression(
    gain: AnimatedAudioScalar, item: RenderAudioItem
) -> str:
    db_expression = _animated_expression(
        gain,
        item,
        value_name="gain",
        minimum=None,
        maximum=None,
    )
    return f"pow(10,({db_expression})/20)"


def _animated_expression(
    animated: AnimatedAudioScalar,
    item: RenderAudioItem,
    *,
    value_name: str,
    minimum: float | None,
    maximum: float | None,
) -> str:
    values = [animated.initial, *(point.value for point in animated.keyframes)]
    if any(not math.isfinite(value) for value in values):
        raise UnsupportedAudioControlError(
            f"{item.path} has non-finite {value_name} automation"
        )
    if minimum is not None and any(value < minimum for value in values):
        raise UnsupportedAudioControlError(
            f"{item.path} has {value_name} below {minimum}"
        )
    if maximum is not None and any(value > maximum for value in values):
        raise UnsupportedAudioControlError(
            f"{item.path} has {value_name} above {maximum}"
        )
    for point in animated.keyframes:
        if point.aux_value is not None:
            raise UnsupportedAudioControlError(
                f"{item.path} has uncalibrated {value_name} auxValue"
            )
    points = tuple(animated.keyframes)
    if not points:
        return _float_text(animated.initial)
    relative = tuple((point.time - item.source_start, point) for point in points)
    expression = _float_text(points[-1].value)
    for (left_time, left), (right_time, right) in reversed(tuple(zip(relative, relative[1:]))):
        if right_time <= left_time:
            raise UnsupportedAudioControlError(
                f"{item.path} has duplicate or reversed {value_name} keyframes"
            )
        progress = f"((t-{_number_text(left_time)})/{_number_text(right_time - left_time)})"
        shaped = _interpolation_expression(progress, left.interp, left.curve, item.path)
        value = (
            f"({_float_text(left.value)}+({_float_text(right.value)}-"
            f"{_float_text(left.value)})*{shaped})"
        )
        expression = f"if(lt(t,{_number_text(right_time)}),{value},{expression})"
    first_time, first = relative[0]
    return f"if(lt(t,{_number_text(first_time)}),{_float_text(first.value)},{expression})"


def _interpolation_expression(
    progress: str, interp: str, curve: str, path: str
) -> str:
    name = (interp or curve or "linear").replace("-", "").casefold()
    if name == "linear":
        return progress
    if name in {"ease", "smooth"}:
        return f"(({progress})*({progress})*(3-2*({progress})))"
    if name == "easein":
        return f"(({progress})*({progress}))"
    if name == "easeout":
        return f"(1-(1-({progress}))*(1-({progress})))"
    raise UnsupportedAudioControlError(
        f"{path} requests unsupported audio interpolation {interp!r}/{curve!r}"
    )


def _apply_fades(
    builder: _GraphBuilder,
    source: str,
    fades: tuple[AudioFade, ...],
    duration: Fraction,
    *,
    stem: str,
) -> str:
    current = source
    for index, fade in enumerate(fades):
        if fade.duration < 0 or fade.duration > duration:
            raise UnsupportedAudioControlError(
                f"audio fade duration {fade.duration} is outside item duration {duration}"
            )
        curve = _fade_curve(fade.curve)
        if fade.kind == "in":
            spec = f"afade=t=in:st=0:d={_number_text(fade.duration)}:curve={curve}"
        elif fade.kind == "out":
            start = duration - fade.duration
            spec = f"afade=t=out:st={_number_text(start)}:d={_number_text(fade.duration)}:curve={curve}"
        else:
            raise UnsupportedAudioControlError(f"unknown fade kind {fade.kind!r}")
        current = builder.chain(current, (spec,), stem=f"{stem}_{index}")
    return current


def _fade_curve(name: str) -> str:
    normalized = name.replace("-", "").casefold()
    mapping = {
        "linear": "tri",
        "tri": "tri",
        "ease": "qsin",
        "easein": "qsin",
        "easeout": "qsin",
        "easeinout": "hsin",
        "smooth": "hsin",
    }
    try:
        return mapping[normalized]
    except KeyError as error:
        raise UnsupportedAudioControlError(
            f"unsupported audio fade curve {name!r}"
        ) from error


def _mute_amplitude_expression(mute: AudioMuteRange, item: RenderAudioItem) -> str:
    source_start = (
        mute.source_start if mute.source_start is not None else item.source_start
    )
    start = source_start - item.source_start
    end = item.source_duration if mute.duration is None else start + mute.duration
    start = max(Fraction(0), start)
    end = min(item.source_duration, end)
    if end <= start:
        return "1"
    fade_in = next((fade for fade in mute.fades if fade.kind == "in"), None)
    fade_out = next((fade for fade in mute.fades if fade.kind == "out"), None)
    if len([fade for fade in mute.fades if fade.kind == "in"]) > 1 or len(
        [fade for fade in mute.fades if fade.kind == "out"]
    ) > 1:
        raise UnsupportedAudioControlError(f"{item.path} has duplicate mute fades")
    expression = f"if(between(t,{_number_text(start)},{_number_text(end)}),0,1)"
    if fade_in is not None and fade_in.duration > 0:
        ramp_start = max(Fraction(0), start - fade_in.duration)
        width = start - ramp_start
        if width > 0:
            ramp = f"(1-(t-{_number_text(ramp_start)})/{_number_text(width)})"
            expression = (
                f"if(between(t,{_number_text(ramp_start)},{_number_text(start)}),"
                f"{ramp},{expression})"
            )
    if fade_out is not None and fade_out.duration > 0:
        ramp_end = min(item.source_duration, end + fade_out.duration)
        width = ramp_end - end
        if width > 0:
            ramp = f"((t-{_number_text(end)})/{_number_text(width)})"
            expression = (
                f"if(between(t,{_number_text(end)},{_number_text(ramp_end)}),"
                f"{ramp},{expression})"
            )
    return expression


def _apply_panner(
    builder: _GraphBuilder,
    source: str,
    panner: AudioPanner,
    item: RenderAudioItem,
    *,
    output_layout: str,
    ffmpeg_layout: str,
    stem: str,
) -> str:
    mode = (panner.mode or "stereo").casefold()
    if mode not in {"stereo", "balance"}:
        raise UnsupportedAudioControlError(
            f"{item.path} requests unsupported panner mode {panner.mode!r}"
        )
    amount = _animated_expression(
        panner.amount,
        item,
        value_name="pan",
        minimum=-1.0,
        maximum=1.0,
    )
    if output_layout != "stereo":
        if all(value == 0 for value in [panner.amount.initial, *(point.value for point in panner.amount.keyframes)]):
            return source
        raise UnsupportedAudioControlError(
            f"{item.path} has non-zero stereo panning in {output_layout} output"
        )
    left = builder.label(f"{stem}_left")
    right = builder.label(f"{stem}_right")
    builder.add(
        (source,),
        (f"channelsplit=channel_layout={ffmpeg_layout}",),
        (left, right),
    )
    # Final Cut's Stereo Left/Right panner uses a constant-power curve.  A
    # genuine Final Cut 12.3 export at amount=65 measured -11.23 dB between
    # the channels; cos/sin quarter-circle panning predicts -10.99 dB.  The
    # former linear balance expression predicted only -9.12 dB.
    left_gain = f"cos((({amount})+1)*PI/4)"
    right_gain = f"sin((({amount})+1)*PI/4)"
    left_out = builder.chain(
        left, (f"volume='{left_gain}':eval=frame",), stem=f"{stem}_left_gain"
    )
    right_out = builder.chain(
        right, (f"volume='{right_gain}':eval=frame",), stem=f"{stem}_right_gain"
    )
    joined = builder.label(f"{stem}_join")
    builder.add(
        (left_out, right_out),
        ("join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR",),
        (joined,),
    )
    return joined


def _role_window_expression(layer: AudioControlLayer, item: RenderAudioItem) -> str:
    source_start = (
        layer.source_start if layer.source_start is not None else item.source_start
    )
    start = source_start - item.source_start
    duration = (
        layer.source_duration
        if layer.source_duration is not None
        else item.source_duration
    )
    end = start + duration
    return f"if(between(t,{_number_text(start)},{_number_text(end)}),1,0)"


def _build_item_retime_execution(
    item: RenderAudioItem,
    *,
    sample_rate: int,
    ffmpeg_layout: str,
    freeze_audio_policy: AudioFreezePolicy,
):
    """Build one item's retime execution plan without emitting any graph.

    Split out from ``_apply_retime`` so ``_build_item_graph`` can inspect the
    plan's segments -- specifically whether ANY segment reads source media --
    before it decides to build the source ``atrim``/``pan`` route.  A fully
    frozen item consumes no media and must skip that route (otherwise its ``pan``
    output is left unconnected and the graph is rejected).  Touches nothing on the
    builder, so calling it before the media route does not perturb label order.
    """
    points = tuple(
        RetimePoint(
            timeline_time=point.output_time,
            source_time=point.source_time - item.source_start,
            interpolation=point.interp,
        )
        for point in item.retime
    )
    try:
        retime_map = RetimeMap.from_points_visible(points, item.duration)
    except ValueError as error:
        raise UnsupportedAudioControlError(
            f"{item.path} has invalid or nonlinear audio retiming: {error}"
        ) from error
    for segment in retime_map.segments:
        low = min(segment.source_start, segment.source_end)
        high = max(segment.source_start, segment.source_end)
        if low < 0 or high > item.source_duration:
            raise UnsupportedAudioControlError(
                f"{item.path} retime source range [{low}, {high}] is outside "
                f"the bound source duration {item.source_duration}"
            )
    return build_retime_execution_plan(
        retime_map,
        video_frame_duration=Fraction(1, 30),
        include_audio=True,
        audio_sample_rate=sample_rate,
        audio_channel_layout=ffmpeg_layout,
        preserve_audio_pitch=item.preserves_pitch,
        freeze_audio_policy=freeze_audio_policy,
    )


def _apply_retime(
    builder: _GraphBuilder,
    source: str,
    item: RenderAudioItem,
    execution,
) -> tuple[str, tuple[str, ...]]:
    """Emit the retime segment graph for a precomputed ``execution``.

    ``source`` is the upstream media label.  A pure-silence (fully frozen)
    execution reads nothing from ``source`` -- the caller passes the raw source
    label as an unused placeholder and skips building the media route.
    """
    output = builder.label(f"{item.id}_retime")
    prefix = builder.label(f"{item.id}_retime_prefix")
    try:
        retime_segments = build_audio_filtergraph_segments(
            execution,
            input_label=source,
            output_label=output,
            label_prefix=prefix,
        )
    except AudioFreezeBehaviorBlocked as error:
        raise AudioFreezeCalibrationError(
            f"{item.path} contains freeze audio without a calibration policy"
        ) from error
    # Structured splice (see the source-instance retime above): byte-identical
    # to the old add_raw(graph.split(";")) but exposes filter nodes to PyAV.
    for in_labels, filters, out_labels in retime_segments:
        builder.add(in_labels, filters, out_labels)
    audio_only_filters = {
        "anullsrc",
        "areverse",
        "aresample",
        "asetpts",
        "asetrate",
        "asplit",
        "atempo",
        "atrim",
        "concat",
    }
    return output, tuple(
        name for name in required_retime_filters(execution) if name in audio_only_filters
    )


def probe_stock_ffmpeg_audio_capabilities(
    plan: AudioExecutionPlan,
    *,
    ffmpeg_path: str | Path = "ffmpeg",
) -> FFmpegAudioCapabilityReport:
    """Read one installed FFmpeg's filters and compare the exact requirement set."""

    try:
        version = subprocess.run(
            (str(ffmpeg_path), "-hide_banner", "-version"),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        filters = subprocess.run(
            (str(ffmpeg_path), "-hide_banner", "-filters"),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise MissingFFmpegAudioCapability(
            f"could not inspect FFmpeg executable {str(ffmpeg_path)!r}: {error}"
        ) from error
    available = _parse_filter_names(filters.stdout)
    missing = tuple(name for name in plan.required_filters if name not in available)
    return FFmpegAudioCapabilityReport(
        executable=str(ffmpeg_path),
        version_line=version.stdout.splitlines()[0] if version.stdout else "unknown",
        required_filters=plan.required_filters,
        available_filters=tuple(sorted(available)),
        missing_filters=missing,
    )


def run_audio_execution_plan(
    plan: AudioExecutionPlan,
    output_path: str | Path,
    *,
    ffmpeg_path: str | Path = "ffmpeg",
    codec: str = "pcm_s24le",
    timeout_seconds: float = 120,
) -> subprocess.CompletedProcess[str]:
    """Run a frozen plan without a shell after checking its stock filters."""

    capability = probe_stock_ffmpeg_audio_capabilities(plan, ffmpeg_path=ffmpeg_path)
    capability.require_supported()
    try:
        return subprocess.run(
            plan.command(output_path, ffmpeg_path=ffmpeg_path, codec=codec),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise AudioExecutionError(f"FFmpeg audio render failed: {error}") from error


def _time_to_sample(value: Fraction, sample_rate: int) -> int:
    exact = value * sample_rate
    if exact < 0:
        raise AudioExecutionError("audio timeline delay cannot be negative")
    return (exact.numerator * 2 + exact.denominator) // (2 * exact.denominator)


def _number_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    with localcontext() as context:
        context.prec = 24
        rendered = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return rendered.rstrip("0").rstrip(".")


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _float_text(value: float) -> str:
    if not math.isfinite(value):
        raise UnsupportedAudioControlError("audio expression value must be finite")
    return format(value, ".12g")


def _parse_filter_names(output: str) -> set[str]:
    pattern = re.compile(r"^\s*[.TSCAX|]{2,6}\s+([A-Za-z0-9_]+)\s+", re.MULTILINE)
    return {match.group(1) for match in pattern.finditer(output)}


__all__ = [
    "AudioAssetBinding",
    "AudioAssetBindingError",
    "AudioExecutionError",
    "AudioExecutionPlan",
    "AudioFreezeCalibrationError",
    "AudioItemExecution",
    "AudioStreamBinding",
    "AudioStreamResolutionError",
    "FFmpegAudioCapabilityReport",
    "MissingFFmpegAudioCapability",
    "UnsupportedAudioControlError",
    "build_audio_execution_plan",
    "probe_audio_asset",
    "probe_stock_ffmpeg_audio_capabilities",
    "run_audio_execution_plan",
]
