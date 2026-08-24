"""Backend-neutral audio-delivery resolver for the PyAV tensor path.

Architecture map
================

This is Stage 1 of the PyAV audio-delivery unification (plan:
``pyav-audio-delivery-unification.md``).  It reproduces the *resolution* half of
``legacy_ffmpeg.ffmpeg._build_audio_execution`` -- probe audible assets, record
the loud ``omitted`` finding for declared-but-undecodable audio, drop those
items, and build the ``AudioExecutionPlan`` -- but WITHOUT the legacy function's
argv/input_paths side-effect.  The PyAV renderer (chunk 2) feeds
``execution.inputs`` to ``abuffer`` sources instead of appending ``-i`` entries.

Why this exists
---------------
The legacy resolver mutates a shared ``argv``/``input_paths`` list because the
CPU path builds one ffmpeg CLI process.  The PyAV path has no argv: it opens the
delivery container in-process (``tensor/encode.py``) and builds the audio graph
node-by-node.  So the coupling to argv must be removed while every loud finding
is preserved exactly.

TEMPORARY DUPLICATION: the neutral logic here is COPIED from, not shared with,
``_build_audio_execution`` -- that legacy function is still imported by the CPU /
segment paths (``legacy_ffmpeg/ffmpeg.py``), which are being deprecated but not
yet removed.  Delete the legacy copy when the CPU/Vulkan backends go.

Main callers:
- ``tensor`` PyAV delivery (chunk 2) -- not wired yet.
- Stage-1 unit coverage under ``experimental_tests``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Literal

from ..core.audio_execution import (
    AudioAssetBinding,
    AudioAssetBindingError,
    AudioExecutionError,
    AudioExecutionPlan,
    build_audio_execution_plan,
    probe_audio_asset,
    reject_unsupported_output_layout,
)
from ..core.errors import RenderCapabilityError
from ..core.model import RenderDocument
from ..core.report import CompatibilityReport


# The PyAV audio graph has no shared argv: its inputs are opened directly as
# ``abuffer`` sources starting at index 0 (there is no video input 0 / silence
# input 1 preceding them, unlike the legacy single-process assembly).
_TENSOR_AUDIO_INPUT_INDEX_OFFSET = 0


@dataclass(frozen=True)
class AudioDeliveryResolution:
    """The resolved delivery audio for one document.

    ``execution is None`` means pure silence: the timeline declares no audible
    audio (or none was requested), so the PyAV path must synthesise the
    calibrated ``anullsrc`` bed itself rather than build an ``amix`` over one
    silent stream.  ``mode`` mirrors the legacy ``CompositionAudioRuntime`` mode.
    """

    execution: AudioExecutionPlan | None
    effective_bindings: tuple[AudioAssetBinding, ...]
    mode: Literal["render", "silence"]
    output_sample_rate: int
    output_layout: str
    output_duration: Fraction


def audio_delivery_codec(output_profile: str) -> str:
    """Delivery audio codec name for a tensor output profile.

    Both tensor exits (``delivery`` H.264/mp4 and ``delivery_alpha`` ProRes/mov)
    carry AAC, exactly as the deleted ``tensor/assemble.py`` argument table pinned
    them.  A profile that declares no audio codec is a loud error, never a silent
    default.

    Why this exists: chunk 2 deletes the single-process assembly argv, so the
    ``-c:a`` choice that lived in the shared FFmpeg profile table now lives here
    as a codec name the encoder adds to its own container.
    """

    if output_profile in ("delivery", "delivery_alpha"):
        return "aac"
    raise RenderCapabilityError(
        f"output profile {output_profile!r} declares no delivery audio codec"
    )


def video_only_silence_resolution(document: RenderDocument) -> AudioDeliveryResolution:
    """A silence-bed resolution for a ``--video-only`` tensor render.

    The output still carries a sequence-length silence AAC track (the executor's
    ``_probe_output_audio`` requires one for every backend, and the old
    assembly muxed the same bed); the caller records the loud ``omitted`` finding
    for the dropped source audio, so ``--strict`` still fails.
    """

    return AudioDeliveryResolution(
        execution=None,
        effective_bindings=(),
        mode="silence",
        output_sample_rate=(
            document.audio.sample_rate if document.audio is not None else 48_000
        ),
        output_layout=audio_delivery_layout(document),
        output_duration=(
            document.audio.sequence_duration
            if document.audio is not None
            else document.duration
        ),
    )


def audio_delivery_layout(document: RenderDocument) -> str:
    """FFmpeg channel-layout name for the document's output audio.

    Adapted from ``legacy_ffmpeg.ffmpeg._audio_output_layout`` so the tensor path
    does not import a legacy private helper. Surround (5.1) output is rejected up
    front here -- the same capability gap ``resolve_audio_delivery`` rejects -- so
    this mapper never emits the ``5.1`` layout name that libav would refuse.
    """

    if document.audio is None:
        return "stereo"
    reject_unsupported_output_layout(document.audio.layout)
    return {
        "mono": "mono",
        "stereo": "stereo",
    }[document.audio.layout]


def resolve_audio_delivery(
    document: RenderDocument,
    *,
    ffprobe: str,
    report: CompatibilityReport,
) -> AudioDeliveryResolution:
    """Resolve independent audio IR into a PyAV-ready execution plan.

    What it does, step by step:
    1. If the document declares no audio, or nothing on the timeline is audible,
       short-circuit to a ``silence`` resolution (``execution is None``) -- the
       PyAV path synthesises the bed; no ``amix`` over a single silent stream.
    2. Collect asset paths from the document bindings, rejecting conflicting
       bindings for one resource id (loud raise).
    3. Probe every audible asset.  An asset that declares audio but carries no
       decodable stream records a VERBATIM ``omitted`` finding and is dropped
       from the effective plan; its interval renders silent.  No silent failure:
       the finding makes a ``--strict`` render fail.
    4. Build the ``AudioExecutionPlan`` at the tensor input offset (0).  Unlike
       the legacy resolver this appends NOTHING to any argv -- the caller reads
       ``execution.inputs`` to open ``abuffer`` sources.

    Main callers:
    - PyAV tensor delivery (chunk 2).

    Why this exists: see the module docstring (argv-free, backend-neutral copy of
    ``_build_audio_execution``'s resolution behaviour).
    """

    if document.audio is None:
        return AudioDeliveryResolution(
            execution=None,
            effective_bindings=(),
            mode="silence",
            output_sample_rate=48_000,
            output_layout="stereo",
            output_duration=document.duration,
        )
    # Reject surround (5.1) output at plan time, before any asset is probed or any
    # graph node built, so the tensor render fails with a clear capability message
    # instead of a cryptic libav ``Errno 22``. This also covers the silent-timeline
    # short-circuit below, whose ``surround`` bed FFmpeg would likewise reject.
    reject_unsupported_output_layout(document.audio.layout)
    if not any(item.audible for item in document.audio.items):
        return AudioDeliveryResolution(
            execution=None,
            effective_bindings=(),
            mode="silence",
            output_sample_rate=document.audio.sample_rate,
            output_layout=document.audio.layout,
            output_duration=document.duration,
        )

    asset_paths: dict[str, Path] = {}
    for binding in document.asset_bindings:
        if binding.resource_id is None:
            continue
        previous = asset_paths.get(binding.resource_id)
        if previous is not None and previous != binding.path:
            raise RenderCapabilityError(
                f"audio asset {binding.resource_id!r} has conflicting bindings"
            )
        asset_paths[binding.resource_id] = binding.path

    audible_ids = sorted(
        {item.asset_id for item in document.audio.items if item.audible}
    )
    probed: dict[str, AudioAssetBinding] = {}
    missing_stream_ids: set[str] = set()
    try:
        for asset_id in audible_ids:
            path = asset_paths.get(asset_id)
            if path is None or not path.is_file():
                raise RenderCapabilityError(
                    f"audible audio asset {asset_id!r} has no local binding"
                )
            try:
                probed[asset_id] = probe_audio_asset(
                    asset_id,
                    path,
                    ffprobe_path=ffprobe,
                )
            except AudioAssetBindingError as error:
                if "has no audio streams" not in str(error):
                    raise
                missing_stream_ids.add(asset_id)
        for item in document.audio.items:
            if item.asset_id not in missing_stream_ids or not item.audible:
                continue
            report.add(
                outcome="omitted",
                portable_status="unsupported",
                fcpxml_path=item.path,
                construct="missing audio stream",
                timeline_start=item.absolute_start,
                timeline_duration=item.duration,
                disposition=(
                    f"asset {item.asset_id!r} declares audio but the bound media "
                    "contains no decodable audio stream; the interval is silent"
                ),
            )
        effective_plan = (
            replace(
                document.audio,
                items=tuple(
                    item
                    for item in document.audio.items
                    if item.asset_id not in missing_stream_ids
                ),
            )
            if missing_stream_ids
            else document.audio
        )
        execution = build_audio_execution_plan(
            effective_plan,
            probed,
            output_label="aout",
            input_index_offset=_TENSOR_AUDIO_INPUT_INDEX_OFFSET,
        )
    except AudioExecutionError as error:
        raise RenderCapabilityError(
            f"audio graph cannot be executed: {error}"
        ) from error

    effective_bindings = tuple(
        probed[asset_id] for asset_id in execution.input_asset_ids
    )
    # If every audible item was dropped for a missing stream, the effective plan
    # renders silence even though the document declared audio.
    mode: Literal["render", "silence"] = (
        "render" if any(item.audible for item in effective_plan.items) else "silence"
    )
    return AudioDeliveryResolution(
        execution=execution if mode == "render" else None,
        effective_bindings=effective_bindings if mode == "render" else (),
        mode=mode,
        output_sample_rate=effective_plan.sample_rate,
        output_layout=effective_plan.layout,
        output_duration=effective_plan.sequence_duration,
    )
