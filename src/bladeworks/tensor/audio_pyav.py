"""In-process PyAV audio delivery: build the calibrated graph node-by-node.

Architecture map
================

This is Stage 2b of the PyAV audio-delivery unification (plan:
``pyav-audio-delivery-unification.md``).  The calibrated audio semantics live in
``core.audio_execution`` as an ``AudioExecutionPlan`` -- a stock-FFmpeg filtergraph
plus the input media it references.  The CPU path handed that graph, as a
``filter_complex`` *string*, to a second ffmpeg process.  PyAV 16 exposes **no**
``filter_complex`` string parser, so this module rebuilds the identical graph
**node-by-node** from the plan's structured ``graph_segments`` and runs it inside
this process on the same libav the video encoder already links.

Stages, top to bottom:

    open_sources(plan)          : decode each ``I:a:O`` media stream -> abuffer     (per source key)
    build_graph(plan)           : one ``graph.add`` per AudioFilterNode, wired by pad labels
      + terminal aresample/aformat -> abuffersink                                   (target rate/layout, fltp)
    pump(graph)                 : push decoded frames, EOF, pull output frames       (fltp @ target rate/layout)

The only thing the CLI did implicitly and we make explicit is the boundary: the
``abuffer`` source formats (read off the decoded streams) and a terminal
``aresample`` + ``aformat=sample_fmts=fltp`` that pins the delivery rate/layout/
format the encoder wants -- everything *inside* the graph is the same libav filter
with the same args as the reference, so parity is true by construction (proven in
text by the Stage 2a byte gate, and perceptually by ``test_tensor_audio_parity``).

Why this exists
---------------
Folding audio into the tensor encoder's own container (``encode.py``) deletes the
second ffmpeg process entirely.  This module produces the AAC-ready
``av.AudioFrame`` stream that ``VideoEncoder`` muxes beside the video.

Main callers:
- ``tensor.encode``/``executor`` delivery path (the frames it yields).
- ``experimental_tests.core.test_tensor_audio_parity`` (rendered to a wav).
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import av

from ..core.audio_execution import (
    AudioExecutionPlan,
    AudioGraphSegment,
    AudioRawSegment,
)
from .errors import TensorRenderError


# A media source pad label is ``inputIndex:a:audioOrdinal`` (e.g. ``0:a:0``).
# Every other label is an internal graph pad allocated by ``_GraphBuilder``.
_SOURCE_LABEL_RE = re.compile(r"^(\d+):a:(\d+)$")

# The delivery audio is always float planar so the AAC encoder takes it directly.
_DELIVERY_SAMPLE_FORMAT = "fltp"


def _is_source_label(label: str) -> bool:
    return _SOURCE_LABEL_RE.fullmatch(label) is not None


class _MediaSource:
    """One decoded ``I:a:O`` stream feeding one ``abuffer`` graph source.

    Holds the decoded frames plus the format they must be pushed at.  The frames
    are decoded eagerly (audio is small) before the graph is configured so the
    ``abuffer`` can be built from the real stream format -- exactly the format
    the CLI's ``-i`` decode would have produced.
    """

    def __init__(self, path: Path, audio_ordinal: int) -> None:
        self.path = path
        self.audio_ordinal = audio_ordinal
        self.frames: list[av.AudioFrame] = []
        self.sample_rate: int = 0
        self.format_name: str = ""
        self.layout_name: str = ""
        self.time_base: Fraction = Fraction(1, 48_000)
        self._decode()

    def _decode(self) -> None:
        with av.open(str(self.path)) as container:
            audio_streams = list(container.streams.audio)
            if self.audio_ordinal >= len(audio_streams):
                raise TensorRenderError(
                    f"audio source {self.path} has no audio stream ordinal "
                    f"{self.audio_ordinal} ({len(audio_streams)} present)"
                )
            stream = audio_streams[self.audio_ordinal]
            self.time_base = stream.time_base or Fraction(1, 48_000)
            for frame in container.decode(stream):
                # A decoded frame with no owning stream time_base still needs one
                # for the abuffer; use the stream's.
                self.frames.append(frame)
        if not self.frames:
            raise TensorRenderError(
                f"audio source {self.path} stream {self.audio_ordinal} decoded no frames"
            )
        first = self.frames[0]
        self.sample_rate = int(first.sample_rate)
        self.format_name = first.format.name
        self.layout_name = first.layout.name


def _add_filter(graph: av.filter.Graph, name: str, args: str):
    """Add one libav filter, passing args only when present.

    ``graph.add(name)`` with no args is not the same call as ``graph.add(name,
    "")`` for every filter, so branch explicitly.
    """

    if args:
        return graph.add(name, args)
    return graph.add(name)


def _build_graph(
    plan: AudioExecutionPlan,
    sources: dict[str, _MediaSource],
) -> tuple[av.filter.Graph, dict[str, "_Pad"], object]:
    """Rebuild ``plan.graph_segments`` as a live ``av.filter.Graph``.

    Returns the configured graph, the source abuffer contexts (by label) and the
    ``abuffersink`` context.  Raises loudly on any ``AudioRawSegment`` -- chunk 2
    made every retime chain structured, so a raw segment means a regression that
    would otherwise need the banned string parser.
    """

    if not plan.graph_segments:
        raise TensorRenderError("audio execution plan carries no structured graph_segments")

    graph = av.filter.Graph()
    produced: dict[str, _Pad] = {}

    # Source abuffers first, so their output pads exist before any chain links
    # to them.
    source_contexts: dict[str, av.filter.context.FilterContext] = {}
    for label, source in sources.items():
        buffer = graph.add_abuffer(
            sample_rate=source.sample_rate,
            format=source.format_name,
            layout=source.layout_name,
            time_base=source.time_base,
        )
        source_contexts[label] = buffer
        produced[label] = _Pad(buffer, 0)

    for segment in plan.graph_segments:
        if isinstance(segment, AudioRawSegment):
            raise TensorRenderError(
                "audio graph still contains an unstructured retime segment; the "
                "PyAV builder needs node/edge form (see build_audio_filtergraph_segments)"
            )
        if not isinstance(segment, AudioGraphSegment):
            raise TensorRenderError(f"unknown audio graph segment {segment!r}")
        nodes = [_add_filter(graph, node.name, node.args) for node in segment.filters]
        # Chain the filters of this one filterchain linearly.
        for previous, following in zip(nodes, nodes[1:]):
            previous.link_to(following, 0, 0)
        # Wire this chain's inputs from already-produced pads.
        for input_index, label in enumerate(segment.in_labels):
            pad = produced.get(label)
            if pad is None:
                raise TensorRenderError(
                    f"audio graph references pad {label!r} before it is produced"
                )
            pad.context.link_to(nodes[0], pad.output_index, input_index)
        # Register this chain's output pads.
        for output_index, label in enumerate(segment.out_labels):
            produced[label] = _Pad(nodes[-1], output_index)

    final_pad = produced.get(plan.output_label)
    if final_pad is None:
        raise TensorRenderError(
            f"audio graph never produced its output label {plan.output_label!r}"
        )

    # Explicit boundary negotiation: pin the delivery rate/layout/format the CLI
    # used to leave to the encoder's implicit conversion.
    resample = graph.add("aresample", str(plan.sample_rate))
    reformat = graph.add(
        "aformat",
        f"sample_fmts={_DELIVERY_SAMPLE_FORMAT}:channel_layouts={plan.ffmpeg_layout}",
    )
    sink = graph.add("abuffersink")
    final_pad.context.link_to(resample, final_pad.output_index, 0)
    resample.link_to(reformat, 0, 0)
    reformat.link_to(sink, 0, 0)
    graph.configure()
    return graph, source_contexts, sink


class _Pad:
    """One produced output pad: a filter context plus which output index it is."""

    __slots__ = ("context", "output_index")

    def __init__(self, context: av.filter.context.FilterContext, output_index: int) -> None:
        self.context = context
        self.output_index = output_index


def _pump(
    sink,
    source_contexts: dict[str, av.filter.context.FilterContext],
    sources: dict[str, _MediaSource],
) -> list[av.AudioFrame]:
    """Feed every decoded source frame, signal EOF, and drain the sink.

    Generator sources inside the graph (``anullsrc`` beds) are driven by libav
    itself; only the real media abuffers are fed here.
    """

    for label, context in source_contexts.items():
        for frame in sources[label].frames:
            context.push(frame)
        context.push(None)  # EOF this source
    output: list[av.AudioFrame] = []
    while True:
        try:
            output.append(sink.pull())
        except av.error.EOFError:
            break
        except av.error.BlockingIOError as error:  # pragma: no cover - defensive
            raise TensorRenderError(
                "audio graph sink starved after all sources reached EOF; the "
                "graph did not terminate as the calibrated plan requires"
            ) from error
    if not output:
        raise TensorRenderError("audio graph produced no output frames")
    return output


def _open_sources(plan: AudioExecutionPlan) -> dict[str, _MediaSource]:
    """Open and decode every ``I:a:O`` media stream the graph references."""

    wanted: dict[str, tuple[int, int]] = {}
    for segment in plan.graph_segments:
        if not isinstance(segment, AudioGraphSegment):
            continue
        for label in segment.in_labels:
            match = _SOURCE_LABEL_RE.fullmatch(label)
            if match is None:
                continue
            wanted[label] = (int(match.group(1)), int(match.group(2)))
    sources: dict[str, _MediaSource] = {}
    for label, (input_index, audio_ordinal) in wanted.items():
        if input_index >= len(plan.inputs):
            raise TensorRenderError(
                f"audio graph references input {input_index} but the plan lists "
                f"{len(plan.inputs)} input(s); the plan must use input offset 0"
            )
        sources[label] = _MediaSource(plan.inputs[input_index], audio_ordinal)
    return sources


def render_execution_frames(plan: AudioExecutionPlan) -> list[av.AudioFrame]:
    """Render one ``AudioExecutionPlan`` to delivery frames (fltp, target rate/layout).

    Main callers:
    - ``build_delivery_frames`` (the encoder feed).
    - ``render_execution_to_wav`` (the perceptual parity gate).
    """

    sources = _open_sources(plan)
    _graph, source_contexts, sink = _build_graph(plan, sources)
    return _pump(sink, source_contexts, sources)


def render_silence_frames(
    *,
    sample_rate: int,
    ffmpeg_layout: str,
    duration: Fraction,
) -> list[av.AudioFrame]:
    """Render the calibrated sequence-length silence bed as delivery frames.

    Used when the resolver reports ``mode == "silence"`` (no audible source, or a
    video-declared-but-undecodable timeline).  Mirrors the CLI's
    ``anullsrc -> atrim -> asetpts`` bed rather than an ``amix`` over one silent
    stream.
    """

    if sample_rate <= 0 or duration <= 0:
        raise TensorRenderError("silence bed needs a positive sample rate and duration")
    graph = av.filter.Graph()
    source = graph.add("anullsrc", f"r={sample_rate}:cl={ffmpeg_layout}")
    trim = graph.add("atrim", f"duration={_number_text(duration)}")
    reset = graph.add("asetpts", "PTS-STARTPTS")
    reformat = graph.add(
        "aformat",
        f"sample_fmts={_DELIVERY_SAMPLE_FORMAT}:channel_layouts={ffmpeg_layout}",
    )
    sink = graph.add("abuffersink")
    source.link_to(trim, 0, 0)
    trim.link_to(reset, 0, 0)
    reset.link_to(reformat, 0, 0)
    reformat.link_to(sink, 0, 0)
    graph.configure()
    output: list[av.AudioFrame] = []
    while True:
        try:
            output.append(sink.pull())
        except av.error.EOFError:
            break
    if not output:
        raise TensorRenderError("silence bed produced no frames")
    return output


def _number_text(value: Fraction) -> str:
    """Exact decimal-ish text for a Fraction duration (matches audio_execution)."""

    if value.denominator == 1:
        return str(value.numerator)
    return repr(float(value))


def render_execution_to_wav(plan: AudioExecutionPlan, out_path: str | Path) -> Path:
    """Render an execution plan to a ``pcm_f32le`` wav for the parity gate.

    Encoding the delivery frames to lossless float PCM keeps the comparator's own
    floor far below the perceptual ceiling.

    Main callers:
    - ``test_tensor_audio_parity`` (candidate vs stock-FFmpeg reference).
    """

    out_path = Path(out_path)
    frames = render_execution_frames(plan)
    _encode_frames_to_wav(frames, out_path, sample_rate=plan.sample_rate, layout=plan.ffmpeg_layout)
    return out_path


def _encode_frames_to_wav(
    frames: Iterable[av.AudioFrame],
    out_path: Path,
    *,
    sample_rate: int,
    layout: str,
) -> None:
    with av.open(str(out_path), mode="w") as container:
        stream = container.add_stream("pcm_f32le", rate=sample_rate)
        stream.layout = layout
        sample_index = 0
        for frame in frames:
            frame.pts = sample_index
            frame.time_base = Fraction(1, sample_rate)
            sample_index += frame.samples
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
