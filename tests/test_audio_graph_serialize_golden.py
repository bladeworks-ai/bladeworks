"""Stage 2a gate: the structured audio graph serialises to the HEAD goldens.

Architecture map
================

This is the *mechanical, no-audio* half of the PyAV audio-delivery unification
(plan: ``pyav-audio-delivery-unification.md``).  It proves that after
``core/audio_execution.py`` was refactored to build a STRUCTURED node/edge graph
(``AudioGraphSegment`` / ``AudioFilterNode``) instead of raw strings, the graph
still serialises **byte-for-byte** to the calibrated ``filter_complex`` that HEAD
produced.

Why this exists
---------------
PyAV 16 has no ``filter_complex`` string parser, so chunk 2 must rebuild the
graph node-by-node from ``AudioExecutionPlan.graph_segments``.  The transcription
risk -- "is the structured form faithful to the calibrated graph?" -- is retired
here in TEXT, deterministically, before any libav graph is built.  With the
interior pinned byte-identical, the perceptual Stage 0 gate only has to cover the
source/sink boundary.

The goldens in ``_audio_delivery_goldens.json`` were captured from the emitter
BEFORE the refactor.  If a legitimate graph change lands, regenerate them with:

    python -m tests._audio_delivery_capture

Main callers: pytest (experimental renderer job).
"""

from __future__ import annotations

import json
from pathlib import Path

from bladeworks.core.audio_execution import (
    AudioFilterNode,
    AudioGraphSegment,
    build_audio_execution_plan,
    serialize_segments,
)
from tests._audio_delivery_corpus import (
    CAPABILITY_REJECTED_CASES,
    structural_cases,
)


_GOLDEN_PATH = Path(__file__).with_name("_audio_delivery_goldens.json")


def _goldens() -> dict[str, str]:
    return json.loads(_GOLDEN_PATH.read_text())


def test_corpus_and_goldens_cover_exactly_the_same_cases() -> None:
    """Fail loudly if a case was added/removed without regenerating goldens.

    A silent mismatch would let a new (untested) graph shape slip through or an
    orphaned golden hide a deleted case -- neither is a silent failure we accept.
    """

    # Capability-rejected cases (e.g. surround output) have NO serialisable graph,
    # so they carry no golden; every other case must map 1:1 to a golden.
    corpus_names = {case.name for case in structural_cases()} - CAPABILITY_REJECTED_CASES
    golden_names = set(_goldens())
    assert corpus_names == golden_names, (
        "corpus/goldens drift -- regenerate goldens via "
        "_audio_delivery_capture; "
        f"corpus-only={sorted(corpus_names - golden_names)} "
        f"golden-only={sorted(golden_names - corpus_names)}"
    )


def test_structured_graph_serialises_to_head_goldens() -> None:
    """serialize_segments(graph_segments) == HEAD filter_complex, byte-for-byte."""

    goldens = _goldens()
    for case in structural_cases():
        if case.name in CAPABILITY_REJECTED_CASES:
            continue  # no renderable graph -- asserted to reject elsewhere
        execution = build_audio_execution_plan(case.plan, case.bindings)
        rebuilt = serialize_segments(execution.graph_segments)
        # The plan's own string and the structured re-serialisation must both
        # equal the pre-refactor golden -- the single producer, no drift.
        assert execution.filter_complex == goldens[case.name], case.name
        assert rebuilt == goldens[case.name], case.name


def test_filter_nodes_split_name_args_without_losing_bytes() -> None:
    """Every node's raw text == name[=args] -- proves chunk 2's split is safe.

    Chunk 2 builds one libav filter per node from ``name``/``args``.  That split
    is on the FIRST ``=`` only; here we confirm recombining reproduces ``raw``
    for every node the corpus emits (including args that themselves contain
    ``=`` like ``anullsrc=r=48000:cl=stereo`` and quoted ``volume`` expressions).
    """

    seen = 0
    for case in structural_cases():
        if case.name in CAPABILITY_REJECTED_CASES:
            continue  # no renderable graph -- asserted to reject elsewhere
        execution = build_audio_execution_plan(case.plan, case.bindings)
        for segment in execution.graph_segments:
            if not isinstance(segment, AudioGraphSegment):
                continue  # AudioRawSegment (retime splice) has no node split yet
            for node in segment.filters:
                assert isinstance(node, AudioFilterNode)
                recombined = node.name if not node.args else f"{node.name}={node.args}"
                assert recombined == node.raw, node.raw
                assert "=" not in node.name
                seen += 1
    assert seen > 0, "corpus produced no structured filter nodes"
