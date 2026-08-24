"""Regenerate the audio-delivery golden ``filter_complex`` fixtures.

Run as a module from the repo root:

    python -m tests._audio_delivery_capture

Writes ``_audio_delivery_goldens.json`` next to this file: one calibrated
``filter_complex`` string per structural corpus case.  Regenerate ONLY when a
deliberate, reviewed graph change lands -- the whole point of the Stage 2a gate
is that the structured emitter reproduces these byte-for-byte, so a diff here is
a real behaviour change that must be explained in the PR.

Why a module (not a test): capturing must never run inside the gate that checks
against the capture, or the gate would tautologically pass.
"""

from __future__ import annotations

import json
from pathlib import Path

from bladeworks.core.audio_execution import build_audio_execution_plan
from tests._audio_delivery_corpus import (
    structural_cases,
)

_GOLDEN_PATH = Path(__file__).with_name("_audio_delivery_goldens.json")


def capture() -> dict[str, str]:
    goldens: dict[str, str] = {}
    for case in structural_cases():
        execution = build_audio_execution_plan(case.plan, case.bindings)
        goldens[case.name] = execution.filter_complex
    return goldens


def main() -> None:
    goldens = capture()
    _GOLDEN_PATH.write_text(json.dumps(goldens, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(goldens)} goldens -> {_GOLDEN_PATH}")


if __name__ == "__main__":
    main()
