"""The exact text format every backend must use for FFmpeg numbers and durations.

Why this exists
---------------
The tensor renderer is measured against the FFmpeg emitter's *emitted text*: an
expression string that differs in one digit produces different pixels, so these two
formatters are a **shared contract**, not an emitter implementation detail.  That is
why they live in ``core`` rather than ``legacy_ffmpeg``.

Before the package split they were private helpers inside ``ffmpeg.py`` --
``_seconds`` duplicated verbatim in ``composition_cpu_emitter.py`` -- and the tensor
port reached across the backend boundary to import them by their underscore names.
One definition, imported by both backends, is what keeps the two from drifting.

Main callers:
- ``legacy_ffmpeg.ffmpeg`` and ``legacy_ffmpeg.composition_cpu_emitter``, which alias
  these to their historical ``_number`` / ``_seconds`` names.
- ``tensor.fx_basic``, ``tensor.tr_equirect``.
"""

from __future__ import annotations

from fractions import Fraction


def format_number(value: float) -> str:
    """Format one scalar for an FFmpeg filter argument or expression.

    A value within 1e-9 of an integer prints as that integer with no decimal point;
    anything else prints at 8 decimal places with trailing zeros -- and then any bare
    trailing point -- stripped.
    """

    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def format_seconds(value: Fraction) -> str:
    """Format an AVOption duration; FFmpeg expressions accept fractions, durations do not.

    A whole number of seconds prints as a bare integer; anything else prints at 12
    decimal places, trailing zeros stripped.
    """

    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.12f}".rstrip("0").rstrip(".")
