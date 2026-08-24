"""Shared fixed-point primitives for the legacy-swscale ``initFilter`` ports.

Why this exists
---------------
``sampler.py`` (the calibrated bilinear whole-raster minifier) and
``fx_branched.py`` (the Callout effect's per-frame bicubic ``scale``) each port
FFmpeg n8.0.1 ``libswscale/utils.c`` ``initFilter`` for one resampling axis.
The two ports are the SAME algorithm with different *kernels*: swscale computes
the raw per-output taps from either a support-scaled triangle (bilinear) or a
Keys bicubic polynomial, then runs an identical fixed-point tail on them --
near-zero tap reduction, SIMD-width alignment, out-of-raster border folding, and
error-diffused normalisation to a power-of-two ``one``. That tail, plus the two
C-arithmetic division helpers it leans on, used to be transcribed three times
(``sampler.py``, ``fx_branched.py``, ``decode.py``). This module owns one copy.

The split is deliberate: each caller keeps its *kernel* visible (the triangle or
the cubic, with its own fone / size / position derivation) because that is the
part that genuinely differs; only the shared fixed-point tail lives here. The
tail is parameterised by two flags that capture the ONLY structural differences
between the two ports:

* ``apply_align_quirk`` -- swscale drops a size-1 vertical filter's alignment
  from 2 to 1 (the MMX / NEON "unscaled vertical" special case). The bicubic
  port hits this (its ``scale`` can be 1:1 on an axis); the bilinear minifier is
  strictly reducing on both axes and never does, so it passes ``False``.
* ``reform_truncates`` -- after alignment, swscale rebuilds each row to exactly
  the aligned width. The bicubic port transcribes that faithfully (truncating a
  row whose aligned width fell BELOW the built width). The bilinear port's
  transcription only ever *pads* (``row + [0] * (aligned - len(row))``, a no-op
  when ``aligned < len(row)``), so it keeps any tail taps. These differ only in
  the rare near-identity-large-reduction corner, but they are not the same
  operation, so each caller pins its own behaviour rather than sharing one.

Main callers:
- ``sampler._swscale_bilinear_filter`` (bilinear triangle taps).
- ``fx_branched.sws_bicubic_filter`` (Keys bicubic taps).

Numeric equivalence with the three pre-consolidation transcriptions is proven by
a standalone sweep harness (see the PR #2810 review notes for #8); every
``(src, dst)`` pair returns byte-identical positions and integer tap tables.
"""

from __future__ import annotations

# ``SWS_MAX_REDUCE_CUTOFF`` (libswscale/utils.c): the fraction of a tap's scale
# below which ``initFilter`` treats a coefficient as insignificant when trimming
# each row's near-zero head and tail.
SWS_MAX_REDUCE_CUTOFF = 0.002


def c_div(a: int, b: int) -> int:
    """C integer division: truncates toward zero. ``b`` may be negative.

    The three former copies differed only in the ``b == 0`` sign test
    (``b > 0`` vs ``b >= 0``), which is unreachable because ``abs(a) // abs(b)``
    raises ``ZeroDivisionError`` first; this canonical copy uses ``b > 0``.
    """

    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def rounded_div(a: int, b: int) -> int:
    """FFmpeg ``ROUNDED_DIV(a, b)`` for a positive ``b``: ``(a +- b/2) / b`` with C truncation."""

    return c_div(a + (b >> 1) if a >= 0 else a - (b >> 1), b)


def finalize_swscale_filter(
    positions: list[int],
    filt: list[list[int]],
    filter_size: int,
    fone: int,
    *,
    filter_align: int,
    one: int,
    apply_align_quirk: bool,
    reform_truncates: bool,
    src_len: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """The shared ``initFilter`` tail: reduce -> align -> reform -> border-fix -> normalise.

    Takes the raw per-output ``positions`` and ``filt`` rows a kernel just built
    (each row is ``filter_size`` fixed-point coefficients; ``fone`` is the
    kernel's 64-bit tap scale, ``one`` the output tap scale, e.g. ``1 << 14``),
    and returns ``(positions, taps)`` with every row summing to ``one``.

    The steps, in swscale order:

    1. Reduce: walking outputs from last to first (the pass is order-dependent),
       strip near-zero taps off the left (shifting the row and advancing its
       position, but never past the next output's position) and count near-zero
       taps on the right; ``min_filter_size`` is the widest surviving row.
    2. Align: round ``min_filter_size`` up to ``filter_align``. When
       ``apply_align_quirk`` is set, a size-1 row with alignment 2 drops the
       alignment to 1 first (the unscaled-vertical special case).
    3. Reform: rebuild every row to the aligned width -- truncating-or-padding
       when ``reform_truncates`` (bicubic), pad-only otherwise (bilinear).
    4. Border-fix: fold taps that point outside ``[0, src_len)`` back onto the
       nearest edge sample.
    5. Normalise: divide each row down to sum ``one`` with +-1 error diffusion.
    """

    dst_len = len(filt)
    filter2_size = filter_size
    cutoff_limit = SWS_MAX_REDUCE_CUTOFF * fone

    # (1) Reduce: strip near-zero left taps (shift) and count near-zero right
    # taps; the surviving width is the largest over all outputs. Last-to-first.
    min_filter_size = 0
    for i in range(dst_len - 1, -1, -1):
        row = filt[i]
        minimum = filter2_size
        cutoff = 0
        for _ in range(filter2_size):
            cutoff += abs(row[0])
            if float(cutoff) > cutoff_limit:
                break
            if i < dst_len - 1 and positions[i] >= positions[i + 1]:
                break
            for k in range(1, filter2_size):
                row[k - 1] = row[k]
            row[filter2_size - 1] = 0
            positions[i] += 1
        cutoff = 0
        for j in range(filter2_size - 1, 0, -1):
            cutoff += abs(row[j])
            if float(cutoff) > cutoff_limit:
                break
            minimum -= 1
        min_filter_size = max(min_filter_size, minimum)

    # (2) Align to the host SIMD width (with the size-1 vertical drop-to-1 quirk).
    if apply_align_quirk and min_filter_size == 1 and filter_align == 2:
        filter_align = 1
    assert min_filter_size > 0
    aligned = (min_filter_size + (filter_align - 1)) & ~(filter_align - 1)

    # (3) Reform every row to the aligned width.
    if reform_truncates:
        filt = [[row[j] if j < filter2_size else 0 for j in range(aligned)] for row in filt]
    else:
        filt = [row + [0] * (aligned - len(row)) for row in filt]

    # (4) Fold out-of-raster support onto the nearest edge sample.
    for i in range(dst_len):
        row = filt[i]
        if positions[i] < 0:
            for j in range(1, aligned):
                left = max(j + positions[i], 0)
                row[left] += row[j]
                row[j] = 0
            positions[i] = 0
        if positions[i] + aligned > src_len:
            shift = positions[i] + min(aligned - src_len, 0)
            acc = 0
            for j in range(aligned - 1, -1, -1):
                if positions[i] + j >= src_len:
                    acc += row[j]
                    row[j] = 0
            for j in range(aligned - 1, -1, -1):
                row[j] = 0 if j < shift else row[j - shift]
            positions[i] -= shift
            row[src_len - 1 - positions[i]] += acc
        assert 0 <= positions[i] < src_len

    # (5) Normalise to ``one`` with error diffusion.
    out: list[tuple[int, ...]] = []
    for row in filt:
        total = (sum(row) + one // 2) // one
        if total == 0:
            total = 1
        error = 0
        taps: list[int] = []
        for value in row:
            v = value + error
            int_v = rounded_div(v, total)
            taps.append(int_v)
            error = v - int_v * total
        out.append(tuple(taps))
    return tuple(positions), tuple(out)
