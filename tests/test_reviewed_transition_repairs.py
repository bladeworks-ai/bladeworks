"""Experimental regression boundaries for the August 2026 reviewed transition repairs.

Architecture map
================

authoritative human verdict + genuine Final Cut frame sequence
    -> one narrowly owned default expression
    -> structural assertion for the rejected semantic
    -> compact stock-FFmpeg render for source ownership and endpoint timing.

The broad cohort suites already prove registry dispatch and deterministic
execution.  These tests record *why* the six defaults changed, so a future
retune cannot quietly restore the rejected direction, one-card Clothesline,
always-closed Curtains, dense triangle grid, or early Reflection snap.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from bladeworks.transitions.cohort import (
    ARROWS_EXPRESSION,
    CLOCK_EXPRESSION,
    CURTAINS_EXPRESSION,
    SWING_EXPRESSION,
)
from bladeworks.transitions.panel_motion import (
    CLOTHESLINE_EXPRESSION,
    REFLECTION_EXPRESSION,
)


def _render_color_frames(expression: str) -> tuple[bytes, ...]:
    """Render one second of red-to-blue evidence through stock FFmpeg.

    Main callers:
    - The Clothesline and Curtains lifecycle tests below.

    Solid, asymmetric ownership colors make the three lifecycle states easy
    to distinguish: outgoing red, incoming blue, and renderer-owned void or
    curtain pixels.  Raw RGB output avoids container and codec variation.
    """

    width = 48
    height = 72
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=red:size={width}x{height}:rate=30:d=2",
            "-f",
            "lavfi",
            "-i",
            f"color=blue:size={width}x{height}:rate=30:d=2",
            "-filter_complex",
            (
                "[0:v][1:v]xfade=transition=custom:duration=1:offset=0:"
                f"expr='{expression}'"
            ),
            "-frames:v",
            "31",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")[-2000:]
    frame_size = width * height * 3
    assert len(completed.stdout) == 31 * frame_size
    return tuple(
        completed.stdout[index * frame_size : (index + 1) * frame_size]
        for index in range(31)
    )


def _color_fractions(frame: bytes) -> tuple[float, float, float]:
    pixels = len(frame) // 3
    red = blue = 0
    for offset in range(0, len(frame), 3):
        r, g, b = frame[offset : offset + 3]
        red += r > 200 and g < 80 and b < 80
        blue += b > 200 and r < 80 and g < 80
    red_fraction = red / pixels
    blue_fraction = blue / pixels
    return red_fraction, blue_fraction, 1 - red_fraction - blue_fraction


def test_rejected_swing_and_clock_directions_stay_reversed() -> None:
    assert "if(lt((1-P),0.38),0.18*" in SWING_EXPRESSION
    assert "0.28*(1-" in SWING_EXPRESSION
    assert "PI/2+atan2(Y-H/2,X-W/2)" in CLOCK_EXPRESSION


def test_arrows_remains_a_coarse_inward_object_field() -> None:
    assert "mod(X,(W/3))" in ARROWS_EXPRESSION
    assert "mod(Y,(H/4))" in ARROWS_EXPRESSION
    assert "-0.82*(1-" in ARROWS_EXPRESSION
    assert "0.82*" in ARROWS_EXPRESSION
    assert "min(255,0.88*" not in ARROWS_EXPRESSION


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_clothesline_has_outgoing_black_incoming_lifecycle() -> None:
    frames = _render_color_frames(CLOTHESLINE_EXPRESSION)
    start_red, start_blue, _ = _color_fractions(frames[0])
    gap_red, gap_blue, gap_void = _color_fractions(frames[5])
    incoming_red, incoming_blue, incoming_void = _color_fractions(frames[8])
    settled_red, settled_blue, _ = _color_fractions(frames[15])

    assert start_red > 0.95 and start_blue < 0.01
    assert gap_void > 0.90 and gap_red < 0.05 and gap_blue < 0.05
    assert incoming_blue > 0.05 and incoming_void > 0.05 and incoming_red < 0.05
    assert settled_blue > 0.95 and settled_red < 0.01


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_default_curtains_closes_holds_and_reopens() -> None:
    frames = _render_color_frames(CURTAINS_EXPRESSION)
    closing_red, closing_blue, _ = _color_fractions(frames[3])
    held_red, held_blue, _ = _color_fractions(frames[15])
    reopening_red, reopening_blue, _ = _color_fractions(frames[23])

    assert closing_red > 0.05 and closing_blue < 0.01
    assert held_red < 0.01 and held_blue < 0.01
    assert reopening_blue > 0.05 and reopening_red < 0.01


def test_reflection_keeps_both_turns_and_does_not_snap_at_eighty_percent() -> None:
    assert "(1-P)/0.36" in REFLECTION_EXPRESSION
    assert "((1-P)-0.66)/0.30" in REFLECTION_EXPRESSION
    assert "gte((1-P),0.96)" in REFLECTION_EXPRESSION
    assert "gte((1-P),0.82)" not in REFLECTION_EXPRESSION


def test_only_the_six_rejected_defaults_changed() -> None:
    """Freeze every approved custom expression across this repair wave.

    The authoritative review rejected exactly these six defaults.  Comparing
    the approved expressions to their pre-review hashes makes preservation a
    direct test instead of relying on visual spot checks of neighboring cases.
    """

    import hashlib

    from bladeworks.transitions.cohort import CUSTOM_IMPLEMENTATIONS

    reviewed = {
        "cohort_swing_default",
        "cohort_arrows_default",
        "cohort_clock_default",
        "cohort_curtains_default",
        "cohort_clothesline_default",
        "cohort_reflection_default",
    }
    approved_hashes = {
        "cohort_center_default": "31faf667b22858b13ad35dd06b61e448b1938a2ad2426ff3ce7e65a9fc9a847b",
        "cohort_deco_default": "e6dcda80e224054b890347d7fdafb6473c10329f0374940bb9037391602a70c8",
        "cohort_divide_default": "c2d837d1b105537ce7bd8bc625bc0d4ae195aec7f87b36d837fded7c8d749b43",
        "cohort_flash_default": "44bb6ca98acd29587d76ed9f1ae49fcc7f7a1724f1f161960a6eff14c6896f9c",
        "cohort_flip_default": "2dc64dacd1e9767bdccd0d435823aa4b44cca82047a18c1593bf2885e361b9f6",
        "cohort_lens_flare_default": "a349e174dc55f4dee11bf1d599aa4c45c3cc142941059083095f89b06113c459",
        "cohort_multi_flip_default": "5adc09d27456148b38d5af458443bd999e87a00ceae1a88eea2fcf428b8c2659",
        "cohort_page_curl_default": "b8d2192f92101df5b94f1911e8059996d2b3eb53d3ad3e39dbb35501c1acf25f",
        "cohort_pinwheel_default": "c02468b025f8ee707ef21b3e1748ae160675db43409141d28687e4ba73158db4",
        "cohort_rotate_default": "a7c2370c38e864e50d1a29ea158782ea31f2d86617d972d367dcc359433ee165",
        "cohort_scale_default": "cf16670de0833fc050d30bfb83472d6ca3d21d2b39407d19e79e7c82f173e1d6",
        "cohort_spin_default": "dda64538fb7446fd4ebff11eb33b443b88f155f43ae0e4142b666c8cbb79b5ef",
        "cohort_static_default": "18b44cc0766af65604c7b9fc852ed030ad1be466b12ff8537252685351d55077",
        "cohort_swap_default": "d4b4bc41fba9883cd3ecc26d09ad030815a583f24414f2c17323b0d5d04eb197",
        "cohort_switch_default": "e2a3eaceb9cda33f5605f42597ecf2ebe8ffb9b2798370ca9e9fe5245066f7b3",
        "cohort_veil_default": "c207335446bf6b62b0ff01858d3e208d4cbcada83cdab0d31156daa4216ab260",
    }
    current = {
        implementation_id: hashlib.sha256(expression.encode()).hexdigest()
        for implementation_id, expression in CUSTOM_IMPLEMENTATIONS.items()
        if implementation_id not in reviewed
    }
    for implementation_id, expected_hash in approved_hashes.items():
        assert current[implementation_id] == expected_hash
