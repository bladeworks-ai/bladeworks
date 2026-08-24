"""G2 per-item golden harness for tensor ``xfade=custom`` transitions (F0 seam + F1 + F2 ids).

What it does
------------
``GOLDEN_CASES`` keys every admitted registry id -- flat ``xfade`` ids
(``transitions.ADMITTED_XFADE_IDS``) and 360° ids (``tr_equirect.ADMITTED_EQUIRECT_IDS``) --
by ``(id, parameter values)``: the compiler's DEFAULT parameter values for the id (resolved
from the capability registry through the compiler's own ``contract.resolve_parameter_values``
+ ``semantic_parameter_values`` path, aliases included) plus every structurally different
parameter branch the registry can select (Circle close vs open, Static styles, direction /
anchor / preset branches, Arrows end caps + motion blur, Curtains animations, Black Hole's one
accepted branch, 360° direction / speed+soften / border / slices variants).  The expression
text is never pasted: each case resolves through the SAME builder the plan uses
(``transitions.stock.build_stock_transition_plan`` /
``transitions.equirectangular.build_equirectangular_transition_plan``), so a golden can only
prove what the renderer runs.  For every case the harness:

1. writes two synthetic RGBA plates (gradients + shapes + semi-transparent alpha,
   ``test_tensor_expr.synthetic_plate``),
2. runs the CPU builder's exact filter shape through the ``ffmpeg`` CLI --
   ``[a]format=gbrap [b]format=gbrap -> xfade=transition=custom:duration=F/fps:offset=0:expr='...'``
   -- for ``F`` frames at a frame-exact timebase (so ``P = 1 - k/F`` in float32, see
   ``expr.xfade_progress``), capturing raw ``gbrap`` planes,
3. runs ``expr.xfade_custom_rgba`` on the same bytes for the same ``k``, and compares per frame.

Gates (per case ``tier``, none loosened beyond the documented class):

* **float64 CPU is bit-exact** for EVERY case and frame -- the semantics proof (grammar, GBRA
  plane order, endpoint guard, alpha plane through the expression, float32 ``P``, and -- since
  the F1 batch -- libm-faithful transcendentals: ``expr._libm_unary`` routes CPU float64
  ``sin/cos/exp/atan2/pow/...`` through numpy = libm, which is what decides Clock's ``0.001``
  solid edge and the 360° Circle Wipe's great-circle ties on exact pixel centres).
* tier ``exact`` (blends, mattes, light/deco): **float32 (the MPS runtime dtype) is within
  1 code** (truncation-boundary flips at mathematically exact integers, e.g. the nearest-sampler
  probe's ``b0*(1-P)+A*P`` with ``b0 == A``, see ``expr.quantize_uint8``).
* tier ``sampler`` (nearest-neighbour geometry -- inverse affine cards, page curl, pushes --
  and hard, unfeathered mattes -- great circle, pinwheel wedge, reflection cards): float32 is
  within 1 code except for **ties**: where the exact source coordinate (or matte edge) lands
  within float32 rounding of an integer / a pixel centre (identity maps written as ``X/W*W``,
  ``X - k/F*W`` translations, hinge rotations, ``lt`` edges) the double reference and float32
  pick neighbouring pixels / the other side, so a value is off by the local source gradient
  or the A/B difference.  Gate: at most ``SAMPLER_TIE_BUDGET`` (2%) of values may be off by more than 1
  code (measured worst: Clothesline right-to-left 1.75% -- its opening card is the identity
  map ``X/W*W`` for 3 of the 7 frames; most sampler cases are 0..1e-5).  Whole-frame ties
  (``k/F*W`` integral for every ``k``) would put entire frames one pixel off in float32; the
  harness runs ``FRAMES = 7`` (prime) so those systematic ties do not mask the per-pixel gate,
  and the runtime note lives in ``tensor/REFERENCE_DISCREPANCIES.md``-style reporting.
* tier ``float64`` (Static): the float32 field is a different noise realization (Static's
  ``sin`` hash needs double precision), so the renderer evaluates it on CPU float64
  (``transitions.FLOAT64_XFADE_IDS``); only the float64 gate applies and a float32 evaluation
  is asserted to be *far* off, so the tier cannot silently become unnecessary.

Coverage rules (tests enforce them): every admitted id has a default-parameter case; every
case id is admitted; probes are the only cases with pasted text (the ``PLANE`` literal
tripwire asserts the literal ``(158, 52, 218)`` in GBRA comes out as RGB ``(218, 158, 52)``).

Also here: the ``transitions.xfade_custom`` module contract on premultiplied linear sides
(k = 0 identity, alpha plane from the expression, unknown / un-admitted id rejects), the
Static float64 detour, and plan admission (Lens Flare + 360° Gaussian Blur admitted; 360° Bloom
rejected loudly).  End-to-end renders per registry family live in
``test_tensor_transition_golden_f1f2.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("av")
pytest.importorskip("PIL")

from bladeworks.core.capabilities import CapabilityRegistry  # noqa: E402
from bladeworks.core.compiler import compile_fcpxml  # noqa: E402
from tests.test_tensor_expr import (  # noqa: E402
    HEIGHT,
    WIDTH,
    diff_histogram,
    ffmpeg_binary,
    plate_tensor,
    read_gbrap_frames,
    run_ffmpeg,
    synthetic_plate,
    write_plate,
)
from bladeworks.core.model import Parameter  # noqa: E402
from bladeworks.tensor import TensorRenderUnsupported, build_tensor_plan  # noqa: E402
from bladeworks.tensor import expr, tr_equirect, transitions  # noqa: E402
from bladeworks.tensor.color import linearize  # noqa: E402
from bladeworks.transitions import cohort, panel_motion  # noqa: E402
from bladeworks.transitions import contract as transition_contract  # noqa: E402
from bladeworks.transitions import equirectangular as equirect_registry  # noqa: E402
from bladeworks.transitions.stock import build_stock_transition_plan  # noqa: E402

FPS = 25
FRAMES = 7  # prime: keeps k/F*W and k/F*H off exact integers (see module doc, tier ``sampler``)
SAMPLER_TIE_BUDGET = 0.02


# --------------------------------------------------------------------------- case table


@dataclass(frozen=True)
class GoldenCase:
    """One (id, parameter values) golden.

    ``overrides`` are applied over the compiler's default parameter values: exact registry
    keys (``cohort.*_KEY`` / ``panel_motion.*_KEY`` / ``"7"``) as FCPXML would author them
    for handler ``xfade``; semantic names (``direction`` / ``speed`` / ...) for handler
    ``equirectangular`` (the compiler hands the 360° builder semantic values,
    ``equirectangular.semantic_parameter_values``).
    """

    handler: str                              # "xfade" | "equirectangular" | "probe"
    xfade_id: str
    family: str
    overrides: Mapping[str, str] = field(default_factory=dict)
    tier: str = "exact"                       # "exact" | "sampler" | "float64"
    text: Optional[str] = None                # probes only (pasted synthetic text)


# Synthetic probes on top of the registry strings.
PLANE_LITERAL = "if(gte(P,1),A,if(lte(P,0),B,if(eq(PLANE,0),158,if(eq(PLANE,1),52,if(eq(PLANE,2),218,255)))))"
SAMPLER_PROBE = (
    "if(gte(P,1),A,if(lte(P,0),B,"
    "if(lt(P,0.5),a2(X+0.7,Y-3.2)*0.5+b1(W-1-X,Y*1.5)*0.5,"
    "if(eq(PLANE,3),a3(X,Y),b0(X*2-W/2,Y+0.5)*(1-P)+A*P))))"
)


def _x(xfade_id: str, family: str, tier: str = "exact", **overrides: str) -> GoldenCase:
    return GoldenCase("xfade", xfade_id, family, dict(overrides), tier)


def _e(xfade_id: str, tier: str = "exact", **overrides: str) -> GoldenCase:
    return GoldenCase("equirectangular", xfade_id, "equirect", dict(overrides), tier)


_K = {  # short aliases for the exact Final Cut serialization keys the compiler resolves
    "center_dir": cohort.CENTER_DIRECTION_KEY, "center_edge": cohort.CENTER_EDGE_TYPE_KEY,
    "clock_dir": cohort.CLOCK_DIRECTION_KEY, "clock_edge": cohort.CLOCK_EDGE_TYPE_KEY,
    "curl_preset": cohort.PAGE_CURL_PRESET_KEY, "curl_dir": cohort.PAGE_CURL_DIRECTION_KEY,
    "swap_dir": cohort.SWAP_DIRECTION_KEY, "static_style": cohort.STATIC_STYLE_KEY,
    "rotate_dir": cohort.ROTATE_DIRECTION_KEY, "rotate_black": cohort.ROTATE_BACKGROUND_KEY,
    "swing_anchor": cohort.SWING_ANCHOR_KEY, "swing_dir": cohort.SWING_DIRECTION_KEY,
    "swing_black": cohort.SWING_BACKGROUND_KEY, "switch_dir": cohort.SWITCH_DIRECTION_KEY,
    "arrows_cap": cohort.ARROWS_END_CAP_KEY, "arrows_blur": cohort.ARROWS_MOTION_BLUR_KEY,
    "curtains": cohort.CURTAINS_ANIMATION_KEY,
    "divide_sections": panel_motion.DIVIDE_SECTIONS_KEY, "clothesline_dir": panel_motion.CLOTHESLINE_DIRECTION_KEY,
    "spin_dir": panel_motion.SPIN_DIRECTION_KEY, "flip_dir": panel_motion.FLIP_DIRECTION_KEY,
    "pinwheel_black": panel_motion.PINWHEEL_BACKGROUND_KEY, "reflection_dir": panel_motion.REFLECTION_DIRECTION_KEY,
    "scale_dir": panel_motion.SCALE_DIRECTION_KEY,
}


def _kv(**named: str) -> dict[str, str]:
    return {_K[name]: value for name, value in named.items()}


GOLDEN_CASES: dict[str, GoldenCase] = {
    # ---- probes -----------------------------------------------------------------
    "probe_plane_literal": GoldenCase("probe", "probe_plane_literal", "probe", text=PLANE_LITERAL),
    "probe_nearest_samplers": GoldenCase("probe", "probe_nearest_samplers", "probe", text=SAMPLER_PROBE),
    # ---- light / deco (light_deco.CUSTOM_IMPLEMENTATIONS) ------------------------
    "cohort_bloom_default": _x("cohort_bloom_default", "light_deco"),
    "cohort_flash_default": _x("cohort_flash_default", "light_deco"),
    "cohort_lens_flare_default": _x("cohort_lens_flare_default", "light_deco"),
    "cohort_deco_default": _x("cohort_deco_default", "light_deco"),
    # ---- cohort dynamic (cohort.build_cohort_transition_plan) --------------------
    "cohort_center_default": _x("cohort_center_default", "cohort_dynamic"),                    # Automatic -> Open, feather
    "cohort_center_close": _x("cohort_center_default", "cohort_dynamic", **_kv(center_dir="1")),
    "cohort_center_solid": _x("cohort_center_default", "cohort_dynamic", **_kv(center_edge="0")),
    "cohort_center_close_solid": _x("cohort_center_default", "cohort_dynamic", **_kv(center_dir="1", center_edge="0")),
    "cohort_clock_default": _x("cohort_clock_default", "cohort_dynamic"),
    "cohort_clock_ccw": _x("cohort_clock_default", "cohort_dynamic", **_kv(clock_dir="1")),
    "cohort_clock_solid": _x("cohort_clock_default", "cohort_dynamic", **_kv(clock_edge="0")),
    "cohort_clock_ccw_solid": _x("cohort_clock_default", "cohort_dynamic", **_kv(clock_dir="1", clock_edge="0")),
    "cohort_page_curl_default": _x("cohort_page_curl_default", "cohort_dynamic"),              # Right, Automatic -> Open
    "cohort_page_curl_close": _x("cohort_page_curl_default", "cohort_dynamic", "sampler", **_kv(curl_dir="1")),
    "cohort_page_curl_left": _x("cohort_page_curl_default", "cohort_dynamic", **_kv(curl_preset="1")),
    "cohort_page_curl_left_close": _x("cohort_page_curl_default", "cohort_dynamic", "sampler", **_kv(curl_preset="1", curl_dir="1")),
    "cohort_swap_default": _x("cohort_swap_default", "cohort_dynamic", "sampler"),             # Right
    "cohort_swap_left": _x("cohort_swap_default", "cohort_dynamic", "sampler", **_kv(swap_dir="0")),
    "cohort_static_default": _x("cohort_static_default", "cohort_dynamic", "float64"),          # Style A
    "cohort_static_style_b": _x("cohort_static_default", "cohort_dynamic", "float64", **_kv(static_style="1")),
    "cohort_rotate_default": _x("cohort_rotate_default", "cohort_dynamic", "sampler"),
    "cohort_rotate_ccw": _x("cohort_rotate_default", "cohort_dynamic", "sampler", **_kv(rotate_dir="1")),
    "cohort_rotate_black": _x("cohort_rotate_default", "cohort_dynamic", "sampler", **_kv(rotate_black="1")),
    "cohort_rotate_ccw_black": _x("cohort_rotate_default", "cohort_dynamic", "sampler", **_kv(rotate_dir="1", rotate_black="1")),
    "cohort_swing_default": _x("cohort_swing_default", "cohort_dynamic", "sampler"),           # Top, Away, no black
    "cohort_swing_top_towards": _x("cohort_swing_default", "cohort_dynamic", "sampler", **_kv(swing_dir="0")),
    "cohort_swing_right": _x("cohort_swing_default", "cohort_dynamic", "sampler", **_kv(swing_anchor="0")),
    "cohort_swing_left_towards_black": _x("cohort_swing_default", "cohort_dynamic", "sampler", **_kv(swing_anchor="1", swing_dir="0", swing_black="1")),
    "cohort_swing_bottom": _x("cohort_swing_default", "cohort_dynamic", "sampler", **_kv(swing_anchor="3")),
    "cohort_swing_top_black": _x("cohort_swing_default", "cohort_dynamic", "sampler", **_kv(swing_black="1")),
    "cohort_switch_default": _x("cohort_switch_default", "cohort_dynamic", "sampler"),         # From Left
    "cohort_switch_from_right": _x("cohort_switch_default", "cohort_dynamic", "sampler", **_kv(switch_dir="2")),
    "cohort_arrows_default": _x("cohort_arrows_default", "cohort_dynamic"),                    # Arrow cap
    "cohort_arrows_round": _x("cohort_arrows_default", "cohort_dynamic", **_kv(arrows_cap="3")),
    "cohort_arrows_square": _x("cohort_arrows_default", "cohort_dynamic", **_kv(arrows_cap="4")),
    "cohort_arrows_none": _x("cohort_arrows_default", "cohort_dynamic", **_kv(arrows_cap="5")),
    "cohort_arrows_bevel_blur": _x("cohort_arrows_default", "cohort_dynamic", **_kv(arrows_cap="6", arrows_blur="1")),
    "cohort_arrows_motion_blur": _x("cohort_arrows_default", "cohort_dynamic", **_kv(arrows_blur="1")),
    "cohort_curtains_default": _x("cohort_curtains_default", "cohort_dynamic"),                # Open & Close
    "cohort_curtains_open_only": _x("cohort_curtains_default", "cohort_dynamic", **_kv(curtains="1")),
    "cohort_curtains_close_only": _x("cohort_curtains_default", "cohort_dynamic", **_kv(curtains="2")),
    "cohort_veil_default": _x("cohort_veil_default", "cohort_dynamic"),
    # ---- panel motion (panel_motion.build_panel_motion_expression) ---------------
    "cohort_divide_default": _x("cohort_divide_default", "panel_motion", "sampler"),          # 4 sections
    "cohort_divide_three": _x("cohort_divide_default", "panel_motion", "sampler", **_kv(divide_sections="1")),
    "cohort_divide_two": _x("cohort_divide_default", "panel_motion", "sampler", **_kv(divide_sections="2")),
    "cohort_spin_default": _x("cohort_spin_default", "panel_motion", "sampler"),              # Automatic -> In
    "cohort_spin_out": _x("cohort_spin_default", "panel_motion", "sampler", **_kv(spin_dir="2")),
    "cohort_clothesline_default": _x("cohort_clothesline_default", "panel_motion", "sampler"),
    "cohort_clothesline_ltr": _x("cohort_clothesline_default", "panel_motion", "sampler", **_kv(clothesline_dir="1")),
    "cohort_flip_default": _x("cohort_flip_default", "panel_motion", "sampler"),              # Right
    "cohort_flip_left": _x("cohort_flip_default", "panel_motion", "sampler", **_kv(flip_dir="1")),
    "cohort_flip_up": _x("cohort_flip_default", "panel_motion", "sampler", **_kv(flip_dir="2")),
    "cohort_flip_down": _x("cohort_flip_default", "panel_motion", "sampler", **_kv(flip_dir="3")),
    "cohort_scale_default": _x("cohort_scale_default", "panel_motion"),                        # Up
    "cohort_scale_down": _x("cohort_scale_default", "panel_motion", **_kv(scale_dir="1")),
    "cohort_scale_in": _x("cohort_scale_default", "panel_motion", **_kv(scale_dir="2")),
    "cohort_scale_out": _x("cohort_scale_default", "panel_motion", **_kv(scale_dir="3")),
    "cohort_multi_flip_default": _x("cohort_multi_flip_default", "panel_motion", "sampler"),
    "cohort_pinwheel_default": _x("cohort_pinwheel_default", "panel_motion", "sampler"),      # hard atan2 wedge matte
    "cohort_pinwheel_black": _x("cohort_pinwheel_default", "panel_motion", "sampler", **_kv(pinwheel_black="1")),
    "cohort_reflection_default": _x("cohort_reflection_default", "panel_motion", "sampler"),  # From Left; hard card mattes
    "cohort_reflection_from_right": _x("cohort_reflection_default", "panel_motion", "sampler", **_kv(reflection_dir="1")),
    # ---- stock / Circle / Black Hole -------------------------------------------
    "fall_default": _x("fall_default", "stock", "sampler"),
    "squares_tile_reveal_default": _x("squares_tile_reveal_default", "stock"),
    "circle_default": _x("circle_default", "circle"),                                          # open
    "circle_close": _x("circle_default", "circle", **{"7": "1"}),
    "black_hole_default": _x("black_hole_default", "black_hole"),                              # the only accepted branch
    # ---- 360° (equirectangular.build_equirectangular_transition_plan) -----------
    "equirect_circle_wipe_default": _e("equirect_circle_wipe", "sampler"),                    # hard great-circle matte
    "equirect_circle_wipe_ease_both": _e("equirect_circle_wipe", "sampler", speed="3"),        # Soften Edges is not an authorable key here
    "equirect_circle_wipe_border": _e("equirect_circle_wipe", "sampler", border="1"),
    "equirect_divide_default": _e("equirect_divide", "sampler"),                              # East & West, 3 bands
    "equirect_divide_west": _e("equirect_divide", "sampler", direction="1"),
    "equirect_divide_east_soft_ease": _e("equirect_divide", "sampler", direction="0", speed="5", soften_edges="0.4"),
    "equirect_divide_six_slices": _e("equirect_divide", "sampler", slices="1", spacing="0.2"),
    "equirect_push_default": _e("equirect_push", "sampler"),                                  # East
    "equirect_push_west": _e("equirect_push", "sampler", direction="1"),
    "equirect_push_soft_ease": _e("equirect_push", "sampler", soften_edges="0.5", speed="3"),
    "equirect_reveal_wipe_default": _e("equirect_reveal_wipe"),
    "equirect_reveal_wipe_soft_ease": _e("equirect_reveal_wipe", soften_edges="0.5", speed="3"),
    "equirect_reveal_wipe_border": _e("equirect_reveal_wipe", border="1"),
    "equirect_slide_default": _e("equirect_slide", "sampler"),
    "equirect_slide_west": _e("equirect_slide", "sampler", direction="1"),
    "equirect_slide_soft_ease": _e("equirect_slide", "sampler", soften_edges="0.5", speed="3"),
    "equirect_wipe_default": _e("equirect_wipe", "sampler"),                                  # 3.4x zoomed outgoing
    "equirect_wipe_west": _e("equirect_wipe", "sampler", direction="1"),
    "equirect_wipe_soft_ease": _e("equirect_wipe", "sampler", soften_edges="0.5", speed="3"),
    "equirect_wipe_border": _e("equirect_wipe", "sampler", border="1"),
}


# --------------------------------------------------------------------------- resolution (the compiler's path)


def _registry_entry(handler: str, xfade_id: str):
    matches = [
        entry for entry in CapabilityRegistry.load().entries
        if entry.kind == "transition" and entry.handler == handler and (entry.xfade or {}).get("id") == xfade_id
    ]
    assert len(matches) == 1, f"expected exactly one {handler} capability with xfade id {xfade_id!r}, got {len(matches)}"
    return matches[0]


def _authored(overrides: Mapping[str, str]) -> tuple[Parameter, ...]:
    return tuple(Parameter(name=None, key=key, value=value) for key, value in overrides.items())


def compiler_parameter_values(case: GoldenCase) -> dict[str, Any]:
    """``RenderTransition.parameter_values`` as ``compiler._compile_transition`` would resolve them
    for ``case.overrides`` authored as ``<param key=... value=...>`` (defaults + aliases included)."""

    entry = _registry_entry(case.handler, case.xfade_id)
    if case.handler == "xfade":
        specs = transition_contract.parse_parameter_specs(entry.parameters, max_slots=None)
        resolved = transition_contract.resolve_parameter_values(specs, _authored(case.overrides))
        return transition_contract.semantic_parameter_values(specs, resolved)
    specs = equirect_registry.parse_equirectangular_parameter_specs(entry.parameters)
    exact_keys = equirect_registry.EQUIRECTANGULAR_PARAMETER_KEYS.get(case.xfade_id, {})
    authored = _authored({exact_keys[name]: value for name, value in case.overrides.items()})
    resolved = equirect_registry.resolve_equirectangular_parameter_values(specs, authored)
    return equirect_registry.semantic_parameter_values(case.xfade_id, resolved)


def resolve_case_expression(case: GoldenCase) -> str:
    """The expression text the plan would evaluate for this case (through the plan's own resolvers)."""

    if case.handler == "probe":
        assert case.text is not None
        return case.text
    values = compiler_parameter_values(case)
    if case.handler == "xfade":
        return transitions.resolve_xfade_expression(case.xfade_id, values)
    return tr_equirect.resolve_equirectangular_expression(case.xfade_id, values)


# --------------------------------------------------------------------------- ffmpeg reference


@pytest.fixture(scope="module")
def plates(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    directory = tmp_path_factory.mktemp("xfade")
    a, b = synthetic_plate(1), synthetic_plate(2)
    return write_plate(directory / "a.png", a), write_plate(directory / "b.png", b), a, b


def ffmpeg_xfade_custom(a_png: Path, b_png: Path, expression: str, raw: Path, *, frames: int = FRAMES, fps: int = FPS) -> torch.Tensor:
    """The CPU builder's xfade shape on two looped PNGs -> ``[frames, 4, H, W]`` RGBA code values."""

    duration = frames / fps
    graph = (
        "[0:v]format=gbrap[a];[1:v]format=gbrap[b];"
        f"[a][b]xfade=transition=custom:duration={duration}:offset=0:expr='{expression}',format=gbrap"
    )
    run_ffmpeg([
        "-loop", "1", "-framerate", str(fps), "-t", str(duration), "-i", str(a_png),
        "-loop", "1", "-framerate", str(fps), "-t", str(duration), "-i", str(b_png),
        "-filter_complex", graph, "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "gbrap", str(raw),
    ])
    out = read_gbrap_frames(raw, WIDTH, HEIGHT)
    assert out.shape[0] == frames, f"ffmpeg produced {out.shape[0]} frames, expected {frames}"
    return out


def _run_all_frames(parsed: expr.Expr, a: torch.Tensor, b: torch.Tensor, reference: torch.Tensor) -> dict[int, int]:
    histogram: dict[int, int] = {}
    for k in range(FRAMES):
        got = expr.xfade_custom_rgba(parsed, a, b, progress=expr.xfade_progress(k, FRAMES))
        for value, count in diff_histogram(got, reference[k]).items():
            histogram[value] = histogram.get(value, 0) + count
    return dict(sorted(histogram.items()))


# --------------------------------------------------------------------------- coverage rules


# Admitted ids that are NOT ``xfade=custom`` expressions in the reference (their own kind, own
# golden): the 360° Gaussian Blur graph (``tr_equirect.gaussian_blur_side``, goldens in
# test_tensor_transition_golden_f1f2.py).
NON_EXPRESSION_ADMITTED: frozenset[tuple[str, str]] = frozenset({("equirectangular", "equirect_gaussian_blur")})


def _admitted_ids() -> set[tuple[str, str]]:
    admitted = {("xfade", item) for item in transitions.ADMITTED_XFADE_IDS} | {
        ("equirectangular", item) for item in tr_equirect.ADMITTED_EQUIRECT_IDS
    }
    assert NON_EXPRESSION_ADMITTED <= admitted
    return admitted - NON_EXPRESSION_ADMITTED


def test_every_admitted_id_has_a_default_golden_and_every_case_is_admitted() -> None:
    defaults = {(case.handler, case.xfade_id) for case in GOLDEN_CASES.values() if case.handler != "probe" and not case.overrides}
    missing = sorted(_admitted_ids() - defaults)
    assert not missing, f"admitted ids without a default-parameter golden: {missing}"
    stray = sorted({(case.handler, case.xfade_id) for case in GOLDEN_CASES.values() if case.handler != "probe"} - _admitted_ids())
    assert not stray, f"golden cases for ids that are not admitted: {stray}"
    assert transitions.FLOAT64_XFADE_IDS <= set(transitions.ADMITTED_XFADE_IDS)
    for label, case in GOLDEN_CASES.items():
        if case.handler == "xfade":
            expected = "float64" if case.xfade_id in transitions.FLOAT64_XFADE_IDS else None
            assert expected is None or case.tier == expected, f"{label}: tier {case.tier!r} vs FLOAT64_XFADE_IDS"


def test_every_pure_custom_registry_id_is_admitted() -> None:
    """Every capability the compiler can emit as a pure custom expression (mode custom, no
    prefilter, at its default parameters) is admitted -- so a new registry id is a loud test
    failure here, not a silently rejected transition."""

    pure: set[tuple[str, str]] = set()
    for entry in CapabilityRegistry.load().entries:
        if entry.kind != "transition" or entry.handler not in {"xfade", "equirectangular"}:
            continue
        xfade_id = str(entry.xfade["id"])
        case = GoldenCase(entry.handler, xfade_id, "?")
        values = compiler_parameter_values(case)
        plan = (
            build_stock_transition_plan(xfade_id, values) if entry.handler == "xfade"
            else equirect_registry.build_equirectangular_transition_plan(xfade_id, values)
        )
        if plan.mode == "custom" and plan.expression is not None and plan.prefilter is None:
            pure.add((entry.handler, xfade_id))
    assert pure == _admitted_ids(), f"pure-custom registry ids vs admitted: {sorted(pure ^ _admitted_ids())}"


def test_variant_cases_resolve_to_distinct_branches() -> None:
    """Every override case selects a *structurally different* expression from its id's default."""

    by_id: dict[tuple[str, str], dict[str, str]] = {}
    for label, case in GOLDEN_CASES.items():
        if case.handler != "probe":
            by_id.setdefault((case.handler, case.xfade_id), {})[label] = resolve_case_expression(case)
    for key, texts in by_id.items():
        assert len(set(texts.values())) == len(texts), f"{key}: cases resolve to the same expression: {sorted(texts)}"


# --------------------------------------------------------------------------- the golden


@pytest.mark.parametrize("label", sorted(GOLDEN_CASES))
def test_xfade_custom_golden(label: str, plates: tuple[Path, Path, np.ndarray, np.ndarray], tmp_path: Path) -> None:
    case = GOLDEN_CASES[label]
    a_png, b_png, a_plate, b_plate = plates
    text = resolve_case_expression(case)
    reference = ffmpeg_xfade_custom(a_png, b_png, text, tmp_path / f"{label}.raw")
    parsed = expr.parse(text)
    total = FRAMES * 4 * HEIGHT * WIDTH

    histogram64 = _run_all_frames(parsed, plate_tensor(a_plate, torch.float64), plate_tensor(b_plate, torch.float64), reference)
    print(f"xfade {label} float64: {histogram64}")
    assert max(histogram64) == 0, f"{label} float64 (CPU, libm-faithful) is not bit-exact: {histogram64}"

    histogram32 = _run_all_frames(parsed, plate_tensor(a_plate, torch.float32), plate_tensor(b_plate, torch.float32), reference)
    beyond_one = sum(count for value, count in histogram32.items() if value > 1) / total
    print(f"xfade {label} float32 ({case.tier}): {histogram32} beyond-1-code fraction {beyond_one:.2e}")
    if case.tier == "exact":
        assert max(histogram32) <= 1, f"{label} float32: |diff| histogram {histogram32} exceeds 1 code"
    elif case.tier == "sampler":
        assert beyond_one <= SAMPLER_TIE_BUDGET, (
            f"{label} float32: {beyond_one:.3%} of values beyond 1 code exceeds the sampling-tie budget "
            f"{SAMPLER_TIE_BUDGET:.0%}: {histogram32}"
        )
    elif case.tier == "float64":
        # The float32 field is a different noise realization: assert it stays far off so the
        # CPU float64 detour (transitions.FLOAT64_XFADE_IDS) cannot silently become unnecessary.
        assert beyond_one > 0.25, f"{label}: float32 unexpectedly close ({beyond_one:.3%} beyond 1 code); revisit FLOAT64_XFADE_IDS"
    else:
        raise AssertionError(f"unknown tier {case.tier!r}")

    if label == "probe_plane_literal":
        # GBRA tripwire: literal planes (0,1,2,3) = (158,52,218,255) must land as RGB (218,158,52).
        got = expr.xfade_custom_rgba(parsed, plate_tensor(a_plate, torch.float32), plate_tensor(b_plate, torch.float32), progress=0.5)
        assert got[:, 0, 0].tolist() == [218.0, 158.0, 52.0, 255.0]
        assert reference[3][:, 0, 0].tolist() == [218.0, 158.0, 52.0, 255.0]


# --------------------------------------------------------------------------- module contract


def _premultiplied_linear(plate: np.ndarray) -> torch.Tensor:
    code = plate_tensor(plate, torch.float32) / 255.0
    alpha = code[3:4]
    return torch.cat((linearize(code[:3]) * alpha, alpha), dim=0)


def test_xfade_custom_module_contract(plates: tuple[Path, Path, np.ndarray, np.ndarray]) -> None:
    _, _, a_plate, b_plate = plates
    a, b = _premultiplied_linear(a_plate), _premultiplied_linear(b_plate)
    # k = 0 (P = 1): the outgoing side, untouched (no code-space round trip).
    assert transitions.xfade_custom(a, b, xfade_id="cohort_flash_default", frame_index=0, frame_count=8) is a
    # Mid-flash: the expression yields 255 on every plane, alpha included -> opaque white,
    # even where both sides were semi-transparent (ffmpeg runs PLANE 3 through the expression).
    mid = transitions.xfade_custom(a, b, xfade_id="cohort_flash_default", frame_index=4, frame_count=8)
    assert mid.shape == a.shape and torch.allclose(mid, torch.ones_like(mid))
    # Lens Flare mid-frame: finite, premultiplied (rgb <= alpha), and not either side.
    flare = transitions.xfade_custom(a, b, xfade_id="cohort_lens_flare_default", frame_index=3, frame_count=8)
    assert torch.isfinite(flare).all()
    assert (flare[:3] <= flare[3:4] + 1e-6).all()
    assert not torch.allclose(flare, a) and not torch.allclose(flare, b)
    # Static takes the CPU float64 detour and comes back on the sides' device / dtype.
    static = transitions.xfade_custom(a, b, xfade_id="cohort_static_default", frame_index=3, frame_count=8)
    assert static.dtype == a.dtype and static.device == a.device and static.shape == a.shape
    assert torch.isfinite(static).all()
    if torch.backends.mps.is_available():
        # The detour hops device before dtype (MPS refuses float64 even in transit) and lands back on MPS.
        on_mps = transitions.xfade_custom(a.to("mps"), b.to("mps"), xfade_id="cohort_static_default", frame_index=3, frame_count=8)
        assert on_mps.device.type == "mps" and on_mps.dtype == torch.float32
        assert torch.allclose(on_mps.cpu(), static, atol=1e-6)
    with pytest.raises(TensorRenderUnsupported, match="not ported"):
        transitions.xfade_expression("no_such_xfade")
    with pytest.raises(TensorRenderUnsupported, match="not ported"):
        tr_equirect.resolve_equirectangular_expression("equirect_bloom_default")
    with pytest.raises(TensorRenderUnsupported, match="not a pure custom expression"):
        tr_equirect.resolve_equirectangular_expression("equirect_gaussian_blur")  # its own kind, not expression text
    with pytest.raises(TensorRenderUnsupported, match="not ported"):
        transitions.xfade_expression("equirect_push")  # a 360° id is not a flat xfade id


def test_lowered_payload_matches_the_reference_resolution() -> None:
    """``_lower_xfade`` / ``_lower_equirectangular`` hand the apply port exactly the text the CPU
    builder resolves for the same ``RenderTransition.parameter_values`` (Static flagged float64)."""

    from fractions import Fraction

    from bladeworks.core.model import RenderTransition

    def item(handler: str, xfade_id: str, values: Mapping[str, Any]) -> RenderTransition:
        return RenderTransition(
            path="/t", absolute_start=Fraction(0), duration=Fraction(1), uid=None, name=xfade_id,
            handler=handler, params=(), xfade_id=xfade_id, parameter_values=dict(values),
        )

    ctx = transitions.LowerContext(width=WIDTH, height=HEIGHT, frame_duration=Fraction(1, 25), frame_count=FRAMES)
    for label, case in GOLDEN_CASES.items():
        if case.handler == "probe":
            continue
        values = compiler_parameter_values(case)
        lowered = transitions.lower_transition(item(case.handler, case.xfade_id, values), ctx)
        assert lowered.kind == "xfade_custom" and lowered.xfade_id == case.xfade_id, label
        assert lowered.payload.expression == resolve_case_expression(case), label
        assert lowered.payload.float64 == (case.tier == "float64"), label


# --------------------------------------------------------------------------- plan admission

_MOTION_UIDS = {
    "Lens Flare": ".../Transitions.localized/Lights.localized/Lens Flare.localized/Lens Flare.motr",
    "360° Gaussian Blur": ".../Transitions.localized/360°.localized/360° Gaussian Blur.localized/360° Gaussian Blur.motr",
    "360° Bloom": ".../Transitions.localized/360°.localized/360° Bloom.localized/360° Bloom.motr",
}


def _media(directory: Path, name: str, source: str) -> Path:
    path = directory / f"{name}.mp4"
    if not path.exists():
        run_ffmpeg(["-f", "lavfi", "-i", f"{source}:s={WIDTH}x{HEIGHT}:r=24:d=3", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-preset", "veryfast", str(path)])
    return path


def _project(directory: Path, transition_name: str) -> Path:
    a = _media(directory, "a", "gradients=seed=1:speed=0.02:c0=red:c1=teal:c2=yellow:c3=purple")
    b = _media(directory, "b", "gradients=seed=7:speed=0.05:c0=orange:c1=navy:c2=white:c3=black")
    uid = _MOTION_UIDS[transition_name]
    source = directory / f"{transition_name.replace(' ', '_').replace('°', '')}.fcpxml"
    source.write_text(f'''<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
<resources>
  <format id="fmt" frameDuration="1/24s" width="{WIDTH}" height="{HEIGHT}" colorSpace="1-1-1 (Rec. 709)"/>
  <asset id="a" start="0s" duration="3s" hasVideo="1" hasAudio="0" format="fmt"><media-rep kind="original-media" src="{a.as_uri()}"/></asset>
  <asset id="b" start="0s" duration="3s" hasVideo="1" hasAudio="0" format="fmt"><media-rep kind="original-media" src="{b.as_uri()}"/></asset>
  <effect id="tr" name="{transition_name}" uid="{uid}"/>
</resources>
<library><event name="t"><project name="t">
<sequence format="fmt" duration="3s"><spine>
  <asset-clip ref="a" offset="0s" start="0s" duration="2s"/>
  <transition name="{transition_name}" offset="3/2s" duration="1s"><filter-video ref="tr" name="{transition_name}"/></transition>
  <asset-clip ref="b" offset="2s" start="1/2s" duration="1s"/>
</spine></sequence></project></event></library></fcpxml>''', encoding="utf-8")
    return source


def test_plan_admits_lens_flare_and_gaussian_and_rejects_360_bloom(tmp_path: Path) -> None:
    ffmpeg_binary()
    plan = build_tensor_plan(compile_fcpxml(_project(tmp_path, "Lens Flare")).render)
    (transition,) = plan.transitions
    assert transition.kind == "xfade_custom" and transition.xfade_id == "cohort_lens_flare_default"
    assert transition.frame_count == 24
    (gaussian,) = build_tensor_plan(compile_fcpxml(_project(tmp_path, "360° Gaussian Blur")).render).transitions
    assert gaussian.kind == "equirect_gaussian_blur" and gaussian.xfade_id == "equirect_gaussian_blur"
    assert gaussian.payload.duration_text == "1" and gaussian.payload.ownership_expression == tr_equirect.gaussian_ownership_expression()
    with pytest.raises(TensorRenderUnsupported, match=r"bloom graph.*transition \(other\)"):
        build_tensor_plan(compile_fcpxml(_project(tmp_path, "360° Bloom")).render)
