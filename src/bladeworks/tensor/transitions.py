"""Transition ports: Cross Dissolve, registry ``xfade=custom`` expressions, and the handler registry.

Architecture map
================

    plan.build_tensor_plan
        -> lower_transition(item, LowerContext) -> Lowered(kind, xfade_id, payload)
             handler "cross_dissolve"  -> kind "cross_dissolve"
             handler "xfade"           -> transitions.stock.build_stock_transition_plan(xfade_id,
                                          parameter_values)  (the SAME registry resolution the CPU
                                          builder runs, parameter-aware: cohort dynamic branches,
                                          panel motion, light/deco, stock, Circle, Black Hole)
                                            mode "custom" + admitted id -> kind "xfade_custom"
                                            native modes / prefilters    -> loud reject (F3/F5)
             other handlers            -> HANDLERS[handler] (tr_handlers.py: fade_color,
                                          wipe, slide_push; tr_equirect.py: equirectangular ->
                                          xfade_custom for the six expression ids, its own
                                          equirect_gaussian_blur kind for 360° Gaussian Blur)
                                          else loud reject
    renderer._FrameComposer.compose
        -> apply_transition(spec.kind, spec.payload, a, b, ApplyContext)
             -> TRANSITIONS[spec.kind].apply(payload, a, b, ctx)

Every transition is one ``Transition`` object -- ``apply(payload, A, B, ctx) -> frame`` --
registered once under its ``kind`` in the single ``TRANSITIONS`` registry.  A ``Transition``
is authored either as hand-written torch (Cross Dissolve, Wipe, Slide/Push) or as an ffmpeg
registry expression evaluated in torch (``xfade=custom`` and everything that lowers to it);
that is a difference of *how the kernel is written*, not a different lifecycle.  The one real
capability axis is ``Lowered.needs_history``: temporal kernels (Phase-5 earthquake / flashback)
read a per-side three-frame window, so the renderer builds ``a_history``/``b_history`` only when
the lowered transition sets it -- no renderer-side knowledge of which family is temporal.

    cross_dissolve : premultiplied linear-light lerp at Final Cut's frame centres
    xfade_custom   : registry expression evaluated by tensor/expr.py
                       premultiplied linear -> straight 0..255 code (``color.premultiplied_to_code``,
                       the CPU builder's ``format=gbrap`` side adaptation)
                       -> expr.xfade_custom_rgba (vf_xfade custom loop, GBRA planes, P = 1 - k/F)
                       -> ``color.code_to_premultiplied``

Cross Dissolve (calibrated): for the k-th of F transition frames the incoming
weight is ``(k + 0.5) / F`` (the legacy module's ``clip((N-0.5)/F,0,1)`` with a
one-based ``N``).

xfade-custom: the CPU reference materializes both sides as 8-bit encoded
*straight*-alpha ``gbrap`` (``ffmpeg._adapt_transition_side_to_encoded``) and
runs ``xfade=transition=custom:expr=<registry string>`` with ``offset=0`` and
``duration=F/fps``, so ``P = 1 - k/F`` (float32, see ``expr.xfade_progress``)
and every plane -- alpha included -- is the expression's output.  This module
evaluates the *same registry string* through ``tensor/expr.py``; nothing is
hand-ported, so a port cannot drift from the reference's expression text.

Admission (append-only): ``ADMITTED_XFADE_IDS`` lists the ``xfade_id``s whose
resolved expressions have a per-item golden in
``experimental_tests/core/test_tensor_transition_golden.py`` (ffmpeg ``xfade=custom``
on plates for the compiler's default parameter values and every parameter branch, plus
an end-to-end SSIM gate per registry family in ``test_tensor_transition_golden_f1f2.py``
vs the CPU reference).  An expression outside the admitted set is a loud
``TensorRenderUnsupported`` at plan time even though the evaluator could run it --
proof before pixels.

Evaluation dtype: the expression runs in the sides' dtype on the sides' device (float32 on
MPS); float32 breaks exact-integer sampling ties differently from the double reference (a
nearest-neighbour source pixel or a hard matte edge flips where a coordinate lands within
float32 rounding of an integer -- SSIM-invisible, see the golden's ``sampler`` tier).  Ids in
``FLOAT64_XFADE_IDS`` (Static's ``sin`` hash) are evaluated on the CPU in float64 -- where the
evaluator is bit-exact with the reference -- and copied back (``XfadePayload.float64``).

Port contract (handlers other than cross_dissolve / xfade)
----------------------------------------------------------
* ``lower(item, LowerContext) -> Lowered`` at plan time: validate the authored
  parameters exactly like the CPU emitter for that handler and raise
  ``support.reject(...)`` for anything not honoured; ``payload`` is the port's
  own frozen dataclass (no torch objects).
* ``Transition.apply(payload, a, b, ApplyContext) -> canvas``: ``a`` (outgoing) and
  ``b`` (incoming) are the two composed sides, premultiplied linear RGBA ``[4, H, W]``
  on the project canvas (transparent where the side has no pixels); the result
  is composited over the layers below the pair.  ``ctx.frame_index`` is ``k``
  (0-based within the owned transition window), ``ctx.frame_count`` is ``F``.
* Ports register from their own module: a ``Transition`` object via ``register`` and
  its plan-time lowerer via ``register_handler``; an id-specialized xfade lowerer via
  ``register_xfade_ids``.  This module imports the port modules at the bottom.
  Registering the same kind / handler / xfade id twice is an error.

Endpoint frame: at ``k = 0`` (``P = 1``) the outgoing side is returned as-is
without the code-space round trip.  The registry ``_endpoint`` guard yields
``A`` there anyway; skipping the round trip avoids quantizing (truncating) an
already exact side and equals the reference frame, which *is* the 8-bit side.
``P <= 0`` never occurs (``k < F``).

Main callers:
- ``plan.build_tensor_plan`` (``lower_transition``).
- ``renderer._FrameComposer.compose`` (``apply_transition``).
- ``experimental_tests/core/test_tensor_transition_golden.py`` (``xfade_custom``, ``xfade_expression``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, ClassVar, Final, Iterable, Mapping, Optional

import torch

from ..core.model import RenderTransition
from ..transitions.stock import build_stock_transition_plan
from .color import code_to_premultiplied, premultiplied_to_code
from .errors import TensorRenderUnsupported
from .expr import Expr, parse, xfade_custom_rgba, xfade_progress
from .support import reject

# xfade ids the plan admits (append-only; every entry has a golden in
# test_tensor_transition_golden.py: the compiler's default parameter values plus every
# structurally different parameter branch, float64 CPU bit-exact vs ``ffmpeg xfade=custom``,
# and one end-to-end SSIM render per registry family in
# test_tensor_transition_golden_f1f2.py).  Ids resolve through
# ``transitions.stock.build_stock_transition_plan`` with the item's compiler-resolved
# parameter values, so admitting an id admits every parameter branch the registry can
# select for it (the golden table enumerates the branches).
ADMITTED_XFADE_IDS: Final[tuple[str, ...]] = (
    # light / deco (light_deco.CUSTOM_IMPLEMENTATIONS; F0 seam + F1)
    "cohort_bloom_default",
    "cohort_flash_default",
    "cohort_lens_flare_default",
    "cohort_deco_default",
    # cohort dynamic branches (cohort.build_cohort_transition_plan; F1)
    "cohort_center_default",
    "cohort_clock_default",
    "cohort_page_curl_default",
    "cohort_swap_default",
    "cohort_static_default",
    "cohort_rotate_default",
    "cohort_swing_default",
    "cohort_switch_default",
    "cohort_arrows_default",
    "cohort_curtains_default",
    "cohort_veil_default",
    # panel motion (panel_motion.build_panel_motion_expression; F1)
    "cohort_divide_default",
    "cohort_spin_default",
    "cohort_clothesline_default",
    "cohort_flip_default",
    "cohort_scale_default",
    "cohort_multi_flip_default",
    "cohort_pinwheel_default",
    "cohort_reflection_default",
    # stock (stock.CUSTOM_IMPLEMENTATIONS), Circle (param "7"), Black Hole (carryovers; F1)
    "fall_default",
    "squares_tile_reveal_default",
    "circle_default",
    "black_hole_default",
)

# Admitted ids whose expression is evaluated in float64 on the CPU even when the renderer
# runs float32 on MPS.  Static's snow is ``abs(mod(sin(X*12.9898+Y*78.233+...)*43758.5453,1))``:
# float32 carries ~1e-3 absolute error into the ``sin`` argument, so the float32 field is a
# *different* noise realization (83% of values off in the golden) -- statistically alike,
# pixel-different, SSIM-fatal.  In CPU float64 the evaluator is bit-exact with the reference
# (libm-faithful transcendental path, ``expr._libm_unary``); the pass costs ~9 ms at 640x360.
FLOAT64_XFADE_IDS: Final[frozenset[str]] = frozenset({"cohort_static_default"})


@dataclass(frozen=True)
class LowerContext:
    """What a port may know about the transition at plan time."""

    width: int
    height: int
    frame_duration: Fraction
    frame_count: int      # F: owned transition frames


@dataclass(frozen=True)
class ApplyContext:
    frame_index: int      # k, 0-based within the owned window
    frame_count: int      # F
    width: int
    height: int
    frame_duration: Fraction
    # Oldest to newest raw composed sides for temporal transition kernels.
    # Empty means the caller did not supply history; ports that require it
    # must reject instead of silently substituting the current frame.
    a_history: tuple[torch.Tensor, ...] = ()
    b_history: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class Lowered:
    """Result of lowering one ``RenderTransition``: the apply-port kind, its payload, and capability.

    ``kind`` selects the ``Transition`` in ``TRANSITIONS`` at apply time.  ``needs_history``
    is the single capability axis (see the module doc): a temporal kernel sets it on the
    lowered instance so the renderer assembles the per-side three-frame history windows only
    for transitions that actually consume them.
    """

    kind: str
    payload: Any = None
    xfade_id: Optional[str] = None
    needs_history: bool = False


class Transition(ABC):
    """One transition behavior: composite two composed sides at a progress point.

    ``kind`` is the stable registry label carried on ``Lowered.kind`` / ``TransitionSpec.kind``
    (used by ``apply_transition`` and asserted by the goldens); ``apply`` is the render-time
    kernel.  A transition is stateless in the sense that all frame-window state it may need
    arrives through ``ApplyContext`` (``a_history``/``b_history``); the temporal capability is
    declared per lowered instance via ``Lowered.needs_history``, not on the class, because one
    family (Phase-5) is temporal for some prefilters and not others.
    """

    kind: ClassVar[str]

    @abstractmethod
    def apply(self, payload: Any, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
        ...


TransitionLower = Callable[[RenderTransition, LowerContext], Lowered]

# The single apply registry: one Transition object per ``kind``.
TRANSITIONS: Final[dict[str, Transition]] = {}
# Plan-time lowerers keyed by the authored ``handler``.
HANDLERS: Final[dict[str, TransitionLower]] = {}
# Per-xfade-id lowering overrides, consulted by the ``xfade`` handler before the stock path.
# This is the explicit extension point that replaces the old monkeypatch of the xfade handler
# (Phase-5 registers its ids here at import; the default path is the stock custom expression).
XFADE_ID_ROUTES: Final[dict[str, TransitionLower]] = {}


def register(transition: Transition) -> None:
    if transition.kind in TRANSITIONS:
        raise AssertionError(f"transition kind {transition.kind!r} registered twice")
    TRANSITIONS[transition.kind] = transition


def register_handler(handler: str, lower: TransitionLower) -> None:
    if handler in HANDLERS:
        raise AssertionError(f"transition handler {handler!r} registered twice")
    HANDLERS[handler] = lower


def register_xfade_ids(xfade_ids: Iterable[str], lower: TransitionLower) -> None:
    """Route specific xfade ids to an alternate lowerer (the explicit anti-monkeypatch)."""

    for xfade_id in xfade_ids:
        if xfade_id in XFADE_ID_ROUTES:
            raise AssertionError(f"xfade id {xfade_id!r} routed twice")
        XFADE_ID_ROUTES[xfade_id] = lower


# ---------------------------------------------------------------------------
# Cross Dissolve
# ---------------------------------------------------------------------------


def cross_dissolve(a: torch.Tensor, b: torch.Tensor, *, frame_index: int, frame_count: int) -> torch.Tensor:
    weight = min(1.0, max(0.0, (frame_index + 0.5) / frame_count))
    return a * (1.0 - weight) + b * weight


class CrossDissolve(Transition):
    kind = "cross_dissolve"

    def apply(self, payload: Any, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
        return cross_dissolve(a, b, frame_index=ctx.frame_index, frame_count=ctx.frame_count)


register(CrossDissolve())


# ---------------------------------------------------------------------------
# xfade custom (registry expressions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XfadePayload:
    xfade_id: str
    expression: str      # resolved registry text (parameter-aware); ``expr.parse`` caches by text
    float64: bool = False  # evaluate on CPU float64 whatever the renderer device (FLOAT64_XFADE_IDS)


def resolve_xfade_expression(xfade_id: str, parameter_values: Optional[Mapping[str, Any]] = None) -> str:
    """The reference's resolved ``xfade=custom`` expression text for an admitted id, loudly.

    Runs the very registry resolution the CPU builder runs
    (``build_stock_transition_plan(xfade_id, parameter_values)``); refuses ids
    outside ``ADMITTED_XFADE_IDS`` and plans that are not pure custom expressions.
    """

    if xfade_id not in ADMITTED_XFADE_IDS:
        raise TensorRenderUnsupported(
            f"xfade-custom transition {xfade_id!r} is not ported to the tensor renderer "
            f"(admitted: {', '.join(ADMITTED_XFADE_IDS)})"
        )
    plan = build_stock_transition_plan(xfade_id, dict(parameter_values or {}))
    if plan.mode != "custom" or plan.expression is None or plan.prefilter is not None:
        raise TensorRenderUnsupported(
            f"xfade transition {xfade_id!r} is admitted but resolves to mode={plan.mode!r} "
            f"prefilter={plan.prefilter!r}, not a pure custom expression"
        )
    return plan.expression


def xfade_expression(xfade_id: str, parameter_values: Optional[Mapping[str, Any]] = None) -> Expr:
    """The parsed registry expression for an admitted id (raises ``TensorRenderUnsupported`` otherwise)."""

    return parse(resolve_xfade_expression(xfade_id, parameter_values))


def xfade_custom(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    xfade_id: str,
    frame_index: int,
    frame_count: int,
    parameter_values: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """Evaluate a registry xfade-custom expression on premultiplied linear sides (see module doc)."""

    return _xfade_custom_expr(
        xfade_expression(xfade_id, parameter_values), a, b,
        frame_index=frame_index, frame_count=frame_count, float64=xfade_id in FLOAT64_XFADE_IDS,
    )


def _xfade_custom_expr(
    expr: Expr, a: torch.Tensor, b: torch.Tensor, *, frame_index: int, frame_count: int, float64: bool = False
) -> torch.Tensor:
    progress = xfade_progress(frame_index, frame_count)
    if progress >= 1.0:
        return a
    a_code, b_code = premultiplied_to_code(a), premultiplied_to_code(b)
    if float64:
        # CPU float64 is bit-exact with the reference (libm-faithful transcendental path);
        # the detour costs one [4, H, W] copy each way (see FLOAT64_XFADE_IDS).
        # Device hop before the dtype change: MPS refuses to materialize float64 even in transit.
        a_code, b_code = a_code.to("cpu").to(torch.float64), b_code.to("cpu").to(torch.float64)
    out_code = xfade_custom_rgba(expr, a_code, b_code, progress=progress)
    return code_to_premultiplied(out_code.to(a.dtype).to(a.device))


class XfadeCustom(Transition):
    kind = "xfade_custom"

    def apply(self, payload: XfadePayload, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
        return _xfade_custom_expr(
            parse(payload.expression), a, b,
            frame_index=ctx.frame_index, frame_count=ctx.frame_count, float64=payload.float64,
        )


register(XfadeCustom())


def _lower_xfade(item: RenderTransition, ctx: LowerContext) -> Lowered:
    if item.xfade_id is None:
        raise reject("transition (other)", f"{item.path}: xfade transition without a registry id")
    # Id-specialized routes (Phase-5) take over before the stock custom-expression path.
    route = XFADE_ID_ROUTES.get(item.xfade_id)
    if route is not None:
        return route(item, ctx)
    plan = build_stock_transition_plan(item.xfade_id, dict(item.parameter_values or {}))
    if plan.mode == "custom" and plan.expression is not None and plan.prefilter is None:
        if item.xfade_id not in ADMITTED_XFADE_IDS:
            raise reject(
                "transition (other)",
                f"{item.path}: transition {item.name!r} xfade_id={item.xfade_id!r} is a custom "
                "expression but is not in transitions.ADMITTED_XFADE_IDS (needs its golden)",
            )
        return Lowered(
            kind="xfade_custom",
            xfade_id=item.xfade_id,
            payload=XfadePayload(item.xfade_id, plan.expression, float64=item.xfade_id in FLOAT64_XFADE_IDS),
        )
    raise reject(
        "transition (other)",
        f"{item.path}: transition {item.name!r} xfade_id={item.xfade_id!r} resolves to "
        f"mode={plan.mode!r} prefilter={plan.prefilter!r} (native / prefilter modes: F3/F5)",
    )


def _lower_cross_dissolve(item: RenderTransition, ctx: LowerContext) -> Lowered:
    return Lowered(kind="cross_dissolve")


register_handler("cross_dissolve", _lower_cross_dissolve)
register_handler("xfade", _lower_xfade)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def lower_transition(item: RenderTransition, ctx: LowerContext) -> Lowered:
    """Lower one calibrated ``RenderTransition`` through its handler port or reject loudly."""

    lower = HANDLERS.get(item.handler or "")
    if lower is None:
        raise reject(
            "transition (other)",
            f"{item.path}: transition {item.name!r} handler={item.handler!r} xfade_id={item.xfade_id!r}",
        )
    lowered = lower(item, ctx)
    if lowered.kind not in TRANSITIONS:
        raise AssertionError(f"{item.path}: handler {item.handler!r} lowered to unregistered kind {lowered.kind!r}")
    return lowered


def apply_transition(kind: str, payload: Any, a: torch.Tensor, b: torch.Tensor, ctx: ApplyContext) -> torch.Tensor:
    """Run the ``Transition`` for a lowered kind on its two composed sides."""

    return TRANSITIONS[kind].apply(payload, a, b, ctx)


# Port modules register on import (one owner per file; append-only).
from . import tr_equirect as _tr_equirect  # noqa: E402,F401  (F2: equirectangular)
from . import tr_handlers as _tr_handlers  # noqa: E402,F401  (F4: fade_color, wipe, slide_push)
from . import tr_phase5 as _tr_phase5  # noqa: E402,F401  (F5: prioritized blur / warp cohort)
