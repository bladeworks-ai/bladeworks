"""The tensor renderer's single table of construct support (A0-lite freeze).

What it does
------------
Every construct the planner can meet is named here once with its status:

* ``supported``  -- the plan lowers it and the renderer executes it;
* ``rejected``   -- the plan raises ``TensorRenderUnsupported`` naming it
                    (loud, never approximated), with the sprint task that owns
                    the port.

``reject(construct, detail)`` is the one way a plan-time rejection is raised,
so "what does the tensor backend not do yet" is answerable by reading this
file, and every rejection message carries the same shape.

Main callers:
- ``plan.build_tensor_plan`` and its helpers.
- Docs / the future plan-coverage audit (``scripts/final_cut``), which
  histogram rejection messages by construct.

Why this exists:
The skeleton scattered ``if ...: problems.append(...)`` ladders through
``plan.py``; parallel agents porting features would each edit that ladder.
One table keeps the support surface reviewable and the ladder append-only.

Row naming (for agents requesting rows in their reports)
--------------------------------------------------------
* One row per *construct as the planner sees it*, phrased as a noun in the
  compiler's vocabulary (``adjust-crop trim/crop/pan``, ``retime reverse``,
  ``group container frame rate differs from project``), never as a symptom or an
  ffmpeg filter name.  Group by the section headers below.
* A row states what the *reference* would do that this backend does not; the
  ``detail`` argument to ``reject`` carries the clip path and the authored
  values, the row carries neither.
* ``supported`` rows exist so the surface is enumerable; they are never passed
  to ``reject`` (that is an assertion).  When a port lands, flip the row to
  ``supported`` and delete the owner task; do not leave both a supported and
  a rejected row for the same construct.
* Rejected rows carry the PYTORCH_MVP_PLAN.md task that owns the port
  (``"X6"``, ``"E1-E4 (...)"``) or ``""`` when the rejection is a caller /
  compiler bug that will never be ported.
* Sub-cases share one row when they share an owner and a fix (``"group
  transform / crop / opacity / blend"``); split rows only when the ports land
  separately.
"""

from __future__ import annotations

from typing import Final, Literal, Mapping

from .errors import TensorRenderUnsupported

Status = Literal["supported", "rejected"]

# construct -> (status, owner task in PYTORCH_MVP_PLAN.md or "" when supported)
SUPPORT: Final[Mapping[str, tuple[Status, str]]] = {
    # ---- clip kinds / sources -------------------------------------------
    "asset-clip": ("supported", ""),
    "still image": ("supported", ""),
    "title / caption / Custom Solid raster": ("supported", ""),
    "gap (spine)": ("supported", ""),
    # The two "runtime raster ..." rows are caller bugs, not port gaps: the
    # caller (``executor.execute_render``) owns the raster pass and must hand
    # its ``text_images`` to ``build_tensor_plan(rasters=...)``.  Custom Solid
    # is likewise the only generator with a portable adapter *anywhere*
    # (``text.resolve_generator_clip_raster``), so none of the three has an
    # owning sprint task.
    "runtime raster not resolved by the caller": ("rejected", ""),
    "runtime raster for a non-raster clip": ("rejected", ""),
    "generator (not Custom Solid)": ("rejected", ""),
    "raster speed (title / caption)": ("rejected", "X5 rasters"),
    "clip kind": ("rejected", "X6 scopes"),
    # Not a port gap: a clip whose pixels are not on disk is an error in the
    # bindings or in the caller's raster pass, so it stays rejected forever.
    "media file (missing or unreadable)": ("rejected", ""),
    "source display rotation metadata": ("supported", ""),
    "non-square pixel aspect": ("rejected", "post-MVP spatial intrinsics"),
    "spatial intrinsics (360 / stereo / stabilization / rolling shutter)": ("rejected", "X10"),
    "source pixel format": ("supported", ""),
    # Colour-in policy (X10): swscale-exact decode for planar 8/10/12-bit YUV/YUVA
    # 4:2:0 / 4:2:2 / 4:4:4 with bt709 / bt601 / bt2020nc x limited / full tags
    # (decode.resolve_source_color), plus the frozen Rec.2020 HLG/PQ to Rec.709
    # SDR color-conform LUTs.
    "source pixel format (unsupported)": ("rejected", "X10"),
    "source colour matrix (unsupported)": ("rejected", "X10"),
    "source colour transfer (HLG / PQ)": ("supported", "X10 HDR-to-SDR LUT"),
    "source HDR metadata (malformed)": ("rejected", "X10 HDR-to-SDR LUT"),
    "source HDR alpha (unsupported)": ("rejected", "X10 HDR-to-SDR LUT"),
    # ---- xyzt geometry -----------------------------------------------------
    "conform fit/fill/none": ("supported", ""),
    "conform (other)": ("rejected", "X1"),
    "adjust-transform (static)": ("supported", ""),
    "adjust-transform (animated)": ("supported", ""),
    "adjust-crop trim/crop/pan": ("supported", ""),
    "corner pin": ("supported", ""),
    # ---- time --------------------------------------------------------------
    "retime forward / variable / freeze": ("supported", ""),
    "retime reverse": ("supported", ""),
    "forward/freeze retime inside a retimed group": ("supported", ""),
    "transition endpoint holds": ("supported", ""),
    # ---- opacity / blend --------------------------------------------------
    "opacity + fades + opacity animation": ("supported", ""),
    "blend mode (reviewed standard)": ("supported", ""),
    "blend mode (unknown or uncalibrated)": ("rejected", "X4 blend modes"),
    # ---- effects / transitions --------------------------------------------
    # Effects lower through the port registry in tensor/effects.py (leaf effects after
    # crop/conform, before the spatial tail; group effects on the rendered scope's canvas).
    "effect (ported handler, leaf or group)": ("supported", ""),
    "effect (unported handler)": ("rejected", "E1-E4 (register a port in tensor/fx_*.py)"),
    "effect (unsupported parameters)": ("rejected", "E1-E4 (the owning port)"),
    "effect with explicit numeric shape/draw/color/luma mask": ("supported", ""),
    "effect with opaque, tracked, Magnetic, Auto, or ML mask": ("rejected", "X8 masks"),
    # Transitions lower through the port registry in tensor/transitions.py: xfade ids
    # resolve through transitions.stock.build_stock_transition_plan (parameter-aware) and
    # must be in transitions.ADMITTED_XFADE_IDS (one golden per id); other handlers
    # (fade_color / wipe / slide_push / equirectangular) register from tr_*.py.
    "transition cross_dissolve / xfade custom (admitted registry ids) / ported handler": ("supported", ""),
    "transition (unsupported parameters)": ("rejected", "F4 (the owning handler port)"),
    "transition (other)": ("rejected", "F1-F5 (append ids to transitions.ADMITTED_XFADE_IDS or register a handler port in tensor/tr_*.py)"),
    # X7-lite: a side is every marked participant layer (connected lanes, group
    # leaves) composed full-canvas; overlapping transitions are independent items;
    # a handler-less transition is the reference's explicit hard cut (skipped).
    "transition sides with several participants (connected lanes / group leaves)": ("supported", ""),
    "overlapping transitions (independent participants)": ("supported", ""),
    "transition without a portable handler (hard cut)": ("supported", ""),
    # A side with no video participant is a compile error in the reference
    # ("refusing to silently render it as a cut"), so it stays rejected forever.
    "transition without both participants": ("rejected", ""),
    # ---- scopes -----------------------------------------------------------
    # X6: every container is either INERT (Fit same-aspect / None same-size, no
    # transform-crop-opacity-blend-effects: folded onto its leaves, the flat
    # fast path) or RENDERED (its own container canvas, placed like a leaf:
    # crop / conform / transform / animation / opacity, group effects, native
    # cadence conversion, and recursive exact retime) -- see plan.py "Group scopes".
    "inert group scope (fit, same aspect)": ("supported", ""),
    "inert group with several leaves / lanes / offsets / gaps": ("supported", ""),
    "constant-speed retimed inert group": ("supported", "rendered as an explicit clock boundary"),
    "group transform / crop / opacity": ("supported", ""),
    "group conform none / fit onto a different container size": ("supported", ""),
    "group effect on a composed group surface (any leaves)": ("supported", ""),
    "effect on a retimed group": ("supported", ""),
    "constant-speed retimed rendered group (transform / crop / opacity / effects)": ("supported", ""),
    "nested rendered scopes (compound in compound, mc-clip angle scope)": ("supported", ""),
    "transition on a rendered group (finished surface as the side)": ("supported", ""),
    "transition inside a retimed group": ("supported", ""),
    "group blend mode (reviewed standard)": ("supported", ""),
    "group retime map (variable forward / freeze)": ("supported", ""),
    "group container frame rate differs from project": ("supported", ""),
    "group retime map (reverse boundary)": ("supported", ""),
}


def reject(construct: str, detail: str) -> TensorRenderUnsupported:
    """Return the loud rejection for ``construct`` (callers ``raise reject(...)``)."""

    entry = SUPPORT.get(construct)
    if entry is None:
        raise KeyError(f"unknown support construct {construct!r}; add it to tensor/support.py")
    status, owner = entry
    if status == "supported":
        raise AssertionError(f"{construct!r} is marked supported; do not reject it")
    suffix = f" [port: {owner}]" if owner else ""
    return TensorRenderUnsupported(f"{detail}: tensor renderer does not support {construct}{suffix}")


def supported_constructs() -> tuple[str, ...]:
    return tuple(name for name, (status, _) in SUPPORT.items() if status == "supported")


def rejected_constructs() -> tuple[str, ...]:
    return tuple(name for name, (status, _) in SUPPORT.items() if status == "rejected")
