"""Build the tensor renderer's frame-grid plan from a compiled RenderDocument.

Architecture map
================

    RenderDocument
        -> group scopes (``_classify_scopes``): every container (compound
           ``ref-clip``, multicam ``mc-clip`` + its ``mc-source`` angle scope,
           ``clip`` / ``sync-clip`` / ``audition``) is either
             INERT     -- a pass-through wrapper (Fit same-aspect / None same-size,
                          no transform / crop / opacity / blend / effects): it folds
                          onto its descendants (placement affine ``canvas_to_owner``)
                          and never owns
                          a canvas -- the exact flat fast path (X6-lite), or
             RENDERED  -- a ``ScopeSpec``: it owns a transparent canvas of its
                          container size on which its children (leaves, child
                          scopes, transitions) are composed at render time, and it
                          is placed into its OWNER's canvas like a leaf (crop /
                          conform / transform / animation homography, group
                          effects, group opacity) -- "Group scopes (X6)" below
        -> per enabled video leaf (``_lower_leaf``) -> ``LayerSpec``:
             source        : decoded video, or a raster held for the whole window
                             (still image, title/caption, Custom Solid; "Raster
                             sources (X5)" below)
             owned frame window (midpoint ownership, expanded over adjacent
                             calibrated transitions like the legacy
                             ``_expanded_schedule``)
             SourceClock   : local time -> exact file-local source instant
             GeometryPlan / OpacityPlan : the live exact kernels (``snapshot(t)``)
             canvas_to_owner : inert-ancestor placement onto the owner canvas
             EffectSpec    : leaf effects through the ports (``tensor/effects.py``)
             owner scope   : the nearest rendered ancestor (None = the root canvas)
             DecodeRaster  : the raster the decoder should produce (``decode_policy.py``:
                             native for export, near the visible footprint for seek / scan)
        -> per rendered scope (``_lower_scope``) -> ``ScopeSpec`` (same fields on
           its container -> parent-container geometry, plus a composable
           ``ScopeTimeMap`` clock boundary)
        -> per calibrated transition (``_lower_transitions``): owner scope, owned
           window on the owner's local frame grid, the two participant SIDES
           (leaves and rendered scopes at the transition's frontier, sorted by
           (lane, document order) -- "Transition sides (X7)" below), z-key and
           the transition port's lowered payload
        -> TensorRenderPlan (frozen; ``layers_at`` sorts by the hierarchical ``z_key``)

Only *semantics already computed by the compiler* live here: geometry,
animation, retime and opacity are evaluated by the shared exact kernels in
``geometry.py`` / ``retime.py`` / ``animation.py`` / ``compositor.py``.  This
module never re-derives a convention (rotation sign, Y flip, units, anchor,
ownership) -- it wires clip fields into those kernels and rejects, loudly and
by name (``support.py``), every construct it does not lower yet.

Main callers:
- ``renderer.render_document`` (via ``build_tensor_plan``).
- ``scripts/final_cut/fcpxml_tensor_render_bench.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import numpy as np

from ..core.animation import ken_burns_progress
from ..core.compositor import (
    CompositorError,
    CompositorWindow,
    OpacityPlan,
    resolve_blend_mode,
)
from ..core.geometry import (
    FrameGeometry,
    GeometryError,
    GeometryPlan,
    GeometrySnapshot,
    GeometryWindow,
    SourceRect,
    resolve_crop_camera_placement,
)
from ..core.model import AlphaHandling, CropRect
from ..core.model import RenderClip, RenderDocument, ResolvedEffect
from ..core.retime import RetimeMap
from ..core.retime_execution import resolve_owned_frame_window, resolve_video_frame_ownership
from .decode import SourceColor, VideoProbe, check_source_color, probe_video
from .decode_policy import DecodePolicy, DecodeRaster, resolve_decode_raster
from .errors import TensorRenderUnsupported
from .support import reject
from . import effects as effect_ports
from . import transitions as transition_ports
from .resolution import OutputResolution
from .effects import EffectSpec
from .fx_color import reference_effect_link

CropMode = Optional[Literal["trim", "crop", "pan"]]
SourceKind = Literal["video", "raster"]


def owned_decode_time(
    source_time: Fraction,
    *,
    direction: str,
    source_frame_duration: Optional[Fraction],
) -> Fraction:
    """Convert a mapped source coordinate to the decoder request instant.

    Forward ownership can keep the authored coordinate. Reverse ownership uses
    the left limit at an exact source-frame high edge, so it requests the
    selected preceding frame's exact start. Freeze follows forward ownership.

    Main callers:
    - ``LayerSpec.source_time``.
    - Directional frame-selection goldens.
    """

    if direction != "reverse":
        return source_time
    if source_frame_duration is None:
        raise TensorRenderUnsupported(
            "reverse playback requires the source format frame duration"
        )
    return resolve_video_frame_ownership(
        source_time,
        frame_duration=source_frame_duration,
        direction="reverse",
    ).source_frame_start


@dataclass(frozen=True)
class SourceClock:
    """Local time -> exact file-local source instant for one layer.

    Mirrors the legacy builder's two paths (``ffmpeg.py`` around
    ``_retime_map_with_transition_holds``):

    * ``retime_map is None``: a simple 1x forward clip.  Source extends
      linearly into transition handles (``source_start - pre_roll``); when the
      handle would leave ``[0, asset_duration]`` the legacy path switches to
      endpoint holds, expressed here as clamping the clip-local time.
    * ``retime_map`` set: any retimed clip.  The authored map is exact over the
      clip interval and the transition-only handles hold the nearest endpoint
      (``RetimeMap.with_endpoint_holds`` semantics == clamping clip time).

    ``clip_start`` is on the layer's *local* clock (absolute timeline time,
    or a retimed group's pad clock -- see ``LocalClock``); the sampled instant
    of output frame ``n`` is the frame *start* ``n * frame_duration`` (the
    legacy layer clock ``(on-1)*fd``), while canvas ownership uses frame
    midpoints.
    """

    clip_start: Fraction
    clip_duration: Fraction
    source_start: Fraction
    speed: Fraction
    retime_map: Optional[RetimeMap]
    hold_ends: bool
    # Output (timeline) frame cadence of this layer. Endpoint holds repeat the
    # last *output* frame, so the reverse tail correction subtracts one of these
    # from ``clip_duration`` -- see ``_held_clip_time``.
    frame_duration: Fraction

    @classmethod
    def constant(
        cls, *, clip_start: Fraction, clip_duration: Fraction, frame_duration: Fraction
    ) -> "SourceClock":
        """The clock of a raster layer: every output instant samples source 0.

        Why this exists: stills, titles, captions and Custom Solid generators
        are a *single* image looped for the clip's whole window (the reference
        feeds them as ``-loop 1 -framerate fps`` inputs and only trims them,
        ``ffmpeg.py:1338`` / ``ffmpeg.py:9853``), so they have no source
        timeline at all.  Expressing that as ``speed = 0`` keeps one clock type
        instead of teaching every caller about a second one.
        """

        return cls(
            clip_start=clip_start,
            clip_duration=clip_duration,
            source_start=Fraction(0),
            speed=Fraction(0),
            retime_map=None,
            hold_ends=False,
            frame_duration=frame_duration,
        )

    def source_time(self, local_time: Fraction) -> Fraction:
        return self.source_sample(local_time)[0]

    def source_sample(self, local_time: Fraction) -> tuple[Fraction, str]:
        """Return the mapped source instant and its directional ownership rule.

        Main callers:
        - ``LayerSpec.source_time`` when converting a retime coordinate into
          the exact source frame that the decoder must fetch.

        Why this exists:
        Forward and reverse playback select different frames at an exact source
        boundary. Keeping the direction beside the mapped rational instant
        prevents the decoder layer from trying to infer it from request order.
        """

        clip_time = local_time - self.clip_start
        if self.retime_map is None:
            if self.hold_ends:
                clip_time = self._held_clip_time(clip_time)
            direction = "reverse" if self.speed < 0 else ("freeze" if self.speed == 0 else "forward")
            return self.source_start + clip_time * self.speed, direction
        clip_time = self._held_clip_time(clip_time)
        sample = self.retime_map.sample(clip_time)
        return sample.source_time, sample.segment_kind

    def _tail_plays_in_reverse(self) -> bool:
        """True when the clip's LAST output frame is decoded in reverse.

        For a linear clip (no retime map) that is simply ``speed < 0``; for an
        authored retime map it is the kind of the final segment -- the one that
        owns ``clip_duration``.  Reverse is the only direction whose endpoint
        hold needs the ``- frame_duration`` correction in ``_held_clip_time``.
        """

        if self.retime_map is None:
            return self.speed < 0
        return self.retime_map.segments[-1].kind == "reverse"

    def _held_clip_time(self, clip_time: Fraction) -> Fraction:
        """Clamp a clip-local instant into this layer's endpoint-hold window.

        Endpoint holds (transition handles / post-roll) repeat the clip's
        nearest endpoint FRAME while the render window extends past the clip.
        The head always holds clip-local ``0``; the tail holds the last real
        OUTPUT frame.

        Why the tail is not simply ``clip_duration``:
        A forward tail sampled at ``clip_duration`` floors onto a source frame
        that still exists, so holding that boundary is harmless.  A REVERSE tail
        instead takes the LEFT limit at the boundary: sampling ``clip_duration``
        maps to the source-0 low edge, and reverse frame ownership then rolls one
        frame *below* it to ``-frame_duration`` -- a negative decode instant the
        decoder rejects (the bug this guards against).  Holding
        ``clip_duration - frame_duration`` (the last real output frame) instead
        repeats the correct final reversed frame, matching the tail clamp already
        used by ``ScopeTimeMap.child_frame`` and ``LocalClock.pad_frame``.
        Forward and freeze tails keep the exact ``clip_duration`` bound, so their
        reference parity is unchanged.
        """

        upper = self.clip_duration
        if self._tail_plays_in_reverse():
            upper = max(Fraction(0), self.clip_duration - self.frame_duration)
        return min(max(clip_time, Fraction(0)), upper)


@dataclass(frozen=True)
class LocalClock:
    """Output time -> the clock a leaf's kernels run on inside a constant-speed retimed group.

    The reference composes a retimed compound / multicam group on its *source
    pad* (children placed at ``group_start + (source offset)``, ``compiler.
    _resolve_virtual_video_source`` / ``ffmpeg._compose_group_scopes``) and
    then retimes that finished pad as one clip instance
    (``ffmpeg._retime_source_instance``: ``rebase_source_retime`` +
    ``build_retime_execution_plan``, endpoint holds over the group's transition
    handles).  Flattening the group therefore means every leaf inside it runs
    its own kernels (source clock, geometry, opacity, effects ``N``, Ken Burns)
    on that pad clock:

        pad(T) = origin + floor(clamp(T - origin, 0, hold - fd) * speed / fd) * fd

    ``origin`` is the group's output start, ``hold`` its output duration (the
    handles hold the pad's endpoint FRAMES: the first, and the last output
    frame inside the group -- ``_retime_map_with_transition_holds`` "hold its
    nearest endpoint frame"), ``speed`` its constant rate, and the floor is the
    pad-frame selection of the reference's retime execution ("Final Cut's
    default floor sampling", ``retime_execution``): the pad only exists at
    frame instants, so every kernel of the leaf is evaluated at the selected
    pad frame's start.  The pad grid is anchored at ``origin`` (Final Cut
    authors group starts on the frame grid, where this equals the reference's
    absolute pad grid).

    A leaf on pad frames ``[K1, K2)`` (``K = ceil((pad time - origin) / fd)``)
    is composited on the output frames whose selected pad frame lies in that
    range -- ``LayerSpec.first_frame`` / ``end_frame`` are derived from that
    rule in ``_retimed_leaf_frames`` (not from midpoint ownership of the mapped
    interval: a leaf starting on an odd pad frame at 2x starts half an output
    frame late, and the reference's pad shows nothing on that frame).

    Main callers: ``LayerSpec.local_time`` (every kernel evaluation of a layer).
    """

    origin: Fraction
    speed: Fraction
    hold: Fraction
    frame_duration: Fraction

    def pad_frame(self, absolute_time: Fraction) -> int:
        """The pad frame (index from ``origin``) the reference's retime selects at ``absolute_time``."""

        last = max(Fraction(0), self.hold - self.frame_duration)
        pad_time = min(max(absolute_time - self.origin, Fraction(0)), last) * self.speed
        return math.floor(pad_time / self.frame_duration)

    def time(self, absolute_time: Fraction) -> Fraction:
        return self.origin + self.pad_frame(absolute_time) * self.frame_duration

    def output_frame_range(self, pad_first: int, pad_end: int, *, output_end: int) -> tuple[int, int]:
        """Output frames ``[first, end)`` whose selected pad frame lies in ``[pad_first, pad_end)``.

        ``pad_frame(n * fd) >= K``  <=>  ``(n * fd - origin) * speed >= K * fd``; the
        end is capped at ``output_end`` (the group's own owned end frame) because
        the end hold keeps selecting the pad's last frame forever.
        """

        fd, s = self.frame_duration, self.speed
        first = math.ceil((self.origin + pad_first * fd / s) / fd)
        end = min(output_end, math.ceil((self.origin + pad_end * fd / s) / fd))
        return first, end


@dataclass(frozen=True)
class ScopeTimeMap:
    """Map one parent sample through a scope's native output and source clocks.

    Architecture map:
    parent frame start
        -> floor onto the scope's native output cadence
        -> clamp to the visible instance for endpoint holds
        -> exact ``RetimeMap`` lookup in source coordinates
        -> floor onto the child story's native cadence

    The first floor is frame-rate normalization of the completed scope. The
    retime lookup then selects which child-story frame feeds that native output
    frame. Keeping both operations on the scope boundary makes recursion
    composable: an inner scope receives the exact frame selected by its parent
    and applies its own boundary independently.

    Main callers: ``renderer._FrameComposer.placed_scope``.
    """

    parent_start: Fraction
    parent_duration: Fraction
    native_output_start: Fraction
    timeline_start: Fraction
    source_origin: Fraction
    parent_frame_duration: Fraction
    child_frame_duration: Fraction
    parent_frame_origin: int
    native_frame_origin: int
    child_frame_origin: int
    retime_map: RetimeMap

    def native_output_frame(self, parent_frame: int) -> int:
        """Native scope-output frame owning one parent frame start (FCP floor sampling)."""

        local_parent_time = (
            parent_frame - self.parent_frame_origin
        ) * self.parent_frame_duration
        return self.native_frame_origin + math.floor(
            local_parent_time / self.child_frame_duration
        )

    def child_frame(self, native_output_frame: int) -> int:
        """Child-story frame selected for one native scope-output frame.

        Endpoint handles hold the first or last native output frame inside the
        instance before the exact retime map is evaluated. Directional frame
        ownership is then applied at this scope boundary, so nested reverse maps
        compose independently instead of leaking a parent cadence into a child.
        """

        output_time = self.native_output_start + (
            native_output_frame - self.native_frame_origin
        ) * self.child_frame_duration
        first_time = self.parent_start
        last_time = max(first_time, self.parent_start + self.parent_duration - self.child_frame_duration)
        held_time = min(max(output_time, first_time), last_time)
        timeline_time = self.timeline_start + (held_time - self.parent_start)
        sample = self.retime_map.sample(timeline_time)
        direction = "reverse" if sample.segment_kind == "reverse" else "forward"
        ownership = resolve_video_frame_ownership(
            sample.source_time,
            frame_duration=self.child_frame_duration,
            frame_grid_origin=self.source_origin,
            direction=direction,
        )
        return self.child_frame_origin + ownership.source_start_frame


@dataclass(frozen=True, kw_only=True)
class PlacedItem:
    """What a leaf layer and a rendered group scope share: one placed raster on a frame grid.

    Two clocks: ``first_frame`` / ``end_frame`` are OUTPUT frames (when the item
    is composited); ``render_start`` / ``local_first_frame`` /
    ``local_frame_count`` are on the item's LOCAL clock (identical to the output
    clock unless ``local_clock`` is set), which is what ``geometry``,
    ``opacity`` and the effect frame counter run on.

    ``frame`` maps the item's raster (a decoded / rasterized source for a leaf,
    the container canvas for a scope) onto the canvas its geometry is authored
    against; ``canvas_to_owner`` then places that canvas onto the OWNER canvas
    (the nearest rendered ancestor scope, or the root) through the inert
    ancestors in between.

    Why this exists: the reference places a finished group surface with the very
    same chain it uses for a leaf (``_group_video_chain``: crop / conform,
    effects, transform / animation, then opacity + blend in the parent fold), so
    one set of kernels serves both -- ``ScopeSpec`` differs from ``LayerSpec``
    only in where its pixels come from.
    """

    path: str
    lane: int
    document_order: int
    first_frame: int
    end_frame: int
    # Local-clock render window: its start instant and its frame extent
    # (handles included, not clamped to the document) -- the legacy layer
    # stream ``F`` and its first frame index on the local frame grid.
    render_start: Fraction
    local_first_frame: int
    local_frame_count: int
    # Cadence of the canvas that owns this item's placement. Root items use
    # the project cadence; scope children use their owner's native cadence.
    frame_duration: Fraction
    frame: FrameGeometry
    geometry: GeometryPlan
    conform: Literal["fit", "fill", "none"]
    crop_mode: CropMode
    # 3x3 row-major affine placing the item's canvas onto its owner's canvas
    # (identity unless inert ancestors sit between the two).
    canvas_to_owner: tuple[float, ...]
    opacity: OpacityPlan
    # Canonical mode name from the shared core contract. Keeping this on the
    # placed item lets a rendered group use the same parent-fold operation as
    # an ordinary leaf.
    blend_mode: str = "Normal"
    effects: tuple[EffectSpec, ...] = ()
    local_clock: Optional[LocalClock] = None
    # The rendered scope this item is stacked in (None = the root canvas).
    owner_id: Optional[str] = None
    # (lane, document_order) of every INERT ancestor between the owner and this
    # item, outermost first: the reference composes a group as ONE raster at the
    # group's own (lane, document_order), so an item folded out of an inert
    # group must sort at its group's position in the owner stack (``z_key``).
    z_prefix: tuple[tuple[int, int], ...] = ()

    @property
    def z_key(self) -> tuple[tuple[int, int], ...]:
        """Hierarchical z-key in the owner stack: inert-ancestor pairs, then own (lane, document_order)."""

        return (*self.z_prefix, (self.lane, self.document_order))

    def local_time(self, frame: int, frame_duration: Fraction) -> Fraction:
        """The local-clock instant sampled for output frame ``frame`` (its start)."""

        absolute = frame * frame_duration
        return absolute if self.local_clock is None else self.local_clock.time(absolute)

    def local_frame(self, frame: int, frame_duration: Fraction) -> int:
        """Local frame-grid index of output frame ``frame`` (the effect counter ``N`` grid).

        Identity for ordinary items; inside a retimed group it is the pad
        frame (from the group's origin) the reference's retime selects for
        this output frame (``LocalClock.pad_frame``).
        """

        if self.local_clock is None:
            return frame
        return self.local_clock.pad_frame(frame * frame_duration)

    def geometry_at(self, frame: int, frame_duration: Fraction) -> GeometrySnapshot:
        snapshot = self.geometry.snapshot(self.local_time(frame, frame_duration))
        if self.crop_mode == "pan":
            snapshot = self._pan_camera(snapshot, self.local_frame(frame, frame_duration))
        return snapshot

    def _pan_camera(self, snapshot: GeometrySnapshot, local_frame: int) -> GeometrySnapshot:
        """Ken Burns on the legacy layer clock: progress = k / (F - 1) over the render window.

        ``GeometryPlan.snapshot`` evaluates the calibrated progress curve on
        clip time ``k / F`` and holds it during transition handles; the CPU
        reference (``ffmpeg._ken_burns_camera_expressions``) drives the same
        curve with the layer's frame counter normalized so the *last* frame of
        the render window is exactly 1.  The tensor renderer soft-matches the
        reference, so it follows the emitted convention here (ledgered in
        ``tensor/REFERENCE_DISCREPANCIES.md``).
        """

        crop = self.geometry.crop
        assert crop is not None and len(crop.active_rects) == 2
        first, last = crop.active_rects
        count = max(2, self.local_frame_count)
        progress = ken_burns_progress(Fraction(local_frame - self.local_first_frame), Fraction(count - 1))
        rect = CropRect(
            left=first.left + (last.left - first.left) * progress,
            top=first.top + (last.top - first.top) * progress,
            right=first.right + (last.right - first.right) * progress,
            bottom=first.bottom + (last.bottom - first.bottom) * progress,
        )
        unit = self.frame.source_height / 100.0
        source_rect = SourceRect(
            x=unit * rect.left,
            y=unit * rect.top,
            width=self.frame.source_width - unit * rect.left - unit * rect.right,
            height=self.frame.source_height - unit * rect.top - unit * rect.bottom,
        )
        camera = resolve_crop_camera_placement(
            self.frame, source_rect, self.conform, allow_outside_source=True
        )
        return replace(snapshot, source_rect=source_rect, camera_placement=camera)

    def opacity_at(self, frame: int, frame_duration: Fraction) -> float:
        return float(self.opacity.snapshot(self.local_time(frame, frame_duration) - self.render_start).result)

    @property
    def canvas_matrix(self) -> np.ndarray:
        return np.array(self.canvas_to_owner, dtype=np.float64).reshape(3, 3)

    @property
    def is_static(self) -> bool:
        return not self.geometry.has_animation

    def active(self, frame: int) -> bool:
        return self.first_frame <= frame < self.end_frame


@dataclass(frozen=True, kw_only=True)
class LayerSpec(PlacedItem):
    """One video leaf on the output frame grid, holding the live exact kernels (see ``PlacedItem``)."""

    clip_id: str
    # The file the pixels come from: the clip's media for video and still
    # assets, the caller-resolved runtime PNG for titles / captions / solids.
    media_path: Path
    clock: SourceClock
    # How ``decode.open_source`` should produce this layer's pixels: a decoded
    # video stream, or one image decoded once and held for the whole window.
    source_kind: SourceKind = "video"
    # Exact source cadence from the asset's FCPXML format. Reverse playback
    # needs it to apply high-edge ownership before asking the decoder.
    source_frame_duration: Optional[Fraction] = None
    # PyAV exposes encoded pixels and does not apply the container display
    # matrix. Keeping the decoder verdict explicit prevents a future normalized
    # decoder from applying the same rotation twice.
    source_rotation_degrees: int = 0
    decoder_applied_orientation: bool = False
    # True when this leaf's own retime or any ancestor scope can request source
    # frames in descending order. ``decode.open_source`` uses it to enable the
    # bounded GOP cache even when the leaf itself has a forward source clock.
    reverse_decode_cache: bool = False
    # The immediate ancestor scope (inert or rendered; None on the spine).
    nearest_scope_id: Optional[str] = None
    # Reference stage order for a direct child of an expanded root scope
    # (``ffmpeg._video_chain`` with ``layer.target_surface``): crop / conform onto
    # the container canvas FIRST (clipping Fill overscan there), then the
    # transform on that canvas -- two resamplings, not the fused one-homography
    # path of spine leaves (ledger row 3).
    staged: bool = False
    # The raster ``decode.open_source`` asks the decoder for (``decode_policy.py``).
    # None for raster layers.  ``frame`` ALWAYS keeps the native display raster;
    # when this is a downscaled request the renderer samples the smaller raster
    # through ``decode_policy.decoded_to_native_matrix`` -- geometry never
    # re-derives itself from the decoded size.
    decode_raster: Optional[DecodeRaster] = None
    # ``None`` is the explicit un-authored state and resolves to straight only
    # when the decoder actually encounters an alpha-carrying source format.
    alpha_handling: Optional[AlphaHandling] = None
    source_has_alpha: bool = False

    def source_time(self, frame: int, frame_duration: Fraction) -> Fraction:
        source_time, direction = self.clock.source_sample(self.local_time(frame, frame_duration))
        try:
            return owned_decode_time(
                source_time,
                direction=direction,
                source_frame_duration=self.source_frame_duration,
            )
        except TensorRenderUnsupported as error:
            raise TensorRenderUnsupported(f"{self.path}: {error}") from error

    @property
    def canvas_to_project(self) -> tuple[float, ...]:
        """Compatibility name for ``canvas_to_owner`` (identical when the owner is the root)."""

        return self.canvas_to_owner


@dataclass(frozen=True, kw_only=True)
class ScopeSpec(PlacedItem):
    """One RENDERED group scope: a transparent canvas of its container size, placed like a leaf.

    ``scope_id`` is the compiled group clip id; children (``LayerSpec`` /
    ``ScopeSpec`` / ``TransitionSpec``) name it as their ``owner_id`` /
    ``scope_id``.  ``width`` x ``height`` is the container canvas the children
    are composed on (``frame.source_*``); ``frame.project_*`` is the immediate
    parent scope's container (the project for a root scope).

    ``time_map`` first floor-samples the completed surface at this scope's
    native cadence, then applies this scope's exact retime to select a child
    story frame. Nested scopes repeat that operation independently.

    ``effects_on_container`` selects the reference's stage order for group
    effects: a root scope's expanded chain (``ffmpeg._group_video_chain``) runs
    them on the container surface BEFORE conform / transform; a nested scope's
    chain (``_legacy_group_video_chain``) runs them after crop / conform and
    before the transform, exactly like a leaf.

    ``expand_children`` mirrors ``ffmpeg._plan_group_execution``: a root scope
    with direct leaves composes them on a surface that grows to hold every
    child pixel (a transformed child may leave the container and a group
    transform may bring it back), so the renderer sizes the canvas per frame
    from the children's quads; every other scope clips at its container.
    """

    scope_id: str
    width: int
    height: int
    time_map: ScopeTimeMap
    effects_on_container: bool = True
    expand_children: bool = False


@dataclass(frozen=True)
class TransitionSpec:
    """One calibrated transition on its owner scope's frame grid, with its two composed sides.

    ``outgoing_clip_ids`` / ``incoming_clip_ids`` are the items of each side --
    leaf clip ids and rendered scope ids at the transition's frontier -- bottom
    to top (sorted by (lane, document_order) like the reference's
    ``_compose_transition_side``); ``lane`` / ``document_order`` is the z-key of
    the composed result in the owner's stack (``_transition_composite_item``:
    the minimum over all participants).  ``scope_id`` is the rendered scope
    whose stack composes the item (None = root); ``first_frame`` / ``end_frame``
    are on that scope's LOCAL frame grid (the pad grid inside a retimed group).
    """

    path: str
    kind: str                       # apply-port key in tensor/transitions.py (TRANSITIONS)
    xfade_id: Optional[str]
    first_frame: int
    end_frame: int
    outgoing_clip_ids: tuple[str, ...]
    incoming_clip_ids: tuple[str, ...]
    lane: int
    document_order: int
    frame_duration: Fraction
    payload: Any = None             # port-owned lowered payload (see transitions.Lowered)
    needs_history: bool = False     # the temporal capability: renderer builds side history iff set
    scope_id: Optional[str] = None
    # Inert ancestors between the owner and the transition (see ``PlacedItem.z_prefix``).
    z_prefix: tuple[tuple[int, int], ...] = ()

    @property
    def z_key(self) -> tuple[tuple[int, int], ...]:
        return (*self.z_prefix, (self.lane, self.document_order))

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.first_frame

    @property
    def participant_ids(self) -> frozenset[str]:
        return frozenset(self.outgoing_clip_ids) | frozenset(self.incoming_clip_ids)

    def active(self, local_frame: int) -> bool:
        return self.first_frame <= local_frame < self.end_frame


@dataclass(frozen=True)
class TensorRenderPlan:
    width: int
    height: int
    frame_duration: Fraction
    frame_count: int
    layers: tuple[LayerSpec, ...]
    transitions: tuple[TransitionSpec, ...]
    scopes: tuple[ScopeSpec, ...] = ()

    @property
    def has_mixed_scope_rates(self) -> bool:
        """Whether a scope's completed native surface needs cadence conversion."""

        return any(
            scope.time_map.child_frame_duration != scope.time_map.parent_frame_duration
            for scope in self.scopes
        )

    @property
    def requires_random_access_sources(self) -> bool:
        """Whether recursive scope mapping can repeat or skip child frames."""

        # Even a 1x scope may select a non-zero source window. The current
        # prefetch worker assumes project-frame indices and cannot represent a
        # recursively mapped request schedule, so every explicit scope uses
        # the seekable lazy source pool until prefetch accepts a frame schedule.
        return bool(self.scopes) or any(
            layer.clock.retime_map is not None
            and any(segment.kind == "reverse" for segment in layer.clock.retime_map.segments)
            for layer in self.layers
        )

    def layers_at(self, frame: int) -> list[LayerSpec]:
        """Active layers bottom-to-top (hierarchical ``z_key``), across every scope."""

        return sorted((layer for layer in self.layers if layer.active(frame)), key=lambda layer: layer.z_key)

    def transitions_at(self, frame: int) -> list[TransitionSpec]:
        """Root-owned transitions active at OUTPUT frame ``frame`` (scope-owned ones run on local grids)."""

        return [item for item in self.transitions if item.scope_id is None and item.active(frame)]

    def scope(self, scope_id: str) -> ScopeSpec:
        for scope in self.scopes:
            if scope.scope_id == scope_id:
                return scope
        raise KeyError(scope_id)


# --------------------------------------------------------------------------- raster sources
#
# Raster sources (X5): stills, titles, captions, Custom Solid generators
# ---------------------------------------------------------------------
# Four compiled constructs share one pixel shape -- *one* image held for the
# clip's whole window -- and therefore one lowering here:
#
#   still image      ``clip.is_still`` (an asset with no duration), pixels on
#                    disk at ``clip.media_path``;
#   title / caption  ``clip.kind in {"title", "caption"}``, pixels rasterized
#                    before the render by ``text.resolve_text_clip_raster``;
#   Custom Solid     ``clip.generator_plan.execution == "solid_color"`` (also
#                    ``is_still``), pixels from ``resolve_generator_clip_raster``.
#
# The last two have no ``media_path`` at all: the *caller* rasterizes them (the
# executor does this in ``execute_render`` and passes the result as
# ``text_images``) and hands the mapping to ``build_tensor_plan(rasters=...)``.
# Both resolvers write a full project-size straight-alpha RGBA PNG once
# (``text.py:459`` / ``text.py:637``), so text placement is already baked in at
# the reference's 1920x1080 design space scaled by ``project_height / 1080``
# (font size ``text.py:441``; anchor and baseline ``text.py:475-479``) and
# nothing about it is re-derived here.
#
# What the reference does with a raster (the parity oracle):
#   * input is ``-loop 1 -framerate fps`` (``ffmpeg._append_video_input``,
#     ``ffmpeg.py:1338``), so the file-local source range is meaningless and the
#     source-range validations are skipped (``ffmpeg.py:1751``, ``:1765``);
#   * the video chain is just ``trim=duration=<render window>``
#     (``ffmpeg.py:9853``) -- *no* retime plan is built for a still or a
#     title/caption (``ffmpeg.py:1712``), and the ``setpts=PTS*1/speed`` speed
#     scaling is skipped for stills (``ffmpeg.py:9887``);
#   * everything downstream (``format=rgba``, geometry, opacity, overlay) is the
#     *same* code as for video, with ``FrameGeometry.source_width/height`` taken
#     from the probed raster (``ffmpeg.py:10501``).
# So the tensor side needs exactly two departures from the video path: a
# constant ``SourceClock`` and a ``RasterSource`` instead of a ``ClipDecoder``.
#
# Gaps: a spine ``<gap>`` never becomes a ``RenderClip``.  ``compiler.py:448``
# only records it as a spine item that advances the cursor, and
# ``story_ir.py:508`` keeps its connected children as ordinary clips.  There is
# nothing to skip and nothing to reject: the frames a gap covers simply have no
# active layer, and ``renderer.render_document`` paints its opaque black canvas
# -- which is what the reference's base ``color=black`` input shows there too.
# The same holds for a gap *inside* a compound / multicam story: no leaf, so
# the (transparent) group contributes nothing there.


def _requires_runtime_raster(clip: RenderClip) -> bool:
    """True when the caller must supply a rasterized image for this clip.

    Mirrors ``ffmpeg._requires_runtime_raster`` (``ffmpeg.py:1348``) and
    ``executor._runtime_raster_clips`` (``executor.py:109``) -- the same
    predicate the reference uses to decide whether a clip's FFmpeg input is the
    resolved PNG.  It is stated a third time here (rather than imported)
    because ``executor`` imports this package, so the dependency may only run
    one way, and importing the 12k-line ``ffmpeg`` module for six lines of
    predicate would drag the legacy emitter into every tensor import.

    ``video_disposition`` is authoritative: a title whose font did not resolve
    comes back as ``omit_transparent`` and owns *no* raster.
    """

    disposition = clip.video_disposition
    if disposition is None:
        raise TensorRenderUnsupported(
            f"{clip.path}: enabled video clip has no compiled video disposition"
        )
    return bool(
        clip.enabled
        and disposition.execution == "composite"
        and (
            bool(clip.missing_media_locators)
            or clip.kind in {"title", "caption"}
            or (
                clip.generator_plan is not None
                and clip.generator_plan.execution == "solid_color"
            )
        )
    )


def _resolve_source(clip: RenderClip, rasters: Mapping[str, Path]) -> tuple[Path, SourceKind]:
    """Return ``(pixel file, source kind)`` for one composited clip, loudly.

    Pythonese:
    1. Refuse a clip kind whose pixels this module cannot locate at all (an
       unlowered container), before anything looks at the filesystem.
    2. If the clip owns a runtime raster (title / caption / Custom Solid), the
       caller must have resolved it; a missing entry is the caller's bug, not a
       reason to invent pixels, so reject by name.
    3. Otherwise a raster entry for this clip is a caller/compiler
       disagreement -- the reference raises the same conflict
       (``ffmpeg.py:1657``) -- and the clip's own media is the source.
    4. A generator that is not Custom Solid has no portable raster adapter at
       all, so name it here rather than let it fall through to "no media file".
    5. A still asset uses its media file but the *raster* clock; anything else
       is an ordinary decoded video stream.

    ``video``, not just ``asset-clip``, is an accepted kind: the compiler
    lowers ``<asset-clip>`` and ``<video>`` through the very same branch
    (``compiler.py:981``), and Final Cut writes still images and generators as
    ``<video>`` elements.

    Main callers: ``_lower_leaf``.
    """

    if clip.kind not in {"asset-clip", "video", "title", "caption"}:
        raise reject("clip kind", f"{clip.path}: kind {clip.kind!r}")
    if _requires_runtime_raster(clip):
        path = rasters.get(clip.id)
        if path is None:
            raise reject(
                "runtime raster not resolved by the caller",
                f"{clip.path}: {clip.kind} composites a runtime raster but the caller passed no "
                "rasters[clip.id] (pass the executor's text_images to build_tensor_plan)",
            )
        if not path.is_file():
            raise reject(
                "media file (missing or unreadable)",
                f"{clip.path}: resolved runtime raster {path} is not a file",
            )
        return path, "raster"
    if clip.id in rasters:
        raise reject(
            "runtime raster for a non-raster clip",
            f"{clip.path}: caller supplied {rasters[clip.id]} for a clip whose compiled "
            "disposition does not composite a runtime raster",
        )
    if clip.generator_plan is not None:
        # Custom Solid is the only calibrated generator adapter; the compiler
        # turns every other Motion generator into an explicit transparent
        # omission (``compiler.py:1027``), so reaching this point means some
        # caller forced its disposition back to composite.
        raise reject(
            "generator (not Custom Solid)",
            f"{clip.path}: generator execution {clip.generator_plan.execution!r}",
        )
    if clip.media_path is None or not clip.media_path.is_file():
        raise reject(
            "media file (missing or unreadable)",
            f"{clip.path}: media {clip.media_path} is not a file",
        )
    return clip.media_path, ("raster" if clip.is_still else "video")


# --------------------------------------------------------------------------- rejection


def _tensor_blend_mode(value: str | None, *, path: str) -> str:
    """Resolve one layer or group blend name for the tensor parent fold.

    Main callers:
    - ``_reject_unsupported_clip`` for leaf capability validation.
    - ``_classify_scopes`` for group capability validation.
    - ``_lower_leaf`` and ``_lower_scope`` when storing the canonical mode on
      the placed item.

    Why this exists:
    The CPU graph and tensor plan must share the same mode vocabulary. Calling
    ``resolve_blend_mode`` here preserves loud errors for unknown or explicitly
    uncalibrated modes instead of turning them into Normal by omission.
    """

    try:
        return resolve_blend_mode(value).canonical_name
    except CompositorError as error:
        raise reject(
            "blend mode (unknown or uncalibrated)",
            f"{path}: blend mode {value!r}: {error}",
        ) from error


def _reject_unsupported_clip(clip: RenderClip, *, source_kind: SourceKind) -> None:
    """Name the first construct outside the supported class, loudly (see support.py).

    The clip *kind* and where its pixels live are already settled by
    ``_resolve_source``; ``source_kind`` is its verdict, and it selects which
    time rules apply (a raster has no source timeline).
    """

    _tensor_blend_mode(clip.blend_mode, path=clip.path)
    if clip.conform_type not in ("fit", "fill", "none"):
        raise reject("conform (other)", f"{clip.path}: conform {clip.conform_type!r}")
    if source_kind == "raster":
        # A raster has no temporal content, so retiming it is the identity and
        # the reference simply never builds a retime plan for one
        # (``ffmpeg.py:1712``).  Ignore ``retime_map`` exactly as the reference
        # does.  The one uncovered case is a title/caption (not ``is_still``)
        # with ``speed != 1``: there the emitted graph still applies
        # ``setpts=PTS*1/speed`` to the trimmed constant stream
        # (``ffmpeg.py:9887``), changing the layer stream's length in a way the
        # tensor loop has no equivalent for -- refuse it rather than guess.
        if not clip.is_still and clip.speed != 1:
            raise reject("raster speed (title / caption)", f"{clip.path}: speed {clip.speed}")
    elif clip.retime_map is not None:
        if any(segment.kind == "reverse" for segment in clip.retime_map.segments) and clip.source_frame_duration is None:
            raise TensorRenderUnsupported(
                f"{clip.path}: reverse playback requires an exact source format frame duration"
            )
        if clip.retime_map.timeline_start != 0 or clip.retime_map.timeline_duration != clip.duration:
            raise TensorRenderUnsupported(
                f"{clip.path}: retime map domain [{clip.retime_map.timeline_start}, "
                f"{clip.retime_map.timeline_end}] is not the clip interval [0, {clip.duration}]"
            )
    intrinsics = clip.spatial_intrinsics
    if intrinsics is not None:
        display = intrinsics.display
        if display is not None and (display.pixel_aspect_h, display.pixel_aspect_v) != (1, 1):
            raise reject("non-square pixel aspect", f"{clip.path}: {display}")
        for name in ("stereo", "stabilization", "transform_360", "reorientation_360", "orientation_360", "rolling_shutter", "cinematic"):
            if getattr(intrinsics, name) is not None:
                raise reject("spatial intrinsics (360 / stereo / stabilization / rolling shutter)", f"{clip.path}: {name}")


def _reject_unsupported_source(
    clip: RenderClip,
    probe: VideoProbe,
    *,
    media_path: Path,
    source_kind: SourceKind,
) -> Optional[SourceColor]:
    """Refuse a probed raster the sampler would place wrongly (rotation / non-square pixels).

    ``media_path`` is the file the pixels actually come from, which for a title,
    caption or Custom Solid is the caller-resolved PNG rather than
    ``clip.media_path`` (there is none).
    """

    if probe.rotation_degrees % 360 not in {0, 90, 180, 270}:
        raise TensorRenderUnsupported(
            f"{clip.path}: {media_path} display rotation {probe.rotation_degrees} is not a quarter turn"
        )
    if probe.sample_aspect_ratio != (1, 1):
        raise reject("non-square pixel aspect", f"{clip.path}: {media_path} SAR {probe.sample_aspect_ratio}")
    if source_kind == "video":
        # Colour-in policy (X10): resolve supported Rec.2020 HLG/PQ through the
        # frozen SDR LUT and fail at plan time on malformed HDR tags, exotic
        # matrices or pixel formats instead of on the first decoded frame.
        return check_source_color(probe, subject=f"{clip.path}: {media_path}")
    return None


# --------------------------------------------------------------------------- group scopes
#
# Group scopes (X6): inert folds + rendered scope canvases
# --------------------------------------------------------
# A compound (``ref-clip``), multicam (``mc-clip``: outer instance scope + inner
# ``mc-source`` angle scope, one video leaf per angle story item), ``clip`` /
# ``sync-clip`` / ``audition`` container compiles to a ``group_scope`` plus the
# ordinary leaf clips below it (``ancestor_clip_ids``).  The reference
# rasterizes each scope: children composed on the container canvas
# (``_compose_item_batch``), then the group chain (retime, effects, crop /
# conform / transform), then the finished raster joins the parent stack at
# (``scope.lane``, ``scope.document_order``) with the group's opacity
# (``ffmpeg._compose_group_scopes`` -> ``_group_video_chain`` /
# ``_legacy_group_video_chain``; the plan-authoritative emitter's
# ``composition_cpu_recursive_scope`` does the same for the Fit / crop-camera
# shapes it accepts).
#
# The tensor plan mirrors that with two kinds of scope:
#
#   INERT (folded)   a scope whose raster fold provably changes no pixel: Fit
#               onto a same-aspect parent canvas / None onto an equal one, no
#               transform / crop / opacity / blend / effects, container frame
#               rate == parent.  It owns no canvas: its placement is a uniform
#               affine composed into its descendants' ``canvas_to_owner``
#               (``_scope_placement``), its descendants sort at ITS position in
#               the owner stack (their ``z_prefix`` carries the inert chain's
#               (lane, document_order) pairs -- the reference composes a group
#               as one raster there), and a transition on it takes its LEAVES
#               as side participants (pixel-identical to composing the inert
#               surface).  This is the exact flat fast path (X6-lite).
#
#   RENDERED (``ScopeSpec``)   everything else: the renderer composes the
#               scope's children on a transparent canvas of the container size
#               (``_FrameComposer.render_scope``), applies the group effects,
#               and places that canvas into the owner's canvas through the
#               scope's own ``GeometryPlan`` (crop / conform / transform /
#               animation -- the same homography as a leaf, source raster =
#               the container), multiplied by the group ``OpacityPlan``.
#               Its ``ScopeTimeMap`` samples the native completed surface and
#               maps the exact authored retime at this one recursive boundary.
#               Transitions on it take the finished scope as the side
#               participant (typed frontier; ledger row 5).
#
# Ownership: the reference never clips a leaf to its container's window
# (``_plan_group_execution``: a root scope's window is the union of its leaves'
# render windows; compound / multicam stories are already clipped by
# ``_clip_story_nodes``), so a leaf keeps its own owned window; a rendered scope
# owns its transition-expanded interval (``_expanded_window``) on its owner's
# grid, or its pad interval mapped through a retimed ancestor
# (``_retimed_item_frames``).
#
# Canvases: a scope's children render on ITS CONTAINER (``container_context``,
# else the scope's own canvas context, else the project); its geometry maps that
# container onto the IMMEDIATE parent scope's container (the project at the
# root) -- multicam's outer / inner scopes both carry the multicam format as
# container, so the inner is an identity Fit and the outer conforms the format
# onto the project.  A leaf's own canvas (``clip.canvas_context``) must equal
# its immediate scope's container -- checked loudly.
#
# Rejects (``support.py``): a non-Normal group blend and group reverse retiming.
# Forward/freeze piecewise maps and differing container rates execute at each
# recursive ``ScopeSpec.time_map`` boundary.


def _is_constant_speed_retime(scope: RenderClip) -> bool:
    """True when the group's retime map is one linear forward segment (constant speed)."""

    retime_map = scope.retime_map
    assert retime_map is not None
    if len(retime_map.segments) != 1:
        return False
    segment = retime_map.segments[0]
    if segment.timeline_start != 0 or segment.timeline_end != scope.duration or segment.kind != "forward":
        return False
    return (segment.source_end - segment.source_start) == scope.duration * scope.speed


def _is_retimed(scope: RenderClip) -> bool:
    """A group whose source pad is retimed (``ffmpeg._source_instance_requires_retime`` + speed)."""

    if scope.speed != 1:
        return True
    retime_map = scope.retime_map
    if retime_map is None:
        return False
    segments = retime_map.segments
    return not (len(segments) == 1 and segments[0].kind == "forward" and segments[0].rate == 1)


def _has_spatial_adjustment(clip: RenderClip) -> bool:
    """A non-identity transform, a transform animation or a corner pin (``ffmpeg._has_spatial_adjustment``)."""

    transform = clip.transform
    animation = clip.transform_animation
    if transform is not None and transform.enabled:
        if (
            tuple(transform.position) != (0.0, 0.0)
            or tuple(transform.scale) != (1.0, 1.0)
            or transform.rotation != 0.0
            or tuple(transform.anchor) != (0.0, 0.0)
        ):
            return True
        if animation is not None and any((animation.position, animation.scale, animation.rotation, animation.anchor)):
            return True
    return bool(clip.corner_pin is not None and clip.corner_pin.enabled)


def _has_opacity(clip: RenderClip) -> bool:
    return clip.blend_opacity != 1.0 or clip.opacity_animation is not None or clip.opacity_fade is not None


@dataclass(frozen=True)
class _PlanCanvas:
    """Resolve every authored canvas into one output-resolution coordinate system.

    Source rasters are deliberately excluded: decoded video and runtime title
    images retain their native dimensions and are sampled by the existing
    geometry kernels. Only composition canvases are scaled here.

    Main callers:
    - ``build_tensor_plan`` and its leaf/scope lowering helpers.
    """

    document: RenderDocument
    output_resolution: Optional[OutputResolution]

    @property
    def root(self) -> tuple[int, int]:
        if self.output_resolution is None:
            return self.document.width, self.document.height
        return self.output_resolution.width, self.output_resolution.height

    def scale(self, width: int, height: int) -> tuple[int, int]:
        if self.output_resolution is None:
            return width, height
        if (width, height) == (self.document.width, self.document.height):
            return self.root
        return (
            max(1, round(width * self.output_resolution.scale_x)),
            max(1, round(height * self.output_resolution.scale_y)),
        )


def _authored_canvas_size(clip: RenderClip, document: RenderDocument) -> tuple[int, int]:
    return clip.canvas_width or document.width, clip.canvas_height or document.height


def _authored_container_size(scope: RenderClip, document: RenderDocument) -> tuple[int, int]:
    context = scope.container_context
    if context is not None and context.width and context.height:
        return int(context.width), int(context.height)
    return _authored_canvas_size(scope, document)


def _container_size(scope: RenderClip, document: RenderDocument, canvas: _PlanCanvas) -> tuple[int, int]:
    """The canvas a scope's children are composed on."""

    context = scope.container_context
    if context is not None and context.width and context.height:
        return canvas.scale(int(context.width), int(context.height))
    return _canvas_size(scope, document, canvas)


def _canvas_size(clip: RenderClip, document: RenderDocument, canvas: _PlanCanvas) -> tuple[int, int]:
    return canvas.scale(
        clip.canvas_width or document.width,
        clip.canvas_height or document.height,
    )


@dataclass(frozen=True)
class _ScopeTree:
    """The compiled group scopes with their parent links and inert / rendered verdicts.

    ``parent`` maps a scope id to its immediate ancestor scope id (None at the
    root); ``rendered`` names the scopes that own a canvas.  Ancestor lists are
    outermost first (``RenderClip.ancestor_clip_ids`` order).
    """

    scopes: Mapping[str, RenderClip]
    parent: Mapping[str, Optional[str]]
    rendered: frozenset[str]

    def ancestors(self, clip: RenderClip) -> tuple[RenderClip, ...]:
        return tuple(self.scopes[gid] for gid in clip.ancestor_clip_ids if gid in self.scopes)

    def parent_canvas(self, scope: RenderClip, document: RenderDocument, canvas: _PlanCanvas) -> tuple[int, int]:
        """The immediate parent scope's container (the project at the root)."""

        parent_id = self.parent[scope.id]
        if parent_id is None:
            return canvas.root
        return _container_size(self.scopes[parent_id], document, canvas)

    def authored_parent_canvas(self, scope: RenderClip, document: RenderDocument) -> tuple[int, int]:
        parent_id = self.parent[scope.id]
        if parent_id is None:
            return document.width, document.height
        return _authored_container_size(self.scopes[parent_id], document)


def _classify_scopes(document: RenderDocument, canvas: _PlanCanvas) -> _ScopeTree:
    """Decide, per scope, inert (folded) vs rendered, and reject what neither can express.

    Pythonese:
    1. Reject the constructs no scope kind lowers: an unknown or uncalibrated
       group blend mode, a reverse group retime map, or an unported group
       effect handler.
    2. A scope is INERT when its fold changes no pixel: Fit onto a same-aspect
       parent container (or None onto an equal one), no spatial adjustment, no
       crop, unit static opacity, no effects.  Everything else is RENDERED.

    Main callers: ``build_tensor_plan``.
    """

    scopes = {scope.id: scope for scope in document.group_scopes}
    parent: dict[str, Optional[str]] = {}
    rendered: set[str] = set()
    for scope in document.group_scopes:
        parent_ids = [gid for gid in scope.ancestor_clip_ids if gid in scopes]
        parent[scope.id] = parent_ids[-1] if parent_ids else None
    tree = _ScopeTree(scopes=scopes, parent=parent, rendered=frozenset())
    for scope in document.group_scopes:
        blend_mode = _tensor_blend_mode(scope.blend_mode, path=scope.path)
        for effect in scope.effects:
            if effect.execution == "apply" and (effect.handler or "") not in effect_ports.EFFECT_PORTS:
                raise reject("effect (unported handler)", f"{scope.path}: {effect.name!r} (handler {effect.handler!r})")
        if scope.conform_type not in ("fit", "fill", "none"):
            raise reject("conform (other)", f"{scope.path}: conform {scope.conform_type!r}")
        container_frame_duration = scope.container_frame_duration or document.frame_duration
        canvas_frame_duration = scope.canvas_frame_duration or document.frame_duration
        inner_width, inner_height = _container_size(scope, document, canvas)
        outer_width, outer_height = tree.parent_canvas(scope, document, canvas)
        inert_placement = (
            (scope.conform_type == "fit" and inner_width * outer_height == inner_height * outer_width)
            or (scope.conform_type == "none" and (inner_width, inner_height) == (outer_width, outer_height))
        )
        inert = (
            inert_placement
            and not _has_spatial_adjustment(scope)
            and not (scope.crop is not None and scope.crop.enabled)
            and not _has_opacity(scope)
            and blend_mode == "Normal"
            and not any(effect.execution == "apply" for effect in scope.effects)
        )
        # A retimed group that holds a transition cannot be folded: the transition
        # runs on the group's pad clock and its window is scheduled on the OWNER's
        # local grid, so the group itself must be the owner (composed on its pad
        # clock, then retimed as one surface).  Folded, the window would be tested
        # against the parent's output clock -- a hard cut plus a fade-from-black.
        holds_transition = _is_retimed(scope) and any(
            item.handler is not None and scope.id in item.ancestor_group_ids for item in document.transitions
        )
        # Retime and cadence conversion are semantic boundaries even when the
        # spatial fold is identity. They must consume the completed child
        # surface, never be distributed over leaves.
        boundary_clock = _is_retimed(scope) or container_frame_duration != canvas_frame_duration
        if not inert or holds_transition or boundary_clock:
            rendered.add(scope.id)
    return _ScopeTree(scopes=scopes, parent=parent, rendered=frozenset(rendered))


def _scope_placement(
    scope: RenderClip,
    document: RenderDocument,
    tree: _ScopeTree,
    canvas: _PlanCanvas,
) -> np.ndarray:
    """The inert scope's container -> parent-container affine (Fit / equal-size None)."""

    inner_width, inner_height = _container_size(scope, document, canvas)
    outer_width, outer_height = tree.parent_canvas(scope, document, canvas)
    if scope.conform_type == "fit":
        scale = min(outer_width / inner_width, outer_height / inner_height)
        return np.array(
            [[scale, 0.0, (outer_width - scale * inner_width) / 2.0],
             [0.0, scale, (outer_height - scale * inner_height) / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return np.eye(3, dtype=np.float64)  # "none" with equal sizes (validated by _classify_scopes)


@dataclass(frozen=True)
class _Ownership:
    """What an item's ancestor scopes resolve to (see the section comment).

    ``owner`` is the nearest rendered ancestor (None = root); ``canvas_to_owner``
    composes the inert scopes between the item and its owner; ``nearest`` is the
    immediate ancestor scope (the canvas the item's geometry is authored
    against). Every retimed scope is rendered, so retiming is never propagated
    globally through this ownership record.
    """

    owner: Optional[RenderClip]
    nearest: Optional[RenderClip]
    canvas_to_owner: np.ndarray
    # The inert scopes between the owner and the item, outermost first (the
    # item's ``z_prefix``: it sorts at their position in the owner stack).
    inert_chain: tuple[RenderClip, ...] = ()

    @property
    def z_prefix(self) -> tuple[tuple[int, int], ...]:
        return tuple((scope.lane, scope.document_order) for scope in self.inert_chain)

def _resolve_ownership(
    clip: RenderClip,
    document: RenderDocument,
    tree: _ScopeTree,
    canvas: _PlanCanvas,
) -> _Ownership:
    """Walk the item's ancestor scopes (outermost first) into one ``_Ownership``.

    The placement composes inner-first: iterating outer -> inner, ``matrix =
    matrix @ placement(scope)`` yields ``P(outer) @ P(inner)``, i.e. inner
    container -> ... -> owner canvas.  A rendered ancestor resets the fold
    (its own placement is its ``ScopeSpec``'s business).
    """

    ancestors = tree.ancestors(clip)
    owner: Optional[RenderClip] = None
    matrix = np.eye(3, dtype=np.float64)
    inert_chain: list[RenderClip] = []
    for scope in ancestors:
        if scope.id in tree.rendered:
            owner = scope
            matrix = np.eye(3, dtype=np.float64)
            inert_chain = []
        else:
            matrix = matrix @ _scope_placement(scope, document, tree, canvas)
            inert_chain.append(scope)
    return _Ownership(
        owner=owner,
        nearest=ancestors[-1] if ancestors else None,
        canvas_to_owner=matrix,
        inert_chain=tuple(inert_chain),
    )


# --------------------------------------------------------------------------- lowering


def _source_clock(
    clip: RenderClip,
    *,
    source_kind: SourceKind,
    clip_start: Fraction,
    clip_duration: Fraction,
    render_start: Fraction,
    render_end: Fraction,
    frame_duration: Fraction,
) -> SourceClock:
    """Choose the legacy path (constant raster vs linear 1x vs exact map with holds).

    All times are on the layer's local clock; ``clip_start`` / ``clip_duration``
    are the clip's own interval there.  ``frame_duration`` is the layer's output
    cadence, carried onto the clock so a reverse endpoint hold can repeat the
    last real output frame instead of rolling past the source-0 boundary.
    """

    if source_kind == "raster":
        # Stills, titles, captions and solids are one looped image: the
        # reference skips both the retime plan and the speed setpts for them
        # (``ffmpeg.py:1712`` / ``ffmpeg.py:9887``) and only trims the stream.
        return SourceClock.constant(
            clip_start=clip_start,
            clip_duration=clip_duration,
            frame_duration=frame_duration,
        )
    speed = clip.speed
    pre_roll = clip_start - render_start
    post_roll = render_end - (clip_start + clip_duration)
    expanded = render_start != clip_start or render_end != clip_start + clip_duration
    expanded_source_start = clip.source_start - pre_roll * speed
    expanded_source_end = clip.source_start + (clip_duration + post_roll) * speed
    requires_hold = expanded and (
        expanded_source_start < 0
        or (clip.asset_source_duration is not None and expanded_source_end > clip.asset_source_duration)
    )
    retime_map = clip.retime_map
    simple_forward = retime_map is None or (
        len(retime_map.segments) == 1
        and retime_map.segments[0].kind == "forward"
        and retime_map.segments[0].rate == 1
    )
    if speed == 1 and simple_forward:
        return SourceClock(
            clip_start=clip_start,
            clip_duration=clip_duration,
            source_start=clip.source_start,
            speed=Fraction(1),
            retime_map=None,
            hold_ends=requires_hold,
            frame_duration=frame_duration,
        )
    if retime_map is None:
        return SourceClock(
            clip_start=clip_start,
            clip_duration=clip_duration,
            source_start=clip.source_start,
            speed=speed,
            retime_map=None,
            hold_ends=requires_hold,
            frame_duration=frame_duration,
        )
    return SourceClock(
        clip_start=clip_start,
        clip_duration=clip_duration,
        source_start=clip.source_start,
        speed=speed,
        retime_map=retime_map,
        hold_ends=True,
        frame_duration=frame_duration,
    )


def _geometry_plan(
    clip: RenderClip,
    *,
    frame: FrameGeometry,
    clip_start: Fraction,
    clip_duration: Fraction,
    render_start: Fraction,
    render_end: Fraction,
) -> GeometryPlan:
    animation = clip.transform_animation
    transform_is_animated = bool(
        animation and any((animation.position, animation.scale, animation.rotation, animation.anchor))
    )
    try:
        return GeometryPlan(
            frame=frame,
            window=GeometryWindow(
                clip_start=clip_start,
                clip_duration=clip_duration,
                render_start=render_start,
                render_duration=render_end - render_start,
            ),
            transform=clip.transform,
            transform_animation=clip.transform_animation,
            crop=clip.crop,
            conform=clip.conform_type,
            corners=clip.corner_pin,
            allow_mirrored_scale=not transform_is_animated,
        )
    except GeometryError as error:
        raise TensorRenderUnsupported(f"{clip.path}: invalid geometry: {error}") from error


def _opacity_plan(clip: RenderClip, *, clip_start: Fraction, clip_duration: Fraction, render_start: Fraction, render_end: Fraction) -> OpacityPlan:
    """Mirror ``ffmpeg._layer_opacity_plan``: handles clamp, sub-frame snaps collapse to the window."""

    pre_roll = clip_start - render_start
    post_roll = render_end - (clip_start + clip_duration)
    if pre_roll < 0 or post_roll < 0:
        if clip.opacity_animation is not None or clip.opacity_fade is not None:
            raise TensorRenderUnsupported(
                f"{clip.path}: trimmed animated opacity requires source-window clock slicing"
            )
        pre_roll = Fraction(0)
        post_roll = Fraction(0)
        clip_duration = render_end - render_start
    return OpacityPlan(
        window=CompositorWindow(
            clip_duration=clip_duration,
            transition_pre_roll=pre_roll,
            transition_post_roll=post_roll,
        ),
        static_opacity=clip.blend_opacity,
        animation=clip.opacity_animation,
        fade=clip.opacity_fade,
        expression_time_origin=render_start,
    )


def _effect_specs(
    effects: tuple[ResolvedEffect, ...],
    *,
    clip_path: str,
    frame_origin: int,
    canvas: tuple[int, int],
    authored_canvas: tuple[int, int],
    frame_duration: Fraction,
    clip_duration: Fraction,
    source_start: Fraction,
    playback_rate: Fraction,
    retime_map: Optional[RetimeMap] = None,
    source_colorspace: str = "unknown",
    source_color_range: str = "unknown",
    reference_effect_link: Optional[str] = None,
) -> tuple[EffectSpec, ...]:
    """Lower an owner's applied effects (leaf clip or folded group) through the effect ports.

    ``execution != "apply"`` entries are the reference's warn-and-ignore identities
    (``ResolvedEffect.execution``), so they lower to nothing here too.  ``frame_origin``
    is the frame (on the layer's local frame grid, ``LayerSpec.local_frame``) the
    effect's ``N`` counts from: the leaf's (or the owning group chain's) first frame.
    """

    ctx = effect_ports.LowerContext(
        clip_path=clip_path,
        width=canvas[0],
        height=canvas[1],
        frame_duration=frame_duration,
        clip_duration=clip_duration,
        source_start=source_start,
        playback_rate=playback_rate,
        retime_map=retime_map,
        coordinate_scale_x=canvas[0] / authored_canvas[0],
        coordinate_scale_y=canvas[1] / authored_canvas[1],
        source_colorspace=source_colorspace,
        source_color_range=source_color_range,
        reference_effect_link=reference_effect_link,
    )
    return tuple(
        effect_ports.lower_effect(effect, ctx, frame_origin=frame_origin)
        for effect in effects
        if effect.execution == "apply"
    )


def _expanded_window(clip: RenderClip) -> tuple[Fraction, Fraction]:
    """Return ``(render_start, render_end)`` on the item's local clock: its interval expanded over its transitions.

    Exactly ``ffmpeg._expanded_schedule`` (the compiler marks ``transition_in/out``
    on the spine item and on every descendant overlapping the transition,
    ``compiler._apply_group_transition``); a group scope carries its own spine
    transitions the same way (``_completed_group_output_window``).
    """

    start = clip.absolute_start
    end = clip.end
    if clip.transition_in is not None:
        start = min(start, clip.transition_in.absolute_start)
    if clip.transition_out is not None:
        end = max(end, clip.transition_out.end)
    return max(Fraction(0), start), end


def _retimed_item_frames(
    clip: RenderClip,
    retimed: RenderClip,
    local_clock: LocalClock,
    *,
    document: RenderDocument,
) -> Optional[tuple[int, int, int, int]]:
    """Output ``(first_frame, end_frame)`` and local ``(first pad frame, pad frame count)`` of an item under a retimed group.

    Pythonese:
    1. The item lives on pad frames ``[K1, K2)`` of its group's pad
       (``K = ceil((pad time - origin) / fd)``: the reference enables the item
       on the pad frames whose instant is >= its start and < its end).
    2. Its output frames are those whose selected pad frame lies in that
       range (``LocalClock.output_frame_range``), capped at the group's own
       owned end (the end hold selects the last pad frame forever).
    3. A transition *on the retimed group* (the compiler marks the item)
       expands it over the handle only when it sits at the corresponding pad
       edge: the held pad frame shows exactly the items active on it.  A
       transition *inside* the retimed group is on the pad clock already
       (``clip.transition_in/out`` inside the pad interval) -- its expansion
       is part of the item's pad interval.
    """

    fd = document.frame_duration
    origin = local_clock.origin
    start, end_time = clip.absolute_start, clip.end
    for item in (clip.transition_in, clip.transition_out):
        if item is not None and retimed.id in item.ancestor_group_ids:
            start = min(start, item.absolute_start)
            end_time = max(end_time, item.end)
    pad_first = math.ceil((max(start, origin) - origin) / fd)
    pad_end = math.ceil((end_time - origin) / fd)
    group_window = resolve_owned_frame_window(retimed.absolute_start, retimed.end, frame_duration=fd)
    first, end = local_clock.output_frame_range(pad_first, pad_end, output_end=group_window.end_frame)
    last_pad_frame = local_clock.pad_frame(retimed.end)
    if clip.transition_in is not None and retimed.id not in clip.transition_in.ancestor_group_ids and pad_first <= 0 < pad_end:
        first = min(first, resolve_owned_frame_window(clip.transition_in.absolute_start, clip.transition_in.end, frame_duration=fd).first_frame)
    if clip.transition_out is not None and retimed.id not in clip.transition_out.ancestor_group_ids and pad_first <= last_pad_frame < pad_end:
        end = max(end, resolve_owned_frame_window(clip.transition_out.absolute_start, clip.transition_out.end, frame_duration=fd).end_frame)
    if end <= first:
        # Shorter than one output frame at this rate: no selected pad frame is
        # the item's, so the reference never shows it (an empty range, not an
        # error) -- the caller drops the item.
        return None
    return first, end, pad_first, max(1, pad_end - pad_first)


@dataclass(frozen=True)
class _ItemWindow:
    """The frame windows of one placed item on both clocks (see ``PlacedItem``)."""

    first_frame: int
    end_frame: int
    render_start: Fraction
    render_end: Fraction
    local_first_frame: int
    local_frame_count: int


def _item_window(
    clip: RenderClip,
    ownership: _Ownership,
    local_clock: Optional[LocalClock],
    *,
    document: RenderDocument,
    frame_duration: Optional[Fraction] = None,
) -> Optional[_ItemWindow]:
    """Resolve the OUTPUT owned window and the LOCAL render window of a leaf or a rendered scope."""

    frame_duration = frame_duration or document.frame_duration
    if local_clock is None:
        start, end = _expanded_window(clip)
        window = resolve_owned_frame_window(start, end, frame_duration=frame_duration)
        return _ItemWindow(window.first_frame, window.end_frame, window.start, window.end, window.first_frame, window.frame_count)
    raise TensorRenderUnsupported(
        f"{clip.path}: legacy flattened LocalClock reached recursive scope lowering"
    )


def _lower_leaf(
    clip: RenderClip,
    *,
    document: RenderDocument,
    tree: _ScopeTree,
    rasters: Mapping[str, Path],
    probes: dict[Path, VideoProbe],
    canvas: _PlanCanvas,
    decode_policy: DecodePolicy = DecodePolicy.NATIVE,
) -> Optional[LayerSpec]:
    """Lower one enabled video leaf (plus its folded inert ancestors) to a ``LayerSpec``.

    Returns ``None`` only for a retimed-group leaf that owns no output frame
    (``_retimed_item_frames``); every other unsupported case raises.

    Pythonese:
    1. Locate the pixels (media file or caller raster) and reject unsupported
       clip / source constructs by name.
    2. Resolve the ancestors: owner scope, inert placement, retimed clock.
    3. Resolve the OUTPUT owned frame window (midpoint ownership over the
       transition-expanded interval) -- this is when the layer is composited
       -- and its LOCAL clock window -- what every kernel evaluates on.
    4. Build the source clock, geometry and opacity kernels on the local clock
       with the clip's own local interval.
    5. Lower the leaf's effects (``N`` from the leaf's first local frame) on
       the leaf's conformed canvas.
    6. Ask ``decode_policy.resolve_decode_raster`` which raster the decoder
       should produce for this leaf (native under ``DecodePolicy.NATIVE``).

    Main callers: ``build_tensor_plan``.
    """

    media_path, source_kind = _resolve_source(clip, rasters)
    _reject_unsupported_clip(clip, source_kind=source_kind)
    probe = probes.get(media_path)
    if probe is None:
        probe = probes[media_path] = probe_video(media_path)
    source_color = _reject_unsupported_source(
        clip,
        probe,
        media_path=media_path,
        source_kind=source_kind,
    )

    ownership = _resolve_ownership(clip, document, tree, canvas)
    reverse_decode_cache = (
        clip.retime_map is not None
        and any(segment.kind == "reverse" for segment in clip.retime_map.segments)
    ) or any(
        scope.retime_map is not None
        and any(segment.kind == "reverse" for segment in scope.retime_map.segments)
        for scope in tree.ancestors(clip)
    )
    frame_duration = (
        ownership.owner.container_frame_duration
        if ownership.owner is not None and ownership.owner.container_frame_duration is not None
        else document.frame_duration
    )
    local_clock = None
    window = _item_window(
        clip,
        ownership,
        local_clock,
        document=document,
        frame_duration=frame_duration,
    )
    if window is None:
        return None
    clip_start, clip_duration = clip.absolute_start, clip.duration
    render_start, render_end = window.render_start, window.render_end

    clock = _source_clock(
        clip,
        source_kind=source_kind,
        clip_start=clip_start,
        clip_duration=clip_duration,
        render_start=render_start,
        render_end=render_end,
        frame_duration=frame_duration,
    )
    first_local_time = local_clock.time(window.first_frame * frame_duration) if local_clock is not None else render_start
    if clock.source_time(first_local_time) < 0:
        raise TensorRenderUnsupported(
            f"{clip.path}: first sampled source instant {clock.source_time(first_local_time)} is negative"
        )
    canvas_width, canvas_height = _canvas_size(clip, document, canvas)
    if ownership.nearest is not None and (canvas_width, canvas_height) != _container_size(ownership.nearest, document, canvas):
        raise TensorRenderUnsupported(
            f"{clip.path}: leaf canvas {canvas_width}x{canvas_height} differs from its scope "
            f"{ownership.nearest.path} container {_container_size(ownership.nearest, document, canvas)}"
        )
    display_width, display_height = probe.width, probe.height
    if probe.rotation_degrees % 360 in {90, 270}:
        display_width, display_height = display_height, display_width
    frame = FrameGeometry(
        source_width=display_width,
        source_height=display_height,
        project_width=canvas_width,
        project_height=canvas_height,
    )
    geometry = _geometry_plan(
        clip, frame=frame, clip_start=clip_start, clip_duration=clip_duration,
        render_start=render_start, render_end=render_end,
    )
    try:
        opacity = _opacity_plan(
            clip, clip_start=clip_start, clip_duration=clip_duration,
            render_start=render_start, render_end=render_end,
        )
    except CompositorError as error:
        raise TensorRenderUnsupported(f"{clip.path}: invalid opacity automation: {error}") from error

    # Effect stage (E6): the reference emits leaf effects after crop/conform and
    # before the spatial tail (``ffmpeg._video_chain`` -> ``_ordered_effect_filters``
    # appended to ``initial_filters``; the fused ``_static_conform_affine_filter``
    # path is refused when ``clip.effects`` is non-empty), so they run on the
    # conformed canvas (``clip.canvas_*``, the same ``FrameGeometry`` project size
    # the reference's ``_geometry_plan`` uses).  ``N`` counts from the layer's
    # first render frame (``_timeline_placement_filters(layer.render_start)`` after
    # ``setpts=PTS-STARTPTS``).
    crop_mode: CropMode = None
    if clip.crop is not None and clip.crop.enabled:
        crop_mode = clip.crop.mode.strip().lower()  # type: ignore[assignment]
    # Rasters (titles / stills / solids) reach the reference's effect chain as untagged RGBA;
    # video sources carry their stream tags (frame-header tags for ProRes, see probe_video).
    # The negotiated link of the reference's effect stack (pixel format / matrix / range)
    # depends on those tags AND on whether a resampling geometry stage precedes the effects
    # (``fx_color.reference_effect_link``); YUV-bridge ports lower against it.
    source_colorspace = probe.color_space if source_kind == "video" else "unknown"
    source_color_range = probe.color_range if source_kind == "video" else "unknown"
    effects = _effect_specs(
        clip.effects,
        clip_path=clip.path,
        frame_origin=window.local_first_frame,
        canvas=(canvas_width, canvas_height),
        authored_canvas=_authored_canvas_size(clip, document),
        frame_duration=frame_duration,
        clip_duration=clip.duration,
        source_start=clip.source_start,
        playback_rate=clip.speed,
        retime_map=clip.retime_map,
        source_colorspace=source_colorspace,
        source_color_range=source_color_range,
        reference_effect_link=reference_effect_link(
            source_color_space=source_colorspace,
            source_color_range=source_color_range,
            source_width=probe.width,
            source_height=probe.height,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            conform=clip.conform_type,
            crop_mode=crop_mode,
        ),
    )
    # A direct child of an expanded root scope is staged by the reference
    # (conform onto the container, then the transform on that container surface).
    nearest = ownership.nearest
    staged = nearest is not None and not tree.ancestors(nearest) and _has_spatial_adjustment(clip)
    layer = LayerSpec(
        clip_id=clip.id,
        path=clip.path,
        media_path=media_path,
        source_kind=source_kind,
        source_frame_duration=clip.source_frame_duration,
        source_rotation_degrees=probe.rotation_degrees,
        decoder_applied_orientation=False,
        alpha_handling=clip.alpha_handling,
        source_has_alpha=bool(source_color and source_color.has_alpha),
        reverse_decode_cache=reverse_decode_cache,
        lane=clip.lane,
        document_order=clip.document_order,
        first_frame=window.first_frame,
        end_frame=(
            min(window.end_frame, document.frame_count)
            if ownership.owner is None
            else window.end_frame
        ),
        render_start=render_start,
        local_first_frame=window.local_first_frame,
        local_frame_count=window.local_frame_count,
        frame_duration=frame_duration,
        clock=clock,
        frame=frame,
        geometry=geometry,
        conform=clip.conform_type,  # type: ignore[arg-type]
        crop_mode=crop_mode,
        canvas_to_owner=tuple(float(v) for v in ownership.canvas_to_owner.reshape(-1)),
        opacity=opacity,
        effects=effects,
        local_clock=local_clock,
        owner_id=ownership.owner.id if ownership.owner is not None else None,
        z_prefix=ownership.z_prefix,
        nearest_scope_id=nearest.id if nearest is not None else None,
        staged=staged,
        blend_mode=_tensor_blend_mode(clip.blend_mode, path=clip.path),
    )
    return replace(
        layer,
        decode_raster=resolve_decode_raster(layer, probe=probe, policy=decode_policy),
    )


def _scope_effect_origin(scope: RenderClip, tree: _ScopeTree, window: _ItemWindow, layers: tuple[LayerSpec, ...], ownership: _Ownership, *, frame_duration: Fraction) -> int:
    """First frame (local grid) of the scope's effect chain -- where its ``N`` is 0.

    ``ffmpeg._group_video_chain`` trims the composed pad at ``execution.render_start``
    and restarts PTS there (``_plan_group_execution``): for a root scope with
    direct leaves that is the union of the direct leaves' render windows (the
    earliest leaf's first local frame); for a retimed root scope it is the
    completed output window (``_completed_group_output_window``: the scope's
    transition-expanded start, where the retimed pad stream begins); for every
    other scope (nested, or root without a direct leaf) it is
    ``scope.absolute_start``, and ``trim`` keeps the first frame whose start is
    >= it, hence the ceiling.
    """

    if not tree.ancestors(scope):
        if _is_retimed(scope):
            return window.local_first_frame
        direct = [layer.local_first_frame for layer in layers if layer.nearest_scope_id == scope.id]
        if direct:
            return min(direct)
    return math.ceil(scope.absolute_start / frame_duration)


def _with_transition_markers(scope: RenderClip, leaves: list[RenderClip]) -> RenderClip:
    """A copy of the scope carrying the transitions ON it (or above it) as ``transition_in/out``.

    The compiler marks transitions on leaves only (``_apply_group_transition``
    iterates ``self.clips``; group scopes never receive ``transition_in/out``),
    so a scope's transition-expanded window (the reference's
    ``_completed_group_output_window`` intent: the finished surface is a side
    participant over the whole handle) is derived from its descendant leaves:
    the earliest incoming / latest outgoing transition marked on a leaf that is
    not authored inside the scope (a transition inside the scope expands its own
    leaves on the scope's clock and needs no scope expansion).
    """

    incoming = None
    outgoing = None
    for leaf in leaves:
        if scope.id not in leaf.ancestor_clip_ids:
            continue
        item = leaf.transition_in
        if item is not None and scope.id not in item.ancestor_group_ids and (incoming is None or item.absolute_start < incoming.absolute_start):
            incoming = item
        item = leaf.transition_out
        if item is not None and scope.id not in item.ancestor_group_ids and (outgoing is None or item.end > outgoing.end):
            outgoing = item
    if incoming is None and outgoing is None:
        return scope
    return replace(scope, transition_in=incoming, transition_out=outgoing)


def _lower_scope(
    scope: RenderClip,
    *,
    document: RenderDocument,
    tree: _ScopeTree,
    layers: tuple[LayerSpec, ...],
    leaves: list[RenderClip],
    canvas: _PlanCanvas,
) -> Optional[ScopeSpec]:
    """Lower one RENDERED group scope to a ``ScopeSpec`` (see the section comment).

    Pythonese:
    1. Resolve ancestors and the scope placement window on its parent's clock.
    2. Its geometry kernel maps the container onto the immediate parent's
       container: crop / conform / transform / animation from the scope clip.
    3. Its opacity kernel and its effects (``N`` from the group chain's first
       frame) come from the scope clip too; effects run on the container
       (root scope) or after crop / conform (nested), like the reference.
    4. Publish a composable ``ScopeTimeMap`` for native cadence and retime.

    Main callers: ``build_tensor_plan`` (after every leaf is lowered).
    """

    scope = _with_transition_markers(scope, leaves)
    ownership = _resolve_ownership(scope, document, tree, canvas)
    frame_duration = scope.canvas_frame_duration or document.frame_duration
    child_frame_duration = scope.container_frame_duration or document.frame_duration
    local_clock = None
    window = _item_window(
        scope,
        ownership,
        local_clock,
        document=document,
        frame_duration=frame_duration,
    )
    if window is None:
        return None
    clip_start, clip_duration = scope.absolute_start, scope.duration
    render_start, render_end = window.render_start, window.render_end
    width, height = _container_size(scope, document, canvas)
    parent_width, parent_height = tree.parent_canvas(scope, document, canvas)
    frame = FrameGeometry(source_width=width, source_height=height, project_width=parent_width, project_height=parent_height)
    geometry = _geometry_plan(scope, frame=frame, clip_start=clip_start, clip_duration=clip_duration, render_start=render_start, render_end=render_end)
    try:
        opacity = _opacity_plan(scope, clip_start=clip_start, clip_duration=clip_duration, render_start=render_start, render_end=render_end)
    except CompositorError as error:
        raise TensorRenderUnsupported(f"{scope.path}: invalid opacity automation: {error}") from error
    crop_mode: CropMode = None
    if scope.crop is not None and scope.crop.enabled:
        crop_mode = scope.crop.mode.strip().lower()  # type: ignore[assignment]
    is_root = not tree.ancestors(scope)
    effects_on_container = is_root
    effect_canvas = (width, height) if effects_on_container else (parent_width, parent_height)
    authored_effect_canvas = (
        _authored_container_size(scope, document)
        if effects_on_container
        else tree.authored_parent_canvas(scope, document)
    )
    effects = _effect_specs(
        scope.effects,
        clip_path=scope.path,
        frame_origin=_scope_effect_origin(
            scope, tree, window, layers, ownership,
            frame_duration=child_frame_duration,
        ),
        canvas=effect_canvas,
        authored_canvas=authored_effect_canvas,
        frame_duration=child_frame_duration,
        clip_duration=scope.duration,
        source_start=scope.source_start,
        playback_rate=scope.speed,
        retime_map=scope.retime_map,
        # The composed surface reaches the reference's effect chain after a bare
        # ``format=rgba`` (untagged): the ``rgba`` link with the 601 default table.
        source_colorspace="unknown",
        source_color_range="unknown",
        reference_effect_link=reference_effect_link(
            source_color_space="unknown",
            source_color_range="unknown",
            source_width=width,
            source_height=height,
            canvas_width=effect_canvas[0],
            canvas_height=effect_canvas[1],
            conform=scope.conform_type,
            crop_mode=crop_mode,
            folded_group_effect=True,
        ),
    )
    has_direct_leaves = any(layer.nearest_scope_id == scope.id for layer in layers)
    assert scope.retime_map is not None
    time_map = ScopeTimeMap(
        parent_start=scope.absolute_start,
        parent_duration=scope.duration,
        native_output_start=render_start,
        timeline_start=scope.retime_map.timeline_start,
        source_origin=(
            scope.source_window_origin
            if scope.source_window_origin is not None
            else scope.source_start
        ),
        parent_frame_duration=frame_duration,
        child_frame_duration=child_frame_duration,
        # FFmpeg trims the completed child at this scope boundary, resets its
        # PTS to zero, performs ``fps=...:round=up``, and only then places the
        # result on the parent timeline. These three origins preserve that
        # local sampling phase instead of accidentally anchoring it to project
        # time zero.
        parent_frame_origin=window.first_frame,
        native_frame_origin=math.ceil(render_start / child_frame_duration),
        child_frame_origin=math.ceil(scope.absolute_start / child_frame_duration),
        retime_map=scope.retime_map,
    )
    return ScopeSpec(
        scope_id=scope.id,
        path=scope.path,
        lane=scope.lane,
        document_order=scope.document_order,
        first_frame=window.first_frame,
        end_frame=(
            min(window.end_frame, document.frame_count)
            if ownership.owner is None
            else window.end_frame
        ),
        render_start=render_start,
        local_first_frame=window.local_first_frame,
        local_frame_count=window.local_frame_count,
        frame_duration=frame_duration,
        frame=frame,
        geometry=geometry,
        conform=scope.conform_type,  # type: ignore[arg-type]
        crop_mode=crop_mode,
        canvas_to_owner=tuple(float(v) for v in ownership.canvas_to_owner.reshape(-1)),
        opacity=opacity,
        effects=effects,
        local_clock=local_clock,
        owner_id=ownership.owner.id if ownership.owner is not None else None,
        z_prefix=ownership.z_prefix,
        width=width,
        height=height,
        time_map=time_map,
        effects_on_container=effects_on_container,
        expand_children=is_root and has_direct_leaves,
        blend_mode=_tensor_blend_mode(scope.blend_mode, path=scope.path),
    )


# --------------------------------------------------------------------------- transitions
#
# Transition sides (X7)
# ---------------------
# The reference (``ffmpeg._build_stock_transition_groups``) renders a calibrated
# transition as ONE composite item:
#
#   participants  every video layer whose ``clip.transition_out`` (outgoing side)
#                 / ``clip.transition_in`` (incoming side) is the transition
#                 (``_transition_participants``) -- the compiler marks the spine
#                 item AND every descendant / connected clip overlapping the
#                 transition interval (``_apply_group_transition``), so a
#                 connected clip on a higher lane beside the outgoing clip is
#                 an outgoing participant, and both leaves of a compound are
#                 participants of a transition on the compound;
#   frontier      the typed side (``_transition_side_frontiers``): a marked leaf
#                 below a RENDERED scope inside the transition's owner collapses
#                 to that scope's finished surface (crop / transform / opacity /
#                 effects / retime included -- ledger row 5 records that the
#                 emitted graph composes the raw leaf branches instead); leaves
#                 under inert scopes stay leaves (pixel-identical);
#   sides         each side is a zero-based, full-canvas composition of its
#                 participants sorted by (lane, document_order), each placed
#                 with its own geometry / opacity and clipped to its own render
#                 window (``_compose_transition_side``): the tensor
#                 ``renderer._FrameComposer.side`` source-overs the active side
#                 items onto a transparent canvas of the owner scope's size;
#   domain        Cross Dissolve runs on the calibrated linear sides; every
#                 other module is fed 8-bit encoded straight sides
#                 (``_adapt_transition_side_to_encoded``) -- that round trip is
#                 the transition port's business (``tensor/transitions.py``);
#   stack         participants are removed from the ordinary overlay pass for
#                 the transition's owned window (``_enable_without_intervals``)
#                 and the transition output is inserted ONCE at
#                 (min lane, min document_order) over all participants
#                 (``_transition_composite_item``) in the OWNER scope's stack
#                 (the nearest rendered ancestor of the transition, or root);
#                 inside a retimed scope the window is on the pad grid;
#   overlaps      transitions with overlapping windows are independent items;
#                 a layer marked by two overlapping transitions is composed
#                 into both sides (the reference does the same);
#   hard cuts     a transition without a portable handler (``handler is None``,
#                 "unknown transition becomes a hard cut") marks no
#                 participant and emits nothing: it is skipped here too;
#                 handled transitions lower regardless of their portable
#                 status label (the reference renders them), and the port
#                 rejects loudly what it cannot do.


def _lower_transitions(
    document: RenderDocument,
    layers: tuple[LayerSpec, ...],
    scopes: tuple[ScopeSpec, ...],
    tree: _ScopeTree,
    root_size: tuple[int, int],
) -> tuple[TransitionSpec, ...]:
    """Lower every handled calibrated transition to its sides + port payload (see the section comment)."""

    clips_by_id = {clip.id: clip for clip in document.clips}
    z_key: dict[str, tuple[tuple[int, int], ...]] = {layer.clip_id: layer.z_key for layer in layers}
    z_key.update({scope.scope_id: scope.z_key for scope in scopes})
    scope_ids = {scope.scope_id for scope in scopes}
    disabled_scopes = {scope.id for scope in document.group_scopes if not scope.enabled}
    transitions: list[TransitionSpec] = []
    for item in document.transitions:
        if item.handler is None:
            # The reference marks no participants and emits no module for it: an
            # explicit hard cut (``compiler._compile_transition`` omission_reason).
            continue
        if any(gid in disabled_scopes for gid in item.ancestor_group_ids):
            # Its leaves were dropped with the disabled group; the reference skips a
            # transition with no branches (``ffmpeg.py`` ``if not any(path == ...)``).
            continue
        # No label gate: ``ffmpeg._build_stock_transition_groups`` renders every handled
        # transition regardless of ``portable_status`` (Circle / Squares are labelled
        # ``unsupported`` yet render through xfade); only ``handler is None`` is a hard cut.
        # What the ports cannot lower rejects loudly inside ``lower_transition``.
        owner_id = next((gid for gid in reversed(item.ancestor_group_ids) if gid in scope_ids), None)
        owner_size = root_size
        frame_duration = document.frame_duration
        if owner_id is not None:
            owner_scope = next(scope for scope in scopes if scope.scope_id == owner_id)
            owner_size = (owner_scope.width, owner_scope.height)
            frame_duration = owner_scope.time_map.child_frame_duration
        window = resolve_owned_frame_window(item.absolute_start, item.end, frame_duration=frame_duration)
        # Lowering goes through the transition ports (tensor/transitions.py): the same
        # registry resolution the CPU builder runs, admitted ids only, loud otherwise.
        lowered = transition_ports.lower_transition(
            item,
            transition_ports.LowerContext(
                width=owner_size[0],
                height=owner_size[1],
                frame_duration=frame_duration,
                frame_count=window.frame_count,
            ),
        )

        def frontier(layer: LayerSpec) -> str:
            # The first rendered scope strictly below the owner on the leaf's ancestry.
            for gid in clips_by_id[layer.clip_id].ancestor_clip_ids:
                if gid in scope_ids and gid != owner_id and (owner_id is None or owner_id in tree.scopes[gid].ancestor_clip_ids):
                    return gid
            return layer.clip_id

        def side(marker: str) -> tuple[str, ...]:
            ids: list[str] = []
            for layer in layers:
                if getattr(clips_by_id[layer.clip_id], marker) is item:
                    participant = frontier(layer)
                    if participant not in ids:
                        ids.append(participant)
            return tuple(sorted(ids, key=z_key.__getitem__))

        outgoing, incoming = side("transition_out"), side("transition_in")
        if not outgoing or not incoming:
            raise reject(
                "transition without both participants",
                f"{item.path}: outgoing={outgoing!r}, incoming={incoming!r}",
            )
        # The composite sits at (min lane, min document_order) over the participants
        # (``_transition_composite_item``) at the transition's own depth: the inert
        # scopes between the owner and the transition are its z-prefix, and every
        # participant's key continues that prefix with the pair the minimum is over.
        z_prefix = tuple(
            (tree.scopes[gid].lane, tree.scopes[gid].document_order)
            for gid in item.ancestor_group_ids[_after(item.ancestor_group_ids, owner_id):]
            if gid in tree.scopes and gid not in scope_ids
        )
        depth = len(z_prefix)
        keys = [z_key[cid] for cid in (*outgoing, *incoming)]
        if any(key[:depth] != z_prefix or len(key) <= depth for key in keys):
            raise TensorRenderUnsupported(f"{item.path}: participant z-keys {keys!r} do not continue the transition prefix {z_prefix!r}")
        transitions.append(
            TransitionSpec(
                path=item.path,
                kind=lowered.kind,
                xfade_id=lowered.xfade_id,
                first_frame=window.first_frame,
                end_frame=window.end_frame,
                outgoing_clip_ids=outgoing,
                incoming_clip_ids=incoming,
                lane=min(key[depth][0] for key in keys),
                document_order=min(key[depth][1] for key in keys),
                frame_duration=frame_duration,
                payload=lowered.payload,
                needs_history=lowered.needs_history,
                scope_id=owner_id,
                z_prefix=z_prefix,
            )
        )
    return tuple(transitions)


def _after(ancestor_ids: tuple[str, ...], owner_id: Optional[str]) -> int:
    """Index just past ``owner_id`` in an outer->inner ancestor list (0 when the owner is the root)."""

    return 0 if owner_id is None else ancestor_ids.index(owner_id) + 1


# --------------------------------------------------------------------------- entry point


def build_tensor_plan(
    document: RenderDocument,
    *,
    rasters: Optional[Mapping[str, Path]] = None,
    output_resolution: Optional[OutputResolution] = None,
    decode_policy: DecodePolicy = DecodePolicy.NATIVE,
) -> TensorRenderPlan:
    """Return the frame-grid plan for ``document`` or raise TensorRenderUnsupported.

    ``rasters`` maps a clip id to the executor's resolved runtime raster (the
    ``text_images`` dict ``execute_render`` builds from
    ``resolve_text_clip_raster`` / ``resolve_generator_clip_raster``): titles,
    captions and solid generators arrive as project-space straight-alpha PNGs.
    A clip that composites a runtime raster and has no entry rejects loudly --
    see ``_resolve_source`` and the raster-sources section above.

    ``decode_policy`` selects the decode raster each video leaf asks its
    decoder for (``decode_policy.py``): ``NATIVE`` (the default, and the export
    contract) decodes every source at its own raster; ``VISIBLE`` is the
    interactive seek / scan contract that lets ordinary leaves decode near
    their visible output contribution.  It never changes geometry or timing.
    """

    if output_resolution is not None and (
        output_resolution.source_width != document.width
        or output_resolution.source_height != document.height
    ):
        raise ValueError(
            "output resolution source raster does not match the render document: "
            f"{output_resolution.source_width}x{output_resolution.source_height} != "
            f"{document.width}x{document.height}"
        )
    canvas = _PlanCanvas(document=document, output_resolution=output_resolution)
    rasters = dict(rasters or {})
    tree = _classify_scopes(document, canvas)
    disabled_scopes = {scope.id for scope in document.group_scopes if not scope.enabled}

    def composited(clip: RenderClip) -> bool:
        if not clip.enabled or not clip.has_video:
            return False
        # A disabled container suppresses its complete internal composition
        # (``ffmpeg._compose_group_scopes``).
        if any(gid in disabled_scopes for gid in clip.ancestor_clip_ids):
            return False
        # The compiler already decided an ``omit_transparent`` interval renders
        # transparent (an unadapted generator or a title whose font did not
        # resolve). Missing media instead owns a visible runtime raster. This
        # is a *compiled* semantic with a compatibility
        # finding attached, not a silent skip, and the reference drops the same
        # clip from its input list (``ffmpeg.py:1665``).
        return not (clip.video_disposition is not None and clip.video_disposition.execution == "omit_transparent")

    leaves = [clip for clip in document.clips if composited(clip)]
    probes: dict[Path, VideoProbe] = {}
    layers = tuple(
        layer
        for layer in (
            _lower_leaf(
                clip,
                document=document,
                tree=tree,
                rasters=rasters,
                probes=probes,
                canvas=canvas,
                decode_policy=decode_policy,
            )
            for clip in leaves
        )
        if layer is not None
    )
    scopes = tuple(
        spec
        for spec in (
            _lower_scope(
                scope,
                document=document,
                tree=tree,
                layers=layers,
                leaves=leaves,
                canvas=canvas,
            )
            for scope in document.group_scopes
            if scope.id in tree.rendered and scope.enabled and not any(gid in disabled_scopes for gid in scope.ancestor_clip_ids)
        )
        if spec is not None
    )

    consumed_rasters = {layer.clip_id for layer in layers if layer.source_kind == "raster"}
    stale_rasters = sorted(set(rasters) - consumed_rasters)
    if stale_rasters:
        # An entry for a clip that never became a raster layer means the caller
        # rasterized against a different document than the one being planned.
        raise reject(
            "runtime raster for a non-raster clip",
            f"rasters carry entries for clips that own no raster layer: {stale_rasters}",
        )

    return TensorRenderPlan(
        width=canvas.root[0],
        height=canvas.root[1],
        frame_duration=document.frame_duration,
        frame_count=document.frame_count,
        layers=layers,
        transitions=_lower_transitions(document, layers, scopes, tree, canvas.root),
        scopes=scopes,
    )
