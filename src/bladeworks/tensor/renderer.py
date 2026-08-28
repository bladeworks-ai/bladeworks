"""The per-frame tensor loop (serial reference and pipelined production paths).

Architecture map
----------------
    render_document(document, pipelined=True|False)
        sources : pipeline.DirectSources   (serial: decode inline)      | SourcePool
                  pipeline.PrefetchSources (pipelined: decode threads)  |
        sink    : encode.VideoEncoder      (serial: encode inline)      | .write_planes
                  encode.EncoderThread     (pipelined: bounded queue)   |
        for frame n:
            sources.before_frame(n)                      open (soon-)active layers
            canvas = _FrameComposer.compose(n)           GPU: place layers / transition / over
            planes = encoder.canvas_to_planes(canvas)     GPU: code space -> yuv420p (or yuva444p10, alpha policy)
            _FrameExit.submit(planes)                    the ONE device->host copy per frame
                                                         (MPS: event-fenced, one frame behind)
                -> sink.write_planes(host)               encode inline or hand to the thread
            sources.after_frame(n)                       close ended layers

The two paths share every pixel-producing line (``_FrameComposer``,
``canvas_to_yuv420p``); they differ only in *where* decode and encode run.
That is what makes them frame-identical (gated by
``test_tensor_pipeline.py``): same decode order per layer, same tensors, same
exit function, same encoder settings.

What one frame does (Pythonese)
-------------------------------
1. Take the layers active at ``n``, bottom to top.
2. For each, ask its source pool for the exact owning source frame
   (``SourceClock`` decides the source time), linearize it, apply the source
   alpha window (trim / crop / pan / conform none), and warp it onto the
   project with the layer's homography for this frame
   (``GeometryPlan.snapshot(t)`` -> ``sampler.layer_homography``); when the
   layer carries ported effects, conform to the clip canvas first, run the
   effects, then apply the composed corner-pin/affine warp (the reference's
   stage order).  Premultiply after the warp, multiply by the exact opacity
   (``OpacityPlan.snapshot``).
3. Build the frame's stack: every active layer that is not a participant of
   an active transition, every active child SCOPE (rendered recursively on
   its own container canvas, then placed like a leaf -- ``placed_scope``),
   plus one item per active transition (its two sides composed from their
   participant items -- ``side`` -- and handed to the transition module) at
   the transition's z-key; sort by (lane, document order) and source-over in
   that order (the reference's fold of ``composite_items`` per scope).
4. Exit: linear -> code space -> 8-bit RGB codes -> Rec.709 limited yuv420p
   on the GPU (``encode.canvas_to_yuv420p``), one download, encode.

Stage stats (``RenderStats``): ``decode_wait_seconds`` is GPU-thread time
blocked on sources (serial: the decode itself; pipelined: queue waits),
``gpu_seconds`` is kernel-issue time for compose + exit (MPS is asynchronous,
so execution mostly lands in ``download_seconds``, the ``.cpu()`` sync),
``encode_wait_seconds`` is GPU-thread time handing frames to the sink (serial:
the encode itself; pipelined: waits on a full encoder queue), and
``encoder_busy_seconds`` is time inside PyAV encode + mux wherever it ran.
When the pipeline works, wall ~= max(stage) rather than the sum.

Main callers:
- ``scripts/final_cut/fcpxml_tensor_render_bench.py``.
- ``experimental_tests/core/test_tensor_*.py``.
- the executor's ``--backend tensor`` path (U2).
"""

from __future__ import annotations

import fcntl
import math
import tempfile
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import torch

from ..core.model import RenderDocument
from .blend import composite_layers
from .color import linearize
from .composite import opaque_black, over, unpremultiply
from .decode_policy import DecodePolicy, decoded_to_native_matrix, scale_alpha_window
from .effects import CanvasPlacement, apply_effects
from .encode import (
    EncoderAudio,
    EncoderThread,
    PixelPolicy,
    VideoEncoder,
    canvas_to_yuv420p,
    yuv420p_ndarray,
)
from .errors import TensorRenderError
from .pipeline import DirectSources, PrefetchSources, SourcePool
from .plan import LayerSpec, ScopeSpec, TensorRenderPlan, TransitionSpec, build_tensor_plan
from .resolution import OutputResolution
from .sampler import (
    GridCache,
    apply_display_rotation,
    apply_alpha_window,
    apply_matrix,
    composed_matrix,
    conform_matrix,
    is_identity,
    layer_homography,
    premultiply,
    resize_exact_aspect_opaque,
    source_alpha_window,
    uses_exact_aspect_minification,
    warp,
)
from .transitions import ApplyContext as TransitionContext, apply_transition


@dataclass
class RenderStats:
    frames: int = 0
    wall_seconds: float = 0.0
    first_frame_seconds: Optional[float] = None
    device: str = ""
    decoded_source_frames: int = 0
    max_open_decoders: int = 0
    encoder: str = ""
    notes: list[str] = field(default_factory=list)
    # Pipeline configuration and per-stage accounting (see module docstring).
    pipelined: bool = False
    prefetch_depth: int = 0
    encoder_queue_depth: int = 0
    decode_wait_seconds: float = 0.0
    upload_seconds: float = 0.0
    gpu_seconds: float = 0.0
    download_seconds: float = 0.0
    encode_wait_seconds: float = 0.0
    encoder_busy_seconds: float = 0.0
    max_encoder_queue_depth: int = 0
    mean_encoder_queue_depth: float = 0.0
    mean_prefetch_queue_depth: float = 0.0
    prefetch_starved_frames: int = 0

    @property
    def ms_per_frame(self) -> float:
        return 1000.0 * self.wall_seconds / self.frames if self.frames else 0.0


@dataclass(frozen=True)
class FrameWindow:
    """An output-frame interval, with an exclusive ``end_frame``.

    Why this exists: export, seek, and scan all execute the same compositor;
    only the visible interval and output sink differ. Keeping range validation
    in this small value object prevents each caller from inventing subtly
    different clipping or off-by-one rules.

    Main callers:
    - ``render_document`` for bounded renders.
    - future seek and scan session adapters.
    """

    start_frame: int
    end_frame: int

    @classmethod
    def full(cls, frame_count: int) -> "FrameWindow":
        return cls(0, frame_count)

    def validate(self, frame_count: int) -> None:
        if self.start_frame < 0:
            raise TensorRenderError("frame window start must be >= 0")
        if self.end_frame > frame_count:
            raise TensorRenderError(
                f"frame window end {self.end_frame} exceeds project frame count {frame_count}"
            )
        if self.end_frame <= self.start_frame:
            raise TensorRenderError(
                "frame window must contain at least one frame "
                f"(got [{self.start_frame}, {self.end_frame}))"
            )

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class ComposedFrame:
    """One opaque Bladeworks frame ready for a live YUV media adapter.

    ``yuv420p`` has shape ``[height * 3 / 2, width]`` in the layout accepted
    by ``av.VideoFrame.from_ndarray``. It is produced directly from the shared
    compositor and never passes through an MP4 container.
    """

    frame: int
    time: Fraction
    duration: Fraction
    width: int
    height: int
    yuv420p: np.ndarray


class TensorRenderSession:
    """Keep a compiled plan, decoders, and compositor warm for seek and scan.

    Architecture map:

        RenderDocument + target resolution
            -> one TensorRenderPlan
            -> lazy DirectSources + one _FrameComposer
            -> frames([start, end))
                -> hidden temporal preroll
                -> composed YUV420 frames with exact project timestamps

    The class is intentionally synchronous and single-caller. Preview service
    orchestration owns locking, cancellation, pacing, and last-request-wins
    publication. This keeps transport and renderer lifetimes separate.

    ``decode_policy`` (``decode_policy.py``) is how the preview asks ordinary
    leaves to decode near their visible output contribution
    (``DecodePolicy.VISIBLE``); the default keeps every source native so tests
    and oracles that compare against export stay byte-identical.  It only
    applies when this session builds its own plan.

    Main callers:
    - the localhost preview ``FrameProducer`` adapter.
    - renderer tests and local performance benchmarks.
    """

    def __init__(
        self,
        document: RenderDocument,
        *,
        output_resolution: Optional[OutputResolution] = None,
        device: Optional[str] = None,
        decoder_threads: int = 2,
        plan: Optional[TensorRenderPlan] = None,
        decode_policy: DecodePolicy = DecodePolicy.NATIVE,
    ) -> None:
        if plan is not None and output_resolution is not None:
            raise ValueError("pass either plan or output_resolution, not both")
        if plan is not None and decode_policy != DecodePolicy.NATIVE:
            raise ValueError("decode_policy is applied when the session builds its plan; pass it into build_tensor_plan instead")
        self.document = document
        self.plan = plan or build_tensor_plan(
            document,
            output_resolution=output_resolution,
            decode_policy=decode_policy,
        )
        self.device = _select_device(device)
        self.sources = DirectSources(
            self.plan,
            device=self.device,
            decoder_threads=decoder_threads,
        )
        self.composer = _FrameComposer(
            self.plan,
            device=self.device,
            sources=self.sources,
        )
        self._closed = False

    def frames(
        self,
        window: FrameWindow,
        *,
        is_cancelled: Optional[CancellationCheck] = None,
    ) -> Iterator[ComposedFrame]:
        """Compose a bounded sequential frame stream without creating a file."""

        if self._closed:
            raise TensorRenderError("tensor render session is closed")
        window.validate(self.plan.frame_count)
        internal_start = _temporal_preroll_start(self.plan, window.start_frame)
        for frame in range(internal_start, window.end_frame):
            if is_cancelled is not None and is_cancelled():
                raise TensorRenderError(f"render cancelled before project frame {frame}")
            self.sources.before_frame(frame)
            canvas = self.composer.compose(frame)
            self.sources.after_frame(frame)
            if frame < window.start_frame:
                continue
            planes = canvas_to_yuv420p(canvas).cpu()
            if is_cancelled is not None and is_cancelled():
                raise TensorRenderError(f"render cancelled before publishing project frame {frame}")
            yield ComposedFrame(
                frame=frame,
                time=frame * self.plan.frame_duration,
                duration=self.plan.frame_duration,
                width=self.plan.width,
                height=self.plan.height,
                yuv420p=yuv420p_ndarray(
                    planes,
                    height=self.plan.height,
                    width=self.plan.width,
                ),
            )

    def seek(
        self,
        frame: int,
        *,
        is_cancelled: Optional[CancellationCheck] = None,
    ) -> ComposedFrame:
        """Compose exactly one visible project frame, including hidden history."""

        if self.plan.frame_count == 0:
            if frame != 0:
                raise TensorRenderError("empty timeline only has playhead 0")
            canvas = self.composer.compose(0)
            planes = canvas_to_yuv420p(canvas).cpu()
            return ComposedFrame(
                frame=0,
                time=Fraction(0),
                duration=self.plan.frame_duration,
                width=self.plan.width,
                height=self.plan.height,
                yuv420p=yuv420p_ndarray(
                    planes,
                    height=self.plan.height,
                    width=self.plan.width,
                ),
            )
        return next(
            self.frames(
                FrameWindow(frame, frame + 1),
                is_cancelled=is_cancelled,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.sources.close()

    def __enter__(self) -> "TensorRenderSession":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def require_torch_device(name: str) -> None:
    """Fail loudly when an explicit device cannot be used.

    Why this exists:
        Fleet/gym on Apple Silicon always request ``mps``. The auto path may
        still fall through to CPU, but an explicit ``mps`` (or ``cuda``) pin
        must never silently land on another device.

    Main callers:
        ``_select_device``, the ``fcpxml render --device`` CLI, and
        ``fcpxml server run --device``.
    """

    requested = str(name).strip()
    if requested in {"", "auto"}:
        raise TensorRenderError("require_torch_device needs an explicit device name, not auto/empty")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise TensorRenderError(
            "requested torch device 'mps' but torch.backends.mps is not available "
            "(Apple Silicon GPU / Metal is required for Fleet Bladeworks renders)"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise TensorRenderError(
            "requested torch device 'cuda' but torch.cuda is not available"
        )
    torch.device(requested)


def _select_device(name: Optional[str]) -> torch.device:
    if name:
        requested = str(name).strip()
        if requested != "auto":
            require_torch_device(requested)
            return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_MPS_SLOT_LOCK = Path(tempfile.gettempdir()) / "spellshot-bladeworks-mps.lock"


@contextmanager
def exclusive_mps_slot(device: torch.device):
    """Serialize MPS tensor renders across processes on one Mac.

    ginartbox is one 16 GB M4. Concurrent episode servers may compile and
    serve source/media in parallel, but two tensor loops on the same GPU
    OOM. CPU/CUDA callers skip the lock.

    Main callers: ``render_document``.
    """

    if device.type != "mps":
        yield
        return
    _MPS_SLOT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = open(_MPS_SLOT_LOCK, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class _FrameComposer:
    """Builds one output canvas per frame from a plan and a source pool (all GPU work).

    Why this exists: the serial and pipelined loops must produce identical
    pixels; keeping every compositing line in one object that both drive
    (with only the ``SourcePool`` differing) makes that true by construction.

    Scopes (X6, Pythonese):
    ``compose(frame)`` renders the ROOT stack onto opaque black by calling
    ``render_scope(None, ...)``.  A stack is the owner's active leaves, active
    child scopes and active transitions (each transition consumes its
    participants), sorted by (lane, document_order) and source-over'd in that
    order.  A child scope item is produced by ``placed_scope``: a transparent
    canvas of the scope's container size (or the per-frame expanded surface of
    a root scope with direct leaves), filled by ``render_scope`` recursively,
    then group effects, unpremultiply, the scope's own homography (crop /
    conform / transform, ``layer_homography`` with the surface as the source),
    premultiply, group opacity -- the same tail as a leaf.  Inside a retimed
    scope evaluates its own ``ScopeTimeMap`` boundary before recursing.
    """

    def __init__(self, plan: TensorRenderPlan, *, device: torch.device, sources: SourcePool, transparent_root: bool = False) -> None:
        self.plan = plan
        self.device = device
        self.sources = sources
        self.transparent_root = transparent_root
        self.layers_by_id = {layer.clip_id: layer for layer in plan.layers}
        self.scopes_by_id = {scope.scope_id: scope for scope in plan.scopes}
        # Stack membership per owner scope (None = root), in plan order.
        self.stack_layers: dict[Optional[str], list[LayerSpec]] = {}
        self.stack_scopes: dict[Optional[str], list[ScopeSpec]] = {}
        self.stack_transitions: dict[Optional[str], list[TransitionSpec]] = {}
        for layer in plan.layers:
            self.stack_layers.setdefault(layer.owner_id, []).append(layer)
        for scope in plan.scopes:
            self.stack_scopes.setdefault(scope.owner_id, []).append(scope)
        for transition in plan.transitions:
            self.stack_transitions.setdefault(transition.scope_id, []).append(transition)
        # Direct leaves of each scope (its expanded surface follows their quads).
        self.direct_leaves: dict[str, list[LayerSpec]] = {}
        for layer in plan.layers:
            if layer.nearest_scope_id is not None:
                self.direct_leaves.setdefault(layer.nearest_scope_id, []).append(layer)
        self.grids = GridCache()
        self._probe = torch.zeros(1, device=device)
        # Completed scope surfaces are reused when several parent frames floor
        # to the same native frame, and when a transition asks for the same side
        # more than once. The small LRU stays lazy and bounds device memory.
        self._scope_cache: "OrderedDict[tuple[object, ...], torch.Tensor]" = OrderedDict()
        self._scope_cache_limit = max(4, 2 * len(plan.scopes))
        # Temporal transitions use the current and two preceding raw composed
        # side frames. Preprocessing then runs at each frame's own local time.
        self._transition_side_cache: "OrderedDict[tuple[object, ...], torch.Tensor]" = OrderedDict()
        # This LRU only needs to span the small live working set, not the whole
        # project.  A temporal transition reads 3 outgoing + 3 incoming raw sides
        # per output frame, rising to 4 per side (8 total) while two consecutive
        # output frames overlap and share history.  16 holds that window even with
        # a couple of temporal transitions in flight through nested scopes.  The
        # old ``6 * len(plan.transitions)`` scaled with the project-wide transition
        # count -- hundreds of slots for a long timeline -- even though only the
        # last frame or two of any single transition is ever live.  The bound only
        # affects cross-frame reuse (within one frame ``side_history`` holds its own
        # references), so shrinking it changes memory, never any pixel.
        self._transition_side_cache_limit = 16

    # ------------------------------------------------------------------ clocks / surfaces

    def child_local_frame(self, owner: Optional[ScopeSpec], frame: int) -> int:
        """The current owner's already-mapped native child frame."""

        return frame

    def scope_surface(self, scope: ScopeSpec, frame: int) -> tuple[int, int, int, int]:
        """``(origin_x, origin_y, width, height)`` of the scope's canvas at ``frame``.

        The container, unless the scope is a root scope with direct leaves
        (``expand_children``): then the integer union of the container and the
        direct leaves' placed quads plus the reference's 2 px transparent guard
        (``ffmpeg._plan_group_execution`` / ``render_surface_for_quads``: pixels
        a transformed child pushes outside the container are kept for the group
        transform, and group effects run on that guarded surface -- an edge
        clamp / blur there pulls in transparent guard, not the border pixel).
        """

        left, top, right, bottom = 0.0, 0.0, float(scope.width), float(scope.height)
        if not scope.expand_children:
            return 0, 0, scope.width, scope.height
        fd = scope.time_map.child_frame_duration
        for layer in self.direct_leaves.get(scope.scope_id, ()):
            if not layer.active(frame):
                continue
            matrix = layer.canvas_matrix
            for x, y in layer.geometry_at(frame, fd).composed_quad:
                hx, hy, hw = matrix @ np.array([x, y, 1.0])
                px, py = hx / hw, hy / hw
                left, top, right, bottom = min(left, px), min(top, py), max(right, px), max(bottom, py)
        guard = 2
        origin_x, origin_y = math.floor(left) - guard, math.floor(top) - guard
        return origin_x, origin_y, math.ceil(right) + guard - origin_x, math.ceil(bottom) + guard - origin_y

    # ------------------------------------------------------------------ items

    def placed(self, layer: LayerSpec, frame: int, *, out_size: tuple[int, int], offset: tuple[int, int] = (0, 0)) -> torch.Tensor:
        """Fetch, window, warp, premultiply and fade one layer for output frame ``frame``.

        ``out_size`` is the owner canvas ``(height, width)`` the layer is placed
        onto and ``offset`` that canvas's origin in owner coordinates (a root
        scope's expanded surface).
        """

        dev = self.device
        out_height, out_width = out_size
        source = self.sources.frame(layer, frame)
        if layer.source_rotation_degrees and not layer.decoder_applied_orientation:
            source = apply_display_rotation(source, layer.source_rotation_degrees)
        channels, source_height, source_width = source.shape
        # Decoder-side downscale (preview only, ``decode_policy.py``): the layer's
        # geometry stays authored against the NATIVE raster (``layer.frame``);
        # the decoded raster is smaller and is sampled through the fixed
        # ``decoded -> native`` scale appended to every source-side matrix below.
        raster = layer.decode_raster
        downscaled = raster is not None and not raster.is_native
        expected_raster = raster.display_size if downscaled else (layer.frame.source_width, layer.frame.source_height)
        if (source_width, source_height) != expected_raster:
            raise TensorRenderError(
                f"{layer.path}: decoded raster {source_width}x{source_height} differs from the "
                f"{'requested decode raster' if downscaled else 'probed'} {expected_raster[0]}x{expected_raster[1]}"
            )
        decoded_to_native = decoded_to_native_matrix(raster) if downscaled else np.eye(3, dtype=np.float64)
        snapshot = layer.geometry_at(frame, layer.frame_duration)
        # Straight source alpha (rasters) or opaque, then the crop/conform window
        # (native pixel indices; re-expressed on the decoded grid when downscaled).
        base_alpha = source[3:4] if channels == 4 else source.new_ones((1, source_height, source_width))
        window = source_alpha_window(snapshot, frame=layer.frame, conform=layer.conform, crop_mode=layer.crop_mode)
        if downscaled:
            window = scale_alpha_window(window, raster)
        alpha = apply_alpha_window(base_alpha, window)
        straight = torch.cat((linearize(source[:3]), alpha), dim=0)
        canvas_matrix = _translation(-offset[0], -offset[1]) @ layer.canvas_matrix
        # The calibrated whole-raster minification is a NATIVE-raster path: a
        # downscaled leaf already arrives near its output footprint, so it takes
        # the general homography (a near-identity bilinear tap).
        exact_aspect_minification = not downscaled and uses_exact_aspect_minification(
            snapshot,
            frame=layer.frame,
            conform=layer.conform,
            crop_mode=layer.crop_mode,
            source_is_opaque=layer.source_kind == "video" and channels == 3,
        )
        if exact_aspect_minification:
            # This calibrated path is deliberately staged: encoded whole-raster
            # minification first, then linear-light effects and authored spatial
            # geometry. Fusing it into the homography would collapse the broad
            # minification footprint back to one aliased 2x2 sample.
            resized_rgb = resize_exact_aspect_opaque(
                source[:3],
                height=layer.frame.project_height,
                width=layer.frame.project_width,
            )
            opaque = resized_rgb.new_ones(
                (1, layer.frame.project_height, layer.frame.project_width)
            )
            canvas = premultiply(torch.cat((linearize(resized_rgb), opaque), dim=0))
            if layer.effects:
                canvas = apply_effects(
                    canvas,
                    layer.effects,
                    frame=layer.local_frame(frame, layer.frame_duration),
                    frame_duration=layer.frame_duration,
                )
            composed = canvas_matrix @ composed_matrix(snapshot, layer.frame)
            if not is_identity(composed) or (
                layer.frame.project_width,
                layer.frame.project_height,
            ) != (out_width, out_height):
                grid = self.grids.grid_for(
                    f"{layer.clip_id}:composed",
                    composed,
                    out_height=out_height,
                    out_width=out_width,
                    source_height=layer.frame.project_height,
                    source_width=layer.frame.project_width,
                    device=dev,
                )
                canvas = warp(canvas, grid)
        elif layer.effects or layer.staged:
            # Reference stage order: crop/conform -> effects -> corner pin -> transform
            # (and the staged conform of a root scope's direct leaf: the container clips
            # first, the transform moves the clipped canvas).
            conform = conform_matrix(snapshot, layer.frame, layer.conform) @ decoded_to_native
            canvas_width, canvas_height = layer.frame.project_width, layer.frame.project_height
            # The composed (post-effects) transform, clip canvas -> output. Built
            # BEFORE the conform warp because the overscan surface is bounded by
            # the canvas region this transform can actually sample.
            canvas_composed = canvas_matrix @ composed_matrix(snapshot, layer.frame)
            # Effects run on a surface that keeps the conform's overscan, so a
            # later pan / zoom transform samples the real image instead of the
            # black left by a premature crop-to-canvas (see ``_overscan_surface``).
            # ``preserve`` is gated to the EFFECTS trigger: a pure ``staged``
            # container leaf keeps its intentional clip (``preserve=False`` ->
            # identity surface), and any clip whose conformed content already fits
            # the canvas is byte-identical to the previous path.
            surface_conform, (surface_width, surface_height), surface_origin = _overscan_surface(
                conform,
                source_width=source_width,
                source_height=source_height,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                preserve=bool(layer.effects) and not layer.staged,
                sample_bound=_sampled_canvas_bound(
                    canvas_composed,
                    out_width=out_width,
                    out_height=out_height,
                    canvas_width=canvas_width,
                    canvas_height=canvas_height,
                ),
            )
            grid = self.grids.grid_for(
                f"{layer.clip_id}:conform", surface_conform,
                out_height=surface_height, out_width=surface_width,
                source_height=source_height, source_width=source_width, device=dev,
            )
            canvas = premultiply(warp(straight, grid))
            if layer.effects:
                # ``N`` counts on the layer's local frame grid (the pad grid inside a
                # retimed group; the output grid otherwise), like the reference chain.
                # On an enlarged surface the placement tells the effects where the
                # clip canvas sits so every canvas-relative kernel keeps its
                # clip-canvas coordinate system (``effects`` module doc). When the
                # surface IS the canvas the call is the plain, pre-overscan one.
                placement = None
                if surface_origin != (0, 0) or (surface_width, surface_height) != (canvas_width, canvas_height):
                    placement = CanvasPlacement(
                        width=canvas_width,
                        height=canvas_height,
                        origin_x=surface_origin[0],
                        origin_y=surface_origin[1],
                    )
                canvas = apply_effects(
                    canvas,
                    layer.effects,
                    frame=layer.local_frame(frame, layer.frame_duration),
                    frame_duration=layer.frame_duration,
                    **({} if placement is None else {"placement": placement}),
                )
            # ``translate(surface_origin)`` re-expresses the (possibly enlarged,
            # origin-shifted) surface in clip-canvas coords before the composed
            # transform. It is identity when the surface is the clip canvas, so
            # this collapses to ``canvas_matrix @ composed_matrix`` unchanged.
            composed = canvas_composed @ _translation(surface_origin[0], surface_origin[1])
            if not is_identity(composed) or (surface_width, surface_height) != (out_width, out_height):
                grid = self.grids.grid_for(
                    f"{layer.clip_id}:composed", composed,
                    out_height=out_height, out_width=out_width,
                    source_height=surface_height, source_width=surface_width, device=dev,
                )
                canvas = warp(canvas, grid)
        else:
            homography = layer_homography(
                snapshot, frame=layer.frame, conform=layer.conform, canvas_to_project=canvas_matrix
            ) @ decoded_to_native
            grid = self.grids.grid_for(
                layer.clip_id, homography,
                out_height=out_height, out_width=out_width,
                source_height=source_height, source_width=source_width, device=dev,
            )
            canvas = premultiply(warp(straight, grid))
        opacity = layer.opacity_at(frame, layer.frame_duration)
        if opacity != 1.0:
            canvas = canvas * opacity  # premultiplied: opacity scales every channel
        return canvas

    def placed_scope(self, scope: ScopeSpec, frame: int, *, out_size: tuple[int, int], offset: tuple[int, int] = (0, 0)) -> torch.Tensor:
        """Render, native-sample, and place one completed child scope.

        ``frame`` is on the parent clock. The boundary first selects a native
        output frame, then maps that through this scope's retime into the child
        story. Effects and geometry run on the native output frame; opacity is
        applied after cadence normalization on the parent frame.
        """

        dev = self.device
        out_height, out_width = out_size
        native_frame = scope.time_map.native_output_frame(frame)
        child_frame = scope.time_map.child_frame(native_frame)
        cache_key = (scope.scope_id, native_frame, out_height, out_width, offset)
        cached = self._scope_cache.get(cache_key)
        if cached is not None:
            self._scope_cache.move_to_end(cache_key)
            opacity = scope.opacity_at(frame, scope.frame_duration)
            return cached if opacity == 1.0 else cached * opacity

        origin_x, origin_y, width, height = self.scope_surface(scope, child_frame)
        canvas = self._probe.new_zeros((4, height, width))
        canvas = self.render_scope(scope, child_frame, canvas, offset=(origin_x, origin_y))
        local_frame = native_frame
        native_fd = scope.time_map.child_frame_duration
        if scope.effects and scope.effects_on_container:
            # Root scope (``_group_video_chain``): effects on the composed surface,
            # before conform / transform.
            canvas = apply_effects(canvas, scope.effects, frame=local_frame, frame_duration=native_fd)
        snapshot = scope.geometry_at(native_frame, native_fd)
        straight = unpremultiply(canvas)
        window = source_alpha_window(snapshot, frame=scope.frame, conform=scope.conform, crop_mode=scope.crop_mode)
        if window is not None:
            x0, x1, y0, y1 = window
            shifted = (max(0, x0 - origin_x), max(0, x1 - origin_x), max(0, y0 - origin_y), max(0, y1 - origin_y))
            straight = torch.cat((straight[:3], apply_alpha_window(straight[3:4], shifted)), dim=0)
        surface_to_container = _translation(origin_x, origin_y)
        canvas_matrix = _translation(-offset[0], -offset[1]) @ scope.canvas_matrix
        if scope.effects and not scope.effects_on_container:
            # Nested scope (``_legacy_group_video_chain``): crop / conform -> effects -> transform.
            conform = conform_matrix(snapshot, scope.frame, scope.conform) @ surface_to_container
            grid = self.grids.grid_for(
                f"{scope.scope_id}:conform", conform,
                out_height=scope.frame.project_height, out_width=scope.frame.project_width,
                source_height=height, source_width=width, device=dev,
            )
            placed = premultiply(warp(straight, grid))
            placed = apply_effects(placed, scope.effects, frame=local_frame, frame_duration=native_fd)
            composed = canvas_matrix @ composed_matrix(snapshot, scope.frame)
            if not is_identity(composed) or (scope.frame.project_width, scope.frame.project_height) != (out_width, out_height):
                grid = self.grids.grid_for(
                    f"{scope.scope_id}:composed", composed,
                    out_height=out_height, out_width=out_width,
                    source_height=scope.frame.project_height, source_width=scope.frame.project_width, device=dev,
                )
                placed = warp(placed, grid)
        else:
            homography = layer_homography(
                snapshot, frame=scope.frame, conform=scope.conform, canvas_to_project=canvas_matrix
            ) @ surface_to_container
            grid = self.grids.grid_for(
                scope.scope_id, homography,
                out_height=out_height, out_width=out_width,
                source_height=height, source_width=width, device=dev,
            )
            placed = premultiply(warp(straight, grid))
        self._scope_cache[cache_key] = placed
        self._scope_cache.move_to_end(cache_key)
        while len(self._scope_cache) > self._scope_cache_limit:
            self._scope_cache.popitem(last=False)
        opacity = scope.opacity_at(frame, scope.frame_duration)
        if opacity != 1.0:
            placed = placed * opacity
        return placed

    def item_result(self, item_id: str, frame: int, *, out_size: tuple[int, int], offset: tuple[int, int]) -> Optional[torch.Tensor]:
        """A leaf or a rendered scope placed onto its owner canvas, or None when inactive."""

        layer = self.layers_by_id.get(item_id)
        if layer is not None:
            return self.placed(layer, frame, out_size=out_size, offset=offset) if layer.active(frame) else None
        scope = self.scopes_by_id[item_id]
        return self.placed_scope(scope, frame, out_size=out_size, offset=offset) if scope.active(frame) else None

    def item_blend_mode(self, item_id: str) -> str:
        """Return the canonical parent-fold blend mode for a leaf or scope."""

        layer = self.layers_by_id.get(item_id)
        if layer is not None:
            return layer.blend_mode
        return self.scopes_by_id[item_id].blend_mode

    def side(self, item_ids: tuple[str, ...], frame: int, *, out_size: tuple[int, int], offset: tuple[int, int] = (0, 0)) -> torch.Tensor:
        """One transition side: its participant items (bottom to top) over a transparent canvas.

        Mirrors ``ffmpeg._compose_transition_side``: a zero-based full-canvas
        composition of every marked item, each with its own placement, opacity,
        and blend mode, clipped to its own render window (an inactive
        participant contributes nothing).
        """

        canvas = self._probe.new_zeros((4, out_size[0], out_size[1]))
        for item_id in item_ids:
            result = self.item_result(item_id, frame, out_size=out_size, offset=offset)
            if result is not None:
                canvas = composite_layers(canvas, result, self.item_blend_mode(item_id))
        return canvas

    def transition_result(self, transition: TransitionSpec, frame: int, local_frame: int, *, out_size: tuple[int, int], offset: tuple[int, int]) -> torch.Tensor:
        """The transition module's output for output frame ``frame`` (premultiplied linear)."""

        def side_history(label: str, item_ids: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
            """Return FFmpeg tmix's oldest-to-newest three-frame input window.

            FFmpeg initializes tmix by cloning its first input frame, so local
            indices before zero clamp to the transition's first owned frame.
            Root rendering is chronological and finds prior frames in this
            cache. Recursive clocks use DirectSources, so a missing historical
            surface can be reconstructed by its local frame index.

            Main callers: temporal transition kernels through the context below.
            """

            history: list[torch.Tensor] = []
            for delta in (2, 1, 0):
                available_delta = min(delta, max(0, local_frame - transition.first_frame))
                history_frame = frame - available_delta
                history_local_frame = local_frame - available_delta
                key = (
                    transition.path,
                    label,
                    history_frame,
                    history_local_frame,
                    out_size,
                    offset,
                )
                side = self._transition_side_cache.get(key)
                if side is None:
                    side = self.side(item_ids, history_frame, out_size=out_size, offset=offset)
                    self._transition_side_cache[key] = side
                    self._transition_side_cache.move_to_end(key)
                    while len(self._transition_side_cache) > self._transition_side_cache_limit:
                        self._transition_side_cache.popitem(last=False)
                else:
                    self._transition_side_cache.move_to_end(key)
                history.append(side)
            return tuple(history)

        temporal = transition.needs_history
        if temporal:
            a_history = side_history("outgoing", transition.outgoing_clip_ids)
            b_history = side_history("incoming", transition.incoming_clip_ids)
            outgoing, incoming = a_history[-1], b_history[-1]
        else:
            a_history = b_history = ()
            outgoing = self.side(
                transition.outgoing_clip_ids,
                frame,
                out_size=out_size,
                offset=offset,
            )
            incoming = self.side(
                transition.incoming_clip_ids,
                frame,
                out_size=out_size,
                offset=offset,
            )

        return apply_transition(
            transition.kind,
            transition.payload,
            outgoing,
            incoming,
            TransitionContext(
                frame_index=local_frame - transition.first_frame,
                frame_count=transition.frame_count,
                width=out_size[1],
                height=out_size[0],
                frame_duration=transition.frame_duration,
                a_history=a_history,
                b_history=b_history,
            ),
        )

    # ------------------------------------------------------------------ stacks

    def render_scope(self, owner: Optional[ScopeSpec], frame: int, canvas: torch.Tensor, *, offset: tuple[int, int] = (0, 0)) -> torch.Tensor:
        """Source-over the owner's stack (leaves + child scopes + transition items, sorted by z-key) onto ``canvas``.

        The reference removes every participant from the ordinary overlay pass
        for the transition's owned window (``_enable_without_intervals``) and
        inserts the transition output once at (min lane, min document_order)
        over all participants (``_transition_composite_item``); a layer marked
        by two overlapping transitions is composed into both.
        """

        owner_id = None if owner is None else owner.scope_id
        out_size = (int(canvas.shape[1]), int(canvas.shape[2]))
        local_frame = self.child_local_frame(owner, frame)
        transitions = [item for item in self.stack_transitions.get(owner_id, ()) if item.active(local_frame)]
        consumed: set[str] = set()
        for transition in transitions:
            consumed |= transition.participant_ids
        # (hierarchical z_key, insertion index) -> item; the index keeps two
        # transition items sharing a participant in plan order (a stable sort).
        # ``z_key`` = the inert ancestors' (lane, document_order) pairs then the
        # item's own, so items folded out of an inert group sort at the group's
        # position (the reference composes a group as one raster there).
        stack: list[tuple[tuple[tuple[tuple[int, int], ...], int], object]] = [
            ((layer.z_key, 0), layer)
            for layer in self.stack_layers.get(owner_id, ())
            if layer.active(frame) and layer.clip_id not in consumed
        ]
        stack.extend(
            ((scope.z_key, 0), scope)
            for scope in self.stack_scopes.get(owner_id, ())
            if scope.active(frame) and scope.scope_id not in consumed
        )
        stack.extend(
            ((transition.z_key, index + 1), transition)
            for index, transition in enumerate(transitions)
        )
        stack.sort(key=lambda item: item[0])
        for _key, item in stack:
            if isinstance(item, TransitionSpec):
                canvas = over(canvas, self.transition_result(item, frame, local_frame, out_size=out_size, offset=offset))
            elif isinstance(item, ScopeSpec):
                canvas = composite_layers(
                    canvas,
                    self.placed_scope(item, frame, out_size=out_size, offset=offset),
                    item.blend_mode,
                )
            else:
                canvas = composite_layers(
                    canvas,
                    self.placed(item, frame, out_size=out_size, offset=offset),
                    item.blend_mode,
                )
        return canvas

    def compose(self, frame: int) -> torch.Tensor:
        """Compose over transparency, then apply the delivery background once.

        A blend mode must distinguish "no lower layer" from an authored black
        layer. Starting the fold on opaque black made a bottom Multiply or
        Darken clip turn black, despite the blend contract's rule that a
        transparent lower pixel reveals the foreground unchanged. The alpha
        delivery path keeps the completed transparent surface; ordinary
        delivery flattens it onto black after the layer fold.

        Main callers: ``TensorRenderSession.frames`` and ``seek``.
        """
        plan = self.plan
        canvas = self._probe.new_zeros((4, plan.height, plan.width))
        composed = self.render_scope(None, frame, canvas)
        if self.transparent_root:
            return composed
        return over(opaque_black(plan.height, plan.width, like=self._probe), composed)

    def forget_layer(self, clip_id: str) -> None:
        for key in (clip_id, f"{clip_id}:conform", f"{clip_id}:composed"):
            self.grids.forget(key)

    def forget_ended_scopes(self, frame: int) -> None:
        for scope in self.plan.scopes:
            if scope.end_frame <= frame + 1:
                self.forget_layer(scope.scope_id)


def _translation(x: float, y: float) -> np.ndarray:
    return np.array([[1.0, 0.0, float(x)], [0.0, 1.0, float(y)], [0.0, 0.0, 1.0]], dtype=np.float64)


Rect = tuple[float, float, float, float]  # (left, top, right, bottom) edge coordinates

# Bilinear sampling reads the 2x2 neighbourhood of each output pixel centre; the mapped
# output RECT already covers the centres, so one extra source pixel (rounded up to two)
# on every side is enough for any scale.
_SAMPLE_MARGIN_PX = 2.0


def _sampled_canvas_bound(
    composed: np.ndarray,
    *,
    out_width: int,
    out_height: int,
    canvas_width: int,
    canvas_height: int,
    margin: float = _SAMPLE_MARGIN_PX,
) -> Optional[Rect]:
    """The clip-canvas region the composed warp can sample, or ``None`` for "unbounded".

    ``composed`` maps clip-canvas edge coordinates to output edge coordinates
    (``canvas_matrix @ composed_matrix``). Inverse-mapping the output rectangle's
    four corners gives the canvas-space quad the output reads; its bounding box
    plus ``margin`` bounds every bilinear tap. Overscan outside it can never reach
    the output, so ``_overscan_surface`` need not allocate it.

    Two special cases:
    * identity onto a same-sized output: the composed warp is SKIPPED by the
      caller and the surface is read 1:1, so the bound is the canvas itself with
      NO margin -- this is what collapses the identity case to exactly the clip
      canvas (the byte-identical pre-overscan path);
    * a singular matrix or a corner mapped behind the projective camera
      (``w <= 0``, a degenerate corner pin): the quad is not a finite rectangle,
      so the bound is dropped (``None``) and the caller keeps the full union --
      the previous behaviour, more memory but never a missing pixel.

    Main callers: ``_FrameComposer._compose_leaf`` (staged effects branch).
    """

    if is_identity(composed) and (out_width, out_height) == (canvas_width, canvas_height):
        return (0.0, 0.0, float(canvas_width), float(canvas_height))
    try:
        inverse = np.linalg.inv(composed)
    except np.linalg.LinAlgError:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for x, y in ((0.0, 0.0), (float(out_width), 0.0), (0.0, float(out_height)), (float(out_width), float(out_height))):
        hx, hy, hw = inverse @ np.array([x, y, 1.0])
        if hw <= 1e-12:
            return None
        xs.append(float(hx / hw))
        ys.append(float(hy / hw))
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


def _overscan_surface(
    conform: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    canvas_width: int,
    canvas_height: int,
    preserve: bool,
    sample_bound: Optional[Rect] = None,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Size the staged-effect intermediate surface to keep (only the useful) conform overscan.

    Returns ``(surface_conform, (surface_width, surface_height), (origin_x, origin_y))``
    where ``origin`` is the clip-canvas coordinate of the surface's top-left pixel
    (integers, ``<= 0``; the surface always contains the whole clip canvas).

    Why this exists
    ---------------
    The staged leaf path (``_compose_leaf`` effects branch) warps the source
    through the spatial ``conform`` onto an intermediate raster, runs the clip's
    effects on it, then warps that raster through the composed transform. When it
    conformed onto exactly the CLIP CANVAS, any source content the conform placed
    OUTSIDE the canvas (an aspect-mismatched ``fill``, a ``crop`` / Ken-Burns
    camera, an oversized ``none``) was discarded before the transform ran -- so a
    transform that pans or zooms that overscan back into frame exposed black,
    while Final Cut (and the effect-free fused ``layer_homography`` path) show the
    image. See ``REFERENCE_DISCREPANCIES.md`` rows 3 and 26.

    What it does
    ------------
    With ``preserve`` set, the surface is the integer bounding box of the clip
    canvas rectangle UNION the part of the conformed source quad's bounding box
    that lies inside ``sample_bound`` (the canvas region the composed transform
    can read, ``_sampled_canvas_bound``; ``None`` = keep all of it), and
    ``surface_conform`` is the conform pre-translated so it targets that
    surface's origin. Effects then run at the same pixel density as before -- the
    surface only GROWS to hold overscan the transform will actually sample.

    Why the bound: without it the surface is the whole conformed quad, and a
    tight ``crop`` / Ken-Burns camera (source scaled up N times) allocates an
    N-squared intermediate for a fixed-size output -- gigabytes for a 10x punch-in.

    Non-regression invariant
    -------------------------
    When the conformed content already fits inside the clip canvas (``fit``,
    ``none``-smaller, matched-aspect ``fill``), or the bound admits none of the
    overscan (an identity transform, whose bound is the canvas itself), the union
    equals the canvas, so this returns the canvas size, a zero origin, and the
    conform unchanged -- the warp, the effect input, and the composed step are
    then byte-identical to the previous canvas-only path. ``preserve=False`` also
    returns that identity, so the intentional container clip of a ``staged`` leaf
    is never widened.

    Main callers: ``_FrameComposer._compose_leaf``; unit-tested directly.
    """

    if not preserve:
        return conform, (canvas_width, canvas_height), (0, 0)
    corners = (
        (0.0, 0.0),
        (float(source_width), 0.0),
        (0.0, float(source_height)),
        (float(source_width), float(source_height)),
    )
    content = apply_matrix(conform, corners)
    left, top = min(point[0] for point in content), min(point[1] for point in content)
    right, bottom = max(point[0] for point in content), max(point[1] for point in content)
    if sample_bound is not None:
        left, top = max(left, sample_bound[0]), max(top, sample_bound[1])
        right, bottom = min(right, sample_bound[2]), min(bottom, sample_bound[3])
    xs = [0.0, float(canvas_width)] + ([left, right] if right > left else [])
    ys = [0.0, float(canvas_height)] + ([top, bottom] if bottom > top else [])
    surface_left, surface_top = math.floor(min(xs)), math.floor(min(ys))
    surface_right, surface_bottom = math.ceil(max(xs)), math.ceil(max(ys))
    if (surface_left, surface_top, surface_right, surface_bottom) == (0, 0, canvas_width, canvas_height):
        return conform, (canvas_width, canvas_height), (0, 0)
    surface_conform = _translation(-surface_left, -surface_top) @ conform
    return (
        surface_conform,
        (surface_right - surface_left, surface_bottom - surface_top),
        (surface_left, surface_top),
    )


FrameProgress = Callable[[int, int], None]
CancellationCheck = Callable[[], bool]


def _has_temporal_transitions(plan: TensorRenderPlan) -> bool:
    """Use the unified transition capability for bounded-render preroll."""

    return any(transition.needs_history for transition in plan.transitions)


def _temporal_preroll_start(plan: TensorRenderPlan, visible_start: int) -> int:
    """Return the earliest frame needed to reproduce ``visible_start`` exactly.

    Temporal transition kernels consume the current side plus its two prior
    side surfaces. A full render naturally leaves those surfaces in the
    composer's cache. A bounded render must quietly compose the same short
    history before it emits its first visible frame. Composing a frame before
    the transition starts is harmless; once the transition activates, its
    kernel clamps history to its first owned frame just like FFmpeg.

    Main callers: ``render_document`` before constructing its source pool.
    """

    # Nested-scope transitions run on a mapped child clock, so a root-frame
    # activity check is not sufficient. Two root frames is the conservative
    # bounded history and is still constant-time for seek/scan startup.
    return max(0, visible_start - 2) if _has_temporal_transitions(plan) else visible_start


class _FrameExit:
    """Device->host download + hand-off to the encoder sink, one frame behind on MPS.

    Pythonese: ``submit(planes)`` queues the download of this frame's yuv
    planes (``non_blocking``) and records a device event behind it, then
    completes the *previous* frame: wait for its event (which returns as soon
    as that frame's work is done, while this frame's kernels keep running),
    give its host bytes to the sink, and drop the host buffers its uploads
    were reading from.  ``finish()`` completes the last frame.  On non-MPS
    devices the download is a plain blocking ``.cpu()`` (CPU device: no copy).

    Why this exists: MPS has one in-order stream and a blocking ``.cpu()``
    drains everything queued, so the GPU idles while the CPU issues the next
    frame's kernels; fencing with ``torch.mps.Event`` (measured: the event
    wait returns before later work completes and the host bytes are valid;
    serial-vs-pipelined output stays byte-identical on torch 2.11 and 2.13)
    overlaps kernel issue of frame n with execution of frame n-1.

    Known cost: on torch 2.11 ``Event.synchronize()`` holds the GIL for the
    whole wait (2.13 releases it), which starves the decode / encoder threads
    of Python time on older torch; upgrade rather than work around it.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        sink,
        sources: SourcePool,
        stats: RenderStats,
        height: int,
        width: int,
        frame_count: int,
        progress: Optional[FrameProgress],
    ) -> None:
        self.device = device
        self.sink = sink
        self.sources = sources
        self.stats = stats
        self.height = height
        self.width = width
        self.frame_count = frame_count
        self.progress = progress
        self.deferred = device.type == "mps"
        self._pending: Optional[tuple[torch.Tensor, object, list[torch.Tensor]]] = None

    def submit(self, planes: torch.Tensor) -> None:
        held = self.sources.take_held()
        if not self.deferred:
            started = time.perf_counter()
            host = planes.cpu()          # blocking: the one device sync per frame
            del held                     # uploads have landed
            self.stats.download_seconds += time.perf_counter() - started
            self._write(host)
            return
        host = planes.to("cpu", non_blocking=True)
        event = torch.mps.Event()
        event.record()
        previous, self._pending = self._pending, (host, event, held)
        if previous is not None:
            self._complete(previous)

    def _complete(self, pending: tuple[torch.Tensor, object, list[torch.Tensor]]) -> None:
        host, event, held = pending
        started = time.perf_counter()
        event.synchronize()              # waits for this frame's work only
        held.clear()
        self.stats.download_seconds += time.perf_counter() - started
        self._write(host)

    def _write(self, host: torch.Tensor) -> None:
        started = time.perf_counter()
        self.sink.write_planes(host)
        self.stats.encode_wait_seconds += time.perf_counter() - started
        self.stats.frames += 1
        if self.progress is not None:
            self.progress(self.stats.frames, self.frame_count)

    def finish(self) -> None:
        pending, self._pending = self._pending, None
        if pending is not None:
            self._complete(pending)


def render_document(
    document: RenderDocument,
    *,
    output_path: Path,
    device: Optional[str] = None,
    codec: str = "libx264",
    preset: str = "medium",
    crf: int = 18,
    decoder_threads: int = 2,
    encoder_threads: int = 0,
    plan: Optional[TensorRenderPlan] = None,
    output_resolution: Optional[OutputResolution] = None,
    pipelined: bool = True,
    prefetch: int = 8,
    encoder_queue: int = 6,
    worker_uploads: bool = False,
    pixel_policy: PixelPolicy = "opaque",
    progress: Optional[FrameProgress] = None,
    audio: Optional["EncoderAudio"] = None,
    window: Optional[FrameWindow] = None,
    is_cancelled: Optional[CancellationCheck] = None,
) -> RenderStats:
    """Render ``document`` (video only) to ``output_path``; return timing + stage stats.

    ``pipelined=True`` (default) runs decode on per-layer worker threads
    (``prefetch`` frames ahead) and encode on an encoder thread behind a queue
    of ``encoder_queue`` frames; ``pipelined=False`` is the serial reference
    loop. Both produce identical output (see the module docstring).
    ``window`` selects an output-frame interval using an exclusive end. When
    omitted, the complete document is rendered exactly as before. Temporal
    preroll required by a transition is composed but not encoded or counted.
    ``progress(completed_frames, total_frames)`` runs only after a frame has
    crossed the device boundary and been handed to the encoder sink. The
    callback is optional so programmatic renders remain quiet by default.
    ``is_cancelled`` is checked between frames and raises loudly after all
    renderer resources have been closed.
    """

    if plan is not None and output_resolution is not None:
        raise ValueError("pass either plan or output_resolution, not both")
    plan = plan or build_tensor_plan(document, output_resolution=output_resolution)
    window = window or FrameWindow.full(plan.frame_count)
    window.validate(plan.frame_count)
    internal_start = _temporal_preroll_start(plan, window.start_frame)
    dev = _select_device(device)
    bounded_temporal_preroll = internal_start < window.start_frame
    source_pipelined = (
        pipelined
        and not plan.requires_random_access_sources
        and not bounded_temporal_preroll
    )
    notes = [f"pixel_policy={pixel_policy}"] if pixel_policy != "opaque" else []
    if pipelined and not source_pipelined:
        reason = (
            "bounded_temporal_preroll"
            if bounded_temporal_preroll
            else "recursive_scope_clocks"
        )
        notes.append(f"source_prefetch=disabled_for_{reason}")
    stats = RenderStats(
        device=str(dev),
        encoder=f"{codec}:{preset}:crf{crf}" if codec == "libx264" else codec,
        notes=notes,
        pipelined=pipelined,
        prefetch_depth=prefetch if source_pipelined else 0,
        encoder_queue_depth=encoder_queue if pipelined else 0,
    )
    mps_slot = exclusive_mps_slot(dev)
    mps_slot.__enter__()
    started = time.monotonic()
    encoder = None
    sink = None
    sources = None

    def close_quietly(resource) -> None:
        # Error path only: the loop's exception is the root cause; never mask it.
        try:
            resource.close()
        except Exception:  # noqa: BLE001
            pass

    try:
        encoder = VideoEncoder(
            output_path,
            width=plan.width,
            height=plan.height,
            frame_duration=plan.frame_duration,
            codec=codec,
            preset=preset,
            crf=crf,
            threads=encoder_threads,
            pixel_policy=pixel_policy,
            audio=audio,
        )
        sink = EncoderThread(encoder, queue_depth=encoder_queue) if pipelined else encoder
        sources = (
            PrefetchSources(
                plan,
                device=dev,
                decoder_threads=decoder_threads,
                prefetch=prefetch,
                worker_uploads=worker_uploads,
                start_frame=internal_start,
            )
            if source_pipelined
            else DirectSources(plan, device=dev, decoder_threads=decoder_threads)
        )
        composer = _FrameComposer(plan, device=dev, sources=sources, transparent_root=(pixel_policy == "alpha"))
        exit_stage = _FrameExit(
            device=dev,
            sink=sink,
            sources=sources,
            stats=stats,
            height=plan.height,
            width=plan.width,
            frame_count=window.frame_count,
            progress=progress,
        )
        layers_by_id = composer.layers_by_id
        open_layers: set[str] = set()

        try:
            for frame in range(internal_start, window.end_frame):
                if is_cancelled is not None and is_cancelled():
                    raise TensorRenderError(f"render cancelled before project frame {frame}")
                sources.before_frame(frame)
                active_ids = {layer.clip_id for layer in plan.layers_at(frame)}
                open_layers |= active_ids
                wait_before = sources.stats.wait_seconds
                gpu_started = time.perf_counter()
                canvas = composer.compose(frame)
                visible = frame >= window.start_frame
                planes = encoder.canvas_to_planes(canvas) if visible else None
                gpu_elapsed = time.perf_counter() - gpu_started
                decode_wait = sources.stats.wait_seconds - wait_before
                stats.decode_wait_seconds += decode_wait
                stats.gpu_seconds += gpu_elapsed - decode_wait
                if planes is not None:
                    exit_stage.submit(planes)   # download (event-fenced on MPS) + hand to the sink
                if stats.first_frame_seconds is None and stats.frames:
                    stats.first_frame_seconds = time.monotonic() - started
                # Close sources (and cached grids) whose layer ended with this frame.
                sources.after_frame(frame)
                for clip_id in [cid for cid in open_layers if layers_by_id[cid].end_frame <= frame + 1]:
                    open_layers.discard(clip_id)
                    composer.forget_layer(clip_id)
                composer.forget_ended_scopes(frame)
            exit_stage.finish()
            if stats.first_frame_seconds is None and stats.frames:
                stats.first_frame_seconds = time.monotonic() - started
        except BaseException:
            close_quietly(sources)
            close_quietly(sink)
            raise
        # Flush the encoder (re-raises an encoder-thread failure), then release the sources.
        close_started = time.perf_counter()
        sink.close()
        stats.encode_wait_seconds += time.perf_counter() - close_started
        sources.close()
        stats.wall_seconds = time.monotonic() - started
        stats.upload_seconds = sources.stats.upload_seconds
        stats.decoded_source_frames = sources.stats.decoded_source_frames
        stats.max_open_decoders = sources.stats.max_open_decoders
        stats.encoder_busy_seconds = encoder.busy_seconds
        if isinstance(sink, EncoderThread):
            stats.max_encoder_queue_depth = sink.stats.max_queue_depth
            stats.mean_encoder_queue_depth = sink.stats.queue_depth_sum / max(1, sink.stats.frames)
        served = max(1, sources.stats.frames_served)
        stats.mean_prefetch_queue_depth = sources.stats.queue_depth_sum / served
        stats.prefetch_starved_frames = sources.stats.starved_frames
        if stats.frames != window.frame_count:
            raise TensorRenderError(f"rendered {stats.frames} frames, expected {window.frame_count}")
        return stats
    except BaseException:
        if sources is not None:
            close_quietly(sources)
        if sink is not None:
            close_quietly(sink)
        raise
    finally:
        mps_slot.__exit__(None, None, None)
