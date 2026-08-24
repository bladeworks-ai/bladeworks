"""Tensor (PyTorch) renderer for compiled FCPXML documents.

Architecture map
================

    RenderDocument (compiler.py)
        -> plan.build_tensor_plan      : layers + scopes + transitions on the output frame
                                         grid; each layer holds the LIVE exact kernels
                                         (GeometryPlan / OpacityPlan / RetimeMap-backed
                                         SourceClock); group scopes (X6): INERT containers
                                         fold onto their leaves (placement affine, LocalClock
                                         = constant-speed pad clock), every other container is
                                         a ScopeSpec (own container canvas, placed like a leaf:
                                         crop / conform / transform / animation, group effects,
                                         group opacity, child pad clock); transition sides =
                                         participant leaves / rendered scopes at the typed
                                         frontier with a z-key (X7)
        -> renderer.render_document    : per-frame loop; _FrameComposer.compose renders the
                                         root stack, render_scope recurses into rendered
                                         scopes (leaves + child scopes + one item per active
                                         transition, side() composes each side's participants);
                                         root scopes with direct leaves compose on the
                                         reference's expanded guarded surface
             decode.ClipDecoder        : PyAV decode, exact source-frame ownership; probe_video;
                                         swscale-exact colour-in (709/601/2020 x limited/full,
                                         8/10-bit 4:2:0/4:2:2/4:4:4; HLG/PQ use
                                         frozen Rec.2020-to-Rec.709 SDR LUTs) (X10)
             decode.RasterSource       : stills / title / solid PNG rasters (straight alpha)
             pipeline.PrefetchSources  : per-layer decode threads feeding the GPU thread
             sampler.*                 : GeometryPlan.snapshot(t) -> one homography ->
                                         grid_sample (crop / conform / corner pin / transform /
                                         Ken Burns / animation, straight RGBA, transparent border)
             composite.over            : calibrated premultiplied linear source-over
             effects.*                 : effect PORT registry (leaf + folded group effects on
                                         the conformed canvas, after conform / before spatial)
               fx_basic / fx_warp / fx_color : E1 / E2 / E4 port batches (one owner per file)
             transitions.*             : transition PORT registry: Cross Dissolve (linear
                                         premultiplied lerp), registry xfade-custom expressions
                                         resolved through transitions.stock (admitted ids)
               tr_handlers / tr_equirect     : F4 (fade_color, wipe, slide_push) / F2 (360°)
             expr.*                    : FFmpeg expression parser + torch evaluator
                                         (xfade / geq environments, GBRA plane order)
             encode.VideoEncoder       : PyAV libx264 (or videotoolbox), Rec.709 limited;
                                         pixel_policy "alpha" = ProRes 4444 straight alpha
                                         (executor output_profile delivery_alpha, CLI --alpha)
        audio_pyav.*                   : the calibrated audio graph rebuilt node-by-node in
                                         libav (av.filter.Graph) + muxed into the encoder's
                                         own container -- no second ffmpeg process (U1)
        audio_delivery.*               : backend-neutral audio resolver (probe/omit/build plan)
        support.SUPPORT                : the one table of supported / rejected constructs

Invariants (frozen for the sprint; see PYTORCH_MVP_PLAN.md §0.1 / §3.2)
-----------------------------------------------------------------------
* Working space: calibrated linear light (exponent 1.94, ``compositor.py``),
  float32, channels-first ``[4, H, W]``; canvases are premultiplied.
* Alpha association is per operation: sources are linearized as *straight*
  colour, windowed and warped straight (like the reference ``perspective``
  over a transparent border), premultiplied *after* the warp; Cross Dissolve
  and layer source-over run premultiplied; code-space kernels (light/deco)
  unpremultiply -> encode -> formula -> linearize -> premultiply.
* Coordinates: edge coordinates, output pixel centres at ``(x+0.5, y+0.5)``,
  ``grid_sample(align_corners=False, padding_mode="zeros")``, fp32 grids.
* Time: exact ``Fraction`` on the CPU; frame ``n`` samples its start
  ``n * frame_duration``; canvas ownership by frame midpoint
  (``resolve_owned_frame_window``); source ownership = last decoded frame with
  pts <= instant.
* No silent fallback: every unsupported construct raises
  ``TensorRenderUnsupported`` at plan time naming the construct
  (``support.reject``); the parity reference is the CPU ``reference`` profile.

File ownership (one owner per file; shared types only through plan.py / support.py)
------------------------------------------------------------------------------------
    plan.py / support.py / errors.py     lowering + support table          (front end)
    sampler.py                            geometry -> grid_sample kernel     (X1/A5)
    decode.py                             PyAV decode + probe                (A3)
    composite.py / color.py               over + transfer                    (A6)
    effects.py / transitions.py           port registries + built-ins       (core; ports append)
    fx_basic.py / fx_warp.py / fx_color.py effect ports                       (E1 / E2 / E4)
    tr_handlers.py / tr_equirect.py       transition ports                   (F4 / F2)
    expr.py                               expression evaluator               (F0)
    encode.py                             encoder (video + interleaved audio) (A7)
    audio_pyav.py / audio_delivery.py     in-process audio graph + resolver  (U1)
    pipeline.py                           source pools (prefetch threads)    (A3/A9)
    renderer.py                           the loop (serial + pipelined)      (A9)

Why this exists
---------------
The legacy CPU path emits one project-long FFmpeg filtergraph and pays for it
in graph pathologies (project-long decoders, format negotiation, unbounded
link FIFOs).  This package executes the same compiled document as a plain
per-frame loop over tensors: only the clips active at frame ``n`` are decoded,
every operator is a few lines of readable array math, the calibrated
linear-light semantics are the product path (float32 on GPU costs nothing),
and frame count is exact by construction.  Semantics (geometry, animation,
retime, opacity, ownership) come from the shared exact kernels -- only pixels
are re-implemented here.

Audio stays on the calibrated FFmpeg *semantics* (``core/audio_execution.py``)
but is no longer a separate process: ``audio_pyav.py`` rebuilds that exact graph
node-by-node in libav (PyAV 16 has no ``filter_complex`` parser) and ``encode.py``
muxes its AAC output into the video container.  The decoded delivery audio is
bit-identical to the CPU render.
"""

from .decode_policy import DecodePolicy, DecodeRaster
from .errors import TensorRenderError, TensorRenderUnsupported
from .plan import TensorRenderPlan, build_tensor_plan
from .renderer import ComposedFrame, FrameWindow, RenderStats, TensorRenderSession, render_document
from .resolution import OutputResolution, RenderMode, ResolutionProfile, resolve_output_resolution
from .support import SUPPORT, rejected_constructs, supported_constructs

__all__ = [
    "ComposedFrame",
    "DecodePolicy",
    "DecodeRaster",
    "FrameWindow",
    "OutputResolution",
    "RenderStats",
    "RenderMode",
    "ResolutionProfile",
    "SUPPORT",
    "TensorRenderError",
    "TensorRenderPlan",
    "TensorRenderSession",
    "TensorRenderUnsupported",
    "build_tensor_plan",
    "rejected_constructs",
    "render_document",
    "resolve_output_resolution",
    "supported_constructs",
]
