"""Source pools for the render loop: direct (serial) and prefetching (pipelined).

Architecture map
----------------
    renderer.render_document
        -> SourcePool.before_frame(n)      : open sources that are (about to be) active
        -> SourcePool.frame(layer, n)      : the source raster owning frame n of ``layer``
        -> SourcePool.after_frame(n)       : close sources whose layer ended with n
        -> SourcePool.close()              : release everything (also on error)

    DirectSources    opens each layer's FrameSource on first use and calls
                     ``frame_at`` inline -- the serial reference behaviour.
    PrefetchSources  one worker thread per active layer runs ``frame_at`` for the
                     layer's whole frame schedule ahead of the GPU thread into a
                     bounded queue; the GPU thread pops (frame index checked) so it
                     never blocks on PyAV or the host->device upload.

Both pools produce the same tensors for the same (layer, frame): the schedule
a prefetch worker follows is exactly the sequence of ``frame_at`` calls the
serial loop would make (every frame in ``[first_frame, end_frame)`` in order,
each layer decoded once per frame), so decoding is deterministic and the two
loops are frame-identical.

Threading rules (see PYTORCH_MVP_PLAN.md §3.3 and the MPS gotchas)
------------------------------------------------------------------
* Workers never touch the device (default ``worker_uploads=False``): they run
  the ``FrameSource`` on the CPU device -- for video via
  ``ClipDecoder.packed_at``, which hands back the raw planes (``SourceFrame``)
  -- and the GPU thread performs the one upload per layer-frame
  (``non_blocking``, host buffer held until that frame's device sync) plus the
  yuv->RGB conversion (``decode.planes_to_rgb``, the same kernel the serial
  loop's ``frame_at`` runs, so both loops are frame-identical by construction).
  Measured on MPS: a blocking upload issued from a worker drains the single
  in-order stream while holding its dispatch queue, stalling the GPU thread's
  kernel issue (3 ms -> 21 ms per frame at 720p); ``worker_uploads=True``
  keeps that mode available for measurement / other backends.
* No silent hangs: a worker that raises pushes its exception to the consumer,
  which re-raises with the layer path; the consumer polls with a timeout and
  checks the worker is alive, so a dead worker cannot stall the render.
* Bounded memory: ``prefetch`` frames per layer at most, and only layers whose
  ``first_frame`` is within ``open_ahead`` frames of the current frame are open.

Main callers: ``renderer.render_document`` (chooses the pool by ``pipelined``).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Protocol

import torch

from .decode import ClipDecoder, FrameSource, HDRFrame, SourceFrame, open_source, planes_to_rgb
from .errors import TensorRenderError
from .hdr import hdr_to_sdr
from .plan import LayerSpec, TensorRenderPlan


@dataclass
class SourcePoolStats:
    decoded_source_frames: int = 0
    max_open_decoders: int = 0
    wait_seconds: float = 0.0            # GPU-thread time spent inside ``frame`` (decode or queue wait, incl. upload)
    upload_seconds: float = 0.0          # prefetch with worker_uploads=False: GPU-thread host->device copies
    starved_frames: int = 0              # prefetch only: pops that found the layer queue empty
    queue_depth_sum: int = 0             # prefetch only: backlog observed at pop time (for the mean)
    frames_served: int = 0


class SourcePool(Protocol):
    stats: SourcePoolStats

    def before_frame(self, frame: int) -> None: ...

    def frame(self, layer: LayerSpec, frame: int) -> torch.Tensor: ...

    def take_held(self) -> list[torch.Tensor]: ...

    def after_frame(self, frame: int) -> None: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- serial


class DirectSources:
    """Serial pool: decode inline on the calling thread (the reference behaviour)."""

    def __init__(self, plan: TensorRenderPlan, *, device: torch.device, decoder_threads: int) -> None:
        self._plan = plan
        self._device = device
        self._threads = decoder_threads
        self._layers = {layer.clip_id: layer for layer in plan.layers}
        self._open: dict[str, FrameSource] = {}
        # Recursive clocks can revisit old source times, so root-frame lifetime
        # is not enough to close a decoder. Keep a bounded insertion-ordered LRU
        # instead of retaining every source in a long library render.
        self._random_access_limit = 16
        self.stats = SourcePoolStats()

    def before_frame(self, frame: int) -> None:
        return None

    def frame(self, layer: LayerSpec, frame: int) -> torch.Tensor:
        started = time.perf_counter()
        source = self._open.get(layer.clip_id)
        if source is None:
            if self._plan.requires_random_access_sources:
                while len(self._open) >= self._random_access_limit:
                    self._release(next(iter(self._open)))
            source = open_source(layer, device=self._device, threads=self._threads)
            self._open[layer.clip_id] = source
            self.stats.max_open_decoders = max(self.stats.max_open_decoders, len(self._open))
        elif self._plan.requires_random_access_sources:
            # A normal dict preserves insertion order. Reinsert the active
            # decoder so the first entry remains the least recently used.
            self._open.pop(layer.clip_id)
            self._open[layer.clip_id] = source
        tensor = source.frame_at(layer.source_time(frame, layer.frame_duration))
        self.stats.wait_seconds += time.perf_counter() - started
        self.stats.frames_served += 1
        return tensor

    def take_held(self) -> list[torch.Tensor]:
        return []

    def after_frame(self, frame: int) -> None:
        if self._plan.requires_random_access_sources:
            # Nested native frames can be repeated or skipped relative to the
            # project frame. Keep these lazy decoders open until render end;
            # root-frame indices cannot safely decide their lifetime.
            return
        for clip_id in [cid for cid in self._open if self._layers[cid].end_frame <= frame + 1]:
            self._release(clip_id)

    def _release(self, clip_id: str) -> None:
        source = self._open.pop(clip_id)
        self.stats.decoded_source_frames += source.frames_decoded
        source.close()

    def close(self) -> None:
        for clip_id in list(self._open):
            self._release(clip_id)


# --------------------------------------------------------------------------- prefetch


@dataclass
class _Decoded:
    frame: int
    # Either a device/CPU RGB(A) tensor from ``frame_at`` (rasters, or ``worker_uploads``)
    # or the raw planes of a video frame (``SourceFrame``) the GPU thread converts.
    payload: "torch.Tensor | SourceFrame"


@dataclass
class _WorkerFailed:
    error: BaseException


class _Done:
    pass


_DONE = _Done()


@dataclass
class _LayerWorker:
    """One layer's decode thread + bounded queue of ``_Decoded`` frames."""

    layer: LayerSpec
    frame_duration: Fraction
    device: torch.device          # where the GPU thread wants the tensor
    decode_device: torch.device   # where the worker's FrameSource materializes it
    threads: int
    depth: int
    stop: threading.Event
    start_frame: int
    frames: "queue.Queue" = field(init=False)
    thread: threading.Thread = field(init=False)
    frames_decoded: int = 0
    failure: Optional[BaseException] = None

    def __post_init__(self) -> None:
        self.frames = queue.Queue(maxsize=self.depth)
        self.thread = threading.Thread(target=self._run, name=f"tensor-decode:{self.layer.clip_id}", daemon=True)
        self.thread.start()

    def _put(self, item: object) -> bool:
        """Blocking put that gives up when the pool is stopping; True if delivered."""

        while not self.stop.is_set():
            try:
                self.frames.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        source: Optional[FrameSource] = None
        try:
            source = open_source(self.layer, device=self.decode_device, threads=self.threads)
            # Video with the GPU thread converting: hand over raw planes, no CPU colour math.
            hand_planes = self.decode_device.type == "cpu" and isinstance(source, ClipDecoder)
            # A bounded render may begin in the middle of this layer. Starting
            # at the requested internal frame keeps the worker's first item in
            # lockstep with the GPU consumer instead of decoding discarded
            # timeline history from ``layer.first_frame``.
            for frame in range(max(self.layer.first_frame, self.start_frame), self.layer.end_frame):
                if self.stop.is_set():
                    return
                source_time = self.layer.source_time(frame, self.frame_duration)
                payload = source.packed_at(source_time) if hand_planes else source.frame_at(source_time)
                if not self._put(_Decoded(frame, payload)):
                    return
            self._put(_DONE)
        except BaseException as exc:  # noqa: BLE001 - delivered to the GPU thread
            self.failure = exc
            self._put(_WorkerFailed(exc))
        finally:
            if source is not None:
                self.frames_decoded = source.frames_decoded
                source.close()

    def pop(self, expected_frame: int, stats: SourcePoolStats, held: list[torch.Tensor]) -> torch.Tensor:
        """Take the next decoded frame; it must be ``expected_frame`` (loud otherwise)."""

        depth = self.frames.qsize()
        stats.queue_depth_sum += depth
        if depth == 0:
            stats.starved_frames += 1
        while True:
            try:
                item = self.frames.get(timeout=0.5)
                break
            except queue.Empty:
                if self.failure is not None:
                    self._raise_failure()
                if not self.thread.is_alive():
                    raise TensorRenderError(
                        f"{self.layer.path}: decode worker exited before delivering frame {expected_frame}"
                    )
        if isinstance(item, _WorkerFailed):
            self._raise_failure()
        if isinstance(item, _Done):
            raise TensorRenderError(
                f"{self.layer.path}: decode worker finished before frame {expected_frame} was requested"
            )
        assert isinstance(item, _Decoded)
        if item.frame != expected_frame:
            raise TensorRenderError(
                f"{self.layer.path}: prefetch schedule mismatch (got frame {item.frame}, GPU thread asked for {expected_frame})"
            )
        payload = item.payload
        if isinstance(payload, torch.Tensor) and payload.device == self.device:
            return payload
        # Worker decoded on the CPU: the GPU thread owns the upload and (for
        # raw video planes, ``SourceFrame``) the yuv->RGB conversion.  The
        # upload is queued (non_blocking) so it does not drain the device
        # stream; the host tensor is kept alive in ``held`` until the caller
        # has passed a device sync for this frame -- freeing it earlier is the
        # measured MPS frame-corruption gotcha.
        started = time.perf_counter()
        if isinstance(payload, SourceFrame):
            held.append(payload.planes)
            planes = payload.planes.to(self.device, non_blocking=True)
            tensor = planes_to_rgb(planes, payload.layout, payload.color)
        elif isinstance(payload, HDRFrame):
            held.append(payload.rgb)
            rgb = payload.rgb.to(self.device, non_blocking=True)
            tensor = hdr_to_sdr(rgb, payload.transfer)
        else:
            held.append(payload)
            tensor = payload.to(self.device, non_blocking=True)
        stats.upload_seconds += time.perf_counter() - started
        return tensor

    def _raise_failure(self) -> None:
        assert self.failure is not None
        raise TensorRenderError(f"{self.layer.path}: decode worker failed: {self.failure!r}") from self.failure

    def finish(self) -> None:
        """After the layer's last frame: expect the end sentinel, join, surface late failures."""

        if self.failure is None and self.thread.is_alive():
            try:
                item = self.frames.get(timeout=30.0)
            except queue.Empty as exc:
                raise TensorRenderError(f"{self.layer.path}: decode worker did not finish after its last frame") from exc
            if isinstance(item, _WorkerFailed):
                self._raise_failure()
            if not isinstance(item, _Done):
                raise TensorRenderError(f"{self.layer.path}: decode worker produced frames past the layer end")
        self.thread.join(timeout=30.0)
        if self.thread.is_alive():
            raise TensorRenderError(f"{self.layer.path}: decode worker did not exit")
        if self.failure is not None:
            self._raise_failure()


class PrefetchSources:
    """Pipelined pool: a decode thread per (soon-to-be) active layer, ``prefetch`` frames ahead.

    Why this exists: at 720p the serial loop spent about a third of its
    per-frame budget in PyAV decode + host->device upload; running that in
    worker threads (PyAV releases the GIL while decoding) lets it overlap the
    GPU thread's compositing and the encoder thread's x264 work, so wall time
    approaches the slowest stage instead of their sum.
    """

    def __init__(
        self,
        plan: TensorRenderPlan,
        *,
        device: torch.device,
        decoder_threads: int,
        prefetch: int = 8,
        open_ahead: Optional[int] = None,
        worker_uploads: bool = False,
        start_frame: int = 0,
    ) -> None:
        if prefetch < 1:
            raise TensorRenderError("prefetch depth must be >= 1")
        self._plan = plan
        self._device = device
        # MPS has one in-order stream and a blocking host->device copy drains it
        # while holding the dispatch queue, so uploads issued from decode
        # workers stall the GPU thread's kernel issue (measured: 3 ms -> 21 ms
        # per frame at 720p).  Default: workers decode + convert on the CPU
        # device and the GPU thread performs the single upload per layer-frame.
        self._decode_device = device if worker_uploads else torch.device("cpu")
        self._threads = decoder_threads
        self._depth = prefetch
        self._open_ahead = prefetch if open_ahead is None else open_ahead
        self._start_frame = start_frame
        self._pending = sorted(
            (layer for layer in plan.layers if layer.end_frame > start_frame),
            key=lambda layer: layer.first_frame,
        )
        self._workers: dict[str, _LayerWorker] = {}
        self._stop = threading.Event()
        self._held: list[torch.Tensor] = []
        self.stats = SourcePoolStats()

    def before_frame(self, frame: int) -> None:
        while self._pending and self._pending[0].first_frame <= frame + self._open_ahead:
            layer = self._pending.pop(0)
            self._workers[layer.clip_id] = _LayerWorker(
                layer=layer, frame_duration=self._plan.frame_duration, device=self._device,
                decode_device=self._decode_device, threads=self._threads, depth=self._depth, stop=self._stop,
                start_frame=self._start_frame,
            )
            self.stats.max_open_decoders = max(self.stats.max_open_decoders, len(self._workers))

    def frame(self, layer: LayerSpec, frame: int) -> torch.Tensor:
        worker = self._workers.get(layer.clip_id)
        if worker is None:
            raise TensorRenderError(f"{layer.path}: no decode worker open at frame {frame}")
        started = time.perf_counter()
        tensor = worker.pop(frame, self.stats, self._held)
        self.stats.wait_seconds += time.perf_counter() - started
        self.stats.frames_served += 1
        return tensor

    def take_held(self) -> list[torch.Tensor]:
        """Hand the caller the host buffers behind this frame's queued uploads.

        The caller must keep them alive until the device has passed the point
        where those uploads were issued (its per-frame download / event sync),
        then drop them; releasing earlier is the MPS frame-corruption gotcha.
        """

        held, self._held = self._held, []
        return held

    def after_frame(self, frame: int) -> None:
        for clip_id in [cid for cid, w in self._workers.items() if w.layer.end_frame <= frame + 1]:
            worker = self._workers.pop(clip_id)
            worker.finish()
            self.stats.decoded_source_frames += worker.frames_decoded

    def close(self) -> None:
        self._stop.set()
        for worker in self._workers.values():
            # Drain so a worker blocked on a full queue can observe ``stop``.
            while True:
                try:
                    worker.frames.get_nowait()
                except queue.Empty:
                    break
            worker.thread.join(timeout=30.0)
            self.stats.decoded_source_frames += worker.frames_decoded
        self._workers.clear()
