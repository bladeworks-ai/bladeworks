"""Raw-frame preview media sink: uncompressed frames over a WebSocket.

Architecture map
================

    preview scan/seek thread                     WebSocket send task
        -> RawFrameMediaSink.write_video             -> RawFrameMediaSink.get()
        -> YUV420 -> RGBA on the worker thread        -> asyncio.to_thread(get)
        -> struct-framed bytes                        -> websocket.send_bytes
        -> bounded thread-safe queue (backpressure)

Why this exists
---------------
The WebRTC sink (``webrtc.py``) is built for lossy, jitter-buffered network
delivery: aiortc encodes every frame (VP8/H.264) and the browser holds a
receiver jitter buffer before compositing. On loopback that machinery adds
~1 second of glass-to-glass latency for zero benefit — there is no network to
be resilient against. This sink is the local-first alternative: it hands the
compositor's YUV frame straight to the browser as raw RGBA over a WebSocket,
which the client paints with ``canvas.putImageData``. Measured seek->glass
latency drops from ~1000 ms to ~50 ms.

The transport is deliberately dumb: no codec, no container, no timestamps on
the wire beyond a frame index. The backend scan loop (``session.py``) is the
master clock — it already paces both video and audio to wall-clock realtime —
so the client can paint video on arrival and schedule audio on arrival and the
two stay in sync at the source.

Wire format (little-endian, no struct padding)
----------------------------------------------
* video: ``<B kind=0><I frameIndex><H width><H height>`` + width*height*4 RGBA
* audio: ``<B kind=1><d timeSeconds><I sampleRate><B channels>`` + interleaved
  signed 16-bit PCM (numSamples = remainingBytes / (channels * 2))

Backpressure
------------
The queue is bounded exactly like the WebRTC track queue. If the client cannot
keep up, ``write_*`` blocks up to two seconds and then fails loudly rather than
growing memory without bound — no silent frame dropping.
"""

from __future__ import annotations

import queue
import struct
import threading
from dataclasses import dataclass

import av
import numpy as np

from ..tensor.renderer import ComposedFrame
from .contracts import ComposedAudioFrame, PreviewAPIError


# Sender-loop signals returned by ``RawFrameMediaSink.get``. Distinct sentinels
# so the WebSocket handler can tell "closed, stop" from "idle, keep polling".
CLOSED = object()
EMPTY = object()

_KIND_VIDEO = 0
_KIND_AUDIO = 1


@dataclass(frozen=True)
class RawFramePayload:
    """One payload tagged with the control generation that produced it."""

    generation: int
    data: bytes


def _rgba_bytes(frame: ComposedFrame) -> bytes:
    """Convert one composited YUV420 frame to browser-ready RGBA bytes.

    Runs on the scan/seek worker thread (never the event loop) so the
    color-space conversion cost stays off the HTTP transport. PyAV's swscale
    path is the same well-optimized converter the WebRTC sink already relies
    on for ``av.VideoFrame`` construction, so this adds no new dependency.
    """

    video_frame = av.VideoFrame.from_ndarray(frame.yuv420p, format="yuv420p")
    rgba = video_frame.to_ndarray(format="rgba")
    return np.ascontiguousarray(rgba).tobytes()


class RawFrameMediaSink:
    """One preview session's outgoing raw-frame queue.

    The session writes frames from a worker thread; the WebSocket handler
    drains them from the event loop through ``get``. The queue is the only
    shared state and it is thread-safe.
    """

    def __init__(self, *, queue_depth: int = 16) -> None:
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=queue_depth)
        self._closed = False
        self._lock = threading.Lock()
        self._generation = 0

    # -- writer side (scan/seek worker thread) ----------------------------

    def _enqueue(self, payload: bytes) -> None:
        with self._lock:
            if self._closed:
                raise PreviewAPIError(
                    "preview_failed",
                    "Raw-frame media sink is closed.",
                    status=500,
                )
            item = RawFramePayload(generation=self._generation, data=payload)
        try:
            self._queue.put(item, timeout=2.0)
        except queue.Full as error:
            raise PreviewAPIError(
                "preview_failed",
                "Raw-frame preview queue remained full for two seconds.",
                status=500,
            ) from error

    def write_video(self, frame: ComposedFrame) -> None:
        header = struct.pack("<BIHH", _KIND_VIDEO, frame.frame, frame.width, frame.height)
        self._enqueue(header + _rgba_bytes(frame))

    def write_still(self, frame: ComposedFrame) -> None:
        """Send one paused seek frame.

        Unlike the WebRTC sink this needs no duplicate frame: there is no
        receiver jitter buffer to coax into releasing a lone frame, so a single
        raw frame paints immediately.
        """

        self.write_video(frame)

    def write_audio(self, frame: ComposedAudioFrame) -> None:
        # ``samples`` is planar (channels, numSamples); transpose to interleaved
        # (numSamples, channels) exactly as the WebRTC sink does before packing.
        interleaved = np.ascontiguousarray(frame.samples.T).astype("<i2", copy=False)
        channels = int(frame.samples.shape[0])
        header = struct.pack(
            "<BdIB",
            _KIND_AUDIO,
            float(frame.time),
            int(frame.sample_rate),
            channels,
        )
        self._enqueue(header + interleaved.tobytes())

    # -- reader side (WebSocket handler, via asyncio.to_thread) ------------

    def get(self, timeout: float) -> object:
        """Block up to ``timeout`` seconds for the next payload.

        Returns a generation-tagged payload, ``EMPTY`` when the wait timed out
        (so the caller can re-check for client disconnect), or ``CLOSED`` once
        the session has torn the sink down. The route checks the generation
        immediately before sending, which also invalidates an item that was
        dequeued just before a control-boundary flush.
        """

        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                return CLOSED if self._closed else EMPTY
        return CLOSED if item is CLOSED else item

    def is_current(self, item: RawFramePayload) -> bool:
        """Return whether a dequeued item still belongs to the active scan."""

        with self._lock:
            return not self._closed and item.generation == self._generation

    def flush(self) -> None:
        """Drop every buffered payload that has not yet been sent.

        The FIFO carries no generation tag, so once a control boundary supersedes
        the old scan the frames and audio it already enqueued must be discarded
        or the browser would paint stale video (playing on past a pause, or the
        wrong region after a seek) before the new still or playback stream.

        Safe to call from the control thread while the WebSocket reader is
        draining: ``queue.Queue`` is thread-safe and this only removes payloads
        that are already buffered. A ``CLOSED`` sentinel is only enqueued by
        ``close``, which is never concurrent with a control boundary, so it is
        never discarded here.
        """

        with self._lock:
            self._generation += 1
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # Best-effort wake for a reader blocked in ``get``. If the queue is full
        # the reader is already draining and will observe ``_closed`` on its
        # next timeout, so dropping the sentinel here is safe.
        try:
            self._queue.put_nowait(CLOSED)
        except queue.Full:
            pass


class RawFrameMediaFactory:
    """Create raw-frame sinks. No SDP negotiation — there is no peer."""

    def __init__(self, *, queue_depth: int = 16) -> None:
        self.queue_depth = queue_depth

    def open(self) -> RawFrameMediaSink:
        return RawFrameMediaSink(queue_depth=self.queue_depth)
