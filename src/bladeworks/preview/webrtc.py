"""aiortc media adapter for Bladeworks's composed YUV preview frames.

Architecture map
================

    HTTP worker thread
        -> AiortcMediaFactory.negotiate(SDP offer)
        -> one dedicated asyncio loop + RTCPeerConnection
        -> TensorVideoTrack and optional TensorAudioTrack
        -> bounded asyncio queues

    preview scan/seek thread
        -> AiortcMediaSink.write_video(ComposedFrame)
        -> thread-safe bounded queue put
        -> TensorVideoTrack.recv() / TensorAudioTrack.recv()
        -> av.VideoFrame(yuv420p) / av.AudioFrame(s16)
        -> aiortc encoders and browser WebRTC tracks

The queue is intentionally bounded. If the browser or encoder cannot consume
frames, ``write_video`` applies backpressure instead of growing memory without
limit. Project timestamps remain in SSE; the RTP track uses a monotonic 90 kHz
clock so seeking backward cannot publish non-monotonic transport timestamps.

When the browser offer includes audio, scan PCM uses a separate WebRTC audio
track. Seek writes only video and leaves audio idle.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from fractions import Fraction
from typing import Optional

import av
import numpy as np
from aiortc import MediaStreamTrack, RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from ..tensor.renderer import ComposedFrame
from .contracts import ComposedAudioFrame, PreviewAPIError, SessionDescription


_CLOSED = object()


class TensorVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, *, queue_depth: int) -> None:
        super().__init__()
        self.frames: asyncio.Queue[object] = asyncio.Queue(maxsize=queue_depth)
        self._next_pts = 0

    async def recv(self) -> av.VideoFrame:
        item = await self.frames.get()
        if item is _CLOSED:
            raise asyncio.CancelledError
        assert isinstance(item, ComposedFrame)
        frame = av.VideoFrame.from_ndarray(item.yuv420p, format="yuv420p")
        frame.pts = self._next_pts
        frame.time_base = Fraction(1, 90_000)
        self._next_pts += max(1, round(90_000 * item.duration))
        return frame

    async def close_queue(self) -> None:
        if not self.frames.full():
            self.frames.put_nowait(_CLOSED)
        self.stop()


class TensorAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, *, queue_depth: int) -> None:
        super().__init__()
        self.frames: asyncio.Queue[object] = asyncio.Queue(maxsize=queue_depth)
        self._next_pts = 0

    async def recv(self) -> av.AudioFrame:
        item = await self.frames.get()
        if item is _CLOSED:
            raise asyncio.CancelledError
        assert isinstance(item, ComposedAudioFrame)
        packed = np.ascontiguousarray(item.samples.T).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(
            packed,
            format="s16",
            layout=item.layout,
        )
        frame.sample_rate = item.sample_rate
        frame.pts = self._next_pts
        frame.time_base = Fraction(1, item.sample_rate)
        self._next_pts += item.samples.shape[1]
        return frame

    async def close_queue(self) -> None:
        if not self.frames.full():
            self.frames.put_nowait(_CLOSED)
        self.stop()


class AiortcMediaSink:
    """One peer connection and its outgoing Bladeworks video track."""

    def __init__(self, *, queue_depth: int = 16) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="tensor-preview-webrtc",
            daemon=True,
        )
        self._thread.start()
        self._loop_ready.wait()
        self._queue_depth = queue_depth
        self._connection: Optional[RTCPeerConnection] = None
        self._track: Optional[TensorVideoTrack] = None
        self._audio_track: Optional[TensorAudioTrack] = None
        self._closed = False

    def _run_loop(self) -> None:
        """Own the complete asyncio loop and executor lifetime.

        Main callers:
        - the dedicated ``tensor-preview-webrtc`` thread created by this sink.

        Why this exists:
        aiortc performs codec work through the loop's default executor. Merely
        closing the event loop leaves those non-daemon worker threads alive,
        so Python can hang during final interpreter shutdown after a browser
        tab disappears with an active preview. Stop the executor explicitly
        before this thread exits.
        """

        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
            self._loop.close()

    def negotiate(self, offer: SessionDescription) -> SessionDescription:
        future = asyncio.run_coroutine_threadsafe(self._negotiate(offer), self._loop)
        try:
            return future.result(timeout=20.0)
        except Exception as error:
            self.close()
            raise PreviewAPIError(
                "preview_failed",
                f"WebRTC negotiation failed: {error}",
                status=500,
            ) from error

    async def _negotiate(self, offer: SessionDescription) -> SessionDescription:
        # This service is localhost-only. Host ICE candidates are sufficient
        # and avoid a public STUN round trip during every session creation.
        connection = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        # Store ownership immediately so a failure in SDP validation or answer
        # creation is still cleaned up by ``negotiate``'s exception path.
        self._connection = connection
        await connection.setRemoteDescription(
            RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        )
        offered_kinds = {transceiver.kind for transceiver in connection.getTransceivers()}
        if "video" not in offered_kinds:
            raise PreviewAPIError(
                "preview_failed",
                "WebRTC offer does not contain a video media section.",
                status=400,
            )
        track = TensorVideoTrack(queue_depth=self._queue_depth)
        connection.addTrack(track)
        audio_track = None
        if "audio" in offered_kinds:
            audio_track = TensorAudioTrack(queue_depth=25)
            connection.addTrack(audio_track)
        answer = await connection.createAnswer()
        await connection.setLocalDescription(answer)
        assert connection.localDescription is not None
        self._track = track
        self._audio_track = audio_track
        return SessionDescription(
            type=connection.localDescription.type,
            sdp=connection.localDescription.sdp,
        )

    def write_video(self, frame: ComposedFrame) -> None:
        if self._closed or self._track is None:
            raise PreviewAPIError(
                "preview_failed",
                "WebRTC media sink is closed or was not negotiated.",
                status=500,
            )
        future = asyncio.run_coroutine_threadsafe(self._track.frames.put(frame), self._loop)
        try:
            future.result(timeout=2.0)
        except concurrent.futures.TimeoutError as error:
            future.cancel()
            raise PreviewAPIError(
                "preview_failed",
                "WebRTC video queue remained full for two seconds.",
                status=500,
            ) from error

    def write_still(self, frame: ComposedFrame) -> None:
        """Queue a seek frame twice so the receiver jitter buffer releases it.

        A lone RTP video frame may remain buffered until the receiver observes
        the following timestamp. Two identical pixels with consecutive
        transport timestamps make paused seek visible without creating an MP4
        or continuously encoding the held frame.
        """

        self.write_video(frame)
        self.write_video(frame)

    def write_audio(self, frame: ComposedAudioFrame) -> None:
        if self._closed or self._audio_track is None:
            raise PreviewAPIError(
                "preview_failed",
                "WebRTC offer did not negotiate an audio track for an audible project.",
                status=500,
            )
        future = asyncio.run_coroutine_threadsafe(
            self._audio_track.frames.put(frame),
            self._loop,
        )
        try:
            future.result(timeout=2.0)
        except concurrent.futures.TimeoutError as error:
            future.cancel()
            raise PreviewAPIError(
                "preview_failed",
                "WebRTC audio queue remained full for two seconds.",
                status=500,
            ) from error

    def flush(self) -> None:
        """Drop server-side queued frames when a control boundary supersedes them.

        The track queues live on this sink's private event loop, so the drain
        runs there via ``run_coroutine_threadsafe``. Only the not-yet-sent
        server-side backlog is cleared; frames already handed to the peer sit in
        the receiver's own jitter buffer beyond our reach. A no-op once closed or
        before negotiation.
        """

        if self._closed or not self._loop.is_running():
            return
        future = asyncio.run_coroutine_threadsafe(self._flush_async(), self._loop)
        future.result(timeout=2.0)

    async def _flush_async(self) -> None:
        for track in (self._track, self._audio_track):
            if track is None:
                continue
            while True:
                try:
                    track.frames.get_nowait()
                except asyncio.QueueEmpty:
                    break

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._close_async(), self._loop)
            try:
                future.result(timeout=5.0)
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=5.0)

    async def _close_async(self) -> None:
        if self._track is not None:
            await self._track.close_queue()
        if self._audio_track is not None:
            await self._audio_track.close_queue()
        if self._connection is not None:
            await self._connection.close()


class AiortcMediaFactory:
    def __init__(self, *, queue_depth: int = 16) -> None:
        self.queue_depth = queue_depth

    def negotiate(self, offer: SessionDescription):
        sink = AiortcMediaSink(queue_depth=self.queue_depth)
        answer = sink.negotiate(offer)
        return answer, sink
