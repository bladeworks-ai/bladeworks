"""Thread-safe replayable SSE event storage for preview sessions.

Each subscriber gets all events after its ``Last-Event-ID`` and then waits for
new records. A bounded log permits ordinary browser reconnects without making
session memory grow with playback duration.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Iterator, Mapping

from .contracts import PreviewEvent


class EventStream:
    def __init__(self, *, history_limit: int = 512) -> None:
        self._condition = threading.Condition()
        self._events: deque[PreviewEvent] = deque(maxlen=history_limit)
        self._next_id = 1
        self._closed = False

    def publish(self, event: str, data: Mapping[str, object]) -> PreviewEvent:
        with self._condition:
            record = PreviewEvent(self._next_id, event, dict(data))
            self._next_id += 1
            self._events.append(record)
            self._condition.notify_all()
            return record

    def subscribe(self, *, after_id: int = 0) -> Iterator[PreviewEvent]:
        cursor = after_id
        while True:
            with self._condition:
                available = [event for event in self._events if event.id > cursor]
                if not available and not self._closed:
                    self._condition.wait(timeout=15.0)
                    available = [event for event in self._events if event.id > cursor]
                if not available and self._closed:
                    return
            for event in available:
                cursor = event.id
                yield event

    def snapshot(self, *, after_id: int = 0) -> tuple[PreviewEvent, ...]:
        """Return currently retained events without waiting for a subscriber."""

        with self._condition:
            return tuple(event for event in self._events if event.id > after_id)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
