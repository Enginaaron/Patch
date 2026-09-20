import queue
import threading
from typing import Any

# Bridges the background SearchWorker thread (Spec 12) to the async SSE
# endpoint (Spec 13). A plain thread-safe queue.Queue per subscriber --
# publish() is called from the worker's own thread and just needs to hand
# events off without blocking it; the SSE generator (a sync generator, same
# pattern as the existing MJPEG stream) blocks on its own queue with a
# timeout, which is fine since StreamingResponse runs sync generators in a
# thread, not on the asyncio event loop.


class SearchEventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, list["queue.Queue[dict[str, Any]]"]] = {}

    def subscribe(self, search_id: str) -> "queue.Queue[dict[str, Any]]":
        q: "queue.Queue[dict[str, Any]]" = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(search_id, []).append(q)
        return q

    def unsubscribe(self, search_id: str, q: "queue.Queue[dict[str, Any]]") -> None:
        with self._lock:
            subs = self._subscribers.get(search_id)
            if not subs:
                return
            if q in subs:
                subs.remove(q)
            if not subs:
                self._subscribers.pop(search_id, None)

    def publish(self, search_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subscribers.get(search_id, []))
        for q in subs:
            q.put(event)


event_bus = SearchEventBus()
