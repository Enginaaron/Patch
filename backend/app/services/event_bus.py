import queue
import threading
from typing import Any

# Bridges the rover controller's mission thread (app/rover/controller.py,
# which replaced the per-search SearchWorker thread of Spec 12) to the SSE
# endpoint (Spec 13). A plain thread-safe queue.Queue per subscriber --
# publish() is called from the mission thread (or from an API call that
# stops / cancels / answers a candidate) and just needs to hand events off
# without blocking. The SSE generator in routers/searches.py is async: it
# polls its queue with get_nowait() from the event loop instead of blocking
# a threadpool thread per open search page.


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
