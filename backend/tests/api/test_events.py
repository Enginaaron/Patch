"""GET /api/searches/{id}/events (server-sent events).

The stream is an async generator; most tests iterate it directly so they can
act between two messages, every wait bounded by a timeout. Motion and
detections are SIMULATED.
"""

import asyncio
import json
import threading

from app.db import get_session
from app.models import RoverMovement, SearchStatus
from app.routers import searches
from app.services.event_bus import event_bus
from tests.api.helpers import create_search, nothing_to_find, target_in_view
from tests.rover.helpers import make_search

TIMEOUT = 10.0


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, TIMEOUT))


def _data(message: str) -> dict:
    assert message.startswith("data: ") and message.endswith("\n\n")
    return json.loads(message[len("data: "):])


def _stream(search_id: str, **kwargs):
    kwargs.setdefault("poll_seconds", 0.005)
    return searches._event_stream(search_id, **kwargs)


def _track_subscriptions(monkeypatch) -> dict:
    """Counts live bus subscriptions, to show the stream always unsubscribes."""
    live = {"count": 0}
    subscribe, unsubscribe = event_bus.subscribe, event_bus.unsubscribe

    def counted_subscribe(search_id):
        live["count"] += 1
        return subscribe(search_id)

    def counted_unsubscribe(search_id, q):
        live["count"] -= 1
        unsubscribe(search_id, q)

    monkeypatch.setattr(event_bus, "subscribe", counted_subscribe)
    monkeypatch.setattr(event_bus, "unsubscribe", counted_unsubscribe)
    return live


def test_first_message_is_the_current_state_and_a_terminal_event_closes_the_stream(client, rover, tap, monkeypatch):
    h = rover(*target_in_view())
    live = _track_subscriptions(monkeypatch)
    sid = create_search(client, "my bottle")["search_id"]
    tap.wait_phase(sid, "waiting_for_confirmation")

    async def scenario():
        stream = _stream(sid)
        first = _data(await anext(stream))
        # The user cancels from another thread while the stream is open.
        await asyncio.to_thread(h.controller.cancel_search, sid)
        return first, [_data(message) async for message in stream]

    first, rest = _run(scenario())

    assert first["type"] == "search_status" and first["terminal"] is False
    assert first["payload"]["status"] == "CANDIDATE_PENDING" and first["payload"]["terminal"] is False
    movement = first["payload"]["movement"]
    assert (movement["phase"], movement["active"], movement["search_id"]) == ("waiting_for_confirmation", True, sid)

    # The halt is reported first, the terminal event last -- and then the stream ends by itself.
    assert [(e["type"], e.get("terminal", False)) for e in rest] == [("movement", False), ("search_cancelled", True)]
    assert rest[0]["payload"]["reason"] == "cancelled"
    assert live["count"] == 0


def test_a_finished_search_gets_one_message_and_the_stream_closes(client, rover, monkeypatch):
    rover(*target_in_view())
    live = _track_subscriptions(monkeypatch)
    cancelled = make_search("my bottle", status=SearchStatus.CANCELLED)
    arrived = make_search("my bottle", status=SearchStatus.FOUND)
    found_only = make_search("my bottle", status=SearchStatus.FOUND)
    with get_session() as session:
        session.add(RoverMovement(search_id=arrived, phase="arrived", message="I'm next to your bottle", reason="proximity_confirmed"))
        session.add(RoverMovement(search_id=found_only, phase="target_lost", message="I lost sight of it", reason="not_visible"))
        session.commit()

    async def collect(search_id):
        return [_data(message) async for message in _stream(search_id)]

    for search_id, status in ((cancelled, "CANCELLED"), (arrived, "FOUND")):
        messages = _run(collect(search_id))
        assert len(messages) == 1
        assert messages[0]["type"] == "search_status" and messages[0]["terminal"] is True
        assert messages[0]["payload"]["status"] == status and messages[0]["payload"]["terminal"] is True
    assert live["count"] == 0

    # FOUND alone is NOT terminal: the rover has not arrived, the search can
    # still be resumed, so the stream stays open (here: until a keep-alive).
    async def found_but_not_arrived():
        stream = _stream(found_only, keepalive_seconds=0.02)
        first = _data(await anext(stream))
        second = await anext(stream)
        await stream.aclose()
        return first, second

    first, second = _run(found_but_not_arrived())
    assert first["terminal"] is False and first["payload"]["movement"]["phase"] == "target_lost"
    assert second == ": keep-alive\n\n"
    assert live["count"] == 0                            # closing the stream unsubscribed it


def test_nothing_published_while_the_snapshot_is_read_gets_lost(client, rover, monkeypatch):
    """Subscribe BEFORE the snapshot: an event that fires while the current
    state is being read must still reach the client."""
    rover(*target_in_view())
    sid = make_search("my bottle")
    snapshot = searches._status_snapshot

    def snapshot_with_a_concurrent_change(search_id):
        state = snapshot(search_id)                      # reads "SEARCHING"...
        event_bus.publish(search_id, {"type": "candidate_found", "payload": {"candidate_id": "c1"}})   # ...then it changes
        return state

    monkeypatch.setattr(searches, "_status_snapshot", snapshot_with_a_concurrent_change)

    async def scenario():
        stream = _stream(sid)
        first = _data(await anext(stream))
        second = _data(await anext(stream))
        await stream.aclose()
        return first, second

    first, second = _run(scenario())
    assert first["payload"]["status"] == "SEARCHING"
    assert second == {"type": "candidate_found", "payload": {"candidate_id": "c1"}}


def test_arrival_closes_the_stream_and_other_events_do_not(client, rover):
    rover(*target_in_view())
    sid = make_search("my bottle", status=SearchStatus.FOUND)

    async def scenario():
        stream = _stream(sid)
        await anext(stream)
        event_bus.publish(sid, {"type": "movement", "payload": {"phase": "approaching"}})
        event_bus.publish(sid, {"type": "vision_error", "payload": {"message": "timeout"}})
        event_bus.publish(sid, {"type": "movement", "payload": {"phase": "arrived"}, "terminal": True})
        event_bus.publish(sid, {"type": "movement", "payload": {"phase": "never sent"}})
        return [_data(message) async for message in stream]

    messages = _run(scenario())
    assert [m["payload"].get("phase", m["type"]) for m in messages] == ["approaching", "vision_error", "arrived"]
    assert messages[-1]["terminal"] is True


def test_the_endpoint_streams_a_finished_search_and_404s_an_unknown_one(client, rover):
    rover(*target_in_view())
    sid = make_search("my bottle", status=SearchStatus.CANCELLED)

    response = client.get(f"/api/searches/{sid}/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    messages = [_data(chunk + "\n\n") for chunk in response.text.split("\n\n") if chunk]
    assert len(messages) == 1 and messages[0]["terminal"] is True and messages[0]["payload"]["status"] == "CANCELLED"

    assert client.get("/api/searches/nope/events").status_code == 404


def test_the_endpoint_follows_a_live_search_until_it_is_cancelled(client, rover, tap, monkeypatch):
    h = rover(*nothing_to_find())
    sid = create_search(client, "my bottle")["search_id"]
    assert h.drive.wait_for_moves(1)

    # The test client only returns once the stream has ended, so the cancel
    # has to come from another thread -- after the first message was computed.
    first_message_ready = threading.Event()
    snapshot = searches._status_snapshot

    def snapshot_then_signal(search_id):
        state = snapshot(search_id)
        first_message_ready.set()
        return state

    monkeypatch.setattr(searches, "_status_snapshot", snapshot_then_signal)

    def cancel_once_streaming():
        if first_message_ready.wait(TIMEOUT):
            h.controller.cancel_search(sid)

    canceller = threading.Thread(target=cancel_once_streaming, daemon=True)
    canceller.start()
    response = client.get(f"/api/searches/{sid}/events")
    canceller.join(TIMEOUT)

    assert response.status_code == 200
    messages = [_data(chunk + "\n\n") for chunk in response.text.split("\n\n") if chunk.startswith("data: ")]
    assert messages[0]["type"] == "search_status" and messages[0]["payload"]["status"] == "SEARCHING"
    assert messages[0]["payload"]["movement"]["active"] is True and messages[0]["terminal"] is False
    assert messages[-1] == {"type": "search_cancelled", "payload": {"status": "CANCELLED", "reason": "cancelled"}, "terminal": True}
    assert h.controller.active_search_id() is None and not h.world.is_moving()
