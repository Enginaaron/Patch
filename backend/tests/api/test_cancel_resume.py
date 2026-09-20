"""Cancel (halts first, idempotent), resume, and what a backend restart looks
like through the API. Motion and detections are SIMULATED."""

import threading

from app.db import get_session
from app.models import RoverMovement, Search, SearchStatus
from tests.api.helpers import create_search, detail, nothing_to_find, now, target_in_view


def _status(search_id: str) -> SearchStatus:
    with get_session() as session:
        return session.get(Search, search_id).status


def test_cancel_halts_the_rover_first_and_is_idempotent(client, rover, tap):
    h = rover(*nothing_to_find())
    sid = create_search(client, "my bottle")["search_id"]
    assert h.drive.wait_for_moves(2)                     # mid-scan: the wheels are being pulsed

    response = client.post(f"/api/searches/{sid}/cancel")
    answered_at = now()

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "CANCELLED" and body["ended_at"] is not None
    assert body["terminal"] is True and body["can_resume"] is False
    assert (body["movement"]["phase"], body["movement"]["reason"], body["movement"]["active"]) == ("stopped", "cancelled", False)
    assert h.controller.active_search_id() is None

    # Halt first, then the terminal event.
    types = [(e["type"], e["payload"].get("reason")) for e in tap.events(sid) if e["type"] in ("movement", "search_cancelled")]
    assert types.index(("movement", "cancelled")) < types.index(("search_cancelled", "cancelled"))
    assert tap.wait_type(sid, "search_cancelled").get("terminal") is True

    # By the time the API answered, the wheels were stopped for good.
    assert h.controller.join_missions(5)
    assert h.drive.records[-1].command == "stop" and not h.world.is_moving()
    assert h.moves_since(answered_at) == []

    # Again: same answer, no second terminal event, still nothing moving.
    again = client.post(f"/api/searches/{sid}/cancel")
    assert again.status_code == 200 and again.json()["status"] == "CANCELLED"
    assert len(tap.events(sid, "search_cancelled")) == 1
    assert h.moves_since(answered_at) == []

    assert client.post("/api/searches/nope/cancel").status_code == 404


def test_cancel_while_the_question_is_open_withdraws_it(client, rover, tap):
    h = rover(*target_in_view())
    sid = create_search(client, "my bottle")["search_id"]
    candidate_id = tap.wait_type(sid, "candidate_found")["payload"]["candidate_id"]
    tap.wait_phase(sid, "waiting_for_confirmation")

    body = client.post(f"/api/searches/{sid}/cancel").json()

    assert body["status"] == "CANCELLED" and body["pending_candidate"] is None   # nothing left to answer
    late = client.post(f"/api/searches/{sid}/candidates/{candidate_id}/accept")
    assert late.status_code == 409 and late.json()["detail"]["code"] == "invalid_transition"
    assert _status(sid) == SearchStatus.CANCELLED and h.drive.moves == []


def test_late_vision_result_after_a_cancel_changes_nothing(client, rover, tap):
    h = rover(*target_in_view())
    h.vision.block_event = threading.Event()             # the first look "hangs" with a positive answer pending
    sid = create_search(client, "my bottle")["search_id"]
    assert h.vision.entered_event.wait(5)

    assert client.post(f"/api/searches/{sid}/cancel").status_code == 200   # does not wait for the hanging call
    h.vision.release()
    assert h.controller.join_missions(5)

    body = detail(client, sid)
    assert body["status"] == "CANCELLED" and body["pending_candidate"] is None
    assert tap.events(sid, "candidate_found") == [] and h.drive.moves == []


def test_cancelling_a_resting_search_leaves_the_active_one_alone(client, rover, tap):
    h = rover(*nothing_to_find())
    resting = create_search(client, "my bottle")["search_id"]
    assert h.drive.wait_for_moves(1)
    assert client.post("/api/rover/stop").status_code == 200
    active = create_search(client, "my laptop")["search_id"]
    moves_before = len(h.drive.moves)

    assert client.post(f"/api/searches/{resting}/cancel").status_code == 200

    assert _status(resting) == SearchStatus.CANCELLED
    assert h.controller.active_search_id() == active and detail(client, active)["movement"]["active"] is True
    assert h.drive.wait_for_moves(moves_before + 2)      # ...and it keeps scanning


def test_stop_keeps_the_search_and_resume_continues_it(client, rover, tap):
    h = rover(*nothing_to_find())
    sid = create_search(client, "my bottle")["search_id"]
    assert h.drive.wait_for_moves(2)

    stopped = client.post("/api/rover/stop")
    stopped_at = now()

    assert stopped.status_code == 200
    body = detail(client, sid)
    assert body["status"] == "SEARCHING"                 # the search itself is untouched...
    assert (body["movement"]["phase"], body["movement"]["reason"], body["movement"]["active"]) == ("stopped", "user_stop", False)
    assert body["can_resume"] is True and body["terminal"] is False   # ...and resumable
    assert h.controller.join_missions(5) and h.moves_since(stopped_at) == []

    resumed = client.post(f"/api/searches/{sid}/resume")

    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["movement"]["active"] is True and resumed.json()["can_resume"] is False
    assert h.drive.wait_for_moves(len(h.drive.moves) + 2)
    assert h.controller.active_search_id() == sid


def test_resume_is_refused_when_it_does_not_apply(client, rover, tap):
    h = rover(*target_in_view())
    cancelled = create_search(client, "my bottle")["search_id"]
    tap.wait_phase(cancelled, "waiting_for_confirmation")
    client.post(f"/api/searches/{cancelled}/cancel")
    active = create_search(client, "my other bottle")["search_id"]
    tap.wait_phase(active, "waiting_for_confirmation")

    # Busy with another search: 409 rover_busy (checked before anything else).
    busy = client.post(f"/api/searches/{cancelled}/resume")
    assert busy.status_code == 409 and busy.json()["detail"]["code"] == "rover_busy"
    assert busy.json()["detail"]["active_search_id"] == active

    # Resuming the search that is already running is a harmless no-op.
    generation = h.controller.snapshot()["generation"]
    same = client.post(f"/api/searches/{active}/resume")
    assert same.status_code == 200 and h.controller.snapshot()["generation"] == generation

    client.post("/api/rover/stop")
    # The rover is free now, but a CANCELLED search stays cancelled...
    gone = client.post(f"/api/searches/{cancelled}/resume")
    assert gone.status_code == 409 and gone.json()["detail"]["code"] == "invalid_transition"
    # ...and an unanswered question has to be answered, not driven past.
    unanswered = client.post(f"/api/searches/{active}/resume")
    assert unanswered.status_code == 409 and unanswered.json()["detail"]["code"] == "invalid_transition"
    assert detail(client, active)["can_resume"] is False

    assert client.post("/api/searches/nope/resume").status_code == 404
    assert h.drive.moves == []


def test_after_a_restart_nothing_moves_until_the_user_resumes(client, rover, tap):
    h = rover(*nothing_to_find())
    sid = create_search(client, "my bottle")["search_id"]
    assert h.drive.wait_for_moves(2)

    # The backend dies mid-mission: the last thing on disk is an ACTIVE phase.
    h.controller.shutdown()
    with get_session() as session:
        row = session.get(RoverMovement, sid)
        row.phase, row.message, row.reason = "scanning", "Looking for your bottle…", None
        session.add(row)
        session.commit()
    records_before = len(h.drive.records)
    h.new_controller()                                   # the new process: no live mission

    body = detail(client, sid)

    assert body["status"] == "SEARCHING"
    assert (body["movement"]["phase"], body["movement"]["reason"], body["movement"]["active"]) == ("stopped", "interrupted", False)
    assert body["can_resume"] is True and body["terminal"] is False
    assert len(h.drive.records) == records_before        # reporting it did not touch the wheels
    state = client.get("/api/rover/state").json()
    assert state["active_search_id"] is None

    assert client.post(f"/api/searches/{sid}/resume").status_code == 200
    assert h.drive.wait_for_moves(len(h.drive.moves) + 1)
    assert detail(client, sid)["movement"]["active"] is True


def test_answering_a_question_after_a_restart_does_not_start_the_wheels(client, rover, tap):
    h = rover(*target_in_view())
    sid = create_search(client, "my bottle")["search_id"]
    candidate_id = tap.wait_type(sid, "candidate_found")["payload"]["candidate_id"]
    tap.wait_phase(sid, "waiting_for_confirmation")
    h.controller.shutdown()
    with get_session() as session:
        row = session.get(RoverMovement, sid)
        row.phase, row.reason = "waiting_for_confirmation", None
        session.add(row)
        session.commit()
    h.new_controller()

    body = client.post(f"/api/searches/{sid}/candidates/{candidate_id}/accept").json()

    # The identification is recorded, the body stays where it is until "resume".
    assert body["status"] == "FOUND" and body["accepted_candidate"]["candidate_id"] == candidate_id
    assert (body["movement"]["phase"], body["movement"]["reason"], body["movement"]["active"]) == ("stopped", "interrupted", False)
    assert body["can_resume"] is True and h.drive.moves == []

    mark = tap.mark()
    assert client.post(f"/api/searches/{sid}/resume").status_code == 200
    rest = tap.wait_phase(sid, "arrived", "target_lost", "error", "stopped", "exhausted", timeout=20, after=mark)
    assert rest["payload"]["phase"] == "arrived"
    assert detail(client, sid)["terminal"] is True
