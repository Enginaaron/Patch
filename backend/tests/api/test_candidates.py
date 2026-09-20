"""Accept / reject through the API. FOUND is the identification being
confirmed; arrival is only ever movement.phase == "arrived". Motion and
detections are SIMULATED."""

import threading

from sqlmodel import select

from app.db import get_session
from app.models import Candidate, CandidateDecision, Find
from app.rover.simulation import SimVision, build_scenario
from tests.api.helpers import create_search, detail, now, target_in_view
from tests.rover.helpers import fast_config, make_world

ACTIVE = {"scanning", "checking", "waiting_for_confirmation", "centering", "approaching"}


def _finds(search_id: str) -> int:
    with get_session() as session:
        return len(session.exec(select(Find).where(Find.search_id == search_id)).all())


def _decision(candidate_id: str) -> CandidateDecision:
    with get_session() as session:
        return session.get(Candidate, candidate_id).decision


def _pending(client, tap, text="my bottle", **kwargs) -> tuple[str, str]:
    sid = create_search(client, text)["search_id"]
    candidate_id = tap.wait_type(sid, "candidate_found", **kwargs)["payload"]["candidate_id"]
    tap.wait_phase(sid, "waiting_for_confirmation")
    return sid, candidate_id


def test_accept_means_found_and_only_proximity_means_arrived(client, rover, tap):
    h = rover(*target_in_view())
    sid, candidate_id = _pending(client, tap)
    # Hold the rover's next look, so the state right after the acceptance can
    # be inspected: accepted, but nowhere near the item yet.
    h.vision.block_event = threading.Event()

    response = client.post(f"/api/searches/{sid}/candidates/{candidate_id}/accept")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "FOUND" and body["ended_at"] is not None
    assert body["pending_candidate"] is None
    accepted = body["accepted_candidate"]
    assert accepted["candidate_id"] == candidate_id and accepted["description"] == "a blue water bottle (simulated)"
    assert accepted["crop_url"].startswith(f"/media/searches/{sid}/")
    # FOUND, yet the rover has not moved an inch and is not "arrived".
    assert (body["movement"]["phase"], body["movement"]["active"]) == ("checking", True)
    assert body["terminal"] is False and body["can_resume"] is False
    assert h.drive.moves == [] and _finds(sid) == 1
    assert tap.wait_type(sid, "candidate_accepted")["payload"] == {"candidate_id": candidate_id, "status": "FOUND"}

    h.vision.release()
    arrived = tap.wait_phase(sid, "arrived", timeout=20)

    assert arrived.get("terminal") is True and arrived["payload"]["reason"] == "proximity_confirmed"
    body = detail(client, sid)
    assert body["status"] == "FOUND" and body["terminal"] is True and body["can_resume"] is False
    assert (body["movement"]["phase"], body["movement"]["active"]) == ("arrived", False)
    assert _finds(sid) == 1                               # arriving creates nothing new
    # It got there through bounded pulses, every one with the watchdog armed.
    moves = h.drive.moves
    assert any(move.command == "forward" for move in moves)
    assert all(move.ttl is not None and 0 < move.ttl < 2.0 for move in moves)
    assert h.drive.records[-1].command == "stop" and not h.world.is_moving()


def test_reject_goes_back_to_searching_and_the_next_object_is_offered(client, rover, tap):
    cfg = fast_config()
    world = build_scenario("two_bottles", make_world(cfg))
    h = rover(cfg, world)
    sid, decoy_id = _pending(client, tap, "my water bottle")
    assert detail(client, sid)["pending_candidate"]["description"] == "a green glass bottle on the floor"
    mark = tap.mark()

    response = client.post(f"/api/searches/{sid}/candidates/{decoy_id}/reject")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["accepted_candidate"] is None and body["terminal"] is False
    # The mission resumes at once, so by now it may already be further along.
    assert body["movement"]["active"] is True and body["movement"]["phase"] in ACTIVE
    assert _decision(decoy_id) == CandidateDecision.REJECTED and _finds(sid) == 0
    assert tap.wait_type(sid, "candidate_rejected")["payload"] == {"candidate_id": decoy_id, "status": "SEARCHING"}

    second = tap.wait_type(sid, "candidate_found", after=mark)["payload"]
    tap.wait_phase(sid, "waiting_for_confirmation", after=mark)
    assert second["candidate_id"] != decoy_id
    pending = detail(client, sid)["pending_candidate"]
    assert pending["candidate_id"] == second["candidate_id"]
    assert pending["description"] == "a blue water bottle beside a backpack"   # never the rejected one again
    assert any(move.command == "turn_right" for move in h.drive.moves)          # it had to keep scanning to get there


def test_decisions_that_no_longer_apply_are_conflicts(client, rover, tap):
    h = rover(*target_in_view())
    sid, candidate_id = _pending(client, tap)
    h.vision.block_event = threading.Event()   # park the approach: this test is about the lifecycle only
    assert client.post(f"/api/searches/{sid}/candidates/{candidate_id}/accept").status_code == 200

    again = client.post(f"/api/searches/{sid}/candidates/{candidate_id}/accept")
    late_no = client.post(f"/api/searches/{sid}/candidates/{candidate_id}/reject")

    for response in (again, late_no):
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "invalid_transition"
        assert response.json()["detail"]["message"]
    assert _finds(sid) == 1 and _decision(candidate_id) == CandidateDecision.ACCEPTED
    assert len(tap.events(sid, "candidate_accepted")) == 1 and tap.events(sid, "candidate_rejected") == []
    assert detail(client, sid)["status"] == "FOUND"


def test_unknown_searches_and_candidates_are_404(client, rover, tap):
    rover(*target_in_view())
    sid, candidate_id = _pending(client, tap)

    assert client.post(f"/api/searches/nope/candidates/{candidate_id}/accept").status_code == 404
    assert client.post(f"/api/searches/{sid}/candidates/nope/accept").status_code == 404
    assert client.post(f"/api/searches/{sid}/candidates/nope/reject").status_code == 404
    assert client.get("/api/searches/nope").status_code == 404

    # None of that touched the real candidate.
    body = detail(client, sid)
    assert body["status"] == "CANDIDATE_PENDING" and body["pending_candidate"]["candidate_id"] == candidate_id


def test_a_candidate_of_another_search_cannot_be_decided_here(client, rover, tap):
    h = rover(*target_in_view())
    first, first_candidate = _pending(client, tap)
    client.post("/api/rover/stop")                       # frees the rover; the question stays open
    second = create_search(client, "my other bottle")["search_id"]
    tap.wait_type(second, "candidate_found")

    response = client.post(f"/api/searches/{second}/candidates/{first_candidate}/accept")

    assert response.status_code == 404
    assert _decision(first_candidate) == CandidateDecision.PENDING and _finds(second) == 0
    assert detail(client, second)["status"] == "CANDIDATE_PENDING"
    assert h.drive.moves == []


def test_a_candidate_without_a_location_can_be_asked_about_but_never_driven_to(client, rover, tap):
    cfg, world = target_in_view()
    h = rover(cfg, world, vision=SimVision(world, found_without_box=True))
    sid, candidate_id = _pending(client, tap)
    pending = detail(client, sid)["pending_candidate"]
    assert pending["box"] is None and pending["crop_url"] == pending["image_url"]   # the crop is the full frame

    accepted_at = now()
    assert client.post(f"/api/searches/{sid}/candidates/{candidate_id}/accept").status_code == 200
    rest = tap.wait_phase(sid, "target_lost", "arrived", "stopped", "error")

    assert (rest["payload"]["phase"], rest["payload"]["reason"]) == ("target_lost", "no_box")
    body = detail(client, sid)
    assert body["status"] == "FOUND" and body["terminal"] is False and body["can_resume"] is True
    assert h.drive.moves == [] and h.moves_since(accepted_at) == []   # not one motor command
