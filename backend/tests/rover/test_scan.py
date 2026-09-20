"""Scanning, candidate questions, rejection memory and the scan budgets.
All detections here are SIMULATED (app.rover.simulation)."""

from pathlib import Path

import pytest

from app.config import settings
from app.models import CandidateDecision, SearchStatus
from app.rover.simulation import build_scenario
from tests.rover.helpers import (
    add_bottle,
    candidates_of,
    fast_config,
    finds_of,
    get_search,
    make_world,
    movement_row,
    search_status,
)


def _two_bottles(harness, **cfg_overrides):
    cfg = fast_config(**cfg_overrides)
    return harness(cfg, build_scenario("two_bottles", make_world(cfg)))


def test_scan_exhaustion_is_bounded(harness):
    cfg = fast_config(scan_max_steps=6)
    h = harness(cfg)                                   # empty world: nothing to find
    sid, log = h.search("my bottle")

    started = h.controller.start_search(sid)
    assert started.phase.value == "scanning" and started.active
    assert started.scan_budget == 6 and started.vision_mode == "simulation" and started.driver == "sim"

    end = log.wait_rest()
    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("exhausted", "scan_budget")
    assert end["payload"]["message"] == "I looked all around and couldn't find it"
    assert end["payload"]["scan_steps"] == 6 and end["payload"]["active"] is False
    assert "terminal" not in end                        # only "arrived" is terminal

    # Exactly the budget, every pulse identical, every pulse watchdog-armed.
    moves = h.drive.moves
    assert [(m.command, m.speed) for m in moves] == [("turn_right", cfg.turn_speed)] * 6
    assert all(m.ttl == pytest.approx(cfg.scan_pulse_seconds + cfg.pulse_watchdog_margin_seconds) for m in moves)
    assert h.drive.records[-1].command == "stop" and not h.world.is_moving()
    assert h.world.pose()[2] == pytest.approx(6 * 20.0)  # six 20 degree steps (simulated)

    # The recognition lifecycle is untouched; the body's state is persisted.
    assert search_status(sid) == SearchStatus.SEARCHING
    row = movement_row(sid)
    assert (row.phase, row.reason) == ("exhausted", "scan_budget")
    assert h.controller.active_search_id() is None
    assert h.tracking.value is None


def test_scan_direction_left(harness):
    h = harness(fast_config(scan_max_steps=3, scan_direction="left"))
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    log.wait_rest()
    assert [m.command for m in h.drive.moves] == ["turn_left"] * 3
    assert h.world.pose()[2] == pytest.approx(-60.0)


def test_periodic_omni_check_despite_yolo_misses(harness):
    """The local detector sees nothing at all, yet every third stop still gets
    an OMNI check -- which is what finds the target."""
    cfg = fast_config(omni_check_every_steps=3)
    world = make_world(cfg)
    add_bottle(world, bearing=120, distance=1.5, is_target=True)
    h = harness(cfg, world)
    h.vision.screen_blind = True
    sid, log = h.search("my bottle")

    h.controller.start_search(sid)
    found = log.wait_type("candidate_found")

    assert found["payload"]["description"] == "target_bottle (simulated)"
    assert len(h.drive.moves) == 6                      # headings 0..120 in 20 degree steps
    assert h.vision.screen_calls == 7                   # screened at every stop ...
    assert h.vision.identify_calls == 3                 # ... OMNI only at stops 0, 3 and 6
    assert search_status(sid) == SearchStatus.CANDIDATE_PENDING


def test_yolo_hit_triggers_omni_immediately(harness):
    cfg = fast_config(omni_check_every_steps=50)        # the periodic check would never fire
    world = make_world(cfg)
    add_bottle(world, bearing=40, distance=1.5, is_target=True)
    h = harness(cfg, world)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    log.wait_type("candidate_found")
    # Stop 0 (first view always checked) + the first stop where YOLO saw a bottle.
    assert h.vision.identify_calls == 2
    assert len(h.drive.moves) == 1


def test_target_without_category_uses_omni_on_every_stop(harness):
    cfg = fast_config()
    world = make_world(cfg)
    world.add_at("keys", bearing=60, distance=1.0, coco_class="keys", description="a bunch of keys (simulated)",
                 width=0.10, height=0.05, is_target=True)
    h = harness(cfg, world)
    sid, log = h.search("my keys")

    state = h.controller.start_search(sid)
    assert state.category is None
    log.wait_type("candidate_found")

    assert h.vision.screen_calls == 0
    assert h.vision.identify_calls == len(h.drive.moves) + 1
    assert h.controller.state_for(sid).category is None


def test_category_is_exposed_in_movement_state(harness):
    h = harness(fast_config(scan_max_steps=1))
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    end = log.wait_rest()
    assert end["payload"]["category"] == "bottle"


def test_candidate_found_payload_images_and_waiting(harness, media_root):
    h = _two_bottles(harness)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)

    found = log.wait_type("candidate_found")
    waiting = log.wait_phase("waiting_for_confirmation")

    payload = found["payload"]
    assert set(payload) == {"candidate_id", "image_url", "crop_url", "box", "description"}
    assert payload["description"] == "a green glass bottle on the floor"   # the decoy comes first
    assert payload["image_url"].startswith(f"/media/searches/{sid}/") and payload["image_url"].endswith("_full.jpg")
    assert payload["crop_url"].endswith("_crop.jpg")
    for url in (payload["image_url"], payload["crop_url"]):
        assert (media_root / url.removeprefix("/media/")).is_file()

    [candidate] = candidates_of(sid)
    assert candidate.id == payload["candidate_id"] and candidate.decision == CandidateDecision.PENDING
    assert search_status(sid) == SearchStatus.CANDIDATE_PENDING

    assert waiting["payload"]["message"] == "Is this your bottle?"            # "my bottle", said back
    assert waiting["payload"]["active"] is True and waiting["payload"]["candidates_presented"] == 1
    assert h.controller.active_search_id() == sid
    assert h.tracking.value == payload["box"]

    # Parked on the question: wheels stopped, and they stay stopped.
    moves_before = len(h.drive.moves)
    assert h.drive.records[-1].command == "stop" and not h.world.is_moving()
    assert not h.drive.wait_for_moves(moves_before + 1, timeout=0.3)


def test_rejection_resumes_and_the_same_object_is_not_shown_again(harness):
    h = _two_bottles(harness)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    decoy = log.wait_type("candidate_found")["payload"]

    state = h.controller.reject_candidate(sid, decoy["candidate_id"])
    assert state.phase.value == "scanning" and state.active
    rejected = log.wait_type("candidate_rejected")
    assert rejected["payload"] == {"candidate_id": decoy["candidate_id"], "status": "SEARCHING"}

    target = log.wait_type("candidate_found")["payload"]
    assert target["description"] == "a blue water bottle beside a backpack"
    decisions = [c.decision for c in candidates_of(sid)]
    assert decisions == [CandidateDecision.REJECTED, CandidateDecision.PENDING]
    # The model was told what was rejected (description + crop), on every later call.
    later = h.vision.requests[-1]
    assert [hint.description for hint in later.rejected] == [decoy["description"]]
    assert later.rejected[0].crop_jpeg is not None and later.rejected[0].crop_jpeg[:2] == b"\xff\xd8"
    assert log.of_type("candidate_suppressed") == []


def test_acceptance_marks_found_and_starts_the_approach(harness):
    h = _two_bottles(harness)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    first = log.wait_type("candidate_found")["payload"]
    h.controller.reject_candidate(sid, first["candidate_id"])
    second = log.wait_type("candidate_found")["payload"]

    state = h.controller.accept_candidate(sid, second["candidate_id"])
    assert state.phase.value == "checking" and state.active
    accepted = log.wait_type("candidate_accepted")
    assert accepted["payload"] == {"candidate_id": second["candidate_id"], "status": "FOUND"}

    # FOUND is the user's confirmation of the identification -- recorded at once,
    # long before (and regardless of whether) the rover gets there.
    search = get_search(sid)
    assert search.status == SearchStatus.FOUND and search.ended_at is not None
    [find] = finds_of(sid)
    assert find.crop_path == second["crop_url"].removeprefix("/media/")
    assert h.controller.state_for(sid).phase.value != "arrived"


def test_suppression_by_bearing_when_vision_ignores_hints(harness):
    h = _two_bottles(harness)
    h.vision.ignore_rejections = True                  # a model that keeps reporting the decoy
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    decoy = log.wait_type("candidate_found")["payload"]
    h.controller.reject_candidate(sid, decoy["candidate_id"])

    target = log.wait_type("candidate_found")["payload"]
    assert target["description"] == "a blue water bottle beside a backpack"

    suppressed = log.of_type("candidate_suppressed")
    assert len(suppressed) >= 2                        # seen again from several headings, never re-asked
    assert all(
        e["payload"] == {"description": decoy["description"], "reason": "previously_rejected"} for e in suppressed
    )
    assert [c.description for c in candidates_of(sid)] == [decoy["description"], target["description"]]


def test_candidate_budget(harness):
    h = _two_bottles(harness, max_candidates=1)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    decoy = log.wait_type("candidate_found")["payload"]
    h.controller.reject_candidate(sid, decoy["candidate_id"])

    end = log.wait_rest()
    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("exhausted", "candidate_budget")
    assert len(candidates_of(sid)) == 1                # the second match was never turned into a question
    assert search_status(sid) == SearchStatus.SEARCHING
    assert h.drive.records[-1].command == "stop"


def test_low_likelihood_match_is_not_presented(harness):
    cfg = fast_config(scan_max_steps=4, candidate_likelihood_threshold="high")
    world = make_world(cfg)
    add_bottle(world, bearing=0, distance=1.5, is_target=True)
    h = harness(cfg, world)
    h.vision.likelihood = "medium"
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    end = log.wait_rest()
    assert end["payload"]["reason"] == "scan_budget"
    assert candidates_of(sid) == [] and log.of_type("candidate_found") == []


def test_candidate_images_are_real_jpegs_under_media_root(harness):
    h = _two_bottles(harness)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    payload = log.wait_type("candidate_found")["payload"]
    full = Path(settings.media_root, payload["image_url"].removeprefix("/media/")).read_bytes()
    crop = Path(settings.media_root, payload["crop_url"].removeprefix("/media/")).read_bytes()
    assert full[:2] == b"\xff\xd8" and crop[:2] == b"\xff\xd8"
    assert len(crop) < len(full)                        # the crop really is a cut-out


def test_omni_gets_the_full_text_while_messages_say_it_back_naturally(harness):
    """Category comes from the comprehension seam; OMNI still receives exactly
    what the user asked for; landmarks gate nothing; and what the rover SAYS
    is the item, not the whole sentence."""
    from types import SimpleNamespace

    from app.rover.controller import spoken_target

    raw = "Find my bottle beside my backpack"

    def resolver(text):
        assert text == raw
        # Like the rule-based adapter: target_text is the raw text, trimmed.
        return SimpleNamespace(category="bottle", landmarks=("backpack",), target_text=raw)

    cfg = fast_config()
    world = make_world(cfg)
    add_bottle(world, bearing=0, distance=1.5, is_target=True)      # no backpack anywhere in this world
    h = harness(cfg, world, resolver=resolver)
    sid, log = h.search(raw)
    h.controller.start_search(sid)
    log.wait_type("candidate_found")                               # found although the landmark is absent
    waiting = log.wait_phase("waiting_for_confirmation")

    assert waiting["payload"]["message"] == "Is this your bottle beside your backpack?"
    assert waiting["payload"]["category"] == "bottle"
    assert all(request.target_text == raw for request in h.vision.requests)

    assert spoken_target("my keys") == "your keys"
    assert spoken_target("Where is my phone?") == "your phone"
    assert spoken_target("can you find the red mug on my desk please") == "the red mug on your desk"
    assert spoken_target("phone") == "your phone"
    assert spoken_target("mystery novel") == "your mystery novel"    # whole words only
    assert spoken_target("get well card") == "your get well card"    # "get" is only a lead-in before my/the/a
    assert spoken_target("   ") == "your item"


def test_a_failing_resolver_falls_back_to_omni_sampling(harness):
    def broken(text):
        raise RuntimeError("comprehension provider is down")

    cfg = fast_config()
    world = make_world(cfg)
    add_bottle(world, bearing=20, distance=1.5, is_target=True)
    h = harness(cfg, world, resolver=broken)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    log.wait_type("candidate_found")
    assert h.vision.screen_calls == 0                               # no category -> OMNI on every stop
    assert h.controller.state_for(sid).category is None
