"""Centring, incremental approach, conservative arrival, and every way an
approach stops short. All detections and all motion here are SIMULATED."""

import pytest

from app.models import SearchStatus
from app.rover.simulation import SimVision, build_scenario
from tests.rover.helpers import (
    ScriptedVision,
    add_bottle,
    candidates_of,
    fast_config,
    finds_of,
    make_world,
    search_status,
)


def _accept_first_candidate(h, sid, log):
    h.controller.start_search(sid)
    candidate = log.wait_type("candidate_found")["payload"]
    h.controller.accept_candidate(sid, candidate["candidate_id"])
    return candidate


def _single_target(harness, *, bearing=25.0, distance=1.5, vision_cls=SimVision, **cfg_overrides):
    cfg = fast_config(**cfg_overrides)
    world = make_world(cfg)
    add_bottle(world, bearing=bearing, distance=distance, is_target=True)
    return harness(cfg, world, vision=vision_cls(world))


def test_centre_then_approach_slow_zone_and_arrival(harness):
    h = _single_target(harness, bearing=25.0)           # visible from heading 0, well right of centre
    sid, log = h.search("my bottle")
    start_distance = h.world.distance_to("target_bottle")

    candidate = _accept_first_candidate(h, sid, log)
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("arrived", "proximity_confirmed")
    assert end["payload"]["message"] == "I'm next to your bottle"
    assert end.get("terminal") is True                  # arrival is the terminal movement event
    assert end["payload"]["active"] is False

    moves = h.drive.moves
    # 1) The target was right of centre, so the first motion is a turn toward it ...
    assert candidate["box"][1] > 500
    assert moves[0].command == "turn_right" and moves[0].speed == h.cfg.turn_speed
    assert h.cfg.turn_pulse_min_seconds <= moves[0].ttl - h.cfg.pulse_watchdog_margin_seconds <= h.cfg.turn_pulse_max_seconds
    # 2) ... then forward pulses, full speed first, slow once inside the slow zone, never fast again.
    forward = [m for m in moves if m.command == "forward"]
    speeds = [m.speed for m in forward]
    assert speeds[0] == h.cfg.forward_speed and speeds[-1] == h.cfg.slow_forward_speed
    first_slow = speeds.index(h.cfg.slow_forward_speed)
    assert all(s == h.cfg.forward_speed for s in speeds[:first_slow])
    assert all(s == h.cfg.slow_forward_speed for s in speeds[first_slow:])
    slow_ttl = h.cfg.slow_forward_pulse_seconds + h.cfg.pulse_watchdog_margin_seconds
    assert all(m.ttl == pytest.approx(slow_ttl) for m in forward[first_slow:])
    # 3) Every pulse is followed by a stop before the next one (move, stop, look, move...).
    records = h.drive.records
    for index, record in enumerate(records[:-1]):
        if record.command != "stop":
            assert records[index + 1].command == "stop"
    assert records[-1].command == "stop" and not h.world.is_moving()

    phases, messages = log.phases(), log.messages()
    assert "centering" in phases and "approaching" in phases
    assert "Almost there — slowing down" in messages
    assert phases.index("centering") < phases.index("approaching") < phases.index("arrived")

    # The simulated rover really did close in, and stopped short of the object.
    assert 0.3 < h.world.distance_to("target_bottle") < 0.7 < start_distance
    assert abs(h.world.relative_bearing("target_bottle")) < 10.0
    assert end["payload"]["target_width"] > 0.1 and end["payload"]["approach_pulses"] == len(moves)

    # Arrival does not touch the recognition lifecycle: FOUND since acceptance.
    assert search_status(sid) == SearchStatus.FOUND and len(finds_of(sid)) == 1
    assert h.tracking.value is None                     # cleared on every resting outcome
    assert h.controller.active_search_id() is None


def test_target_left_of_centre_turns_left(harness):
    h = _single_target(harness, bearing=-25.0)
    sid, log = h.search("my bottle")
    candidate = _accept_first_candidate(h, sid, log)
    log.wait_phase("approaching")
    assert candidate["box"][3] < 500
    assert h.drive.moves[0].command == "turn_left"


def test_arrival_needs_repeated_stationary_confirmations(harness):
    h = _single_target(harness, bearing=0.0, distance=0.9, arrival_confirmations=3)
    sid, log = h.search("my bottle")
    _accept_first_candidate(h, sid, log)
    end = log.wait_rest()
    assert end["payload"]["phase"] == "arrived"

    last_move = h.drive.moves[-1].at
    after = [call for call in h.vision.calls if call.at > last_move]
    # Three different frames were inspected after the wheels last moved ...
    assert len({call.frame_seq for call in after}) >= 3
    # ... and the final word came from OMNI, not from a YOLO box.
    assert after[-1].kind == "identify"
    assert all(record.command == "stop" for record in h.drive.records if record.at > last_move)


def test_found_alone_never_means_arrived(harness):
    """OMNI keeps saying "found", with a small box that never grows: the rover
    must conclude it is making no progress -- never that it arrived."""
    small = [450, 470, 550, 530]
    cfg = fast_config(stall_pulses=3)
    h = harness(cfg, vision=ScriptedVision(lambda index: {"box": small}))
    sid, log = h.search("my bottle")
    _accept_first_candidate(h, sid, log)
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("stopped", "no_progress")
    assert end["payload"]["message"] == "I'm not getting any closer — stopping"
    assert "arrived" not in log.phases()
    assert [m.command for m in h.drive.moves] == ["forward"] * 3     # exactly stall_pulses, then stop
    assert h.drive.records[-1].command == "stop"


def test_one_big_box_is_not_arrival(harness):
    """A single arrival-sized observation that does not repeat is discarded:
    the rover double-checks from a standstill, sees the evidence is gone, and
    carries on (here: into the no-progress stop) instead of claiming arrival."""
    usual = [350, 450, 650, 550]      # 0.10 x 0.30: not arrival-sized for a bottle
    flicker = [275, 415, 725, 585]    # 0.17 x 0.45: wide enough to look like arrival, once
    h = harness(fast_config(), vision=ScriptedVision(lambda index: {"box": flicker if index == 2 else usual}))
    sid, log = h.search("my bottle")
    _accept_first_candidate(h, sid, log)      # call 1 = candidate, call 2 = reacquire (the flicker)
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("stopped", "no_progress")
    assert "I think I'm close — double-checking" in log.messages()    # the evidence WAS examined ...
    assert "arrived" not in log.phases()                                # ... and did not hold


def test_wheels_stall_stops_with_no_progress(harness):
    h = _single_target(harness, bearing=0.0, stall_pulses=3)
    h.world.stuck = True                                 # wheels spin, the rover does not advance
    sid, log = h.search("my bottle")
    _accept_first_candidate(h, sid, log)
    end = log.wait_rest()
    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("stopped", "no_progress")
    assert [m.command for m in h.drive.moves] == ["forward"] * 3
    assert search_status(sid) == SearchStatus.FOUND      # still resumable


# --- progress is judged against a ratcheting baseline, not the previous frame (C02 / C03) ----


def _stuck_approach(harness, *, distance=1.5, **vision_fields):
    """Accept the bottle dead ahead, with the wheels spinning on the spot from
    then on. Returns (harness, resting payload, number of forward pulses)."""
    cfg = fast_config(stall_pulses=3, approach_max_pulses=40)
    world = make_world(cfg)
    add_bottle(world, bearing=0.0, distance=distance, is_target=True)
    h = harness(cfg, world, vision=SimVision(world, **vision_fields))
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    candidate = log.wait_type("candidate_found")["payload"]
    h.world.stuck = True                                  # e.g. pushed up against a chair leg
    h.controller.accept_candidate(sid, candidate["candidate_id"])
    end = log.wait_rest(timeout=30)["payload"]
    assert h.world.distance_to("target_bottle") == pytest.approx(distance)     # it really never advanced
    return h, end, [m.command for m in h.drive.moves].count("forward")


@pytest.mark.parametrize("jitter", [1, 2, 3])
def test_box_jitter_cannot_hide_a_stalled_rover(harness, jitter):
    """A box that wobbles by a unit or two "grows" by 3% every other frame.
    Compared with the previous frame that reset the stall counter for ever
    (40 pulses into an obstacle); against a baseline it stops within a few."""
    h, end, forward = _stuck_approach(harness, box_jitter=jitter)
    assert (end["phase"], end["reason"]) == ("stopped", "no_progress")
    assert forward <= h.cfg.stall_pulses + 2
    assert "arrived" not in [m.command for m in h.drive.moves]


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_random_box_jitter_cannot_hide_a_stalled_rover(harness, seed):
    # +/-5 units of noise can only ratchet the baseline up ~10 times before
    # nothing above it is left to "grow" into.
    h, end, forward = _stuck_approach(harness, box_jitter=5, box_jitter_seed=seed)
    assert (end["phase"], end["reason"]) == ("stopped", "no_progress")
    assert forward <= 14 < h.cfg.approach_max_pulses


@pytest.mark.parametrize("scale", [1.05, 0.95])
def test_yolo_and_omni_disagreeing_about_the_box_cannot_hide_a_stall(harness, scale):
    """No random noise at all: OMNI's box is simply a little looser / tighter
    than YOLO's, so every revalidation looked like a shrink followed by growth."""
    h, end, forward = _stuck_approach(harness, identify_box_scale=scale)
    assert (end["phase"], end["reason"]) == ("stopped", "no_progress")
    assert forward <= h.cfg.stall_pulses + 2


@pytest.mark.parametrize(
    "distance, metres_per_pulse",
    [
        (4.0, 0.10),    # far target: 2.5% growth per pulse
        (5.0, 0.10),    # 2.0% per pulse
        (2.0, 0.04),    # slow rover: 2% per pulse, 0.016 m slow-zone pulses (~2.2% each)
        (1.0, 0.04),    # the same rover starting just outside the slow zone
    ],
)
def test_slow_but_real_progress_is_never_no_progress(harness, distance, metres_per_pulse):
    """Growth per pulse is ~ step / distance, so a healthy approach to a far
    target (or by a slow rover, or inside the slow zone) grows the box by less
    than min_progress_fraction per pulse. That is progress, not a stall."""
    cfg = fast_config(approach_max_pulses=200, approach_max_seconds=60.0)
    # The near-target case drops below the threshold in the slow zone, not
    # necessarily on its very first full-speed pulse.
    slow_step = metres_per_pulse * (cfg.slow_forward_speed / cfg.forward_speed) * (cfg.slow_forward_pulse_seconds / cfg.forward_pulse_seconds)
    assert slow_step / (distance - slow_step) < cfg.min_progress_fraction
    world = make_world(cfg, metres_per_pulse=metres_per_pulse)
    add_bottle(world, bearing=0.0, distance=distance, is_target=True)
    h = harness(cfg, world)
    sid, log = h.search("my bottle")
    _accept_first_candidate(h, sid, log)
    end = log.wait_rest(timeout=60)

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("arrived", "proximity_confirmed")
    assert h.world.distance_to("target_bottle") < 0.7


def test_a_shrinking_box_never_lowers_the_progress_baseline(harness):
    """Boxes that shrink and then merely recover are not progress."""
    sizes = {1: 100, 2: 100, 3: 90, 4: 100, 5: 92, 6: 101}          # half-widths; never 3% above 100
    def answer(index):
        half = sizes.get(index, 100)
        return {"box": [500 - half, 500 - half // 2, 500 + half, 500 + half // 2]}

    h = harness(fast_config(stall_pulses=3), vision=ScriptedVision(answer))
    sid, log = h.search("my bottle")
    _accept_first_candidate(h, sid, log)
    end = log.wait_rest()
    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("stopped", "no_progress")
    assert [m.command for m in h.drive.moves] == ["forward"] * 3


def test_turn_jitter_cannot_hide_a_rover_that_does_not_rotate(harness):
    """The turn half of the same defect: the offset wobbles, the rover does not
    turn. Previously every other pulse counted as "more centred"."""
    def answer(index):
        shift = 0 if index % 2 else 12                                # offset 0.120 / 0.132, for ever
        return {"box": [350, 570 + shift, 650, 670 + shift]}

    h = harness(fast_config(stall_pulses=3), vision=ScriptedVision(answer))
    sid, log = h.search("my bottle")
    _accept_first_candidate(h, sid, log)
    end = log.wait_rest()
    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("stopped", "no_progress")
    assert 3 <= len(h.drive.moves) <= 5 and {m.command for m in h.drive.moves} == {"turn_right"}


@pytest.mark.parametrize("bearing, command", [(12.0, "turn_right"), (25.0, "turn_right"), (-25.0, "turn_left")])
def test_a_turn_that_does_not_rotate_the_rover_is_no_progress_not_a_mix_up(harness, bearing, command):
    """L01: wheels spin, the rover does not rotate. With a large offset the box
    fails the association with where it SHOULD be after the turn -- but it is
    exactly where it was before. That is a stalled turn, with one object in
    view, not "I'm not sure which one is your bottle"."""
    h = _single_target(harness, bearing=bearing, stall_pulses=3)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    candidate = log.wait_type("candidate_found")["payload"]
    h.world.turn_stuck = True
    h.controller.accept_candidate(sid, candidate["candidate_id"])
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("stopped", "no_progress")
    assert [m.command for m in h.drive.moves] == [command] * 3
    assert h.world.pose()[2] == pytest.approx(0.0)


def test_a_box_that_matches_neither_prediction_nor_the_old_view_is_still_a_mix_up(harness):
    """L01 must not weaken the identity gate: after a turn, a box that is
    neither where the target should be now nor where it was before is another
    object."""
    before = [350, 800, 650, 900]                                     # offset +0.35: a big turn is commanded
    elsewhere = [350, 80, 650, 180]                                   # far from both the prediction and `before`
    vision = ScriptedVision(lambda index: {"box": before if index <= 2 else elsewhere})
    h = harness(fast_config(), vision=vision)
    sid, log = h.search("my bottle")
    _accept_first_candidate(h, sid, log)
    end = log.wait_rest()
    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("target_lost", "identity_uncertain")
    assert [m.command for m in h.drive.moves] == ["turn_right"]


def test_approach_pulse_budget(harness):
    h = _single_target(harness, bearing=0.0, distance=2.5, approach_max_pulses=2)
    sid, log = h.search("my bottle")
    _accept_first_candidate(h, sid, log)
    end = log.wait_rest()
    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("stopped", "approach_budget")
    assert len(h.drive.moves) == 2 and h.drive.records[-1].command == "stop"


def test_candidate_without_box_may_be_asked_but_never_moves(harness):
    h = _single_target(harness, bearing=0.0)
    h.vision.found_without_box = True
    sid, log = h.search("my bottle")

    h.controller.start_search(sid)
    candidate = log.wait_type("candidate_found")["payload"]
    assert candidate["box"] is None
    assert candidate["crop_url"] == candidate["image_url"]          # crop falls back to the full frame
    [row] = candidates_of(sid)
    assert row.bbox_json == "null"

    h.controller.accept_candidate(sid, candidate["candidate_id"])
    end = log.wait_rest()
    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("target_lost", "no_box")
    assert h.drive.moves == []                                      # ZERO motor commands, ever
    assert search_status(sid) == SearchStatus.FOUND
    # No crop of a box-less candidate is sent as the "confirmed target".
    assert all(request.confirmed_crop is None for request in h.vision.requests)


def test_boxless_acceptance_never_moves_even_when_a_box_shows_up_afterwards(harness):
    """C01: the user said yes to a full frame, not to an object. A box that
    OMNI produces afterwards (here: for a DIFFERENT object) has no tie to what
    was accepted, so it must not authorise a single pulse -- or even a look."""
    other_object = [350, 700, 650, 800]
    vision = ScriptedVision(lambda index: {"box": None if index == 1 else other_object})
    h = harness(fast_config(), vision=vision)
    sid, log = h.search("my bottle")
    candidate = _accept_first_candidate(h, sid, log)
    assert candidate["box"] is None
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("target_lost", "no_box")
    assert end["payload"]["message"] == (
        "I think I can see your bottle, but I can't tell where it is, so I'm staying put"
    )
    assert h.controller.join_missions(2.0)
    assert h.drive.moves == []                                      # zero drive calls
    assert vision.identify_calls == 1                               # ended before any identify call
    assert all(record.command == "stop" for record in h.drive.records)
    assert search_status(sid) == SearchStatus.FOUND


class _BoxlessFirst(SimVision):
    """First answer (the candidate) describes the user's bottle without a box;
    later answers are "the single strongest match", with a box -- the decoy."""

    def identify(self, packet, request):
        first = self.identify_calls == 0
        self.found_without_box = first
        self.world.get("decoy_bottle").matches_query = not first
        return super().identify(packet, request)


def test_boxless_acceptance_is_never_an_approach_to_the_strongest_lookalike(harness):
    """The reviewers' scenario: two bottles in view, box-less question accepted,
    OMNI then localises the bigger decoy. The rover used to drive to the decoy
    and announce it had arrived."""
    cfg = fast_config()
    world = make_world(cfg)
    add_bottle(world, "users_bottle", bearing=-12, distance=2.2, is_target=True, description="a blue bottle")
    add_bottle(world, "decoy_bottle", bearing=12, distance=1.4, description="a green bottle")
    h = harness(cfg, world, vision=_BoxlessFirst(world))
    sid, log = h.search("my bottle")
    candidate = _accept_first_candidate(h, sid, log)
    assert (candidate["description"], candidate["box"]) == ("a blue bottle", None)
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("target_lost", "no_box")
    assert h.drive.moves == [] and "arrived" not in log.phases()
    assert h.world.distance_to("decoy_bottle") == pytest.approx(1.4)


def test_a_found_search_with_a_boxless_acceptance_cannot_be_resumed(harness):
    """Resume would only loop back into the same dead end -- or, worse, start
    moving once OMNI does return a box. It is refused, and the API's
    ``can_resume`` flag says so."""
    from app.rover.types import InvalidTransitionError

    h = _single_target(harness, bearing=0.0)
    h.vision.found_without_box = True
    sid, log = h.search("my bottle")
    _accept_first_candidate(h, sid, log)
    assert log.wait_rest()["payload"]["reason"] == "no_box"
    assert h.controller.join_missions(2.0)

    h.vision.found_without_box = False                              # OMNI localises from now on
    assert h.controller.can_resume(sid) is False
    records = len(h.drive.records)
    with pytest.raises(InvalidTransitionError):
        h.controller.resume(sid)
    assert h.controller.active_search_id() is None
    assert len(h.drive.records) == records and h.drive.moves == []

    # The same after a restart (nothing but the database to go by).
    reborn = h.new_controller()
    try:
        assert reborn.can_resume(sid) is False
        with pytest.raises(InvalidTransitionError):
            reborn.resume(sid)
        assert h.drive.moves == []
    finally:
        reborn.shutdown()


class _VanishingVision(SimVision):
    """Removes the target from the world when the Nth identify call starts."""

    vanish_on_call = 0
    vanished_at = None

    def identify(self, packet, request):
        if self.identify_calls + 1 == self.vanish_on_call:
            self.world.remove_object("target_bottle")
            self.vanished_at = packet.captured_at
        return super().identify(packet, request)


def test_target_loss_stops_and_stays_put(harness):
    h = _single_target(harness, bearing=0.0, vision_cls=_VanishingVision, omni_revalidate_every_pulses=2)
    h.vision.vanish_on_call = 3          # 1 = candidate, 2 = reacquire, 3 = first revalidation
    sid, log = h.search("my bottle")
    _accept_first_candidate(h, sid, log)
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("target_lost", "not_visible")
    assert h.vision.vanished_at is not None
    assert [m for m in h.drive.moves if m.at > h.vision.vanished_at] == []     # looked again, never moved
    assert h.vision.identify_calls == 2 + h.cfg.reacquire_attempts             # stationary re-observations
    assert h.tracking.value is None


def test_yolo_miss_alone_is_never_target_lost(harness):
    """The local detector goes blind during the approach; OMNI still sees the
    target, so the rover carries on and arrives."""
    h = _single_target(harness, bearing=0.0)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    candidate = log.wait_type("candidate_found")["payload"]
    h.vision.screen_blind = True
    h.controller.accept_candidate(sid, candidate["candidate_id"])
    end = log.wait_rest()
    assert end["payload"]["phase"] == "arrived"
    assert "target_lost" not in log.phases()


def test_ambiguous_association_falls_back_to_omni(harness):
    """Two similar bottles side by side: YOLO boxes cannot say which one the
    user accepted, so every observation is revalidated by OMNI."""
    cfg = fast_config(omni_revalidate_every_pulses=50)   # periodic revalidation out of the picture
    world = make_world(cfg)
    add_bottle(world, "target_bottle", bearing=0, distance=1.5, is_target=True)
    add_bottle(world, "twin_bottle", bearing=6, distance=1.5, width=0.075, height=0.24)
    h = harness(cfg, world)
    sid, log = h.search("my bottle")
    candidate = _accept_first_candidate(h, sid, log)
    assert candidate["description"] == "target_bottle (simulated)"
    end = log.wait_rest()
    assert end["payload"]["phase"] == "arrived"
    assert h.world.distance_to("target_bottle") < h.world.distance_to("twin_bottle")

    moves = h.drive.moves
    identify_times = [call.at for call in h.vision.calls if call.kind == "identify"]
    for first, second in zip(moves[:3], moves[1:4]):
        assert any(first.at < at < second.at for at in identify_times), "observation was not revalidated by OMNI"


def test_unambiguous_track_uses_yolo_between_revalidations(harness):
    h = _single_target(harness, bearing=0.0, omni_revalidate_every_pulses=50)
    sid, log = h.search("my bottle")
    _accept_first_candidate(h, sid, log)
    assert log.wait_rest()["payload"]["phase"] == "arrived"
    moves = h.drive.moves
    identify_times = [call.at for call in h.vision.calls if call.kind == "identify"]
    # Between the early pulses nobody asked OMNI: a unique YOLO box carried the track.
    assert not any(moves[0].at < at < moves[2].at for at in identify_times)


def test_identity_mismatch_stops_immediately(harness):
    """OMNI suddenly reports a different object (far from where the target must
    be): identity is uncertain, so the rover stops rather than chase it."""
    cfg = fast_config(omni_revalidate_every_pulses=2)
    world = make_world(cfg)
    add_bottle(world, "target_bottle", bearing=0, distance=1.5, is_target=True)
    add_bottle(world, "other_bottle", bearing=25, distance=1.8)
    h = harness(cfg, world)
    h.vision.wrong_object_after = 1      # reacquire is honest, the next revalidation is not
    sid, log = h.search("my bottle")
    candidate = _accept_first_candidate(h, sid, log)
    assert candidate["description"] == "target_bottle (simulated)"
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("target_lost", "identity_uncertain")
    assert len(h.drive.moves) == 2                       # the two pulses before the revalidation, none after
    assert h.drive.records[-1].command == "stop"
    assert h.world.distance_to("target_bottle") > 1.0    # nowhere near "arrived"


def test_two_bottles_full_mission(harness):
    cfg = fast_config()
    world = build_scenario("two_bottles", make_world(cfg))
    h = harness(cfg, world)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    decoy = log.wait_type("candidate_found")["payload"]
    h.controller.reject_candidate(sid, decoy["candidate_id"])
    target = log.wait_type("candidate_found")["payload"]
    h.controller.accept_candidate(sid, target["candidate_id"])
    end = log.wait_rest()

    assert end["payload"]["phase"] == "arrived"
    assert h.world.distance_to("target_bottle") < 0.7 < h.world.distance_to("decoy_bottle")
    # The accepted crop travels with every approach-time identification.
    approach_requests = [r for r in h.vision.requests if r.confirmed_crop is not None]
    assert approach_requests and all(r.confirmed_crop[:2] == b"\xff\xd8" for r in approach_requests)
