"""Races around the generation token: cancel during inference, cancel during
a pulse, replacement while the old thread is still unwinding, decisions with
no live mission, overrides and shutdown. Everything is SIMULATED; every wait
is on an event or a queue with a timeout."""

import threading
import time

import pytest

from app.config import settings
from app.models import CandidateDecision, SearchStatus
from app.rover.mission import SCAN, Mission
from app.rover.simulation import SimVision, build_scenario
from app.rover.store import RoverStore
from app.rover.types import (
    InvalidTransitionError,
    MissionCancelled,
    MovementPhase,
    RoverBusyError,
    RoverError,
    SearchNotFoundError,
    StopReason,
)
from tests.rover.helpers import (
    add_bottle,
    candidates_of,
    fast_config,
    finds_of,
    make_search,
    make_world,
    movement_row,
    search_status,
)


def _visible_target(harness, *, vision_cls=SimVision, bearing=0.0, **cfg_overrides):
    cfg = fast_config(**cfg_overrides)
    world = make_world(cfg)
    add_bottle(world, bearing=bearing, distance=1.5, is_target=True)
    return harness(cfg, world, vision=vision_cls(world))


def _search_files(media_root, search_id):
    folder = media_root / "searches" / search_id
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


# --- cancellation during inference ---------------------------------------------------


def test_cancel_during_inference_returns_fast_and_drops_the_late_result(harness, media_root):
    h = _visible_target(harness)
    h.vision.block_event = threading.Event()             # OMNI "hangs" with a positive answer pending
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    assert h.vision.entered_event.wait(5), "inference never started"

    before = time.monotonic()
    assert h.controller.cancel_search(sid) is True
    took = time.monotonic() - before
    cancelled_at = time.monotonic()

    assert took < 0.5                                     # did not wait for the hanging call
    assert search_status(sid) == SearchStatus.CANCELLED
    state = h.controller.state_for(sid)
    assert (state.phase, state.reason, state.active) == (MovementPhase.STOPPED, "cancelled", False)
    assert h.controller.active_search_id() is None

    # The rover halts first, then the terminal event goes out.
    stopped = log.wait_phase("stopped")
    terminal = log.wait_type("search_cancelled")
    assert stopped["payload"]["reason"] == "cancelled"
    assert terminal == {"type": "search_cancelled", "payload": {"status": "CANCELLED", "reason": "cancelled"}, "terminal": True}

    # Now the "found, high" answer finally arrives -- from a dead generation.
    h.vision.block_event.set()
    assert h.controller.join_missions(5)
    assert candidates_of(sid) == []
    assert search_status(sid) == SearchStatus.CANCELLED   # not flipped back to CANDIDATE_PENDING
    assert log.of_type("candidate_found") == []
    assert _search_files(media_root, sid) == []
    assert [r for r in h.drive.records if r.at > cancelled_at] == []   # not one command after the cancel
    assert h.drive.moves == []


class _CancelsWhileAnswering(SimVision):
    """The cancel lands at the exact moment the positive result is produced."""

    controller = None
    search_id = None

    def identify(self, packet, request):
        result = super().identify(packet, request)
        self.controller.cancel_search(self.search_id)
        return result


def test_result_that_arrives_with_the_cancel_creates_nothing(harness, media_root):
    h = _visible_target(harness, vision_cls=_CancelsWhileAnswering)
    sid, log = h.search("my bottle")
    h.vision.controller, h.vision.search_id = h.controller, sid
    h.controller.start_search(sid)
    log.wait_type("search_cancelled")
    assert h.controller.join_missions(5)

    assert candidates_of(sid) == [] and log.of_type("candidate_found") == []
    assert search_status(sid) == SearchStatus.CANCELLED
    assert _search_files(media_root, sid) == []
    assert h.drive.moves == []


def test_stale_mission_cannot_create_a_candidate(harness, media_root):
    """White-box: drive the candidate path directly with a mission whose
    generation is no longer current. Nothing may be written, files included."""
    h = _visible_target(harness)
    sid, _log = h.search("my bottle")
    stale = Mission(search_id=sid, generation=h.controller._generation - 1, kind=SCAN, target_text="my bottle")
    packet = h.camera.latest_packet()
    result = h.vision.identify(packet, _request())

    with pytest.raises(MissionCancelled):
        h.controller._present_candidate(stale, packet, result, result.box)

    assert candidates_of(sid) == [] and search_status(sid) == SearchStatus.SEARCHING
    assert _search_files(media_root, sid) == []


def test_store_refuses_a_candidate_for_a_search_that_is_not_searching():
    store = RoverStore()
    sid = make_search("my bottle", status=SearchStatus.CANCELLED)
    created = store.create_candidate(sid, full_image_path="a.jpg", crop_path="b.jpg", box=None, description="x")
    assert created is None
    assert candidates_of(sid) == [] and search_status(sid) == SearchStatus.CANCELLED


def _request():
    from app.rover.types import IdentifyRequest

    return IdentifyRequest(target_text="my bottle")


# --- cancellation during a motor pulse -------------------------------------------------


def test_cancel_during_a_pulse_stops_the_wheels_promptly(harness):
    cfg = fast_config(scan_pulse_seconds=1.5)             # a pulse long enough to be caught in the act
    h = harness(cfg)                                      # empty world: goes straight to scanning
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    assert h.drive.wait_for_moves(1, timeout=5)
    assert h.world.is_moving()

    before = time.monotonic()
    h.controller.cancel_search(sid)
    after = time.monotonic()

    last = h.drive.records[-1]
    assert last.command == "stop" and before <= last.at <= after
    assert after - before < 0.5
    assert not h.world.is_moving()
    # The mission thread unwinds at once instead of finishing its 1.5 s pulse ...
    assert h.controller.join_missions(1.0)
    # ... and neither it nor anything else touched the motors again.
    assert h.drive.records[-1] is last
    [pulse] = h.drive.moves
    assert pulse.ttl == pytest.approx(cfg.scan_pulse_seconds + cfg.pulse_watchdog_margin_seconds)
    assert 0 < h.world.pose()[2] < 1500.0 * 0.9           # turned only for as long as it actually ran
    assert search_status(sid) == SearchStatus.CANCELLED


def test_every_pulse_of_a_whole_mission_arms_the_watchdog(harness):
    cfg = fast_config()
    h = harness(cfg, build_scenario("two_bottles", make_world(cfg)))
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    h.controller.reject_candidate(sid, log.wait_type("candidate_found")["payload"]["candidate_id"])
    h.controller.accept_candidate(sid, log.wait_type("candidate_found")["payload"]["candidate_id"])
    assert log.wait_rest()["payload"]["phase"] == "arrived"

    moves = h.drive.moves
    assert {m.command for m in moves} == {"turn_right", "forward"}
    longest = max(cfg.scan_pulse_seconds, cfg.turn_pulse_max_seconds, cfg.forward_pulse_seconds)
    for move in moves:
        assert move.ttl is not None
        assert cfg.pulse_watchdog_margin_seconds < move.ttl <= longest + cfg.pulse_watchdog_margin_seconds + 1e-9
    assert h.world.watchdog_trips == 0                    # every pulse was stopped by the mission itself


# --- replacement and busy ----------------------------------------------------------------


class _FirstScreenBlocks(SimVision):
    """The first screen() call (mission A, on its own thread) hangs until
    released -- long after A has been replaced."""

    def __init__(self, world):
        super().__init__(world)
        self.first_entered = threading.Event()
        self.release_first = threading.Event()
        self._first_taken = False

    def release(self):
        super().release()
        self.release_first.set()

    def screen(self, packet, coco_class):
        with self._lock:
            first, self._first_taken = not self._first_taken, True
        if first:
            self.first_entered.set()
            self.release_first.wait()
        return super().screen(packet, coco_class)


def test_replacement_and_stale_mission_cleanup(harness):
    h = _visible_target(harness, vision_cls=_FirstScreenBlocks)
    sid_a, log_a = h.search("my bottle")
    sid_b, log_b = h.search("my other bottle")

    h.controller.start_search(sid_a)
    assert h.vision.first_entered.wait(5)
    thread_a = h.controller._mission.thread

    state_b = h.controller.start_search(sid_b, replace=True)
    assert state_b.search_id == sid_b and state_b.active

    # A was cancelled exactly like cancel_search, with reason "replaced".
    assert search_status(sid_a) == SearchStatus.CANCELLED
    assert log_a.wait_phase("stopped")["payload"]["reason"] == "replaced"
    assert log_a.wait_type("search_cancelled")["payload"] == {"status": "CANCELLED", "reason": "replaced"}
    assert h.controller.state_for(sid_a).reason == "replaced"          # from the RoverMovement table

    # B gets all the way to its question while A's thread is still stuck.
    candidate_b = log_b.wait_type("candidate_found")["payload"]
    log_b.wait_phase("waiting_for_confirmation")
    assert thread_a.is_alive()
    assert h.tracking.value == candidate_b["box"]
    records_before = len(h.drive.records)
    tracking_before = len(h.tracking.history)

    # Now let A's thread run into its cleanup.
    h.vision.release_first.set()
    thread_a.join(5)
    assert not thread_a.is_alive()

    assert h.tracking.value == candidate_b["box"]                       # B's tracking box survived
    assert len(h.tracking.history) == tracking_before                   # A never touched it
    assert len(h.drive.records) == records_before                       # A issued no stop, no command
    assert h.controller.active_search_id() == sid_b                     # B is still registered
    assert h.controller.state_for(sid_b).phase == MovementPhase.WAITING_FOR_CONFIRMATION
    assert candidates_of(sid_a) == [] and log_a.of_type("candidate_found") == []
    assert search_status(sid_b) == SearchStatus.CANDIDATE_PENDING

    # And B is fully functional afterwards.
    h.controller.accept_candidate(sid_b, candidate_b["candidate_id"])
    assert log_b.wait_rest()["payload"]["phase"] == "arrived"


def test_second_search_is_refused_while_busy(harness):
    h = _visible_target(harness)
    sid_a, log_a = h.search("my bottle")
    sid_b, _log_b = h.search("my other bottle")
    h.controller.start_search(sid_a)
    log_a.wait_phase("waiting_for_confirmation")          # "active" includes waiting for an answer
    generation = h.controller.snapshot()["generation"]

    with pytest.raises(RoverBusyError) as busy:
        h.controller.start_search(sid_b)
    assert busy.value.active_search_id == sid_a and busy.value.active_target_text == "my bottle"
    with pytest.raises(RoverBusyError):
        h.controller.resume(sid_b)

    # Nothing about A changed, nothing about B started.
    snapshot = h.controller.snapshot()
    assert snapshot["active_search_id"] == sid_a and snapshot["generation"] == generation
    assert snapshot["movement"]["phase"] == "waiting_for_confirmation"
    assert search_status(sid_a) == SearchStatus.CANDIDATE_PENDING
    assert search_status(sid_b) == SearchStatus.SEARCHING and movement_row(sid_b) is None
    assert h.controller.state_for(sid_b).phase == MovementPhase.IDLE


def test_starting_the_active_search_again_is_a_no_op(harness):
    h = _visible_target(harness)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    log.wait_phase("waiting_for_confirmation")
    generation = h.controller.snapshot()["generation"]
    thread = h.controller._mission.thread

    again = h.controller.start_search(sid)
    assert again.phase == MovementPhase.WAITING_FOR_CONFIRMATION
    assert h.controller.snapshot()["generation"] == generation
    assert h.controller._mission.thread is thread
    assert h.controller.resume(sid).phase == MovementPhase.WAITING_FOR_CONFIRMATION


def test_start_search_validates_the_search(harness):
    h = harness(fast_config())
    with pytest.raises(SearchNotFoundError):
        h.controller.start_search("no-such-search")
    cancelled = make_search("my bottle", status=SearchStatus.CANCELLED)
    with pytest.raises(InvalidTransitionError):
        h.controller.start_search(cancelled)
    assert h.drive.records == [] and h.controller.active_search_id() is None


# --- stop, manual override, resume -----------------------------------------------------------


def test_estop_halts_leaves_the_search_resumable_and_resume_continues(harness):
    cfg = fast_config(scan_pulse_seconds=1.5)
    world = make_world(cfg)
    h = harness(cfg, world)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    assert h.drive.wait_for_moves(1, timeout=5)

    state = h.controller.stop()
    assert (state.phase, state.reason, state.active) == (MovementPhase.STOPPED, "user_stop", False)
    assert h.drive.records[-1].command == "stop" and not h.world.is_moving()
    assert search_status(sid) == SearchStatus.SEARCHING            # untouched
    assert h.controller.join_missions(1.0)

    resumed = h.controller.resume(sid)
    assert resumed.phase == MovementPhase.SCANNING and resumed.active
    assert h.drive.wait_for_moves(2, timeout=5)                     # it is scanning again
    h.controller.stop()
    # A stop with nothing running still stops the wheels and changes nothing else.
    count = len(h.drive.records)
    idle = h.controller.stop()
    assert idle.phase == MovementPhase.STOPPED and len(h.drive.records) == count + 1


def test_manual_override_invalidates_and_never_cuts_the_manual_command(harness):
    cfg = fast_config(scan_pulse_seconds=1.5)
    h = harness(cfg)
    sid, log = h.search("my bottle")

    h.controller.manual_override()                                  # idle: a no-op
    assert h.drive.records == []

    h.controller.start_search(sid)
    assert h.drive.wait_for_moves(1, timeout=5)

    h.controller.manual_override()
    state = h.controller.state_for(sid)
    assert (state.phase, state.reason) == (MovementPhase.STOPPED, "manual_override")
    assert h.drive.records[-1].command == "stop"
    assert search_status(sid) == SearchStatus.SEARCHING
    assert log.wait_phase("stopped")["payload"]["message"] == "Manual control took over — search paused"

    # The human now drives. The dying mission thread wakes up inside its
    # pulse; it must NOT send the stop it would normally end the pulse with.
    h.drive.drive("forward", 0.5, ttl=2.0)
    assert h.controller.join_missions(1.0)
    assert h.drive.records[-1].command == "forward"
    assert h.world.is_moving()
    h.drive.stop()


def test_decisions_are_refused_once_cancelled_or_decided(harness):
    h = _visible_target(harness)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    candidate_id = log.wait_type("candidate_found")["payload"]["candidate_id"]

    with pytest.raises(SearchNotFoundError):
        h.controller.accept_candidate(sid, "no-such-candidate")
    with pytest.raises(SearchNotFoundError):
        h.controller.accept_candidate("no-such-search", candidate_id)

    h.controller.cancel_search(sid)
    moves = len(h.drive.moves)
    with pytest.raises(InvalidTransitionError):                     # accept after cancel
        h.controller.accept_candidate(sid, candidate_id)
    with pytest.raises(InvalidTransitionError):
        h.controller.reject_candidate(sid, candidate_id)
    with pytest.raises(InvalidTransitionError):
        h.controller.resume(sid)

    assert search_status(sid) == SearchStatus.CANCELLED and finds_of(sid) == []
    assert candidates_of(sid)[0].decision == CandidateDecision.PENDING   # "unchanged" on cancel
    assert h.controller.join_missions(2.0) and len(h.drive.moves) == moves
    assert log.of_type("candidate_accepted") == []


def test_double_accept_has_exactly_one_effect(harness):
    h = _visible_target(harness)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    candidate_id = log.wait_type("candidate_found")["payload"]["candidate_id"]

    h.controller.accept_candidate(sid, candidate_id)
    with pytest.raises(InvalidTransitionError):
        h.controller.accept_candidate(sid, candidate_id)
    with pytest.raises(InvalidTransitionError):
        h.controller.reject_candidate(sid, candidate_id)

    assert log.wait_rest()["payload"]["phase"] == "arrived"          # the mission was not disturbed
    assert len(finds_of(sid)) == 1 and len(log.of_type("candidate_accepted")) == 1
    assert search_status(sid) == SearchStatus.FOUND
    with pytest.raises(InvalidTransitionError):                       # already there
        h.controller.resume(sid)


def test_cancel_is_idempotent_and_unknown_searches_report_false(harness):
    h = _visible_target(harness)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    log.wait_phase("waiting_for_confirmation")

    assert h.controller.cancel_search(sid) is True
    assert h.controller.cancel_search(sid) is True
    assert h.controller.cancel_search("no-such-search") is False
    assert len(log.of_type("search_cancelled")) == 1
    assert search_status(sid) == SearchStatus.CANCELLED


def test_cancel_after_found_stops_the_approach(harness):
    h = _visible_target(harness, forward_pulse_seconds=1.5)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    h.controller.accept_candidate(sid, log.wait_type("candidate_found")["payload"]["candidate_id"])
    assert h.drive.wait_for_moves(1, timeout=5)                      # approaching, mid-pulse

    h.controller.cancel_search(sid)
    assert h.drive.records[-1].command == "stop" and not h.world.is_moving()
    assert search_status(sid) == SearchStatus.CANCELLED
    assert h.controller.join_missions(1.0)
    assert "arrived" not in log.phases()


# --- restart recovery --------------------------------------------------------------------------


def test_restart_reports_interrupted_and_nothing_moves_until_resume(harness):
    cfg = fast_config()
    h = harness(cfg, build_scenario("two_bottles", make_world(cfg)))
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    decoy = log.wait_type("candidate_found")["payload"]
    log.wait_phase("waiting_for_confirmation")
    assert movement_row(sid).phase == "waiting_for_confirmation"

    # "Restart": the old process is gone without a clean shutdown; a fresh
    # controller comes up on the same database.
    with h.controller._lock:
        h.controller._generation += 1                                 # the old thread is dead to the world
    reborn = h.new_controller()
    try:
        state = reborn.state_for(sid)
        assert (state.phase, state.reason, state.active) == (MovementPhase.STOPPED, "interrupted", False)
        assert reborn.active_search_id() is None
        assert reborn.state_for("never-seen").phase == MovementPhase.IDLE
        with pytest.raises(InvalidTransitionError):                   # the question must be answered first
            reborn.resume(sid)

        # The user answers the question that was on screen: recorded, but the
        # rover does not move on its own after a restart.
        commands = len(h.drive.records)
        after = reborn.reject_candidate(sid, decoy["candidate_id"])
        assert after.phase == MovementPhase.STOPPED and after.active is False
        assert search_status(sid) == SearchStatus.SEARCHING
        assert log.wait_type("candidate_rejected")["payload"]["candidate_id"] == decoy["candidate_id"]
        assert reborn.active_search_id() is None and len(h.drive.records) == commands

        # Explicit resume: scanning continues, and the rejection is remembered
        # across the restart (loaded from the database), so the decoy is not
        # offered again.
        assert reborn.resume(sid).phase == MovementPhase.SCANNING
        target = log.wait_type("candidate_found")["payload"]
        assert target["description"] == "a blue water bottle beside a backpack"
        assert reborn.state_for(sid).candidates_presented == 2

        # Same again for an accept that arrives with no live mission.
        reborn.stop()
        assert log.wait_phase("stopped")["payload"]["reason"] == "user_stop"
        accepted = reborn.accept_candidate(sid, target["candidate_id"])
        assert accepted.phase == MovementPhase.STOPPED and accepted.active is False
        assert search_status(sid) == SearchStatus.FOUND and len(finds_of(sid)) == 1
        moves = len(h.drive.moves)
        assert reborn.join_missions(2.0) and len(h.drive.moves) == moves

        resumed = reborn.resume(sid)                                  # FOUND + accepted -> approach mission
        assert resumed.phase == MovementPhase.CHECKING and resumed.active
        assert log.wait_rest()["payload"]["phase"] == "arrived"
        assert h.world.distance_to("target_bottle") < 0.7
    finally:
        reborn.shutdown()


# --- shutdown --------------------------------------------------------------------------------------


def test_shutdown_stops_everything_and_is_idempotent(harness):
    cfg = fast_config(scan_pulse_seconds=1.5)
    h = harness(cfg)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    assert h.drive.wait_for_moves(1, timeout=5)
    thread = h.controller._mission.thread

    before = time.monotonic()
    h.controller.shutdown()
    assert time.monotonic() - before < 2.0
    assert not thread.is_alive()                                       # joined
    assert h.drive.records[-1].command == "stop" and not h.world.is_moving()
    state = h.controller.state_for(sid)
    assert (state.phase, state.reason) == (MovementPhase.STOPPED, "shutdown")
    assert movement_row(sid).reason == "shutdown"
    assert search_status(sid) == SearchStatus.SEARCHING                # resumable after the restart

    h.controller.shutdown()                                            # idempotent
    assert h.controller.snapshot()["shutdown"] is True
    with pytest.raises(RoverError):
        h.controller.start_search(sid)
    with pytest.raises(RoverError):
        h.controller.resume(sid)
    assert len(h.drive.moves) == 1


def test_shutdown_does_not_wait_for_a_hanging_inference(harness):
    h = _visible_target(harness)
    h.vision.block_event = threading.Event()
    sid, _log = h.search("my bottle")
    h.controller.start_search(sid)
    assert h.vision.entered_event.wait(5)

    before = time.monotonic()
    h.controller.shutdown()
    assert time.monotonic() - before < 2.0
    assert h.controller.join_missions(0.1)
    assert h.controller.state_for(sid).reason == "shutdown"


def test_stop_reason_is_recorded(harness):
    h = _visible_target(harness)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    log.wait_phase("waiting_for_confirmation")
    h.controller.stop(StopReason.MANUAL_OVERRIDE)
    assert h.controller.state_for(sid).reason == "manual_override"
    assert settings.motor_driver == "sim"


def test_can_resume_tracks_what_resume_would_do(harness):
    h = _visible_target(harness)
    sid, log = h.search("my bottle")
    other = make_search("my other bottle")
    assert h.controller.can_resume(sid) is True                      # SEARCHING, nothing running
    assert h.controller.can_resume("no-such-search") is False

    h.controller.start_search(sid)
    candidate_id = log.wait_type("candidate_found")["payload"]["candidate_id"]
    log.wait_phase("waiting_for_confirmation")
    assert h.controller.can_resume(sid) is False                     # it is running
    assert h.controller.can_resume(other) is False                   # the rover is busy

    h.controller.stop()
    assert log.wait_phase("stopped")["payload"]["reason"] == "user_stop"
    assert h.controller.can_resume(sid) is False                     # CANDIDATE_PENDING: answer first
    assert h.controller.can_resume(other) is True
    h.controller.accept_candidate(sid, candidate_id)
    assert h.controller.can_resume(sid) is True                      # FOUND, accepted, not arrived

    h.controller.resume(sid)
    assert log.wait_rest()["payload"]["phase"] == "arrived"
    assert h.controller.can_resume(sid) is False                     # already there
    h.controller.cancel_search(other)
    assert h.controller.can_resume(other) is False                   # cancelled searches stay cancelled
