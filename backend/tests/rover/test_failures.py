"""Camera and inference failures, and the simulated-vision safety interlock.
In every case the required outcome is the same: wheels stopped."""

import threading

import pytest

from app.models import SearchStatus
from app.rover.mission import SCAN, Mission, MissionEnd
from app.rover.simulation import SimCamera
from app.rover.types import MissionCancelled, MovementPhase
from app.rover.vision import LiveVision
from tests.rover.helpers import add_bottle, candidates_of, fast_config, make_world, movement_row, search_status


def test_no_frames_at_all_is_an_error_without_any_motion(harness):
    h = harness(fast_config(frame_timeout_seconds=0.05, max_frame_failures=3))
    h.camera.fail = True
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("error", "stale_frames")
    assert h.drive.moves == [] and h.vision.identify_calls == 0
    assert h.drive.records[-1].command == "stop"
    assert search_status(sid) == SearchStatus.SEARCHING
    assert movement_row(sid).phase == "error"


def test_camera_dying_mid_scan_stops_the_rover(harness):
    h = harness(fast_config(frame_timeout_seconds=0.05))
    h.camera.fail_after = 3                             # three good frames, then nothing
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("error", "stale_frames")
    assert len(h.drive.moves) == 3                      # one pulse per good frame, none on faith afterwards
    assert h.drive.records[-1].command == "stop" and not h.world.is_moving()


def test_frame_that_does_not_advance_is_never_reused(harness):
    """A stuck camera driver keeps returning the same frame. The controller
    must not act on a frame from before the last movement."""
    h = harness(fast_config(frame_timeout_seconds=0.05))
    h.camera.frozen = True
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("error", "stale_frames")
    assert len(h.drive.moves) == 1                      # the first (genuinely fresh) frame allowed one pulse
    assert len({call.frame_seq for call in h.vision.calls}) == 1


class _EarlyCamera(SimCamera):
    """Violates the contract differently: new seq, but stamped before the
    wheels stopped (as if it had been captured mid-motion)."""

    def wait_for_frame(self, after_seq, not_before, timeout, cancel=None):
        packet = super().wait_for_frame(after_seq, not_before, timeout, cancel)
        if packet is None or packet.seq == 1:
            return packet
        return type(packet)(packet.frame, packet.seq, not_before - 1.0, packet.captured_wall)


def test_frame_captured_before_the_wheels_settled_is_rejected(harness):
    cfg = fast_config()
    world = make_world(cfg)
    h = harness(cfg, world, camera=_EarlyCamera(world))
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    end = log.wait_rest()
    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("error", "stale_frames")
    assert len(h.drive.moves) == 1


def test_inference_failure_publishes_errors_then_stops(harness):
    cfg = fast_config(max_vision_errors=3)
    world = make_world(cfg)
    add_bottle(world, bearing=0, distance=1.5, is_target=True)
    h = harness(cfg, world)
    h.vision.fail_with = RuntimeError("gateway exploded")
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("error", "inference_failed")
    errors = log.of_type("vision_error")
    assert [e["payload"] for e in errors] == [{"message": "gateway exploded"}] * 3
    assert h.drive.moves == []                          # retried from a standstill, never moved blind
    assert candidates_of(sid) == []
    # Each retry looked at a NEW frame.
    seqs = [call.frame_seq for call in h.vision.calls if call.kind == "identify"]
    assert len(seqs) == 3 and len(set(seqs)) == 3


def test_a_success_resets_the_error_count(harness):
    cfg = fast_config(max_vision_errors=3)
    world = make_world(cfg)
    add_bottle(world, bearing=0, distance=1.5, is_target=True)
    h = harness(cfg, world)
    h.vision.fail_with, h.vision.fail_times = RuntimeError("hiccup"), 2
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)

    candidate = log.wait_type("candidate_found")["payload"]   # third call succeeds
    assert len(log.of_type("vision_error")) == 2
    # Two more failures after a success still do not add up to three.
    h.vision.fail_times = 2
    h.controller.accept_candidate(sid, candidate["candidate_id"])
    end = log.wait_rest()
    assert end["payload"]["phase"] == "arrived"
    assert len(log.of_type("vision_error")) == 4


def test_inference_timeout_counts_as_failure(harness):
    cfg = fast_config(inference_timeout_seconds=0.1, max_vision_errors=2)
    h = harness(cfg)
    h.vision.block_event = threading.Event()            # every call hangs (released at teardown)
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("error", "inference_failed")
    messages = [e["payload"]["message"] for e in log.of_type("vision_error")]
    assert len(messages) == 2 and all("timed out" in m for m in messages)
    assert h.drive.moves == []


def test_simulated_vision_with_a_real_driver_refuses_to_start(harness):
    cfg = fast_config()
    world = make_world(cfg)
    add_bottle(world, bearing=0, distance=1.5, is_target=True)
    h = harness(cfg, world, driver_name="gpio")         # SimVision + something that is not the sim driver
    sid, log = h.search("my bottle")

    state = h.controller.start_search(sid)

    assert (state.phase, state.reason) == (MovementPhase.ERROR, "unsafe_vision")
    assert state.active is False and state.driver == "gpio" and state.vision_mode == "simulation"
    assert h.controller.active_search_id() is None
    assert h.controller.join_missions(1.0)              # no mission thread was even started
    assert h.drive.moves == []                          # zero drive calls
    assert h.vision.identify_calls == 0 and candidates_of(sid) == []
    event = log.wait_phase("error", timeout=2)
    assert event["payload"]["reason"] == "unsafe_vision"
    # resume() is refused the same way.
    assert h.controller.resume(sid).reason == "unsafe_vision"
    assert h.drive.moves == []


def test_pulse_itself_refuses_simulated_vision_on_a_real_driver(harness):
    """Defence in depth: even if a mission somehow existed, the pulse checks
    the pairing again at the moment it would start the wheels."""
    h = harness(fast_config(), driver_name="gpio")
    controller = h.controller
    mission = Mission(search_id="s", generation=controller._generation, kind=SCAN, target_text="x")
    with controller._lock:
        controller._mission = mission
        controller._state.phase = MovementPhase.SCANNING
    with pytest.raises(MissionEnd) as refused:
        controller._motion.pulse(mission, "forward", 0.4, 0.01)
    assert refused.value.reason.value == "unsafe_vision"
    assert h.drive.records == []


def test_pulse_refuses_outside_moving_phases_and_for_stale_missions(harness):
    h = harness(fast_config())
    controller = h.controller
    mission = Mission(search_id="s", generation=controller._generation, kind=SCAN, target_text="x")
    with controller._lock:
        controller._mission = mission
        controller._state.phase = MovementPhase.WAITING_FOR_CONFIRMATION
    assert controller._motion.pulse(mission, "forward", 0.4, 0.01) is False
    with pytest.raises(ValueError):
        controller._motion.pulse(mission, "backward", 0.4, 0.01)   # autonomy never reverses

    with controller._lock:
        controller._state.phase = MovementPhase.SCANNING
        controller._generation += 1                                  # anything invalidated the mission
    with pytest.raises(MissionCancelled):
        controller._motion.pulse(mission, "forward", 0.4, 0.01)
    assert h.drive.records == []


class _NoLocalDetector:
    @staticmethod
    def is_available() -> bool:
        return False


def test_live_vision_without_a_key_never_fabricates_a_detection(harness):
    """conftest blanks OMNI_API_KEY. The live adapter must fail loudly, and the
    mission must end in ERROR with no candidate and no movement."""
    cfg = fast_config(max_vision_errors=2)
    world = make_world(cfg)
    add_bottle(world, bearing=0, distance=1.5, is_target=True)
    h = harness(cfg, world, vision=LiveVision(cfg, detector=_NoLocalDetector()))
    sid, log = h.search("my bottle")

    state = h.controller.start_search(sid)
    assert state.vision_mode == "live"
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("error", "inference_failed")
    assert [e["payload"]["message"] for e in log.of_type("vision_error")] == ["OMNI_API_KEY is not set"] * 2
    assert candidates_of(sid) == [] and h.drive.moves == []


# --- garbage detections and out-of-band database changes ---------------------------------

_INVERTED = [600, 500, 400, 450]          # ymin > ymax, xmin > xmax


def test_invalid_box_is_never_offered_as_a_candidate(harness):
    from tests.rover.helpers import ScriptedVision

    h = harness(fast_config(scan_max_steps=3), vision=ScriptedVision(lambda index: {"box": _INVERTED}))
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    end = log.wait_rest()
    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("exhausted", "scan_budget")
    assert candidates_of(sid) == [] and log.of_type("candidate_found") == []


def test_invalid_box_during_the_approach_is_an_error_not_a_heading(harness):
    from tests.rover.helpers import ScriptedVision

    good = [350, 450, 650, 550]
    h = harness(fast_config(), vision=ScriptedVision(lambda index: {"box": good if index == 1 else _INVERTED}))
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    h.controller.accept_candidate(sid, log.wait_type("candidate_found")["payload"]["candidate_id"])
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("error", "invalid_detection")
    assert h.vision.identify_calls == 1 + h.cfg.reacquire_attempts      # stationary retries, then give up
    assert h.drive.moves == []                                          # garbage never steered the rover


def test_search_cancelled_behind_the_controllers_back_creates_no_candidate(harness, media_root):
    """The legacy cancel path wrote CANCELLED straight to the database. The
    mission is still 'current', but the compare-and-set on Search.status must
    refuse to create a candidate (and must not flip the search back)."""
    from sqlmodel import Session

    from app.db import engine
    from app.models import Search

    cfg = fast_config()
    world = make_world(cfg)
    add_bottle(world, bearing=0, distance=1.5, is_target=True)
    h = harness(cfg, world)
    h.vision.block_event = threading.Event()
    sid, log = h.search("my bottle")
    h.controller.start_search(sid)
    assert h.vision.entered_event.wait(5)

    with Session(engine) as session:                  # not through the controller
        search = session.get(Search, sid)
        search.status = SearchStatus.CANCELLED
        session.add(search)
        session.commit()
    h.vision.block_event.set()                        # the positive result now arrives
    end = log.wait_rest()

    assert (end["payload"]["phase"], end["payload"]["reason"]) == ("stopped", "cancelled")
    assert candidates_of(sid) == [] and log.of_type("candidate_found") == []
    assert search_status(sid) == SearchStatus.CANCELLED
    folder = media_root / "searches" / sid
    assert not folder.exists() or list(folder.iterdir()) == []         # the saved images were removed again
    assert h.drive.moves == []
