"""The simulated backend end to end, on its PRODUCTION wiring.

The other API tests inject a controller built by the test. This one does not:
the real app lifespan runs with ROVER_SIMULATION on, so the path under test is
the one the laptop demo uses --

    lifespan -> install_simulation(camera_service) -> camera_service.start()
             -> drive_service.start() (sim motor driver) -> get_rover_controller()
             -> TeeDrive(drive_service, SimDrive) + camera_service + SimVision

and everything after that goes over HTTP, the way the frontend drives it.

Everything about motion and detection here is SIMULATED; nothing in this file
says anything about the physical rover. No camera, GPIO or network is touched.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.rover.controller as controller_module
from app.config import settings
from app.main import app
from app.rover import simulation
from app.rover.controller import _build_default_controller as build_default_controller
from app.rover.controller import get_rover_controller
from app.services.camera_service import camera_service
from app.services.drive_service import drive_service
from tests.api.helpers import create_search, detail

# Time compression, done the way scripts/demo_rover_sim.py does it: pulses are
# SPEED times shorter and the (made-up) simulated rover is SPEED times quicker,
# so every pulse turns / advances exactly as far as it would in real time.
SPEED = 10.0
_DIVIDED = (
    "rover_scan_pulse_seconds",
    "rover_turn_pulse_min_seconds",
    "rover_turn_pulse_max_seconds",
    "rover_forward_pulse_seconds",
    "rover_slow_forward_pulse_seconds",
)


@pytest.fixture
def simulated_backend(request, monkeypatch, tap):
    """The real app, started through its lifespan in simulation mode, on the
    process-wide camera / drive services (the ones the default controller is
    built on). Yields ``(client, world)``. Parametrise indirectly with a
    scenario name (default ``two_bottles``)."""
    scenario = getattr(request, "param", "two_bottles")
    # tests/api/conftest.py forbids the default controller; this test is about it.
    monkeypatch.setattr(controller_module, "_build_default_controller", build_default_controller)
    monkeypatch.setattr(settings, "rover_simulation", True)
    monkeypatch.setattr(settings, "rover_sim_scenario", scenario)
    assert settings.motor_driver == "sim"          # tests/conftest.py; the lifespan refuses anything else
    for name in _DIVIDED:
        monkeypatch.setattr(settings, name, getattr(settings, name) / SPEED)
    monkeypatch.setattr(settings, "rover_turn_degrees_per_second", settings.rover_turn_degrees_per_second * SPEED)
    monkeypatch.setattr(settings, "rover_settle_seconds", 0.0)
    # Generous: the margin only bounds a runaway pulse, it never slows the test,
    # and a loaded machine must not turn a late stop into a watchdog trip.
    monkeypatch.setattr(settings, "rover_pulse_watchdog_margin_seconds", 0.5)
    if scenario == "empty":
        # Nothing to find: keep scanning until the test stops it, however slow the machine.
        monkeypatch.setattr(settings, "rover_scan_max_steps", 100_000)

    simulation.reset_sim_world()
    world = simulation.get_sim_world()             # built from the settings above, as in production
    world.forward_metres_per_second *= SPEED       # the one rate that is not a setting

    # install_simulation() swaps the shared camera's source for the synthetic
    # one BEFORE the camera starts, so no real device is ever opened. Put the
    # original back afterwards so nothing leaks into other tests.
    original_source = camera_service._camera
    try:
        with TestClient(app) as client:            # runs the real lifespan
            yield client, world
        # Only reached after a passing test: the lifespan shutdown released
        # the motor driver and the camera, whatever the mission was doing.
        assert drive_service.driver_name == "none" and drive_service.state()["command"] == "stop"
        assert not camera_service.is_available() and not world.is_moving()
    finally:
        camera_service.stop()                      # no-ops after a clean shutdown
        drive_service.close()
        camera_service.set_source(original_source)
        simulation.reset_sim_world()


def _wait_for_question(tap, client, search_id: str, *, after: int = 0) -> dict:
    """Block until the rover is parked on a candidate question (event bus, no
    polling), then read it the way the frontend does: from the detail."""
    tap.wait_type(search_id, "candidate_found", after=after, timeout=20)
    tap.wait_phase(search_id, "waiting_for_confirmation", after=after, timeout=20)
    body = detail(client, search_id)
    assert body["status"] == "CANDIDATE_PENDING" and body["pending_candidate"] is not None
    assert (body["movement"]["phase"], body["movement"]["active"]) == ("waiting_for_confirmation", True)
    return body


def test_simulated_backend_scans_asks_twice_and_arrives_at_the_accepted_bottle(simulated_backend, tap):
    client, world = simulated_backend

    # Startup moved nothing, and the wiring is the simulated one.
    state = client.get("/api/rover/state").json()
    assert state["active_search_id"] is None
    assert (state["drive"]["command"], state["drive"]["driver"]) == ("stop", "sim")
    assert state["camera"]["available"] is True and not world.is_moving()
    assert world.pose() == (0.0, 0.0, 0.0)

    sid = create_search(client, "Find my water bottle beside my backpack")["search_id"]

    # 1) Scanning right meets the decoy first. Wheels are stopped for the question.
    first = _wait_for_question(tap, client, sid)
    decoy = first["pending_candidate"]
    assert decoy["description"] == "a green glass bottle on the floor"
    assert first["movement"]["vision_mode"] == "simulation" and first["movement"]["driver"] == "sim"
    assert first["resolved_target"]["category"] == "bottle"
    assert client.get(decoy["crop_url"]).status_code == 200 and client.get(decoy["image_url"]).status_code == 200
    assert not world.is_moving() and client.get("/api/drive/state").json()["command"] == "stop"
    heading_at_decoy = world.pose()[2]
    assert heading_at_decoy > 0                           # it had to turn right to get here

    # 2) "No": back to SEARCHING, keeps scanning, offers the OTHER bottle next.
    mark = tap.mark()
    rejected = client.post(f"/api/searches/{sid}/candidates/{decoy['candidate_id']}/reject")
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["status"] == "SEARCHING"
    second = _wait_for_question(tap, client, sid, after=mark)
    target = second["pending_candidate"]
    assert target["candidate_id"] != decoy["candidate_id"]
    assert target["description"] == "a blue water bottle beside a backpack"
    assert world.pose()[2] > heading_at_decoy             # further round: it kept scanning
    assert world.pose()[:2] == (0.0, 0.0)                 # ...and never drove forward before a "yes"

    # 3) "Yes": FOUND at once -- which is the identification, not an arrival.
    accepted = client.post(f"/api/searches/{sid}/candidates/{target['candidate_id']}/accept")
    assert accepted.status_code == 200, accepted.text
    body = accepted.json()
    assert body["status"] == "FOUND" and body["terminal"] is False
    assert body["movement"]["phase"] != "arrived"
    assert body["accepted_candidate"]["candidate_id"] == target["candidate_id"]

    # 4) Centre + approach in bounded pulses until proximity is confirmed.
    arrived = tap.wait_phase(sid, "arrived", timeout=30)
    assert arrived.get("terminal") is True and arrived["payload"]["reason"] == "proximity_confirmed"

    final = detail(client, sid)
    assert final["status"] == "FOUND" and final["terminal"] is True and final["can_resume"] is False
    assert (final["movement"]["phase"], final["movement"]["active"]) == ("arrived", False)
    assert final["movement"]["approach_pulses"] > 0

    # The wheels are stopped -- per the drive service (the only path to the
    # motor driver) and per the simulated world -- and every mission pulse was
    # ended by the controller itself, never by the watchdog.
    state = client.get("/api/rover/state").json()
    assert state["active_search_id"] is None
    drive = state["drive"]
    assert (drive["command"], drive["left"], drive["right"]) == ("stop", 0.0, 0.0)
    assert drive["deadline_in"] is None and drive["watchdog_trips"] == 0
    assert not world.is_moving() and world.watchdog_trips == 0
    # Simulated rover: next to the accepted bottle, facing it, away from the decoy.
    assert world.distance_to("target_bottle") < 0.8 < world.distance_to("decoy_bottle")
    assert abs(world.relative_bearing("target_bottle")) < 10.0

    # A finished search ends its event stream by itself (first message, then close).
    with client.stream("GET", f"/api/searches/{sid}/events") as stream:
        lines = [line for line in stream.iter_lines() if line.startswith("data:")]
    assert len(lines) == 1 and '"terminal": true' in lines[0]


@pytest.mark.parametrize("simulated_backend", ["empty"], indirect=True)
def test_simulated_backend_stop_resume_and_cancel_over_http(simulated_backend, tap):
    client, world = simulated_backend
    sid = create_search(client, "my bottle")["search_id"]   # empty room: it scans until somebody stops it
    tap.wait_phase(sid, "scanning", timeout=20)
    controller = get_rover_controller()

    stopped = client.post("/api/rover/stop")

    assert stopped.status_code == 200, stopped.text
    body = stopped.json()
    assert body["active_search_id"] is None and body["drive"]["command"] == "stop"
    assert not world.is_moving()
    after = detail(client, sid)
    assert after["status"] == "SEARCHING" and after["can_resume"] is True and after["terminal"] is False
    assert (after["movement"]["phase"], after["movement"]["reason"]) == ("stopped", "user_stop")
    # The mission thread is gone, so nothing can move again by itself.
    assert controller.join_missions(5)
    pose = world.pose()

    mark = tap.mark()
    resumed = client.post(f"/api/searches/{sid}/resume")
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["movement"]["active"] is True
    assert world.pose()[:2] == pose[:2]                     # scanning only ever turns on the spot
    tap.wait_phase(sid, "scanning", after=mark, timeout=20)

    cancelled = client.post(f"/api/searches/{sid}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    body = cancelled.json()
    assert body["status"] == "CANCELLED" and body["terminal"] is True and body["can_resume"] is False
    assert (body["movement"]["phase"], body["movement"]["active"]) == ("stopped", False)
    assert controller.join_missions(5)
    assert client.get("/api/drive/state").json()["command"] == "stop" and not world.is_moving()
    assert tap.wait_type(sid, "search_cancelled", after=mark).get("terminal") is True
