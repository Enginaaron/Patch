"""App lifespan: nothing moves on startup; on shutdown the mission stops, then
the motors and the camera are released -- whatever fails on the way. Runs on a
fake camera source and the sim motor driver; no hardware is touched."""

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.camera.base import CameraSource
from app.config import settings
from app.rover import simulation
from app.rover.controller import set_rover_controller
from app.services.camera_service import CameraService
from app.services.drive_service import DriveService
from tests.api.helpers import create_search, detail, nothing_to_find, target_in_view


class _Frames(CameraSource):
    """A camera that needs no device."""

    def __init__(self) -> None:
        self.started = self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def get_frame(self) -> np.ndarray:
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def stop(self) -> None:
        self.stopped += 1


@pytest.fixture
def services(monkeypatch):
    """Real DriveService / CameraService instances (not the process-wide
    singletons) swapped into app.main, on the sim driver and a fake source."""
    source = _Frames()
    drive, camera = DriveService(), CameraService(source=source)
    monkeypatch.setattr(main, "drive_service", drive)
    monkeypatch.setattr(main, "camera_service", camera)
    yield drive, camera, source
    # Through the classes: a test may have replaced close()/stop() on the instance.
    DriveService.close(drive)
    CameraService.stop(camera)


def _enter_lifespan() -> None:
    async def run() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(asyncio.wait_for(run(), 15))


def test_startup_moves_nothing_and_shutdown_stops_mission_motors_and_camera(services, rover, tap):
    drive, camera, source = services
    h = rover(*nothing_to_find())

    with TestClient(main.app) as live:                   # runs the real lifespan
        assert live.get("/api/health").json() == {"status": "ok"}
        assert camera.is_available() and source.started == 1
        assert drive.driver_name == "sim" and drive.state()["command"] == "stop"
        assert h.drive.records == [] and h.controller.active_search_id() is None   # nothing moved by itself

        sid = create_search(live, "my bottle")["search_id"]
        assert h.drive.wait_for_moves(2)
        assert detail(live, sid)["movement"]["active"] is True

    # Shutdown with a mission mid-scan:
    state = h.controller.state_for(sid)
    assert (state.phase.value, state.reason, state.active) == ("stopped", "shutdown", False)
    assert h.controller.join_missions(5)
    assert h.drive.records[-1].command == "stop" and not h.world.is_moving()
    assert drive.driver_name == "none"                   # motor driver released
    assert not camera.is_available() and source.stopped >= 1
    # The search itself survives a restart: it is resumable, not cancelled.
    assert tap.events(sid, "search_cancelled") == []


class _ExplodingController:
    def shutdown(self) -> None:
        raise RuntimeError("controller shutdown blew up")


def test_motors_and_camera_are_released_even_when_the_controller_fails_to_stop(services):
    drive, camera, source = services
    set_rover_controller(_ExplodingController())

    _enter_lifespan()                                    # must not raise

    assert drive.driver_name == "none" and drive.state()["command"] == "stop"
    assert not camera.is_available() and source.stopped >= 1


def test_the_camera_is_released_even_when_the_motor_driver_fails_to_close(services, rover, monkeypatch):
    drive, camera, source = services
    rover(*target_in_view())

    def broken_close() -> None:
        raise RuntimeError("driver close blew up")

    monkeypatch.setattr(drive, "close", broken_close)

    _enter_lifespan()                                    # must not raise

    assert not camera.is_available() and source.stopped >= 1


def test_simulation_is_refused_with_a_real_motor_driver(services, rover, monkeypatch):
    drive, camera, source = services
    rover(*target_in_view())
    monkeypatch.setattr(settings, "rover_simulation", True)
    monkeypatch.setattr(settings, "motor_driver", "gpio")

    with pytest.raises(RuntimeError, match="MOTOR_DRIVER=sim"):
        _enter_lifespan()

    # It refused before anything came up: no camera, and above all no motor driver.
    assert source.started == 0 and not camera.is_available()
    assert drive.driver_name == "none"


def test_simulation_swaps_in_the_synthetic_camera_before_the_camera_starts(services, rover, monkeypatch):
    drive, camera, source = services
    rover(*target_in_view())
    monkeypatch.setattr(settings, "rover_simulation", True)   # MOTOR_DRIVER is already "sim" in tests
    simulation.reset_sim_world()
    seen: dict = {}

    async def run() -> None:
        async with main.lifespan(main.app):
            packet = await asyncio.to_thread(camera.wait_for_frame, 0, 0.0, 10.0)
            seen["shape"] = None if packet is None else packet.frame.shape
            seen["driver"] = drive.driver_name

    try:
        asyncio.run(asyncio.wait_for(run(), 20))
    finally:
        simulation.reset_sim_world()

    assert seen == {"shape": (480, 640, 3), "driver": "sim"}   # the SIMULATED world's frames, not the fake source's
    assert source.started == 0
