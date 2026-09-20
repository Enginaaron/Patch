"""Shared scaffolding for the API tests (see conftest.py for the fixtures).

Everything about motion and detection here is SIMULATED: a real
``RoverController`` runs on the ports from ``app.rover.simulation``. Waiting is
done on a tap of the event bus (a condition variable with a timeout), never
with fixed sleeps.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from fastapi.testclient import TestClient
from sqlmodel import select

from app.db import get_session
from app.models import Search
from app.rover.config import RoverConfig
from app.rover.controller import RoverController, set_rover_controller
from app.rover.simulation import RecordingDrive, SimCamera, SimDrive, SimVision, SimWorld, TeeDrive
from app.services.drive_service import drive_service
from tests.rover.helpers import add_bottle, fast_config, make_world


class BusTap:
    """Sees every event published on the bus, for any search -- including the
    ones published before a test could know the new search's id."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._events: list[tuple[str, dict]] = []

    def record(self, search_id: str, event: dict) -> None:
        with self._cond:
            self._events.append((search_id, event))
            self._cond.notify_all()

    def mark(self) -> int:
        """Pass as ``after=`` to only match events published from now on."""
        with self._cond:
            return len(self._events)

    def events(self, search_id: str, event_type: str | None = None) -> list[dict]:
        with self._cond:
            return [
                event
                for sid, event in self._events
                if sid == search_id and (event_type is None or event.get("type") == event_type)
            ]

    def wait_for(
        self, search_id: str, predicate: Callable[[dict], bool], *, timeout: float = 10.0, after: int = 0, what: str = "event"
    ) -> dict:
        def find() -> dict | None:
            for sid, event in self._events[after:]:
                if sid == search_id and predicate(event):
                    return event
            return None

        with self._cond:
            if not self._cond.wait_for(lambda: find() is not None, timeout=timeout):
                seen = [self._label(event) for sid, event in self._events if sid == search_id]
                raise AssertionError(f"timed out waiting for {what}; saw {seen}")
            return find()

    def wait_type(self, search_id: str, event_type: str, **kwargs) -> dict:
        return self.wait_for(search_id, lambda e: e.get("type") == event_type, what=event_type, **kwargs)

    def wait_phase(self, search_id: str, *phases: str, **kwargs) -> dict:
        wanted = set(phases)
        return self.wait_for(
            search_id,
            lambda e: e.get("type") == "movement" and e["payload"]["phase"] in wanted,
            what=f"phase in {sorted(wanted)}",
            **kwargs,
        )

    @staticmethod
    def _label(event: dict) -> str:
        if event.get("type") == "movement":
            return f"movement:{event['payload']['phase']}:{event['payload']['reason']}"
        return str(event.get("type"))


class ApiRover:
    """A real controller on simulated ports, installed as THE controller the
    routers use."""

    def __init__(
        self,
        cfg: RoverConfig | None = None,
        world: SimWorld | None = None,
        *,
        vision=None,
        through_drive_service: bool = False,
        driver_name: str | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else fast_config()
        self.world = world if world is not None else make_world(self.cfg)
        sim_drive = SimDrive(self.world)
        # through_drive_service: the production wiring of the laptop demo --
        # mission commands reach the real drive service (sim motor driver) and
        # are mirrored into the world, so manual and mission commands meet in
        # the same place, as they do on the rover.
        inner = TeeDrive(drive_service, sim_drive) if through_drive_service else sim_drive
        # driver_name impersonates a real motor driver (the wrapped port still only simulates).
        self.drive = RecordingDrive(inner, driver_name=driver_name)
        self.camera = SimCamera(self.world)
        self.vision = vision if vision is not None else SimVision(self.world)
        self.controllers: list[RoverController] = []
        self.controller = self.new_controller()

    def new_controller(self) -> RoverController:
        """Build + install a controller on the same world and database. Called
        a second time, this is what a backend restart looks like: no live
        mission, a fresh generation counter."""
        controller = RoverController(self.drive, self.camera, self.vision, self.cfg, tracking_sink=lambda box: None)
        self.controllers.append(controller)
        self.controller = controller
        set_rover_controller(controller)
        return controller

    def moves_since(self, moment: float) -> list:
        return [record for record in self.drive.moves if record.at > moment]

    def close(self) -> None:
        release = getattr(self.vision, "release", None)
        if callable(release):
            release()  # unblock a hanging identify() BEFORE shutdown joins the thread
        for controller in self.controllers:
            controller.shutdown()


# --- worlds ---------------------------------------------------------------------


def target_in_view(**cfg_overrides) -> tuple[RoverConfig, SimWorld]:
    """The user's bottle stands straight ahead: the very first look finds it,
    so the mission parks on the candidate question without a single pulse."""
    cfg = fast_config(**cfg_overrides)
    world = make_world(cfg)
    add_bottle(world, bearing=0.0, distance=1.5, is_target=True, description="a blue water bottle (simulated)")
    return cfg, world


def nothing_to_find(**cfg_overrides) -> tuple[RoverConfig, SimWorld]:
    """An empty room and a huge scan budget: the mission keeps pulsing until
    somebody stops it."""
    cfg_overrides.setdefault("scan_max_steps", 100_000)
    cfg = fast_config(**cfg_overrides)
    return cfg, make_world(cfg)


# --- helpers ----------------------------------------------------------------------


def create_search(client: TestClient, text: str = "my bottle", **form) -> dict:
    response = client.post("/api/searches", data={"target_text": text, **form})
    assert response.status_code == 200, response.text
    return response.json()


def detail(client: TestClient, search_id: str) -> dict:
    response = client.get(f"/api/searches/{search_id}")
    assert response.status_code == 200, response.text
    return response.json()


def search_count() -> int:
    with get_session() as session:
        return len(session.exec(select(Search.id)).all())


def now() -> float:
    return time.monotonic()
