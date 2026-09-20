"""Fixtures for the API tests.

The routers are exercised through FastAPI's ``TestClient`` against a REAL
``RoverController`` wired to simulated ports (``tests/api/helpers.py``). The
app lifespan is deliberately not run by the client: it would try to open a real
camera. The lifespan has its own tests, on fakes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.rover.controller as controller_module
from app.main import app
from app.rover.controller import set_rover_controller
from app.services.drive_service import drive_service
from app.services.event_bus import event_bus
from tests.api.helpers import ApiRover, BusTap


@pytest.fixture(autouse=True)
def _no_real_controller(monkeypatch):
    """A test that forgets to install a simulated controller must fail loudly
    instead of quietly building one on the real camera and drive services."""

    def refuse():
        raise AssertionError("install a simulated controller with the `rover` fixture first")

    monkeypatch.setattr(controller_module, "_build_default_controller", refuse)
    yield
    set_rover_controller(None)


@pytest.fixture
def tap(monkeypatch) -> BusTap:
    bus_tap = BusTap()
    publish = event_bus.publish

    def tapped(search_id: str, event: dict) -> None:
        bus_tap.record(search_id, event)
        publish(search_id, event)

    monkeypatch.setattr(event_bus, "publish", tapped)
    return bus_tap


@pytest.fixture
def rover(tap):
    """Factory for ``ApiRover``. Depends on ``tap`` so that the bus is tapped
    before any mission can publish."""
    built: list[ApiRover] = []

    def factory(*args, **kwargs) -> ApiRover:
        instance = ApiRover(*args, **kwargs)
        built.append(instance)
        return instance

    yield factory
    set_rover_controller(None)
    for instance in built:
        instance.close()
    drive_service.close()  # retires a watchdog armed by a manual drive command


@pytest.fixture
def client() -> TestClient:
    # Not used as a context manager on purpose: the lifespan (real camera) must not run.
    return TestClient(app)
