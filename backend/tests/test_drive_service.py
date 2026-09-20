"""DriveService deadman (ttl watchdog) behaviour, observed at the motor driver.

A recording fake driver stands in for the hardware; every assertion is about
the wheel speeds it actually received and when. Lower time bounds are exact
(the wheels must never be cut early); upper bounds are generous so a busy
machine cannot make the suite flaky.

Service state is always read through state(): it takes the service lock, so it
cannot observe the instant between the watchdog's motor write and its
bookkeeping.
"""

import threading
import time
from unittest.mock import patch

import pytest

from app.config import settings
from app.motors.base import MotorDriver
from app.rover.types import DrivePort
from app.services.drive_service import COMMANDS, DriveService, _make_driver, drive_service

WAIT = 5.0
STOPPED = (0.0, 0.0)


class RecordingDriver(MotorDriver):
    name = "fake"

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.calls: list[tuple[float, float]] = []
        self.stamps: list[float] = []
        self.fail_remaining = 0  # refuse this many upcoming set_speeds() calls
        self.fail_stops_remaining = 0  # refuse this many upcoming *stop* writes only
        self.failures = 0
        self.closed = False

    def set_speeds(self, left: float, right: float) -> None:
        with self._cond:
            is_stop = (left, right) == STOPPED
            if self.fail_remaining > 0 or (is_stop and self.fail_stops_remaining > 0):
                if self.fail_remaining > 0:
                    self.fail_remaining -= 1
                else:
                    self.fail_stops_remaining -= 1
                self.failures += 1
                self._cond.notify_all()
                raise RuntimeError("driver fault")
            self.calls.append((left, right))
            self.stamps.append(time.monotonic())
            self._cond.notify_all()

    def close(self) -> None:
        super().close()
        self.closed = True

    def wait_for_calls(self, count: int, timeout: float = WAIT) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: len(self.calls) >= count, timeout)


@pytest.fixture(autouse=True)
def _neutral_calibration():
    with patch.multiple(settings, motor_left_scale=1.0, motor_right_scale=1.0, drive_max_speed=1.0):
        yield


@pytest.fixture
def driver() -> RecordingDriver:
    return RecordingDriver()


@pytest.fixture
def service(driver: RecordingDriver):
    # Swapped in without start(), exactly like test_motor_driver.py does: the
    # watchdog has to come up lazily when the first ttl is armed.
    svc = DriveService()
    svc._driver = driver
    yield svc
    svc.close()


# --- the deadman ----------------------------------------------------------


def test_ttl_stops_the_wheels_without_any_help_from_the_caller(service, driver):
    service.drive("forward", 0.5, ttl=30.0)
    state = service.state()
    assert state["command"] == "forward"
    assert 25.0 < state["deadline_in"] <= 30.0

    before = time.monotonic()
    service.drive("forward", 0.5, ttl=0.15)

    # The caller now does nothing at all -- as if it were stuck in a hung request.
    assert driver.wait_for_calls(3)
    assert driver.calls == [(0.5, 0.5), (0.5, 0.5), STOPPED]
    assert driver.stamps[2] - before >= 0.15  # never cut early

    state = service.state()
    assert (state["command"], state["left"], state["right"]) == ("stop", 0.0, 0.0)
    assert state["watchdog_trips"] == 1
    assert state["deadline_in"] is None


def test_set_speeds_honours_ttl_too(service, driver):
    service.set_speeds(0.3, -0.2, command="custom", ttl=0.05)
    assert driver.wait_for_calls(2)
    assert driver.calls == [(0.3, -0.2), STOPPED]
    assert service.state()["command"] == "stop"


def test_a_new_command_replaces_the_previous_deadline(service, driver):
    # The first ttl is long enough that a slow machine cannot trip it between
    # the two commands, and short enough to fire well inside the second one.
    service.drive("forward", 0.5, ttl=0.25)
    rearmed = time.monotonic()
    service.drive("turn_left", 0.4, ttl=0.6)

    assert driver.wait_for_calls(3)
    assert driver.calls == [(0.5, 0.5), (-0.4, 0.4), STOPPED]
    # The 250 ms deadline died with the command it belonged to.
    assert driver.stamps[2] - rearmed >= 0.6
    assert service.state()["watchdog_trips"] == 1


def test_a_shorter_deadline_takes_effect_immediately(service, driver):
    service.drive("forward", 0.5, ttl=60.0)
    service.drive("forward", 0.5, ttl=0.05)
    assert driver.wait_for_calls(3)  # well before the first command's 60 s
    assert driver.calls[-1] == STOPPED


def test_stop_clears_the_deadline(service, driver):
    service.drive("forward", 0.5, ttl=0.25)
    service.stop()

    assert driver.calls == [(0.5, 0.5), STOPPED]
    assert service.state()["deadline_in"] is None
    # Nothing further happens when the old deadline passes.
    assert not driver.wait_for_calls(3, timeout=0.5)
    assert service.state()["watchdog_trips"] == 0


def test_ttl_none_clears_the_deadline_and_never_auto_stops(service, driver):
    service.drive("forward", 0.5, ttl=0.25)
    service.drive("forward", 0.6)  # ttl=None: caller takes responsibility for stopping

    assert service.state()["deadline_in"] is None
    assert not driver.wait_for_calls(3, timeout=0.5)  # the old 250 ms deadline never fires
    assert driver.calls == [(0.5, 0.5), (0.6, 0.6)]
    state = service.state()
    assert (state["command"], state["left"], state["watchdog_trips"]) == ("forward", 0.6, 0)


def test_commands_without_ttl_never_start_a_watchdog(service, driver):
    service.drive("forward", 0.5)
    service.stop()
    assert service._watchdog is None
    assert driver.calls == [(0.5, 0.5), STOPPED]


def test_a_stop_command_is_not_given_a_deadline(service, driver):
    service.drive("stop", ttl=0.05)
    assert service.state()["deadline_in"] is None
    assert not driver.wait_for_calls(2, timeout=0.2)
    assert service.state()["watchdog_trips"] == 0


def test_watchdog_works_again_after_a_trip(service, driver):
    service.drive("forward", 0.5, ttl=0.05)
    assert driver.wait_for_calls(2)
    service.drive("backward", 0.5, ttl=0.05)
    assert driver.wait_for_calls(4)
    assert driver.calls == [(0.5, 0.5), STOPPED, (-0.5, -0.5), STOPPED]
    assert service.state()["watchdog_trips"] == 2


# --- driver faults --------------------------------------------------------


def test_watchdog_survives_a_driver_exception_and_retries_the_stop(service, driver):
    driver.fail_stops_remaining = 2  # the watchdog's first two stop attempts blow up
    service.drive("forward", 0.5, ttl=0.05)

    assert driver.wait_for_calls(2)
    assert driver.calls == [(0.5, 0.5), STOPPED]
    assert driver.failures == 2
    assert service.state()["watchdog_trips"] == 1
    assert service._watchdog.is_alive()

    # ...and it is still on duty for the next command.
    service.drive("turn_right", 0.4, ttl=0.05)
    assert driver.wait_for_calls(4)
    assert driver.calls[-1] == STOPPED
    assert service.state()["watchdog_trips"] == 2


def test_a_refused_stop_is_retried_by_the_watchdog(service, driver):
    service.drive("forward", 0.5)  # no ttl: only an explicit stop would end this
    driver.fail_remaining = 1
    with pytest.raises(RuntimeError, match="driver fault"):
        service.stop()

    assert driver.wait_for_calls(2)
    assert driver.calls == [(0.5, 0.5), STOPPED]
    assert service.state()["command"] == "stop"


def test_a_refused_drive_command_is_reported_and_leaves_the_wheels_stopped(service, driver):
    driver.fail_remaining = 1
    with pytest.raises(RuntimeError, match="driver fault"):
        service.drive("forward", 0.5, ttl=10.0)

    assert service.state()["command"] == "stop"  # the refused command was never recorded as running
    assert driver.wait_for_calls(1)
    assert driver.calls == [STOPPED]


# --- validation -----------------------------------------------------------


@pytest.mark.parametrize("ttl", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_ttl_is_rejected_before_any_motor_command(service, driver, ttl):
    with pytest.raises(ValueError):
        service.drive("forward", 0.5, ttl=ttl)
    assert driver.calls == []
    assert service.state()["command"] == "stop"


def test_non_finite_speed_is_rejected_before_any_motor_command(service, driver):
    with pytest.raises(ValueError):
        service.drive("forward", float("nan"))
    with pytest.raises(ValueError):
        service.set_speeds(float("inf"), 0.0)
    with pytest.raises(ValueError):
        service.drive("sideways", 0.5)
    assert driver.calls == []


def test_command_table_is_unchanged():
    assert COMMANDS["turn_left"] == (-1.0, 1.0)
    assert COMMANDS["turn_right"] == (1.0, -1.0)
    assert COMMANDS["forward"] == (1.0, 1.0)
    assert COMMANDS["stop"] == STOPPED


# --- lifecycle ------------------------------------------------------------


def test_close_stops_motors_retires_the_watchdog_and_releases_the_driver(driver):
    service = DriveService()
    service._driver = driver
    service.drive("forward", 0.5, ttl=60.0)
    watchdog = service._watchdog
    assert watchdog is not None and watchdog.is_alive()

    service.close()

    assert driver.calls[0] == (0.5, 0.5)
    assert driver.calls[-1] == STOPPED
    assert driver.closed
    assert not watchdog.is_alive()
    assert service.driver_name == "none"
    state = service.state()
    assert (state["command"], state["left"], state["right"]) == ("stop", 0.0, 0.0)
    assert state["deadline_in"] is None
    assert state["driver"] == "none"

    service.close()  # idempotent


def test_start_brings_up_the_watchdog_and_the_service_restarts_after_close():
    service = DriveService()
    assert (service.driver_name, service.is_simulated) == ("none", False)

    with patch.object(settings, "motor_driver", "sim"):
        service.start()
    try:
        assert (service.driver_name, service.is_simulated) == ("sim", True)
        assert service._watchdog is not None and service._watchdog.is_alive()
        first_driver = service._driver
        service.start()  # a second start must not build (and leak) another driver
        assert service._driver is first_driver
    finally:
        service.close()
    assert service._watchdog is None

    with patch.object(settings, "motor_driver", "sim"):
        service.start()
    try:
        sim = service._driver
        service.drive("forward", 0.5, ttl=0.05)
        deadline = time.monotonic() + WAIT
        while service.state()["watchdog_trips"] == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert service.state()["watchdog_trips"] == 1
        assert (sim.left, sim.right) == STOPPED
    finally:
        service.close()


def test_driver_name_reports_the_real_driver(service, driver):
    assert (service.driver_name, service.is_simulated) == ("fake", False)
    driver.name = "gpio"
    assert (service.driver_name, service.is_simulated) == ("gpio", False)
    assert service.state()["driver"] == "gpio"


def test_satisfies_the_drive_port():
    assert isinstance(DriveService(), DrivePort)
    assert isinstance(drive_service, DriveService)


def test_state_keeps_existing_keys_and_adds_watchdog_fields(service):
    assert service.state() == {
        "command": "stop",
        "left": 0.0,
        "right": 0.0,
        "driver": "fake",
        "watchdog_trips": 0,
        "deadline_in": None,
    }


# --- _make_driver must keep failing loudly ---------------------------------


def test_unknown_driver_raises_and_leaves_no_driver():
    service = DriveService()
    with patch.object(settings, "motor_driver", "warp"):
        with pytest.raises(ValueError, match="Unknown motor driver"):
            _make_driver()
        with pytest.raises(ValueError, match="Unknown motor driver"):
            service.start()
    assert (service.driver_name, service.is_simulated) == ("none", False)


def test_failed_gpio_driver_raises_instead_of_falling_back_to_sim():
    service = DriveService()
    with patch.object(settings, "motor_driver", "gpio"):
        with patch("app.motors.gpio.GpioMotorDriver", side_effect=RuntimeError("GPIO unavailable")):
            with pytest.raises(RuntimeError, match="GPIO unavailable"):
                service.start()
    assert (service.driver_name, service.is_simulated) == ("none", False)
