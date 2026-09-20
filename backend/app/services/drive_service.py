"""High-level differential drive.

Turns named commands (forward, turn_left, rotate_right, stop, …) into signed
left/right wheel speeds and pushes them to the active :class:`MotorDriver`.
Also the single source of truth for "what is Patch doing right now", which the
rover controller and the UI both read.

This is the only path to the motor driver, so the deadman lives here: a
command may carry a ``ttl``, and an independent watchdog thread zeroes the
wheels at that deadline even if the caller that started them is blocked
forever (hung inference, stalled camera, dropped browser connection).
"""

import logging
import math
import threading
import time

from app.config import settings
from app.motors.base import MotorDriver
from app.motors.sim import SimMotorDriver

logger = logging.getLogger(__name__)

# command -> (left, right) as fractions of the requested speed
COMMANDS: dict[str, tuple[float, float]] = {
    "forward": (1.0, 1.0),
    "backward": (-1.0, -1.0),
    "turn_left": (-1.0, 1.0),   # rotate in place, counter-clockwise
    "turn_right": (1.0, -1.0),  # rotate in place, clockwise
    "veer_left": (0.3, 1.0),    # gentle arc while moving forward
    "veer_right": (1.0, 0.3),
    "stop": (0.0, 0.0),
}

# How soon the watchdog retries a stop the driver refused.
_WATCHDOG_RETRY_SECONDS = 0.1


def _make_driver() -> MotorDriver:
    # Deliberately no fallback: a rover configured for real motors must fail
    # loudly rather than quietly "drive" a simulator.
    if settings.motor_driver == "gpio":
        from app.motors.gpio import GpioMotorDriver

        return GpioMotorDriver()
    if settings.motor_driver == "sim":
        return SimMotorDriver()
    raise ValueError(f"Unknown motor driver: {settings.motor_driver!r}")


class DriveService:
    def __init__(self) -> None:
        self._driver: MotorDriver | None = None
        self._lock = threading.Lock()
        # Shares _lock: a command and its deadline change together, so the
        # watchdog can never zero a newer command using an older deadline.
        self._wake = threading.Condition(self._lock)
        self._deadline: float | None = None  # time.monotonic() at which the wheels must be stopped
        self._watchdog: threading.Thread | None = None
        self._watchdog_id = 0  # bumped by close(); retires the running watchdog thread
        self.watchdog_trips = 0
        self.command = "stop"
        self.left = 0.0
        self.right = 0.0

    def start(self) -> None:
        with self._lock:
            if self._driver is not None:
                return  # already started; building a second GPIO driver would fight over the pins
            self._driver = _make_driver()  # raises (and leaves us driverless) on a bad/failed driver
            self._ensure_watchdog()
            name = self._driver.name
        logger.info("drive service using '%s' motor driver", name)

    @property
    def driver_name(self) -> str:
        driver = self._driver
        return driver.name if driver is not None else "none"

    @property
    def is_simulated(self) -> bool:
        return self.driver_name == "sim"

    def set_speeds(self, left: float, right: float, command: str = "custom", ttl: float | None = None) -> None:
        """Apply wheel speeds. ``ttl`` (seconds) arms the watchdog: the wheels
        are stopped at that deadline unless another command arrives first.
        Every command replaces the previous deadline; ``ttl=None`` clears it."""
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Wheel speeds must be finite")
        if ttl is not None and (not math.isfinite(ttl) or ttl <= 0):
            raise ValueError("ttl must be a positive number of seconds, or None")
        cap = max(0.0, min(1.0, settings.drive_max_speed))
        # Apply equally in forward/reverse; zero remains exactly zero. Limit
        # after calibration so a boosted wheel cannot exceed the power cap.
        left = max(-cap, min(cap, left * settings.motor_left_scale))
        right = max(-cap, min(cap, right * settings.motor_right_scale))
        with self._lock:
            try:
                if self._driver is not None:
                    self._driver.set_speeds(left, right)
            except Exception:
                # The wheels may be in any state now (the refused command
                # could even have been a stop). Have the watchdog force a stop
                # and keep retrying, instead of trusting the caller to recover.
                logger.error("motor driver refused %r; the watchdog will force a stop", command)
                self._deadline = time.monotonic()
                self._ensure_watchdog()
                self._wake.notify_all()
                raise
            self.left, self.right, self.command = left, right, command
            if ttl is None or (left == 0.0 and right == 0.0):
                self._deadline = None  # nothing to time out
            else:
                self._deadline = time.monotonic() + ttl
                self._ensure_watchdog()
            self._wake.notify_all()

    def drive(self, command: str, speed: float | None = None, ttl: float | None = None) -> None:
        if command not in COMMANDS:
            raise ValueError(f"unknown drive command: {command!r}")
        if speed is not None and not math.isfinite(speed):
            # min()/max() would quietly turn NaN into full speed.
            raise ValueError("Drive speed must be finite")
        speed = settings.drive_max_speed if speed is None else max(0.0, min(1.0, speed))
        lf, rf = COMMANDS[command]
        self.set_speeds(lf * speed, rf * speed, command=command, ttl=ttl)

    def stop(self) -> None:
        self.drive("stop")

    def state(self) -> dict:
        with self._lock:
            deadline = self._deadline
            return {
                "command": self.command,
                "left": self.left,
                "right": self.right,
                "driver": self._driver.name if self._driver else "none",
                "watchdog_trips": self.watchdog_trips,
                "deadline_in": None if deadline is None else max(0.0, deadline - time.monotonic()),
            }

    def close(self) -> None:
        """Stop the motors, retire the watchdog thread, release the driver.
        Safe to call twice; the service can be start()ed again afterwards."""
        with self._lock:
            driver, self._driver = self._driver, None
            if driver is not None:
                try:
                    driver.set_speeds(0.0, 0.0)
                except Exception:  # noqa: BLE001 - still release the driver below; its close() stops too
                    logger.exception("failed to stop the motors while closing the drive service")
            self.left, self.right, self.command = 0.0, 0.0, "stop"
            self._deadline = None
            watchdog, self._watchdog = self._watchdog, None
            self._watchdog_id += 1
            self._wake.notify_all()
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=2)
        if driver is not None:
            driver.close()

    # --- watchdog ----------------------------------------------------------

    def _ensure_watchdog(self) -> None:
        """Start the watchdog thread if it is not running. Caller holds _lock.

        Lazy on purpose: tests (and scripts) swap ``_driver`` in without
        calling start(), and a ttl must still be honoured there.
        """
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, args=(self._watchdog_id,), name="drive-watchdog", daemon=True
        )
        self._watchdog.start()

    def _watchdog_loop(self, watchdog_id: int) -> None:
        failures = 0
        with self._lock:
            while watchdog_id == self._watchdog_id:
                if self._deadline is None:
                    self._wake.wait()  # idle until a ttl is armed (or close())
                    continue
                remaining = self._deadline - time.monotonic()
                if remaining > 0:
                    self._wake.wait(remaining)  # a newer command re-arms and wakes us early
                    continue
                try:
                    if self._driver is not None:
                        self._driver.set_speeds(0.0, 0.0)
                except Exception:  # noqa: BLE001 - must never kill the watchdog
                    # Keep the deadline armed so the stop is retried; this
                    # thread is the last line of defence for the wheels.
                    failures += 1
                    if failures == 1 or failures % 50 == 0:
                        logger.exception("drive watchdog could not stop the motors (attempt %d); retrying", failures)
                    self._wake.wait(_WATCHDOG_RETRY_SECONDS)
                    continue
                failures = 0
                logger.warning("drive watchdog stopped the wheels (last command %r)", self.command)
                self.left, self.right, self.command = 0.0, 0.0, "stop"
                self._deadline = None
                self.watchdog_trips += 1


drive_service = DriveService()
