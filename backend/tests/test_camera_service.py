"""CameraService freshness guarantees, driven by a fake source the test controls.

No real camera: the fake delivers exactly one frame per ``push()`` and can be
made to fail, raise or stall on demand. Staleness is tested by shifting the
service's clock instead of sleeping.
"""

import threading
import time

import numpy as np
import pytest

import app.services.camera_service as camera_module
from app.camera.base import CameraSource
from app.config import settings
from app.rover.types import CameraPort, FramePacket
from app.services.camera_service import CameraService

WAIT = 5.0  # generous upper bound for things that normally take milliseconds


class FakeSource(CameraSource):
    """Delivers one frame per push(); otherwise the read stalls, like a hung device."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending = 0
        self._mode = "frames"  # "frames" | "fail" | "raise"
        self._released = False
        self._value = 0
        self.start_error: Exception | None = None
        self.started = 0
        self.stopped = 0
        self.stalled = False  # a read is currently hung waiting for a push()
        self.unblock_on_stop = True

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        with self._cond:
            self._released = False
        self.started += 1

    def get_frame(self) -> np.ndarray | None:
        with self._cond:
            while True:
                if self._released:
                    return None
                if self._mode == "fail":
                    return None
                if self._mode == "raise":
                    raise OSError("usb glitch")
                if self._pending > 0:
                    self._pending -= 1
                    self._value += 1
                    return np.full((4, 6, 3), self._value, dtype=np.uint8)
                self.stalled = True
                self._cond.wait()
                self.stalled = False

    def stop(self) -> None:
        self.stopped += 1
        if self.unblock_on_stop:
            self.unblock()

    # --- test controls ---

    def push(self, count: int = 1) -> None:
        with self._cond:
            self._pending += count
            self._cond.notify_all()

    def set_mode(self, mode: str) -> None:
        with self._cond:
            self._mode = mode
            self._cond.notify_all()

    def unblock(self) -> None:
        with self._cond:
            self._released = True
            self._cond.notify_all()


class ShiftedClock:
    """Stands in for the ``time`` module inside camera_service: real time plus
    an offset the test moves forward, so "two seconds later" costs nothing."""

    def __init__(self) -> None:
        self.offset = 0.0

    def monotonic(self) -> float:
        return time.monotonic() + self.offset

    def time(self) -> float:
        return time.time() + self.offset


def wait_until(predicate, timeout: float = WAIT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class Waiter:
    """Runs wait_for_frame() on a thread so the test can poke the service meanwhile."""

    def __init__(self, service: CameraService, **kwargs) -> None:
        self.result: FramePacket | None = None
        self.elapsed: float | None = None
        self.done = threading.Event()
        self._service = service
        self._kwargs = kwargs
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        start = time.monotonic()
        self.result = self._service.wait_for_frame(**self._kwargs)
        self.elapsed = time.monotonic() - start
        self.done.set()

    def is_blocked(self) -> bool:
        """True when the call has not returned after a short grace period."""
        return not self.done.wait(0.05)


@pytest.fixture
def source() -> FakeSource:
    return FakeSource()


@pytest.fixture
def service(source: FakeSource):
    svc = CameraService(source)
    svc.start()
    yield svc
    source.unblock()  # never leave stop() waiting on a stalled fake read
    svc.stop()


def next_packet(service: CameraService, source: FakeSource, after_seq: int) -> FramePacket:
    source.push()
    packet = service.wait_for_frame(after_seq=after_seq, not_before=0.0, timeout=WAIT)
    assert packet is not None
    return packet


# --- stamping -------------------------------------------------------------


def test_successful_reads_are_stamped_with_increasing_seq_and_time(service, source):
    before_mono, before_wall = time.monotonic(), time.time()
    first = next_packet(service, source, after_seq=0)
    second = next_packet(service, source, after_seq=first.seq)
    after_mono, after_wall = time.monotonic(), time.time()

    assert (first.seq, second.seq) == (1, 2)
    assert before_mono <= first.captured_at <= second.captured_at <= after_mono
    assert before_wall <= first.captured_wall <= second.captured_wall <= after_wall
    assert int(first.frame[0, 0, 0]) == 1 and int(second.frame[0, 0, 0]) == 2


def test_failed_reads_do_not_refresh_seq_or_timestamps(service, source):
    good = next_packet(service, source, after_seq=0)

    source.set_mode("fail")
    assert wait_until(lambda: service.health()["consecutive_failures"] >= 2)

    latest = service.latest_packet()
    assert latest is not None
    assert latest.seq == good.seq
    assert latest.captured_at == good.captured_at
    assert latest.captured_wall == good.captured_wall
    health = service.health()
    assert health["last_seq"] == good.seq
    assert health["last_failure_age_seconds"] is not None

    source.set_mode("frames")
    recovered = next_packet(service, source, after_seq=good.seq)
    assert recovered.seq == good.seq + 1
    assert service.health()["consecutive_failures"] == 0


def test_a_raising_source_counts_as_failure_and_does_not_kill_capture(service, source):
    good = next_packet(service, source, after_seq=0)

    source.set_mode("raise")
    assert wait_until(lambda: service.health()["consecutive_failures"] >= 2)
    assert service.latest_packet().seq == good.seq

    source.set_mode("frames")
    assert next_packet(service, source, after_seq=good.seq).seq == good.seq + 1


# --- staleness ------------------------------------------------------------


def test_get_frame_goes_none_once_the_latest_frame_is_stale(service, source, monkeypatch):
    clock = ShiftedClock()
    monkeypatch.setattr(camera_module, "time", clock)
    monkeypatch.setattr(settings, "camera_stale_after_seconds", 2.0)
    packet = next_packet(service, source, after_seq=0)

    clock.offset = 1.0
    assert service.get_frame() is not None

    # The source is now stalled: no new frames, and "time passes".
    clock.offset = 2.5
    assert service.get_frame() is None
    assert service.latest_packet(max_age_seconds=2.0) is None
    assert service.health()["last_frame_age_seconds"] >= 2.5

    # Explicit opt-outs still see the old frame, clearly identified by its seq.
    assert service.get_frame(max_age_seconds=None) is not None
    assert service.get_frame(max_age_seconds=10.0) is not None
    assert service.latest_packet().seq == packet.seq

    # A new successful read makes the camera fresh again.
    next_packet(service, source, after_seq=packet.seq)
    assert service.get_frame() is not None


def test_a_dead_capture_stops_serving_the_old_frame(service, source, monkeypatch):
    clock = ShiftedClock()
    monkeypatch.setattr(camera_module, "time", clock)
    next_packet(service, source, after_seq=0)

    source.set_mode("fail")
    assert wait_until(lambda: service.health()["consecutive_failures"] >= 1)
    clock.offset = settings.camera_stale_after_seconds + 1.0

    assert service.get_frame() is None
    assert service.is_available()  # opened, but producing nothing: health() tells the story
    assert service.health()["consecutive_failures"] >= 1


def test_stale_limit_of_zero_in_settings_disables_the_default_check(service, source, monkeypatch):
    clock = ShiftedClock()
    monkeypatch.setattr(camera_module, "time", clock)
    monkeypatch.setattr(settings, "camera_stale_after_seconds", 0.0)
    next_packet(service, source, after_seq=0)
    clock.offset = 3600.0
    assert service.get_frame() is not None


def test_readers_get_independent_copies(service, source):
    next_packet(service, source, after_seq=0)
    frame = service.get_frame()
    frame[:] = 99
    assert int(service.get_frame()[0, 0, 0]) == 1
    packet = service.latest_packet()
    packet.frame[:] = 77
    assert int(service.latest_packet().frame[0, 0, 0]) == 1


# --- wait_for_frame -------------------------------------------------------


def test_wait_for_frame_requires_a_newer_seq(service, source):
    first = next_packet(service, source, after_seq=0)

    # The frame that is already there does not count.
    start = time.monotonic()
    assert service.wait_for_frame(after_seq=first.seq, not_before=0.0, timeout=0.05) is None
    assert 0.04 <= time.monotonic() - start < 2.0

    waiter = Waiter(service, after_seq=first.seq, not_before=0.0, timeout=WAIT)
    assert waiter.is_blocked()
    source.push()
    assert waiter.done.wait(WAIT)
    assert waiter.result is not None and waiter.result.seq == first.seq + 1


def test_wait_for_frame_ignores_frames_captured_before_not_before(service, source):
    early = next_packet(service, source, after_seq=0)

    # seq is new enough, but the frame predates the moment the caller cares about.
    boundary = time.monotonic()
    assert early.captured_at <= boundary
    waiter = Waiter(service, after_seq=0, not_before=boundary, timeout=WAIT)
    assert waiter.is_blocked()

    source.push()
    assert waiter.done.wait(WAIT)
    assert waiter.result is not None
    assert waiter.result.seq == early.seq + 1
    assert waiter.result.captured_at >= boundary


def test_wait_for_frame_times_out_when_every_frame_is_too_early(service, source):
    far_future = time.monotonic() + 1000.0
    source.push(3)
    assert wait_until(lambda: service.health()["last_seq"] == 3)
    start = time.monotonic()
    assert service.wait_for_frame(after_seq=0, not_before=far_future, timeout=0.1) is None
    assert 0.09 <= time.monotonic() - start < 2.0


def test_wait_for_frame_returns_none_promptly_when_cancelled(service, source):
    cancel = threading.Event()
    waiter = Waiter(service, after_seq=0, not_before=0.0, timeout=60.0, cancel=cancel)
    assert waiter.is_blocked()

    cancel.set()
    assert waiter.done.wait(WAIT)
    assert waiter.result is None
    assert waiter.elapsed < 2.0  # nowhere near the 60 s timeout


def test_wait_for_frame_with_cancel_already_set_never_returns_a_frame(service, source):
    next_packet(service, source, after_seq=0)
    cancel = threading.Event()
    cancel.set()
    assert service.wait_for_frame(after_seq=0, not_before=0.0, timeout=WAIT, cancel=cancel) is None


def test_wait_for_frame_still_delivers_when_a_cancel_event_is_supplied(service, source):
    waiter = Waiter(service, after_seq=0, not_before=0.0, timeout=WAIT, cancel=threading.Event())
    assert waiter.is_blocked()
    source.push()
    assert waiter.done.wait(WAIT)
    assert waiter.result is not None and waiter.result.seq == 1


def test_one_frame_wakes_every_waiter(service, source):
    waiters = [Waiter(service, after_seq=0, not_before=0.0, timeout=WAIT) for _ in range(4)]
    source.push()
    for waiter in waiters:
        assert waiter.done.wait(WAIT)
        assert waiter.result is not None and waiter.result.seq == 1


# --- stop / restart / sources ---------------------------------------------


def test_stop_wakes_waiters_even_while_the_capture_read_is_stalled(source, monkeypatch):
    # stop() will sit in its join for as long as the read stays hung; make that
    # far longer than the waiter is allowed, so only the wake-up can pass this.
    monkeypatch.setattr(camera_module, "_JOIN_TIMEOUT_SECONDS", 30.0)
    service = CameraService(source)
    service.start()
    source.unblock_on_stop = False  # the read stays hung; only the service can wake the waiter
    assert wait_until(lambda: source.stalled)
    waiter = Waiter(service, after_seq=0, not_before=0.0, timeout=60.0)
    assert waiter.is_blocked()

    stopper = threading.Thread(target=service.stop, daemon=True)
    stopper.start()
    try:
        assert waiter.done.wait(WAIT)  # woken by stop() itself, not by the hung read ending
        assert waiter.result is None
        assert stopper.is_alive()  # stop() is still stuck behind the hung read
    finally:
        source.unblock()
        stopper.join(WAIT)
    assert not stopper.is_alive()


def test_stop_forgets_the_last_frame_and_refuses_new_waits(service, source):
    next_packet(service, source, after_seq=0)
    service.stop()

    assert source.stopped == 1
    assert not service.is_available()
    assert service.get_frame(max_age_seconds=None) is None
    assert service.latest_packet() is None
    start = time.monotonic()
    assert service.wait_for_frame(after_seq=0, not_before=0.0, timeout=30.0) is None
    assert time.monotonic() - start < 2.0  # a stopped camera will never deliver: do not sit out the timeout
    assert service.health()["available"] is False


def test_seq_keeps_increasing_across_a_restart(service, source):
    first = next_packet(service, source, after_seq=0)
    service.stop()
    service.start()
    assert source.started == 2
    again = next_packet(service, source, after_seq=first.seq)
    assert again.seq == first.seq + 1


def test_a_read_that_outlives_stop_never_publishes(source, monkeypatch):
    monkeypatch.setattr(camera_module, "_JOIN_TIMEOUT_SECONDS", 0.05)
    service = CameraService(source)
    service.start()
    first = next_packet(service, source, after_seq=0)

    source.unblock_on_stop = False
    assert wait_until(lambda: source.stalled)  # the capture thread is back inside a hung read
    old_thread = service._thread
    service.stop()  # gives up on the hung read after 50 ms
    assert old_thread.is_alive()

    source.push()  # the hung read finally returns a frame...
    old_thread.join(WAIT)
    assert not old_thread.is_alive()
    assert service.latest_packet() is None  # ...which is dropped
    assert service.health()["last_seq"] == first.seq
    source.unblock()


def test_start_is_idempotent(service, source):
    thread = service._thread
    service.start()
    assert service._thread is thread
    assert source.started == 1


def test_unavailable_camera_reports_the_error_and_can_be_retried(source):
    source.start_error = RuntimeError("could not open camera at index 0")
    service = CameraService(source)
    service.start()

    assert not service.is_available()
    assert service.error() == "could not open camera at index 0"
    assert service.get_frame() is None
    start = time.monotonic()
    assert service.wait_for_frame(after_seq=0, not_before=0.0, timeout=30.0) is None
    assert time.monotonic() - start < 2.0
    health = service.health()
    assert health["available"] is False and health["error"] == "could not open camera at index 0"

    source.start_error = None
    service.start()
    try:
        assert service.is_available() and service.error() is None
        assert next_packet(service, source, after_seq=0).seq == 1
    finally:
        source.unblock()
        service.stop()


def test_set_source_only_while_stopped(service, source):
    replacement = FakeSource()
    with pytest.raises(RuntimeError):
        service.set_source(replacement)

    first = next_packet(service, source, after_seq=0)
    service.stop()
    service.set_source(replacement)
    service.start()
    try:
        assert replacement.started == 1
        packet = next_packet(service, replacement, after_seq=first.seq)
        assert packet.seq == first.seq + 1
        assert int(packet.frame[0, 0, 0]) == 1  # the replacement's first frame, not the old source's
    finally:
        replacement.unblock()


def test_health_shape_and_camera_port(service, source):
    assert isinstance(service, CameraPort)
    assert service.health() == {
        "available": True,
        "error": None,
        "last_seq": 0,
        "last_frame_age_seconds": None,
        "consecutive_failures": 0,
        "last_failure_age_seconds": None,
    }
    next_packet(service, source, after_seq=0)
    health = service.health()
    assert health["last_seq"] == 1
    assert 0.0 <= health["last_frame_age_seconds"] < WAIT


def test_module_singleton_is_a_camera_service():
    assert isinstance(camera_module.camera_service, CameraService)
