"""Shared camera: one capture thread, many readers.

Every successful read is stamped (sequence number + monotonic/wall time) and
stored as a :class:`FramePacket`, so consumers can prove a frame is fresh. The
rover controller relies on this: it must never act on a frame captured before
the wheels stopped, and a dead capture must read as "no frame", not as the
last good frame forever.
"""

import logging
import threading
import time

import numpy as np

from app.camera.base import CameraSource
from app.camera.webcam import WebcamCamera
from app.config import settings
from app.rover.types import FramePacket

logger = logging.getLogger(__name__)

# How long the capture thread backs off after a failed read.
_FAILURE_BACKOFF_SECONDS = 0.1
# Floor on the capture period. A webcam read blocks for a frame interval by
# itself; a synthetic source may return instantly, and without this the
# capture thread would spin a whole core (and starve the GIL on the Pi).
_MIN_FRAME_INTERVAL_SECONDS = 1 / 120
# wait_for_frame() re-checks its cancel event at least this often.
_CANCEL_POLL_SECONDS = 0.05
# How long stop() waits for the capture thread before releasing the device anyway.
_JOIN_TIMEOUT_SECONDS = 2.0


class _UseDefault:
    """Sentinel type: "use settings.camera_stale_after_seconds"."""


_USE_DEFAULT = _UseDefault()


class CameraService:
    def __init__(self, source: CameraSource | None = None) -> None:
        self._camera: CameraSource = source if source is not None else WebcamCamera(settings.camera_index)
        # Serialises start()/stop()/set_source(). Kept separate from _cond so
        # a slow device open/release never blocks readers or waiters.
        self._lifecycle_lock = threading.Lock()
        # One condition guards all capture state and wakes wait_for_frame().
        self._cond = threading.Condition()
        self._packet: FramePacket | None = None
        # Never reset, not even across stop()/start(): consumers remember the
        # last seq they used and wait for "seq > that".
        self._seq = 0
        self._consecutive_failures = 0
        self._last_failure_at: float | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        # Bumped by every start()/stop(). A capture thread that was stalled
        # inside a read when stop() gave up on it must not publish afterwards,
        # and a waiter must not sleep through a stop()/start() pair.
        self._run_id = 0
        self._available = False
        self._error: str | None = None

    # --- lifecycle ---------------------------------------------------------

    def set_source(self, source: CameraSource) -> None:
        """Swap the capture source (the simulation injects a synthetic one).
        Only allowed while stopped -- the capture thread owns the source."""
        with self._lifecycle_lock:
            if self._running:
                raise RuntimeError("cannot change the camera source while the camera is running")
            self._camera = source

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._running:
                return
            try:
                self._camera.start()
            except Exception as exc:  # noqa: BLE001 - a missing camera must not take the backend down
                with self._cond:
                    self._available = False
                    self._error = str(exc)
                logger.warning("camera unavailable: %s", exc)
                return

            with self._cond:
                self._available = True
                self._error = None
                self._running = True
                self._run_id += 1
                self._consecutive_failures = 0
                self._last_failure_at = None
                self._stop_event = threading.Event()
                self._thread = threading.Thread(
                    target=self._run,
                    args=(self._run_id, self._camera, self._stop_event),
                    name="camera-capture",
                    daemon=True,
                )
                self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            with self._cond:
                thread = self._thread
                self._thread = None
                self._running = False
                self._available = False
                self._run_id += 1
                # Frames from before a stop are never served again.
                self._packet = None
                self._stop_event.set()
                self._cond.notify_all()  # wake every wait_for_frame() caller now
            # Join before releasing: closing a capture device under a read
            # that is still in flight is not safe with every OpenCV backend.
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
            self._camera.stop()

    # --- capture thread ----------------------------------------------------

    def _run(self, run_id: int, camera: CameraSource, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            started = time.monotonic()
            failure: Exception | None = None
            try:
                frame = camera.get_frame()
            except Exception as exc:  # noqa: BLE001 - a flaky source must not kill the capture thread
                frame, failure = None, exc

            if frame is None:
                self._record_failure(run_id, failure)
                stop_event.wait(_FAILURE_BACKOFF_SECONDS)
                continue

            # Stamp when the read returned. This is an upper bound: the pixels
            # can be a little older (driver buffering), never newer, so callers
            # comparing against "wheels stopped at T" must add a settle margin.
            captured_at = time.monotonic()
            captured_wall = time.time()
            with self._cond:
                if run_id != self._run_id:
                    return  # stopped (or restarted) while this read was in flight
                if self._consecutive_failures:
                    logger.info("camera recovered after %d failed reads", self._consecutive_failures)
                self._seq += 1
                self._packet = FramePacket(
                    frame=frame, seq=self._seq, captured_at=captured_at, captured_wall=captured_wall
                )
                self._consecutive_failures = 0
                self._cond.notify_all()

            spare = _MIN_FRAME_INTERVAL_SECONDS - (time.monotonic() - started)
            if spare > 0:
                stop_event.wait(spare)

    def _record_failure(self, run_id: int, exc: Exception | None) -> None:
        # Deliberately leaves seq and the packet's timestamps alone: a failed
        # read must make the last frame look older, never fresher.
        with self._cond:
            if run_id != self._run_id:
                return
            self._consecutive_failures += 1
            self._last_failure_at = time.monotonic()
            failures = self._consecutive_failures
        # Reads fail ~10x a second while a camera is down: log the start of a
        # streak and then only occasionally.
        if failures == 1 or failures % 100 == 0:
            logger.warning("camera read failed (%d consecutive)%s", failures, f": {exc!r}" if exc else "")

    # --- readers -----------------------------------------------------------

    @staticmethod
    def _copy(packet: FramePacket) -> FramePacket:
        return FramePacket(
            frame=packet.frame.copy(),
            seq=packet.seq,
            captured_at=packet.captured_at,
            captured_wall=packet.captured_wall,
        )

    def latest_packet(self, max_age_seconds: float | None = None) -> FramePacket | None:
        """Latest packet (frame copied), or None when there is none or it is
        older than ``max_age_seconds``. ``None`` disables the age check."""
        with self._cond:
            packet = self._packet
            if packet is None:
                return None
            if max_age_seconds is not None and time.monotonic() - packet.captured_at > max_age_seconds:
                return None
            return self._copy(packet)

    def get_frame(self, max_age_seconds: float | None | _UseDefault = _USE_DEFAULT) -> np.ndarray | None:
        """Copy of the latest frame, or None when it is stale.

        The default limit is ``settings.camera_stale_after_seconds`` (a value
        <= 0 there disables the check), so a dead capture stops feeding old
        frames to the stream, chat and debug consumers. Pass ``None`` to get
        the latest frame regardless of age.
        """
        if isinstance(max_age_seconds, _UseDefault):
            configured = settings.camera_stale_after_seconds
            max_age_seconds = configured if configured > 0 else None
        packet = self.latest_packet(max_age_seconds)
        return None if packet is None else packet.frame

    def wait_for_frame(
        self,
        after_seq: int,
        not_before: float,
        timeout: float,
        cancel: threading.Event | None = None,
    ) -> FramePacket | None:
        """Block until a packet with ``seq > after_seq`` and ``captured_at >=
        not_before`` (monotonic) exists. Returns None on timeout, when
        ``cancel`` is set, or when the camera is stopped (nothing will come)."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            run_id = self._run_id
            while True:
                if cancel is not None and cancel.is_set():
                    return None
                packet = self._packet
                if packet is not None and packet.seq > after_seq and packet.captured_at >= not_before:
                    return self._copy(packet)
                if not self._running or run_id != self._run_id:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                # An Event cannot notify a Condition, so poll it in short slices.
                self._cond.wait(remaining if cancel is None else min(remaining, _CANCEL_POLL_SECONDS))

    # --- status ------------------------------------------------------------

    def is_available(self) -> bool:
        return self._available

    def error(self) -> str | None:
        return self._error

    def health(self) -> dict:
        now = time.monotonic()
        with self._cond:
            packet = self._packet
            return {
                "available": self._available,
                "error": self._error,
                "last_seq": self._seq,
                "last_frame_age_seconds": None if packet is None else max(0.0, now - packet.captured_at),
                "consecutive_failures": self._consecutive_failures,
                "last_failure_age_seconds": (
                    None if self._last_failure_at is None else max(0.0, now - self._last_failure_at)
                ),
            }


camera_service = CameraService()
