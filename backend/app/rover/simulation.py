"""A small simulated world for tests and the laptop demo.

EVERYTHING in this module is simulated: the rover pose, the camera frames and
the detections. It exists so the controller's behaviour (scan, candidate
questions, rejection memory, centring, approach, arrival, every stop path) can
be exercised repeatably without a camera, a network or motors. It says nothing
about how the physical rover will behave -- turn rates, forward speed and
apparent sizes here are made-up numbers.

``SimVision.is_simulated`` is True, and the controller refuses to move whenever
simulated vision is paired with a motor driver that is not the sim driver, so
nothing in here can ever turn a real wheel.

Conventions: world units are metres; +y is heading 0 and +x is heading 90
(headings grow clockwise, so turning RIGHT increases the heading and slides
everything in view to the LEFT). Objects stand on the floor.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.camera.base import CameraSource
from app.config import settings
from app.rover.config import RoverConfig
from app.rover.geometry import area, is_valid_box
from app.rover.types import (
    Box,
    CameraPort,
    DrivePort,
    FramePacket,
    IdentifyRequest,
    IdentifyResult,
    LocalDetection,
    VisionPort,
)

logger = logging.getLogger(__name__)

# A pulse that was stopped at >= this fraction of its nominal time is
# integrated at exactly the nominal time, so OS timer jitter does not leak
# into the simulated pose (tests stay deterministic).
_NOMINAL_TOLERANCE = 0.9
_MIN_DEPTH_METRES = 0.05


@dataclass
class SimObject:
    id: str
    coco_class: str
    description: str
    x: float
    y: float
    width: float                       # physical size, metres
    height: float
    is_target: bool = False            # ground truth: THE item the user means
    matches_query: bool = True         # looks like the description, so identify() may report it
    confidence: float = 0.85           # what the simulated local detector reports
    color: tuple[int, int, int] = (200, 120, 40)   # BGR, for the rendered frame


@dataclass(frozen=True)
class Projection:
    obj: SimObject
    box: Box                 # clipped to the frame
    visible_fraction: float  # visible area / full projected area


@dataclass
class _Motion:
    command: str
    speed: float
    ttl: float | None
    started: float
    origin: tuple[float, float, float]


class SimWorld:
    """Pose + objects + a pinhole camera. One lock guards all of it."""

    def __init__(
        self,
        *,
        hfov_degrees: float = 62.0,
        frame_size: tuple[int, int] = (640, 480),
        camera_height: float = 0.12,
        turn_degrees_per_second: float = 85.0,
        reference_turn_speed: float = 0.4,
        forward_metres_per_second: float = 0.25,
        reference_forward_speed: float = 0.45,
        watchdog_margin_seconds: float = 0.15,
    ) -> None:
        self.lock = threading.RLock()
        self.hfov_degrees = hfov_degrees
        self.frame_size = frame_size
        self.camera_height = camera_height
        # Rates hold at the reference speeds and scale linearly with speed.
        self.turn_degrees_per_second = turn_degrees_per_second
        self.reference_turn_speed = reference_turn_speed
        self.forward_metres_per_second = forward_metres_per_second
        self.reference_forward_speed = reference_forward_speed
        # SimDrive is told ttl = pulse + margin; knowing the margin lets it
        # recover the nominal pulse time.
        self.watchdog_margin_seconds = watchdog_margin_seconds
        self.stuck = False            # wheels spin, the rover does not advance (stall tests)
        self.turn_stuck = False       # wheels spin, the rover does not rotate (turn-stall tests)
        self.watchdog_trips = 0
        self.objects: list[SimObject] = []
        self._pose = (0.0, 0.0, 0.0)  # x, y, heading (degrees)
        self._motion: _Motion | None = None

    @classmethod
    def for_config(cls, config: RoverConfig, **kwargs) -> "SimWorld":
        """A world whose (made-up) physics agree with the controller's
        calibration constants, so dead-reckoning and reality line up."""
        params = dict(
            hfov_degrees=config.camera_hfov_degrees,
            turn_degrees_per_second=config.turn_degrees_per_second,
            reference_turn_speed=config.turn_speed,
            reference_forward_speed=config.forward_speed,
            watchdog_margin_seconds=config.pulse_watchdog_margin_seconds,
        )
        params.update(kwargs)
        return cls(**params)

    # --- objects ------------------------------------------------------------

    def add_object(self, obj: SimObject) -> SimObject:
        with self.lock:
            self.objects.append(obj)
        return obj

    def add_at(self, obj_id: str, *, bearing: float, distance: float, **fields) -> SimObject:
        """Place an object by bearing (degrees clockwise from heading 0) and
        distance from the origin."""
        rad = math.radians(bearing)
        return self.add_object(SimObject(id=obj_id, x=distance * math.sin(rad), y=distance * math.cos(rad), **fields))

    def remove_object(self, obj_id: str) -> None:
        with self.lock:
            self.objects = [o for o in self.objects if o.id != obj_id]

    def get(self, obj_id: str) -> SimObject | None:
        with self.lock:
            return next((o for o in self.objects if o.id == obj_id), None)

    def target(self) -> SimObject | None:
        with self.lock:
            return next((o for o in self.objects if o.is_target), None)

    # --- pose & motion --------------------------------------------------------

    def set_pose(self, x: float, y: float, heading: float) -> None:
        with self.lock:
            self._motion = None
            self._pose = (x, y, heading)

    def pose(self) -> tuple[float, float, float]:
        """Current pose, including any motion in progress."""
        with self.lock:
            if self._motion is None:
                return self._pose
            motion = self._motion
            elapsed = time.monotonic() - motion.started
            if motion.ttl is not None:
                elapsed = min(elapsed, motion.ttl)
            return self._integrate(motion, elapsed)

    def is_moving(self) -> bool:
        with self.lock:
            motion = self._motion
            if motion is None:
                return False
            return motion.ttl is None or (time.monotonic() - motion.started) < motion.ttl

    def apply_command(self, command: str, speed: float | None, ttl: float | None) -> None:
        with self.lock:
            self.halt()
            if command == "stop":
                return
            self._motion = _Motion(
                command=command,
                speed=1.0 if speed is None else max(0.0, min(1.0, speed)),
                ttl=ttl,
                started=time.monotonic(),
                origin=self._pose,
            )

    def halt(self) -> None:
        """Settle the motion in progress into the pose."""
        with self.lock:
            motion = self._motion
            if motion is None:
                return
            elapsed = time.monotonic() - motion.started
            if motion.ttl is not None:
                nominal = max(0.0, motion.ttl - self.watchdog_margin_seconds)
                if elapsed >= motion.ttl:
                    # Nobody stopped us in time: the (simulated) watchdog did.
                    elapsed = motion.ttl
                    self.watchdog_trips += 1
                elif elapsed >= nominal * _NOMINAL_TOLERANCE:
                    elapsed = nominal
            self._pose = self._integrate(motion, elapsed)
            self._motion = None

    def _integrate(self, motion: _Motion, elapsed: float) -> tuple[float, float, float]:
        x, y, heading = motion.origin
        turn_rate = (
            0.0 if self.turn_stuck else self.turn_degrees_per_second * motion.speed / max(self.reference_turn_speed, 1e-6)
        )
        speed = 0.0 if self.stuck else self.forward_metres_per_second * motion.speed / max(self.reference_forward_speed, 1e-6)
        if motion.command == "turn_right":
            heading += turn_rate * elapsed
        elif motion.command == "turn_left":
            heading -= turn_rate * elapsed
        elif motion.command in ("forward", "backward", "veer_left", "veer_right"):
            sign = -1.0 if motion.command == "backward" else 1.0
            rad = math.radians(heading)
            x += sign * speed * elapsed * math.sin(rad)
            y += sign * speed * elapsed * math.cos(rad)
            if motion.command == "veer_left":
                heading -= 0.25 * turn_rate * elapsed
            elif motion.command == "veer_right":
                heading += 0.25 * turn_rate * elapsed
        return (x, y, heading)

    def set_forward_step(self, metres: float, pulse_seconds: float) -> None:
        """Make one forward pulse of ``pulse_seconds`` at the reference forward
        speed cover ``metres`` (slow-zone pulses scale down from there). For
        "far target / slow rover" tests: growth per pulse is ~ step / distance."""
        if not (metres >= 0 and pulse_seconds > 0):
            raise ValueError("metres must be >= 0 and pulse_seconds > 0")
        with self.lock:
            self.forward_metres_per_second = metres / pulse_seconds

    def distance_to(self, obj_id: str) -> float:
        obj = self.get(obj_id)
        if obj is None:
            raise KeyError(obj_id)
        x, y, _ = self.pose()
        return math.hypot(obj.x - x, obj.y - y)

    def relative_bearing(self, obj_id: str) -> float:
        """Degrees from the camera axis to the object; negative = to the left."""
        obj = self.get(obj_id)
        if obj is None:
            raise KeyError(obj_id)
        x, y, heading = self.pose()
        bearing = math.degrees(math.atan2(obj.x - x, obj.y - y))
        return (bearing - heading + 180.0) % 360.0 - 180.0

    # --- camera model -----------------------------------------------------------

    def project(self, obj: SimObject) -> Projection | None:
        x, y, heading = self.pose()
        rad = math.radians(heading)
        dx, dy = obj.x - x, obj.y - y
        forward = dx * math.sin(rad) + dy * math.cos(rad)
        right = dx * math.cos(rad) - dy * math.sin(rad)
        if forward <= _MIN_DEPTH_METRES:
            return None

        frame_w, frame_h = self.frame_size
        tan_h = math.tan(math.radians(self.hfov_degrees / 2))
        tan_v = tan_h * frame_h / frame_w
        cx = 0.5 + (right / forward) / (2 * tan_h)
        half_w = (obj.width / forward) / (2 * tan_h) / 2
        top = 0.5 + ((self.camera_height - obj.height) / forward) / (2 * tan_v)
        bottom = 0.5 + (self.camera_height / forward) / (2 * tan_v)
        xmin, xmax = cx - half_w, cx + half_w

        vis_xmin, vis_xmax = max(0.0, xmin), min(1.0, xmax)
        vis_top, vis_bottom = max(0.0, top), min(1.0, bottom)
        if vis_xmax <= vis_xmin or vis_bottom <= vis_top:
            return None
        full_area = (xmax - xmin) * (bottom - top)
        visible_fraction = ((vis_xmax - vis_xmin) * (vis_bottom - vis_top)) / full_area if full_area > 0 else 0.0
        box = [round(vis_top * 1000), round(vis_xmin * 1000), round(vis_bottom * 1000), round(vis_xmax * 1000)]
        if not is_valid_box(box):
            return None
        return Projection(obj=obj, box=box, visible_fraction=visible_fraction)

    def visible(self, min_fraction: float = 0.6) -> list[Projection]:
        with self.lock:
            projections = [self.project(obj) for obj in self.objects]
        return [p for p in projections if p is not None and p.visible_fraction >= min_fraction]

    def render(self) -> np.ndarray:
        """A synthetic BGR frame: wall, floor, one filled rectangle per object."""
        frame_w, frame_h = self.frame_size
        frame = np.full((frame_h, frame_w, 3), (225, 220, 210), dtype=np.uint8)
        frame[frame_h // 2 :, :] = (150, 160, 170)
        with self.lock:
            heading = self.pose()[2]
            projections = [p for p in (self.project(obj) for obj in self.objects) if p is not None]
        # Far objects first, so nearer ones are drawn over them.
        projections.sort(key=lambda p: area(p.box))
        for projection in projections:
            ymin, xmin, ymax, xmax = projection.box
            left, right = round(xmin / 1000 * frame_w), round(xmax / 1000 * frame_w)
            top, bottom = round(ymin / 1000 * frame_h), round(ymax / 1000 * frame_h)
            cv2.rectangle(frame, (left, top), (right, bottom), projection.obj.color, thickness=-1)
            cv2.rectangle(frame, (left, top), (right, bottom), (40, 40, 40), thickness=1)
            cv2.putText(
                frame, projection.obj.id, (left, max(12, top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1, cv2.LINE_AA
            )
        cv2.putText(
            frame, f"SIMULATION  heading {heading % 360:.0f} deg", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 160), 1, cv2.LINE_AA
        )
        return frame

    # --- ports ------------------------------------------------------------------

    def make_drive(self) -> "SimDrive":
        return SimDrive(self)

    def make_camera(self, **kwargs) -> "SimCamera":
        return SimCamera(self, **kwargs)

    def make_vision(self, **kwargs) -> "SimVision":
        return SimVision(self, **kwargs)


class SimDrive:
    """``DrivePort`` that moves the simulated rover. Honours ``ttl`` itself: a
    command nobody stops ends at its deadline."""

    driver_name = "sim"

    def __init__(self, world: SimWorld) -> None:
        self._world = world

    def drive(self, command: str, speed: float | None = None, ttl: float | None = None) -> None:
        self._world.apply_command(command, speed, ttl)

    def stop(self) -> None:
        self._world.halt()


@dataclass(frozen=True)
class DriveRecord:
    command: str            # "stop" for stop()
    speed: float | None
    ttl: float | None
    at: float               # time.monotonic() when the call arrived


class RecordingDrive:
    """``DrivePort`` wrapper that records every call, for asserting on the
    exact motor commands a mission issued. ``driver_name`` can be overridden
    to impersonate a real driver (the wrapped port still only simulates)."""

    def __init__(self, inner: DrivePort, driver_name: str | None = None) -> None:
        self._inner = inner
        self._driver_name = driver_name
        self._cond = threading.Condition()
        self._records: list[DriveRecord] = []

    @property
    def driver_name(self) -> str:
        return self._driver_name if self._driver_name is not None else self._inner.driver_name

    def drive(self, command: str, speed: float | None = None, ttl: float | None = None) -> None:
        self._record(DriveRecord(command, speed, ttl, time.monotonic()))
        self._inner.drive(command, speed, ttl=ttl)

    def stop(self) -> None:
        self._record(DriveRecord("stop", None, None, time.monotonic()))
        self._inner.stop()

    def _record(self, record: DriveRecord) -> None:
        with self._cond:
            self._records.append(record)
            self._cond.notify_all()

    @property
    def records(self) -> list[DriveRecord]:
        with self._cond:
            return list(self._records)

    @property
    def moves(self) -> list[DriveRecord]:
        """Everything except stops: the commands that actually turn wheels."""
        return [r for r in self.records if r.command != "stop"]

    def wait_for_moves(self, count: int = 1, timeout: float = 5.0) -> bool:
        with self._cond:
            return self._cond.wait_for(
                lambda: sum(1 for r in self._records if r.command != "stop") >= count, timeout=timeout
            )


class TeeDrive:
    """Sends every command to the real drive service (so the UI and the
    watchdog behave as in production) and mirrors it into the sim world."""

    def __init__(self, primary: DrivePort, mirror: DrivePort) -> None:
        self._primary = primary
        self._mirror = mirror

    @property
    def driver_name(self) -> str:
        return self._primary.driver_name

    def drive(self, command: str, speed: float | None = None, ttl: float | None = None) -> None:
        self._primary.drive(command, speed, ttl=ttl)
        self._mirror.drive(command, speed, ttl=ttl)

    def stop(self) -> None:
        try:
            self._primary.stop()
        finally:
            self._mirror.stop()


class SimCamera(CameraSource):
    """Synthetic camera. Works directly as the controller's ``CameraPort``
    (frames are rendered on demand with increasing ``seq`` and real monotonic
    timestamps) and as a ``CameraSource`` for ``camera_service`` in the demo."""

    def __init__(self, world: SimWorld, *, fps: float = 15.0) -> None:
        self._world = world
        self._fps = fps
        self._lock = threading.Lock()
        self._seq = 0
        self._latest: FramePacket | None = None
        self.fail = False          # no frames at all (unplugged camera)
        self.fail_after: int | None = None   # ...after this many captured frames
        self.frozen = False        # keeps handing out the last frame (a stuck driver)
        # A capture that DIED (see die()): the last frame it produced is kept,
        # exactly like camera_service keeps its last packet after read failures.
        self.dead = False
        self.dead_ignores_not_before = False

    # CameraPort ----------------------------------------------------------------

    def _capture(self) -> FramePacket:
        with self._lock:
            self._seq += 1
            packet = FramePacket(
                frame=self._world.render(), seq=self._seq, captured_at=time.monotonic(), captured_wall=time.time()
            )
            self._latest = packet
            return packet

    def _failing(self) -> bool:
        return self.fail or (self.fail_after is not None and self._seq >= self.fail_after)

    def die(self, *, ignore_not_before: bool = False) -> FramePacket:
        """The capture dies NOW, the way a real one does: one last frame was
        captured a moment ago (new seq, genuine timestamp) and no frame ever
        follows. ``wait_for_frame`` keeps offering that last packet for as long
        as it satisfies the port contract (seq > after_seq and captured_at >=
        not_before) -- it has no idea how OLD the packet is, which is precisely
        what the controller's frame-age bound has to catch. With
        ``ignore_not_before`` the port even breaks the contract and hands the
        packet out regardless of ``not_before``. Returns that last packet."""
        packet = self._capture()
        self.dead_ignores_not_before = ignore_not_before
        self.dead = True
        return packet

    def revive(self) -> None:
        self.dead = False
        self.dead_ignores_not_before = False

    def latest_packet(self, max_age_seconds: float | None = None) -> FramePacket | None:
        if self._failing():
            return None
        keep_last = (self.frozen or self.dead) and self._latest is not None
        packet = self._latest if keep_last else self._capture()
        if max_age_seconds is not None and time.monotonic() - packet.captured_at > max_age_seconds:
            return None
        return packet

    @staticmethod
    def _wait_until(when: float, cancel: threading.Event | None) -> bool:
        """False when cancelled before ``when``."""
        while True:
            remaining = when - time.monotonic()
            if remaining <= 0:
                return True
            if cancel is not None:
                if cancel.wait(remaining):
                    return False
            else:
                time.sleep(remaining)

    def wait_for_frame(
        self,
        after_seq: int,
        not_before: float,
        timeout: float,
        cancel: threading.Event | None = None,
    ) -> FramePacket | None:
        deadline = time.monotonic() + timeout
        if cancel is not None and cancel.is_set():
            return None
        if self.dead:
            last = self._latest
            if last is not None and last.seq > after_seq and (
                self.dead_ignores_not_before or last.captured_at >= not_before
            ):
                return last        # "fresh" by seq and not_before, however old it really is
            self._wait_until(deadline, cancel)
            return None
        if self._failing() or not_before > deadline:
            self._wait_until(deadline, cancel)
            return None
        if not self._wait_until(not_before, cancel):
            return None
        if self.frozen and self._latest is not None:
            return self._latest  # deliberately violates the freshness contract
        return self._capture()

    # CameraSource (for camera_service in the demo server) ------------------------

    def start(self) -> None:
        return None

    def get_frame(self) -> np.ndarray | None:
        # The capture loop calls this back to back; a real camera blocks for
        # one frame interval, so do the same.
        time.sleep(1.0 / self._fps)
        if self._failing() or self.dead:
            return None
        return self._world.render()

    def stop(self) -> None:
        return None


@dataclass
class VisionCall:
    kind: str        # "screen" | "identify"
    frame_seq: int
    at: float        # time.monotonic() when the call started


@dataclass
class SimVision:
    """Simulated recognition bound to a ``SimWorld``.

    ``identify`` reports the single strongest visible object that matches the
    query (like OMNI: one candidate, no alternatives) and honours ``rejected``
    hints by description. With a ``confirmed_crop`` it looks for the SAME
    physical object that was reported when the user accepted. The public
    attributes below inject failures.
    """

    world: SimWorld
    min_visible_fraction: float = 0.6
    latency_seconds: float = 0.0
    likelihood: str = "high"
    screening_available: bool = True     # False -> screen() returns None (no local detector)
    screen_blind: bool = False           # True -> the local detector sees nothing
    found_without_box: bool = False      # identify() says found=true but gives no location
    ignore_rejections: bool = False      # a model that does not follow the rejected hints
    fail_with: Exception | None = None   # identify() raises this...
    fail_times: int | None = None        # ...this many times (None = every time)
    block_event: threading.Event | None = None   # identify() hangs until this is set
    wrong_object_after: int | None = None        # after N confirmed-target calls, report another object
    # Detector noise. Real boxes are never pixel-stable, and two models never
    # agree on a box; the controller's progress checks must survive both.
    box_jitter: int = 0                  # +/- this many 0-1000 units on every reported box edge
    box_jitter_seed: int | None = None   # None -> alternately shrunk / exact; int -> seeded random per edge
    identify_box_scale: float = 1.0      # OMNI boxes systematically looser (>1) / tighter (<1) than YOLO's
    is_simulated: bool = True

    # Observability for tests: set when an identify() call starts; every call
    # and every identify request is logged.
    entered_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    calls: list[VisionCall] = field(default_factory=list, init=False, repr=False)
    requests: list[IdentifyRequest] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _last_reported_id: str | None = field(default=None, init=False, repr=False)
    _confirmed_id: str | None = field(default=None, init=False, repr=False)
    _confirmed_calls: int = field(default=0, init=False, repr=False)
    _jitter_tick: int = field(default=0, init=False, repr=False)
    _jitter_rng: random.Random | None = field(default=None, init=False, repr=False)

    @property
    def identify_calls(self) -> int:
        with self._lock:
            return sum(1 for call in self.calls if call.kind == "identify")

    @property
    def screen_calls(self) -> int:
        with self._lock:
            return sum(1 for call in self.calls if call.kind == "screen")

    def release(self) -> None:
        """Unblock every hanging identify() call (test teardown)."""
        if self.block_event is not None:
            self.block_event.set()

    def _noisy(self, box: Box, scale: float = 1.0) -> Box:
        """``box`` as a real detector would report it: optionally scaled about
        its centre, then jittered. Falls back to the exact box if the noise
        would make it invalid (tiny boxes)."""
        ymin, xmin, ymax, xmax = box
        if scale != 1.0:
            cy, cx = (ymin + ymax) / 2, (xmin + xmax) / 2
            half_h, half_w = (ymax - ymin) / 2 * scale, (xmax - xmin) / 2 * scale
            ymin, xmin, ymax, xmax = round(cy - half_h), round(cx - half_w), round(cy + half_h), round(cx + half_w)
        amp = int(self.box_jitter)
        if amp > 0:
            with self._lock:
                if self.box_jitter_seed is None:
                    # Deterministic worst case for a "compare with the previous
                    # frame" progress check: shrunk, exact, shrunk, exact, ...
                    k = amp if self._jitter_tick % 2 == 0 else 0
                    deltas = (k, k, -k, -k)
                else:
                    if self._jitter_rng is None:
                        self._jitter_rng = random.Random(self.box_jitter_seed)
                    deltas = tuple(self._jitter_rng.randint(-amp, amp) for _ in range(4))
                self._jitter_tick += 1
            ymin, xmin, ymax, xmax = ymin + deltas[0], xmin + deltas[1], ymax + deltas[2], xmax + deltas[3]
        noisy = [max(0, min(1000, int(v))) for v in (ymin, xmin, ymax, xmax)]
        return noisy if is_valid_box(noisy) else list(box)

    # VisionPort -------------------------------------------------------------------

    def screen(self, packet: FramePacket, coco_class: str) -> list[LocalDetection] | None:
        with self._lock:
            self.calls.append(VisionCall("screen", packet.seq, time.monotonic()))
        if not self.screening_available:
            return None
        if self.screen_blind:
            return []
        detections = [
            LocalDetection(box=self._noisy(p.box), confidence=p.obj.confidence)
            for p in self.world.visible(self.min_visible_fraction)
            if p.obj.coco_class == coco_class
        ]
        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections

    def identify(self, packet: FramePacket, request: IdentifyRequest) -> IdentifyResult:
        started = time.monotonic()
        with self._lock:
            self.calls.append(VisionCall("identify", packet.seq, started))
            self.requests.append(request)
        self.entered_event.set()

        block = self.block_event
        if block is not None:
            block.wait()
        if self.latency_seconds > 0:
            time.sleep(self.latency_seconds)

        with self._lock:
            if self.fail_with is not None and (self.fail_times is None or self.fail_times > 0):
                if self.fail_times is not None:
                    self.fail_times -= 1
                raise self.fail_with
            chosen = self._choose(request)
            if chosen is not None:
                self._last_reported_id = chosen.obj.id

        latency = time.monotonic() - started
        if chosen is None:
            return IdentifyResult(
                found=False, likelihood="low", box=None, description="nothing matching in view",
                latency_seconds=latency, frame_seq=packet.seq,
            )
        return IdentifyResult(
            found=True,
            likelihood=self.likelihood,
            box=None if self.found_without_box else self._noisy(chosen.box, self.identify_box_scale),
            description=chosen.obj.description,
            latency_seconds=latency,
            frame_seq=packet.seq,
        )

    def _choose(self, request: IdentifyRequest) -> Projection | None:
        # The rover is stationary whenever it looks, so "what is visible now"
        # is what the frame shows.
        matching = [p for p in self.world.visible(self.min_visible_fraction) if p.obj.matches_query]

        if request.confirmed_crop is not None:
            if self._confirmed_id is None:
                target = self.world.target()
                self._confirmed_id = self._last_reported_id or (target.id if target else None)
            self._confirmed_calls += 1
            if self.wrong_object_after is not None and self._confirmed_calls > self.wrong_object_after:
                matching = [p for p in matching if p.obj.id != self._confirmed_id]
            else:
                matching = [p for p in matching if p.obj.id == self._confirmed_id]
        elif not self.ignore_rejections:
            rejected = {hint.description for hint in request.rejected}
            matching = [p for p in matching if p.obj.description not in rejected]

        if not matching:
            return None
        # "Single strongest candidate": the biggest in view.
        return max(matching, key=lambda p: area(p.box))


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------

SCENARIOS = ("two_bottles", "single_bottle", "empty")


def build_scenario(name: str, world: SimWorld | None = None) -> SimWorld:
    """Populate a world. ``two_bottles``: scanning right from heading 0 meets a
    decoy bottle first (bearing ~50 deg) and the user's bottle later (~135 deg),
    standing beside a backpack."""
    world = world if world is not None else SimWorld()
    key = (name or "").strip().lower()
    if key == "two_bottles":
        world.add_at(
            "decoy_bottle", bearing=50, distance=1.5, coco_class="bottle",
            description="a green glass bottle on the floor", width=0.08, height=0.26, color=(60, 140, 60),
        )
        world.add_at(
            "target_bottle", bearing=135, distance=1.8, coco_class="bottle",
            description="a blue water bottle beside a backpack", width=0.08, height=0.25,
            is_target=True, confidence=0.9, color=(200, 90, 30),
        )
        world.add_at(
            "backpack", bearing=148, distance=1.9, coco_class="backpack",
            description="a grey backpack", width=0.32, height=0.45, matches_query=False, color=(110, 110, 110),
        )
    elif key == "single_bottle":
        world.add_at(
            "target_bottle", bearing=40, distance=1.5, coco_class="bottle",
            description="a blue water bottle", width=0.08, height=0.25, is_target=True, color=(200, 90, 30),
        )
    elif key == "empty":
        pass
    else:
        raise ValueError(f"unknown simulation scenario {name!r}; expected one of {SCENARIOS}")
    return world


# ----------------------------------------------------------------------
# ROVER_SIMULATION=true wiring (laptop demo inside the real server)
# ----------------------------------------------------------------------

_world: SimWorld | None = None
_world_lock = threading.Lock()


def ensure_simulation_allowed() -> None:
    """Simulated detections must never drive real motors: refuse to start the
    simulation unless the motor driver is the sim driver."""
    if settings.motor_driver != "sim":
        raise RuntimeError(
            f"ROVER_SIMULATION=true requires MOTOR_DRIVER=sim (got {settings.motor_driver!r}): "
            "simulated detections must never move real motors"
        )


def get_sim_world() -> SimWorld:
    """The process-wide simulated world for ``settings.rover_sim_scenario``."""
    global _world
    with _world_lock:
        if _world is None:
            _world = build_scenario(settings.rover_sim_scenario, SimWorld.for_config(RoverConfig.from_settings()))
        return _world


def reset_sim_world() -> None:
    global _world
    with _world_lock:
        _world = None


def install_simulation(camera_service=None) -> SimWorld:
    """Call from the app lifespan BEFORE ``camera_service.start()`` when
    ``settings.rover_simulation`` is on: refuses a non-sim motor driver, then
    points the shared camera service at the synthetic camera so ``/video`` and
    the controller both see the simulated world."""
    ensure_simulation_allowed()
    world = get_sim_world()
    if camera_service is None:
        from app.services.camera_service import camera_service as shared_camera_service

        camera_service = shared_camera_service
    camera_service.set_source(SimCamera(world))
    logger.warning("SIMULATION: camera frames and detections are synthetic (scenario %r)", settings.rover_sim_scenario)
    return world


def build_simulation_ports(
    drive_service: DrivePort, camera_service: CameraPort, config: RoverConfig
) -> tuple[DrivePort, CameraPort, VisionPort]:
    """Ports for ``get_rover_controller()`` in simulation mode."""
    ensure_simulation_allowed()
    world = get_sim_world()
    return TeeDrive(drive_service, SimDrive(world)), camera_service, SimVision(world)
