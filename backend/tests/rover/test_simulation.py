"""The simulated world itself: its camera model has to agree with the
controller's geometry (turn right -> things slide left, closer -> bigger), its
drive has to honour ttl, and its wiring must refuse real motors."""

import threading
import time

import numpy as np
import pytest

from app.config import settings
from app.rover import controller as controller_module
from app.rover import geometry, simulation
from app.rover.controller import get_rover_controller, set_rover_controller
from app.rover.simulation import (
    RecordingDrive,
    SimCamera,
    SimDrive,
    SimVision,
    TeeDrive,
    build_scenario,
    build_simulation_ports,
    ensure_simulation_allowed,
)
from app.rover.types import CameraPort, DrivePort, IdentifyRequest, RejectedHint, VisionPort
from tests.rover.helpers import add_bottle, fast_config, make_world


def _world_with_bottle(bearing=0.0, distance=1.5):
    cfg = fast_config()
    world = make_world(cfg)
    add_bottle(world, bearing=bearing, distance=distance, is_target=True)
    return cfg, world


def test_ports_satisfy_the_protocols():
    _, world = _world_with_bottle()
    assert isinstance(SimDrive(world), DrivePort) and isinstance(RecordingDrive(SimDrive(world)), DrivePort)
    assert isinstance(TeeDrive(SimDrive(world), SimDrive(world)), DrivePort)
    assert isinstance(SimCamera(world), CameraPort)
    assert isinstance(SimVision(world), VisionPort) and SimVision(world).is_simulated is True
    assert SimDrive(world).driver_name == "sim"


def test_object_ahead_is_centred_and_behind_is_invisible():
    _, world = _world_with_bottle(bearing=0.0)
    [projection] = world.visible()
    assert geometry.offset_from_center(projection.box) == pytest.approx(0.0, abs=0.002)
    world.set_pose(0.0, 0.0, 180.0)
    assert world.visible() == []


def test_turning_right_slides_objects_left_as_geometry_predicts():
    cfg, world = _world_with_bottle(bearing=10.0)
    before = world.visible()[0].box
    world.set_pose(0.0, 0.0, 15.0)                       # rover turned right by 15 degrees
    after = world.visible()[0].box
    assert geometry.center_x(after) < geometry.center_x(before)
    predicted = geometry.predict_box_after_turn(before, 15.0, cfg.camera_hfov_degrees)
    # Linear prediction vs pinhole "reality": close enough to associate.
    assert abs(geometry.center_x(predicted) - geometry.center_x(after)) < 0.05
    assert geometry.turn_command_for_offset(geometry.offset_from_center(before)) == "turn_right"


def test_getting_closer_makes_the_box_bigger():
    _, world = _world_with_bottle(bearing=0.0, distance=2.0)
    far = world.visible()[0].box
    world.set_pose(0.0, 1.0, 0.0)
    near = world.visible()[0].box
    assert geometry.width(near) == pytest.approx(2 * geometry.width(far), rel=0.1)
    assert geometry.height(near) == pytest.approx(2 * geometry.height(far), rel=0.05)


def test_mostly_clipped_objects_are_not_reported():
    _, world = _world_with_bottle(bearing=31.5)          # straddles the right edge of a 62 degree view
    projection = world.project(world.get("target_bottle"))
    assert projection is not None and projection.visible_fraction < 0.6
    assert world.visible() == []


def test_drive_integrates_the_nominal_pulse_not_timer_jitter():
    cfg, world = _world_with_bottle()
    drive = SimDrive(world)
    ttl = 0.02 + cfg.pulse_watchdog_margin_seconds
    drive.drive("turn_right", cfg.turn_speed, ttl=ttl)
    threading.Event().wait(0.03)                         # "0.02 s" as an OS timer delivers it
    drive.stop()
    assert world.pose()[2] == pytest.approx(20.0)        # exactly 1000 deg/s x 0.02 s

    drive.drive("forward", cfg.forward_speed, ttl=ttl)
    threading.Event().wait(0.03)
    drive.stop()
    x, y, _ = world.pose()
    assert (x, y) == pytest.approx((0.1 * np.sin(np.radians(20.0)), 0.1 * np.cos(np.radians(20.0))))
    # Speed scales the (made-up) rates linearly.
    drive.drive("turn_left", cfg.turn_speed / 2, ttl=ttl)
    threading.Event().wait(0.03)
    drive.stop()
    assert world.pose()[2] == pytest.approx(10.0)


def test_drive_stopped_early_only_moves_for_the_time_it_ran():
    cfg, world = _world_with_bottle()
    drive = SimDrive(world)
    drive.drive("turn_right", cfg.turn_speed, ttl=1.0 + cfg.pulse_watchdog_margin_seconds)
    threading.Event().wait(0.02)
    assert world.is_moving()
    drive.stop()
    assert 0.0 < world.pose()[2] < 500.0                 # nowhere near the nominal 1000 degrees
    assert not world.is_moving()


def test_drive_honours_ttl_when_nobody_stops_it():
    world = make_world(fast_config(), watchdog_margin_seconds=0.0)
    drive = SimDrive(world)
    drive.drive("turn_right", 0.4, ttl=0.03)
    threading.Event().wait(0.08)
    assert not world.is_moving()                          # the deadline ended the motion by itself
    assert world.pose()[2] == pytest.approx(30.0)         # ... at exactly ttl
    drive.stop()
    assert world.pose()[2] == pytest.approx(30.0) and world.watchdog_trips == 1


def test_stuck_world_turns_but_does_not_advance():
    cfg, world = _world_with_bottle()
    world.stuck = True
    drive = SimDrive(world)
    drive.drive("forward", cfg.forward_speed, ttl=0.02 + cfg.pulse_watchdog_margin_seconds)
    threading.Event().wait(0.03)
    drive.stop()
    assert world.pose()[:2] == (0.0, 0.0)


def test_camera_frames_are_fresh_sequenced_and_cancellable():
    _, world = _world_with_bottle()
    camera = SimCamera(world)
    first = camera.wait_for_frame(after_seq=-1, not_before=time.monotonic(), timeout=1.0)
    not_before = time.monotonic() + 0.03
    second = camera.wait_for_frame(after_seq=first.seq, not_before=not_before, timeout=1.0)
    assert second.seq > first.seq and second.captured_at >= not_before
    assert first.frame.shape == (480, 640, 3) and first.frame.dtype == np.uint8
    assert camera.latest_packet().seq > second.seq

    # A frame that cannot exist before the timeout: None, after the timeout.
    started = time.monotonic()
    assert camera.wait_for_frame(after_seq=0, not_before=time.monotonic() + 5.0, timeout=0.05) is None
    assert 0.04 <= time.monotonic() - started < 1.0

    cancel = threading.Event()
    cancel.set()
    started = time.monotonic()
    assert camera.wait_for_frame(after_seq=0, not_before=time.monotonic() + 5.0, timeout=5.0, cancel=cancel) is None
    assert time.monotonic() - started < 0.5

    camera.fail = True
    assert camera.wait_for_frame(after_seq=0, not_before=time.monotonic(), timeout=0.02) is None
    assert camera.latest_packet() is None and camera.get_frame() is None


def test_camera_doubles_as_a_camera_source_showing_the_world():
    from app.camera.base import CameraSource

    _, world = _world_with_bottle()
    camera = SimCamera(world, fps=200)
    assert isinstance(camera, CameraSource)
    camera.start()
    with_bottle = camera.get_frame()
    world.remove_object("target_bottle")
    without = camera.get_frame()
    camera.stop()
    assert with_bottle.shape == (480, 640, 3)
    assert np.any(with_bottle != without)                 # the bottle was really drawn


def test_vision_reports_one_strongest_match_and_honours_rejections():
    cfg = fast_config()
    world = make_world(cfg)
    add_bottle(world, "near", bearing=-10, distance=1.0, description="near bottle")
    add_bottle(world, "far", bearing=10, distance=2.0, description="far bottle", is_target=True)
    world.add_at("bag", bearing=0, distance=1.5, coco_class="backpack", description="a bag",
                 width=0.3, height=0.4, matches_query=False)
    vision = SimVision(world)
    packet = SimCamera(world).latest_packet()

    assert len(vision.screen(packet, "bottle")) == 2 and len(vision.screen(packet, "backpack")) == 1
    first = vision.identify(packet, IdentifyRequest(target_text="my bottle"))
    assert (first.found, first.description, first.frame_seq) == (True, "near bottle", packet.seq)
    assert geometry.is_valid_box(first.box)

    hinted = IdentifyRequest(target_text="my bottle", rejected=[RejectedHint("near bottle")])
    assert vision.identify(packet, hinted).description == "far bottle"
    both = IdentifyRequest(target_text="my bottle", rejected=[RejectedHint("near bottle"), RejectedHint("far bottle")])
    nothing = vision.identify(packet, both)
    assert (nothing.found, nothing.box) == (False, None)

    vision.ignore_rejections = True
    assert vision.identify(packet, both).description == "near bottle"
    vision.screening_available = False
    assert vision.screen(packet, "bottle") is None


def test_two_bottles_scenario_meets_the_decoy_first_when_scanning_right():
    world = build_scenario("two_bottles", make_world(fast_config()))
    first_seen = {}
    for heading in range(0, 360, 5):
        world.set_pose(0.0, 0.0, float(heading))
        for projection in world.visible():
            first_seen.setdefault(projection.obj.id, heading)
    assert first_seen["decoy_bottle"] < first_seen["target_bottle"]
    assert world.target().id == "target_bottle"
    # The two are never in view together, so "which bottle?" has one answer per view.
    for heading in range(0, 360, 5):
        world.set_pose(0.0, 0.0, float(heading))
        bottles = [p.obj.id for p in world.visible() if p.obj.coco_class == "bottle"]
        assert len(bottles) <= 1
    with pytest.raises(ValueError):
        build_scenario("no-such-scenario")


# --- ROVER_SIMULATION wiring ----------------------------------------------------------


class _FakeDriveService:
    driver_name = "sim"

    def __init__(self):
        self.calls = []

    def drive(self, command, speed=None, ttl=None):
        self.calls.append((command, speed, ttl))

    def stop(self):
        self.calls.append(("stop", None, None))


def test_simulation_is_refused_unless_the_motor_driver_is_sim(monkeypatch):
    assert settings.motor_driver == "sim"
    ensure_simulation_allowed()                            # fine with the sim driver

    monkeypatch.setattr(settings, "motor_driver", "gpio")
    with pytest.raises(RuntimeError, match="MOTOR_DRIVER=sim"):
        ensure_simulation_allowed()
    with pytest.raises(RuntimeError):
        build_simulation_ports(_FakeDriveService(), object(), fast_config())
    with pytest.raises(RuntimeError):
        simulation.install_simulation(camera_service=object())

    # The lazy singleton refuses too, before touching any real service.
    monkeypatch.setattr(settings, "rover_simulation", True)
    set_rover_controller(None)
    with pytest.raises(RuntimeError, match="MOTOR_DRIVER=sim"):
        get_rover_controller()
    assert controller_module._controller is None


def test_simulation_ports_mirror_commands_into_the_world(monkeypatch):
    monkeypatch.setattr(settings, "rover_sim_scenario", "two_bottles")
    simulation.reset_sim_world()
    try:
        service, camera = _FakeDriveService(), object()
        drive, camera_port, vision = build_simulation_ports(service, camera, fast_config())
        assert camera_port is camera and vision.is_simulated and drive.driver_name == "sim"
        world = simulation.get_sim_world()
        assert vision.world is world and world.target().id == "target_bottle"

        drive.drive("turn_right", 0.4, ttl=1.0)
        assert world.is_moving()
        drive.stop()
        assert not world.is_moving() and world.pose()[2] > 0
        assert service.calls == [("turn_right", 0.4, 1.0), ("stop", None, None)]
    finally:
        simulation.reset_sim_world()


def test_install_simulation_points_the_camera_service_at_the_sim_camera():
    class _CameraService:
        source = None

        def set_source(self, source):
            self.source = source

    simulation.reset_sim_world()
    try:
        service = _CameraService()
        world = simulation.install_simulation(camera_service=service)
        assert isinstance(service.source, SimCamera) and world is simulation.get_sim_world()
    finally:
        simulation.reset_sim_world()


def test_set_rover_controller_installs_the_singleton(harness):
    h = harness(fast_config())
    set_rover_controller(h.controller)
    try:
        assert get_rover_controller() is h.controller
    finally:
        set_rover_controller(None)
