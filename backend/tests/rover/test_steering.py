import pytest

from app.rover.config import RoverConfig
from app.rover.proximity import ProximityProfile
from app.rover.proximity import profile_for
from app.rover.steering import ApproachSteering


@pytest.mark.parametrize('error', [-0.49, -0.35, 0.35, 0.49])
def test_off_centre_target_drives_and_steers_together(error):
    control = ApproachSteering(RoverConfig())
    wheels = control.wheels(error)
    assert not wheels.pivot
    assert wheels.left > 0 and wheels.right > 0
    assert (wheels.left > wheels.right) == (error > 0)


def test_deadzone_has_hysteresis_in_frame_width_units():
    control = ApproachSteering(RoverConfig(steer_filter_alpha=1))
    assert control.wheels(0.23).label == 'forward'  # 11.5% of frame
    assert control.wheels(0.29).label == 'forward'  # retain straight until 15%
    assert control.wheels(0.31).label == 'veer_right'
    assert control.wheels(0.26).label == 'veer_right'
    assert control.wheels(0.23).label == 'forward'


def test_pivot_only_near_edge_with_hysteresis():
    control = ApproachSteering(RoverConfig(steer_filter_alpha=1))
    assert not control.wheels(0.51).pivot
    assert control.wheels(0.56).pivot
    assert control.wheels(0.52).pivot
    assert not control.wheels(0.49).pivot


def test_all_nonzero_wheels_exceed_effective_minimum_without_exceeding_cap():
    cfg = RoverConfig()
    for i in range(-100, 101):
        for slow in [True, False]:
            wheels = ApproachSteering(cfg).wheels(i / 100, slow=slow)
            for value in [wheels.left, wheels.right]:
                assert cfg.approach_min_motor <= abs(value) <= cfg.approach_wheel_max


def test_error_filter_softens_one_frame_direction_reversal():
    control = ApproachSteering(RoverConfig())
    control.wheels(0.4)
    wheels = control.wheels(-0.4)
    assert abs(wheels.error) < 0.1


def test_arrival_stops_earlier_and_does_not_chatter():
    control = ApproachSteering(RoverConfig())
    profile = ProximityProfile(arrive_width=0.2, arrive_height=0.5)
    assert not control.arrival([400, 425, 600, 575], profile)  # width .15 < .16
    assert control.arrival([400, 415, 600, 585], profile)      # .17 enters earlier than old .20
    assert control.arrival([400, 425, 600, 575], profile)      # .15 holds above exit .14
    assert not control.arrival([400, 435, 600, 565], profile)  # .13 releases


def test_nonfinite_error_refuses_motion():
    with pytest.raises(ValueError):
        ApproachSteering(RoverConfig()).wheels(float('nan'))


def test_longer_travel_is_reserved_for_aligned_distant_target():
    cfg = RoverConfig()
    control = ApproachSteering(cfg)
    forward = control.wheels(0.0)
    assert forward.left == forward.right == pytest.approx(0.65)
    assert control.pulse_seconds(forward, slow=False) == pytest.approx(0.65)
    near = control.wheels(0.0, slow=True)
    assert near.left == near.right == pytest.approx(0.35)
    assert control.pulse_seconds(near, slow=True) == pytest.approx(0.20)
    for error in (-0.8, -0.4, 0.4, 0.8):
        control = ApproachSteering(cfg)
        turn = control.wheels(error)
        assert control.pulse_seconds(turn, slow=False) == pytest.approx(0.25)


def test_shorter_command_lease_is_respected():
    control = ApproachSteering(RoverConfig(lost_target_hold_seconds=0.1))
    for error in (0.0, 0.4, 0.8):
        assert control.pulse_seconds(control.wheels(error), slow=False) <= 0.1


def test_far_narrow_bottle_cannot_arrive_on_height_alone():
    control = ApproachSteering(RoverConfig())
    profile = profile_for('bottle')
    # Reproduces the reported 8.9% width, even with ample apparent height.
    assert not control.arrival([200, 456, 700, 545], profile)
    assert not control.arrival([300, 456, 980, 545], profile)  # bottom-edge shortcut must not override
    assert not control.arrival([200, 370, 850, 630], profile)  # still about 2.57 feet away
    assert control.arrival([100, 330, 950, 670], profile)     # about 1.96 feet away


def test_large_sideways_target_must_be_aligned_before_arrival():
    control = ApproachSteering(RoverConfig())
    profile = profile_for('bottle')
    assert not control.arrival([100, 555, 800, 895], profile)
    assert control.arrival([100, 330, 800, 670], profile)
    assert control.arrival([100, 470, 800, 810], profile)  # alignment exit hysteresis
    assert not control.arrival([100, 555, 800, 895], profile)


@pytest.mark.parametrize('distance_ft,arrived', [(7.5, False), (3, False), (2.1, False), (1.9, True)])
def test_physical_bottle_calibration_stops_at_two_feet(distance_ft, arrived):
    # Apparent width grows inversely with distance for the same bottle/camera.
    width = 0.089 * 7.5 / distance_ft
    half = round(width * 500)
    assert ApproachSteering(RoverConfig()).arrival(
        [100, 500 - half, 950, 500 + half], profile_for('bottle')) is arrived


def test_straight_driving_hysteresis_band_cannot_prevent_arrival():
    control = ApproachSteering(RoverConfig(steer_filter_alpha=1))
    assert control.wheels(0.0).label == 'forward'
    assert control.wheels(0.28).label == 'forward'
    assert control.arrival([100, 470, 800, 810], profile_for('bottle'))
