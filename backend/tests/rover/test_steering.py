import pytest

from app.rover.config import RoverConfig
from app.rover.proximity import ProximityProfile
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
