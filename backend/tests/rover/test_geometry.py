"""Box interpretation, turn direction, prediction and association."""

import math

import pytest

from app.rover import geometry
from app.rover.types import LocalDetection
from tests.rover.helpers import fast_config

# [ymin, xmin, ymax, xmax] on the 0-1000 scale
LEFT_BOX = [400, 100, 600, 200]
RIGHT_BOX = [400, 800, 600, 900]
CENTRE_BOX = [300, 450, 700, 550]


@pytest.mark.parametrize(
    "box",
    [
        [0, 0, 1000, 1000],
        [400, 100, 600, 200],
        (10, 20, 30, 40),
    ],
)
def test_valid_boxes(box):
    assert geometry.is_valid_box(box)


@pytest.mark.parametrize(
    "box",
    [
        None,
        "0,0,10,10",
        [1, 2, 3],
        [1, 2, 3, 4, 5],
        [400, 200, 600, 200],      # zero width
        [600, 100, 400, 200],      # ymin > ymax
        [400, 300, 600, 200],      # xmin > xmax
        [-1, 0, 10, 10],
        [0, 0, 10, 1001],
        [0.0, 0.0, 10.0, 10.0],    # floats are not the wire format
        [True, False, True, True],
    ],
)
def test_invalid_boxes_are_rejected(box):
    assert not geometry.is_valid_box(box)


def test_box_is_ymin_xmin_ymax_xmax():
    box = [100, 200, 500, 800]
    assert geometry.center_x(box) == pytest.approx(0.5)
    assert geometry.center_y(box) == pytest.approx(0.3)
    assert geometry.width(box) == pytest.approx(0.6)
    assert geometry.height(box) == pytest.approx(0.4)
    assert geometry.area(box) == pytest.approx(0.24)


def test_offset_sign_and_turn_direction():
    # Target left of the image centre -> negative offset -> turn LEFT toward it.
    assert geometry.offset_from_center(LEFT_BOX) == pytest.approx(-0.35)
    assert geometry.turn_command_for_offset(geometry.offset_from_center(LEFT_BOX)) == "turn_left"
    assert geometry.offset_from_center(RIGHT_BOX) == pytest.approx(0.35)
    assert geometry.turn_command_for_offset(geometry.offset_from_center(RIGHT_BOX)) == "turn_right"
    assert geometry.offset_from_center(CENTRE_BOX) == pytest.approx(0.0)


def test_iou():
    assert geometry.iou(CENTRE_BOX, CENTRE_BOX) == pytest.approx(1.0)
    assert geometry.iou(LEFT_BOX, RIGHT_BOX) == 0.0
    half_overlap = geometry.iou([0, 0, 100, 100], [0, 50, 100, 150])
    assert half_overlap == pytest.approx(1 / 3)


def test_turning_right_moves_objects_left():
    # 31 degrees of a 62 degree field of view is half the frame.
    predicted = geometry.predict_box_after_turn(RIGHT_BOX, 31.0, 62.0)
    assert predicted == [400, 300, 600, 400]
    assert geometry.width(predicted) == pytest.approx(geometry.width(RIGHT_BOX))


def test_turning_left_moves_objects_right_and_may_leave_the_frame():
    predicted = geometry.predict_box_after_turn(RIGHT_BOX, -15.5, 62.0)
    assert predicted == [400, 1050, 600, 1150]
    # A prediction is not clamped (its size must stay comparable) ...
    assert not geometry.is_valid_box(predicted)
    # ... and an exact turn toward a target centres it.
    offset = geometry.offset_from_center(RIGHT_BOX)
    centred = geometry.predict_box_after_turn(RIGHT_BOX, offset * 62.0, 62.0)
    assert geometry.offset_from_center(centred) == pytest.approx(0.0, abs=1e-3)


def test_predict_rejects_nonsense_fov():
    with pytest.raises(ValueError):
        geometry.predict_box_after_turn(CENTRE_BOX, 10.0, 0.0)


def test_bearing_combines_heading_and_offset():
    assert geometry.bearing_of(CENTRE_BOX, 90.0, 62.0) == pytest.approx(90.0)
    assert geometry.bearing_of(RIGHT_BOX, 20.0, 62.0) == pytest.approx(20.0 + 0.35 * 62.0)
    assert geometry.bearing_of(LEFT_BOX, 20.0, 62.0) == pytest.approx(20.0 - 0.35 * 62.0)


def test_angle_difference_wraps():
    assert geometry.angle_difference(350.0, 10.0) == pytest.approx(20.0)
    assert geometry.angle_difference(10.0, 370.0) == pytest.approx(0.0)
    assert geometry.angle_difference(0.0, 180.0) == pytest.approx(180.0)
    assert geometry.normalize_degrees(190.0) == pytest.approx(-170.0)


def _det(box, confidence=0.9):
    return LocalDetection(box=box, confidence=confidence)


def test_associate_picks_the_detection_that_continues_the_track():
    cfg = fast_config()
    near = _det([310, 460, 710, 560])
    far = _det([300, 900, 700, 1000])          # centre shift 0.45 > 0.30 gate
    result = geometry.associate([far, near], CENTRE_BOX, cfg)
    assert result.match is near
    assert not result.ambiguous
    assert result.passed == 1


def test_associate_gates_on_size_change():
    cfg = fast_config()
    tiny = _det([480, 490, 520, 510])          # 5x narrower, 10x shorter
    huge = _det([0, 250, 1000, 750])           # 5x wider
    assert geometry.associate([tiny, huge], CENTRE_BOX, cfg).match is None
    grown = _det([250, 430, 750, 570])         # 1.4x wider, 1.25x taller: a forward pulse
    assert geometry.associate([grown], CENTRE_BOX, cfg).match is grown


def test_associate_flags_two_similar_objects_as_ambiguous():
    cfg = fast_config()
    a = _det([300, 450, 700, 550])
    b = _det([300, 540, 700, 640])             # same size, 0.09 of the frame to the right
    result = geometry.associate([a, b], CENTRE_BOX, cfg)
    assert result.match is a                   # still reports the best ...
    assert result.ambiguous                    # ... but it must not be trusted without OMNI
    assert result.passed == 2


def test_associate_is_not_ambiguous_when_the_runner_up_is_clearly_worse():
    cfg = fast_config()
    a = _det([300, 450, 700, 550])
    b = _det([300, 700, 700, 800])             # passes the 0.30 gate but only just
    result = geometry.associate([a, b], CENTRE_BOX, cfg)
    assert result.match is a
    assert not result.ambiguous


def test_associate_ignores_invalid_boxes_and_empty_input():
    cfg = fast_config()
    garbage = _det([700, 550, 300, 450])
    assert geometry.associate([garbage], CENTRE_BOX, cfg).match is None
    assert geometry.associate([], CENTRE_BOX, cfg) == geometry.AssociationResult(None, False, 0)


def test_associate_accepts_an_unclamped_prediction():
    cfg = fast_config()
    predicted = geometry.predict_box_after_turn([400, 850, 600, 950], -6.2, 62.0)  # slides to 950..1050
    clipped = _det([400, 940, 600, 1000])
    assert geometry.associate([clipped], predicted, cfg).match is clipped


def test_association_score_is_monotonic():
    best = geometry._association_score(0.0, 1.0, 0.3, 2.0)
    worse_shift = geometry._association_score(0.15, 1.0, 0.3, 2.0)
    worse_size = geometry._association_score(0.0, math.sqrt(2.0), 0.3, 2.0)
    assert best == pytest.approx(1.0)
    assert worse_shift == pytest.approx(0.75)
    assert worse_size == pytest.approx(0.75)
