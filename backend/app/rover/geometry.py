"""Pure box geometry for the rover controller.

Box convention (shared with omni_vision / yolo_detector / bbox):
``[ymin, xmin, ymax, xmax]`` integers on a 0-1000 scale, origin top-left.
Everything here is a pure function so the steering decisions can be tested
without a camera, a model or motors.

Angles: headings grow clockwise, so turning RIGHT increases the heading and
makes everything in view slide LEFT. All angle maths is approximate -- the
rover has no encoders and the camera model is a simple linear one
(``hfov`` degrees spread evenly across the frame width).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from app.rover.config import RoverConfig
from app.rover.types import Box, LocalDetection

# Two detections that both pass the association gates are "comparable" (and
# the association therefore ambiguous) when their scores differ by less than
# this. Scores are in [0, 1]; see _association_score().
AMBIGUITY_SCORE_MARGIN = 0.25


def is_valid_box(box: object) -> bool:
    """True for a well-formed ``[ymin, xmin, ymax, xmax]`` box: four ints in
    [0, 1000] with positive width and height. Anything else (wrong length,
    floats, bools, inverted or zero-area boxes) must never steer the rover."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    for value in box:
        # bool is an int subclass; a [True, False, ...] "box" is garbage.
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        if not 0 <= value <= 1000:
            return False
    ymin, xmin, ymax, xmax = box
    return ymin < ymax and xmin < xmax


def center_x(box: Sequence[int]) -> float:
    """Horizontal centre as a fraction of the frame width (0 = left edge)."""
    return (box[1] + box[3]) / 2000


def center_y(box: Sequence[int]) -> float:
    """Vertical centre as a fraction of the frame height (0 = top edge)."""
    return (box[0] + box[2]) / 2000


def width(box: Sequence[int]) -> float:
    return (box[3] - box[1]) / 1000


def height(box: Sequence[int]) -> float:
    return (box[2] - box[0]) / 1000


def area(box: Sequence[int]) -> float:
    return max(0.0, width(box)) * max(0.0, height(box))


def offset_from_center(box: Sequence[int]) -> float:
    """``center_x - 0.5``: negative when the target is LEFT of the image centre."""
    return center_x(box) - 0.5


def iou(a: Sequence[int], b: Sequence[int]) -> float:
    """Intersection over union of two boxes (0 when they do not overlap)."""
    inter_h = min(a[2], b[2]) - max(a[0], b[0])
    inter_w = min(a[3], b[3]) - max(a[1], b[1])
    if inter_h <= 0 or inter_w <= 0:
        return 0.0
    intersection = inter_h * inter_w
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def turn_command_for_offset(offset: float) -> str:
    """Drive command that turns the rover TOWARD a target at ``offset``
    (``center_x - 0.5``). Target left of centre -> ``turn_left``."""
    return "turn_left" if offset < 0 else "turn_right"


def normalize_degrees(angle: float) -> float:
    """Wrap an angle into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def angle_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two headings/bearings, in degrees."""
    return abs(normalize_degrees(a - b))


def predict_box_after_turn(box: Sequence[int], heading_delta_deg: float, hfov_deg: float) -> Box:
    """Where ``box`` should appear after the rover rotates in place.

    ``heading_delta_deg`` > 0 means the rover turned RIGHT, which moves every
    object LEFT by ``delta / hfov`` of the frame width. The result is a
    *prediction* used only for association: it is deliberately not clamped to
    the frame (so the width stays comparable) and may therefore fail
    ``is_valid_box``.
    """
    if hfov_deg <= 0:
        raise ValueError("hfov_deg must be positive")
    shift = round(-heading_delta_deg / hfov_deg * 1000)
    return [int(box[0]), int(box[1]) + shift, int(box[2]), int(box[3]) + shift]


def bearing_of(box: Sequence[int], heading_deg: float, hfov_deg: float) -> float:
    """Approximate world bearing of a box: the rover's (estimated) heading plus
    the box's angular offset from the image centre."""
    return heading_deg + offset_from_center(box) * hfov_deg


@dataclass(frozen=True)
class AssociationResult:
    """Outcome of matching this frame's detections to the tracked target.

    ``match`` is the best detection that passed the gates (or None).
    ``ambiguous`` means a second detection passed too with a comparable score:
    a per-frame detector cannot say which of two similar objects is the one
    the user accepted, so the caller must revalidate identity (OMNI) instead
    of trusting ``match``.
    """

    match: LocalDetection | None
    ambiguous: bool
    passed: int = 0


def _size_ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return math.inf
    return max(a / b, b / a)


def _association_score(shift: float, ratio: float, max_shift: float, max_ratio: float) -> float:
    """1.0 = exactly where and as big as predicted; 0.0 = right on both gates."""
    shift_term = shift / max_shift if max_shift > 0 else 0.0
    ratio_term = math.log(ratio) / math.log(max_ratio) if max_ratio > 1 else 0.0
    return 1.0 - 0.5 * min(1.0, shift_term) - 0.5 * min(1.0, ratio_term)


def associate(
    detections: Sequence[LocalDetection],
    predicted_box: Sequence[int],
    cfg: RoverConfig,
    *,
    ambiguity_margin: float = AMBIGUITY_SCORE_MARGIN,
) -> AssociationResult:
    """Pick the detection that continues the tracked target.

    Gates: the box centre may move at most ``cfg.association_max_center_shift``
    (fraction of the frame) from the prediction, and neither its width nor its
    height may change by more than ``cfg.association_max_size_ratio``.
    Invalid boxes are ignored.
    """
    max_shift = cfg.association_max_center_shift
    max_ratio = cfg.association_max_size_ratio
    pred_cx, pred_cy = center_x(predicted_box), center_y(predicted_box)
    pred_w, pred_h = width(predicted_box), height(predicted_box)

    scored: list[tuple[float, LocalDetection]] = []
    for det in detections:
        if not is_valid_box(det.box):
            continue
        shift = math.hypot(center_x(det.box) - pred_cx, center_y(det.box) - pred_cy)
        if shift > max_shift:
            continue
        ratio = max(_size_ratio(width(det.box), pred_w), _size_ratio(height(det.box), pred_h))
        if ratio > max_ratio:
            continue
        scored.append((_association_score(shift, ratio, max_shift, max_ratio), det))

    if not scored:
        return AssociationResult(match=None, ambiguous=False, passed=0)

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    ambiguous = len(scored) > 1 and (best_score - scored[1][0]) < ambiguity_margin
    return AssociationResult(match=best, ambiguous=ambiguous, passed=len(scored))
