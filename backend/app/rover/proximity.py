"""Conservative "am I next to it?" heuristic.

A single forward RGB camera gives no reliable distance. In particular:

* **No universal box size means "arrived".** A suitcase fills the frame from
  two metres away; a phone never does. Thresholds are therefore per target
  category, configurable, and only ever *evidence* -- the controller requires
  several consecutive stationary observations (the last one validated by OMNI)
  before it reports ``arrived``.
* **Timed movement does not measure distance.** A 0.4 s forward pulse covers a
  different distance on carpet, on tiles, and on a low battery. The controller
  never counts pulses to decide it has arrived; it only looks at what the
  camera shows after each pulse.

The default numbers below are starting points for a controlled-area demo and
have NOT been calibrated on the physical rover. Override them with
``ROVER_PROXIMITY_PROFILES_JSON``, e.g. ``{"bottle": {"arrive_width": 0.2}}``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import Sequence

from app.rover import geometry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProximityProfile:
    """Apparent-size thresholds (fractions of the frame) for one category."""

    arrive_width: float
    arrive_height: float
    # Box bottom this close to the bottom of the frame (0-1000 scale) suggests
    # the object is about to slide under the camera's view.
    bottom_edge: int = 960


# Keyed by COCO class (what comprehension.resolve_target returns as category).
DEFAULT_PROFILES: dict[str, ProximityProfile] = {
    "bottle": ProximityProfile(arrive_width=0.16, arrive_height=0.50),
    "cell phone": ProximityProfile(arrive_width=0.22, arrive_height=0.30),
    "remote": ProximityProfile(arrive_width=0.25, arrive_height=0.25),
    "book": ProximityProfile(arrive_width=0.35, arrive_height=0.35),
    "laptop": ProximityProfile(arrive_width=0.50, arrive_height=0.45),
    "backpack": ProximityProfile(arrive_width=0.45, arrive_height=0.60),
    "handbag": ProximityProfile(arrive_width=0.40, arrive_height=0.50),
    "suitcase": ProximityProfile(arrive_width=0.45, arrive_height=0.70),
    "refrigerator": ProximityProfile(arrive_width=0.60, arrive_height=0.90),
    "default": ProximityProfile(arrive_width=0.35, arrive_height=0.50),
}


def _fraction(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0.0 < float(value) <= 1.0 else None


def load_profiles(profiles_json: str = "") -> dict[str, ProximityProfile]:
    """Defaults merged with a JSON override. Invalid JSON, or an invalid entry,
    is logged and ignored -- a typo in .env must not change how close the rover
    drives to things in some surprising way."""
    profiles = dict(DEFAULT_PROFILES)
    text = (profiles_json or "").strip()
    if not text:
        return profiles

    try:
        overrides = json.loads(text)
    except ValueError as exc:
        logger.warning("ignoring ROVER_PROXIMITY_PROFILES_JSON (invalid JSON): %s", exc)
        return profiles
    if not isinstance(overrides, dict):
        logger.warning("ignoring ROVER_PROXIMITY_PROFILES_JSON: expected a JSON object")
        return profiles

    for name, fields in overrides.items():
        key = str(name).strip().lower()
        if not key or not isinstance(fields, dict):
            logger.warning("ignoring proximity profile %r: expected an object", name)
            continue
        profile = profiles.get(key, profiles["default"])
        valid = True
        for field_name in ("arrive_width", "arrive_height"):
            if field_name in fields:
                value = _fraction(fields[field_name])
                if value is None:
                    valid = False
                    break
                profile = replace(profile, **{field_name: value})
        if valid and "bottom_edge" in fields:
            edge = fields["bottom_edge"]
            if isinstance(edge, bool) or not isinstance(edge, int) or not 0 <= edge <= 1000:
                valid = False
            else:
                profile = replace(profile, bottom_edge=edge)
        if not valid:
            logger.warning("ignoring proximity profile %r: values out of range", name)
            continue
        profiles[key] = profile
    return profiles


def profile_for(category: str | None, profiles: dict[str, ProximityProfile] | None = None) -> ProximityProfile:
    """Profile for a resolved category; unknown / None -> the ``default`` entry."""
    table = profiles if profiles is not None else DEFAULT_PROFILES
    if category:
        found = table.get(category.strip().lower())
        if found is not None:
            return found
    return table.get("default", DEFAULT_PROFILES["default"])


def arrival_evidence(box: Sequence[int], profile: ProximityProfile) -> bool:
    """True when ONE observation looks close enough for this category.

    This is evidence, not arrival: no universal box size means "arrived", and
    the caller must see it repeatedly from a standstill. It is never derived
    from how long the wheels ran -- timed movement does not measure distance.
    """
    box_width = geometry.width(box)
    box_height = geometry.height(box)
    if box_width >= profile.arrive_width or box_height >= profile.arrive_height:
        return True
    # Touching the bottom of the frame while already fairly large: the object
    # is about to leave the camera's view, so going further would lose it.
    return box[2] >= profile.bottom_edge and box_height >= 0.5 * profile.arrive_height


def in_slow_zone(box: Sequence[int], profile: ProximityProfile, fraction: float) -> bool:
    """True once the apparent size reaches ``fraction`` of the arrival
    threshold: the controller then uses shorter, slower forward pulses."""
    if arrival_evidence(box, profile):
        return True
    return (
        geometry.width(box) >= fraction * profile.arrive_width
        or geometry.height(box) >= fraction * profile.arrive_height
    )
