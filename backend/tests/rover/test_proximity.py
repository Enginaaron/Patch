"""Target-specific, configurable arrival evidence."""

import logging

from app.rover.proximity import (
    DEFAULT_PROFILES,
    ProximityProfile,
    arrival_evidence,
    in_slow_zone,
    load_profiles,
    profile_for,
)

PROFILE = ProximityProfile(arrive_width=0.30, arrive_height=0.50, bottom_edge=960)


def _box(width: float, height: float, *, ymax: int = 700) -> list[int]:
    """Centred horizontally, bottom edge at ``ymax``."""
    half = round(width * 500)
    return [ymax - round(height * 1000), 500 - half, ymax, 500 + half]


def test_small_box_is_not_arrival():
    assert not arrival_evidence(_box(0.10, 0.20), PROFILE)


def test_wide_or_tall_enough_is_evidence():
    assert arrival_evidence(_box(0.30, 0.20), PROFILE)
    assert arrival_evidence(_box(0.10, 0.50), PROFILE)


def test_bottom_edge_rule_needs_a_reasonably_large_box():
    # Touching the bottom of the frame and at least half the arrival height.
    assert arrival_evidence(_box(0.10, 0.25, ymax=980), PROFILE)
    # Touching the bottom but tiny: a far-away object low in the frame.
    assert not arrival_evidence(_box(0.10, 0.10, ymax=980), PROFILE)
    # Large-ish but nowhere near the bottom edge.
    assert not arrival_evidence(_box(0.10, 0.25, ymax=700), PROFILE)


def test_no_universal_size_means_arrived():
    """The same box is arrival evidence for a bottle and nothing of the sort
    for a suitcase."""
    box = _box(0.20, 0.30)
    assert arrival_evidence(box, profile_for("bottle"))
    assert not arrival_evidence(box, profile_for("suitcase"))


def test_slow_zone_starts_before_arrival():
    assert not in_slow_zone(_box(0.10, 0.20), PROFILE, 0.7)
    assert in_slow_zone(_box(0.22, 0.20), PROFILE, 0.7)     # 0.22 >= 0.7 * 0.30
    assert in_slow_zone(_box(0.10, 0.36), PROFILE, 0.7)     # 0.36 >= 0.7 * 0.50
    assert in_slow_zone(_box(0.40, 0.60), PROFILE, 0.7)     # arrival evidence implies slow zone


def test_defaults_cover_the_screened_categories():
    for category in ("bottle", "cell phone", "remote", "book", "laptop", "backpack", "handbag", "suitcase", "refrigerator"):
        assert category in DEFAULT_PROFILES
    assert profile_for(None) == DEFAULT_PROFILES["default"]
    assert profile_for("teapot") == DEFAULT_PROFILES["default"]
    assert profile_for("  Bottle ") == DEFAULT_PROFILES["bottle"]


def test_json_overrides_merge_with_defaults():
    profiles = load_profiles('{"bottle": {"arrive_width": 0.2}, "mug": {"arrive_height": 0.4, "bottom_edge": 900}}')
    assert profiles["bottle"].arrive_width == 0.2
    assert profiles["bottle"].arrive_height == DEFAULT_PROFILES["bottle"].arrive_height
    # A new category starts from the default profile.
    assert profiles["mug"] == ProximityProfile(
        arrive_width=DEFAULT_PROFILES["default"].arrive_width, arrive_height=0.4, bottom_edge=900
    )
    assert profiles["laptop"] == DEFAULT_PROFILES["laptop"]
    # The module-level defaults are never mutated.
    assert DEFAULT_PROFILES["bottle"].arrive_width != 0.2


def test_invalid_json_is_logged_and_ignored(caplog):
    with caplog.at_level(logging.WARNING, logger="app.rover.proximity"):
        assert load_profiles("{not json") == DEFAULT_PROFILES
        assert load_profiles("[1, 2]") == DEFAULT_PROFILES
    assert "invalid JSON" in caplog.text
    assert "expected a JSON object" in caplog.text


def test_invalid_entries_are_ignored_individually(caplog):
    with caplog.at_level(logging.WARNING, logger="app.rover.proximity"):
        profiles = load_profiles(
            '{"bottle": {"arrive_width": 7}, "book": {"arrive_height": "big"}, '
            '"remote": {"bottom_edge": 5000}, "laptop": 3, "handbag": {"arrive_width": 0.5}}'
        )
    for untouched in ("bottle", "book", "remote", "laptop"):
        assert profiles[untouched] == DEFAULT_PROFILES[untouched]
    assert profiles["handbag"].arrive_width == 0.5
    assert caplog.text.count("ignoring proximity profile") == 4
