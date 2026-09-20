"""The LIVE recognition path, across every module it crosses, with only the
network call faked.

The other mission tests run on ``SimVision``; ``test_live_vision.py`` checks
``LiveVision`` against injected fakes. Neither runs the path the real rover
uses as one piece:

    controller -> LiveVision.identify -> search_worker._prepare_frame
               -> omni_vision.detect_* -> build_content (rejected crops read back
                  from disk, the user-confirmed crop) -> SearchDetection
               -> candidate images saved -> approach

Here that whole chain is real. The one thing replaced is ``omni_vision._call``
-- the single function that talks to the gateway -- by a stand-in that answers
from the simulated world. Motion, frames and the "model" are SIMULATED; this
says nothing about what the real OMNI model does with these prompts.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from types import SimpleNamespace

from PIL import Image

from app.config import settings
from app.models import CandidateDecision
from app.rover.geometry import area
from app.rover.simulation import SimWorld, build_scenario
from app.rover.vision import LiveVision
from app.services import omni_vision
from app.services.omni_vision import CONFIRMED_LABEL, CURRENT_FRAME_LABEL, DetectionResult, SearchDetection
from tests.rover.helpers import candidates_of, fast_config, finds_of, make_world, search_status

TARGET_TEXT = "my water bottle beside my backpack"
DECOY = "a green glass bottle on the floor"
TARGET = "a blue water bottle beside a backpack"


class _NoLocalDetector:
    """YOLO is optional (torch is not installed here, and must not be needed):
    without it the controller asks OMNI on every stop."""

    @staticmethod
    def is_available() -> bool:
        return False


class _FakeGateway:
    """Stands in for ``omni_vision._call``. Like the real model it sees only
    the prompt it is sent: it honours the REJECTED descriptions listed there,
    and once a USER-CONFIRMED crop is attached it sticks to the object it
    reported when the user said yes."""

    def __init__(self, world: SimWorld) -> None:
        self.world = world
        self.lock = threading.Lock()
        self.contents: list[list[dict]] = []
        self.moving_during_call: list[bool] = []
        self._last_reported: str | None = None
        self._confirmed: str | None = None

    def __call__(self, content: list[dict], *, target_text: str) -> DetectionResult:
        assert target_text == TARGET_TEXT
        with self.lock:
            self.moving_during_call.append(self.world.is_moving())
            self.contents.append(content)
            prompt = content[0]["text"]
            matching = [p for p in self.world.visible(0.6) if p.obj.matches_query]
            if CONFIRMED_LABEL in _labels(content):
                self._confirmed = self._confirmed or self._last_reported
                matching = [p for p in matching if p.obj.id == self._confirmed]
            else:
                matching = [p for p in matching if f'"{p.obj.description}"' not in prompt]
            if not matching:
                detection = SearchDetection(found=False, likelihood="low", box_2d=None, description="nothing matching")
            else:
                best = max(matching, key=lambda p: area(p.box))
                self._last_reported = best.obj.id
                detection = SearchDetection(
                    found=True, likelihood="high", box_2d=list(best.box), description=best.obj.description
                )
            return DetectionResult(detection=detection, latency_seconds=0.0)


def _labels(content: list[dict]) -> list[str]:
    return [item["text"] for item in content[1:] if item["type"] == "text"]


def _image_after(content: list[dict], label: str) -> Image.Image:
    """The picture attached right after ``label``, decoded."""
    index = next(i for i, item in enumerate(content) if item.get("text") == label)
    url = content[index + 1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    image.load()
    return image


def test_live_path_rejection_memory_confirmed_crop_and_arrival(harness, monkeypatch, media_root):
    monkeypatch.setattr(settings, "omni_api_key", "test-key")   # never used: the gateway call is the fake below
    cfg = fast_config()
    world = build_scenario("two_bottles", make_world(cfg))
    gateway = _FakeGateway(world)
    monkeypatch.setattr(omni_vision, "_call", gateway)
    h = harness(cfg, world, vision=LiveVision(cfg, detector=_NoLocalDetector()))
    sid, log = h.search(TARGET_TEXT)

    h.controller.start_search(sid)

    # 1) The decoy comes first. Nothing has been rejected yet: plain Mode A prompt.
    decoy = log.wait_type("candidate_found")["payload"]
    log.wait_phase("waiting_for_confirmation")
    assert decoy["description"] == DECOY
    asked_with = len(gateway.contents)
    assert all("REJECTED" not in content[0]["text"] and _labels(content) == [] for content in gateway.contents)
    h.controller.reject_candidate(sid, decoy["candidate_id"])

    # 2) The next question is about the OTHER bottle, and every prompt since the
    #    "no" carried the rejected description plus its crop, read back from disk.
    target = log.wait_type("candidate_found")["payload"]
    log.wait_phase("waiting_for_confirmation")
    assert target["description"] == TARGET and target["candidate_id"] != decoy["candidate_id"]
    since_rejection = gateway.contents[asked_with:]
    assert since_rejection
    for content in since_rejection:
        assert f'REJECTED OBJECT 1: "{DECOY}" (image below)' in content[0]["text"]
        assert _labels(content) == ["REJECTED OBJECT 1:", CURRENT_FRAME_LABEL]
        crop, frame = _image_after(content, "REJECTED OBJECT 1:"), _image_after(content, CURRENT_FRAME_LABEL)
        assert crop.width < frame.width and crop.height < frame.height   # the crop, not a whole frame
    accepted_with = len(gateway.contents)
    h.controller.accept_candidate(sid, target["candidate_id"])

    # 3) Approach: every look now carries the crop the user confirmed.
    final = log.wait_rest(timeout=30)["payload"]
    assert (final["phase"], final["reason"]) == ("arrived", "proximity_confirmed")
    assert final["vision_mode"] == "live" and final["driver"] == "sim"
    approach = gateway.contents[accepted_with:]
    assert approach
    for content in approach:
        assert CONFIRMED_LABEL in _labels(content) and _labels(content)[-1] == CURRENT_FRAME_LABEL
        assert "SAME physical object" in content[0]["text"]
        confirmed, frame = _image_after(content, CONFIRMED_LABEL), _image_after(content, CURRENT_FRAME_LABEL)
        assert confirmed.width < frame.width and confirmed.height < frame.height

    # OMNI always got the user's full wording, and only ever with the wheels stopped.
    assert all(TARGET_TEXT in content[0]["text"] for content in gateway.contents)
    assert gateway.moving_during_call and not any(gateway.moving_during_call)

    # Recognition lifecycle and the files behind the candidate URLs.
    assert search_status(sid).value == "FOUND" and len(finds_of(sid)) == 1
    rows = candidates_of(sid)
    assert [row.decision for row in rows] == [CandidateDecision.REJECTED, CandidateDecision.ACCEPTED]
    for row in rows:
        assert (media_root / row.crop_path).is_file() and (media_root / row.full_image_path).is_file()

    # Bounded pulses, every one with the watchdog armed; stopped next to the right bottle.
    moves = h.drive.moves
    assert any(move.command == "forward" for move in moves)
    assert all(move.ttl is not None and 0 < move.ttl < 2.0 for move in moves)
    assert h.drive.records[-1].command == "stop" and not world.is_moving() and world.watchdog_trips == 0
    assert world.distance_to("target_bottle") < 0.8 < world.distance_to("decoy_bottle")


def test_wrong_localization_never_creates_candidate_or_moves_rover(harness, monkeypatch):
    monkeypatch.setattr(settings, "omni_api_key", "test-key")
    cfg = fast_config()
    cfg.max_vision_errors = 2
    world = build_scenario("two_bottles", make_world(cfg))

    def create(**kwargs):
        if kwargs["max_tokens"] == 80:
            reply = {"matches": False}
        else:
            reply = {"found": True, "likelihood": "high", "bbox_2d": [48, 53, 96, 107],
                     "description": "a blue water bottle beside a backpack"}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))])

    monkeypatch.setattr(omni_vision, "_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    h = harness(cfg, world, vision=LiveVision(cfg, detector=_NoLocalDetector()))
    sid, log = h.search(TARGET_TEXT)
    h.controller.start_search(sid)
    final = log.wait_rest()["payload"]
    assert (final["phase"], final["reason"]) == ("error", "inference_failed")
    assert candidates_of(sid) == []
    assert h.drive.moves == []
    assert not world.is_moving()
