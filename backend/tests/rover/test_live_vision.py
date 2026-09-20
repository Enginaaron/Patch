"""LiveVision: the adapter between the controller and omni_vision / yolo_detector.

The OMNI functions are injected fakes with the exact keyword-only signature
the spec gives ``detect_text_only`` / ``detect_personalized``, so a call with
the wrong shape fails here with a TypeError. Nothing touches the network or a
model."""

import time
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from app.config import settings
from app.rover.types import FramePacket, IdentifyRequest, RejectedHint, VisionPort
from app.rover.vision import LiveVision
from tests.rover.helpers import fast_config


@dataclass
class _Detection:
    found: bool = True
    likelihood: str = "high"
    box_2d: list | None = None
    description: str = "a blue bottle"


@dataclass
class _Result:
    detection: _Detection
    latency_seconds: float = 1.25


class _Omni:
    def __init__(self, detection: _Detection | None = None, error: Exception | None = None) -> None:
        self.detection = detection or _Detection(box_2d=[100, 200, 300, 400])
        self.error = error
        self.calls: list[dict] = []

    def detect_text_only(self, target_text, scene_image, *, rejected=None, confirmed_crop=None):
        self.calls.append(dict(mode="text", target_text=target_text, scene_image=scene_image,
                               rejected=rejected, confirmed_crop=confirmed_crop))
        if self.error:
            raise self.error
        return _Result(self.detection)

    def detect_personalized(self, target_text, reference_images, scene_image, *, rejected=None, confirmed_crop=None):
        self.calls.append(dict(mode="personalized", target_text=target_text, reference_images=reference_images,
                               scene_image=scene_image, rejected=rejected, confirmed_crop=confirmed_crop))
        if self.error:
            raise self.error
        return _Result(self.detection)


def _packet(seq: int = 7) -> FramePacket:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, 320:] = (0, 128, 255)
    return FramePacket(frame=frame, seq=seq, captured_at=time.monotonic(), captured_wall=time.time())


def _vision(omni: _Omni, **kwargs) -> LiveVision:
    return LiveVision(
        fast_config(), detect_text_only=omni.detect_text_only, detect_personalized=omni.detect_personalized, **kwargs
    )


@pytest.fixture
def omni_key(monkeypatch):
    monkeypatch.setattr(settings, "omni_api_key", "test-key")


def test_is_a_live_vision_port():
    vision = LiveVision(fast_config())
    assert isinstance(vision, VisionPort)
    assert vision.is_simulated is False


def test_blank_key_raises_instead_of_inventing_a_detection():
    omni = _Omni()
    assert settings.omni_api_key == ""                     # tests/conftest.py blanks it
    with pytest.raises(RuntimeError, match="OMNI_API_KEY is not set"):
        _vision(omni).identify(_packet(), IdentifyRequest(target_text="my bottle"))
    assert omni.calls == []


def test_text_only_call_shape_and_result_mapping(omni_key):
    omni = _Omni()
    result = _vision(omni).identify(_packet(seq=7), IdentifyRequest(target_text="my bottle beside my backpack"))

    [call] = omni.calls
    assert call["mode"] == "text"
    assert call["target_text"] == "my bottle beside my backpack"   # OMNI always gets the full text
    assert call["rejected"] is None and call["confirmed_crop"] is None   # -> today's prompt, byte for byte

    # The scene is Aleesha's 768px JPEG, and the pixels the box refers to come back with it.
    decoded = cv2.imdecode(np.frombuffer(call["scene_image"], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (576, 768)
    assert result.scene_jpeg == call["scene_image"] and result.resized_frame.shape[:2] == (576, 768)

    assert (result.found, result.likelihood, result.box, result.description) == (
        True, "high", [100, 200, 300, 400], "a blue bottle",
    )
    assert result.latency_seconds == 1.25 and result.frame_seq == 7


def test_reference_images_use_the_personalized_mode(omni_key):
    omni = _Omni()
    request = IdentifyRequest(target_text="my keys", reference_images=[b"ref-1", b"ref-2"])
    _vision(omni).identify(_packet(), request)
    [call] = omni.calls
    assert call["mode"] == "personalized" and call["reference_images"] == [b"ref-1", b"ref-2"]
    assert call["confirmed_crop"] is None


def test_confirmed_crop_alone_uses_the_personalized_mode(omni_key):
    omni = _Omni()
    request = IdentifyRequest(target_text="my bottle", confirmed_crop=b"accepted-crop")
    _vision(omni).identify(_packet(), request)
    [call] = omni.calls
    assert call["mode"] == "personalized" and call["reference_images"] == []
    assert call["confirmed_crop"] == b"accepted-crop"


def test_rejected_hints_are_passed_through(omni_key):
    omni = _Omni()
    hints = [RejectedHint("a green bottle", b"crop"), RejectedHint("a red mug")]
    _vision(omni).identify(_packet(), IdentifyRequest(target_text="my bottle", rejected=hints))
    assert omni.calls[0]["rejected"] == hints


def test_not_found_maps_to_no_box(omni_key):
    omni = _Omni(_Detection(found=False, likelihood="low", box_2d=None, description="nothing like that here"))
    result = _vision(omni).identify(_packet(), IdentifyRequest(target_text="my bottle"))
    assert (result.found, result.box) == (False, None)


def test_gateway_errors_propagate_to_the_controller(omni_key):
    omni = _Omni(error=TimeoutError("gateway timed out"))
    with pytest.raises(TimeoutError):
        _vision(omni).identify(_packet(), IdentifyRequest(target_text="my bottle"))


# --- local screening ---------------------------------------------------------------


@dataclass
class _ClassDetection:
    box_2d: list
    confidence: float


class _Detector:
    def __init__(self, detections=None, available=True, error=None):
        self.detections, self.available, self.error = detections or [], available, error
        self.calls = []

    def is_available(self):
        return self.available

    def detect_class_all(self, frame, coco_class):
        self.calls.append((frame.shape, coco_class))
        if self.error:
            raise self.error
        return self.detections


def test_screen_returns_every_valid_detection():
    detector = _Detector([
        _ClassDetection([100, 100, 300, 200], 0.9),
        _ClassDetection([300, 200, 100, 100], 0.8),      # inverted: dropped
        _ClassDetection([400, 600, 700, 700], 0.6),
    ])
    detections = LiveVision(fast_config(), detector=detector).screen(_packet(), "bottle")
    assert [(d.box, d.confidence) for d in detections] == [([100, 100, 300, 200], 0.9), ([400, 600, 700, 700], 0.6)]
    assert detector.calls == [((480, 640, 3), "bottle")]


def test_screen_empty_list_means_looked_and_saw_nothing():
    assert LiveVision(fast_config(), detector=_Detector([])).screen(_packet(), "bottle") == []


def test_screen_is_none_when_unavailable_disabled_or_broken():
    unavailable = _Detector(available=False)
    assert LiveVision(fast_config(), detector=unavailable).screen(_packet(), "bottle") is None
    assert unavailable.calls == []

    disabled = _Detector([_ClassDetection([1, 2, 3, 4], 0.9)])
    assert LiveVision(fast_config(local_detection_enabled=False), detector=disabled).screen(_packet(), "bottle") is None
    assert disabled.calls == []

    broken = _Detector(error=RuntimeError("model exploded"))
    vision = LiveVision(fast_config(), detector=broken)
    assert vision.screen(_packet(), "bottle") is None      # never raises: it is only an accelerator
    assert vision.screen(_packet(), "bottle") is None
