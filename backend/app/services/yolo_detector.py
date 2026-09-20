import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# torch / ultralytics are imported lazily, on first model load, instead of at
# module import. They are heavy (slow startup on the Pi) and optional: when
# they are not installed the rover falls back to OMNI sampling instead of the
# whole backend failing to import. See _configure_threads() for why the thread
# caps still run before any inference.


def _configure_threads() -> None:
    """PyTorch's CPU inference and OpenCV's internal preprocessing (resize,
    color conversion, etc.) each maintain their OWN thread pool, and both
    default to every available core -- measured live on a 16-core dev machine,
    a single YOLOv8n call at ~10fps was consuming ~7.5 cores continuously.
    That directly contradicts "fast, local, free... cheap". Capping all three
    pools (intra-op, inter-op, OpenCV) keeps each inference call fast (the nano
    model doesn't need 16-way parallelism to hit real-time) without the
    contention. torch.set_num_interop_threads() can only be called once,
    before any inter-op parallel work has happened, so this runs right after
    the first torch import and before the model is loaded or used."""
    import cv2
    import torch

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass  # already initialized elsewhere; not worth failing startup over
    cv2.setNumThreads(2)


# Acceleration path only, per Spec 11 -- YOLO never decides "this is the
# user's item," only "something worth checking is here." Most free-text
# targets won't map to any of YOLOv8n's 80 fixed COCO classes; that's
# expected, not a gap to close here. The demo target for this hackathon is
# "phone" -- the map is written to extend easily, but only that class is
# validated end-to-end.
COCO_TARGET_MAP = {
    "phone": "cell phone",
    "cell phone": "cell phone",
    "iphone": "cell phone",
    "backpack": "backpack",
    "bag": "backpack",
    "handbag": "handbag",
    "purse": "handbag",
    "suitcase": "suitcase",
    "remote": "remote",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "book": "book",
    "bottle": "bottle",
    "laptop": "laptop",
}


def match_coco_class(target_text: str) -> str | None:
    """COCO class to screen for, or None to fall back to OMNI sampling.

    Delegates to the comprehension adapter, which matches COCO_TARGET_MAP
    keywords on word boundaries and only inside the target phrase. The old
    substring match screened for a backpack when asked for "my bottle beside
    my backpack", and for a phone on "headphones".
    """
    # Imported here: comprehension imports COCO_TARGET_MAP from this module.
    from app.services.comprehension import resolve_target

    return resolve_target(target_text).category


_model_lock = threading.Lock()
_model = None
_unavailable_reason: str | None = None


def _get_model():
    global _model, _unavailable_reason
    if _model is None:
        with _model_lock:
            if _model is None:
                if _unavailable_reason is not None:
                    raise RuntimeError(_unavailable_reason)
                try:
                    _configure_threads()
                    from ultralytics import YOLO
                except Exception as exc:  # noqa: BLE001 - missing/broken optional dependency
                    _unavailable_reason = f"local YOLO detection unavailable: {exc}"
                    logger.warning("%s -- falling back to OMNI sampling", _unavailable_reason)
                    raise RuntimeError(_unavailable_reason) from exc
                logger.info("loading YOLOv8n model...")
                _model = YOLO("yolov8n.pt")
    return _model


def is_available() -> bool:
    """True when the local detector can run (loads the model on first call)."""
    try:
        _get_model()
    except Exception:  # noqa: BLE001
        return False
    return True


@dataclass(frozen=True)
class ClassDetection:
    box_2d: list[int]
    confidence: float


def detect_class_all(frame: np.ndarray, coco_class: str, min_confidence: float = 0.0) -> list[ClassDetection]:
    """Every detection of `coco_class` in `frame` (BGR), highest confidence
    first, as box_2d [ymin, xmin, ymax, xmax] on the 0-1000 scale.

    YOLO is a per-frame detector, not an identity tracker: two bottles give two
    boxes with nothing tying either to "the one the user accepted". Callers
    that need identity must associate boxes across frames themselves and
    revalidate with OMNI (see app/rover).
    """
    model = _get_model()
    results = model.predict(frame, verbose=False)[0]

    height, width = frame.shape[:2]
    detections: list[ClassDetection] = []

    for box in results.boxes:
        class_name = model.names[int(box.cls[0])]
        if class_name != coco_class:
            continue
        conf = float(box.conf[0])
        if conf < min_confidence:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            ClassDetection(
                box_2d=[
                    max(0, min(1000, round(y1 / height * 1000))),
                    max(0, min(1000, round(x1 / width * 1000))),
                    max(0, min(1000, round(y2 / height * 1000))),
                    max(0, min(1000, round(x2 / width * 1000))),
                ],
                confidence=conf,
            )
        )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def detect_class(frame: np.ndarray, coco_class: str) -> list[int] | None:
    """Run YOLOv8n on `frame` (BGR, as produced by camera_service).

    Returns box_2d [ymin, xmin, ymax, xmax] on the same 0-1000 scale used
    everywhere else in this codebase (Spec 7/10), for the highest-confidence
    detection of `coco_class`, or None if it isn't present in this frame.
    """
    detections = detect_class_all(frame, coco_class)
    return detections[0].box_2d if detections else None


# Shared state so the MJPEG stream (a separate consumer of the same latest
# frame, per Spec 3/4) can draw whatever the YOLO-gated worker last saw,
# without the two being coupled beyond this one value.
_tracking_lock = threading.Lock()
_tracking_box: list[int] | None = None


def set_tracking_box(box_2d: list[int] | None) -> None:
    global _tracking_box
    with _tracking_lock:
        _tracking_box = box_2d


def get_tracking_box() -> list[int] | None:
    with _tracking_lock:
        return _tracking_box
