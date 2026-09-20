import logging
import threading

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# PyTorch's CPU inference and OpenCV's internal preprocessing (resize, color
# conversion, etc.) each maintain their OWN thread pool, and both default to
# every available core -- measured live on this machine (16 logical cores),
# that meant a single YOLOv8n call at ~10fps was consuming ~7.5 cores
# continuously. That directly contradicts "fast, local, free... cheap" --
# it's supposed to be the lightweight layer, not the thing maxing out the
# machine. Capping all three pools (intra-op, inter-op, OpenCV) keeps each
# inference call fast (the nano model doesn't need 16-way parallelism to hit
# real-time) without the contention. torch.set_num_interop_threads() can
# only be called once, before any inter-op parallel work has happened, so
# this must run at import time, before any inference.
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
    """Normalize target_text and substring-match against COCO_TARGET_MAP."""
    normalized = target_text.strip().lower()
    for keyword, coco_class in COCO_TARGET_MAP.items():
        if keyword in normalized:
            return coco_class
    return None


_model_lock = threading.Lock()
_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info("loading YOLOv8n model...")
                _model = YOLO("yolov8n.pt")
    return _model


def detect_class(frame: np.ndarray, coco_class: str) -> list[int] | None:
    """Run YOLOv8n on `frame` (BGR, as produced by camera_service).

    Returns box_2d [ymin, xmin, ymax, xmax] on the same 0-1000 scale used
    everywhere else in this codebase (Spec 7/10), for the highest-confidence
    detection of `coco_class`, or None if it isn't present in this frame.
    """
    model = _get_model()
    results = model.predict(frame, verbose=False)[0]

    height, width = frame.shape[:2]
    best_box: list[int] | None = None
    best_conf = -1.0

    for box in results.boxes:
        class_name = model.names[int(box.cls[0])]
        if class_name != coco_class:
            continue
        conf = float(box.conf[0])
        if conf <= best_conf:
            continue
        best_conf = conf
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        best_box = [
            max(0, min(1000, round(y1 / height * 1000))),
            max(0, min(1000, round(x1 / width * 1000))),
            max(0, min(1000, round(y2 / height * 1000))),
            max(0, min(1000, round(x2 / width * 1000))),
        ]

    return best_box


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
