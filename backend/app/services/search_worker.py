"""Frame / image helpers for search detection, plus compatibility shims.

What changed (for Aleesha)
--------------------------
This module used to run one ``SearchWorker`` thread per search that polled the
shared camera frame, gated OMNI on YOLO, and created candidates. Now that the
rover moves on its own, a search and the wheels have to be driven by ONE loop
(look only while stopped, move only between looks, stop the instant the user
says so), so that loop lives in ``app/rover/controller.py``. The recognition
pipeline itself is unchanged and still yours:

* the controller calls ``omni_vision.detect_text_only`` /
  ``detect_personalized`` and ``yolo_detector`` through
  ``app/rover/vision.py`` (``LiveVision``);
* it prepares frames and saves candidate images with the helpers below
  (``_prepare_frame``, ``_save_search_image``, ``_crop_jpeg``,
  ``_meets_threshold``), so candidate images, ``Candidate`` rows and the
  ``candidate_found`` event payload are exactly what the frontend already
  expects.

Moving the loop also removed four races the per-search thread had:
``stop()`` only set an event (an OMNI call in flight still finished and acted),
``_check_frame`` could complete after a cancel, ``_create_candidate`` did not
re-check the search status (a CANCELLED search could flip back to
CANDIDATE_PENDING), and an old worker's cleanup could unregister its
replacement. The controller's generation token closes all four.

``start_worker`` / ``stop_worker`` remain as thin shims so existing callers
keep working.
"""

import io
import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.services.bbox import crop_to_box

logger = logging.getLogger(__name__)

FRAME_TARGET_WIDTH = 768
JPEG_QUALITY = 85

_LIKELIHOOD_ORDER = {"low": 0, "medium": 1, "high": 2}


def start_worker(search_id: str) -> None:
    """Compatibility shim: start the rover mission for ``search_id``.

    Delegates to ``RoverController.start_search``. Raises ``RoverBusyError``
    when another search currently owns the rover (there is one body; the old
    per-search threads could run side by side, missions cannot).
    """
    # Imported here: the controller imports this module's helpers.
    from app.rover.controller import get_rover_controller

    get_rover_controller().start_search(search_id)


def stop_worker(search_id: str) -> None:
    """Compatibility shim: halt the mission of ``search_id`` if it is the
    active one. Does NOT touch ``Search.status`` -- use
    ``RoverController.cancel_search`` to cancel a search."""
    from app.rover.controller import get_rover_controller

    get_rover_controller().stop_search(search_id)


def _prepare_frame(frame: np.ndarray) -> tuple[np.ndarray, bytes]:
    """Resize to ~768px wide (preserving aspect ratio) and JPEG-encode.

    Returns the resized BGR array too, so a later crop uses the exact same
    pixel dimensions box_2d was computed against.
    """
    height, width = frame.shape[:2]
    scale = FRAME_TARGET_WIDTH / width
    resized = cv2.resize(frame, (FRAME_TARGET_WIDTH, round(height * scale)))

    ok, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("failed to JPEG-encode frame")
    return resized, buffer.tobytes()


def _crop_jpeg(resized_frame: np.ndarray, box_2d: list[int]) -> bytes:
    """JPEG of ``box_2d`` cut out of the resized BGR frame the box refers to
    (the crop step of the old ``SearchWorker._create_candidate``)."""
    rgb_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
    crop = crop_to_box(rgb_image, box_2d)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _save_search_image(search_id: str, image_bytes: bytes, suffix: str) -> str:
    search_dir = Path(settings.media_root) / "searches" / search_id
    search_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4()}_{suffix}.jpg"
    (search_dir / filename).write_bytes(image_bytes)
    return f"searches/{search_id}/{filename}"


def _meets_threshold(likelihood: str, threshold: str | None = None) -> bool:
    """``likelihood`` >= ``threshold`` (default: CANDIDATE_LIKELIHOOD_THRESHOLD)."""
    minimum = settings.candidate_likelihood_threshold if threshold is None else threshold
    return _LIKELIHOOD_ORDER[likelihood] >= _LIKELIHOOD_ORDER[minimum]
