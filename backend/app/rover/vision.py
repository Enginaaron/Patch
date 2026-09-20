"""Live recognition for the rover: Aleesha's OMNI + YOLO pipeline behind the
``VisionPort`` the controller depends on.

Nothing here decides anything about movement, and nothing here can invent a
detection: without an OMNI key ``identify`` raises, and without a local
detector ``screen`` reports "unavailable" (None) so the controller falls back
to asking OMNI on every stop.

What the two layers can and cannot say:

* YOLO (``screen``) is a per-frame category detector. It does not track
  identity -- two bottles are two boxes, with nothing tying either of them to
  the one the user accepted.
* OMNI (``identify``) returns the single strongest candidate for the
  description. The schema has no list of alternatives.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from app.config import settings
from app.rover.config import RoverConfig
from app.rover.geometry import is_valid_box
from app.rover.types import FramePacket, IdentifyRequest, IdentifyResult, LocalDetection

logger = logging.getLogger(__name__)


class LiveVision:
    """``VisionPort`` backed by ``omni_vision`` and ``yolo_detector``.

    The callables are injectable so tests never import a model or touch the
    network; by default they resolve lazily to the real modules.
    """

    is_simulated = False

    def __init__(
        self,
        config: RoverConfig | None = None,
        *,
        detect_text_only: Callable[..., Any] | None = None,
        detect_personalized: Callable[..., Any] | None = None,
        detector: Any | None = None,
        prepare_frame: Callable[[Any], tuple[Any, bytes]] | None = None,
    ) -> None:
        self._config = config
        self._detect_text_only = detect_text_only
        self._detect_personalized = detect_personalized
        self._detector = detector
        self._prepare_frame = prepare_frame
        self._screen_failure_logged = False

    # --- lazy wiring to Aleesha's modules ---------------------------------

    def _local_detection_enabled(self) -> bool:
        if self._config is not None:
            return self._config.local_detection_enabled
        return settings.local_detection_enabled

    def _yolo(self) -> Any:
        if self._detector is None:
            from app.services import yolo_detector

            self._detector = yolo_detector
        return self._detector

    def _omni(self) -> tuple[Callable[..., Any], Callable[..., Any]]:
        if self._detect_text_only is None or self._detect_personalized is None:
            from app.services import omni_vision

            self._detect_text_only = self._detect_text_only or omni_vision.detect_text_only
            self._detect_personalized = self._detect_personalized or omni_vision.detect_personalized
        return self._detect_text_only, self._detect_personalized

    def _prepare(self, frame: Any) -> tuple[Any, bytes]:
        if self._prepare_frame is None:
            from app.services.search_worker import _prepare_frame

            self._prepare_frame = _prepare_frame
        return self._prepare_frame(frame)

    # --- VisionPort ---------------------------------------------------------

    def screen(self, packet: FramePacket, coco_class: str) -> list[LocalDetection] | None:
        """Every YOLO detection of ``coco_class`` in this frame, or None when
        local screening cannot run (disabled, torch/ultralytics missing, or the
        detector failed). Never raises: screening is only an accelerator."""
        if not self._local_detection_enabled():
            return None
        try:
            detector = self._yolo()
            if not detector.is_available():
                return None
            raw = detector.detect_class_all(packet.frame, coco_class)
        except Exception:  # noqa: BLE001 - the accelerator must never break a mission
            if not self._screen_failure_logged:
                logger.exception("local screening failed; falling back to OMNI sampling")
                self._screen_failure_logged = True
            return None

        detections: list[LocalDetection] = []
        for det in raw:
            box = list(det.box_2d)
            if not is_valid_box(box):
                logger.warning("dropping invalid local detection box %r", box)
                continue
            detections.append(LocalDetection(box=box, confidence=float(det.confidence)))
        return detections

    def identify(self, packet: FramePacket, request: IdentifyRequest) -> IdentifyResult:
        """One blocking OMNI call on this frame. Raises on any failure; the
        controller bounds it with a timeout and drops late results."""
        if not settings.omni_api_key.strip():
            # Never fabricate a detection: no key means no recognition.
            raise RuntimeError("OMNI_API_KEY is not set")

        resized_frame, scene_jpeg = self._prepare(packet.frame)
        # Preserve aa_dev's blur gate on real camera frames. A rejected frame
        # causes a stationary retry, never a no-match that authorizes movement.
        if self._prepare_frame is not None and hasattr(resized_frame, "shape"):
            from app.services.search_worker import _laplacian_variance

            if _laplacian_variance(resized_frame) < settings.blur_variance_threshold:
                raise RuntimeError("Camera image is too blurry; keep the camera clear and steady")
        detect_text_only, detect_personalized = self._omni()
        rejected = list(request.rejected) or None

        if request.reference_images or request.confirmed_crop is not None:
            result = detect_personalized(
                request.target_text,
                list(request.reference_images),
                scene_jpeg,
                rejected=rejected,
                confirmed_crop=request.confirmed_crop,
            )
        else:
            result = detect_text_only(
                request.target_text,
                scene_jpeg,
                rejected=rejected,
                confirmed_crop=None,
            )

        detection = result.detection
        box = list(detection.box_2d) if detection.box_2d is not None else None
        return IdentifyResult(
            found=bool(detection.found),
            likelihood=detection.likelihood,
            box=box,
            description=detection.description,
            latency_seconds=float(result.latency_seconds),
            frame_seq=packet.seq,
            resized_frame=resized_frame,
            scene_jpeg=scene_jpeg,
        )
