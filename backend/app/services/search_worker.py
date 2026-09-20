import io
import json
import logging
import threading
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image
from sqlmodel import select

from app.config import settings
from app.db import get_session
from app.models import Candidate, CandidateDecision, ReferenceImage, Search, SearchStatus
from app.services.bbox import crop_to_box
from app.services.camera_service import camera_service
from app.services.event_bus import event_bus
from app.services.omni_vision import DetectionResult, detect_personalized, detect_text_only
from app.services.yolo_detector import detect_class, match_coco_class, set_tracking_box

logger = logging.getLogger(__name__)

FRAME_TARGET_WIDTH = 768
JPEG_QUALITY = 85

# YOLO polls much faster than AI_SAMPLE_INTERVAL_SECONDS since it's meant to
# be the "near-instant" local layer -- the OMP_WAIT_POLICY/KMP_BLOCKTIME fix
# (see main.py) stopped the OpenMP/MKL thread pool from spin-waiting between
# calls, which was what actually drove CPU usage. 5Hz (200ms) still reads as
# near-instant against OMNI's multi-second latency while keeping total CPU
# modest on less powerful hardware than this dev machine's 16 cores.
YOLO_POLL_INTERVAL_SECONDS = 0.2

_LIKELIHOOD_ORDER = {"low": 0, "medium": 1, "high": 2}

# Spec 15: how many of the most-recently-rejected crops get fed back into
# subsequent OMNI prompts as "don't suggest this again" context.
REJECTED_CONTEXT_LIMIT = 2

# Spec 21: OMNI failure backoff -- doubles from this base on each consecutive
# failure, capped at the max, reset to normal cadence on the next success.
# Deliberately independent of AI_SAMPLE_INTERVAL_SECONDS/YOLO_POLL_INTERVAL_SECONDS:
# this is about not hammering a failing OMNI endpoint, not camera sampling rate.
VISION_BACKOFF_BASE_SECONDS = 2.0
VISION_BACKOFF_MAX_SECONDS = 30.0

_active_workers: dict[str, "SearchWorker"] = {}
_registry_lock = threading.Lock()


def start_worker(search_id: str) -> None:
    with _registry_lock:
        if search_id in _active_workers:
            return
        worker = SearchWorker(search_id)
        _active_workers[search_id] = worker
    worker.start()


def stop_worker(search_id: str) -> None:
    with _registry_lock:
        worker = _active_workers.pop(search_id, None)
    if worker:
        worker.stop()


def _unregister(search_id: str) -> None:
    with _registry_lock:
        _active_workers.pop(search_id, None)


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


def _laplacian_variance(frame: np.ndarray) -> float:
    """Standard variance-of-Laplacian blur metric. Computed on the same
    resized frame that would be sent to OMNI, not the raw camera frame, so
    the configured threshold means the same thing regardless of the camera's
    native resolution."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _save_search_image(search_id: str, image_bytes: bytes, suffix: str) -> str:
    search_dir = Path(settings.media_root) / "searches" / search_id
    search_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4()}_{suffix}.jpg"
    (search_dir / filename).write_bytes(image_bytes)
    return f"searches/{search_id}/{filename}"


def _meets_threshold(likelihood: str) -> bool:
    return _LIKELIHOOD_ORDER[likelihood] >= _LIKELIHOOD_ORDER[settings.candidate_likelihood_threshold]


def _load_recent_rejected_crops(search_id: str) -> list[bytes]:
    with get_session() as session:
        rejected = session.exec(
            select(Candidate)
            .where(Candidate.search_id == search_id, Candidate.decision == CandidateDecision.REJECTED)
            .order_by(Candidate.created_at.desc())
            .limit(REJECTED_CONTEXT_LIMIT)
        ).all()
        crop_paths = [c.crop_path for c in rejected]

    return [Path(settings.media_root, p).read_bytes() for p in crop_paths]


class SearchWorker:
    """Polls the shared camera frame for one Search and calls OMNI at most
    once every AI_SAMPLE_INTERVAL_SECONDS. Strictly sequential -- the loop
    blocks on each OMNI call before considering the next frame, so there is
    never more than one request in flight and nothing is ever queued; the
    next call always uses whatever frame is latest when the previous one
    returns, never a frame captured mid-flight.
    """

    def __init__(self, search_id: str) -> None:
        self.search_id = search_id
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._call_count = 0
        self._consecutive_failures = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        # _tracking_box is a single global (yolo_detector.py), not scoped to
        # this search -- clear it before dispatching so a newly-started
        # search never renders a stale box left behind by whichever search
        # last used the YOLO-gated path, including one that isn't this one.
        set_tracking_box(None)
        try:
            search_info = self._load_search_info()
            if search_info is None:
                return
            target_text, _ = search_info

            coco_class = match_coco_class(target_text) if settings.local_detection_enabled else None
            if coco_class:
                logger.info(
                    "search %s: target %r mapped to COCO class %r, using YOLO-gated flow",
                    self.search_id,
                    target_text,
                    coco_class,
                )
                self._run_yolo_gated(coco_class)
            else:
                logger.info(
                    "search %s: no COCO mapping for %r, using Spec 12 timed sampling",
                    self.search_id,
                    target_text,
                )
                self._run_timed_sampling()
        finally:
            set_tracking_box(None)
            _unregister(self.search_id)

    def _run_timed_sampling(self) -> None:
        """Spec 12's original behavior, unchanged: sample every
        AI_SAMPLE_INTERVAL_SECONDS and send straight to OMNI. This is the
        fallback path for any target that doesn't map to a COCO class."""
        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            search_info = self._load_search_info()
            if search_info is None:
                return
            target_text, reference_paths = search_info

            frame = camera_service.get_frame()
            if frame is None:
                self._wait_remaining(loop_start, settings.ai_sample_interval_seconds)
                continue

            if self._check_frame(frame, target_text, reference_paths):
                return  # candidate created, pause

            self._wait_remaining(loop_start, settings.ai_sample_interval_seconds)

    def _run_yolo_gated(self, coco_class: str) -> None:
        """Spec 11: YOLO polls the shared frame continuously and cheaply.
        It only ever decides "something worth checking is here" -- OMNI still
        makes the actual match decision, via the same _check_frame path as
        the timed-sampling flow. Fires OMNI on a brand new detection, or
        after LOCAL_DETECTION_COOLDOWN_SECONDS if the object has stayed in
        view continuously (never on every single frame)."""
        was_present = False
        last_omni_call_time: float | None = None

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            search_info = self._load_search_info()
            if search_info is None:
                return
            target_text, reference_paths = search_info

            frame = camera_service.get_frame()
            if frame is None:
                set_tracking_box(None)
                was_present = False
                self._wait_remaining(loop_start, YOLO_POLL_INTERVAL_SECONDS)
                continue

            try:
                box_2d = detect_class(frame, coco_class)
            except Exception:
                logger.exception("YOLO detection failed for search %s", self.search_id)
                box_2d = None

            if box_2d is None:
                set_tracking_box(None)
                was_present = False
                self._wait_remaining(loop_start, YOLO_POLL_INTERVAL_SECONDS)
                continue

            # Draw the live tracking box regardless of whether this frame
            # also triggers an OMNI call -- it's a separate, faster-feedback
            # signal than the OMNI-confirmed candidate.
            set_tracking_box(box_2d)

            is_new_detection = not was_present
            cooldown_elapsed = (
                last_omni_call_time is None
                or (time.monotonic() - last_omni_call_time) >= settings.local_detection_cooldown_seconds
            )
            was_present = True

            if not (is_new_detection or cooldown_elapsed):
                self._wait_remaining(loop_start, YOLO_POLL_INTERVAL_SECONDS)
                continue

            last_omni_call_time = time.monotonic()
            if self._check_frame(frame, target_text, reference_paths):
                return  # candidate created, pause

            self._wait_remaining(loop_start, YOLO_POLL_INTERVAL_SECONDS)

    def _check_frame(self, frame: np.ndarray, target_text: str, reference_paths: list[str]) -> bool:
        """Prepares `frame`, calls OMNI (Mode A or B), and creates a
        Candidate if it meets the likelihood threshold. Shared by both the
        timed-sampling and YOLO-gated flows -- "same downstream logic" per
        Spec 11. Returns True if a candidate was created (caller should stop
        polling)."""
        try:
            resized_frame, scene_jpeg = _prepare_frame(frame)
        except Exception:
            logger.exception("frame prep failed for search %s", self.search_id)
            return False

        # Spec 21: skip inference on frames too blurry to be meaningfully
        # judged -- deliberately not a failure (no backoff, no vision_error).
        blur_variance = _laplacian_variance(resized_frame)
        if blur_variance < settings.blur_variance_threshold:
            logger.info(
                "search %s: skipping blurry frame (variance=%.1f < threshold=%.1f)",
                self.search_id,
                blur_variance,
                settings.blur_variance_threshold,
            )
            return False

        rejected_images = _load_recent_rejected_crops(self.search_id) or None

        try:
            if reference_paths:
                reference_bytes = [Path(settings.media_root, p).read_bytes() for p in reference_paths]
                result = detect_personalized(
                    target_text, reference_bytes, scene_jpeg, rejected_images=rejected_images
                )
            else:
                result = detect_text_only(target_text, scene_jpeg, rejected_images=rejected_images)
        except Exception as exc:
            # Spec 21: covers timeout, network failure, malformed OMNI output,
            # and invalid box_2d alike -- all of these surface as exceptions
            # from detect_personalized/detect_text_only (network/timeout from
            # the httpx/openai client, malformed JSON from _extract_json,
            # invalid box_2d from SearchDetection's pydantic validator).
            self._consecutive_failures += 1
            backoff_seconds = min(
                VISION_BACKOFF_BASE_SECONDS * (2 ** (self._consecutive_failures - 1)),
                VISION_BACKOFF_MAX_SECONDS,
            )
            logger.exception(
                "search %s: OMNI detection call failed (%d consecutive), backing off %.1fs",
                self.search_id,
                self._consecutive_failures,
                backoff_seconds,
            )
            event_bus.publish(
                self.search_id,
                {"type": "vision_error", "payload": {"message": str(exc)}},
            )
            # Interruptible sleep -- a cancel/reject during backoff still
            # stops promptly rather than waiting out the full delay.
            self._stop_event.wait(backoff_seconds)
            return False

        if self._consecutive_failures > 0:
            # Spec 21: "resume normal-cadence sampling automatically once a
            # request succeeds" -- tell the frontend explicitly too, since it
            # has no other signal that a quiet (nothing-found) success ever
            # happened to know the earlier vision_error is now stale.
            event_bus.publish(self.search_id, {"type": "vision_recovered", "payload": {}})
        self._consecutive_failures = 0

        self._call_count += 1
        logger.info(
            "search %s: OMNI call #%d complete in %.2fs (found=%s likelihood=%s, rejected_context=%d)",
            self.search_id,
            self._call_count,
            result.latency_seconds,
            result.detection.found,
            result.detection.likelihood,
            len(rejected_images or []),
        )

        if result.detection.found and _meets_threshold(result.detection.likelihood):
            self._create_candidate(resized_frame, scene_jpeg, result)
            return True
        return False

    def _load_search_info(self) -> tuple[str, list[str]] | None:
        with get_session() as session:
            search = session.get(Search, self.search_id)
            if search is None or search.status != SearchStatus.SEARCHING:
                logger.info(
                    "search %s no longer SEARCHING (status=%s), stopping worker",
                    self.search_id,
                    search.status if search else "missing",
                )
                return None

            reference_images = session.exec(
                select(ReferenceImage).where(ReferenceImage.item_id == search.item_id)
            ).all()
            return search.target_text, [r.image_path for r in reference_images]

    def _wait_remaining(self, loop_start: float, interval: float) -> None:
        elapsed = time.monotonic() - loop_start
        remaining = interval - elapsed
        if remaining > 0:
            self._stop_event.wait(remaining)

    def _create_candidate(self, resized_frame: np.ndarray, scene_jpeg: bytes, result: DetectionResult) -> None:
        full_path = _save_search_image(self.search_id, scene_jpeg, "full")

        box_2d = result.detection.box_2d
        if box_2d is not None:
            rgb_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            crop = crop_to_box(rgb_image, box_2d)
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=JPEG_QUALITY)
            crop_path = _save_search_image(self.search_id, buf.getvalue(), "crop")
        else:
            # found=true with no box is allowed by the schema -- fall back to
            # the full frame so Candidate.crop_path always has a valid image.
            crop_path = full_path

        with get_session() as session:
            search = session.get(Search, self.search_id)
            if search is None:
                return

            candidate = Candidate(
                search_id=self.search_id,
                full_image_path=full_path,
                crop_path=crop_path,
                bbox_json=json.dumps(box_2d),
                description=result.detection.description,
                decision=CandidateDecision.PENDING,
            )
            session.add(candidate)

            search.status = SearchStatus.CANDIDATE_PENDING
            session.add(search)
            session.commit()

            # session.commit() expires ORM attributes -- capture what we need
            # for the event payload while still inside the session, same fix
            # as the identical DetachedInstanceError hit in Spec 5.
            candidate_id = candidate.id

        logger.info("search %s: candidate created, status -> CANDIDATE_PENDING", self.search_id)
        event_bus.publish(
            self.search_id,
            {
                "type": "candidate_found",
                "payload": {
                    "candidate_id": candidate_id,
                    "image_url": f"/media/{full_path}",
                    "crop_url": f"/media/{crop_path}",
                    "box": box_2d,
                    "description": result.detection.description,
                },
            },
        )
