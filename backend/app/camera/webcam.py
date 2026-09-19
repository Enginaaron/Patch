import logging

import cv2
import numpy as np

from app.camera.base import CameraSource

logger = logging.getLogger(__name__)


class WebcamCamera(CameraSource):
    def __init__(self, camera_index: int) -> None:
        self._camera_index = camera_index
        self._capture: cv2.VideoCapture | None = None

    def start(self) -> None:
        capture = cv2.VideoCapture(self._camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"could not open camera at index {self._camera_index}")
        self._capture = capture
        logger.info("opened VideoCapture on camera index %s", self._camera_index)

    def get_frame(self) -> np.ndarray | None:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
