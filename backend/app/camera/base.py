from abc import ABC, abstractmethod

import numpy as np


class CameraSource(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def get_frame(self) -> np.ndarray | None: ...

    @abstractmethod
    def stop(self) -> None: ...
