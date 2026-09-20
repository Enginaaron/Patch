from abc import ABC, abstractmethod

import numpy as np


class CameraSource(ABC):
    """A frame producer driven by CameraService's capture thread.

    ``get_frame()`` is called in a loop from that one thread. It should block
    until the next frame is ready (a synthetic source paces itself), return a
    new BGR array each time -- the service keeps a reference to it -- and
    return None when a read fails. The service timestamps successful reads
    only, so a source must never hand back an old frame to paper over a
    failure: that would make a dead camera look alive to the rover.
    """

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def get_frame(self) -> np.ndarray | None: ...

    @abstractmethod
    def stop(self) -> None: ...
