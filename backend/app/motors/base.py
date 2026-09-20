from abc import ABC, abstractmethod


class MotorDriver(ABC):
    """Lowest-level drive interface: set signed speeds for the two wheels.

    Speeds are normalized to ``[-1.0, 1.0]`` where ``+1`` is full forward and
    ``-1`` is full reverse for that wheel. Everything above this (turning,
    scanning, approaching) is expressed in terms of ``set_speeds``.
    """

    name: str = "motor"

    @abstractmethod
    def set_speeds(self, left: float, right: float) -> None: ...

    def stop(self) -> None:
        self.set_speeds(0.0, 0.0)

    def close(self) -> None:  # release hardware resources, if any
        self.stop()
