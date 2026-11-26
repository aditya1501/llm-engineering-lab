from __future__ import annotations

from abc import ABC, abstractmethod

from halludetect.types import Detection, Example


class Detector(ABC):
    name: str = "base"
    #: True if the method can point at specific unsupported claims.
    claim_level: bool = False

    @abstractmethod
    def detect(self, example: Example) -> Detection: ...
