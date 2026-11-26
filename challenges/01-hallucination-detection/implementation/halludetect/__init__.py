"""Challenge 01: hallucination detection for context-grounded QA."""

from halludetect.dataset import load_dataset
from halludetect.types import Detection, Example

__all__ = ["Detection", "Example", "load_dataset"]
