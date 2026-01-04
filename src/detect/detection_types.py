"""Data types for object detection."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Detection:
    """Represents a detected object."""

    bbox: List[int]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    frame_id: int
    track_id: Optional[int] = None  # Assigned by tracker

    def __post_init__(self) -> None:
        """Validate bbox format."""
        if len(self.bbox) != 4:
            raise ValueError("bbox must have 4 elements [x1, y1, x2, y2]")



