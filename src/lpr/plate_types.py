"""Data types for license plate recognition."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PlateResult:
    """Represents a detected and recognized license plate."""

    text: str  # Plate text or "UNKNOWN"
    confidence: float  # 0.0 - 1.0
    bbox: List[int]  # [x1, y1, x2, y2] in frame coordinates
    vehicle_track_id: Optional[int] = None  # Associated vehicle track ID

