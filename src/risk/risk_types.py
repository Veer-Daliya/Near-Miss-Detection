"""Data types for risk assessment module."""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ObjectState:
    """Represents the state of a tracked object at a given moment."""

    track_id: int
    position_image: Tuple[float, float]  # bbox center (x, y)
    position_ground: Optional[Tuple[float, float]]  # meters from camera
    velocity: Optional[Tuple[float, float]]  # m/s (vx, vy)
    class_name: str  # "person", "car", "truck", etc.
    bbox: List[int]  # [x1, y1, x2, y2]


@dataclass
class CollisionRisk:
    """Represents a collision risk assessment between a pedestrian and vehicle."""

    pedestrian_track_id: int
    vehicle_track_id: int
    ttc: float  # Time To Collision in seconds
    min_distance: float  # meters at closest point
    risk_level: str  # "critical" (<1.5s), "warning" (<3s), "safe" (>=3s)
    frame_number: int
