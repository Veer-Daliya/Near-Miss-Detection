"""Risk assessment module for near-miss detection.

This module provides trajectory tracking, collision prediction, and
near-miss event detection for pedestrian-vehicle interactions.
"""

from .collision_predictor import CollisionPredictor
from .near_miss_detector import NearMissDetector
from .risk_types import CollisionRisk, ObjectState
from .trajectory import TrajectoryTracker

__all__ = [
    "CollisionPredictor",
    "CollisionRisk",
    "NearMissDetector",
    "ObjectState",
    "TrajectoryTracker",
]
