"""Collision prediction using Closest Point of Approach (CPA)."""

from typing import Optional

import numpy as np

from .risk_types import CollisionRisk, ObjectState


class CollisionPredictor:
    """Predicts collision risk between pedestrians and vehicles using CPA.

    Uses the Closest Point of Approach (CPA) algorithm to estimate:
    - Time to collision (TTC): when objects will be closest
    - Minimum distance: how close they will get

    Risk levels:
    - Critical: TTC < 1.5s AND distance_at_cpa < 2m
    - Warning: TTC < 3.0s AND distance_at_cpa < 3m
    - Safe: otherwise
    """

    # Risk level thresholds
    CRITICAL_TTC = 1.5  # seconds
    CRITICAL_DISTANCE = 2.0  # meters
    WARNING_TTC = 3.0  # seconds
    WARNING_DISTANCE = 3.0  # meters

    def __init__(
        self,
        critical_ttc: float = CRITICAL_TTC,
        critical_distance: float = CRITICAL_DISTANCE,
        warning_ttc: float = WARNING_TTC,
        warning_distance: float = WARNING_DISTANCE,
        max_prediction_time: float = 10.0,
    ):
        """Initialize the collision predictor.

        Args:
            critical_ttc: TTC threshold for critical risk (seconds).
            critical_distance: Distance threshold for critical risk (meters).
            warning_ttc: TTC threshold for warning risk (seconds).
            warning_distance: Distance threshold for warning risk (meters).
            max_prediction_time: Maximum time horizon for predictions (seconds).
        """
        self.critical_ttc = critical_ttc
        self.critical_distance = critical_distance
        self.warning_ttc = warning_ttc
        self.warning_distance = warning_distance
        self.max_prediction_time = max_prediction_time

    def predict(
        self,
        pedestrian: ObjectState,
        vehicle: ObjectState,
        frame_number: int,
    ) -> Optional[CollisionRisk]:
        """Predict collision risk between a pedestrian and vehicle.

        Uses Closest Point of Approach (CPA) calculation:
        - t_cpa = -dot(relative_position, relative_velocity) / |relative_velocity|^2
        - distance_at_cpa = |relative_position + relative_velocity * t_cpa|

        Args:
            pedestrian: State of the pedestrian.
            vehicle: State of the vehicle.
            frame_number: Current frame number for the assessment.

        Returns:
            CollisionRisk if both objects have ground positions and velocities,
            None otherwise.
        """
        # Need ground positions and velocities for both objects
        if pedestrian.position_ground is None or vehicle.position_ground is None:
            return None
        if pedestrian.velocity is None or vehicle.velocity is None:
            return None

        # Calculate relative position and velocity (vehicle relative to pedestrian)
        rel_pos = np.array(
            [
                vehicle.position_ground[0] - pedestrian.position_ground[0],
                vehicle.position_ground[1] - pedestrian.position_ground[1],
            ]
        )
        rel_vel = np.array(
            [
                vehicle.velocity[0] - pedestrian.velocity[0],
                vehicle.velocity[1] - pedestrian.velocity[1],
            ]
        )

        # Calculate time to closest point of approach
        rel_vel_squared = np.dot(rel_vel, rel_vel)

        if rel_vel_squared < 1e-10:
            # Objects are moving together or both stationary
            # Use current distance as minimum distance
            t_cpa = 0.0
            min_distance = float(np.linalg.norm(rel_pos))
        else:
            # t_cpa = -dot(relative_position, relative_velocity) / |relative_velocity|^2
            t_cpa = -np.dot(rel_pos, rel_vel) / rel_vel_squared

            # Clamp to valid time range (0 to max_prediction_time)
            # Negative t_cpa means closest point was in the past
            t_cpa = max(0.0, min(t_cpa, self.max_prediction_time))

            # Calculate position at CPA
            pos_at_cpa = rel_pos + rel_vel * t_cpa
            min_distance = float(np.linalg.norm(pos_at_cpa))

        # Classify risk level
        risk_level = self._classify_risk(t_cpa, min_distance)

        return CollisionRisk(
            pedestrian_track_id=pedestrian.track_id,
            vehicle_track_id=vehicle.track_id,
            ttc=t_cpa,
            min_distance=min_distance,
            risk_level=risk_level,
            frame_number=frame_number,
        )

    def _classify_risk(self, ttc: float, min_distance: float) -> str:
        """Classify risk level based on TTC and minimum distance.

        Args:
            ttc: Time to collision in seconds.
            min_distance: Minimum distance at closest point in meters.

        Returns:
            Risk level string: "critical", "warning", or "safe".
        """
        if ttc < self.critical_ttc and min_distance < self.critical_distance:
            return "critical"
        elif ttc < self.warning_ttc and min_distance < self.warning_distance:
            return "warning"
        else:
            return "safe"
