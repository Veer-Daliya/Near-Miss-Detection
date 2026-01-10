"""Trajectory tracking with velocity estimation."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


class TrajectoryTracker:
    """Tracks object trajectories and estimates velocities.

    Stores position history per track_id (last 10 positions) and calculates
    velocity from position deltas with exponential moving average smoothing.
    """

    def __init__(self, history_length: int = 10, ema_alpha: float = 0.3):
        """Initialize the trajectory tracker.

        Args:
            history_length: Number of positions to store per track.
            ema_alpha: Smoothing factor for exponential moving average (0-1).
                      Higher values give more weight to recent observations.
        """
        self.history_length = history_length
        self.ema_alpha = ema_alpha

        # Store position and timestamp history per track_id
        self._position_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self._timestamp_history: Dict[int, List[float]] = defaultdict(list)

        # Store smoothed velocity per track_id
        self._smoothed_velocity: Dict[int, Optional[Tuple[float, float]]] = {}

    def update(
        self, track_id: int, position: Tuple[float, float], timestamp: float
    ) -> Optional[Tuple[float, float]]:
        """Update trajectory with new position and return estimated velocity.

        Args:
            track_id: Unique identifier for the tracked object.
            position: Ground plane position (x, y) in meters.
            timestamp: Time of observation in seconds.

        Returns:
            Estimated velocity (vx, vy) in m/s, or None if insufficient history.
        """
        positions = self._position_history[track_id]
        timestamps = self._timestamp_history[track_id]

        # Add new observation
        positions.append(position)
        timestamps.append(timestamp)

        # Trim to history length
        if len(positions) > self.history_length:
            positions.pop(0)
            timestamps.pop(0)

        # Need at least 2 positions to calculate velocity
        if len(positions) < 2:
            return None

        # Calculate instantaneous velocity from last two positions
        dt = timestamps[-1] - timestamps[-2]
        if dt <= 0:
            return self._smoothed_velocity.get(track_id)

        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        instant_velocity = (dx / dt, dy / dt)

        # Apply exponential moving average smoothing
        prev_velocity = self._smoothed_velocity.get(track_id)
        if prev_velocity is None:
            smoothed = instant_velocity
        else:
            smoothed = (
                self.ema_alpha * instant_velocity[0]
                + (1 - self.ema_alpha) * prev_velocity[0],
                self.ema_alpha * instant_velocity[1]
                + (1 - self.ema_alpha) * prev_velocity[1],
            )

        self._smoothed_velocity[track_id] = smoothed
        return smoothed

    def get_velocity(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Get the current smoothed velocity for a track.

        Args:
            track_id: Unique identifier for the tracked object.

        Returns:
            Smoothed velocity (vx, vy) in m/s, or None if not available.
        """
        return self._smoothed_velocity.get(track_id)

    def get_position_history(
        self, track_id: int
    ) -> List[Tuple[float, float]]:
        """Get the position history for a track.

        Args:
            track_id: Unique identifier for the tracked object.

        Returns:
            List of positions, oldest to newest.
        """
        return list(self._position_history.get(track_id, []))

    def clear_track(self, track_id: int) -> None:
        """Remove all history for a track.

        Args:
            track_id: Unique identifier for the tracked object to clear.
        """
        self._position_history.pop(track_id, None)
        self._timestamp_history.pop(track_id, None)
        self._smoothed_velocity.pop(track_id, None)

    def clear_all(self) -> None:
        """Remove all tracking history."""
        self._position_history.clear()
        self._timestamp_history.clear()
        self._smoothed_velocity.clear()
