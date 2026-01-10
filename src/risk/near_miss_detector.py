"""Near miss detection orchestrating trajectory tracking and collision prediction."""

from typing import Any, Dict, List, Optional, Tuple

from .collision_predictor import CollisionPredictor
from .risk_types import CollisionRisk, ObjectState
from .trajectory import TrajectoryTracker


class NearMissDetector:
    """Detects near-miss events between pedestrians and vehicles.

    Orchestrates trajectory tracking and collision prediction to identify
    potential collision risks in video frames.
    """

    # Object classes considered as pedestrians
    PEDESTRIAN_CLASSES = {"person", "pedestrian"}

    # Object classes considered as vehicles
    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}

    def __init__(
        self,
        trajectory_history_length: int = 10,
        velocity_ema_alpha: float = 0.3,
        critical_ttc: float = 1.5,
        critical_distance: float = 2.0,
        warning_ttc: float = 3.0,
        warning_distance: float = 3.0,
        fps: float = 30.0,
    ):
        """Initialize the near miss detector.

        Args:
            trajectory_history_length: Number of positions to store per track.
            velocity_ema_alpha: Smoothing factor for velocity EMA.
            critical_ttc: TTC threshold for critical risk (seconds).
            critical_distance: Distance threshold for critical risk (meters).
            warning_ttc: TTC threshold for warning risk (seconds).
            warning_distance: Distance threshold for warning risk (meters).
            fps: Frames per second of the video (for timestamp calculation).
        """
        self.trajectory_tracker = TrajectoryTracker(
            history_length=trajectory_history_length,
            ema_alpha=velocity_ema_alpha,
        )
        self.collision_predictor = CollisionPredictor(
            critical_ttc=critical_ttc,
            critical_distance=critical_distance,
            warning_ttc=warning_ttc,
            warning_distance=warning_distance,
        )
        self.fps = fps

    def process_frame(
        self,
        frame_number: int,
        detections: List[Dict[str, Any]],
        ground_plane_estimator: Any,
    ) -> List[CollisionRisk]:
        """Process a frame and return collision risk events.

        Args:
            frame_number: Current frame number.
            detections: List of detection dictionaries with keys:
                - track_id: int
                - bbox: List[int] [x1, y1, x2, y2]
                - class_name: str
            ground_plane_estimator: Object with method to convert image coords
                to ground plane coordinates. Expected interface:
                - image_to_ground(x, y) -> Optional[Tuple[float, float]]

        Returns:
            List of CollisionRisk events for this frame.
        """
        timestamp = frame_number / self.fps

        # Convert detections to ObjectStates with ground positions and velocities
        object_states = self._build_object_states(
            detections, ground_plane_estimator, timestamp
        )

        # Separate pedestrians and vehicles
        pedestrians = [
            obj
            for obj in object_states
            if obj.class_name.lower() in self.PEDESTRIAN_CLASSES
        ]
        vehicles = [
            obj
            for obj in object_states
            if obj.class_name.lower() in self.VEHICLE_CLASSES
        ]

        # Check all pedestrian-vehicle pairs for collision risk
        collision_risks: List[CollisionRisk] = []
        for pedestrian in pedestrians:
            for vehicle in vehicles:
                risk = self.collision_predictor.predict(
                    pedestrian, vehicle, frame_number
                )
                if risk is not None:
                    collision_risks.append(risk)

        return collision_risks

    def _build_object_states(
        self,
        detections: List[Dict[str, Any]],
        ground_plane_estimator: Any,
        timestamp: float,
    ) -> List[ObjectState]:
        """Convert detections to ObjectStates with ground positions and velocities.

        Args:
            detections: List of detection dictionaries.
            ground_plane_estimator: Ground plane coordinate converter.
            timestamp: Current timestamp in seconds.

        Returns:
            List of ObjectState instances.
        """
        object_states: List[ObjectState] = []

        for det in detections:
            track_id = det.get("track_id")
            bbox = det.get("bbox", [])
            class_name = det.get("class_name", "unknown")

            if track_id is None or len(bbox) < 4:
                continue

            # Calculate bbox center
            x1, y1, x2, y2 = bbox[:4]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            position_image = (center_x, center_y)

            # Convert to ground plane coordinates
            position_ground = self._get_ground_position(
                ground_plane_estimator, center_x, y2  # Use bottom of bbox for ground
            )

            # Update trajectory and get velocity
            velocity: Optional[Tuple[float, float]] = None
            if position_ground is not None:
                velocity = self.trajectory_tracker.update(
                    track_id, position_ground, timestamp
                )

            object_states.append(
                ObjectState(
                    track_id=track_id,
                    position_image=position_image,
                    position_ground=position_ground,
                    velocity=velocity,
                    class_name=class_name,
                    bbox=list(bbox[:4]),
                )
            )

        return object_states

    def _get_ground_position(
        self,
        ground_plane_estimator: Any,
        x: float,
        y: float,
    ) -> Optional[Tuple[float, float]]:
        """Get ground plane position from image coordinates.

        Args:
            ground_plane_estimator: Ground plane coordinate converter.
            x: Image x coordinate.
            y: Image y coordinate.

        Returns:
            Ground plane position (x, y) in meters, or None if unavailable.
        """
        if ground_plane_estimator is None:
            return None

        try:
            # Try common interface patterns
            if hasattr(ground_plane_estimator, "project_to_ground"):
                result = ground_plane_estimator.project_to_ground((x, y))
                if result is not None and len(result) >= 2:
                    return (float(result[0]), float(result[1]))
            elif hasattr(ground_plane_estimator, "image_to_ground"):
                result = ground_plane_estimator.image_to_ground(x, y)
                if result is not None and len(result) >= 2:
                    return (float(result[0]), float(result[1]))
            elif hasattr(ground_plane_estimator, "transform"):
                result = ground_plane_estimator.transform(x, y)
                if result is not None and len(result) >= 2:
                    return (float(result[0]), float(result[1]))
        except (TypeError, ValueError, AttributeError):
            pass

        return None

    def get_risks_by_level(
        self, risks: List[CollisionRisk], level: str
    ) -> List[CollisionRisk]:
        """Filter collision risks by risk level.

        Args:
            risks: List of CollisionRisk events.
            level: Risk level to filter by ("critical", "warning", or "safe").

        Returns:
            Filtered list of CollisionRisk events.
        """
        return [r for r in risks if r.risk_level == level]

    def reset(self) -> None:
        """Reset all tracking state."""
        self.trajectory_tracker.clear_all()
