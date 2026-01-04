"""Visualization utilities for drawing detections."""

from typing import List

import cv2
import numpy as np

from src.detect.detection_types import Detection
from src.lpr.plate_types import PlateResult


def draw_detections_with_labels(
    frame: np.ndarray,
    detections: List[Detection],
    plate_results: List[PlateResult],
    show_track_ids: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes with labels (plate text + track ID).

    Args:
        frame: Input frame (BGR format)
        detections: List of detections
        plate_results: List of plate results
        show_track_ids: Whether to show track IDs

    Returns:
        Annotated frame
    """
    # Copy frame for annotation (required since we modify it)
    annotated = frame.copy()

    # Pre-compute plate lookup for faster access (O(1) instead of O(n) per vehicle)
    plate_lookup = {p.vehicle_track_id: p for p in plate_results if p.vehicle_track_id is not None}

    for detection in detections:
        bbox = detection.bbox
        track_id = detection.track_id or 0
        class_name = detection.class_name

        # Find associated plate for vehicles (optimized: O(1) lookup)
        plate_text = None
        plate_confidence = None

        if class_name in ["car", "truck", "bus", "motorcycle"]:
            plate = plate_lookup.get(track_id)
            if plate:
                plate_text = plate.text
                plate_confidence = plate.confidence

        # Build label based on conditional logic
        if class_name in ["car", "truck", "bus", "motorcycle"]:
            # Vehicle: Conditional display
            if plate_text and plate_confidence and plate_confidence > 0.7:
                # High confidence plate detected
                if show_track_ids:
                    label = f"{plate_text} [{track_id}]"
                else:
                    label = plate_text
                label_color = (0, 255, 0)  # Green

            elif plate_text and plate_confidence and plate_confidence > 0.4:
                # Low confidence plate
                if show_track_ids:
                    label = f"{plate_text}? [{track_id}]"
                else:
                    label = f"{plate_text}?"
                label_color = (0, 255, 255)  # Yellow

            else:
                # No plate detected - fallback
                label = f"Vehicle [{track_id}]"
                label_color = (0, 0, 255)  # Red

        else:
            # Non-vehicle (pedestrian, bicycle, etc.)
            if show_track_ids:
                label = f"{class_name.capitalize()} [{track_id}]"
            else:
                label = class_name.capitalize()
            label_color = (255, 255, 255)  # White

        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), label_color, 2)

        # Draw label with background for visibility
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Background rectangle
        cv2.rectangle(
            annotated,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 5, y1),
            (0, 0, 0),  # Black background
            -1,
        )

        # Text
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            label_color,
            2,
        )

    return annotated

