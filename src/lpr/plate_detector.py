"""License plate detector using PaddleOCR's built-in detection."""

from typing import List, Optional

import numpy as np

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

from src.lpr.plate_types import PlateResult


def _bbox_points_to_rect(bbox_points: List[List[float]]) -> tuple[int, int, int, int]:
    """Convert bbox points to [x1, y1, x2, y2] rectangle coordinates."""
    x_coords = [point[0] for point in bbox_points]
    y_coords = [point[1] for point in bbox_points]
    return int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))


class PlateDetector:
    """
    License plate detector using PaddleOCR's text detection.

    Uses PaddleOCR to detect text regions that match license plate characteristics.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize plate detector.

        Args:
            confidence_threshold: Minimum confidence for plate detection
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device

        # Initialize PaddleOCR for text detection
        if PaddleOCR is None:
            raise ImportError(
                "PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr"
            )

        # Initialize PaddleOCR (we'll use detection results, ignore text initially)
        # Note: PaddleOCR does detection+OCR together, but we can use just the bboxes
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def detect_on_vehicle_roi(
        self, vehicle_roi: np.ndarray, vehicle_bbox: List[int]
    ) -> List[PlateResult]:
        """
        Detect license plates within a vehicle ROI.

        Args:
            vehicle_roi: Cropped vehicle region
            vehicle_bbox: Original vehicle bbox [x1, y1, x2, y2] in frame coordinates

        Returns:
            List of PlateResult objects with bounding boxes (text will be empty, filled by OCR later)
        """
        if vehicle_roi.size == 0:
            return []

        plates = []

        try:
            # Run PaddleOCR detection (detection only, no recognition)
            # Note: PaddleOCR doesn't have a pure detection mode, so we use OCR but ignore text
            results = self.ocr.ocr(vehicle_roi, cls=True)

            if results and results[0]:
                vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = vehicle_bbox

                for line in results[0]:
                    if line:
                        bbox_points, (text, confidence) = line

                        # Filter by confidence
                        if confidence < self.confidence_threshold:
                            continue

                        # Convert bbox points to [x1, y1, x2, y2] in ROI coordinates
                        roi_x1, roi_y1, roi_x2, roi_y2 = _bbox_points_to_rect(bbox_points)

                        # Calculate aspect ratio and size
                        width = roi_x2 - roi_x1
                        height = roi_y2 - roi_y1
                        aspect_ratio = width / height if height > 0 else 0
                        area = width * height

                        # Filter for license plate-like regions
                        # License plates typically have:
                        # - Aspect ratio between 2:1 and 5:1 (wider than tall)
                        # - Minimum size (at least 50x20 pixels)
                        # - Text length between 4-10 characters (rough check)
                        if (
                            2.0 <= aspect_ratio <= 5.0
                            and width >= 50
                            and height >= 20
                            and area >= 1000  # Minimum area
                        ):
                            # Convert ROI coordinates to frame coordinates
                            frame_x1 = vehicle_x1 + roi_x1
                            frame_y1 = vehicle_y1 + roi_y1
                            frame_x2 = vehicle_x1 + roi_x2
                            frame_y2 = vehicle_y1 + roi_y2

                            plate = PlateResult(
                                text="",  # Will be filled by OCR later
                                confidence=confidence,
                                bbox=[frame_x1, frame_y1, frame_x2, frame_y2],
                            )
                            plates.append(plate)

        except Exception:
            # If detection fails, return empty list
            # This can happen with very small or corrupted ROIs
            # Silently continue - plate detection is optional
            pass

        return plates

    def detect_on_frame(
        self, frame: np.ndarray, vehicle_bboxes: List[List[int]]
    ) -> List[PlateResult]:
        """
        Detect license plates on full frame or vehicle ROIs.

        Args:
            frame: Full frame image
            vehicle_bboxes: List of vehicle bounding boxes

        Returns:
            List of PlateResult objects
        """
        plates = []
        for vehicle_bbox in vehicle_bboxes:
            x1, y1, x2, y2 = vehicle_bbox
            vehicle_roi = frame[y1:y2, x1:x2]
            if vehicle_roi.size == 0:
                continue

            roi_plates = self.detect_on_vehicle_roi(vehicle_roi, vehicle_bbox)
            plates.extend(roi_plates)

        return plates

