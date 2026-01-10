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
        confidence_threshold: float = 0.3,  # Lowered from 0.5 for better detection
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

        # Initialize PaddleOCR with GPU support
        # Set PaddlePaddle device before initializing PaddleOCR
        gpu_available = False
        try:
            import paddle
            import os
            
            # Try to set GPU device if available
            if paddle.device.is_compiled_with_cuda():
                try:
                    # Set PaddlePaddle to use GPU
                    paddle.device.set_device("gpu")
                    gpu_available = True
                    # Also set environment variable for PaddleOCR
                    os.environ["USE_GPU"] = "1"
                except Exception:
                    # GPU not available or error setting device
                    paddle.device.set_device("cpu")
                    gpu_available = False
            else:
                # Check if we can use CPU with optimizations
                paddle.device.set_device("cpu")
                gpu_available = False
        except (ImportError, AttributeError):
            # PaddlePaddle not available or error
            gpu_available = False
        
        # Initialize PaddleOCR
        # PaddleOCR will use GPU if PaddlePaddle device is set to GPU
        # Try with use_angle_cls, fallback without it for newer versions
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en")
        except (TypeError, ValueError):
            # Newer PaddleOCR versions may not support use_angle_cls
            self.ocr = PaddleOCR(lang="en")
        
        if gpu_available:
            print("Plate detector initialized with GPU acceleration")
        else:
            print("Plate detector initialized (CPU mode)")
            print("  Note: For GPU support, install: pip install paddlepaddle-gpu")
            print("  (Note: GPU support requires CUDA, not available on macOS)")

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
            # cls parameter not supported in this PaddleOCR version
            results = self.ocr.ocr(vehicle_roi)

            if not results:
                return []

            # Handle new PaddleOCR API (OCRResult object) vs old API (list)
            ocr_result = results[0] if isinstance(results, list) else results
            
            # Check if it's the new OCRResult format
            if hasattr(ocr_result, 'text_lines') or hasattr(ocr_result, 'rec_res'):
                # New API format - extract from OCRResult object
                if hasattr(ocr_result, 'text_lines') and ocr_result.text_lines:
                    text_lines = ocr_result.text_lines
                elif hasattr(ocr_result, 'rec_res') and ocr_result.rec_res:
                    text_lines = ocr_result.rec_res
                else:
                    return []
            elif isinstance(ocr_result, list):
                # Old API format - list of results
                text_lines = ocr_result
            else:
                return []

            vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = vehicle_bbox

            for line in text_lines:
                if not line:
                    continue
                
                # Handle different line formats
                if isinstance(line, tuple) and len(line) == 2:
                    bbox_points, (_, confidence) = line
                elif isinstance(line, dict):
                    # New format might be dict
                    bbox_points = line.get('bbox', [])
                    confidence = line.get('confidence', 0.0)
                else:
                    continue

                # Filter by confidence
                if confidence < self.confidence_threshold:
                    continue

                # Convert bbox points to [x1, y1, x2, y2] in ROI coordinates
                if isinstance(bbox_points, list) and len(bbox_points) > 0:
                    roi_x1, roi_y1, roi_x2, roi_y2 = _bbox_points_to_rect(bbox_points)
                else:
                    continue

                # Calculate aspect ratio and size
                width = roi_x2 - roi_x1
                height = roi_y2 - roi_y1
                aspect_ratio = width / height if height > 0 else 0
                area = width * height

                # Filter for license plate-like regions
                # License plates typically have:
                # - Aspect ratio between 1.5:1 and 6:1 (wider than tall, more lenient)
                # - Minimum size (at least 30x15 pixels - more lenient for distant vehicles)
                # - Minimum area (lowered for smaller plates)
                if (
                    1.5 <= aspect_ratio <= 6.0  # More lenient aspect ratio
                    and width >= 30  # Lowered from 50
                    and height >= 15  # Lowered from 20
                    and area >= 450  # Lowered from 1000 (30*15 = 450)
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

        except Exception as e:
            # Log error for debugging (but don't fail completely)
            # This can happen with very small or corrupted ROIs
            import sys
            print(f"Warning: Plate detection error on vehicle ROI: {e}", file=sys.stderr)
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

