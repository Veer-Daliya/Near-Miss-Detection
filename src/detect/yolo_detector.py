"""YOLOv8 detector for pedestrians and vehicles."""

from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from src.detect.detection_types import Detection


class YOLODetector:
    """YOLOv8-based object detector for pedestrians and vehicles."""

    # COCO class IDs for objects we care about
    COCO_CLASSES = {
        0: "person",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_size: str = "m",
        confidence_threshold: float = 0.4,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize YOLO detector.

        Args:
            model_size: Model size - 'n', 's', 'm', 'l', 'x'
            confidence_threshold: Minimum confidence for detections
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.confidence_threshold = confidence_threshold
        model_name = f"yolov8{model_size}.pt"
        self.model = YOLO(model_name)
        self.device = device

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input image frame (BGR format)
            frame_id: Frame identifier

        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, device=self.device)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class info
                class_id = int(box.cls[0])
                if class_id not in self.COCO_CLASSES:
                    continue  # Skip classes we don't care about

                class_name = self.COCO_CLASSES[class_id]
                confidence = float(box.conf[0])

                # Get bounding box (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = [int(x1), int(y1), int(x2), int(y2)]

                detection = Detection(
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    frame_id=frame_id,
                )
                detections.append(detection)

        return detections

