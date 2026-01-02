"""Detection module for objects, pedestrians, and vehicles."""

from src.detect.detection_types import Detection
from src.detect.yolo_detector import YOLODetector

__all__ = ["Detection", "YOLODetector"]



