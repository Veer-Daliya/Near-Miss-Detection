"""YOLOv10 detector for pedestrians and vehicles."""

from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from src.detect.detection_types import Detection
from src.utils.gpu_utils import optimize_gpu_settings


def detect_batch(
    model: YOLO,
    frames: List[np.ndarray],
    frame_ids: List[int],
    confidence_threshold: float,
    device: str,
) -> List[List[Detection]]:
    """
    Detect objects in a batch of frames for better GPU utilization.

    Args:
        model: YOLO model instance
        frames: List of input frames (BGR format)
        frame_ids: List of frame identifiers
        confidence_threshold: Minimum confidence for detections
        device: Device to use ('cuda', 'mps', 'cpu')

    Returns:
        List of detection lists (one per frame)
    """
    if not frames:
        return []

    # Use half precision (FP16) on GPU for faster inference
    half = device in ["cuda", "mps"]

    # Batch inference - YOLO handles batching internally
    results = model(
        frames,
        conf=confidence_threshold,
        device=device,
        half=half,
        verbose=False,
        imgsz=640,
        max_det=300,
    )

    # Process results for each frame
    all_detections = []
    for frame_idx, result in enumerate(results):
        detections = []
        boxes = result.boxes

        for box in boxes:
            # Get class info
            class_id = int(box.cls[0])
            if class_id not in YOLODetector.COCO_CLASSES:
                continue  # Skip classes we don't care about

            class_name = YOLODetector.COCO_CLASSES[class_id]
            confidence = float(box.conf[0])

            # Get bounding box (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox = [int(x1), int(y1), int(x2), int(y2)]

            frame_id = frame_ids[frame_idx] if frame_idx < len(frame_ids) else frame_idx

            detection = Detection(
                bbox=bbox,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                frame_id=frame_id,
            )
            detections.append(detection)

        all_detections.append(detections)

    return all_detections


class YOLODetector:
    """YOLOv10-based object detector for pedestrians and vehicles."""

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
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.confidence_threshold = confidence_threshold

        # Try to find model in data/models/ first, then fall back to Ultralytics auto-download
        model_name = f"yolov10{model_size}.pt"
        project_root = Path(__file__).parent.parent.parent
        local_model_path = project_root / "data" / "models" / model_name

        if local_model_path.exists():
            model_path = str(local_model_path)
            print(f"Using local model: {model_path}")
        else:
            # Fall back to Ultralytics auto-download (will download to default location)
            model_path = model_name
            print(f"Model not found locally, will download: {model_name}")

        # Auto-detect best device if not specified
        if device is None:
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    device = "mps"  # Apple Silicon GPU
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        # Optimize GPU settings
        optimize_gpu_settings(device)

        self.device = device

        # Load model and explicitly set device
        self.model = YOLO(model_path)

        # Move model to device explicitly for better GPU utilization
        if self.device is not None and self.device in ["cuda", "mps"]:
            try:
                import torch

                # Ensure model is on the correct device
                if hasattr(self.model.model, "to"):
                    self.model.model.to(self.device)  # type: ignore
            except Exception:
                pass  # YOLO handles device placement internally

        print(f"YOLO detector using device: {self.device}")

        # Print GPU memory info if available
        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    print(f"GPU: {torch.cuda.get_device_name(0)}")
                    print(
                        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
                    )
            except Exception:
                pass

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input image frame (BGR format)
            frame_id: Frame identifier

        Returns:
            List of Detection objects
        """
        # Run inference with optimizations
        # Use half precision (FP16) on GPU for faster inference
        half = self.device in ["cuda", "mps"]
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            half=half,  # Use FP16 on GPU (2x faster)
            verbose=False,  # Suppress verbose output
            imgsz=640,  # Standard input size for optimal GPU utilization
            max_det=300,  # Limit max detections for faster NMS
        )

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

    def detect_batch(
        self, frames: List[np.ndarray], frame_ids: List[int]
    ) -> List[List[Detection]]:
        """
        Detect objects in a batch of frames for better GPU utilization.

        This method processes multiple frames at once, which is more efficient
        for GPU inference than processing frames one at a time.

        Args:
            frames: List of input frames (BGR format)
            frame_ids: List of frame identifiers

        Returns:
            List of detection lists (one per frame)
        """
        return detect_batch(
            self.model, frames, frame_ids, self.confidence_threshold, self.device
        )
