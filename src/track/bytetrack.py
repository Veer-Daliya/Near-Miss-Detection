"""ByteTrack multi-object tracker wrapper using YOLO's built-in tracking."""

from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from src.detect.detection_types import Detection


class ByteTracker:
    """
    ByteTrack multi-object tracker using YOLO's built-in tracking.
    
    Provides stable track IDs across frames using ByteTrack algorithm.
    """

    def __init__(
        self,
        model: YOLO,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ) -> None:
        """
        Initialize ByteTracker.

        Args:
            model: YOLO model instance (for tracking)
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IOU threshold for matching
        """
        self.model = model
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_count = 0  # Track frame count for persist

    def update(self, detections: List[Detection], frame: np.ndarray, frame_id: int = 0) -> List[Detection]:
        """
        Update tracker and return detections with track IDs.
        
        IMPORTANT: This uses YOLO's track() which does detection AND tracking in one pass.
        This ensures bboxes and track IDs correspond correctly from the same detection.

        Args:
            detections: List of detections (ignored - we re-detect for consistency)
            frame: Current frame
            frame_id: Frame identifier

        Returns:
            List of detections with track IDs assigned (bboxes and IDs from same detection pass)
        """
        self.frame_count += 1
        
        # Use YOLO's built-in tracking (ByteTrack is default)
        # persist=True maintains track IDs across frames
        # This does BOTH detection AND tracking in one pass, ensuring bboxes match track IDs
        # Detect device for optimization
        device = None
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except ImportError:
            pass
        
        half = device in ["cuda", "mps"] if device else False
        
        results = self.model.track(
            frame,
            conf=self.track_thresh,
            persist=True,  # Maintain tracks across frames
            verbose=False,
            max_det=300,  # Limit max detections to speed up NMS
            device=device,  # Explicitly set device for better GPU utilization
            half=half,  # Use FP16 on GPU for faster inference
            imgsz=640,  # Standard input size for optimal GPU utilization
        )

        # Extract tracked detections directly from YOLO tracking results
        # This ensures bboxes and track IDs come from the same detection pass
        tracked_detections = []
        
        # Map class IDs to class names (from YOLODetector)
        COCO_CLASSES = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                # Get track IDs (may be None if tracking not available)
                track_ids = None
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    # Get bbox from tracked results (ensures consistency)
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    
                    # Only process classes we care about
                    if class_id not in COCO_CLASSES:
                        continue
                    
                    # Get track ID (if available)
                    track_id = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None
                    
                    tracked_det = Detection(
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        class_id=class_id,
                        class_name=COCO_CLASSES[class_id],
                        confidence=confidence,
                        frame_id=frame_id,
                        track_id=track_id,
                    )
                    tracked_detections.append(tracked_det)

        return tracked_detections

    def _find_track_by_iou(
        self, detection: Detection, track_id_map: dict
    ) -> Optional[int]:
        """Find track ID by IOU matching."""
        best_iou = 0.3  # Minimum threshold
        best_track_id = None

        for (x1, y1, x2, y2, cls_id), track_id in track_id_map.items():
            if cls_id != detection.class_id:
                continue

            iou = self._calculate_iou(
                np.array([x1, y1, x2, y2]), np.array(detection.bbox)
            )
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id

        return best_track_id

    @staticmethod
    def _calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union (IOU) between two bboxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

