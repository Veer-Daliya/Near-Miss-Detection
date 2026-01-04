"""Video reader for various input sources."""

from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


class VideoReader:
    """Reads video from files, RTSP streams, or webcams."""

    def __init__(self, source: str, fps: Optional[int] = None) -> None:
        """
        Initialize video reader.

        Args:
            source: Video file path, RTSP URL, or webcam device ID
            fps: Target FPS for frame sampling (None = use source FPS)
        """
        self.source = source
        self.target_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.source_fps: Optional[float] = None

    def __enter__(self) -> "VideoReader":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def open(self) -> None:
        """Open video source."""
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.source}")

        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.source_fps <= 0:
            self.source_fps = 30.0  # Default fallback

    def close(self) -> None:
        """Close video source."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int, float]]:
        """Iterate over frames."""
        if self.cap is None:
            self.open()

        frame_skip = 1
        if self.target_fps and self.source_fps:
            frame_skip = max(1, int(self.source_fps / self.target_fps))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Skip frames if target FPS is lower than source FPS
            if self.frame_count % frame_skip == 0:
                timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield frame, self.frame_count, timestamp

            self.frame_count += 1

    def get_frame_count(self) -> int:
        """Get total number of frames."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self) -> float:
        """Get source FPS."""
        return self.source_fps or 30.0

    def get_width(self) -> int:
        """Get video width."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self) -> int:
        """Get video height."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

