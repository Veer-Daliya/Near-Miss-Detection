"""Road marking detection using Canny edge detection and Hough Transform."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Line:
    """Represents a detected line segment."""

    x1: int
    y1: int
    x2: int
    y2: int
    angle: float  # Angle in degrees (0-180)
    length: float  # Length of the line segment

    def to_array(self) -> np.ndarray:
        """Convert line to numpy array [x1, y1, x2, y2]."""
        return np.array([self.x1, self.y1, self.x2, self.y2])


@dataclass
class LineGroup:
    """Represents a group of parallel lines."""

    lines: List[Line]
    average_angle: float
    average_length: float

    def get_representative_line(self) -> Line:
        """Get the longest line as representative of the group."""
        return max(self.lines, key=lambda line: line.length)


class LineDetector:
    """Detects road markings and lane lines using edge detection and Hough Transform."""

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 50,
        min_line_length: int = 50,
        max_line_gap: int = 10,
        angle_tolerance: float = 5.0,
        temporal_smoothing: bool = False,
        smoothing_alpha: float = 0.7,
    ) -> None:
        """
        Initialize line detector.

        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            hough_threshold: Accumulator threshold for Hough lines
            min_line_length: Minimum line length to detect
            max_line_gap: Maximum gap between line segments to connect
            angle_tolerance: Tolerance in degrees for grouping parallel lines
            temporal_smoothing: If True, apply temporal smoothing across frames
            smoothing_alpha: Smoothing factor (0-1), higher = more weight to new detections
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.angle_tolerance = angle_tolerance
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_alpha = smoothing_alpha
        # Store previous frame's lines for temporal smoothing
        self.prev_lane_lines: List[Line] = []
        self.prev_road_edges: List[Line] = []

    def detect_lines(self, image: np.ndarray) -> List[Line]:
        """
        Detect lines in an image using Canny edge detection and Hough Transform.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            List of detected Line objects
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return []

        # Convert to Line objects
        detected_lines: List[Line] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle (0-180 degrees)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if angle < 0:
                angle += 180

            # Calculate length
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            detected_lines.append(Line(x1, y1, x2, y2, angle, length))

        return detected_lines

    def filter_horizontal_lines(
        self, lines: List[Line], min_angle: float = 80.0, max_angle: float = 100.0
    ) -> List[Line]:
        """
        Filter lines that are approximately horizontal (lane markings).

        Args:
            lines: List of detected lines
            min_angle: Minimum angle in degrees (default: 80, near horizontal)
            max_angle: Maximum angle in degrees (default: 100, near horizontal)

        Returns:
            Filtered list of horizontal lines
        """
        filtered = []
        for line in lines:
            # Normalize angle to 0-180 range
            angle = line.angle
            if angle > 90:
                angle = 180 - angle

            # Check if line is approximately horizontal
            if min_angle <= line.angle <= max_angle:
                filtered.append(line)

        return filtered

    def filter_vertical_lines(
        self, lines: List[Line], min_angle: float = 0.0, max_angle: float = 10.0
    ) -> List[Line]:
        """
        Filter lines that are approximately vertical (road edges).

        Args:
            lines: List of detected lines
            min_angle: Minimum angle in degrees (default: 0, vertical)
            max_angle: Maximum angle in degrees (default: 10, near vertical)

        Returns:
            Filtered list of vertical lines
        """
        filtered = []
        for line in lines:
            # Normalize angle to 0-90 range for vertical lines
            angle = line.angle
            if angle > 90:
                angle = 180 - angle

            # Check if line is approximately vertical
            if min_angle <= angle <= max_angle:
                filtered.append(line)

        return filtered

    def group_parallel_lines(self, lines: List[Line]) -> List[LineGroup]:
        """
        Group lines that are approximately parallel.

        Args:
            lines: List of detected lines

        Returns:
            List of LineGroup objects, each containing parallel lines
        """
        if not lines:
            return []

        groups: List[LineGroup] = []
        used = [False] * len(lines)

        for i, line in enumerate(lines):
            if used[i]:
                continue

            # Start a new group with this line
            group_lines = [line]
            used[i] = True

            # Find all lines parallel to this one
            for j, other_line in enumerate(lines):
                if used[j] or i == j:
                    continue

                # Calculate angle difference
                angle_diff = abs(line.angle - other_line.angle)
                # Handle wrap-around (e.g., 179 and 1 degrees are similar)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff

                if angle_diff <= self.angle_tolerance:
                    group_lines.append(other_line)
                    used[j] = True

            # Calculate average angle and length for the group
            avg_angle = float(np.mean([line.angle for line in group_lines]))
            avg_length = float(np.mean([line.length for line in group_lines]))

            groups.append(LineGroup(group_lines, avg_angle, avg_length))

        return groups

    def detect_lane_markings(self, image: np.ndarray) -> List[Line]:
        """
        Detect lane markings (horizontal lines) in the image.

        Args:
            image: Input image

        Returns:
            List of detected lane marking lines
        """
        all_lines = self.detect_lines(image)
        horizontal_lines = self.filter_horizontal_lines(all_lines)

        if self.temporal_smoothing and self.prev_lane_lines:
            horizontal_lines = self._smooth_lines(
                horizontal_lines, self.prev_lane_lines
            )

        if self.temporal_smoothing:
            self.prev_lane_lines = horizontal_lines

        return horizontal_lines

    def detect_road_edges(self, image: np.ndarray) -> List[Line]:
        """
        Detect road edges (vertical lines) in the image.

        Args:
            image: Input image

        Returns:
            List of detected road edge lines
        """
        all_lines = self.detect_lines(image)
        vertical_lines = self.filter_vertical_lines(all_lines)

        if self.temporal_smoothing and self.prev_road_edges:
            vertical_lines = self._smooth_lines(vertical_lines, self.prev_road_edges)

        if self.temporal_smoothing:
            self.prev_road_edges = vertical_lines

        return vertical_lines

    def _smooth_lines(
        self, current_lines: List[Line], previous_lines: List[Line]
    ) -> List[Line]:
        """
        Smooth lines temporally by matching and averaging with previous frame's lines.

        Args:
            current_lines: Lines detected in current frame
            previous_lines: Lines from previous frame

        Returns:
            Smoothed lines
        """
        if not previous_lines:
            return current_lines

        if not current_lines:
            return previous_lines

        # Match current lines to previous lines based on position and angle
        matched_lines: List[Line] = []
        used_prev = [False] * len(previous_lines)

        for curr_line in current_lines:
            best_match_idx = -1
            best_distance = float("inf")

            # Find closest matching previous line
            for i, prev_line in enumerate(previous_lines):
                if used_prev[i]:
                    continue

                # Calculate distance between line midpoints
                curr_mid_x = (curr_line.x1 + curr_line.x2) / 2
                curr_mid_y = (curr_line.y1 + curr_line.y2) / 2
                prev_mid_x = (prev_line.x1 + prev_line.x2) / 2
                prev_mid_y = (prev_line.y1 + prev_line.y2) / 2

                distance = np.sqrt(
                    (curr_mid_x - prev_mid_x) ** 2 + (curr_mid_y - prev_mid_y) ** 2
                )

                # Also check angle similarity
                angle_diff = abs(curr_line.angle - prev_line.angle)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff

                # Combined distance metric
                combined_distance = distance + angle_diff * 5  # Weight angle difference

                if (
                    combined_distance < best_distance and combined_distance < 100
                ):  # Threshold
                    best_distance = combined_distance
                    best_match_idx = i

            if best_match_idx >= 0:
                # Smooth with previous line
                prev_line = previous_lines[best_match_idx]
                used_prev[best_match_idx] = True

                # Exponential moving average
                alpha = self.smoothing_alpha
                smoothed_x1 = int(curr_line.x1 * alpha + prev_line.x1 * (1 - alpha))
                smoothed_y1 = int(curr_line.y1 * alpha + prev_line.y1 * (1 - alpha))
                smoothed_x2 = int(curr_line.x2 * alpha + prev_line.x2 * (1 - alpha))
                smoothed_y2 = int(curr_line.y2 * alpha + prev_line.y2 * (1 - alpha))

                # Recalculate angle and length
                smoothed_angle = np.degrees(
                    np.arctan2(smoothed_y2 - smoothed_y1, smoothed_x2 - smoothed_x1)
                )
                if smoothed_angle < 0:
                    smoothed_angle += 180
                smoothed_length = np.sqrt(
                    (smoothed_x2 - smoothed_x1) ** 2 + (smoothed_y2 - smoothed_y1) ** 2
                )

                matched_lines.append(
                    Line(
                        smoothed_x1,
                        smoothed_y1,
                        smoothed_x2,
                        smoothed_y2,
                        smoothed_angle,
                        smoothed_length,
                    )
                )
            else:
                # No match found, use current line (new detection)
                matched_lines.append(curr_line)

        # Add unmatched previous lines (with decay)
        for i, prev_line in enumerate(previous_lines):
            if not used_prev[i]:
                # Keep previous line but fade it out
                matched_lines.append(prev_line)

        return matched_lines

    def reset_temporal_state(self) -> None:
        """Reset temporal smoothing state (call when starting a new video)."""
        self.prev_lane_lines = []
        self.prev_road_edges = []

    def visualize_lines(
        self,
        image: np.ndarray,
        lines: List[Line],
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw detected lines on an image.

        Args:
            image: Input image
            lines: List of lines to draw
            color: BGR color tuple for lines

        Returns:
            Image with lines drawn
        """
        result = image.copy()
        for line in lines:
            cv2.line(result, (line.x1, line.y1), (line.x2, line.y2), color, 2)
        return result

    def visualize_line_groups(
        self, image: np.ndarray, groups: List[LineGroup]
    ) -> np.ndarray:
        """
        Draw line groups on an image, with each group in a different color.

        Args:
            image: Input image
            groups: List of line groups to draw

        Returns:
            Image with line groups drawn
        """
        result = image.copy()

        # Generate distinct colors for each group
        colors = [
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        for idx, group in enumerate(groups):
            color = colors[idx % len(colors)]
            for line in group.lines:
                cv2.line(result, (line.x1, line.y1), (line.x2, line.y2), color, 2)

        return result
