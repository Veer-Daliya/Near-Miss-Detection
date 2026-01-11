"""Ground plane estimation with multiple fallback methods.

This module provides robust ground plane estimation using three fallback methods:
1. Lane-based: Uses detected lane lines to compute vanishing point and homography
2. Horizon-based: Detects horizon line and estimates ground plane from it
3. Size-based: Uses known pedestrian height to estimate distance (always works)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .line_detector import Line, LineDetector


class EstimationMethod(Enum):
    """Enumeration of ground plane estimation methods."""

    LANE_BASED = "lane_based"
    HORIZON_BASED = "horizon_based"
    SIZE_BASED = "size_based"
    NONE = "none"


@dataclass
class GroundPlaneEstimate:
    """Represents a ground plane estimate with metadata."""

    method: EstimationMethod
    confidence: float  # 0.0 to 1.0
    homography: Optional[np.ndarray]  # 3x3 homography matrix (if available)
    horizon_y: Optional[float]  # Y-coordinate of horizon line (if detected)
    vanishing_point: Optional[Tuple[float, float]]  # (x, y) vanishing point
    frame_number: int  # Frame when this estimate was computed


class GroundPlaneEstimator:
    """Robust ground plane estimator with multiple fallback methods.

    Tries estimation methods in order of accuracy:
    1. Lane-based (best): Uses lane lines to compute vanishing point and homography
    2. Horizon-based (good): Detects horizon line from the image
    3. Size-based (fallback): Uses pedestrian height to estimate distance

    Features:
    - Temporal smoothing using exponential moving average
    - Result caching (updates every N frames)
    - Confidence scoring for each estimate
    """

    # Default camera intrinsics for a typical dashcam (can be overridden)
    DEFAULT_FOCAL_LENGTH = 800.0  # pixels
    DEFAULT_CAMERA_HEIGHT = 1.2  # meters (typical dashcam height)

    # Reference pedestrian dimensions
    PEDESTRIAN_HEIGHT_METERS = 1.7  # Average adult height in meters
    PEDESTRIAN_REFERENCE_DISTANCE = 10.0  # Reference distance in meters
    PEDESTRIAN_REFERENCE_HEIGHT_PIXELS = 170.0  # Expected height at reference distance

    def __init__(
        self,
        focal_length: float = DEFAULT_FOCAL_LENGTH,
        camera_height: float = DEFAULT_CAMERA_HEIGHT,
        smoothing_alpha: float = 0.3,
        cache_frames: int = 30,
        min_lane_lines: int = 2,
        horizon_detection_threshold: float = 0.5,
    ) -> None:
        """Initialize the ground plane estimator.

        Args:
            focal_length: Camera focal length in pixels
            camera_height: Camera height above ground in meters
            smoothing_alpha: EMA smoothing factor (0-1, lower = more smoothing)
            cache_frames: Number of frames to cache results before recomputing
            min_lane_lines: Minimum lane lines required for lane-based method
            horizon_detection_threshold: Confidence threshold for horizon detection
        """
        self.focal_length = focal_length
        self.camera_height = camera_height
        self.smoothing_alpha = smoothing_alpha
        self.cache_frames = cache_frames
        self.min_lane_lines = min_lane_lines
        self.horizon_detection_threshold = horizon_detection_threshold

        # Initialize line detector for lane detection
        self.line_detector = LineDetector(
            canny_low=50,
            canny_high=150,
            hough_threshold=50,
            min_line_length=50,
            max_line_gap=10,
            temporal_smoothing=True,
            smoothing_alpha=0.7,
        )

        # State for caching and smoothing
        self._frame_count: int = 0
        self._cached_estimate: Optional[GroundPlaneEstimate] = None
        self._smoothed_horizon_y: Optional[float] = None
        self._smoothed_vanishing_point: Optional[Tuple[float, float]] = None
        self._smoothed_homography: Optional[np.ndarray] = None

        # Image dimensions (set on first frame)
        self._image_width: int = 0
        self._image_height: int = 0

    def estimate(
        self, image: np.ndarray, force_update: bool = False
    ) -> GroundPlaneEstimate:
        """Estimate the ground plane from an image.

        Tries methods in order: lane-based -> horizon-based -> size-based

        Args:
            image: Input image (BGR format)
            force_update: If True, bypass cache and recompute

        Returns:
            GroundPlaneEstimate with the best available estimation
        """
        self._frame_count += 1
        self._image_height, self._image_width = image.shape[:2]

        # Check cache
        if (
            not force_update
            and self._cached_estimate is not None
            and (self._frame_count - self._cached_estimate.frame_number)
            < self.cache_frames
        ):
            return self._cached_estimate

        # Try methods in order of preference
        estimate = self._try_lane_based(image)

        if estimate is None or estimate.confidence < 0.5:
            horizon_estimate = self._try_horizon_based(image)
            if horizon_estimate is not None and (
                estimate is None or horizon_estimate.confidence > estimate.confidence
            ):
                estimate = horizon_estimate

        if estimate is None or estimate.confidence < 0.3:
            size_estimate = self._try_size_based()
            if size_estimate is not None and (
                estimate is None or size_estimate.confidence > estimate.confidence
            ):
                estimate = size_estimate

        # Fallback to size-based if nothing else worked
        if estimate is None:
            estimate = GroundPlaneEstimate(
                method=EstimationMethod.SIZE_BASED,
                confidence=0.2,
                homography=None,
                horizon_y=self._image_height * 0.4,  # Default horizon at 40% from top
                vanishing_point=None,
                frame_number=self._frame_count,
            )

        # Apply temporal smoothing
        estimate = self._apply_temporal_smoothing(estimate)

        # Cache the result
        self._cached_estimate = estimate

        return estimate

    def _try_lane_based(self, image: np.ndarray) -> Optional[GroundPlaneEstimate]:
        """Try lane-based ground plane estimation.

        Uses detected lane lines to compute vanishing point and derive homography.

        Args:
            image: Input image

        Returns:
            GroundPlaneEstimate if successful, None otherwise
        """
        # Detect all lines
        all_lines = self.line_detector.detect_lines(image)

        if len(all_lines) < self.min_lane_lines:
            return None

        # Filter to get lane lines (lines that converge toward horizon)
        # Lane lines typically have angles between 20-80 degrees or 100-160 degrees
        lane_lines = self._filter_lane_lines(all_lines)

        if len(lane_lines) < self.min_lane_lines:
            return None

        # Compute vanishing point from lane lines
        vanishing_point = self._compute_vanishing_point(lane_lines)

        if vanishing_point is None:
            return None

        # Validate vanishing point (should be near top half of image)
        vp_x, vp_y = vanishing_point
        if vp_y > self._image_height * 0.6 or vp_y < -self._image_height:
            return None

        # Compute homography from vanishing point
        homography = self._compute_homography_from_vanishing_point(vanishing_point)

        # Compute confidence based on number of lines and their consistency
        confidence = self._compute_lane_confidence(lane_lines, vanishing_point)

        return GroundPlaneEstimate(
            method=EstimationMethod.LANE_BASED,
            confidence=confidence,
            homography=homography,
            horizon_y=vp_y,
            vanishing_point=vanishing_point,
            frame_number=self._frame_count,
        )

    def _filter_lane_lines(self, lines: List[Line]) -> List[Line]:
        """Filter lines to keep only those that could be lane lines.

        Lane lines typically have angles that would converge toward the horizon.

        Args:
            lines: All detected lines

        Returns:
            Filtered list of potential lane lines
        """
        lane_lines = []
        for line in lines:
            angle = line.angle
            # Lane lines typically have angles 20-80 or 100-160 degrees
            # (converging toward horizon)
            if (20 <= angle <= 80) or (100 <= angle <= 160):
                # Also filter by position (should be in lower 2/3 of image)
                mid_y = (line.y1 + line.y2) / 2
                if mid_y > self._image_height * 0.33:
                    lane_lines.append(line)
        return lane_lines

    def _compute_vanishing_point(
        self, lines: List[Line]
    ) -> Optional[Tuple[float, float]]:
        """Compute vanishing point from a set of lines.

        Uses least squares intersection of all line pairs.

        Args:
            lines: List of lane lines

        Returns:
            (x, y) vanishing point or None if computation fails
        """
        if len(lines) < 2:
            return None

        # Convert lines to homogeneous representation (ax + by + c = 0)
        homo_lines = []
        for line in lines:
            x1, y1, x2, y2 = line.x1, line.y1, line.x2, line.y2
            # Line equation: (y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x1 = 0
            a = y2 - y1
            b = -(x2 - x1)
            c = (x2 - x1) * y1 - (y2 - y1) * x1
            norm = np.sqrt(a * a + b * b)
            if norm > 0:
                homo_lines.append((a / norm, b / norm, c / norm))

        if len(homo_lines) < 2:
            return None

        # Compute intersections of all line pairs and average
        intersections = []
        for i in range(len(homo_lines)):
            for j in range(i + 1, len(homo_lines)):
                a1, b1, c1 = homo_lines[i]
                a2, b2, c2 = homo_lines[j]

                # Solve for intersection
                det = a1 * b2 - a2 * b1
                if abs(det) < 1e-10:
                    continue  # Lines are parallel

                x = (b1 * c2 - b2 * c1) / det
                y = (a2 * c1 - a1 * c2) / det

                # Only keep intersections above the bottom of the image
                if y < self._image_height:
                    intersections.append((x, y))

        if not intersections:
            return None

        # Use median to be robust to outliers
        x_coords = [p[0] for p in intersections]
        y_coords = [p[1] for p in intersections]

        return (float(np.median(x_coords)), float(np.median(y_coords)))

    def _compute_homography_from_vanishing_point(
        self, vanishing_point: Tuple[float, float]
    ) -> np.ndarray:
        """Compute a ground plane homography from the vanishing point.

        Creates a homography that maps image coordinates to ground plane coordinates.

        Args:
            vanishing_point: (x, y) vanishing point

        Returns:
            3x3 homography matrix
        """
        vp_x, vp_y = vanishing_point
        _cx = self._image_width / 2  # noqa: F841
        _cy = self._image_height / 2  # noqa: F841

        # Create a simple homography based on vanishing point
        # This maps the image to a bird's-eye view perspective
        # Source points: corners of lower portion of image
        src_pts = np.array(
            [
                [0, self._image_height],
                [self._image_width, self._image_height],
                [self._image_width, vp_y + (self._image_height - vp_y) * 0.3],
                [0, vp_y + (self._image_height - vp_y) * 0.3],
            ],
            dtype=np.float32,
        )

        # Destination points: rectangle (bird's eye view)
        width = 400
        height = 600
        dst_pts = np.array(
            [
                [0, height],
                [width, height],
                [width, 0],
                [0, 0],
            ],
            dtype=np.float32,
        )

        homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return homography  # type: ignore[return-value]

    def _compute_lane_confidence(
        self, lines: List[Line], vanishing_point: Tuple[float, float]
    ) -> float:
        """Compute confidence score for lane-based estimation.

        Args:
            lines: Detected lane lines
            vanishing_point: Computed vanishing point

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from number of lines
        line_count_score = min(len(lines) / 10.0, 1.0)

        # Check how well lines converge to vanishing point
        vp_x, vp_y = vanishing_point
        convergence_errors = []

        for line in lines:
            # Extend line to vanishing point y-coordinate
            dx = line.x2 - line.x1
            dy = line.y2 - line.y1
            if abs(dy) < 1e-10:
                continue

            # Find x at vanishing point y
            t = (vp_y - line.y1) / dy
            x_at_vp = line.x1 + t * dx

            error = abs(x_at_vp - vp_x)
            convergence_errors.append(error)

        if convergence_errors:
            avg_error = float(np.mean(convergence_errors))
            convergence_score = max(0.0, 1.0 - avg_error / self._image_width)
        else:
            convergence_score = 0.5

        # Total line length as quality indicator
        total_length = sum(line.length for line in lines)
        length_score = float(min(total_length / (self._image_width * 2), 1.0))

        # Weighted combination
        confidence = (
            0.3 * line_count_score + 0.5 * convergence_score + 0.2 * length_score
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def _try_horizon_based(self, image: np.ndarray) -> Optional[GroundPlaneEstimate]:
        """Try horizon-based ground plane estimation.

        Detects the horizon line from edge patterns and gradients.

        Args:
            image: Input image

        Returns:
            GroundPlaneEstimate if successful, None otherwise
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detect horizontal edges (horizon is often a strong horizontal edge)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = np.abs(sobel_y)

        # Look for the horizon in the upper 60% of the image
        search_region = sobel_y[: int(self._image_height * 0.6), :]

        # Sum along rows to find strongest horizontal edge
        row_sums = np.sum(search_region, axis=1)

        # Apply smoothing to row sums
        kernel_size = max(5, self._image_height // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = cv2.GaussianBlur(
            row_sums.reshape(-1, 1), (1, kernel_size), 0
        ).flatten()

        # Find the row with maximum edge response
        if len(smoothed) == 0:
            return None

        horizon_y_local = int(np.argmax(smoothed))
        horizon_y = float(horizon_y_local)

        # Compute confidence based on edge strength and position
        max_response = smoothed[horizon_y_local]
        mean_response = np.mean(smoothed)

        if mean_response < 1e-10:
            return None

        edge_strength_ratio = max_response / mean_response
        position_score = 1 - abs(horizon_y / self._image_height - 0.35) / 0.35

        confidence = min(0.5 * (edge_strength_ratio / 5.0) + 0.5 * position_score, 0.85)

        if confidence < self.horizon_detection_threshold:
            return None

        # Create homography from horizon
        homography = self._compute_homography_from_horizon(horizon_y)

        return GroundPlaneEstimate(
            method=EstimationMethod.HORIZON_BASED,
            confidence=confidence,
            homography=homography,
            horizon_y=horizon_y,
            vanishing_point=(self._image_width / 2, horizon_y),
            frame_number=self._frame_count,
        )

    def _compute_homography_from_horizon(self, horizon_y: float) -> np.ndarray:
        """Compute homography from detected horizon line.

        Args:
            horizon_y: Y-coordinate of the horizon line

        Returns:
            3x3 homography matrix
        """
        # Similar to vanishing point method, but centered
        src_pts = np.array(
            [
                [0, self._image_height],
                [self._image_width, self._image_height],
                [self._image_width, horizon_y + (self._image_height - horizon_y) * 0.2],
                [0, horizon_y + (self._image_height - horizon_y) * 0.2],
            ],
            dtype=np.float32,
        )

        width = 400
        height = 600
        dst_pts = np.array(
            [
                [0, height],
                [width, height],
                [width, 0],
                [0, 0],
            ],
            dtype=np.float32,
        )

        homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return homography  # type: ignore[return-value]

    def _try_size_based(self) -> Optional[GroundPlaneEstimate]:
        """Create a size-based ground plane estimate.

        This method always works as a fallback, using assumed camera parameters.

        Returns:
            GroundPlaneEstimate with moderate confidence
        """
        # Default horizon at 40% from top of image
        default_horizon_y = self._image_height * 0.4

        # Create default homography
        homography = self._compute_homography_from_horizon(default_horizon_y)

        return GroundPlaneEstimate(
            method=EstimationMethod.SIZE_BASED,
            confidence=0.4,  # Moderate confidence for fallback method
            homography=homography,
            horizon_y=default_horizon_y,
            vanishing_point=(self._image_width / 2, default_horizon_y),
            frame_number=self._frame_count,
        )

    def _apply_temporal_smoothing(
        self, estimate: GroundPlaneEstimate
    ) -> GroundPlaneEstimate:
        """Apply temporal smoothing to the estimate.

        Uses exponential moving average to smooth values across frames.

        Args:
            estimate: Current frame's estimate

        Returns:
            Smoothed estimate
        """
        alpha = self.smoothing_alpha

        # Smooth horizon_y
        if estimate.horizon_y is not None:
            if self._smoothed_horizon_y is None:
                self._smoothed_horizon_y = estimate.horizon_y
            else:
                self._smoothed_horizon_y = (
                    alpha * estimate.horizon_y + (1 - alpha) * self._smoothed_horizon_y
                )

        # Smooth vanishing point
        if estimate.vanishing_point is not None:
            if self._smoothed_vanishing_point is None:
                self._smoothed_vanishing_point = estimate.vanishing_point
            else:
                vp_x = (
                    alpha * estimate.vanishing_point[0]
                    + (1 - alpha) * self._smoothed_vanishing_point[0]
                )
                vp_y = (
                    alpha * estimate.vanishing_point[1]
                    + (1 - alpha) * self._smoothed_vanishing_point[1]
                )
                self._smoothed_vanishing_point = (vp_x, vp_y)

        # Create smoothed estimate
        smoothed_estimate = GroundPlaneEstimate(
            method=estimate.method,
            confidence=estimate.confidence,
            homography=estimate.homography,
            horizon_y=self._smoothed_horizon_y,
            vanishing_point=self._smoothed_vanishing_point,
            frame_number=estimate.frame_number,
        )

        # Recompute homography with smoothed values if we have them
        if self._smoothed_vanishing_point is not None:
            smoothed_estimate = GroundPlaneEstimate(
                method=estimate.method,
                confidence=estimate.confidence,
                homography=self._compute_homography_from_vanishing_point(
                    self._smoothed_vanishing_point
                ),
                horizon_y=self._smoothed_horizon_y,
                vanishing_point=self._smoothed_vanishing_point,
                frame_number=estimate.frame_number,
            )

        return smoothed_estimate

    def project_to_ground(
        self, image_point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Project an image point to ground coordinates in meters.

        Args:
            image_point: (x, y) pixel coordinates in the image

        Returns:
            (x, z) ground coordinates in meters, where x is lateral position
            and z is distance from camera. Returns None if projection fails.
        """
        if self._cached_estimate is None:
            return None

        estimate = self._cached_estimate
        x_img, y_img = image_point

        # Method 1: Use homography if available (works for any camera angle)
        if estimate.homography is not None:
            try:
                # Apply homography
                pt = np.array([[x_img, y_img]], dtype=np.float32).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(pt, estimate.homography)
                tx, ty = transformed[0, 0]

                # Convert to meters (scale based on camera height)
                # The homography maps to a 400x600 pixel bird's eye view
                # We assume this corresponds to roughly 10m x 20m on the ground
                x_meters = (tx - 200) * 10.0 / 400.0  # Center and scale
                z_meters = (600 - ty) * 20.0 / 600.0  # Flip and scale

                return (float(x_meters), float(z_meters))
            except cv2.error:
                pass

        # Method 2: Size-based fallback using vertical position
        # Objects lower in the image are closer to camera
        if self._image_height > 0:
            # Normalize y position (0 = top, 1 = bottom)
            y_normalized = y_img / self._image_height

            # Map to distance: objects at bottom are ~2m, at top are ~50m
            # Using exponential mapping for more realistic distances
            z_meters = 2.0 + (1.0 - y_normalized) * 30.0

            # Lateral position based on x offset from center
            x_offset = x_img - self._image_width / 2
            x_meters = x_offset * z_meters / self.focal_length

            return (float(x_meters), float(z_meters))

        return None

    def estimate_distance(self, bbox: Tuple[int, int, int, int]) -> float:
        """Estimate distance to an object based on its bounding box.

        Uses size-based estimation as the primary method, with optional
        refinement from ground plane projection.

        Args:
            bbox: Bounding box as (x1, y1, x2, y2)

        Returns:
            Estimated distance in meters
        """
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        _bbox_width = x2 - x1  # noqa: F841
        bbox_center_x = (x1 + x2) / 2
        bbox_bottom_y = y2

        # Method 1: Size-based estimation (always available)
        # Distance = (reference_height * reference_distance) / apparent_height
        if bbox_height > 0:
            size_based_distance = (
                self.PEDESTRIAN_HEIGHT_METERS
                * self.PEDESTRIAN_REFERENCE_DISTANCE
                * self.PEDESTRIAN_REFERENCE_HEIGHT_PIXELS
            ) / (bbox_height * self.PEDESTRIAN_REFERENCE_DISTANCE)

            # Simplify: Distance = (focal_length * real_height) / apparent_height
            # Using empirical reference values
            size_based_distance = (
                self.focal_length * self.PEDESTRIAN_HEIGHT_METERS / bbox_height
            )
        else:
            size_based_distance = float("inf")

        # Method 2: Ground plane projection (if available and reliable)
        if self._cached_estimate is not None and self._cached_estimate.confidence > 0.5:
            ground_coords = self.project_to_ground((bbox_center_x, bbox_bottom_y))
            if ground_coords is not None:
                _, z_meters = ground_coords
                if 0 < z_meters < 100:  # Sanity check
                    # Blend size-based and ground-based estimates
                    # Weight by confidence
                    confidence = self._cached_estimate.confidence
                    blended_distance = (
                        confidence * z_meters + (1 - confidence) * size_based_distance
                    )
                    return float(blended_distance)

        return float(size_based_distance)

    def get_current_method(self) -> EstimationMethod:
        """Get the current estimation method being used.

        Returns:
            The EstimationMethod enum value for the current estimate
        """
        if self._cached_estimate is None:
            return EstimationMethod.NONE
        return self._cached_estimate.method

    def get_confidence(self) -> float:
        """Get the confidence of the current estimate.

        Returns:
            Confidence score between 0 and 1
        """
        if self._cached_estimate is None:
            return 0.0
        return self._cached_estimate.confidence

    def get_horizon_y(self) -> Optional[float]:
        """Get the current horizon Y-coordinate.

        Returns:
            Y-coordinate of the horizon line in pixels, or None
        """
        if self._cached_estimate is None:
            return None
        return self._cached_estimate.horizon_y

    def reset(self) -> None:
        """Reset the estimator state.

        Call this when starting a new video or scene.
        """
        self._frame_count = 0
        self._cached_estimate = None
        self._smoothed_horizon_y = None
        self._smoothed_vanishing_point = None
        self._smoothed_homography = None
        self.line_detector.reset_temporal_state()

    def visualize(
        self,
        image: np.ndarray,
        draw_horizon: bool = True,
        draw_vanishing_point: bool = True,
        draw_grid: bool = False,
    ) -> np.ndarray:
        """Visualize the ground plane estimate on an image.

        Args:
            image: Input image
            draw_horizon: Whether to draw the horizon line
            draw_vanishing_point: Whether to draw the vanishing point
            draw_grid: Whether to draw a perspective grid

        Returns:
            Image with visualization overlays
        """
        result = image.copy()

        if self._cached_estimate is None:
            return result

        estimate = self._cached_estimate

        # Draw horizon line
        if draw_horizon and estimate.horizon_y is not None:
            horizon_y = int(estimate.horizon_y)
            # Color based on confidence (green = high, yellow = medium, red = low)
            if estimate.confidence > 0.7:
                color = (0, 255, 0)
            elif estimate.confidence > 0.4:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.line(result, (0, horizon_y), (self._image_width, horizon_y), color, 2)

            # Add label
            label = f"{estimate.method.value} ({estimate.confidence:.2f})"
            cv2.putText(
                result,
                label,
                (10, horizon_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Draw vanishing point
        if draw_vanishing_point and estimate.vanishing_point is not None:
            vp_x, vp_y = estimate.vanishing_point
            if 0 <= vp_x < self._image_width and 0 <= vp_y < self._image_height:
                cv2.circle(result, (int(vp_x), int(vp_y)), 8, (255, 0, 255), -1)
                cv2.circle(result, (int(vp_x), int(vp_y)), 12, (255, 0, 255), 2)

        # Draw perspective grid
        if draw_grid and estimate.horizon_y is not None:
            self._draw_perspective_grid(result, estimate)

        return result

    def _draw_perspective_grid(
        self, image: np.ndarray, estimate: GroundPlaneEstimate
    ) -> None:
        """Draw a perspective grid on the image.

        Args:
            image: Image to draw on (modified in place)
            estimate: Ground plane estimate
        """
        if estimate.vanishing_point is None or estimate.horizon_y is None:
            return

        vp_x, vp_y = estimate.vanishing_point
        horizon_y = estimate.horizon_y

        # Draw converging lines from bottom corners to vanishing point
        grid_color = (128, 128, 128)
        num_lines = 5

        for i in range(num_lines):
            # Left side
            x_left = int(i * self._image_width / num_lines)
            cv2.line(
                image,
                (x_left, self._image_height),
                (int(vp_x), int(vp_y)),
                grid_color,
                1,
            )
            # Right side
            x_right = int((num_lines - i) * self._image_width / num_lines)
            cv2.line(
                image,
                (x_right, self._image_height),
                (int(vp_x), int(vp_y)),
                grid_color,
                1,
            )

        # Draw horizontal lines (perspective corrected)
        num_horizontal = 5
        for i in range(1, num_horizontal + 1):
            # Interpolate y position with perspective
            t = i / (num_horizontal + 1)
            y = int(horizon_y + (self._image_height - horizon_y) * t)

            # Calculate x bounds at this y level
            if abs(vp_y - self._image_height) > 1e-10:
                ratio = (y - vp_y) / (self._image_height - vp_y)
                x_left = int(vp_x - ratio * vp_x)
                x_right = int(vp_x + ratio * (self._image_width - vp_x))
            else:
                x_left = 0
                x_right = self._image_width

            cv2.line(image, (x_left, y), (x_right, y), grid_color, 1)
