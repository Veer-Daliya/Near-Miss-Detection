#!/usr/bin/env python3
"""Test script to verify road marking detection on images and videos."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.ground_plane import LineDetector, Line  # noqa: E402
from src.detect import YOLODetector  # noqa: E402
from src.detect.detection_types import Detection  # noqa: E402


def test_line_detection_image(
    image_path: str,
    output_path: Optional[str] = None,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 50,
    min_line_length: int = 50,
    max_line_gap: int = 10,
    show_groups: bool = False,
) -> None:
    """
    Test line detection on a single image.

    Args:
        image_path: Path to input image
        output_path: Path to save annotated output (optional)
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        hough_threshold: Accumulator threshold for Hough lines
        min_line_length: Minimum line length to detect
        max_line_gap: Maximum gap between line segments to connect
        show_groups: If True, show line groups instead of all lines
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Loaded image: {image.shape[1]}x{image.shape[0]}")

    # Initialize detector
    detector = LineDetector(
        canny_low=canny_low,
        canny_high=canny_high,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
    )

    # Detect all lines
    print("Detecting lines...")
    all_lines = detector.detect_lines(image)
    print(f"Detected {len(all_lines)} total lines")

    # Detect lane markings (horizontal lines)
    lane_lines = detector.detect_lane_markings(image)
    print(f"Detected {len(lane_lines)} lane marking lines")

    # Detect road edges (vertical lines)
    road_edges = detector.detect_road_edges(image)
    print(f"Detected {len(road_edges)} road edge lines")

    # Group parallel lines
    line_groups = detector.group_parallel_lines(all_lines)
    print(f"Grouped into {len(line_groups)} parallel line groups")

    # Visualize results
    if show_groups:
        result = detector.visualize_line_groups(image, line_groups)
        print("Visualizing line groups (different colors per group)")
    else:
        # Create composite visualization
        result = image.copy()

        # Draw all lines in gray
        for line in all_lines:
            cv2.line(result, (line.x1, line.y1), (line.x2, line.y2), (128, 128, 128), 1)

        # Draw lane markings in green
        for line in lane_lines:
            cv2.line(result, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), 2)

        # Draw road edges in blue
        for line in road_edges:
            cv2.line(result, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 0), 2)

        print("Visualization: Gray=all lines, Green=lane markings, Blue=road edges")

    # Save or display
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Saved annotated image to {output_path}")
    else:
        cv2.imshow("Line Detection", result)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def draw_detections(
    image: np.ndarray, detections: List[Detection], color: tuple, label_prefix: str = ""
) -> np.ndarray:
    """Draw detections on image with labels."""
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{label_prefix}{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return annotated


def test_line_detection_video(
    video_path: str,
    output_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 50,
    min_line_length: int = 50,
    max_line_gap: int = 10,
    show_groups: bool = False,
    show_detections: bool = False,
    detection_model: str = "yolov8m",
    detection_confidence: float = 0.4,
) -> None:
    """
    Test line detection on a video file.

    Args:
        video_path: Path to input video
        output_path: Path to save annotated output video (optional)
        max_frames: Maximum number of frames to process (None = all)
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        hough_threshold: Accumulator threshold for Hough lines
        min_line_length: Minimum line length to detect
        max_line_gap: Maximum gap between line segments to connect
        show_groups: If True, show line groups instead of all lines
        show_detections: If True, also show YOLO detections (bboxes)
        detection_model: YOLO model to use for detections
        detection_confidence: Confidence threshold for detections
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video from {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # Initialize video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Will save output to {output_path}")

    # Initialize line detector with temporal smoothing enabled
    detector = LineDetector(
        canny_low=canny_low,
        canny_high=canny_high,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
        temporal_smoothing=True,
        smoothing_alpha=0.7,  # 70% weight to new detection, 30% to previous
    )

    # Initialize YOLO detector if detections are requested
    yolo_detector = None
    if show_detections:
        print(f"Initializing YOLO detector ({detection_model})...")
        # Extract model size (e.g., "yolov8m" -> "m")
        model_size = detection_model.replace("yolov8", "").replace("yolov10", "")
        if not model_size:
            model_size = "m"  # Default to medium
        yolo_detector = YOLODetector(
            model_size=model_size,
            confidence_threshold=detection_confidence,
        )
        print("YOLO detector ready")

    frame_count = 0
    processed_frames = 0
    frame_skip = 5  # Process every 5th frame for detection, but draw on all frames

    print("Processing video frames...")
    print(
        "Note: Lines detected every 5 frames, but drawn on all frames for smooth visualization"
    )

    # Store detected lines for drawing on skipped frames
    current_lane_lines: List[Line] = []
    current_road_edges: List[Line] = []
    current_all_lines: List[Line] = []

    # Store detections
    current_detections: List[Detection] = []

    # For stable counter display (running average)
    line_count_history: List[int] = []
    lane_count_history: List[int] = []
    edge_count_history: List[int] = []
    history_window = 10  # Average over last 10 detection cycles

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect lines every N frames
        if frame_count % frame_skip == 0:
            processed_frames += 1

            # Detect lines
            current_all_lines = detector.detect_lines(frame)
            current_lane_lines = detector.detect_lane_markings(frame)
            current_road_edges = detector.detect_road_edges(frame)

            # Detect objects if requested
            if show_detections and yolo_detector:
                current_detections = yolo_detector.detect(frame, frame_id=frame_count)

            # Update count history for stable display
            line_count_history.append(len(current_all_lines))
            lane_count_history.append(len(current_lane_lines))
            edge_count_history.append(len(current_road_edges))

            # Keep only last N counts
            if len(line_count_history) > history_window:
                line_count_history.pop(0)
                lane_count_history.pop(0)
                edge_count_history.pop(0)

        # Always draw lines (using most recent detection)
        if show_groups:
            line_groups = detector.group_parallel_lines(current_all_lines)
            result = detector.visualize_line_groups(frame, line_groups)
        else:
            result = frame.copy()

            # Draw all lines in gray
            for line in current_all_lines:
                cv2.line(
                    result, (line.x1, line.y1), (line.x2, line.y2), (128, 128, 128), 1
                )

            # Draw lane markings in green
            for line in current_lane_lines:
                cv2.line(result, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), 2)

            # Draw road edges in blue
            for line in current_road_edges:
                cv2.line(result, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 0), 2)

            # Draw detections if requested
            if show_detections:
                # Separate pedestrians and vehicles
                pedestrians = [
                    d for d in current_detections if d.class_name == "person"
                ]
                vehicles = [d for d in current_detections if d.class_name != "person"]

                # Draw pedestrians in red
                result = draw_detections(result, pedestrians, (0, 0, 255), "PED ")

                # Draw vehicles in yellow
                result = draw_detections(result, vehicles, (0, 255, 255), "VEH ")

        # Calculate stable counts (running average)
        if line_count_history:
            avg_lines = int(np.mean(line_count_history))
            avg_lanes = int(np.mean(lane_count_history))
            avg_edges = int(np.mean(edge_count_history))
        else:
            avg_lines = len(current_all_lines)
            avg_lanes = len(current_lane_lines)
            avg_edges = len(current_road_edges)

        # Add text overlay with stable counts
        cv2.putText(
            result,
            f"Frame {frame_count}/{total_frames} | Lines: {avg_lines} | "
            f"Lanes: {avg_lanes} | Edges: {avg_edges}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if writer:
            writer.write(result)
        else:
            cv2.imshow("Line Detection", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Check max_frames limit on total frames, not detection cycles
        if max_frames and frame_count >= max_frames:
            print(f"Reached max_frames limit ({max_frames} total frames)")
            break

        if processed_frames % 10 == 0:
            print(
                f"Processed {processed_frames} detection cycles ({frame_count} total frames)..."
            )

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video to {output_path}")
    cv2.destroyAllWindows()

    print(
        f"Processed {processed_frames} detection cycles ({frame_count} total frames) from video"
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test road marking detection")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input image or video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output (optional)",
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=50,
        help="Lower threshold for Canny edge detection (default: 50)",
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=150,
        help="Upper threshold for Canny edge detection (default: 150)",
    )
    parser.add_argument(
        "--hough-threshold",
        type=int,
        default=50,
        help="Accumulator threshold for Hough lines (default: 50)",
    )
    parser.add_argument(
        "--min-line-length",
        type=int,
        default=50,
        help="Minimum line length to detect (default: 50)",
    )
    parser.add_argument(
        "--max-line-gap",
        type=int,
        default=10,
        help="Maximum gap between line segments (default: 10)",
    )
    parser.add_argument(
        "--show-groups",
        action="store_true",
        help="Show line groups instead of all lines",
    )
    parser.add_argument(
        "--show-detections",
        action="store_true",
        help="Also show YOLO detections (bboxes for pedestrians and vehicles)",
    )
    parser.add_argument(
        "--detection-model",
        type=str,
        default="yolov8m",
        help="YOLO model to use for detections (default: yolov8m)",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.4,
        help="Confidence threshold for detections (default: 0.4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (video only)",
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source file not found: {args.source}")
        return

    # Determine if input is image or video
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    video_extensions = {".mp4", ".mov", ".avi", ".mkv"}

    if source_path.suffix.lower() in image_extensions:
        print("Detected image input")
        test_line_detection_image(
            str(source_path),
            args.output,
            args.canny_low,
            args.canny_high,
            args.hough_threshold,
            args.min_line_length,
            args.max_line_gap,
            args.show_groups,
        )
    elif source_path.suffix.lower() in video_extensions:
        print("Detected video input")
        test_line_detection_video(
            str(source_path),
            args.output,
            args.max_frames,
            args.canny_low,
            args.canny_high,
            args.hough_threshold,
            args.min_line_length,
            args.max_line_gap,
            args.show_groups,
            args.show_detections,
            args.detection_model,
            args.detection_confidence,
        )
    else:
        print(f"Error: Unsupported file format: {source_path.suffix}")
        print(
            "Supported formats: images (.jpg, .png, .bmp) or videos (.mp4, .mov, .avi, .mkv)"
        )


if __name__ == "__main__":
    main()
