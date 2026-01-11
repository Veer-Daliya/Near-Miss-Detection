#!/usr/bin/env python3
"""Test script to verify spatial filtering on a single image."""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.detect import YOLODetector  # noqa: E402
from src.filter import filter_pedestrians_in_vehicles  # noqa: E402


def draw_detections(
    image: np.ndarray, detections, color: tuple, label_prefix: str = ""
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


def test_spatial_filtering(
    image_path: str,
    output_path: Optional[str] = None,
    model_size: str = "m",
    confidence_threshold: float = 0.4,
    overlap_threshold: float = 0.7,
) -> None:
    """
    Test spatial filtering on a single image.

    Args:
        image_path: Path to input image
        output_path: Path to save annotated output (optional)
        model_size: YOLO model size
        confidence_threshold: Detection confidence threshold
        overlap_threshold: Overlap threshold for filtering (0.0-1.0)
    """
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Initialize detector
    print("Initializing YOLO detector...")
    detector = YOLODetector(
        model_size=model_size, confidence_threshold=confidence_threshold
    )

    # Run detection
    print("\nRunning detection...")
    detections = detector.detect(image, frame_id=0)

    # Separate pedestrians and vehicles
    pedestrians_before = [d for d in detections if d.class_name == "person"]
    vehicles = [
        d for d in detections if d.class_name in ["car", "truck", "bus", "motorcycle"]
    ]

    print("\nBefore filtering:")
    print(f"  Total detections: {len(detections)}")
    print(f"  Pedestrians: {len(pedestrians_before)}")
    print(f"  Vehicles: {len(vehicles)}")

    # Apply spatial filtering
    print(f"\nApplying spatial filtering (overlap_threshold={overlap_threshold})...")
    filtered_detections = filter_pedestrians_in_vehicles(
        detections, overlap_threshold=overlap_threshold
    )

    # Count filtered results
    pedestrians_after = [d for d in filtered_detections if d.class_name == "person"]
    filtered_out = len(pedestrians_before) - len(pedestrians_after)

    print("\nAfter filtering:")
    print(f"  Total detections: {len(filtered_detections)}")
    print(f"  Pedestrians: {len(pedestrians_after)}")
    print(f"  Vehicles: {len(vehicles)}")
    print(f"  Pedestrians filtered out: {filtered_out}")

    # Show which pedestrians were filtered
    if filtered_out > 0:
        print("\nFiltered pedestrians (inside vehicles):")
        for ped in pedestrians_before:
            if ped not in pedestrians_after:
                print(
                    f"  - {ped.class_name} at {ped.bbox} (conf: {ped.confidence:.2f})"
                )

    # Create visualization
    print("\nCreating visualization...")

    # Before filtering (all detections)
    before_image = draw_detections(image, detections, (0, 255, 0), "")
    cv2.putText(
        before_image,
        f"Before Filtering: {len(pedestrians_before)} pedestrians",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # After filtering
    after_image = draw_detections(image, filtered_detections, (255, 0, 0), "")
    cv2.putText(
        after_image,
        f"After Filtering: {len(pedestrians_after)} pedestrians ({filtered_out} removed)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    # Combine side by side
    combined = np.hstack([before_image, after_image])

    # Save or display
    if output_path:
        cv2.imwrite(output_path, combined)
        print(f"\nAnnotated image saved to: {output_path}")
    else:
        # Auto-generate output path
        input_path = Path(image_path)
        auto_output_path = (
            input_path.parent / f"{input_path.stem}_filtering_test{input_path.suffix}"
        )
        cv2.imwrite(str(auto_output_path), combined)
        print(f"\nAnnotated image saved to: {auto_output_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pedestrians before: {len(pedestrians_before)}")
    print(f"Pedestrians after: {len(pedestrians_after)}")
    print(f"Filtered out: {filtered_out}")
    print(f"Overlap threshold: {overlap_threshold}")


def main():
    parser = argparse.ArgumentParser(
        description="Test spatial filtering on a single image"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output (default: auto-generated)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="m",
        choices=["n", "s", "m", "l", "x"],
        help="YOLO model size",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.7,
        help="Overlap threshold for filtering (0.0-1.0, default: 0.7)",
    )

    args = parser.parse_args()

    test_spatial_filtering(
        image_path=args.image,
        output_path=args.output,
        model_size=args.model_size,
        confidence_threshold=args.confidence,
        overlap_threshold=args.overlap_threshold,
    )


if __name__ == "__main__":
    main()
