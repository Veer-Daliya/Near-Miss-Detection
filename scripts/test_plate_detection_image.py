#!/usr/bin/env python3
"""Diagnostic script to test license plate detection on images with no filtering."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.detect import YOLODetector  # noqa: E402
from src.lpr.ocr import PlateOCR  # noqa: E402

# Cache vehicle class names
_VEHICLE_CLASSES = frozenset(["car", "truck", "bus", "motorcycle"])


def save_yolo_detection_incremental(
    output_dir: str, image_name: str, detections: list
) -> None:
    """
    Save YOLO detections incrementally to JSON Lines file.
    Uses JSON Lines format (.jsonl) for efficient append-only writes.

    Args:
        output_dir: Output directory
        image_name: Image filename
        detections: List of detection dictionaries
    """
    jsonl_path = os.path.join(output_dir, "yolo_detections.jsonl")

    # Prepare image data
    image_data = {
        "image": image_name,
        "detections": detections,
        "detection_count": len(detections),
    }

    # Append as a single line (JSON Lines format)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(image_data) + "\n")


def save_ocr_detection_incremental(
    output_dir: str, image_name: str, ocr_results: list
) -> None:
    """
    Save OCR detections incrementally to JSON Lines file.
    Uses JSON Lines format (.jsonl) for efficient append-only writes.

    Args:
        output_dir: Output directory
        image_name: Image filename
        ocr_results: List of OCR result dictionaries
    """
    jsonl_path = os.path.join(output_dir, "ocr_bboxes.jsonl")

    # Prepare image data
    image_data = {
        "image": image_name,
        "ocr_results": ocr_results,
        "ocr_count": len(ocr_results),
    }

    # Append as a single line (JSON Lines format)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(image_data) + "\n")


def _standardize_bbox_from_polygon(polygon) -> tuple[int, int, int, int]:
    """
    Standardize bbox extraction from polygon points.
    Handles various polygon formats from PaddleOCR/EasyOCR.

    Returns: (x1, y1, x2, y2) in standard format
    """
    try:
        import numpy as np

        poly_array = np.array(polygon)

        # Handle different array shapes
        if poly_array.ndim == 1:
            # Flat array [x1, y1, x2, y2, ...]
            if len(poly_array) >= 4:
                x_coords = poly_array[::2]  # Even indices
                y_coords = poly_array[1::2]  # Odd indices
            else:
                raise ValueError("Polygon array too short")
        elif poly_array.ndim == 2:
            # 2D array - could be (n_points, 2) or (2, n_points)
            if poly_array.shape[1] == 2:
                # Standard format: (n_points, 2)
                x_coords = poly_array[:, 0]
                y_coords = poly_array[:, 1]
            elif poly_array.shape[0] == 2:
                # Transposed: (2, n_points)
                x_coords = poly_array[0, :]
                y_coords = poly_array[1, :]
            else:
                raise ValueError(f"Unexpected polygon shape: {poly_array.shape}")
        else:
            raise ValueError(f"Unexpected polygon dimensions: {poly_array.ndim}")

        x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
        x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))

        return x1, y1, x2, y2
    except Exception as e:
        # Fallback to manual extraction
        if isinstance(polygon, (list, tuple)) and len(polygon) > 0:
            if isinstance(polygon[0], (list, tuple, np.ndarray)):
                x_list = [float(p[0]) for p in polygon]
                y_list = [float(p[1]) for p in polygon]
                return (
                    int(min(x_list)),
                    int(min(y_list)),
                    int(max(x_list)),
                    int(max(y_list)),
                )
        raise ValueError(f"Could not parse polygon: {polygon}, error: {e}")


def detect_all_text_regions(image: np.ndarray, ocr, min_confidence: float = 0.0):
    """
    Detect ALL text regions using OCR with no filtering.

    Returns list of detection dicts with bbox, text, confidence and other metadata.
    """
    all_detections: List[dict[str, Any]] = []

    try:
        # Get the underlying OCR instance
        # PlateOCR wraps either PaddleOCR (ocr.ocr) or EasyOCR (ocr.reader)
        if hasattr(ocr, "ocr"):
            # PaddleOCR - the .ocr attribute is the PaddleOCR instance, which has .ocr() method
            ocr_instance = ocr.ocr
            results = ocr_instance.ocr(image)
            print(f"DEBUG: PaddleOCR returned type: {type(results)}, value: {results}")
        elif hasattr(ocr, "reader"):
            # EasyOCR - convert to PaddleOCR-like format
            results_easyocr = ocr.reader.readtext(image)
            print(f"DEBUG: EasyOCR returned {len(results_easyocr)} detections")
            # Convert EasyOCR format to PaddleOCR-like format
            results = [[(bbox, (text, conf))] for bbox, text, conf in results_easyocr]
        else:
            print("Error: Unknown OCR engine type")
            return all_detections

        print(f"DEBUG: Results after processing: {results}")
        if not results:
            print("DEBUG: Results is empty or None")
            return all_detections

        # Handle EasyOCR format (list of lists) vs PaddleOCR format
        if hasattr(ocr, "reader"):
            # EasyOCR - results is already in the right format
            text_lines = []
            for result_group in results:
                if result_group:
                    text_lines.extend(result_group)
            print(f"DEBUG: EasyOCR text_lines count: {len(text_lines)}")
        else:
            # PaddleOCR - Handle new dictionary format vs old tuple format
            print(f"DEBUG: Processing PaddleOCR results, type: {type(results)}")
            ocr_result = (
                results[0]
                if isinstance(results, list) and len(results) > 0
                else results
            )

            # Check if it's the new dictionary format (PaddleOCR v2+)
            if isinstance(ocr_result, dict):
                print("DEBUG: Found dictionary format (new PaddleOCR API)")
                rec_texts = ocr_result.get("rec_texts", [])
                rec_scores = ocr_result.get("rec_scores", [])
                rec_polys = ocr_result.get("rec_polys", [])
                print(f"DEBUG: Found {len(rec_texts)} text detections")

                # Convert to old format for compatibility
                text_lines = []
                for i, (text, score, poly) in enumerate(
                    zip(rec_texts, rec_scores, rec_polys)
                ):
                    # Convert polygon to bbox points format using standardized function
                    try:
                        x1, y1, x2, y2 = _standardize_bbox_from_polygon(poly)
                        # Convert back to bbox_points format for compatibility
                        bbox_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    except Exception as e:
                        print(f"Warning: Could not parse polygon {i}: {e}")
                        # Fallback: try direct extraction
                        import numpy as np

                        poly_array = np.array(poly)
                        if poly_array.ndim == 2 and poly_array.shape[1] == 2:
                            bbox_points = [
                                [int(poly_array[j][0]), int(poly_array[j][1])]
                                for j in range(poly_array.shape[0])
                            ]
                        else:
                            continue
                    text_lines.append((bbox_points, (text, float(score))))
                print(f"DEBUG: Converted to {len(text_lines)} text lines")
            elif hasattr(ocr_result, "text_lines") or hasattr(ocr_result, "rec_res"):
                print("DEBUG: Found OCRResult object with attributes")
                if hasattr(ocr_result, "text_lines") and ocr_result.text_lines:
                    text_lines = ocr_result.text_lines
                    print(f"DEBUG: Using text_lines, count: {len(text_lines)}")
                elif hasattr(ocr_result, "rec_res") and ocr_result.rec_res:
                    text_lines = ocr_result.rec_res
                    print(f"DEBUG: Using rec_res, count: {len(text_lines)}")
                else:
                    print("DEBUG: OCRResult has no valid text_lines or rec_res")
                    return all_detections
            elif isinstance(ocr_result, list):
                text_lines = ocr_result
                print(
                    f"DEBUG: ocr_result is list (old format), count: {len(text_lines)}"
                )
            else:
                print(f"DEBUG: Unknown ocr_result format: {type(ocr_result)}")
                return all_detections

        for line in text_lines:
            if not line:
                continue

            # Handle different line formats
            if isinstance(line, tuple) and len(line) == 2:
                bbox_points, (text, confidence) = line
            elif isinstance(line, dict):
                bbox_points = line.get("bbox", [])
                text = line.get("text", "")
                confidence = line.get("confidence", 0.0)
            else:
                continue

            # Only filter by minimum confidence (no other filters)
            if confidence < min_confidence:
                continue

            # Convert bbox points to [x1, y1, x2, y2] - use standardized function
            if isinstance(bbox_points, list) and len(bbox_points) > 0:
                try:
                    roi_x1, roi_y1, roi_x2, roi_y2 = _standardize_bbox_from_polygon(
                        bbox_points
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not parse bbox_points: {bbox_points}, error: {e}"
                    )
                    continue

                width = roi_x2 - roi_x1
                height = roi_y2 - roi_y1
                aspect_ratio = width / height if height > 0 else 0
                area = width * height

                all_detections.append(
                    {
                        "bbox": [roi_x1, roi_y1, roi_x2, roi_y2],
                        "text": text,
                        "confidence": confidence,
                        "width": width,
                        "height": height,
                        "aspect_ratio": aspect_ratio,
                        "area": area,
                    }
                )

    except Exception as e:
        print(f"Error in text detection: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()

    return all_detections


def process_image(
    image_path: str,
    output_path: Optional[str] = None,
    model_size: str = "m",
    confidence_threshold: float = 0.4,
    plate_confidence_threshold: float = 0.0,  # No filtering by default
    add_margin: bool = True,
    margin_pixels: int = 20,
):
    """
    Process a single image and detect license plates with no filtering.

    Args:
        image_path: Path to input image
        output_path: Path to save annotated output (optional)
        model_size: YOLO model size
        confidence_threshold: Vehicle detection confidence threshold
        plate_confidence_threshold: Minimum confidence for text detection (0.0 = no filter)
        add_margin: Whether to add margin around vehicle bboxes
        margin_pixels: Number of pixels to add as margin
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

    # Initialize OCR
    print("Initializing OCR...")
    ocr = None
    try:
        # Try EasyOCR first (better on Apple Silicon), then PaddleOCR
        try:
            ocr = PlateOCR(ocr_engine="easyocr")
            print("Using EasyOCR")
        except ImportError:
            ocr = PlateOCR(ocr_engine="paddleocr")
            print("Using PaddleOCR")
    except ImportError as e:
        print(f"Error: OCR not available. {e}")
        return

    # Detect vehicles
    print("\nDetecting vehicles...")
    detections = detector.detect(image, frame_id=0)
    vehicles = [d for d in detections if d.class_name in _VEHICLE_CLASSES]
    print(f"Found {len(vehicles)} vehicles")

    for i, vehicle in enumerate(vehicles):
        print(
            f"  Vehicle {i + 1}: {vehicle.class_name} (confidence: {vehicle.confidence:.2f})"
        )
        print(f"    Bbox: {vehicle.bbox}")

    # Create annotated image
    annotated = image.copy()

    # Draw vehicle bounding boxes
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{vehicle.class_name} {vehicle.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Detect text on vehicle ROIs (production approach)
    print("\n" + "=" * 60)
    print("DETECTING TEXT ON VEHICLE ROIs")
    print("=" * 60)

    # Collect all OCR results for saving
    all_ocr_results = []
    total_roi_detections = 0
    for i, vehicle in enumerate(vehicles):
        print(f"\nVehicle {i + 1} ({vehicle.class_name}):")
        x1, y1, x2, y2 = vehicle.bbox

        # Add margin if requested
        if add_margin:
            img_h, img_w = image.shape[:2]
            x1 = max(0, x1 - margin_pixels)
            y1 = max(0, y1 - margin_pixels)
            x2 = min(img_w, x2 + margin_pixels)
            y2 = min(img_h, y2 + margin_pixels)
            print(f"  Original bbox: {vehicle.bbox}")
            print(f"  With margin ({margin_pixels}px): [{x1}, {y1}, {x2}, {y2}]")

        vehicle_roi = image[y1:y2, x1:x2]

        if vehicle_roi.size == 0:
            print("  Empty ROI, skipping")
            continue

        print(f"  ROI size: {vehicle_roi.shape[1]}x{vehicle_roi.shape[0]}")

        # Detect text in ROI
        roi_text = detect_all_text_regions(
            vehicle_roi, ocr, min_confidence=plate_confidence_threshold
        )
        print(f"  Found {len(roi_text)} text regions in ROI")
        total_roi_detections += len(roi_text)

        for j, det in enumerate(roi_text):
            # Convert ROI coordinates to frame coordinates
            frame_x1 = x1 + det["bbox"][0]
            frame_y1 = y1 + det["bbox"][1]
            frame_x2 = x1 + det["bbox"][2]
            frame_y2 = y1 + det["bbox"][3]

            print(f"    Text {j + 1}: '{det['text']}' (conf: {det['confidence']:.3f})")
            print(f"      ROI bbox: {det['bbox']}")
            print(f"      Frame bbox: [{frame_x1}, {frame_y1}, {frame_x2}, {frame_y2}]")
            print(
                f"      Size: {det['width']}x{det['height']}, Aspect: {det['aspect_ratio']:.2f}"
            )

            # Store OCR result with frame coordinates
            ocr_result = {
                "vehicle_index": i,
                "vehicle_class": vehicle.class_name,
                "vehicle_bbox": vehicle.bbox,
                "roi_bbox": det["bbox"],
                "frame_bbox": [frame_x1, frame_y1, frame_x2, frame_y2],
                "text": det["text"],
                "confidence": det["confidence"],
                "width": det["width"],
                "height": det["height"],
                "aspect_ratio": det["aspect_ratio"],
                "area": det["area"],
                "source": "ocr",
            }
            all_ocr_results.append(ocr_result)

            # Draw on annotated image
            cv2.rectangle(
                annotated, (frame_x1, frame_y1), (frame_x2, frame_y2), (255, 0, 0), 2
            )
            label = f"{det['text']} ({det['confidence']:.2f})"
            cv2.putText(
                annotated,
                label,
                (frame_x1, frame_y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    # Prepare output directory
    input_path = Path(image_path)
    output_dir = input_path.parent
    image_name = input_path.name

    # Save YOLO detections incrementally (JSON Lines format)
    yolo_detections = [
        {
            "bbox": d.bbox,
            "class_name": d.class_name,
            "class_id": d.class_id if hasattr(d, "class_id") else None,
            "confidence": d.confidence,
            "track_id": d.track_id if hasattr(d, "track_id") else None,
        }
        for d in detections
    ]
    save_yolo_detection_incremental(str(output_dir), image_name, yolo_detections)
    print(
        f"\nYOLO detections saved to: {os.path.join(output_dir, 'yolo_detections.jsonl')}"
    )

    # Save OCR detections incrementally (JSON Lines format)
    if all_ocr_results:
        save_ocr_detection_incremental(str(output_dir), image_name, all_ocr_results)
        print(
            f"OCR detections saved to: {os.path.join(output_dir, 'ocr_bboxes.jsonl')}"
        )

    # Save annotated image
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"\nAnnotated image saved to: {output_path}")
    else:
        # Auto-generate output path
        input_path = Path(image_path)
        auto_output_path = (
            input_path.parent / f"{input_path.stem}_annotated{input_path.suffix}"
        )
        cv2.imwrite(str(auto_output_path), annotated)
        print(f"\nAnnotated image saved to: {auto_output_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Vehicles detected: {len(vehicles)}")
    print(f"Text regions detected (vehicle ROIs): {total_roi_detections}")


def main():
    parser = argparse.ArgumentParser(
        description="Test license plate detection on images with no filtering"
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
        help="Vehicle detection confidence threshold",
    )
    parser.add_argument(
        "--plate-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence for text detection (0.0 = no filter)",
    )
    parser.add_argument(
        "--no-margin",
        action="store_true",
        help="Don't add margin around vehicle bboxes",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=20,
        help="Margin pixels to add around vehicle bboxes",
    )

    args = parser.parse_args()

    process_image(
        image_path=args.image,
        output_path=args.output,
        model_size=args.model_size,
        confidence_threshold=args.confidence,
        plate_confidence_threshold=args.plate_confidence,
        add_margin=not args.no_margin,
        margin_pixels=args.margin,
    )


if __name__ == "__main__":
    main()
