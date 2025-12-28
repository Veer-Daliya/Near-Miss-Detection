#!/usr/bin/env python3
"""Main script to run object detection pipeline."""

import argparse
import json
import os
from typing import List

import cv2
from tqdm import tqdm

from src.detect import Detection, YOLODetector
from src.ingest import VideoReader
from src.lpr import PlateDetector, PlateOCR, PlateResult
from src.utils import draw_detections_with_labels


# Cache vehicle class names for faster filtering
_VEHICLE_CLASSES = frozenset(["car", "truck", "bus", "motorcycle"])


def associate_plates_to_vehicles(
    detections: List[Detection], plate_results: List[PlateResult]
) -> List[PlateResult]:
    """
    Associate plate results with vehicle detections.
    
    Optimized: Filters vehicles once and uses spatial proximity.
    """
    if not plate_results:
        return plate_results
    
    # Filter vehicles once (O(n) instead of O(n*m))
    vehicles = [d for d in detections if d.class_name in _VEHICLE_CLASSES]
    
    if not vehicles:
        return plate_results
    
    # Pre-compute vehicle centers for faster distance calculation
    vehicle_centers = []
    for vehicle in vehicles:
        cx = (vehicle.bbox[0] + vehicle.bbox[2]) * 0.5
        cy = (vehicle.bbox[1] + vehicle.bbox[3]) * 0.5
        vehicle_centers.append((cx, cy, vehicle))
    
    # Associate each plate to nearest vehicle
    for plate in plate_results:
        if not plate.bbox:
            continue
            
        plate_cx = (plate.bbox[0] + plate.bbox[2]) * 0.5
        plate_cy = (plate.bbox[1] + plate.bbox[3]) * 0.5
        
        best_vehicle = None
        min_distance_sq = float("inf")
        
        # Find closest vehicle (using squared distance to avoid sqrt)
        for vx, vy, vehicle in vehicle_centers:
            dx = plate_cx - vx
            dy = plate_cy - vy
            distance_sq = dx * dx + dy * dy
            
            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                best_vehicle = vehicle
        
        if best_vehicle and best_vehicle.track_id:
            plate.vehicle_track_id = best_vehicle.track_id
    
    return plate_results


def process_video(
    source: str,
    output_dir: str,
    model_size: str = "m",
    confidence_threshold: float = 0.4,
    fps: int = 10,
    save_annotated: bool = True,
    save_json: bool = True,
) -> None:
    """
    Process video and run detection pipeline.

    Args:
        source: Video file path, RTSP URL, or webcam device ID
        output_dir: Directory to save outputs
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        confidence_threshold: Detection confidence threshold
        fps: Target FPS for processing
        save_annotated: Whether to save annotated video
        save_json: Whether to save JSON results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize models
    print("Initializing detection models...")
    detector = YOLODetector(
        model_size=model_size, confidence_threshold=confidence_threshold
    )
    
    # Initialize plate detector (requires PaddleOCR)
    plate_detector = None
    try:
        plate_detector = PlateDetector(confidence_threshold=0.5)
        print("Plate detector initialized (PaddleOCR)")
    except ImportError:
        print("Warning: Plate detection not available. Install PaddleOCR for license plate detection.")

    # Initialize OCR (optional - may fail if not installed)
    ocr = None
    try:
        ocr = PlateOCR(ocr_engine="paddleocr")
        print("OCR initialized (PaddleOCR)")
    except ImportError:
        try:
            ocr = PlateOCR(ocr_engine="easyocr")
            print("OCR initialized (EasyOCR)")
        except ImportError:
            print("Warning: OCR not available. Install PaddleOCR or EasyOCR for plate text extraction.")

    # Open video
    print(f"Opening video source: {source}")
    with VideoReader(source, fps=fps) as reader:
        # Get video properties
        width = reader.get_width()
        height = reader.get_height()
        source_fps = reader.get_fps()

        # Setup video writer for annotated output
        video_writer = None
        if save_annotated:
            output_video_path = os.path.join(output_dir, "annotated_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, source_fps, (width, height)
            )

        # Get total frame count for progress bar
        total_frames = reader.get_frame_count()
        if total_frames == 0:
            # If frame count unavailable (RTSP streams, etc.), use None for indeterminate
            total_frames = None

        # Process frames
        all_results = []
        frame_count = 0
        track_id_counter = 0

        # Create progress bar
        pbar = tqdm(
            desc="Processing frames",
            total=total_frames,
            unit="frame",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            disable=False,
        )

        for frame, frame_id, timestamp in reader:
            frame_count += 1

            # Run detection
            detections = detector.detect(frame, frame_id=frame_id)

            # Update progress bar description with current stats
            pbar.set_description(
                f"Processing frames (Frame {frame_count}, {len(detections)} detections)"
            )

            # Assign simple track IDs (in production, use ByteTrack)
            for detection in detections:
                if detection.track_id is None:
                    track_id_counter += 1
                    detection.track_id = track_id_counter

            # Detect plates on vehicle ROIs
            plate_results = []
            if plate_detector:
                # Filter vehicles once using cached set (faster than list comprehension each time)
                vehicle_bboxes = [
                    d.bbox for d in detections if d.class_name in _VEHICLE_CLASSES
                ]
                if vehicle_bboxes:
                    plate_results = plate_detector.detect_on_frame(frame, vehicle_bboxes)

            # Extract plate text with OCR
            if ocr:
                for plate in plate_results:
                    if plate.bbox:
                        x1, y1, x2, y2 = plate.bbox
                        plate_crop = frame[y1:y2, x1:x2]
                        if plate_crop.size > 0:
                            ocr_result = ocr.extract_text(plate_crop)
                            plate.text = ocr_result.text
                            plate.confidence = ocr_result.confidence

            # Associate plates with vehicles
            plate_results = associate_plates_to_vehicles(detections, plate_results)

            # Visualize
            annotated_frame = draw_detections_with_labels(
                frame, detections, plate_results, show_track_ids=True
            )

            # Save annotated frame
            if video_writer:
                video_writer.write(annotated_frame)

            # Store results
            frame_result = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "detections": [
                    {
                        "bbox": d.bbox,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "track_id": d.track_id,
                    }
                    for d in detections
                ],
                "plates": [
                    {
                        "text": p.text,
                        "confidence": p.confidence,
                        "bbox": p.bbox,
                        "vehicle_track_id": p.vehicle_track_id,
                    }
                    for p in plate_results
                ],
            }
            all_results.append(frame_result)

            # Update progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()

        # Cleanup
        if video_writer:
            video_writer.release()

    # Save JSON results
    if save_json:
        json_path = os.path.join(output_dir, "detections.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {json_path}")

    if save_annotated:
        print(f"Annotated video saved to {os.path.join(output_dir, 'annotated_output.mp4')}")

    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total detections: {sum(len(r['detections']) for r in all_results)}")
    print(f"Total plates detected: {sum(len(r['plates']) for r in all_results)}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run object detection pipeline")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video file path, RTSP URL, or webcam device ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs",
        help="Output directory for results",
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
        "--fps",
        type=int,
        default=10,
        help="Target FPS for processing",
    )
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Don't save annotated video",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Don't save JSON results",
    )

    args = parser.parse_args()

    process_video(
        source=args.source,
        output_dir=args.output_dir,
        model_size=args.model_size,
        confidence_threshold=args.confidence,
        fps=args.fps,
        save_annotated=not args.no_annotated,
        save_json=not args.no_json,
    )


if __name__ == "__main__":
    main()

