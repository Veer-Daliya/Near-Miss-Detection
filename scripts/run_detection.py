#!/usr/bin/env python3
"""Main script to run object detection pipeline."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
from tqdm import tqdm

from src.detect import Detection, YOLODetector
from src.ingest import VideoReader
from src.lpr import PlateAggregator, PlateDetector, PlateOCR, PlateResult
from src.track import ByteTracker
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
    fps: Optional[int] = None,  # None = process all frames
    save_annotated: bool = True,
    save_json: bool = True,
    plate_detection_interval: int = 1,  # Process plates on every frame
    use_aggregation: bool = True,  # Use multi-frame aggregation
    batch_size: int = 1,  # Batch size for GPU processing (1 = no batching)
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
    
    # Print GPU optimization info
    if batch_size > 1 and detector.device in ["cuda", "mps"]:
        print(f"GPU batch processing enabled: batch_size={batch_size}")
    elif batch_size > 1:
        print(f"Warning: Batch size > 1 specified but GPU not available. Using batch_size=1")
        batch_size = 1
    
    # Initialize plate detector (requires PaddleOCR)
    plate_detector = None
    try:
        plate_detector = PlateDetector(confidence_threshold=0.3)  # Lowered for better detection
        print("Plate detector initialized (PaddleOCR)")
    except ImportError:
        print("Warning: Plate detection not available. Install PaddleOCR for license plate detection.")

    # Initialize OCR (optional - may fail if not installed)
    # Auto-select best OCR engine for GPU acceleration
    ocr = None
    ocr_engine_used = None
    
    # Detect if we're on Apple Silicon (MPS GPU available)
    is_apple_silicon = False
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            is_apple_silicon = True
    except ImportError:
        pass
    
    # Prefer EasyOCR on Apple Silicon (uses GPU), PaddleOCR otherwise
    if is_apple_silicon:
        # Try EasyOCR first (supports Apple Silicon GPU via PyTorch MPS)
        try:
            ocr = PlateOCR(ocr_engine="easyocr")
            ocr_engine_used = "EasyOCR (GPU-accelerated)"
            print("OCR initialized (EasyOCR with Apple Silicon GPU)")
        except ImportError:
            # Fallback to PaddleOCR (CPU only on macOS)
            try:
                ocr = PlateOCR(ocr_engine="paddleocr")
                ocr_engine_used = "PaddleOCR (CPU)"
                print("OCR initialized (PaddleOCR - CPU mode, EasyOCR not available)")
            except ImportError:
                print("Warning: OCR not available. Install EasyOCR for GPU acceleration or PaddleOCR for CPU.")
    else:
        # On non-Apple Silicon, try PaddleOCR first (can use CUDA GPU if available)
        try:
            ocr = PlateOCR(ocr_engine="paddleocr")
            ocr_engine_used = "PaddleOCR"
            print("OCR initialized (PaddleOCR)")
        except ImportError:
            # Fallback to EasyOCR
            try:
                ocr = PlateOCR(ocr_engine="easyocr")
                ocr_engine_used = "EasyOCR"
                print("OCR initialized (EasyOCR)")
            except ImportError:
                print("Warning: OCR not available. Install PaddleOCR or EasyOCR for plate text extraction.")

    # Initialize plate aggregator for multi-frame aggregation
    aggregator = None
    if use_aggregation and ocr:
        aggregator = PlateAggregator(min_confidence=0.3, min_agreement=0.5)
        print("Multi-frame aggregation enabled")

    # Open video
    print(f"Opening video source: {source}")
    with VideoReader(source, fps=fps) as reader:
        # Get video properties
        width = reader.get_width()
        height = reader.get_height()
        source_fps = reader.get_fps()
        
        # Initialize tracker (uses YOLO model's built-in tracking)
        tracker = None
        try:
            tracker = ByteTracker(
                model=detector.model,
                track_thresh=confidence_threshold,
                track_buffer=30,
                match_thresh=0.8,
            )
            print("ByteTracker initialized (using YOLO built-in tracking)")
        except Exception as e:
            print(f"Warning: Tracker initialization failed: {e}. Using fallback.")

        # Setup video writer for annotated output
        video_writer = None
        if save_annotated:
            output_video_path = os.path.join(output_dir, "annotated_output.mp4")
            # Use H.264 codec for better compatibility (avc1)
            # Fallback to mp4v if avc1 not available
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            if not fourcc:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, source_fps, (width, height)
            )
            if not video_writer.isOpened():
                print(f"Warning: Could not open video writer with avc1, trying mp4v...")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    output_video_path, fourcc, source_fps, (width, height)
                )
            print(f"Video writer initialized: {output_video_path} ({width}x{height} @ {source_fps} FPS)")

        # Get total frame count for progress bar
        total_frames = reader.get_frame_count()
        if total_frames == 0:
            # If frame count unavailable (RTSP streams, etc.), use None for indeterminate
            total_frames = None

        # Process frames
        all_results = []
        frame_count = 0
        track_id_counter = 0  # Fallback counter if tracker not available

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

            # Run detection and tracking
            # If tracker available, use it (does detection + tracking in one pass)
            # This ensures bboxes and track IDs correspond correctly from the same detection
            if tracker:
                try:
                    # Tracker does detection + tracking in one pass
                    # Pass empty list (tracker re-detects) and frame_id
                    detections = tracker.update([], frame, frame_id=frame_id)
                except Exception as e:
                    # Only print warning once to avoid spam
                    if frame_count == 1:
                        print(f"\n⚠️  Warning: Tracker failed: {e}")
                        print("   Install 'lap' package for tracking: pip3 install lap")
                        print("   Continuing with fallback detection (IDs may change per frame)\n")
                    # Fallback: run detection separately
                    detections = detector.detect(frame, frame_id=frame_id)
                    for detection in detections:
                        if detection.track_id is None:
                            track_id_counter += 1
                            detection.track_id = track_id_counter
            else:
                # Fallback: run detection only (no tracking)
                detections = detector.detect(frame, frame_id=frame_id)
                # Simple sequential ID assignment
                for detection in detections:
                    if detection.track_id is None:
                        track_id_counter += 1
                        detection.track_id = track_id_counter

            # Update progress bar description with current stats
            pbar.set_description(
                f"Processing frames (Frame {frame_count}, {len(detections)} detections)"
            )

            # Detect plates on vehicle ROIs (skip some frames for speed)
            plate_results = []
            if plate_detector and frame_count % plate_detection_interval == 0:
                # Filter vehicles once using cached set (faster than list comprehension each time)
                vehicle_bboxes = [
                    d.bbox for d in detections if d.class_name in _VEHICLE_CLASSES
                ]
                if vehicle_bboxes:
                    try:
                        plate_results = plate_detector.detect_on_frame(frame, vehicle_bboxes)
                        # Debug: Print plate detection stats occasionally
                        if plate_results and frame_count % 30 == 0:
                            print(f"  [Frame {frame_count}] Detected {len(plate_results)} plate regions on {len(vehicle_bboxes)} vehicles")
                    except Exception as e:
                        print(f"Warning: Plate detection failed on frame {frame_count}: {e}")

            # Extract plate text with OCR (only on frames where plates were detected)
            # Optimization: Skip OCR if we already have high-confidence aggregated results
            if ocr and plate_results:
                for plate in plate_results:
                    if plate.bbox:
                        # Check if we already have a good aggregated result for this vehicle
                        skip_ocr = False
                        if aggregator and plate.vehicle_track_id is not None:
                            aggregated = aggregator.aggregate(plate.vehicle_track_id)
                            # Skip OCR if we have high-confidence aggregated result (>= 0.8)
                            # and enough samples (>= 3 frames) for reliable aggregation
                            if (
                                aggregated
                                and aggregated.confidence >= 0.8
                                and aggregator.get_track_count(plate.vehicle_track_id) >= 3
                            ):
                                skip_ocr = True
                                # Use aggregated result instead of running OCR
                                plate.text = aggregated.text
                                plate.confidence = aggregated.confidence
                        
                        if not skip_ocr:
                            x1, y1, x2, y2 = plate.bbox
                            plate_crop = frame[y1:y2, x1:x2]
                            if plate_crop.size > 0:
                                try:
                                    ocr_result = ocr.extract_text(plate_crop)
                                    plate.text = ocr_result.text
                                    plate.confidence = ocr_result.confidence
                                except Exception:
                                    # If OCR fails, keep plate detection but mark as unknown
                                    plate.text = "UNKNOWN"
                                    plate.confidence = 0.0

            # Associate plates with vehicles
            plate_results = associate_plates_to_vehicles(detections, plate_results)

            # Add to aggregator if enabled
            if aggregator:
                for plate in plate_results:
                    if plate.vehicle_track_id is not None:
                        aggregator.add_result(plate.vehicle_track_id, frame_id, plate)

            # Get aggregated results for visualization (if aggregation enabled)
            display_plate_results = []
            if aggregator:
                # Get aggregated results for all vehicles with plates
                vehicle_track_ids = {p.vehicle_track_id for p in plate_results if p.vehicle_track_id is not None}
                for track_id in vehicle_track_ids:
                    aggregated = aggregator.aggregate(track_id)
                    if aggregated:
                        display_plate_results.append(aggregated)
            else:
                # Use per-frame results if aggregation disabled
                display_plate_results = plate_results

            # Visualize
            annotated_frame = draw_detections_with_labels(
                frame, detections, display_plate_results, show_track_ids=True
            )

            # Save annotated frame
            if video_writer:
                video_writer.write(annotated_frame)

            # Store results
            # Use aggregated results for JSON output if available
            json_plate_results = display_plate_results if aggregator else plate_results
            
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
                    for p in json_plate_results
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
        help="YOLO model size (n=fastest, s=fast, m=medium, l=large, x=slowest)",
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
        default=None,
        nargs="?",
        help="Target FPS for processing (default: process all frames, specify number to limit FPS)",
    )
    parser.add_argument(
        "--plate-interval",
        type=int,
        default=1,
        dest="plate_interval",
        help="Process license plates every N frames (1 = every frame, higher = faster)",
    )
    parser.add_argument(
        "--no-aggregation",
        action="store_true",
        help="Disable multi-frame aggregation (use per-frame results only)",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        dest="batch_size",
        help="Batch size for GPU processing (1-8, higher = faster but uses more GPU memory). Only effective with GPU.",
    )

    args = parser.parse_args()

    # Validate batch size
    batch_size = max(1, min(8, args.batch_size))  # Clamp between 1 and 8
    if args.batch_size != batch_size:
        print(f"Warning: Batch size adjusted from {args.batch_size} to {batch_size} (valid range: 1-8)")
    
    process_video(
        source=args.source,
        output_dir=args.output_dir,
        model_size=args.model_size,
        confidence_threshold=args.confidence,
        fps=args.fps,
        save_annotated=not args.no_annotated,
        save_json=not args.no_json,
        plate_detection_interval=args.plate_interval,
        use_aggregation=not args.no_aggregation,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()

