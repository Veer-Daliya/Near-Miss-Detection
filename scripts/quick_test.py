#!/usr/bin/env python3
"""Quick test script - processes only first 50 frames with fast settings."""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detect import YOLODetector  # noqa: E402
from src.ingest import VideoReader  # noqa: E402
from tqdm import tqdm  # noqa: E402


def quick_test(source: str, max_frames: int = 50) -> None:
    """Process first N frames quickly."""
    print(f"Quick test: Processing first {max_frames} frames from {source}")
    print("Using fast settings: model=n, no plate detection, no OCR\n")
    
    # Initialize detector with smallest model
    detector = YOLODetector(model_size="n", confidence_threshold=0.4)
    
    # Open video
    with VideoReader(source) as reader:
        total_frames = reader.get_frame_count()
        print(f"Video has {total_frames} total frames")
        print(f"Processing first {max_frames} frames only\n")
        
        frame_count = 0
        total_detections = 0
        
        pbar = tqdm(total=min(max_frames, total_frames), desc="Processing", unit="frame")
        
        for frame, frame_id, timestamp in reader:
            if frame_count >= max_frames:
                break
                
            # Run detection only (no tracking, no plates, no OCR)
            detections = detector.detect(frame, frame_id=frame_id)
            total_detections += len(detections)
            
            # Update progress
            pbar.set_description(f"Frame {frame_count+1}/{max_frames} ({len(detections)} detections)")
            pbar.update(1)
            
            frame_count += 1
        
        pbar.close()
        
        print("\n✅ Quick test complete!")
        print(f"   Processed: {frame_count} frames")
        print(f"   Total detections: {total_detections}")
        print(f"   Average detections per frame: {total_detections/frame_count:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test - process first 50 frames")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video file path",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Maximum frames to process (default: 50)",
    )
    
    args = parser.parse_args()
    quick_test(args.source, args.max_frames)



