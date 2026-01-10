#!/usr/bin/env python3
"""Test script to verify track separation functionality."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detect.detection_types import Detection
from src.track import (
    filter_tracks_by_class,
    get_pedestrian_tracks,
    get_vehicle_tracks,
    group_detections_by_track_id,
    separate_pedestrian_and_vehicle_tracks,
    validate_track_ids,
)


def test_track_separation():
    """Test track separation functions with sample detections."""
    print("=" * 60)
    print("Testing Track Separation Functions")
    print("=" * 60)
    
    # Create sample detections with track IDs
    detections = [
        # Pedestrian track 1 (frame 0)
        Detection(
            bbox=[100, 100, 150, 200],
            class_id=0,
            class_name="person",
            confidence=0.9,
            frame_id=0,
            track_id=1,
        ),
        # Pedestrian track 1 (frame 1)
        Detection(
            bbox=[105, 102, 155, 202],
            class_id=0,
            class_name="person",
            confidence=0.88,
            frame_id=1,
            track_id=1,
        ),
        # Vehicle track 1 (frame 0)
        Detection(
            bbox=[200, 200, 300, 250],
            class_id=2,
            class_name="car",
            confidence=0.95,
            frame_id=0,
            track_id=2,
        ),
        # Vehicle track 1 (frame 1)
        Detection(
            bbox=[205, 200, 305, 250],
            class_id=2,
            class_name="car",
            confidence=0.94,
            frame_id=1,
            track_id=2,
        ),
        # Pedestrian track 2 (frame 1)
        Detection(
            bbox=[400, 300, 450, 400],
            class_id=0,
            class_name="person",
            confidence=0.85,
            frame_id=1,
            track_id=3,
        ),
        # Vehicle track 2 (frame 0)
        Detection(
            bbox=[500, 400, 600, 450],
            class_id=7,
            class_name="truck",
            confidence=0.92,
            frame_id=0,
            track_id=4,
        ),
    ]
    
    print(f"\nTotal detections: {len(detections)}")
    print(f"Frames: {set(d.frame_id for d in detections)}")
    print(f"Track IDs: {set(d.track_id for d in detections if d.track_id is not None)}")
    
    # Test 1: Group by track ID
    print("\n" + "-" * 60)
    print("Test 1: Group detections by track ID")
    print("-" * 60)
    tracks = group_detections_by_track_id(detections)
    print(f"Found {len(tracks)} tracks:")
    for track_id, track_detections in tracks.items():
        print(f"  Track {track_id}: {len(track_detections)} detections")
    
    # Test 2: Separate pedestrian and vehicle tracks
    print("\n" + "-" * 60)
    print("Test 2: Separate pedestrian and vehicle tracks")
    print("-" * 60)
    ped_tracks, veh_tracks = separate_pedestrian_and_vehicle_tracks(detections)
    print(f"Pedestrian tracks: {len(ped_tracks)}")
    for track_id, track_detections in ped_tracks.items():
        print(f"  Track {track_id}: {len(track_detections)} detections")
    
    print(f"\nVehicle tracks: {len(veh_tracks)}")
    for track_id, track_detections in veh_tracks.items():
        print(f"  Track {track_id}: {len(track_detections)} detections")
    
    # Test 3: Get pedestrian tracks (convenience function)
    print("\n" + "-" * 60)
    print("Test 3: Get pedestrian tracks (convenience function)")
    print("-" * 60)
    ped_tracks_2 = get_pedestrian_tracks(detections)
    print(f"Found {len(ped_tracks_2)} pedestrian tracks")
    assert len(ped_tracks_2) == len(ped_tracks), "Pedestrian track counts should match"
    print("✓ Pedestrian track counts match")
    
    # Test 4: Get vehicle tracks (convenience function)
    print("\n" + "-" * 60)
    print("Test 4: Get vehicle tracks (convenience function)")
    print("-" * 60)
    veh_tracks_2 = get_vehicle_tracks(detections)
    print(f"Found {len(veh_tracks_2)} vehicle tracks")
    assert len(veh_tracks_2) == len(veh_tracks), "Vehicle track counts should match"
    print("✓ Vehicle track counts match")
    
    # Test 5: Filter tracks by class
    print("\n" + "-" * 60)
    print("Test 5: Filter tracks by class")
    print("-" * 60)
    all_tracks = group_detections_by_track_id(detections)
    car_tracks = filter_tracks_by_class(all_tracks, "car")
    print(f"Car tracks: {len(car_tracks)}")
    for track_id in car_tracks:
        print(f"  Track {track_id}")
    
    person_tracks = filter_tracks_by_class(all_tracks, "person")
    print(f"\nPerson tracks: {len(person_tracks)}")
    for track_id in person_tracks:
        print(f"  Track {track_id}")
    
    # Test 6: Validate track IDs
    print("\n" + "-" * 60)
    print("Test 6: Validate track IDs")
    print("-" * 60)
    is_valid = validate_track_ids(detections)
    print(f"Track IDs are valid: {is_valid}")
    assert is_valid, "Track IDs should be valid"
    print("✓ Track IDs are valid")
    
    # Test 7: Test with invalid track (same track_id, different classes)
    print("\n" + "-" * 60)
    print("Test 7: Test validation with invalid track")
    print("-" * 60)
    invalid_detections = detections.copy()
    # Add a detection with same track_id but different class (should fail validation)
    invalid_detections.append(
        Detection(
            bbox=[600, 500, 700, 550],
            class_id=2,  # car
            class_name="car",
            confidence=0.9,
            frame_id=2,
            track_id=1,  # Same track_id as pedestrian track 1
        )
    )
    is_valid_invalid = validate_track_ids(invalid_detections)
    print(f"Invalid track IDs validation: {is_valid_invalid}")
    assert not is_valid_invalid, "Invalid track IDs should fail validation"
    print("✓ Invalid track IDs correctly detected")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nPhase 1.3 Track Separation: COMPLETE")
    print("\nFunctions implemented:")
    print("  ✓ group_detections_by_track_id()")
    print("  ✓ separate_pedestrian_and_vehicle_tracks()")
    print("  ✓ filter_tracks_by_class()")
    print("  ✓ get_pedestrian_tracks()")
    print("  ✓ get_vehicle_tracks()")
    print("  ✓ validate_track_ids()")


if __name__ == "__main__":
    test_track_separation()

