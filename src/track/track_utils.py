"""Utility functions for track management and separation."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.detect.detection_types import Detection

# Vehicle class names (matching YOLO COCO classes)
_VEHICLE_CLASSES = frozenset(["car", "truck", "bus", "motorcycle"])


def group_detections_by_track_id(
    detections: List[Detection],
) -> Dict[Optional[int], List[Detection]]:
    """
    Group detections by their track ID.
    
    Args:
        detections: List of detections with track IDs
        
    Returns:
        Dictionary mapping track_id -> list of detections for that track
    """
    tracks: Dict[Optional[int], List[Detection]] = defaultdict(list)
    
    for detection in detections:
        track_id = detection.track_id
        tracks[track_id].append(detection)
    
    return dict(tracks)


def get_track_class(track_detections: List[Detection]) -> Optional[str]:
    """
    Get the class name for a track (assumes all detections in track have same class).
    
    Args:
        track_detections: List of detections belonging to the same track
        
    Returns:
        Class name (e.g., "person", "car") or None if empty
    """
    if not track_detections:
        return None
    
    # All detections in a track should have the same class
    # Return the most common class (in case of any inconsistencies)
    class_counts: Dict[str, int] = defaultdict(int)
    for det in track_detections:
        class_counts[det.class_name] += 1
    
    return max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None


def separate_pedestrian_and_vehicle_tracks(
    detections: List[Detection],
) -> Tuple[Dict[Optional[int], List[Detection]], Dict[Optional[int], List[Detection]]]:
    """
    Separate detections into pedestrian tracks and vehicle tracks.
    
    Groups detections by track_id and separates them by class.
    
    Args:
        detections: List of all detections (pedestrians and vehicles)
        
    Returns:
        Tuple of (pedestrian_tracks, vehicle_tracks) dictionaries
        Each dictionary maps track_id -> list of detections for that track
    """
    # Group all detections by track_id
    all_tracks = group_detections_by_track_id(detections)
    
    pedestrian_tracks: Dict[Optional[int], List[Detection]] = {}
    vehicle_tracks: Dict[Optional[int], List[Detection]] = {}
    
    for track_id, track_detections in all_tracks.items():
        if not track_detections:
            continue
        
        # Get the class for this track
        class_name = get_track_class(track_detections)
        
        if class_name == "person":
            pedestrian_tracks[track_id] = track_detections
        elif class_name in _VEHICLE_CLASSES:
            vehicle_tracks[track_id] = track_detections
    
    return pedestrian_tracks, vehicle_tracks


def filter_tracks_by_class(
    tracks: Dict[Optional[int], List[Detection]],
    class_name: str,
) -> Dict[Optional[int], List[Detection]]:
    """
    Filter tracks to only include those matching a specific class.
    
    Args:
        tracks: Dictionary mapping track_id -> list of detections
        class_name: Class name to filter by (e.g., "person", "car")
        
    Returns:
        Filtered dictionary containing only tracks of the specified class
    """
    filtered: Dict[Optional[int], List[Detection]] = {}
    
    for track_id, track_detections in tracks.items():
        track_class = get_track_class(track_detections)
        if track_class == class_name:
            filtered[track_id] = track_detections
    
    return filtered


def get_pedestrian_tracks(
    detections: List[Detection],
) -> Dict[Optional[int], List[Detection]]:
    """
    Get all pedestrian tracks from detections.
    
    Convenience function that groups detections and filters for pedestrians.
    
    Args:
        detections: List of all detections
        
    Returns:
        Dictionary mapping track_id -> list of pedestrian detections
    """
    tracks = group_detections_by_track_id(detections)
    return filter_tracks_by_class(tracks, "person")


def get_vehicle_tracks(
    detections: List[Detection],
) -> Dict[Optional[int], List[Detection]]:
    """
    Get all vehicle tracks from detections.
    
    Convenience function that groups detections and filters for vehicles.
    
    Args:
        detections: List of all detections
        
    Returns:
        Dictionary mapping track_id -> list of vehicle detections
    """
    tracks = group_detections_by_track_id(detections)
    vehicle_tracks: Dict[Optional[int], List[Detection]] = {}
    
    for track_id, track_detections in tracks.items():
        track_class = get_track_class(track_detections)
        if track_class in _VEHICLE_CLASSES:
            vehicle_tracks[track_id] = track_detections
    
    return vehicle_tracks


def get_track_detections_at_frame(
    track_detections: List[Detection],
    frame_id: int,
) -> Optional[Detection]:
    """
    Get the detection for a specific track at a specific frame.
    
    Args:
        track_detections: List of detections for a track
        frame_id: Frame ID to look up
        
    Returns:
        Detection at the specified frame, or None if not found
    """
    for detection in track_detections:
        if detection.frame_id == frame_id:
            return detection
    return None


def get_active_track_ids(
    detections: List[Detection],
    frame_id: int,
) -> List[Optional[int]]:
    """
    Get all track IDs that are active (have detections) at a specific frame.
    
    Args:
        detections: List of all detections
        frame_id: Frame ID to check
        
    Returns:
        List of track IDs active at the specified frame
    """
    active_track_ids = set()
    
    for detection in detections:
        if detection.frame_id == frame_id and detection.track_id is not None:
            active_track_ids.add(detection.track_id)
    
    return list(active_track_ids)


def validate_track_ids(detections: List[Detection]) -> bool:
    """
    Validate that track IDs are maintained correctly across frames.
    
    Checks that:
    - Track IDs are consistent (same track_id has same class_name)
    - Track IDs are not None (if tracking is working)
    
    Args:
        detections: List of detections to validate
        
    Returns:
        True if track IDs are valid, False otherwise
    """
    tracks = group_detections_by_track_id(detections)
    
    for track_id, track_detections in tracks.items():
        if track_id is None:
            # None track IDs are okay (untracked detections)
            continue
        
        # Check that all detections in a track have the same class
        classes = {det.class_name for det in track_detections}
        if len(classes) > 1:
            # Track has multiple classes - this is an error
            return False
    
    return True

