"""Spatial filtering to remove pedestrians inside vehicles."""

from typing import List

from src.detect.detection_types import Detection


# Vehicle class names (matching YOLO COCO classes)
_VEHICLE_CLASSES = frozenset(["car", "truck", "bus", "motorcycle"])


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union (IOU) between two bounding boxes.

    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IOU value between 0.0 and 1.0
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # No intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_overlap_ratio(ped_bbox: List[int], vehicle_bbox: List[int]) -> float:
    """
    Calculate what fraction of pedestrian bbox overlaps with vehicle bbox.

    This is different from IOU - it measures how much of the pedestrian
    is inside the vehicle, not the mutual overlap.

    Args:
        ped_bbox: Pedestrian bounding box [x1, y1, x2, y2]
        vehicle_bbox: Vehicle bounding box [x1, y1, x2, y2]

    Returns:
        Ratio of pedestrian area that overlaps with vehicle (0.0 to 1.0)
    """
    x1_p, y1_p, x2_p, y2_p = ped_bbox
    x1_v, y1_v, x2_v, y2_v = vehicle_bbox

    # Calculate intersection
    x1_i = max(x1_p, x1_v)
    y1_i = max(y1_p, y1_v)
    x2_i = min(x2_p, x2_v)
    y2_i = min(y2_p, y2_v)

    # No intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    ped_area = (x2_p - x1_p) * (y2_p - y1_p)

    return intersection / ped_area if ped_area > 0 else 0.0


def filter_pedestrians_in_vehicles(
    detections: List[Detection],
    overlap_threshold: float = 0.7,
) -> List[Detection]:
    """
    Filter out pedestrian detections that are inside vehicle bounding boxes.

    This prevents tracking people sitting in cars, which would cause false
    positives in near miss detection.

    Args:
        detections: List of all detections (pedestrians and vehicles)
        overlap_threshold: Minimum overlap ratio to consider pedestrian as "inside"
                         (default: 0.7 = 70% of pedestrian bbox must overlap)

    Returns:
        Filtered list of detections with pedestrians inside vehicles removed
    """
    # Separate pedestrians and vehicles
    pedestrians = [d for d in detections if d.class_name == "person"]
    vehicles = [d for d in detections if d.class_name in _VEHICLE_CLASSES]

    # If no vehicles, return all detections unchanged
    if not vehicles:
        return detections

    # Filter pedestrians
    filtered_pedestrians = []
    for ped in pedestrians:
        is_inside_vehicle = False

        for vehicle in vehicles:
            # Calculate overlap ratio (how much of pedestrian is inside vehicle)
            overlap_ratio = calculate_overlap_ratio(ped.bbox, vehicle.bbox)

            # If pedestrian is mostly inside vehicle bbox, filter it out
            if overlap_ratio >= overlap_threshold:
                is_inside_vehicle = True
                break

        # Only keep pedestrians NOT inside vehicles
        if not is_inside_vehicle:
            filtered_pedestrians.append(ped)

    # Combine filtered pedestrians with all vehicles (vehicles unchanged)
    filtered_detections = filtered_pedestrians + vehicles

    return filtered_detections


def separate_pedestrians_and_vehicles(
    detections: List[Detection],
) -> tuple[List[Detection], List[Detection]]:
    """
    Separate detections into pedestrians and vehicles.

    Args:
        detections: List of all detections

    Returns:
        Tuple of (pedestrians, vehicles) lists
    """
    pedestrians = [d for d in detections if d.class_name == "person"]
    vehicles = [d for d in detections if d.class_name in _VEHICLE_CLASSES]

    return pedestrians, vehicles
