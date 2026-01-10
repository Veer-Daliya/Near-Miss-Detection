# Properly Filtered Detections: Explanation

This document explains what "properly filtered" means in the context of pedestrian and vehicle detection.

---

## What Does "Properly Filtered" Mean?

### In Simple Terms
"Properly filtered" means:
- **Only detecting what you need** (pedestrians and vehicles)
- **Removing false positives** (things detected incorrectly)
- **Removing unwanted detections** (people inside cars)
- **Ensuring quality** (high confidence, correct classes)

---

## Current State of Your System

### What You Already Have
✅ **Detection**: YOLO detects `person`, `car`, `truck`, `bus`, `motorcycle`  
✅ **Tracking**: ByteTrack assigns track IDs to all detections  
✅ **Class Filtering**: Only processes classes you care about

### What Might Be Missing
❓ **Spatial Filtering**: Removing people inside vehicles  
❓ **Confidence Filtering**: Ensuring only high-confidence detections  
❓ **Size Filtering**: Removing very small detections (likely false positives)  
❓ **Track Quality**: Ensuring tracks are stable and consistent

---

## Types of Filtering Needed

### 1. Class Filtering (Already Done)
**What it does**: Only keep detections of classes you care about

**Current implementation**:
```python
# In ByteTracker, you already filter by class:
COCO_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
# Only process these classes
```

**Status**: ✅ **Already working**

---

### 2. Confidence Filtering (May Need Tuning)
**What it does**: Only keep detections above a confidence threshold

**Current implementation**:
```python
# In YOLODetector, you set confidence threshold:
detector = YOLODetector(confidence_threshold=0.4)
```

**What to check**:
- Are you getting too many false positives? → Increase threshold
- Missing real detections? → Decrease threshold
- Current threshold (0.4) is reasonable, but may need tuning per scene

**Status**: ✅ **Working, may need tuning**

---

### 3. Spatial Filtering (NOT YET IMPLEMENTED) ⚠️
**What it does**: Remove pedestrian detections that are inside vehicle bboxes

**Why needed**: 
- People sitting in cars get detected as "person"
- These are NOT pedestrians on the road
- They cause false positives in near miss detection

**How to implement**:
```python
def filter_pedestrians_in_vehicles(pedestrians, vehicles):
    """
    Remove pedestrian detections that are inside vehicle bboxes.
    
    Logic:
    - For each pedestrian, check if it overlaps significantly with any vehicle
    - If overlap > threshold (e.g., 70%), remove the pedestrian
    - This prevents tracking people inside cars
    """
    filtered_peds = []
    
    for ped in pedestrians:
        is_inside_vehicle = False
        
        for vehicle in vehicles:
            # Calculate overlap ratio
            overlap = calculate_bbox_overlap(ped.bbox, vehicle.bbox)
            
            # If pedestrian is mostly inside vehicle bbox
            if overlap > 0.7:  # 70% overlap threshold
                is_inside_vehicle = True
                break
        
        # Only keep pedestrians NOT inside vehicles
        if not is_inside_vehicle:
            filtered_peds.append(ped)
    
    return filtered_peds
```

**Status**: ❌ **Needs to be implemented**

---

### 4. Size Filtering (May Need Addition)
**What it does**: Remove very small detections (likely false positives or too far away)

**Why needed**:
- Very small bboxes (e.g., < 20x20 pixels) are often false positives
- Or they're so far away they're not useful for near miss detection
- Reduces noise in the system

**How to implement**:
```python
def filter_by_size(detections, min_width=20, min_height=20):
    """Remove detections smaller than minimum size."""
    filtered = []
    for det in detections:
        width = det.bbox[2] - det.bbox[0]
        height = det.bbox[3] - det.bbox[1]
        
        if width >= min_width and height >= min_height:
            filtered.append(det)
    
    return filtered
```

**Status**: ❓ **May be useful, but not critical**

---

### 5. Track Quality Filtering (May Need Addition)
**What it does**: Only keep tracks that are stable and consistent

**Why needed**:
- Some tracks are noisy (ID switches, false detections)
- Unstable tracks give bad risk scores
- Better to filter out low-quality tracks

**How to check**:
- Track length (how many frames the track exists)
- Track consistency (does bbox size change dramatically?)
- Detection confidence over time

**Status**: ❓ **Nice to have, but not critical for MVP**

---

## What "Properly Filtered" Means for Your System

### Minimum Requirements (Must Have)
1. ✅ **Class filtering** - Only pedestrians and vehicles (already done)
2. ✅ **Confidence filtering** - Remove low-confidence detections (already done)
3. ❌ **Spatial filtering** - Remove people inside vehicles (needs implementation)

### Nice to Have (Can Add Later)
4. ❓ **Size filtering** - Remove very small detections
5. ❓ **Track quality filtering** - Only keep stable tracks

---

## Recommended Implementation Order

### Phase 1: Critical Filtering (Do First)
1. **Implement spatial filtering** - Remove people inside vehicles
   - This is the most important missing piece
   - Prevents false positives in near miss detection
   - Should be done before risk scoring

### Phase 2: Optional Filtering (Can Add Later)
2. **Add size filtering** - If you see many small false positives
3. **Add track quality filtering** - If tracks are unstable

---

## Summary

**"Properly filtered" means**:
- ✅ Detecting only relevant classes (pedestrians, vehicles)
- ✅ Using appropriate confidence thresholds
- ❌ **Removing people inside vehicles** (needs implementation)
- ❓ Optionally filtering by size and track quality

**Current Status**: Your system is **mostly filtered**, but **spatial filtering** (removing people in vehicles) is the key missing piece.

---

## Next Steps

1. **Verify current filtering**: Check if class and confidence filtering are working
2. **Implement spatial filtering**: Add code to remove people inside vehicles
3. **Test**: Verify filtering works correctly on your videos
4. **Tune**: Adjust thresholds based on results

---

## References

- Bbox Overlap Calculation: IOU (Intersection over Union) for detecting overlaps
- False Positive Reduction: Techniques for improving detection quality
- Spatial Reasoning: Understanding object relationships in images

