# Using Three Points on a Car for Ground Plane Estimation

This document explains how to use points on a vehicle to help estimate the ground plane.

---

## The Concept

### What You're Trying to Do
Use points on a detected vehicle to help establish the ground plane. The idea is that vehicles are on the ground, so their contact points with the ground can help define the plane.

### The Challenge
- **3 points define a plane** in 3D space (mathematically correct)
- **But homography needs 4+ correspondences** (image points ↔ world points)
- So 3 points alone aren't enough for homography calculation

---

## Option A: Use 4 Points (Wheel Contact Points) ⭐ RECOMMENDED

### How It Works
- Detect a vehicle in the image
- Identify 4 points where wheels touch the ground
- These 4 points are on the ground plane
- Use these 4 points + known/estimated distances → calculate homography

### Steps
1. **Detect vehicle**: Use YOLO to get vehicle bbox
2. **Estimate wheel positions**: 
   - Front wheels: bottom-left and bottom-right of bbox (approximately)
   - Rear wheels: slightly forward from bottom corners
3. **Get ground contact points**: Points where wheels touch ground (bottom of bbox)
4. **Estimate real-world positions**: 
   - Assume standard car width (1.8m) → distance between left/right wheels
   - Estimate car length from bbox aspect ratio
5. **Calculate homography**: Use 4 image points ↔ 4 world points

### Advantages
- Uses detected vehicles (already have them)
- 4 points = enough for homography
- Can be automated

### Disadvantages
- Wheel positions are estimates (not exact)
- Assumes standard car dimensions
- Less accurate than using road markings

### Implementation
```python
def get_wheel_points(vehicle_bbox):
    """Estimate 4 wheel contact points from vehicle bbox."""
    x1, y1, x2, y2 = vehicle_bbox
    
    # Front wheels (bottom of bbox, slightly inset)
    front_left = (x1 + (x2-x1)*0.2, y2)   # 20% from left
    front_right = (x1 + (x2-x1)*0.8, y2)  # 80% from left
    
    # Rear wheels (slightly forward from bottom)
    rear_left = (x1 + (x2-x1)*0.2, y2 - (y2-y1)*0.1)  # 10% up from bottom
    rear_right = (x1 + (x2-x1)*0.8, y2 - (y2-y1)*0.1)
    
    return [front_left, front_right, rear_left, rear_right]
```

---

## Option B: Use 3 Points + Constraint

### How It Works
- Use 3 points on the vehicle
- Add constraint: "vehicle is on the ground plane"
- This constraint provides the 4th piece of information needed

### Steps
1. **Select 3 points**: Could be:
   - 3 corners of vehicle bbox bottom edge
   - 3 wheel positions (if only 3 visible)
   - Any 3 points on vehicle that touch ground
2. **Add constraint**: Assume vehicle bottom is on ground plane
3. **Calculate homography**: Use 3 points + constraint → solve for homography

### Advantages
- Works with 3 points
- Uses detected vehicles

### Disadvantages
- More complex math (constrained optimization)
- Less accurate than 4 points
- Constraint assumption may not always be true (hills, ramps)

### Implementation Complexity
- **High**: Requires constrained optimization
- **Not recommended** unless you have specific reason

---

## Option C: Use Road Markings Instead (SIMPLER EXPLANATION)

### Simple Explanation
Instead of using points on the car, use points on the **road itself**:
- Road markings (lane lines, crosswalks)
- Road edges
- Any features on the ground

### Why This Is Better
- ✅ More accurate (road markings are on the ground)
- ✅ Easier to detect (use line detection)
- ✅ More stable (don't depend on vehicle detection)
- ✅ Standard approach in computer vision

### How It Works (Simple)
1. **Find road markings**: Detect lines on the road (like lane markings)
2. **Pick 4 points**: Choose 4 corners/intersections of road markings
3. **Estimate distances**: Guess or measure distances between points
4. **Calculate homography**: Use 4 image points ↔ 4 world points

### Example
```
Image:                    Real World:
[Road marking corners] →  [Known/estimated distances]

Point 1 (image) → (0m, 0m)   in world
Point 2 (image) → (5m, 0m)    in world  
Point 3 (image) → (0m, 10m)   in world
Point 4 (image) → (5m, 10m)   in world
```

### Why This Is Recommended
- **Standard approach**: Used in most computer vision systems
- **More reliable**: Road markings are designed to be visible
- **Easier to implement**: Can use existing line detection algorithms
- **Works better**: More accurate than using vehicle points

---

## Comparison

| Method | Points Needed | Accuracy | Complexity | Best For |
|--------|---------------|----------|------------|----------|
| 4 Wheel Points | 4 | Medium | Medium | Scenes with clear vehicles |
| 3 Points + Constraint | 3 | Low-Medium | High | Special cases only |
| Road Markings | 4+ | High | Low-Medium | Most scenes ⭐ |

---

## Recommendation

**Don't use 3 car points** - instead:

1. **Use road markings** (Option C) - more accurate and standard
2. **OR use 4 wheel points** (Option A) - if road markings aren't visible
3. **Avoid 3 points** (Option B) - too complex for little benefit

---

## Practical Approach

### Step 1: Try Road Markings First
- Detect lane lines, road edges, crosswalks
- Use 4+ intersection points
- Calculate homography

### Step 2: Fallback to Vehicle Points
- If road markings aren't clear
- Use 4 wheel contact points from vehicles
- Estimate distances from vehicle dimensions

### Step 3: Manual Selection (Last Resort)
- If automatic methods fail
- Let user manually select 4 points on ground
- User provides/estimates real-world coordinates

---

## References

- Homography Calculation: Requires minimum 4 point correspondences
- Vehicle Dimension Estimation: Using known vehicle sizes for calibration
- Road Marking Detection: Computer vision techniques for detecting road features

