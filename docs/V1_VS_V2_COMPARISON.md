# v1 vs v2: Image-Space vs Ground Plane Approach

This document explains the difference between v1 (image-space) and v2 (ground plane) approaches for near miss detection.

---

## v1: Image-Space Approach (Simpler, Faster)

### What It Means
- All calculations done in **pixel coordinates** (2D image space)
- No camera calibration needed
- No ground plane estimation required
- Works immediately with any camera setup

### How It Works
- Measure distances in **pixels** (not meters)
- Calculate speeds in **pixels per frame** (not meters per second)
- Risk scoring based on **relative** distances and speeds
- Time-to-contact calculated in **frames** (not seconds)

### Example Calculations
```python
# v1: Image-space calculations
pedestrian_centroid = (320, 480)  # pixels
vehicle_centroid = (640, 360)     # pixels
distance_pixels = sqrt((640-320)^2 + (360-480)^2) = 360 pixels

# Speed in pixels per frame
speed_pixels_per_frame = distance_change / frame_difference

# Time-to-contact in frames
ttc_frames = distance_pixels / closing_speed_pixels_per_frame
```

### Advantages
- ✅ **Fast to implement** (1-2 weeks)
- ✅ **Works immediately** - no setup required
- ✅ **No calibration needed** - works with any camera
- ✅ **Good for relative risk scoring** - can compare scenarios
- ✅ **Less complex** - easier to debug and maintain

### Disadvantages
- ❌ **Less accurate** - distances are in pixels, not real-world
- ❌ **Can't calculate real speeds** - only relative speeds
- ❌ **Depends on camera angle** - same distance looks different from different angles
- ❌ **Harder to set thresholds** - "50 pixels" means different things in different scenes

### Use Cases
- Quick prototype/MVP
- Relative risk comparison (which scenario is riskier?)
- Scenes where absolute accuracy isn't critical
- When camera calibration isn't possible

---

## v2: Ground Plane Approach (More Accurate, More Complex)

### What It Means
- Transform image coordinates → **real-world coordinates** (meters)
- Requires **ground plane calibration** (homography matrix)
- All calculations done in **3D world space**
- Can calculate real distances, speeds, and times

### How It Works
1. **Estimate ground plane**: Calculate homography matrix (see GROUND_PLANE_METHODS.md)
2. **Transform points**: Convert image coordinates → ground plane coordinates
3. **Calculate in real-world**: Measure distances in **meters**, speeds in **m/s**
4. **Real TTC**: Time-to-contact in **seconds** (not frames)

### Example Calculations
```python
# v2: Ground plane calculations
# Step 1: Transform to ground plane
pedestrian_world = homography.transform(pedestrian_image_point)  # (2.5m, 0m)
vehicle_world = homography.transform(vehicle_image_point)        # (5.0m, 0m)

# Step 2: Calculate real distance
distance_meters = sqrt((5.0-2.5)^2 + (0-0)^2) = 2.5 meters

# Step 3: Calculate real speed (m/s)
speed_mps = distance_change_meters / time_seconds

# Step 4: Real time-to-contact (seconds)
ttc_seconds = distance_meters / closing_speed_mps
```

### Advantages
- ✅ **More accurate** - real-world measurements
- ✅ **Real speeds** - can calculate actual vehicle speeds (km/h, mph)
- ✅ **Real distances** - know actual separation distance
- ✅ **Real TTC** - time-to-contact in seconds (more meaningful)
- ✅ **Better thresholds** - can use real-world values (e.g., "2 meters" is always 2 meters)
- ✅ **More professional** - suitable for production systems

### Disadvantages
- ❌ **More complex** - requires ground plane estimation
- ❌ **Takes longer** - 2-3 weeks for robust implementation
- ❌ **Requires calibration** - needs setup per camera
- ❌ **May need recalibration** - if camera moves (PTZ cameras)
- ❌ **More error-prone** - calibration errors affect all calculations

### Use Cases
- Production systems requiring accuracy
- When real-world measurements are needed
- Systems that need to report actual speeds/distances
- When thresholds need to be consistent across cameras

---

## Comparison Table

| Feature | v1 (Image-Space) | v2 (Ground Plane) |
|---------|------------------|-------------------|
| **Implementation Time** | 1-2 weeks | 2-3 weeks |
| **Setup Required** | None | Ground plane calibration |
| **Accuracy** | Medium (relative) | High (absolute) |
| **Distance Units** | Pixels | Meters |
| **Speed Units** | Pixels/frame | m/s, km/h |
| **TTC Units** | Frames | Seconds |
| **Calibration Needed** | No | Yes |
| **Complexity** | Low | Medium-High |
| **Best For** | MVP, prototypes | Production systems |

---

## Is v2 a Lengthy Process?

### Short Answer
**Yes, but manageable if broken into phases.**

### Detailed Breakdown

#### Phase 1: Ground Plane Estimation (1-2 weeks)
- Detect road markings/lines
- Calculate vanishing points
- Estimate homography matrix
- Test and validate

#### Phase 2: Coordinate Transformation (3-5 days)
- Implement point transformation (image → ground plane)
- Transform all detections/tracks
- Validate transformations

#### Phase 3: Update Risk Scoring (1 week)
- Rewrite risk scoring to use real-world coordinates
- Update all calculations (distance, speed, TTC)
- Test with sample data

#### Phase 4: Testing & Refinement (1 week)
- Test on multiple videos
- Refine thresholds
- Handle edge cases

**Total Time: 3-4 weeks** (if working full-time)

### Can It Be Done Faster?
- **Yes, if**: You use manual point selection (Method 3) instead of automatic vanishing points
- **Yes, if**: You start with simple homography and refine later
- **Yes, if**: You focus on MVP first, then add accuracy improvements

---

## Recommendation

### For Your Project
**Start with v2 (Ground Plane)** because:
1. You want real-world accuracy
2. You have time to implement properly
3. It's more professional for production use
4. You can break it into phases

### Phased Approach (Recommended)
1. **Week 1**: Implement ground plane estimation (vanishing points)
2. **Week 2**: Implement coordinate transformation + basic risk scoring
3. **Week 3**: Refine and test
4. **Week 4**: Polish and handle edge cases

This way you have a working system after 2 weeks, then refine for 2 more weeks.

---

## Migration Path (v1 → v2)

If you start with v1 and want to upgrade later:

1. Keep v1 code as fallback
2. Add ground plane estimation module
3. Add coordinate transformation layer
4. Update risk scoring to use transformed coordinates
5. Test both v1 and v2, compare results
6. Switch to v2 when confident

This allows you to have a working system while building v2.

---

## References

- Image-Space vs World-Space: Understanding coordinate systems in computer vision
- Homography Transformation: Converting between image and world coordinates
- Ground Plane Estimation: Methods for finding the ground plane in images

