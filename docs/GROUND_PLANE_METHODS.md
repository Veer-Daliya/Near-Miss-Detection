# Ground Plane Estimation Methods

This document explains different methods for estimating the ground plane without knowing camera height/orientation.

---

## Method 1: Vanishing Points (Parallel Lines) ⭐ RECOMMENDED FOR V1

### How It Works
- Detect parallel lines in the image (lane markings, road edges, building edges)
- Find where these parallel lines converge → this is the vanishing point
- The vanishing point gives you the horizon line
- Combine horizon line + known distances → estimate homography matrix

### Steps
1. **Detect lines**: Use edge detection (Canny) + line detection (Hough Transform)
2. **Group parallel lines**: Cluster lines that have similar angles
3. **Find vanishing point**: Calculate intersection of parallel lines
4. **Estimate horizon**: Use vanishing point to find horizon line
5. **Calculate homography**: Use horizon + assumptions about road width/distance

### Advantages
- Works automatically from video frames
- No manual point selection needed
- Works well with clear road markings

### Disadvantages
- Requires visible parallel lines (may not work in all scenes)
- Less accurate if lines are not perfectly parallel
- Needs some assumptions about road dimensions

### Implementation Complexity
- **Medium**: Requires line detection, clustering, and geometric calculations
- **Time**: 2-3 days for robust implementation

---

## Method 2: Known Object Sizes

### How It Works
- Use known real-world dimensions (e.g., average car width = 1.8 meters)
- Measure pixel size of objects in image
- Calculate scale factor (pixels per meter)
- Combine with vanishing points → full homography

### Steps
1. **Detect vehicles**: Use YOLO to detect cars
2. **Measure pixel width**: Calculate bbox width in pixels
3. **Estimate real width**: Assume average car width (1.8m) or use vehicle type
4. **Calculate scale**: pixels_per_meter = pixel_width / real_width
5. **Combine with vanishing point**: Use scale + vanishing point → homography

### Advantages
- Uses objects already detected (cars)
- Can be automated
- Works even without clear road markings

### Disadvantages
- Assumes standard car sizes (may vary)
- Less accurate for distant objects
- Needs vehicle classification to know size

### Implementation Complexity
- **Medium**: Requires vehicle detection + size estimation
- **Time**: 1-2 days

---

## Method 3: Manual Point Selection

### How It Works
- User manually selects 4+ points on the ground in the image
- User provides corresponding real-world coordinates (or estimates)
- Calculate homography matrix from point correspondences

### Steps
1. **Display frame**: Show user a video frame
2. **Select points**: User clicks 4+ points on ground (e.g., corners of road markings)
3. **Provide coordinates**: User enters real-world coordinates OR estimates distances
4. **Calculate homography**: Use cv2.findHomography() with point pairs
5. **Validate**: Test homography by transforming known points

### Advantages
- Most accurate if done correctly
- Works in any scene (doesn't need parallel lines)
- User has full control

### Disadvantages
- Requires manual intervention
- Needs user to know/estimate real-world coordinates
- Must be redone if camera moves

### Implementation Complexity
- **Low**: Simple point selection + homography calculation
- **Time**: 1 day

---

## Method 4: Auto-Calibration from Traffic

### How It Works
- Track vehicles moving on the ground plane
- Use motion vectors + assumptions → estimate plane parameters
- Vehicles moving in straight lines → gives direction vectors
- Combine multiple vehicle tracks → estimate ground plane

### Steps
1. **Track vehicles**: Use ByteTrack to get vehicle trajectories
2. **Filter straight motion**: Find vehicles moving in straight lines
3. **Extract motion vectors**: Get direction of vehicle movement
4. **Estimate plane**: Use motion vectors + assumptions → ground plane
5. **Refine**: Use multiple tracks to improve accuracy

### Advantages
- Fully automatic
- Uses existing tracking data
- Adapts to camera angle automatically

### Disadvantages
- Requires vehicles moving in scene
- Less accurate than other methods
- Complex to implement correctly

### Implementation Complexity
- **High**: Requires advanced geometric calculations
- **Time**: 3-5 days

---

## Comparison Table

| Method | Accuracy | Automation | Complexity | Best For |
|--------|----------|------------|------------|----------|
| Vanishing Points | High | Automatic | Medium | Scenes with road markings |
| Known Object Sizes | Medium | Automatic | Medium | Scenes with vehicles |
| Manual Selection | Very High | Manual | Low | Any scene, one-time setup |
| Auto-Calibration | Medium | Automatic | High | Scenes with moving vehicles |

---

## Recommendation

**For v1 (MVP)**: Use **Method 1 (Vanishing Points)** - best balance of accuracy and automation

**For v2 (Future)**: Consider **Method 3 (Manual Selection)** for highest accuracy, or combine methods

---

## References

- OpenCV Homography: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
- Vanishing Point Detection: Computer Vision algorithms for detecting vanishing points
- Camera Calibration: Understanding camera parameters and coordinate transformations

