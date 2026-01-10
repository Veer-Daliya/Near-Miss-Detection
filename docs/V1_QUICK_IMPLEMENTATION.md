# v1 Quick Implementation Plan (< 1 Week)

This document outlines a realistic plan for implementing v1 (image-space) near miss detection in less than a week.

---

## Important Clarification: v1 vs Vanishing Points

### v1 (Image-Space) = NO Vanishing Points, NO Homography

**v1 approach**:
- All calculations in **pixel coordinates**
- **No ground plane** needed
- **No homography** needed
- **No vanishing points** needed
- Works immediately with any camera

**If you want vanishing points**:
- That's for **v2 (ground plane)** approach
- Vanishing points are used to **estimate homography**
- Requires ground plane calibration
- Takes longer to implement

### The Three Point System

**If you're doing v1 (no homography)**:
- ❌ **Don't need 3 points** - that was for homography estimation
- ❌ **Don't need vanishing points** - that's for ground plane
- ✅ **Just use pixel coordinates directly**

**If you're doing v2 (with homography)**:
- ✅ **Need 4+ points** (not 3) for homography
- ✅ **Use vanishing points** to help estimate ground plane
- ✅ **Transform coordinates** to world space

---

## Realistic Timeline: < 1 Week

### Is It Possible?
**Yes, but only for MVP (Minimum Viable Product)** - a basic working system, not polished.

### What You Can Achieve in < 1 Week

**Day 1-2**: Filtering & Basic Risk Scoring
- Filter people inside vehicles
- Calculate basic risk metrics (distance, approach rate)
- Simple risk tiers

**Day 3-4**: Near Miss Detection
- Detect near miss events
- Store event windows
- Basic evidence collection

**Day 5**: Testing & Refinement
- Test on your video
- Fix bugs
- Tune thresholds

**Day 6-7**: Polish (if time)
- Improve visualization
- Add more metrics
- Documentation

---

## v1 Implementation Plan (No Vanishing Points)

### Phase 1: Filtering (Day 1 - 4 hours)

**Goal**: Remove people inside vehicles

**Tasks**:
1. Create `src/filter/spatial_filter.py`
2. Implement bbox overlap calculation
3. Filter pedestrians with >70% overlap with vehicles
4. Test on sample video

**Code Structure**:
```python
def filter_pedestrians_in_vehicles(pedestrians, vehicles):
    """Remove pedestrian detections inside vehicle bboxes."""
    filtered = []
    for ped in pedestrians:
        is_inside = False
        for vehicle in vehicles:
            overlap = calculate_overlap_ratio(ped.bbox, vehicle.bbox)
            if overlap > 0.7:
                is_inside = True
                break
        if not is_inside:
            filtered.append(ped)
    return filtered
```

---

### Phase 2: Basic Risk Scoring (Day 1-2 - 8 hours)

**Goal**: Calculate risk metrics in image-space (pixels)

**Tasks**:
1. Create `src/risk/risk_scorer.py`
2. Calculate distance between pedestrian and vehicle (pixels)
3. Calculate approach rate (pixels per frame)
4. Calculate time-to-contact proxy (frames)
5. Assign risk tiers

**Metrics to Calculate**:
- **Distance**: `sqrt((cx_ped - cx_veh)^2 + (cy_ped - cy_veh)^2)` in pixels
- **Approach Rate**: Change in distance per frame (pixels/frame)
- **TTC Proxy**: `distance / approach_rate` in frames
- **Risk Tier**: Based on TTC and distance thresholds

**Code Structure**:
```python
def calculate_risk_score(pedestrian_track, vehicle_track):
    """Calculate risk score in image-space."""
    # Get current positions
    ped_center = get_centroid(pedestrian_track.current_bbox)
    veh_center = get_centroid(vehicle_track.current_bbox)
    
    # Calculate distance (pixels)
    distance = calculate_distance(ped_center, veh_center)
    
    # Calculate approach rate (pixels per frame)
    prev_distance = calculate_previous_distance(pedestrian_track, vehicle_track)
    approach_rate = prev_distance - distance  # positive = getting closer
    
    # Calculate TTC proxy (frames)
    if approach_rate > 0:
        ttc_frames = distance / approach_rate
    else:
        ttc_frames = float('inf')  # Moving apart
    
    # Assign risk tier
    risk_tier = assign_risk_tier(ttc_frames, distance)
    
    return RiskScore(distance, approach_rate, ttc_frames, risk_tier)
```

---

### Phase 3: Near Miss Detection (Day 2-3 - 8 hours)

**Goal**: Detect when risk exceeds threshold

**Tasks**:
1. Create `src/near_miss/detector.py`
2. Monitor risk scores for all pedestrian-vehicle pairs
3. Detect when risk tier becomes "high" or "critical"
4. Create near miss event
5. Store event window (frames before/during/after)

**Code Structure**:
```python
def detect_near_miss(pedestrian_track, vehicle_track, risk_score):
    """Detect near miss event."""
    if risk_score.tier in ["high", "critical"]:
        event = NearMissEvent(
            pedestrian_id=pedestrian_track.track_id,
            vehicle_id=vehicle_track.track_id,
            timestamp=current_timestamp,
            risk_score=risk_score,
            frame_id=current_frame_id
        )
        return event
    return None
```

---

### Phase 4: Track Management (Day 3 - 4 hours)

**Goal**: Maintain track histories for risk calculation

**Tasks**:
1. Store track positions over time
2. Calculate velocities from position history
3. Maintain separate lists for pedestrians vs vehicles
4. Handle track IDs and continuity

**Code Structure**:
```python
class TrackHistory:
    """Store track position history."""
    def __init__(self, track_id):
        self.track_id = track_id
        self.positions = []  # List of (frame_id, bbox, centroid)
        self.velocities = []  # List of velocities
    
    def add_position(self, frame_id, bbox, centroid):
        self.positions.append((frame_id, bbox, centroid))
        if len(self.positions) > 1:
            velocity = self._calculate_velocity()
            self.velocities.append(velocity)
```

---

### Phase 5: Integration (Day 4 - 6 hours)

**Goal**: Integrate all components into main pipeline

**Tasks**:
1. Update `scripts/run_detection.py` to include filtering
2. Add risk scoring to main loop
3. Add near miss detection
4. Store results (JSON output)
5. Add visualization (optional)

**Pipeline Flow**:
```
Frame → Detection → Tracking → Filtering → Risk Scoring → Near Miss Detection → Output
```

---

### Phase 6: Testing & Tuning (Day 5 - 4 hours)

**Goal**: Test on your video and tune thresholds

**Tasks**:
1. Run on your video
2. Check if detections are correct
3. Verify filtering works
4. Tune risk thresholds
5. Fix bugs

---

## What You WON'T Have Time For (< 1 Week)

❌ **Impact Detection** - Too complex, save for later  
❌ **Ground Plane** - Not needed for v1  
❌ **Vanishing Points** - Not needed for v1  
❌ **Homography** - Not needed for v1  
❌ **Real-world Metrics** - Will use pixel-based metrics  
❌ **Polished UI** - Basic visualization only  
❌ **Comprehensive Testing** - Basic testing only  

---

## Minimal Viable Product (MVP) Features

✅ **Filter people inside vehicles**  
✅ **Calculate risk scores** (distance, approach rate, TTC proxy)  
✅ **Detect near miss events**  
✅ **Store event data** (JSON output)  
✅ **Basic visualization** (optional)  

---

## File Structure (Minimal)

```
src/
├── filter/
│   ├── __init__.py
│   └── spatial_filter.py          # Filter people in vehicles
├── risk/
│   ├── __init__.py
│   ├── risk_scorer.py              # Calculate risk metrics
│   └── risk_types.py                # Risk data types
├── near_miss/
│   ├── __init__.py
│   ├── detector.py                  # Detect near miss events
│   └── event_types.py               # Event data types
└── track/
    └── track_history.py             # Track position history (add to existing)
```

---

## Risk Thresholds (Starting Points - Will Need Tuning)

```python
RISK_THRESHOLDS = {
    "critical": {
        "ttc_frames": 10,      # < 10 frames until contact
        "distance_pixels": 50  # < 50 pixels apart
    },
    "high": {
        "ttc_frames": 30,
        "distance_pixels": 100
    },
    "medium": {
        "ttc_frames": 60,
        "distance_pixels": 200
    },
    "low": {
        "ttc_frames": float('inf'),
        "distance_pixels": float('inf')
    }
}
```

**Note**: These are starting points - you'll need to tune based on your video!

---

## Daily Breakdown

### Day 1 (8 hours)
- Morning: Filtering implementation (4 hours)
- Afternoon: Basic risk scoring (4 hours)

### Day 2 (8 hours)
- Morning: Complete risk scoring (4 hours)
- Afternoon: Near miss detection (4 hours)

### Day 3 (8 hours)
- Morning: Track history management (4 hours)
- Afternoon: Integration into pipeline (4 hours)

### Day 4 (8 hours)
- Morning: Complete integration (4 hours)
- Afternoon: Basic testing (4 hours)

### Day 5 (4-8 hours)
- Testing on your video
- Bug fixes
- Threshold tuning

### Day 6-7 (Optional)
- Polish
- More testing
- Documentation

---

## Success Criteria (MVP)

- [ ] System filters people inside vehicles
- [ ] Risk scores calculated for pedestrian-vehicle pairs
- [ ] Near miss events detected when risk is high
- [ ] Results saved to JSON
- [ ] Works on your test video (may need threshold tuning)

---

## Tips for < 1 Week Implementation

1. **Focus on MVP** - Don't try to build everything
2. **Start simple** - Get basic version working first
3. **Test early** - Test on your video as soon as possible
4. **Iterate quickly** - Fix bugs, tune thresholds, repeat
5. **Don't perfect** - Get it working, polish later
6. **Use existing code** - Build on your detection/tracking code

---

## If You Have More Time Later

You can always add:
- Impact detection
- Ground plane (v2)
- Vanishing points
- Better visualization
- More sophisticated risk metrics
- Real-world units

But for < 1 week, focus on MVP!

---

## Next Steps

1. **Decide**: v1 (image-space, no vanishing points) or v2 (ground plane, with vanishing points)?
2. **If v1**: Follow this plan, skip vanishing points
3. **If v2**: Need more time (2-3 weeks minimum)

**Recommendation**: For < 1 week, go with **v1 (no vanishing points)** - it's much faster!

