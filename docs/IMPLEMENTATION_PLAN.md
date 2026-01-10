# Near Miss Detection Implementation Plan

This document outlines the structured plan for implementing near miss detection using v2 (ground plane) approach.

---

## Phase 1: Foundation & Filtering (Week 1)

### 1.1 Verify Current Detection/Tracking
- [ ] Verify pedestrian detection is working (bboxes created)
- [ ] Verify pedestrian tracking is working (track IDs assigned)
- [ ] Test on sample video to confirm detection quality

### 1.2 Implement Spatial Filtering
- [ ] Create function to filter people inside vehicles
- [ ] Calculate bbox overlap ratios
- [ ] Remove pedestrian detections with >70% overlap with vehicles
- [ ] Test filtering on sample video

### 1.3 Separate Track Lists
- [ ] Create separate lists for pedestrian tracks vs vehicle tracks
- [ ] Add helper functions to filter tracks by class
- [ ] Ensure track IDs are maintained correctly

**Deliverable**: Clean, filtered detections with separate pedestrian/vehicle tracks

---

## Phase 2: Ground Plane Estimation (Week 1-2)

### 2.1 Road Marking Detection
- [ ] Implement line detection (Canny edge detection + Hough Transform)
- [ ] Detect lane markings and road edges
- [ ] Filter and group parallel lines
- [ ] Test on sample video

### 2.2 Vanishing Point Calculation
- [ ] Calculate vanishing points from parallel lines
- [ ] Find horizon line from vanishing points
- [ ] Validate vanishing point detection
- [ ] Handle edge cases (no clear lines, etc.)

### 2.3 Homography Estimation
- [ ] Select 4+ points on ground plane (from road markings or manual)
- [ ] Estimate/measure real-world distances
- [ ] Calculate homography matrix using cv2.findHomography()
- [ ] Validate homography by transforming test points

**Deliverable**: Working ground plane calibration with homography matrix

---

## Phase 3: Coordinate Transformation (Week 2)

### 3.1 Transform Detections
- [ ] Transform pedestrian centroids to ground plane coordinates
- [ ] Transform vehicle centroids to ground plane coordinates
- [ ] Store both image and world coordinates
- [ ] Test transformations

### 3.2 Transform Tracks
- [ ] Transform track histories to ground plane
- [ ] Calculate velocities in world space (m/s)
- [ ] Store transformed trajectories
- [ ] Validate track transformations

**Deliverable**: All detections and tracks transformed to ground plane coordinates

---

## Phase 4: Physics-Based Risk Scoring (Week 2-3)

### 4.1 Calculate Approach Rate
- [ ] Calculate distance between pedestrian and vehicle (in meters)
- [ ] Track distance over time
- [ ] Calculate closing speed (m/s)
- [ ] Compute approach rate metric

### 4.2 Calculate Bbox Growth Rate
- [ ] Track vehicle bbox size over time
- [ ] Calculate expansion rate toward pedestrian
- [ ] Use as proxy for approach

### 4.3 Calculate Time-to-Contact (TTC)
- [ ] Calculate TTC in real seconds (not frames)
- [ ] TTC = distance / closing_speed
- [ ] Handle edge cases (stationary objects, etc.)

### 4.4 Calculate Minimum Predicted Separation
- [ ] Predict trajectories using linear extrapolation
- [ ] Find minimum distance over prediction horizon
- [ ] Use for risk assessment

### 4.5 Compute Risk Tiers
- [ ] Define risk tiers (low/medium/high/critical)
- [ ] Use TTC and minimum separation as inputs
- [ ] Assign risk tier to each pedestrian-vehicle pair
- [ ] Test risk scoring on sample scenarios

**Deliverable**: Working risk scoring system with real-world metrics

---

## Phase 5: Near Miss Detection (Week 3)

### 5.1 Detect Near Miss Events
- [ ] Define near miss criteria (risk tier + thresholds)
- [ ] Detect when risk exceeds threshold
- [ ] Create near miss event objects
- [ ] Store event metadata (timestamp, IDs, risk score)

### 5.2 Store Event Windows
- [ ] Capture frames before near miss (e.g., 30 frames)
- [ ] Capture frames during near miss
- [ ] Capture frames after near miss (e.g., 30 frames)
- [ ] Store bboxes for all frames in window

### 5.3 Evidence Collection
- [ ] Save annotated frames with bboxes
- [ ] Store detection/tracking data for event window
- [ ] Create event summary with key metrics

**Deliverable**: Near miss detection system with evidence collection

---

## Phase 6: Impact Detection (Week 3-4)

### 6.1 Detect Impact Candidates
- [ ] Detect when pedestrian and vehicle bboxes overlap
- [ ] Detect when tracks are within threshold distance
- [ ] Create candidate event windows (±1 second)

### 6.2 Analyze Impact Signals
- [ ] Velocity discontinuity (sudden pedestrian velocity change)
- [ ] Fall-like motion (bbox aspect ratio + downward motion)
- [ ] Track disappearance (pedestrian lost near vehicle)
- [ ] Vehicle deceleration (sudden slowdown)

### 6.3 Generate Impact Events
- [ ] Combine signals to compute impact confidence
- [ ] Create impact event objects
- [ ] Store impact metadata
- [ ] Link to associated tracks

**Deliverable**: Impact detection system with confidence scores

---

## Phase 7: Integration & Testing (Week 4)

### 7.1 Integrate All Components
- [ ] Connect filtering → ground plane → risk scoring → near miss → impact
- [ ] Ensure data flows correctly through pipeline
- [ ] Handle edge cases and errors

### 7.2 Testing
- [ ] Test on sample video
- [ ] Validate ground plane calibration
- [ ] Check risk scoring accuracy
- [ ] Verify near miss detection
- [ ] Test impact detection

### 7.3 Refinement
- [ ] Tune thresholds based on test results
- [ ] Improve filtering if needed
- [ ] Optimize performance
- [ ] Add error handling

**Deliverable**: Complete, tested near miss detection system

---

## Phase 8: Advanced Features (Week 4+)

### 8.1 LPR on Impact Only
- [ ] Gate existing LPR by impact events
- [ ] Only extract plates when impact detected
- [ ] Store plate results with impact events

### 8.2 VLM Escalation (Optional)
- [ ] Integrate VLM for high-risk events
- [ ] Add rate limiting and caching
- [ ] Implement fallback to physics-only

**Deliverable**: Complete system with all features

---

## File Structure

```
src/
├── filter/
│   ├── __init__.py
│   └── spatial_filter.py          # Filter people in vehicles
├── ground_plane/
│   ├── __init__.py
│   ├── line_detector.py            # Detect road markings
│   ├── vanishing_point.py          # Calculate vanishing points
│   └── homography.py               # Estimate homography matrix
├── transform/
│   ├── __init__.py
│   └── coordinate_transform.py     # Transform to ground plane
├── risk/
│   ├── __init__.py
│   ├── risk_scorer.py              # Calculate risk metrics
│   └── risk_types.py                # Risk data types
├── near_miss/
│   ├── __init__.py
│   ├── detector.py                  # Detect near miss events
│   └── event_types.py               # Event data types
└── impact/
    ├── __init__.py
    ├── detector.py                  # Detect impact events
    └── impact_types.py               # Impact data types
```

---

## Testing Strategy

### Unit Tests
- Test each component independently
- Test filtering functions
- Test coordinate transformations
- Test risk calculations

### Integration Tests
- Test full pipeline on sample videos
- Validate end-to-end flow
- Check data consistency

### Validation
- Compare results with manual annotations
- Validate ground plane accuracy
- Check risk scoring makes sense

---

## Success Criteria

- [ ] Ground plane calibration works on sample video
- [ ] Risk scoring produces reasonable scores
- [ ] Near miss events detected correctly
- [ ] Impact events detected with good accuracy
- [ ] System runs in real-time (or acceptable latency)
- [ ] Code is well-documented and maintainable

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Phase 1 | Week 1 | Filtered detections |
| Phase 2 | Week 1-2 | Ground plane calibration |
| Phase 3 | Week 2 | Coordinate transformation |
| Phase 4 | Week 2-3 | Risk scoring |
| Phase 5 | Week 3 | Near miss detection |
| Phase 6 | Week 3-4 | Impact detection |
| Phase 7 | Week 4 | Integration & testing |
| Phase 8 | Week 4+ | Advanced features |

**Total: 3-4 weeks** for complete implementation

---

## Notes

- Start with MVP (Phases 1-5) for working system
- Add impact detection (Phase 6) for complete system
- Advanced features (Phase 8) can be added incrementally
- Test frequently on real video data
- Iterate based on results

