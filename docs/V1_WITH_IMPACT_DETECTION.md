# v1 Implementation Plan WITH Impact Detection (< 1.5 Weeks)

This document outlines a realistic plan for implementing v1 (image-space) near miss detection **WITH impact detection** in approximately 1-1.5 weeks.

---

## Revised Timeline: Including Impact Detection

### Is < 1 Week Still Possible?
**Barely, but risky**. More realistic: **1-1.5 weeks** (7-10 days) for a working system with impact detection.

### What Changes?
- **Add impact detection** (simplified but functional)
- **Extend timeline** slightly (1-1.5 weeks instead of < 1 week)
- **Focus on MVP** - basic impact detection, not perfect

---

## Simplified Impact Detection Approach

### What We'll Implement (Simplified Version)

Instead of all complex signals, we'll use **simpler, faster methods**:

1. **Overlap Detection** - When pedestrian and vehicle bboxes overlap
2. **Track Loss** - Pedestrian track disappears near vehicle
3. **Proximity + Velocity** - Very close + moving together

**Skip for now** (too complex):
- ❌ Velocity discontinuity (requires complex analysis)
- ❌ Fall-like motion detection (requires pose estimation)
- ❌ Vehicle deceleration (requires speed calculation)

### Why This Works
- ✅ **Faster to implement** (1-2 days vs 3-5 days)
- ✅ **Catches most impacts** (overlap is strong signal)
- ✅ **Good enough for MVP** (can refine later)
- ✅ **Fits in timeline** (adds ~2-3 days)

---

## Revised Daily Breakdown

### Day 1 (8 hours): Filtering & Basic Risk Scoring
- **Morning (4h)**: Filter people inside vehicles
- **Afternoon (4h)**: Basic risk scoring (distance, approach rate, TTC)

### Day 2 (8 hours): Near Miss Detection
- **Morning (4h)**: Complete risk scoring
- **Afternoon (4h)**: Near miss detection + event storage

### Day 3 (8 hours): Impact Detection - Part 1
- **Morning (4h)**: Overlap detection
- **Afternoon (4h)**: Track loss detection

### Day 4 (8 hours): Impact Detection - Part 2
- **Morning (4h)**: Proximity + velocity analysis
- **Afternoon (4h)**: Impact event generation + confidence scoring

### Day 5 (8 hours): Integration
- **Morning (4h)**: Integrate impact detection into pipeline
- **Afternoon (4h)**: Gate LPR by impact events

### Day 6 (6-8 hours): Testing & Tuning
- Test on your video
- Fix bugs
- Tune thresholds
- Validate impact detection

### Day 7 (Optional): Polish
- Improve confidence scoring
- Add more edge cases
- Documentation

**Total: 6-7 days** (1-1.5 weeks)

---

## Impact Detection Implementation Details

### Signal 1: Overlap Detection (Simplest, Strongest)

**How it works**:
- When pedestrian and vehicle bboxes overlap significantly
- Overlap > threshold (e.g., 30-50% IOU)
- Strong indicator of impact

**Implementation**:
```python
def detect_overlap_impact(pedestrian_bbox, vehicle_bbox, threshold=0.3):
    """Detect impact based on bbox overlap."""
    iou = calculate_iou(pedestrian_bbox, vehicle_bbox)
    if iou > threshold:
        return True, iou  # Impact detected, confidence = IOU
    return False, 0.0
```

**Time to implement**: 2-3 hours

---

### Signal 2: Track Loss Detection

**How it works**:
- Pedestrian track disappears while near vehicle
- Track was close to vehicle (< threshold distance)
- Track doesn't reappear within N frames

**Implementation**:
```python
def detect_track_loss_impact(pedestrian_track, vehicle_track, frames_since_loss=5):
    """Detect impact based on track loss."""
    if pedestrian_track.is_lost:
        # Check if was close to vehicle before loss
        last_distance = calculate_distance(
            pedestrian_track.last_position,
            vehicle_track.current_position
        )
        if last_distance < 100:  # pixels threshold
            if pedestrian_track.frames_lost > frames_since_loss:
                return True, 0.7  # Medium confidence
    return False, 0.0
```

**Time to implement**: 3-4 hours

---

### Signal 3: Proximity + Velocity (Simplified)

**How it works**:
- Pedestrian and vehicle very close (< threshold)
- Moving toward each other (approaching)
- High confidence if both conditions met

**Implementation**:
```python
def detect_proximity_velocity_impact(pedestrian_track, vehicle_track):
    """Detect impact based on proximity and velocity."""
    distance = calculate_distance(
        pedestrian_track.current_position,
        vehicle_track.current_position
    )
    
    # Very close
    if distance < 50:  # pixels
        # Check if approaching (simplified - just check if distance decreasing)
        if len(pedestrian_track.positions) > 1 and len(vehicle_track.positions) > 1:
            prev_distance = calculate_previous_distance(pedestrian_track, vehicle_track)
            if distance < prev_distance:  # Getting closer
                return True, 0.6  # Medium-high confidence
    return False, 0.0
```

**Time to implement**: 3-4 hours

---

### Combining Signals

**Confidence Scoring**:
```python
def detect_impact(pedestrian_track, vehicle_track):
    """Detect impact using multiple signals."""
    signals = []
    confidence = 0.0
    
    # Signal 1: Overlap
    has_overlap, overlap_iou = detect_overlap_impact(
        pedestrian_track.current_bbox,
        vehicle_track.current_bbox
    )
    if has_overlap:
        signals.append("overlap")
        confidence = max(confidence, overlap_iou)
    
    # Signal 2: Track loss
    has_loss, loss_conf = detect_track_loss_impact(pedestrian_track, vehicle_track)
    if has_loss:
        signals.append("track_loss")
        confidence = max(confidence, loss_conf)
    
    # Signal 3: Proximity + velocity
    has_prox, prox_conf = detect_proximity_velocity_impact(pedestrian_track, vehicle_track)
    if has_prox:
        signals.append("proximity_velocity")
        confidence = max(confidence, prox_conf)
    
    # Multiple signals = higher confidence
    if len(signals) >= 2:
        confidence = min(1.0, confidence * 1.2)  # Boost if multiple signals
    
    # Impact if confidence > threshold
    if confidence > 0.5:  # Threshold
        return True, confidence, signals
    
    return False, 0.0, []
```

**Time to implement**: 2-3 hours

**Total for impact detection**: ~10-14 hours (1.5-2 days)

---

## Updated File Structure

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
├── impact/                          # NEW
│   ├── __init__.py
│   ├── detector.py                  # Detect impact events
│   ├── signals.py                   # Impact detection signals
│   └── impact_types.py              # Impact data types
└── track/
    └── track_history.py             # Track position history
```

---

## Updated Pipeline Flow

```
Frame → Detection → Tracking → Filtering → Risk Scoring → Near Miss Detection
                                                              ↓
                                                         Impact Detection
                                                              ↓
                                                         LPR (on impact)
                                                              ↓
                                                         Output
```

---

## Impact Detection Requirements

### Input
- Pedestrian tracks (with position history)
- Vehicle tracks (with position history)
- Current frame detections

### Output
```python
@dataclass
class ImpactEvent:
    event_id: str
    timestamp: float
    frame_id: int
    pedestrian_track_id: int
    vehicle_track_id: int
    confidence: float  # 0.0 - 1.0
    signals: List[str]  # ["overlap", "track_loss", "proximity_velocity"]
    evidence_frame_ids: List[int]  # Frames around impact
```

---

## Integration with LPR

### Before (Without Impact Detection)
```python
# LPR triggered by risk
if risk_tier in ["high", "critical"]:
    extract_license_plate(vehicle)
```

### After (With Impact Detection)
```python
# LPR triggered by impact
if impact_event:
    extract_license_plate(impact_event.vehicle_track_id)
```

---

## Testing Strategy

### Unit Tests
- Test overlap detection
- Test track loss detection
- Test proximity + velocity
- Test confidence scoring

### Integration Tests
- Test on sample videos with known impacts
- Validate impact detection accuracy
- Check false positive rate
- Verify LPR triggering

### Validation
- Compare detected impacts with ground truth
- Tune thresholds based on results
- Measure precision/recall

---

## Success Criteria

- [ ] Impact detection identifies actual collisions
- [ ] Low false positive rate (< 10% ideally)
- [ ] Reasonable recall (> 70% ideally)
- [ ] LPR triggered correctly on impacts
- [ ] System runs in acceptable time

---

## What You'll Have After 1-1.5 Weeks

✅ **Complete System**:
1. Detection & Tracking (already exists)
2. Filtering (people in vehicles)
3. Risk Scoring (image-space)
4. Near Miss Detection
5. **Impact Detection** (simplified but functional)
6. **LPR on Impact** (gated correctly)
7. Output (JSON, events, evidence)

✅ **Fulfills All Requirements**:
- ✅ Collision Risk Detection
- ✅ **Impact Detection** (simplified)
- ✅ License Plate Extraction (on impact)

---

## Trade-offs

### What You Get
- ✅ Complete system (all 3 requirements)
- ✅ Impact detection works
- ✅ LPR correctly gated

### What You Sacrifice
- ⚠️ Simplified impact detection (not perfect)
- ⚠️ May have some false positives/negatives
- ⚠️ Takes 1-1.5 weeks (not < 1 week)

### Can Be Improved Later
- Add velocity discontinuity analysis
- Add fall-like motion detection
- Add vehicle deceleration detection
- Improve confidence scoring

---

## Recommendation

**Go with this plan** (1-1.5 weeks with impact detection):
- ✅ Fulfills all requirements
- ✅ Realistic timeline
- ✅ Functional system
- ✅ Can refine later

**Don't try to do it in < 1 week** - too risky, will likely fail or be incomplete.

---

## Branch Structure (One Branch Per Day/Phase)

### Recommended Branch Names

- **Day 1**: `veer/filtering` - Filter people inside vehicles
- **Day 2**: `veer/risk-scoring` - Risk scoring implementation
- **Day 3**: `veer/near-miss` - Near miss detection
- **Day 4-5**: `veer/impact-detection` - Impact detection
- **Day 6**: `veer/integration` - Integrate all components
- **Day 7**: `veer/polish` - Testing and refinement

### Branch Workflow

1. Create branch from `veer/detecting-objects` (or previous completed branch)
2. Work on feature for that day
3. Test and commit
4. Merge back to `veer/detecting-objects` when complete (or keep separate)
5. Create next branch from updated base
6. Repeat

### Benefits

- ✅ Focused work per feature
- ✅ Easier to test and review
- ✅ Cleaner commit history
- ✅ Can merge incrementally
- ✅ Easy to rollback if needed

---

## Next Steps

1. **Accept timeline**: 1-1.5 weeks (not < 1 week)
2. **Create branch**: `veer/filtering` from `veer/detecting-objects`
3. **Start implementation**: Begin with filtering (Day 1)
4. **Test early**: Test impact detection as soon as possible
5. **Iterate**: Tune thresholds based on results

This gives you a **complete, working system** that fulfills all requirements!

