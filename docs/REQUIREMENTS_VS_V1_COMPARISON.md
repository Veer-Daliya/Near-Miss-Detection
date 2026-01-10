# README Requirements vs v1 Quick Implementation

This document compares what the README requires vs what the v1 quick implementation (< 1 week) can actually deliver.

---

## README Core Requirements (3 Objectives)

### 1. Collision Risk Detection ✅ CAN DELIVER
**Requirement**: Identify pedestrian–vehicle interactions where a collision is likely within a configurable horizon (default: 1–3 seconds). Output a risk score and alert tier.

**v1 Quick Implementation**:
- ✅ Can calculate risk scores in image-space (pixels)
- ✅ Can assign risk tiers (low/medium/high/critical)
- ✅ Can detect near miss events
- ⚠️ **Caveat**: Time horizon is in **frames**, not real seconds (but can estimate based on FPS)

**Status**: **FULFILLS REQUIREMENT** (with image-space limitations)

---

### 2. Impact Detection ❌ CANNOT DELIVER IN < 1 WEEK
**Requirement**: Detect when an actual collision/impact event has occurred. Output event timestamp, confidence, and associated actor IDs.

**v1 Quick Implementation**:
- ❌ **Explicitly excluded** from < 1 week plan (too complex)
- ❌ Requires velocity discontinuity analysis
- ❌ Requires fall-like motion detection
- ❌ Requires track disappearance analysis
- ❌ Requires vehicle deceleration detection

**Status**: **DOES NOT FULFILL REQUIREMENT**

**Workaround**: Can detect "near miss" events (high risk), but cannot distinguish actual impacts from near misses.

---

### 3. License Plate Extraction ⚠️ PARTIAL
**Requirement**: When an impact is confirmed (or high-confidence), extract the license plate of the involved vehicle. Return plate text, confidence score, and evidence frames.

**v1 Quick Implementation**:
- ✅ LPR system already exists and works
- ❌ **Cannot gate by impact** (no impact detection)
- ✅ Can gate by **high-risk events** instead
- ✅ Can return plate text, confidence, evidence frames

**Status**: **PARTIALLY FULFILLS** (works, but triggered by risk instead of impact)

---

## README System Architecture Requirements

### Module 1: Ingest ✅ ALREADY EXISTS
- ✅ Video file reading
- ✅ Frame sampling
- ✅ Timestamp handling

### Module 2: Detection ✅ ALREADY EXISTS
- ✅ YOLO detection
- ✅ Pedestrian + vehicle detection
- ✅ Bounding boxes, classes, confidence

### Module 3: Tracking ✅ ALREADY EXISTS
- ✅ ByteTrack tracking
- ✅ Stable track IDs
- ✅ Track histories

### Module 4: Risk Scoring ✅ CAN DELIVER
- ✅ Physics-based risk scoring
- ✅ Image-space TTC proxy
- ✅ Risk tiers
- ❌ VLM escalation (optional, not critical)

### Module 5: Impact Detection ❌ CANNOT DELIVER
- ❌ Candidate event windows
- ❌ Velocity change detection
- ❌ Fall detection
- ❌ Track disappearance analysis

### Module 6: Vehicle Association ❌ CANNOT DELIVER (depends on impact)
- ❌ Links impact to vehicle
- ✅ Can link risk events to vehicles (workaround)

### Module 7: License Plate Recognition ✅ ALREADY EXISTS
- ✅ Plate detection
- ✅ OCR
- ✅ Multi-frame aggregation
- ⚠️ Triggered by risk, not impact

### Module 8: Output ✅ CAN DELIVER
- ✅ Risk tier alerts
- ❌ Impact event records (no impact detection)
- ✅ Plate results (triggered by risk)
- ✅ JSON/file outputs

---

## Summary Table

| Requirement | README Expects | v1 Quick (< 1 week) | Status |
|-------------|----------------|---------------------|--------|
| **Collision Risk Detection** | ✅ Required | ✅ Can deliver | ✅ **FULFILLS** |
| **Impact Detection** | ✅ Required | ❌ Cannot deliver | ❌ **MISSING** |
| **License Plate Extraction** | ✅ Required (on impact) | ⚠️ Partial (on risk) | ⚠️ **PARTIAL** |
| **Risk Scoring** | ✅ Required | ✅ Can deliver | ✅ **FULFILLS** |
| **Near Miss Detection** | ✅ Required | ✅ Can deliver | ✅ **FULFILLS** |
| **VLM Escalation** | Optional | ❌ Not included | ⚠️ **OPTIONAL** |

---

## What v1 Quick Implementation CAN Deliver

### ✅ Full Working System
1. **Detection & Tracking** (already exists)
2. **Filtering** (remove people in vehicles)
3. **Risk Scoring** (image-space, pixel-based)
4. **Near Miss Detection** (high-risk events)
5. **LPR on High-Risk Events** (not impacts, but high-risk)
6. **Output** (JSON, events, evidence)

### ✅ Meets Core Purpose
- Can detect dangerous pedestrian-vehicle interactions
- Can identify when collision is likely
- Can extract license plates for high-risk events
- Can provide evidence and alerts

---

## What v1 Quick Implementation CANNOT Deliver

### ❌ Missing Features
1. **Impact Detection** - Cannot distinguish actual collisions from near misses
2. **Impact-Triggered LPR** - LPR triggered by risk, not confirmed impact
3. **VLM Escalation** - Not included (optional anyway)

---

## Can v1 Quick Implementation "Work"?

### Short Answer: **YES, but with limitations**

### What "Works" Means:
- ✅ **Functional**: System runs, detects risks, outputs results
- ✅ **Useful**: Can identify dangerous situations
- ✅ **Complete MVP**: Has core risk detection functionality
- ⚠️ **Incomplete**: Missing impact detection (one of 3 core objectives)

### What "Fulfills Requirements" Means:
- ⚠️ **Partially**: Fulfills 2 of 3 core objectives
- ❌ **Not fully**: Missing impact detection requirement
- ✅ **Close enough**: For MVP/demo purposes, may be acceptable

---

## Recommendations

### Option 1: Accept Partial Fulfillment (Recommended for < 1 week)
**What to do**:
- Build v1 quick implementation
- Document that impact detection is "future work"
- Use "high-risk events" instead of "impacts" for LPR triggering
- **Status**: Functional MVP, but not complete system

**Pros**:
- ✅ Achievable in < 1 week
- ✅ Core functionality works
- ✅ Can demo and test
- ✅ Can add impact detection later

**Cons**:
- ❌ Doesn't fulfill all README requirements
- ❌ Cannot distinguish impacts from near misses

---

### Option 2: Extend Timeline to Include Impact Detection
**What to do**:
- Add 3-5 more days for impact detection
- Implement basic impact signals
- **Total time**: ~1.5-2 weeks

**Pros**:
- ✅ Fulfills all README requirements
- ✅ Complete system

**Cons**:
- ❌ Takes longer than 1 week
- ❌ More complex

---

### Option 3: Simplified Impact Detection (Compromise)
**What to do**:
- Implement basic impact detection (overlap + track loss)
- Skip complex signals (velocity discontinuity, fall detection)
- **Total time**: ~1 week + 2-3 days

**Pros**:
- ✅ Fulfills requirement (basic version)
- ✅ Reasonable timeline

**Cons**:
- ⚠️ Less accurate impact detection
- ⚠️ May have false positives

---

## Conclusion

### Does v1 Quick Implementation "Work"?
**YES** - It's a functional system that can detect risks and near misses.

### Does it Fulfill All README Requirements?
**NO** - It's missing impact detection (1 of 3 core objectives).

### Is It Acceptable?
**DEPENDS ON YOUR GOALS**:
- ✅ **For MVP/Demo**: Yes, acceptable
- ✅ **For Testing**: Yes, can test risk detection
- ❌ **For Production**: May need impact detection
- ❌ **For Complete System**: Missing one requirement

### Recommendation
**Build v1 quick implementation** and document that:
1. Impact detection is "Phase 2" work
2. LPR is triggered by "high-risk events" instead of "impacts"
3. System fulfills 2 of 3 core objectives
4. Can be extended later with impact detection

This gives you a **working, useful system** in < 1 week, even if it's not 100% complete.

---

## Next Steps

1. **Decide**: Accept partial fulfillment or extend timeline?
2. **If accept partial**: Build v1 quick implementation, document limitations
3. **If extend**: Add impact detection to timeline (1.5-2 weeks total)
4. **If compromise**: Implement basic impact detection (1 week + 2-3 days)

