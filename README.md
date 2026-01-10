# Near-Miss Detection System

A computer vision pipeline for detecting pedestrian–vehicle collision risk, confirmed impacts, and automatic license plate extraction from fixed and PTZ camera feeds.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Assumptions & Camera Context](#assumptions--camera-context)
- [System Architecture](#system-architecture)
- [Module Details](#module-details)
- [Computational Cost Guidance](#computational-cost-guidance)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Output Schema](#output-schema)

---

## Feature Status

| Feature | Status | Module |
|---------|--------|--------|
| Object Detection (YOLO) | Implemented | `src/detect/` |
| Multi-Object Tracking (ByteTrack) | Implemented | `src/track/` |
| Pedestrian-in-Vehicle Filtering | Implemented | `src/filter/` |
| License Plate Detection | Implemented | `src/lpr/plate_detector.py` |
| License Plate OCR | Implemented | `src/lpr/ocr.py` |
| Multi-Frame Plate Aggregation | Implemented | `src/lpr/aggregator.py` |
| Ground Plane Estimation | Implemented | `src/ground_plane/` |
| Near-Miss Detection (TTC) | Implemented | `src/risk/` |
| Trajectory Tracking | Implemented | `src/risk/trajectory.py` |
| Collision Prediction | Implemented | `src/risk/collision_predictor.py` |
| Impact Detection | Planned | - |
| VLM Escalation | Planned | - |

### Quick Start with Near-Miss Detection

```bash
python scripts/run_detection.py --source video.mp4 --enable-near-miss
```

---

## Problem Statement

This system addresses three core objectives:

1. **Collision Risk Detection** — Identify pedestrian–vehicle interactions where a collision is likely within a configurable horizon (default: 1–3 seconds). Output a risk score and alert tier.

2. **Impact Detection** — Detect when an actual collision/impact event has occurred. Output event timestamp, confidence, and associated actor IDs.

3. **License Plate Extraction** — When an impact is confirmed (or high-confidence), extract the license plate of the involved vehicle. Return plate text, confidence score, and evidence frames.

---

## Assumptions & Camera Context

### Camera Types

| Type | Notes |
|------|-------|
| **Fixed cameras** | Static viewpoint; calibration is stable once set. |
| **PTZ cameras** | Variable zoom, pan, tilt; requires handling of camera motion events, rolling calibration challenges, and tracker resets. |

### PTZ-Specific Considerations

- Detect "camera moved" periods via background flow spikes or PTZ metadata.
- Reset or compensate trackers after significant pan/tilt/zoom changes.
- Zoom changes affect apparent object size and velocity—normalize where possible.

### Scope Constraints (v1)

- **No full 3D scene understanding** — No lane/crosswalk segmentation required in v1.
- **Hooks for future extensions** — Architecture allows plugging in scene context, depth estimation, or semantic maps later.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CAMERA FEEDS                                    │
│                        (RTSP / VOD / File Input)                            │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. INGEST                                                                   │
│     - RTSP stream reader / VOD file loader                                  │
│     - Frame sampling (configurable FPS)                                     │
│     - PTZ metadata extraction (if available)                                │
│     Output: frame_id, timestamp, image, ptz_state (optional)                │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. ACTOR DETECTION                                                          │
│     - Detect pedestrians + vehicles (YOLO-family recommended)               │
│     - Output bounding boxes, class labels, confidence scores                │
│     Output: List[Detection] per frame                                       │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. MULTI-OBJECT TRACKING                                                    │
│     - Assign stable IDs across frames (ByteTrack / DeepSORT)                │
│     - Build tracklets with position history                                 │
│     - Handle occlusions, re-ID, PTZ motion resets                           │
│     Output: List[Track] with smoothed trajectories                          │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
              ▼                                       ▼
┌──────────────────────────────┐       ┌──────────────────────────────────────┐
│  4. RISK SCORING             │       │  5. IMPACT DETECTION                  │
│     - Physics-based (fast)   │       │     - Candidate event windows         │
│     - Image-space TTC proxy  │       │     - Velocity change / fall detect   │
│     - Optional: VLM escalate │       │     - Track disappearance analysis    │
│     Output: risk_score, tier │       │     Output: ImpactEvent (if any)      │
└──────────────┬───────────────┘       └──────────────────┬───────────────────┘
               │                                          │
               │    ┌─────────────────────────────────────┘
               │    │
               ▼    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. VEHICLE ASSOCIATION                                                      │
│     - Link impact event to specific vehicle track                           │
│     - Resolve ambiguity via proximity + motion + timing                     │
│     Output: associated_vehicle_id                                           │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼ (only on confirmed/high-confidence impact)
┌─────────────────────────────────────────────────────────────────────────────┐
│  7. LICENSE PLATE RECOGNITION                                                │
│     - Plate detection on vehicle ROI                                        │
│     - Best-frame selection (visibility, resolution, angle)                  │
│     - OCR + multi-frame aggregation                                         │
│     Output: plate_text, plate_confidence, evidence_frames                   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  8. OUTPUT + ALERTING                                                        │
│     - Risk tier alerts (low / medium / high / critical)                     │
│     - Impact event records                                                  │
│     - Plate results with evidence                                           │
│     - Webhook / queue / file outputs                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Module | Input | Output |
|--------|-------|--------|
| Ingest | RTSP URL / file path | `Frame` (image, timestamp, metadata) |
| Detection | Frame | `List[Detection]` (bbox, class, conf) |
| Tracking | Detections (stream) | `List[Track]` (id, positions, velocities) |
| Risk Scoring | Tracks | `RiskAssessment` (score, tier, pairs) |
| Impact Detection | Tracks + Frames | `ImpactEvent` (time, ped_id, vehicle_id, conf) |
| Vehicle Association | ImpactEvent + Tracks | `vehicle_id` |
| LPR | Vehicle ROI + Frames | `PlateResult` (text, conf, evidence) |
| Output | All events | JSON / Webhook / Alerts |

---

## Module Details

### 1. Detection + Tracking

#### Detector

- **Recommended**: YOLOv10 (faster, better for large objects like vehicles)
- **Classes**: `person`, `car`, `truck`, `bus`, `motorcycle`, `bicycle`
- **Confidence threshold**: 0.3–0.5 (tune per deployment)

#### Tracker

- **Recommended**: ByteTrack (state-of-the-art, no re-ID features needed for baseline)
- **Alternative**: DeepSORT if re-ID across occlusions is critical
- **Track smoothing**: Kalman filter on centroid + bbox dimensions
- **Re-ID strategy**: Appearance embedding (optional for v2)

#### PTZ Handling

```python
# Pseudocode for PTZ motion detection
def detect_camera_motion(prev_frame, curr_frame, threshold=0.6):
    """Detect significant camera movement via background flow."""
    flow = compute_optical_flow(prev_frame, curr_frame)
    median_magnitude = np.median(np.linalg.norm(flow, axis=-1))
    return median_magnitude > threshold

# On camera motion detected:
# - Option A: Reset all trackers (simple, robust)
# - Option B: Apply global motion compensation (complex, fragile)
```

### 2. Physics-Based Risk Scoring (Fast Baseline)

Operates entirely in **image space** — no calibration required for v1.

#### Metrics

| Metric | Description |
|--------|-------------|
| **Relative approach rate** | Rate of decrease in distance between pedestrian and vehicle centroids (pixels/frame) |
| **Box growth rate** | Rate of vehicle bbox expansion toward pedestrian (proxy for approach) |
| **Time-to-contact (TTC) proxy** | `distance / closing_speed` in image space (frames until overlap) |
| **Minimum predicted separation** | Predicted closest approach over horizon using linear extrapolation |

#### Risk Tiers

```python
def compute_risk_tier(ttc_proxy: float, min_separation: float) -> str:
    if ttc_proxy < 10 and min_separation < 50:  # frames, pixels
        return "critical"
    elif ttc_proxy < 30 and min_separation < 100:
        return "high"
    elif ttc_proxy < 60 and min_separation < 200:
        return "medium"
    else:
        return "low"
```

#### Optional Calibration Mode

If a ground-plane homography is available:
- Transform image coordinates to world coordinates (meters).
- Compute true TTC in seconds.
- Improve accuracy for angled cameras.

### 3. VLM Escalation (API-Based)

**Strategy**: Only invoke VLM when physics model outputs `high` or `critical` risk, or when uncertainty is high.

#### Sampling Policy

```python
VLM_CONFIG = {
    "trigger_tiers": ["high", "critical"],
    "frame_count": 5,           # frames to send
    "time_span_sec": 1.0,       # span of clip
    "max_calls_per_minute": 10, # rate limit
    "timeout_sec": 5.0,
    "fallback": "use_physics_only"
}
```

#### Redaction Policy

- Blur all faces (apply face detection + Gaussian blur).
- Crop to interaction region (union of pedestrian + vehicle bboxes + margin).
- Do not send full frames to minimize data exposure.

#### Prompt Template

```
You are analyzing a potential pedestrian-vehicle collision.

Context:
- Camera: {camera_id}
- Timestamp: {timestamp}
- Physics risk score: {risk_score} ({risk_tier})

Images attached show {frame_count} frames spanning {time_span}s.

Classify this interaction:
1. NEAR_MISS - Close call but no contact
2. LIKELY_IMPACT - Contact appears imminent or just occurred
3. CONFIRMED_IMPACT - Clear evidence of collision
4. UNCERTAIN - Cannot determine from available frames

Provide:
- classification: one of the above
- confidence: 0.0 to 1.0
- rationale: brief explanation (1-2 sentences)

Respond in JSON format only.
```

#### Cost/Latency Controls

- **Caching**: Cache results by frame hash for 5 minutes (avoid re-processing same clip).
- **Rate limiting**: Max N calls/minute per camera, with backpressure.
- **Fallback**: If VLM times out or errors, fall back to physics-only classification.
- **Batching**: Queue multiple escalations and batch if supported by API.

### 4. Impact Detection

#### Candidate Event Windows

Generate candidate windows when:
- Pedestrian track and vehicle track bboxes overlap or are within threshold distance.
- Duration: analyze ±1 second around the proximity event.

#### Signals

| Signal | Description | Weight |
|--------|-------------|--------|
| **Velocity discontinuity** | Sudden change in pedestrian velocity (>2σ from recent mean) | High |
| **Fall-like motion** | Bbox aspect ratio change + downward centroid motion | Medium |
| **Track disappearance** | Pedestrian track lost while vehicle track continues in proximity | Medium |
| **Vehicle deceleration** | Sudden vehicle slowdown near pedestrian | Low |

#### Occlusion Guardrails

- Do not flag track loss as impact if pedestrian reappears within N frames.
- Check if another object (not the vehicle) occluded the pedestrian.

#### Output

```python
@dataclass
class ImpactEvent:
    event_id: str
    timestamp: float
    frame_id: int
    pedestrian_track_id: int
    vehicle_track_id: int
    confidence: float  # 0.0 - 1.0
    signals: List[str]  # which signals triggered
    evidence_frame_ids: List[int]
```

### 5. Vehicle Association

When multiple vehicles are near the impact:

1. **Proximity score**: Inverse distance to pedestrian at impact time.
2. **Motion alignment**: Is vehicle moving toward pedestrian?
3. **Overlap timing**: Which vehicle bbox overlapped pedestrian first?
4. **Track continuity**: Prefer vehicles with stable tracks (not just appeared).

```python
def associate_vehicle(impact: ImpactEvent, nearby_vehicles: List[Track]) -> int:
    scores = []
    for v in nearby_vehicles:
        prox = 1.0 / (distance(v, impact.pedestrian_position) + 1e-6)
        motion = cosine_similarity(v.velocity, direction_to_pedestrian)
        overlap = overlap_iou(v.bbox, impact.pedestrian_bbox)
        scores.append(prox * 0.4 + motion * 0.3 + overlap * 0.3)
    return nearby_vehicles[np.argmax(scores)].track_id
```

### 6. License Plate Recognition (On Impact Only)

Triggered only after impact is confirmed or high-confidence.

#### Pipeline

1. **Frame selection**: Score frames by plate visibility:
   - Vehicle bbox size (larger = better)
   - Plate region not occluded
   - Low motion blur (sharpness metric)
   - Frontal/rear angle preferred

2. **Plate detection**: Run plate detector on vehicle ROI.
   - Recommended: YOLO-based plate detector or OpenALPR.

3. **OCR**: Extract text from plate crop.
   - Recommended: PaddleOCR or EasyOCR.

4. **Aggregation**: Combine results across frames.
   - Majority vote on characters.
   - Confidence-weighted averaging.
   - Require minimum agreement threshold.

#### Output

```python
@dataclass
class PlateResult:
    plate_text: str           # e.g., "ABC1234" or "UNKNOWN"
    plate_confidence: float   # 0.0 - 1.0
    plate_bbox: List[int]     # [x1, y1, x2, y2] in best frame
    best_frame_id: int
    evidence_frame_ids: List[int]
    evidence_timestamps: List[float]
    ocr_candidates: List[dict]  # all readings with confidences
```

---

## Computational Cost Guidance

### Why Full Monocular Depth is Optional

| Approach | Cost | Accuracy | Recommendation |
|----------|------|----------|----------------|
| Image-space risk | Low | Sufficient for most scenes | **v1 default** |
| Homography (ground plane) | Low | Good if calibrated | v1 optional |
| Monocular depth (MiDaS, DepthAnything) | High (~50-100ms/frame) | Better for complex scenes | v2 second-pass |

### Staged Approach

**v1 (Current)**
- Always-on: Detection + Tracking + Image-space physics risk.
- On-demand: VLM escalation for high-risk windows.
- On-impact: LPR pipeline.

**v2 (Future)**
- Add monocular depth as second-pass on high-risk windows only.
- Use depth for improved TTC and occlusion reasoning.
- Still not run on every frame — too expensive.

### PTZ Cost Notes

PTZ adds complexity:
- Camera motion detection adds ~5ms/frame.
- Tracker resets cause temporary ID churn.
- Zoom normalization requires additional computation.

**v1 recommendation**: Use simple heuristics (reset on motion), avoid complex compensation.

---

## Evaluation

### Risk Scoring Metrics

| Metric | Description |
|--------|-------------|
| **Lead time** | Seconds before impact that risk was flagged (higher = better) |
| **False alarms/hour/camera** | Rate of false high-risk alerts |
| **PR-AUC** | Precision-recall on labeled risky interactions |
| **Tier accuracy** | Correct tier assignment on labeled events |

### Impact Detection Metrics

| Metric | Description |
|--------|-------------|
| **Event precision** | Fraction of detected impacts that are true |
| **Event recall** | Fraction of true impacts that are detected |
| **Time tolerance** | Allow ±0.5s for timestamp matching |

### LPR Metrics

| Metric | Description |
|--------|-------------|
| **Exact match accuracy** | Plate text matches ground truth exactly |
| **Character error rate** | Levenshtein distance / plate length |
| **Confidence calibration** | High-confidence predictions should be correct |
| **Unreadable rate** | Fraction returning "UNKNOWN" (lower is better, but not at cost of accuracy) |

---

## Deployment

### Inference Modes

| Mode | Use Case | Latency Target |
|------|----------|----------------|
| **Edge** | Real-time alerting, low-bandwidth sites | <100ms end-to-end |
| **Server** | Centralized processing, higher accuracy models | <500ms acceptable |
| **Batch/Offline** | Post-incident review, forensics | Throughput-focused |

### Latency Targets

- Frame ingest → risk score: **<100ms** (edge), **<300ms** (server)
- Impact detection → LPR result: **<2s** (acceptable for post-event)
- VLM escalation: **<5s** (async, non-blocking)

### Monitoring

- Model drift detection (detection confidence distribution shift).
- Lighting change alerts (mean frame brightness tracking).
- PTZ zoom distribution (are we always zoomed in? out?).
- Track ID churn rate (high = tracker instability).

### Privacy

- **Face blurring**: Apply before any frame leaves the system or is stored.
- **Retention policy**: Store only event clips (impact + context window), delete raw stream.
- **Access control**: Role-based access to evidence frames.
- **Audit logging**: Track all access to plate data.

---

## Repository Structure

```
Near-Miss-Detection/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── pyproject.toml
│
├── src/
│   ├── __init__.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── rtsp_reader.py
│   │   ├── vod_reader.py
│   │   ├── frame_sampler.py
│   │   └── ptz_metadata.py
│   │
│   ├── detect/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   ├── yolo_wrapper.py
│   │   └── detection_types.py
│   │
│   ├── track/
│   │   ├── __init__.py
│   │   ├── tracker.py
│   │   ├── bytetrack.py
│   │   ├── tracklet.py
│   │   └── ptz_motion.py
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── physics_scorer.py
│   │   ├── risk_types.py
│   │   └── homography.py
│   │
│   ├── impact/
│   │   ├── __init__.py
│   │   ├── impact_detector.py
│   │   ├── signals.py
│   │   └── vehicle_association.py
│   │
│   ├── lpr/
│   │   ├── __init__.py
│   │   ├── plate_detector.py
│   │   ├── ocr.py
│   │   ├── frame_selector.py
│   │   └── aggregator.py
│   │
│   ├── vlm/
│   │   ├── __init__.py
│   │   ├── escalation.py
│   │   ├── prompts.py
│   │   ├── redaction.py
│   │   └── rate_limiter.py
│   │
│   ├── output/
│   │   ├── __init__.py
│   │   ├── alerter.py
│   │   ├── webhook.py
│   │   └── file_writer.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── geometry.py
│       ├── video.py
│       ├── logging.py
│       └── config.py
│
├── configs/
│   ├── default.yaml
│   ├── fixed_camera.yaml
│   ├── ptz_camera.yaml
│   └── evaluation.yaml
│
├── docs/
│   ├── session-plan.md
│   ├── architecture.md
│   └── api.md
│
├── tests/
│   ├── __init__.py
│   ├── test_detect/
│   ├── test_track/
│   ├── test_risk/
│   ├── test_impact/
│   ├── test_lpr/
│   └── fixtures/
│
├── scripts/
│   ├── run_pipeline.py
│   ├── evaluate.py
│   └── demo.py
│
└── data/
    ├── samples/           # sample videos for testing
    ├── models/            # model weights (gitignored)
    └── outputs/           # pipeline outputs (gitignored)
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for real-time)
- FFmpeg (for video decoding)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/InspiritAI/Near-Miss-Detection.git
cd Near-Miss-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
pip install paddlepaddle paddleocr
pip install lap

# Run detection
python scripts/run_detection.py --source data/samples/traffic_video.mp4
```

**For detailed installation and usage instructions, see the [Getting Started Guide](docs/getting-started.md).**

### Configuration

#### Fixed Camera Example (`configs/fixed_camera.yaml`)

```yaml
camera:
  type: fixed
  source: "rtsp://192.168.1.100:554/stream1"
  fps: 10

detection:
  model: yolov10m
  confidence_threshold: 0.4
  classes: [person, car, truck, bus, motorcycle]

tracking:
  algorithm: bytetrack
  max_age: 30
  min_hits: 3

risk:
  horizon_frames: 30  # ~3 seconds at 10 fps
  ttc_critical: 10
  ttc_high: 30
  min_separation_critical: 50
  min_separation_high: 100

vlm:
  enabled: true
  provider: openai
  model: gpt-4-vision-preview
  trigger_tiers: [high, critical]
  rate_limit_per_minute: 10

lpr:
  enabled: true
  trigger_on: impact_confirmed
  min_plate_confidence: 0.7

output:
  webhook_url: null
  save_events: true
  output_dir: ./data/outputs
```

#### PTZ Camera Example (`configs/ptz_camera.yaml`)

```yaml
camera:
  type: ptz
  source: "rtsp://192.168.1.101:554/stream1"
  fps: 10
  ptz_metadata_enabled: true

detection:
  model: yolov10m
  confidence_threshold: 0.4
  classes: [person, car, truck, bus, motorcycle]

tracking:
  algorithm: bytetrack
  max_age: 30
  min_hits: 3
  ptz_motion_threshold: 0.6  # reset trackers if exceeded
  ptz_reset_strategy: full   # or 'compensate'

risk:
  horizon_frames: 30
  ttc_critical: 10
  ttc_high: 30
  # Disable homography for PTZ (calibration not stable)
  use_homography: false

vlm:
  enabled: true
  provider: openai
  model: gpt-4-vision-preview
  trigger_tiers: [high, critical]
  rate_limit_per_minute: 10

lpr:
  enabled: true
  trigger_on: impact_confirmed
  min_plate_confidence: 0.7

output:
  webhook_url: null
  save_events: true
  output_dir: ./data/outputs
```

### Running the Pipeline

```bash
# Run on a video file
python scripts/run_detection.py --source data/samples/traffic_video.mp4

# Run with custom settings
python scripts/run_detection.py --source video.mp4 --model-size s --fps 10

# Run on RTSP stream
python scripts/run_detection.py --source rtsp://192.168.1.100:554/stream1
```

**For detailed run options and configurations, see the [Running Guide](docs/running.md).**

---

## Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[Getting Started](docs/getting-started.md)** - Installation, setup, and basic usage
- **[Running the System](docs/running.md)** - Detailed run options, configurations, and examples
- **[Performance Guide](docs/performance.md)** - Optimization tips and code-level optimizations
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Session Plan](docs/session-plan.md)** - Development roadmap and architecture

---

## Output Schema

### Risk Assessment Output

```json
{
  "frame_id": 1234,
  "timestamp": 1703678400.123,
  "camera_id": "cam_001",
  "risk_assessments": [
    {
      "pedestrian_track_id": 5,
      "vehicle_track_id": 12,
      "risk_score": 0.85,
      "risk_tier": "high",
      "ttc_proxy_frames": 15,
      "min_separation_px": 80,
      "vlm_escalated": false
    }
  ]
}
```

### Impact Event Output

```json
{
  "event_id": "impact_20231227_143022_001",
  "event_type": "impact",
  "timestamp": 1703678422.456,
  "frame_id": 1456,
  "camera_id": "cam_001",
  "pedestrian_track_id": 5,
  "vehicle_track_id": 12,
  "confidence": 0.92,
  "detection_signals": ["velocity_discontinuity", "track_overlap"],
  "vlm_classification": "CONFIRMED_IMPACT",
  "vlm_confidence": 0.88,
  "vlm_rationale": "Vehicle bumper makes contact with pedestrian, pedestrian falls backward.",
  "evidence_frames": [1450, 1453, 1456, 1459, 1462],
  "plate_result": {
    "plate_text": "ABC1234",
    "plate_confidence": 0.94,
    "plate_bbox": [120, 340, 220, 380],
    "best_frame_id": 1460,
    "evidence_frame_ids": [1458, 1460, 1465],
    "ocr_candidates": [
      {"text": "ABC1234", "confidence": 0.94},
      {"text": "ABC1Z34", "confidence": 0.72}
    ]
  }
}
```

### Alert Output

```json
{
  "alert_id": "alert_20231227_143020_001",
  "alert_type": "risk_critical",
  "timestamp": 1703678420.100,
  "camera_id": "cam_001",
  "message": "Critical collision risk detected",
  "risk_tier": "critical",
  "pedestrian_track_id": 5,
  "vehicle_track_id": 12,
  "ttc_seconds": 1.5,
  "recommended_action": "immediate_review"
}
```

---

## License

See [LICENSE](LICENSE) for details.
