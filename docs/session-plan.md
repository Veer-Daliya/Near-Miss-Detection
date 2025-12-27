# Session Plan: Sessions 4–10

This document outlines the development plan for the next 7 sessions, building the Near-Miss Detection system from foundational components to a fully integrated pipeline.

---

## Session Overview

| Session | Focus | Key Deliverable |
|---------|-------|-----------------|
| 4 | Repo scaffolding + Ingest + Detector baseline | Working detection on sample video |
| 5 | Tracking + Tracklets + PTZ motion handling | Stable tracks with IDs across frames |
| 6 | Physics risk scorer + Thresholds + Eval harness | Risk scores for pedestrian-vehicle pairs |
| 7 | Impact candidate generation + Impact heuristics | Impact event detection baseline |
| 8 | VLM escalation integration | VLM classification with cost controls |
| 9 | LPR pipeline | Plate extraction on impact events |
| 10 | End-to-end integration + Metrics + Demo | Complete working system |

---

## Session 4: Repo Scaffolding + Ingest + Detector Baseline

### Goal
Establish the repository structure, implement video ingestion, and get a working object detector producing bounding boxes on sample footage.

### Scope
- Create the full directory structure as specified in README.
- Implement frame ingestion from video files (VOD first, RTSP stub).
- Integrate YOLOv8 for pedestrian and vehicle detection.
- Output detections to JSON for inspection.

### Tasks

- [ ] **Scaffolding**
  - Create directory structure: `src/`, `configs/`, `docs/`, `tests/`, `scripts/`
  - Add `requirements.txt` with core dependencies:
    ```
    ultralytics>=8.0.0
    opencv-python>=4.8.0
    numpy>=1.24.0
    pyyaml>=6.0
    ```
  - Add `pyproject.toml` with project metadata
  - Create `src/__init__.py` and submodule `__init__.py` files

- [ ] **Ingest module**
  - Implement `src/ingest/vod_reader.py`:
    - Read video file with OpenCV
    - Yield frames with frame_id and timestamp
  - Implement `src/ingest/frame_sampler.py`:
    - Configurable FPS sampling (e.g., process every Nth frame)
  - Add config loading in `src/utils/config.py`

- [ ] **Detection module**
  - Implement `src/detect/yolo_wrapper.py`:
    - Load YOLOv8 model (configurable size: n/s/m/l)
    - Run inference, filter by classes of interest
    - Return list of `Detection` objects
  - Define `src/detect/detection_types.py`:
    ```python
    @dataclass
    class Detection:
        bbox: List[int]  # [x1, y1, x2, y2]
        class_id: int
        class_name: str
        confidence: float
        frame_id: int
    ```

- [ ] **Pipeline script**
  - Create `scripts/run_detection.py`:
    - Load config
    - Read video frames
    - Run detector on each frame
    - Save detections to JSON

- [ ] **Sample data**
  - Add 1-2 short sample videos to `data/samples/` (or document where to download)

### Deliverables

| File | Description |
|------|-------------|
| `requirements.txt` | Core dependencies |
| `src/ingest/vod_reader.py` | Video file reader |
| `src/ingest/frame_sampler.py` | FPS sampling utility |
| `src/detect/yolo_wrapper.py` | YOLOv8 wrapper |
| `src/detect/detection_types.py` | Detection dataclass |
| `scripts/run_detection.py` | Detection-only pipeline |
| `configs/default.yaml` | Default configuration |

### Acceptance Criteria

- [ ] Running `python scripts/run_detection.py --source data/samples/test.mp4` produces a JSON file with detections
- [ ] Detections include pedestrians (`person`) and vehicles (`car`, `truck`, `bus`)
- [ ] Each detection has bbox, class, confidence, and frame_id
- [ ] Config file controls model size, confidence threshold, and target classes
- [ ] Code passes basic linting (`ruff check src/`)

### Stretch Goals

- [ ] Add RTSP reader stub (connect but don't fully implement)
- [ ] Add visualization script that draws bboxes on frames
- [ ] Dockerize the environment

---

## Session 5: Tracking + Tracklets + PTZ Motion Handling

### Goal
Implement multi-object tracking to assign stable IDs across frames, build tracklets with position history, and add basic PTZ camera motion detection with tracker reset logic.

### Scope
- Integrate ByteTrack for multi-object tracking.
- Build tracklet data structures with smoothed trajectories.
- Detect camera motion events (for PTZ) and reset/invalidate tracks appropriately.
- Output tracks to JSON with position history.

### Tasks

- [ ] **Tracking module**
  - Implement `src/track/bytetrack.py`:
    - Wrap ByteTrack library (use `supervision` or standalone implementation)
    - Accept detections, return tracked objects with IDs
  - Implement `src/track/tracklet.py`:
    - Define `Track` dataclass:
      ```python
      @dataclass
      class Track:
          track_id: int
          class_name: str
          positions: List[Tuple[int, int]]  # centroid history
          bboxes: List[List[int]]           # bbox history
          velocities: List[Tuple[float, float]]  # computed velocities
          frame_ids: List[int]
          is_active: bool
      ```
    - Implement tracklet manager that maintains track state

- [ ] **PTZ motion handling**
  - Implement `src/track/ptz_motion.py`:
    - Compute background optical flow between consecutive frames
    - Detect "camera moved" when median flow magnitude exceeds threshold
    - Return boolean flag per frame
  - Add tracker reset logic:
    - When camera motion detected, mark all tracks as inactive
    - Clear position histories
    - Allow new tracks to form

- [ ] **Track smoothing**
  - Add Kalman filter or simple exponential smoothing to centroid positions
  - Compute velocity from smoothed positions

- [ ] **Integration**
  - Update `scripts/run_pipeline.py` (rename from `run_detection.py`):
    - Run detection
    - Run tracking
    - Output tracks with full history

- [ ] **Tests**
  - Add `tests/test_track/test_bytetrack.py`:
    - Test that consistent detections get same ID
    - Test that track history grows over frames
  - Add `tests/test_track/test_ptz_motion.py`:
    - Test motion detection on synthetic flow

### Deliverables

| File | Description |
|------|-------------|
| `src/track/bytetrack.py` | ByteTrack wrapper |
| `src/track/tracklet.py` | Track dataclass and manager |
| `src/track/ptz_motion.py` | Camera motion detector |
| `scripts/run_pipeline.py` | Detection + tracking pipeline |
| `tests/test_track/` | Tracking unit tests |

### Acceptance Criteria

- [ ] Running pipeline on sample video produces tracks with stable IDs
- [ ] Same pedestrian maintains same track_id across frames (when visible)
- [ ] Tracks include position history and computed velocities
- [ ] PTZ motion detection correctly identifies camera movement in test cases
- [ ] Tracker resets when camera motion exceeds threshold (configurable)
- [ ] Output JSON includes track_id, positions, velocities, frame_ids

### Stretch Goals

- [ ] Add DeepSORT as alternative tracker (config-switchable)
- [ ] Implement track interpolation for short occlusions
- [ ] Add re-ID feature extraction for track recovery

---

## Session 6: Physics Risk Scorer + Thresholds + Evaluation Harness

### Goal
Implement image-space physics risk scoring for pedestrian-vehicle pairs, define risk thresholds/tiers, and create an evaluation framework for measuring risk prediction performance.

### Scope
- Compute relative approach rate, TTC proxy, and minimum predicted separation.
- Assign risk tiers (low/medium/high/critical) based on configurable thresholds.
- Build evaluation harness that measures lead time and false alarm rate on labeled data.

### Tasks

- [ ] **Risk scoring module**
  - Implement `src/risk/physics_scorer.py`:
    - For each (pedestrian, vehicle) pair:
      - Compute centroid distance
      - Compute closing speed (rate of distance change)
      - Compute TTC proxy = distance / closing_speed
      - Predict future positions via linear extrapolation
      - Compute minimum separation over horizon
    - Return `RiskAssessment` per pair
  - Define `src/risk/risk_types.py`:
    ```python
    @dataclass
    class RiskAssessment:
        pedestrian_track_id: int
        vehicle_track_id: int
        risk_score: float        # 0.0 - 1.0
        risk_tier: str           # low/medium/high/critical
        ttc_proxy_frames: float
        min_separation_px: float
        closing_speed_px_per_frame: float
    ```

- [ ] **Threshold configuration**
  - Add risk thresholds to config:
    ```yaml
    risk:
      horizon_frames: 30
      tier_thresholds:
        critical:
          ttc_max: 10
          separation_max: 50
        high:
          ttc_max: 30
          separation_max: 100
        medium:
          ttc_max: 60
          separation_max: 200
    ```

- [ ] **Optional homography support**
  - Add `src/risk/homography.py`:
    - Load homography matrix from config (if provided)
    - Transform image points to ground plane coordinates
    - Compute TTC in meters/seconds instead of pixels/frames
  - Keep optional: risk scorer works without it

- [ ] **Evaluation harness**
  - Create `scripts/evaluate.py`:
    - Load ground truth labels (JSON format with interaction annotations)
    - Run pipeline on video
    - Compare predictions to labels
    - Compute metrics:
      - Lead time (frames/seconds before labeled impact that risk was flagged)
      - False alarms per hour
      - Precision/Recall at each tier threshold
  - Define ground truth label schema:
    ```json
    {
      "events": [
        {
          "type": "near_miss",
          "frame_start": 100,
          "frame_end": 150,
          "pedestrian_id": 5,
          "vehicle_id": 12
        }
      ]
    }
    ```

- [ ] **Integration**
  - Update `scripts/run_pipeline.py`:
    - Run detection → tracking → risk scoring
    - Output risk assessments per frame

### Deliverables

| File | Description |
|------|-------------|
| `src/risk/physics_scorer.py` | Physics-based risk computation |
| `src/risk/risk_types.py` | RiskAssessment dataclass |
| `src/risk/homography.py` | Optional ground-plane transform |
| `scripts/evaluate.py` | Evaluation script |
| `configs/evaluation.yaml` | Evaluation configuration |

### Acceptance Criteria

- [ ] Risk scorer produces risk tier for each pedestrian-vehicle pair per frame
- [ ] Higher risk (lower TTC, smaller separation) → higher tier
- [ ] Risk output includes ttc_proxy, min_separation, closing_speed
- [ ] Evaluation script runs and outputs metrics on labeled test data
- [ ] Lead time metric correctly measures frames before event that risk was flagged
- [ ] Homography mode (when configured) produces TTC in seconds

### Stretch Goals

- [ ] Add risk visualization overlay on video frames
- [ ] Compute PR-AUC across threshold sweep
- [ ] Add box growth rate as additional risk signal

---

## Session 7: Impact Candidate Generation + Initial Impact Heuristics

### Goal
Implement impact detection by generating candidate event windows based on proximity and analyzing heuristic signals (velocity changes, track loss) to classify actual impacts.

### Scope
- Generate candidate windows when pedestrian and vehicle tracks overlap or are close.
- Implement signal extractors: velocity discontinuity, track disappearance, etc.
- Combine signals to produce impact confidence score.
- Output impact events with evidence frames.

### Tasks

- [ ] **Impact detection module**
  - Implement `src/impact/impact_detector.py`:
    - Scan track pairs for proximity events (bbox overlap or distance < threshold)
    - Generate candidate windows: ±N frames around proximity event
    - Score each candidate using signals
    - Return `ImpactEvent` for candidates above threshold
  - Define `ImpactEvent` in `src/impact/__init__.py`:
    ```python
    @dataclass
    class ImpactEvent:
        event_id: str
        timestamp: float
        frame_id: int
        pedestrian_track_id: int
        vehicle_track_id: int
        confidence: float
        signals: Dict[str, float]  # signal_name -> score
        evidence_frame_ids: List[int]
    ```

- [ ] **Signal extractors**
  - Implement `src/impact/signals.py`:
    - `velocity_discontinuity(track, frame_id)`: Detect sudden velocity change
    - `track_disappearance(track, frame_id, window)`: Detect track loss with vehicle nearby
    - `bbox_overlap(ped_track, vehicle_track, frame_id)`: Compute IOU at frame
    - `aspect_ratio_change(track, frame_id)`: Detect fall-like motion (optional)
  - Each signal returns a score in [0, 1]

- [ ] **Signal combination**
  - Weight and combine signals:
    ```python
    weights = {
        "velocity_discontinuity": 0.35,
        "track_disappearance": 0.25,
        "bbox_overlap": 0.25,
        "aspect_ratio_change": 0.15
    }
    confidence = sum(weights[s] * signals[s] for s in signals)
    ```
  - Configurable threshold for impact classification

- [ ] **Vehicle association**
  - Implement `src/impact/vehicle_association.py`:
    - When impact detected, if multiple vehicles nearby, select most likely:
      - Closest at impact frame
      - Moving toward pedestrian
      - Highest bbox overlap

- [ ] **Occlusion guardrails**
  - Do not flag track disappearance if:
    - Pedestrian reappears within N frames
    - Another object (not target vehicle) occluded pedestrian

- [ ] **Integration**
  - Update `scripts/run_pipeline.py`:
    - Run detection → tracking → risk → impact detection
    - Output impact events

### Deliverables

| File | Description |
|------|-------------|
| `src/impact/impact_detector.py` | Main impact detection logic |
| `src/impact/signals.py` | Signal extractors |
| `src/impact/vehicle_association.py` | Multi-vehicle resolution |
| `tests/test_impact/` | Impact detection tests |

### Acceptance Criteria

- [ ] Pipeline detects impact candidates when pedestrian and vehicle overlap
- [ ] Impact events include confidence score and contributing signals
- [ ] Velocity discontinuity signal fires on sudden pedestrian movement change
- [ ] Track disappearance signal fires when pedestrian track lost near vehicle
- [ ] Vehicle association correctly selects the involved vehicle when multiple are near
- [ ] Evidence frames are collected around impact time

### Stretch Goals

- [ ] Add fall detection via pose estimation
- [ ] Add vehicle braking signal (sudden vehicle velocity change)
- [ ] Tune weights via simple grid search on labeled data

---

## Session 8: VLM Escalation Integration + Prompts + Safety/Cost Controls

### Goal
Integrate VLM (Vision-Language Model) API for classifying high-risk interactions and ambiguous impact candidates, with proper cost controls, rate limiting, and fallback mechanisms.

### Scope
- Implement VLM escalation triggered by high risk tier or uncertain impact.
- Frame selection and redaction (face blurring, cropping).
- Prompt engineering for classification.
- Rate limiting, caching, timeout handling.

### Tasks

- [ ] **VLM module structure**
  - Implement `src/vlm/escalation.py`:
    - Check if escalation is needed (risk tier in trigger list, or impact confidence uncertain)
    - Select frames for VLM (configurable count, span)
    - Prepare payload and call VLM API
    - Parse response and return classification

- [ ] **Frame preparation**
  - Implement `src/vlm/redaction.py`:
    - Face detection (use simple model or blur all person bboxes except target)
    - Gaussian blur on detected faces
    - Crop to interaction region (union of involved bboxes + margin)
  - Frame encoding (base64 for API)

- [ ] **Prompt engineering**
  - Implement `src/vlm/prompts.py`:
    - Template with placeholders for context
    - Clear classification categories:
      - `NEAR_MISS`
      - `LIKELY_IMPACT`
      - `CONFIRMED_IMPACT`
      - `UNCERTAIN`
    - Request JSON response with classification, confidence, rationale

- [ ] **Cost/safety controls**
  - Implement `src/vlm/rate_limiter.py`:
    - Token bucket or sliding window rate limiter
    - Configurable max calls per minute per camera
    - Backpressure: queue or skip if limit exceeded
  - Add caching:
    - Hash frame content
    - Cache VLM results for N minutes
    - Skip API call on cache hit
  - Timeout handling:
    - Configurable timeout (default 5s)
    - On timeout: fall back to physics-only classification
    - Log timeout for monitoring

- [ ] **VLM providers**
  - Support OpenAI GPT-4 Vision as primary
  - Add provider abstraction for future providers (Anthropic, Google, etc.)
  - Configure via:
    ```yaml
    vlm:
      provider: openai
      model: gpt-4-vision-preview
      api_key_env: OPENAI_API_KEY
      timeout_sec: 5.0
      rate_limit_per_minute: 10
    ```

- [ ] **Integration**
  - Update pipeline to call VLM on:
    - Risk tier `high` or `critical`
    - Impact confidence between 0.4 and 0.8 (uncertain)
  - Merge VLM result into output

### Deliverables

| File | Description |
|------|-------------|
| `src/vlm/escalation.py` | Main escalation logic |
| `src/vlm/redaction.py` | Face blur and cropping |
| `src/vlm/prompts.py` | Prompt templates |
| `src/vlm/rate_limiter.py` | Rate limiting |
| `tests/test_vlm/` | VLM module tests (mock API) |

### Acceptance Criteria

- [ ] VLM is called only when trigger conditions are met (not on every frame)
- [ ] Frames sent to VLM have faces blurred
- [ ] Frames are cropped to interaction region
- [ ] Rate limiter correctly throttles requests
- [ ] Cache prevents duplicate API calls for same frames
- [ ] Timeout results in graceful fallback
- [ ] VLM response is parsed and included in event output

### Stretch Goals

- [ ] Add batching: send multiple escalations in one API call if supported
- [ ] Implement cost tracking (estimate tokens, log cost per call)
- [ ] Add prompt A/B testing framework

---

## Session 9: LPR Pipeline + Best-Frame Selection + Aggregation

### Goal
Implement license plate recognition that activates on confirmed or high-confidence impacts, selects the best frames for plate visibility, runs detection and OCR, and aggregates results across frames.

### Scope
- Frame selection based on plate visibility criteria.
- Plate detection on vehicle ROI.
- OCR on plate crops.
- Multi-frame aggregation with confidence weighting.

### Tasks

- [ ] **Frame selection**
  - Implement `src/lpr/frame_selector.py`:
    - Given vehicle track and impact event, score frames by:
      - Vehicle bbox size (larger = better resolution)
      - Estimated plate visibility (rear/front of vehicle facing camera)
      - Motion blur (compute Laplacian variance)
      - Temporal proximity to impact (prefer frames just before/after)
    - Return top N frames for LPR

- [ ] **Plate detection**
  - Implement `src/lpr/plate_detector.py`:
    - Crop vehicle ROI with margin
    - Run plate detection model (YOLO-based or specialized)
    - Return plate bbox within vehicle crop
    - Handle cases: no plate found, multiple plates (pick largest/most confident)

- [ ] **OCR**
  - Implement `src/lpr/ocr.py`:
    - Crop plate region
    - Run OCR (PaddleOCR or EasyOCR)
    - Return text candidates with confidence
    - Apply character normalization (0/O, 1/I disambiguation)

- [ ] **Aggregation**
  - Implement `src/lpr/aggregator.py`:
    - Collect OCR results from multiple frames
    - Alignment: handle different text lengths
    - Voting:
      - Character-level majority vote
      - Confidence-weighted averaging
    - Output final plate text with confidence
    - Return "UNKNOWN" if confidence below threshold or too much disagreement

- [ ] **PlateResult output**
  - Define in `src/lpr/__init__.py`:
    ```python
    @dataclass
    class PlateResult:
        plate_text: str
        plate_confidence: float
        plate_bbox: List[int]
        best_frame_id: int
        evidence_frame_ids: List[int]
        evidence_timestamps: List[float]
        ocr_candidates: List[Dict]  # per-frame results
    ```

- [ ] **Integration**
  - Update pipeline:
    - On impact event with confidence > threshold, trigger LPR
    - Add PlateResult to ImpactEvent output

- [ ] **Tests**
  - Add `tests/test_lpr/`:
    - Test frame selector ranking
    - Test aggregator voting logic
    - Test OCR with mock results

### Deliverables

| File | Description |
|------|-------------|
| `src/lpr/frame_selector.py` | Best-frame selection |
| `src/lpr/plate_detector.py` | Plate detection |
| `src/lpr/ocr.py` | OCR wrapper |
| `src/lpr/aggregator.py` | Multi-frame aggregation |
| `tests/test_lpr/` | LPR unit tests |

### Acceptance Criteria

- [ ] LPR triggers only on confirmed/high-confidence impacts
- [ ] Frame selector correctly ranks frames by visibility criteria
- [ ] Plate detector finds plates in vehicle ROI
- [ ] OCR extracts text from plate crops
- [ ] Aggregator produces consensus plate text from multiple frames
- [ ] Output includes plate_text, confidence, and evidence frames
- [ ] "UNKNOWN" returned when plate cannot be read with sufficient confidence

### Stretch Goals

- [ ] Add plate format validation (regex for state/country patterns)
- [ ] Implement super-resolution on plate crops
- [ ] Add plate tracking across frames for better aggregation

---

## Session 10: End-to-End Integration + Metrics + Hardening + Demo

### Goal
Integrate all modules into a complete end-to-end system, implement comprehensive metrics and monitoring, harden error handling, and create a demo showcasing the full pipeline.

### Scope
- Full pipeline integration with all modules.
- Comprehensive metrics collection and logging.
- Error handling and recovery.
- Demo script and visualization.
- Documentation finalization.

### Tasks

- [ ] **End-to-end pipeline**
  - Implement `src/pipeline.py`:
    - Orchestrate all modules:
      1. Ingest → frames
      2. Detection → detections
      3. Tracking → tracks
      4. Risk scoring → risk assessments
      5. Impact detection → impact events
      6. VLM escalation → classifications
      7. LPR → plate results
      8. Output → alerts/events
    - Handle module failures gracefully
    - Support both streaming and batch modes

- [ ] **Output module**
  - Implement `src/output/alerter.py`:
    - Generate alerts for risk tiers
    - Generate event records for impacts
  - Implement `src/output/webhook.py`:
    - POST events to configured webhook URL
    - Retry logic with exponential backoff
  - Implement `src/output/file_writer.py`:
    - Write events to JSON files
    - Write evidence frames to disk

- [ ] **Metrics and monitoring**
  - Implement `src/utils/metrics.py`:
    - Track:
      - Frame processing latency
      - Detection count per class
      - Active track count
      - Risk tier distribution
      - Impact event count
      - VLM call count/latency
      - LPR success rate
    - Expose metrics endpoint (optional: Prometheus format)
  - Implement `src/utils/logging.py`:
    - Structured logging (JSON format)
    - Log levels: DEBUG, INFO, WARNING, ERROR
    - Include camera_id, frame_id in context

- [ ] **Error handling and recovery**
  - Handle:
    - Video source disconnection → reconnect with backoff
    - Detection model failure → skip frame, log error
    - Tracking failure → reset tracker
    - VLM timeout → use fallback
    - LPR failure → return UNKNOWN
  - Circuit breaker for external services (VLM API)

- [ ] **Demo script**
  - Create `scripts/demo.py`:
    - Run pipeline on sample video
    - Visualize:
      - Bounding boxes with track IDs
      - Risk tier overlay (color-coded)
      - Impact event markers
      - Plate text overlay
    - Output annotated video or display live

- [ ] **Evaluation update**
  - Update `scripts/evaluate.py`:
    - Add impact detection metrics:
      - Event precision/recall with ±0.5s tolerance
    - Add LPR metrics:
      - Exact match accuracy
      - Confidence calibration
      - Unreadable rate
    - Generate evaluation report (Markdown or HTML)

- [ ] **Documentation**
  - Finalize `README.md` with actual usage examples
  - Add `docs/api.md` with module interfaces
  - Add `docs/architecture.md` with detailed diagrams
  - Update docstrings throughout codebase

- [ ] **Configuration validation**
  - Add config schema validation
  - Fail fast on invalid config with clear error messages

### Deliverables

| File | Description |
|------|-------------|
| `src/pipeline.py` | Main orchestrator |
| `src/output/alerter.py` | Alert generation |
| `src/output/webhook.py` | Webhook integration |
| `src/output/file_writer.py` | File output |
| `src/utils/metrics.py` | Metrics collection |
| `src/utils/logging.py` | Structured logging |
| `scripts/demo.py` | Visualization demo |
| `docs/api.md` | API documentation |
| `docs/architecture.md` | Architecture docs |

### Acceptance Criteria

- [ ] Full pipeline runs end-to-end on sample video without crashing
- [ ] Risk assessments, impact events, and plate results are all produced
- [ ] Output is written to configured destination (file/webhook)
- [ ] Metrics are collected and can be queried
- [ ] Logs are structured and include relevant context
- [ ] Demo script produces annotated visualization
- [ ] Evaluation script produces complete metrics report
- [ ] Documentation is complete and accurate

### Stretch Goals

- [ ] Add real-time dashboard (Grafana or simple web UI)
- [ ] Implement A/B testing for risk thresholds
- [ ] Add Kubernetes deployment manifests
- [ ] Create Docker Compose for local multi-component testing

---

## Session Summary

After completing these 7 sessions, the system will have:

1. **Robust ingest** supporting VOD and RTSP with PTZ awareness
2. **Accurate detection** using YOLOv8 for pedestrians and vehicles
3. **Stable tracking** with ByteTrack and PTZ motion handling
4. **Physics-based risk scoring** with configurable thresholds
5. **Impact detection** using heuristic signals
6. **VLM escalation** with cost controls
7. **License plate recognition** triggered on impacts
8. **Complete pipeline** with monitoring and evaluation

### Dependencies Between Sessions

```
Session 4 (Ingest + Detect)
    │
    ▼
Session 5 (Tracking)
    │
    ▼
Session 6 (Risk Scoring) ──────────────┐
    │                                  │
    ▼                                  ▼
Session 7 (Impact Detection) ──► Session 8 (VLM)
    │                                  │
    ▼                                  │
Session 9 (LPR) ◄──────────────────────┘
    │
    ▼
Session 10 (Integration)
```

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| VLM API costs | Rate limiting, caching, trigger only on high-risk |
| PTZ complexity | Start with simple reset, add compensation in v2 |
| LPR accuracy | Multi-frame aggregation, confidence thresholds |
| Latency | Two-pass architecture, async VLM, edge optimization |
| False positives | Conservative thresholds, VLM verification |

### Post-Session 10 Roadmap

- v2: Monocular depth integration for improved TTC
- v2: Scene context (lanes, crosswalks) for risk refinement
- v2: Re-ID for track recovery across occlusions
- v2: Pose estimation for fall detection
- v2: Multi-camera tracking

