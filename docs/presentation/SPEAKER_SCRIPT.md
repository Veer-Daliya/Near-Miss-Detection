# Speaker Script: Near-Miss Detection Presentation

> **Total Presentation Time:** ~15-20 minutes (excluding Q&A)
> **Target Audience:** College admissions, technical reviewers, AI/ML researchers

---

## SLIDE 1: Title Slide (30 seconds)

**What to say:**

"Good morning/afternoon. My name is Veer Jhaveri, and today I'm excited to present my research on real-time pedestrian-vehicle collision risk assessment.

This project combines state-of-the-art deep learning with classical physics to create a system that can actually predict when a pedestrian and vehicle might collide—before it happens.

Let me take you through how we built this system and what we achieved."

**Key points to emphasize:**
- Your personal investment in the project
- The practical, life-saving application

---

## SLIDE 2: Agenda (15 seconds)

**What to say:**

"Here's our roadmap for today. I'll start with why this problem matters, explain the technical challenges we faced, walk you through our solution architecture, highlight our key innovations, show you our results, and then discuss future directions.

Let's begin with why this work is important."

---

## SLIDE 3: The Problem - Pedestrian Safety Crisis (45 seconds)

**What to say:**

"Let's start with some sobering statistics. The World Health Organization reports 1.35 million road traffic deaths every year—that's more than 3,700 people dying every single day.

Pedestrians account for 23% of these fatalities. That's over 300,000 people per year.

Current traffic monitoring systems have fundamental limitations:
- Human operators simply cannot monitor dozens of camera feeds 24/7
- Most systems are reactive—they record incidents but don't prevent them
- There's no early warning capability to alert drivers or pedestrians

This is the gap our research addresses."

**Pause for effect after the statistics.**

---

## SLIDE 4: Our Vision - Proactive Collision Prevention (30 seconds)

**What to say:**

"Our vision is to transform traffic monitoring from reactive to proactive.

The pipeline works like this: we detect pedestrians and vehicles in every frame, track them across time to understand their trajectories, predict their future positions using physics, and generate alerts before a collision can occur.

Our goal is to provide 1.5 to 3 seconds of advance warning. Why these numbers? At 30 mph, a car travels about 40 feet per second. A 2-second warning gives enough time for braking or evasive maneuvers.

This is the difference between recording an accident and preventing one."

**Gesture to the pipeline diagram as you explain each stage.**

---

## SLIDE 5: Technical Challenges (45 seconds)

**What to say:**

"Building this system required solving four key technical challenges:

**First, real-time performance.** Early warning is useless if it comes late. We need to process at least 10 frames per second with sub-100ms latency, and we want to run on consumer hardware, not expensive data center GPUs.

**Second, stable tracking.** We need to maintain consistent identities for each person and vehicle across frames, even when they're temporarily hidden behind other objects.

**Third, metric-space prediction.** To predict if two objects will collide, we need to know their real-world positions in meters, not just pixel coordinates. And crucially, we don't have camera calibration—we can't ask every traffic camera operator for their lens specifications.

**Fourth, false positive control.** Not every close proximity is dangerous. A passenger visible through a car window shouldn't trigger an alert.

Let me show you how we addressed each of these."

---

## SLIDE 6: System Architecture Overview (45 seconds)

**What to say:**

"Here's our complete system architecture. It's an 8-module pipeline that processes video in real-time.

Starting from the top left:
- **Video Ingestion** handles RTSP streams, video files, or webcam feeds
- **YOLOv10 Detection** identifies pedestrians and vehicles with 94% accuracy
- **ByteTrack** assigns stable IDs with less than 5% identity switches
- **Spatial Filtering** removes false positives like passengers in vehicles

Then in the second row:
- **Ground Plane Estimation** converts image coordinates to real-world meters
- **Trajectory Analysis** computes velocities and predicts future positions
- **CPA Risk Scoring** calculates time-to-collision
- **License Plate Recognition** identifies vehicles involved in incidents

Finally, the **Output module** generates alerts and evidence.

Let me dive deeper into the key innovations."

**Point to each module as you describe it.**

---

## SLIDE 7: Key Innovation #1 - Ground Plane Estimation (60 seconds)

**What to say:**

"Our first key innovation addresses the camera calibration problem.

To predict collisions, we need real-world distances in meters, not pixels. Traditional approaches require knowing camera parameters—focal length, mounting height, orientation. But traffic cameras rarely come with this documentation.

Our solution is a three-method cascade:

**First**, we try **Lane-Based Estimation**. If we can detect lane markings in the image, we compute where the parallel lane lines converge—the vanishing point. This gives us enough geometric information to build a homography matrix that transforms image coordinates to a bird's-eye view ground plane. This is the most accurate method, but only works 62% of the time when lanes are visible.

**If that fails**, we try **Horizon Detection**. Finding the horizon line tells us the camera's pitch angle, which constrains the geometry. This works 78% of the time.

**As a final fallback**, we use **Size-Based Estimation**. We know the average pedestrian is 1.7 meters tall. By measuring how big a pedestrian appears in pixels, we can estimate their distance from the camera. This always works, though with less precision.

The cascade achieves 100% coverage with a mean position error of just 1.1 meters—good enough for collision prediction."

**This is a key differentiator—spend time here.**

---

## SLIDE 8: Key Innovation #2 - Physics-Based Collision Prediction (60 seconds)

**What to say:**

"Our second innovation is using physics-based collision prediction with the Closest Point of Approach algorithm.

This is a classical algorithm from maritime and aviation collision avoidance, which we've adapted for pedestrian-vehicle interactions.

The math is elegant. Given the positions and velocities of a pedestrian and vehicle, we can compute exactly when they'll be closest to each other, and how close they'll get.

The diagram shows this visually. The pedestrian is moving up and to the right, the vehicle is moving down and to the left. The CPA is where they'll be closest. If that distance is small and happening soon, we have a collision risk.

Why use physics instead of deep learning? Three reasons:

**First, interpretability.** We can explain exactly why an alert was generated: 'These two objects will be 0.5 meters apart in 1.2 seconds.'

**Second, no training data required.** We don't need thousands of labeled collision videos.

**Third, guaranteed behavior.** The physics model won't suddenly fail in a weird edge case—its limitations are well-understood.

We classify risks into three tiers shown in the table: Critical for imminent collisions, Warning for concerning proximity, and Safe otherwise."

---

## SLIDE 9: Key Innovation #3 - Multi-Frame License Plate Aggregation (45 seconds)

**What to say:**

"Our third innovation improves license plate recognition reliability.

When a collision event is detected, we need to identify the vehicle. But single-frame OCR only achieves about 71% accuracy. Motion blur, partial occlusions, and lighting variations all cause errors. And one wrong character means identifying the wrong vehicle.

Our solution is multi-frame aggregation. Instead of reading the plate once, we read it across multiple frames and use confidence-weighted voting to determine each character.

The diagram shows an example. Frame 1 reads the fourth character as '1', but frames 2, 3, and 4 all read it as 'I'. Our voting algorithm produces 'ABCI234' as the consensus.

We require at least 3 frames to agree before reporting a result. This reduces false positives from single-frame OCR errors.

The result: we improved accuracy from 71% to nearly 90%. That's a 60% reduction in errors."

---

## SLIDE 10: Results - System Performance (45 seconds)

**What to say:**

"Let me share our results.

On detection and tracking: YOLOv10 achieves 94.2% mAP for pedestrian and vehicle detection. ByteTrack maintains stable IDs with less than 5% identity switches even through occlusions. Our spatial filtering eliminates 99% of false positives from passengers inside vehicles.

For collision prediction: We achieve 87% precision and 92% recall on critical risk events. That means we catch 92% of actual near-misses while keeping false alarms low. The warning tier has slightly lower precision, which is acceptable since it's meant as an early heads-up.

On the right, you can see our latency breakdown. Detection takes about 75ms, tracking adds 25ms, and risk scoring is very fast at 8ms. The total pipeline runs at over 10 frames per second, meeting our real-time requirement.

License plate recognition is slower at 400ms, but it only runs on-demand when we detect a collision event."

**Point to specific numbers as you mention them.**

---

## SLIDE 11: Results - Early Warning Capability (30 seconds)

**What to say:**

"This graph shows our early warning capability in action.

The x-axis is time before a potential collision, with zero being the collision point. The y-axis shows the risk level our system detects.

As a pedestrian and vehicle approach each other, our system first triggers a Warning alert at about 3 seconds before collision. This escalates to Critical at 1.5 seconds.

That 1.5 to 3 second warning window is exactly what we targeted. At 30 mph, 2 seconds gives a driver time to brake and reduce speed by over 20 mph before impact.

This is the difference between a fatality and a survivable incident—or avoiding the collision entirely."

**Let this sink in—it's the core value proposition.**

---

## SLIDE 12: System in Action / Demo (60 seconds)

**If showing a video demo:**

"Let me show you the system in action with a brief demo.

Watch the top of the screen—you'll see pedestrians labeled with green boxes and vehicles in blue. The track IDs remain stable as they move.

Now watch what happens as the vehicle approaches the pedestrian... there's the Warning alert. The system detected that based on their current trajectories, they'll be dangerously close in 2.3 seconds.

You can see the risk tier updating in real-time. And when they get closer... Critical alert at 1.5 seconds."

**If no video available:**

"The system provides several key visualizations:
- Bounding boxes with persistent track IDs
- Color-coded risk tier overlays
- Trajectory prediction lines showing where objects are heading
- License plate readouts for identified vehicles

The annotated video output can be used for post-incident review or real-time monitoring."

---

## SLIDE 13: Architecture Benefits (30 seconds)

**What to say:**

"Beyond the algorithms, we designed the architecture for real-world deployment.

**Modularity**: Each component can be tested and upgraded independently. If a better detector comes out next year, we can swap it in without touching the risk scoring.

**Flexibility**: The same codebase supports fixed cameras with stable geometry and PTZ cameras with moving viewpoints. Configuration is YAML-based, so operators can tune thresholds without code changes.

**Robustness**: Every critical path has fallbacks. If lane detection fails, we try horizon detection. If that fails, we use size-based estimation. The system never gives up.

**Extensibility**: We've built hooks for future enhancements like vision-language model verification and monocular depth estimation.

This is production-ready code with GPU acceleration and streaming output."

---

## SLIDE 14: Future Work (30 seconds)

**What to say:**

"Looking ahead, we have several planned extensions.

In **Phase 2**, we'll add impact detection—recognizing when a collision has actually occurred by analyzing velocity discontinuities, fall-like motion patterns, and sudden track losses.

**Phase 3** integrates Vision-Language Models. When the physics model flags a high-risk event, we can send frames to a VLM for semantic verification. 'Is this person about to be hit, or are they just walking near a parked car?'

**Phase 4** explores learning-based prediction. Our current physics model assumes constant velocity. Social force models and intent prediction could anticipate evasive actions.

Longer-term research includes monocular depth estimation for better 3D understanding, cross-camera tracking for wide-area coverage, and scaling to thousands of cameras."

---

## SLIDE 15: Conclusion (45 seconds)

**What to say:**

"To summarize: we've built NearMiss, a complete real-time system for pedestrian-vehicle collision risk assessment.

Our key contributions are:
1. A novel ground plane estimation cascade that works without camera calibration
2. Physics-based collision prediction using the Closest Point of Approach algorithm
3. Multi-frame OCR aggregation that dramatically improves license plate recognition

The results speak for themselves: we provide 1.5 to 3 seconds of advance warning, achieve 89% F1 score on critical risk detection, run in real-time at 10+ FPS, and identify vehicles with 90% accuracy.

But beyond the numbers, this work is about saving lives. Every year, over 300,000 pedestrians die in traffic accidents worldwide. If systems like this can prevent even a fraction of those deaths, we've made a meaningful difference.

I'm happy to take any questions."

**End with confidence and make eye contact.**

---

## SLIDE 16: Questions (Open-ended)

**Prepared Answers for Anticipated Questions:**

### Q: "How does this compare to Tesla Autopilot or other ADAS systems?"

"Great question. ADAS systems like Tesla's are onboard the vehicle with access to multiple sensors—cameras, radar, ultrasonics. Our system works with existing infrastructure cameras with no vehicle modification. They're complementary: ADAS protects the vehicle's occupants and immediate surroundings; our system provides city-wide monitoring."

### Q: "What happens with unusual weather or lighting?"

"YOLOv10 is trained on diverse conditions, so detection remains robust. However, extreme conditions like heavy fog or complete darkness would degrade performance. We're exploring infrared camera support for nighttime operation."

### Q: "Why physics-based instead of deep learning for prediction?"

"Two reasons: interpretability and data requirements. Deep learning trajectory predictors like Social-GAN need thousands of labeled trajectories. Our physics approach works immediately and its failure modes are well-understood. We can always add learning-based components later for complex scenes."

### Q: "What's the deployment cost?"

"The software runs on a single GPU. An RTX 3080 costs around $700 and can process 10+ cameras. Compared to hiring human monitors, the ROI is significant."

### Q: "What was the hardest part of this project?"

"The ground plane estimation cascade was the most challenging. We went through several iterations before finding a robust solution that works across different camera angles and scene types. It required combining classical computer vision (vanishing points, homography) with practical engineering (fallbacks, caching, temporal smoothing)."

### Q: "What did you learn from this project?"

"I learned that real-world systems require much more than just the core algorithm. You need fallbacks, error handling, performance optimization, and careful API design. I also gained deep appreciation for how classical techniques (like CPA from maritime navigation) can complement modern deep learning."

---

## Presentation Tips

1. **Pace yourself.** You have plenty of material. Don't rush.

2. **Make eye contact.** Look at different parts of the audience, not just the screen.

3. **Use the diagrams.** Point to specific parts of architecture and graphs as you explain.

4. **Pause after key statistics.** Let the 1.35 million deaths number sink in.

5. **Show enthusiasm** when discussing innovations. This is YOUR work—be proud of it.

6. **Handle unknowns gracefully.** "That's a great question. I haven't explored that specifically, but based on our architecture, I would approach it by..."

7. **End strong.** The conclusion should feel like a natural landing, not an abrupt stop.

---

## Technical Details for Deep Questions

If asked about implementation specifics:

- **YOLOv10**: Uses CSPNet backbone, efficient attention, ~50-100ms inference on RTX 3080
- **ByteTrack**: Two-stage IoU matching, 30-frame track buffer, no appearance features needed
- **CPA Formula**: t_CPA = -(r · w) / |w|², where r is relative position, w is relative velocity
- **Homography**: 3x3 matrix mapping image pixels to ground plane meters
- **EMA Smoothing**: α = 0.3, applied to velocity estimates and ground plane parameters
- **LPR**: PaddleOCR for detection, confidence-weighted character voting across 3+ frames

---

*Good luck with your presentation!*
