# Running the Detection System

This guide covers all the ways to run the detection pipeline, from quick tests to production configurations.

## Quick Start

### Basic Run (Recommended for First Time)

Processes everything with default settings:

```bash
python scripts/run_detection.py --source data/samples/traffic_video.mp4
```

**What it does:**
- ✅ Processes ALL frames
- ✅ Detects plates on EVERY frame
- ✅ Uses YOLOv10 medium model
- ✅ Uses ByteTracker (stable IDs)
- ✅ Multi-frame aggregation enabled
- ✅ Saves annotated video + JSON

**Best for:** First run, testing, when you want complete results

## Running Methods

### Option 1: Shell Script (Easiest - Recommended)

**Basic Usage:**
```bash
./run.sh
```

This will:
- ✅ Create virtual environment (if needed)
- ✅ Install all dependencies (if needed)
- ✅ Run detection with default settings
- ✅ Use video at `data/samples/traffic_video.mp4`

**Custom Video File:**
```bash
./run.sh /path/to/your/video.mp4
```

**Custom Settings:**
```bash
# Set environment variables before running
export MODEL_SIZE="n"      # n, s, m, l, x
export FPS="5"             # Target FPS
export PLATE_INTERVAL="5"  # Process plates every N frames

./run.sh
```

### Option 2: Python Script

**Basic Usage:**
```bash
python3 run.py
```

**Custom Video File:**
```bash
python3 run.py /path/to/your/video.mp4
```

**Custom Settings:**
```bash
python3 run.py data/samples/traffic_video.mp4 s 10 3
# Arguments: [video_path] [model_size] [fps] [plate_interval]
```

### Option 3: Direct Command

**Basic Detection (All Defaults):**
```bash
python scripts/run_detection.py --source data/samples/traffic_video.mp4
```

**With Custom Settings:**
```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --model-size s \
  --fps 10 \
  --plate-interval 3
```

## Run Configurations

### Fast Processing (Good for Testing)

Process fewer frames for faster results:

```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --fps 5 \
  --plate-interval 5
```

**What it does:**
- Processes every 5th frame (5x faster)
- Detects plates every 5th frame
- Still uses tracking and aggregation

**Best for:** Quick testing, long videos, when speed matters

### Maximum Speed (Fastest)

Process as fast as possible:

```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --model-size n \
  --fps 2 \
  --plate-interval 10 \
  --no-annotated
```

**What it does:**
- Uses nano model (fastest)
- Processes every 2nd frame
- Plates every 10th frame
- No video output (JSON only)

**Estimated time: 3-5 minutes** (instead of 30+ minutes)

**Best for:** Very long videos, quick testing, when you only need data

### Maximum Accuracy (Best Results)

Highest quality detection:

```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --model-size l \
  --confidence 0.5 \
  --plate-interval 1
```

**What it does:**
- Uses large model (most accurate)
- Higher confidence threshold
- Plates on every frame
- Slower but best quality

**Best for:** Important videos, when accuracy is critical

### Balanced (Recommended for Production)

Good balance of speed and accuracy:

```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --model-size s \
  --fps 10 \
  --plate-interval 3
```

**What it does:**
- Small model (fast)
- Processes 10 FPS
- Plates every 3rd frame
- Good balance

**Best for:** Production use, regular processing

### JSON Only (No Video)

Skip video encoding for faster processing:

```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --no-annotated
```

**What it does:**
- Processes all frames
- Saves JSON results only
- No video output (faster)

**Best for:** When you only need data, batch processing

### Per-Frame Results (No Aggregation)

Get individual frame results without aggregation:

```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --no-aggregation
```

**What it does:**
- Processes all frames
- No multi-frame aggregation
- Each frame has independent results

**Best for:** Debugging, when you want per-frame accuracy

## Complete Parameter Reference

### Required
- `--source` - Video file path, RTSP URL, or webcam ID (e.g., "0")

### Optional Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--output-dir` | `data/outputs` | Any directory | Where to save results |
| `--model-size` | `m` | `n`, `s`, `m`, `l`, `x` | Model size (n=fastest, x=most accurate) |
| `--confidence` | `0.4` | `0.0-1.0` | Detection confidence threshold |
| `--fps` | `None` (all frames) | Any number | Target FPS (lower = faster) |
| `--plate-interval` | `1` (every frame) | Any number | Process plates every N frames |
| `--no-annotated` | False | Flag | Don't save annotated video |
| `--no-json` | False | Flag | Don't save JSON results |
| `--no-aggregation` | False | Flag | Disable multi-frame aggregation |

## Model Size Comparison

| Size | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| `n` (nano) | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Fast testing, real-time |
| `s` (small) | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced, production |
| `m` (medium) | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Default, good balance |
| `l` (large) | ⚡⚡ | ⭐⭐⭐⭐⭐⭐ | High accuracy needed |
| `x` (xlarge) | ⚡ | ⭐⭐⭐⭐⭐⭐⭐ | Maximum accuracy |

## Recommended Configurations

### 🚀 Quick Test (2-3 minutes)
```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --model-size n \
  --fps 5 \
  --plate-interval 10 \
  --no-annotated
```

### ⚖️ Balanced (10-15 minutes)
```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --model-size s \
  --fps 10 \
  --plate-interval 3
```

### 🎯 Full Quality (20-40 minutes)
```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --model-size m \
  --confidence 0.4
```

### 🔬 Maximum Accuracy (30-60 minutes)
```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --model-size l \
  --confidence 0.5 \
  --plate-interval 1
```

## Common Use Cases

### "I want to test quickly"
```bash
python scripts/run_detection.py --source data/samples/traffic_video.mp4 --model-size n --fps 5 --no-annotated
```

### "I want the best results"
```bash
python scripts/run_detection.py --source data/samples/traffic_video.mp4 --model-size l --confidence 0.5
```

### "I want to process faster"
```bash
python scripts/run_detection.py --source data/samples/traffic_video.mp4 --fps 5 --plate-interval 5
```

### "I only need the data, not the video"
```bash
python scripts/run_detection.py --source data/samples/traffic_video.mp4 --no-annotated
```

### "I want to see per-frame results"
```bash
python scripts/run_detection.py --source data/samples/traffic_video.mp4 --no-aggregation
```

## Input Sources

### Video File
```bash
python scripts/run_detection.py --source data/samples/traffic_video.mp4
```

### Different Video Location
```bash
# Absolute path
python scripts/run_detection.py --source "/path/to/your/video.mp4"

# Or copy it first
cp /path/to/your/video.mp4 data/samples/
python scripts/run_detection.py --source data/samples/your_video.mp4
```

### RTSP Camera Stream
```bash
python scripts/run_detection.py --source rtsp://192.168.1.100:554/stream1
```

### Webcam (Live Processing)
```bash
python scripts/run_detection.py --source 0
```

## Output Files

After running, check `data/outputs/`:

1. **annotated_output.mp4**
   - Video with bounding boxes and labels
   - Shows: Vehicle type, track ID, license plate text
   - Colors: Green (high conf plate), Yellow (low conf), Red (no plate)

2. **detections.json**
   - All detection results in JSON format
   - Frame-by-frame data
   - Includes: bboxes, classes, track IDs, plate text

View results:
```bash
open data/outputs/annotated_output.mp4
```

## What Happens When You Run

1. **Checks virtual environment** - Creates if needed
2. **Checks dependencies** - Installs if missing
3. **Checks video file** - Verifies it exists
4. **Runs detection** - Processes video with AI models
5. **Saves results** - Creates annotated video and JSON

## Expected Output

You'll see progress indicators:

```
Initializing detection models...
YOLO detector using device: mps
ByteTracker initialized (using YOLO built-in tracking)
Plate detector initialized (PaddleOCR)
OCR initialized (PaddleOCR)
Multi-frame aggregation enabled
Opening video source: data/samples/traffic_video.mp4
Processing frames...
Processing frames (Frame 1, 15 detections):   0%|          | 1/300 [00:00<02:30]
```

After completion:
```
Processing frames (Frame 300, 12 detections): 100%|██████████| 300/300 [02:30<00:00]

Results saved to data/outputs/detections.json
Annotated video saved to data/outputs/annotated_output.mp4

Processing complete!
Total frames processed: 300
Total detections: 5372
Total plates detected: 5
```

## Tips

1. **First time?** Use the basic command
2. **Long video?** Use `--fps 5` to process faster
3. **Need accuracy?** Use `--model-size l` or `x`
4. **Testing?** Use `--no-annotated` to skip video encoding
5. **Crowded scene?** Lower `--confidence` to 0.3
6. **Clean results?** Raise `--confidence` to 0.6

## Troubleshooting

### Script Not Executable
```bash
chmod +x run.sh
chmod +x run.py
```

### Permission Denied
```bash
# Use python3 explicitly
python3 run.py
```

### Video Not Found
```bash
# Provide full path
./run.sh /full/path/to/video.mp4
```

### Dependencies Not Installing
```bash
# Activate venv first
source venv/bin/activate
pip install -r requirements.txt
pip install paddlepaddle paddleocr
pip install lap
```

## Quick Reference Card

```bash
# BASIC
python scripts/run_detection.py --source data/samples/traffic_video.mp4

# FAST
python scripts/run_detection.py --source data/samples/traffic_video.mp4 --model-size n --fps 5

# ACCURATE
python scripts/run_detection.py --source data/samples/traffic_video.mp4 --model-size l --confidence 0.5

# JSON ONLY
python scripts/run_detection.py --source data/samples/traffic_video.mp4 --no-annotated

# CUSTOM OUTPUT
python scripts/run_detection.py --source data/samples/traffic_video.mp4 --output-dir my_results
```




