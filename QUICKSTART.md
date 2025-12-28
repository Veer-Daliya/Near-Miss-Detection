# Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install OCR (choose one):**
```bash
# Option 1: PaddleOCR (recommended)
pip install paddlepaddle paddleocr

# Option 2: EasyOCR (alternative)
pip install easyocr
```

## Using Your Downloaded Video

### Step 1: Place your video in the project

Move your downloaded video to the `data/samples/` directory:

```bash
# Example: If your video is in Downloads
mv ~/Downloads/your_traffic_video.mp4 data/samples/

# Or copy it
cp ~/Downloads/your_traffic_video.mp4 data/samples/
```

### Step 2: Run detection

```bash
python scripts/run_detection.py --source data/samples/your_traffic_video.mp4
```

### Step 3: Check results

Results will be saved in `data/outputs/`:
- `annotated_output.mp4` - Video with bounding boxes and labels
- `detections.json` - All detection results in JSON format

## Command Line Options

```bash
python scripts/run_detection.py --source VIDEO_PATH [OPTIONS]

Required:
  --source          Video file, RTSP URL, or webcam device ID

Optional:
  --output-dir      Output directory (default: data/outputs)
  --model-size      YOLO model size: n, s, m, l, x (default: m)
  --confidence      Detection confidence threshold 0.0-1.0 (default: 0.4)
  --fps             Target FPS for processing (default: 10)
  --no-annotated    Don't save annotated video
  --no-json         Don't save JSON results
```

## Examples

### Process a video file:
```bash
python scripts/run_detection.py --source data/samples/traffic.mp4
```

### Process with custom settings:
```bash
python scripts/run_detection.py \
  --source data/samples/traffic.mp4 \
  --model-size s \
  --confidence 0.5 \
  --fps 15
```

### Process RTSP stream:
```bash
python scripts/run_detection.py --source rtsp://192.168.1.100:554/stream1
```

### Process webcam:
```bash
python scripts/run_detection.py --source 0
```

## Output Format

The JSON output contains:
- Frame-by-frame detections
- Bounding boxes for each object
- Class names (person, car, truck, bus, etc.)
- Confidence scores
- Track IDs
- License plate text (if detected)

## Troubleshooting

### OCR not working?
- Make sure PaddleOCR or EasyOCR is installed
- Check that license plates are visible and readable in the video

### Detection not working?
- Try lowering confidence threshold: `--confidence 0.3`
- Try different model size: `--model-size s` (smaller, faster)

### Video won't open?
- Check file path is correct
- Ensure video format is supported (.mp4, .avi, .mov)
- Try absolute path instead of relative

