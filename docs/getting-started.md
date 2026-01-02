# Getting Started

This guide will help you set up and run the Near-Miss Detection system.

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for real-time processing)
- FFmpeg (for video decoding)

## Installation

### Step 1: Create Virtual Environment (Recommended)

Using a virtual environment isolates dependencies and prevents conflicts with other projects.

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# On Windows, use:
# venv\Scripts\activate
```

After activation, your terminal prompt should show `(venv)`.

**Why use a virtual environment?**
- Isolates dependencies - prevents conflicts with other Python projects
- Cleaner system - doesn't pollute your global Python installation
- Reproducible - same environment every time
- Easier to manage - can delete and recreate easily

### Step 2: Install Core Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all core dependencies
pip install -r requirements.txt
```

This installs:
- opencv-python (for video processing)
- numpy (for array operations)
- ultralytics (for YOLO detection)

### Step 3: Install OCR (Choose One)

**Option 1: PaddleOCR (Recommended)**
```bash
pip install paddlepaddle paddleocr
```

**Option 2: EasyOCR (Alternative)**
```bash
pip install easyocr
```

**Note:** You only need ONE OCR engine. PaddleOCR is recommended for better accuracy.

### Step 4: Install Tracking Dependency

```bash
pip install lap
```

This package is required for ByteTracker to work properly.

### Step 5: Verify Installation

Test that everything is installed:

```bash
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
python3 -c "from ultralytics import YOLO; print('Ultralytics: OK')"
python3 -c "from paddleocr import PaddleOCR; print('PaddleOCR: OK')"
python3 -c "import lap; print('LAP: OK')"
```

## Quick Install (All at Once)

```bash
# Navigate to project directory
cd "/path/to/Near-Miss-Detection"

# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install everything
pip install --upgrade pip
pip install -r requirements.txt
pip install paddlepaddle paddleocr
pip install lap

# Verify
python3 -c "from ultralytics import YOLO; from paddleocr import PaddleOCR; import lap; print('✓ All packages installed')"
```

## Using Your Video

### Step 1: Place Your Video

Move your video to the `data/samples/` directory:

```bash
# Example: If your video is in Downloads
mv ~/Downloads/your_traffic_video.mp4 data/samples/

# Or copy it
cp ~/Downloads/your_traffic_video.mp4 data/samples/
```

### Step 2: Run Detection

```bash
# Make sure venv is activated
source venv/bin/activate

# Run detection
python scripts/run_detection.py --source data/samples/your_traffic_video.mp4
```

### Step 3: Check Results

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
  --plate-interval  Process plates every N frames (default: 1)
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

### Installation Issues

**If pip install fails:**
- Make sure you're in the virtual environment (see `(venv)` in prompt)
- Try: `pip install --upgrade pip` first
- On Mac, you might need: `pip3 install` instead of `pip install`

**If PaddleOCR installation is slow:**
- It's normal - PaddleOCR downloads model files (~500MB-1GB)
- Be patient, it only happens once

**If you get permission errors:**
- Make sure you're using a virtual environment (not system Python)
- Don't use `sudo` - use virtual environment instead
- Use `python3` explicitly: `python3 -m venv venv`

**Virtual environment not activating?**
```bash
# Make sure you're in the project directory
cd "/path/to/Near-Miss-Detection"

# Try explicit path
source ./venv/bin/activate
```

**Packages not found after activation?**
```bash
# Make sure venv is activated (you should see (venv) in prompt)
# Then reinstall
pip install -r requirements.txt
pip install lap
```

### Runtime Issues

**OCR not working?**
- Make sure PaddleOCR or EasyOCR is installed
- Check that license plates are visible and readable in the video

**Detection not working?**
- Try lowering confidence threshold: `--confidence 0.3`
- Try different model size: `--model-size s` (smaller, faster)

**Video won't open?**
- Check file path is correct
- Ensure video format is supported (.mp4, .avi, .mov)
- Try absolute path instead of relative

**Deactivate virtual environment (when done):**
```bash
deactivate
```

## Next Steps

- See [Running the System](running.md) for detailed run options and configurations
- See [Performance Tips](performance.md) for optimization guidance
- See [Troubleshooting](troubleshooting.md) for common issues

