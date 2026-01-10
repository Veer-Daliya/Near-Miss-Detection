# Setup Guide for Lenovo Laptop

This guide will help you set up the Near-Miss Detection project on your Lenovo laptop from scratch.

---

## 📋 Table of Contents

1. [Quick Overview](#quick-overview)
2. [Step-by-Step Setup](#step-by-step-setup)
3. [Models Being Used](#models-being-used)
4. [Outputs and Samples](#outputs-and-samples)
5. [Running the System](#running-the-system)
6. [Sharing Your Work (Email vs Git)](#sharing-your-work-email-vs-git)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Overview

**What this project does:**
- Detects pedestrians and vehicles in video
- Tracks objects across frames
- Extracts license plate text from vehicles
- Generates annotated videos with bounding boxes

**What you need:**
- Python 3.9 or higher
- Virtual environment (we'll create this)
- Video file to process (or use sample video)

---

## 📝 Step-by-Step Setup

### Step 1: Navigate to Project Directory

Open terminal/command prompt and go to your project folder:

```bash
cd "/Users/vdaliya27/Documents/GitHub AI Project/Near-Miss-Detection"
```

**Windows users:** Use backslashes or forward slashes:
```bash
cd "C:\Users\YourName\Documents\GitHub AI Project\Near-Miss-Detection"
```

### Step 2: Create Virtual Environment

**Why?** Virtual environments keep your project dependencies separate from other Python projects.

```bash
# Create virtual environment
python -m venv venv

# On Windows, if python doesn't work, try:
# python3 -m venv venv
# or
# py -3 -m venv venv
```

This creates a `venv` folder in your project directory.

### Step 3: Activate Virtual Environment

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

**Windows PowerShell:**
```bash
venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your terminal prompt.

### Step 4: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 5: Install Core Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` - Video processing
- `numpy` - Array operations
- `ultralytics` - YOLO object detection

### Step 6: Install OCR Engine

**For NVIDIA GPU (recommended on Lenovo with NVIDIA graphics):**
```bash
pip install paddlepaddle-gpu paddleocr
```

**For CPU-only (if no NVIDIA GPU or if GPU install fails):**
```bash
pip install paddlepaddle paddleocr
```

**Alternative OCR (if PaddleOCR doesn't work):**
```bash
pip install easyocr
```

### Step 7: Install Tracking Dependency

```bash
pip install lap
```

Required for ByteTracker to work.

### Step 8: Verify Installation

Test that everything works:

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR: OK')"
python -c "import lap; print('LAP: OK')"
```

If all commands succeed, you're ready to go!

---

## 🤖 Models Being Used

### Detection Models (YOLO)

The system uses **YOLOv8** or **YOLOv10** models for object detection. These models are automatically downloaded on first use.

**Available model sizes:**
- `yolov8n` / `yolov10n` - Nano (smallest, fastest, least accurate)
- `yolov8s` / `yolov10s` - Small (balanced)
- `yolov8m` / `yolov10m` - Medium (default, good balance) ⭐
- `yolov8l` / `yolov10l` - Large (slower, more accurate)
- `yolov8x` / `yolov10x` - Extra Large (slowest, most accurate)

**Model location:** Models are stored in `data/models/` after first download.

**What they detect:**
- `person` - Pedestrians
- `car` - Cars
- `truck` - Trucks
- `bus` - Buses
- `motorcycle` - Motorcycles
- `bicycle` - Bicycles

### OCR Models (License Plate Recognition)

**PaddleOCR** (recommended):
- Automatically downloads model files (~500MB-1GB) on first use
- Supports GPU acceleration (CUDA) for NVIDIA GPUs
- Falls back to CPU if GPU not available
- Model files stored in: `~/.paddleocr/` (hidden folder in home directory)

**EasyOCR** (alternative):
- Also downloads models on first use
- Works on CPU and GPU
- Model files stored in: `~/.EasyOCR/` (hidden folder)

**What they do:**
- Detect text regions in images
- Extract license plate text from vehicle images
- Provide confidence scores for each reading

---

## 📁 Outputs and Samples

### Sample Videos

**Location:** `data/samples/`

**Current samples:**
- `traffic_video.mp4` - Example traffic video for testing

**To add your own video:**
1. Copy your video file to `data/samples/`
2. Run detection on it (see Running section below)

**Supported formats:**
- `.mp4` (recommended)
- `.avi`
- `.mov`
- `.mkv`
- Any format supported by OpenCV

### Output Files

**Location:** `data/outputs/`

**Generated files:**

1. **`annotated_output.mp4`**
   - Video with bounding boxes drawn around detected objects
   - Shows: object type, confidence, track ID, license plate text
   - Same resolution and frame rate as input video

2. **`detections.json`**
   - Complete detection results in JSON format
   - Contains:
     - Frame-by-frame detections
     - Bounding box coordinates
     - Class names and confidence scores
     - Track IDs (for tracking objects across frames)
     - License plate text (if detected)

**Example JSON structure:**
```json
{
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
      "detections": [
        {
          "class_name": "car",
          "confidence": 0.95,
          "bbox": [100, 200, 300, 400],
          "track_id": 1,
          "plate_text": "ABC1234",
          "plate_confidence": 0.87
        }
      ]
    }
  ]
}
```

**Note:** Output files are overwritten each time you run detection. To keep multiple runs, rename them or use `--output-dir` to specify a different folder.

---

## 🎬 Running the System

### Basic Usage

**Process a video file:**
```bash
# Make sure venv is activated first!
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows

# Run detection
python scripts/run_detection.py --source data/samples/traffic_video.mp4
```

**Process your own video:**
```bash
python scripts/run_detection.py --source data/samples/your_video.mp4
```

### Command Line Options

```bash
python scripts/run_detection.py --source VIDEO_PATH [OPTIONS]

Required:
  --source          Video file path, RTSP URL, or webcam device ID (0, 1, etc.)

Optional:
  --output-dir      Where to save outputs (default: data/outputs)
  --model-size      YOLO model: n, s, m, l, x (default: m)
  --confidence      Detection threshold 0.0-1.0 (default: 0.4)
  --fps             Target FPS for processing (default: 10)
  --plate-interval  Process plates every N frames (default: 1)
  --no-annotated    Don't save annotated video
  --no-json         Don't save JSON results
```

### Examples

**Quick test with small model (faster):**
```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --model-size s \
  --confidence 0.3
```

**High accuracy (slower):**
```bash
python scripts/run_detection.py \
  --source data/samples/traffic_video.mp4 \
  --model-size l \
  --confidence 0.5
```

**Process webcam (if you have one):**
```bash
python scripts/run_detection.py --source 0
```

**Process RTSP stream:**
```bash
python scripts/run_detection.py --source rtsp://192.168.1.100:554/stream1
```

---

## 📧 Sharing Your Work (Email vs Git)

### Can You Just Email Instead of Git?

**Yes, absolutely!** You can share your work via email or any file-sharing method. Here are your options:

### Option 1: Email the Project Folder

**What to send:**
1. **The entire project folder** (zipped)
   - Right-click project folder → "Compress" (Mac) or "Send to → Compressed folder" (Windows)
   - Creates a `.zip` file
   - Email the zip file

**What to include:**
- ✅ All source code (`src/` folder)
- ✅ Configuration files (`configs/`, `requirements.txt`, etc.)
- ✅ Scripts (`scripts/` folder)
- ✅ Documentation (`docs/` folder)
- ❌ **Don't include:** `venv/` folder (too large, ~500MB-1GB)
- ❌ **Don't include:** `data/models/` (models auto-download)
- ❌ **Don't include:** `data/outputs/` (generated files)

**How to exclude files:**
- Create a `.zip` manually and exclude `venv/`, `data/models/`, `data/outputs/`
- Or use `.gitignore` patterns (see below)

### Option 2: Use File Sharing Services

**Google Drive / Dropbox / OneDrive:**
1. Upload project folder (excluding `venv/`)
2. Share link via email

**WeTransfer / SendAnywhere:**
- Good for large files
- Upload zip file, get link, email the link

### Option 3: Git (If You Want to Learn)

**If you want to use Git later:**
```bash
# Initialize git (one time)
git init

# Add files (excluding venv and outputs)
git add src/ scripts/ configs/ docs/ requirements.txt README.md

# Commit
git commit -m "My changes"

# Push to GitHub (if you have a repo)
git push origin main
```

**But you don't have to!** Email works perfectly fine for sharing code.

### What to Include When Sharing

**Include:**
- ✅ `src/` - All source code
- ✅ `scripts/` - Run scripts
- ✅ `configs/` - Configuration files
- ✅ `docs/` - Documentation
- ✅ `requirements.txt` - Dependencies list
- ✅ `README.md` - Project readme
- ✅ `SETUP_GUIDE.md` - This file!

**Exclude (too large or auto-generated):**
- ❌ `venv/` - Virtual environment (can be recreated)
- ❌ `data/models/` - YOLO models (auto-download)
- ❌ `data/outputs/` - Generated output files
- ❌ `__pycache__/` - Python cache files
- ❌ `.pyc` files - Compiled Python files

---

## 🔧 Troubleshooting

### Virtual Environment Issues

**"venv folder not created":**
```bash
# Make sure you're in the project directory
cd "/Users/vdaliya27/Documents/GitHub AI Project/Near-Miss-Detection"

# Try with python3 explicitly
python3 -m venv venv

# On Windows, try:
py -3 -m venv venv
```

**"venv not activating":**
```bash
# Mac/Linux - use full path
source ./venv/bin/activate

# Windows - use full path
.\venv\Scripts\activate

# Windows PowerShell - may need to allow scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Installation Issues

**"pip install fails":**
```bash
# Make sure venv is activated (you see (venv) in prompt)
# Upgrade pip first
pip install --upgrade pip

# Try installing one package at a time
pip install opencv-python
pip install numpy
pip install ultralytics
```

**"PaddleOCR installation is slow":**
- This is normal! PaddleOCR downloads large model files (~500MB-1GB)
- Be patient, it only happens once
- Make sure you have internet connection

**"Permission denied":**
- Don't use `sudo` - use virtual environment instead
- Make sure venv is activated
- Use `python -m pip install` instead of just `pip install`

### Runtime Issues

**"No module named 'src'":**
```bash
# Make sure you're in the project root directory
cd "/Users/vdaliya27/Documents/GitHub AI Project/Near-Miss-Detection"

# Then run the script
python scripts/run_detection.py --source data/samples/traffic_video.mp4
```

**"Video won't open":**
- Check file path is correct
- Use absolute path: `/full/path/to/video.mp4`
- Check video format is supported (.mp4, .avi, .mov)

**"Detection not working":**
- Try lowering confidence: `--confidence 0.3`
- Try smaller model: `--model-size s`
- Check video has visible objects

**"OCR not working":**
- Make sure PaddleOCR or EasyOCR is installed
- Check license plates are visible in video
- Try different video with clearer plates

### GPU Issues (NVIDIA)

**"GPU not detected":**
- Install CUDA toolkit (if NVIDIA GPU)
- Install `paddlepaddle-gpu` instead of `paddlepaddle`
- Check GPU drivers are installed

**"CUDA out of memory":**
- Use smaller model: `--model-size s` or `--model-size n`
- Process fewer frames: `--fps 5`

### Getting Help

**Check existing documentation:**
- `docs/getting-started.md` - Detailed installation guide
- `docs/troubleshooting.md` - More troubleshooting tips
- `docs/running.md` - Running options and examples

**Common solutions:**
1. Always activate venv before running: `source venv/bin/activate`
2. Make sure you're in project root directory
3. Check all dependencies are installed: `pip list`
4. Try with sample video first: `data/samples/traffic_video.mp4`

---

## ✅ Quick Checklist

Before running detection, make sure:

- [ ] Virtual environment created (`venv/` folder exists)
- [ ] Virtual environment activated (see `(venv)` in prompt)
- [ ] All dependencies installed (`pip list` shows packages)
- [ ] Sample video exists (`data/samples/traffic_video.mp4`)
- [ ] You're in project root directory
- [ ] Output directory exists (`data/outputs/`)

---

## 🎯 Next Steps

1. **Set up the environment** (follow steps above)
2. **Test with sample video:**
   ```bash
   python scripts/run_detection.py --source data/samples/traffic_video.mp4
   ```
3. **Check outputs** in `data/outputs/`
4. **Try your own video** by placing it in `data/samples/`
5. **Share your work** via email or file sharing

---

**Need help?** Check `docs/troubleshooting.md` or review the error message carefully - most issues are environment-related and easy to fix!



