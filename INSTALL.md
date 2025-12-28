# Installation Guide

## Step 1: Create Virtual Environment (Recommended)

It's best practice to use a virtual environment to avoid conflicts with other projects.

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# On Windows, use:
# venv\Scripts\activate
```

After activation, your terminal prompt should show `(venv)`.

## Step 2: Install Core Dependencies

```bash
# Install all core dependencies
pip install -r requirements.txt
```

This installs:
- opencv-python (for video processing)
- numpy (for array operations)
- ultralytics (for YOLOv8 detection)

## Step 3: Install OCR (Choose One)

### Option 1: PaddleOCR (Recommended)
```bash
pip install paddlepaddle paddleocr
```

### Option 2: EasyOCR (Alternative)
```bash
pip install easyocr
```

**Note:** You only need ONE OCR engine. PaddleOCR is recommended for better accuracy.

## Step 4: Verify Installation

Test that everything is installed:

```bash
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
python3 -c "from ultralytics import YOLO; print('Ultralytics: OK')"
python3 -c "from paddleocr import PaddleOCR; print('PaddleOCR: OK')"
```

## Troubleshooting

### If pip install fails:
- Make sure you're in the virtual environment (see `(venv)` in prompt)
- Try: `pip install --upgrade pip` first
- On Mac, you might need: `pip3 install` instead of `pip install`

### If PaddleOCR installation is slow:
- It's normal - PaddleOCR downloads model files (~500MB)
- Be patient, it only happens once

### If you get permission errors:
- Make sure you're using a virtual environment (not system Python)
- Don't use `sudo` - use virtual environment instead

## Quick Install (All at Once)

```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install everything
pip install -r requirements.txt
pip install paddlepaddle paddleocr

# Verify
python3 -c "from ultralytics import YOLO; print('Ready!')"
```

