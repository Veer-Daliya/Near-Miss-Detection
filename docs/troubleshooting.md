# Troubleshooting Guide

Common issues and their solutions.

## Tracking Issues

### Problem: Missing `lap` Package

**Symptoms:**
- Tracker failing every frame
- IDs changing per frame
- Very slow processing (~3 seconds per frame instead of milliseconds)
- Repeated warnings about missing package

**Solution:**

Install the missing dependency:

```bash
pip install lap
```

This installs the Linear Assignment Problem solver needed for ByteTracker.

**After installing, run again:**
```bash
python scripts/run_detection.py --source data/samples/traffic_video.mp4 --model-size s --fps 10 --plate-interval 3
```

**What will happen after fix:**
- ✅ Tracker will work properly
- ✅ Stable track IDs across frames
- ✅ Much faster processing (~50-100ms per frame instead of 3 seconds)
- ✅ No more repeated warnings

**Why it's slow:**
The tracker tries to install `lap` on EVERY frame (1800 times!), which is why it's so slow. Once `lap` is installed, this stops and processing becomes fast.

**Alternative: Disable Tracking Temporarily**
If you can't install `lap` right now, the code will fall back to simple ID assignment. It will work but IDs will change per frame (not ideal for your use case).

## Installation Issues

### Virtual Environment Not Activating

**Symptoms:**
- `source venv/bin/activate` doesn't work
- No `(venv)` in prompt

**Solutions:**

```bash
# Make sure you're in the project directory
cd "/path/to/Near-Miss-Detection"

# Try explicit path
source ./venv/bin/activate

# Or recreate venv
python3 -m venv venv
source venv/bin/activate
```

### Packages Not Found After Activation

**Symptoms:**
- Import errors even after installing packages
- "No module named X" errors

**Solutions:**

```bash
# Make sure venv is activated (you should see (venv) in prompt)
# Then reinstall
pip install -r requirements.txt
pip install paddlepaddle paddleocr
pip install lap
```

### Permission Errors

**Symptoms:**
- Permission denied errors during installation
- Cannot write to directories

**Solutions:**

```bash
# Use python3 explicitly
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt

# Don't use sudo - use virtual environment instead
```

### PaddleOCR Installation is Slow

**Symptoms:**
- Installation takes 3-8 minutes
- Appears to hang

**Solution:**
- This is **normal** - PaddleOCR downloads model files (~500MB-1GB)
- Be patient, it only happens once
- Check your internet connection if it's taking longer

### pip Install Fails

**Symptoms:**
- Installation errors
- Package not found

**Solutions:**

```bash
# Make sure you're in the virtual environment (see (venv) in prompt)
# Upgrade pip first
pip install --upgrade pip

# On Mac, you might need pip3 instead of pip
pip3 install -r requirements.txt
```

## Runtime Issues

### Video Won't Open

**Symptoms:**
- "Cannot open video" error
- File not found errors

**Solutions:**

```bash
# Check file path is correct
ls -lh data/samples/traffic_video.mp4

# Try absolute path instead of relative
python scripts/run_detection.py --source "/full/path/to/video.mp4"

# Ensure video format is supported (.mp4, .avi, .mov)
# Check file permissions
chmod 644 data/samples/traffic_video.mp4
```

### Detection Not Working

**Symptoms:**
- No detections found
- Very few detections

**Solutions:**

```bash
# Try lowering confidence threshold
python scripts/run_detection.py --source video.mp4 --confidence 0.3

# Try different model size (smaller is faster, larger is more accurate)
python scripts/run_detection.py --source video.mp4 --model-size s

# Check that video has visible objects
# Ensure video quality is good
```

### OCR Not Working

**Symptoms:**
- No license plates detected
- Plate text is empty or "UNKNOWN"

**Solutions:**

```bash
# Make sure PaddleOCR or EasyOCR is installed
pip install paddlepaddle paddleocr

# Check that license plates are visible and readable in the video
# Try processing more frames
python scripts/run_detection.py --source video.mp4 --plate-interval 1

# Lower confidence threshold for plate detection
```

### GPU Not Detected

**Symptoms:**
- Processing is slow
- See "cpu" instead of "mps" in output

**Expected output:**
```
YOLO detector using device: mps
```

**If you see `cpu` instead:**
- GPU acceleration isn't working (but code still works)
- On Apple Silicon, MPS should work automatically
- Check that you have the latest PyTorch/Ultralytics versions

### Script Not Executable

**Symptoms:**
- Permission denied when running scripts
- Cannot execute `./run.sh`

**Solutions:**

```bash
# Make scripts executable
chmod +x run.sh
chmod +x run.py

# Or run directly with Python
python3 run.py
```

### Processing is Very Slow

**Symptoms:**
- Takes hours instead of minutes
- Very slow per-frame processing

**Solutions:**

1. **Check if `lap` is installed** (most common cause)
   ```bash
   pip install lap
   ```

2. **Use faster settings:**
   ```bash
   python scripts/run_detection.py \
     --source video.mp4 \
     --model-size n \
     --fps 5 \
     --plate-interval 10 \
     --no-annotated
   ```

3. **Check GPU acceleration:**
   - Should see "mps" in output, not "cpu"

4. **Reduce frame processing:**
   - Use `--fps 5` to process fewer frames
   - Use `--plate-interval 10` to skip plate detection

## Time Estimates

### Setup Commands (First Time Only)

| Command | Time |
|---------|------|
| Navigate to project | < 1 second |
| Create virtual environment | 5-15 seconds |
| Activate virtual environment | < 1 second |
| Upgrade pip | 10-30 seconds |
| Install core dependencies | 2-5 minutes |
| Install PaddleOCR | 3-8 minutes |
| Install tracking dependency (lap) | 10-30 seconds |

**Total Setup Time: 5-15 minutes**

### Processing Time

For a typical 175MB video (~1800 frames at 10 FPS):

| Setting | Frames Processed | Estimated Time |
|---------|------------------|----------------|
| `--fps 10` (balanced) | ~180 frames | **15-30 minutes** |
| `--fps 5` (faster) | ~90 frames | **8-15 minutes** |
| `--fps None` (all frames) | ~1800 frames | **2-4 hours** |
| `--model-size n` (nano) | ~180 frames | **10-20 minutes** |
| `--model-size l` (large) | ~180 frames | **30-60 minutes** |

**Per-frame processing time:**
- With GPU (MPS): ~5-10 seconds per frame
- Without GPU (CPU): ~15-30 seconds per frame
- With tracking working: ~3-5 seconds per frame

**Factors affecting processing time:**
- ✅ GPU acceleration (MPS) - **2-3x faster**
- ✅ Model size (n=nano fastest, x=xlarge slowest)
- ✅ FPS setting (lower = fewer frames = faster)
- ✅ Plate detection interval (higher = faster)
- ❌ Missing `lap` package - causes 3+ seconds per frame slowdown

### Quick Run Commands (After Setup)

| Task | Time |
|------|------|
| Activate venv | < 1 second |
| Run detection (balanced) | 15-30 minutes |
| Run detection (fast) | 8-15 minutes |
| Run detection (all frames) | 2-4 hours |

## Common Error Messages

### "No module named 'lap'"
```bash
pip install lap
```

### "No module named 'paddleocr'"
```bash
pip install paddlepaddle paddleocr
```

### "venv: command not found"
```bash
# Make sure you're in the project directory
cd "/path/to/Near-Miss-Detection"
python3 -m venv venv
```

### "Permission denied"
```bash
# Don't use sudo - use virtual environment instead
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

### "Cannot open video"
```bash
# Check file path
ls -lh data/samples/traffic_video.mp4

# Try absolute path
python scripts/run_detection.py --source "/full/path/to/video.mp4"
```

## Getting Help

If you're still experiencing issues:

1. **Check the logs** - Look for error messages in the output
2. **Verify installation** - Run verification commands from [Getting Started](getting-started.md)
3. **Check system requirements** - Ensure Python 3.9+ is installed
4. **Review configuration** - Check that all settings are correct

## Prevention Tips

1. **Always use virtual environment** - Prevents dependency conflicts
2. **Install `lap` immediately** - Critical for tracking performance
3. **Use appropriate model size** - Don't use largest model unless needed
4. **Monitor processing speed** - Stop and adjust if too slow
5. **Keep dependencies updated** - Regularly update packages




