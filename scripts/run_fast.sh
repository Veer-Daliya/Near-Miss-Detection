#!/bin/bash

# Near-Miss Detection - FAST Mode (< 5 minutes)
# Optimized for speed: uses nano model, processes fewer frames, skips video encoding

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Near-Miss Detection - FAST MODE${NC}"
echo -e "${BLUE}Target: < 5 minutes${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Navigate to project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
echo -e "${GREEN}✓${NC} Project directory: $SCRIPT_DIR"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"
echo ""

# Check if dependencies are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python -c "import ultralytics" 2>/dev/null; then
    echo -e "${YELLOW}Installing core dependencies...${NC}"
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    echo -e "${GREEN}✓${NC} Core dependencies installed"
else
    echo -e "${GREEN}✓${NC} Core dependencies already installed"
fi

if ! python -c "import paddleocr" 2>/dev/null; then
    echo -e "${YELLOW}Installing PaddleOCR...${NC}"
    pip install paddlepaddle paddleocr
    echo -e "${GREEN}✓${NC} PaddleOCR installed"
else
    echo -e "${GREEN}✓${NC} PaddleOCR already installed"
fi

if ! python -c "import lap" 2>/dev/null; then
    echo -e "${YELLOW}Installing tracking dependency (lap)...${NC}"
    pip install lap
    echo -e "${GREEN}✓${NC} Tracking dependency installed"
else
    echo -e "${GREEN}✓${NC} Tracking dependency already installed"
fi
echo ""

# Check if video file exists
VIDEO_FILE="data/samples/traffic_video.mp4"
if [ "$#" -ge 1 ]; then
    VIDEO_FILE="$1"
fi

if [ ! -f "$VIDEO_FILE" ]; then
    echo -e "${YELLOW}⚠ Warning: Video file not found at $VIDEO_FILE${NC}"
    echo -e "${YELLOW}Please provide video file path as argument:${NC}"
    echo -e "${BLUE}  ./run_fast.sh /path/to/your/video.mp4${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Video file found: $VIDEO_FILE"
echo ""

# FAST MODE SETTINGS
MODEL_SIZE="n"          # nano - fastest model
FPS="2"                 # Process every 2nd frame (very fast)
PLATE_INTERVAL="10"     # Process plates every 10th frame (skip most)

echo -e "${BLUE}FAST MODE Settings:${NC}"
echo -e "  Video: $VIDEO_FILE"
echo -e "  Model: YOLOv10${MODEL_SIZE} (nano - fastest)"
echo -e "  FPS: ${FPS} (processes fewer frames)"
echo -e "  Plate Interval: ${PLATE_INTERVAL} (skips most plate processing)"
echo -e "  Video Output: DISABLED (JSON only - faster)"
echo ""
echo -e "${YELLOW}⚠ Note: This mode prioritizes speed over accuracy${NC}"
echo -e "${YELLOW}  - Uses smallest model (may miss some objects)${NC}"
echo -e "${YELLOW}  - Processes fewer frames${NC}"
echo -e "${YELLOW}  - Skips video encoding (JSON results only)${NC}"
echo ""

# Run detection in FAST MODE
python scripts/run_detection.py \
    --source "$VIDEO_FILE" \
    --model-size "$MODEL_SIZE" \
    --fps "$FPS" \
    --plate-interval "$PLATE_INTERVAL" \
    --no-annotated

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Detection Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Results saved in: ${BLUE}data/outputs/${NC}"
echo -e "  - detections.json (detection data)"
echo -e "  - No video output (disabled for speed)"
echo ""

