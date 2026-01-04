#!/bin/bash

# Near-Miss Detection - Automated Run Script
# This script sets up the environment and runs the detection program

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Near-Miss Detection - Setup & Run${NC}"
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
if [ ! -f "$VIDEO_FILE" ]; then
    echo -e "${YELLOW}⚠ Warning: Video file not found at $VIDEO_FILE${NC}"
    echo -e "${YELLOW}Please provide video file path as argument:${NC}"
    echo -e "${BLUE}  ./run.sh /path/to/your/video.mp4${NC}"
    echo ""
    read -p "Enter video file path (or press Enter to use default): " USER_VIDEO
    if [ -n "$USER_VIDEO" ]; then
        VIDEO_FILE="$USER_VIDEO"
    else
        echo -e "${YELLOW}Exiting. Please place video at $VIDEO_FILE or provide path.${NC}"
        exit 1
    fi
fi

if [ ! -f "$VIDEO_FILE" ]; then
    echo -e "${YELLOW}Error: Video file not found: $VIDEO_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Video file found: $VIDEO_FILE"
echo ""

# Run detection
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Detection...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Default arguments (balanced settings)
MODEL_SIZE="${MODEL_SIZE:-s}"
FPS="${FPS:-10}"
PLATE_INTERVAL="${PLATE_INTERVAL:-3}"

# Allow command line arguments to override defaults
if [ "$#" -ge 1 ]; then
    VIDEO_FILE="$1"
fi
if [ "$#" -ge 2 ]; then
    MODEL_SIZE="$2"
fi
if [ "$#" -ge 3 ]; then
    FPS="$3"
fi
if [ "$#" -ge 4 ]; then
    PLATE_INTERVAL="$4"
fi

echo -e "${BLUE}Settings:${NC}"
echo -e "  Video: $VIDEO_FILE"
echo -e "  Model: YOLOv10$MODEL_SIZE"
echo -e "  FPS: $FPS"
echo -e "  Plate Interval: $PLATE_INTERVAL"
echo ""

python scripts/run_detection.py \
    --source "$VIDEO_FILE" \
    --model-size "$MODEL_SIZE" \
    --fps "$FPS" \
    --plate-interval "$PLATE_INTERVAL"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Detection Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Results saved in: ${BLUE}data/outputs/${NC}"
echo -e "  - annotated_output.mp4"
echo -e "  - detections.json"
echo ""

