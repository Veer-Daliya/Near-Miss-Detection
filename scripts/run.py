#!/usr/bin/env python3
"""
Near-Miss Detection - Python Wrapper Script
Automatically sets up environment and runs detection.
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

def print_colored(message, color=Colors.NC):
    """Print colored message."""
    print(f"{color}{message}{Colors.NC}")

def check_command(cmd):
    """Check if command exists."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False

def run_command(cmd, description=""):
    """Run command and handle errors."""
    if description:
        print_colored(f"  {description}...", Colors.YELLOW)
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if description:
            print_colored(f"  ✓ {description} complete", Colors.GREEN)
        return result
    except subprocess.CalledProcessError as e:
        print_colored(f"  ✗ Error: {e}", Colors.RED)
        print_colored(f"  Output: {e.stdout}", Colors.RED)
        print_colored(f"  Error: {e.stderr}", Colors.RED)
        sys.exit(1)

def main():
    """Main function."""
    print_colored("=" * 50, Colors.BLUE)
    print_colored("Near-Miss Detection - Setup & Run", Colors.BLUE)
    print_colored("=" * 50, Colors.BLUE)
    print()
    
    # Get project directory
    project_dir = Path(__file__).parent.absolute()
    os.chdir(project_dir)
    print_colored(f"✓ Project directory: {project_dir}", Colors.GREEN)
    print()
    
    # Check Python version
    python_cmd = "python3" if sys.platform != "win32" else "python"
    
    # Create virtual environment if needed
    venv_path = project_dir / "venv"
    if not venv_path.exists():
        print_colored("Creating virtual environment...", Colors.YELLOW)
        run_command(f"{python_cmd} -m venv venv", "Creating venv")
    else:
        print_colored("✓ Virtual environment already exists", Colors.GREEN)
    print()
    
    # Determine activation script
    if sys.platform == "win32":
        pip_cmd = venv_path / "Scripts" / "pip"
        python_venv = venv_path / "Scripts" / "python"
    else:
        pip_cmd = venv_path / "bin" / "pip"
        python_venv = venv_path / "bin" / "python"
    
    # Check and install dependencies
    print_colored("Checking dependencies...", Colors.YELLOW)
    
    # Check ultralytics
    check_cmd = f'"{python_venv}" -c "import ultralytics" 2>/dev/null'
    if not check_command(check_cmd):
        print_colored("Installing core dependencies...", Colors.YELLOW)
        run_command(f'"{pip_cmd}" install --upgrade pip --quiet')
        run_command(f'"{pip_cmd}" install -r requirements.txt')
    else:
        print_colored("✓ Core dependencies installed", Colors.GREEN)
    
    # Check paddleocr
    check_cmd = f'"{python_venv}" -c "import paddleocr" 2>/dev/null'
    if not check_command(check_cmd):
        print_colored("Installing PaddleOCR...", Colors.YELLOW)
        run_command(f'"{pip_cmd}" install paddlepaddle paddleocr')
    else:
        print_colored("✓ PaddleOCR installed", Colors.GREEN)
    
    # Check lap
    check_cmd = f'"{python_venv}" -c "import lap" 2>/dev/null'
    if not check_command(check_cmd):
        print_colored("Installing tracking dependency (lap)...", Colors.YELLOW)
        run_command(f'"{pip_cmd}" install lap')
    else:
        print_colored("✓ Tracking dependency installed", Colors.GREEN)
    print()
    
    # Check video file
    video_file = Path("data/samples/traffic_video.mp4")
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        video_file = Path(sys.argv[1])
    
    if not video_file.exists():
        print_colored(f"⚠ Warning: Video file not found: {video_file}", Colors.YELLOW)
        print_colored("Usage: python run.py [video_path] [model_size] [fps] [plate_interval]", Colors.BLUE)
        print_colored("Example: python run.py data/samples/traffic_video.mp4 s 10 3", Colors.BLUE)
        sys.exit(1)
    
    print_colored(f"✓ Video file found: {video_file}", Colors.GREEN)
    print()
    
    # Get arguments
    model_size = sys.argv[2] if len(sys.argv) > 2 else "s"
    fps = sys.argv[3] if len(sys.argv) > 3 else "10"
    plate_interval = sys.argv[4] if len(sys.argv) > 4 else "3"
    
    # Run detection
    print_colored("=" * 50, Colors.BLUE)
    print_colored("Starting Detection...", Colors.BLUE)
    print_colored("=" * 50, Colors.BLUE)
    print()
    
    print_colored("Settings:", Colors.BLUE)
    print_colored(f"  Video: {video_file}")
    print_colored(f"  Model: YOLOv10{model_size}")
    print_colored(f"  FPS: {fps}")
    print_colored(f"  Plate Interval: {plate_interval}")
    print()
    
    # Run the detection script
    detection_cmd = (
        f'"{python_venv}" scripts/run_detection.py '
        f'--source "{video_file}" '
        f'--model-size {model_size} '
        f'--fps {fps} '
        f'--plate-interval {plate_interval}'
    )
    
    run_command(detection_cmd, "Running detection")
    
    print()
    print_colored("=" * 50, Colors.GREEN)
    print_colored("Detection Complete!", Colors.GREEN)
    print_colored("=" * 50, Colors.GREEN)
    print()
    print_colored("Results saved in: data/outputs/", Colors.BLUE)
    print_colored("  - annotated_output.mp4", Colors.BLUE)
    print_colored("  - detections.json", Colors.BLUE)
    print()

if __name__ == "__main__":
    main()

