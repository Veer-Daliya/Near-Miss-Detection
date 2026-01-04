#!/bin/bash
# Run detection script with caffeinate to prevent sleep

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Use caffeinate to prevent sleep while processing
# -d: prevents display from sleeping
# -i: prevents system from idle sleeping
# -m: prevents disk from idle sleeping
# -s: prevents system from sleeping (only on AC power)
# -u: simulates user activity

echo "Starting detection with caffeinate (computer will stay awake)..."
echo "Press Ctrl+C to stop"
echo ""

# Run the detection script with caffeinate
caffeinate -d -i -m -s -u python scripts/run_detection.py "$@"

