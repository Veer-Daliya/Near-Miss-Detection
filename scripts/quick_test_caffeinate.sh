#!/bin/bash
# Quick test with caffeinate to prevent sleep

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Running quick test with caffeinate (computer will stay awake)..."
echo ""

# Run quick test with caffeinate
caffeinate -d -i -m -s -u python scripts/quick_test.py "$@"

