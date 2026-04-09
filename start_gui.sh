#!/bin/bash
# Quick start script for Dataset Pipeline GUI

cd "$(dirname "$0")" || exit 1

echo "================================"
echo "Dataset Pipeline GUI"
echo "================================"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
echo "Installing dependencies..."
pip install --quiet --upgrade pip pandas scikit-learn 2>/dev/null

echo ""
echo "✓ Environment ready"
echo ""
echo "Starting GUI application..."
echo ""

# Run the GUI
python3 run_dataset_gui.py

# Deactivate on exit
deactivate
