#!/bin/bash

# Drone Tracking Server Startup Script

echo "Drone Tracking Server"
echo "========================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if evio is installed
if ! python -c "import evio" 2>/dev/null; then
    echo "[WARNING] evio library not found. Installing..."
    cd evio
    pip install -e .
    cd ..
    echo "[OK] evio installed"
else
    echo "[OK] evio library found"
fi

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "[WARNING] Flask not found. Installing dependencies..."
    pip install -r requirements.txt
    echo "[OK] Dependencies installed"
else
    echo "[OK] Flask found"
fi

echo ""
echo "Starting server..."
echo "Dashboard: http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

python app.py
