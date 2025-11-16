#!/bin/bash

# Setup script for Drone Tracking Server

echo "Drone Tracking Server - Setup"
echo "================================="
echo ""

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed"
    echo "Please install Python 3 first"
    exit 1
fi

echo "[OK] Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment
if [ -d "venv" ]; then
    echo "[WARNING] Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        echo "[OK] Removed old virtual environment"
    else
        echo "Using existing virtual environment"
    fi
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install build tools
echo "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel -q

# Install evio library
echo "Installing evio library..."
cd evio
pip install -e . -q
cd ..
echo "[OK] evio installed"

# Install other dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q
echo "[OK] Dependencies installed"

echo ""
echo "Setup complete!"
echo ""
echo "To start the server, run:"
echo "  ./start_server.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python app.py"
