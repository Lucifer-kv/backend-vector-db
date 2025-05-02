#!/bin/bash
echo "Setting up virtual environment..."
python -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running tests..."
pytest tests/

echo "Starting FastAPI server..."
python main.py