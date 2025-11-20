#!/bin/bash
# Bash script to set up the virtual environment

echo "Setting up Readmission Agents project..."

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete! Virtual environment is activated."
echo "To activate manually, run: source venv/bin/activate"

