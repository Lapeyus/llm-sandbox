#!/bin/bash

# Set the environment name
ENV_NAME="ocr_env"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python 3 is not installed. Please install it and try again."
    exit 1
fi

# Create the virtual environment
echo "[INFO] Creating virtual environment: $ENV_NAME..."
python3 -m venv $ENV_NAME

# Check if the environment was created successfully
if [ ! -d "$ENV_NAME" ]; then
    echo "[ERROR] Failed to create virtual environment: $ENV_NAME."
    exit 1
fi

# Activate the virtual environment
echo "[INFO] Activating virtual environment..."
source $ENV_NAME/bin/activate

# Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f requirements.txt ]; then
    echo "[INFO] Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "[ERROR] requirements.txt not found. Please ensure the file exists and try again."
    deactivate
    exit 1
fi

echo "[INFO] Setup complete. Virtual environment '$ENV_NAME' is ready."

# Provide manual activation instructions
echo "[INFO] To manually activate the environment, run: source $ENV_NAME/bin/activate"