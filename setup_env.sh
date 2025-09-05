#!/usr/bin/env bash
# Create and activate Conda environment for deepfake model

ENV_NAME=df
PYTHON_VERSION=3.10

echo "Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "Installing dependencies..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install opencv-python tqdm matplotlib pyyaml scikit-learn

echo "Setup complete. To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"

