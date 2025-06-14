#!/bin/bash

# Exit on error
set -e

# # Create and activate conda environment
# echo "Creating conda environment..."
# conda create -n project_env python=3.9 -y
# source activate project_env

# # Install required packages
# echo "Installing required packages..."
# pip install torch tensorflow safetensors
# pip install -r requirements.txt --force-reinstall

# # check model parameters
# conda run -n project_env python check_para.py --framwork pytorch
# #or
# conda run -n project_env python check_para.py --framework tensorflow

# Run the project script
echo "Running project script..."
bash run.sh

# Deactivate conda environment
echo "Deactivating conda environment..."
conda deactivate
