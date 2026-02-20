#!/bin/bash
set -e

conda create -n prometheus python=3.11 -y

# Source conda so 'conda activate' works in a script
eval "$(conda shell.bash hook)"
conda activate prometheus

# Install PyTorch (change cu121 to match your CUDA version from nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install LLaMA Factory
pip install -e .

# Install DeepSpeed for multi-GPU training
pip install deepspeed

pip install huggingface_hub

# Generate training data from HuggingFace
python convert_prometheus.py

echo "Setup complete! Run training with: bash run_gemma2_prometheus.sh"

conda activate prometheus


python -c "from huggingface_hub import login; login()"