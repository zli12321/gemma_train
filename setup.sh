conda create -n prometheus python=3.11 
conda init
conda activate prometheus

# For CUDA 12.1 (most common on modern servers)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

cd ./gemma_train

pip install -e .

pip install deepspeed

bash run_gemma2_prometheus.sh

bash run_gemma2_prometheus.sh