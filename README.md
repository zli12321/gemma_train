# Gemma-2B SFT on Prometheus Feedback Collection

Fine-tune [google/gemma-2b](https://huggingface.co/google/gemma-2b) on the [Prometheus Feedback Collection](https://huggingface.co/datasets/prometheus-eval/Feedback-Collection) dataset using [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory).

## Setup

```bash
git clone https://github.com/zli12321/gemma_train.git
bash setup.sh
```

This will:
1. Create a conda environment (`prometheus`, Python 3.11)
2. Install PyTorch, LLaMA Factory, and DeepSpeed
3. Download and convert the Prometheus dataset

> **Note:** Edit the CUDA version in `setup.sh` if your server is not on CUDA 12.1. Check with `nvidia-smi`.

## Training

```bash
eval "$(conda shell.bash hook)"
conda activate prometheus
bash run_gemma2_prometheus.sh
```

Training config: `examples/train_full/gemma2_full_sft_prometheus.yaml`

## Configuration

| Parameter | Value |
|---|---|
| Model | google/gemma-2b |
| Fine-tuning | Full parameter |
| Dataset | Prometheus Feedback Collection (99,952 examples) |
| GPUs | 4 (adjustable via `NPROC_PER_NODE` in `run_gemma2_prometheus.sh`) |
| DeepSpeed | ZeRO Stage 3 |
| Batch size | 2 per device × 4 accumulation × 4 GPUs = 32 effective |
| Learning rate | 1e-5 (cosine schedule) |
| Epochs | 3 |
| Cutoff length | 2048 tokens |

## Output

Checkpoints are saved to `saves/gemma-2b/full/sft-prometheus/`.
