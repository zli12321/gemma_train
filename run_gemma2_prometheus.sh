#!/bin/bash

# Set your HuggingFace token here or run `huggingface-cli login` before this script
# export HF_TOKEN="hf_your_token_here"

NPROC_PER_NODE=4 llamafactory-cli train \
    examples/train_full/gemma2_full_sft_prometheus.yaml
