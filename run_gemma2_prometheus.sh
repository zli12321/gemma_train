#!/bin/bash

NPROC_PER_NODE=4 llamafactory-cli train \
    examples/train_full/gemma2_full_sft_prometheus.yaml
