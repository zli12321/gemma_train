#!/bin/bash

NPROC_PER_NODE=4

llamafactory-cli train \
    --config examples/train_full/gemma2_full_sft_prometheus.yaml \
    --nproc_per_node $NPROC_PER_NODE
