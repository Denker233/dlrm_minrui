#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python ../dlrm_s_pytorch.py \
--use-gpu \
--inference-only \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--mini-batch-size=16384 \
--test-mini-batch-size=16384 \
--test-num-workers=16 \
--print-freq=100 \
--print-time \
--num-batches=1000 \
--sketch-flag \
--compress-rate=0.001 \
--hash-rate=0.5 \
--sketch-threshold=1 \
--cat-path="$HOME/expr/dlrm_minrui/criteo_24days/sparse" \
--dense-path="$HOME/expr/dlrm_minrui/criteo_24days/dense" \
--label-path="$HOME/expr/dlrm_minrui/criteo_24days/label" \
--count-path="$HOME/expr/dlrm_minrui/criteo_24days/processed_count.bin" \
2>&1 | tee inference_sketch.log

echo "done"
