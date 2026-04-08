#!/bin/bash

echo "CPU-only mini test (will be VERY slow)"
echo "WARNING: Training 20k batches on CPU would take DAYS"
echo "Running only 100 batches for demonstration..."

python dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--mini-batch-size=32 \
--print-freq=20 \
--test-freq=50 \
--test-mini-batch-size=128 \
--test-num-workers=4 \
--num-batches=100 \
--cat-path="$HOME/expr/dlrm_minrui/criteo_24days/sparse" \
--dense-path="$HOME/expr/dlrm_minrui/criteo_24days/dense" \
--label-path="$HOME/expr/dlrm_minrui/criteo_24days/label" \
--count-path="$HOME/expr/dlrm_minrui/criteo_24days/processed_count.bin" \
2>&1 | tee cpu_test.log

echo ""
echo "Test complete. Note: 100 batches is NOT enough for good accuracy"
tail -20 cpu_test.log
