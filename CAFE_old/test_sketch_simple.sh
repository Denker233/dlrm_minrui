#!/bin/bash

echo "Testing sketch after fix (500 batches quick test)"

python dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--loss-function=bce \
--mini-batch-size=128 \
--print-freq=100 \
--test-freq=250 \
--test-mini-batch-size=2048 \
--sketch-flag \
--compress-rate=0.001 \
--hash-rate=0.5 \
--num-batches=500 \
--cat-path="$HOME/expr/dlrm_minrui/criteo_24days/sparse" \
--dense-path="$HOME/expr/dlrm_minrui/criteo_24days/dense" \
--label-path="$HOME/expr/dlrm_minrui/criteo_24days/label" \
--count-path="$HOME/expr/dlrm_minrui/criteo_24days/processed_count.bin" \
2>&1 | tee sketch_test_500.log

if [ $? -eq 0 ]; then
    echo ""
    echo "✓✓✓ SUCCESS! Sketch compression works!"
    echo ""
    grep "accuracy.*auc" sketch_test_500.log | tail -1
else
    echo ""
    echo "✗✗✗ Still failing. Check sketch_test_500.log for errors"
fi

