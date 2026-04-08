#!/bin/bash
# quick_compare.sh - Quick comparison with cache dropping

# Check for sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Error: Must run with sudo for cache dropping"
    echo "Usage: sudo ./quick_compare.sh"
    exit 1
fi

# Get the actual user's home directory (not root's)
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)

# Get Python path from the user's environment
PYTHON_PATH=$(su - $ACTUAL_USER -c "which python")

echo "Running as: $ACTUAL_USER"
echo "Home directory: $ACTUAL_HOME"
echo "Python path: $PYTHON_PATH"
echo ""

# Verify Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python not found at $PYTHON_PATH"
    echo "Please activate your conda environment and run again"
    exit 1
fi

drop_cache() {
    echo "Dropping cache..."
    sync
    echo 3 > /proc/sys/vm/drop_caches
    sleep 2
}

echo "============================================"
echo "SKETCH (CAFE) VERSION"
echo "============================================"
drop_cache
cd "$ACTUAL_HOME/expr/dlrm_minrui/CAFE_old"
time $PYTHON_PATH dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --test-mini-batch-size=2048 \
    --num-workers=0 \
    --test-num-workers=0 \
    --print-time \
    --sketch-flag \
    --compress-rate=0.007 \
    --hash-rate=0.2 \
    --cat-path="$ACTUAL_HOME/expr/dlrm_minrui/criteo_24days/sparse" \
    --dense-path="$ACTUAL_HOME/expr/dlrm_minrui/criteo_24days/dense" \
    --label-path="$ACTUAL_HOME/expr/dlrm_minrui/criteo_24days/label" \
    --count-path="$ACTUAL_HOME/expr/dlrm_minrui/criteo_24days/processed_count.bin" \
    --load-model="cafe_experiments_20260120_045901/exp1_150x_conservative/model.pt" \
    --inference-only 2>&1 | grep -E "MLP time|embedding time|interaction time|total time|accuracy"

echo ""
echo "============================================"
echo "ORIGINAL DLRM VERSION"
echo "============================================"
drop_cache
cd "$ACTUAL_HOME/expr/dlrm_minrui"
time $PYTHON_PATH dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --test-mini-batch-size=2048 \
    --num-workers=0 \
    --test-num-workers=0 \
    --print-time \
    --processed-data-file="$ACTUAL_HOME/expr/dlrm_minrui/criteo/kaggle_processed_sparse.bin" \
    --inference-only 2>&1 | grep -E "MLP time|embedding time|interaction time|total time|accuracy"
