#!/bin/bash
# Full 24-day Criteo Terabyte: preprocess + train.  Days 0-22 train, day 23 test.
#
# Applied optimisations (all verified bit-identical vs the reference):
#   P1  vectorised categorical remap        (data_utils._remap_cat_vectorised)
#   P2  savez instead of savez_compressed   (3 hot sites)
#   MP  --dataset-multiprocessing, bounded into waves so it cannot OOM
# Deliberately NOT applied: int32 payloads (P3), direct-parquet ingest (P4).
#
# --test-freq is set above the iteration count so the model is evaluated exactly
# once, at the end.  The checkpoint is only written when test accuracy improves,
# so more evaluations would make it "best-of-N selected on the day-23 test set"
# -- the same set the DC benchmark later reports AUC against.
cd /home/cc/expr/dlrm_minrui
source dlrm_env/bin/activate
export CRITEO_DAYS=24
export TMPDIR=/mnt/nvme1/tmp; mkdir -p $TMPDIR
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24     # 48 measured no faster (memory-bound)
export DLRM_MP_CONCURRENCY=8                      # npz build: ~29 GiB/worker
export DLRM_MP_CONCURRENCY_PROC=6                 # remap:     ~48 GiB/worker
set -o pipefail

echo "===== 24-day preprocess + train ====="
date
python -u dlrm_s_pytorch.py \
    --arch-sparse-feature-size=64 \
    --arch-mlp-bot="13-512-256-64" \
    --arch-mlp-top="512-512-256-1" \
    --max-ind-range=10000000 \
    --data-generation=dataset \
    --data-set=terabyte \
    --memory-map \
    --dataset-multiprocessing \
    --raw-data-file=/home/cc/input/terabyte/day \
    --processed-data-file=/home/cc/input/terabyte/terabyte_processed.npz \
    --loss-function=bce --round-targets=True \
    --learning-rate=0.1 --mini-batch-size=2048 --nepochs=1 --num-workers=0 \
    --print-freq=10240 --print-time --test-freq=3000000 \
    --test-mini-batch-size=16384 --test-num-workers=16 \
    --save-model="./models/dlrm_terabyte_24day.pt" 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -vE '^Load [0-9]+/'
echo "TB24_TRAIN_RC=${PIPESTATUS[0]}"
date
ls -la models/
echo "TB24_SETUP_DONE"
