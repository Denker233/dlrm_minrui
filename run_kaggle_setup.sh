#!/bin/bash
# Criteo Kaggle setup: preprocess -> train the canonical D=16 checkpoint.
cd /home/cc/expr/dlrm_minrui
source dlrm_env/bin/activate
export TMPDIR=/mnt/nvme0/tmp; mkdir -p $TMPDIR
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
set -o pipefail
python -u dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/home/cc/input/train.txt \
    --processed-data-file=/home/cc/input/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce --round-targets=True \
    --learning-rate=0.1 --mini-batch-size=128 --nepochs=1 --num-workers=0 \
    --print-freq=8192 --print-time --test-freq=50000 \
    --test-mini-batch-size=16384 --test-num-workers=16 \
    --save-model="./models/dlrm_kaggle_correct.pt" 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -vE '^Load [0-9]+/'
echo "KAGGLE_TRAIN_RC=${PIPESTATUS[0]}"
ls -la models/
echo "KAGGLE_SETUP_DONE"
