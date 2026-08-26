#!/bin/bash
# Full Criteo Terabyte (4-day) setup: fetch -> convert -> preprocess -> train.
cd /home/cc/expr/dlrm_minrui
source dlrm_env/bin/activate
export CRITEO_DAYS=4
export TMPDIR=/mnt/nvme0/tmp; mkdir -p $TMPDIR
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
set -o pipefail

echo "===== [1/2] fetch + convert 4 days ====="
i=0
for d in 2015-02-15 2015-02-16 2015-02-17 2015-02-18; do
  ./tb_fetch_day.sh "$d" "$i" || { echo "TB_FETCH_FAILED day_$i"; exit 1; }
  df -h /mnt/nvme0 | tail -1
  i=$((i+1))
done
echo "TB_RAW_ALL_DONE"; ls -la /home/cc/input/terabyte/

echo "===== [2/2] preprocess + train (1 epoch, days 0-2 train / day 3 test) ====="
# data_utils prints one progress line PER ROW using \r; strip it or the log hits tens of GB.
python -u dlrm_s_pytorch.py \
    --arch-sparse-feature-size=64 \
    --arch-mlp-bot="13-512-256-64" \
    --arch-mlp-top="512-512-256-1" \
    --max-ind-range=10000000 \
    --data-generation=dataset \
    --data-set=terabyte \
    --memory-map \
    --raw-data-file=/home/cc/input/terabyte/day \
    --processed-data-file=/home/cc/input/terabyte/terabyte_processed.npz \
    --loss-function=bce --round-targets=True \
    --learning-rate=0.1 --mini-batch-size=2048 --nepochs=1 --num-workers=0 \
    --print-freq=2048 --print-time --test-freq=102400 \
    --test-mini-batch-size=16384 --test-num-workers=16 \
    --save-model="./models/dlrm_terabyte_4day.pt" 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -vE '^Load [0-9]+/'
echo "TB_TRAIN_RC=${PIPESTATUS[0]}"
ls -la models/
echo "TB_SETUP_DONE"
