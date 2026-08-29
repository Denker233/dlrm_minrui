# Brief for the Claude Code agent on the A100 node

## Context (read first)

You are the GPU half of a two-node pipeline. A CPU node ("dlrm-tb", 10.52.2.91, user
`cc`) is currently preprocessing the full 24-day Criteo Terabyte dataset (~9 h,
finishing roughly 2026-08-27 07:00 UTC) and will push the training-ready files to
this machine incrementally. **Your job: validate the GPU path now, receive the data,
then train one epoch of DLRM and hand the checkpoint back.**

The research purpose: the trained model is the input to a DC embedding-compression
study. The checkpoint itself is the deliverable — training-time accuracy printouts are
uninformative (the test set has a 3.2% positive rate, so ~96.8% accuracy is the
majority-class floor; AUC is measured later, separately).

Repo: `https://github.com/Denker233/dlrm_minrui`, branch `codecs_display`.
The full work log is in `SETUP_PLAN.md` in the repo if you need history.

## What will arrive (do NOT regenerate any of this)

Into `~/input/terabyte/`, via rsync from dlrm-tb:

- `day_0_reordered.npz` … `day_23_reordered.npz` — one per day, ~28 GiB each
  (X_cat stored as **int32**; the loader's collate does
  `np.array(..., dtype=np.int64)` so no code change is needed)
- `day_fea_count.npz`, `day_day_count.npz` — small metadata; **required** by the loader
- possibly `day_transfer.done` — marker that the set is complete

Days 0–22 are training, day 23 is test (MLPerf convention). Total ~0.68 TB.
**Never** run preprocessing here — the loader short-circuits to the `_reordered.npz`
files when all 24 exist plus the two metadata files. If it ever starts printing
"Reading raw data" or "Loading file day_0" followed by dictionary building, STOP —
something is missing; check with the other node rather than letting it rebuild.

## Task 1 — environment + smoke test (do immediately, no data needed)

```bash
tmux new -s a100          # everything in tmux; SSH drops must not kill work
git clone -b codecs_display https://github.com/Denker233/dlrm_minrui && cd dlrm_minrui
python3 -m venv gpu_env && source gpu_env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # NOT +cpu
pip install numpy scikit-learn pandas
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Smoke test — synthetic data at the EXACT 24-day table sizes (49.0M rows, 11.68 GiB
fp32 of embeddings; fits easily in 80 GB):

```bash
python dlrm_s_pytorch.py \
  --arch-embedding-size="10000000-32448-15529-7289-19879-3-6611-1384-62-10000000-1262224-284369-10-2208-10424-89-4-963-14-10000000-6919756-10000000-397757-11440-103-35" \
  --arch-sparse-feature-size=64 \
  --arch-mlp-bot="13-512-256-64" --arch-mlp-top="512-512-256-1" \
  --data-generation=random --data-size=2048000 \
  --loss-function=bce --round-targets=True --learning-rate=0.1 \
  --mini-batch-size=2048 --num-batches=200 --print-freq=50 --print-time \
  --use-gpu
```

This validates CUDA sparse-EmbeddingBag gradients (`sparse=True` is hardcoded in the
model) and reports real ms/it. Known risk: sparse CUDA grads are torch-version
sensitive. If it crashes, capture the traceback; fallback is converting small tables
to dense grads while keeping the 5 big tables sparse — but report before patching.

Also record: `df -h /` (disk headroom vs the incoming 0.68 TB) and `nvidia-smi`.

## Task 2 — periodic checkpointing patch (before training starts)

The stock script saves only when test accuracy improves, and we evaluate exactly once
at the end — so add unconditional periodic saves. In `dlrm_s_pytorch.py`, inside the
training loop (near the existing `should_test` logic), add: every 250,000 iterations,
`torch.save` the same `model_metrics_dict` structure to
`./models/ckpt_it{j}.pt` (distinct files, never overwrite the main path).
~2.3M total iterations → ~9 intermediate checkpoints. These give the researcher a
DC-compression-loss vs training-progress curve. Keep the patch minimal: no change to
optimizer state, evaluation, or the final-save logic.

## Task 3 — training (when the data + `.done` marker are present)

```bash
cd ~/dlrm_minrui && source gpu_env/bin/activate
export CRITEO_DAYS=24
mkdir -p models logs
python -u dlrm_s_pytorch.py \
    --arch-sparse-feature-size=64 \
    --arch-mlp-bot="13-512-256-64" \
    --arch-mlp-top="512-512-256-1" \
    --max-ind-range=10000000 \
    --data-generation=dataset \
    --data-set=terabyte \
    --memory-map \
    --raw-data-file=$HOME/input/terabyte/day \
    --processed-data-file=$HOME/input/terabyte/terabyte_processed.npz \
    --loss-function=bce --round-targets=True \
    --learning-rate=0.1 --mini-batch-size=2048 --nepochs=1 --num-workers=0 \
    --print-freq=10240 --print-time --test-freq=3000000 \
    --test-mini-batch-size=16384 --test-num-workers=16 \
    --use-gpu \
    --save-model="./models/dlrm_terabyte_24day.pt" 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -vE '^Load [0-9]+/' \
  | tee logs/train_24day_gpu.log
```

Config notes — these are deliberate, do not "improve" them:
- `--test-freq=3000000` > total iterations ⇒ exactly ONE evaluation at the end.
  The save is gated on `is_best`, so multiple evals would make the checkpoint
  best-of-N selected on the same test set the later DC study reports AUC against.
- `--num-workers=0` is REQUIRED: the dataset's day-switching is stateful
  (`day_boundary` bug) — workers >0 crash, and naive fixes silently read wrong rows.
- Same arch/lr/batch as the CPU 4-day baseline so results are comparable.
- CPU-node reference throughput was 36.6 ms/it; expect a large GPU speedup, but the
  single-process loader may become the bottleneck. Report the observed ms/it — if
  loading dominates, note it, don't restructure mid-run.
- Expect a several-minute stall at each of the 23 day boundaries (the loader inflates
  the next ~28 GiB day synchronously). That is normal, not a hang.

## Task 4 — when training finishes

1. Verify `TB24`-style completion: exit code 0, one final `Testing at` block, and
   `models/dlrm_terabyte_24day.pt` exists (~12 GB).
2. rsync the final checkpoint AND the periodic `ckpt_it*.pt` files back:
   `rsync -av models/*.pt cc@10.52.2.91:/home/cc/expr/dlrm_minrui/models/gpu24/`
3. Report: final test accuracy line, wall time, ms/it, and any anomalies.
   Do NOT delete the local copies until the other side confirms receipt.

## Ground rules

- Everything long-running goes in tmux.
- Don't push to GitHub (no credentials configured; the researcher pushes manually).
- If disk runs short mid-transfer, say so immediately — the sending node can switch
  to compressed npz (~0.15 TB) at the cost of slower loading.
- When in doubt about anything preprocessing-related: the answer is on the other node;
  ask the researcher rather than regenerating data here.
