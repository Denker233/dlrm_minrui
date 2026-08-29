# Setup, Progress & Findings — dlrm_minrui on the `cc` NVMe node

**Autonomous work log.** Started 2026-08-23. Branch `codecs_display`.
Everything below was measured on this machine unless explicitly marked otherwise.

---

## 0. Machine

| Item | Value |
|---|---|
| CPU | 2× Intel Xeon E5-2670 v3, 24C/48T (Haswell) — **AVX2 only, no AVX-512** |
| Governor | `performance` (was `schedutil`, pinned at 1200 MHz → now 3100 MHz) |
| RAM | 503 GB |
| NVMe 0 | Intel P3600 2 TB → `/mnt/nvme0` (dataset), **NUMA node 0** |
| NVMe 1 | Intel P3600 2 TB → `/mnt/nvme1` (benchmarks), **NUMA node 1** |
| Python | 3.10.12 · torch 2.13.0+cpu · venv `dlrm_env` |

> README quotes dav1d AVX-512 decode numbers. This node cannot execute AVX-512;
> any decode figure re-measured here is AVX2 and must be labelled as such.

---

## 1. Status

### Phase A — Environment ✅
- [x] venv, torch, C++ extension, ffmpeg, liburing, fio
- [x] `nvme-cli`, `reptyr` installed
- [x] `bench_iouring` built; patched to evict only its own pages via
      `posix_fadvise(DONTNEED)` instead of a global `drop_caches` (which would have
      flushed 306 GB of page cache under the running training job ~100×)

### Phase B — Datasets & models ✅
- [x] Criteo Kaggle preprocessed + D=16 model trained (`dlrm_kaggle_correct.pt`)
- [x] Criteo Terabyte **4 days** fetched, converted, preprocessed (770.4M samples)
- [x] Terabyte D=64 model trained 1 epoch → `models/dlrm_terabyte_4day.pt` (12.4 GB)
- [x] Baseline AUC measured: **0.789235** (4.10M samples) / **0.788249** (20.5M samples)
      — the README's 0.768820 came from an undertrained model and is superseded

### Phase C — NVMe cold-embedding benchmark ✅
- [x] `results/nvme_benchmark.md` Part 1 (conservative) and Part 2 (optimised)
- [x] Optimised baseline: **9.30 ms → 0.39 ms (24×)**

### Phase D — Experiments
- [x] D1 Terabyte DC sweep on the trained model → `results/unified_comparison.md`
- [x] D2 CAFE+ baseline (pre-existing)
- [x] D7 **per-dimension block means** → `results/perdim_analysis.md` (new, best method)
- [x] D8 **sort-key decomposition** → `results/sort_key_analysis.md`
- [x] D9 INT8 / INT4 baselines measured in the same framework
- [ ] D3 TT-Rec baseline
- [ ] D4 Multi-tenant serving demo
- [ ] D5 Threshold-based dispatch (drop the bitmap, verify the 136× claim)
- [ ] D6 Model loading-time benchmark

### Phase E — Full 24-day Terabyte ⏳ NEXT
See section 4 for the speed plan.

---

## 2. KEY FINDINGS

### F1. The README's Terabyte tables are all superseded
Baseline AUC moved **0.768820 → 0.789235**. Every Terabyte delta in the README was
computed against the undertrained model. Re-measured numbers are in
`results/unified_comparison.md`.

### F2. Per-dimension block means dominate scalar block means
| config | mem | ratio | ΔAUC% |
|---|--:|--:|--:|
| scalar block=16 @1% hot | 43.4 MB | 273× | −0.0874 |
| **per-dim block=256 @0.5% hot** | **32.9 MB** | **360×** | **−0.0455** |

24% less memory **and** 48% less AUC loss — strictly dominant. Scalar survives on the
Pareto frontier only at the extreme corner (414× at −0.1268%).

Larger blocks are better for per-dim (block=256 beats block=16 on the frontier):
per-dimension structure is shared across many rows, so coarse blocks capture nearly all
of it. **block=256 was the largest tested and won everywhere — the optimum may lie beyond.**

### F3. Scalar block means barely beat zeroing
```
zero cold rows      ΔAUC −0.1451%
scalar block mean   ΔAUC −0.0946%   1.5× better than zero
per-dim block=256   ΔAUC −0.0301%   4.8× better than zero
```
The current method is much closer to "zero the cold rows" than it looks.

### F4. Error decomposition — only 0.3% of the error is orderable at D=64
```
sum (w[r,d] − mu_B)^2 = sum (w[r,d] − mu_r)^2   ← within-row, ordering-INDEPENDENT
                      + D * sum (mu_r − mu_B)^2 ← between-row, the only orderable term
```
Within-row is **99.7–100.0%** of the error at D=64 (97.2–99.6% at D=16). Sorting by row
mean is *exactly* optimal for the between-row term, so value-sort is the optimum, not a
heuristic. Keeping a **per-row** mean instead of a per-block mean is pointless: 16× the
storage to remove 0.3% of the error.

### F5. PCA-sort does not replicate — and never beat value-sort
value-sort −0.0341% vs pca-sort −0.0338% at 4.3% hot on Terabyte: indistinguishable.
PCA was *worse* than value-sort on all 8 tables tested, Kaggle included. On Kaggle
corr(PC1, row mean) = 0.93–0.99, i.e. nearly the same permutation, so the README's
"17–26% less AUC loss" claim has no mechanism behind it. **Recommend dropping PCA-sort.**

### F6. "SSD is infeasible at 91 ms" does not survive
That figure is a SATA drive with serial `pread`. Optimised NVMe (O_DIRECT 512 B, IOPOLL,
registered files/buffers, 8 threads NUMA-pinned, 2 drives striped) does the same work in
**1.79 ms**, or **0.39 ms** with a sector-packed layout. In a full forward pass the SSD
path costs **9.35 ms** vs DC's 7.14 ms. Honest claim: DC is faster *and* needs no drives
and no I/O threads — not that SSD is unusable.

### F7. INT8 is a much stronger baseline than the README implies
INT8 whole-table: **+0.0017% AUC, 4×, 6.90 ms** — faster than fp32. DC's case against
INT8 is *memory* (62× less), not accuracy or speed. **INT4 is not a competitor**:
−0.2553% at 8×, dominated by per-dim on every axis (31–45× less memory *and* better AUC).

### F8. C fusion changes the latency ranking; fp32 does not benefit
| config | PyTorch | C fused | speedup |
|---|--:|--:|--:|
| fp32 | 9.70 | 11.10 | **0.87×** |
| INT8 | 8.64 | 7.40 | 1.17× |
| DC value-sort 1% | 12.34 | 7.41 | 1.66× |
| SSD cold 1% | 14.36 | 9.35 | 1.54× |

fp32 got *slower* in C — `index_select` is already one tuned op, nothing to fuse. So the
honest fp32 baseline is PyTorch's. We predicted SSD would gain much less than DC because
its device I/O is irreducible; **that was wrong** — Python assembly was a bigger share
than the device time.

### F9. ~99% of the embedding table is still at random initialisation
Median row norm **0.00147**; DLRM's init `U(±sqrt(1/n))` predicts **0.00146** for n=10M.
Cold rows are not "trained toward zero", they are *untrained*. This is why DC works so
well here, and it is the main external-validity risk: with 24 days the same 10M capped
slots receive ~6× more updates, so DC would be discarding real signal instead of
initialisation noise. **Direction is clear; magnitude unknown — this is why Phase E matters.**

### F10. Preprocessing was ~10× slower than necessary
Per Terabyte day: a Python loop doing 5.1e9 dict lookups at 854 ns each (~72 min) plus
`np.savez_compressed` at **9 MB/s** (~90 min). Fixed in `data_utils.py`:
`pandas.Index.get_indexer` (validated bit-identical on 5.1e9 values of day_0) and plain
`savez`. **~150 min/day → ~15 min/day.**

### F11. 24 days will not change the compression ratios
`max_ind_range=10000000` caps each table. The largest is already at 9,990,976 — 0.09%
below the cap. Full 24 days saturates to exactly 10M: 48.5M → 49.0M rows, **+0.9%**.
Ratios are structurally unchanged; only the embedding *content* changes (see F9).

---

## 3. Artefacts produced

| File | What |
|---|---|
| `results/unified_comparison.md` | fp32 / INT8 / INT4 / DC / SSD, PyTorch vs C fused |
| `results/perdim_analysis.md` | scalar vs per-dimension block means + Pareto |
| `results/sort_key_analysis.md` | error decomposition, why PCA-sort fails |
| `results/nvme_benchmark.md` | NVMe baseline, conservative + optimised (24×) |
| `bench_unified_terabyte.py` | one code path, real storage layouts, AUC + forward time |
| `bench_cpp_terabyte.py` | C-fused counterpart |
| `bench_perdim_terabyte.py` | per-dim sweep incl. INT8/INT4 |
| `libembfwd.c` | fused C gathers: fp32 / INT8 / INT4 / DC scalar / DC per-dim / SSD |
| `libcoldread.c`, `bench_iouring_opt.c` | optimised io_uring engine |
| `analyze_sort_keys.py`, `analyze_dc_variants.py` | the decomposition analyses |
| `data_utils.py` | vectorised remap + plain savez (10× preprocessing) |

---

## 4. PHASE E — Full 24-day Terabyte: speed plan

**Why:** F9 says our result may be an artefact of an undertrained table. 24 days is the
only way to know. **Not** for bigger tables — F11 shows those are already saturated.

### Measured per-stage costs (from the 4-day run's own timestamps)

| stage | 4-day actual | per day |
|---|--:|--:|
| fetch parquet + convert to TSV | 00:34→01:00 | ~6 min |
| TSV → `day_N.npz` (`getCriteoAdData`) | 01:00→13:03 | **~180 min** |
| → `day_N_processed.npz` | 13:03→22:42 | ~145 min (now ~15, patched) |
| reorder/memmap | 22:42→03:25 | ~70 min (now ~5, patched) |
| training 1 epoch, 589M samples | 03:30→09:26 | 6 h total |

### The big win: skip the TSV entirely (measured)

The HuggingFace source is **typed parquet**; the current pipeline serialises it to TSV
text and parses it back with a Python per-row loop (`data_utils.py:1031-1060`: `split`,
26 `int(x,16)` calls through a `lambda`, a per-row `np.array` allocation, and a progress
print *per row*). That is the ~180 min/day.

pyarrow keeps the hex strings in one contiguous byte buffer, and nulls occupy zero bytes,
so valid 8-char values are densely packed. Reshape to `(n,8)`, map ASCII→nibble with a
256-entry LUT, combine with shifts — no per-row work:

```
full part end-to-end (parquet read + convert): 1.16 s / 594,104 rows
  -> 6.4 min/day   vs ~186 min/day for fetch+TSV+parse   = ~28x
```
Verified equal to `int(x,16)` including nulls/empties (which map to 0, as the TSV path does).

**It also never writes the 47 GB/day TSV — saving ~1.13 TB across 24 days**, which was
the binding disk constraint.

### Revised 24-day budget

| stage | naive | with fixes |
|---|--:|--:|
| parquet → `day_N.npz` (direct, vectorised) | ~74 h | **~4 h** (parallel across days) |
| → processed npz (patched, F10) | ~58 h | **~6 h** → ~2 h parallel |
| reorder/memmap (patched) | ~28 h | **~1 h** |
| training 1 epoch, 23 days / 3.4B samples | ~36 h | **~36 h** (unavoidable) |
| **total** | **~196 h (8 days)** | **~43 h** |

Training dominates and cannot be shortened without changing the science.

### Remaining speed measures
1. Run 4–6 days concurrently in the ingest stages (independent, box has 48 threads).
2. `--dataset-multiprocessing` (`dlrm_s_pytorch.py:1480`, unused today).
3. `--test-freq` > nbatches so it evaluates **once**, not 3×.
4. `--num-workers` 8–16 for the training loader (the 4-day run used 0).

### Disk — the binding constraint

| item | size |
|---|--:|
| raw text, 24 days | **0 — never materialised** (direct parquet path) |
| `day_*.npz` | ~264 GiB |
| `day_*_processed.npz` (int32, plain) | ~700 GiB |
| intermediates + reordered | ~1.4 TB |
| **peak, direct-parquet path** | **~2.3 TB** |
| **available** | **1.3 TB on nvme0 + 1.7 TB on nvme1** |

**Mandatory:** delete each day's raw text immediately after its `.npz` is built, and
delete `*_intermediate_*` once `*_reordered.npz` exists (verified safe: they are
referenced only inside `concatCriteoAdData`, and the loader short-circuits on
`_reordered.npz`). Stage across **both** drives. Even so this is tight — a streaming
delete-as-you-go pipeline is required, not a build-everything-then-clean approach.

### Risk register
- Criteo download availability/rate for 24 days is unverified.
- 3.5 TB peak vs 3.0 TB total — needs the streaming design to work first time.
- ~58 h wall time; the session must survive disconnects (tmux).
- If 24-day DC loss grows sharply (F9), that is a **finding**, not a failure.

---

## 5. PHASE E EXECUTION — 24-day run (started 2026-08-26)

**Config:** days 0-22 train / day 23 test (MLPerf convention, `self.day = days-1`).
D=64, max-ind-range=10M, lr=0.1, batch 2048, 1 epoch. Script: `run_tb24_setup.sh`.

### Optimisations applied (user-approved subset)
| id | change | status |
|---|---|---|
| P1 | vectorised categorical remap | **bit-identical**, verified on 5.1e9 values of day_0 |
| P2 | `savez` not `savez_compressed` (3 sites) | **byte-identical payload** (same md5 of the extracted .npy) |
| MP | `--dataset-multiprocessing` | **bit-identical** (dict merge is `for day in range(days)`) |
| P3 | int32 payloads | **reverted** at user request — float64 kept |
| P4 | direct-parquet ingest | **not used** at user request (written + validated, unused) |
| T1 | `--num-workers > 0` | **BLOCKED** — upstream bug, see below |
| T3 | OMP 24 -> 48 | **no benefit**: 36.60 vs 35.35 ms/it, wall 780 vs 787 s |

### Two upstream OOM bugs found and fixed
`--dataset-multiprocessing` starts **all** days at once in both stages. At 24 days:
- npz build: 24 x ~29 GiB = **701 GiB** -> OOM (503 GiB box)
- remap:     24 x ~48 GiB = **1.2 TB**  -> OOM

Patched both into bounded waves (`DLRM_MP_CONCURRENCY=8`, `DLRM_MP_CONCURRENCY_PROC=6`).
Waves change scheduling only; the merge order is unchanged, so output stays bit-identical.

### T1 is unusable (upstream bug, and the crash is protective)
`AttributeError: 'CriteoDataset' object has no attribute 'day_boundary'`
(`dlrm_data_pytorch.py:286`). `day_boundary` is set only inside
`if index == self.offset_per_file[self.day]`. With workers, only the worker that
receives index 0 ever sets it. Merely initialising it would make workers past day 0
compute `i = index - <wrong boundary>` and **silently read the wrong rows**.

### Stale-artefact hazard (would have corrupted the run silently)
With 24 days every dense category ID changes, but several writes are guarded by
`if not path.exists(...)`, so the 4-day `_fea_count.npz`, `_fea_dict_*.npz` and
`_day_count.npz` would **not** be overwritten, and `processCriteoAdData` prints
"Using existing" for `day_0..3_processed.npz` — mixing old and new ID spaces with no error.
Moved 84 GB of those to `/mnt/ssd4/tb_4day_backup` (kept, so the 4-day results stay
reproducible). `day_0..3.npz` are retained: they hold `int(hex,16) % max_ind_range`,
which is dictionary-independent.

### test-freq
Set to 3,000,000 (> the ~2.19M iterations) so the model is evaluated **once, at the end**.
Rationale: the checkpoint is written only `if is_best`, so N evaluations would select it
as best-of-N **on the day-23 test set the DC benchmark then reports AUC against**. One
evaluation gives the final model with no selection effect. (MLPerf's `run_and_time.sh`
uses 102400 because its metric is time-to-AUC-threshold; that rationale does not apply here.)

### Storage
All four SATA SSDs scanned (95-99% zeros, no filesystem, no recoverable data), formatted
`-m 0`, mounted `/mnt/ssd1..4`, fstab `nofail`. **9.5 TB free** vs a ~2.5 TB peak.
Raw TSVs for days 4-23 land on the SSDs and are symlinked into the dataset dir.

### Expected timeline (from measured per-day costs)
| stage | estimate |
|---|--:|
| fetch 20 days | ~2 h |
| TSV -> day_N.npz (24 days, 8 concurrent) | ~6.6 h |
| -> processed npz (24 days, 6 concurrent) | ~0.7 h |
| reorder | ~2 h |
| training 2.19M iterations @ 36.6 ms | ~22.3 h |
| day-boundary loads + 1 test pass | ~0.9 h |
| **total** | **~35 h (1.5 days)** |

### Run-killer caught before preprocessing started (2026-08-26)
The pipeline writes **all four output categories into one directory** and deletes
nothing. Cumulative on `/mnt/nvme0` (1.6 TB free):

| stage | cumulative |
|---|--:|
| day_N.npz | 0.69 TB |
| + processed | 1.82 TB |
| + intermediates | 3.19 TB |
| + reordered | **4.33 TB** |

It would have died ~6 h in, during the intermediates stage -- after the expensive npz
work. The "8.65 TB free" figure was misleading: that is the sum over six *separate*
mounts and the code can only use one.

**Fix:** 120 pre-created symlinks distributing outputs by category. Verified that
`np.savez`/`np.save` follow dangling symlinks and create the target, and that
`path.exists()` reports a dangling link as absent (so nothing is wrongly skipped as
"using existing").

| mount | holds | needs | free | margin |
|---|---|--:|--:|--:|
| nvme0 | day_N.npz x24 | 0.69 T | 1.52 T | 0.83 T |
| ssd1 / ssd2 | processed x12 each | 0.57 T | 1.31 / 1.59 T | 0.75 / 1.02 T |
| ssd3 / ssd4 | intermediates x12 each | 0.69 T | 1.30 / 1.25 T | 0.61 / 0.56 T |
| **nvme1** | **reordered x24** | 1.13 T | 1.68 T | 0.54 T |

Side benefit: reordered files (re-read at every training day boundary) now sit on fast
NVMe rather than SATA.

Observed data scale: days average 44 GiB -> **4.72 B rows** across 24 days
(canonical figure is ~4.37 B).

### Phase E execution log (actuals)
| stage | actual |
|---|--:|
| fetch 20 days (+ the day_23 recovery) | ~2.5 h |
| counting pass | 0.6 h |
| TSV -> day_N.npz, 3 waves of 8 | ~5.6 h |
| dict merge (serial, via Manager IPC) | ~1.0 h |
| remap, 4 waves of 6 (P1+P2) | ~1.25 h |
| reorder 1st pass (serial, 24x ~10 min) | ~4.1 h |
| reorder 2nd pass (serial, 24x ~7 min) | ~2.6 h |
| **preprocessing total** | **~16.5 h** (est. was ~9-10 h; reorder passes underestimated) |
| training start | 2026-08-27 ~14:35 UTC (CPU fallback; A100 address never provided) |

All 24 reordered files verified on nvme1 (1.3 TB). Intermediates (~1.3 TB, ssd3/4)
deleted after verification. day_23.npz shipped-set totals: X_cat/X_int/y all int32-safe
(round-trip verified per file by the shipper when it runs).

---

## 6. PHASE E COMPLETE (2026-08-28)

- 24-day dataset preprocessed here (16.5 h with P1+P2+bounded-wave MP), shipped to a
  Chameleon A100 as int32 (~665 GB, every file round-trip-verified), trained there in
  16.2 h (`--use-gpu`, RC=0). `models/gpu24/dlrm_terabyte_24day.pt` + 8 periodic ckpts.
- **Baseline AUC 0.798717** on day-23 (MLPerf convention).
- DC results on the fully-trained model: **per-dim block=256: 248x @ -0.060% / 359x @
  -0.092%**. Scalar and INT4 degrade more. `results/perdim_24day_analysis.md`.
- **F9 resolved with a 9-point curve** (`results/dc_vs_training_progress.json`):
  DC loss grows monotonically with training progress (scalar 2.5x, per-dim 3.5x from
  12%->100%), decelerating near convergence; per-dim's advantage stable at 2.1-3.4x
  throughout.
- Remaining (user decisions): CPU fallback run (~72%, redundant — kill or keep for a
  numerics cross-check); README/PROPOSAL update with 24-day numbers; git push (commit
  `7324a72` + new work still local-only); A100 lease release (all artefacts pulled).

### CPU fallback run completed (2026-08-29 ~04:00 UTC)
`TB24_TRAIN_RC=0`, wall ~37.5 h (14:35 Aug 27 -> ~04:00 Aug 29, slowed at times by
concurrent benchmarks/shipping). **Final test accuracy 96.663% — identical to the GPU
run's 96.663% to every printed digit**: two independent machines (Haswell CPU vs A100
CUDA), same data order, matching outcomes. CPU checkpoint at `models/dlrm_terabyte_24day.pt`
(GPU copy at `models/gpu24/`). All training in the project is now complete.

### Follow-on experiments dispatched to the A100 node (briefs sent 2026-08-29)
A: clean serving-latency suite on the idle EPYC 7763 (repairs the contaminated timing
columns; modern-CPU numbers). B: GPU DC inference benchmark (fp32/INT8/DC-scalar/DC-
per-dim, GPU-resident, CUDA-event timing, AUC anchors from the CPU run for validation,
GPU-memory/multi-tenancy accounting). Results to be pulled back when the agent reports.
