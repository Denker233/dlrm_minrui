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

### Budget at today's (already patched) speeds

| stage | 4 days actual | 24 days naive | with the plan below |
|---|--:|--:|--:|
| download + convert | 8.5 h | ~51 h | **~9 h** (parallel fetch) |
| npz build | 8.5 h | ~51 h | **~9 h** (parallel days) |
| processed npz | 10 h → 1 h (patched) | ~6 h | **~1.5 h** (parallel days) |
| reorder/memmap | 4.5 h → ~0.3 h (patched) | ~2 h | **~2 h** |
| training 1 epoch | 6 h | ~36 h | **~36 h** (unavoidable) |
| **total** | ~38 h | **~146 h** | **~58 h** |

### Speed measures, in order of payoff

1. **Parallel day fetch + convert (biggest win).** `tb_fetch_day.sh` runs one day at a
   time; days are independent. Run 4–6 concurrently — network-bound, and the box has 48
   threads. 51 h → ~9 h.
2. **`--dataset-multiprocessing`** (`dlrm_s_pytorch.py:1480`, currently unused) parallelises
   per-day processing across processes.
3. **The `data_utils.py` patches are already in** (F10) — 10× on the processed-npz stage.
4. **Skip the reorder second pass if possible** — it exists to shuffle across days; with
   23 training days the benefit is marginal but the cost is 23 × ~1 min (now that it is
   plain `savez`). Keep it.
5. **`--test-freq`**: set > nbatches so it evaluates **once at the end**. The 4-day run did
   3 full passes over 181M test samples; at 24 days that is hours wasted.
6. **`--num-workers`**: the 4-day run used 0. Use 8–16 for the training loader.
7. **Training itself is ~36 h and cannot be shortened** without changing the science.
   Run it under tmux, monitored.

### Disk — the binding constraint

| item | size |
|---|--:|
| raw text, 24 days | ~1.15 TB |
| `day_*.npz` | ~264 GiB |
| `day_*_processed.npz` (int32, plain) | ~700 GiB |
| intermediates + reordered | ~1.4 TB |
| **peak if nothing is deleted** | **~3.5 TB** |
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
