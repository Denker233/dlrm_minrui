# DC Embedding Compression — Complete Progress Record & Findings

**Period:** 2026-08-23 → 2026-08-30 · **Branch:** `codecs_cpu_gpu`
Companion docs: `SETUP_PLAN.md` (chronological work log),
`results/perdim_24day_analysis.md` (definitive 24-day results),
`results/unified_comparison.md`, `results/sort_key_analysis.md`, `results/nvme_benchmark.md`.

---

## 1. What was accomplished, in order

| # | Milestone | Outcome |
|--|---|---|
| 1 | 4-day Terabyte model retrained (1 full epoch, CPU) | baseline AUC 0.789235 — superseded the undertrained 0.768820 all prior Terabyte tables used |
| 2 | Unified benchmark framework | one code path measuring AUC **and** full forward latency, each config gathering from its **real** storage layout (fp32 / uint8 / nibble-packed / DC / SSD), PyTorch and C-fused variants |
| 3 | Optimised NVMe baseline | 9.30 → 0.39 ms for 1,660 cold reads (24x): O_DIRECT 512 B, IOPOLL, registered buffers, NUMA-pinned threads, 2-drive striping, sector packing |
| 4 | **Per-dimension block means discovered** | Pareto-dominant over scalar DC on both datasets |
| 5 | Error decomposition (theory) | explains per-dim, proves value-sort optimal, predicts cross-dataset behaviour |
| 6 | Kaggle (D=16) cross-check | per-dim block=256 is *free* there (same bytes as scalar, 2.1x less loss) |
| 7 | Preprocessing rebuilt | ~10x faster (vectorised remap bit-identical on 5.1e9 values; plain savez byte-identical; two upstream multiprocessing OOM bugs fixed with bounded waves) |
| 8 | **Full 24-day Terabyte pipeline** | fetched (1.03 TB), preprocessed 16.5 h across 6 mounts, shipped as int32 (~665 GB, round-trip-verified) to a Chameleon A100 |
| 9 | **24-day model trained** | A100, 16.2 h, days 0-22 / day 23 (MLPerf split), RC=0; independent CPU run matched final test accuracy (96.663%) to every digit; SHA-256-verified transfer back |
| 10 | DC on the fully-trained model | headline results below; 8 periodic checkpoints → training-progress curve |
| 11 | Serving study on 3 hardware generations | 2015 Haswell, EPYC 7763, A100 80GB |
| 12 | Block-size bracket 512/1024 | 256 confirmed a plateau, not a truncated search |

## 2. Definitive numbers (24-day model, day-23 test, baseline AUC 0.798717)

| method | mem | ratio | ΔAUC% |
|---|--:|--:|--:|
| INT8 whole-table | 3.3 GB | 4x | −0.0005 |
| INT4 whole-table | 1.6 GB | 8x | −0.3419 |
| DC scalar block=16 @1% | 48 MB | 273x | −0.1351 |
| **DC per-dim block=256 @1%** | **53 MB** | **248x** | **−0.0596** |
| **DC per-dim block=256 @0.5%** | **36 MB** | **359x** | **−0.0917** |
| (per-dim block=512 @0.5%) | 33 MB | 393x | −0.0915 |

## 3. Key findings

**F-A. Per-dimension block means dominate scalar block means** — 1.8–2.9x less AUC loss
at equal or lower memory, on 4-day and 24-day Terabyte and on Kaggle. At D=16 the
per-dim block=256 variant costs byte-for-byte the same as scalar block=16.

**F-B. The error decomposition explains everything.** Scalar-DC squared error splits
exactly into a within-row term (ordering-independent; **99.7–100% of the error at
D=64**, 97–99.6% at D=16) and a between-row term (the only part sorting can touch).
Consequences, all verified: value-sort is *exactly optimal* (not a heuristic); PCA-sort
cannot beat it and measured worse on all 8 tables tested — **the README's old 17–26%
PCA claim does not replicate**; per-row means are pointless (16x storage for ≤0.3% of
the error); per-dim means are the only variant attacking the dominant term; and the
per-dim advantage is larger at D=64 than D=16, as predicted.

**F-C. Scalar DC barely beats zeroing** (1.5x); per-dim beats zeroing 4.8x. The old
method was much closer to the "cold=0" floor than its framing implied.

**F-D. Compressibility depends on training budget.** The 4-day model's cold rows sat at
random init (median row norm 0.00147 vs the init-theoretic 0.00146 — untrained noise).
The 9-checkpoint curve shows DC loss growing monotonically with training (scalar 2.5x,
per-dim 3.5x from 12%→100% trained) and **flattening at convergence**. Method ranking
stable throughout. Report compression results with the training config attached.

**F-E. Serving latency (batch 2048, full forward):**
- 2015 Haswell (C-fused): DC per-dim 7.41 ms ≈ INT8 7.40 < fp32 9.70
- EPYC 7763 (idle, 24T): **DC at fp32 parity** (10.1 vs 9.5 ms, within noise) at
  248–359x; INT8 *loses* its edge on modern CPUs (10.97 ms)
- A100 (GPU-resident, validated to <1e-6 vs CPU AUC): fp32 fastest (1.49 ms — HBM
  makes 13 GB cheap); DC 4.6–5.6 ms but **~170 models/80 GB vs 6 → 28x multi-tenancy**
- Implementation matters: PyTorch DC was *slower* than fp32 on CPU; fused C made it
  faster. fp32 gained nothing from C (index_select is already one tuned op).

**F-F. The honest SSD story.** "SSD infeasible at 91 ms" was SATA + serial pread. An
optimised NVMe path reaches 1.79 ms (0.39 ms sector-packed) for the same reads and
9.35 ms end-to-end — viable and lossless. DC's real advantage: faster, zero extra
hardware, zero I/O threads.

**F-G. INT4 is dominated everywhere** (−0.25 to −0.34% at only 8x; worse on the trained
model). INT8 is the baseline to respect: essentially lossless at 4x.

**F-H. On GPU, DC's footprint is dominated by its index maps** (410 MB of hot_pos/
cold_rank vs 19–151 MB of data) — threshold-based dispatch (drop the maps) is the
highest-value next experiment, and grows in importance with table size.

**F-I. Block=256 is a plateau**: 512/1024 change AUC by ≤0.007pp (noise) for ~10% more
ratio.

**F-J. Engineering findings:** preprocessing was 10x slower than necessary (854 ns/eleme
nt Python dict loop; zlib at 9 MB/s); `--dataset-multiprocessing` OOMs at 24 days
without bounded waves (701 GB / 1.2 TB spikes); `num-workers>0` crashes on the stateful
day-boundary logic (and naive fixes would silently read wrong rows); npz day-file
inflation cost the GPU run ~12 of its 16 h (single-threaded zip path — raw binary
format recommended before any future epoch).

## 4. Corrections to previously-published claims

1. All Terabyte tables built on baseline 0.768820 are superseded (undertrained model).
2. PCA-sort recommendation withdrawn (F-B): use value-sort.
3. "SSD cold serving infeasible (91.3 ms)" restated per F-F.
4. Timing columns of the 24-day AUC sweep were contamination-flagged and re-measured
   clean on the EPYC (results/a100/).

## 5. Artefact map

Models: `models/dlrm_terabyte_4day.pt` · `models/dlrm_terabyte_24day.pt` (CPU) ·
`models/gpu24/dlrm_terabyte_24day.pt` + `ckpt_it{250k..2M}.pt` (GPU, SHA-verified) ·
`models/dlrm_kaggle_correct.pt`.
Data (local): 24-day reordered npz on `/mnt/nvme1/tbout` (float64) and on the A100 as
int32; raw + intermediates cleaned per the disk plan.
Benchmarks: `bench_unified_terabyte.py` (framework) · `bench_cpp_terabyte.py` /
`libembfwd.c` (C-fused) · `bench_perdim_terabyte.py` / `bench_perdim_kaggle.py` ·
`bench_curve_terabyte.py` · `bench_gpu_inference.py` (A100) · `bench_iouring_opt.c` /
`libcoldread.c` (NVMe) · `tb_ship_reordered.py`, `run_tb24_setup.sh` (pipeline).
Results: `results/perdim_24day_analysis.md` (start here) · `results/a100/` ·
`results/*.json` per experiment.

## 6. Open items

D5 threshold dispatch (GPU-motivated, F-H) · multi-epoch curve extension · loader raw-
binary format · PROPOSAL rewrite around per-dim · A100 lease release decision.

---

## 7. Addendum (2026-08-30 → 09-01): third dataset + ordering study

**Third dataset (Avazu_x1, non-Criteo, D=16, all-categorical; test AUC 0.7609, in the
FuxiCTR reference band).** Everything replicates: per-dim > scalar at every hot
fraction (1.6x to >8x less loss); scalar ~= zero within 0.001pp; per-dim @0.5% is
LOSSLESS (-0.0015 to +0.0009%). New observation: zeroing *more* cold rows can hurt
*less* (mid-frequency embeddings carry overfit noise — compression as regularisation).
`results/avazu_analysis.md`.

**F-K. Ordering study — the sort is essential, and PC1 beats value-sort for per-dim.**
Unsorted per-dim loses 3.5-3.8x more AUC than sorted (and is worse than sorted scalar):
the ordering *creates* the block homogeneity per-dim means exploit. The scalar
optimality proof for value-sort does not extend to per-dim (vector summaries make
ordering a clustering problem); measured on the 24-day model, PC1-sort/zorder give
10-17% less loss than value-sort. **New best operating points: 248x @ -0.053% (pc1,
1% hot) and 359x @ -0.076% (zorder, 0.5%).** pc1|pc2-lex underperformed despite a good
MSE proxy score — AUC-validate orderings, don't trust proxies.
`results/perdim_sort_screen.json`, `results/perdim_order_ablation.json`,
`results/a100/perdim_order_hf005.json`.

**F-L. Ordering has no latency cost — measured, twice, order-reversed.** All orderings
latency-equivalent within run noise (~±0.7 ms); first-config-in-run position effects
exceed any between-ordering gap. Encode cost: value ~6 s, pc1 ~17 s, zorder ~28 s
(one-time, all 48.4M cold rows). Methodological: single-run cross-config timing
comparisons on this class of machine carry ±0.5-1 ms noise — do not interpret smaller
differences, anywhere in this project's tables.

**Final recommended method: per-dim block means, block=256, 4-bit, PC1-sorted.**
Every clause experiment-backed: 248x @ -0.053% / 359x @ -0.076% on the 24-day MLPerf
config; latency-equivalent to all alternatives; ~17 s one-time encode.
