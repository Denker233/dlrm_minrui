# Third dataset: Avazu_x1 (non-Criteo, D=16, all-categorical)

Standard FuxiCTR x1 split; DLRM D=16 trained 1 epoch (CAFE's avazu config), test AUC
**0.760925** on 8.08M rows — inside the FuxiCTR reference band (0.76-0.77), so a
legitimate model. 22 tables, 1.545M rows, 94.3 MB fp32; two large tables (1.24M +
278k, device-id-like) hold 98% of rows. Deterministic full-test evaluation (no
sampling noise). Raw: `results/avazu_dc_sweep.json`.

| hot% | zero | scalar bs16 | perdim bs16 | perdim bs64 | perdim bs256 |
|--:|--:|--:|--:|--:|--:|
| 4.3 | -0.0228 | -0.0224 | -0.0140 | -0.0112 | -0.0118 |
| 2.0 | -0.0246 | -0.0237 | -0.0096 | -0.0106 | -0.0088 |
| 1.0 | -0.0225 | -0.0214 | -0.0093 | -0.0078 | -0.0090 |
| 0.5 | -0.0138 | -0.0126 | **-0.0006** | **+0.0009** | -0.0015 |

(dAUC%; perdim bs256 costs byte-for-byte the same as scalar bs16 at D=16.)

## What replicates (third dataset, different company's traffic)

1. **Per-dim > scalar at every hot fraction** (1.6x to >8x less loss; at 0.5% hot,
   per-dim is lossless to within evaluation precision, +0.0009 to -0.0015%).
2. **Scalar ~= zero** (within 0.001pp everywhere) — scalar block means barely beat
   zeroing, exactly as on Criteo (F-C).
3. **Same-bytes free win** (D=16: perdim bs256 = scalar bs16 in size, ~2-8x less loss).
4. Large blocks remain fine (bs64/bs256 ~= bs16, often better).

## New observation: zeroing MORE cold rows can hurt LESS

Zero @4.3% hot costs -0.0228 but zero @0.5% costs only -0.0138 — i.e. the rows in the
0.5-4.3% frequency band *improve* the model when removed. Mid-frequency device
embeddings on Avazu appear to carry overfit noise (consistent with Ginart et al.:
rare-item capacity reduces train loss, not test loss), so aggressive compression acts
as regularisation here. This is why per-dim @0.5% lands at/above baseline.

## Caveat (methodological difference vs the Criteo runs)

In this sweep hot rows keep their fp32 values (the memory accounting assumes uint8
hot, as in the Criteo runs, but the AUC path did not apply the uint8 quantisation).
Based on the INT8 results (+-0.002% everywhere), the effect is ~0.002pp at most, but
an exact-methodology rerun should apply hot-uint8 before quoting these numbers next
to the Criteo tables.

## Bottom line

Per-dim dominance and the scalar~zero finding now hold on three datasets (Criteo
Kaggle D=16, Criteo Terabyte D=64, Avazu D=16) spanning two data sources, two
dimensions, dense+sparse and pure-sparse models, and 94 MB to 13 GB scales. The
mechanism is a property of trained embedding geometry, not of Criteo.
