#!/usr/bin/env python3
"""
Recompute the matched-loss comparison with FULLY CONSISTENT memory accounting.

Audit findings:
  Bug 1: DC sweep hardcoded +6.0 MB bitmap; pruning sweep used computed 4.02 MB.
         Fix: use consistent computed bitmap (n_total_large / 8 / 1024 / 1024).
  Bug 2: DC sweep used `int(n*hf)` for hot count; pruning used `int(round(...))`.
         Both give identical counts at our hot fractions; not load-bearing.
  Bug 3: DC sweep cold_mb used float division (n_cold/rpb) instead of ceil.
         Difference is at most 8 bytes per table; not load-bearing.

Other terms (small_mb, hot_mb formula, AUC computation, freq profiling) match.
"""
import json
import math

# Constants matching both scripts
EMB_DIM = 16
TABLES = [2, 3, 9, 11, 15, 20, 23, 25]  # large tables on Kaggle

with open("results/dct_domain/kaggle_dc_blocksize_sweep.json") as f:
    dc_data = json.load(f)
with open("results/dct_domain/kaggle_pruning_sweep.json") as f:
    pr_data = json.load(f)

baseline_auc = dc_data["baseline_auc"]
total_emb_mb = dc_data["total_emb_mb"]

# Recover the actual row counts from the existing data.
# We can read total_hot_rows from hot_mb: total_hot_rows = hot_mb * 1024^2 / EMB_DIM
# bitmap = sum_table(n_rows) / 8

# Trust the pruning sweep's bitmap_mb (it's computed from real n_total_large)
BITMAP_MB = pr_data["rows"][0]["bitmap_mb"]
SMALL_MB = pr_data["small_mb"]
print(f"Canonical accounting:")
print(f"  baseline_auc = {baseline_auc:.6f}")
print(f"  total_emb_mb = {total_emb_mb:.2f}")
print(f"  bitmap_mb    = {BITMAP_MB:.4f}  (n_total_large / 8 / 1024^2)")
print(f"  small_mb     = {SMALL_MB:.4f}")
print()

def recompute_total(hot_mb, cold_mb):
    """Total memory with consistent overhead."""
    return hot_mb + cold_mb + SMALL_MB + BITMAP_MB

def recompute_ratio(hot_mb, cold_mb):
    return total_emb_mb / recompute_total(hot_mb, cold_mb)

# Recompute DC rows
print("=" * 100)
print("DC sweep (recomputed with consistent accounting)")
print("=" * 100)
print(f"{'Hot%':>5} {'Mode':>14} {'AUC':>11} {'Loss':>10} {'HotMB':>7} {'ColdMB':>8} "
      f"{'Total v1':>9} {'Ratio v1':>9} {'Total v2':>9} {'Ratio v2':>9}")
print("-" * 100)

dc_recomputed = []
for r in dc_data["rows"]:
    new_total = recompute_total(r["hot_mb"], r["cold_mb"])
    new_ratio = recompute_ratio(r["hot_mb"], r["cold_mb"])
    name = "zero" if r["rpb"] == 0 else f"DC rpb={r['rpb']}"
    print(f"{r['hot_fraction']*100:>4.1f}% {name:>14} {r['auc']:>11.6f} {r['loss_pct']:>+9.4f}% "
          f"{r['hot_mb']:>6.2f} {r['cold_mb']:>7.2f} "
          f"{r['total_mb']:>8.2f} {r['ratio']:>7.1f}x "
          f"{new_total:>8.2f} {new_ratio:>7.1f}x")
    dc_recomputed.append({**r, "total_mb_v2": new_total, "ratio_v2": new_ratio})

# Pruning rows already have correct accounting; just copy
print()
print("=" * 100)
print("Pruning sweep (uint8 kept, already-consistent)")
print("=" * 100)
print(f"{'Sparsity':>9} {'Keep%':>6} {'AUC':>11} {'Loss':>10} {'KeptMB':>8} {'Total':>9} {'Ratio':>9}")
print("-" * 100)
pr_uint8 = [r for r in pr_data["rows"] if r["variant"] == "uint8_kept"]
for r in pr_uint8:
    print(f"{r['sparsity_pct']:>8.1f}% {100-r['sparsity_pct']:>5.2f}% {r['auc']:>11.6f} "
          f"{r['loss_pct']:>+9.4f}% {r['kept_mb']:>7.2f} "
          f"{r['total_mb']:>8.2f} {r['ratio']:>7.1f}x")

# Verify: at matched hot/sparsity, DC zero and pruning uint8 should give same total
print()
print("=" * 100)
print("VERIFICATION: DC zero vs pruning uint8 at matched hot fraction (should be IDENTICAL)")
print("=" * 100)
for hf in [0.043, 0.02, 0.01]:
    sp = (1 - hf) * 100
    dc_zero = [r for r in dc_recomputed if r["hot_fraction"] == hf and r["rpb"] == 0][0]
    # Find closest sparsity in pruning
    closest = min(pr_uint8, key=lambda x: abs(x["sparsity_pct"] - sp))
    sp_match = abs(closest["sparsity_pct"] - sp) < 0.01
    print(f"  hot={hf*100:.1f}% (sp={sp:.1f}%):")
    print(f"    DC zero v2: total={dc_zero['total_mb_v2']:.4f} MB, ratio={dc_zero['ratio_v2']:.2f}x, AUC={dc_zero['auc']:.6f}")
    print(f"    Pruning sp={closest['sparsity_pct']}%: total={closest['total_mb']:.4f} MB, ratio={closest['ratio']:.2f}x, AUC={closest['auc']:.6f}")
    auc_diff = dc_zero['auc'] - closest['auc']
    total_diff = dc_zero['total_mb_v2'] - closest['total_mb']
    print(f"    Δ total: {total_diff:+.4f} MB | Δ AUC: {auc_diff:+.6f}")
    if not sp_match:
        print(f"    (sparsity mismatch: {sp:.2f} vs {closest['sparsity_pct']:.2f})")
    print()

# Recompute matched-loss table with consistent accounting
def interp(target_loss, rows):
    pts = sorted([(r["loss_pct"], r["ratio"]) for r in rows])
    losses = [p[0] for p in pts]
    ratios = [p[1] for p in pts]
    if target_loss <= losses[0]: return ratios[0]
    if target_loss >= losses[-1]: return ratios[-1]
    for i in range(len(losses) - 1):
        if losses[i] <= target_loss <= losses[i+1]:
            t = (target_loss - losses[i]) / (losses[i+1] - losses[i])
            return ratios[i] * (1 - t) + ratios[i+1] * t
    return ratios[-1]

print("=" * 100)
print("CORRECTED MATCHED-LOSS TABLE (consistent accounting)")
print("=" * 100)
print(f"{'DC config':>20} {'DC ratio':>10} {'AUC loss':>10} {'Pr ratio':>10} {'Gap':>10}")
print("-" * 100)
matched = []
dc_pts = [r for r in dc_recomputed if r["rpb"] > 0]
dc_pts.sort(key=lambda r: r["ratio_v2"])
for r in dc_pts:
    name = f"rpb={r['rpb']} hot={r['hot_fraction']*100:.1f}%"
    target = r["loss_pct"]
    pr_ratio = interp(target, pr_uint8)
    gap = pr_ratio / r["ratio_v2"]
    direction = "pruning wins" if gap > 1.02 else ("DC wins" if gap < 0.98 else "TIE")
    print(f"{name:>20} {r['ratio_v2']:>9.1f}x {target:>+9.4f}% {pr_ratio:>9.1f}x "
          f"{gap:>8.2f}x ({direction})")
    matched.append({"config": name, "dc_ratio": r["ratio_v2"], "auc_loss": target,
                    "pr_ratio_at_same_loss": pr_ratio, "gap": gap})

# Save
with open("results/dct_domain/matched_loss_v2.json", "w") as f:
    json.dump({
        "baseline_auc": baseline_auc,
        "total_emb_mb": total_emb_mb,
        "bitmap_mb": BITMAP_MB,
        "small_mb": SMALL_MB,
        "dc_recomputed": dc_recomputed,
        "pr_uint8": pr_uint8,
        "matched_loss": matched,
    }, f, indent=2)
print(f"\nSaved results/dct_domain/matched_loss_v2.json")

# Summary stats
gaps = [m["gap"] for m in matched]
print(f"\nGap stats: min={min(gaps):.2f}x, median={sorted(gaps)[len(gaps)//2]:.2f}x, max={max(gaps):.2f}x")
