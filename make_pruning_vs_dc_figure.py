#!/usr/bin/env python3
"""
Compare DC sweep against frequency-pruning sweep on the same Pareto plot.
Compute matched-loss table: at each DC point, what does pruning need to match?
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "results/dct_domain"

with open("results/dct_domain/kaggle_dc_blocksize_sweep.json") as f:
    dc_data = json.load(f)
with open("results/dct_domain/kaggle_pruning_sweep.json") as f:
    pr_data = json.load(f)

baseline_auc = dc_data["baseline_auc"]
total_emb_mb = dc_data["total_emb_mb"]
print(f"Baseline AUC = {baseline_auc:.6f}, total fp32 = {total_emb_mb:.0f} MB")

dc_rows = dc_data["rows"]
pr_uint8 = [r for r in pr_data["rows"] if r["variant"] == "uint8_kept"]
pr_fp32 = [r for r in pr_data["rows"] if r["variant"] == "fp32_kept"]

# Sort by ratio
pr_uint8.sort(key=lambda r: r["ratio"])
pr_fp32.sort(key=lambda r: r["ratio"])

# ============================================================
# Matched-loss table: at each DC operating point, find what pruning needs
# ============================================================

def interpolate_pruning_compression(target_loss, pruning_rows):
    """At target AUC loss, return interpolated compression ratio from pruning sweep."""
    # Sort by loss
    pts = sorted([(r["loss_pct"], r["ratio"]) for r in pruning_rows])
    losses = [p[0] for p in pts]
    ratios = [p[1] for p in pts]
    if target_loss <= losses[0]:
        return ratios[0]
    if target_loss >= losses[-1]:
        return ratios[-1]
    for i in range(len(losses) - 1):
        if losses[i] <= target_loss <= losses[i+1]:
            t = (target_loss - losses[i]) / (losses[i+1] - losses[i])
            return ratios[i] * (1 - t) + ratios[i+1] * t
    return ratios[-1]

print("\n" + "=" * 96)
print("MATCHED-LOSS COMPARISON: at each DC AUC, what compression ratio does pruning need?")
print("=" * 96)
print(f"{'DC config':>18} {'DC ratio':>10} {'AUC loss':>10} {'Pruning ratio':>15} {'Ratio gap':>12}")
print("-" * 96)

dc_pts = [r for r in dc_rows if r["rpb"] > 0]
dc_pts.sort(key=lambda r: r["ratio"])

matched_loss_rows = []
for r in dc_pts:
    name = f"rpb={r['rpb']} hot={r['hot_fraction']*100:.1f}%"
    target = r["loss_pct"]
    pr_ratio = interpolate_pruning_compression(target, pr_uint8)
    gap = pr_ratio / r["ratio"]
    direction = "pruning wins" if pr_ratio > r["ratio"] else "DC wins"
    print(f"{name:>18} {r['ratio']:>9.1f}x {target:>+9.4f}% {pr_ratio:>14.1f}x "
          f"{gap:>10.2f}x ({direction})")
    matched_loss_rows.append({
        "dc_config": name, "dc_ratio": r["ratio"], "auc_loss": target,
        "pruning_ratio_at_same_loss": pr_ratio, "gap": gap,
    })

# ============================================================
# Pareto figure: DC vs pruning
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)

# Pruning lines
ax.plot([r["ratio"] for r in pr_uint8], [r["loss_pct"] for r in pr_uint8],
        "o-", color="#d62728", markersize=5, linewidth=1.5,
        label="Pruning (uint8 kept)", zorder=3)
ax.plot([r["ratio"] for r in pr_fp32], [r["loss_pct"] for r in pr_fp32],
        "s--", color="#9467bd", markersize=4, linewidth=1.0, alpha=0.55,
        label="Pruning (fp32 kept)", zorder=2)

# DC scatter, colored by hot fraction, marker by rpb
hot_colors = {0.043: "#1f77b4", 0.02: "#ff7f0e", 0.01: "#2ca02c"}
rpb_markers = {1: "o", 4: "s", 8: "^", 16: "D"}
for hf in [0.043, 0.02, 0.01]:
    subset = [r for r in dc_rows if r["hot_fraction"] == hf and r["rpb"] > 0]
    subset.sort(key=lambda r: r["ratio"])
    for r in subset:
        ax.scatter(r["ratio"], r["loss_pct"], color=hot_colors[hf],
                   marker=rpb_markers[r["rpb"]], s=85, edgecolors="white",
                   linewidths=1.0, zorder=4)
    # Connect DC points at same hot fraction
    ax.plot([r["ratio"] for r in subset], [r["loss_pct"] for r in subset],
            "-", color=hot_colors[hf], alpha=0.3, linewidth=1)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#d62728", marker="o", linewidth=1.5, label="Pruning (uint8)"),
    Line2D([0], [0], color="#9467bd", marker="s", linewidth=1.0, linestyle="--",
           label="Pruning (fp32)"),
    Line2D([0], [0], color="#1f77b4", marker="o", linewidth=0, markersize=8,
           label="DC @ 4.3% hot"),
    Line2D([0], [0], color="#ff7f0e", marker="o", linewidth=0, markersize=8,
           label="DC @ 2.0% hot"),
    Line2D([0], [0], color="#2ca02c", marker="o", linewidth=0, markersize=8,
           label="DC @ 1.0% hot"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9, frameon=True)

ax.set_xscale("log")
ax.set_xlabel("Compression ratio (vs fp32, log scale)")
ax.set_ylabel("AUC loss (%)")
ax.set_title("DC vs frequency-pruning Pareto on Criteo Kaggle (D=16)")
ax.grid(True, alpha=0.3, which="both")
ax.invert_yaxis()

plt.tight_layout()
out = os.path.join(OUT_DIR, "kaggle_dc_vs_pruning_pareto.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
print(f"\nSaved {out}")
plt.close()

# Save the matched-loss table
with open(os.path.join(OUT_DIR, "matched_loss_table.json"), "w") as f:
    json.dump({
        "baseline_auc": baseline_auc,
        "matched_loss_rows": matched_loss_rows,
    }, f, indent=2)
print(f"Saved {OUT_DIR}/matched_loss_table.json")
