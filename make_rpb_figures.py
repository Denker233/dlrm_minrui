#!/usr/bin/env python3
"""
Generate two figures from results/dct_domain/kaggle_dc_blocksize_sweep.json:

1. Pareto curve: AUC loss (y) vs compression ratio (x), one line per hot fraction.
2. Linearity: AUC recovery over zero (y) vs cold-storage MB (x), one line per hot.
   Should show approximately linear trend → "DC info is linearly compressible per MB."
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "results/dct_domain/kaggle_dc_blocksize_sweep.json"
OUT_DIR = "results/dct_domain"

with open(DATA_PATH) as f:
    data = json.load(f)

baseline_auc = data["baseline_auc"]
total_emb_mb = data["total_emb_mb"]
rows = data["rows"]

# Group by hot fraction
hot_fractions = sorted(set(r["hot_fraction"] for r in rows), reverse=True)

# Colors and markers
colors = {0.043: "#1f77b4", 0.02: "#ff7f0e", 0.01: "#2ca02c"}
markers = {0: "X", 1: "o", 4: "s", 8: "^", 16: "D"}
labels = {0: "zero", 1: "DC rpb=1", 4: "DC rpb=4", 8: "DC rpb=8", 16: "DC rpb=16"}

# ============================================================
# Figure 1: Pareto curve (compression ratio vs AUC loss)
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)

for hf in hot_fractions:
    subset = [r for r in rows if r["hot_fraction"] == hf]
    subset.sort(key=lambda r: r["ratio"])
    ratios = [r["ratio"] for r in subset]
    losses = [r["loss_pct"] for r in subset]
    ax.plot(ratios, losses, "-", color=colors[hf], alpha=0.55, linewidth=1.4)
    for r in subset:
        rpb = r["rpb"]
        ax.scatter(r["ratio"], r["loss_pct"], color=colors[hf],
                   marker=markers[rpb], s=70, edgecolors="white", linewidths=0.9,
                   label=f"{labels[rpb]} @ {int(hf*100*10)/10:.1f}% hot" if hf == hot_fractions[0] else None,
                   zorder=3)

# Annotate just the rpb labels next to points at 4.3% hot (clearest line)
for r in [r for r in rows if r["hot_fraction"] == 0.043]:
    rpb = r["rpb"]
    name = labels[rpb]
    ax.annotate(name, (r["ratio"], r["loss_pct"]),
                textcoords="offset points", xytext=(6, 5),
                fontsize=8, color="#444")

# Hot fraction lines legend
from matplotlib.lines import Line2D
hf_handles = [Line2D([0], [0], color=colors[h], linewidth=2,
                     label=f"{int(h*1000)/10:.1f}% hot") for h in hot_fractions]
mk_handles = [Line2D([0], [0], color="gray", marker=markers[r], linestyle="",
                     markersize=7, label=labels[r]) for r in [0, 1, 4, 8, 16]]
leg1 = ax.legend(handles=hf_handles, loc="upper left", fontsize=9, frameon=True,
                 title="Hot fraction")
ax.add_artist(leg1)
ax.legend(handles=mk_handles, loc="upper right", fontsize=9, frameon=True, title="Cold")

ax.set_xscale("log")
ax.set_xlabel("Compression ratio (vs fp32, log scale)")
ax.set_ylabel("AUC loss (%)")
ax.set_title("DC block-size sweep: Pareto frontier on Criteo Kaggle (D=16)")
ax.grid(True, alpha=0.3, which="both")
ax.invert_yaxis()  # Lower AUC loss is better → put it on top

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "kaggle_rpb_pareto.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.savefig(out1.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved {out1}")
plt.close()

# ============================================================
# Figure 2: Linearity-per-MB
#   x = cold storage MB (zero is at x=0)
#   y = AUC recovery over zero (= zero_loss - method_loss)
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)

for hf in hot_fractions:
    subset = [r for r in rows if r["hot_fraction"] == hf]
    # Find zero baseline
    zero_row = [r for r in subset if r["rpb"] == 0][0]
    zero_loss = zero_row["loss_pct"]
    # Sort by cold_mb
    dc_rows = sorted([r for r in subset if r["rpb"] > 0], key=lambda r: r["cold_mb"])
    xs = [0.0] + [r["cold_mb"] for r in dc_rows]
    ys = [0.0] + [zero_loss - r["loss_pct"] for r in dc_rows]
    ax.plot(xs, ys, "-", color=colors[hf], alpha=0.55, linewidth=1.4,
            label=f"{int(hf*1000)/10:.1f}% hot")
    for x, y, r in zip(xs[1:], ys[1:], dc_rows):
        rpb = r["rpb"]
        ax.scatter(x, y, color=colors[hf], marker=markers[rpb],
                   s=70, edgecolors="white", linewidths=0.9, zorder=3)
        ax.annotate(f"rpb={rpb}", (x, y),
                    textcoords="offset points", xytext=(6, -3),
                    fontsize=7.5, color="#444")
    # Mark zero
    ax.scatter(0, 0, color=colors[hf], marker="X", s=70,
               edgecolors="white", linewidths=0.9, zorder=3)

ax.set_xlabel("Cold storage (MB)")
ax.set_ylabel("AUC recovery over zero-out (%)")
ax.set_title("DC info is approximately linear in cold-storage MB (Kaggle, D=16)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", title="Hot fraction", fontsize=9)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "kaggle_rpb_linearity.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.savefig(out2.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved {out2}")
plt.close()

# ============================================================
# Print numerical efficiency table for sanity check
# ============================================================
print("\nRecovery efficiency (% AUC recovered per MB of cold storage):")
print(f"{'Hot%':>6} {'rpb=1':>10} {'rpb=4':>10} {'rpb=8':>10} {'rpb=16':>10}")
for hf in hot_fractions:
    subset = [r for r in rows if r["hot_fraction"] == hf]
    zero_loss = [r for r in subset if r["rpb"] == 0][0]["loss_pct"]
    line = f"{hf*100:>5.1f}%"
    for rpb in [1, 4, 8, 16]:
        rr = [r for r in subset if r["rpb"] == rpb][0]
        recovery = zero_loss - rr["loss_pct"]
        eff = recovery / rr["cold_mb"]
        line += f"  {eff:>8.5f}"
    print(line)
