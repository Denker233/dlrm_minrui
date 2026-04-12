#!/usr/bin/env python3
"""Regenerate Pareto + linearity figures using corrected (consistent-accounting) data."""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "results/dct_domain"

with open("results/dct_domain/matched_loss_v2.json") as f:
    d = json.load(f)

dc_recomputed = d["dc_recomputed"]
pr_uint8 = d["pr_uint8"]

hot_colors = {0.043: "#1f77b4", 0.02: "#ff7f0e", 0.01: "#2ca02c"}
rpb_markers = {0: "X", 1: "o", 4: "s", 8: "^", 16: "D"}
labels = {0: "zero", 1: "DC rpb=1", 4: "DC rpb=4", 8: "DC rpb=8", 16: "DC rpb=16"}

# ============================================================
# Figure: DC Pareto with rpb sweep (CORRECTED)
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)

for hf in [0.043, 0.02, 0.01]:
    subset = [r for r in dc_recomputed if r["hot_fraction"] == hf]
    subset.sort(key=lambda r: r["ratio_v2"])
    ratios = [r["ratio_v2"] for r in subset]
    losses = [r["loss_pct"] for r in subset]
    ax.plot(ratios, losses, "-", color=hot_colors[hf], alpha=0.55, linewidth=1.4)
    for r in subset:
        rpb = r["rpb"]
        ax.scatter(r["ratio_v2"], r["loss_pct"], color=hot_colors[hf],
                   marker=rpb_markers[rpb], s=70, edgecolors="white", linewidths=0.9, zorder=3)

# rpb annotations on the 4.3% line
for r in [r for r in dc_recomputed if r["hot_fraction"] == 0.043]:
    rpb = r["rpb"]
    ax.annotate(labels[rpb], (r["ratio_v2"], r["loss_pct"]),
                textcoords="offset points", xytext=(6, 5),
                fontsize=8, color="#444")

from matplotlib.lines import Line2D
hf_handles = [Line2D([0], [0], color=hot_colors[h], linewidth=2,
                     label=f"{h*100:.1f}% hot") for h in [0.043, 0.02, 0.01]]
mk_handles = [Line2D([0], [0], color="gray", marker=rpb_markers[r], linestyle="",
                     markersize=7, label=labels[r]) for r in [0, 1, 4, 8, 16]]
leg1 = ax.legend(handles=hf_handles, loc="upper left", fontsize=9, frameon=True, title="Hot fraction")
ax.add_artist(leg1)
ax.legend(handles=mk_handles, loc="upper right", fontsize=9, frameon=True, title="Cold")

ax.set_xscale("log")
ax.set_xlabel("Compression ratio (vs fp32, log scale)")
ax.set_ylabel("AUC loss (%)")
ax.set_title("DC sweep on Criteo Kaggle (D=16, corrected accounting)")
ax.grid(True, alpha=0.3, which="both")
ax.invert_yaxis()
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "kaggle_rpb_pareto_v2.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.savefig(out1.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved {out1}")
plt.close()

# ============================================================
# Figure: linearity-per-MB (CORRECTED)
# ============================================================
fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)

for hf in [0.043, 0.02, 0.01]:
    subset = [r for r in dc_recomputed if r["hot_fraction"] == hf]
    zero_row = [r for r in subset if r["rpb"] == 0][0]
    zero_loss = zero_row["loss_pct"]
    dc_rows = sorted([r for r in subset if r["rpb"] > 0], key=lambda r: r["cold_mb"])
    xs = [0.0] + [r["cold_mb"] for r in dc_rows]
    ys = [0.0] + [zero_loss - r["loss_pct"] for r in dc_rows]
    ax.plot(xs, ys, "-", color=hot_colors[hf], alpha=0.55, linewidth=1.4,
            label=f"{hf*100:.1f}% hot")
    for x, y, r in zip(xs[1:], ys[1:], dc_rows):
        ax.scatter(x, y, color=hot_colors[hf], marker=rpb_markers[r["rpb"]],
                   s=70, edgecolors="white", linewidths=0.9, zorder=3)
        ax.annotate(f"rpb={r['rpb']}", (x, y),
                    textcoords="offset points", xytext=(6, -3),
                    fontsize=7.5, color="#444")
    ax.scatter(0, 0, color=hot_colors[hf], marker="X", s=70,
               edgecolors="white", linewidths=0.9, zorder=3)

ax.set_xlabel("Cold storage (MB)")
ax.set_ylabel("AUC recovery over zero-out (%)")
ax.set_title("DC info recovery is linear in cold-storage budget (Kaggle, D=16)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", title="Hot fraction", fontsize=9)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "kaggle_rpb_linearity_v2.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.savefig(out2.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved {out2}")
plt.close()

# ============================================================
# Figure: DC vs pruning Pareto (CORRECTED)
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)

# Pruning curve
pr_sorted = sorted(pr_uint8, key=lambda r: r["ratio"])
ax.plot([r["ratio"] for r in pr_sorted], [r["loss_pct"] for r in pr_sorted],
        "o-", color="#d62728", markersize=5, linewidth=1.8,
        label="Pruning (uint8 kept)", zorder=3)

# DC scatter
for hf in [0.043, 0.02, 0.01]:
    subset = sorted([r for r in dc_recomputed if r["hot_fraction"] == hf and r["rpb"] > 0],
                    key=lambda r: r["ratio_v2"])
    for r in subset:
        ax.scatter(r["ratio_v2"], r["loss_pct"], color=hot_colors[hf],
                   marker=rpb_markers[r["rpb"]], s=85, edgecolors="white",
                   linewidths=1.0, zorder=4)
    ax.plot([r["ratio_v2"] for r in subset], [r["loss_pct"] for r in subset],
            "-", color=hot_colors[hf], alpha=0.4, linewidth=1)

legend_elements = [
    Line2D([0], [0], color="#d62728", marker="o", linewidth=1.8, label="Pruning"),
    Line2D([0], [0], color="#1f77b4", marker="o", linewidth=0, markersize=8, label="DC @ 4.3% hot"),
    Line2D([0], [0], color="#ff7f0e", marker="o", linewidth=0, markersize=8, label="DC @ 2.0% hot"),
    Line2D([0], [0], color="#2ca02c", marker="o", linewidth=0, markersize=8, label="DC @ 1.0% hot"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9, frameon=True)

ax.set_xscale("log")
ax.set_xlabel("Compression ratio (vs fp32, log scale)")
ax.set_ylabel("AUC loss (%)")
ax.set_title("DC vs frequency pruning (Kaggle D=16, consistent accounting)")
ax.grid(True, alpha=0.3, which="both")
ax.invert_yaxis()
plt.tight_layout()
out3 = os.path.join(OUT_DIR, "kaggle_dc_vs_pruning_pareto_v2.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.savefig(out3.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved {out3}")
plt.close()

# ============================================================
# Sanity check: linearity efficiency with corrected accounting
# ============================================================
print("\nRecovery efficiency (% AUC per MB cold storage), with corrected accounting:")
print(f"{'Hot%':>6} {'rpb=1':>10} {'rpb=4':>10} {'rpb=8':>10} {'rpb=16':>10}")
for hf in [0.043, 0.02, 0.01]:
    subset = [r for r in dc_recomputed if r["hot_fraction"] == hf]
    zero_loss = [r for r in subset if r["rpb"] == 0][0]["loss_pct"]
    line = f"{hf*100:>5.1f}%"
    for rpb in [1, 4, 8, 16]:
        rr = [r for r in subset if r["rpb"] == rpb][0]
        recovery = zero_loss - rr["loss_pct"]
        eff = recovery / rr["cold_mb"]
        line += f"  {eff:>8.5f}"
    print(line)
