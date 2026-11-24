# -*- coding: utf-8 -*-
"""Ablation figure (Figure 2): word-level F1 (bars) + version-hallucination rate (line), Table 3 data.

Run from the repository root:  python src/plot_ablation.py  ->  figures/fig2_ablation.png

The paper version omits the in-figure title; the Chinese paper localizes the axis labels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# ---- font: labels are ASCII, so a plain sans-serif is used first; the CJK fonts are
# kept only as a fallback so a relabelled run on a Chinese-locale machine still renders
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "PingFang SC", "Hiragino Sans GB",
                                  "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---- data (Table 3, 100-case test set)
configs = ["Baseline\nRAG", "Proposer\nonly", "Proposer\n+Reviser", "Full\npipeline"]
f1      = np.array([0.2629, 0.2330, 0.2363, 0.2135])   # word-level F1
halluc  = np.array([0.35,   0.11,   0.03,   0.00])     # version-hallucination rate

x = np.arange(len(configs))

fig, ax = plt.subplots(figsize=(7.2, 4.466), dpi=300)

# ---- left axis: F1 bars
bars = ax.bar(x, f1, width=0.58, color="#4472C4", edgecolor="none",
              zorder=3, label="Word-level F1")
ax.set_ylim(0, 0.32)
ax.set_ylabel("Word-level F1", fontsize=11)
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.grid(axis="y", ls="--", lw=0.7, alpha=0.35)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)

# ---- right axis: version-hallucination rate line
ax2 = ax.twinx()
line, = ax2.plot(x, halluc, color="#C0504D", marker="o", ms=8,
                 mfc="white", mew=1.4, lw=1.6, zorder=4, label="Version-hallucination rate")
ax2.set_ylim(0, 0.40)
ax2.set_ylabel("Version-hallucination rate", fontsize=11)
ax2.yaxis.set_major_locator(MultipleLocator(0.05))
ax2.spines["top"].set_visible(False)

# ---- value labels: F1 inside the bars, hallucination labels in white boxes above markers
for xi, v in zip(x, f1):
    ax.text(xi, v - 0.012, f"{v:.3f}", ha="center", va="top",
            fontsize=9.5, color="white", fontweight="bold", zorder=5)

for xi, v in zip(x, halluc):
    dy = 13 if xi != 0 else 13
    ax2.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, dy), textcoords="offset points",
                 ha="center", va="bottom", fontsize=9.5, color="#B03A37", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#C0504D", lw=0.6, alpha=0.95))

# ---- axes and legend
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10.5)
ax.tick_params(axis="y", labelsize=9.5)
ax2.tick_params(axis="y", labelsize=9.5)
ax.legend([bars, line], ["Word-level F1", "Version-hallucination rate"], loc="upper right",
          fontsize=9.5, frameon=False)

fig.tight_layout()
fig.savefig("figures/fig2_ablation.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved figures/fig2_ablation.png")
