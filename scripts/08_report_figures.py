"""Compact multi-panel benchmark figure for the interim report.

Usage: python scripts/08_report_figures.py  ->  figures/report_bench.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from retpanel.config import FIGURES, RESULTS
from retpanel import plotting as P

P.setup()
SLOT = {"de": 0, "hvg": 1, "random": 2, "concrete": 3, "concrete_adv1": 4}
LABEL = {"de": "Wilcoxon DE", "hvg": "top-variance (HVG)", "random": "random HVGs",
         "concrete": "concrete selector", "concrete_adv1": "concrete + donor-adv."}
PANELS = [("major_donor", "wang2022", "a  Major classes → Wang 2022 (external)"),
          ("BC_donor", "heldout_donors", "b  Bipolar subtypes (14)"),
          ("RGC_donor", "heldout_donors", "c  RGC subtypes (10)"),
          ("AC_donor", "heldout_donors", "d  Amacrine subtypes (71)")]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
for ax, (run, split, title) in zip(axes, PANELS):
    df = pd.read_csv(RESULTS / run / "metrics_per_fold.csv")
    s = df[df.split == split]
    full = s[s.method == "full_lr"].macro_auprc.mean()
    ax.axhline(full, color=P.TEXT2, linewidth=1, linestyle="--")
    ax.text(200, full, "3,000 genes ", va="bottom", ha="right", fontsize=7, color=P.TEXT2)
    for m, slot in SLOT.items():
        g = s[s.method == m].groupby("k").macro_auprc.agg(["mean", "std"]).reset_index()
        c = P.SERIES[slot]
        ax.plot(g.k, g["mean"], color=c, marker="osD^v"[slot], markersize=5, label=LABEL[m],
                linewidth=1.8 if len(g) > 1 else 0)
        ax.fill_between(g.k, g["mean"] - g["std"], g["mean"] + g["std"], color=c, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 200], ["10", "25", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("panel size k")
    ax.set_title(title, loc="left", fontsize=9)
axes[0].set_ylabel("macro auPR")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIGURES / "report_bench.png")
print("wrote", FIGURES / "report_bench.png")
