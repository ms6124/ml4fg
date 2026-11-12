"""Plot macro auPR vs panel size for every finished benchmark run in results/<task>_<split>/.

Usage: python scripts/06_plot_benchmark.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from retpanel.config import FIGURES, RESULTS
from retpanel import plotting as P

P.setup()
# fixed method -> slot mapping so colours follow the method across all figures
SLOT = {"de": 0, "hvg": 1, "random": 2, "concrete": 3, "concrete_adv1": 4}
LABEL = {"de": "Wilcoxon DE panel", "hvg": "top-variance (HVG) panel", "random": "random HVG panel",
         "concrete": "concrete selector", "concrete_adv1": "concrete + donor-adversarial"}

for d in sorted(RESULTS.glob("*_*")):
    f = d / "metrics_per_fold.csv"
    if not d.is_dir() or not f.exists():
        continue
    df = pd.read_csv(f)
    splits = [s for s in ["heldout_donors", "wang2022"] if s in set(df.split)]
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 3.6), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        s = df[df.split == sp]
        full = s[s.method == "full_lr"].macro_auprc
        if len(full):
            ax.axhline(full.mean(), color=P.TEXT2, linewidth=1, linestyle="--")
            ax.text(s.k[s.method != "full_lr"].max() if (s.method != "full_lr").any() else 1, full.mean(),
                    " 3,000-gene LR", va="bottom", ha="right", fontsize=7, color=P.TEXT2)
        for m, slot in SLOT.items():
            g = s[s.method == m].groupby("k").macro_auprc.agg(["mean", "std"]).reset_index()
            if g.empty:
                continue
            c = P.SERIES[slot]
            ax.plot(g.k, g["mean"], color=c, marker="osD^v"[slot], markersize=5, label=LABEL[m])
            ax.fill_between(g.k, g["mean"] - g["std"], g["mean"] + g["std"], color=c, alpha=0.12, linewidth=0)
        ax.set_xscale("log")
        ks = sorted(s.k[s.method != "full_lr"].unique())
        ax.set_xticks(ks, [str(k) for k in ks]); ax.minorticks_off()
        ax.set_xlabel("panel size k (genes)")
        ax.set_ylabel("macro one-vs-rest auPR")
        held = "Held-out lab (Shekhar, Harvard)" if d.name.endswith("_lab") else "Held-out HRCA donors"
        ax.set_title({"heldout_donors": held, "wang2022": "External: Wang 2022 (no retraining)"}[sp], loc="left")
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{d.name}: performance vs panel size (mean ± sd over folds)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / f"bench_{d.name}.png")
    plt.close(fig)
    print("wrote", FIGURES / f"bench_{d.name}.png")

# ---- one combined table: mean macro auPR per task / split / method / k
rows = []
for d in sorted(RESULTS.glob("*_*")):
    f = d / "metrics_per_fold.csv"
    if d.is_dir() and f.exists():
        m = pd.read_csv(f)
        g = m.groupby(["split", "method", "k"]).macro_auprc.agg(["mean", "std", "count"]).reset_index()
        g.insert(0, "run", d.name)
        rows.append(g)
allres = pd.concat(rows)
allres.to_csv(RESULTS / "summary_all_runs.csv", index=False)
print("wrote", RESULTS / "summary_all_runs.csv")
