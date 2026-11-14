"""Figures for the final report. Each figure is skipped if its inputs do not exist yet.

  figures/final_bench.png        1 - macro auPR vs panel size (log-log), all methods, 4 tasks
  figures/final_paired.png       concrete - DE difference with paired donor-bootstrap 95% CIs
  figures/final_donor_probe.png  donor identifiability of each panel (post-hoc probe)
  figures/final_lambda.png       donor-adversarial weight sweep
  figures/final_atac.png         promoter-accessibility specificity: learned vs DE vs matched null

Results are read from results/<task>_<split>_final when present, else the interim folders.

Usage: python scripts/14_final_figures.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retpanel import plotting as P
from retpanel.config import FIGURES, RESULTS

P.setup()
TASKS = [("major", "wang2022", "a  Major classes → Wang 2022"),
         ("BC", "heldout_donors", "b  Bipolar subtypes (14)"),
         ("RGC", "heldout_donors", "c  RGC subtypes (10)"),
         ("AC", "heldout_donors", "d  Amacrine subtypes (71)")]
STYLE = {  # method -> (label, colour slot, marker)
    "de": ("Wilcoxon DE", 0, "o"), "hvg": ("top-variance (HVG)", 1, "s"),
    "random": ("random HVGs", 2, "D"), "concrete": ("concrete", 3, "^"),
    "concrete_deinit": ("concrete, DE init", 4, "v"), "concrete_adv1": ("concrete + donor-adv.", 6, "P"),
}


def run_path(task, split="donor"):
    for tag in (f"{task}_{split}_final", f"{task}_{split}"):
        if (RESULTS / tag).exists():
            return RESULTS / tag
    return None


def save(fig, name):
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print("wrote", FIGURES / name)


# ------------------------------------------------------------------ 1. benchmark, error scale
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, (task, split, title) in zip(axes, TASKS):
    d = run_path(task)
    if d is None:
        ax.set_visible(False)
        continue
    df = pd.read_csv(d / "metrics_per_fold.csv")
    s = df[df.split == split].assign(err=lambda x: 1 - x.macro_auprc)
    full = s[s.method == "full_lr"].err.mean()
    ax.axhline(full, color=P.TEXT2, linewidth=1, linestyle="--")
    ax.text(10, full, " 3,000 genes", va="bottom", fontsize=7, color=P.TEXT2)
    for m, (label, slot, mk) in STYLE.items():
        g = s[s.method == m].groupby("k").err.agg(["mean", "std"]).reset_index()
        if g.empty:
            continue
        c = P.SERIES[slot]
        ax.plot(g.k, g["mean"], color=c, marker=mk, markersize=5, label=label,
                linewidth=1.8 if len(g) > 1 else 0, alpha=0.5 if m == "random" else 1)
        ax.fill_between(g.k, (g["mean"] - g["std"]).clip(1e-4), g["mean"] + g["std"],
                        color=c, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10, 25, 50, 100, 200], ["10", "25", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlabel("panel size k")
    ax.set_title(title, loc="left", fontsize=9)
axes[0].set_ylabel("1 − macro auPR (log scale)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.07))
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "final_bench.png")

# ------------------------------------------------------------------ 2. paired differences vs DE
rows = []
for task, split, title in TASKS:
    d = run_path(task)
    if d is None or not (d / "paired_bootstrap.csv").exists():
        continue
    pb = pd.read_csv(d / "paired_bootstrap.csv")
    want = "wang2022" if task == "major" else "heldout_hrca"
    pb = pb[(pb.comparison == "vs DE") & (pb["eval"] == want) & pb.method_a.isin(["concrete", "concrete_deinit"])]
    rows.append(pb.assign(task=title[3:]))
if rows:
    pb = pd.concat(rows)
    groups = list(pb.groupby(["task", "method_a"], sort=False))
    n_rows = pb.groupby(["task", "method_a", "k"]).ngroups
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * n_rows + 0.25 * len(groups) + 1))
    yt, yl, y0 = [], [], 0
    for (task, m), g in groups:
        c = P.SERIES[STYLE[m][1]]
        for k, gk in g.groupby("k"):
            seeds = sorted(gk.seed.unique())
            for j, (_, r) in enumerate(gk.sort_values("seed").iterrows()):
                yy = y0 + (j - (len(seeds) - 1) / 2) * 0.18  # seeds side by side on one row
                ax.errorbar(r["diff"], yy, xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]],
                            fmt=STYLE[m][2], color=c, markersize=4, capsize=2, linewidth=1)
            yt.append(y0)
            yl.append(f"{task}  {STYLE[m][0]}  k={k}")
            y0 += 1
        y0 += 0.6
    ax.axvline(0, color=P.TEXT2, linewidth=1)
    ax.set_yticks(yt, yl, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("macro auPR difference vs DE panel (paired donor bootstrap, 95% CI)")
    save(fig, "final_paired.png")

# ------------------------------------------------------------------ 3. donor probe
probe = []
for task, _, title in TASKS:
    d = run_path(task)
    if d is not None and (d / "donor_probe.csv").exists():
        dp = pd.read_csv(d / "donor_probe.csv")
        per_panel = dp.groupby(["method", "k", "fold", "seed"]).norm.mean().reset_index()
        probe.append(per_panel.assign(task=f"{task} (held-out donors)"))
if probe:
    dp = pd.concat(probe)
    ks = sorted(set(dp[dp.method != "hvg_universe"].k))
    k_show = 50 if 50 in ks else ks[-1]
    dp = dp[(dp.k == k_show) | (dp.method == "hvg_universe")]
    order = ["hvg_universe", "hvg", "de", "concrete", "concrete_deinit"] + sorted(
        m for m in dp.method.unique() if "adv" in m)
    order = [m for m in order if m in dp.method.unique()]
    tasks = list(dp.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.2 * len(tasks), 2.8), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], tasks):
        g = dp[dp.task == t].groupby("method").norm.agg(["mean", "std"]).reindex(order)
        x = np.arange(len(order))
        ax.bar(x, g["mean"], yerr=g["std"], color=P.SERIES[0], width=0.7, capsize=2)
        ax.set_xticks(x, [m.replace("hvg_universe", "all 3k HVGs").replace("concrete", "conc.")
                          for m in order], rotation=45, ha="right", fontsize=7)
        ax.set_title(t, loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel(f"donor identifiability, k = {k_show}\n(0 = chance, 1 = perfect)")
    save(fig, "final_donor_probe.png")

# ------------------------------------------------------------------ 4. lambda sweep
lam_rows = []
for task, _, _ in TASKS:
    for split in ("donor", "lab"):
        d = run_path(task, split)
        if d is None:
            continue
        df = pd.read_csv(d / "metrics_per_fold.csv")
        for m in df.method.unique():
            mm = re.fullmatch(r"concrete(?:_adv([\d.]+))?", m)
            if not mm:
                continue
            for ev in ("heldout_donors", "wang2022"):
                v = df[(df.method == m) & (df.split == ev)].groupby("k").macro_auprc.mean()
                for k, val in v.items():
                    test = {"heldout_donors": "held-out " + ("lab" if split == "lab" else "donors"),
                            "wang2022": "Wang"}[ev]
                    lam_rows.append({"series": f"{task}, {split} split → {test}", "lam": float(mm.group(1) or 0),
                                     "k": k, "auprc": val})
lam = pd.DataFrame(lam_rows)
if not lam.empty:
    lam = lam[lam.groupby(["series", "k"]).lam.transform("nunique") > 2]  # only real sweeps
if not lam.empty:
    k_lam = 50 if 50 in set(lam.k) else lam.k.max()
    lam = lam[lam.k == k_lam]
if not lam.empty:
    fig, ax = plt.subplots(figsize=(4.8, 3))
    for i, (sname, g) in enumerate(lam.groupby("series")):
        g = g.sort_values("lam")
        ax.plot(g.lam, g.auprc, marker="o", color=P.SERIES[i % len(P.SERIES)], label=sname)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("donor-adversarial weight λ")
    ax.set_ylabel(f"macro auPR (k = {k_lam})")
    ax.legend(fontsize=7)
    save(fig, "final_lambda.png")

# ------------------------------------------------------------------ 5. ATAC specificity
d = run_path("major")
if d is not None and (d / "atac_panel_scores.csv").exists():
    ps = pd.read_csv(d / "atac_panel_scores.csv")
    ks = sorted(ps.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.6 * len(ks), 2.8), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], ks):
        g = ps[ps.k == k]
        methods = [m for m in ["de", "concrete", "concrete_deinit", "hvg"] if m in g.method.unique()]
        for i, m in enumerate(methods):
            gm = g[g.method == m]
            ax.scatter(np.full(len(gm), i) + np.linspace(-0.12, 0.12, len(gm)), gm.spec,
                       color=P.SERIES[STYLE[m][1]], s=14, zorder=3)
            ax.hlines(gm.null_spec_mean.mean(), i - 0.3, i + 0.3, color=P.TEXT2, linestyle=":", linewidth=1)
        ax.set_xticks(range(len(methods)), [STYLE[m][0] for m in methods], rotation=30, ha="right", fontsize=7)
        ax.set_title(f"k = {k}", loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("promoter accessibility specificity\n(log2, target class vs others)")
    fig.text(0.99, 0.01, "dotted: expression-matched random genes", ha="right", fontsize=7, color=P.TEXT2)
    save(fig, "final_atac.png")
