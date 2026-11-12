"""Exploratory data analysis for the interim report.

Produces (figures/ and results/):
  eda_summary.csv                 cells / genes / donors / studies per dataset
  eda_class_counts.png            major-class composition: full HRCA atlas, dev subsample, Wang
  eda_donors.png                  cells per donor, grouped by study (lab)
  eda_qc.png                      QC metric distributions, HRCA vs Wang
  eda_subtypes.png                subtype counts per subclass (BC / RGC / AC)
  eda_pca_variance.png            PCA scree + per-PC variance explained by class vs donor vs study
  eda_pca_by_class.png            PC1/PC2 small multiples by major class
  eda_umap_by_class.png           UMAP small multiples by major class
  eda_umap_by_dataset.png         UMAP small multiples by study (HRCA labs + Wang) -> batch effects
  eda_umap_bc_subtypes.png        UMAP of bipolar subtypes
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from retpanel.config import FIGURES, PROC, RESULTS, SEED
from retpanel.data import SHARED_MAJOR
from retpanel import plotting as P
from retpanel.preprocess import lognorm

P.setup()
h = ad.read_h5ad(PROC / "hrca_dev.h5ad")
w = ad.read_h5ad(PROC / "wang_test.h5ad")
full = pd.read_parquet(PROC / "hrca_full_obs.parquet")
hm = h[h.obs.in_major.values]

# ---------------------------------------------------------------- summary table
rows = []
for name, o, nv in [("HRCA full atlas (snRNA)", full, None), ("HRCA dev: major task", hm.obs, h.n_vars),
                    ("HRCA dev: subtype task", h.obs[h.obs.in_subtype], h.n_vars), ("Wang 2022 (test)", w.obs, w.n_vars)]:
    rows.append({"dataset": name, "cells": len(o), "genes": nv, "donors": o.donor_id.nunique(),
                 "studies": o.study_name.nunique(),
                 "major_classes": o.major.nunique() if "major" in o else np.nan})
summ = pd.DataFrame(rows)
summ.to_csv(RESULTS / "eda_summary.csv", index=False)
print(summ.to_string(index=False))
print("\nHRCA studies x institute:\n", full.groupby(["study_name", "institute"]).donor_id.nunique())

# ---------------------------------------------------------------- class composition
order = full.major.value_counts().index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
for ax, (t, s) in zip(axes, [("HRCA atlas (all snRNA cells)", full.major), ("HRCA dev subsample (major task)", hm.obs.major),
                             ("Wang 2022 external test", w.obs.major)]):
    vc = s.value_counts().reindex(order).dropna()
    P.hbar(ax, vc.index.tolist(), vc.values)
    ax.set_title(t, loc="left")
    ax.set_xlabel("cells")
fig.suptitle("Major-class composition is dominated by rods; the dev subsample caps each (donor, class)",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "eda_class_counts.png"); plt.close(fig)

# ---------------------------------------------------------------- donors per study
d = full.groupby(["study_name", "donor_id"]).size().rename("cells").reset_index()
studies = d.groupby("study_name").cells.sum().sort_values(ascending=False).index
fig, axes = plt.subplots(1, len(studies), figsize=(2.4 * len(studies), 3), sharey=True)
for ax, st in zip(np.atleast_1d(axes), studies):
    v = np.sort(d[d.study_name == st].cells.values)[::-1]
    ax.bar(np.arange(len(v)), v, color=P.HIGHLIGHT, width=0.8)
    ax.set_title(f"{st}\n{len(v)} donors", loc="left", fontsize=8)
    ax.set_xticks([]); ax.grid(axis="x", visible=False)
np.atleast_1d(axes)[0].set_ylabel("cells per donor")
fig.suptitle("HRCA snRNA: cells per donor, by contributing study", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "eda_donors.png"); plt.close(fig)

# ---------------------------------------------------------------- QC
fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
for ax, col, lab, logx in [(axes[0], "n_genes", "genes detected per cell", True),
                           (axes[1], "total_counts", "UMIs per cell", True),
                           (axes[2], "pct_mt", "% mitochondrial UMIs", False)]:
    for i, (nm, o) in enumerate([("HRCA dev", h.obs), ("Wang 2022", w.obs)]):
        x = o[col].astype(float).values
        bins = np.logspace(np.log10(max(x.min(), 1)), np.log10(x.max()), 60) if logx else np.linspace(0, max(x.max(), 1), 60)
        ax.hist(x, bins=bins, histtype="step", density=True, color=P.SERIES[i], label=nm, linewidth=1.5)
    if logx: ax.set_xscale("log")
    ax.set_xlabel(lab); ax.set_yticks([])
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Per-cell QC after filtering (≥300 genes, ≤10% mito, Scrublet doublets removed)",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "eda_qc.png"); plt.close(fig)

# ---------------------------------------------------------------- subtypes
fs = full[full.subtype.notna() & (full.subtype != "nan") & (full.subtype != "None")]
fig, axes = plt.subplots(1, 3, figsize=(13, 9), gridspec_kw={"width_ratios": [1, 1, 1]})
for ax, sub in zip(axes, ["BC", "RGC", "AC"]):
    vc = fs[fs.subtype.str.startswith(sub + ":")].subtype.str.split(":", n=1).str[1].value_counts()
    if sub == "AC":
        vc = vc.head(40)
    P.hbar(ax, vc.index.tolist(), vc.values, log=True)
    ax.tick_params(axis="y", labelsize=6)
    if vc.min() < 300:
        ax.axvline(300, color=P.TEXT2, linewidth=1, linestyle="--")
        ax.text(300, len(vc) - 0.5, " min. 300 cells\n kept for task", fontsize=6, color=P.TEXT2, va="bottom")
    ax.set_title(f"{sub} subtypes ({fs.subtype.str.startswith(sub + ':').sum():,} cells)" + (" — top 40" if sub == "AC" else ""), loc="left")
    ax.set_xlabel("cells in atlas (log)")
fig.suptitle("Fine subtypes are highly imbalanced (harder task)", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(FIGURES / "eda_subtypes.png"); plt.close(fig)

# ---------------------------------------------------------------- PCA / UMAP on combined major-task cells
rng = np.random.default_rng(SEED)
take_w = rng.choice(w.n_obs, min(20000, w.n_obs), replace=False)
comb = ad.concat([hm, w[np.sort(take_w)]], join="inner", label="src", keys=["HRCA", "Wang2022"], index_unique="-")
comb.layers["counts"] = comb.X.copy()
comb.X = lognorm(comb.X)
sc.pp.highly_variable_genes(comb, n_top_genes=3000, flavor="seurat_v3", layer="counts")
c = comb[:, comb.var.highly_variable].copy()
sc.pp.scale(c, max_value=10)
sc.tl.pca(c, n_comps=50, random_state=SEED)

# variance of each PC explained by class / donor / study (one-way ANOVA R^2)
def r2(pc, g):
    g = pd.Series(g).astype(str).values
    df = pd.DataFrame({"x": pc, "g": g})
    return 1 - df.groupby("g").x.transform(lambda v: v - v.mean()).pow(2).sum() / ((pc - pc.mean()) ** 2).sum()

pcs = c.obsm["X_pca"]
ve = c.uns["pca"]["variance_ratio"]
rec = []
for i in range(20):
    rec.append({"PC": i + 1, "var_ratio": ve[i], "R2_class": r2(pcs[:, i], c.obs.major),
                "R2_donor": r2(pcs[:, i], c.obs.donor_id), "R2_study": r2(pcs[:, i], c.obs.study_name)})
pv = pd.DataFrame(rec)
pv.to_csv(RESULTS / "eda_pca_variance.csv", index=False)
# donor R^2 within class (residual donor effect after removing class means)
resid = pcs[:, :20] - pd.DataFrame(pcs[:, :20]).groupby(c.obs.major.values).transform("mean").values
within = [r2(resid[:, i], c.obs.donor_id) for i in range(20)]
pv["R2_donor_within_class"] = within
pv.to_csv(RESULTS / "eda_pca_variance.csv", index=False)
print(pv.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].bar(pv.PC, pv.var_ratio * 100, color=P.HIGHLIGHT, width=0.75)
axes[0].set_xlabel("principal component"); axes[0].set_ylabel("% variance (of top-3000 HVGs)")
axes[0].set_title("Scree", loc="left"); axes[0].grid(axis="x", visible=False)
for i, (col, lab) in enumerate([("R2_class", "major class"), ("R2_study", "study / lab"),
                                ("R2_donor_within_class", "donor (within class)")]):
    axes[1].plot(pv.PC, pv[col], color=P.SERIES[i], marker="o", markersize=4, label=lab)
axes[1].set_xlabel("principal component"); axes[1].set_ylabel("fraction of PC variance (R²)")
axes[1].set_ylim(0, 1); axes[1].legend(fontsize=8)
for a in axes:
    a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes[1].set_title("What drives each PC", loc="left")
fig.tight_layout(); fig.savefig(FIGURES / "eda_pca_variance.png"); plt.close(fig)

order = [m for m in SHARED_MAJOR if m in set(c.obs.major)] + sorted(set(c.obs.major) - set(SHARED_MAJOR))
P.highlight_grid(pcs[:, :2], c.obs.major.values, order, "PC1 vs PC2 (HRCA dev + Wang), by major class",
                 FIGURES / "eda_pca_by_class.png")

sc.pp.neighbors(c, n_neighbors=15, n_pcs=30, random_state=SEED)
sc.tl.umap(c, random_state=SEED)
emb = c.obsm["X_umap"]
P.highlight_grid(emb, c.obs.major.values, order, "UMAP (HRCA dev + Wang, 30 PCs), by major class",
                 FIGURES / "eda_umap_by_class.png")
st_order = c.obs.study_name.value_counts().index.tolist()
P.highlight_grid(emb, c.obs.study_name.values, st_order, "UMAP by study — uncorrected batch structure",
                 FIGURES / "eda_umap_by_dataset.png", ncols=4)
c.obs[["src", "study_name", "donor_id", "major"]].assign(umap1=emb[:, 0], umap2=emb[:, 1],
                                                       pc1=pcs[:, 0], pc2=pcs[:, 1]).to_parquet(RESULTS / "eda_embedding.parquet")

# ---------------------------------------------------------------- bipolar subtypes UMAP
b = h[(h.obs.in_subtype & (h.obs.subclass == "BC")).values].copy()
b.layers["counts"] = b.X.copy(); b.X = lognorm(b.X)
sc.pp.highly_variable_genes(b, n_top_genes=2000, flavor="seurat_v3", layer="counts")
b = b[:, b.var.highly_variable].copy(); sc.pp.scale(b, max_value=10)
sc.tl.pca(b, n_comps=30, random_state=SEED); sc.pp.neighbors(b, n_pcs=30, random_state=SEED); sc.tl.umap(b, random_state=SEED)
lab = b.obs.subtype.str.split(":", n=1).str[1].values
P.highlight_grid(b.obsm["X_umap"], lab, pd.Series(lab).value_counts().index.tolist(),
                 "UMAP of bipolar subtypes (HRCA subtype-task cells)", FIGURES / "eda_umap_bc_subtypes.png", s=2)
print("EDA done.")
