"""Non-learned gene-panel baselines, all choosing from the same training-set HVG universe.

* random_panel : k genes drawn uniformly from the HVGs.
* hvg_panel    : the k most variable genes (seurat_v3 rank on training donors).
* de_panel     : Wilcoxon one-vs-rest (scanpy) per class, merged round-robin so each
                 class contributes its top up-regulated markers until k distinct genes.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def random_panel(genes: list[str], k: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return list(rng.choice(genes, k, replace=False))


def hvg_panel(hvg_rank: pd.Series, k: int) -> list[str]:
    return list(hvg_rank.sort_values().index[:k])


def wilcoxon_rankings(X_log, genes: list[str], labels: np.ndarray, max_per_class: int = 3000,
                      seed: int = 0) -> dict[str, list[str]]:
    """Per-class gene ranking from scanpy Wilcoxon one-vs-rest on (training) log-normalized data.

    Cells are capped per class to keep the test tractable; only genes with positive
    log fold-change are kept (markers, not anti-markers).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(str)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    a = ad.AnnData(X_log[keep], obs=pd.DataFrame({"y": pd.Categorical(labels[keep])}),
                   var=pd.DataFrame(index=genes))
    a.uns["log1p"] = {"base": None}
    sc.tl.rank_genes_groups(a, "y", method="wilcoxon", use_raw=False, tie_correct=False)
    res = a.uns["rank_genes_groups"]
    out = {}
    for c in a.obs.y.cat.categories:
        names = np.asarray(res["names"][c])
        lfc = np.asarray(res["logfoldchanges"][c])
        out[c] = list(names[lfc > 0])
    return out


def de_panel(rankings: dict[str, list[str]], k: int) -> list[str]:
    """Round-robin merge of per-class rankings into k distinct genes."""
    panel, seen = [], set()
    depth = 0
    classes = sorted(rankings)
    max_depth = max(len(v) for v in rankings.values())
    while len(panel) < k and depth < max_depth:
        for c in classes:
            if depth < len(rankings[c]):
                g = rankings[c][depth]
                if g not in seen:
                    seen.add(g)
                    panel.append(g)
                    if len(panel) == k:
                        break
        depth += 1
    return panel
