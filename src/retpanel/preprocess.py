"""Leakage-safe preprocessing: everything data-dependent is fit on training donors only."""
from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def qc_filter(adata: ad.AnnData, min_genes: int = 300, max_pct_mt: float = 10.0) -> ad.AnnData:
    """Basic per-cell QC on raw counts; adds n_genes / total_counts / pct_mt to obs."""
    adata.var["mt"] = adata.var["feature_name"].astype(str).str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata.obs["n_genes"] = adata.obs["n_genes_by_counts"]
    adata.obs["pct_mt"] = adata.obs["pct_counts_mt"]
    keep = (adata.obs.n_genes >= min_genes) & (adata.obs.pct_mt <= max_pct_mt)
    print(f"QC: keeping {keep.sum():,}/{adata.n_obs:,} cells (min_genes={min_genes}, pct_mt<={max_pct_mt})")
    return adata[keep.values].copy()


def run_scrublet(adata: ad.AnnData, batch_key: str, seed: int = 0) -> ad.AnnData:
    """Scrublet per batch on raw counts; adds doublet_score / predicted_doublet."""
    sc.pp.scrublet(adata, batch_key=batch_key, random_state=seed, verbose=False)
    return adata


def lognorm(X) -> sp.csr_matrix:
    """log1p(counts per 10k); purely per-cell so it is leakage-free."""
    X = sp.csr_matrix(X, dtype=np.float32, copy=True)
    s = np.asarray(X.sum(1)).ravel()
    s[s == 0] = 1
    X = sp.diags((1e4 / s).astype(np.float32)) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def select_hvg(counts_train: ad.AnnData, n_top: int, batch_key: str | None = "donor_id") -> pd.DataFrame:
    """Seurat-v3 HVGs on training cells' raw counts. Returns var table with `hvg_rank`."""
    a = ad.AnnData(counts_train.X, obs=counts_train.obs[[batch_key]] if batch_key else None,
                   var=pd.DataFrame(index=counts_train.var_names))
    # Batches with too few cells destabilise the loess fit; collapse them.
    if batch_key:
        vc = a.obs[batch_key].value_counts()
        small = vc[vc < 200].index
        a.obs[batch_key] = a.obs[batch_key].astype(str).where(~a.obs[batch_key].isin(small), "_small")
    sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor="seurat_v3", batch_key=batch_key, span=0.5)
    v = a.var.copy()
    rank_col = "highly_variable_rank"
    v["hvg_rank"] = v[rank_col].fillna(np.inf)
    return v


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X) -> "Scaler":
        X = X.toarray() if sp.issparse(X) else X
        m = X.mean(0)
        s = X.std(0)
        s[s < 1e-6] = 1.0
        return cls(m.astype(np.float32), s.astype(np.float32))

    def transform(self, X, clip: float = 10.0) -> np.ndarray:
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        Z = (X - self.mean) / self.std
        return np.clip(Z, -clip, clip).astype(np.float32)
