"""Loading, label harmonization and stratified subsampling of the raw h5ad files."""
from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Label harmonization
# ---------------------------------------------------------------------------
# Major classes shared between HRCA and Wang 2022 (Cell Ontology label -> class).
# Anything not listed is kept under its own name and only used inside HRCA.
CELLTYPE_TO_MAJOR = {
    "retinal rod cell": "Rod",
    "retinal cone cell": "Cone",
    "L/M cone": "Cone",
    "S cone cell": "Cone",
    "retinal bipolar neuron": "BC",
    "rod bipolar cell": "BC",
    "ON-bipolar cell": "BC",
    "OFF-bipolar cell": "BC",
    "amacrine cell": "AC",
    "retina horizontal cell": "HC",
    "retinal ganglion cell": "RGC",
    "Mueller cell": "MG",
    "astrocyte": "Astro",
    "microglial cell": "Microglia",
}
SHARED_MAJOR = ["Rod", "Cone", "BC", "AC", "HC", "RGC", "MG", "Astro", "Microglia"]

# Wang 2022 author labels -> intermediate labels comparable to HRCA subclasses.
WANG_CUSTOM_TO_MAJOR = {
    "Rod": "Rod", "Cone": "Cone",
    "ON-cone bipolar": "BC", "OFF-cone bipolar": "BC", "Rod bipolar": "BC",
    "GABA-amacrine": "AC", "Gly-amacrine": "AC", "AII-amacrine": "AC",
    "Horizontal": "HC", "Retinal ganglion cell": "RGC", "Muller glia": "MG",
    "Astrocyte": "Astro", "Microglia": "Microglia",
}


def _major_rule(ct: str) -> str:
    """Collapse fine Cell Ontology labels (as used by HRCA) to a major class.
    Order matters: 'cone bipolar' / 'rod bipolar' are bipolar cells, not photoreceptors."""
    s = ct.lower()
    if s in CELLTYPE_TO_MAJOR_LOWER:
        return CELLTYPE_TO_MAJOR_LOWER[s]
    if "bipolar" in s or s == "offx cell":
        return "BC"
    if "amacrine" in s:
        return "AC"
    if "ganglion" in s:
        return "RGC"
    if "horizontal" in s:
        return "HC"
    if "cone" in s:
        return "Cone"
    if "pigment epithelial" in s:
        return "RPE"
    return ct


CELLTYPE_TO_MAJOR_LOWER = {k.lower(): v for k, v in CELLTYPE_TO_MAJOR.items()}


def map_major(cell_type: pd.Series) -> pd.Series:
    s = cell_type.astype(str)
    uniq = {u: _major_rule(u) for u in s.unique()}
    return s.map(uniq)


# ---------------------------------------------------------------------------
# h5ad helpers
# ---------------------------------------------------------------------------
def read_obs(path) -> pd.DataFrame:
    with h5py.File(path, "r") as h:
        return ad.io.read_elem(h["obs"])


def read_var(path, raw: bool = True) -> pd.DataFrame:
    with h5py.File(path, "r") as h:
        g = h["raw"]["var"] if raw and "raw" in h else h["var"]
        return ad.io.read_elem(g)


def read_csr_rows(path, rows: np.ndarray, group: str = "raw/X", chunk_rows: int = 100_000) -> sp.csr_matrix:
    """Read selected rows of an on-disk CSR matrix by streaming contiguous chunks.

    Much faster than anndata backed fancy-indexing when `rows` is a dense-ish
    random subset of a very large file.
    """
    rows = np.sort(np.asarray(rows))
    with h5py.File(path, "r") as h:
        g = h[group]
        n, ncol = g.attrs["shape"]
        indptr = g["indptr"][:]
        parts = []
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            sel = rows[(rows >= start) & (rows < stop)]
            if sel.size == 0:
                continue
            lo, hi = indptr[start], indptr[stop]
            data = g["data"][lo:hi]
            idx = g["indices"][lo:hi]
            block = sp.csr_matrix((data, idx, indptr[start:stop + 1] - lo), shape=(stop - start, ncol))
            parts.append(block[sel - start])
            print(f"  rows {start:>9,}-{stop:>9,}: kept {sel.size:,}", flush=True)
    return sp.vstack(parts, format="csr")


def lognorm_to_counts(X: sp.csr_matrix) -> sp.csr_matrix:
    """Invert log1p(CP10k): expm1 then divide each row by its smallest nonzero.

    Valid when every cell has at least one gene observed with a single count
    (true for essentially every droplet). Returns rounded integer counts.
    """
    X = X.tocsr(copy=True).astype(np.float64)
    X.data = np.expm1(X.data)
    mins = np.minimum.reduceat(X.data, X.indptr[:-1]) if X.nnz else np.ones(X.shape[0])
    lens = np.diff(X.indptr)
    mins[lens == 0] = 1.0
    X.data /= np.repeat(mins, lens)
    err = np.abs(X.data - np.round(X.data)).max() if X.nnz else 0.0
    X.data = np.round(X.data)
    return X.astype(np.float32), err


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------
def stratified_cap(df: pd.DataFrame, by: list[str], cap: int, seed: int = 0) -> np.ndarray:
    """Return integer positions with at most `cap` cells per group in `by`."""
    rng = np.random.default_rng(seed)
    pos = np.arange(len(df))
    keep = []
    for _, idx in df.groupby(by, observed=True).indices.items():
        idx = pos[idx]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))
