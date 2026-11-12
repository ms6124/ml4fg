"""Fetch only the .obs tables of the HRCA BC / AC / RGC subclass h5ad files over HTTP.

These files hold the fine subtype labels (author_cell_type). Their cells are a
subset of the "all cells" snRNA file and share `observation_joinid`, so we join
the labels instead of downloading ~21 GB of expression data we already have.

Usage: python scripts/01_fetch_subtype_obs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anndata as ad
import fsspec
import h5py

from retpanel.config import PROC, URLS

KEEP = ["observation_joinid", "donor_id", "study_name", "author_cell_type", "cell_type"]

for key, cls in [("hrca_bc_subset", "BC"), ("hrca_ac_subset", "AC"), ("hrca_rgc_subset", "RGC")]:
    out = PROC / f"{key}_obs.parquet"
    if out.exists():
        print("exists:", out)
        continue
    with fsspec.open(URLS[key], "rb", block_size=16 * 2**20) as f:
        with h5py.File(f, "r") as h:
            obs = ad.io.read_elem(h["obs"])
    obs = obs[KEEP].copy()
    obs["subclass"] = cls
    obs.index.name = "barcode"
    obs.reset_index().to_parquet(out)
    print(cls, obs.shape, obs.author_cell_type.nunique(), "subtypes ->", out)
