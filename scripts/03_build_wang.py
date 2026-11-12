"""Build the external Wang et al. 2022 multiome-RNA test set.

CELLxGENE stores this dataset as log1p(CP10k) only, so integer counts are recovered
exactly (see retpanel.data.lognorm_to_counts) to share the HRCA preprocessing.

Output: data/processed/wang_test.h5ad (raw counts in X, harmonized labels in obs)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anndata as ad
import numpy as np
import pandas as pd

from retpanel.config import PROC, RAW, SEED
from retpanel.data import WANG_CUSTOM_TO_MAJOR, lognorm_to_counts
from retpanel.preprocess import qc_filter, run_scrublet

src = RAW / "wang2022_multiome_rna.h5ad"
a = ad.read_h5ad(src)
print(a)

counts, err = lognorm_to_counts(a.X)
print(f"recovered counts; max |x - round(x)| before rounding = {err:.2e}")
a.X = counts

obs = pd.DataFrame(index=a.obs_names)
obs["donor_id"] = a.obs["donor_id"].astype(str).values
obs["sample_id"] = a.obs["biosample_id"].astype(str).values
obs["study_name"] = "Wang2022"
obs["dataset"] = "Wang2022"
obs["cell_type"] = a.obs["cell_type"].astype(str).values
obs["author_cell_type"] = a.obs["cell_type__custom"].astype(str).values
obs["major"] = obs["author_cell_type"].map(WANG_CUSTOM_TO_MAJOR)
# coarse BC / AC subclasses usable for a limited external check of the subtype task
obs["wang_subclass"] = obs["author_cell_type"]
obs["sex"] = a.obs["sex"].astype(str).values
obs["age"] = a.obs["age"].values
assert obs["major"].notna().all()

var = a.var[["feature_name", "feature_biotype"]].copy()
var["feature_name"] = var["feature_name"].astype(str)
w = ad.AnnData(a.X, obs=obs, var=var)
w.obsm["X_umap_author"] = a.obsm["X_UMAP"]

w = qc_filter(w)
w = run_scrublet(w, batch_key="sample_id", seed=SEED)
print("Scrublet predicted doublet rate per donor:\n", w.obs.groupby("donor_id")["predicted_doublet"].mean())
n0 = w.n_obs
w = w[~w.obs["predicted_doublet"].values].copy()
print(f"removed {n0 - w.n_obs} predicted doublets; final {w.n_obs:,} cells")
print(w.obs["major"].value_counts())

out = PROC / "wang_test.h5ad"
w.write_h5ad(out, compression="gzip")
print("wrote", out)
