"""Build the HRCA development set from the 3.2M-cell snRNA "all cells" h5ad.

Steps
1. Read obs only; harmonize major classes; attach fine BC/AC/RGC subtype labels from
   the subclass files (joined on observation_joinid).
2. Pre-filter on author QC metrics, exclude any donor shared with the Wang test set.
3. Two stratified subsamples (union is loaded once):
     major task   : <= CAP_PER_DONOR_CLASS cells per (donor, major class)
     subtype task : <= CAP_PER_DONOR_SUBTYPE cells per (donor, subtype), subtypes with
                    >= MIN_CELLS_SUBTYPE atlas cells
4. Stream raw counts for the selected rows; QC + Scrublet per donor; restrict genes to
   those shared with Wang.

Output: data/processed/hrca_dev.h5ad
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anndata as ad
import numpy as np
import pandas as pd

from retpanel.config import (CAP_PER_DONOR_CLASS, CAP_PER_DONOR_SUBTYPE, MIN_CELLS_MAJOR, MIN_CELLS_SUBTYPE,
                             PROC, RAW, SEED)
from retpanel.data import map_major, read_csr_rows, read_obs, read_var, stratified_cap
from retpanel.preprocess import qc_filter, run_scrublet

src = RAW / "hrca_snrna_all.h5ad"
obs = read_obs(src)
print("HRCA all-cells obs:", obs.shape)
for c in ["study_name", "institute", "cell_type", "tissue", "is_primary_data"]:
    if c in obs:
        print(f"\n{c}:\n{obs[c].value_counts().head(30).to_string()}")

obs["major"] = map_major(obs["cell_type"])
obs["row"] = np.arange(len(obs))

# --- fine subtype labels from subclass files -------------------------------------------
subs = pd.concat([pd.read_parquet(PROC / f"hrca_{s}_subset_obs.parquet") for s in ("bc", "ac", "rgc")])
subs = subs.drop_duplicates("observation_joinid", keep=False)  # a cell in >1 subclass file is ambiguous
subs["subtype"] = subs["subclass"] + ":" + subs["author_cell_type"].astype(str)
obs = obs.merge(subs[["observation_joinid", "subclass", "subtype"]], on="observation_joinid", how="left")
print("\ncells with a fine subtype label:", obs["subtype"].notna().sum())
# compact copy of full-atlas metadata for EDA (atlas-level composition, not just the subsample)
_cols = ["donor_id", "study_name", "institute", "major", "subtype", "nFeature_RNA", "nCount_RNA", "percent.mt"]
_full = obs[[c for c in _cols if c in obs]].copy()
for c in _full.columns:
    if _full[c].dtype == object or str(_full[c].dtype) == "category":
        _full[c] = _full[c].astype(str)
_full.to_parquet(PROC / "hrca_full_obs.parquet")

# --- donor overlap with Wang 2022 -------------------------------------------------------
wang_donors = {"LGS1", "LGS2", "LGS3", "LVG1"}
overlap = obs["study_name"].astype(str).str.contains("Wang|Sui", case=False) | obs["donor_id"].astype(str).isin(wang_donors)
print(f"cells from Wang-like studies/donors (excluded): {overlap.sum()}")
obs = obs[~overlap.values]

# --- pre-filter on author QC so caps are not wasted on cells we would drop ---------------
qc = (obs["nFeature_RNA"].astype(float) >= 300) & (obs["percent.mt"].astype(float) <= 10)
print(f"author-metric pre-QC: keeping {qc.sum():,}/{len(obs):,}")
obs = obs[qc.values]

# drop major classes too rare to evaluate across donor folds
vc = obs["major"].value_counts()
rare = vc[vc < MIN_CELLS_MAJOR].index.tolist()
print("dropping rare major classes:", rare)
obs = obs[~obs["major"].isin(rare)].reset_index(drop=True)

sel_major = stratified_cap(obs, ["donor_id", "major"], CAP_PER_DONOR_CLASS, seed=SEED)

sub = obs[obs["subtype"].notna()]
svc = sub["subtype"].value_counts()
sub = sub[sub["subtype"].isin(svc[svc >= MIN_CELLS_SUBTYPE].index)]
sel_sub = sub.index.values[stratified_cap(sub, ["donor_id", "subtype"], CAP_PER_DONOR_SUBTYPE, seed=SEED)]

obs["in_major"] = False
obs.loc[sel_major, "in_major"] = True
obs["in_subtype"] = False
obs.loc[sel_sub, "in_subtype"] = True
take = obs[obs["in_major"] | obs["in_subtype"]].copy()
print(f"\nselected: major-task {len(sel_major):,}  subtype-task {len(sel_sub):,}  union {len(take):,}")

# --- stream counts -------------------------------------------------------------------------
X = read_csr_rows(src, take["row"].values, group="raw/X")
take = take.sort_values("row")
var = read_var(src, raw=True)
var = var[["feature_name"] + [c for c in ["feature_biotype"] if c in var]].copy()
var["feature_name"] = var["feature_name"].astype(str)

keep_cols = ["donor_id", "study_name", "institute", "sampleid", "library_id", "tissue", "sex", "donor_age",
             "cell_type", "author_cell_type", "major", "subclass", "subtype", "in_major", "in_subtype",
             "observation_joinid"]
o = take[[c for c in keep_cols if c in take]].copy()
o.index = take["observation_joinid"].astype(str).values
for c in o.columns:
    if o[c].dtype == object or str(o[c].dtype) == "category":
        o[c] = o[c].astype(str)
o["dataset"] = "HRCA"
h = ad.AnnData(X, obs=o, var=var)

# genes shared with Wang (both are CELLxGENE Ensembl IDs)
wang_var = ad.read_h5ad(PROC / "wang_test.h5ad", backed="r").var_names
shared = h.var_names.intersection(wang_var)
print(f"genes: HRCA {h.n_vars}, Wang {len(wang_var)}, shared {len(shared)}")
h = h[:, shared].copy()

h = qc_filter(h)
h = run_scrublet(h, batch_key="donor_id", seed=SEED)
print("Scrublet doublet rate (median over donors):",
      h.obs.groupby("donor_id")["predicted_doublet"].mean().median())
n0 = h.n_obs
h = h[~h.obs["predicted_doublet"].values].copy()
print(f"removed {n0 - h.n_obs} predicted doublets; final {h.n_obs:,} cells")
print(h.obs.loc[h.obs.in_major, "major"].value_counts().to_string())
print("subtypes:", h.obs.loc[h.obs.in_subtype, "subtype"].nunique())

out = PROC / "hrca_dev.h5ad"
h.write_h5ad(out, compression="gzip")
print("wrote", out)
