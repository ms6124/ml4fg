"""Paths, dataset URLs and global constants shared by all scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _d in (RAW, PROC, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 0

# CELLxGENE Discover h5ad assets (schema 5+; raw counts in .raw.X).
# HRCA collection 4c6eaf5c-6d57-4c76-b1e9-60df8c655f1e (Li et al., Nat Genet 2026)
# Wang 2022 collection 348da6dc-5bf6-435d-adc5-37747b9ae38a (Wang et al., Cell Genomics 2022)
URLS = {
    "hrca_snrna_all": "https://datasets.cellxgene.cziscience.com/25f1558d-c1f8-49f8-addc-b778d8cbd4aa.h5ad",
    "wang2022_multiome_rna": "https://datasets.cellxgene.cziscience.com/74ec3a77-f1ff-4e90-af45-b8937cea0a8f.h5ad",
    # Subclass files: only .obs is fetched (remotely) to get fine subtype labels.
    "hrca_bc_subset": "https://datasets.cellxgene.cziscience.com/6c9f72ef-bc79-4a98-a323-d26252cf5dab.h5ad",
    "hrca_ac_subset": "https://datasets.cellxgene.cziscience.com/7710a02a-55b2-4a81-a3d4-aba6b217386e.h5ad",
    "hrca_rgc_subset": "https://datasets.cellxgene.cziscience.com/e4c0ab7b-35da-4d47-a285-f25db28c1624.h5ad",
}

# Subsampling budget for the HRCA development set.
CAP_PER_DONOR_CLASS = 400   # major-class task: max cells per (donor, class) -> ~300k cells
MIN_CELLS_MAJOR = 2000      # drop major classes rarer than this in the atlas (RPE contamination, 863 cells)
CAP_PER_DONOR_SUBTYPE = 60  # subtype task: max cells per (donor, subtype)
MIN_CELLS_SUBTYPE = 300     # drop subtypes with fewer cells than this in the atlas

N_HVG = 3000
PANEL_SIZES = [10, 25, 50, 100, 200]
N_FOLDS = 5
