"""Helpers shared by the post-hoc analysis scripts (09-14).

They rebuild a task's cells, labels and folds exactly as scripts/05_benchmark.py does, and
read the panels that the benchmark wrote to results/<run>/panels.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .config import N_FOLDS, PROC, RESULTS, SEED
from .splits import check_no_donor_leak, donor_folds


def load_task(task: str):
    """HRCA development cells for one task, filtered and labelled as in 05_benchmark.py."""
    h = ad.read_h5ad(PROC / "hrca_dev.h5ad")
    if task == "major":
        h = h[h.obs.in_major.values].copy()
        label_col = "major"
    else:
        h = h[(h.obs.in_subtype & (h.obs.subclass == task)).values].copy()
        label_col = "subtype"
    classes = sorted(h.obs[label_col].unique())
    y = h.obs[label_col].map({c: i for i, c in enumerate(classes)}).values
    return h, label_col, classes, y


def task_folds(h: ad.AnnData, label_col: str, split: str) -> np.ndarray:
    """Fold id per cell. Donor split: 5 donor-grouped folds. Lab split: Shekhar lab = fold 0,
    Chen lab = -1 (training only)."""
    if split == "donor":
        fold = donor_folds(h.obs, label_col, N_FOLDS, SEED)
        check_no_donor_leak(h.obs, fold)
        return fold
    inst = h.obs.institute.astype(str).values
    return np.where(inst == "Harvard University", 0, -1)


def run_dir(run: str) -> Path:
    d = RESULTS / run
    if not d.exists():
        raise SystemExit(f"{d} not found: run scripts/05_benchmark.py first")
    return d


def parse_run(run: str) -> tuple[str, str]:
    """'major_donor_final' -> ('major', 'donor')."""
    task, split = run.split("_")[:2]
    return task, split


def load_panels(run: str) -> pd.DataFrame:
    """One row per stored panel: fold, method, k, seed, genes (symbols).

    Keys look like 'fold0|de|50' or 'fold0|concrete|50|s1'. The per-fold HVG universe is
    stored under method 'hvg_universe'.
    """
    raw = json.load(open(run_dir(run) / "panels.json"))
    rows = []
    for key, genes in raw.items():
        parts = key.split("|")
        seed = int(parts[3][1:]) if len(parts) == 4 else 0
        method = parts[1]
        if method.startswith("random_s"):
            method, seed = "random", int(method.split("_s")[1])
        rows.append({"fold": int(parts[0][4:]), "method": method, "k": int(parts[2]),
                     "seed": seed, "genes": list(genes)})
    return pd.DataFrame(rows)


def symbol_index(var: pd.DataFrame) -> pd.Series:
    """Gene symbol -> integer column position (first occurrence wins for duplicated symbols)."""
    sym = var["feature_name"].astype(str).reset_index(drop=True)
    s = pd.Series(np.arange(len(sym)), index=sym.values)
    return s[~s.index.duplicated()]


def symbol_to_ensembl(var: pd.DataFrame) -> pd.Series:
    s = pd.Series(var.index.astype(str), index=var["feature_name"].astype(str).values)
    return s[~s.index.duplicated()]
