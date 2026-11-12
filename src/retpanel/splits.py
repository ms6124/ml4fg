"""Donor-grouped and lab-held-out splits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def donor_folds(obs: pd.DataFrame, label: str, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign each cell a fold id so that every donor lives in exactly one fold.

    Stratifies on the label so rare classes appear in every test fold where possible.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(obs), -1, dtype=int)
    for f, (_, te) in enumerate(sgkf.split(np.zeros(len(obs)), obs[label].values, obs["donor_id"].values)):
        fold[te] = f
    assert (fold >= 0).all()
    return fold


def check_no_donor_leak(obs: pd.DataFrame, fold: np.ndarray) -> None:
    per_donor = pd.Series(fold).groupby(obs["donor_id"].values).nunique()
    assert (per_donor == 1).all(), "a donor appears in more than one fold"
