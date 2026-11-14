"""Evaluation metrics: macro one-vs-rest auPR, per-class auPR, macro-F1, donor bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, balanced_accuracy_score


def per_class_auprc(y: np.ndarray, proba: np.ndarray, classes: list[str]) -> pd.Series:
    out = {}
    for j, c in enumerate(classes):
        pos = y == j
        if pos.sum() == 0:
            continue  # class absent from this test set: auPR undefined
        out[c] = average_precision_score(pos, proba[:, j])
    return pd.Series(out)


def summarize(y: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict:
    pc = per_class_auprc(y, proba, classes)
    pred = proba.argmax(1)
    present = np.unique(y)
    return {
        "macro_auprc": pc.mean(),
        "macro_f1": f1_score(y, pred, labels=present, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "n_classes_eval": len(pc),
        **{f"auprc[{k}]": v for k, v in pc.items()},
    }


def _weighted_ap(pos_sorted: np.ndarray, w_sorted: np.ndarray) -> float:
    """Average precision with per-cell weights, given cells pre-sorted by decreasing score."""
    wp = w_sorted * pos_sorted
    tp = np.cumsum(wp)
    fp = np.cumsum(w_sorted - wp)
    denom = wp.sum()
    if denom == 0:
        return np.nan
    prec = tp / np.maximum(tp + fp, 1e-12)
    return float((prec * wp).sum() / denom)


def paired_donor_bootstrap(y, proba_a, proba_b, donors, n_classes: int, n_boot: int = 1000,
                           seed: int = 0) -> dict:
    """Macro-auPR difference (a - b) with a *paired* donor bootstrap.

    Each replicate resamples donors once and scores both methods on the same weights, so
    the interval reflects the difference itself rather than two overlapping marginal CIs.
    Returns the point estimate, 95% CI, and a two-sided bootstrap p-value for diff = 0.
    """
    rng = np.random.default_rng(seed)
    uniq, d_idx = np.unique(np.asarray(donors), return_inverse=True)
    present = [j for j in range(n_classes) if (y == j).any()]
    prep = []
    for P in (proba_a, proba_b):
        orders = {j: np.argsort(-P[:, j], kind="stable") for j in present}
        prep.append((orders, {j: (y[orders[j]] == j).astype(np.float64) for j in present}))

    def macro(w):
        return [np.nanmean([_weighted_ap(pos[j], w[orders[j]]) for j in present]) for orders, pos in prep]

    a0, b0 = macro(np.ones(len(y)))
    diffs = []
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(np.float64)
        a, b = macro(mult[d_idx])
        diffs.append(a - b)
    diffs = np.asarray(diffs)
    # floor at 1/n_boot: with B replicates a p-value below 1/B cannot be resolved
    p = max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / n_boot)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"auprc_a": a0, "auprc_b": b0, "diff": a0 - b0, "ci_lo": lo, "ci_hi": hi,
            "p_boot": min(p, 1.0), "n_donors": len(uniq), "n_boot": n_boot}


def donor_bootstrap(y, proba, donors, classes, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """95% CI of macro auPR from resampling *donors* (not cells) with replacement.

    Implemented as donor-multiplicity weights on cells, so each class is sorted once
    (fast even for ~80 classes x ~200k cells). Ties in scores are broken by sort order.
    """
    rng = np.random.default_rng(seed)
    uniq, d_idx = np.unique(np.asarray(donors), return_inverse=True)
    present = [j for j in range(len(classes)) if (y == j).any()]
    orders = {j: np.argsort(-proba[:, j], kind="stable") for j in present}
    pos = {j: (y[orders[j]] == j).astype(np.float64) for j in present}
    vals = []
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(np.float64)
        w = mult[d_idx]
        aps = [_weighted_ap(pos[j], w[orders[j]]) for j in present]
        vals.append(np.nanmean(aps))
    return tuple(np.percentile(vals, [2.5, 97.5]))
