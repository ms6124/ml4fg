"""Consensus panels: the k genes picked most often across folds (and seeds) for a method.

Learned panels vary from fold to fold. A single deliverable panel is the k genes with the
highest selection frequency. Because the consensus pools every fold's training donors, it
cannot be scored by HRCA cross-validation without leakage. For the major task we score it
on Wang 2022, which was never used for selection: an L2 LR is fit on HRCA cells and applied
to Wang, exactly as in the benchmark.

Usage: python scripts/11_consensus_panels.py --run major_donor_final --ks 25 50 \
           --methods de concrete concrete_deinit hvg
Output: results/<run>/consensus_panels.csv  (method, k, rank, gene, frequency)
        results/<run>/consensus_wang.csv    (major task only)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anndata as ad
import numpy as np
import pandas as pd

from retpanel.analysis import load_panels, load_task, parse_run, run_dir, symbol_index
from retpanel.config import PROC
from retpanel.data import SHARED_MAJOR, stratified_cap
from retpanel.metrics import summarize
from retpanel.models import fit_logreg, predict_proba_full
from retpanel.preprocess import Scaler, lognorm

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--ks", type=int, nargs="+", default=[50])
p.add_argument("--methods", nargs="+", default=["de", "hvg", "concrete", "concrete_deinit"])
p.add_argument("--train-cap", type=int, default=150, help="HRCA cells per (donor, class) for the final LR")
args = p.parse_args()

task, _ = parse_run(args.run)
out = run_dir(args.run)
panels = load_panels(args.run)

cons = []
for m in args.methods:
    for k in args.ks:
        sub = panels[(panels.method == m) & (panels.k == k)]
        if sub.empty:
            print(f"skip {m} k={k}: not in panels.json")
            continue
        n = len(sub)
        counts = pd.Series([g for gs in sub.genes for g in gs]).value_counts()
        # ties broken by first appearance, so the order is deterministic
        first = {}
        for gs in sub.genes:
            for i, g in enumerate(gs):
                first.setdefault(g, len(first))
        top = sorted(counts.index, key=lambda g: (-counts[g], first[g]))[:k]
        for r, g in enumerate(top):
            cons.append({"method": m, "k": k, "rank": r + 1, "gene": g,
                         "frequency": counts[g] / n, "n_panels": n})
cons = pd.DataFrame(cons)
cons.to_csv(out / "consensus_panels.csv", index=False)
print(cons.groupby(["method", "k"]).frequency.describe().round(2).to_string())

if task == "major":
    h, label_col, classes, y = load_task(task)
    keep = stratified_cap(h.obs, ["donor_id", label_col], args.train_cap, seed=0)
    h, y = h[keep].copy(), y[keep]
    w = ad.read_h5ad(PROC / "wang_test.h5ad")[:, h.var_names].copy()
    cidx = {c: i for i, c in enumerate(classes)}
    shared = [cidx[c] for c in SHARED_MAJOR if c in cidx]
    yw = np.array([shared.index(cidx[c]) for c in w.obs.major])
    Xh, Xw = lognorm(h.X), lognorm(w.X)
    col_of = symbol_index(h.var)
    rows = []
    for (m, k), g in cons.groupby(["method", "k"]):
        cols = col_of.reindex(g.sort_values("rank").gene).dropna().astype(int).values
        sc = Scaler.fit(Xh[:, cols])
        clf = fit_logreg(sc.transform(Xh[:, cols]), y)
        pw = predict_proba_full(clf, sc.transform(Xw[:, cols]), len(classes))[:, shared]
        pw = pw / pw.sum(1, keepdims=True).clip(1e-12)
        r = summarize(yw, pw, [classes[i] for i in shared])
        rows.append({"method": m, "k": k, **r})
        print(f"consensus {m:>16s} k={k:<4d} Wang macro-auPR={r['macro_auprc']:.4f}")
    pd.DataFrame(rows).to_csv(out / "consensus_wang.csv", index=False)
