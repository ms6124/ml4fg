"""Post-hoc donor probe: how much donor identity does each frozen gene panel carry?

The donor-adversarial head's own accuracy is not a fair measure (it is trained *against*
the selector). Instead, for every stored panel we train a fresh classifier to predict the
donor from the panel genes, separately within each cell class, using only the fold's
held-out donors (the panel never saw them). Cells of each donor are split 50/50 into
probe-train and probe-test halves.

Reported per (panel, class): balanced accuracy and its chance level 1/n_donors, plus a
normalised score (bacc - chance) / (1 - chance): 0 = no donor signal, 1 = donor fully
identifiable. The fold's full 3,000-HVG universe is included as an upper reference.

Usage: python scripts/09_donor_probe.py --run major_donor_final [--ks 50] [--cap 150]
Output: results/<run>/donor_probe.csv, donor_probe_summary.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from retpanel.analysis import load_panels, load_task, parse_run, run_dir, symbol_index, task_folds
from retpanel.data import stratified_cap
from retpanel.preprocess import Scaler, lognorm

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="results folder, e.g. major_donor_final")
p.add_argument("--ks", type=int, nargs="*", default=None, help="panel sizes to probe (default: all)")
p.add_argument("--methods", nargs="*", default=None, help="default: every stored method")
p.add_argument("--cap", type=int, default=150, help="max cells per (donor, class) in the probe")
p.add_argument("--min-cells", type=int, default=20, help="donors need this many cells of a class")
p.add_argument("--min-donors", type=int, default=3)
p.add_argument("--no-universe", action="store_true", help="skip the 3,000-HVG reference (slow)")
args = p.parse_args()

task, split = parse_run(args.run)
out = run_dir(args.run)
h, label_col, classes, y = load_task(task)
fold = task_folds(h, label_col, split)
donors = h.obs.donor_id.astype(str).values
X_log = lognorm(h.X)
col_of = symbol_index(h.var)

panels = load_panels(args.run)
if args.ks:
    panels = panels[panels.k.isin(args.ks) | (panels.method == "hvg_universe")]
if args.methods:
    panels = panels[panels.method.isin(args.methods) | (panels.method == "hvg_universe")]
if args.no_universe:
    panels = panels[panels.method != "hvg_universe"]
panels = panels[panels.method.isin(["hvg_universe"]) | ~panels.method.str.endswith("_mlp")]
rng = np.random.default_rng(0)


def probe(cols: np.ndarray, cells: np.ndarray, cls: int):
    """Balanced accuracy of predicting donor from `cols` among cells of one class."""
    d = donors[cells]
    vc = pd.Series(d).value_counts()
    ok = vc[vc >= args.min_cells].index
    if len(ok) < args.min_donors:
        return None
    cells = cells[np.isin(d, ok)]
    d = donors[cells]
    tr_mask = np.zeros(len(cells), bool)
    for dn in ok:  # half of each donor's cells train the probe
        idx = np.where(d == dn)[0]
        tr_mask[rng.choice(idx, len(idx) // 2, replace=False)] = True
    X = X_log[cells][:, cols]
    sc = Scaler.fit(X[tr_mask])
    Ztr, Zte = sc.transform(X[tr_mask]), sc.transform(X[~tr_mask])
    clf = LogisticRegression(C=1.0, max_iter=500, tol=1e-3).fit(Ztr, d[tr_mask])
    bacc = balanced_accuracy_score(d[~tr_mask], clf.predict(Zte))
    chance = 1 / len(ok)
    return {"class": classes[cls], "n_donors": len(ok), "n_cells": len(cells),
            "bacc": bacc, "chance": chance, "norm": (bacc - chance) / (1 - chance)}


rows = []
for f in sorted(panels.fold.unique()):
    te = np.where(fold == f)[0]
    te = te[stratified_cap(h.obs.iloc[te], ["donor_id", label_col], args.cap, seed=0)]
    fp = panels[panels.fold == f]
    print(f"fold {f}: {len(te):,} held-out cells, {len(fp)} panels", flush=True)
    for _, r in fp.iterrows():
        cols = col_of.reindex(r.genes).dropna().astype(int).values
        for c in range(len(classes)):
            res = probe(cols, te[y[te] == c], c)
            if res:
                rows.append({"fold": f, "method": r.method, "k": r.k, "seed": r.seed,
                             "n_genes_found": len(cols), **res})
    pd.DataFrame(rows).to_csv(out / "donor_probe.csv", index=False)

df = pd.DataFrame(rows)
# average over classes first (each class counts once), then over folds/seeds
per_panel = df.groupby(["method", "k", "fold", "seed"])[["bacc", "chance", "norm"]].mean().reset_index()
summary = per_panel.groupby(["method", "k"])[["bacc", "chance", "norm"]].agg(["mean", "std"]).round(4)
summary.to_csv(out / "donor_probe_summary.csv")
print(summary.to_string())
