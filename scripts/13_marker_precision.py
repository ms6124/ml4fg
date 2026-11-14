"""Overlap of gene panels with known retinal major-class markers (major task).

For each stored panel:
  precision@k   = fraction of panel genes that are known markers
  attainable    = known markers that are in that fold's 3,000-HVG universe (a panel can only
                  pick from there), and recall = markers picked / attainable markers
  classes_hit   = number of the 9 classes with at least one of their markers in the panel

Marker list: resources/retina_markers.csv (canonical markers, one class each).
NOTE: verify this list against the marker tables in Li et al. 2026 (HRCA) before the final.

Usage: python scripts/13_marker_precision.py --run major_donor_final
Output: results/<run>/marker_precision.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from retpanel.analysis import load_panels, parse_run, run_dir
from retpanel.config import ROOT

p = argparse.ArgumentParser()
p.add_argument("--run", default="major_donor_final")
p.add_argument("--markers", default=str(ROOT / "resources" / "retina_markers.csv"))
args = p.parse_args()

assert parse_run(args.run)[0] == "major", "the marker list covers the major classes only"
markers = pd.read_csv(args.markers)
cls_of = dict(zip(markers.gene, markers["class"]))
panels = load_panels(args.run)
universe = {r.fold: set(r.genes) for _, r in panels[panels.method == "hvg_universe"].iterrows()}

rows = []
for _, r in panels[panels.method != "hvg_universe"].iterrows():
    hits = [g for g in r.genes if g in cls_of]
    uni = universe.get(r.fold, set(cls_of))
    attainable = [g for g in cls_of if g in uni]
    rows.append({"fold": r.fold, "method": r.method, "k": r.k, "seed": r.seed,
                 "n_markers": len(hits), "precision": len(hits) / r.k,
                 "attainable": len(attainable),
                 "recall": len(hits) / max(len(attainable), 1),
                 "classes_hit": len({cls_of[g] for g in hits}),
                 "markers": ";".join(f"{g}({cls_of[g]})" for g in hits)})
df = pd.DataFrame(rows)
df.to_csv(run_dir(args.run) / "marker_precision.csv", index=False)
print(df.groupby(["method", "k"])[["precision", "recall", "classes_hit", "attainable"]]
      .mean().round(3).to_string())
