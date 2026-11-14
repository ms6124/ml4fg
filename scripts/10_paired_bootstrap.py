"""Paired donor-bootstrap tests between methods, from the predictions saved in oof.npz.

Comparisons at every saved panel size k (one row per concrete seed):
  * each learned panel vs the DE panel        (does learning beat DE?)
  * concrete_adv*  vs concrete                (does the donor adversary help?)
  * concrete_deinit vs concrete               (does the DE warm start help?)
  * every panel vs the 3,000-gene LR          (cost of compressing to k genes)

HRCA: out-of-fold predictions, resampling HRCA donors.
Wang 2022 (major task only): fold models' probabilities are averaged, then Wang's 4 donors
are resampled. With 4 donors that interval is very coarse, so we also write per-donor
macro auPR (wang_per_donor.csv), which is the more honest view.

Usage: python scripts/10_paired_bootstrap.py --run major_donor_final [--n-boot 1000]
Output: results/<run>/paired_bootstrap.csv, wang_per_donor.csv
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from retpanel.analysis import run_dir
from retpanel.metrics import paired_donor_bootstrap, per_class_auprc

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--n-boot", type=int, default=1000)
args = p.parse_args()

out = run_dir(args.run)
z = np.load(out / "oof.npz", allow_pickle=False)


def parse(prefix):
    """{(method, k, seed): array} for keys like 'oof__concrete__50__s1'."""
    res = {}
    for key in z.files:
        if not key.startswith(prefix + "__"):
            continue
        parts = key[len(prefix) + 2:].split("__")
        seed = int(parts[2][1:]) if len(parts) == 3 else 0
        res[(parts[0], int(parts[1]), seed)] = z[key].astype(np.float32)
    return res


def pairs(keys):
    """Yield (a, b, label) comparisons available among the stored keys."""
    for (m, k, s) in sorted(keys):
        if m == "full_lr":
            continue
        if m != "de" and ("de", k, 0) in keys:
            yield (m, k, s), ("de", k, 0), "vs DE"
        full = [x for x in keys if x[0] == "full_lr"]
        if full:
            yield (m, k, s), full[0], "vs full LR"
        mm = re.match(r"(concrete(?:_deinit)?)_adv[\d.]+$", m)
        if mm and (mm.group(1), k, s) in keys:
            yield (m, k, s), (mm.group(1), k, s), "adversary effect"
        if m == "concrete_deinit" and ("concrete", k, s) in keys:
            yield (m, k, s), ("concrete", k, s), "DE warm start effect"


rows = []
y, donors = z["y"], z["donors"]
oof = parse("oof")
n_classes = len(z["classes"])
for a, b, label in pairs(set(oof)):
    ok = ~np.isnan(oof[a][:, 0]) & ~np.isnan(oof[b][:, 0])
    r = paired_donor_bootstrap(y[ok], oof[a][ok], oof[b][ok], donors[ok], n_classes, args.n_boot)
    rows.append({"eval": "heldout_hrca", "comparison": label, "method_a": a[0], "method_b": b[0],
                 "k": a[1], "seed": a[2], **r})
    print(f"HRCA  {a[0]:>22s} - {b[0]:<16s} k={a[1]:<4d} s{a[2]}  diff={r['diff']:+.4f} "
          f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p={r['p_boot']:.3f}", flush=True)

if "wang_y" in z.files:
    wy, wd, wcls = z["wang_y"], z["wang_donors"], z["wang_classes"]
    wang = {key: P.mean(0) for key, P in parse("wang").items()}  # average the fold models
    for a, b, label in pairs(set(wang)):
        r = paired_donor_bootstrap(wy, wang[a], wang[b], wd, len(wcls), args.n_boot)
        rows.append({"eval": "wang2022", "comparison": label, "method_a": a[0], "method_b": b[0],
                     "k": a[1], "seed": a[2], **r})
    per_donor = []
    for (m, k, s), P in wang.items():
        for dn in np.unique(wd):
            sel = wd == dn
            pc = per_class_auprc(wy[sel], P[sel], list(wcls))
            per_donor.append({"method": m, "k": k, "seed": s, "donor": dn, "n_cells": int(sel.sum()),
                              "macro_auprc": pc.mean(), **{f"auprc[{c}]": v for c, v in pc.items()}})
    pd.DataFrame(per_donor).to_csv(out / "wang_per_donor.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv(out / "paired_bootstrap.csv", index=False)
print(f"wrote {out / 'paired_bootstrap.csv'} ({len(df)} comparisons)")
