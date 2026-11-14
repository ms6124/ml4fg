"""Chromatin-accessibility support of gene panels (Wang 2022 pseudobulk ATAC), major task.

For each panel gene we take its *target class* = the HRCA class with the highest mean RNA
expression, and ask whether its promoter is specifically open in that class:
  spec = log2(CPM+1) in the target class - mean log2(CPM+1) over the other 8 classes
  hit  = the most accessible class is the target class
ATAC cell types are averaged within major class (ON/OFF/rod BC -> BC, three AC types -> AC).

Course feedback: nearly any marker passes this check, so beating random genes is a weak
claim. We therefore report three things per panel (fold, seed):
  1. observed mean spec / hit rate
  2. an expression-matched random null (each gene swapped for a random gene in the same
     decile of maximum class expression, drawn from that fold's 3,000-HVG universe)
  3. learned panel vs the same fold's DE panel of the same size: difference in mean spec,
     with a label-permutation p-value (genes of both panels pooled and reshuffled)

Usage: python scripts/12_atac_panels.py --run major_donor_final --ks 25 50 \
           --methods concrete concrete_deinit hvg [--region promoter|genebody]
Inputs: data/processed/atac_promoter_cpm.parquet (scripts/07_atac_promoters.py)
Output: results/<run>/atac_gene_scores.csv, atac_panel_scores.csv, atac_vs_de.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from retpanel.analysis import load_panels, load_task, parse_run, run_dir, symbol_to_ensembl
from retpanel.config import PROC
from retpanel.data import stratified_cap
from retpanel.preprocess import lognorm

p = argparse.ArgumentParser()
p.add_argument("--run", default="major_donor_final")
p.add_argument("--ks", type=int, nargs="+", default=[50])
p.add_argument("--methods", nargs="+", default=["concrete", "concrete_deinit", "hvg"])
p.add_argument("--region", default="promoter", choices=["promoter", "genebody"])
p.add_argument("--n-perm", type=int, default=2000)
args = p.parse_args()

task, _ = parse_run(args.run)
assert task == "major", "ATAC cell types map onto the major classes only"
out = run_dir(args.run)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- ATAC: genes x major classes
atac = pd.read_parquet(PROC / f"atac_{args.region}_cpm.parquet").set_index("gene_id")
ct_cols = [c for c in atac.columns if "|" in c]
major_of = {c: c.split("|")[1] for c in ct_cols}
A = np.log2(atac[ct_cols] + 1).T.groupby(major_of).mean().T  # genes x majors

# ---------------------------------------------------------------- RNA: mean expression per class
h, label_col, classes, y = load_task("major")
keep = stratified_cap(h.obs, ["donor_id", label_col], 100, seed=0)
X = lognorm(h.X[keep])
onehot = sp.csr_matrix((np.ones(len(keep)), (y[keep], np.arange(len(keep)))), shape=(len(classes), len(keep)))
R = pd.DataFrame((onehot @ X).toarray() / np.bincount(y[keep])[:, None], index=classes,
                 columns=h.var_names.astype(str)).T  # genes (ensembl) x classes
ens = symbol_to_ensembl(h.var)

genes_ok = R.index.intersection(A.index)
R, A = R.loc[genes_ok, classes], A.loc[genes_ok, classes]
target = R.values.argmax(1)
Av = A.values
spec_all = pd.Series(Av[np.arange(len(Av)), target]
                     - (Av.sum(1) - Av[np.arange(len(Av)), target]) / (len(classes) - 1), index=genes_ok)
hit_all = pd.Series(Av.argmax(1) == target, index=genes_ok)
maxexpr = pd.Series(R.values.max(1), index=genes_ok)
target_name = pd.Series(np.array(classes)[target], index=genes_ok)


def to_ens(symbols):
    e = ens.reindex(symbols).dropna()
    return [g for g in e.values if g in spec_all.index]


panels = load_panels(args.run)
universe = {r.fold: to_ens(r.genes) for _, r in panels[panels.method == "hvg_universe"].iterrows()}

gene_rows, panel_rows, de_rows = [], [], []
for f in sorted(panels.fold.unique()):
    uni = pd.Index(universe.get(f) or list(spec_all.index))
    bins = pd.qcut(maxexpr.loc[uni], 10, labels=False, duplicates="drop")
    by_bin = {b: uni[bins.values == b].values for b in np.unique(bins)}
    for k in args.ks:
        fp = panels[(panels.fold == f) & (panels.k == k)]
        de = fp[fp.method == "de"]
        de_genes = to_ens(de.iloc[0].genes) if len(de) else None
        for _, r in fp[fp.method.isin(args.methods + ["de"])].iterrows():
            g = to_ens(r.genes)
            obs_spec, obs_hit = spec_all.loc[g].mean(), hit_all.loc[g].mean()
            for gi in g:
                gene_rows.append({"fold": f, "method": r.method, "k": k, "seed": r.seed, "gene_id": gi,
                                  "target": target_name[gi], "spec": spec_all[gi], "hit": hit_all[gi]})
            # expression-matched null: same expression-decile composition, random genes
            gb = bins.reindex(g).dropna().astype(int)
            null_spec, null_hit = [], []
            for _ in range(args.n_perm):
                draw = np.concatenate([rng.choice(by_bin[b], n, replace=False)
                                       for b, n in gb.value_counts().items()])
                null_spec.append(spec_all.loc[draw].mean())
                null_hit.append(hit_all.loc[draw].mean())
            null_spec = np.asarray(null_spec)
            panel_rows.append({"fold": f, "method": r.method, "k": k, "seed": r.seed, "n_genes": len(g),
                               "spec": obs_spec, "hit_rate": obs_hit,
                               "null_spec_mean": null_spec.mean(), "null_spec_sd": null_spec.std(),
                               "null_hit_mean": np.mean(null_hit),
                               "p_vs_null": (1 + (null_spec >= obs_spec).sum()) / (1 + args.n_perm)})
            # learned vs DE panel of the same size and fold
            if de_genes and r.method != "de":
                a, b = spec_all.loc[g].values, spec_all.loc[de_genes].values
                pooled = np.concatenate([a, b])
                obs = a.mean() - b.mean()
                perm = np.empty(args.n_perm)
                for i in range(args.n_perm):
                    s = rng.permutation(pooled)
                    perm[i] = s[:len(a)].mean() - s[len(a):].mean()
                de_rows.append({"fold": f, "method": r.method, "k": k, "seed": r.seed,
                                "spec_method": a.mean(), "spec_de": b.mean(), "diff": obs,
                                "hit_method": hit_all.loc[g].mean(), "hit_de": hit_all.loc[de_genes].mean(),
                                "n_shared_genes": len(set(g) & set(de_genes)),
                                "p_perm": (1 + (np.abs(perm) >= abs(obs)).sum()) / (1 + args.n_perm)})
    print(f"fold {f} done", flush=True)

pd.DataFrame(gene_rows).to_csv(out / "atac_gene_scores.csv", index=False)
ps = pd.DataFrame(panel_rows)
ps.to_csv(out / "atac_panel_scores.csv", index=False)
pd.DataFrame(de_rows).to_csv(out / "atac_vs_de.csv", index=False)
print(ps.groupby(["method", "k"])[["spec", "null_spec_mean", "hit_rate", "null_hit_mean", "p_vs_null"]]
      .mean().round(3).to_string())
if de_rows:
    print(pd.DataFrame(de_rows).groupby(["method", "k"])[["diff", "p_perm"]].mean().round(3).to_string())
