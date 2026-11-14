"""Donor-held-out benchmark of gene panels vs the full-HVG classifier.

For each fold (all fitted quantities use training donors only):
  1. seurat_v3 HVGs (3000) on training counts; z-score parameters on training cells.
  2. full      : L2 multinomial logistic regression on all 3000 HVGs.
  3. panels    : random (3 seeds), hvg (top-k variable), de (Wilcoxon one-vs-rest,
                 round-robin per class), [concrete (Gumbel-softmax selector, optional
                 donor-adversarial GRL head)] -- each evaluated with the SAME downstream
                 classifier (L2 logistic regression on the k panel genes).
  4. Evaluate on held-out donors and (major task) on the external Wang 2022 data.

Tasks: major (HRCA major classes) | BC | RGC | AC (fine subtypes within a subclass)
Splits: donor (StratifiedGroupKFold by donor) | lab (train Baylor/Chen, test Harvard/Shekhar)

Examples
  python scripts/05_benchmark.py --task major
  python scripts/05_benchmark.py --task BC --ks 10 25 50 100
  python scripts/05_benchmark.py --task major --concrete-ks 50 --adv 0 1
  python scripts/05_benchmark.py --task major --split lab
  # final-report sweep: concrete at every k, 3 seeds, plain and DE-initialised, lambda sweep
  python scripts/05_benchmark.py --task RGC --concrete-ks 10 25 50 100 200 --seeds 0 1 2 \
      --init random de --adv 0 1 --tag final

Outputs (results/<task>_<split>[_tag]/): metrics_per_fold.csv, summary.csv, panels.json
(gene symbols per fold/method/k, plus each fold's 3,000-HVG universe), pooled_oof_bootstrap.csv,
per_class_auprc_pooled.csv and oof.npz (out-of-fold + Wang probabilities for --save-oof-ks,
used by 10_paired_bootstrap.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anndata as ad
import numpy as np
import pandas as pd

from retpanel.config import N_FOLDS, N_HVG, PANEL_SIZES, PROC, RESULTS, SEED
from retpanel.data import SHARED_MAJOR
from retpanel.metrics import donor_bootstrap, per_class_auprc, summarize
from retpanel.models import ConcreteConfig, fit_logreg, predict_concrete, predict_proba_full, train_concrete
from retpanel.panels import de_panel, hvg_panel, random_panel, wilcoxon_rankings
from retpanel.preprocess import Scaler, lognorm, select_hvg
from retpanel.splits import check_no_donor_leak, donor_folds

p = argparse.ArgumentParser()
p.add_argument("--task", default="major", choices=["major", "BC", "RGC", "AC"])
p.add_argument("--split", default="donor", choices=["donor", "lab"])
p.add_argument("--ks", type=int, nargs="+", default=PANEL_SIZES)
p.add_argument("--folds", type=int, default=N_FOLDS)
p.add_argument("--max-folds", type=int, default=None, help="run only the first N folds (quick runs)")
p.add_argument("--n-random", type=int, default=3)
p.add_argument("--concrete-ks", type=int, nargs="*", default=[])
p.add_argument("--adv", type=float, nargs="*", default=[0.0], help="donor-adversarial lambdas for concrete")
p.add_argument("--seeds", type=int, nargs="+", default=[SEED], help="concrete training seeds")
p.add_argument("--init", nargs="+", default=["random"], choices=["random", "de"],
               help="concrete selector initialisation(s): random logits or warm start from the DE panel")
p.add_argument("--save-oof-ks", type=int, nargs="*", default=[50],
               help="panel sizes whose out-of-fold probabilities are saved to oof.npz")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--no-full", action="store_true")
p.add_argument("--tag", default="")
args = p.parse_args()

tag = f"{args.task}_{args.split}{('_' + args.tag) if args.tag else ''}"
out_dir = RESULTS / tag
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "per_class_auprc_pooled.csv").unlink(missing_ok=True)  # written in append mode below
t0 = time.time()

# ------------------------------------------------------------------ data
h = ad.read_h5ad(PROC / "hrca_dev.h5ad")
if args.task == "major":
    h = h[h.obs.in_major.values].copy()
    label_col = "major"
else:
    h = h[(h.obs.in_subtype & (h.obs.subclass == args.task)).values].copy()
    label_col = "subtype"
classes = sorted(h.obs[label_col].unique())
cidx = {c: i for i, c in enumerate(classes)}
y = h.obs[label_col].map(cidx).values
donors = h.obs.donor_id.values
X_log = lognorm(h.X)  # per-cell normalisation: no cross-cell information
print(f"[{tag}] {h.n_obs:,} cells, {h.n_vars:,} genes, {len(classes)} classes, {len(set(donors))} donors")

wang = None
if args.task == "major":
    w = ad.read_h5ad(PROC / "wang_test.h5ad")
    w = w[:, h.var_names].copy()  # same gene space (hrca_dev is already restricted to shared genes)
    wang = {"X": lognorm(w.X), "y": w.obs.major.map(cidx).values, "donor": w.obs.donor_id.values}
    shared_idx = [cidx[c] for c in SHARED_MAJOR if c in cidx]

if args.split == "donor":
    fold = donor_folds(h.obs, label_col, args.folds, SEED)
    check_no_donor_leak(h.obs, fold)
    fold_ids = list(range(args.folds))
else:
    inst = h.obs.institute.astype(str).values
    fold = np.where(inst == "Harvard University", 0, -1)
    fold_ids = [0]
    print("lab split test cells:", (fold == 0).sum(), "train cells:", (fold == -1).sum())
if args.max_folds:
    fold_ids = fold_ids[: args.max_folds]

symbol = h.var["feature_name"].astype(str)

# ------------------------------------------------------------------ loop
records, panels_out = [], {}
oof = {}  # method -> (n_cells, n_classes) out-of-fold probabilities
wang_proba = {}  # method key -> {fold: (n_wang, n_shared) probabilities}


def evaluate(method, k, f, proba_te, te, extra=None, proba_w=None):
    rec = {"method": method, "k": k, "fold": f, "split": "heldout_donors", **summarize(y[te], proba_te, classes)}
    if extra:
        rec.update(extra)
    records.append(rec)
    key = f"{method}|{k}" + (f"|s{extra['seed']}" if extra and "seed" in extra else "")
    oof.setdefault(key, np.full((len(y), len(classes)), np.nan, dtype=np.float32))[te] = proba_te
    if proba_w is not None:
        pw = proba_w[:, shared_idx]
        pw = pw / pw.sum(1, keepdims=True).clip(1e-12)
        if k in args.save_oof_ks or method == "full_lr":
            wang_proba.setdefault(key, {})[f] = pw.astype(np.float16)
        yw = np.array([shared_idx.index(v) for v in wang["y"]])
        rw = {"method": method, "k": k, "fold": f, "split": "wang2022",
              **summarize(yw, pw, [classes[i] for i in shared_idx])}
        if extra:
            rw.update(extra)
        records.append(rw)
    print(f"  {method:>14s} k={str(k):>4s}  macro-auPR={rec['macro_auprc']:.4f}  macro-F1={rec['macro_f1']:.4f}"
          + (f"  | Wang auPR={records[-1]['macro_auprc']:.4f}" if proba_w is not None else ""), flush=True)


for f in fold_ids:
    tr = np.where(fold != f)[0]
    te = np.where(fold == f)[0]
    print(f"\n=== fold {f}: train {len(tr):,} cells / {len(set(donors[tr]))} donors; "
          f"test {len(te):,} cells / {len(set(donors[te]))} donors ({time.time() - t0:.0f}s)")
    hv = select_hvg(h[tr], N_HVG, batch_key="donor_id")
    hvg = hv.index[np.isfinite(hv.hvg_rank)][:N_HVG]
    hvg = hv.loc[hvg].sort_values("hvg_rank").index
    gpos = h.var_names.get_indexer(hvg)
    Xtr, Xte = X_log[tr][:, gpos], X_log[te][:, gpos]
    sc_ = Scaler.fit(Xtr)
    Ztr, Zte = sc_.transform(Xtr), sc_.transform(Xte)
    Zw = sc_.transform(wang["X"][:, gpos]) if wang else None
    hvg_list = list(hvg)
    panels_out[f"fold{f}|hvg_universe|{N_HVG}"] = symbol.loc[hvg_list].tolist()

    def run_lr(method, k, cols, extra=None):
        clf = fit_logreg(Ztr[:, cols], y[tr], seed=SEED)
        pte = predict_proba_full(clf, Zte[:, cols], len(classes))
        pw = predict_proba_full(clf, Zw[:, cols], len(classes)) if wang else None
        evaluate(method, k, f, pte, te, extra, pw)

    if not args.no_full:
        run_lr("full_lr", N_HVG, np.arange(len(hvg_list)))

    rank_series = pd.Series(np.arange(len(hvg_list)), index=hvg_list)
    rankings = wilcoxon_rankings(Xtr, hvg_list, y[tr], seed=SEED)
    for k in args.ks:
        sel = {"hvg": hvg_panel(rank_series, k), "de": de_panel(rankings, k)}
        for s in range(args.n_random):
            sel[f"random_s{s}"] = random_panel(hvg_list, k, seed=SEED + 100 * f + s)
        for m, genes in sel.items():
            cols = rank_series.loc[genes].values
            run_lr(m if not m.startswith("random") else "random", k, cols,
                   extra={"seed": int(m.split("_s")[1])} if m.startswith("random") else None)
            if not m.startswith("random") or m == "random_s0":
                panels_out[f"fold{f}|{m}|{k}"] = symbol.loc[genes].tolist()

    multi_seed = len(args.seeds) > 1
    for k in args.concrete_ks:
        de_cols = rank_series.loc[de_panel(rankings, k)].values
        for init in args.init:
            base = "concrete" if init == "random" else "concrete_deinit"
            for lam in args.adv:
                name = base if lam == 0 else f"{base}_adv{lam:g}"
                for s in args.seeds:
                    print(f"  training {name} k={k} seed={s}")
                    cfg = ConcreteConfig(k=k, epochs=args.epochs, adv_lambda=lam, seed=s,
                                         init_cols=de_cols if init == "de" else None)
                    net, hist = train_concrete(Ztr, y[tr], donors[tr], len(classes), cfg)
                    cols = net.selector.panel()
                    genes = [hvg_list[i] for i in cols]
                    sfx = f"|s{s}" if multi_seed else ""
                    panels_out[f"fold{f}|{name}|{k}{sfx}"] = symbol.loc[genes].tolist()
                    hist_name = f"history_fold{f}_{name}_k{k}" + (f"_s{s}" if multi_seed else "")
                    pd.DataFrame(hist).to_csv(out_dir / f"{hist_name}.csv", index=False)
                    extra = {"n_unique_argmax": net.selector.n_unique_argmax()}
                    if multi_seed:
                        extra["seed"] = s
                    # end-to-end network (selector argmax + MLP head)
                    pte = predict_concrete(net, Zte)
                    pw = predict_concrete(net, Zw) if wang else None
                    evaluate(name + "_mlp", k, f, pte, te, dict(extra), pw)
                    # same downstream classifier as every other panel
                    run_lr(name, k, cols, extra=dict(extra))

    pd.DataFrame(records).to_csv(out_dir / "metrics_per_fold.csv", index=False)
    json.dump(panels_out, open(out_dir / "panels.json", "w"), indent=1)

# ------------------------------------------------------------------ pooled OOF + donor bootstrap CI
df = pd.DataFrame(records)
pooled = []
for key, P in oof.items():
    m, k = key.split("|")[:2]
    ok = ~np.isnan(P[:, 0])
    pc = per_class_auprc(y[ok], P[ok], classes)
    # CIs only for deterministic panels; random-panel spread is reported across seeds/folds instead
    lo, hi = (np.nan, np.nan) if m == "random" else donor_bootstrap(y[ok], P[ok], donors[ok], classes, n_boot=200, seed=SEED)
    pooled.append({"method": m, "k": int(k), "seed": key.split("|")[2] if key.count("|") == 2 else "", "pooled_macro_auprc": pc.mean(), "ci_lo": lo, "ci_hi": hi})
    pc.rename(key).to_frame().T.to_csv(out_dir / "per_class_auprc_pooled.csv", mode="a",
                                       header=not (out_dir / "per_class_auprc_pooled.csv").exists())
pooled = pd.DataFrame(pooled)
pooled.to_csv(out_dir / "pooled_oof_bootstrap.csv", index=False)

# ------------------------------------------------------------------ saved predictions for paired tests
save = {"y": y, "donors": donors.astype(str), "classes": np.array(classes), "fold": fold}
for key, P in oof.items():
    m, k = key.split("|")[:2]
    if (int(k) in args.save_oof_ks or m == "full_lr") and m != "random":
        save["oof__" + key.replace("|", "__")] = P.astype(np.float16)
if wang is not None:
    save["wang_y"] = np.array([shared_idx.index(v) for v in wang["y"]])
    save["wang_donors"] = wang["donor"].astype(str)
    save["wang_classes"] = np.array([classes[i] for i in shared_idx])
    for key, per_fold in wang_proba.items():
        if not key.startswith("random"):
            save["wang__" + key.replace("|", "__")] = np.stack([per_fold[f] for f in sorted(per_fold)])
np.savez_compressed(out_dir / "oof.npz", **save)

summary = (df.groupby(["split", "method", "k"])[["macro_auprc", "macro_f1", "balanced_acc"]]
           .agg(["mean", "std"]).round(4))
summary.to_csv(out_dir / "summary.csv")
print("\n", summary.to_string())
print("\npooled OOF (donor-bootstrap 95% CI):\n", pooled.round(4).to_string(index=False))
print(f"done in {time.time() - t0:.0f}s -> {out_dir}")
