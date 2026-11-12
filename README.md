# Compact gene panels for retinal cell-type annotation

ML4FG project (Luana Liao, Sky Sun). We learn small gene panels (k = 10–200) that classify human
retinal cell types across donors and labs, and compare them to a 3,000-gene classifier, to
DE / HVG / random panels of the same size, and later to single-cell foundation models.

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

`tiledbsoma` (CELLxGENE Census) does not load on macOS 14, so we pull h5ad files from the
CELLxGENE Discover API directly.

## Data

| file | source | notes |
|---|---|---|
| `data/raw/hrca_snrna_all.h5ad` (37.7 GB) | HRCA, Li et al. 2026, CELLxGENE collection `4c6eaf5c…` | 3.18 M nuclei, 104 donors; raw counts in `.raw.X` |
| `data/processed/hrca_{bc,ac,rgc}_subset_obs.parquet` | HRCA subclass files (obs only, fetched over HTTP) | fine subtype labels joined on `observation_joinid` |
| `data/raw/wang2022_multiome_rna.h5ad` (0.4 GB) | Wang et al. 2022, CELLxGENE collection `348da6dc…` | 51.6 k nuclei, 4 donors; X is log1p(CP10k) and counts are recovered exactly |

```bash
curl -o data/raw/hrca_snrna_all.h5ad        https://datasets.cellxgene.cziscience.com/25f1558d-c1f8-49f8-addc-b778d8cbd4aa.h5ad
curl -o data/raw/wang2022_multiome_rna.h5ad https://datasets.cellxgene.cziscience.com/74ec3a77-f1ff-4e90-af45-b8937cea0a8f.h5ad
```

## Pipeline

| step | script | output |
|---|---|---|
| 1 | `scripts/01_fetch_subtype_obs.py` | BC / AC / RGC subtype labels |
| 2 | `scripts/03_build_wang.py` | `data/processed/wang_test.h5ad` (count recovery, QC, Scrublet) |
| 3 | `scripts/02_build_hrca.py` | `data/processed/hrca_dev.h5ad`: stratified subsample for both tasks, QC, Scrublet, genes shared with Wang |
| 4 | `scripts/04_eda.py` | `figures/eda_*.png`, `results/eda_*.csv` |
| 5 | `scripts/05_benchmark.py --task {major,BC,RGC,AC} [--split lab] [--concrete-ks 50 --adv 0 1]` | `results/<task>_<split>/` |
| 6 | `scripts/06_plot_benchmark.py` | `figures/bench_*.png` |

Steps 2 and 3 must run in this order because HRCA genes are intersected with the Wang genes.

## Leakage rules

* Splits are by donor (`StratifiedGroupKFold`), and the code asserts that no donor appears in two folds.
* HVGs (seurat_v3), z-score parameters, Wilcoxon DE rankings and every learned panel are fit on training donors only.
* Wang 2022 is never used for training or model selection. HRCA donors from Wang-like studies are excluded.

## Methods implemented

* `full_lr`: L2 multinomial logistic regression on 3,000 HVGs.
* `random`, `hvg`, `de`: panel baselines. `de` uses scanpy's Wilcoxon one-vs-rest and merges per-class top markers round-robin.
* `concrete`: Gumbel-softmax selector (Abid et al. 2019) with an MLP head. Optional donor-adversarial head through gradient reversal (Ganin et al. 2016); the adversary also sees the class label, so it penalises donor signal beyond class composition.
* Every panel is scored with the same downstream classifier (logistic regression on the k genes). The end-to-end concrete MLP is reported separately (`concrete_mlp`).
* Metrics: macro one-vs-rest auPR (primary), per-class auPR, macro-F1, balanced accuracy, and 95 % donor-bootstrap CIs on pooled out-of-fold predictions.

## Interim status (2026-11-08)

All runs below are complete. Figures are in `figures/`, tables in `results/` (`summary_all_runs.csv` combines every run).
Numbers are macro one-vs-rest auPR, mean over 5 donor folds.

| task | full LR (3000) | DE k=10 / 25 / 50 | HVG k=50 | concrete k=50 | concrete + adv k=50 |
|---|---|---|---|---|---|
| major, held-out donors | 1.000 | 0.960 / 0.996 / 0.999 | 0.964 | 0.999 | 0.998 |
| major → Wang 2022 | 0.967 | 0.786 / 0.901 / 0.943 | 0.887 | 0.944 | 0.946 |
| major, held-out lab → Wang | 0.968 | 0.808 / 0.923 / 0.946 | 0.914 | 0.959 | 0.938 |
| BC subtypes (14) | 0.999 | 0.904 / 0.984 / 0.996 | 0.990 | 0.997 | 0.995 |
| RGC subtypes (10) | 0.982 | 0.794 / 0.925 / 0.955 | 0.936 | 0.904 | 0.907 |
| AC subtypes (71) | 0.982 | 0.368 / 0.804 / 0.944 | 0.875 | 0.956 | 0.953 |

Findings so far
* Major classes are easy within HRCA, as expected. The real cost of a small panel shows up on the external lab, where rare glia (microglia, astrocytes) lose the most.
* DE (Wilcoxon) panels are a strong baseline. The concrete selector only beats DE with many classes (AC). It loses on RGC, which has little training data and is 95 % midget cells.
* The donor-adversarial head (λ = 1) does not help yet. The class-conditional adversary only reaches about 5 % donor accuracy (chance is about 1.2 %), so selected genes carry little donor signal.
* Wang multiome nuclei have about 2× fewer UMIs and genes than HRCA, a depth shift on top of the lab effect.

Known issues / fixed along the way
* PyTorch 2.15's MPS backend crashes on macOS 14, so training runs on CPU (set `RETPANEL_MPS=1` to try MPS).
* The concrete selector needed its own learning rate for the selector logits (5e-2), T0 = 2, annealing over 75 % of training, and class-balanced cross-entropy. Without these it collapses to the majority class.
* The donor bootstrap was rewritten as weighted AP (one sort per class). The first major/BC/RGC runs pooled the 3 random-panel seeds under one key. Per-fold numbers are unaffected; re-run those tasks to get per-seed pooled CIs.
* `scripts/07_atac_promoters.py` has produced `data/processed/atac_{promoter,genebody}_cpm.parquet`. Canonical markers pass a sanity check (ARR3 → cone, GRM6 → ON-BC, RLBP1 → MG, PTPRC → microglia).
