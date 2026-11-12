"""Cell-type pseudobulk chromatin accessibility at gene promoters (Wang et al. 2022 ATAC).

Input : data/raw/atac/GSM5866073_<celltype>_frags.tsv.gz  (author pseudobulk fragment files)
        data/raw/atac/gencode.v32.basic.annotation.gtf.gz  (GENCODE 32 = Ensembl 98, as HRCA)
Output: data/processed/atac_promoter_cpm.parquet   genes x cell types, Tn5 insertions in TSS +/- 2 kb (CPM)
        data/processed/atac_genebody_cpm.parquet   genes x cell types, gene body + 2 kb upstream (CPM)

Each fragment contributes two Tn5 insertion sites (its ends). Counts are normalised by
each cell type's total insertions (CPM), so cell types with more nuclei are comparable.

Usage: python scripts/07_atac_promoters.py
"""
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from retpanel.config import PROC, RAW

ATAC = RAW / "atac"
FLANK = 2000
CELLTYPES = {  # file tag -> (fine label, major class shared with the RNA task)
    "Rod": ("Rod", "Rod"), "Cone": ("Cone", "Cone"),
    "ONconebipolar": ("ON-cone bipolar", "BC"), "OFFconebipolar": ("OFF-cone bipolar", "BC"),
    "Rodbipolar": ("Rod bipolar", "BC"),
    "GABAamacrine": ("GABA-amacrine", "AC"), "Glyamacrine": ("Gly-amacrine", "AC"),
    "AIIamacrine": ("AII-amacrine", "AC"),
    "Horizontal": ("Horizontal", "HC"), "Retinalganglioncell": ("RGC", "RGC"),
    "Mullerglia": ("Muller glia", "MG"), "Astrocyte": ("Astrocyte", "Astro"), "Microglia": ("Microglia", "Microglia"),
}


def load_genes(gtf: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 8)
            if p[2] != "gene":
                continue
            attrs = dict(kv.strip().split(" ", 1) for kv in p[8].strip().rstrip(";").split(";") if kv.strip())
            gid = attrs["gene_id"].strip('"').split(".")[0]
            rows.append((gid, attrs["gene_name"].strip('"'), p[0], int(p[3]), int(p[4]), p[6]))
    g = pd.DataFrame(rows, columns=["gene_id", "gene_name", "chrom", "start", "end", "strand"])
    g["tss"] = np.where(g.strand == "+", g.start, g.end)
    g["p_lo"], g["p_hi"] = g.tss - FLANK, g.tss + FLANK
    g["b_lo"] = np.where(g.strand == "+", g.start - FLANK, g.start)
    g["b_hi"] = np.where(g.strand == "+", g.end, g.end + FLANK)
    return g.drop_duplicates("gene_id").reset_index(drop=True)


def count_file(path: Path, genes: pd.DataFrame, chunksize: int = 5_000_000):
    by_chr = {c: d.index.values for c, d in genes.groupby("chrom")}
    prom = np.zeros(len(genes))
    body = np.zeros(len(genes))
    total = 0
    reader = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "s", "e"],
                         dtype={"chrom": str, "s": np.int64, "e": np.int64}, chunksize=chunksize, comment="#")
    for i, ch in enumerate(reader):
        total += 2 * len(ch)
        for c, d in ch.groupby("chrom", sort=False):
            if c not in by_chr:
                continue
            pos = np.sort(np.concatenate([d.s.values, d.e.values - 1]))
            gi = by_chr[c]
            for lo_col, hi_col, acc in (("p_lo", "p_hi", prom), ("b_lo", "b_hi", body)):
                lo = genes[lo_col].values[gi]
                hi = genes[hi_col].values[gi]
                acc[gi] += np.searchsorted(pos, hi, "right") - np.searchsorted(pos, lo, "left")
        print(f"    chunk {i}: {total / 2:,.0f} fragments", flush=True)
    return prom, body, total


genes = load_genes(ATAC / "gencode.v32.basic.annotation.gtf.gz")
print("genes:", len(genes))
P, B, tot = {}, {}, {}
for tag, (fine, major) in CELLTYPES.items():
    f = ATAC / f"GSM5866073_{tag}_frags.tsv.gz"
    if not f.exists():
        print("missing", f)
        continue
    print(tag, flush=True)
    p, b, t = count_file(f, genes)
    P[fine], B[fine], tot[fine] = p / t * 1e6, b / t * 1e6, t

idx = pd.MultiIndex.from_frame(genes[["gene_id", "gene_name"]])
cols = pd.MultiIndex.from_tuples([(f, dict(CELLTYPES.values())[f]) for f in P], names=["celltype", "major"])
for name, M in (("promoter", P), ("genebody", B)):
    df = pd.DataFrame(np.column_stack([M[c] for c in P]), index=idx, columns=cols)
    df.columns = [f"{a}|{b}" for a, b in df.columns]
    df.reset_index().to_parquet(PROC / f"atac_{name}_cpm.parquet")
pd.Series(tot, name="insertions").to_csv(PROC / "atac_total_insertions.csv")
print("wrote", PROC / "atac_promoter_cpm.parquet")
