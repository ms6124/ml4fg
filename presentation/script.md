# Final presentation script (draft)

Slides: https://claude.ai/artifact/VvCgTkzoo9ip27Zmp9jTS1 (the same text is in each slide's speaker notes).
Time limit: 1 + 2N = **5 minutes** for N = 2. About 130 spoken words per minute, so roughly 650 words total.
Bracketed parts wait on the final runs (`bash scripts/run_final.sh`).

| # | slide | speaker | time |
|---|---|---|---|
| 1 | Title | Luana | 0:15 |
| 2 | Spatial methods only measure the genes you pick in advance | Luana | 0:40 |
| 3 | Every method picks k genes, then the same classifier is scored | Luana | 0:45 |
| 4 | A concrete selector learns the panel, and an adversary hides the donor | Luana | 0:45 |
| 5 | In the atlas 25 genes are enough, but a new lab costs accuracy | Sky | 0:45 |
| 6 | Learning the panel helps only when there are many subtypes | Sky | 0:45 |
| 7 | [Donor signal and chromatin support] | Sky | 0:45 |
| 8 | Takeaways | Sky | 0:20 |
| | | | **5:00** |

---

**1. Title (Luana)**
Hi, I'm Luana, and this is Sky. Our project asks how few genes you need to tell retinal cell types
apart, and whether a gene panel learned by a neural network does better than simply picking known
marker genes.

**2. Problem (Luana)**
Spatial transcriptomics methods such as MERFISH and Xenium show where each cell sits in a tissue,
but they only measure the genes you designed probes for, usually a few dozen to a few hundred. So
the gene panel has to be chosen before the experiment, and it has to work on new samples.
We ask how small that panel can be and still identify retinal cell types in donors and labs it has
never seen.
We train on the Human Retina Cell Atlas: 3.2 million nuclei from 104 donors. We test on an
independent dataset from another lab, which also measured chromatin accessibility with ATAC-seq in
the same nuclei.

**3. Approach (Luana)**
Every method follows the same recipe. We start from the 3,000 most variable genes. A selector picks
k of them, from 10 up to 200. Then the same logistic regression classifies the cells using only
those k genes, so any difference between methods comes from the genes, not the classifier.
Our simple baselines pick genes by differential expression, which means the top marker genes of
each cell type, or by variance, or at random.
We test on held-out donors, on a held-out lab, and on the new dataset. We also check whether the
chosen genes have open chromatin in the cell types they mark.

**4. Learned selector (Luana)**
Our learned method is a concrete selector. Each of its k slots holds a soft choice over all 3,000
genes. During training we sharpen these choices until each slot picks exactly one gene. Because the
choice is differentiable, the genes are learned together with a small neural network that predicts
the cell type.
We also add a donor classifier that tries to guess which donor a cell came from, using only the
selected genes. Its gradient is reversed, so the selector is pushed away from genes that give away
the donor. The goal is a panel that transfers to new people.
Now Sky will show what we found.

**5. Result 1 (Sky)**
This figure shows error, one minus macro auPR, on a log scale, so lower is better. Blue is the
marker-gene panel from differential expression, orange picks genes by variance, and green is random.
The dashed line is the model with all 3,000 genes.
Inside the atlas, major classes are nearly solved: 25 marker genes already reach an auPR of 0.996.
On the new lab's data, in panel a, everything gets worse. The 3,000-gene model drops to 0.967, and a
50-gene marker panel to 0.943. Most of that loss is in rare glia, such as microglia.
Picking genes by variance alone almost misses cones on the new data, so choosing marker genes per
cell type matters.

**6. Result 2 (Sky)**
The subtypes are where the methods differ. Each number is the learned panel's score minus the marker
panel's score, at 50 genes.
*(click)* For bipolar subtypes, they tie.
*(click)* For amacrine cells, which have 71 subtypes, the learned panel wins. Fifty genes is fewer than
one marker per subtype, and the learned panel finds genes that separate several subtypes at once.
*(click)* For ganglion cells, it clearly loses. That task has the least training data, and one
subtype, the midget cells, makes up 95 percent of the cells.
[After the final runs: say whether starting the selector from the marker panel closes this gap.]

**7. Result 3 (Sky)** [fill in after the final runs]
Next we asked whether the panels encode which donor a cell came from. With the donor adversary
turned on, accuracy did not change. When we train a separate donor classifier on each frozen panel,
[result].
Finally, the chromatin check. Learned panel genes have [more / similar / less] open chromatin in their
target cell type than marker-panel genes of the same size [p = ___].

**8. Takeaways (Sky)**
To sum up. First, about 25 marker genes are enough to identify the major retinal cell classes in new
donors, but a new lab is a harder test. Second, learning the panel beats marker genes when there are
many subtypes, but not when training data is scarce. Third, [donor and chromatin takeaway].
Thanks for watching.

---

## Recording notes

- Introduce yourselves (slide 1). Define every acronym out loud the first time (auPR, DE, ATAC-seq).
- Point at the panel you are talking about on slide 5, or the instructor's tip: reveal elements as
  you talk (slide 6 already reveals one card per click).
- Rehearse with a timer. If you run long, cut the variance sentence on slide 5 first, then the
  "goal is a panel that transfers" sentence on slide 4.
- Zoom is fine for recording. Faces optional.
