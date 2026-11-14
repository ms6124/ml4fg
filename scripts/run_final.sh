#!/usr/bin/env bash
# Experiments for the final report, in priority order.
#
#   bash scripts/run_final.sh            # everything, in order
#   bash scripts/run_final.sh 1 2        # only stages 1 and 2
#
# Needs data/processed/{hrca_dev.h5ad, wang_test.h5ad, atac_promoter_cpm.parquet}.
# Times are rough CPU estimates from the interim runs (about 1.5 min per concrete training
# for the major task, 1.7 min for AC, 0.5 min for BC and 0.3 min for RGC). Each command
# writes to results/<task>_<split>_final/ and keeps its log next to it.
#
# Apple GPU: try `export RETPANEL_MPS=1` first on a short run. MPS crashed on macOS 14
# with torch 2.15; on newer macOS it may work and would cut the training time a lot.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
KS="10 25 50 100 200"
STAGES=${*:-"1 2 3 4"}

bench() {  # bench <log name> <args...>
  local log=results/log_final_$1.txt; shift
  echo ">>> $PY scripts/05_benchmark.py $*  (log: $log)"
  "$PY" scripts/05_benchmark.py "$@" --tag final 2>&1 | tee "$log"
}

for s in $STAGES; do case $s in
1)  # Core result: concrete selector at every k, 3 seeds, random vs DE initialisation.
    # RGC ~1 h, BC ~1.5 h, AC ~4 h, major ~4 h. Run overnight; the smaller tasks first.
    bench RGC_donor   --task RGC   --concrete-ks $KS --seeds 0 1 2 --init random de
    bench BC_donor    --task BC    --concrete-ks $KS --seeds 0 1 2 --init random de
    bench major_donor --task major --concrete-ks $KS --seeds 0 1 2 --init random de
    bench AC_donor    --task AC    --concrete-ks $KS --seeds 0 1 2 --init random de
    ;;
2)  # Harder generalisation: train on the Chen lab, test on the Shekhar lab, for subtypes too.
    # One fold each, so cheap (~1.5 h total). The major lab run doubles as the lambda sweep.
    bench RGC_lab   --task RGC   --split lab --concrete-ks 25 50 --seeds 0 1 2 --init random de
    bench BC_lab    --task BC    --split lab --concrete-ks 25 50 --seeds 0 1 2 --init random de
    bench AC_lab    --task AC    --split lab --concrete-ks 25 50 --seeds 0 1 2 --init random de
    bench major_lab --task major --split lab --concrete-ks 25 50 --seeds 0 1 2 --adv 0 0.1 0.3 1 3
    ;;
3)  # Post-hoc analyses on the stage 1-2 outputs (minutes each, ATAC ~10-20 min).
    for r in major_donor BC_donor RGC_donor AC_donor major_lab BC_lab RGC_lab AC_lab; do
      [ -d results/${r}_final ] || continue
      "$PY" scripts/10_paired_bootstrap.py --run ${r}_final
      "$PY" scripts/09_donor_probe.py --run ${r}_final --ks 25 50
    done
    "$PY" scripts/11_consensus_panels.py --run major_donor_final --ks 25 50 \
          --methods de hvg concrete concrete_deinit
    for t in BC RGC AC; do
      "$PY" scripts/11_consensus_panels.py --run ${t}_donor_final --ks 50 --methods de concrete concrete_deinit
    done
    "$PY" scripts/12_atac_panels.py --run major_donor_final --ks 25 50 --methods concrete concrete_deinit hvg
    "$PY" scripts/13_marker_precision.py --run major_donor_final
    ;;
4)  "$PY" scripts/14_final_figures.py
    ;;
*)  echo "unknown stage $s"; exit 1 ;;
esac; done
