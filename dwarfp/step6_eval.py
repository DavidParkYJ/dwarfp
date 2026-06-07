"""step6_eval.py — Path-based Adaptive Weighting: full headline evaluation.

This is the canonical, paper-managed entry point for the headline comparison
(Tables tab:aggregate, tab:ood, tab:results): RF vs the proposed method (CPFW)
vs WRF(Winham) vs KNORA-E vs KNORA-U, all sharing the same forest per split,
30 repeats x 36 datasets, weight table estimated by 5-fold CV on the training
fold only.

Method (per tree vote):
    w = P(correct | forest_pb, pattern, pred_class)
      / P(correct | forest_pb, pred_class)
  E[w | forest_pb, pred_class] = 1  (no systematic class or confidence bias)

Eval contract:
  Primary:   accuracy vs RF (Wilcoxon signed-rank, 36 datasets x 30 repeats)
  Secondary: minority and majority recall must not regress vs RF (>0.2pp)

Single engine: the CPFW core lives in `dwarfp.common` (cpfw_collect_table /
cpfw_build_weight_table / cpfw_predict_proba) and the full runner + canonical
CSV writer live in `dwarfp.compare_baselines`.  This module is the managed
`stepN` entry that drives that engine, so the headline numbers and every
downstream consumer (step8/step9/step11) read one artifact: results_baselines.csv.

Run:
    python -m dwarfp.step6_eval
"""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
from dwarfp.compare_baselines import run


if __name__ == "__main__":
    run()
