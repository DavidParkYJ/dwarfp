# dwarfp — Path-based Adaptive Weighting for Random Forests

Reproduction code for the paper *"Decision-Path Patterns as Tree Reliability
Signals: Path-based Adaptive Weighting for Random Forest Classification."*
https://arxiv.org/abs/2605.20716

Near the decision boundary, the sequence of majority-class labels along a
tree's root-to-leaf path (its **flip pattern**) carries reliability signal not
captured by the forest-level vote. This repo contains the method and the full
experimental pipeline used to evaluate it.

> This repository contains only the code and datasets needed to **verify the
> paper's empirical results**. The paper text itself is not included.

**Naming.** The method is called *Path-based Adaptive Weighting* in the paper.
In the code it appears as `CPFW` (Conditional Path-Flip Weighting), the
working-title acronym used during development. The two names refer to the same
method: everywhere the paper writes "the proposed method" or "Path-based
Adaptive Weighting," the corresponding code identifier is `CPFW`. The code
identifier was kept (for now) to avoid the churn and diff-noise of renaming
functions and the CSV columns referenced from already-saved result files.

`CPFW` is the single, consistent reference token for the method across the
reproduction pipeline — symbols (`cpfw_*`, `CPFW_N_PROB`/`CPFW_N_CLS`), the
`CPFW_*` / `d_CPFW_*` columns of `results_baselines.csv`, and the `"CPFW"`
display label. A future rename to the final paper name is therefore a single
case-insensitive replace of `cpfw`/`CPFW`, plus regenerating `results_baselines.csv`
so its column headers follow (the cell values are unchanged). The exploratory
`exp_*.py` scripts still use the older `FW` / `Proposed` aliases for the method
and would be folded into that same rename.

## Layout

```
dwarfp/                      method + experiment code
  common.py                  dataset loading, pattern classification, tree walk,
                             CPFW core (cpfw_collect_table/_build/_predict)
  step1_flip_patterns.py     pattern distribution           (tab:pattern-dist)
  step2_pattern_accuracy.py  pooled pattern accuracy         (tab:raw-acc)
  step3_class_confound.py    class confound diagnostic       (tab:confound)
  step3b_naive_ablation.py   naive vs class-conditional      (tab:naive-ablation)
  step5_conditional_signal.py within-cell signal + region    (tab:cond-signal, tab:region-pat)
  step5b_region_best.py      best pattern per region         (tab:region-best, -full)
  step6_eval.py              canonical headline entry: RF vs CPFW vs WRF vs
                             KNORA-E/U   (tab:aggregate, tab:ood, tab:results)
  compare_baselines.py       engine behind step6_eval (same numbers); writes
                             the canonical results_baselines.csv
  step6b_size_effect.py      accuracy gain by dataset size   (tab:size, tab:size-full)
  step7_tree_sweep.py        tree-count robustness           (tab:trees)
  step8_boundary_mass.py     boundary mass M (OOB)           (tab:mass-spread-full, col M)
  step9_boundary_spread.py   boundary spread S (OOB)         (tab:mass-spread-full, col S)
  step10_weight_amplification.py CV-honest K* amplification  (tab:amp-sweep-full)
  step11_mass_spread.py      M*S product, quintiles, Pearson r, amplification-by-quintile
                             (tab:mass-spread, tab:mass-spread-full; the
                             bottom-quintile K*=0 share = 212/240 = 88%)
  fig_synthetic_2d.py        synthetic 2D visualisation      (Figure 1)
  eval_magnitude.py          effect-size helper
  download_datasets.py       (re)builds data_cache/ from UCI/OpenML
data_cache/                  preprocessed datasets (.pkl): the 36 used in the
                             paper plus 7 extra bundled but excluded from the
                             published evaluation (see dwarfp/common.py DATASETS)
requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12. The `deslib==0.3.7` + `scikit-learn>=1.6`
compatibility shim is applied inside the scripts; no manual patch needed.

## Reproduce

Run from the repository root (scripts resolve `data_cache/` relative to it).
Datasets are bundled, so no download is required.

```bash
# Main results — aggregate / OOD / full per-dataset tables
#   (36 datasets x 30 repeats x 5 methods, ~12 min / 20 cores)
#   Writes results_baselines.csv, consumed by step8/step9/step11 below.
python -m dwarfp.step6_eval
#   equivalent: python -c "from dwarfp import compare_baselines as cb; cb.run()"

# Diagnostics — pattern distribution / accuracy / confound / conditional signal
python -m dwarfp.step1_flip_patterns         # tab:pattern-dist
python -m dwarfp.step2_pattern_accuracy      # tab:raw-acc
python -m dwarfp.step3_class_confound        # tab:confound
python -m dwarfp.step3b_naive_ablation       # tab:naive-ablation
python -m dwarfp.step5_conditional_signal    # tab:cond-signal, tab:region-pat
python -m dwarfp.step5b_region_best          # tab:region-best, tab:region-best-full

# Robustness — tree count and dataset size
python -m dwarfp.step7_tree_sweep            # tab:trees
python -m dwarfp.step6b_size_effect          # tab:size, tab:size-full

# Applicability indicator (M*S) and weight amplification.
#   step8/step9 read results_baselines.csv; step11 reads step8/step9 output
#   (and step10's when present), so run them in this order.
python -m dwarfp.step8_boundary_mass         # boundary mass M  -> results_fp_share.csv
python -m dwarfp.step9_boundary_spread       # boundary spread S -> results_boundary_spread.csv
python -m dwarfp.step10_weight_amplification # tab:amp-sweep-full -> results_weight_amplification.csv
python -m dwarfp.step11_mass_spread          # tab:mass-spread, tab:mass-spread-full, Pearson r

# Synthetic 2D figure — Figure 1
python -m dwarfp.fig_synthetic_2d
```

Every paper table maps to a `stepN` script; see the layout above for the
table-to-script mapping. `step11` reads the CSVs written by `step6_eval`,
`step8`, and `step9`, so run those first; for the amplification-by-quintile
figures it uses `step10`'s output when present and otherwise falls back to the
tracked `results_alpha_ms_cv.csv`, so it reproduces from a clean clone.

All methods share the same train/test split and the same trained forest within
each repeat (seed = `42 + repeat`); the proposed method's weight table is
estimated by 5-fold CV on the training fold only (test data never used).
