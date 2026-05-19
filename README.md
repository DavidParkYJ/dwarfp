# dwarfp — Decision-Path Flip-Pattern Weighting for Random Forests

Reproduction code for the paper *"Decision-Path Patterns as Tree Reliability
Signals: Path-based Adaptive Weighting for Random Forest Classification."*

Near the decision boundary, the sequence of majority-class labels along a
tree's root-to-leaf path (its **flip pattern**) carries reliability signal not
captured by the forest-level vote. This repo contains the method and the full
experimental pipeline used to evaluate it.

> This repository contains only the code and datasets needed to **verify the
> paper's empirical results**. The paper text itself is not included.

## Layout

```
dwarfp/                      method + experiment code
  common.py                  dataset loading, pattern classification, tree walk
  compare_baselines.py       main evaluation: RF vs Proposed vs WRF vs KNORA-E/U
  step1_flip_patterns.py     pattern distribution           (Table 2)
  step2_pattern_accuracy.py  pooled pattern accuracy         (Table 3)
  step3_class_confound.py    class confound diagnostic       (Table 4)
  step5_conditional_signal.py within-cell conditional signal (Table 5)
  step6_eval.py              full evaluation (same as compare_baselines)
  step7_tree_sweep.py        tree-count robustness           (Table 11)
  fig_synthetic_2d.py        synthetic 2D visualisation      (Figure 1)
  eval_magnitude.py          effect-size helper
  download_datasets.py       (re)builds data_cache/ from UCI/OpenML
data_cache/                  37 preprocessed benchmark datasets (.pkl, bundled)
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
# Main results — Tables 7-10  (30 datasets x 30 repeats x 5 methods, ~12 min / 20 cores)
python -c "from dwarfp import compare_baselines as cb; cb.run()"

# Tree-count robustness — Table 11
python -m dwarfp.step7_tree_sweep

# Diagnostics — Tables 2-5
python -m dwarfp.step1_flip_patterns
python -m dwarfp.step2_pattern_accuracy
python -m dwarfp.step3_class_confound
python -m dwarfp.step5_conditional_signal

# Synthetic 2D figure — Figure 1
python -m dwarfp.fig_synthetic_2d
```

All methods share the same train/test split and the same trained forest within
each repeat (seed = `42 + repeat`); the proposed method's weight table is
estimated by 5-fold CV on the training fold only (test data never used).
