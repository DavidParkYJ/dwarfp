"""step1_flip_patterns.py — Define flip patterns and show their distribution.

Motivation: RF aggregates tree leaf votes, discarding within-tree path
structure. Before asking whether this matters, we need to show that
decision paths have interpretable, diverse structure worth examining.

This script:
  (a) Defines the 6 flip pattern types with examples
  (b) Shows pattern frequency distribution across all 30 datasets

No modelling — just path traversal on a fitted RF.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dwarfp.common import load, classify_pattern, walk_tree, PATTERNS, N_PAT, DATASETS

warnings.filterwarnings("ignore")

N_ESTIMATORS = 150
TEST_SIZE = 0.3
SEED = 42


def _distribution(name):
    X, y = load(name)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    (tr, te), = sss.split(X, y)
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
                                bootstrap=True, random_state=SEED,
                                n_jobs=1).fit(X[tr], y[tr])
    counts = np.zeros(N_PAT, dtype=int)
    for est in rf.estimators_:
        for labels, _ in walk_tree(est, X[te]):
            counts[classify_pattern(labels)] += 1
    return counts / counts.sum()


def run():
    print("=== Flip Pattern Definitions ===\n")
    defs = [
        ("noflip",    "majority class never changes root → leaf"),
        ("early_sw",  "first flip in first 1/3 of path, stable after 2/3"),
        ("late_sw",   "first (and only) flip after 2/3 of path"),
        ("oscillat",  "≥2 direction reversals along path"),
        ("recover",   "1 reversal, then stabilises before 2/3 of path"),
        ("other",     "does not fit the above categories"),
    ]
    for name, desc in defs:
        print(f"  {name:10s}  {desc}")

    print("\n=== Pattern Distribution across 30 datasets (% of tree×point pairs) ===\n")
    header = f'{"dataset":14s}' + ''.join(f'{p:>10s}' for p in PATTERNS)
    print(header)
    print("-" * len(header))

    dists = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_distribution)(name) for name in DATASETS)

    for name, dist in zip(DATASETS, dists):
        row = f'{name:14s}' + ''.join(f'{100*v:9.1f}%' for v in dist)
        print(row)

    mean_dist = np.mean(dists, axis=0)
    print("-" * len(header))
    print(f'{"mean":14s}' + ''.join(f'{100*v:9.1f}%' for v in mean_dist))

    print("\n=== Key observations ===")
    print(f"noflip dominates ({100*mean_dist[0]:.0f}% mean) — most tree paths are stable.")
    print(f"late_sw ({100*mean_dist[2]:.0f}%) and early_sw ({100*mean_dist[1]:.0f}%) are the"
          f" next most common — paths that flip near boundary.")
    print("Pattern mix varies across datasets → structure is dataset-specific.")


if __name__ == "__main__":
    run()
