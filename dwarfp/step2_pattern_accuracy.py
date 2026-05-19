"""step2_pattern_accuracy.py — Raw pattern × accuracy signal.

Claim: pattern type predicts individual tree accuracy.

Method: aggregate (pattern, correct?) across all trees × test points × datasets.
No conditioning yet — this is the raw signal before any controls.

Expected: noflip and recover have higher accuracy; late_sw and oscillat lower.
This motivates using pattern as a reliability signal.
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
REPEATS = 5
TEST_SIZE = 0.3
SEED = 42


def _accumulate(name, rep):
    X, y = load(name)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
                                bootstrap=True, random_state=SEED + rep,
                                n_jobs=1).fit(Xtr, ytr)
    classes = rf.classes_
    # R[pat, {sumc, n}]
    R = np.zeros((N_PAT, 2))
    for est in rf.estimators_:
        for (labels, lv), yi in zip(walk_tree(est, Xte), yte):
            pred = classes[int(np.argmax(lv))]
            c = 1.0 if pred == yi else 0.0
            R[classify_pattern(labels), 0] += c
            R[classify_pattern(labels), 1] += 1
    return R


def run():
    print(f"repeats={REPEATS}  n_estimators={N_ESTIMATORS}\n")
    print("Accumulating (pattern, correct) over all datasets × repeats × trees...")

    jobs = [(name, r) for name in DATASETS for r in range(REPEATS)]
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_accumulate)(name, r) for name, r in jobs)

    R_global = sum(results)

    print("\n=== Raw pattern accuracy (pooled across 30 datasets) ===\n")
    print(f'{"pattern":10s}  {"acc":>7}  {"n":>12}  {"rel_to_mean":>12}')
    print("-" * 50)

    overall = R_global[:, 0].sum() / R_global[:, 1].sum()
    for pi, pat in enumerate(PATTERNS):
        v = R_global[pi]
        if v[1] > 0:
            acc = v[0] / v[1]
            print(f'{pat:10s}  {acc:7.4f}  {int(v[1]):12,d}  {acc - overall:+12.4f}')

    print(f'\n{"overall":10s}  {overall:7.4f}')

    print("\n=== Per-dataset pattern accuracy spread ===")
    print("(max acc - min acc across patterns, per dataset)")
    print()

    jobs2 = [(name, r) for name in DATASETS for r in range(REPEATS)]
    results2 = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_accumulate)(name, r) for name, r in jobs2)

    # re-aggregate per dataset
    ds_R = {}
    for (name, _), R in zip(jobs2, results2):
        ds_R[name] = ds_R.get(name, np.zeros((N_PAT, 2))) + R

    print(f'{"dataset":14s}  {"spread":>8}  {"best":>10}  {"worst":>10}')
    print("-" * 50)
    spreads = []
    for name in DATASETS:
        R = ds_R[name]
        accs = [(R[pi, 0] / R[pi, 1], PATTERNS[pi])
                for pi in range(N_PAT) if R[pi, 1] >= 500]
        if len(accs) >= 2:
            best = max(accs)
            worst = min(accs)
            spread = best[0] - worst[0]
            spreads.append(spread)
            print(f'{name:14s}  {spread:8.4f}  {best[1]:>10s}  {worst[1]:>10s}')
    print(f'\nmean spread: {np.mean(spreads):.4f}')
    print("\nNote: spread here is uncontrolled — class confound examined in step3.")


if __name__ == "__main__":
    run()
