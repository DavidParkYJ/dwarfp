"""step5b_region_best.py — Per-dataset region-best pattern on 17 design datasets.

For each dataset, determines the best pattern in each (forest-probability
region, predicted class) cell.  Shows that the best pattern varies across
datasets, motivating per-dataset weight estimation.

Output: table for paper Section 4.7.
"""

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(PROJECT_ROOT) / "archive"))
from dwarfp.common import (load, classify_pattern, walk_tree,
                            PATTERNS, N_PAT, DATASETS)

warnings.filterwarnings("ignore")

N_ESTIMATORS = 150
REPEATS = 5
TEST_SIZE = 0.3
SEED = 42
MIN_N = 30

DESIGN_DATASETS = DATASETS[:17]

N_PROB = 5
BUCK_LABELS = ["[.5,.6)", "[.6,.7)", "[.7,.8)", "[.8,.9)", "[.9,1.]"]


def _bucket_fp(fp):
    fp = max(0.5, min(0.9999, fp))
    return min(4, int((fp - 0.5) / 0.1))


def _accumulate(name, rep):
    X, y = load(name)
    cls, cnt = np.unique(y, return_counts=True)
    minority = int(cls[np.argmin(cnt)])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
                                bootstrap=True, random_state=SEED + rep,
                                n_jobs=1).fit(Xtr, ytr)
    classes = rf.classes_
    forest_proba = rf.predict_proba(Xte)

    R = np.zeros((N_PROB, N_PAT, 2, 2))
    for est in rf.estimators_:
        for i, (labels, lv) in enumerate(walk_tree(est, Xte)):
            pred = classes[int(np.argmax(lv))]
            c = 1.0 if pred == yte[i] else 0.0
            ci = 1 if int(pred) == minority else 0
            pred_idx = np.searchsorted(classes, pred)
            fp = float(forest_proba[i, pred_idx])
            pb = _bucket_fp(fp)
            pat = classify_pattern(labels)
            R[pb, pat, ci, 0] += c
            R[pb, pat, ci, 1] += 1
    return R


def _best_pattern(R, pb, ci):
    """Return (best_pattern_name, accuracy, n_valid_patterns) or (None, ...)."""
    best_acc = -1.0
    best_p = None
    n_valid = 0
    for p in range(N_PAT):
        if R[pb, p, ci, 1] >= MIN_N:
            n_valid += 1
            acc = R[pb, p, ci, 0] / R[pb, p, ci, 1]
            if acc > best_acc:
                best_acc = acc
                best_p = p
    if best_p is None:
        return None, np.nan, 0
    return PATTERNS[best_p], best_acc, n_valid


def run():
    print(f"datasets={len(DESIGN_DATASETS)}  repeats={REPEATS}  "
          f"n_estimators={N_ESTIMATORS}  min_n={MIN_N}\n")

    # Collect per-dataset R tables
    dataset_R = {}
    for name in DESIGN_DATASETS:
        jobs = [(name, r) for r in range(REPEATS)]
        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_accumulate)(n, r) for n, r in jobs)
        dataset_R[name] = sum(results)
        print(f"  {name} done", flush=True)

    # --- Table 1: best pattern per dataset × region (majority pred) ---
    for ci, clab in [(0, "maj"), (1, "min")]:
        print(f"\n=== Best pattern per dataset × region, pred={clab} ===\n")
        header = f'{"dataset":14s}' + ''.join(f'{bl:>10s}' for bl in BUCK_LABELS)
        print(header)
        print("-" * len(header))
        for name in DESIGN_DATASETS:
            R = dataset_R[name]
            row = f'{name:14s}'
            for pb in range(N_PROB):
                bp, _, _ = _best_pattern(R, pb, ci)
                row += f'{(bp or "--"):>10s}'
            print(row)

    # --- Summary: distinct best counts per region ---
    print("\n=== Distinct best-pattern count per region ===\n")
    print(f'{"region":8s} {"class":5s} {"n_distinct":>10s} {"patterns":30s}')
    print("-" * 60)
    for pb in range(N_PROB):
        for ci, clab in [(0, "maj"), (1, "min")]:
            bests = set()
            for name in DESIGN_DATASETS:
                R = dataset_R[name]
                bp, _, _ = _best_pattern(R, pb, ci)
                if bp:
                    bests.add(bp)
            print(f'{BUCK_LABELS[pb]:8s} {clab:5s} {len(bests):>10d} '
                  f'{", ".join(sorted(bests)):30s}')

    # --- Save CSV ---
    out_path = Path(__file__).resolve().parent / "results_region_best.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "region", "class",
                         "best_pattern", "best_acc", "n_valid_patterns"])
        for name in DESIGN_DATASETS:
            R = dataset_R[name]
            for pb in range(N_PROB):
                for ci, clab in [(0, "maj"), (1, "min")]:
                    bp, ba, nv = _best_pattern(R, pb, ci)
                    writer.writerow([name, BUCK_LABELS[pb], clab,
                                     bp or "", f"{ba:.4f}" if bp else "", nv])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()
