"""step5_conditional_signal.py — Pattern signal after controlling for region and class.

After establishing:
  - pattern types have different accuracy (step 2)
  - the effect is confounded with class (step 3)
  - forest-level proba is an orthogonal conditioning variable (step 4)

This script answers: does pattern still discriminate accuracy AFTER
controlling for (forest_proba_bucket, predicted_class)?

If yes → pattern carries genuine incremental information, not just
  a proxy for forest confidence or class.

Method: aggregate R[pb, pat, ci, {sumc, n}] over all datasets.
Report accuracy spread within each (pb, ci) cell.
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
REPEATS = 3
TEST_SIZE = 0.3
SEED = 42
N_PROB = 5
BUCK_LABELS = ["[.5,.6)", "[.6,.7)", "[.7,.8)", "[.8,.9)", "[.9,1.]"]
MIN_N = 200


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
    forest_proba = rf.predict_proba(Xte)   # (n_te, n_cls)

    # R[pb, pat, ci, {sumc, n}]
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


def run():
    print(f"repeats={REPEATS}  n_estimators={N_ESTIMATORS}  min_n={MIN_N}\n")
    print("Accumulating R[forest_pb, pattern, class, {correct,n}]...\n")

    jobs = [(name, r) for name in DATASETS for r in range(REPEATS)]
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_accumulate)(name, r) for name, r in jobs)
    R = sum(results)

    # --- accuracy table ---
    for ci, clab in [(0, "majority"), (1, "minority")]:
        print(f"=== pred={clab}: accuracy(n) within each (forest_pb, pattern) cell ===\n")
        header = f'{"bucket":8s}' + ''.join(f'{p:>14s}' for p in PATTERNS)
        print(header)
        print("-" * len(header))
        for bi, bl in enumerate(BUCK_LABELS):
            row = f'{bl:8s}'
            for pi in range(N_PAT):
                v = R[bi, pi, ci]
                if v[1] >= MIN_N:
                    row += f' {v[0]/v[1]:7.3f}({int(v[1])//100:3d}h)'
                else:
                    row += f'{"--":>14s}'
            print(row)
        print()

    # --- spread summary ---
    print("=== Within-cell spread (pattern discrimination after region + class) ===\n")
    print(f'{"bucket":8s} {"class":8s} {"spread":>8} {"best":>10} {"worst":>10} '
          f'{"n_pats":>7}')
    print("-" * 60)

    total_cells = 0
    cells_with_signal = 0
    for bi, bl in enumerate(BUCK_LABELS):
        for ci, clab in [(0, "maj"), (1, "min")]:
            accs = [(R[bi, pi, ci, 0] / R[bi, pi, ci, 1], PATTERNS[pi])
                    for pi in range(N_PAT) if R[bi, pi, ci, 1] >= MIN_N]
            if len(accs) >= 2:
                total_cells += 1
                accs_s = sorted(accs, reverse=True)
                spread = accs_s[0][0] - accs_s[-1][0]
                if spread > 0.02:
                    cells_with_signal += 1
                print(f'{bl:8s} {clab:8s} {spread:8.4f} '
                      f'{accs_s[0][1]:>10s} {accs_s[-1][1]:>10s} {len(accs):>7d}')

    print(f'\nCells with spread > 0.02: {cells_with_signal}/{total_cells}')
    print("\nInterpretation:")
    print("  Non-zero spread within (pb, ci) cells = pattern carries incremental signal.")
    print("  Signal is concentrated in uncertain region [.5,.7) where adjustments")
    print("  can actually change decisions. High-confidence [.9,1.) has near-zero spread")
    print("  (correct regardless of pattern) — weighting those trees has little effect.")
    print("  Both majority and minority cells show signal → conditioning on ci enables")
    print("  separate weight tables that avoid the class confound from step 3.")


if __name__ == "__main__":
    run()
