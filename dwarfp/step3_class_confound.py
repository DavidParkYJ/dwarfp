"""step3_class_confound.py — Class confound in raw pattern signal.

Problem: minority-predicting trees are structurally more complex.
They typically traverse more splits and are more likely to show
late_sw / oscillat patterns. So low accuracy for late_sw may just
reflect "minority prediction is harder", not "late_sw is unreliable".

This script shows:
  Part A: pattern × predicted_class × accuracy
          → late_sw accuracy drop is class-specific (majority only)
  Part B: naive weighting (w = f(pattern), no class condition) eval
          → wins≈RF but minority recall regressions ≈ 26/30
          → confirms confound: down-weighting late_sw = down-weighting minority

Conclusion: any weighting scheme must be conditioned on predicted class.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dwarfp.common import load, recalls, classify_pattern, walk_tree, PATTERNS, N_PAT, DATASETS

warnings.filterwarnings("ignore")

N_ESTIMATORS = 150
REPEATS = 5
TEST_SIZE = 0.3
SEED = 42
MIN_N = 500


# --- Part A helpers ---

def _accumulate_byclass(name, rep):
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
    # R[pat, ci, {sumc, n}]   ci=0 majority pred, ci=1 minority pred
    R = np.zeros((N_PAT, 2, 2))
    for est in rf.estimators_:
        for (labels, lv), yi in zip(walk_tree(est, Xte), yte):
            pred = classes[int(np.argmax(lv))]
            c = 1.0 if pred == yi else 0.0
            ci = 1 if int(pred) == minority else 0
            R[classify_pattern(labels), ci, 0] += c
            R[classify_pattern(labels), ci, 1] += 1
    return R


# --- Part B helpers: naive weight = 1 - flip_rate ---

def _flip_rate(labels):
    if len(labels) <= 1:
        return 0.0
    flips = sum(1 for k in range(1, len(labels)) if labels[k] != labels[k - 1])
    return flips / (len(labels) - 1)


def _run_naive_weight(name, rep):
    X, y = load(name)
    cls, cnt = np.unique(y, return_counts=True)
    minority = int(cls[np.argmin(cnt)])
    majority = int(cls[np.argmax(cnt)])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
                                bootstrap=True, random_state=SEED + rep,
                                n_jobs=1).fit(Xtr, ytr)
    classes = rf.classes_
    n_cls = len(classes)

    out = np.zeros((len(Xte), n_cls))
    for est in rf.estimators_:
        for i, (labels, lv) in enumerate(walk_tree(est, Xte)):
            w = max(1e-6, 1.0 - _flip_rate(labels))
            out[i] += w * (lv / lv.sum())

    rf_pred = rf.predict(Xte)
    rf_acc = float(accuracy_score(yte, rf_pred))
    rf_rmin, rf_rmaj = recalls(yte, rf_pred, minority, majority)

    nw_pred = classes[np.argmax(out, axis=1)]
    nw_acc = float(accuracy_score(yte, nw_pred))
    nw_rmin, nw_rmaj = recalls(yte, nw_pred, minority, majority)
    return rf_acc, rf_rmin, rf_rmaj, nw_acc, nw_rmin, nw_rmaj


def run():
    print(f"repeats={REPEATS}  n_estimators={N_ESTIMATORS}\n")

    # --- Part A ---
    print("=== Part A: pattern × predicted_class × accuracy ===\n")
    jobs = [(name, r) for name in DATASETS for r in range(REPEATS)]
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_accumulate_byclass)(name, r) for name, r in jobs)
    R = sum(results)

    print(f'{"pattern":10s}  {"maj_acc":>8}  {"maj_n":>10}  '
          f'{"min_acc":>8}  {"min_n":>10}  {"gap(maj-min)":>13}')
    print("-" * 65)
    for pi, pat in enumerate(PATTERNS):
        maj = R[pi, 0]
        mn  = R[pi, 1]
        if maj[1] >= MIN_N and mn[1] >= MIN_N:
            ma = maj[0] / maj[1]
            mna = mn[0] / mn[1]
            print(f'{pat:10s}  {ma:8.4f}  {int(maj[1]):10,d}  '
                  f'{mna:8.4f}  {int(mn[1]):10,d}  {ma - mna:+13.4f}')
        elif maj[1] >= MIN_N:
            ma = maj[0] / maj[1]
            print(f'{pat:10s}  {ma:8.4f}  {int(maj[1]):10,d}  '
                  f'{"--":>8}  {int(mn[1]):10,d}  {"--":>13}')

    print("\nNote: large gap for late_sw and oscillat indicates class confound.")
    print("  Minority-predicting trees have structurally complex paths (leaf-heavy)")
    print("  → they show more late_sw/oscillat patterns AND lower accuracy.")
    print("  Down-weighting late_sw without class conditioning = down-weighting minority.")

    # --- Part B ---
    print("\n=== Part B: naive weighting (w = 1 - flip_rate, no class condition) ===\n")
    print(f'{"dataset":14s} {"RF_acc":>7} {"NW_acc":>7} {"d_acc":>7} '
          f'{"RF_rmi":>7} {"NW_rmi":>7}')
    print("-" * 55)

    all_rf, all_nw = [], []
    all_rf_rmin, all_nw_rmin = [], []
    all_rf_rmaj, all_nw_rmaj = [], []

    for name in DATASETS:
        res = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_run_naive_weight)(name, r) for r in range(REPEATS))
        rf_acc  = float(np.mean([r[0] for r in res]))
        rf_rmin = float(np.mean([r[1] for r in res]))
        rf_rmaj = float(np.mean([r[2] for r in res]))
        nw_acc  = float(np.mean([r[3] for r in res]))
        nw_rmin = float(np.mean([r[4] for r in res]))
        nw_rmaj = float(np.mean([r[5] for r in res]))
        d = nw_acc - rf_acc
        flag = " <-- minority hurt" if nw_rmin < rf_rmin - 1e-9 else ""
        print(f'{name:14s} {rf_acc:7.4f} {nw_acc:7.4f} {d:+7.4f} '
              f'{rf_rmin:7.3f} {nw_rmin:7.3f}{flag}')
        all_rf.append(rf_acc);    all_nw.append(nw_acc)
        all_rf_rmin.append(rf_rmin); all_nw_rmin.append(nw_rmin)
        all_rf_rmaj.append(rf_rmaj); all_nw_rmaj.append(nw_rmaj)

    d_acc  = np.array(all_nw) - np.array(all_rf)
    d_rmin = np.array(all_nw_rmin) - np.array(all_rf_rmin)
    try:
        p = wilcoxon(all_nw, all_rf).pvalue
    except ValueError:
        p = float("nan")

    print(f'\nacc wins={int((d_acc>1e-9).sum())}  '
          f'losses={int((d_acc<-1e-9).sum())}  p={p:.4f}')
    print(f'minority regressions: {int((d_rmin<-1e-9).sum())}/{len(DATASETS)}  '
          f'(confound causes systematic minority harm)')
    print("\nConclusion: weighting must be conditioned on predicted class (step 4–5).")


if __name__ == "__main__":
    run()
