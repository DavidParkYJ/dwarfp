"""step3b_naive_ablation.py — Naive weighting ablation (30 repeats).

Compares RF vs naive weighting (w = 1 - flip_rate, no class conditioning)
using the same 30-repeat protocol as step6_eval.py.

Purpose: produce the results table for Section 5.4 (Why class conditioning).
"""

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(PROJECT_ROOT) / "archive"))
from dwarfp.common import load, recalls, walk_tree, DATASETS

warnings.filterwarnings("ignore")

N_ESTIMATORS = 150
REPEATS = 30
TEST_SIZE = 0.3
SEED = 42


def _flip_rate(labels):
    if len(labels) <= 1:
        return 0.0
    flips = sum(1 for k in range(1, len(labels)) if labels[k] != labels[k - 1])
    return flips / (len(labels) - 1)


def _run_one(name, rep):
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

    # Naive weighted prediction
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

    out_path = Path(__file__).resolve().parent / "results_naive_ablation.csv"

    print(f'{"dataset":16s} {"n":>5} {"RF_acc":>7} {"NW_acc":>7} {"d_acc":>7} '
          f'{"RF_rmi":>7} {"NW_rmi":>7} {"RF_rma":>7} {"NW_rma":>7}')
    print("-" * 85)

    rows = []
    all_rf_acc, all_nw_acc = [], []
    all_rf_rmin, all_nw_rmin = [], []
    all_rf_rmaj, all_nw_rmaj = [], []

    for name in DATASETS:
        res = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_run_one)(name, r) for r in range(REPEATS))
        X, _ = load(name)
        n = len(X)

        rf_acc  = float(np.mean([r[0] for r in res]))
        rf_rmin = float(np.mean([r[1] for r in res]))
        rf_rmaj = float(np.mean([r[2] for r in res]))
        nw_acc  = float(np.mean([r[3] for r in res]))
        nw_rmin = float(np.mean([r[4] for r in res]))
        nw_rmaj = float(np.mean([r[5] for r in res]))
        d = nw_acc - rf_acc

        print(f'{name:16s} {n:5d} {rf_acc:7.4f} {nw_acc:7.4f} {d:+7.4f} '
              f'{rf_rmin:7.3f} {nw_rmin:7.3f} {rf_rmaj:7.3f} {nw_rmaj:7.3f}')

        rows.append({
            "dataset": name, "n": n,
            "rf_acc": rf_acc, "nw_acc": nw_acc,
            "rf_rmin": rf_rmin, "nw_rmin": nw_rmin,
            "rf_rmaj": rf_rmaj, "nw_rmaj": nw_rmaj,
        })
        all_rf_acc.append(rf_acc);    all_nw_acc.append(nw_acc)
        all_rf_rmin.append(rf_rmin);  all_nw_rmin.append(nw_rmin)
        all_rf_rmaj.append(rf_rmaj);  all_nw_rmaj.append(nw_rmaj)

    # Save CSV
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_path}")

    # Summary
    d_acc  = np.array(all_nw_acc)  - np.array(all_rf_acc)
    d_rmin = np.array(all_nw_rmin) - np.array(all_rf_rmin)
    d_rmaj = np.array(all_nw_rmaj) - np.array(all_rf_rmaj)

    wins   = int((d_acc >  1e-9).sum())
    losses = int((d_acc < -1e-9).sum())
    ties   = len(DATASETS) - wins - losses
    try:
        p = wilcoxon(all_nw_acc, all_rf_acc).pvalue
    except ValueError:
        p = float("nan")

    rmin_worse = int((d_rmin < -0.005).sum())
    rmaj_worse = int((d_rmaj < -0.005).sum())

    print(f'\n=== SUMMARY: Naive weighting vs RF ===')
    print(f'  acc   mean_d={d_acc.mean():+.4f}  W={wins} T={ties} L={losses}  p={p:.4f}')
    print(f'  r_min mean_d={d_rmin.mean():+.4f}  worse(>0.5pp)={rmin_worse}/{len(DATASETS)}')
    print(f'  r_maj mean_d={d_rmaj.mean():+.4f}  worse(>0.5pp)={rmaj_worse}/{len(DATASETS)}')


if __name__ == "__main__":
    run()
