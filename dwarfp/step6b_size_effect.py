"""step6b_size_effect.py — Per-dataset accuracy delta (RF vs CPFW) for size effect appendix.

Runs RF + CPFW only (no WRF/KNORA) with 30 repeats to get full-precision
per-dataset deltas.  Saves CSV for the appendix table.
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
from dwarfp.common import (load, recalls, DATASETS,
                            cpfw_collect_table, cpfw_build_weight_table,
                            cpfw_predict_proba)

warnings.filterwarnings("ignore")

N_ESTIMATORS = 300
REPEATS = 30
TEST_SIZE = 0.3
SEED = 42
N_CV = 5
MIN_N = 30


def _run_one(name, rep):
    X, y = load(name)
    cls, cnt = np.unique(y, return_counts=True)
    minority = int(cls[np.argmin(cnt)])
    majority = int(cls[np.argmax(cnt)])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                max_features="sqrt", bootstrap=True,
                                random_state=SEED + rep, n_jobs=1).fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)
    rf_acc = float(accuracy_score(yte, rf_pred))
    rf_rmin, rf_rmaj = recalls(yte, rf_pred, minority, majority)

    R = cpfw_collect_table(Xtr, ytr, minority, SEED + rep,
                           n_estimators=N_ESTIMATORS, n_cv=N_CV)
    W = cpfw_build_weight_table(R, min_n=MIN_N)
    wp = cpfw_predict_proba(rf, Xte, minority, W)
    cpfw_pred = rf.classes_[np.argmax(wp, axis=1)]
    cpfw_acc = float(accuracy_score(yte, cpfw_pred))
    cpfw_rmin, cpfw_rmaj = recalls(yte, cpfw_pred, minority, majority)
    return rf_acc, rf_rmin, rf_rmaj, cpfw_acc, cpfw_rmin, cpfw_rmaj


def run():
    print(f"repeats={REPEATS}  n_estimators={N_ESTIMATORS}\n")

    rows = []
    rf_a, cp_a = [], []
    rf_rmi, cp_rmi = [], []
    rf_rma, cp_rma = [], []
    for name in DATASETS:
        print(f"  {name}...", end="", flush=True)
        res = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_run_one)(name, r) for r in range(REPEATS))
        X, _ = load(name)
        n = len(X)
        rf_acc  = float(np.mean([r[0] for r in res]))
        rf_rmin = float(np.mean([r[1] for r in res]))
        rf_rmaj = float(np.mean([r[2] for r in res]))
        cpfw_acc  = float(np.mean([r[3] for r in res]))
        cpfw_rmin = float(np.mean([r[4] for r in res]))
        cpfw_rmaj = float(np.mean([r[5] for r in res]))
        d_acc  = cpfw_acc - rf_acc
        d_rmin = cpfw_rmin - rf_rmin
        d_rmaj = cpfw_rmaj - rf_rmaj
        print(f"  d_acc={d_acc:+.4f}")
        rf_a.append(rf_acc); cp_a.append(cpfw_acc)
        rf_rmi.append(rf_rmin); cp_rmi.append(cpfw_rmin)
        rf_rma.append(rf_rmaj); cp_rma.append(cpfw_rmaj)
        rows.append({
            "dataset": name, "n": n,
            "rf_acc": f"{rf_acc:.4f}", "cpfw_acc": f"{cpfw_acc:.4f}",
            "d_acc": f"{d_acc:+.4f}",
            "rf_rmin": f"{rf_rmin:.4f}", "cpfw_rmin": f"{cpfw_rmin:.4f}",
            "d_rmin": f"{d_rmin:+.4f}",
            "rf_rmaj": f"{rf_rmaj:.4f}", "cpfw_rmaj": f"{cpfw_rmaj:.4f}",
            "d_rmaj": f"{d_rmaj:+.4f}",
        })

    out_path = Path(__file__).resolve().parent / "results_size_effect.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_path}")

    # ── Aggregate vs RF ──────────────────────────────────────────────
    d      = np.array(cp_a) - np.array(rf_a)
    drmin  = np.array(cp_rmi) - np.array(rf_rmi)
    drmaj  = np.array(cp_rma) - np.array(rf_rma)
    W = int((d > 1e-9).sum())
    L = int((d < -1e-9).sum())
    p = wilcoxon(np.array(cp_a), np.array(rf_a)).pvalue
    print(f"\n=== CPFW vs RF (aggregate, {len(DATASETS)} datasets) ===")
    print(f"  d_acc  mean={d.mean():+.4f}  W={W} T={len(DATASETS)-W-L} L={L}  "
          f"Wilcoxon p={p:.4f}")
    print(f"  d_rmin mean={drmin.mean():+.4f}  "
          f"worse(>0.2pp)={int((drmin < -0.002).sum())}/{len(DATASETS)}")
    print(f"  d_rmaj mean={drmaj.mean():+.4f}  "
          f"worse(>0.2pp)={int((drmaj < -0.002).sum())}/{len(DATASETS)}")

    # ── Size effect ──────────────────────────────────────────────────
    ns = np.array([int(r["n"]) for r in rows])
    med = np.median(ns)
    small = d[ns <= med]
    large = d[ns > med]
    print(f"\nMedian n = {int(med)}")
    print(f"Small (n<={int(med)}): mean_d={small.mean():+.4f}  "
          f"wins={int((small > 1e-9).sum())}/{len(small)}")
    print(f"Large (n>{int(med)}):  mean_d={large.mean():+.4f}  "
          f"wins={int((large > 1e-9).sum())}/{len(large)}")


if __name__ == "__main__":
    run()
