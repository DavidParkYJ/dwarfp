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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(PROJECT_ROOT) / "archive"))
from dwarfp.common import (load, recalls, classify_pattern, walk_tree,
                            DATASETS)

warnings.filterwarnings("ignore")

N_ESTIMATORS = 150
REPEATS = 30
TEST_SIZE = 0.3
SEED = 42
N_CV = 5
MIN_N = 30
N_PROB = 10
N_PAT = 6
N_CLS = 2


def _bucket_fp(fp):
    return min(9, int((fp - 0.5) / 0.05))


def _collect_table(X_tr, y_tr, minority, seed):
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=seed)
    R = np.zeros((N_PROB, N_PAT, N_CLS, 2))
    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                    max_features="sqrt", bootstrap=True,
                                    random_state=seed, n_jobs=1
                                    ).fit(X_tr[tr_idx], y_tr[tr_idx])
        classes = rf.classes_
        forest_proba = rf.predict_proba(X_tr[val_idx])
        for est in rf.estimators_:
            for j, (labels, lv) in enumerate(walk_tree(est, X_tr[val_idx])):
                pred = classes[int(np.argmax(lv))]
                c = 1.0 if pred == y_tr[val_idx[j]] else 0.0
                ci = 1 if int(pred) == minority else 0
                pred_idx = np.searchsorted(classes, pred)
                fp = float(forest_proba[j, pred_idx])
                R[_bucket_fp(fp), classify_pattern(labels), ci, 0] += c
                R[_bucket_fp(fp), classify_pattern(labels), ci, 1] += 1
    return R


def _build_weight_table(R):
    W = np.ones((N_PROB, N_PAT, N_CLS))
    for pb in range(N_PROB):
        for ci in range(N_CLS):
            marg = R[pb, :, ci, :].sum(axis=0)
            p_marg = marg[0] / marg[1] if marg[1] >= MIN_N else None
            for pat in range(N_PAT):
                v = R[pb, pat, ci]
                if v[1] >= MIN_N and p_marg and p_marg > 0:
                    W[pb, pat, ci] = (v[0] / v[1]) / p_marg
    return W


def _weighted_predict(rf, Xte, minority, W):
    classes = rf.classes_
    n_cls = len(classes)
    out = np.zeros((len(Xte), n_cls))
    forest_proba = rf.predict_proba(Xte)
    tree_data = []
    for est in rf.estimators_:
        t = est.tree_
        tree_data.append((t.children_left, t.children_right,
                          np.argmax(t.value[:, 0, :], axis=1),
                          t.feature, t.threshold, t.value))
    for i in range(len(Xte)):
        xi = Xte[i]
        psum = np.zeros(n_cls)
        wsum = 0.0
        for (cl, cr, nlab, feat, thr, val) in tree_data:
            node = 0
            labels = [int(nlab[node])]
            while cl[node] != cr[node]:
                node = cl[node] if xi[feat[node]] <= thr[node] else cr[node]
                labels.append(int(nlab[node]))
            lv = val[node, 0, :]
            pred = classes[int(np.argmax(lv))]
            ci = 1 if int(pred) == minority else 0
            pred_idx = np.searchsorted(classes, pred)
            fp = float(forest_proba[i, pred_idx])
            w = float(W[_bucket_fp(fp), classify_pattern(labels), ci])
            psum += w * (lv / lv.sum())
            wsum += w
        out[i] = psum / wsum if wsum > 0 else np.ones(n_cls) / n_cls
    return out


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

    R = _collect_table(Xtr, ytr, minority, SEED + rep)
    W = _build_weight_table(R)
    wp = _weighted_predict(rf, Xte, minority, W)
    cpfw_pred = rf.classes_[np.argmax(wp, axis=1)]
    cpfw_acc = float(accuracy_score(yte, cpfw_pred))
    cpfw_rmin, cpfw_rmaj = recalls(yte, cpfw_pred, minority, majority)
    return rf_acc, rf_rmin, rf_rmaj, cpfw_acc, cpfw_rmin, cpfw_rmaj


def run():
    print(f"repeats={REPEATS}  n_estimators={N_ESTIMATORS}\n")

    rows = []
    for name in DATASETS:
        print(f"  {name}...", end="", flush=True)
        res = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_run_one)(name, r) for r in range(REPEATS))
        X, _ = load(name)
        n = len(X)
        rf_acc  = float(np.mean([r[0] for r in res]))
        rf_rmin = float(np.mean([r[1] for r in res]))
        cpfw_acc  = float(np.mean([r[3] for r in res]))
        cpfw_rmin = float(np.mean([r[4] for r in res]))
        d_acc = cpfw_acc - rf_acc
        d_rmin = cpfw_rmin - rf_rmin
        print(f"  d_acc={d_acc:+.4f}")
        rows.append({
            "dataset": name, "n": n,
            "rf_acc": f"{rf_acc:.4f}", "cpfw_acc": f"{cpfw_acc:.4f}",
            "d_acc": f"{d_acc:+.4f}",
            "rf_rmin": f"{rf_rmin:.4f}", "cpfw_rmin": f"{cpfw_rmin:.4f}",
            "d_rmin": f"{d_rmin:+.4f}",
        })

    out_path = Path(__file__).resolve().parent / "results_size_effect.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_path}")

    # Summary
    ns = np.array([int(r["n"]) for r in rows])
    d = np.array([float(r["d_acc"]) for r in rows])
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
