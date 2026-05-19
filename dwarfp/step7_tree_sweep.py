"""step7_tree_sweep.py — CPFW d_acc stability across N_ESTIMATORS.

Optimisation: pre-computes leaf patterns per tree (DFS once), then uses
est.apply(X) for leaf lookup + array indexing. Avoids decision_path(),
padded matrix construction, and batch pattern classification per sample.

Tree counts: [100, 150, 300, 500, 1000]
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dwarfp.common import load, DATASETS, precompute_leaf_patterns

warnings.filterwarnings("ignore")

TREE_COUNTS = [100, 150, 300, 500, 1000]
REPEATS = 30
TEST_SIZE = 0.3
SEED = 42
N_CV = 5
MIN_N = 30
N_PROB = 10
N_PAT = 6
N_CLS = 2


def _bucket_fp(fp_arr):
    return np.minimum(9, ((fp_arr - 0.5) / 0.05).astype(int))


def _collect_table(X_tr, y_tr, minority, seed, n_est):
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=seed)
    R = np.zeros((N_PROB, N_PAT, N_CLS, 2))

    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        rf = RandomForestClassifier(n_estimators=n_est, max_features="sqrt",
                                    bootstrap=True, random_state=seed,
                                    n_jobs=1).fit(X_tr[tr_idx], y_tr[tr_idx])
        classes = rf.classes_
        X_val, y_val = X_tr[val_idx], y_tr[val_idx]
        n_val = len(val_idx)
        forest_proba = rf.predict_proba(X_val)

        for est in rf.estimators_:
            leaf_pat = precompute_leaf_patterns(est)    # O(n_leaves), once
            leaf_ids = est.apply(X_val)                 # C code, (n_val,)
            t = est.tree_

            pat      = leaf_pat[leaf_ids]               # array indexing
            lv_mat   = t.value[leaf_ids, 0, :]          # (n_val, n_cls)
            pred_idx = np.argmax(lv_mat, axis=1)
            pred_cls = classes[pred_idx]

            fp  = forest_proba[np.arange(n_val), pred_idx]
            pb  = _bucket_fp(fp)
            ci  = (pred_cls == minority).astype(int)
            cor = (pred_cls == y_val).astype(np.float64)

            np.add.at(R[:, :, :, 0], (pb, pat, ci), cor)
            np.add.at(R[:, :, :, 1], (pb, pat, ci), 1.0)

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
    n_cls   = len(classes)
    n_te    = len(Xte)
    psum    = np.zeros((n_te, n_cls))
    wsum    = np.zeros(n_te)

    forest_proba = rf.predict_proba(Xte)

    for est in rf.estimators_:
        leaf_pat = precompute_leaf_patterns(est)
        leaf_ids = est.apply(Xte)
        t = est.tree_

        pat      = leaf_pat[leaf_ids]
        lv_mat   = t.value[leaf_ids, 0, :]
        lv_norm  = lv_mat / lv_mat.sum(axis=1, keepdims=True)
        pred_idx = np.argmax(lv_mat, axis=1)
        pred_cls = classes[pred_idx]

        fp = forest_proba[np.arange(n_te), pred_idx]
        pb = _bucket_fp(fp)
        ci = (pred_cls == minority).astype(int)

        w = W[pb, pat, ci]
        psum += w[:, np.newaxis] * lv_norm
        wsum += w

    safe = np.where(wsum > 0, wsum, 1.0)
    return psum / safe[:, np.newaxis]


def _run_one(name, rep, n_est):
    X, y = load(name)
    cls, cnt = np.unique(y, return_counts=True)
    minority = int(cls[np.argmin(cnt)])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    R  = _collect_table(Xtr, ytr, minority, SEED + rep, n_est)
    W  = _build_weight_table(R)
    rf = RandomForestClassifier(n_estimators=n_est, max_features="sqrt",
                                bootstrap=True, random_state=SEED + rep,
                                n_jobs=1).fit(Xtr, ytr)

    rf_acc = float(accuracy_score(yte, rf.predict(Xte)))
    wp     = _weighted_predict(rf, Xte, minority, W)
    fw_acc = float(accuracy_score(yte, rf.classes_[np.argmax(wp, axis=1)]))
    return rf_acc, fw_acc


def run():
    datasets = DATASETS
    n_ds = len(datasets)
    all_d  = np.zeros((len(TREE_COUNTS), n_ds))
    all_rf = np.zeros((len(TREE_COUNTS), n_ds))

    hdr = f"{'trees':>6}  {'RF_acc':>7}  {'FW_acc':>7}  {'d_acc':>8}  {'wins':>5}  {'p':>7}"
    print(f"Tree sweep: {TREE_COUNTS}  repeats={REPEATS}  datasets={n_ds}\n")
    print(hdr)
    print("-" * len(hdr))

    for i, n_est in enumerate(TREE_COUNTS):
        print(f"  running n_estimators={n_est}...", flush=True)
        rf_accs, fw_accs = [], []
        for name in datasets:
            res = Parallel(n_jobs=-1, prefer="processes")(
                delayed(_run_one)(name, r, n_est) for r in range(REPEATS))
            rf_accs.append(float(np.mean([r[0] for r in res])))
            fw_accs.append(float(np.mean([r[1] for r in res])))

        rf_accs = np.array(rf_accs)
        fw_accs = np.array(fw_accs)
        d = fw_accs - rf_accs
        all_d[i]  = d
        all_rf[i] = rf_accs
        wins = int((d > 1e-9).sum())
        try:
            p = wilcoxon(fw_accs, rf_accs).pvalue
        except ValueError:
            p = float("nan")
        print(f"{n_est:>6}  {rf_accs.mean():7.4f}  {fw_accs.mean():7.4f}  "
              f"{d.mean():+8.4f}  {wins:>3}/{n_ds}  {p:7.4f}", flush=True)

    print("\n== Per-dataset d_acc trend ==")
    print(f"{'dataset':16s}", end="")
    for n in TREE_COUNTS:
        print(f"  {n:>5}", end="")
    print(f"  {'trend':>6}")
    print("-" * (16 + 8 * len(TREE_COUNTS) + 8))

    trends = []
    for j, name in enumerate(datasets):
        row = all_d[:, j]
        slope = np.polyfit(np.log(TREE_COUNTS), row, 1)[0]
        trend = "grow" if slope > 5e-5 else ("shrink" if slope < -5e-5 else "stable")
        trends.append(trend)
        print(f"{name:16s}", end="")
        for v in row:
            print(f"  {v:+.4f}", end="")
        print(f"  {trend:>6}")

    from collections import Counter
    c = Counter(trends)
    print(f"\nTrend summary: grow={c['grow']}  stable={c['stable']}  shrink={c['shrink']}")
    print(f"RF acc gain (150→1000): {all_rf[-1].mean() - all_rf[1].mean():+.4f}")


if __name__ == "__main__":
    run()
