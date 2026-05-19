"""eval_magnitude.py — Ablation: restrict weighting to uncertain region.

step5 showed signal collapses in [.8,1.):
  [.9,1.) spread: maj=0.021, min=0.023
  [.8,.9) spread: maj=0.046, min=0.063

Variants:
  A  MAX_PB=9  all buckets (current default)
  B  MAX_PB=5  fp < 0.80
  C  MAX_PB=3  fp < 0.70

For pb > MAX_PB: w = 1.0 (falls back to standard RF vote).
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
from dwarfp.common import load, recalls, classify_pattern, DATASETS

warnings.filterwarnings("ignore")

N_ESTIMATORS = 150
REPEATS = 20
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
            t = est.tree_
            cl, cr = t.children_left, t.children_right
            nlab = np.argmax(t.value[:, 0, :], axis=1)
            for j, i in enumerate(val_idx):
                xi = X_tr[i]
                node = 0
                labels = [int(nlab[node])]
                while cl[node] != cr[node]:
                    node = (cl[node] if xi[t.feature[node]] <= t.threshold[node]
                            else cr[node])
                    labels.append(int(nlab[node]))
                lv = t.value[node, 0, :]
                pred = classes[int(np.argmax(lv))]
                c = 1.0 if pred == y_tr[i] else 0.0
                ci = 1 if int(pred) == minority else 0
                fpred_idx = np.searchsorted(classes, pred)
                fp = float(forest_proba[j, fpred_idx])
                R[_bucket_fp(fp), classify_pattern(labels), ci, 0] += c
                R[_bucket_fp(fp), classify_pattern(labels), ci, 1] += 1
    return R


def _build_weight_table(R, max_pb):
    W = np.ones((N_PROB, N_PAT, N_CLS))
    for pb in range(min(max_pb + 1, N_PROB)):
        for ci in range(N_CLS):
            marg = R[pb, :, ci, :].sum(axis=0)
            p_marg = marg[0] / marg[1] if marg[1] >= MIN_N else None
            for pat in range(N_PAT):
                v = R[pb, pat, ci]
                if v[1] >= MIN_N and p_marg and p_marg > 0:
                    W[pb, pat, ci] = (v[0] / v[1]) / p_marg
    return W


def _run_one(name, rep, max_pb):
    X, y = load(name)
    cls, cnt = np.unique(y, return_counts=True)
    minority = int(cls[np.argmin(cnt)])
    majority = int(cls[np.argmax(cnt)])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    R = _collect_table(Xtr, ytr, minority, SEED + rep)
    W = _build_weight_table(R, max_pb)

    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                max_features="sqrt", bootstrap=True,
                                random_state=SEED + rep, n_jobs=1).fit(Xtr, ytr)
    classes = rf.classes_
    n_cls = len(classes)
    forest_proba = rf.predict_proba(Xte)

    tree_data = []
    for est in rf.estimators_:
        t = est.tree_
        tree_data.append((t.children_left, t.children_right,
                          np.argmax(t.value[:, 0, :], axis=1),
                          t.feature, t.threshold, t.value))

    out = np.zeros((len(Xte), n_cls))
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
            fpred_idx = np.searchsorted(classes, pred)
            fp = float(forest_proba[i, fpred_idx])
            w = float(W[_bucket_fp(fp), classify_pattern(labels), ci])
            psum += w * (lv / lv.sum())
            wsum += w
        out[i] = psum / wsum if wsum > 0 else np.ones(n_cls) / n_cls

    rf_pred = rf.predict(Xte)
    rf_acc  = float(accuracy_score(yte, rf_pred))
    rf_rmin, rf_rmaj = recalls(yte, rf_pred, minority, majority)

    fw_pred = classes[np.argmax(out, axis=1)]
    fw_acc  = float(accuracy_score(yte, fw_pred))
    fw_rmin, fw_rmaj = recalls(yte, fw_pred, minority, majority)
    return rf_acc, rf_rmin, rf_rmaj, fw_acc, fw_rmin, fw_rmaj


def _eval_variant(datasets, max_pb, label):
    all_rf, all_fw = [], []
    all_rf_rmin, all_fw_rmin = [], []
    all_rf_rmaj, all_fw_rmaj = [], []
    all_n = []
    n_by_ds = {name: len(load(name)[0]) for name in datasets}

    for name in datasets:
        res = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_run_one)(name, r, max_pb) for r in range(REPEATS))
        all_rf.append(float(np.mean([r[0] for r in res])))
        all_fw.append(float(np.mean([r[3] for r in res])))
        all_rf_rmin.append(float(np.mean([r[1] for r in res])))
        all_fw_rmin.append(float(np.mean([r[4] for r in res])))
        all_rf_rmaj.append(float(np.mean([r[2] for r in res])))
        all_fw_rmaj.append(float(np.mean([r[5] for r in res])))
        all_n.append(n_by_ds[name])

    d_acc  = np.array(all_fw)  - np.array(all_rf)
    d_rmin = np.array(all_fw_rmin) - np.array(all_rf_rmin)
    d_rmaj = np.array(all_fw_rmaj) - np.array(all_rf_rmaj)
    wins   = int((d_acc >  1e-9).sum())
    losses = int((d_acc < -1e-9).sum())
    try:
        p = wilcoxon(all_fw, all_rf).pvalue
    except ValueError:
        p = float("nan")

    ns = np.array(all_n)
    med = np.median(ns)
    sm = d_acc[ns <= med]; lg = d_acc[ns > med]

    thresh = 0.50 + (max_pb + 1) * 0.05
    print(f"\n{'='*62}")
    print(f"Variant {label}  max_pb={max_pb}  fp < {thresh:.2f}")
    print(f"  acc   wins={wins:2d}  ties={len(datasets)-wins-losses:2d}  "
          f"losses={losses:2d}  p={p:.4f}  mean_d={d_acc.mean():+.4f}")
    print(f"  recall regressions: minority {int((d_rmin<-1e-9).sum())}/{len(datasets)}  "
          f"majority {int((d_rmaj<-1e-9).sum())}/{len(datasets)}")
    print(f"  d_rmin={d_rmin.mean():+.4f}  d_rmaj={d_rmaj.mean():+.4f}")
    print(f"  size:  small(n<={int(med):4d}) d={sm.mean():+.4f} w={int((sm>1e-9).sum())}/{len(sm)}"
          f"   large(n>{int(med):4d}) d={lg.mean():+.4f} w={int((lg>1e-9).sum())}/{len(lg)}")
    return d_acc.mean(), wins, p


def run():
    print(f"Uncertain-region ablation  "
          f"({len(DATASETS)} datasets × {REPEATS} repeats)\n")
    print("A = all buckets (current), B = fp<0.80, C = fp<0.70")

    summary = []
    for label, max_pb in [("A", 9), ("B", 5), ("C", 3)]:
        mean_d, wins, p = _eval_variant(DATASETS, max_pb, label)
        summary.append((label, max_pb, mean_d, wins, p))

    print(f"\n{'='*62}")
    print(f"{'variant':>8} {'fp_thresh':>10} {'mean_d':>8} {'wins':>6} {'p':>8}")
    print("-" * 45)
    for label, max_pb, mean_d, wins, p in summary:
        thresh = 0.50 + (max_pb + 1) * 0.05
        print(f"{label:>8} {'<'+f'{thresh:.2f}':>10} {mean_d:>+8.4f} {wins:>6} {p:>8.4f}")


if __name__ == "__main__":
    run()
