"""compare_baselines.py — CPFW vs RF vs WRF(Winham) vs KNORA-E vs KNORA-U.

Same split / same forest where possible.
Reports: accuracy, minority recall, majority recall per dataset.
Wilcoxon signed-rank for each method vs RF.

NOTE: This is a decision-support script, not paper code.
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

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(PROJECT_ROOT) / "archive"))
from dwarfp.common import load, recalls, classify_pattern, walk_tree, DATASETS

# DES imports — deslib 0.3.7 calls _validate_data which was removed in
# sklearn >=1.6.  Patch BaseDS to use sklearn.utils.validation instead.
from deslib.base import BaseDS
from sklearn.utils.validation import validate_data as _sklearn_validate_data
if not hasattr(BaseDS, '_validate_data'):
    BaseDS._validate_data = lambda self, *a, **kw: _sklearn_validate_data(self, *a, **kw)
from deslib.des import KNORAU, KNORAE

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


# ── CPFW helpers (from step6_eval.py) ─────────────────────────────────

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


# ── WRF (Winham 2013) ───────────────────────────────────────────────

def _oob_indices(rf, n_train):
    out = []
    for est in rf.estimators_:
        rs = est.random_state
        rng = np.random.RandomState(rs)
        sample_indices = rng.randint(0, n_train, n_train)
        in_bag = np.zeros(n_train, dtype=bool)
        in_bag[sample_indices] = True
        out.append(np.where(~in_bag)[0])
    return out


def _wrf_predict(rf, X_train, y_train, X_test):
    """WRF: tree weight = 1/(1-OOB_acc)."""
    oob_idx_list = _oob_indices(rf, len(y_train))
    weights = np.ones(len(rf.estimators_))
    for j, est in enumerate(rf.estimators_):
        idx = oob_idx_list[j]
        if len(idx) == 0:
            continue
        acc = float(np.mean(est.predict(X_train[idx]) == y_train[idx]))
        weights[j] = 1.0 / max(1.0 - acc, 1e-3)

    classes = rf.classes_
    n_cls = len(classes)
    out = np.zeros((len(X_test), n_cls))
    for j, est in enumerate(rf.estimators_):
        proba = est.predict_proba(X_test)
        # align columns to forest classes
        aligned = np.zeros((len(X_test), n_cls))
        for ci, c in enumerate(est.classes_):
            idx_in_forest = np.searchsorted(classes, c)
            aligned[:, idx_in_forest] = proba[:, ci]
        out += weights[j] * aligned
    out /= weights.sum()
    return classes[np.argmax(out, axis=1)]


# ── Main runner ───────────────────────────────────────────────────────

def _ensure_deslib_patch():
    """Ensure deslib patch is applied in subprocess too."""
    from deslib.base import BaseDS
    if not hasattr(BaseDS, '_validate_data'):
        from sklearn.utils.validation import validate_data as _vd
        BaseDS._validate_data = lambda self, *a, **kw: _vd(self, *a, **kw)


def _run_one(name, rep):
    _ensure_deslib_patch()
    X, y = load(name)
    cls, cnt = np.unique(y, return_counts=True)
    minority = int(cls[np.argmin(cnt)])
    majority = int(cls[np.argmax(cnt)])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    # Shared forest
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                max_features="sqrt", bootstrap=True,
                                random_state=SEED + rep, n_jobs=1).fit(Xtr, ytr)

    # 1) RF
    rf_pred = rf.predict(Xte)
    rf_acc = float(accuracy_score(yte, rf_pred))
    rf_rmin, rf_rmaj = recalls(yte, rf_pred, minority, majority)

    # 2) CPFW
    R = _collect_table(Xtr, ytr, minority, SEED + rep)
    W = _build_weight_table(R)
    wp = _weighted_predict(rf, Xte, minority, W)
    cpfw_pred = rf.classes_[np.argmax(wp, axis=1)]
    cpfw_acc = float(accuracy_score(yte, cpfw_pred))
    cpfw_rmin, cpfw_rmaj = recalls(yte, cpfw_pred, minority, majority)

    # 3) WRF (Winham) — same forest, reweight by OOB acc
    wrf_pred = _wrf_predict(rf, Xtr, ytr, Xte)
    wrf_acc = float(accuracy_score(yte, wrf_pred))
    wrf_rmin, wrf_rmaj = recalls(yte, wrf_pred, minority, majority)

    # 4) KNORA-E — same forest's estimators
    try:
        kne = KNORAE(rf.estimators_, random_state=SEED + rep)
        kne.fit(Xtr, ytr)
        kne_pred = kne.predict(Xte)
        kne_acc = float(accuracy_score(yte, kne_pred))
        kne_rmin, kne_rmaj = recalls(yte, kne_pred, minority, majority)
    except Exception:
        kne_acc, kne_rmin, kne_rmaj = np.nan, np.nan, np.nan

    # 5) KNORA-U — same forest's estimators
    try:
        knu = KNORAU(rf.estimators_, random_state=SEED + rep)
        knu.fit(Xtr, ytr)
        knu_pred = knu.predict(Xte)
        knu_acc = float(accuracy_score(yte, knu_pred))
        knu_rmin, knu_rmaj = recalls(yte, knu_pred, minority, majority)
    except Exception:
        knu_acc, knu_rmin, knu_rmaj = np.nan, np.nan, np.nan

    return {
        "rf":   (rf_acc, rf_rmin, rf_rmaj),
        "cpfw": (cpfw_acc, cpfw_rmin, cpfw_rmaj),
        "wrf": (wrf_acc, wrf_rmin, wrf_rmaj),
        "kne":  (kne_acc, kne_rmin, kne_rmaj),
        "knu":  (knu_acc, knu_rmin, knu_rmaj),
    }


METHODS = ["rf", "cpfw", "wrf", "kne", "knu"]
LABELS = {"rf": "RF", "cpfw": "CPFW", "wrf": "WRF", "kne": "KNE", "knu": "KNU"}


def run(datasets=None):
    datasets = datasets or DATASETS
    print(f"repeats={REPEATS}  n_estimators={N_ESTIMATORS}  "
          f"datasets={len(datasets)}")
    print(f"Methods: RF, CPFW, WRF(Winham), KNORA-E, KNORA-U")
    print(f"Same forest shared across all methods per split.\n")

    # header
    hd = f'{"dataset":16s} {"n":>5}'
    for m in METHODS:
        hd += f' {LABELS[m]+"_acc":>8}'
    hd += '  |'
    for m in METHODS:
        hd += f' {LABELS[m]+"_rmi":>8}'
    print(hd)
    print("-" * len(hd))

    # storage
    all_res = {m: {"acc": [], "rmin": [], "rmaj": []} for m in METHODS}

    for name in datasets:
        res_list = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_run_one)(name, r) for r in range(REPEATS))

        X, y = load(name)
        n = len(X)

        row = f'{name:16s} {n:5d}'
        for m in METHODS:
            vals = [r[m][0] for r in res_list]
            mean_acc = float(np.nanmean(vals))
            all_res[m]["acc"].append(mean_acc)
            row += f' {mean_acc:8.4f}'

        row += '  |'
        for m in METHODS:
            vals = [r[m][1] for r in res_list]
            mean_rmin = float(np.nanmean(vals))
            all_res[m]["rmin"].append(mean_rmin)
            row += f' {mean_rmin:8.3f}'

        # also collect rmaj silently
        for m in METHODS:
            vals = [r[m][2] for r in res_list]
            all_res[m]["rmaj"].append(float(np.nanmean(vals)))

        print(row)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY vs RF")
    print("=" * 70)

    rf_acc = np.array(all_res["rf"]["acc"])
    rf_rmin = np.array(all_res["rf"]["rmin"])
    rf_rmaj = np.array(all_res["rf"]["rmaj"])

    for m in METHODS:
        if m == "rf":
            continue
        acc = np.array(all_res[m]["acc"])
        rmin = np.array(all_res[m]["rmin"])
        rmaj = np.array(all_res[m]["rmaj"])

        d_acc = acc - rf_acc
        d_rmin = rmin - rf_rmin
        d_rmaj = rmaj - rf_rmaj

        wins = int((d_acc > 1e-9).sum())
        losses = int((d_acc < -1e-9).sum())
        ties = len(datasets) - wins - losses

        try:
            p = wilcoxon(acc, rf_acc).pvalue
        except ValueError:
            p = float("nan")

        rmin_worse = int((d_rmin < -0.005).sum())
        rmaj_worse = int((d_rmaj < -0.005).sum())

        print(f"\n{LABELS[m]} vs RF:")
        print(f"  acc   mean_d={d_acc.mean():+.4f}  "
              f"W={wins} T={ties} L={losses}  Wilcoxon p={p:.4f}")
        print(f"  minority recall  mean_d={d_rmin.mean():+.4f}  "
              f"worse(>0.5pp)={rmin_worse}/{len(datasets)}")
        print(f"  majority recall  mean_d={d_rmaj.mean():+.4f}  "
              f"worse(>0.5pp)={rmaj_worse}/{len(datasets)}")


if __name__ == "__main__":
    run()
