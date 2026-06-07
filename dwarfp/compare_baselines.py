"""compare_baselines.py — headline evaluation engine.

CPFW (proposed) vs RF vs WRF(Winham) vs KNORA-E vs KNORA-U, same split / same
forest per repeat. Reports accuracy, minority recall, majority recall per
dataset and the Wilcoxon signed-rank test for each method vs RF; writes the
canonical results_baselines.csv (with full-precision d_*_acc columns consumed
by step8/step9/step11).

This module is the engine behind the managed entry point `step6_eval`; the
CPFW core itself lives in `dwarfp.common` (cpfw_collect_table /
cpfw_build_weight_table / cpfw_predict_proba). Run via:
    python -m dwarfp.step6_eval     # canonical
    python -c "from dwarfp import compare_baselines as cb; cb.run()"  # equivalent
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

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(PROJECT_ROOT) / "archive"))
from dwarfp.common import (load, recalls, DATASETS,
                            cpfw_collect_table, cpfw_build_weight_table,
                            cpfw_predict_proba)

# DES imports — deslib 0.3.7 calls _validate_data which was removed in
# sklearn >=1.6.  Patch BaseDS to use sklearn.utils.validation instead.
from deslib.base import BaseDS
from sklearn.utils.validation import validate_data as _sklearn_validate_data
if not hasattr(BaseDS, '_validate_data'):
    BaseDS._validate_data = lambda self, *a, **kw: _sklearn_validate_data(self, *a, **kw)
from deslib.des import KNORAU, KNORAE

warnings.filterwarnings("ignore")

N_ESTIMATORS = 300
REPEATS = 30
TEST_SIZE = 0.3
SEED = 42
N_CV = 5
MIN_N = 30


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

    # 2) CPFW (shared vectorized impl from common; identical to step6b)
    R = cpfw_collect_table(Xtr, ytr, minority, SEED + rep,
                           n_estimators=N_ESTIMATORS, n_cv=N_CV)
    W = cpfw_build_weight_table(R, min_n=MIN_N)
    wp = cpfw_predict_proba(rf, Xte, minority, W)
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

    # ── Persist per-dataset acc/rmin/rmaj ──────────────────────────────
    # In addition to the 4-decimal-rounded acc/rmin/rmaj per method, we
    # store full-precision differences (vs RF) rounded to 4 decimals.
    # This avoids precision loss that would otherwise occur if a reader
    # subtracted the 4-decimal columns directly.
    import csv as _csv
    out_path = Path(__file__).resolve().parent / "results_baselines.csv"
    fieldnames = ["dataset", "n"]
    for m in METHODS:
        fieldnames += [f"{LABELS[m]}_acc", f"{LABELS[m]}_rmin", f"{LABELS[m]}_rmaj"]
    for m in METHODS:
        if m == "rf":
            continue
        fieldnames += [f"d_{LABELS[m]}_acc",
                       f"d_{LABELS[m]}_rmin",
                       f"d_{LABELS[m]}_rmaj"]
    with open(out_path, "w", newline="") as fcsv:
        w = _csv.DictWriter(fcsv, fieldnames=fieldnames)
        w.writeheader()
        for i, name in enumerate(datasets):
            X, _ = load(name)
            row = {"dataset": name, "n": len(X)}
            for m in METHODS:
                row[f"{LABELS[m]}_acc"]  = f"{all_res[m]['acc'][i]:.4f}"
                row[f"{LABELS[m]}_rmin"] = f"{all_res[m]['rmin'][i]:.4f}"
                row[f"{LABELS[m]}_rmaj"] = f"{all_res[m]['rmaj'][i]:.4f}"
            for m in METHODS:
                if m == "rf":
                    continue
                d_acc  = all_res[m]['acc'][i]  - all_res['rf']['acc'][i]
                d_rmin = all_res[m]['rmin'][i] - all_res['rf']['rmin'][i]
                d_rmaj = all_res[m]['rmaj'][i] - all_res['rf']['rmaj'][i]
                row[f"d_{LABELS[m]}_acc"]  = f"{d_acc:+.4f}"
                row[f"d_{LABELS[m]}_rmin"] = f"{d_rmin:+.4f}"
                row[f"d_{LABELS[m]}_rmaj"] = f"{d_rmaj:+.4f}"
            w.writerow(row)
    print(f"\nSaved: {out_path}")

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

        rmin_worse = int((d_rmin < -0.002).sum())
        rmaj_worse = int((d_rmaj < -0.002).sum())

        print(f"\n{LABELS[m]} vs RF:")
        print(f"  acc   mean_d={d_acc.mean():+.4f}  "
              f"W={wins} T={ties} L={losses}  Wilcoxon p={p:.4f}")
        print(f"  minority recall  mean_d={d_rmin.mean():+.4f}  "
              f"worse(>0.2pp)={rmin_worse}/{len(datasets)}")
        print(f"  majority recall  mean_d={d_rmaj.mean():+.4f}  "
              f"worse(>0.2pp)={rmaj_worse}/{len(datasets)}")


if __name__ == "__main__":
    run()
