"""step7_tree_sweep.py — Proposed-vs-RF accuracy stability across N_ESTIMATORS.

Uses the shared vectorized CPFW core from `dwarfp.common`, identical to
step6b and compare_baselines.

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
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(PROJECT_ROOT) / "archive"))
from dwarfp.common import (load, DATASETS,
                            cpfw_collect_table, cpfw_build_weight_table,
                            cpfw_predict_proba)

warnings.filterwarnings("ignore")

TREE_COUNTS = [100, 150, 300, 500, 1000]
REPEATS = 30
TEST_SIZE = 0.3
SEED = 42
N_CV = 5
MIN_N = 30


def _run_one(name, rep, n_est):
    X, y = load(name)
    cls, cnt = np.unique(y, return_counts=True)
    minority = int(cls[np.argmin(cnt)])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    rf = RandomForestClassifier(n_estimators=n_est, max_features="sqrt",
                                bootstrap=True, random_state=SEED + rep,
                                n_jobs=1).fit(Xtr, ytr)

    rf_acc = float(accuracy_score(yte, rf.predict(Xte)))

    R  = cpfw_collect_table(Xtr, ytr, minority, SEED + rep,
                            n_estimators=n_est, n_cv=N_CV)
    W  = cpfw_build_weight_table(R, min_n=MIN_N)
    wp = cpfw_predict_proba(rf, Xte, minority, W)
    fw_acc = float(accuracy_score(yte, rf.classes_[np.argmax(wp, axis=1)]))
    return rf_acc, fw_acc


def run():
    datasets = DATASETS
    n_ds = len(datasets)

    hdr = f"{'trees':>6}  {'RF_acc':>7}  {'FW_acc':>7}  {'d_acc':>8}  {'wins':>5}  {'p':>7}"
    print(f"Tree sweep: {TREE_COUNTS}  repeats={REPEATS}  datasets={n_ds}\n")
    print(hdr)
    print("-" * len(hdr))

    for n_est in TREE_COUNTS:
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
        wins = int((d > 1e-9).sum())
        try:
            p = wilcoxon(fw_accs, rf_accs).pvalue
        except ValueError:
            p = float("nan")
        print(f"{n_est:>6}  {rf_accs.mean():7.4f}  {fw_accs.mean():7.4f}  "
              f"{d.mean():+8.4f}  {wins:>3}/{n_ds}  {p:7.4f}", flush=True)


if __name__ == "__main__":
    run()
