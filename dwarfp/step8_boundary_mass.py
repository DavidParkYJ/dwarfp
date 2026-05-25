"""step8_boundary_mass.py — Per-dataset boundary mass M.

Sample-level definition: M is the fraction of training samples whose
**OOB-only forest probability** for the winning class falls in [0.4, 0.6).
The forest probability for each sample is computed from the trees that
have that sample out-of-bag (`rf.oob_decision_function_`), so no test
labels are used and the indicator can be read off a single RF fit's
free OOB by-product before the method is applied.

Produces the M column of Appendix B.4 (Per-Dataset Boundary Mass and
Spread). Outputs results_fp_share.csv with all 5 bucket shares for
diagnostics (`[.0,.2)` through `[.8,1.]`).
"""

import csv
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(PROJECT_ROOT) / "archive"))   # leaf_aware_rf for pickle
from dwarfp.common import load, DATASETS

N_ESTIMATORS = 300
TEST_SIZE = 0.3
REPEATS = 30
SEED = 42
N_BUCK = 5
BUCK_LABELS = ["[.0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1.]"]


def _fp_share_one(name, rep):
    """Per-sample OOB forest-probability distribution across the 5 buckets.

    For each training sample, the forest probability is the OOB-only vote
    (the OOB decision function), so the boundary indicator does not depend
    on the test set or on any in-bag overfit.
    """
    X, y = load(name)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    Xtr, ytr = X[tr], y[tr]

    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                max_features="sqrt", bootstrap=True,
                                oob_score=True, random_state=SEED + rep,
                                n_jobs=1).fit(Xtr, ytr)
    oob_proba = rf.oob_decision_function_                          # (n_train, n_cls)
    valid = ~np.isnan(oob_proba).any(axis=1)
    sample_fp = oob_proba[valid].max(axis=1)
    bins = np.minimum(N_BUCK - 1, (sample_fp * N_BUCK).astype(int))
    counts = np.bincount(bins, minlength=N_BUCK).astype(float)
    return counts / counts.sum() if counts.sum() > 0 else np.zeros(N_BUCK)


def run():
    print(f"datasets={len(DATASETS)}  repeats={REPEATS}  n_estimators={N_ESTIMATORS}")
    print("sample-level OOB forest probability (rf.oob_decision_function_)\n")
    cb = {r["dataset"]: float(r["d_CPFW_acc"])
          for r in csv.DictReader(open(
              Path(__file__).resolve().parent / "results_baselines.csv"))}

    header = f"{'dataset':22s} " + " ".join(f"{lbl:>8}" for lbl in BUCK_LABELS) + f"  {'Δacc':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for name in DATASETS:
        res = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_fp_share_one)(name, rep) for rep in range(REPEATS))
        share = np.mean(res, axis=0)
        d_acc = cb.get(name, float("nan"))
        rows.append({"dataset": name, "share_46": float(share[2]),
                     "share_all": share.tolist(), "d_acc": d_acc})
        print(f"{name:22s} " + " ".join(f"{s*100:>7.1f}%" for s in share)
              + f"  {d_acc:+.4f}")

    out_path = Path(__file__).resolve().parent / "results_fp_share.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset"] + BUCK_LABELS + ["d_acc"])
        for r in rows:
            w.writerow([r["dataset"]] + [f"{s:.4f}" for s in r["share_all"]]
                       + [f"{r['d_acc']:+.4f}"])
    print(f"\nSaved: {out_path}")

    x = np.array([r["share_46"] for r in rows])
    y = np.array([r["d_acc"] for r in rows])
    print(f"\nCorrelation: fp[.4,.6) sample share vs Δacc (Proposed − RF) on {len(rows)} datasets")
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    print(f"  Pearson r = {pr:+.4f}  (p={pp:.4f})")
    print(f"  Spearman ρ = {sr:+.4f}  (p={sp:.4f})")

    rows_sorted = sorted(rows, key=lambda r: -r["share_46"])
    print(f"\nTop 5 boundary-mass datasets:")
    for r in rows_sorted[:5]:
        print(f"  {r['dataset']:22s} fp[.4,.6)={r['share_46']*100:5.1f}%  Δacc={r['d_acc']:+.4f}")
    print(f"\nBottom 5 boundary-mass datasets:")
    for r in rows_sorted[-5:]:
        print(f"  {r['dataset']:22s} fp[.4,.6)={r['share_46']*100:5.1f}%  Δacc={r['d_acc']:+.4f}")


if __name__ == "__main__":
    run()
