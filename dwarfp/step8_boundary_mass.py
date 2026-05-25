"""step8_boundary_mass.py — Measure RF fp ∈ [0.4, 0.6) share on all 36 datasets,
correlate with Proposed vs RF Δacc. Produces the boundary mass `M` column of
the per-dataset breakdown in Appendix B.4 (Per-Dataset Boundary Mass and
Spread). Outputs results_fp_share.csv.

The boundary-region share is a single RF-only indicator (no weighted
prediction needed) that pre-flags where the method is expected to help.
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
    X, y = load(name)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    Xtr, Xte = X[tr], X[te]
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                max_features="sqrt", bootstrap=True,
                                random_state=SEED + rep, n_jobs=1).fit(Xtr, y[tr])
    forest_proba = rf.predict_proba(Xte)
    n_te = len(Xte)
    fps = []
    for est in rf.estimators_:
        leaf_ids = est.apply(Xte)
        lv = est.tree_.value[leaf_ids, 0, :]
        pred_idx = np.argmax(lv, axis=1)
        fps.append(forest_proba[np.arange(n_te), pred_idx])
    fps = np.concatenate(fps)
    bins = np.minimum(N_BUCK - 1, (fps * N_BUCK).astype(int))
    counts = np.bincount(bins, minlength=N_BUCK).astype(float)
    return counts / counts.sum()


def run():
    print(f"datasets={len(DATASETS)}  repeats={REPEATS}  n_estimators={N_ESTIMATORS}\n")
    # Load Δacc from compare_baselines CSV (shared-forest pipeline)
    cb = {r["dataset"]: float(r["CPFW_acc"]) - float(r["RF_acc"])
          for r in csv.DictReader(open(
              Path(__file__).resolve().parent / "results_baselines.csv"))}

    # Header
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

    # Save CSV
    out_path = Path(__file__).resolve().parent / "results_fp_share.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset"] + BUCK_LABELS + ["d_acc"])
        for r in rows:
            w.writerow([r["dataset"]] + [f"{s:.4f}" for s in r["share_all"]]
                       + [f"{r['d_acc']:+.4f}"])
    print(f"\nSaved: {out_path}")

    # Correlations
    x = np.array([r["share_46"] for r in rows])
    y = np.array([r["d_acc"] for r in rows])
    print(f"\nCorrelation: fp[.4,.6) share vs Δacc (CPFW - RF) on {len(rows)} datasets")
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    print(f"  Pearson r = {pr:+.4f}  (p={pp:.4f})")
    print(f"  Spearman ρ = {sr:+.4f}  (p={sp:.4f})")

    # Top-5 boundary mass datasets
    rows_sorted = sorted(rows, key=lambda r: -r["share_46"])
    print(f"\nTop 5 boundary-mass datasets:")
    for r in rows_sorted[:5]:
        print(f"  {r['dataset']:22s} fp[.4,.6)={r['share_46']*100:5.1f}%  Δacc={r['d_acc']:+.4f}")
    print(f"\nBottom 5 boundary-mass datasets:")
    for r in rows_sorted[-5:]:
        print(f"  {r['dataset']:22s} fp[.4,.6)={r['share_46']*100:5.1f}%  Δacc={r['d_acc']:+.4f}")


if __name__ == "__main__":
    run()
