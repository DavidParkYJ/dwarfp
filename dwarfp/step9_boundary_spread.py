"""step9_boundary_spread.py — Per-dataset boundary pattern-accuracy spread.

Sample-level definition: starting from the boundary sample set
B = {x : OOB-only forest probability max ∈ [0.4, 0.6)} (see step8),
for each x ∈ B accumulate (predicted-class, pattern, correctness) over
the trees that have x out-of-bag. Within each predicted-class subgroup
compute the max−min of pattern accuracies across the 6 flip patterns
(spread); fall back to the available subgroup if either is sparse,
otherwise average the two.

S is read off the OOB by-product of a single RF fit (no test labels)
and reproduces the test-side measurement closely (Pearson r ≈ 0.97 on
36 datasets), while being deployable as a true pre-flag indicator
before any test-set prediction.

Produces the S column of Appendix B.4. Outputs
results_boundary_spread.csv.
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
sys.path.insert(0, str(Path(PROJECT_ROOT) / "archive"))
from dwarfp.common import load, walk_tree_batch, N_PAT, DATASETS

N_ESTIMATORS = 300
TEST_SIZE = 0.3
REPEATS = 30
SEED = 42
MIN_N_CELL = 1     # release cell threshold; class fallback handles sparsity


def _oob_indices_for_tree(est, n_train):
    rng = np.random.RandomState(est.random_state)
    sample_indices = rng.randint(0, n_train, n_train)
    in_bag = np.zeros(n_train, dtype=bool)
    in_bag[sample_indices] = True
    return np.where(~in_bag)[0]


def _accumulate(name, rep):
    """Returns R[ci, pat, {sumc, n}] aggregated over OOB tree×boundary-sample
    pairs (boundary defined by sample-level OOB forest probability)."""
    X, y = load(name)
    cls, cnt = np.unique(y, return_counts=True)
    minority = int(cls[np.argmin(cnt)])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    Xtr, ytr = X[tr], y[tr]
    n_train = len(Xtr)

    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                max_features="sqrt", bootstrap=True,
                                oob_score=True, random_state=SEED + rep,
                                n_jobs=1).fit(Xtr, ytr)

    oob_proba = rf.oob_decision_function_                          # (n_train, n_cls)
    valid = ~np.isnan(oob_proba).any(axis=1)
    sample_fp = np.full(n_train, np.nan)
    sample_fp[valid] = oob_proba[valid].max(axis=1)
    boundary_mask = (sample_fp >= 0.4) & (sample_fp < 0.6)

    R = np.zeros((2, N_PAT, 2))               # [ci, pat, {sumc, n}]
    for est in rf.estimators_:
        oob_idx = _oob_indices_for_tree(est, n_train)
        if len(oob_idx) == 0:
            continue
        oob_b = oob_idx[boundary_mask[oob_idx]]
        if len(oob_b) == 0:
            continue
        _, leaf_pat, _, pred_cls = walk_tree_batch(est, Xtr[oob_b])
        ci = (pred_cls == minority).astype(int)
        cor = (pred_cls == ytr[oob_b]).astype(np.float64)
        np.add.at(R[:, :, 0], (ci, leaf_pat), cor)
        np.add.at(R[:, :, 1], (ci, leaf_pat), 1.0)
    return R


def _spread_per_class(R_pooled, ci, min_n=MIN_N_CELL):
    """max - min of pattern accuracy in tree-class ci. NaN if too few cells."""
    accs = []
    for pat in range(N_PAT):
        v = R_pooled[ci, pat]
        if v[1] >= min_n:
            accs.append(v[0] / v[1])
    if len(accs) < 2:
        return float("nan"), len(accs)
    return float(max(accs) - min(accs)), len(accs)


def run():
    print(f"datasets={len(DATASETS)}  repeats={REPEATS}  n_estimators={N_ESTIMATORS}")
    print("sample-level OOB region: forest_proba.max ∈ [.4,.6)")
    print(f"class: per-(tree, sample) predicted class; cell min_n={MIN_N_CELL}\n")
    cb = {r["dataset"]: float(r["CPFW_acc"]) - float(r["RF_acc"])
          for r in csv.DictReader(open(
              Path(__file__).resolve().parent / "results_baselines.csv"))}

    header = (f"{'dataset':22s} "
              f"{'maj_spread':>10}  {'maj_pats':>8}  "
              f"{'min_spread':>10}  {'min_pats':>8}  "
              f"{'avg_spread':>10}  {'Δacc':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for name in DATASETS:
        R_list = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_accumulate)(name, rep) for rep in range(REPEATS))
        R_pooled = sum(R_list)
        maj_spread, maj_nc = _spread_per_class(R_pooled, 0)
        min_spread, min_nc = _spread_per_class(R_pooled, 1)
        vals = [v for v in (maj_spread, min_spread) if not np.isnan(v)]
        avg_spread = float(np.mean(vals)) if vals else float("nan")
        d_acc = cb.get(name, float("nan"))
        rows.append({
            "dataset": name,
            "maj_spread": maj_spread, "maj_pats": maj_nc,
            "min_spread": min_spread, "min_pats": min_nc,
            "avg_spread": avg_spread, "d_acc": d_acc,
        })
        ms_s = f"{maj_spread:>10.4f}" if not np.isnan(maj_spread) else f"{'--':>10}"
        mn_s = f"{min_spread:>10.4f}" if not np.isnan(min_spread) else f"{'--':>10}"
        av_s = f"{avg_spread:>10.4f}" if not np.isnan(avg_spread) else f"{'--':>10}"
        print(f"{name:22s} {ms_s}  {maj_nc:>8d}  {mn_s}  {min_nc:>8d}  "
              f"{av_s}  {d_acc:+8.4f}")

    out_path = Path(__file__).resolve().parent / "results_boundary_spread.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "maj_spread", "maj_n_patterns",
                    "min_spread", "min_n_patterns", "avg_spread", "d_acc"])
        for r in rows:
            w.writerow([r["dataset"],
                        f"{r['maj_spread']:.4f}" if not np.isnan(r['maj_spread']) else "",
                        r["maj_pats"],
                        f"{r['min_spread']:.4f}" if not np.isnan(r['min_spread']) else "",
                        r["min_pats"],
                        f"{r['avg_spread']:.4f}" if not np.isnan(r['avg_spread']) else "",
                        f"{r['d_acc']:+.4f}"])
    print(f"\nSaved: {out_path}\n")

    def _corr(key):
        x = np.array([r[key] for r in rows])
        y = np.array([r["d_acc"] for r in rows])
        valid = ~np.isnan(x)
        x, y = x[valid], y[valid]
        n = len(x)
        pr, pp = pearsonr(x, y) if n >= 3 else (np.nan, np.nan)
        sr, sp = spearmanr(x, y) if n >= 3 else (np.nan, np.nan)
        print(f"{key:>10s}  n={n}  Pearson r={pr:+.4f} (p={pp:.4f})  "
              f"Spearman ρ={sr:+.4f} (p={sp:.4f})")

    print("Correlation: boundary spread vs Δacc (Proposed − RF)")
    _corr("maj_spread")
    _corr("min_spread")
    _corr("avg_spread")

    rows_sorted = sorted([r for r in rows if not np.isnan(r["avg_spread"])],
                        key=lambda r: -r["avg_spread"])
    print("\nTop 5 boundary spread:")
    for r in rows_sorted[:5]:
        print(f"  {r['dataset']:22s} avg_spread={r['avg_spread']:.4f}  Δacc={r['d_acc']:+.4f}")
    print("Bottom 5 boundary spread:")
    for r in rows_sorted[-5:]:
        print(f"  {r['dataset']:22s} avg_spread={r['avg_spread']:.4f}  Δacc={r['d_acc']:+.4f}")


if __name__ == "__main__":
    run()
