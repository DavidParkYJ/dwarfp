"""step5_conditional_signal.py — Table 5: conditional pattern signal.

Region = per-tree fp = P̂_RF(x)[pred_t] in [0,1] — the forest's support for the
TREE's own predicted class.  Because the region co-refers with the tree's
predicted class, a cell keyed by (region, tree class) is already a coherent
population: no cell mixes trees that agree with the forest's vote with trees
that dissent.  No forest-class axis is needed (cf. the per-sample region, whose
unsigned max-confidence cell would pool agreeing and dissenting trees).

    cell = (region, tree_class)

The within-cell pattern-accuracy spread is then genuine reliability signal.
Displayed at a coarse 0.20-width bucket resolution (5x2 = 10 cells) for
readability; the deployed weight table uses the 0.10-width buckets of step6.

All working datasets pooled, test-set diagnostic (matches step6's protocol).
"""

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "archive"))  # leaf_aware_rf — needed to unpickle
from dwarfp.common import load, walk_tree_batch, PATTERNS, N_PAT, DATASETS

warnings.filterwarnings("ignore")

N_ESTIMATORS = 300
REPEATS = 5
TEST_SIZE = 0.3
SEED = 42
N_REG = 5
MIN_N = 200
REG_LABELS = ["[.0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1.]"]
CL = {0: "maj", 1: "min"}


def _bucket(fp):
    return np.minimum(N_REG - 1, (np.asarray(fp) * N_REG).astype(int))


def _accumulate(name, rep):
    X, y = load(name)
    cls, cnt = np.unique(y, return_counts=True)
    minority = int(cls[np.argsort(cnt)[0]])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
                                bootstrap=True, random_state=SEED + rep,
                                n_jobs=1).fit(Xtr, ytr)
    n_te = len(Xte)
    forest_proba = rf.predict_proba(Xte)
    R = np.zeros((N_REG, N_PAT, 2, 2))             # region, pattern, tree-class
    for est in rf.estimators_:
        _, leaf_pat, leaf_val, pred_cls = walk_tree_batch(est, Xte)
        pred_idx = np.argmax(leaf_val, axis=1)
        fp = forest_proba[np.arange(n_te), pred_idx]
        pb = _bucket(fp)
        ci = (pred_cls == minority).astype(int)
        cor = (pred_cls == yte).astype(np.float64)
        np.add.at(R[:, :, :, 0], (pb, leaf_pat, ci), cor)
        np.add.at(R[:, :, :, 1], (pb, leaf_pat, ci), 1.0)
    return R


def run():
    jobs = [(n, r) for n in DATASETS for r in range(REPEATS)]
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_accumulate)(n, r) for n, r in jobs)
    R = sum(results)

    print("Table 5: conditional pattern signal under per-tree-fixed region")
    print(f"(fp = P_RF[pred_t], [0,1], {N_REG} buckets of {1/N_REG:.2f})")
    print(f"{len(DATASETS)} datasets x {REPEATS} repeats pooled, MIN_N={MIN_N}\n")
    hd = (f'{"region":9s} {"tree":5s} {"spread":>8} {"best":>10} {"worst":>10} '
          f'{"marg_acc":>9} {"n":>10}')
    print(hd)
    print("-" * len(hd))
    sig = tot_cells = 0
    rows = []
    for rb in range(N_REG):
        for ci in (0, 1):
            cell = R[rb, :, ci, :]
            tot = cell.sum(axis=0)
            marg = tot[0] / tot[1] if tot[1] > 0 else np.nan
            accs = [(cell[p, 0] / cell[p, 1], PATTERNS[p])
                    for p in range(N_PAT) if cell[p, 1] >= MIN_N]
            if len(accs) >= 2:
                s = sorted(accs, reverse=True)
                spread, best, worst = s[0][0] - s[-1][0], s[0][1], s[-1][1]
                sp = f'{spread:>8.3f}'
                tot_cells += 1
                sig += spread > 0.02
            else:
                spread, best, worst = np.nan, "--", "--"
                sp = f'{"--":>8}'
            print(f'{REG_LABELS[rb]:9s} {CL[ci]:5s} {sp} {best:>10} '
                  f'{worst:>10} {marg:>9.3f} {int(tot[1]):>10d}')
            rows.append([REG_LABELS[rb], CL[ci], spread, best, worst,
                         marg, int(tot[1])])
        print()
    print(f"cells with spread > 0.02: {sig}/{tot_cells}")

    out = _ROOT / "dwarfp" / "results_cond_signal.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "tree_cls", "spread", "best", "worst",
                    "marg_acc", "n"])
        for r in rows:
            sp = f"{r[2]:.5f}" if not np.isnan(r[2]) else ""
            w.writerow([r[0], r[1], sp, r[3], r[4], f"{r[5]:.5f}", r[6]])
    print(f"\nwrote {out}")

    # ── Per-(region, pattern) accuracy, pooled across both tree classes ──
    print("\n=== Per-pattern accuracy by region "
          "(pooled across both tree classes) ===\n")
    hd2 = f'{"region":9s}' + ''.join(f'{p:>10s}' for p in PATTERNS) + \
          f'{"marg":>10}'
    print(hd2)
    print("-" * len(hd2))
    pat_rows = []
    for rb in range(N_REG):
        cell = R[rb, :, :, :].sum(axis=1)                # (N_PAT, 2)
        tot  = cell.sum(axis=0)                          # over patterns
        marg = tot[0] / tot[1] if tot[1] > 0 else np.nan
        accs = []
        for p in range(N_PAT):
            n_p = cell[p, 1]
            accs.append(cell[p, 0] / n_p if n_p >= MIN_N else np.nan)
        row_str = f'{REG_LABELS[rb]:9s}' + ''.join(
            f'{a:>10.4f}' if not np.isnan(a) else f'{"--":>10}'
            for a in accs) + f'{marg:>10.4f}'
        print(row_str)
        pat_rows.append([REG_LABELS[rb]] +
                        [f"{a:.5f}" if not np.isnan(a) else "" for a in accs] +
                        [f"{marg:.5f}"])

    out2 = _ROOT / "dwarfp" / "results_cond_signal_region_pattern.csv"
    with open(out2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region"] + list(PATTERNS) + ["marg"])
        w.writerows(pat_rows)
    print(f"\nwrote {out2}")


if __name__ == "__main__":
    run()
