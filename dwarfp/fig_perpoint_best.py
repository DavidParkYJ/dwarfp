"""fig_perpoint_best.py — Per-point best pattern scatter for 4 synthetic 2D datasets.

For each data point, determines the best pattern in its (forest-probability
bucket, predicted-class) cell and colours the point accordingly.
Marker shape distinguishes true class.

Shows that the optimal pattern type varies spatially within a single dataset,
motivating per-cell weight estimation rather than a global pattern ranking.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.datasets import make_moons, make_circles
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dwarfp.common import precompute_leaf_patterns, PATTERNS

N_SAMPLES = 2000
N_ESTIMATORS = 300
SEED = 42
N_CV = 5
MIN_N = 20
N_PAT = 6

PAT_COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12", "#95a5a6"]


# ── synthetic datasets (same as fig_synthetic_2d.py) ─────────────────

def make_diagonal(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    y = ((X[:, 1] > X[:, 0] + rng.normal(0, 0.4, n))).astype(int)
    return X, y


def make_overlap(n, seed):
    rng = np.random.default_rng(seed)
    n0, n1 = n // 2, n - n // 2
    X0 = rng.normal(loc=[0, 0], scale=1.0, size=(n0, 2))
    X1 = rng.normal(loc=[1.2, 1.2], scale=1.0, size=(n1, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)
    return X, y


SYNTH_DATASETS = [
    ("Diagonal", lambda: make_diagonal(N_SAMPLES, SEED)),
    ("Moons",    lambda: make_moons(n_samples=N_SAMPLES, noise=0.30,
                                     random_state=SEED)),
    ("Circles",  lambda: make_circles(n_samples=N_SAMPLES, noise=0.15,
                                       factor=0.4, random_state=SEED)),
    ("Overlap",  lambda: make_overlap(N_SAMPLES, SEED)),
]


# ── helpers ──────────────────────────────────────────────────────────

def _coarse_bucket(fp):
    return np.minimum(4, ((fp - 0.5) / 0.10).astype(int))


def collect_per_sample(X, y):
    """For each sample (via CV), return per-sample best pattern.

    Returns
    -------
    best_pat : (n,) int — best pattern index for each sample's (pb, ci) cell
    fp_vals  : (n,) float — forest probability for each sample
    pred_cls : (n,) int — forest-predicted class for each sample
    """
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)

    # Accumulate cell-level accuracy: R[cb, pat, ci] = (sum_correct, count)
    R = np.zeros((5, N_PAT, 2, 2))

    # Also store per-sample info to map back
    sample_fp = np.zeros(len(X))
    sample_pred = np.zeros(len(X), dtype=int)
    sample_count = np.zeros(len(X))  # how many folds this sample appeared in

    for tr_idx, val_idx in skf.split(X, y):
        rf_cv = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_features="sqrt",
            bootstrap=True, random_state=SEED, n_jobs=1
        ).fit(X[tr_idx], y[tr_idx])
        classes = rf_cv.classes_
        X_val, y_val = X[val_idx], y[val_idx]
        forest_proba = rf_cv.predict_proba(X_val)
        forest_pred = classes[np.argmax(forest_proba, axis=1)]

        # Store sample-level forest prediction
        sample_fp[val_idx] = np.max(forest_proba, axis=1)
        sample_pred[val_idx] = forest_pred.astype(int)
        sample_count[val_idx] += 1

        for est in rf_cv.estimators_:
            leaf_pat = precompute_leaf_patterns(est)
            leaf_ids = est.apply(X_val)
            t = est.tree_
            lv = t.value[leaf_ids, 0, :]
            pred_idx = np.argmax(lv, axis=1)
            pred_cls = classes[pred_idx]

            fp_tree = forest_proba[np.arange(len(X_val)), pred_idx]
            cb = _coarse_bucket(fp_tree)
            pat = leaf_pat[leaf_ids]
            ci = pred_cls.astype(int)
            correct = (pred_cls == y_val).astype(float)

            for j in range(len(X_val)):
                R[cb[j], pat[j], ci[j], 0] += correct[j]
                R[cb[j], pat[j], ci[j], 1] += 1

    # Compute best pattern per (cb, ci) cell
    best_pat_table = np.zeros((5, 2), dtype=int)
    for cb in range(5):
        for ci in range(2):
            best_acc = -1.0
            best_p = 0
            for p in range(N_PAT):
                if R[cb, p, ci, 1] >= MIN_N:
                    acc = R[cb, p, ci, 0] / R[cb, p, ci, 1]
                    if acc > best_acc:
                        best_acc = acc
                        best_p = p
            best_pat_table[cb, ci] = best_p

    # Map back to samples
    cb_sample = _coarse_bucket(sample_fp)
    best_pat = np.array([best_pat_table[cb_sample[i], sample_pred[i]]
                         for i in range(len(X))])

    return best_pat, sample_fp, sample_pred


# ── plotting ─────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for idx, (name, gen_fn) in enumerate(SYNTH_DATASETS):
        print(f"  {name}...", flush=True)
        X, y = gen_fn()
        best_pat, fp_vals, pred_cls = collect_per_sample(X, y)

        ax = axes[idx // 2, idx % 2]

        # Plot each point: color = best pattern, marker = true class
        markers = ["o", "s"]
        for c in [0, 1]:
            mask_c = (y == c)
            for p in range(N_PAT):
                mask = mask_c & (best_pat == p)
                if mask.sum() == 0:
                    continue
                ax.scatter(X[mask, 0], X[mask, 1],
                           c=PAT_COLORS[p], marker=markers[c],
                           s=18, alpha=0.6, edgecolors="white",
                           linewidths=0.3, zorder=2)

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    # Shared legend on the right side
    pat_handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=PAT_COLORS[p], markersize=8,
                          label=PATTERNS[p])
                   for p in range(N_PAT)]
    class_handles = [Line2D([0], [0], marker=m, color="w",
                            markerfacecolor="gray", markersize=8,
                            label=f"Class {c}")
                     for c, m in enumerate(["o", "s"])]
    fig.legend(handles=pat_handles + class_handles,
               loc="center right", ncol=1, fontsize=9,
               title="Best pattern in\n(region, pred class) cell",
               title_fontsize=9, framealpha=0.95,
               bbox_to_anchor=(1.01, 0.5))

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    out_path = (Path(__file__).resolve().parent.parent
                / "paper" / "fig_perpoint_best.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
