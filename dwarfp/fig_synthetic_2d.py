"""fig_synthetic_2d.py — 2D synthetic visualisation (4 datasets).

Rows: diagonal / moons / circles / overlap
Cols: confidence heatmap (+ class labels) | pattern accuracy pred=0 | pred=1

Bar charts include RF marginal accuracy as dashed reference line.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
GRID_RES = 200

PB_LABELS = ["[.0\n.2)", "[.2\n.4)", "[.4\n.6)", "[.6\n.8)", "[.8\n1.]"]
PAT_COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12", "#95a5a6"]
# Scatter point colors: deep variants of the heatmap blue/green endpoints,
# so class 0 dots blend into the blue heatmap region (darker) and class 1
# dots into the green region (darker).
DATA_COLORS = ["#0e3a8a", "#0a5f33"]    # deep blue (class 0), deep green (class 1)
# Bar-chart borders + very-light facecolors to match predicted-class context
BAR_BORDER_COLORS = ["#0e3a8a", "#0a5f33"]   # pred=Class 0 (blue), pred=Class 1 (green)
BAR_FACE_COLORS   = ["#eaf0fb", "#eaf5ee"]   # very light blue / very light green

# Heatmap colormap for P̂_RF[class 1] in [0,1] — light/pastel so points pop out:
#   0 = light blue (class 0 region), 1 = light green (class 1 region),
#   0.5 = light red (boundary), 0.25/0.75 = light yellow (intermediate).
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "blue_yellow_red_yellow_green_light",
    [(0.00, "#a8bce8"),   # light blue
     (0.25, "#fbe89e"),   # light yellow
     (0.50, "#e9a5a0"),   # light red
     (0.75, "#fbe89e"),   # light yellow
     (1.00, "#a8d4ba")],  # light green
)


# ── synthetic datasets ──────────────────────────────────────────────

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
    ("Moons",    lambda: make_moons(n_samples=N_SAMPLES, noise=0.30, random_state=SEED)),
    ("Circles",  lambda: make_circles(n_samples=N_SAMPLES, noise=0.15,
                                       factor=0.4, random_state=SEED)),
    ("Overlap",  lambda: make_overlap(N_SAMPLES, SEED)),
]


# ── helpers ─────────────────────────────────────────────────────────

def _coarse_bucket(fp):
    return np.minimum(4, (fp * 5).astype(int))   # per-tree fp in [0,1], 0.20 wide


def collect_records(X, y):
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    fp_list, pat_list, ci_list, cor_list = [], [], [], []

    for tr_idx, val_idx in skf.split(X, y):
        rf_cv = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_features="sqrt",
            bootstrap=True, random_state=SEED, n_jobs=1
        ).fit(X[tr_idx], y[tr_idx])
        classes = rf_cv.classes_
        X_val, y_val = X[val_idx], y[val_idx]
        forest_proba = rf_cv.predict_proba(X_val)

        for est in rf_cv.estimators_:
            leaf_pat = precompute_leaf_patterns(est)
            leaf_ids = est.apply(X_val)
            t = est.tree_
            lv = t.value[leaf_ids, 0, :]
            pred_idx = np.argmax(lv, axis=1)
            pred_cls = classes[pred_idx]

            fp_list.append(forest_proba[np.arange(len(X_val)), pred_idx])
            pat_list.append(leaf_pat[leaf_ids])
            ci_list.append(pred_cls.astype(int))
            cor_list.append((pred_cls == y_val).astype(float))

    return (np.concatenate(fp_list), np.concatenate(pat_list).astype(int),
            np.concatenate(ci_list), np.concatenate(cor_list))


def build_tables(fp_all, pat_all, ci_all, cor_all):
    """Returns (acc_table, cnt_table, marginal_acc) — all shape (5, ..., 2)."""
    cb_all = _coarse_bucket(fp_all)
    acc = np.full((5, N_PAT, 2), np.nan)
    cnt = np.zeros((5, N_PAT, 2), dtype=int)
    marg_acc = np.full((5, 2), np.nan)  # RF baseline: P(correct | pb, ci)

    for cpb in range(5):
        for ci in range(2):
            m_cell = (cb_all == cpb) & (ci_all == ci)
            n_cell = m_cell.sum()
            if n_cell >= MIN_N:
                marg_acc[cpb, ci] = cor_all[m_cell].mean()
            for pat in range(N_PAT):
                m = m_cell & (pat_all == pat)
                n = m.sum()
                cnt[cpb, pat, ci] = n
                if n >= MIN_N:
                    acc[cpb, pat, ci] = cor_all[m].mean()

    return acc, cnt, marg_acc


# ── plotting ────────────────────────────────────────────────────────

def plot_row(fig, gs, row_idx, name, X, y, acc, cnt, marg_acc):
    n_rows = 4

    # --- col 0: confidence heatmap with class labels ---
    ax_map = fig.add_subplot(gs[row_idx, 0])
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_features="sqrt",
                                bootstrap=True, random_state=SEED).fit(X, y)
    margin = 0.6
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, GRID_RES),
                         np.linspace(y_min, y_max, GRID_RES))
    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = rf.predict_proba(grid)
    p1 = proba[:, 1].reshape(xx.shape)   # P̂_RF[class 1] in [0,1]

    im = ax_map.contourf(xx, yy, p1, levels=np.linspace(0.0, 1.0, 41),
                         cmap=HEAT_CMAP, alpha=0.70)

    # data points — deep variants of the heatmap blue/green to stand out
    markers = ["o", "s"]
    for c in [0, 1]:
        m = y == c
        ax_map.scatter(X[m, 0], X[m, 1], c=DATA_COLORS[c], s=14, alpha=0.85,
                       edgecolors="white", linewidths=0.3, marker=markers[c],
                       label=f"Class {c}", zorder=3)

    ax_map.set_ylabel(f"{name}\n$x_2$", fontsize=10, fontweight="bold")
    if row_idx == 0:
        ax_map.set_title(r"$\hat{P}_{\mathrm{RF}}[\mathrm{class}\,1]$"
                         " + class labels", fontsize=10)
        ax_map.legend(loc="upper right", fontsize=7, markerscale=1.5,
                      framealpha=0.9)
    if row_idx == n_rows - 1:
        ax_map.set_xlabel("$x_1$")

    # --- col 1-2: pattern accuracy bars with RF reference ---
    class_labels = ["pred = Class 0", "pred = Class 1"]
    ax_first_bar = None
    for ci in range(2):
        ax = fig.add_subplot(gs[row_idx, 1 + ci])
        # Border + very-light background colored by predicted class:
        #   Class 0 = blue, Class 1 = green
        ax.set_facecolor(BAR_FACE_COLORS[ci])
        for spine in ax.spines.values():
            spine.set_edgecolor(BAR_BORDER_COLORS[ci])
            spine.set_linewidth(2.0)
        if ci == 0:
            ax_first_bar = ax
        x_pos = np.arange(5)
        width = 0.12
        offsets = np.arange(N_PAT) - (N_PAT - 1) / 2

        for p in range(N_PAT):
            vals = []
            for cpb in range(5):
                v = acc[cpb, p, ci]
                vals.append(v if not np.isnan(v) else 0)
            ax.bar(x_pos + offsets[p] * width, vals, width,
                   label=PATTERNS[p] if (row_idx == 0 and ci == 0) else "",
                   color=PAT_COLORS[p], alpha=0.85)
            # x mark for insufficient data
            for k in range(5):
                if cnt[k, p, ci] < MIN_N:
                    ax.scatter(x_pos[k] + offsets[p] * width,
                               0.02, marker="x", c="red", s=15, zorder=5)

        # RF marginal accuracy as reference line (step function)
        rf_vals = []
        for cpb in range(5):
            v = marg_acc[cpb, ci]
            rf_vals.append(v if not np.isnan(v) else 0)
        ax.plot(x_pos, rf_vals, color="black", ls="--", lw=1.5, marker="d",
                ms=4, zorder=6, label="RF avg" if (row_idx == 0 and ci == 0) else "")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(PB_LABELS if row_idx == n_rows - 1 else [""] * 5,
                           fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", ls="--", lw=0.5, alpha=0.3)
        if row_idx == 0:
            ax.set_title(class_labels[ci], fontsize=10)
        if ci == 0:
            ax.set_ylabel("Tree accuracy")
        if row_idx == n_rows - 1:
            ax.set_xlabel("Forest confidence region")

    return im, ax_first_bar  # return pred=Class 0 bar axis (has labels)


def main():
    fig = plt.figure(figsize=(16, 17), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, width_ratios=[1.3, 1, 1])

    last_im = None
    first_bar_ax = None
    for i, (name, gen_fn) in enumerate(SYNTH_DATASETS):
        print(f"  {name}...", flush=True)
        X, y = gen_fn()
        fp_all, pat_all, ci_all, cor_all = collect_records(X, y)
        acc, cnt, marg_acc = build_tables(fp_all, pat_all, ci_all, cor_all)
        last_im, bar_ax = plot_row(fig, gs, i, name, X, y, acc, cnt, marg_acc)
        if i == 0:
            first_bar_ax = bar_ax

    # shared legend — attach to existing first-row bar chart axis
    handles, labels = first_bar_ax.get_legend_handles_labels()
    if handles:
        first_bar_ax.legend(handles, labels, fontsize=7, loc="upper left",
                            ncol=2, title="Pattern type", title_fontsize=8,
                            framealpha=0.95)

    # colorbar at bottom of heatmap column
    cbar_ax = fig.add_axes([0.03, 0.01, 0.28, 0.010])
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal",
                        label=r"$\hat{P}_{\mathrm{RF}}[\mathrm{class}\,1]$"
                              "  (0 = class 0,  0.5 = boundary,  1 = class 1)")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    out_path = Path(__file__).resolve().parent.parent / "paper" / "fig_synthetic_2d.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
