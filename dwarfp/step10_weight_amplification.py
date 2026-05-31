"""step10_weight_amplification.py — CV-honest K selection for weight amplification.

K is selected WITHIN the same 5-fold CV that builds the weight table:
  1. 5-fold CV on training data: accumulate R (as usual) AND cache
     per-fold validation-set per-tree intermediates (pb, pat, ci, lv_norm, y_val).
  2. Build W from R.
  3. For each K candidate, replay cached CV predictions with amplified W
     → compute CV accuracy.  Pick K* = argmax.
  4. Apply W with K* on test set.

This makes K a data-driven per-dataset parameter estimated without
test-data leakage.  If CV-selected K consistently improves over K=0,
the amplification is a validated technique, not a post-hoc observation.

Outputs: results_weight_amplification.csv
"""

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(PROJECT_ROOT) / "archive"))
from dwarfp.common import (
    load, recalls, walk_tree_batch, N_PAT, DATASETS,
    cpfw_build_weight_table,
    CPFW_N_PROB, CPFW_N_CLS, cpfw_bucket_fp,
)

warnings.filterwarnings("ignore")

N_ESTIMATORS = 300
REPEATS = 30
TEST_SIZE = 0.3
SEED = 42
MIN_N = 30
N_CV = 5

K_VALUES = [0, 10, 20, 30]


# ── OOB helpers ──────────────────────────────────────────────────────

def _oob_indices_for_tree(est, n_train):
    rng = np.random.RandomState(est.random_state)
    sample_indices = rng.randint(0, n_train, n_train)
    in_bag = np.zeros(n_train, dtype=bool)
    in_bag[sample_indices] = True
    return np.where(~in_bag)[0]


def _compute_M(rf):
    oob_proba = rf.oob_decision_function_
    valid = ~np.isnan(oob_proba).any(axis=1)
    sample_fp = oob_proba[valid].max(axis=1)
    boundary = (sample_fp >= 0.4) & (sample_fp < 0.6)
    return float(boundary.sum() / len(sample_fp)) if len(sample_fp) > 0 else 0.0


def _compute_S(rf, X_tr, y_tr, minority):
    n_train = len(X_tr)
    oob_proba = rf.oob_decision_function_
    valid = ~np.isnan(oob_proba).any(axis=1)
    sample_fp = np.full(n_train, np.nan)
    sample_fp[valid] = oob_proba[valid].max(axis=1)
    boundary_mask = (sample_fp >= 0.4) & (sample_fp < 0.6)

    R = np.zeros((2, N_PAT, 2))
    for est in rf.estimators_:
        oob_idx = _oob_indices_for_tree(est, n_train)
        if len(oob_idx) == 0:
            continue
        oob_b = oob_idx[boundary_mask[oob_idx]]
        if len(oob_b) == 0:
            continue
        _, leaf_pat, _, pred_cls = walk_tree_batch(est, X_tr[oob_b])
        ci = (pred_cls == minority).astype(int)
        cor = (pred_cls == y_tr[oob_b]).astype(np.float64)
        np.add.at(R[:, :, 0], (ci, leaf_pat), cor)
        np.add.at(R[:, :, 1], (ci, leaf_pat), 1.0)

    spreads = []
    for ci in range(2):
        accs = []
        for pat in range(N_PAT):
            if R[ci, pat, 1] >= 1:
                accs.append(R[ci, pat, 0] / R[ci, pat, 1])
        if len(accs) >= 2:
            spreads.append(max(accs) - min(accs))
    return float(np.mean(spreads)) if spreads else 0.0


def _amplify_weight_table(W, alpha):
    return np.maximum(1.0 + alpha * (W - 1.0), 0.01)


# ── CV with cached intermediates ─────────────────────────────────────

def _collect_table_with_cache(X_tr, y_tr, minority, seed):
    """5-fold CV: build R AND cache per-fold per-tree intermediates."""
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=seed)
    R = np.zeros((CPFW_N_PROB, N_PAT, CPFW_N_CLS, 2))
    fold_caches = []  # list of (tree_cache_list, y_val, n_val, classes)

    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_features="sqrt",
            bootstrap=True, random_state=seed, n_jobs=1
        ).fit(X_tr[tr_idx], y_tr[tr_idx])
        classes = rf.classes_
        X_val, y_val = X_tr[val_idx], y_tr[val_idx]
        n_val = len(val_idx)
        forest_proba = rf.predict_proba(X_val)

        tree_cache = []
        for est in rf.estimators_:
            _, leaf_pat, leaf_val, pred_cls = walk_tree_batch(est, X_val)
            lv_norm = leaf_val / leaf_val.sum(axis=1, keepdims=True)
            pred_idx = np.argmax(leaf_val, axis=1)
            fp = forest_proba[np.arange(n_val), pred_idx]
            pb = cpfw_bucket_fp(fp)
            ci = (pred_cls == minority).astype(int)
            cor = (pred_cls == y_val).astype(np.float64)

            # Accumulate R
            np.add.at(R[:, :, :, 0], (pb, leaf_pat, ci), cor)
            np.add.at(R[:, :, :, 1], (pb, leaf_pat, ci), 1.0)

            tree_cache.append({"pb": pb, "pat": leaf_pat, "ci": ci,
                               "lv_norm": lv_norm})

        fold_caches.append((tree_cache, y_val, n_val, classes))

    return R, fold_caches


def _select_K_from_cv(fold_caches, W, M, S):
    """Sweep K on cached CV validation predictions, return best K."""
    best_K = 0
    best_acc = -1.0
    for K in K_VALUES:
        alpha = 1.0 + K * M * S
        W_amp = _amplify_weight_table(W, alpha)
        total_correct = 0
        total_n = 0
        for tree_cache, y_val, n_val, classes in fold_caches:
            n_cls = len(classes)
            psum = np.zeros((n_val, n_cls))
            wsum = np.zeros(n_val)
            for tc in tree_cache:
                w = W_amp[tc["pb"], tc["pat"], tc["ci"]]
                psum += w[:, np.newaxis] * tc["lv_norm"]
                wsum += w
            safe = np.where(wsum > 0, wsum, 1.0)
            proba = psum / safe[:, np.newaxis]
            pred = classes[np.argmax(proba, axis=1)]
            total_correct += int((pred == y_val).sum())
            total_n += n_val
        cv_acc = total_correct / total_n
        if cv_acc > best_acc:
            best_acc = cv_acc
            best_K = K
    return best_K, best_acc


# ── Main run ─────────────────────────────────────────────────────────

def _run_one(name, rep):
    X, y = load(name)
    cls, cnt = np.unique(y, return_counts=True)
    minority = int(cls[np.argmin(cnt)])
    majority = int(cls[np.argmax(cnt)])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=SEED + rep)
    (tr, te), = sss.split(X, y)
    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    # Outer RF for OOB indicators + test prediction
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                max_features="sqrt", bootstrap=True,
                                oob_score=True,
                                random_state=SEED + rep, n_jobs=1
                                ).fit(X_tr, y_tr)
    classes = rf.classes_
    n_cls = len(classes)
    n_te = len(X_te)

    # RF baseline
    rf_pred = rf.predict(X_te)
    rf_acc = float(accuracy_score(y_te, rf_pred))
    rf_rmin, rf_rmaj = recalls(y_te, rf_pred, minority, majority)

    # OOB indicators
    M = _compute_M(rf)
    S = _compute_S(rf, X_tr, y_tr, minority)

    # CV: build W + cache for K selection
    R, fold_caches = _collect_table_with_cache(X_tr, y_tr, minority,
                                                seed=SEED + rep)
    W = cpfw_build_weight_table(R, min_n=MIN_N)

    # CV-honest K selection
    K_star, cv_acc = _select_K_from_cv(fold_caches, W, M, S)

    # Cache test-set per-tree intermediates (one pass)
    forest_proba = rf.predict_proba(X_te)
    test_cache = []
    for est in rf.estimators_:
        _, leaf_pat, leaf_val, pred_cls = walk_tree_batch(est, X_te)
        lv_norm = leaf_val / leaf_val.sum(axis=1, keepdims=True)
        pred_idx = np.argmax(leaf_val, axis=1)
        fp = forest_proba[np.arange(n_te), pred_idx]
        pb = cpfw_bucket_fp(fp)
        ci = (pred_cls == minority).astype(int)
        test_cache.append({"pb": pb, "pat": leaf_pat, "ci": ci,
                           "lv_norm": lv_norm})

    # Predict test with all K values (for comparison) + K*
    row = {
        "dataset": name, "rep": rep,
        "M": M, "S": S, "MS": M * S,
        "K_star": K_star,
        "rf_acc": rf_acc, "rf_rmin": rf_rmin, "rf_rmaj": rf_rmaj,
    }

    for K in K_VALUES:
        alpha = 1.0 + K * M * S
        W_amp = _amplify_weight_table(W, alpha)
        psum = np.zeros((n_te, n_cls))
        wsum = np.zeros(n_te)
        for tc in test_cache:
            w = W_amp[tc["pb"], tc["pat"], tc["ci"]]
            psum += w[:, np.newaxis] * tc["lv_norm"]
            wsum += w
        safe = np.where(wsum > 0, wsum, 1.0)
        proba = psum / safe[:, np.newaxis]
        pred = classes[np.argmax(proba, axis=1)]
        acc = float(accuracy_score(y_te, pred))
        rmin, rmaj = recalls(y_te, pred, minority, majority)
        tag = f"K{K}"
        row[f"{tag}_acc"] = acc
        row[f"{tag}_rmin"] = rmin
        row[f"{tag}_rmaj"] = rmaj

    # K* test result
    alpha_star = 1.0 + K_star * M * S
    W_star = _amplify_weight_table(W, alpha_star)
    psum = np.zeros((n_te, n_cls))
    wsum = np.zeros(n_te)
    for tc in test_cache:
        w = W_star[tc["pb"], tc["pat"], tc["ci"]]
        psum += w[:, np.newaxis] * tc["lv_norm"]
        wsum += w
    safe = np.where(wsum > 0, wsum, 1.0)
    proba = psum / safe[:, np.newaxis]
    pred = classes[np.argmax(proba, axis=1)]
    row["Kstar_acc"] = float(accuracy_score(y_te, pred))
    row["Kstar_rmin"], row["Kstar_rmaj"] = recalls(y_te, pred, minority, majority)

    return row


def run():
    print(f"CV-honest K selection: alpha = 1 + K*M*S, K in {K_VALUES}")
    print(f"datasets={len(DATASETS)}  repeats={REPEATS}  "
          f"n_estimators={N_ESTIMATORS}  min_n={MIN_N}\n")

    all_rows = []
    for name in DATASETS:
        res = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_run_one)(name, rep) for rep in range(REPEATS))
        all_rows.extend(res)

        rf_m = np.mean([r["rf_acc"] for r in res])
        k0   = np.mean([r["K0_acc"] for r in res])
        ks   = np.mean([r["Kstar_acc"] for r in res])
        ms   = np.mean([r["MS"] for r in res])
        # K* distribution
        k_counts = {}
        for r in res:
            k = r["K_star"]
            k_counts[k] = k_counts.get(k, 0) + 1
        k_dist = " ".join(f"K{k}:{c}" for k, c in sorted(k_counts.items()))
        print(f"{name:22s}  RF={rf_m:.4f}  K0={k0:.4f}  K*={ks:.4f}"
              f"(d={ks-rf_m:+.4f})  M*S={ms:.4f}  [{k_dist}]",
              flush=True)

    # ── Save CSV ──
    fields = ["dataset", "rep", "M", "S", "MS", "K_star",
              "rf_acc", "rf_rmin", "rf_rmaj",
              "Kstar_acc", "Kstar_rmin", "Kstar_rmaj"]
    for K in K_VALUES:
        tag = f"K{K}"
        fields += [f"{tag}_acc", f"{tag}_rmin", f"{tag}_rmaj"]
    out_path = Path(__file__).resolve().parent / "results_weight_amplification.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nSaved: {out_path}")

    # ── Aggregate ──
    print("\n" + "=" * 80)
    print("AGGREGATE")
    print("=" * 80)

    ds_stats = {}
    for name in DATASETS:
        reps = [r for r in all_rows if r["dataset"] == name]
        d = {"rf_acc": np.mean([r["rf_acc"] for r in reps]),
             "rf_rmin": np.mean([r["rf_rmin"] for r in reps]),
             "rf_rmaj": np.mean([r["rf_rmaj"] for r in reps]),
             "MS": np.mean([r["MS"] for r in reps]),
             "Kstar_acc": np.mean([r["Kstar_acc"] for r in reps]),
             "Kstar_rmin": np.mean([r["Kstar_rmin"] for r in reps]),
             "Kstar_rmaj": np.mean([r["Kstar_rmaj"] for r in reps])}
        for K in K_VALUES:
            tag = f"K{K}"
            d[f"{tag}_acc"]  = np.mean([r[f"{tag}_acc"]  for r in reps])
            d[f"{tag}_rmin"] = np.mean([r[f"{tag}_rmin"] for r in reps])
            d[f"{tag}_rmaj"] = np.mean([r[f"{tag}_rmaj"] for r in reps])
        # K* distribution
        k_counts = {}
        for r in reps:
            k = r["K_star"]
            k_counts[k] = k_counts.get(k, 0) + 1
        d["K_dist"] = k_counts
        ds_stats[name] = d

    def _wilcoxon_safe(x):
        x = x[x != 0]
        if len(x) < 2: return float("nan"), float("nan")
        s, p = wilcoxon(x)
        return s, p

    def _wtl(d):
        w = int(np.sum(d > 0))
        l = int(np.sum(d < 0))
        t = len(d) - w - l
        return w, t, l

    # K* vs RF
    d_acc = np.array([ds_stats[n]["Kstar_acc"] - ds_stats[n]["rf_acc"]
                      for n in DATASETS])
    d_rmin = np.array([ds_stats[n]["Kstar_rmin"] - ds_stats[n]["rf_rmin"]
                       for n in DATASETS])
    d_rmaj = np.array([ds_stats[n]["Kstar_rmaj"] - ds_stats[n]["rf_rmaj"]
                       for n in DATASETS])
    w, t, l = _wtl(d_acc)
    _, p = _wilcoxon_safe(d_acc)
    n_hmin = int(np.sum(d_rmin < -0.002))
    n_hmaj = int(np.sum(d_rmaj < -0.002))
    print(f"\nK* vs RF:  mean={d_acc.mean():+.4f}  W/T/L={w}/{t}/{l}"
          f"  p={p:.4f}  min_regr={n_hmin}/{len(DATASETS)}"
          f"  maj_regr={n_hmaj}/{len(DATASETS)}")

    # K* vs K=0
    d_k0 = np.array([ds_stats[n]["Kstar_acc"] - ds_stats[n]["K0_acc"]
                      for n in DATASETS])
    w, t, l = _wtl(d_k0)
    _, p = _wilcoxon_safe(d_k0)
    print(f"K* vs K=0: mean={d_k0.mean():+.4f}  W/T/L={w}/{t}/{l}"
          f"  p={p:.4f}")

    # Fixed K vs RF for comparison
    print(f"\n--- Fixed K vs RF (for comparison) ---")
    print(f"{'setting':>8} {'mean_d':>10} {'W/T/L':>8} {'p':>10}"
          f" {'min_regr':>9} {'maj_regr':>9}")
    print("-" * 60)
    for K in K_VALUES:
        tag = f"K{K}"
        d = np.array([ds_stats[n][f"{tag}_acc"] - ds_stats[n]["rf_acc"]
                      for n in DATASETS])
        dr_min = np.array([ds_stats[n][f"{tag}_rmin"] - ds_stats[n]["rf_rmin"]
                           for n in DATASETS])
        dr_maj = np.array([ds_stats[n][f"{tag}_rmaj"] - ds_stats[n]["rf_rmaj"]
                           for n in DATASETS])
        w, t, l = _wtl(d)
        _, p = _wilcoxon_safe(d)
        hmin = int(np.sum(dr_min < -0.002))
        hmaj = int(np.sum(dr_maj < -0.002))
        print(f"{'K='+str(K):>8} {d.mean():>+10.4f} {w:>2}/{t:>2}/{l:>1}"
              f" {p:>10.4f} {hmin:>5}/{len(DATASETS)}"
              f" {hmaj:>5}/{len(DATASETS)}")
    # K* row
    d = np.array([ds_stats[n]["Kstar_acc"] - ds_stats[n]["rf_acc"]
                  for n in DATASETS])
    dr_min = np.array([ds_stats[n]["Kstar_rmin"] - ds_stats[n]["rf_rmin"]
                       for n in DATASETS])
    dr_maj = np.array([ds_stats[n]["Kstar_rmaj"] - ds_stats[n]["rf_rmaj"]
                       for n in DATASETS])
    w, t, l = _wtl(d)
    _, p = _wilcoxon_safe(d)
    hmin = int(np.sum(dr_min < -0.002))
    hmaj = int(np.sum(dr_maj < -0.002))
    print(f"{'K*':>8} {d.mean():>+10.4f} {w:>2}/{t:>2}/{l:>1}"
          f" {p:>10.4f} {hmin:>5}/{len(DATASETS)}"
          f" {hmaj:>5}/{len(DATASETS)}")

    # K* distribution across all datasets
    print(f"\n--- K* selection distribution (across all 36 x 30 = 1080 runs) ---")
    total_counts = {}
    for name in DATASETS:
        for k, c in ds_stats[name]["K_dist"].items():
            total_counts[k] = total_counts.get(k, 0) + c
    for k in sorted(total_counts):
        pct = total_counts[k] / (len(DATASETS) * REPEATS) * 100
        print(f"  K={k:>2}: {total_counts[k]:>4} ({pct:.1f}%)")

    # Top M*S datasets
    top7 = sorted(DATASETS, key=lambda n: -ds_stats[n]["MS"])[:7]
    print(f"\n--- Top 7 M*S datasets ---")
    print(f"{'dataset':22s} {'M*S':>6} {'RF':>7} {'K0':>7} {'K*':>7}"
          f" {'dK*':>7} {'K* dist':>20}")
    print("-" * 85)
    for name in top7:
        d = ds_stats[name]
        k_dist = " ".join(f"{k}:{c}" for k, c in
                          sorted(d["K_dist"].items()))
        print(f"{name:22s} {d['MS']:.4f} {d['rf_acc']:.4f} "
              f"{d['K0_acc']:.4f} {d['Kstar_acc']:.4f}"
              f" {d['Kstar_acc']-d['rf_acc']:+.4f} {k_dist:>20}")


if __name__ == "__main__":
    run()
