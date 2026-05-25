"""common.py — shared utilities for CPFW experiments."""

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_cache"


class Dataset:
    """Minimal dataset container compatible with data_cache pickle format."""
    def __init__(self, X, y):
        self.X = X
        self.y = y

DATASETS = [
    # 17 design (method tuned here)
    "hepatitis", "wine", "heart-disease", "parkinsons", "sonar",
    "heart-statlog", "haberman", "spect", "ionosphere", "australian",
    "breast-w", "transfusion", "diabetes", "wdbc", "banknote",
    "german-credit", "tic-tac-toe",
    # 13 OOD (held out from method design)
    "spambase", "eeg-eye-state", "magic-gamma", "yeast-me2", "oil",
    "phoneme", "kr-vs-kp", "jm1", "electricity", "adult",
    "musk", "nomao", "bank-marketing",
    # 6 OOD added in v1.4 expansion (hard UCI, RF-weak boundary-rich)
    "breast-cancer-uci", "mammographic-mass", "wine-quality-red",
    "wine-quality-white", "thoracic-surgery", "default-credit-card",
]

PATTERNS = ["noflip", "early_sw", "late_sw", "oscillat", "recover", "other"]
N_PAT = len(PATTERNS)


def load(name):
    with open(DATA_DIR / f"{name}.pkl", "rb") as f:
        d = pickle.load(f)
    return np.asarray(d.X, dtype=float), np.asarray(d.y)


def recalls(y_true, y_pred, minority, majority):
    rmin = recall_score(y_true, y_pred, labels=[minority],
                        average="macro", zero_division=0)
    rmaj = recall_score(y_true, y_pred, labels=[majority],
                        average="macro", zero_division=0)
    return rmin, rmaj


def classify_pattern(labels):
    """Classify flip pattern of a decision path.

    labels: list[int] — node majority-class at each step from root to leaf.

    Returns index into PATTERNS:
        0 noflip      no class change root→leaf
        1 early_sw    first flip in first 1/3, stable after 2/3
        2 late_sw     first flip after 2/3 of path
        3 oscillat    ≥2 direction reversals
        4 recover     1 reversal, stabilises before 2/3
        5 other       doesn't fit above
    """
    L = len(labels)
    slots = L - 1
    if slots < 1:
        return 0
    flips = [k for k in range(1, L) if labels[k] != labels[k - 1]]
    if not flips:
        return 0
    pos = ([(k - 1) / (slots - 1) for k in flips]
           if slots >= 2 else [0.5] * len(flips))
    first_p, last_p = pos[0], pos[-1]
    dirs = [labels[flips[j]] - labels[flips[j] - 1] for j in range(len(flips))]
    n_rev = sum(1 for j in range(1, len(dirs)) if dirs[j] != dirs[j - 1])
    if n_rev >= 2:
        return 3
    if n_rev == 1:
        return 4 if last_p < 2/3 else 3
    if first_p < 1/3 and last_p < 2/3:
        return 1
    if first_p > 2/3:
        return 2
    return 5


def classify_patterns_batch(labels_mat, lengths):
    """Vectorized pattern classification for a batch of decision paths.

    labels_mat : (n, max_depth) int8 array, padded with -1
    lengths    : (n,) int array — actual path length per sample

    Returns (n,) int array of pattern indices (same encoding as classify_pattern).

    Key identity used: for binary labels, consecutive flips always alternate
    direction, so n_rev = max(0, n_flips - 1).
    """
    n, max_len = labels_mat.shape
    if max_len < 2:
        return np.zeros(n, dtype=np.int8)

    valid_pair = (labels_mat[:, :-1] >= 0) & (labels_mat[:, 1:] >= 0)
    flips = (labels_mat[:, 1:] != labels_mat[:, :-1]) & valid_pair  # (n, max_len-1)

    n_flips = flips.sum(axis=1)
    has_flip = n_flips > 0
    n_rev = np.maximum(0, n_flips - 1)  # binary label identity

    slots = lengths - 1  # number of transitions
    # Normalised position denominator: slots-1 when slots>=2, else 1 (fallback)
    safe_denom = np.where(slots >= 2, slots - 1, 1).astype(np.float64)

    first_flip_col = np.argmax(flips, axis=1)                            # (n,)
    last_flip_col  = (max_len - 2) - np.argmax(flips[:, ::-1], axis=1)  # (n,)

    # slots==1: original code uses pos=0.5 (avoids division by zero)
    single_slot = has_flip & (slots < 2)
    first_pos = np.where(
        has_flip & ~single_slot, first_flip_col / safe_denom,
        np.where(single_slot, 0.5, 0.0))
    last_pos  = np.where(
        has_flip & ~single_slot, last_flip_col  / safe_denom,
        np.where(single_slot, 0.5, 0.0))

    pattern = np.zeros(n, dtype=np.int8)
    pattern[has_flip & (n_rev == 0) & (first_pos < 1/3) & (last_pos < 2/3)] = 1  # early_sw
    pattern[has_flip & (n_rev == 0) & (first_pos > 2/3)]                          = 2  # late_sw
    pattern[has_flip & ((n_rev >= 2) | ((n_rev == 1) & (last_pos >= 2/3)))]       = 3  # oscillat
    pattern[has_flip & (n_rev == 1) & (last_pos < 2/3)]                           = 4  # recover
    # other: has_flip & n_rev==0 & not early_sw & not late_sw
    is_other = (has_flip & (n_rev == 0)
                & ~((first_pos < 1/3) & (last_pos < 2/3))
                & ~(first_pos > 2/3))
    pattern[is_other] = 5
    return pattern


def _enumerate_leaf_label_paths(estimator):
    """DFS over the tree, returning (leaf_ids, labels_mat, lengths).

    labels_mat : (n_leaves, max_depth) int8 padded with -1
    lengths    : (n_leaves,) int — actual path length per leaf
    """
    t = estimator.tree_
    cl, cr = t.children_left, t.children_right
    nlab = np.argmax(t.value[:, 0, :], axis=1)

    leaf_ids, leaf_labels = [], []
    stack = [(0, [int(nlab[0])])]
    while stack:
        node, labels = stack.pop()
        if cl[node] == cr[node]:                         # leaf
            leaf_ids.append(node)
            leaf_labels.append(labels)
        else:
            stack.append((cr[node], labels + [int(nlab[cr[node]])]))
            stack.append((cl[node], labels + [int(nlab[cl[node]])]))

    n = len(leaf_ids)
    max_len = max(len(l) for l in leaf_labels)
    mat = np.full((n, max_len), -1, dtype=np.int8)
    lens = np.array([len(l) for l in leaf_labels])
    for i, l in enumerate(leaf_labels):
        mat[i, :len(l)] = l
    return np.array(leaf_ids), mat, lens


def precompute_leaf_patterns(estimator):
    """Pre-compute flip pattern index for every leaf in a fitted tree.

    Returns an array of shape (n_nodes,) where result[leaf_id] gives the
    pattern index for that leaf.  Only leaf entries are meaningful.
    """
    leaf_ids, mat, lens = _enumerate_leaf_label_paths(estimator)
    pats = classify_patterns_batch(mat, lens)
    result = np.zeros(estimator.tree_.node_count, dtype=np.int8)
    result[leaf_ids] = pats
    return result


def precompute_leaf_flip_rate(estimator):
    """Pre-compute flip rate per leaf (= k/d for k flips over depth-d path).

    Returns an array of shape (n_nodes,) where result[leaf_id] gives the
    flip rate for that leaf's root-to-leaf path. Only leaf entries are
    meaningful.
    """
    leaf_ids, mat, lens = _enumerate_leaf_label_paths(estimator)
    valid = (mat[:, :-1] >= 0) & (mat[:, 1:] >= 0)
    flips = ((mat[:, 1:] != mat[:, :-1]) & valid).sum(axis=1)
    slots = np.maximum(lens - 1, 1)  # avoid div-by-zero for trivial paths
    rates = flips / slots
    result = np.zeros(estimator.tree_.node_count, dtype=np.float64)
    result[leaf_ids] = rates
    return result


def walk_tree(estimator, X):
    """Yield (labels, leaf_value) for each sample in X through one tree.

    labels : list[int] — majority-class label at each node from root to leaf
    leaf_value : np.ndarray — unnormalised class counts at leaf

    Kept for legacy callers in archive/exp_*; new code uses
    `walk_tree_batch` which is the vectorized equivalent and used
    across all paper-related scripts.
    """
    t = estimator.tree_
    cl, cr = t.children_left, t.children_right
    nlab = np.argmax(t.value[:, 0, :], axis=1)
    for xi in X:
        node = 0
        labels = [int(nlab[node])]
        while cl[node] != cr[node]:
            node = cl[node] if xi[t.feature[node]] <= t.threshold[node] else cr[node]
            labels.append(int(nlab[node]))
        yield labels, t.value[node, 0, :]


def walk_tree_batch(estimator, X):
    """Vectorized counterpart of `walk_tree`.

    Returns the per-sample (leaf_id, pattern_idx, leaf_value, pred_cls)
    for all samples in X through one tree, with no Python-level loop
    over samples.

        leaf_ids : (n,) int       — leaf node index per sample
        leaf_pat : (n,) int8      — flip pattern index per sample's leaf
        leaf_val : (n, n_cls)     — unnormalised class counts at leaf
        pred_cls : (n,) class lbl — argmax class at leaf
    """
    t = estimator.tree_
    leaf_ids = estimator.apply(X)
    leaf_pat_table = precompute_leaf_patterns(estimator)
    leaf_pat = leaf_pat_table[leaf_ids]
    leaf_val = t.value[leaf_ids, 0, :]
    pred_idx = np.argmax(leaf_val, axis=1)
    pred_cls = estimator.classes_[pred_idx]
    return leaf_ids, leaf_pat, leaf_val, pred_cls


# ── CPFW core (vectorized, shared by step6b and compare_baselines) ───
from sklearn.ensemble import RandomForestClassifier as _RFC
from sklearn.model_selection import StratifiedKFold as _SKF

CPFW_N_PROB = 10  # 10 buckets of width 0.10 over [0, 1]
CPFW_N_CLS = 2


def cpfw_bucket_fp(fp):
    """Bucket per-tree forest probability `fp ∈ [0,1]` into 10 buckets."""
    return np.minimum(CPFW_N_PROB - 1,
                      (np.asarray(fp) * CPFW_N_PROB).astype(int))


def cpfw_collect_table(X_tr, y_tr, minority, seed,
                       n_estimators=300, n_cv=5):
    """Collect per-cell (correct, total) counts via K-fold CV on the
    training fold. Returns R with shape (N_PROB, N_PAT, N_CLS, 2).

    All inner operations are vectorized; no per-sample Python loop.
    """
    skf = _SKF(n_splits=n_cv, shuffle=True, random_state=seed)
    R = np.zeros((CPFW_N_PROB, N_PAT, CPFW_N_CLS, 2))
    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        rf = _RFC(n_estimators=n_estimators,
                  max_features="sqrt", bootstrap=True,
                  random_state=seed, n_jobs=1
                  ).fit(X_tr[tr_idx], y_tr[tr_idx])
        classes = rf.classes_
        X_val, y_val = X_tr[val_idx], y_tr[val_idx]
        n_val = len(val_idx)
        forest_proba = rf.predict_proba(X_val)
        for est in rf.estimators_:
            _, leaf_pat, leaf_val, pred_cls = walk_tree_batch(est, X_val)
            pred_idx = np.argmax(leaf_val, axis=1)
            fp = forest_proba[np.arange(n_val), pred_idx]
            pb = cpfw_bucket_fp(fp)
            ci = (pred_cls == minority).astype(int)
            cor = (pred_cls == y_val).astype(np.float64)
            np.add.at(R[:, :, :, 0], (pb, leaf_pat, ci), cor)
            np.add.at(R[:, :, :, 1], (pb, leaf_pat, ci), 1.0)
    return R


def cpfw_build_weight_table(R, min_n=30):
    """Build weight table W[pb, pat, ci] = P(corr|pb,pat,ci) / P(corr|pb,ci).

    Cells with fewer than `min_n` observations fall back to w = 1.0.
    """
    W = np.ones((CPFW_N_PROB, N_PAT, CPFW_N_CLS))
    for pb in range(CPFW_N_PROB):
        for ci in range(CPFW_N_CLS):
            marg = R[pb, :, ci, :].sum(axis=0)
            p_marg = marg[0] / marg[1] if marg[1] >= min_n else None
            for pat in range(N_PAT):
                v = R[pb, pat, ci]
                if v[1] >= min_n and p_marg and p_marg > 0:
                    W[pb, pat, ci] = (v[0] / v[1]) / p_marg
    return W


def cpfw_predict_proba(rf, X_te, minority, W):
    """Vectorized weighted prediction. Returns (n_te, n_cls) probabilities."""
    classes = rf.classes_
    n_cls = len(classes)
    n_te = len(X_te)
    psum = np.zeros((n_te, n_cls))
    wsum = np.zeros(n_te)
    forest_proba = rf.predict_proba(X_te)
    for est in rf.estimators_:
        _, leaf_pat, leaf_val, pred_cls = walk_tree_batch(est, X_te)
        lv_norm = leaf_val / leaf_val.sum(axis=1, keepdims=True)
        pred_idx = np.argmax(leaf_val, axis=1)
        fp = forest_proba[np.arange(n_te), pred_idx]
        pb = cpfw_bucket_fp(fp)
        ci = (pred_cls == minority).astype(int)
        w = W[pb, leaf_pat, ci]
        psum += w[:, np.newaxis] * lv_norm
        wsum += w
    safe = np.where(wsum > 0, wsum, 1.0)
    return psum / safe[:, np.newaxis]
