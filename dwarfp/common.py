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
    # original 17
    "hepatitis", "wine", "heart-disease", "parkinsons", "sonar",
    "heart-statlog", "haberman", "spect", "ionosphere", "australian",
    "breast-w", "transfusion", "diabetes", "wdbc", "banknote",
    "german-credit", "tic-tac-toe",
    # added 5 (first expansion)
    "spambase", "eeg-eye-state", "magic-gamma", "yeast-me2", "oil",
    # added 8 (second expansion, medium-large)
    "phoneme", "kr-vs-kp", "jm1", "electricity", "adult",
    "musk", "nomao", "bank-marketing",
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


def precompute_leaf_patterns(estimator):
    """Pre-compute flip pattern index for every leaf in a fitted tree.

    Returns an array of shape (n_nodes,) where result[leaf_id] gives the
    pattern index for that leaf.  Only leaf entries are meaningful.

    Cost: one DFS per tree — O(n_nodes × avg_depth), negligible vs sample work.
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

    # batch classify all leaves at once
    n = len(leaf_ids)
    max_len = max(len(l) for l in leaf_labels)
    mat = np.full((n, max_len), -1, dtype=np.int8)
    lens = np.array([len(l) for l in leaf_labels])
    for i, l in enumerate(leaf_labels):
        mat[i, :len(l)] = l
    pats = classify_patterns_batch(mat, lens)

    result = np.zeros(t.node_count, dtype=np.int8)
    result[np.array(leaf_ids)] = pats
    return result


def walk_tree(estimator, X):
    """Yield (labels, leaf_value) for each sample in X through one tree.

    labels : list[int] — majority-class label at each node from root to leaf
    leaf_value : np.ndarray — unnormalised class counts at leaf
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
