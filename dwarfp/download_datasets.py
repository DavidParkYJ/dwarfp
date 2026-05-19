"""download_datasets.py — Fetch 8 additional medium-large binary datasets.

Target: n >= 2000, minority% >= 10%, binary classification.
Source: OpenML via sklearn.datasets.fetch_openml.

Saves to data_cache/ in the same pickle format as existing datasets.
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dwarfp.common import Dataset

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_cache"

TARGETS = [
    # (save_name,  openml_name_or_id,  version,  subsample_n)
    ("phoneme",       "phoneme",        1,        None),    # 5.4K, 5 feat, 29% min
    ("kr-vs-kp",      "kr-vs-kp",       1,        None),    # 3.2K, 36 feat, balanced
    ("jm1",           "jm1",            1,        None),    # 10.9K, 21 feat, 19% min
    ("electricity",   "electricity",    1,        5000),    # 45K → 5K subsample
    ("adult",         "adult",          2,        5000),    # 48K → 5K subsample
    ("musk",          "musk",           1,        None),    # 6.6K, 166 feat, 43% min
    ("nomao",         "nomao",          1,        5000),    # 34.5K → 5K subsample
    ("bank-marketing","bank-marketing", 1,        5000),    # 41K → 5K subsample
]


def _to_binary(X_raw, y_raw):
    """Encode X as float array and y as 0/1. For multiclass, keep top-2 classes."""
    # encode X
    if isinstance(X_raw, pd.DataFrame):
        Xc = X_raw.copy()
        for col in Xc.columns:
            s = Xc[col]
            if s.dtype == object or str(s.dtype) == "category":
                le = LabelEncoder()
                Xc[col] = le.fit_transform(s.astype(str).fillna("NA"))
        Xc = Xc.fillna(Xc.median(numeric_only=True))
        X = Xc.values.astype(float)
    else:
        X = np.asarray(X_raw, dtype=float)

    # encode y
    if isinstance(y_raw, pd.Series):
        ys = y_raw.astype(str)
    else:
        ys = pd.Series(np.asarray(y_raw).astype(str))

    cls, cnt = np.unique(ys, return_counts=True)
    if len(cls) > 2:
        top2 = cls[np.argsort(-cnt)[:2]]
        mask = ys.isin(top2)
        X = X[mask.values]
        ys = ys[mask].reset_index(drop=True)
        cls, cnt = np.unique(ys, return_counts=True)

    le = LabelEncoder()
    y = le.fit_transform(ys).astype(int)
    return X, y




def download_one(save_name, openml_name, version, subsample_n, seed=42):
    out_path = DATA_DIR / f"{save_name}.pkl"
    if out_path.exists():
        print(f"  {save_name:20s}  already cached, skip")
        return True

    print(f"  {save_name:20s}  fetching '{openml_name}' v{version} ...", end="", flush=True)
    try:
        bunch = fetch_openml(openml_name, version=version, as_frame=True,
                             parser="auto")
        X, y = _to_binary(bunch.data, bunch.target)
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    # drop rows with NaN
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    # subsample if requested (stratified)
    if subsample_n and len(X) > subsample_n:
        rng = np.random.RandomState(seed)
        cls, cnt = np.unique(y, return_counts=True)
        idx = []
        for c, n in zip(cls, cnt):
            ci = np.where(y == c)[0]
            take = int(subsample_n * n / len(y))
            take = max(take, 50)
            idx.extend(rng.choice(ci, min(take, len(ci)), replace=False))
        idx = np.array(idx)
        rng.shuffle(idx)
        X, y = X[idx], y[idx]

    cls, cnt = np.unique(y, return_counts=True)
    min_pct = 100 * cnt.min() / cnt.sum()

    if len(X) < 1000:
        print(f"  SKIP (n={len(X)} too small)")
        return False
    if min_pct < 8.0:
        print(f"  SKIP (minority={min_pct:.1f}% too imbalanced)")
        return False

    with open(out_path, "wb") as f:
        pickle.dump(Dataset(X, y), f)

    print(f"  OK  n={len(X):6d}  p={X.shape[1]:3d}  min%={min_pct:.1f}%")
    return True


def run():
    print(f"Downloading datasets to {DATA_DIR}\n")
    ok, fail = 0, 0
    for args in TARGETS:
        success = download_one(*args)
        if success:
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} saved, {fail} failed/skipped")


if __name__ == "__main__":
    run()
