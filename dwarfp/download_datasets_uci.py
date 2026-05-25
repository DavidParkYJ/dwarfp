"""download_datasets_uci.py — Fetch 6 hard datasets directly from UCI repository.

Bypasses OpenML (currently returning 504). Each dataset has its own file
format on UCI; this script handles .data, .csv (;-delimited), .arff, .xls.

Skipped: kc1, steel-plates-faults (OpenML-only).
"""

import io
import pickle
import ssl
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dwarfp.common import Dataset

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_cache"

# UCI direct URLs (ml-databases paths still serve the static files)
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"
TARGETS = {
    "breast-cancer-uci":   f"{BASE}/breast-cancer/breast-cancer.data",
    "mammographic-mass":   f"{BASE}/mammographic-masses/mammographic_masses.data",
    "wine-quality-red":    f"{BASE}/wine-quality/winequality-red.csv",
    "wine-quality-white":  f"{BASE}/wine-quality/winequality-white.csv",
    "thoracic-surgery":    f"{BASE}/00277/ThoraricSurgery.arff",
    "default-credit-card": f"{BASE}/00350/default%20of%20credit%20card%20clients.xls",
}


def _fetch(url, timeout=30):
    # UCI's TLS cert sometimes expires; allow unverified SSL as fallback.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()


def _encode_X(df):
    Xc = df.copy()
    for col in Xc.columns:
        s = Xc[col]
        if not pd.api.types.is_numeric_dtype(s):
            le = LabelEncoder()
            Xc[col] = le.fit_transform(s.astype(str).fillna("NA"))
    Xc = Xc.fillna(Xc.median(numeric_only=True))
    return Xc.values.astype(float)


def _save(save_name, X, y):
    cls, cnt = np.unique(y, return_counts=True)
    nan_mask = ~np.isnan(X).any(axis=1)
    X, y = X[nan_mask], y[nan_mask]
    cls, cnt = np.unique(y, return_counts=True)
    min_pct = 100 * cnt.min() / cnt.sum()
    out_path = DATA_DIR / f"{save_name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(Dataset(X, y), f)
    print(f"  saved n={len(X):6d}  p={X.shape[1]:3d}  "
          f"classes={cls.tolist()}  min%={min_pct:.1f}%")


def fetch_breast_cancer(save_name, url):
    # 286 rows, 10 cols (1 target + 9 features), missing="?"
    raw = _fetch(url).decode("utf-8")
    cols = ["target", "age", "menopause", "tumor-size", "inv-nodes",
            "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"]
    df = pd.read_csv(io.StringIO(raw), header=None, names=cols, na_values="?")
    y_raw = df["target"]
    Xdf = df.drop(columns=["target"])
    y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(int)
    X = _encode_X(Xdf)
    _save(save_name, X, y)


def fetch_mammographic_mass(save_name, url):
    # 961 rows, 6 cols: 5 features + severity target (last col)
    raw = _fetch(url).decode("utf-8")
    cols = ["BI-RADS", "age", "shape", "margin", "density", "severity"]
    df = pd.read_csv(io.StringIO(raw), header=None, names=cols, na_values="?")
    y = df["severity"].fillna(-1).astype(int).values
    keep = y >= 0
    df = df[keep]; y = y[keep]
    X = _encode_X(df.drop(columns=["severity"]))
    _save(save_name, X, y)


def fetch_wine_quality(save_name, url):
    raw = _fetch(url).decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), sep=";")
    quality = df["quality"].values
    y = (quality >= 6).astype(int)
    X = _encode_X(df.drop(columns=["quality"]))
    _save(save_name, X, y)


def fetch_thoracic_surgery(save_name, url):
    raw = _fetch(url).decode("utf-8")
    # ARFF: skip lines starting with @, read remaining as CSV
    lines = [ln for ln in raw.splitlines()
             if ln and not ln.startswith("@") and not ln.startswith("%")]
    body = "\n".join(lines)
    df = pd.read_csv(io.StringIO(body), header=None)
    # last column is target (T=died/F=survived)
    y_raw = df.iloc[:, -1]
    y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(int)
    X = _encode_X(df.iloc[:, :-1])
    _save(save_name, X, y)


def fetch_default_credit_card(save_name, url, subsample_n=5000, seed=42):
    raw = _fetch(url, timeout=60)
    # XLS file: header is row 1, data starts row 2, first col is ID
    df = pd.read_excel(io.BytesIO(raw), header=1)
    df = df.drop(columns=[c for c in df.columns if str(c).upper() == "ID"])
    target_col = [c for c in df.columns if "default" in str(c).lower()][0]
    y = df[target_col].astype(int).values
    X_df = df.drop(columns=[target_col])
    X = _encode_X(X_df)
    # stratified subsample to 5000
    if len(X) > subsample_n:
        rng = np.random.RandomState(seed)
        cls, cnt = np.unique(y, return_counts=True)
        idx = []
        for c, n in zip(cls, cnt):
            ci = np.where(y == c)[0]
            take = max(int(subsample_n * n / len(y)), 50)
            idx.extend(rng.choice(ci, min(take, len(ci)), replace=False))
        idx = np.array(idx)
        rng.shuffle(idx)
        X, y = X[idx], y[idx]
    _save(save_name, X, y)


FETCHERS = {
    "breast-cancer-uci":   fetch_breast_cancer,
    "mammographic-mass":   fetch_mammographic_mass,
    "wine-quality-red":    fetch_wine_quality,
    "wine-quality-white":  fetch_wine_quality,
    "thoracic-surgery":    fetch_thoracic_surgery,
    "default-credit-card": fetch_default_credit_card,
}


def run():
    print(f"Downloading 6 UCI datasets to {DATA_DIR}\n")
    ok, fail = 0, 0
    for name, url in TARGETS.items():
        out = DATA_DIR / f"{name}.pkl"
        if out.exists():
            print(f"  {name:24s}  already cached, skip"); ok += 1; continue
        print(f"  {name:24s}  fetching ...", end="", flush=True)
        try:
            FETCHERS[name](name, url)
            ok += 1
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            fail += 1
    print(f"\nDone: {ok} saved, {fail} failed")


if __name__ == "__main__":
    run()
