"""step11_mass_spread.py — M·S applicability indicator: product, quintiles, correlation.

Assembles the boundary-mass x boundary-spread analysis (paper Sec. 6.4-6.5,
Tables tab:mass-spread and tab:mass-spread-full) from the OOB by-products
already produced by the upstream steps:

  - step8_boundary_mass        -> results_fp_share.csv         (M = OOB fp[.4,.6) share)
  - step9_boundary_spread      -> results_boundary_spread.csv  (S = avg boundary spread)
  - compare_baselines / step6  -> results_baselines.csv        (Δacc = d_CPFW_acc)
  - step10_weight_amplification-> results_weight_amplification.csv (K0 / K* gains)
        (that file is .gitignored; if absent the script falls back to the
         tracked, byte-compatible results_alpha_ms_cv.csv, so every figure
         reproduces from a clean clone without re-running step10)

The canonical M and S are the OOB sample-level indicators (step8, step9);
step10's per-rep M/S are internal to the amplification factor only and are
NOT used here.

Run step8, step9 first (step10 optional — see fall-back above). Produces
results_mass_spread.csv (per-dataset) and results_mass_spread_quintile.csv
(quintile summary incl. the bottom-quintile K*=0 share), and prints:
  - tab:mass-spread-full  per-dataset M, S, M·S, Δacc (sorted by M·S desc)
  - Pearson r(M·S, Δacc)  (paper: +0.840)
  - tab:mass-spread       M·S quintile mean Δacc + W/T/L
  - amplification by quintile (top: +0.99pp K=0 -> +1.48pp K*; bottom-quintile
    K*=0 in 212/240 = 88% of runs)
"""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)
from dwarfp.common import DATASETS

HERE = Path(__file__).resolve().parent

# Quintile sizes for 36 datasets: Q1 holds the extra (36 = 8 + 7*4).
QUINTILE_SIZES = [8, 7, 7, 7, 7]


def _read_csv(name):
    with open(HERE / name) as f:
        return list(csv.DictReader(f))


def _load_inputs():
    """Join M (step8), S (step9), Δacc (baselines) per dataset."""
    M = {r["dataset"]: float(r["[.4,.6)"]) for r in _read_csv("results_fp_share.csv")}
    S = {r["dataset"]: float(r["avg_spread"])
         for r in _read_csv("results_boundary_spread.csv")}
    dacc = {r["dataset"]: float(r["d_CPFW_acc"])
            for r in _read_csv("results_baselines.csv")}

    rows = []
    for name in DATASETS:
        if name not in M or name not in S or name not in dacc:
            print(f"  [warn] missing input for {name}; skipped")
            continue
        rows.append({"dataset": name, "M": M[name], "S": S[name],
                     "MS": M[name] * S[name], "dacc": dacc[name]})
    return rows


# Amplification table source. step10 writes results_weight_amplification.csv,
# but that file is .gitignored; the byte-compatible, version-controlled twin is
# results_alpha_ms_cv.csv (identical schema and numbers). Prefer step10's fresh
# output when present, else fall back to the tracked file so the quintile
# statistics (incl. the bottom-quintile K*=0 share) reproduce from a clean clone.
_AMP_SOURCES = ["results_weight_amplification.csv", "results_alpha_ms_cv.csv"]


def _amp_source():
    for fn in _AMP_SOURCES:
        if (HERE / fn).exists():
            return fn
    return None


def _amp_by_quintile(quintile_names, amp_csv):
    """Mean K0 / K* gain per quintile + bottom-quintile K*=0 run counts.

    Reads the per-(dataset, rep) amplification table (step10 output, or the
    tracked results_alpha_ms_cv.csv twin).
    """
    amp = _read_csv(amp_csv)
    # per-dataset mean gains and per-dataset K*=0 run counts
    by_ds = {}
    for r in amp:
        d = r["dataset"]
        by_ds.setdefault(d, {"k0": [], "kstar": [], "kstar0": 0, "n": 0})
        rf = float(r["rf_acc"])
        by_ds[d]["k0"].append(float(r["K0_acc"]) - rf)
        by_ds[d]["kstar"].append(float(r["Kstar_acc"]) - rf)
        by_ds[d]["n"] += 1
        if int(float(r["K_star"])) == 0:
            by_ds[d]["kstar0"] += 1

    out = []
    for qi, names in enumerate(quintile_names, 1):
        k0 = [np.mean(by_ds[n]["k0"]) for n in names if n in by_ds]
        ks = [np.mean(by_ds[n]["kstar"]) for n in names if n in by_ds]
        runs = sum(by_ds[n]["n"] for n in names if n in by_ds)
        ks0 = sum(by_ds[n]["kstar0"] for n in names if n in by_ds)
        out.append({"q": qi, "k0_mean": float(np.mean(k0)),
                    "kstar_mean": float(np.mean(ks)),
                    "kstar0_runs": ks0, "n_runs": runs,
                    "kstar0_share": ks0 / runs if runs else float("nan")})
    return out


def run():
    rows = _load_inputs()
    if not rows:
        print("No inputs. Run step8, step9, compare_baselines first.")
        return

    # ── tab:mass-spread-full — per-dataset, sorted by M·S desc ──────────
    rows_desc = sorted(rows, key=lambda r: -r["MS"])
    print("Table B.4 (tab:mass-spread-full): per-dataset M, S, M·S, Δacc\n")
    print(f'{"dataset":20s} {"M":>7} {"S":>7} {"M*S":>8} {"d_acc":>9}')
    print("-" * 55)
    for r in rows_desc:
        print(f'{r["dataset"]:20s} {r["M"]:7.3f} {r["S"]:7.3f} '
              f'{r["MS"]:8.4f} {r["dacc"]:+9.4f}')

    out_path = HERE / "results_mass_spread.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "M", "S", "MS", "d_acc"])
        for r in rows_desc:
            w.writerow([r["dataset"], f'{r["M"]:.4f}', f'{r["S"]:.4f}',
                        f'{r["MS"]:.4f}', f'{r["dacc"]:+.4f}'])
    print(f"\nSaved: {out_path}")

    # ── Pearson r(M·S, Δacc) — paper: +0.840 ───────────────────────────
    ms = np.array([r["MS"] for r in rows])
    da = np.array([r["dacc"] for r in rows])
    pr, pp = pearsonr(ms, da)
    print(f"\nPearson r(M·S, Δacc) = {pr:+.4f}  (p={pp:.4g}, n={len(rows)})")

    # ── tab:mass-spread — quintiles by M·S (ascending) ─────────────────
    rows_asc = sorted(rows, key=lambda r: r["MS"])
    quintiles, quintile_names, i = [], [], 0
    for size in QUINTILE_SIZES:
        chunk = rows_asc[i:i + size]
        quintiles.append(chunk)
        quintile_names.append([r["dataset"] for r in chunk])
        i += size

    print("\nTable (tab:mass-spread): mean Δacc by M·S quintile\n")
    print(f'{"Quintile":12s} {"M*S range":>20} {"mean d_acc":>11} {"W/T/L":>8}')
    print("-" * 54)
    labels = ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]
    qstats = []
    for lab, chunk in zip(labels, quintiles):
        d = np.array([r["dacc"] for r in chunk])
        w = int((d > 1e-9).sum())
        l = int((d < -1e-9).sum())
        t = len(chunk) - w - l
        lo, hi = chunk[0]["MS"], chunk[-1]["MS"]
        print(f'{lab:12s} [{lo:.4f}, {hi:.4f}]   {d.mean():+11.4f} '
              f'{f"{w}/{t}/{l}":>8}')
        qstats.append({"quintile": lab, "ms_lo": lo, "ms_hi": hi,
                       "mean_dacc": float(d.mean()), "W": w, "T": t, "L": l})

    # ── amplification by quintile (Sec. 6.5) ───────────────────────────
    # The bottom-quintile K*=0 share is the figure quoted in Sec. 6.5: at
    # M·S <= 0.0028 (Q1) amplification almost never fires, so those datasets
    # are left effectively unchanged. Reproduced here as 212/240 = 88%.
    amp_csv = _amp_source()
    if amp_csv:
        amp = _amp_by_quintile(quintile_names, amp_csv)
        top, bot = amp[4], amp[0]
        for q, st in zip(amp, qstats):
            st["amp_k0_mean"] = q["k0_mean"]
            st["amp_kstar_mean"] = q["kstar_mean"]
            st["kstar0_runs"] = q["kstar0_runs"]
            st["n_runs"] = q["n_runs"]
            st["kstar0_share"] = q["kstar0_share"]
        print(f"\nAmplification by quintile (source: {amp_csv}):")
        print(f'  top quintile (Q5): {top["k0_mean"]*100:+.2f}pp (K=0) '
              f'-> {top["kstar_mean"]*100:+.2f}pp (K*)')
        print(f'  bottom quintile (Q1): {bot["k0_mean"]*100:+.2f}pp (K=0); '
              f'K*=0 in {bot["kstar0_runs"]}/{bot["n_runs"]} = '
              f'{bot["kstar0_share"]*100:.0f}% of runs')
    else:
        print(f"\n[skip] none of {_AMP_SOURCES} found "
              "(run step10 for amplification-by-quintile).")

    # ── persist the quintile summary (managed artifact) ────────────────
    sum_path = HERE / "results_mass_spread_quintile.csv"
    cols = ["quintile", "ms_lo", "ms_hi", "mean_dacc", "W", "T", "L",
            "amp_k0_mean", "amp_kstar_mean", "kstar0_runs", "n_runs",
            "kstar0_share"]
    fmt = {"ms_lo": "{:.4f}", "ms_hi": "{:.4f}", "mean_dacc": "{:+.4f}",
           "amp_k0_mean": "{:+.4f}", "amp_kstar_mean": "{:+.4f}",
           "kstar0_share": "{:.4f}"}
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for st in qstats:
            w.writerow({k: (fmt[k].format(st[k]) if k in fmt and k in st
                            else st.get(k, "")) for k in cols})
    print(f"\nSaved: {sum_path}")


if __name__ == "__main__":
    run()
