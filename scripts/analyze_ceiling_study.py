"""Test the thermal-gap hypothesis on the multi-city ceiling sweep.

paper_v2 Section III-C states the built-vegetated LST gap "rank-orders the
ceiling perfectly across all tested cities" -- on four cities. With four
points a perfect Spearman rho carries essentially no evidence (any monotone
covariate has a 1-in-24 chance of ranking four points perfectly by luck), so
the claim needs a larger sample before it can be stated.

This ranks every measured covariate against the ceiling and reports how the
perfect-ordering claim fares out of sample.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RESULTS = OUT / "ceiling_study.jsonl"

COVARIATES = [
    ("veg_imp_lst_gap_c", "built-vegetated LST gap (C)"),
    ("lst_spread_c", "LST spread p98-p2 (C)"),
    ("impervious_std", "impervious fraction sigma"),
    ("impervious_mean", "impervious fraction mean"),
    ("vegetation_mean", "vegetation fraction mean"),
    ("class_entropy", "class entropy (bits)"),
    ("water_mean", "water fraction mean"),
    ("lst_mean_c", "mean LST (C)"),
]


def main() -> None:
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if np.isfinite(r.get("ceiling_r", np.nan))
            and np.isfinite(r.get("veg_imp_lst_gap_c", np.nan))]
    n = len(rows)
    print(f"cities with a usable ceiling: {n}\n")
    if n < 5:
        print("too few cities to analyse")
        return

    y = np.array([r["ceiling_r"] for r in rows])
    ymulti = np.array([r["ceiling_multi_r"] for r in rows])

    print(f"ceiling r  : min {y.min():+.3f}  median {np.median(y):+.3f}  "
          f"max {y.max():+.3f}")
    print(f"multi R    : min {ymulti.min():+.3f}  median {np.median(ymulti):+.3f}  "
          f"max {ymulti.max():+.3f}")
    print()

    print("=" * 78)
    print("SINGLE-COVARIATE ASSOCIATION WITH THE CEILING")
    print("=" * 78)
    print(f"{'covariate':<34} {'Pearson':>9} {'Spearman':>9} {'p (Spm)':>10}")
    print("-" * 78)
    ranked = []
    for key, label in COVARIATES:
        x = np.array([r[key] for r in rows], dtype=float)
        ok = np.isfinite(x)
        if ok.sum() < 5:
            continue
        pr = stats.pearsonr(x[ok], y[ok])
        sp = stats.spearmanr(x[ok], y[ok])
        ranked.append((abs(sp.statistic), label, pr.statistic, sp.statistic,
                       sp.pvalue))
        print(f"{label:<34} {pr.statistic:>+9.3f} {sp.statistic:>+9.3f} "
              f"{sp.pvalue:>10.3g}")
    ranked.sort(reverse=True)
    print()
    print(f"  strongest single predictor: {ranked[0][1]} "
          f"(Spearman {ranked[0][3]:+.3f}, p = {ranked[0][4]:.3g})")

    # ---- the paper's specific claim ----
    print()
    print("=" * 78)
    print("DOES THE GAP RANK-ORDER THE CEILING 'PERFECTLY'?")
    print("=" * 78)
    gap = np.array([r["veg_imp_lst_gap_c"] for r in rows])
    sp = stats.spearmanr(gap, y)
    order_gap = np.argsort(gap)
    order_ceil = np.argsort(y)
    perfect = bool(np.array_equal(order_gap, order_ceil))
    n_pairs = 0
    n_conc = 0
    for i, j in combinations(range(len(rows)), 2):
        n_pairs += 1
        if (gap[i] - gap[j]) * (y[i] - y[j]) > 0:
            n_conc += 1
    print(f"  n = {len(rows)} cities, Spearman rho = {sp.statistic:+.3f} "
          f"(p = {sp.pvalue:.3g})")
    print(f"  perfect rank ordering: {'YES' if perfect else 'NO'}")
    print(f"  concordant pairs: {n_conc}/{n_pairs} ({n_conc / n_pairs:.0%})")

    # worst violations of the claimed ordering
    viol = []
    for i, j in combinations(range(len(rows)), 2):
        if (gap[i] - gap[j]) * (y[i] - y[j]) < 0:
            viol.append((abs(gap[i] - gap[j]), rows[i], rows[j]))
    viol.sort(reverse=True, key=lambda t: t[0])
    if viol:
        print("\n  largest ordering violations (bigger gap, lower ceiling):")
        for _, a, b in viol[:5]:
            hi, lo = (a, b) if a["veg_imp_lst_gap_c"] > b["veg_imp_lst_gap_c"] else (b, a)
            print(f"    {hi['city']:<15} gap {hi['veg_imp_lst_gap_c']:+5.2f} "
                  f"ceiling {hi['ceiling_r']:+.3f}   vs   "
                  f"{lo['city']:<15} gap {lo['veg_imp_lst_gap_c']:+5.2f} "
                  f"ceiling {lo['ceiling_r']:+.3f}")

    # ---- how noisy is a single-day ceiling? ----
    sds = [r["ceiling_r_sd"] for r in rows
           if np.isfinite(r.get("ceiling_r_sd", np.nan))]
    ndays = [r.get("n_days", 1) for r in rows]
    if sds:
        print()
        print("=" * 78)
        print("SINGLE-DAY NOISE (why the ceiling is now a multi-day mean)")
        print("=" * 78)
        print(f"  days per city : median {int(np.median(ndays))}, "
              f"range {min(ndays)}-{max(ndays)}")
        print(f"  per-city sd of the daily ceiling: median {np.median(sds):.3f}, "
              f"max {max(sds):.3f}")
        worst = max(rows, key=lambda r: r.get("ceiling_r_sd", 0))
        print(f"  noisiest city : {worst['city']} "
              f"({worst['ceiling_r']:+.3f} +- {worst['ceiling_r_sd']:.3f}, "
              f"range [{worst['ceiling_r_min']:+.3f}, {worst['ceiling_r_max']:+.3f}] "
              f"over {worst['n_days']} days)")
        print(f"  a single-day ceiling therefore carries roughly +-{np.median(sds):.2f}")
        print(f"  of noise, which is the scale of several cities' worth of signal.")

    # ---- is the association just the arid tail? ----
    print()
    print("=" * 78)
    print("ROBUSTNESS: IS THE ASSOCIATION ONLY THE ARID TAIL?")
    print("=" * 78)
    print("  The worry is that cloud dropout removes humid cities, leaving a")
    print("  sample whose correlation is carried entirely by a few desert")
    print("  outliers. Refitting on subsets tests that directly.")
    for label, keep in (
        ("all cities", np.ones(len(rows), dtype=bool)),
        ("excluding negative-ceiling cities", y > 0),
        ("excluding the 5 most arid (highest mean LST)",
         np.argsort(np.argsort([-r["lst_mean_c"] for r in rows])) >= 5),
        ("humid half only (below-median mean LST)",
         np.array([r["lst_mean_c"] for r in rows]) < np.median(
             [r["lst_mean_c"] for r in rows])),
    ):
        k = np.asarray(keep, dtype=bool)
        if k.sum() < 6:
            continue
        sp2 = stats.spearmanr(gap[k], y[k])
        print(f"  {label:<44} n={int(k.sum()):>3}  "
              f"rho={sp2.statistic:+.3f}  p={sp2.pvalue:.3g}")

    # ---- multivariate ----
    print()
    print("=" * 78)
    print("MULTIVARIATE FIT")
    print("=" * 78)
    keys = ["veg_imp_lst_gap_c", "lst_spread_c", "impervious_std", "class_entropy"]
    X = np.column_stack([[r[k] for r in rows] for k in keys])
    ok = np.all(np.isfinite(X), axis=1)
    A = np.column_stack([X[ok], np.ones(ok.sum())])
    beta, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    pred = A @ beta
    R = float(np.corrcoef(pred, y[ok])[0, 1])
    print(f"  predictors: {', '.join(keys)}")
    print(f"  multiple R = {R:.3f}  (R^2 = {R**2:.3f}) on n = {int(ok.sum())}")
    for k, b in zip(keys, beta[:-1]):
        print(f"    {k:<24} beta = {b:+.4f}")

    gap_only = np.column_stack([X[ok][:, 0], np.ones(ok.sum())])
    b1, *_ = np.linalg.lstsq(gap_only, y[ok], rcond=None)
    R1 = float(np.corrcoef(gap_only @ b1, y[ok])[0, 1])
    print(f"  gap alone  : R = {R1:.3f} (R^2 = {R1**2:.3f})")

    (OUT / "ceiling_study_analysis.json").write_text(json.dumps({
        "n_cities": n,
        "spearman_gap_vs_ceiling": float(sp.statistic),
        "p_value": float(sp.pvalue),
        "perfect_rank_ordering": perfect,
        "concordant_fraction": n_conc / n_pairs,
        "multivariate_R": R, "gap_only_R": R1,
    }, indent=2))
    print(f"\nwrote outputs/ceiling_study_analysis.json")


if __name__ == "__main__":
    main()
