"""Every wide-sweep number, derived from the single `full_sweep.jsonl` pass.

Supersedes `analyze_ceiling_study.py`, `run_dual_target_analysis.py` and the
four-city `run_channel_prescreen.py`. Those were written incrementally and
read different artifacts with different date handling; this reads one file
produced by one pass on the final pipeline, so the gap heuristic and the
channel screen are finally on the same footing.

Produces, all from the same cities and the same days:

  1. distributions of both ceiling definitions;
  2. the gap heuristic against both targets, with bootstrap CIs and the
     paired test of the difference;
  3. robustness subsets against both targets;
  4. the NDVI channel screen across the whole sample rather than four
     cities, including whether the gain is predictable from aridity --
     which is the claim the paper actually wants to make;
  5. LaTeX tables for the paper.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SRC = OUT / "full_sweep.jsonl"
B_BOOT = 5000
SEED = 0

COVARIATES = [
    ("veg_imp_lst_gap_c", "built-vegetated LST gap"),
    ("lst_mean_c", "mean LST"),
    ("water_mean", "water fraction"),
    ("impervious_mean", "impervious fraction mean"),
    ("lst_spread_c", "LST spread (p98-p2)"),
    ("impervious_std", "impervious fraction sd"),
    ("vegetation_mean", "vegetation fraction mean"),
    ("class_entropy", "class entropy"),
]


def spearman_ci(x, y, b=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    reps = np.full(b, np.nan)
    for i in range(b):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) > 0 and np.std(y[idx]) > 0:
            reps[i] = stats.spearmanr(x[idx], y[idx]).statistic
    return tuple(float(v) for v in np.nanpercentile(reps, [2.5, 97.5]))


def paired_rho_diff(x, ya, yb, b=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    reps = np.full(b, np.nan)
    for i in range(b):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0:
            continue
        reps[i] = (stats.spearmanr(x[idx], ya[idx]).statistic
                   - stats.spearmanr(x[idx], yb[idx]).statistic)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return float(np.nanmean(reps)), float(lo), float(hi)


def concordance(x, y):
    tot = con = 0
    for i, j in combinations(range(len(x)), 2):
        tot += 1
        if (x[i] - x[j]) * (y[i] - y[j]) > 0:
            con += 1
    return con, tot


def col(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


def main() -> None:
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if np.isfinite(r.get("ceiling_r", np.nan))
            and np.isfinite(r.get("ceiling_multi_r", np.nan))]
    n = len(rows)
    gap = col(rows, "veg_imp_lst_gap_c")
    uni = col(rows, "ceiling_r")
    mul = col(rows, "ceiling_multi_r")
    lst = col(rows, "lst_mean_c")
    gain = col(rows, "ndvi_gain")
    gain_sd = col(rows, "ndvi_gain_sd")

    print(f"cities: {n}   days per city: median "
          f"{int(np.median(col(rows, 'n_days')))} "
          f"(range {int(col(rows, 'n_days').min())}-{int(col(rows, 'n_days').max())})")
    print(f"  univariate  r : median {np.median(uni):+.3f}  "
          f"[{uni.min():+.3f}, {uni.max():+.3f}]  negatives {(uni < 0).sum()}")
    print(f"  multivariate R: median {np.median(mul):+.3f}  "
          f"[{mul.min():+.3f}, {mul.max():+.3f}]  negatives {(mul < 0).sum()}")
    print(f"  median per-city sd: univariate "
          f"{np.median(col(rows, 'ceiling_r_sd')):.3f}, multivariate "
          f"{np.median(col(rows, 'ceiling_multi_r_sd')):.3f}")

    # ---- 1. gap heuristic, both targets ----
    print()
    print("=" * 94)
    print("GAP HEURISTIC, BOTH TARGETS")
    print("=" * 94)
    su, sm = stats.spearmanr(gap, uni), stats.spearmanr(gap, mul)
    cu, cm = spearman_ci(gap, uni), spearman_ci(gap, mul)
    ccu, cpu = concordance(gap, uni)
    ccm, cpm = concordance(gap, mul)
    print(f"  vs univariate  : rho {su.statistic:+.3f} "
          f"[{cu[0]:+.3f}, {cu[1]:+.3f}]  p {su.pvalue:.3g}  "
          f"concordant {100 * ccu / cpu:.0f} pct")
    print(f"  vs multivariate: rho {sm.statistic:+.3f} "
          f"[{cm[0]:+.3f}, {cm[1]:+.3f}]  p {sm.pvalue:.3g}  "
          f"concordant {100 * ccm / cpm:.0f} pct")
    d, dlo, dhi = paired_rho_diff(gap, uni, mul)
    print(f"  paired difference: {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
          f"{'  (excludes zero)' if dlo > 0 else '  (includes zero)'}")

    print()
    print(f"{'covariate':<30} {'rho(uni)':>9} {'rho(multi)':>11} {'p(multi)':>11}")
    print("-" * 94)
    cov_out = []
    for key, label in COVARIATES:
        x = col(rows, key)
        a, b_ = stats.spearmanr(x, uni), stats.spearmanr(x, mul)
        print(f"{label:<30} {a.statistic:>+9.3f} {b_.statistic:>+11.3f} "
              f"{b_.pvalue:>11.3g}")
        cov_out.append(dict(covariate=label, rho_uni=a.statistic,
                            rho_multi=b_.statistic, p_multi=b_.pvalue))

    # ---- 2. robustness ----
    print()
    print("=" * 94)
    print("ROBUSTNESS SUBSETS")
    print("=" * 94)
    med = float(np.median(lst))
    subsets = [
        ("all cities", np.ones(n, dtype=bool)),
        ("excluding negative univariate ceilings", uni > 0),
        ("excluding 5 most arid", np.argsort(np.argsort(-lst)) >= 5),
        ("humid half (below-median mean LST)", lst < med),
    ]
    print(f"{'subset':<42} {'n':>4} {'rho(uni)':>9} {'p':>10} "
          f"{'rho(multi)':>11} {'p':>10}")
    print("-" * 94)
    sub_out = []
    for label, k in subsets:
        k = np.asarray(k, dtype=bool)
        if k.sum() < 6:
            continue
        a, b_ = stats.spearmanr(gap[k], uni[k]), stats.spearmanr(gap[k], mul[k])
        flag = "" if b_.pvalue < 0.05 else "  n.s."
        print(f"{label:<42} {int(k.sum()):>4} {a.statistic:>+9.3f} "
              f"{a.pvalue:>10.3g} {b_.statistic:>+11.3f} {b_.pvalue:>10.3g}{flag}")
        sub_out.append(dict(subset=label, n=int(k.sum()),
                            rho_uni=a.statistic, p_uni=a.pvalue,
                            rho_multi=b_.statistic, p_multi=b_.pvalue))

    # ---- 3. NDVI channel screen, whole sample ----
    ok = np.isfinite(gain)
    print()
    print("=" * 94)
    print(f"NDVI CHANNEL SCREEN ACROSS {int(ok.sum())} CITIES")
    print("=" * 94)
    print(f"  gain: median {np.median(gain[ok]):+.4f}  "
          f"[{gain[ok].min():+.4f}, {gain[ok].max():+.4f}]")
    robust = ok & (gain - 2 * gain_sd > 0)
    print(f"  cities where gain - 2sd > 0 (robustly positive): "
          f"{int(robust.sum())} of {int(ok.sum())}")

    order = np.argsort(-np.where(ok, gain, -np.inf))
    print(f"\n  {'city':<16} {'gain':>9} {'sd':>8} {'3ch':>8} {'+NDVI':>8}  robust")
    print("  " + "-" * 62)
    for i in list(order[:8]) + list(order[max(0, int(ok.sum()) - 4):int(ok.sum())]):
        r = rows[i]
        print(f"  {r['city']:<16} {gain[i]:>+9.4f} {gain_sd[i]:>8.4f} "
              f"{r.get('ceiling_multi_r_ndvimask', float('nan')):>8.3f} "
              f"{r.get('ceiling_ndvi_r', float('nan')):>8.3f}  "
              f"{'yes' if robust[i] else ''}")

    print("\n  is the NDVI gain predictable from city character?")
    for key, label in (("lst_mean_c", "mean LST (aridity proxy)"),
                       ("veg_imp_lst_gap_c", "built-vegetated gap"),
                       ("ceiling_multi_r", "multivariate ceiling"),
                       ("vegetation_mean", "vegetation fraction")):
        x = col(rows, key)[ok]
        sp = stats.spearmanr(x, gain[ok])
        print(f"    {label:<28} rho {sp.statistic:>+7.3f}  p {sp.pvalue:.3g}")

    # ---- LaTeX ----
    nl = chr(92) + chr(92)
    pct = chr(92) + "%"
    tex = [
        r"\begin{table}[ht]", r"  \centering",
        r"  \caption{Built--vegetated LST gap as a predictor of the land-cover",
        f"  data ceiling across {n} US cities. All quantities are multi-day",
        r"  means from a single sweep. Spearman $\rho$ with percentile",
        f"  bootstrap 95{pct} CIs ($B={B_BOOT}$).}}",
        r"  \label{tab:gap_dual}", r"  \small",
        r"  \begin{tabular}{lcc}", r"    \toprule",
        f"    & Univariate $r$ & Multivariate $R$ {nl}", r"    \midrule",
        f"    Spearman $\\rho$ & ${su.statistic:+.3f}$ & ${sm.statistic:+.3f}$ {nl}",
        f"    95{pct} CI & [${cu[0]:+.3f}$, ${cu[1]:+.3f}$] & "
        f"[${cm[0]:+.3f}$, ${cm[1]:+.3f}$] {nl}",
        f"    Concordant pairs & {100 * ccu / cpu:.0f}{pct} & {100 * ccm / cpm:.0f}{pct} {nl}",
        f"    Cities with negative ceiling & {(uni < 0).sum()} & {(mul < 0).sum()} {nl}",
        f"    Median per-city sd & ${np.median(col(rows, 'ceiling_r_sd')):.3f}$ & "
        f"${np.median(col(rows, 'ceiling_multi_r_sd')):.3f}$ {nl}",
        r"    \bottomrule", r"  \end{tabular}", r"\end{table}",
    ]
    (OUT / "full_sweep_tables.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (OUT / "full_sweep_analysis.json").write_text(json.dumps(dict(
        n_cities=n,
        headline=dict(rho_uni=su.statistic, ci_uni=list(cu), p_uni=su.pvalue,
                      rho_multi=sm.statistic, ci_multi=list(cm), p_multi=sm.pvalue,
                      paired_difference=dict(mean=d, ci=[dlo, dhi])),
        covariates=cov_out, subsets=sub_out,
        ndvi=dict(n_cities=int(ok.sum()), n_robust=int(robust.sum()),
                  median_gain=float(np.median(gain[ok]))),
    ), indent=2), encoding="utf-8")
    print("\nwrote outputs/full_sweep_analysis.json and full_sweep_tables.tex")


if __name__ == "__main__":
    main()
