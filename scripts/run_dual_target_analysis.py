"""Gap heuristic against BOTH ceiling definitions, with uncertainty.

paper_v3 argues the univariate ceiling is unsound because it can go negative
in arid cities, then validates the built-vegetated gap heuristic against that
same univariate ceiling, justifying the choice as the only metric feasible to
average at scale. It is not the only one: `ceiling_multi_r` is computed and
averaged for every city and sits in `ceiling_study.jsonl`. The choice is
therefore free, and it matters -- the association is materially weaker
against the sound metric, and the humid-subset robustness check does not
survive the swap.

This produces what is needed to report both honestly:

  * per-covariate association against each target;
  * percentile bootstrap CIs on Spearman rho, so the two can be compared
    rather than merely contrasted;
  * a PAIRED bootstrap on rho(uni) - rho(multi), which is the actual test of
    whether the univariate target flatters the heuristic. Pairing matters:
    both targets come from the same cities and the same imagery, so their
    sampling noise is highly correlated and comparing two overlapping CIs
    would understate the difference;
  * robustness subsets and four-predictor fits for both.

Emits a LaTeX table ready to drop into the paper.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
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
    """Percentile CI for Spearman rho by resampling cities with replacement."""
    rng = np.random.default_rng(seed)
    n = len(x)
    reps = np.full(b, np.nan)
    for i in range(b):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) > 0 and np.std(y[idx]) > 0:
            reps[i] = stats.spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return float(lo), float(hi)


def paired_rho_difference(x, y_a, y_b, b=B_BOOT, seed=SEED):
    """CI for rho(x, y_a) - rho(x, y_b) over shared city resamples."""
    rng = np.random.default_rng(seed)
    n = len(x)
    reps = np.full(b, np.nan)
    for i in range(b):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0:
            continue
        reps[i] = (stats.spearmanr(x[idx], y_a[idx]).statistic
                   - stats.spearmanr(x[idx], y_b[idx]).statistic)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return float(np.nanmean(reps)), float(lo), float(hi)


def tex_sci(v: float, sig: int = 1) -> str:
    """LaTeX scientific notation, e.g. 2.7e-14 -> 2.7\times10^{-14}."""
    if v >= 0.01:
        return f"{v:.2f}"
    mant, exp = f"{v:.{sig}e}".split("e")
    return f"{mant}" + chr(92) + "times10^{" + str(int(exp)) + "}"


def concordance(x, y):
    n_pairs = n_conc = 0
    for i, j in combinations(range(len(x)), 2):
        n_pairs += 1
        if (x[i] - x[j]) * (y[i] - y[j]) > 0:
            n_conc += 1
    return n_conc, n_pairs


def multi_fit(rows, y, keys):
    X = np.column_stack([[r[k] for r in rows] for k in keys]
                        + [np.ones(len(rows))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.corrcoef(X @ beta, y)[0, 1])


def main() -> None:
    src = (OUT / "ceiling_study.jsonl").read_text().splitlines()
    rows = [json.loads(l) for l in src if l.strip()]
    rows = [r for r in rows
            if np.isfinite(r.get("ceiling_r", np.nan))
            and np.isfinite(r.get("ceiling_multi_r", np.nan))
            and np.isfinite(r.get("veg_imp_lst_gap_c", np.nan))]
    n = len(rows)
    gap = np.array([r["veg_imp_lst_gap_c"] for r in rows])
    uni = np.array([r["ceiling_r"] for r in rows])
    mul = np.array([r["ceiling_multi_r"] for r in rows])
    lst = np.array([r["lst_mean_c"] for r in rows])

    print(f"n = {n} cities, multi-day means, B = {B_BOOT} bootstrap replicates")
    print(f"  univariate  r: min {uni.min():+.3f}  median {np.median(uni):+.3f}"
          f"  max {uni.max():+.3f}   negatives: {(uni < 0).sum()}")
    print(f"  multivariate R: min {mul.min():+.3f}  median {np.median(mul):+.3f}"
          f"  max {mul.max():+.3f}   negatives: {(mul < 0).sum()}")

    print()
    print("=" * 92)
    print("COVARIATE ASSOCIATION, BOTH TARGETS")
    print("=" * 92)
    print(f"{'covariate':<30} {'rho(uni)':>9} {'95% CI':>18} "
          f"{'rho(mult)':>10} {'95% CI':>18}")
    print("-" * 92)
    cov_rows = []
    for key, label in COVARIATES:
        x = np.array([r[key] for r in rows], dtype=float)
        su, sm = stats.spearmanr(x, uni), stats.spearmanr(x, mul)
        cu, cm = spearman_ci(x, uni), spearman_ci(x, mul)
        print(f"{label:<30} {su.statistic:>+9.3f} "
              f"[{cu[0]:+.3f},{cu[1]:+.3f}] {sm.statistic:>+10.3f} "
              f"[{cm[0]:+.3f},{cm[1]:+.3f}]")
        cov_rows.append(dict(covariate=label,
                             rho_uni=su.statistic, ci_uni=list(cu), p_uni=su.pvalue,
                             rho_multi=sm.statistic, ci_multi=list(cm),
                             p_multi=sm.pvalue))

    print()
    print("=" * 92)
    print("HEADLINE: BUILT-VEGETATED GAP")
    print("=" * 92)
    su, sm = stats.spearmanr(gap, uni), stats.spearmanr(gap, mul)
    cu, cm = spearman_ci(gap, uni), spearman_ci(gap, mul)
    ccu, cpu = concordance(gap, uni)
    ccm, cpm = concordance(gap, mul)
    pct_u = 100.0 * ccu / cpu
    pct_m = 100.0 * ccm / cpm
    print(f"  vs univariate  : rho = {su.statistic:+.3f} "
          f"[{cu[0]:+.3f}, {cu[1]:+.3f}]  p = {su.pvalue:.3g}   "
          f"concordant {ccu}/{cpu} ({pct_u:.0f} pct)")
    print(f"  vs multivariate: rho = {sm.statistic:+.3f} "
          f"[{cm[0]:+.3f}, {cm[1]:+.3f}]  p = {sm.pvalue:.3g}   "
          f"concordant {ccm}/{cpm} ({pct_m:.0f} pct)")

    d, dlo, dhi = paired_rho_difference(gap, uni, mul)
    print()
    print(f"  paired difference rho(uni) - rho(multi) = {d:+.3f} "
          f"[{dlo:+.3f}, {dhi:+.3f}]")
    if dlo > 0:
        print("  -> the univariate target gives a SIGNIFICANTLY stronger")
        print("     association. Part of what the heuristic tracks is the arid")
        print("     sign-flip the gap and the univariate ceiling share, rather")
        print("     than ceiling height as such.")
    else:
        print("  -> the two targets are not distinguishable at this sample size.")

    print()
    print("=" * 92)
    print("ROBUSTNESS SUBSETS, BOTH TARGETS")
    print("=" * 92)
    med = float(np.median(lst))
    subsets = [
        ("all cities", np.ones(n, dtype=bool)),
        ("excluding negative univariate ceilings", uni > 0),
        ("excluding 5 most arid (highest mean LST)",
         np.argsort(np.argsort(-lst)) >= 5),
        ("humid half (below-median mean LST)", lst < med),
    ]
    print(f"{'subset':<42} {'n':>4} {'rho(uni)':>9} {'p':>10} "
          f"{'rho(mult)':>10} {'p':>10}")
    print("-" * 92)
    sub_rows = []
    for label, k in subsets:
        k = np.asarray(k, dtype=bool)
        if k.sum() < 6:
            continue
        a, b_ = stats.spearmanr(gap[k], uni[k]), stats.spearmanr(gap[k], mul[k])
        flag = "" if b_.pvalue < 0.05 else "  <- n.s."
        print(f"{label:<42} {int(k.sum()):>4} {a.statistic:>+9.3f} "
              f"{a.pvalue:>10.3g} {b_.statistic:>+10.3f} {b_.pvalue:>10.3g}{flag}")
        sub_rows.append(dict(subset=label, n=int(k.sum()),
                             rho_uni=a.statistic, p_uni=a.pvalue,
                             rho_multi=b_.statistic, p_multi=b_.pvalue))

    print()
    print("=" * 92)
    print("FOUR-PREDICTOR FIT, BOTH TARGETS")
    print("=" * 92)
    keys = ["veg_imp_lst_gap_c", "lst_spread_c", "impervious_std", "class_entropy"]
    Ru, Rm = multi_fit(rows, uni, keys), multi_fit(rows, mul, keys)
    gu, gm = multi_fit(rows, uni, keys[:1]), multi_fit(rows, mul, keys[:1])
    print(f"  gap alone    -> uni R = {gu:.3f} (R2 {gu ** 2:.3f})   "
          f"multi R = {gm:.3f} (R2 {gm ** 2:.3f})")
    print(f"  4 predictors -> uni R = {Ru:.3f} (R2 {Ru ** 2:.3f})   "
          f"multi R = {Rm:.3f} (R2 {Rm ** 2:.3f})")

    # ---------------- LaTeX ----------------
    hk = lst < med
    hu, hm = stats.spearmanr(gap[hk], uni[hk]), stats.spearmanr(gap[hk], mul[hk])
    pct = chr(92) + "%"          # escaped percent for LaTeX
    nl = chr(92) + chr(92)       # row terminator
    tex = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Built--vegetated LST gap as a predictor of the land-cover",
        f"  data ceiling across {n} US cities (multi-day means). Reported",
        r"  against both ceiling definitions: the univariate correlation with",
        r"  impervious fraction, and the multiple correlation over the three",
        r"  channels the simulator consumes. Spearman $\rho$ with percentile",
        f"  bootstrap 95{pct} CIs ($B={B_BOOT}$).}}",
        r"  \label{tab:gap_dual}",
        r"  \small",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        f"    & Univariate $r$ & Multivariate $R$ {nl}",
        r"    \midrule",
        f"    Spearman $\\rho$ & ${su.statistic:+.3f}$ & ${sm.statistic:+.3f}$ {nl}",
        f"    95{pct} CI & [${cu[0]:+.3f}$, ${cu[1]:+.3f}$] & "
        f"[${cm[0]:+.3f}$, ${cm[1]:+.3f}$] {nl}",
        f"    $p$ & ${tex_sci(su.pvalue)}$ & ${tex_sci(sm.pvalue)}$ {nl}",
        f"    Concordant pairs & {ccu}/{cpu} ({pct_u:.0f}{pct}) & "
        f"{ccm}/{cpm} ({pct_m:.0f}{pct}) {nl}",
        r"    \midrule",
        f"    Humid half, $n={int(hk.sum())}$: $\\rho$ & ${hu.statistic:+.3f}$ & "
        f"${hm.statistic:+.3f}$ {nl}",
        f"    \\quad $p$ & ${tex_sci(hu.pvalue)}$ & ${tex_sci(hm.pvalue)}$ {nl}",
        r"    \midrule",
        "    Paired $" + chr(92) + "Delta" + chr(92) + "rho$ (uni $-$ multi) & "
        + chr(92) + "multicolumn{2}{c}{$"
        + f"{d:+.3f}$ [${dlo:+.3f}$, ${dhi:+.3f}$]" + "} " + nl,
        f"    Cities with negative ceiling & {(uni < 0).sum()} & {(mul < 0).sum()} {nl}",
        f"    Four-predictor $R$ & ${Ru:.3f}$ & ${Rm:.3f}$ {nl}",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    (OUT / "gap_dual_target.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (OUT / "gap_dual_target.json").write_text(json.dumps(dict(
        n_cities=n, b_bootstrap=B_BOOT,
        headline=dict(rho_uni=su.statistic, ci_uni=list(cu), p_uni=su.pvalue,
                      rho_multi=sm.statistic, ci_multi=list(cm),
                      p_multi=sm.pvalue,
                      concordant_uni=[ccu, cpu], concordant_multi=[ccm, cpm],
                      paired_difference=dict(mean=d, ci=[dlo, dhi])),
        covariates=cov_rows, subsets=sub_rows,
        four_predictor=dict(uni=Ru, multi=Rm),
        gap_only=dict(uni=gu, multi=gm),
    ), indent=2), encoding="utf-8")
    print("\nwrote outputs/gap_dual_target.json and outputs/gap_dual_target.tex")


if __name__ == "__main__":
    main()
