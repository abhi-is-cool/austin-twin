"""Is arid-city ceiling inversion climatic, or an artifact of the US sample?

The 49-city sweep was entirely American, so "arid" was confounded with
"American" and the inversion result could not distinguish a climatic
mechanism from a continental one. This analyses the extension, in which
every city -- US and not -- is a 25 km disc, so the two groups differ in
geography and nothing else.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
rows = [json.loads(l) for l in (OUT / "global_extension.jsonl").read_text().splitlines() if l.strip()]

def col(k): return np.array([r[k] for r in rows], float)
uni, mul, gap, lst = col("ceiling_r"), col("ceiling_multi_r"), col("veg_imp_lst_gap_c"), col("lst_mean_c")
grp = np.array([r["group"] for r in rows])
us = np.array([g.endswith("_us") for g in grp])
arid = np.array([g.startswith("arid") for g in grp])

print(f"{len(rows)} cities, uniform 25 km disc definition\n")
print(f"{'city':<15} {'group':<16} {'uni':>7} {'multi':>7} {'gap':>7} {'LST':>6} {'n':>3}")
print("-" * 70)
for r in sorted(rows, key=lambda r: r["ceiling_r"]):
    flag = "  INVERTS" if r["ceiling_r"] < 0 else ""
    print(f"{r['city']:<15} {r['group']:<16} {r['ceiling_r']:>+7.3f} "
          f"{r['ceiling_multi_r']:>+7.3f} {r['veg_imp_lst_gap_c']:>+7.2f} "
          f"{r['lst_mean_c']:>6.1f} {r['n_days']:>3}{flag}")

print()
print("=" * 70)
print("IS INVERSION CLIMATIC OR CONTINENTAL?")
print("=" * 70)
for label, m in (("arid, non-US", arid & ~us), ("arid, US", arid & us),
                 ("humid/temperate, non-US", ~arid & ~us),
                 ("humid/temperate, US", ~arid & us)):
    if m.sum() == 0:
        continue
    n_inv = int((uni[m] < 0).sum())
    print(f"  {label:<26} n={int(m.sum()):>2}  inverting {n_inv}/{int(m.sum())}"
          f"  median uni {np.median(uni[m]):+.3f}  median gap {np.median(gap[m]):+.2f}")

inv = [r for r in rows if r["ceiling_r"] < 0]
nonus_inv = [r for r in inv if not r["group"].endswith("_us")]
print(f"\n  inverting cities outside the US: {len(nonus_inv)}")
print("   ", ", ".join(sorted(r["city"] for r in nonus_inv)))

# arid vs humid, pooled across continents
a, h = uni[arid], uni[~arid]
u = stats.mannwhitneyu(a, h, alternative="less")
print(f"\n  arid vs humid/temperate ceilings: median {np.median(a):+.3f} vs "
      f"{np.median(h):+.3f}, Mann-Whitney p = {u.pvalue:.3g}")
# US vs non-US *within* arid: if continental, these should differ
au, an = uni[arid & us], uni[arid & ~us]
if len(au) >= 3 and len(an) >= 3:
    u2 = stats.mannwhitneyu(au, an)
    print(f"  arid US vs arid non-US:          median {np.median(au):+.3f} vs "
          f"{np.median(an):+.3f}, p = {u2.pvalue:.3g}"
          f"  -> {'no continental effect detected' if u2.pvalue > 0.05 else 'differ'}")

print()
print("=" * 70)
print("GAP-CEILING ASSOCIATION ON THIS SAMPLE")
print("=" * 70)
for label, y in (("univariate", uni), ("multivariate", mul)):
    sp = stats.spearmanr(gap, y)
    print(f"  vs {label:<13} rho = {sp.statistic:+.3f}  p = {sp.pvalue:.3g}")
sp = stats.spearmanr(lst, uni)
print(f"  mean LST vs univariate ceiling  rho = {sp.statistic:+.3f}  p = {sp.pvalue:.3g}")

(OUT / "global_extension_analysis.json").write_text(json.dumps({
    "n_cities": len(rows),
    "inverting_nonus": sorted(r["city"] for r in nonus_inv),
    "n_inverting": len(inv),
    "gap_rho_multi": float(stats.spearmanr(gap, mul).statistic),
    "gap_rho_uni": float(stats.spearmanr(gap, uni).statistic),
}, indent=2))
print("\nwrote outputs/global_extension_analysis.json")
