"""Does the arid inversion survive a receptive-field-matched ceiling?

Tests three things on the 28-city global sample:

  1. Is the receptive-field gain concentrated in arid cities, as the four
     focus cities suggested?
  2. Do the cities whose univariate ceiling is *negative* -- the inversion
     cases the paper is built on -- recover a positive context ceiling?
  3. Does the built--vegetated thermal-gap heuristic still predict the
     ceiling once the ceiling can see neighbourhoods, or is the heuristic
     itself scoped to pixel-scale models?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SRC = OUT / "arid_receptive_field.jsonl"
RESULT = OUT / "arid_receptive_field_analysis.json"

ARID = {"arid_nonus", "arid_us"}


def sp(a, b):
    r, p = stats.spearmanr(a, b)
    return float(r), float(p)


def main() -> None:
    recs = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    g = {k: np.array([r[k] for r in recs], dtype=float) for k in
         ("linear", "per_cell", "context", "gain", "lst_mean_c",
          "published_uni", "veg_imp_lst_gap_c")}
    arid = np.array([r["group"] in ARID for r in recs])
    names = [r["city"] for r in recs]
    res = {"n_cities": len(recs), "n_arid": int(arid.sum())}

    print(f"{len(recs)} cities, {int(arid.sum())} arid\n")

    print("=== 1. Is the gain concentrated in arid cities? ===")
    r, p = sp(g["gain"], g["lst_mean_c"])
    print(f"  gain vs mean LST (aridity proxy)   rho = {r:+.3f}  p = {p:.2g}")
    res["gain_vs_aridity"] = [r, p]
    r2, p2 = sp(g["gain"], g["per_cell"])
    print(f"  gain vs per-cell ceiling           rho = {r2:+.3f}  p = {p2:.2g}")
    res["gain_vs_percell"] = [r2, p2]
    u, pu = stats.mannwhitneyu(g["gain"][arid], g["gain"][~arid])
    print(f"  arid median gain {np.median(g['gain'][arid]):+.3f}  vs "
          f"non-arid {np.median(g['gain'][~arid]):+.3f}   "
          f"Mann-Whitney p = {pu:.3g}")
    res["gain_arid_vs_other_p"] = float(pu)

    print("\n=== 2. Do the inverting cities recover? ===")
    inv = g["published_uni"] < 0
    print(f"  {int(inv.sum())} cities have a negative univariate ceiling")
    print(f"  {'city':<15}{'uni':>8}{'perCell':>9}{'context':>9}")
    for i in np.where(inv)[0]:
        print(f"  {names[i]:<15}{g['published_uni'][i]:>+8.3f}"
              f"{g['per_cell'][i]:>+9.3f}{g['context'][i]:>+9.3f}")
    res["inverting"] = dict(
        n=int(inv.sum()),
        min_context=float(g["context"][inv].min()),
        median_context=float(np.median(g["context"][inv])),
    )
    print(f"  -> every inverting city has context ceiling >= "
          f"{g['context'][inv].min():+.3f}")

    print("\n=== Is ANY city data-limited under a matched ceiling? ===")
    order = np.argsort(g["context"])
    for i in order[:5]:
        print(f"  {names[i]:<15} context {g['context'][i]:+.3f}"
              f"   (per-cell {g['per_cell'][i]:+.3f})")
    res["context_min"] = float(g["context"].min())
    res["context_median"] = float(np.median(g["context"]))
    print(f"  lowest context ceiling in the sample: {g['context'].min():+.3f}")
    print(f"  median: {np.median(g['context']):+.3f}")
    spread_pc = float(g["per_cell"].max() - g["per_cell"].min())
    spread_cx = float(g["context"].max() - g["context"].min())
    print(f"  spread across cities: per-cell {spread_pc:.3f} -> "
          f"context {spread_cx:.3f}")
    res["spread_per_cell"], res["spread_context"] = spread_pc, spread_cx

    print("\n=== 3. Does the thermal-gap heuristic survive? ===")
    for tgt in ("linear", "per_cell", "context"):
        r, p = sp(g["veg_imp_lst_gap_c"], g[tgt])
        print(f"  gap vs {tgt:<9} ceiling   rho = {r:+.3f}  p = {p:.2g}")
        res[f"gap_vs_{tgt}"] = [r, p]

    print("\n=== Does aridity still predict the ceiling? ===")
    for tgt in ("linear", "per_cell", "context"):
        r, p = sp(g["lst_mean_c"], g[tgt])
        print(f"  mean LST vs {tgt:<9} ceiling   rho = {r:+.3f}  p = {p:.2g}")
        res[f"aridity_vs_{tgt}"] = [r, p]
    for tgt in ("per_cell", "context"):
        u, p = stats.mannwhitneyu(g[tgt][arid], g[tgt][~arid])
        print(f"  arid vs non-arid {tgt:<9}  median "
              f"{np.median(g[tgt][arid]):+.3f} vs {np.median(g[tgt][~arid]):+.3f}"
              f"   p = {p:.3g}")
        res[f"arid_vs_other_{tgt}_p"] = float(p)

    RESULT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
