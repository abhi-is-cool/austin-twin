"""Does the 49-city gap heuristic survive a receptive-field-matched ceiling?

The published headline is Spearman rho = +0.69 between the built--vegetated
thermal gap and the multivariate ceiling across 49 US cities. That ceiling
is a pixel-scale quantity, and the global sample showed pixel scale is the
wrong scale for arid cities. So the question is whether the heuristic
predicts recoverable signal, or only predicts signal recoverable
per-pixel.

Reports, for each ceiling specification:
  * gap vs ceiling, and aridity vs ceiling
  * whether any city is data-limited
  * a paired bootstrap on the difference in rho, since the two rhos are
    computed on the same 49 cities and their marginal intervals overlap
    in ways that say nothing about the difference
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SRC = OUT / "sweep_receptive_field.jsonl"
RESULT = OUT / "sweep_receptive_field_analysis.json"

N_BOOT = 5000
SEED = 20260921


def sp(a, b):
    r, p = stats.spearmanr(a, b)
    return float(r), float(p)


def paired_rho_ci(x, a, b, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI for rho(x,a) - rho(x,b), resampling cities together."""
    rng = np.random.default_rng(seed)
    n = len(x)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0:
            d[i] = np.nan
            continue
        d[i] = (stats.spearmanr(x[idx], a[idx]).statistic
                - stats.spearmanr(x[idx], b[idx]).statistic)
    d = d[np.isfinite(d)]
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(np.mean(d < 0)))


def main() -> None:
    recs = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    keys = ("linear", "per_cell", "context", "gain", "lst_mean_c",
            "published_uni", "published_multi", "veg_imp_lst_gap_c")
    g = {k: np.array([r[k] for r in recs], dtype=float) for k in keys}
    names = [r["city"] for r in recs]
    res = {"n_cities": len(recs)}
    print(f"{len(recs)} US cities\n")

    print("=== The headline association, under each specification ===")
    for tgt, lab in (("published_multi", "published multi"),
                     ("linear", "linear (recomputed)"),
                     ("per_cell", "nonlinear per-cell"),
                     ("context", "nonlinear + context")):
        r, p = sp(g["veg_imp_lst_gap_c"], g[tgt])
        print(f"  gap vs {lab:<22} rho = {r:+.3f}  p = {p:.2g}")
        res[f"gap_vs_{tgt}"] = [r, p]

    lo, hi, pneg = paired_rho_ci(g["veg_imp_lst_gap_c"], g["context"],
                                 g["published_multi"])
    d = res["gap_vs_context"][0] - res["gap_vs_published_multi"][0]
    print(f"\n  paired difference (context - published): {d:+.3f}"
          f"  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  -> {'weakens significantly' if hi < 0 else 'no significant change'}"
          f"  (P(diff<0) = {pneg:.3f})")
    res["paired_rho_diff"] = dict(delta=d, ci=[lo, hi], p_negative=pneg)

    print("\n=== Aridity against the ceiling ===")
    for tgt in ("published_multi", "per_cell", "context"):
        r, p = sp(g["lst_mean_c"], g[tgt])
        print(f"  mean LST vs {tgt:<18} rho = {r:+.3f}  p = {p:.2g}")
        res[f"aridity_vs_{tgt}"] = [r, p]

    print("\n=== Is any US city data-limited under a matched ceiling? ===")
    order = np.argsort(g["context"])
    for i in order[:6]:
        print(f"  {names[i]:<16} context {g['context'][i]:+.3f}"
              f"   (per-cell {g['per_cell'][i]:+.3f},"
              f" published {g['published_multi'][i]:+.3f})")
    res["context_min"] = float(np.nanmin(g["context"]))
    res["context_median"] = float(np.nanmedian(g["context"]))
    res["n_negative_published"] = int((g["published_uni"] < 0).sum())
    res["n_negative_context"] = int((g["context"] < 0).sum())
    print(f"  lowest {np.nanmin(g['context']):+.3f}, "
          f"median {np.nanmedian(g['context']):+.3f}")
    print(f"  cities with negative univariate ceiling: "
          f"{res['n_negative_published']}")
    print(f"  cities with negative context ceiling   : "
          f"{res['n_negative_context']}")

    print("\n=== Gain structure ===")
    r, p = sp(g["gain"], g["per_cell"])
    print(f"  gain vs per-cell ceiling  rho = {r:+.3f}  p = {p:.2g}")
    res["gain_vs_percell"] = [r, p]
    r, p = sp(g["gain"], g["lst_mean_c"])
    print(f"  gain vs mean LST          rho = {r:+.3f}  p = {p:.2g}")
    res["gain_vs_aridity"] = [r, p]
    med = float(np.median(g["lst_mean_c"]))
    hot, cool = g["lst_mean_c"] >= med, g["lst_mean_c"] < med
    u, p = stats.mannwhitneyu(g["gain"][hot], g["gain"][cool])
    print(f"  hot half median gain {np.median(g['gain'][hot]):+.3f} vs "
          f"cool half {np.median(g['gain'][cool]):+.3f}   p = {p:.3g}")
    res["gain_hot_vs_cool_p"] = float(p)
    print(f"  spread across cities: per-cell "
          f"{np.nanmax(g['per_cell']) - np.nanmin(g['per_cell']):.3f}"
          f" -> context {np.nanmax(g['context']) - np.nanmin(g['context']):.3f}")

    RESULT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
