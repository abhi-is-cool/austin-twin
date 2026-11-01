"""Multi-day ceilings for the four transfer focus cities, both definitions.

paper_v3 states that the three-channel ceiling behind the transfer table
"has not been independently re-verified against a multi-day pool", and that
there is no mechanistic reason to expect it is more stable than the
univariate one. This closes that caveat.

Unlike the 49-city sweep, which uses one shared summer pool, each focus city
gets a pool matching the season of its own transfer test days -- Austin and
Phoenix in summer, Denver in summer, Miami in spring. The ceiling is the
denominator of a "percent of ceiling" statement, so it has to describe the
same period as the numerator; scoring an April transfer against a July
ceiling would compare different atmospheres.

Reports both ceiling definitions as multi-day means with spread, and
recomputes each city's percent-of-ceiling against the multivariate figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from austin_twin.ceiling import ceiling_for_landuse
from austin_twin.cities import CITIES
from austin_twin.grid import build_grid, fetch_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

MIN_COVERAGE = 0.60


def _days(prefix: str, days: list[int]) -> list[str]:
    return [f"{prefix}-{d:02d}" for d in days]


# Pools centred on each city's own transfer window (see bootstrap_uncertainty:
# Austin Aug 21/22, Phoenix Jul 22/24, Denver Jul, Miami Apr 16/19).
POOLS = {
    "austin": _days("2024-08", [3, 6, 9, 11, 14, 16, 19, 21, 22, 25, 28, 31]),
    "phoenix": _days("2024-07", [2, 5, 8, 11, 14, 17, 20, 22, 24, 27, 29, 31]),
    "denver": _days("2024-07", [2, 5, 8, 11, 14, 17, 20, 23, 25, 28, 30]),
    "miami": _days("2024-04", [2, 5, 8, 11, 14, 16, 19, 22, 25, 28, 30]),
}

# Published single-day ceilings and best transfer scores (paper Table IV).
PUBLISHED = {
    "austin": dict(uni=0.671, multi=0.704, best_model=None),
    "phoenix": dict(uni=0.363, multi=0.412, best_model=0.083),
    "denver": dict(uni=0.131, multi=0.395, best_model=0.004),
    "miami": dict(uni=0.827, multi=0.843, best_model=0.832),
}


def main() -> None:
    rows = []
    for name, dates in POOLS.items():
        city = CITIES[name]
        print(f"\n=== {name} ({len(dates)} candidate days, "
              f"{dates[0][:7]}) ===")
        boundary = fetch_boundary(city, cache_path=RAW / f"{name}_boundary.geojson")
        grid = build_grid(boundary, resolution_m=500.0)
        landuse = build_worldcover_landuse(boundary, grid, RAW,
                                           tile=city.worldcover_tile)
        n_city = int(grid.mask.sum())

        per_day = []
        for d in dates:
            try:
                lst = fetch_aqua_lst(d, grid, RAW / "modis")
            except Exception as exc:
                print(f"    {d}  fetch failed: {type(exc).__name__}")
                continue
            cov = float(np.isfinite(lst).sum()) / max(n_city, 1)
            if cov < MIN_COVERAGE:
                continue
            res = ceiling_for_landuse(lst, landuse)
            per_day.append((d, res))
            print(f"    {d}  cov {cov:>4.0%}   uni {res.ceiling_r:+.3f}   "
                  f"multi {res.ceiling_multi_r:+.3f}")

        if len(per_day) < 2:
            print(f"    too few clear days ({len(per_day)})")
            continue

        uni = np.array([r.ceiling_r for _, r in per_day])
        mul = np.array([r.ceiling_multi_r for _, r in per_day])
        gap = np.array([r.veg_imp_lst_gap_c for _, r in per_day])
        rows.append(dict(
            city=name, n_days=len(per_day), dates=[d for d, _ in per_day],
            uni_mean=float(uni.mean()), uni_sd=float(uni.std(ddof=1)),
            uni_min=float(uni.min()), uni_max=float(uni.max()),
            multi_mean=float(mul.mean()), multi_sd=float(mul.std(ddof=1)),
            multi_min=float(mul.min()), multi_max=float(mul.max()),
            gap_mean=float(gap.mean()),
            per_day={d: dict(uni=float(r.ceiling_r), multi=float(r.ceiling_multi_r))
                     for d, r in per_day},
        ))

    # ---------------- summary ----------------
    print()
    print("=" * 96)
    print("MULTI-DAY CEILINGS FOR THE FOUR FOCUS CITIES")
    print("=" * 96)
    print(f"{'city':<9} {'n':>3} | {'uni published':>13} {'uni multi-day':>21} | "
          f"{'multi published':>15} {'multi multi-day':>21}")
    print("-" * 96)
    for r in rows:
        pub = PUBLISHED[r["city"]]
        print(f"{r['city']:<9} {r['n_days']:>3} | {pub['uni']:>+13.3f} "
              f"{r['uni_mean']:>+9.3f} +-{r['uni_sd']:.3f} "
              f"[{r['uni_min']:+.2f},{r['uni_max']:+.2f}] | "
              f"{pub['multi']:>+15.3f} "
              f"{r['multi_mean']:>+9.3f} +-{r['multi_sd']:.3f} "
              f"[{r['multi_min']:+.2f},{r['multi_max']:+.2f}]")

    print()
    print("=" * 96)
    print("PERCENT OF CEILING, RECOMPUTED")
    print("=" * 96)
    print(f"{'city':<9} {'model':>8} | {'vs published multi':>19} "
          f"{'vs multi-day multi':>19}  {'shift':>8}")
    print("-" * 96)
    for r in rows:
        pub = PUBLISHED[r["city"]]
        if pub["best_model"] is None:
            continue
        a = pub["best_model"] / pub["multi"]
        b = pub["best_model"] / r["multi_mean"] if r["multi_mean"] != 0 else float("nan")
        r["pct_published"] = a
        r["pct_multiday"] = b
        print(f"{r['city']:<9} {pub['best_model']:>+8.3f} | {a:>18.0%} "
              f"{b:>18.0%}  {b - a:>+8.0%}")

    print()
    print("stability comparison (sd of the daily estimate):")
    for r in rows:
        ratio = (r["multi_sd"] / r["uni_sd"]) if r["uni_sd"] > 0 else float("nan")
        verdict = ("more stable" if ratio < 0.9 else
                   "less stable" if ratio > 1.1 else "comparable")
        print(f"  {r['city']:<9} uni sd {r['uni_sd']:.3f}   "
              f"multi sd {r['multi_sd']:.3f}   ratio {ratio:.2f}  ({verdict})")

    (OUT / "focus_city_ceilings.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote outputs/focus_city_ceilings.json")


if __name__ == "__main__":
    main()
