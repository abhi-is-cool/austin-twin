"""Scale the ceiling diagnostic to many cities and test the thermal-gap rule.

paper_v2 asserts the built-vegetated LST gap "rank-orders the ceiling
perfectly across all tested cities" on four points. This runs the same
diagnostic over a large city list so the claim can be tested rather than
asserted, and so the covariate that *predicts* a low ceiling can be
identified by regression rather than by narrative.

No simulator, no calibration, no ERA5 -- just a boundary, land cover and one
MODIS day per city. City configs are derived from centroids by
`cities.city_from_point` (UTM zone and WorldCover tile), which
tests/test_city_config.py pins against the four hand-written configs.

Resumable: each finished city is appended to the output file and skipped on
a re-run, so a long sweep can be interrupted.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from austin_twin.ceiling import ceiling_for_landuse
from austin_twin.cities import city_from_point
from austin_twin.grid import build_grid, fetch_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
RESULTS = OUT / "ceiling_study.jsonl"

# Candidate MODIS days, spread across the northern warm season.
#
# The first version used four July dates and kept the single highest-coverage
# day. That had two problems. Cloud dropout is not independent of the variable
# under study -- humid cities cloud out far more than arid ones -- so 18 of 55
# cities were lost and the surviving sample skewed arid, which is exactly the
# tail driving the gap-ceiling association. And a one-day ceiling is noisy:
# Phoenix moved from +0.363 to -0.149 two weeks apart.
#
# Both are addressed by scanning a wider pool and averaging the ceiling over
# every sufficiently clear day instead of picking one. More dates recovers
# humid cities; averaging gives each city an uncertainty estimate.
CANDIDATE_DATES = [
    "2024-06-16", "2024-06-24", "2024-07-02", "2024-07-08",
    "2024-07-16", "2024-07-24", "2024-08-01", "2024-08-09",
    "2024-08-17", "2024-08-25", "2024-09-02", "2024-09-10",
]
MIN_COVERAGE = 0.60   # per-day, to accept that day into the average
MIN_DAYS = 2          # a city needs at least this many clear days

# name, OSM query, lat, lon. Chosen to span aridity, latitude and city size,
# which is what the thermal-gap hypothesis needs to be tested against.
CITIES = [
    ("phoenix", "Phoenix, Arizona, USA", 33.45, -112.07),
    ("tucson", "Tucson, Arizona, USA", 32.22, -110.97),
    ("las_vegas", "Las Vegas, Nevada, USA", 36.17, -115.14),
    ("albuquerque", "Albuquerque, New Mexico, USA", 35.08, -106.65),
    ("el_paso", "El Paso, Texas, USA", 31.76, -106.49),
    ("denver", "Denver, Colorado, USA", 39.74, -104.99),
    ("salt_lake_city", "Salt Lake City, Utah, USA", 40.76, -111.89),
    ("boise", "Boise, Idaho, USA", 43.62, -116.20),
    ("reno", "Reno, Nevada, USA", 39.53, -119.81),
    ("los_angeles", "Los Angeles, California, USA", 34.05, -118.24),
    ("san_diego", "San Diego, California, USA", 32.72, -117.16),
    ("sacramento", "Sacramento, California, USA", 38.58, -121.49),
    ("fresno", "Fresno, California, USA", 36.74, -119.78),
    ("san_jose", "San Jose, California, USA", 37.34, -121.89),
    ("portland", "Portland, Oregon, USA", 45.52, -122.68),
    ("seattle", "Seattle, Washington, USA", 47.61, -122.33),
    ("spokane", "Spokane, Washington, USA", 47.66, -117.43),
    ("billings", "Billings, Montana, USA", 45.79, -108.55),
    ("dallas", "Dallas, Texas, USA", 32.78, -96.80),
    ("houston", "Houston, Texas, USA", 29.76, -95.37),
    ("san_antonio", "San Antonio, Texas, USA", 29.42, -98.49),
    ("austin", "Austin, Texas, USA", 30.27, -97.74),
    ("oklahoma_city", "Oklahoma City, Oklahoma, USA", 35.47, -97.52),
    ("tulsa", "Tulsa, Oklahoma, USA", 36.15, -95.99),
    ("wichita", "Wichita, Kansas, USA", 37.69, -97.34),
    ("kansas_city", "Kansas City, Missouri, USA", 39.10, -94.58),
    ("omaha", "Omaha, Nebraska, USA", 41.26, -95.93),
    ("des_moines", "Des Moines, Iowa, USA", 41.59, -93.62),
    ("minneapolis", "Minneapolis, Minnesota, USA", 44.98, -93.27),
    ("milwaukee", "Milwaukee, Wisconsin, USA", 43.04, -87.91),
    ("chicago", "Chicago, Illinois, USA", 41.88, -87.63),
    ("indianapolis", "Indianapolis, Indiana, USA", 39.77, -86.16),
    ("detroit", "Detroit, Michigan, USA", 42.33, -83.05),
    ("columbus", "Columbus, Ohio, USA", 39.96, -83.00),
    ("cleveland", "Cleveland, Ohio, USA", 41.50, -81.69),
    ("pittsburgh", "Pittsburgh, Pennsylvania, USA", 40.44, -79.996),
    ("buffalo", "Buffalo, New York, USA", 42.89, -78.88),
    ("st_louis", "St. Louis, Missouri, USA", 38.63, -90.20),
    ("memphis", "Memphis, Tennessee, USA", 35.15, -90.05),
    ("nashville", "Nashville, Tennessee, USA", 36.16, -86.78),
    ("louisville", "Louisville, Kentucky, USA", 38.25, -85.76),
    ("atlanta", "Atlanta, Georgia, USA", 33.75, -84.39),
    ("charlotte", "Charlotte, North Carolina, USA", 35.23, -80.84),
    ("raleigh", "Raleigh, North Carolina, USA", 35.78, -78.64),
    ("richmond", "Richmond, Virginia, USA", 37.54, -77.44),
    ("baltimore", "Baltimore, Maryland, USA", 39.29, -76.61),
    ("philadelphia", "Philadelphia, Pennsylvania, USA", 39.95, -75.17),
    ("boston", "Boston, Massachusetts, USA", 42.36, -71.06),
    ("jacksonville", "Jacksonville, Florida, USA", 30.33, -81.66),
    ("orlando", "Orlando, Florida, USA", 28.54, -81.38),
    ("tampa", "Tampa, Florida, USA", 27.95, -82.46),
    ("new_orleans", "New Orleans, Louisiana, USA", 29.95, -90.07),
    ("birmingham", "Birmingham, Alabama, USA", 33.52, -86.80),
    ("little_rock", "Little Rock, Arkansas, USA", 34.75, -92.29),
    ("bakersfield", "Bakersfield, California, USA", 35.37, -119.02),
]


def load_done() -> dict:
    if not RESULTS.exists():
        return {}
    out = {}
    for line in RESULTS.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["city"]] = rec
    return out


def clear_days(city, grid):
    """Every candidate day clearing MIN_COVERAGE, as (date, lst) pairs."""
    n_city = int(grid.mask.sum())
    out, best_cov = [], 0.0
    for d in CANDIDATE_DATES:
        try:
            lst = fetch_aqua_lst(d, grid, RAW / "modis")
        except Exception:
            continue
        cov = float(np.isfinite(lst).sum()) / max(n_city, 1)
        best_cov = max(best_cov, cov)
        if cov >= MIN_COVERAGE:
            out.append((d, lst))
    return out, best_cov


def aggregate(results):
    """Mean of each numeric field across days, plus spread on the ceiling.

    Averaging is what turns a noisy single-day number into an estimate with
    an uncertainty attached, so the ceiling can be quoted with a standard
    deviation and a range rather than as a bare point value.
    """
    keys = [k for k, v in results[0].to_dict().items()
            if isinstance(v, (int, float))]
    agg = {}
    for k in keys:
        vals = [getattr(r, k) for r in results]
        vals = [v for v in vals if np.isfinite(v)]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    ceilings = [r.ceiling_r for r in results if np.isfinite(r.ceiling_r)]
    agg["ceiling_r_sd"] = (float(np.std(ceilings, ddof=1))
                           if len(ceilings) > 1 else 0.0)
    agg["ceiling_r_min"] = float(np.min(ceilings)) if ceilings else float("nan")
    agg["ceiling_r_max"] = float(np.max(ceilings)) if ceilings else float("nan")
    agg["n_days"] = len(results)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    done = load_done()
    todo = [c for c in CITIES if c[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} cities already done, {len(todo)} to run\n")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    for name, query, lat, lon in todo:
        city = city_from_point(name, query, lat, lon)
        try:
            boundary = fetch_boundary(city, cache_path=RAW / f"{name}_boundary.geojson")
            grid = build_grid(boundary, resolution_m=500.0)
            landuse = build_worldcover_landuse(boundary, grid, RAW,
                                               tile=city.worldcover_tile)
            days, best_cov = clear_days(city, grid)
            if len(days) < MIN_DAYS:
                print(f"  {name:<16} SKIP  {len(days)} clear day(s) "
                      f"(best coverage {best_cov:.0%})")
                continue
            per_day = [ceiling_for_landuse(lst, landuse) for _, lst in days]
            agg = aggregate(per_day)
            rec = dict(city=name, lat=lat, lon=lon,
                       dates=[d for d, _ in days],
                       per_day_ceiling={d: float(r.ceiling_r)
                                        for (d, _), r in zip(days, per_day)},
                       utm=city.utm_crs, tile=city.worldcover_tile,
                       n_city_cells=int(grid.mask.sum()), **agg)
            with open(RESULTS, "a") as fh:
                fh.write(json.dumps(rec) + chr(10))
            print(f"  {name:<16} n={agg['n_days']:>2}d  "
                  f"ceiling {agg['ceiling_r']:+.3f} +-{agg['ceiling_r_sd']:.3f} "
                  f"[{agg['ceiling_r_min']:+.3f},{agg['ceiling_r_max']:+.3f}]  "
                  f"multi {agg['ceiling_multi_r']:+.3f}  "
                  f"gap {agg['veg_imp_lst_gap_c']:+5.2f} C")
        except Exception as exc:
            print(f"  {name:<16} FAILED {type(exc).__name__}: {str(exc)[:70]}")
            traceback.print_exc(limit=1)

    print(f"\nresults in {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
