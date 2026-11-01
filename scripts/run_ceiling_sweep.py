"""Compute the land-cover data ceiling across cities.

Two modes:

  --validate  (default) recompute the ceilings recorded in NUMBERS.md section 14
              for the four study cities on their published dates. This is the
              correctness gate: if the sweep cannot reproduce +0.671 / +0.363 /
              +0.131 / +0.827, it cannot be trusted on unseen cities.

  --cities / --dates  run the same diagnostic over an arbitrary city list.

The diagnostic needs no simulator, no calibration and no ERA5 — only a
boundary, a land-cover raster and one MODIS day per city — so it scales to a
large city list far more cheaply than the transfer experiment does.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from austin_twin.ceiling import ceiling_for_landuse
from austin_twin.cities import CITIES, CityConfig, city_from_point
from austin_twin.grid import build_grid, fetch_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

# NUMBERS.md section 14: ceiling = r(MODIS LST, impervious_frac) on that date.
PUBLISHED = {
    "austin": ("2024-08-22", +0.671),
    "phoenix": ("2024-07-22", +0.363),
    "denver": ("2024-07-25", +0.131),
    "miami": ("2024-04-16", +0.827),
}


def ceiling_for_city(city: CityConfig, date_iso: str, resolution_m: float = 500.0):
    boundary = fetch_boundary(
        city, cache_path=RAW / f"{city.name}_boundary.geojson"
    )
    grid = build_grid(boundary, resolution_m=resolution_m)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=city.worldcover_tile)
    lst = fetch_aqua_lst(date_iso, grid, RAW / "modis")
    res = ceiling_for_landuse(lst, landuse)
    return res, grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", default=True)
    ap.add_argument("--resolution", type=float, default=500.0)
    ap.add_argument("--cities", nargs="*", default=None,
                    help="name:osm_query:lat:lon quadruples for an ad-hoc sweep")
    ap.add_argument("--dates", nargs="*", default=None,
                    help="one MODIS date per --cities entry")
    args = ap.parse_args()

    rows = []

    if args.cities:
        jobs = []
        for spec, date_iso in zip(args.cities, args.dates or []):
            name, query, lat, lon = spec.split(":")
            jobs.append((city_from_point(name, query, float(lat), float(lon)),
                         date_iso, None))
    else:
        jobs = [(CITIES[n], d, ref) for n, (d, ref) in PUBLISHED.items()]

    for city, date_iso, ref in jobs:
        print(f"\n=== {city.name}  {date_iso}  "
              f"({city.utm_crs}, tile {city.worldcover_tile}) ===")
        try:
            res, grid = ceiling_for_city(city, date_iso, args.resolution)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:160]}")
            continue

        ny, nx = grid.shape
        print(f"  grid {ny}x{nx}, {int(grid.mask.sum())} city cells, "
              f"coverage {res.coverage:.1%} ({res.n_valid} valid)")
        print(f"  ceiling r (uni)    {res.ceiling_r:+.4f}"
              + (f"   published {ref:+.3f}   delta {res.ceiling_r - ref:+.4f}"
                 if ref is not None else ""))
        print(f"  ceiling R (3-chan) {res.ceiling_multi_r:+.4f}")
        print(f"  LST mean / spread  {res.lst_mean_c:.1f} C / "
              f"{res.lst_spread_c:.1f} C")
        print(f"  impervious mean/sd {res.impervious_mean:.3f} / "
              f"{res.impervious_std:.3f}")
        print(f"  vegetation / water {res.vegetation_mean:.3f} / "
              f"{res.water_mean:.3f}")
        print(f"  class entropy      {res.class_entropy:.3f} bits")
        print(f"  built - vegetated  {res.veg_imp_lst_gap_c:+.2f} C")

        row = {"city": city.name, "date": date_iso,
               "published_ceiling": ref, **res.to_dict()}
        rows.append(row)

    if jobs and all(r.get("published_ceiling") is not None for r in rows):
        print("\n" + "=" * 72)
        print("VALIDATION vs NUMBERS.md section 14")
        print("=" * 72)
        ok = True
        for r in rows:
            d = r["ceiling_r"] - r["published_ceiling"]
            # Published values are quoted to 3 decimals; 0.01 allows for that
            # plus any boundary/version drift in the inputs.
            good = abs(d) < 0.01
            ok &= good
            print(f"  {r['city']:<10} {r['ceiling_r']:+.4f} vs "
                  f"{r['published_ceiling']:+.3f}   delta {d:+.4f}   "
                  f"{'MATCH' if good else 'MISMATCH'}")
        print(f"\n  RESULT: {'CEILING SWEEP VALIDATED' if ok else 'MISMATCH'}")

    out = OUT / "ceiling_sweep.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
