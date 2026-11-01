"""Extend the ceiling sweep beyond the US, to test whether inversion is climatic.

Every city -- US and non-US alike -- is defined as a 25 km-radius disc around
its centre rather than by its OSM administrative boundary, because those are
not comparable across countries (see scripts/disc_boundary.py). US cities are
included under the same definition as a comparability bridge, so any
US/non-US difference cannot be an artifact of how a "city" was delimited.

The 49-city sample is entirely American, so "arid" in it means "American
Southwest" and the arid-inversion result cannot distinguish a climatic
effect from a continental one. Adding a sixth Southwest city would not help;
adding arid cities on other continents is the test.

Three groups, chosen so the comparison is interpretable:

  * ARID NON-US -- Middle East, North Africa, Australia, the Atacama,
    southern Africa, central and south Asia. If inversion is climatic it
    should appear here too.
  * ARID US (extreme) -- Yuma, Palm Springs, Las Cruces, drier than any city
    already in the sample. Tests whether inversion deepens with aridity
    within one continent.
  * HUMID / TEMPERATE NON-US -- controls. Without them the extension only
    adds arid points and the gap-ceiling association could not be retested
    on the new sample.

Everything is hemisphere-aware: southern-hemisphere cities get an austral
summer pool, so every city is sampled in its own warm season. No ERA5 and
no simulator are involved -- the ceiling is a property of the data.

Writes `global_extension.jsonl` in the same schema as `full_sweep.jsonl`,
so the two can be concatenated for analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_full_sweep import (  # noqa: E402
    SPREAD_FIELDS, aggregate, ndvi_window, per_day_record,
)
from run_ceiling_study import MIN_COVERAGE, MIN_DAYS  # noqa: E402
from run_channel_prescreen import fetch_ndvi  # noqa: E402

from austin_twin.cities import city_from_point  # noqa: E402
from disc_boundary import DEFAULT_RADIUS_KM, disc_boundary  # noqa: E402
from austin_twin.grid import build_grid  # noqa: E402
from austin_twin.modis import fetch_aqua_lst  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
RESULTS = OUT / "global_extension.jsonl"

# Boreal summer, matching the US sweep exactly.
NORTH_DATES = [
    "2024-06-16", "2024-06-24", "2024-07-02", "2024-07-08",
    "2024-07-16", "2024-07-24", "2024-08-01", "2024-08-09",
    "2024-08-17", "2024-08-25", "2024-09-02", "2024-09-10",
]
# Austral summer, the same 12-date spacing shifted six months.
SOUTH_DATES = [
    "2024-12-16", "2024-12-24", "2025-01-02", "2025-01-08",
    "2025-01-16", "2025-01-24", "2025-02-01", "2025-02-09",
    "2025-02-17", "2025-02-25", "2025-03-02", "2025-03-10",
]

# name, OSM query, lat, lon, group
CITIES = [
    # --- arid, outside the US ---
    ("riyadh", "Riyadh, Saudi Arabia", 24.71, 46.68, "arid_nonus"),
    ("dubai", "Dubai, United Arab Emirates", 25.20, 55.27, "arid_nonus"),
    ("cairo", "Cairo, Egypt", 30.04, 31.24, "arid_nonus"),
    ("kuwait_city", "Kuwait City, Kuwait", 29.38, 47.98, "arid_nonus"),
    ("amman", "Amman, Jordan", 31.95, 35.93, "arid_nonus"),
    ("marrakesh", "Marrakesh, Morocco", 31.63, -7.99, "arid_nonus"),
    ("jodhpur", "Jodhpur, Rajasthan, India", 26.29, 73.03, "arid_nonus"),
    ("tashkent", "Tashkent, Uzbekistan", 41.30, 69.24, "arid_nonus"),
    ("hermosillo", "Hermosillo, Sonora, Mexico", 29.07, -110.96, "arid_nonus"),
    ("antofagasta", "Antofagasta, Chile", -23.65, -70.40, "arid_nonus"),
    ("alice_springs", "Alice Springs, Northern Territory, Australia",
     -23.70, 133.88, "arid_nonus"),
    ("perth", "Perth, Western Australia, Australia", -31.95, 115.86, "arid_nonus"),
    ("windhoek", "Windhoek, Namibia", -22.56, 17.08, "arid_nonus"),
    ("mendoza", "Mendoza, Argentina", -32.89, -68.84, "arid_nonus"),
    # --- extreme-arid US: does inversion deepen within one continent? ---
    ("yuma", "Yuma, Arizona, USA", 32.69, -114.63, "arid_us"),
    ("palm_springs", "Palm Springs, California, USA", 33.83, -116.55, "arid_us"),
    ("las_cruces", "Las Cruces, New Mexico, USA", 32.32, -106.76, "arid_us"),
    # --- humid / temperate controls, outside the US ---
    ("madrid", "Madrid, Spain", 40.42, -3.70, "temperate_nonus"),
    ("milan", "Milan, Italy", 45.46, 9.19, "temperate_nonus"),
    ("kuala_lumpur", "Kuala Lumpur, Malaysia", 3.14, 101.69, "humid_nonus"),
    ("bangkok", "Bangkok, Thailand", 13.76, 100.50, "humid_nonus"),
    ("manila", "Manila, Philippines", 14.60, 120.98, "humid_nonus"),
    ("lagos", "Lagos, Nigeria", 6.52, 3.38, "humid_nonus"),
    ("sao_paulo", "São Paulo, Brazil", -23.55, -46.63, "humid_nonus"),
    ("sydney", "Sydney, New South Wales, Australia", -33.87, 151.21, "temperate_nonus"),
    # --- US bridge: same disc definition, so the two samples are directly
    # comparable. Includes all five cities that invert under the OSM
    # definition, plus humid/temperate US controls. ---
    ("phoenix_d", "Phoenix, Arizona, USA", 33.45, -112.07, "arid_us"),
    ("tucson_d", "Tucson, Arizona, USA", 32.22, -110.97, "arid_us"),
    ("albuquerque_d", "Albuquerque, New Mexico, USA", 35.08, -106.65, "arid_us"),
    ("boise_d", "Boise, Idaho, USA", 43.62, -116.20, "arid_us"),
    ("billings_d", "Billings, Montana, USA", 45.79, -108.55, "arid_us"),
    ("austin_d", "Austin, Texas, USA", 30.27, -97.74, "temperate_us"),
    ("houston_d", "Houston, Texas, USA", 29.76, -95.37, "humid_us"),
    ("atlanta_d", "Atlanta, Georgia, USA", 33.75, -84.39, "humid_us"),
    ("chicago_d", "Chicago, Illinois, USA", 41.88, -87.63, "humid_us"),
    ("seattle_d", "Seattle, Washington, USA", 47.61, -122.33, "temperate_us"),
]


def load_done() -> set[str]:
    if not RESULTS.exists():
        return set()
    return {json.loads(l)["city"] for l in RESULTS.read_text().splitlines() if l.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    done = load_done()
    todo = [c for c in CITIES if c[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} done, {len(todo)} to run\n")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    for name, query, lat, lon, group in todo:
        dates = NORTH_DATES if lat >= 0 else SOUTH_DATES
        season = "boreal" if lat >= 0 else "austral"
        city = city_from_point(name, query, lat, lon)
        try:
            boundary = disc_boundary(lat, lon, city.utm_crs)
            grid = build_grid(boundary, resolution_m=500.0)
            landuse = build_worldcover_landuse(boundary, grid, RAW,
                                               tile=city.worldcover_tile)
            n_city = int(grid.mask.sum())

            recs, used, n_ndvi = [], [], 0
            for d in dates:
                try:
                    lst = fetch_aqua_lst(d, grid, RAW / "modis")
                except Exception:
                    continue
                if float(np.isfinite(lst).sum()) / max(n_city, 1) < MIN_COVERAGE:
                    continue
                try:
                    ndvi = fetch_ndvi(grid, ndvi_window(d), RAW / "modis_ndvi")
                except Exception:
                    ndvi = None
                rec = per_day_record(lst, ndvi, landuse)
                if np.isfinite(rec.get("ndvi_gain", np.nan)):
                    n_ndvi += 1
                recs.append(rec)
                used.append(d)

            if len(recs) < MIN_DAYS:
                print(f"  {name:<15} SKIP  {len(recs)} clear day(s) [{season}]")
                continue

            agg = aggregate(recs)
            out = dict(city=name, lat=lat, lon=lon, group=group, season=season,
                       boundary="disc", radius_km=DEFAULT_RADIUS_KM,
                       utm=city.utm_crs, tile=city.worldcover_tile,
                       n_city_cells=n_city, dates=used, n_ndvi_days=n_ndvi,
                       per_day={d: {k: r.get(k) for k in SPREAD_FIELDS}
                                for d, r in zip(used, recs)},
                       **agg)
            with open(RESULTS, "a") as fh:
                fh.write(json.dumps(out) + "\n")

            inv = "  INVERTS" if agg["ceiling_r"] < 0 else ""
            print(f"  {name:<15} {group:<16} n={agg['n_days']:>2}d  "
                  f"uni {agg['ceiling_r']:+.3f}  multi {agg['ceiling_multi_r']:+.3f}  "
                  f"gap {agg['veg_imp_lst_gap_c']:+5.2f}  "
                  f"LST {agg['lst_mean_c']:5.1f}{inv}")
        except Exception as exc:
            print(f"  {name:<15} FAILED {type(exc).__name__}: {str(exc)[:70]}")
            traceback.print_exc(limit=1)

    print(f"\nresults in {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
