"""One definitive pass: every per-city quantity, from one artifact.

Rationale. The results have been assembled incrementally, and each partial
re-run settled only the numbers it touched. By now the wide sweep and the
channel screen sit on different footings: the gap-ceiling association rests
on 49 cities with multi-day means, while the NDVI channel screen rests on
four cities, and the two were computed by separate scripts with separate
date handling. Any further partial fix risks another inconsistency.

This computes everything in a single pass, per city per clear day, on the
final corrected pipeline (multi-tile land cover, grid-keyed raster caches,
debiased estimators, multi-day averaging), and writes one artifact that all
downstream analysis reads:

    city x day -> univariate ceiling r
                  multivariate ceiling R (impervious, vegetation, water)
                  multivariate ceiling R + NDVI (four channels)
                  built-vegetated LST gap
                  every covariate in ceiling.CeilingResult

Aggregated per city as mean / sd / min / max over its clear days, so every
published figure can carry a spread rather than being a single-day point.

Deliberately uses one shared warm-season pool for all cities. Comparability
across cities is what the wide sweep is for; the transfer-denominator run
(`run_focus_city_ceilings.py`) uses per-city seasonal pools instead, because
there the ceiling must describe the same period as the score it divides.

Resumable: finished cities are appended and skipped on re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ceiling_study import CANDIDATE_DATES, CITIES, MIN_COVERAGE, MIN_DAYS  # noqa: E402
from run_channel_prescreen import fetch_ndvi  # noqa: E402

from austin_twin.ceiling import _multiple_r, ceiling_for_landuse  # noqa: E402
from austin_twin.cities import city_from_point  # noqa: E402
from austin_twin.grid import build_grid, fetch_boundary  # noqa: E402
from austin_twin.modis import fetch_aqua_lst  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
RESULTS = OUT / "full_sweep.jsonl"

# Fields aggregated as mean/sd across a city's clear days.
SPREAD_FIELDS = ["ceiling_r", "ceiling_multi_r", "ceiling_ndvi_r",
                 "ndvi_gain", "veg_imp_lst_gap_c"]


def ndvi_window(date_iso: str, half: int = 8) -> str:
    d = date.fromisoformat(date_iso)
    lo = (d - timedelta(days=half)).isoformat()
    hi = (d + timedelta(days=half)).isoformat()
    return f"{lo}/{hi}"


def load_done() -> set[str]:
    if not RESULTS.exists():
        return set()
    return {json.loads(l)["city"]
            for l in RESULTS.read_text().splitlines() if l.strip()}


def per_day_record(lst, ndvi, landuse):
    """All quantities for one city-day. `ndvi` may be None."""
    res = ceiling_for_landuse(lst, landuse)
    rec = res.to_dict()
    rec["ceiling_ndvi_r"] = float("nan")
    rec["ndvi_gain"] = float("nan")

    if ndvi is None:
        return rec

    mask = landuse["city_mask"].values.astype(bool)
    v = np.isfinite(lst) & mask & np.isfinite(ndvi)
    if int(v.sum()) < 10:
        return rec

    y = lst[v].astype(float)
    imp = landuse["impervious_frac"].values[v].astype(float)
    veg = landuse["vegetation_frac"].values[v].astype(float)
    wat = landuse["water_mask"].values[v].astype(float)
    nd = ndvi[v].astype(float)

    # Recompute the 3-channel R on exactly the NDVI-valid cells so the gain
    # is a like-for-like difference rather than a comparison across masks.
    r3 = _multiple_r(y, np.column_stack([imp, veg, wat]))
    r4 = _multiple_r(y, np.column_stack([imp, veg, wat, nd]))
    rec["ceiling_multi_r_ndvimask"] = r3
    rec["ceiling_ndvi_r"] = r4
    rec["ndvi_gain"] = r4 - r3
    return rec


def aggregate(day_recs: list[dict]) -> dict:
    """Mean of every numeric field, plus sd/min/max for the headline ones."""
    keys = [k for k, v in day_recs[0].items() if isinstance(v, (int, float))]
    agg = {}
    for k in keys:
        vals = [r[k] for r in day_recs
                if k in r and isinstance(r[k], (int, float)) and np.isfinite(r[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    for k in SPREAD_FIELDS:
        vals = [r[k] for r in day_recs
                if k in r and isinstance(r[k], (int, float)) and np.isfinite(r[k])]
        agg[f"{k}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg[f"{k}_min"] = float(np.min(vals)) if vals else float("nan")
        agg[f"{k}_max"] = float(np.max(vals)) if vals else float("nan")
        agg[f"{k}_n"] = len(vals)
    agg["n_days"] = len(day_recs)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-ndvi", action="store_true")
    args = ap.parse_args()

    done = load_done()
    todo = [c for c in CITIES if c[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} cities done, {len(todo)} to run, "
          f"{len(CANDIDATE_DATES)} candidate dates, NDVI={'off' if args.no_ndvi else 'on'}\n")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    for name, query, lat, lon in todo:
        city = city_from_point(name, query, lat, lon)
        try:
            boundary = fetch_boundary(city, cache_path=RAW / f"{name}_boundary.geojson")
            grid = build_grid(boundary, resolution_m=500.0)
            landuse = build_worldcover_landuse(boundary, grid, RAW,
                                               tile=city.worldcover_tile)
            n_city = int(grid.mask.sum())

            day_recs, dates_used, n_ndvi = [], [], 0
            for d in CANDIDATE_DATES:
                try:
                    lst = fetch_aqua_lst(d, grid, RAW / "modis")
                except Exception:
                    continue
                if float(np.isfinite(lst).sum()) / max(n_city, 1) < MIN_COVERAGE:
                    continue

                ndvi = None
                if not args.no_ndvi:
                    try:
                        ndvi = fetch_ndvi(grid, ndvi_window(d), RAW / "modis_ndvi")
                    except Exception:
                        ndvi = None

                rec = per_day_record(lst, ndvi, landuse)
                if np.isfinite(rec.get("ndvi_gain", np.nan)):
                    n_ndvi += 1
                day_recs.append(rec)
                dates_used.append(d)

            if len(day_recs) < MIN_DAYS:
                print(f"  {name:<16} SKIP  {len(day_recs)} clear day(s)")
                continue

            agg = aggregate(day_recs)
            out = dict(city=name, lat=lat, lon=lon, utm=city.utm_crs,
                       tile=city.worldcover_tile, n_city_cells=n_city,
                       dates=dates_used, n_ndvi_days=n_ndvi,
                       per_day={d: {k: r.get(k) for k in SPREAD_FIELDS}
                                for d, r in zip(dates_used, day_recs)},
                       **agg)
            with open(RESULTS, "a") as fh:
                fh.write(json.dumps(out) + "\n")

            gs = agg.get("ndvi_gain", float("nan"))
            print(f"  {name:<16} n={agg['n_days']:>2}d "
                  f"(ndvi {n_ndvi:>2})  "
                  f"uni {agg['ceiling_r']:+.3f}+-{agg['ceiling_r_sd']:.3f}  "
                  f"multi {agg['ceiling_multi_r']:+.3f}+-{agg['ceiling_multi_r_sd']:.3f}  "
                  f"gap {agg['veg_imp_lst_gap_c']:+5.2f}  "
                  f"ndvi gain {gs:+.4f}+-{agg['ndvi_gain_sd']:.4f}")
        except Exception as exc:
            print(f"  {name:<16} FAILED {type(exc).__name__}: {str(exc)[:70]}")
            traceback.print_exc(limit=1)

    print(f"\nresults in {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
