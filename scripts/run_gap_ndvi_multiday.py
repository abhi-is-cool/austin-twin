"""Multi-day built-vegetated gap and multi-day NDVI pre-screen.

Two figures in the paper are still single-day while everything around them
has moved to multi-day means:

  * the built-vegetated LST gap quoted per focus city (Section III-C
    bullets), where Denver's value even changes sign once averaged; and
  * the NDVI channel pre-screen, whose whole claim is that a candidate
    channel can be evaluated cheaply -- which is only worth believing if the
    verdict is stable across days.

Both are recomputed here over the same per-city seasonal pools used by
`run_focus_city_ceilings.py`, in a single pass so each clear day is fetched
once and used for both.

For each clear day the NDVI composite covering that date is fetched (MOD13A1
is a 16-day product, so nearby dates often share one composite and the asset
cache absorbs the repeats).
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_channel_prescreen import fetch_ndvi  # noqa: E402
from run_focus_city_ceilings import POOLS, MIN_COVERAGE  # noqa: E402

from austin_twin.ceiling import _multiple_r, ceiling_for_landuse  # noqa: E402
from austin_twin.cities import CITIES  # noqa: E402
from austin_twin.grid import build_grid, fetch_boundary  # noqa: E402
from austin_twin.modis import fetch_aqua_lst  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

# Single-day values currently printed in the paper.
PUBLISHED_GAP = {"miami": 8.37, "austin": 2.11, "phoenix": 1.35, "denver": 0.28}
PUBLISHED_NDVI = {"denver": 0.1863, "phoenix": 0.0492,
                  "austin": 0.0228, "miami": 0.0030}


def ndvi_window(date_iso: str, half: int = 8) -> str:
    d = date.fromisoformat(date_iso)
    return f"{(d - timedelta(days=half)).isoformat()}/{(d + timedelta(days=half)).isoformat()}"


def main() -> None:
    rows = []
    for name, dates in POOLS.items():
        city = CITIES[name]
        print(f"\n=== {name} ===")
        boundary = fetch_boundary(city, cache_path=RAW / f"{name}_boundary.geojson")
        grid = build_grid(boundary, resolution_m=500.0)
        landuse = build_worldcover_landuse(boundary, grid, RAW,
                                           tile=city.worldcover_tile)
        mask = landuse["city_mask"].values.astype(bool)
        imp = landuse["impervious_frac"].values
        veg = landuse["vegetation_frac"].values
        wat = landuse["water_mask"].values
        n_city = int(mask.sum())

        gaps, three_ch, four_ch, used = [], [], [], []
        for d in dates:
            try:
                lst = fetch_aqua_lst(d, grid, RAW / "modis")
            except Exception:
                continue
            if float(np.isfinite(lst).sum()) / max(n_city, 1) < MIN_COVERAGE:
                continue
            res = ceiling_for_landuse(lst, landuse)

            try:
                ndvi = fetch_ndvi(grid, ndvi_window(d), RAW / "modis_ndvi")
            except Exception as exc:
                print(f"    {d}  NDVI unavailable ({type(exc).__name__})")
                gaps.append(res.veg_imp_lst_gap_c)
                used.append(d)
                continue

            v = np.isfinite(lst) & mask & np.isfinite(ndvi)
            y = lst[v].astype(float)
            X3 = np.column_stack([imp[v], veg[v], wat[v]]).astype(float)
            X4 = np.column_stack([imp[v], veg[v], wat[v], ndvi[v]]).astype(float)
            r3, r4 = _multiple_r(y, X3), _multiple_r(y, X4)

            gaps.append(res.veg_imp_lst_gap_c)
            three_ch.append(r3)
            four_ch.append(r4)
            used.append(d)
            print(f"    {d}  gap {res.veg_imp_lst_gap_c:+6.2f} C   "
                  f"3ch {r3:+.3f}   +NDVI {r4:+.3f}   gain {r4 - r3:+.4f}")

        if not gaps:
            print("    no usable days")
            continue

        g = np.array(gaps, dtype=float)
        rec = dict(city=name, n_days=len(used), dates=used,
                   gap_mean=float(g.mean()),
                   gap_sd=float(g.std(ddof=1)) if len(g) > 1 else 0.0,
                   gap_min=float(g.min()), gap_max=float(g.max()),
                   published_gap=PUBLISHED_GAP.get(name))
        if three_ch:
            a, b = np.array(three_ch), np.array(four_ch)
            gain = b - a
            rec.update(n_ndvi_days=len(a),
                       three_ch_mean=float(a.mean()),
                       with_ndvi_mean=float(b.mean()),
                       gain_mean=float(gain.mean()),
                       gain_sd=float(gain.std(ddof=1)) if len(gain) > 1 else 0.0,
                       gain_min=float(gain.min()), gain_max=float(gain.max()),
                       published_gain=PUBLISHED_NDVI.get(name))
        rows.append(rec)

    # ---------------- gap summary ----------------
    print()
    print("=" * 88)
    print("BUILT-VEGETATED GAP: published single day vs multi-day mean")
    print("=" * 88)
    print(f"{'city':<9} {'n':>3} {'published':>10} {'multi-day':>11} {'sd':>7} "
          f"{'range':>18} {'sign flip':>10}")
    print("-" * 88)
    for r in rows:
        flip = ("YES" if r["published_gap"] is not None
                and np.sign(r["published_gap"]) != np.sign(r["gap_mean"]) else "")
        print(f"{r['city']:<9} {r['n_days']:>3} {r['published_gap']:>+10.2f} "
              f"{r['gap_mean']:>+11.2f} {r['gap_sd']:>7.2f} "
              f"[{r['gap_min']:+6.2f},{r['gap_max']:+6.2f}] {flip:>10}")

    # ---------------- NDVI summary ----------------
    print()
    print("=" * 88)
    print("NDVI PRE-SCREEN: published single day vs multi-day mean")
    print("=" * 88)
    print(f"{'city':<9} {'n':>3} {'3ch':>8} {'+NDVI':>8} {'gain':>9} {'sd':>7} "
          f"{'range':>18} {'published':>10}")
    print("-" * 88)
    for r in rows:
        if "gain_mean" not in r:
            continue
        print(f"{r['city']:<9} {r['n_ndvi_days']:>3} {r['three_ch_mean']:>+8.3f} "
              f"{r['with_ndvi_mean']:>+8.3f} {r['gain_mean']:>+9.4f} "
              f"{r['gain_sd']:>7.4f} [{r['gain_min']:+7.4f},{r['gain_max']:+7.4f}] "
              f"{r['published_gain']:>+10.4f}")

    print()
    print("verdict stability (is the gain distinguishable from zero?):")
    for r in rows:
        if "gain_mean" not in r:
            continue
        lo = r["gain_mean"] - 2 * r["gain_sd"]
        hi = r["gain_mean"] + 2 * r["gain_sd"]
        verdict = ("worth integrating" if lo > 0.05 else
                   "positive but modest" if lo > 0 else "not distinguishable from 0")
        print(f"  {r['city']:<9} gain {r['gain_mean']:+.4f} "
              f"+-2sd [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    (OUT / "gap_ndvi_multiday.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote outputs/gap_ndvi_multiday.json")


if __name__ == "__main__":
    main()
