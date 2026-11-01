"""Is the "ceiling" actually a ceiling?

paper_v2 defines it as Pearson r between MODIS LST and *impervious fraction*
and calls it "the maximum correlation any land-cover-only model can achieve".
But the simulator consumes three land-cover channels (impervious, vegetation,
water), and the multiple correlation of a three-predictor fit is >= the
univariate correlation of any one of them by construction. So a land-cover
model can exceed this "ceiling" without anything being wrong.

That matters because the paper reports exactly that happening -- Miami
Scheme A reaches +0.832 against a ceiling of +0.827 -- and explains it as
bootstrap sampling variability. If the cause is instead definitional, the
explanation is wrong and the abstract's claim is too strong.

This computes both quantities on cached data for all four cities.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from austin_twin.cities import CITIES
from austin_twin.grid import build_grid, fetch_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

DATES = {
    "austin": "2024-08-22",
    "phoenix": "2024-07-22",
    "denver": "2024-07-25",
    "miami": "2024-04-16",
}
# Best model score per city from paper_v2 Table IV (Austin = ship reference).
BEST_MODEL = {
    "austin": 0.758, "phoenix": 0.083, "denver": 0.004, "miami": 0.832,
}


def multiple_r(y: np.ndarray, X: np.ndarray) -> float:
    """Multiple correlation: r between y and its least-squares projection."""
    A = np.column_stack([X, np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    if np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(pred, y)[0, 1])


def main() -> None:
    rows = []
    for name, date_iso in DATES.items():
        city = CITIES[name]
        boundary = fetch_boundary(city, cache_path=RAW / f"{name}_boundary.geojson")
        grid = build_grid(boundary, resolution_m=500.0)
        landuse = build_worldcover_landuse(boundary, grid, RAW,
                                           tile=city.worldcover_tile)
        lst = fetch_aqua_lst(date_iso, grid, RAW / "modis")

        mask = landuse["city_mask"].values.astype(bool)
        valid = np.isfinite(lst) & mask
        y = lst[valid].astype(float)
        imp = landuse["impervious_frac"].values[valid].astype(float)
        veg = landuse["vegetation_frac"].values[valid].astype(float)
        wat = landuse["water_mask"].values[valid].astype(float)

        uni = float(np.corrcoef(y, imp)[0, 1])
        multi = multiple_r(y, np.column_stack([imp, veg, wat]))
        best = BEST_MODEL[name]

        rows.append(dict(city=name, date=date_iso, univariate_ceiling=uni,
                         multivariate_ceiling=multi, best_model_r=best,
                         model_over_uni=best / uni if uni else float("nan"),
                         model_over_multi=best / multi if multi else float("nan")))

    print(f"{'city':<9} {'uni r(LST,imp)':>15} {'multi R (3ch)':>14} "
          f"{'best model':>11} {'model/uni':>10} {'model/multi':>12}")
    print("-" * 78)
    for r in rows:
        print(f"{r['city']:<9} {r['univariate_ceiling']:>15.4f} "
              f"{r['multivariate_ceiling']:>14.4f} {r['best_model_r']:>11.3f} "
              f"{r['model_over_uni']:>10.2f}x {r['model_over_multi']:>11.2f}x")

    print()
    over_uni = [r for r in rows if r["best_model_r"] > r["univariate_ceiling"]]
    over_multi = [r for r in rows if r["best_model_r"] > r["multivariate_ceiling"]]
    print(f"cities where model exceeds the univariate 'ceiling'   : "
          f"{len(over_uni)} of {len(rows)}  "
          f"({', '.join(r['city'] for r in over_uni) or 'none'})")
    print(f"cities where model exceeds the multivariate ceiling   : "
          f"{len(over_multi)} of {len(rows)}  "
          f"({', '.join(r['city'] for r in over_multi) or 'none'})")

    (OUT / "ceiling_definition_diag.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote outputs/ceiling_definition_diag.json")


if __name__ == "__main__":
    main()
