"""Does allowing nonlinearity change the data ceiling?

The linear ceiling is already known to be violated: the shipped Austin
configuration scores r = +0.754 on held-out days against an Austin
multivariate ceiling of +0.693 on those same days. Before mapping ceilings
across hundreds of cities it is worth knowing whether that violation is
specific to Austin's tuned physics or whether land cover generally
supports more than a linear fit extracts.

Three estimates per city-day, on identical cells:

  linear      multiple R, the published definition
  blocked CV  gradient boosting, folds are contiguous 6x6 tiles
  random CV   gradient boosting, folds assigned per cell

`random CV` is a deliberate control, not a candidate estimator. Adjacent
500 m cells are strongly autocorrelated, so random folds train on a test
cell's own neighbours and the score inflates toward 1.0. If the two CV
numbers are close, the blocking is doing nothing and the design is wrong;
if random is far higher, the leak is real and blocking is what makes the
nonlinear estimate trustworthy.

Also reports the KSG mutual-information bound, which is distribution-free
and so should sit at or above the achievable figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from austin_twin.ceiling import _multiple_r  # noqa: E402
from austin_twin.ceiling_nonlinear import (  # noqa: E402
    N_FOLDS, SEED, cv_nonlinear_r, mi_bound_r, spatial_block_folds,
)
from austin_twin.cities import CITIES  # noqa: E402
from austin_twin.grid import build_grid, fetch_boundary  # noqa: E402
from austin_twin.modis import fetch_aqua_lst  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
FOCUS = OUT / "focus_city_ceilings.json"
RESULT = OUT / "nonlinear_ceiling_pilot.json"

# Austin's shipped held-out score, for the comparison that motivated this.
AUSTIN_SHIP_R = 0.754
AUSTIN_TEST_DAYS = ["2024-08-21", "2024-08-22"]


def random_folds(n: int, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_folds, size=n)


def main() -> None:
    focus = json.loads(FOCUS.read_text())
    out = []

    for rec in focus:
        name = rec["city"]
        city = CITIES[name]
        boundary = fetch_boundary(city, cache_path=RAW / f"{name}_boundary.geojson")
        grid = build_grid(boundary, resolution_m=500.0)
        landuse = build_worldcover_landuse(boundary, grid, RAW,
                                           tile=city.worldcover_tile)
        mask = landuse["city_mask"].values.astype(bool)
        chans = [landuse[v].values.astype(float) for v in
                 ("impervious_frac", "vegetation_frac", "water_mask")]

        per_day = {}
        print(f"\n{name}  ({len(rec['dates'])} days)")
        print(f"  {'date':<12}{'linear':>9}{'blockCV':>9}{'randCV':>9}{'MI':>9}{'n':>7}")
        for d in rec["dates"]:
            try:
                lst = fetch_aqua_lst(d, grid, RAW / "modis")
            except Exception as exc:
                print(f"  {d:<12}  fetch failed: {type(exc).__name__}")
                continue
            valid = mask & np.isfinite(lst)
            if int(valid.sum()) < 200:
                continue
            y = lst[valid].astype(float)
            X = np.column_stack([c[valid] for c in chans])

            lin = _multiple_r(y, X)
            blk, _ = cv_nonlinear_r(y, X, spatial_block_folds(valid))
            rnd, _ = cv_nonlinear_r(y, X, random_folds(y.size))
            mi = mi_bound_r(y, X)
            per_day[d] = dict(linear=lin, block_cv=blk, random_cv=rnd,
                              mi=mi, n=int(y.size))
            print(f"  {d:<12}{lin:>+9.3f}{blk:>+9.3f}{rnd:>+9.3f}"
                  f"{mi:>+9.3f}{y.size:>7d}")

        if not per_day:
            continue

        def mean(k):
            v = [r[k] for r in per_day.values() if np.isfinite(r[k])]
            return float(np.mean(v)) if v else float("nan")

        agg = {k: mean(k) for k in ("linear", "block_cv", "random_cv", "mi")}
        agg.update(city=name, n_days=len(per_day),
                   published_multi=rec["multi_mean"],
                   gain=agg["block_cv"] - agg["linear"],
                   leak=agg["random_cv"] - agg["block_cv"],
                   per_day=per_day)
        out.append(agg)
        print(f"  {'MEAN':<12}{agg['linear']:>+9.3f}{agg['block_cv']:>+9.3f}"
              f"{agg['random_cv']:>+9.3f}{agg['mi']:>+9.3f}")

    RESULT.write_text(json.dumps(out, indent=1))

    print("\n" + "=" * 68)
    print(f"{'city':<10}{'linear':>9}{'blockCV':>9}{'gain':>8}{'MI':>8}{'leak':>8}")
    print("-" * 68)
    for a in out:
        print(f"{a['city']:<10}{a['linear']:>+9.3f}{a['block_cv']:>+9.3f}"
              f"{a['gain']:>+8.3f}{a['mi']:>+8.3f}{a['leak']:>+8.3f}")

    au = next((a for a in out if a["city"] == "austin"), None)
    if au:
        td = [au["per_day"][d] for d in AUSTIN_TEST_DAYS if d in au["per_day"]]
        if td:
            lin_t = float(np.mean([t["linear"] for t in td]))
            blk_t = float(np.mean([t["block_cv"] for t in td]))
            print("\nAustin, on the two held-out days Table 2 scores:")
            print(f"  shipped model          {AUSTIN_SHIP_R:+.3f}")
            print(f"  linear ceiling         {lin_t:+.3f}"
                  f"   -> model is {AUSTIN_SHIP_R / lin_t * 100:.0f}% of it")
            print(f"  nonlinear ceiling      {blk_t:+.3f}"
                  f"   -> model is {AUSTIN_SHIP_R / blk_t * 100:.0f}% of it")
            verdict = ("RESOLVED: the nonlinear ceiling bounds the model"
                       if blk_t >= AUSTIN_SHIP_R else
                       "STILL VIOLATED: even the nonlinear ceiling is exceeded")
            print(f"  {verdict}")

    print(f"\nwrote {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
