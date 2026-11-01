"""Is the receptive-field gain real, or is it support matching?

MYD11A1 LST is natively ~1 km. We warp it to a 500 m grid with
nearest-neighbour resampling, so each thermal pixel is duplicated across
several cells -- measured at 7.55x for Austin, leaving 380 distinct LST
values among 2,868 cells. Land cover, aggregated from 10 m WorldCover,
genuinely varies at 500 m.

That asymmetry gives the receptive-field result a competing explanation.
The predictors sit at a finer support than the target, and smoothing land
cover at sigma = 1-2 cells (0.5-1 km) is precisely what is needed to bring
the two supports into alignment. Under that reading the "context gain" is
an ordinary geostatistical effect, not evidence that models reading across
space exceed per-cell ceilings.

The two explanations make the same prediction at 500 m, so 500 m cannot
separate them. Coarsening can. Block-aggregate both land cover and LST to
a common support at or beyond the thermal pixel, then smooth only at
scales strictly wider than that new pixel:

  * gain persists  -> spatial reach is doing the work; support matching
                      cannot explain a gain measured beyond the pixel
  * gain vanishes  -> the original finding was support matching

Aggregation is by nanmean over factor x factor blocks, so a coarse cell is
the mean of its valid fine cells. The CV block side is reduced in step, to
hold the fold geometry at a constant physical scale (~3 km) rather than a
constant cell count.
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
    cv_nonlinear_r, spatial_block_folds, spatial_context_channels,
)
from austin_twin.cities import CITIES  # noqa: E402
from austin_twin.grid import build_grid, fetch_boundary  # noqa: E402
from austin_twin.modis import fetch_aqua_lst  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
RESULT = OUT / "support_matching_test.json"

# factor -> (cell size m, CV block side in coarse cells, smoothing sigmas)
# sigmas are chosen so the smallest is strictly wider than the coarse cell,
# which is what makes a surviving gain attributable to reach rather than support.
LEVELS = {
    1: (500,  6, (1.0, 2.0, 4.0)),   # as published
    2: (1000, 3, (1.0, 2.0, 4.0)),   # at the MODIS native pixel
    3: (1500, 2, (1.0, 2.0, 3.0)),   # beyond it
}


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Mean over f x f blocks, ignoring NaN. Trims any partial edge block."""
    if f == 1:
        return a
    ny, nx = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    b = a[:ny, :nx].reshape(ny // f, f, nx // f, f)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3))


def duplication_factor(lst: np.ndarray, valid: np.ndarray) -> float:
    v = lst[valid]
    if v.size == 0:
        return float("nan")
    return float(v.size / len(np.unique(np.round(v, 4))))


def main() -> None:
    focus = json.loads((OUT / "focus_city_ceilings.json").read_text())
    out = []

    for rec in focus:
        name = rec["city"]
        city = CITIES[name]
        boundary = fetch_boundary(city, cache_path=RAW / f"{name}_boundary.geojson")
        grid = build_grid(boundary, resolution_m=500.0)
        landuse = build_worldcover_landuse(boundary, grid, RAW,
                                           tile=city.worldcover_tile)
        mask = landuse["city_mask"].values.astype(bool)
        chans0 = [landuse[v].values.astype(float) for v in
                  ("impervious_frac", "vegetation_frac", "water_mask")]

        print(f"\n{name}")
        print(f"  {'cell':>6}{'dup':>7}{'n':>7}{'linear':>9}"
              f"{'perCell':>9}{'context':>9}{'gain':>8}")

        city_rows = {}
        for f, (metres, blk, sigmas) in LEVELS.items():
            per_day = {}
            for d in rec["dates"]:
                try:
                    lst0 = fetch_aqua_lst(d, grid, RAW / "modis")
                except Exception:
                    continue
                keep0 = mask & np.isfinite(lst0)
                if int(keep0.sum()) < 200:
                    continue

                # aggregate LST and land cover onto the same coarse support
                lst = block_mean(np.where(keep0, lst0, np.nan), f)
                frac = block_mean(keep0.astype(float), f)
                chans = [block_mean(np.where(keep0, c, np.nan), f) for c in chans0]
                # a coarse cell counts only if most of it was valid
                valid = np.isfinite(lst) & (frac > 0.5)
                for c in chans:
                    valid &= np.isfinite(c)
                if int(valid.sum()) < 200:
                    continue

                y = lst[valid]
                folds = spatial_block_folds(valid, block_side=blk)
                X1 = np.column_stack([c[valid] for c in chans])
                ctx = spatial_context_channels(chans, valid, sigmas)
                Xc = np.column_stack([c[valid] for c in ctx])

                per_day[d] = dict(
                    linear=_multiple_r(y, X1),
                    per_cell=cv_nonlinear_r(y, X1, folds)[0],
                    context=cv_nonlinear_r(y, Xc, folds)[0],
                    n=int(y.size),
                    dup=duplication_factor(lst0, keep0),
                )
            if not per_day:
                print(f"  {metres:>6}   too few cells after aggregation")
                continue

            def mean(k):
                v = [r[k] for r in per_day.values() if np.isfinite(r[k])]
                return float(np.mean(v)) if v else float("nan")

            row = {k: mean(k) for k in ("linear", "per_cell", "context", "n", "dup")}
            row["gain"] = row["context"] - row["per_cell"]
            row.update(cell_m=metres, block_side=blk, sigmas=list(sigmas),
                       n_days=len(per_day))
            city_rows[metres] = row
            print(f"  {metres:>6}{row['dup']:>7.2f}{row['n']:>7.0f}"
                  f"{row['linear']:>+9.3f}{row['per_cell']:>+9.3f}"
                  f"{row['context']:>+9.3f}{row['gain']:>+8.3f}")

        out.append(dict(city=name, levels=city_rows))

    RESULT.write_text(json.dumps(out, indent=1))

    print("\n" + "=" * 66)
    print("CONTEXT GAIN BY SUPPORT")
    print(f"{'city':<10}{'500 m':>10}{'1000 m':>10}{'1500 m':>10}")
    print("-" * 66)
    for o in out:
        g = {m: r["gain"] for m, r in o["levels"].items()}
        cells = [f"{g[m]:+.3f}" if m in g else "  --  " for m in (500, 1000, 1500)]
        print(f"{o['city']:<10}" + "".join(f"{c:>10}" for c in cells))
    print()
    print("Gain surviving at 1000-1500 m, where the smoothing scales exceed")
    print("the thermal pixel, cannot be explained by support matching.")
    print(f"\nwrote {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
