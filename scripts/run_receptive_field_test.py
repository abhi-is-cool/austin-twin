"""Is the ceiling violation a receptive-field mismatch?

The pilot left one thing unexplained. Austin's shipped model scores
r = +0.754 on its two held-out days, above both the linear ceiling
(+0.693) and the cross-validated nonlinear ceiling (+0.681) on the same
days. Allowing nonlinearity did not close the gap, so nonlinearity is not
what the model is exploiting.

Hypothesis: it is spatial reach, not functional form. The ceiling
regresses LST on each cell's own land cover. The simulator diffuses heat,
so a modelled cell depends on its neighbours' land cover as well. The
model is using information the ceiling never offered its estimator.

Test: give the ceiling estimator neighbourhood-averaged copies of the same
three channels, at scales bracketing the simulator's own diffusion length
(~590 m, a little over one cell), and recompute. If the gap closes, the
ceiling was mis-specified rather than wrong in principle, and the fix is
to match the ceiling's receptive field to the model's.

This matters well beyond Austin. Convolutional and foundation-model UHI
methods all have receptive fields much wider than one pixel, so a per-cell
ceiling systematically understates what they can achieve -- and an atlas
built on per-cell ceilings would mislabel cities as hopeless.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
RESULT = OUT / "receptive_field_test.json"

AUSTIN_SHIP_R = 0.754
SIGMAS = (1.0, 2.0, 4.0)          # cells: ~0.5, 1, 2 km on the 500 m grid


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
        chans = [landuse[v].values.astype(float) for v in
                 ("impervious_frac", "vegetation_frac", "water_mask")]

        rows = {}
        print(f"\n{name}")
        print(f"  {'date':<12}{'linear':>9}{'percell':>9}{'context':>9}{'delta':>9}")
        for d in rec["dates"]:
            try:
                lst = fetch_aqua_lst(d, grid, RAW / "modis")
            except Exception:
                continue
            valid = mask & np.isfinite(lst)
            if int(valid.sum()) < 200:
                continue
            y = lst[valid].astype(float)
            folds = spatial_block_folds(valid)

            X1 = np.column_stack([c[valid] for c in chans])
            ctx = spatial_context_channels(chans, valid, SIGMAS)
            Xc = np.column_stack([c[valid] for c in ctx])

            lin = _multiple_r(y, X1)
            per_cell, _ = cv_nonlinear_r(y, X1, folds)
            context, _ = cv_nonlinear_r(y, Xc, folds)
            rows[d] = dict(linear=lin, per_cell=per_cell, context=context,
                           n=int(y.size))
            print(f"  {d:<12}{lin:>+9.3f}{per_cell:>+9.3f}{context:>+9.3f}"
                  f"{context - per_cell:>+9.3f}")

        if not rows:
            continue

        def mean(k):
            v = [r[k] for r in rows.values() if np.isfinite(r[k])]
            return float(np.mean(v)) if v else float("nan")

        agg = dict(city=name, n_days=len(rows),
                   linear=mean("linear"), per_cell=mean("per_cell"),
                   context=mean("context"), per_day=rows)
        agg["context_gain"] = agg["context"] - agg["per_cell"]
        out.append(agg)
        print(f"  {'MEAN':<12}{agg['linear']:>+9.3f}{agg['per_cell']:>+9.3f}"
              f"{agg['context']:>+9.3f}{agg['context_gain']:>+9.3f}")

    RESULT.write_text(json.dumps(out, indent=1))

    print("\n" + "=" * 72)
    print(f"{'city':<10}{'linear':>9}{'per-cell':>10}{'context':>9}{'gain':>8}")
    print("-" * 72)
    for a in out:
        print(f"{a['city']:<10}{a['linear']:>+9.3f}{a['per_cell']:>+10.3f}"
              f"{a['context']:>+9.3f}{a['context_gain']:>+8.3f}")

    au = next((a for a in out if a["city"] == "austin"), None)
    if au:
        days = ["2024-08-21", "2024-08-22"]
        td = [au["per_day"][d] for d in days if d in au["per_day"]]
        if td:
            lin = float(np.mean([t["linear"] for t in td]))
            pc = float(np.mean([t["per_cell"] for t in td]))
            cx = float(np.mean([t["context"] for t in td]))
            print("\nAustin, on the two held-out days Table 2 scores:")
            print(f"  shipped model                 {AUSTIN_SHIP_R:+.3f}")
            print(f"  linear, per-cell ceiling      {lin:+.3f}"
                  f"   model = {AUSTIN_SHIP_R / lin * 100:3.0f}%")
            print(f"  nonlinear, per-cell ceiling   {pc:+.3f}"
                  f"   model = {AUSTIN_SHIP_R / pc * 100:3.0f}%")
            print(f"  nonlinear, with context       {cx:+.3f}"
                  f"   model = {AUSTIN_SHIP_R / cx * 100:3.0f}%")
            if cx >= AUSTIN_SHIP_R:
                print("  RESOLVED: a receptive-field-matched ceiling bounds the model.")
            else:
                print("  NOT RESOLVED: context helps but the model still clears it.")

    print(f"\nwrote {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
