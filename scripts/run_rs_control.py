"""Does high r_s actually mean "r_s is mis-parameterised", or just "less ET"?

The sweep found test r rising monotonically with r_s_leaf all the way to the
edge of the grid (800 s/m, 8x the FAO-56 reference-grass value), recovering
61% of the PM regression. That has two very different readings:

  (a) the stomatal resistance is genuinely mis-parameterised for a
      water-limited summer city, and a physically higher value fixes it; or
  (b) r_s -> infinity drives Q_LE -> 0, so the "improvement" is just the
      model switching evapotranspiration off, and PM-ET is net harmful here.

These are distinguishable. If (b), then pushing r_s far beyond any physical
value keeps improving the score and converges on a no-ET control. If (a),
the curve turns over at a finite, defensible value.

Runs the extended grid plus two controls: PM with r_s effectively infinite,
and the linear model with k_ET pinned to zero (no ET at all).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from austin_twin.arco_era5 import forcing_for_modis_date
from austin_twin.calibration import calibrate, load_calibrated_config
from austin_twin.cities import AUSTIN
from austin_twin.grid import build_grid, fetch_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
TRAIN = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST = ["2024-08-21", "2024-08-22"]


def main() -> None:
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW, tile=AUSTIN.worldcover_tile)
    tr = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TRAIN}
    te = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TEST}
    fo = {d: forcing_for_modis_date(AUSTIN, d, cache_dir=RAW / "arco_era5")
          for d in TRAIN + TEST}
    ship = load_calibrated_config(OUT / "calibrated_config_arco.json",
                                  base_config=SimConfig(duration_hours=48.0))
    D = float(ship.diffusion_m2_s)

    def fit(cfg):
        cal = calibrate(landuse, tr, te, base_config=cfg,
                        fixed_coefs={"diffusion_m2_s": D,
                                     "et_coeff": float(cfg.et_coeff)},
                        forcings=fo, popsize=8, maxiter=12, seed=0, verbose=False)
        return (float(np.mean([m.pearson_r for m in cal.test_metrics_after])),
                float(np.mean([m.rmse for m in cal.test_metrics_after])))

    rows = []
    print(f"{'config':<34} {'test r':>9} {'RMSE':>8}")
    print("-" * 54)

    for rs in (100.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 1e6):
        cfg = replace(ship, use_pm_et=True, pm_vpd_mode="air", pm_lai_k=3.0,
                      pm_r_s_leaf=float(rs))
        r, rm = fit(cfg)
        tag = "  (FAO-56 default)" if rs == 100.0 else ""
        print(f"{'PM, r_s_leaf = ' + format(rs, '.0f'):<34} {r:>+9.4f} {rm:>8.4f}{tag}")
        rows.append(dict(config=f"pm_rs_{rs:.0f}", r_s_leaf=rs, test_r=r, test_rmse=rm))

    # Control: linear form with ET switched off entirely.
    cfg0 = replace(ship, use_pm_et=False, et_coeff=0.0)
    r0, rm0 = fit(cfg0)
    print(f"{'CONTROL: no ET at all (k_ET = 0)':<34} {r0:>+9.4f} {rm0:>8.4f}")
    rows.append(dict(config="no_et_control", test_r=r0, test_rmse=rm0))

    # Reference points.
    lin, _ = fit(replace(ship, use_pm_et=False))
    print(f"{'reference: linear ET, k_ET fitted':<34} {lin:>+9.4f}")
    rows.append(dict(config="linear_et_reference", test_r=lin))

    print()
    print("=" * 62)
    best = max((x for x in rows if "r_s_leaf" in x), key=lambda x: x["test_r"])
    turns_over = best["r_s_leaf"] < 1e6
    print(f"  best r_s in grid: {best['r_s_leaf']:.0f} s/m -> r = {best['test_r']:+.4f}")
    print(f"  curve turns over below the ceiling: {'YES' if turns_over else 'NO'}")
    print(f"  no-ET control: r = {r0:+.4f}")
    gap = best["test_r"] - r0
    print(f"  best PM minus no-ET control: {gap:+.4f}")
    # The earlier version of this branch tested abs(gap) and so reported
    # "PM beats the no-ET control" for a gap of -0.073, i.e. when PM was
    # comfortably WORSE. Sign first, magnitude second.
    if gap > 0.02:
        print("  -> PM at its best r_s beats the no-ET control, so r_s is doing")
        print("     real work beyond suppressing evapotranspiration.")
    elif abs(gap) <= 0.02:
        print("  -> PM at its best r_s is indistinguishable from switching ET off.")
    else:
        print(f"  -> PM at its best r_s is still {abs(gap):.3f} BELOW simply")
        print("     switching ET off. Tuning r_s narrows the gap but never")
        print("     closes it: the formulation is a net loss at every r_s.")

    # The sweep moves only the vegetation stomatal path. r_s_water (1 s/m)
    # and r_s_bare (2000 s/m) are untouched, so r_s_leaf -> infinity does NOT
    # converge on the no-ET control -- water and bare cells keep evaporating.
    # That is why the curve can turn over rather than rising to the control.
    print("     (the sweep moves the vegetation path only; water and bare")
    print("      conductances are fixed, so r_s_leaf -> inf is not 'no ET')")

    (OUT / "rs_control.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote outputs/rs_control.json")


if __name__ == "__main__":
    main()
