"""Does T_i-dependent VPD recover the Penman-Monteith regression?

paper_v2 Table V records canonical PM collapsing held-out r from +0.758 to
+0.416, and attributes it to two mechanisms: LAI saturation and VPD
decoupling. Its own numbers say the split is not even -- linear LAI recovers
only ~0.08 of the 0.34 -- so VPD decoupling should dominate. The Discussion
proposes big-leaf bulk PM with T_i-dependent VPD as the fix.

This measures it. Each variant is *recalibrated* (Stage 2, D pinned, ARCO
forcing), because comparing ET formulations at coefficients fitted for a
different formulation would confound the comparison.

Grid:
  vpd_mode  air      canonical FAO-56, spatially uniform VPD (baseline)
            surface  e_sat(T_i) in the bounded PM form
            bulk     true big-leaf, unbounded (capped at 1.5 x Rn)
  lai_k     3.0      canonical, saturates by f_veg ~ 0.5
            1.0      softened
            linear   LAI proportional to f_veg (k -> 0 limit)

Linear ET is included as the reference point.
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

TRAIN_DATES = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST_DATES = ["2024-08-21", "2024-08-22"]

# paper_v2 Table V, for reference.
PUBLISHED = {"linear ET + ERA5 (ship)": 0.758,
             "PM canonical (K=3)": 0.416,
             "PM linear LAI": 0.493}

_LIN = dict(pm_lai_mode="linear")

VARIANTS = [
    ("linear ET (reference)", dict(use_pm_et=False)),
    ("PM air VPD, k=3 (canonical)", dict(use_pm_et=True, pm_vpd_mode="air", pm_lai_k=3.0)),
    ("PM air VPD, k=1", dict(use_pm_et=True, pm_vpd_mode="air", pm_lai_k=1.0)),
    ("PM air VPD, linear LAI", dict(use_pm_et=True, pm_vpd_mode="air", **_LIN)),
    ("PM surface VPD, k=3", dict(use_pm_et=True, pm_vpd_mode="surface", pm_lai_k=3.0)),
    ("PM surface VPD, k=1", dict(use_pm_et=True, pm_vpd_mode="surface", pm_lai_k=1.0)),
    ("PM surface VPD, linear LAI", dict(use_pm_et=True, pm_vpd_mode="surface", **_LIN)),
    ("big-leaf bulk, k=3", dict(use_pm_et=True, pm_vpd_mode="bulk", pm_lai_k=3.0,
                                pm_max_evaporative_fraction=1.5)),
    ("big-leaf bulk, linear LAI", dict(use_pm_et=True, pm_vpd_mode="bulk",
                                       pm_max_evaporative_fraction=1.5, **_LIN)),
]


def main() -> None:
    print("[setup] grid, land cover, MODIS, ARCO forcing...")
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=AUSTIN.worldcover_tile)
    modis_train = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TRAIN_DATES}
    modis_test = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TEST_DATES}
    forcings = {d: forcing_for_modis_date(AUSTIN, d, cache_dir=RAW / "arco_era5")
                for d in TRAIN_DATES + TEST_DATES}

    ship = load_calibrated_config(OUT / "calibrated_config_arco.json",
                                  base_config=SimConfig(duration_hours=48.0))
    D_pinned = float(ship.diffusion_m2_s)
    print(f"[setup] D pinned at {D_pinned:.3f}; each variant re-fits "
          f"3 absorptions + k_ET\n")

    print(f"{'variant':<30} {'test r':>9} {'test RMSE':>10} "
          f"{'vs linear':>10} {'k_ET':>8}")
    print("-" * 72)

    rows = []
    ref_r = None
    for label, overrides in VARIANTS:
        cfg = replace(ship, **overrides)
        try:
            cal = calibrate(
                landuse, modis_train, modis_test,
                base_config=cfg,
                fixed_coefs={"diffusion_m2_s": D_pinned},
                forcings=forcings,
                popsize=8, maxiter=12, seed=0, verbose=False,
            )
        except Exception as exc:
            print(f"{label:<30} FAILED {type(exc).__name__}: {str(exc)[:40]}")
            continue

        test_r = float(np.mean([m.pearson_r for m in cal.test_metrics_after]))
        test_rmse = float(np.mean([m.rmse for m in cal.test_metrics_after]))
        if ref_r is None:
            ref_r = test_r
        print(f"{label:<30} {test_r:>+9.4f} {test_rmse:>10.4f} "
              f"{test_r - ref_r:>+10.4f} {cal.optimal_coefs['et_coeff']:>8.2f}")
        rows.append(dict(variant=label, test_r=test_r, test_rmse=test_rmse,
                         delta_vs_linear=test_r - ref_r,
                         coefs={k: float(v) for k, v in cal.optimal_coefs.items()},
                         **overrides))

    print()
    print("=" * 72)
    print("MECHANISM DECOMPOSITION")
    print("=" * 72)
    by = {r["variant"]: r["test_r"] for r in rows}
    lin = by.get("linear ET (reference)")
    canon = by.get("PM air VPD, k=3 (canonical)")
    if lin is not None and canon is not None:
        total = canon - lin
        print(f"  total PM regression            : {total:+.4f}")
        lai_only = by.get("PM air VPD, linear LAI")
        if lai_only is not None:
            frac = (lai_only - canon) / abs(total) if total else float("nan")
            print(f"  recovered by LAI fix alone     : "
                  f"{lai_only - canon:+.4f}  ({frac:.0%} of the regression)")
        vpd_only = by.get("PM surface VPD, k=3")
        if vpd_only is not None:
            frac = (vpd_only - canon) / abs(total) if total else float("nan")
            print(f"  recovered by VPD fix alone     : "
                  f"{vpd_only - canon:+.4f}  ({frac:.0%} of the regression)")
        both = by.get("PM surface VPD, linear LAI")
        if both is not None:
            frac = (both - canon) / abs(total) if total else float("nan")
            print(f"  recovered by both fixes        : "
                  f"{both - canon:+.4f}  ({frac:.0%} of the regression)")
            print(f"  best PM vs linear ET           : {both - lin:+.4f}")

    print()
    print("  paper_v2 Table V for reference:")
    for k, v in PUBLISHED.items():
        print(f"    {k:<30} {v:+.3f}")

    (OUT / "pm_variants.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote outputs/pm_variants.json")


if __name__ == "__main__":
    main()
