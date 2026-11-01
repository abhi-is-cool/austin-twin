"""Re-run the two-stage ship calibration on zero-authentication forcing.

`run_era5_calibration.py` produced the published ship configuration using
ERA5 fetched from the Copernicus CDS, which needs a registered account. This
is the same procedure driven by ARCO-ERA5 instead, so the shipped
coefficients are themselves derived from anonymously-retrievable data and
the pipeline is credential-free end to end.

`validate_arco_era5.py` already established that the two sources agree when
the coefficients are held fixed (test mean r +0.7579 vs +0.7557). This step
closes the remaining gap: that the *fit* does not depend on CDS either.

Procedure is deliberately identical to the published one:
  Stage 1 -- anchor diffusion to a 200 m literature half-cooling distance,
             using synthetic forcing (weather-independent, so unchanged).
  Stage 2 -- differential evolution over 4 free coefficients (3 absorptions
             + et_coeff) with D pinned, scoring each MODIS date against a
             simulation driven by that date's own forcing.

Same DE settings (popsize=8, maxiter=12, seed=0) and the same train/test
split, so any difference in the result is attributable to the forcing
source rather than to the search.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from austin_twin.arco_era5 import forcing_for_modis_date
from austin_twin.calibration import (
    calibrate, format_metrics_table, load_calibrated_config,
)
from austin_twin.cities import AUSTIN
from austin_twin.cooling_response import (
    find_D_for_target_half_cooling, run_full_analysis,
)
from austin_twin.grid import build_grid, fetch_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

CAL_TWO = OUT / "calibrated_config_two_stage.json"   # synthetic-forcing base
CAL_CDS = OUT / "calibrated_config_era5.json"        # published ship config
CAL_ARCO = OUT / "calibrated_config_arco.json"

TRAIN_DATES = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST_DATES = ["2024-08-21", "2024-08-22"]
TARGET_HALF_COOLING_M = 200.0
D_SEARCH_RANGE = (1.0, 50.0)
D_SEARCH_N = 12


def main() -> None:
    published = json.loads(CAL_CDS.read_text())

    print("[1/6] Austin grid + WorldCover...")
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=AUSTIN.worldcover_tile)

    print("[2/6] base config (two-stage, synthetic-forcing calibration)...")
    base = SimConfig(duration_hours=48.0)
    legacy_cfg = replace(load_calibrated_config(CAL_TWO, base_config=base),
                         use_pm_et=False)

    print(f"[3/6] STAGE 1: sweep D for half-cooling = {TARGET_HALF_COOLING_M:.0f} m")
    D_target, sweep = find_D_for_target_half_cooling(
        landuse, legacy_cfg,
        target_half_cooling_m=TARGET_HALF_COOLING_M,
        D_range=D_SEARCH_RANGE, n=D_SEARCH_N,
    )
    print(f"      -> D = {D_target:.3f} m2/s "
          f"(published CDS run selected {published['diffusion_m2_s']:.3f})")

    print("[4/6] MODIS days + ARCO-ERA5 forcings...")
    modis_train = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TRAIN_DATES}
    modis_test = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TEST_DATES}
    cache = RAW / "arco_era5"
    forcings = {d: forcing_for_modis_date(AUSTIN, d, cache_dir=cache)
                for d in TRAIN_DATES + TEST_DATES}

    print(f"[5/6] STAGE 2: DE on 4 coefs, D={D_target:.3f} pinned...")
    cal = calibrate(
        landuse, modis_train, modis_test,
        base_config=legacy_cfg,
        fixed_coefs={"diffusion_m2_s": D_target},
        forcings=forcings,
        popsize=8, maxiter=12, seed=0, verbose=True,
    )

    print()
    print(format_metrics_table(cal.train_metrics_before,
                               cal.train_metrics_after, "train"))
    print()
    print(format_metrics_table(cal.test_metrics_before,
                               cal.test_metrics_after, "test"))

    test_r = float(np.mean([m.pearson_r for m in cal.test_metrics_after]))
    test_rmse = float(np.mean([m.rmse for m in cal.test_metrics_after]))
    train_r = float(np.mean([m.pearson_r for m in cal.train_metrics_after]))

    print()
    print("=" * 78)
    print("ARCO-calibrated ship config vs published CDS ship config")
    print("=" * 78)
    print(f"{'coefficient':<26} {'ARCO':>12} {'CDS':>12} {'delta':>12}")
    print("-" * 78)
    lit = {
        "absorption_impervious": "0.80-0.95",
        "absorption_vegetation": "0.50-0.65",
        "absorption_water": "0.06-0.10",
        "et_coeff": "-",
        "diffusion_m2_s": "-",
    }
    for name in ("absorption_impervious", "absorption_vegetation",
                 "absorption_water", "et_coeff", "diffusion_m2_s"):
        a, c = cal.optimal_coefs[name], published[name]
        print(f"{name:<26} {a:>12.6f} {c:>12.6f} {a - c:>+12.6f}"
              f"   lit {lit[name]}")

    print()
    print(f"{'metric':<26} {'ARCO':>12} {'CDS':>12} {'delta':>12}")
    print("-" * 78)
    for label, a, c in (
        ("train mean r", train_r, published["train_r_mean_after"]),
        ("test mean r", test_r, published["test_r_mean_after"]),
        ("test mean RMSE", test_rmse, published["test_rmse_mean_after"]),
    ):
        print(f"{label:<26} {a:>12.4f} {c:>12.4f} {a - c:>+12.4f}")

    print()
    print("  per-day held-out r:")
    for m in cal.test_metrics_after:
        c = published["test_per_day_r_after"][m.date]
        print(f"    {m.date}  ARCO {m.pearson_r:+.4f}   CDS {c:+.4f}   "
              f"delta {m.pearson_r - c:+.4f}")

    print()
    print("[6/6] cooling response under the ARCO-calibrated config...")
    report = run_full_analysis(landuse, cal.optimal_config)
    print(f"      citywide slope per +10% canopy : "
          f"{report.response_curve.citywide_slope_per_10pct:+.3f} C")
    print(f"      local slope per +10% canopy    : "
          f"{report.response_curve.planted_slope_per_10pct:+.3f} C")
    print(f"      half-cooling distance          : "
          f"{report.distance_decay.half_cooling_m:.0f} m "
          f"(lambda = {report.distance_decay.decay_length_m:.0f} m)")

    CAL_ARCO.write_text(json.dumps({
        **cal.optimal_coefs,
        "use_pm_et": False,
        "forcing_source": "arco_era5 (gs://gcp-public-data-arco-era5, anonymous)",
        "train_dates": TRAIN_DATES,
        "test_dates": TEST_DATES,
        "diffusion_target_half_cooling_m": TARGET_HALF_COOLING_M,
        "train_r_mean_after": train_r,
        "test_r_mean_after": test_r,
        "test_rmse_mean_after": test_rmse,
        "test_per_day_r_after": {m.date: float(m.pearson_r)
                                 for m in cal.test_metrics_after},
        "test_per_day_rmse_after": {m.date: float(m.rmse)
                                    for m in cal.test_metrics_after},
        "cooling_response": {
            "citywide_slope_per_10pct":
                float(report.response_curve.citywide_slope_per_10pct),
            "planted_slope_per_10pct":
                float(report.response_curve.planted_slope_per_10pct),
            "half_cooling_m": float(report.distance_decay.half_cooling_m),
            "decay_length_m": float(report.distance_decay.decay_length_m),
        },
        "comparison_vs_published_cds": {
            "test_r_mean_cds": published["test_r_mean_after"],
            "delta_test_r_mean": test_r - published["test_r_mean_after"],
        },
        "stage1_sweep": sweep,
        "history": cal.history,
    }, indent=2))
    print(f"\ndone. wrote {CAL_ARCO.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
