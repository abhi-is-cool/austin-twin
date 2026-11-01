"""Re-run the ship-config block bootstrap under ARCO-ERA5 forcing.

paper_v2 reports ARCO-recalibrated coefficients in Section II-C but keeps the
CDS-derived score and CI in Table II. `bootstrap_uncertainty.py` cannot be
re-run as-is: it reads CDS NetCDF files from `data/raw/era5/` that no longer
exist and require credentials to obtain.

This reuses that script's estimator verbatim -- same 6x6 block tiles, same
B = 2000, same per-day resampling and percentile CI -- and only swaps the
forcing source, so the resulting interval is directly comparable to the
published one.

Both configurations are evaluated so the comparison is like-for-like:
  * ship-CDS   : published CDS coefficients, ARCO forcing
  * ship-ARCO  : ARCO-recalibrated coefficients, ARCO forcing
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_uncertainty import (  # noqa: E402  (path set above)
    BLOCK_SIDE, N_BOOT, SEED,
    _peak_frame, _valid_cells_2d, bootstrap_r_rmse,
)

from austin_twin.arco_era5 import forcing_for_modis_date  # noqa: E402
from austin_twin.calibration import load_calibrated_config  # noqa: E402
from austin_twin.cities import AUSTIN  # noqa: E402
from austin_twin.grid import build_grid, fetch_boundary  # noqa: E402
from austin_twin.modis import fetch_aqua_lst  # noqa: E402
from austin_twin.simulator import SimConfig, run  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
TEST_DATES = ["2024-08-21", "2024-08-22"]

# Table II ship row as printed in paper_v2.
PUBLISHED_R = (0.758, 0.722, 0.788)
PUBLISHED_RMSE = (1.42, 1.36, 1.48)


def days_data_for(cfg, landuse, grid, forcings):
    out = []
    city_mask = landuse["city_mask"].values.astype(bool)
    for d in TEST_DATES:
        sim = _peak_frame(run(landuse, cfg, forcing=forcings[d]))
        obs = fetch_aqua_lst(d, grid, RAW / "modis")
        out.append((sim, obs, _valid_cells_2d(sim, obs, city_mask)))
    return out


def main() -> None:
    print("[setup] grid, land cover, MODIS, ARCO forcing...")
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=AUSTIN.worldcover_tile)
    forcings = {d: forcing_for_modis_date(AUSTIN, d, cache_dir=RAW / "arco_era5")
                for d in TEST_DATES}

    base = SimConfig(duration_hours=48.0)
    configs = {
        "ship-CDS coefs (ARCO forcing)":
            replace(load_calibrated_config(OUT / "calibrated_config_era5.json",
                                           base_config=base), use_pm_et=False),
        "ship-ARCO coefs (ARCO forcing)":
            replace(load_calibrated_config(OUT / "calibrated_config_arco.json",
                                           base_config=base), use_pm_et=False),
    }

    print(f"[bootstrap] B={N_BOOT}, block side={BLOCK_SIDE} cells, seed={SEED}\n")
    results = {}
    for label, cfg in configs.items():
        dd = days_data_for(cfg, landuse, grid, forcings)
        res = bootstrap_r_rmse(dd, B=N_BOOT, L=BLOCK_SIDE, seed=SEED)
        r_hat, (r_lo, r_hi) = res["r_hat"], res["r_ci"]
        rmse_hat, (rmse_lo, rmse_hi) = res["rmse_hat"], res["rmse_ci"]
        results[label] = dict(r=r_hat, r_ci=[r_lo, r_hi],
                              rmse=rmse_hat, rmse_ci=[rmse_lo, rmse_hi])
        print(f"  {label}")
        print(f"    r    = {r_hat:+.4f}  [{r_lo:+.4f}, {r_hi:+.4f}]")
        print(f"    RMSE = {rmse_hat:.4f}  [{rmse_lo:.4f}, {rmse_hi:.4f}]")

    print()
    print("=" * 74)
    print("vs paper_v2 Table II ship row")
    print("=" * 74)
    print(f"  published : r = {PUBLISHED_R[0]:+.3f} "
          f"[{PUBLISHED_R[1]:+.3f}, {PUBLISHED_R[2]:+.3f}]   "
          f"RMSE = {PUBLISHED_RMSE[0]:.2f} "
          f"[{PUBLISHED_RMSE[1]:.2f}, {PUBLISHED_RMSE[2]:.2f}]")
    arco = results["ship-ARCO coefs (ARCO forcing)"]
    print(f"  ARCO      : r = {arco['r']:+.3f} "
          f"[{arco['r_ci'][0]:+.3f}, {arco['r_ci'][1]:+.3f}]   "
          f"RMSE = {arco['rmse']:.2f} "
          f"[{arco['rmse_ci'][0]:.2f}, {arco['rmse_ci'][1]:.2f}]")
    print(f"  delta r point estimate: {arco['r'] - PUBLISHED_R[0]:+.4f}")

    (OUT / "bootstrap_arco.json").write_text(json.dumps({
        "B": N_BOOT, "block_side": BLOCK_SIDE, "seed": SEED,
        "test_dates": TEST_DATES,
        "published_table_ii": {"r": PUBLISHED_R, "rmse": PUBLISHED_RMSE},
        "results": results,
    }, indent=2))
    print(f"\nwrote outputs/bootstrap_arco.json")


if __name__ == "__main__":
    main()
