"""Check that zero-auth ARCO-ERA5 forcing reproduces the CDS ship config.

Published ship configuration (outputs/calibrated_config_era5.json), which
was calibrated and scored against CDS-fetched ERA5:

    held-out test mean r = +0.7557,  mean RMSE = 1.4269 C
    per day  2024-08-21: r = +0.7408
             2024-08-22: r = +0.7706

If ARCO-ERA5 is the same data, driving the same coefficients with ARCO
forcing must land on the same scores. A mismatch means a convention
differs (accumulation window, bbox snapping, time alignment) and the
zero-auth swap is not yet safe to publish behind.

Note the ship coefficients are held fixed here — this validates the
*forcing source*, not the calibration. Re-running the DE calibration on
ARCO forcing is a separate step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from austin_twin.arco_era5 import forcing_for_modis_date
from austin_twin.calibration import evaluate, load_calibrated_config
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

PUBLISHED = json.loads((OUT / "calibrated_config_era5.json").read_text())
TARGET_TEST_R = PUBLISHED["test_r_mean_after"]
TARGET_TEST_RMSE = PUBLISHED["test_rmse_mean_after"]
TARGET_PER_DAY_R = PUBLISHED["test_per_day_r_after"]


def main() -> None:
    print("[1/4] Austin grid + WorldCover...")
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=AUSTIN.worldcover_tile)

    print("[2/4] MODIS days...")
    modis = {d: fetch_aqua_lst(d, grid, RAW / "modis")
             for d in TRAIN_DATES + TEST_DATES}

    print("[3/4] ARCO-ERA5 forcings (first call ~45 s per window, then cached)...")
    cache = RAW / "arco_era5"
    forcings = {}
    for d in TRAIN_DATES + TEST_DATES:
        f = forcing_for_modis_date(AUSTIN, d, cache_dir=cache)
        forcings[d] = f
        print(f"      {d}: T_air {f.t_air_c.min():.1f}..{f.t_air_c.max():.1f} C  "
              f"T_dew mean {f.t_dew_c.mean():.1f} C  "
              f"wind mean {f.wind_speed_m_s.mean():.1f} m/s  "
              f"solar max {f.solar_w_m2.max():.0f} W/m2")

    print("[4/4] scoring ship config under ARCO forcing...")
    cfg = load_calibrated_config(OUT / "calibrated_config_era5.json",
                                 base_config=SimConfig(duration_hours=48.0))
    print("      ship coefficients:")
    for name in ("absorption_impervious", "absorption_vegetation",
                 "absorption_water", "et_coeff", "diffusion_m2_s"):
        print(f"        {name:<24} {getattr(cfg, name):.6f}")

    train = evaluate(landuse, cfg, {d: modis[d] for d in TRAIN_DATES},
                     forcings=forcings)
    test = evaluate(landuse, cfg, {d: modis[d] for d in TEST_DATES},
                    forcings=forcings)

    print()
    print("=" * 74)
    print("ARCO-ERA5 (zero-auth) vs published CDS ERA5")
    print("=" * 74)
    for m in train:
        print(f"  TRAIN {m.date}  r = {m.pearson_r:+.4f}  rmse = {m.rmse:.4f}")
    print(f"  TRAIN MEAN      r = {np.mean([m.pearson_r for m in train]):+.4f}  "
          f"rmse = {np.mean([m.rmse for m in train]):.4f}")
    print(f"    (published train mean r = {PUBLISHED['train_r_mean_after']:+.4f})")
    print()

    ok = True
    for m in test:
        tgt = TARGET_PER_DAY_R[m.date]
        d = m.pearson_r - tgt
        good = abs(d) < 0.01
        ok &= good
        print(f"  TEST  {m.date}  r = {m.pearson_r:+.4f}  "
              f"(CDS {tgt:+.4f}, delta {d:+.4f})  "
              f"{'ok' if good else 'DIVERGED'}")

    mean_r = float(np.mean([m.pearson_r for m in test]))
    mean_rmse = float(np.mean([m.rmse for m in test]))
    dr = mean_r - TARGET_TEST_R
    drmse = mean_rmse - TARGET_TEST_RMSE
    print()
    print(f"  TEST MEAN  r    = {mean_r:+.4f}   CDS {TARGET_TEST_R:+.4f}   "
          f"delta {dr:+.4f}")
    print(f"  TEST MEAN  RMSE = {mean_rmse:.4f}   CDS {TARGET_TEST_RMSE:.4f}   "
          f"delta {drmse:+.4f}")

    # 0.01 in r is well inside the published bootstrap CI half-width (~0.033)
    # and is the level at which "same data, different mirror" should agree.
    verdict = ok and abs(dr) < 0.01
    print()
    print(f"  VERDICT: {'ARCO-ERA5 MATCHES CDS' if verdict else 'MISMATCH - investigate'}")

    (OUT / "arco_era5_validation.json").write_text(json.dumps({
        "published_cds": {
            "test_r_mean": TARGET_TEST_R,
            "test_rmse_mean": TARGET_TEST_RMSE,
            "test_per_day_r": TARGET_PER_DAY_R,
        },
        "arco": {
            "test_r_mean": mean_r,
            "test_rmse_mean": mean_rmse,
            "test_per_day_r": {m.date: float(m.pearson_r) for m in test},
            "train_r_mean": float(np.mean([m.pearson_r for m in train])),
        },
        "delta_test_r_mean": dr,
        "matches": bool(verdict),
    }, indent=2))
    print(f"  wrote outputs/arco_era5_validation.json")


if __name__ == "__main__":
    main()
