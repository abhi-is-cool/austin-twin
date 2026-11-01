"""Paired bootstrap CI for the ERA5 contribution, under ARCO coefficients.

paper_v2 Table II reported the ERA5 contribution as a paired difference,
Delta r = +0.0105 with 95% CI [+0.0051, +0.0160], computed under CDS
forcing and the CDS-fitted coefficients. paper_v3 carries the ARCO
recalibration, for which only the point estimate (+0.007) was available, so
the CI was correctly left unstated rather than reused.

This recomputes it with the same estimator `bootstrap_uncertainty.py` uses
(6x6 blocks, B = 2000, one shared tile-origin draw per replicate applied to
both configurations), so the interval is directly comparable to the
published one.

The pairing is what makes this the right test: the two configurations differ
by ~0.007 while each individual CI is ~0.03 wide, so unpaired intervals
overlap almost completely and say nothing. Sharing the resampled blocks
cancels the spatial-sampling noise common to both.

  A = ship, ARCO-recalibrated coefficients, ARCO-ERA5 forcing
  B = two-stage config, synthetic diurnal forcing (the pre-ERA5 baseline)

Valid masks are intersected across A and B first, as the estimator requires.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_uncertainty import (  # noqa: E402
    BLOCK_SIDE, N_BOOT, SEED, _peak_frame, paired_delta_r,
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

PUBLISHED_V2 = (0.0105, 0.0051, 0.0160)  # CDS-era paired delta and CI


def main() -> None:
    print("[setup] grid, land cover, MODIS, ARCO forcing...")
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=AUSTIN.worldcover_tile)
    city_mask = landuse["city_mask"].values.astype(bool)
    forcings = {d: forcing_for_modis_date(AUSTIN, d, cache_dir=RAW / "arco_era5")
                for d in TEST_DATES}

    base = SimConfig(duration_hours=48.0)
    cfg_a = replace(load_calibrated_config(OUT / "calibrated_config_arco.json",
                                           base_config=base), use_pm_et=False)
    cfg_b = replace(load_calibrated_config(OUT / "calibrated_config_two_stage.json",
                                           base_config=base), use_pm_et=False)

    days_a, days_b = [], []
    for d in TEST_DATES:
        obs = fetch_aqua_lst(d, grid, RAW / "modis")
        sim_a = _peak_frame(run(landuse, cfg_a, forcing=forcings[d]))
        sim_b = _peak_frame(run(landuse, cfg_b))  # synthetic diurnal
        # One shared mask so the pairing is exact.
        valid = (np.isfinite(sim_a) & np.isfinite(sim_b)
                 & np.isfinite(obs) & city_mask)
        days_a.append((sim_a, obs, valid))
        days_b.append((sim_b, obs, valid))
        print(f"  {d}: {int(valid.sum())} jointly-valid cells")

    print(f"\n[bootstrap] B={N_BOOT}, block side={BLOCK_SIDE}, seed={SEED}")
    res = paired_delta_r(days_a, days_b, B=N_BOOT, L=BLOCK_SIDE, seed=SEED)
    lo, hi = res["delta_ci"]
    delta = res["delta_hat"]

    print()
    print("=" * 70)
    print("PAIRED ERA5 CONTRIBUTION (ARCO coefficients)")
    print("=" * 70)
    print(f"  Delta r = {delta:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  published (v2, CDS)  = {PUBLISHED_V2[0]:+.4f} "
          f"[{PUBLISHED_V2[1]:+.4f}, {PUBLISHED_V2[2]:+.4f}]")
    excludes_zero = lo > 0 or hi < 0
    print(f"  excludes zero: {'YES' if excludes_zero else 'NO'}")
    if not excludes_zero:
        print("  -> the ERA5 gain is NOT distinguishable from zero under the")
        print("     ARCO recalibration, unlike the CDS-era result.")

    (OUT / "paired_era5_ci_arco.json").write_text(json.dumps({
        "delta_hat": delta, "delta_ci": [lo, hi],
        "B": N_BOOT, "block_side": BLOCK_SIDE, "seed": SEED,
        "published_v2_cds": {"delta": PUBLISHED_V2[0],
                             "ci": list(PUBLISHED_V2[1:])},
        "excludes_zero": bool(excludes_zero),
    }, indent=2))
    print(f"\nwrote outputs/paired_era5_ci_arco.json")


if __name__ == "__main__":
    main()
