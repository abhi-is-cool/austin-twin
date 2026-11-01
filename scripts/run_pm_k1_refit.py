"""Refit the two K=1 PM variants with k_ET excluded, for protocol parity.

Audit finding (2): paper Table tab:pm states every variant was recalibrated
with k_ET excluded from the search, but the two K=1 rows carried over from
the earlier sweep, in which k_ET was still being searched. Same protocol as
the other rows, so the table is internally consistent.
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
RAW, OUT = ROOT / "data" / "raw", ROOT / "outputs"
TRAIN = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST = ["2024-08-21", "2024-08-22"]
PUBLISHED = {"PM, air VPD, K=1": 0.485, "PM, surface VPD, K=1": 0.509}

boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
grid = build_grid(boundary, resolution_m=500.0)
landuse = build_worldcover_landuse(boundary, grid, RAW, tile=AUSTIN.worldcover_tile)
tr = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TRAIN}
te = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TEST}
fo = {d: forcing_for_modis_date(AUSTIN, d, cache_dir=RAW / "arco_era5") for d in TRAIN + TEST}
ship = load_calibrated_config(OUT / "calibrated_config_arco.json",
                              base_config=SimConfig(duration_hours=48.0))
D = float(ship.diffusion_m2_s)

rows = []
print(f"{'variant':<26} {'refit r':>9} {'published':>11} {'delta':>8}")
print("-" * 58)
for label, mode in (("PM, air VPD, K=1", "air"), ("PM, surface VPD, K=1", "surface")):
    cfg = replace(ship, use_pm_et=True, pm_vpd_mode=mode, pm_lai_k=1.0)
    cal = calibrate(landuse, tr, te, base_config=cfg,
                    fixed_coefs={"diffusion_m2_s": D, "et_coeff": float(cfg.et_coeff)},
                    forcings=fo, popsize=8, maxiter=12, seed=0, verbose=False)
    r = float(np.mean([m.pearson_r for m in cal.test_metrics_after]))
    p = PUBLISHED[label]
    print(f"{label:<26} {r:>+9.4f} {p:>+11.3f} {r - p:>+8.4f}")
    rows.append(dict(variant=label, refit_test_r=r, published=p, delta=r - p))
(OUT / "pm_k1_refit.json").write_text(json.dumps(rows, indent=2))
print("\nwrote outputs/pm_k1_refit.json")
