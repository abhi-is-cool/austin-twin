"""Reproduce the published single-stage MODIS scores through both simulators.

Target (paper Table II row 1 / NUMBERS.md section 5 head-to-head):

    single-stage coefficients, synthetic forcing, current mixed-layer
    water defaults, held-out Aug 21 + Aug 22 2024
        mean Pearson r = +0.850      mean RMSE = 1.089 C

Note the labelling wrinkle this pins down: `outputs/calibrated_config.json`
records +0.855 / 1.070, because those coefficients were *calibrated* under
the legacy proportional-damping water buffer that the simulator no longer
implements. Re-scoring them under the current mixed-layer water physics is
what gives +0.850 / 1.089, and that is the number the paper prints.

Everything this script fetches is zero-authentication: the OSM boundary,
the WorldCover COG on public S3, and MODIS via the Planetary Computer STAC
anonymous signer. No ERA5, hence no CDS credentials.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from austin_twin import simulator_torch as st
from austin_twin.calibration import (
    _anomaly_stats,
    _peak_frame,
    load_calibrated_config,
)
from austin_twin.cities import AUSTIN
from austin_twin.forcing import Forcing
from austin_twin.grid import build_grid, fetch_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import run
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

TRAIN_DATES = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST_DATES = ["2024-08-21", "2024-08-22"]

# Published targets for the held-out test split.
TARGET_R = 0.850
TARGET_RMSE = 1.089
# Per-day, from NUMBERS.md section 5 (these are the legacy-water values, so
# they are reported for context but not asserted against).
LEGACY_PER_DAY_R = {"2024-08-21": 0.828, "2024-08-22": 0.882}


def load_inputs(resolution_m: float = 500.0):
    print("[data] Austin boundary via OSM...")
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")

    print(f"[data] building {resolution_m:.0f} m grid...")
    grid = build_grid(boundary, resolution_m=resolution_m)
    ny, nx = grid.shape
    print(f"[data]   grid {ny}x{nx}, {int(grid.mask.sum())} cells inside city limits")

    print("[data] ESA WorldCover v200 (public S3 COG)...")
    landuse = build_worldcover_landuse(
        boundary, grid, RAW, tile=AUSTIN.worldcover_tile
    )

    modis = {}
    for date in TRAIN_DATES + TEST_DATES:
        print(f"[data] MODIS Aqua LST {date} (Planetary Computer STAC)...")
        modis[date] = fetch_aqua_lst(date, grid, RAW / "modis")

    return grid, landuse, modis


def score(peak: np.ndarray, modis: dict[str, np.ndarray], dates: list[str]):
    out = {}
    for d in dates:
        rmse, r, _, _ = _anomaly_stats(peak, modis[d])
        out[d] = (r, rmse)
    return out


def summarize(label: str, per_day: dict[str, tuple[float, float]]) -> tuple[float, float]:
    mean_r = float(np.mean([v[0] for v in per_day.values()]))
    mean_rmse = float(np.mean([v[1] for v in per_day.values()]))
    days = "  ".join(f"{d}: r={v[0]:+.4f} rmse={v[1]:.4f}" for d, v in per_day.items())
    print(f"  {label:<26} {days}")
    print(f"  {'':<26} MEAN r={mean_r:+.4f}  rmse={mean_rmse:.4f}")
    return mean_r, mean_rmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=float, default=500.0)
    args = ap.parse_args()

    grid, landuse, modis = load_inputs(args.resolution)

    cfg = load_calibrated_config(ROOT / "outputs" / "calibrated_config.json")
    print()
    print("single-stage coefficients under current mixed-layer water defaults:")
    for name in st.COEF_NAMES:
        print(f"    {name:<24} {getattr(cfg, name):.6f}")
    print(f"    water_heat_capacity_ratio {cfg.water_heat_capacity_ratio}")
    print(f"    water_evap_w_m2          {cfg.water_evap_w_m2}")

    # The synthetic forcing `run()` would build internally; constructed
    # explicitly so both simulators are driven by the identical object.
    forcing = Forcing.synthetic_diurnal(
        duration_hours=cfg.duration_hours,
        air_temp_mean_c=cfg.air_temp_c,
        air_temp_amplitude_c=cfg.diurnal_amplitude_c,
        dewpoint_c=cfg.default_dewpoint_c,
        wind_speed_m_s=cfg.default_wind_speed_m_s,
        peak_solar_w_m2=cfg.peak_solar_w_m2,
        pressure_pa=cfg.default_pressure_pa,
    )

    results: dict[str, tuple[float, float]] = {}

    print()
    print("=" * 78)
    print("NUMPY (reference implementation)")
    print("=" * 78)
    res = run(landuse, cfg, forcing=forcing)
    peak_np = _peak_frame(res.temperature, res.times_hours)
    print(f"  peak frame: mean {np.nanmean(peak_np):.2f} C, "
          f"{int(np.isfinite(peak_np).sum())} valid cells")
    summarize("TRAIN", score(peak_np, modis, TRAIN_DATES))
    results["numpy"] = summarize("TEST (held out)", score(peak_np, modis, TEST_DATES))

    import torch

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        for compiled in ([False, True] if dev == "cuda" else [False]):
            tag = f"torch {dev}" + (" +compile" if compiled else "")
            print()
            print("=" * 78)
            print(tag.upper())
            print("=" * 78)
            fields = st.static_fields_from_landuse(landuse, device=dev)
            fs = st.forcing_series(forcing, cfg, device=dev)
            out = st.run_batch(fields, {}, fs, cfg, compile_step=compiled)
            peak_t = out.detach().cpu().numpy()[0]

            both = np.isfinite(peak_np) & np.isfinite(peak_t)
            dmax = float(np.abs(peak_np[both] - peak_t[both]).max())
            print(f"  max |diff| vs numpy peak frame: {dmax:.3e} C")
            results[tag] = summarize("TEST (held out)", score(peak_t, modis, TEST_DATES))

    # ---------------- verdict ----------------
    print()
    print("=" * 78)
    print("VERDICT vs PUBLISHED")
    print("=" * 78)
    print(f"  published (paper Table II / NUMBERS.md sec 5): "
          f"r = {TARGET_R:+.3f}, RMSE = {TARGET_RMSE:.3f} C")
    print()
    ok = True
    for label, (r, rmse) in results.items():
        dr, drmse = r - TARGET_R, rmse - TARGET_RMSE
        # The published figures are quoted to 3 decimals, so agreement can
        # only be asserted at that granularity.
        good = abs(dr) < 5e-4 and abs(drmse) < 5e-4
        ok &= good
        print(f"  {label:<22} r = {r:+.4f} ({dr:+.4f})   "
              f"RMSE = {rmse:.4f} ({drmse:+.4f})   "
              f"{'MATCH' if good else 'MISMATCH'}")

    print()
    print("  per-day, for context (NUMBERS.md sec 5 quotes legacy-water values):")
    numpy_days = score(peak_np, modis, TEST_DATES)
    for d in TEST_DATES:
        print(f"    {d}  reproduced r = {numpy_days[d][0]:+.4f}   "
              f"legacy-water published r = {LEGACY_PER_DAY_R[d]:+.4f}")

    out_path = ROOT / "outputs" / "reproduction_check.json"
    out_path.write_text(json.dumps({
        "target": {"test_r": TARGET_R, "test_rmse": TARGET_RMSE},
        "reproduced": {k: {"test_r": v[0], "test_rmse": v[1]}
                       for k, v in results.items()},
        "all_match": bool(ok),
    }, indent=2))
    print(f"\n  wrote {out_path.relative_to(ROOT)}")
    print(f"\n  RESULT: {'REPRODUCED' if ok else 'DID NOT REPRODUCE'}")


if __name__ == "__main__":
    main()
