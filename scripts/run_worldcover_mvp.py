"""End-to-end run on ESA WorldCover land cover for Austin.

Pipeline:
  1. Fetch Austin city-limits boundary (cached).
  2. Build 500 m grid.
  3. Fetch + resample ESA WorldCover 10 m -> 500 m (cached locally after first
     remote read; subsequent runs are instant).
  4. Plot the land-use channels.
  5. Run the baseline 48 h simulation; write animated GIF.
  6. Run the canonical counterfactual scenarios on the real-data baseline.
  7. Write a scenario comparison plot and a numerical summary.
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from austin_twin.calibration import load_calibrated_config
from austin_twin.counterfactual import CANONICAL_SCENARIOS, run_scenarios, summarize
from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.simulator import SimConfig, run, safe_dt
from austin_twin.viz import animate_temperature, plot_landuse, plot_scenario_comparison
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CAL = OUT / "calibrated_config.json"

# Grid resolution (m). Override with AUSTIN_GRID_M env var, e.g.
# AUSTIN_GRID_M=100 python scripts/run_worldcover_mvp.py
GRID_RES_M = float(os.environ.get("AUSTIN_GRID_M", "500"))


def _make_config(dx_m: float) -> SimConfig:
    """Load calibrated config (if available), then CFL-adjust dt for `dx_m`.

    The frame_stride is chosen so the stored history has roughly the same number
    of frames regardless of resolution (~ one frame every 10 simulated minutes).
    """
    base = SimConfig(duration_hours=48.0)
    if CAL.exists():
        print(f"[config] loading calibrated coefficients from {CAL.name}")
        cfg = load_calibrated_config(CAL, base_config=base)
    else:
        print("[config] using default SimConfig (no calibration file found)")
        cfg = base
    dt_max = safe_dt(cfg.diffusion_m2_s, dx_m)
    dt = min(cfg.dt_seconds, dt_max)
    # Round to a nice number <= dt_max.
    dt = float(int(dt))
    stride = max(1, int(round(600.0 / dt)))  # ~one stored frame per 10 sim min
    print(f"[config] grid {dx_m:.0f} m  ->  dt = {dt:.0f} s (CFL-safe; limit {dt_max:.1f} s), "
          f"frame_stride = {stride}")
    return replace(cfg, dt_seconds=dt, frame_stride=stride)


def main() -> None:
    res_tag = f"{int(GRID_RES_M)}m"
    # Suffix output paths only when overriding the default 500 m grid, so the
    # canonical run (no env var) writes to the unsuffixed paths the README
    # links to.
    suffix = "" if GRID_RES_M == 500.0 else f"{suffix}"
    print(f"[1/5] Austin boundary + {res_tag} grid...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=GRID_RES_M)
    print(f"      grid {grid.shape}, {int(grid.mask.sum())} cells inside city")

    print("[2/5] ESA WorldCover land cover...")
    # 500 m cache lives at the parent dir for backwards compatibility; other
    # resolutions get their own subdirectory.
    wc_cache = RAW / "worldcover" if GRID_RES_M == 500.0 else RAW / "worldcover" / res_tag
    landuse = build_worldcover_landuse(boundary, grid, cache_dir=wc_cache)
    plot_landuse(landuse, OUT / f"landuse_worldcover{suffix}.png")

    # Print channel statistics so we can see this is meaningfully different from OSM.
    mask = landuse["city_mask"].values
    for ch in ("impervious_frac", "vegetation_frac", "water_mask"):
        vals = landuse[ch].values[mask]
        print(f"      {ch:>18}: mean={vals.mean():.3f}  median={np.median(vals):.3f}  "
              f"max={vals.max():.3f}  p95={np.percentile(vals, 95):.3f}")

    print("[3/5] Baseline 48 h simulation...")
    cfg = _make_config(GRID_RES_M)
    result = run(landuse, cfg)
    animate_temperature(result, landuse, OUT / f"temperature_worldcover{suffix}.gif", stride=6, fps=8)

    finite = result.temperature[np.isfinite(result.temperature)]
    print(f"      T range: {finite.min():.2f} - {finite.max():.2f} °C")
    # Peak UHI (max - min within a single frame, citywide max over time).
    per_frame_uhi = np.array([
        np.nanmax(f) - np.nanmin(f) for f in result.temperature
        if np.isfinite(f).any()
    ])
    print(f"      peak instantaneous UHI: {per_frame_uhi.max():.2f} °C")

    # Skip counterfactuals at fine resolutions — they take minutes each.
    skip_scenarios = GRID_RES_M < 250.0
    if skip_scenarios:
        print(f"[4/5] skipping {len(CANONICAL_SCENARIOS)} counterfactuals "
              f"(grid {res_tag} is too fine; would take minutes each)")
    else:
        print(f"[4/5] Running {len(CANONICAL_SCENARIOS)} counterfactuals on real data...")
        runs = run_scenarios(landuse, CANONICAL_SCENARIOS, config=cfg)
        plot_scenario_comparison(runs, OUT / f"scenarios_worldcover{suffix}.png")

        table = summarize(runs)
        print()
        print(table)
        (OUT / f"scenario_summary_worldcover{suffix}.txt").write_text(table + "\n")

    print("\n[5/5] done.")
    print(f"  - {OUT / f'landuse_worldcover{suffix}.png'}")
    print(f"  - {OUT / f'temperature_worldcover{suffix}.gif'}")
    if not skip_scenarios:
        print(f"  - {OUT / f'scenarios_worldcover{suffix}.png'}")
        print(f"  - {OUT / f'scenario_summary_worldcover{suffix}.txt'}")


if __name__ == "__main__":
    main()
