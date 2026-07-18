"""Validation + counterfactuals under the linear-ET + ERA5 calibrated config.

Companion to `run_era5_calibration.py` (the ship config after the
Phase-2 PM negative-result audit). After that script writes
`outputs/calibrated_config_era5.json`, this one re-runs:
  - 2024-08-19 spatial validation vs MODIS with ERA5 forcing;
  - canonical counterfactual scenarios with a shared ERA5 forcing so
    scenario ΔT isolates the land-use intervention.

Outputs (all *_era5.png / *_era5.txt so we don't overwrite legacy or PM):
  - outputs/validation_modis_era5.png
  - outputs/validation_scatter_era5.png
  - outputs/scenarios_era5.png
  - outputs/scenario_summary_era5.txt
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from austin_twin.calibration import load_calibrated_config
from austin_twin.counterfactual import CANONICAL_SCENARIOS, run_scenarios, summarize
from austin_twin.forcing import Forcing
from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig, run
from austin_twin.viz import plot_scenario_comparison
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CAL_ERA5 = OUT / "calibrated_config_era5.json"
ERA5_NC = RAW / "era5" / "era5_2024-08-10_2024-08-23.nc"
VALIDATION_DATE = "2024-08-19"
UTC_OFFSET_HOURS = 5


def _forcing_for(date_iso: str) -> Forcing:
    prev = date.fromisoformat(date_iso) - timedelta(days=1)
    return Forcing.from_era5(
        ERA5_NC, start_iso=f"{prev.isoformat()}T{UTC_OFFSET_HOURS:02d}:00",
        duration_hours=48.0,
    )


def _era5_config() -> SimConfig:
    base = SimConfig(duration_hours=48.0)
    cfg = load_calibrated_config(CAL_ERA5, base_config=base)
    return replace(cfg, use_pm_et=False)


def _peak_frame(result):
    T, times = result.temperature, result.times_hours
    mean_t = np.array([np.nanmean(f) if np.isfinite(f).any() else -np.inf for f in T])
    masked = np.where(times >= 24.0, mean_t, -np.inf)
    idx = int(np.argmax(masked))
    return T[idx], times[idx], idx


def _anomaly(field: np.ndarray) -> np.ndarray:
    return field - float(np.nanmean(field))


def _run_validation(landuse, grid, cfg: SimConfig) -> None:
    print(f"\n=== Validation vs MODIS on {VALIDATION_DATE} (linear+ERA5) ===")
    lst_obs = fetch_aqua_lst(VALIDATION_DATE, grid, RAW / "modis")
    forcing = _forcing_for(VALIDATION_DATE)
    result = run(landuse, cfg, forcing=forcing)
    T_sim, t_peak, _ = _peak_frame(result)

    finite_obs = lst_obs[np.isfinite(lst_obs)]
    valid = np.isfinite(lst_obs) & np.isfinite(T_sim)
    o = _anomaly(np.where(valid, lst_obs, np.nan))[valid]
    s = _anomaly(np.where(valid, T_sim, np.nan))[valid]
    r = float(np.corrcoef(o, s)[0, 1])
    rmse = float(np.sqrt(np.mean((o - s) ** 2)))
    mae = float(np.mean(np.abs(o - s)))

    print(f"  peak-heat sim frame t = {t_peak:.1f} h")
    print(f"  MODIS   °C: min={finite_obs.min():.2f}  mean={finite_obs.mean():.2f}  "
          f"max={finite_obs.max():.2f}  spread={finite_obs.max()-finite_obs.min():.2f}")
    print(f"  sim T   °C: min={np.nanmin(T_sim):.2f}  mean={np.nanmean(T_sim):.2f}  "
          f"max={np.nanmax(T_sim):.2f}  spread={np.nanmax(T_sim)-np.nanmin(T_sim):.2f}")
    print(f"  Pearson r = {r:+.3f}   RMSE = {rmse:.2f} °C   MAE = {mae:.2f} °C")

    extent = (
        float(landuse["x"].min()), float(landuse["x"].max()),
        float(landuse["y"].min()), float(landuse["y"].max()),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)

    ax = axes[0, 0]
    vmin_o, vmax_o = float(np.nanpercentile(lst_obs, 2)), float(np.nanpercentile(lst_obs, 98))
    im = ax.imshow(lst_obs, extent=extent, origin="upper", cmap="inferno", vmin=vmin_o, vmax=vmax_o)
    ax.set_title(f"MODIS Aqua LST ({VALIDATION_DATE}, ~1:30 pm)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.85, label="°C")

    ax = axes[0, 1]
    vmin_s, vmax_s = float(np.nanpercentile(T_sim, 2)), float(np.nanpercentile(T_sim, 98))
    im = ax.imshow(T_sim, extent=extent, origin="upper", cmap="inferno", vmin=vmin_s, vmax=vmax_s)
    ax.set_title(f"Simulator T (linear-ET + ERA5, t = {t_peak:.1f} h)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.85, label="°C")

    obs_anom = _anomaly(np.where(valid, lst_obs, np.nan))
    sim_anom = _anomaly(np.where(valid, T_sim, np.nan))
    vmax_a = max(float(np.nanpercentile(np.abs(obs_anom), 98)),
                 float(np.nanpercentile(np.abs(sim_anom), 98)))
    ax = axes[1, 0]
    im = ax.imshow(obs_anom, extent=extent, origin="upper", cmap="RdBu_r", vmin=-vmax_a, vmax=vmax_a)
    ax.set_title("MODIS anomaly (LST - city mean)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.85, label="°C")

    ax = axes[1, 1]
    im = ax.imshow(sim_anom, extent=extent, origin="upper", cmap="RdBu_r", vmin=-vmax_a, vmax=vmax_a)
    ax.set_title(f"linear-ET + ERA5 anomaly\nPearson r = {r:+.3f}, RMSE = {rmse:.2f} °C, MAE = {mae:.2f} °C")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.85, label="°C")

    fig.suptitle(
        f"Linear-ET + ERA5 forcing vs MODIS LST — {VALIDATION_DATE}\n"
        "Ship config after Phase-2 PM negative-result audit.",
        fontsize=12,
    )
    out_val = OUT / "validation_modis_era5.png"
    fig.savefig(out_val, dpi=130)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(o, s, s=6, alpha=0.3)
    lim = max(abs(o.min()), abs(o.max()), abs(s.min()), abs(s.max()))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
    ax.set_xlabel("MODIS anomaly (°C)")
    ax.set_ylabel("linear-ET + ERA5 anomaly (°C)")
    ax.set_title(f"Per-cell scatter (n = {valid.sum()})  Pearson r = {r:+.3f}  RMSE = {rmse:.2f} °C")
    ax.legend(); ax.grid(alpha=0.3)
    out_scatter = OUT / "validation_scatter_era5.png"
    fig2.savefig(out_scatter, dpi=130)
    plt.close(fig2)

    print(f"  wrote {out_val} and {out_scatter}")


def _run_counterfactuals(landuse, cfg: SimConfig) -> None:
    print(f"\n=== Counterfactuals under linear-ET + ERA5 (shared {VALIDATION_DATE} forcing) ===")
    forcing = _forcing_for(VALIDATION_DATE)
    runs = run_scenarios(landuse, CANONICAL_SCENARIOS, config=cfg, forcing=forcing)

    plot_scenario_comparison(runs, OUT / "scenarios_era5.png")
    table = summarize(runs)
    print(table)
    (OUT / "scenario_summary_era5.txt").write_text(
        f"# Counterfactuals under linear-ET + ERA5 calibration ({CAL_ERA5.name})\n"
        f"# Shared forcing: ERA5 for {VALIDATION_DATE}\n"
        + table + "\n"
    )
    print(f"  wrote {OUT / 'scenarios_era5.png'} and {OUT / 'scenario_summary_era5.txt'}")


def main() -> None:
    print("[1/3] Austin grid + WorldCover land cover...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, cache_dir=RAW / "worldcover")

    print("[2/3] Loading linear-ET + ERA5 calibrated config...")
    if not CAL_ERA5.exists():
        raise SystemExit(f"missing {CAL_ERA5}; run scripts/run_era5_calibration.py first")
    cfg = _era5_config()
    print(f"      absorptions: imp={cfg.absorption_impervious:.3f}  "
          f"veg={cfg.absorption_vegetation:.3f}  water={cfg.absorption_water:.3f}")
    print(f"      et_coeff = {cfg.et_coeff:.3f}   diffusion_m2_s = {cfg.diffusion_m2_s:.3f}   "
          f"use_pm_et = {cfg.use_pm_et}")

    print("[3/3] validation + counterfactuals...")
    _run_validation(landuse, grid, cfg)
    _run_counterfactuals(landuse, cfg)


if __name__ == "__main__":
    main()
