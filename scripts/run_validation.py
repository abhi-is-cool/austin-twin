"""Validate the simulator's spatial UHI pattern against MODIS Aqua LST.

We compare the *spatial pattern* of the UHI rather than absolute temperature:
the simulator uses a synthetic 30 +/- 6 C diurnal forcing while the validation
day (2024-08-19, heart of a real Austin heat wave) had a real mean ~46 C in
the satellite scene. Subtracting the citywide spatial mean from each field
removes that absolute-offset bias and lets us judge whether the model puts
the heat in the right *places*.

Metrics reported:
  - Pearson r between simulator and MODIS spatial anomalies
  - RMSE of the difference in anomaly space
  - Mean absolute error per cell
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig, run
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

VALIDATION_DATE = "2024-08-19"  # clear-sky, peak of 2024 Austin heat wave


def _anomaly(field: np.ndarray) -> np.ndarray:
    """Subtract spatial mean over valid cells; preserves NaN."""
    mean = float(np.nanmean(field))
    return field - mean


def main() -> None:
    print("[1/4] Austin grid + WorldCover land cover...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, cache_dir=RAW / "worldcover")

    print(f"[2/4] MODIS Aqua LST for {VALIDATION_DATE}...")
    lst_obs = fetch_aqua_lst(VALIDATION_DATE, grid, RAW / "modis")
    finite_obs = lst_obs[np.isfinite(lst_obs)]
    print(f"      coverage: {finite_obs.size}/{int(grid.mask.sum())} city cells"
          f"  ({100 * finite_obs.size / int(grid.mask.sum()):.1f}%)")
    print(f"      MODIS LST °C: min={finite_obs.min():.2f}  mean={finite_obs.mean():.2f}  "
          f"max={finite_obs.max():.2f}  spread={finite_obs.max()-finite_obs.min():.2f}")

    print("[3/4] Simulator 48 h run, taking peak-heat afternoon frame...")
    result = run(landuse, SimConfig(duration_hours=48.0))
    # Pick the citywide-mean-hottest frame in the second simulated day (after spinup).
    T = result.temperature
    times = result.times_hours
    second_day = times >= 24.0
    mean_t = np.array([
        np.nanmean(T[i]) if np.isfinite(T[i]).any() else -np.inf
        for i in range(T.shape[0])
    ])
    mean_t_2nd = np.where(second_day, mean_t, -np.inf)
    peak_idx = int(np.argmax(mean_t_2nd))
    T_sim = T[peak_idx]
    print(f"      peak-heat frame: t = {times[peak_idx]:.1f} h")
    print(f"      simulator T °C: min={np.nanmin(T_sim):.2f}  mean={np.nanmean(T_sim):.2f}  "
          f"max={np.nanmax(T_sim):.2f}  spread={np.nanmax(T_sim)-np.nanmin(T_sim):.2f}")

    # --- Comparison ---
    print("[4/4] computing fit metrics + writing plot...")
    valid = np.isfinite(lst_obs) & np.isfinite(T_sim)
    n_valid = int(valid.sum())
    obs_anom = _anomaly(np.where(valid, lst_obs, np.nan))
    sim_anom = _anomaly(np.where(valid, T_sim, np.nan))

    # Pearson r and RMSE over the anomaly fields.
    o = obs_anom[valid]
    s = sim_anom[valid]
    r = float(np.corrcoef(o, s)[0, 1])
    rmse = float(np.sqrt(np.mean((o - s) ** 2)))
    mae = float(np.mean(np.abs(o - s)))
    obs_range = float(o.max() - o.min())
    sim_range = float(s.max() - s.min())

    print(f"\n  Spatial UHI pattern fit on {n_valid} jointly-valid cells:")
    print(f"    Pearson r           = {r:+.3f}")
    print(f"    RMSE of anomaly     = {rmse:.2f} °C")
    print(f"    MAE  of anomaly     = {mae:.2f} °C")
    print(f"    MODIS spread        = {obs_range:.2f} °C")
    print(f"    simulator spread    = {sim_range:.2f} °C")

    # --- Plot: 4-panel (MODIS abs, sim abs, MODIS anomaly vs sim anomaly with shared cbar, residual) ---
    extent = (
        float(landuse["x"].min()), float(landuse["x"].max()),
        float(landuse["y"].min()), float(landuse["y"].max()),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)

    # Top-left: MODIS absolute LST.
    ax = axes[0, 0]
    vmin_o, vmax_o = float(np.nanpercentile(lst_obs, 2)), float(np.nanpercentile(lst_obs, 98))
    im = ax.imshow(lst_obs, extent=extent, origin="upper", cmap="inferno",
                   vmin=vmin_o, vmax=vmax_o)
    ax.set_title(f"MODIS Aqua LST  ({VALIDATION_DATE}, ~1:30 pm)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.85, label="°C")

    # Top-right: simulator T at peak.
    ax = axes[0, 1]
    vmin_s, vmax_s = float(np.nanpercentile(T_sim, 2)), float(np.nanpercentile(T_sim, 98))
    im = ax.imshow(T_sim, extent=extent, origin="upper", cmap="inferno",
                   vmin=vmin_s, vmax=vmax_s)
    ax.set_title(f"Simulator T  (t = {times[peak_idx]:.1f} h, synthetic forcing)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.85, label="°C")

    # Bottom-left: side-by-side anomalies on shared scale.
    ax = axes[1, 0]
    vmax_anom = max(float(np.nanpercentile(np.abs(obs_anom), 98)),
                    float(np.nanpercentile(np.abs(sim_anom), 98)))
    im = ax.imshow(obs_anom, extent=extent, origin="upper", cmap="RdBu_r",
                   vmin=-vmax_anom, vmax=vmax_anom)
    ax.set_title("MODIS anomaly (LST - city mean)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.85, label="°C")

    # Bottom-right: simulator anomaly on the same scale, then residual annotated.
    ax = axes[1, 1]
    im = ax.imshow(sim_anom, extent=extent, origin="upper", cmap="RdBu_r",
                   vmin=-vmax_anom, vmax=vmax_anom)
    ax.set_title(f"Simulator anomaly\nPearson r = {r:+.3f}, RMSE = {rmse:.2f} °C, MAE = {mae:.2f} °C")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.85, label="°C")

    fig.suptitle(
        f"Simulator vs MODIS LST — {VALIDATION_DATE}\n"
        f"Comparison is of spatial pattern (anomaly from citywide mean) — "
        f"absolute T differs because simulator uses synthetic forcing.",
        fontsize=12,
    )
    out_path = OUT / "validation_modis.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    # --- Per-cell scatter plot ---
    fig2, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(o, s, s=6, alpha=0.3)
    lim = max(abs(o.min()), abs(o.max()), abs(s.min()), abs(s.max()))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
    ax.set_xlabel("MODIS anomaly (°C)")
    ax.set_ylabel("Simulator anomaly (°C)")
    ax.set_title(f"Per-cell scatter (n = {n_valid})\nPearson r = {r:+.3f}, RMSE = {rmse:.2f} °C")
    ax.legend()
    ax.grid(alpha=0.3)
    fig2.savefig(OUT / "validation_scatter.png", dpi=130)
    plt.close(fig2)

    print(f"\ndone. wrote {OUT / 'validation_modis.png'} and {OUT / 'validation_scatter.png'}.")


if __name__ == "__main__":
    main()
