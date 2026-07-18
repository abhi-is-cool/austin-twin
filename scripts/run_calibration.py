"""Calibrate simulator coefficients against multi-day MODIS LST.

Train days: 2024-08-11, 2024-08-16, 2024-08-19 (all clear-sky, >98% coverage)
Hold-out:   2024-08-21, 2024-08-22 (also clear, withheld from optimization)

Writes a calibrated SimConfig as JSON, prints train/test metrics, and emits a
re-validation plot for Aug 19 using the calibrated config (so we can directly
compare against the uncalibrated outputs/validation_modis.png).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from austin_twin.calibration import (
    CalibrationResult, calibrate, evaluate, format_metrics_table,
)
from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig, run
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

TRAIN_DATES = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST_DATES = ["2024-08-21", "2024-08-22"]


def _load_modis(dates: list[str], grid) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for d in dates:
        print(f"  fetching {d}...")
        out[d] = fetch_aqua_lst(d, grid, RAW / "modis")
    return out


def _peak_frame(T: np.ndarray, times_h: np.ndarray) -> tuple[np.ndarray, float]:
    second_day = times_h >= 24.0
    means = np.array([
        float(np.nanmean(T[i])) if np.isfinite(T[i]).any() else -np.inf
        for i in range(T.shape[0])
    ])
    masked = np.where(second_day, means, -np.inf)
    idx = int(np.argmax(masked))
    return T[idx], float(times_h[idx])


def _save_revalidation_plot(landuse, cfg: SimConfig, lst: np.ndarray, label: str, out_path: Path) -> None:
    result = run(landuse, cfg)
    T_peak, t_h = _peak_frame(result.temperature, result.times_hours)
    valid = np.isfinite(T_peak) & np.isfinite(lst)
    obs_anom = lst - np.nanmean(lst[valid])
    sim_anom = T_peak - np.nanmean(T_peak[valid])
    obs_anom_v = obs_anom[valid]; sim_anom_v = sim_anom[valid]
    rmse = float(np.sqrt(np.mean((obs_anom_v - sim_anom_v) ** 2)))
    r = float(np.corrcoef(obs_anom_v, sim_anom_v)[0, 1])

    extent = (
        float(landuse["x"].min()), float(landuse["x"].max()),
        float(landuse["y"].min()), float(landuse["y"].max()),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    vmax = float(max(np.nanpercentile(np.abs(obs_anom), 98),
                     np.nanpercentile(np.abs(sim_anom), 98)))

    im0 = axes[0].imshow(obs_anom, extent=extent, origin="upper", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"MODIS anomaly  (Aug 19 2024)"); axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im0, ax=axes[0], shrink=0.85, label="°C")

    im1 = axes[1].imshow(sim_anom, extent=extent, origin="upper", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"Simulator anomaly ({label})\nr = {r:+.3f}, RMSE = {rmse:.2f} °C")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], shrink=0.85, label="°C")

    # Scatter
    axes[2].scatter(obs_anom_v, sim_anom_v, s=6, alpha=0.3)
    lim = max(abs(obs_anom_v.min()), abs(obs_anom_v.max()),
              abs(sim_anom_v.min()), abs(sim_anom_v.max()))
    axes[2].plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
    axes[2].set_xlabel("MODIS anomaly (°C)"); axes[2].set_ylabel("Simulator anomaly (°C)")
    axes[2].set_title(f"Per-cell scatter\nr = {r:+.3f}, RMSE = {rmse:.2f} °C")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle(f"Validation on Aug 19 2024 — {label}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    print("[1/5] Austin grid + WorldCover land cover...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, cache_dir=RAW / "worldcover")

    print("[2/5] MODIS train days...")
    modis_train = _load_modis(TRAIN_DATES, grid)
    print("[3/5] MODIS test days...")
    modis_test = _load_modis(TEST_DATES, grid)

    print("[4/5] running calibration...")
    base = SimConfig(duration_hours=48.0)
    cal: CalibrationResult = calibrate(
        landuse, modis_train, modis_test,
        base_config=base, popsize=8, maxiter=12, verbose=True,
    )

    print("\n----- TRAIN metrics -----")
    print(format_metrics_table(cal.train_metrics_before, cal.train_metrics_after, "train"))
    print("\n----- TEST  metrics -----")
    print(format_metrics_table(cal.test_metrics_before, cal.test_metrics_after, "test"))

    print("\n----- OPTIMAL COEFFICIENTS -----")
    for name, val in cal.optimal_coefs.items():
        default = getattr(SimConfig(), name)
        print(f"  {name:<26} default={default:>7.3f}   optimal={val:>7.3f}")

    # Persist results.
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibrated_config.json").write_text(json.dumps({
        **cal.optimal_coefs,
        "train_rmse_mean_before": float(np.mean([m.rmse for m in cal.train_metrics_before])),
        "train_rmse_mean_after": float(np.mean([m.rmse for m in cal.train_metrics_after])),
        "test_rmse_mean_before": float(np.mean([m.rmse for m in cal.test_metrics_before])),
        "test_rmse_mean_after": float(np.mean([m.rmse for m in cal.test_metrics_after])),
        "train_r_mean_before": float(np.mean([m.pearson_r for m in cal.train_metrics_before])),
        "train_r_mean_after": float(np.mean([m.pearson_r for m in cal.train_metrics_after])),
        "test_r_mean_before": float(np.mean([m.pearson_r for m in cal.test_metrics_before])),
        "test_r_mean_after": float(np.mean([m.pearson_r for m in cal.test_metrics_after])),
        "history": cal.history,
    }, indent=2))

    print("\n[5/5] re-validation plots (Aug 19) for direct before/after comparison...")
    _save_revalidation_plot(landuse, base, modis_train["2024-08-19"],
                             label="uncalibrated", out_path=OUT / "validation_uncalibrated.png")
    _save_revalidation_plot(landuse, cal.optimal_config, modis_train["2024-08-19"],
                             label="calibrated", out_path=OUT / "validation_calibrated.png")

    print(f"\ndone. wrote {OUT / 'calibrated_config.json'}, "
          f"{OUT / 'validation_uncalibrated.png'}, {OUT / 'validation_calibrated.png'}.")


if __name__ == "__main__":
    main()
