"""Two-stage calibration: legacy linear-ET term + real ERA5 forcing.

This is the "ship" config after the Phase-2 audit. The PM-ET experiment
(see run_pm_calibration.py + Discussion in NUMBERS.md) is documented as
a negative result; we retain the linear ET term because canonical
FAO-56 PM decouples LE from T_surf, which collapses the spatial-UHI
pattern correlation against MODIS (r 0.75 -> 0.42). The ERA5 forcing
upgrade, however, is neutral to spatial pattern (Δr ≈ -0.01) and gives
per-date absolute temperature fields, so we adopt it independently.

Same two-stage split as run_two_stage_calibration.py:
  Stage 1 -- anchor diffusion to literature-derived half-cooling (200 m).
  Stage 2 -- DE on 4 free coefs (absorptions + et_coeff) against MODIS
             with per-date ERA5 forcing driving each simulator run.

Output: outputs/calibrated_config_era5.json + outputs/era5_calibration.png.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from austin_twin.calibration import (
    calibrate, format_metrics_table, load_calibrated_config,
)
from austin_twin.cooling_response import find_D_for_target_half_cooling, run_full_analysis
from austin_twin.forcing import Forcing
from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CAL_TWO = OUT / "calibrated_config_two_stage.json"       # legacy synthetic-forcing config
CAL_ERA5 = OUT / "calibrated_config_era5.json"

TRAIN_DATES = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST_DATES = ["2024-08-21", "2024-08-22"]
TARGET_HALF_COOLING_M = 200.0
D_SEARCH_RANGE = (1.0, 50.0)
D_SEARCH_N = 12

ERA5_NC = RAW / "era5" / "era5_2024-08-10_2024-08-23.nc"

# Austin CDT = UTC-5 in August. Start each 48h sim at midnight local of the
# day before the MODIS date; peak-heat frame at ~t=37h lands ~13:00 CDT on
# the MODIS date, matching the Aqua overpass (~13:30 local).
UTC_OFFSET_HOURS = 5


def _forcing_for_date(date_iso: str, duration_hours: float = 48.0) -> Forcing:
    d = date.fromisoformat(date_iso)
    prev_day = d - timedelta(days=1)
    return Forcing.from_era5(
        ERA5_NC,
        start_iso=f"{prev_day.isoformat()}T{UTC_OFFSET_HOURS:02d}:00",
        duration_hours=duration_hours,
    )


def _load_modis(dates: list[str], grid) -> dict[str, np.ndarray]:
    return {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in dates}


def _per_day_metrics_block(label: str, metrics_before, metrics_after) -> str:
    """Wider-format per-day table for the paper update."""
    lines = [
        f"----- {label.upper()} per-day metrics (linear-ET + ERA5) -----",
        f"{'date':<12} {'r before':>10} {'r after':>10} {'RMSE before':>13} "
        f"{'RMSE after':>12} {'sim spread':>12} {'obs spread':>12}",
        "-" * 84,
    ]
    for mb, ma in zip(metrics_before, metrics_after):
        lines.append(
            f"{mb.date:<12} {mb.pearson_r:>+10.3f} {ma.pearson_r:>+10.3f} "
            f"{mb.rmse:>12.3f}° {ma.rmse:>11.3f}° {ma.sim_spread:>11.2f}° {ma.obs_spread:>11.2f}°"
        )
    r_b = float(np.mean([m.pearson_r for m in metrics_before]))
    r_a = float(np.mean([m.pearson_r for m in metrics_after]))
    rmse_b = float(np.mean([m.rmse for m in metrics_before]))
    rmse_a = float(np.mean([m.rmse for m in metrics_after]))
    lines.append(
        f"{'MEAN':<12} {r_b:>+10.3f} {r_a:>+10.3f} {rmse_b:>12.3f}° {rmse_a:>11.3f}°"
    )
    return "\n".join(lines)


def main() -> None:
    print("[1/7] Austin grid + WorldCover land cover...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, cache_dir=RAW / "worldcover")

    print("[2/7] Loading legacy two-stage config (synthetic-forcing calibration) for comparison...")
    base = SimConfig(duration_hours=48.0)
    legacy_cfg = load_calibrated_config(CAL_TWO, base_config=base) if CAL_TWO.exists() else base
    legacy_cfg = replace(legacy_cfg, use_pm_et=False)
    print(f"      legacy diffusion_m2_s = {legacy_cfg.diffusion_m2_s:.3f}")

    print(f"[3/7] STAGE 1: sweep D to hit half-cooling = {TARGET_HALF_COOLING_M:.0f} m")
    print("      (uses synthetic forcing; anchors physics decay length, not weather)")
    D_target, sweep = find_D_for_target_half_cooling(
        landuse, legacy_cfg,
        target_half_cooling_m=TARGET_HALF_COOLING_M,
        D_range=D_SEARCH_RANGE, n=D_SEARCH_N,
    )
    for row in sweep:
        if np.isfinite(row["half_cooling_m"]):
            print(f"        D={row['D']:>6.2f}  half-cooling={row['half_cooling_m']:>5.0f} m  "
                  f"λ={row['decay_length_m']:>5.0f} m  peak ΔT={row['peak_dt_c']:.2f} °C")
    print(f"      -> selected diffusion_m2_s = {D_target:.3f} m²/s")

    print("[4/7] MODIS train + test days + per-date ERA5 forcings...")
    modis_train = _load_modis(TRAIN_DATES, grid)
    modis_test = _load_modis(TEST_DATES, grid)
    forcings = {d: _forcing_for_date(d) for d in TRAIN_DATES + TEST_DATES}
    for d, f in forcings.items():
        print(f"      {d}: T_air max={f.t_air_c.max():.1f} °C, T_dew mean={f.t_dew_c.mean():.1f} °C, "
              f"wind mean={f.wind_speed_m_s.mean():.1f} m/s")

    print(f"[5/7] STAGE 2: DE on 4 coefs (3 absorptions + et_coeff), D={D_target:.3f} pinned...")
    cal = calibrate(
        landuse, modis_train, modis_test,
        base_config=replace(legacy_cfg, use_pm_et=False),
        fixed_coefs={"diffusion_m2_s": D_target},
        forcings=forcings,
        popsize=8, maxiter=12, verbose=True,
    )

    train_table = format_metrics_table(cal.train_metrics_before, cal.train_metrics_after, "train")
    test_table = format_metrics_table(cal.test_metrics_before, cal.test_metrics_after, "test")
    train_paper = _per_day_metrics_block("train", cal.train_metrics_before, cal.train_metrics_after)
    test_paper = _per_day_metrics_block("test", cal.test_metrics_before, cal.test_metrics_after)

    print("\n" + train_table)
    print("\n" + test_table)

    print("\n----- Paper-formatted per-day metrics -----\n")
    print(train_paper)
    print()
    print(test_paper)

    print("\n----- COEFFICIENTS: legacy synthetic-forcing vs linear-ET + ERA5 -----")
    for name in ("absorption_impervious", "absorption_vegetation", "absorption_water",
                 "et_coeff", "diffusion_m2_s"):
        v_legacy = getattr(legacy_cfg, name)
        v_new = cal.optimal_coefs[name]
        print(f"  {name:<26} legacy = {v_legacy:>8.3f}   linear+ERA5 = {v_new:>8.3f}")

    print("\n[6/7] Cooling response under linear-ET + ERA5 config (synthetic forcing for analysis)...")
    report_new = run_full_analysis(landuse, cal.optimal_config)
    print(f"  citywide slope per +10% canopy : {report_new.response_curve.citywide_slope_per_10pct:+.3f} °C")
    print(f"  local slope per +10% canopy    : {report_new.response_curve.planted_slope_per_10pct:+.3f} °C")
    print(f"  half-cooling distance          : {report_new.distance_decay.half_cooling_m:.0f} m "
          f"(λ = {report_new.distance_decay.decay_length_m:.0f} m)")

    print("  (comparison) same analysis under legacy synthetic-forcing config...")
    report_legacy = run_full_analysis(landuse, legacy_cfg)
    print(f"  legacy citywide per +10% : {report_legacy.response_curve.citywide_slope_per_10pct:+.3f} °C")
    print(f"  legacy local per +10%    : {report_legacy.response_curve.planted_slope_per_10pct:+.3f} °C")
    print(f"  legacy half-cooling      : {report_legacy.distance_decay.half_cooling_m:.0f} m")

    CAL_ERA5.write_text(json.dumps({
        **cal.optimal_coefs,
        "use_pm_et": False,
        "era5_source": str(ERA5_NC.name),
        "train_dates": TRAIN_DATES,
        "test_dates": TEST_DATES,
        "diffusion_target_half_cooling_m": TARGET_HALF_COOLING_M,
        "train_rmse_mean_before": float(np.mean([m.rmse for m in cal.train_metrics_before])),
        "train_rmse_mean_after": float(np.mean([m.rmse for m in cal.train_metrics_after])),
        "test_rmse_mean_before": float(np.mean([m.rmse for m in cal.test_metrics_before])),
        "test_rmse_mean_after": float(np.mean([m.rmse for m in cal.test_metrics_after])),
        "train_r_mean_before": float(np.mean([m.pearson_r for m in cal.train_metrics_before])),
        "train_r_mean_after": float(np.mean([m.pearson_r for m in cal.train_metrics_after])),
        "test_r_mean_before": float(np.mean([m.pearson_r for m in cal.test_metrics_before])),
        "test_r_mean_after": float(np.mean([m.pearson_r for m in cal.test_metrics_after])),
        "train_per_day_r_after": {m.date: float(m.pearson_r) for m in cal.train_metrics_after},
        "test_per_day_r_after": {m.date: float(m.pearson_r) for m in cal.test_metrics_after},
        "train_per_day_rmse_after": {m.date: float(m.rmse) for m in cal.train_metrics_after},
        "test_per_day_rmse_after": {m.date: float(m.rmse) for m in cal.test_metrics_after},
        "stage1_sweep": sweep,
        "history": cal.history,
    }, indent=2))

    print("[7/7] figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    ax = axes[0]
    valid_sweep = [s for s in sweep if np.isfinite(s["half_cooling_m"])]
    ax.semilogx([s["D"] for s in valid_sweep], [s["half_cooling_m"] for s in valid_sweep], "o-")
    ax.axhspan(100.0, 300.0, alpha=0.15, color="green", label="lit. band 100-300 m")
    ax.axvline(D_target, color="red", ls="--", label=f"selected D = {D_target:.2f}")
    ax.axhline(TARGET_HALF_COOLING_M, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Diffusion coefficient D (m²/s)")
    ax.set_ylabel("Half-cooling distance (m)")
    ax.set_title("Stage 1: decay-length anchoring")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    deltas_pct = [d * 100 for d in report_legacy.response_curve.deltas]
    ax.plot(deltas_pct, report_legacy.response_curve.citywide_mean_dt, "o-", color="C0", label="legacy synthetic, citywide")
    ax.plot(deltas_pct, report_new.response_curve.citywide_mean_dt, "s-", color="C2", label="linear+ERA5, citywide")
    ax.plot(deltas_pct, report_legacy.response_curve.planted_mean_dt, "o--", color="C0", alpha=0.5, label="legacy synthetic, planted")
    ax.plot(deltas_pct, report_new.response_curve.planted_mean_dt, "s--", color="C2", alpha=0.5, label="linear+ERA5, planted")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Δ canopy applied citywide (%)")
    ax.set_ylabel("ΔT at peak heat (°C)")
    ax.set_title("Cooling response: legacy vs linear+ERA5")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for label, rep, color in [("legacy synthetic", report_legacy, "C0"), ("linear+ERA5", report_new, "C2")]:
        d = rep.distance_decay.distances_m
        mag = -rep.distance_decay.dt_mean
        valid = np.isfinite(mag)
        ax.plot(d[valid] / 1000.0, mag[valid], "o-", color=color,
                label=f"{label} (λ = {rep.distance_decay.decay_length_m:.0f} m)")
    ax.axvspan(0.1, 0.3, alpha=0.10, color="green", label="lit. band")
    ax.set_xlabel("Distance from patch center (km)")
    ax.set_ylabel("|ΔT| (°C)")
    ax.set_title("Distance decay: legacy vs linear+ERA5")
    ax.set_xlim(0, 4)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Linear-ET + ERA5 forcing (ship config) vs legacy synthetic-forcing", fontsize=12)
    out_plot = OUT / "era5_calibration.png"
    fig.savefig(out_plot, dpi=130)
    plt.close(fig)

    print(f"\ndone. wrote {CAL_ERA5}, {out_plot}.")


if __name__ == "__main__":
    main()
