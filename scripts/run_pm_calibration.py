"""Two-stage calibration with Penman-Monteith ET + real ERA5 forcing.

This is the Phase-2 counterpart to `run_two_stage_calibration.py`:
  - Same two-stage split (literature-anchored diffusion, then MODIS-fit
    absorptions).
  - `use_pm_et=True` instead of the legacy linear ET term.
  - Per-date ERA5 forcing driving each MODIS training/test comparison.

`et_coeff` is pinned in Stage 2 (PM-ET ignores it) so DE only searches
the three absorptions. The Stage-1 D sweep uses synthetic forcing (its
purpose is to anchor decay length, which is a physics property not a
weather property).

Output: outputs/calibrated_config_pm.json + outputs/pm_calibration.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from austin_twin.calibration import (
    calibrate, evaluate, format_metrics_table, load_calibrated_config,
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
CAL_TWO = OUT / "calibrated_config_two_stage.json"    # legacy for comparison
CAL_PM = OUT / "calibrated_config_pm.json"

TRAIN_DATES = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST_DATES = ["2024-08-21", "2024-08-22"]
TARGET_HALF_COOLING_M = 200.0
D_SEARCH_RANGE = (1.0, 50.0)
D_SEARCH_N = 12

# ERA5 window that covers all training and test dates plus a 1-day buffer.
ERA5_NC = RAW / "era5" / "era5_2024-08-10_2024-08-23.nc"

# Austin is UTC-5 in Aug (CDT). We start each 48-hour simulation at
# midnight local of the day BEFORE the MODIS date, so the second-day peak
# frame at ~t=37h lands around 13:00 CDT on the MODIS date, matching the
# Aqua overpass (~13:30 local).
UTC_OFFSET_HOURS = 5


def _forcing_for_date(date_iso: str, duration_hours: float = 48.0) -> Forcing:
    """Build a 48h ERA5 forcing anchored so peak-heat lands on `date_iso`."""
    from datetime import date, timedelta
    d = date.fromisoformat(date_iso)
    prev_day = d - timedelta(days=1)
    start_utc = f"{prev_day.isoformat()}T{UTC_OFFSET_HOURS:02d}:00"
    return Forcing.from_era5(ERA5_NC, start_iso=start_utc, duration_hours=duration_hours)


def _load_modis(dates: list[str], grid) -> dict[str, np.ndarray]:
    return {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in dates}


def _make_pm_base(single_cfg: SimConfig) -> SimConfig:
    """Start from the single-stage calibrated coefs but switch on PM-ET."""
    from dataclasses import replace
    return replace(single_cfg, use_pm_et=True, duration_hours=48.0)


def main() -> None:
    print("[1/7] Austin grid + WorldCover land cover...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, cache_dir=RAW / "worldcover")

    print("[2/7] Loading two-stage (legacy linear-ET) config for comparison...")
    base_single = SimConfig(duration_hours=48.0)
    single_cfg = load_calibrated_config(CAL_TWO, base_config=base_single) if CAL_TWO.exists() else base_single
    pm_base = _make_pm_base(single_cfg)
    print(f"      linear-ET diffusion_m2_s = {single_cfg.diffusion_m2_s:.3f}")
    print(f"      use_pm_et flipped to True; other coefs inherited")

    print(f"[3/7] STAGE 1: sweeping D to find target half-cooling = {TARGET_HALF_COOLING_M:.0f} m")
    print("      (uses synthetic forcing; D anchors the physics decay length, not weather)")
    D_target, sweep = find_D_for_target_half_cooling(
        landuse, pm_base, target_half_cooling_m=TARGET_HALF_COOLING_M,
        D_range=D_SEARCH_RANGE, n=D_SEARCH_N,
    )
    for row in sweep:
        if np.isfinite(row["half_cooling_m"]):
            print(f"        D = {row['D']:>6.2f}  ->  half-cooling = {row['half_cooling_m']:>5.0f} m  "
                  f"(λ = {row['decay_length_m']:>5.0f} m, peak ΔT = {row['peak_dt_c']:.2f} °C)")
    print(f"      -> selected diffusion_m2_s = {D_target:.3f} m²/s")

    print("[4/7] MODIS train + test days + per-date ERA5 forcings...")
    modis_train = _load_modis(TRAIN_DATES, grid)
    modis_test = _load_modis(TEST_DATES, grid)
    forcings = {d: _forcing_for_date(d) for d in TRAIN_DATES + TEST_DATES}
    for d, f in forcings.items():
        print(f"      {d}: ERA5 T_air max={f.t_air_c.max():.1f} °C, T_dew mean={f.t_dew_c.mean():.1f} °C, "
              f"wind mean={f.wind_speed_m_s.mean():.1f} m/s")

    print(f"[5/7] STAGE 2: DE on 3 absorption coefs with diffusion={D_target:.3f}, PM-ET on, "
          "et_coeff pinned (unused)...")
    cal = calibrate(
        landuse, modis_train, modis_test,
        base_config=pm_base,
        fixed_coefs={
            "diffusion_m2_s": D_target,
            "et_coeff": pm_base.et_coeff,  # value irrelevant under PM; just remove from search space
        },
        forcings=forcings,
        popsize=8, maxiter=12, verbose=True,
    )

    print("\n----- TRAIN metrics (PM-ET + ERA5) -----")
    print(format_metrics_table(cal.train_metrics_before, cal.train_metrics_after, "train"))
    print("\n----- TEST  metrics (PM-ET + ERA5) -----")
    print(format_metrics_table(cal.test_metrics_before, cal.test_metrics_after, "test"))

    print("\n----- COEFFICIENTS: legacy two-stage vs PM+ERA5 -----")
    for name in ("absorption_impervious", "absorption_vegetation", "absorption_water",
                 "et_coeff", "diffusion_m2_s"):
        v_legacy = getattr(single_cfg, name)
        v_pm = cal.optimal_coefs[name]
        note = "  (unused under PM)" if name == "et_coeff" else ""
        print(f"  {name:<26} legacy = {v_legacy:>8.3f}   PM+ERA5 = {v_pm:>8.3f}{note}")

    print("\n[6/7] Cooling response under PM+ERA5 config (synthetic forcing for the analysis run)...")
    report_pm = run_full_analysis(landuse, cal.optimal_config)
    print(f"  citywide slope per +10% canopy: {report_pm.response_curve.citywide_slope_per_10pct:+.3f} °C")
    print(f"  local slope per +10% canopy   : {report_pm.response_curve.planted_slope_per_10pct:+.3f} °C")
    print(f"  half-cooling distance         : {report_pm.distance_decay.half_cooling_m:.0f} m "
          f"(λ = {report_pm.distance_decay.decay_length_m:.0f} m)")

    print("  (comparison) cooling response under legacy two-stage config...")
    report_legacy = run_full_analysis(landuse, single_cfg)
    print(f"  legacy citywide per +10% : {report_legacy.response_curve.citywide_slope_per_10pct:+.3f} °C")
    print(f"  legacy local per +10%    : {report_legacy.response_curve.planted_slope_per_10pct:+.3f} °C")
    print(f"  legacy half-cooling      : {report_legacy.distance_decay.half_cooling_m:.0f} m")

    CAL_PM.write_text(json.dumps({
        **cal.optimal_coefs,
        "use_pm_et": True,
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
    ax.set_title("Stage 1: PM-ET decay-length anchoring")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    deltas_pct = [d * 100 for d in report_legacy.response_curve.deltas]
    ax.plot(deltas_pct, report_legacy.response_curve.citywide_mean_dt, "o-", color="C0", label="legacy linear-ET, citywide")
    ax.plot(deltas_pct, report_pm.response_curve.citywide_mean_dt, "s-", color="C1", label="PM+ERA5, citywide")
    ax.plot(deltas_pct, report_legacy.response_curve.planted_mean_dt, "o--", color="C0", alpha=0.5, label="legacy, planted")
    ax.plot(deltas_pct, report_pm.response_curve.planted_mean_dt, "s--", color="C1", alpha=0.5, label="PM+ERA5, planted")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Δ canopy applied citywide (%)")
    ax.set_ylabel("ΔT at peak heat (°C)")
    ax.set_title("Cooling response: legacy vs PM+ERA5")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for label, rep, color in [("legacy linear-ET", report_legacy, "C0"), ("PM+ERA5", report_pm, "C1")]:
        d = rep.distance_decay.distances_m
        mag = -rep.distance_decay.dt_mean
        valid = np.isfinite(mag)
        ax.plot(d[valid] / 1000.0, mag[valid], "o-", color=color,
                label=f"{label} (λ = {rep.distance_decay.decay_length_m:.0f} m)")
    ax.axvspan(0.1, 0.3, alpha=0.10, color="green", label="lit. band")
    ax.set_xlabel("Distance from patch center (km)")
    ax.set_ylabel("|ΔT| (°C)")
    ax.set_title("Distance decay: legacy vs PM+ERA5")
    ax.set_xlim(0, 4)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("PM-ET + ERA5 calibration vs legacy two-stage linear-ET", fontsize=12)
    out_plot = OUT / "pm_calibration.png"
    fig.savefig(out_plot, dpi=130)
    plt.close(fig)

    print(f"\ndone. wrote {CAL_PM}, {out_plot}.")


if __name__ == "__main__":
    main()
