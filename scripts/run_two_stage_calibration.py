"""Two-stage calibration: literature-anchored diffusion + MODIS-fit absorptions.

Motivation: in the single-stage calibration, `diffusion_m2_s` was free to take
any value that helped match the MODIS baseline pattern. It landed at 18.3 m²/s,
which gives a patch half-cooling distance of ~610 m — 2-6x longer than the
100-300 m range published in the urban-forestry-cooling literature. The
diffusion coefficient was doing double duty: baseline smoothing AND controlling
the spatial extent of intervention effects, with no constraint on the latter.

This script splits the calibration into two stages with separate objectives:

  Stage 1: anchor diffusion to literature.
    Sweep D over [1, 50] m²/s, run a 1 km canopy-patch experiment per value,
    extract the half-cooling distance, and pick the D that lands closest to
    the 200 m midpoint of the literature band.

  Stage 2: refit absorption/ET against MODIS.
    Hold diffusion fixed at the Stage-1 value; run differential-evolution on
    the remaining 4 coefficients (absorption_impervious/vegetation/water,
    et_coeff) against the same three Aug 2024 training days.

We then report:
  - the trade-off in MODIS fit quality (RMSE before/after on train + test);
  - the trade-off in cooling-decay length (before: 610 m, after: ~target);
  - the resulting coefficient values vs the single-stage calibration.

Output: outputs/calibrated_config_two_stage.json and a diagnostic figure.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from austin_twin.calibration import (
    CalibrationResult, calibrate, evaluate, format_metrics_table, load_calibrated_config,
)
from austin_twin.cooling_response import (
    find_D_for_target_half_cooling, run_full_analysis,
)
from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CAL_SINGLE = OUT / "calibrated_config.json"
CAL_TWO = OUT / "calibrated_config_two_stage.json"

TRAIN_DATES = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST_DATES = ["2024-08-21", "2024-08-22"]
TARGET_HALF_COOLING_M = 200.0  # midpoint of 100-300 m literature band
D_SEARCH_RANGE = (1.0, 50.0)
D_SEARCH_N = 12


def _load_modis(dates: list[str], grid) -> dict[str, np.ndarray]:
    return {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in dates}


def main() -> None:
    print("[1/6] Austin grid + WorldCover land cover...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, cache_dir=RAW / "worldcover")

    print("[2/6] Loading single-stage calibrated config (for comparison)...")
    base = SimConfig(duration_hours=48.0)
    single_cfg = load_calibrated_config(CAL_SINGLE, base_config=base) if CAL_SINGLE.exists() else base
    print(f"      single-stage diffusion_m2_s = {single_cfg.diffusion_m2_s:.3f}")

    print(f"[3/6] STAGE 1: sweeping D to find target half-cooling = {TARGET_HALF_COOLING_M:.0f} m...")
    # Start from the single-stage config so absorptions/ET are reasonable.
    D_target, sweep = find_D_for_target_half_cooling(
        landuse, single_cfg, target_half_cooling_m=TARGET_HALF_COOLING_M,
        D_range=D_SEARCH_RANGE, n=D_SEARCH_N,
    )
    print(f"      sweep results (D, half_cooling):")
    for row in sweep:
        if np.isfinite(row["half_cooling_m"]):
            print(f"        D = {row['D']:>6.2f}  ->  half-cooling = {row['half_cooling_m']:>5.0f} m  "
                  f"(λ = {row['decay_length_m']:>5.0f} m, peak ΔT = {row['peak_dt_c']:.2f} °C)")
        else:
            print(f"        D = {row['D']:>6.2f}  ->  (no fit)")
    print(f"      -> selected diffusion_m2_s = {D_target:.3f} m²/s")

    print("[4/6] MODIS train + test days...")
    modis_train = _load_modis(TRAIN_DATES, grid)
    modis_test = _load_modis(TEST_DATES, grid)

    print(f"[5/6] STAGE 2: DE on 4 absorption/ET coefs with diffusion pinned to {D_target:.3f}...")
    cal: CalibrationResult = calibrate(
        landuse, modis_train, modis_test,
        base_config=base,
        fixed_coefs={"diffusion_m2_s": D_target},
        popsize=8, maxiter=12, verbose=True,
    )

    print("\n----- TRAIN metrics (two-stage) -----")
    print(format_metrics_table(cal.train_metrics_before, cal.train_metrics_after, "train"))
    print("\n----- TEST  metrics (two-stage) -----")
    print(format_metrics_table(cal.test_metrics_before, cal.test_metrics_after, "test"))

    print("\n----- COEFFICIENTS: single-stage vs two-stage -----")
    for name in ("absorption_impervious", "absorption_vegetation", "absorption_water",
                 "et_coeff", "diffusion_m2_s"):
        v_single = getattr(single_cfg, name)
        v_two = cal.optimal_coefs[name]
        print(f"  {name:<26} single = {v_single:>8.3f}   two-stage = {v_two:>8.3f}")

    # Persist results.
    CAL_TWO.write_text(json.dumps({
        **cal.optimal_coefs,
        "diffusion_target_half_cooling_m": TARGET_HALF_COOLING_M,
        "diffusion_search_range": list(D_SEARCH_RANGE),
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

    # ----- Trade-off comparison: single vs two-stage on the same test set -----
    print("\n[6/6] direct head-to-head: single-stage vs two-stage on Aug 19, plus cooling response check...")
    single_test = evaluate(landuse, single_cfg, modis_test)
    two_test = cal.train_metrics_after  # this is on the test set under the new config
    two_test_real = evaluate(landuse, cal.optimal_config, modis_test)

    mean_rmse_single = float(np.mean([m.rmse for m in single_test]))
    mean_r_single = float(np.mean([m.pearson_r for m in single_test]))
    mean_rmse_two = float(np.mean([m.rmse for m in two_test_real]))
    mean_r_two = float(np.mean([m.pearson_r for m in two_test_real]))
    print(f"  single-stage MODIS test : r = {mean_r_single:+.3f}  RMSE = {mean_rmse_single:.3f} °C")
    print(f"  two-stage    MODIS test : r = {mean_r_two:+.3f}  RMSE = {mean_rmse_two:.3f} °C")

    # Cooling response under both configs.
    print("\n  running full cooling response under TWO-STAGE config...")
    report_two = run_full_analysis(landuse, cal.optimal_config)
    print(f"  two-stage citywide slope per +10% canopy: {report_two.response_curve.citywide_slope_per_10pct:+.3f} °C")
    print(f"  two-stage local slope per +10% canopy   : {report_two.response_curve.planted_slope_per_10pct:+.3f} °C")
    print(f"  two-stage half-cooling distance         : {report_two.distance_decay.half_cooling_m:.0f} m "
          f"(λ = {report_two.distance_decay.decay_length_m:.0f} m)")

    print("  running full cooling response under SINGLE-STAGE config...")
    report_single = run_full_analysis(landuse, single_cfg)
    print(f"  single-stage citywide slope per +10% canopy: {report_single.response_curve.citywide_slope_per_10pct:+.3f} °C")
    print(f"  single-stage local slope per +10% canopy   : {report_single.response_curve.planted_slope_per_10pct:+.3f} °C")
    print(f"  single-stage half-cooling distance         : {report_single.distance_decay.half_cooling_m:.0f} m "
          f"(λ = {report_single.distance_decay.decay_length_m:.0f} m)")

    # ---- Diagnostic figure ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # (1) Stage 1 sweep: half-cooling vs D
    ax = axes[0]
    valid_sweep = [s for s in sweep if np.isfinite(s["half_cooling_m"])]
    Ds = [s["D"] for s in valid_sweep]
    hcs = [s["half_cooling_m"] for s in valid_sweep]
    ax.semilogx(Ds, hcs, "o-")
    ax.axhspan(100.0, 300.0, alpha=0.15, color="green", label="lit. half-cooling band 100-300 m")
    ax.axvline(D_target, color="red", linestyle="--", label=f"selected D = {D_target:.2f}")
    ax.axhline(TARGET_HALF_COOLING_M, color="gray", linestyle=":", lw=0.8)
    ax.set_xlabel("Diffusion coefficient D (m² / s)")
    ax.set_ylabel("Half-cooling distance (m)")
    ax.set_title("Stage 1: anchor D to literature decay scale")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (2) Cooling response comparison: ΔT vs Δcanopy, single vs two-stage
    ax = axes[1]
    deltas_pct = [d * 100 for d in report_single.response_curve.deltas]
    ax.plot(deltas_pct, report_single.response_curve.citywide_mean_dt, "o-",
            color="C0", label="single-stage, citywide")
    ax.plot(deltas_pct, report_two.response_curve.citywide_mean_dt, "s-",
            color="C1", label="two-stage, citywide")
    ax.plot(deltas_pct, report_single.response_curve.planted_mean_dt, "o--",
            color="C0", alpha=0.6, label="single-stage, planted")
    ax.plot(deltas_pct, report_two.response_curve.planted_mean_dt, "s--",
            color="C1", alpha=0.6, label="two-stage, planted")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Δ canopy applied citywide (%)")
    ax.set_ylabel("ΔT at peak heat (°C)")
    ax.set_title("Cooling response: single vs two-stage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (3) Distance decay: |ΔT| vs distance, single vs two-stage
    ax = axes[2]
    for label, rep, color in [("single-stage", report_single, "C0"), ("two-stage", report_two, "C1")]:
        d = rep.distance_decay.distances_m
        mag = -rep.distance_decay.dt_mean
        valid = np.isfinite(mag)
        ax.plot(d[valid] / 1000.0, mag[valid], "o-", color=color, label=f"{label} (λ = {rep.distance_decay.decay_length_m:.0f} m)")
    ax.axvspan(0.1, 0.3, alpha=0.10, color="green", label="lit. half-cooling band")
    ax.set_xlabel("Distance from canopy patch center (km)")
    ax.set_ylabel("Cooling magnitude |ΔT| (°C)")
    ax.set_title("Distance decay: single vs two-stage")
    ax.set_xlim(0, 4)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Two-stage calibration: literature-anchored D + MODIS-fit absorption/ET", fontsize=12)
    out_plot = OUT / "two_stage_calibration.png"
    fig.savefig(out_plot, dpi=130)
    plt.close(fig)

    print(f"\ndone. wrote {CAL_TWO}, {out_plot}.")


if __name__ == "__main__":
    main()
