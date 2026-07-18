"""Extract canopy-cooling metrics in the form the UFOR literature reports.

Three reviewer-defensible numbers from the simulator:

  (1) °C of citywide cooling per +10% canopy   -- slope of sweep
  (2) °C of LOCAL cooling per +10% canopy      -- slope inside planted cells
  (3) Cooling decay length λ (m) around a 1 km canopy patch -- exp fit

Approximate literature bands hard-coded for reference; they reflect roughly
the central range of values reported across multiple UFOR meta-analyses and
city-scale studies. Specific papers should be substituted by the reader/
reviewer based on their preferred sources.

The output is one figure ([1] sweep with both slopes annotated, [2] cooling
vs distance with exponential fit) plus a short table comparing model values
to literature bands.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from austin_twin.calibration import load_calibrated_config
from austin_twin.cooling_response import run_full_analysis
from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.simulator import SimConfig
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CAL = OUT / "calibrated_config.json"

# Approximate literature reference bands.
# These are CENTRAL ranges synthesized across multiple UFOR studies — they are
# meant as orientation, not citation. A reviewer can swap them.
LIT_CITYWIDE_PER_10PCT = (0.10, 0.50)   # °C, citywide average per +10% canopy
LIT_LOCAL_PER_10PCT = (1.0, 3.0)        # °C, at the planting site
LIT_DECAY_LENGTH_M = (100.0, 300.0)     # m, half-cooling distance around patches


def main() -> None:
    print("[1/3] Austin grid + WorldCover land cover...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, cache_dir=RAW / "worldcover")

    base = SimConfig(duration_hours=48.0)
    cfg = load_calibrated_config(CAL, base_config=base) if CAL.exists() else base
    print(f"[2/3] running canopy sensitivity + patch decay (using "
          f"{'CALIBRATED' if CAL.exists() else 'default'} config)...")
    report = run_full_analysis(landuse, cfg)

    rc = report.response_curve
    dd = report.distance_decay

    # ---------- numerical summary ----------
    sweep_lines = ["  canopy +Δ  | citywide ΔT  | planted ΔT  | spillover ΔT"]
    sweep_lines.append("  " + "-" * 56)
    for d, cw, pl, sp in zip(rc.deltas, rc.citywide_mean_dt,
                              rc.planted_mean_dt, rc.spillover_mean_dt):
        sweep_lines.append(
            f"  {int(d*100):>4}%       | {cw:>+9.3f} °C | {pl:>+8.3f} °C | {sp:>+9.3f} °C"
        )

    half_str = (f"{dd.half_cooling_m:.0f} m  (λ = {dd.decay_length_m:.0f} m)"
                if np.isfinite(dd.half_cooling_m) else "n/a (decay too weak)")

    model_vs_lit_lines = [
        "Metric                                  | Model (calibrated) | Literature band (approx)",
        "-" * 95,
        f"Citywide cooling per +10% canopy        | {rc.citywide_slope_per_10pct:>+7.3f} °C        | "
        f"{LIT_CITYWIDE_PER_10PCT[0]:.2f} - {LIT_CITYWIDE_PER_10PCT[1]:.2f} °C",
        f"Local cooling per +10% canopy (planted) | {rc.planted_slope_per_10pct:>+7.3f} °C        | "
        f"{LIT_LOCAL_PER_10PCT[0]:.1f} - {LIT_LOCAL_PER_10PCT[1]:.1f} °C",
        f"Patch peak cooling (1 km canopy patch)  | {dd.peak_dt_c:>7.2f} °C        | (study-dependent)",
        f"Patch half-cooling distance             | {half_str:<18} | "
        f"{LIT_DECAY_LENGTH_M[0]:.0f} - {LIT_DECAY_LENGTH_M[1]:.0f} m",
    ]

    table = (
        "Canopy sweep (citywide canopy increment Δ):\n"
        + "\n".join(sweep_lines)
        + "\n\nModel vs literature reference bands:\n"
        + "\n".join(model_vs_lit_lines)
    )
    print()
    print(table)
    (OUT / "cooling_response.txt").write_text(table + "\n")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Panel 1: ΔT vs Δcanopy
    ax = axes[0]
    deltas_pct = [d * 100 for d in rc.deltas]
    ax.plot(deltas_pct, rc.citywide_mean_dt, "o-", color="C0", label="citywide mean ΔT")
    ax.plot(deltas_pct, rc.planted_mean_dt, "s-", color="C3", label="planted-cell mean ΔT")
    ax.plot(deltas_pct, rc.spillover_mean_dt, "^-", color="C2",
            label="spillover (Δveg ≈ 0)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Δ canopy fraction applied citywide (%)")
    ax.set_ylabel("ΔT vs baseline at peak heat (°C)")
    ax.set_title(
        f"Canopy sensitivity sweep\n"
        f"slope citywide = {rc.citywide_slope_per_10pct:+.3f} °C / +10% canopy   |   "
        f"slope local = {rc.planted_slope_per_10pct:+.3f} °C / +10% canopy"
    )
    # Shade literature reference band for citywide slope, projected to total ΔT
    # at each delta — the band is on the SLOPE, so the band shown is
    # delta × literature_slope.
    deltas_arr = np.array(deltas_pct)
    band_lo = -LIT_CITYWIDE_PER_10PCT[0] * deltas_arr / 10.0
    band_hi = -LIT_CITYWIDE_PER_10PCT[1] * deltas_arr / 10.0
    ax.fill_between(deltas_pct, band_lo, band_hi, alpha=0.12, color="C0",
                    label=f"lit. band, citywide ({LIT_CITYWIDE_PER_10PCT[0]:.2f}-{LIT_CITYWIDE_PER_10PCT[1]:.2f} °C/+10%)")
    band_lo_local = -LIT_LOCAL_PER_10PCT[0] * deltas_arr / 10.0
    band_hi_local = -LIT_LOCAL_PER_10PCT[1] * deltas_arr / 10.0
    ax.fill_between(deltas_pct, band_lo_local, band_hi_local, alpha=0.12, color="C3",
                    label=f"lit. band, local ({LIT_LOCAL_PER_10PCT[0]:.1f}-{LIT_LOCAL_PER_10PCT[1]:.1f} °C/+10%)")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)

    # Panel 2: ΔT vs distance from canopy patch
    ax = axes[1]
    mag = -dd.dt_mean  # cooling magnitude (positive)
    mag_p5 = -dd.dt_p95  # cooling magnitude lower
    mag_p95 = -dd.dt_p5  # cooling magnitude upper
    valid = np.isfinite(mag)
    ax.fill_between(dd.distances_m[valid] / 1000.0, mag_p5[valid], mag_p95[valid],
                    alpha=0.15, color="C1", label="5-95 % range per bin")
    ax.plot(dd.distances_m[valid] / 1000.0, mag[valid], "o-", color="C1",
            label="bin mean cooling")
    # Exponential fit overlay
    if np.isfinite(dd.decay_length_m):
        d_smooth = np.linspace(0, dd.distances_m[valid].max(), 200)
        fit_y = dd.peak_dt_c * np.exp(-d_smooth / dd.decay_length_m)
        ax.plot(d_smooth / 1000.0, fit_y, "k--", lw=1.2,
                label=f"exp fit, λ = {dd.decay_length_m:.0f} m")
        if np.isfinite(dd.half_cooling_m):
            ax.axvline(dd.half_cooling_m / 1000.0, color="k", lw=0.8, ls=":")
            ax.text(dd.half_cooling_m / 1000.0 + 0.05, ax.get_ylim()[1] * 0.6,
                    f"half-cooling\n{dd.half_cooling_m:.0f} m",
                    fontsize=9, va="top")
    # Literature half-cooling band as vertical shade
    ax.axvspan(LIT_DECAY_LENGTH_M[0] / 1000.0, LIT_DECAY_LENGTH_M[1] / 1000.0,
               alpha=0.10, color="green",
               label=f"lit. half-cooling band ({LIT_DECAY_LENGTH_M[0]:.0f}-{LIT_DECAY_LENGTH_M[1]:.0f} m)")
    ax.set_xlabel("Distance from canopy patch center (km)")
    ax.set_ylabel("Cooling magnitude |ΔT| (°C)")
    ax.set_title(
        f"Patch distance decay\n1 km canopy patch in central Austin, calibrated D = "
        f"{cfg.diffusion_m2_s:.1f} m²/s"
    )
    ax.set_xlim(0, 6)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle("Counterfactual cooling response in the form UFOR literature reports", fontsize=12)
    out_plot = OUT / "cooling_response.png"
    fig.savefig(out_plot, dpi=130)
    plt.close(fig)

    print(f"\n[3/3] done. wrote {OUT / 'cooling_response.txt'} and {out_plot}.")


if __name__ == "__main__":
    main()
