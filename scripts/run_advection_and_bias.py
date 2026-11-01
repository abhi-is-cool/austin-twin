"""Two smaller diagnostics: wind advection, and whether the ~7 C bias matters.

ADVECTION (paper_v2 Section V proposes replacing Fickian D with
advection-diffusion driven by ERA5 wind). Two things to establish:
  * the timestep cost, since the advective CFL |u| dt/dx <= 1 binds far
    earlier than the diffusive one at realistic wind speeds, and
  * whether it actually improves held-out r.
Each variant is recalibrated so the comparison is not confounded by
coefficients fitted for a different transport operator.

ABSOLUTE BIAS. paper_v2 lists a ~7 C mean bias under "honest limitations"
and the results are anomaly-based, which invites the assumption that the
bias simply cancels. It does not: the ET term is
    k_ET * f_veg * max(T - T_air, 0)
which is nonlinear with a clamp, so an offset moves the model onto a
different part of that term. Sweeping the longwave coefficient changes the
absolute bias; if r moves with it, the bias is coupled to skill and is not
a purely cosmetic limitation.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from austin_twin.arco_era5 import forcing_for_modis_date
from austin_twin.calibration import calibrate, evaluate, load_calibrated_config
from austin_twin.cities import AUSTIN
from austin_twin.grid import build_grid, fetch_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig, run
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

TRAIN = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST = ["2024-08-21", "2024-08-22"]


def _peak(res):
    T, t = res.temperature, res.times_hours
    means = np.array([np.nanmean(f) if np.isfinite(f).any() else -np.inf for f in T])
    return T[int(np.argmax(np.where(t >= 24.0, means, -np.inf)))]


def main() -> None:
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW, tile=AUSTIN.worldcover_tile)
    modis_tr = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TRAIN}
    modis_te = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TEST}
    forcings = {d: forcing_for_modis_date(AUSTIN, d, cache_dir=RAW / "arco_era5")
                for d in TRAIN + TEST}

    ship = load_calibrated_config(OUT / "calibrated_config_arco.json",
                                  base_config=SimConfig(duration_hours=48.0))
    dx = 500.0

    # ---------------- advective CFL cost ----------------
    print("=" * 76)
    print("1. ADVECTIVE CFL COST")
    print("=" * 76)
    umax = max(float(np.max(np.abs(f.u_wind_m_s) + np.abs(f.v_wind_m_s)))
               for f in forcings.values())
    dt_max = dx / umax
    print(f"  max (|u|+|v|) over the calibration window : {umax:.2f} m/s")
    print(f"  max stable dt at dx = 500 m               : {dt_max:.0f} s")
    print(f"  published dt                              : 600 s")
    print(f"  steps per 48 h at that dt                 : "
          f"{int(48 * 3600 / min(dt_max, 600.0)):,} (vs 288)")

    dt_adv = 120.0  # comfortably inside the limit above
    print(f"  using dt = {dt_adv:.0f} s for advection runs "
          f"({int(48 * 3600 / dt_adv):,} steps)\n")

    # ---------------- advection on/off, both recalibrated ----------------
    print("=" * 76)
    print("2. DOES ADVECTION IMPROVE HELD-OUT r?")
    print("=" * 76)
    rows = []
    variants = [
        ("no advection, dt=600", dict(use_advection=False, dt_seconds=600.0)),
        ("no advection, dt=120", dict(use_advection=False, dt_seconds=dt_adv)),
        ("advection 100%, dt=120", dict(use_advection=True, dt_seconds=dt_adv,
                                        advection_scale=1.0)),
        ("advection 50%,  dt=120", dict(use_advection=True, dt_seconds=dt_adv,
                                        advection_scale=0.5)),
        ("advection 25%,  dt=120", dict(use_advection=True, dt_seconds=dt_adv,
                                        advection_scale=0.25)),
    ]
    print(f"{'variant':<26} {'test r':>9} {'test RMSE':>10} {'D':>8} {'k_ET':>7}")
    print("-" * 66)
    for label, over in variants:
        cfg = replace(ship, **over)
        try:
            cal = calibrate(landuse, modis_tr, modis_te, base_config=cfg,
                            fixed_coefs={"diffusion_m2_s": float(ship.diffusion_m2_s)},
                            forcings=forcings, popsize=8, maxiter=12, seed=0,
                            verbose=False)
        except Exception as exc:
            print(f"{label:<26} FAILED {type(exc).__name__}: {str(exc)[:30]}")
            continue
        r = float(np.mean([m.pearson_r for m in cal.test_metrics_after]))
        rm = float(np.mean([m.rmse for m in cal.test_metrics_after]))
        print(f"{label:<26} {r:>+9.4f} {rm:>10.4f} "
              f"{cal.optimal_coefs['diffusion_m2_s']:>8.3f} "
              f"{cal.optimal_coefs['et_coeff']:>7.2f}")
        rows.append(dict(variant=label, test_r=r, test_rmse=rm, **over))

    # ---------------- does the absolute bias couple to skill? ----------------
    print()
    print("=" * 76)
    print("3. DOES THE ABSOLUTE BIAS COUPLE TO SKILL?")
    print("=" * 76)
    print(f"{'lw_coeff':>9} {'mean bias C':>12} {'test r':>9} {'test RMSE':>10}")
    print("-" * 44)
    bias_rows = []
    for lw in (4.0, 6.0, 8.0, 12.0, 16.0, 24.0):
        cfg = replace(ship, lw_coeff=float(lw))
        mets = evaluate(landuse, cfg, modis_te, forcings=forcings)
        r = float(np.mean([m.pearson_r for m in mets]))
        rm = float(np.mean([m.rmse for m in mets]))
        biases = []
        for d in TEST:
            sim = _peak(run(landuse, cfg, forcing=forcings[d]))
            obs = modis_te[d]
            v = np.isfinite(sim) & np.isfinite(obs)
            biases.append(float(np.mean(sim[v]) - np.mean(obs[v])))
        bias = float(np.mean(biases))
        print(f"{lw:>9.1f} {bias:>12.2f} {r:>+9.4f} {rm:>10.4f}")
        bias_rows.append(dict(lw_coeff=lw, mean_bias_c=bias, test_r=r, test_rmse=rm))

    rs = [b["test_r"] for b in bias_rows]
    biases = [b["mean_bias_c"] for b in bias_rows]
    print()
    print(f"  r range across the sweep : {min(rs):+.4f} .. {max(rs):+.4f} "
          f"(spread {max(rs) - min(rs):.4f})")
    print(f"  bias range               : {min(biases):+.2f} .. {max(biases):+.2f} C")
    if len(set(np.round(rs, 4))) > 1:
        print("  -> r DOES move with the absolute bias: the two are coupled,")
        print("     so the bias is not purely cosmetic even in anomaly space.")
    else:
        print("  -> r is flat: anomaly normalization does absorb the bias here.")

    (OUT / "advection_and_bias.json").write_text(json.dumps(
        {"max_wind_sum_m_s": umax, "dt_max_s": dt_max,
         "advection": rows, "bias_sweep": bias_rows}, indent=2))
    print(f"\nwrote outputs/advection_and_bias.json")


if __name__ == "__main__":
    main()
