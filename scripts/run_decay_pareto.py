"""The MODIS-fidelity vs intervention-realism trade-off, as an actual curve.

Contribution #2 claims the two-stage calibration "quantifies the trade-off
between MODIS pattern fidelity and intervention spatial realism", but the
paper evaluates it at two points (D = 1.0 and D = 18.3). This sweeps D across
three decades, including below the published search floor of 1.0, and for
each D:

  * re-fits the four free coefficients against MODIS with that D pinned
    (Stage 2, exactly as the ship procedure does), and
  * measures held-out test r and the patch half-cooling distance.

That pair is the Pareto front the contribution promises.

Both half-cooling estimators are reported side by side, because the legacy
one is biased by its own fit support (see `patch_distance_decay`) and the
published 366 m comes from it.

Stage 2 dominates the cost: one DE run per D value. `--quick` reduces the DE
budget for a first look.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from austin_twin.arco_era5 import forcing_for_modis_date
from austin_twin.calibration import calibrate, load_calibrated_config
from austin_twin.cities import AUSTIN
from austin_twin.cooling_response import patch_distance_decay
from austin_twin.grid import build_grid, fetch_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

TRAIN_DATES = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST_DATES = ["2024-08-21", "2024-08-22"]
LIT_BAND = (100.0, 300.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smaller DE budget and fewer D values")
    args = ap.parse_args()

    popsize, maxiter = (4, 5) if args.quick else (8, 12)
    n_D = 8 if args.quick else 14
    # Span below the published floor (1.0) up to the single-stage optimum.
    Ds = list(np.geomspace(0.05, 20.0, n_D))

    print("[setup] Austin grid + WorldCover + MODIS + ARCO-ERA5...")
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=AUSTIN.worldcover_tile)
    modis_train = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TRAIN_DATES}
    modis_test = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TEST_DATES}
    forcings = {d: forcing_for_modis_date(AUSTIN, d, cache_dir=RAW / "arco_era5")
                for d in TRAIN_DATES + TEST_DATES}

    base = replace(
        load_calibrated_config(OUT / "calibrated_config_two_stage.json",
                               base_config=SimConfig(duration_hours=48.0)),
        use_pm_et=False,
    )

    print(f"[sweep] {len(Ds)} D values, DE popsize={popsize} maxiter={maxiter}\n")
    print(f"{'D':>8} {'test r':>9} {'test RMSE':>10} "
          f"{'edge half':>10} {'ctr half':>10} {'legacy':>9} {'peak dT':>9} "
          f"{'k_ET':>7} {'in band':>8}")
    print("-" * 94)

    rows = []
    for D in Ds:
        cal = calibrate(
            landuse, modis_train, modis_test,
            base_config=base,
            fixed_coefs={"diffusion_m2_s": float(D)},
            forcings=forcings,
            popsize=popsize, maxiter=maxiter, seed=0, verbose=False,
        )
        cfg = cal.optimal_config
        test_r = float(np.mean([m.pearson_r for m in cal.test_metrics_after]))
        test_rmse = float(np.mean([m.rmse for m in cal.test_metrics_after]))

        rob = patch_distance_decay(landuse, cfg, estimator="robust")
        leg = patch_distance_decay(landuse, cfg, estimator="legacy")
        edge = rob.half_cooling_from_edge_m
        # The literature band is edge-referenced, so that is what we compare.
        inband = np.isfinite(edge) and LIT_BAND[0] <= edge <= LIT_BAND[1]

        print(f"{D:>8.3f} {test_r:>+9.4f} {test_rmse:>10.4f} "
              f"{edge:>10.1f} {rob.half_cooling_m:>10.1f} "
              f"{leg.half_cooling_m:>9.1f} "
              f"{rob.peak_dt_c:>9.2f} {cal.optimal_coefs['et_coeff']:>7.2f} "
              f"{'YES' if inband else '':>8}")

        rows.append(dict(
            D=float(D), test_r=test_r, test_rmse=test_rmse,
            half_cooling_edge_m=float(edge),
            half_cooling_robust_m=float(rob.half_cooling_m),
            half_cooling_legacy_m=float(leg.half_cooling_m),
            decay_length_robust_m=float(rob.decay_length_m),
            peak_dt_c=float(rob.peak_dt_c),
            n_bins_robust=int(rob.n_bins_fit), n_bins_legacy=int(leg.n_bins_fit),
            coefs={k: float(v) for k, v in cal.optimal_coefs.items()},
        ))

    # ---- Pareto analysis ----
    ok = [r for r in rows if np.isfinite(r["half_cooling_edge_m"])]
    inband = [r for r in ok
              if LIT_BAND[0] <= r["half_cooling_edge_m"] <= LIT_BAND[1]]

    print()
    print("=" * 86)
    print("PARETO SUMMARY (robust estimator)")
    print("=" * 86)
    best_r = max(ok, key=lambda r: r["test_r"])
    print(f"  best test r overall      : r = {best_r['test_r']:+.4f} at "
          f"D = {best_r['D']:.3f}  (edge half-cooling "
          f"{best_r['half_cooling_edge_m']:.0f} m)")
    if inband:
        best_in = max(inband, key=lambda r: r["test_r"])
        print(f"  best test r inside band  : r = {best_in['test_r']:+.4f} at "
              f"D = {best_in['D']:.3f}  (edge half-cooling "
              f"{best_in['half_cooling_edge_m']:.0f} m)")
        print(f"  cost of staying in band  : "
              f"{best_in['test_r'] - best_r['test_r']:+.4f} r")
    else:
        print("  no sweep point lands inside the 100-300 m literature band")

    ship = min(ok, key=lambda r: abs(r["D"] - 1.0))
    print(f"  published ship D = 1.0   : r = {ship['test_r']:+.4f}, "
          f"edge half-cooling {ship['half_cooling_edge_m']:.0f} m, "
          f"centre {ship['half_cooling_robust_m']:.0f} m, "
          f"legacy {ship['half_cooling_legacy_m']:.0f} m")

    # Monotonicity of each estimator across the sweep.
    for tag in ("edge", "robust", "legacy"):
        vals = [r[f"half_cooling_{tag}_m"] for r in rows
                if np.isfinite(r[f"half_cooling_{tag}_m"])]
        drops = int((np.diff(vals) < 0).sum())
        print(f"  {tag:<6} estimator monotonic in D: "
              f"{'yes' if drops == 0 else f'NO ({drops} reversals)'}")

    (OUT / "decay_pareto.json").write_text(json.dumps({
        "lit_band_m": LIT_BAND, "popsize": popsize, "maxiter": maxiter,
        "rows": rows,
    }, indent=2))
    print(f"\nwrote outputs/decay_pareto.json")


if __name__ == "__main__":
    main()
