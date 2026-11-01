"""Is k_ET identified in PM mode, and does r_s explain the missing 77%?

Three things, in increasing cost.

1. VERIFY k_ET IS INERT. `simulator.run` references `cfg.et_coeff` only in
   the non-PM branch: with `use_pm_et=True` the latent flux comes entirely
   from `latent_heat_flux_w_m2`. So k_ET should have *no* effect on a PM run.
   If that holds, the earlier caveat -- "k_ET pins at its optimizer bound, so
   the PM variants are not cleanly identified" -- was the wrong diagnosis.
   The values it produced (7.8, 19.9, 40.7, 40.9, 44.2, 47.0, 54.3, 54.9)
   were scattered across the range rather than piled at 60, which is what an
   unidentified parameter looks like, not a constrained one.

2. RE-RUN THE PM SWEEP WITHOUT k_ET. Even though k_ET does nothing in PM
   mode, differential evolution was still searching over it, so the PM
   variants received roughly three quarters of the effective budget that
   linear ET did on the three absorptions that actually matter. That is a
   real confound in the 23%/77% split -- not boundary-pinning, but wasted
   optimiser effort. Refitting on the three live parameters is the fair
   comparison and costs no more.

3. CALIBRATE r_s. The stomatal resistance `r_s_leaf` has been fixed at the
   FAO-56 reference-grass value of 100 s/m in every result so far and was
   never calibrated, which makes it the other prime suspect for the part of
   the regression neither named mechanism explains. Swept with the three
   absorptions refit at each value.
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
from austin_twin.simulator import SimConfig
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

TRAIN = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST = ["2024-08-21", "2024-08-22"]

# Only the three absorptions are live under PM; k_ET is not referenced.
PM_FREE = {"diffusion_m2_s", "et_coeff"}          # -> pinned, so not searched
LINEAR_FREE = {"diffusion_m2_s"}                  # k_ET stays live for linear ET

VARIANTS = [
    ("linear ET (reference)", dict(use_pm_et=False), LINEAR_FREE),
    ("PM air VPD, k=3 (canonical)",
     dict(use_pm_et=True, pm_vpd_mode="air", pm_lai_k=3.0), PM_FREE),
    ("PM air VPD, linear LAI",
     dict(use_pm_et=True, pm_vpd_mode="air", pm_lai_mode="linear"), PM_FREE),
    ("PM surface VPD, k=3",
     dict(use_pm_et=True, pm_vpd_mode="surface", pm_lai_k=3.0), PM_FREE),
    ("PM surface VPD, linear LAI",
     dict(use_pm_et=True, pm_vpd_mode="surface", pm_lai_mode="linear"), PM_FREE),
    ("big-leaf bulk, k=3",
     dict(use_pm_et=True, pm_vpd_mode="bulk", pm_lai_k=3.0,
          pm_max_evaporative_fraction=1.5), PM_FREE),
]

R_S_GRID = [25.0, 50.0, 100.0, 200.0, 400.0, 800.0]


def load_inputs():
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=AUSTIN.worldcover_tile)
    modis_tr = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TRAIN}
    modis_te = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in TEST}
    forcings = {d: forcing_for_modis_date(AUSTIN, d, cache_dir=RAW / "arco_era5")
                for d in TRAIN + TEST}
    return landuse, modis_tr, modis_te, forcings


def score(landuse, cfg, modis, forcings):
    m = evaluate(landuse, cfg, modis, forcings=forcings)
    return (float(np.mean([x.pearson_r for x in m])),
            float(np.mean([x.rmse for x in m])))


def main() -> None:
    landuse, modis_tr, modis_te, forcings = load_inputs()
    ship = load_calibrated_config(OUT / "calibrated_config_arco.json",
                                  base_config=SimConfig(duration_hours=48.0))
    D = float(ship.diffusion_m2_s)
    results = {}

    # ---------- 1. is k_ET inert under PM? ----------
    print("=" * 78)
    print("1. IS k_ET INERT IN PM MODE?")
    print("=" * 78)
    base_pm = replace(ship, use_pm_et=True, pm_vpd_mode="surface", pm_lai_k=3.0)
    r_lo, _ = score(landuse, replace(base_pm, et_coeff=5.0), modis_te, forcings)
    r_hi, _ = score(landuse, replace(base_pm, et_coeff=60.0), modis_te, forcings)
    print(f"  PM surface VPD, k_ET =  5.0 : test r = {r_lo:+.6f}")
    print(f"  PM surface VPD, k_ET = 60.0 : test r = {r_hi:+.6f}")
    inert = abs(r_hi - r_lo) < 1e-9
    print(f"  difference = {r_hi - r_lo:+.3e}  ->  "
          f"{'k_ET IS INERT under PM' if inert else 'k_ET has an effect'}")

    # Contrast: the same test under linear ET, where k_ET must matter.
    base_lin = replace(ship, use_pm_et=False)
    l_lo, _ = score(landuse, replace(base_lin, et_coeff=5.0), modis_te, forcings)
    l_hi, _ = score(landuse, replace(base_lin, et_coeff=60.0), modis_te, forcings)
    print(f"  (linear ET, same swing     : {l_lo:+.4f} -> {l_hi:+.4f}, "
          f"delta {l_hi - l_lo:+.4f})")
    results["k_et_inert_under_pm"] = bool(inert)

    # ---------- 2. PM sweep with the dead dimension removed ----------
    print()
    print("=" * 78)
    print("2. PM VARIANTS, REFIT ON LIVE PARAMETERS ONLY")
    print("=" * 78)
    print(f"{'variant':<30} {'test r':>9} {'RMSE':>8} {'vs linear':>10} {'free':>5}")
    print("-" * 78)
    rows, ref = [], None
    for label, over, pinned in VARIANTS:
        cfg = replace(ship, **over)
        fixed = {k: float(getattr(cfg, k)) for k in pinned}
        fixed["diffusion_m2_s"] = D
        try:
            cal = calibrate(landuse, modis_tr, modis_te, base_config=cfg,
                            fixed_coefs=fixed, forcings=forcings,
                            popsize=8, maxiter=12, seed=0, verbose=False)
        except Exception as exc:
            print(f"{label:<30} FAILED {type(exc).__name__}")
            continue
        r = float(np.mean([m.pearson_r for m in cal.test_metrics_after]))
        rm = float(np.mean([m.rmse for m in cal.test_metrics_after]))
        if ref is None:
            ref = r
        n_free = 5 - len(fixed)
        print(f"{label:<30} {r:>+9.4f} {rm:>8.4f} {r - ref:>+10.4f} {n_free:>5}")
        rows.append(dict(variant=label, test_r=r, test_rmse=rm,
                         delta_vs_linear=r - ref, n_free=n_free,
                         coefs={k: float(v) for k, v in cal.optimal_coefs.items()}))
    results["variants"] = rows

    by = {r["variant"]: r["test_r"] for r in rows}
    lin, canon = by.get("linear ET (reference)"), by.get("PM air VPD, k=3 (canonical)")
    if lin is not None and canon is not None:
        total = canon - lin
        print()
        print(f"  total PM regression: {total:+.4f}")
        for name, key in (("LAI fix alone", "PM air VPD, linear LAI"),
                          ("VPD fix alone", "PM surface VPD, k=3"),
                          ("both fixes", "PM surface VPD, linear LAI"),
                          ("big-leaf bulk", "big-leaf bulk, k=3")):
            v = by.get(key)
            if v is not None:
                print(f"    {name:<16} {v - canon:+.4f}  "
                      f"({(v - canon) / abs(total):.0%} of the regression)")
        results["regression_total"] = total

    # ---------- 3. does r_s explain the residual? ----------
    print()
    print("=" * 78)
    print("3. STOMATAL RESISTANCE SWEEP (absorptions refit at each value)")
    print("=" * 78)
    print(f"{'r_s_leaf':>9} {'test r':>9} {'RMSE':>8} {'vs canonical':>13}")
    print("-" * 78)
    rs_rows = []
    for rs in R_S_GRID:
        cfg = replace(ship, use_pm_et=True, pm_vpd_mode="air",
                      pm_lai_k=3.0, pm_r_s_leaf=float(rs))
        fixed = {"diffusion_m2_s": D, "et_coeff": float(cfg.et_coeff)}
        cal = calibrate(landuse, modis_tr, modis_te, base_config=cfg,
                        fixed_coefs=fixed, forcings=forcings,
                        popsize=8, maxiter=12, seed=0, verbose=False)
        r = float(np.mean([m.pearson_r for m in cal.test_metrics_after]))
        rm = float(np.mean([m.rmse for m in cal.test_metrics_after]))
        mark = "  <- FAO-56 default" if rs == 100.0 else ""
        print(f"{rs:>9.0f} {r:>+9.4f} {rm:>8.4f} "
              f"{r - (canon or 0):>+13.4f}{mark}")
        rs_rows.append(dict(r_s_leaf=rs, test_r=r, test_rmse=rm))
    results["r_s_sweep"] = rs_rows

    best = max(rs_rows, key=lambda x: x["test_r"])
    print()
    print(f"  best r_s_leaf = {best['r_s_leaf']:.0f} s/m -> r = {best['test_r']:+.4f}")
    if canon is not None and lin is not None:
        rec = best["test_r"] - canon
        print(f"  recovers {rec:+.4f} of the {canon - lin:+.4f} regression "
              f"({rec / abs(canon - lin):.0%})")

    (OUT / "pm_identification.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote outputs/pm_identification.json")


if __name__ == "__main__":
    main()
