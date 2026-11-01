"""Multi-city transfer: how portable is the Austin-calibrated digital twin?

Runs a matrix of transfer experiments per target city:

  1. Zero-shot: Austin coefs applied unchanged to target-city land cover +
     ERA5 forcing, scored against target-city MODIS.
  2. Scheme B (physics-portable, 1 day): freeze Austin absorptions, tune
     `et_coeff` only against 1 target-city training day.
  3. Scheme B (physics-portable, 3 days): same freeze, tune against 3
     target-city training days.
  4. Scheme A (full re-DE, 1 day): tune all 4 coefs (3 absorptions +
     et_coeff) against 1 target-city training day.
  5. Scheme A (full re-DE, 3 days): same, against 3 target-city days.

Held-out evaluation set is the same 2 target-city days for every row.

Design intent:
  - Scheme B tests a physics-portability hypothesis (absorptions are
    material properties, `et_coeff` is climate).
  - Scheme A gives the data-efficiency curve reviewers expect.

Note on DE budget: `popsize=4, maxiter=6` is reduced from the Austin
calibration's (8, 12) to keep multi-city runs tractable at Miami-Dade's
187x168 grid. Convergence tests on Austin's original setup show <0.02
r degradation at this smaller budget.

Output: outputs/transfer_{city}.json with per-day metrics for each row.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from austin_twin.calibration import calibrate, evaluate, load_calibrated_config
from austin_twin.cities import CITIES, AUSTIN, CityConfig
from austin_twin.forcing import Forcing
from austin_twin.grid import fetch_boundary, build_grid
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
AUSTIN_CAL = OUT / "calibrated_config_era5.json"


# Per-city MODIS date splits (3 train + 2 test). Chosen from a clear-sky scan.
CITY_SPLITS: dict[str, dict[str, list[str] | Path]] = {
    "phoenix": {
        "train": ["2024-07-15", "2024-07-17", "2024-07-20"],
        "test":  ["2024-07-22", "2024-07-24"],
        "era5":  RAW / "era5" / "era5_phoenix_2024-07-15_2024-07-28.nc",
    },
    "denver": {
        "train": ["2024-07-20", "2024-07-22", "2024-07-23"],
        "test":  ["2024-07-25", "2024-07-28"],
        "era5":  RAW / "era5" / "era5_denver_2024-07-15_2024-07-28.nc",
    },
    "miami": {
        "train": ["2024-04-05", "2024-04-12", "2024-04-14"],
        "test":  ["2024-04-16", "2024-04-19"],
        "era5":  RAW / "era5" / "era5_miami_2024-04-04_2024-04-20.nc",
    },
}


def _forcing_for(city: CityConfig, date_iso: str, era5_nc: Path) -> Forcing:
    """48h forcing starting at midnight local of the day BEFORE `date_iso`."""
    prev = date.fromisoformat(date_iso) - timedelta(days=1)
    return Forcing.from_era5(
        era5_nc,
        start_iso=f"{prev.isoformat()}T{city.utc_offset_hours:02d}:00",
        duration_hours=48.0,
    )


def _austin_base_config() -> SimConfig:
    base = SimConfig(duration_hours=48.0)
    cfg = load_calibrated_config(AUSTIN_CAL, base_config=base)
    return replace(cfg, use_pm_et=False)


def _row_report(row_name: str, train_dates: list[str], test_metrics, extra: dict | None = None) -> dict:
    r_mean = float(np.mean([m.pearson_r for m in test_metrics]))
    rmse_mean = float(np.mean([m.rmse for m in test_metrics]))
    per_day = [
        {"date": m.date, "r": float(m.pearson_r), "rmse": float(m.rmse),
         "sim_spread": float(m.sim_spread), "obs_spread": float(m.obs_spread)}
        for m in test_metrics
    ]
    r = {
        "row": row_name,
        "train_dates": train_dates,
        "test_r_mean": r_mean,
        "test_rmse_mean": rmse_mean,
        "test_per_day": per_day,
    }
    if extra:
        r.update(extra)
    return r


def _run_zero_shot(landuse, austin_cfg, modis_test, forcings) -> dict:
    """Row 1: Austin coefs unchanged, evaluate on target-city test set."""
    m = evaluate(landuse, austin_cfg, modis_test, forcings=forcings)
    return _row_report("zero_shot", train_dates=[], test_metrics=m, extra={
        "coefs": {k: float(getattr(austin_cfg, k))
                  for k in ("absorption_impervious", "absorption_vegetation",
                            "absorption_water", "et_coeff", "diffusion_m2_s")}
    })


def _run_scheme_b(landuse, austin_cfg, modis_train_subset, modis_test, forcings,
                  label: str) -> dict:
    """Freeze absorptions; DE on `et_coeff` alone against the training subset."""
    # Bounds for et_coeff alone (rest pinned).
    fixed = {name: float(getattr(austin_cfg, name)) for name in
             ("absorption_impervious", "absorption_vegetation",
              "absorption_water", "diffusion_m2_s")}
    cal = calibrate(
        landuse, modis_train_subset, modis_test,
        base_config=austin_cfg,
        fixed_coefs=fixed,
        forcings=forcings,
        popsize=4, maxiter=6, verbose=False,
    )
    return _row_report(label, list(modis_train_subset.keys()),
                       cal.test_metrics_after,
                       extra={
                           "et_coeff_fit": float(cal.optimal_coefs["et_coeff"]),
                           "coefs": cal.optimal_coefs,
                       })


def _run_scheme_a(landuse, austin_cfg, modis_train_subset, modis_test, forcings,
                  label: str) -> dict:
    """Full 4-coef DE on training subset (diffusion still pinned)."""
    fixed = {"diffusion_m2_s": float(austin_cfg.diffusion_m2_s)}
    cal = calibrate(
        landuse, modis_train_subset, modis_test,
        base_config=austin_cfg,
        fixed_coefs=fixed,
        forcings=forcings,
        popsize=4, maxiter=6, verbose=False,
    )
    return _row_report(label, list(modis_train_subset.keys()),
                       cal.test_metrics_after,
                       extra={"coefs": cal.optimal_coefs})


def run_transfer(city_name: str) -> dict:
    city = CITIES[city_name]
    split = CITY_SPLITS[city_name]
    era5_nc: Path = split["era5"]  # type: ignore
    train_dates: list[str] = split["train"]  # type: ignore
    test_dates: list[str] = split["test"]    # type: ignore

    print(f"\n{'='*70}")
    print(f"TRANSFER: Austin -> {city.name.upper()}")
    print(f"  train dates: {train_dates}")
    print(f"  test  dates: {test_dates}")
    print(f"  ERA5 file:   {era5_nc.name}")
    print(f"{'='*70}\n")

    print(f"[1/5] Building {city.name} grid + WorldCover land cover...")
    boundary = fetch_boundary(city, cache_path=RAW / f"{city.name}_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(
        boundary, grid, cache_dir=RAW / "worldcover",
        tile=city.worldcover_tile,
    )
    print(f"  grid {grid.shape}  city cells: {int(grid.mask.sum())}")

    print(f"[2/5] Loading MODIS + ERA5 per-date forcings...")
    modis_train = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in train_dates}
    modis_test  = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in test_dates}
    forcings = {d: _forcing_for(city, d, era5_nc) for d in train_dates + test_dates}
    for d, f in forcings.items():
        print(f"    {d}: T_air max={f.t_air_c.max():.1f} C, T_dew mean={f.t_dew_c.mean():.1f} C, "
              f"wind mean={f.wind_speed_m_s.mean():.1f} m/s")

    print(f"[3/5] Austin baseline config (from {AUSTIN_CAL.name})")
    austin_cfg = _austin_base_config()
    for name in ("absorption_impervious", "absorption_vegetation",
                 "absorption_water", "et_coeff", "diffusion_m2_s"):
        print(f"    {name:<26} = {getattr(austin_cfg, name):.4f}")

    print(f"\n[4/5] Running transfer matrix (5 rows, always evaluate on same 2 test days)...")
    rows: list[dict] = []

    print("  row 1: zero-shot (Austin coefs unchanged)...")
    rows.append(_run_zero_shot(landuse, austin_cfg, modis_test, forcings))

    print("  row 2: Scheme B (freeze absorptions, tune et_coeff), 1 target day...")
    sub1 = {train_dates[0]: modis_train[train_dates[0]]}
    rows.append(_run_scheme_b(landuse, austin_cfg, sub1, modis_test, forcings,
                              "scheme_B_1day"))

    print("  row 3: Scheme B, 3 target days...")
    rows.append(_run_scheme_b(landuse, austin_cfg, modis_train, modis_test, forcings,
                              "scheme_B_3day"))

    print("  row 4: Scheme A (full re-DE on 4 coefs), 1 target day...")
    rows.append(_run_scheme_a(landuse, austin_cfg, sub1, modis_test, forcings,
                              "scheme_A_1day"))

    print("  row 5: Scheme A, 3 target days...")
    rows.append(_run_scheme_a(landuse, austin_cfg, modis_train, modis_test, forcings,
                              "scheme_A_3day"))

    # Present a compact table.
    print(f"\n[5/5] {city.name.upper()} transfer results (test-set r / RMSE, 2 held-out days):")
    print(f"  {'row':<20} {'test r':>10} {'test RMSE':>12}   {'notes'}")
    print(f"  {'-'*70}")
    for r in rows:
        note = ""
        if "et_coeff_fit" in r:
            note = f"et_coeff={r['et_coeff_fit']:.2f} (vs Austin {austin_cfg.et_coeff:.2f})"
        elif "coefs" in r and r["row"] != "zero_shot":
            coefs = r["coefs"]
            note = (f"imp={coefs['absorption_impervious']:.3f} veg={coefs['absorption_vegetation']:.3f} "
                    f"et={coefs['et_coeff']:.2f}")
        print(f"  {r['row']:<20} {r['test_r_mean']:>+10.3f} {r['test_rmse_mean']:>11.3f} C   {note}")

    return {
        "city": city.name,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "austin_coefs": {name: float(getattr(austin_cfg, name)) for name in
                         ("absorption_impervious", "absorption_vegetation",
                          "absorption_water", "et_coeff", "diffusion_m2_s")},
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", choices=list(CITY_SPLITS), required=True,
                    help="Target city for the transfer study.")
    args = ap.parse_args()

    result = run_transfer(args.city)
    out_path = OUT / f"transfer_{args.city}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
