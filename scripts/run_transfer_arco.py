"""Re-run the transfer matrix under the ARCO ship coefficients and forcing.

Audit finding (1): the paper reports the ARCO-recalibrated Austin
coefficients in Section II-C and an ARCO-derived ceiling row in the transfer
table, but the transfer *scores* in that table came from
`outputs/transfer_*.json`, whose stored `austin_coefs` are the CDS values.
The two differ most in alpha_veg (0.4343 -> 0.4641, about 7 %) and k_ET
(10.975 -> 8.677) -- and vegetation absorption drives exactly the
vegetation/impervious contrast transfer depends on, so the substitution is
not obviously harmless.

This repeats `run_transfer.py`'s protocol exactly -- same train/test splits,
same five rows, same DE budget (popsize 4, maxiter 6) -- changing only the
two things the audit flagged: the Austin config comes from
`calibrated_config_arco.json`, and the per-date forcing comes from
ARCO-ERA5 rather than a CDS NetCDF (which is also what makes this runnable
at all; the CDS files were never on this machine).
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_transfer import (  # noqa: E402
    CITY_SPLITS, _run_scheme_a, _run_scheme_b, _run_zero_shot,
)

from austin_twin.arco_era5 import forcing_for_modis_date  # noqa: E402
from austin_twin.calibration import load_calibrated_config  # noqa: E402
from austin_twin.cities import CITIES  # noqa: E402
from austin_twin.grid import build_grid, fetch_boundary  # noqa: E402
from austin_twin.modis import fetch_aqua_lst  # noqa: E402
from austin_twin.simulator import SimConfig  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

# Published (CDS-coefficient) values, from paper_v3 (3) Table IV.
PUBLISHED = {
    "denver":  {"zero_shot": 0.004, "scheme_B_1day": 0.003, "scheme_B_3day": 0.003,
                "scheme_A_1day": 0.001, "scheme_A_3day": 0.001},
    "phoenix": {"zero_shot": 0.083, "scheme_B_1day": 0.075, "scheme_B_3day": 0.074,
                "scheme_A_1day": 0.039, "scheme_A_3day": 0.036},
    "miami":   {"zero_shot": 0.760, "scheme_B_1day": 0.787, "scheme_B_3day": 0.787,
                "scheme_A_1day": 0.832, "scheme_A_3day": 0.832},
}


def run_city(name: str, austin_cfg) -> dict:
    city = CITIES[name]
    split = CITY_SPLITS[name]
    train_dates, test_dates = split["train"], split["test"]

    print(f"\n=== {name.upper()}  train {train_dates}  test {test_dates} ===")
    boundary = fetch_boundary(city, cache_path=RAW / f"{name}_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=city.worldcover_tile)
    modis_train = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in train_dates}
    modis_test = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in test_dates}
    forcings = {d: forcing_for_modis_date(city, d, cache_dir=RAW / "arco_era5")
                for d in train_dates + test_dates}

    sub1 = {train_dates[0]: modis_train[train_dates[0]]}
    rows = [
        _run_zero_shot(landuse, austin_cfg, modis_test, forcings),
        _run_scheme_b(landuse, austin_cfg, sub1, modis_test, forcings, "scheme_B_1day"),
        _run_scheme_b(landuse, austin_cfg, modis_train, modis_test, forcings, "scheme_B_3day"),
        _run_scheme_a(landuse, austin_cfg, sub1, modis_test, forcings, "scheme_A_1day"),
        _run_scheme_a(landuse, austin_cfg, modis_train, modis_test, forcings, "scheme_A_3day"),
    ]

    pub = PUBLISHED[name]
    print(f"  {'row':<16} {'ARCO r':>9} {'published':>11} {'delta':>9}")
    print("  " + "-" * 48)
    for r in rows:
        p = pub.get(r["row"])
        d = r["test_r_mean"] - p if p is not None else float("nan")
        flag = "  <-- moves" if abs(d) >= 0.01 else ""
        print(f"  {r['row']:<16} {r['test_r_mean']:>+9.3f} {p:>+11.3f} "
              f"{d:>+9.3f}{flag}")
    return dict(city=name, train_dates=train_dates, test_dates=test_dates,
                austin_coefs={k: float(getattr(austin_cfg, k)) for k in
                              ("absorption_impervious", "absorption_vegetation",
                               "absorption_water", "et_coeff", "diffusion_m2_s")},
                rows=rows)


def main() -> None:
    austin_cfg = replace(
        load_calibrated_config(OUT / "calibrated_config_arco.json",
                               base_config=SimConfig(duration_hours=48.0)),
        use_pm_et=False)
    print("Austin ship coefficients (ARCO):")
    for k in ("absorption_impervious", "absorption_vegetation",
              "absorption_water", "et_coeff", "diffusion_m2_s"):
        print(f"  {k:<26} {getattr(austin_cfg, k):.4f}")

    out = {}
    for name in ("denver", "phoenix", "miami"):
        try:
            out[name] = run_city(name, austin_cfg)
        except Exception as exc:
            print(f"  {name}: FAILED {type(exc).__name__}: {str(exc)[:90]}")

    print()
    print("=" * 72)
    print("TRANSFER MATRIX UNDER ARCO COEFFICIENTS")
    print("=" * 72)
    print(f"{'Scheme':<24} {'Denver':>10} {'Phoenix':>10} {'Miami':>10}")
    print("-" * 72)
    labels = [("zero_shot", "Zero-shot"),
              ("scheme_B_1day", "Scheme B, 1 day"),
              ("scheme_B_3day", "Scheme B, 3 days"),
              ("scheme_A_1day", "Scheme A, 1 day"),
              ("scheme_A_3day", "Scheme A, 3 days")]
    for key, label in labels:
        cells = []
        for c in ("denver", "phoenix", "miami"):
            v = next((r["test_r_mean"] for r in out.get(c, {}).get("rows", [])
                      if r["row"] == key), None)
            cells.append(f"{v:+.3f}" if v is not None else "  ---  ")
        print(f"{label:<24} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10}")

    print()
    biggest = 0.0
    for c in out:
        for r in out[c]["rows"]:
            p = PUBLISHED[c].get(r["row"])
            if p is not None:
                biggest = max(biggest, abs(r["test_r_mean"] - p))
    print(f"largest |delta| vs published: {biggest:.4f}")
    print("  -> " + ("conclusions unchanged; differences are below rounding "
                     "at 3 dp" if biggest < 0.005 else
                     "at least one cell moves materially; the table needs "
                     "updating"))

    (OUT / "transfer_arco.json").write_text(json.dumps(out, indent=2))
    print("\nwrote outputs/transfer_arco.json")


if __name__ == "__main__":
    main()
