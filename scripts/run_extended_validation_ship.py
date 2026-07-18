"""Extended validation (18 days across 6 seasonal categories) on the ship config.

Companion to `run_extended_validation.py`. The original ran the single-stage
config against 18 MODIS days with synthetic diurnal forcing everywhere. This
one runs the ship config (linear-ET + per-date ERA5 forcing) — the honest
ship-config generalization test, since coefficients and forcing are jointly
calibrated in Phase 2.

Same date selection (loaded from `outputs/extended_validation_selection.json`
so the two runs score identical days). ERA5 pulled from the per-window
NetCDFs under `data/raw/era5/` (see `run_extended_validation_ship.py`
comment block for which windows are needed).

Regression baselines are re-fit here for completeness on the same Aug 11/16/19
training days as the original — they're independent of the simulator config
and give the same numbers as `run_extended_validation.py`.

Output: `outputs/extended_validation_ship.txt` + `.json` (per-day metrics and
per-category means).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from austin_twin import baseline_regression as breg
from austin_twin.calibration import evaluate as evaluate_simulator, load_calibrated_config
from austin_twin.cities import AUSTIN
from austin_twin.forcing import Forcing
from austin_twin.grid import fetch_boundary, build_grid
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CAL_SHIP = OUT / "calibrated_config_era5.json"
SEL_CACHE = OUT / "extended_validation_selection.json"
ERA5_DIR = RAW / "era5"

TRAIN_DATES = ["2024-08-11", "2024-08-16", "2024-08-19"]

# Which ERA5 NetCDF covers which validation date. Each window must include
# the day BEFORE the validation date (that's when the 48h sim starts, at
# midnight CDT/CST = UTC-5 or UTC-6). Any window that fully contains
# [date - 1 day, date + 1 day] works.
_ERA5_WINDOWS: list[tuple[str, str, Path]] = [
    ("2024-08-10", "2024-08-23", ERA5_DIR / "era5_2024-08-10_2024-08-23.nc"),
    ("2024-06-07", "2024-09-09", ERA5_DIR / "era5_austin_2024-06-07_2024-09-09.nc"),
    ("2024-04-02", "2024-04-23", ERA5_DIR / "era5_austin_2024-04-02_2024-04-23.nc"),
    ("2024-10-14", "2024-10-25", ERA5_DIR / "era5_austin_2024-10-14_2024-10-25.nc"),
    ("2023-12-26", "2024-01-30", ERA5_DIR / "era5_austin_2023-12-26_2024-01-30.nc"),
    ("2023-08-07", "2023-08-19", ERA5_DIR / "era5_austin_2023-08-07_2023-08-19.nc"),
]


def _find_era5_for(date_iso: str) -> Path:
    """Return the ERA5 NetCDF whose window contains date_iso and the prior day."""
    d = date.fromisoformat(date_iso)
    prev = d - timedelta(days=1)
    for win_start, win_end, path in _ERA5_WINDOWS:
        s = date.fromisoformat(win_start)
        e = date.fromisoformat(win_end)
        if s <= prev and d <= e:
            return path
    raise RuntimeError(f"No ERA5 window found for {date_iso} (need day-before + day-of).")


def _forcing_for_date(date_iso: str) -> Forcing:
    """48 h ERA5 forcing starting at midnight local of the day before date_iso."""
    prev = date.fromisoformat(date_iso) - timedelta(days=1)
    # UTC offset: Austin is CDT (-5) Mar-Nov, CST (-6) Nov-Mar. Use -5 for
    # spring/summer/fall selection days, -6 for winter validation days.
    month = date.fromisoformat(date_iso).month
    utc_offset = 6 if month in (12, 1, 2) else 5
    return Forcing.from_era5(
        _find_era5_for(date_iso),
        start_iso=f"{prev.isoformat()}T{utc_offset:02d}:00",
        duration_hours=48.0,
    )


@dataclass
class Pick:
    date: str
    coverage: float
    mean_lst_c: float


def main() -> None:
    print("[1/4] Austin grid + WorldCover land cover...")
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(
        boundary, grid, cache_dir=RAW / "worldcover", tile=AUSTIN.worldcover_tile,
    )

    print(f"[2/4] Loading pre-selected 18 validation days from {SEL_CACHE.name}...")
    if not SEL_CACHE.exists():
        raise SystemExit(f"missing {SEL_CACHE}; run scripts/run_extended_validation.py first "
                         "to generate the day selection")
    raw = json.loads(SEL_CACHE.read_text())
    selection: dict[str, list[Pick]] = {cat: [Pick(**d) for d in lst] for cat, lst in raw.items()}
    all_dates = [p.date for picks in selection.values() for p in picks]
    print(f"      {len(all_dates)} validation days across {len(selection)} categories")

    print("[3/4] Verifying all ERA5 windows exist + building per-date forcings...")
    forcings: dict[str, Forcing] = {}
    for d in all_dates:
        forcings[d] = _forcing_for_date(d)
        f = forcings[d]
        print(f"      {d}: T_air {f.t_air_c.min():.1f}-{f.t_air_c.max():.1f} °C, "
              f"T_dew mean {f.t_dew_c.mean():.1f} °C, wind mean {f.wind_speed_m_s.mean():.1f} m/s")

    print("[4/4] Ship config + MODIS + regression baselines...")
    modis_days = {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in all_dates}
    train_modis = {d: modis_days[d] if d in modis_days else fetch_aqua_lst(d, grid, RAW / "modis")
                   for d in TRAIN_DATES}
    reg_simple = breg.fit(landuse, train_modis, use_position=False)
    reg_pos = breg.fit(landuse, train_modis, use_position=True)

    base = SimConfig(duration_hours=48.0)
    cfg = replace(load_calibrated_config(CAL_SHIP, base_config=base), use_pm_et=False)
    print(f"      ship coefs: imp={cfg.absorption_impervious:.3f}  veg={cfg.absorption_vegetation:.3f}  "
          f"wat={cfg.absorption_water:.3f}  et={cfg.et_coeff:.3f}  D={cfg.diffusion_m2_s:.3f}")

    sim_metrics = evaluate_simulator(landuse, cfg, modis_days, forcings=forcings)
    sim_by_date = {m.date: m for m in sim_metrics}
    reg_simple_metrics = breg.evaluate(reg_simple, landuse, modis_days)
    reg_pos_metrics = breg.evaluate(reg_pos, landuse, modis_days)

    # --- Per-day and per-category tables (same layout as legacy) ---
    rows: list[str] = [
        "Per-day metrics on anomaly fields (SHIP CONFIG, linear-ET + per-date ERA5; "
        "regression trained on Aug 11/16/19 only)",
        "  category                              date         | mean LST | ship r/RMSE    | reg r/RMSE     | reg+pos r/RMSE",
        "  " + "-" * 130,
    ]
    grouped: dict[str, list[tuple[str, float, tuple[float, float], tuple[float, float], tuple[float, float]]]] = {}
    for cat, picks in selection.items():
        grouped[cat] = []
        for p in picks:
            sm = sim_by_date.get(p.date)
            if sm is None:
                continue
            rs = reg_simple_metrics[p.date]
            rp = reg_pos_metrics[p.date]
            sim_t = (sm.pearson_r, sm.rmse)
            grouped[cat].append((p.date, p.mean_lst_c, sim_t, rs, rp))
            rows.append(
                f"  {cat:<37} {p.date:<12} | {p.mean_lst_c:>+7.1f}°C "
                f"| {sm.pearson_r:+.3f} / {sm.rmse:.2f} "
                f"| {rs[1]:+.3f} / {rs[0]:.2f} "
                f"| {rp[1]:+.3f} / {rp[0]:.2f}"
            )

    rows.append("")
    rows.append("Mean metrics per category:")
    rows.append(
        f"  {'category':<37} {'n':>3} {'mean LST':>10} "
        f"{'ship r':>8} {'ship RMSE':>10} {'reg r':>8} {'reg RMSE':>10} "
        f"{'reg+pos r':>11} {'reg+pos RMSE':>13}"
    )
    rows.append("  " + "-" * 130)
    summary: dict[str, dict[str, float]] = {}
    for cat, items in grouped.items():
        if not items:
            continue
        mean_lst = float(np.mean([x[1] for x in items]))
        ship_r = float(np.mean([x[2][0] for x in items]))
        ship_rmse = float(np.mean([x[2][1] for x in items]))
        reg_r = float(np.mean([x[3][1] for x in items]))
        reg_rmse = float(np.mean([x[3][0] for x in items]))
        regp_r = float(np.mean([x[4][1] for x in items]))
        regp_rmse = float(np.mean([x[4][0] for x in items]))
        summary[cat] = dict(n=len(items), mean_lst=mean_lst,
                            ship_r=ship_r, ship_rmse=ship_rmse,
                            reg_r=reg_r, reg_rmse=reg_rmse,
                            reg_pos_r=regp_r, reg_pos_rmse=regp_rmse)
        rows.append(
            f"  {cat:<37} {len(items):>3} {mean_lst:>+9.1f}°C "
            f"{ship_r:>+8.3f} {ship_rmse:>10.2f} "
            f"{reg_r:>+8.3f} {reg_rmse:>10.2f} "
            f"{regp_r:>+11.3f} {regp_rmse:>13.2f}"
        )

    table = "\n".join(rows)
    print()
    print(table)
    (OUT / "extended_validation_ship.txt").write_text(table + "\n")

    # Also emit JSON for downstream table-fill scripts.
    json_out = {
        "config": "ship (linear-ET + per-date ERA5)",
        "summary": summary,
        "per_day": {p.date: {
            "category": cat,
            "mean_lst_c": p.mean_lst_c,
            "ship_r": sim_by_date[p.date].pearson_r,
            "ship_rmse": sim_by_date[p.date].rmse,
            "reg_r": reg_simple_metrics[p.date][1],
            "reg_rmse": reg_simple_metrics[p.date][0],
            "reg_pos_r": reg_pos_metrics[p.date][1],
            "reg_pos_rmse": reg_pos_metrics[p.date][0],
        } for cat, picks in selection.items() for p in picks if p.date in sim_by_date},
    }
    (OUT / "extended_validation_ship.json").write_text(json.dumps(json_out, indent=2, default=float))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cat_color = {c: color_cycle[i % len(color_cycle)] for i, c in enumerate(grouped)}

    for cat, items in grouped.items():
        if not items:
            continue
        lsts = [x[1] for x in items]
        rs_sim = [x[2][0] for x in items]
        rmses_sim = [x[2][1] for x in items]
        axes[0].scatter(lsts, rs_sim, color=cat_color[cat], label=cat, s=60, edgecolor="k", linewidth=0.5)
        axes[1].scatter(lsts, rmses_sim, color=cat_color[cat], label=cat, s=60, edgecolor="k", linewidth=0.5)

    axes[0].set_xlabel("Mean MODIS LST on validation day (°C)")
    axes[0].set_ylabel("Ship-config Pearson r")
    axes[0].set_title("Ship config: spatial-pattern fidelity by season / extremeness")
    axes[0].axhline(0.0, color="gray", lw=0.5)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].set_xlabel("Mean MODIS LST on validation day (°C)")
    axes[1].set_ylabel("Ship-config anomaly RMSE (°C)")
    axes[1].set_title("Ship config: anomaly RMSE by season / extremeness")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Extended validation: ship config (linear-ET + per-date ERA5) on 18 out-of-cal days",
                 fontsize=12)
    out_plot = OUT / "extended_validation_ship.png"
    fig.savefig(out_plot, dpi=130)
    plt.close(fig)
    print(f"\nwrote {OUT / 'extended_validation_ship.txt'}, {OUT / 'extended_validation_ship.json'}, {out_plot}.")


if __name__ == "__main__":
    main()
