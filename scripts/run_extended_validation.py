"""Extended validation: does the simulator generalize outside the calibration heat wave?

The calibrated SimConfig (coefficients fit on 2024-08-11/16/19) is held FIXED.
We test it on MODIS Aqua LST days drawn from non-extreme summer, spring, fall,
winter, and a different year — categories the calibrator never saw.

Same anomaly-normalization protocol as the original validation: each field has
its citywide spatial mean subtracted before scoring, so the absolute-T mismatch
between the simulator's synthetic forcing and the actual day's weather is
factored out. We are measuring spatial pattern fidelity only.

Pipeline:
  1. For each category, scan a date range with probe_coverage (fast, no warp)
     and keep the 2-3 clearest days.
  2. Cache the selection to outputs/extended_validation_selection.json so re-runs
     skip the scan.
  3. Fetch + evaluate calibrated simulator and both regression baselines on
     every selected day. Report metrics broken out by category.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from austin_twin import baseline_regression as breg
from austin_twin.calibration import evaluate as evaluate_simulator, load_calibrated_config
from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.modis import fetch_aqua_lst, probe_coverage
from austin_twin.simulator import SimConfig
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CAL = OUT / "calibrated_config.json"
SEL_CACHE = OUT / "extended_validation_selection.json"

# Categories and the date ranges to scan for clear days. Each range is
# (start, end_inclusive). Aug 2024 heat-wave days are the calibration set and
# are reported separately for context.
CATEGORIES: dict[str, list[tuple[str, str]]] = {
    "heat_wave_aug_2024 (in-distribution)": [("2024-08-11", "2024-08-22")],
    "summer_non_extreme_2024":              [("2024-06-01", "2024-06-30"),
                                             ("2024-09-01", "2024-09-30")],
    "spring_2024":                          [("2024-04-01", "2024-05-15")],
    "fall_2024":                            [("2024-10-15", "2024-11-30")],
    "winter_2023_2024":                     [("2023-12-15", "2024-02-15")],
    "summer_2023 (different year)":         [("2023-08-01", "2023-08-31")],
}
DAYS_PER_CATEGORY = 3
MIN_COVERAGE = 0.85  # require 85% Austin coverage to count as "clear"

# Re-use the same regression training days as the rest of the project.
TRAIN_DATES = ["2024-08-11", "2024-08-16", "2024-08-19"]


@dataclass
class Pick:
    date: str
    coverage: float
    mean_lst_c: float


def _daterange(start: str, end: str):
    s = date.fromisoformat(start); e = date.fromisoformat(end)
    d = s
    while d <= e:
        yield d.isoformat()
        d += timedelta(days=1)


def _scan_category(category: str, ranges: list[tuple[str, str]], grid, max_per_category: int) -> list[Pick]:
    """Probe every day in `ranges`; return the top-N by coverage above threshold."""
    print(f"  scanning '{category}'...")
    cands: list[Pick] = []
    for start, end in ranges:
        for d in _daterange(start, end):
            cov, mean_c, _n = probe_coverage(d, grid)
            if cov >= MIN_COVERAGE:
                cands.append(Pick(date=d, coverage=cov, mean_lst_c=mean_c))
    cands.sort(key=lambda p: p.coverage, reverse=True)
    picks = cands[:max_per_category]
    for p in picks:
        print(f"    -> {p.date}  cov={p.coverage:.2%}  mean LST={p.mean_lst_c:.1f}°C")
    return picks


def _select(grid) -> dict[str, list[Pick]]:
    """Either scan, or read the cached selection from disk."""
    if SEL_CACHE.exists():
        print(f"[select] re-using cached selection from {SEL_CACHE.name}")
        raw = json.loads(SEL_CACHE.read_text())
        return {cat: [Pick(**d) for d in lst] for cat, lst in raw.items()}

    print("[select] scanning MODIS for clear-sky candidate days (one-time, ~5-10 min)...")
    out: dict[str, list[Pick]] = {}
    for category, ranges in CATEGORIES.items():
        max_n = DAYS_PER_CATEGORY if "heat_wave" not in category else 5  # we already have these
        out[category] = _scan_category(category, ranges, grid, max_n)
    SEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SEL_CACHE.write_text(json.dumps(
        {cat: [asdict(p) for p in lst] for cat, lst in out.items()},
        indent=2,
    ))
    return out


def _category_for_date(d: str, selection: dict[str, list[Pick]]) -> str:
    for cat, picks in selection.items():
        if any(p.date == d for p in picks):
            return cat
    return "unknown"


def main() -> None:
    print("[1/4] Austin grid + WorldCover land cover...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, cache_dir=RAW / "worldcover")

    print("[2/4] Selecting clear-sky candidate days per category...")
    selection = _select(grid)
    all_dates = [p.date for picks in selection.values() for p in picks]
    print(f"      {len(all_dates)} validation days across {len(selection)} categories")

    print("[3/4] Fetching MODIS LST for every selected day (cached after first run)...")
    modis_days: dict[str, np.ndarray] = {}
    for d in all_dates:
        modis_days[d] = fetch_aqua_lst(d, grid, RAW / "modis")

    # Fit regression baselines on the same Aug 2024 train days as before.
    print("[4/4] Fitting regression baselines on Aug 2024 train days, evaluating everywhere...")
    train_modis = {d: modis_days[d] if d in modis_days else fetch_aqua_lst(d, grid, RAW / "modis")
                   for d in TRAIN_DATES}
    reg_simple = breg.fit(landuse, train_modis, use_position=False)
    reg_pos = breg.fit(landuse, train_modis, use_position=True)

    # Score the calibrated simulator on every day (one sim run, evaluated per day).
    base = SimConfig(duration_hours=48.0)
    cfg = load_calibrated_config(CAL, base_config=base) if CAL.exists() else base
    sim_metrics = evaluate_simulator(landuse, cfg, modis_days)
    sim_by_date = {m.date: m for m in sim_metrics}

    # Score regressions on every day.
    reg_simple_metrics = breg.evaluate(reg_simple, landuse, modis_days)
    reg_pos_metrics = breg.evaluate(reg_pos, landuse, modis_days)

    # --- Build per-category aggregates ---
    rows: list[str] = [
        "Per-day metrics on anomaly fields (calibrated coefficients FIXED, regression trained on Aug 11/16/19 only)",
        "  category                              date         | mean LST | sim r/RMSE     | reg r/RMSE     | reg+pos r/RMSE",
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
        f"{'sim r':>8} {'sim RMSE':>10} {'reg r':>8} {'reg RMSE':>10} "
        f"{'reg+pos r':>11} {'reg+pos RMSE':>13}"
    )
    rows.append("  " + "-" * 130)
    summary: dict[str, dict[str, float]] = {}
    for cat, items in grouped.items():
        if not items:
            continue
        mean_lst = float(np.mean([x[1] for x in items]))
        sim_r = float(np.mean([x[2][0] for x in items]))
        sim_rmse = float(np.mean([x[2][1] for x in items]))
        reg_r = float(np.mean([x[3][1] for x in items]))
        reg_rmse = float(np.mean([x[3][0] for x in items]))
        regp_r = float(np.mean([x[4][1] for x in items]))
        regp_rmse = float(np.mean([x[4][0] for x in items]))
        summary[cat] = dict(n=len(items), mean_lst=mean_lst,
                            sim_r=sim_r, sim_rmse=sim_rmse,
                            reg_r=reg_r, reg_rmse=reg_rmse,
                            reg_pos_r=regp_r, reg_pos_rmse=regp_rmse)
        rows.append(
            f"  {cat:<37} {len(items):>3} {mean_lst:>+9.1f}°C "
            f"{sim_r:>+8.3f} {sim_rmse:>10.2f} "
            f"{reg_r:>+8.3f} {reg_rmse:>10.2f} "
            f"{regp_r:>+11.3f} {regp_rmse:>13.2f}"
        )

    print()
    table = "\n".join(rows)
    print(table)
    (OUT / "extended_validation.txt").write_text(table + "\n")

    # --- One plot: per-day sim Pearson r as a function of mean LST, colored by category ---
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
    axes[0].set_ylabel("Calibrated simulator Pearson r")
    axes[0].set_title("Spatial pattern fidelity by season / extremeness")
    axes[0].axhline(0.0, color="gray", lw=0.5)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].set_xlabel("Mean MODIS LST on validation day (°C)")
    axes[1].set_ylabel("Calibrated simulator anomaly RMSE (°C)")
    axes[1].set_title("Anomaly RMSE by season / extremeness")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Extended validation: simulator generalization outside calibration regime", fontsize=12)
    out_plot = OUT / "extended_validation.png"
    fig.savefig(out_plot, dpi=130)
    plt.close(fig)
    print(f"\ndone. wrote {OUT / 'extended_validation.txt'} and {out_plot}.")


if __name__ == "__main__":
    main()
