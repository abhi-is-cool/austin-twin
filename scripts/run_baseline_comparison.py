"""Head-to-head: calibrated simulator vs linear-regression baselines.

Asks "does the physics simulator actually beat a 3-coefficient linear regression
on the same land-cover channels?" This is the question every reviewer asks first
about a physics-based UHI model.

Three predictors, same train/test split as scripts/run_calibration.py:
  - simulator   : calibrated SimConfig from outputs/calibrated_config.json
  - regression  : LST anomaly ~ impervious + vegetation + water  (lstsq)
  - regression* : above + (x, y, x^2, y^2, xy)  (strictly more info than simulator)

Train days: 2024-08-11, 16, 19. Test days: 2024-08-21, 22. All metrics are
computed on spatial anomaly fields (each field mean-subtracted before scoring),
matching the simulator-validation protocol.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from austin_twin import baseline_regression as breg
from austin_twin.calibration import (
    DayMetrics, evaluate as evaluate_simulator, load_calibrated_config,
)
from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CAL = OUT / "calibrated_config.json"

TRAIN_DATES = ["2024-08-11", "2024-08-16", "2024-08-19"]
TEST_DATES = ["2024-08-21", "2024-08-22"]
VIS_DATE = "2024-08-19"  # the figure goes on the peak heat-wave day


def _load_modis(dates: list[str], grid) -> dict[str, np.ndarray]:
    return {d: fetch_aqua_lst(d, grid, RAW / "modis") for d in dates}


def main() -> None:
    print("[1/4] Austin grid + WorldCover land cover...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, cache_dir=RAW / "worldcover")

    print("[2/4] MODIS LST for train + test...")
    modis_train = _load_modis(TRAIN_DATES, grid)
    modis_test = _load_modis(TEST_DATES, grid)

    print("[3/4] Fitting regression baselines on the same train days...")
    reg_simple = breg.fit(landuse, modis_train, use_position=False)
    reg_pos = breg.fit(landuse, modis_train, use_position=True)

    # Print learned coefficients so the regression isn't a black box.
    print(f"      simple regression coefs (anomaly ~ feats):")
    for name, c in zip(reg_simple.feature_names, reg_simple.coefs[:-1]):
        print(f"        {name:<24} {c:+.3f}")
    print(f"        {'intercept':<24} {reg_simple.coefs[-1]:+.3f}")

    print("[4/4] Scoring simulator (calibrated) and both regressions on all days...")
    base = SimConfig(duration_hours=48.0)
    cfg = load_calibrated_config(CAL, base_config=base) if CAL.exists() else base
    sim_train: list[DayMetrics] = evaluate_simulator(landuse, cfg, modis_train)
    sim_test: list[DayMetrics] = evaluate_simulator(landuse, cfg, modis_test)
    reg_simple_all = breg.evaluate(reg_simple, landuse, {**modis_train, **modis_test})
    reg_pos_all = breg.evaluate(reg_pos, landuse, {**modis_train, **modis_test})

    # --- Build the metrics table ---
    def _row(date: str, simulator: DayMetrics, split: str) -> str:
        rmse_s = simulator.rmse; r_s = simulator.pearson_r
        rmse_r1, r_r1 = reg_simple_all[date]
        rmse_r2, r_r2 = reg_pos_all[date]
        return (f"  {split:<6} {date:<12} | "
                f"sim   r={r_s:+.3f} RMSE={rmse_s:.3f} | "
                f"reg   r={r_r1:+.3f} RMSE={rmse_r1:.3f} | "
                f"reg+pos r={r_r2:+.3f} RMSE={rmse_r2:.3f}")

    lines = [
        "Per-day metrics on anomaly fields (spatial-mean subtracted)",
        "  split  date         |    simulator (calibrated) |    landuse-only lstsq  |  landuse+position lstsq",
        "  " + "-" * 110,
    ]
    for m in sim_train:
        lines.append(_row(m.date, m, "train"))
    for m in sim_test:
        lines.append(_row(m.date, m, "test"))

    # --- Aggregated means ---
    def _mean(metric: str, sim_metrics: list[DayMetrics], reg_dict: dict[str, tuple[float, float]]) -> tuple[float, float]:
        sim_vals = [getattr(m, metric) for m in sim_metrics]
        reg_vals = [reg_dict[m.date][0 if metric == "rmse" else 1] for m in sim_metrics]
        return float(np.mean(sim_vals)), float(np.mean(reg_vals))

    lines.append("")
    lines.append("Aggregated mean metrics:")
    for split, group in [("TRAIN", sim_train), ("TEST ", sim_test)]:
        sim_rmse = float(np.mean([m.rmse for m in group]))
        sim_r = float(np.mean([m.pearson_r for m in group]))
        rs1 = [reg_simple_all[m.date] for m in group]
        rs2 = [reg_pos_all[m.date] for m in group]
        reg_rmse = float(np.mean([x[0] for x in rs1]))
        reg_r = float(np.mean([x[1] for x in rs1]))
        reg2_rmse = float(np.mean([x[0] for x in rs2]))
        reg2_r = float(np.mean([x[1] for x in rs2]))
        lines.append(f"  {split} (n={len(group)})  | "
                     f"sim   r={sim_r:+.3f}  RMSE={sim_rmse:.3f} | "
                     f"reg   r={reg_r:+.3f}  RMSE={reg_rmse:.3f} | "
                     f"reg+pos r={reg2_r:+.3f}  RMSE={reg2_rmse:.3f}")

    table = "\n".join(lines)
    print()
    print(table)
    (OUT / "baseline_comparison.txt").write_text(table + "\n")

    # --- 4-panel figure on the visualization day ---
    lst_vis = (modis_train | modis_test)[VIS_DATE]
    sim_metric_for_vis = next((m for m in sim_train + sim_test if m.date == VIS_DATE), None)
    _save_four_panel(landuse, lst_vis, cfg, reg_simple, reg_pos, sim_metric_for_vis,
                     date=VIS_DATE, out_path=OUT / "baseline_comparison.png")

    print(f"\ndone. wrote {OUT / 'baseline_comparison.txt'} and {OUT / 'baseline_comparison.png'}.")


def _save_four_panel(landuse, lst, cfg, reg_simple, reg_pos, sim_metric, date, out_path):
    """MODIS anomaly + three model anomaly maps, shared color scale."""
    from austin_twin.simulator import run
    # Run simulator to get its peak-heat frame.
    result = run(landuse, cfg)
    times = result.times_hours
    T = result.temperature
    mean_t = np.array([
        float(np.nanmean(T[i])) if np.isfinite(T[i]).any() else -np.inf
        for i in range(T.shape[0])
    ])
    masked = np.where(times >= 24.0, mean_t, -np.inf)
    T_sim = T[int(np.argmax(masked))]

    pred_simple = reg_simple.predict_map(landuse)
    pred_pos = reg_pos.predict_map(landuse)

    city = landuse["city_mask"].values
    valid = city & np.isfinite(lst) & np.isfinite(T_sim) & np.isfinite(pred_simple)
    lst_anom = lst - np.nanmean(lst[valid])
    sim_anom = T_sim - np.nanmean(T_sim[valid])
    reg_anom = pred_simple - np.nanmean(pred_simple[valid])
    reg_pos_anom = pred_pos - np.nanmean(pred_pos[valid])

    def _rmse_r(p):
        po = p[valid]; oo = lst_anom[valid]
        po = po - po.mean(); oo = oo - oo.mean()
        return float(np.sqrt(np.mean((po - oo) ** 2))), float(np.corrcoef(po, oo)[0, 1])

    rmse_sim, r_sim = _rmse_r(sim_anom)
    rmse_reg, r_reg = _rmse_r(reg_anom)
    rmse_pos, r_pos = _rmse_r(reg_pos_anom)

    extent = (
        float(landuse["x"].min()), float(landuse["x"].max()),
        float(landuse["y"].min()), float(landuse["y"].max()),
    )
    vmax = float(max(np.nanpercentile(np.abs(lst_anom), 98),
                     np.nanpercentile(np.abs(sim_anom), 98),
                     np.nanpercentile(np.abs(reg_anom), 98),
                     np.nanpercentile(np.abs(reg_pos_anom), 98)))

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), constrained_layout=True)
    for ax, arr, title in zip(
        axes,
        [lst_anom, sim_anom, reg_anom, reg_pos_anom],
        [f"MODIS anomaly\n({date})",
         f"Calibrated simulator\nr = {r_sim:+.3f}, RMSE = {rmse_sim:.2f} °C",
         f"Linear regression (landuse only)\nr = {r_reg:+.3f}, RMSE = {rmse_reg:.2f} °C",
         f"Linear regression (landuse + position)\nr = {r_pos:+.3f}, RMSE = {rmse_pos:.2f} °C"],
    ):
        im = ax.imshow(arr, extent=extent, origin="upper", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.85, label="°C")

    fig.suptitle(
        f"Reviewer baseline check on {date}: physics vs lstsq on the same channels",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
