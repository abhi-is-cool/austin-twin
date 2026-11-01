"""Plot land-use channels and render simulation output as an animated GIF."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .simulator import SimResult

if TYPE_CHECKING:
    from .counterfactual import ScenarioRun
    from .sensitivity import ZoneMetrics


def plot_landuse(landuse: xr.Dataset, out_path: Path, title: str | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    extent = (
        float(landuse["x"].min()),
        float(landuse["x"].max()),
        float(landuse["y"].min()),
        float(landuse["y"].max()),
    )
    for ax, name, cmap in zip(
        axes,
        ["impervious_frac", "vegetation_frac", "water_mask"],
        ["pink_r", "Greens", "Blues"],
    ):
        arr = landuse[name].values.astype(float)
        arr = np.where(landuse["city_mask"].values, arr, np.nan)
        im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
    if title is None:
        source = landuse.attrs.get("source", "unknown")
        title = f"Land-use channels (Austin) — source: {source}"
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def animate_temperature(
    result: SimResult,
    landuse: xr.Dataset,
    out_path: Path,
    stride: int = 6,
    fps: int = 8,
) -> None:
    """Render the temperature history as an animated GIF.

    `stride` thins frames; default keeps one frame per hour at dt=600s.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extent = (
        float(landuse["x"].min()),
        float(landuse["x"].max()),
        float(landuse["y"].min()),
        float(landuse["y"].max()),
    )

    T_all = result.temperature[::stride]
    times = result.times_hours[::stride]
    finite = T_all[np.isfinite(T_all)]
    vmin, vmax = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))

    frames = []
    for T, t_h in zip(T_all, times):
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        im = ax.imshow(T, extent=extent, origin="upper", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"Surface T  |  t = {t_h:5.1f} h")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.85, label="°C")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps, loop=0)


def _peak_frame_index(result: SimResult) -> int:
    """Time index where the citywide mean temperature is highest."""
    T = result.temperature
    mean_t = np.nanmean(T.reshape(T.shape[0], -1), axis=1)
    return int(np.nanargmax(mean_t))


def plot_scenario_comparison(
    runs: dict[str, "ScenarioRun"],
    out_path: Path,
    baseline_name: str = "baseline",
) -> None:
    """Two-row grid: peak-heat absolute T (top) and ΔT vs baseline (bottom).

    Columns = scenarios in insertion order.
    """
    base_run = runs[baseline_name]
    peak_idx = _peak_frame_index(base_run.result)
    base_T = base_run.result.temperature[peak_idx]

    extent = (
        float(base_run.landuse["x"].min()),
        float(base_run.landuse["x"].max()),
        float(base_run.landuse["y"].min()),
        float(base_run.landuse["y"].max()),
    )

    # Shared color scales.
    finite_base = base_T[np.isfinite(base_T)]
    vmin_T, vmax_T = float(np.percentile(finite_base, 2)), float(np.percentile(finite_base, 98))

    all_diffs = []
    for name, r in runs.items():
        if name == baseline_name:
            continue
        dT = r.result.temperature[peak_idx] - base_T
        all_diffs.append(dT[np.isfinite(dT)])
    if all_diffs:
        stacked = np.concatenate(all_diffs)
        vmax_dT = float(np.nanpercentile(np.abs(stacked), 99))
    else:
        vmax_dT = 1.0

    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 9), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    im_top = None
    im_dT = None
    for col, (name, r) in enumerate(runs.items()):
        T_peak = r.result.temperature[peak_idx]
        ax_top = axes[0, col]
        im_top = ax_top.imshow(T_peak, extent=extent, origin="upper", cmap="inferno",
                                vmin=vmin_T, vmax=vmax_T)
        ax_top.set_title(f"{name}\n(t = {r.result.times_hours[peak_idx]:.1f} h)", fontsize=10)
        ax_top.set_xticks([])
        ax_top.set_yticks([])

        ax_bot = axes[1, col]
        if name == baseline_name:
            ax_bot.text(0.5, 0.5, "reference", ha="center", va="center", transform=ax_bot.transAxes,
                        fontsize=11, color="gray")
            ax_bot.set_xticks([])
            ax_bot.set_yticks([])
            continue
        dT = T_peak - base_T
        im_dT = ax_bot.imshow(dT, extent=extent, origin="upper", cmap="RdBu_r",
                               vmin=-vmax_dT, vmax=vmax_dT)
        cooled = np.nanmean(dT[np.isfinite(dT)])
        ax_bot.set_title(f"ΔT vs baseline\nmean = {cooled:+.2f} °C", fontsize=10)
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])

    fig.colorbar(im_top, ax=axes[0, :].tolist(), shrink=0.85, label="°C")
    if im_dT is not None:
        fig.colorbar(im_dT, ax=axes[1, :].tolist(), shrink=0.85, label="ΔT (°C)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_zone_sensitivity(
    metrics: list["ZoneMetrics"],
    landuse: xr.Dataset,
    out_path: Path,
    top_n: int = 3,
) -> None:
    """Three-panel diagnostic for the zonal sensitivity analysis.

    Left   : baseline impervious fraction with zone outlines and labels.
    Middle : choropleth of citywide cooling efficiency per zone (top-N starred).
    Right  : choropleth of local mean ΔT per zone (negative = zone cools itself).
    """
    ny, nx = landuse["city_mask"].values.shape
    extent = (
        float(landuse["x"].min()),
        float(landuse["x"].max()),
        float(landuse["y"].min()),
        float(landuse["y"].max()),
    )

    # Build zone-id raster: zones are ranked by efficiency for color sorting.
    ranked = sorted(metrics, key=lambda m: m.efficiency, reverse=True)
    rank_by_label = {m.zone.label: i for i, m in enumerate(ranked)}
    top_labels = {m.zone.label for m in ranked[:top_n]}

    efficiency_raster = np.full((ny, nx), np.nan, dtype=np.float32)
    local_raster = np.full((ny, nx), np.nan, dtype=np.float32)
    for m in metrics:
        efficiency_raster[m.zone.cell_mask] = m.efficiency
        local_raster[m.zone.cell_mask] = m.local_mean_dt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # --- Left: baseline impervious + zone labels ---
    ax = axes[0]
    imp = landuse["impervious_frac"].values.astype(float)
    imp = np.where(landuse["city_mask"].values, imp, np.nan)
    im0 = ax.imshow(imp, extent=extent, origin="upper", cmap="pink_r", vmin=0, vmax=1)
    ax.set_title("Baseline impervious fraction\n(zones labeled)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im0, ax=ax, shrink=0.8)
    for m in metrics:
        ys, xs = np.where(m.zone.cell_mask)
        if ys.size == 0:
            continue
        cy = float(landuse["y"].values[int(np.median(ys))])
        cx = float(landuse["x"].values[int(np.median(xs))])
        color = "red" if m.zone.label in top_labels else "black"
        weight = "bold" if m.zone.label in top_labels else "normal"
        ax.text(cx, cy, m.zone.label, ha="center", va="center",
                fontsize=9, color=color, fontweight=weight,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.7))

    # --- Middle: citywide cooling efficiency ---
    ax = axes[1]
    finite_eff = efficiency_raster[np.isfinite(efficiency_raster)]
    vmax_e = float(np.percentile(finite_eff, 99))
    im1 = ax.imshow(efficiency_raster, extent=extent, origin="upper",
                    cmap="Blues", vmin=0, vmax=vmax_e)
    ax.set_title("Cooling efficiency per zone\n(°C·m² of citywide cooling per m² converted)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im1, ax=ax, shrink=0.8)
    for m in ranked[:top_n]:
        ys, xs = np.where(m.zone.cell_mask)
        if ys.size == 0:
            continue
        cy = float(landuse["y"].values[int(np.median(ys))])
        cx = float(landuse["x"].values[int(np.median(xs))])
        rank = rank_by_label[m.zone.label] + 1
        ax.text(cx, cy, f"#{rank}\n{m.zone.label}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="red", ec="white"))

    # --- Right: local mean ΔT per zone ---
    ax = axes[2]
    finite_loc = local_raster[np.isfinite(local_raster)]
    if finite_loc.size:
        vmax_l = float(np.nanpercentile(np.abs(finite_loc), 99))
    else:
        vmax_l = 1.0
    im2 = ax.imshow(local_raster, extent=extent, origin="upper",
                    cmap="RdBu_r", vmin=-vmax_l, vmax=vmax_l)
    ax.set_title("Local mean ΔT inside each zone\n(blue = the zone cools itself)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im2, ax=ax, shrink=0.8, label="°C")

    fig.suptitle(f"Targeted canopy intervention — Austin (top {top_n} zones starred)", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
