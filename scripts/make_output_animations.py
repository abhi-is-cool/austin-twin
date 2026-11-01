"""Render the Austin simulation as 2D and 3D animations.

Runs the shipped configuration -- the ARCO-recalibrated coefficients, ARCO
forcing, 48 hours -- and writes two GIFs:

  * `austin_surface_2d.gif` -- map view of surface temperature with a
    citywide-mean trace underneath, so the diurnal cycle is legible as a
    curve and not only as a colour shift.
  * `austin_surface_3d.gif` -- the same field as a relief surface, height
    and colour both temperature, on a slow turntable. Vertical relief
    carries the urban-core anomaly better than colour alone, which is the
    one thing the flat map genuinely hides.

Both use fixed colour and height limits across all frames: a per-frame
rescale would make the night frames look as hot as noon.

Cells outside the city boundary are NaN and simply are not drawn, so the
city footprint reads as its own silhouette.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from austin_twin.arco_era5 import forcing_for_modis_date  # noqa: E402
from austin_twin.calibration import load_calibrated_config  # noqa: E402
from austin_twin.cities import AUSTIN  # noqa: E402
from austin_twin.grid import build_grid, fetch_boundary  # noqa: E402
from austin_twin.simulator import SimConfig, run  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
FIGS = OUT / "figures"
DATE = "2024-08-21"

CMAP = "inferno"
BG = "#14140f"
FG = "#f2f1ea"
MUTED = "#8a8880"


def simulate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T[t,y,x] masked to the city, times in hours, city mask)."""
    print("[setup] boundary, grid, land cover, ARCO forcing...")
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=AUSTIN.worldcover_tile)
    forcing = forcing_for_modis_date(AUSTIN, DATE, cache_dir=RAW / "arco_era5")

    cfg = replace(
        load_calibrated_config(OUT / "calibrated_config_arco.json",
                               base_config=SimConfig(duration_hours=48.0)),
        use_pm_et=False,
    )
    print(f"[run] 48 h, dx=500 m, grid {grid.shape}, "
          f"{int(grid.mask.sum())} city cells")
    result = run(landuse, cfg, forcing=forcing)

    city = landuse["city_mask"].values.astype(bool)
    T = np.where(city[None, :, :], result.temperature, np.nan)
    return T.astype(np.float32), result.times_hours, city


def thin(T: np.ndarray, times: np.ndarray, per_hour: float = 1.0):
    """Keep roughly `per_hour` frames per simulated hour."""
    dt_h = float(times[1] - times[0])
    stride = max(1, int(round(1.0 / (per_hour * dt_h))))
    return T[::stride], times[::stride]


def limits(T: np.ndarray) -> tuple[float, float]:
    finite = T[np.isfinite(T)]
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


def crop(T: np.ndarray) -> np.ndarray:
    """Trim the all-NaN margin around the city so the footprint fills the frame."""
    valid = np.isfinite(T).any(axis=0)
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    return T[:, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _style(ax) -> None:
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def render_2d(T, times, out_path: Path, fps: int) -> None:
    vmin, vmax = limits(T)
    mean_t = np.nanmean(T.reshape(len(T), -1), axis=1)
    frames = []
    for i, (field, t_h) in enumerate(zip(T, times)):
        fig = plt.figure(figsize=(6.4, 6.6), facecolor=BG)
        gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 1.0], hspace=0.18,
                              left=0.10, right=0.90, top=0.93, bottom=0.10)
        ax = fig.add_subplot(gs[0])
        ax.set_facecolor(BG)
        im = ax.imshow(field, cmap=CMAP, vmin=vmin, vmax=vmax, origin="upper",
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Austin surface temperature   t = {t_h:5.1f} h",
                     color=FG, fontsize=12, pad=10)
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label("°C", color=FG, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(MUTED)

        axl = fig.add_subplot(gs[1])
        _style(axl)
        axl.plot(times, mean_t, color=MUTED, lw=1.2)
        axl.plot(times[: i + 1], mean_t[: i + 1], color="#eb6834", lw=1.8)
        axl.scatter([t_h], [mean_t[i]], s=22, color="#eb6834", zorder=3)
        axl.set_xlim(float(times[0]), float(times[-1]))
        axl.set_xlabel("hours since start", color=MUTED, fontsize=9)
        axl.set_ylabel("citywide mean °C", color=MUTED, fontsize=8)

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"[2d] {out_path.name}  {len(frames)} frames")


def render_3d(T, times, out_path: Path, fps: int, spin_deg: float) -> None:
    vmin, vmax = limits(T)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(CMAP) if hasattr(cm, "get_cmap") else plt.get_cmap(CMAP)
    ny, nx = T.shape[1], T.shape[2]
    Y, X = np.mgrid[0:ny, 0:nx]

    frames = []
    n = len(T)
    for i, (field, t_h) in enumerate(zip(T, times)):
        fig = plt.figure(figsize=(7.2, 4.9), facecolor=BG)
        ax = fig.add_subplot(111, projection="3d", facecolor=BG)
        # Height and colour are the same variable on purpose: the relief is
        # the message, the colour just makes it readable at low grazing angle.
        ax.plot_surface(X, Y, field, facecolors=cmap(norm(field)),
                        rstride=1, cstride=1, linewidth=0, antialiased=False,
                        shade=False)
        ax.set_zlim(vmin, vmax)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_box_aspect((nx / max(ny, 1), 1.0, 0.50), zoom=1.02)
        ax.view_init(elev=52, azim=-60 + spin_deg * i / max(n - 1, 1))
        ax.set_axis_off()
        ax.set_title(f"Austin surface temperature as relief   t = {t_h:5.1f} h",
                     color=FG, fontsize=12, y=0.97)

        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(m, ax=ax, shrink=0.58, pad=0.0, fraction=0.035)
        cb.set_label("°C", color=FG, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(MUTED)

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"[3d] {out_path.name}  {len(frames)} frames")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--per-hour", type=float, default=1.0)
    ap.add_argument("--spin", type=float, default=50.0,
                    help="degrees of azimuth swept across the 3D animation")
    ap.add_argument("--only", choices=["2d", "3d"], default=None)
    args = ap.parse_args()

    T, times, _ = simulate()
    Tf, tf = thin(T, times, args.per_hour)
    Tf = crop(Tf)
    vmin, vmax = limits(Tf)
    print(f"[frames] {len(Tf)}   T range {vmin:.1f} to {vmax:.1f} °C")

    if args.only != "3d":
        render_2d(Tf, tf, FIGS / "austin_surface_2d.gif", args.fps)
    if args.only != "2d":
        render_3d(Tf, tf, FIGS / "austin_surface_3d.gif", args.fps, args.spin)


if __name__ == "__main__":
    main()
