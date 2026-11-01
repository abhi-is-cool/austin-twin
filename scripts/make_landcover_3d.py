"""Land cover as 3D relief, and land cover against temperature.

  1. `landcover_3d.png` -- Austin's three input channels (impervious,
     vegetation, water) beside observed LST, each as relief. These are
     everything the simulator consumes about the surface; seeing them as
     terrain makes the composite-absorption step legible.

  2. `landcover_vs_lst_3d.png` / `.gif` -- Austin and Riyadh, impervious
     relief beside LST relief, on a full 360-degree turntable. This is the
     inversion in one image: the built core rises as a plateau in *both*
     cities' impervious channel, and the temperature surface rises with it
     in Austin and sinks with it in Riyadh. Nothing about the land-cover
     input distinguishes the two cases -- which is exactly why a univariate
     land-cover ceiling reports "no model can work" for the arid one.

Land cover does not evolve, so the animation is a pure rotation rather
than a time series.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from disc_boundary import disc_boundary  # noqa: E402
from make_extra_figures import (  # noqa: E402
    BG, FG, HOT, MUTED, bar, best_day, crop_box, relief, sweep_dates,
)

from austin_twin.cities import city_from_point  # noqa: E402
from austin_twin.grid import build_grid, fetch_boundary  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
FIGS = OUT / "figures"

IMP = "cividis"
VEG = "YlGn"
WAT = colors.LinearSegmentedColormap.from_list(
    "water", [BG, "#3987e5"])


def load_city(name, query, lat, lon, kind):
    """Return (impervious, vegetation, water, lst, city_mask), all 2D."""
    city = city_from_point(name, query, lat, lon)
    if kind == "disc":
        boundary = disc_boundary(lat, lon, city.utm_crs)
    else:
        boundary = fetch_boundary(city, cache_path=RAW / f"{name}_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=city.worldcover_tile)
    day, lst = best_day(grid, sweep_dates(name) or [])
    mask = landuse["city_mask"].values.astype(bool)
    keep = mask & np.isfinite(lst)
    out = {}
    for key, var in (("imp", "impervious_frac"), ("veg", "vegetation_frac"),
                     ("wat", "water_mask")):
        out[key] = np.where(keep, landuse[var].values.astype(float), np.nan)
    out["lst"] = np.where(keep, lst, np.nan)
    out["mask"] = keep
    out["day"] = day
    return out


def panel(fig, pos, field, cmap_name, lo, hi, azim, title, label,
          z_aspect=0.38, zoom=1.22):
    """One relief panel. The land-cover channels are flatter than the
    temperature ones on purpose: WorldCover fractions at 500 m are close to
    binary cell-to-cell, so a tall z-exaggeration turns the built-up core
    into visual noise instead of the plateau it is."""
    ax = fig.add_subplot(*pos, projection="3d", facecolor=BG)
    cmap = (cmap_name if hasattr(cmap_name, "__call__")
            else plt.get_cmap(cmap_name))
    norm = colors.Normalize(vmin=lo, vmax=hi)
    relief(ax, field, cmap, norm, azim, (lo, hi), title,
           z_aspect=z_aspect, zoom=zoom, fontsize=11, title_y=1.05)
    bar(fig, ax, cmap, norm, label)
    return ax


# --------------------------------------------------- 1. Austin channels

def figure_channels(azim: float) -> None:
    print("[1/2] Austin input channels")
    d = load_city("austin", "Austin, Texas, USA", 30.27, -97.74, "osm")
    rs, cs = crop_box(d["mask"])
    imp, veg, wat, lst = (d[k][rs, cs] for k in ("imp", "veg", "wat", "lst"))
    fin = lst[np.isfinite(lst)]
    lo, hi = float(np.percentile(fin, 2)), float(np.percentile(fin, 98))
    print(f"      {d['day']}   impervious mean {np.nanmean(imp):.2f}   "
          f"vegetation mean {np.nanmean(veg):.2f}   "
          f"water cells {int(np.nansum(wat))}")

    fig = plt.figure(figsize=(11.4, 7.0), facecolor=BG)
    panel(fig, (2, 2, 1), imp, IMP, 0.0, 1.0, azim, "impervious fraction", "")
    panel(fig, (2, 2, 2), veg, VEG, 0.0, 1.0, azim, "vegetation fraction", "")
    panel(fig, (2, 2, 3), wat, WAT, 0.0, 1.0, azim, "water mask", "")
    panel(fig, (2, 2, 4), lst, HOT, lo, hi, azim,
          f"observed LST  {d['day']}", "°C", z_aspect=0.50, zoom=1.22)
    fig.suptitle("Austin: everything the simulator sees about the surface",
                 color=FG, fontsize=13, y=0.99)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.88, bottom=0.03,
                        wspace=0.02, hspace=0.20)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "landcover_3d.png"
    fig.savefig(out, dpi=165, facecolor=BG)
    plt.close(fig)
    print(f"      {out.name}")


# ------------------------------------------ 2. Austin vs Riyadh, 4 panels

PAIR = [
    ("austin", "Austin, Texas, USA", 30.27, -97.74, "osm", "Austin, TX"),
    ("riyadh", "Riyadh, Saudi Arabia", 24.71, 46.68, "disc", "Riyadh, SA"),
]


def _pair_frame(cities, azim):
    fig = plt.figure(figsize=(11.4, 7.4), facecolor=BG)
    for j, (d, label) in enumerate(cities):
        imp, lst = d["imp_c"], d["lst_c"]
        panel(fig, (2, 2, 2 * j + 1), imp, IMP, 0.0, 1.0, azim,
              f"{label} — impervious fraction", "")
        panel(fig, (2, 2, 2 * j + 2), lst, HOT, d["lo"], d["hi"], azim,
              f"{label} — LST   r = {d['r']:+.3f}", "°C",
              z_aspect=0.50, zoom=1.22)
    fig.suptitle("Same land-cover shape, opposite temperature response",
                 color=FG, fontsize=13, y=0.99)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.88, bottom=0.03,
                        wspace=0.02, hspace=0.24)
    return fig


def figure_pair(azim: float, frames_n: int, fps: int, animate: bool) -> None:
    print("[2/2] Austin vs Riyadh")
    cities = []
    for name, query, lat, lon, kind, label in PAIR:
        d = load_city(name, query, lat, lon, kind)
        rs, cs = crop_box(d["mask"])
        d["imp_c"], d["lst_c"] = d["imp"][rs, cs], d["lst"][rs, cs]
        fin = d["lst_c"][np.isfinite(d["lst_c"])]
        d["lo"], d["hi"] = (float(np.percentile(fin, 2)),
                            float(np.percentile(fin, 98)))
        v = d["mask"]
        d["r"] = float(np.corrcoef(d["lst"][v], d["imp"][v])[0, 1])
        print(f"      {name:<8} {d['day']}  r(LST, impervious) = {d['r']:+.3f}  "
              f"grid {d['imp_c'].shape}")
        cities.append((d, label))

    FIGS.mkdir(parents=True, exist_ok=True)
    fig = _pair_frame(cities, azim)
    still = FIGS / "landcover_vs_lst_3d.png"
    fig.savefig(still, dpi=165, facecolor=BG)
    plt.close(fig)
    print(f"      {still.name}")

    if not animate:
        return
    frames = []
    for i in range(frames_n):
        fig = _pair_frame(cities, azim + 360.0 * i / frames_n)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        if (i + 1) % 12 == 0:
            print(f"      frame {i + 1}/{frames_n}")
    out = FIGS / "landcover_vs_lst_3d.gif"
    imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"      {out.name}  {len(frames)} frames")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--fps", type=int, default=9)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--only", choices=["channels", "pair"], default=None)
    args = ap.parse_args()

    if args.only in (None, "channels"):
        figure_channels(args.azim)
    if args.only in (None, "pair"):
        figure_pair(args.azim, args.frames, args.fps, not args.no_gif)


if __name__ == "__main__":
    main()
