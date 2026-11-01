"""Print figures sized for MDPI Remote Sensing (single column, ~16 cm text).

The conference figure is 3.4 in wide for an IEEE column; dropped into a
16 cm MDPI text block it would sit at half width with 6.6 pt type. These
are re-rendered at 6.3 in with type scaled to match, on a white ground.

Produces:
  fig_inversion_wide.pdf  -- Austin/Riyadh, impervious vs LST as relief
  fig_validation.pdf      -- simulated vs MODIS, absolute and anomaly
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_uncertainty import (  # noqa: E402
    _peak_frame, _r_and_rmse, _valid_cells_2d,
)
from make_extra_figures import DATE, RAW, austin_setup, crop_box, ship_config  # noqa: E402
from make_landcover_3d import PAIR, load_city  # noqa: E402

from austin_twin.modis import fetch_aqua_lst  # noqa: E402

FIGS = ROOT / "outputs" / "figures"
INK, PAPER = "#000000", "#ffffff"
HOT, DIV, IMP = "inferno", "RdBu_r", "cividis"


def relief(ax, field, cmap, lo, hi, title, azim, z_aspect, fs):
    ny, nx = field.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    norm = colors.Normalize(vmin=lo, vmax=hi)
    ax.plot_surface(X, Y, field, facecolors=plt.get_cmap(cmap)(norm(field)),
                    rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=False)
    ax.set_zlim(lo, hi)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_box_aspect((nx / max(ny, 1), 1.0, z_aspect), zoom=1.42)
    ax.view_init(elev=54, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor(PAPER)
    ax.set_title(title, color=INK, fontsize=fs, pad=-2)


def figure_inversion(width, height, fs, dpi, azim=-60.0):
    print("[1/2] inversion, MDPI width")
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
        print(f"      {name:<8} {d['day']}  r = {d['r']:+.3f}  "
              f"LST {d['lo']:.1f}-{d['hi']:.1f} C")
        cities.append((d, label))

    fig = plt.figure(figsize=(width, height), facecolor=PAPER)
    for j, (d, label) in enumerate(cities):
        ax = fig.add_subplot(2, 2, 2 * j + 1, projection="3d", facecolor=PAPER)
        relief(ax, d["imp_c"], IMP, 0.0, 1.0,
               f"({chr(97 + 2 * j)}) {label}: impervious fraction", azim, 0.34, fs)
        ax = fig.add_subplot(2, 2, 2 * j + 2, projection="3d", facecolor=PAPER)
        relief(ax, d["lst_c"], HOT, d["lo"], d["hi"],
               f"({chr(98 + 2 * j)}) {label}: LST, $r$ = {d['r']:+.2f}",
               azim, 0.50, fs)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.96, bottom=0.0,
                        wspace=0.0, hspace=0.04)
    out = FIGS / "fig_inversion_wide.pdf"
    fig.savefig(out, dpi=dpi, facecolor=PAPER, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=PAPER,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"      {out.name}")


def figure_validation(width, fs, dpi):
    print("[2/2] simulated vs observed, print")
    from austin_twin.simulator import run as run_sim

    grid, landuse, forcing = austin_setup()
    sim = _peak_frame(run_sim(landuse, ship_config(), forcing=forcing))
    obs = fetch_aqua_lst(DATE, grid, RAW / "modis")
    city = landuse["city_mask"].values.astype(bool)
    valid = _valid_cells_2d(sim, obs, city)

    r, rmse = _r_and_rmse(sim[valid], obs[valid])
    offset = float(np.mean(sim[valid]) - np.mean(obs[valid]))
    print(f"      r = {r:+.3f}  anomaly RMSE = {rmse:.2f} C  "
          f"offset = {offset:+.2f} C  n = {int(valid.sum())}")

    show = np.where(valid, 1.0, np.nan)
    s_m, o_m = sim * show, obs * show
    res = (s_m - np.mean(sim[valid])) - (o_m - np.mean(obs[valid]))
    rs, cs = crop_box(valid)
    s_m, o_m, res = s_m[rs, cs], o_m[rs, cs], res[rs, cs]
    s_a, o_a = s_m - np.mean(sim[valid]), o_m - np.mean(obs[valid])

    both = np.concatenate([s_m[np.isfinite(s_m)], o_m[np.isfinite(o_m)]])
    vmin, vmax = float(np.percentile(both, 1)), float(np.percentile(both, 99))
    ba = np.concatenate([s_a[np.isfinite(s_a)], o_a[np.isfinite(o_a)]])
    amax = float(np.percentile(np.abs(ba), 98))
    rmax = float(np.percentile(np.abs(res[np.isfinite(res)]), 98))

    ny, nx = s_m.shape
    height = width / 3.0 * (ny / nx) * 2.0 + 0.5
    fig, axes = plt.subplots(2, 3, figsize=(width, height), facecolor=PAPER)
    panels = [
        (s_m, HOT, vmin, vmax, "(a) Simulated", r"$^{\circ}$C"),
        (o_m, HOT, vmin, vmax, "(b) MODIS observed", r"$^{\circ}$C"),
        (s_m - o_m, DIV, None, None, "(c) Raw difference", r"$\Delta T$ ($^{\circ}$C)"),
        (s_a, DIV, -amax, amax, "(d) Simulated anomaly", r"$\Delta T$ ($^{\circ}$C)"),
        (o_a, DIV, -amax, amax, "(e) Observed anomaly", r"$\Delta T$ ($^{\circ}$C)"),
        (res, DIV, -rmax, rmax, "(f) Residual of anomalies", r"$\Delta T$ ($^{\circ}$C)"),
    ]
    raw = s_m - o_m
    gmax = float(np.percentile(np.abs(raw[np.isfinite(raw)]), 98))
    for ax, (field, cmap, lo, hi, title, lab) in zip(axes.ravel(), panels):
        if lo is None:
            lo, hi = -gmax, gmax
        ax.set_facecolor(PAPER)
        im = ax.imshow(field, cmap=cmap, vmin=lo, vmax=hi, origin="upper",
                       interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, color=INK, fontsize=fs, pad=3)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(lab, color=INK, fontsize=fs - 1)
        cb.ax.tick_params(colors=INK, labelsize=fs - 2)
        cb.outline.set_edgecolor("#999999")
    fig.tight_layout(pad=0.4)
    out = FIGS / "fig_validation.pdf"
    fig.savefig(out, dpi=dpi, facecolor=PAPER, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=PAPER,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"      {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=6.3, help="inches (~16 cm)")
    ap.add_argument("--height", type=float, default=5.2)
    ap.add_argument("--fontsize", type=float, default=9.0)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--only", choices=["inv", "val"], default=None)
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "inv"):
        figure_inversion(args.width, args.height, args.fontsize, args.dpi)
    if args.only in (None, "val"):
        figure_validation(args.width, args.fontsize, args.dpi)


if __name__ == "__main__":
    main()
