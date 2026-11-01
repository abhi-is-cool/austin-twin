"""Print version of the Austin/Riyadh inversion figure, for the paper.

The screen version (`landcover_vs_lst_3d.png`) is dark-ground, 11 inches
wide and carries four colourbars. None of that survives a single IEEE
column: it would be reduced to a third of its width, the dark ground would
flood the page with ink, and the colourbars would be unreadable at that
scale.

This renders the same four panels for print instead -- white ground, no
colourbars, ranges moved into the panel titles and the caption, type sized
so it is still legible at 3.4 in wide. The message is the shape, not the
value: the built core is a plateau in *both* cities' impervious channel,
and the temperature surface rises with it in Austin and sinks with it in
Riyadh.
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

from make_landcover_3d import PAIR, load_city  # noqa: E402
from make_extra_figures import crop_box  # noqa: E402

FIGS = ROOT / "outputs" / "figures"

INK = "#000000"
PAPER = "#ffffff"
IMP_CMAP = "cividis"
LST_CMAP = "inferno"


def relief_print(ax, field, cmap, lo, hi, title, azim, z_aspect):
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
    ax.set_title(title, color=INK, fontsize=6.6, pad=-2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=3.4, help="inches (one IEEE column)")
    ap.add_argument("--height", type=float, default=2.8)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

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
        print(f"  {name:<8} {d['day']}  r = {d['r']:+.3f}  "
              f"LST {d['lo']:.1f}-{d['hi']:.1f} C  grid {d['imp_c'].shape}")
        cities.append((d, label))

    fig = plt.figure(figsize=(args.width, args.height), facecolor=PAPER)
    for j, (d, label) in enumerate(cities):
        ax = fig.add_subplot(2, 2, 2 * j + 1, projection="3d", facecolor=PAPER)
        relief_print(ax, d["imp_c"], IMP_CMAP, 0.0, 1.0,
                     f"{label}: impervious", args.azim, 0.34)
        ax = fig.add_subplot(2, 2, 2 * j + 2, projection="3d", facecolor=PAPER)
        relief_print(ax, d["lst_c"], LST_CMAP, d["lo"], d["hi"],
                     f"{label}: LST, $r$ = {d['r']:+.2f}", args.azim, 0.50)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.97, bottom=0.0,
                        wspace=0.0, hspace=0.02)
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIGS / f"fig_inversion.{ext}"
        fig.savefig(out, dpi=args.dpi, facecolor=PAPER,
                    bbox_inches="tight", pad_inches=0.01)
        print(f"  wrote {out.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
