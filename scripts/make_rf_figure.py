"""Figure: a ceiling bounds a model only at its own receptive field.

The JURSE draft shipped with fig1_ceiling_definition.pdf under a caption
describing per-cell versus context-matched ceilings. That figure is the
univariate/multivariate scatter -- it is about channel count, not spatial
support -- so it argued a point the paper does not make. This is the
figure the paper actually needs.

One job: show that the CNN crosses the per-cell ceiling as soon as it has
any spatial reach, tracks the context ceiling when its reach matches, and
passes the context ceiling when its reach exceeds the specification. Four
small multiples carry it; a single panel with four cities plus eight
reference lines does not read at column width.

Panels share both axes so only the outer labels are drawn, which is what
buys the room to run this at IEEEtran single-column width (3.5in).

Colours are categorical (identity: which ceiling), taken in fixed slot
order from the same reference palette as make_figure1.py. The two ceilings
are additionally distinguished by line style, so identity is never carried
by colour alone.

Writes PDF (paper) and PNG (slides), light by default, `--dark` for a dark
deck.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "cnn_validation.json"
FIGS = ROOT / "outputs" / "figures"

# Same validated palette as make_figure1.py.
LIGHT = dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
             muted="#8a8880", s1="#2a78d6", s2="#eb6834", ring="#fcfcfb")
DARK = dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
            muted="#7d7b73", s1="#3987e5", s2="#d95926", ring="#1a1a19")

RF = np.array([1, 7, 13])
ORDER = ["austin", "phoenix", "denver", "miami"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()
    C = DARK if args.dark else LIGHT

    recs = {r["city"]: r for r in json.loads(SRC.read_text())}

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.45), sharex=True,
                             sharey=True)
    fig.patch.set_facecolor(C["surface"])

    for ax, city in zip(axes.ravel(), ORDER):
        r = recs[city]
        ax.set_facecolor(C["surface"])

        cnn = np.array([r["cnn1"], r["cnn7"], r["cnn13"]])
        per_cell, context = r["per_cell"], r["context"]

        # Above the context ceiling a matched percent-of-ceiling figure
        # exceeds 100%. Very low contrast: context, not a mark.
        ax.axhspan(context, 1.02, color=C["s2"], alpha=0.08, zorder=0,
                   linewidth=0)

        ax.axhline(per_cell, color=C["s1"], lw=1.1, ls=(0, (4, 2.5)),
                   zorder=2)
        ax.axhline(context, color=C["s2"], lw=1.1, ls=(0, (1.4, 1.6)),
                   zorder=2)

        ax.plot(RF, cnn, color=C["primary"], lw=1.3, zorder=3)
        ax.scatter(RF, cnn, s=26, c=C["primary"], edgecolors=C["ring"],
                   linewidths=1.0, zorder=4)

        ax.text(0.035, 0.97, city.title(), transform=ax.transAxes,
                fontsize=8.5, color=C["primary"], ha="left", va="top",
                bbox=dict(facecolor=C["surface"], edgecolor="none",
                          boxstyle="round,pad=0.12"))

        ax.set_xticks(RF)
        ax.set_xlim(-0.6, 14.6)
        ax.set_ylim(0.30, 1.02)
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.grid(True, color=C["muted"], alpha=0.16, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C["muted"])
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=C["secondary"], labelsize=7, length=2.5,
                       pad=2)

    # Direct-labelled once, in the panel with the most room, rather than in
    # a legend box that would cost a fifth of the figure at this size.
    a = axes[0, 0]
    a.text(14.2, recs["austin"]["per_cell"] - 0.03, "per-cell ceiling",
           fontsize=6.4, color=C["s1"], ha="right", va="top")
    a.text(14.2, recs["austin"]["context"] - 0.025, "context ceiling",
           fontsize=6.4, color=C["s2"], ha="right", va="top")

    fig.tight_layout(rect=(0.055, 0.075, 1, 1), h_pad=0.9, w_pad=1.1)

    fig.text(0.53, 0.018, "CNN receptive field (cells, 500 m each)",
             fontsize=8, color=C["primary"], ha="center", va="bottom")
    fig.text(0.016, 0.54, "Correlation with observed LST", fontsize=8,
             color=C["primary"], ha="left", va="center", rotation=90)

    FIGS.mkdir(parents=True, exist_ok=True)
    suffix = "_dark" if args.dark else ""
    for ext in ("pdf", "png"):
        out = FIGS / f"fig_receptive_field{suffix}.{ext}"
        fig.savefig(out, dpi=300, facecolor=C["surface"])
        print(f"wrote {out.relative_to(ROOT)}")

    print()
    for city in ORDER:
        r = recs[city]
        print(f"  {city:8} per-cell {r['per_cell']:+.3f}  "
              f"RF=1 {r['cnn1']:+.3f}  context {r['context']:+.3f}  "
              f"RF=7 {r['cnn7']:+.3f}  RF=13 {r['cnn13']:+.3f}")
    print()
    print("  RF=1 within 0.07 of per-cell in all four: "
          f"{all(abs(recs[c]['cnn1'] - recs[c]['per_cell']) < 0.07 for c in ORDER)}")
    print("  RF=7 at or below context in all four:     "
          f"{all(recs[c]['cnn7'] <= recs[c]['context'] for c in ORDER)}")
    print("  RF=13 above context in all four:          "
          f"{all(recs[c]['cnn13'] > recs[c]['context'] for c in ORDER)}")


if __name__ == "__main__":
    main()
