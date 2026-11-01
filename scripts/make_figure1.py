"""Figure 1: why the univariate data ceiling is the wrong definition.

The figure has one job: show that the univariate ceiling reports "no
land-cover model can work" for five arid cities, while the multivariate
ceiling over the channels the model actually consumes is positive
everywhere. Two facts have to land at a glance:

  * every city sits above the y = x line -- the multiple correlation is
    >= the univariate one by construction, so a three-channel model can
    exceed a one-channel "ceiling" with nothing wrong; and
  * five cities fall left of x = 0, where a "percent of ceiling" is
    undefined.

A scatter against the diagonal carries both in one panel, which a paired
bar or dumbbell over 49 cities would not do at slide size.

Colours are categorical (identity: two groups of city), taken in fixed slot
order from the reference palette and validated for CVD separation before
use. The highlighted group is additionally direct-labelled, so identity is
never carried by colour alone.

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
SRC = ROOT / "outputs" / "full_sweep.jsonl"
FIGS = ROOT / "outputs" / "figures"

# Reference palette, validated: node validate_palette.js "#2a78d6,#eb6834"
# -> all six checks PASS (worst adjacent CVD dE 24.7 protan, normal 33.6).
LIGHT = dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
             muted="#8a8880", s1="#2a78d6", s2="#eb6834", ring="#fcfcfb")
DARK = dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
            muted="#7d7b73", s1="#3987e5", s2="#d95926", ring="#1a1a19")

# Hand-placed, in offset points, with leader lines. Boise and Albuquerque
# sit 0.03 apart in R, so an automatic placer puts one label through the
# other's marker; the leaders make the association explicit either way.
# All offsets point right: a left-pointing label on Boise, the leftmost
# city, overruns the panel into the y-axis title.
LABEL_OFFSETS = {
    "boise": (30, 46), "albuquerque": (56, -2), "tucson": (38, -4),
    "phoenix": (2, 28), "billings": (26, 14),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()
    C = DARK if args.dark else LIGHT

    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    uni = np.array([r["ceiling_r"] for r in rows])
    mul = np.array([r["ceiling_multi_r"] for r in rows])
    names = [r["city"] for r in rows]
    neg = uni < 0

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    fig.patch.set_facecolor(C["surface"])
    ax.set_facecolor(C["surface"])

    lo, hi = -0.42, 0.88
    # No city falls below R = 0.22, so anchoring y at the data rather than at
    # x's range removes roughly a third of the panel as dead space.
    ylo = 0.12

    # Region where the univariate ceiling is negative and a percentage of it
    # is undefined. Kept very low contrast: context, not a mark.
    ax.axvspan(lo, 0, color=C["s2"], alpha=0.07, zorder=0, linewidth=0)
    ax.axvline(0, color=C["muted"], lw=1.0, zorder=1)

    # y = x. Everything must sit above it; that is the structural point.
    ax.plot([ylo, hi], [ylo, hi], ls=(0, (5, 4)), lw=1.5,
            color=C["muted"], zorder=1)
    ax.text(0.76, 0.735, "multivariate = univariate", color=C["muted"],
            fontsize=9, rotation=42, ha="center", va="center",
            rotation_mode="anchor")

    # Marks: >=8px, 2px surface ring so overlapping points stay separable.
    ax.scatter(uni[~neg], mul[~neg], s=78, c=C["s1"], edgecolors=C["ring"],
               linewidths=1.6, zorder=3, label=f"univariate ≥ 0  (n={int((~neg).sum())})")
    ax.scatter(uni[neg], mul[neg], s=110, c=C["s2"], edgecolors=C["ring"],
               linewidths=1.6, zorder=4, marker="D",
               label=f"univariate < 0  (n={int(neg.sum())})")

    for i, nm in enumerate(names):
        if not neg[i]:
            continue
        dx, dy = LABEL_OFFSETS.get(nm, (10, 10))
        ax.annotate(nm.replace("_", " ").title(), (uni[i], mul[i]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=9.5, color=C["primary"], zorder=5,
                    ha="left" if dx > 0 else "right", va="center",
                    arrowprops=dict(arrowstyle="-", color=C["muted"],
                                    lw=0.8, shrinkA=2, shrinkB=6,
                                    connectionstyle="arc3,rad=0"))

    ax.text(-0.398, 0.155,
            "univariate ceiling says\nno land-cover model can work",
            fontsize=9.5, color=C["secondary"], ha="left", va="bottom",
            linespacing=1.35)

    ax.set_xlim(lo, hi)
    ax.set_ylim(ylo, hi)
    ax.set_xlabel("Univariate ceiling  $r$(LST, impervious fraction)",
                  fontsize=11, color=C["primary"], labelpad=8)
    ax.set_ylabel("Multivariate ceiling  $R$ (impervious, vegetation, water)",
                  fontsize=11, color=C["primary"], labelpad=8)
    ax.set_title("The univariate ceiling is negative where the multivariate one is not",
                 fontsize=12.5, color=C["primary"], pad=14, loc="left")

    ax.grid(True, color=C["muted"], alpha=0.18, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["muted"])
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=C["secondary"], labelsize=10)

    leg = ax.legend(loc="upper left", frameon=False, fontsize=10,
                    handletextpad=0.6, borderaxespad=0.8)
    for t in leg.get_texts():
        t.set_color(C["primary"])

    fig.text(0.012, 0.015,
             f"{len(rows)} US cities · multi-day means · "
             f"multivariate never negative (min {mul.min():+.2f})",
             fontsize=8.5, color=C["muted"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    FIGS.mkdir(parents=True, exist_ok=True)
    suffix = "_dark" if args.dark else ""
    for ext in ("pdf", "png"):
        out = FIGS / f"fig1_ceiling_definition{suffix}.{ext}"
        fig.savefig(out, dpi=220, facecolor=C["surface"])
        print(f"wrote {out.relative_to(ROOT)}")

    print(f"\n  cities: {len(rows)}   univariate negative: {int(neg.sum())}"
          f"   multivariate negative: {int((mul < 0).sum())}")
    print(f"  all points above y=x: {bool(np.all(mul >= uni - 1e-9))}")
    print(f"  max multivariate among negative-univariate: "
          f"{mul[neg].max():+.3f}")


if __name__ == "__main__":
    main()
