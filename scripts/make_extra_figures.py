"""Three further renders of the shipped pipeline.

  1. `canopy_counterfactual_3d.gif` -- baseline surface temperature as
     relief beside the cooling field from the +20% canopy scenario, both on
     a slow turntable. The counterfactual result is a single number in the
     text (-0.74 °C citywide); as relief it becomes a shape, and the shape
     is not uniform -- cooling concentrates where there was impervious
     surface to convert.

  2. `sim_vs_modis.png` -- simulated peak-heat field against the MODIS
     observation for the same day, absolute on the top row and anomaly on
     the bottom. Table II scores mean-subtracted fields, so the bottom row
     is what the reported r and RMSE actually measure; the top row keeps
     the absolute cold offset visible instead of quietly differencing it
     away. Showing simulation alone would invite the reader to assume
     agreement.

  3. `arid_contrast.png` -- Austin, Phoenix and Riyadh: observed LST, the
     impervious channel, and LST against impervious per cell. This is what
     the ceiling diagnostic detects. In Austin the relationship is positive,
     in Riyadh it is negative -- built surface is *cooler* than the desert
     around it -- and a univariate "ceiling" is then undefined.

Figure 3 uses no simulator at all: inversion is a property of the data.
Riyadh uses the 25 km disc definition it was measured under; Austin and
Phoenix use their OSM boundaries, as in the 49-city sweep.
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_uncertainty import (  # noqa: E402
    _peak_frame, _r_and_rmse, _valid_cells_2d,
)
from disc_boundary import disc_boundary  # noqa: E402

from austin_twin.arco_era5 import forcing_for_modis_date  # noqa: E402
from austin_twin.calibration import load_calibrated_config  # noqa: E402
from austin_twin.cities import AUSTIN, city_from_point  # noqa: E402
from austin_twin.counterfactual import CANONICAL_SCENARIOS, run_scenarios  # noqa: E402
from austin_twin.grid import build_grid, fetch_boundary  # noqa: E402
from austin_twin.modis import fetch_aqua_lst  # noqa: E402
from austin_twin.simulator import SimConfig  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
FIGS = OUT / "figures"
DATE = "2024-08-21"

BG = "#14140f"
FG = "#f2f1ea"
MUTED = "#8a8880"
HOT = "inferno"
DIV = "RdBu_r"


# ---------------------------------------------------------------- helpers

def ship_config() -> SimConfig:
    return replace(
        load_calibrated_config(OUT / "calibrated_config_arco.json",
                               base_config=SimConfig(duration_hours=48.0)),
        use_pm_et=False,
    )


def austin_setup():
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=AUSTIN.worldcover_tile)
    forcing = forcing_for_modis_date(AUSTIN, DATE, cache_dir=RAW / "arco_era5")
    return grid, landuse, forcing


def crop_box(mask: np.ndarray) -> tuple[slice, slice]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def relief(ax, field, cmap, norm, spin_azim, zlim, title,
           z_aspect=0.50, zoom=1.02, elev=52.0, fontsize=11, title_y=0.97):
    ny, nx = field.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    ax.plot_surface(X, Y, field, facecolors=cmap(norm(field)),
                    rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=False)
    ax.set_zlim(*zlim)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_box_aspect((nx / max(ny, 1), 1.0, z_aspect), zoom=zoom)
    ax.view_init(elev=elev, azim=spin_azim)
    ax.set_axis_off()
    ax.set_title(title, color=FG, fontsize=fontsize, y=title_y)


def bar(fig, ax, cmap, norm, label):
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(m, ax=ax, shrink=0.55, pad=0.0, fraction=0.032)
    cb.set_label(label, color=FG, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(MUTED)


def pearson(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------- 1. counterfactual

def figure_counterfactual(fps: int, spin: float) -> None:
    print("[1/3] counterfactual relief")
    grid, landuse, forcing = austin_setup()
    wanted = [s for s in CANONICAL_SCENARIOS
              if s.name in ("baseline", "canopy_plus_20")]
    runs = run_scenarios(landuse, wanted, config=ship_config(), forcing=forcing)

    city = landuse["city_mask"].values.astype(bool)
    base = np.where(city[None], runs["baseline"].result.temperature, np.nan)
    scen = np.where(city[None], runs["canopy_plus_20"].result.temperature, np.nan)
    times = runs["baseline"].result.times_hours

    dt_h = float(times[1] - times[0])
    stride = max(1, int(round(1.0 / dt_h)))
    base, scen, times = base[::stride], scen[::stride], times[::stride]
    rs, cs = crop_box(city)
    base, scen = base[:, rs, cs], scen[:, rs, cs]
    delta = scen - base

    fin_b = base[np.isfinite(base)]
    vmin, vmax = float(np.percentile(fin_b, 1)), float(np.percentile(fin_b, 99))
    fin_d = delta[np.isfinite(delta)]
    dmax = float(np.percentile(np.abs(fin_d), 99.5))
    # counterfactual.summarize averages over every (cell, timestep) pair, not
    # over the final frame; report both so the animation's per-frame caption
    # cannot be mistaken for the headline number.
    print(f"      mean dT over the whole run: {float(np.nanmean(delta)):+.2f} °C")
    print(f"      mean dT at the final frame: {float(np.nanmean(delta[-1])):+.2f} °C")

    n_hot = colors.Normalize(vmin=vmin, vmax=vmax)
    n_div = colors.TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
    c_hot, c_div = plt.get_cmap(HOT), plt.get_cmap(DIV)

    frames = []
    n = len(base)
    for i in range(n):
        az = -60 + spin * i / max(n - 1, 1)
        fig = plt.figure(figsize=(11.0, 4.1), facecolor=BG)
        a1 = fig.add_subplot(121, projection="3d", facecolor=BG)
        relief(a1, base[i], c_hot, n_hot, az, (vmin, vmax),
               f"baseline   t = {times[i]:5.1f} h")
        bar(fig, a1, c_hot, n_hot, "°C")
        a2 = fig.add_subplot(122, projection="3d", facecolor=BG)
        relief(a2, delta[i], c_div, n_div, az, (-dmax, dmax),
               f"+20% canopy − baseline   mean {np.nanmean(delta[i]):+.2f} °C")
        bar(fig, a2, c_div, n_div, "ΔT  °C")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "canopy_counterfactual_3d.gif"
    imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"      {out.name}  {len(frames)} frames")


# ---------------------------------------------------------- 2. sim vs obs

def figure_sim_vs_obs() -> None:
    print("[2/3] simulated vs observed")
    grid, landuse, forcing = austin_setup()
    from austin_twin.simulator import run as run_sim

    sim = _peak_frame(run_sim(landuse, ship_config(), forcing=forcing))
    obs = fetch_aqua_lst(DATE, grid, RAW / "modis")
    city = landuse["city_mask"].values.astype(bool)
    valid = _valid_cells_2d(sim, obs, city)

    # Table II reports r and RMSE on mean-subtracted fields (see
    # bootstrap_uncertainty._r_and_rmse): the slab has no absolute-temperature
    # anchor, so only the spatial pattern is being scored. Quoting a raw RMSE
    # here would look like a different — and much worse — model.
    r, rmse = _r_and_rmse(sim[valid], obs[valid])
    offset = float(np.mean(sim[valid]) - np.mean(obs[valid]))
    print(f"      r = {r:+.3f}   anomaly RMSE = {rmse:.2f} °C   "
          f"mean offset = {offset:+.2f} °C   n = {int(valid.sum())}")

    show = np.where(valid, 1.0, np.nan)
    s_m, o_m = sim * show, obs * show
    # Residual of the anomalies, matching how the score is computed.
    res = (s_m - np.mean(sim[valid])) - (o_m - np.mean(obs[valid]))
    rs, cs = crop_box(valid)
    s_m, o_m, res = s_m[rs, cs], o_m[rs, cs], res[rs, cs]


    # Top row absolute, bottom row anomaly. On a shared absolute scale the
    # simulation just looks uniformly dim, which hides the pattern agreement
    # the score is actually about; on anomalies alone the cold offset would
    # vanish entirely. Both rows, so neither fact is hidden by the other.
    s_anom = s_m - np.mean(sim[valid])
    o_anom = o_m - np.mean(obs[valid])
    raw = s_m - o_m

    both = np.concatenate([s_m[np.isfinite(s_m)], o_m[np.isfinite(o_m)]])
    vmin, vmax = float(np.percentile(both, 1)), float(np.percentile(both, 99))
    ba = np.concatenate([s_anom[np.isfinite(s_anom)], o_anom[np.isfinite(o_anom)]])
    amax = float(np.percentile(np.abs(ba), 98))
    gmax = float(np.percentile(np.abs(raw[np.isfinite(raw)]), 98))
    rmax = float(np.percentile(np.abs(res[np.isfinite(res)]), 98))

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.8), facecolor=BG)
    panels = [
        (s_m, HOT, dict(vmin=vmin, vmax=vmax), "simulated (ship config)", "°C"),
        (o_m, HOT, dict(vmin=vmin, vmax=vmax), f"MODIS Aqua observed  {DATE}", "°C"),
        (raw, DIV, dict(vmin=-gmax, vmax=gmax), "raw difference (carries the offset)", "ΔT  °C"),
        (s_anom, DIV, dict(vmin=-amax, vmax=amax), "simulated anomaly", "ΔT  °C"),
        (o_anom, DIV, dict(vmin=-amax, vmax=amax), "observed anomaly", "ΔT  °C"),
        (res, DIV, dict(vmin=-rmax, vmax=rmax), "residual of anomalies — what is scored", "ΔT  °C"),
    ]
    for ax, (field, cmap, kw, title, lab) in zip(axes.ravel(), panels):
        ax.set_facecolor(BG)
        im = ax.imshow(field, cmap=cmap, origin="upper",
                       interpolation="nearest", **kw)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, color=FG, fontsize=11, pad=8)
        cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
        cb.set_label(lab, color=FG, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(MUTED)

    fig.suptitle(f"Austin, {DATE}   r = {r:+.3f}   anomaly RMSE = {rmse:.2f} °C   "
                 f"(mean offset {offset:+.2f} °C, removed)   "
                 f"n = {int(valid.sum())} cells",
                 color=FG, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "sim_vs_modis.png"
    fig.savefig(out, dpi=170, facecolor=BG)
    plt.close(fig)
    print(f"      {out.name}")


# ------------------------------------------------------ 3. arid contrast

# name, OSM query, lat, lon, boundary kind, label
ARID_PANEL = [
    ("austin", "Austin, Texas, USA", 30.27, -97.74, "osm", "Austin, TX  (humid subtropical)"),
    ("phoenix", "Phoenix, Arizona, USA", 33.45, -112.07, "osm", "Phoenix, AZ  (hot desert)"),
    ("riyadh", "Riyadh, Saudi Arabia", 24.71, 46.68, "disc", "Riyadh, SA  (hot desert, 25 km disc)"),
]


def multiday_r(city_name: str) -> tuple[float, float]:
    """(univariate, multivariate) multi-day mean ceiling from the sweep artifacts."""
    for f in ("full_sweep.jsonl", "global_extension.jsonl"):
        p = OUT / f
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["city"] == city_name:
                return float(rec["ceiling_r"]), float(rec["ceiling_multi_r"])
    return float("nan"), float("nan")


def best_day(grid, dates: list[str]) -> tuple[str, np.ndarray]:
    """The listed date with the most valid LST cells (all are already cached)."""
    best, best_n, best_lst = None, -1, None
    for d in dates:
        try:
            lst = fetch_aqua_lst(d, grid, RAW / "modis")
        except Exception:
            continue
        n = int(np.isfinite(lst).sum())
        if n > best_n:
            best, best_n, best_lst = d, n, lst
    if best is None:
        raise RuntimeError("no usable LST date")
    return best, best_lst


def sweep_dates(city_name: str) -> list[str]:
    for f in ("full_sweep.jsonl", "global_extension.jsonl"):
        p = OUT / f
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip() and json.loads(line)["city"] == city_name:
                return json.loads(line)["dates"]
    return []


def figure_arid_contrast() -> None:
    print("[3/3] arid contrast")
    rows = []
    for name, query, lat, lon, kind, label in ARID_PANEL:
        city = city_from_point(name, query, lat, lon)
        if kind == "disc":
            boundary = disc_boundary(lat, lon, city.utm_crs)
        else:
            boundary = fetch_boundary(city,
                                      cache_path=RAW / f"{name}_boundary.geojson")
        grid = build_grid(boundary, resolution_m=500.0)
        landuse = build_worldcover_landuse(boundary, grid, RAW,
                                           tile=city.worldcover_tile)
        dates = sweep_dates(name) or [DATE]
        day, lst = best_day(grid, dates)

        mask = landuse["city_mask"].values.astype(bool)
        imp = landuse["impervious_frac"].values.astype(float)
        v = mask & np.isfinite(lst) & np.isfinite(imp)
        r_day = pearson(lst[v], imp[v])
        r_uni, r_multi = multiday_r(name)
        print(f"      {name:<9} {day}  r(day) = {r_day:+.3f}   "
              f"multi-day r = {r_uni:+.3f}   R = {r_multi:+.3f}   n = {int(v.sum())}")
        rows.append(dict(label=label, lst=np.where(v, lst, np.nan),
                         imp=np.where(v, imp, np.nan), v=v, day=day,
                         r_day=r_day, r_uni=r_uni, r_multi=r_multi))

    fig, axes = plt.subplots(len(rows), 3, figsize=(12.4, 4.15 * len(rows)),
                             facecolor=BG)
    for j, row in enumerate(rows):
        rs, cs = crop_box(row["v"])
        lst_c, imp_c = row["lst"][rs, cs], row["imp"][rs, cs]

        ax = axes[j, 0]
        ax.set_facecolor(BG)
        fin = lst_c[np.isfinite(lst_c)]
        im = ax.imshow(lst_c, cmap=HOT, origin="upper", interpolation="nearest",
                       vmin=float(np.percentile(fin, 2)),
                       vmax=float(np.percentile(fin, 98)))
        ax.set_title(f"{row['label']}\nMODIS LST  {row['day']}",
                     color=FG, fontsize=10, pad=6)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("°C", color=FG, fontsize=8)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
        cb.outline.set_edgecolor(MUTED)

        ax2 = axes[j, 1]
        ax2.set_facecolor(BG)
        im2 = ax2.imshow(imp_c, cmap="cividis", origin="upper",
                         interpolation="nearest", vmin=0.0, vmax=1.0)
        ax2.set_title("impervious fraction", color=FG, fontsize=10, pad=6)
        cb2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
        cb2.ax.tick_params(colors=MUTED, labelsize=7)
        cb2.outline.set_edgecolor(MUTED)

        for a in (ax, ax2):
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)

        ax3 = axes[j, 2]
        ax3.set_facecolor(BG)
        x = row["imp"][row["v"]]
        y = row["lst"][row["v"]]
        colour = "#2a78d6" if row["r_day"] > 0 else "#eb6834"
        ax3.scatter(x, y, s=3, alpha=0.16, color=colour, edgecolors="none")
        if len(x) > 2:
            b, a0 = np.polyfit(x, y, 1)
            xs = np.linspace(float(x.min()), float(x.max()), 50)
            ax3.plot(xs, a0 + b * xs, color=FG, lw=1.6)
        ax3.set_xlabel("impervious fraction", color=MUTED, fontsize=9)
        ax3.set_ylabel("LST  °C", color=MUTED, fontsize=9)
        ax3.tick_params(colors=MUTED, labelsize=8)
        for s in ax3.spines.values():
            s.set_color(MUTED)
        ax3.set_title(f"r = {row['r_day']:+.3f} this day" + chr(10) +
                      f"multi-day r = {row['r_uni']:+.3f},  R = {row['r_multi']:+.3f}",
                      color=FG, fontsize=10, pad=6)

    fig.suptitle("What the ceiling diagnostic sees: land cover against temperature",
                 color=FG, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "arid_contrast.png"
    fig.savefig(out, dpi=165, facecolor=BG)
    plt.close(fig)
    print(f"      {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--spin", type=float, default=50.0)
    ap.add_argument("--only", choices=["cf", "obs", "arid"], default=None)
    args = ap.parse_args()

    if args.only in (None, "cf"):
        figure_counterfactual(args.fps, args.spin)
    if args.only in (None, "obs"):
        figure_sim_vs_obs()
    if args.only in (None, "arid"):
        figure_arid_contrast()


if __name__ == "__main__":
    main()
