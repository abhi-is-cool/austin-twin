"""Populate `tab:bootstrap_ci` and `tab:sensitivity` in main.tex.

Two analyses, both driven by re-simulating the ship configuration (and
its comparators) against the per-cell MODIS observations on their
respective held-out test days:

1.  Spatial block bootstrap for held-out test-set r and RMSE
    (`tab:bootstrap_ci`). Six configurations, each evaluated on its own
    2 held-out days:

      - Single-stage, legacy water     : Austin Aug 21, 22
      - Two-stage, synthetic forcing   : Austin Aug 21, 22
      - Ship (two-stage + ERA5)        : Austin Aug 21, 22
      - PM + ERA5 (canonical, K = 3)   : Austin Aug 21, 22
      - Phoenix, zero-shot             : Phoenix Jul 22, 24
      - Miami, zero-shot               : Miami Apr 16, 19

    Bootstrap: 6x6 block tile resampling (L = 6 cells side, ~3 km), pool
    resampled cells across both test days per replicate, B = 2000 reps.
    Report percentile 95 % CI on r and (where applicable) RMSE.

    Paired-difference CI on Δr = r_ship − r_two_stage_synthetic: same
    resampled cells drive both r's per replicate; report percentile CI
    of the difference. This is the correct test for "is the ERA5 gain
    distinguishable from zero" (individual CIs overlap by construction
    when Δr ≈ 0.01 and CI ≈ 0.02).

2.  One-at-a-time coefficient sensitivity for the ship configuration
    (`tab:sensitivity`). Four free coefficients (α_imp, α_veg, α_wat,
    k_ET); each perturbed independently by ±10 % and ±25 % from its
    calibrated value; report Δr and Δ RMSE relative to the calibrated
    ship-config test metrics.

    DE population std. dev. column is left as "—" since the calibration
    pipeline never captured the final DE population (would require a
    re-run with a callback that logs it).

Output: outputs/bootstrap_uncertainty.json + a formatted stdout table
that can be pasted straight into main.tex.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from austin_twin.calibration import load_calibrated_config
from austin_twin.cities import AUSTIN, PHOENIX, MIAMI, CityConfig
from austin_twin.forcing import Forcing
from austin_twin.grid import fetch_boundary, build_grid
from austin_twin.modis import fetch_aqua_lst
from austin_twin.simulator import SimConfig, run
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

# --- Bootstrap parameters (from the paper spec) ---
BLOCK_SIDE = 6           # cells; 6 x 6 tile ≈ 3 km x 3 km
N_BOOT = 2000
PERTURBATIONS = [-0.25, -0.10, 0.10, 0.25]
SEED = 0


# --- Per-configuration recipe ---

_ERA5_AUSTIN = RAW / "era5" / "era5_2024-08-10_2024-08-23.nc"
_ERA5_PHOENIX = RAW / "era5" / "era5_phoenix_2024-07-15_2024-07-28.nc"
_ERA5_MIAMI = RAW / "era5" / "era5_miami_2024-04-04_2024-04-20.nc"


def _forcing_for(city: CityConfig, era5_nc: Path, date_iso: str) -> Forcing:
    prev = date.fromisoformat(date_iso) - timedelta(days=1)
    return Forcing.from_era5(
        era5_nc,
        start_iso=f"{prev.isoformat()}T{city.utc_offset_hours:02d}:00",
        duration_hours=48.0,
    )


def _peak_frame(result):
    T = result.temperature
    times = result.times_hours
    means = np.array([np.nanmean(f) if np.isfinite(f).any() else -np.inf for f in T])
    idx = int(np.argmax(np.where(times >= 24.0, means, -np.inf)))
    return T[idx]


def _load_ship_landuse(city: CityConfig):
    boundary = fetch_boundary(city, cache_path=RAW / f"{city.name}_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(
        boundary, grid, cache_dir=RAW / "worldcover", tile=city.worldcover_tile,
    )
    return grid, landuse


def _sim_and_obs_pair(city, era5_nc, cfg, date_iso, grid, landuse):
    """Return (sim_peak_frame, modis_frame) as 2D arrays over grid.shape."""
    forcing = _forcing_for(city, era5_nc, date_iso) if era5_nc is not None else None
    result = run(landuse, cfg, forcing=forcing)
    sim = _peak_frame(result)
    obs = fetch_aqua_lst(date_iso, grid, RAW / "modis")
    return sim, obs


# --- Bootstrap machinery ---

def _valid_cells_2d(sim, obs, city_mask):
    """Return a boolean mask (ny, nx) of jointly-valid cells inside city."""
    return np.isfinite(sim) & np.isfinite(obs) & city_mask


def _r_and_rmse(sim_flat, obs_flat):
    """Pearson r and RMSE on anomaly-normalized (mean-subtracted) inputs."""
    if sim_flat.size < 2:
        return np.nan, np.nan
    s = sim_flat - np.mean(sim_flat)
    o = obs_flat - np.mean(obs_flat)
    denom = float(np.std(s) * np.std(o))
    if denom == 0.0:
        return np.nan, np.nan
    r = float(np.corrcoef(s, o)[0, 1])
    rmse = float(np.sqrt(np.mean((s - o) ** 2)))
    return r, rmse


def _tile_origins(ny, nx, L, rng):
    """Sample tile top-left corners uniformly with replacement, enough to
    approximately match the total valid-cell count."""
    # Cover the domain with ceil(n_total / L^2) tiles.
    n_tiles = int(np.ceil(ny * nx / (L * L)))
    rows = rng.integers(0, max(1, ny - L + 1), size=n_tiles)
    cols = rng.integers(0, max(1, nx - L + 1), size=n_tiles)
    return rows, cols


def _sample_tile_origins_per_day(days_data, L, rng):
    """Return a list (one per day) of (rows, cols) tile-origin arrays.

    Each day gets an independent set of draws — same block grid definition
    (L-cell tile side), separate random origins per day. This matches the
    per-day resampling structure the paper's estimator requires: each day
    is treated as its own spatial-correlation domain."""
    origins = []
    for sim_arr, _, _ in days_data:
        ny, nx = sim_arr.shape
        origins.append(_tile_origins(ny, nx, L, rng))
    return origins


def _apply_tiles(days_data, tile_origins_per_day, L):
    """For each day, extract cells covered by the given tile origins and
    that are jointly valid, returning [(sim_flat, obs_flat) per day]."""
    per_day = []
    for (sim_arr, obs_arr, valid_mask), (rows, cols) in zip(days_data, tile_origins_per_day):
        ny, nx = sim_arr.shape
        sim_parts = []
        obs_parts = []
        for r0, c0 in zip(rows, cols):
            r1 = min(r0 + L, ny)
            c1 = min(c0 + L, nx)
            tile_valid = valid_mask[r0:r1, c0:c1]
            if not tile_valid.any():
                continue
            sim_parts.append(sim_arr[r0:r1, c0:c1][tile_valid])
            obs_parts.append(obs_arr[r0:r1, c0:c1][tile_valid])
        if sim_parts:
            per_day.append((np.concatenate(sim_parts), np.concatenate(obs_parts)))
        else:
            per_day.append((np.array([]), np.array([])))
    return per_day


def _per_day_avg_stat(day_arrays):
    """Return (mean per-day r, mean per-day RMSE) using per-day anomaly
    normalization, matching the paper's `evaluate()` convention.

    Days that yield too few cells to compute a valid statistic are
    dropped from the average (they contribute neither r nor RMSE)."""
    rs, rmses = [], []
    for s, o in day_arrays:
        r, rmse = _r_and_rmse(s, o)
        if np.isfinite(r):
            rs.append(r)
        if np.isfinite(rmse):
            rmses.append(rmse)
    r_avg = float(np.mean(rs)) if rs else np.nan
    rmse_avg = float(np.mean(rmses)) if rmses else np.nan
    return r_avg, rmse_avg


def _percentile_ci(values, level=95):
    lo = float(np.nanpercentile(values, (100 - level) / 2))
    hi = float(np.nanpercentile(values, 100 - (100 - level) / 2))
    return lo, hi


def _point_estimate_per_day_avg(days_data):
    """Deterministic point estimate: per-day anomaly r/RMSE averaged.
    Matches `evaluate()` -> mean of per-day metrics used throughout the paper."""
    full_arrays = [(s[v], o[v]) for s, o, v in days_data]
    return _per_day_avg_stat(full_arrays)


def bootstrap_r_rmse(days_data, B=N_BOOT, L=BLOCK_SIDE, seed=SEED):
    """95 % percentile CI for per-day-averaged r and RMSE via block bootstrap.

    Replicate procedure (matches paper convention):
      - draw an independent tile-origin sequence for each day;
      - for each day, extract jointly-valid cells covered by those tiles;
      - compute per-day anomaly r and RMSE on those cells;
      - the replicate statistic is the mean of the per-day values.
    """
    r_hat, rmse_hat = _point_estimate_per_day_avg(days_data)

    rng = np.random.default_rng(seed)
    r_reps = np.empty(B)
    rmse_reps = np.empty(B)
    for b in range(B):
        origins = _sample_tile_origins_per_day(days_data, L, rng)
        per_day = _apply_tiles(days_data, origins, L)
        r_reps[b], rmse_reps[b] = _per_day_avg_stat(per_day)
    return {
        "r_hat": r_hat, "r_ci": _percentile_ci(r_reps),
        "rmse_hat": rmse_hat, "rmse_ci": _percentile_ci(rmse_reps),
    }


def paired_delta_r(days_data_A, days_data_B, B=N_BOOT, L=BLOCK_SIDE, seed=SEED):
    """CI for r_A - r_B under the paper's per-day-averaged r estimator.

    Per replicate: draw ONE tile-origin sequence per day; apply those same
    origins to both configs' sim arrays (identical valid masks are used for
    A and B — caller intersects them beforehand). Compute per-day-averaged
    r for each config, take the difference. This preserves the paired
    structure across BOTH days and configs, so the CI reflects the ERA5
    contribution net of shared spatial-sampling noise."""
    assert len(days_data_A) == len(days_data_B)
    r_a_hat, _ = _point_estimate_per_day_avg(days_data_A)
    r_b_hat, _ = _point_estimate_per_day_avg(days_data_B)
    delta_hat = r_a_hat - r_b_hat

    rng = np.random.default_rng(seed)
    delta_reps = np.empty(B)
    for b in range(B):
        origins = _sample_tile_origins_per_day(days_data_A, L, rng)  # shared
        per_day_A = _apply_tiles(days_data_A, origins, L)
        per_day_B = _apply_tiles(days_data_B, origins, L)
        r_a, _ = _per_day_avg_stat(per_day_A)
        r_b, _ = _per_day_avg_stat(per_day_B)
        delta_reps[b] = r_a - r_b
    return {"delta_hat": float(delta_hat), "delta_ci": _percentile_ci(delta_reps)}


# --- OAT sensitivity ---

_COEF_NAMES = ("absorption_impervious", "absorption_vegetation",
               "absorption_water", "et_coeff")


def oat_sensitivity(base_cfg, city, era5_nc, test_dates, grid, landuse):
    """OAT ±10 %, ±25 % on 4 free coefs. Report Δr and ΔRMSE vs the base
    config's own held-out r/RMSE (pooled across test days, jointly-valid
    cells; no bootstrap here — deterministic point estimates)."""
    def _per_day_avg_r_rmse(cfg):
        pairs = [_sim_and_obs_pair(city, era5_nc, cfg, d, grid, landuse) for d in test_dates]
        day_arrays = []
        for sim, obs in pairs:
            v = _valid_cells_2d(sim, obs, landuse["city_mask"].values)
            day_arrays.append((sim[v], obs[v]))
        return _per_day_avg_stat(day_arrays)

    r_base, rmse_base = _per_day_avg_r_rmse(base_cfg)

    rows = []
    for coef in _COEF_NAMES:
        row = {"coef": coef, "base_value": float(getattr(base_cfg, coef))}
        for pct in PERTURBATIONS:
            perturbed = replace(base_cfg, **{coef: getattr(base_cfg, coef) * (1.0 + pct)})
            r_p, rmse_p = _per_day_avg_r_rmse(perturbed)
            row[f"delta_r_{int(pct*100):+d}pct"] = r_p - r_base
            row[f"delta_rmse_{int(pct*100):+d}pct"] = rmse_p - rmse_base
        # Rank by sum of |Δr| across the two ±25 % perturbations (matches
        # what the paper displays; keeps ranking stable to the visible cells).
        row["abs_delta_25"] = abs(row["delta_r_-25pct"]) + abs(row["delta_r_+25pct"])
        rows.append(row)

    ranked = sorted(rows, key=lambda x: -x["abs_delta_25"])
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return {"r_base": float(r_base), "rmse_base": float(rmse_base),
            "rows": rows, "ranking_metric": "|Δr(-25%)| + |Δr(+25%)|"}


# --- Main driver ---

def _build_days_data(city, era5_nc, cfg, test_dates):
    grid, landuse = _load_ship_landuse(city)
    city_mask = landuse["city_mask"].values
    days = []
    for d in test_dates:
        sim, obs = _sim_and_obs_pair(city, era5_nc, cfg, d, grid, landuse)
        v = _valid_cells_2d(sim, obs, city_mask)
        days.append((sim, obs, v))
    return grid, landuse, days


def main():
    print(f"[boot] BLOCK_SIDE={BLOCK_SIDE} cells (~{BLOCK_SIDE*500/1000:.1f} km),  "
          f"B={N_BOOT},  seed={SEED}")

    austin_test = ["2024-08-21", "2024-08-22"]
    phoenix_test = ["2024-07-22", "2024-07-24"]
    miami_test = ["2024-04-16", "2024-04-19"]

    base_48h = SimConfig(duration_hours=48.0)

    # ---- Configurations for tab:bootstrap_ci ----
    configs = {
        "single_stage_legacy_water": {
            "cfg": load_calibrated_config(OUT / "calibrated_config.json", base_48h),
            "city": AUSTIN, "era5": None, "test": austin_test,
            "label": "Single-stage, legacy water",
        },
        "two_stage_synthetic": {
            "cfg": replace(load_calibrated_config(OUT / "calibrated_config_two_stage.json", base_48h),
                           use_pm_et=False),
            "city": AUSTIN, "era5": None, "test": austin_test,
            "label": "Two-stage, synthetic forcing",
        },
        "ship_two_stage_era5": {
            "cfg": replace(load_calibrated_config(OUT / "calibrated_config_era5.json", base_48h),
                           use_pm_et=False),
            "city": AUSTIN, "era5": _ERA5_AUSTIN, "test": austin_test,
            "label": "Ship (two-stage + ERA5)",
        },
        "pm_era5_K3": {
            "cfg": replace(load_calibrated_config(OUT / "calibrated_config_pm.json", base_48h),
                           use_pm_et=True),
            "city": AUSTIN, "era5": _ERA5_AUSTIN, "test": austin_test,
            "label": "PM + ERA5 (canonical, K=3)",
        },
        "phoenix_zero_shot": {
            "cfg": replace(load_calibrated_config(OUT / "calibrated_config_era5.json", base_48h),
                           use_pm_et=False),
            "city": PHOENIX, "era5": _ERA5_PHOENIX, "test": phoenix_test,
            "label": "Phoenix, zero-shot",
        },
        "miami_zero_shot": {
            "cfg": replace(load_calibrated_config(OUT / "calibrated_config_era5.json", base_48h),
                           use_pm_et=False),
            "city": MIAMI, "era5": _ERA5_MIAMI, "test": miami_test,
            "label": "Miami, zero-shot",
        },
    }

    results = {}
    # Cache city-level (grid, landuse, days_data) so paired analyses reuse them.
    per_config_days = {}
    for key, spec in configs.items():
        print(f"\n[boot] === {spec['label']} ===")
        grid, landuse, days = _build_days_data(
            spec["city"], spec["era5"], spec["cfg"], spec["test"],
        )
        per_config_days[key] = (grid, landuse, days)
        for i, (_, _, v) in enumerate(days):
            print(f"       day {spec['test'][i]}: {int(v.sum())} jointly-valid cells")
        stats = bootstrap_r_rmse(days)
        r_ci = stats["r_ci"]; rmse_ci = stats["rmse_ci"]
        print(f"       r  = {stats['r_hat']:+.3f}  [{r_ci[0]:+.3f}, {r_ci[1]:+.3f}]")
        print(f"       RMSE = {stats['rmse_hat']:.3f}  [{rmse_ci[0]:.3f}, {rmse_ci[1]:.3f}]")
        results[key] = {"label": spec["label"], **stats}

    # ---- Paired-difference CI: ship − two_stage_synthetic ----
    print(f"\n[boot] === Paired Δr: ship (ERA5) − two-stage (synthetic) ===")
    _, _, days_ship = per_config_days["ship_two_stage_era5"]
    _, _, days_syn = per_config_days["two_stage_synthetic"]
    # Ensure the two share grid shape and valid masks (same city, same test days).
    # Take the intersection of jointly-valid masks so cells line up.
    aligned_ship, aligned_syn = [], []
    for (s1, o1, v1), (s2, o2, v2) in zip(days_ship, days_syn):
        v = v1 & v2
        aligned_ship.append((s1, o1, v))
        aligned_syn.append((s2, o2, v))
    paired = paired_delta_r(aligned_ship, aligned_syn)
    print(f"       Δr_hat = {paired['delta_hat']:+.4f}  "
          f"95 % CI = [{paired['delta_ci'][0]:+.4f}, {paired['delta_ci'][1]:+.4f}]")
    zero_in_ci = paired['delta_ci'][0] <= 0 <= paired['delta_ci'][1]
    print(f"       zero in CI? {zero_in_ci}   -> "
          f"{'CIs overlap, Δr indistinguishable from zero' if zero_in_ci else 'Δr distinguishable from zero'}")
    results["paired_delta_ship_minus_syn"] = paired

    # ---- OAT sensitivity on ship config ----
    print(f"\n[boot] === OAT sensitivity (ship config, Austin test days) ===")
    ship_cfg = configs["ship_two_stage_era5"]["cfg"]
    grid_a, landuse_a, _ = per_config_days["ship_two_stage_era5"]
    oat = oat_sensitivity(ship_cfg, AUSTIN, _ERA5_AUSTIN, austin_test, grid_a, landuse_a)
    print(f"       ship baseline (pooled): r = {oat['r_base']:+.3f}  RMSE = {oat['rmse_base']:.3f}")
    print(f"       {'coef':<26} {'-25%':>10} {'-10%':>10} {'+10%':>10} {'+25%':>10}  {'rank'}")
    for row in oat["rows"]:
        print(f"       {row['coef']:<26} "
              f"{row['delta_r_-25pct']:>+10.4f} "
              f"{row['delta_r_-10pct']:>+10.4f} "
              f"{row['delta_r_+10pct']:>+10.4f} "
              f"{row['delta_r_+25pct']:>+10.4f}  "
              f"{row['rank']}")
    results["oat_sensitivity"] = oat

    # ---- Persist ----
    out_path = OUT / "bootstrap_uncertainty.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[boot] wrote {out_path}")

    # ---- Paper-ready tables ----
    print("\n" + "=" * 78)
    print("PAPER TABLE: tab:bootstrap_ci")
    print("=" * 78)
    order = ["single_stage_legacy_water", "two_stage_synthetic",
             "ship_two_stage_era5", "pm_era5_K3",
             "phoenix_zero_shot", "miami_zero_shot"]
    for k in order:
        s = results[k]
        r_lo, r_hi = s["r_ci"]
        rmse_lo, rmse_hi = s["rmse_ci"]
        rmse_str = f"${s['rmse_hat']:.2f}$ [${rmse_lo:.2f}$, ${rmse_hi:.2f}$]"
        print(f"  {s['label']:<35} & ${s['r_hat']:+.3f}$ [${r_lo:+.3f}$, ${r_hi:+.3f}$] "
              f"& {rmse_str} \\\\")

    print("\n" + "=" * 78)
    print("PAPER TABLE: tab:sensitivity")
    print("=" * 78)
    coef_tex = {
        "absorption_impervious": r"$\alpha_{\mathrm{imp}}$",
        "absorption_vegetation": r"$\alpha_{\mathrm{veg}}$",
        "absorption_water": r"$\alpha_{\mathrm{wat}}$",
        "et_coeff": r"$k_{\mathrm{ET}}$",
    }
    for row in oat["rows"]:
        print(f"  {coef_tex[row['coef']]:<28} & "
              f"${row['delta_r_-25pct']:+.3f}$ & "
              f"${row['delta_r_+25pct']:+.3f}$ & --- & {row['rank']} \\\\")

    print("\n" + "=" * 78)
    print(f"FRAMING SIGNAL (Section 6): paired Δr(ship − synthetic) = "
          f"{paired['delta_hat']:+.4f}, 95% CI [{paired['delta_ci'][0]:+.4f}, "
          f"{paired['delta_ci'][1]:+.4f}]")
    if zero_in_ci:
        print("  -> keep 'neutral' framing (Δr indistinguishable from zero)")
    else:
        print("  -> switch framing to 'marginally positive but small' "
              "(Δr distinguishable from zero)")
    print("=" * 78)


if __name__ == "__main__":
    main()
