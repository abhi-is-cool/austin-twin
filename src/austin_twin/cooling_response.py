"""Extract simulator outputs in the form the UFOR literature reports them.

Three quantities, all derivable from short canopy-perturbation experiments:

  - canopy_response_curve: city-mean ΔT as a function of fractional canopy
    increase. The slope (°C per +10% canopy) is the most directly comparable
    quantity to published meta-analyses (e.g., Bowler 2010, Park 2017, Zardo
    2017, Wang 2018). We also decompose into mean ΔT inside the cells that
    were actually planted ("local cooling") and outside them ("spillover via
    diffusion").

  - patch_distance_decay: drop a single circular max-canopy patch in central
    Austin, measure |ΔT| as a function of distance from the patch center,
    then fit an exponential cooling extent λ. Literature typically reports
    100-300 m as the half-cooling distance for substantial canopy patches.

  - tabulated_summary: bundles the numbers into one row a reviewer can drop
    next to published bands.

All metrics are computed from the second-day peak-heat frame of the simulator,
matching the validation protocol used elsewhere in the project.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from .simulator import SimConfig, SimResult, run


@dataclass
class CanopyResponseCurve:
    deltas: list[float]
    citywide_mean_dt: list[float]
    planted_mean_dt: list[float]
    spillover_mean_dt: list[float]
    citywide_slope_per_10pct: float
    planted_slope_per_10pct: float


@dataclass
class DistanceDecay:
    distances_m: np.ndarray
    dt_mean: np.ndarray
    dt_p5: np.ndarray
    dt_p95: np.ndarray
    peak_dt_c: float            # mean cooling magnitude at the patch (d ≈ 0)
    decay_length_m: float       # e-folding scale of |ΔT| vs distance
    half_cooling_m: float       # distance at which |ΔT| drops to half of peak
    n_bins_fit: int


@dataclass
class CoolingResponseReport:
    response_curve: CanopyResponseCurve
    distance_decay: DistanceDecay
    config: SimConfig = field(repr=False)


# ---------- internal helpers ----------

def _peak_frame(result: SimResult) -> np.ndarray:
    """Return the citywide-mean-hottest frame in the second simulated day."""
    T = result.temperature
    times = result.times_hours
    second = times >= 24.0
    means = np.array([
        float(np.nanmean(T[i])) if np.isfinite(T[i]).any() else -np.inf
        for i in range(T.shape[0])
    ])
    return T[int(np.argmax(np.where(second, means, -np.inf)))]


def _city_centroid(landuse: xr.Dataset) -> tuple[float, float]:
    mask = landuse["city_mask"].values
    xs = landuse["x"].values
    ys = landuse["y"].values
    cx = float(np.mean(xs[mask.any(axis=0)]))
    cy = float(np.mean(ys[mask.any(axis=1)]))
    return cx, cy


def _apply_canopy_global(landuse: xr.Dataset, delta: float) -> xr.Dataset:
    """Add `delta` to vegetation_frac citywide, debit from impervious."""
    out = landuse.copy(deep=True)
    veg = out["vegetation_frac"].values
    imp = out["impervious_frac"].values
    veg_new = np.clip(veg + delta, 0.0, 1.0)
    actual = veg_new - veg
    imp_new = np.clip(imp - actual, 0.0, 1.0)
    out["vegetation_frac"].values[...] = veg_new
    out["impervious_frac"].values[...] = imp_new
    return out


def _apply_canopy_patch(
    landuse: xr.Dataset,
    cx: float, cy: float,
    radius_m: float,
    target_veg: float,
) -> tuple[xr.Dataset, np.ndarray]:
    """Inside a radius around (cx, cy), raise vegetation_frac to at least
    `target_veg`. Returns (new_dataset, patch_mask)."""
    out = landuse.copy(deep=True)
    xx, yy = np.meshgrid(out["x"].values, out["y"].values)
    patch = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius_m**2
    veg = out["vegetation_frac"].values
    imp = out["impervious_frac"].values
    veg_new = veg.copy()
    veg_new[patch] = np.maximum(veg[patch], target_veg)
    actual = veg_new - veg
    imp_new = np.clip(imp - actual, 0.0, 1.0)
    out["vegetation_frac"].values[...] = veg_new
    out["impervious_frac"].values[...] = imp_new
    return out, patch


# ---------- main analyses ----------

def canopy_response(
    landuse: xr.Dataset,
    config: SimConfig,
    deltas: list[float] | None = None,
    baseline_T_peak: np.ndarray | None = None,
) -> CanopyResponseCurve:
    """Citywide canopy sweep at four canopy increments."""
    if deltas is None:
        deltas = [0.05, 0.10, 0.20, 0.40]
    city = landuse["city_mask"].values
    if baseline_T_peak is None:
        baseline_T_peak = _peak_frame(run(landuse, config))

    citywide: list[float] = []
    planted: list[float] = []
    spillover: list[float] = []
    for delta in deltas:
        perturbed = _apply_canopy_global(landuse, delta)
        T_peak = _peak_frame(run(perturbed, config))
        dT = T_peak - baseline_T_peak
        veg_change = (perturbed["vegetation_frac"].values
                      - landuse["vegetation_frac"].values)
        planted_mask = (veg_change > 0.5 * delta) & city
        spillover_mask = (veg_change < 0.1 * delta) & city
        citywide.append(float(np.nanmean(dT[city])))
        planted.append(float(np.nanmean(dT[planted_mask])) if planted_mask.any() else 0.0)
        spillover.append(float(np.nanmean(dT[spillover_mask])) if spillover_mask.any() else 0.0)

    # Linear fit: ΔT = slope * delta. Multiply by 0.1 to get °C per +10% canopy.
    deltas_arr = np.array(deltas)
    citywide_slope = float(np.polyfit(deltas_arr, citywide, 1)[0]) * 0.1
    planted_slope = float(np.polyfit(deltas_arr, planted, 1)[0]) * 0.1
    return CanopyResponseCurve(
        deltas=list(deltas),
        citywide_mean_dt=citywide,
        planted_mean_dt=planted,
        spillover_mean_dt=spillover,
        citywide_slope_per_10pct=citywide_slope,
        planted_slope_per_10pct=planted_slope,
    )


def patch_distance_decay(
    landuse: xr.Dataset,
    config: SimConfig,
    patch_radius_m: float = 1000.0,
    target_veg: float = 1.0,
    bin_width_m: float = 250.0,
    max_distance_m: float = 8000.0,
    baseline_T_peak: np.ndarray | None = None,
) -> DistanceDecay:
    """Single circular max-canopy patch in central Austin → cooling vs distance."""
    if baseline_T_peak is None:
        baseline_T_peak = _peak_frame(run(landuse, config))

    cx, cy = _city_centroid(landuse)
    perturbed, _ = _apply_canopy_patch(landuse, cx, cy, patch_radius_m, target_veg)
    T_peak = _peak_frame(run(perturbed, config))
    dT = T_peak - baseline_T_peak  # negative inside the cooling footprint

    xx, yy = np.meshgrid(landuse["x"].values, landuse["y"].values)
    distances = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    city = landuse["city_mask"].values
    valid = city & np.isfinite(dT)

    edges = np.arange(0.0, max_distance_m + bin_width_m, bin_width_m)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dt_mean = np.full(centers.shape, np.nan)
    dt_p5 = np.full(centers.shape, np.nan)
    dt_p95 = np.full(centers.shape, np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = valid & (distances >= lo) & (distances < hi)
        if m.any():
            vals = dT[m]
            dt_mean[i] = float(np.mean(vals))
            dt_p5[i] = float(np.percentile(vals, 5))
            dt_p95[i] = float(np.percentile(vals, 95))

    # Fit exponential decay |ΔT|(d) = A exp(-d/λ) on bins where cooling is large
    # enough to be unambiguous (> 0.05 °C below noise floor of the simulator).
    cooling_mag = -dt_mean
    fit_mask = np.isfinite(cooling_mag) & (cooling_mag > 0.05)
    if fit_mask.sum() >= 3:
        d = centers[fit_mask]
        magn = cooling_mag[fit_mask]
        slope, intercept = np.polyfit(d, np.log(magn), 1)
        lam = -1.0 / slope if slope < 0 else float("inf")
        A = float(np.exp(intercept))
        half = lam * np.log(2.0) if np.isfinite(lam) else float("nan")
        peak = float(cooling_mag[0]) if np.isfinite(cooling_mag[0]) else A
    else:
        lam = float("nan"); half = float("nan"); peak = float("nan")

    return DistanceDecay(
        distances_m=centers,
        dt_mean=dt_mean,
        dt_p5=dt_p5,
        dt_p95=dt_p95,
        peak_dt_c=peak,
        decay_length_m=float(lam),
        half_cooling_m=float(half),
        n_bins_fit=int(fit_mask.sum()),
    )


def diffusion_decay_sweep(
    landuse: xr.Dataset,
    base_config: SimConfig,
    diffusion_values: list[float],
    patch_radius_m: float = 1000.0,
    target_veg: float = 1.0,
) -> list[dict]:
    """For each D, rebuild a baseline and compute the patch half-cooling distance.

    Returns a list of {D, half_cooling_m, decay_length_m, peak_dt_c} dicts.
    Each D requires two simulator runs (baseline + perturbed); a CFL violation
    is caught and recorded as NaN so the sweep is robust to bad values.
    """
    from dataclasses import replace as _replace

    out: list[dict] = []
    for D in diffusion_values:
        cfg = _replace(base_config, diffusion_m2_s=float(D))
        try:
            baseline_T_peak = _peak_frame(run(landuse, cfg))
            decay = patch_distance_decay(
                landuse, cfg,
                patch_radius_m=patch_radius_m,
                target_veg=target_veg,
                baseline_T_peak=baseline_T_peak,
            )
            out.append(dict(
                D=float(D),
                half_cooling_m=float(decay.half_cooling_m),
                decay_length_m=float(decay.decay_length_m),
                peak_dt_c=float(decay.peak_dt_c),
                n_bins_fit=int(decay.n_bins_fit),
            ))
        except ValueError:
            out.append(dict(D=float(D), half_cooling_m=float("nan"),
                            decay_length_m=float("nan"),
                            peak_dt_c=float("nan"), n_bins_fit=0))
    return out


def find_D_for_target_half_cooling(
    landuse: xr.Dataset,
    base_config: SimConfig,
    target_half_cooling_m: float = 200.0,
    D_range: tuple[float, float] = (1.0, 50.0),
    n: int = 12,
) -> tuple[float, list[dict]]:
    """Sweep D log-uniformly and return the value whose half-cooling distance
    is closest to `target_half_cooling_m`. Also returns the full sweep for
    plotting.

    Linearly interpolates between bracketing D values to refine the estimate
    when the target falls between grid points.
    """
    Ds = list(np.geomspace(D_range[0], D_range[1], n))
    sweep = diffusion_decay_sweep(landuse, base_config, Ds)
    half = np.array([s["half_cooling_m"] for s in sweep])
    valid = np.isfinite(half)
    if not valid.any():
        raise RuntimeError("No valid half-cooling values across diffusion sweep")

    D_arr = np.array([s["D"] for s in sweep])
    # Restrict to valid points and interpolate in log-D space.
    log_D = np.log(D_arr[valid])
    hc = half[valid]
    order = np.argsort(log_D)
    log_D = log_D[order]; hc = hc[order]
    # The half-cooling distance generally increases with D; we look for the
    # crossing of target_half_cooling_m.
    if hc[0] >= target_half_cooling_m:
        return float(np.exp(log_D[0])), sweep  # smallest D already above target
    if hc[-1] <= target_half_cooling_m:
        return float(np.exp(log_D[-1])), sweep  # largest D still below target
    # Find bracketing pair.
    above = np.where(hc >= target_half_cooling_m)[0][0]
    below = above - 1
    # Linear interp on log(D) vs hc.
    t = (target_half_cooling_m - hc[below]) / (hc[above] - hc[below])
    log_D_target = log_D[below] + t * (log_D[above] - log_D[below])
    return float(np.exp(log_D_target)), sweep


def run_full_analysis(landuse: xr.Dataset, config: SimConfig) -> CoolingResponseReport:
    """Run baseline once, then both analyses sharing that baseline."""
    baseline_T_peak = _peak_frame(run(landuse, config))
    response = canopy_response(landuse, config, baseline_T_peak=baseline_T_peak)
    decay = patch_distance_decay(landuse, config, baseline_T_peak=baseline_T_peak)
    return CoolingResponseReport(response_curve=response, distance_decay=decay, config=config)
