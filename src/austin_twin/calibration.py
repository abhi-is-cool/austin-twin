"""Calibrate simulator coefficients against MODIS LST.

To use a previously-calibrated config:

    from austin_twin.calibration import load_calibrated_config
    cfg = load_calibrated_config(Path("outputs/calibrated_config.json"))


We tune five coefficients that drive spatial UHI variation:
    absorption_impervious, absorption_vegetation, absorption_water,
    et_coeff, diffusion_m2_s
Coefficients that only shift the spatial mean (air_temp_c, diurnal_amplitude_c,
lw_coeff, cell_heat_capacity) are kept fixed because the objective is anomaly
RMSE — they cancel out under mean-subtraction.

Objective:
    f(coefs) = mean over training days of anomaly RMSE between simulator
               peak-heat frame (second simulated day) and MODIS LST_Day,
               with both fields anomaly-normalized by subtracting their
               citywide spatial mean.

Optimizer: scipy.optimize.differential_evolution (gradient-free, robust for
mixed-scale 5D problems with a smooth-ish objective).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
from scipy.optimize import differential_evolution

from typing import TYPE_CHECKING

from .simulator import SimConfig, run

if TYPE_CHECKING:
    from .forcing import Forcing

# Bounds for the 5 tunable coefficients (chosen to span physically plausible
# values; the upper bound on diffusion_m2_s keeps us safely under the CFL
# limit of 0.25 at dt=600 s, dx=500 m, which corresponds to D ~ 104).
COEF_NAMES = (
    "absorption_impervious",
    "absorption_vegetation",
    "absorption_water",
    "et_coeff",
    "diffusion_m2_s",
)
BOUNDS = [
    (0.80, 0.98),  # absorption_impervious
    (0.30, 0.70),  # absorption_vegetation
    # Water absorbs only 6-15 % of shortwave (highly reflective at high sun
    # angles). The legacy (0.05, 0.50) range exists in the single-stage config
    # to compensate for the proportional-damping water buffer, which the
    # current simulator no longer uses; with the shallow mixed-layer model
    # the physical band is enforced here.
    (0.04, 0.18),  # absorption_water
    (5.0, 60.0),   # et_coeff (W/m² per unit veg per K)
    (5.0, 100.0),  # diffusion_m2_s
]


@dataclass
class DayMetrics:
    date: str
    rmse: float
    pearson_r: float
    sim_spread: float
    obs_spread: float


@dataclass
class CalibrationResult:
    optimal_config: SimConfig
    optimal_coefs: dict[str, float]
    train_metrics_before: list[DayMetrics]
    train_metrics_after: list[DayMetrics]
    test_metrics_before: list[DayMetrics]
    test_metrics_after: list[DayMetrics]
    history: list[float]


def _config_from_vector(
    base: SimConfig,
    x: Sequence[float],
    free_names: Sequence[str] = COEF_NAMES,
) -> SimConfig:
    """Build a SimConfig by overriding `free_names` on `base` with values from `x`."""
    return replace(base, **{name: float(val) for name, val in zip(free_names, x)})


def _peak_frame(T: np.ndarray, times_h: np.ndarray) -> np.ndarray:
    """Return the citywide-mean-hottest frame in the second simulated day."""
    second_day = times_h >= 24.0
    means = np.array([
        float(np.nanmean(T[i])) if np.isfinite(T[i]).any() else -np.inf
        for i in range(T.shape[0])
    ])
    masked = np.where(second_day, means, -np.inf)
    return T[int(np.argmax(masked))]


def _anomaly_stats(sim: np.ndarray, obs: np.ndarray) -> tuple[float, float, float, float]:
    """Return (RMSE, Pearson r, sim_spread, obs_spread) on jointly-valid cells."""
    valid = np.isfinite(sim) & np.isfinite(obs)
    if not valid.any():
        return float("inf"), 0.0, 0.0, 0.0
    s = sim[valid] - np.mean(sim[valid])
    o = obs[valid] - np.mean(obs[valid])
    rmse = float(np.sqrt(np.mean((s - o) ** 2)))
    if np.std(s) > 0 and np.std(o) > 0:
        r = float(np.corrcoef(s, o)[0, 1])
    else:
        r = 0.0
    return rmse, r, float(s.max() - s.min()), float(o.max() - o.min())


def evaluate(
    landuse: xr.Dataset,
    config: SimConfig,
    modis_days: dict[str, np.ndarray],
    forcings: "dict[str, Forcing] | None" = None,
) -> list[DayMetrics]:
    """Score simulator peak-heat frame against each provided MODIS day.

    If `forcings` is given, the simulator is re-run per date with that
    date's real atmospheric forcing (from ERA5). If `forcings` is None,
    the simulator is run once with a synthetic diurnal and that same
    peak frame is compared to every MODIS day (legacy behavior).
    """
    out: list[DayMetrics] = []
    if forcings is None:
        result = run(landuse, config)
        T_peak = _peak_frame(result.temperature, result.times_hours)
        for date, lst in modis_days.items():
            rmse, r, ss, os_ = _anomaly_stats(T_peak, lst)
            out.append(DayMetrics(date=date, rmse=rmse, pearson_r=r,
                                  sim_spread=ss, obs_spread=os_))
        return out

    for date, lst in modis_days.items():
        if date not in forcings:
            raise KeyError(f"forcings dict missing MODIS date {date!r}")
        result = run(landuse, config, forcing=forcings[date])
        T_peak = _peak_frame(result.temperature, result.times_hours)
        rmse, r, ss, os_ = _anomaly_stats(T_peak, lst)
        out.append(DayMetrics(date=date, rmse=rmse, pearson_r=r,
                              sim_spread=ss, obs_spread=os_))
    return out


def _objective_factory(
    landuse: xr.Dataset,
    modis_train: dict[str, np.ndarray],
    base: SimConfig,
    free_names: Sequence[str] = COEF_NAMES,
    forcings: "dict[str, Forcing] | None" = None,
):
    def _f(x: np.ndarray) -> float:
        try:
            cfg = _config_from_vector(base, x, free_names=free_names)
            if forcings is None:
                result = run(landuse, cfg)
                T_peak = _peak_frame(result.temperature, result.times_hours)
                if not np.isfinite(T_peak).any():
                    return 1e3
                rmses = [_anomaly_stats(T_peak, lst)[0] for lst in modis_train.values()]
            else:
                rmses = []
                for date, lst in modis_train.items():
                    result = run(landuse, cfg, forcing=forcings[date])
                    T_peak = _peak_frame(result.temperature, result.times_hours)
                    if not np.isfinite(T_peak).any():
                        return 1e3
                    rmses.append(_anomaly_stats(T_peak, lst)[0])
        except ValueError:
            return 1e3
        return float(np.mean(rmses))
    return _f


def calibrate(
    landuse: xr.Dataset,
    modis_train: dict[str, np.ndarray],
    modis_test: dict[str, np.ndarray],
    base_config: SimConfig | None = None,
    fixed_coefs: dict[str, float] | None = None,
    forcings: "dict[str, Forcing] | None" = None,
    popsize: int = 8,
    maxiter: int = 12,
    seed: int = 0,
    verbose: bool = True,
) -> CalibrationResult:
    """Run differential-evolution calibration; report train/test metrics.

    If `fixed_coefs` is given, those coefficients are pinned to the supplied
    values on `base_config` and the optimizer only searches over the remaining
    free coefficients. This supports two-stage calibration: pin diffusion to a
    literature-derived value, then fit absorption/ET to MODIS.

    If `forcings` is given (dict[date_str -> Forcing]), each MODIS date is
    scored against its own simulator run driven by that date's atmospheric
    forcing (from ERA5). Otherwise the simulator runs once with a synthetic
    diurnal and the same peak frame is compared to all dates (legacy mode).
    """
    base = base_config or SimConfig()
    if fixed_coefs:
        unknown = set(fixed_coefs) - set(COEF_NAMES)
        if unknown:
            raise ValueError(f"fixed_coefs contains unknown names: {unknown}")
        base = replace(base, **{k: float(v) for k, v in fixed_coefs.items()})
        free_names = tuple(n for n in COEF_NAMES if n not in fixed_coefs)
        free_bounds = [b for n, b in zip(COEF_NAMES, BOUNDS) if n not in fixed_coefs]
    else:
        free_names = COEF_NAMES
        free_bounds = BOUNDS

    before_train = evaluate(landuse, base, modis_train, forcings=forcings)
    before_test = evaluate(landuse, base, modis_test, forcings=forcings)

    objective = _objective_factory(landuse, modis_train, base,
                                   free_names=free_names, forcings=forcings)
    history: list[float] = []

    def _callback(intermediate_result):  # scipy >= 1.15 passes an OptimizeResult-like object
        val = float(intermediate_result.fun)
        history.append(val)
        if verbose:
            print(f"  [calibrate] iter {len(history):2d}  RMSE = {val:.3f} °C")
        return False

    if verbose:
        free_str = ", ".join(free_names)
        print(f"[calibrate] popsize={popsize}, maxiter={maxiter}, "
              f"free coefs: [{free_str}], max evals ~ {popsize * len(free_bounds) * maxiter}")
    res = differential_evolution(
        objective,
        bounds=free_bounds,
        popsize=popsize,
        maxiter=maxiter,
        tol=0.005,
        seed=seed,
        polish=True,
        workers=1,  # sim is fast; serialization overhead would dominate
        callback=_callback,
        updating="deferred",
    )

    optimal = _config_from_vector(base, res.x, free_names=free_names)
    after_train = evaluate(landuse, optimal, modis_train, forcings=forcings)
    after_test = evaluate(landuse, optimal, modis_test, forcings=forcings)

    # Record optimal coefs for ALL names (fixed ones come from base).
    full_coefs = {name: float(getattr(optimal, name)) for name in COEF_NAMES}

    return CalibrationResult(
        optimal_config=optimal,
        optimal_coefs=full_coefs,
        train_metrics_before=before_train,
        train_metrics_after=after_train,
        test_metrics_before=before_test,
        test_metrics_after=after_test,
        history=history,
    )


def load_calibrated_config(
    path: Path,
    base_config: SimConfig | None = None,
) -> SimConfig:
    """Load a calibrated SimConfig from the JSON written by run_calibration.py.

    Only the five tuned coefficients are overridden; everything else inherits
    from `base_config` (defaults to fresh SimConfig).
    """
    base = base_config or SimConfig()
    data = json.loads(path.read_text())
    overrides = {name: float(data[name]) for name in COEF_NAMES if name in data}
    return replace(base, **overrides)


def format_metrics_table(metrics_before: list[DayMetrics], metrics_after: list[DayMetrics], label: str) -> str:
    """Side-by-side before/after metric table for one split (train or test)."""
    lines = [
        f"{label.upper():<10} {'date':<12} {'RMSE before':>12} {'RMSE after':>12} "
        f"{'r before':>10} {'r after':>10} {'spread before':>14} {'spread after':>14}",
        "-" * 96,
    ]
    for mb, ma in zip(metrics_before, metrics_after):
        assert mb.date == ma.date
        lines.append(
            f"{'':<10} {mb.date:<12} {mb.rmse:>11.3f}° {ma.rmse:>11.3f}° "
            f"{mb.pearson_r:>+10.3f} {ma.pearson_r:>+10.3f} "
            f"{mb.sim_spread:>13.2f}° {ma.sim_spread:>13.2f}°"
        )
    mean_rmse_before = float(np.mean([m.rmse for m in metrics_before]))
    mean_rmse_after = float(np.mean([m.rmse for m in metrics_after]))
    mean_r_before = float(np.mean([m.pearson_r for m in metrics_before]))
    mean_r_after = float(np.mean([m.pearson_r for m in metrics_after]))
    lines.append(
        f"{'':<10} {'MEAN':<12} {mean_rmse_before:>11.3f}° {mean_rmse_after:>11.3f}° "
        f"{mean_r_before:>+10.3f} {mean_r_after:>+10.3f}"
    )
    return "\n".join(lines)
