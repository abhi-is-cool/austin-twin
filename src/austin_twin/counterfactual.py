"""Counterfactual urban-planning scenarios.

A `Scenario` is a named function that perturbs the landuse Dataset.
`run_scenarios` runs the simulator on the baseline plus each scenario and
returns the full set of SimResults indexed by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import xarray as xr

from .simulator import SimConfig, SimResult, run

Perturbation = Callable[[xr.Dataset], xr.Dataset]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    apply: Perturbation


# ---------- helper geometry ----------

def _city_center(landuse: xr.Dataset) -> tuple[float, float]:
    mask = landuse["city_mask"].values
    xs = landuse["x"].values
    ys = landuse["y"].values
    cx = float(np.mean(xs[mask.any(axis=0)]))
    cy = float(np.mean(ys[mask.any(axis=1)]))
    return cx, cy


def _radial_mask(landuse: xr.Dataset, cx: float, cy: float, radius_m: float) -> np.ndarray:
    xx, yy = np.meshgrid(landuse["x"].values, landuse["y"].values)
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius_m**2


# ---------- scenarios ----------

def _baseline(landuse: xr.Dataset) -> xr.Dataset:
    return landuse


def _canopy_plus_20(landuse: xr.Dataset) -> xr.Dataset:
    """Increase vegetation fraction by 0.2 everywhere; convert from impervious."""
    out = landuse.copy(deep=True)
    veg_old = out["vegetation_frac"].values
    imp_old = out["impervious_frac"].values
    veg_new = np.clip(veg_old + 0.2, 0.0, 1.0)
    delta = veg_new - veg_old  # actual increase after clipping
    imp_new = np.clip(imp_old - delta, 0.0, 1.0)
    out["vegetation_frac"].values[...] = veg_new
    out["impervious_frac"].values[...] = imp_new
    return out


def _cool_roofs_downtown(landuse: xr.Dataset) -> xr.Dataset:
    """Apply 0.35 albedo boost in a 3 km radius around downtown."""
    out = landuse.copy(deep=True)
    cx, cy = _city_center(out)
    mask = _radial_mask(out, cx, cy, radius_m=3000.0)
    boost = np.zeros_like(out["impervious_frac"].values, dtype=np.float32)
    boost[mask] = 0.35
    out["albedo_boost"] = (("y", "x"), boost)
    return out


def _river_greenway(landuse: xr.Dataset) -> xr.Dataset:
    """Add a 1 km vegetation buffer along the synthetic Colorado River."""
    out = landuse.copy(deep=True)
    water = out["water_mask"].values
    if water.sum() == 0:
        return out
    # Dilate water mask by ~1 km using a separable rectangular kernel.
    dx = float(abs(out["x"].values[1] - out["x"].values[0]))
    radius_cells = max(1, int(round(1000.0 / dx)))
    dilated = _box_dilate(water > 0, radius_cells)
    buffer_only = dilated & ~(water > 0)
    veg = out["vegetation_frac"].values
    imp = out["impervious_frac"].values
    # In the buffer, raise veg to at least 0.7 and pull impervious down to compensate.
    target_veg = np.maximum(veg, 0.7)
    delta = np.where(buffer_only, target_veg - veg, 0.0)
    veg_new = veg + delta
    imp_new = np.clip(imp - delta, 0.0, 1.0)
    out["vegetation_frac"].values[...] = veg_new
    out["impervious_frac"].values[...] = imp_new
    return out


def _suburban_densification(landuse: xr.Dataset) -> xr.Dataset:
    """Control scenario: raise impervious fraction in the suburban ring."""
    out = landuse.copy(deep=True)
    cx, cy = _city_center(out)
    inner = _radial_mask(out, cx, cy, radius_m=3000.0)
    outer = _radial_mask(out, cx, cy, radius_m=10000.0)
    ring = outer & ~inner
    veg = out["vegetation_frac"].values
    imp = out["impervious_frac"].values
    bump = np.where(ring, 0.15, 0.0).astype(np.float32)
    imp_new = np.clip(imp + bump, 0.0, 1.0)
    delta = imp_new - imp
    veg_new = np.clip(veg - delta, 0.0, 1.0)
    out["impervious_frac"].values[...] = imp_new
    out["vegetation_frac"].values[...] = veg_new
    return out


def _box_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Boolean dilation with a (2r+1)x(2r+1) square structuring element."""
    out = mask.copy()
    for _ in range(radius):
        shifted = np.zeros_like(out)
        shifted[1:, :] |= out[:-1, :]
        shifted[:-1, :] |= out[1:, :]
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out = out | shifted
    return out


CANONICAL_SCENARIOS: list[Scenario] = [
    Scenario("baseline", "Unmodified synthetic land-use", _baseline),
    Scenario("canopy_plus_20", "+20% tree canopy citywide (converts impervious -> veg)", _canopy_plus_20),
    Scenario("cool_roofs_downtown", "0.35 albedo boost within 3 km of downtown", _cool_roofs_downtown),
    Scenario("river_greenway", "1 km vegetation buffer along the Colorado River", _river_greenway),
    Scenario("suburban_densification", "Control: +15% impervious in 3-10 km ring", _suburban_densification),
]


@dataclass
class ScenarioRun:
    scenario: Scenario
    landuse: xr.Dataset
    result: SimResult


def run_scenarios(
    baseline_landuse: xr.Dataset,
    scenarios: list[Scenario],
    config: SimConfig | None = None,
    forcing=None,
) -> dict[str, ScenarioRun]:
    """Run the simulator on the baseline plus each scenario.

    All scenarios share the same `forcing` (a `Forcing` object) so the
    ΔT between them isolates the land-use intervention rather than
    conflating it with weather variation. If `forcing` is None, each
    run builds a synthetic diurnal from `config` (legacy behavior).
    """
    cfg = config or SimConfig()
    runs: dict[str, ScenarioRun] = {}
    for sc in scenarios:
        lu = sc.apply(baseline_landuse)
        result = run(lu, cfg, forcing=forcing)
        runs[sc.name] = ScenarioRun(scenario=sc, landuse=lu, result=result)
    return runs


def summarize(runs: dict[str, ScenarioRun], baseline_name: str = "baseline") -> str:
    """Return a one-table string summary of cooling vs the baseline."""
    base = runs[baseline_name].result.temperature
    finite_base = np.isfinite(base).all(axis=0)
    lines = [
        f"{'scenario':<26} {'mean ΔT':>10} {'max cooling':>13} {'%cells<-1°C':>13}",
        "-" * 64,
    ]
    for name, r in runs.items():
        T = r.result.temperature
        dT = T - base
        valid_dT = dT[:, finite_base]
        mean_dT = float(np.nanmean(valid_dT))
        max_cool = float(np.nanmin(valid_dT))  # most negative = most cooling
        # Fraction of (cell, time) pairs cooled by more than 1 °C.
        frac_cooled = float(np.nanmean(valid_dT < -1.0)) * 100.0
        lines.append(
            f"{name:<26} {mean_dT:>+9.2f}°C {max_cool:>+12.2f}°C {frac_cooled:>12.1f}%"
        )
    return "\n".join(lines)
