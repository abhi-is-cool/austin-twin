"""Spatially targeted intervention analysis.

Partition the city into zones, add canopy to one zone at a time, and measure
how much each zone's intervention cools the city. The output answers:

  "If we can only plant trees in one neighborhood, where do we get the most
   cooling per acre converted?"

Three policy-relevant metrics per zone:
  - local_mean_dt   : avg ΔT inside the zone
  - citywide_mean_dt: avg ΔT across all city cells (captures diffusion spillover)
  - efficiency      : total degree-m² of cooling per m² of impervious converted
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import xarray as xr
from tqdm import tqdm

from .grid import CityGrid
from .simulator import SimConfig, SimResult, run

CANOPY_DELTA = 0.20  # the fractional vegetation increase to apply per zone


@dataclass(frozen=True)
class Zone:
    label: str
    cell_mask: np.ndarray  # bool, grid.shape — True where zone covers a city cell

    @property
    def n_cells(self) -> int:
        return int(self.cell_mask.sum())


@dataclass
class ZoneMetrics:
    zone: Zone
    delta_t_mean: np.ndarray  # shape grid.shape — time-averaged ΔT per cell
    area_converted_m2: float
    local_mean_dt: float
    citywide_mean_dt: float
    total_cooling_degc_m2: float  # integral of -ΔT × cell_area over city (positive = cooling)
    efficiency: float  # total_cooling_degc_m2 / area_converted_m2


def make_grid_zones(grid: CityGrid, n_x: int = 6, n_y: int = 5, min_cells: int = 6) -> list[Zone]:
    """Tile the city with a regular n_y x n_x grid of zones.

    Only returns zones that contain at least `min_cells` city cells; this drops
    zones that fall mostly outside Austin's irregular boundary.
    """
    ny, nx = grid.shape
    inside = grid.mask

    # Define zone edges from the row/column indices that contain ANY inside-city cell,
    # so we don't waste zones on the empty border around the boundary bbox.
    rows_with_city = np.where(inside.any(axis=1))[0]
    cols_with_city = np.where(inside.any(axis=0))[0]
    r_lo, r_hi = int(rows_with_city.min()), int(rows_with_city.max()) + 1
    c_lo, c_hi = int(cols_with_city.min()), int(cols_with_city.max()) + 1

    row_edges = np.linspace(r_lo, r_hi, n_y + 1).astype(int)
    col_edges = np.linspace(c_lo, c_hi, n_x + 1).astype(int)

    zones: list[Zone] = []
    row_labels = list(string.ascii_uppercase[:n_y])  # A..E for n_y=5
    for i in range(n_y):
        for j in range(n_x):
            mask = np.zeros((ny, nx), dtype=bool)
            mask[row_edges[i]:row_edges[i + 1], col_edges[j]:col_edges[j + 1]] = True
            mask &= inside
            if mask.sum() < min_cells:
                continue
            label = f"{row_labels[i]}{j + 1}"
            zones.append(Zone(label=label, cell_mask=mask))
    return zones


def _apply_canopy_in_zone(landuse: xr.Dataset, zone: Zone, delta: float = CANOPY_DELTA) -> xr.Dataset:
    """Bump vegetation by `delta` inside `zone`; reduce impervious to compensate."""
    out = landuse.copy(deep=True)
    veg_old = out["vegetation_frac"].values
    imp_old = out["impervious_frac"].values
    veg_new = veg_old.copy()
    veg_new[zone.cell_mask] = np.clip(veg_old[zone.cell_mask] + delta, 0.0, 1.0)
    actual_delta = veg_new - veg_old  # clipped delta per cell
    imp_new = np.clip(imp_old - actual_delta, 0.0, 1.0)
    out["vegetation_frac"].values[...] = veg_new
    out["impervious_frac"].values[...] = imp_new
    return out


def _zone_area_converted_m2(landuse: xr.Dataset, zone: Zone, grid: CityGrid, delta: float = CANOPY_DELTA) -> float:
    veg_old = landuse["vegetation_frac"].values[zone.cell_mask]
    actual_delta = np.minimum(delta, 1.0 - veg_old)
    cell_area = grid.resolution_m ** 2
    return float(np.sum(actual_delta) * cell_area)


def evaluate_zones(
    landuse: xr.Dataset,
    grid: CityGrid,
    zones: Iterable[Zone],
    baseline_result: SimResult,
    config: SimConfig | None = None,
) -> list[ZoneMetrics]:
    """For each zone, run the canopy-perturbed simulation and compute metrics."""
    cfg = config or SimConfig()
    base_T = baseline_result.temperature  # (T, y, x)
    city = landuse["city_mask"].values
    cell_area = grid.resolution_m ** 2
    n_city_cells = int(city.sum())

    metrics: list[ZoneMetrics] = []
    for z in tqdm(list(zones), desc="zone sensitivities"):
        perturbed = _apply_canopy_in_zone(landuse, z)
        result = run(perturbed, cfg)
        dt = result.temperature - base_T  # negative = cooling
        dt_mean = np.nanmean(dt, axis=0)  # shape grid.shape

        # Local ΔT — mean ΔT in zone cells (over time and over cells).
        local_mean = float(np.nanmean(dt_mean[z.cell_mask]))

        # Citywide ΔT.
        citywide_mean = float(np.nanmean(dt_mean[city]))

        # Total cooling in degree·m². Negative ΔT contributes positively to cooling.
        total_cooling = float(-np.nansum(dt_mean[city]) * cell_area)

        area_conv = _zone_area_converted_m2(landuse, z, grid)
        efficiency = total_cooling / area_conv if area_conv > 0 else 0.0

        metrics.append(ZoneMetrics(
            zone=z,
            delta_t_mean=dt_mean,
            area_converted_m2=area_conv,
            local_mean_dt=local_mean,
            citywide_mean_dt=citywide_mean,
            total_cooling_degc_m2=total_cooling,
            efficiency=efficiency,
        ))

    return metrics


def rank_zones(metrics: list[ZoneMetrics]) -> list[ZoneMetrics]:
    """Sort by efficiency (cooling per m² of conversion), descending."""
    return sorted(metrics, key=lambda m: m.efficiency, reverse=True)


def format_ranking(metrics: list[ZoneMetrics]) -> str:
    """Pretty-print the ranking as a table."""
    ranked = rank_zones(metrics)
    lines = [
        f"{'rank':<5} {'zone':<6} {'cells':>6} {'area_conv':>14} "
        f"{'local ΔT':>10} {'city ΔT':>10} {'efficiency':>16}",
        f"{'':<5} {'':<6} {'':>6} {'(hectare)':>14} {'(°C)':>10} {'(°C)':>10} "
        f"{'(°C·m²/m²)':>16}",
        "-" * 78,
    ]
    for i, m in enumerate(ranked, start=1):
        lines.append(
            f"{i:<5} {m.zone.label:<6} {m.zone.n_cells:>6d} "
            f"{m.area_converted_m2 / 1e4:>14,.1f} "
            f"{m.local_mean_dt:>+10.3f} {m.citywide_mean_dt:>+10.4f} "
            f"{m.efficiency:>16.3e}"
        )
    return "\n".join(lines)
