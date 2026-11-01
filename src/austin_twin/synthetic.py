"""Synthetic land-use generator for MVP simulation.

Produces three normalized channels: vegetation fraction, impervious fraction,
and a water mask. The pattern is hand-crafted to be qualitatively Austin-like
(downtown core, suburban ring, parks, the Colorado River) so the simulator
output is visually interpretable before real Landsat/OSM data is wired in.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .grid import CityGrid


def _gaussian_blob(yy: np.ndarray, xx: np.ndarray, cy: float, cx: float, sigma: float) -> np.ndarray:
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))


def generate_landuse(grid: CityGrid, seed: int = 0) -> xr.Dataset:
    """Generate synthetic land-use channels aligned to `grid`."""
    rng = np.random.default_rng(seed)
    ny, nx = grid.shape
    xx, yy = np.meshgrid(grid.x, grid.y)

    # City centroid in projected coords (downtown core).
    cx = float(np.mean(grid.x[grid.mask.any(axis=0)]))
    cy = float(np.mean(grid.y[grid.mask.any(axis=1)]))

    # Downtown: tight blob of high impervious / low vegetation.
    downtown = _gaussian_blob(yy, xx, cy, cx, sigma=2500.0)

    # Suburban ring: broader blob, moderate impervious.
    suburb = _gaussian_blob(yy, xx, cy, cx, sigma=8000.0) - downtown
    suburb = np.clip(suburb, 0, 1)

    # Parks: a few random low-impervious / high-veg patches.
    parks = np.zeros_like(downtown)
    for _ in range(8):
        py = rng.uniform(grid.y.min(), grid.y.max())
        px = rng.uniform(grid.x.min(), grid.x.max())
        parks += _gaussian_blob(yy, xx, py, px, sigma=rng.uniform(800, 1800))
    parks = np.clip(parks, 0, 1)

    # Colorado River: a sinusoidal band running roughly E-W through downtown.
    river_y_center = cy - 500.0
    river_amplitude = 1500.0
    river_wavelength = 12000.0
    river_thickness = 250.0  # half-width in meters
    river_curve = river_y_center + river_amplitude * np.sin(2 * np.pi * (xx - cx) / river_wavelength)
    water = (np.abs(yy - river_curve) < river_thickness).astype(np.float32)

    # Compose channels.
    impervious = np.clip(0.85 * downtown + 0.45 * suburb - 0.6 * parks, 0.05, 0.95)
    vegetation = np.clip(0.15 + 0.7 * parks + 0.3 * (1 - suburb) - 0.7 * downtown, 0.0, 1.0)

    # Water overrides everything.
    impervious = np.where(water > 0, 0.0, impervious)
    vegetation = np.where(water > 0, 0.0, vegetation)

    # Zero-out cells outside city limits so downstream math stays clean.
    mask = grid.mask
    impervious = np.where(mask, impervious, 0.0).astype(np.float32)
    vegetation = np.where(mask, vegetation, 0.0).astype(np.float32)
    water = np.where(mask, water, 0.0).astype(np.float32)

    return xr.Dataset(
        data_vars={
            "impervious_frac": (("y", "x"), impervious),
            "vegetation_frac": (("y", "x"), vegetation),
            "water_mask": (("y", "x"), water),
            "city_mask": (("y", "x"), mask),
        },
        coords={"x": grid.x, "y": grid.y},
        attrs={"source": "synthetic", "seed": seed},
    )
