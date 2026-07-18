"""Build a regular metric grid over a city-limits boundary.

Multi-city aware: the grid uses whatever UTM CRS the input boundary is
already in, so any northern-hemisphere city works. Austin-specific
helpers (`fetch_austin_boundary`, `AUSTIN_CRS`) are retained as
back-compat shims — new code should prefer `fetch_boundary(city)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import xarray as xr
from shapely.geometry import box

from .cities import AUSTIN, CityConfig

# EPSG:32614 = UTM zone 14N, covers Austin; kept as a public constant for
# legacy imports (e.g. viz.py, some scripts).
AUSTIN_CRS = AUSTIN.utm_crs
GEO_CRS = "EPSG:4326"


@dataclass
class CityGrid:
    """A regular metric grid clipped to a city boundary."""

    boundary: gpd.GeoDataFrame  # in its own UTM CRS
    resolution_m: float
    x: np.ndarray  # cell-center eastings, shape (nx,)
    y: np.ndarray  # cell-center northings, shape (ny,), decreasing (north-up)
    mask: np.ndarray  # bool, shape (ny, nx); True = inside city limits

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    @property
    def crs(self) -> str:
        return str(self.boundary.crs)

    def to_dataset(self) -> xr.Dataset:
        return xr.Dataset(
            data_vars={"city_mask": (("y", "x"), self.mask)},
            coords={"x": self.x, "y": self.y},
            attrs={"crs": self.crs, "resolution_m": self.resolution_m},
        )


def fetch_boundary(city: CityConfig, cache_path: Path | None = None) -> gpd.GeoDataFrame:
    """Pull a city-limits polygon from OSM and reproject to the city's UTM CRS."""
    if cache_path and cache_path.exists():
        return gpd.read_file(cache_path).to_crs(city.utm_crs)

    gdf = ox.geocode_to_gdf(city.osm_query)
    gdf = gdf.to_crs(city.utm_crs)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_crs(GEO_CRS).to_file(cache_path, driver="GeoJSON")
    return gdf


def fetch_austin_boundary(cache_path: Path | None = None) -> gpd.GeoDataFrame:
    """Back-compat shim for pre-multi-city code. Prefer `fetch_boundary(AUSTIN)`."""
    return fetch_boundary(AUSTIN, cache_path=cache_path)


def build_grid(boundary: gpd.GeoDataFrame, resolution_m: float = 500.0) -> CityGrid:
    """Construct cell-center coordinates and an inside-city mask.

    Uses whatever CRS the input `boundary` is in — must be a metric CRS
    (typically the city's UTM zone) for `resolution_m` to be meaningful.
    """
    minx, miny, maxx, maxy = boundary.total_bounds
    # Snap bounds outward to whole multiples of resolution for clean coords.
    minx = np.floor(minx / resolution_m) * resolution_m
    miny = np.floor(miny / resolution_m) * resolution_m
    maxx = np.ceil(maxx / resolution_m) * resolution_m
    maxy = np.ceil(maxy / resolution_m) * resolution_m

    x = np.arange(minx + resolution_m / 2, maxx, resolution_m)
    y = np.arange(maxy - resolution_m / 2, miny, -resolution_m)  # north-up

    # Mask by point-in-polygon at cell centers, using the boundary's own CRS.
    xx, yy = np.meshgrid(x, y)
    flat_pts = gpd.GeoSeries(gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=boundary.crs)
    inside = flat_pts.within(boundary.unary_union).to_numpy().reshape(xx.shape)

    return CityGrid(boundary=boundary, resolution_m=resolution_m, x=x, y=y, mask=inside)
