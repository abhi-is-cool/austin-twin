"""Build a real land-use Dataset from OpenStreetMap features.

Drop-in replacement for `synthetic.generate_landuse`: returns the same channels
(impervious_frac, vegetation_frac, water_mask, city_mask) so downstream code
(simulator, counterfactuals) works unchanged.

OSM queries hit the Overpass API and are slow (minutes for a city) — every
layer is cached as GeoParquet under `data/raw/osm/` after the first fetch.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio.features
import xarray as xr
from rasterio.transform import from_origin

from .grid import AUSTIN_CRS, GEO_CRS, CityGrid


# Tag sets per layer. Roads list is restricted to drivable + residential so we
# don't pull every footpath (massive feature count, negligible impervious area).
_LAYER_TAGS: dict[str, dict] = {
    "buildings": {"building": True},
    "roads": {
        "highway": [
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "unclassified", "residential", "motorway_link", "trunk_link",
            "primary_link", "secondary_link", "tertiary_link",
        ]
    },
    "water": {"natural": "water", "waterway": ["river", "stream", "canal"]},
    "parks": {
        "leisure": ["park", "garden", "nature_reserve"],
        "landuse": ["grass", "forest", "meadow", "recreation_ground", "cemetery"],
        "natural": ["wood", "scrub", "grassland"],
    },
}

# Approximate road half-widths (meters) used to buffer line geometries into
# polygons. Real road widths vary; these are a coarse proxy.
_ROAD_HALF_WIDTH_M = {
    "motorway": 12.0, "trunk": 10.0, "primary": 8.0, "secondary": 7.0,
    "tertiary": 6.0, "unclassified": 5.0, "residential": 5.0,
    "motorway_link": 6.0, "trunk_link": 5.0, "primary_link": 5.0,
    "secondary_link": 4.0, "tertiary_link": 4.0,
}
_DEFAULT_ROAD_HALF_WIDTH = 5.0


def _cache_path(cache_dir: Path, layer: str) -> Path:
    return cache_dir / f"{layer}.gpkg"


def fetch_layer(
    layer: str,
    boundary: gpd.GeoDataFrame,
    cache_dir: Path,
) -> gpd.GeoDataFrame:
    """Fetch one OSM layer, reproject to UTM, and cache as GeoPackage."""
    path = _cache_path(cache_dir, layer)
    if path.exists():
        return gpd.read_file(path).to_crs(AUSTIN_CRS)

    boundary_wgs84 = boundary.to_crs(GEO_CRS)
    poly_wgs84 = boundary_wgs84.geometry.union_all()
    tags = _LAYER_TAGS[layer]

    print(f"  fetching OSM layer '{layer}' (this may take 30s-5min)...")
    gdf = ox.features_from_polygon(poly_wgs84, tags=tags)
    if gdf.empty:
        gdf = gpd.GeoDataFrame(geometry=[], crs=GEO_CRS)
    gdf = gdf.to_crs(AUSTIN_CRS)

    # Discard rows with null/invalid geometries.
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].copy()

    if layer == "roads":
        # Buffer line geometries by their per-highway-class half-width.
        def _buffer(row):
            hw_class = row.get("highway")
            if isinstance(hw_class, list):
                hw_class = hw_class[0]
            half_w = _ROAD_HALF_WIDTH_M.get(hw_class, _DEFAULT_ROAD_HALF_WIDTH)
            return row.geometry.buffer(half_w)
        gdf["geometry"] = gdf.apply(_buffer, axis=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    # GeoPackage can't carry list-valued OSM attribute columns; drop them so
    # the only column we need (geometry) plus simple attributes survives.
    keep_cols = ["geometry"]
    if "highway" in gdf.columns:
        gdf["highway"] = gdf["highway"].astype(str)
        keep_cols.append("highway")
    gdf = gdf[keep_cols]
    gdf.to_file(path, driver="GPKG", layer=layer)
    print(f"    cached {len(gdf):,} features -> {path.name}")
    return gdf


def rasterize_fraction(
    features: gpd.GeoDataFrame,
    grid: CityGrid,
    supersample: int = 10,
) -> np.ndarray:
    """Compute per-cell area fraction covered by `features`.

    Uses binary rasterization at `supersample * grid_resolution`, then
    block-averages back down. Accurate to ~1/supersample^2 per cell.
    """
    if features.empty:
        return np.zeros(grid.shape, dtype=np.float32)

    ny, nx = grid.shape
    sub_res = grid.resolution_m / supersample
    sub_ny, sub_nx = ny * supersample, nx * supersample

    # Affine transform: origin at top-left of the grid extent.
    x_left = float(grid.x[0]) - grid.resolution_m / 2.0
    y_top = float(grid.y[0]) + grid.resolution_m / 2.0
    transform = from_origin(x_left, y_top, sub_res, sub_res)

    geoms = features.geometry.values
    raster = rasterio.features.rasterize(
        shapes=geoms,
        out_shape=(sub_ny, sub_nx),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=False,
    )
    # Block-average from (sub_ny, sub_nx) -> (ny, nx).
    fraction = raster.reshape(ny, supersample, nx, supersample).mean(axis=(1, 3))
    return fraction.astype(np.float32)


def build_osm_landuse(
    boundary: gpd.GeoDataFrame,
    grid: CityGrid,
    cache_dir: Path,
) -> xr.Dataset:
    """Construct the same Dataset schema as `synthetic.generate_landuse`."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[osm] fetching layers (first run is slow; later runs hit cache)...")
    buildings = fetch_layer("buildings", boundary, cache_dir)
    roads = fetch_layer("roads", boundary, cache_dir)
    water = fetch_layer("water", boundary, cache_dir)
    parks = fetch_layer("parks", boundary, cache_dir)

    print("[osm] rasterizing to grid...")
    building_frac = rasterize_fraction(buildings, grid)
    road_frac = rasterize_fraction(roads, grid)
    water_frac = rasterize_fraction(water, grid)
    park_frac = rasterize_fraction(parks, grid)

    # Channel composition:
    #   impervious = buildings + roads (capped at 1)
    #   water_mask = 1 where water fraction > 0.3 (a cell is "water" if >30% wet)
    #   vegetation = park_frac, plus residual land that's not impervious or water
    #                weighted at 0.5 (OSM under-tags vegetation outside parks)
    impervious = np.clip(building_frac + road_frac, 0.0, 1.0)
    water_mask = (water_frac > 0.3).astype(np.float32)
    # Where water dominates, zero out impervious so the channels stay coherent.
    impervious = np.where(water_mask > 0, 0.0, impervious)

    residual = np.clip(1.0 - impervious - water_mask, 0.0, 1.0)
    vegetation = np.clip(park_frac + 0.5 * residual, 0.0, 1.0)
    # Don't let veg + impervious + water exceed 1.
    overshoot = np.maximum(vegetation + impervious + water_mask - 1.0, 0.0)
    vegetation = np.clip(vegetation - overshoot, 0.0, 1.0)

    # Mask outside city.
    inside = grid.mask
    impervious = np.where(inside, impervious, 0.0).astype(np.float32)
    vegetation = np.where(inside, vegetation, 0.0).astype(np.float32)
    water_mask = np.where(inside, water_mask, 0.0).astype(np.float32)

    return xr.Dataset(
        data_vars={
            "impervious_frac": (("y", "x"), impervious),
            "vegetation_frac": (("y", "x"), vegetation),
            "water_mask": (("y", "x"), water_mask),
            "city_mask": (("y", "x"), inside),
        },
        coords={"x": grid.x, "y": grid.y},
        attrs={"source": "osm", "cache_dir": str(cache_dir)},
    )
