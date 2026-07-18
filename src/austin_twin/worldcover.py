"""Fetch ESA WorldCover 10 m land cover and resample to the Austin grid.

WorldCover v200 (2021 epoch) is hosted as public COGs on AWS S3 in eu-central-1.
The tile covering Austin is `N30W099` (spans 30-33°N, 96-99°W).

Class codes used:
   10 Tree cover, 20 Shrubland, 30 Grassland, 40 Cropland, 50 Built-up,
   60 Bare/sparse vegetation, 80 Permanent water bodies, 90 Herbaceous wetland.

We aggregate from 10 m -> 500 m by counting class memberships per block.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.warp
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from .grid import AUSTIN_CRS, CityGrid

_WC_TILE_URL_TEMPLATE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
# Default Austin tile for back-compat with legacy callers.
_WC_TILE_URL = _WC_TILE_URL_TEMPLATE.format(tile="N30W099")

# Class -> human label (for diagnostics)
_WC_CLASSES = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "crop",
    50: "built",
    60: "bare",
    80: "water",
    90: "wetland",
}


def fetch_worldcover_for_grid(
    grid: CityGrid,
    cache_path: Path,
    tile: str = "N30W099",
) -> np.ndarray:
    """Return a 10 m class raster aligned with the grid (shape ny*50, nx*50).

    `tile` is the WorldCover v200 3° tile ID covering the grid (e.g.
    "N30W099" for Austin, "N33W114" for Phoenix, "N24W081" for Miami).
    Cached locally as a GeoTIFF (~30 MB per city).
    """
    if cache_path.exists():
        with rasterio.open(cache_path) as src:
            return src.read(1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    ny, nx = grid.shape
    sub = 50  # 500 m / 10 m
    sub_ny, sub_nx = ny * sub, nx * sub
    sub_res = grid.resolution_m / sub  # 10 m

    x_left = float(grid.x[0]) - grid.resolution_m / 2.0
    y_top = float(grid.y[0]) + grid.resolution_m / 2.0
    dst_transform = from_origin(x_left, y_top, sub_res, sub_res)

    tile_url = _WC_TILE_URL_TEMPLATE.format(tile=tile)
    print(f"[worldcover] reading remote tile {tile} (~30 MB) via HTTP range reads...")
    env_opts = dict(
        GDAL_HTTP_MULTIPLEX="YES",
        GDAL_HTTP_VERSION="2",
        VSI_CACHE="YES",
        CPL_VSIL_CURL_USE_HEAD="NO",
    )
    with rasterio.Env(**env_opts):
        with rasterio.open(f"/vsicurl/{tile_url}") as src:
            dst = np.zeros((sub_ny, sub_nx), dtype=np.uint8)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=grid.crs,
                resampling=Resampling.nearest,
                num_threads=4,
            )

    with rasterio.open(
        cache_path, "w",
        driver="GTiff",
        height=sub_ny, width=sub_nx,
        count=1, dtype=np.uint8,
        crs=grid.crs, transform=dst_transform,
        compress="lzw", tiled=True,
    ) as dst_file:
        dst_file.write(dst, 1)

    print(f"[worldcover] cached to {cache_path} ({sub_ny}x{sub_nx} @ 10m)")
    return dst


def class_fractions(classes_10m: np.ndarray, grid: CityGrid) -> dict[int, np.ndarray]:
    """Compute per-cell fraction of each WorldCover class on the 500 m grid.

    Returns a dict {class_code: fraction_array} where each array has shape grid.shape.
    Fractions sum to <=1 per cell (zero where all pixels were nodata).
    """
    ny, nx = grid.shape
    sub = 50
    # Block-reshape to (ny, sub, nx, sub) for vectorized counting.
    blocks = classes_10m.reshape(ny, sub, nx, sub)
    total = (blocks != 0).sum(axis=(1, 3)).astype(np.float32)  # 0 = nodata
    total = np.maximum(total, 1.0)  # avoid divide-by-zero

    out: dict[int, np.ndarray] = {}
    for code in _WC_CLASSES:
        count = (blocks == code).sum(axis=(1, 3)).astype(np.float32)
        out[code] = count / total
    return out


def build_worldcover_landuse(
    boundary: gpd.GeoDataFrame,
    grid: CityGrid,
    cache_dir: Path,
    tile: str = "N30W099",
) -> xr.Dataset:
    """Build a landuse Dataset (same schema as synthetic / OSM) from WorldCover.

    `tile` is the WorldCover v200 3° tile ID that covers the grid. Defaults
    to Austin's tile for back-compat; multi-city callers should pass the
    target city's tile ID (see `austin_twin.cities`).
    """
    cache_name = "worldcover_10m.tif" if tile == "N30W099" else f"worldcover_{tile}_10m.tif"
    classes = fetch_worldcover_for_grid(grid, cache_dir / cache_name, tile=tile)
    fr = class_fractions(classes, grid)

    impervious = fr[50]
    water = fr[80]
    vegetation = fr[10] + fr[20] + fr[30] + fr[40] + fr[90]
    bare = fr[60]
    # Treat 'bare' as a modest absorber (low vegetation cooling, no impervious
    # build-up). Allocate it half to impervious-like (low ET) and half to a
    # "neutral residual" we leave unbinned. Simplest: lump bare into vegetation
    # at half weight to capture some sparse-veg cooling without overstating it.
    vegetation = vegetation + 0.5 * bare

    # Mask outside city.
    inside = grid.mask
    impervious = np.where(inside, impervious, 0.0).astype(np.float32)
    vegetation = np.where(inside, vegetation, 0.0).astype(np.float32)
    water = np.where(inside, water, 0.0).astype(np.float32)

    return xr.Dataset(
        data_vars={
            "impervious_frac": (("y", "x"), impervious),
            "vegetation_frac": (("y", "x"), vegetation),
            # Keep the channel name 'water_mask' for compatibility with the
            # simulator + counterfactual code, even though it is now a
            # continuous fraction in [0, 1] rather than a binary mask.
            "water_mask": (("y", "x"), water),
            "city_mask": (("y", "x"), inside),
        },
        coords={"x": grid.x, "y": grid.y},
        attrs={
            "source": "esa_worldcover_v200_2021",
            "tile": "N30W099",
            "note": "water_mask is a continuous fraction (not strictly binary)",
        },
    )
