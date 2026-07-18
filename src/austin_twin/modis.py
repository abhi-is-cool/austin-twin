"""Fetch MODIS Aqua (MYD11A1) daily LST via Microsoft Planetary Computer.

We use the afternoon Aqua overpass (~1:30 pm local) because it is the closest
to peak surface temperature, where the urban heat island signal is strongest.

LST is stored as scaled uint16 with scale factor 0.02 K and fill value 0.
After reading we apply: T[°C] = raw * 0.02 - 273.15  and mask raw == 0.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from .grid import AUSTIN_CRS, CityGrid

_STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
_COLLECTION = "modis-11A1-061"
_LST_SCALE = 0.02
_LST_FILL = 0  # raw values of 0 are nodata


def _open_catalog() -> pystac_client.Client:
    return pystac_client.Client.open(
        _STAC_API, modifier=planetary_computer.sign_inplace
    )


def find_aqua_lst_items(
    date_iso: str,
    grid: CityGrid,
) -> list:
    """Return all MYD11A1 (Aqua) items intersecting the city bbox on `date_iso`.

    Austin straddles MODIS tile boundary h09v05 / h09v06, so we typically get
    one item per tile and mosaic them on reprojection.
    """
    boundary = grid.boundary.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = boundary.total_bounds
    catalog = _open_catalog()
    search = catalog.search(
        collections=[_COLLECTION],
        bbox=[float(minx), float(miny), float(maxx), float(maxy)],
        datetime=date_iso,
    )
    items = [it for it in search.items() if it.id.startswith("MYD")]
    return items


def probe_coverage(
    date_iso: str,
    grid: CityGrid,
    half_window_px: int = 25,
) -> tuple[float, float, int]:
    """Quickly check LST coverage on a date without warping/caching.

    Reads a (2*half_window_px)^2 pixel window centered on the grid's own
    city centroid (in MODIS sinusoidal space) and returns
    (coverage_fraction, mean_LST_C, n_valid). Coverage is the fraction of
    the window with valid (cloud-free) LST. Returns (0, nan, 0) if no
    MYD11A1 items are found for that date over the city bbox.

    Multi-city aware: derives the centroid from `grid.boundary` rather
    than hardcoding Austin. Used by extended-validation scans to filter
    candidate dates cheaply before paying the full fetch+warp+cache cost.
    """
    items = find_aqua_lst_items(date_iso, grid)
    if not items:
        return 0.0, float("nan"), 0
    # City centroid: compute in the boundary's own metric CRS (silences a
    # spurious "centroid in geographic CRS" warning), then reproject the
    # single point to lon/lat.
    centroid_metric = gpd.GeoSeries(
        [grid.boundary.geometry.centroid.iloc[0]], crs=grid.boundary.crs
    ).to_crs("EPSG:4326").iloc[0]
    lon, lat = float(centroid_metric.x), float(centroid_metric.y)
    # STAC can return multiple tiles for a city (h09v04, h09v05, h10v04, ...);
    # for the centroid probe we want the tile that actually contains the
    # centroid. Iterate through items and use the first one that does.
    for it in items:
        href = it.assets["LST_Day_1km"].href
        with rasterio.open(href) as src:
            pt_proj = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
            col, row = ~src.transform * (pt_proj[0][0], pt_proj[1][0])
            col, row = int(col), int(row)
            p = half_window_px
            ny_s, nx_s = src.shape
            if not (0 <= row < ny_s and 0 <= col < nx_s):
                continue
            win = src.read(1, window=(
                (max(0, row - p), min(ny_s, row + p)),
                (max(0, col - p), min(nx_s, col + p)),
            ))
        valid = win > 0
        total = int(win.size)
        if not valid.any() or total == 0:
            continue
        cel = win[valid] * _LST_SCALE - 273.15
        return float(valid.sum() / total), float(cel.mean()), int(valid.sum())
    return 0.0, float("nan"), 0


def fetch_aqua_lst(
    date_iso: str,
    grid: CityGrid,
    cache_dir: Path,
) -> np.ndarray:
    """Fetch + mosaic + reproject MODIS Aqua LST_Day to the city grid (500 m).

    Returns LST in Celsius, shape grid.shape, with NaN for fill/nodata cells.
    Cached locally after first fetch.
    """
    # Cache is keyed by CRS so Austin / Phoenix / Denver / Miami don't
    # collide on the same filename with different destination grids.
    crs_dir = str(grid.crs).replace(":", "_")
    cache_path = cache_dir / crs_dir / f"modis_aqua_lst_day_{date_iso}.tif"
    if cache_path.exists():
        with rasterio.open(cache_path) as src:
            return src.read(1)

    items = find_aqua_lst_items(date_iso, grid)
    if not items:
        raise RuntimeError(f"No MYD11A1 items for {date_iso} over city bbox")
    print(f"[modis] found {len(items)} Aqua items: {[it.id.split('.')[2] for it in items]}")

    ny, nx = grid.shape
    x_left = float(grid.x[0]) - grid.resolution_m / 2.0
    y_top = float(grid.y[0]) + grid.resolution_m / 2.0
    dst_transform = from_origin(x_left, y_top, grid.resolution_m, grid.resolution_m)

    # Warp each tile in raw uint16 space (nearest-neighbor keeps the fill
    # value of 0 from propagating into valid neighbors). Mosaic by taking
    # the first valid value per destination cell.
    out_raw = np.zeros((ny, nx), dtype=np.uint16)
    valid_mask = np.zeros((ny, nx), dtype=bool)
    for it in items:
        href = it.assets["LST_Day_1km"].href
        print(f"[modis]   warping {it.id}...")
        with rasterio.open(href) as src:
            buf = np.zeros((ny, nx), dtype=np.uint16)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=buf,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=grid.crs,
                resampling=Resampling.nearest,
                src_nodata=_LST_FILL,
                dst_nodata=_LST_FILL,
                num_threads=4,
            )
        new_valid = (buf != _LST_FILL) & ~valid_mask
        out_raw = np.where(new_valid, buf, out_raw)
        valid_mask = valid_mask | new_valid

    out = np.where(
        valid_mask, out_raw.astype(np.float32) * _LST_SCALE - 273.15, np.nan
    ).astype(np.float32)
    # Restrict to within city boundary.
    out = np.where(grid.mask, out, np.nan).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        cache_path, "w",
        driver="GTiff", height=ny, width=nx, count=1, dtype=np.float32,
        crs=grid.crs, transform=dst_transform, nodata=np.nan,
        compress="lzw", tiled=True,
    ) as dst:
        dst.write(out, 1)
    print(f"[modis] cached -> {cache_path.name}")
    return out
