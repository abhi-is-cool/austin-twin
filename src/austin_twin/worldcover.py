"""Fetch ESA WorldCover 10 m land cover and resample to the Austin grid.

WorldCover v200 (2021 epoch) is hosted as public COGs on AWS S3 in eu-central-1.
The tile covering Austin is `N30W099` (spans 30-33°N, 96-99°W).

Class codes used:
   10 Tree cover, 20 Shrubland, 30 Grassland, 40 Cropland, 50 Built-up,
   60 Bare/sparse vegetation, 80 Permanent water bodies, 90 Herbaceous wetland.

We aggregate from 10 m to the model grid by counting class memberships
per block. A city may straddle several 3-degree tiles; the full set is
derived from its boundary and mosaicked (see `tiles_for_boundary`).
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

from .grid import AUSTIN_CRS, GEO_CRS, CityGrid

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


_WC_NATIVE_RES_M = 10.0


def subsample_factor(resolution_m: float) -> int:
    """10 m WorldCover pixels per model cell along one axis.

    The aggregation used to assume 50 (500 m / 10 m) in two places, which
    silently pinned the whole pipeline to a 500 m grid. Deriving it from the
    grid is what lets the same code run at 250 m, 100 m or 30 m.
    """
    sub = int(round(resolution_m / _WC_NATIVE_RES_M))
    if sub < 1:
        raise ValueError(
            f"grid resolution {resolution_m} m is finer than WorldCover's "
            f"{_WC_NATIVE_RES_M:.0f} m native pixel; aggregation is undefined"
        )
    return sub


def ensure_local_tile(tile: str, tile_dir: Path) -> Path:
    """Download a WorldCover COG once and return the local path.

    Reading the tile through GDAL's `/vsicurl` handler issues one HTTP range
    request per internal block. These tiles are ~120 MB with thousands of
    blocks, and at the ~0.5 s round-trip this endpoint gives, warping the
    Austin window that way stalls for tens of minutes. One sequential GET of
    the whole tile takes about 12 s at 10 MB/s, so fetching once and warping
    from local disk is dramatically faster — and the tile is reusable, since
    a 3° tile covers many cities.

    Downloads to a `.part` file and renames on completion, so an interrupted
    fetch cannot leave a truncated tile that later looks like a valid cache.
    """
    tile_dir.mkdir(parents=True, exist_ok=True)
    dest = tile_dir / f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    if dest.exists():
        return dest

    import requests  # local import: only the fetch path needs it

    url = _WC_TILE_URL_TEMPLATE.format(tile=tile)
    part = dest.with_suffix(".part")
    print(f"[worldcover] downloading tile {tile} -> {dest.name}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        with open(part, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                got += len(chunk)
        if total and got != total:
            part.unlink(missing_ok=True)
            raise IOError(
                f"WorldCover tile {tile} truncated: got {got} of {total} bytes"
            )
    part.replace(dest)
    print(f"[worldcover] tile cached ({got / 1e6:.0f} MB)")
    return dest


def tile_name(lat: float, lon: float) -> str:
    """ESA WorldCover v200 tile ID whose 3-degree cell contains this point."""
    tlat = int(np.floor(lat / 3.0) * 3)
    tlon = int(np.floor(lon / 3.0) * 3)
    ns = "N" if tlat >= 0 else "S"
    ew = "W" if tlon < 0 else "E"
    return f"{ns}{abs(tlat):02d}{ew}{abs(tlon):03d}"


def tiles_for_boundary(boundary: gpd.GeoDataFrame) -> list[str]:
    """Every WorldCover tile intersecting a boundary's lon/lat bounding box.

    Tiles are 3 deg x 3 deg, so a city sitting near a multiple of 3 straddles
    two or four of them. Choosing a single tile from the city centroid
    silently leaves the overhang as nodata -- Denver loses its western ~12 km
    that way, and it is not the only city affected. Deriving the set from the
    bounding box removes the whole class of error.
    """
    minx, miny, maxx, maxy = boundary.to_crs(GEO_CRS).total_bounds
    lat0 = int(np.floor(miny / 3.0) * 3)
    lat1 = int(np.floor(maxy / 3.0) * 3)
    lon0 = int(np.floor(minx / 3.0) * 3)
    lon1 = int(np.floor(maxx / 3.0) * 3)
    out = []
    for lat in range(lat0, lat1 + 3, 3):
        for lon in range(lon0, lon1 + 3, 3):
            out.append(tile_name(lat + 0.5, lon + 0.5))
    return sorted(set(out))


def fetch_worldcover_for_grid(
    grid: CityGrid,
    cache_path: Path,
    tile: str | None = "N30W099",
    tiles: list[str] | None = None,
) -> np.ndarray:
    """Return a 10 m class raster aligned with the grid (shape ny*sub, nx*sub).

    `tile` is the WorldCover v200 3° tile ID covering the grid (e.g.
    "N30W099" for Austin, "N33W114" for Phoenix, "N24W081" for Miami).
    The warped result is cached locally as a GeoTIFF; the source tile is
    cached separately by `ensure_local_tile`.
    """
    if tiles is None:
        tiles = [tile] if tile else []
    if not tiles:
        raise ValueError("no WorldCover tiles specified")

    if cache_path.exists():
        with rasterio.open(cache_path) as src:
            return src.read(1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    ny, nx = grid.shape
    sub = subsample_factor(grid.resolution_m)
    sub_ny, sub_nx = ny * sub, nx * sub
    sub_res = grid.resolution_m / sub  # ~10 m

    x_left = float(grid.x[0]) - grid.resolution_m / 2.0
    y_top = float(grid.y[0]) + grid.resolution_m / 2.0
    dst_transform = from_origin(x_left, y_top, sub_res, sub_res)

    dst = np.zeros((sub_ny, sub_nx), dtype=np.uint8)  # 0 = nodata
    for t in tiles:
        tile_path = ensure_local_tile(t, cache_path.parent / "worldcover_tiles")
        print(f"[worldcover] warping {t} to {grid.resolution_m:.0f} m grid...")
        buf = np.zeros((sub_ny, sub_nx), dtype=np.uint8)
        with rasterio.open(tile_path) as src:
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=buf,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=grid.crs,
                resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0,
                num_threads=4,
            )
        # Fill only cells nothing has written yet, so tile overlaps are
        # harmless and the first tile supplying a pixel wins.
        fill = (dst == 0) & (buf != 0)
        dst = np.where(fill, buf, dst)
        if len(tiles) > 1:
            print(f"[worldcover]   {t} contributed {int(fill.sum()):,} pixels")

    # Write via a .part file and rename: a reader racing this write must see
    # either no file or a complete one. A partially-written GeoTIFF reads as
    # all-zero land cover, which yields a multiple correlation of ~1e-16 --
    # a number that looks like a result rather than a failure.
    part_path = cache_path.with_suffix(".part")
    with rasterio.open(
        part_path, "w",
        driver="GTiff",
        height=sub_ny, width=sub_nx,
        count=1, dtype=np.uint8,
        crs=grid.crs, transform=dst_transform,
        compress="lzw", tiled=True,
    ) as dst_file:
        dst_file.write(dst, 1)
    part_path.replace(cache_path)

    print(f"[worldcover] cached to {cache_path} "
          f"({sub_ny}x{sub_nx} @ {sub_res:.0f} m, sub={sub})")
    return dst


def class_fractions(classes_10m: np.ndarray, grid: CityGrid) -> dict[int, np.ndarray]:
    """Compute per-cell fraction of each WorldCover class on the model grid.

    Returns a dict {class_code: fraction_array} where each array has shape grid.shape.
    Fractions sum to <=1 per cell (zero where all pixels were nodata).
    """
    ny, nx = grid.shape
    sub = subsample_factor(grid.resolution_m)
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
    auto_tiles: bool = True,
) -> xr.Dataset:
    """Build a landuse Dataset (same schema as synthetic / OSM) from WorldCover.

    With `auto_tiles` (the default) the tile set is derived from the
    boundary's bounding box, so a city straddling a 3-degree tile edge is
    covered completely. `tile` is then only a hint and is unioned in. Pass
    `auto_tiles=False` to force the single named tile, which reproduces the
    older single-tile behaviour.
    """
    # The warped raster depends on the tile set, the resolution *and* the
    # city's own grid extent. Keying on the tile alone (as this did) collides
    # for any two cities sharing a 3-degree tile; adding resolution alone is
    # still not enough. grid_key covers CRS, resolution, shape and origin.
    from .grid import grid_key

    if auto_tiles:
        tiles = tiles_for_boundary(boundary)
        if tile and tile not in tiles:
            tiles = sorted(set(tiles) | {tile})
    else:
        tiles = [tile]

    cache_name = (f"worldcover_{'-'.join(tiles)}_{grid.resolution_m:.0f}m"
                  f"_{grid_key(grid)}.tif")
    classes = fetch_worldcover_for_grid(grid, cache_dir / cache_name,
                                        tile=None, tiles=tiles)
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
