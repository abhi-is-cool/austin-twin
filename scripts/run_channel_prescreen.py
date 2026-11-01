"""Pre-screen a candidate input channel against the ceiling, with no model run.

paper_v2's Discussion proposes adding a channel that distinguishes irrigated
from natural vegetation (NLCD irrigated crops, or a satellite ET product) to
break the Phoenix/Denver class-separability limit. Two problems with those
specific suggestions:

  * NLCD is US-only, so it cannot support a global multi-city study.
  * A satellite ET product is partly *derived from* MODIS LST, so feeding it
    back in as a predictor of LST risks inflating the ceiling circularly.

MODIS NDVI (MOD13A1, 500 m, 16-day) avoids both: it is global, it is a
reflectance-based vegetation index rather than a thermal retrieval, and it
separates irrigated green vegetation from spectrally-similar but sparse
desert scrub -- exactly the Phoenix failure mode.

The screen itself is the cheap part and is the point of the exercise: the
ceiling is a property of the *data*, so a candidate channel can be evaluated
by recomputing the ceiling with it included. If the ceiling does not move,
no model using that channel can do better, and the expensive integration
work is unnecessary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from austin_twin.ceiling import _multiple_r
from austin_twin.cities import CITIES
from austin_twin.grid import build_grid, fetch_boundary
from austin_twin.modis import download_asset, fetch_aqua_lst
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
_NDVI_COLLECTION = "modis-13A1-061"
_NDVI_ASSET = "500m_16_days_NDVI"
_NDVI_SCALE = 1e-4
_NDVI_FILL = -3000

DATES = {
    "austin": ("2024-08-22", "2024-08-01/2024-09-01"),
    "phoenix": ("2024-07-22", "2024-07-01/2024-08-01"),
    "denver": ("2024-07-25", "2024-07-01/2024-08-01"),
    "miami": ("2024-04-16", "2024-04-01/2024-05-01"),
}


def fetch_ndvi(grid, window: str, cache_dir: Path) -> np.ndarray:
    """MODIS NDVI mosaicked onto the model grid, NaN where fill."""
    boundary = grid.boundary.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = boundary.total_bounds
    cat = pystac_client.Client.open(_STAC, modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=[_NDVI_COLLECTION],
                            bbox=[float(minx), float(miny), float(maxx), float(maxy)],
                            datetime=window).items())
    if not items:
        raise RuntimeError(f"no {_NDVI_COLLECTION} items in {window}")

    ny, nx = grid.shape
    x_left = float(grid.x[0]) - grid.resolution_m / 2.0
    y_top = float(grid.y[0]) + grid.resolution_m / 2.0
    dst_transform = from_origin(x_left, y_top, grid.resolution_m, grid.resolution_m)

    out = np.full((ny, nx), np.nan, dtype=np.float32)
    filled = np.zeros((ny, nx), dtype=bool)
    for it in items:
        href = it.assets[_NDVI_ASSET].href
        local = download_asset(href, cache_dir / f"{it.id}_NDVI.tif")
        with rasterio.open(local) as src:
            buf = np.full((ny, nx), _NDVI_FILL, dtype=np.int16)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1), destination=buf,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=grid.crs,
                resampling=Resampling.nearest,
                src_nodata=_NDVI_FILL, dst_nodata=_NDVI_FILL, num_threads=4,
            )
        good = (buf != _NDVI_FILL) & ~filled
        out = np.where(good, buf.astype(np.float32) * _NDVI_SCALE, out)
        filled |= good
    return out


def main() -> None:
    rows = []
    print(f"{'city':<9} {'uni imp':>8} {'3-chan':>8} {'+NDVI':>8} "
          f"{'gain':>7} {'NDVI cov':>9}")
    print("-" * 56)

    for name, (date_iso, window) in DATES.items():
        city = CITIES[name]
        boundary = fetch_boundary(city, cache_path=RAW / f"{name}_boundary.geojson")
        grid = build_grid(boundary, resolution_m=500.0)
        landuse = build_worldcover_landuse(boundary, grid, RAW,
                                           tile=city.worldcover_tile)
        lst = fetch_aqua_lst(date_iso, grid, RAW / "modis")
        ndvi = fetch_ndvi(grid, window, RAW / "modis_ndvi")

        mask = landuse["city_mask"].values.astype(bool)
        valid = np.isfinite(lst) & mask & np.isfinite(ndvi)
        y = lst[valid].astype(float)
        imp = landuse["impervious_frac"].values[valid].astype(float)
        veg = landuse["vegetation_frac"].values[valid].astype(float)
        wat = landuse["water_mask"].values[valid].astype(float)
        nd = ndvi[valid].astype(float)

        uni = float(np.corrcoef(y, imp)[0, 1])
        three = _multiple_r(y, np.column_stack([imp, veg, wat]))
        four = _multiple_r(y, np.column_stack([imp, veg, wat, nd]))
        cov = float(valid.sum()) / max(int((np.isfinite(lst) & mask).sum()), 1)

        print(f"{name:<9} {uni:>8.4f} {three:>8.4f} {four:>8.4f} "
              f"{four - three:>+7.4f} {cov:>8.0%}")
        rows.append(dict(city=name, date=date_iso, univariate=uni,
                         three_channel=three, with_ndvi=four,
                         gain=four - three, ndvi_coverage=cov))

    print()
    print("=" * 56)
    print("INTERPRETATION")
    print("=" * 56)
    for r in rows:
        verdict = ("worth integrating" if r["gain"] > 0.05
                   else "marginal" if r["gain"] > 0.02 else "will not help")
        print(f"  {r['city']:<9} ceiling gain {r['gain']:+.4f}  ->  {verdict}")

    (OUT / "channel_prescreen.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote outputs/channel_prescreen.json")


if __name__ == "__main__":
    main()
