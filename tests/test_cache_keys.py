"""Raster cache keys must distinguish cities that share a CRS or a tile.

The multi-city sweep exposed this: Austin, Dallas and Oklahoma City all sit
in EPSG:32614, and many US cities share a 3-degree WorldCover tile. The old
keys (CRS for MODIS, tile for land cover) collided, which threw shape errors
here but would have silently swapped data between two cities whose grids
happened to have equal shapes.
"""
from __future__ import annotations

import numpy as np

from austin_twin.grid import CityGrid, grid_key

RES = 500.0


def _grid(x0: float, y0: float, ny: int = 40, nx: int = 30,
          crs: str = "EPSG:32614", res: float = RES) -> CityGrid:
    x = np.arange(nx, dtype=float) * res + x0
    y = np.arange(ny, dtype=float)[::-1] * res + y0

    class _B:  # stands in for the boundary GeoDataFrame
        pass

    b = _B()
    b.crs = crs
    g = CityGrid(boundary=b, resolution_m=res, x=x, y=y,
                 mask=np.ones((ny, nx), dtype=bool))
    return g


def test_same_crs_different_extent_differs():
    """Two cities in one UTM zone must not share a key."""
    austin = _grid(620_000.0, 3_330_000.0)
    dallas = _grid(700_000.0, 3_620_000.0)
    assert grid_key(austin) != grid_key(dallas)


def test_same_extent_different_resolution_differs():
    assert grid_key(_grid(620_000.0, 3_330_000.0, res=500.0)) != \
           grid_key(_grid(620_000.0, 3_330_000.0, res=100.0))


def test_same_origin_different_shape_differs():
    assert grid_key(_grid(620_000.0, 3_330_000.0, ny=40, nx=30)) != \
           grid_key(_grid(620_000.0, 3_330_000.0, ny=30, nx=40))


def test_different_crs_differs():
    assert grid_key(_grid(620_000.0, 3_330_000.0, crs="EPSG:32614")) != \
           grid_key(_grid(620_000.0, 3_330_000.0, crs="EPSG:32613"))


def test_key_is_stable_and_short():
    g = _grid(620_000.0, 3_330_000.0)
    assert grid_key(g) == grid_key(_grid(620_000.0, 3_330_000.0))
    assert len(grid_key(g)) == 10
