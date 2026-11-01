"""Tile-set derivation must cover cities that straddle a 3-degree boundary.

Choosing one tile from the city centroid silently drops whatever hangs over
the edge as nodata. Denver is the known case (its western ~12 km sits in
N39W108, not N39W105), but the failure is generic and silent, so it is
pinned here rather than special-cased.
"""
from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from austin_twin.worldcover import tile_name, tiles_for_boundary


def _bbox(minlon, minlat, maxlon, maxlat):
    return gpd.GeoDataFrame(
        geometry=[box(minlon, minlat, maxlon, maxlat)], crs="EPSG:4326")


def test_tile_name_matches_known_tiles():
    assert tile_name(30.27, -97.74) == "N30W099"   # Austin
    assert tile_name(33.45, -112.07) == "N33W114"  # Phoenix
    assert tile_name(25.76, -80.19) == "N24W081"   # Miami
    assert tile_name(39.74, -104.99) == "N39W105"  # Denver


def test_interior_city_needs_one_tile():
    # Austin sits well inside N30W099.
    assert tiles_for_boundary(_bbox(-98.0, 30.1, -97.5, 30.5)) == ["N30W099"]


def test_denver_straddles_two_tiles():
    """Denver crosses -105, so it needs N39W105 and N39W108."""
    tiles = tiles_for_boundary(_bbox(-105.11, 39.61, -104.60, 39.91))
    assert set(tiles) == {"N39W105", "N39W108"}


def test_corner_city_needs_four_tiles():
    tiles = tiles_for_boundary(_bbox(-105.1, 38.9, -104.9, 39.1))
    assert set(tiles) == {"N36W108", "N36W105", "N39W108", "N39W105"}


@pytest.mark.parametrize("lon", [-104.99, -105.01])
def test_tile_edge_is_not_off_by_one(lon):
    """A point either side of -105 must land in the correct tile."""
    expected = "N39W105" if lon > -105 else "N39W108"
    assert tile_name(39.74, lon) == expected
