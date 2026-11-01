"""Pin the CityConfig derivation rules against the hand-written configs.

A 100-city sweep cannot have hand-picked UTM zones and WorldCover tile IDs,
so those are derived from the city centroid. The four configs in `cities.py`
were worked out by hand for the published study, which makes them an
independent check that the derivation is right — if the rules reproduce all
four, they can be trusted on cities nobody has checked.
"""
from __future__ import annotations

import pytest

from austin_twin.cities import (
    AUSTIN, DENVER, MIAMI, PHOENIX,
    city_from_point, utm_crs_for, worldcover_tile_for,
)

# Approximate centroids of the four published study cities. Miami's config
# uses Miami-Dade County rather than the city proper (see cities.py), so the
# centroid is the county's.
CENTROIDS = [
    ("austin", AUSTIN, 30.27, -97.74),
    ("phoenix", PHOENIX, 33.45, -112.07),
    ("denver", DENVER, 39.74, -104.99),
    ("miami", MIAMI, 25.76, -80.19),
]


@pytest.mark.parametrize("name,cfg,lat,lon", CENTROIDS)
def test_utm_crs_matches_handwritten(name, cfg, lat, lon):
    assert utm_crs_for(lat, lon) == cfg.utm_crs


@pytest.mark.parametrize("name,cfg,lat,lon", CENTROIDS)
def test_worldcover_tile_matches_handwritten(name, cfg, lat, lon):
    assert worldcover_tile_for(lat, lon) == cfg.worldcover_tile


@pytest.mark.parametrize("name,cfg,lat,lon", CENTROIDS)
def test_generated_config_covers_city(name, cfg, lat, lon):
    """The generated ERA5 bbox must contain the centroid and be non-degenerate."""
    gen = city_from_point(name, cfg.osm_query, lat, lon,
                          utc_offset_hours=cfg.utc_offset_hours)
    assert gen.era5_south < lat < gen.era5_north
    assert gen.era5_west < lon < gen.era5_east
    assert gen.utm_crs == cfg.utm_crs
    assert gen.worldcover_tile == cfg.worldcover_tile


def test_southern_hemisphere_and_eastern_longitude():
    """Rules must not be quietly northern/western-only.

    Sao Paulo (-23.55, -46.63) and Nairobi (-1.29, 36.82) exercise the S/E
    naming branches and the 327xx UTM band that the four US cities never hit.
    """
    assert utm_crs_for(-23.55, -46.63) == "EPSG:32723"
    assert worldcover_tile_for(-23.55, -46.63) == "S24W048"

    assert utm_crs_for(-1.29, 36.82) == "EPSG:32737"
    assert worldcover_tile_for(-1.29, 36.82) == "S03E036"

    # Northern + eastern: Delhi.
    assert utm_crs_for(28.61, 77.21) == "EPSG:32643"
    assert worldcover_tile_for(28.61, 77.21) == "N27E075"


def test_utc_offset_estimate_is_used_only_when_omitted():
    explicit = city_from_point("x", "q", 30.27, -97.74, utc_offset_hours=5)
    assert explicit.utc_offset_hours == 5

    estimated = city_from_point("x", "q", 30.27, -97.74)
    # -(-97.74)/15 = 6.5 -> banker's-safe round gives 7 or 6; either way it is
    # a standard-time estimate, and notably NOT Austin's DST value of 5.
    assert estimated.utc_offset_hours in (6, 7)
