"""Per-city configuration for the multi-city transfer study.

Austin remains the source (calibration) city. Phoenix and Miami are
target cities used to characterize how far the calibrated coefficients
port with zero, few-shot, and full re-fit target-city MODIS data.

Each `CityConfig` bundles:
  - OSM query string for the city-limits boundary;
  - local UTM CRS (metric grid, minimal distortion);
  - ERA5 bbox (with buffer so we get the surrounding grid points);
  - ESA WorldCover tile ID that covers the city;
  - UTC offset for aligning simulator "second-day peak frame" to the
    MODIS Aqua ~13:30-local overpass.

Adding a new city: pick a UTM zone from the longitude (⌊(lon+180)/6⌋+1
=> N in the northern hemisphere; EPSG code = 32600 + zone), find the
WorldCover tile at the SW corner rounded down to a 3° multiple (tiles
named `N{lat}{W|E}{lon}`; span 3°×3°), and fill in the bbox.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CityConfig:
    name: str
    osm_query: str
    utm_crs: str                     # e.g. "EPSG:32614" for UTM zone 14 N
    # ERA5 bbox in degrees, N/W/S/E order (matches cdsapi convention).
    era5_north: float
    era5_west: float
    era5_south: float
    era5_east: float
    # ESA WorldCover v200 3° tile ID covering the city, e.g. "N30W099".
    worldcover_tile: str
    # Local time offset from UTC in hours during the calibration window.
    # Austin: CDT = -5. Miami: EDT = -4. Phoenix: MST = -7 (no DST).
    utc_offset_hours: int


AUSTIN = CityConfig(
    name="austin",
    osm_query="Austin, Texas, USA",
    utm_crs="EPSG:32614",
    era5_north=30.85, era5_west=-98.20, era5_south=29.85, era5_east=-97.30,
    worldcover_tile="N30W099",
    utc_offset_hours=5,   # CDT (August)
)

PHOENIX = CityConfig(
    name="phoenix",
    osm_query="Phoenix, Arizona, USA",
    utm_crs="EPSG:32612",
    era5_north=34.00, era5_west=-112.70, era5_south=33.00, era5_east=-111.50,
    worldcover_tile="N33W114",
    utc_offset_hours=7,   # MST year-round (Arizona doesn't observe DST)
)

DENVER = CityConfig(
    name="denver",
    osm_query="Denver, Colorado, USA",
    utm_crs="EPSG:32613",   # UTM zone 13N
    # Denver is at 39.74 N, -104.99 W. Bbox buffered ~0.5 deg so ERA5's
    # 0.25 deg grid gives >=3x3 points around the city.
    era5_north=40.30, era5_west=-105.60, era5_south=39.20, era5_east=-104.40,
    # City proper crosses -105 W by ~12 km on the west; that sliver falls
    # into the neighboring WorldCover tile N39W108 and will be nodata
    # here. The eastern ~90 % of Denver (downtown, most residential) is
    # fully covered by N39W105 (39-42 N, 105-102 W).
    worldcover_tile="N39W105",
    utc_offset_hours=6,     # MDT = UTC-6 (Colorado observes DST)
)


MIAMI = CityConfig(
    name="miami",
    # "Miami city limits" is only 55 km²; too small at 500 m to give a stable
    # MODIS sample. Miami-Dade County (~6,300 km²) covers the urban Miami
    # heat-island footprint including Miami Beach, Hialeah, Kendall.
    osm_query="Miami-Dade County, Florida, USA",
    utm_crs="EPSG:32617",
    era5_north=26.30, era5_west=-80.80, era5_south=25.20, era5_east=-79.60,
    worldcover_tile="N24W081",
    utc_offset_hours=4,   # EDT (August)
)


CITIES: dict[str, CityConfig] = {c.name: c for c in (AUSTIN, PHOENIX, DENVER, MIAMI)}
