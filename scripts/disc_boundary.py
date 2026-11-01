"""Uniform disc city definition, for cross-country comparability.

OSM place queries return administrative units, and those are not comparable
across countries: "Riyadh, Saudi Arabia" yields a 375,478 km2 province while
"Perth, Western Australia" yields a 6 km2 CBD council. The variation is
already present inside the US sample too (Austin 725 km2 vs Miami-Dade
~6,300), it is simply less extreme.

For a comparison whose whole question is whether arid-city inversion is
climatic or continental, an administrative definition is a confound: it
varies with national practice rather than with anything physical. A disc of
fixed radius around the city centre removes it, and guarantees every city
carries a similar urban/peri-urban mix.

25 km is chosen to sit inside the US sample's own range (~1,963 km2, between
Austin and Miami-Dade, close to Phoenix's 1,344).
"""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point

DEFAULT_RADIUS_KM = 25.0


def disc_boundary(lat: float, lon: float, utm_crs: str,
                  radius_km: float = DEFAULT_RADIUS_KM) -> gpd.GeoDataFrame:
    """A circular city footprint of `radius_km` centred on (lat, lon)."""
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(utm_crs)
    return gpd.GeoDataFrame(geometry=[pt.iloc[0].buffer(radius_km * 1000.0)],
                            crs=utm_crs)
