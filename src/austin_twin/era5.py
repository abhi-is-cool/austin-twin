"""ERA5 hourly forcing fetcher for Austin.

This module wraps `cdsapi` to download the surface fields the PM-ET path
needs (t2m, d2m, u10, v10, ssrd, sp) for a date window covering Austin
and writes them to a NetCDF that `Forcing.from_era5` can load.

Why it lives in its own module
------------------------------
The CDS API requires registered credentials (~/.cdsapirc with a URL and a
personal API key). We don't want every simulator import to drag in
`cdsapi` or fail at import time on machines without credentials. So:

  - The simulator and Forcing class never import this module.
  - Only `scripts/run_era5_fetch.py` (or an explicit user call) does.
  - The `cdsapi` import is local to the fetch function so a stale install
    doesn't break unrelated module imports.

Phase 2 of the PM-ET work begins by running this script once to populate
`data/raw/era5/` with NetCDF files for the validation date range.

Auth setup (one-time, in a future session):
  1. Register at https://cds.climate.copernicus.eu/
  2. Accept the ERA5 dataset license at
     https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
  3. Copy your API key from the profile page to ~/.cdsapirc:
        url: https://cds.climate.copernicus.eu/api
        key: <your-uid>:<your-api-key>
  4. `pip install cdsapi xarray netcdf4`
  5. `python scripts/run_era5_fetch.py --start 2023-08-01 --end 2023-08-31`
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

# Austin bbox with ~0.25 deg buffer so we get the ring of cells surrounding
# the city (ERA5 native resolution is 0.25 deg ~ 28 km, so this guarantees
# at least 3x3 grid points around downtown).
_AUSTIN_NORTH = 30.85
_AUSTIN_WEST = -98.20
_AUSTIN_SOUTH = 29.85
_AUSTIN_EAST = -97.30

_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "surface_pressure",
]

_HOURS = [f"{h:02d}:00" for h in range(24)]


@dataclass(frozen=True)
class Era5Request:
    """Inputs the CDS fetcher needs. Bounds default to Austin + buffer."""
    start_date: date
    end_date: date
    out_path: Path
    north: float = _AUSTIN_NORTH
    west: float = _AUSTIN_WEST
    south: float = _AUSTIN_SOUTH
    east: float = _AUSTIN_EAST

    def date_range(self) -> list[date]:
        days = (self.end_date - self.start_date).days
        return [self.start_date + timedelta(days=i) for i in range(days + 1)]


def fetch_era5_hourly(req: Era5Request, overwrite: bool = False) -> Path:
    """Download ERA5 hourly fields for the requested window into a NetCDF.

    Idempotent: if `req.out_path` already exists and `overwrite=False`, the
    function is a no-op and returns the existing path. This matters because
    a single ERA5 request can take 5-60 minutes in the CDS queue.

    The cdsapi import is local so a missing `cdsapi` install only breaks
    this call, not module import.
    """
    if req.out_path.exists() and not overwrite:
        return req.out_path
    req.out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cdsapi  # noqa: WPS433 -- local import is intentional
    except ImportError as exc:
        raise RuntimeError(
            "cdsapi is not installed. Run `pip install cdsapi xarray netcdf4` "
            "and ensure ~/.cdsapirc is configured (see era5.py docstring)."
        ) from exc

    years = sorted({d.year for d in req.date_range()})
    months = sorted({d.month for d in req.date_range()})
    days = sorted({d.day for d in req.date_range()})

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": _VARIABLES,
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in months],
            "day": [f"{d:02d}" for d in days],
            "time": _HOURS,
            "area": [req.north, req.west, req.south, req.east],  # N, W, S, E
            "format": "netcdf",
        },
        str(req.out_path),
    )
    return req.out_path
