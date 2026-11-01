"""Zero-authentication ERA5 forcing from Google's ARCO-ERA5 mirror.

`era5.py` fetches the same fields from the Copernicus CDS, which requires a
registered account and an API key in `~/.cdsapirc`. That is the one input in
this pipeline that is not anonymously retrievable, and it also queues for
5-60 minutes per request, which makes any multi-city sweep impractical.

The obvious alternative — Planetary Computer's `era5-pds` collection, which
would have reused a dependency this project already has — is a dead end: the
collection exists but carries no items after 2022, so it has nothing for the
August 2024 calibration window.

ARCO-ERA5 is the identical ERA5 reanalysis republished by Google as a public
zarr store readable anonymously over HTTP. Verified equivalences:

  * `surface_solar_radiation_downwards` is hourly-accumulated J/m^2, the same
    convention CDS single-levels uses, so the `/3600` conversion in
    `Forcing.from_era5` carries over unchanged (a clear Austin August
    afternoon peaks at 3.43e6 J/m^2 = 953 W/m^2).
  * Temperatures are K, pressure Pa, winds m/s — all as CDS delivers them.

Two layout differences the loader has to absorb:

  * Longitudes run 0..360 rather than signed, so a CDS-style bbox needs
    wrapping.
  * Latitudes descend from +90, so the latitude slice runs north-to-south.

Cost: the store is chunked one timestep per chunk across the whole globe, so
a 48 h window costs about 45 s regardless of how small the spatial box is.
Extracted series are therefore cached to disk per (city, window); a second
call is effectively free. For a many-city sweep over shared dates, prefer
fetching each timestep once and extracting every city from it over calling
this per city.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .cities import CityConfig
from .forcing import Forcing

ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ARCO long names -> the short names `Forcing.from_era5` works in.
_VARS = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "surface_solar_radiation_downwards": "ssrd",
    "surface_pressure": "sp",
}

_DATASET = None


def _open_store(store: str = ARCO_STORE):
    """Open (and memoise) the ARCO zarr store for anonymous access."""
    global _DATASET
    if _DATASET is None:
        import xarray as xr  # local import: keeps zarr optional

        _DATASET = xr.open_zarr(
            store, chunks=None, storage_options={"token": "anon"},
            decode_timedelta=True,
        )
    return _DATASET


def _extract(city: CityConfig, start_iso: str, duration_hours: float,
             store: str) -> dict[str, np.ndarray]:
    ds = _open_store(store)

    # CDS bbox is signed; ARCO longitudes are 0..360.
    west, east = city.era5_west % 360, city.era5_east % 360
    descending = bool(ds.latitude.values[0] > ds.latitude.values[-1])
    lat_slice = (slice(city.era5_north, city.era5_south) if descending
                 else slice(city.era5_south, city.era5_north))

    t0 = np.datetime64(start_iso)
    t1 = t0 + np.timedelta64(int(duration_hours), "h")
    win = ds.sel(latitude=lat_slice, longitude=slice(west, east),
                 time=slice(t0, t1))

    if win.sizes["time"] == 0:
        raise ValueError(
            f"ARCO-ERA5 has no data for {city.name} in "
            f"[{start_iso}, +{duration_hours}h)"
        )
    if win.sizes["latitude"] == 0 or win.sizes["longitude"] == 0:
        raise ValueError(
            f"ARCO-ERA5 bbox for {city.name} selected no grid points "
            f"(N{city.era5_north} W{city.era5_west} "
            f"S{city.era5_south} E{city.era5_east})"
        )

    out = {
        short: win[long].mean(dim=["latitude", "longitude"]).values.astype(float)
        for long, short in _VARS.items()
    }
    times = win["time"].values
    out["times_hours"] = ((times - times[0]) / np.timedelta64(1, "h")).astype(float)
    return out


def load_arco_forcing(
    city: CityConfig,
    start_iso: str,
    duration_hours: float = 48.0,
    cache_dir: Path | None = None,
    store: str = ARCO_STORE,
) -> Forcing:
    """Build a `Forcing` for one simulation window from ARCO-ERA5.

    Field-for-field equivalent to `Forcing.from_era5` on a CDS NetCDF: the
    six surface fields are averaged over the city's ERA5 bbox and converted
    to the simulator's units.
    """
    key = f"arco_{city.name}_{start_iso.replace(':', '')}_{int(duration_hours)}h.npz"
    cache_path = (cache_dir / key) if cache_dir else None

    if cache_path is not None and cache_path.exists():
        raw = {k: v for k, v in np.load(cache_path).items()}
    else:
        raw = _extract(city, start_iso, duration_hours, store)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, **raw)

    # ssrd is J/m^2 accumulated over the preceding hour; the simulator wants
    # a mean flux over that hour. Identical handling to the CDS path.
    solar = np.clip(raw["ssrd"] / 3600.0, 0.0, None)
    wind = np.sqrt(raw["u10"] ** 2 + raw["v10"] ** 2)

    return Forcing(
        times_hours=raw["times_hours"],
        t_air_c=raw["t2m"] - 273.15,
        t_dew_c=raw["d2m"] - 273.15,
        wind_speed_m_s=wind,
        solar_w_m2=solar,
        pressure_pa=raw["sp"],
        source=f"arco_era5:{city.name}:{start_iso}",
        u_wind_m_s=raw["u10"],
        v_wind_m_s=raw["v10"],
    )


def forcing_for_modis_date(
    city: CityConfig,
    date_iso: str,
    duration_hours: float = 48.0,
    cache_dir: Path | None = None,
) -> Forcing:
    """Forcing window aligned to a MODIS date, matching the published runs.

    `run_era5_calibration.py` starts each 48 h simulation at local midnight
    of the day *before* the MODIS date, so the peak frame near t=37 h lands
    around 13:00 local — the Aqua overpass. Local midnight is expressed in
    UTC via the city's offset.
    """
    prev_day = date.fromisoformat(date_iso) - timedelta(days=1)
    start_iso = f"{prev_day.isoformat()}T{city.utc_offset_hours:02d}:00"
    return load_arco_forcing(city, start_iso, duration_hours, cache_dir=cache_dir)
