"""Time-varying atmospheric forcing for the slab simulator.

A `Forcing` object bundles the atmospheric fields the simulator needs at
each timestep: air temperature, dewpoint, wind speed, solar radiation, and
surface pressure. Forcings are stored as 1-D arrays over time (spatially
uniform across the city), which is appropriate at the ~50 km extent of
Austin and the 0.25 ° native resolution of ERA5 reanalysis.

Two construction paths:

  - `Forcing.synthetic_diurnal(...)`: reproduces the legacy sinusoidal
    diurnal cycle the simulator used before PM-ET, so existing scripts
    that don't supply a Forcing continue to work unchanged.

  - `Forcing.from_era5(...)`: reads the NetCDF produced by
    `scripts/run_era5_fetch.py` and aligns it to the simulator's time grid.
    This is the real-forcing path; requires CDS API credentials to fetch.

The simulator's `run()` accepts a `Forcing` (or builds a synthetic one
from `SimConfig`) and interpolates field values at each timestep via
`sample(t_hours)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Forcing:
    """Hourly atmospheric forcing, spatially uniform across the city.

    All arrays are 1-D in time. Times are in decimal hours from t = 0 of
    the simulation (typically aligned with midnight local of a chosen
    start date).
    """
    times_hours: np.ndarray             # shape (n,)
    t_air_c: np.ndarray                 # shape (n,)
    t_dew_c: np.ndarray                 # shape (n,)
    wind_speed_m_s: np.ndarray          # shape (n,)
    solar_w_m2: np.ndarray              # shape (n,)
    pressure_pa: np.ndarray             # shape (n,)
    source: str = "synthetic"

    def __post_init__(self) -> None:
        n = len(self.times_hours)
        for name in ("t_air_c", "t_dew_c", "wind_speed_m_s",
                     "solar_w_m2", "pressure_pa"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(
                    f"Forcing field '{name}' length {len(arr)} ≠ times_hours length {n}"
                )

    def sample(self, t_hours: float) -> dict[str, float]:
        """Linearly interpolate every field to a single time `t_hours`.

        Out-of-range times saturate at the first / last sample (no
        extrapolation), so a 48 h simulation that runs slightly past the
        loaded forcing range degrades to a constant rather than blowing up.
        """
        t = float(np.clip(t_hours, self.times_hours[0], self.times_hours[-1]))
        return dict(
            t_air_c=float(np.interp(t, self.times_hours, self.t_air_c)),
            t_dew_c=float(np.interp(t, self.times_hours, self.t_dew_c)),
            wind_speed_m_s=float(np.interp(t, self.times_hours, self.wind_speed_m_s)),
            solar_w_m2=float(np.interp(t, self.times_hours, self.solar_w_m2)),
            pressure_pa=float(np.interp(t, self.times_hours, self.pressure_pa)),
        )

    # ----- constructors -----

    @classmethod
    def synthetic_diurnal(
        cls,
        duration_hours: float,
        air_temp_mean_c: float = 30.0,
        air_temp_amplitude_c: float = 6.0,
        air_temp_min_offset_hours: float = 6.0,  # T_air minimum at ~6am
        dewpoint_c: float = 22.0,                # typical Austin August
        wind_speed_m_s: float = 3.0,             # climatological default
        peak_solar_w_m2: float = 900.0,
        pressure_pa: float = 1.01325e5,
        sample_step_hours: float = 0.25,
    ) -> "Forcing":
        """Build a back-compatible synthetic forcing matching the legacy
        diurnal cycle the simulator used before PM-ET.

        T_air follows the same sinusoid the old code used; dewpoint,
        wind, and pressure are climatological constants reasonable for
        Austin summer (override per call when running other seasons).
        """
        t = np.arange(0.0, duration_hours + sample_step_hours, sample_step_hours)
        # Matches the legacy _ambient_temp(t, mean, amplitude) in simulator.py:
        # mean + amplitude * sin(2π * (t - 9) / 24), so min ~6am, max ~3pm.
        t_air = air_temp_mean_c + air_temp_amplitude_c * np.sin(
            2.0 * np.pi * (t - 9.0) / 24.0
        )
        # Half-sine daytime solar (peaks at solar noon ~13:00), zero at night.
        hour_of_day = np.mod(t, 24.0)
        solar = np.where(
            (hour_of_day >= 6.0) & (hour_of_day <= 20.0),
            peak_solar_w_m2 * np.sin(np.pi * (hour_of_day - 6.0) / 14.0),
            0.0,
        )
        return cls(
            times_hours=t,
            t_air_c=t_air,
            t_dew_c=np.full_like(t, dewpoint_c),
            wind_speed_m_s=np.full_like(t, wind_speed_m_s),
            solar_w_m2=solar,
            pressure_pa=np.full_like(t, pressure_pa),
            source="synthetic_diurnal",
        )

    @classmethod
    def from_era5(
        cls,
        nc_path: Path,
        start_iso: str,
        duration_hours: float,
    ) -> "Forcing":
        """Load ERA5 hourly forcing for a time window into a Forcing object.

        CDS-Beta returns a zip containing two NetCDFs (instant variables
        and hourly-accumulated variables), so this loader:
          - unzips transparently if `nc_path` is actually a zip (the
            fetcher writes .nc but the CDS response is a zip);
          - merges the instant and accum datasets on `valid_time`;
          - handles the modern `valid_time` dim name as well as the
            legacy `time` name.

        Spatially averages across whatever grid points fall inside the
        requested Austin bbox (~2-12 ERA5 cells at 0.25 deg resolution).
        """
        import xarray as xr  # noqa: WPS433 -- local import is intentional
        import zipfile

        ds = _open_era5_dataset(nc_path, xr, zipfile)
        # Standardize the time dim to "time".
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})

        t0 = np.datetime64(start_iso)
        t1 = t0 + np.timedelta64(int(duration_hours), "h")
        ds_window = ds.sel(time=slice(t0, t1))
        if len(ds_window.time) == 0:
            raise ValueError(
                f"ERA5 file {nc_path} has no data in [{start_iso}, "
                f"{start_iso} + {duration_hours}h). File range: "
                f"{ds.time.values[0]} to {ds.time.values[-1]}"
            )

        def _mean(var_name: str) -> np.ndarray:
            return ds_window[var_name].mean(
                dim=[d for d in ("latitude", "longitude")
                     if d in ds_window[var_name].dims]
            ).values

        times = ds_window["time"].values
        times_hours = (times - times[0]) / np.timedelta64(1, "h")

        t_air_c = _mean("t2m") - 273.15
        t_dew_c = _mean("d2m") - 273.15
        u10 = _mean("u10")
        v10 = _mean("v10")
        wind = np.sqrt(u10 * u10 + v10 * v10)
        # ssrd is J/m² accumulated over the previous hour. Divide by 3600
        # to convert to mean W/m² over that hour.
        ssrd = _mean("ssrd") / 3600.0
        solar = np.clip(ssrd, 0.0, None)
        sp = _mean("sp")  # ERA5 surface pressure in Pa.

        return cls(
            times_hours=times_hours.astype(float),
            t_air_c=t_air_c.astype(float),
            t_dew_c=t_dew_c.astype(float),
            wind_speed_m_s=wind.astype(float),
            solar_w_m2=solar.astype(float),
            pressure_pa=sp.astype(float),
            source=f"era5:{nc_path.name}:{start_iso}",
        )


def _open_era5_dataset(nc_path: "Path", xr, zipfile):
    """Open an ERA5 file returned by CDS-Beta.

    If `nc_path` is a plain NetCDF, opens it directly. If it's actually
    a zip (CDS-Beta returns a zip for multi-stepType requests), unpacks
    the inner .nc files into a per-zip subdirectory (`unpacked/<stem>/`)
    and merges the instant + accum datasets. Per-zip subdirs prevent
    Austin/Phoenix/Miami zips (which all contain identically-named inner
    files) from colliding on the same cache paths.
    """
    if not zipfile.is_zipfile(nc_path):
        return xr.open_dataset(nc_path)

    unpack_dir = nc_path.parent / "unpacked" / nc_path.stem
    unpack_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(nc_path) as zf:
        inner_names = zf.namelist()
        for name in inner_names:
            target = unpack_dir / name
            if not target.exists():
                zf.extract(name, unpack_dir)

    datasets = [xr.open_dataset(unpack_dir / name) for name in inner_names]
    if len(datasets) == 1:
        return datasets[0]
    return xr.merge(datasets, compat="override")
