"""Baseline physics-inspired surface-temperature simulator.

Per-cell update each timestep dt (seconds):

    dT/dt = solar_in(t, land) - radiative_out(T) - evapotranspiration(veg, T)
          + diffusion(T)

This is a deliberately simple zero-layer slab. It is not meant to be a
calibrated land-surface model — the goal is a qualitatively correct UHI
pattern (downtown > suburbs > parks > water) that we can validate against
satellite LST once real data is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from tqdm import tqdm


@dataclass
class SimConfig:
    dt_seconds: float = 600.0  # 10-minute timestep
    duration_hours: float = 48.0
    air_temp_c: float = 30.0  # mean ambient
    diurnal_amplitude_c: float = 6.0  # day/night swing in ambient
    peak_solar_w_m2: float = 900.0
    diffusion_m2_s: float = 50.0  # turbulent horizontal mixing
    cell_heat_capacity: float = 2.0e6  # J / (m^2 * K), thin slab approximation
    # Per-land-type absorption (fraction of incident shortwave retained as heat)
    absorption_impervious: float = 0.90
    absorption_vegetation: float = 0.55
    # Water absorbs ~6-10 % of shortwave (highly reflective at high sun angles).
    # The physically realistic value is used here; if you switch back to the
    # legacy proportional-damping water buffer you'd want a much higher value.
    absorption_water: float = 0.08
    # Evapotranspiration cooling strength (W/m^2 per unit veg_frac per K above air)
    et_coeff: float = 25.0
    # Net longwave loss coefficient (W/m^2 per K above sky temp)
    lw_coeff: float = 8.0
    # Shallow mixed-layer water model parameters. A water cell has thermal
    # mass `water_heat_capacity_ratio` times larger than a land slab (default
    # 4.0 ≈ a 2 m water column with rho*c_p_water = 4.186 MJ/m³/K, vs the
    # 2 MJ/m²/K land slab). Latent heat flux is a constant cooling term
    # representing summer-over-water evaporation (~80-150 W/m² typical).
    # Note: water_evap_w_m2 is ignored when use_pm_et is True; PM-ET computes
    # the latent flux over water cells from r_s = r_s_water = 0.
    water_heat_capacity_ratio: float = 4.0
    water_evap_w_m2: float = 100.0
    # Penman-Monteith ET switch. When False (default), the simulator uses the
    # legacy linear ET term `et_coeff * f_veg * (T-T_air)`. When True, it
    # uses the big-leaf PM-ET from penman_monteith.py, which requires a
    # Forcing object (or builds a synthetic one with climatological VPD/wind).
    use_pm_et: bool = False
    # Climatological defaults used when `use_pm_et=True` but no Forcing is
    # supplied. Dewpoint 22 °C ≈ Austin August median.
    default_dewpoint_c: float = 22.0
    default_wind_speed_m_s: float = 3.0
    default_pressure_pa: float = 1.01325e5
    # Store every Nth integration step in the history tensor (memory control
    # at fine grids). 1 = keep every step.
    frame_stride: int = 1


def safe_dt(diffusion_m2_s: float, dx_m: float, cfl_safety: float = 0.20) -> float:
    """Return the largest dt that satisfies the explicit-scheme CFL constraint.

    For 2D explicit diffusion the stability limit is D*dt/dx^2 <= 0.25; we use
    a safety factor (default 0.20) to stay comfortably under the threshold.
    """
    return cfl_safety * dx_m * dx_m / max(diffusion_m2_s, 1e-9)


@dataclass
class SimResult:
    temperature: np.ndarray  # (n_steps, ny, nx), Celsius
    times_hours: np.ndarray  # (n_steps,)
    config: SimConfig = field(repr=False)


def _laplacian(T: np.ndarray) -> np.ndarray:
    """5-point Laplacian with reflective (Neumann) boundary conditions."""
    L = np.zeros_like(T)
    L[1:-1, 1:-1] = (
        T[:-2, 1:-1] + T[2:, 1:-1] + T[1:-1, :-2] + T[1:-1, 2:] - 4 * T[1:-1, 1:-1]
    )
    # Reflective edges: copy interior neighbor into ghost cell -> zero flux.
    L[0, 1:-1] = T[1, 1:-1] + T[0, :-2] + T[0, 2:] - 3 * T[0, 1:-1]
    L[-1, 1:-1] = T[-2, 1:-1] + T[-1, :-2] + T[-1, 2:] - 3 * T[-1, 1:-1]
    L[1:-1, 0] = T[:-2, 0] + T[2:, 0] + T[1:-1, 1] - 3 * T[1:-1, 0]
    L[1:-1, -1] = T[:-2, -1] + T[2:, -1] + T[1:-1, -2] - 3 * T[1:-1, -1]
    return L


def run(
    landuse: xr.Dataset,
    config: SimConfig | None = None,
    forcing: "Forcing | None" = None,
) -> SimResult:
    """Integrate the slab heat equation forward in time.

    If `forcing` is provided it is used directly. Otherwise a synthetic
    diurnal forcing is built from `config` (preserving the legacy behavior
    of every script that doesn't know about the new Forcing API).

    The ET term is selected by `config.use_pm_et`:
      - False (default): legacy linear k_ET * f_veg * max(T-T_air, 0).
      - True: big-leaf Penman-Monteith from penman_monteith.py; in this
        mode `et_coeff` is unused.
    """
    # Local import keeps Forcing optional for scripts that never touch it.
    from .forcing import Forcing  # noqa: WPS433
    from . import penman_monteith as pm

    cfg = config or SimConfig()

    if forcing is None:
        forcing = Forcing.synthetic_diurnal(
            duration_hours=cfg.duration_hours,
            air_temp_mean_c=cfg.air_temp_c,
            air_temp_amplitude_c=cfg.diurnal_amplitude_c,
            dewpoint_c=cfg.default_dewpoint_c,
            wind_speed_m_s=cfg.default_wind_speed_m_s,
            peak_solar_w_m2=cfg.peak_solar_w_m2,
            pressure_pa=cfg.default_pressure_pa,
        )

    impervious = landuse["impervious_frac"].values.astype(np.float32)
    vegetation = landuse["vegetation_frac"].values.astype(np.float32)
    water = landuse["water_mask"].values.astype(np.float32)
    city = landuse["city_mask"].values.astype(bool)

    # Composite absorption per cell.
    absorption = (
        cfg.absorption_impervious * impervious
        + cfg.absorption_vegetation * vegetation
        + cfg.absorption_water * water
    )
    # Cells with no land class (outside city) get a neutral absorption.
    absorption = np.where(absorption > 0, absorption, cfg.absorption_vegetation)

    # Optional per-cell albedo boost (e.g., for cool-roof counterfactuals).
    # albedo_boost in [0, 1] reduces solar absorption multiplicatively.
    if "albedo_boost" in landuse:
        boost = landuse["albedo_boost"].values.astype(np.float32)
        absorption = absorption * (1.0 - np.clip(boost, 0.0, 1.0))

    # Spatial step from coordinate spacing (assume uniform).
    dx = float(abs(landuse["x"].values[1] - landuse["x"].values[0]))

    n_steps = int(cfg.duration_hours * 3600.0 / cfg.dt_seconds)
    ny, nx = impervious.shape
    stride = max(1, int(cfg.frame_stride))
    n_stored = n_steps // stride + 1  # frame 0 plus every `stride`-th step

    # Initial condition: mean of the forcing's T_air series across the whole
    # window. For synthetic diurnal forcing this is (numerically) `cfg.air_temp_c`
    # over integer periods, so pre-existing synthetic-forcing runs are unchanged.
    # For real ERA5 forcing this replaces the legacy summer default (30 °C) that
    # left the sim 20-25 °C above equilibrium on cold-season days and produced a
    # spurious decay pattern rather than a weather-driven one (see NUMBERS.md §7).
    T = np.full((ny, nx), float(np.mean(forcing.t_air_c)), dtype=np.float32)

    history = np.zeros((n_stored, ny, nx), dtype=np.float32)
    times = np.zeros(n_stored, dtype=np.float32)
    history[0] = T
    times[0] = 0.0
    next_store = 1  # index in `history` for the next frame to write

    diffusion_factor = cfg.diffusion_m2_s * cfg.dt_seconds / (dx * dx)
    if diffusion_factor > 0.25:
        raise ValueError(
            f"explicit-scheme CFL violated: D*dt/dx^2 = {diffusion_factor:.3f} > 0.25. "
            f"Reduce diffusion_m2_s, reduce dt_seconds, or coarsen the grid."
        )

    # Per-cell effective heat capacity: water cells are a shallow mixed-layer
    # with thermal mass `water_heat_capacity_ratio` times larger than the land
    # slab. This replaces the legacy proportional-damping water buffer.
    cell_capacity_eff = cfg.cell_heat_capacity * (
        1.0 + water * (cfg.water_heat_capacity_ratio - 1.0)
    )

    for step in tqdm(range(n_steps), desc="simulating"):
        t_hours = step * cfg.dt_seconds / 3600.0
        sample = forcing.sample(t_hours)
        S = sample["solar_w_m2"]
        T_air = sample["t_air_c"]

        # Energy fluxes (W/m^2)
        Q_solar = absorption * S
        Q_lw = cfg.lw_coeff * (T - T_air)
        if cfg.use_pm_et:
            R_net = Q_solar - Q_lw
            Q_et = pm.latent_heat_flux_w_m2(
                R_net_w_m2=R_net,
                T_air_c=T_air,
                T_dew_c=sample["t_dew_c"],
                f_veg=vegetation,
                f_water=water,
                wind_speed_m_s=sample["wind_speed_m_s"],
                pressure_pa=sample["pressure_pa"],
            )
            Q_evap_water = 0.0
        else:
            Q_et = cfg.et_coeff * vegetation * np.maximum(T - T_air, 0.0)
            Q_evap_water = cfg.water_evap_w_m2 * water
        Q_net = Q_solar - Q_lw - Q_et - Q_evap_water  # W/m^2

        dT_energy = Q_net * cfg.dt_seconds / cell_capacity_eff
        dT_diff = diffusion_factor * _laplacian(T)

        T = T + dT_energy + dT_diff

        # Store every `stride`-th frame in history (memory control). T itself
        # stays finite so the Laplacian doesn't propagate NaNs on next step.
        if (step + 1) % stride == 0 and next_store < n_stored:
            history[next_store] = np.where(city, T, np.nan)
            times[next_store] = t_hours + cfg.dt_seconds / 3600.0
            next_store += 1

    # Mask the t=0 frame too, for consistent visualization.
    history[0] = np.where(city, history[0], np.nan)

    return SimResult(temperature=history, times_hours=times, config=cfg)
