"""Penman-Monteith latent-heat flux (canonical form) for the slab simulator.

Why this exists
---------------
The legacy linear ET term
    Q_LE = k_ET * f_veg * max(T_surf - T_air, 0)
saturates incorrectly at high vegetation fractions and undershoots peak
summer ET rates by roughly an order of magnitude. The cooling_response
analysis showed local cooling per +10 % canopy stays at ~0.5 °C
regardless of calibration: the linear form simply cannot extract enough
latent heat from a fully-vegetated cell.

This module implements the canonical FAO-56 (Allen et al. 1998) form of
the Penman-Monteith equation:

    Q_LE = [ Δ · Rn + ρ_air · c_p · VPD / r_a ]
           ----------------------------------
           [ Δ + γ · (1 + r_s / r_a) ]

with:
  - Δ   = slope of saturation-vapor-pressure curve at T_air [Pa/K]
  - Rn  = net radiation absorbed by the surface [W/m²]
  - VPD = e_sat(T_air) - e_air [Pa]
  - r_a = aerodynamic resistance [s/m] -- function of wind
  - r_s = bulk surface resistance [s/m] -- function of land cover via LAI
  - γ   = psychrometric constant [Pa/K]

Why canonical PM and not "big-leaf bulk":
  The big-leaf bulk form Q_LE = ρ·c_p · (e_sat(T_surf) - e_a) / γ / (r_a+r_s)
  uses surface saturation pressure for stronger T_surf feedback, but it is
  NOT bounded by energy availability and can return values multiples of
  the incoming radiation. Canonical PM is bounded (Q_LE ≤ ~1.26 × Rn under
  reference conditions) and matches the form on which the entire UFOR /
  land-surface literature reports values.

  The cost: PM is a steady-state diagnostic and does not include explicit
  T_surf feedback. In our slab model the feedback enters through the next
  timestep: a hotter slab radiates more (Q_LW), which lowers Rn, which
  lowers PM-ET on the next step.

Defaults follow FAO-56 reference grass (h_canopy = 0.12 m, r_s_leaf = 100
s/m) so that PM-ET reduces to the FAO ET₀ benchmark when f_veg = 1. For
sparse vegetation, r_s rises as LAI shrinks; for water, r_s -> 0; for
bare/impervious, r_s -> very large -> Q_LE -> 0.

Reference: Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998).
FAO-56, chapters 1-4.
"""
from __future__ import annotations

import numpy as np

# Physical constants
_RHO_AIR_KG_PER_M3 = 1.225
_CP_AIR_J_PER_KG_PER_K = 1005.0
_VON_KARMAN = 0.41
_DEFAULT_SURFACE_PRESSURE_PA = 1.01325e5

# FAO-56 reference grass parameters (h_canopy = 0.12 m).
# Yields r_a ≈ 208 / u_2m  s/m at z = 2 m, matching Allen 1998 Eq. 4.
_REF_CANOPY_HEIGHT_M = 0.12
_REF_D_M = 2.0 / 3.0 * _REF_CANOPY_HEIGHT_M           # zero-plane displacement
_REF_Z0M_M = 0.123 * _REF_CANOPY_HEIGHT_M             # roughness for momentum
_REF_Z0H_M = 0.1 * _REF_Z0M_M                         # roughness for heat
_REF_Z_M = 2.0                                         # reference height

# Surface resistance parameters
_R_S_LEAF_S_PER_M = 100.0        # per-leaf-area stomatal resistance (FAO grass)
_R_S_BARE_S_PER_M = 2000.0       # bare/impervious soil -- essentially no ET
_R_S_WATER_S_PER_M = 1.0         # free water -- aerodynamic-limited; small but nonzero
_LAI_MAX = 5.0                   # closed-canopy LAI
_LAI_K = 3.0                     # f_veg -> LAI saturation rate


# ---------- thermodynamic helpers ----------

def saturation_vapor_pressure_kpa(T_celsius: float | np.ndarray) -> float | np.ndarray:
    """Tetens formula. Returns e_sat in kPa for T in °C."""
    return 0.6108 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))


def saturation_vapor_pressure_slope_kpa_per_k(T_celsius: float | np.ndarray) -> float | np.ndarray:
    """d(e_sat)/dT at T (kPa/K)."""
    e_sat = saturation_vapor_pressure_kpa(T_celsius)
    return 4098.0 * e_sat / (T_celsius + 237.3) ** 2


def psychrometric_constant_kpa_per_k(pressure_pa: float | np.ndarray = _DEFAULT_SURFACE_PRESSURE_PA) -> float | np.ndarray:
    """γ = 0.665e-3 × P[kPa]  (Allen 1998 Eq. 8)."""
    return 0.665e-3 * (np.asarray(pressure_pa) / 1000.0)


# ---------- resistances ----------

def aerodynamic_resistance_s_per_m(
    wind_speed_m_s: float | np.ndarray,
    z_m: float = _REF_Z_M,
    z_h: float = _REF_Z_M,
    h_canopy_m: float = _REF_CANOPY_HEIGHT_M,
) -> float | np.ndarray:
    """FAO-56 Eq. 4: log-law r_a with separate momentum / heat roughness.

        r_a = ln((z_m-d)/z_om) * ln((z_h-d)/z_oh) / (κ² · u_z)

    Defaults give r_a ≈ 208/u_2m for FAO reference grass. Stability
    corrections are ignored (valid for daily timescales and near-neutral
    afternoons; less accurate for individual stable nights).
    """
    d = 2.0 / 3.0 * h_canopy_m
    z_om = 0.123 * h_canopy_m
    z_oh = 0.1 * z_om
    u = np.maximum(np.asarray(wind_speed_m_s, dtype=np.float64), 0.5)
    log_m = np.log((z_m - d) / z_om)
    log_h = np.log((z_h - d) / z_oh)
    return log_m * log_h / (_VON_KARMAN * _VON_KARMAN * u)


def leaf_area_index(
    f_veg: np.ndarray,
    lai_max: float = _LAI_MAX,
    k: float = _LAI_K,
    mode: str = "exponential",
) -> np.ndarray:
    """Leaf area index as a function of vegetation fraction.

    "exponential" (default, canonical): LAI = LAI_max · (1 - exp(-k · f_veg)).
    Saturating: f_veg = 0 -> 0; f_veg -> 1 -> LAI_max. At the canonical
    k = 3 it reaches 84 % of LAI_max by f_veg = 0.5, which flattens the ET
    gradient across most Austin cells (median f_veg = 0.66).

    "linear": LAI = LAI_max · f_veg. This is what "linear LAI" should mean —
    the same range, without the saturation. Note that taking k -> 0 in the
    exponential form does *not* produce this: it collapses to
    LAI_max · k · f_veg, i.e. near-zero LAI everywhere, which suppresses ET
    entirely rather than linearising it.
    """
    f = np.asarray(f_veg)
    if mode == "linear":
        return lai_max * np.clip(f, 0.0, 1.0)
    if mode != "exponential":
        raise ValueError(f"unknown LAI mode {mode!r}")
    return lai_max * (1.0 - np.exp(-k * f))


def surface_resistance_s_per_m(
    f_veg: np.ndarray,
    f_water: np.ndarray,
    r_s_leaf: float = _R_S_LEAF_S_PER_M,
    r_s_bare: float = _R_S_BARE_S_PER_M,
    r_s_water: float = _R_S_WATER_S_PER_M,
    lai_max: float = _LAI_MAX,
    lai_k: float = _LAI_K,
    lai_mode: str = "exponential",
) -> np.ndarray:
    """Bulk surface resistance as a parallel combination of partitions.

        1 / r_s = f_veg / r_s_canopy(LAI) + f_water / r_s_water + f_bare / r_s_bare

    where r_s_canopy = r_s_leaf / max(LAI, eps). Water dominates the bulk
    conductance when present because r_s_water is small.
    """
    f_veg = np.asarray(f_veg, dtype=np.float64)
    f_water = np.asarray(f_water, dtype=np.float64)
    f_bare = np.clip(1.0 - f_veg - f_water, 0.0, 1.0)

    lai = leaf_area_index(f_veg, lai_max=lai_max, k=lai_k, mode=lai_mode)
    r_s_canopy = np.where(lai > 1e-3, r_s_leaf / np.maximum(lai, 1e-3), np.inf)

    g_veg = np.where(np.isfinite(r_s_canopy), f_veg / np.maximum(r_s_canopy, 1e-6), 0.0)
    g_water = f_water / max(r_s_water, 1e-6)
    g_bare = f_bare / max(r_s_bare, 1e-6)
    g_total = g_veg + g_water + g_bare
    return np.where(g_total > 0, 1.0 / g_total, np.inf)


# ---------- main entry point ----------

def latent_heat_flux_w_m2(
    R_net_w_m2: float | np.ndarray,
    T_air_c: float | np.ndarray,
    T_dew_c: float | np.ndarray,
    f_veg: np.ndarray,
    f_water: np.ndarray,
    wind_speed_m_s: float | np.ndarray = 3.0,
    pressure_pa: float | np.ndarray = _DEFAULT_SURFACE_PRESSURE_PA,
    T_surf_c: np.ndarray | None = None,
    vpd_mode: str = "air",
    lai_k: float = _LAI_K,
    lai_mode: str = "exponential",
    r_s_leaf: float = _R_S_LEAF_S_PER_M,
    max_evaporative_fraction: float | None = None,
) -> np.ndarray:
    """Penman-Monteith latent heat flux Q_LE (W/m²).

    `vpd_mode` selects how the vapour-pressure deficit is formed, which is
    the whole subject of the ET diagnostic:

      "air"      Canonical FAO-56. VPD = e_sat(T_air) - e_sat(T_dew), with
                 both driven by spatially uniform ERA5 fields. Because
                 nothing in the numerator varies per cell except Rn, the
                 T_surf feedback that drives spatial contrast in the linear
                 ET term is absent -- the diagnosed cause of most of the
                 published PM regression.

      "surface"  Canonical PM structure, but VPD = e_sat(T_i) - e_sat(T_dew).
                 Restores per-cell surface-temperature feedback while keeping
                 the energy-bounded PM denominator. This is the formulation
                 the paper's Discussion proposes.

      "bulk"     True big-leaf bulk form,
                 Q_LE = rho c_p (e_sat(T_i) - e_a) / (gamma (r_a + r_s)).
                 Strongest feedback, but not bounded by available energy --
                 it can exceed Rn, which is why the module originally
                 avoided it. Included so the trade-off can be measured
                 rather than assumed.

    `lai_k` exposes the LAI saturation rate so the second mechanism (LAI
    saturating at f_veg ~ 0.5) can be varied independently of VPD.

    `max_evaporative_fraction`, if given, caps Q_LE at that multiple of
    max(Rn, 0). Use with "bulk" to keep an unbounded form numerically sane.

    Returns Q_LE in W/m², clipped to be non-negative (no condensation).
    """
    if vpd_mode not in ("air", "surface", "bulk"):
        raise ValueError(f"unknown vpd_mode {vpd_mode!r}")
    if vpd_mode != "air" and T_surf_c is None:
        raise ValueError(f"vpd_mode={vpd_mode!r} requires T_surf_c")

    gamma = psychrometric_constant_kpa_per_k(pressure_pa) * 1000.0
    e_air = saturation_vapor_pressure_kpa(T_dew_c) * 1000.0

    r_a = aerodynamic_resistance_s_per_m(wind_speed_m_s)
    r_s = surface_resistance_s_per_m(f_veg, f_water, lai_k=lai_k,
                                     lai_mode=lai_mode, r_s_leaf=r_s_leaf)

    if vpd_mode == "bulk":
        e_surf = saturation_vapor_pressure_kpa(T_surf_c) * 1000.0
        vpd = np.maximum(e_surf - e_air, 0.0)
        q_le = (_RHO_AIR_KG_PER_M3 * _CP_AIR_J_PER_KG_PER_K * vpd
                / (gamma * (r_a + r_s)))
    else:
        # Δ is evaluated at the temperature driving the deficit.
        T_delta = T_air_c if vpd_mode == "air" else T_surf_c
        delta = saturation_vapor_pressure_slope_kpa_per_k(T_delta) * 1000.0
        e_ref = saturation_vapor_pressure_kpa(
            T_air_c if vpd_mode == "air" else T_surf_c
        ) * 1000.0
        vpd = np.maximum(e_ref - e_air, 0.0)

        numerator = (delta * R_net_w_m2
                     + _RHO_AIR_KG_PER_M3 * _CP_AIR_J_PER_KG_PER_K * vpd / r_a)
        denominator = delta + gamma * (1.0 + r_s / r_a)
        q_le = numerator / denominator

    q_le = np.clip(np.where(np.isfinite(q_le), q_le, 0.0), 0.0, None)
    if max_evaporative_fraction is not None:
        cap = max_evaporative_fraction * np.maximum(R_net_w_m2, 0.0)
        q_le = np.minimum(q_le, cap)
    return q_le
