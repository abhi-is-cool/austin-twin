"""Batched PyTorch port of the slab surface-temperature simulator.

This is a faithful re-implementation of `simulator.run()` with two changes
that matter for scale:

  1. **Batched over parameter sets.** Every tunable coefficient carries a
     leading batch dimension `B`, so an entire differential-evolution
     population (or a Monte-Carlo ensemble) integrates in one pass. A
     54x54 Austin grid cannot saturate a GPU on its own; a batch of
     hundreds of them can.

  2. **No history tensor.** The only frame any downstream metric needs is
     the citywide-mean-hottest frame of the second simulated day, so the
     peak frame is tracked incrementally. Memory is O(B*ny*nx) instead of
     O(B*n_steps*ny*nx), which is what makes fine grids affordable.

Everything else is deliberately identical to the NumPy path, including the
Laplacian's zeroed corners (see `_laplacian`). `tests/test_torch_parity.py`
pins the two implementations together.

The forward pass is autograd-safe: all coefficient tensors flow through
differentiably, so `torch.autograd.grad(rmse, coefs)` works. That is the
property the NumPy version cannot provide and the reason for this port.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from .simulator import SimConfig

# The five coefficients `calibration.COEF_NAMES` tunes, in the same order.
COEF_NAMES = (
    "absorption_impervious",
    "absorption_vegetation",
    "absorption_water",
    "et_coeff",
    "diffusion_m2_s",
)


@dataclass
class StaticFields:
    """Per-cell inputs that do not vary across the batch (one city, one grid)."""

    impervious: torch.Tensor  # (ny, nx)
    vegetation: torch.Tensor  # (ny, nx)
    water: torch.Tensor  # (ny, nx)
    city: torch.Tensor  # (ny, nx) bool
    albedo_boost: torch.Tensor | None  # (ny, nx) or None
    dx_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.impervious.shape)  # type: ignore[return-value]


@dataclass
class ForcingSeries:
    """Forcing pre-sampled onto the integrator's own timestep grid.

    `simulator.run()` calls `Forcing.sample(t)` once per step, which is a
    scalar `np.interp` per field. Pre-sampling hoists that out of the loop
    and keeps the GPU path free of host round-trips.
    """

    solar_w_m2: torch.Tensor  # (n_steps,)
    t_air_c: torch.Tensor  # (n_steps,)
    t_air_mean_c: float  # initial condition


def _laplacian(T: torch.Tensor, corner_mask: torch.Tensor) -> torch.Tensor:
    """5-point Laplacian, reflective (Neumann) edges, zeroed corners.

    Replicate padding reproduces the NumPy edge rows exactly: a ghost cell
    equal to its interior neighbour turns the 5-point stencil into the
    3-neighbour form `simulator._laplacian` writes by hand.

    The four corners are a genuine quirk of the NumPy implementation rather
    than a physical choice — it allocates `L = np.zeros_like(T)` and its four
    edge assignments all exclude the corners, so they stay 0 instead of
    getting the 2-neighbour stencil replicate padding would give them. They
    sit far outside every city polygon we run, so the choice is immaterial
    physically, but reproducing it is what lets the parity test assert
    agreement rather than approximate agreement.
    """
    Tp = F.pad(T.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    L = (
        Tp[:, :-2, 1:-1]
        + Tp[:, 2:, 1:-1]
        + Tp[:, 1:-1, :-2]
        + Tp[:, 1:-1, 2:]
        - 4.0 * T
    )
    # Multiplicative rather than in-place assignment so autograd stays happy.
    return L * corner_mask


def _step(
    T: torch.Tensor,
    S: torch.Tensor,
    T_air: torch.Tensor,
    absorption: torch.Tensor,
    vegetation: torch.Tensor,
    water: torch.Tensor,
    k_et: torch.Tensor,
    cell_capacity_eff: torch.Tensor,
    diffusion_factor: torch.Tensor,
    corner_mask: torch.Tensor,
    lw_coeff: float,
    water_evap_w_m2: float,
    dt: float,
) -> torch.Tensor:
    """One explicit Euler step of the slab energy balance.

    Factored out of `run_batch` purely so it can be handed to
    `torch.compile`: eager mode issues roughly a dozen tiny elementwise
    kernels per step, and at Austin's grid size the launch overhead of
    those dwarfs their arithmetic. Fusing them into one kernel is worth
    about 4x, and the fusion is what makes fine grids practical.
    """
    Q_solar = absorption * S
    Q_lw = lw_coeff * (T - T_air)
    Q_et = k_et * vegetation * (T - T_air).clamp(min=0.0)
    Q_evap_water = water_evap_w_m2 * water
    Q_net = Q_solar - Q_lw - Q_et - Q_evap_water

    dT_energy = Q_net * dt / cell_capacity_eff
    dT_diff = diffusion_factor * _laplacian(T, corner_mask)
    return T + dT_energy + dT_diff


_COMPILED_STEP = None


def _get_step(compile_step: bool):
    """Return the step function, compiling (and caching) on first request."""
    global _COMPILED_STEP
    if not compile_step:
        return _step
    if _COMPILED_STEP is None:
        # No cudagraphs: the 288-step chain feeds each output straight into
        # the next input, which cudagraph trees invalidate between replays.
        # Fusion is where essentially all of the win is anyway.
        _COMPILED_STEP = torch.compile(_step, mode="max-autotune-no-cudagraphs")
    return _COMPILED_STEP


def _corner_mask(ny: int, nx: int, **kw) -> torch.Tensor:
    m = torch.ones((1, ny, nx), **kw)
    m[0, 0, 0] = 0.0
    m[0, 0, -1] = 0.0
    m[0, -1, 0] = 0.0
    m[0, -1, -1] = 0.0
    return m


def static_fields_from_landuse(
    landuse: xr.Dataset,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> StaticFields:
    """Move the land-cover channels of a landuse Dataset onto the device."""
    def _t(name: str) -> torch.Tensor:
        return torch.as_tensor(
            landuse[name].values.astype(np.float32), device=device, dtype=dtype
        )

    x = landuse["x"].values
    dx = float(abs(x[1] - x[0]))

    boost = _t("albedo_boost") if "albedo_boost" in landuse else None

    return StaticFields(
        impervious=_t("impervious_frac"),
        vegetation=_t("vegetation_frac"),
        water=_t("water_mask"),
        city=torch.as_tensor(
            landuse["city_mask"].values.astype(bool), device=device
        ),
        albedo_boost=boost,
        dx_m=dx,
    )


def forcing_series(
    forcing,
    cfg: SimConfig,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ForcingSeries:
    """Pre-sample a `Forcing` onto the integrator timestep grid.

    Interpolation is done in float64 by `np.interp` and then narrowed, which
    is exactly what the NumPy path does: `Forcing.sample` returns Python
    floats and NEP 50 weak-scalar promotion narrows them at the multiply.
    """
    n_steps = int(cfg.duration_hours * 3600.0 / cfg.dt_seconds)
    t_h = np.arange(n_steps, dtype=np.float64) * cfg.dt_seconds / 3600.0
    t_clipped = np.clip(t_h, forcing.times_hours[0], forcing.times_hours[-1])

    solar = np.interp(t_clipped, forcing.times_hours, forcing.solar_w_m2)
    t_air = np.interp(t_clipped, forcing.times_hours, forcing.t_air_c)

    return ForcingSeries(
        solar_w_m2=torch.as_tensor(solar.astype(np.float32), device=device, dtype=dtype),
        t_air_c=torch.as_tensor(t_air.astype(np.float32), device=device, dtype=dtype),
        t_air_mean_c=float(np.mean(forcing.t_air_c)),
    )


def _as_batch(
    value, B: int, device, dtype, name: str
) -> torch.Tensor:
    t = torch.as_tensor(value, device=device, dtype=dtype)
    if t.ndim == 0:
        t = t.expand(B).clone()
    if t.shape != (B,):
        raise ValueError(f"coefficient {name!r} has shape {tuple(t.shape)}, expected ({B},)")
    return t


def run_batch(
    fields: StaticFields,
    coefs: Mapping[str, torch.Tensor | float],
    forcing: ForcingSeries,
    cfg: SimConfig,
    check_cfl: bool = True,
    compile_step: bool = False,
) -> torch.Tensor:
    """Integrate `B` parameter sets forward and return their peak frames.

    Returns a `(B, ny, nx)` tensor: the citywide-mean-hottest frame of the
    second simulated day, with cells outside the city set to NaN — the same
    field `calibration._peak_frame` extracts from a `SimResult`.

    `coefs` maps any of `COEF_NAMES` to either a scalar or a `(B,)` tensor;
    anything absent falls back to the value on `cfg`. Passing tensors with
    `requires_grad=True` makes the returned frame differentiable in them.

    Raises on a CFL violation only when `check_cfl` is True. Calibration
    drivers want `False` plus a finite-mask check, so that one bad member of
    a population does not abort the whole batch.

    `compile_step=True` routes the inner step through `torch.compile`. It
    costs a one-off compilation of roughly a minute and pays back about 4x
    per run thereafter, so it is worth it for calibration sweeps and fine
    grids but not for a single coarse simulation.
    """
    if cfg.use_pm_et:
        raise NotImplementedError(
            "the Penman-Monteith ET path is not ported yet; use the NumPy "
            "simulator for PM runs (see simulator.run(use_pm_et=True))"
        )

    device = fields.impervious.device
    dtype = fields.impervious.dtype

    # Batch width is set by the widest coefficient supplied.
    B = 1
    for name in COEF_NAMES:
        v = coefs.get(name)
        if isinstance(v, torch.Tensor) and v.ndim == 1:
            B = max(B, int(v.shape[0]))

    def _coef(name: str) -> torch.Tensor:
        v = coefs.get(name, getattr(cfg, name))
        return _as_batch(v, B, device, dtype, name)

    a_imp = _coef("absorption_impervious")
    a_veg = _coef("absorption_vegetation")
    a_wat = _coef("absorption_water")
    k_et = _coef("et_coeff")
    D = _coef("diffusion_m2_s")

    ny, nx = fields.shape
    b1 = (B, 1, 1)

    # --- per-cell composite absorption, one map per batch member ---
    absorption = (
        a_imp.view(b1) * fields.impervious
        + a_veg.view(b1) * fields.vegetation
        + a_wat.view(b1) * fields.water
    )
    # Cells with no land class (outside the city) fall back to the vegetation
    # coefficient, matching `np.where(absorption > 0, absorption, a_veg)`.
    absorption = torch.where(absorption > 0, absorption, a_veg.view(b1))

    if fields.albedo_boost is not None:
        absorption = absorption * (1.0 - fields.albedo_boost.clamp(0.0, 1.0))

    # --- CFL ---
    dx = fields.dx_m
    diffusion_factor = D * (cfg.dt_seconds / (dx * dx))  # (B,)
    if check_cfl and bool((diffusion_factor > 0.25).any()):
        worst = float(diffusion_factor.max())
        raise ValueError(
            f"explicit-scheme CFL violated: max D*dt/dx^2 = {worst:.3f} > 0.25. "
            f"Reduce diffusion_m2_s, reduce dt_seconds, or coarsen the grid."
        )

    # --- per-cell effective heat capacity (shallow mixed-layer water) ---
    cell_capacity_eff = cfg.cell_heat_capacity * (
        1.0 + fields.water * (cfg.water_heat_capacity_ratio - 1.0)
    )

    corner_mask = _corner_mask(ny, nx, device=device, dtype=dtype)
    city_f = fields.city.to(dtype)
    n_city = float(fields.city.sum())

    T = torch.full((B, ny, nx), forcing.t_air_mean_c, device=device, dtype=dtype)

    n_steps = int(cfg.duration_hours * 3600.0 / cfg.dt_seconds)
    best_mean = torch.full((B,), -float("inf"), device=device, dtype=dtype)
    # NaN rather than zeros: a member that never records a finite peak (i.e.
    # one that diverged under check_cfl=False) must come back non-finite so
    # the caller's isfinite guard rejects it. Zeros would be scored as a
    # legitimate constant field. The NumPy path cannot reach this state at
    # all — it raises on CFL violation instead of integrating — so there is
    # no parity obligation here, only a safety one.
    best_T = torch.full_like(T, float("nan"))

    dt = cfg.dt_seconds
    step_fn = _get_step(compile_step)
    k_et_b = k_et.view(b1)
    dfac_b = diffusion_factor.view(b1)

    for step in range(n_steps):
        T = step_fn(
            T,
            forcing.solar_w_m2[step],
            forcing.t_air_c[step],
            absorption,
            fields.vegetation,
            fields.water,
            k_et_b,
            cell_capacity_eff,
            dfac_b,
            corner_mask,
            cfg.lw_coeff,
            cfg.water_evap_w_m2,
            dt,
        )

        # Peak frame is only ever sought in the second simulated day.
        if (step + 1) * dt / 3600.0 >= 24.0:
            m = (T * city_f).sum(dim=(1, 2)) / n_city
            # A non-finite mean means at least one city cell is non-finite,
            # so the isfinite guard also keeps a partially-diverged member
            # from banking a garbage frame that happens to compare high.
            newer = (m > best_mean) & torch.isfinite(m)
            best_mean = torch.where(newer, m, best_mean)
            best_T = torch.where(newer.view(B, 1, 1), T, best_T)

    nan = torch.tensor(float("nan"), device=device, dtype=dtype)
    return torch.where(fields.city.unsqueeze(0), best_T, nan)


def anomaly_stats(
    sim: torch.Tensor,
    obs: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched anomaly RMSE and Pearson r over `valid` cells.

    Mirrors `calibration._anomaly_stats`: both fields are mean-subtracted
    over the jointly-valid cells before comparison, so a uniform offset
    between simulator and satellite does not enter the score.

    `sim` is (B, ny, nx), `obs` is (ny, nx), `valid` is a (ny, nx) bool mask.
    Returns two `(B,)` tensors.
    """
    s = sim[:, valid]  # (B, n_valid)
    o = obs[valid]  # (n_valid,)
    s = s - s.mean(dim=1, keepdim=True)
    o = o - o.mean()

    rmse = torch.sqrt(((s - o) ** 2).mean(dim=1))

    s_sd = s.std(dim=1, unbiased=False)
    o_sd = o.std(unbiased=False)
    cov = (s * o).mean(dim=1)
    r = torch.where(
        (s_sd > 0) & (o_sd > 0), cov / (s_sd * o_sd), torch.zeros_like(cov)
    )
    return rmse, r


def valid_mask(
    sim_like_city: torch.Tensor, obs: torch.Tensor
) -> torch.Tensor:
    """Cells where both the simulator and the observation are finite.

    The simulator is NaN exactly outside the city mask and MODIS is NaN
    outside the city or under cloud, so this is shared across the batch and
    worth hoisting out of the objective.
    """
    return torch.isfinite(sim_like_city) & torch.isfinite(obs)
