"""Pin the PyTorch port to the NumPy simulator.

These tests are deliberately offline: they build a grid and land-cover
pattern analytically instead of fetching OSM/WorldCover, so parity is
testable in CI and on a machine with no credentials. Reproducing the
*published* scores against real MODIS is a separate script — this file
only asserts that the two integrators compute the same thing.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from austin_twin import simulator_torch as st
from austin_twin.calibration import _anomaly_stats, _peak_frame
from austin_twin.forcing import Forcing
from austin_twin.grid import CityGrid
from austin_twin.simulator import SimConfig, run
from austin_twin.synthetic import generate_landuse

RES_M = 500.0
NY, NX = 54, 54


def _offline_grid() -> CityGrid:
    """An Austin-sized grid with an elliptical city mask and no network I/O.

    `generate_landuse` only touches `.shape`, `.x`, `.y` and `.mask`, so the
    boundary GeoDataFrame the real `build_grid` would attach is not needed.
    """
    x = np.arange(NX, dtype=float) * RES_M + 620_000.0
    y = np.arange(NY, dtype=float)[::-1] * RES_M + 3_330_000.0
    xx, yy = np.meshgrid(
        (np.arange(NX) - NX / 2) / (NX / 2), (np.arange(NY) - NY / 2) / (NY / 2)
    )
    mask = (xx**2 + (yy / 0.85) ** 2) < 0.9
    return CityGrid(boundary=None, resolution_m=RES_M, x=x, y=y, mask=mask)


@pytest.fixture(scope="module")
def landuse():
    return generate_landuse(_offline_grid(), seed=0)


@pytest.fixture(scope="module")
def cfg():
    # frame_stride=1 so the NumPy peak-frame search sees every step, which is
    # what the torch path tracks incrementally.
    return SimConfig(duration_hours=48.0, dt_seconds=600.0, frame_stride=1)


@pytest.fixture(scope="module")
def forcing(cfg):
    return Forcing.synthetic_diurnal(
        duration_hours=cfg.duration_hours,
        air_temp_mean_c=cfg.air_temp_c,
        air_temp_amplitude_c=cfg.diurnal_amplitude_c,
        dewpoint_c=cfg.default_dewpoint_c,
        wind_speed_m_s=cfg.default_wind_speed_m_s,
        peak_solar_w_m2=cfg.peak_solar_w_m2,
        pressure_pa=cfg.default_pressure_pa,
    )


def _numpy_peak(landuse, cfg, forcing, **overrides) -> np.ndarray:
    from dataclasses import replace

    result = run(landuse, replace(cfg, **overrides), forcing=forcing)
    return _peak_frame(result.temperature, result.times_hours)


def _torch_peak(landuse, cfg, forcing, device, coefs) -> np.ndarray:
    fields = st.static_fields_from_landuse(landuse, device=device)
    fs = st.forcing_series(forcing, cfg, device=device)
    out = st.run_batch(fields, coefs, fs, cfg)
    return out.detach().cpu().numpy()


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_peak_frame_matches_numpy(landuse, cfg, forcing, device):
    """A single default-coefficient run must agree cell-for-cell."""
    ref = _numpy_peak(landuse, cfg, forcing)
    got = _torch_peak(landuse, cfg, forcing, device, {})[0]

    assert np.isnan(ref).sum() == np.isnan(got).sum()
    both = np.isfinite(ref) & np.isfinite(got)
    assert both.sum() > 1000, "city mask should leave plenty of valid cells"

    diff = np.abs(ref[both] - got[both])
    # float32 accumulation over 288 steps in a different reduction order;
    # a millikelvin ceiling is far below any physically meaningful signal.
    assert diff.max() < 1e-3, f"max |diff| = {diff.max():.3e} °C"


@pytest.mark.parametrize("device", DEVICES)
def test_batch_matches_individual_runs(landuse, cfg, forcing, device):
    """B parameter sets in one pass == B separate NumPy runs."""
    members = [
        dict(absorption_impervious=0.90, absorption_vegetation=0.55,
             absorption_water=0.08, et_coeff=25.0, diffusion_m2_s=50.0),
        dict(absorption_impervious=0.80, absorption_vegetation=0.43,
             absorption_water=0.18, et_coeff=11.0, diffusion_m2_s=1.0),
        dict(absorption_impervious=0.95, absorption_vegetation=0.31,
             absorption_water=0.05, et_coeff=60.0, diffusion_m2_s=18.3),
        dict(absorption_impervious=0.85, absorption_vegetation=0.65,
             absorption_water=0.12, et_coeff=5.0, diffusion_m2_s=90.0),
    ]
    refs = np.stack([_numpy_peak(landuse, cfg, forcing, **m) for m in members])

    batched = {
        name: torch.tensor([m[name] for m in members], dtype=torch.float32,
                           device=device)
        for name in st.COEF_NAMES
    }
    got = _torch_peak(landuse, cfg, forcing, device, batched)

    assert got.shape == refs.shape
    both = np.isfinite(refs) & np.isfinite(got)
    diff = np.abs(refs[both] - got[both])
    assert diff.max() < 1e-3, f"max |diff| = {diff.max():.3e} °C"


@pytest.mark.parametrize("device", DEVICES)
def test_anomaly_stats_matches_numpy(landuse, cfg, forcing, device):
    """The scoring function must agree with `calibration._anomaly_stats`."""
    sim = _numpy_peak(landuse, cfg, forcing)

    # A stand-in observation: the simulator field warped and noised, with a
    # patch of cloud knocked out, so the valid mask is a strict subset.
    rng = np.random.default_rng(1)
    obs = 1.3 * sim + 2.0 + rng.normal(0, 0.4, sim.shape).astype(np.float32)
    obs[10:18, 22:34] = np.nan

    ref_rmse, ref_r, _, _ = _anomaly_stats(sim, obs)

    sim_t = torch.as_tensor(sim, device=device)
    obs_t = torch.as_tensor(obs, device=device)
    valid = st.valid_mask(sim_t, obs_t)
    rmse, r = st.anomaly_stats(sim_t.unsqueeze(0), obs_t, valid)

    assert float(rmse[0]) == pytest.approx(ref_rmse, abs=1e-4)
    assert float(r[0]) == pytest.approx(ref_r, abs=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_gradients_flow_to_all_coefficients(landuse, cfg, forcing, device):
    """The whole point of the port: d(score)/d(coefficient) must exist.

    Differential evolution cannot see these gradients, which is what caps
    the NumPy path at five scalar coefficients.
    """
    fields = st.static_fields_from_landuse(landuse, device=device)
    fs = st.forcing_series(forcing, cfg, device=device)

    coefs = {
        name: torch.tensor([float(getattr(cfg, name))], device=device,
                           requires_grad=True)
        for name in st.COEF_NAMES
    }
    sim = st.run_batch(fields, coefs, fs, cfg)

    rng = np.random.default_rng(2)
    obs_np = sim.detach().cpu().numpy()[0] + rng.normal(0, 0.5, fields.shape)
    obs = torch.as_tensor(obs_np.astype(np.float32), device=device)
    valid = st.valid_mask(sim[0], obs)

    rmse, _ = st.anomaly_stats(sim, obs, valid)
    rmse[0].backward()

    for name, t in coefs.items():
        assert t.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(t.grad).all(), f"non-finite gradient for {name}"
        assert float(t.grad.abs().max()) > 0.0, f"zero gradient for {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_step_matches_numpy(landuse, cfg, forcing):
    """Kernel fusion must not move the answer.

    Fusion reassociates the elementwise chain, so this is a slightly
    looser check than eager parity — but it still has to land far below
    any temperature difference that would change a conclusion.
    """
    ref = _numpy_peak(landuse, cfg, forcing)

    fields = st.static_fields_from_landuse(landuse, device="cuda")
    fs = st.forcing_series(forcing, cfg, device="cuda")
    got = st.run_batch(fields, {}, fs, cfg, compile_step=True)
    got = got.detach().cpu().numpy()[0]

    both = np.isfinite(ref) & np.isfinite(got)
    diff = np.abs(ref[both] - got[both])
    assert diff.max() < 1e-3, f"max |diff| = {diff.max():.3e} °C"


def test_cfl_violation_raises(landuse, cfg, forcing):
    fields = st.static_fields_from_landuse(landuse)
    fs = st.forcing_series(forcing, cfg)
    coefs = {"diffusion_m2_s": torch.tensor([500.0])}
    with pytest.raises(ValueError, match="CFL"):
        st.run_batch(fields, coefs, fs, cfg)


def test_cfl_check_can_be_disabled(landuse, cfg, forcing):
    """One diverged member must not abort the population, nor score as valid.

    A calibration objective rejects a member via `isfinite`, so a diverged
    run has to come back non-finite. Returning a constant field instead
    would be scored as a legitimate (merely bad) simulation.
    """
    fields = st.static_fields_from_landuse(landuse)
    fs = st.forcing_series(forcing, cfg)
    coefs = {"diffusion_m2_s": torch.tensor([50.0, 500.0])}
    out = st.run_batch(fields, coefs, fs, cfg, check_cfl=False)
    city = fields.city.numpy()
    assert np.isfinite(out[0].numpy()[city]).all(), "stable member stayed finite"
    assert not np.isfinite(out[1].numpy()[city]).any(), "diverged member is all-NaN"
