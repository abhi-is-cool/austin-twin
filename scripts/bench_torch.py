"""Benchmark the batched torch simulator against the NumPy reference.

The question this answers is not "is the GPU faster" but "at what scale does
the GPU start to matter", which is the only thing that decides whether a
CUDA port is worth building a research programme on.

Reports cell-updates/s so the resolution sweep is comparable across grids.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from austin_twin import simulator_torch as st
from austin_twin.calibration import _peak_frame
from austin_twin.forcing import Forcing
from austin_twin.grid import CityGrid
from austin_twin.simulator import SimConfig, run, safe_dt
from austin_twin.synthetic import generate_landuse

# Austin's real extent, ~27 km across; the cell count is what changes.
EXTENT_M = 27_000.0
N_STEPS = 288  # 48 h at dt = 600 s, the published configuration


def make_grid(res_m: float) -> CityGrid:
    n = int(round(EXTENT_M / res_m))
    x = np.arange(n, dtype=float) * res_m + 620_000.0
    y = np.arange(n, dtype=float)[::-1] * res_m + 3_330_000.0
    xx, yy = np.meshgrid((np.arange(n) - n / 2) / (n / 2),
                         (np.arange(n) - n / 2) / (n / 2))
    mask = (xx ** 2 + (yy / 0.85) ** 2) < 0.9
    return CityGrid(boundary=None, resolution_m=res_m, x=x, y=y, mask=mask)


def cfl_safe_D(res_m: float, dt: float) -> float:
    """Largest D that keeps the explicit scheme stable at this dx and dt."""
    return 0.20 * res_m * res_m / dt


def timed(fn, warmup: int = 1, repeat: int = 3, cuda: bool = False) -> float:
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    has_cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"device {torch.cuda.get_device_name(0) if has_cuda else 'CPU only'}")
    if has_cuda:
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"compute capability sm_{cap[0]}{cap[1]} | {vram:.1f} GB VRAM")
    print()

    cfg = SimConfig(frame_stride=1, duration_hours=48.0, dt_seconds=600.0)

    # ---------- 1. single run at the published 500 m grid ----------
    print("=" * 74)
    print("1. SINGLE 48-h RUN, 500 m grid (the published configuration)")
    print("=" * 74)
    grid = make_grid(500.0)
    landuse = generate_landuse(grid, seed=0)
    forcing = Forcing.synthetic_diurnal(
        duration_hours=cfg.duration_hours,
        air_temp_mean_c=cfg.air_temp_c,
        air_temp_amplitude_c=cfg.diurnal_amplitude_c,
        peak_solar_w_m2=cfg.peak_solar_w_m2,
    )
    ny, nx = grid.shape
    print(f"grid {ny}x{nx} = {ny * nx} cells, {int(grid.mask.sum())} inside city\n")

    t_np = timed(lambda: run(landuse, cfg, forcing=forcing), warmup=1, repeat=3)
    print(f"  numpy              {t_np * 1e3:8.1f} ms")

    for dev in (["cpu", "cuda"] if has_cuda else ["cpu"]):
        fields = st.static_fields_from_landuse(landuse, device=dev)
        fs = st.forcing_series(forcing, cfg, device=dev)
        t = timed(lambda: st.run_batch(fields, {}, fs, cfg),
                  cuda=(dev == "cuda"))
        print(f"  torch {dev:<5}        {t * 1e3:8.1f} ms   "
              f"({t_np / t:5.2f}x vs numpy)")

    # ---------- 2. batch scaling ----------
    print()
    print("=" * 74)
    print("2. BATCH SCALING at 500 m  (a DE population in one pass)")
    print("=" * 74)
    print(f"{'batch':>7} | {'cpu ms':>9} {'cpu sim/s':>11} | "
          f"{'cuda ms':>9} {'cuda sim/s':>11} | "
          f"{'+comp ms':>9} {'+comp sim/s':>11} | "
          f"{'vs cpu':>7} {'vs numpy':>9}")
    print("-" * 104)

    for B in (1, 16, 64, 256, 1024):
        row = {}
        for dev in (["cpu", "cuda"] if has_cuda else ["cpu"]):
            fields = st.static_fields_from_landuse(landuse, device=dev)
            fs = st.forcing_series(forcing, cfg, device=dev)
            coefs = {
                "diffusion_m2_s": torch.linspace(1.0, 90.0, B, device=dev),
                "et_coeff": torch.linspace(5.0, 60.0, B, device=dev),
            }
            t = timed(lambda: st.run_batch(fields, coefs, fs, cfg),
                      warmup=1, repeat=3, cuda=(dev == "cuda"))
            row[dev] = t
            if dev == "cuda":
                tc = timed(
                    lambda: st.run_batch(fields, coefs, fs, cfg,
                                         compile_step=True),
                    warmup=1, repeat=3, cuda=True)
                row["cuda+compile"] = tc
        cpu_t = row["cpu"]
        if has_cuda:
            cu_t, cc_t = row["cuda"], row["cuda+compile"]
            # Throughput ratios, not wall-time ratios: numpy runs one sim
            # per t_np, the batch runs B sims per cc_t.
            vs_cpu = (B / cc_t) / (B / cpu_t)
            vs_numpy = (B / cc_t) / (1.0 / t_np)
            print(f"{B:>7} | {cpu_t * 1e3:9.1f} {B / cpu_t:11.1f} | "
                  f"{cu_t * 1e3:9.1f} {B / cu_t:11.1f} | "
                  f"{cc_t * 1e3:9.1f} {B / cc_t:11.1f} | "
                  f"{vs_cpu:6.1f}x {vs_numpy:8.0f}x")
        else:
            print(f"{B:>7} | {cpu_t * 1e3:9.1f} {B / cpu_t:11.1f}")

    # ---------- 3. resolution scaling ----------
    print()
    print("=" * 74)
    print("3. RESOLUTION SCALING  (B=32; D pinned to the CFL limit per dx)")
    print("=" * 74)
    print(f"{'res':>6} {'grid':>12} {'cells':>10} | {'cpu ms':>9} | "
          f"{'cuda ms':>9} {'speedup':>8} | {'cuda Mcell-upd/s':>17}")
    print("-" * 88)

    B = 32
    for res in (500.0, 250.0, 100.0, 30.0):
        g = make_grid(res)
        lu = generate_landuse(g, seed=0)
        gy, gx = g.shape
        D = cfl_safe_D(res, cfg.dt_seconds)
        row = {}
        for dev in (["cpu", "cuda"] if has_cuda else ["cpu"]):
            fields = st.static_fields_from_landuse(lu, device=dev)
            fs = st.forcing_series(forcing, cfg, device=dev)
            coefs = {"diffusion_m2_s": torch.full((B,), D, device=dev)}
            # 30 m on CPU is slow enough that repeats are not worth the wall time.
            rep = 1 if (dev == "cpu" and res <= 100.0) else 3
            t = timed(lambda: st.run_batch(fields, coefs, fs, cfg,
                                           compile_step=(dev == "cuda")),
                      warmup=1, repeat=rep, cuda=(dev == "cuda"))
            row[dev] = t
        cells = gy * gx
        if has_cuda:
            upd = B * cells * N_STEPS / row["cuda"] / 1e6
            print(f"{res:>5.0f}m {f'{gy}x{gx}':>12} {cells:>10,} | "
                  f"{row['cpu'] * 1e3:9.1f} | {row['cuda'] * 1e3:9.1f} "
                  f"{row['cpu'] / row['cuda']:7.1f}x | {upd:17.0f}")
        else:
            print(f"{res:>5.0f}m {f'{gy}x{gx}':>12} {cells:>10,} | "
                  f"{row['cpu'] * 1e3:9.1f}")

    if has_cuda:
        print()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak VRAM allocated across all benchmarks: {peak:.2f} GB "
              f"of {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---------- 4. what a real 30 m run actually costs ----------
    print()
    print("=" * 74)
    print("4. CFL REALITY CHECK at the literature-anchored D = 1.0 m^2/s")
    print("=" * 74)
    for res in (500.0, 100.0, 30.0):
        dt = safe_dt(1.0, res)
        steps = int(48 * 3600 / min(dt, 600.0))
        print(f"  {res:>5.0f} m : max stable dt = {min(dt, 600.0):8.1f} s "
              f"-> {steps:6,} steps per 48 h")


if __name__ == "__main__":
    main()
