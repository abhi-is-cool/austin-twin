"""Is the 363 m half-cooling distance structural, or a selection artifact?

The paper reports a patch half-cooling distance of 363 m against a
literature band of 100-300 m, and attributes the gap to the diffusion-anchor
trade-off. But `find_D_for_target_half_cooling` early-returns the smallest D
in its range whenever that D's half-cooling already exceeds the target, so it
never inspects the rest of the sweep -- and the recorded sweep is
non-monotonic at the low end (D=1.000 -> 363 m, D=1.427 -> 282 m).

Two things to establish:
  1. Is the low-D non-monotonicity real, or noise from fitting an exponential
     to very few distance bins (n_bins_fit = 4 at D = 1.0)?
  2. What half-cooling distance is actually reachable, and at what D?

Stage 1 uses synthetic forcing and is weather-independent, so this needs no
ERA5 and no network beyond the cached land cover.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from austin_twin.calibration import load_calibrated_config
from austin_twin.cities import AUSTIN
from austin_twin.cooling_response import diffusion_decay_sweep
from austin_twin.grid import build_grid, fetch_boundary
from austin_twin.simulator import SimConfig, safe_dt
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

TARGET = 200.0
LIT_BAND = (100.0, 300.0)


def main() -> None:
    boundary = fetch_boundary(AUSTIN, cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, RAW,
                                       tile=AUSTIN.worldcover_tile)

    base = replace(
        load_calibrated_config(OUT / "calibrated_config_two_stage.json",
                               base_config=SimConfig(duration_hours=48.0)),
        use_pm_et=False,
    )

    # Dense low-D sweep, extended below the published search floor of 1.0 so
    # the advisor's "go below the floor" question is answered too. CFL is an
    # upper bound on D, so smaller values are always stable.
    Ds = sorted(set(
        [round(float(d), 4) for d in np.geomspace(0.10, 6.0, 28)]
        + [1.0, 1.427, 2.037, 2.906, 4.148]  # published sweep points
    ))
    print(f"CFL ceiling at dx=500 m, dt=600 s: D <= "
          f"{0.25 * 500 * 500 / 600:.1f} m2/s (all sweep values safe)\n")

    sweep = diffusion_decay_sweep(landuse, base, Ds)

    print(f"{'D':>8} {'half-cool m':>12} {'lambda m':>10} "
          f"{'peak dT C':>10} {'bins':>5}  {'in lit band':>11}")
    print("-" * 66)
    rows = []
    for s in sweep:
        hc = s["half_cooling_m"]
        inband = (np.isfinite(hc) and LIT_BAND[0] <= hc <= LIT_BAND[1])
        print(f"{s['D']:>8.3f} {hc:>12.1f} {s['decay_length_m']:>10.1f} "
              f"{s['peak_dt_c']:>10.2f} {s['n_bins_fit']:>5d}  "
              f"{'YES' if inband else '':>11}")
        rows.append(s)

    valid = [s for s in sweep if np.isfinite(s["half_cooling_m"])]
    best = min(valid, key=lambda s: abs(s["half_cooling_m"] - TARGET))
    inband = [s for s in valid if LIT_BAND[0] <= s["half_cooling_m"] <= LIT_BAND[1]]

    print()
    print("=" * 66)
    print(f"closest to {TARGET:.0f} m target : D = {best['D']:.3f}  "
          f"-> {best['half_cooling_m']:.1f} m  (bins={best['n_bins_fit']})")
    print(f"published ship selection  : D = 1.000  -> "
          f"{[s for s in valid if abs(s['D']-1.0) < 1e-6][0]['half_cooling_m']:.1f} m")
    print(f"values inside {LIT_BAND[0]:.0f}-{LIT_BAND[1]:.0f} m band : "
          f"{len(inband)} of {len(valid)}")
    if inband:
        lo = min(inband, key=lambda s: s["D"])
        hi = max(inband, key=lambda s: s["D"])
        print(f"  D range achieving in-band: {lo['D']:.3f} .. {hi['D']:.3f}")

    # Monotonicity check over the low-D region only.
    low = [s for s in valid if s["D"] <= 5.0]
    hcs = [s["half_cooling_m"] for s in low]
    diffs = np.diff(hcs)
    n_down = int((diffs < 0).sum())
    print()
    print(f"monotonicity over D <= 5: {n_down} of {len(diffs)} steps decrease "
          f"({'NON-MONOTONIC' if n_down else 'monotonic increasing'})")
    thin = [s for s in low if s["n_bins_fit"] <= 4]
    print(f"points fit on <=4 distance bins (fragile): {len(thin)} of {len(low)}")

    (OUT / "diffusion_anchor_diag.json").write_text(json.dumps({
        "target_m": TARGET,
        "lit_band_m": LIT_BAND,
        "sweep": rows,
        "best_D": best["D"],
        "best_half_cooling_m": best["half_cooling_m"],
        "n_in_band": len(inband),
    }, indent=2))
    print(f"\nwrote outputs/diffusion_anchor_diag.json")


if __name__ == "__main__":
    main()
