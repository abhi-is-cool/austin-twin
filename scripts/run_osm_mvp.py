"""Run the simulator on OSM-derived land use and compare to the synthetic MVP.

First run pulls buildings/roads/water/parks from OSM via Overpass — this is
slow (minutes). Subsequent runs read cached GeoParquet from data/raw/osm/.
"""
from __future__ import annotations

from pathlib import Path

from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.osm_landuse import build_osm_landuse
from austin_twin.simulator import SimConfig, run
from austin_twin.synthetic import generate_landuse
from austin_twin.viz import animate_temperature, plot_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"


def main() -> None:
    print("[1/5] Austin boundary...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")

    print("[2/5] building 500m grid...")
    grid = build_grid(boundary, resolution_m=500.0)
    print(f"      grid shape: {grid.shape}, cells inside city: {int(grid.mask.sum())}")

    print("[3/5] OSM land use...")
    osm_lu = build_osm_landuse(boundary, grid, cache_dir=RAW / "osm")

    print("[4/5] synthetic land use (for comparison)...")
    syn_lu = generate_landuse(grid, seed=0)

    print("[5/5] simulating both + writing outputs...")
    plot_landuse(osm_lu, OUT / "landuse_osm.png")
    plot_landuse(syn_lu, OUT / "landuse_synthetic.png")

    cfg = SimConfig(duration_hours=48.0)
    osm_result = run(osm_lu, cfg)
    syn_result = run(syn_lu, cfg)

    animate_temperature(osm_result, osm_lu, OUT / "temperature_osm.gif", stride=6, fps=8)
    animate_temperature(syn_result, syn_lu, OUT / "temperature_synthetic.gif", stride=6, fps=8)

    # Quick numerical summary at peak heat.
    import numpy as np
    for label, lu, r in [("OSM", osm_lu, osm_result), ("synthetic", syn_lu, syn_result)]:
        T = r.temperature
        mean_t = np.nanmean(T.reshape(T.shape[0], -1), axis=1)
        peak_idx = int(np.nanargmax(mean_t))
        T_peak = T[peak_idx]
        finite = T_peak[np.isfinite(T_peak)]
        imp_mean = float(np.nanmean(lu["impervious_frac"].values[lu["city_mask"].values]))
        veg_mean = float(np.nanmean(lu["vegetation_frac"].values[lu["city_mask"].values]))
        water_cells = int(lu["water_mask"].values.sum())
        print(
            f"  {label:<10}  peak t={r.times_hours[peak_idx]:5.1f}h  "
            f"T min={finite.min():5.2f}  mean={finite.mean():5.2f}  max={finite.max():5.2f}  "
            f"UHI={finite.max()-finite.min():.2f}°C  "
            f"|  imp_mean={imp_mean:.2f}  veg_mean={veg_mean:.2f}  water_cells={water_cells}"
        )

    print(f"\ndone. outputs in {OUT}/.")


if __name__ == "__main__":
    main()
