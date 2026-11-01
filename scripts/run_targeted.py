"""Zonal sensitivity analysis: where in Austin does canopy planting help most?

Partitions the WorldCover-driven baseline into a 6x5 zonal grid (~25-30 zones
covering the irregular city footprint), runs +20% canopy in one zone at a time,
and ranks zones by cooling efficiency (citywide °C·m² of cooling per m² of
impervious converted).
"""
from __future__ import annotations

from pathlib import Path

from austin_twin.calibration import load_calibrated_config
from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.sensitivity import (
    evaluate_zones, format_ranking, make_grid_zones, rank_zones,
)
from austin_twin.simulator import SimConfig, run
from austin_twin.viz import plot_zone_sensitivity
from austin_twin.worldcover import build_worldcover_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CAL = OUT / "calibrated_config.json"


def main() -> None:
    print("[1/4] Austin grid + WorldCover land cover...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = build_worldcover_landuse(boundary, grid, cache_dir=RAW / "worldcover")

    print("[2/4] baseline 48 h simulation (one time)...")
    base = SimConfig(duration_hours=48.0)
    if CAL.exists():
        print(f"      using calibrated coefficients from {CAL.name}")
        cfg = load_calibrated_config(CAL, base_config=base)
    else:
        print("      using default SimConfig (no calibration file found)")
        cfg = base
    baseline = run(landuse, cfg)

    print("[3/4] running zonal sensitivities...")
    zones = make_grid_zones(grid, n_x=6, n_y=5, min_cells=6)
    print(f"      {len(zones)} zones with >=6 city cells")
    metrics = evaluate_zones(landuse, grid, zones, baseline, cfg)

    print("[4/4] writing outputs...")
    plot_zone_sensitivity(metrics, landuse, OUT / "zone_sensitivity.png", top_n=3)

    table = format_ranking(metrics)
    print()
    print(table)
    (OUT / "zone_ranking.txt").write_text(table + "\n")

    # Brief narrative of top zones.
    ranked = rank_zones(metrics)
    print("\nTop-3 zones for canopy investment in Austin (by citywide cooling efficiency):")
    for i, m in enumerate(ranked[:3], start=1):
        area_ha = m.area_converted_m2 / 1e4
        print(
            f"  #{i}  zone {m.zone.label}: convert {area_ha:,.1f} ha of impervious -> "
            f"local cooling {m.local_mean_dt:+.2f}°C, "
            f"citywide cooling {m.citywide_mean_dt:+.3f}°C, "
            f"efficiency {m.efficiency:.2e} °C·m² per m²"
        )

    print(f"\ndone. wrote {OUT / 'zone_sensitivity.png'} and {OUT / 'zone_ranking.txt'}.")


if __name__ == "__main__":
    main()
