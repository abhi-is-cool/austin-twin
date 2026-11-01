"""End-to-end MVP: Austin boundary -> grid -> synthetic land-use -> simulator -> GIF."""
from __future__ import annotations

from pathlib import Path

from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.simulator import SimConfig, run
from austin_twin.synthetic import generate_landuse
from austin_twin.viz import animate_temperature, plot_landuse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"


def main() -> None:
    print("[1/4] fetching Austin city limits (cached after first run)...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")

    print("[2/4] building 500m grid over city limits...")
    grid = build_grid(boundary, resolution_m=500.0)
    print(f"      grid shape: {grid.shape}, cells inside city: {int(grid.mask.sum())}")

    print("[3/4] generating synthetic land-use channels...")
    landuse = generate_landuse(grid, seed=0)
    plot_landuse(landuse, OUT / "landuse.png")

    print("[4/4] running 48h baseline simulation...")
    result = run(landuse, SimConfig(duration_hours=48.0, dt_seconds=600.0))
    animate_temperature(result, landuse, OUT / "temperature.gif", stride=6, fps=8)

    print(f"done. wrote {OUT / 'landuse.png'} and {OUT / 'temperature.gif'}.")


if __name__ == "__main__":
    main()
