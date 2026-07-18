"""Run baseline + canonical counterfactual scenarios and emit comparison plots."""
from __future__ import annotations

from pathlib import Path

from austin_twin.calibration import load_calibrated_config
from austin_twin.counterfactual import CANONICAL_SCENARIOS, run_scenarios, summarize
from austin_twin.grid import build_grid, fetch_austin_boundary
from austin_twin.simulator import SimConfig
from austin_twin.synthetic import generate_landuse
from austin_twin.viz import plot_scenario_comparison

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CAL = OUT / "calibrated_config.json"


def main() -> None:
    print("[1/3] preparing baseline land-use...")
    boundary = fetch_austin_boundary(cache_path=RAW / "austin_boundary.geojson")
    grid = build_grid(boundary, resolution_m=500.0)
    landuse = generate_landuse(grid, seed=0)

    base = SimConfig(duration_hours=48.0)
    if CAL.exists():
        print(f"[2/3] using calibrated coefficients from {CAL.name}")
        cfg = load_calibrated_config(CAL, base_config=base)
    else:
        print("[2/3] using default SimConfig (no calibration file found)")
        cfg = base
    print(f"      running {len(CANONICAL_SCENARIOS)} scenarios x 48h each...")
    runs = run_scenarios(landuse, CANONICAL_SCENARIOS, config=cfg)

    print("[3/3] writing comparison plot + summary table...")
    plot_scenario_comparison(runs, OUT / "scenarios.png")

    table = summarize(runs)
    print()
    print(table)
    (OUT / "scenario_summary.txt").write_text(table + "\n")

    print(f"\ndone. wrote {OUT / 'scenarios.png'} and {OUT / 'scenario_summary.txt'}.")


if __name__ == "__main__":
    main()
