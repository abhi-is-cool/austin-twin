# austin-twin

Digital twin of Austin for urban heat dynamics. Pulls real city geography and
land cover, simulates surface temperature evolution with a physics-inspired
slab heat model, runs counterfactual urban-planning scenarios, and validates
the spatial UHI pattern against MODIS satellite observations.

Current results (ship config, linear-ET + ERA5 forcing per MODIS date):
simulator reproduces the Austin urban heat island with **Pearson r = +0.76 /
RMSE 1.43 °C** on held-out MODIS LST days (Aug 21-22, 2024) after
calibration on three other August 2024 days. Absolute surface temperatures
now line up too: sim mean 40.8 °C vs MODIS 47.6 °C on 2024-08-19, with
matching ~15 °C spatial spread — the synthetic-forcing anomaly
normalization is no longer required for interpretability.

Higher raw r on this test set is available under earlier configs
(single-stage synthetic, r = +0.86, RMSE 1.07 °C) that traded physical
plausibility (unrealistic diffusion, no mixed-layer water) for a tighter
anomaly fit; the ship config keeps the physical constraints from the
two-stage / mixed-layer / ERA5 upgrades. See
[NUMBERS.md §13](NUMBERS.md) for the full audit trail, including the
Penman-Monteith ET experiment recorded as a documented negative result.

## Layout

### Core modules ([src/austin_twin/](src/austin_twin/))

- [cities.py](src/austin_twin/cities.py) — per-city config for the multi-city transfer study (Austin, Phoenix, Denver, Miami-Dade): OSM query, UTM CRS, ERA5 bbox, WorldCover tile, UTC offset
- [grid.py](src/austin_twin/grid.py) — fetch a city-limits polygon (via OSM), build a 500 m UTM grid — multi-city aware via `fetch_boundary(city)`
- [synthetic.py](src/austin_twin/synthetic.py) — synthetic land-use channels for fast iteration / testing
- [osm_landuse.py](src/austin_twin/osm_landuse.py) — OSM-driven land use (buildings, roads, water, parks) cached as GeoPackage
- [worldcover.py](src/austin_twin/worldcover.py) — ESA WorldCover 10 m land cover via public AWS COG (no auth)
- [simulator.py](src/austin_twin/simulator.py) — slab heat equation: solar absorption / radiative loss / evapotranspiration / horizontal diffusion. CFL-stable, ~1 s per 48 h run. Accepts an optional `Forcing` object (ERA5 or synthetic).
- [forcing.py](src/austin_twin/forcing.py) — atmospheric forcing container. `Forcing.synthetic_diurnal(...)` reproduces the legacy sinusoidal cycle; `Forcing.from_era5(...)` loads ERA5 hourly t2m/d2m/wind/ssrd/sp for a chosen window.
- [penman_monteith.py](src/austin_twin/penman_monteith.py) — canonical FAO-56 PM latent-heat flux. **Not the ship default** (see NUMBERS.md §13 for the documented negative result); kept for reproducibility and future physics work.
- [era5.py](src/austin_twin/era5.py) — CDS-Beta ERA5 fetcher (lazy `cdsapi` import; requires `~/.cdsapirc`).
- [counterfactual.py](src/austin_twin/counterfactual.py) — `Scenario` API + canonical scenarios (canopy, cool roofs, river greenway, densification)
- [sensitivity.py](src/austin_twin/sensitivity.py) — zonal sensitivity analysis: where in Austin does canopy planting cool the most per acre converted?
- [modis.py](src/austin_twin/modis.py) — MODIS Aqua LST via Microsoft Planetary Computer STAC (no auth)
- [calibration.py](src/austin_twin/calibration.py) — multi-day, train/test calibration of simulator coefficients via differential evolution against MODIS anomaly RMSE
- [viz.py](src/austin_twin/viz.py) — land-use plots, temperature GIF, scenario comparison, zonal impact maps

### Scripts ([scripts/](scripts/))

- [run_mvp.py](scripts/run_mvp.py) — boundary → grid → synthetic land use → baseline simulation → GIF
- [run_counterfactuals.py](scripts/run_counterfactuals.py) — synthetic land use + 5 canonical intervention scenarios
- [run_osm_mvp.py](scripts/run_osm_mvp.py) — OSM-derived land use vs synthetic, side-by-side
- [run_worldcover_mvp.py](scripts/run_worldcover_mvp.py) — ESA WorldCover land use + counterfactuals on real Austin
- [run_targeted.py](scripts/run_targeted.py) — zonal sensitivity analysis (find best zones for canopy investment)
- [run_validation.py](scripts/run_validation.py) — simulator vs MODIS LST validation on 2024-08-19
- [run_calibration.py](scripts/run_calibration.py) — calibrate 5 simulator coefficients vs 3 MODIS days, evaluate on 2 held-out days, write calibrated config
- [run_two_stage_calibration.py](scripts/run_two_stage_calibration.py) — literature-anchored diffusion + MODIS-fit absorption/ET (synthetic forcing)
- [run_era5_fetch.py](scripts/run_era5_fetch.py) — fetch ERA5 hourly forcing for Austin into `data/raw/era5/` (requires CDS credentials; see [era5.py](src/austin_twin/era5.py) docstring)
- [run_era5_calibration.py](scripts/run_era5_calibration.py) — **ship-config calibration**: two-stage linear-ET + per-date ERA5 forcing
- [run_era5_downstream.py](scripts/run_era5_downstream.py) — validation + counterfactuals under the ship config
- [run_pm_calibration.py](scripts/run_pm_calibration.py) / [run_pm_downstream.py](scripts/run_pm_downstream.py) — reproducibility for the Penman-Monteith negative-result experiment (NUMBERS.md §13)
- [run_transfer.py](scripts/run_transfer.py) — **multi-city transfer study** (Austin → Phoenix, Denver, Miami): 5-row matrix per target city (zero-shot, Scheme B freeze-abs 1/3 days, Scheme A full-DE 1/3 days). See NUMBERS.md §14

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python 3.11+ recommended. Tested on Apple Silicon (M1) — all dependencies have
arm64 wheels.

## Run

Start with the WorldCover end-to-end pipeline (the "real" baseline):

```bash
python scripts/run_worldcover_mvp.py                    # default 500 m
AUSTIN_GRID_M=100 python scripts/run_worldcover_mvp.py  # 100 m (~30 s, sharper)
```

First run downloads:
- Austin city boundary from OSM (~5 s, cached as GeoJSON)
- ESA WorldCover 10 m tile covering Austin (~30 MB, cached as GeoTIFF per grid resolution)

The default 500 m run writes to unsuffixed paths (`landuse_worldcover.png`, etc.).
Non-default resolutions add a suffix (`landuse_worldcover_100m.png`,
`temperature_worldcover_100m.gif`) so they don't overwrite the canonical
outputs. Subsequent runs hit the cache and finish in seconds.

Then explore the analyses:

```bash
python scripts/run_targeted.py          # zonal sensitivity (best canopy zones)
python scripts/run_validation.py        # validate against MODIS LST
python scripts/run_counterfactuals.py   # canonical intervention scenarios
```

To reproduce the ship-config Phase-2 pipeline (real ERA5 atmospheric
forcing per MODIS date, linear-ET calibrated against it):

```bash
# One-time: register at cds.climate.copernicus.eu, accept ERA5 licence,
# put your API key in ~/.cdsapirc (see src/austin_twin/era5.py docstring).
python scripts/run_era5_fetch.py --start 2024-08-10 --end 2024-08-23
python scripts/run_era5_calibration.py    # writes outputs/calibrated_config_era5.json
python scripts/run_era5_downstream.py     # writes outputs/validation_modis_era5.png etc.
```

## Key outputs (in `outputs/`)

- `landuse_worldcover.png` — real Austin land cover (impervious / vegetation / water)
- `temperature_worldcover.gif` — 48 h surface T animation
- `scenarios_worldcover.png` — five intervention scenarios with ΔT vs baseline
- `zone_sensitivity.png` — choropleth ranking of zones by canopy cooling efficiency
- `validation_modis.png` — simulator vs MODIS LST (Aug 19 2024) anomaly comparison
- `validation_scatter.png` — per-cell scatter, Pearson r and RMSE
- `validation_uncalibrated.png` / `validation_calibrated.png` — direct before/after view of the calibration effect on Aug 19
- `calibrated_config.json` — five tuned coefficients + per-day RMSE/r metrics from the single-stage calibration run
- `calibrated_config_era5.json` — **ship config**: two-stage linear-ET + per-date ERA5 forcing
- `validation_modis_era5.png` / `validation_scatter_era5.png` — ship-config validation vs MODIS on 2024-08-19
- `scenarios_era5.png` / `scenario_summary_era5.txt` — counterfactuals under ship config
- `era5_calibration.png` — 3-panel side-by-side of ship vs legacy: Stage-1 sweep, cooling response, distance decay

## Data sources (all free, no auth)

- City boundary — OpenStreetMap via `osmnx.geocode_to_gdf`
- OSM features — Overpass API via `osmnx.features_from_polygon` (cached as GeoPackage)
- Land cover — ESA WorldCover v200 (2021) public COG on AWS S3
- Surface temperature — MODIS MYD11A1 v6.1 via Microsoft Planetary Computer STAC
- Atmospheric forcing — ERA5 hourly reanalysis (t2m, d2m, u10, v10, ssrd, sp) via the Copernicus Climate Data Store (CDS-Beta API). Requires a free account and one-time licence acceptance; see [era5.py](src/austin_twin/era5.py) docstring.

## Method summary

**Grid.** Configurable cell size in UTM Zone 14N (EPSG:32614), clipped to the Austin city-limits polygon. Default 500 m (94 × 73 array, ~2,896 city cells); 100 m option supported via `AUSTIN_GRID_M=100` (465 × 362 array, ~72,486 city cells). The simulator auto-adjusts its timestep to stay CFL-safe given the chosen resolution and diffusion coefficient; `frame_stride` is auto-chosen so history memory stays bounded regardless of resolution (one stored frame per ~10 simulated minutes).

**Land-cover composition.** ESA WorldCover classes mapped to three channels:
- impervious_frac ← Built-up (class 50)
- vegetation_frac ← Tree / Shrub / Grass / Crop / Wetland (+ half of Bare)
- water_mask ← Permanent water (class 80), kept as continuous fraction

**Simulator.** Per-cell energy balance, 600 s timestep, 48 h horizon:
```
dT/dt = solar_in(t, land) - radiative_out(T) - evapotranspiration(veg, T)
      + diffusion(T)
```
- Solar: half-sine diurnal forcing, absorption weighted by land-cover composition
- Radiative: Newtonian cooling toward ambient
- Evapotranspiration: proportional to vegetation_frac × (T - T_air), one-sided
- Diffusion: 5-point Laplacian, reflective BCs, CFL-checked
- Water cells: relax toward diurnal mean proportional to per-cell water fraction (thermal-mass approximation)

**Counterfactuals.** Each `Scenario` is a `Callable[[Dataset], Dataset]` that perturbs land-use channels. Five canonical scenarios shipped: baseline, +20% canopy, cool roofs downtown (3 km radius, albedo +0.35), 1 km river greenway, suburban densification.

**Zonal sensitivity.** City is partitioned into a 6×5 zone grid (~27 zones with ≥ 6 city cells after edge trimming). For each zone, +20% canopy is added only inside; the simulator runs; we compute local mean ΔT, citywide mean ΔT, total cooling in °C·m², and efficiency = total cooling / area converted. Top zones can absorb the most cooling per dollar of intervention.

**Validation.** MODIS Aqua LST (1:30 pm overpass) on 2024-08-19 for the primary visualization; extended to 18 dates spanning heat-wave and shoulder-season conditions for out-of-distribution testing. Anomaly normalization (subtracting the citywide spatial mean) is still used for cross-config comparability, but under the ship config the absolute-T match is close enough that raw T comparison is also informative.

**Calibration.** Four spatial-variation-driving coefficients (`absorption_impervious`, `absorption_vegetation`, `absorption_water`, `et_coeff`) are tuned by `scipy.optimize.differential_evolution` to minimize mean anomaly RMSE across **three training MODIS days** (Aug 11, 16, 19). Two **held-out days** (Aug 21, 22) are scored only after optimization completes — they never enter the loss. `diffusion_m2_s` is pinned via a Stage-1 sweep to the literature-derived value that gives a 100-300 m half-cooling distance (~1.0 m²/s). Under the ship config each simulator run is driven by the specific ERA5 hourly forcing (t2m, d2m, wind, ssrd, sp) for the day being scored, so the calibrator sees real atmospheric conditions per date rather than one synthetic representative.

**Ship config (linear-ET + ERA5) mean metrics:**

| split | RMSE before | RMSE after | r before | r after |
|---|---|---|---|---|
| train (3 days) | 1.560 °C | **1.547 °C** | +0.714 | **+0.732** |
| test (2 days, held-out) | 1.440 °C | **1.427 °C** | +0.735 | **+0.756** |

Test improves more than train — the calibration generalizes. Test r is
marginally above the synthetic-forcing two-stage baseline (+0.756 vs
+0.747), confirming ERA5 forcing does not degrade spatial pattern
correlation while adding per-date absolute temperatures. See
[NUMBERS.md §13](NUMBERS.md) for per-day metrics and the Penman-Monteith
negative-result ablation.

To use the ship config in any script:

```python
from pathlib import Path
from austin_twin.calibration import load_calibrated_config
cfg = load_calibrated_config(Path("outputs/calibrated_config_era5.json"))
```

## Status

| stage | status |
|---|---|
| Spatial grid + city boundary | ✅ |
| Baseline slab simulator (CFL-stable) | ✅ |
| Counterfactual engine + 5 canonical scenarios | ✅ |
| Real land cover (OSM + ESA WorldCover) | ✅ |
| Zonal sensitivity / targeted intervention analysis | ✅ |
| Validation against satellite LST (MODIS) | ✅ |
| Calibration of simulator coefficients vs MODIS (train/test split) | ✅ |
| Real climate forcing (ERA5, hourly, per-date) | ✅ |
| Penman-Monteith ET experiment (documented negative result) | ✅ (see NUMBERS.md §13) |
| Multi-city transfer (Phoenix, Denver, Miami) | ✅ (see NUMBERS.md §14) |
| ML residual / correction layer | ⬜ |

## Limitations to know about

- **Ship-config sim spread now matches MODIS spread** (~15 °C on 2024-08-19, vs ~15.8 °C observed). The remaining anomaly RMSE (~1.4 °C) is dominated by sub-500 m features (highway hot streaks, fine agricultural patterns) that the grid resolution can't resolve.
- **ET term is empirical, not physics-based.** Canonical FAO-56 Penman-Monteith was implemented and tested against the same MODIS train/test set with ERA5 forcing; it regresses test r from +0.75 to +0.42 because it decouples LE from surface temperature (see [NUMBERS.md §13](NUMBERS.md) for the mechanism-level ablation). A T_surf-driven bulk form with a Priestley-Taylor cap is a plausible physics-based upgrade that preserves the surface-temperature feedback the linear term retains; deferred.
- **500 m grid smooths highway-scale features.** MODIS shows distinct I-35 / MoPac hot streaks that 500 m averaging hides. The 100 m grid (`AUSTIN_GRID_M=100`) resolves these — peak UHI grows from 11.4 °C → 13.7 °C and individual highway corridors become visible — but is ~25× more work per simulation, which makes the full counterfactual + zonal sensitivity pipelines too slow for interactive use at 100 m. Calibration is still done at 500 m; the coefficients transfer reasonably but a re-fit at 100 m would be more rigorous (~30 h of compute).
- **The water buffer is a mixed-layer thermal-mass model** (4× land heat capacity), a step up from the earlier single-parameter relaxation. Adequate at daytime; less accurate at night.

## License

GNU General Public License v3.0 or later (see [LICENSE](LICENSE)).
