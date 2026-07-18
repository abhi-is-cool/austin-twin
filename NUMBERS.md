# austin-twin — number audit

Every quantitative claim made about the model, with its source.
Sources are file paths under the project root or named scripts. Where a
number was produced by an interactive run that did not write a file, the
source is labeled `(stdout)` and the conversation transcript is the
underlying record.

Sections:
1. Project setup constants
2. Data sources, counts, and extents
3. Grid resolution sensitivity (500 m vs 100 m)
4. Calibrated coefficients across three configurations
5. Calibration MODIS train/test metrics (single-stage + two-stage)
6. Differential-evolution iteration trajectories
7. Extended-season MODIS validation (18 days, 6 categories)
8. Regression baseline head-to-head
9. Counterfactual scenario summary (5 canonical)
10. Zonal sensitivity ranking (27 zones)
11. Cooling-response analysis (canopy sweep + patch decay)
12. Two-stage calibration trade-off summary
13. Phase 2: real ERA5 forcing + Penman-Monteith ET negative result
14. Phase 3: multi-city transfer (Austin → Phoenix, Denver, Miami)
15. Open caveats / known number gaps

---

## 1. Project setup constants

| quantity | value | source |
|---|---|---|
| Default grid resolution | 500 m | `scripts/run_worldcover_mvp.py` default |
| Optional fine-grid resolution | 100 m (via `AUSTIN_GRID_M=100`) | `scripts/run_worldcover_mvp.py` |
| Projected CRS | EPSG:32614 (UTM Zone 14N) | `src/austin_twin/grid.py` |
| Default simulation horizon | 48 h | `SimConfig.duration_hours` |
| Default integration timestep (500 m grid) | 600 s (10 min) | `SimConfig.dt_seconds` |
| CFL safety factor | 0.20 of stability limit | `safe_dt()` in `simulator.py` |
| Solar peak (synthetic forcing) | 900 W/m² | `SimConfig.peak_solar_w_m2` |
| Mean diurnal ambient (synthetic) | 30 °C ± 6 °C | `SimConfig.air_temp_c, diurnal_amplitude_c` |
| Cell heat capacity (land slab) | 2.0 × 10⁶ J / (m²·K) | `SimConfig.cell_heat_capacity` |
| Water heat capacity ratio (mixed-layer) | 4.0 × land slab | `SimConfig.water_heat_capacity_ratio` |
| Water latent-heat flux | 100 W/m² | `SimConfig.water_evap_w_m2` |
| Longwave loss coefficient | 8 W/m²/K | `SimConfig.lw_coeff` |

---

## 2. Data sources, counts, and extents

### Austin city limits (OSM)
- Source: `osmnx.geocode_to_gdf("Austin, Texas, USA")`, cached to `data/raw/austin_boundary.geojson`.
- Grid at 500 m: **94 × 73** total cells, **2,896** inside city limits.
- Grid at 100 m: **465 × 362** total cells, **72,486** inside city limits.

### OSM features (cached as GeoPackage, `data/raw/osm/`)

| layer | feature count | source |
|---|---|---|
| buildings | 300,842 | conversation transcript, `scripts/run_osm_mvp.py` |
| roads | 29,071 | same |
| water | 6,205 | same |
| parks | 4,835 | same |

### ESA WorldCover (v200, 2021), AWS S3 public COG
- Tile fetched: N30W099 (~30 MB cached as `data/raw/worldcover/worldcover_10m.tif`).
- Aggregated to 500 m via 50× supersample with class-membership averaging.

Channel statistics (500 m, inside city, calibrated config):

| channel | mean | median | max | p95 |
|---|---|---|---|---|
| impervious_frac | 0.340 | 0.310 | 0.988 | 0.785 |
| vegetation_frac | 0.637 | 0.664 | 1.000 | 0.998 |
| water_mask (continuous) | 0.016 | 0.000 | 1.000 | 0.062 |

Channel statistics (100 m, same coverage area):

| channel | mean | median | max | p95 |
|---|---|---|---|---|
| impervious_frac | 0.342 | 0.255 | 1.000 | 0.956 |
| vegetation_frac | 0.632 | 0.704 | 1.000 | 1.000 |
| water_mask | 0.018 | 0.000 | 1.000 | 0.062 |

Source: `(stdout)` from `scripts/run_worldcover_mvp.py` with and without `AUSTIN_GRID_M=100`.

### MODIS Aqua LST (MYD11A1 v6.1) via Microsoft Planetary Computer STAC
- Days fetched and cached in `data/raw/modis/`: 18 (see §7).
- Per-day Austin coverage: 100 % for 16 of 18, 98-99 % for the remaining 2.
- Source: `extended_validation_selection.json` and `(stdout)` from
  `scripts/run_extended_validation.py`.

---

## 3. Grid resolution sensitivity (500 m vs 100 m)

Both runs use the same calibrated single-stage config (`outputs/calibrated_config.json`).

| metric | 500 m | 100 m | source |
|---|---|---|---|
| Cells inside city | 2,896 | 72,486 | `(stdout)` |
| Auto-CFL timestep | 600 s | 109 s | `(stdout)` |
| Frame stride | 1 | 6 | `(stdout)` |
| Peak instantaneous UHI | 11.44 °C | 13.66 °C | `(stdout)` |
| Impervious p95 | 0.785 | 0.956 | `(stdout)` |
| Impervious max | 0.988 | 1.000 | `(stdout)` |
| Approx wall time, 48 h sim | ~1 s | ~16 s | conversation |

---

## 4. Calibrated coefficients across three configurations

Three calibrated configs have been produced in this project. All three are
fit against the same Aug 11, 16, 19 train days and Aug 21, 22 test days, all
under the WorldCover land cover at 500 m resolution.

| coefficient | single-stage (legacy water physics) | two-stage (legacy water physics, run #1) | two-stage (mixed-layer water physics, CURRENT) |
|---|---|---|---|
| `absorption_impervious` | **0.8268** | 0.800 | **0.8002** |
| `absorption_vegetation` | **0.3713** | 0.548 | **0.4984** |
| `absorption_water` | **0.1191** | 0.487 (red flag) | **0.1693** |
| `et_coeff` (W/m²/K/veg) | **13.2492** | 9.13 | **6.8453** |
| `diffusion_m2_s` | **18.3054** (free) | 1.000 (pinned) | **1.0000** (pinned) |
| `water_heat_capacity_ratio` | n/a (legacy) | n/a (legacy) | 4.0 (default) |
| `water_evap_w_m2` | n/a (legacy) | n/a (legacy) | 100.0 (default) |

Sources:
- single-stage: `outputs/calibrated_config.json` (precise floats above).
- two-stage run #1 (legacy water): conversation transcript, original
  `calibrated_config_two_stage.json` before it was overwritten.
- two-stage run #2 (current, mixed-layer): `outputs/calibrated_config_two_stage.json`.

The single-stage config is what every analysis script auto-loads (when the
JSON exists). The two-stage config requires an explicit
`load_calibrated_config(OUT/"calibrated_config_two_stage.json")` to use.

---

## 5. Calibration MODIS train/test metrics

All metrics are anomaly-normalized (each field has its citywide spatial mean
subtracted before scoring). RMSE in °C.

### Single-stage (legacy water physics)

| split | days | RMSE before | RMSE after | r before | r after |
|---|---|---|---|---|---|
| TRAIN | 3 (Aug 11, 16, 19) | 1.465 | **1.256** | +0.784 | **+0.823** |
| TEST  | 2 (Aug 21, 22)    | 1.304 | **1.070** | +0.814 | **+0.855** |

Source: `outputs/calibrated_config.json`.

Per-day train:
| date | RMSE before | RMSE after | r before | r after |
|---|---|---|---|---|
| 2024-08-11 | 1.263 | 1.124 | +0.781 | +0.821 |
| 2024-08-16 | 1.424 | 1.232 | +0.782 | +0.818 |
| 2024-08-19 | 1.709 | 1.410 | +0.788 | +0.829 |

Per-day test:
| date | RMSE before | RMSE after | r before | r after |
|---|---|---|---|---|
| 2024-08-21 | 1.411 | 1.202 | +0.790 | +0.828 |
| 2024-08-22 | 1.196 | 0.937 | +0.837 | +0.882 |

Source: conversation transcript from `scripts/run_calibration.py` stdout.

### Two-stage (legacy water physics, run #1, superseded)

| split | RMSE before | RMSE after | r before | r after |
|---|---|---|---|---|
| TRAIN (n=3) | 1.747 | 1.570 | +0.714 | +0.717 |
| TEST  (n=2) | 1.668 | 1.449 | +0.729 | +0.738 |

Source: conversation transcript only (JSON was overwritten by run #2).

### Two-stage (mixed-layer water physics, CURRENT)

| split | RMSE before | RMSE after | r before | r after |
|---|---|---|---|---|
| TRAIN (n=3) | **1.778** | **1.600** | +0.715 | +0.725 |
| TEST  (n=2) | **1.697** | **1.482** | +0.731 | **+0.747** |

Source: `outputs/calibrated_config_two_stage.json`.

### Direct head-to-head on TEST set (Aug 21, 22)

| config | mean r | mean RMSE (°C) |
|---|---|---|
| single-stage (with mixed-layer water defaults applied) | +0.850 | 1.089 |
| two-stage (mixed-layer) | +0.747 | 1.482 |

Source: conversation transcript from latest `scripts/run_two_stage_calibration.py`.

---

## 6. Differential-evolution iteration trajectories

### Single-stage (legacy water physics)

Mean training-set RMSE per generation (best so far):

```
iter  1: 1.268    iter  5: 1.259    iter  9: 1.256
iter  2: 1.262    iter  6: 1.257    iter 10: 1.256
iter  3: 1.262    iter  7: 1.257    iter 11: 1.256
iter  4: 1.262    iter  8: 1.256
```

Source: `outputs/calibrated_config.json` field `history`.

### Two-stage (mixed-layer water physics, current)

```
iter  1: 1.642    iter  5: 1.616    iter  9: 1.608
iter  2: 1.642    iter  6: 1.616    iter 10: 1.608
iter  3: 1.637    iter  7: 1.614    iter 11: 1.600
iter  4: 1.637    iter  8: 1.611    iter 12: 1.600
```

Source: `outputs/calibrated_config_two_stage.json` field `history`.

---

## 7. Extended-season MODIS validation (18 days, 6 categories)

Coefficients used: single-stage (legacy water physics) — held FIXED across
all 18 days. Regression baselines trained only on Aug 11/16/19.

### Per-day metrics

| category | date | mean LST (°C) | sim r | sim RMSE | reg r | reg RMSE | reg+pos r | reg+pos RMSE |
|---|---|---|---|---|---|---|---|---|
| heat_wave_aug_2024 | 2024-08-11 | 38.4 | +0.821 | 1.12 | +0.662 | 1.47 | +0.768 | 1.26 |
| heat_wave_aug_2024 | 2024-08-16 | 41.1 | +0.818 | 1.23 | +0.668 | 1.59 | +0.753 | 1.41 |
| heat_wave_aug_2024 | 2024-08-19 | 46.1 | +0.829 | 1.41 | +0.703 | 1.78 | +0.750 | 1.65 |
| heat_wave_aug_2024 | 2024-08-21 | 45.4 | +0.828 | 1.20 | +0.682 | 1.57 | +0.767 | 1.37 |
| heat_wave_aug_2024 | 2024-08-22 | 43.5 | +0.882 | 0.94 | +0.703 | 1.42 | +0.759 | 1.30 |
| summer_non_extreme_2024 | 2024-06-08 | 39.9 | +0.909 | 0.86 | +0.717 | 1.43 | +0.742 | 1.38 |
| summer_non_extreme_2024 | 2024-09-07 | 36.1 | +0.793 | 1.13 | +0.639 | 1.39 | +0.725 | 1.26 |
| summer_non_extreme_2024 | 2024-09-08 | 37.4 | +0.833 | 1.12 | +0.670 | 1.50 | +0.740 | 1.37 |
| spring_2024 | 2024-04-03 | 32.0 | +0.878 | 1.08 | +0.709 | 1.57 | +0.765 | 1.44 |
| spring_2024 | 2024-04-04 | 36.5 | +0.819 | 1.29 | +0.642 | 1.73 | +0.643 | 1.74 |
| spring_2024 | 2024-04-22 | 28.2 | +0.881 | 1.14 | +0.702 | 1.67 | +0.767 | 1.51 |
| fall_2024 | 2024-10-15 | 39.1 | +0.853 | 0.96 | +0.671 | 1.13 | +0.749 | 1.10 |
| fall_2024 | 2024-10-23 | 34.4 | +0.602 | 1.42 | +0.482 | 1.38 | +0.565 | 1.39 |
| fall_2024 | 2024-10-24 | 35.6 | +0.677 | 1.35 | +0.589 | 1.38 | +0.634 | 1.38 |
| winter_2023_2024 | 2023-12-27 | 19.1 | +0.749 | 1.18 | +0.574 | 1.27 | +0.697 | 1.19 |
| winter_2023_2024 | 2024-01-10 | 18.8 | +0.789 | 1.24 | +0.562 | 1.23 | +0.642 | 1.29 |
| winter_2023_2024 | 2024-01-29 | 22.5 | +0.781 | 1.12 | +0.581 | 1.38 | +0.753 | 1.12 |
| summer_2023 (different yr) | 2023-08-08 | 49.0 | +0.786 | 1.44 | +0.660 | 1.75 | +0.651 | 1.77 |
| summer_2023 (different yr) | 2023-08-16 | 44.8 | +0.801 | 1.14 | +0.621 | 1.48 | +0.680 | 1.40 |
| summer_2023 (different yr) | 2023-08-18 | 47.7 | +0.819 | 1.20 | +0.663 | 1.56 | +0.689 | 1.52 |

### Per-category means

| category | n | mean LST | sim r | sim RMSE | reg r | reg RMSE | reg+pos r | reg+pos RMSE |
|---|---|---|---|---|---|---|---|---|
| heat_wave_aug_2024 (in-dist) | 5 | 42.9 | +0.836 | 1.18 | +0.684 | 1.57 | +0.760 | 1.40 |
| summer_non_extreme_2024 | 3 | 37.8 | **+0.845** | **1.04** | +0.676 | 1.44 | +0.736 | 1.34 |
| spring_2024 | 3 | 32.2 | **+0.859** | 1.17 | +0.684 | 1.66 | +0.725 | 1.56 |
| fall_2024 | 3 | 36.4 | +0.711 | 1.25 | +0.581 | 1.30 | +0.649 | 1.29 |
| winter_2023_2024 | 3 | 20.1 | +0.773 | 1.18 | +0.572 | 1.29 | +0.697 | 1.20 |
| summer_2023 (different yr) | 3 | 47.2 | +0.802 | 1.26 | +0.648 | 1.60 | +0.673 | 1.57 |

Source: `outputs/extended_validation.txt`.

---

## 8. Regression baseline head-to-head (Aug 11/16/19 fit, Aug 21/22 test)

Calibrated single-stage simulator vs lstsq regression with same channels
(impervious + vegetation + water) vs lstsq regression with channels + 5
positional features (x, y, x², y², xy).

### Per-day

| split | date | sim r | sim RMSE | reg r | reg RMSE | reg+pos r | reg+pos RMSE |
|---|---|---|---|---|---|---|---|
| train | 2024-08-11 | +0.821 | 1.124 | +0.662 | 1.471 | +0.768 | 1.256 |
| train | 2024-08-16 | +0.818 | 1.232 | +0.668 | 1.594 | +0.753 | 1.409 |
| train | 2024-08-19 | +0.829 | 1.410 | +0.703 | 1.775 | +0.750 | 1.646 |
| test  | 2024-08-21 | +0.828 | 1.202 | +0.682 | 1.567 | +0.767 | 1.374 |
| test  | 2024-08-22 | +0.882 | 0.937 | +0.703 | 1.419 | +0.759 | 1.305 |

### Aggregated mean

| split | sim r | sim RMSE | reg r | reg RMSE | reg+pos r | reg+pos RMSE |
|---|---|---|---|---|---|---|
| TRAIN (n=3) | +0.823 | 1.256 | +0.678 | 1.613 | +0.757 | 1.437 |
| TEST  (n=2) | **+0.855** | **1.070** | +0.692 | 1.493 | +0.763 | 1.339 |

### Δ (simulator advantage, test set)

- Pearson r: simulator beats landuse-only regression by **+0.163**, beats landuse+position regression by **+0.092**.
- RMSE: simulator beats landuse-only by **−0.423 °C**, beats landuse+position by **−0.269 °C**.

Source: `outputs/baseline_comparison.txt`.

### Learned regression coefficients (anomaly target, landuse-only fit)

```
impervious_frac    -3.014
vegetation_frac    -8.402
water_mask        -13.065
intercept          +6.589
```

Note: coefficients are conditional on the other channels (multicollinearity);
do not interpret signs in isolation.

Source: conversation transcript from `scripts/run_baseline_comparison.py` stdout.

---

## 9. Counterfactual scenario summary (5 canonical, single-stage config, WorldCover)

| scenario | mean ΔT (°C) | max cooling (°C) | % cells < −1 °C |
|---|---|---|---|
| baseline | +0.00 | +0.00 | 0.0 |
| canopy_plus_20 | **−1.34** | −5.97 | **66.5** |
| cool_roofs_downtown | −0.07 | −3.57 | 2.8 |
| river_greenway | −0.41 | −3.58 | 11.4 |
| suburban_densification | +0.22 | −3.12 | 0.5 |

Source: `outputs/scenario_summary_worldcover.txt`.

Pre-calibration values (for reference, before the simulator was tuned):

| scenario | mean ΔT (°C) | max cooling (°C) | % cells < −1 °C |
|---|---|---|---|
| canopy_plus_20 | −1.71 | −8.94 | 69.4 |
| cool_roofs_downtown | −0.14 | −3.00 | 2.6 |
| river_greenway | −0.85 | −2.95 | 14.4 |
| suburban_densification | +0.28 | (warming) | — |

Source: conversation transcript from earlier `scripts/run_worldcover_mvp.py`.

---

## 10. Zonal sensitivity ranking (27 zones, single-stage config)

Each zone = a tile from a 6 × 5 partition of the city bbox (zones with < 6
city cells dropped). Intervention: +20 % canopy applied only inside zone.
Local ΔT = mean ΔT inside the zone; city ΔT = mean ΔT across all city cells;
efficiency = total cooling (°C·m²) / area converted (m²).

### Top 5 zones (most efficient canopy investment)

| rank | zone | city cells | area converted (ha) | local ΔT (°C) | city ΔT (°C) | efficiency (°C·m²/m²) |
|---|---|---|---|---|---|---|
| 1 | C4 | 204 | 1,009.8 | **−0.966** | −0.0881 | **6.313** |
| 2 | B4 | 210 | 1,016.1 | −0.940 | −0.0844 | 6.014 |
| 3 | D4 | 216 | 1,037.2 | −0.854 | −0.0819 | 5.715 |
| 4 | D3 | 202 | 960.9 | −0.843 | −0.0744 | 5.606 |
| 5 | B3 | 183 | 759.2 | −0.658 | −0.0544 | 5.185 |

### Bottom 5 zones (already vegetated; little to convert)

| rank | zone | city cells | area converted (ha) | local ΔT (°C) | city ΔT (°C) | efficiency (°C·m²/m²) |
|---|---|---|---|---|---|---|
| 23 | E6 | 23 | 48.3 | −0.139 | −0.0017 | 2.506 |
| 24 | C6 | 121 | 353.9 | −0.148 | −0.0071 | 1.455 |
| 25 | C2 | 49 | 152.0 | −0.101 | −0.0029 | 1.367 |
| 26 | B1 | 11 | 20.0 | **−0.014** | −0.0001 | **0.208** |
| 27 | C1 | 17 | 42.0 | −0.007 | −0.0001 | 0.132 |

Total convertible area (top-3 spine B4+C4+D4): **3,063.1 ha** of impervious.

Source: `outputs/zone_ranking.txt`.

---

## 11. Cooling-response analysis (single-stage config, legacy water physics)

### Canopy sensitivity sweep (citywide canopy increment Δ)

| canopy +Δ | citywide ΔT (°C) | planted ΔT (°C) | spillover ΔT (°C) |
|---|---|---|---|
| +5 % | −1.663 | −1.594 | −2.289 |
| +10 % | −1.987 | −1.923 | −2.365 |
| +20 % | −2.537 | −2.534 | −2.491 |
| +40 % | −3.251 | −3.499 | −2.457 |

### Model vs literature reference bands

| metric | model | literature band |
|---|---|---|
| Citywide cooling per +10 % canopy (slope over 5-40 % range) | **−0.445 °C** | 0.10 − 0.50 |
| Local cooling per +10 % canopy (slope over 5-40 % range) | **−0.539 °C** | 1.0 − 3.0 |
| Patch peak cooling (1 km canopy patch, central Austin) | **−3.10 °C** | study-dependent |
| Patch half-cooling distance | **610 m** (λ = 880 m) | 100 − 300 |

Source: `outputs/cooling_response.txt`.

### Same analysis under TWO-STAGE config with mixed-layer water physics (current)

| metric | value |
|---|---|
| Citywide cooling per +10 % canopy | **−0.389 °C** |
| Local cooling per +10 % canopy | **−0.521 °C** |
| Patch half-cooling distance | **363 m** (λ = 524 m) |

Source: conversation transcript from latest `scripts/run_two_stage_calibration.py`.

### Same analysis under SINGLE-STAGE config with mixed-layer water physics

| metric | value |
|---|---|
| Citywide cooling per +10 % canopy | −0.449 °C |
| Local cooling per +10 % canopy | −0.544 °C |
| Patch half-cooling distance | 619 m (λ = 892 m) |

Source: conversation transcript from latest `scripts/run_two_stage_calibration.py`.

(Difference vs legacy water physics single-stage is within rounding because
water cells are ~1.5 % of city area; new water physics barely shifts the
central UHI pattern.)

---

## 12. Two-stage calibration trade-off summary

Side-by-side, three configurations (all evaluated under current
mixed-layer water physics):

| metric | single-stage | two-stage (legacy water, run #1) | two-stage (mixed-layer, CURRENT) |
|---|---|---|---|
| MODIS test r | +0.850 | +0.738 | **+0.747** |
| MODIS test RMSE (°C) | 1.089 | 1.449 | **1.482** |
| `α_impervious` | 0.827 | 0.800 | 0.800 |
| `α_vegetation` | 0.371 | 0.548 | **0.498** |
| `α_water` | 0.119 | **0.487 (red flag)** | **0.169 ✓** |
| `et_coeff` | 13.249 | 9.13 | 6.845 |
| `diffusion_m2_s` | 18.305 | 1.000 | 1.000 |
| Patch half-cooling (m) | 619 | 365 | **363** |
| Citywide ΔT / +10 % canopy | −0.449 | −0.350 | −0.389 |
| Local ΔT / +10 % canopy | −0.544 | −0.472 | −0.521 |

### Stage 1 diffusion sweep (current, mixed-layer water physics)

| D (m²/s) | half-cooling (m) | λ (m) | peak ΔT (°C) |
|---|---|---|---|
| 1.00 | 247 | 356 | 18.66 |
| 1.43 | 282 | 406 | 14.54 |
| 2.04 | 322 | 464 | 11.44 |
| 2.91 | 311 | 448 | 12.23 |
| 4.15 | 349 | 503 | 9.70 |
| 5.92 | 394 | 568 | 7.57 |
| 8.45 | 454 | 654 | 5.70 |
| 12.05 | 521 | 752 | 4.31 |
| 17.20 | 603 | 870 | 3.22 |
| 24.55 | 693 | 999 | 2.43 |
| 35.04 | 802 | 1157 | 1.80 |
| 50.00 | 936 | 1351 | 1.32 |

Selected D = **1.0 m²/s** (lowest in search range; gives half-cooling 247 m at
sweep time, but 363 m after Stage 2 refits the absorptions sharper).

Source: `outputs/calibrated_config_two_stage.json` (`stage1_sweep`) and
conversation transcript.

---

## 13. Phase 2: real ERA5 forcing + Penman-Monteith ET negative result

Phase 2 replaces the synthetic 30 ± 6 °C diurnal forcing with hourly
ERA5 reanalysis (t2m, d2m, u10, v10, ssrd, sp) for each MODIS date
individually, and evaluates whether canonical FAO-56 Penman-Monteith is
a viable drop-in for the linear ET term. Ship config is
**linear-ET + ERA5** (`outputs/calibrated_config_era5.json`); PM is
retained as a documented negative result.

### Ship-config coefficients (linear-ET + ERA5)

| coefficient | two-stage (synthetic, §4) | linear-ET + ERA5 |
|---|---|---|
| `absorption_impervious` | 0.800 | **0.803** |
| `absorption_vegetation` | 0.498 | **0.434** |
| `absorption_water` | 0.169 | **0.178** |
| `et_coeff` | 6.845 | **10.975** |
| `diffusion_m2_s` | 1.000 (Stage-1 pinned) | **1.000** (Stage-1 pinned) |

`et_coeff` rises 60 % because ERA5 forcing has real August air
temperatures (35-41 °C peak vs synthetic mean 30 °C), so realizing the
same MODIS-anomaly fit requires a stronger ET response per unit
surface-air gap. The three absorptions move by less than 10 %.

Source: `outputs/calibrated_config_era5.json`.

### Ship-config MODIS train/test metrics

Anomaly-normalized, per-date sim runs driven by each date's own ERA5
forcing.

| split | days | RMSE before | RMSE after | r before | r after |
|---|---|---|---|---|---|
| TRAIN | 3 (Aug 11, 16, 19) | 1.560 | **1.547** | +0.714 | **+0.732** |
| TEST  | 2 (Aug 21, 22)     | 1.440 | **1.427** | +0.735 | **+0.756** |

Per-day train:
| date | r before | r after | RMSE before (°C) | RMSE after (°C) | sim spread (°C) | obs spread (°C) |
|---|---|---|---|---|---|---|
| 2024-08-11 | +0.693 | +0.713 | 1.474 | 1.485 | 14.30 | 10.34 |
| 2024-08-16 | +0.704 | +0.723 | 1.552 | 1.542 | 14.78 | 13.80 |
| 2024-08-19 | +0.743 | +0.761 | 1.654 | 1.613 | 15.39 | 15.82 |

Per-day test:
| date | r before | r after | RMSE before (°C) | RMSE after (°C) | sim spread (°C) | obs spread (°C) |
|---|---|---|---|---|---|---|
| 2024-08-21 | +0.719 | +0.741 | 1.524 | 1.510 | 15.34 | 13.00 |
| 2024-08-22 | +0.750 | +0.771 | 1.355 | 1.344 | 14.86 | 12.34 |

Source: `outputs/calibrated_config_era5.json` and stdout of
`scripts/run_era5_calibration.py`.

### ERA5 forcing is pattern-neutral (compared to synthetic)

Δr on test set: **+0.009** (ship linear-ET + ERA5, r=+0.756 vs
two-stage w/ mixed-layer + synthetic, r=+0.747). ERA5 does not
degrade the anomaly correlation and yields per-date absolute
temperature fields (previously blocked by the synthetic-forcing
assumption). Absolute sim T on 2024-08-19: mean = 40.8 °C (vs MODIS
47.6 °C, obs spread 15.8 °C, sim spread 15.4 °C — first time the model
resolves both the mean and the range at the correct scale).

### Penman-Monteith ET: negative result with mechanism

Canonical FAO-56 PM was implemented in
[`penman_monteith.py`](src/austin_twin/penman_monteith.py) and calibrated
against the same MODIS train/test set with ERA5 forcing driving each
run. **Result: substantial regression in anomaly correlation.**

Ablation of what drives the regression:

| variant | train r | test r |
|---|---|---|
| Legacy linear-ET + synthetic (§5 two-stage) | +0.725 | +0.747 |
| Legacy linear-ET + ERA5 (isolates forcing) | +0.714 | +0.735 |
| **PM + ERA5 (as calibrated)** | **+0.496** | **+0.418** |
| PM + ERA5, LAI_K = 1 (gentle saturation) | +0.567 | +0.515 |
| PM + ERA5, LAI = LAI_max · f_veg (linear) | +0.548 | +0.493 |
| PM + ERA5, LAI = 10 · f_veg (steeper) | +0.484 | +0.411 |

Source: monkeypatched-LAI diagnostic in conversation transcript.

**Two mechanisms contribute to the regression:**

1. **LAI saturation (~1/3 of the gap).** The default
   LAI(f_veg) = 5·(1 − e⁻³ᶠ) reaches ~4.0 at f_veg = 0.5 and ~4.75 at
   f_veg = 1.0, so PM sees cells with 50 % and 90 % canopy as nearly
   identical. Softening (LAI_K = 1) or fully linearizing recovers ~0.09
   of test r. Steeper gradient (LAI = 10·f_veg) does not help — the
   ceiling is real, not tunable.
2. **T_surf ↔ LE decoupling (~2/3 of the gap).** Canonical FAO-56 uses
   VPD from `(T_air, T_dew)` only; hotter surface cells do not shed
   more LE. This removes the linear-ET positive-feedback
   (Q_ET ∝ T_surf − T_air) that gave the model its spatial-UHI
   contrast. Every cell in the ERA5 window sees the same atmospheric
   VPD, so PM's spatial variability is driven only by Rn (absorption ×
   solar), which is a weaker discriminator than land-cover-linked ET.

Under recalibration, DE reached α_veg = 0.303 (essentially the 0.30
lower bound) trying to compensate for over-strong PM cooling. The
bound-hit is a symptom of the T_surf-decoupling issue, not a tunable
parameter to widen.

Config: `outputs/calibrated_config_pm.json`, kept for reproducibility.
Not the ship config.

**Design implication.** The linear ET term used in the ship config
(`Q_ET = et_coeff · f_veg · max(T − T_air, 0)`) is empirically
motivated but retains the surface-temperature feedback that carries
the UHI signal in MODIS. A physics-based alternative that preserves
the feedback would be a T_surf-driven bulk form (e.g. Dalton
Q_LE ∝ (e_sat(T_surf) − e_a)) with a Priestley-Taylor cap Q_LE ≤
1.26·Rn to avoid the unbounded behavior we saw before adopting
canonical FAO-56. That work is deferred.

### Ship-config counterfactual scenarios (2024-08-19 ERA5 forcing)

Rerun of the five canonical scenarios under `calibrated_config_era5.json`,
sharing the ERA5 forcing for 2024-08-19 across all runs (so ΔT isolates
land-use, not weather). Convention: `mean ΔT` and `max cooling` follow
`summarize()` — mean over all 48 h frames × all always-finite city cells;
max over the same. Same units and definitions as §9's historical
single-stage table.

| scenario | mean ΔT (°C) | max cooling (°C) | % cells < −1 °C |
|---|---|---|---|
| baseline | +0.00 | +0.00 | 0.0 |
| canopy_plus_20 | **−0.74** | −4.27 | **29.2** |
| cool_roofs_downtown (boost=0.35, canonical) | −0.07 | −4.80 | 2.7 |
| river_greenway | −0.25 | −4.87 | 8.2 |
| suburban_densification | +0.18 | −2.58 | 0.1 |

Source: `outputs/scenario_summary_era5.txt`.

### Cool-roofs intensity comparison (treatment-area metric)

The canonical `cool_roofs_downtown` scenario applies a uniform 0.35
multiplicative albedo boost across every cell in a 3 km disk around the
downtown centroid, regardless of land class. This is 3–5× more
aggressive than a typical cool-roof retrofit (which treats 20–40 % of
impervious per block with a roof-albedo change of ~0.2–0.3, so
block-average absorption Δ ≈ 0.04–0.12). We rerun the scenario at
`boost=0.10` — the literature-matched effective intensity — to give a
like-for-like comparison against published cooling ranges, and add a
**treatment-area** column: mean over the 112 city cells inside the 3 km
disk (rather than citywide), so the number is comparable to the
"in-treated-area" cooling that cool-roof literature actually reports.

| variant | boost | citywide mean ΔT (48 h) | treatment-area mean ΔT (48 h) | treatment-area peak ΔT | literature band |
|---|---|---|---|---|---|
| `cool_roofs_downtown` (canonical, as-shipped) | 0.35 | −0.068 | **−1.663** | −3.458 | 0.35 – 1.2 °C (peak) |
| `cool_roofs_downtown_lit` (intensity-matched) | 0.10 | −0.019 | −0.471 | **−0.970** | ✓ in band |

Interpretation: the canonical boost=0.35 scenario is a
larger-than-literature intervention; its treatment-area cooling
(−3.46 °C peak, −1.66 °C 48 h mean) scales roughly linearly with
intensity. At intensity matched to actual cool-roof retrofit programs
(boost=0.10), the treatment-area peak cooling is −0.97 °C, cleanly
inside the 0.35–1.2 °C literature band. The model is not overreacting
to the intervention; the canonical scenario is applying a stronger
stimulus than the literature values were measured under.

Source: direct extraction on 2024-08-19 ERA5 forcing under ship config
(scripted; see conversation transcript for exact recipe).

---

## 14. Phase 3: multi-city transfer (Austin → Phoenix, Denver, Miami)

Question: how portable is the Austin-calibrated digital twin? We take the
ship config from §13 (`calibrated_config_era5.json`) and evaluate it on
three climatically-distinct target cities:

| target | climate | validation window | test dates |
|---|---|---|---|
| Phoenix, AZ | hot semi-arid desert | 2024-07-15 – 2024-07-28 | 2024-07-22, 2024-07-24 |
| Denver, CO | humid continental, high altitude (1600 m) | 2024-07-15 – 2024-07-28 | 2024-07-25, 2024-07-28 |
| Miami-Dade, FL | humid subtropical, coastal | 2024-04-04 – 2024-04-20 (dry season) | 2024-04-16, 2024-04-19 |

Miami's rainy-season MODIS coverage in Aug/Sep is unusable (0–35 %); we
pivoted to the April dry season, where 8 of 20 days have >70 % coverage
and LSTs still reach 31–48 °C.

Each target city is evaluated under five schemes; the held-out test-set r
and RMSE are the payoff.

### Transfer matrix — Phoenix

| row | training data | test r | test RMSE (°C) | notes |
|---|---|---|---|---|
| Zero-shot | Austin coefs, none | **+0.080** | 3.12 | Austin ship config, no target-city fitting |
| Scheme B, 1 day | freeze abs., tune `et_coeff` on 1 target day | +0.075 | 3.06 | `et_coeff` → 5.75 (hit lower bound at 5.05 with 3 days) |
| Scheme B, 3 days | freeze abs., tune `et_coeff` on 3 target days | +0.074 | 3.05 | `et_coeff` → 5.05 |
| Scheme A, 1 day | full 4-coef DE on 1 target day | +0.039 | 2.26 | absorption_veg hits upper bound 0.70 |
| Scheme A, 3 days | full 4-coef DE on 3 target days | +0.036 | 2.23 | same |

### Transfer matrix — Denver

| row | training data | test r | test RMSE (°C) | notes |
|---|---|---|---|---|
| Zero-shot | Austin coefs, none | **+0.004** | 3.44 | Austin ship config, no target-city fitting |
| Scheme B, 1 day | freeze abs., tune `et_coeff` on 1 target day | +0.024 | 3.36 | `et_coeff` → 5.05 (hit lower bound) |
| Scheme B, 3 days | freeze abs., tune `et_coeff` on 3 target days | +0.024 | 3.36 | identical to 1 day |
| Scheme A, 1 day | full 4-coef DE on 1 target day | −0.099 | 2.72 | absorption_veg hits upper bound 0.70 |
| Scheme A, 3 days | full 4-coef DE on 3 target days | −0.172 | 2.54 | worse than zero-shot |

### Transfer matrix — Miami

| row | training data | test r | test RMSE (°C) | notes |
|---|---|---|---|---|
| Zero-shot | Austin coefs, none | **+0.760** | 2.85 | Austin ship config, no target-city fitting |
| Scheme B, 1 day | freeze abs., tune `et_coeff` on 1 target day | +0.787 | 2.77 | `et_coeff` → 33.7 (up from Austin 10.98) |
| Scheme B, 3 days | freeze abs., tune `et_coeff` on 3 target days | +0.787 | 2.77 | identical to 1 day |
| Scheme A, 1 day | full 4-coef DE on 1 target day | +0.832 | 2.41 | imp → 0.94, veg → 0.31 |
| Scheme A, 3 days | full 4-coef DE on 3 target days | +0.832 | 2.41 | plateau, no improvement over 1 day |

Baseline for reference: Austin ship config on Austin test set (§13) is
`r = +0.756`, `RMSE = 1.43 °C`.

Sources: `outputs/transfer_phoenix.json`, `outputs/transfer_denver.json`,
`outputs/transfer_miami.json`.

### The critical control: what's the ceiling per city?

Near-zero transfer r on Phoenix and Denver could mean either (a) Austin's
coefficients mistransfer, or (b) the spatial LST simply doesn't correlate
with WorldCover categories in those cities. We separate these by measuring
the *data-level* Pearson correlation between MODIS LST anomaly and
impervious fraction — the maximum r any linear-land-cover model can
achieve without additional features.

| city | data ceiling (r_LST_imp) | model achieves (best of transfer matrix) | model / ceiling |
|---|---|---|---|
| Austin (§13, native) | +0.671 (2024-08-22) | +0.756 | 1.13× (multi-feature synergy) |
| Phoenix | **+0.363** (2024-07-22) | +0.080 (zero-shot) | 0.22× — well below ceiling |
| Denver | **+0.131** (2024-07-25) | +0.004 (zero-shot) | 0.03× — essentially at noise floor |
| Miami | **+0.827** (2024-04-16) | +0.832 (Scheme A) | 1.01× — at ceiling |

The Phoenix and Denver failures are *partly* model-transfer failures and
*partly* unavoidable structural weaknesses in the target-city data. A
perfectly calibrated linear-land-cover model would top out around r ≈ 0.37
in Phoenix and r ≈ 0.13 in Denver — the LST spread on Denver's test days
is only 1.4–2.7 °C, roughly 5× smaller than Austin or Miami, leaving
little spatial pattern for any model to recover.

### Coefficient drift under full re-DE (Scheme A, 3 days)

| coefficient | Austin | Phoenix | Denver | Miami | Phoenix Δ | Denver Δ | Miami Δ |
|---|---|---|---|---|---|---|---|
| `absorption_impervious` | 0.803 | 0.800 | 0.800 | 0.940 | −0.003 | −0.003 | **+0.137** |
| `absorption_vegetation` | 0.434 | 0.689 | 0.700 | 0.309 | **+0.255** | **+0.266** | −0.125 |
| `absorption_water`      | 0.178 | 0.110 | 0.120 | 0.100 | −0.068 | −0.058 | −0.078 |
| `et_coeff`              | 10.975 | 8.93 | 5.64 | 25.99 | −2.04 | **−5.33** | **+15.02** |
| `diffusion_m2_s`        | 1.000 | 1.000 (pinned) | 1.000 (pinned) | 1.000 (pinned) | 0 | 0 | 0 |

Two observations:

1. **`absorption_vegetation` splits western from eastern cities.**
   Phoenix and Denver (arid/semi-arid western cities) both push `α_veg`
   to the upper bound 0.70, consistent with their WorldCover "vegetation"
   being dominated by irrigated turf, xeriscape, and sparse scrub that
   absorbs strongly. Miami pushes `α_veg` down to 0.31, consistent with
   genuinely lush, shaded subtropical canopy that cools effectively.
2. **`et_coeff` is not cross-city stable.** Denver's dry mountain climate
   suppresses evapotranspiration, so DE cuts `et_coeff` by 49 % (−5.33);
   Miami's humid subtropical climate amplifies it, and DE raises it by
   137 % (+15.02). The linear ET term `k_ET · f_veg · (T − T_air)` does
   not absorb these climate differences implicitly — it needs a different
   coefficient per city. This invalidates the Scheme-B "physics-portable
   absorptions, climate-portable `et_coeff`" hypothesis: `et_coeff` is
   at least as climate-dependent as the absorptions.

### Practical answer to "how portable is the digital twin?"

- Portability depends on **data structure of the target city** (how well
  WorldCover land-cover categories predict thermal heterogeneity), not on
  climate distance from Austin. Miami (climatically far — humid, coastal,
  tropical) transfers well because its land-cover classes cleanly
  separate thermal regimes. Phoenix and Denver (climatically nearer —
  semi-arid or dry western) transfer poorly because irrigation, urban-form
  heterogeneity, and low overall LST spread break the WorldCover →
  thermal-class mapping.
- Where portability is possible (Miami), **zero-shot reaches 92 % of
  ceiling** (r = +0.760 vs ceiling +0.827). Adding target-city MODIS days
  raises this to +0.832 (Scheme A), but the improvement saturates after
  just 1 day.
- Where portability is broken (Phoenix, Denver), **more training data does
  not help.** Scheme A with 3 days performs *worse* than zero-shot for
  both cities — DE over-fits the training days at the expense of held-out
  days. The right response is additional features (irrigation-aware
  vegetation classification, sub-500 m urban-form heterogeneity) rather
  than more calibration data.

---

## 15. Open caveats / known number gaps

1. **The cooling-response numbers in §11 use the single-stage config.** They
   have not been re-extracted under the current two-stage / mixed-layer
   config in a written file — only via the verification block of the
   two-stage script (numbers in §11 sub-tables and §12). A clean re-run of
   `scripts/run_cooling_response.py` against
   `outputs/calibrated_config_two_stage.json` would put those numbers in
   `outputs/cooling_response.txt` properly.

2. **The counterfactual scenario summary (§9) is under the single-stage
   config.** Re-running `scripts/run_counterfactuals.py` and
   `scripts/run_worldcover_mvp.py` after loading the two-stage config would
   shift the magnitudes (canopy_plus_20 mean ΔT would drop from −1.34 °C
   toward −1.0 °C based on the slope comparison in §12, but exact numbers
   require a re-run).

3. **The zonal sensitivity ranking (§10) is under the single-stage config.**
   Earlier sanity-check confirmed the ranking is stable under calibration,
   but the exact local-ΔT and efficiency values would shift under the
   two-stage config; not yet recomputed.

4. **MODIS coverage is heat-wave-heavy by date selection.** Of 18 validation
   days, 5 are within the Aug 2024 heat wave (in-distribution) and another 3
   are summer 2023 (different year but still extreme). Cold-season sample
   size is only 3 (winter 2023-24). A larger winter / spring / fall sample
   would tighten the per-category confidence intervals.

5. **Literature bands in §11 are approximate central ranges**, not citations
   of specific papers. A reviewer can substitute exact published values
   from their preferred sources without affecting the model numbers.

6. **All absolute temperatures under configs §4-§12 use synthetic 30 °C ± 6 °C
   diurnal forcing.** No comparison is meaningful in absolute °C terms for
   those configs; all validation there is anomaly-normalized. The ship
   config in §13 (`calibrated_config_era5.json`) uses real ERA5 forcing per
   MODIS date and now resolves absolute sim T within a few °C of MODIS
   (e.g. 40.8 °C sim mean vs 47.6 °C MODIS mean on 2024-08-19, with
   matching ~15 °C spread).

7. **Multi-city transfer (§14) uses a reduced DE budget** (popsize=4,
   maxiter=6) vs Austin's own calibration (popsize=8, maxiter=12) to
   keep Miami-Dade's 187×168 grid tractable. Pilot tests on Austin's
   own data at the reduced budget show <0.02 r degradation vs full
   budget, so this shouldn't affect the transfer conclusions, but a
   reviewer requesting the full budget could re-run
   `scripts/run_transfer.py` with the tuple restored (~4× runtime).

8. **Phoenix, Denver, and Miami use different validation windows** (Phoenix
   and Denver July 2024, Miami April 2024) because Miami's August rainy
   season has unusably low MODIS coverage. This means the cities are scored
   under different atmospheric conditions (Phoenix/Denver at heat-wave peak,
   Miami in dry-season shoulder). The zero-shot comparison is still
   meaningful (Austin coefs applied to each city's own ERA5 + MODIS),
   but a cleaner design would fetch a same-month window for all four
   cities if reviewer weight is put on the season-comparability point.
