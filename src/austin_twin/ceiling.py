"""The land-cover data-ceiling diagnostic, and the covariates that explain it.

The ceiling is the Pearson correlation between observed MODIS LST and
impervious fraction across a city's valid cells: the best any model driven
by land cover alone could score there. Comparing a model's transfer score
against it separates "the model failed" from "the input data cannot express
this city's thermal structure".

Two things make this cheap to run at scale, and both matter for turning the
published two-city result into a real study:

  * It needs no simulator, no calibration and no ERA5 — only MODIS LST and a
    land-cover raster. There is nothing to fit, so there is no train/test
    split to get wrong.
  * Pearson r is invariant to mean subtraction, so scoring against the LST
    anomaly and against raw LST give identical values.

`CeilingResult` also carries the covariates a hundred-city study would
regress the ceiling against — the point being to predict *which* cities have
low ceilings, rather than observing one case (Phoenix) and generalising from
it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class CeilingResult:
    """Ceiling plus the city descriptors that might explain it."""

    ceiling_r: float          # r(LST, impervious_frac) — univariate
    ceiling_multi_r: float    # multiple R on (impervious, vegetation, water)
    n_valid: int              # cells contributing
    coverage: float           # fraction of city cells with valid LST

    lst_mean_c: float
    lst_spread_c: float       # p98 - p2, robust stand-in for dynamic range

    impervious_mean: float
    impervious_std: float     # no spatial contrast -> nothing to correlate with
    vegetation_mean: float
    water_mean: float

    class_entropy: float      # Shannon entropy over the 3 channels, in bits
    veg_imp_lst_gap_c: float  # mean LST of built cells minus vegetated cells

    def to_dict(self) -> dict:
        return asdict(self)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _multiple_r(y: np.ndarray, X: np.ndarray) -> float:
    """Multiple correlation: r between y and its least-squares projection on X.

    This is the quantity that actually bounds a *linear* land-cover model,
    because the simulator consumes three channels rather than one. The
    univariate r(LST, impervious) is only a bound on a model using impervious
    fraction alone, and a three-channel model can exceed it by construction —
    which is what makes the univariate version unsuitable as a stated ceiling.

    Note it still does not bound a *nonlinear* model: the simulator's clamped
    ET term and diffusion coupling capture structure no linear combination of
    the channels can, so Austin exceeds even this.
    """
    if y.size < 5:
        return float("nan")
    A = np.column_stack([X, np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    if np.std(pred) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(pred, y)[0, 1])


def _entropy_bits(fracs: list[np.ndarray]) -> float:
    """Shannon entropy of the mean class mixture, in bits.

    A city whose cells are nearly all one class has little for any
    land-cover model to separate, independent of how the classes behave
    thermally. Low entropy is therefore a candidate ceiling predictor
    distinct from the thermal-behaviour failure Phoenix exhibits.
    """
    p = np.array([float(np.mean(f)) for f in fracs], dtype=float)
    total = p.sum()
    if total <= 0:
        return float("nan")
    p = p / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def data_ceiling(
    lst_c: np.ndarray,
    impervious: np.ndarray,
    vegetation: np.ndarray,
    water: np.ndarray,
    city_mask: np.ndarray,
    built_threshold: float = 0.5,
    veg_threshold: float = 0.5,
) -> CeilingResult:
    """Compute the data ceiling and its covariates for one city-day.

    All arrays share the model grid's shape. `lst_c` is Celsius with NaN for
    cloud and out-of-city cells.
    """
    valid = np.isfinite(lst_c) & city_mask
    n_city = int(city_mask.sum())
    n_valid = int(valid.sum())
    coverage = (n_valid / n_city) if n_city else 0.0

    if n_valid < 3:
        nan = float("nan")
        return CeilingResult(
            ceiling_r=nan, ceiling_multi_r=nan,
            n_valid=n_valid, coverage=coverage,
            lst_mean_c=nan, lst_spread_c=nan,
            impervious_mean=nan, impervious_std=nan,
            vegetation_mean=nan, water_mean=nan,
            class_entropy=nan, veg_imp_lst_gap_c=nan,
        )

    lst_v = lst_c[valid].astype(float)
    imp_v = impervious[valid].astype(float)
    veg_v = vegetation[valid].astype(float)
    wat_v = water[valid].astype(float)

    built = imp_v >= built_threshold
    vegd = veg_v >= veg_threshold
    gap = (float(lst_v[built].mean() - lst_v[vegd].mean())
           if built.any() and vegd.any() else float("nan"))

    p2, p98 = np.percentile(lst_v, [2, 98])

    return CeilingResult(
        ceiling_r=_pearson(lst_v, imp_v),
        ceiling_multi_r=_multiple_r(lst_v, np.column_stack([imp_v, veg_v, wat_v])),
        n_valid=n_valid,
        coverage=coverage,
        lst_mean_c=float(lst_v.mean()),
        lst_spread_c=float(p98 - p2),
        impervious_mean=float(imp_v.mean()),
        impervious_std=float(imp_v.std()),
        vegetation_mean=float(veg_v.mean()),
        water_mean=float(wat_v.mean()),
        class_entropy=_entropy_bits([imp_v, veg_v, wat_v]),
        veg_imp_lst_gap_c=gap,
    )


def ceiling_for_landuse(
    lst_c: np.ndarray, landuse, **kwargs
) -> CeilingResult:
    """Convenience wrapper taking the landuse Dataset the pipeline builds."""
    return data_ceiling(
        lst_c=lst_c,
        impervious=landuse["impervious_frac"].values,
        vegetation=landuse["vegetation_frac"].values,
        water=landuse["water_mask"].values,
        city_mask=landuse["city_mask"].values.astype(bool),
        **kwargs,
    )
