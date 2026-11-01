"""Linear-regression baseline for the simulator validation comparison.

Predicts per-cell LST anomaly from WorldCover land-cover channels (and,
optionally, smooth positional features). Trained and evaluated with the same
splits used for simulator calibration so the head-to-head is apples-to-apples:

  - simple variant : impervious_frac, vegetation_frac, water_mask
    (the same three channels the simulator integrates over time)
  - position variant: above + (x_norm, y_norm, x_norm^2, y_norm^2, x_norm*y_norm)
    (strictly more information than the simulator gets — a stress test)

No regularization (problem is over-determined: thousands of rows, <= 9 columns).
Anomaly normalization: each day's MODIS LST has its citywide spatial mean
subtracted before being stacked into the training set. Predictions are also
mean-subtracted at evaluation time, identical to how the simulator output is
treated in scripts/run_validation.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

FEATURE_NAMES_SIMPLE = ("impervious_frac", "vegetation_frac", "water_mask")
FEATURE_NAMES_POSITION = FEATURE_NAMES_SIMPLE + (
    "x_norm", "y_norm", "x_norm*y_norm", "x_norm^2", "y_norm^2",
)


def _build_features(landuse: xr.Dataset, use_position: bool) -> np.ndarray:
    """Per-cell feature matrix of shape (ny*nx, n_features). Row order is C."""
    imp = landuse["impervious_frac"].values.ravel()
    veg = landuse["vegetation_frac"].values.ravel()
    wat = landuse["water_mask"].values.ravel()
    feats = [imp, veg, wat]
    if use_position:
        xs = landuse["x"].values
        ys = landuse["y"].values
        xx, yy = np.meshgrid(xs, ys)  # both shape (ny, nx)
        # Normalize position by spread of the bounding box so coefficients are
        # comparable in magnitude to the land-cover features (which are in [0,1]).
        x_norm = ((xx - xx.mean()) / xx.std()).ravel()
        y_norm = ((yy - yy.mean()) / yy.std()).ravel()
        feats += [x_norm, y_norm, x_norm * y_norm, x_norm**2, y_norm**2]
    return np.stack(feats, axis=1).astype(np.float64)


@dataclass
class RegressionModel:
    coefs: np.ndarray  # shape (n_features + 1,), last entry is intercept
    use_position: bool
    feature_names: tuple[str, ...]

    def predict_map(self, landuse: xr.Dataset) -> np.ndarray:
        """Per-cell predicted LST anomaly. NaN outside city mask."""
        feats = _build_features(landuse, self.use_position)
        pred = feats @ self.coefs[:-1] + self.coefs[-1]
        pred = pred.reshape(landuse["city_mask"].values.shape)
        return np.where(landuse["city_mask"].values, pred, np.nan).astype(np.float32)


def fit(
    landuse: xr.Dataset,
    modis_train: dict[str, np.ndarray],
    use_position: bool = False,
) -> RegressionModel:
    """Fit linear regression of LST anomaly on landuse features."""
    feats_all = _build_features(landuse, use_position)  # (n_cells_flat, n_feat)
    city = landuse["city_mask"].values.ravel()

    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    for _date, lst in modis_train.items():
        lst_flat = lst.ravel()
        valid = city & np.isfinite(lst_flat)
        if not valid.any():
            continue
        anom = lst_flat[valid] - np.mean(lst_flat[valid])
        X_blocks.append(feats_all[valid])
        y_blocks.append(anom)

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    X_aug = np.column_stack([X, np.ones(X.shape[0])])
    coefs, _resid, _rank, _sv = np.linalg.lstsq(X_aug, y, rcond=None)

    names = FEATURE_NAMES_POSITION if use_position else FEATURE_NAMES_SIMPLE
    return RegressionModel(coefs=coefs, use_position=use_position, feature_names=names)


def evaluate_day(model: RegressionModel, landuse: xr.Dataset, lst: np.ndarray) -> tuple[float, float, int]:
    """Return (anomaly RMSE °C, Pearson r, n_valid_cells) for one MODIS day.

    Predictions are spatially mean-subtracted before comparison to match the
    treatment of the simulator in scripts/run_validation.py.
    """
    pred = model.predict_map(landuse)
    valid = np.isfinite(pred) & np.isfinite(lst)
    if not valid.any():
        return float("nan"), float("nan"), 0
    p = pred[valid] - np.mean(pred[valid])
    o = lst[valid] - np.mean(lst[valid])
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    r = float(np.corrcoef(p, o)[0, 1]) if np.std(p) > 0 and np.std(o) > 0 else 0.0
    return rmse, r, int(valid.sum())


def evaluate(model: RegressionModel, landuse: xr.Dataset, modis_days: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """Return {date: (rmse, r)} for each day."""
    out: dict[str, tuple[float, float]] = {}
    for date, lst in modis_days.items():
        rmse, r, _n = evaluate_day(model, landuse, lst)
        out[date] = (rmse, r)
    return out
