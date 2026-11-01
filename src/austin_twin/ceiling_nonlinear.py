"""A model-class-free data ceiling.

`ceiling.py` defines the ceiling as a multiple correlation, which bounds a
*linear* land-cover model and nothing more. That is a real limitation, not
a technicality: the shipped Austin configuration scores r = +0.754 on
held-out days against an Austin multivariate ceiling of +0.693 on those
same days. A clamped ET term and diffusion coupling express structure no
linear combination of three channels can, so the model clears its own
"ceiling" by roughly 9%. Reporting a percent-of-ceiling against a quantity
models routinely exceed is not meaningful, and mapping it across hundreds
of cities would only scale the error.

This module estimates what land cover can support without assuming a
functional form, by two routes that fail in opposite directions:

  * `cv_nonlinear_r` -- gradient boosting under spatial block
    cross-validation. An *achievable* figure: some model reached it, so the
    true ceiling is at least this high. It can underestimate if the
    estimator is a poor fit.
  * `mi_bound_r` -- a Kraskov k-NN mutual-information estimate converted to
    a correlation equivalent. An *upper* bound in principle, since it is
    distribution-free, but MI estimation in three dimensions on a few
    thousand autocorrelated cells is noisy and biases upward.

Neither alone is trustworthy. Read together they bracket the ceiling, and
the width of the bracket is itself the honest statement of how well the
ceiling is known for that city.

SPATIAL BLOCK CROSS-VALIDATION IS NOT OPTIONAL HERE. Neighbouring 500 m
cells are strongly autocorrelated, so a random train/test split puts a
cell's own neighbours in the training set and the fitted model effectively
memorises the field. That inflates the nonlinear estimate toward 1.0 and
would manufacture the conclusion that land cover explains everything.
Folds are therefore contiguous tiles, matching the block bootstrap used
for the confidence intervals elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

BLOCK_SIDE = 6      # cells; matches bootstrap_uncertainty.BLOCK_SIDE
N_FOLDS = 5
SEED = 20260921


@dataclass
class NonlinearCeiling:
    """Both estimates plus what is needed to judge them."""

    linear_r: float        # multiple R, the existing definition
    cv_nonlinear_r: float  # spatially blocked CV, gradient boosting
    mi_bound_r: float      # mutual-information upper bound, r-equivalent
    n_valid: int
    n_folds_used: int
    block_side: int

    @property
    def nonlinear_gain(self) -> float:
        """How much allowing nonlinearity buys. Large means the linear
        ceiling was understating what land cover supports."""
        return self.cv_nonlinear_r - self.linear_r

    def to_dict(self) -> dict:
        d = asdict(self)
        d["nonlinear_gain"] = self.nonlinear_gain
        return d


def spatial_block_folds(
    valid: np.ndarray,
    block_side: int = BLOCK_SIDE,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> np.ndarray:
    """Assign each valid cell to a fold by contiguous tile.

    `valid` is a 2D boolean mask. Returns a 1D array of fold indices, one
    per valid cell in row-major order, so it lines up with `field[valid]`.

    Whole tiles go to one fold, so a test cell's immediate neighbours are
    almost always held out with it rather than trained on.
    """
    ny, nx = valid.shape
    tile_row = np.arange(ny)[:, None] // block_side
    tile_col = np.arange(nx)[None, :] // block_side
    tile_id = tile_row * (nx // block_side + 1) + tile_col

    ids = np.unique(tile_id[valid])
    rng = np.random.default_rng(seed)
    assignment = rng.integers(0, n_folds, size=ids.size)
    lookup = dict(zip(ids.tolist(), assignment.tolist()))
    return np.array([lookup[t] for t in tile_id[valid]], dtype=int)


def cv_nonlinear_r(
    y: np.ndarray,
    X: np.ndarray,
    folds: np.ndarray,
    seed: int = SEED,
) -> tuple[float, int]:
    """Out-of-fold correlation from gradient boosting, blocked spatially.

    Returns (r, n_folds_actually_used). Predictions are pooled across folds
    before correlating, rather than averaging per-fold correlations, so a
    fold that happens to span little thermal range cannot dominate.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    if y.size < 200:
        return float("nan"), 0

    pred = np.full(y.shape, np.nan)
    used = 0
    for f in np.unique(folds):
        train, test = folds != f, folds == f
        # A fold with too little training or test data cannot be scored.
        if train.sum() < 100 or test.sum() < 20:
            continue
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_depth=4,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
        used += 1

    ok = np.isfinite(pred)
    if used < 2 or ok.sum() < 50:
        return float("nan"), used
    if np.std(pred[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan"), used
    return float(np.corrcoef(pred[ok], y[ok])[0, 1]), used


def _knn_mutual_information(y: np.ndarray, X: np.ndarray, k: int = 5) -> float:
    """Kraskov--Stoegbauer--Grassberger estimate of I(y; X), in nats.

    Estimator 1 of Kraskov et al. (2004): for each point, find the distance
    to its k-th neighbour in the joint space under the Chebyshev metric,
    then count how many neighbours fall strictly inside that distance in
    each marginal space.
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    n = y.size
    if n <= k + 1:
        return float("nan")

    # Standardise so no channel dominates the joint metric by its units.
    def z(a):
        s = np.std(a, axis=0)
        s = np.where(s == 0, 1.0, s)
        return (a - np.mean(a, axis=0)) / s

    Xs, ys = z(X), z(y.reshape(-1, 1))
    joint = np.hstack([Xs, ys])

    eps = cKDTree(joint).query(joint, k=k + 1, p=np.inf)[0][:, k]
    # Strictly-inside counts; the tiny shrink avoids counting the k-th point.
    nx = cKDTree(Xs).query_ball_point(Xs, r=eps * (1 - 1e-9), p=np.inf,
                                      return_length=True) - 1
    ny_ = cKDTree(ys).query_ball_point(ys, r=eps * (1 - 1e-9), p=np.inf,
                                       return_length=True) - 1
    mi = (digamma(k) + digamma(n)
          - np.mean(digamma(np.maximum(nx, 0) + 1)
                    + digamma(np.maximum(ny_, 0) + 1)))
    return float(max(mi, 0.0))


def mi_bound_r(y: np.ndarray, X: np.ndarray, k: int = 5,
               max_n: int = 4000, seed: int = SEED) -> float:
    """Mutual information converted to an equivalent correlation.

    Under joint normality I = -0.5 log(1 - rho^2), so rho = sqrt(1 - e^{-2I}).
    Real fields are not jointly normal, but the inversion is monotone and
    keeps the number on the same scale as every other ceiling reported here,
    which is what makes it comparable at a glance.

    Subsamples above `max_n` points: the KSG estimator is O(n log n) per
    tree query and the estimate is stable well below the full city.
    """
    if y.size < 200:
        return float("nan")
    if y.size > max_n:
        idx = np.random.default_rng(seed).choice(y.size, max_n, replace=False)
        y, X = y[idx], X[idx]
    mi = _knn_mutual_information(y, X, k=k)
    if not np.isfinite(mi):
        return float("nan")
    return float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * mi))))


def nonlinear_ceiling(
    lst_c: np.ndarray,
    channels: list[np.ndarray],
    valid: np.ndarray,
    block_side: int = BLOCK_SIDE,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> NonlinearCeiling:
    """Linear, cross-validated nonlinear and MI-bound ceilings for one city-day.

    `valid` must already encode the city mask and finite-LST condition;
    `channels` are the 2D land-cover fields the model would consume.
    """
    from .ceiling import _multiple_r

    y = lst_c[valid].astype(float)
    X = np.column_stack([c[valid].astype(float) for c in channels])

    lin = _multiple_r(y, X)
    folds = spatial_block_folds(valid, block_side, n_folds, seed)
    nl, used = cv_nonlinear_r(y, X, folds, seed)
    mi = mi_bound_r(y, X, seed=seed)

    return NonlinearCeiling(
        linear_r=lin,
        cv_nonlinear_r=nl,
        mi_bound_r=mi,
        n_valid=int(y.size),
        n_folds_used=used,
        block_side=block_side,
    )


def spatial_context_channels(
    channels: list[np.ndarray],
    valid: np.ndarray,
    sigmas: tuple[float, ...] = (1.0, 2.0, 4.0),
) -> list[np.ndarray]:
    """Add neighbourhood-averaged copies of each land-cover channel.

    The ceiling as originally defined regresses LST on each cell's *own*
    land cover. A simulator with a diffusion term does not work that way:
    heat spreads, so a cell's modelled temperature depends on its
    neighbours' land cover too. Any model with a receptive field wider than
    one cell -- a diffusion scheme, a convolutional network, a model given
    neighbourhood features -- therefore has access to information the
    per-cell ceiling never measured, and can exceed it without anything
    being wrong.

    That makes the per-cell ceiling a bound on per-cell models specifically,
    not on land-cover models in general. Matching the ceiling's information
    to the model's receptive field is what makes the comparison fair.

    Sigmas are in cells, so on the 500 m grid (1, 2, 4) spans roughly
    0.5-2 km. The shipped diffusivity D = 1.0 m^2 s^-1 over 48 h gives a
    diffusion length of about 590 m, a little over one cell, so that range
    brackets the simulator's own reach.

    Smoothing is mask-aware: cells outside the city are excluded from both
    numerator and denominator, so the city edge is not contaminated by
    whatever lies beyond it.
    """
    from scipy.ndimage import gaussian_filter

    weight = valid.astype(float)
    denom = {s: gaussian_filter(weight, s, mode="constant") for s in sigmas}

    out = list(channels)
    for c in channels:
        masked = np.where(valid, np.nan_to_num(c), 0.0)
        for s in sigmas:
            num = gaussian_filter(masked, s, mode="constant")
            d = denom[s]
            out.append(np.divide(num, d, out=np.zeros_like(num), where=d > 1e-6))
    return out
