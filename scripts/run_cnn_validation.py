"""Does a real convolutional model behave the way the ceiling argument predicts?

The paper argues that a ceiling bounds a model only when their receptive
fields match, and that convolutional and foundation-model approaches have
receptive fields far wider than one pixel. Both claims are made about CNNs
without ever testing one. This does.

A fully-convolutional network is trained to predict LST from the three
land-cover channels, under the same spatial block folds used everywhere
else, at three receptive fields:

    RF = 1    all 1x1 convolutions -- a per-cell model with a neural net
              attached. This is the control. If the ceiling argument is
              right it should land near the per-cell GBM ceiling, showing
              that nothing is gained by "being a neural network".
    RF = 7    three stacked 3x3 convolutions, ~3.5 km
    RF = 13   six stacked 3x3 convolutions, ~6.5 km

The predictions are:

  * RF=1  ~ per-cell ceiling           (architecture alone buys nothing)
  * RF=7  > per-cell ceiling           (spatial reach is what buys skill)
  * RF=7  ~< context ceiling           (the matched ceiling bounds it)
  * RF=13 may exceed the context ceiling, which would show our particular
    three-Gaussian specification is itself too narrow -- the principle
    holding while our instance of it does not.

Only the LST target is ever masked out of the loss. Land cover is an
input available everywhere in the city, so a convolution reading a test
cell's neighbours is using information a deployed model would also have;
no label information crosses a fold boundary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from austin_twin.ceiling import _multiple_r  # noqa: E402
from austin_twin.ceiling_nonlinear import (  # noqa: E402
    cv_nonlinear_r, spatial_block_folds, spatial_context_channels,
)
from austin_twin.cities import CITIES  # noqa: E402
from austin_twin.grid import build_grid, fetch_boundary  # noqa: E402
from austin_twin.modis import fetch_aqua_lst  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
RESULT = OUT / "cnn_validation.json"

SEED = 20260925
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEPTHS = {1: 0, 7: 3, 13: 6}     # receptive field -> number of 3x3 layers
WIDTH, EPOCHS, LR, WD = 16, 400, 3e-3, 1e-3


class FCN(nn.Module):
    """Fully convolutional. `depth` 3x3 layers give a receptive field of
    1 + 2*depth; depth 0 is a per-cell multilayer perceptron."""

    def __init__(self, in_ch: int, depth: int, width: int = WIDTH):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(depth):
            layers += [nn.Conv2d(c, width, 3, padding=1), nn.ReLU()]
            c = width
        # two 1x1 layers so depth 0 is still a nonlinear per-cell model,
        # which keeps the control a fair comparison rather than a linear one
        layers += [nn.Conv2d(c, width, 1), nn.ReLU(), nn.Conv2d(width, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_predict(X, y, valid, train_mask, depth, seed=SEED):
    """Train on `train_mask` cells, return a prediction for every cell."""
    torch.manual_seed(seed)
    xb = torch.from_numpy(X[None]).float().to(DEV)          # (1, C, H, W)
    yb = torch.from_numpy(y[None, None]).float().to(DEV)    # (1, 1, H, W)
    mb = torch.from_numpy(train_mask[None, None]).float().to(DEV)

    model = FCN(X.shape[0], depth).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    denom = mb.sum().clamp(min=1.0)
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred = model(xb)
        loss = (((pred - yb) ** 2) * mb).sum() / denom
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(xb)[0, 0].cpu().numpy()


def cnn_cv_r(chans, lst, valid, folds, depth):
    """Out-of-fold correlation for a CNN at the given receptive field."""
    H, W = valid.shape
    # standardise inputs; zero outside the city, with a validity channel
    stack = []
    for c in chans:
        v = np.where(valid, np.nan_to_num(c), 0.0).astype(np.float32)
        sd = v[valid].std() or 1.0
        stack.append((v - v[valid].mean()) / sd)
    stack.append(valid.astype(np.float32))
    X = np.stack(stack)

    ys = np.zeros((H, W), np.float32)
    mu, sd = lst[valid].mean(), (lst[valid].std() or 1.0)
    ys[valid] = ((lst[valid] - mu) / sd).astype(np.float32)

    pred = np.full((H, W), np.nan, np.float32)
    fold_ids = np.unique(folds)
    idx = np.argwhere(valid)
    for f in fold_ids:
        te = np.zeros((H, W), bool)
        tr = np.zeros((H, W), bool)
        te[tuple(idx[folds == f].T)] = True
        tr[tuple(idx[folds != f].T)] = True
        if tr.sum() < 100 or te.sum() < 20:
            continue
        out = fit_predict(X, ys, valid, tr, depth)
        pred[te] = out[te]

    ok = np.isfinite(pred) & valid
    if ok.sum() < 50 or np.std(pred[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(pred[ok], ys[ok])[0, 1])


def main() -> None:
    print(f"device: {DEV}\n")
    focus = json.loads((OUT / "focus_city_ceilings.json").read_text())
    out = []

    for rec in focus:
        name = rec["city"]
        city = CITIES[name]
        boundary = fetch_boundary(city, cache_path=RAW / f"{name}_boundary.geojson")
        grid = build_grid(boundary, resolution_m=500.0)
        landuse = build_worldcover_landuse(boundary, grid, RAW,
                                           tile=city.worldcover_tile)
        mask = landuse["city_mask"].values.astype(bool)
        chans = [landuse[v].values.astype(float) for v in
                 ("impervious_frac", "vegetation_frac", "water_mask")]

        print(f"{name}")
        print(f"  {'date':<12}{'linear':>8}{'perCell':>9}{'context':>9}"
              f"{'cnn1':>8}{'cnn7':>8}{'cnn13':>8}")
        per_day = {}
        for d in rec["dates"]:
            try:
                lst = fetch_aqua_lst(d, grid, RAW / "modis")
            except Exception:
                continue
            valid = mask & np.isfinite(lst)
            if int(valid.sum()) < 200:
                continue
            y = lst[valid].astype(float)
            folds = spatial_block_folds(valid)
            X1 = np.column_stack([c[valid] for c in chans])
            Xc = np.column_stack([c[valid] for c in
                                  spatial_context_channels(chans, valid)])
            row = dict(
                linear=_multiple_r(y, X1),
                per_cell=cv_nonlinear_r(y, X1, folds)[0],
                context=cv_nonlinear_r(y, Xc, folds)[0],
            )
            for rf, depth in DEPTHS.items():
                row[f"cnn{rf}"] = cnn_cv_r(chans, lst, valid, folds, depth)
            per_day[d] = row
            print(f"  {d:<12}{row['linear']:>+8.3f}{row['per_cell']:>+9.3f}"
                  f"{row['context']:>+9.3f}{row['cnn1']:>+8.3f}"
                  f"{row['cnn7']:>+8.3f}{row['cnn13']:>+8.3f}")

        if not per_day:
            continue

        def mean(k):
            v = [r[k] for r in per_day.values() if np.isfinite(r[k])]
            return float(np.mean(v)) if v else float("nan")

        agg = {k: mean(k) for k in
               ("linear", "per_cell", "context", "cnn1", "cnn7", "cnn13")}
        agg.update(city=name, n_days=len(per_day), per_day=per_day)
        out.append(agg)
        print(f"  {'MEAN':<12}{agg['linear']:>+8.3f}{agg['per_cell']:>+9.3f}"
              f"{agg['context']:>+9.3f}{agg['cnn1']:>+8.3f}"
              f"{agg['cnn7']:>+8.3f}{agg['cnn13']:>+8.3f}\n")

    RESULT.write_text(json.dumps(out, indent=1))

    print("=" * 74)
    print(f"{'city':<9}{'perCell':>9}{'cnn RF=1':>10}{'context':>9}"
          f"{'cnn RF=7':>10}{'cnn RF=13':>11}")
    print("-" * 74)
    for a in out:
        print(f"{a['city']:<9}{a['per_cell']:>+9.3f}{a['cnn1']:>+10.3f}"
              f"{a['context']:>+9.3f}{a['cnn7']:>+10.3f}{a['cnn13']:>+11.3f}")
    print()
    print("control : RF=1 should sit near the per-cell ceiling")
    print("claim   : RF=7/13 should exceed it, and be bounded by context")
    print(f"\nwrote {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
