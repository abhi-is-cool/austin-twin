"""Does the arid inversion dissolve once the ceiling can see neighbourhoods?

On the four focus cities the receptive-field gain was largest in the arid
ones (Phoenix +0.307, Denver +0.333) and smallest in humid Miami (+0.065).
That ordering suggests the inversion is a pixel-scale effect: the built
core still separates thermally from the desert around it even where
individual built and vegetated cells do not. If so, arid cities are poorly
served by pixel-scale land-cover models specifically, rather than being
intrinsically resistant to land-cover-driven modelling -- a very different
conclusion, and one that changes what a readiness atlas should report.

Four cities cannot settle it. This runs the same comparison across the
28-city global extension, which uses the uniform 25 km disc definition and
spans five continents, so the arid group is not an American artifact.

For each city and each of its cloud-free dates:
    linear      multiple R on three per-cell channels
    per_cell    gradient boosting on the same three, blocked CV
    context     gradient boosting on those plus neighbourhood averages

Then tests whether the gain is larger in arid cities, and whether cities
whose per-cell ceiling is negative recover a positive context ceiling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from disc_boundary import disc_boundary  # noqa: E402

from austin_twin.ceiling import _multiple_r  # noqa: E402
from austin_twin.ceiling_nonlinear import (  # noqa: E402
    cv_nonlinear_r, spatial_block_folds, spatial_context_channels,
)
from austin_twin.cities import city_from_point  # noqa: E402
from austin_twin.grid import build_grid  # noqa: E402
from austin_twin.modis import fetch_aqua_lst  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
SRC = OUT / "global_extension.jsonl"
RESULT = OUT / "arid_receptive_field.jsonl"

ARID_GROUPS = {"arid_nonus", "arid_us"}


def done() -> set[str]:
    if not RESULT.exists():
        return set()
    return {json.loads(l)["city"] for l in RESULT.read_text().splitlines() if l.strip()}


def main() -> None:
    records = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    have = done()
    todo = [r for r in records if r["city"] not in have]
    print(f"{len(have)} done, {len(todo)} to run\n")
    print(f"{'city':<15}{'grp':<16}{'linear':>8}{'perCell':>9}"
          f"{'context':>9}{'gain':>8}{'n':>3}")
    print("-" * 70)

    for rec in todo:
        name = rec["city"]
        try:
            city = city_from_point(name, name, rec["lat"], rec["lon"])
            boundary = disc_boundary(rec["lat"], rec["lon"], rec["utm"],
                                     radius_km=rec["radius_km"])
            grid = build_grid(boundary, resolution_m=500.0)
            landuse = build_worldcover_landuse(boundary, grid, RAW,
                                               tile=city.worldcover_tile)
            mask = landuse["city_mask"].values.astype(bool)
            chans = [landuse[v].values.astype(float) for v in
                     ("impervious_frac", "vegetation_frac", "water_mask")]

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
                per_day[d] = dict(
                    linear=_multiple_r(y, X1),
                    per_cell=cv_nonlinear_r(y, X1, folds)[0],
                    context=cv_nonlinear_r(y, Xc, folds)[0],
                    n=int(y.size),
                )
            if not per_day:
                print(f"{name:<15}{rec['group']:<16}  no usable day")
                continue

            def mean(k):
                v = [r[k] for r in per_day.values() if np.isfinite(r[k])]
                return float(np.mean(v)) if v else float("nan")

            out = dict(city=name, group=rec["group"], lat=rec["lat"],
                       lon=rec["lon"], lst_mean_c=rec["lst_mean_c"],
                       published_uni=rec["ceiling_r"],
                       published_multi=rec["ceiling_multi_r"],
                       veg_imp_lst_gap_c=rec["veg_imp_lst_gap_c"],
                       linear=mean("linear"), per_cell=mean("per_cell"),
                       context=mean("context"), n_days=len(per_day),
                       per_day=per_day)
            out["gain"] = out["context"] - out["per_cell"]
            with open(RESULT, "a") as fh:
                fh.write(json.dumps(out) + "\n")

            print(f"{name:<15}{rec['group']:<16}{out['linear']:>+8.3f}"
                  f"{out['per_cell']:>+9.3f}{out['context']:>+9.3f}"
                  f"{out['gain']:>+8.3f}{out['n_days']:>3d}")
        except Exception as exc:
            print(f"{name:<15}{rec['group']:<16}  FAILED "
                  f"{type(exc).__name__}: {str(exc)[:40]}")

    print(f"\nresults in {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
