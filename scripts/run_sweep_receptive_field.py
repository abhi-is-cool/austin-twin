"""Receptive-field-matched ceilings for the 49-city US sweep.

The global extension showed that the arid inversion is a pixel-scale
effect and that no city there is data-limited once the ceiling can read
neighbourhoods. The 49-city US sweep, which carries the headline
gap-versus-ceiling association (Spearman rho = +0.69), is still computed
at pixel scale, so that association and everything fitted against it
remain statements about pixel-scale models only.

This recomputes the sweep's ceilings three ways -- linear per-cell,
nonlinear per-cell, and nonlinear with neighbourhood context -- on the
cities' own OSM boundaries and their own accepted cloud-free dates, so
the result is directly comparable to `full_sweep.jsonl` rather than to
the disc-defined global sample.

Resumable: re-running skips cities already written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ceiling_study import CITIES  # noqa: E402

from austin_twin.ceiling import _multiple_r  # noqa: E402
from austin_twin.ceiling_nonlinear import (  # noqa: E402
    cv_nonlinear_r, spatial_block_folds, spatial_context_channels,
)
from austin_twin.cities import city_from_point  # noqa: E402
from austin_twin.grid import build_grid, fetch_boundary  # noqa: E402
from austin_twin.modis import fetch_aqua_lst  # noqa: E402
from austin_twin.worldcover import build_worldcover_landuse  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
SRC = OUT / "full_sweep.jsonl"
RESULT = OUT / "sweep_receptive_field.jsonl"


def done() -> set[str]:
    if not RESULT.exists():
        return set()
    return {json.loads(l)["city"] for l in RESULT.read_text().splitlines() if l.strip()}


def main() -> None:
    sweep = {json.loads(l)["city"]: json.loads(l)
             for l in SRC.read_text().splitlines() if l.strip()}
    queries = {c[0]: c for c in CITIES}
    have = done()
    todo = [n for n in sweep if n not in have]
    print(f"{len(have)} done, {len(todo)} to run\n")
    print(f"{'city':<16}{'linear':>8}{'perCell':>9}{'context':>9}{'gain':>8}{'n':>4}")
    print("-" * 54)

    for name in todo:
        rec = sweep[name]
        try:
            _, query, lat, lon = queries[name]
            city = city_from_point(name, query, lat, lon)
            boundary = fetch_boundary(city,
                                      cache_path=RAW / f"{name}_boundary.geojson")
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
                print(f"{name:<16}  no usable day")
                continue

            def mean(k):
                v = [r[k] for r in per_day.values() if np.isfinite(r[k])]
                return float(np.mean(v)) if v else float("nan")

            out = dict(city=name, lat=rec["lat"], lon=rec["lon"],
                       lst_mean_c=rec["lst_mean_c"],
                       published_uni=rec["ceiling_r"],
                       published_multi=rec["ceiling_multi_r"],
                       veg_imp_lst_gap_c=rec["veg_imp_lst_gap_c"],
                       ndvi_gain=rec.get("ndvi_gain"),
                       linear=mean("linear"), per_cell=mean("per_cell"),
                       context=mean("context"), n_days=len(per_day),
                       per_day=per_day)
            out["gain"] = out["context"] - out["per_cell"]
            with open(RESULT, "a") as fh:
                fh.write(json.dumps(out) + "\n")
            print(f"{name:<16}{out['linear']:>+8.3f}{out['per_cell']:>+9.3f}"
                  f"{out['context']:>+9.3f}{out['gain']:>+8.3f}{out['n_days']:>4d}")
        except Exception as exc:
            print(f"{name:<16}  FAILED {type(exc).__name__}: {str(exc)[:40]}")

    print(f"\nresults in {RESULT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
