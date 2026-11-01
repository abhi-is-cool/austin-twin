"""Fetch ERA5 hourly forcing for Austin into data/raw/era5/.

Phase 2 of the PM-ET work. Requires:
  - CDS account at https://cds.climate.copernicus.eu/
  - ERA5 dataset license accepted on the CDS web UI
  - ~/.cdsapirc populated with `url:` and `key:`
  - `pip install cdsapi xarray netcdf4`

Usage:
  python scripts/run_era5_fetch.py --start 2023-08-15 --end 2023-08-22

The CDS queue typically takes 5-60 min per request. The fetch is idempotent:
re-running with the same output path is a no-op unless --overwrite is set.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from austin_twin.era5 import Era5Request, fetch_era5_hourly


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument(
        "--out-dir",
        default="data/raw/era5",
        help="Cache directory (file is named era5_<start>_<end>.nc)",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_path = Path(args.out_dir) / f"era5_{args.start}_{args.end}.nc"

    print(f"[era5] requesting {args.start} -> {args.end} -> {out_path}")
    print("[era5] CDS queue can take 5-60 min; this script blocks until done.")
    path = fetch_era5_hourly(
        Era5Request(start_date=start, end_date=end, out_path=out_path),
        overwrite=args.overwrite,
    )
    print(f"[era5] wrote {path}")


if __name__ == "__main__":
    main()
