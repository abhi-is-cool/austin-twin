"""Check every candidate boundary BEFORE paying for tiles and imagery.

OSM place queries do not reliably return city limits outside the US: asking
for "Riyadh, Saudi Arabia" returns Riyadh Province at 375,478 km2, against
Austin's 725 km2. A province-sized polygon would still produce a
plausible-looking ceiling -- computed over open desert rather than a city --
so this has to be caught by inspection, not by the numbers looking odd.

Costs one geocode per city and no downloads.
"""
from __future__ import annotations
import sys
from pathlib import Path
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_global_extension import CITIES  # noqa: E402
from austin_twin.cities import city_from_point  # noqa: E402
from austin_twin.grid import fetch_boundary  # noqa: E402
from austin_twin.worldcover import tiles_for_boundary  # noqa: E402

RAW = ROOT / "data" / "raw"
# Austin 725, Phoenix 1344, Miami-Dade ~6300 km2 -- the US sample's own range.
PLAUSIBLE_MAX_KM2 = 8000.0

print(f"{'city':<15} {'area km2':>10} {'tiles':>6}  verdict")
print("-" * 62)
bad = []
for name, query, lat, lon, group in CITIES:
    city = city_from_point(name, query, lat, lon)
    try:
        b = fetch_boundary(city, cache_path=RAW / f"{name}_boundary.geojson")
    except Exception as exc:
        print(f"{name:<15} {'--':>10} {'--':>6}  GEOCODE FAILED ({type(exc).__name__})")
        bad.append((name, "geocode failed")); continue
    area = float(b.to_crs(b.estimate_utm_crs()).area.sum()) / 1e6
    tiles = len(tiles_for_boundary(b))
    if area > PLAUSIBLE_MAX_KM2:
        verdict = f"TOO LARGE ({area/725:.0f}x Austin) -- likely a region"
        bad.append((name, f"{area:,.0f} km2"))
    elif tiles > 4:
        verdict = f"{tiles} tiles -- check extent"
        bad.append((name, f"{tiles} tiles"))
    else:
        verdict = "ok"
    print(f"{name:<15} {area:>10,.0f} {tiles:>6}  {verdict}")

print()
if bad:
    print(f"{len(bad)} of {len(CITIES)} need a more specific OSM query:")
    for n, why in bad:
        print(f"  {n}: {why}")
else:
    print("all boundaries plausible")
