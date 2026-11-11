import json
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

GEOJSON_PATH = Path("maxar-open-data/datasets/Marshall-Fire-21-Update/10200100BCB1A500.geojson")
OUTPUT_DIR = Path("data/Marshall-Fire-21-Update/10200100BCB1A500")
ASSET_KEYS = ["visual", "ms_analytic", "pan_analytic", "data-mask"]

with GEOJSON_PATH.open() as f:
    features = json.load(f)["features"]

print(f"Found {len(features)} tiles")

for feature in features:
    props = feature["properties"]
    quadkey = props["quadkey"]
    tile_dir = OUTPUT_DIR / quadkey
    tile_dir.mkdir(parents=True, exist_ok=True)

    for key in ASSET_KEYS:
        url = props[key]
        if not url:
            print(f"skip (not available): {key} for {quadkey}")
            continue
        dest = tile_dir / url.rsplit("/", 1)[-1]

        with urlopen(Request(url, method="HEAD"), timeout=15) as resp:
            remote_size = int(resp.headers["Content-Length"])

        if dest.exists() and dest.stat().st_size == remote_size:
            print(f"skip (complete): {dest}")
            continue
        print(f"downloading: {url} -> {dest}")
        urlretrieve(url, dest)

print("Done.")
