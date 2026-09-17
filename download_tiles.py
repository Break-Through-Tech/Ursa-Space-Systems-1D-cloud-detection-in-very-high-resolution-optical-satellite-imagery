import json
from pathlib import Path
from urllib.request import urlretrieve

GEOJSON_PATH = Path("maxar-open-data/datasets/HurricaneHelene-Oct24/1030010105AA2600.geojson")
OUTPUT_DIR = Path("data/HurricaneHelene-Oct24/1030010105AA2600")
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
        dest = tile_dir / url.rsplit("/", 1)[-1]
        if dest.exists():
            print(f"skip (exists): {dest}")
            continue
        print(f"downloading: {url} -> {dest}")
        urlretrieve(url, dest)

print("Done.")
