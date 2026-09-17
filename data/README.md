# Data

Raw satellite tiles are not stored in this repo (they're large binary `.tif`/`.gpkg`
files) — instead they live in two places:

- **Shared Drive folder:** https://drive.google.com/drive/folders/1NDecVVRKXe07abG5RTNFLwYXUM-ng-Dg?usp=sharing
- **Regenerate locally:** run [`download_tiles.py`](../download_tiles.py) from the repo
  root. It reads tile metadata from the Maxar Open Data STAC catalog and re-downloads
  the same files directly from Maxar's public S3 bucket, so the Drive copy and a
  freshly generated local copy are always identical.

Currently covers: `HurricaneHelene-Oct24` / catalog id `1030010105AA2600` (9 tiles —
`visual`, `ms_analytic`, `pan_analytic`, `data-mask` assets each).
