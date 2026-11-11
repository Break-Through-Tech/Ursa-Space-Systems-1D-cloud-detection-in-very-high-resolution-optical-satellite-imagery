# Data

Raw satellite tiles are not stored in this repo (they're large binary `.tif`/`.gpkg`
files) — instead they live in two places:

- **Shared Drive folder:** https://drive.google.com/drive/folders/1NDecVVRKXe07abG5RTNFLwYXUM-ng-Dg?usp=sharing
- **Regenerate locally:** run [`download_tiles.py`](../download_tiles.py) from the repo
  root. It reads tile metadata from the Maxar Open Data STAC catalog and re-downloads
  the same files directly from Maxar's public S3 bucket, so the Drive copy and a
  freshly generated local copy are always identical.

Currently covers:

- `HurricaneHelene-Oct24` / catalog id `1030010105AA2600` (9 tiles)
- `Brazil-Flooding-May24` / catalog id `10300100887C0F00` (6 tiles, reported by
  the team to have no cloud cover)
- `Marshall-Fire-21-Update` / catalog id `10200100BCB1A500` (15 tiles; the
  catalog does not provide `ms_analytic` assets for this event)

Available assets vary by event. The downloader skips assets that are not
provided by the catalog; Marshall Fire tiles have `visual`, `pan_analytic`, and
`data-mask` assets.

## Human-verified cloud reference labels

The GeoPackage `data-mask` assets describe image footprints; they are not cloud
labels. To create a small reference set, use the RGB `visual.tif` tiles with the
VS Code Jupyter reviewer in
[`notebooks/review_reference_labels.py`](../notebooks/review_reference_labels.py).
The script samples up to three 1024 x 1024 patches from each tile, stratified by
the fraction of pixels marked cloud-like by a brightness/saturation heuristic.
It prefers patches with at least 80% valid imagery, avoiding empty borders; if a
tile has no such patch, it uses its non-empty patches and records that fallback.
These masks are candidate labels only and must be checked and corrected by a
human before they are used as reference data.

Use [`notebooks/Review_Cloud_Labels.ipynb`](../notebooks/Review_Cloud_Labels.ipynb)
in VS Code with the Jupyter extension. The coordinator runs its setup and
generation cells once using the local `data/` folder. The generated review
bundle is stored at `data/reference-review-v2/` and contains small RGB patches,
candidate masks, and a manifest; the original TIFFs remain ignored by Git.
To view every candidate mask over its RGB patch without opening the interactive
reviewer one patch at a time, run the batch-preview cell in the notebook. It
saves blue-overlay JPEGs under `data/reference-review-v2/candidate-overlays/`;
the original candidate PNG masks are unchanged.

In VS Code's Source Control view, the coordinator commits and pushes
`data/reference-review-v2/manifest.json`, `patches/`, and
`candidate-overlays/`.
Other reviewers pull the repository, open the same notebook, skip patch
generation, and run the review cell with their own stable reviewer ID. Each
person's masks, tags, and status are saved separately in
`data/reference-review-v2/annotations/<reviewer_id>/`; commit and push only that
person's annotation folder to share progress. Pull the latest branch before
starting a review session and push the annotation changes afterward. This keeps
review work in the repository, avoids Drive access and re-uploading large TIFFs,
and lets each person resume their annotations after reopening VS Code.

The coordinator needs local copies of the raw TIFFs to generate patches.
Reviewers only need the committed patch bundle and code; they do not need the
source TIFFs. Ensure all reviewers use the same generated patch bundle and
reviewer IDs remain unique.

The reviewer starts with HSV threshold pseudo-labels (`0` clear/non-cloud,
`1` candidate cloud, `255` ignore) for Hurricane Helene. For an event explicitly
listed in `cloud_free_events`, valid pixels start as `0` (clear) while the
separate candidate mask still shows where the threshold predicted cloud. This
lets reviewers inspect and correct potential false positives in the reported
cloud-free event; it does not automatically certify those pixels as reference
labels. Draw polygons on the RGB view, choose a label/tag, and apply the
correction. Ambiguous pixels should be marked ignore; clearly identifiable thin
cloud can be labeled cloud and tagged. Smoke and bright ground are non-cloud and
can be tagged for targeted analysis. A patch remains `pending` until a reviewer
enters their name and selects **Mark reviewed**. Only patches with a matching
`annotations/<reviewer_id>/*_review.json` status of `reviewed` should be
included in an evaluation set.
The candidate mask is retained as `*_candidate.png`; reviewed labels, tags, and
reviewer metadata are saved separately.
Use `load_reviewed_labels(review_dir, patch_id, reviewer_id)` from the same
module to load a final mask and tags; it raises an error for patches that are
still pending.

To prepare reviewed patches for the DCP benchmark, run
`python notebooks/prepare_reviewed_eval.py` from the repository root. The
script writes a reusable `outputs/reviewed_eval_set/` with the expected
`dev/{images,masks}` and `test/{images,masks}` layout, plus
`split_manifest.csv` and `dataset_info.json`. It includes only patches marked
`reviewed`, checks their RGB/mask pairs, and assigns whole `(event, tile)` groups
to splits with seed 42 (25% of each event's reviewed tiles held out for test).
Generated copies are ignored by Git and can be recreated from the committed
review bundle.

This tile-grouped split prevents patches from a source tile crossing splits,
but both sets still share the same two events. Since the current Brazil labels
are cloud-free and Hurricane Helene supplies the cloud-positive examples, this
is a preliminary within-event benchmark, not an independent event-level test.
The current annotations are also single-reviewer labels; scores should be
treated accordingly.

The sampler tries to include low-candidate and high-candidate patches from each
tile, avoids empty image borders, and records event/tile IDs in the manifest.
Treat the Brazil tiles as a valuable negative-scene reference after review. For
threshold selection and final scoring, keep the event/scene split independent:
with only these two events, holding Brazil out can test false positives on a
cloud-free scene, but it cannot also provide an independent cloudy-scene test.
