"""Build a tile-grouped dev/test set from reviewed reference-label patches."""

import argparse
import csv
import json
import math
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


LABEL_VALUES = {0, 1, 255}


def prepare_reviewed_eval(
    review_root: str | Path,
    output_root: str | Path,
    seed: int = 42,
    test_fraction: float = 0.25,
) -> Path:
    """Stage reviewed RGB/mask pairs with parent tiles kept in one split."""
    review_root = Path(review_root).resolve()
    output_root = Path(output_root).resolve()
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    manifest_path = review_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing patch manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patches = manifest.get("patches")
    if not isinstance(patches, list):
        raise ValueError("Patch manifest must contain a 'patches' list")
    patches_by_id = {patch.get("id"): patch for patch in patches}
    if None in patches_by_id or len(patches_by_id) != len(patches):
        raise ValueError("Patch manifest contains missing or duplicate IDs")

    reviews_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    review_files = sorted((review_root / "annotations").glob("*/**/*_review.json"))
    if not review_files:
        raise FileNotFoundError(
            f"No review records found under {review_root / 'annotations'}"
        )
    for review_path in review_files:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        patch_id = review.get("patch_id")
        if review.get("status") != "reviewed":
            continue
        if patch_id not in patches_by_id:
            raise ValueError(
                f"Reviewed patch {patch_id!r} is not in the patch manifest"
            )
        if patch_id in reviews_by_id:
            raise ValueError(
                f"Multiple reviewed annotations found for {patch_id!r}; "
                "select a single reference annotation before preparing evaluation"
            )
        patch = patches_by_id[patch_id]
        if review.get("event_id") != patch.get("event_id"):
            raise ValueError(f"Event metadata differs for patch {patch_id}")
        reviews_by_id[patch_id] = (review_path, review)

    if not reviews_by_id:
        raise ValueError("No patches have a review status of 'reviewed'")

    grouped: dict[str, set[str]] = defaultdict(set)
    records: list[dict[str, Any]] = []
    for patch_id, (review_path, review) in sorted(reviews_by_id.items()):
        patch = patches_by_id[patch_id]
        event_id = patch.get("event_id")
        tile_id = patch.get("tile_id")
        reviewer_id = review.get("reviewer_id")
        if not all(isinstance(value, str) and value for value in (event_id, tile_id, reviewer_id)):
            raise ValueError(f"Missing event, tile, or reviewer metadata for {patch_id}")
        if review_path.parent.name != reviewer_id:
            raise ValueError(f"Reviewer folder differs from review record for {patch_id}")

        rgb_path = review_root / "patches" / f"{patch_id}_rgb.jpg"
        mask_path = (
            review_root / "annotations" / reviewer_id / f"{patch_id}_labels.png"
        )
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing reviewed RGB patch: {rgb_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing reviewed label mask: {mask_path}")

        with Image.open(rgb_path) as image:
            rgb_size = image.size
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image)
        if mask.ndim != 2 or mask.shape[::-1] != rgb_size:
            raise ValueError(f"RGB/mask dimensions differ for {patch_id}")
        values, counts = np.unique(mask, return_counts=True)
        if not set(map(int, values)).issubset(LABEL_VALUES):
            raise ValueError(f"Mask for {patch_id} contains unsupported labels")
        pixel_counts = {int(value): int(count) for value, count in zip(values, counts)}
        if pixel_counts.get(0, 0) + pixel_counts.get(1, 0) == 0:
            raise ValueError(f"Mask for {patch_id} has no scorable pixels")

        grouped[event_id].add(tile_id)
        records.append({
            "patch_id": patch_id,
            "event_id": event_id,
            "tile_id": tile_id,
            "reviewer_id": reviewer_id,
            "reviewed_at_utc": review.get("reviewed_at_utc", ""),
            "rgb_path": rgb_path,
            "mask_path": mask_path,
            "pixel_counts": pixel_counts,
        })

    rng = random.Random(seed)
    test_groups: set[tuple[str, str]] = set()
    for event_id, tile_ids in sorted(grouped.items()):
        shuffled_tiles = sorted(tile_ids)
        if len(shuffled_tiles) < 2:
            raise ValueError(
                f"Event {event_id!r} needs at least two reviewed tiles "
                "to appear in both dev and test"
            )
        rng.shuffle(shuffled_tiles)
        test_count = max(1, math.ceil(len(shuffled_tiles) * test_fraction))
        if test_count >= len(shuffled_tiles):
            raise ValueError(f"Not enough reviewed tiles to split event {event_id!r}")
        test_groups.update((event_id, tile_id) for tile_id in shuffled_tiles[:test_count])

    for record in records:
        record["split"] = (
            "test"
            if (record["event_id"], record["tile_id"]) in test_groups
            else "dev"
        )

    expected_ids = {
        split: {record["patch_id"] for record in records if record["split"] == split}
        for split in ("dev", "test")
    }
    for split, patch_ids in expected_ids.items():
        for folder in ("images", "masks"):
            existing = {
                path.stem.removesuffix("_mask")
                for path in (output_root / split / folder).glob("*.png")
            }
            stale = existing - patch_ids
            if stale:
                raise FileExistsError(
                    f"Stale generated files exist in {output_root / split / folder}: "
                    f"{sorted(stale)[:5]}. Move or remove the generated eval_set "
                    "before rebuilding it."
                )

    output_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        split = record["split"]
        image_dir = output_root / split / "images"
        mask_dir = output_root / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(record["rgb_path"]) as image:
            image.convert("RGB").save(image_dir / f"{record['patch_id']}.png")
        with Image.open(record["mask_path"]) as mask_image:
            mask_image.convert("L").save(mask_dir / f"{record['patch_id']}_mask.png")

    split_rows = []
    split_counts: dict[str, Counter[int]] = {
        "dev": Counter(),
        "test": Counter(),
    }
    for record in records:
        split_counts[record["split"]].update(record["pixel_counts"])
        split_rows.append({
            key: record[key]
            for key in (
                "patch_id", "split", "event_id", "tile_id",
                "reviewer_id", "reviewed_at_utc",
            )
        })

    split_manifest = output_root / "split_manifest.csv"
    with split_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)

    split_info = {
        "source": str(review_root),
        "seed": seed,
        "test_fraction_per_event": test_fraction,
        "split_group": "event_id + tile_id",
        "assignment": "deterministic shuffled tile groups, stratified by event",
        "review_status_required": "reviewed",
        "annotation_policy": (
            "One reviewed annotation per patch; this does not establish "
            "inter-reviewer agreement."
        ),
        "patch_counts": {
            split: len(expected_ids[split]) for split in ("dev", "test")
        },
        "tile_counts": {
            split: len({
                (record["event_id"], record["tile_id"])
                for record in records if record["split"] == split
            })
            for split in ("dev", "test")
        },
        "pixel_counts": {
            split: {str(label): split_counts[split].get(label, 0) for label in (0, 1, 255)}
            for split in ("dev", "test")
        },
    }
    (output_root / "dataset_info.json").write_text(
        json.dumps(split_info, indent=2) + "\n", encoding="utf-8"
    )

    for split in ("dev", "test"):
        counts = split_info["pixel_counts"][split]
        if counts["0"] == 0 or counts["1"] == 0:
            warnings.warn(
                f"The {split} split does not contain both cloud and non-cloud "
                "reference pixels; some metrics may be undefined.",
                stacklevel=2,
            )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("data/reference-review-v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/reviewed_eval_set"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    args = parser.parse_args()
    output_root = prepare_reviewed_eval(
        args.review_root,
        args.output_root,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )
    print(f"Prepared reviewed evaluation data at {output_root}")


if __name__ == "__main__":
    main()
