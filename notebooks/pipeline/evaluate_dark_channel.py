from pathlib import Path
import csv
import json
import time

import numpy as np
from PIL import Image

from . import evaluate_hsv as benchmark
from .methods.dark_channel import DarkChannelPrior, compute_dark_channel


DEFAULT_PATCH_SIZES = (3, 7, 15, 31)
DEFAULT_THRESHOLDS = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)


def _image_paths(eval_root, split):
    paths = sorted((Path(eval_root) / split / "images").glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No evaluation PNGs found for {split}")
    return paths


def _pooled_summary(rows, patch_size, threshold, split, label_status):
    counts = {
        key: sum(row[key] for row in rows)
        for key in ("tp", "fp", "fn", "tn")
    }
    return {
        "method": "dark_channel_prior",
        "split": split,
        "label_status": label_status,
        "aggregation": "pooled_pixels",
        "n_images": len(rows),
        "patch_size": patch_size,
        "dark_channel_threshold": threshold,
        **benchmark.metrics_from_counts(**counts),
    }


def sweep_dev(
    eval_root,
    out_root,
    label_status,
    patch_sizes=DEFAULT_PATCH_SIZES,
    thresholds=DEFAULT_THRESHOLDS,
):
    """Choose patch size and threshold by pooled dev F1; never reads test labels."""
    patch_sizes = tuple(patch_sizes)
    thresholds = tuple(float(value) for value in thresholds)
    if len(set(patch_sizes)) < 3:
        raise ValueError("The dev sweep requires at least three patch sizes")
    if any(size <= 0 or size % 2 == 0 for size in patch_sizes):
        raise ValueError("All patch sizes must be positive odd integers")
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Thresholds must be a non-empty sequence in [0, 1]")

    images = _image_paths(eval_root, "dev")
    per_image = []
    summaries = []
    for image_path in images:
        rgb, ref, valid, _ = benchmark.load_example(
            eval_root, "dev", image_path
        )
        dark_channels = {
            size: compute_dark_channel(rgb, size, valid)
            for size in patch_sizes
        }
        for size in patch_sizes:
            for threshold in thresholds:
                prediction = (
                    (dark_channels[size] >= threshold) & valid
                ).astype(np.uint8)
                scores = benchmark.score_prediction(ref, prediction, valid)
                if scores["n_scored"] == 0:
                    raise ValueError(
                        f"No scorable reference pixels for {image_path.stem}"
                    )
                per_image.append({
                    "image_id": image_path.stem,
                    "label_status": label_status,
                    "patch_size": size,
                    "dark_channel_threshold": threshold,
                    **scores,
                })

    for size in patch_sizes:
        for threshold in thresholds:
            subset = [
                row for row in per_image
                if row["patch_size"] == size
                and row["dark_channel_threshold"] == threshold
            ]
            summaries.append(
                _pooled_summary(
                    subset, size, threshold, "dev", label_status
                )
            )

    summaries.sort(
        key=lambda row: (
            row["f1"], row["iou"], row["precision"],
            -row["patch_size"], -row["dark_channel_threshold"],
        ),
        reverse=True,
    )
    best = summaries[0]
    output = Path(out_root) / "dev"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("patch_size_threshold_sweep_per_image.csv", per_image),
        ("patch_size_threshold_sweep_summary.csv", summaries),
    ):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    config = {
        "patch_size": best["patch_size"],
        "threshold": best["dark_channel_threshold"],
        "selected_on": "dev pooled-pixel F1",
        "label_status": label_status,
    }
    (output / "selected_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return config, summaries


def _diagnostic_windows(rgb, ref, pred, valid, window_size=256, stride=128):
    height, width = ref.shape
    candidates = []
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom = min(top + window_size, height)
            right = min(left + window_size, width)
            if bottom <= top or right <= left:
                continue
            scored = valid[top:bottom, left:right] & (ref[top:bottom, left:right] != 255)
            fp = int(np.count_nonzero(
                scored & (pred[top:bottom, left:right] == 1)
                & (ref[top:bottom, left:right] == 0)
            ))
            fn = int(np.count_nonzero(
                scored & (pred[top:bottom, left:right] == 0)
                & (ref[top:bottom, left:right] == 1)
            ))
            candidates.append((fp + fn, fp, fn, top, left, bottom, right))
    return sorted(candidates, reverse=True)


def _save_failure_cases(
    test_examples, output_dir, label_status, limit=3
):
    """Save top error-heavy crops and concise, explicitly provisional notes."""
    cases = []
    for image_path, rgb, ref, pred, valid in test_examples:
        for total, fp, fn, top, left, bottom, right in _diagnostic_windows(
            rgb, ref, pred, valid
        ):
            cases.append({
                "image_id": image_path.stem,
                "total_errors": total,
                "fp": fp,
                "fn": fn,
                "top": top,
                "left": left,
                "bottom": bottom,
                "right": right,
                "rgb": rgb,
                "ref": ref,
                "pred": pred,
                "valid": valid,
            })
    cases.sort(key=lambda case: (case["total_errors"], case["fp"] + case["fn"]), reverse=True)

    selected = [case for case in cases if case["total_errors"] > 0][:limit]
    if len(selected) < limit:
        selected.extend(
            case for case in cases
            if case["total_errors"] == 0
            and all(
                (case["image_id"], case["top"], case["left"])
                != (item["image_id"], item["top"], item["left"])
                for item in selected
            )
        )
        selected = selected[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    notes = []
    for number, case in enumerate(selected, start=1):
        fp, fn = case["fp"], case["fn"]
        if fp and fp >= fn:
            note = (
                "False-positive-heavy crop: inspect whether bright ground, "
                "smoke, or local contrast triggered the cloud threshold."
            )
            kind = "false_positive_heavy"
        elif fn:
            note = (
                "False-negative-heavy crop: inspect thin or low-contrast "
                "cloud and whether the neighborhood size erased its boundary."
            )
            kind = "false_negative_heavy"
        else:
            note = "No scored mismatch in this crop; saved as a sanity-check example."
            kind = "no_observed_error"

        stem = (
            f"case_{number:02d}_{case['image_id']}"
            f"_r{case['top']:05d}_c{case['left']:05d}"
        )
        bounds = (
            slice(case["top"], case["bottom"]),
            slice(case["left"], case["right"]),
        )
        benchmark.save_comparison(
            case["rgb"][bounds], case["ref"][bounds],
            case["pred"][bounds], case["valid"][bounds],
            output_dir / f"{stem}.png",
            f"{stem} | FP={fp} FN={fn} | {kind}",
        )
        notes.append({
            "case": stem,
            "image_id": case["image_id"],
            "row_start": case["top"],
            "column_start": case["left"],
            "false_positives": fp,
            "false_negatives": fn,
            "case_type": kind,
            "label_status": label_status,
            "review_note": note,
        })

    notes_path = output_dir / "failure_case_notes.csv"
    if notes:
        with notes_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(notes[0]))
            writer.writeheader()
            writer.writerows(notes)
    return notes


def evaluate_test(eval_root, out_root, label_status, patch_size, threshold):
    """Run one frozen dev-selected configuration on held-out test images."""
    images = _image_paths(eval_root, "test")
    method = DarkChannelPrior(patch_size=patch_size, threshold=threshold)
    output = Path(out_root) / "test" / method.name
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    examples = []
    for image_path in images:
        rgb, ref, valid, valid_source = benchmark.load_example(
            eval_root, "test", image_path
        )
        features = benchmark.features_from_rgb(rgb)
        start = time.perf_counter()
        prediction = method.predict(features, valid)
        elapsed = time.perf_counter() - start
        scores = benchmark.score_prediction(ref, prediction, valid)
        if scores["n_scored"] == 0:
            raise ValueError(
                f"No scorable reference pixels for {image_path.stem}"
            )
        np.save(output / f"{image_path.stem}_mask.npy", prediction)
        Image.fromarray(prediction).save(output / f"{image_path.stem}_mask.png")
        benchmark.save_comparison(
            rgb, ref, prediction, valid,
            output / f"{image_path.stem}_comparison.png",
            f"{image_path.stem} | dark_channel_prior | {label_status}",
        )
        rows.append({
            "image_id": image_path.stem,
            "split": "test",
            "method": method.name,
            "label_status": label_status,
            "valid_source": valid_source,
            "patch_size": patch_size,
            "dark_channel_threshold": threshold,
            "predict_seconds": elapsed,
            **scores,
        })
        examples.append((image_path, rgb, ref, prediction, valid))

    summary = _pooled_summary(
        rows, patch_size, threshold, "test", label_status
    )
    for name, data in (
        ("metrics_per_image.csv", rows),
        ("metrics_summary.csv", [summary]),
    ):
        with (Path(out_root) / "test" / name).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    notes = _save_failure_cases(
        examples,
        Path(out_root) / "test" / method.name / "failure_cases",
        label_status=label_status,
    )
    print(f"Dark Channel Prior test label status: {label_status}")
    print(f"Saved {len(notes)} diagnostic crops and notes.")
    return rows, summary, notes
