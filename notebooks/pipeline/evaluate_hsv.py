# 3. Add the evaluation and image-saving code
from pathlib import Path
import csv
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .methods.hsv_threshold import HSVThreshold
from .methods.hsv_otsu import HSVOtsu
from .methods.hot import HOTBaseline


def features_from_rgb(rgb):
    """Use already normalized evaluation RGB without another stretch."""
    rgb = np.asarray(rgb, dtype=np.float32)

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected an RGB image")

    v = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)

    s = np.divide(
        v - minimum,
        v,
        out=np.zeros_like(v),
        where=v > 0,
    )

    # NIR/NDVI placeholders are unused by these two HSV methods.
    zeros = np.zeros_like(v)

    return np.dstack([rgb, v, s, zeros, zeros]).astype(np.float32)


def load_example(eval_root, split, image_path):
    image_path = Path(image_path)
    folder = Path(eval_root) / split
    ref_path = folder / "masks" / f"{image_path.stem}_mask.png"

    rgb = (
        np.asarray(
            Image.open(image_path).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
    )
    ref = np.asarray(Image.open(ref_path))

    if ref.ndim != 2 or ref.shape != rgb.shape[:2]:
        raise ValueError(
            f"Image/mask dimensions differ: {image_path.stem}"
        )

    if not np.isin(ref, [0, 1, 255]).all():
        raise ValueError("Reference labels must be 0, 1, or 255")

    valid_path = (
        folder / "valid" / f"{image_path.stem}_valid.png"
    )

    if valid_path.exists():
        valid = np.asarray(Image.open(valid_path)) > 0

        if valid.shape != ref.shape:
            raise ValueError("Validity mask dimensions differ")

        valid_source = "independent_valid_mask"
    else:
        # Use benchmark coverage when no independent valid mask exists.
        valid = ref != 255
        valid_source = "reference_coverage"

    return rgb, ref, valid, valid_source


def safe_ratio(a, b):
    return float(a / b) if b else float("nan")


def metrics_from_counts(tp, fp, fn, tn):
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_scored": int(tp + fp + fn + tn),
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "iou": safe_ratio(tp, tp + fp + fn),
    }


def score_prediction(ref, pred, valid):
    if not (ref.shape == pred.shape == valid.shape):
        raise ValueError(
            "Reference, prediction, and validity grids must match"
        )

    if not np.isin(pred, [0, 1]).all():
        raise ValueError("Prediction must contain only 0 and 1")

    scored = valid & (ref != 255)
    y = ref[scored] == 1
    p = pred[scored] == 1

    return metrics_from_counts(
        np.count_nonzero(y & p),
        np.count_nonzero(~y & p),
        np.count_nonzero(y & ~p),
        np.count_nonzero(~y & ~p),
    )


def save_comparison(rgb, ref, pred, valid, path, title):
    scored = valid & (ref != 255)

    reference = np.zeros((*ref.shape, 3), dtype=np.uint8)
    reference[ref == 1] = [255, 220, 0]
    reference[~scored] = [100, 100, 100]

    prediction = np.repeat(
        (pred * 255)[..., None], 3, axis=-1
    )

    errors = (rgb * 90).astype(np.uint8)
    errors[scored & (pred == 1) & (ref == 0)] = [255, 0, 0]
    errors[scored & (pred == 0) & (ref == 1)] = [0, 255, 255]
    errors[~scored] = [100, 100, 100]

    panels = [
        ("RGB", (rgb * 255).astype(np.uint8)),
        ("Reference: yellow=cloud", reference),
        ("Prediction: white=cloud", prediction),
        ("Errors: red=FP cyan=FN", errors),
    ]

    canvas = Image.new("RGB", (1024, 300), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill="black")

    for i, (label, array) in enumerate(panels):
        draw.text((i * 256 + 4, 24), label, fill="black")

        resampling = (
            Image.Resampling.LANCZOS
            if i == 0 else Image.Resampling.NEAREST
        )

        panel = ImageOps.contain(
            Image.fromarray(array),
            (252, 252),
            method=resampling,
        )
        canvas.paste(panel, (i * 256 + 2, 46))

    canvas.save(path)


def evaluate(
    eval_root,
    split="dev",
    out_root="outputs/classical_reviewed_benchmark",
    label_status="heuristic",
    v_min=0.70,
    s_max=0.20,
    hot_threshold=0.20,
    hot_slope=0.5,
    hot_intercept=0.0,
):
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")

    if label_status not in {
        "heuristic", "human_verified", "human_reviewed_single_reviewer"
    }:
        raise ValueError("Invalid label_status")

    images = sorted(
        (Path(eval_root) / split / "images").glob("*.png")
    )

    if not images:
        raise FileNotFoundError(
            f"No evaluation PNGs found for {split}"
        )

    fixed = HSVThreshold(v_min=v_min, s_max=s_max)
    fixed.name = "hsv_fixed"
    fixed.last_thresholds = {
        "v_threshold": float(v_min),
        "s_threshold": float(s_max),
        "threshold_source": "fixed",
    }

    methods = [
        fixed,
        HSVOtsu(s_max=s_max, v_fallback=v_min),
        HOTBaseline(
            threshold=hot_threshold,
            slope=hot_slope,
            intercept=hot_intercept,
        ),
    ]
    rows = []

    for image_path in images:
        rgb, ref, valid, valid_source = load_example(
            eval_root, split, image_path
        )
        features = features_from_rgb(rgb)

        for method in methods:
            start = time.perf_counter()
            pred = method.predict(features, valid)
            elapsed = time.perf_counter() - start

            scores = score_prediction(ref, pred, valid)

            folder = Path(out_root) / split / method.name
            folder.mkdir(parents=True, exist_ok=True)

            np.save(
                folder / f"{image_path.stem}_mask.npy",
                pred,
            )
            Image.fromarray(pred).save(
                folder / f"{image_path.stem}_mask.png"
            )

            save_comparison(
                rgb, ref, pred, valid,
                folder / f"{image_path.stem}_comparison.png",
                f"{image_path.stem} | {method.name} | {label_status}",
            )

            rows.append({
                "image_id": image_path.stem,
                "split": split,
                "method": method.name,
                "label_status": label_status,
                "valid_source": valid_source,
                **method.last_thresholds,
                "predict_seconds": elapsed,
                "cloud_percent_valid": (
                    100 * safe_ratio(
                        pred[valid].sum(), valid.sum()
                    )
                ),
                **scores,
            })

    destination = Path(out_root) / split

    metric_fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))
    with (destination / "metrics_per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=metric_fields
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = []

    for method in methods:
        subset = [
            row for row in rows
            if row["method"] == method.name
        ]
        counts = {
            key: sum(row[key] for row in subset)
            for key in ["tp", "fp", "fn", "tn"]
        }

        summary.append({
            "method": method.name,
            "split": split,
            "label_status": label_status,
            "aggregation": "pooled_pixels",
            "n_images": len(subset),
            **metrics_from_counts(**counts),
        })

    with (destination / "metrics_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary[0])
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved {split} masks, comparisons, and CSVs.")
    print("Reference status:", label_status)

    return rows, summary
