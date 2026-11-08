"""Tune the HOT-style RGB threshold on dev, then evaluate frozen settings."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from . import evaluate_hsv as benchmark
from .methods.hot import HOTBaseline


DEFAULT_THRESHOLDS = tuple(
    round(float(value), 3) for value in np.arange(-0.25, 0.501, 0.025)
)


def sweep_dev(
    eval_root,
    out_root,
    label_status,
    thresholds=DEFAULT_THRESHOLDS,
    slope=0.5,
    intercept=0.0,
):
    """Select HOT threshold by pooled dev F1; does not read test labels."""
    thresholds = tuple(float(value) for value in thresholds)
    if not thresholds or not all(
        math.isfinite(value) and -1 <= value <= 1 for value in thresholds
    ):
        raise ValueError("thresholds must be non-empty values in [-1, 1]")

    image_paths = sorted(
        (Path(eval_root) / "dev" / "images").glob("*.png")
    )
    if not image_paths:
        raise FileNotFoundError("No evaluation PNGs found for dev")

    per_image = []
    for image_path in image_paths:
        rgb, ref, valid, _ = benchmark.load_example(
            eval_root, "dev", image_path
        )
        features = benchmark.features_from_rgb(rgb)
        for threshold in thresholds:
            method = HOTBaseline(threshold, slope, intercept)
            prediction = method.predict(features, valid)
            scores = benchmark.score_prediction(ref, prediction, valid)
            per_image.append({
                "image_id": image_path.stem,
                "label_status": label_status,
                **method.last_thresholds,
                **scores,
            })

    summaries = []
    for threshold in thresholds:
        rows = [row for row in per_image if row["hot_threshold"] == threshold]
        counts = {
            key: sum(row[key] for row in rows)
            for key in ("tp", "fp", "fn", "tn")
        }
        summaries.append({
            "method": "hot_rgb",
            "split": "dev",
            "label_status": label_status,
            "aggregation": "pooled_pixels",
            "n_images": len(rows),
            "hot_threshold": threshold,
            "hot_slope": float(slope),
            "hot_intercept": float(intercept),
            **benchmark.metrics_from_counts(**counts),
        })

    summaries.sort(
        key=lambda row: (
            row["f1"],
            row["iou"],
            row["precision"],
            -row["hot_threshold"],
        ),
        reverse=True,
    )
    best = summaries[0]
    output = Path(out_root) / "dev"
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("threshold_sweep_per_image.csv", per_image),
        ("threshold_sweep_summary.csv", summaries),
    ):
        with (output / filename).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    config = {
        "threshold": best["hot_threshold"],
        "slope": float(slope),
        "intercept": float(intercept),
        "selected_on": "dev pooled-pixel F1",
        "label_status": label_status,
    }
    (output / "selected_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return config, summaries


def run_benchmark(eval_root, out_root, label_status):
    config, sweep = sweep_dev(eval_root, out_root, label_status)
    dev_rows, dev_summary = benchmark.evaluate(
        eval_root,
        split="dev",
        out_root=out_root,
        label_status=label_status,
        hot_threshold=config["threshold"],
        hot_slope=config["slope"],
        hot_intercept=config["intercept"],
    )
    test_rows, test_summary = benchmark.evaluate(
        eval_root,
        split="test",
        out_root=out_root,
        label_status=label_status,
        hot_threshold=config["threshold"],
        hot_slope=config["slope"],
        hot_intercept=config["intercept"],
    )
    return config, sweep, dev_rows, dev_summary, test_rows, test_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--label-status",
        default="human_reviewed_single_reviewer",
    )
    args = parser.parse_args()
    config, _, _, dev_summary, _, test_summary = run_benchmark(
        args.eval_root, args.out_root, args.label_status
    )
    print("Dev-selected HOT configuration:", json.dumps(config))
    print("Dev summary:", json.dumps(dev_summary, indent=2))
    print("Test summary:", json.dumps(test_summary, indent=2))


if __name__ == "__main__":
    main()
