"""Run a pretrained RGB cloud SegFormer on reviewed patches."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from huggingface_hub import hf_hub_download
from PIL import Image
import segmentation_models_pytorch as smp

from . import evaluate_hsv as benchmark


MODEL_ID = "Burdenthrive/cloud-detection-segformer-mit_b4-RGB"
MODEL_REVISION = "4e2d31d9b36d196d6e6f91ba5b4d68763853bf17"
WEIGHTS_FILENAME = "segformer-mit_b4-RGB.pth"
TILE_SIZE = 512
CLASS_NAMES = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
LABEL_STATUS_VALUES = {
    "heuristic",
    "human_verified",
    "human_reviewed_single_reviewer",
}


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Model checkpoint must contain a state dictionary")
    for key in ("state_dict", "model_state_dict"):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Model checkpoint has no state dictionary")
    if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise ValueError("Model checkpoint contains non-tensor state entries")
    return checkpoint


def load_model(device):
    """Load the pinned MIT-licensed cloud checkpoint without remote code."""
    checkpoint_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=WEIGHTS_FILENAME,
        revision=MODEL_REVISION,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    state_dict = _extract_state_dict(checkpoint)

    prefixes = ("module.", "model.", "module.model.", "segformer.")
    candidates = [state_dict]
    for prefix in prefixes:
        candidate = {
            key.removeprefix(prefix): value
            for key, value in state_dict.items()
        }
        if len(candidate) == len(state_dict):
            candidates.append(candidate)

    model = smp.Segformer(
        encoder_name="mit_b4",
        encoder_weights=None,
        in_channels=3,
        classes=len(CLASS_NAMES),
    )
    expected_keys = set(model.state_dict())
    matched = next(
        (candidate for candidate in candidates if set(candidate) == expected_keys),
        None,
    )
    if matched is None:
        raise ValueError(
            "Checkpoint parameters do not match the documented SegFormer "
            "MiT-B4 RGB architecture"
        )
    model.load_state_dict(matched, strict=True)
    model.to(device)
    model.eval()
    return model


def _predict_tile(model, rgb_tile, device):
    image = torch.from_numpy(
        np.ascontiguousarray(rgb_tile.transpose(2, 0, 1))
    ).to(device=device, dtype=torch.float32)
    image = image.unsqueeze(0)
    with torch.inference_mode():
        logits = model(image)
        if logits.ndim != 4 or logits.shape[1] != len(CLASS_NAMES):
            raise ValueError(
                "Model output must have shape (1, 4, height, width)"
            )
        if logits.shape[-2:] != rgb_tile.shape[:2]:
            logits = functional.interpolate(
                logits,
                size=rgb_tile.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
        probabilities = torch.softmax(logits, dim=1)
        cloud_probability = probabilities[:, 1] + probabilities[:, 2]
        prediction = cloud_probability[0] >= 0.5
    return prediction.to(device="cpu").numpy().astype(np.uint8)


def predict_cloud_mask(model, rgb):
    """Apply the model in 512-pixel RGB windows and combine cloud classes."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("RGB values must be finite and in [0, 1]")
    height, width = rgb.shape[:2]
    if height % TILE_SIZE or width % TILE_SIZE:
        raise ValueError(
            f"Image dimensions must be divisible by {TILE_SIZE}"
        )

    device = next(model.parameters()).device
    prediction = np.zeros((height, width), dtype=np.uint8)
    for top in range(0, height, TILE_SIZE):
        for left in range(0, width, TILE_SIZE):
            tile = rgb[
                top:top + TILE_SIZE,
                left:left + TILE_SIZE,
            ]
            prediction[
                top:top + TILE_SIZE,
                left:left + TILE_SIZE,
            ] = _predict_tile(model, tile, device)
    return prediction


def evaluate_split(
    eval_root, out_root, split, label_status, device=None, model=None
):
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if label_status not in LABEL_STATUS_VALUES:
        raise ValueError("Invalid label_status")
    image_paths = sorted(
        (Path(eval_root) / split / "images").glob("*.png")
    )
    if not image_paths:
        raise FileNotFoundError(f"No evaluation PNGs found for {split}")

    device = device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    if model is None:
        model = load_model(torch.device(device))
    rows = []
    output = Path(out_root) / split / "segformer_cloudsen12plus"
    output.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        rgb, ref, valid, valid_source = benchmark.load_example(
            eval_root, split, image_path
        )
        start = time.perf_counter()
        prediction = predict_cloud_mask(model, rgb)
        elapsed = time.perf_counter() - start
        scores = benchmark.score_prediction(ref, prediction, valid)
        np.save(output / f"{image_path.stem}_mask.npy", prediction)
        Image.fromarray(prediction).save(
            output / f"{image_path.stem}_mask.png"
        )
        benchmark.save_comparison(
            rgb,
            ref,
            prediction,
            valid,
            output / f"{image_path.stem}_comparison.png",
            f"{image_path.stem} | SegFormer CloudSEN12Plus | {label_status}",
        )
        rows.append({
            "image_id": image_path.stem,
            "split": split,
            "method": "segformer_cloudsen12plus_mit_b4_rgb",
            "label_status": label_status,
            "valid_source": valid_source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "device": str(device),
            "predict_seconds": elapsed,
            "cloud_percent_valid": (
                100 * benchmark.safe_ratio(
                    prediction[valid].sum(), valid.sum()
                )
            ),
            **scores,
        })

    with (output / "metrics_per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        key: sum(row[key] for row in rows)
        for key in ("tp", "fp", "fn", "tn")
    }
    summary = {
        "method": "segformer_cloudsen12plus_mit_b4_rgb",
        "split": split,
        "label_status": label_status,
        "aggregation": "pooled_pixels",
        "n_images": len(rows),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": str(device),
        **benchmark.metrics_from_counts(**counts),
    }
    with (output / "metrics_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    metadata = {
        **summary,
        "class_names": list(CLASS_NAMES),
        "cloud_classes": ["thick_cloud", "thin_cloud"],
        "cloud_probability_threshold": 0.5,
        "input_channels": ["red", "green", "blue"],
        "input_normalization": "uint8 RGB divided by 255, no rescaling",
        "window_size": TILE_SIZE,
        "source_dataset": "CloudSEN12Plus Sentinel-2 RGB",
        "license": "MIT",
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("dev", "test", "both"), default="both"
    )
    parser.add_argument(
        "--label-status",
        default="human_reviewed_single_reviewer",
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        help="Default: MPS, then CUDA, then CPU when available.",
    )
    args = parser.parse_args()
    splits = ("dev", "test") if args.split == "both" else (args.split,)
    device = args.device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = load_model(torch.device(device))
    for split in splits:
        _, summary = evaluate_split(
            args.eval_root,
            args.out_root,
            split,
            args.label_status,
            device=device,
            model=model,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
