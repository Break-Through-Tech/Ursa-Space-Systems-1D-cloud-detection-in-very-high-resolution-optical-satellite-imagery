"""Evaluate the Cloud-Adapter HRC_WHU checkpoint on the reviewed dev/test patches.

The checkpoint outputs a binary class map (argmax), so there is no threshold
or other parameter to tune on dev. Dev and test are scored independently and
never pooled together. Scoring reuses the shared benchmark helpers in
``evaluate_hsv`` so results are directly comparable with HOT, the dark-channel
prior, HSV and SegFormer.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import evaluate_hsv as benchmark


MODEL_REPO = "XavierJiezou/cloud-adapter-models"
MODEL_REVISION = "cf76783e0696f014e251f71d1b42f74b9bf0f0f6"
CHECKPOINT_NAME = "checkpoints/cloud-adapter/hrc_whu_full_weight.pth"
CLOUD_ADAPTER_COMMIT = "78c7ea50611b2381f45206a4269a2346116f4525"
METHOD = "cloud_adapter_hrc_whu"
WINDOW_SIZE = 256
LABEL_STATUS_VALUES = {
    "heuristic",
    "human_verified",
    "human_reviewed_single_reviewer",
}
# The original three-patch dev pilot, kept so it can be reproduced with --pilot.
PILOT_PATCH_IDS = (
    "Brazil-Flooding-May24_213131112121_r07168_c06144",
    "HurricaneHelene-Oct24_031313233020_r13312_c15360",
    "HurricaneHelene-Oct24_031313232113_r16384_c16384",
)


def _load_model(cloud_adapter_root, checkpoint_path, device):
    import sys

    import torch
    from mmengine import Config
    from mmseg.apis import init_model

    custom_code = Path(cloud_adapter_root) / "hugging_face"
    config = custom_code / "cloud-adapter-configs" / "binary_classes_256x256.py"
    if not config.is_file():
        raise FileNotFoundError(f"Missing Cloud-Adapter config: {config}")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    sys.path.insert(0, str(custom_code))
    from cloud_adapter.cloud_adapter_dinov2 import (  # noqa: F401
        CloudAdapterDinoVisionTransformer,
    )

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if device not in ("cpu", "mps") and not device.startswith("cuda"):
        raise ValueError(f"Unsupported inference device: {device}")

    cfg = Config.fromfile(str(config))
    # Match the checkpoint's 256-pixel training setup so its positional
    # embeddings load instead of being skipped by the published 512 config.
    cfg.model.backbone.img_size = WINDOW_SIZE
    cfg.model.data_preprocessor.size = (WINDOW_SIZE, WINDOW_SIZE)
    model = init_model(cfg, str(checkpoint_path), device=device)
    model.eval()
    return model


def _predict_native_patch(model, rgb_uint8, torch, SegDataSample):
    """Predict 256-pixel native-resolution windows and stitch them back."""
    height, width = rgb_uint8.shape[:2]
    if height % WINDOW_SIZE or width % WINDOW_SIZE:
        raise ValueError(
            f"Patch dimensions must be multiples of {WINDOW_SIZE}"
        )
    prediction = np.zeros((height, width), dtype=np.uint8)
    for top in range(0, height, WINDOW_SIZE):
        for left in range(0, width, WINDOW_SIZE):
            tile_rgb = rgb_uint8[
                top:top + WINDOW_SIZE,
                left:left + WINDOW_SIZE,
            ]
            tile_bgr = np.ascontiguousarray(tile_rgb[..., ::-1])
            sample = SegDataSample()
            sample.set_metainfo({
                "ori_shape": (WINDOW_SIZE, WINDOW_SIZE),
                "img_shape": (WINDOW_SIZE, WINDOW_SIZE),
                "pad_shape": (WINDOW_SIZE, WINDOW_SIZE),
                "scale_factor": (1.0, 1.0),
            })
            inputs = torch.from_numpy(
                tile_bgr.transpose(2, 0, 1)
            ).float()
            with torch.inference_mode():
                result = model.test_step({
                    "inputs": [inputs],
                    "data_samples": [sample],
                })[0]
            tile_mask = (
                result.pred_sem_seg.data.squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            if tile_mask.shape != (WINDOW_SIZE, WINDOW_SIZE):
                raise ValueError(
                    f"Unexpected model output shape {tile_mask.shape}"
                )
            if not np.isin(tile_mask, (0, 1)).all():
                raise ValueError("Cloud-Adapter output is not binary")
            prediction[
                top:top + WINDOW_SIZE,
                left:left + WINDOW_SIZE,
            ] = tile_mask
    return prediction


def _image_paths(eval_root, split, pilot):
    folder = Path(eval_root) / split / "images"
    if pilot:
        paths = [folder / f"{patch_id}.png" for patch_id in PILOT_PATCH_IDS]
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing pilot patches: {missing}")
        return paths
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No evaluation PNGs found in {folder}")
    return paths


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    """Replace NaN (undefined metrics) with None so the JSON stays valid."""
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def evaluate_split(
    model,
    eval_root,
    output_root,
    split,
    label_status,
    device,
    pilot=False,
):
    import torch
    from mmseg.structures import SegDataSample

    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if pilot and split != "dev":
        raise ValueError("The pilot subset is dev-only")

    output = Path(output_root) / split / METHOD
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for image_path in _image_paths(eval_root, split, pilot):
        rgb, ref, valid, valid_source = benchmark.load_example(
            eval_root, split, image_path
        )
        # load_example returns uint8/255; recover the exact uint8 values.
        rgb_uint8 = np.rint(rgb * 255).astype(np.uint8)

        start = time.perf_counter()
        prediction = _predict_native_patch(
            model, rgb_uint8, torch, SegDataSample
        )
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
            f"{image_path.stem} | Cloud-Adapter HRC_WHU | {label_status}",
        )
        rows.append({
            "image_id": image_path.stem,
            "split": split,
            "method": METHOD,
            "label_status": label_status,
            "valid_source": valid_source,
            "model_id": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "device": device,
            "predict_seconds": elapsed,
            "cloud_percent_valid": (
                100 * benchmark.safe_ratio(
                    prediction[valid].sum(), valid.sum()
                )
            ),
            **scores,
        })
        print(
            f"[{split}] {image_path.stem}: F1={scores['f1']:.3f} "
            f"IoU={scores['iou']:.3f} ({elapsed:.2f} s)",
            flush=True,
        )

    _write_csv(output / "metrics_per_image.csv", rows)

    counts = {
        key: sum(row[key] for row in rows)
        for key in ("tp", "fp", "fn", "tn")
    }
    seconds = [row["predict_seconds"] for row in rows]
    summary = {
        "method": METHOD,
        "split": split,
        "label_status": label_status,
        "aggregation": "pooled_pixels",
        "n_images": len(rows),
        "model_id": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "device": device,
        "mean_predict_seconds": float(np.mean(seconds)),
        **benchmark.metrics_from_counts(**counts),
    }
    _write_csv(output / "metrics_summary.csv", [summary])

    metadata = {
        **summary,
        "subset": "pilot (3 fixed dev patches)" if pilot else "all patches",
        "patch_ids": [row["image_id"] for row in rows],
        "checkpoint": CHECKPOINT_NAME,
        "cloud_adapter_source_commit": CLOUD_ADAPTER_COMMIT,
        "dataset": "HRC_WHU Google Earth RGB (reported 0.5-15 m resolution)",
        "maxar_input_resolution": "approximately 0.3-0.5 m",
        "window_size": WINDOW_SIZE,
        "windows_per_patch": (1024 // WINDOW_SIZE) ** 2,
        "config_input_size_override": [WINDOW_SIZE, WINDOW_SIZE],
        "input_dtype_range": "uint8 [0, 255]",
        "input_channel_order": "BGR to MMSegmentation preprocessor",
        "normalization": {
            "configured_mean_rgb": [123.675, 116.28, 103.53],
            "configured_std_rgb": [58.395, 57.12, 57.375],
            "description": "Applied by the config data preprocessor after BGR-to-RGB conversion",
        },
        "cloud_class": 1,
        "decision_rule": "argmax of binary output; no threshold tuned",
        "ignored_label": 255,
        "caveats": [
            "Initial single-reviewer labels; inter-reviewer agreement is unknown.",
            "Both splits contain the same two events (Hurricane Helene, Brazil Flooding).",
            "HRC_WHU resolution is closer to Maxar than Sentinel-2 but is not an exact sensor/domain match.",
        ],
    }
    (output / "run_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return rows, summary


def run(
    eval_root,
    output_root,
    cloud_adapter_root,
    checkpoint_path,
    split="both",
    label_status="human_reviewed_single_reviewer",
    device="cuda:0",
    pilot=False,
):
    """Load the checkpoint once and evaluate each requested split separately."""
    import mmcv
    import mmengine
    import mmseg
    import torch

    if label_status not in LABEL_STATUS_VALUES:
        raise ValueError(f"Unsupported label_status: {label_status}")
    if pilot:
        split = "dev"
    splits = ("dev", "test") if split == "both" else (split,)

    model = _load_model(cloud_adapter_root, checkpoint_path, device)
    versions = {
        "torch": torch.__version__,
        "mmcv": mmcv.__version__,
        "mmengine": mmengine.__version__,
        "mmseg": mmseg.__version__,
    }
    print("Library versions:", versions)

    summaries = {}
    for name in splits:
        _, summaries[name] = evaluate_split(
            model,
            eval_root,
            output_root,
            name,
            label_status,
            device,
            pilot=pilot,
        )
        metadata_path = Path(output_root) / name / METHOD / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["versions"] = versions
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    return summaries


def main(argv=None):
    """CLI entry point so the runner can use a separate Python environment."""
    import argparse
    import os

    # Colab exports MPLBACKEND for its notebook kernel; the venv lacks that backend.
    os.environ["MPLBACKEND"] = "Agg"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cloud-adapter-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--split", choices=("dev", "test", "both"), default="both"
    )
    parser.add_argument(
        "--label-status",
        choices=sorted(LABEL_STATUS_VALUES),
        default="human_reviewed_single_reviewer",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Only run the original three dev patches.",
    )
    args = parser.parse_args(argv)
    summaries = run(
        eval_root=args.eval_root,
        output_root=args.output_root,
        cloud_adapter_root=args.cloud_adapter_root,
        checkpoint_path=args.checkpoint,
        split=args.split,
        label_status=args.label_status,
        device=args.device,
        pilot=args.pilot,
    )
    for name, summary in summaries.items():
        print(
            f"{name}: n={summary['n_images']} "
            f"precision={summary['precision']:.3f} "
            f"recall={summary['recall']:.3f} "
            f"F1={summary['f1']:.3f} IoU={summary['iou']:.3f}"
        )


if __name__ == "__main__":
    main()
