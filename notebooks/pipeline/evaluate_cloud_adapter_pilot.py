"""Evaluate the Cloud-Adapter HRC_WHU checkpoint on three dev patches."""

import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


MODEL_REPO = "XavierJiezou/cloud-adapter-models"
MODEL_REVISION = "cf76783e0696f014e251f71d1b42f74b9bf0f0f6"
CHECKPOINT_NAME = "checkpoints/cloud-adapter/hrc_whu_full_weight.pth"
CLOUD_ADAPTER_COMMIT = "78c7ea50611b2381f45206a4269a2346116f4525"
WINDOW_SIZE = 256
PATCH_IDS = (
    "Brazil-Flooding-May24_213131112121_r07168_c06144",
    "HurricaneHelene-Oct24_031313233020_r13312_c15360",
    "HurricaneHelene-Oct24_031313232113_r16384_c16384",
)


def _score(ref, pred):
    scored = ref != 255
    y = ref[scored] == 1
    p = pred[scored] == 1
    tp = int(np.count_nonzero(y & p))
    fp = int(np.count_nonzero(~y & p))
    fn = int(np.count_nonzero(y & ~p))
    tn = int(np.count_nonzero(~y & ~p))
    f1_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_scored": tp + fp + fn + tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / f1_denominator if f1_denominator else None,
        "iou": tp / iou_denominator if iou_denominator else None,
    }


def _load_model(cloud_adapter_root, checkpoint_path, device):
    import torch
    from mmseg.apis import init_model
    from mmengine import Config

    custom_code = Path(cloud_adapter_root) / "hugging_face"
    config = custom_code / "cloud-adapter-configs" / "binary_classes_256x256.py"
    if not config.is_file():
        raise FileNotFoundError(f"Missing Cloud-Adapter config: {config}")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    import sys

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


def _predict_native_patch(model, rgb, torch, SegDataSample):
    height, width = rgb.shape[:2]
    if height % WINDOW_SIZE or width % WINDOW_SIZE:
        raise ValueError(
            f"Patch dimensions must be multiples of {WINDOW_SIZE}"
        )
    prediction = np.zeros((height, width), dtype=np.uint8)
    for top in range(0, height, WINDOW_SIZE):
        for left in range(0, width, WINDOW_SIZE):
            tile_rgb = rgb[
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


def _save_comparison(rgb, ref, pred, output_path, title):
    import matplotlib.pyplot as plt

    scored = ref != 255
    ref_vis = np.zeros((*ref.shape, 3), dtype=np.uint8)
    ref_vis[ref == 1] = (255, 220, 0)
    ref_vis[~scored] = (100, 100, 100)
    pred_vis = np.repeat((pred * 255)[..., None], 3, axis=2)
    errors = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    errors[scored & (pred == 1) & (ref == 0)] = (255, 0, 0)
    errors[scored & (pred == 0) & (ref == 1)] = (0, 255, 255)
    errors[~scored] = (100, 100, 100)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for axis, image, label in zip(
        axes,
        (rgb, ref_vis, pred_vis, errors),
        ("RGB", "Reference (yellow cloud)", "Model (white cloud)", "Errors (red FP, cyan FN)"),
    ):
        axis.imshow(image)
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def run_pilot(
    eval_root,
    output_root,
    cloud_adapter_root,
    checkpoint_path,
    device="cuda:0",
):
    """Run fixed checkpoint on three selected dev patches; no threshold tuning."""
    import torch
    import mmcv
    import mmengine
    import mmseg
    from mmseg.structures import SegDataSample

    eval_root = Path(eval_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model = _load_model(cloud_adapter_root, checkpoint_path, device)
    records = []
    pooled = {key: 0 for key in ("tp", "fp", "fn", "tn")}

    for patch_id in PATCH_IDS:
        image_path = eval_root / "dev" / "images" / f"{patch_id}.png"
        mask_path = eval_root / "dev" / "masks" / f"{patch_id}_mask.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing dev image/mask pair for {patch_id}"
            )
        rgb = np.asarray(
            Image.open(image_path).convert("RGB"),
            dtype=np.uint8,
        )
        ref = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if rgb.shape != (1024, 1024, 3) or ref.shape != (1024, 1024):
            raise ValueError(f"Expected 1024x1024 pair for {patch_id}")
        if not np.isin(ref, (0, 1, 255)).all():
            raise ValueError(f"Unsupported reference labels in {patch_id}")

        start = time.perf_counter()
        pred = _predict_native_patch(model, rgb, torch, SegDataSample)
        elapsed = time.perf_counter() - start
        metrics = _score(ref, pred)
        for key in pooled:
            pooled[key] += metrics[key]
        records.append({
            "patch_id": patch_id,
            "predict_seconds": elapsed,
            **metrics,
        })

        np.save(output_root / f"{patch_id}_prediction.npy", pred)
        Image.fromarray(pred).save(
            output_root / f"{patch_id}_prediction.png"
        )
        _save_comparison(
            rgb,
            ref,
            pred,
            output_root / f"{patch_id}_comparison.png",
            (
                f"{patch_id} | F1={metrics['f1']:.3f}, "
                f"IoU={metrics['iou']:.3f}"
                if metrics["f1"] is not None
                else f"{patch_id} | F1 undefined, IoU undefined"
            ),
        )

    tp, fp, fn, tn = (pooled[key] for key in ("tp", "fp", "fn", "tn"))
    pooled_metrics = {
        **pooled,
        "n_scored": tp + fp + fn + tn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        "iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
    }
    metadata = {
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "checkpoint": CHECKPOINT_NAME,
        "cloud_adapter_source_commit": CLOUD_ADAPTER_COMMIT,
        "dataset": "HRC_WHU Google Earth RGB (reported 0.5-15 m resolution)",
        "maxar_input_resolution": "approximately 0.3-0.5 m",
        "input_size": [WINDOW_SIZE, WINDOW_SIZE],
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
        "ignored_label": 255,
        "reviewer": "reviewer1",
        "split": "dev only",
        "patch_ids": list(PATCH_IDS),
        "per_patch_metrics": records,
        "pooled_metrics": pooled_metrics,
        "versions": {
            "torch": torch.__version__,
            "mmcv": mmcv.__version__,
            "mmengine": mmengine.__version__,
            "mmseg": mmseg.__version__,
        },
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0)
            if device.startswith("cuda")
            else "Apple Metal Performance Shaders"
            if device == "mps"
            else "CPU"
        ),
        "caveats": [
            "Three patches only; exploratory pilot, not model ranking.",
            "Initial single-reviewer labels; the second review is not incorporated.",
            "HRC_WHU resolution is closer to Maxar than Sentinel-2 but is not an exact sensor/domain match.",
        ],
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_root / "metrics_per_patch.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records, pooled_metrics, metadata
