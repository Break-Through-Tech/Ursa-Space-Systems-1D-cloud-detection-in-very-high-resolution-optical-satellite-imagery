# Reviewed-patch cloud benchmark

The classical runner includes fixed HSV, Otsu HSV, and the HOT-style
`blue - 0.5 * red` RGB score. HOT's threshold is selected on the development
split by pooled-pixel F1, then frozen before the test split is evaluated.

The pretrained comparison uses
[Burdenthrive/cloud-detection-segformer-mit_b4-RGB](https://huggingface.co/Burdenthrive/cloud-detection-segformer-mit_b4-RGB),
a 4-class RGB SegFormer fine-tuned on CloudSEN12Plus. It uses RGB divided by
255, predicts 512-pixel windows, and maps the thick- and thin-cloud class
probabilities to cloud with a fixed 0.5 threshold. Cloud shadow is treated as
non-cloud to match the binary reference labels. The runner downloads the pinned
checkpoint (about 738 MB) to the Hugging Face cache on first use; it does not
execute model code from the Hub.

From the repository root, install the model dependencies and prepare the
reviewed patches:

```bash
python -m pip install -r requirements.txt
python notebooks/prepare_reviewed_eval.py
```

Run the classical and pretrained comparisons:

```bash
PYTHONPATH=notebooks python -m pipeline.evaluate_hot \
  --eval-root outputs/reviewed_eval_set \
  --out-root outputs/hot_reviewed_benchmark \
  --label-status human_reviewed_single_reviewer

PYTHONPATH=notebooks python -m pipeline.evaluate_pretrained_cloud \
  --eval-root outputs/reviewed_eval_set \
  --out-root outputs/pretrained_cloud_benchmark \
  --split both \
  --label-status human_reviewed_single_reviewer
```

Outputs include pooled and per-patch metrics, masks, comparison images, and
the selected HOT threshold/model metadata. Treat results as preliminary: the
reference set has one reviewer, the development/test split is tile-grouped
but both splits contain the same two events, and the pretrained model was
trained on lower-resolution Sentinel-2 RGB imagery rather than Maxar VHR.

## Closer-resolution pretrained checkpoint pilot

[`../Cloud_Adapter_HRC_WHU_Pilot.ipynb`](../Cloud_Adapter_HRC_WHU_Pilot.ipynb)
contains a Colab setup option for the Cloud-Adapter binary checkpoint trained
on HRC_WHU Google Earth RGB imagery (reported 0.5–15 m). The runner was also
executed locally on Apple MPS. It evaluates three fixed dev patches only. Each
1024 × 1024 patch is tiled into sixteen native-resolution 256 × 256 crops to
match the checkpoint's training crop size. The published inference config
specified 512 pixels, which caused the checkpoint's learned positional
embeddings to be skipped; the runner overrides the backbone and preprocessor
input sizes to 256, matching the official HRC training config. It supplies
uint8 BGR values to the MMSegmentation test step, which converts them to RGB
and applies mean `[123.675, 116.28, 103.53]` and std
`[58.395, 57.12, 57.375]`. This differs from the Cloud-Adapter demo's global
min-max scaling.

The Cloud-Adapter checkpoint is about 1.32 GB and requires the official custom
MMSegmentation modules and compiled MMCV operators, so this pilot is not
included in the lightweight local benchmark dependencies. On this Mac, MPS
inference required `PYTORCH_ENABLE_MPS_FALLBACK=1` for the unsupported bicubic
interpolation operation.

The same three-patch dev subset produced these pooled-pixel F1/IoU results:

| Method | Clear Brazil | Helene, ~4% cloud | Helene, ~76% cloud | Pooled |
|---|---:|---:|---:|---:|
| HOT RGB | 0.000 / 0.000 | 0.377 / 0.232 | 0.947 / 0.900 | 0.928 / 0.866 |
| CloudSEN12Plus SegFormer | undefined | 0.565 / 0.394 | 0.946 / 0.897 | 0.914 / 0.842 |
| Cloud-Adapter HRC_WHU | 0.000 / 0.000 | 0.777 / 0.635 | 0.963 / 0.929 | 0.954 / 0.912 |

These are exploratory dev-set results, not a model ranking: HOT's threshold
was tuned on dev, the sample is only three patches from two events, and labels
are from the initial single-reviewer pass. HRC_WHU is closer to Maxar in
resolution and shares RGB binary cloud segmentation as the task, but it is
Google Earth imagery and is not an exact sensor/domain match; the much finer
Maxar pixels also change the ground area represented by each 256-pixel crop.
The outputs and run metadata are saved under
`outputs/cloud_adapter_hrc_whu_pilot/`.
