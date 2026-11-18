# Cloud Detection in Very-High-Resolution Optical Satellite Imagery

**Classical signal processing vs. pretrained deep learning models**

Break Through Tech AI Studio, Fall 2026 · Host company: Ursa Space Systems

---

### 👥 **Team Members**

| Name                | GitHub Handle       | Contribution |
|---------------------|---------------------|--------------|
| Lisa Wilder         | @Wilder407          | TBD          |
| Aqila Nasiry        | @Devcode5           | TBD          |
| Samanvitha Paderthi | @samanvithapaderthi | TBD          |
| Harshita Kumari     | @hk10877            | TBD          |
| Justin Vu           | @justincvu          | TBD          |
| Star Vann           | @starieeee          | TBD          |
| Zahraa Al-Naami     | @alnaamizahraa      | TBD          |

---

## 🎯 **Project Highlights**

- Built a reproducible benchmark on ~30–50 cm Maxar Open Data RGB imagery. It compares two classical cloud-detection methods, a HOT-style haze index and the dark-channel prior (DCP), against a pretrained deep learning model, Cloud-Adapter (HRC_WHU checkpoint).
- Created a human-reviewed reference set of 45 patches (1024 × 1024) and split it into dev and test sets by tile. Classical thresholds were tuned on dev only, then frozen before the single test run. Cloud-Adapter needed no tuning.
- On the held-out test split, Cloud-Adapter scored highest (F1 0.965, IoU 0.932). DCP was close behind (F1 0.929, IoU 0.867), with no learned weights. HOT scored F1 0.858, IoU 0.751.
- Also evaluated, as supporting comparisons: two simpler HSV threshold methods, and a SegFormer checkpoint trained on 10 m Sentinel-2 imagery that was set aside because its training resolution is far coarser than Maxar's.
- These results are preliminary; see [Results](#-results--key-findings).

---

## 👩🏽‍💻 **Setup and Installation**

### 1. Clone and install

```bash
git clone https://github.com/Break-Through-Tech/Ursa-Space-Systems-1D-cloud-detection-in-very-high-resolution-optical-satellite-imagery.git
cd Ursa-Space-Systems-1D-cloud-detection-in-very-high-resolution-optical-satellite-imagery
python -m pip install -r requirements.txt
```

[`requirements.txt`](requirements.txt) covers the classical methods and the SegFormer model. Cloud-Adapter has its own, heavier environment and runs in Colab (see step 4).

### 2. Get the imagery

Raw tiles (`.tif`, `.gpkg`) are not stored in git. Either:

- download them from the team's shared Drive folder, or
- run [`download_tiles.py`](download_tiles.py) from the repo root, which pulls the same files from the Maxar Open Data STAC catalog.

[`data/README.md`](data/README.md) lists the events and tiles, and explains how the reference labels were made.

### 3. Build the evaluation set and run the classical benchmarks

```bash
python notebooks/prepare_reviewed_eval.py

PYTHONPATH=notebooks python -m pipeline.evaluate_hot \
  --eval-root outputs/reviewed_eval_set \
  --out-root outputs/hot_reviewed_benchmark \
  --label-status human_reviewed_single_reviewer
```

This runs HOT and both HSV methods. The dark-channel prior is run with [`notebooks/pipeline/evaluate_dark_channel.py`](notebooks/pipeline/evaluate_dark_channel.py), and the SegFormer comparison with [`evaluate_pretrained_cloud.py`](notebooks/pipeline/evaluate_pretrained_cloud.py). [`notebooks/pipeline/README.md`](notebooks/pipeline/README.md) covers the runners, preprocessing and output layout.

### 4. Run Cloud-Adapter (Colab GPU)

Open the [Cloud-Adapter HRC-WHU notebook in Colab](https://colab.research.google.com/drive/1arhVOTe_zwg_0YTPH7AJ3LoSXU1UiAfp?usp=sharing) (source: [`notebooks/Cloud_Adapter_HRC_WHU_Pilot.ipynb`](notebooks/Cloud_Adapter_HRC_WHU_Pilot.ipynb)) with a GPU runtime. It:

1. installs MMSegmentation, MMCV and the other dependencies into a separate Python 3.10 environment;
2. downloads the ~1.32 GB checkpoint;
3. builds the evaluation set from the committed reviews (in a reused runtime, delete `outputs/reviewed_eval_set` first so it is rebuilt);
4. runs all 30 dev and 15 test patches, taking about 1.6 s per patch on a Colab GPU.

Download the zip at the end and unzip it into `outputs/cloud_adapter_benchmark/`.

---

## 🏗️ **Project Overview**

This project is part of the Break Through Tech AI Program's AI Studio, with Ursa Space Systems as the host company.

Ursa Space Systems turns satellite data into intelligence for commercial and government clients. Clouds in commercial very-high-resolution (VHR) optical imagery hide ground features and degrade every downstream product built on that imagery. Ursa needs a dependable way to mask cloud-covered regions automatically.

**Objective:** produce a clear, reproducible comparison of cloud-masking approaches for VHR RGB imagery, rather than a single accuracy number:

1. At least two classical methods and one pretrained model, each runnable end to end.
2. Evaluation against a small human-reviewed reference set, using IoU, precision, recall and F1 on cloud vs. non-cloud pixels.
3. A qualitative error analysis of where each method fails, for example on thin cloud, smoke, or bright non-cloud surfaces.

The full brief is in [`Challenge-Project-Overview.md`](Challenge-Project-Overview.md). The evaluation rules are in [`Evaluation_Protocol.pdf`](Evaluation_Protocol.pdf).

---

## 📊 **Data Exploration**

### Dataset

- **Source:** [Maxar Open Data Program](https://radiantearth.github.io/stac-browser/#/external/maxar-opendata.s3.amazonaws.com/events) (now Vantor)
- **Format:** cloud-optimized GeoTIFFs, ~30–50 cm resolution
- **Events used:**
  - Hurricane Helene, Oct 2024 (9 tiles)
  - Brazil Flooding, May 2024 (6 tiles; reported to be cloud-free)
  - Marshall Fire (15 tiles; no multispectral asset)

### Reference labels

There are no ground-truth cloud masks for this imagery. The team created reference labels as follows:

1. Up to three 1024 × 1024 patches were sampled from each tile, stratified by heuristic cloud fraction.
2. Candidate masks were generated with a brightness/saturation heuristic.
3. Each mask was reviewed and corrected by one team member ([`notebooks/Review_Cloud_Labels.ipynb`](notebooks/Review_Cloud_Labels.ipynb)).
4. [`notebooks/prepare_reviewed_eval.py`](notebooks/prepare_reviewed_eval.py) split the reviewed patches by tile (seed 42, 25% of each event's tiles held out for test):

| Split | Patches | Tiles |
|---|---:|---:|
| dev  | 30 | 10 |
| test | 15 | 5  |

### Preprocessing

The source imagery was downsampled to 10% resolution using bilinear resampling. RGB values were normalized to [0, 1] with percentile winsorization. NIR was percentile-normalized, and NDVI was derived from NIR and red. Pixels marked invalid by the imagery mask were excluded. Feature derivation is in [`notebooks/Ursa_Space_1D.ipynb`](notebooks/Ursa_Space_1D.ipynb), and the analysis is in [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb).

### EDA summary

We explored seven engineered features (red, green, blue, brightness, saturation, NIR and NDVI) on the nine Hurricane Helene tiles. After invalid and NoData pixels were removed, about 12.1 million pixels remained.

- RGB and brightness overlap strongly (Spearman 0.93–0.97), and so do NIR and NDVI (0.93).
- Using brightness > 0.70 and saturation < 0.20 as a rough proxy, about 7.7% of valid pixels looked cloud-like. This is a heuristic observation, not a ground-truth cloud estimate.
- NIR was higher in pixels flagged as cloud-like. Because NIR and NDVI are so correlated, adding either band should be tested by ablation (RGB vs. RGB+NIR vs. RGB+NIR+NDVI) rather than assumed to help.

---

## 🧠 **Model Development**

### Classical methods ([`notebooks/pipeline/methods/`](notebooks/pipeline/methods/))

| Method | Idea | Parameters |
|---|---|---|
| **HOT RGB** | Haze index `blue − 0.5 × red` | threshold 0.225 |
| **Dark-channel prior** | Haze and cloud raise the local minimum across channels | 31 px window, threshold 0.35 |
| HSV fixed (supporting) | Bright, low-saturation pixels are cloud | V ≥ 0.70, S ≤ 0.20 (fixed) |
| HSV Otsu (supporting) | The brightness threshold is chosen per image with Otsu's method | S ≤ 0.20 |

HOT and DCP thresholds were picked by pooled-pixel F1 on the **dev** split, then frozen before the test run. All four methods implement the shared `CloudDetector` interface in [`base.py`](notebooks/pipeline/base.py).

### Pretrained model: Cloud-Adapter

[Cloud-Adapter](https://huggingface.co/XavierJiezou/cloud-adapter-models) adapts a DINOv2 vision foundation model for cloud segmentation. We use its binary checkpoint trained on **HRC_WHU**, which is Google Earth RGB imagery at 0.5–15 m. That is the closest public training resolution we found to Maxar's ~0.3–0.5 m.

- Each 1024 × 1024 patch is split into sixteen 256 × 256 crops at native resolution, matching the checkpoint's training crop size.
- The model outputs cloud / clear directly, so there is no threshold to tune and nothing was fit to dev.
- It was applied zero-shot, with no fine-tuning.

Runner: [`evaluate_cloud_adapter_pilot.py`](notebooks/pipeline/evaluate_cloud_adapter_pilot.py). It uses the same scoring code as the classical methods.

### Candidate set aside: SegFormer (CloudSEN12Plus)

We first evaluated a [SegFormer mit_b4 RGB](https://huggingface.co/Burdenthrive/cloud-detection-segformer-mit_b4-RGB) checkpoint trained on CloudSEN12Plus. That dataset is Sentinel-2 imagery at 10 m, roughly 20–30× coarser than Maxar. We switched to Cloud-Adapter because its training resolution is much closer. SegFormer results are kept below for reference.

---

## 📈 **Results & Key Findings**

### Main comparison: held-out test split (15 patches, pooled pixels)

| Method | Precision | Recall | F1 | IoU | Dev F1 (30 patches) |
|---|---:|---:|---:|---:|---:|
| **Cloud-Adapter (HRC_WHU)** | 0.960 | 0.970 | **0.965** | **0.932** | 0.969 |
| Dark-channel prior | 0.930 | 0.927 | 0.929 | 0.867 | 0.950 |
| HOT RGB | 0.820 | 0.899 | 0.858 | 0.751 | 0.943 |

Every method was scored on the same 15 test patches with the same code. Cloud-Adapter ran on a Colab GPU at about 1.6 s per patch. Per-patch metrics, masks and comparison images are in `outputs/cloud_adapter_benchmark/`, `outputs/dark_channel_benchmark/` and `outputs/hot_reviewed_benchmark/`.

### Supporting comparisons (same test split)

| Method | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| SegFormer (CloudSEN12Plus, set aside) | 0.941 | 0.985 | 0.963 | 0.928 |
| HSV fixed | 0.862 | 0.701 | 0.773 | 0.630 |
| HSV Otsu | 0.334 | 0.781 | 0.467 | 0.305 |

### Findings so far

- **Cloud-Adapter is the strongest and most consistent method.** It scored F1 0.969 on dev and 0.965 on test without any tuning.
- **DCP is a strong classical option.** It comes within about 0.04 F1 of Cloud-Adapter on test, needs no GPU or model download, and only dropped slightly from dev to test (0.950 → 0.929).
- **HOT does not hold up on test.** Its F1 fell from 0.943 on dev to 0.858 on test. On one cloud-free test patch it marked about 27% of the pixels as cloud, which suggests bright clear surfaces trigger the haze index.
- **SegFormer matched Cloud-Adapter on test but not on dev.** On dev it over-predicted cloud badly (F1 0.880, precision 0.80). That inconsistency is another reason to prefer Cloud-Adapter.
- **Sparse cloud is the hardest case, even for Cloud-Adapter.** On dev patches with under 1% cloud, its F1 ranges from 0.005 to 0.56. On test it missed one small cloud entirely (about 5,200 pixels). On cloud-free patches it flags up to about 0.9% of pixels as cloud.
- **Otsu thresholding performs poorly on patches with little or no cloud**, because it always splits the image into two classes.

### Caveats

- Each patch's labels come from a single reviewer, so inter-reviewer agreement is unknown.
- Both splits contain only Hurricane Helene and Brazil Flooding, so cross-event generalization has not been tested.
- 15 test patches is a small sample.
- HRC_WHU is closer to Maxar's resolution than Sentinel-2, but it is still Google Earth imagery rather than the same sensor.

---

## 🚀 **Next Steps**

- Wrap Cloud-Adapter as a `CloudDetector` in the shared pipeline ([#22](https://github.com/Break-Through-Tech/Ursa-Space-Systems-1D-cloud-detection-in-very-high-resolution-optical-satellite-imagery/issues/22)).
- Build a qualitative error analysis of the sparse-cloud and cloud-free failures above.
- Strengthen the reference labels with a second reviewer and a stress set of smoke and bright surfaces ([#20](https://github.com/Break-Through-Tech/Ursa-Space-Systems-1D-cloud-detection-in-very-high-resolution-optical-satellite-imagery/issues/20)).
- Combine all methods into one benchmark table and a final notebook ([#23](https://github.com/Break-Through-Tech/Ursa-Space-Systems-1D-cloud-detection-in-very-high-resolution-optical-satellite-imagery/issues/23), [#25](https://github.com/Break-Through-Tech/Ursa-Space-Systems-1D-cloud-detection-in-very-high-resolution-optical-satellite-imagery/issues/25)).
- Possible stretch goals:
  - test generalization across events, such as Marshall Fire smoke;
  - run an NIR ablation;
  - fine-tune a small U-Net;
  - build a cloud-cover percentage / "usable imagery" filter.

---

## 📝 **License**

To be decided with the Challenge Advisor.

---

## 📄 **References**

- Maxar Open Data Program: https://www.maxar.com/open-data
- He, Sun & Tang (2009). *Single Image Haze Removal Using Dark Channel Prior.* CVPR.
- Zhang, Guindon & Cihlar (2002). *An image transform to characterize and compensate for spatial variations in thin cloud contamination of Landsat images* (HOT). Remote Sensing of Environment.
- Aybar et al. (2024). *CloudSEN12+: The largest dataset of expert-labeled pixels for cloud and cloud shadow detection in Sentinel-2.* Data in Brief.
- Zou et al. (2024). *Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images* (Cloud-Adapter).

---

## 🙏 **Acknowledgements**

Thanks to our Challenge Advisor John Savage (Ursa Space Systems), our AI Studio Coach Julio Contreras, and the Break Through Tech AI team.
