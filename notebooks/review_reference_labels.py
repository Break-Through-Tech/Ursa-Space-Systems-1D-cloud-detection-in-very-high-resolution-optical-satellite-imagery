"""Create candidate cloud labels and review them interactively in Colab.

Candidate masks are only starting points. A patch is part of the reference set
only after a human explicitly marks it reviewed in the interactive reviewer.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

LABEL_CLEAR = 0
LABEL_CLOUD = 1
LABEL_IGNORE = 255

DEFAULT_BRIGHTNESS_THRESHOLD = 0.70
DEFAULT_SATURATION_THRESHOLD = 0.20


def pseudo_cloud_mask(
    rgb: np.ndarray,
    valid: np.ndarray,
    brightness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD,
    saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
) -> np.ndarray:
    """Return uint8 candidate labels: clear=0, cloud=1, invalid=255."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must contain 8-bit channel values")
    if valid.shape != rgb.shape[:2]:
        raise ValueError("valid must match the image height and width")
    if valid.dtype != np.bool_:
        raise ValueError("valid must be a boolean mask")
    if not 0 <= brightness_threshold <= 1:
        raise ValueError("brightness_threshold must be between 0 and 1")
    if not 0 <= saturation_threshold <= 1:
        raise ValueError("saturation_threshold must be between 0 and 1")

    scaled = rgb.astype(np.float32) / 255.0
    maximum = np.max(scaled, axis=-1)
    minimum = np.min(scaled, axis=-1)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    labels = np.zeros(valid.shape, dtype=np.uint8)
    labels[(maximum > brightness_threshold) & (saturation < saturation_threshold) & valid] = (
        LABEL_CLOUD
    )
    labels[~valid] = LABEL_IGNORE
    return labels


def _read_rgb_patch(
    src: Any,
    row: int,
    column: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    from rasterio.windows import Window

    window = Window(column, row, patch_size, patch_size)
    bands = src.read(
        [1, 2, 3],
        window=window,
        boundless=True,
        fill_value=0,
    )
    band_masks = src.read_masks(
        [1, 2, 3],
        window=window,
        boundless=True,
    )
    rgb = np.moveaxis(bands, 0, -1)
    if rgb.dtype != np.uint8:
        raise ValueError(
            f"Expected 8-bit RGB visual imagery, found {rgb.dtype} in {src.name}"
        )
    valid = np.all(band_masks > 0, axis=0)
    return rgb, valid


def _patch_locations(height: int, width: int, patch_size: int) -> list[tuple[int, int]]:
    return [
        (row, column)
        for row in range(0, height, patch_size)
        for column in range(0, width, patch_size)
    ]


def _event_name(source_root: Path, tile_path: Path) -> str:
    relative_parts = tile_path.relative_to(source_root).parts
    if len(relative_parts) == 3:
        return source_root.name
    return relative_parts[0] if len(relative_parts) > 1 else source_root.name


def _select_stratified(
    candidates: list[dict[str, Any]],
    count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    ordered = sorted(shuffled, key=lambda item: item["pseudo_cloud_fraction"])
    indices = np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)
    return [ordered[index] for index in indices]


def prepare_review_set(
    input_dir: str | Path,
    output_dir: str | Path,
    patch_size: int = 1024,
    patches_per_tile: int = 3,
    minimum_valid_fraction: float = 0.80,
    cloud_free_events: Sequence[str] = (),
    seed: int = 42,
    brightness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD,
    saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
) -> Path:
    """Prepare a small, repeatable review sample from every discovered visual tile."""
    import rasterio

    source_root = Path(input_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if patches_per_tile <= 0:
        raise ValueError("patches_per_tile must be positive")
    if not 0 < minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be in (0, 1]")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {source_root}")
    if (destination / "manifest.json").exists():
        raise FileExistsError(
            f"{destination / 'manifest.json'} already exists; choose a new output "
            "directory so existing human review work is not overwritten."
        )

    visual_tiles = sorted(source_root.rglob("*-visual.tif"))
    if not visual_tiles:
        raise FileNotFoundError(f"No *-visual.tif files found under {source_root}")

    available_events = {_event_name(source_root, path) for path in visual_tiles}
    clear_event_names = set(cloud_free_events)
    unknown_clear_events = clear_event_names - available_events
    if unknown_clear_events:
        raise ValueError(
            "cloud_free_events did not match discovered event directories: "
            + ", ".join(sorted(unknown_clear_events))
        )
    rng = np.random.default_rng(seed)
    selected_patches: list[dict[str, Any]] = []
    skipped_tiles: list[str] = []
    for tile_path in visual_tiles:
        with rasterio.open(tile_path) as src:
            if src.count < 3:
                raise ValueError(f"Expected at least three RGB bands in {tile_path}")
            candidates: list[dict[str, Any]] = []
            for row, column in _patch_locations(src.height, src.width, patch_size):
                rgb, valid = _read_rgb_patch(src, row, column, patch_size)
                valid_count = int(valid.sum())
                if valid_count == 0:
                    fraction = 0.0
                else:
                    labels = pseudo_cloud_mask(
                        rgb,
                        valid,
                        brightness_threshold,
                        saturation_threshold,
                    )
                    fraction = float(np.count_nonzero(labels == LABEL_CLOUD) / valid_count)
                candidates.append(
                    {
                        "row": row,
                        "column": column,
                        "pseudo_cloud_fraction": fraction,
                        "valid_fraction": valid_count / (patch_size * patch_size),
                    }
                )

            nonempty = [item for item in candidates if item["valid_fraction"] > 0]
            if not nonempty:
                skipped_tiles.append(tile_path.relative_to(source_root).as_posix())
                continue
            preferred = [
                item
                for item in nonempty
                if item["valid_fraction"] >= minimum_valid_fraction
            ]
            selection_pool = preferred or nonempty
            chosen = _select_stratified(selection_pool, patches_per_tile, rng)
            relative_path = tile_path.relative_to(source_root)
            event_id = _event_name(source_root, tile_path)
            tile_id = tile_path.parent.name
            scene_is_cloud_free = event_id in clear_event_names
            for item in chosen:
                patch_id = (
                    f"{event_id}_{tile_id}_r{item['row']:05d}_c{item['column']:05d}"
                )
                selected_patches.append(
                    {
                        **item,
                        "id": patch_id,
                        "event_id": event_id,
                        "tile_id": tile_id,
                        "source": relative_path.as_posix(),
                        "scene_condition": (
                            "reported_cloud_free"
                            if scene_is_cloud_free
                            else "cloudy_or_unverified"
                        ),
                        "used_partial_valid_fallback": not bool(preferred),
                    }
                )

    if not selected_patches:
        raise ValueError("No valid image patches were found in the input tiles")

    patches_dir = destination / "patches"
    patches_dir.mkdir(parents=True, exist_ok=False)
    for patch in selected_patches:
        tile_path = source_root / patch["source"]
        with rasterio.open(tile_path) as src:
            rgb, valid = _read_rgb_patch(
                src,
                patch["row"],
                patch["column"],
                patch_size,
            )
        candidate = pseudo_cloud_mask(
            rgb,
            valid,
            brightness_threshold,
            saturation_threshold,
        )
        patch_id = patch["id"]
        Image.fromarray(rgb).save(patches_dir / f"{patch_id}_rgb.jpg", quality=95)
        Image.fromarray(candidate).save(patches_dir / f"{patch_id}_candidate.png")

    manifest = {
        "input_directory": source_root.name,
        "patch_size": patch_size,
        "minimum_valid_fraction": minimum_valid_fraction,
        "cloud_free_events": sorted(clear_event_names),
        "seed": seed,
        "pseudo_label_method": "HSV value/saturation threshold on 8-bit RGB",
        "brightness_threshold": brightness_threshold,
        "saturation_threshold": saturation_threshold,
        "label_values": {
            "clear_or_non_cloud": LABEL_CLEAR,
            "cloud": LABEL_CLOUD,
            "ignore": LABEL_IGNORE,
        },
        "review_required_before_evaluation": True,
        "tiles_skipped_no_valid_patches": skipped_tiles,
        "patches": selected_patches,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def _reviewer_directory(output_dir: str | Path, reviewer_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", reviewer_id):
        raise ValueError("reviewer_id may contain only letters, numbers, '_' and '-'")
    return Path(output_dir).expanduser().resolve() / "annotations" / reviewer_id


def create_candidate_overlay_previews(
    patches_dir: str | Path,
    output_dir: str | Path | None = None,
    cloud_opacity: float = 0.55,
) -> list[Path]:
    """Save human-viewable RGB previews with candidate cloud pixels in blue."""
    patch_root = Path(patches_dir).expanduser().resolve()
    if not patch_root.is_dir():
        raise FileNotFoundError(f"Patch directory does not exist: {patch_root}")
    if not 0 <= cloud_opacity <= 1:
        raise ValueError("cloud_opacity must be between 0 and 1")

    candidates = sorted(patch_root.glob("*_candidate.png"))
    if not candidates:
        raise FileNotFoundError(f"No *_candidate.png files found under {patch_root}")

    preview_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else patch_root.parent / "candidate-overlays"
    )
    preview_root.mkdir(parents=True, exist_ok=True)
    previews: list[Path] = []
    for candidate_path in candidates:
        patch_id = candidate_path.name.removesuffix("_candidate.png")
        rgb_path = patch_root / f"{patch_id}_rgb.jpg"
        if not rgb_path.is_file():
            raise FileNotFoundError(
                f"RGB patch for {candidate_path.name} not found: {rgb_path}"
            )

        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(candidate_path) as image:
            labels = np.asarray(image, dtype=np.uint8)
        if labels.shape != rgb.shape[:2]:
            raise ValueError(
                f"Mask dimensions for {candidate_path.name} do not match {rgb_path.name}"
            )
        if not np.isin(labels, (LABEL_CLEAR, LABEL_CLOUD, LABEL_IGNORE)).all():
            raise ValueError(f"Unexpected label values in {candidate_path}")

        overlay = rgb.astype(np.float32)
        cloud = labels == LABEL_CLOUD
        overlay[cloud] = (
            (1 - cloud_opacity) * overlay[cloud]
            + cloud_opacity * np.array((13, 115, 255), dtype=np.float32)
        )
        ignored = labels == LABEL_IGNORE
        overlay[ignored] = (
            (1 - cloud_opacity) * overlay[ignored]
            + cloud_opacity * np.array((255, 179, 0), dtype=np.float32)
        )

        preview_path = preview_root / f"{patch_id}_candidate_overlay.jpg"
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
            preview_path,
            quality=95,
        )
        previews.append(preview_path)

    return previews


def load_reviewed_labels(
    output_dir: str | Path,
    patch_id: str,
    reviewer_id: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load one reviewer's mask and tags; reject pending pseudo-labels."""
    destination = Path(output_dir).expanduser().resolve()
    patch_dir = _reviewer_directory(destination, reviewer_id)
    stem = patch_dir / patch_id
    review_path = stem.with_name(stem.name + "_review.json")
    if not review_path.exists():
        raise ValueError(f"Patch {patch_id} has not been marked reviewed")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("status") != "reviewed":
        raise ValueError(f"Patch {patch_id} is still pending human review")
    if review.get("reviewer_id") != reviewer_id:
        raise ValueError(f"Review metadata for {patch_id} does not match {reviewer_id}")
    labels = np.asarray(
        Image.open(stem.with_name(stem.name + "_labels.png")),
        dtype=np.uint8,
    )
    with np.load(stem.with_name(stem.name + "_tags.npz"), allow_pickle=False) as saved:
        tags = {
            name: np.asarray(saved[name], dtype=np.uint8).copy()
            for name in ("smoke", "bright_ground", "thin_cloud")
        }
    if any(tag.shape != labels.shape for tag in tags.values()):
        raise ValueError(f"Tag masks for {patch_id} do not match the label dimensions")
    if not np.isin(labels, (LABEL_CLEAR, LABEL_CLOUD, LABEL_IGNORE)).all():
        raise ValueError(f"Patch {patch_id} contains unexpected label values")
    return labels, tags


def launch_reviewer(output_dir: str | Path, reviewer_id: str) -> None:
    """Open the reviewer, saving edits separately under this reviewer's ID."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display
    from matplotlib.path import Path as PolygonPath
    from matplotlib.widgets import PolygonSelector

    destination = Path(output_dir).expanduser().resolve()
    annotation_dir = _reviewer_directory(destination, reviewer_id)
    manifest_path = destination / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Review manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patches = manifest["patches"]
    if not patches:
        raise ValueError("The review manifest contains no patches")
    patches_dir = destination / "patches"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    patch_by_id = {patch["id"]: patch for patch in patches}

    def annotation_paths(patch_id: str) -> dict[str, Path]:
        stem = annotation_dir / patch_id
        return {
            "labels": stem.with_name(stem.name + "_labels.png"),
            "tags": stem.with_name(stem.name + "_tags.npz"),
            "review": stem.with_name(stem.name + "_review.json"),
        }

    def review_status(patch_id: str) -> str:
        review_path = annotation_paths(patch_id)["review"]
        if not review_path.exists():
            return "pending"
        return json.loads(review_path.read_text(encoding="utf-8"))["status"]

    patch_options = [
        (
            f"{patch['id']} | candidate cloud {patch['pseudo_cloud_fraction']:.1%} "
            f"| {review_status(patch['id'])}",
            patch["id"],
        )
        for patch in patches
    ]
    patch_select = widgets.Dropdown(options=patch_options, description="Patch")
    tool_select = widgets.Dropdown(
        options=[
            ("Cloud = 1", "cloud"),
            ("Clear / non-cloud = 0", "clear"),
            ("Ignore = 255", "ignore"),
            ("Smoke tag (non-cloud)", "smoke"),
            ("Bright-ground tag (non-cloud)", "bright_ground"),
            ("Thin-cloud tag (cloud)", "thin_cloud"),
        ],
        description="Polygon",
    )
    reviewer_input = widgets.Text(description="Reviewer")
    apply_button = widgets.Button(description="Apply polygon", button_style="primary")
    undo_button = widgets.Button(description="Undo")
    save_button = widgets.Button(description="Save draft")
    reviewed_button = widgets.Button(description="Mark reviewed", button_style="success")
    status = widgets.HTML()

    figure, axes = plt.subplots(1, 3, figsize=(16, 6))
    figure.tight_layout()
    state: dict[str, Any] = {
        "patch_id": "",
        "rgb": None,
        "candidate": None,
        "labels": None,
        "tags": {},
        "undo": None,
        "dirty": False,
        "vertices": None,
    }
    image_artists = []
    overlay_artists = []
    titles = ("RGB imagery", "Threshold candidate", "Human-reviewed labels")
    for axis, title in zip(axes, titles):
        image_artists.append(axis.imshow(np.zeros((1, 1, 3), dtype=np.uint8)))
        overlay_artists.append(
            axis.imshow(np.zeros((1, 1, 4), dtype=np.float32))
        )
        axis.set_title(title)
        axis.axis("off")

    def store_polygon(vertices: list[tuple[float, float]]) -> None:
        state["vertices"] = vertices

    selector = PolygonSelector(axes[0], store_polygon, useblit=False)

    def patch_paths(patch_id: str) -> dict[str, Path]:
        stem = patches_dir / patch_id
        return {
            "rgb": stem.with_name(stem.name + "_rgb.jpg"),
            "candidate": stem.with_name(stem.name + "_candidate.png"),
        }

    def load_patch(patch_id: str) -> None:
        paths = patch_paths(patch_id)
        saved = annotation_paths(patch_id)
        state["patch_id"] = patch_id
        state["rgb"] = np.asarray(Image.open(paths["rgb"]).convert("RGB"))
        state["candidate"] = np.asarray(Image.open(paths["candidate"]), dtype=np.uint8)
        if saved["labels"].exists() and saved["tags"].exists():
            state["labels"] = np.asarray(Image.open(saved["labels"]), dtype=np.uint8)
            with np.load(saved["tags"], allow_pickle=False) as saved_tags:
                state["tags"] = {
                    name: np.asarray(saved_tags[name], dtype=np.uint8).copy()
                    for name in ("smoke", "bright_ground", "thin_cloud")
                }
        else:
            if saved["labels"].exists() or saved["tags"].exists() or saved["review"].exists():
                raise ValueError(
                    f"Incomplete saved annotation for {patch_id} under {annotation_dir}"
                )
            state["labels"] = state["candidate"].copy()
            if patch_by_id[patch_id]["scene_condition"] == "reported_cloud_free":
                valid = state["candidate"] != LABEL_IGNORE
                state["labels"][valid] = LABEL_CLEAR
            state["tags"] = {
                name: np.zeros(state["candidate"].shape, dtype=np.uint8)
                for name in ("smoke", "bright_ground", "thin_cloud")
            }
        state["undo"] = None
        state["dirty"] = False
        render()

    def overlay(labels: np.ndarray, candidate: bool = False) -> np.ndarray:
        rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
        if candidate:
            rgba[labels == LABEL_CLOUD] = (0.1, 0.45, 1.0, 0.42)
            rgba[labels == LABEL_IGNORE] = (1.0, 0.75, 0.0, 0.4)
        else:
            rgba[labels == LABEL_CLOUD] = (0.1, 0.8, 0.2, 0.45)
            rgba[labels == LABEL_IGNORE] = (1.0, 0.7, 0.0, 0.45)
        return rgba

    def render() -> None:
        rgb = state["rgb"]
        candidate = state["candidate"]
        labels = state["labels"]
        if rgb is None or candidate is None or labels is None:
            return
        for artist in image_artists:
            artist.set_data(rgb)
        overlay_artists[0].set_data(np.zeros((*rgb.shape[:2], 4), dtype=np.float32))
        overlay_artists[1].set_data(overlay(candidate, candidate=True))
        overlay_artists[2].set_data(overlay(labels))
        for axis in axes:
            axis.set_xlim(-0.5, rgb.shape[1] - 0.5)
            axis.set_ylim(rgb.shape[0] - 0.5, -0.5)
        selector.set_active(True)
        figure.canvas.draw_idle()
        paths = annotation_paths(state["patch_id"])
        review_status = "pending"
        if paths["review"].exists():
            review_status = json.loads(paths["review"].read_text(encoding="utf-8"))[
                "status"
            ]
        if state["dirty"]:
            review_status = "pending (unsaved edits)"
        cloud_count = int(np.count_nonzero(labels == LABEL_CLOUD))
        ignore_count = int(np.count_nonzero(labels == LABEL_IGNORE))
        ignored_fraction = ignore_count / labels.size
        ignored_warning = " | FLAG: over 20% ignored" if ignored_fraction > 0.20 else ""
        patch = patch_by_id[state["patch_id"]]
        fallback_warning = (
            " | partial-valid fallback"
            if patch["used_partial_valid_fallback"]
            else ""
        )
        status.value = (
            f"<b>{state['patch_id']}</b> — status: <b>{review_status}</b> "
            f"| valid: {patch['valid_fraction']:.1%} "
            f"| cloud pixels: {cloud_count:,} "
            f"| ignored: {ignore_count:,} ({ignored_fraction:.1%})"
            f"{ignored_warning}{fallback_warning} "
            "| draw a polygon on the left, choose a class, then apply."
        )

    def save_patch(review_status: str) -> None:
        patch_id = state["patch_id"]
        paths = annotation_paths(patch_id)
        Image.fromarray(state["labels"]).save(paths["labels"])
        np.savez_compressed(paths["tags"], **state["tags"])
        patch = patch_by_id[patch_id]
        review_data = {
            "patch_id": patch_id,
            "status": review_status,
            "reviewer_id": reviewer_id,
            "reviewer": reviewer_input.value.strip(),
            "reviewed_at_utc": (
                datetime.now(timezone.utc).isoformat()
                if review_status == "reviewed"
                else None
            ),
            "pseudo_cloud_fraction": patch["pseudo_cloud_fraction"],
            "event_id": patch["event_id"],
            "scene_condition": patch["scene_condition"],
            "valid_fraction": patch["valid_fraction"],
            "ignored_fraction": float(
                np.count_nonzero(state["labels"] == LABEL_IGNORE) / state["labels"].size
            ),
            "final_cloud_fraction": float(
                np.count_nonzero(state["labels"] == LABEL_CLOUD)
                / max(1, np.count_nonzero(state["candidate"] != LABEL_IGNORE))
            ),
            "tagged_pixel_counts": {
                name: int(np.count_nonzero(mask))
                for name, mask in state["tags"].items()
            },
        }
        paths["review"].write_text(
            json.dumps(review_data, indent=2) + "\n",
            encoding="utf-8",
        )
        state["dirty"] = False
        patch_select.options = [
            (
                f"{item['id']} | candidate cloud "
                f"{item['pseudo_cloud_fraction']:.1%} | {review_status(item['id'])}",
                item["id"],
            )
            for item in patches
        ]
        render()

    def on_apply(_button: widgets.Button) -> None:
        vertices = state["vertices"]
        if vertices is None or len(vertices) < 3:
            status.value = "Draw and close a polygon first."
            return
        path = PolygonPath(vertices)
        x0 = max(0, int(np.floor(np.min(np.asarray(vertices)[:, 0]))))
        x1 = min(state["labels"].shape[1], int(np.ceil(np.max(np.asarray(vertices)[:, 0]))) + 1)
        y0 = max(0, int(np.floor(np.min(np.asarray(vertices)[:, 1]))))
        y1 = min(state["labels"].shape[0], int(np.ceil(np.max(np.asarray(vertices)[:, 1]))) + 1)
        if x0 >= x1 or y0 >= y1:
            status.value = "The polygon must overlap the image."
            return
        xx, yy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        points = np.column_stack((xx.ravel(), yy.ravel()))
        inside = path.contains_points(points).reshape(yy.shape)
        inside &= state["candidate"][y0:y1, x0:x1] != LABEL_IGNORE
        if not np.any(inside):
            status.value = "The polygon contains no valid image pixels."
            return

        state["undo"] = (
            state["labels"].copy(),
            {name: mask.copy() for name, mask in state["tags"].items()},
        )
        label_region = state["labels"][y0:y1, x0:x1]
        tag_regions = {
            name: mask[y0:y1, x0:x1] for name, mask in state["tags"].items()
        }
        action = tool_select.value
        if action in ("cloud", "clear", "ignore"):
            label_region[inside] = {
                "cloud": LABEL_CLOUD,
                "clear": LABEL_CLEAR,
                "ignore": LABEL_IGNORE,
            }[action]
            for tag_region in tag_regions.values():
                tag_region[inside] = 0
        else:
            label_region[inside] = LABEL_CLOUD if action == "thin_cloud" else LABEL_CLEAR
            for name, tag_region in tag_regions.items():
                tag_region[inside] = int(name == action)
        state["vertices"] = None
        state["dirty"] = True
        render()

    def on_undo(_button: widgets.Button) -> None:
        if state["undo"] is None:
            status.value = "No polygon action to undo."
            return
        state["labels"], state["tags"] = state["undo"]
        state["undo"] = None
        state["dirty"] = True
        render()

    def on_save_draft(_button: widgets.Button) -> None:
        save_patch("pending")

    def on_mark_reviewed(_button: widgets.Button) -> None:
        if not reviewer_input.value.strip():
            status.value = "Enter a reviewer name before marking this patch reviewed."
            return
        save_patch("reviewed")

    def on_patch_change(change: dict[str, Any]) -> None:
        if change.get("name") != "value" or not change.get("new"):
            return
        if state["dirty"]:
            save_patch("pending")
        load_patch(change["new"])

    apply_button.on_click(on_apply)
    undo_button.on_click(on_undo)
    save_button.on_click(on_save_draft)
    reviewed_button.on_click(on_mark_reviewed)
    patch_select.observe(on_patch_change)
    controls = widgets.HBox(
        [
            patch_select,
            tool_select,
            apply_button,
            undo_button,
            save_button,
            reviewer_input,
            reviewed_button,
        ],
        layout=widgets.Layout(flex_flow="row wrap"),
    )
    display(widgets.VBox([controls, status]))
    display(figure.canvas)
    load_patch(patches[0]["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--patches-per-tile", type=int, default=3)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.80)
    parser.add_argument(
        "--cloud-free-event",
        action="append",
        default=[],
        help="Event directory name known to be cloud-free; may be repeated.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--brightness-threshold",
        type=float,
        default=DEFAULT_BRIGHTNESS_THRESHOLD,
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=DEFAULT_SATURATION_THRESHOLD,
    )
    args = parser.parse_args()
    output = prepare_review_set(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        patches_per_tile=args.patches_per_tile,
        minimum_valid_fraction=args.minimum_valid_fraction,
        cloud_free_events=args.cloud_free_event,
        seed=args.seed,
        brightness_threshold=args.brightness_threshold,
        saturation_threshold=args.saturation_threshold,
    )
    print(f"Prepared human-review patches under: {output}")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["tiles_skipped_no_valid_patches"]:
        print("Skipped tiles with no valid image pixels:", manifest["tiles_skipped_no_valid_patches"])
    print(
        "Open a notebook with the ipympl backend and call "
        "launch_reviewer(review_dir, reviewer_id='your_id')."
    )


if __name__ == "__main__":
    main()
