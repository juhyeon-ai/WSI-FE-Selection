"""SAM 1 on whole-slide thumbnails and explicit level-0 coordinate mapping."""
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


def build_instance_map(masks, rgb, min_area):
    instance = np.zeros(rgb.shape[:2], dtype=np.int32)
    kept = []
    for mask in masks:
        seg = np.asarray(mask["segmentation"], dtype=bool)
        if seg.sum() >= min_area and rgb[seg].mean() <= 250:
            kept.append(seg)
    # Small masks overwrite large masks in overlapping areas.
    for region_id, seg in enumerate(sorted(kept, key=lambda s: -int(s.sum())), 1):
        instance[seg] = region_id
    if not instance.any():
        raise ValueError("SAM produced no usable regions; inspect thumbnail and SAM settings")
    return instance


def generate_regions(config, rows):
    output = Path(config["work_dir"]) / "sam"
    output.mkdir(parents=True, exist_ok=True)
    pending = [row for row in rows if not (output / f"{row['slide_id']}.npz").exists()]
    if not pending:
        print("SAM: all selected slides already have region maps")
        return

    import openslide
    import torch
    from PIL import Image
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    checkpoint = Path(config["sam_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Download the SAM 1 checkpoint first: {checkpoint}")
    device = "cpu" if config["gpu"] < 0 else f"cuda:{config['gpu']}"
    sam = sam_model_registry[config["sam_model_type"]](checkpoint=str(checkpoint)).to(device)
    generator = SamAutomaticMaskGenerator(
        sam, points_per_side=config["points_per_side"],
        pred_iou_thresh=config["pred_iou_thresh"],
        stability_score_thresh=config["stability_score_thresh"],
        min_mask_region_area=config["min_mask_region_area"],
    )
    for row in pending:
        sid = row["slide_id"]
        target = output / f"{sid}.npz"
        with openslide.OpenSlide(row["wsi_path"]) as slide:
            width, height = slide.dimensions
            thumb = slide.get_thumbnail((config["thumbnail_size"], config["thumbnail_size"])).convert("RGB")
        rgb = np.asarray(thumb)
        with torch.inference_mode():
            masks = generator.generate(rgb)
        instance = build_instance_map(masks, rgb, config["min_mask_region_area"])
        thumb.save(output / f"{sid}.jpg")
        rng = np.random.default_rng(0)
        colors = rng.integers(0, 256, size=(int(instance.max()) + 1, 3), dtype=np.uint8)
        colors[0] = 255
        overlay = (0.6 * rgb + 0.4 * colors[instance]).astype(np.uint8)
        Image.fromarray(overlay).save(output / f"{sid}_overlay.jpg")
        temporary = output / f"{sid}.tmp.npz"
        np.savez_compressed(temporary, regions=instance, level0_width=width, level0_height=height)
        temporary.replace(target)
        print(f"SAM: {sid}, {len(np.unique(instance[instance > 0]))} regions")


def map_regions(coords, patch_size, instance, width, height):
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("coords must be finite (n_patches, 2) level-0 top-left coordinates")
    if not np.isfinite(patch_size) or patch_size <= 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid level-0 patch/slide dimensions")
    if np.any(coords < 0) or np.any(coords[:, 0] >= width) or np.any(coords[:, 1] >= height):
        raise ValueError("Patch origins fall outside the slide")
    h, w = instance.shape
    scale = np.array([w / width, h / height])
    starts = np.floor(coords * scale).astype(int)
    ends = np.ceil((coords + patch_size) * scale).astype(int)
    result = np.full(len(coords), -1, dtype=np.int32)
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(starts, ends)):
        values = instance[y0:min(h, y1), x0:min(w, x1)].ravel()
        values = values[values > 0]
        if values.size:
            ids, counts = np.unique(values, return_counts=True)
            result[i] = ids[np.argmax(counts)]
    return result


def merge_features(config, rows):
    work = Path(config["work_dir"])
    mapping = json.loads((work / "label_mapping.json").read_text())
    root = work / "trident" / "patches"
    for row in rows:
        sid = row["slide_id"]
        with np.load(work / "sam" / f"{sid}.npz") as pack:
            instance = pack["regions"]
            width, height = int(pack["level0_width"]), int(pack["level0_height"])
        # All quickstart FEs share a patch grid. Retain only one slide's mapping;
        # recompute if a caller supplies different coordinates or patch sizes.
        cached_coords, cached_size, cached_regions = None, None, None
        for encoder in config["encoders"]:
            output = work / "features" / encoder
            output.mkdir(parents=True, exist_ok=True)
            source = root / f"features_{encoder}" / f"{sid}.h5"
            destination = output / f"{sid}.h5"
            with h5py.File(source, "r") as stream:
                coords = stream["coords"][:]
                features = stream["features"]
                if features.ndim != 2 or not features.shape[0] or len(coords) != features.shape[0]:
                    raise ValueError(f"Empty or misaligned features: {source}")
                attrs = stream["coords"].attrs
                patch_size = attrs.get("patch_size_level0")
                if patch_size is None:
                    # TRIDENT retains these attributes in feature HDF5 files.
                    required = ["patch_size", "level0_magnification", "target_magnification"]
                    if not all(key in attrs for key in required):
                        raise ValueError(f"Missing level-0 patch size metadata: {source}")
                    patch_size = float(attrs["patch_size"]) * float(attrs["level0_magnification"]) / float(attrs["target_magnification"])
                patch_size = float(patch_size)
                if cached_size != patch_size or not np.array_equal(cached_coords, coords):
                    cached_regions = map_regions(coords, patch_size, instance, width, height)
                    cached_coords, cached_size = coords, patch_size
                regions = cached_regions
                if not np.any(regions > 0):
                    raise ValueError(f"No patches map to SAM regions: {sid}; inspect overlay")
            temporary = destination.with_suffix(".tmp.h5")
            shutil.copy2(source, temporary)
            with h5py.File(temporary, "r+") as stream:
                for key in ["sam_region", "sam_cluster", "label"]:
                    if key in stream:
                        del stream[key]
                stream.create_dataset("sam_region", data=regions, compression="gzip")
                stream["sam_cluster"] = stream["sam_region"]  # compatibility hard link
                stream.create_dataset("label", data=np.int64(mapping[row["label"]]))
                stream.attrs["slide_id"] = sid
                stream.attrs["sam_mapping"] = "positive-region majority within patch box; unmatched=-1"
                stream.attrs["sam_coverage"] = float(np.mean(regions > 0))
            temporary.replace(destination)
            print(f"Merged: {encoder}/{sid}, coverage={np.mean(regions > 0):.1%}")
