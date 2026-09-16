"""Prepare a small cohort and compare WSI feature extractors without MIL training."""
import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

from .manifest import prepare_sample, read_manifest
from suitability_score.config import DEFAULT_CONFIG_PATH, load_config as load_score_config

DEFAULTS = {
    "n_slides": 30, "sampling": "balanced", "gpu": 0,
    "magnification": 20, "patch_size": 256, "batch_size": 64, "segmenter": "hest",
    "thumbnail_size": 2048, "sam_model_type": "vit_h", "points_per_side": 32,
    "pred_iou_thresh": 0.2, "stability_score_thresh": 0.3, "min_mask_region_area": 256,
}
STAGES = ["sample", "extract", "sam", "merge", "score"]


def load_config(path):
    path = Path(path).resolve()
    supplied = json.loads(path.read_text())
    allowed = set(DEFAULTS) | {"manifest", "work_dir", "trident_dir", "sam_checkpoint", "encoders", "score_config", "seed"}
    if not isinstance(supplied, dict):
        raise ValueError("Workflow config must be a JSON object")
    if set(supplied) - allowed:
        raise ValueError(f"Unknown workflow settings: {sorted(set(supplied) - allowed)}")
    config = {**DEFAULTS, **supplied}
    for key in ["manifest", "work_dir", "trident_dir", "sam_checkpoint"]:
        value = Path(config[key]).expanduser()
        config[key] = str((path.parent / value).resolve())
    score_path = Path(config.get("score_config", DEFAULT_CONFIG_PATH)).expanduser()
    config["score_config"] = str((path.parent / score_path).resolve())
    # Legacy workflow seed overrides both sampling and metric randomness.
    overrides = {"shared": {"seed": config["seed"]}} if "seed" in config else None
    config["scoring"] = load_score_config(config["score_config"], overrides=overrides)
    config["seed"] = config["scoring"]["shared"]["seed"]
    if not isinstance(config.get("encoders"), list) or not config["encoders"]:
        raise ValueError("encoders must be a nonempty list of distinct TRIDENT encoder names")
    for encoder in config["encoders"]:
        if not isinstance(encoder, str) or not encoder or Path(encoder).name != encoder or encoder in {".", ".."}:
            raise ValueError("Invalid encoder name")
    if len(set(config["encoders"])) != len(config["encoders"]):
        raise ValueError("encoders must be distinct")
    for key in ["n_slides", "patch_size", "batch_size", "thumbnail_size", "points_per_side"]:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["magnification"]) not in (int, float) or not math.isfinite(config["magnification"]) or config["magnification"] <= 0:
        raise ValueError("magnification must be positive")
    if config["sampling"] not in {"balanced", "proportional"}:
        raise ValueError("sampling must be balanced or proportional")
    if config["sam_model_type"] not in {"vit_h", "vit_l", "vit_b"}:
        raise ValueError("sam_model_type must be vit_h, vit_l or vit_b")
    if type(config["gpu"]) is not int or config["gpu"] < -1:
        raise ValueError("gpu must be -1 (CPU) or a nonnegative GPU index")
    for key in ["pred_iou_thresh", "stability_score_thresh"]:
        if type(config[key]) not in (int, float) or not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if type(config["min_mask_region_area"]) is not int or config["min_mask_region_area"] < 0:
        raise ValueError("min_mask_region_area must be a nonnegative integer")
    return config


def extraction_commands(config):
    work = Path(config["work_dir"])
    script = Path(config["trident_dir"]) / "run_batch_of_slides.py"
    common = [sys.executable, str(script), "--wsi_dir", str(work / "wsis"),
              "--job_dir", str(work / "trident"), "--gpus", str(config["gpu"]),
              "--coords_dir", "patches", "--mag", str(config["magnification"]),
              "--patch_size", str(config["patch_size"]), "--reader_type", "openslide"]
    yield common + ["--task", "seg", "--segmenter", config["segmenter"]]
    yield common + ["--task", "coords"]
    for encoder in config["encoders"]:
        yield common + ["--task", "feat", "--patch_encoder", encoder,
                        "--batch_size", str(config["batch_size"])]


def run(config, stages, dry_run=False):
    if dry_run:
        print(json.dumps(config, indent=2))
        print("Stages:", ", ".join(stages))
        if "extract" in stages:
            for command in extraction_commands(config):
                print(shlex.join(command))
        return
    if "extract" in stages and not (Path(config["trident_dir"]) / "run_batch_of_slides.py").is_file():
        raise FileNotFoundError("TRIDENT is missing; run bash scripts/setup_dependencies.sh first")
    if "sam" in stages and not Path(config["sam_checkpoint"]).is_file():
        raise FileNotFoundError(f"Download the SAM 1 checkpoint before this run: {config['sam_checkpoint']}")
    work = Path(config["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    snapshot = work / "config.resolved.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != config:
        raise ValueError("Config differs from the existing run; choose a fresh work_dir to avoid stale artifacts")
    snapshot.write_text(json.dumps(config, indent=2))
    for stage in stages:
        print(f"\n=== {stage} ===", flush=True)
        if stage == "sample":
            prepare_sample(config)
            continue
        rows = read_manifest(work / "sample.csv", check_paths=stage in {"extract", "sam"})
        if stage == "extract":
            for command in extraction_commands(config):
                print(shlex.join(command), flush=True)
                subprocess.run(command, check=True)
            for encoder in config["encoders"]:
                for row in rows:
                    expected = work / "trident" / "patches" / f"features_{encoder}" / f"{row['slide_id']}.h5"
                    if not expected.is_file():
                        raise FileNotFoundError(f"TRIDENT did not produce {expected}; inspect its run report")
        elif stage == "sam":
            from .regions import generate_regions
            generate_regions(config, rows)
        elif stage == "merge":
            from .regions import merge_features
            merge_features(config, rows)
        elif stage == "score":
            from suitability_score.__main__ import evaluate
            evaluate(work / "features", work / "scores", config["encoders"], config=config["scoring"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--dry-run", action="store_true", help="Show resolved paths and TRIDENT commands without loading WSIs/models")
    args = parser.parse_args()
    try:
        if args.stages != [stage for stage in STAGES if stage in args.stages]:
            raise ValueError("Stages must be unique and in sample, extract, sam, merge, score order")
        run(load_config(args.config), args.stages, args.dry_run)
    except ModuleNotFoundError as error:
        parser.exit(2, f"Missing dependency: {error}. Activate your environment and run bash scripts/setup_dependencies.sh.\n")
    except (ValueError, FileNotFoundError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
