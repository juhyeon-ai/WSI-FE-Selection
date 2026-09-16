"""One command to compare candidate FEs on the same cohort."""
import argparse
import csv
import json
import platform
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np

from .effective_dimension import pca_effective_dim
from .linear_probing import linear_probing_score
from .nesum import nesum_score
from .sam_cluster import sam_ratio_score
from .sam_lp import sam_lp_score
from .self_cluster import self_cluster_score_one
from .config import ALL_METRICS, DEFAULT_CONFIG, load_config, metric_kwargs

DEFAULT_METRICS = DEFAULT_CONFIG["metrics"]


def load_slide(path, needs_sam, needs_label, max_patches, seed):
    """Validate every feature row, retaining only the deterministic scoring sample.

    Read contiguous blocks instead of loading a whole WSI or issuing thousands
    of HDF5 fancy-index reads. Keep full coordinates/IDs for cross-FE checks.
    """
    with h5py.File(path, "r") as stream:
        features = stream["features"]
        if features.ndim != 2 or min(features.shape) < 1:
            raise ValueError(f"{path}: features must be a nonempty, finite 2-D array")
        count, dimension = features.shape
        indices = (np.sort(np.random.default_rng(seed).choice(count, max_patches, replace=False))
                   if count > max_patches else None)
        z = np.empty((min(count, max_patches), dimension), dtype=np.float32)
        # Approximately 8 MiB per source block, including float64 input files.
        block_rows = max(1, (8 * 1024**2) // (dimension * max(features.dtype.itemsize, 4)))
        for start in range(0, count, block_rows):
            stop = min(start + block_rows, count)
            block = np.asarray(features[start:stop], dtype=np.float32)
            if not np.isfinite(block).all():
                raise ValueError(f"{path}: features must be a nonempty, finite 2-D array")
            if indices is None:
                z[start:stop] = block
            else:
                lo, hi = np.searchsorted(indices, [start, stop])
                z[lo:hi] = block[indices[lo:hi] - start]
        y = None
        if needs_label:
            label = np.asarray(stream["label"])
            if label.size != 1 or label.dtype.kind not in "iu":
                raise ValueError(f"{path}: label must be a scalar integer class ID")
            y = int(label.reshape(-1)[0])
        regions = None
        if needs_sam:
            key = "sam_region" if "sam_region" in stream else "sam_cluster"
            regions = np.asarray(stream[key])
            if regions.shape != (count,) or regions.dtype.kind not in "iu":
                raise ValueError(f"{path}: SAM IDs must be integer (n_patches,) aligned with features")
            if np.any(regions < -1):
                raise ValueError(f"{path}: only -1 is accepted as an unmatched-region sentinel")
            if not np.any(regions >= 0):
                raise ValueError(f"{path}: no assigned SAM patches")
        coords = stream["coords"][:] if "coords" in stream else None
    return z, y, regions, coords, count, indices


def evaluate(feature_root, output_dir, encoders=None, metrics=None, seed=None, reduce_dim=None, max_patches=None, config=None):
    from sklearn.exceptions import ConvergenceWarning

    root, output = Path(feature_root), Path(output_dir)
    overrides = {"shared": {key: value for key, value in
                 [("seed", seed), ("reduce_dim", reduce_dim), ("max_patches", max_patches)] if value is not None}}
    if metrics is not None:
        overrides["metrics"] = metrics
    settings = load_config(config, overrides=overrides)
    metrics = settings["metrics"]
    seed = settings["shared"]["seed"]
    reduce_dim = settings["shared"]["reduce_dim"]
    max_patches = settings["shared"]["max_patches"]
    kwargs = {metric: metric_kwargs(settings, metric) for metric in metrics}
    encoders = encoders or sorted(p.name for p in root.iterdir() if p.is_dir())
    if not encoders or len(set(encoders)) != len(encoders):
        raise ValueError("Provide one or more distinct encoder directories")
    if len(set(metrics)) != len(metrics) or not set(metrics).issubset(ALL_METRICS):
        raise ValueError("Choose distinct metrics from the supported list")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if reduce_dim < 2 or max_patches < 50:
        raise ValueError("reduce_dim must be >=2 and max_patches >=50")
    inventories = {fe: {p.stem for p in (root / fe).glob('*.h5')} for fe in encoders}
    slides = sorted(inventories[encoders[0]])
    if not slides or any(inventories[fe] != set(slides) for fe in encoders):
        raise ValueError("Each FE must contain exactly the same nonempty slide set; no silent intersection is used")
    needs_sam = bool(set(metrics) & {"sam_cluster", "sam_lp"})
    needs_label = bool(set(metrics) & {"linear_probing", "sam_lp", "logme"})
    expected_labels, expected_coords, expected_regions = None, {}, {}
    rows, diagnostics = [], []
    for fe in encoders:
        start = time.perf_counter()
        bags, regions, labels = [], [], []
        for slide_index, sid in enumerate(slides):
            path = root / fe / f"{sid}.h5"
            z, y, c, coords, total_patches, indices = load_slide(
                path, needs_sam, needs_label, max_patches, seed + slide_index)
            if bags and z.shape[1] != bags[0].shape[1]:
                raise ValueError(f"{fe}: feature dimension changes between slides")
            # Coordinate equality also protects patch subsampling across encoders.
            if coords is not None and (coords.shape != (total_patches, 2) or not np.isfinite(coords).all()):
                raise ValueError(f"{fe}/{sid}: coords must be finite (n_patches, 2)")
            if fe == encoders[0]:
                expected_coords[sid] = (total_patches, coords)
                expected_regions[sid] = c
            count, reference = expected_coords[sid]
            if total_patches != count or (reference is None) != (coords is None) or (reference is not None and not np.array_equal(reference, coords)):
                raise ValueError(f"{fe}/{sid}: patch count/order/coordinates differ across FEs")
            if needs_sam and not np.array_equal(expected_regions[sid], c):
                raise ValueError(f"{fe}/{sid}: SAM region assignments differ across FEs")
            if indices is not None:
                c = c[indices] if c is not None else None
            bags.append(z)
            labels.append(y)
            regions.append(c)
            if c is not None:
                assigned = c >= 0
                sizes = np.unique(c[assigned], return_counts=True)[1]
                diagnostics.append({"encoder": fe, "slide_id": sid, "patches_scored": len(z), "sam_coverage": float(assigned.mean()), "sam_regions": len(sizes)})
                min_ratio = settings["sam_cluster"]["min_cluster_ratio"]
                min_size = max(2, int(assigned.sum() * min_ratio)) if min_ratio > 0 else 1
                if "sam_cluster" in metrics and (assigned.sum() < 50 or np.sum(sizes >= min_size) < 2):
                    raise ValueError(f"{fe}/{sid}: SAM-Cluster needs >=50 assigned patches and >=2 non-tiny regions; inspect masks or choose metrics without sam_cluster")
        if expected_labels is None:
            expected_labels = labels
        elif labels != expected_labels:
            raise ValueError(f"{fe}: labels differ from other FEs")
        if needs_label:
            if len(set(labels)) < 2:
                raise ValueError("Supervised scores require at least two sampled classes")
            # LogME requires contiguous class IDs.
            _, y = np.unique(labels, return_inverse=True)
        else:
            y = np.zeros(len(slides), dtype=int)
        sam_bags, sam_ids = [], []
        if needs_sam:
            for z, c in zip(bags, regions):
                assigned = c >= 0
                # The common full-coverage case can share the original arrays.
                sam_bags.append(z if assigned.all() else z[assigned])
                sam_ids.append(c if assigned.all() else c[assigned])
        scores = {"encoder": fe}
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            for metric in metrics:
                if metric == "nesum":
                    value = np.mean([nesum_score(z, **kwargs[metric]) for z in bags])
                elif metric == "self_cluster":
                    options = dict(kwargs[metric])
                    aggregate = options.pop("aggregate")
                    values = [self_cluster_score_one(z, **options) for z in bags]
                    value = np.median(values) if aggregate == "median" else np.mean(values)
                elif metric == "sam_cluster":
                    value = np.mean([sam_ratio_score(z, c, **kwargs[metric]) for z, c in zip(sam_bags, sam_ids)])
                elif metric == "linear_probing":
                    value = linear_probing_score(bags, y, **kwargs[metric])
                elif metric == "sam_lp":
                    value = sam_lp_score(sam_bags, sam_ids, y, **kwargs[metric])
                elif metric == "effective_dimension":
                    value = np.mean([pca_effective_dim(z, **kwargs[metric]) for z in bags])
                elif metric == "logme":
                    from .logme import logme_score
                    value = logme_score(bags, y, **kwargs[metric])
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                if not np.isfinite(value):
                    raise ValueError(f"{fe}/{metric} returned a non-finite score")
                scores[metric] = float(value)
        scores["seconds"] = time.perf_counter() - start
        rows.append(scores)
        print(json.dumps(scores), flush=True)
        # Do not retain this encoder's feature arrays while loading the next FE.
        del bags, regions, sam_bags, sam_ids, z, c
    # Average ranks avoid combining incompatible raw score scales. Ties share rank.
    from scipy.stats import rankdata
    rank_rows = [{"encoder": fe} for fe in encoders]
    for metric in metrics:
        ranks = rankdata([-row[metric] for row in rows], method="average")
        for row, rank in zip(rank_rows, ranks):
            row[metric + "_rank"] = float(rank)
    for row in rank_rows:
        row["mean_rank"] = float(np.mean([row[metric + "_rank"] for metric in metrics]))
    rank_rows.sort(key=lambda r: (r["mean_rank"], r["encoder"]))
    output.mkdir(parents=True, exist_ok=True)
    for name, records in [("scores.csv", rows), ("ranking.csv", rank_rows), ("coverage.csv", diagnostics)]:
        if records:
            with (output / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        elif name == "coverage.csv":
            (output / name).unlink(missing_ok=True)
    metadata = {"encoders": encoders, "slides": slides, "metrics": metrics, "seed": seed,
                "reduce_dim": reduce_dim, "max_patches": max_patches,
                "scoring_config": settings,
                "sam_cluster": {**settings["sam_cluster"], "threshold": settings["sam_cluster"]["ratio_sim_threshold"],
                                "topk_exclude": settings["sam_cluster"]["ratio_topk_exclude"],
                                "rng_seeds": list(metric_kwargs(settings, "sam_cluster")["rng_seeds"])},
                "sam_lp": settings["sam_lp"],
                "python": platform.python_version(),
                "packages": {name: version(name) for name in ["numpy", "h5py", "scikit-learn", "scipy"]},
                "interpretation": "Higher raw scores; lower mean rank. Mean rank is a heuristic, not a validated paper metric. LP/SAM-LP are in-sample log-likelihoods, not test performance."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    print(f"Saved scores and ranking to {output}")
    return rows, rank_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True, help="Contains one directory per FE, each with slide_id.h5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoders", nargs="+")
    parser.add_argument("--config", type=Path, help="Scoring JSON; defaults to configs/suitability.json")
    parser.add_argument("--metrics", nargs="+", choices=ALL_METRICS, help="Override config metrics")
    parser.add_argument("--seed", type=int, help="Override config shared.seed")
    parser.add_argument("--reduce-dim", type=int, help="Override config shared.reduce_dim")
    parser.add_argument("--max-patches", type=int, help="Override config shared.max_patches")
    args = parser.parse_args()
    try:
        evaluate(**vars(args))
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
