"""Execution optimizations must preserve sampling, validation and region mapping."""
import builtins
import json

import h5py
import numpy as np
import pytest

from preprocessing import regions as region_ops
from suitability_score.__main__ import load_slide
from suitability_score import sam_lp


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("cap", [60, 6000])
def test_block_loading_matches_full_read_sampling(tmp_path, dtype, cap):
    # Cross the loader's block boundary with both input precisions.
    rng = np.random.default_rng(20)
    features = rng.normal(size=(4200, 512)).astype(dtype)
    regions = np.arange(len(features)) % 7
    coords = np.column_stack([np.arange(len(features)), np.zeros(len(features))])
    path = tmp_path / "slide.h5"
    with h5py.File(path, "w") as stream:
        stream.create_dataset("features", data=features, chunks=(128, 512), compression="gzip")
        stream["sam_region"] = regions
        stream["coords"] = coords
        stream["label"] = 3
    z, y, c, actual_coords, count, indices = load_slide(path, True, True, cap, 7)
    selected = (np.sort(np.random.default_rng(7).choice(len(features), cap, replace=False))
                if cap < len(features) else np.arange(len(features)))
    np.testing.assert_array_equal(z, features.astype(np.float32)[selected])
    np.testing.assert_array_equal(c, regions)
    np.testing.assert_array_equal(actual_coords, coords)
    assert count == len(features) and y == 3
    if cap < count:
        np.testing.assert_array_equal(indices, selected)
    else:
        assert indices is None


def test_unsampled_nonfinite_feature_still_fails(tmp_path):
    path = tmp_path / "slide.h5"
    count, cap, seed = 4200, 60, 7
    sampled = np.random.default_rng(seed).choice(count, cap, replace=False)
    unsampled = np.setdiff1d(np.arange(4100, count), sampled)[-1]
    with h5py.File(path, "w") as stream:
        features = stream.create_dataset("features", (count, 512), dtype="float32")
        features[unsampled, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        load_slide(path, False, False, cap, seed)


def test_completed_sam_stage_does_not_import_models(tmp_path, monkeypatch):
    folder = tmp_path / "sam"
    folder.mkdir()
    np.savez_compressed(folder / "slide.npz", regions=np.ones((2, 2)))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"torch", "openslide", "segment_anything"}:
            pytest.fail(f"Completed SAM stage imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    region_ops.generate_regions({"work_dir": str(tmp_path)}, [{"slide_id": "slide"}])


@pytest.mark.parametrize("changed", [None, "order", "patch_size"])
def test_merge_reuses_mapping_only_for_identical_geometry(tmp_path, monkeypatch, changed):
    (tmp_path / "sam").mkdir()
    (tmp_path / "label_mapping.json").write_text(json.dumps({"tumor": 0}))
    instance = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
    np.savez_compressed(tmp_path / "sam/slide.npz", regions=instance, level0_width=80, level0_height=40)
    geometries = []
    for encoder in ["a", "b"]:
        folder = tmp_path / f"trident/patches/features_{encoder}"
        folder.mkdir(parents=True)
        coords = np.array([[20, 0], [40, 0]])
        size = 20
        if encoder == "b" and changed == "order":
            coords = coords[::-1]
        if encoder == "b" and changed == "patch_size":
            size = 60
        geometries.append((coords, size))
        with h5py.File(folder / "slide.h5", "w") as stream:
            stream["features"] = np.ones((2, 4))
            stream["coords"] = coords
            stream["coords"].attrs["patch_size_level0"] = size
    original = region_ops.map_regions
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(region_ops, "map_regions", counted)
    region_ops.merge_features({"work_dir": str(tmp_path), "encoders": ["a", "b"]},
                              [{"slide_id": "slide", "label": "tumor"}])
    assert len(calls) == (1 if changed is None else 2)
    for encoder, (coords, size) in zip(["a", "b"], geometries):
        with h5py.File(tmp_path / f"features/{encoder}/slide.h5") as stream:
            np.testing.assert_array_equal(stream["sam_region"][:], original(coords, size, instance, 80, 40))


@pytest.mark.parametrize("em_iters", [0, 1, 3])
def test_sam_lp_computes_fixed_centroids_at_most_once(tmp_path, monkeypatch, em_iters):
    rng = np.random.default_rng(5)
    bags = [rng.normal(size=(96, 16)).astype(np.float32) for _ in range(6)]
    clusters = [np.repeat(np.arange(3), 32) for _ in bags]
    labels = np.arange(6) % 2
    original = sam_lp.sam_cluster_centroids
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sam_lp, "sam_cluster_centroids", counted)
    score = sam_lp.sam_lp_score(bags, clusters, labels, em_iters=em_iters, reduce_dim=8)
    assert np.isfinite(score)
    assert len(calls) == (len(bags) if em_iters else 0)
