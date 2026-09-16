import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from preprocessing.__main__ import extraction_commands, load_config, run
from preprocessing.manifest import read_manifest, select_slides, prepare_sample, write_manifest
from preprocessing.regions import build_instance_map, map_regions, merge_features
from suitability_score.__main__ import ALL_METRICS, evaluate


def cohort(tmp_path):
    rows = []
    for i, label in enumerate(["a"] * 8 + ["b"] * 3 + ["c"]):
        source = tmp_path / f"s{i}.svs"
        source.touch()
        rows.append(dict(slide_id=f"s{i}", wsi_path=str(source), label=label, split="train", patient_id=f"p{i}"))
    return rows


@pytest.mark.parametrize("strategy", ["balanced", "proportional"])
def test_sample_exact_class_coverage_and_reproducibility(tmp_path, strategy):
    rows = cohort(tmp_path)
    for n in range(3, 13):
        sample = select_slides(rows, n, 4, strategy)
        assert len(sample) == n
        assert len({r["slide_id"] for r in sample}) == n
        assert {r["label"] for r in sample} == {"a", "b", "c"}
        assert sample == select_slides(list(reversed(rows)), n, 4, strategy)
    with pytest.raises(ValueError):
        select_slides(rows, 2)
    with pytest.raises(ValueError):
        select_slides(rows, 13)


def test_manifest_detects_patient_leakage_and_duplicate_sources(tmp_path):
    rows = cohort(tmp_path)
    path = tmp_path / "slides.csv"
    rows[1]["patient_id"] = rows[0]["patient_id"]
    rows[1]["split"] = "test"
    write_manifest(path, rows)
    with pytest.raises(ValueError, match="patient"):
        read_manifest(path)
    rows[1]["patient_id"] = "other"
    rows[1]["wsi_path"] = rows[0]["wsi_path"]
    write_manifest(path, rows)
    with pytest.raises(ValueError, match="more than once"):
        read_manifest(path)


def test_sample_excludes_test_and_preserves_originals(tmp_path):
    rows = cohort(tmp_path)
    rows[0]["split"] = "test"
    manifest = tmp_path / "slides.csv"
    write_manifest(manifest, rows)
    config = dict(work_dir=str(tmp_path / "run"), manifest=str(manifest), n_slides=6, seed=0, sampling="balanced")
    selected = prepare_sample(config)
    assert "s0" not in {r["slide_id"] for r in selected}
    assert all(p.is_symlink() for p in (tmp_path / "run/wsis").iterdir())
    assert selected == prepare_sample(config)
    with pytest.raises(ValueError, match="differs"):
        prepare_sample({**config, "n_slides": 7})


def test_mapping_uses_box_majority_non_square_dimensions_and_unmatched():
    instance = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 3, 3]], dtype=np.int32)
    coords = np.array([[0, 0], [40, 0], [0, 80], [40, 80]])
    # Level-0 width/height 80x120 -> thumbnail 4x3; different axis scales.
    assert map_regions(coords, 40, instance, 80, 120).tolist() == [1, 2, -1, 3]
    with pytest.raises(ValueError, match="outside"):
        map_regions(np.array([[81, 0]]), 40, instance, 80, 120)


def test_small_masks_override_large_and_white_is_filtered():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    large = np.ones((8, 8), dtype=bool)
    small = np.zeros((8, 8), dtype=bool)
    small[:2, :2] = True
    result = build_instance_map([{"segmentation": small}, {"segmentation": large}], rgb, 2)
    assert result[0, 0] != result[-1, -1]
    with pytest.raises(ValueError, match="no usable"):
        build_instance_map([{"segmentation": large}], np.full_like(rgb, 255), 2)


def test_merge_copies_h5_and_retains_patch_order(tmp_path):
    raw = tmp_path / "trident/patches/features_fake"
    raw.mkdir(parents=True)
    (tmp_path / "sam").mkdir()
    (tmp_path / "label_mapping.json").write_text('{"tumor": 0}')
    source = raw / "slide.h5"
    with h5py.File(source, "w") as stream:
        stream["features"] = np.arange(12).reshape(4, 3)
        stream["coords"] = np.array([[0, 0], [40, 0], [0, 40], [40, 40]])
        stream["coords"].attrs["patch_size_level0"] = 40
    np.savez_compressed(tmp_path / "sam/slide.npz", regions=np.array([[1, 2], [1, 2]]), level0_width=80, level0_height=80)
    merge_features(dict(work_dir=str(tmp_path), encoders=["fake"]), [dict(slide_id="slide", label="tumor")])
    with h5py.File(source) as original, h5py.File(tmp_path / "features/fake/slide.h5") as merged:
        assert "label" not in original
        np.testing.assert_array_equal(original["features"], merged["features"])
        assert merged["sam_region"][:].tolist() == [1, 2, 1, 2]
        assert merged["sam_region"].id == merged["sam_cluster"].id
        assert merged["label"][()] == 0


def synthetic_features(tmp_path):
    root = tmp_path / "features"
    rng = np.random.default_rng(10)
    for sid in range(6):
        regions = np.repeat(np.arange(3), 32)
        centers = rng.normal(size=(3, 12))
        z = (centers[regions] + 0.1 * rng.normal(size=(96, 12))).astype(np.float32)
        for encoder in ["fe_a", "fe_b"]:
            folder = root / encoder
            folder.mkdir(parents=True, exist_ok=True)
            with h5py.File(folder / f"slide_{sid}.h5", "w") as stream:
                stream["features"] = z
                stream["sam_region"] = regions
                stream["coords"] = np.column_stack([np.arange(96), np.zeros(96)])
                stream["label"] = sid % 2
    return root


def test_all_seven_metrics_outputs_and_tied_rankings(tmp_path):
    root = synthetic_features(tmp_path)
    scores, ranks = evaluate(root, tmp_path / "scores", metrics=ALL_METRICS)
    for metric in ALL_METRICS:
        assert np.isfinite(scores[0][metric])
        assert scores[0][metric] == pytest.approx(scores[1][metric])
        assert ranks[0][metric + "_rank"] == 1.5
    assert ranks[0]["mean_rank"] == 1.5
    assert set(p.name for p in (tmp_path / "scores").iterdir()) == {"scores.csv", "ranking.csv", "coverage.csv", "metadata.json"}
    metadata = json.loads((tmp_path / "scores/metadata.json").read_text())
    assert metadata["sam_lp"]["em_iters"] == 1
    assert metadata["sam_cluster"]["threshold"] == 0.95


@pytest.mark.parametrize("invalid", ["missing_slide", "label", "coords", "nan", "region_shape", "region_assignments"])
def test_invalid_inputs_fail_instead_of_ranking(tmp_path, invalid):
    root = synthetic_features(tmp_path)
    path = root / "fe_b/slide_0.h5"
    if invalid == "missing_slide":
        path.unlink()
    else:
        with h5py.File(path, "r+") as stream:
            if invalid == "label":
                stream["label"][()] = 3
            elif invalid == "coords":
                stream["coords"][0, 0] = 1000
            elif invalid == "nan":
                stream["features"][0, 0] = np.nan
            elif invalid == "region_assignments":
                stream["sam_region"][0] = 99
            else:
                del stream["sam_region"]
                stream["sam_region"] = np.array([0, 1])
    with pytest.raises(ValueError):
        evaluate(root, tmp_path / "scores", metrics=["sam_lp"])
    assert not (tmp_path / "scores/ranking.csv").exists()


def test_config_paths_and_trident_shared_grid(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    file = config_dir / "config.json"
    file.write_text(json.dumps(dict(manifest="../slides.csv", work_dir="../run", trident_dir="../trident", sam_checkpoint="../sam.pth", encoders=["resnet50", "uni_v1"])))
    config = load_config(file)
    assert config["work_dir"] == str(tmp_path / "run")
    commands = list(extraction_commands(config))
    assert [c[c.index("--task") + 1] for c in commands] == ["seg", "coords", "feat", "feat"]
    assert all(c[c.index("--coords_dir") + 1] == "patches" for c in commands)
    run(config, ["sample", "extract"], dry_run=True)
    assert not (tmp_path / "run").exists()


def test_direct_metric_defaults_match_integrated_defaults(tmp_path):
    from suitability_score.config import DEFAULT_CONFIG
    from suitability_score.effective_dimension import pca_effective_dim
    from suitability_score.linear_probing import linear_probing_score
    from suitability_score.logme import logme_score
    from suitability_score.nesum import nesum_score
    from suitability_score.sam_cluster import sam_cluster_score
    from suitability_score.sam_lp import sam_lp_score
    from suitability_score.self_cluster import self_cluster_score_slides

    root = synthetic_features(tmp_path)
    bags, regions, labels = [], [], []
    for path in sorted((root / "fe_a").glob("*.h5")):
        with h5py.File(path) as stream:
            bags.append(stream["features"][:])
            regions.append(stream["sam_region"][:])
            labels.append(stream["label"][()])
    y = np.array(labels)
    expected = {
        "nesum": np.mean([nesum_score(z) for z in bags]),
        "self_cluster": self_cluster_score_slides(bags, regions, y),
        "sam_cluster": sam_cluster_score(bags, regions, y),
        "linear_probing": linear_probing_score(bags, y),
        "sam_lp": sam_lp_score(bags, regions, y),
        "effective_dimension": np.mean([pca_effective_dim(z) for z in bags]),
        "logme": logme_score(bags, y),
    }
    actual, _ = evaluate(root, tmp_path / "scores", encoders=["fe_a"], metrics=ALL_METRICS)
    for metric, value in expected.items():
        assert actual[0][metric] == pytest.approx(value)
    # This is the released scientific default, not the historical top-k preset.
    assert DEFAULT_CONFIG["sam_lp"]["em_iters"] == 1
    assert DEFAULT_CONFIG["sam_lp"]["select_mode"] == "soft"


def test_custom_config_changes_real_computation_and_records_effective_values(tmp_path):
    root = synthetic_features(tmp_path)
    baseline, _ = evaluate(root, tmp_path / "baseline", metrics=["linear_probing"])
    path = tmp_path / "custom.json"
    path.write_text(json.dumps({
        "metrics": ["linear_probing", "sam_lp", "sam_cluster"],
        "shared": {"reduce_dim": 8, "seed": 7, "max_patches": 60},
        "linear_probe": {"C": 0.2, "max_iter": 500},
        "sam_lp": {"em_iters": 0, "beta": 3},
        "sam_cluster": {"n_pairs": 400, "ratio_sim_threshold": 0.7,
                        "rng_seed_offsets": [2, 4], "dtype": "float64"},
    }))
    actual, _ = evaluate(root, tmp_path / "custom_scores", config=path)
    assert actual[0]["linear_probing"] != pytest.approx(baseline[0]["linear_probing"])
    # Disabling the EM update makes SAM-LP identical to the mean-pooling probe.
    assert actual[0]["sam_lp"] == pytest.approx(actual[0]["linear_probing"])
    meta = json.loads((tmp_path / "custom_scores/metadata.json").read_text())
    assert meta["scoring_config"]["linear_probe"]["C"] == 0.2
    assert meta["sam_cluster"]["threshold"] == 0.7
    assert meta["sam_cluster"]["rng_seeds"] == [9, 11]
    assert meta["sam_cluster"]["max_seconds"] is None
    with (tmp_path / "custom_scores/coverage.csv").open() as stream:
        assert {row["patches_scored"] for row in csv.DictReader(stream)} == {"60"}


@pytest.mark.parametrize("override", [
    {"sam_lp": {"betta": 4}},
    {"shared": {"reduce_dim": 0}},
    {"shared": {"seed": True}},
    {"shared": {"max_patches": 20}},
    {"sam_cluster": {"n_pairs": 10}},
    {"sam_cluster": {"max_seconds": float("inf")}},
    {"sam_cluster": {"rng_seed_offsets": []}},
    {"sam_lp": {"select_mode": "invalid"}},
    {"metrics": []},
])
def test_invalid_scoring_config_rejected_before_outputs(tmp_path, override):
    with pytest.raises(ValueError):
        evaluate(tmp_path / "absent_features", tmp_path / "scores", config=override)
    assert not (tmp_path / "scores").exists()


def test_config_cli_overrides_and_defaults_are_not_mutated(tmp_path):
    import os
    import subprocess
    import sys
    from suitability_score.config import load_config as load_scoring_config

    root = synthetic_features(tmp_path)
    config_file = tmp_path / "custom.json"
    config_file.write_text(json.dumps({"metrics": ["nesum"], "shared": {"seed": 5, "reduce_dim": 10}}))
    result = subprocess.run([
        sys.executable, "-m", "suitability_score", "--config", str(config_file),
        "--feature-root", str(root), "--output-dir", str(tmp_path / "scores"),
        "--seed", "9", "--reduce-dim", "8", "--metrics", "linear_probing",
    ], capture_output=True, text=True, env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
    assert result.returncode == 0, result.stderr
    metadata = json.loads((tmp_path / "scores/metadata.json").read_text())
    assert metadata["seed"] == 9 and metadata["reduce_dim"] == 8
    assert metadata["metrics"] == ["linear_probing"]
    assert load_scoring_config()["shared"]["seed"] == 0
    assert load_scoring_config(config_file)["shared"]["seed"] == 5


def test_preprocessing_passes_and_snapshots_scoring_config(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "score.json").write_text(json.dumps({"shared": {"seed": 6}, "sam_lp": {"beta": 3}}))
    workflow = configs / "workflow.json"
    workflow.write_text(json.dumps(dict(manifest="../slides.csv", work_dir="../run", trident_dir="../trident",
                                        sam_checkpoint="../sam.pth", encoders=["fe_a"], score_config="score.json")))
    config = load_config(workflow)
    assert config["seed"] == 6
    assert config["scoring"]["sam_lp"]["beta"] == 3
    work = tmp_path / "run"
    work.mkdir()
    write_manifest(work / "sample.csv", cohort(tmp_path)[:2])
    called = []
    monkeypatch.setattr("suitability_score.__main__.evaluate", lambda *a, **kw: called.append(kw))
    run(config, ["score"])
    assert called[0]["config"] == config["scoring"]
    saved = json.loads((work / "config.resolved.json").read_text())
    assert saved["scoring"]["sam_lp"]["beta"] == 3
    # Editing the referenced config is detected even though its path is unchanged.
    (configs / "score.json").write_text(json.dumps({"sam_lp": {"beta": 4}}))
    with pytest.raises(ValueError, match="differs"):
        run(load_config(workflow), ["score"])


def test_sampling_does_not_require_test_images(tmp_path):
    rows = cohort(tmp_path)
    rows[0]["split"] = "test"
    Path(rows[0]["wsi_path"]).unlink()
    file = tmp_path / "slides.csv"
    write_manifest(file, rows)
    selected = prepare_sample(dict(work_dir=str(tmp_path / "run"), manifest=str(file), n_slides=6, seed=0, sampling="balanced"))
    assert len(selected) == 6
    assert all(row["split"] == "train" for row in selected)


def test_manifest_reports_bad_csv_row(tmp_path):
    file = tmp_path / "slides.csv"
    file.write_text("slide_id,wsi_path,label,split\ns1,/data/a.svs,tumor,train,extra\n")
    with pytest.raises(ValueError, match="row 2"):
        read_manifest(file)


def test_workflow_rejects_typos_and_invalid_values(tmp_path):
    file = tmp_path / "config.json"
    base = dict(manifest="slides.csv", work_dir="run", trident_dir="trident", sam_checkpoint="sam.pth", encoders=["resnet50"])
    for change in [{"batch_sze": 32}, {"gpu": -2}, {"sam_model_type": "sam2"}, {"n_slides": True}, {"pred_iou_thresh": 1.1}]:
        file.write_text(json.dumps({**base, **change}))
        with pytest.raises(ValueError):
            load_config(file)


def test_missing_checkpoint_fails_before_expensive_stages(tmp_path):
    config = dict(work_dir=str(tmp_path / "run"), sam_checkpoint=str(tmp_path / "missing.pth"))
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run(config, ["sample", "sam"])
    assert not (tmp_path / "run").exists()


def test_non_sam_rerun_clears_old_coverage(tmp_path):
    root = synthetic_features(tmp_path)
    output = tmp_path / "scores"
    evaluate(root, output, metrics=["sam_lp"])
    assert (output / "coverage.csv").exists()
    evaluate(root, output, metrics=["nesum"])
    assert not (output / "coverage.csv").exists()
