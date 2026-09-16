"""Validate a cohort manifest and select exactly n training slides."""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_manifest(path, check_paths=False):
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"slide_id", "wsi_path", "label", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns {sorted(required)}")
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Manifest row {line} has the wrong number of columns")
            rows.append({k: v.strip() for k, v in row.items()})
    if not rows:
        raise ValueError("Manifest is empty")
    seen, paths, patient_splits = set(), set(), defaultdict(set)
    for row in rows:
        sid = row["slide_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sid) or sid in seen:
            raise ValueError(f"Invalid or duplicate slide_id: {sid!r}")
        seen.add(sid)
        if not row["label"] or not row["wsi_path"]:
            raise ValueError(f"Missing label or wsi_path for {sid}")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"{sid}: split must be train, val, or test")
        wsi = Path(row["wsi_path"]).expanduser()
        wsi = (path.parent / wsi).resolve() if not wsi.is_absolute() else wsi.resolve()
        if str(wsi) in paths:
            raise ValueError(f"WSI occurs more than once: {wsi}")
        paths.add(str(wsi))
        if check_paths and not wsi.is_file():
            raise FileNotFoundError(f"{sid}: {wsi}")
        row["wsi_path"] = str(wsi)
        if "patient_id" in row:
            if not row["patient_id"]:
                raise ValueError("patient_id must be present for every row when supplied")
            patient_splits[row["patient_id"]].add(row["split"])
    if any(len(splits) > 1 for splits in patient_splits.values()):
        raise ValueError("A patient occurs in multiple splits; correct the split before selection")
    return rows


def select_slides(rows, n, seed=0, strategy="balanced"):
    train = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["slide_id"])
    groups = defaultdict(list)
    for row in train:
        groups[row["label"]].append(row)
    if not groups or n < len(groups) or n > len(train):
        raise ValueError(f"n_slides must be between {len(groups)} classes and {len(train)} training slides")
    if strategy not in {"balanced", "proportional"}:
        raise ValueError("sampling must be balanced or proportional")
    rng = np.random.default_rng(seed)
    labels = sorted(groups)
    counts = np.array([len(groups[label]) for label in labels])
    # Reserve one per class, then allocate to the largest target deficit.
    target = np.full(len(labels), n / len(labels)) if strategy == "balanced" else n * counts / counts.sum()
    quota = np.ones(len(labels), dtype=int)
    for _ in range(n - len(labels)):
        deficit = np.where(quota < counts, target - quota, -np.inf)
        quota[np.argmax(deficit)] += 1
    selected = []
    for label, k in zip(labels, quota):
        indices = rng.choice(len(groups[label]), size=k, replace=False)
        selected.extend(groups[label][i] for i in indices)
    return sorted(selected, key=lambda row: row["slide_id"])


def write_manifest(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_sample(config):
    work = Path(config["work_dir"])
    rows = read_manifest(config["manifest"])
    selected = select_slides(rows, config["n_slides"], config["seed"], config["sampling"])
    # Only the sampled training slides need local image files. Validate all of
    # them before staging any output, including slides later in the sample.
    for row in selected:
        source = Path(row["wsi_path"])
        if not source.is_file():
            raise FileNotFoundError(f"{row['slide_id']}: {source}")
        if source.suffix.lower() not in {".svs", ".ndpi", ".tif", ".tiff"}:
            raise ValueError("Quickstart supports single-file OpenSlide WSIs: svs, ndpi, tif, tiff")
    sample_path = work / "sample.csv"
    if sample_path.exists():
        if read_manifest(sample_path) != selected:
            raise ValueError("Existing sample differs; use a new work_dir for a different sample")
    work.mkdir(parents=True, exist_ok=True)
    staged = work / "wsis"
    staged.mkdir(exist_ok=True)
    expected = {r["slide_id"] + Path(r["wsi_path"]).suffix for r in selected}
    if {p.name for p in staged.iterdir()} - expected:
        raise ValueError("Unexpected slides in staged WSI directory; use a fresh work_dir")
    for row in selected:
        source = Path(row["wsi_path"])
        target = staged / (row["slide_id"] + source.suffix)
        if target.is_symlink() and target.resolve() == source:
            continue
        if target.exists() or target.is_symlink():
            raise ValueError(f"Conflicting staged WSI: {target}")
        target.symlink_to(source)
    write_manifest(sample_path, selected)
    labels = sorted({r["label"] for r in rows if r["split"] == "train"})
    (work / "label_mapping.json").write_text(json.dumps({label: i for i, label in enumerate(labels)}, indent=2))
    print(f"Selected {len(selected)} training slides: {dict(Counter(r['label'] for r in selected))}")
    return selected
