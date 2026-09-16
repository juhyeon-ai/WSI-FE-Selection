# Workflow guide

[Back to the project](../README.md) · [Metric reference](../suitability_score/README.md) · [Implementation notes](implementation_notes.md)

## Quickstart

### 1. Clone and install

Use Linux and Python 3.10 or 3.11. A CUDA GPU is recommended for WSI feature extraction and SAM ViT-H. Metric evaluation alone can run on CPU.

```bash
git clone https://github.com/juhyeon-ai/WSI-FE-Selection.git
cd WSI-FE-Selection
python3 -m venv .venv  # Use Python 3.10 or 3.11.
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a compatible **PyTorch + torchvision** pair for your machine using the [PyTorch installer](https://pytorch.org/get-started/locally/), then run:

```bash
bash scripts/setup_dependencies.sh
```

The setup script clones [TRIDENT](https://github.com/mahmoodlab/TRIDENT) and [Segment Anything (SAM 1)](https://github.com/facebookresearch/segment-anything) into `third_party/`, verifies pinned commits, and installs both. See [implementation notes](../docs/implementation_notes.md) for the revisions. The script also checks dependency consistency with `pip check`. Keep these commands in a dedicated virtual environment.

Download the [official SAM 1 ViT-H checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints):

```bash
mkdir -p checkpoints
curl -fL --retry 3 -o checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Some pathology encoders require access to their Hugging Face model repositories. Obtain access for each candidate and authenticate before extraction:

```bash
python -m pip install huggingface_hub
hf auth login
```

TRIDENT downloads supported encoder weights when needed. See its [encoder documentation](https://github.com/mahmoodlab/TRIDENT/blob/main/README.md) for model-specific dependencies and access requirements. To check the mechanics before using gated models, set `encoders` to `["resnet50"]`; comparing FEs requires multiple candidates.

### 2. Prepare WSIs and a label manifest

Copy [`examples/slides.csv`](../examples/slides.csv) and fill in your own paths:

```csv
slide_id,wsi_path,label,split,patient_id
slide_001,/data/slides/slide_001.svs,normal,train,patient_001
slide_002,/data/slides/slide_002.svs,tumor,train,patient_002
slide_003,/data/slides/slide_003.svs,tumor,test,patient_003
```

- `slide_id`: unique, using letters, digits, `.`, `_`, or `-`; also becomes the output filename.
- `wsi_path`: absolute path, or a path relative to the CSV. The quickstart supports single-file OpenSlide-readable `.svs`, `.ndpi`, `.tif`, and `.tiff` slides with usable resolution metadata.
- `label`: nonempty class label. A consistent integer mapping is saved automatically.
- `split`: `train`, `val`, or `test`. **Only training slides are sampled.**
- `patient_id`: optional; if present, required for every row and checked for overlap between splits. Without it, you must establish patient-level splits yourself.

Sampling is by slide, not patient. Repeated slides from one patient can be selected. Grade labels are treated as classification labels for LP/SAM-LP; the quickstart does not compute QWK or survival scores. The example CSV contains placeholders, not downloadable data.

Edit [`configs/quickstart.json`](../configs/quickstart.json):

| Setting | Meaning |
| --- | --- |
| `manifest` | Your CSV path |
| `work_dir` | Output directory; choose a new one for a new sample/configuration |
| `encoders` | Candidate TRIDENT patch encoder names |
| `n_slides` | **Total** number of training slides, not slides per class |
| `sampling` | `balanced` for approximately equal class counts, or `proportional` to preserve class frequencies as closely as possible |
| `score_config` | Shared metric config, relative to this workflow JSON; its `shared.seed` controls slide/patch sampling and metric randomness |
| `gpu` | GPU index; `-1` for CPU |
| `magnification`, `patch_size` | Shared patch grid for every FE |

Metric hyperparameters are managed together in [`configs/suitability.json`](../configs/suitability.json). Edit that file, or copy it and point `score_config` to the copy. The released settings are identical for direct Python calls and the integrated runner. See the [configuration reference](../suitability_score/README.md#one-configuration-for-every-entry-point).

All config paths are relative to the **config file**, independent of your shell directory. Every class is represented. Request at least as many slides as classes, and no more than the available training slides. The default `n_slides=30` requires your own cohort with at least 30 training slides; choose a smaller value for a small pilot.

### 3. Run the suitability workflow

Preview paths and TRIDENT commands:

```bash
python -m preprocessing --config configs/quickstart.json --dry-run
```

Run all stages:

```bash
python -m preprocessing --config configs/quickstart.json
```

Or run them separately:

```bash
# Label-aware sample; stage symlinks to the original WSIs.
python -m preprocessing --config configs/quickstart.json --stages sample
# Shared tissue segmentation/patch coordinates; features per candidate FE → .h5.
python -m preprocessing --config configs/quickstart.json --stages extract
# Full-slide thumbnails → SAM 1 regions, once per slide.
python -m preprocessing --config configs/quickstart.json --stages sam
# Copy feature files and add patch-aligned SAM region IDs and slide labels.
python -m preprocessing --config configs/quickstart.json --stages merge
# NESum, Self-Cluster, SAM-Cluster, Linear Probing, and SAM-LP.
python -m preprocessing --config configs/quickstart.json --stages score
```

Re-running the same config reuses the saved sample, TRIDENT outputs and SAM maps; if all SAM maps exist, the SAM stage skips model loading entirely. Merging and scoring are recomputed. Mapping is shared across FEs with identical coordinates and patch sizes. Source WSIs and TRIDENT feature files are preserved. A changed config requires a new `work_dir` to avoid mixing stale results. Inspect `sam/*_overlay.jpg` and region coverage before interpreting scores.

### 4. Read the scores and shortlist FEs

```text
runs/quickstart/
├── sample.csv                 # Exact sampled cohort, reused across FEs
├── label_mapping.json
├── config.resolved.json
├── wsis/                      # Symlinks, not copies of WSIs
├── trident/patches/            # Shared patch grid and raw features_<encoder>/
├── sam/                       # Thumbnails, overlays, region maps + dimensions
├── features/<encoder>/*.h5    # Features + labels + aligned SAM IDs
└── scores/
    ├── scores.csv             # Raw scores; higher is better
    ├── ranking.csv            # Per-metric ranks and mean rank; lower is better
    ├── coverage.csv           # SAM coverage / number of regions per slide
    └── metadata.json          # Seeds, metric settings, slides, software versions
```

**We recommend considering NESum, Self-Cluster, SAM-Cluster, Linear Probing, and SAM-LP together.** They describe different aspects of representation quality. Look for candidates that rank well across several metrics, then validate a shortlist with your downstream task. `Linear Prob.` in the paper refers to Linear Probing.

`ranking.csv` averages **ranks**, since raw metric scales differ. This equal-weight mean rank is a practical screening heuristic, **not an additional metric validated in the paper**. Keep the individual rankings visible, especially when they disagree. LP/SAM-LP scores are in-sample log-likelihoods, not accuracy, calibrated performance estimates, or held-out validation results. Repeat with different seeds and fresh run directories to assess selection stability.

## Already have feature files?

Install only `requirements.txt`, arrange files as `features/<encoder>/<slide_id>.h5`, then run:

```bash
python -m pip install -r requirements.txt
python -m suitability_score --feature-root /data/features --output-dir runs/my_scores
```

See the [HDF5 contract and all seven metrics](../suitability_score/README.md). Labels and SAM regions are only required by the metrics that use them:

```bash
python -m suitability_score --feature-root /data/features --output-dir runs/unsupervised \
  --metrics nesum self_cluster effective_dimension
```


## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `No module named ...` | Activate `.venv` and install `requirements.txt` for scoring, or run `bash scripts/setup_dependencies.sh` for preprocessing. |
| PyTorch / torchvision import error | Install a mutually compatible pair for your Python/CUDA environment using the official PyTorch installer. |
| `libGL.so.1` or `libglib-2.0.so` missing | On minimal Ubuntu servers, install the system libraries required by OpenCV, typically `libgl1` and `libglib2.0-0`. |
| Encoder download returns 401 / 403 | Obtain access to that encoder's model repository and authenticate with `hf auth login`. |
| Missing SAM checkpoint | Follow the download command above; make sure the checkpoint matches `sam_model_type`. |
| Invalid `n_slides` | n is a total, not per-class count. It must be at least the number of classes and no more than the training rows. |
| Example manifest cannot find WSIs | `examples/slides.csv` is a format template. Replace its placeholder paths with your own data. |
| `Config differs from the existing run` | Use a new `work_dir` when changing cohort or scoring settings. |
| Features or slide sets differ between FEs | Use the same sample and patch grid for every FE. Failed or missing slides are not silently dropped. |
| Too few assigned SAM patches / regions | Inspect `sam/*_overlay.jpg`. Check thumbnail quality, SAM parameters, and the number of tissue patches; do not interpret a failed region mapping as an FE score. |
| Missing magnification / MPP | Confirm the WSI includes resolution metadata usable by OpenSlide/TRIDENT. The quickstart does not infer unknown physical resolution. |

The manifest may contain held-out slides whose images are not local. Only sampled training WSI files must exist for preprocessing. Patient split overlap is still checked across the complete manifest.

For reproducibility reports, include your command, config, package versions and the relevant error output. Avoid sharing private WSI files or identifying patient information in public issues.
