# Implementation and reproducibility notes

## Scope

The quickstart is an executable workflow for **a new cohort**. It reuses the repository's seven existing metric implementations. It does not claim to regenerate the published result tables without the original datasets, model weights, sampled cohorts and experimental settings. The [paper](paper.pdf) remains the source for method definitions and experiments; [paper_details.md](paper_details.md) retains the research explanations and tables from the previous README.

## Third-party sources

The setup script verifies these inspected revisions:

| Project | Revision |
| --- | --- |
| [TRIDENT](https://github.com/mahmoodlab/TRIDENT) | `e4c98b2280d5b905fe83b00f01afac0f76755880` |
| [SAM 1](https://github.com/facebookresearch/segment-anything) | `dca509fe793f601edb92606367a655c15ac00fdf` |

The clones and checkpoints are ignored by Git. Setup stops if an existing clone differs, rather than resetting local changes. Both third-party projects retain their own licenses. The Python requirements specify supported ranges, not a complete platform-specific lockfile. Preserve `pip freeze` output with experiments and record encoder weight revisions if model repositories change.

## New-cohort defaults versus paper reproduction

| Aspect | New workflow |
| --- | --- |
| Cohort | Label-aware training-slide sampling; exact n, one or more per class; balanced by default |
| Repeats | One run per config/seed; repeat manually with fresh run directories |
| WSI patches | Shared 20x, 256px grid by default; configurable to match the intended FE/task protocol |
| SAM input | Full-slide RGB thumbnail, longest side <=2048, original aspect ratio; no ROI crop or letterboxing |
| SAM coverage | White masks (mean intensity >250) and tiny masks are filtered; no nearest-region fill for uncovered patches |
| Mapping | Majority positive region within each patch's projected bounding box; independent x/y scales from actual thumbnail dimensions |
| Missing regions | Unmatched patches get -1; removed for SAM metrics and reported in coverage |
| SAM-LP | Existing mean-pooling initialization and final L2 normalization retained; soft selection, size weights and one EM step explicitly selected |
| SAM-Cluster | Existing Monte Carlo estimator, normalized centroid similarities, score clipping and degenerate-pair fallback retained; threshold 0.95, top-k exclusion disabled |
| Combined ranking | Mean of per-metric FE ranks; an unvalidated practical heuristic |

The paper's Eq. 5 describes initialization from weighted normalized prototypes. The current metric function instead initializes from the mean of patch embeddings. Changing that scientific implementation is outside this repository-organization pass. Likewise the original research preprocessing used ROI-based thumbnail preparation and coverage filling; the portable quickstart uses the explicitly described full-slide mapping. These differences can change numerical scores.

The previous README incorrectly printed SAM-Cluster's threshold as `95`; the camera-ready paper specifies **0.95**. All metric defaults now come from [`configs/suitability.json`](../configs/suitability.json), shared by the Python functions and integrated runners. The released defaults use 128-dimensional Gaussian projection (except Effective Dimension), SAM-Cluster threshold 0.95 with no top-k exclusion, and SAM-LP with one EM step, soft selection and linear size weighting. These replace historical standalone defaults such as SAM-LP's top-k selection and two iterations. The [configuration guide](../suitability_score/README.md#one-configuration-for-every-entry-point) explains overrides and the complete parameter list.

## Data and resuming

The sample is saved before any extraction. Only training rows are used. Patient-level split overlap is checked when `patient_id` is supplied; sampling remains slide-level. The preprocessing workflow stages symlinks using `slide_id` as the basename so TRIDENT outputs match the manifest. Multi-file WSI formats such as MRXS are not handled by this staging helper.

The resolved workflow config, including the referenced scoring configuration, is saved and checked on subsequent runs. Scoring metadata records every effective metric setting, including any CLI overrides. Do not replace the contents of a source WSI, checkpoint, or installed third-party checkout midway through a run; use a new output directory when inputs change. TRIDENT and SAM reuse completed outputs. HDF5 merging writes copies through temporary files; raw extracted features are not modified.

The scoring runner verifies equal slide sets, patch counts, coordinate order when available, and slide labels across FEs. It does not silently drop missing slides. The quickstart uses the same patch grid for all FEs; this tests FE suitability under that grid, not each FE's individually optimal imaging protocol.

## Local validation

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q
python -m preprocessing --config configs/quickstart.json --dry-run
```

Synthetic tests cover sample size and class coverage, patient split leakage, deterministic patch/region mapping, copied HDF5 outputs, all seven score paths, ranking ties and invalid cross-FE inputs. These tests do not validate SAM segmentation quality or GPU feature extraction. A real WSI pilot with accessible encoder weights and the SAM checkpoint is required to validate the installed preprocessing stack.

### Release review — 2026-09-16

The publishable working-tree files were copied to a clean directory, excluding local data, outputs and third-party clones. In a newly created Python 3.11 environment:

- Core dependency installation and `pip check` completed successfully.
- All **33 tests** passed, including shared defaults, custom configs, invalid inputs and patch/region alignment.
- The README's synthetic CPU demo produced scores and rankings for both artificial encoders.
- The preprocessing dry run resolved the config and generated the expected TRIDENT commands.
- Local documentation links, heading anchors, image references, citation YAML and CI YAML were checked.

Core packages tested: NumPy 2.4.6, SciPy 1.17.1, h5py 3.16.0, scikit-learn 1.7.2 and Numba 0.67.0. These are tested versions, not a mandatory lockfile. GitHub Actions is configured for Python 3.10 and 3.11; remote CI results are available after pushing the workflow.

Full TRIDENT/SAM dependency installation and real-WSI model inference were **not** validated in this release review. Use a small real cohort to validate the full stack, model access and segmentation quality before starting a large extraction run. The [third-party terms](third_party.md) also apply to the optional preprocessing stack.

### Execution review

The subsequent execution review preserved the scoring definitions, random seeds, patch selection and default configuration. It introduced block-based HDF5 loading with full-row validation, avoided full-coverage SAM feature copies, released each FE's arrays before loading the next, reused identical patch-grid mappings, skipped SAM model loading when all maps already exist, and cached fixed centroids within each SAM-LP call.

**45 tests passed** in the same fresh core environment, including compressed float32/float64 HDF5 sampling across read-block boundaries, non-finite values outside the scoring sample, changed mapping geometry, resumed SAM execution and multiple EM iterations.

The pre-optimization snapshot and updated code were measured in separate processes, with NumPy/BLAS threads set to one. Values below are medians of three local synthetic CPU runs. Times measure the scoring or merge function, excluding process startup and top-level imports; memory is peak process RSS. [Measurement details and individual trials](results/execution_review.json) are provided for context.

| Workload | Before | After |
| --- | ---: | ---: |
| Default five scores: peak memory | 417.85 MiB | 284.40 MiB |
| Default five scores: elapsed time | 1.813 s | 1.717 s |
| Shared-grid region mapping and HDF5 merge | 0.317 s | 0.084 s |

The scoring fixture contained **8 slides × 2 FEs**, with 20,000 float32 patches of 512 dimensions per slide/FE; defaults retained 6,000 patches and projected to 128 dimensions. Peak memory fell by about **32%**. Default scores and ranks were exactly equal before and after in this environment. The merge fixture used one slide, four FEs, 10,000 patches and 32-dimensional features; every merged data array was exactly equal.

A separate comparison exercised all seven metrics, partial SAM coverage, different FE dimensions, patch subsampling and three SAM-LP EM iterations under both soft and top-k selection. Scores, rankings and coverage also matched exactly. These are local synthetic checks, not a claim of end-to-end WSI speedup. Real-WSI extraction, SAM inference and GPU memory remain unmeasured; merge copying and full input validation still incur disk I/O.
