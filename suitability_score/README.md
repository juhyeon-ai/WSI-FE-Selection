# Suitability scores

Run from the repository root:

```bash
python -m suitability_score --feature-root /data/features --output-dir runs/scores
```

The default evaluates the five complementary metrics recommended for screening. All metrics are available through one entry point:

```bash
python -m suitability_score --feature-root /data/features --output-dir runs/all_scores \
  --metrics nesum self_cluster sam_cluster linear_probing sam_lp effective_dimension logme
```

| CLI name | Implementation | Labels | SAM | Reported score |
| --- | --- | --- | --- | --- |
| `nesum` | [nesum.py](nesum.py) | No | No | Mean per-slide normalized eigensum |
| `self_cluster` | [self_cluster.py](self_cluster.py) | No | No | Mean per-slide Self-Cluster statistic |
| `sam_cluster` | [sam_cluster.py](sam_cluster.py) | No | Yes | Mean per-slide regional cohesion/separation score |
| `linear_probing` | [linear_probing.py](linear_probing.py) | Yes | No | Training average log-likelihood of a mean-pooling linear probe |
| `sam_lp` | [sam_lp.py](sam_lp.py) | Yes | Yes | Training average log-likelihood after regional reweighting |
| `effective_dimension` | [effective_dimension.py](effective_dimension.py) | No | No | Mean number of PCs explaining 99% variance |
| `logme` | [logme.py](logme.py) | Yes | No | Evidence on mean-pooled slide features |

The runner ranks these scores in descending order following the repository's suitability-score convention. `ranking.csv` uses average ranks for ties; lower mean rank is better. Joint ranking is a screening heuristic, not a paper-validated ensemble. Do not average raw scores with incompatible units. For one encoder, all ranks are 1 and do not establish superiority.

## HDF5 input contract

```text
features/
├── resnet50/
│   ├── slide_001.h5
│   └── slide_002.h5
└── uni_v1/
    ├── slide_001.h5
    └── slide_002.h5
```

| Dataset | Shape / dtype | Required |
| --- | --- | --- |
| `features` | `(n_patches, feature_dim)`, finite numeric values | Always |
| `label` | Scalar or one-element integer array | LP, SAM-LP, LogME |
| `sam_region` | `(n_patches,)`, integers | SAM-Cluster, SAM-LP |
| `coords` | `(n_patches, 2)`, level-0 top-left `(x, y)` | Preprocessing merge; recommended for score-only use |

`sam_cluster` is accepted as an alternative key. New merged files store both names as HDF5 hard links to the same array. IDs >=0 are assigned regions; **-1 means unmatched** and is removed only for SAM metrics. The new preprocessing code assigns regions starting at 1. If an existing file uses 0 for background, convert its unmatched IDs to -1 before evaluation.

All FE directories must have exactly the same nonempty slide set, labels, patch counts, and coordinates/order when coordinates are present. Never silently compare different cohorts. A feature dimension may differ between FEs, but must be constant across slides within each FE. At least two classes are required for supervised scores. The score-only CLI trusts that your input comes from training slides; split filtering is performed by the preprocessing workflow.

## Settings and interpretation

- Defaults: `--seed 0`, `--reduce-dim 128`, `--max-patches 6000` per slide. The same patch indices are used across FEs. HDF5 files are read in blocks; only the capped feature samples for one FE are retained in memory.
- All six metrics that support Gaussian random projection default to `reduce_dim=128`, both through Python functions and the CLI. Self-Cluster defaults to `reduce_method="gaussian"`. Effective Dimension measures the original feature space and has no projection target. If the input dimension is <=128, most metrics leave it unchanged; SAM-Cluster always projects to the requested dimension. See [implementation notes](../docs/implementation_notes.md).
- SAM-Cluster: threshold 0.95, no top-k exclusion, 20,000 sampled pairs per internal seed, two internal seeds, uniform cluster sampling, tiny-region filtering at 0.5%. Time-based truncation is disabled by default for repeatable sampling. The inherited implementation clips negative scores to 0 and falls back to distinct-cluster comparisons if every pair is excluded.
- SAM-LP: one EM iteration, soft selection, linear region-size prior, beta 5, minimum 20 patches per centroid. The existing implementation initializes with mean pooling and normalizes its slide representations. If all regions are too small, it uses a whole-slide centroid. These implementation choices are disclosed rather than presented as exact reproduction of the paper's equations.
- LP and SAM-LP use logistic regression with `C=1` and at most 2,000 solver iterations. Solver non-convergence fails the run instead of producing a misleading ranking.
- `coverage.csv` reports patch count, assigned-region fraction and number of regions after the patch cap. Inspect low coverage and overlays. SAM-Cluster rejects slides with fewer than 50 assigned patches or fewer than two sufficiently large regions.
- Scores indicate relative suitability within a shared cohort and protocol. They are not held-out accuracy, AUROC, QWK, or clinical performance. Repeat subset sampling with new seeds; compare the individual metric rankings before selecting a shortlist.

## Memory and runtime

The runner validates every feature row, including rows outside the scoring sample, using approximately 8 MiB source blocks. This bounds loading memory while preserving full input checks; the cap reduces retained features and computation, not the amount of input read from disk. Each file is opened once for its features, label, coordinates and region IDs.

The sampled float32 feature arrays use roughly `4 × slides × max_patches × feature_dim` bytes per FE when every slide reaches the cap. For example, 30 slides at 6,000 patches and 1,024 dimensions need about **703 MiB for those arrays alone**. Allow additional memory for metric calculations, full coordinates/region IDs and filtered copies when SAM coverage is incomplete. Encoders are evaluated sequentially and their feature arrays released before loading the next encoder. Reducing `reduce_dim` alone does not shrink the retained input arrays.

Full-coverage SAM metrics reuse the sampled arrays without copying them. SAM-LP prepares its fixed region centroids once, even when multiple EM iterations are requested. See the [execution review](../docs/implementation_notes.md#execution-review) for a measured synthetic comparison and validation limits.

## One configuration for every entry point

[`configs/suitability.json`](../configs/suitability.json) is the single source of metric defaults. Python functions, `python -m suitability_score`, and the full preprocessing workflow use this configuration. There are no separate scientific presets for standalone functions and the runner.

To keep the released defaults, run the usual commands without any overrides. To change settings, copy the JSON and edit the values you need:

```bash
cp configs/suitability.json configs/my_suitability.json
python -m suitability_score \
  --feature-root /data/features --output-dir runs/custom_scores \
  --config configs/my_suitability.json
```

Partial JSON overrides are also accepted. For example:

```json
{
  "shared": {"reduce_dim": 256, "seed": 42},
  "sam_lp": {"beta": 3.0}
}
```

Unspecified values inherit the released defaults. Unknown keys, invalid modes and out-of-range values fail before scoring. Precedence is **default JSON → custom JSON → explicit CLI flags** (`--metrics`, `--seed`, `--reduce-dim`, `--max-patches`). The effective configuration is saved under `scoring_config` in `scores/metadata.json`.

For the full WSI pipeline, set `"score_config": "my_suitability.json"` in your workflow JSON. This path is relative to the workflow JSON. Sampling uses `shared.seed` from the scoring config. A legacy workflow-level `seed`, if supplied, explicitly overrides both sampling and scoring seeds. The resolved workflow snapshot includes the complete scoring settings, so changes to the referenced file require a fresh `work_dir`.

### Python usage

```python
from suitability_score.sam_lp import sam_lp_score
from suitability_score.config import load_config, metric_kwargs

# The same released defaults used by the CLI.
score = sam_lp_score(patch_embeddings, sam_regions, labels)

# Apply a custom config to a direct function call.
config = load_config("configs/my_suitability.json")
score = sam_lp_score(
    patch_embeddings, sam_regions, labels,
    **metric_kwargs(config, "sam_lp"),
)
```

Function defaults are loaded when the Python modules are imported. Restart an existing notebook kernel after editing the default JSON, or use `load_config` and `metric_kwargs` to load changes explicitly. Keyword arguments can still override individual settings. For single-slide Self-Cluster calls, remove the `aggregate` key from `metric_kwargs`; it is used by the slide-list API and runner.

To disable projection in direct LP, SAM-LP, LogME or NESum calls, pass `reduce_dim=None`; for Self-Cluster and SAM-Cluster, pass `reduce_method="none"`. The shared JSON dimension remains a positive target dimension; Effective Dimension always measures the original feature space.

## Other hyperparameters

These are the defaults in the shared JSON, used by both direct calls and integrated scoring:

| Config section | Settings |
| --- | --- |
| `shared` | `reduce_dim=128`, `seed=0`, `max_patches=6000`, `eps=1e-12` |
| `linear_probe` | `C=1.0`, `max_iter=2000`; shared by LP and SAM-LP |
| `nesum` | `center=false` |
| `self_cluster` | `reduce_method="gaussian"`, `aggregate="mean"` |
| `sam_cluster` | `ratio_sim_threshold=0.95`, `ratio_topk_exclude=0`, `n_pairs=20000`, `min_cluster_ratio=0.005` |
| `sam_cluster` sampling | `rng_seed_offsets=[0,1]`, `cluster_sampling="uniform"`, `oversample_factor=6.0`, `max_seconds=null`, `dtype="float32"` |
| `sam_lp` | `em_iters=1`, `select_mode="soft"`, `weight_mode="linear"`, `beta=5.0`, `min_cluster_size=20`, `topk=1` |
| `effective_dimension` | `explained_var=0.99`, `center=true` |
| `logme` | `warmup_numba=false` |

`max_seconds=null` disables the time budget; a positive number sets a budget in seconds and can make sampling dependent on runtime. Internal SAM-Cluster seeds are `shared.seed + offset`. `topk` is used only if SAM-LP's selection mode is changed to `"topk"`; beta is used in `"soft"` mode. The shared patch cap is passed to SAM-Cluster as well as the runner, so a custom cap is not silently limited to a second hard-coded value. Other direct metric functions operate on the bags you pass them; the runner handles the common patch cap and input checks.
