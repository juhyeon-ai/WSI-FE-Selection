# Method and results

[Project overview](../README.md) · [Paper](paper.pdf) · [Workflow guide](quickstart.md) · [Benchmark CSV](results/table1.csv)

Research background, benchmark results and resource measurements. These are reported research results; see [implementation notes](implementation_notes.md) for the scope of the new-cohort workflow.

## Overview

WSI-based computational pathology relies on the Multiple Instance Learning (MIL) framework, in which a **Feature Extractor (FE)** and an **Aggregator** jointly determine performance. With pathology foundation models proliferating rapidly (Virchow2, UNI, CONCH, Phikon, …), there is **no universally optimal FE**, and identifying the best one per cohort traditionally demands exhaustive, prohibitively expensive MIL training.

**Why FE selection matters.** In our preliminary study (10 FEs × 9 aggregators × 8 datasets), performance swings dramatically with the FE choice — e.g., on Camelyon16 an ImageNet ResNet50 reaches 76.59% while Virchow2 reaches 97.98% (a 20+ point gap). No single FE wins everywhere: Virchow2 is best on Camelyon16, CONCH-v1 on BRACS, and UNI-v1 on PANDA. Yet exhaustively finding the best FE costs **450 full MIL training runs per dataset** (10 FEs × 9 aggregators × 5 seeds), scaling linearly with candidates and dataset size.

**Our solution.** A **pre-evaluation framework** that ranks candidate FEs *before* training. Existing transferability metrics from general computer vision assume single-image inputs and ignore the bag-of-instances nature and spatial heterogeneity of WSIs. We instead leverage SAM to infer region-level organization and evaluate representation quality at the bag level, yielding two WSI-aware metrics:

- **SAM-Cluster** — an *unsupervised* metric measuring structure-aware feature separability.
- **SAM-LP** — a *supervised* metric measuring diagnostic task alignment while mitigating signal dilution.

---

## Key features

- **No MIL training** — Rank candidate FEs in minutes rather than weeks.
- **Structure-aware metrics** — Use SAM masks as pseudo-structural labels, moving beyond the standard i.i.d. patch assumption.
- **High rank correlation** — Strong Spearman correlation with downstream MIL performance across 8 public WSI datasets.
- **Cost-efficient & scalable** — Reliable on a small random subset of WSIs (e.g., 30 slides); SAM runs **once per slide**, so cost scales at $\mathcal{O}(1)$ w.r.t. the number of candidate FEs.

---

## Method

We define a **Suitability Score** computed from pre-extracted features on a small random WSI subset. Given candidate FEs $\Phi$, we select

$$
\phi^{*} = \arg\max_{\phi \in \Phi} S_{\text{metric}}(\phi).
$$

**Setup.** The $i$-th WSI is a bag of patches $X_i = \{x_{i,j}\}_{j=1}^{K_i}$ with label $Y_i \in \{1,\dots,C\}$. An FE $\phi:\mathcal{X}\to\mathbb{R}^d$ maps each patch to an embedding $z_{i,j}$, and SAM provides masks $\mathcal{M}=\{M_k\}$ used as pseudo-structural regions.

### SAM-Cluster (Unsupervised)

An ideal FE keeps **high consistency within a tissue structure** and **clear boundaries between structures**. We measure intra-region cohesion and a *relaxed* inter-region separability, where mask pairs with overly similar centroids are excluded to mitigate SAM over-segmentation. With $L_2$-normalized region mean $c_k = \tfrac{1}{|Z^{(k)}|}\sum_{z\in Z^{(k)}} z$ and exclusion set $\mathcal{E} = \{(k,l)\mid c_k^\top c_l > \tau\}$:

$$
S_{intra} = \mathbb{E}_{k}\!\left[\mathbb{E}_{z_i,z_j\in Z^{(k)}}\!\left[(z_i^\top z_j)^2\right]\right],
\qquad
S_{inter} = \mathbb{E}_{(k,l)\notin\mathcal{E}}\!\left[\mathbb{E}_{z_i\in Z^{(k)},\,z_j\in Z^{(l)}}\!\left[(z_i^\top z_j)^2\right]\right]
$$

$$
S_{\text{SAM-Cluster}}(\phi) = 1 - \frac{S_{inter}}{S_{intra}}
$$

A higher score indicates superior preservation of histological boundaries.

### SAM-LP (Supervised)

Conventional mean-pooling Linear Probing suffers from **signal dilution**, where diagnostic signals from small regions (e.g., micrometastases) are overwhelmed by abundant background tissue. SAM-LP reconstructs each slide as a weighted sum of **structural prototypes** $c_{i,k}$ (the $L_2$-normalized mask centroids), with weights initialized proportional to mask size and refined by a **single EM update**:

$$
\tilde{z}_i^{(t)} = \sum_{k}\alpha_{i,k}^{(t)}\,c_{i,k}, \quad \text{s.t.}\ \sum_k \alpha_{i,k}^{(t)} = 1
$$

**M-Step** — fix $\tilde{z}_i^{(t)}$, update the linear classifier $\theta^{(t)}$:

$$
\theta^{(t)} = \arg\max_{\theta}\frac{1}{N}\sum_{i=1}^{N}\log P\!\left(Y_i\mid \tilde{z}_i^{(t)};\theta\right)
$$

**E-Step** — reweight prototypes by their contribution to predicting $Y_i$ ($\beta$ controls sharpness):

$$
\alpha_{i,k}^{(t+1)} = \frac{\alpha_{i,k}^{(0)}\exp\!\big(\beta\log P(Y_i\mid c_{i,k};\theta^{(t)})\big)}{\sum_{j}\alpha_{i,j}^{(0)}\exp\!\big(\beta\log P(Y_i\mid c_{i,j};\theta^{(t)})\big)}
$$

The final score evaluates task alignment with the optimized $\tilde{z}_i^{*}$ and $\theta^{*}$:

$$
S_{\text{SAM-LP}}(\phi) = \frac{1}{N}\sum_{i=1}^{N}\log P\!\left(Y_i\mid \tilde{z}_i^{*};\theta^{*}\right)
$$

> **Default hyperparameters** (used across all experiments): similarity threshold `τ = 0.95` for SAM-Cluster, attention sharpness `β = 5` for SAM-LP. Both metrics are stable over broad ranges and need no dataset-specific tuning.

### Baseline Metrics

We benchmark against four unsupervised and three supervised pre-evaluation metrics:

| Metric | Setting | Label-free | Core idea |
| :-- | :-- | :--: | :-- |
| Effective Dimension | Unsupervised (richness) | ✅ | # principal components explaining 99% variance |
| NESum | Unsupervised (richness) | ✅ | feature-space isotropy |
| Self-Cluster | Unsupervised (separability) | ✅ | global patch clustering quality |
| **SAM-Cluster (Ours)** | Unsupervised (separability) | ✅ | structure-aware regional cohesion vs. separability |
| LogME | Supervised | ❌ | maximum label evidence, no gradient optimization |
| Linear Probing | Supervised | ❌ | linear classifier on mean-pooled slide embedding |
| **SAM-LP (Ours)** | Supervised | ❌ | structure-aware prototypes refined by EM |

---

## Running the code

See the [quickstart](../README.md#quickstart) for the executable workflow. Full MIL training is only needed to reproduce downstream benchmark comparisons, not to select an FE on a new dataset. Benchmark code and splits are in [`Benchmark-MIL/`](../Benchmark-MIL/); reference results are in [`Ground Truth/`](../Ground%20Truth/).

---

## Results

### Rank Correlation with Downstream MIL Performance

Spearman's $\rho$ between each metric and the GT MIL ranking across 8 datasets. **Bold** marks the best metric in each row. Standard deviations (after `±`) come from repeating random subset sampling 5× (seeds 0–4); each subset is 30 WSIs, except PANDA (600 WSIs, due to its smaller slide size).

| Dataset | Eff. Dim | NESum | Self-Cluster | **SAM-Cluster** | LogME | Linear Prob. | **SAM-LP** |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Camelyon16 | 0.1030±.000 | 0.3576±.036 | 0.3697±.016 | **0.3939**±.052 | -0.1903±.209 | 0.3091±.030 | 0.3212±.174 |
| BRACS | -0.3939±.000 | 0.0909±.022 | 0.1273±.025 | 0.3212±.062 | -0.2024±.136 | 0.2364±.082 | **0.3697**±.099 |
| UBC-OCEAN | 0.2242±.000 | 0.6485±.011 | 0.6121±.020 | 0.6727±.050 | 0.1806±.100 | **0.8303**±.057 | **0.8303**±.037 |
| TCGA-GLIOMA | 0.3818±.000 | 0.3455±.059 | 0.1879±.032 | 0.3697±.083 | **0.5103**±.087 | 0.4545±.101 | 0.4424±.093 |
| TCGA-NSCLC | -0.1636±.000 | 0.4061±.025 | 0.3212±.051 | 0.4909±.022 | 0.1976±.196 | 0.6364±.079 | **0.9515**±.045 |
| TCGA-RCC | 0.0303±.005 | 0.5273±.059 | 0.6485±.063 | 0.7212±.038 | 0.4836±.010 | 0.6364±.066 | **0.7939**±.066 |
| Histai-skin-b1 | 0.3697±.049 | 0.5879±.000 | 0.6606±.000 | **0.7091**±.027 | 0.1515±.165 | 0.6727±.044 | **0.7091**±.087 |
| PANDA | 0.5030±.000 | 0.5879±.000 | 0.5636±.000 | 0.6000±.005 | **0.7430**±.034 | 0.6121±.000 | 0.5879±.072 |
| **Average** | 0.1318 | 0.4439 | 0.4364 | **0.5348** | 0.2342 | 0.5485 | **0.6258** |

**Highlights**

- **SAM-Cluster** (avg. **0.5348**) surpasses Self-Cluster (0.4364) and NESum (0.4439) on all 8 datasets — significant under a Wilcoxon signed-rank test (*p* = 0.0078). Gains are largest on structurally complex datasets (Camelyon16, BRACS, Histai-skin-b1).
- **SAM-LP** (avg. **0.6258**) outperforms Linear Probing (0.5485), with the largest gain on TCGA-NSCLC (0.6364 → **0.9515**).

### When SAM-LP Helps Most (Structural Complexity)

SAM-LP's advantage grows with slide structural complexity (avg. patches × SAM regions):

- **Low complexity → LP suffices:** PANDA (201 patches, 19.7 regions), TCGA-GLIOMA (2,994 patches, 53.2 regions); on par for UBC-OCEAN (3,175 patches, 70.5 regions).
- **High complexity → SAM-LP wins:** BRACS (3,618 / 128.6), TCGA-NSCLC (3,834 / 90.3), TCGA-RCC (4,000 / 71.9).

> **Guidance:** SAM-LP is most beneficial on large, structurally heterogeneous slides where signal dilution is severe.

### Aggregator-Agnostic Robustness

The averaged GT is representative rather than aggregator-specific: it correlates with individual aggregator rankings at **mean ρ = 0.8205**, and with the best-performing aggregator per dataset at **ρ = 0.9299**. Per-aggregator stratification:

| Metric | Best-aligned aggregator(s) | ρ | Avg. across 9 aggregators |
| :-- | :-- | --: | --: |
| **SAM-LP** | Mean Pooling | 0.6078 | **0.5688 ± 0.0226** |
| | RRT-MIL | 0.5955 | |
| **SAM-Cluster** | CLAM | 0.5260 | — |

As expected, SAM-LP's linear-probing formulation aligns most closely with Mean Pooling; SAM-Cluster shows no clear architecture-specific pattern.

### Robustness to Sample Size & Top-*k* Selection

- **Sample size.** SAM-Cluster outperforms all baselines even at *n* = 5. SAM-LP approaches its maximum at *n* = 30, whereas LP needs *n* ≥ 80; SAM-LP's peak (ρ = 0.6576) exceeds LP's (ρ = 0.6167) — higher reliability with far fewer slides.
- **Top-*k*.** SAM-Cluster retains stable screening (62.5%) even within a narrow Top-2 space; SAM-LP includes the optimal FE in **100%** of datasets within the Top-5.

---

## Computational efficiency

Exhaustive selection trains every aggregator for every candidate FE over all slides:

$$
C_{\text{exhaustive}} = N \times |\Phi| \times (K \times C_{\text{FE}} + M \times C_{\text{aggregator}})
$$

Our framework uses only $n \ll N$ slides, no aggregator training, and a single SAM pass per slide:

$$
C_{\text{ours}} = n \times \big[\, |\Phi| \times (K \times C_{\text{FE}} + \alpha) + C_{\text{SAM}} \,\big]
\quad\Longrightarrow\quad \frac{C_{\text{ours}}}{C_{\text{exhaustive}}} \approx \frac{n}{N}
$$

On Camelyon16 (RTX A6000): feature extraction ≈ **343 TFLOPs/WSI**, SAM ≈ **2.98 TFLOPs** (< 1% of a single FE extraction), metric overhead $\alpha$ ≈ **3.7 MFLOPs**.

- **Single-pass prior** — SAM masks computed once per WSI, $\mathcal{O}(1)$ w.r.t. $|\Phi|$.
- **Lightweight scoring** — LP and SAM-LP fit a linear classifier on pre-extracted embeddings; no MIL aggregator is trained.
- **Subset reliability** — Cost scales only with the number of sampled WSIs.

### Per-Component Resource Tables

All measurements were taken on a single **RTX A6000** GPU using Camelyon16 (avg. *K* = 4,090 patches per WSI).

<details>
<summary><b>Feature Extractors</b> (FLOPs & runtime per WSI; peak memory at batch size 256)</summary>

| Feature Extractor | Params (M) | FLOPs / WSI | Peak Mem (GB) | Time / WSI (s) |
| :-- | --: | --: | --: | --: |
| conch_v1 | 395.2 | 277.4 TFLOPs | 9.62 | 46.41 |
| conch_v15 | 306.1 | —¹ | 6.22 | 27.20 |
| hibou_b | 85.7 | 96.4 TFLOPs | 3.93 | 14.95 |
| lunit-vits16 | 21.7 | 17.4 TFLOPs | 1.18 | 3.06 |
| musk | 675.2 | 783.5 TFLOPs | 14.62 | 47.31 |
| phikon_v2 | 303.4 | 244.2 TFLOPs | 4.28 | 32.40 |
| resnet50 | 8.5 | 13.5 TFLOPs | 2.87 | 2.22 |
| uni_v1 | 303.4 | 244.2 TFLOPs | 1.93 | 6.29 |
| uni_v2² | 681.4 | 737.8 TFLOPs | 4.51 | 17.59 |
| virchow2 | 631.2 | 673.2 TFLOPs | 3.87 | 17.61 |
| **Average** | **341.2** | **343.1 TFLOPs** | **5.30** | **21.51** |

¹ The fvcore tracer could not trace the dynamic positional-embedding interpolation in `conch_v15`; only FLOPs are omitted.
² UNI v2 was profiled with the same architecture under random initialization due to gated HuggingFace access. Weights do not affect FLOPs, memory, or runtime.

</details>

<details>
<summary><b>SAM</b> (run once per slide, SAM ViT-H on a 2048px thumbnail, <code>points_per_side=32</code>)</summary>

| Model | Params (M) | FLOPs | Peak Mem (GB) | Time / Thumbnail (s) |
| :-- | --: | --: | --: | --: |
| SAM ViT-H | 641.1 | 2.98 TFLOPs | 8.09 | 2.28 ± 0.001 |

</details>

<details>
<summary><b>MIL Aggregators</b> (one forward + backward step, feature dim 1024, K = 4,090 instances)</summary>

| Aggregator | Params (M) | FLOPs (fwd) | Peak Mem (GB) | Time / Step (ms) |
| :-- | --: | --: | --: | --: |
| Mean Pooling | 0.002 | 0.000002 GFLOPs | 0.067 | 0.45 |
| AB-MIL | 0.13 | 0.54 GFLOPs | 0.042 | 1.28 |
| DS-MIL | 1.19 | 4.84 GFLOPs | 0.103 | 2.43 |
| CLAM-MB | 0.79 | 3.22 GFLOPs | 0.102 | 2.13 |
| Trans-MIL | 2.67 | 27.26 GFLOPs | 0.690 | 26.01 |
| DTFD-MIL³ | 0.66 | 2.68 GFLOPs | 0.078 | 1.12 |
| WiKG | 1.58 | 21.46 GFLOPs | 0.408 | 11.44 |
| RRT-MIL | 4.33 | 18.26 GFLOPs | 0.263 | 7.21 |
| ILRA | 3.72 | 9.21 GFLOPs | 0.237 | 11.98 |
| **Average** | **1.67** | **9.72 GFLOPs** | **0.221** | **7.1** |

³ DTFD-MIL was approximated with a single `dimReduction → attention` pass instead of the full pseudo-bag multi-stage structure (`numGroup=4`); interpret as a lower-bound estimate of training cost.

</details>

<details>
<summary><b>SAM-LP / SAM-Cluster</b> (no backpropagation; per FE on a 30-WSI sample)</summary>

| Metric | Params | FLOPs | Peak GPU Mem | Wall-clock / FE, 30 WSI |
| :-- | :-- | :-- | :-- | --: |
| SAM-LP | – | –⁴ | <0.1 GB | 7.44 s |
| SAM-Cluster | – | –⁴ | <0.1 GB | 5.77 s |

⁴ Direct FLOP counting was not performed because these metrics are implemented mainly with CPU-based NumPy / scikit-learn operations unsuitable for fvcore tracing. The “≈3.7 MFLOPs” value in the paper is an analytical estimate of the metric overhead, not a direct measurement here.

</details>

---

## Reproducibility details

### SAM Configuration (Main 8-Dataset Pipeline)

Verified from the `sam_cluster` attributes embedded in generated `.h5` files (e.g., `features_sam1/.../normal_002.h5`) and cross-checked with `batch_sam1_trident_inject.py`.

| Item | Value | Evidence |
| :-- | :-- | :-- |
| Model | SAM ViT-H (`sam_vit_h_4b8939.pth`) | CLI default `--sam1-model-type vit_h`; `.h5` attr `sam_version_tag: SAMversion1` |
| Thumbnail size | 2048px long side, letterboxed | `.h5` attr `sam_target_side: 2048` |
| Thumbnail generation | OpenSlide level 5, 32× downsample, white background outside GeoJSON ROI | Metadata in `SAM_thumbnails/*.json` |
| `points_per_side` | 32 | CLI default; matches profiling configuration |
| `pred_iou_thresh` | 0.2 | CLI default |
| `stability_score_thresh` | 0.3 | CLI default |
| `min_mask_region_area` | 256 px | CLI default |
| Patch-to-mask mapping | Majority vote inside patch box in thumbnail space | `.h5` attr `mapping_used` |

> ⚠️ A separate ablation/visualization script, `run_sam1_on_3Dmixed_annotation_part.sh`, uses different settings (`points_per_side=16`, `pred_iou=0.1`) for the independent `3Dmixed_annotation_part` dataset only. These were **not** used for the main 8-dataset experiments (Table 1, SAM-LP, SAM-Cluster). Post-processing settings can affect the number of retained masks and overall runtime.

### Metric Hyperparameters

| Hyperparameter | Metric | Value | Notes |
| :-- | :-- | :-- | :-- |
| `τ` (centroid-similarity threshold, Eq. for $S_{inter}$) | SAM-Cluster | 0.95 | Stable over a broad range |
| `β` (attention sharpness, E-Step) | SAM-LP | 5 | Stable over a broad range |
| Subset size `n` | both | 30 WSIs (PANDA: 20× instances) | Repeated 5× (seeds 0–4) |

### Dataset Scale Statistics

Patch-level statistics computed from the extracted `.h5` files (FE: `conch_v1`). Useful for gauging slide complexity and interpreting when structure-aware metrics such as SAM-LP help most.

| Dataset | Mean Patches | Std | Median | Min | Max | P25 | P75 | P90 | P95 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| BRACS | 3617.56 | 1800.04 | 3467.0 | 110 | 7667 | 2154.0 | 5059.0 | 5955.6 | 6803.6 |
| Camelyon16 | 4312.09 | 2029.50 | 4277.0 | 131 | 10512 | 3046.0 | 5298.0 | 6708.6 | 7655.8 |
| Histai-skin-b1 | 2085.27 | 1710.72 | 1610.5 | 103 | 8268 | 808.5 | 3020.25 | 4303.5 | 5322.15 |
| PANDA | 200.60 | 76.22 | 202.0 | 12 | 702 | 155.0 | 244.25 | 290.1 | 323.0 |
| TCGA-GLIOMA | 2994.24 | 1943.67 | 2763.0 | 190 | 7612 | 1413.0 | 4441.0 | 5878.4 | 6334.4 |
| TCGA-HNSC | 3389.83 | 1640.73 | 3594.0 | 424 | 6240 | 2061.25 | 4807.75 | 5222.7 | 5565.35 |
| TCGA-NSCLC | 3834.36 | 2203.68 | 3726.0 | 241 | 9309 | 2028.5 | 5204.75 | 6854.8 | 7152.4 |
| TCGA-OV | 4301.63 | 1734.29 | 4431.0 | 275 | 7513 | 3242.25 | 5767.25 | 6155.4 | 6448.25 |
| TCGA-STAD | 3088.37 | 1541.15 | 3403.0 | 146 | 5971 | 2009.0 | 3898.0 | 5187.9 | 5461.25 |
| TCGA-RCC | 3999.75 | 1742.54 | 3912.0 | 112 | 8173 | 2832.0 | 5256.0 | 6233.2 | 6589.0 |
| UBC-OCEAN | 3175.48 | 2100.96 | 2782.0 | 9 | 9527 | 1564.0 | 4730.0 | 5662.4 | 6636.8 |

---

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) under grant number **RS-2022-NR068758**. The authors also gratefully acknowledge the generous support of the **Seegene Medical Foundation**, Republic of Korea.

We thank the developers of [Trident](https://github.com/mahmoodlab/TRIDENT), the [Segment Anything Model](https://github.com/facebookresearch/segment-anything), and the authors of the foundation models and MIL aggregators benchmarked in this study.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{kim2026samcluster,
  title     = {{SAM-Cluster} and {SAM-LP}: Structure-Aware Evaluation Metrics for
               Selecting WSI Feature Extractors without MIL Training},
  author    = {Kim, Juhyeon and Ssemakula, Paul and Song, Chanjae and Yi, Mun Yong},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026},
  publisher = {Springer},
  note      = {Oral Presentation}
}
```
<!-- TODO: update volume/pages/DOI once the MICCAI 2026 proceedings are published. -->

---

## License

Original WSI-FE-Selection code is released under [Apache-2.0](../LICENSE). Third-party code, models, datasets and the paper retain their own terms; see [third-party notes](third_party.md).

---

## Contact

For questions about the paper or code, please open an [issue](https://github.com/juhyeon-ai/WSI-FE-Selection/issues) or contact:

- **Juhyeon Kim** — `wedsed123@kaist.ac.kr`
- **Mun Yong Yi** (corresponding) — `munyi@kaist.ac.kr`