# License and third-party software

Original WSI-FE-Selection code is licensed under [Apache-2.0](../LICENSE), permitting research and commercial use subject to its terms. This license does not replace licenses attached to third-party code, datasets, pretrained weights or the paper.

## Optional preprocessing stack

| Component | Source | Terms |
| --- | --- | --- |
| TRIDENT | [Mahmood Lab](https://github.com/mahmoodlab/TRIDENT) | The pinned checkout declares **CC-BY-NC-ND 4.0** and limits use to non-commercial academic research; commercial use requires permission from its authors. See [TRIDENT's terms](https://github.com/mahmoodlab/TRIDENT#license-and-terms-of-use). |
| Segment Anything (SAM 1) | [Meta](https://github.com/facebookresearch/segment-anything) | See the upstream [Apache-2.0 license](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE) and checkpoint terms. |
| Patch feature extractors | Model repositories selected through TRIDENT | Each encoder and its weights have separate terms, including possible gated access. |

These repositories are cloned into the Git-ignored `third_party/` directory by the setup script. Their source and weights are not redistributed as part of this repository. Using Apache-2.0 for WSI-FE-Selection does **not** make the full TRIDENT-based workflow available for unrestricted commercial use.

The scoring entry point accepts existing HDF5 features and does not import TRIDENT, SAM or PyTorch. For commercial use, supply features from a pipeline and models for which you have the appropriate rights.

## Research benchmark code and data

`Benchmark-MIL/` contains the historical downstream benchmark implementations. Preserve existing attributions and consult the original model implementations before redistribution or commercial reuse; this repository's license does not relicense upstream material. The benchmark code is separate from the supported suitability-scoring quickstart.

`Ground Truth/` and `Benchmark-MIL/dataset/data_split/` contain research results and split metadata, not WSI image datasets. Obtain the source datasets under their original access conditions. The camera-ready manuscript is provided for reading and citation; the software license does not relicense the paper.
