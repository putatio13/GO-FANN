# Third-party notices

GO-FANN incorporates source from [Tinkerver/ann_search](https://github.com/Tinkerver/ann_search), the repository linked by [FlashANNS (arXiv v1)](https://arxiv.org/html/2507.10070v1). See [acknowledgments](docs/acknowledgments.md) for the paper citation and implementation lineage. The FlashANNS authors have authorized the use, modification and redistribution of the incorporated code under the MIT License.

| File | Retained license |
| --- | --- |
| `third_party/cuvs_rabitq/` | NVIDIA cuVS RaBitQ implementation; Apache License, Version 2.0. File-level copyright and license notices are retained; see [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). |
| `src/index/shared_resources.hpp` | Apache License, Version 2.0; the original file-level notice is preserved unchanged. The full license is included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). |

CUDA, cuVS, RAFT, RMM, GDRCopy, OpenSSL and other separately installed dependencies remain subject to their respective licenses. Dataset and pretrained/quantized artifact rights are not granted by this repository's software license; those inputs are not bundled here.
