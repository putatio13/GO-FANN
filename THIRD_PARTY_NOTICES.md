# Third-party notices

The [MIT License](LICENSE) covers only original GO-FANN contributions. It does not grant rights in inherited third-party code, including inherited portions of modified files. Existing copyright and license notices are retained.

GO-FANN incorporates source derived from [Tinkerver/ann_search](https://github.com/Tinkerver/ann_search), the repository linked by [FlashANNS (arXiv v1)](https://arxiv.org/html/2507.10070v1). Its upstream repository provides no repository-wide license grant. Rights in that source remain with its copyright holders. Licensing and redistribution of the inherited portions are subject to permission from those holders, separately from the MIT license for GO-FANN’s original contributions. The paper citation and implementation lineage are given in [acknowledgments](docs/acknowledgments.md).

| File | Retained license |
| --- | --- |
| `third_party/cuvs_rabitq/` | NVIDIA cuVS RaBitQ implementation; Apache License, Version 2.0. File-level copyright and license notices are retained; see [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). |
| `src/index/shared_resources.hpp` | Apache License, Version 2.0; the original file-level notice is preserved unchanged. The full license is included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). |

The source lineage from FlashANNS and the paper's relationship to VecFlow are documented in [docs/acknowledgments.md](docs/acknowledgments.md). The MIT license does not replace a retained third-party file-level license.

CUDA, cuVS, RAFT, RMM, GDRCopy, OpenSSL and other separately installed dependencies remain subject to their respective licenses. Dataset and pretrained/quantized artifact rights are not granted by this repository's software license; those inputs are not bundled here.
