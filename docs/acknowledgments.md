# Source lineage and acknowledgments

GO-FANN incorporates source from [Tinkerver/ann_search](https://github.com/Tinkerver/ann_search), the repository linked in the artifact-availability statement of [FlashANNS (arXiv v1)](https://arxiv.org/html/2507.10070v1). FlashANNS is described in Yang Xiao, Mo Sun, Ziyu Song, Bing Tian, Jie Zhang, Jie Sun and Zeke Wang, [*Breaking the Storage-Compute Bottleneck in Billion-Scale ANNS: A GPU-Driven Asynchronous I/O Framework*](https://arxiv.org/abs/2507.10070v1), arXiv:2507.10070v1.

The label graph/posting design also draws on VecFlow.

GO-FANN uses PQ, RaBitQ, CAGRA, cuVS, RAFT, RMM and GDRCopy. The RaBitQ artifact generator includes the cuVS implementation files listed in [third-party notices](../THIRD_PARTY_NOTICES.md), with their original NVIDIA copyright and Apache-2.0 notices retained.

We thank the FlashANNS author team for permission to use, modify and publicly redistribute the inherited source under MIT. The [MIT License](../LICENSE) covers these portions and original GO-FANN contributions; original attribution and copyright notices are retained. Other third-party files retain their own terms; see [third-party notices](../THIRD_PARTY_NOTICES.md), including the Apache-2.0 notices for `src/index/shared_resources.hpp` and the vendored RaBitQ implementation.

## Implementation map

| GO-FANN mechanism | Implementation |
| --- | --- |
| Heterogeneous predicates and residual checks | `src/predicates/`, `src/planning/`, and manifest loading in `src/data/` |
| Graph/posting access selection and route preparation | `src/planning/` and range indexes in `src/index/` |
| CPU–GPU service, reranking and overlap | `src/execution/`, integrated with the PQ and RaBitQ backends |
| Vector quantization and filtered data preparation | `tools/quantization/` and `scripts/` |

These modules integrate GO-FANN's mechanisms with the inherited ANN implementation and the libraries credited above. See [the source layout](../src/README.md) for their responsibilities and [the experiment controls](../configs/README.md#experiment-controls) for the evaluated settings.
