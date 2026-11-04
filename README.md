# GO-FANN

GO-FANN is a CPU–GPU cooperative filtered approximate nearest-neighbor search system. It represents categorical, numeric and string predicates through graph and posting indexes, compiles queries into routes, and overlaps GPU search with CPU service and exact reranking.

| Dataset | Workloads | Vectors / search |
| --- | --- | --- |
| YFCC10M | Categorical | 10M × 192 uint8, PQ; additional Full-GPU configuration |
| arxiv-large | EMIS and range | 2.7M × 4096 FP32, one-bit RaBitQ |
| LAION100M | Categorical, range, exact string, fuzzy string and hybrid | 100M × 512 FP16, one-bit RaBitQ |

## Getting started

1. [Create the Conda environment, build and run a synthetic example](docs/build.md) for the runtime and quantizers.
2. [Prepare data and quantize vectors](scripts/README.md).
3. [Choose a configuration, build graphs and run search](configs/README.md).

Run commands from the repository root; configuration paths are relative to the working directory. Run GPU jobs sequentially on an idle device. The [reproduction guide](REPRODUCE.md) maps experiments to commands and reference curves. Reference details: [input formats](docs/data-and-workflows.md) and [experiment parameters](docs/experiment-protocol.md).

## Layout

| Directory | Purpose |
| --- | --- |
| `src/`, `include/` | Native graph builder and PQ, RaBitQ and Full-GPU search |
| `configs/` | Dataset and workload configurations |
| `scripts/` | Data preparation, predicate indexes, exact ground truth and quantization CLI |
| `tools/quantization/`, `third_party/` | Native quantizers and their dependencies |
| `tests/` | Python, C++ and CUDA tests; development checks in `tests/tools/` |
| `docs/` | Build instructions, input formats, experiment parameters and attribution |

## Credit and license

GO-FANN builds on the FlashANNS implementation; see [acknowledgments](docs/acknowledgments.md) for source lineage and provenance.

Original GO-FANN contributions are covered by the scoped [MIT License](LICENSE). It does not license inherited FlashANNS code; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and third-party terms.
