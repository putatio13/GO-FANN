# GO-FANN

GO-FANN is a CPU–GPU cooperative filtered approximate nearest-neighbor search system. It represents categorical, numeric and string predicates through graph and posting indexes, compiles queries into routes, and overlaps GPU search with CPU service and exact reranking.

| Dataset | Workloads | Vectors / search |
| --- | --- | --- |
| YFCC10M | Categorical | 10M × 192 uint8, PQ; additional Full-GPU configuration |
| arxiv-large | EMIS and range | 2.7M × 4096 FP32, one-bit RaBitQ |
| LAION100M | Categorical, range, exact string, fuzzy string and hybrid | 100M × 512 FP16, one-bit RaBitQ |

## Getting started

1. [Follow the reproduction guide](REPRODUCE.md): synthetic check, first dataset run and figure/table commands.
2. [Install the environment and build](docs/build.md), then [download prepared inputs and quantize vectors](scripts/README.md).
3. [Select workload configurations](configs/README.md); use [profiling recipes](docs/profiling.md) for stage and kernel measurements.

Run commands from the repository root; configuration paths are relative to the working directory. Run GPU jobs sequentially on an idle device. Reference details: [input formats](docs/data-and-workflows.md) and [experiment parameters](docs/experiment-protocol.md).

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

GO-FANN’s original contributions and inherited FlashANNS portions are covered by the [MIT License](LICENSE), with permission from the FlashANNS author team. Other third-party components retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
