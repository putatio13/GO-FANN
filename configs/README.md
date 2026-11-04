# Workload configurations

Run from the GO-FANN root; all paths are relative to the working directory. Prepare the [input files](../docs/data-and-workflows.md) and [quantized vectors](../scripts/README.md) first.

| Dataset / workload | Search configuration | Graph build configuration |
| --- | --- | --- |
| YFCC10M categorical | `yfcc10m/categorical.json` | `yfcc10m/build.json` |
| YFCC10M Full-GPU | `yfcc10m/full_gpu.json` | `yfcc10m/build.json` |
| arxiv-large EMIS | `arxiv/emis.json` | `arxiv/build_emis.json` |
| arxiv-large range | `arxiv/range.json` | `arxiv/build_range.json` |
| LAION100M categorical | `laion100m/categorical.json` | `laion100m/build_labels.json` |
| LAION100M range | `laion100m/range.json` | `laion100m/build_range.json` |
| LAION100M exact string | `laion100m/string_exact.json` | `laion100m/build_labels.json` |
| LAION100M fuzzy string | `laion100m/string_fuzzy.json` | `laion100m/build_labels.json` |
| LAION100M hybrid | `laion100m/hybrid.json` | Both LAION build configurations |

## Build and run

Choose an idle GPU and run each command sequentially. Set `GPU=0` (or another idle physical device) in your shell. For YFCC:

```bash
mkdir -p artifacts/yfcc10m/labels outputs
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann --config=configs/yfcc10m/build.json
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann --config=configs/yfcc10m/categorical.json
```

LAION hybrid needs both label and interval graphs:

```bash
mkdir -p artifacts/laion100m/labels artifacts/laion100m/range outputs
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann --config=configs/laion100m/build_labels.json
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann --config=configs/laion100m/build_range.json
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann --config=configs/laion100m/hybrid.json
```

Other workloads use the same invocation with the configurations in the table. The builder creates the identity map and metadata beside the graph; low-frequency posting paths are derived from `graph_path`. Keep the explicit search-side identity and metadata paths paired with the graph when relocating artifacts.

Set a separate output path for each experiment. Native configurations sweep the configured queue sizes in `benchmark` mode. Full-GPU uses `search` mode; run one point at a time with `--search-queue-size=N`. See [experiment parameters](../docs/experiment-protocol.md) for workload settings and evaluation.

For a single native search, override the mode, queue size and result destination together:

```bash
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann --config=configs/yfcc10m/categorical.json \
  --mode=search --search-queue-size=128 --output-csv-path=outputs/yfcc10m_q128.csv
```

## Experiment controls

Each JSON groups input settings, execution settings, experiment controls and measurement/output settings. Keep the workload's inputs and resource settings, then edit the relevant control in place; separate variant files are unnecessary.

| Factor | Configuration controls |
| --- | --- |
| Fuzzy branch pruning | `query_manifest_max_fuzzy_terms`: 8, 16, 32, 32768; `query_manifest_fuzzy_cap_policy=coverage_cost` |
| Range access selection | `range_posting_mode=select`, `range_posting_scan_threshold`: 20000, 50000, 100000000; graph-only uses `off` and 0 |
| Range–label AND access | `range_label_and_policy`: `auto`, `label_prefer`, `range_prefer`, `dual` |
| Predicate penalty | `predicate_bias_alpha`: 0, 0.15, 0.3 |
| Rerank placement | `highfreq_rerank_mode`, `highfreq_rerank_streaming` |
| CPU/GPU execution overlap | `cpu_rerank_schedule`: `no_overlap`, `batch_overlap`, `streaming`; `native` preserves the individual streaming flags |

Use matching selected queries and ground truth when evaluating range–label AND controls. `benchmark_queue_sizes` controls the recall/throughput sweep; `benchmark_output` names its summary and per-point result files. The native configurations use `predicate_bias_alpha` for predicate-aware navigation. Leave `quick_filter_bonus=1` unchanged; it supplies the `b1_000` component of benchmark filenames and does not change scores on this path.

For CPU exact rerank, compare the execution schedules with separate outputs:

```bash
for schedule in no_overlap batch_overlap streaming; do
  CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann --config=configs/laion100m/hybrid.json \
    --cpu-rerank-schedule="$schedule" --benchmark-output="outputs/hybrid_${schedule}.csv"
done
```

`no_overlap` finishes GPU search before CPU rerank; `batch_overlap` allows completed GPU batches to rerank while later GPU work runs; `streaming` reranks completed query results as they arrive. The explicit schedule controls both graph and posting streaming. `native` retains each configuration's individual flags. CPU worker counts and required host graph services remain active in every mode.
