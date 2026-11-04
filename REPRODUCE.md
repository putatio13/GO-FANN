# Reproducing experiments

## Quick path

Prerequisites: Linux x86-64, an NVIDIA datacenter GPU supporting GPUDirect RDMA/GDRCopy, and a compatible [CUDA toolchain and driver](docs/build.md#environment). Installing/loading `gdrdrv` requires administrator privileges or assistance, with user access to `/dev/gdrdrv`. See [hardware and storage](#hardware-and-storage) for dataset-specific resources.

Run from the repository root, with one GPU job at a time on an idle device.

1. [Install the Conda environment and GDRCopy, then build](docs/build.md#environment). Installation time depends on the host and downloads.
2. **Check the runtime without downloading datasets:** after the build, run the synthetic command below. Success writes `build/gpu-smoke/VALIDATED.json`.
3. [Acquire the YFCC inputs](scripts/README.md#base-vectors-and-upstream-inputs) for the example below. For arxiv/LAION, also [download and verify prepared queries/metadata](scripts/README.md#prepared-queries-and-metadata) (about 3.35 GB total); base vectors are separate.
4. [Quantize](scripts/README.md#quantize-vectors), then [build and search YFCC](#first-dataset-run). Its recorded graph build took about 22 minutes, excluding downloads, quantization and loading.
5. Evaluate the saved IDs against matching GT with the command below; all workloads use mean per-query Recall@10.
6. [Choose a figure](#figures-and-table) to redraw recorded results or plot new measurements. Advanced capture commands live in [profiling.md](docs/profiling.md).

```bash
GPU=${GPU:-0}
python -m tests.tools.run_gpu_smoke --root=build/gpu-smoke --gpu="$GPU" --arch=sm_70
```

The examples use `sm_70` for the tested V100; match the build architecture and `--arch` to your GPU. Use a fresh smoke output directory each time. The runner generates its own small dataset, checks CUDA components and builds/searches the synthetic graph. Environment installation and full-dataset preparation are separate from this check.

## First dataset run

After acquiring the YFCC files and generating PQ artifacts, build its graph and run one benchmark point:

```bash
GPU=${GPU:-0}
mkdir -p artifacts/yfcc10m/labels outputs
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann --config=configs/yfcc10m/build.json
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann --config=configs/yfcc10m/categorical.json \
  --benchmark-queue-sizes=64
python -m scripts.evaluate \
  --results outputs/yfcc10m_categorical.b1_000.q64.f100.results.csv \
  --ground-truth data/yfcc10m/queries/categorical/gt.ibin \
  --base-rows 10000000 --k 10 \
  --timed-runs outputs/yfcc10m_categorical.b1_000.q64.f100.timed_runs.csv
```

The search performs two warmups and three timed repetitions. Evaluation reports `recall_at_10` and throughput from the timed-run file. Compare QPS at similar recall using the same inputs and timing scope; hardware, queue size and CPU parallelism affect throughput. See [evaluation details](docs/experiment-protocol.md#evaluation) and [NUMA placement](#hardware-and-storage) before throughput comparisons.

## Figures and table

Find the paper figure by its content or filename below. Render the supplied measurements with:

```bash
python -m pip install matplotlib
python tools/reproduce/plot.py --figure main --output-dir outputs/figures
```

Replace `main` with the identifier below. Each call writes `<identifier>.pdf` and `.png` into a fresh destination. To plot new measurements, write the [matching CSV schema](tools/reproduce/README.md) and add `--input-dir PATH`; plotting recorded CSVs does not run experiments. [Baseline sources/settings](docs/baselines.md) describe the recorded comparison curves.

| Paper figure / table | Plot identifier | New measurements: configuration or command | Required inputs | Time to plan for |
| --- | --- | --- | --- | --- |
| Overall performance — `eval_main_results_compact.pdf` | `main` | [Workload configurations and evaluation](#main-search-curves) | All eight workloads and matching GT | Builds: YFCC ~22 min; arxiv ~10–16 min per workload; LAION labels ~8.7 h + intervals ~2.8 h. Search adds loading and queue sweeps. |
| Query optimization — `eval_query_planning_single_column_row.pdf` | `planning` | [Controls](configs/README.md#experiment-controls); [AND subset command](#query-and-execution-controls) | LAION fuzzy, range and 15K AND subset | Reuse indexes; one sweep per cap, access policy or threshold. |
| Predicate penalty — `eval_penalty_compact.pdf` | `penalty` | `predicate_bias_alpha` in `{0, 0.15, 0.3}` via [controls](configs/README.md#experiment-controls) | YFCC, arxiv range, LAION 15K AND subset | Reuse indexes; three coefficient sweeps per workload. |
| Rerank placement — `eval_rerank_placement_split.pdf` | `placement` | [CPU/GPU commands](#rerank-placement) | YFCC, arxiv EMIS, LAION categorical | Two placement sweeps per workload; GPU inline has additional resident-memory needs. |
| Execution overlap — `eval_execution_overlap_split.pdf` | `overlap` | [Three-schedule command](configs/README.md#experiment-controls) | YFCC, LAION hybrid and fuzzy | Reuse indexes; three schedule sweeps per workload. |
| Search-window breakdown — `eval_search_window_split.pdf` | `window` | [Capture → convert → plot](docs/profiling.md#search-window-breakdown) | All eight workloads; one selected queue each | Loading plus warmups/timed capture for each workload. |
| Index construction/storage table | [index-costs.csv](tools/reproduce/reference/index-costs.csv) | [Label/range graph builders](configs/README.md); record elapsed time and graph/sidecar sizes | LAION labels/ranges and base vectors | GO-FANN builds: labels ~8.7 h; intervals ~2.8 h. |

Build times are recorded on the [experimental host](#hardware-and-storage). Download, quantization, loading and sweep durations depend on hardware and selected points; no fixed end-to-end duration is assumed. Prepared GT avoids exact-neighbor recomputation over 100M vectors. The mathematical planning illustration and system diagrams are explanatory figures, not benchmark outputs.

## Main search curves

Reuse the [first-dataset search command](#first-dataset-run) with a configuration from the table below. Set `--benchmark-queue-sizes` to the desired points, or omit it for the configuration's full sweep. Keep `GPU` set to an idle device and create `outputs/` with `mkdir -p outputs` before these experiments.

Build the corresponding graphs first using the [configuration table](configs/README.md). Use these configurations and matching ground truth for the eight panels:

| Panel | Configuration | GT under `data/` |
| --- | --- | --- |
| YFCC10M | `configs/yfcc10m/categorical.json` | `yfcc10m/queries/categorical/gt.ibin` |
| arxiv EMIS | `configs/arxiv/emis.json` | `arxiv/queries/emis/ground_truth_emis.ivecs` |
| arxiv range | `configs/arxiv/range.json` | `arxiv/queries/range/ground_truth_r.ivecs` |
| LAION categorical | `configs/laion100m/categorical.json` | `laion100m/queries/categorical/gt/gt.single.top10.ibin` |
| LAION range | `configs/laion100m/range.json` | `laion100m/queries/range/gt/gt.range.top10.ibin` |
| LAION exact string | `configs/laion100m/string_exact.json` | `laion100m/queries/string_exact/gt/gt.string.top10.ibin` |
| LAION fuzzy string | `configs/laion100m/string_fuzzy.json` | `laion100m/queries/string_fuzzy/gt/gt.string.top10.ibin` |
| LAION hybrid | `configs/laion100m/hybrid.json` | `laion100m/queries/hybrid/gt/gt.hybrid.top10.ibin` |

LAION query vectors are sampled from base rows satisfying their predicates. The query point remains eligible in exact ground truth and is included in Recall@10; the workloads use the same query/GT inputs across methods. See [query formats](docs/data-and-workflows.md#queries-and-exact-ground-truth).

Each native benchmark configuration sweeps queue sizes, performs two warmups and three timed repetitions, and writes one result-ID file and timing file per point. Evaluate a point with the matching GT:

```bash
python -m scripts.evaluate \
  --results outputs/laion100m_hybrid.b1_000.q64.f10.results.csv \
  --ground-truth data/laion100m/queries/hybrid/gt/gt.hybrid.top10.ibin \
  --base-rows 100000000 --k 10 \
  --timed-runs outputs/laion100m_hybrid.b1_000.q64.f10.timed_runs.csv
```

## Query and execution controls

The [experiment controls](configs/README.md#experiment-controls) map each factor to its configuration keys. Use separate output filenames for every variant.

Range–label AND panels use the supplied 15,000-query subset, including its own aligned GT. Create a working configuration from the hybrid configuration:

```bash
python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('configs/laion100m/hybrid.json').read_text())
root = 'data/laion100m/queries/range_label_and/'
config.update(query_vector_path=root + 'query.hvecs',
              query_manifest_jsonl_path=root + 'query_manifest.jsonl',
              query_range_path=root + 'query_range.bin',
              benchmark_output='outputs/range_label_and.csv')
Path('outputs/range_label_and.json').write_text(json.dumps(config, indent=2))
PY
for policy in auto label_prefer range_prefer dual; do
  CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann --config=outputs/range_label_and.json \
    --range-label-and-policy="$policy" \
    --benchmark-output="outputs/range_label_and_${policy}.csv"
done
```

Evaluate these points against `data/laion100m/queries/range_label_and/gt/gt.top10.ibin`. For overlap, use the [three-schedule command](configs/README.md#experiment-controls) with the full workload inputs.

## Rerank placement

The placement CSV uses the following treatment identifiers. Both use exact
reranking and retain the workload's predicate coefficient.

| CSV variant | `highfreq_rerank_mode` | `cpu_rerank_schedule` | `highfreq_rerank_streaming` | `pq.posting_streaming` / `rabitq_posting_streaming` |
| --- | --- | --- | --- | --- |
| `cpu_batch` (CPU rerank) | `cpu` | `no_overlap` | `false` | `false` |
| `gpu_inline` (GPU inline rerank) | `gpu` | `native` | `false` | `false` |

Run the YFCC CPU and GPU treatments in sequence:

```bash
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann \
  --config=configs/yfcc10m/categorical.json \
  --highfreq-rerank-mode=cpu --highfreq-rerank-algorithm=exact \
  --cpu-rerank-schedule=no_overlap --predicate-mode=gpu_exact_bias \
  --predicate-bias-alpha=0.15 --benchmark-queue-sizes=16,24,32,64,128 \
  --benchmark-output=outputs/placement_yfcc_cpu.csv
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann \
  --config=configs/yfcc10m/categorical.json \
  --highfreq-rerank-mode=gpu --highfreq-rerank-algorithm=exact \
  --cpu-rerank-schedule=native --highfreq-rerank-streaming=0 \
  --pq-posting-streaming=0 --predicate-mode=gpu_exact_bias \
  --predicate-bias-alpha=0.15 --benchmark-queue-sizes=16,24,32,64,128 \
  --benchmark-output=outputs/placement_yfcc_gpu.csv
```

For arxiv EMIS and LAION categorical, use the [workload-specific queues, predicate settings and memory requirements](docs/profiling.md#gpu-rerank-settings). That section also describes GPU-post and supported backend/storage combinations. Evaluate each treatment with its matching GT above.

## Hardware and storage

The experiments use a Tesla V100-SXM2 with 32 GiB GPU memory, two Intel Xeon Platinum 8260 CPUs (48 physical cores total), and 256 GiB host memory. The CPU–GPU service channel requires GDRCopy and a working `gdrdrv` module. See [installation](docs/build.md) for the toolchain and driver setup.

| LAION100M component | Size |
| --- | ---: |
| FP16 base vectors | 95.7 GiB |
| Label graphs and structural metadata | 63.9 GiB |
| Interval graphs and structural metadata | 47.7 GiB |

These LAION100M persistent components subtotal about 207.3 GiB, before quantization, predicate metadata, downloaded NPY/parquet shards and preparation scratch files. Other workloads have their own storage requirements. Keep original downloads and generated outputs on a volume with sufficient free space before starting graph construction. Search keeps graphs in host memory and reads FP16 vectors through mmap.

On multi-socket hosts, bind CPUs and memory to the NUMA node local to the GPU for consistent placement, especially with resident arxiv vectors. Inspect the machine topology before selecting the node:

```bash
nvidia-smi topo -m
numactl --hardware
GPU=0
NUMA_NODE=0  # Set from the topology above; GPU 0 is local to node 0 on the experimental host.
CUDA_VISIBLE_DEVICES="$GPU" numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
  ./src/go-fann --config=configs/arxiv/emis.json
```

The reference arxiv EMIS launch command did not explicitly bind CPUs or memory. Record CPU/memory binding with new throughput measurements and keep it fixed across comparisons.
