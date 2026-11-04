# Reproducing experiments

Run commands from the repository root. Start with the [environment and synthetic GPU checks](docs/build.md), then acquire the [vectors and prepared inputs](scripts/README.md). The synthetic checks generate their own small datasets, quantize them and exercise the runtime; they do not require LAION downloads. Use a fresh output directory for each check.

## Hardware and storage

The experiments use a Tesla V100-SXM2 with 32 GiB GPU memory, two Intel Xeon Platinum 8260 CPUs (48 physical cores total), and 256 GiB host memory. The CPU–GPU service channel requires GDRCopy and a working `gdrdrv` module. See [installation](docs/build.md) for the toolchain and driver setup.

| LAION100M component | Size |
| --- | ---: |
| FP16 base vectors | 95.7 GiB |
| Label graphs and structural metadata | 63.9 GiB |
| Interval graphs and structural metadata | 47.7 GiB |

These persistent components total about 207.3 GiB. Also budget space for quantization, predicate metadata, downloaded NPY/parquet shards and preparation scratch files. Keep original downloads and generated outputs on a volume with sufficient free space before starting graph construction. Search keeps graphs in host memory and reads FP16 vectors through mmap.

Recorded graph construction times on the experimental host are approximately 22 minutes for YFCC10M, 10–16 minutes for the arxiv workloads, 8.7 hours for LAION label graphs and 2.8 hours for LAION interval graphs. Prepared query/GT packages avoid repeating exact-neighbor computation over 100M vectors. Build time, input loading, query preparation and timed search are separate stages.

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

## Main search curves

Set `GPU` to an idle physical device and run one GPU job at a time:

```bash
GPU=0
mkdir -p outputs
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann --config=configs/laion100m/hybrid.json
```

Build the corresponding graphs first using the [configuration table](configs/README.md). For the overall-performance figure (`eval_main_results_compact.pdf`), run these configurations:

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

Use `recall_at_10`, the mean per-query Recall@10, for every workload; see [evaluation details](docs/experiment-protocol.md#evaluation). Compare throughput at similar recall, using the same inputs and timing scope. Hardware, queue size and CPU parallelism affect throughput.

## Query and execution controls

The [experiment controls](configs/README.md#experiment-controls) map each factor to its configuration keys. Use separate output filenames for every variant.

| Figure content / file | Workloads and controls |
| --- | --- |
| Query optimization, `eval_query_planning_single_column_row.pdf` | Fuzzy: `query_manifest_max_fuzzy_terms`; range: `range_posting_mode` and `range_posting_scan_threshold`; range–label AND: `range_label_and_policy` |
| Predicate penalty, `eval_penalty_compact.pdf` | `predicate_bias_alpha` on the workloads named in each panel |
| Rerank placement, `eval_rerank_placement_split.pdf` | CPU and GPU exact rerank; placement parameters and commands below |
| Execution overlap, `eval_execution_overlap_split.pdf` | `cpu_rerank_schedule=no_overlap`, `batch_overlap`, `streaming` |
| Search-window breakdown, `eval_search_window_split.pdf` | Capture PQ/RaBitQ stages, convert to `window.csv`, then plot with `--figure window` |
| Index construction and storage table | Label/range build configurations; elapsed build time and graph/structural sidecar sizes |

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

### Rerank placement

The placement CSV uses the following treatment identifiers. Both use exact
reranking and retain the workload's predicate coefficient.

| CSV variant | `highfreq_rerank_mode` | `cpu_rerank_schedule` | `highfreq_rerank_streaming` | `pq.posting_streaming` / `rabitq_posting_streaming` |
| --- | --- | --- | --- | --- |
| `cpu_batch` (CPU rerank) | `cpu` | `no_overlap` | `false` | `false` |
| `gpu_inline` (GPU inline rerank) | `gpu` | `native` | `false` | `false` |

The graph switch is `highfreq_rerank_streaming`; the posting switch is
`pq.posting_streaming` for PQ or `rabitq_posting_streaming` for RaBitQ
(CLI `--pq-posting-streaming=0` or `--rabitq-posting-streaming=0`).

`cpu_batch` runs CPU reranking after GPU search completes. CLI options override
JSON values; `no_overlap` resolves both streaming switches to false, while
`native` preserves them. Set the predicate policy explicitly to retain the
workload's coefficient during a placement comparison.

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

For arxiv EMIS, use its configuration, `--rabitq-posting-streaming=0`,
`--predicate-bias-alpha=0`, and queues `16,32,64,128,256`. For LAION categorical,
use queues `16,24,32,64,128` and alpha zero. RaBitQ inline reranking requires
`--rerank-vector-storage-mode=default`: floating-point base vectors are resident
FP32 in host memory, including when the input is FP16. Account for this memory
when switching from the supplied LAION configuration's `mmap_fp16` storage.
Evaluate each result against its matching GT in the main-results table.

RaBitQ also supports `--highfreq-rerank-mode=gpu_post` for floating-point
vectors: it reranks the traversal shortlist on the GPU and can use
`mmap_fp16` storage. This mode is separate from inline reranking and from the
Full-GPU search backend. For example:

```bash
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann \
  --config=configs/laion100m/categorical.json \
  --highfreq-rerank-mode=gpu_post --highfreq-rerank-algorithm=exact \
  --cpu-rerank-schedule=native --highfreq-rerank-streaming=0 \
  --rabitq-posting-streaming=0 --predicate-mode=gpu_exact_bias \
  --predicate-bias-alpha=0 --benchmark-queue-sizes=24 \
  --benchmark-output=outputs/categorical_gpu_post.csv
```

GPU graph reranking uses mailbox transport and coupled workers. PQ supports
uint8 inline reranking; RaBitQ supports uint8 and floating-point inline
reranking, plus floating-point GPU-post. Range-posting and dual range-graph
execution use CPU exact reranking. With explicit predicate settings, unsupported
storage or execution combinations are rejected rather than changing the
requested predicate or penalty.

### Search-window breakdown

Capture a single benchmark point using a fresh trace filename. The same command
works with PQ and RaBitQ configurations; warmups are excluded from the capture.
Install Matplotlib once in the active Python environment before plotting.

```bash
python -m pip install matplotlib
mkdir -p outputs/profile/yfcc
GO_FANN_PERFORMANCE_TRACE=outputs/profile/yfcc/stages.jsonl \
CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann \
  --config=configs/yfcc10m/categorical.json --benchmark-queue-sizes=24 \
  --benchmark-output=outputs/profile/yfcc/run.csv
python tools/reproduce/profile.py \
  --input yfcc10m outputs/profile/yfcc/stages.jsonl \
    outputs/profile/yfcc/run.b1_000.q24.f100.timed_runs.csv \
  --output outputs/profile/window.csv
python tools/reproduce/plot.py --figure window --workloads yfcc10m \
  --input-dir outputs/profile --output-dir outputs/profile/figures
```

Use a new output directory for another capture. The converter joins completed
trace iterations to their exact benchmark-point and timed-run records. It
rejects missing or duplicate iterations and inconsistent wall times. Supply
one `--input WORKLOAD TRACE TIMED_CSV` triple per workload to combine captures.
The diagnostic points in the reference figure use:

| Workload key | Search configuration | Queue |
| --- | --- | ---: |
| `yfcc10m` | `configs/yfcc10m/categorical.json` | 24 |
| `arxiv_emis` | `configs/arxiv/emis.json` | 32 |
| `arxiv_range` | `configs/arxiv/range.json` | 32 |
| `categorical_single` | `configs/laion100m/categorical.json` | 24 |
| `pure_range` | `configs/laion100m/range.json` | 40 |
| `string_exact` | `configs/laion100m/string_exact.json` | 24 |
| `string_fuzzy` | `configs/laion100m/string_fuzzy.json` | 16 |
| `hybrid` | `configs/laion100m/hybrid.json` | 40 |

Keep one queue and one finish-count/bonus combination per capture. Omitting
`--workloads` from the plot command requires all eight workloads; no missing
measurements are filled from reference data.

These are exclusive **host-observed search-window times**. Graph/posting spans
include service waits; simultaneous search and CPU reranking count once as
overlap. Copying, merging, preparation and remaining work complete the window.
The chart divides summed phase times by summed wall time across repetitions.
Collection adds overhead, so use uninstrumented runs for throughput comparisons.
The supplied `reference/window.csv` remains available to redraw the recorded
figure. Nsight is not required for this stage breakdown.

### Native graph-route phases

The default build can report RaBitQ graph-route phase timings without Nsight:

```bash
mkdir -p outputs/graph-phases
GO_FANN_RABITQ_GRAPH_PROFILE=1 CUDA_VISIBLE_DEVICES="$GPU" \
  ./src/go-fann --config=configs/arxiv/emis.json \
  --benchmark-queue-sizes=32 --warmup-runs=2 --timed-runs=1 \
  --benchmark-output=outputs/graph-phases/emis.csv \
  > outputs/graph-phases/emis.log 2>&1
```

For arxiv range, use `configs/arxiv/range.json` and separate output filenames.
Read `[rabitq graph profile]` and `[rabitq gpu phases]` between the timed-run
start/end markers, excluding warmups. The fields `io_wait_ms`,
`rabitq_score_ms`, `frontier_ms` and `other_ms` accumulate work across graph
routes. Concurrent routes contribute separately; these totals are not the
exclusive wall-clock windows above or GPU utilization.

### Nsight kernel metrics and timelines

Install Nsight Compute (`ncu`) and Nsight Systems (`nsys`) separately and make
them available on `PATH`. The command syntax below was checked with Compute
2025.1.1 and Systems 2024.6.2. NCU requires access to GPU performance counters.
Use the normal build/runtime environment, an idle GPU and a new output
directory for each capture; dataset and memory requirements are unchanged.

These API-gated commands use the **RaBitQ** backend, which starts capture at
the first timed iteration after warmups. Select one queue and one benchmark
point. PQ does not provide these profiler API markers.

For NCU, application replay repeats host execution as well as device work.
The hybrid graph kernel needs early CPU service so its host requests can be
answered while NCU intercepts the launch:

```bash
NCU_METRICS=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
mkdir -p outputs/ncu-hybrid
CUDA_VISIBLE_DEVICES="$GPU" \
GO_FANN_CUDA_PROFILER_CAPTURE_RANGE=1 GO_FANN_PROFILE_HOST_SERVICE_EARLY=1 \
  env -u GO_FANN_PERFORMANCE_TRACE -u GO_FANN_CPU_RERANK_PROFILE \
  ncu --target-processes all \
  --replay-mode application --app-replay-match grid --app-replay-mode strict \
  --profile-from-start off --kernel-name regex:native_rabitq_search_kernel \
  --launch-skip 0 --launch-count 1 --cache-control none --clock-control none \
  --metrics "$NCU_METRICS" --export outputs/ncu-hybrid/report \
  ./src/go-fann --config=configs/laion100m/hybrid.json \
  --benchmark-queue-sizes=40 --warmup-runs=2 --timed-runs=1 \
  --benchmark-output=outputs/ncu-hybrid/run.csv
ncu --import outputs/ncu-hybrid/report.ncu-rep \
  --page raw --csv > outputs/ncu-hybrid/metrics.csv
```

Use the following substitutions to capture other kernels, with distinct output
directories. Set early host service to `0` for the two posting captures.

| Configuration | Queue | Kernel-name regex |
| --- | ---: | --- |
| `configs/laion100m/string_fuzzy.json` | 16 | `lowfreq_rabitq_resident_posting_kernel` |
| `configs/laion100m/range.json` | 40 | `range_posting_rabitq_scan_kernel` |
| `configs/laion100m/hybrid.json` | 40 | `native_rabitq_search_kernel` |

`--launch-count 1` collects the first matching timed kernel; for hybrid this is
the label-graph provider, not both graph providers combined. Warmups are already
outside capture, so `--launch-skip` remains zero. Early service uses the
configuration's host-memory graph and coupled request ring. Keep file-based
host/CPU traces disabled during NCU application replay: each replay process
would otherwise target the same trace file. The metrics describe kernel
execution, SM throughput, active-warp occupancy, DRAM throughput and
long-scoreboard stalls separately. Replay repeats loading and warmups; its
application CSV is not a set of independent throughput repetitions.

For a single-run CUDA/CPU timeline with NVTX stage ranges:

```bash
mkdir -p outputs/nsys-hybrid
GO_FANN_CUDA_PROFILER_CAPTURE_RANGE=1 \
GO_FANN_PERFORMANCE_TRACE=outputs/nsys-hybrid/stages.jsonl \
CUDA_VISIBLE_DEVICES="$GPU" nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=outputs/nsys-hybrid/report \
  ./src/go-fann --config=configs/laion100m/hybrid.json \
  --benchmark-queue-sizes=40 --warmup-runs=2 --timed-runs=1 \
  --benchmark-output=outputs/nsys-hybrid/run.csv
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace,cuda_api_sum,nvtx_pushpop_trace \
  --format csv --output outputs/nsys-hybrid/summary \
  outputs/nsys-hybrid/report.nsys-rep
```

Open `report.nsys-rep` in Nsight Systems or inspect the exported CSVs. The
CUDA trace shows device intervals; NVTX ranges show host stages, including
service waits and joins. Overlapping ranges must not be summed as sequential
latency. Profiling adds overhead; use ordinary uninstrumented runs for QPS.

## Reference curves and plotting

[tools/reproduce](tools/reproduce/README.md) contains the numerical inputs for the paper's evaluation figures, including the recorded baseline curves, and one plotting entry point. [Baseline settings and sources](docs/baselines.md) describe the comparison methods and their parameter sweeps.

```bash
python tools/reproduce/plot.py --figure main --output-dir outputs/figures
```

Use the figure-specific CSV schema described there to plot new measurements with `--input-dir`. Rendering the supplied CSVs plots recorded measurements; running and evaluating the configurations above produces new measurements. The mathematical planning illustration and system diagrams are explanatory figures, not benchmark outputs.
