# Profiling and GPU reranking

Use these recipes after the [build](build.md) and [data preparation](../scripts/README.md) steps. Run from the repository root on an idle GPU, one job at a time; set `GPU=0` or another idle physical device. Use a fresh output directory for each capture.

- [Search-window breakdown](#search-window-breakdown): collect the paper's stage breakdown with the native tracer.
- [Native graph-route phases](#native-graph-route-phases): inspect accumulated RaBitQ route timings.
- [Nsight kernel metrics and timelines](#nsight-kernel-metrics-and-timelines): collect NCU counters or an Nsys timeline.
- [GPU rerank settings](#gpu-rerank-settings): adapt the paper's placement comparison to other workloads or use GPU-post.

## Search-window breakdown

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

## Native graph-route phases

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

## Nsight kernel metrics and timelines

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

## GPU rerank settings

The [placement comparison](../REPRODUCE.md#rerank-placement) defines `cpu_batch` and `gpu_inline` and gives the two YFCC commands. The graph switch is `highfreq_rerank_streaming`; the posting switch is `pq.posting_streaming` for PQ or `rabitq_posting_streaming` for RaBitQ (CLI `--pq-posting-streaming=0` or `--rabitq-posting-streaming=0`).

CLI options override JSON values. `no_overlap` runs CPU reranking after GPU search and resolves both streaming switches to false; `native` preserves their explicit values. Keep the workload's predicate policy and coefficient fixed during a placement comparison.

For arxiv EMIS, use its configuration, `--rabitq-posting-streaming=0`,
`--predicate-bias-alpha=0`, and queues `16,32,64,128,256`. For LAION categorical,
use queues `16,24,32,64,128` and alpha zero. RaBitQ inline reranking requires
`--rerank-vector-storage-mode=default`: floating-point base vectors are resident
FP32 in host memory, including when the input is FP16. Account for this memory
when switching from the supplied LAION configuration's `mmap_fp16` storage.
Evaluate each result against its matching GT in the [workload table](../REPRODUCE.md#main-search-curves).

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
