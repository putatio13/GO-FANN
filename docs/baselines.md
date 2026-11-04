# Baseline implementations and measured settings

The main-results CSV contains GO-FANN and 14 comparison labels, including
GO-FANN Full-GPU. The tables below describe the implementations and parameter
sweeps behind those measurements. This repository contains GO-FANN; external
implementations are available from the sources linked below. The adaptation
column identifies the benchmark-specific changes used for the recorded curves.
GO-FANN Full-GPU is runnable with the included configuration.

## Implementations

The implementations use the upstream base revisions below with the listed
benchmark adaptations.

| Figure label | Upstream base revision / version | Adaptation used by the benchmark |
|---|---|---|
| GO-FANN Full-GPU | This repository; [configuration](../configs/yfcc10m/full_gpu.json) | Resident YFCC vectors and graph, exact GPU distances |
| VecFlow | [VecFlow](https://github.com/Supercomputing-System-AI-Lab/VecFlow), base `8f7837aca26e2ee579ca5215b81aa783975125cc` | Multi-label greedy API and filtering extensions |
| ACORN-1 | [ACORN](https://github.com/TAG-Research/ACORN), base `c259f11cdbec880671658eaa0d8747e2cc9de79d` | SIEVE's `AcornIndexUint8`/`AcornIndexFloat` bindings, including predicate/range support and search-step control |
| ACORN-γ | Same ACORN base and bindings as ACORN-1 | Workload-specific `gamma` and `m_beta`, listed below |
| SIEVE | [SIEVE](https://github.com/BillyZhaohengLi/SIEVE-vldb25), base `9f128f4916ca2fc5cf839a292359050be85eab0f` | Partitioned-index bindings; also supplies the ACORN bindings above |
| ELI | [LabelANN](https://github.com/mingyu-hkustgz/LabelANN), base `a3372a65e5cb57819472ba039d0e45073df46fd4` | Elastic HNSW; containment labels converted to 1-based IDs |
| BigFANN | [BigFANN](https://github.com/zhouzixiang2004/BigFANN), base `ec36a43c993489fbfd8baa8c71412c1317e7e7da` | Typed vector readers, saved-index query driver, FP16 LAION storage and memory-management changes |
| CGIF | [CGIF](https://github.com/rutgers-db/CGIF), base `aa525c18b11026414365818ffaa90af113d50053` | Filter adapters; raw FP32 arxiv reader; FP16 LAION storage and general-predicate execution |
| UNG | Base `92b58ffd600d3a51bb6537460dae1079150b5fa5` | Dataset-specific label widths, frequency remapping and large-vector support |
| SeRF | [SeRF](https://github.com/rutgers-db/SeRF), base `4075c03e2d04f1398c556e9324f15d0473db984b` | Serial construction helper and saved-index range-query wrapper |
| iRangeGraph | [iRangeGraph](https://github.com/sklkdd/iRangeGraph), base `dcfa0506dabab0fe77fd885eb2f294933d5ea8f1` | Attribute-order permutation, range conversion and threaded query wrapper |
| Faiss IVF-PQ | ACORN's Faiss fork, base `c259f11cdbec880671658eaa0d8747e2cc9de79d` | CPU IVF-PQ followed by predicate filtering; not the stock PyPI Faiss package |
| Milvus HNSW | `milvusdb/milvus:v2.6.20`; client `pymilvus==2.6.17` | Dataset import, scalar-filter translation and concurrent request driver |
| Milvus IVF-PQ | Same Milvus server/client versions | IVF-PQ collection with the same predicate translation |

## Index and search settings

The table reports the execution settings of the selected curves. Query
parallelism and search breadth were varied by workload. `T` means query
workers; Milvus request concurrency is listed separately. QPS measures total
query throughput, without per-core normalization.
All evaluate top-10, using 100K YFCC queries, 10K queries per arxiv or non-hybrid LAION workload, and 50K
LAION hybrid queries. Sweeps include settings below the main figure's 0.6 recall
cutoff; the reference CSV contains the plotted subset. Listed search ranges
vary traversal breadth or candidate budget to measure the recall–throughput
trade-off.

| Method | Index / fixed settings | Search sweep and execution |
|---|---|---|
| GO-FANN Full-GPU | YFCC degree 32, adaptive stopping | Queue sizes `{16,32,48,64,96,128,192,256,384,512}`; three timed runs |
| VecFlow | YFCC degree 16, specificity threshold 2000, multi-label greedy, `use_legacy_filter=true` | iTopK `{8,16,32,64,128,256,512}`; automatic batch size through 256, batch 1024 at 512; 10 warmups and 100 timed runs on V100 |
| ACORN-1 | `M=32,gamma=1,m_beta=32,max_steps=0`; arxiv EMIS/range `efConstruction=32` | YFCC: `ef={10,60,110,160,210,260,310,360,410,460,510}`, 1T. EMIS: `{10,15,20,25,30,50,100,250,500,750,1000}`, 1T. arxiv range: `{50,100,250,500,750,1000,1500,2000}`, 64T |
| ACORN-γ | `(M,gamma,m_beta)` = `(32,80,64)` YFCC, `(48,15,48)` EMIS, `(32,12,16)` arxiv range; `efConstruction={2560,720,384}` respectively; `max_steps=0` | YFCC: same `ef` set as ACORN-1, 48T. EMIS plotted `ef={100,250,500,750,1000,1500,2000}`, 48T. arxiv range: same `ef` set as ACORN-1, 24T |
| SIEVE | `M=16` YFCC, `M=32` arxiv; arxiv `efConstruction=40`; EMIS index budget 2, range budget 0 | YFCC: `ef=10..110` in steps of 10, 24T. EMIS: `{150,200,300,500,750,1000,1500,2000,3000,5000}`, 1T. arxiv range: `10..110` in steps of 10 plus `{128,160,200,256,512}`, 48T |
| ELI | arxiv EMIS elastic factor 0.2, containment search | `ef={10,16,32,64,128,256,275,300,400}`, 32T |
| BigFANN | YFCC uint8, 5 IVF probes; arxiv FP32; LAION categorical FP16, graph degree 12/shared degree 1 | Beam: YFCC `{10,20,40,80,120,240,500,1000}`, 96T; EMIS `{10,20,40,80,120,320,1000,2000}`, 48T; LAION `{10,20,40,80,120,240,500,1000,2000}`, 48T. One warmup, median of three timed passes |
| CGIF | LAION 4096 lists and FP16 vectors; dataset-specific filter adapters | YFCC `ef={10,40,120,320,1000}`, 1T; arxiv `{10,40,120,320}`, 1T; all LAION workloads `{10,20,30,40,60,80,120,160,240,320}`, 48T |
| UNG | `R=32,Lbuild=100,alpha=1.2`, cross-edge count 6, 16 entry points | YFCC and EMIS `Lsearch={10,50,100,300,500,1000,1200,3000}`, 24T |
| SeRF | arxiv range: index K 100, construction `ef=100`, maximum `ef=600`; serial index construction | Search `ef={8,16,32,64,128,256,512,1024}`, 96T |
| iRangeGraph | arxiv range: `M=24,efConstruction=100` | Search `ef={1,2,3,4,5,6,8,10,15,20,30,50,100,200}`, 64T |
| Faiss IVF-PQ | LAION: 16384 lists, 64 PQ subquantizers, 8 bits, 1M training rows | `nprobe={16,64,256}` × candidate count `{100,300,1000,3000}`, plus `(256,10000)` and `(256,30000)`; 24T. The candidate count is before filtering; output remains top-10 |
| Milvus HNSW | `M=32,efConstruction=200`, standard filtering; YFCC centered int8 vectors, LAION FP16 | YFCC `ef={10,12,16,24,32,64,128,256,512}`, 24 concurrent requests. LAION `{10,12,16,24,32,64}`, 512 requests |
| Milvus IVF-PQ | LAION: 16384 lists, 64 PQ subquantizers, 8 bits, standard filtering, FP16 vectors | `nprobe={1,4,16,64,256}`; concurrent requests: categorical/range 512, exact string 384, fuzzy string/hybrid 256 |

For YFCC, VecFlow's `use_legacy_filter=true` selects its single-residual-label filter: the least-frequent query label selects the graph/list, and the remaining label is checked during search. The 100K queries contain one or two labels, so this path represents their full conjunction; it does not disable filtering.

LAION string and hybrid adapters evaluate the released token vocabulary and
predicates, preserving vector IDs. Raw-caption substring matching is not an
interchangeable input. Milvus uses token-based padded-caption expressions and
an NGRAM scalar index for these predicates. Faiss applies predicates to its
unfiltered candidate list, so both `nprobe` and candidate count affect recall.

For a new GO-FANN Full-GPU curve, build the YFCC graph and acquire its inputs as
in [the experiment instructions](../REPRODUCE.md), then run one search per queue.
The Full-GPU configuration runs a single point; it does not use the PQ/RaBitQ
benchmark sweep:

```bash
GPU=0
mkdir -p outputs/full_gpu
for queue in 16 32 48 64 96 128 192 256 384 512; do
  CUDA_VISIBLE_DEVICES="$GPU" ./src/go-fann \
    --config=configs/yfcc10m/full_gpu.json --search-queue-size="$queue" \
    --output-csv-path="outputs/full_gpu/q${queue}.results.csv" \
    > "outputs/full_gpu/q${queue}.log" 2>&1
done
python -m scripts.evaluate --results outputs/full_gpu/q192.results.csv \
  --ground-truth data/yfcc10m/queries/categorical/gt.ibin \
  --base-rows 10000000 --k 10
```

Use `recall_at_10`, as for the other workloads. Full-GPU writes result IDs and prints the
average timed kernel duration to its log; it does not produce the PQ/RaBitQ
`timed_runs.csv`. For original-query throughput, divide the query count by
`FullGPU kernel time avg` in seconds; the separately printed effective route QPS
counts routed searches and is not interchangeable with original-query QPS.

To compare independently obtained baseline measurements, write `dataset,method,recall_at_10,qps` rows using the panel and
method names in [main.csv](../tools/reproduce/reference/main.csv), then use
[the plotter](../tools/reproduce/README.md). The source identities and settings
above document the recorded comparison, including the adaptations and
workload-specific settings needed when comparing external implementations.
