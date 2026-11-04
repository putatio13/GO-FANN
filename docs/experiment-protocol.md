# Experiment parameters

Run workload configurations from the repository root after preparing matching data, quantization and graph artifacts. Use one GPU search job at a time. Native benchmark configurations run two warmups and three timed repetitions for each queue size.

| Configuration under `configs/` | Queries | Queue sizes | CPU rerank workers | Predicate penalty |
| --- | --- | --- | --- | --- |
| `yfcc10m/categorical.json` | 100,000 | 16, 32, 48, 64, 96, 128, 192, 256, 384, 512 | 16 | 0.15 |
| `arxiv/emis.json` | 10,000 | 16, 32, 64, 128, 256, 512 | 32 | 0 |
| `arxiv/range.json` | 10,000 | 16, 32, 64, 128, 192, 256, 384, 512 | 16 | 0 |
| `laion100m/categorical.json` | 10,000 | 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 | 32 | 0 |
| `laion100m/string_exact.json` | 10,000 | 12, 14, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 | 32 | 0 |
| `laion100m/string_fuzzy.json` | 10,000 | 11, 12, 14, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 | 32 | 0 |
| `laion100m/range.json` | 10,000 | 16, 18, 20, 22, 24, 32, 40, 48, 64, 96, 128, 192, 256, 384, 512 | 32 | 0.15 |
| `laion100m/hybrid.json` | 50,000 | 32, 40, 48, 64, 96, 128, 192, 256, 384, 512 | 32 | 0.15 |

All workloads use graph degree 32 and top-10 results (`topk=10`). `search_finish_count` is a search stopping parameter, independent of the number of returned neighbors. Main configurations keep adjacency in host memory, use eight CPU service threads and streaming CPU exact rerank, and set resident graph `prefetch_hops=3`. Label graph/posting thresholds are 2,000 for YFCC/arxiv and 10,000 for LAION. See [configuration selection](../configs/README.md) for graph pairings and launch commands.

## Workload settings

- YFCC categorical sets `and_dual_route_posting_ratio_threshold=1`. This threshold controls whether an AND label pair uses both routes; changing it changes search work and recall.
- Pure-range configurations load interval graphs only; LAION hybrid loads both label and interval graphs with automatic route selection. LAION range/hybrid use an ordered-posting selection threshold of 20,000; arxiv range remains graph-based (`range_posting_mode=off`).
- LAION configurations use `host_reply_wait_policy=adaptive`. Categorical search sets `rabitq_posting_streaming=false`.
- LAION exact and range use `rabitq_posting_schedule=graph_first`; fuzzy uses `posting_first`.
- LAION range disables low-frequency route merging and low-frequency RaBitQ rerank, sets `lowfreq_full_filter_mode=off`, and uses `range_posting_shortlist=100`. Hybrid also disables low-frequency route merging.
- arxiv uses resident FP32 originals for exact reranking. LAION uses mmap-backed FP16 originals. One-bit RaBitQ approximate scoring uses its INT8 scoring path independently of exact-rerank vector representation.
- Standalone fuzzy search caps expansion at 8 branches; hybrid uses 16, with `coverage_cost` pruning. Exact ground truth uses uncapped matching-token unions.

Native PQ/RaBitQ configurations automatically select GPU CSR residual checks; `predicate_bias_alpha` sets the penalty listed above. The separate Full-GPU backend uses its own predicate interface. Manifest family/subtype selection and string expansion settings are explicit in each workload JSON. See [experiment controls](../configs/README.md#experiment-controls) for the supported ablation settings.

Use the root `predicate_*` keys shown in the examples; they take precedence over backend-specific predicate aliases. CLI options override JSON. A nested `pq.host_reply_wait_policy` overrides the root wait policy for both native backends.

## Evaluation

Compare returned IDs against exact ground truth for the same selected query rows. Preserve query order across vectors, manifests, ranges and GT. From the repository root, evaluate a saved benchmark point on the CPU:

```bash
python3 -m scripts.evaluate \
  --results outputs/laion100m_hybrid.b1_000.q64.f10.results.csv \
  --ground-truth data/laion100m/queries/hybrid/gt/gt.hybrid.top10.ibin \
  --base-rows 100000000 --k 10 \
  --timed-runs outputs/laion100m_hybrid.b1_000.q64.f10.timed_runs.csv
```

Use the filenames produced by your run. The evaluator accepts headerless integer prediction CSVs, `.ibin` GT (IDs only or IDs followed by distances), and variable-length `.ivecs` GT. It validates every ID and requires matching query counts, then scores the first `k` positions. `-1` and `4294967295` denote missing IDs; duplicate IDs and malformed rows are errors. Optional `--base-rows` also checks dataset bounds.

Use **Recall@10 averaged over queries** for every workload: for each query, divide the number of correct returned IDs by the number of valid exact top-10 neighbors, then average over nonempty GT rows. The primary JSON field is `recall_at_10` (or `recall_at_<k>` with another `--k`). This is macro recall; it equals total-hit aggregation when all GT rows contain 10 valid neighbors, as in the reference results.

For custom inputs with fewer valid neighbors, use the actual GT row length as the denominator. Empty GT rows are excluded and counted in `zero_gt_queries`; recall is `null` if all rows are empty. Prediction rows must contain at least `k` slots.

Omit `--timed-runs` for recall only. With a matching native per-point timing file, `timing` reports the timed QPS values, arithmetic mean and sample standard deviation (`null` for one run), excluding warmups. Its result paths must match the evaluated file; relative paths resolve from the working directory. The saved IDs are the final result for that point, while the timing file records its repetitions.

`yfcc10m/full_gpu.json` places graph and vector payloads on the GPU. Native CPU-service and CPU-rerank parameters do not apply to this backend.

The [data inventory](data-artifacts.json) records archive and extracted-file hashes for all supplied LAION query/GT and metadata packages, including the hybrid workload, and arxiv metadata. After extraction, verify the supplied files:

```bash
python3 -m tests.tools.verify_paper_inputs --root=.
```

This command reads complete files with bounded-memory hashing and reports missing or mismatching required inputs. To verify the downloaded archive parts and `SHA256SUMS` instead, use `--root=downloads --archives`. Base vectors and separately downloaded upstream YFCC/arxiv queries and GT are not members of these packages.
