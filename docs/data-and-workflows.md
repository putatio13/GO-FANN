# Input formats

## Vectors and predicates

| Format | Layout |
| --- | --- |
| `.ibin`, `.u8bin`, `.bin` vector input | Little-endian uint32 row count and dimension, then row-major uint8 values |
| `.fbin` | The same header, then row-major FP32 values |
| `.bvecs`, `.fvecs`, `.hvecs` | Repeated uint32 dimension followed by one uint8, FP32 or FP16 row |
| `.spmat` | Sparse CSR labels, as read and written by the preparation tools |

YFCC uses uint8 vectors, arxiv FP32, and LAION FP16. Vectors, labels, graphs, quantized artifacts and predicate indexes must use the same base row IDs and row limit.

LAION100M is the first 100M rows of the released 512-dimensional image embeddings, concatenating NPY shards by numeric shard ID; see the [download and conversion commands](../scripts/README.md#laion100m-vectors). A single concatenated NPY must preserve that order. The FP16 export does not normalize or reorder vectors. LAION preparation also consumes row maps and token maps. Row maps associate `row_id` with `shard_row_id` and `metadata_shard_id` (or `shard_id`). Token maps contain contiguous, zero-based `token_id` values and `token` text; ngram indexes use the full token map.

`scripts.laion.metadata` produces those maps and `base.spmat` from matching parquet/image shards. Its `base_ranges.fbin` has three FP32 columns: width, height and similarity. Pass columns `0,1` to `range_index hierarchy` for the width/height workloads. Preserve the generated vocabulary together with its labels and query token IDs.

## Graph and quantization artifacts

Graph construction writes `graph.bin`, `graph.identity.groups`, `graph.lowfreq.bin`, `graph.seed`, `graph.orig_label.seed` and `graph.metadata`. Keep these sidecars together. Search loads the serialized graph into host memory; its header identifies the format independently of the filename suffix.

PQ uses `pq.pivots_path` for the pivot/codebook container and `pq.compressed_path` for row-ordered uint8 codes. RaBitQ uses `rabitq_artifact_path` for the native sample-major artifact. The [quantization commands](../scripts/README.md#2-quantize-vectors) produce these formats directly.

## Queries and exact ground truth

| Input | Configuration field |
| --- | --- |
| Query vectors | `query_vector_path` |
| Categorical query labels | `query_label_spmat_path` |
| JSONL predicate manifest | `query_manifest_jsonl_path` |
| String/operator index | `query_manifest_operator_index_path` |
| Base numeric values | `base_range_fbin_path` |
| Query numeric bounds | `query_range_path` |
| Interval-label metadata | `range_label_manifest_path` |

Selected manifest rows, query vectors, labels, range rows and exact GT must preserve identical query order. Each selected manifest row identifies its source vector with `query_vector_id`. LAION query vectors are sampled from base rows that satisfy the query predicate, rather than from a held-out vector set. Exact GT does not exclude that base row, so a self-match can contribute to Recall@10. Identical vectors may tie at zero distance; the GT preserves the supplied neighbor ordering. LAION range inputs use width and height; arxiv uses its corresponding numeric values and interval metadata. `range_index hierarchy` writes `data/laion100m/range/{base_range.fbin,base.virtual.spmat,manifest.json}`. Its selected column positions become query predicate column IDs; preserve that mapping when exporting a source matrix with extra columns.

`query` writes `query_manifest.jsonl`, `query.hvecs` and `query_range.bin` under the selected workload directory. Token postings (`artifacts/laion100m/token_postings.bin`) map token IDs to base rows for exact GT; the operator index (`data/laion100m/token_operator_index.bin`) stores normalized token/ngram information for predicate expansion. They are different formats.

Pass metadata outputs directly to `hybrid_queries` with `--base-spmat PATH --base-range-fbin PATH --scales 100M`. Use the three-column source ranges when the input predicates refer to width, height and similarity. Atomic/string roots contain `query_templates.jsonl` or `query_manifest.jsonl`, either directly or under the scale directory. The generator writes its outputs under `<out-root>/<scale>/`; it preserves base row order. Existing scale-directory layouts remain accepted.

Exact GT contains neighbors and distances for the selected predicates. Fuzzy GT evaluates the uncapped union of matching tokens; approximate-search expansion caps do not apply to GT. The [data inventory](data-artifacts.json) records file identities and public destinations for the supplied query/GT and metadata packages.
