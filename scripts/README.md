# Preparation tools

Run from the repository root. Create the [Conda environment](../docs/build.md) for vector and metadata preparation. For GPU ground-truth generation, also install the PyTorch version shown there. `requirements.txt` and `requirements-data.txt` list dependencies for existing environments. See [input formats](../docs/data-and-workflows.md) for file layouts and row-order requirements.

## 1. Use prepared inputs (recommended)

The packages supply LAION queries/GT and arxiv/LAION predicate metadata. YFCC inputs and arxiv query vectors/GT come from [upstream](#base-vectors-and-upstream-inputs). **Base vectors, graph indexes and quantized vectors are acquired or built separately.** Follow the sections in order; optional regeneration is in [Part 2](#2-generate-inputs-optional).

### Prepared queries and metadata

Download the data attachments from the [GO-FANN data release](https://github.com/putatio13/GO-FANN/releases/tag/paper-data). They extract to the paths used by the configurations:

| Package | Contents |
| --- | --- |
| `go-fann-arxiv-metadata.tar.gz` | arxiv CSR labels, numeric attributes and query bounds |
| `go-fann-laion100m-queries.tar.gz` | All five LAION query sets and exact GT; the 15K range–label AND subset for ablations |
| `go-fann-laion100m-metadata.tar.gz.part*` | Base labels, numeric attributes, fixed token vocabulary and string operator index |
| `go-fann-laion100m-range-indexes.tar.gz` | Interval labels and ordered range postings |

Using the GitHub CLI, download and verify the attachments, then extract from the repository root:

```bash
mkdir -p downloads
gh release download paper-data --repo putatio13/GO-FANN --dir downloads \
  --pattern 'go-fann-*.tar.gz*' --pattern SHA256SUMS
python -m tests.tools.verify_paper_inputs --root downloads --archives
tar -xzf downloads/go-fann-arxiv-metadata.tar.gz
tar -xzf downloads/go-fann-laion100m-queries.tar.gz
cat downloads/go-fann-laion100m-metadata.tar.gz.part* | tar -xz
tar -xzf downloads/go-fann-laion100m-range-indexes.tar.gz
python -m tests.tools.verify_paper_inputs --root .
```

Extract into a fresh data directory to preserve any existing inputs. Each package carries payload checksums; [the inventory](../docs/data-artifacts.json) also records archive and file identities. Query vectors, predicates and GT are supplied together in matching row order. For the standard experiments, proceed to vector acquisition, quantization and graph construction; no GT recomputation is needed.

### Base vectors and upstream inputs

Use the [BigANN YFCC dataset definitions](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py) and [arxiv-large dataset](https://huggingface.co/datasets/SPCL/arxiv-for-fanns-large). Place or link the downloaded files as follows:

| Downloaded file | Local destination |
| --- | --- |
| YFCC `base.10M.u8bin` | `data/yfcc10m/base.u8bin` |
| YFCC `base.metadata.10M.spmat` | `data/yfcc10m/base.spmat` |
| YFCC `query.public.100K.u8bin` | `data/yfcc10m/queries/categorical/query.u8bin` |
| YFCC `query.metadata.public.100K.spmat` | `data/yfcc10m/queries/categorical/query.spmat` |
| YFCC `GT.public.ibin` | `data/yfcc10m/queries/categorical/gt.ibin` |
| arxiv `database_vectors.fvecs` | `data/arxiv/base.fvecs` |
| arxiv `query_vectors.fvecs` | `data/arxiv/queries/{emis,range}/query.fvecs` |
| arxiv `ground_truth_emis.ivecs`, `ground_truth_r.ivecs` | Corresponding workload directory, retaining `.ivecs` |

YFCC files are hosted under the [YFCC benchmark directory](https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/). Use the upstream arxiv vectors and GT listed above, preserving their row order.

#### LAION100M vectors

LAION100M uses the first 100,000,000 **image embeddings** from [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/), in numeric NPY shard order and original row order within each shard. Obtain the vectors and matching metadata from [Academic Torrents](https://academictorrents.com/details/34b94abbcefef5a240358b9acd7920c8b675aacc) using its [.torrent file](https://academictorrents.com/download/34b94abbcefef5a240358b9acd7920c8b675aacc.torrent). The torrent info hash is `34b94abbcefef5a240358b9acd7920c8b675aacc`.

In your BitTorrent client's file selection, select image shards `img_emb_0.npy` through `img_emb_100.npy` and the matching `metadata_0.parquet` through `metadata_100.parquet`. These are under `laion400m-met-release/laion400m-embeddings/{images,metadata}/`. Use numeric suffixes when selecting files; the file list is lexicographically ordered. After the client completes its piece verification, set `LAION_SOURCE` to the downloaded `laion400m-embeddings` directory:

```bash
LAION_SOURCE=/path/to/laion400m-met-release/laion400m-embeddings
python3 -m scripts.laion.base export-vectors \
  --input "$LAION_SOURCE/images" --output data/laion100m/base.hvecs \
  --rows 100000000 --dim 512
```

Shard sizes vary: this prefix uses all rows of shards 0–99 and 312,973 rows of shard 100. The exporter reads each NPY shape, writes FP16 `.hvecs` in chunks, and preserves values without normalization. Output size is 102,800,000,000 bytes (about 95.7 GiB). If the complete embeddings are already concatenated in the same order into one 2D NPY, pass that file to `--input` instead. Existing output files are protected; `--chunk-rows` controls conversion memory.

### Quantize vectors

[Build](../docs/build.md) the runtime and quantizers first. Select an idle GPU and use a new output directory; existing directories are protected. These commands produce the paths already used by the supplied configurations:

```bash
CUDA_VISIBLE_DEVICES="${GPU:-0}" python3 -m scripts.quantize pq \
  --input data/yfcc10m/base.u8bin --output-dir artifacts/yfcc10m/pq \
  --pq-dim 64 --training-rows 65536 --chunk-rows 8192 --seed 137
CUDA_VISIBLE_DEVICES="${GPU:-0}" python3 -m scripts.quantize rabitq \
  --input data/arxiv/base.fvecs --output-dir artifacts/arxiv/rabitq \
  --n-lists 1024 --bits 1 --training-rows 262144 --chunk-rows 8192 --seed 137
CUDA_VISIBLE_DEVICES="${GPU:-0}" python3 -m scripts.quantize rabitq \
  --input data/laion100m/base.hvecs --output-dir artifacts/laion100m/rabitq \
  --n-lists 1024 --bits 1 --training-rows 262144 --chunk-rows 8192 --seed 137
```

PQ produces `pq.pivots.bin` and `pq.codes.bin`; RaBitQ produces `rabitq.b1.bin` for `--bits 1`. Both save `parameters.json`. RaBitQ generation includes conversion to the native runtime format.

Use `--row-limit` for a dataset prefix and `--dim` to validate dimensions. PQ supports 16 or 64 equal-sized, 8-bit subspaces; the input dimension must divide evenly. RaBitQ streams encoding chunks by default; `--bits 1 2` produces multiple widths and `--build-mode batch` selects a resident build. See `python3 -m scripts.quantize <pq|rabitq> --help` for all options.

### Build graphs and search

Follow the [configuration guide](../configs/README.md) to select the graph builders and run the matching search workload. Development checks are documented in [build.md](../docs/build.md#cpu-and-preparation-checks).

## 2. Generate inputs (optional)

The supplied query package is the standard input for the experiments. The query-selection tools below consume user-provided atomic candidate manifests (`SOURCE_RANGE_MANIFEST`, `query_templates.jsonl` or `query_manifest.jsonl`); they do not generate those starting candidates from raw captions. Use the supplied queries to reproduce the experiments.

The preparation commands below support generating metadata, constructing predicate indexes and computing exact GT for your own inputs. Retain the supplied token map when preparing metadata for the supplied queries, so token IDs remain aligned.

### LAION tokens and attributes

Install `requirements-data.txt`, then generate labels and attributes for the same prefix:

```bash
python3 -m nltk.downloader stopwords
python3 -m scripts.laion.metadata \
  --metadata "$LAION_SOURCE/metadata" --images "$LAION_SOURCE/images" \
  --output artifacts/laion100m/metadata --rows 100000000
for name in base.spmat token_map.parquet row_map.parquet; do
  ln -s "$(pwd)/artifacts/laion100m/metadata/$name" "data/laion100m/$name"
done
SOURCE_RANGE_FBIN=artifacts/laion100m/metadata/base_ranges.fbin
```

The producer lowercases captions with DuckDB, extracts `[a-z][a-z0-9]{2,}` tokens, removes English stopwords and deduplicates within each row. It retains all remaining tokens, expands the fixed categorical aliases, and assigns the fixed category IDs first, followed by remaining tokens in descending document frequency with lexical ties. For existing queries or indexes, pass their `--token-map PATH` to retain those IDs; unknown tokens fail validation. Numeric columns are width, height and similarity: original dimensions take precedence over resized dimensions, and invalid values become NaN without dropping rows. `row_map.parquet` records the original shard/row positions; `category_canonicalization.csv` records category aliases and IDs.

Use a new output directory. `--batch-rows` bounds record batches and `--memory-limit` sets DuckDB's memory budget; intermediate data can use temporary disk space. Query `query_vector_id` values index these same base rows. The remaining stages are listed below; module names are relative to `scripts.laion`.

| Stage | Module / subcommand | Inputs → outputs |
| --- | --- | --- |
| Caption and attribute preparation | `metadata` | Matching parquet/NPY prefix → labels, token/row maps and numeric values |
| Base vectors / query materialization | `base export-vectors` / `base` | NPY prefix → base vectors; aligned labels/ranges and selected manifest → query sidecars |
| Numeric index | `range_index hierarchy`, then `range_index postings` | Width/height values → interval labels, hierarchy and ordered postings |
| String index | `string_index` | Sparse labels and token map → token postings |
| Query selection | `range_queries`, `hybrid_queries` | Atomic candidates and predicate indexes → range or pairwise hybrid manifests |
| Query materialization | `query` | Selected manifest and row map → aligned vectors/ranges and operator sidecars |
| Exact GT | `categorical_gt`, `range_gt`, `string_gt`, `hybrid_gt` | Matching vectors and predicates → exact neighbors/distances |
| Shard assembly (optional) | `assemble split`, `assemble combine-range`, `assemble combine-hybrid` | Query subsets or completed GT shards → aligned outputs |

Append `--help` to a module/subcommand for its input paths and options, for example:

```bash
python3 -m scripts.laion.base --help
python3 -m scripts.laion.range_index postings --help
```

### LAION numeric inputs

Set `SOURCE_RANGE_FBIN` to the aligned source numeric matrix and `SOURCE_RANGE_MANIFEST` to the candidate range-query JSONL. The example extracts width and height from columns 0 and 1; adjust `--columns` if their source positions differ.

```bash
python3 -m scripts.laion.range_index hierarchy \
  --source-base-range-fbin "$SOURCE_RANGE_FBIN" --columns 0,1 --levels 32,256
python3 -m scripts.laion.range_queries \
  --source-manifest "$SOURCE_RANGE_MANIFEST" \
  --value-index-root artifacts/laion100m/range/value_index \
  --output-root artifacts/laion100m/range/query_candidates
python3 -m scripts.laion.range_index postings \
  --source-root artifacts/laion100m/range/value_index \
  --output artifacts/laion100m/range/ordered_postings.bin
python3 -m scripts.laion.base \
  --runtime-root data/laion100m \
  --query-manifest artifacts/laion100m/range/query_candidates/100M/query_manifest.jsonl \
  --base-vector-path data/laion100m/base.hvecs \
  --data-label-spmat data/laion100m/range/base.virtual.spmat \
  --base-range-fbin data/laion100m/range/base_range.fbin \
  --range-label-manifest data/laion100m/range/manifest.json --include-range \
  --output-root data/laion100m/queries/range \
  --filtered-manifest-output query_manifest.jsonl
```

The range generator builds missing value indexes and exact rectangle counts. Its count cache is bound to the source manifest and base values. Use a fresh output/index directory when those inputs change.

For strings, run `query` with an explicit source `--query-manifest`, per-workload `--output-root`, and `--families string --subtypes exact` (or `fuzzy`). Its ngram output defaults to `token_ngram_index.bin` under that output directory; relative output overrides also resolve there.

When generating new candidates or exact GT, build token postings and the full-vocabulary Python ngram index from aligned base labels and the fixed token map:

```bash
python -m scripts.laion.string_index \
  --base-spmat data/laion100m/base.spmat --token-map data/laion100m/token_map.parquet \
  --output-path artifacts/laion100m/token_postings.bin \
  --ngram-output artifacts/laion100m/token_ngram_index.bin
```

`hybrid_queries` and `hybrid_gt` use these artifact paths by default; `string_gt` accepts the ngram path through `--token-ngram-index`. This Python index is separate from the supplied `data/laion100m/token_operator_index.bin` used by native search. Keep the supplied operator index at its configured path. Prepared query/GT packages do not require this regeneration step.

For pairwise hybrid generation, pass `hybrid_queries --base-spmat artifacts/laion100m/metadata/base.spmat --base-range-fbin artifacts/laion100m/metadata/base_ranges.fbin`, together with `--atomic-root`, `--string-root` and a fresh `--out-root`. The two explicit base files support one `--scales` value and preserve the metadata producer's row order and width/height/similarity columns. The atomic/string roots must contain their candidate `query_templates.jsonl` or `query_manifest.jsonl`; `--range-query-root` can provide a separate range manifest. Existing per-scale inputs can instead use `--runtime-root`.

### Exact ground truth

Generate range GT independently of search results:

```bash
CUDA_VISIBLE_DEVICES="${GPU:-0}" python3 -m scripts.laion.range_gt \
  --query-hvecs data/laion100m/queries/range/query.hvecs \
  --query-range-bin data/laion100m/queries/range/query_range.bin \
  --output-root data/laion100m/queries/range/gt --topk 10
```

Run GPU GT jobs sequentially on an idle GPU. When computing GT in shards, split the materialized queries first, compute GT per shard, then combine. `range_gt --search-results-csv PATH` optionally also evaluates an existing result; the general [evaluation command](../docs/experiment-protocol.md#evaluation) works with all workloads. Build and search settings come from `configs/`; preparation scripts only generate input artifacts.
