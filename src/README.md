# Native source layout

| Directory | Responsibility |
| --- | --- |
| `app/` | CLI, configuration and backend dispatch |
| `data/` | Dataset loading, vector representation and CPU distance helpers |
| `index/` | CAGRA construction, graph formats and range postings |
| `planning/` | Query preparation and route planning |
| `search/pq/`, `search/rabitq/`, `search/full_gpu/` | Search backends |
| `search/common/` | Shared kernels, heaps, merging and navigation |
| `execution/` | CPU–GPU channels, host service, posting completion and reranking |
| `predicates/` | GPU predicate storage and checks |
| `profiling/` | Timing, tracing and CSV output |
| `../include/` | Shared heap, hash and type headers |
| `../tests/native/` | Native tests |

`app/main.cu` dispatches the requested mode and backend. Data and index metadata feed query planning; search uses predicate checks and CPU–GPU channels to fetch host-resident data and rerank candidates.

Build with `make exe` from the repository root, or `make -C src exe`. The executable is `src/go-fann`; see [build instructions](../docs/build.md) for dependencies and tests.
