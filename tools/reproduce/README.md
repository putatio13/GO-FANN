# Paper figures and reference results

The `reference/` CSVs contain the measurements used by the paper's evaluation
figures.

| Paper figure or table | CSV | Content |
|---|---|---|
| Recall@10–QPS | `main.csv` | 419 points across eight workloads, including baselines |
| Query planning optimizations | `planning.csv` | 100 points for fuzzy pruning, range access and AND driving access |
| Predicate-aware penalty | `penalty.csv` | 45 points across three workloads |
| Rerank placement | `placement.csv` | 30 points across three workloads |
| Execution overlap | `overlap.csv` | 45 points across three workloads |
| Search-window breakdown | `window.csv` | Three timed samples for each of eight workloads |
| Index storage and construction time | `index-costs.csv` | Seven LAION100M index records |

Render a figure from the repository root (Python and Matplotlib required):

```bash
python -m pip install matplotlib
python tools/reproduce/plot.py --figure main --output-dir outputs/figures
python tools/reproduce/plot.py --figure planning --output-dir outputs/figures
python tools/reproduce/plot.py --figure penalty --output-dir outputs/figures
python tools/reproduce/plot.py --figure placement --output-dir outputs/figures
python tools/reproduce/plot.py --figure overlap --output-dir outputs/figures
python tools/reproduce/plot.py --figure window --output-dir outputs/figures
```

Each command writes a PDF and PNG. The renderer keeps every recorded point,
including coincident measurements, and displays the full recall range.
It uses a common layout rather than reproducing the paper's inset and typography.
The window chart divides summed exclusive phase times by summed wall time across
runs; it does not average per-run percentages.
To plot a selected workload from a new capture, add `--workloads yfcc10m`
to the window command. Without this option, the window figure requires all
eight workloads. `profile.py` converts completed native traces and matching
timed-run CSVs into the window schema; follow the
[capture and conversion commands](../../docs/profiling.md#search-window-breakdown).
The [placement and stage-collection instructions](../../REPRODUCE.md#rerank-placement)
define the placement variants and explain which measurements can be collected
with the released runtime.

Curves connect all recorded points in recall order. Faiss IVF-PQ sweeps both
`nprobe` and the candidate count, so connecting the combined grid can produce
nonmonotonic lines. The LAION Milvus HNSW curves each retain six measured points;
several have identical recall and similar QPS and therefore overlap visually.
No envelope filtering, smoothing or deduplication is applied. See
[baseline settings](../../docs/baselines.md) for the parameter grids.

To plot a new experiment, write the corresponding CSV into a separate directory
and pass `--input-dir PATH`. Use `dataset,method,recall_at_10,qps` for main results;
use `family,variant,recall_at_10,qps_mean` for planning and penalty, adding
`workload` for placement and overlap. Keep the panel identifiers and timing
column names in the reference CSVs. Additional columns retain the available
queue parameters, timing statistics and query counts; blank values mean that
the original summary did not supply that field. The plotting command reads CSV
measurements; it does not run searches. See [REPRODUCE.md](../../REPRODUCE.md)
for running and evaluating GO-FANN.

In `index-costs.csv`, storage uses bytes and GiB; construction fields use seconds
and minutes as named. The two Milvus times are approximate interrupted-session
totals, as marked by `build_evidence`, rather than continuous build timings.
