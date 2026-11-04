#!/usr/bin/env python3
"""Render paper recall/QPS curves from reference or newly measured CSVs."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


PANELS = {
    "main": [
        ("yfcc10m", "YFCC10M"), ("arxiv-emis", "arxiv EMIS"),
        ("arxiv-range", "arxiv range"),
        ("laion-categorical-single", "LAION categorical"),
        ("laion-range", "LAION range"), ("laion-exact", "LAION exact string"),
        ("laion-fuzzy", "LAION fuzzy string"), ("laion-hybrid", "LAION hybrid"),
    ],
    "planning": [
        ("fuzzy_cap", "Fuzzy branch pruning"),
        ("pure_range", "Range access selection"),
        ("range_label_and", "Range–label AND driving access"),
    ],
    "penalty": [
        ("predicate_yfcc10m", "Penalty: YFCC10M"),
        ("predicate_arxiv_range", "Penalty: arxiv range"),
        ("predicate_range_label_and", "Penalty: LAION range–label AND"),
    ],
    "placement": [
        ("placement/yfcc10m", "Rerank: YFCC10M"),
        ("placement/arxiv_emis", "Rerank: arxiv EMIS"),
        ("placement/categorical_single", "Rerank: LAION categorical"),
    ],
    "overlap": [
        ("pipeline/yfcc10m", "Overlap: YFCC10M"),
        ("pipeline/hybrid", "Overlap: LAION hybrid"),
        ("pipeline/string_fuzzy", "Overlap: LAION fuzzy string"),
    ],
}
WINDOW_WORKLOADS = [
    ("yfcc10m", "YFCC10M"), ("arxiv_emis", "arxiv EMIS"),
    ("arxiv_range", "arxiv range"), ("categorical_single", "LAION categorical"),
    ("pure_range", "LAION range"), ("string_exact", "LAION exact string"),
    ("string_fuzzy", "LAION fuzzy string"), ("hybrid", "LAION hybrid"),
]
WINDOW_GROUPS = [
    ("Search only", ("graph_phase_ms", "posting_phase_ms")),
    ("Search + rerank", ("gpu_cpu_rerank_overlap_ms",)),
    ("Exposed rerank", ("cpu_rerank_exposed_ms",)),
    ("Other", ("explicit_copy_ms", "result_merge_ms", "prepare_ms", "other_sync_cleanup_ms")),
]
LABELS = {
    "cap8": "Cap 8", "cap16": "Cap 16", "cap32": "Cap 32", "nocap": "No cap",
    "graph_only": "Graph only", "auto20k": "Auto 20K", "auto50k": "Auto 50K",
    "posting_only_100m": "Posting only", "auto": "Auto", "dual": "Dual",
    "label_prefer": "Prefer label", "range_prefer": "Prefer range",
    "lambda_0": "λ = 0", "lambda_0.15": "λ = 0.15", "lambda_0.3": "λ = 0.3",
    "gpu_inline": "GPU inline rerank", "cpu_batch": "CPU rerank",
    "no_overlap": "Serial", "batch_overlap": "Stage overlap", "streaming": "Streaming overlap",
}


def load_curves(path, figure):
    """Keep every row, including coincident points, and reject invalid measurements."""
    panel_keys = {key for key, _ in PANELS[figure]}
    curves = defaultdict(lambda: defaultdict(list))
    qps_key = "qps" if figure == "main" else "qps_mean"
    required = {"recall_at_10", qps_key}
    required.update(("dataset", "method") if figure == "main" else ("family", "variant"))
    if figure in ("placement", "overlap"):
        required.add("workload")
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: {', '.join(sorted(required))}")
        for line, row in enumerate(reader, 2):
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError(f"{path}:{line}: malformed CSV row")
            if figure == "main":
                panel, label = row["dataset"], row["method"]
            else:
                panel, label = row["family"], row["variant"]
                if panel in ("placement", "pipeline"):
                    panel += "/" + row["workload"]
            if panel not in panel_keys or not label.strip():
                raise ValueError(f"{path}:{line}: unknown panel or empty curve label")
            try:
                recall, qps = float(row["recall_at_10"]), float(row[qps_key])
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: recall/QPS must be numeric") from exc
            if not math.isfinite(recall) or not 0 <= recall <= 1 or not math.isfinite(qps) or qps <= 0:
                raise ValueError(f"{path}:{line}: require finite recall in [0, 1] and positive QPS")
            curves[panel][label].append((recall, qps))
    if missing := panel_keys - curves.keys():
        raise ValueError(f"{path}: missing panels: {', '.join(sorted(missing))}")
    return curves


def load_windows(path, workloads=None):
    """Aggregate exclusive phase times as sum(phase) / sum(wall) per workload."""
    requested = list(dict(WINDOW_WORKLOADS)) if workloads is None else list(workloads)
    if (not requested or len(requested) != len(set(requested)) or
            any(key not in dict(WINDOW_WORKLOADS) for key in requested)):
        raise ValueError("select distinct known window workloads")
    samples = defaultdict(list)
    fields = [field for _, group in WINDOW_GROUPS for field in group]
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(fields + ["workload", "q", "run", "wall_ms"]).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing timing columns")
        for line, row in enumerate(reader, 2):
            if row["workload"] not in dict(WINDOW_WORKLOADS):
                raise ValueError(f"{path}:{line}: unknown workload")
            try:
                values = {field: float(row[field]) for field in fields + ["wall_ms"]}
                run, queue = int(row["run"]), int(row["q"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: invalid timing value") from exc
            if run < 0 or queue <= 0 or any(not math.isfinite(v) or v < 0 for v in values.values()):
                raise ValueError(f"{path}:{line}: require nonnegative finite times and valid run/queue")
            wall = values["wall_ms"]
            if wall <= 0 or not math.isclose(sum(values[f] for f in fields), wall, rel_tol=1e-6, abs_tol=1e-3):
                raise ValueError(f"{path}:{line}: exclusive phase times must sum to positive wall time")
            samples[row["workload"]].append((run, queue, values))
    shares = {}
    for workload, records in samples.items():
        if len({run for run, _, _ in records}) != len(records) or len({q for _, q, _ in records}) != 1:
            raise ValueError(f"{path}: {workload}: missing runs, duplicate run IDs or mixed queues")
    for workload in requested:
        records = samples[workload]
        if not records:
            raise ValueError(f"{path}: {workload}: missing runs")
        wall = sum(values["wall_ms"] for _, _, values in records)
        shares[workload] = [100 * sum(values[f] for _, _, values in records for f in fields) / wall
                            for _, fields in WINDOW_GROUPS]
    return shares


def render_windows(shares):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    workloads = [(key, dict(WINDOW_WORKLOADS)[key]) for key in shares]
    height = 4.5 if len(workloads) == len(WINDOW_WORKLOADS) else max(2, .4 * len(workloads) + 1.3)
    fig, ax = plt.subplots(figsize=(8, height), layout="constrained")
    left = [0.] * len(workloads)
    for i, (label, _) in enumerate(WINDOW_GROUPS):
        values = [shares[key][i] for key, _ in workloads]
        ax.barh(range(len(left)), values, left=left, label=label)
        left = [a + b for a, b in zip(left, values)]
    ax.set_yticks(range(len(left)), [label for _, label in workloads])
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="Search window (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=4, fontsize=8, frameon=False)
    return fig


def render(curves, figure):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 4 if figure == "main" else 3
    rows = math.ceil(len(PANELS[figure]) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.1 * columns, 3.6 * rows),
                             squeeze=False, layout="constrained")
    labels = list(dict.fromkeys(label for key, _ in PANELS[figure] for label in curves[key]))
    palette = plt.get_cmap("tab20")
    markers = ("o", "s", "^", "D", "v", "P", "X")
    styles = {label: {"color": palette(i % 20), "marker": markers[i % len(markers)],
                      "linestyle": ("-", "--", "-.")[i % 3]}
              for i, label in enumerate(labels)}
    for i, ((key, title), ax) in enumerate(zip(PANELS[figure], axes.flat)):
        for label, points in curves[key].items():
            # Stable sorting matches the paper's recall/QPS curves, without filtering.
            ordered = sorted(points, key=lambda point: point[0])
            x, y = zip(*ordered)
            ax.plot(x, y, label=LABELS.get(label, label), markersize=3,
                    linewidth=1, **styles[label])
        ax.set(title=f"({chr(97 + i)}) {title}", xlabel="Recall@10", ylabel="QPS", yscale="log")
        ax.grid(True, alpha=.25)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=[*PANELS, "window"], required=True)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).parent / "reference")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", choices=list(dict(WINDOW_WORKLOADS)),
                        help="window only: render these captured workloads (default: all eight)")
    args = parser.parse_args()
    if args.workloads is not None and args.figure != "window":
        parser.error("--workloads is only supported for --figure window")
    try:
        path = args.input_dir / f"{args.figure}.csv"
        curves = load_windows(path, args.workloads) if args.figure == "window" else load_curves(path, args.figure)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    paths = [args.output_dir / f"{args.figure}.{ext}" for ext in ("pdf", "png")]
    if any(path.exists() for path in paths):
        parser.error("output exists; select a new output directory")
    fig = render_windows(curves) if args.figure == "window" else render(curves, args.figure)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        fig.savefig(path, dpi=180)
    import matplotlib.pyplot as plt
    plt.close(fig)
    if args.figure == "window":
        print(f"Rendered {len(curves)} workload timing breakdowns to {args.output_dir}")
    else:
        count = sum(len(points) for panel in curves.values() for points in panel.values())
        print(f"Rendered {count} points in {len(curves)} panels to {args.output_dir}")


if __name__ == "__main__":
    main()
