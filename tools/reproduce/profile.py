#!/usr/bin/env python3
"""Convert completed native host-span captures and timed-run CSVs to window.csv."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import tempfile


WORKLOADS = (
    "yfcc10m", "arxiv_emis", "arxiv_range", "categorical_single", "pure_range",
    "string_exact", "string_fuzzy", "hybrid",
)
CATEGORIES = (
    "graph_phase", "posting_phase", "cpu_rerank_exposed", "gpu_cpu_rerank_overlap",
    "explicit_copy", "result_merge", "prepare", "other_sync_cleanup",
)
FIELDS = ("workload", "q", "run", "wall_ms") + tuple(name + "_ms" for name in CATEGORIES)
SPAN_CATEGORIES = {
    "graph_launch_to_complete": "graph",
    "lowfreq_kernel_launch_to_complete": "posting",
    "range_posting_kernel_launch_to_complete": "posting",
    "highfreq_cpu_rerank_batch": "cpu",
    "lowfreq_cpu_finalize": "cpu",
    "range_posting_cpu_rerank": "cpu",
    "graph_result_d2h": "copy",
    "graph_distance_d2h": "copy",
    "lowfreq_h2d": "copy",
    "lowfreq_shortlist_d2h": "copy",
    "lowfreq_distance_d2h": "copy",
    "range_posting_d2h": "copy",
    "merge_routes_to_original": "merge",
    "final_route_merge": "merge",
    "dual_graph_final_merge": "merge",
    "graph_prepare": "prepare",
    "lowfreq_prepare": "prepare",
    "iteration_prepare": "prepare",
    "lowfreq_candidate_prepare": "prepare",
    "iteration_prepare_outside_search_window": "prepare",
}
POINT_PATTERN = re.compile(r"(?P<backend>rabitq_)?b[0-9]+(?:\.[0-9]+)?_q(?P<queue>[1-9][0-9]*)_f[1-9][0-9]*")
WALL_TOLERANCE_NS = 50_000


def integer(value, name, minimum=0, maximum=(1 << 63) - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def number(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def point_identity(point_id):
    match = POINT_PATTERN.fullmatch(point_id) if isinstance(point_id, str) else None
    if not match:
        raise ValueError("point_id must match bN_qN_fN or rabitq_bN_qN_fN")
    return ("rabitq" if match["backend"] else "pq", int(match["queue"]))


def csv_integer(value, name, minimum=0):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer(int(value), name, minimum)


def load_timings(path):
    """Read one selected benchmark point; never combine distinct search policies."""
    rows = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"point_id", "phase", "run_index", "wall_ms"}
        if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                or not required.issubset(reader.fieldnames)):
            raise ValueError("timed CSV needs unique point_id,phase,run_index,wall_ms columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("malformed timed CSV row")
            point_identity(row["point_id"])
            if row["phase"] != "timed":
                raise ValueError("only timed iterations belong in the timed CSV")
            run = csv_integer(row["run_index"], "run_index")
            key = (row["point_id"], row["phase"], run)
            if key in rows:
                raise ValueError(f"duplicate timed iteration: {key}")
            row["wall_ms"] = number(row["wall_ms"], "wall_ms", positive=True)
            if "qps" in row:
                number(row["qps"], "qps", positive=True)
            if "query_count" in row:
                csv_integer(row["query_count"], "query_count", minimum=1)
            rows[key] = row
    if not rows:
        raise ValueError("timed CSV has no iterations")
    if len({key[0] for key in rows}) != 1:
        raise ValueError("select one benchmark point per workload capture")
    return rows


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def validate_event(event):
    if not isinstance(event, dict):
        raise ValueError("trace record must be a JSON object")
    if type(event.get("schema_version")) is not int or event["schema_version"] != 1:
        raise ValueError("unsupported trace schema_version; expected 1")
    backend, queue = point_identity(event.get("point_id"))
    if event.get("backend") != backend:
        raise ValueError("backend disagrees with point_id")
    if integer(event.get("queue"), "queue", minimum=1) != queue:
        raise ValueError("queue disagrees with point_id")
    if event.get("phase") != "timed":
        raise ValueError("trace capture must contain timed iterations only")
    run = integer(event.get("run"), "run")
    kind = event.get("kind")
    if kind == "complete":
        integer(event.get("event_count"), "event_count", minimum=1)
    elif kind in ("span", "counter"):
        if not isinstance(event.get("name"), str) or not event["name"]:
            raise ValueError("event name must be a nonempty string")
        start = integer(event.get("start_ns"), "start_ns")
        end = integer(event.get("end_ns"), "end_ns")
        if end < start:
            raise ValueError("end_ns precedes start_ns")
        for field in ("context", "thread"):
            if field in event:
                integer(event[field], field, maximum=(1 << 64) - 1)
        if kind == "counter":
            if start != end:
                raise ValueError("counter timestamps must be equal")
            if not isinstance(event.get("value"), (int, float)):
                raise ValueError("counter value must be numeric")
            number(event["value"], "counter value")
    else:
        raise ValueError(f"unknown trace kind: {kind}")
    return (event["point_id"], event["phase"], run)


def completed_iterations(path):
    """Stream contiguous iterations, buffering only the current iteration's spans."""
    seen = set()
    pending_key = None
    events = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                if not line.endswith("\n"):
                    raise ValueError("unterminated trace record")
                event = json.loads(line, object_pairs_hook=unique_object,
                                   parse_constant=reject_constant)
                key = validate_event(event)
                if key in seen:
                    raise ValueError(f"duplicate or late completed iteration: {key}")
                if pending_key is not None and key != pending_key:
                    raise ValueError("iteration changed before its completion record")
                if event["kind"] == "complete":
                    if event["event_count"] != len(events):
                        raise ValueError("completion event_count does not match records")
                    seen.add(key)
                    yield key, events
                    pending_key, events = None, []
                else:
                    pending_key = key
                    events.append(event)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if events:
        raise ValueError(f"{path}: incomplete iteration (missing completion record)")
    if not seen:
        raise ValueError(f"{path}: no completed iterations")


def category(active):
    graph, posting, cpu = active["graph"] > 0, active["posting"] > 0, active["cpu"] > 0
    if (graph or posting) and cpu:
        return "gpu_cpu_rerank_overlap"
    if graph:
        return "graph_phase"
    if posting:
        return "posting_phase"
    if cpu:
        return "cpu_rerank_exposed"
    for source, target in (("copy", "explicit_copy"), ("merge", "result_merge"),
                           ("prepare", "prepare")):
        if active[source] > 0:
            return target
    return "other_sync_cleanup"


def partition(events, start, end):
    """Partition clipped half-open spans in integer nanoseconds, counting overlaps once."""
    if end <= start:
        raise ValueError("search window must have positive duration")
    boundaries = defaultdict(Counter)
    boundaries[start], boundaries[end]  # Include uncovered leading/trailing time.
    for event in events:
        kind = SPAN_CATEGORIES.get(event.get("name"))
        if event["kind"] != "span" or kind is None:
            continue
        left, right = max(start, event["start_ns"]), min(end, event["end_ns"])
        if left < right:
            boundaries[left][kind] += 1
            boundaries[right][kind] -= 1
    totals = dict.fromkeys(CATEGORIES, 0)
    active = Counter()
    previous = start
    for position, changes in sorted(boundaries.items()):
        totals[category(active)] += position - previous
        active.update(changes)
        previous = position
    if sum(totals.values()) != end - start or any(value < 0 for value in totals.values()):
        raise ValueError("non-conserving time partition")
    if totals["other_sync_cleanup"] == end - start:
        raise ValueError("search window has no positive classified work spans")
    return totals


def search_window(events, wall_ms):
    spans = [event for event in events if event["kind"] == "span"]
    dual = [event for event in spans if event["name"] == "dual_graph_search_window"]
    providers = [event for event in spans if event["name"] == "provider_search_window"]
    if len(dual) == 1:
        window = dual[0]
        if any(event["start_ns"] < window["start_ns"] or event["end_ns"] > window["end_ns"]
               for event in providers):
            raise ValueError("provider window lies outside the dual search window")
    elif not dual and len(providers) == 1:
        window = providers[0]
    else:
        raise ValueError("need exactly one dual search window or one provider search window")
    start, end = window["start_ns"], window["end_ns"]
    if end <= start or abs((end - start) - wall_ms * 1e6) > WALL_TOLERANCE_NS:
        raise ValueError("search window duration disagrees with timed wall_ms")
    counters = [event for event in events
                if event["kind"] == "counter" and event["name"] == "reported_search_wall_ms"]
    if len(counters) > 1:
        raise ValueError("duplicate reported_search_wall_ms counter")
    if counters and abs(counters[0]["value"] - wall_ms) * 1e6 > WALL_TOLERANCE_NS:
        raise ValueError("reported_search_wall_ms counter disagrees with timed wall_ms")
    return start, end


def convert_capture(workload, trace, timing_csv):
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload: {workload}; choose from {', '.join(WORKLOADS)}")
    timing = load_timings(timing_csv)
    remaining = set(timing)
    rows = []
    for key, events in completed_iterations(trace):
        if key not in remaining:
            raise ValueError(f"trace iteration missing from timed CSV: {key}")
        remaining.remove(key)
        start, end = search_window(events, timing[key]["wall_ms"])
        totals = partition(events, start, end)
        rows.append(dict(workload=workload, q=point_identity(key[0])[1], run=key[2],
                         wall_ms=(end - start) / 1e6,
                         **{name + "_ms": value / 1e6 for name, value in totals.items()}))
    if remaining:
        raise ValueError(f"timed iterations missing from trace: {sorted(remaining)}")
    return sorted(rows, key=lambda row: row["run"])


def convert(inputs, output):
    """Validate all captures before atomically installing a fresh output file."""
    output = Path(output)
    paths = [Path(path).resolve() for _, trace, timing in inputs for path in (trace, timing)]
    if output.resolve() in paths:
        raise ValueError("output must not overwrite an input")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    if not inputs or len({item[0] for item in inputs}) != len(inputs):
        raise ValueError("provide one distinct input triple per workload")
    if len(paths) != len(set(paths)):
        raise ValueError("each input trace and timed CSV must be distinct")
    rows = []
    for workload, trace, timing in inputs:
        rows.extend(convert_capture(workload, trace, timing))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=output.parent, prefix=f".{output.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)  # Exclusive installation also protects a racing writer.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs=3, action="append", required=True,
                        metavar=("WORKLOAD", "TRACE_JSONL", "TIMED_CSV"),
                        help="one point per workload; repeat for multiple workloads")
    parser.add_argument("--output", type=Path, required=True, help="new window CSV path")
    args = parser.parse_args(argv)
    try:
        rows = convert(args.input, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote {len(rows)} timed iterations to {args.output}")


if __name__ == "__main__":
    main()
