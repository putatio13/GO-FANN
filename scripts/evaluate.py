#!/usr/bin/env python3
"""Evaluate saved ANN result IDs against aligned ground truth, on the CPU."""

import argparse
from contextlib import closing
import csv
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys


UINT32_MAX = (1 << 32) - 1
MISSING = {-1, UINT32_MAX}
INTEGER = re.compile(r"[+-]?[0-9]+")


def integer(value, location):
    if not INTEGER.fullmatch(value.strip()):
        raise ValueError(f"{location}: expected an integer, got {value!r}")
    return int(value)


def validated_ids(ids, location, base_rows):
    """Validate the entire row, including entries beyond the scoring cutoff."""
    seen = set()
    upper = UINT32_MAX if base_rows is None else base_rows
    for value in ids:
        if value in MISSING:
            continue
        if not 0 <= value < upper:
            raise ValueError(f"{location}: ID {value} is outside [0, {upper})")
        if value in seen:
            raise ValueError(f"{location}: duplicate ID {value}")
        seen.add(value)
    return ids


def csv_rows(source, path):
    reader = csv.reader(source, strict=True)
    while True:
        try:
            row = next(reader, None)
        except csv.Error as error:
            raise ValueError(f"{path}: line {reader.line_num}: {error}") from error
        if row is None:
            return
        yield row


def prediction_rows(path, k, base_rows):
    with path.open(newline="", encoding="utf-8") as source:
        width = None
        for row_number, row in enumerate(csv_rows(source, path), 1):
            location = f"{path}: row {row_number}"
            if not row:
                raise ValueError(f"{location}: empty prediction row")
            if len(row) < k:
                raise ValueError(f"{location}: expected at least {k} IDs, got {len(row)}")
            if width is not None and len(row) != width:
                raise ValueError(f"{location}: inconsistent prediction width")
            width = len(row)
            ids = [integer(value, location) for value in row]
            yield validated_ids(ids, location, base_rows)


def ground_truth_rows(path, k, base_rows):
    if path.suffix not in {".ibin", ".ivecs"}:
        raise ValueError(f"{path}: ground truth must have .ibin or .ivecs extension")
    with path.open("rb") as source:
        size = source.seek(0, 2)
        source.seek(0)
        if path.suffix == ".ibin":
            header = source.read(8)
            if len(header) != 8:
                raise ValueError(f"{path}: truncated ibin header")
            rows, width = struct.unpack("<II", header)
            if rows == 0 or width < k:
                raise ValueError(f"{path}: ibin requires nonzero rows and width >= {k}")
            payload_size = 4 * rows * width
            if size not in {8 + payload_size, 8 + 2 * payload_size}:
                raise ValueError(f"{path}: ibin size does not match IDs or IDs-plus-distances layout")
            for row_number in range(1, rows + 1):
                payload = source.read(width * 4)
                if len(payload) != width * 4:
                    raise ValueError(f"{path}: row {row_number}: truncated IDs")
                ids = struct.unpack(f"<{width}I", payload)
                yield validated_ids(ids, f"{path}: row {row_number}", base_rows)
        else:
            row_number = 0
            while header := source.read(4):
                row_number += 1
                location = f"{path}: row {row_number}"
                if len(header) != 4:
                    raise ValueError(f"{location}: truncated ivecs row header")
                width, = struct.unpack("<i", header)
                if width < 0:
                    raise ValueError(f"{location}: negative ivecs row length")
                if 4 * width > size - source.tell():
                    raise ValueError(f"{location}: truncated ivecs IDs")
                payload = source.read(width * 4)
                if len(payload) != width * 4:
                    raise ValueError(f"{location}: truncated ivecs IDs")
                ids = struct.unpack(f"<{width}i", payload)
                yield validated_ids(ids, location, base_rows)


def timing_summary(path, results, query_count):
    required = {"phase", "run_index", "point_id", "wall_ms", "qps", "query_count", "result_path"}
    qps_runs, wall_ms_runs = [], []
    run_indices = set()
    point_id = None
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv_rows(source, path)
        fields = next(reader, None)
        if (not fields or not required.issubset(fields) or "" in fields
                or len(fields) != len(set(fields))):
            raise ValueError(f"{path}: invalid timed-runs header; required fields: {', '.join(sorted(required))}")
        for row_number, values in enumerate(reader, 2):
            location = f"{path}: row {row_number}"
            if len(values) != len(fields):
                raise ValueError(f"{location}: field count does not match header")
            row = dict(zip(fields, values))
            if row["phase"] == "warmup":
                continue
            if row["phase"] != "timed":
                raise ValueError(f"{location}: unknown timing phase {row['phase']!r}")
            if not row["point_id"] or (point_id is not None and row["point_id"] != point_id):
                raise ValueError(f"{location}: timed records must identify one nonempty point_id")
            point_id = row["point_id"]
            run_index = integer(row["run_index"], location)
            if run_index < 0 or run_index in run_indices:
                raise ValueError(f"{location}: negative or duplicate timed run_index")
            run_indices.add(run_index)
            if integer(row["query_count"], location) != query_count:
                raise ValueError(f"{location}: query_count differs from evaluated results")
            if not row["result_path"] or Path(row["result_path"]).resolve() != results.resolve():
                raise ValueError(f"{location}: result_path differs from evaluated results")
            try:
                wall_ms, qps = float(row["wall_ms"]), float(row["qps"])
            except ValueError as error:
                raise ValueError(f"{location}: wall_ms and qps must be numeric") from error
            if not all(math.isfinite(value) and value > 0 for value in (wall_ms, qps)):
                raise ValueError(f"{location}: wall_ms and qps must be finite and positive")
            if not math.isclose(qps, query_count * 1000 / wall_ms, rel_tol=1e-8):
                raise ValueError(f"{location}: qps disagrees with query_count * 1000 / wall_ms")
            qps_runs.append(qps)
            wall_ms_runs.append(wall_ms)
    if not qps_runs:
        raise ValueError(f"{path}: no timed records")
    return {
        "timed_run_count": len(qps_runs),
        "qps_runs": qps_runs,
        "wall_ms_runs": wall_ms_runs,
        "qps_mean": statistics.mean(qps_runs),
        "qps_std": statistics.stdev(qps_runs) if len(qps_runs) > 1 else None,
    }


def evaluate(results, ground_truth, *, k=10, base_rows=None, timed_runs=None):
    if k <= 0:
        raise ValueError("k must be positive")
    if base_rows is not None and not 0 < base_rows <= UINT32_MAX:
        raise ValueError(f"base-rows must be in [1, {UINT32_MAX}]")
    results, ground_truth = Path(results), Path(ground_truth)
    query_count = hits = valid_gt = zero_gt_queries = missing_slots = 0
    macro_sum = 0.0
    with closing(prediction_rows(results, k, base_rows)) as predictions, closing(
        ground_truth_rows(ground_truth, k, base_rows)
    ) as truth:
        for row_number, gt_row in enumerate(truth, 1):
            predicted_row = next(predictions, None)
            if predicted_row is None:
                raise ValueError(f"{results}: missing prediction for GT row {row_number}")
            expected = {value for value in gt_row[:k] if value not in MISSING}
            predicted = {value for value in predicted_row[:k] if value not in MISSING}
            row_hits = len(expected & predicted)
            hits += row_hits
            valid_gt += len(expected)
            missing_slots += sum(value in MISSING for value in predicted_row[:k])
            query_count += 1
            if expected:
                macro_sum += row_hits / len(expected)
            else:
                zero_gt_queries += 1
        if next(predictions, None) is not None:
            raise ValueError(f"{results}: more prediction rows than ground truth")
    if query_count == 0:
        raise ValueError(f"{ground_truth}: no query rows")
    nonempty = query_count - zero_gt_queries
    return {
        "k": k,
        f"recall_at_{k}": macro_sum / nonempty if nonempty else None,
        "query_count": query_count,
        "hits": hits,
        "valid_gt": valid_gt,
        "zero_gt_queries": zero_gt_queries,
        "missing_prediction_slots": missing_slots,
        "timing": timing_summary(Path(timed_runs), results, query_count) if timed_runs else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Headerless integer ID CSV, one row per query.")
    parser.add_argument("--ground-truth", required=True, type=Path, help="Aligned .ibin or variable-length .ivecs GT.")
    parser.add_argument("--base-rows", type=int, help="Base vector count; enables dataset bounds validation.")
    parser.add_argument("--k", type=int, default=10, help="Mean per-query recall over the first k positions (default: 10).")
    parser.add_argument("--timed-runs", type=Path, help="Optional matching native per-point timed-runs CSV.")
    args = parser.parse_args()
    try:
        report = evaluate(args.results, args.ground_truth, k=args.k,
                          base_rows=args.base_rows, timed_runs=args.timed_runs)
    except (OSError, ValueError, csv.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
