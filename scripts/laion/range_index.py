#!/usr/bin/env python3
"""Build LAION width/height independent range-bucket artifacts.

The LAION100M source range file stores width, height, and similarity in one
three-column fbin. This script extracts width/height, creates a two-column base
range fbin, and writes a range-only spmat where each dimension owns an
independent virtual-label bucket hierarchy.

The default bucket policy partitions observed unique values into non-empty
contiguous buckets. That avoids empty manifest intervals that can make the
pure-range cover planner leave gaps for queries.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_ROOT = Path("data/laion100m/range")

FBIN_HEADER_DTYPE = np.dtype("<u4")
FBIN_VALUE_DTYPE = np.dtype("<f4")
SPMAT_HEADER_DTYPE = np.dtype("<i8")
SPMAT_INDEX_DTYPE = np.dtype("<i4")
SPMAT_DATA_DTYPE = np.dtype("<f4")


def csv_ints(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def csv_names(raw: str) -> list[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("expected at least one name")
    return values


def parse_levels(raw: str) -> list[int]:
    levels = sorted(csv_ints(raw))
    if any(level <= 0 for level in levels):
        raise ValueError("--levels must contain positive integers")
    if len(set(levels)) != len(levels):
        raise ValueError("--levels cannot contain duplicate entries")
    for previous, current in zip(levels, levels[1:]):
        if current % previous != 0:
            raise ValueError(
                "each --levels value must be an integer multiple of the previous value"
            )
    return levels


def default_path(output_root: Path, explicit: Path | None, name: str) -> Path:
    if explicit is not None:
        return explicit
    return output_root / name


def tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def prepare_tmp(path: Path, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    tmp = tmp_path(path)
    if tmp.exists():
        tmp.unlink()
    return tmp


def publish_tmp(tmp: Path, path: Path) -> None:
    tmp.replace(path)


def inspect_fbin(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=FBIN_HEADER_DTYPE, count=2)
    if header.size != 2:
        raise ValueError(f"invalid fbin header in {path}")
    rows, cols = [int(value) for value in header.tolist()]
    expected_size = 8 + rows * cols * FBIN_VALUE_DTYPE.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path} size={actual_size} does not match fbin shape {(rows, cols)}")
    return rows, cols


def open_fbin_payload(path: Path, rows: int, cols: int) -> np.memmap:
    return np.memmap(path, dtype=FBIN_VALUE_DTYPE, mode="r", offset=8, shape=(rows, cols))


def update_value_counts(counts: dict[float, int], values: np.ndarray) -> None:
    unique, unique_counts = np.unique(values, return_counts=True)
    for value, count in zip(unique.tolist(), unique_counts.tolist()):
        counts[float(value)] = counts.get(float(value), 0) + int(count)


def collect_stats_and_write_base(
    source: np.memmap,
    source_path: Path,
    output_path: Path,
    selected_columns: list[int],
    range_names: list[str],
    edge_policy: str,
    chunk_rows: int,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[float, int]]]:
    rows = int(source.shape[0])
    range_count = len(selected_columns)
    mins = np.full(range_count, math.inf, dtype=np.float64)
    maxs = np.full(range_count, -math.inf, dtype=np.float64)
    sums = np.zeros(range_count, dtype=np.float64)
    finite_counts = np.zeros(range_count, dtype=np.int64)
    value_counts: list[dict[float, int]] = [dict() for _ in selected_columns]

    tmp = prepare_tmp(output_path, overwrite)
    start_time = time.monotonic()
    with tmp.open("wb") as handle:
        np.asarray([rows, range_count], dtype=FBIN_HEADER_DTYPE).tofile(handle)
        for begin in range(0, rows, chunk_rows):
            end = min(rows, begin + chunk_rows)
            chunk = np.asarray(source[begin:end, selected_columns], dtype=FBIN_VALUE_DTYPE)
            finite = np.isfinite(chunk)
            if not bool(finite.all()):
                bad_positions = np.argwhere(~finite)
                first_bad = bad_positions[0]
                bad_row = begin + int(first_bad[0])
                bad_col = selected_columns[int(first_bad[1])]
                raise ValueError(
                    f"non-finite value in selected range column: row={bad_row} column={bad_col}"
                )

            for range_pos, name in enumerate(range_names):
                values = chunk[:, range_pos]
                mins[range_pos] = min(mins[range_pos], float(values.min()))
                maxs[range_pos] = max(maxs[range_pos], float(values.max()))
                sums[range_pos] += values.astype(np.float64, copy=False).sum()
                finite_counts[range_pos] += values.shape[0]
                update_value_counts(value_counts[range_pos], values)

            chunk.astype(FBIN_VALUE_DTYPE, copy=False).tofile(handle)
            if end == rows or end % (10 * chunk_rows) == 0:
                elapsed = time.monotonic() - start_time
                print(f"[base-range] rows {end}/{rows} ({end / rows:.1%}) elapsed={elapsed:.1f}s")

    publish_tmp(tmp, output_path)

    stats = []
    for range_pos, name in enumerate(range_names):
        unique_count = len(value_counts[range_pos]) if edge_policy == "balanced_unique" else None
        stats.append(
            {
                "range_index": range_pos,
                "name": name,
                "source_base_range_fbin": str(source_path),
                "source_column": int(selected_columns[range_pos]),
                "rows": rows,
                "finite_rows": int(finite_counts[range_pos]),
                "min": float(mins[range_pos]),
                "max": float(maxs[range_pos]),
                "mean": float(sums[range_pos] / max(1, finite_counts[range_pos])),
                "unique_value_count": unique_count,
            }
        )
    return stats, value_counts


def nextafter_high(value: float) -> float:
    return float(np.nextafter(np.float32(value), np.float32(math.inf), dtype=np.float32))


def balanced_unique_finest_edges(value_counts: dict[float, int], num_bins: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sorted(value_counts.keys()), dtype=np.float64)
    counts = np.asarray([value_counts[float(value)] for value in values], dtype=np.int64)
    if values.size < num_bins:
        raise ValueError(
            f"cannot create {num_bins} non-empty buckets from only {values.size} unique values"
        )

    cumsum = np.cumsum(counts, dtype=np.int64)
    total = int(cumsum[-1])
    edges = np.empty(num_bins + 1, dtype=np.float64)
    bucket_counts = np.empty(num_bins, dtype=np.int64)
    edges[0] = float(values[0])

    start = 0
    for bucket in range(num_bins):
        remaining = num_bins - bucket
        if remaining == 1:
            end = values.size
        else:
            target = total * float(bucket + 1) / float(num_bins)
            proposed = int(np.searchsorted(cumsum, target, side="left")) + 1
            min_end = start + 1
            max_end = values.size - (remaining - 1)
            end = max(min_end, min(proposed, max_end))

        bucket_counts[bucket] = int(counts[start:end].sum())
        if bucket + 1 < num_bins:
            left = float(values[end - 1])
            right = float(values[end])
            boundary = (left + right) / 2.0
            if not (left < boundary < right):
                boundary = np.nextafter(left, math.inf)
            if not boundary > edges[bucket]:
                boundary = np.nextafter(edges[bucket], math.inf)
            edges[bucket + 1] = float(boundary)
        start = end

    edges[-1] = nextafter_high(float(values[-1]))
    if not bool(np.all(np.diff(edges) > 0.0)):
        raise ValueError("balanced_unique edge generation produced non-increasing edges")
    return edges, bucket_counts


def bucket_counts_from_unique(value_counts: dict[float, int], edges: np.ndarray) -> np.ndarray:
    values = np.asarray(sorted(value_counts.keys()), dtype=np.float64)
    counts = np.asarray([value_counts[float(value)] for value in values], dtype=np.int64)
    bins = np.searchsorted(edges, values, side="right") - 1
    bins = np.clip(bins, 0, edges.size - 2)
    return np.bincount(bins, weights=counts, minlength=edges.size - 1).astype(np.int64, copy=False)


def build_range_edges(
    stats: dict[str, Any],
    value_counts: dict[float, int],
    levels: list[int],
    edge_policy: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    finest_bins = levels[-1]
    if edge_policy == "balanced_unique":
        finest_edges, finest_counts = balanced_unique_finest_edges(value_counts, finest_bins)
    else:
        low = float(stats["min"])
        high = nextafter_high(float(stats["max"]))
        finest_edges = np.linspace(low, high, finest_bins + 1, dtype=np.float64)
        finest_counts = bucket_counts_from_unique(value_counts, finest_edges)

    edges_by_level: dict[int, np.ndarray] = {}
    counts_by_level: dict[int, np.ndarray] = {}
    for level_bins in levels:
        ratio = finest_bins // level_bins
        if ratio <= 0 or finest_bins % level_bins != 0:
            raise ValueError(f"finest level {finest_bins} is not divisible by level {level_bins}")
        edges = finest_edges[::ratio].copy()
        if edges.size != level_bins + 1:
            edges = np.concatenate([edges, finest_edges[-1:]])
        edges[-1] = finest_edges[-1]
        counts = bucket_counts_from_unique(value_counts, edges)
        edges_by_level[level_bins] = edges
        counts_by_level[level_bins] = counts

    counts_by_level[finest_bins] = finest_counts
    return edges_by_level, counts_by_level


def build_manifest(
    source_path: Path,
    base_range_path: Path,
    spmat_path: Path,
    stats: list[dict[str, Any]],
    selected_columns: list[int],
    range_names: list[str],
    source_column_names: list[str],
    levels: list[int],
    all_edges: list[dict[int, np.ndarray]],
    all_counts: list[dict[int, np.ndarray]],
    edge_policy: str,
) -> dict[str, Any]:
    ranges = []
    running_label_offset = 0
    for range_index, name in enumerate(range_names):
        range_levels = []
        level_offset = 0
        for level_index, num_bins in enumerate(levels):
            edges = all_edges[range_index][num_bins]
            counts = all_counts[range_index][num_bins]
            partitions = []
            for bin_index in range(num_bins):
                label_id = running_label_offset + level_offset + bin_index
                partitions.append(
                    {
                        "label_id": int(label_id),
                        "bin_index": int(bin_index),
                        "range_index": int(range_index),
                        "interval": [float(edges[bin_index]), float(edges[bin_index + 1])],
                        "assigned_rows": int(counts[bin_index]),
                    }
                )
            range_levels.append(
                {
                    "level_index": int(level_index),
                    "num_bins": int(num_bins),
                    "label_id_start": int(running_label_offset + level_offset),
                    "label_id_end_exclusive": int(running_label_offset + level_offset + num_bins),
                    "partitions": partitions,
                }
            )
            level_offset += num_bins

        range_stat = dict(stats[range_index])
        range_stat["edge_policy"] = edge_policy
        ranges.append(
            {
                "range_index": int(range_index),
                "name": name,
                "source_column": int(selected_columns[range_index]),
                "source_column_name": source_column_names[range_index],
                "domain": {
                    "low": float(all_edges[range_index][levels[0]][0]),
                    "high": float(all_edges[range_index][levels[0]][-1]),
                },
                "virtual_label_count": int(sum(levels)),
                "levels": range_levels,
                "base_stats": range_stat,
            }
        )
        running_label_offset += sum(levels)

    return {
        "version": 2,
        "artifact_kind": "merged_multi_range_virtual_labels",
        "dataset_size": int(stats[0]["rows"]) if stats else 0,
        "range_count": len(ranges),
        "enum_label_count": 0,
        "virtual_label_count": int(len(ranges) * sum(levels)),
        "merged_virtual_label_count": int(len(ranges) * sum(levels)),
        "merged_label_count": int(len(ranges) * sum(levels)),
        "edge_policy": edge_policy,
        "levels": [int(level) for level in levels],
        "source": {
            "source_base_range_fbin": str(source_path),
            "base_range_fbin": str(base_range_path),
            "virtual_spmat": str(spmat_path),
        },
        "ranges": ranges,
    }


def write_json_atomic(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    tmp = prepare_tmp(path, overwrite)
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    publish_tmp(tmp, path)


def write_spmat(
    source: np.memmap,
    output_path: Path,
    selected_columns: list[int],
    levels: list[int],
    all_edges: list[dict[int, np.ndarray]],
    chunk_rows: int,
    overwrite: bool,
) -> None:
    rows = int(source.shape[0])
    range_count = len(selected_columns)
    labels_per_row = range_count * len(levels)
    label_count_per_range = sum(levels)
    total_label_count = range_count * label_count_per_range
    nnz = rows * labels_per_row
    if total_label_count > np.iinfo(SPMAT_INDEX_DTYPE).max:
        raise ValueError("spmat column count exceeds int32 index range")

    level_offsets = np.asarray([sum(levels[:idx]) for idx in range(len(levels))], dtype=np.int64)
    range_offsets = np.asarray(
        [range_idx * label_count_per_range for range_idx in range(range_count)],
        dtype=np.int64,
    )

    tmp = prepare_tmp(output_path, overwrite)
    start_time = time.monotonic()
    with tmp.open("wb") as handle:
        np.asarray([rows, total_label_count, nnz], dtype=SPMAT_HEADER_DTYPE).tofile(handle)

        indptr_chunk = max(1, min(5_000_000, chunk_rows * 5))
        for begin in range(0, rows + 1, indptr_chunk):
            end = min(rows + 1, begin + indptr_chunk)
            indptr = np.arange(begin, end, dtype=SPMAT_HEADER_DTYPE) * labels_per_row
            indptr.tofile(handle)

        for begin in range(0, rows, chunk_rows):
            end = min(rows, begin + chunk_rows)
            values = np.asarray(source[begin:end, selected_columns], dtype=np.float64)
            out = np.empty((end - begin, labels_per_row), dtype=SPMAT_INDEX_DTYPE)
            out_col = 0
            for range_pos in range(range_count):
                range_values = values[:, range_pos]
                for level_pos, num_bins in enumerate(levels):
                    edges = all_edges[range_pos][num_bins]
                    bins = np.searchsorted(edges, range_values, side="right") - 1
                    bins = np.clip(bins, 0, num_bins - 1).astype(np.int64, copy=False)
                    labels = range_offsets[range_pos] + level_offsets[level_pos] + bins
                    out[:, out_col] = labels.astype(SPMAT_INDEX_DTYPE, copy=False)
                    out_col += 1
            out.reshape(-1).tofile(handle)
            if end == rows or end % (10 * chunk_rows) == 0:
                elapsed = time.monotonic() - start_time
                print(f"[spmat:index] rows {end}/{rows} ({end / rows:.1%}) elapsed={elapsed:.1f}s")

        data_chunk = max(1, min(50_000_000, chunk_rows * labels_per_row * 10))
        remaining = nnz
        ones = np.ones(data_chunk, dtype=SPMAT_DATA_DTYPE)
        written = 0
        while remaining > 0:
            take = min(remaining, data_chunk)
            ones[:take].tofile(handle)
            remaining -= take
            written += take
            if remaining == 0 or written % (data_chunk * 10) == 0:
                elapsed = time.monotonic() - start_time
                print(f"[spmat:data] nnz {written}/{nnz} ({written / nnz:.1%}) elapsed={elapsed:.1f}s")

    publish_tmp(tmp, output_path)


def summarize_buckets(
    range_names: list[str],
    levels: list[int],
    all_counts: list[dict[int, np.ndarray]],
) -> list[dict[str, Any]]:
    out = []
    for range_index, name in enumerate(range_names):
        for num_bins in levels:
            counts = all_counts[range_index][num_bins]
            nonzero = counts[counts > 0]
            out.append(
                {
                    "range_index": int(range_index),
                    "name": name,
                    "num_bins": int(num_bins),
                    "zero_bins": int(np.count_nonzero(counts == 0)),
                    "min_nonzero_rows": int(nonzero.min()) if nonzero.size else 0,
                    "max_rows": int(counts.max()) if counts.size else 0,
                    "mean_rows": float(counts.mean()) if counts.size else 0.0,
                    "p99_rows": float(np.quantile(counts, 0.99)) if counts.size else 0.0,
                }
            )
    return out


def validate_no_empty_bins(bucket_summary: list[dict[str, Any]], allow_empty_bins: bool) -> None:
    offenders = [entry for entry in bucket_summary if entry["zero_bins"] > 0]
    if offenders and not allow_empty_bins:
        preview = "; ".join(
            f"{entry['name']} bins={entry['num_bins']} zero={entry['zero_bins']}"
            for entry in offenders[:4]
        )
        raise ValueError(
            "empty range buckets are unsafe for pure-range cover planning; "
            f"pass --allow-empty-bins only for debugging. Offenders: {preview}"
        )


import os
import struct

MAGIC = b"FARPIDX1"

VERSION = 1

HEADER = struct.Struct("<8sIIQQII")

DESCRIPTOR = struct.Struct("<IIQQQQQQQQ")

ALIGNMENT = 64


def align_up(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def write_padding(handle, target_offset: int) -> None:
    current = handle.tell()
    if current > target_offset:
        raise RuntimeError(f"writer passed target offset {target_offset}: {current}")
    if current < target_offset:
        handle.write(b"\0" * (target_offset - current))


def load_dimension(source_root: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.load(source_root / f"{name}.unique_values.f32.npy", mmap_mode="r")
    offsets = np.load(source_root / f"{name}.offsets.u64.npy", mmap_mode="r")
    sample_ids = np.load(source_root / f"{name}.rows_by_value.u32.npy", mmap_mode="r")
    if values.dtype != np.dtype("<f4") or values.ndim != 1 or values.size == 0:
        raise RuntimeError(f"{name}: unique values must be nonempty little-endian float32")
    if offsets.dtype != np.dtype("<u8") or offsets.ndim != 1 or offsets.size != values.size + 1:
        raise RuntimeError(f"{name}: offsets must be little-endian uint64 with unique_count + 1 rows")
    if sample_ids.dtype != np.dtype("<u4") or sample_ids.ndim != 1 or sample_ids.size == 0:
        raise RuntimeError(f"{name}: sample ids must be nonempty little-endian uint32")
    return values, offsets, sample_ids


def validate_dimension(
    name: str,
    values: np.ndarray,
    offsets: np.ndarray,
    sample_ids: np.ndarray,
    chunk_rows: int,
) -> None:
    if not np.all(np.isfinite(values)) or not np.all(values[1:] > values[:-1]):
        raise RuntimeError(f"{name}: unique values are not finite and strictly increasing")
    if int(offsets[0]) != 0 or int(offsets[-1]) != sample_ids.size:
        raise RuntimeError(f"{name}: offsets do not span the sample-id permutation")
    if not np.all(offsets[1:] >= offsets[:-1]):
        raise RuntimeError(f"{name}: offsets are not monotonic")
    for begin in range(0, sample_ids.size, chunk_rows):
        chunk = sample_ids[begin : begin + chunk_rows]
        if chunk.size and int(chunk.max()) >= sample_ids.size:
            raise RuntimeError(f"{name}: sample id outside [0, {sample_ids.size})")


def add_hierarchy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-base-range-fbin", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-range-output", type=Path, default=None)
    parser.add_argument("--virtual-spmat-output", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--columns", type=str, default="0,1")
    parser.add_argument("--range-names", type=str, default="width,height")
    parser.add_argument("--source-column-names", type=str, default="width,height")
    parser.add_argument("--levels", type=str, default="256")
    parser.add_argument(
        "--edge-policy",
        choices=("balanced_unique", "uniform"),
        default="balanced_unique",
        help="Bucket edge construction policy. balanced_unique creates non-empty observed-value buckets.",
    )
    parser.add_argument(
        "--allow-empty-bins",
        action="store_true",
        help="Allow zero-row buckets. Usually unsafe for pure-range routing; mainly for debugging.",
    )
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")


def add_postings_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--dimensions",
        type=str,
        default="width,height",
        help="Comma-separated dimension names in runtime range-index order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validation-chunk-rows", type=int, default=8_000_000)


def hierarchy(args: argparse.Namespace) -> None:
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    levels = parse_levels(args.levels)
    selected_columns = csv_ints(args.columns)
    range_names = csv_names(args.range_names)
    source_column_names = csv_names(args.source_column_names)
    if not (len(selected_columns) == len(range_names) == len(source_column_names)):
        raise ValueError("--columns, --range-names, and --source-column-names must have the same length")

    output_root = args.output_root
    base_range_output = default_path(output_root, args.base_range_output, "base_range.fbin")
    spmat_output = default_path(
        output_root,
        args.virtual_spmat_output,
        "base.virtual.spmat",
    )
    manifest_output = default_path(
        output_root,
        args.manifest_output,
        "manifest.json",
    )
    summary_output = default_path(
        output_root,
        args.summary_output,
        "summary.json",
    )

    rows, cols = inspect_fbin(args.source_base_range_fbin)
    for column in selected_columns:
        if column < 0 or column >= cols:
            raise ValueError(f"selected column {column} is outside source fbin column count {cols}")
    source = open_fbin_payload(args.source_base_range_fbin, rows, cols)

    print(
        f"[start] source={args.source_base_range_fbin} rows={rows} cols={cols} "
        f"selected_columns={selected_columns} levels={levels} edge_policy={args.edge_policy}"
    )
    stats, value_counts = collect_stats_and_write_base(
        source=source,
        source_path=args.source_base_range_fbin,
        output_path=base_range_output,
        selected_columns=selected_columns,
        range_names=range_names,
        edge_policy=args.edge_policy,
        chunk_rows=args.chunk_rows,
        overwrite=args.overwrite,
    )

    all_edges = []
    all_counts = []
    for range_pos, range_stat in enumerate(stats):
        edges_by_level, counts_by_level = build_range_edges(
            range_stat,
            value_counts[range_pos],
            levels,
            args.edge_policy,
        )
        all_edges.append(edges_by_level)
        all_counts.append(counts_by_level)

    bucket_summary = summarize_buckets(range_names, levels, all_counts)
    validate_no_empty_bins(bucket_summary, args.allow_empty_bins)

    write_spmat(
        source=source,
        output_path=spmat_output,
        selected_columns=selected_columns,
        levels=levels,
        all_edges=all_edges,
        chunk_rows=args.chunk_rows,
        overwrite=args.overwrite,
    )

    manifest = build_manifest(
        source_path=args.source_base_range_fbin,
        base_range_path=base_range_output,
        spmat_path=spmat_output,
        stats=stats,
        selected_columns=selected_columns,
        range_names=range_names,
        source_column_names=source_column_names,
        levels=levels,
        all_edges=all_edges,
        all_counts=all_counts,
        edge_policy=args.edge_policy,
    )
    write_json_atomic(manifest_output, manifest, args.overwrite)

    summary = {
        "source_base_range_fbin": str(args.source_base_range_fbin),
        "base_range_fbin": str(base_range_output),
        "virtual_spmat": str(spmat_output),
        "manifest": str(manifest_output),
        "rows": rows,
        "range_count": len(selected_columns),
        "levels": levels,
        "edge_policy": args.edge_policy,
        "labels_per_row": len(selected_columns) * len(levels),
        "spmat": {
            "rows": rows,
            "cols": len(selected_columns) * sum(levels),
            "nnz": rows * len(selected_columns) * len(levels),
            "bytes": spmat_output.stat().st_size,
        },
        "base_range": {
            "rows": rows,
            "cols": len(selected_columns),
            "bytes": base_range_output.stat().st_size,
        },
        "range_stats": stats,
        "bucket_summary": bucket_summary,
    }
    write_json_atomic(summary_output, summary, args.overwrite)

    print(f"Saved base ranges: {base_range_output}")
    print(f"Saved virtual labels: {spmat_output}")
    print(f"Saved manifest: {manifest_output}")
    print(f"Saved summary: {summary_output}")
    print(json.dumps({"bucket_summary": bucket_summary, "spmat": summary["spmat"]}, indent=2))


def postings(args: argparse.Namespace) -> None:
    names = [value.strip() for value in args.dimensions.split(",") if value.strip()]
    if not names or len(set(names)) != len(names):
        raise RuntimeError("--dimensions must contain unique nonempty names")
    if args.validation_chunk_rows <= 0:
        raise RuntimeError("--validation-chunk-rows must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output}")

    dimensions = []
    row_count: int | None = None
    for name in names:
        values, offsets, sample_ids = load_dimension(args.source_root, name)
        validate_dimension(name, values, offsets, sample_ids, args.validation_chunk_rows)
        if row_count is None:
            row_count = int(sample_ids.size)
        elif sample_ids.size != row_count:
            raise RuntimeError(f"{name}: row count differs from the first dimension")
        dimensions.append((name, values, offsets, sample_ids))
    assert row_count is not None

    header_bytes = HEADER.size + DESCRIPTOR.size * len(dimensions)
    cursor = align_up(header_bytes)
    descriptors = []
    for range_index, (_, values, offsets, sample_ids) in enumerate(dimensions):
        values_offset = cursor
        values_bytes = int(values.nbytes)
        cursor = align_up(cursor + values_bytes)
        offsets_offset = cursor
        offsets_bytes = int(offsets.nbytes)
        cursor = align_up(cursor + offsets_bytes)
        sample_ids_offset = cursor
        sample_ids_bytes = int(sample_ids.nbytes)
        cursor = align_up(cursor + sample_ids_bytes)
        descriptors.append(
            (
                range_index,
                0,
                int(values.size),
                row_count,
                values_offset,
                offsets_offset,
                sample_ids_offset,
                values_bytes,
                offsets_bytes,
                sample_ids_bytes,
            )
        )
    file_bytes = cursor

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_name(args.output.name + ".tmp")
    try:
        with temp_path.open("wb") as handle:
            handle.write(
                HEADER.pack(
                    MAGIC,
                    VERSION,
                    header_bytes,
                    file_bytes,
                    row_count,
                    len(dimensions),
                    0,
                )
            )
            for descriptor in descriptors:
                handle.write(DESCRIPTOR.pack(*descriptor))
            for (_, values, offsets, sample_ids), descriptor in zip(dimensions, descriptors):
                write_padding(handle, descriptor[4])
                values.tofile(handle)
                write_padding(handle, descriptor[5])
                offsets.tofile(handle)
                write_padding(handle, descriptor[6])
                sample_ids.tofile(handle)
            write_padding(handle, file_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, args.output)
    finally:
        temp_path.unlink(missing_ok=True)

    summary = {
        "format": "flashanns_range_posting_index",
        "version": VERSION,
        "path": str(args.output.resolve()),
        "file_bytes": file_bytes,
        "row_count": row_count,
        "range_count": len(dimensions),
        "dimensions": [
            {
                "range_index": range_index,
                "name": name,
                "unique_count": int(values.size),
                "row_count": int(sample_ids.size),
            }
            for range_index, (name, values, _, sample_ids) in enumerate(dimensions)
        ],
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build range hierarchies and ordered posting indexes.")
    commands = parser.add_subparsers(dest="command", required=True)
    hierarchy_parser = commands.add_parser("hierarchy", help="Build range hierarchy labels and metadata")
    add_hierarchy_arguments(hierarchy_parser)
    hierarchy_parser.set_defaults(run=hierarchy)
    postings_parser = commands.add_parser("postings", help="Serialize ordered range value postings")
    add_postings_arguments(postings_parser)
    postings_parser.set_defaults(run=postings)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
