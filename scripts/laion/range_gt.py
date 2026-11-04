#!/usr/bin/env python3
"""Generate exact LAION width/height pure-range ground truth.

This workload has empty query spmat rows and stores the true predicates in
query_range.bin.  The script builds a compact value-to-row index for the two
range dimensions, enumerates exact 2D range candidates, then computes L2 top-k
over hvecs with PyTorch.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


from scripts.laion.categorical_gt import (
    UINT32_MAX,
    hvecs_records,
    inspect_hvecs,
    worker_compute_topk,
    write_dist_fbin,
    write_gt_ibin,
)


DEFAULT_DATA_ROOT = Path("data/laion100m")
DEFAULT_BASE_HVECS = DEFAULT_DATA_ROOT / "base.hvecs"
DEFAULT_BASE_RANGE = DEFAULT_DATA_ROOT / "range/base_range.fbin"

FBIN_HEADER_DTYPE = np.dtype("<u4")
FBIN_VALUE_DTYPE = np.dtype("<f4")
QUERY_RANGE_DTYPE = np.dtype(
    [
        ("mode", np.uint8),
        ("upper", np.float32),
        ("lower", np.float32),
    ],
    align=True,
)
QUERY_RANGE_PACKED_DTYPE = np.dtype(
    [
        ("mode", np.uint8),
        ("upper", np.float32),
        ("lower", np.float32),
    ],
    align=False,
)

MODE_NO_RANGE = 0
MODE_GT = 1
MODE_LT = 2
MODE_BETWEEN = 3


@dataclass(frozen=True)
class FbinInfo:
    rows: int
    cols: int


@dataclass
class RangeValueIndex:
    dim: int
    name: str
    values: np.ndarray
    offsets: np.ndarray
    rows_by_value: np.ndarray

    def bounds(self, query_record: np.void) -> tuple[int, int]:
        mode = int(query_record["mode"])
        lower = np.float32(query_record["lower"])
        upper = np.float32(query_record["upper"])
        if mode == MODE_NO_RANGE:
            return 0, int(self.values.size)
        if mode == MODE_GT:
            return int(np.searchsorted(self.values, lower, side="right")), int(self.values.size)
        if mode == MODE_LT:
            return 0, int(np.searchsorted(self.values, upper, side="left"))
        if mode == MODE_BETWEEN:
            return (
                int(np.searchsorted(self.values, lower, side="left")),
                int(np.searchsorted(self.values, upper, side="left")),
            )
        raise ValueError(f"unsupported range mode={mode}")

    def estimate_count(self, query_record: np.void) -> int:
        lo, hi = self.bounds(query_record)
        if hi <= lo:
            return 0
        return int(self.offsets[hi] - self.offsets[lo])

    def rows_for(self, query_record: np.void) -> np.ndarray:
        lo, hi = self.bounds(query_record)
        if hi <= lo:
            return np.empty(0, dtype=np.uint32)
        begin = int(self.offsets[lo])
        end = int(self.offsets[hi])
        return np.asarray(self.rows_by_value[begin:end], dtype=np.uint32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-hvecs", type=Path, default=DEFAULT_BASE_HVECS)
    parser.add_argument("--query-hvecs", type=Path, required=True)
    parser.add_argument("--base-range-fbin", type=Path, default=DEFAULT_BASE_RANGE)
    parser.add_argument("--query-range-bin", type=Path, required=True)
    parser.add_argument("--search-results-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=Path("artifacts/laion100m/range/value_index"))
    parser.add_argument("--tag", type=str, default="gt.range")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--recall-k", type=int, default=10)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--row-batch", type=int, default=131_072)
    parser.add_argument("--query-batch", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--skip-recall",
        action="store_true",
        help="Generate exact GT without reading a search-results CSV or reporting recall.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_recall_inputs(args: argparse.Namespace) -> None:
    if args.plan_only or args.skip_recall or args.search_results_csv is None:
        return
    if not args.search_results_csv.exists():
        raise FileNotFoundError(args.search_results_csv)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output_path(path: Path, overwrite: bool) -> None:
    ensure_parent(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    ensure_parent(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)


def inspect_fbin(path: Path) -> FbinInfo:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=FBIN_HEADER_DTYPE, count=2)
    if header.size != 2:
        raise ValueError(f"invalid fbin header in {path}")
    rows, cols = [int(value) for value in header.tolist()]
    expected_size = 8 + rows * cols * FBIN_VALUE_DTYPE.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path} size={actual_size} does not match fbin shape {(rows, cols)}")
    return FbinInfo(rows=rows, cols=cols)


def open_fbin_payload(path: Path, info: FbinInfo) -> np.memmap:
    return np.memmap(path, dtype=FBIN_VALUE_DTYPE, mode="r", offset=8, shape=(info.rows, info.cols))


def read_query_range_bin(path: Path) -> np.ndarray:
    def decode_records(payload: bytes, query_count: int, range_count: int) -> np.ndarray:
        total_records = query_count * range_count
        aligned_bytes = total_records * QUERY_RANGE_DTYPE.itemsize
        packed_bytes = total_records * QUERY_RANGE_PACKED_DTYPE.itemsize
        if len(payload) == aligned_bytes:
            data = np.frombuffer(payload, dtype=QUERY_RANGE_DTYPE, count=total_records).copy()
            return data.reshape(query_count, range_count)
        if len(payload) == packed_bytes:
            packed = np.frombuffer(payload, dtype=QUERY_RANGE_PACKED_DTYPE, count=total_records)
            data = np.empty(total_records, dtype=QUERY_RANGE_DTYPE)
            data["mode"] = packed["mode"]
            data["upper"] = packed["upper"]
            data["lower"] = packed["lower"]
            return data.reshape(query_count, range_count)
        raise ValueError(
            f"query range payload size mismatch in {path}: "
            f"query_count={query_count} range_count={range_count} payload={len(payload)}"
        )

    with path.open("rb") as handle:
        first = np.fromfile(handle, dtype=np.dtype("<u4"), count=1)
        if first.size != 1:
            raise ValueError(f"invalid query range header in {path}")
        query_count = int(first[0])
        payload = handle.read()

    legacy_aligned = query_count * QUERY_RANGE_DTYPE.itemsize
    legacy_packed = query_count * QUERY_RANGE_PACKED_DTYPE.itemsize
    if len(payload) in (legacy_aligned, legacy_packed):
        return decode_records(payload, query_count=query_count, range_count=1)
    if len(payload) < 4:
        raise ValueError(f"truncated query range payload in {path}")
    range_count = int(np.frombuffer(payload[:4], dtype=np.dtype("<u4"), count=1)[0])
    if range_count <= 0:
        raise ValueError(f"invalid range_count={range_count} in {path}")
    return decode_records(payload[4:], query_count=query_count, range_count=range_count)


def build_range_value_index(
    base_ranges: np.memmap,
    dim: int,
    name: str,
    index_root: Path,
    rebuild: bool,
) -> RangeValueIndex:
    values_path = index_root / f"{name}.unique_values.f32.npy"
    offsets_path = index_root / f"{name}.offsets.u64.npy"
    rows_path = index_root / f"{name}.rows_by_value.u32.npy"
    if not rebuild and values_path.exists() and offsets_path.exists() and rows_path.exists():
        return RangeValueIndex(
            dim=dim,
            name=name,
            values=np.load(values_path),
            offsets=np.load(offsets_path),
            rows_by_value=np.load(rows_path, mmap_mode="r"),
        )

    started = time.monotonic()
    print(f"[index] build dim={dim} name={name} rows={base_ranges.shape[0]}", flush=True)
    dim_values = base_ranges[:, dim]
    order = np.argsort(dim_values, kind="stable")
    sorted_values = np.asarray(dim_values[order], dtype=np.float32)
    if sorted_values.size == 0:
        raise ValueError("cannot build range index for empty base ranges")
    change = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), change.astype(np.int64, copy=False)))
    ends = np.concatenate((change.astype(np.int64, copy=False), np.asarray([sorted_values.size], dtype=np.int64)))
    unique_values = np.asarray(sorted_values[starts], dtype=np.float32)
    offsets = np.concatenate((starts, np.asarray([sorted_values.size], dtype=np.int64))).astype(np.uint64)
    rows_by_value = np.asarray(order, dtype=np.uint32)

    index_root.mkdir(parents=True, exist_ok=True)
    save_npy_atomic(values_path, unique_values)
    save_npy_atomic(offsets_path, offsets)
    save_npy_atomic(rows_path, rows_by_value)
    elapsed = time.monotonic() - started
    print(
        f"[index] wrote name={name} unique_values={unique_values.size} "
        f"rows={rows_by_value.size} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return RangeValueIndex(
        dim=dim,
        name=name,
        values=unique_values,
        offsets=offsets,
        rows_by_value=np.load(rows_path, mmap_mode="r"),
    )


def float32_bits(value: Any) -> int:
    return int(np.asarray(np.float32(value), dtype=np.float32).view(np.uint32))


def query_key(query_ranges: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            int(record["mode"]),
            float32_bits(record["lower"]),
            float32_bits(record["upper"]),
        )
        for record in query_ranges
    )


def value_matches(record: np.void, values: np.ndarray) -> np.ndarray:
    mode = int(record["mode"])
    if mode == MODE_NO_RANGE:
        return np.ones(values.shape[0], dtype=bool)
    lower = np.float32(record["lower"])
    upper = np.float32(record["upper"])
    if mode == MODE_GT:
        return values > lower
    if mode == MODE_LT:
        return values < upper
    if mode == MODE_BETWEEN:
        return (values >= lower) & (values < upper)
    raise ValueError(f"unsupported range mode={mode}")


def candidate_rows_for_query(
    query_ranges: np.ndarray,
    indexes: list[RangeValueIndex],
    base_ranges: np.memmap,
) -> tuple[np.ndarray, dict[str, Any]]:
    estimates = [index.estimate_count(query_ranges[index.dim]) for index in indexes]
    chosen_pos = int(np.argmin(np.asarray(estimates, dtype=np.int64)))
    chosen = indexes[chosen_pos]
    rows = chosen.rows_for(query_ranges[chosen.dim])
    before_other = int(rows.size)
    if rows.size == 0:
        return rows, {
            "chosen_dim": chosen.dim,
            "estimate_counts": estimates,
            "one_dim_rows": 0,
            "candidate_rows": 0,
        }

    for index in indexes:
        if index.dim == chosen.dim:
            continue
        record = query_ranges[index.dim]
        if int(record["mode"]) == MODE_NO_RANGE:
            continue
        other_values = np.asarray(base_ranges[rows.astype(np.int64, copy=False), index.dim], dtype=np.float32)
        rows = rows[value_matches(record, other_values)]
        if rows.size == 0:
            break

    if rows.size > 1:
        rows = np.sort(rows.astype(np.uint32, copy=False), kind="quicksort")
    return rows.astype(np.uint32, copy=False), {
        "chosen_dim": chosen.dim,
        "estimate_counts": estimates,
        "one_dim_rows": before_other,
        "candidate_rows": int(rows.size),
    }


def stats_for_counts(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": int(arr.max()),
        "zero": int(np.count_nonzero(arr == 0)),
        "lt_10": int(np.count_nonzero(arr < 10)),
        "lt_100": int(np.count_nonzero(arr < 100)),
    }


def read_search_csv(path: Path, query_count: int, width: int) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",", dtype=np.uint32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[0] < query_count:
        raise ValueError(f"{path} has {matrix.shape[0]} rows, expected at least {query_count}")
    if matrix.shape[1] < width:
        raise ValueError(f"{path} has width {matrix.shape[1]}, expected at least {width}")
    return np.asarray(matrix[:query_count, :width], dtype=np.uint32)


def compute_recall(
    search_ids: np.ndarray,
    gt_ids: np.ndarray,
    recall_k: int,
) -> dict[str, Any]:
    if search_ids.shape[0] != gt_ids.shape[0]:
        raise ValueError(f"search/GT row mismatch: {search_ids.shape[0]} vs {gt_ids.shape[0]}")
    k = min(int(recall_k), search_ids.shape[1], gt_ids.shape[1])
    hits = np.zeros(search_ids.shape[0], dtype=np.int32)
    gt_valid_counts = np.zeros(search_ids.shape[0], dtype=np.int32)
    pred_valid_counts = np.zeros(search_ids.shape[0], dtype=np.int32)
    for query_idx in range(search_ids.shape[0]):
        gt_valid = [int(value) for value in gt_ids[query_idx, :k].tolist() if int(value) != int(UINT32_MAX)]
        pred_valid = [int(value) for value in search_ids[query_idx, :k].tolist() if int(value) != int(UINT32_MAX)]
        gt_valid_counts[query_idx] = len(gt_valid)
        pred_valid_counts[query_idx] = len(pred_valid)
        if gt_valid and pred_valid:
            hits[query_idx] = len(set(gt_valid).intersection(pred_valid))

    fixed_total = int(search_ids.shape[0] * k)
    valid_total = int(gt_valid_counts.sum())
    nonempty = gt_valid_counts > 0
    per_query_valid = np.full(search_ids.shape[0], np.nan, dtype=np.float64)
    per_query_valid[nonempty] = hits[nonempty] / gt_valid_counts[nonempty]
    return {
        "recall_k": int(k),
        "query_count": int(search_ids.shape[0]),
        "hits": int(hits.sum()),
        "fixed_total": fixed_total,
        "recall_fixed_k_denominator": float(hits.sum() / fixed_total) if fixed_total else math.nan,
        "valid_gt_total": valid_total,
        "recall_valid_gt_denominator": float(hits.sum() / valid_total) if valid_total else math.nan,
        "mean_query_recall_valid_gt": float(np.nanmean(per_query_valid)) if np.any(nonempty) else math.nan,
        "zero_gt_queries": int(np.count_nonzero(gt_valid_counts == 0)),
        "full_gt_topk_queries": int(np.count_nonzero(gt_valid_counts >= k)),
        "pred_valid_count_stats": stats_for_counts(pred_valid_counts),
        "gt_valid_count_stats": stats_for_counts(gt_valid_counts),
        "hit_count_stats": stats_for_counts(hits),
    }


def write_summary(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    import torch
    if args.topk <= 0 or args.recall_k <= 0:
        raise ValueError("--topk and --recall-k must be positive")
    if args.recall_k > args.topk:
        raise ValueError("--recall-k must not exceed --topk")
    if args.row_batch <= 0 or args.query_batch <= 0:
        raise ValueError("--row-batch and --query-batch must be positive")

    for required in (args.base_hvecs, args.query_hvecs, args.base_range_fbin, args.query_range_bin):
        if not required.exists():
            raise FileNotFoundError(required)
    validate_recall_inputs(args)

    total_started = time.monotonic()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    base_info = inspect_hvecs(args.base_hvecs)
    query_info = inspect_hvecs(args.query_hvecs)
    range_info = inspect_fbin(args.base_range_fbin)
    if base_info.rows != range_info.rows:
        raise ValueError(f"base rows {base_info.rows} != range rows {range_info.rows}")
    if base_info.dim != query_info.dim:
        raise ValueError(f"base/query dim mismatch: {base_info.dim} vs {query_info.dim}")
    if range_info.cols != 2:
        raise ValueError(f"expected width/height range_count=2, got {range_info.cols}")

    query_ranges = read_query_range_bin(args.query_range_bin)
    query_count = min(query_info.rows, query_ranges.shape[0])
    if args.query_limit > 0:
        query_count = min(query_count, int(args.query_limit))
    if query_count <= 0:
        raise ValueError("no queries selected")
    query_ranges = query_ranges[:query_count].copy()
    if query_ranges.shape[1] != range_info.cols:
        raise ValueError(f"query/base range dim mismatch: {query_ranges.shape[1]} vs {range_info.cols}")

    base_ranges = open_fbin_payload(args.base_range_fbin, range_info)
    indexes = [
        build_range_value_index(base_ranges, 0, "width", args.index_root, args.rebuild_index),
        build_range_value_index(base_ranges, 1, "height", args.index_root, args.rebuild_index),
    ]

    groups: dict[tuple[tuple[int, int, int], ...], list[int]] = defaultdict(list)
    for query_idx in range(query_count):
        groups[query_key(query_ranges[query_idx])].append(query_idx)
    group_items = sorted(groups.items(), key=lambda item: item[1][0])
    print(
        f"[gt] queries={query_count} unique_range_predicates={len(group_items)} "
        f"topk={args.topk} plan_only={args.plan_only}",
        flush=True,
    )

    final_ids = np.full((query_count, args.topk), UINT32_MAX, dtype=np.uint32)
    final_dist = np.full((query_count, args.topk), np.inf, dtype=np.float32)
    candidate_counts = np.zeros(query_count, dtype=np.int64)
    one_dim_counts = np.zeros(query_count, dtype=np.int64)
    chosen_dims = np.zeros(query_count, dtype=np.int8)
    group_candidate_cache: dict[tuple[tuple[int, int, int], ...], int] = {}

    device = torch.device(args.device)
    if not args.plan_only:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        base_records = hvecs_records(args.base_hvecs, base_info)
        query_records = hvecs_records(args.query_hvecs, query_info)
    else:
        base_records = None
        query_records = None

    candidate_started = time.monotonic()
    topk_seconds = 0.0
    for group_idx, (key, qid_list) in enumerate(group_items, start=1):
        first_qid = qid_list[0]
        rows, candidate_info = candidate_rows_for_query(query_ranges[first_qid], indexes, base_ranges)
        qids = np.asarray(qid_list, dtype=np.int64)
        candidate_counts[qids] = int(rows.size)
        one_dim_counts[qids] = int(candidate_info["one_dim_rows"])
        chosen_dims[qids] = int(candidate_info["chosen_dim"])
        group_candidate_cache[key] = int(rows.size)

        if not args.plan_only and rows.size > 0:
            assert base_records is not None and query_records is not None
            topk_started = time.monotonic()
            ids, dist = worker_compute_topk(
                base_records,
                query_records,
                rows,
                qids,
                args.topk,
                device,
                args.row_batch,
                args.query_batch,
            )
            topk_seconds += time.monotonic() - topk_started
            final_ids[qids] = ids
            final_dist[qids] = dist

        if (
            group_idx == 1
            or group_idx == len(group_items)
            or (args.progress_every > 0 and group_idx % args.progress_every == 0)
        ):
            elapsed = time.monotonic() - candidate_started
            done_queries = sum(len(item[1]) for item in group_items[:group_idx])
            print(
                f"[gt] group {group_idx}/{len(group_items)} queries_done={done_queries}/{query_count} "
                f"last_q={first_qid} reuse={len(qid_list)} candidates={rows.size} "
                f"elapsed={elapsed:.1f}s topk={topk_seconds:.1f}s",
                flush=True,
            )

    candidate_path = output_root / f"{args.tag}.candidate_counts.npy"
    chosen_path = output_root / f"{args.tag}.chosen_dims.npy"
    if candidate_path.exists() and args.overwrite:
        candidate_path.unlink()
    if chosen_path.exists() and args.overwrite:
        chosen_path.unlink()
    if not candidate_path.exists():
        save_npy_atomic(candidate_path, candidate_counts)
    if not chosen_path.exists():
        save_npy_atomic(chosen_path, chosen_dims)

    outputs: dict[str, Any] = {
        "candidate_counts": str(candidate_path),
        "chosen_dims": str(chosen_path),
    }
    recall_summary: dict[str, Any] | None = None
    if not args.plan_only:
        top100_path = output_root / f"{args.tag}.top{args.topk}.ibin"
        top10_path = output_root / f"{args.tag}.top{args.recall_k}.ibin"
        dist100_path = output_root / f"{args.tag}.top{args.topk}.dist.fbin"
        dist10_path = output_root / f"{args.tag}.top{args.recall_k}.dist.fbin"
        outputs["topk_gt"] = write_gt_ibin(top100_path, final_ids, args.overwrite)
        outputs["topk_distances"] = write_dist_fbin(dist100_path, final_dist, args.overwrite)
        if args.recall_k == args.topk:
            outputs["recall_k_gt"] = outputs["topk_gt"]
            outputs["recall_k_distances"] = outputs["topk_distances"]
        else:
            outputs["recall_k_gt"] = write_gt_ibin(top10_path, final_ids[:, : args.recall_k], args.overwrite)
            outputs["recall_k_distances"] = write_dist_fbin(dist10_path, final_dist[:, : args.recall_k], args.overwrite)

        if not args.skip_recall and args.search_results_csv is not None:
            search_ids = read_search_csv(args.search_results_csv, query_count, args.recall_k)
            recall_summary = compute_recall(search_ids, final_ids[:, : args.recall_k], args.recall_k)
            recall_path = output_root / f"{args.tag}.recall_at_{args.recall_k}.json"
            write_summary(recall_path, recall_summary, args.overwrite)
            outputs["recall"] = str(recall_path)

    modes_by_dim = []
    for dim in range(query_ranges.shape[1]):
        modes_by_dim.append({str(int(k)): int(v) for k, v in Counter(query_ranges[:, dim]["mode"].tolist()).items()})
    summary = {
        "source": {
            "base_hvecs": str(args.base_hvecs),
            "query_hvecs": str(args.query_hvecs),
            "base_range_fbin": str(args.base_range_fbin),
            "query_range_bin": str(args.query_range_bin),
            "search_results_csv": (
                None if args.skip_recall or args.search_results_csv is None else str(args.search_results_csv)
            ),
        },
        "outputs": outputs,
        "query_count": int(query_count),
        "unique_range_predicates": int(len(group_items)),
        "range_modes_by_dim": modes_by_dim,
        "candidate_stats": stats_for_counts(candidate_counts),
        "one_dim_candidate_stats": stats_for_counts(one_dim_counts),
        "chosen_dim_counts": {str(int(k)): int(v) for k, v in Counter(chosen_dims.tolist()).items()},
        "gt_stats": {
            "valid_topk_stats": stats_for_counts(np.count_nonzero(final_ids != UINT32_MAX, axis=1))
            if not args.plan_only
            else None,
            "finite_distance_count": int(np.isfinite(final_dist).sum()) if not args.plan_only else None,
        },
        "recall": recall_summary,
        "config": {
            "tag": args.tag,
            "topk": int(args.topk),
            "recall_k": int(args.recall_k),
            "row_batch": int(args.row_batch),
            "query_batch": int(args.query_batch),
            "device": args.device,
            "plan_only": bool(args.plan_only),
            "skip_recall": bool(args.skip_recall),
        },
        "timings": {
            "candidate_and_gt_seconds": float(time.monotonic() - candidate_started),
            "topk_seconds": float(topk_seconds),
            "total_seconds": float(time.monotonic() - total_started),
        },
    }
    summary_path = output_root / f"{args.tag}.summary.json"
    write_summary(summary_path, summary, args.overwrite)
    print(f"[gt] wrote summary {summary_path}", flush=True)
    if recall_summary is not None:
        print(
            f"[recall] R@{recall_summary['recall_k']} fixed={recall_summary['recall_fixed_k_denominator']:.6f} "
            f"valid_gt={recall_summary['recall_valid_gt_denominator']:.6f} "
            f"hits={recall_summary['hits']}/{recall_summary['valid_gt_total']} valid_gt",
            flush=True,
        )
    print(f"[gt] total_seconds={summary['timings']['total_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
