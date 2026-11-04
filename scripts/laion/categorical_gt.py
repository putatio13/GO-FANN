#!/usr/bin/env python3
"""Generate exact LAION categorical ground truth with predicate reuse.

The categorical workload has many query instances that share the same
predicate. This script materializes the categorical-only query vector/manifest
files, builds postings only for categorical tokens used by the queries, and
computes exact L2 top-k within the predicate candidate set.

For speed, single and OR predicates are token-centric:

- single(label) uses top-k over that label posting.
- or(label_a, label_b) merges top-k(label_a) and top-k(label_b) for the same
  query, which is exact for a union.

AND predicates are handled per unique predicate by intersecting posting lists
and evaluating all query instances that share that predicate in one GPU batch.
"""

from __future__ import annotations

import argparse
import heapq
import json
import multiprocessing as mp
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


from scripts.laion.base import HvecsInfo, hvecs_records, inspect_hvecs, SpmatInfo, inspect_spmat

DEFAULT_DATA_ROOT = Path("data/laion100m")
DEFAULT_BASE_HVECS = DEFAULT_DATA_ROOT / "base.hvecs"

SPMAT_HEADER_DTYPE = np.dtype("<i8")
SPMAT_INDEX_DTYPE = np.dtype("<i4")
HVECS_DIM_DTYPE = np.dtype("<u4")
HVECS_VALUE_DTYPE = np.dtype("<f2")
GT_DTYPE = np.dtype("<u4")
DIST_DTYPE = np.dtype("<f4")
UINT32_MAX = np.uint32(0xFFFFFFFF)

QUERY_RANGE_DTYPE = np.dtype(
    [
        ("mode", np.uint8),
        ("upper", np.float32),
        ("lower", np.float32),
    ],
    align=True,
)


@dataclass
class QueryRecord:
    local_idx: int
    full_idx: int
    subtype: str
    token_ids: tuple[int, ...]
    predicate_key: str
    manifest: dict[str, Any]
    candidate_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--full-query-hvecs", type=Path, required=True)
    parser.add_argument("--base-hvecs", type=Path, default=DEFAULT_BASE_HVECS)
    parser.add_argument("--data-label-spmat", type=Path, default=DEFAULT_DATA_ROOT / "base.spmat")
    parser.add_argument("--base-range-fbin", type=Path, default=DEFAULT_DATA_ROOT / "range/base_range.fbin")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Comma-separated CUDA device ids. auto uses all visible GPUs, preferring ids >=2 first.",
    )
    parser.add_argument("--spmat-row-batch", type=int, default=1_000_000)
    parser.add_argument("--row-batch", type=int, default=131_072)
    parser.add_argument("--query-batch", type=int, default=64)
    parser.add_argument("--limit-per-subtype", type=int, default=0, help="Debug/smoke cap per categorical subtype.")
    parser.add_argument("--max-and-predicates", type=int, default=0, help="Debug/smoke cap for unique AND predicates.")
    parser.add_argument("--reuse-postings", action="store_true", help="Reuse existing posting .npy files when present.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output_path(path: Path, overwrite: bool) -> None:
    ensure_parent(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")


def write_hvecs(path: Path, matrix: np.ndarray, overwrite: bool) -> dict[str, Any]:
    if matrix.ndim != 2:
        raise ValueError("hvecs matrix must be 2D")
    ensure_output_path(path, overwrite)
    dim = int(matrix.shape[1])
    records = np.empty(matrix.shape[0], dtype=[("dim", "<u4"), ("vec", "<f2", (dim,))])
    records["dim"] = dim
    records["vec"] = matrix.astype(np.float16, copy=False)
    records.tofile(path)
    return {
        "path": str(path),
        "rows": int(matrix.shape[0]),
        "dim": dim,
        "bytes": path.stat().st_size,
        "format": "hvecs",
        "payload_dtype": "float16",
    }


def write_gt_ibin(path: Path, matrix: np.ndarray, overwrite: bool) -> dict[str, Any]:
    if matrix.ndim != 2:
        raise ValueError("GT matrix must be 2D")
    ensure_output_path(path, overwrite)
    with path.open("wb") as handle:
        np.asarray([matrix.shape[0], matrix.shape[1]], dtype=GT_DTYPE).tofile(handle)
        matrix.astype(GT_DTYPE, copy=False).tofile(handle)
    return {"path": str(path), "rows": int(matrix.shape[0]), "topk": int(matrix.shape[1]), "bytes": path.stat().st_size}


def write_dist_fbin(path: Path, matrix: np.ndarray, overwrite: bool) -> dict[str, Any]:
    if matrix.ndim != 2:
        raise ValueError("distance matrix must be 2D")
    ensure_output_path(path, overwrite)
    with path.open("wb") as handle:
        np.asarray([matrix.shape[0], matrix.shape[1]], dtype=np.dtype("<u4")).tofile(handle)
        matrix.astype(DIST_DTYPE, copy=False).tofile(handle)
    return {"path": str(path), "rows": int(matrix.shape[0]), "topk": int(matrix.shape[1]), "bytes": path.stat().st_size}


def write_empty_query_range(path: Path, query_count: int, range_count: int, overwrite: bool) -> dict[str, Any]:
    ensure_output_path(path, overwrite)
    ranges = np.empty((query_count, range_count), dtype=QUERY_RANGE_DTYPE)
    ranges["mode"] = np.uint8(0)
    ranges["upper"] = np.float32(0.0)
    ranges["lower"] = np.float32(0.0)
    with path.open("wb") as handle:
        np.asarray([query_count, range_count], dtype=np.dtype("<u4")).tofile(handle)
        ranges.reshape(-1).tofile(handle)
    return {"path": str(path), "rows": query_count, "range_count": range_count, "bytes": path.stat().st_size}


def read_fbin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.dtype("<u4"), count=2)
    if header.size != 2:
        raise ValueError(f"invalid fbin header in {path}")
    return int(header[0]), int(header[1])


def canonical_tokens(subtype: str, predicates: list[dict[str, Any]]) -> tuple[int, ...]:
    token_ids = tuple(int(predicate["token_id"]) for predicate in predicates)
    if subtype in {"and", "or"}:
        return tuple(sorted(token_ids))
    return token_ids


def load_categorical_queries(path: Path, limit_per_subtype: int) -> list[QueryRecord]:
    queries: list[QueryRecord] = []
    per_subtype = Counter()
    with path.open("r", encoding="utf-8") as handle:
        full_idx = 0
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            family = str(record.get("family", "")).lower()
            subtype = str(record.get("subtype", "")).lower()
            if family != "categorical":
                full_idx += 1
                continue
            if subtype not in {"single", "and", "or"}:
                raise ValueError(f"unsupported categorical subtype on line {line_number}: {subtype}")
            if limit_per_subtype > 0 and per_subtype[subtype] >= limit_per_subtype:
                full_idx += 1
                continue
            predicates = record.get("predicates") or []
            if not predicates:
                raise ValueError(f"categorical query line {line_number} has no predicates")
            tokens = canonical_tokens(subtype, predicates)
            if subtype == "single" and len(tokens) != 1:
                raise ValueError(f"single query line {line_number} has {len(tokens)} token ids")
            if subtype in {"and", "or"} and len(tokens) < 2:
                raise ValueError(f"{subtype} query line {line_number} has fewer than 2 token ids")
            try:
                candidate_count = int(record.get("candidate_count", -1))
            except (TypeError, ValueError):
                candidate_count = -1
            queries.append(
                QueryRecord(
                    local_idx=len(queries),
                    full_idx=full_idx,
                    subtype=subtype,
                    token_ids=tokens,
                    predicate_key=str(record.get("predicate_key", "")),
                    manifest=record,
                    candidate_count=candidate_count,
                )
            )
            per_subtype[subtype] += 1
            full_idx += 1
    if not queries:
        raise RuntimeError(f"no categorical queries selected from {path}")
    return queries


def analyze_predicates(queries: list[QueryRecord]) -> dict[str, Any]:
    by_subtype: dict[str, list[QueryRecord]] = defaultdict(list)
    by_predicate: dict[tuple[str, tuple[int, ...]], list[QueryRecord]] = defaultdict(list)
    for query in queries:
        by_subtype[query.subtype].append(query)
        by_predicate[(query.subtype, query.token_ids)].append(query)

    subtype_summary: dict[str, Any] = {}
    for subtype, rows in sorted(by_subtype.items()):
        pred_groups: dict[tuple[int, ...], list[QueryRecord]] = defaultdict(list)
        for query in rows:
            pred_groups[query.token_ids].append(query)
        reuse = [len(group) for group in pred_groups.values()]
        candidate_counts = [
            group[0].candidate_count for group in pred_groups.values() if group[0].candidate_count >= 0
        ]
        subtype_summary[subtype] = {
            "queries": len(rows),
            "unique_predicates": len(pred_groups),
            "reuse_min": min(reuse),
            "reuse_median": float(np.median(np.asarray(reuse, dtype=np.float64))),
            "reuse_max": max(reuse),
            "candidate_count_unique_sum": int(sum(candidate_counts)),
            "candidate_count_query_naive_sum": int(sum(query.candidate_count for query in rows if query.candidate_count >= 0)),
            "candidate_count_min": int(min(candidate_counts)) if candidate_counts else None,
            "candidate_count_median": float(np.median(np.asarray(candidate_counts, dtype=np.float64))) if candidate_counts else None,
            "candidate_count_max": int(max(candidate_counts)) if candidate_counts else None,
            "top_reused_predicates": [
                {
                    "tokens": list(tokens),
                    "reuse": len(group),
                    "candidate_count": int(group[0].candidate_count),
                }
                for tokens, group in sorted(pred_groups.items(), key=lambda item: len(item[1]), reverse=True)[:10]
            ],
        }
    return {
        "query_count": len(queries),
        "selected_by_subtype": dict(Counter(query.subtype for query in queries)),
        "unique_predicates_total": len(by_predicate),
        "by_subtype": subtype_summary,
    }


def build_categorical_query_artifacts(
    queries: list[QueryRecord],
    full_query_path: Path,
    output_query_path: Path,
    output_manifest_path: Path,
    output_range_path: Path,
    range_count: int,
    overwrite: bool,
) -> dict[str, Any]:
    full_info = inspect_hvecs(full_query_path)
    records = hvecs_records(full_query_path, full_info)
    full_indices = np.asarray([query.full_idx for query in queries], dtype=np.int64)
    if np.any(full_indices < 0) or np.any(full_indices >= full_info.rows):
        raise ValueError("categorical query full_idx is outside full query vector file")
    selected_vectors = np.asarray(records["vec"][full_indices], dtype=np.float16)
    query_info = write_hvecs(output_query_path, selected_vectors, overwrite)

    ensure_output_path(output_manifest_path, overwrite)
    with output_manifest_path.open("w", encoding="utf-8") as handle:
        for query in queries:
            record = dict(query.manifest)
            record["flashanns_categorical_query_index"] = query.local_idx
            record["flashanns_full_query_index"] = query.full_idx
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    range_info = write_empty_query_range(output_range_path, len(queries), range_count, overwrite)
    return {
        "query_hvecs": query_info,
        "query_manifest": {
            "path": str(output_manifest_path),
            "rows": len(queries),
            "bytes": output_manifest_path.stat().st_size,
        },
        "query_range": range_info,
    }


def build_token_postings(
    spmat_path: Path,
    spmat: SpmatInfo,
    token_ids: list[int],
    postings_dir: Path,
    row_batch: int,
    reuse_existing: bool,
) -> dict[int, int]:
    postings_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        token: postings_dir.joinpath(f"token_{token}.npy").exists()
        for token in token_ids
    }
    if reuse_existing and all(existing.values()):
        return {token: int(np.load(postings_dir / f"token_{token}.npy", mmap_mode="r").shape[0]) for token in token_ids}

    missing = [token for token in token_ids if not existing[token] or not reuse_existing]
    for token in missing:
        path = postings_dir / f"token_{token}.npy"
        if path.exists():
            path.unlink()

    token_to_local = {token: idx for idx, token in enumerate(token_ids)}
    remap = np.full(spmat.cols, -1, dtype=np.int32)
    remap[np.asarray(token_ids, dtype=np.int64)] = np.arange(len(token_ids), dtype=np.int32)
    chunks: list[list[np.ndarray]] = [[] for _ in token_ids]

    indptr = np.memmap(spmat_path, dtype=SPMAT_HEADER_DTYPE, mode="r", offset=spmat.indptr_offset, shape=(spmat.rows + 1,))
    indices = np.memmap(spmat_path, dtype=SPMAT_INDEX_DTYPE, mode="r", offset=spmat.indices_offset, shape=(spmat.nnz,))
    started = time.monotonic()
    row_step = max(1, int(row_batch))
    for row_begin in range(0, spmat.rows, row_step):
        row_end = min(spmat.rows, row_begin + row_step)
        pos_begin = int(indptr[row_begin])
        pos_end = int(indptr[row_end])
        labels = indices[pos_begin:pos_end]
        local = remap[labels]
        mask = local >= 0
        if np.any(mask):
            row_counts = np.diff(indptr[row_begin:row_end + 1]).astype(np.int32, copy=False)
            row_ids = np.repeat(np.arange(row_begin, row_end, dtype=np.uint32), row_counts)
            matched_rows = row_ids[mask]
            matched_local = local[mask]
            order = np.argsort(matched_local, kind="stable")
            sorted_local = matched_local[order]
            sorted_rows = matched_rows[order]
            split_points = np.flatnonzero(sorted_local[1:] != sorted_local[:-1]) + 1
            starts = np.concatenate((np.asarray([0], dtype=np.int64), split_points))
            ends = np.concatenate((split_points, np.asarray([sorted_local.size], dtype=np.int64)))
            for start, end in zip(starts, ends):
                local_id = int(sorted_local[int(start)])
                chunks[local_id].append(np.asarray(sorted_rows[start:end], dtype=np.uint32))
        if row_end == spmat.rows or row_end % (10 * row_step) == 0:
            elapsed = time.monotonic() - started
            print(
                f"[postings] rows {row_end}/{spmat.rows} "
                f"({row_end / spmat.rows:.1%}) elapsed={elapsed:.1f}s",
                flush=True,
            )

    sizes: dict[int, int] = {}
    for token in token_ids:
        local_id = token_to_local[token]
        if chunks[local_id]:
            values = np.concatenate(chunks[local_id]).astype(np.uint32, copy=False)
        else:
            values = np.empty(0, dtype=np.uint32)
        np.save(postings_dir / f"token_{token}.npy", values)
        sizes[token] = int(values.size)
    return sizes


def resolve_devices(raw: str) -> list[int]:
    import torch
    if raw.strip().lower() != "auto":
        devices = [int(item.strip()) for item in raw.split(",") if item.strip()]
        if not devices:
            raise ValueError("--devices resolved to an empty list")
        return devices
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; exact GT generation needs at least one GPU")
    count = torch.cuda.device_count()
    if count <= 0:
        raise RuntimeError("torch sees zero CUDA devices")
    return list(range(2, count)) + list(range(0, min(2, count)))


def posting_path(postings_dir: Path, token_id: int) -> Path:
    return postings_dir / f"token_{token_id}.npy"


def assign_tasks(
    token_tasks: list[dict[str, Any]],
    and_tasks: list[dict[str, Any]],
    devices: list[int],
) -> list[dict[str, Any]]:
    assignments = [{"device": device, "token_tasks": [], "and_tasks": [], "cost": 0} for device in devices]
    heap = [(0, idx) for idx in range(len(assignments))]
    heapq.heapify(heap)
    all_tasks = [("token", task) for task in token_tasks] + [("and", task) for task in and_tasks]
    all_tasks.sort(key=lambda item: int(item[1]["cost"]), reverse=True)
    for kind, task in all_tasks:
        cost, idx = heapq.heappop(heap)
        if kind == "token":
            assignments[idx]["token_tasks"].append(task)
        else:
            assignments[idx]["and_tasks"].append(task)
        cost += int(task["cost"])
        assignments[idx]["cost"] = cost
        heapq.heappush(heap, (cost, idx))
    return assignments


def worker_compute_topk(
    base_records: np.memmap,
    query_records: np.memmap,
    candidate_rows: np.ndarray,
    qids: np.ndarray,
    topk: int,
    device: torch.device,
    row_batch: int,
    query_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    out_ids = np.full((qids.size, topk), int(UINT32_MAX), dtype=np.uint32)
    out_dist = np.full((qids.size, topk), np.inf, dtype=np.float32)
    if candidate_rows.size == 0 or qids.size == 0:
        return out_ids, out_dist

    row_step = max(1, int(row_batch))
    query_step = max(1, int(query_batch))
    for q_begin in range(0, qids.size, query_step):
        q_end = min(qids.size, q_begin + query_step)
        qids_batch = qids[q_begin:q_end]
        query_np = np.asarray(query_records["vec"][qids_batch.astype(np.int64)], dtype=np.float16)
        query_gpu = torch.from_numpy(query_np).to(device=device, dtype=torch.float32)
        query_norm = torch.sum(query_gpu * query_gpu, dim=1)
        best_dist: torch.Tensor | None = None
        best_ids: torch.Tensor | None = None
        for row_begin in range(0, candidate_rows.size, row_step):
            row_end = min(candidate_rows.size, row_begin + row_step)
            rows_np = candidate_rows[row_begin:row_end]
            base_np = np.asarray(base_records["vec"][rows_np.astype(np.int64)], dtype=np.float16)
            base_gpu = torch.from_numpy(base_np).to(device=device, dtype=torch.float32)
            base_norm = torch.sum(base_gpu * base_gpu, dim=1)
            dist = query_norm[:, None] + base_norm[None, :] - 2.0 * torch.matmul(query_gpu, base_gpu.T)
            # Norm expansion can round identical fp16 vectors slightly below zero.
            dist.clamp_min_(0.0)
            keep = min(topk, dist.shape[1])
            chunk_dist, chunk_pos = torch.topk(dist, k=keep, dim=1, largest=False, sorted=True)
            row_ids_gpu = torch.from_numpy(rows_np.astype(np.int64, copy=False)).to(device=device)
            chunk_ids = row_ids_gpu[chunk_pos]
            if best_dist is None:
                merged_dist = chunk_dist
                merged_ids = chunk_ids
            else:
                merged_dist = torch.cat((best_dist, chunk_dist), dim=1)
                merged_ids = torch.cat((best_ids, chunk_ids), dim=1)
            keep = min(topk, merged_dist.shape[1])
            best_dist, top_pos = torch.topk(merged_dist, k=keep, dim=1, largest=False, sorted=True)
            best_ids = torch.gather(merged_ids, 1, top_pos)
            del base_gpu, base_norm, dist, chunk_dist, chunk_pos, chunk_ids, merged_dist, merged_ids
        if best_dist is not None and best_ids is not None:
            ids_cpu = best_ids.cpu().numpy().astype(np.uint32, copy=False)
            dist_cpu = best_dist.cpu().numpy().astype(np.float32, copy=False)
            out_ids[q_begin:q_end, : ids_cpu.shape[1]] = ids_cpu
            out_dist[q_begin:q_end, : dist_cpu.shape[1]] = dist_cpu
        del query_gpu, query_norm, best_dist, best_ids
    return out_ids, out_dist


def worker_main(config: dict[str, Any], assignment: dict[str, Any]) -> str:
    import torch
    worker_idx = int(assignment["worker_idx"])
    device_id = int(assignment["device"])
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    base_info = inspect_hvecs(Path(config["base_hvecs"]))
    query_info = inspect_hvecs(Path(config["query_hvecs"]))
    base_records = hvecs_records(Path(config["base_hvecs"]), base_info)
    query_records = hvecs_records(Path(config["query_hvecs"]), query_info)
    postings_dir = Path(config["postings_dir"])
    topk = int(config["topk"])
    row_batch = int(config["row_batch"])
    query_batch = int(config["query_batch"])
    partial_path = Path(config["partials_dir"]) / f"worker_{worker_idx}_gpu{device_id}.npz"

    token_query_ids: list[np.ndarray] = []
    token_ids: list[np.ndarray] = []
    token_top_ids: list[np.ndarray] = []
    token_top_dist: list[np.ndarray] = []
    and_query_ids: list[np.ndarray] = []
    and_top_ids: list[np.ndarray] = []
    and_top_dist: list[np.ndarray] = []
    and_candidate_counts: list[np.ndarray] = []

    started = time.monotonic()
    print(
        f"[worker {worker_idx}] gpu={device_id} token_tasks={len(assignment['token_tasks'])} "
        f"and_tasks={len(assignment['and_tasks'])} cost={assignment['cost']}",
        flush=True,
    )

    for task_idx, task in enumerate(assignment["token_tasks"], start=1):
        token = int(task["token"])
        qids = np.asarray(task["qids"], dtype=np.int64)
        rows = np.load(posting_path(postings_dir, token), mmap_mode="r")
        ids, dist = worker_compute_topk(base_records, query_records, rows, qids, topk, device, row_batch, query_batch)
        token_query_ids.append(qids.astype(np.int32))
        token_ids.append(np.full(qids.shape, token, dtype=np.int32))
        token_top_ids.append(ids)
        token_top_dist.append(dist)
        print(
            f"[worker {worker_idx}] token {task_idx}/{len(assignment['token_tasks'])} "
            f"token={token} posting={rows.shape[0]} q={qids.size}",
            flush=True,
        )

    for task_idx, task in enumerate(assignment["and_tasks"], start=1):
        qids = np.asarray(task["qids"], dtype=np.int64)
        token_tuple = tuple(int(token) for token in task["tokens"])
        rows = np.load(posting_path(postings_dir, token_tuple[0]), mmap_mode="r")
        candidate = np.asarray(rows, dtype=np.uint32)
        for token in token_tuple[1:]:
            next_rows = np.load(posting_path(postings_dir, token), mmap_mode="r")
            candidate = np.intersect1d(candidate, next_rows, assume_unique=True).astype(np.uint32, copy=False)
            if candidate.size == 0:
                break
        ids, dist = worker_compute_topk(
            base_records,
            query_records,
            candidate,
            qids,
            topk,
            device,
            row_batch,
            query_batch,
        )
        and_query_ids.append(qids.astype(np.int32))
        and_top_ids.append(ids)
        and_top_dist.append(dist)
        and_candidate_counts.append(np.full(qids.shape, candidate.size, dtype=np.int64))
        print(
            f"[worker {worker_idx}] and {task_idx}/{len(assignment['and_tasks'])} "
            f"tokens={token_tuple} candidates={candidate.size} q={qids.size}",
            flush=True,
        )

    payload: dict[str, np.ndarray] = {
        "token_query_ids": np.concatenate(token_query_ids) if token_query_ids else np.empty(0, dtype=np.int32),
        "token_ids": np.concatenate(token_ids) if token_ids else np.empty(0, dtype=np.int32),
        "token_top_ids": np.concatenate(token_top_ids, axis=0) if token_top_ids else np.empty((0, topk), dtype=np.uint32),
        "token_top_dist": np.concatenate(token_top_dist, axis=0) if token_top_dist else np.empty((0, topk), dtype=np.float32),
        "and_query_ids": np.concatenate(and_query_ids) if and_query_ids else np.empty(0, dtype=np.int32),
        "and_top_ids": np.concatenate(and_top_ids, axis=0) if and_top_ids else np.empty((0, topk), dtype=np.uint32),
        "and_top_dist": np.concatenate(and_top_dist, axis=0) if and_top_dist else np.empty((0, topk), dtype=np.float32),
        "and_candidate_counts": np.concatenate(and_candidate_counts) if and_candidate_counts else np.empty(0, dtype=np.int64),
        "elapsed_seconds": np.asarray([time.monotonic() - started], dtype=np.float64),
    }
    np.savez(partial_path, **payload)
    print(f"[worker {worker_idx}] wrote {partial_path} elapsed={payload['elapsed_seconds'][0]:.1f}s", flush=True)
    return str(partial_path)


def merge_small_topk(ids_list: list[np.ndarray], dist_list: list[np.ndarray], topk: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.concatenate(ids_list).astype(np.uint32, copy=False)
    dist = np.concatenate(dist_list).astype(np.float32, copy=False)
    valid = ids != UINT32_MAX
    ids = ids[valid]
    dist = dist[valid]
    if ids.size == 0:
        return np.full(topk, UINT32_MAX, dtype=np.uint32), np.full(topk, np.inf, dtype=np.float32)
    order = np.lexsort((ids.astype(np.uint64), dist.astype(np.float64)))
    ids = ids[order]
    dist = dist[order]
    unique_ids, first = np.unique(ids, return_index=True)
    unique_dist = dist[first]
    order = np.lexsort((unique_ids.astype(np.uint64), unique_dist.astype(np.float64)))
    keep = min(topk, order.size)
    out_ids = np.full(topk, UINT32_MAX, dtype=np.uint32)
    out_dist = np.full(topk, np.inf, dtype=np.float32)
    selected = order[:keep]
    out_ids[:keep] = unique_ids[selected]
    out_dist[:keep] = unique_dist[selected]
    return out_ids, out_dist


def merge_partials(
    queries: list[QueryRecord],
    partial_paths: list[Path],
    topk: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    final_ids = np.full((len(queries), topk), UINT32_MAX, dtype=np.uint32)
    final_dist = np.full((len(queries), topk), np.inf, dtype=np.float32)
    token_partials: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    filled = np.zeros(len(queries), dtype=bool)
    and_candidate_counts: list[int] = []

    for path in partial_paths:
        data = np.load(path)
        token_qids = data["token_query_ids"]
        token_ids = data["token_ids"]
        token_top_ids = data["token_top_ids"]
        token_top_dist = data["token_top_dist"]
        for idx in range(token_qids.shape[0]):
            token_partials[(int(token_qids[idx]), int(token_ids[idx]))] = (token_top_ids[idx], token_top_dist[idx])

        and_qids = data["and_query_ids"]
        and_top_ids = data["and_top_ids"]
        and_top_dist = data["and_top_dist"]
        for idx in range(and_qids.shape[0]):
            qid = int(and_qids[idx])
            final_ids[qid] = and_top_ids[idx]
            final_dist[qid] = and_top_dist[idx]
            filled[qid] = True
        if "and_candidate_counts" in data:
            and_candidate_counts.extend(int(value) for value in data["and_candidate_counts"].tolist())

    missing_partials: list[str] = []
    for query in queries:
        if query.subtype == "and":
            continue
        pieces: list[np.ndarray] = []
        dists: list[np.ndarray] = []
        for token in query.token_ids:
            partial = token_partials.get((query.local_idx, int(token)))
            if partial is None:
                missing_partials.append(f"query={query.local_idx} token={token}")
                continue
            ids, dist = partial
            pieces.append(ids)
            dists.append(dist)
        if not pieces:
            continue
        ids, dist = merge_small_topk(pieces, dists, topk)
        final_ids[query.local_idx] = ids
        final_dist[query.local_idx] = dist
        filled[query.local_idx] = True

    if missing_partials:
        preview = ", ".join(missing_partials[:8])
        raise RuntimeError(f"missing token partials: {preview}")
    missing_queries = np.flatnonzero(~filled)
    if missing_queries.size:
        preview = ", ".join(str(int(idx)) for idx in missing_queries[:16])
        raise RuntimeError(f"missing GT rows for query ids: {preview}")
    hit_counts = np.count_nonzero(final_ids != UINT32_MAX, axis=1)
    return final_ids, final_dist, {
        "min_hits": int(hit_counts.min()),
        "median_hits": float(np.median(hit_counts)),
        "max_hits": int(hit_counts.max()),
        "missing_topk_rows": int(np.count_nonzero(hit_counts < topk)),
        "and_candidate_count_min": int(min(and_candidate_counts)) if and_candidate_counts else None,
        "and_candidate_count_median": float(np.median(np.asarray(and_candidate_counts))) if and_candidate_counts else None,
        "and_candidate_count_max": int(max(and_candidate_counts)) if and_candidate_counts else None,
    }


def write_summary(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.spmat_row_batch <= 0 or args.row_batch <= 0 or args.query_batch <= 0:
        raise ValueError("batch sizes must be positive")

    for required in (args.query_manifest, args.full_query_hvecs, args.base_hvecs, args.data_label_spmat, args.base_range_fbin):
        if not required.exists():
            raise FileNotFoundError(required)

    output_root = args.output_root
    postings_dir = output_root / "postings"
    partials_dir = output_root / "partials"
    output_root.mkdir(parents=True, exist_ok=True)
    partials_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    total_started = time.monotonic()
    spmat = inspect_spmat(args.data_label_spmat)
    base_info = inspect_hvecs(args.base_hvecs)
    full_query_info = inspect_hvecs(args.full_query_hvecs)
    if base_info.dim != full_query_info.dim:
        raise ValueError(f"base/query dim mismatch: {base_info.dim} vs {full_query_info.dim}")
    if base_info.rows != spmat.rows:
        raise ValueError(f"base rows {base_info.rows} != spmat rows {spmat.rows}")
    range_rows, range_count = read_fbin_header(args.base_range_fbin)
    if range_rows != spmat.rows:
        raise ValueError(f"range rows {range_rows} != spmat rows {spmat.rows}")

    queries = load_categorical_queries(args.query_manifest, args.limit_per_subtype)
    predicate_analysis = analyze_predicates(queries)
    analysis_path = output_root / "categorical_predicate_analysis.json"
    write_summary(analysis_path, predicate_analysis, args.overwrite)
    print(
        f"[gt] selected categorical queries={len(queries)} "
        f"unique_predicates={predicate_analysis['unique_predicates_total']}",
        flush=True,
    )
    for subtype, info in sorted(predicate_analysis["by_subtype"].items()):
        print(
            f"[gt] subtype={subtype} queries={info['queries']} unique={info['unique_predicates']} "
            f"reuse_median={info['reuse_median']} candidate_unique_sum={info['candidate_count_unique_sum']}",
            flush=True,
        )

    artifact_info = build_categorical_query_artifacts(
        queries,
        args.full_query_hvecs,
        output_root / "query.categorical.hvecs",
        output_root / "query_manifest.categorical.jsonl",
        output_root / "query_range.categorical.bin",
        range_count,
        args.overwrite,
    )

    token_ids = sorted({token for query in queries for token in query.token_ids})
    started = time.monotonic()
    posting_sizes = build_token_postings(
        args.data_label_spmat,
        spmat,
        token_ids,
        postings_dir,
        args.spmat_row_batch,
        args.reuse_postings,
    )
    timings["build_postings_seconds"] = time.monotonic() - started

    token_qids: dict[int, list[int]] = defaultdict(list)
    and_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    and_cost_hints: dict[tuple[int, ...], int] = {}
    for query in queries:
        if query.subtype in {"single", "or"}:
            for token in query.token_ids:
                token_qids[int(token)].append(query.local_idx)
        elif query.subtype == "and":
            and_groups[query.token_ids].append(query.local_idx)
            if query.candidate_count >= 0:
                and_cost_hints[query.token_ids] = query.candidate_count

    token_tasks = [
        {
            "kind": "token",
            "token": int(token),
            "qids": sorted(set(qids)),
            "posting_size": int(posting_sizes[int(token)]),
            "cost": int(posting_sizes[int(token)]) * len(set(qids)),
        }
        for token, qids in sorted(token_qids.items())
    ]
    and_items = sorted(and_groups.items())
    if args.max_and_predicates > 0:
        and_items = and_items[: args.max_and_predicates]
    and_tasks = [
        {
            "kind": "and",
            "tokens": list(tokens),
            "qids": qids,
            "candidate_count_hint": int(and_cost_hints.get(tokens, min(posting_sizes[int(token)] for token in tokens))),
            "cost": int(and_cost_hints.get(tokens, min(posting_sizes[int(token)] for token in tokens))) * len(qids),
        }
        for tokens, qids in and_items
    ]
    if args.max_and_predicates > 0:
        allowed = {qid for task in and_tasks for qid in task["qids"]}
        queries = [query for query in queries if query.subtype != "and" or query.local_idx in allowed]
        raise RuntimeError("--max-and-predicates is for development only and cannot currently remap query ids")

    devices = resolve_devices(args.devices)
    assignments = assign_tasks(token_tasks, and_tasks, devices)
    for idx, assignment in enumerate(assignments):
        assignment["worker_idx"] = idx
    assignment_summary = [
        {
            "worker_idx": int(assignment["worker_idx"]),
            "device": int(assignment["device"]),
            "token_tasks": len(assignment["token_tasks"]),
            "and_tasks": len(assignment["and_tasks"]),
            "cost": int(assignment["cost"]),
        }
        for assignment in assignments
    ]
    print(f"[gt] devices={devices}", flush=True)
    for item in assignment_summary:
        print(f"[gt] assignment {item}", flush=True)

    config = {
        "base_hvecs": str(args.base_hvecs),
        "query_hvecs": str(output_root / "query.categorical.hvecs"),
        "postings_dir": str(postings_dir),
        "partials_dir": str(partials_dir),
        "topk": int(args.topk),
        "row_batch": int(args.row_batch),
        "query_batch": int(args.query_batch),
    }
    started = time.monotonic()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(assignments)) as pool:
        partial_paths = pool.starmap(worker_main, [(config, assignment) for assignment in assignments])
    timings["gpu_exact_seconds"] = time.monotonic() - started

    final_ids, final_dist, gt_stats = merge_partials(queries, [Path(path) for path in partial_paths], args.topk)
    top100_info = write_gt_ibin(output_root / "gt.categorical.top100.ibin", final_ids, args.overwrite)
    top10_info = write_gt_ibin(output_root / "gt.categorical.top10.ibin", final_ids[:, :10], args.overwrite)
    dist100_info = write_dist_fbin(output_root / "gt.categorical.top100.dist.fbin", final_dist, args.overwrite)
    dist10_info = write_dist_fbin(output_root / "gt.categorical.top10.dist.fbin", final_dist[:, :10], args.overwrite)
    timings["total_seconds"] = time.monotonic() - total_started

    summary = {
        "source": {
            "query_manifest": str(args.query_manifest),
            "full_query_hvecs": str(args.full_query_hvecs),
            "base_hvecs": str(args.base_hvecs),
            "data_label_spmat": str(args.data_label_spmat),
            "base_range_fbin": str(args.base_range_fbin),
        },
        "outputs": {
            **artifact_info,
            "top100_gt": top100_info,
            "top10_gt": top10_info,
            "top100_distances": dist100_info,
            "top10_distances": dist10_info,
            "predicate_analysis": str(analysis_path),
        },
        "spmat": {
            "rows": spmat.rows,
            "cols": spmat.cols,
            "nnz": spmat.nnz,
            "format_note": "header + int64 indptr + int32 indices; data payload is not required by this generator",
        },
        "vectors": {
            "base_rows": base_info.rows,
            "query_rows_full": full_query_info.rows,
            "dim": base_info.dim,
            "scalar_type": "float16_hvecs",
        },
        "predicate_analysis": predicate_analysis,
        "token_postings": {
            "token_count": len(token_ids),
            "posting_rows_total": int(sum(posting_sizes.values())),
            "posting_size_min": int(min(posting_sizes.values())),
            "posting_size_median": float(np.median(np.asarray(list(posting_sizes.values()), dtype=np.float64))),
            "posting_size_max": int(max(posting_sizes.values())),
            "path": str(postings_dir),
        },
        "tasks": {
            "token_tasks": len(token_tasks),
            "and_tasks": len(and_tasks),
            "assignments": assignment_summary,
        },
        "gt_stats": gt_stats,
        "timings": timings,
        "notes": [
            "single/or exact GT reuses per-token top-k; OR top-k over a union is the merge of per-token top-k for the same query.",
            "AND exact GT intersects token postings per unique predicate and batches all reused query instances.",
            "top10 GT is the prefix of top100 GT.",
        ],
    }
    summary_path = output_root / "categorical_gt_summary.json"
    write_summary(summary_path, summary, args.overwrite)

    print(f"[gt] wrote top100 {top100_info['path']}", flush=True)
    print(f"[gt] wrote top10  {top10_info['path']}", flush=True)
    print(f"[gt] wrote summary {summary_path}", flush=True)
    print(f"[gt] total_seconds={timings['total_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
