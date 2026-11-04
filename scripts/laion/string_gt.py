#!/usr/bin/env python3
"""Compute exact string ground truth from materialized queries and token postings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import mmap
import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from scripts.laion.categorical_gt import write_gt_ibin, write_dist_fbin
from scripts.laion.base import (
    RuntimeQuery,
    ensure_output_path,
    hvecs_records,
    inspect_hvecs,
    write_hvecs_batch,
)
from scripts.laion.query import load_selected_queries, csv_lower_set, manifest_subtype_key
from scripts.laion.string_index import (
    TokenIdPostingsBinaryLookup,
    TokenIdPostingsDuckDBLookup,
    TokenIdPostingsNpyLookup,
    TokenPostingsDuckDBLookup,
    TokenTextIndex,
    build_token_postings_lookup,
    chunked_sequence,
    detect_token_postings_layout,
    fuzzy_matching_token_ids,
    fuzzy_predicate_mode,
    fuzzy_predicate_pattern,
    load_token_text_index,
    load_token_text_index_from_ngram_binary,
    token_text_from_predicate,
)

DEFAULT_DATA_ROOT = Path("data/laion100m")
DEFAULT_BASE_HVECS = DEFAULT_DATA_ROOT / "base.hvecs"
DEFAULT_TOKEN_POSTINGS = Path("artifacts/laion100m/token_postings.bin")
UINT32_MAX = np.uint32(0xFFFFFFFF)


@dataclass
class StringGTQuery:
    local_idx: int
    query_id: str
    subtype: str
    predicate_key: str
    token_id: int | None = None
    token_text: str | None = None
    fuzzy_mode: str | None = None
    fuzzy_pattern: str | None = None


def mem_available_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        with meminfo.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return max(0, int(parts[1]) * 1024)
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return max(0, int(pages) * int(page_size))
    except (AttributeError, OSError, ValueError):
        return 0


def warm_file_into_page_cache(
    path: Path,
    *,
    available_bytes: int,
    min_headroom_ratio: float = 1.10,
    chunk_bytes: int = 128 << 20,
    full_read_max_bytes: int = 8 << 30,
    sparse_touch_stride_bytes: int = 8 << 20,
) -> dict[str, Any]:
    info = {
        "path": str(path),
        "exists": path.exists(),
        "attempted": False,
        "completed": False,
        "bytes": 0,
        "elapsed_seconds": 0.0,
        "available_bytes_before": int(available_bytes),
        "reason": None,
        "mode": None,
    }
    if not path.exists():
        info["reason"] = "missing"
        return info
    if not path.is_file():
        info["reason"] = "not_a_file"
        return info

    file_size = int(path.stat().st_size)
    info["bytes"] = file_size
    if available_bytes > 0 and available_bytes < int(file_size * float(min_headroom_ratio)):
        info["reason"] = "insufficient_memavailable"
        return info

    started = time.monotonic()
    info["attempted"] = True
    try:
        with path.open("rb", buffering=0) as handle:
            fd = handle.fileno()
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_SEQUENTIAL"):
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                except OSError:
                    pass
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_WILLNEED"):
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
                except OSError:
                    pass
            if file_size <= int(full_read_max_bytes):
                info["mode"] = "full_read"
                buffer = bytearray(max(1, int(chunk_bytes)))
                view = memoryview(buffer)
                warmed = 0
                while True:
                    read_bytes = handle.readinto(view)
                    if not read_bytes:
                        break
                    warmed += int(read_bytes)
                info["completed"] = warmed >= file_size
                info["bytes_warmed"] = int(warmed)
            else:
                info["mode"] = "advisory_sparse_touch"
                stride = max(mmap.PAGESIZE, int(sparse_touch_stride_bytes))
                touch_count = 0
                checksum = 0
                with mmap.mmap(fd, length=0, access=mmap.ACCESS_READ) as mapped:
                    if hasattr(mapped, "madvise") and hasattr(mmap, "MADV_SEQUENTIAL"):
                        try:
                            mapped.madvise(mmap.MADV_SEQUENTIAL)
                        except (AttributeError, OSError, ValueError):
                            pass
                    if hasattr(mapped, "madvise") and hasattr(mmap, "MADV_WILLNEED"):
                        try:
                            mapped.madvise(mmap.MADV_WILLNEED)
                        except (AttributeError, OSError, ValueError):
                            pass
                    for offset in range(0, file_size, stride):
                        checksum ^= mapped[offset]
                        touch_count += 1
                    last_offset = file_size - 1
                    if file_size > 0 and (last_offset % stride) != 0:
                        checksum ^= mapped[last_offset]
                        touch_count += 1
                info["completed"] = True
                info["touch_stride_bytes"] = int(stride)
                info["touch_count"] = int(touch_count)
                info["touch_checksum"] = int(checksum)
    finally:
        info["elapsed_seconds"] = float(time.monotonic() - started)
    return info


def warm_gt_inputs_to_page_cache(
    *,
    base_hvecs: Path,
    token_postings_dir: Path,
    token_ngram_index_path: Path | None,
    postings_layout: str,
) -> list[dict[str, Any]]:
    available = mem_available_bytes()
    targets = [base_hvecs]
    if postings_layout == "token_id_binary":
        targets.append(token_postings_dir)
    if token_ngram_index_path is not None:
        targets.append(token_ngram_index_path)
    results: list[dict[str, Any]] = []
    for path in targets:
        result = warm_file_into_page_cache(path, available_bytes=available)
        status = "done" if result.get("completed") else ("skip" if not result.get("attempted") else "partial")
        print(
            f"[gt] host warmup {status} path={path} bytes={result.get('bytes', 0)} "
            f"elapsed={result.get('elapsed_seconds', 0.0):.2f}s reason={result.get('reason')}",
            flush=True,
        )
        results.append(result)
        available = mem_available_bytes()
    return results


def resolve_cuda_device_ids(raw: str | None) -> list[int]:
    value = str(raw or "").strip()
    if not value or value.lower() == "auto":
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("PyTorch is required to resolve CUDA devices for GT generation.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; GT generation requires at least one CUDA device")
        count = torch.cuda.device_count()
        if count <= 0:
            raise RuntimeError("torch sees zero CUDA devices")
        return list(range(count))

    device_ids: list[int] = []
    for token in value.split(","):
        item = token.strip()
        if not item:
            continue
        if item.lower().startswith("cuda:"):
            item = item.split(":", 1)[1]
        elif item.lower() == "cuda":
            item = "0"
        device_ids.append(int(item))
    if not device_ids:
        raise ValueError(f"unable to resolve CUDA devices from {raw!r}")
    return list(dict.fromkeys(device_ids))


def estimate_gt_work_item_cost(
    item: GTWorkItem,
    token_lookup: Any,
    posting_size_cache: dict[int, int] | None = None,
) -> int:
    token_ids = np.asarray(item.token_ids if item.token_ids is not None else np.empty((0,), dtype=np.uint32), dtype=np.uint32)
    if token_ids.size == 0:
        return max(1, int(item.qids.size))
    posting_size_fn = getattr(token_lookup, "posting_size", None)
    estimated_rows = 0
    if callable(posting_size_fn):
        for token_id in token_ids.astype(np.int64, copy=False).tolist():
            token_id = int(token_id)
            if posting_size_cache is not None and token_id in posting_size_cache:
                estimated_rows += max(0, int(posting_size_cache[token_id]))
                continue
            size = max(0, int(posting_size_fn(token_id)))
            if posting_size_cache is not None:
                posting_size_cache[token_id] = size
            estimated_rows += size
    if estimated_rows <= 0:
        estimated_rows = max(1, int(token_ids.size))
    return max(1, estimated_rows) * max(1, int(item.qids.size))


def assign_gt_work_items(
    items: list[GTWorkItem],
    device_ids: list[int],
    token_lookup: Any,
) -> list[dict[str, Any]]:
    assignments = [{"device_id": device_id, "items": [], "cost": 0} for device_id in device_ids]
    if not assignments:
        return []
    import heapq

    posting_size_cache: dict[int, int] = {}
    heap: list[tuple[int, int]] = [(0, idx) for idx in range(len(assignments))]
    heapq.heapify(heap)
    scored_items = sorted(
        ((estimate_gt_work_item_cost(item, token_lookup, posting_size_cache), item) for item in items),
        key=lambda pair: pair[0],
        reverse=True,
    )
    for cost, item in scored_items:
        current_cost, idx = heapq.heappop(heap)
        new_cost = current_cost + int(cost)
        assignments[idx]["items"].append(item)
        assignments[idx]["cost"] = new_cost
        heapq.heappush(heap, (new_cost, idx))
    return assignments


def numeric_stats(values: Sequence[int] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": int(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def load_string_gt_queries(
    queries: list[RuntimeQuery],
    token_index: TokenTextIndex | None = None,
) -> list[StringGTQuery]:
    out: list[StringGTQuery] = []
    for local_idx, query in enumerate(queries):
        record = query.source
        family, subtype, subtype_name = manifest_subtype_key(record)
        if family != "string" or subtype not in {"exact", "fuzzy"}:
            raise ValueError(
                "--compute-gt currently supports only string.exact/string.fuzzy selections; "
                f"found {subtype_name} at selected row {local_idx}"
            )
        predicates = record.get("predicates") or []
        if not predicates:
            raise ValueError(f"selected row {local_idx} has no predicates")
        query_id = str(record.get("query_id", local_idx))
        predicate_key = str(record.get("predicate_key") or record.get("predicate_id") or query_id)
        if subtype == "exact":
            token_id = predicates[0].get("token_id", predicates[0].get("source_token_id"))
            if token_id is None:
                raise ValueError(f"selected row {local_idx} exact predicate is missing token_id/source_token_id")
            token_text = None
            try:
                token_text = token_text_from_predicate(predicates[0], token_index)
            except ValueError:
                token_text = None
            out.append(
                StringGTQuery(
                    local_idx=local_idx,
                    query_id=query_id,
                    subtype=subtype,
                    predicate_key=f"exact:{int(token_id)}",
                    token_id=int(token_id),
                    token_text=token_text,
                )
            )
            continue
        mode = fuzzy_predicate_mode(predicates[0])
        pattern = fuzzy_predicate_pattern(predicates[0])
        out.append(
            StringGTQuery(
                local_idx=local_idx,
                query_id=query_id,
                subtype=subtype,
                predicate_key=f"fuzzy:{mode}:{pattern}",
                fuzzy_mode=mode,
                fuzzy_pattern=pattern,
            )
        )
    return out


class BaseVectorProvider:
    def __init__(self, base_records: np.memmap):
        self.base_records = base_records

    def fetch_rows(self, rows_np: np.ndarray, dtype: np.dtype) -> np.ndarray:
        return np.asarray(self.base_records["vec"][rows_np], dtype=dtype)

    def summary(self) -> dict[str, Any]:
        return {"mode": "memmap_direct"}


class CachedBaseVectorProvider(BaseVectorProvider):
    def __init__(
        self,
        base_records: np.memmap,
        max_rows: int,
        dim: int,
        cache_dtype: np.dtype = np.float16,
        bulk_direct_threshold_rows: int = 65536,
        disable_cache_sample_rows: int = 262144,
        disable_cache_hit_rate_threshold: float = 0.05,
    ):
        super().__init__(base_records)
        self.max_rows = max(0, int(max_rows))
        self.dim = int(dim)
        self.cache_dtype = np.dtype(cache_dtype)
        self.bulk_direct_threshold_rows = max(1, int(bulk_direct_threshold_rows))
        self.disable_cache_sample_rows = max(0, int(disable_cache_sample_rows))
        self.disable_cache_hit_rate_threshold = float(disable_cache_hit_rate_threshold)
        self.row_to_slot: dict[int, int] = {}
        self.slot_to_row: np.ndarray | None = None
        self.cache: np.ndarray | None = None
        self.next_slot = 0
        self.total_fetch_rows = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cached_rows = 0
        self.direct_bypass_rows = 0
        self.cache_enabled = self.max_rows > 0

    def _ensure_cache_storage(self) -> None:
        if self.max_rows <= 0 or self.cache is not None:
            return
        self.slot_to_row = np.full((self.max_rows,), -1, dtype=np.int64)
        self.cache = np.empty((self.max_rows, self.dim), dtype=self.cache_dtype)

    def fetch_rows(self, rows_np: np.ndarray, dtype: np.dtype) -> np.ndarray:
        rows_np = rows_np.astype(np.int64, copy=False)
        self.total_fetch_rows += int(rows_np.size)
        if self.max_rows <= 0 or rows_np.size == 0:
            self.cache_misses += int(rows_np.size)
            return super().fetch_rows(rows_np, dtype)
        if not self.cache_enabled or rows_np.size >= self.bulk_direct_threshold_rows:
            self.direct_bypass_rows += int(rows_np.size)
            self.cache_misses += int(rows_np.size)
            return super().fetch_rows(rows_np, dtype)

        self._ensure_cache_storage()
        if self.cache is None or self.slot_to_row is None:
            self.cache_misses += int(rows_np.size)
            return super().fetch_rows(rows_np, dtype)

        out = np.empty((rows_np.size, self.dim), dtype=self.cache_dtype)
        missing_positions: list[int] = []
        missing_rows: list[int] = []
        for pos, row_id in enumerate(rows_np.tolist()):
            slot = self.row_to_slot.get(int(row_id))
            if slot is None:
                missing_positions.append(pos)
                missing_rows.append(int(row_id))
            else:
                out[pos] = self.cache[slot]
                self.cache_hits += 1

        if missing_rows:
            unique_missing_rows: list[int] = []
            unique_positions: dict[int, list[int]] = defaultdict(list)
            for pos, row_id in zip(missing_positions, missing_rows):
                if row_id not in unique_positions:
                    unique_missing_rows.append(row_id)
                unique_positions[row_id].append(pos)

            fetched = np.asarray(
                self.base_records["vec"][np.asarray(unique_missing_rows, dtype=np.int64)],
                dtype=self.cache_dtype,
            )
            self.cache_misses += len(missing_rows)
            for local_idx, row_id in enumerate(unique_missing_rows):
                vector = fetched[local_idx]
                for pos in unique_positions[row_id]:
                    out[pos] = vector
                if self.max_rows > 0:
                    slot = self.next_slot % self.max_rows
                    evicted_row = int(self.slot_to_row[slot])
                    if evicted_row >= 0:
                        self.row_to_slot.pop(evicted_row, None)
                    self.cache[slot] = vector
                    self.slot_to_row[slot] = row_id
                    self.row_to_slot[row_id] = slot
                    self.next_slot += 1
            self.cached_rows = min(len(self.row_to_slot), self.max_rows)

        if (
            self.cache_enabled
            and self.disable_cache_sample_rows > 0
            and self.total_fetch_rows >= self.disable_cache_sample_rows
        ):
            hit_rate = 0.0 if self.total_fetch_rows <= 0 else float(self.cache_hits) / float(self.total_fetch_rows)
            if hit_rate < self.disable_cache_hit_rate_threshold:
                self.cache_enabled = False
                self.row_to_slot.clear()
                self.slot_to_row = None
                self.cache = None
                self.cached_rows = 0

        if out.dtype != dtype:
            return out.astype(dtype, copy=False)
        return out

    def summary(self) -> dict[str, Any]:
        hit_rate = 0.0 if self.total_fetch_rows <= 0 else float(self.cache_hits) / float(self.total_fetch_rows)
        return {
            "mode": "host_vector_cache",
            "max_rows": self.max_rows,
            "dim": self.dim,
            "cache_dtype": str(self.cache_dtype),
            "cached_rows": self.cached_rows,
            "total_fetch_rows": self.total_fetch_rows,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "cache_enabled": self.cache_enabled,
            "bulk_direct_threshold_rows": self.bulk_direct_threshold_rows,
            "direct_bypass_rows": self.direct_bypass_rows,
        }


@dataclass
class GTWorkItem:
    group_key: str
    subtype: str
    qids: np.ndarray
    token_count: int
    candidate_rows: np.ndarray | None = None
    token_ids: np.ndarray | None = None


@dataclass
class GTWorkBatch:
    items: list[GTWorkItem]
    union_row_budget: int
    union_token_count: int
    query_count: int


def token_id_batches_by_row_budget(
    token_ids: np.ndarray,
    posting_lookup: Any,
    row_budget: int,
) -> list[np.ndarray]:
    if token_ids.size == 0:
        return []
    if row_budget <= 0 or not hasattr(posting_lookup, "posting_size"):
        return [np.asarray(token_ids, dtype=np.uint32, copy=False)]
    sized = []
    for token_id in token_ids.astype(np.int64, copy=False).tolist():
        size = int(posting_lookup.posting_size(int(token_id)))
        if size <= 0:
            continue
        sized.append((int(token_id), size))
    if not sized:
        return []
    sized.sort(key=lambda item: item[1], reverse=True)
    batches: list[np.ndarray] = []
    current: list[int] = []
    current_rows = 0
    for token_id, posting_size in sized:
        if current and current_rows + posting_size > row_budget:
            batches.append(np.asarray(current, dtype=np.uint32))
            current = []
            current_rows = 0
        current.append(token_id)
        current_rows += posting_size
        if current_rows >= row_budget:
            batches.append(np.asarray(current, dtype=np.uint32))
            current = []
            current_rows = 0
    if current:
        batches.append(np.asarray(current, dtype=np.uint32))
    return batches


def token_id_batches_for_gt_work(
    token_ids: np.ndarray,
    postings_layout: str,
    posting_lookup: Any,
    postings_row_budget: int,
    postings_token_batch_size: int,
) -> list[np.ndarray]:
    token_ids = np.asarray(token_ids, dtype=np.uint32, copy=False)
    if token_ids.size == 0:
        return []
    if postings_layout == "token_id_binary":
        return token_id_batches_by_row_budget(token_ids, posting_lookup, postings_row_budget)
    return [
        np.asarray(token_batch, dtype=np.uint32)
        for token_batch in chunked_sequence(token_ids, postings_token_batch_size)
    ]


def load_posting_rows_for_token_batch(
    token_ids: np.ndarray,
    postings_layout: str,
    token_lookup: Any,
    token_index: TokenTextIndex | None,
) -> np.ndarray:
    token_ids = np.asarray(token_ids, dtype=np.uint32, copy=False)
    if token_ids.size == 0:
        return np.empty((0,), dtype=np.uint32)
    if postings_layout in {"token_id_npy", "duckdb_token_id_parquet", "token_id_binary"}:
        return token_lookup.rows_for_token_ids_union(token_ids)
    if token_index is None:
        raise RuntimeError("text-token postings require token_index")
    return token_lookup.rows_for_token_ids_union(token_ids, token_index)


def merge_topk_cpu(ids_list: list[np.ndarray], dist_list: list[np.ndarray], topk: int) -> tuple[np.ndarray, np.ndarray]:
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


def load_query_vectors(query_records: np.memmap, qids: np.ndarray) -> np.ndarray:
    return np.asarray(query_records["vec"][qids.astype(np.int64, copy=False)], dtype=np.float16)


def topk_from_distance_matrix(dist: np.ndarray, row_ids: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
    if dist.ndim != 2:
        raise ValueError(f"distance matrix must be 2D, got shape={dist.shape}")
    out_ids = np.full((dist.shape[0], topk), UINT32_MAX, dtype=np.uint32)
    out_dist = np.full((dist.shape[0], topk), np.inf, dtype=np.float32)
    if dist.size == 0:
        return out_ids, out_dist

    row_ids = row_ids.astype(np.uint32, copy=False)
    keep = min(topk, dist.shape[1])
    if keep <= 0:
        return out_ids, out_dist

    for local_idx in range(dist.shape[0]):
        row_dist = np.asarray(dist[local_idx], dtype=np.float64)
        order = np.lexsort((row_ids.astype(np.uint64), row_dist))
        selected_ids = row_ids[order]
        selected_dist = row_dist[order]
        if selected_ids.size > 1:
            unique_ids, first = np.unique(selected_ids, return_index=True)
            selected_dist = selected_dist[first]
            selected_ids = unique_ids
            order = np.lexsort((selected_ids.astype(np.uint64), selected_dist.astype(np.float64)))
            selected_ids = selected_ids[order]
            selected_dist = selected_dist[order]
        out_ids[local_idx, : min(keep, selected_ids.size)] = selected_ids[:keep]
        out_dist[local_idx, : min(keep, selected_dist.size)] = selected_dist[:keep]
    return out_ids, out_dist


def cpu_topk_for_query_vectors(
    base_vector_provider: BaseVectorProvider,
    query_vectors: np.ndarray,
    candidate_rows: np.ndarray,
    topk: int,
    row_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    out_ids = np.full((query_vectors.shape[0], topk), UINT32_MAX, dtype=np.uint32)
    out_dist = np.full((query_vectors.shape[0], topk), np.inf, dtype=np.float32)
    if candidate_rows.size == 0 or query_vectors.size == 0:
        return out_ids, out_dist

    query_float = np.asarray(query_vectors, dtype=np.float32)
    query_norm = np.sum(query_float * query_float, axis=1)
    row_step = max(1, int(row_batch))
    best_ids = np.full((query_vectors.shape[0], 0), UINT32_MAX, dtype=np.uint32)
    best_dist = np.full((query_vectors.shape[0], 0), np.inf, dtype=np.float32)

    for row_begin in range(0, candidate_rows.size, row_step):
        row_end = min(candidate_rows.size, row_begin + row_step)
        rows_np = candidate_rows[row_begin:row_end].astype(np.int64, copy=False)
        base_np = base_vector_provider.fetch_rows(rows_np, np.float32)
        base_norm = np.sum(base_np * base_np, axis=1)
        dist = query_norm[:, None] + base_norm[None, :] - 2.0 * (query_float @ base_np.T)
        chunk_ids_cpu, chunk_dist_cpu = topk_from_distance_matrix(dist, rows_np, topk)
        merged_ids = np.empty((query_vectors.shape[0], topk), dtype=np.uint32)
        merged_dist = np.empty((query_vectors.shape[0], topk), dtype=np.float32)
        for local_idx in range(query_vectors.shape[0]):
            merged_ids[local_idx], merged_dist[local_idx] = merge_topk_cpu(
                [best_ids[local_idx], chunk_ids_cpu[local_idx]],
                [best_dist[local_idx], chunk_dist_cpu[local_idx]],
                topk,
            )
        best_ids = merged_ids
        best_dist = merged_dist

    out_ids[:] = best_ids
    out_dist[:] = best_dist
    return out_ids, out_dist


def cpu_exact_topk_for_query_vectors(
    base_vector_provider: BaseVectorProvider,
    query_vectors: np.ndarray,
    candidate_rows: np.ndarray,
    topk: int,
    row_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids, dist = cpu_topk_for_query_vectors(
        base_vector_provider,
        query_vectors,
        candidate_rows,
        topk,
        row_batch,
    )
    if ids.size == 0:
        return ids, dist
    for local_idx in range(ids.shape[0]):
        ids[local_idx], dist[local_idx] = dedupe_topk_rows_cpu(ids[local_idx], dist[local_idx], topk)
    return ids, dist


def gpu_topk_for_query_vectors(
    torch_module: Any,
    base_vector_provider: BaseVectorProvider,
    query_vectors: np.ndarray,
    candidate_rows: np.ndarray,
    topk: int,
    device: Any,
    row_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_gpu = torch_module.from_numpy(np.asarray(query_vectors, dtype=np.float32)).to(
        device=device,
        dtype=torch_module.float32,
    )
    query_norm = torch_module.sum(query_gpu * query_gpu, dim=1)
    return gpu_topk_for_query_tensors(
        torch_module,
        base_vector_provider,
        query_gpu,
        query_norm,
        candidate_rows,
        topk,
        device,
        row_batch,
    )


def gpu_stream_topk_for_candidate_rows(
    torch_module: Any,
    base_vector_provider: BaseVectorProvider,
    query_gpu: Any,
    query_norm: Any,
    candidate_rows: np.ndarray,
    topk: int,
    device: Any,
    row_batch: int,
    best_ids_gpu: Any | None = None,
    best_dist_gpu: Any | None = None,
) -> tuple[Any | None, Any | None]:
    query_count = int(query_gpu.shape[0])
    if candidate_rows.size == 0 or query_count == 0:
        return best_ids_gpu, best_dist_gpu

    row_step = max(1, int(row_batch))
    with torch_module.no_grad():
        for row_begin in range(0, candidate_rows.size, row_step):
            row_end = min(candidate_rows.size, row_begin + row_step)
            rows_np = candidate_rows[row_begin:row_end].astype(np.int64, copy=False)
            base_np = base_vector_provider.fetch_rows(rows_np, np.float16)
            base_gpu = torch_module.from_numpy(base_np).to(device=device, dtype=torch_module.float32)
            base_norm = torch_module.sum(base_gpu * base_gpu, dim=1)
            dist = query_norm[:, None] + base_norm[None, :] - 2.0 * torch_module.matmul(query_gpu, base_gpu.T)
            keep = min(topk, int(dist.shape[1]))
            if keep <= 0:
                del base_gpu, base_norm, dist
                continue
            chunk_dist_gpu, chunk_pos_gpu = torch_module.topk(
                dist,
                k=keep,
                dim=1,
                largest=False,
                sorted=True,
            )
            row_ids_gpu = torch_module.from_numpy(rows_np.astype(np.int64, copy=False)).to(device=device)
            chunk_ids_gpu = row_ids_gpu[chunk_pos_gpu]
            best_ids_gpu, best_dist_gpu = gpu_merge_topk_tensors(
                torch_module,
                best_ids_gpu,
                best_dist_gpu,
                chunk_ids_gpu,
                chunk_dist_gpu,
                topk,
            )
            del base_gpu, base_norm, dist, row_ids_gpu, chunk_ids_gpu, chunk_dist_gpu, chunk_pos_gpu
    return best_ids_gpu, best_dist_gpu


def gpu_topk_for_query_tensors(
    torch_module: Any,
    base_vector_provider: BaseVectorProvider,
    query_gpu: Any,
    query_norm: Any,
    candidate_rows: np.ndarray,
    topk: int,
    device: Any,
    row_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_count = int(query_gpu.shape[0])
    out_ids = np.full((query_count, topk), UINT32_MAX, dtype=np.uint32)
    out_dist = np.full((query_count, topk), np.inf, dtype=np.float32)
    if candidate_rows.size == 0 or query_count == 0:
        return out_ids, out_dist

    best_ids_gpu, best_dist_gpu = gpu_stream_topk_for_candidate_rows(
        torch_module,
        base_vector_provider,
        query_gpu,
        query_norm,
        candidate_rows,
        topk,
        device,
        row_batch,
    )

    if best_ids_gpu is not None and best_dist_gpu is not None:
        best_ids_cpu = best_ids_gpu.cpu().numpy().astype(np.uint32, copy=False)
        best_dist_cpu = best_dist_gpu.cpu().numpy().astype(np.float32, copy=False)
        width = int(best_ids_cpu.shape[1])
        out_ids[:, :width] = best_ids_cpu
        out_dist[:, :width] = best_dist_cpu
    return out_ids, out_dist


def gpu_exact_topk_for_query_tensors(
    torch_module: Any,
    base_vector_provider: BaseVectorProvider,
    query_gpu: Any,
    query_norm: Any,
    candidate_rows: np.ndarray,
    topk: int,
    device: Any,
    row_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids, dist = gpu_topk_for_query_tensors(
        torch_module,
        base_vector_provider,
        query_gpu,
        query_norm,
        candidate_rows,
        topk,
        device,
        row_batch,
    )
    if ids.size == 0:
        return ids, dist
    for local_idx in range(ids.shape[0]):
        ids[local_idx], dist[local_idx] = dedupe_topk_rows_cpu(ids[local_idx], dist[local_idx], topk)
    return ids, dist


def gpu_merge_topk_results(
    torch_module: Any,
    best_ids_gpu: Any | None,
    best_dist_gpu: Any | None,
    batch_ids: np.ndarray,
    batch_dists: np.ndarray,
    topk: int,
    device: Any,
) -> tuple[Any, Any]:
    batch_ids_gpu = torch_module.from_numpy(batch_ids.astype(np.int64, copy=False)).to(device=device)
    batch_dist_gpu = torch_module.from_numpy(batch_dists.astype(np.float32, copy=False)).to(device=device)
    valid_mask = batch_ids_gpu != int(UINT32_MAX)
    batch_dist_gpu = torch_module.where(
        valid_mask,
        batch_dist_gpu,
        torch_module.full_like(batch_dist_gpu, float("inf")),
    )
    if best_ids_gpu is None or best_dist_gpu is None:
        return batch_ids_gpu, batch_dist_gpu

    merged_ids_gpu = torch_module.cat((best_ids_gpu, batch_ids_gpu), dim=1)
    merged_dist_gpu = torch_module.cat((best_dist_gpu, batch_dist_gpu), dim=1)
    keep = min(topk, int(merged_dist_gpu.shape[1]))
    best_dist_gpu, best_pos = torch_module.topk(
        merged_dist_gpu,
        k=keep,
        dim=1,
        largest=False,
        sorted=True,
    )
    best_ids_gpu = torch_module.gather(merged_ids_gpu, 1, best_pos)
    return best_ids_gpu, best_dist_gpu


def gpu_merge_topk_tensors(
    torch_module: Any,
    best_ids_gpu: Any | None,
    best_dist_gpu: Any | None,
    chunk_ids_gpu: Any,
    chunk_dist_gpu: Any,
    topk: int,
) -> tuple[Any, Any]:
    if best_ids_gpu is None or best_dist_gpu is None:
        return chunk_ids_gpu, chunk_dist_gpu
    merged_ids_gpu = torch_module.cat((best_ids_gpu, chunk_ids_gpu), dim=1)
    merged_dist_gpu = torch_module.cat((best_dist_gpu, chunk_dist_gpu), dim=1)
    keep = min(topk, int(merged_dist_gpu.shape[1]))
    best_dist_gpu, best_pos = torch_module.topk(
        merged_dist_gpu,
        k=keep,
        dim=1,
        largest=False,
        sorted=True,
    )
    best_ids_gpu = torch_module.gather(merged_ids_gpu, 1, best_pos)
    return best_ids_gpu, best_dist_gpu


def dedupe_topk_rows_cpu(ids: np.ndarray, dist: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.uint32, copy=False).reshape(-1)
    dist = np.asarray(dist, dtype=np.float32, copy=False).reshape(-1)
    valid = ids != UINT32_MAX
    if not np.any(valid):
        return np.full(topk, UINT32_MAX, dtype=np.uint32), np.full(topk, np.inf, dtype=np.float32)
    ids = ids[valid]
    dist = dist[valid]
    order = np.lexsort((ids.astype(np.uint64), dist.astype(np.float64)))
    ids = ids[order]
    dist = dist[order]
    uniq_ids, first = np.unique(ids, return_index=True)
    uniq_dist = dist[first]
    order = np.lexsort((uniq_ids.astype(np.uint64), uniq_dist.astype(np.float64)))
    keep = min(topk, order.size)
    out_ids = np.full(topk, UINT32_MAX, dtype=np.uint32)
    out_dist = np.full(topk, np.inf, dtype=np.float32)
    selected = order[:keep]
    out_ids[:keep] = uniq_ids[selected]
    out_dist[:keep] = uniq_dist[selected]
    return out_ids, out_dist


def append_work_batch(
    batches: list[GTWorkBatch],
    current_items: list[GTWorkItem],
    current_union_row_budget: int,
    current_union_token_count: int,
    current_query_count: int,
) -> tuple[list[GTWorkItem], int, int, int]:
    if current_items:
        batches.append(
            GTWorkBatch(
                items=list(current_items),
                union_row_budget=int(current_union_row_budget),
                union_token_count=int(current_union_token_count),
                query_count=int(current_query_count),
            )
        )
    return [], 0, 0, 0


def build_exact_gt_work_batches(
    exact_work_items: list[GTWorkItem],
    token_lookup: Any,
    postings_row_budget: int,
    query_batch_size: int,
) -> list[GTWorkBatch]:
    work_batches: list[GTWorkBatch] = []
    current_items: list[GTWorkItem] = []
    current_union_row_budget = 0
    current_union_token_count = 0
    current_query_count = 0
    exact_merge_row_budget = max(postings_row_budget, int(postings_row_budget * 2))
    for item in exact_work_items:
        token_ids = np.asarray(
            item.token_ids if item.token_ids is not None else np.empty((0,), dtype=np.uint32),
            dtype=np.uint32,
        )
        item_budget = 1
        if token_ids.size > 0 and hasattr(token_lookup, "posting_size"):
            item_budget = max(1, sum(int(token_lookup.posting_size(int(token_id))) for token_id in token_ids.tolist()))
        if current_items and (
            current_union_row_budget + item_budget > exact_merge_row_budget
            or current_query_count + int(item.qids.size) > max(1, int(query_batch_size))
        ):
            current_items, current_union_row_budget, current_union_token_count, current_query_count = append_work_batch(
                work_batches,
                current_items,
                current_union_row_budget,
                current_union_token_count,
                current_query_count,
            )
        current_items.append(item)
        current_union_row_budget += item_budget
        current_union_token_count += item.token_count
        current_query_count += int(item.qids.size)
    current_items, current_union_row_budget, current_union_token_count, current_query_count = append_work_batch(
        work_batches,
        current_items,
        current_union_row_budget,
        current_union_token_count,
        current_query_count,
    )
    return work_batches


def gpu_topk_for_queries(
    torch_module: Any,
    base_vector_provider: BaseVectorProvider,
    query_records: np.memmap,
    candidate_rows: np.ndarray,
    qids: np.ndarray,
    topk: int,
    device: Any,
    row_batch: int,
    query_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    out_ids = np.full((qids.size, topk), UINT32_MAX, dtype=np.uint32)
    out_dist = np.full((qids.size, topk), np.inf, dtype=np.float32)
    if candidate_rows.size == 0 or qids.size == 0:
        return out_ids, out_dist

    row_step = max(1, int(row_batch))
    query_step = max(1, int(query_batch))
    candidate_rows = candidate_rows.astype(np.uint32, copy=False)
    with torch_module.no_grad():
        for q_begin in range(0, qids.size, query_step):
            q_end = min(qids.size, q_begin + query_step)
            qids_batch = qids[q_begin:q_end].astype(np.int64, copy=False)
            query_np = load_query_vectors(query_records, qids_batch)
            batch_ids, batch_dist = gpu_topk_for_query_vectors(
                torch_module,
                base_vector_provider,
                query_np,
                candidate_rows,
                topk,
                device,
                row_batch=row_step,
            )
            out_ids[q_begin:q_end] = batch_ids
            out_dist[q_begin:q_end] = batch_dist
    return out_ids, out_dist


def stable_sort_topk_rows(
    ids: np.ndarray,
    dist: np.ndarray,
    invalid_id: np.uint32 = UINT32_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.uint32)
    dist = np.asarray(dist, dtype=np.float32)
    if ids.shape != dist.shape:
        raise ValueError(f"ids/dist shape mismatch: {ids.shape} vs {dist.shape}")
    if ids.ndim != 2:
        raise ValueError(f"expected 2D top-k matrices, got ids.ndim={ids.ndim}")
    row_count, width = ids.shape
    if row_count == 0 or width <= 1:
        return ids, dist
    order = np.empty((row_count, width), dtype=np.int64)
    cols = np.arange(width, dtype=np.int64)
    invalid_id_int = int(invalid_id)
    for row_idx in range(row_count):
        row_ids = ids[row_idx]
        row_dist = dist[row_idx]
        valid = row_ids != invalid_id_int
        if np.any(valid):
            adjusted_dist = np.where(valid, row_dist, np.inf)
            adjusted_ids = np.where(valid, row_ids.astype(np.int64, copy=False), invalid_id_int)
            order[row_idx] = np.lexsort((adjusted_ids, adjusted_dist))
        else:
            order[row_idx] = cols
    row_selector = np.arange(row_count, dtype=np.int64)[:, None]
    return ids[row_selector, order], dist[row_selector, order]


def write_gt_rows_jsonl(
    path: Path,
    gt_queries: list[StringGTQuery],
    ids: np.ndarray,
    dist: np.ndarray,
    candidate_counts: np.ndarray,
    expanded_token_counts: np.ndarray,
    overwrite: bool,
) -> dict[str, Any]:
    ensure_output_path(path, overwrite)
    with path.open("w", encoding="utf-8") as handle:
        for query in gt_queries:
            idx = query.local_idx
            valid = ids[idx] != UINT32_MAX
            row = {
                "flashanns_query_index": idx,
                "query_id": query.query_id,
                "subtype": query.subtype,
                "candidate_count": int(candidate_counts[idx]),
                "expanded_token_count": (
                    None if expanded_token_counts[idx] < 0 else int(expanded_token_counts[idx])
                ),
                "topk": ids[idx][valid].astype(np.uint32).tolist(),
                "distances": dist[idx][valid].astype(np.float32).tolist(),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {"path": str(path), "rows": len(gt_queries), "bytes": path.stat().st_size}


def load_gt_token_index(
    token_map: Path,
    token_ngram_index_path: Path | None,
    ngram_size: int,
    token_map_batch_size: int,
) -> TokenTextIndex:
    if token_ngram_index_path is not None:
        print(f"[gt] loading fuzzy/text index from ngram binary {token_ngram_index_path}", flush=True)
        return load_token_text_index_from_ngram_binary(token_ngram_index_path)
    print(f"[gt] loading token_map fuzzy/text index from {token_map}", flush=True)
    return load_token_text_index(token_map, ngram_size=ngram_size, batch_size=token_map_batch_size)


def compute_string_gpu_gt_worker(
    config: dict[str, Any],
    assignment: dict[str, Any],
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for --compute-gt GPU top-k; install torch in the active environment.") from exc

    device_id = int(assignment["device_id"])
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    cpu_threads = max(1, int(config.get("cpu_threads", 1)))
    if cpu_threads > 0:
        try:
            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    base_hvecs = Path(config["base_hvecs"])
    query_vector_path = Path(config["query_vector_path"])
    token_postings_dir = Path(config["token_postings_dir"])
    token_map = Path(config["token_map"])
    token_ngram_index_path = config.get("token_ngram_index_path")
    if token_ngram_index_path is not None:
        token_ngram_index_path = Path(token_ngram_index_path)

    topk = int(config["topk"])
    row_batch_size = int(config["row_batch_size"])
    query_batch_size = int(config["query_batch_size"])
    postings_layout = str(config["postings_layout"])
    postings_workers = int(config["postings_workers"])
    postings_batch_rows = int(config["postings_batch_rows"])
    postings_token_batch_size = int(config["postings_token_batch_size"])
    max_row_id_exclusive = int(config["max_row_id_exclusive"])
    ngram_size = int(config["ngram_size"])
    token_map_batch_size = int(config["token_map_batch_size"])
    query_count = int(config["query_count"])

    base_info = inspect_hvecs(base_hvecs)
    query_info = inspect_hvecs(query_vector_path)
    if base_info.dim != query_info.dim:
        raise ValueError(f"base/query dim mismatch in worker {device_id}: {base_info.dim} vs {query_info.dim}")
    if query_info.rows != query_count:
        raise ValueError(f"query.hvecs rows {query_info.rows} != selected queries {query_count}")

    base_records = hvecs_records(base_hvecs, base_info)
    query_records = hvecs_records(query_vector_path, query_info)
    base_vector_provider = BaseVectorProvider(base_records)
    token_lookup = build_token_postings_lookup(
        postings_layout,
        token_postings_dir,
        max_row_id_exclusive=max_row_id_exclusive,
        postings_workers=postings_workers,
        postings_batch_rows=postings_batch_rows,
        postings_token_batch_size=postings_token_batch_size,
    )
    token_index: TokenTextIndex | None = None
    if postings_layout == "duckdb_token_text_parquet":
        token_index = load_gt_token_index(token_map, token_ngram_index_path, ngram_size, token_map_batch_size)

    assigned_items = list(assignment["items"])
    exact_work_items = [item for item in assigned_items if item.subtype == "exact"]
    fuzzy_work_items = [item for item in assigned_items if item.subtype == "fuzzy"]
    print(
        f"[gt worker {assignment['worker_idx']}] gpu={device_id} exact={len(exact_work_items)} "
        f"fuzzy={len(fuzzy_work_items)} cost={assignment['cost']}",
        flush=True,
    )

    final_ids = np.full((query_count, topk), UINT32_MAX, dtype=np.uint32)
    final_dist = np.full((query_count, topk), np.inf, dtype=np.float32)
    candidate_counts = np.zeros((query_count,), dtype=np.int64)
    expanded_token_counts = np.full((query_count,), -1, dtype=np.int64)

    started = time.monotonic()

    postings_row_budget = max(1, int(postings_batch_rows))
    exact_work_batches = build_exact_gt_work_batches(
        exact_work_items,
        token_lookup,
        postings_row_budget=postings_row_budget,
        query_batch_size=query_batch_size,
    )
    processed_exact_groups = 0
    for batch_idx, work_batch in enumerate(exact_work_batches, start=1):
        batch_qids = np.concatenate([item.qids for item in work_batch.items]).astype(np.int64, copy=False)
        batch_query_vectors = load_query_vectors(query_records, batch_qids)
        batch_best_ids_cpu = np.full((batch_qids.size, topk), UINT32_MAX, dtype=np.uint32)
        batch_best_dist_cpu = np.full((batch_qids.size, topk), np.inf, dtype=np.float32)
        batch_candidate_counts: dict[int, int] = {}
        qid_to_local: dict[int, int] = {int(qid): idx for idx, qid in enumerate(batch_qids.tolist())}
        item_query_positions: list[np.ndarray] = []
        for item_idx, item in enumerate(work_batch.items):
            local_positions = np.asarray([qid_to_local[int(qid)] for qid in item.qids.tolist()], dtype=np.int64)
            item_query_positions.append(local_positions)
            batch_candidate_counts[item_idx] = 0

        if not batch_qids.size:
            processed_exact_groups += len(work_batch.items)
            continue

        batch_query_gpu = torch.from_numpy(np.asarray(batch_query_vectors, dtype=np.float64)).to(
            device=device,
            dtype=torch.float64,
        )
        batch_query_norm = torch.sum(batch_query_gpu * batch_query_gpu, dim=1)
        per_item_best_ids_cpu: dict[int, np.ndarray] = {}
        per_item_best_dist_cpu: dict[int, np.ndarray] = {}
        union_parts: list[np.ndarray] = []
        item_row_parts: dict[int, list[np.ndarray]] = defaultdict(list)

        for item_idx, item in enumerate(work_batch.items):
            token_ids = np.asarray(
                item.token_ids if item.token_ids is not None else np.empty((0,), dtype=np.uint32),
                dtype=np.uint32,
            )
            token_batches = token_id_batches_for_gt_work(
                token_ids,
                postings_layout,
                token_lookup,
                postings_row_budget,
                postings_token_batch_size,
            )
            item_candidate_count = 0
            for token_batch in token_batches:
                batch_rows = load_posting_rows_for_token_batch(
                    token_batch,
                    postings_layout,
                    token_lookup,
                    token_index,
                )
                if batch_rows.size == 0:
                    continue
                item_candidate_count += int(batch_rows.size)
                batch_rows = batch_rows.astype(np.uint32, copy=False)
                union_parts.append(batch_rows)
                item_row_parts[item_idx].append(batch_rows)
            batch_candidate_counts[item_idx] = item_candidate_count

        if not union_parts:
            processed_exact_groups += len(work_batch.items)
            continue

        union_rows = np.concatenate(union_parts).astype(np.uint32, copy=False)
        if union_rows.size > 1:
            union_rows = np.unique(union_rows)
        item_union_positions: list[np.ndarray] = []
        for item_idx, _item in enumerate(work_batch.items):
            item_rows = item_row_parts.get(item_idx, [])
            if not item_rows:
                item_union_positions.append(np.empty((0,), dtype=np.int64))
                continue
            joined = np.concatenate(item_rows).astype(np.uint32, copy=False)
            if joined.size > 1:
                joined = np.unique(joined)
            item_union_positions.append(
                np.searchsorted(union_rows, joined).astype(np.int64, copy=False)
            )

        row_step = max(1, int(row_batch_size))
        query_pos_gpu_cache: dict[int, Any] = {
            item_idx: torch.from_numpy(local_positions.astype(np.int64, copy=False)).to(device=device)
            for item_idx, local_positions in enumerate(item_query_positions)
            if local_positions.size
        }
        with torch.no_grad():
            for row_begin in range(0, union_rows.size, row_step):
                row_end = min(union_rows.size, row_begin + row_step)
                rows_np = union_rows[row_begin:row_end].astype(np.int64, copy=False)
                base_np = base_vector_provider.fetch_rows(rows_np, np.float16)
                base_gpu = torch.from_numpy(base_np).to(device=device, dtype=torch.float64)
                base_norm = torch.sum(base_gpu * base_gpu, dim=1)
                dist = batch_query_norm[:, None] + base_norm[None, :] - 2.0 * torch.matmul(batch_query_gpu, base_gpu.T)
                row_ids_gpu = torch.from_numpy(rows_np.astype(np.int64, copy=False)).to(device=device)
                for item_idx, item in enumerate(work_batch.items):
                    local_positions = item_query_positions[item_idx]
                    union_pos = item_union_positions[item_idx]
                    if union_pos.size == 0 or local_positions.size == 0:
                        continue
                    lo = np.searchsorted(union_pos, row_begin, side="left")
                    hi = np.searchsorted(union_pos, row_end, side="left")
                    if hi <= lo:
                        continue
                    cols_np = (union_pos[lo:hi] - row_begin).astype(np.int64, copy=False)
                    cols_gpu = torch.from_numpy(cols_np).to(device=device)
                    query_pos_gpu = query_pos_gpu_cache[item_idx]
                    item_dist = dist.index_select(0, query_pos_gpu).index_select(1, cols_gpu)
                    if int(item_dist.shape[1]) <= 0:
                        continue
                    item_row_ids_cpu = row_ids_gpu.index_select(0, cols_gpu).cpu().numpy().astype(np.uint32, copy=False)
                    item_dist_cpu = item_dist.cpu().numpy().astype(np.float64, copy=False)
                    chunk_ids_cpu, chunk_dist_cpu = topk_from_distance_matrix(
                        item_dist_cpu,
                        item_row_ids_cpu,
                        topk,
                    )
                    best_ids_cpu = per_item_best_ids_cpu.get(item_idx)
                    best_dist_cpu = per_item_best_dist_cpu.get(item_idx)
                    if best_ids_cpu is None or best_dist_cpu is None:
                        best_ids_cpu = chunk_ids_cpu
                        best_dist_cpu = chunk_dist_cpu
                    else:
                        merged_ids = np.empty((chunk_ids_cpu.shape[0], topk), dtype=np.uint32)
                        merged_dist = np.empty((chunk_ids_cpu.shape[0], topk), dtype=np.float32)
                        for local_idx in range(chunk_ids_cpu.shape[0]):
                            merged_ids[local_idx], merged_dist[local_idx] = merge_topk_cpu(
                                [best_ids_cpu[local_idx], chunk_ids_cpu[local_idx]],
                                [best_dist_cpu[local_idx], chunk_dist_cpu[local_idx]],
                                topk,
                            )
                        best_ids_cpu = merged_ids
                        best_dist_cpu = merged_dist
                    per_item_best_ids_cpu[item_idx] = best_ids_cpu
                    per_item_best_dist_cpu[item_idx] = best_dist_cpu
                del base_gpu, base_norm, dist, row_ids_gpu

        for item_idx, item in enumerate(work_batch.items):
            local_positions = item_query_positions[item_idx]
            best_ids_cpu = per_item_best_ids_cpu.get(item_idx)
            best_dist_cpu = per_item_best_dist_cpu.get(item_idx)
            if best_ids_cpu is None or best_dist_cpu is None:
                continue
            width = int(best_ids_cpu.shape[1])
            batch_best_ids_cpu[local_positions, :width] = best_ids_cpu
            batch_best_dist_cpu[local_positions, :width] = best_dist_cpu

        final_ids[batch_qids] = batch_best_ids_cpu
        final_dist[batch_qids] = batch_best_dist_cpu
        for item_idx, item in enumerate(work_batch.items):
            candidate_counts[item.qids] = batch_candidate_counts.get(item_idx, 0)
            expanded_token_counts[item.qids] = item.token_count
        processed_exact_groups += len(work_batch.items)
        if progress_interval := int(config.get("progress_interval", 0)):
            if batch_idx % progress_interval == 0 or batch_idx == len(exact_work_batches):
                print(
                    f"[gt worker {assignment['worker_idx']}] exact_batch {batch_idx}/{len(exact_work_batches)} "
                    f"groups={len(work_batch.items)} union_rows={int(union_rows.size)} "
                    f"processed_groups={processed_exact_groups}/{len(exact_work_items)}",
                    flush=True,
                )

    visited_rows_local = np.zeros((max_row_id_exclusive,), dtype=np.bool_) if fuzzy_work_items else None
    for fuzzy_idx, item in enumerate(fuzzy_work_items, start=1):
        token_ids = np.asarray(item.token_ids if item.token_ids is not None else np.empty((0,), dtype=np.uint32), dtype=np.uint32)
        expanded_token_counts[item.qids] = int(item.token_count)
        if token_ids.size == 0:
            candidate_counts[item.qids] = 0
            if progress_interval := int(config.get("progress_interval", 0)):
                if fuzzy_idx % progress_interval == 0 or fuzzy_idx == len(fuzzy_work_items):
                    print(
                        f"[gt worker {assignment['worker_idx']}] fuzzy {fuzzy_idx}/{len(fuzzy_work_items)} q={int(item.qids.size)} candidates=0",
                        flush=True,
                    )
            continue

        token_batches = token_id_batches_for_gt_work(
            token_ids,
            postings_layout,
            token_lookup,
            postings_batch_rows,
            postings_token_batch_size,
        )
        item_query_vectors = load_query_vectors(query_records, item.qids)
        item_query_gpu = torch.from_numpy(np.asarray(item_query_vectors, dtype=np.float32)).to(
            device=device,
            dtype=torch.float32,
        )
        item_query_norm = torch.sum(item_query_gpu * item_query_gpu, dim=1)
        item_best_ids_gpu = None
        item_best_dist_gpu = None
        item_candidate_count = 0
        touched_rows_local: list[np.ndarray] = []

        with torch.no_grad():
            for token_batch in token_batches:
                batch_rows = load_posting_rows_for_token_batch(
                    token_batch,
                    postings_layout,
                    token_lookup,
                    token_index,
                )
                if batch_rows.size == 0:
                    continue
                if visited_rows_local is not None:
                    new_mask = ~visited_rows_local[batch_rows]
                    if not np.any(new_mask):
                        continue
                    batch_rows = batch_rows[new_mask]
                    visited_rows_local[batch_rows] = True
                    touched_rows_local.append(batch_rows)
                if batch_rows.size == 0:
                    continue
                item_candidate_count += int(batch_rows.size)
                item_best_ids_gpu, item_best_dist_gpu = gpu_stream_topk_for_candidate_rows(
                    torch,
                    base_vector_provider,
                    item_query_gpu,
                    item_query_norm,
                    batch_rows.astype(np.uint32, copy=False),
                    topk,
                    device,
                    row_batch=row_batch_size,
                    best_ids_gpu=item_best_ids_gpu,
                    best_dist_gpu=item_best_dist_gpu,
                )

        if item_best_ids_gpu is not None and item_best_dist_gpu is not None:
            item_ids = item_best_ids_gpu.cpu().numpy().astype(np.uint32, copy=False)
            item_dists = item_best_dist_gpu.cpu().numpy().astype(np.float32, copy=False)
            width = int(item_ids.shape[1])
            final_ids[item.qids, :width] = item_ids
            final_dist[item.qids, :width] = item_dists
        candidate_counts[item.qids] = item_candidate_count
        if visited_rows_local is not None:
            for touched in touched_rows_local:
                visited_rows_local[touched] = False
        if progress_interval := int(config.get("progress_interval", 0)):
            if fuzzy_idx % progress_interval == 0 or fuzzy_idx == len(fuzzy_work_items):
                print(
                    f"[gt worker {assignment['worker_idx']}] fuzzy {fuzzy_idx}/{len(fuzzy_work_items)} "
                    f"q={int(item.qids.size)} candidates={item_candidate_count}",
                    flush=True,
                )

    elapsed = time.monotonic() - started
    final_ids, final_dist = stable_sort_topk_rows(final_ids, final_dist)
    hit_counts = np.count_nonzero(final_ids != UINT32_MAX, axis=1)
    return {
        "ids": final_ids,
        "dist": final_dist,
        "candidate_counts": candidate_counts,
        "expanded_token_counts": expanded_token_counts,
        "elapsed_seconds": float(elapsed),
        "worker_idx": int(assignment["worker_idx"]),
        "device_id": int(device_id),
        "exact_items": len(exact_work_items),
        "fuzzy_items": len(fuzzy_work_items),
        "hit_count_stats": numeric_stats(hit_counts),
        "missing_topk_rows": int(np.count_nonzero(hit_counts < topk)),
    }


def compute_string_gpu_gt_multi(
    *,
    queries: list[RuntimeQuery],
    gt_queries: list[StringGTQuery],
    exact_work_items: list[GTWorkItem],
    fuzzy_work_items: list[GTWorkItem],
    query_vector_path: Path,
    base_hvecs: Path,
    token_map: Path,
    token_postings_dir: Path,
    output_root: Path,
    jsonl_output: Path,
    topk: int,
    device_name: str,
    row_batch_size: int,
    query_batch_size: int,
    token_map_batch_size: int,
    ngram_size: int,
    token_ngram_index_path: Path | None,
    postings_workers: int,
    postings_batch_rows: int,
    postings_token_batch_size: int,
    host_vector_cache_gb: float,
    max_row_id_exclusive: int,
    progress_interval: int,
    overwrite: bool,
    host_warmup: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for --compute-gt GPU top-k; install torch in the active environment.") from exc

    device_ids = resolve_cuda_device_ids(device_name)
    if len(device_ids) <= 1:
        raise ValueError("multi-GPU helper called with fewer than two devices")

    base_info = inspect_hvecs(base_hvecs)
    query_info = inspect_hvecs(query_vector_path)
    if base_info.dim != query_info.dim:
        raise ValueError(f"base/query dim mismatch for GT: {base_info.dim} vs {query_info.dim}")
    if query_info.rows != len(queries):
        raise ValueError(f"query.hvecs rows {query_info.rows} != selected queries {len(queries)}")
    if max_row_id_exclusive <= 0:
        max_row_id_exclusive = int(base_info.rows)
    if max_row_id_exclusive > base_info.rows:
        raise ValueError(f"--gt-max-row-id-exclusive {max_row_id_exclusive} exceeds base rows {base_info.rows}")

    started = time.monotonic()
    postings_layout = detect_token_postings_layout(token_postings_dir)
    token_lookup = build_token_postings_lookup(
        postings_layout,
        token_postings_dir,
        max_row_id_exclusive=max_row_id_exclusive,
        postings_workers=postings_workers,
        postings_batch_rows=postings_batch_rows,
        postings_token_batch_size=postings_token_batch_size,
    )
    assignments = assign_gt_work_items(exact_work_items + fuzzy_work_items, device_ids, token_lookup)
    assignments = [assignment for assignment in assignments if assignment["items"]]
    for worker_idx, assignment in enumerate(assignments):
        assignment["worker_idx"] = worker_idx

    if host_vector_cache_gb > 0:
        print("[gt] multi-GPU path uses direct host vectors; ignoring host cache request", flush=True)

    cpu_threads = max(1, int((os.cpu_count() or 1) // max(1, len(assignments))))
    config = {
        "base_hvecs": str(base_hvecs),
        "query_vector_path": str(query_vector_path),
        "token_map": str(token_map),
        "token_ngram_index_path": None if token_ngram_index_path is None else str(token_ngram_index_path),
        "token_postings_dir": str(token_postings_dir),
        "query_count": len(gt_queries),
        "topk": int(topk),
        "row_batch_size": int(row_batch_size),
        "query_batch_size": int(query_batch_size),
        "token_map_batch_size": int(token_map_batch_size),
        "ngram_size": int(ngram_size),
        "postings_layout": postings_layout,
        "postings_workers": int(postings_workers),
        "postings_batch_rows": int(postings_batch_rows),
        "postings_token_batch_size": int(postings_token_batch_size),
        "max_row_id_exclusive": int(max_row_id_exclusive),
        "progress_interval": int(progress_interval),
        "cpu_threads": int(cpu_threads),
    }
    print(f"[gt] multi-GPU devices={device_ids} cpu_threads_per_worker={cpu_threads}", flush=True)
    for assignment in assignments:
        print(
            f"[gt] assignment worker={assignment['worker_idx']} gpu={assignment['device_id']} "
            f"items={len(assignment['items'])} cost={assignment['cost']}",
            flush=True,
        )

    if not assignments:
        if gt_queries:
            raise RuntimeError("multi-GPU GT produced no assigned work items for a non-empty query set")
        final_ids = np.full((0, topk), UINT32_MAX, dtype=np.uint32)
        final_dist = np.full((0, topk), np.inf, dtype=np.float32)
        candidate_counts = np.zeros((0,), dtype=np.int64)
        expanded_token_counts = np.zeros((0,), dtype=np.int64)
        summary = {
            "source": {
                "query_vector": str(query_vector_path),
                "base_hvecs": str(base_hvecs),
                "token_map": str(token_map),
                "token_ngram_index": None if token_ngram_index_path is None else str(token_ngram_index_path),
                "token_postings_dir": str(token_postings_dir),
            },
            "query_count": 0,
            "subtype_counts": {},
            "unique_predicate_groups": 0,
            "topk": int(topk),
            "device": device_name,
            "devices": device_ids,
            "max_row_id_exclusive": int(max_row_id_exclusive),
            "vectors": {
                "base_rows": int(base_info.rows),
                "query_rows": int(query_info.rows),
                "dim": int(base_info.dim),
                "scalar_type": "float16_hvecs",
            },
            "base_vector_provider": {
                "mode": "memmap_direct",
                "workers": 0,
                "cpu_threads_per_worker": 0,
            },
            "candidate_count_stats": numeric_stats(candidate_counts),
            "expanded_token_count_stats": numeric_stats(expanded_token_counts),
            "hit_count_stats": numeric_stats(np.zeros((0,), dtype=np.int64)),
            "missing_topk_rows": 0,
            "elapsed_seconds": float(time.monotonic() - started),
            "worker_elapsed_seconds_sum": 0.0,
            "assignments": [],
            "notes": [
                "multi-GPU GT shards exact/fuzzy work items across devices and uses direct host vectors; host cache is disabled.",
                "Fuzzy expansion uses the full token map and token postings remain the source of truth for GT candidates.",
            ],
        }
        topk_info = write_gt_ibin(output_root / f"gt.string.top{topk}.ibin", final_ids, overwrite)
        dist_info = write_dist_fbin(output_root / f"gt.string.top{topk}.dist.fbin", final_dist, overwrite)
        outputs: dict[str, Any] = {
            f"top{topk}_gt": topk_info,
            f"top{topk}_distances": dist_info,
        }
        if topk > 10:
            outputs["top10_gt"] = write_gt_ibin(output_root / "gt.string.top10.ibin", final_ids[:, :10], overwrite)
            outputs["top10_distances"] = write_dist_fbin(
                output_root / "gt.string.top10.dist.fbin",
                final_dist[:, :10],
                overwrite,
            )
        jsonl_info = write_gt_rows_jsonl(
            jsonl_output,
            gt_queries,
            final_ids,
            final_dist,
            candidate_counts,
            expanded_token_counts,
            overwrite,
        )
        outputs["rows_jsonl"] = jsonl_info
        summary["outputs"] = outputs
        summary_path = output_root / "gt.string.summary.json"
        summary["summary_path"] = str(summary_path)
        ensure_output_path(summary_path, overwrite)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[gt] wrote {topk_info['path']}", flush=True)
        if topk > 10 and "top10_gt" in outputs:
            print(f"[gt] wrote {outputs['top10_gt']['path']}", flush=True)
        print(f"[gt] wrote summary {summary_path}", flush=True)
        return summary

    if len(assignments) == 1:
        worker_results = [compute_string_gpu_gt_worker(config, assignments[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(assignments)) as pool:
            worker_results = pool.starmap(
                compute_string_gpu_gt_worker,
                [(config, assignment) for assignment in assignments],
            )

    final_ids = np.full((len(gt_queries), topk), UINT32_MAX, dtype=np.uint32)
    final_dist = np.full((len(gt_queries), topk), np.inf, dtype=np.float32)
    candidate_counts = np.zeros((len(gt_queries),), dtype=np.int64)
    expanded_token_counts = np.full((len(gt_queries),), -1, dtype=np.int64)
    filled_mask = np.zeros((len(gt_queries),), dtype=bool)
    for worker_result in worker_results:
        mask = worker_result["expanded_token_counts"] >= 0
        if not np.any(mask):
            continue
        overlap = filled_mask & mask
        if np.any(overlap):
            raise RuntimeError("multi-GPU GT merge detected overlapping query rows between workers")
        final_ids[mask] = worker_result["ids"][mask]
        final_dist[mask] = worker_result["dist"][mask]
        candidate_counts[mask] = worker_result["candidate_counts"][mask]
        expanded_token_counts[mask] = worker_result["expanded_token_counts"][mask]
        filled_mask[mask] = True

    if not np.all(filled_mask):
        missing = np.flatnonzero(~filled_mask)
        preview = ", ".join(str(int(idx)) for idx in missing[:16])
        raise RuntimeError(f"multi-GPU GT missing rows for query ids: {preview}")

    final_ids, final_dist = stable_sort_topk_rows(final_ids, final_dist)
    hit_counts = np.count_nonzero(final_ids != UINT32_MAX, axis=1)
    summary = {
        "source": {
            "query_vector": str(query_vector_path),
            "base_hvecs": str(base_hvecs),
            "token_map": str(token_map),
            "token_ngram_index": None if token_ngram_index_path is None else str(token_ngram_index_path),
            "token_postings_dir": str(token_postings_dir),
        },
        "query_count": len(gt_queries),
        "subtype_counts": dict(Counter(query.subtype for query in gt_queries)),
        "unique_predicate_groups": len(exact_work_items) + len(fuzzy_work_items),
        "topk": int(topk),
        "device": device_name,
        "devices": device_ids,
        "max_row_id_exclusive": int(max_row_id_exclusive),
        "vectors": {
            "base_rows": int(base_info.rows),
            "query_rows": int(query_info.rows),
            "dim": int(base_info.dim),
            "scalar_type": "float16_hvecs",
        },
        "base_vector_provider": {
            "mode": "memmap_direct",
            "workers": len(assignments),
            "cpu_threads_per_worker": int(cpu_threads),
        },
        "host_warmup": [] if host_warmup is None else host_warmup,
        "candidate_count_stats": numeric_stats(candidate_counts),
        "expanded_token_count_stats": numeric_stats(expanded_token_counts[expanded_token_counts >= 0]),
        "hit_count_stats": numeric_stats(hit_counts),
        "missing_topk_rows": int(np.count_nonzero(hit_counts < topk)),
        "elapsed_seconds": float(time.monotonic() - started),
        "worker_elapsed_seconds_sum": float(sum(float(worker_result["elapsed_seconds"]) for worker_result in worker_results)),
        "assignments": [
            {
                "worker_idx": int(assignment["worker_idx"]),
                "device_id": int(assignment["device_id"]),
                "items": int(len(assignment["items"])),
                "cost": int(assignment["cost"]),
            }
            for assignment in assignments
        ],
        "notes": [
            "multi-GPU GT shards exact/fuzzy work items across devices and uses direct host vectors; host cache is disabled.",
            "Fuzzy expansion uses the full token map and token postings remain the source of truth for GT candidates.",
        ],
    }

    final_ids, final_dist = stable_sort_topk_rows(final_ids, final_dist)
    topk_info = write_gt_ibin(output_root / f"gt.string.top{topk}.ibin", final_ids, overwrite)
    dist_info = write_dist_fbin(output_root / f"gt.string.top{topk}.dist.fbin", final_dist, overwrite)
    outputs: dict[str, Any] = {
        f"top{topk}_gt": topk_info,
        f"top{topk}_distances": dist_info,
    }
    if topk > 10:
        outputs["top10_gt"] = write_gt_ibin(output_root / "gt.string.top10.ibin", final_ids[:, :10], overwrite)
        outputs["top10_distances"] = write_dist_fbin(
            output_root / "gt.string.top10.dist.fbin",
            final_dist[:, :10],
            overwrite,
        )
    jsonl_info = write_gt_rows_jsonl(
        jsonl_output,
        gt_queries,
        final_ids,
        final_dist,
        candidate_counts,
        expanded_token_counts,
        overwrite,
    )
    outputs["rows_jsonl"] = jsonl_info
    summary["outputs"] = outputs
    summary_path = output_root / "gt.string.summary.json"
    summary["summary_path"] = str(summary_path)
    ensure_output_path(summary_path, overwrite)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[gt] wrote {topk_info['path']}", flush=True)
    if topk > 10 and "top10_gt" in outputs:
        print(f"[gt] wrote {outputs['top10_gt']['path']}", flush=True)
    print(f"[gt] wrote summary {summary_path}", flush=True)
    return summary


def compute_string_gpu_gt(
    *,
    queries: list[RuntimeQuery],
    query_vector_path: Path,
    base_hvecs: Path,
    token_map: Path,
    token_postings_dir: Path,
    output_root: Path,
    jsonl_output: Path,
    topk: int,
    device_name: str,
    row_batch_size: int,
    query_batch_size: int,
    token_map_batch_size: int,
    ngram_size: int,
    token_ngram_index_path: Path | None,
    postings_workers: int,
    postings_batch_rows: int,
    postings_token_batch_size: int,
    host_vector_cache_gb: float,
    max_row_id_exclusive: int,
    progress_interval: int,
    overwrite: bool,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for --compute-gt GPU top-k; install torch in the active environment.") from exc

    if topk <= 0:
        raise ValueError("--gt-topk must be positive")
    if not base_hvecs.exists():
        raise FileNotFoundError(f"--compute-gt requires --base-hvecs to exist: {base_hvecs}")
    if not token_postings_dir.exists():
        raise FileNotFoundError(f"--compute-gt token postings dir does not exist: {token_postings_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; --compute-gt requires a CUDA-capable PyTorch install")

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device_ids = resolve_cuda_device_ids(device_name)
    device_id = int(device_ids[0])

    base_info = inspect_hvecs(base_hvecs)
    query_info = inspect_hvecs(query_vector_path)
    if base_info.dim != query_info.dim:
        raise ValueError(f"base/query dim mismatch for GT: {base_info.dim} vs {query_info.dim}")
    if query_info.rows != len(queries):
        raise ValueError(f"query.hvecs rows {query_info.rows} != selected queries {len(queries)}")
    if max_row_id_exclusive <= 0:
        max_row_id_exclusive = int(base_info.rows)
    if max_row_id_exclusive > base_info.rows:
        raise ValueError(f"--gt-max-row-id-exclusive {max_row_id_exclusive} exceeds base rows {base_info.rows}")

    postings_layout = detect_token_postings_layout(token_postings_dir)
    host_warmup = warm_gt_inputs_to_page_cache(
        base_hvecs=base_hvecs,
        token_postings_dir=token_postings_dir,
        token_ngram_index_path=token_ngram_index_path,
        postings_layout=postings_layout,
    )
    token_index: TokenTextIndex | None = None
    gt_queries = load_string_gt_queries(queries, token_index=None)
    subtype_counts = Counter(query.subtype for query in gt_queries)
    need_token_index = bool(subtype_counts.get("fuzzy", 0)) or postings_layout == "duckdb_token_text_parquet"
    if need_token_index:
        if token_ngram_index_path is not None:
            print(f"[gt] loading fuzzy/text index from ngram binary {token_ngram_index_path}", flush=True)
            token_index = load_token_text_index_from_ngram_binary(token_ngram_index_path)
        else:
            print(f"[gt] loading token_map fuzzy/text index from {token_map}", flush=True)
            token_index = load_token_text_index(token_map, ngram_size=ngram_size, batch_size=token_map_batch_size)
        gt_queries = load_string_gt_queries(queries, token_index=token_index)

    base_records = hvecs_records(base_hvecs, base_info)
    query_records = hvecs_records(query_vector_path, query_info)
    cache_rows = 0
    if host_vector_cache_gb > 0 and len(device_ids) == 1:
        bytes_per_row = int(base_info.dim) * np.dtype(np.float16).itemsize
        cache_rows = max(0, int((host_vector_cache_gb * (1024 ** 3)) // bytes_per_row))
    if cache_rows > 0:
        base_vector_provider: BaseVectorProvider = CachedBaseVectorProvider(
            base_records,
            max_rows=cache_rows,
            dim=int(base_info.dim),
            cache_dtype=np.float16,
        )
        print(f"[gt] enabling host vector cache rows={cache_rows} approx_gb={host_vector_cache_gb:.2f}", flush=True)
    else:
        base_vector_provider = BaseVectorProvider(base_records)
    if postings_layout == "token_id_binary":
        token_lookup: Any = TokenIdPostingsBinaryLookup(
            token_postings_dir,
            max_row_id_exclusive=max_row_id_exclusive,
        )
    elif postings_layout == "token_id_npy":
        token_lookup: Any = TokenIdPostingsNpyLookup(
            token_postings_dir,
            max_row_id_exclusive=max_row_id_exclusive,
        )
    elif postings_layout == "duckdb_token_id_parquet":
        token_lookup = TokenIdPostingsDuckDBLookup(
            token_postings_dir,
            max_row_id_exclusive=max_row_id_exclusive,
            workers=postings_workers,
            batch_rows=postings_batch_rows,
            token_batch_size=postings_token_batch_size,
        )
    else:
        token_lookup = TokenPostingsDuckDBLookup(
            token_postings_dir,
            max_row_id_exclusive=max_row_id_exclusive,
            workers=postings_workers,
            batch_rows=postings_batch_rows,
            token_batch_size=postings_token_batch_size,
        )

    final_ids = np.full((len(gt_queries), topk), UINT32_MAX, dtype=np.uint32)
    final_dist = np.full((len(gt_queries), topk), np.inf, dtype=np.float32)
    candidate_counts = np.zeros((len(gt_queries),), dtype=np.int64)
    expanded_token_counts = np.full((len(gt_queries),), -1, dtype=np.int64)
    postings_row_budget = max(1, int(postings_batch_rows))

    groups: defaultdict[str, list[StringGTQuery]] = defaultdict(list)
    for query in gt_queries:
        groups[query.predicate_key].append(query)

    group_items = list(groups.items())
    visited_rows_shared = np.zeros((max_row_id_exclusive,), dtype=np.bool_) if subtype_counts.get("fuzzy", 0) else None
    exact_token_ids = sorted({int(query.token_id) for query in gt_queries if query.subtype == "exact" and query.token_id is not None})
    if exact_token_ids:
        if postings_layout == "token_id_binary":
            print(f"[gt] binary exact postings are offset-addressable; skipping eager preload for {len(exact_token_ids)} token(s)", flush=True)
        else:
            print(f"[gt] preloading {len(exact_token_ids)} exact token posting set(s)", flush=True)
        if postings_layout in {"token_id_npy", "duckdb_token_id_parquet"}:
            token_lookup.load_token_ids(exact_token_ids)
        else:
            if postings_layout == "token_id_binary":
                pass
            elif token_index is None:
                raise RuntimeError("text-token postings require token_index for exact preloading")
            else:
                exact_tokens = [
                    token_index.id_to_text[token_id]
                    for token_id in exact_token_ids
                ]
                token_lookup.load_tokens(exact_tokens)
    fuzzy_expansion_cache: dict[tuple[str, str], np.ndarray] = {}
    if postings_layout in {"token_id_binary", "token_id_npy"} and hasattr(token_lookup, "posting_size"):
        def _group_sort_key(item: tuple[str, list[StringGTQuery]]) -> tuple[int, str]:
            first_query = item[1][0]
            if first_query.subtype == "exact" and first_query.token_id is not None:
                return (-int(token_lookup.posting_size(int(first_query.token_id))), item[0])
            return (1, item[0])

        group_items.sort(key=_group_sort_key)
    else:
        group_items.sort(key=lambda item: item[0])
    print(
        f"[gt] selected={len(gt_queries)} groups={len(group_items)} "
        f"subtypes={dict(subtype_counts)} device={device_name}",
        flush=True,
    )

    exact_work_items: list[GTWorkItem] = []
    fuzzy_work_items: list[GTWorkItem] = []
    for group_key, group_queries in group_items:
        first = group_queries[0]
        if first.subtype == "exact":
            if first.token_id is None:
                raise ValueError("exact GT query missing token_id")
            token_count = 1
            qids = np.asarray([query.local_idx for query in group_queries], dtype=np.int64)
            exact_work_items.append(
                GTWorkItem(
                    group_key=group_key,
                    subtype=first.subtype,
                    qids=qids,
                    token_count=token_count,
                    token_ids=np.asarray([int(first.token_id)], dtype=np.uint32),
                )
            )
        else:
            if first.fuzzy_mode is None or first.fuzzy_pattern is None:
                raise ValueError("fuzzy GT query missing mode/pattern")
            if token_index is None:
                raise RuntimeError("fuzzy GT requires token_index")
            cache_key = (first.fuzzy_mode, first.fuzzy_pattern)
            token_ids = fuzzy_expansion_cache.get(cache_key)
            if token_ids is None:
                token_ids = fuzzy_matching_token_ids(first.fuzzy_mode, first.fuzzy_pattern, token_index)
                fuzzy_expansion_cache[cache_key] = token_ids
            token_count = int(token_ids.size)
            qids = np.asarray([query.local_idx for query in group_queries], dtype=np.int64)
            fuzzy_work_items.append(
                GTWorkItem(
                    group_key=group_key,
                    subtype=first.subtype,
                    qids=qids,
                    token_count=token_count,
                    token_ids=token_ids.astype(np.uint32, copy=False),
                )
            )

    if len(device_ids) > 1:
        return compute_string_gpu_gt_multi(
            queries=queries,
            gt_queries=gt_queries,
            exact_work_items=exact_work_items,
            fuzzy_work_items=fuzzy_work_items,
            query_vector_path=query_vector_path,
            base_hvecs=base_hvecs,
            token_map=token_map,
            token_postings_dir=token_postings_dir,
            output_root=output_root,
            jsonl_output=jsonl_output,
            topk=topk,
            device_name=device_name,
            row_batch_size=row_batch_size,
            query_batch_size=query_batch_size,
            token_map_batch_size=token_map_batch_size,
            ngram_size=ngram_size,
            token_ngram_index_path=token_ngram_index_path,
            postings_workers=postings_workers,
            postings_batch_rows=postings_batch_rows,
            postings_token_batch_size=postings_token_batch_size,
            host_vector_cache_gb=host_vector_cache_gb,
            max_row_id_exclusive=max_row_id_exclusive,
            progress_interval=progress_interval,
            overwrite=overwrite,
            host_warmup=host_warmup,
        )

    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)

    work_batches: list[GTWorkBatch] = []
    current_items: list[GTWorkItem] = []
    current_union_row_budget = 0
    current_union_token_count = 0
    current_query_count = 0
    exact_merge_row_budget = max(postings_row_budget, int(postings_row_budget * 2))
    for item in exact_work_items:
        token_ids = np.asarray(item.token_ids if item.token_ids is not None else np.empty((0,), dtype=np.uint32), dtype=np.uint32)
        item_budget = 1
        if token_ids.size > 0 and hasattr(token_lookup, "posting_size"):
            item_budget = max(1, sum(int(token_lookup.posting_size(int(token_id))) for token_id in token_ids.tolist()))
        if current_items and (
            current_union_row_budget + item_budget > exact_merge_row_budget
            or current_query_count + int(item.qids.size) > max(1, int(query_batch_size))
        ):
            current_items, current_union_row_budget, current_union_token_count, current_query_count = append_work_batch(
                work_batches,
                current_items,
                current_union_row_budget,
                current_union_token_count,
                current_query_count,
            )
        current_items.append(item)
        current_union_row_budget += item_budget
        current_union_token_count += item.token_count
        current_query_count += int(item.qids.size)
    current_items, current_union_row_budget, current_union_token_count, current_query_count = append_work_batch(
        work_batches,
        current_items,
        current_union_row_budget,
        current_union_token_count,
        current_query_count,
    )

    total_groups_processed = 0
    for batch_idx, work_batch in enumerate(work_batches, start=1):
        batch_qids = np.concatenate([item.qids for item in work_batch.items]).astype(np.int64, copy=False)
        batch_query_vectors = load_query_vectors(query_records, batch_qids)
        batch_query_gpu = None
        batch_query_norm = None
        batch_best_ids_cpu = np.full((batch_qids.size, topk), UINT32_MAX, dtype=np.uint32)
        batch_best_dist_cpu = np.full((batch_qids.size, topk), np.inf, dtype=np.float32)
        batch_candidate_counts: dict[int, int] = {}
        qid_to_local: dict[int, int] = {int(qid): idx for idx, qid in enumerate(batch_qids.tolist())}
        item_query_positions: list[np.ndarray] = []
        for item_idx, item in enumerate(work_batch.items):
            local_positions = np.asarray([qid_to_local[int(qid)] for qid in item.qids.tolist()], dtype=np.int64)
            item_query_positions.append(local_positions)
            batch_candidate_counts[item_idx] = 0

        if not batch_qids.size:
            continue

        batch_query_gpu = torch.from_numpy(np.asarray(batch_query_vectors, dtype=np.float64)).to(
            device=device,
            dtype=torch.float64,
        )
        batch_query_norm = torch.sum(batch_query_gpu * batch_query_gpu, dim=1)
        per_item_best_ids_cpu: dict[int, np.ndarray] = {}
        per_item_best_dist_cpu: dict[int, np.ndarray] = {}
        union_parts: list[np.ndarray] = []
        item_row_parts: dict[int, list[np.ndarray]] = defaultdict(list)

        for item_idx, item in enumerate(work_batch.items):
            token_ids = np.asarray(item.token_ids if item.token_ids is not None else np.empty((0,), dtype=np.uint32), dtype=np.uint32)
            token_batches = token_id_batches_for_gt_work(
                token_ids,
                postings_layout,
                token_lookup,
                postings_row_budget,
                postings_token_batch_size,
            )
            item_candidate_count = 0
            for token_batch in token_batches:
                batch_rows = load_posting_rows_for_token_batch(
                    token_batch,
                    postings_layout,
                    token_lookup,
                    token_index,
                )
                if batch_rows.size == 0:
                    continue
                item_candidate_count += int(batch_rows.size)
                batch_rows = batch_rows.astype(np.uint32, copy=False)
                union_parts.append(batch_rows)
                item_row_parts[item_idx].append(batch_rows)
            batch_candidate_counts[item_idx] = item_candidate_count

        if not union_parts:
            total_groups_processed += len(work_batch.items)
            continue

        union_rows = np.concatenate(union_parts).astype(np.uint32, copy=False)
        if union_rows.size > 1:
            union_rows = np.unique(union_rows)
        item_union_positions: list[np.ndarray] = []
        for item_idx, _item in enumerate(work_batch.items):
            item_rows = item_row_parts.get(item_idx, [])
            if not item_rows:
                item_union_positions.append(np.empty((0,), dtype=np.int64))
                continue
            joined = np.concatenate(item_rows).astype(np.uint32, copy=False)
            if joined.size > 1:
                joined = np.unique(joined)
            item_union_positions.append(
                np.searchsorted(union_rows, joined).astype(np.int64, copy=False)
            )

        row_step = max(1, int(row_batch_size))
        query_pos_gpu_cache: dict[int, Any] = {
            item_idx: torch.from_numpy(local_positions.astype(np.int64, copy=False)).to(device=device)
            for item_idx, local_positions in enumerate(item_query_positions)
            if local_positions.size
        }
        with torch.no_grad():
            for row_begin in range(0, union_rows.size, row_step):
                row_end = min(union_rows.size, row_begin + row_step)
                rows_np = union_rows[row_begin:row_end].astype(np.int64, copy=False)
                base_np = base_vector_provider.fetch_rows(rows_np, np.float16)
                base_gpu = torch.from_numpy(base_np).to(device=device, dtype=torch.float64)
                base_norm = torch.sum(base_gpu * base_gpu, dim=1)
                dist = batch_query_norm[:, None] + base_norm[None, :] - 2.0 * torch.matmul(batch_query_gpu, base_gpu.T)
                row_ids_gpu = torch.from_numpy(rows_np.astype(np.int64, copy=False)).to(device=device)
                for item_idx, item in enumerate(work_batch.items):
                    local_positions = item_query_positions[item_idx]
                    union_pos = item_union_positions[item_idx]
                    if union_pos.size == 0 or local_positions.size == 0:
                        continue
                    lo = np.searchsorted(union_pos, row_begin, side="left")
                    hi = np.searchsorted(union_pos, row_end, side="left")
                    if hi <= lo:
                        continue
                    cols_np = (union_pos[lo:hi] - row_begin).astype(np.int64, copy=False)
                    cols_gpu = torch.from_numpy(cols_np).to(device=device)
                    query_pos_gpu = query_pos_gpu_cache[item_idx]
                    item_dist = dist.index_select(0, query_pos_gpu).index_select(1, cols_gpu)
                    if int(item_dist.shape[1]) <= 0:
                        continue
                    item_row_ids_cpu = row_ids_gpu.index_select(0, cols_gpu).cpu().numpy().astype(np.uint32, copy=False)
                    item_dist_cpu = item_dist.cpu().numpy().astype(np.float64, copy=False)
                    chunk_ids_cpu, chunk_dist_cpu = topk_from_distance_matrix(
                        item_dist_cpu,
                        item_row_ids_cpu,
                        topk,
                    )
                    best_ids_cpu = per_item_best_ids_cpu.get(item_idx)
                    best_dist_cpu = per_item_best_dist_cpu.get(item_idx)
                    if best_ids_cpu is None or best_dist_cpu is None:
                        best_ids_cpu = chunk_ids_cpu
                        best_dist_cpu = chunk_dist_cpu
                    else:
                        merged_ids = np.empty((chunk_ids_cpu.shape[0], topk), dtype=np.uint32)
                        merged_dist = np.empty((chunk_ids_cpu.shape[0], topk), dtype=np.float32)
                        for local_idx in range(chunk_ids_cpu.shape[0]):
                            merged_ids[local_idx], merged_dist[local_idx] = merge_topk_cpu(
                                [best_ids_cpu[local_idx], chunk_ids_cpu[local_idx]],
                                [best_dist_cpu[local_idx], chunk_dist_cpu[local_idx]],
                                topk,
                            )
                        best_ids_cpu = merged_ids
                        best_dist_cpu = merged_dist
                    per_item_best_ids_cpu[item_idx] = best_ids_cpu
                    per_item_best_dist_cpu[item_idx] = best_dist_cpu
                del base_gpu, base_norm, dist, row_ids_gpu

        for item_idx, item in enumerate(work_batch.items):
            local_positions = item_query_positions[item_idx]
            best_ids_cpu = per_item_best_ids_cpu.get(item_idx)
            best_dist_cpu = per_item_best_dist_cpu.get(item_idx)
            if best_ids_cpu is None or best_dist_cpu is None:
                continue
            width = int(best_ids_cpu.shape[1])
            batch_best_ids_cpu[local_positions, :width] = best_ids_cpu
            batch_best_dist_cpu[local_positions, :width] = best_dist_cpu

        final_ids[batch_qids] = batch_best_ids_cpu
        final_dist[batch_qids] = batch_best_dist_cpu
        for item_idx, item in enumerate(work_batch.items):
            candidate_counts[item.qids] = batch_candidate_counts.get(item_idx, 0)
            expanded_token_counts[item.qids] = item.token_count
        total_groups_processed += len(work_batch.items)
        if progress_interval > 0 and (batch_idx % progress_interval == 0 or batch_idx == len(work_batches)):
            print(
                f"[gt] batch {batch_idx}/{len(work_batches)} subtype={work_batch.items[0].subtype} "
                f"groups={len(work_batch.items)} union_rows={union_rows.size} processed_groups={total_groups_processed}/{len(group_items)}",
                flush=True,
            )

    fuzzy_group_batch_rows = max(1, int(postings_row_budget))
    for fuzzy_group_idx, item in enumerate(fuzzy_work_items, start=1):
        token_ids = np.asarray(item.token_ids if item.token_ids is not None else np.empty((0,), dtype=np.uint32), dtype=np.uint32)
        expanded_token_counts[item.qids] = int(item.token_count)
        if token_ids.size == 0:
            candidate_counts[item.qids] = 0
            total_groups_processed += 1
            if progress_interval > 0 and (
                fuzzy_group_idx % progress_interval == 0 or fuzzy_group_idx == len(fuzzy_work_items)
            ):
                print(
                    f"[gt] fuzzy {fuzzy_group_idx}/{len(fuzzy_work_items)} tokens=0 candidates=0 processed_groups={total_groups_processed}/{len(group_items)}",
                    flush=True,
                )
            continue

        token_batches = token_id_batches_for_gt_work(
            token_ids,
            postings_layout,
            token_lookup,
            postings_row_budget,
            postings_token_batch_size,
        )
        item_query_vectors = load_query_vectors(query_records, item.qids)
        item_query_gpu = torch.from_numpy(np.asarray(item_query_vectors, dtype=np.float32)).to(
            device=device,
            dtype=torch.float32,
        )
        item_query_norm = torch.sum(item_query_gpu * item_query_gpu, dim=1)
        item_best_ids_gpu = None
        item_best_dist_gpu = None
        item_candidate_count = 0
        touched_rows_local: list[np.ndarray] = []

        with torch.no_grad():
            for token_batch in token_batches:
                batch_rows = load_posting_rows_for_token_batch(
                    token_batch,
                    postings_layout,
                    token_lookup,
                    token_index,
                )
                if batch_rows.size == 0:
                    continue
                if visited_rows_shared is not None:
                    new_mask = ~visited_rows_shared[batch_rows]
                    if not np.any(new_mask):
                        continue
                    batch_rows = batch_rows[new_mask]
                    visited_rows_shared[batch_rows] = True
                    touched_rows_local.append(batch_rows)
                if batch_rows.size == 0:
                    continue
                item_candidate_count += int(batch_rows.size)
                item_best_ids_gpu, item_best_dist_gpu = gpu_stream_topk_for_candidate_rows(
                    torch,
                    base_vector_provider,
                    item_query_gpu,
                    item_query_norm,
                    batch_rows.astype(np.uint32, copy=False),
                    topk,
                    device,
                    row_batch=min(int(row_batch_size), fuzzy_group_batch_rows),
                    best_ids_gpu=item_best_ids_gpu,
                    best_dist_gpu=item_best_dist_gpu,
                )

        if item_best_ids_gpu is not None and item_best_dist_gpu is not None:
            item_ids = item_best_ids_gpu.cpu().numpy().astype(np.uint32, copy=False)
            item_dists = item_best_dist_gpu.cpu().numpy().astype(np.float32, copy=False)
            width = int(item_ids.shape[1])
            final_ids[item.qids, :width] = item_ids
            final_dist[item.qids, :width] = item_dists
        candidate_counts[item.qids] = item_candidate_count
        if touched_rows_local:
            for touched in touched_rows_local:
                visited_rows_shared[touched] = False
        total_groups_processed += 1
        if progress_interval > 0 and (
            fuzzy_group_idx % progress_interval == 0 or fuzzy_group_idx == len(fuzzy_work_items)
        ):
            print(
                f"[gt] fuzzy {fuzzy_group_idx}/{len(fuzzy_work_items)} tokens={item.token_count} "
                f"candidates={item_candidate_count} processed_groups={total_groups_processed}/{len(group_items)}",
                flush=True,
            )

    if visited_rows_shared is not None:
        visited_rows_shared.fill(False)

    topk_info = write_gt_ibin(output_root / f"gt.string.top{topk}.ibin", final_ids, overwrite)
    dist_info = write_dist_fbin(output_root / f"gt.string.top{topk}.dist.fbin", final_dist, overwrite)
    outputs: dict[str, Any] = {
        f"top{topk}_gt": topk_info,
        f"top{topk}_distances": dist_info,
    }
    if topk > 10:
        outputs["top10_gt"] = write_gt_ibin(output_root / "gt.string.top10.ibin", final_ids[:, :10], overwrite)
        outputs["top10_distances"] = write_dist_fbin(
            output_root / "gt.string.top10.dist.fbin",
            final_dist[:, :10],
            overwrite,
        )
    jsonl_info = write_gt_rows_jsonl(
        jsonl_output,
        gt_queries,
        final_ids,
        final_dist,
        candidate_counts,
        expanded_token_counts,
        overwrite,
    )
    outputs["rows_jsonl"] = jsonl_info

    elapsed = time.monotonic() - started
    hit_counts = np.count_nonzero(final_ids != UINT32_MAX, axis=1)
    summary = {
        "source": {
            "query_vector": str(query_vector_path),
            "base_hvecs": str(base_hvecs),
            "token_map": str(token_map),
            "token_ngram_index": None if token_ngram_index_path is None else str(token_ngram_index_path),
            "token_postings_dir": str(token_postings_dir),
        },
        "outputs": outputs,
        "query_count": len(gt_queries),
        "subtype_counts": dict(subtype_counts),
        "unique_predicate_groups": len(group_items),
        "topk": int(topk),
        "device": str(device),
        "max_row_id_exclusive": int(max_row_id_exclusive),
        "vectors": {
            "base_rows": int(base_info.rows),
            "query_rows": int(query_info.rows),
            "dim": int(base_info.dim),
            "scalar_type": "float16_hvecs",
        },
        "base_vector_provider": base_vector_provider.summary(),
        "host_warmup": host_warmup,
        "token_index": {
            "mode": "not_loaded"
            if token_index is None
            else "loaded",
            "token_count": None if token_index is None else int(token_index.token_count),
            "ngram_size": None if token_index is None else int(token_index.ngram_size),
            "ngram_count": None if token_index is None else int(len(token_index.ngram_index)),
            "skipped_empty_tokens": None if token_index is None else int(token_index.skipped_empty_tokens),
            "max_postings_per_ngram": None if token_index is None else int(token_index.max_postings_per_ngram),
            "token_id_column": None if token_index is None else token_index.token_id_column,
            "token_text_column": None if token_index is None else token_index.token_text_column,
        },
        "candidate_count_stats": numeric_stats(candidate_counts),
        "expanded_token_count_stats": numeric_stats(expanded_token_counts[expanded_token_counts >= 0]),
        "hit_count_stats": numeric_stats(hit_counts),
        "missing_topk_rows": int(np.count_nonzero(hit_counts < topk)),
        "postings_lookup": token_lookup.summary(),
        "elapsed_seconds": float(elapsed),
        "notes": [
            "string.fuzzy GT expands prefix/suffix/contains against the full token_map vocabulary.",
            "Fuzzy expansion uses a full-token-map ngram index and filters the rarest gram candidate set by exact string mode.",
            "Candidate rows are streamed in token batches and merged with exact top-k deduplication on row_id.",
            "Small pairwise batches stay on CPU while larger batches use GPU top-k; candidate base vectors still stream from base.hvecs in row batches.",
            "Exact-only GT with token-id postings skips token_map loading entirely.",
        ],
    }
    summary_path = output_root / "gt.string.summary.json"
    summary["summary_path"] = str(summary_path)
    ensure_output_path(summary_path, overwrite)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[gt] wrote {topk_info['path']}", flush=True)
    if topk > 10 and "top10_gt" in outputs:
        print(f"[gt] wrote {outputs['top10_gt']['path']}", flush=True)
    print(f"[gt] wrote summary {summary_path}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-manifest",
        type=Path,
        required=True,
        help="Selected query manifest already aligned with query.hvecs row order.",
    )
    parser.add_argument(
        "--query-hvecs",
        type=Path,
        required=True,
        help="Existing query.hvecs aligned with --query-manifest.",
    )
    parser.add_argument("--token-map", type=Path, default=DEFAULT_DATA_ROOT / "token_map.parquet")
    parser.add_argument(
        "--token-ngram-index",
        type=Path,
        default=None,
        help="Optional existing token_ngram_index.bin for fuzzy GT. Reuses the binary ngram sidecar instead of rebuilding an index from token_map.parquet.",
    )
    parser.add_argument("--base-hvecs", type=Path, default=DEFAULT_BASE_HVECS)
    parser.add_argument(
        "--token-postings-dir",
        type=Path,
        default=DEFAULT_TOKEN_POSTINGS,
        help="Token postings source. Can be a token-postings binary, token_*.npy, or parquet postings.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jsonl-output", type=Path, default=None)
    parser.add_argument("--families", type=str, default="string")
    parser.add_argument("--subtypes", type=str, default="exact,fuzzy")
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--per-subtype-limit", type=int, default=0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="CUDA device spec for GT. Use auto for all visible GPUs, or a comma-separated list like cuda:0,cuda:1.",
    )
    parser.add_argument("--row-batch-size", type=int, default=131072)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--token-map-batch-size", type=int, default=262144)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--postings-workers", type=int, default=8)
    parser.add_argument("--postings-batch-rows", type=int, default=4_000_000)
    parser.add_argument("--postings-token-batch-size", type=int, default=8192)
    parser.add_argument("--host-vector-cache-gb", type=float, default=0.0)
    parser.add_argument(
        "--max-row-id-exclusive",
        type=int,
        default=0,
        help="Upper row-id bound for postings/base rows. Defaults to base.hvecs row count.",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_output_path(output_root: Path, explicit: Path | None, default_name: str) -> Path:
    if explicit is None:
        return output_root / default_name
    return explicit if explicit.is_absolute() else output_root / explicit


def write_query_hvecs_subset(
    source_query_hvecs: Path,
    selected_query_hvecs: Path,
    selected_query_indices: Sequence[int],
    overwrite: bool,
) -> None:
    """Select source manifest positions, preserving interleaved subtype order."""
    info = inspect_hvecs(source_query_hvecs)
    indices = np.asarray(selected_query_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= info.rows):
        raise ValueError(f"selected query index outside query.hvecs rows={info.rows}")
    if selected_query_hvecs.resolve() == source_query_hvecs.resolve():
        raise ValueError("selected query output must differ from the source")
    ensure_output_path(selected_query_hvecs, overwrite)
    records = hvecs_records(source_query_hvecs, info)
    with selected_query_hvecs.open("wb") as handle:
        for begin in range(0, indices.size, 65536):
            matrix = np.asarray(records["vec"][indices[begin : begin + 65536]], dtype=np.float16)
            write_hvecs_batch(handle, matrix)


def validate_args(args: argparse.Namespace) -> None:
    if args.query_limit < 0 or args.per_subtype_limit < 0:
        raise ValueError("--query-limit and --per-subtype-limit must be non-negative")
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.row_batch_size <= 0 or args.query_batch_size <= 0:
        raise ValueError("--row-batch-size and --query-batch-size must be positive")
    if args.token_map_batch_size <= 0 or args.ngram_size <= 0:
        raise ValueError("--token-map-batch-size and --ngram-size must be positive")
    if args.postings_workers <= 0 or args.postings_batch_rows <= 0 or args.postings_token_batch_size <= 0:
        raise ValueError("--postings-workers, --postings-batch-rows, and --postings-token-batch-size must be positive")
    if args.host_vector_cache_gb < 0:
        raise ValueError("--host-vector-cache-gb must be non-negative")
    if args.max_row_id_exclusive < 0:
        raise ValueError("--max-row-id-exclusive must be non-negative")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    for required in (args.query_manifest, args.query_hvecs, args.token_map, args.base_hvecs, args.token_postings_dir):
        if not required.exists():
            raise FileNotFoundError(required)

    selected = load_selected_queries(
        query_manifest=args.query_manifest,
        families=csv_lower_set(args.families),
        subtypes=csv_lower_set(args.subtypes),
        query_limit=args.query_limit,
        per_subtype_limit=args.per_subtype_limit,
    )
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_output = resolve_output_path(output_root, args.jsonl_output, "gt.string.rows.jsonl")
    selected_query_hvecs = args.query_hvecs
    source_rows = inspect_hvecs(args.query_hvecs).rows
    if selected.source_indices != list(range(source_rows)):
        selected_query_hvecs = output_root / "query.selected.hvecs"
        write_query_hvecs_subset(
            args.query_hvecs,
            selected_query_hvecs,
            selected_query_indices=selected.source_indices,
            overwrite=args.overwrite,
        )
    summary = compute_string_gpu_gt(
        queries=selected.queries,
        query_vector_path=selected_query_hvecs,
        base_hvecs=args.base_hvecs,
        token_map=args.token_map,
        token_ngram_index_path=args.token_ngram_index,
        token_postings_dir=args.token_postings_dir,
        output_root=output_root,
        jsonl_output=jsonl_output,
        topk=args.topk,
        device_name=args.device,
        row_batch_size=args.row_batch_size,
        query_batch_size=args.query_batch_size,
        token_map_batch_size=args.token_map_batch_size,
        ngram_size=args.ngram_size,
        postings_workers=args.postings_workers,
        postings_batch_rows=args.postings_batch_rows,
        postings_token_batch_size=args.postings_token_batch_size,
        host_vector_cache_gb=args.host_vector_cache_gb,
        max_row_id_exclusive=args.max_row_id_exclusive,
        progress_interval=args.progress_interval,
        overwrite=args.overwrite,
    )
    print(f"Wrote GT summary: {summary['summary_path']}")


if __name__ == "__main__":
    main()
