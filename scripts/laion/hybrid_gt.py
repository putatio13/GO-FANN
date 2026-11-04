#!/usr/bin/env python3
"""Compute exact GT for pairwise LAION hybrid query manifests."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.laion.predicates import range_predicate_mask


from scripts.laion.categorical_gt import hvecs_records, inspect_hvecs, read_fbin_header
from scripts.laion.string_index import (
    TokenIdPostingsBinaryLookup,
    fuzzy_matching_token_ids,
    fuzzy_predicate_mode,
    fuzzy_predicate_pattern,
    load_token_text_index_from_ngram_binary,
)
from scripts.laion.string_gt import (
    BaseVectorProvider,
    dedupe_topk_rows_cpu,
    gpu_stream_topk_for_candidate_rows,
    resolve_cuda_device_ids,
    stable_sort_topk_rows,
    write_dist_fbin,
    write_gt_ibin,
)


DEFAULT_DATA_ROOT = Path("data/laion100m")
DEFAULT_BASE_HVECS = DEFAULT_DATA_ROOT / "base.hvecs"
DEFAULT_TOKEN_POSTINGS = Path("artifacts/laion100m/token_postings.bin")
DEFAULT_NGRAM_INDEX = Path("artifacts/laion100m/token_ngram_index.bin")
DEFAULT_RANGE_FBIN = DEFAULT_DATA_ROOT / "range/base_range.fbin"
UINT32_MAX = np.uint32(0xFFFFFFFF)


@dataclass
class HybridQuery:
    local_idx: int
    hvec_idx: int
    query_id: str
    subtype: str
    logic: str
    record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--query-hvecs", type=Path, required=True)
    parser.add_argument("--base-hvecs", type=Path, default=DEFAULT_BASE_HVECS)
    parser.add_argument("--token-postings-bin", type=Path, default=DEFAULT_TOKEN_POSTINGS)
    parser.add_argument("--token-ngram-index", type=Path, default=DEFAULT_NGRAM_INDEX)
    parser.add_argument("--range-fbin", type=Path, default=DEFAULT_RANGE_FBIN)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--device",
        default="auto",
        help="CUDA device selection. auto prefers idle devices >=2 via materialize helper.",
    )
    parser.add_argument("--row-batch-size", type=int, default=131072)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--per-subtype-limit", type=int, default=0)
    parser.add_argument("--load-range-values", action="store_true")
    parser.add_argument("--rowset-cache-gb", type=float, default=16.0)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")


class RowSetProvider:
    def __init__(
        self,
        *,
        token_postings_bin: Path,
        token_ngram_index: Path,
        range_fbin: Path,
        row_scope: int,
        load_range_values: bool,
        rowset_cache_bytes: int,
    ) -> None:
        self.row_scope = int(row_scope)
        self.token_lookup = TokenIdPostingsBinaryLookup(token_postings_bin, max_row_id_exclusive=row_scope)
        self.token_index = load_token_text_index_from_ngram_binary(token_ngram_index)
        rows, cols = read_fbin_header(range_fbin)
        if row_scope > rows:
            raise ValueError(f"row_scope {row_scope} exceeds range rows {rows}")
        values = np.memmap(range_fbin, dtype="<f4", mode="r", offset=8, shape=(rows, cols))
        self.range_values = np.asarray(values[:row_scope], dtype=np.float32) if load_range_values else values
        self.row_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.row_cache_bytes_limit = max(0, int(rowset_cache_bytes))
        self.row_cache_bytes = 0
        self.row_cache_evictions = 0
        self.fuzzy_expansion_cache: dict[str, np.ndarray] = {}

    def child_key(self, child: dict[str, Any]) -> str:
        family = str(child.get("family", "")).lower()
        subtype = str(child.get("subtype", "")).lower()
        key = str(child.get("predicate_id") or child.get("predicate_key") or child.get("query_id") or json.dumps(child, sort_keys=True))
        return f"{family}.{subtype}:{key}"

    def rows_for_child(self, child: dict[str, Any]) -> np.ndarray:
        family = str(child.get("family", "")).lower()
        subtype = str(child.get("subtype", "")).lower()
        cache_key = self.child_key(child)
        cached = self.row_cache.get(cache_key)
        if cached is not None:
            self.row_cache.move_to_end(cache_key)
            return cached
        if family in {"categorical", "string"} and subtype in {"single", "exact"}:
            rows = self.rows_for_token_ids(predicate_token_ids(child))
        elif family == "string" and subtype == "fuzzy":
            rows = self.rows_for_fuzzy(child)
        elif family == "range":
            rows = np.flatnonzero(self.range_mask_for_child(child)).astype(np.uint32, copy=False)
        else:
            raise ValueError(f"unsupported hybrid child {family}.{subtype}")
        rows = filter_rows(rows, self.row_scope)
        self.cache_rows(cache_key, rows)
        return rows

    def cache_rows(self, key: str, rows: np.ndarray) -> None:
        if self.row_cache_bytes_limit <= 0:
            return
        size = int(rows.nbytes)
        if size > self.row_cache_bytes_limit:
            return
        while self.row_cache and self.row_cache_bytes + size > self.row_cache_bytes_limit:
            _, old_rows = self.row_cache.popitem(last=False)
            self.row_cache_bytes -= int(old_rows.nbytes)
            self.row_cache_evictions += 1
        self.row_cache[key] = rows
        self.row_cache_bytes += size

    def rows_for_token_ids(self, token_ids: list[int]) -> np.ndarray:
        clean = sorted({int(token_id) for token_id in token_ids if int(token_id) >= 0})
        if not clean:
            return np.empty(0, dtype=np.uint32)
        if len(clean) == 1:
            return filter_rows(self.token_lookup.load_token_ids(clean).get(clean[0], np.empty(0, dtype=np.uint32)), self.row_scope)
        return filter_rows(self.token_lookup.rows_for_token_ids_union(np.asarray(clean, dtype=np.uint32)), self.row_scope)

    def rows_for_fuzzy(self, child: dict[str, Any]) -> np.ndarray:
        predicates = child.get("predicates") or []
        if not predicates:
            return np.empty(0, dtype=np.uint32)
        mode = fuzzy_predicate_mode(predicates[0])
        pattern = fuzzy_predicate_pattern(predicates[0])
        expansion_key = f"{mode}:{pattern}"
        token_ids = self.fuzzy_expansion_cache.get(expansion_key)
        if token_ids is None:
            token_ids = fuzzy_matching_token_ids(mode, pattern, self.token_index)
            self.fuzzy_expansion_cache[expansion_key] = token_ids
        return self.rows_for_token_ids(token_ids.astype(np.uint32, copy=False).tolist())

    def range_mask_for_child(self, child: dict[str, Any]) -> np.ndarray:
        values = self.range_values[: self.row_scope]
        return range_predicate_mask(values, child.get("predicates") or [])


def predicate_token_ids(child: dict[str, Any]) -> list[int]:
    out = []
    for pred in child.get("predicates") or []:
        token_id = pred.get("token_id", pred.get("source_token_id"))
        if token_id is not None:
            out.append(int(token_id))
    return out


def filter_rows(rows: np.ndarray, row_scope: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.uint32)
    if rows.size == 0:
        return rows
    rows = rows[rows < np.uint32(row_scope)]
    if rows.size > 1:
        rows = np.unique(rows)
    return rows.astype(np.uint32, copy=False)


def hybrid_tree(record: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    tree = record.get("predicate") or {}
    op = str(tree.get("op") or record.get("hybrid_top_logic") or "").lower()
    children = tree.get("children") or []
    if len(children) != 2:
        raise ValueError(f"expected pairwise hybrid predicate children, got {len(children)}")
    return op, children


def manifest_candidate_count(record: dict[str, Any]) -> int | None:
    value = record.get("candidate_count")
    if value is None:
        stats = record.get("stats") or {}
        value = stats.get("true_rows") or stats.get("estimated_rows")
    if value is None:
        return None
    return int(value)


def candidate_row_parts(record: dict[str, Any], provider: RowSetProvider) -> tuple[list[np.ndarray], int]:
    op, children = hybrid_tree(record)
    left = provider.rows_for_child(children[0])
    right = provider.rows_for_child(children[1])
    if op == "and":
        rows = np.intersect1d(left, right, assume_unique=True).astype(np.uint32, copy=False)
        return [rows], int(rows.size)
    if op == "or":
        if left.size >= right.size:
            other_only = np.setdiff1d(right, left, assume_unique=True).astype(np.uint32, copy=False)
            parts = [left, other_only]
        else:
            other_only = np.setdiff1d(left, right, assume_unique=True).astype(np.uint32, copy=False)
            parts = [right, other_only]
        return parts, int(parts[0].size + parts[1].size)
    raise ValueError(f"unsupported hybrid op {op!r}")


def topk_for_candidate_parts(
    *,
    base_provider: BaseVectorProvider,
    query_gpu: Any,
    query_norm: Any,
    candidate_parts: list[np.ndarray],
    topk: int,
    device: Any,
    row_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    out_ids = np.full((int(query_gpu.shape[0]), topk), UINT32_MAX, dtype=np.uint32)
    out_dist = np.full((int(query_gpu.shape[0]), topk), np.inf, dtype=np.float32)
    best_ids_gpu = None
    best_dist_gpu = None
    for rows in candidate_parts:
        rows = np.asarray(rows, dtype=np.uint32)
        if rows.size == 0:
            continue
        best_ids_gpu, best_dist_gpu = gpu_stream_topk_for_candidate_rows(
            torch,
            base_provider,
            query_gpu,
            query_norm,
            rows,
            topk,
            device,
            row_batch_size,
            best_ids_gpu,
            best_dist_gpu,
        )
    if best_ids_gpu is None or best_dist_gpu is None:
        return out_ids, out_dist
    ids_cpu = best_ids_gpu.cpu().numpy().astype(np.uint32, copy=False)
    dist_cpu = best_dist_gpu.cpu().numpy().astype(np.float32, copy=False)
    for row_idx in range(ids_cpu.shape[0]):
        out_ids[row_idx], out_dist[row_idx] = dedupe_topk_rows_cpu(ids_cpu[row_idx], dist_cpu[row_idx], topk)
    return out_ids, out_dist


def query_gpu_batch(
    query_records: np.memmap,
    queries: list[HybridQuery],
    query_indices: list[int],
    device: Any,
) -> tuple[Any, Any]:
    import torch
    hvec_indices = np.asarray([queries[index].hvec_idx for index in query_indices], dtype=np.int64)
    query_np = np.asarray(query_records["vec"][hvec_indices], dtype=np.float16)
    query_gpu = torch.from_numpy(np.asarray(query_np, dtype=np.float32)).to(device=device)
    query_norm = torch.sum(query_gpu * query_gpu, dim=1)
    return query_gpu, query_norm


def merge_query_result(
    final_ids: np.ndarray,
    final_dist: np.ndarray,
    query_idx: int,
    ids: np.ndarray,
    dist: np.ndarray,
    source_idx: int,
    topk: int,
) -> None:
    merged_ids, merged_dist = dedupe_topk_rows_cpu(
        np.concatenate((final_ids[query_idx], ids[source_idx])),
        np.concatenate((final_dist[query_idx], dist[source_idx])),
        topk,
    )
    final_ids[query_idx] = merged_ids
    final_dist[query_idx] = merged_dist


def load_queries(path: Path, query_limit: int, per_subtype_limit: int) -> list[HybridQuery]:
    out: list[HybridQuery] = []
    per_subtype: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            subtype = str(record.get("subtype"))
            if per_subtype_limit > 0 and per_subtype[subtype] >= per_subtype_limit:
                continue
            local_idx = len(out)
            hvec_idx = int(record.get("flashanns_query_index", local_idx))
            out.append(
                HybridQuery(
                    local_idx=local_idx,
                    hvec_idx=hvec_idx,
                    query_id=str(record.get("query_id", local_idx)),
                    subtype=subtype,
                    logic=str(record.get("hybrid_top_logic", "")).lower(),
                    record=record,
                )
            )
            per_subtype[subtype] += 1
            if query_limit > 0 and len(out) >= query_limit:
                break
    if not out:
        raise RuntimeError(f"no queries selected from {path}")
    return out


def write_jsonl(
    path: Path,
    queries: list[HybridQuery],
    ids: np.ndarray,
    dist: np.ndarray,
    candidate_counts: np.ndarray,
    overwrite: bool,
) -> dict[str, Any]:
    ensure_output_path(path, overwrite)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            idx = query.local_idx
            valid = ids[idx] != UINT32_MAX
            row = {
                "flashanns_query_index": idx,
                "query_id": query.query_id,
                "subtype": query.subtype,
                "logic": query.logic,
                "candidate_count": int(candidate_counts[idx]),
                "topk": ids[idx][valid].astype(np.uint32).tolist(),
                "distances": dist[idx][valid].astype(np.float32).tolist(),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {"path": str(path), "rows": len(queries), "bytes": path.stat().st_size}


def main() -> None:
    args = parse_args()
    import torch
    started = time.time()
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    for required in (args.query_manifest, args.query_hvecs, args.base_hvecs, args.token_postings_bin, args.token_ngram_index, args.range_fbin):
        if not required.exists():
            raise FileNotFoundError(required)

    queries = load_queries(args.query_manifest, args.query_limit, args.per_subtype_limit)
    query_info = inspect_hvecs(args.query_hvecs)
    max_hvec_idx = max(query.hvec_idx for query in queries)
    if max_hvec_idx >= query_info.rows:
        raise ValueError(f"query.hvecs rows {query_info.rows} <= max selected vector index {max_hvec_idx}")
    base_info = inspect_hvecs(args.base_hvecs)
    if query_info.dim != base_info.dim:
        raise ValueError(f"query dim {query_info.dim} != base dim {base_info.dim}")

    device_ids = resolve_cuda_device_ids(args.device)
    device = torch.device(f"cuda:{device_ids[0]}")
    print(f"[hybrid-gt] device={device} queries={len(queries)} topk={args.topk}", flush=True)

    query_records = hvecs_records(args.query_hvecs, query_info)
    base_records = hvecs_records(args.base_hvecs, base_info)
    base_provider = BaseVectorProvider(base_records)
    provider = RowSetProvider(
        token_postings_bin=args.token_postings_bin,
        token_ngram_index=args.token_ngram_index,
        range_fbin=args.range_fbin,
        row_scope=base_info.rows,
        load_range_values=args.load_range_values,
        rowset_cache_bytes=int(max(0.0, args.rowset_cache_gb) * (1024 ** 3)),
    )

    final_ids = np.full((len(queries), args.topk), UINT32_MAX, dtype=np.uint32)
    final_dist = np.full((len(queries), args.topk), np.inf, dtype=np.float32)
    candidate_counts = np.zeros((len(queries),), dtype=np.int64)
    subtype_counts: Counter[str] = Counter()
    or_child_groups: dict[str, dict[str, Any]] = {}
    or_group_links: list[tuple[int, str]] = []

    for idx, query in enumerate(queries):
        op, children = hybrid_tree(query.record)
        subtype_counts[query.subtype] += 1
        manifest_count = manifest_candidate_count(query.record)
        if manifest_count is not None:
            candidate_counts[idx] = manifest_count
        if op == "or":
            for child in children:
                key = provider.child_key(child)
                if key not in or_child_groups:
                    or_child_groups[key] = {"child": child, "query_indices": []}
                or_child_groups[key]["query_indices"].append(idx)
                or_group_links.append((idx, key))
            continue
        candidate_parts, candidate_count = candidate_row_parts(query.record, provider)
        candidate_counts[idx] = candidate_count
        query_gpu, query_norm = query_gpu_batch(query_records, queries, [idx], device)
        ids, dist = topk_for_candidate_parts(
            base_provider=base_provider,
            query_gpu=query_gpu,
            query_norm=query_norm,
            candidate_parts=candidate_parts,
            topk=args.topk,
            device=device,
            row_batch_size=args.row_batch_size,
        )
        final_ids[idx : idx + 1] = ids
        final_dist[idx : idx + 1] = dist
        if args.progress_interval > 0 and (idx + 1) % int(args.progress_interval) == 0:
            print(
                f"[hybrid-gt] {idx + 1}/{len(queries)} subtype={query.subtype} candidates={candidate_count}",
                flush=True,
            )

    if or_child_groups:
        processed_or_chunks = 0
        total_or_chunks = sum(
            (len(group["query_indices"]) + max(1, args.query_batch_size) - 1) // max(1, args.query_batch_size)
            for group in or_child_groups.values()
        )
        print(
            f"[hybrid-gt] processing OR child groups={len(or_child_groups)} chunks={total_or_chunks}",
            flush=True,
        )
        batch_size = max(1, int(args.query_batch_size))
        for key, group in sorted(or_child_groups.items()):
            rows = provider.rows_for_child(group["child"])
            qindices = list(group["query_indices"])
            for begin in range(0, len(qindices), batch_size):
                chunk_indices = qindices[begin : begin + batch_size]
                query_gpu, query_norm = query_gpu_batch(query_records, queries, chunk_indices, device)
                ids, dist = topk_for_candidate_parts(
                    base_provider=base_provider,
                    query_gpu=query_gpu,
                    query_norm=query_norm,
                    candidate_parts=[rows],
                    topk=args.topk,
                    device=device,
                    row_batch_size=args.row_batch_size,
                )
                for local_pos, query_idx in enumerate(chunk_indices):
                    merge_query_result(final_ids, final_dist, query_idx, ids, dist, local_pos, args.topk)
                processed_or_chunks += 1
                if args.progress_interval > 0 and processed_or_chunks % int(args.progress_interval) == 0:
                    print(
                        f"[hybrid-gt] OR chunks {processed_or_chunks}/{total_or_chunks} child_rows={int(rows.size)}",
                        flush=True,
                    )
        for query_idx, _ in or_group_links:
            if candidate_counts[query_idx] == 0:
                candidate_parts, candidate_count = candidate_row_parts(queries[query_idx].record, provider)
                candidate_counts[query_idx] = candidate_count

    final_ids, final_dist = stable_sort_topk_rows(final_ids, final_dist)
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"top{args.topk}_gt": write_gt_ibin(args.output_root / f"gt.hybrid.top{args.topk}.ibin", final_ids, args.overwrite),
        f"top{args.topk}_dist": write_dist_fbin(args.output_root / f"gt.hybrid.top{args.topk}.dist.fbin", final_dist, args.overwrite),
        "jsonl": write_jsonl(args.output_root / "gt.hybrid.rows.jsonl", queries, final_ids, final_dist, candidate_counts, args.overwrite),
    }
    if args.topk > 10:
        outputs["top10_gt"] = write_gt_ibin(args.output_root / "gt.hybrid.top10.ibin", final_ids[:, :10], args.overwrite)
        outputs["top10_dist"] = write_dist_fbin(args.output_root / "gt.hybrid.top10.dist.fbin", final_dist[:, :10], args.overwrite)

    summary = {
        "query_manifest": str(args.query_manifest),
        "query_hvecs": str(args.query_hvecs),
        "base_hvecs": str(args.base_hvecs),
        "query_count": len(queries),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "candidate_count_min": int(candidate_counts.min()) if candidate_counts.size else None,
        "candidate_count_median": float(np.median(candidate_counts)) if candidate_counts.size else None,
        "candidate_count_max": int(candidate_counts.max()) if candidate_counts.size else None,
        "candidate_count_sum": int(candidate_counts.sum()),
        "topk": int(args.topk),
        "device": str(device),
        "outputs": outputs,
        "elapsed_sec": time.time() - started,
        "rowset_cache": {
            "row_cache_entries": len(provider.row_cache),
            "row_cache_bytes": int(provider.row_cache_bytes),
            "row_cache_gb_limit": float(args.rowset_cache_gb),
            "row_cache_evictions": int(provider.row_cache_evictions),
            "fuzzy_expansion_cache_entries": len(provider.fuzzy_expansion_cache),
            "token_lookup": provider.token_lookup.summary(),
        },
    }
    (args.output_root / "gt.hybrid.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[hybrid-gt] wrote {args.output_root / f'gt.hybrid.top{args.topk}.ibin'}", flush=True)


if __name__ == "__main__":
    main()
