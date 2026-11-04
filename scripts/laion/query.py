#!/usr/bin/env python3
"""Materialize LAION query vectors, ranges, and a global token ngram sidecar.

This script is intentionally narrower than scripts.laion.base:
it writes query vectors in exactly the selected query-manifest order, builds a
fuzzy-token ngram index from the global runtime token_map.parquet, and can
optionally compute exact GPU GT for selected string exact/fuzzy queries.
Query vectors are reconstructed from an existing full `base.hvecs` when
available and fall back to LAION image-embedding shards otherwise. Query range
rows are emitted for both atomic range and pairwise hybrid manifests; query
labels remain manifest-driven.

Ngram binary layout, little-endian:

- header:
  [magic:16, version:u32, header_bytes:u32, endian:u32, flags:u32,
   label_count:u64, token_count:u64, ngram_size:u64,
   normalizer_version:u64, token_text_bytes:u64, ngram_count:u64,
   ngram_text_bytes:u64, posting_count:u64, checksum_kind:u64,
   reserved:u64, checksum_sha256:32]
- token_ids: u32[token_count]
- token_ngram_counts: u32[token_count]
- token_text_offsets: u64[token_count + 1]
- token_text_payload: normalized ASCII token bytes
- ngram_text_offsets: u64[ngram_count + 1]
- ngram_offsets: u64[ngram_count + 1]
- ngram_text_payload: normalized ngram bytes
- posting_ids: u32[posting_count]

The checksum is SHA-256 over all sections after the header.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.laion.base import (
    MODE_NO_RANGE,
    RowMapLookup,
    RuntimeQuery,
    ensure_output_path,
    hvecs_records,
    inspect_hvecs,
    range_terms,
    read_fbin_header,
    write_hvecs_batch,
    write_query_range_bin,
)


from scripts.laion.string_index import chunked, write_ngram_index

DEFAULT_LAION_ROOT = Path("data/laion400m")
DEFAULT_DATA_ROOT = Path("data/laion100m")
DEFAULT_TOKEN_POSTINGS = Path("artifacts/laion100m/token_postings.bin")


@dataclass
class SelectedQueries:
    queries: list[RuntimeQuery]
    source_indices: list[int]
    summary: dict[str, Any]


@dataclass
class EmbeddingShardLoader:
    root: Path
    cache: dict[int, np.memmap]
    dim: int | None = None
    source_dtype: str | None = None

    def shard_path(self, shard_id: int) -> Path:
        candidates = [
            self.root / f"img_emb_{shard_id}.npy",
            self.root / "images" / f"img_emb_{shard_id}.npy",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"missing LAION embedding shard img_emb_{shard_id}.npy under {self.root}")

    def load(self, shard_id: int) -> np.memmap:
        shard = self.cache.get(shard_id)
        if shard is not None:
            return shard
        path = self.shard_path(shard_id)
        shard = np.load(path, mmap_mode="r")
        if shard.ndim != 2:
            raise ValueError(f"expected 2D embedding shard, got shape={shard.shape} for {path}")
        if self.dim is None:
            self.dim = int(shard.shape[1])
            self.source_dtype = str(shard.dtype)
        elif int(shard.shape[1]) != self.dim:
            raise ValueError(f"embedding shard {path} dim {shard.shape[1]} != expected dim {self.dim}")
        self.cache[shard_id] = shard
        return shard

    def infer_dim(self) -> int:
        if self.dim is not None:
            return self.dim
        candidates = sorted(self.root.glob("img_emb_*.npy"))
        if not candidates:
            candidates = sorted((self.root / "images").glob("img_emb_*.npy"))
        for path in candidates:
            shard = np.load(path, mmap_mode="r")
            if shard.ndim == 2 and shard.shape[1] > 0:
                self.dim = int(shard.shape[1])
                self.source_dtype = str(shard.dtype)
                return self.dim
        raise FileNotFoundError(f"no img_emb_*.npy shards found under {self.root} or {self.root / 'images'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--token-map", type=Path, default=DEFAULT_DATA_ROOT / "token_map.parquet")
    parser.add_argument(
        "--base-range-fbin",
        type=Path,
        default=DEFAULT_DATA_ROOT / "range/base_range.fbin",
        help="Base range matrix used to validate the emitted query range dimension.",
    )
    parser.add_argument("--laion-root", type=Path, default=DEFAULT_LAION_ROOT)
    parser.add_argument(
        "--embedding-root",
        "--npy-root",
        dest="embedding_root",
        type=Path,
        default=None,
        help="Directory containing img_emb_*.npy shards. Defaults to <laion-root>/images.",
    )
    parser.add_argument("--row-map", type=Path, default=DEFAULT_DATA_ROOT / "row_map.parquet")
    parser.add_argument(
        "--base-hvecs",
        type=Path,
        default=DEFAULT_DATA_ROOT / "base.hvecs",
        help="Optional existing full-base .hvecs used to reconstruct query vectors by query_vector_id.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--query-vector-output", type=Path, default=None)
    parser.add_argument("--query-range-output", type=Path, default=None)
    parser.add_argument("--ngram-index-output", type=Path, default=None)
    parser.add_argument("--selected-manifest-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--families", type=str, default="")
    parser.add_argument("--subtypes", type=str, default="")
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--per-subtype-limit", type=int, default=0)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--embedding-batch-size", type=int, default=8192)
    parser.add_argument("--token-map-batch-size", type=int, default=262144)
    parser.add_argument("--row-map-workers", type=int, default=8)
    parser.add_argument("--row-map-batch-size", type=int, default=100000)
    parser.add_argument("--compute-gt", action="store_true", help="Compute exact GPU GT for selected string exact/fuzzy queries.")
    parser.add_argument(
        "--token-postings-dir",
        type=Path,
        default=DEFAULT_TOKEN_POSTINGS,
        help="Token postings binary or NumPy/Parquet directory used for exact/fuzzy GT candidate rows.",
    )
    parser.add_argument(
        "--gt-output-root",
        type=Path,
        default=None,
        help="Optional GT output directory. Defaults to --output-root.",
    )
    parser.add_argument("--gt-topk", type=int, default=100)
    parser.add_argument(
        "--gt-device",
        type=str,
        default="auto",
        help="CUDA device spec for GT. Use auto for all visible GPUs, or a comma-separated list like cuda:0,cuda:1.",
    )
    parser.add_argument("--gt-row-batch-size", type=int, default=131072)
    parser.add_argument("--gt-query-batch-size", type=int, default=32)
    parser.add_argument("--gt-postings-workers", type=int, default=8)
    parser.add_argument("--gt-postings-batch-rows", type=int, default=1000000)
    parser.add_argument("--gt-postings-token-batch-size", type=int, default=8192)
    parser.add_argument("--gt-host-vector-cache-gb", type=float, default=0.0)
    parser.add_argument(
        "--gt-token-ngram-index",
        type=Path,
        default=None,
        help="Optional existing token_ngram_index.bin for fuzzy GT. When set, fuzzy GT reuses it instead of rebuilding an index from token_map.parquet.",
    )
    parser.add_argument(
        "--gt-max-row-id-exclusive",
        type=int,
        default=0,
        help="Upper row-id bound for postings. Defaults to base.hvecs row count.",
    )
    parser.add_argument(
        "--gt-jsonl-output",
        type=Path,
        default=None,
        help="Optional JSONL detail output for GT rows. Defaults to gt.string.rows.jsonl under --gt-output-root.",
    )
    parser.add_argument("--gt-progress-interval", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def csv_lower_set(raw: str) -> set[str]:
    return {item.strip().lower() for item in str(raw or "").split(",") if item.strip()}


def resolve_output_path(output_root: Path, explicit: Path | None, default_name: str) -> Path:
    if explicit is None:
        return output_root / default_name
    return explicit if explicit.is_absolute() else output_root / explicit


def manifest_subtype_key(record: dict[str, Any]) -> tuple[str, str, str]:
    family = str(record.get("family", "")).lower()
    subtype = str(record.get("subtype", "")).lower()
    return family, subtype, f"{family}.{subtype}"


def manifest_record_allowed(
    family: str,
    subtype: str,
    families: set[str],
    subtypes: set[str],
) -> bool:
    if families and family not in families:
        return False
    return not subtypes or subtype in subtypes or f"{family}.{subtype}" in subtypes


def selected_query_range_terms(
    record: dict[str, Any],
    range_count: int,
) -> list[tuple[int, int, float, float]]:
    family = str(record.get("family", "")).lower()
    range_records: list[dict[str, Any]] = []
    if family == "range":
        range_records.append(record)
    elif family == "hybrid":
        predicate = record.get("predicate")
        if not isinstance(predicate, dict):
            raise ValueError("hybrid query has no predicate object")
        children = predicate.get("children")
        if not isinstance(children, list) or len(children) != 2:
            raise ValueError("hybrid query must contain exactly two predicate children")
        range_records.extend(
            child
            for child in children
            if isinstance(child, dict) and str(child.get("family", "")).lower() == "range"
        )
        if len(range_records) > 1:
            raise ValueError("hybrid query contains more than one range child")

    terms: list[tuple[int, int, float, float]] = []
    seen_columns: set[int] = set()
    for range_record in range_records:
        for column, mode, lower, upper in range_terms(range_record):
            if column < 0 or column >= range_count:
                raise ValueError(
                    f"range column {column} is outside base range count {range_count}"
                )
            if column in seen_columns:
                raise ValueError(f"duplicate range column {column}")
            if mode == int(MODE_NO_RANGE):
                raise ValueError("manifest range term unexpectedly uses NO_RANGE mode")
            seen_columns.add(column)
            terms.append((column, mode, lower, upper))
    return terms


def load_selected_queries(
    query_manifest: Path,
    families: set[str],
    subtypes: set[str],
    query_limit: int,
    per_subtype_limit: int,
    range_count: int = 0,
) -> SelectedQueries:
    queries: list[RuntimeQuery] = []
    source_indices: list[int] = []
    scanned_by_subtype: Counter[str] = Counter()
    selected_by_subtype: Counter[str] = Counter()
    skipped_by_filter: Counter[str] = Counter()
    skipped_by_limit: Counter[str] = Counter()
    errors: list[str] = []
    scanned_total = 0

    with query_manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            scanned_total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"query manifest JSON parse error on line {line_number}: {exc}") from exc

            family, subtype, subtype_name = manifest_subtype_key(record)
            scanned_by_subtype[subtype_name] += 1
            if not manifest_record_allowed(family, subtype, families, subtypes):
                skipped_by_filter[subtype_name] += 1
                continue
            if per_subtype_limit > 0 and selected_by_subtype[subtype_name] >= per_subtype_limit:
                skipped_by_limit[subtype_name] += 1
                continue

            row_id = record.get("query_vector_id")
            if row_id is None:
                errors.append(f"line {line_number} {subtype_name}: missing query_vector_id")
                continue
            try:
                row_id = int(row_id)
            except (TypeError, ValueError):
                errors.append(f"line {line_number} {subtype_name}: invalid query_vector_id={row_id!r}")
                continue
            if row_id < 0:
                errors.append(f"line {line_number} {subtype_name}: negative query_vector_id={row_id}")
                continue

            selected = dict(record)
            selected["query_vector_id"] = row_id
            selected["flashanns_query_index"] = len(queries)
            selected["flashanns_query_order"] = "query_manifest_order_after_family_subtype_filters"
            try:
                ranges = selected_query_range_terms(selected, range_count)
            except ValueError as exc:
                errors.append(f"line {line_number} {subtype_name}: invalid range predicate: {exc}")
                continue
            queries.append(RuntimeQuery(source=selected, label_ids=[], ranges=ranges))
            source_indices.append(scanned_total - 1)
            selected_by_subtype[subtype_name] += 1

            if query_limit > 0 and len(queries) >= query_limit:
                break

    if errors:
        preview = "\n".join(errors[:16])
        raise ValueError(
            "Selected query manifest rows must all have valid query_vector_id values; "
            f"first errors:\n{preview}"
        )
    if not queries:
        raise RuntimeError(f"No query rows selected from {query_manifest}; relax filters or limits.")

    row_ids = [int(query.source["query_vector_id"]) for query in queries]
    row_id_hasher = hashlib.sha256()
    row_id_hasher.update(np.asarray(row_ids, dtype=np.dtype("<u8")).tobytes())
    summary = {
        "query_manifest": str(query_manifest),
        "selection_order": "manifest order after family/subtype filters; limits are applied in that order",
        "families_filter": sorted(families),
        "subtypes_filter": sorted(subtypes),
        "query_limit": int(query_limit),
        "per_subtype_limit": int(per_subtype_limit),
        "scanned_total": scanned_total,
        "selected_total": len(queries),
        "scanned_by_subtype": dict(scanned_by_subtype),
        "selected_by_subtype": dict(selected_by_subtype),
        "skipped_by_filter": dict(skipped_by_filter),
        "skipped_by_limit": dict(skipped_by_limit),
        "query_vector_id_checksum_sha256": row_id_hasher.hexdigest(),
        "first_query_vector_ids": row_ids[:16],
        "last_query_vector_ids": row_ids[-16:],
    }
    return SelectedQueries(queries=queries, source_indices=source_indices, summary=summary)


def write_selected_manifest(path: Path, queries: list[RuntimeQuery], overwrite: bool) -> dict[str, Any]:
    ensure_output_path(path, overwrite)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query.source, sort_keys=True) + "\n")
    return {"path": str(path), "rows": len(queries), "bytes": path.stat().st_size}


def write_query_hvecs(
    path: Path,
    queries: list[RuntimeQuery],
    row_lookup: RowMapLookup | None,
    embedding_root: Path,
    base_hvecs: Path | None,
    batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    ensure_output_path(path, overwrite)
    row_ids = [int(query.source["query_vector_id"]) for query in queries]
    if base_hvecs is not None and base_hvecs.exists():
        info = inspect_hvecs(base_hvecs)
        if any(row_id < 0 or row_id >= info.rows for row_id in row_ids):
            raise ValueError(f"query_vector_id exceeds base.hvecs row count {info.rows} in {base_hvecs}")
        records = hvecs_records(base_hvecs, info)
        rows_written = 0
        with path.open("wb") as handle:
            for batch_ids in chunked(row_ids, batch_size):
                batch_rows = np.asarray(batch_ids, dtype=np.int64)
                matrix = np.asarray(records["vec"][batch_rows], dtype=np.float16)
                write_hvecs_batch(handle, matrix)
                rows_written += matrix.shape[0]
        return {
            "path": str(path),
            "rows": rows_written,
            "dim": info.dim,
            "source_dtype": "float16",
            "payload_dtype": "float16",
            "format": "hvecs",
            "bytes": path.stat().st_size,
            "expected_bytes": rows_written * (4 + 2 * info.dim),
            "source": "base_hvecs",
            "source_path": str(base_hvecs),
        }

    if row_lookup is None:
        raise ValueError("row_lookup is required when query vectors are sourced from embedding shards")
    entries = row_lookup.get_entries(row_ids)
    loader = EmbeddingShardLoader(root=embedding_root, cache={})
    dim = loader.infer_dim()
    rows_written = 0

    with path.open("wb") as handle:
        for batch_ids in chunked(row_ids, batch_size):
            matrix = np.empty((len(batch_ids), dim), dtype=np.float16)
            by_shard: dict[int, list[tuple[int, int]]] = {}
            for out_idx, row_id in enumerate(batch_ids):
                entry = entries[int(row_id)]
                by_shard.setdefault(entry.shard_id, []).append((out_idx, entry.shard_row_id))
            for shard_id, locations in by_shard.items():
                shard = loader.load(shard_id)
                shard_rows = np.asarray([shard_row for _out_idx, shard_row in locations], dtype=np.int64)
                if np.any(shard_rows < 0) or np.any(shard_rows >= shard.shape[0]):
                    raise IndexError(f"row_map references row outside embedding shard {shard_id}")
                vectors = np.asarray(shard[shard_rows], dtype=np.float16)
                for local_idx, (out_idx, _shard_row) in enumerate(locations):
                    matrix[out_idx] = vectors[local_idx]
            write_hvecs_batch(handle, matrix)
            rows_written += matrix.shape[0]

    return {
        "path": str(path),
        "rows": rows_written,
        "dim": dim,
        "source_dtype": loader.source_dtype,
        "payload_dtype": "float16",
        "format": "hvecs",
        "bytes": path.stat().st_size,
        "expected_bytes": rows_written * (4 + 2 * dim),
        "source": "laion_images",
        "source_path": str(embedding_root),
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.query_limit < 0 or args.per_subtype_limit < 0:
        raise ValueError("--query-limit and --per-subtype-limit must be non-negative")
    if args.ngram_size <= 0:
        raise ValueError("--ngram-size must be positive")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive")
    if args.token_map_batch_size <= 0:
        raise ValueError("--token-map-batch-size must be positive")
    if args.row_map_workers <= 0 or args.row_map_batch_size <= 0:
        raise ValueError("--row-map-workers and --row-map-batch-size must be positive")
    if args.compute_gt:
        if args.gt_topk <= 0:
            raise ValueError("--gt-topk must be positive")
        if args.gt_row_batch_size <= 0 or args.gt_query_batch_size <= 0:
            raise ValueError("--gt-row-batch-size and --gt-query-batch-size must be positive")
        if args.gt_postings_workers <= 0 or args.gt_postings_batch_rows <= 0 or args.gt_postings_token_batch_size <= 0:
            raise ValueError("--gt-postings-workers, --gt-postings-batch-rows, and --gt-postings-token-batch-size must be positive")
        if args.gt_host_vector_cache_gb < 0:
            raise ValueError("--gt-host-vector-cache-gb must be non-negative")
        if args.gt_max_row_id_exclusive < 0:
            raise ValueError("--gt-max-row-id-exclusive must be non-negative")
        if args.gt_progress_interval < 0:
            raise ValueError("--gt-progress-interval must be non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    embedding_root = args.embedding_root or (args.laion_root / "images")
    query_vector_output = resolve_output_path(output_root, args.query_vector_output, "query.hvecs")
    query_range_output = resolve_output_path(output_root, args.query_range_output, "query_range.bin")
    ngram_index_output = resolve_output_path(output_root, args.ngram_index_output, "token_ngram_index.bin")
    selected_manifest_output = resolve_output_path(
        output_root,
        args.selected_manifest_output,
        "query_manifest.jsonl",
    )
    summary_output = resolve_output_path(output_root, args.summary_output, "query_runtime_summary.json")

    for required in (args.query_manifest, args.token_map, args.base_range_fbin):
        if not required.exists():
            raise FileNotFoundError(required)
    base_range_rows, range_count = read_fbin_header(args.base_range_fbin)
    if base_range_rows <= 0 or range_count <= 0:
        raise ValueError(
            f"invalid base range header rows={base_range_rows} range_count={range_count} "
            f"in {args.base_range_fbin}"
        )

    selected = load_selected_queries(
        query_manifest=args.query_manifest,
        families=csv_lower_set(args.families),
        subtypes=csv_lower_set(args.subtypes),
        query_limit=args.query_limit,
        per_subtype_limit=args.per_subtype_limit,
        range_count=range_count,
    )
    selected_manifest_info = write_selected_manifest(selected_manifest_output, selected.queries, args.overwrite)

    row_lookup: RowMapLookup | None = None
    if not args.base_hvecs.exists():
        if not args.row_map.exists():
            raise FileNotFoundError(args.row_map)
        row_lookup = RowMapLookup(args.row_map, workers=args.row_map_workers, batch_size=args.row_map_batch_size)
    query_vector_info = write_query_hvecs(
        query_vector_output,
        selected.queries,
        row_lookup,
        embedding_root,
        args.base_hvecs if args.base_hvecs.exists() else None,
        args.embedding_batch_size,
        args.overwrite,
    )
    if int(query_vector_info["rows"]) != len(selected.queries):
        raise RuntimeError(
            f"query hvecs row count {query_vector_info['rows']} != selected manifest rows {len(selected.queries)}"
        )
    query_range_info = write_query_range_bin(
        query_range_output,
        selected.queries,
        range_count,
        args.overwrite,
    )

    ngram_info = write_ngram_index(
        ngram_index_output,
        token_map=args.token_map,
        ngram_size=args.ngram_size,
        batch_size=args.token_map_batch_size,
        overwrite=args.overwrite,
    )

    summary = {
        "source": {
            "query_manifest": str(args.query_manifest),
            "token_map": str(args.token_map),
            "row_map": str(args.row_map),
            "laion_root": str(args.laion_root),
            "embedding_root": str(embedding_root),
            "base_hvecs": str(args.base_hvecs) if args.base_hvecs.exists() else None,
        },
        "selection": selected.summary,
        "selected_by_subtype": selected.summary["selected_by_subtype"],
        "query_count": len(selected.queries),
        "query_vector": query_vector_info,
        "query_range": query_range_info,
        "selected_manifest": selected_manifest_info,
        "ngram_index": ngram_info,
        "ngram": {
            "token_count": int(ngram_info["token_count"]),
            "ngram_count": int(ngram_info["ngram_count"]),
            "checksum_sha256": ngram_info["checksum_sha256"],
        },
        "output_paths": {
            "query_vector": str(query_vector_output),
            "query_range": str(query_range_output),
            "selected_manifest": str(selected_manifest_output),
            "ngram_index": str(ngram_index_output),
            "summary": str(summary_output),
        },
        "row_map_lookup": None if row_lookup is None else row_lookup.summary(),
        "notes": [
            "Query vectors are materialized as .hvecs and retain float16 payload rows.",
            "When base.hvecs is available, query vectors are reconstructed from it by query_vector_id before falling back to LAION image shards.",
            "Query vector rows follow selected query_manifest rows exactly; "
            "see selected_manifest flashanns_query_index.",
            "Query range rows are materialized from canonical atomic or hybrid range predicates.",
            "The ngram sidecar is built from the global token_map and is intended for string.fuzzy expansion.",
        ],
    }

    if args.compute_gt:
        from scripts.laion.string_gt import compute_string_gpu_gt

        gt_output_root = args.gt_output_root or output_root
        if not gt_output_root.is_absolute():
            gt_output_root = output_root / gt_output_root
        gt_output_root.mkdir(parents=True, exist_ok=True)
        gt_jsonl_output = resolve_output_path(gt_output_root, args.gt_jsonl_output, "gt.string.rows.jsonl")
        gt_summary = compute_string_gpu_gt(
            queries=selected.queries,
            query_vector_path=query_vector_output,
            base_hvecs=args.base_hvecs,
            token_map=args.token_map,
            token_postings_dir=args.token_postings_dir,
            output_root=gt_output_root,
            jsonl_output=gt_jsonl_output,
            topk=args.gt_topk,
            device_name=args.gt_device,
            row_batch_size=args.gt_row_batch_size,
            query_batch_size=args.gt_query_batch_size,
            token_map_batch_size=args.token_map_batch_size,
            ngram_size=args.ngram_size,
            postings_workers=args.gt_postings_workers,
            postings_batch_rows=args.gt_postings_batch_rows,
            postings_token_batch_size=args.gt_postings_token_batch_size,
            host_vector_cache_gb=args.gt_host_vector_cache_gb,
            token_ngram_index_path=args.gt_token_ngram_index,
            max_row_id_exclusive=args.gt_max_row_id_exclusive,
            progress_interval=args.gt_progress_interval,
            overwrite=args.overwrite,
        )
        summary["gt"] = gt_summary
        summary["gt_output_root"] = str(gt_output_root)

    ensure_output_path(summary_output, args.overwrite)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote query vectors: {query_vector_output} rows={query_vector_info['rows']} dim={query_vector_info['dim']}")
    print(
        f"Wrote ngram index:  {ngram_index_output} tokens={ngram_info['token_count']} "
        f"ngrams={ngram_info['ngram_count']} checksum={ngram_info['checksum_sha256']}"
    )
    if args.compute_gt:
        print(f"Wrote GT:           {summary['gt']['outputs'][f'top{args.gt_topk}_gt']['path']}")
    print(f"Wrote summary:      {summary_output}")


if __name__ == "__main__":
    main()
