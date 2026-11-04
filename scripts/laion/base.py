#!/usr/bin/env python3
"""Materialize LAION100M artifacts for the FlashANNS runtime.

Use `export-vectors --help` to export an NPY image-embedding prefix independently.

Supply aligned base labels, numeric values and a selected query manifest.
Query vectors can be reconstructed either from an existing full `base.hvecs` via `query_vector_id` or, as a fallback, from LAION
`.npy` image-embedding shards. This adapter writes the FlashANNS-native
sidecars that the C++ runtime currently loads directly:

- base/query vectors as float16 .hvecs
- query labels as .spmat when the runtime can consume direct label rows
- query ranges as the multi-range query binary
- filtered query manifest

For `categorical.or` and `string.fuzzy`, use `query_manifest_jsonl_path`
so the runtime can rebuild query label rows with OR/fuzzy semantics from the
manifest itself. Pure range queries can be included only when a range virtual-label
manifest and matching
range-augmented data spmat are supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


from scripts.laion.string_index import write_ngram_index

DEFAULT_LAION_ROOT = Path("data/laion400m")

SPMAT_HEADER_DTYPE = np.dtype("<i8")
SPMAT_INDEX_DTYPE = np.dtype("<i4")
SPMAT_DATA_DTYPE = np.dtype("<f4")
HVECS_DIM_DTYPE = np.dtype("<u4")
HVECS_VALUE_DTYPE = np.dtype("<f2")
QUERY_RANGE_DTYPE = np.dtype(
    [
        ("mode", np.uint8),
        ("upper", np.float32),
        ("lower", np.float32),
    ],
    align=True,
)

MODE_NO_RANGE = np.uint8(0)
MODE_GT = np.uint8(1)
MODE_LT = np.uint8(2)
MODE_BETWEEN = np.uint8(3)


DEFAULT_SUPPORTED_SUBTYPES = {
    ("categorical", "single"),
    ("categorical", "and"),
    ("string", "exact"),
}
RANGE_SUBTYPES = {
    ("range", "similarity"),
    ("range", "width"),
    ("range", "height"),
    ("range", "width_height"),
}
MANIFEST_ONLY_SUBTYPES = {
    ("categorical", "or"),
    ("string", "fuzzy"),
}


@dataclass(frozen=True)
class RowMapEntry:
    row_id: int
    shard_id: int
    shard_row_id: int


@dataclass
class RuntimeQuery:
    source: dict[str, Any]
    label_ids: list[int]
    ranges: list[tuple[int, int, float, float]]


@dataclass(frozen=True)
class HvecsInfo:
    rows: int
    dim: int
    record_bytes: int
    dtype: np.dtype


@dataclass(frozen=True)
class QueryRoutingPlan:
    uses_manifest: bool
    uses_spmat: bool
    uses_ngram_sidecar: bool
    families: list[str]
    subtypes: list[str]


class RowMapLookup:
    def __init__(self, path: Path, workers: int, batch_size: int):
        import duckdb
        import pyarrow.parquet as pq
        self.path = path
        self.batch_size = max(1, int(batch_size))
        self.cache: dict[int, RowMapEntry] = {}
        self.rows_loaded = 0
        self.queries = 0
        schema = pq.read_schema(path)
        if "metadata_shard_id" in schema.names:
            self.shard_col = "metadata_shard_id"
        elif "shard_id" in schema.names:
            self.shard_col = "shard_id"
        else:
            raise ValueError(f"{path} has no metadata_shard_id or shard_id column")
        if "shard_row_id" not in schema.names or "row_id" not in schema.names:
            raise ValueError(f"{path} must contain row_id and shard_row_id columns")

        self.con = duckdb.connect(database=":memory:")
        self.con.execute(f"SET threads={max(1, int(workers))}")
        self.con.execute("CREATE TEMP TABLE wanted_row_ids(row_id BIGINT)")

    def get_entries(self, row_ids: Iterable[int]) -> dict[int, RowMapEntry]:
        unique_ids = sorted({int(row_id) for row_id in row_ids})
        missing = [row_id for row_id in unique_ids if row_id not in self.cache]
        for batch in chunked(missing, self.batch_size):
            self._load_batch(batch)
        still_missing = [row_id for row_id in unique_ids if row_id not in self.cache]
        if still_missing:
            preview = ", ".join(str(row_id) for row_id in still_missing[:8])
            raise KeyError(f"row_map has no entries for row_id(s): {preview}")
        return {row_id: self.cache[row_id] for row_id in unique_ids}

    def _load_batch(self, row_ids: list[int]) -> None:
        if not row_ids:
            return
        self.con.execute("DELETE FROM wanted_row_ids")
        self.con.executemany("INSERT INTO wanted_row_ids VALUES (?)", [(int(row_id),) for row_id in row_ids])
        rows = self.con.execute(
            f"""
            SELECT
              rm.row_id,
              rm.{self.shard_col} AS shard_id,
              rm.shard_row_id
            FROM read_parquet(?) AS rm
            JOIN wanted_row_ids AS wanted
              ON rm.row_id = wanted.row_id
            """,
            [str(self.path)],
        ).fetchall()
        self.queries += 1
        self.rows_loaded += len(rows)
        for row_id, shard_id, shard_row_id in rows:
            self.cache[int(row_id)] = RowMapEntry(
                row_id=int(row_id),
                shard_id=int(shard_id),
                shard_row_id=int(shard_row_id),
            )

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "cached_rows": len(self.cache),
            "rows_loaded": self.rows_loaded,
            "queries": self.queries,
            "batch_size": self.batch_size,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laion-root", type=Path, default=DEFAULT_LAION_ROOT)
    parser.add_argument("--runtime-root", type=Path, required=True)
    query_source = parser.add_mutually_exclusive_group(required=True)
    query_source.add_argument("--query-root", type=Path)
    query_source.add_argument("--query-manifest", type=Path)
    parser.add_argument("--row-map", type=Path, default=None)
    parser.add_argument("--data-label-spmat", type=Path, default=None)
    parser.add_argument("--base-range-fbin", type=Path, default=None)
    parser.add_argument("--token-map", type=Path, default=None)
    parser.add_argument("--range-label-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-vector-path", type=Path, default=None)
    parser.add_argument("--base-vector-output", type=Path, default=None)
    parser.add_argument("--query-vector-output", type=Path, default=None)
    parser.add_argument("--query-label-output", type=Path, default=None)
    parser.add_argument("--query-range-output", type=Path, default=None)
    parser.add_argument("--filtered-manifest-output", type=Path, default=None)
    parser.add_argument("--ngram-index-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--write-base-vectors", action="store_true")
    parser.add_argument(
        "--base-row-limit",
        type=int,
        default=0,
        help="Rows to materialize for base .hvecs. Use 0 for the full runtime row count.",
    )
    parser.add_argument(
        "--max-row-id-exclusive",
        type=int,
        default=0,
        help="Drop query instances whose query_vector_id is outside this row bound. Defaults to base-row-limit when set, otherwise the runtime row count.",
    )
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--per-subtype-limit", type=int, default=0)
    parser.add_argument("--families", type=str, default="")
    parser.add_argument("--subtypes", type=str, default="")
    parser.add_argument(
        "--include-range",
        action="store_true",
        help="Include pure range queries. Requires --range-label-manifest and a matching range-augmented --data-label-spmat.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=8192)
    parser.add_argument("--token-map-batch-size", type=int, default=262144)
    parser.add_argument("--query-manifest-ngram-size", type=int, default=3)
    parser.add_argument("--row-map-workers", type=int, default=8)
    parser.add_argument("--row-map-batch-size", type=int, default=100000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def chunked(items: list[int], size: int) -> Iterable[list[int]]:
    for begin in range(0, len(items), size):
        yield items[begin: begin + size]


def csv_set(raw: str) -> set[str]:
    return {item.strip() for item in str(raw or "").split(",") if item.strip()}


def resolve_path(output_root: Path, explicit: Path | None, default_name: str) -> Path:
    if explicit is None:
        return output_root / default_name
    return explicit if explicit.is_absolute() else output_root / explicit


def ensure_output_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")


@dataclass(frozen=True)
class SpmatInfo:
    rows: int
    cols: int
    nnz: int
    indptr_offset: int
    indices_offset: int


def inspect_spmat(path: Path) -> SpmatInfo:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=SPMAT_HEADER_DTYPE, count=3)
    if header.size != 3:
        raise ValueError(f"invalid spmat header in {path}")
    rows, cols, nnz = [int(value) for value in header.tolist()]
    indptr_offset = 3 * SPMAT_HEADER_DTYPE.itemsize
    indices_offset = indptr_offset + (rows + 1) * SPMAT_HEADER_DTYPE.itemsize
    min_size = indices_offset + nnz * SPMAT_INDEX_DTYPE.itemsize
    actual_size = path.stat().st_size
    if actual_size < min_size:
        raise ValueError(f"truncated spmat {path}: size={actual_size}, expected at least {min_size}")
    return SpmatInfo(rows=rows, cols=cols, nnz=nnz, indptr_offset=indptr_offset, indices_offset=indices_offset)


def read_spmat_header(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=SPMAT_HEADER_DTYPE, count=3)
    if header.size != 3:
        raise ValueError(f"invalid spmat header in {path}")
    rows, cols, nnz = [int(value) for value in header.tolist()]
    return rows, cols, nnz


def read_fbin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.dtype("<u4"), count=2)
    if header.size != 2:
        raise ValueError(f"invalid fbin header in {path}")
    rows, cols = [int(value) for value in header.tolist()]
    return rows, cols


def inspect_hvecs(path: Path) -> HvecsInfo:
    with path.open("rb") as handle:
        first_dim = np.fromfile(handle, dtype=HVECS_DIM_DTYPE, count=1)
    if first_dim.size != 1 or int(first_dim[0]) <= 0:
        raise ValueError(f"invalid hvecs header in {path}")
    dim = int(first_dim[0])
    record_bytes = HVECS_DIM_DTYPE.itemsize + dim * HVECS_VALUE_DTYPE.itemsize
    size = path.stat().st_size
    if size % record_bytes != 0:
        raise ValueError(f"{path} size={size} is not aligned to hvecs record_bytes={record_bytes}")
    dtype = np.dtype([("dim", "<u4"), ("vec", "<f2", (dim,))])
    return HvecsInfo(rows=size // record_bytes, dim=dim, record_bytes=record_bytes, dtype=dtype)


def hvecs_records(path: Path, info: HvecsInfo) -> np.memmap:
    return np.memmap(path, dtype=info.dtype, mode="r", shape=(info.rows,))


def load_embedding_shard(laion_root: Path, shard_id: int, cache: dict[int, np.memmap]) -> np.memmap:
    shard = cache.get(shard_id)
    if shard is None:
        path = laion_root / "images" / f"img_emb_{shard_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"missing LAION embedding shard: {path}")
        shard = np.load(path, mmap_mode="r")
        if shard.ndim != 2:
            raise ValueError(f"expected 2D embedding shard, got shape={shard.shape} for {path}")
        cache[shard_id] = shard
    return shard


def infer_embedding_dim(laion_root: Path) -> int:
    for path in sorted((laion_root / "images").glob("img_emb_*.npy")):
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 2 and arr.shape[1] > 0:
            return int(arr.shape[1])
    raise FileNotFoundError(f"no img_emb_*.npy shards found under {laion_root / 'images'}")


def write_hvecs_batch(handle, matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        raise ValueError("hvecs matrix batch must be 2D")
    dim = int(matrix.shape[1])
    records = np.empty(matrix.shape[0], dtype=[("dim", "<u4"), ("vec", "<f2", (dim,))])
    records["dim"] = dim
    records["vec"] = matrix.astype(np.float16, copy=False)
    records.tofile(handle)


def export_npy_prefix(source: Path, output: Path, rows: int, chunk_rows: int, dim: int) -> None:
    """Write original-order image embeddings, using only chunk-sized allocations."""
    if rows <= 0 or chunk_rows <= 0 or not 0 < dim <= np.iinfo(np.uint32).max:
        raise ValueError("--rows, --chunk-rows and --dim must be positive (dimension fits uint32)")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"{output} already exists; choose a new output path")
    if source.is_dir():
        shards = {}
        for path in source.glob("img_emb_*.npy"):
            match = re.fullmatch(r"img_emb_(\d+)\.npy", path.name)
            if match is None:
                raise ValueError(f"invalid image shard filename: {path.name}")
            shard_id = int(match[1])
            if shard_id in shards:
                raise ValueError(f"duplicate image shard ID {shard_id}")
            shards[shard_id] = path
        paths = sorted(shards.items())
    else:
        paths = [(0, source)]

    # Validate the entire requested prefix before creating output. Retain only
    # file metadata, not a mapping or an array for every source shard.
    selected = []
    remaining = rows
    for expected_id, (shard_id, path) in enumerate(paths):
        if shard_id != expected_id:
            raise ValueError(f"missing image shard img_emb_{expected_id}.npy")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(array, np.memmap):
            if hasattr(array, "close"):
                array.close()
            raise ValueError(f"{path}: expected an NPY array")
        if array.ndim != 2 or array.shape[1] != dim or array.shape[0] == 0:
            raise ValueError(f"{path}: expected a nonempty 2D array with dimension {dim}, got {array.shape}")
        if array.dtype.kind != "f" or array.dtype.itemsize not in (2, 4):
            raise ValueError(f"{path}: expected float16 or float32, got {array.dtype}")
        if path.stat().st_size != array.offset + array.nbytes:
            raise ValueError(f"{path}: NPY payload size does not match its header")
        take = min(remaining, array.shape[0])
        selected.append((path, take))
        del array
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError(f"input contains only {rows - remaining} rows; requested {rows}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            for path, take in selected:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                for begin in range(0, take, chunk_rows):
                    with np.errstate(over="raise", invalid="raise"):
                        matrix = np.asarray(array[begin:min(begin + chunk_rows, take)], dtype="<f2")
                    if not np.isfinite(matrix).all():
                        raise ValueError(f"{path}: non-finite embedding values at rows {begin}:{begin + len(matrix)}")
                    write_hvecs_batch(handle, matrix)
                del array
        # Atomic and exclusive: a concurrently created destination is protected.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink()


def export_vectors_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Export the first N image-embedding rows as FP16 hvecs, preserving original order.")
    parser.add_argument("--input", type=Path, required=True, help="2D NPY or directory containing numerically ordered img_emb_<id>.npy shards")
    parser.add_argument("--output", type=Path, required=True, help="New .hvecs file; existing files are protected")
    parser.add_argument("--rows", type=int, default=100000000)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=512)
    args = parser.parse_args(argv)
    export_npy_prefix(args.input, args.output, args.rows, args.chunk_rows, args.dim)
    print(f"Exported {args.rows} rows x {args.dim} dimensions to {args.output}")


def query_vector_base_hvecs_candidates(
    runtime_root: Path,
    explicit_base_vector_path: Path | None,
    base_vector_output: Path,
) -> list[Path]:
    candidates: list[Path] = []
    for candidate in (
        explicit_base_vector_path,
        base_vector_output,
        runtime_root / "base.hvecs",
    ):
        if candidate is None:
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def csv_lower_set(raw: str) -> set[str]:
    return {item.strip().lower() for item in str(raw or "").split(",") if item.strip()}


def write_query_hvecs(
    path: Path,
    queries: list[RuntimeQuery],
    row_lookup: RowMapLookup | None,
    laion_root: Path | None,
    base_hvecs_path: Path | None,
    batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    ensure_output_path(path, overwrite)
    row_ids = [int(query.source["query_vector_id"]) for query in queries]
    if base_hvecs_path is not None and base_hvecs_path.exists():
        info = inspect_hvecs(base_hvecs_path)
        if any(row_id < 0 or row_id >= info.rows for row_id in row_ids):
            raise ValueError(
                f"query_vector_id exceeds base.hvecs row count {info.rows} in {base_hvecs_path}"
            )
        records = hvecs_records(base_hvecs_path, info)
        rows_written = 0
        with path.open("wb") as handle:
            for batch_ids in chunked(row_ids, max(1, batch_size)):
                batch_rows = np.asarray(batch_ids, dtype=np.int64)
                matrix = np.asarray(records["vec"][batch_rows], dtype=np.float16)
                write_hvecs_batch(handle, matrix)
                rows_written += matrix.shape[0]
        return {
            "path": str(path),
            "rows": rows_written,
            "dim": info.dim,
            "bytes": path.stat().st_size,
            "source": "base_hvecs",
            "source_path": str(base_hvecs_path),
        }

    if row_lookup is None or laion_root is None:
        raise ValueError("row_lookup and laion_root are required when query vectors are sourced from images")
    entries = row_lookup.get_entries(row_ids)
    shard_cache: dict[int, np.memmap] = {}
    dim = infer_embedding_dim(laion_root)
    rows_written = 0
    with path.open("wb") as handle:
        for batch_ids in chunked(row_ids, max(1, batch_size)):
            matrix = np.empty((len(batch_ids), dim), dtype=np.float16)
            by_shard: dict[int, list[tuple[int, int]]] = {}
            for out_idx, row_id in enumerate(batch_ids):
                entry = entries[int(row_id)]
                by_shard.setdefault(entry.shard_id, []).append((out_idx, entry.shard_row_id))
            for shard_id, locations in by_shard.items():
                shard = load_embedding_shard(laion_root, shard_id, shard_cache)
                shard_rows = np.asarray([shard_row for _out_idx, shard_row in locations], dtype=np.int64)
                vectors = np.asarray(shard[shard_rows], dtype=np.float16)
                for local_idx, (out_idx, _shard_row) in enumerate(locations):
                    matrix[out_idx] = vectors[local_idx]
            write_hvecs_batch(handle, matrix)
            rows_written += matrix.shape[0]
    return {
        "path": str(path),
        "rows": rows_written,
        "dim": dim,
        "bytes": path.stat().st_size,
        "source": "laion_images",
        "source_path": str(laion_root / "images"),
    }


def iter_row_map_prefix(path: Path, row_limit: int, columns: list[str], batch_size: int):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    remaining = row_limit
    for batch in parquet.iter_batches(batch_size=max(1, batch_size), columns=columns):
        table = batch.to_pydict()
        row_ids = table["row_id"]
        if not row_ids:
            continue
        if remaining > 0 and len(row_ids) > remaining:
            keep = remaining
            table = {name: values[:keep] for name, values in table.items()}
        yield table
        if remaining > 0:
            remaining -= len(table["row_id"])
            if remaining <= 0:
                break


def write_base_hvecs(
    path: Path,
    row_map_path: Path,
    laion_root: Path,
    rows_to_write: int,
    batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    import pyarrow.parquet as pq
    ensure_output_path(path, overwrite)
    if rows_to_write <= 0:
        raise ValueError("rows_to_write must be positive")
    schema = pq.read_schema(row_map_path)
    if "metadata_shard_id" in schema.names:
        shard_col = "metadata_shard_id"
    elif "shard_id" in schema.names:
        shard_col = "shard_id"
    else:
        raise ValueError(f"{row_map_path} has no metadata_shard_id or shard_id column")

    columns = ["row_id", shard_col, "shard_row_id"]
    shard_cache: dict[int, np.memmap] = {}
    dim = infer_embedding_dim(laion_root)
    expected_row = 0
    rows_written = 0
    next_progress = max(batch_size, 1)
    with path.open("wb") as handle:
        for batch in iter_row_map_prefix(row_map_path, rows_to_write, columns, batch_size):
            row_ids = np.asarray(batch["row_id"], dtype=np.int64)
            shard_ids = np.asarray(batch[shard_col], dtype=np.int64)
            shard_row_ids = np.asarray(batch["shard_row_id"], dtype=np.int64)
            if row_ids.size == 0:
                continue
            expected_ids = np.arange(expected_row, expected_row + row_ids.size, dtype=np.int64)
            if not np.array_equal(row_ids, expected_ids):
                raise ValueError(
                    "row_map_reordered.parquet must be sorted by contiguous row_id for base export; "
                    f"expected row_id {int(expected_ids[0])}, got {int(row_ids[0])}"
                )
            matrix = np.empty((row_ids.size, dim), dtype=np.float16)
            for shard_id in sorted(set(int(value) for value in shard_ids.tolist())):
                mask = shard_ids == shard_id
                shard = load_embedding_shard(laion_root, shard_id, shard_cache)
                matrix[mask] = np.asarray(shard[shard_row_ids[mask]], dtype=np.float16)
            write_hvecs_batch(handle, matrix)
            expected_row += row_ids.size
            rows_written += row_ids.size
            if rows_written >= next_progress or rows_written >= rows_to_write:
                print(
                    f"[materialize] base.hvecs rows {rows_written}/{rows_to_write} "
                    f"bytes={path.stat().st_size}",
                    flush=True,
                )
                while next_progress <= rows_written:
                    next_progress += max(batch_size, 1)
    return {
        "path": str(path),
        "rows": rows_written,
        "dim": dim,
        "bytes": path.stat().st_size,
    }


def write_spmat(path: Path, label_rows: list[list[int]], col_count: int, overwrite: bool) -> dict[str, Any]:
    ensure_output_path(path, overwrite)
    indptr = [0]
    indices: list[int] = []
    for labels in label_rows:
        clean = sorted({int(label) for label in labels})
        for label in clean:
            if label < 0 or label >= col_count:
                raise ValueError(f"query label id {label} is outside data spmat column count {col_count}")
        indices.extend(clean)
        indptr.append(len(indices))

    with path.open("wb") as handle:
        np.asarray([len(label_rows), col_count, len(indices)], dtype=SPMAT_HEADER_DTYPE).tofile(handle)
        np.asarray(indptr, dtype=SPMAT_HEADER_DTYPE).tofile(handle)
        np.asarray(indices, dtype=SPMAT_INDEX_DTYPE).tofile(handle)
        np.ones(len(indices), dtype=SPMAT_DATA_DTYPE).tofile(handle)
    return {
        "path": str(path),
        "rows": len(label_rows),
        "cols": col_count,
        "nnz": len(indices),
        "bytes": path.stat().st_size,
    }


def empty_query_ranges(query_count: int, range_count: int) -> np.ndarray:
    # Zero the aligned record padding as well as named fields so repeated
    # materialization produces byte-identical query_range.bin artifacts.
    ranges = np.zeros((query_count, range_count), dtype=QUERY_RANGE_DTYPE)
    ranges["mode"] = MODE_NO_RANGE
    return ranges


def inclusive_upper_for_runtime(value: float) -> np.float32:
    return np.nextafter(np.float32(value), np.float32(math.inf), dtype=np.float32)


def write_query_range_bin(
    path: Path,
    queries: list[RuntimeQuery],
    range_count: int,
    overwrite: bool,
) -> dict[str, Any]:
    ensure_output_path(path, overwrite)
    ranges = empty_query_ranges(len(queries), range_count)
    active_records = 0
    for query_idx, query in enumerate(queries):
        for column, mode, lower, upper in query.ranges:
            if column < 0 or column >= range_count:
                raise ValueError(f"range column {column} is outside base range count {range_count}")
            ranges["mode"][query_idx, column] = np.uint8(mode)
            ranges["lower"][query_idx, column] = np.float32(lower)
            ranges["upper"][query_idx, column] = np.float32(upper)
            if mode != int(MODE_NO_RANGE):
                active_records += 1

    with path.open("wb") as handle:
        np.asarray([len(queries), range_count], dtype=np.dtype("<u4")).tofile(handle)
        ranges.reshape(-1).tofile(handle)
    return {
        "path": str(path),
        "rows": len(queries),
        "range_count": range_count,
        "active_range_terms": active_records,
        "bytes": path.stat().st_size,
    }


def predicate_label_ids(record: dict[str, Any]) -> list[int]:
    labels = []
    for predicate in record.get("predicates") or []:
        token_id = predicate.get("token_id")
        if token_id is None:
            raise ValueError("predicate is missing token_id")
        labels.append(int(token_id))
    if not labels:
        raise ValueError("query has no predicate token ids")
    return labels


def range_terms(record: dict[str, Any]) -> list[tuple[int, int, float, float]]:
    terms = []
    for predicate in record.get("predicates") or []:
        column = predicate.get("column")
        if column is None:
            raise ValueError("range predicate is missing column")
        op = str(predicate.get("op", "between"))
        lo = predicate.get("lo", predicate.get("lower", predicate.get("min")))
        hi = predicate.get("hi", predicate.get("upper", predicate.get("max")))
        if op == "between":
            if lo is None or hi is None:
                raise ValueError("between range predicate requires lo/hi")
            terms.append((int(column), int(MODE_BETWEEN), float(lo), float(inclusive_upper_for_runtime(float(hi)))))
        elif op == "gt":
            if lo is None:
                raise ValueError("gt range predicate requires lower/lo")
            terms.append((int(column), int(MODE_GT), float(lo), 0.0))
        elif op == "lt":
            if hi is None:
                raise ValueError("lt range predicate requires upper/hi")
            terms.append((int(column), int(MODE_LT), 0.0, float(hi)))
        else:
            raise ValueError(f"unsupported range predicate op: {op}")
    if not terms:
        raise ValueError("range query has no range terms")
    return terms


def subtype_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("family")), str(record.get("subtype"))


def subtype_allowed(record: dict[str, Any], families: set[str], subtypes: set[str]) -> bool:
    family, subtype = subtype_key(record)
    family = family.lower()
    subtype = subtype.lower()
    if families and family not in families:
        return False
    return not subtypes or subtype in subtypes or f"{family}.{subtype}" in subtypes


def convert_record(
    record: dict[str, Any],
    data_label_cols: int,
    range_count: int,
    include_range: bool,
) -> tuple[RuntimeQuery | None, str]:
    family, subtype = subtype_key(record)
    family = family.lower()
    subtype = subtype.lower()
    key = (family, subtype)
    if key in DEFAULT_SUPPORTED_SUBTYPES:
        try:
            labels = predicate_label_ids(record)
        except ValueError as exc:
            return None, f"invalid_label_predicate:{exc}"
        for label in labels:
            if label < 0 or label >= data_label_cols:
                return None, "label_id_out_of_range"
        return RuntimeQuery(record, labels, []), "included"

    if key == ("string", "fuzzy"):
        predicates = record.get("predicates") or []
        if not predicates:
            return None, "invalid_fuzzy_predicate:query has no predicates"
        for predicate in predicates:
            pattern = predicate.get("pattern")
            token_text = predicate.get("token_text")
            source_token_text = predicate.get("source_token_text")
            if pattern is None and token_text is None and source_token_text is None:
                return None, "invalid_fuzzy_predicate:missing pattern text"
        return RuntimeQuery(record, [], []), "included"

    if key in RANGE_SUBTYPES:
        if not include_range:
            return None, "range_requires_include_range"
        try:
            terms = range_terms(record)
        except ValueError as exc:
            return None, f"invalid_range_predicate:{exc}"
        for column, _mode, _lower, _upper in terms:
            if column < 0 or column >= range_count:
                return None, "range_column_out_of_range"
        return RuntimeQuery(record, [], terms), "included"

    if key == ("categorical", "or"):
        try:
            labels = predicate_label_ids(record)
        except ValueError as exc:
            return None, f"invalid_label_predicate:{exc}"
        for label in labels:
            if label < 0 or label >= data_label_cols:
                return None, "label_id_out_of_range"
        return RuntimeQuery(record, labels, []), "included"
    return None, "unsupported_subtype"


def query_manifest_tuple(query: RuntimeQuery) -> tuple[str, str]:
    return (
        str(query.source.get("family", "")).lower(),
        str(query.source.get("subtype", "")).lower(),
    )


def build_query_routing_plan(queries: list[RuntimeQuery]) -> QueryRoutingPlan:
    families = sorted({family for family, _subtype in map(query_manifest_tuple, queries)})
    subtypes = sorted({f"{family}.{subtype}" for family, subtype in map(query_manifest_tuple, queries)})
    uses_manifest = any(query_manifest_tuple(query) in MANIFEST_ONLY_SUBTYPES for query in queries)
    uses_ngram_sidecar = any(query_manifest_tuple(query) == ("string", "fuzzy") for query in queries)
    uses_spmat = not uses_manifest
    return QueryRoutingPlan(
        uses_manifest=uses_manifest,
        uses_spmat=uses_spmat,
        uses_ngram_sidecar=uses_ngram_sidecar,
        families=families,
        subtypes=subtypes,
    )


def validate_query_plan(query_plan: QueryRoutingPlan) -> None:
    if not query_plan.uses_manifest:
        return
    if any(subtype.startswith("range.") for subtype in query_plan.subtypes):
        raise ValueError(
            "mixed manifest-mode and range workloads are not supported by "
            "scripts.laion.base yet; split range queries into a "
            "separate materialization run"
        )


def load_runtime_queries(
    query_manifest: Path,
    data_label_cols: int,
    range_count: int,
    max_row_id_exclusive: int,
    include_range: bool,
    families: set[str],
    subtypes: set[str],
    query_limit: int,
    per_subtype_limit: int,
) -> tuple[list[RuntimeQuery], dict[str, Any]]:
    queries: list[RuntimeQuery] = []
    skipped_by_reason: Counter[str] = Counter()
    scanned_by_subtype: Counter[str] = Counter()
    included_by_subtype: Counter[str] = Counter()
    dropped_by_row_bound = 0
    with query_manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not subtype_allowed(record, families, subtypes):
                continue
            family, subtype = subtype_key(record)
            subtype_name = f"{family}.{subtype}"
            scanned_by_subtype[subtype_name] += 1
            row_id = record.get("query_vector_id")
            if row_id is None:
                skipped_by_reason["missing_query_vector_id"] += 1
                continue
            if int(row_id) >= max_row_id_exclusive:
                dropped_by_row_bound += 1
                continue
            if per_subtype_limit > 0 and included_by_subtype[subtype_name] >= per_subtype_limit:
                skipped_by_reason["per_subtype_limit"] += 1
                continue
            query, reason = convert_record(record, data_label_cols, range_count, include_range)
            if query is None:
                skipped_by_reason[reason] += 1
                continue
            query.source = dict(query.source)
            query.source["flashanns_query_index"] = len(queries)
            query.source["flashanns_runtime_semantics"] = runtime_semantics_note(family, subtype)
            queries.append(query)
            included_by_subtype[subtype_name] += 1
            if query_limit > 0 and len(queries) >= query_limit:
                break

    if not queries:
        raise RuntimeError(
            f"No runtime-compatible queries selected from {query_manifest}; "
            "relax filters or row bounds."
        )
    return queries, {
        "records_included": len(queries),
        "scanned_by_subtype": dict(scanned_by_subtype),
        "included_by_subtype": dict(included_by_subtype),
        "skipped_by_reason": dict(skipped_by_reason),
        "dropped_by_row_bound": dropped_by_row_bound,
    }


def runtime_semantics_note(family: str, subtype: str) -> str:
    if family == "categorical" and subtype in {"single", "and"}:
        return "conjunctive token labels; runtime chooses one primary label and checks remaining labels"
    if family == "string" and subtype == "exact":
        return "single exact token label"
    if family == "range":
        return "pure range query; requires range virtual labels merged into data spmat"
    return "runtime-compatible"


def write_filtered_manifest(path: Path, queries: list[RuntimeQuery], overwrite: bool) -> dict[str, Any]:
    ensure_output_path(path, overwrite)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query.source, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "rows": len(queries),
        "bytes": path.stat().st_size,
    }


def max_manifest_label_id(path: Path) -> int:
    root = json.loads(path.read_text(encoding="utf-8"))
    max_label = -1
    entries = root.get("ranges")
    if isinstance(entries, list):
        ranges = entries
    else:
        ranges = [root]
    for entry in ranges:
        for level in entry.get("levels", []):
            for partition in level.get("partitions", []):
                label = partition.get("label_id")
                if label is not None:
                    max_label = max(max_label, int(label))
    return max_label


def validate_args(args: argparse.Namespace) -> None:
    if args.base_row_limit < 0:
        raise ValueError("--base-row-limit must be non-negative")
    if args.max_row_id_exclusive < 0:
        raise ValueError("--max-row-id-exclusive must be non-negative")
    if args.query_limit < 0 or args.per_subtype_limit < 0:
        raise ValueError("--query-limit and --per-subtype-limit must be non-negative")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive")
    if args.token_map_batch_size <= 0:
        raise ValueError("--token-map-batch-size must be positive")
    if args.query_manifest_ngram_size <= 0:
        raise ValueError("--query-manifest-ngram-size must be positive")
    if args.row_map_workers <= 0 or args.row_map_batch_size <= 0:
        raise ValueError("--row-map-workers and --row-map-batch-size must be positive")
    if args.include_range and args.range_label_manifest is None:
        raise ValueError("--include-range requires --range-label-manifest")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "export-vectors":
        export_vectors_main(sys.argv[2:])
        return
    args = parse_args()
    validate_args(args)

    runtime_root = args.runtime_root
    query_manifest = args.query_manifest or (args.query_root / "query_manifest.jsonl")
    row_map = args.row_map or (runtime_root / "row_map_reordered.parquet")
    data_label_spmat = args.data_label_spmat or (runtime_root / "base_duckdb_reordered.spmat")
    base_range_fbin = args.base_range_fbin or (runtime_root / "base_ranges_reordered.fbin")
    token_map = args.token_map or (runtime_root / "token_map.parquet")
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    base_vector_output = resolve_path(output_root, args.base_vector_output, "base.hvecs")
    base_vector_path = args.base_vector_path or base_vector_output
    query_vector_base_hvecs = next(
        (
            candidate
            for candidate in query_vector_base_hvecs_candidates(runtime_root, args.base_vector_path, base_vector_output)
            if candidate.exists() and candidate.suffix.lower() == ".hvecs"
        ),
        None,
    )
    query_vector_output = resolve_path(output_root, args.query_vector_output, "query.hvecs")
    query_label_output = resolve_path(output_root, args.query_label_output, "query.spmat")
    query_range_output = resolve_path(output_root, args.query_range_output, "query_range.bin")
    filtered_manifest_output = resolve_path(
        output_root,
        args.filtered_manifest_output,
        "query_manifest.flashanns.jsonl",
    )
    ngram_index_output = resolve_path(output_root, args.ngram_index_output, "token_ngram_index.bin")
    summary_output = resolve_path(output_root, args.summary_output, "materialize_summary.json")

    required_inputs = [query_manifest, data_label_spmat, base_range_fbin]
    if args.write_base_vectors or query_vector_base_hvecs is None:
        required_inputs.append(row_map)
    for required in required_inputs:
        if not required.exists():
            raise FileNotFoundError(required)

    data_rows, data_label_cols, data_nnz = read_spmat_header(data_label_spmat)
    range_rows, range_count = read_fbin_header(base_range_fbin)
    if data_rows != range_rows:
        raise ValueError(f"data label rows {data_rows} != base range rows {range_rows}")
    if range_count <= 0:
        raise ValueError(f"invalid base range count {range_count} in {base_range_fbin}")
    if args.range_label_manifest is not None:
        max_range_label = max_manifest_label_id(args.range_label_manifest)
        if max_range_label >= data_label_cols:
            raise ValueError(
                "range label manifest references label ids beyond data spmat columns; "
                "pass the range-augmented data spmat via --data-label-spmat"
            )

    max_row_id_exclusive = args.max_row_id_exclusive
    if max_row_id_exclusive == 0:
        max_row_id_exclusive = args.base_row_limit if args.base_row_limit > 0 else data_rows
    if max_row_id_exclusive > data_rows:
        raise ValueError(f"--max-row-id-exclusive {max_row_id_exclusive} exceeds runtime rows {data_rows}")

    queries, query_load = load_runtime_queries(
        query_manifest=query_manifest,
        data_label_cols=data_label_cols,
        range_count=range_count,
        max_row_id_exclusive=max_row_id_exclusive,
        include_range=args.include_range,
        families=csv_set(args.families),
        subtypes=csv_set(args.subtypes),
        query_limit=args.query_limit,
        per_subtype_limit=args.per_subtype_limit,
    )
    query_plan = build_query_routing_plan(queries)
    validate_query_plan(query_plan)
    if query_plan.uses_ngram_sidecar and not token_map.exists():
        raise FileNotFoundError(
            f"{token_map} is required when selected queries include string.fuzzy"
        )

    row_lookup: RowMapLookup | None = None
    if query_vector_base_hvecs is None:
        row_lookup = RowMapLookup(row_map, workers=args.row_map_workers, batch_size=args.row_map_batch_size)
    query_vector_info = write_query_hvecs(
        query_vector_output,
        queries,
        row_lookup,
        args.laion_root if query_vector_base_hvecs is None else None,
        query_vector_base_hvecs,
        args.embedding_batch_size,
        args.overwrite,
    )
    if query_plan.uses_spmat:
        query_label_info = write_spmat(
            query_label_output,
            [query.label_ids for query in queries],
            data_label_cols,
            args.overwrite,
        )
    else:
        query_label_info = {
            "path": str(query_label_output),
            "rows": len(queries),
            "cols": data_label_cols,
            "nnz": None,
            "bytes": 0,
            "written": False,
            "reason": "query labels are reconstructed from query_manifest_jsonl_path at runtime",
        }
    query_range_info = write_query_range_bin(
        query_range_output,
        queries,
        range_count,
        args.overwrite,
    )
    filtered_manifest_info = write_filtered_manifest(filtered_manifest_output, queries, args.overwrite)
    ngram_info: dict[str, Any] | None = None
    if query_plan.uses_ngram_sidecar:
        ngram_info = write_ngram_index(
            ngram_index_output,
            token_map=token_map,
            ngram_size=args.query_manifest_ngram_size,
            batch_size=args.token_map_batch_size,
            overwrite=args.overwrite,
        )

    base_vector_info: dict[str, Any] = {
        "path": str(base_vector_path),
        "written": False,
        "exists": base_vector_path.exists(),
    }
    if args.write_base_vectors:
        base_rows_to_write = args.base_row_limit if args.base_row_limit > 0 else data_rows
        base_vector_info = write_base_hvecs(
            base_vector_output,
            row_map,
            args.laion_root,
            base_rows_to_write,
            args.embedding_batch_size,
            args.overwrite,
        )
        base_vector_info["written"] = True
        base_vector_info["exists"] = True
        base_vector_path = base_vector_output
    summary = {
        "source": {
            "laion_root": str(args.laion_root),
            "runtime_root": str(runtime_root),
            "query_manifest": str(query_manifest),
            "row_map": str(row_map),
            "data_label_spmat": str(data_label_spmat),
            "base_range_fbin": str(base_range_fbin),
            "query_vector_base_hvecs": None if query_vector_base_hvecs is None else str(query_vector_base_hvecs),
            "token_map": str(token_map) if token_map.exists() else None,
            "range_label_manifest": None if args.range_label_manifest is None else str(args.range_label_manifest),
        },
        "runtime_contract": {
            "data_rows": data_rows,
            "data_label_cols": data_label_cols,
            "data_label_nnz": data_nnz,
            "range_rows": range_rows,
            "range_count": range_count,
            "max_row_id_exclusive": max_row_id_exclusive,
            "supported_direct_label_subtypes": sorted(
                f"{family}.{subtype}" for family, subtype in DEFAULT_SUPPORTED_SUBTYPES
            ),
            "manifest_only_subtypes": sorted(f"{family}.{subtype}" for family, subtype in MANIFEST_ONLY_SUBTYPES),
            "range_included": bool(args.include_range),
        },
        "query_routing": {
            "uses_manifest": query_plan.uses_manifest,
            "uses_spmat": query_plan.uses_spmat,
            "uses_ngram_sidecar": query_plan.uses_ngram_sidecar,
            "families": query_plan.families,
            "subtypes": query_plan.subtypes,
        },
        "query_load": query_load,
        "artifacts": {
            "base_vector": base_vector_info,
            "query_vector": query_vector_info,
            "query_label_spmat": query_label_info,
            "query_range": query_range_info,
            "filtered_manifest": filtered_manifest_info,
            "ngram_index": ngram_info,
        },
        "row_map_lookup": None if row_lookup is None else row_lookup.summary(),
        "warnings": [
            "LAION100M base/query vectors are materialized as .hvecs to preserve float16 payloads.",
            "When an existing full base.hvecs is available, query vectors are reconstructed directly from it by query_vector_id before falling back to LAION image shards.",
            "categorical.or and string.fuzzy use query_manifest_jsonl_path to reconstruct OR/fuzzy label rows from the manifest.",
            "query_label_spmat_path remains the direct path for pure conjunctive categorical/range-compatible workloads.",
            "Range queries require range virtual labels already merged into data_label_spmat plus a matching manifest.",
            "LAION inclusive upper range bounds are written with nextafter(hi, +inf) to match FlashANNS upper-exclusive between filtering.",
        ],
    }
    ensure_output_path(summary_output, args.overwrite)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {len(queries)} runtime-compatible queries")
    print(f"Query vectors: {query_vector_output}")
    print(f"Query labels:  {query_label_output}")
    print(f"Query ranges:  {query_range_output}")
    if args.write_base_vectors:
        print(f"Base vectors:  {base_vector_output}")
    else:
        print(f"Existing base vectors: {base_vector_path}")
    print(f"Summary:       {summary_output}")


if __name__ == "__main__":
    main()
