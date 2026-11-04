#!/usr/bin/env python3
"""Build token postings and share normalized token/ngram binary readers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import hashlib
import os
import re
import struct
import sys
import time
from array import array
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterable

DEFAULT_DATA_ROOT = Path("data/laion100m")
NGRAM_MAGIC = b"FANNSNGRAMIDX\0\0\0"

NGRAM_INDEX_VERSION = 1

NGRAM_NORMALIZER_VERSION = 1

NGRAM_CHECKSUM_KIND_SHA256 = 1

NGRAM_ENDIAN_MARKER = 0x01020304

NGRAM_HEADER = struct.Struct("<16sIIIIQQQQQQQQQQ32s")

POSTINGS_INDEX_MAGIC = b"FANNSTOKPOSTIDX\0"

POSTINGS_INDEX_VERSION = 1

POSTINGS_INDEX_ENDIAN_MARKER = 0x01020304

POSTINGS_INDEX_HEADER = struct.Struct("<16sIIIIQQQQ")

TOKEN_TEXT_COLUMN_CANDIDATES = ("token", "token_text", "text")

DEFAULT_BASE_SPMAT = DEFAULT_DATA_ROOT / "base.spmat"

DEFAULT_TOKEN_MAP = DEFAULT_DATA_ROOT / "token_map.parquet"

DEFAULT_OUTPUT_PATH = Path("artifacts/laion100m/token_postings.bin")


def csv_lower_set(raw: str) -> set[str]:
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def ensure_output_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")


@dataclass
class TokenTextIndex:
    id_to_text: Sequence[str]
    ngram_index: dict[str, Sequence[int]]
    ngram_size: int
    token_id_column: str
    token_text_column: str
    skipped_empty_tokens: int
    max_postings_per_ngram: int

    @property
    def token_count(self) -> int:
        return len(self.id_to_text)


@dataclass(frozen=True)
class TokenPostingsBinaryInfo:
    token_count: int
    posting_count: int
    max_row_id_exclusive: int
    offsets_offset: int
    row_ids_offset: int


@dataclass(frozen=True)
class TokenNgramBinaryInfo:
    label_count: int
    token_count: int
    ngram_size: int
    normalizer_version: int
    token_text_bytes: int
    ngram_count: int
    ngram_text_bytes: int
    posting_count: int
    token_ids_offset: int
    token_ngram_counts_offset: int
    token_text_offsets_offset: int
    token_text_payload_offset: int
    ngram_text_offsets_offset: int
    ngram_offsets_offset: int
    ngram_text_payload_offset: int
    posting_ids_offset: int


class LazyTokenTextTable(Sequence[str]):
    def __init__(
        self,
        label_count: int,
        token_ids: np.ndarray,
        token_text_offsets: np.ndarray,
        token_text_payload: np.ndarray,
    ):
        self.label_count = int(label_count)
        self.token_text_offsets = token_text_offsets
        self.token_text_payload = token_text_payload
        self.slot_by_token_id = np.full((self.label_count,), -1, dtype=np.int64)
        if token_ids.size:
            self.slot_by_token_id[token_ids.astype(np.int64, copy=False)] = np.arange(token_ids.size, dtype=np.int64)
        self.cache: dict[int, str] = {}

    def __len__(self) -> int:
        return self.label_count

    def __getitem__(self, item: int | slice) -> str | list[str]:
        if isinstance(item, slice):
            start, stop, step = item.indices(self.label_count)
            return [self[index] for index in range(start, stop, step)]
        token_id = int(item)
        if token_id < 0 or token_id >= self.label_count:
            raise IndexError(token_id)
        cached = self.cache.get(token_id)
        if cached is not None:
            return cached
        slot = int(self.slot_by_token_id[token_id])
        if slot < 0:
            self.cache[token_id] = ""
            return ""
        begin = int(self.token_text_offsets[slot])
        end = int(self.token_text_offsets[slot + 1])
        text = bytes(self.token_text_payload[begin:end]).decode("ascii")
        self.cache[token_id] = text
        return text


def chunked(items: list[int], size: int) -> Iterable[list[int]]:
    step = max(1, int(size))
    for begin in range(0, len(items), step):
        yield items[begin:begin + step]


def chunked_sequence(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    step = max(1, int(size))
    for begin in range(0, len(items), step):
        yield items[begin:begin + step]


def build_token_postings_lookup(
    postings_layout: str,
    token_postings_dir: Path,
    max_row_id_exclusive: int,
    postings_workers: int,
    postings_batch_rows: int,
    postings_token_batch_size: int,
) -> Any:
    if postings_layout == "token_id_binary":
        return TokenIdPostingsBinaryLookup(
            token_postings_dir,
            max_row_id_exclusive=max_row_id_exclusive,
        )
    if postings_layout == "token_id_npy":
        return TokenIdPostingsNpyLookup(
            token_postings_dir,
            max_row_id_exclusive=max_row_id_exclusive,
        )
    if postings_layout == "duckdb_token_id_parquet":
        return TokenIdPostingsDuckDBLookup(
            token_postings_dir,
            max_row_id_exclusive=max_row_id_exclusive,
            workers=postings_workers,
            batch_rows=postings_batch_rows,
            token_batch_size=postings_token_batch_size,
        )
    return TokenPostingsDuckDBLookup(
        token_postings_dir,
        max_row_id_exclusive=max_row_id_exclusive,
        workers=postings_workers,
        batch_rows=postings_batch_rows,
        token_batch_size=postings_token_batch_size,
    )


def normalize_token_text(text: Any) -> str:
    if text is None:
        return ""
    raw = str(text).encode("utf-8", errors="ignore")
    out = bytearray()
    previous_space = False
    for byte in raw:
        is_digit = 48 <= byte <= 57
        is_upper = 65 <= byte <= 90
        is_lower = 97 <= byte <= 122
        if is_digit or is_upper or is_lower:
            out.append(byte + 32 if is_upper else byte)
            previous_space = False
        elif not previous_space and out:
            out.append(32)
            previous_space = True
    while out and out[-1] == 32:
        out.pop()
    return out.decode("ascii")


def token_ngrams(normalized: str, ngram_size: int) -> list[str]:
    if not normalized:
        return []
    n = max(1, int(ngram_size))
    if len(normalized) <= n:
        return [normalized]
    return sorted({normalized[offset:offset + n] for offset in range(0, len(normalized) - n + 1)})


def token_map_columns(token_map: Path) -> tuple[str, str]:
    import pyarrow.parquet as pq
    schema = pq.read_schema(token_map)
    if "token_id" not in schema.names:
        raise ValueError(f"{token_map} is missing required column: token_id")
    token_column = next((name for name in TOKEN_TEXT_COLUMN_CANDIDATES if name in schema.names), None)
    if token_column is None:
        candidates = ", ".join(TOKEN_TEXT_COLUMN_CANDIDATES)
        raise ValueError(f"{token_map} is missing a token text column; tried: {candidates}")
    return "token_id", token_column


def append_u64(values: array, value: int) -> None:
    if value < 0:
        raise ValueError("cannot append negative value to u64 array")
    values.append(int(value))


def append_u32(values: array, value: int) -> None:
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"value {value} does not fit in u32")
    values.append(int(value))


def build_ngram_parts(
    token_map: Path,
    ngram_size: int,
    batch_size: int,
) -> dict[str, Any]:
    import pyarrow.parquet as pq
    token_id_column, token_text_column = token_map_columns(token_map)
    parquet = pq.ParquetFile(token_map)
    token_ids = array("I")
    token_ngram_counts = array("I")
    token_text_offsets = array("Q", [0])
    token_text_payload = bytearray()
    inverted: defaultdict[str, array] = defaultdict(lambda: array("I"))
    expected_token_id = 0
    skipped_empty_tokens = 0
    max_postings_per_ngram = 0

    for batch in parquet.iter_batches(
        batch_size=max(1, int(batch_size)),
        columns=[token_id_column, token_text_column],
    ):
        data = batch.to_pydict()
        for raw_token_id, raw_token in zip(data[token_id_column], data[token_text_column]):
            if raw_token_id is None:
                raise ValueError(f"{token_map} contains a null token_id at logical row {expected_token_id}")
            token_id = int(raw_token_id)
            if token_id != expected_token_id:
                raise ValueError(
                    f"{token_map} must be sorted by contiguous token_id for this sidecar; "
                    f"expected {expected_token_id}, got {token_id}"
                )
            expected_token_id += 1

            normalized = normalize_token_text(raw_token)
            grams = token_ngrams(normalized, ngram_size)
            if not grams:
                skipped_empty_tokens += 1
                continue

            encoded = normalized.encode("ascii")
            append_u32(token_ids, token_id)
            append_u32(token_ngram_counts, len(grams))
            token_text_payload.extend(encoded)
            append_u64(token_text_offsets, len(token_text_payload))
            for gram in grams:
                postings = inverted[gram]
                append_u32(postings, token_id)
                if len(postings) > max_postings_per_ngram:
                    max_postings_per_ngram = len(postings)

    return {
        "label_count": expected_token_id,
        "token_ids": token_ids,
        "token_ngram_counts": token_ngram_counts,
        "token_text_offsets": token_text_offsets,
        "token_text_payload": bytes(token_text_payload),
        "inverted": dict(inverted),
        "token_id_column": token_id_column,
        "token_text_column": token_text_column,
        "skipped_empty_tokens": skipped_empty_tokens,
        "max_postings_per_ngram": max_postings_per_ngram,
    }


def little_endian_array_bytes(values: array, expected_itemsize: int) -> bytes:
    if values.itemsize != expected_itemsize:
        raise RuntimeError(f"array itemsize {values.itemsize} != expected {expected_itemsize}")
    if sys.byteorder == "little":
        return values.tobytes()
    copy = array(values.typecode, values)
    copy.byteswap()
    return copy.tobytes()


def write_hashed_bytes(handle, hasher: Any, payload: bytes) -> None:
    handle.write(payload)
    hasher.update(payload)


def write_hashed_array(handle, hasher: Any, values: array, expected_itemsize: int) -> None:
    write_hashed_bytes(handle, hasher, little_endian_array_bytes(values, expected_itemsize))


def pack_ngram_header(
    *,
    label_count: int,
    token_count: int,
    ngram_size: int,
    token_text_bytes: int,
    ngram_count: int,
    ngram_text_bytes: int,
    posting_count: int,
    checksum: bytes,
) -> bytes:
    if len(checksum) != 32:
        raise ValueError("ngram checksum must be 32 bytes")
    return NGRAM_HEADER.pack(
        NGRAM_MAGIC,
        NGRAM_INDEX_VERSION,
        NGRAM_HEADER.size,
        NGRAM_ENDIAN_MARKER,
        0,
        int(label_count),
        int(token_count),
        int(ngram_size),
        NGRAM_NORMALIZER_VERSION,
        int(token_text_bytes),
        int(ngram_count),
        int(ngram_text_bytes),
        int(posting_count),
        NGRAM_CHECKSUM_KIND_SHA256,
        0,
        checksum,
    )


def write_ngram_index(
    path: Path,
    token_map: Path,
    ngram_size: int,
    batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    ensure_output_path(path, overwrite)
    parts = build_ngram_parts(token_map, ngram_size=ngram_size, batch_size=batch_size)
    grams = sorted(parts["inverted"])
    ngram_text_offsets = array("Q", [0])
    ngram_offsets = array("Q", [0])
    ngram_text_payload = bytearray()
    posting_count = 0
    for gram in grams:
        encoded = gram.encode("ascii")
        ngram_text_payload.extend(encoded)
        append_u64(ngram_text_offsets, len(ngram_text_payload))
        posting_count += len(parts["inverted"][gram])
        append_u64(ngram_offsets, posting_count)

    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    hasher = hashlib.sha256()
    checksum_placeholder = b"\0" * 32
    with tmp_path.open("wb") as handle:
        handle.write(
            pack_ngram_header(
                label_count=parts["label_count"],
                token_count=len(parts["token_ids"]),
                ngram_size=ngram_size,
                token_text_bytes=len(parts["token_text_payload"]),
                ngram_count=len(grams),
                ngram_text_bytes=len(ngram_text_payload),
                posting_count=posting_count,
                checksum=checksum_placeholder,
            )
        )
        write_hashed_array(handle, hasher, parts["token_ids"], 4)
        write_hashed_array(handle, hasher, parts["token_ngram_counts"], 4)
        write_hashed_array(handle, hasher, parts["token_text_offsets"], 8)
        write_hashed_bytes(handle, hasher, parts["token_text_payload"])
        write_hashed_array(handle, hasher, ngram_text_offsets, 8)
        write_hashed_array(handle, hasher, ngram_offsets, 8)
        write_hashed_bytes(handle, hasher, bytes(ngram_text_payload))
        for gram in grams:
            write_hashed_array(handle, hasher, parts["inverted"][gram], 4)

        checksum = hasher.digest()
        handle.seek(0, os.SEEK_SET)
        handle.write(
            pack_ngram_header(
                label_count=parts["label_count"],
                token_count=len(parts["token_ids"]),
                ngram_size=ngram_size,
                token_text_bytes=len(parts["token_text_payload"]),
                ngram_count=len(grams),
                ngram_text_bytes=len(ngram_text_payload),
                posting_count=posting_count,
                checksum=checksum,
            )
        )

    tmp_path.replace(path)
    return {
        "path": str(path),
        "format": "flashanns_token_ngram_index",
        "magic": NGRAM_MAGIC.rstrip(b"\0").decode("ascii"),
        "version": NGRAM_INDEX_VERSION,
        "header_bytes": NGRAM_HEADER.size,
        "token_id_column": parts["token_id_column"],
        "token_text_column": parts["token_text_column"],
        "label_count": int(parts["label_count"]),
        "token_count": len(parts["token_ids"]),
        "skipped_empty_tokens": int(parts["skipped_empty_tokens"]),
        "ngram_size": int(ngram_size),
        "normalizer_version": NGRAM_NORMALIZER_VERSION,
        "normalizer": "ascii_alnum_lowercase_space_collapse_v1",
        "token_text_bytes": len(parts["token_text_payload"]),
        "ngram_count": len(grams),
        "ngram_text_bytes": len(ngram_text_payload),
        "posting_count": posting_count,
        "max_postings_per_ngram": int(parts["max_postings_per_ngram"]),
        "checksum_kind": "sha256_payload_sections",
        "checksum_sha256": hasher.hexdigest(),
        "bytes": path.stat().st_size,
        "sections": [
            "token_ids:u32[token_count]",
            "token_ngram_counts:u32[token_count]",
            "token_text_offsets:u64[token_count+1]",
            "token_text_payload:u8[token_text_bytes]",
            "ngram_text_offsets:u64[ngram_count+1]",
            "ngram_offsets:u64[ngram_count+1]",
            "ngram_text_payload:u8[ngram_text_bytes]",
            "posting_ids:u32[posting_count]",
        ],
    }


def load_token_text_index(token_map: Path, ngram_size: int, batch_size: int) -> TokenTextIndex:
    import pyarrow.parquet as pq
    token_id_column, token_text_column = token_map_columns(token_map)
    parquet = pq.ParquetFile(token_map)
    id_to_text: list[str] = []
    inverted: defaultdict[str, array] = defaultdict(lambda: array("I"))
    expected_token_id = 0
    skipped_empty_tokens = 0
    max_postings_per_ngram = 0

    for batch in parquet.iter_batches(
        batch_size=max(1, int(batch_size)),
        columns=[token_id_column, token_text_column],
    ):
        data = batch.to_pydict()
        for raw_token_id, raw_token in zip(data[token_id_column], data[token_text_column]):
            if raw_token_id is None:
                raise ValueError(f"{token_map} contains a null token_id at logical row {expected_token_id}")
            token_id = int(raw_token_id)
            if token_id != expected_token_id:
                raise ValueError(
                    f"{token_map} must be sorted by contiguous token_id for GT expansion; "
                    f"expected {expected_token_id}, got {token_id}"
                )
            token_text = str(raw_token or "").lower()
            id_to_text.append(token_text)
            expected_token_id += 1
            grams = token_ngrams(token_text, ngram_size)
            if not grams:
                skipped_empty_tokens += 1
                continue
            for gram in grams:
                postings = inverted[gram]
                append_u32(postings, token_id)
                if len(postings) > max_postings_per_ngram:
                    max_postings_per_ngram = len(postings)

    return TokenTextIndex(
        id_to_text=id_to_text,
        ngram_index=dict(inverted),
        ngram_size=max(1, int(ngram_size)),
        token_id_column=token_id_column,
        token_text_column=token_text_column,
        skipped_empty_tokens=skipped_empty_tokens,
        max_postings_per_ngram=max_postings_per_ngram,
    )


def inspect_token_ngram_binary(path: Path) -> TokenNgramBinaryInfo:
    with path.open("rb") as handle:
        raw = handle.read(NGRAM_HEADER.size)
    if len(raw) != NGRAM_HEADER.size:
        raise ValueError(f"invalid token ngram binary header in {path}")
    (
        magic,
        version,
        header_bytes,
        endian,
        _flags,
        label_count,
        token_count,
        ngram_size,
        normalizer_version,
        token_text_bytes,
        ngram_count,
        ngram_text_bytes,
        posting_count,
        checksum_kind,
        _reserved,
        _checksum_sha256,
    ) = NGRAM_HEADER.unpack(raw)
    if magic.rstrip(b"\0") == b"FANNSOPLEXIDX":
        raise ValueError(f"{path} is a runtime operator index; generate a token ngram index "
                         "with scripts.laion.string_index --ngram-output PATH")
    if magic != NGRAM_MAGIC:
        raise ValueError(f"{path} has unsupported token ngram magic {magic!r}")
    if int(version) != NGRAM_INDEX_VERSION:
        raise ValueError(f"{path} has unsupported token ngram version {version}")
    if int(header_bytes) != NGRAM_HEADER.size:
        raise ValueError(f"{path} header_bytes {header_bytes} != expected {NGRAM_HEADER.size}")
    if int(endian) != NGRAM_ENDIAN_MARKER:
        raise ValueError(f"{path} endian marker mismatch: {endian:#x}")
    if int(checksum_kind) != NGRAM_CHECKSUM_KIND_SHA256:
        raise ValueError(f"{path} checksum kind {checksum_kind} != expected {NGRAM_CHECKSUM_KIND_SHA256}")

    offset = int(header_bytes)
    token_ids_offset = offset
    offset += int(token_count) * np.dtype("<u4").itemsize
    token_ngram_counts_offset = offset
    offset += int(token_count) * np.dtype("<u4").itemsize
    token_text_offsets_offset = offset
    offset += (int(token_count) + 1) * np.dtype("<u8").itemsize
    token_text_payload_offset = offset
    offset += int(token_text_bytes)
    ngram_text_offsets_offset = offset
    offset += (int(ngram_count) + 1) * np.dtype("<u8").itemsize
    ngram_offsets_offset = offset
    offset += (int(ngram_count) + 1) * np.dtype("<u8").itemsize
    ngram_text_payload_offset = offset
    offset += int(ngram_text_bytes)
    posting_ids_offset = offset
    offset += int(posting_count) * np.dtype("<u4").itemsize
    actual_size = path.stat().st_size
    if actual_size < offset:
        raise ValueError(f"truncated token ngram binary {path}: size={actual_size}, expected at least {offset}")

    return TokenNgramBinaryInfo(
        label_count=int(label_count),
        token_count=int(token_count),
        ngram_size=int(ngram_size),
        normalizer_version=int(normalizer_version),
        token_text_bytes=int(token_text_bytes),
        ngram_count=int(ngram_count),
        ngram_text_bytes=int(ngram_text_bytes),
        posting_count=int(posting_count),
        token_ids_offset=token_ids_offset,
        token_ngram_counts_offset=token_ngram_counts_offset,
        token_text_offsets_offset=token_text_offsets_offset,
        token_text_payload_offset=token_text_payload_offset,
        ngram_text_offsets_offset=ngram_text_offsets_offset,
        ngram_offsets_offset=ngram_offsets_offset,
        ngram_text_payload_offset=ngram_text_payload_offset,
        posting_ids_offset=posting_ids_offset,
    )


def load_token_text_index_from_ngram_binary(path: Path) -> TokenTextIndex:
    info = inspect_token_ngram_binary(path)
    token_ids = np.memmap(
        path,
        dtype=np.dtype("<u4"),
        mode="r",
        offset=info.token_ids_offset,
        shape=(info.token_count,),
    )
    token_ngram_counts = np.memmap(
        path,
        dtype=np.dtype("<u4"),
        mode="r",
        offset=info.token_ngram_counts_offset,
        shape=(info.token_count,),
    )
    token_text_offsets = np.memmap(
        path,
        dtype=np.dtype("<u8"),
        mode="r",
        offset=info.token_text_offsets_offset,
        shape=(info.token_count + 1,),
    )
    token_text_payload = np.memmap(
        path,
        dtype=np.dtype("u1"),
        mode="r",
        offset=info.token_text_payload_offset,
        shape=(info.token_text_bytes,),
    )
    ngram_text_offsets = np.memmap(
        path,
        dtype=np.dtype("<u8"),
        mode="r",
        offset=info.ngram_text_offsets_offset,
        shape=(info.ngram_count + 1,),
    )
    ngram_offsets = np.memmap(
        path,
        dtype=np.dtype("<u8"),
        mode="r",
        offset=info.ngram_offsets_offset,
        shape=(info.ngram_count + 1,),
    )
    ngram_text_payload = np.memmap(
        path,
        dtype=np.dtype("u1"),
        mode="r",
        offset=info.ngram_text_payload_offset,
        shape=(info.ngram_text_bytes,),
    )
    posting_ids = np.memmap(
        path,
        dtype=np.dtype("<u4"),
        mode="r",
        offset=info.posting_ids_offset,
        shape=(info.posting_count,),
    )

    ngram_index: dict[str, np.ndarray] = {}
    max_postings_per_ngram = 0
    for gram_idx in range(info.ngram_count):
        begin = int(ngram_text_offsets[gram_idx])
        end = int(ngram_text_offsets[gram_idx + 1])
        gram = bytes(ngram_text_payload[begin:end]).decode("ascii")
        post_begin = int(ngram_offsets[gram_idx])
        post_end = int(ngram_offsets[gram_idx + 1])
        postings = np.asarray(posting_ids[post_begin:post_end], dtype=np.uint32)
        ngram_index[gram] = postings
        if postings.size > max_postings_per_ngram:
            max_postings_per_ngram = int(postings.size)

    id_to_text = LazyTokenTextTable(
        label_count=info.label_count,
        token_ids=np.asarray(token_ids, dtype=np.uint32),
        token_text_offsets=token_text_offsets,
        token_text_payload=token_text_payload,
    )
    skipped_empty_tokens = max(0, info.label_count - info.token_count)
    return TokenTextIndex(
        id_to_text=id_to_text,
        ngram_index=ngram_index,
        ngram_size=info.ngram_size,
        token_id_column="token_id",
        token_text_column="token_ngram_index.bin",
        skipped_empty_tokens=skipped_empty_tokens,
        max_postings_per_ngram=max_postings_per_ngram,
    )


def fuzzy_predicate_mode(predicate: dict[str, Any]) -> str:
    mode = str(predicate.get("fuzzy_match_mode") or predicate.get("match_mode") or "").lower()
    if mode not in {"prefix", "suffix", "contains"}:
        raise ValueError(f"unsupported fuzzy match mode: {mode!r}")
    return mode


def fuzzy_predicate_pattern(predicate: dict[str, Any]) -> str:
    for key in ("pattern", "query", "value", "token_text", "source_token_text"):
        value = predicate.get(key)
        if value is not None and str(value):
            return str(value).lower()
    raise ValueError(f"fuzzy predicate is missing pattern text: {predicate}")


def token_text_from_predicate(predicate: dict[str, Any], token_index: TokenTextIndex | None = None) -> str:
    for key in ("token_text", "source_token_text"):
        value = predicate.get(key)
        if value is not None and str(value):
            return str(value).lower()
    if token_index is None:
        raise ValueError(f"predicate is missing token text and no token_index is available: {predicate}")
    token_id = predicate.get("token_id", predicate.get("source_token_id"))
    if token_id is None:
        raise ValueError(f"predicate is missing token text and token id: {predicate}")
    token_id = int(token_id)
    if token_id < 0 or token_id >= token_index.token_count:
        raise ValueError(f"predicate token_id {token_id} is outside token_map rows={token_index.token_count}")
    return token_index.id_to_text[token_id]


def fuzzy_matching_token_ids(mode: str, pattern: str, token_index: TokenTextIndex) -> np.ndarray:
    pattern = str(pattern or "").lower()
    if not pattern:
        return np.empty((0,), dtype=np.uint32)
    grams = token_ngrams(pattern, token_index.ngram_size)
    candidate_ids: Iterable[int]
    if grams:
        missing = [gram for gram in grams if gram not in token_index.ngram_index]
        if missing:
            return np.empty((0,), dtype=np.uint32)
        gram = min(grams, key=lambda item: len(token_index.ngram_index[item]))
        candidate_ids = token_index.ngram_index[gram]
    else:
        candidate_ids = range(token_index.token_count)

    id_to_text = token_index.id_to_text
    if mode == "prefix":
        matches = [int(token_id) for token_id in candidate_ids if id_to_text[int(token_id)].startswith(pattern)]
    elif mode == "suffix":
        matches = [int(token_id) for token_id in candidate_ids if id_to_text[int(token_id)].endswith(pattern)]
    elif mode == "contains":
        matches = [int(token_id) for token_id in candidate_ids if pattern in id_to_text[int(token_id)]]
    else:
        raise ValueError(f"unsupported fuzzy match mode: {mode}")
    if not matches:
        return np.empty((0,), dtype=np.uint32)
    return np.asarray(matches, dtype=np.uint32)


class TokenPostingsDuckDBLookup:
    def __init__(
        self,
        postings_dir: Path,
        max_row_id_exclusive: int,
        workers: int,
        batch_rows: int,
        token_batch_size: int,
    ):
        try:
            import duckdb
        except ImportError as exc:
            raise SystemExit(
                "duckdb is required for --compute-gt; install it in the LAION torch/parquet environment."
            ) from exc

        if not postings_dir.exists():
            raise FileNotFoundError(f"token postings dir does not exist: {postings_dir}")
        self.postings_dir = postings_dir
        self.max_row_id_exclusive = int(max_row_id_exclusive)
        self.batch_rows = max(1, int(batch_rows))
        self.token_batch_size = max(1, int(token_batch_size))
        self.tokens_requested = 0
        self.tokens_loaded = 0
        self.token_batches = 0
        self.rows_returned_before_union = 0
        self.files_scanned = 0
        self.bucket_dirs = self._discover_bucket_dirs()
        self.bucket_count = max(self.bucket_dirs) + 1 if self.bucket_dirs else 0
        if self.bucket_count <= 0:
            raise FileNotFoundError(f"no posting_bucket=* directories under {postings_dir}")
        self.con = duckdb.connect(database=":memory:")
        self.con.execute(f"SET threads={max(1, int(workers))}")
        self.rows_by_token: dict[str, np.ndarray] = {}
        self.files_by_bucket: dict[int, list[Path]] = {}

    def _discover_bucket_dirs(self) -> dict[int, Path]:
        dirs: dict[int, Path] = {}
        pattern = re.compile(r"posting_bucket=(\d+)$")
        for path in self.postings_dir.iterdir():
            if not path.is_dir():
                continue
            match = pattern.search(path.name)
            if match:
                dirs[int(match.group(1))] = path
        return dirs

    def _files_for_bucket(self, bucket: int) -> list[Path]:
        cached = self.files_by_bucket.get(bucket)
        if cached is not None:
            return cached
        bucket_dir = self.bucket_dirs.get(bucket)
        if bucket_dir is None:
            files: list[Path] = []
        else:
            files = sorted(bucket_dir.glob("*.parquet"))
        self.files_by_bucket[bucket] = files
        return files

    def _load_batch_from_postings(self, bucket: int, tokens: list[str]) -> dict[str, np.ndarray]:
        import pyarrow as pa
        parquet_files = self._files_for_bucket(bucket)
        self.token_batches += 1
        self.files_scanned += len(parquet_files)
        rows_by_token: dict[str, np.ndarray] = {token: np.empty((0,), dtype=np.uint32) for token in tokens}
        if not parquet_files:
            self.tokens_loaded += len(tokens)
            return rows_by_token

        self.con.register("wanted_tokens", pa.table({"token": tokens}))
        rows_by_token_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
        try:
            reader = self.con.execute(
                """
                SELECT p.token, p.row_id
                FROM read_parquet(?, hive_partitioning=true) AS p
                JOIN wanted_tokens AS wanted
                  ON p.token = wanted.token
                WHERE p.row_id < ?
                ORDER BY p.token, p.row_id
                """,
                [[str(path) for path in parquet_files], self.max_row_id_exclusive],
            ).fetch_record_batch(self.batch_rows)

            for batch in reader:
                batch_tokens = batch.column(0).to_pylist()
                batch_rows = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                grouped_rows: dict[str, list[int]] = defaultdict(list)
                for index, token in enumerate(batch_tokens):
                    grouped_rows[str(token)].append(int(batch_rows[index]))
                for token, row_values in grouped_rows.items():
                    rows_by_token_chunks[token].append(np.asarray(row_values, dtype=np.int64))
        finally:
            try:
                self.con.unregister("wanted_tokens")
            except Exception:
                pass

        for token in tokens:
            chunks = rows_by_token_chunks.get(token)
            if chunks:
                rows_array = np.concatenate(chunks)
                if rows_array.size > 1:
                    rows_array = np.unique(rows_array)
                rows_by_token[token] = rows_array.astype(np.uint32, copy=False)
                self.rows_returned_before_union += int(rows_array.size)
        self.tokens_loaded += len(tokens)
        return rows_by_token

    def load_tokens(self, tokens: Sequence[str]) -> dict[str, np.ndarray]:
        import pyarrow as pa
        clean_tokens = sorted({str(token).lower() for token in tokens if str(token or "")})
        if not clean_tokens:
            return {}

        missing = [token for token in clean_tokens if token not in self.rows_by_token]
        self.tokens_requested += len(clean_tokens)
        if missing:
            self.con.register("all_requested_tokens", pa.table({"token": missing}))
            try:
                bucket_rows = self.con.execute(
                    """
                    SELECT token, CAST(hash(token) % ? AS BIGINT) AS bucket
                    FROM all_requested_tokens
                    ORDER BY bucket, token
                    """,
                    [int(self.bucket_count)],
                ).fetchall()
            finally:
                try:
                    self.con.unregister("all_requested_tokens")
                except Exception:
                    pass

            bucketed_tokens: dict[int, list[str]] = defaultdict(list)
            for token, bucket in bucket_rows:
                bucketed_tokens[int(bucket)].append(str(token))

            for bucket in sorted(bucketed_tokens):
                bucket_tokens = bucketed_tokens[bucket]
                for batch_tokens in chunked_sequence(bucket_tokens, self.token_batch_size):
                    batch_rows = self._load_batch_from_postings(bucket, list(batch_tokens))
                    self.rows_by_token.update(batch_rows)

        return {token: self.rows_by_token.get(token, np.empty((0,), dtype=np.uint32)) for token in clean_tokens}

    def rows_for_token_texts_union(self, tokens: Sequence[str]) -> np.ndarray:
        token_rows = self.load_tokens(tokens)
        chunks = [rows for rows in token_rows.values() if rows.size]
        if not chunks:
            return np.empty((0,), dtype=np.uint32)
        rows = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        if rows.size > 1:
            rows = np.unique(rows)
        return rows.astype(np.uint32, copy=False)

    def rows_for_token_ids_union(self, token_ids: np.ndarray, token_index: TokenTextIndex) -> np.ndarray:
        if token_ids.size == 0:
            return np.empty((0,), dtype=np.uint32)
        tokens = [token_index.id_to_text[int(token_id)] for token_id in token_ids]
        return self.rows_for_token_texts_union(tokens)

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "duckdb_token_text_parquet",
            "postings_dir": str(self.postings_dir),
            "max_row_id_exclusive": self.max_row_id_exclusive,
            "bucket_count": self.bucket_count,
            "tokens_requested": self.tokens_requested,
            "tokens_loaded": self.tokens_loaded,
            "token_batches": self.token_batches,
            "token_batch_size": self.token_batch_size,
            "rows_returned_before_union": self.rows_returned_before_union,
            "files_scanned": self.files_scanned,
            "batch_rows": self.batch_rows,
        }


class TokenIdPostingsDuckDBLookup:
    def __init__(
        self,
        postings_dir: Path,
        max_row_id_exclusive: int,
        workers: int,
        batch_rows: int,
        token_batch_size: int,
    ):
        try:
            import duckdb
        except ImportError as exc:
            raise SystemExit(
                "duckdb is required for --compute-gt; install it in the LAION torch/parquet environment."
            ) from exc

        if not postings_dir.exists():
            raise FileNotFoundError(f"token postings dir does not exist: {postings_dir}")
        self.postings_dir = postings_dir
        self.max_row_id_exclusive = int(max_row_id_exclusive)
        self.batch_rows = max(1, int(batch_rows))
        self.token_batch_size = max(1, int(token_batch_size))
        self.tokens_requested = 0
        self.tokens_loaded = 0
        self.token_batches = 0
        self.rows_returned_before_union = 0
        self.files_scanned = 0
        self.bucket_dirs = self._discover_bucket_dirs()
        self.bucket_count = max(self.bucket_dirs) + 1 if self.bucket_dirs else 0
        if self.bucket_count <= 0:
            raise FileNotFoundError(f"no posting_bucket=* directories under {postings_dir}")
        self.con = duckdb.connect(database=":memory:")
        self.con.execute(f"SET threads={max(1, int(workers))}")
        self.rows_by_token_id: dict[int, np.ndarray] = {}
        self.files_by_bucket: dict[int, list[Path]] = {}

    def _discover_bucket_dirs(self) -> dict[int, Path]:
        dirs: dict[int, Path] = {}
        pattern = re.compile(r"posting_bucket=(\d+)$")
        for path in self.postings_dir.iterdir():
            if not path.is_dir():
                continue
            match = pattern.search(path.name)
            if match:
                dirs[int(match.group(1))] = path
        return dirs

    def _files_for_bucket(self, bucket: int) -> list[Path]:
        cached = self.files_by_bucket.get(bucket)
        if cached is not None:
            return cached
        bucket_dir = self.bucket_dirs.get(bucket)
        files = [] if bucket_dir is None else sorted(bucket_dir.glob("*.parquet"))
        self.files_by_bucket[bucket] = files
        return files

    def _load_batch_from_postings(self, bucket: int, token_ids: list[int]) -> dict[int, np.ndarray]:
        import pyarrow as pa
        parquet_files = self._files_for_bucket(bucket)
        self.token_batches += 1
        self.files_scanned += len(parquet_files)
        rows_by_token_id: dict[int, np.ndarray] = {token_id: np.empty((0,), dtype=np.uint32) for token_id in token_ids}
        if not parquet_files:
            self.tokens_loaded += len(token_ids)
            return rows_by_token_id

        self.con.register("wanted_token_ids", pa.table({"token_id": np.asarray(token_ids, dtype=np.int64)}))
        rows_by_token_id_chunks: dict[int, list[np.ndarray]] = defaultdict(list)
        try:
            reader = self.con.execute(
                """
                SELECT p.token_id, p.row_id
                FROM read_parquet(?, hive_partitioning=true) AS p
                JOIN wanted_token_ids AS wanted
                  ON p.token_id = wanted.token_id
                WHERE p.row_id < ?
                ORDER BY p.token_id, p.row_id
                """,
                [[str(path) for path in parquet_files], self.max_row_id_exclusive],
            ).fetch_record_batch(self.batch_rows)

            for batch in reader:
                batch_token_ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                batch_rows = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                grouped_rows: dict[int, list[int]] = defaultdict(list)
                for index, token_id in enumerate(batch_token_ids):
                    grouped_rows[int(token_id)].append(int(batch_rows[index]))
                for token_id, row_values in grouped_rows.items():
                    rows_by_token_id_chunks[token_id].append(np.asarray(row_values, dtype=np.int64))
        finally:
            try:
                self.con.unregister("wanted_token_ids")
            except Exception:
                pass

        for token_id in token_ids:
            chunks = rows_by_token_id_chunks.get(token_id)
            if chunks:
                rows_array = np.concatenate(chunks)
                if rows_array.size > 1:
                    rows_array = np.unique(rows_array)
                rows_by_token_id[token_id] = rows_array.astype(np.uint32, copy=False)
                self.rows_returned_before_union += int(rows_array.size)
        self.tokens_loaded += len(token_ids)
        return rows_by_token_id

    def load_token_ids(self, token_ids: Sequence[int]) -> dict[int, np.ndarray]:
        clean_ids = sorted({int(token_id) for token_id in token_ids if int(token_id) >= 0})
        if not clean_ids:
            return {}
        missing = [token_id for token_id in clean_ids if token_id not in self.rows_by_token_id]
        self.tokens_requested += len(clean_ids)
        if missing:
            bucketed_token_ids: dict[int, list[int]] = defaultdict(list)
            for token_id in missing:
                bucketed_token_ids[token_id % int(self.bucket_count)].append(token_id)
            for bucket in sorted(bucketed_token_ids):
                bucket_token_ids = bucketed_token_ids[bucket]
                for batch_token_ids in chunked_sequence(bucket_token_ids, self.token_batch_size):
                    batch_rows = self._load_batch_from_postings(bucket, [int(token_id) for token_id in batch_token_ids])
                    self.rows_by_token_id.update(batch_rows)
        return {token_id: self.rows_by_token_id.get(token_id, np.empty((0,), dtype=np.uint32)) for token_id in clean_ids}

    def rows_for_token_ids_union(self, token_ids: np.ndarray) -> np.ndarray:
        if token_ids.size == 0:
            return np.empty((0,), dtype=np.uint32)
        token_rows = self.load_token_ids(token_ids.astype(np.int64, copy=False).tolist())
        chunks = [rows for rows in token_rows.values() if rows.size]
        if not chunks:
            return np.empty((0,), dtype=np.uint32)
        rows = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        if rows.size > 1:
            rows = np.unique(rows)
        return rows.astype(np.uint32, copy=False)

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "duckdb_token_id_parquet",
            "postings_dir": str(self.postings_dir),
            "max_row_id_exclusive": self.max_row_id_exclusive,
            "bucket_count": self.bucket_count,
            "tokens_requested": self.tokens_requested,
            "tokens_loaded": self.tokens_loaded,
            "token_batches": self.token_batches,
            "token_batch_size": self.token_batch_size,
            "rows_returned_before_union": self.rows_returned_before_union,
            "files_scanned": self.files_scanned,
            "batch_rows": self.batch_rows,
        }


def inspect_token_postings_binary(path: Path) -> TokenPostingsBinaryInfo:
    with path.open("rb") as handle:
        raw = handle.read(POSTINGS_INDEX_HEADER.size)
    if len(raw) != POSTINGS_INDEX_HEADER.size:
        raise ValueError(f"invalid token postings binary header in {path}")
    (
        magic,
        version,
        header_bytes,
        endian,
        _flags,
        token_count,
        posting_count,
        max_row_id_exclusive,
        _reserved,
    ) = POSTINGS_INDEX_HEADER.unpack(raw)
    if magic != POSTINGS_INDEX_MAGIC:
        raise ValueError(f"{path} has unsupported token postings magic {magic!r}")
    if int(version) != POSTINGS_INDEX_VERSION:
        raise ValueError(f"{path} has unsupported token postings version {version}")
    if int(header_bytes) != POSTINGS_INDEX_HEADER.size:
        raise ValueError(f"{path} header_bytes {header_bytes} != expected {POSTINGS_INDEX_HEADER.size}")
    if int(endian) != POSTINGS_INDEX_ENDIAN_MARKER:
        raise ValueError(f"{path} endian marker mismatch: {endian:#x}")
    token_count = int(token_count)
    posting_count = int(posting_count)
    offsets_offset = int(header_bytes)
    row_ids_offset = offsets_offset + (token_count + 1) * np.dtype("<u8").itemsize
    min_size = row_ids_offset + posting_count * np.dtype("<u4").itemsize
    actual_size = path.stat().st_size
    if actual_size < min_size:
        raise ValueError(f"truncated token postings binary {path}: size={actual_size}, expected at least {min_size}")
    return TokenPostingsBinaryInfo(
        token_count=token_count,
        posting_count=posting_count,
        max_row_id_exclusive=int(max_row_id_exclusive),
        offsets_offset=offsets_offset,
        row_ids_offset=row_ids_offset,
    )


class TokenIdPostingsBinaryLookup:
    def __init__(self, path: Path, max_row_id_exclusive: int):
        if not path.exists():
            raise FileNotFoundError(f"token postings binary does not exist: {path}")
        self.path = path
        self.info = inspect_token_postings_binary(path)
        self.max_row_id_exclusive = min(int(max_row_id_exclusive), self.info.max_row_id_exclusive)
        self.offsets = np.memmap(
            path,
            dtype=np.dtype("<u8"),
            mode="r",
            offset=self.info.offsets_offset,
            shape=(self.info.token_count + 1,),
        )
        self.row_ids = np.memmap(
            path,
            dtype=np.dtype("<u4"),
            mode="r",
            offset=self.info.row_ids_offset,
            shape=(self.info.posting_count,),
        )
        self.tokens_requested = 0
        self.tokens_loaded = 0
        self.rows_returned_before_union = 0

    def posting_size(self, token_id: int) -> int:
        token_id = int(token_id)
        if token_id < 0 or token_id >= self.info.token_count:
            return 0
        return int(self.offsets[token_id + 1] - self.offsets[token_id])

    def _rows_for_token_id(self, token_id: int) -> np.ndarray:
        token_id = int(token_id)
        if token_id < 0 or token_id >= self.info.token_count:
            return np.empty((0,), dtype=np.uint32)
        start = int(self.offsets[token_id])
        end = int(self.offsets[token_id + 1])
        rows = self.row_ids[start:end]
        if 0 < self.max_row_id_exclusive < self.info.max_row_id_exclusive and rows.size:
            rows = rows[rows < self.max_row_id_exclusive]
        self.tokens_loaded += 1
        self.rows_returned_before_union += int(rows.size)
        return rows

    def load_token_ids(self, token_ids: Sequence[int]) -> dict[int, np.ndarray]:
        clean_ids = sorted({int(token_id) for token_id in token_ids if int(token_id) >= 0})
        self.tokens_requested += len(clean_ids)
        return {token_id: self._rows_for_token_id(token_id) for token_id in clean_ids}

    def rows_for_token_ids_union(self, token_ids: np.ndarray) -> np.ndarray:
        if token_ids.size == 0:
            return np.empty((0,), dtype=np.uint32)
        token_rows = self.load_token_ids(token_ids.astype(np.int64, copy=False).tolist())
        chunks = [rows for rows in token_rows.values() if rows.size]
        if not chunks:
            return np.empty((0,), dtype=np.uint32)
        rows = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        if rows.size > 1:
            rows = np.unique(rows)
        return rows.astype(np.uint32, copy=False)

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "token_id_binary",
            "path": str(self.path),
            "token_count": self.info.token_count,
            "posting_count": self.info.posting_count,
            "max_row_id_exclusive": self.max_row_id_exclusive,
            "source_max_row_id_exclusive": self.info.max_row_id_exclusive,
            "tokens_requested": self.tokens_requested,
            "tokens_loaded": self.tokens_loaded,
            "rows_returned_before_union": self.rows_returned_before_union,
        }


def token_id_posting_path(postings_dir: Path, token_id: int) -> Path:
    return postings_dir / f"token_{int(token_id)}.npy"


class TokenIdPostingsNpyLookup:
    def __init__(self, postings_dir: Path, max_row_id_exclusive: int):
        if not postings_dir.exists():
            raise FileNotFoundError(f"token postings dir does not exist: {postings_dir}")
        self.postings_dir = postings_dir
        self.max_row_id_exclusive = int(max_row_id_exclusive)
        self.rows_by_token_id: dict[int, np.ndarray] = {}
        self.tokens_requested = 0
        self.tokens_loaded = 0
        self.rows_returned_before_union = 0
        self.files_scanned = 0

    def _load_token_id(self, token_id: int) -> np.ndarray:
        cached = self.rows_by_token_id.get(token_id)
        if cached is not None:
            return cached
        self.files_scanned += 1
        path = token_id_posting_path(self.postings_dir, token_id)
        if not path.exists():
            rows = np.empty((0,), dtype=np.uint32)
        else:
            rows = np.asarray(np.load(path, mmap_mode="r"), dtype=np.uint32)
            if self.max_row_id_exclusive > 0 and rows.size:
                rows = rows[rows < self.max_row_id_exclusive]
        self.tokens_loaded += 1
        self.rows_returned_before_union += int(rows.size)
        self.rows_by_token_id[token_id] = rows
        return rows

    def posting_size(self, token_id: int) -> int:
        token_id = int(token_id)
        cached = self.rows_by_token_id.get(token_id)
        if cached is not None:
            return int(cached.size)
        path = token_id_posting_path(self.postings_dir, token_id)
        if not path.exists():
            return 0
        rows = np.asarray(np.load(path, mmap_mode="r"), dtype=np.uint32)
        if self.max_row_id_exclusive > 0 and rows.size:
            rows = rows[rows < self.max_row_id_exclusive]
        return int(rows.size)

    def load_token_ids(self, token_ids: Sequence[int]) -> dict[int, np.ndarray]:
        clean_ids = sorted({int(token_id) for token_id in token_ids if int(token_id) >= 0})
        self.tokens_requested += len(clean_ids)
        return {token_id: self._load_token_id(token_id) for token_id in clean_ids}

    def rows_for_token_ids_union(self, token_ids: np.ndarray) -> np.ndarray:
        if token_ids.size == 0:
            return np.empty((0,), dtype=np.uint32)
        token_rows = self.load_token_ids(token_ids.astype(np.int64, copy=False).tolist())
        chunks = [rows for rows in token_rows.values() if rows.size]
        if not chunks:
            return np.empty((0,), dtype=np.uint32)
        rows = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        if rows.size > 1:
            rows = np.unique(rows)
        return rows.astype(np.uint32, copy=False)

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "token_id_npy",
            "postings_dir": str(self.postings_dir),
            "max_row_id_exclusive": self.max_row_id_exclusive,
            "tokens_requested": self.tokens_requested,
            "tokens_loaded": self.tokens_loaded,
            "rows_returned_before_union": self.rows_returned_before_union,
            "files_scanned": self.files_scanned,
        }


def detect_token_postings_layout(postings_dir: Path) -> str:
    if postings_dir.is_file():
        with postings_dir.open("rb") as handle:
            magic = handle.read(16)
        if magic == POSTINGS_INDEX_MAGIC:
            return "token_id_binary"
        raise FileNotFoundError(
            f"unable to detect token postings file layout for {postings_dir}; unsupported magic {magic!r}"
        )
    if any(postings_dir.glob("token_*.npy")):
        return "token_id_npy"
    parquet_file = next(postings_dir.glob("posting_bucket=*/*.parquet"), None)
    if parquet_file is not None:
        import pyarrow.parquet as pq

        schema_names = set(pq.ParquetFile(parquet_file).schema.names)
        if "token_id" in schema_names:
            return "duckdb_token_id_parquet"
        if "token" in schema_names or "token_text" in schema_names or "text" in schema_names:
            return "duckdb_token_text_parquet"
    raise FileNotFoundError(
        f"unable to detect token postings layout under {postings_dir}; expected token_*.npy or posting_bucket=*/*.parquet"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-spmat", type=Path, default=DEFAULT_BASE_SPMAT)
    parser.add_argument("--token-map", type=Path, default=DEFAULT_TOKEN_MAP)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--ngram-output", type=Path,
                        help="Also write a full-vocabulary ngram index for Python query generation and GT.")
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=None,
        help="Optional string query manifest. When set, only token ids needed by this workload are exported; other token ids get empty postings.",
    )
    parser.add_argument("--subtypes", type=str, default="exact,fuzzy")
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--token-map-batch-size", type=int, default=262144)
    parser.add_argument("--spmat-row-batch", type=int, default=1_000_000)
    parser.add_argument("--max-row-id-exclusive", type=int, default=100_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_needed_token_mask(
    query_manifest: Path | None,
    token_map: Path,
    token_count: int,
    subtypes: set[str],
    ngram_size: int,
    token_map_batch_size: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if query_manifest is None:
        return None, {
            "mode": "full_vocab",
            "needed_token_count": token_count,
        }

    token_index = load_token_text_index(token_map, ngram_size=ngram_size, batch_size=token_map_batch_size)
    needed_token_ids: set[int] = set()
    exact_count = 0
    fuzzy_count = 0
    fuzzy_groups: dict[tuple[str, str], int] = {}

    with query_manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("family", "")).lower() != "string":
                continue
            subtype = str(record.get("subtype", "")).lower()
            if subtypes and subtype not in subtypes:
                continue
            predicates = record.get("predicates") or []
            if not predicates:
                raise ValueError(f"query manifest row {line_number} has no predicates")
            predicate = predicates[0]
            if subtype == "exact":
                token_id = predicate.get("token_id", predicate.get("source_token_id"))
                if token_id is None:
                    raise ValueError(f"exact row {line_number} missing token_id/source_token_id")
                needed_token_ids.add(int(token_id))
                exact_count += 1
                continue
            if subtype != "fuzzy":
                continue
            mode = str(
                predicate.get("match_mode")
                or predicate.get("fuzzy_match_mode")
                or predicate.get("mode")
                or ""
            ).lower()
            pattern = str(
                predicate.get("pattern")
                or predicate.get("query")
                or predicate.get("value")
                or predicate.get("token_text")
                or predicate.get("source_token_text")
                or ""
            ).lower()
            if not mode or not pattern:
                raise ValueError(f"fuzzy row {line_number} missing mode/pattern")
            fuzzy_groups[(mode, pattern)] = fuzzy_groups.get((mode, pattern), 0) + 1
            fuzzy_count += 1

    fuzzy_expanded_total = 0
    for mode, pattern in sorted(fuzzy_groups):
        token_ids = fuzzy_matching_token_ids(mode, pattern, token_index)
        fuzzy_expanded_total += int(token_ids.size)
        needed_token_ids.update(int(token_id) for token_id in token_ids.tolist())

    mask = np.zeros((token_count,), dtype=np.bool_)
    if needed_token_ids:
        selected = np.asarray(sorted(needed_token_ids), dtype=np.int64)
        if np.any(selected < 0) or np.any(selected >= token_count):
            raise ValueError("selected token ids exceed token_count")
        mask[selected] = True
    return mask, {
        "mode": "workload_subset",
        "query_manifest": str(query_manifest),
        "subtypes_filter": sorted(subtypes),
        "exact_query_count": exact_count,
        "fuzzy_query_count": fuzzy_count,
        "fuzzy_unique_predicates": len(fuzzy_groups),
        "fuzzy_expanded_total": fuzzy_expanded_total,
        "needed_token_count": int(mask.sum()),
    }


def main() -> None:
    from scripts.laion.base import SPMAT_HEADER_DTYPE, SPMAT_INDEX_DTYPE, inspect_spmat
    args = parse_args()
    if args.spmat_row_batch <= 0:
        raise ValueError("--spmat-row-batch must be positive")
    if args.token_map_batch_size <= 0:
        raise ValueError("--token-map-batch-size must be positive")
    if args.max_row_id_exclusive <= 0:
        raise ValueError("--max-row-id-exclusive must be positive")
    if args.ngram_size <= 0:
        raise ValueError("--ngram-size must be positive")

    for required in (args.base_spmat, args.token_map):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.query_manifest is not None and not args.query_manifest.exists():
        raise FileNotFoundError(args.query_manifest)

    if args.ngram_output is not None:
        reserved = [args.output_path, args.base_spmat, args.token_map,
                    args.output_path.with_suffix(args.output_path.suffix + ".summary.json")]
        if args.query_manifest is not None:
            reserved.append(args.query_manifest)
        if args.ngram_output.resolve() in {path.resolve() for path in reserved}:
            raise ValueError("--ngram-output must differ from input and postings output paths")
        ensure_output_path(args.ngram_output, overwrite=args.overwrite)
    ensure_output_path(args.output_path, overwrite=args.overwrite)
    spmat = inspect_spmat(args.base_spmat)
    if args.max_row_id_exclusive > spmat.rows:
        raise ValueError(f"--max-row-id-exclusive {args.max_row_id_exclusive} exceeds spmat rows {spmat.rows}")

    token_mask, selection_summary = resolve_needed_token_mask(
        args.query_manifest,
        args.token_map,
        spmat.cols,
        csv_lower_set(args.subtypes),
        ngram_size=args.ngram_size,
        token_map_batch_size=args.token_map_batch_size,
    )

    indptr = np.memmap(
        args.base_spmat,
        dtype=SPMAT_HEADER_DTYPE,
        mode="r",
        offset=spmat.indptr_offset,
        shape=(spmat.rows + 1,),
    )
    indices = np.memmap(
        args.base_spmat,
        dtype=SPMAT_INDEX_DTYPE,
        mode="r",
        offset=spmat.indices_offset,
        shape=(spmat.nnz,),
    )

    counts = np.zeros((spmat.cols,), dtype=np.uint64)
    started = time.monotonic()
    row_step = max(1, int(args.spmat_row_batch))

    # Pass 1: count postings per token id.
    for row_begin in range(0, args.max_row_id_exclusive, row_step):
        row_end = min(args.max_row_id_exclusive, row_begin + row_step)
        pos_begin = int(indptr[row_begin])
        pos_end = int(indptr[row_end])
        labels = np.asarray(indices[pos_begin:pos_end], dtype=np.int64)
        if token_mask is not None:
            labels = labels[token_mask[labels]]
        if labels.size:
            counts += np.bincount(labels, minlength=spmat.cols).astype(np.uint64, copy=False)
        if row_end == args.max_row_id_exclusive or row_end % (10 * row_step) == 0:
            elapsed = time.monotonic() - started
            print(
                f"[postings-index] count rows {row_end}/{args.max_row_id_exclusive} "
                f"({row_end / args.max_row_id_exclusive:.1%}) elapsed={elapsed:.1f}s",
                flush=True,
            )

    offsets = np.empty((spmat.cols + 1,), dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    posting_count = int(offsets[-1])
    header_bytes = POSTINGS_INDEX_HEADER.size
    offsets_offset = header_bytes
    row_ids_offset = offsets_offset + offsets.nbytes

    with args.output_path.open("wb") as handle:
        handle.write(
            POSTINGS_INDEX_HEADER.pack(
                POSTINGS_INDEX_MAGIC,
                POSTINGS_INDEX_VERSION,
                POSTINGS_INDEX_HEADER.size,
                POSTINGS_INDEX_ENDIAN_MARKER,
                0,
                int(spmat.cols),
                posting_count,
                int(args.max_row_id_exclusive),
                0,
            )
        )
        handle.truncate(row_ids_offset + posting_count * np.dtype("<u4").itemsize)

    offsets_mmap = np.memmap(
        args.output_path,
        dtype=np.dtype("<u8"),
        mode="r+",
        offset=offsets_offset,
        shape=(spmat.cols + 1,),
    )
    offsets_mmap[:] = offsets
    row_ids_mmap = np.memmap(
        args.output_path,
        dtype=np.dtype("<u4"),
        mode="r+",
        offset=row_ids_offset,
        shape=(posting_count,),
    )
    write_positions = offsets[:-1].copy()

    # Pass 2: fill contiguous row-id slices per token id.
    started = time.monotonic()
    for row_begin in range(0, args.max_row_id_exclusive, row_step):
        row_end = min(args.max_row_id_exclusive, row_begin + row_step)
        pos_begin = int(indptr[row_begin])
        pos_end = int(indptr[row_end])
        labels = np.asarray(indices[pos_begin:pos_end], dtype=np.int64)
        row_counts = np.diff(indptr[row_begin:row_end + 1]).astype(np.int32, copy=False)
        row_ids = np.repeat(np.arange(row_begin, row_end, dtype=np.uint32), row_counts)
        if token_mask is not None:
            selected = token_mask[labels]
            labels = labels[selected]
            row_ids = row_ids[selected]
        if labels.size:
            order = np.argsort(labels, kind="stable")
            sorted_labels = labels[order]
            sorted_rows = row_ids[order]
            split_points = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1 if sorted_labels.size > 1 else np.empty(0, dtype=np.int64)
            starts = np.concatenate((np.asarray([0], dtype=np.int64), split_points))
            ends = np.concatenate((split_points, np.asarray([sorted_labels.size], dtype=np.int64)))
            for start, end in zip(starts, ends):
                token_id = int(sorted_labels[int(start)])
                position = int(write_positions[token_id])
                size = int(end - start)
                row_ids_mmap[position:position + size] = sorted_rows[int(start):int(end)]
                write_positions[token_id] += size
        if row_end == args.max_row_id_exclusive or row_end % (10 * row_step) == 0:
            elapsed = time.monotonic() - started
            print(
                f"[postings-index] fill rows {row_end}/{args.max_row_id_exclusive} "
                f"({row_end / args.max_row_id_exclusive:.1%}) elapsed={elapsed:.1f}s",
                flush=True,
            )

    offsets_mmap.flush()
    row_ids_mmap.flush()
    ngram_summary = None
    if args.ngram_output is not None:
        ngram_summary = write_ngram_index(args.ngram_output, args.token_map, args.ngram_size,
                                          args.token_map_batch_size, args.overwrite)
    summary = {
        "source": {
            "base_spmat": str(args.base_spmat),
            "token_map": str(args.token_map),
            "query_manifest": None if args.query_manifest is None else str(args.query_manifest),
        },
        "output_path": str(args.output_path),
        "token_count": int(spmat.cols),
        "posting_count": posting_count,
        "max_row_id_exclusive": int(args.max_row_id_exclusive),
        "selection": selection_summary,
        "ngram_index": ngram_summary,
        "sample_token_sizes": {
            str(token_id): int(counts[token_id])
            for token_id in np.flatnonzero(counts)[:16].astype(np.int64).tolist()
        },
        "notes": [
            "Offsets are token-id addressed and contiguous across the full vocabulary.",
            "Rows are derived directly from base.spmat and are the source-of-truth for string GT.",
        ],
    }
    summary_path = args.output_path.with_suffix(args.output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote token postings binary {args.output_path} token_count={spmat.cols} posting_count={posting_count}",
        flush=True,
    )
    print(f"Wrote summary {summary_path}", flush=True)


if __name__ == "__main__":
    main()
