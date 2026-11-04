#!/usr/bin/env python3
"""Generate pairwise LAION hybrid query instances from atomic predicate templates.

This generator builds cross-family hybrid workloads over four atom families:
categorical.single, range, string.exact, and string.fuzzy.  It intentionally
skips string.exact x string.fuzzy because both are string predicates.

For every requested pair, both top-level AND and OR predicates can be emitted.
The final selectivity is computed from the true combined row set whenever the
needed row sets can be materialized from the runtime artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from scripts.laion.predicates import range_predicate_mask

from scripts.laion.string_index import (
    TokenIdPostingsBinaryLookup,
    fuzzy_matching_token_ids,
    fuzzy_predicate_mode,
    fuzzy_predicate_pattern,
    load_token_text_index_from_ngram_binary,
)


DEFAULT_TOKEN_POSTINGS = Path("artifacts/laion100m/token_postings.bin")
DEFAULT_NGRAM_INDEX = Path("artifacts/laion100m/token_ngram_index.bin")

FBIN_HEADER_DTYPE = np.dtype("<u4")
ATOM_ORDER = ("categorical", "range", "exact", "fuzzy")
PAIR_FAMILIES = (
    ("categorical", "range"),
    ("categorical", "exact"),
    ("categorical", "fuzzy"),
    ("range", "exact"),
    ("range", "fuzzy"),
)
TOP_LOGICS = ("and", "or")

AND_MIN_SELECTIVITY = 1e-5
OR_MIN_SELECTIVITY = 1e-4
DEFAULT_MAX_SELECTIVITY = 5e-2
DEFAULT_QUERIES_PER_HYBRID_SUBTYPE = 5000


@dataclass(frozen=True)
class PairSpec:
    left: str
    right: str
    logic: str

    @property
    def label(self) -> str:
        return f"{self.left_label}__{self.right_label}.{self.logic}"

    @property
    def left_label(self) -> str:
        return atom_label(self.left)

    @property
    def right_label(self) -> str:
        return atom_label(self.right)


@dataclass
class GenerationStats:
    attempted_pairs: int = 0
    accepted_pairs: int = 0
    below_min_true_selectivity: int = 0
    above_max_true_selectivity: int = 0
    estimated_prefilter_rejects: int = 0
    duplicate_pair: int = 0
    empty_rowset: int = 0
    no_subtype_unique_query_vector: int = 0
    rowset_errors: int = 0
    range_subtype_quota_rejects: int = 0


@dataclass
class CombinedRowSet:
    candidate_count: int
    sample_sources: list[np.ndarray]
    fallback_records: list[dict[str, Any]]


class LruRowCache:
    def __init__(self, max_entries: int):
        self.max_entries = max(0, int(max_entries))
        self.items: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> np.ndarray | None:
        value = self.items.get(key)
        if value is None:
            self.misses += 1
            return None
        self.items.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: np.ndarray) -> None:
        if self.max_entries <= 0:
            return
        self.items[key] = value
        self.items.move_to_end(key)
        while len(self.items) > self.max_entries:
            self.items.popitem(last=False)
            self.evictions += 1

    def summary(self) -> dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "entries": len(self.items),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


class RowSetProvider:
    def __init__(
        self,
        *,
        runtime_root: Path | None,
        scale: str,
        row_scope: int,
        token_postings_bin: Path,
        token_ngram_index: Path,
        row_cache_entries: int,
        range_cache_entries: int,
        load_range_values: bool,
        base_range_fbin: Path | None = None,
    ) -> None:
        self.runtime_root = runtime_root
        self.scale = scale
        self.scale_root = runtime_root / scale if runtime_root is not None else None
        self.row_scope = int(row_scope)
        self.token_lookup = TokenIdPostingsBinaryLookup(token_postings_bin, max_row_id_exclusive=row_scope)
        self.token_index = load_token_text_index_from_ngram_binary(token_ngram_index)
        self.range_fbin = resolve_metadata_path(runtime_root, scale, base_range_fbin,
            ("base_ranges_reordered.fbin", "base_ranges_aligned.fbin", "base_ranges.fbin"))
        rows, cols = read_fbin_header(self.range_fbin)
        if self.row_scope > rows:
            raise ValueError(f"row_scope {self.row_scope} exceeds range rows {rows} in {self.range_fbin}")
        range_values = np.memmap(
            self.range_fbin,
            dtype="<f4",
            mode="r",
            offset=8,
            shape=(rows, cols),
        )
        self.range_values_loaded = bool(load_range_values)
        if load_range_values:
            self.range_values = np.asarray(range_values[: self.row_scope], dtype=np.float32)
        else:
            self.range_values = range_values
        self.row_cache = LruRowCache(row_cache_entries)
        self.range_row_cache = LruRowCache(range_cache_entries)
        self.range_mask_cache = LruRowCache(range_cache_entries)
        self.fuzzy_expansion_cache: dict[str, np.ndarray] = {}

    def rows_for_record(self, record: dict[str, Any]) -> np.ndarray:
        atom = atom_family(record)
        key = str(record.get("predicate_key") or record.get("predicate_id") or record.get("query_id"))
        cache_key = f"{atom}:{key}"
        if atom == "range":
            cached = self.range_row_cache.get(cache_key)
        else:
            cached = self.row_cache.get(cache_key)
        if cached is not None:
            return cached

        if atom == "categorical":
            rows = self.rows_for_token_ids(predicate_token_ids(record))
        elif atom == "exact":
            rows = self.rows_for_token_ids(predicate_token_ids(record))
        elif atom == "fuzzy":
            rows = self.rows_for_fuzzy(record)
        elif atom == "range":
            rows = self.rows_for_range(record)
        else:
            raise ValueError(f"unsupported atom family {atom!r}")

        rows = filter_rows_to_scope(rows, self.row_scope)
        if atom == "range":
            self.range_row_cache.put(cache_key, rows)
        else:
            self.row_cache.put(cache_key, rows)
        return rows

    def count_for_record(self, record: dict[str, Any]) -> int:
        atom = atom_family(record)
        if atom == "range":
            return int(np.count_nonzero(self.range_mask_for_record(record)))
        return int(self.rows_for_record(record).size)

    def rows_for_token_ids(self, token_ids: Sequence[int]) -> np.ndarray:
        clean_ids = sorted({int(token_id) for token_id in token_ids if int(token_id) >= 0})
        if not clean_ids:
            return empty_rows()
        if len(clean_ids) == 1:
            token_rows = self.token_lookup.load_token_ids(clean_ids)
            rows = token_rows.get(clean_ids[0], empty_rows())
            return filter_rows_to_scope(rows, self.row_scope)
        rows = self.token_lookup.rows_for_token_ids_union(np.asarray(clean_ids, dtype=np.uint32))
        return filter_rows_to_scope(rows, self.row_scope)

    def rows_for_fuzzy(self, record: dict[str, Any]) -> np.ndarray:
        predicates = record.get("predicates") or []
        if not predicates:
            return empty_rows()
        predicate = predicates[0]
        mode = fuzzy_predicate_mode(predicate)
        pattern = fuzzy_predicate_pattern(predicate)
        expansion_key = f"{mode}:{pattern}"
        token_ids = self.fuzzy_expansion_cache.get(expansion_key)
        if token_ids is None:
            token_ids = fuzzy_matching_token_ids(mode, pattern, self.token_index)
            self.fuzzy_expansion_cache[expansion_key] = token_ids
        return self.rows_for_token_ids(token_ids.astype(np.uint32, copy=False).tolist())

    def rows_for_range(self, record: dict[str, Any]) -> np.ndarray:
        mask = self.range_mask_for_record(record)
        rows = np.flatnonzero(mask)
        return rows.astype(np.uint32, copy=False)

    def range_intersection(self, range_record: dict[str, Any], rows: np.ndarray) -> np.ndarray:
        rows = filter_rows_to_scope(rows, self.row_scope)
        if rows.size == 0:
            return empty_rows()
        range_mask = self.range_mask_for_record(range_record)
        return rows[range_mask[rows.astype(np.int64, copy=False)]].astype(np.uint32, copy=False)

    def range_mask_for_record(self, record: dict[str, Any]) -> np.ndarray:
        key = str(record.get("predicate_key") or record.get("predicate_id") or record.get("query_id"))
        cache_key = f"range-mask:{key}"
        cached = self.range_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        mask = self.range_mask(record)
        if self.row_scope < mask.size:
            mask = mask[: self.row_scope]
        mask = np.asarray(mask, dtype=bool)
        self.range_mask_cache.put(cache_key, mask)
        return mask

    def range_mask(self, record: dict[str, Any]) -> np.ndarray:
        values = self.range_values[: self.row_scope]
        return range_predicate_mask(values, record.get("predicates") or [])

    def summary(self) -> dict[str, Any]:
        token_summary = self.token_lookup.summary()
        return {
            "row_scope": self.row_scope,
            "range_fbin": str(self.range_fbin),
            "range_values_loaded": self.range_values_loaded,
            "token_postings": token_summary,
            "row_cache": self.row_cache.summary(),
            "range_row_cache": self.range_row_cache.summary(),
            "range_mask_cache": self.range_mask_cache.summary(),
            "fuzzy_expansions": len(self.fuzzy_expansion_cache),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path,
                        help="Parent of per-scale metadata directories; alternatively supply both explicit base files.")
    parser.add_argument("--base-spmat", type=Path, help="Aligned base label CSR for one scale.")
    parser.add_argument("--base-range-fbin", type=Path,
                        help="Aligned numeric matrix from metadata.py (width, height, similarity).")
    parser.add_argument("--atomic-root", type=Path, required=True)
    parser.add_argument("--string-root", type=Path, required=True)
    parser.add_argument(
        "--range-query-root",
        type=Path,
        default=None,
        help=(
            "Optional independent range source. Reads query_templates.jsonl or "
            "query_manifest.jsonl and replaces range records from --atomic-root."
        ),
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--scales", default="100M")
    parser.add_argument("--token-postings-bin", type=Path, default=DEFAULT_TOKEN_POSTINGS)
    parser.add_argument("--token-ngram-index", type=Path, default=DEFAULT_NGRAM_INDEX)
    parser.add_argument("--queries-per-hybrid-subtype", type=int, default=DEFAULT_QUERIES_PER_HYBRID_SUBTYPE)
    parser.add_argument("--min-and-selectivity", type=float, default=AND_MIN_SELECTIVITY)
    parser.add_argument("--min-or-selectivity", type=float, default=OR_MIN_SELECTIVITY)
    parser.add_argument("--max-selectivity", type=float, default=DEFAULT_MAX_SELECTIVITY)
    parser.add_argument("--max-candidate-pairs-per-subtype", type=int, default=0)
    parser.add_argument("--max-atomic-templates-per-family", type=int, default=0)
    parser.add_argument(
        "--range-subtypes",
        default="similarity,width_height",
        help="Comma-separated range template subtypes to include, e.g. width_height.",
    )
    parser.add_argument(
        "--require-range-pair",
        action="store_true",
        help="Generate only pair families that include a range atom.",
    )
    parser.add_argument(
        "--reuse-query-root",
        type=Path,
        default=None,
        help="Optional existing query-set root. Matching records are reused before generating top-up records.",
    )
    parser.add_argument(
        "--reuse-subtypes",
        default="",
        help=(
            "Optional comma-separated hybrid subtype allowlist for --reuse-query-root. "
            "Empty preserves the existing behavior of reusing every eligible subtype."
        ),
    )
    parser.add_argument("--row-limit", type=int, default=0)
    parser.add_argument("--row-cache-entries", type=int, default=256)
    parser.add_argument("--range-row-cache-entries", type=int, default=64)
    parser.add_argument(
        "--max-pairs-per-outer",
        type=int,
        default=128,
        help="Candidate diversity cap when grouping by an expensive outer atom.",
    )
    parser.add_argument(
        "--load-range-values",
        action="store_true",
        help="Load the range fbin slice into RAM instead of using memmap random reads.",
    )
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--estimate-prefilter-factor", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--topup-seed-offset",
        type=int,
        default=7919,
        help="Seed offset used for generated top-up candidates after reusing an existing query load.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    explicit = (args.base_spmat is not None, args.base_range_fbin is not None)
    if any(explicit) and not all(explicit):
        raise SystemExit("--base-spmat and --base-range-fbin must be supplied together")
    if not any(explicit) and args.runtime_root is None:
        raise SystemExit("supply --runtime-root or both --base-spmat and --base-range-fbin")
    if all(explicit) and len(parse_scales(args.scales)) != 1:
        raise SystemExit("explicit base files require exactly one --scales value")
    if args.queries_per_hybrid_subtype < 0:
        raise SystemExit("--queries-per-hybrid-subtype must be non-negative")
    if args.min_and_selectivity < 0 or args.min_or_selectivity < 0 or args.max_selectivity < 0:
        raise SystemExit("selectivity bounds must be non-negative")
    if args.min_and_selectivity > args.max_selectivity or args.min_or_selectivity > args.max_selectivity:
        raise SystemExit("min selectivity must be <= max selectivity")
    for name in (
        "max_candidate_pairs_per_subtype",
        "max_atomic_templates_per_family",
        "row_limit",
        "row_cache_entries",
        "range_row_cache_entries",
        "max_pairs_per_outer",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if args.progress_interval <= 0:
        raise SystemExit("--progress-interval must be positive")
    if args.estimate_prefilter_factor < 1.0:
        raise SystemExit("--estimate-prefilter-factor must be >= 1")
    range_subtypes = parse_csv_lower(args.range_subtypes)
    if not range_subtypes:
        raise SystemExit("--range-subtypes must contain at least one subtype")


def parse_csv_lower(raw: str) -> set[str]:
    return {item.strip().lower() for item in str(raw or "").split(",") if item.strip()}


def parse_scales(raw: str) -> list[str]:
    scales = [part.strip() for part in raw.split(",") if part.strip()]
    if not scales:
        raise SystemExit("At least one scale is required")
    return scales


def read_fbin_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=FBIN_HEADER_DTYPE, count=2)
    if header.size != 2:
        raise ValueError(f"invalid fbin header in {path}")
    return int(header[0]), int(header[1])


def resolve_metadata_path(runtime_root: Path | None, scale: str,
                          explicit: Path | None, names: tuple[str, ...]) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    if runtime_root is None:
        raise ValueError("runtime root or explicit metadata file is required")
    for name in names:
        path = runtime_root / scale / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"expected one of {names} under {runtime_root / scale}")


def read_spmat_rows(runtime_root: Path | None, scale: str, base_spmat: Path | None = None) -> int:
    path = resolve_metadata_path(runtime_root, scale, base_spmat,
        ("base_duckdb_reordered.spmat", "base_duckdb_aligned.spmat", "base.spmat"))
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.dtype("<i8"), count=3)
    if header.size != 3:
        raise ValueError(f"invalid spmat header in {path}")
    return int(header[0])


def effective_row_scope(runtime_root: Path | None, scale: str, row_limit: int,
                        base_spmat: Path | None = None) -> int:
    spmat_rows = read_spmat_rows(runtime_root, scale, base_spmat)
    rows = spmat_rows if row_limit <= 0 else min(spmat_rows, int(row_limit))
    if rows <= 0:
        raise ValueError("effective row scope must be positive")
    return rows


def scale_dir(root: Path, scale: str) -> Path:
    filenames = ("query_templates.jsonl", "query_manifest.jsonl")
    if any((root / name).exists() for name in filenames):
        return root
    return root / scale


def resolve_query_record_path(path: Path, scale: str) -> Path:
    root = scale_dir(path, scale)
    for name in ("query_templates.jsonl", "query_manifest.jsonl"):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no query_templates.jsonl or query_manifest.jsonl found under {root}"
    )


def load_query_records(path: Path, scale: str) -> list[dict[str, Any]]:
    record_path = resolve_query_record_path(path, scale)
    records = []
    with record_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("predicate_id", record.get("query_id"))
            record.setdefault("predicate_key", stable_predicate_key(record))
            record.setdefault("scale", scale)
            records.append(record)
    return records


def load_templates(path: Path, scale: str) -> list[dict[str, Any]]:
    return load_query_records(path, scale)


def group_atomic_templates(
    atomic_records: list[dict[str, Any]],
    string_records: list[dict[str, Any]],
    cap: int,
    seed: int,
    range_subtypes: set[str],
    range_records: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in atomic_records:
        family = str(record.get("family"))
        subtype = str(record.get("subtype"))
        if family == "categorical" and subtype == "single":
            grouped["categorical"].append(record)
        elif (
            range_records is None
            and family == "range"
            and subtype.lower() in range_subtypes
        ):
            grouped["range"].append(record)
    for record in range_records or []:
        family = str(record.get("family"))
        subtype = str(record.get("subtype"))
        if family == "range" and subtype.lower() in range_subtypes:
            grouped["range"].append(record)
    for record in string_records:
        family = str(record.get("family"))
        subtype = str(record.get("subtype"))
        if family == "string" and subtype == "exact":
            grouped["exact"].append(record)
        elif family == "string" and subtype == "fuzzy":
            grouped["fuzzy"].append(record)

    if cap > 0:
        grouped = {
            family: deterministic_cap(records, cap, seed, family)
            for family, records in grouped.items()
        }
    return grouped


def deterministic_cap(records: list[dict[str, Any]], cap: int, seed: int, namespace: str) -> list[dict[str, Any]]:
    if cap <= 0 or len(records) <= cap:
        return list(records)
    ordered = sorted(
        records,
        key=lambda record: stable_unit(seed, namespace, record.get("predicate_id") or record.get("query_id")),
    )
    return ordered[:cap]


def atom_label(atom: str) -> str:
    if atom == "categorical":
        return "categorical.single"
    if atom == "exact":
        return "string.exact"
    if atom == "fuzzy":
        return "string.fuzzy"
    return atom


def atom_family(record: dict[str, Any]) -> str:
    family = str(record.get("family", "")).lower()
    subtype = str(record.get("subtype", "")).lower()
    if family == "categorical" and subtype == "single":
        return "categorical"
    if family == "range":
        return "range"
    if family == "string" and subtype == "exact":
        return "exact"
    if family == "string" and subtype == "fuzzy":
        return "fuzzy"
    raise ValueError(f"unsupported atomic record family/subtype: {family}.{subtype}")


def predicate_token_ids(record: dict[str, Any]) -> list[int]:
    token_ids = []
    for predicate in record.get("predicates") or []:
        token_id = predicate.get("token_id", predicate.get("source_token_id"))
        if token_id is None:
            continue
        token_ids.append(int(token_id))
    return sorted(set(token_ids))


def empty_rows() -> np.ndarray:
    return np.empty((0,), dtype=np.uint32)


def normalize_rows(rows: np.ndarray, row_scope: int) -> np.ndarray:
    if rows.size == 0:
        return empty_rows()
    rows = np.asarray(rows, dtype=np.uint32)
    if row_scope > 0:
        rows = rows[rows < np.uint32(row_scope)]
    if rows.size > 1:
        rows = np.unique(rows)
    return rows.astype(np.uint32, copy=False)


def filter_rows_to_scope(rows: np.ndarray, row_scope: int) -> np.ndarray:
    if rows.size == 0:
        return empty_rows()
    rows = np.asarray(rows, dtype=np.uint32)
    if row_scope > 0:
        rows = rows[rows < np.uint32(row_scope)]
    return rows.astype(np.uint32, copy=False)


def intersect_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return empty_rows()
    return np.intersect1d(left, right, assume_unique=True).astype(np.uint32, copy=False)


def union_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0:
        return right.astype(np.uint32, copy=False)
    if right.size == 0:
        return left.astype(np.uint32, copy=False)
    return np.union1d(left, right).astype(np.uint32, copy=False)


def combine_rows(
    left_record: dict[str, Any],
    right_record: dict[str, Any],
    logic: str,
    provider: RowSetProvider,
) -> np.ndarray:
    left_atom = atom_family(left_record)
    right_atom = atom_family(right_record)
    if left_atom == "range" and right_atom == "range":
        raise ValueError("range x range is not a supported pair")

    if left_atom == "range" or right_atom == "range":
        range_record = left_record if left_atom == "range" else right_record
        other_record = right_record if left_atom == "range" else left_record
        other_rows = provider.rows_for_record(other_record)
        intersection = provider.range_intersection(range_record, other_rows)
        if logic == "and":
            return intersection
        range_rows = provider.rows_for_record(range_record)
        return union_rows(range_rows, other_rows)

    left_rows = provider.rows_for_record(left_record)
    right_rows = provider.rows_for_record(right_record)
    if logic == "and":
        return intersect_rows(left_rows, right_rows)
    return union_rows(left_rows, right_rows)


def combine_row_info(
    left_record: dict[str, Any],
    right_record: dict[str, Any],
    logic: str,
    provider: RowSetProvider,
) -> CombinedRowSet:
    if logic == "and":
        rows = combine_rows(left_record, right_record, logic, provider)
        return CombinedRowSet(candidate_count=int(rows.size), sample_sources=[rows], fallback_records=[])

    left_atom = atom_family(left_record)
    right_atom = atom_family(right_record)
    if left_atom == "range" or right_atom == "range":
        range_record = left_record if left_atom == "range" else right_record
        other_record = right_record if left_atom == "range" else left_record
        other_rows = provider.rows_for_record(other_record)
        range_count = atomic_candidate_count(range_record, provider.row_scope)
        intersection = provider.range_intersection(range_record, other_rows)
        candidate_count = range_count + int(other_rows.size) - int(intersection.size)
        return CombinedRowSet(
            candidate_count=int(candidate_count),
            sample_sources=[other_rows],
            fallback_records=[range_record],
        )

    left_rows = provider.rows_for_record(left_record)
    right_rows = provider.rows_for_record(right_record)
    intersection = intersect_rows(left_rows, right_rows)
    candidate_count = int(left_rows.size) + int(right_rows.size) - int(intersection.size)
    return CombinedRowSet(
        candidate_count=int(candidate_count),
        sample_sources=[left_rows, right_rows],
        fallback_records=[],
    )


def atomic_candidate_count(record: dict[str, Any], row_scope: int) -> int:
    raw = record.get("candidate_count")
    if raw is not None:
        try:
            count = int(raw)
            if row_scope > 0:
                return min(count, int(row_scope))
            return count
        except (TypeError, ValueError):
            pass
    selectivity = float(record.get("selectivity") or 0.0)
    return int(round(selectivity * row_scope))


def estimate_selectivity(left_record: dict[str, Any], right_record: dict[str, Any], logic: str) -> float:
    left = float(left_record.get("selectivity") or 0.0)
    right = float(right_record.get("selectivity") or 0.0)
    if logic == "and":
        return left * right
    return left + right - left * right


def pair_attempt_limit(args: argparse.Namespace, left_count: int, right_count: int) -> int:
    total = left_count * right_count
    if args.max_candidate_pairs_per_subtype > 0:
        return min(total, int(args.max_candidate_pairs_per_subtype))
    derived = max(args.queries_per_hybrid_subtype * 300, args.queries_per_hybrid_subtype)
    return min(total, derived)


def iter_candidate_pairs(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    max_attempts: int,
    seed: int,
    label: str,
    max_pairs_per_outer: int = 0,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    if not left_records or not right_records or max_attempts <= 0:
        return

    left_count = len(left_records)
    right_count = len(right_records)
    left_atom = atom_family(left_records[0])
    right_atom = atom_family(right_records[0])
    if "range" in {left_atom, right_atom}:
        if right_atom == "range":
            outer_side = "right"
            outer_records = right_records
            inner_records = left_records
        else:
            outer_side = "left"
            outer_records = left_records
            inner_records = right_records
        for outer_record, inner_record in iter_grouped_candidate_pairs(
            outer_records,
            inner_records,
            max_attempts,
            seed,
            label,
            max_pairs_per_outer,
        ):
            if outer_side == "left":
                yield outer_record, inner_record
            else:
                yield inner_record, outer_record
        return

    yield from iter_grouped_candidate_pairs(
        left_records,
        right_records,
        max_attempts,
        seed,
        label,
        max_pairs_per_outer,
    )


def iter_grouped_candidate_pairs(
    outer_records: list[dict[str, Any]],
    inner_records: list[dict[str, Any]],
    max_attempts: int,
    seed: int,
    label: str,
    max_pairs_per_outer: int = 0,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    outer_count = len(outer_records)
    inner_count = len(inner_records)
    if not outer_count or not inner_count:
        return
    outer_order = deterministic_order(outer_records, seed, label, "outer")
    step = max(1, int(stable_unit(seed, label, "inner-step") * inner_count))
    while math.gcd(step, inner_count) != 1:
        step += 1
        if step >= inner_count:
            step = 1
            break
    per_outer = inner_count if max_pairs_per_outer <= 0 else min(inner_count, int(max_pairs_per_outer))
    attempts = 0
    outer_cycle = 0
    while attempts < max_attempts:
        made_progress = False
        for outer_pos, outer_index in enumerate(outer_order):
            outer_record = outer_records[outer_index]
            base = int(
                stable_unit(seed, label, "inner-offset", outer_record.get("predicate_id"), outer_cycle) * inner_count
            )
            for inner_rank in range(per_outer):
                if attempts >= max_attempts:
                    return
                inner_index = (base + inner_rank * step) % inner_count
                yield outer_record, inner_records[inner_index]
                attempts += 1
                made_progress = True
        outer_cycle += 1
        if not made_progress:
            return


def deterministic_order(records: list[dict[str, Any]], seed: int, label: str, namespace: str) -> list[int]:
    return sorted(
        range(len(records)),
        key=lambda idx: stable_unit(seed, label, namespace, records[idx].get("predicate_id") or records[idx].get("query_id")),
    )


def selectivity_bounds(spec: PairSpec, args: argparse.Namespace) -> tuple[float, float]:
    if spec.logic == "and":
        return float(args.min_and_selectivity), float(args.max_selectivity)
    return float(args.min_or_selectivity), float(args.max_selectivity)


def choose_query_vector(rows: np.ndarray, used_rows: set[int], seed: int, predicate_id: str) -> int | None:
    if rows.size == 0:
        return None
    start = int(stable_unit(seed, "query-vector", predicate_id) * rows.size)
    for offset in range(rows.size):
        row = int(rows[(start + offset) % rows.size])
        if row not in used_rows:
            return row
    return None


def choose_query_vector_from_info(
    info: CombinedRowSet,
    provider: RowSetProvider,
    used_rows: set[int],
    seed: int,
    predicate_id: str,
) -> int | None:
    for index, rows in enumerate(info.sample_sources):
        row = choose_query_vector(rows, used_rows, seed, f"{predicate_id}:source:{index}")
        if row is not None:
            return row
    for index, record in enumerate(info.fallback_records):
        rows = provider.rows_for_record(record)
        row = choose_query_vector(rows, used_rows, seed, f"{predicate_id}:fallback:{index}")
        if row is not None:
            return row
    return None


def build_record(
    *,
    scale: str,
    spec: PairSpec,
    left_record: dict[str, Any],
    right_record: dict[str, Any],
    rank: int,
    query_vector_id: int,
    candidate_count: int,
    row_scope: int,
    estimated_selectivity: float,
    reused_global: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predicate_id = hybrid_predicate_id(scale, spec, left_record, right_record)
    op = spec.logic
    predicate_tree = {"op": op, "children": [atomic_child(left_record), atomic_child(right_record)]}
    true_selectivity = int(candidate_count) / float(row_scope)
    label = f"{left_record.get('label', left_record.get('query_id'))}__{op}__{right_record.get('label', right_record.get('query_id'))}"
    record = {
        "query_id": f"{predicate_id}-inst-{rank:06d}",
        "predicate_id": predicate_id,
        "predicate_key": "",
        "predicate_instance_rank": rank,
        "predicate_reused": False,
        "predicate_logic": op.upper(),
        "predicate": predicate_tree,
        "predicates": [predicate_tree],
        "scale": scale,
        "family": "hybrid",
        "subtype": spec.label,
        "hybrid_subtype": spec.label,
        "hybrid_top_logic": op,
        "hybrid_atom_families": [spec.left, spec.right],
        "hybrid_atom_types": [spec.left_label, spec.right_label],
        "label": label,
        "is_query_instance": True,
        "query_vector_id": int(query_vector_id),
        "anchor_row_id": int(query_vector_id),
        "anchor_sample_id": None,
        "anchor_source": "true_pairwise_rowset",
        "anchor_sample_method": f"hybrid_true_{op}_rowset",
        "anchor_reused": bool(reused_global),
        "query_vector_reused_across_predicates": bool(reused_global),
        "query_vector_reused_within_subtype": False,
        "selectivity": true_selectivity,
        "true_selectivity": true_selectivity,
        "candidate_count": int(candidate_count),
        "selectivity_band": selectivity_band(true_selectivity),
        "difficulty_band": difficulty_band(true_selectivity),
        "stats": {
            "estimated_selectivity": float(estimated_selectivity),
            "true_selectivity": true_selectivity,
            "estimated_rows": int(round(estimated_selectivity * row_scope)),
            "true_rows": int(candidate_count),
            "selectivity_source": "true_pairwise_rowset",
            "estimate_method": "atomic_selectivity_independence",
            "is_estimate": False,
            "row_scope": int(row_scope),
            "left_predicate_id": left_record.get("predicate_id"),
            "right_predicate_id": right_record.get("predicate_id"),
            "left_candidate_count": int(left_record.get("candidate_count") or 0),
            "right_candidate_count": int(right_record.get("candidate_count") or 0),
        },
        "notes": [
            "pairwise hybrid instance generated from existing atomic predicate templates",
            "query vector sampled from the true combined row set",
            "string.exact x string.fuzzy is intentionally excluded from this workload",
        ],
    }
    record["predicate_key"] = stable_predicate_key(record)
    posting = {
        "query_id": record["query_id"],
        "rank": 0,
        "row_id": int(query_vector_id),
        "sample_id": "",
        "similarity": "",
        "width": "",
        "height": "",
        "source": record["anchor_source"],
        "sample_method": record["anchor_sample_method"],
    }
    return record, posting


def atomic_child(record: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "query_id",
        "predicate_id",
        "predicate_key",
        "family",
        "subtype",
        "label",
        "predicate_logic",
        "predicates",
        "selectivity",
        "candidate_count",
        "selectivity_band",
        "difficulty_band",
        "stats",
    )
    child = {field: record[field] for field in fields if field in record}
    child["type"] = f"{record.get('family')}.{record.get('subtype')}"
    return child


def load_reusable_records(
    *,
    reuse_root: Path | None,
    scale: str,
    specs: list[PairSpec],
    target_per_subtype: int,
    range_subtypes: set[str],
    reuse_subtypes: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, Any]]:
    reusable: dict[str, list[dict[str, Any]]] = defaultdict(list)
    postings_by_query_id: dict[str, dict[str, Any]] = {}
    if reuse_root is None:
        return reusable, postings_by_query_id, {"enabled": False}

    root = scale_dir(reuse_root, scale)
    manifest_path = root / "query_manifest.jsonl"
    postings_path = root / "query_postings.jsonl"
    allowed_subtypes = {spec.label for spec in specs}
    if reuse_subtypes:
        allowed_subtypes &= reuse_subtypes
    seen_query_ids: set[str] = set()
    scanned = 0
    matched = 0

    if not manifest_path.exists():
        raise FileNotFoundError(f"reuse manifest not found: {manifest_path}")
    if postings_path.exists():
        with postings_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                posting = json.loads(line)
                query_id = str(posting.get("query_id"))
                postings_by_query_id.setdefault(query_id, posting)

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            scanned += 1
            record = json.loads(line)
            subtype = str(record.get("subtype"))
            if subtype not in allowed_subtypes:
                continue
            if not record_range_subtypes(record) <= range_subtypes:
                continue
            if len(reusable[subtype]) >= target_per_subtype:
                continue
            query_id = str(record.get("query_id"))
            if query_id in seen_query_ids:
                continue
            seen_query_ids.add(query_id)
            reusable[subtype].append(mark_reused_record(record, len(reusable[subtype])))
            matched += 1

    summary = {
        "enabled": True,
        "root": str(reuse_root),
        "manifest": str(manifest_path),
        "postings": str(postings_path) if postings_path.exists() else None,
        "scanned": scanned,
        "matched": matched,
        "subtype_allowlist": sorted(reuse_subtypes),
        "by_subtype": {subtype: len(records) for subtype, records in sorted(reusable.items())},
    }
    return reusable, postings_by_query_id, summary


def record_range_subtypes(record: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    for child in (record.get("predicate") or {}).get("children") or []:
        if str(child.get("family")) == "range":
            subtype = child.get("subtype")
            if subtype is not None:
                found.add(str(subtype).lower())
    return found


def range_subtype_for_record(record: dict[str, Any]) -> str | None:
    if str(record.get("family", "")).lower() == "range":
        subtype = record.get("subtype")
        return None if subtype is None else str(subtype).lower()
    subtypes = record_range_subtypes(record)
    if not subtypes:
        return None
    if len(subtypes) != 1:
        raise ValueError(f"expected one range subtype, found {sorted(subtypes)}")
    return next(iter(subtypes))


def proportional_range_subtype_targets(
    range_records: list[dict[str, Any]],
    target: int,
) -> dict[str, int]:
    counts = Counter(
        subtype
        for record in range_records
        if (subtype := range_subtype_for_record(record)) is not None
    )
    total = sum(counts.values())
    if total <= 0:
        return {}

    exact = {subtype: target * count / total for subtype, count in counts.items()}
    targets = {subtype: int(math.floor(value)) for subtype, value in exact.items()}
    remainder = int(target) - sum(targets.values())
    order = sorted(counts, key=lambda subtype: (-(exact[subtype] - targets[subtype]), subtype))
    for subtype in order[:remainder]:
        targets[subtype] += 1
    return dict(sorted(targets.items()))


def mark_reused_record(record: dict[str, Any], rank: int) -> dict[str, Any]:
    reused = dict(record)
    reused["predicate_instance_rank"] = int(rank)
    reused.setdefault("notes", [])
    if isinstance(reused["notes"], list):
        reused["notes"] = [*reused["notes"], "reused from existing hybrid query load"]
    reused["reused_from_existing_hybrid_load"] = True
    reused["anchor_sample_method"] = str(reused.get("anchor_sample_method") or "") + "_reused"
    return reused


def posting_for_record(record: dict[str, Any], postings_by_query_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    query_id = str(record.get("query_id"))
    posting = postings_by_query_id.get(query_id)
    if posting is not None:
        return posting
    return {
        "query_id": query_id,
        "rank": 0,
        "row_id": int(record["query_vector_id"]),
        "sample_id": "",
        "similarity": "",
        "width": "",
        "height": "",
        "source": record.get("anchor_source", "reused_existing_hybrid_load"),
        "sample_method": record.get("anchor_sample_method", "reused_existing_hybrid_load"),
    }


def generate_subtype(
    *,
    spec: PairSpec,
    records_by_family: dict[str, list[dict[str, Any]]],
    provider: RowSetProvider,
    row_scope: int,
    args: argparse.Namespace,
    used_rows_global: set[int],
    reusable_records: list[dict[str, Any]],
    reusable_postings_by_query_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    left_records = records_by_family.get(spec.left, [])
    right_records = records_by_family.get(spec.right, [])
    target = int(args.queries_per_hybrid_subtype)
    max_attempts = pair_attempt_limit(args, len(left_records), len(right_records))
    min_sel, max_sel = selectivity_bounds(spec, args)
    range_targets = (
        proportional_range_subtype_targets(records_by_family.get("range", []), target)
        if "range" in {spec.left, spec.right}
        else {}
    )
    range_emitted: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    reusable_quota_rejects = 0
    for record in reusable_records:
        if len(records) >= target:
            break
        range_subtype = range_subtype_for_record(record)
        if range_targets:
            if range_subtype not in range_targets:
                reusable_quota_rejects += 1
                continue
            if range_emitted[range_subtype] >= range_targets[range_subtype]:
                reusable_quota_rejects += 1
                continue
        records.append(record)
        if range_subtype is not None:
            range_emitted[range_subtype] += 1
    postings: list[dict[str, Any]] = [
        posting_for_record(record, reusable_postings_by_query_id)
        for record in records
    ]
    stats = GenerationStats()
    seen_pairs: set[tuple[str, str]] = {reused_pair_key(record) for record in records}
    used_rows_subtype: set[int] = {int(record["query_vector_id"]) for record in records}
    for row_id in used_rows_subtype:
        used_rows_global.add(row_id)

    print(
        f"[pairwise] {spec.label}: left={len(left_records)} right={len(right_records)} "
        f"target={target} reused={len(records)} attempts={max_attempts} selectivity=[{min_sel},{max_sel}]",
        flush=True,
    )

    candidate_seed = args.seed + (int(args.topup_seed_offset) if reusable_records else 0)
    for left_record, right_record in iter_candidate_pairs(
        left_records,
        right_records,
        max_attempts,
        candidate_seed,
        spec.label,
        args.max_pairs_per_outer,
    ):
        if len(records) >= target:
            break
        stats.attempted_pairs += 1
        if stats.attempted_pairs % int(args.progress_interval) == 0:
            print(
                f"[pairwise] {spec.label}: attempted={stats.attempted_pairs} "
                f"accepted={len(records)}/{target}",
                flush=True,
            )
        pair_key = (str(left_record.get("predicate_id")), str(right_record.get("predicate_id")))
        if pair_key in seen_pairs:
            stats.duplicate_pair += 1
            continue
        seen_pairs.add(pair_key)

        range_record = None
        if spec.left == "range":
            range_record = left_record
        elif spec.right == "range":
            range_record = right_record
        range_subtype = (
            None if range_record is None else range_subtype_for_record(range_record)
        )
        if range_targets:
            if range_subtype not in range_targets:
                stats.rowset_errors += 1
                continue
            if range_emitted[range_subtype] >= range_targets[range_subtype]:
                stats.range_subtype_quota_rejects += 1
                continue

        estimated = estimate_selectivity(left_record, right_record, spec.logic)
        if estimated > max_sel * float(args.estimate_prefilter_factor):
            stats.estimated_prefilter_rejects += 1
            continue

        try:
            combined = combine_row_info(left_record, right_record, spec.logic, provider)
        except Exception as exc:  # noqa: BLE001 - emit summary context and continue other pairs.
            stats.rowset_errors += 1
            if stats.rowset_errors <= 5:
                print(f"[pairwise] {spec.label}: rowset error: {exc}", flush=True)
            continue
        if combined.candidate_count <= 0:
            stats.empty_rowset += 1
            continue
        true_selectivity = combined.candidate_count / float(row_scope)
        if true_selectivity < min_sel:
            stats.below_min_true_selectivity += 1
            continue
        if true_selectivity > max_sel:
            stats.above_max_true_selectivity += 1
            continue

        predicate_id = hybrid_predicate_id(provider.scale, spec, left_record, right_record)
        query_vector_id = choose_query_vector_from_info(combined, provider, used_rows_subtype, args.seed, predicate_id)
        if query_vector_id is None:
            stats.no_subtype_unique_query_vector += 1
            continue
        used_rows_subtype.add(query_vector_id)
        record, posting = build_record(
            scale=provider.scale,
            spec=spec,
            left_record=left_record,
            right_record=right_record,
            rank=len(records),
            query_vector_id=query_vector_id,
            candidate_count=combined.candidate_count,
            row_scope=row_scope,
            estimated_selectivity=estimated,
            reused_global=query_vector_id in used_rows_global,
        )
        records.append(record)
        postings.append(posting)
        if range_subtype is not None:
            range_emitted[range_subtype] += 1
        used_rows_global.add(query_vector_id)
        stats.accepted_pairs += 1

    summary = {
        "requested": target,
        "emitted": len(records),
        "reused": len(records) - stats.accepted_pairs,
        "generated": stats.accepted_pairs,
        "reusable_range_quota_rejects": reusable_quota_rejects,
        "shortfall": max(0, target - len(records)),
        "attempted_pairs": stats.attempted_pairs,
        "accepted_pairs": stats.accepted_pairs,
        "below_min_true_selectivity": stats.below_min_true_selectivity,
        "above_max_true_selectivity": stats.above_max_true_selectivity,
        "estimated_prefilter_rejects": stats.estimated_prefilter_rejects,
        "duplicate_pair": stats.duplicate_pair,
        "empty_rowset": stats.empty_rowset,
        "no_subtype_unique_query_vector": stats.no_subtype_unique_query_vector,
        "rowset_errors": stats.rowset_errors,
        "range_subtype_quota_rejects": stats.range_subtype_quota_rejects,
        "range_subtype_targets": range_targets,
        "range_subtype_emitted": dict(sorted(range_emitted.items())),
        "candidate_pair_attempt_limit": max_attempts,
        "selectivity_min": min_sel,
        "selectivity_max": max_sel,
        "atomic_templates": {
            spec.left_label: len(left_records),
            spec.right_label: len(right_records),
        },
        "unique_query_vector_ids": len(used_rows_subtype),
    }
    return records, postings, summary


def reused_pair_key(record: dict[str, Any]) -> tuple[str, str]:
    stats = record.get("stats") or {}
    left = stats.get("left_predicate_id")
    right = stats.get("right_predicate_id")
    if left is not None and right is not None:
        return str(left), str(right)
    children = (record.get("predicate") or {}).get("children") or []
    if len(children) >= 2:
        return (
            str(children[0].get("predicate_id") or children[0].get("query_id")),
            str(children[1].get("predicate_id") or children[1].get("query_id")),
        )
    return str(record.get("predicate_id")), str(record.get("query_id"))


def hybrid_predicate_id(
    scale: str,
    spec: PairSpec,
    left_record: dict[str, Any],
    right_record: dict[str, Any],
) -> str:
    return (
        f"laion-{sanitize_id(scale)}-hybrid-{sanitize_id(spec.label)}-"
        f"{sanitize_id(str(left_record.get('predicate_id') or left_record.get('query_id')))}-"
        f"{sanitize_id(str(right_record.get('predicate_id') or right_record.get('query_id')))}"
    )


def write_query_files(out_dir: Path, records: list[dict[str, Any]], postings: list[dict[str, Any]]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    manifest_path = out_dir / "query_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")
    paths["query_manifest_jsonl"] = str(manifest_path)

    hybrid_path = out_dir / "query_manifest_hybrid.jsonl"
    with hybrid_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")
    paths["query_manifest_hybrid_jsonl"] = str(hybrid_path)

    csv_path = out_dir / "query_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "query_id",
            "predicate_id",
            "predicate_instance_rank",
            "query_vector_id",
            "scale",
            "family",
            "subtype",
            "hybrid_top_logic",
            "hybrid_atom_types",
            "selectivity",
            "candidate_count",
            "selectivity_band",
            "difficulty_band",
            "anchor_sample_method",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "query_id": record["query_id"],
                    "predicate_id": record["predicate_id"],
                    "predicate_instance_rank": record["predicate_instance_rank"],
                    "query_vector_id": record["query_vector_id"],
                    "scale": record["scale"],
                    "family": record["family"],
                    "subtype": record["subtype"],
                    "hybrid_top_logic": record["hybrid_top_logic"],
                    "hybrid_atom_types": ",".join(record["hybrid_atom_types"]),
                    "selectivity": record["selectivity"],
                    "candidate_count": record["candidate_count"],
                    "selectivity_band": record["selectivity_band"],
                    "difficulty_band": record["difficulty_band"],
                    "anchor_sample_method": record["anchor_sample_method"],
                }
            )
    paths["query_manifest_csv"] = str(csv_path)

    postings_path = out_dir / "query_postings.jsonl"
    with postings_path.open("w", encoding="utf-8") as handle:
        for posting in postings:
            handle.write(json.dumps(to_jsonable(posting), sort_keys=True) + "\n")
    paths["query_postings_jsonl"] = str(postings_path)

    postings_csv_path = out_dir / "query_postings.csv"
    with postings_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["query_id", "rank", "row_id", "sample_id", "similarity", "width", "height", "source", "sample_method"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for posting in postings:
            writer.writerow({field: posting.get(field, "") for field in fieldnames})
    paths["query_postings_csv"] = str(postings_csv_path)
    return paths


def pair_specs(args: argparse.Namespace) -> list[PairSpec]:
    specs = []
    for left, right in PAIR_FAMILIES:
        if args.require_range_pair and "range" not in {left, right}:
            continue
        for logic in TOP_LOGICS:
            specs.append(PairSpec(left=left, right=right, logic=logic))
    return specs


def generate_for_scale(args: argparse.Namespace, scale: str) -> dict[str, Any]:
    started = time.time()
    out_dir = args.out_root / scale
    ensure_output_dir(out_dir, args.overwrite)
    row_scope = effective_row_scope(args.runtime_root, scale, args.row_limit, args.base_spmat)
    token_postings_bin = args.token_postings_bin
    atomic_records = load_templates(args.atomic_root, scale)
    string_records = load_templates(args.string_root, scale)
    range_records = None
    range_source_root = args.atomic_root
    if args.range_query_root is not None:
        range_source_root = args.range_query_root
        range_records = load_query_records(args.range_query_root, scale)
    records_by_family = group_atomic_templates(
        atomic_records,
        string_records,
        cap=args.max_atomic_templates_per_family,
        seed=args.seed,
        range_subtypes=parse_csv_lower(args.range_subtypes),
        range_records=range_records,
    )
    specs = pair_specs(args)
    reuse_subtypes = parse_csv_lower(args.reuse_subtypes)
    unknown_reuse_subtypes = reuse_subtypes - {spec.label for spec in specs}
    if unknown_reuse_subtypes:
        raise SystemExit(
            f"--reuse-subtypes contains unknown hybrid subtypes: {sorted(unknown_reuse_subtypes)}"
        )
    required_families = set()
    for spec in specs:
        required_families.add(spec.left)
        required_families.add(spec.right)
    missing = [family for family in ATOM_ORDER if family in required_families and not records_by_family.get(family)]
    if missing:
        raise SystemExit(f"Missing atomic template families: {missing}")

    reusable_by_subtype, reusable_postings, reuse_summary = load_reusable_records(
        reuse_root=args.reuse_query_root,
        scale=scale,
        specs=specs,
        target_per_subtype=int(args.queries_per_hybrid_subtype),
        range_subtypes=parse_csv_lower(args.range_subtypes),
        reuse_subtypes=reuse_subtypes,
    )

    provider = RowSetProvider(
        runtime_root=args.runtime_root,
        scale=scale,
        row_scope=row_scope,
        token_postings_bin=token_postings_bin,
        token_ngram_index=args.token_ngram_index,
        row_cache_entries=args.row_cache_entries,
        range_cache_entries=args.range_row_cache_entries,
        load_range_values=args.load_range_values,
        base_range_fbin=args.base_range_fbin,
    )

    all_records: list[dict[str, Any]] = []
    all_postings: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    used_rows_global: set[int] = set()
    for spec in specs:
        records, postings, summary = generate_subtype(
            spec=spec,
            records_by_family=records_by_family,
            provider=provider,
            row_scope=row_scope,
            args=args,
            used_rows_global=used_rows_global,
            reusable_records=reusable_by_subtype.get(spec.label, []),
            reusable_postings_by_query_id=reusable_postings,
        )
        all_records.extend(records)
        all_postings.extend(postings)
        summaries[spec.label] = summary
        print(
            f"[pairwise] {spec.label}: emitted={summary['emitted']}/{summary['requested']} "
            f"attempted={summary['attempted_pairs']}",
            flush=True,
        )

    output_paths = write_query_files(out_dir, all_records, all_postings)
    schema_path = out_dir / "query_schema.json"
    write_json(schema_path, query_schema())
    output_paths["query_schema_json"] = str(schema_path)

    summary = {
        "scale": scale,
        "stage": "pairwise_hybrid_instances",
        "row_scope": int(row_scope),
        "atomic_roots": {
            "categorical_range": str(args.atomic_root),
            "string": str(args.string_root),
            "range": str(range_source_root),
        },
        "atomic_source_files": {
            "categorical_range": str(resolve_query_record_path(args.atomic_root, scale)),
            "string": str(resolve_query_record_path(args.string_root, scale)),
            "range": str(resolve_query_record_path(range_source_root, scale)),
        },
        "query_counts": query_counts(all_records),
        "hybrid_target": {
            "queries_per_hybrid_subtype": int(args.queries_per_hybrid_subtype),
            "and_selectivity": [float(args.min_and_selectivity), float(args.max_selectivity)],
            "or_selectivity": [float(args.min_or_selectivity), float(args.max_selectivity)],
            "subtypes": [spec.label for spec in specs],
            "requested_by_subtype": {label: int(item["requested"]) for label, item in summaries.items()},
            "emitted_by_subtype": {label: int(item["emitted"]) for label, item in summaries.items()},
            "reused_by_subtype": {label: int(item["reused"]) for label, item in summaries.items()},
            "generated_by_subtype": {label: int(item["generated"]) for label, item in summaries.items()},
            "shortfall_by_subtype": {label: int(item["shortfall"]) for label, item in summaries.items()},
            "shortfall_total": sum(int(item["shortfall"]) for item in summaries.values()),
        },
        "atomic_template_counts": {family: len(records) for family, records in sorted(records_by_family.items())},
        "range_atomic_subtype_counts": dict(
            sorted(
                Counter(
                    subtype
                    for record in records_by_family.get("range", [])
                    if (subtype := range_subtype_for_record(record)) is not None
                ).items()
            )
        ),
        "range_subtypes": sorted(parse_csv_lower(args.range_subtypes)),
        "pair_specs": [spec.label for spec in specs],
        "reuse": reuse_summary,
        "query_instances": {
            "enabled": True,
            "query_instance_count": len(all_records),
            "unique_query_vector_ids_total": len({int(record["query_vector_id"]) for record in all_records}),
            "query_vector_reused_across_predicates_total": sum(
                1 for record in all_records if record.get("query_vector_reused_across_predicates")
            ),
            "query_vector_reused_within_subtype_total": sum(
                1 for record in all_records if record.get("query_vector_reused_within_subtype")
            ),
            "subtype_query_vector_duplicate_total": duplicate_subtype_row_count(all_records),
            "predicate_query_vector_duplicate_total": duplicate_pair_count(all_records),
            "query_vector_reuse_policy": "query_vector_id is unique within each hybrid subtype",
        },
        "rowset_provider": provider.summary(),
        "subtype_generation": summaries,
        "outputs": output_paths,
        "timings_sec": {
            "total": time.time() - started,
        },
        "notes": [
            "categorical atoms use categorical.single only",
            "string.exact x string.fuzzy is intentionally excluded",
            "candidate_count is the true combined row-set cardinality within row_scope",
            "query_postings.* records only the selected query-vector anchor row",
        ],
    }
    summary_path = out_dir / "query_summary.json"
    write_json(summary_path, summary)
    output_paths["query_summary_json"] = str(summary_path)
    manifest = {
        "scale": scale,
        "stage": "pairwise_hybrid_instances",
        "query_counts": query_counts(all_records),
        "parameters": {
            "runtime_root": None if args.runtime_root is None else str(args.runtime_root),
            "base_spmat": None if args.base_spmat is None else str(args.base_spmat),
            "base_range_fbin": str(provider.range_fbin),
            "atomic_root": str(args.atomic_root),
            "string_root": str(args.string_root),
            "range_query_root": None if args.range_query_root is None else str(args.range_query_root),
            "token_postings_bin": str(token_postings_bin),
            "token_ngram_index": str(args.token_ngram_index),
            "row_limit": int(args.row_limit),
            "row_scope": int(row_scope),
            "queries_per_hybrid_subtype": int(args.queries_per_hybrid_subtype),
            "min_and_selectivity": float(args.min_and_selectivity),
            "min_or_selectivity": float(args.min_or_selectivity),
            "max_selectivity": float(args.max_selectivity),
            "max_candidate_pairs_per_subtype": int(args.max_candidate_pairs_per_subtype),
            "max_atomic_templates_per_family": int(args.max_atomic_templates_per_family),
            "range_subtypes": sorted(parse_csv_lower(args.range_subtypes)),
            "require_range_pair": bool(args.require_range_pair),
            "reuse_query_root": None if args.reuse_query_root is None else str(args.reuse_query_root),
            "reuse_subtypes": sorted(reuse_subtypes),
            "row_cache_entries": int(args.row_cache_entries),
            "range_row_cache_entries": int(args.range_row_cache_entries),
            "max_pairs_per_outer": int(args.max_pairs_per_outer),
            "load_range_values": bool(args.load_range_values),
            "seed": int(args.seed),
        },
        "outputs": output_paths,
        "elapsed_sec": time.time() - started,
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def query_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_subtype: Counter[str] = Counter(str(record["subtype"]) for record in records)
    by_logic: Counter[str] = Counter(str(record["hybrid_top_logic"]) for record in records)
    return {
        "total": len(records),
        "hybrid": len(records),
        "by_subtype": dict(sorted(by_subtype.items())),
        "by_top_logic": dict(sorted(by_logic.items())),
    }


def duplicate_pair_count(records: list[dict[str, Any]]) -> int:
    seen: set[tuple[str, int]] = set()
    duplicates = 0
    for record in records:
        key = (str(record["predicate_id"]), int(record["query_vector_id"]))
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def duplicate_subtype_row_count(records: list[dict[str, Any]]) -> int:
    counts: Counter[tuple[str, int]] = Counter(
        (str(record["subtype"]), int(record["query_vector_id"])) for record in records
    )
    return sum(max(0, count - 1) for count in counts.values())


def query_schema() -> dict[str, Any]:
    return {
        "format": "jsonl",
        "record": {
            "family": "hybrid",
            "subtype": "pairwise subtype label such as categorical.single__range.and",
            "hybrid_top_logic": "and or or",
            "hybrid_atom_types": "two atomic predicate types",
            "predicate": "boolean tree with op equal to hybrid_top_logic",
            "predicates": "compatibility wrapper containing the boolean tree",
            "candidate_count": "true combined row-set cardinality within row_scope",
            "selectivity": "candidate_count / row_scope",
            "query_vector_id": "row id sampled from the true combined row set",
        },
        "notes": [
            "categorical atoms are categorical.single only",
            "string exact/fuzzy are not paired with each other",
            "AND lower selectivity bound defaults to 1e-5; OR lower bound defaults to 1e-4",
        ],
    }


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory already exists and is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def stable_unit(seed: int, *parts: object) -> float:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def stable_predicate_key(record: dict[str, Any]) -> str:
    payload = {
        "family": record.get("family"),
        "subtype": record.get("subtype"),
        "predicate_logic": record.get("predicate_logic"),
        "predicates": to_jsonable(record.get("predicates", [])),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def sanitize_id(value: str) -> str:
    import re

    clean = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return clean or "x"


def selectivity_band(value: float) -> str:
    if value < 0.0001:
        return "very_low"
    if value < 0.001:
        return "low"
    if value < 0.01:
        return "mid"
    if value < 0.05:
        return "high"
    if value < 0.20:
        return "very_high"
    return "broad"


def difficulty_band(value: float) -> str:
    band = selectivity_band(value)
    if band in {"very_low", "low"}:
        return "hard"
    if band == "mid":
        return "medium"
    if band == "high":
        return "easy"
    return "very_easy"


def main() -> None:
    args = parse_args()
    manifests = []
    for scale in parse_scales(args.scales):
        manifests.append(generate_for_scale(args, scale))
    run_manifest = {
        "stage": "pairwise_hybrid_instances",
        "out_root": str(args.out_root),
        "scale_manifests": [manifest["outputs"].get("query_summary_json") for manifest in manifests],
        "elapsed_sec": sum(float(manifest.get("elapsed_sec", 0.0)) for manifest in manifests),
    }
    write_json(args.out_root / "run_manifest.json", run_manifest)


if __name__ == "__main__":
    main()
