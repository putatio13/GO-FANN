#!/usr/bin/env python3
"""Generate a broad-tail LAION100M range workload with mixed operators."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DEFAULT_BASE_RANGE = Path("data/laion100m/range/base_range.fbin")

SHAPES = ("width", "height", "width_height")

FIELDS = ("original_width", "original_height")

UINT32_DTYPE = np.dtype("<u4")

FLOAT32_DTYPE = np.dtype("<f4")


@dataclass(frozen=True)
class FbinInfo:
    rows: int
    cols: int


@dataclass(frozen=True)
class ValueIndex:
    dim: int
    name: str
    values: np.ndarray
    offsets: np.ndarray
    rows_by_value: np.ndarray

    def bounds(self, lower: float, upper: float) -> tuple[int, int]:
        lo = int(np.searchsorted(self.values, np.float32(lower), side="left"))
        hi = int(np.searchsorted(self.values, np.float32(upper), side="right"))
        return lo, hi

    def count(self, lower: float, upper: float) -> int:
        lo, hi = self.bounds(lower, upper)
        return int(self.offsets[hi] - self.offsets[lo]) if hi > lo else 0

    def rows(self, lower: float, upper: float) -> np.ndarray:
        lo, hi = self.bounds(lower, upper)
        if hi <= lo:
            return np.empty(0, dtype=np.uint32)
        begin = int(self.offsets[lo])
        end = int(self.offsets[hi])
        return np.asarray(self.rows_by_value[begin:end], dtype=np.uint32)


@dataclass(frozen=True)
class CalibratedInterval:
    lower: float
    upper: float
    actual_count: int
    target_count: int
    anchor_value: float
    quantile_lo: float
    quantile_hi: float


def inspect_fbin(path: Path) -> FbinInfo:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=UINT32_DTYPE, count=2)
    if header.size != 2:
        raise ValueError(f"invalid fbin header in {path}")
    rows, cols = (int(value) for value in header.tolist())
    expected = 8 + rows * cols * FLOAT32_DTYPE.itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"fbin size mismatch in {path}: expected {expected}")
    if cols != 2:
        raise ValueError(f"expected width/height fbin with 2 columns, got {cols}")
    return FbinInfo(rows=rows, cols=cols)


def open_base_ranges(path: Path, info: FbinInfo) -> np.memmap:
    return np.memmap(path, dtype=FLOAT32_DTYPE, mode="r", offset=8, shape=(info.rows, info.cols))


def load_value_index(root: Path, dim: int, name: str, expected_rows: int) -> ValueIndex:
    values_path = root / f"{name}.unique_values.f32.npy"
    offsets_path = root / f"{name}.offsets.u64.npy"
    rows_path = root / f"{name}.rows_by_value.u32.npy"
    for path in (values_path, offsets_path, rows_path):
        if not path.exists():
            raise FileNotFoundError(path)
    values = np.load(values_path, mmap_mode="r")
    offsets = np.load(offsets_path, mmap_mode="r")
    rows = np.load(rows_path, mmap_mode="r")
    if values.ndim != 1 or offsets.shape != (values.size + 1,):
        raise ValueError(f"invalid {name} value-index shape")
    if rows.shape != (expected_rows,) or int(offsets[-1]) != expected_rows:
        raise ValueError(f"{name} value index does not cover {expected_rows} rows")
    return ValueIndex(dim=dim, name=name, values=values, offsets=offsets, rows_by_value=rows)


def validate_value_index(index: ValueIndex, base_ranges: np.ndarray) -> None:
    """Verify the exact stable value/row permutation using bounded temporary arrays."""
    if (index.values.dtype != np.dtype("<f4") or index.offsets.dtype != np.dtype("<u8")
            or index.rows_by_value.dtype != np.dtype("<u4") or index.rows_by_value.ndim != 1
            or index.values.size == 0 or int(index.offsets[0]) != 0):
        raise ValueError(f"invalid {index.name} value-index dtype or shape")
    chunk_rows = 262144
    for start in range(0, len(index.values), chunk_rows):
        stop = min(start + chunk_rows, len(index.values))
        values = index.values[max(0, start - 1):stop]
        offsets = index.offsets[start:stop + 1]
        if (not np.all(np.isfinite(values)) or np.any(values[1:] <= values[:-1])
                or np.any(offsets[1:] <= offsets[:-1])):
            raise ValueError(f"invalid {index.name} value-index ordering")
    last_group = last_id = -1
    for start in range(0, len(base_ranges), chunk_rows):
        stop = min(start + chunk_rows, len(base_ranges))
        ids = index.rows_by_value[start:stop]
        groups = np.searchsorted(index.offsets, np.arange(start, stop), side="right") - 1
        if np.any(ids >= len(base_ranges)):
            raise ValueError(f"invalid {index.name} value-index row ID")
        if (int(groups[0]) == last_group and int(ids[0]) <= last_id) or np.any(
            (groups[1:] == groups[:-1]) & (ids[1:] <= ids[:-1])
        ):
            raise ValueError(f"invalid {index.name} value-index row permutation")
        if not np.array_equal(base_ranges[ids, index.dim], index.values[groups]):
            raise ValueError(f"{index.name} value index does not match base ranges")
        last_group, last_id = int(groups[-1]), int(ids[-1])


def load_or_create_value_indexes(root: Path, base_ranges: np.ndarray) -> list[ValueIndex]:
    names = ("width", "height")
    files = [root / f"{name}.{suffix}.npy" for name in names for suffix in
             ("unique_values.f32", "offsets.u64", "rows_by_value.u32")]
    exists = [path.exists() for path in files]
    if not all(exists):
        if any(exists) or (root.exists() and any(root.iterdir())):
            raise ValueError("incomplete value index; use a new --value-index-root")
        # The same NumPy producer serves query calibration and exact GT; its
        # sorting preserves increasing original row IDs within each value.
        from scripts.laion.range_gt import build_range_value_index
        for dim, name in enumerate(names):
            build_range_value_index(base_ranges, dim, name, root, rebuild=False)
    indexes = [load_value_index(root, dim, name, len(base_ranges)) for dim, name in enumerate(names)]
    # Even an existing index without provenance must match the current values
    # and be a complete permutation. Never trust matching row counts alone.
    for index in indexes:
        validate_value_index(index, base_ranges)
    return indexes


def stable_unit(seed: int, *parts: object) -> float:
    payload = "\0".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return value / float(1 << 64)


def load_source_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("family") == "range" and record.get("subtype") == "width_height":
                predicates = record.get("predicates") or []
                if len(predicates) != 2:
                    raise ValueError(f"line {line_number}: width_height record does not have two predicates")
                records.append(record)
    return records


def source_bounds(record: dict[str, Any]) -> list[tuple[float, float]]:
    by_dim: dict[int, tuple[float, float]] = {}
    for predicate in record.get("predicates") or []:
        dim = int(predicate["column"])
        lower = float(predicate.get("lo", predicate.get("min")))
        upper = float(predicate.get("hi", predicate.get("max")))
        by_dim[dim] = (lower, upper)
    if sorted(by_dim) != [0, 1]:
        raise ValueError(f"record {record.get('query_id')} lacks width/height predicates")
    return [by_dim[0], by_dim[1]]


def candidate_lo_indices(desired: int, anchor: int) -> list[int]:
    candidates = {0, anchor, max(0, min(anchor, desired))}
    for delta in range(-64, 65):
        candidates.add(max(0, min(anchor, desired + delta)))
    return sorted(candidates)


def calibrate_single_interval(
    index: ValueIndex,
    anchor_value: float,
    target_selectivity: float,
    row_count: int,
    min_count: int,
    max_count: int,
    position: float,
) -> CalibratedInterval | None:
    anchor = int(np.searchsorted(index.values, np.float32(anchor_value), side="left"))
    if anchor >= index.values.size or np.float32(index.values[anchor]) != np.float32(anchor_value):
        raise ValueError(f"anchor value {anchor_value} is absent from {index.name} index")
    anchor_count = int(index.offsets[anchor + 1] - index.offsets[anchor])
    if anchor_count > max_count:
        return None

    target_count = max(min_count, min(max_count, int(round(target_selectivity * row_count))))
    anchor_mid = 0.5 * (int(index.offsets[anchor]) + int(index.offsets[anchor + 1]))
    anchor_fraction = 0.10 + 0.80 * position
    desired_rank = int(round(anchor_mid - anchor_fraction * target_count))
    desired_rank = max(0, min(row_count - target_count, desired_rank))
    desired_lo = int(np.searchsorted(index.offsets, desired_rank, side="right") - 1)
    desired_lo = max(0, min(anchor, desired_lo))

    best: tuple[tuple[float, float, int], int, int, int] | None = None
    for lo in candidate_lo_indices(desired_lo, anchor):
        target_end = int(index.offsets[lo]) + target_count
        base_hi = int(np.searchsorted(index.offsets, target_end, side="left"))
        for hi in (base_hi - 1, base_hi, base_hi + 1, anchor + 1):
            hi = max(anchor + 1, min(int(index.values.size), hi))
            if hi <= lo:
                continue
            count = int(index.offsets[hi] - index.offsets[lo])
            if count < min_count or count > max_count:
                continue
            realized_anchor_fraction = (anchor_mid - int(index.offsets[lo])) / max(count, 1)
            score = (
                abs(count - target_count),
                abs(realized_anchor_fraction - anchor_fraction),
                lo,
            )
            if best is None or score < best[0]:
                best = (score, lo, hi, count)
    if best is None:
        return None
    _, lo, hi, count = best
    return CalibratedInterval(
        lower=float(np.float32(index.values[lo])),
        upper=float(np.float32(index.values[hi - 1])),
        actual_count=count,
        target_count=target_count,
        anchor_value=float(np.float32(anchor_value)),
        quantile_lo=float(index.offsets[lo] / row_count),
        quantile_hi=float(index.offsets[hi] / row_count),
    )


def numeric_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def ensure_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output exists: {existing[0]}; pass --overwrite")


OPERATORS = ("between", "gt", "lt")
BANDS = ("low", "medium", "high")


@dataclass(frozen=True)
class SelectivityBand:
    name: str
    lower: float
    upper: float
    lower_inclusive: bool
    weight: float

    def count_bounds(self, row_count: int) -> tuple[int, int]:
        if self.lower_inclusive:
            minimum = int(math.ceil(self.lower * row_count))
        else:
            minimum = int(math.floor(self.lower * row_count)) + 1
        maximum = int(math.floor(self.upper * row_count))
        return minimum, maximum

    def contains_count(self, count: int, row_count: int) -> bool:
        minimum, maximum = self.count_bounds(row_count)
        return minimum <= count <= maximum


@dataclass(frozen=True)
class CalibratedRange:
    op: str
    lower: float | None
    upper: float | None
    value_begin: int
    value_end: int
    actual_count: int
    target_count: int
    quantile_lo: float
    quantile_hi: float


@dataclass(frozen=True)
class SingleSlot:
    shape: str
    op: str
    band: str
    target_selectivity: float


@dataclass(frozen=True)
class SingleAssignment:
    source: dict[str, Any]
    slot: SingleSlot
    calibrated: CalibratedRange
    query_vector_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest", type=Path, required=True
    )
    parser.add_argument(
        "--base-range-fbin", type=Path, default=DEFAULT_BASE_RANGE
    )
    parser.add_argument(
        "--value-index-root", type=Path, required=True,
        help="Validate an existing width/height value index, or build one in a new directory."
    )
    parser.add_argument("--source-rectangle-count-cache", type=Path,
                        help="Exact counts: validate an existing file, or create it if absent. "
                             "Defaults to OUTPUT_ROOT/source_rectangle_counts.json.")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Write manifests under OUTPUT_ROOT/100M.")
    parser.add_argument("--width-count", type=int, default=4000)
    parser.add_argument("--height-count", type=int, default=4000)
    parser.add_argument("--and-count", type=int, default=2000)
    parser.add_argument("--min-selectivity", type=float, default=0.0001)
    parser.add_argument("--low-band-max", type=float, default=0.01)
    parser.add_argument("--medium-band-max", type=float, default=0.02)
    parser.add_argument("--max-selectivity", type=float, default=0.05)
    parser.add_argument("--low-band-ratio", type=float, default=0.70)
    parser.add_argument("--medium-band-ratio", type=float, default=0.20)
    parser.add_argument("--high-band-ratio", type=float, default=0.10)
    parser.add_argument("--between-predicate-ratio", type=float, default=0.60)
    parser.add_argument("--gt-predicate-ratio", type=float, default=0.20)
    parser.add_argument("--lt-predicate-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalized_weights(values: dict[str, float], label: str) -> dict[str, float]:
    if any(value < 0.0 for value in values.values()):
        raise ValueError(f"{label} ratios must be non-negative")
    total = math.fsum(values.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} ratios must sum to 1, got {total}")
    return {key: value / total for key, value in values.items()}


def selectivity_bands(args: argparse.Namespace) -> dict[str, SelectivityBand]:
    if not (
        0.0
        < args.min_selectivity
        < args.low_band_max
        < args.medium_band_max
        < args.max_selectivity
        <= 1.0
    ):
        raise ValueError(
            "selectivity bounds must satisfy 0 < min < low max < medium max < max <= 1"
        )
    weights = normalized_weights(
        {
            "low": args.low_band_ratio,
            "medium": args.medium_band_ratio,
            "high": args.high_band_ratio,
        },
        "selectivity band",
    )
    return {
        "low": SelectivityBand(
            "low", args.min_selectivity, args.low_band_max, True, weights["low"]
        ),
        "medium": SelectivityBand(
            "medium", args.low_band_max, args.medium_band_max, False, weights["medium"]
        ),
        "high": SelectivityBand(
            "high", args.medium_band_max, args.max_selectivity, False, weights["high"]
        ),
    }


def validate_args(args: argparse.Namespace) -> dict[str, SelectivityBand]:
    counts = (args.width_count, args.height_count, args.and_count)
    if any(count < 0 for count in counts) or sum(counts) <= 0:
        raise ValueError("query shape counts must be non-negative and sum to a positive value")
    for path in (args.source_manifest, args.base_range_fbin):
        if not path.exists():
            raise FileNotFoundError(path)
    normalized_weights(
        {
            "between": args.between_predicate_ratio,
            "gt": args.gt_predicate_ratio,
            "lt": args.lt_predicate_ratio,
        },
        "range operator",
    )
    return selectivity_bands(args)


def proportional_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value for key, value in weights.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    ranked = sorted(weights, key=lambda key: (-(raw[key] - counts[key]), list(weights).index(key)))
    for key in ranked[:remainder]:
        counts[key] += 1
    return counts


def split_column_counts(
    totals: dict[str, int], width_count: int, height_count: int
) -> dict[str, dict[str, int]]:
    single_count = width_count + height_count
    width_raw = {key: value * width_count / single_count for key, value in totals.items()}
    width = {key: int(math.floor(value)) for key, value in width_raw.items()}
    remainder = width_count - sum(width.values())
    ranked = sorted(
        totals,
        key=lambda key: (-(width_raw[key] - width[key]), OPERATORS.index(key)),
    )
    for key in ranked:
        if remainder <= 0:
            break
        if width[key] < totals[key]:
            width[key] += 1
            remainder -= 1
    if remainder:
        raise ValueError("cannot split single-column operator quotas")
    height = {key: totals[key] - width[key] for key in totals}
    if sum(height.values()) != height_count:
        raise ValueError("height operator quota does not match requested count")
    return {"width": width, "height": height}


def operator_quotas(args: argparse.Namespace) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    weights = normalized_weights(
        {
            "between": args.between_predicate_ratio,
            "gt": args.gt_predicate_ratio,
            "lt": args.lt_predicate_ratio,
        },
        "range operator",
    )
    active_predicates = args.width_count + args.height_count + 2 * args.and_count
    global_counts = proportional_counts(active_predicates, weights)
    single_counts = dict(global_counts)
    single_counts["between"] -= 2 * args.and_count
    if any(value < 0 for value in single_counts.values()):
        raise ValueError(
            "requested BETWEEN ratio is too small because width_height queries use two BETWEEN terms"
        )
    if sum(single_counts.values()) != args.width_count + args.height_count:
        raise ValueError("single-column operator quotas do not match single query count")
    return global_counts, split_column_counts(single_counts, args.width_count, args.height_count)


def matrix_counts(row_totals: dict[str, int], column_weights: dict[str, float]) -> dict[str, dict[str, int]]:
    rows = list(row_totals)
    columns = list(column_weights)
    column_totals = proportional_counts(sum(row_totals.values()), column_weights)
    result = {row: {column: 0 for column in columns} for row in rows}
    remaining_columns = dict(column_totals)
    for row_index, row in enumerate(rows):
        if row_index == len(rows) - 1:
            result[row] = dict(remaining_columns)
            break
        raw = {column: row_totals[row] * column_weights[column] for column in columns}
        allocated = {
            column: min(int(math.floor(raw[column])), remaining_columns[column])
            for column in columns
        }
        remainder = row_totals[row] - sum(allocated.values())
        ranked = sorted(
            columns,
            key=lambda column: (-(raw[column] - math.floor(raw[column])), columns.index(column)),
        )
        while remainder:
            eligible = [column for column in ranked if allocated[column] < remaining_columns[column]]
            if not eligible:
                raise ValueError("cannot allocate exact operator/selectivity band matrix")
            for column in eligible:
                if remainder == 0:
                    break
                allocated[column] += 1
                remainder -= 1
        result[row] = allocated
        remaining_columns = {
            column: remaining_columns[column] - allocated[column] for column in columns
        }
    for row, expected in row_totals.items():
        if sum(result[row].values()) != expected:
            raise ValueError(f"matrix row {row} does not match expected count")
    return result


def stratified_targets(band: SelectivityBand, count: int) -> list[float]:
    if count == 0:
        return []
    log_lower = math.log(band.lower)
    log_upper = math.log(band.upper)
    return [
        math.exp(log_lower + ((position + 0.5) / count) * (log_upper - log_lower))
        for position in range(count)
    ]


def calibrate_between(
    index: ValueIndex,
    anchor_value: float,
    target_selectivity: float,
    row_count: int,
    band: SelectivityBand,
    position: float,
) -> CalibratedRange | None:
    minimum, maximum = band.count_bounds(row_count)
    interval = calibrate_single_interval(
        index,
        anchor_value,
        target_selectivity,
        row_count,
        minimum,
        maximum,
        position,
    )
    if interval is None:
        return None
    begin, end = index.bounds(interval.lower, interval.upper)
    return CalibratedRange(
        op="between",
        lower=interval.lower,
        upper=interval.upper,
        value_begin=begin,
        value_end=end,
        actual_count=interval.actual_count,
        target_count=interval.target_count,
        quantile_lo=interval.quantile_lo,
        quantile_hi=interval.quantile_hi,
    )


def calibrate_tail(
    index: ValueIndex,
    op: str,
    target_selectivity: float,
    row_count: int,
    band: SelectivityBand,
) -> CalibratedRange | None:
    if op not in {"gt", "lt"}:
        raise ValueError(f"tail calibration does not support op={op}")
    target_count = int(round(target_selectivity * row_count))
    minimum, maximum = band.count_bounds(row_count)
    offsets = index.offsets
    unique_count = int(index.values.size)
    if op == "gt":
        desired_begin_row = row_count - target_count
        center = int(np.searchsorted(offsets, desired_begin_row, side="left"))
    else:
        center = int(np.searchsorted(offsets, target_count, side="left"))

    best: tuple[tuple[int, int], CalibratedRange] | None = None
    for value_boundary in range(max(1, center - 4), min(unique_count - 1, center + 4) + 1):
        if op == "gt":
            count = row_count - int(offsets[value_boundary])
            calibrated = CalibratedRange(
                op=op,
                lower=float(np.float32(index.values[value_boundary - 1])),
                upper=None,
                value_begin=value_boundary,
                value_end=unique_count,
                actual_count=count,
                target_count=target_count,
                quantile_lo=float(offsets[value_boundary] / row_count),
                quantile_hi=1.0,
            )
        else:
            count = int(offsets[value_boundary])
            calibrated = CalibratedRange(
                op=op,
                lower=None,
                upper=float(np.float32(index.values[value_boundary])),
                value_begin=0,
                value_end=value_boundary,
                actual_count=count,
                target_count=target_count,
                quantile_lo=0.0,
                quantile_hi=float(offsets[value_boundary] / row_count),
            )
        if not minimum <= count <= maximum:
            continue
        score = (abs(count - target_count), value_boundary)
        if best is None or score < best[0]:
            best = (score, calibrated)
    return None if best is None else best[1]


def predicate_for_range(dim: int, calibrated: CalibratedRange) -> dict[str, Any]:
    predicate: dict[str, Any] = {
        "column": dim,
        "field": FIELDS[dim],
        "op": calibrated.op,
    }
    if calibrated.op == "between":
        predicate.update(
            {
                "hi": calibrated.upper,
                "inclusive_max": True,
                "inclusive_min": True,
                "lo": calibrated.lower,
                "max": calibrated.upper,
                "min": calibrated.lower,
            }
        )
    elif calibrated.op == "gt":
        predicate.update(
            {
                "inclusive_min": False,
                "lo": calibrated.lower,
                "lower": calibrated.lower,
                "min": calibrated.lower,
            }
        )
    elif calibrated.op == "lt":
        predicate.update(
            {
                "hi": calibrated.upper,
                "inclusive_max": False,
                "max": calibrated.upper,
                "upper": calibrated.upper,
            }
        )
    else:
        raise ValueError(f"unsupported range operator {calibrated.op}")
    return predicate


def candidate_payload_bounds(
    index: ValueIndex, calibrated: CalibratedRange
) -> tuple[int, int]:
    begin = int(index.offsets[calibrated.value_begin])
    end = int(index.offsets[calibrated.value_end])
    if end - begin != calibrated.actual_count:
        raise ValueError("calibrated range count does not match value-index payload")
    return begin, end


def choose_unique_anchor(
    index: ValueIndex,
    calibrated: CalibratedRange,
    used: set[int],
    seed: int,
    *parts: object,
) -> int:
    begin, end = candidate_payload_bounds(index, calibrated)
    count = end - begin
    start = int(stable_unit(seed, "anchor", *parts) * count)
    for offset in range(count):
        row_id = int(index.rows_by_value[begin + ((start + offset) % count)])
        if row_id not in used:
            used.add(row_id)
            return row_id
    raise RuntimeError(f"no unused anchor remains for op={calibrated.op} count={count}")


def band_for_count(
    count: int, row_count: int, bands: dict[str, SelectivityBand]
) -> str | None:
    for name in BANDS:
        if bands[name].contains_count(count, row_count):
            return name
    return None


def load_rectangle_counts(
    path: Path,
    source_manifest: Path,
    base_range_fbin: Path,
    row_count: int,
) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_json_keys)
    if not isinstance(payload, dict):
        raise ValueError("source rectangle cache must be a JSON object")
    stat = base_range_fbin.stat()
    if payload.get("source_manifest") != str(source_manifest):
        raise ValueError("source rectangle cache manifest does not match")
    if type(payload.get("row_count")) is not int or payload["row_count"] != row_count:
        raise ValueError("source rectangle cache row count does not match")
    if payload.get("base_range_size") != stat.st_size:
        raise ValueError("source rectangle cache base-range size does not match")
    if payload.get("base_range_mtime_ns") != stat.st_mtime_ns:
        raise ValueError("source rectangle cache base-range mtime does not match")
    if payload.get("base_range_fbin", str(base_range_fbin)) != str(base_range_fbin):
        raise ValueError("source rectangle cache base-range path does not match")
    if "source_manifest_sha256" in payload:
        if payload["source_manifest_sha256"] != hashlib.sha256(source_manifest.read_bytes()).hexdigest():
            raise ValueError("source rectangle cache manifest contents do not match")
    counts = payload.get("counts_by_predicate_id")
    if not isinstance(counts, dict) or any(
        type(value) is not int or not 0 <= value <= row_count for value in counts.values()
    ):
        raise ValueError("source rectangle cache counts must be integers in [0, row_count]")
    return counts


def unique_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate rectangle cache key: {key}")
        result[key] = value
    return result


def rectangle_key(record: dict[str, Any]) -> tuple[tuple[float, float, bool, bool], ...]:
    """Canonical float32 bounds, including strict gt/lt endpoint semantics."""
    bounds = {}
    for predicate in record["predicates"]:
        dim = predicate["column"]
        if type(dim) is not int or dim not in (0, 1) or dim in bounds:
            raise ValueError("rectangle predicates require distinct columns 0 and 1")
        op = predicate.get("op", "between")
        if op not in ("between", "gt", "lt"):
            raise ValueError(f"unsupported rectangle operator: {op}")
        lo = -math.inf if op == "lt" else float(np.float32(
            predicate.get("lo", predicate.get("min", predicate.get("lower")))))
        hi = math.inf if op == "gt" else float(np.float32(
            predicate.get("hi", predicate.get("max", predicate.get("upper")))))
        if math.isnan(lo) or math.isnan(hi) or lo > hi:
            raise ValueError("invalid rectangle bounds")
        inclusive_lo = predicate.get("inclusive_min", True) if op == "between" else op != "gt"
        inclusive_hi = predicate.get("inclusive_max", True) if op == "between" else op != "lt"
        if type(inclusive_lo) is not bool or type(inclusive_hi) is not bool:
            raise ValueError("rectangle inclusive flags must be boolean")
        bounds[dim] = (lo, hi, inclusive_lo, inclusive_hi)
    if sorted(bounds) != [0, 1]:
        raise ValueError("rectangle requires width and height predicates")
    return tuple(bounds[dim] for dim in (0, 1))


def count_rectangle(
    bounds: tuple[tuple[float, float, bool, bool], ...],
    indexes: list[ValueIndex],
    base_ranges: np.ndarray,
    row_count: int,
    chunk_rows: int = 262144,
) -> int:
    """Count the smaller indexed marginal, intersecting the other in bounded chunks."""
    if not 0 <= row_count <= len(base_ranges) or chunk_rows <= 0:
        raise ValueError("invalid rectangle row prefix or chunk size")
    spans = []
    for index, (lo, hi, inclusive_lo, inclusive_hi) in zip(indexes, bounds):
        begin = int(np.searchsorted(index.values, lo, side="left" if inclusive_lo else "right"))
        end = int(np.searchsorted(index.values, hi, side="right" if inclusive_hi else "left"))
        spans.append((int(index.offsets[begin]), int(index.offsets[max(begin, end)])))
    chosen = min(range(2), key=lambda dim: spans[dim][1] - spans[dim][0])
    begin, end = spans[chosen]
    lo, hi, inclusive_lo, inclusive_hi = bounds[1 - chosen]
    total = 0
    for start in range(begin, end, chunk_rows):
        ids = indexes[chosen].rows_by_value[start:min(start + chunk_rows, end)]
        if np.any(ids >= len(base_ranges)) or np.any(ids < 0):
            raise ValueError("rectangle value index contains an out-of-range row ID")
        ids = ids[ids < row_count]
        values = base_ranges[ids, 1 - chosen]
        lower = values >= lo if inclusive_lo else values > lo
        upper = values <= hi if inclusive_hi else values < hi
        total += int(np.count_nonzero(lower & upper))
    return total


def load_or_create_rectangle_counts(
    path: Path,
    source_manifest: Path,
    base_range_fbin: Path,
    row_count: int,
    records: list[dict[str, Any]],
    indexes: list[ValueIndex],
    base_ranges: np.ndarray,
) -> dict[str, int]:
    keys = {}
    for record in records:
        predicate_id = str(record["predicate_id"])
        key = rectangle_key(record)
        if predicate_id in keys and keys[predicate_id] != key:
            raise ValueError(f"conflicting rectangles for predicate ID {predicate_id}")
        keys[predicate_id] = key
    if path.exists():
        counts = load_rectangle_counts(path, source_manifest, base_range_fbin, row_count)
        if not keys.keys() <= counts.keys():
            raise ValueError("source rectangle cache misses predicate IDs")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "source_manifest_sha256" in payload:
            return counts
        # Historical caches have no manifest fingerprint: verify their counts
        # against the current predicates without modifying the existing cache.
    else:
        counts = None
    manifest_hash = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    stat = base_range_fbin.stat()
    by_rectangle = {}
    exact = {}
    for predicate_id, key in keys.items():
        if key not in by_rectangle:
            by_rectangle[key] = count_rectangle(key, indexes, base_ranges, row_count)
        exact[predicate_id] = by_rectangle[key]
    if counts is not None:
        if any(counts[key] != count for key, count in exact.items()):
            raise ValueError("legacy source rectangle cache counts do not match current inputs")
        return counts
    after = base_range_fbin.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns) or (
        hashlib.sha256(source_manifest.read_bytes()).hexdigest() != manifest_hash
    ):
        raise ValueError("rectangle inputs changed while counting")
    payload = {
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": manifest_hash,
        "base_range_fbin": str(base_range_fbin),
        "base_range_size": stat.st_size,
        "base_range_mtime_ns": stat.st_mtime_ns,
        "row_count": row_count,
        "counts_by_predicate_id": exact,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return exact


def select_evenly(
    records: list[dict[str, Any]], count: int, counts: dict[str, int], seed: int, band: str
) -> list[dict[str, Any]]:
    if len(records) < count:
        raise RuntimeError(f"band={band} has {len(records)} valid source AND records, needs {count}")
    ordered = sorted(
        records,
        key=lambda record: (
            counts[str(record["predicate_id"])],
            stable_unit(seed, "and", band, record["query_id"]),
        ),
    )
    selected = []
    for position in range(count):
        index = int((position + 0.5) * len(ordered) / count)
        selected.append(ordered[index])
    if len({record["query_id"] for record in selected}) != count:
        raise RuntimeError("even AND selection emitted duplicate records")
    return selected


def make_slots(
    shape: str,
    operator_counts: dict[str, int],
    bands: dict[str, SelectivityBand],
    seed: int,
) -> list[SingleSlot]:
    band_weights = {name: bands[name].weight for name in BANDS}
    matrix = matrix_counts(operator_counts, band_weights)
    slots = []
    for op in OPERATORS:
        for band_name in BANDS:
            targets = stratified_targets(bands[band_name], matrix[op][band_name])
            for target in targets:
                slots.append(SingleSlot(shape, op, band_name, target))
    return sorted(
        slots,
        key=lambda slot: (
            OPERATORS.index(slot.op),
            BANDS.index(slot.band),
            stable_unit(
                seed, "slot", shape, slot.op, slot.band, slot.target_selectivity
            ),
        ),
    )


def predicate_matches(value: float, calibrated: CalibratedRange) -> bool:
    if calibrated.op == "between":
        assert calibrated.lower is not None and calibrated.upper is not None
        return calibrated.lower <= value <= calibrated.upper
    if calibrated.op == "gt":
        assert calibrated.lower is not None
        return value > calibrated.lower
    if calibrated.op == "lt":
        assert calibrated.upper is not None
        return value < calibrated.upper
    raise ValueError(f"unsupported range operator {calibrated.op}")


def make_record(
    source: dict[str, Any],
    shape: str,
    shape_index: int,
    predicates: list[dict[str, Any]],
    query_vector_id: int,
    actual_count: int,
    target_selectivity: float,
    row_count: int,
    selectivity_band: str,
) -> dict[str, Any]:
    record = copy.deepcopy(source)
    for stale_key in (
        "anchor_sample_id",
        "consumer",
        "diagnostics",
        "difficulty_band",
        "predicate_instance_rank",
        "similarity_proxy",
    ):
        record.pop(stale_key, None)
    actual_selectivity = actual_count / row_count
    predicate_id = f"laion-100m-range-opmix-{shape}-{shape_index:05d}"
    source_query_vector_id = int(source["query_vector_id"])
    operators = [str(predicate["op"]) for predicate in predicates]
    record.update(
        {
            "actual_selectivity": actual_selectivity,
            "anchor_reused": False,
            "anchor_row_id": query_vector_id,
            "anchor_sample_method": "full_100m_range_value_index",
            "anchor_source": "full_100m_range_value_index",
            "candidate_count": actual_count,
            "family": "range",
            "label": predicate_id,
            "notes": [
                "Deterministic true-hit anchor selected or verified against the full range value index."
            ],
            "predicate_id": predicate_id,
            "predicate_key": hashlib.blake2b(predicate_id.encode("utf-8"), digest_size=8).hexdigest(),
            "predicate_logic": "AND",
            "predicate_reused": False,
            "predicates": predicates,
            "query_id": f"{predicate_id}-qv-{query_vector_id:08d}",
            "query_vector_id": query_vector_id,
            "query_vector_reassigned": query_vector_id != source_query_vector_id,
            "range_operators": operators,
            "range_shape": shape,
            "selectivity": actual_selectivity,
            "selectivity_band": selectivity_band,
            "source_predicate_id": source.get("predicate_id"),
            "source_query_id": source.get("query_id"),
            "source_query_vector_id": source_query_vector_id,
            "stats": {
                "actual_rows": actual_count,
                "actual_selectivity": actual_selectivity,
                "estimate_method": "full_100m_value_index_exact_count",
                "estimated_rows": actual_count,
                "estimated_selectivity": actual_selectivity,
                "is_estimate": False,
                "selectivity_source": "exact_full_range_index",
                "target_rows": int(round(target_selectivity * row_count)),
                "target_selectivity": target_selectivity,
            },
            "subtype": shape,
            "target_selectivity": target_selectivity,
        }
    )
    record.pop("flashanns_query_index", None)
    return record


def main() -> None:
    args = parse_args()
    bands = validate_args(args)
    if args.source_rectangle_count_cache is None:
        args.source_rectangle_count_cache = args.output_root / "source_rectangle_counts.json"
    primary_output = args.output_root / "100M/query_manifest.jsonl"
    if primary_output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {primary_output}; pass --overwrite")

    info = inspect_fbin(args.base_range_fbin)
    base_ranges = open_base_ranges(args.base_range_fbin, info)
    source_records = load_source_records(args.source_manifest)
    requested_total = args.width_count + args.height_count + args.and_count
    if len(source_records) != requested_total:
        raise ValueError(f"found {len(source_records)} source records, requested {requested_total}")
    source_query_ids = [int(record["query_vector_id"]) for record in source_records]
    if len(set(source_query_ids)) != len(source_query_ids):
        raise ValueError("source records do not have unique query_vector_id values")
    if any(row_id < 0 or row_id >= info.rows for row_id in source_query_ids):
        raise ValueError("source query_vector_id is outside the base row range")
    if args.source_rectangle_count_cache.exists():
        load_rectangle_counts(args.source_rectangle_count_cache, args.source_manifest,
                              args.base_range_fbin, info.rows)
    indexes = load_or_create_value_indexes(args.value_index_root, base_ranges)

    rectangle_counts = load_or_create_rectangle_counts(
        args.source_rectangle_count_cache,
        args.source_manifest,
        args.base_range_fbin,
        info.rows,
        source_records,
        indexes,
        base_ranges,
    )
    missing = {
        str(record["predicate_id"])
        for record in source_records
        if str(record["predicate_id"]) not in rectangle_counts
    }
    if missing:
        raise ValueError(f"source rectangle cache misses {len(missing)} predicate ids")

    global_operator_counts, single_operator_counts = operator_quotas(args)
    band_weights = {name: bands[name].weight for name in BANDS}
    and_band_counts = proportional_counts(args.and_count, band_weights)
    selected_and: list[dict[str, Any]] = []
    for band_name in BANDS:
        candidates = [
            record
            for record in source_records
            if bands[band_name].contains_count(
                rectangle_counts[str(record["predicate_id"])], info.rows
            )
        ]
        selected_and.extend(
            select_evenly(candidates, and_band_counts[band_name], rectangle_counts, args.seed, band_name)
        )
    selected_and_ids = {str(record["query_id"]) for record in selected_and}
    if len(selected_and_ids) != args.and_count:
        raise RuntimeError("AND selection is not globally unique")
    available = [record for record in source_records if str(record["query_id"]) not in selected_and_ids]

    slots_by_shape = {
        shape: make_slots(shape, single_operator_counts[shape], bands, args.seed)
        for shape in ("width", "height")
    }
    between_assignments: list[tuple[dict[str, Any], SingleSlot, CalibratedRange]] = []
    for shape in ("width", "height"):
        dim = 0 if shape == "width" else 1
        between_slots = [slot for slot in slots_by_shape[shape] if slot.op == "between"]
        for slot_index, slot in enumerate(between_slots):
            chosen = None
            for record_index, record in enumerate(available):
                row_id = int(record["query_vector_id"])
                calibrated = calibrate_between(
                    indexes[dim],
                    float(base_ranges[row_id, dim]),
                    slot.target_selectivity,
                    info.rows,
                    bands[slot.band],
                    stable_unit(
                        args.seed, "between", shape, slot.band, slot_index, record["query_id"]
                    ),
                )
                if calibrated is not None:
                    chosen = (record_index, record, calibrated)
                    break
            if chosen is None:
                raise RuntimeError(f"cannot find a feasible anchor for {shape} {slot.band} BETWEEN")
            record_index, record, calibrated = chosen
            available.pop(record_index)
            between_assignments.append((record, slot, calibrated))
        print(f"[assign-between] shape={shape} rows={len(between_slots)}", flush=True)

    tail_slots = [
        slot
        for shape in ("width", "height")
        for slot in slots_by_shape[shape]
        if slot.op != "between"
    ]
    tail_slots.sort(
        key=lambda slot: (
            slot.target_selectivity,
            slot.shape,
            OPERATORS.index(slot.op),
            stable_unit(
                args.seed, "tail-order", slot.shape, slot.op, slot.band, slot.target_selectivity
            ),
        )
    )
    available.sort(key=lambda record: stable_unit(args.seed, "source", record["query_id"]))
    if len(available) != len(tail_slots):
        raise RuntimeError(
            f"tail source count {len(available)} does not match tail slot count {len(tail_slots)}"
        )

    used_query_vectors = {int(record["query_vector_id"]) for record in selected_and}
    used_query_vectors.update(int(record["query_vector_id"]) for record, _slot, _cal in between_assignments)
    assignments: list[SingleAssignment] = [
        SingleAssignment(record, slot, calibrated, int(record["query_vector_id"]))
        for record, slot, calibrated in between_assignments
    ]
    for slot_index, (record, slot) in enumerate(zip(available, tail_slots)):
        dim = 0 if slot.shape == "width" else 1
        calibrated = calibrate_tail(
            indexes[dim], slot.op, slot.target_selectivity, info.rows, bands[slot.band]
        )
        if calibrated is None:
            raise RuntimeError(
                f"cannot calibrate {slot.shape} {slot.op} target={slot.target_selectivity}"
            )
        query_vector_id = choose_unique_anchor(
            indexes[dim],
            calibrated,
            used_query_vectors,
            args.seed,
            slot.shape,
            slot.op,
            slot.band,
            slot_index,
        )
        assignments.append(SingleAssignment(record, slot, calibrated, query_vector_id))
    print(f"[assign-tail] rows={len(tail_slots)}", flush=True)

    output_by_shape: dict[str, list[dict[str, Any]]] = {shape: [] for shape in SHAPES}
    assignments_by_shape = {
        shape: [assignment for assignment in assignments if assignment.slot.shape == shape]
        for shape in ("width", "height")
    }
    for shape in ("width", "height"):
        dim = 0 if shape == "width" else 1
        ordered = sorted(
            assignments_by_shape[shape],
            key=lambda item: (
                item.slot.target_selectivity,
                OPERATORS.index(item.slot.op),
                item.source["query_id"],
            ),
        )
        for shape_index, assignment in enumerate(ordered):
            anchor_value = float(base_ranges[assignment.query_vector_id, dim])
            if not predicate_matches(anchor_value, assignment.calibrated):
                raise ValueError(f"single-range anchor {assignment.query_vector_id} is not a true hit")
            if not bands[assignment.slot.band].contains_count(
                assignment.calibrated.actual_count, info.rows
            ):
                raise ValueError("single-range actual count escaped its selectivity band")
            output_by_shape[shape].append(
                make_record(
                    assignment.source,
                    shape,
                    shape_index,
                    [predicate_for_range(dim, assignment.calibrated)],
                    assignment.query_vector_id,
                    assignment.calibrated.actual_count,
                    assignment.slot.target_selectivity,
                    info.rows,
                    assignment.slot.band,
                )
            )

    selected_and.sort(
        key=lambda record: (
            rectangle_counts[str(record["predicate_id"])],
            stable_unit(args.seed, "and-output", record["query_id"]),
        )
    )
    for shape_index, source in enumerate(selected_and):
        row_id = int(source["query_vector_id"])
        bounds = source_bounds(source)
        if not all(
            lower <= float(base_ranges[row_id, dim]) <= upper
            for dim, (lower, upper) in enumerate(bounds)
        ):
            raise ValueError(f"source AND anchor {row_id} is not a true hit")
        actual_count = rectangle_counts[str(source["predicate_id"])]
        band_name = band_for_count(actual_count, info.rows, bands)
        if band_name is None:
            raise ValueError("selected AND record escaped formal selectivity bands")
        predicates = copy.deepcopy(source["predicates"])
        if any(str(predicate.get("op", "between")) != "between" for predicate in predicates):
            raise ValueError("source AND records must contain BETWEEN predicates")
        output_by_shape["width_height"].append(
            make_record(
                source,
                "width_height",
                shape_index,
                predicates,
                row_id,
                actual_count,
                actual_count / info.rows,
                info.rows,
                band_name,
            )
        )

    combined = [record for shape in SHAPES for record in output_by_shape[shape]]
    combined.sort(key=lambda record: stable_unit(args.seed, "combined", record["query_id"]))
    for query_index, record in enumerate(combined):
        record["workload_query_index"] = query_index
    if len({int(record["query_vector_id"]) for record in combined}) != len(combined):
        raise RuntimeError("output query vectors are not globally unique")

    scale_root = args.output_root / "100M"
    outputs = {
        "combined": scale_root / "query_manifest.jsonl",
        "range": scale_root / "query_manifest_range.jsonl",
        "width": scale_root / "query_manifest_width.jsonl",
        "height": scale_root / "query_manifest_height.jsonl",
        "width_height": scale_root / "query_manifest_width_height.jsonl",
        "summary": scale_root / "query_summary.json",
        "manifest": scale_root / "manifest.json",
        "schema": scale_root / "query_schema.json",
        "run_manifest": args.output_root / "run_manifest.json",
    }
    ensure_outputs(outputs.values(), args.overwrite)
    scale_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(outputs["combined"], combined)
    write_jsonl(outputs["range"], combined)
    for shape in SHAPES:
        write_jsonl(outputs[shape], output_by_shape[shape])

    summary_by_shape = {}
    for shape in SHAPES:
        rows = output_by_shape[shape]
        summary_by_shape[shape] = {
            "count": len(rows),
            "active_range_terms": sum(len(record["predicates"]) for record in rows),
            "operator_predicate_counts": dict(
                sorted(Counter(op for record in rows for op in record["range_operators"]).items())
            ),
            "selectivity_band_counts": dict(sorted(Counter(record["selectivity_band"] for record in rows).items())),
            "actual_rows": numeric_stats(record["candidate_count"] for record in rows),
            "actual_selectivity": numeric_stats(record["actual_selectivity"] for record in rows),
            "target_selectivity": numeric_stats(record["target_selectivity"] for record in rows),
            "absolute_selectivity_error": numeric_stats(
                abs(record["actual_selectivity"] - record["target_selectivity"]) for record in rows
            ),
            "query_vector_reassigned_count": sum(bool(record["query_vector_reassigned"]) for record in rows),
        }
    actual_operator_counts = Counter(
        op for record in combined for op in record["range_operators"]
    )
    if dict(actual_operator_counts) != global_operator_counts:
        raise RuntimeError(
            f"operator counts mismatch: actual={dict(actual_operator_counts)} expected={global_operator_counts}"
        )
    summary = {
        "scale": "100M",
        "family": "range",
        "query_count": len(combined),
        "active_range_predicate_count": sum(actual_operator_counts.values()),
        "query_counts_by_subtype": {shape: len(output_by_shape[shape]) for shape in SHAPES},
        "query_vector_unique_count": len({int(record["query_vector_id"]) for record in combined}),
        "operator_predicate_counts": dict(sorted(actual_operator_counts.items())),
        "selectivity_band_counts": dict(sorted(Counter(record["selectivity_band"] for record in combined).items())),
        "selectivity_bands": {
            name: {
                "lower": bands[name].lower,
                "lower_inclusive": bands[name].lower_inclusive,
                "upper": bands[name].upper,
                "upper_inclusive": True,
                "ratio": bands[name].weight,
            }
            for name in BANDS
        },
        "by_subtype": summary_by_shape,
        "source": {
            "source_manifest": str(args.source_manifest),
            "source_rectangle_count_cache": str(args.source_rectangle_count_cache),
            "base_range_fbin": str(args.base_range_fbin),
            "value_index_root": str(args.value_index_root),
        },
        "method": {
            "between_single": "anchor-containing empirical interval calibrated within its selectivity band",
            "one_sided_single": "empirical CDF tail followed by a deterministic unique true-hit anchor selection",
            "width_height": "source BETWEEN rectangle rebucketed by exact full-100M intersection count",
            "operator_allocation": (
                "exact quotas over all active range predicates; "
                "width_height contributes two BETWEEN predicates"
            ),
            "target_allocation": "70/20/10 exact query quotas over low/medium/high selectivity bands",
        },
        "seed": args.seed,
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    schema = {
        "family": "range",
        "operators": {
            "between": "inclusive lower and upper source bounds",
            "gt": "strict lower-bound predicate",
            "lt": "strict upper-bound predicate",
        },
        "subtypes": {
            "width": {"logic": "single", "active_columns": [0]},
            "height": {"logic": "single", "active_columns": [1]},
            "width_height": {"logic": "and", "active_columns": [0, 1]},
        },
        "range_columns": {"original_width": 0, "original_height": 1},
        "selectivity": "exact full-100M candidate count divided by row count",
    }
    outputs["schema"].write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs["manifest"].write_text(
        json.dumps(
            {
                "scale": "100M",
                "query_count": len(combined),
                "query_manifest": str(outputs["combined"]),
                "query_summary": str(outputs["summary"]),
                "query_schema": str(outputs["schema"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["run_manifest"].write_text(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "counts": {
                    "width": args.width_count,
                    "height": args.height_count,
                    "width_height": args.and_count,
                },
                "operator_predicate_counts": global_operator_counts,
                "selectivity_band_query_counts": proportional_counts(len(combined), band_weights),
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] output={args.output_root} queries={len(combined)} "
        f"operators={dict(sorted(actual_operator_counts.items()))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
