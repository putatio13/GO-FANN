#!/usr/bin/env python3
"""Split query runtimes and assemble per-subtype ground truth in manifest order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from collections import Counter, defaultdict
from scripts.laion.categorical_gt import (
    hvecs_records,
    inspect_hvecs,
    write_hvecs,
    write_dist_fbin,
    write_gt_ibin,
)


def sanitize(value: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "unknown"


def ensure_output_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_matrix(path: Path, dtype: np.dtype) -> np.ndarray:
    with path.open("rb") as handle:
        header = np.fromfile(handle, dtype=np.dtype("<u4"), count=2)
    if header.size != 2:
        raise ValueError(f"invalid matrix header in {path}")
    rows, cols = int(header[0]), int(header[1])
    expected_bytes = 8 + rows * cols * np.dtype(dtype).itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"matrix payload size mismatch in {path}: expected {expected_bytes} bytes")
    matrix = np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(rows, cols))
    return np.asarray(matrix)


def write_rows_jsonl(
    path: Path,
    ordered_rows: list[dict[str, Any]],
    subtype_gt_rows: dict[str, list[dict[str, Any]]],
    subtype_local_indices: list[int],
    overwrite: bool,
) -> dict[str, Any]:
    ensure_output_path(path, overwrite)
    with path.open("w", encoding="utf-8") as handle:
        for global_idx, manifest_row in enumerate(ordered_rows):
            subtype = str(manifest_row.get("subtype", "unknown"))
            local_idx = subtype_local_indices[global_idx]
            source = dict(subtype_gt_rows[subtype][local_idx])
            source["flashanns_query_index"] = global_idx
            source["flashanns_subtype_query_index"] = local_idx
            source["flashanns_parent_query_index"] = int(manifest_row.get("flashanns_query_index", global_idx))
            source["query_id"] = str(manifest_row.get("query_id", source.get("query_id", global_idx)))
            source["subtype"] = subtype
            handle.write(json.dumps(source, sort_keys=True) + "\n")
    return {"path": str(path), "rows": len(ordered_rows), "bytes": path.stat().st_size}

UINT32_MAX = np.iinfo(np.uint32).max


def load_subtype_outputs(
    root: Path,
    subtype: str,
    expected_count: int,
    gt_tag: str,
    topk: int,
) -> dict[str, Any]:
    subtype_root = root / sanitize(subtype)
    manifest_path = subtype_root / "runtime/query_manifest.flashanns.jsonl"
    gt_root = subtype_root / "gt"
    ids_path = gt_root / f"{gt_tag}.top{topk}.ibin"
    distances_path = gt_root / f"{gt_tag}.top{topk}.dist.fbin"
    candidate_path = gt_root / f"{gt_tag}.candidate_counts.npy"
    chosen_path = gt_root / f"{gt_tag}.chosen_dims.npy"
    for required in (manifest_path, ids_path, distances_path, candidate_path, chosen_path):
        if not required.exists():
            raise FileNotFoundError(required)

    manifest = load_jsonl(manifest_path)
    ids = read_matrix(ids_path, np.dtype("<u4"))
    distances = read_matrix(distances_path, np.dtype("<f4"))
    candidate_counts = np.load(candidate_path)
    chosen_dims = np.load(chosen_path)
    expected_matrix_shape = (expected_count, topk)
    if len(manifest) != expected_count:
        raise ValueError(f"{subtype} manifest rows {len(manifest)} != {expected_count}")
    if ids.shape != expected_matrix_shape:
        raise ValueError(f"{subtype} ids shape {ids.shape} != {expected_matrix_shape}")
    if distances.shape != expected_matrix_shape:
        raise ValueError(
            f"{subtype} distances shape {distances.shape} != {expected_matrix_shape}"
        )
    if candidate_counts.shape != (expected_count,):
        raise ValueError(f"{subtype} candidate-count shape mismatch: {candidate_counts.shape}")
    if chosen_dims.shape != (expected_count,):
        raise ValueError(f"{subtype} chosen-dim shape mismatch: {chosen_dims.shape}")

    query_ids = [str(row.get("query_id")) for row in manifest]
    if any(query_id == "None" for query_id in query_ids):
        raise ValueError(f"{subtype} manifest contains a missing query_id")
    if len(set(query_ids)) != expected_count:
        raise ValueError(f"{subtype} manifest query_id values are not unique")
    if any(str(row.get("subtype")) != subtype for row in manifest):
        raise ValueError(f"{subtype} runtime manifest contains another subtype")
    return {
        "manifest": manifest,
        "ids": ids,
        "distances": distances,
        "candidate_counts": np.asarray(candidate_counts, dtype=np.int64),
        "chosen_dims": np.asarray(chosen_dims, dtype=np.int8),
        "query_id_to_local": {query_id: index for index, query_id in enumerate(query_ids)},
    }


def validate_gt(ids: np.ndarray, distances: np.ndarray, num_points: int) -> None:
    if np.any(ids == UINT32_MAX):
        raise ValueError("combined GT contains invalid uint32 sentinel ids")
    if np.any(ids >= num_points):
        raise ValueError(f"combined GT contains ids outside [0, {num_points})")
    if ids.shape[1] > 1:
        sorted_ids = np.sort(ids, axis=1)
        if np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
            raise ValueError("combined GT contains duplicate ids within a row")
    if not np.all(np.isfinite(distances)):
        raise ValueError("combined GT contains non-finite distances")
    if np.any(distances < 0.0):
        raise ValueError("combined GT contains negative squared L2 distances")
    if np.any(distances[:, 1:] < distances[:, :-1]):
        raise ValueError("combined GT distances are not nondecreasing within each row")


def write_rows(
    path: Path,
    all_rows: list[dict[str, Any]],
    local_indices: list[int],
    candidate_counts: np.ndarray,
    overwrite: bool,
) -> dict[str, Any]:
    ensure_output_path(path, overwrite)
    with path.open("w", encoding="utf-8") as handle:
        for global_index, (row, local_index) in enumerate(zip(all_rows, local_indices)):
            payload = {
                "candidate_count": int(candidate_counts[global_index]),
                "flashanns_query_index": global_index,
                "flashanns_subtype_query_index": local_index,
                "query_id": str(row["query_id"]),
                "query_vector_id": int(row["query_vector_id"]),
                "subtype": str(row["subtype"]),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return {"path": str(path), "rows": len(all_rows), "bytes": path.stat().st_size}


def combine(args: argparse.Namespace) -> dict[str, Any]:
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")
    for required in (args.all_query_manifest, args.by_subtype_root):
        if not required.exists():
            raise FileNotFoundError(required)

    all_rows = load_jsonl(args.all_query_manifest)
    if not all_rows:
        raise RuntimeError(f"empty all-query manifest: {args.all_query_manifest}")
    query_ids = [str(row.get("query_id")) for row in all_rows]
    if any(query_id == "None" for query_id in query_ids):
        raise ValueError("all-query manifest contains a missing query_id")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("all-query manifest query_id values are not unique")

    subtype_counts = Counter(str(row.get("subtype")) for row in all_rows)
    subtype_outputs = {
        subtype: load_subtype_outputs(
            args.by_subtype_root,
            subtype,
            count,
            args.gt_tag,
            args.topk,
        )
        for subtype, count in sorted(subtype_counts.items())
    }

    combined_ids = np.empty((len(all_rows), args.topk), dtype=np.uint32)
    combined_distances = np.empty((len(all_rows), args.topk), dtype=np.float32)
    candidate_counts = np.empty(len(all_rows), dtype=np.int64)
    chosen_dims = np.empty(len(all_rows), dtype=np.int8)
    local_indices = []
    for global_index, row in enumerate(all_rows):
        subtype = str(row["subtype"])
        query_id = str(row["query_id"])
        output = subtype_outputs[subtype]
        try:
            local_index = output["query_id_to_local"][query_id]
        except KeyError as exc:
            raise ValueError(f"query_id {query_id} is missing from subtype {subtype}") from exc
        local_indices.append(local_index)
        combined_ids[global_index] = output["ids"][local_index]
        combined_distances[global_index] = output["distances"][local_index]
        candidate_counts[global_index] = output["candidate_counts"][local_index]
        chosen_dims[global_index] = output["chosen_dims"][local_index]
        manifest_count = int(row.get("candidate_count", -1))
        if candidate_counts[global_index] != manifest_count:
            raise ValueError(
                f"query_id {query_id} candidate count {candidate_counts[global_index]} "
                f"!= manifest {manifest_count}"
            )

    validate_gt(combined_ids, combined_distances, args.num_points)
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"top{args.topk}_gt": write_gt_ibin(
            args.output_root / f"{args.gt_tag}.top{args.topk}.ibin",
            combined_ids,
            args.overwrite,
        ),
        f"top{args.topk}_dist": write_dist_fbin(
            args.output_root / f"{args.gt_tag}.top{args.topk}.dist.fbin",
            combined_distances,
            args.overwrite,
        ),
    }
    if args.topk > 10:
        outputs["top10_gt"] = write_gt_ibin(
            args.output_root / f"{args.gt_tag}.top10.ibin",
            combined_ids[:, :10],
            args.overwrite,
        )
        outputs["top10_dist"] = write_dist_fbin(
            args.output_root / f"{args.gt_tag}.top10.dist.fbin",
            combined_distances[:, :10],
            args.overwrite,
        )

    candidate_path = args.output_root / f"{args.gt_tag}.candidate_counts.npy"
    chosen_path = args.output_root / f"{args.gt_tag}.chosen_dims.npy"
    ensure_output_path(candidate_path, args.overwrite)
    ensure_output_path(chosen_path, args.overwrite)
    np.save(candidate_path, candidate_counts)
    np.save(chosen_path, chosen_dims)
    outputs["candidate_counts"] = str(candidate_path)
    outputs["chosen_dims"] = str(chosen_path)
    outputs["rows"] = write_rows(
        args.output_root / f"{args.gt_tag}.rows.jsonl",
        all_rows,
        local_indices,
        candidate_counts,
        args.overwrite,
    )

    summary = {
        "all_query_manifest": str(args.all_query_manifest),
        "by_subtype_root": str(args.by_subtype_root),
        "candidate_count_max": int(candidate_counts.max()),
        "candidate_count_median": float(np.median(candidate_counts)),
        "candidate_count_min": int(candidate_counts.min()),
        "candidate_count_p90": float(np.quantile(candidate_counts, 0.90)),
        "candidate_count_sum": int(candidate_counts.sum()),
        "distance_finite_count": int(np.isfinite(combined_distances).sum()),
        "gt_tag": args.gt_tag,
        "num_points": int(args.num_points),
        "outputs": outputs,
        "query_count": len(all_rows),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "topk": int(args.topk),
    }
    summary_path = args.output_root / f"{args.gt_tag}.summary.json"
    ensure_output_path(summary_path, args.overwrite)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_manifest(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    ensure_output_path(path, overwrite)
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            item = dict(row)
            item["flashanns_query_index"] = index
            item["flashanns_parent_query_index"] = int(row.get("flashanns_query_index", index))
            handle.write(json.dumps(item, sort_keys=True) + "\n")


def add_combine_hybrid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--all-query-manifest", type=Path, required=True)
    parser.add_argument("--by-subtype-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")


def add_combine_range_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--all-query-manifest", type=Path, required=True)
    parser.add_argument("--by-subtype-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gt-tag", default="gt.range")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--num-points", type=int, default=100_000_000)
    parser.add_argument("--overwrite", action="store_true")


def add_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--query-hvecs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")


def combine_hybrid(args: argparse.Namespace) -> None:
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    for required in (args.all_query_manifest, args.by_subtype_root):
        if not required.exists():
            raise FileNotFoundError(required)

    all_rows = load_jsonl(args.all_query_manifest)
    if not all_rows:
        raise RuntimeError(f"empty all-query manifest: {args.all_query_manifest}")

    subtype_local_indices: list[int] = []
    subtype_seen: Counter[str] = Counter()
    for row in all_rows:
        subtype = str(row.get("subtype", "unknown"))
        subtype_local_indices.append(subtype_seen[subtype])
        subtype_seen[subtype] += 1

    subtype_ids: dict[str, np.ndarray] = {}
    subtype_dist: dict[str, np.ndarray] = {}
    subtype_gt_rows: dict[str, list[dict[str, Any]]] = {}
    subtype_positions: dict[str, dict[str, int]] = {}
    for subtype, count in sorted(subtype_seen.items()):
        subtype_dir = args.by_subtype_root / sanitize(subtype) / "gt"
        ids_path = subtype_dir / f"gt.hybrid.top{args.topk}.ibin"
        dist_path = subtype_dir / f"gt.hybrid.top{args.topk}.dist.fbin"
        rows_path = subtype_dir / "gt.hybrid.rows.jsonl"
        for required in (ids_path, dist_path, rows_path):
            if not required.exists():
                raise FileNotFoundError(required)
        ids = read_matrix(ids_path, np.dtype("<u4"))
        dist = read_matrix(dist_path, np.dtype("<f4"))
        gt_rows = load_jsonl(rows_path)
        if ids.shape != dist.shape:
            raise ValueError(f"{subtype} ids/dist shape mismatch: {ids.shape} vs {dist.shape}")
        if ids.shape != (count, args.topk):
            raise ValueError(f"{subtype} expected {(count, args.topk)}, got {ids.shape}")
        if len(gt_rows) != count:
            raise ValueError(f"{subtype} rows JSONL expected {count}, got {len(gt_rows)}")
        query_ids = [str(row.get("query_id")) for row in gt_rows]
        if "None" in query_ids or len(set(query_ids)) != count:
            raise ValueError(f"{subtype} GT query_id values must be present and unique")
        expected_ids = [str(row.get("query_id")) for row in all_rows if str(row.get("subtype", "unknown")) == subtype]
        if len(set(expected_ids)) != count or set(expected_ids) != set(query_ids):
            raise ValueError(f"{subtype} GT query IDs differ from the source manifest")
        if any(str(row.get("subtype", subtype)) != subtype for row in gt_rows):
            raise ValueError(f"{subtype} GT rows contain another subtype")
        subtype_positions[subtype] = {query_id: index for index, query_id in enumerate(query_ids)}
        subtype_ids[subtype] = ids
        subtype_dist[subtype] = dist
        subtype_gt_rows[subtype] = gt_rows

    combined_ids = np.empty((len(all_rows), args.topk), dtype=np.uint32)
    combined_dist = np.empty((len(all_rows), args.topk), dtype=np.float32)
    candidate_counts = np.empty((len(all_rows),), dtype=np.int64)
    for global_idx, row in enumerate(all_rows):
        subtype = str(row.get("subtype", "unknown"))
        local_idx = subtype_positions[subtype][str(row["query_id"])]
        subtype_local_indices[global_idx] = local_idx
        combined_ids[global_idx] = subtype_ids[subtype][local_idx]
        combined_dist[global_idx] = subtype_dist[subtype][local_idx]
        candidate_counts[global_idx] = int(subtype_gt_rows[subtype][local_idx].get("candidate_count", 0))

    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"top{args.topk}_gt": write_gt_ibin(args.output_root / f"gt.hybrid.top{args.topk}.ibin", combined_ids, args.overwrite),
        f"top{args.topk}_dist": write_dist_fbin(args.output_root / f"gt.hybrid.top{args.topk}.dist.fbin", combined_dist, args.overwrite),
        "jsonl": write_rows_jsonl(
            args.output_root / "gt.hybrid.rows.jsonl",
            all_rows,
            subtype_gt_rows,
            subtype_local_indices,
            args.overwrite,
        ),
    }
    if args.topk > 10:
        outputs["top10_gt"] = write_gt_ibin(args.output_root / "gt.hybrid.top10.ibin", combined_ids[:, :10], args.overwrite)
        outputs["top10_dist"] = write_dist_fbin(args.output_root / "gt.hybrid.top10.dist.fbin", combined_dist[:, :10], args.overwrite)

    summary = {
        "all_query_manifest": str(args.all_query_manifest),
        "by_subtype_root": str(args.by_subtype_root),
        "query_count": len(all_rows),
        "subtype_counts": dict(sorted(subtype_seen.items())),
        "candidate_count_min": int(candidate_counts.min()),
        "candidate_count_median": float(np.median(candidate_counts)),
        "candidate_count_max": int(candidate_counts.max()),
        "candidate_count_sum": int(candidate_counts.sum()),
        "topk": int(args.topk),
        "outputs": outputs,
    }
    (args.output_root / "gt.hybrid.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote combined GT rows={len(all_rows)} topk={args.topk} to {args.output_root}")


def combine_range(args: argparse.Namespace) -> None:
    summary = combine(args)
    print(
        f"Wrote combined range GT rows={summary['query_count']} "
        f"topk={summary['topk']} to {args.output_root}"
    )


def split(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.query_manifest)
    hinfo = inspect_hvecs(args.query_hvecs)
    if hinfo.rows != len(rows):
        raise ValueError(f"query.hvecs rows {hinfo.rows} != manifest rows {len(rows)}")
    records = hvecs_records(args.query_hvecs, hinfo)

    by_subtype: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_subtype[str(row.get("subtype", "unknown"))].append(index)

    names = [sanitize(subtype) for subtype in by_subtype]
    if len(set(names)) != len(names):
        raise ValueError("subtype names collide after filename sanitization")

    summary = {
        "source_manifest": str(args.query_manifest),
        "source_query_hvecs": str(args.query_hvecs),
        "total_rows": len(rows),
        "subtypes": {},
    }
    for subtype, indices in sorted(by_subtype.items()):
        out_dir = args.output_root / sanitize(subtype)
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = np.asarray(indices, dtype=np.int64)
        subtype_rows = [rows[int(i)] for i in idx]
        write_manifest(out_dir / "query_manifest.selected.jsonl", subtype_rows, args.overwrite)
        matrix = np.asarray(records["vec"][idx], dtype=np.float16)
        hvec_info = write_hvecs(out_dir / "query.hvecs", matrix, args.overwrite)
        subtype_summary = {
            "subtype": subtype,
            "rows": len(indices),
            "parent_indices_min": int(idx.min()),
            "parent_indices_max": int(idx.max()),
            "query_hvecs": hvec_info,
            "manifest": str(out_dir / "query_manifest.selected.jsonl"),
        }
        (out_dir / "split_summary.json").write_text(
            json.dumps(subtype_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary["subtypes"][subtype] = subtype_summary

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote split runtime to {args.output_root} subtypes={len(by_subtype)} rows={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, add_arguments, run in (
        ("split", add_split_arguments, split),
        ("combine-range", add_combine_range_arguments, combine_range),
        ("combine-hybrid", add_combine_hybrid_arguments, combine_hybrid),
    ):
        command_parser = commands.add_parser(command)
        add_arguments(command_parser)
        command_parser.set_defaults(run=run)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
