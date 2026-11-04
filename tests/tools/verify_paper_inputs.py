#!/usr/bin/env python3
"""Verify frozen paper input identities; does not download or generate data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "docs/data-artifacts.json"


def manifest_entries(manifest: dict, archives: bool) -> list[dict]:
    """Select downloaded assets or extracted files from the package inventory."""
    if manifest.get("schema_version") != 1:
        raise ValueError("expected manifest schema_version 1")
    packages = manifest.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ValueError("manifest must contain a nonempty packages list")
    entries = []
    for package in packages:
        field = "assets" if archives else "files"
        if not isinstance(package, dict) or not isinstance(package.get(field), list) or not package[field]:
            raise ValueError(f"each package requires a nonempty {field} list")
        entries.extend(package[field])
    if archives:
        entries.append(manifest.get("checksum_asset"))
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each inventory entry must be an object")
        normalized.append({
            "path": entry.get("name" if archives else "path"),
            "bytes": entry.get("bytes"), "sha256": entry.get("sha256"),
        })
    return normalized


def verify_inputs(root: Path, manifest: dict, *, archives: bool = False) -> dict:
    """Hash complete files with bounded memory and report every missing/mismatched input."""
    artifacts = manifest_entries(manifest, archives)
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("root must be a directory")
    entries = []
    seen = set()
    for entry in artifacts:
        relative = entry["path"]
        if not isinstance(relative, str) or not relative:
            raise ValueError("artifact requires a nonempty path relative to root")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"artifact path must stay under root: {relative}")
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"artifact path escapes root: {relative}")
        if target in seen:
            raise ValueError(f"duplicate artifact: {relative}")
        seen.add(target)
        expected_size = entry.get("bytes")
        expected_hash = entry.get("sha256")
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError(f"invalid byte count: {relative}")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"invalid SHA256: {relative}")
        entries.append((entry, target, relative))

    results = []
    for entry, target, relative in entries:
        result = {"path": relative}
        try:
            with target.open("rb") as stream:
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(block)
                    digest.update(block)
            result.update(bytes=size, sha256=digest.hexdigest())
            result["status"] = (
                "verified" if size == entry["bytes"] and digest.hexdigest() == entry["sha256"]
                else "mismatch"
            )
        except FileNotFoundError:
            result["status"] = "missing"
        except OSError as exc:
            result.update(status="unreadable", error=str(exc))
        results.append(result)
    return {
        "workloads": manifest.get("workloads"),
        "scope": "archives" if archives else "files",
        "ok": all(r["status"] == "verified" for r in results),
        "verified_files": sum(r["status"] == "verified" for r in results),
        "artifacts": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory containing the manifest-relative input paths")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archives", action="store_true", help="verify downloaded archive parts and SHA256SUMS instead of extracted files")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        report = verify_inputs(args.root, manifest, archives=args.archives)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
