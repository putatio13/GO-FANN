#!/usr/bin/env python3
"""Check the generated range posting binary against the runtime format contract."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HEADER = struct.Struct("<8sIIQQII")
DESCRIPTOR = struct.Struct("<IIQQQQQQQQ")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="flashanns-range-posting-generator-") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "source"
        source.mkdir()
        dimensions = {
            "width": (
                np.asarray([1.0, 2.0, 4.0], dtype="<f4"),
                np.asarray([0, 2, 4, 6], dtype="<u8"),
                np.asarray([1, 5, 0, 4, 2, 3], dtype="<u4"),
            ),
            "height": (
                np.asarray([10.0, 20.0], dtype="<f4"),
                np.asarray([0, 3, 6], dtype="<u8"),
                np.asarray([2, 3, 5, 0, 1, 4], dtype="<u4"),
            ),
        }
        for name, (values, offsets, sample_ids) in dimensions.items():
            np.save(source / f"{name}.unique_values.f32.npy", values)
            np.save(source / f"{name}.offsets.u64.npy", offsets)
            np.save(source / f"{name}.rows_by_value.u32.npy", sample_ids)

        output = tmp_path / "posting.bin"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.laion.range_index",
                "postings",
                "--source-root",
                str(source),
                "--dimensions",
                "width,height",
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        payload = output.read_bytes()
        magic, version, header_bytes, file_bytes, row_count, range_count, reserved = (
            HEADER.unpack_from(payload)
        )
        assert magic == b"FARPIDX1"
        assert version == 1 and reserved == 0
        assert file_bytes == len(payload) and row_count == 6 and range_count == 2
        assert header_bytes == HEADER.size + 2 * DESCRIPTOR.size

        for range_index, (name, expected) in enumerate(dimensions.items()):
            descriptor = DESCRIPTOR.unpack_from(payload, HEADER.size + range_index * DESCRIPTOR.size)
            (
                stored_index,
                stored_reserved,
                unique_count,
                stored_rows,
                values_offset,
                offsets_offset,
                ids_offset,
                values_bytes,
                offsets_bytes,
                ids_bytes,
            ) = descriptor
            values, offsets, sample_ids = expected
            assert stored_index == range_index and stored_reserved == 0
            assert unique_count == values.size and stored_rows == sample_ids.size
            assert payload[values_offset : values_offset + values_bytes] == values.tobytes()
            assert payload[offsets_offset : offsets_offset + offsets_bytes] == offsets.tobytes()
            assert payload[ids_offset : ids_offset + ids_bytes] == sample_ids.tobytes()

        summary = json.loads(output.with_suffix(".bin.summary.json").read_text())
        assert summary["row_count"] == 6
        assert [entry["name"] for entry in summary["dimensions"]] == ["width", "height"]


if __name__ == "__main__":
    main()
