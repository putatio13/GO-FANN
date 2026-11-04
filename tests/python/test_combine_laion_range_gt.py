#!/usr/bin/env python3

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

import scripts.laion.assemble as combine_range
from scripts.laion.categorical_gt import write_dist_fbin, write_gt_ibin


class CombineLaionRangeGtTest(unittest.TestCase):
    def test_combines_interleaved_subtypes_by_query_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            all_rows = [
                {"query_id": "w0", "query_vector_id": 10, "subtype": "width", "candidate_count": 3},
                {"query_id": "h0", "query_vector_id": 20, "subtype": "height", "candidate_count": 4},
                {"query_id": "w1", "query_vector_id": 30, "subtype": "width", "candidate_count": 5},
            ]
            all_manifest = root / "all.jsonl"
            self._write_jsonl(all_manifest, all_rows)
            self._write_subtype(
                root,
                "width",
                [all_rows[2], all_rows[0]],
                np.asarray([[30, 31], [10, 11]], dtype=np.uint32),
                np.asarray([[0.3, 0.4], [0.1, 0.2]], dtype=np.float32),
                np.asarray([5, 3], dtype=np.int64),
            )
            self._write_subtype(
                root,
                "height",
                [all_rows[1]],
                np.asarray([[20, 21]], dtype=np.uint32),
                np.asarray([[0.5, 0.6]], dtype=np.float32),
                np.asarray([4], dtype=np.int64),
            )
            output = root / "combined"

            summary = combine_range.combine(
                Namespace(
                    all_query_manifest=all_manifest,
                    by_subtype_root=root,
                    output_root=output,
                    gt_tag="gt.range",
                    topk=2,
                    num_points=100,
                    overwrite=False,
                )
            )

            ids = combine_range.read_matrix(output / "gt.range.top2.ibin", np.dtype("<u4"))
            self.assertEqual(ids.tolist(), [[10, 11], [20, 21], [30, 31]])
            self.assertEqual(np.load(output / "gt.range.candidate_counts.npy").tolist(), [3, 4, 5])
            self.assertEqual(summary["subtype_counts"], {"height": 1, "width": 2})

    def test_top10_combination_writes_each_output_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"query_id": "w0", "query_vector_id": 0,
                     "subtype": "width", "candidate_count": 10}]
            manifest = root / "all.jsonl"
            self._write_jsonl(manifest, rows)
            ids = np.arange(10, dtype=np.uint32).reshape(1, 10)
            distances = np.arange(10, dtype=np.float32).reshape(1, 10)
            self._write_subtype(root, "width", rows, ids, distances, np.array([10]))
            output = root / "combined"
            summary = combine_range.combine(Namespace(
                all_query_manifest=manifest, by_subtype_root=root,
                output_root=output, gt_tag="gt.range", topk=10,
                num_points=100, overwrite=False,
            ))
            actual = combine_range.read_matrix(output / "gt.range.top10.ibin", np.dtype("<u4"))
            np.testing.assert_array_equal(actual, ids)
            self.assertEqual(summary["topk"], 10)
            self.assertTrue((output / "gt.range.summary.json").is_file())

    def test_rejects_negative_squared_l2_distance(self):
        with self.assertRaisesRegex(ValueError, "negative squared L2"):
            combine_range.validate_gt(
                np.asarray([[1, 2]], dtype=np.uint32),
                np.asarray([[-1e-6, 0.1]], dtype=np.float32),
                num_points=100,
            )

    def test_rejects_duplicate_ids_within_row(self):
        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            combine_range.validate_gt(
                np.asarray([[1, 1]], dtype=np.uint32),
                np.asarray([[0.0, 0.1]], dtype=np.float32),
                num_points=100,
            )

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def _write_subtype(
        self,
        root: Path,
        subtype: str,
        rows: list[dict],
        ids: np.ndarray,
        distances: np.ndarray,
        candidate_counts: np.ndarray,
    ) -> None:
        subtype_root = root / subtype
        self._write_jsonl(subtype_root / "runtime/query_manifest.flashanns.jsonl", rows)
        gt_root = subtype_root / "gt"
        gt_root.mkdir(parents=True, exist_ok=True)
        write_gt_ibin(gt_root / f"gt.range.top{ids.shape[1]}.ibin", ids, False)
        write_dist_fbin(gt_root / f"gt.range.top{ids.shape[1]}.dist.fbin", distances, False)
        np.save(gt_root / "gt.range.candidate_counts.npy", candidate_counts)
        np.save(gt_root / "gt.range.chosen_dims.npy", np.zeros(len(rows), dtype=np.int8))


if __name__ == "__main__":
    unittest.main()
