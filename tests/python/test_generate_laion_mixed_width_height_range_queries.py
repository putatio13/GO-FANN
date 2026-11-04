#!/usr/bin/env python3

import unittest
import json
import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import scripts.laion.range_queries as mixed_range
import scripts.laion.range_queries as operator_mixed
import scripts.laion.base as materialize


class MixedWidthHeightRangeTest(unittest.TestCase):
    def test_ratio_sum_preserves_near_tie_quota_across_python_versions(self):
        weights = operator_mixed.normalized_weights(
            {"low": 0.7, "medium": 0.2, "high": 0.1}, "selectivity")
        self.assertEqual(operator_mixed.proportional_counts(2, weights),
                         {"low": 1, "medium": 1, "high": 0})
        self.assertEqual(operator_mixed.proportional_counts(2000, weights),
                         {"low": 1400, "medium": 400, "high": 200})

    def test_calibrated_interval_contains_anchor_and_uses_exact_count(self):
        index = mixed_range.ValueIndex(
            dim=0,
            name="width",
            values=np.asarray([10, 20, 30, 40], dtype=np.float32),
            offsets=np.asarray([0, 2, 5, 9, 10], dtype=np.uint64),
            rows_by_value=np.arange(10, dtype=np.uint32),
        )

        interval = mixed_range.calibrate_single_interval(
            index=index,
            anchor_value=30,
            target_selectivity=0.4,
            row_count=10,
            min_count=1,
            max_count=10,
            position=0.5,
        )

        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertLessEqual(interval.lower, 30)
        self.assertGreaterEqual(interval.upper, 30)
        self.assertEqual(interval.actual_count, index.count(interval.lower, interval.upper))

    def test_operator_shape_counts_preserve_requested_margins(self):
        requested = {"width": 4, "height": 4, "width_height": 2}
        args = SimpleNamespace(
            width_count=requested["width"],
            height_count=requested["height"],
            and_count=requested["width_height"],
            between_predicate_ratio=0.60,
            gt_predicate_ratio=0.20,
            lt_predicate_ratio=0.20,
        )
        global_counts, single_counts = operator_mixed.operator_quotas(args)
        self.assertEqual(sum(global_counts.values()), 12)
        for shape in ("width", "height"):
            self.assertEqual(sum(single_counts[shape].values()), requested[shape])
        self.assertEqual(
            global_counts["between"] - sum(single_counts[s]["between"] for s in single_counts),
            2 * requested["width_height"],
        )

    def test_materializer_accepts_single_range_subtypes(self):
        self.assertIn(("range", "width"), materialize.RANGE_SUBTYPES)
        self.assertIn(("range", "height"), materialize.RANGE_SUBTYPES)
        self.assertIn(("range", "width_height"), materialize.RANGE_SUBTYPES)

    def test_empty_query_ranges_zeroes_aligned_padding(self):
        ranges = materialize.empty_query_ranges(2, 2)

        self.assertEqual(ranges.tobytes(), bytes(ranges.nbytes))

    def test_operator_quotas_include_and_between_terms(self):
        args = SimpleNamespace(
            width_count=4000,
            height_count=4000,
            and_count=2000,
            between_predicate_ratio=0.60,
            gt_predicate_ratio=0.20,
            lt_predicate_ratio=0.20,
        )

        global_counts, single_counts = operator_mixed.operator_quotas(args)

        self.assertEqual(global_counts, {"between": 7200, "gt": 2400, "lt": 2400})
        self.assertEqual(
            single_counts,
            {
                "width": {"between": 1600, "gt": 1200, "lt": 1200},
                "height": {"between": 1600, "gt": 1200, "lt": 1200},
            },
        )

    def test_operator_band_matrix_preserves_both_margins(self):
        matrix = operator_mixed.matrix_counts(
            {"between": 1600, "gt": 1200, "lt": 1200},
            {"low": 0.70, "medium": 0.20, "high": 0.10},
        )

        self.assertEqual(
            {operator: sum(row.values()) for operator, row in matrix.items()},
            {"between": 1600, "gt": 1200, "lt": 1200},
        )
        self.assertEqual(
            {
                band: sum(matrix[operator][band] for operator in operator_mixed.OPERATORS)
                for band in operator_mixed.BANDS
            },
            {"low": 2800, "medium": 800, "high": 400},
        )

    def test_tail_calibration_matches_strict_runtime_semantics(self):
        index = mixed_range.ValueIndex(
            dim=0,
            name="width",
            values=np.asarray([10, 20, 30, 40, 50], dtype=np.float32),
            offsets=np.asarray([0, 2, 5, 9, 12, 20], dtype=np.uint64),
            rows_by_value=np.arange(20, dtype=np.uint32),
        )
        band = operator_mixed.SelectivityBand("test", 0.20, 0.60, True, 1.0)

        greater = operator_mixed.calibrate_tail(index, "gt", 0.40, 20, band)
        less = operator_mixed.calibrate_tail(index, "lt", 0.40, 20, band)

        self.assertIsNotNone(greater)
        self.assertIsNotNone(less)
        assert greater is not None and less is not None
        self.assertEqual(greater.actual_count, 8)
        self.assertEqual(greater.lower, 40)
        self.assertTrue(operator_mixed.predicate_matches(50, greater))
        self.assertFalse(operator_mixed.predicate_matches(40, greater))
        self.assertEqual(less.actual_count, 9)
        self.assertEqual(less.upper, 40)
        self.assertTrue(operator_mixed.predicate_matches(30, less))
        self.assertFalse(operator_mixed.predicate_matches(40, less))

    def test_generated_record_replaces_stale_anchor_provenance(self):
        source = {
            "query_vector_id": 7,
            "anchor_row_id": 7,
            "anchor_sample_id": 3,
            "consumer": {"stale": True},
            "diagnostics": {"stale": True},
            "difficulty_band": "hard",
            "predicate_instance_rank": 12,
            "similarity_proxy": {"stale": True},
            "query_id": "source-query",
            "predicate_id": "source-predicate",
        }

        record = operator_mixed.make_record(
            source=source,
            shape="width",
            shape_index=0,
            predicates=[{"column": 0, "op": "gt", "lo": 10}],
            query_vector_id=99,
            actual_count=20,
            target_selectivity=0.2,
            row_count=100,
            selectivity_band="high",
        )

        self.assertEqual(record["anchor_row_id"], 99)
        self.assertEqual(record["anchor_source"], "full_100m_range_value_index")
        self.assertNotIn("anchor_sample_id", record)
        self.assertNotIn("consumer", record)
        self.assertNotIn("diagnostics", record)
        self.assertNotIn("difficulty_band", record)
        self.assertNotIn("predicate_instance_rank", record)
        self.assertNotIn("similarity_proxy", record)


class ExactRectangleCountsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = np.array([[1, 2], [1, 2], [2, 3], [3, 1], [4, 4], [5, 2]], dtype="<f4")
        self.base_path = self.root / "base.fbin"
        with self.base_path.open("wb") as handle:
            np.array(self.base.shape, dtype="<u4").tofile(handle)
            self.base.tofile(handle)
        self.indexes = []
        for dim in range(2):
            values, counts = np.unique(self.base[:, dim], return_counts=True)
            self.indexes.append(mixed_range.ValueIndex(
                dim, ("width", "height")[dim], values,
                np.r_[0, np.cumsum(counts)].astype(np.uint64),
                np.argsort(self.base[:, dim], kind="stable").astype(np.uint32)))
        self.records = [self.record("a", {"op": "between", "lo": 1, "hi": 3},
                                    {"op": "between", "lo": 2, "hi": 3})]
        self.manifest = self.root / "source.jsonl"
        self.manifest.write_text("".join(json.dumps(r) + "\n" for r in self.records))
        self.cache = self.root / "counts.json"

    @staticmethod
    def record(identifier, width, height):
        return {"predicate_id": identifier, "predicates": [dict(width, column=0), dict(height, column=1)]}

    def generate(self):
        return mixed_range.load_or_create_rectangle_counts(
            self.cache, self.manifest, self.base_path, len(self.base),
            self.records, self.indexes, self.base)

    def test_exact_counts_match_independent_oracle_including_prefix(self):
        cases = [
            ({"op": "gt", "lo": 1}, {"op": "lt", "hi": 4}, lambda x, y: x > 1 and y < 4),
            ({"op": "between", "lo": 1, "hi": 3}, {"op": "between", "lo": 2, "hi": 3},
             lambda x, y: 1 <= x <= 3 and 2 <= y <= 3),
            ({"op": "between", "lo": 1, "hi": 3, "inclusive_min": False},
             {"op": "between", "lo": 2, "hi": 3, "inclusive_max": False},
             lambda x, y: 1 < x <= 3 and 2 <= y < 3),
            ({"op": "gt", "lo": 9}, {"op": "lt", "hi": 0}, lambda x, y: False),
        ]
        for width, height, oracle in cases:
            for prefix in (0, 2, 5, 6):
                key = mixed_range.rectangle_key(self.record("a", width, height))
                self.assertEqual(mixed_range.count_rectangle(key, self.indexes, self.base, prefix, 2),
                                 sum(oracle(x, y) for x, y in self.base[:prefix]))

    def test_fresh_cache_deduplicates_rectangles_and_reuses_without_counting(self):
        self.records.append(dict(self.records[0], predicate_id="b"))
        self.manifest.write_text(json.dumps(self.records))
        with patch.object(mixed_range, "count_rectangle", wraps=mixed_range.count_rectangle) as count:
            self.assertEqual(self.generate(), {"a": 3, "b": 3})
            self.assertEqual(count.call_count, 1)
        before = self.cache.read_bytes()
        with patch.object(mixed_range, "count_rectangle", side_effect=AssertionError("must reuse")):
            self.assertEqual(self.generate(), {"a": 3, "b": 3})
        self.assertEqual(self.cache.read_bytes(), before)

    def test_legacy_cache_is_checked_without_rewriting(self):
        self.generate()
        payload = json.loads(self.cache.read_text())
        del payload["source_manifest_sha256"]
        self.cache.write_text(json.dumps(payload))
        before = self.cache.read_bytes()
        self.assertEqual(self.generate(), {"a": 3})
        self.assertEqual(self.cache.read_bytes(), before)
        self.records[0]["predicates"][0]["hi"] = 1
        with self.assertRaisesRegex(ValueError, "legacy.*do not match"):
            self.generate()

    def test_stale_manifest_and_base_are_rejected(self):
        self.generate()
        before = self.cache.read_bytes()
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "manifest contents"):
            self.generate()
        self.assertEqual(self.cache.read_bytes(), before)
        self.manifest.write_text(self.manifest.read_text()[:-1])
        stat = self.base_path.stat()
        os.utime(self.base_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        with self.assertRaisesRegex(ValueError, "mtime"):
            self.generate()

    def test_malformed_counts_and_duplicate_json_keys_are_rejected(self):
        self.generate()
        payload = json.loads(self.cache.read_text())
        for value in (-1, 7, 1.5, True, "3", None):
            payload["counts_by_predicate_id"]["a"] = value
            self.cache.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "counts must be integers"):
                self.generate()
        self.cache.write_text('{"row_count": 6, "row_count": 6}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.generate()

    def test_missing_ids_and_conflicting_predicates_are_rejected(self):
        self.generate()
        self.records.append(dict(self.records[0], predicate_id="b"))
        with self.assertRaisesRegex(ValueError, "misses"):
            self.generate()
        self.records = [self.records[0], self.record("a", {"op": "gt", "lo": 2},
                                                   {"op": "lt", "hi": 4})]
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.generate()

    def test_cli_requires_sources_and_defaults_to_fresh_counts(self):
        with patch("sys.argv", ["range_queries"]), self.assertRaises(SystemExit):
            mixed_range.parse_args()
        with patch("sys.argv", ["range_queries", "--source-manifest", str(self.manifest),
                                "--value-index-root", str(self.root), "--output-root", str(self.root / "out")]):
            args = mixed_range.parse_args()
        self.assertIsNone(args.source_rectangle_count_cache)
        self.assertEqual(str(args.base_range_fbin), "data/laion100m/range/base_range.fbin")
        self.assertEqual(args.seed, 20260727)
        self.assertEqual((args.width_count, args.height_count, args.and_count), (4000, 4000, 2000))
        self.assertEqual((args.low_band_ratio, args.medium_band_ratio, args.high_band_ratio), (.7, .2, .1))

    def test_fresh_generation_preserves_fixed_seed_manifest_bytes(self):
        base = np.array([(x, y) for x in range(100) for y in range(100)], dtype="<f4")
        with self.base_path.open("wb") as handle:
            np.array(base.shape, dtype="<u4").tofile(handle)
            base.tofile(handle)
        records = [dict(
            family="range", subtype="width_height", query_id=f"q{i}", predicate_id=f"p{i}",
            query_vector_id=i, predicates=[dict(column=0, op="between", lo=0, hi=i % 3),
                                           dict(column=1, op="between", lo=0, hi=99)]) for i in range(10)]
        self.manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
        output = self.root / "out"
        argv = ["range_queries", "--source-manifest", str(self.manifest),
                "--base-range-fbin", str(self.base_path), "--value-index-root", str(self.root / "index"),
                "--output-root", str(output), "--width-count", "4", "--height-count", "4", "--and-count", "2"]
        with patch("sys.argv", argv):
            mixed_range.main()
        # Captured by running the preceding generator against this same exact
        # count fixture; this covers selected anchors, predicates and row order.
        manifest = output / "100M/query_manifest.jsonl"
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(),
                         "d7e23bb5e7be5dc1cafd79b8f82250cce572192cbd903267d36ecd4123a1fbac")

        count_path = output / "source_rectangle_counts.json"
        before = count_path.read_bytes()
        with patch("sys.argv", argv + ["--overwrite"]), patch.object(
            mixed_range, "count_rectangle", side_effect=AssertionError("must reuse")):
            mixed_range.main()
        self.assertEqual(count_path.read_bytes(), before)
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(),
                         "d7e23bb5e7be5dc1cafd79b8f82250cce572192cbd903267d36ecd4123a1fbac")

    def test_index_bootstrap_and_stale_or_partial_index_rejection(self):
        root = self.root / "index"
        indexes = mixed_range.load_or_create_value_indexes(root, self.base)
        for expected, actual in zip(self.indexes, indexes):
            np.testing.assert_array_equal(actual.values, expected.values)
            np.testing.assert_array_equal(actual.offsets, expected.offsets)
            np.testing.assert_array_equal(actual.rows_by_value, expected.rows_by_value)
        changed = self.base.copy()
        changed[0, 0] = 99
        with self.assertRaisesRegex(ValueError, "does not match"):
            mixed_range.load_or_create_value_indexes(root, changed)
        rows = np.load(root / "width.rows_by_value.u32.npy")
        rows[1] = rows[0]
        np.save(root / "width.rows_by_value.u32.npy", rows)
        with self.assertRaisesRegex(ValueError, "row permutation"):
            mixed_range.load_or_create_value_indexes(root, self.base)
        (root / "height.offsets.u64.npy").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            mixed_range.load_or_create_value_indexes(root, self.base)



if __name__ == "__main__":
    unittest.main()
