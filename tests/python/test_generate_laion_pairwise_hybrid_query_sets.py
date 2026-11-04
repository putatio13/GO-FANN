#!/usr/bin/env python3

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import scripts.laion.hybrid_queries as hybrid
from scripts.laion.predicates import range_predicate_mask


class GenerateLaionPairwiseHybridQuerySetsTest(unittest.TestCase):
    def test_metadata_cli_requires_a_single_aligned_pair(self):
        required = ["hybrid_queries", "--atomic-root", "atomic", "--string-root", "strings",
                    "--out-root", "out"]
        for extra in ([], ["--base-spmat", "labels"],
                      ["--base-spmat", "labels", "--base-range-fbin", "ranges", "--scales", "10M,100M"]):
            with patch("sys.argv", required + extra), self.assertRaises(SystemExit):
                hybrid.parse_args()
        with patch("sys.argv", required + ["--base-spmat", "labels", "--base-range-fbin", "ranges"]):
            args = hybrid.parse_args()
        self.assertIsNone(args.runtime_root)
        self.assertEqual(args.base_spmat, Path("labels"))

    def test_current_and_historical_metadata_layouts_keep_row_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scale = root / "100M"
            scale.mkdir()
            for name in ("base_duckdb_reordered.spmat", "base_duckdb_aligned.spmat", "base.spmat"):
                with self.subTest(name=name):
                    path = scale / name
                    path.write_bytes(np.asarray([4, 1, 0], "<i8").tobytes() + bytes(5 * 8))
                    self.assertEqual(hybrid.effective_row_scope(root, "100M", 0), 4)
                    self.assertEqual(hybrid.effective_row_scope(root, "100M", 2), 2)
                    path.unlink()
            with self.assertRaises(FileNotFoundError):
                hybrid.read_spmat_rows(root, "100M")
            # An explicit missing file must not silently fall back to a legacy file.
            (scale / "base_duckdb_reordered.spmat").write_bytes(np.asarray([4, 1, 0], "<i8").tobytes())
            with self.assertRaises(FileNotFoundError):
                hybrid.read_spmat_rows(root, "100M", root / "missing.spmat")

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("pyarrow", "duckdb", "nltk")),
                         "raw parquet preparation dependencies")
    def test_raw_metadata_outputs_feed_actual_hybrid_predicates(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from scripts.laion import metadata, string_index

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, images, output = root / "raw", root / "images", root / "metadata"
            source.mkdir(); images.mkdir()
            pq.write_table(pa.table({"caption": ["cat", "dog", "cat dog", "cat"],
                                     "width": [30, 10, 40, 20], "height": [2, 4, 1, 3],
                                     "similarity": [.1, .2, .3, .4]}), source / "metadata_0.parquet")
            np.save(images / "img_emb_0.npy", np.ones((4, 2), "<f2"))
            metadata.prepare(source, images, output, rows=4, batch_rows=2)
            labels, ranges = output / "base.spmat", output / "base_ranges.fbin"
            tokens = output / "token_map.parquet"
            postings, ngrams = root / "postings.bin", root / "ngrams.bin"
            with patch("sys.argv", ["string_index", "--base-spmat", str(labels), "--token-map", str(tokens),
                                    "--output-path", str(postings), "--max-row-id-exclusive", "4",
                                    "--ngram-output", str(ngrams),
                                    "--spmat-row-batch", "2"]):
                string_index.main()
            scope = hybrid.effective_row_scope(None, "100M", 0, labels)
            provider = hybrid.RowSetProvider(runtime_root=None, scale="100M", row_scope=scope,
                token_postings_bin=postings, token_ngram_index=ngrams, row_cache_entries=4,
                range_cache_entries=4, load_range_values=False, base_range_fbin=ranges)
            table = pq.read_table(tokens).to_pydict()
            cat = dict(zip(table["token"], table["token_id"]))["cat"]
            categorical = {"family": "categorical", "subtype": "single", "predicate_id": "cat",
                           "predicates": [{"token_id": cat}]}
            numeric = {"family": "range", "subtype": "width", "predicate_id": "wide",
                       "predicates": [{"column": 0, "op": "gt", "lo": 20}]}
            np.testing.assert_array_equal(provider.rows_for_record(categorical), [0, 2, 3])
            np.testing.assert_array_equal(provider.rows_for_record(numeric), [0, 2])
            np.testing.assert_array_equal(provider.range_intersection(numeric, provider.rows_for_record(categorical)), [0, 2])
            np.testing.assert_array_equal(provider.range_values[:, 0], [30, 10, 40, 20])
            self.assertEqual(provider.range_fbin, ranges)
            for label_name, range_name in (("base_duckdb_reordered.spmat", "base_ranges_reordered.fbin"),
                                           ("base_duckdb_aligned.spmat", "base_ranges_aligned.fbin"),
                                           ("base.spmat", "base_ranges.fbin")):
                with self.subTest(layout=range_name):
                    runtime = root / range_name
                    (runtime / "100M").mkdir(parents=True)
                    (runtime / "100M" / label_name).write_bytes(labels.read_bytes())
                    (runtime / "100M" / range_name).write_bytes(ranges.read_bytes())
                    legacy = hybrid.RowSetProvider(runtime_root=runtime, scale="100M",
                        row_scope=hybrid.effective_row_scope(runtime, "100M", 0),
                        token_postings_bin=postings, token_ngram_index=ngrams, row_cache_entries=4,
                        range_cache_entries=4, load_range_values=False)
                    np.testing.assert_array_equal(legacy.rows_for_record(numeric), [0, 2])
                    np.testing.assert_array_equal(legacy.rows_for_record(categorical), [0, 2, 3])

    def test_range_mask_honors_strict_and_inclusive_operators(self):
        values = np.asarray(
            [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, np.nan]],
            dtype=np.float32,
        )

        greater = range_predicate_mask(values, [{"column": 0, "op": "gt", "lo": 20.0}])
        less = range_predicate_mask(values, [{"column": 0, "op": "lt", "hi": 30.0}])
        inclusive = range_predicate_mask(
            values,
            [
                {
                    "column": 0,
                    "op": "between",
                    "lo": 20.0,
                    "hi": 30.0,
                    "inclusive_min": True,
                    "inclusive_max": True,
                }
            ],
        )
        exclusive = range_predicate_mask(
            values,
            [
                {
                    "column": 0,
                    "op": "between",
                    "lo": 20.0,
                    "hi": 40.0,
                    "inclusive_min": False,
                    "inclusive_max": False,
                }
            ],
        )

        np.testing.assert_array_equal(greater, [False, False, True, True])
        np.testing.assert_array_equal(less, [True, True, False, False])
        np.testing.assert_array_equal(inclusive, [False, True, True, False])
        np.testing.assert_array_equal(exclusive, [False, False, True, False])

    def test_query_manifest_is_accepted_as_atomic_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scale_root = root / "100M"
            scale_root.mkdir()
            record = {
                "query_id": "range-q0",
                "family": "range",
                "subtype": "width",
                "predicates": [{"column": 0, "op": "gt", "lo": 100.0}],
            }
            (scale_root / "query_manifest.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )

            loaded = hybrid.load_query_records(root, "100M")

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["predicate_id"], "range-q0")
        self.assertTrue(loaded[0]["predicate_key"])

    def test_reuse_subtype_allowlist_excludes_old_range_hybrids(self):
        categorical_exact = self.hybrid_record(
            "categorical.single__string.exact.and",
            10,
            [
                self.child("categorical", "single", "category"),
                self.child("string", "exact", "exact"),
            ],
        )
        categorical_range = self.hybrid_record(
            "categorical.single__range.and",
            11,
            [
                self.child("categorical", "single", "category"),
                self.child("range", "width_height", "range"),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scale_root = root / "100M"
            scale_root.mkdir()
            (scale_root / "query_manifest.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in (categorical_exact, categorical_range)),
                encoding="utf-8",
            )
            specs = [
                hybrid.PairSpec("categorical", "exact", "and"),
                hybrid.PairSpec("categorical", "range", "and"),
            ]

            reusable, _, summary = hybrid.load_reusable_records(
                reuse_root=root,
                scale="100M",
                specs=specs,
                target_per_subtype=5,
                range_subtypes={"width", "height", "width_height"},
                reuse_subtypes={"categorical.single__string.exact.and"},
            )

        self.assertEqual(list(reusable), ["categorical.single__string.exact.and"])
        self.assertEqual(summary["matched"], 1)

    def test_range_subtype_targets_preserve_source_ratio(self):
        records = []
        for subtype, count in (("width", 4), ("height", 4), ("width_height", 2)):
            records.extend(
                {"family": "range", "subtype": subtype, "query_id": f"{subtype}-{index}"}
                for index in range(count)
            )

        targets = hybrid.proportional_range_subtype_targets(records, 5000)

        self.assertEqual(targets, {"height": 2000, "width": 2000, "width_height": 1000})

    @staticmethod
    def child(family: str, subtype: str, predicate_id: str) -> dict:
        return {"family": family, "subtype": subtype, "predicate_id": predicate_id}

    @staticmethod
    def hybrid_record(subtype: str, row_id: int, children: list[dict]) -> dict:
        return {
            "query_id": f"query-{row_id}",
            "predicate_id": f"predicate-{row_id}",
            "query_vector_id": row_id,
            "subtype": subtype,
            "predicate": {"op": "and", "children": children},
        }


if __name__ == "__main__":
    unittest.main()
