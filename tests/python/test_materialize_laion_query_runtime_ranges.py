#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path



import scripts.laion.query as runtime


def range_child(column: int, op: str = "between") -> dict:
    predicate = {"column": column, "op": op}
    if op in {"between", "gt"}:
        predicate["lo"] = 10.0
    if op in {"between", "lt"}:
        predicate["hi"] = 20.0
    return {
        "family": "range",
        "subtype": "width" if column == 0 else "height",
        "predicates": [predicate],
    }


def hybrid_record(range_atom: dict, query_vector_id: int = 7) -> dict:
    return {
        "family": "hybrid",
        "subtype": "categorical.single__range.or",
        "query_vector_id": query_vector_id,
        "predicate": {
            "op": "or",
            "children": [
                {
                    "family": "categorical",
                    "subtype": "single",
                    "predicates": [{"op": "eq", "token_id": 3}],
                },
                range_atom,
            ],
        },
    }


class MaterializeHybridQueryRangesTest(unittest.TestCase):
    def test_extracts_range_from_canonical_hybrid_children(self) -> None:
        terms = runtime.selected_query_range_terms(hybrid_record(range_child(1)), 2)
        self.assertEqual(len(terms), 1)
        column, mode, lower, upper = terms[0]
        self.assertEqual((column, mode, lower), (1, 3, 10.0))
        self.assertGreater(upper, 20.0)

    def test_label_only_query_produces_no_active_range(self) -> None:
        record = {
            "family": "string",
            "subtype": "exact",
            "query_vector_id": 9,
            "predicates": [{"op": "eq", "token_id": 4}],
        }
        self.assertEqual(runtime.selected_query_range_terms(record, 2), [])

    def test_rejects_duplicate_or_out_of_range_dimensions(self) -> None:
        duplicate = hybrid_record(
            {
                "family": "range",
                "subtype": "width_height",
                "predicates": [
                    {"column": 0, "op": "gt", "lo": 1.0},
                    {"column": 0, "op": "lt", "hi": 2.0},
                ],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate range column"):
            runtime.selected_query_range_terms(duplicate, 2)
        with self.assertRaisesRegex(ValueError, "outside base range count"):
            runtime.selected_query_range_terms(hybrid_record(range_child(2)), 2)

    def test_selected_manifest_and_binary_rows_stay_aligned(self) -> None:
        records = [
            hybrid_record(range_child(0, "gt"), query_vector_id=11),
            {
                "family": "string",
                "subtype": "exact",
                "query_vector_id": 12,
                "predicates": [{"op": "eq", "token_id": 8}],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "query_manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            selected = runtime.load_selected_queries(
                manifest,
                families=set(),
                subtypes=set(),
                query_limit=0,
                per_subtype_limit=0,
                range_count=2,
            )
            output = root / "query_range.bin"
            info = runtime.write_query_range_bin(output, selected.queries, 2, False)
            self.assertEqual(info["rows"], 2)
            self.assertEqual(info["range_count"], 2)
            self.assertEqual(info["active_range_terms"], 1)
            header = runtime.np.fromfile(output, dtype="<u4", count=2)
            self.assertEqual(header.tolist(), [2, 2])


if __name__ == "__main__":
    unittest.main()
