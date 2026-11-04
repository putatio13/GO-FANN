#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import scripts.laion.range_gt as range_gt


class GenerateLaionWidthHeightRangeGtTest(unittest.TestCase):
    def test_generation_only_accepts_custom_queries(self):
        args = SimpleNamespace(
            plan_only=False,
            skip_recall=False,
            query_hvecs=Path("custom/query.hvecs"),
            query_range_bin=Path("custom/query_range.bin"),
            search_results_csv=None,
        )

        range_gt.validate_recall_inputs(args)

    def test_skip_recall_accepts_custom_queries(self):
        args = SimpleNamespace(
            plan_only=False,
            skip_recall=True,
            query_hvecs=Path("custom/query.hvecs"),
            query_range_bin=Path("custom/query_range.bin"),
            search_results_csv=None,
        )

        range_gt.validate_recall_inputs(args)

    def test_explicit_search_results_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                plan_only=False,
                skip_recall=False,
                query_hvecs=Path("custom/query.hvecs"),
                query_range_bin=Path("custom/query_range.bin"),
                search_results_csv=Path(tmp) / "missing.csv",
            )

            with self.assertRaises(FileNotFoundError):
                range_gt.validate_recall_inputs(args)


if __name__ == "__main__":
    unittest.main()
