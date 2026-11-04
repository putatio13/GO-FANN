#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path

import numpy as np


from scripts.laion import range_index as range_artifacts


class RangeHierarchyGeneratorTest(unittest.TestCase):
    def test_accepts_integer_multiple_levels(self) -> None:
        self.assertEqual(range_artifacts.parse_levels("32,256"), [32, 256])

    def test_rejects_nondivisible_levels(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            range_artifacts.parse_levels("32,250")

    def test_coarse_edges_are_nested_in_finest_edges(self) -> None:
        counts = {float(value): 1 for value in range(256)}
        stats = {"min": 0.0, "max": 255.0}
        edges, bucket_counts = range_artifacts.build_range_edges(
            stats, counts, [32, 256], "balanced_unique"
        )
        np.testing.assert_array_equal(edges[32], edges[256][::8])
        np.testing.assert_array_equal(bucket_counts[32], np.full(32, 8))
        np.testing.assert_array_equal(bucket_counts[256], np.ones(256))


if __name__ == "__main__":
    unittest.main()
