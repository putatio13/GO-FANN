"""CPU fixtures for strict result/GT alignment, recall and native timings."""

import csv
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from scripts.evaluate import evaluate, UINT32_MAX


ROOT = Path(__file__).resolve().parents[2]
TIMING_FIELDS = ["phase", "run_index", "point_id", "wall_ms", "qps", "query_count", "result_path"]


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results.csv"
        self.gt = self.root / "gt.ivecs"
        self.timing = self.root / "timed_runs.csv"

    def predictions(self, rows):
        with self.results.open("w", newline="") as output:
            csv.writer(output).writerows(rows)

    def ivecs(self, rows):
        self.gt = self.root / "gt.ivecs"
        self.gt.write_bytes(b"".join(struct.pack(f"<{len(row) + 1}i", len(row), *row) for row in rows))

    def ibin(self, rows, distances=False):
        self.gt = self.root / "gt.ibin"
        width = len(rows[0])
        ids = [value & UINT32_MAX for row in rows for value in row]
        payload = struct.pack("<II", len(rows), width) + struct.pack(f"<{len(ids)}I", *ids)
        if distances:
            payload += struct.pack(f"<{len(ids)}f", *([0.25] * len(ids)))
        self.gt.write_bytes(payload)

    def run_evaluate(self, **kwargs):
        return evaluate(self.results, self.gt, k=kwargs.pop("k", 2), **kwargs)

    def timing_rows(self, rows=None):
        if rows is None:
            rows = [self.timed_row(0, 1000), self.timed_row(1, 2000)]
        with self.timing.open("w", newline="") as output:
            writer = csv.DictWriter(output, TIMING_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def timed_row(self, index, duration, **kwargs):
        row = dict(phase="timed", run_index=index, point_id="q64", wall_ms=duration,
                   qps=2000 / duration, query_count=2, result_path=str(self.results))
        row.update(kwargs)
        return row

    def test_unequal_and_empty_gt_recall(self):
        self.predictions([[1, 8], [3, -1], [9, UINT32_MAX]])
        self.ivecs([[1, 2], [3], []])
        report = self.run_evaluate(base_rows=10)
        self.assertEqual(report["hits"], 2)
        self.assertEqual(report["valid_gt"], 3)
        self.assertEqual(report["recall_at_2"], .75)
        self.assertEqual(report["zero_gt_queries"], 1)
        self.assertEqual(report["missing_prediction_slots"], 2)
        self.assertIsNone(report["timing"])

    def test_all_empty_gt_has_undefined_recall(self):
        self.predictions([[-1, UINT32_MAX], [1, 2]])
        for rows in ([[], []], [[-1, -1], [-1, -1]]):
            self.ivecs(rows)
            report = self.run_evaluate()
            self.assertIsNone(report["recall_at_2"])
            self.assertEqual(report["zero_gt_queries"], 2)
            self.assertNotIn("NaN", json.dumps(report, allow_nan=False))

    def test_binary_layouts_agree(self):
        self.predictions([[1, 8], [3, -1]])
        reports = []
        self.ivecs([[1, 2], [3, -1]])
        reports.append(self.run_evaluate())
        for distances in (False, True):
            self.ibin([[1, 2], [3, -1]], distances)
            reports.append(self.run_evaluate())
        self.assertTrue(all(report == reports[0] for report in reports))

    def test_top10_primary_metric_is_mean_per_query(self):
        self.predictions([list(range(5)) + list(range(20, 25)),
                          [10, 11] + [-1] * 8])
        self.ivecs([list(range(10)), [10, 11]])
        report = self.run_evaluate(k=10)
        self.assertEqual(report["recall_at_10"], .75)
        self.assertEqual({key for key in report if key.startswith("recall_")}, {"recall_at_10"})
        self.assertNotIn("recall_at_2", report)
        self.ivecs([list(range(10)), list(range(10, 20))])
        report = self.run_evaluate(k=10)
        self.assertEqual(report["recall_at_10"], .35)

    def test_first_k_positions_without_filling_missing(self):
        self.predictions([[1, 4, 3]])
        self.ivecs([[1, -1, 3, 4]])
        report = self.run_evaluate()
        self.assertEqual(report["hits"], 1)
        self.assertEqual(report["valid_gt"], 1)
        self.assertEqual(report["recall_at_2"], 1)

    def test_single_query_and_single_column_multiple_rows(self):
        for rows in ([[1]], [[1], [2], [3]]):
            self.predictions(rows)
            self.ivecs(rows)
            report = self.run_evaluate(k=1)
            self.assertEqual(report["query_count"], len(rows))
            self.assertEqual(report["recall_at_1"], 1)

    def test_prediction_shape_and_integer_errors(self):
        self.ivecs([[1, 2], [3, 4]])
        malformed = ["", "\n", "1,2\n", "1,2\n3,4\n5,6\n", "1,2\n3\n",
                     "1,2\n3,4,5\n", "1,2\n\n", "1,2\n3,\n", "1,2\n3,4.0\n",
                     "1,2\n3,1_0\n", "1,2\n3,nan\n", '1,2\n"3,4\n']
        for content in malformed:
            with self.subTest(content=content):
                self.results.write_text(content)
                with self.assertRaises((ValueError, csv.Error)):
                    self.run_evaluate()

    def test_duplicate_and_invalid_prediction_ids_including_tail(self):
        self.ivecs([[1, 2]])
        for row in ([1, 1], [-2, 1], [10, 1], [1, UINT32_MAX + 1], [1, 2, 1], [1, 2, 10]):
            with self.subTest(row=row):
                self.predictions([row])
                with self.assertRaises(ValueError):
                    self.run_evaluate(base_rows=10)

    def test_optional_base_rows_and_id_domain(self):
        self.predictions([[UINT32_MAX - 1, UINT32_MAX]])
        self.ibin([[UINT32_MAX - 1, UINT32_MAX]])
        self.assertEqual(self.run_evaluate()["recall_at_2"], 1)
        self.assertEqual(self.run_evaluate(base_rows=UINT32_MAX)["recall_at_2"], 1)
        for base_rows in (0, -1, UINT32_MAX + 1, 10):
            with self.assertRaises(ValueError):
                self.run_evaluate(base_rows=base_rows)
        self.predictions([[UINT32_MAX + 1, -1]])
        with self.assertRaises(ValueError):
            self.run_evaluate()

    def test_duplicate_and_invalid_gt_ids_including_tail(self):
        self.predictions([[1, 2]])
        for row in ([1, 1], [-2, 1], [10, 1], [1, 2, 1], [1, 2, 10]):
            with self.subTest(row=row):
                self.ivecs([row])
                with self.assertRaises(ValueError):
                    self.run_evaluate(base_rows=10)

    def test_ibin_corruption(self):
        self.predictions([[1, 2]])
        self.ibin([[1, 2]])
        good = self.gt.read_bytes()
        malformed = [good[:4], good[:-1], good + b"x", good + b"\0" * 12,
                     struct.pack("<II", 0, 2), struct.pack("<II", 1, 0),
                     struct.pack("<III", 1, 1, 1)]
        self.ibin([[1, 2]], distances=True)
        malformed.append(self.gt.read_bytes()[:-1])
        for content in malformed:
            with self.subTest(content=content):
                self.gt.write_bytes(content)
                with self.assertRaises(ValueError):
                    self.run_evaluate()

    def test_ivecs_corruption_and_extra_rows(self):
        self.predictions([[1, 2]])
        self.ivecs([[1, 2]])
        good = self.gt.read_bytes()
        for content in (b"", b"\0", struct.pack("<i", -1), good[:-1], good + b"\0",
                        good + struct.pack("<i", 0), struct.pack("<i", 2**31 - 1)):
            with self.subTest(content=content):
                self.gt.write_bytes(content)
                with self.assertRaises(ValueError):
                    self.run_evaluate()

    def test_bad_k_and_format(self):
        self.predictions([[1, 2]])
        self.ivecs([[1, 2]])
        for k in (0, -1, 3):
            with self.assertRaises(ValueError):
                self.run_evaluate(k=k)
        wrong = self.root / "gt.bin"
        wrong.write_bytes(self.gt.read_bytes())
        with self.assertRaisesRegex(ValueError, "extension"):
            evaluate(self.results, wrong)

    def setup_timed_fixture(self):
        self.predictions([[1, 2], [3, 4]])
        self.ivecs([[1, 2], [3, 4]])

    def test_timing_arithmetic_mean_sample_std_and_warmups(self):
        self.setup_timed_fixture()
        self.timing_rows([self.timed_row(0, 1, phase="warmup"),
                          self.timed_row(0, 1000), self.timed_row(1, 2000)])
        report = self.run_evaluate(timed_runs=self.timing)["timing"]
        self.assertEqual(report["qps_runs"], [2, 1])
        self.assertEqual(report["wall_ms_runs"], [1000, 2000])
        self.assertEqual(report["qps_mean"], 1.5)
        self.assertAlmostEqual(report["qps_std"], math.sqrt(.5))
        self.assertEqual(report["timed_run_count"], 2)
        self.timing_rows([self.timed_row(0, 1000)])
        self.assertIsNone(self.run_evaluate(timed_runs=self.timing)["timing"]["qps_std"])

    def test_invalid_timing_records(self):
        self.setup_timed_fixture()
        variants = [dict(phase="build"), dict(run_index="0.5"), dict(run_index=-1),
                    dict(query_count=3), dict(result_path=""), dict(result_path="other.csv"),
                    dict(point_id=""), dict(wall_ms=0), dict(wall_ms="nan"),
                    dict(wall_ms="oops"), dict(qps="inf"), dict(qps=-1), dict(qps=2001)]
        for override in variants:
            with self.subTest(override=override):
                self.timing_rows([self.timed_row(0, 1, **override)])
                with self.assertRaises(ValueError):
                    self.run_evaluate(timed_runs=self.timing)
        for rows in ([], [self.timed_row(0, 1, phase="warmup")],
                     [self.timed_row(0, 1), self.timed_row(0, 2)],
                     [self.timed_row(0, 1), self.timed_row(1, 2, point_id="q128")]):
            self.timing_rows(rows)
            with self.assertRaises(ValueError):
                self.run_evaluate(timed_runs=self.timing)

    def test_malformed_timing_schema(self):
        self.setup_timed_fixture()
        self.timing_rows()
        good = self.timing.read_text()
        for content in ("", "phase,run_index\ntimed,0\n", good.replace("phase,", "qps,", 1),
                        good + "timed,2\n", good.replace("q64,1000", "q64,extra,1000", 1)):
            with self.subTest(content=content):
                self.timing.write_text(content)
                with self.assertRaises(ValueError):
                    self.run_evaluate(timed_runs=self.timing)

    def cli(self, *extra, cwd=ROOT):
        return subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"),
                               "--results", str(self.results), "--ground-truth", str(self.gt),
                               "--k", "2", *extra], cwd=cwd, text=True, capture_output=True)

    def test_cli_json_and_failure_has_no_summary(self):
        self.setup_timed_fixture()
        result = self.cli("--base-rows", "10")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["hits"], 4)
        self.assertEqual(json.loads(result.stdout)["recall_at_2"], 1)
        self.timing_rows([self.timed_row(0, 1, query_count=3)])
        result = self.cli("--timed-runs", str(self.timing))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(str(self.timing), result.stderr)
        self.assertIn("row 2", result.stderr)

    def test_relative_timing_path_and_relocated_artifacts(self):
        self.setup_timed_fixture()
        self.timing_rows([self.timed_row(0, 1000, result_path="results.csv")])
        valid = self.cli("--timed-runs", str(self.timing), cwd=self.root)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        wrong_cwd = self.cli("--timed-runs", str(self.timing))
        self.assertNotEqual(wrong_cwd.returncode, 0)
        self.timing_rows([self.timed_row(0, 1000)])
        relocated = self.root / "relocated" / "results.csv"
        relocated.parent.mkdir()
        relocated.write_bytes(self.results.read_bytes())
        self.results = relocated
        mismatch = self.cli("--timed-runs", str(self.timing))
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertEqual(mismatch.stdout, "")
        self.assertEqual(self.cli().returncode, 0)


if __name__ == "__main__":
    unittest.main()
