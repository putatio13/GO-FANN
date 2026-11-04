import csv
import importlib.util
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("reproduce_plot", ROOT / "tools/reproduce/plot.py")
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


class ReproducePlotTests(unittest.TestCase):
    def test_window_aggregation_weights_wall_time(self):
        shares = PLOT.load_windows(ROOT / "tools/reproduce/reference/window.csv")
        self.assertEqual(len(shares), 8)
        for values in shares.values():
            self.assertAlmostEqual(sum(values), 100, places=5)
        fields = [field for _, group in PLOT.WINDOW_GROUPS for field in group]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "window.csv"
            rows = []
            for workload, _ in PLOT.WINDOW_WORKLOADS:
                for run, wall, search in [(0, 100, 100), (1, 300, 0)]:
                    row = dict.fromkeys(fields, 0)
                    row.update(workload=workload, q=32, run=run, wall_ms=wall,
                               graph_phase_ms=search, cpu_rerank_exposed_ms=wall-search)
                    rows.append(row)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            for values in PLOT.load_windows(path).values():
                self.assertEqual(values, [25, 0, 75, 0])
            rows[0]["graph_phase_ms"] = 101
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "must sum"):
                PLOT.load_windows(path)

    def test_window_subset_keeps_full_input_validation(self):
        with (ROOT / "tools/reproduce/reference/window.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "window.csv"
            def write(values):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(values)
            subset = [row for row in rows if row["workload"] == "yfcc10m"]
            write(subset)
            actual = PLOT.load_windows(path, ["yfcc10m"])
            self.assertEqual(list(actual), ["yfcc10m"])
            self.assertEqual(actual["yfcc10m"], PLOT.load_windows(ROOT / "tools/reproduce/reference/window.csv")["yfcc10m"])
            with self.assertRaisesRegex(ValueError, "missing runs"):
                PLOT.load_windows(path)
            for selection in ([], ["unknown"], ["yfcc10m", "yfcc10m"], ["hybrid"]):
                with self.subTest(selection=selection), self.assertRaises(ValueError):
                    PLOT.load_windows(path, selection)
            unselected = next(row for row in rows if row["workload"] == "hybrid")
            for invalid in (subset + [unselected, unselected],
                            subset + [unselected | {"wall_ms": "nan"}],
                            subset + [unselected | {"workload": "unknown"}]):
                write(invalid)
                with self.assertRaises(ValueError):
                    PLOT.load_windows(path, ["yfcc10m"])

    def test_window_subset_cli_and_renderer(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib not installed")
        shares = PLOT.load_windows(ROOT / "tools/reproduce/reference/window.csv", ["hybrid"])
        figure = PLOT.render_windows(shares)
        try:
            ax = figure.axes[0]
            self.assertEqual([text.get_text() for text in ax.get_yticklabels()], ["LAION hybrid"])
            self.assertEqual(len(ax.patches), 4)
        finally:
            plt.close(figure)
        with tempfile.TemporaryDirectory() as root:
            command = [sys.executable, str(ROOT / "tools/reproduce/plot.py"), "--figure", "window",
                       "--workloads", "hybrid", "--output-dir", root]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(root) / "window.pdf").is_file())
            self.assertTrue((Path(root) / "window.png").is_file())
            command[command.index("window")] = "main"
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("only supported", result.stderr)

    def test_reference_panel_and_curve_counts(self):
        expected = {"main": (419, 8, 51), "planning": (100, 3, 12), "penalty": (45, 3, 9), "placement": (30, 3, 6), "overlap": (45, 3, 9)}
        for figure, (points, panels, lines) in expected.items():
            with self.subTest(figure=figure):
                curves = PLOT.load_curves(ROOT / f"tools/reproduce/reference/{figure}.csv", figure)
                self.assertEqual(len(curves), panels)
                self.assertEqual(sum(len(panel) for panel in curves.values()), lines)
                self.assertEqual(sum(len(line) for panel in curves.values() for line in panel.values()), points)

    def test_renderer_preserves_every_point_including_duplicates(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("Matplotlib not installed")
        curves = {key: {"test": [(0.9, 20), (0.8, 30), (0.9, 20)]}
                  for key, _ in PLOT.PANELS["planning"]}
        figure = PLOT.render(curves, "planning")
        try:
            for ax in figure.axes:
                self.assertEqual(list(ax.lines[0].get_xdata()), [0.8, 0.9, 0.9])
                self.assertEqual(list(ax.lines[0].get_ydata()), [30, 20, 20])
        finally:
            plt.close(figure)

    def test_invalid_measurements_and_missing_panels_fail(self):
        rows = [{"family": key, "variant": "test", "recall_at_10": "0.9", "qps_mean": "20"}
                for key, _ in PLOT.PANELS["planning"]]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "planning.csv"
            def write(values):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(values)
            for key, value in (("recall_at_10", "nan"), ("recall_at_10", "1.1"),
                               ("qps_mean", "inf"), ("qps_mean", "0"),
                               ("qps_mean", "bad"), ("family", "unknown")):
                with self.subTest(key=key, value=value):
                    write([{**rows[0], key: value}, *rows[1:]])
                    with self.assertRaises(ValueError):
                        PLOT.load_curves(path, "planning")
            write(rows[:1])
            with self.assertRaisesRegex(ValueError, "missing panels"):
                PLOT.load_curves(path, "planning")


if __name__ == "__main__":
    unittest.main()
