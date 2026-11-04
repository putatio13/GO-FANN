import csv
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("reproduce_profile", ROOT / "tools/reproduce/profile.py")
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def event(name, start, end, **extra):
    return dict(schema_version=1, backend="pq", point_id="b1_q24_f100", phase="timed",
                queue=24, run=0, kind="span", name=name, start_ns=start, end_ns=end, **extra)


def complete(events):
    return {key: events[0][key] for key in
            ("schema_version", "backend", "point_id", "phase", "queue", "run")} | {
                "kind": "complete", "event_count": len(events)}


def fixture_events():
    # 100 ns total, all eight output categories, two overlapping graph spans.
    return [event("provider_search_window", 100, 200),
            event("graph_prepare", 95, 110),
            event("graph_launch_to_complete", 110, 140),
            event("graph_launch_to_complete", 120, 135),
            event("highfreq_cpu_rerank_batch", 130, 160),
            event("lowfreq_kernel_launch_to_complete", 150, 170),
            event("lowfreq_h2d", 170, 180),
            event("merge_routes_to_original", 180, 190)]


def write_capture(root, events=None, rows=None):
    root = Path(root)
    events = fixture_events() if events is None else events
    trace = root / "stages.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in [*events, complete(events)]))
    timing = root / "run.timed_runs.csv"
    rows = rows if rows is not None else [dict(point_id="b1_q24_f100", phase="timed",
                                             run_index=0, wall_ms=0.0001,
                                             qps=1e10, query_count=1)]
    with timing.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return trace, timing


class ReproduceProfileTests(unittest.TestCase):
    def test_hand_calculated_exclusive_partition_and_cli(self):
        with tempfile.TemporaryDirectory() as root:
            trace, timing = write_capture(root)
            output = Path(root) / "window.csv"
            command = [sys.executable, str(ROOT / "tools/reproduce/profile.py"),
                       "--input", "yfcc10m", str(trace), str(timing), "--output", str(output)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open() as stream:
                row = next(csv.DictReader(stream))
            expected = dict(graph_phase=20, posting_phase=10, cpu_rerank_exposed=10,
                            gpu_cpu_rerank_overlap=20, explicit_copy=10, result_merge=10,
                            prepare=10, other_sync_cleanup=10)
            for key, value in expected.items():
                self.assertAlmostEqual(float(row[key + "_ms"]) * 1e6, value)
            self.assertEqual(row["q"], "24")
            self.assertEqual(row["run"], "0")
            again = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(again.returncode, 2)
            self.assertIn("already exists", again.stderr)

    def test_random_intervals_against_independent_tick_oracle(self):
        rng = random.Random(4291)
        names = list(PROFILE.SPAN_CATEGORIES) + ["diagnostic_only"]
        for case in range(500):
            events = [event("graph_launch_to_complete", 10, 11)]
            for _ in range(rng.randrange(1, 40)):
                start = rng.randrange(0, 61)
                events.append(event(rng.choice(names), start, rng.randrange(start, 62)))
            rng.shuffle(events)
            expected = dict.fromkeys(PROFILE.CATEGORIES, 0)
            for tick in range(10, 50):
                active = {e["name"] for e in events if e["start_ns"] <= tick < e["end_ns"]}
                graph = "graph_launch_to_complete" in active
                posting = bool(active & {"lowfreq_kernel_launch_to_complete", "range_posting_kernel_launch_to_complete"})
                cpu = bool(active & {"highfreq_cpu_rerank_batch", "lowfreq_cpu_finalize", "range_posting_cpu_rerank"})
                if cpu and (graph or posting):
                    category = "gpu_cpu_rerank_overlap"
                elif graph:
                    category = "graph_phase"
                elif posting:
                    category = "posting_phase"
                elif cpu:
                    category = "cpu_rerank_exposed"
                elif active & {"graph_result_d2h", "graph_distance_d2h", "lowfreq_h2d", "lowfreq_shortlist_d2h", "lowfreq_distance_d2h", "range_posting_d2h"}:
                    category = "explicit_copy"
                elif active & {"merge_routes_to_original", "final_route_merge", "dual_graph_final_merge"}:
                    category = "result_merge"
                elif active & {"lowfreq_prepare", "iteration_prepare", "graph_prepare", "lowfreq_candidate_prepare", "iteration_prepare_outside_search_window"}:
                    category = "prepare"
                else:
                    category = "other_sync_cleanup"
                expected[category] += 1
            with self.subTest(case=case):
                actual = PROFILE.partition(events, 10, 50)
                self.assertEqual(actual, expected)
                self.assertEqual(sum(actual.values()), 40)

    def test_adjacent_nested_zero_and_clipped_spans(self):
        events = [event("graph_launch_to_complete", 0, 20),
                  event("graph_launch_to_complete", 3, 5),
                  event("highfreq_cpu_rerank_batch", 5, 5),
                  event("highfreq_cpu_rerank_batch", 20, 30),
                  event("lowfreq_h2d", 30, 100)]
        totals = PROFILE.partition(events, 10, 40)
        self.assertEqual([totals[k] for k in ("graph_phase", "cpu_rerank_exposed", "explicit_copy")], [10, 10, 10])
        self.assertEqual(sum(totals.values()), 30)

    def test_dual_outer_window_contains_providers_and_merge(self):
        events = [event("dual_graph_search_window", 100, 300),
                  event("provider_search_window", 100, 220),
                  event("provider_search_window", 150, 250),
                  event("graph_launch_to_complete", 100, 250),
                  event("dual_graph_final_merge", 250, 290)]
        self.assertEqual(PROFILE.search_window(events, 0.0002), (100, 300))
        totals = PROFILE.partition(events, 100, 300)
        self.assertEqual(totals["graph_phase"], 150)
        self.assertEqual(totals["result_merge"], 40)
        self.assertEqual(totals["other_sync_cleanup"], 10)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            PROFILE.search_window(events[1:], 0.0002)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            PROFILE.search_window(events + [events[0]], 0.0002)
        with self.assertRaisesRegex(ValueError, "outside"):
            PROFILE.search_window(events + [event("provider_search_window", 0, 300)], 0.0002)

    def test_exact_endpoints_are_not_synthesized_from_rounded_wall(self):
        events = [event("provider_search_window", 100, 1_000_100),
                  event("graph_launch_to_complete", 100, 1_000_100)]
        self.assertEqual(PROFILE.search_window(events, 1.02), (100, 1_000_100))
        with self.assertRaisesRegex(ValueError, "duration disagrees"):
            PROFILE.search_window(events, 1.06)
        counter = event("reported_search_wall_ms", 1_000_200, 1_000_200)
        counter.update(kind="counter", value=2)
        with self.assertRaisesRegex(ValueError, "counter disagrees"):
            PROFILE.search_window(events + [counter], 1)
        counter["value"] = 1
        self.assertEqual(PROFILE.search_window(events + [counter], 1), (100, 1_000_100))
        with self.assertRaisesRegex(ValueError, "duplicate reported"):
            PROFILE.search_window(events + [counter, counter], 1)

    def test_no_coarse_window_or_unclassified_empty_bar(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            PROFILE.search_window([event("dual_graph_iteration", 0, 100)], 0.0001)
        with self.assertRaisesRegex(ValueError, "no positive classified"):
            PROFILE.partition([event("diagnostic_only", 0, 100)], 0, 100)

    def test_schema_rejects_malformed_identity_and_values(self):
        bad = [("schema_version", 0), ("schema_version", True), ("backend", "other"),
               ("point_id", "unknown"), ("point_id", "rabitq_b1_q24_f100"),
               ("queue", 25), ("queue", True), ("run", -1), ("run", 0.5),
               ("phase", "warmup"), ("kind", "unknown"), ("name", ""),
               ("start_ns", -1), ("start_ns", 201), ("start_ns", 1.0),
               ("end_ns", float("nan")), ("end_ns", 1 << 63), ("context", -1)]
        for key, value in bad:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                PROFILE.validate_event(event("graph_prepare", 100, 200) | {key: value})
        for value in (float("nan"), float("inf"), -1, True, "2"):
            counter = event("counter", 100, 100)
            counter.update(kind="counter", value=value)
            with self.subTest(counter=value), self.assertRaises(ValueError):
                PROFILE.validate_event(counter)
        counter = event("counter", 100, 101)
        counter.update(kind="counter", value=1)
        with self.assertRaisesRegex(ValueError, "timestamps must be equal"):
            PROFILE.validate_event(counter)

    def test_completion_truncation_order_and_duplicate_guards(self):
        events = fixture_events()
        valid = [*events, complete(events)]
        cases = [events, valid + [valid[-1]], valid + [events[0]],
                 [*events, complete(events) | {"event_count": len(events) + 1}],
                 [events[0], events[1] | {"run": 1}],
                 [complete(events) | {"event_count": 0}]]
        with tempfile.TemporaryDirectory() as root:
            trace = Path(root) / "trace.jsonl"
            for records in cases:
                with self.subTest(records=records):
                    trace.write_text("".join(json.dumps(row) + "\n" for row in records))
                    with self.assertRaises(ValueError):
                        list(PROFILE.completed_iterations(trace))
            for text in ("", "{\n", "{}", '{"schema_version":1,"schema_version":1}\n',
                         '{"schema_version":NaN}\n'):
                trace.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    list(PROFILE.completed_iterations(trace))

    def test_timing_requires_exact_complete_join(self):
        with tempfile.TemporaryDirectory() as root:
            trace, timing = write_capture(root)
            text = timing.read_text()
            for replacement in (text + text.splitlines()[-1] + "\n",
                                text.replace(",timed,0,", ",timed,1,"),
                                text + text.splitlines()[-1].replace(",timed,0,", ",timed,1,") + "\n",
                                text.replace("b1_q24_f100", "b2_q24_f100")):
                timing.write_text(replacement)
                with self.assertRaises(ValueError):
                    PROFILE.convert_capture("yfcc10m", trace, timing)
            timing.write_text(text + text.splitlines()[-1].replace("f100", "f200") + "\n")
            with self.assertRaisesRegex(ValueError, "one benchmark point"):
                PROFILE.load_timings(timing)

    def test_bad_timed_fields(self):
        row = dict(point_id="b1_q24_f100", phase="timed", run_index=0, wall_ms=0.0001,
                   qps=1e10, query_count=1)
        with tempfile.TemporaryDirectory() as root:
            for key, value in (("wall_ms", "nan"), ("wall_ms", "inf"), ("wall_ms", 0),
                               ("run_index", "0.5"), ("run_index", -1), ("phase", "warmup"),
                               ("qps", -1), ("query_count", 0)):
                _, timing = write_capture(root, rows=[row | {key: value}])
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    PROFILE.load_timings(timing)
            timing.write_text("point_id,phase,run_index,wall_ms,wall_ms\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                PROFILE.load_timings(timing)

    def test_multiple_runs_backends_and_workloads(self):
        with tempfile.TemporaryDirectory() as root:
            inputs = []
            for workload in ("yfcc10m", "hybrid"):
                path = Path(root) / workload
                path.mkdir()
                rows, records = [], []
                for run in (2, 0, 1):
                    events = fixture_events()
                    for item in events:
                        item["run"] = run
                        if workload == "hybrid":
                            item.update(point_id="rabitq_b1_q24_f100", backend="rabitq")
                    records.extend([*events, complete(events)])
                    rows.append(dict(point_id=events[0]["point_id"], phase="timed",
                                     run_index=run, wall_ms=0.0001))
                trace, timing = write_capture(path, rows=rows)
                trace.write_text("".join(json.dumps(record) + "\n" for record in records))
                inputs.append((workload, trace, timing))
            result = PROFILE.convert(inputs, Path(root) / "window.csv")
            self.assertEqual([row["run"] for row in result], [0, 1, 2, 0, 1, 2])
            self.assertEqual({row["workload"] for row in result}, {"yfcc10m", "hybrid"})
            for row in result:
                self.assertAlmostEqual(sum(row[name + "_ms"] for name in PROFILE.CATEGORIES), row["wall_ms"])

    def test_real_native_collector_to_converter(self):
        compiler = shutil.which("g++")
        roots = [Path(sys.prefix)]
        for name in ("CONDA_PREFIX", "CUDA_PATH"):
            if os.environ.get(name):
                roots.append(Path(os.environ[name]))
        nvcc = shutil.which("nvcc")
        if nvcc:
            roots.append(Path(nvcc).resolve().parent.parent)
        includes = [root / suffix for root in roots for suffix in
                    ("include", "targets/x86_64-linux/include")]
        nvtx = next((path for path in includes if (path / "nvtx3/nvToolsExt.h").is_file()), None)
        if not compiler or nvtx is None:
            self.skipTest("native integration requires g++ and CUDA/Conda NVTX headers")
        source = r'''#include "profiling/performance_trace.h"
#include <cstdlib>
namespace trace = go_fann::performance_trace;
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    setenv("GO_FANN_PERFORMANCE_TRACE", argv[1], 1);
    trace::Capture capture("pq", "b1.000000_q24_f100", "timed", 24, 0);
    auto span = [](const char* name, long start, long end) {
        trace::span(name, nullptr, trace::Time(std::chrono::nanoseconds(start)),
                    trace::Time(std::chrono::nanoseconds(end)));
    };
    span("provider_search_window", 100, 200);
    span("graph_launch_to_complete", 100, 170);
    span("highfreq_cpu_rerank_batch", 150, 190);
    span("merge_routes_to_original", 190, 200);
    trace::value("reported_search_wall_ms", nullptr, 0.0001);
    capture.finish();
}
'''
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            cpp = path / "fixture.cpp"
            cpp.write_text(source)
            executable = path / "fixture"
            build = subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(ROOT / "src"),
                                    "-I", str(nvtx), str(cpp), "-ldl", "-o", str(executable)],
                                   capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            trace, timing = write_capture(root, rows=[dict(point_id="b1.000000_q24_f100",
                                                         phase="timed", run_index=0, wall_ms=0.0001)])
            trace.unlink()
            result = subprocess.run([str(executable), str(trace)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            row = PROFILE.convert_capture("yfcc10m", trace, timing)[0]
            self.assertEqual(row["graph_phase_ms"], 50 / 1e6)
            self.assertEqual(row["gpu_cpu_rerank_overlap_ms"], 20 / 1e6)
            self.assertEqual(row["cpu_rerank_exposed_ms"], 20 / 1e6)
            self.assertEqual(row["result_merge_ms"], 10 / 1e6)

    def test_output_protection_and_invalid_inputs_leave_no_output(self):
        with tempfile.TemporaryDirectory() as root:
            trace, timing = write_capture(root)
            output = Path(root) / "window.csv"
            inputs = [("yfcc10m", trace, timing)]
            for args in ([], inputs + inputs, [("unknown", trace, timing)],
                         inputs + [("hybrid", trace, timing)]):
                with self.assertRaises(ValueError):
                    PROFILE.convert(args, output)
                self.assertFalse(output.exists())
            original = trace.read_bytes()
            with self.assertRaisesRegex(ValueError, "overwrite an input"):
                PROFILE.convert(inputs, trace)
            self.assertEqual(trace.read_bytes(), original)
            output.symlink_to(Path(root) / "missing.csv")
            with self.assertRaises(FileExistsError):
                PROFILE.convert(inputs, output)
            output.unlink()
            real_link = os.link
            def racing_link(source, destination):
                Path(destination).write_text("other writer")
                real_link(source, destination)
            with mock.patch.object(PROFILE.os, "link", side_effect=racing_link):
                with self.assertRaises(FileExistsError):
                    PROFILE.convert(inputs, output)
            self.assertEqual(output.read_text(), "other writer")
            self.assertFalse(list(Path(root).glob(".window.csv.*.tmp")))


if __name__ == "__main__":
    unittest.main()
