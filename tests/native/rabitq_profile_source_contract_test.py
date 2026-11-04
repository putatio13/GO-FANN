#!/usr/bin/env python3

import unittest
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] / "src" / "search/rabitq/backend_rabitq.cu").read_text(
    encoding="utf-8"
)
MAKEFILE = (Path(__file__).resolve().parents[2] / "src" / "Makefile").read_text(
    encoding="utf-8"
)


class RabitqProfileSourceContractTest(unittest.TestCase):
    def test_synchronous_profiler_launch_can_start_mailbox_service_first(self) -> None:
        helper = SOURCE[SOURCE.index("bool profile_early_host_service_enabled()"):
                        SOURCE.index("bool mailbox_rtt_profile_enabled()")]
        self.assertIn('value == nullptr || std::strcmp(value, "0") == 0', helper)
        self.assertIn('std::strcmp(value, "1") == 0', helper)
        body = SOURCE[SOURCE.index("void execute_rabitq_graph_batch("):]
        early = body.index("if (early_host_service) scan_thread = start_host_service();")
        launch = body.index("native_rabitq_search_kernel<QS", early)
        ordinary = body.index("if (!early_host_service) scan_thread = start_host_service();")
        self.assertLess(early, launch)
        self.assertLess(launch, ordinary)
        self.assertIn("early profiling host service requires a memory graph and coupled request ring", body[:early])
        cleanup = body.index("request_ring_cancelled.store(true", ordinary)
        self.assertIn("publish_finished_for_host_service(ctx, query_indices)", body[cleanup:cleanup+350])
        self.assertIn("scan_thread.join()", body[cleanup:cleanup+350])

    def test_capture_range_is_opt_in_and_timed_only(self) -> None:
        self.assertIn(
            'std::getenv("GO_FANN_CUDA_PROFILER_CAPTURE_RANGE")', SOURCE
        )
        self.assertIn("cuda_profiler_capture_range_enabled() && run == 0", SOURCE)
        self.assertIn('" wall_time_ns=" << timed_start_wall_ns', SOURCE)
        self.assertIn('" wall_time_ns=" << timed_end_wall_ns', SOURCE)
        start = SOURCE.index("cudaProfilerStart()")
        execute = SOURCE.index("execute_benchmark_iteration(false)", start)
        stop = SOURCE.index("cudaProfilerStop()", execute)
        timed_start = SOURCE.index("[benchmark timed-run start]", start)
        self.assertLess(start, timed_start)
        self.assertLess(timed_start, execute)
        self.assertLess(execute, stop)

    def test_phase_snapshot_is_compile_time_opt_in(self) -> None:
        self.assertIn("RABITQ_PHASE_SNAPSHOT_PROFILE ?= 0", MAKEFILE)
        self.assertIn(
            "-DGO_FANN_RABITQ_PHASE_SNAPSHOT_PROFILE=", MAKEFILE
        )
        self.assertIn("#if GO_FANN_RABITQ_PHASE_SNAPSHOT_PROFILE", SOURCE)
        self.assertIn(
            'std::getenv("GO_FANN_RABITQ_PHASE_SNAPSHOT_DIR")', SOURCE
        )

    def test_phase_snapshot_schema_preserves_sm_and_phase_counts(self) -> None:
        self.assertIn("(warp_kernel::get_smid() << 8)", SOURCE)
        self.assertIn(
            '"sample,wall_time_ns,copy_us,sm_id,inactive,wait,score,frontier,done\\n"',
            SOURCE,
        )
        for phase in ("Wait", "Score", "Frontier", "Done"):
            self.assertIn(f"RabitqWarpPhase::{phase}", SOURCE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
