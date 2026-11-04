#!/usr/bin/env python3
import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2] / "src"
ASYNC_HEADER = (ROOT / "execution/async_io.h").read_text()
ASYNC_SOURCE = (ROOT / "execution/async_io.cpp").read_text()
RABITQ = (ROOT / "search/rabitq/backend_rabitq.cu").read_text()
DATALOADER = (ROOT / "data/dataloader.cpp").read_text()


def function_source(source: str, name: str) -> str:
    match = re.search(r"\b" + re.escape(name) + r"\s*\(", source)
    if not match:
        raise AssertionError(f"function {name} not found")
    body_start = source.find("{", match.end())
    depth = 0
    for position in range(body_start, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[match.start():position + 1]
    raise AssertionError(f"function {name} has no complete body")


class CandidateStreamSourceContractTest(unittest.TestCase):
    def test_shared_protocol_is_source_agnostic(self) -> None:
        self.assertIn("struct HostCandidateSourceView", ASYNC_HEADER)
        self.assertIn("QueryCandidateSource source_kind", ASYNC_HEADER)
        self.assertIn("kCandidatePayloadIdsWord", ASYNC_HEADER)
        self.assertIn("void host_candidate_scan_and_set", ASYNC_HEADER)
        service = function_source(ASYNC_SOURCE, "host_candidate_scan_and_set")
        self.assertIn("HostCandidateSourceView", service)
        self.assertIn("host_candidate_matches_route", service)
        self.assertNotIn("range_posting::", service)

    def test_rabitq_keeps_resident_and_streamed_execution_separate(self) -> None:
        self.assertIn("candidate_stream_rabitq_scan_kernel", RABITQ)
        plan = function_source(RABITQ, "build_range_posting_plan")
        self.assertIn("QueryRouteExecution::HostCandidateStream", plan)
        self.assertIn("resident_plan_positions", plan)
        self.assertIn("stream_plan_positions", plan)
        run = function_source(RABITQ, "run_range_posting_rabitq_search")
        self.assertIn("execute_subset(ctx.plan.resident_plan_positions, false,", run)
        self.assertIn("execute_subset(ctx.plan.resident_postfilter_plan_positions,", run)
        self.assertIn("execute_subset(ctx.plan.stream_plan_positions, true,", run)
        self.assertIn("host_candidate_scan_and_set", run)

    def test_residual_labels_trigger_shared_lazy_csr_load(self) -> None:
        loader = function_source(DATALOADER, "ensure_host_candidate_label_data")
        self.assertIn("QueryRouteExecution::HostCandidateStream", loader)
        self.assertIn("sparse_label_row_span", loader)
        self.assertIn("load_spmat_labels", loader)
        self.assertGreaterEqual(DATALOADER.count("ensure_host_candidate_label_data(runtime, config)"), 2)


if __name__ == "__main__":
    unittest.main()
