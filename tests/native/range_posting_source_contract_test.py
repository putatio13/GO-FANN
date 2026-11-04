#!/usr/bin/env python3

import re
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2] / "src"
BACKEND = (SRC_DIR / "search/rabitq/backend_rabitq.cu").read_text(encoding="utf-8")
BACKEND_HEADER = (SRC_DIR / "search/rabitq/backend_rabitq.cuh").read_text(encoding="utf-8")
HEAP_HEADER = (SRC_DIR / "search/common/posting_shortlist_heap.cuh").read_text(encoding="utf-8")
MAIN = (SRC_DIR / "app/main.cu").read_text(encoding="utf-8")
CONFIG_HEADER = (SRC_DIR / "app/config_loader.h").read_text(encoding="utf-8")
CONFIG_SOURCE = (SRC_DIR / "app/config_loader.cpp").read_text(encoding="utf-8")
DATALOADER = (SRC_DIR / "data/dataloader.cpp").read_text(encoding="utf-8")
DATALOADER_HEADER = (SRC_DIR / "data/dataloader.h").read_text(encoding="utf-8")


def function_source(source: str, name: str, occurrence: int = 0) -> str:
    matches = list(re.finditer(rf"\b{name}\s*\(", source))
    if occurrence >= len(matches):
        raise AssertionError(f"function {name} occurrence {occurrence} not found")
    start = matches[occurrence].start()
    brace = source.find("{", matches[occurrence].end())
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"function {name} has an unterminated body")


class RangePostingSourceContractTest(unittest.TestCase):
    def test_options_are_plumbed_end_to_end(self) -> None:
        for field in (
            "range_posting_index_path",
            "range_posting_mode",
            "range_posting_scan_threshold",
            "range_posting_shortlist",
            "range_posting_postfilter_shortlist_multiplier",
            "range_posting_and_filter_mode",
            "range_label_and_policy",
            "range_posting_drop_wide_hybrid_or",
        ):
            self.assertIn(field, CONFIG_HEADER)
            self.assertGreaterEqual(CONFIG_SOURCE.count(f'"{field}"'), 2)
            self.assertIn(field, BACKEND_HEADER)
            self.assertIn(field, MAIN)
        self.assertIn("--range-posting-mode", MAIN)
        self.assertIn("--range-label-and-policy", MAIN)
        self.assertIn("--range-posting-drop-wide-hybrid-or", MAIN)
        self.assertIn('set("range_posting_mode", options.range_posting_mode)', MAIN)

    def test_plan_handles_required_clause_routes_and_legacy_threshold(self) -> None:
        plan = function_source(BACKEND, "build_range_posting_plan")
        self.assertIn("runtime.original_query_labels_matrix", plan)
        self.assertIn("enabled range posting requires at least one active range predicate", plan)
        self.assertIn("query_range_mode > 3", plan)
        self.assertIn("runtime.route_providers", plan)
        self.assertIn("QueryRouteProvider::RangePosting", plan)
        self.assertIn("runtime.query_ranges, route_idx", plan)
        self.assertIn("QueryRouteExecution::HostCandidateStream", plan)
        self.assertIn("stream_plan_positions", plan)
        self.assertIn("resident_plan_positions", plan)
        self.assertIn("replaced_routes", plan)
        self.assertIn("if (labels.begin != labels.end)", plan)
        self.assertIn("choose_smallest_slice", plan)
        self.assertIn("options.range_posting_scan_threshold", plan)
        self.assertIn("selected_original_queries[original_query] = 1", plan)
        self.assertIn("selected_routes[route_idx] = 1", plan)
        self.assertIn("representative_routes[original_query]", plan)
        self.assertIn("options.range_posting_drop_wide_hybrid_or", plan)
        self.assertIn("runtime.original_query_clauses.has_clauses", plan)
        self.assertIn("runtime.original_query_label_ops[original_query] == QueryLabelOp::Or", plan)
        self.assertIn("QueryRouteExecution::GpuResidentScan", plan)
        self.assertIn("dropped_wide_hybrid_or_routes", plan)
        self.assertIn("dropped_wide_hybrid_or_rows", plan)

    def test_dataloader_owns_the_production_access_path_decision(self) -> None:
        planner = function_source(DATALOADER, "plan_graph_range_access")
        self.assertIn("clause_has_lowfreq", planner)
        self.assertIn("label_prefer", planner)
        self.assertIn("range_prefer", planner)
        self.assertIn("dual", planner)
        self.assertIn("GpuResidentScanPostFilter", planner)
        self.assertIn("eligible", planner)
        self.assertNotIn("slice.count <= config.shortlist", planner)
        self.assertIn("HostCandidateStream", planner)
        self.assertIn("range_graph_and_original_queries", planner)
        load = function_source(DATALOADER, "load_graph_runtime")
        self.assertIn("range_posting::Index::load", load)
        self.assertIn("plan_graph_range_access", load)

        lowering = function_source(BACKEND, "build_range_posting_plan")
        planned = lowering.index("runtime.range_access_planner_profile.planned")
        legacy = lowering.index("choose_smallest_slice", planned)
        self.assertLess(planned, legacy)
        self.assertIn("runtime.route_range_posting_begins", lowering[planned:legacy])
        self.assertIn("runtime.route_enabled", lowering[planned:legacy])

    def test_planned_routes_carry_clause_assignments_and_coalesce(self) -> None:
        planner = function_source(DATALOADER, "plan_graph_range_access")
        for symbol in (
            "route_clause_indptr",
            "range_graph_clause_indptr",
            "representative_by_key",
            "route_clause_ids",
        ):
            self.assertIn(symbol, planner)
        for helper in (
            "sample_matches_route_clause_labels",
            "sample_matches_range_graph_clause_labels",
        ):
            self.assertIn(helper, DATALOADER_HEADER)
        self.assertIn("sample_matches_range_graph_clause_labels", BACKEND)

    def test_backend_reuses_dataloader_owned_index(self) -> None:
        search = function_source(BACKEND, "run_rabitq_search")
        self.assertIn("runtime.range_posting_index", search)
        self.assertNotIn("range_posting::Index::load", search)

    def test_select_skips_graph_routes_while_hybrid_keeps_them(self) -> None:
        iteration = function_source(BACKEND, "execute_rabitq_iteration")
        select = iteration.index("range_posting::Mode::Select")
        graph = iteration.index("execute_rabitq_graph_batch")
        posting = iteration.index("run_range_posting_rabitq_search")
        self.assertLess(select, graph)
        self.assertLess(graph, posting)
        self.assertIn("plan.replaced_routes", iteration[select:graph])
        self.assertIn("high_query_ids.erase", iteration[select:graph])
        self.assertIn("low_query_ids.erase", iteration[select:graph])
        self.assertIn("merge_exact_result_row", iteration[posting:])

    def test_load_config_advertises_range_provider_only_when_enabled(self) -> None:
        load_config = function_source(BACKEND, "make_rabitq_load_config")
        self.assertIn("kQueryRouteProviderLabel", load_config)
        self.assertIn("range_posting::Mode::Off", load_config)
        self.assertIn("kQueryRouteProviderRangePosting", load_config)

    def test_kernel_skips_single_range_recheck_and_checks_multi_range(self) -> None:
        kernel = function_source(BACKEND, "range_posting_rabitq_scan_kernel")
        self.assertIn("d_route_to_original_query", kernel)
        self.assertIn("count_matching_active_ranges", kernel)
        self.assertIn("active_range_count > 1", kernel)
        self.assertIn("matched_active_dims == active_range_count", kernel)
        self.assertIn("exact_pass = __shfl_sync", kernel)
        self.assertLess(kernel.index("exact_pass == 0"),
                        kernel.index("score_rabitq_code_device"))

    def test_postfilter_multiplier_splits_route_capacity_and_heap_is_bounded(self) -> None:
        init = function_source(BACKEND, "init_range_posting_context")
        run = function_source(BACKEND, "run_range_posting_rabitq_search")
        self.assertIn("postfilter_shortlist_k", init)
        self.assertIn("resident_postfilter_plan_positions", init)
        self.assertIn("ctx.postfilter_shortlist_k", run)
        self.assertIn("execute_subset(ctx.plan.resident_postfilter_plan_positions", run)
        self.assertIn("ordinary_active_blocks_per_sm", run)
        self.assertIn("postfilter_active_blocks_per_sm", run)
        self.assertIn("cudaOccupancyMaxActiveBlocksPerMultiprocessor", run)
        self.assertIn("posting_shortlist_heap::insert", BACKEND)
        self.assertIn("posting_shortlist_heap::sort", BACKEND)
        self.assertIn("range posting postfilter shortlist multiplier", BACKEND)
        for symbol in ("insert", "sort", "sift_down", "pair_is_worse"):
            self.assertIn(symbol, HEAP_HEADER)

        rerank = function_source(BACKEND, "lowfreq_rabitq_host_rerank_range")
        self.assertIn("rerank_stable_batches != UINT32_MAX", rerank)
        self.assertGreaterEqual(run.count("UINT32_MAX"), 2)

    def test_posting_ids_are_resident_and_exact_rerank_is_reused(self) -> None:
        init = function_source(BACKEND, "init_range_posting_context")
        self.assertIn("cudaMalloc(&ctx.d_sample_ids", init)
        self.assertIn("posting_index.dimension(range_index)", init)
        run = function_source(BACKEND, "run_range_posting_rabitq_search")
        self.assertIn("lowfreq_rabitq_host_rerank", run)
        self.assertIn("options.highfreq_rerank_threads", run)
        self.assertIn("ctx.shortlist_k", run)
        self.assertIn("postfilter_residual_pass_rows", run)
        self.assertIn("exact_rerank_rows", run)
        self.assertIn("sample_matches_route_predicate", run)

    def test_search_reports_end_to_end_qps(self) -> None:
        search = function_source(BACKEND, "run_rabitq_search")
        self.assertIn("[rabitq-e2e]", search)
        self.assertIn("result.wall_ms", search)
        self.assertIn("result.qps", search)


if __name__ == "__main__":
    unittest.main()
