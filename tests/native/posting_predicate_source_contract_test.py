"""Guard production wiring that the standalone predicate/heap fixture cannot cover."""
from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parents[2] / "src" / "search/rabitq/backend_rabitq.cu").read_text()


def body(name):
    # Skip declarations and calls; definitions have a brace after the signature.
    import re
    candidates = list(re.finditer(r"(?:void|bool)\s+" + name + r"\s*\(", SOURCE))
    for candidate in candidates:
        cursor, parentheses = candidate.end(), 1
        while parentheses:
            parentheses += (SOURCE[cursor] == "(") - (SOURCE[cursor] == ")")
            cursor += 1
        opening = SOURCE.find("{", cursor)
        if ";" in SOURCE[cursor:opening]:
            continue
        depth = 1
        end = opening + 1
        while depth:
            depth += (SOURCE[end] == "{") - (SOURCE[end] == "}")
            end += 1
        return SOURCE[opening:end]
    raise AssertionError(f"Missing definition: {name}")


class PostingPredicateSourceContract(unittest.TestCase):
    def test_direct_backend_rejects_invalid_or_unsupported_score_order(self):
        code = body("validate_rabitq_search_options")
        for guard in ('posting_predicate_order != "filter_first"',
                      'posting_predicate_order != "score_first"',
                      'posting_predicate_order == "score_first"',
                      'query_worker_mode != "coupled"',
                      'graph_transport != "mailbox"',
                      'graph_storage_mode != "memory"',
                      'options.latency_persistent_queue',
                      'options.posting_persistent_workers',
                      'rabitq_predicate_storage != "csr"',
                      'rabitq_predicate_mode != "gpu_exact"',
                      'rabitq_predicate_mode != "gpu_exact_bias"'):
            self.assertIn(guard, code)
        self.assertNotIn("options.use_lane_architecture", SOURCE)
        self.assertNotIn("runtime.graph_on_disk", SOURCE)
        self.assertIn('fail_rabitq("posting_predicate_order must be', code)
        self.assertIn('fail_rabitq("score-first posting requires', code)

    def test_resident_producers_filter_original_sample_before_score_and_heap(self):
        for name in ("run_lowfreq_rabitq_resident_scan_one_block",
                     "range_posting_rabitq_scan_kernel"):
            with self.subTest(name=name):
                code = body(name)
                predicate = code.index("gpu_predicate::matches(posting_predicate, route_idx, sample_id)")
                score = code.index("score_rabitq_code_device<")
                heap = code.index("posting_shortlist_heap::insert<")
                self.assertLess(code.index("sample_id >= base_rows"), predicate)
                self.assertLess(predicate, score)
                self.assertLess(score, heap)
                self.assertIn("__shfl_sync", code[predicate:score])
                self.assertIn("continue;", code[predicate:score])
                self.assertIn("posting_predicate.base_ranges = d_base_range_values", code)
                self.assertNotIn("gpu_predicate::priority", code)

    def test_score_first_uses_valid_heap_before_full_predicate(self):
        for name in ("run_lowfreq_rabitq_resident_scan_one_block",
                     "range_posting_rabitq_scan_kernel"):
            with self.subTest(name=name):
                code = body(name)
                score = code.index("score_rabitq_code_device<")
                gate = code.index("posting_shortlist_heap::can_enter<")
                predicate = code.index(
                    "gpu_predicate::matches(posting_predicate, route_idx, sample_id)", score)
                insertion = code.index("posting_shortlist_heap::insert<")
                self.assertLess(score, gate)
                self.assertLess(gate, predicate)
                self.assertLess(predicate, insertion)
                self.assertIn("__shfl_sync", code[predicate:insertion])
                self.assertIn("continue;", code[predicate:insertion])
                self.assertIn("score_first_requested &&", code)
                self.assertIn("!score_first", code[:score])
                self.assertNotIn("break;", code[score:insertion])
                self.assertIn("dist, sample_id, best_dist, best_ids, shortlist_k", code[gate:predicate])

    def test_score_first_dispatch_defaults_and_diagnostics(self):
        self.assertIn("bool score_first_requested = false", SOURCE)
        for provider in ("label", "range"):
            self.assertIn(f"graph_ctx.{provider}_posting_predicate.view,\n"
                          '                            options.posting_predicate_order == "score_first"', SOURCE)
        self.assertIn('storage=csr mode=" << options.posting_predicate_order', SOURCE)
        self.assertIn('exact_pass_rows_available = false', SOURCE)
        self.assertIn('exact_pass_rows_available=" << range_posting_stats.exact_pass_rows_available', SOURCE)
        self.assertIn('? std::to_string(range_posting_stats.exact_pass_rows) : "unavailable"', SOURCE)
        self.assertIn('posting_predicate,\n        score_first_requested);',
                      body("lowfreq_rabitq_resident_posting_kernel"))

    def test_independent_provider_plans_reach_kernel_launches(self):
        self.assertIn("runtime, go_fann::route_predicate::Producer::LabelPosting, hybrid_residual_runtime", SOURCE)
        self.assertIn("runtime, go_fann::route_predicate::Producer::RangePosting,", SOURCE)
        self.assertIn("graph_ctx.label_posting_predicate.view,", SOURCE)
        self.assertIn("graph_ctx.range_posting_predicate.view,", SOURCE)
        self.assertIn("cleanup_posting_predicate_context(ctx.label_posting_predicate)", SOURCE)
        self.assertIn("cleanup_posting_predicate_context(ctx.range_posting_predicate)", SOURCE)

    def test_posting_upload_shares_csr_without_graph_bias_or_rerank_trust(self):
        code = body("init_posting_predicate_context")
        self.assertIn("acquire_gpu_predicate_index(*plan.label_source", code)
        self.assertIn("view.csr_offsets = ctx.index->csr_offsets", code)
        self.assertIn("view.csr_labels = ctx.index->csr_labels", code)
        self.assertNotIn("view.alpha =", code)
        self.assertNotIn("view.pending_base_stop =", code)
        self.assertNotIn("device_predicate_verified", code)
        self.assertIn("cpu_final_validation=unverified_routes_only", code)

    def test_unsupported_and_host_prefilter_modes_are_not_enabled(self):
        supported = body("gpu_posting_predicate_supported")
        for guard in ('== "gpu_exact"', '== "gpu_exact_bias"',
                      '== "csr"',
                      '== "coupled"', '== "mailbox"', '!options.latency_persistent_queue'):
            self.assertIn(guard, supported)
        label = body("needs_label_posting_predicate_context")
        self.assertIn("!defaults::lowfreq_full_filter_mode_is_pre", label)
        self.assertIn("!lowfreq_use_filter_mask_enabled()", label)


if __name__ == "__main__":
    unittest.main()
