"""Link device helper fixtures to the native main-workload call sites.

These checks supplement (and do not replace) the CUDA semantic fixture.
"""
from pathlib import Path
import unittest

SOURCE = (Path(__file__).resolve().parents[2] / "src" / "search/rabitq/backend_rabitq.cu").read_text()


class MainPredicateSourceContract(unittest.TestCase):
    def test_parallel_proposed_checks_precede_ordered_lane_zero_insertion(self):
        begin = SOURCE.index("uint64_t insert_start = 0;")
        end = SOURCE.index("// Lane 0 mutates the shared frontier", begin)
        insertion = SOURCE[begin:end]
        prepare = insertion.index("gpu_predicate::prepare_candidate_pass_masks(")
        lane_zero = insertion.index("if (subtid == 0)")
        self.assertLess(prepare, lane_zero)
        self.assertIn("if constexpr (kPredicateBias)", insertion[:prepare])
        self.assertIn("predicate_pass_masks[cid][i / 32]", insertion[lane_zero:])
        self.assertIn("predicate_match_observed(predicate, route_idx, sample_idx, true)",
                      insertion[lane_zero:])
        helper = (Path(__file__).resolve().parents[2] / "src" / "predicates/gpu_predicate_parallel.cuh").read_text()
        self.assertIn("#define GO_FANN_GPU_PREDICATE_PARALLEL 1", helper)
        self.assertIn("if (view.counters || !enabled(view, route)) return false;", helper)
        self.assertIn("__ballot_sync(0xffffffffu, pass)", helper)
        self.assertIn("__syncwarp();", helper)

    def test_reserved_pending_candidates_are_in_stop_comparison(self):
        update = SOURCE[SOURCE.index("float pending_score = lane0_state[cid].next.first;"):
                        SOURCE.index("int prefetch_vector_index = 0;")]
        self.assertIn("predicate.pending_base_stop && gpu_predicate::enabled(predicate, route_idx)", update)
        self.assertIn("gpu_predicate::minimum_pending_base(", update)
        self.assertIn("q[cid], heap_size[cid], lane0_state[cid].now,", update)
        self.assertIn("lane0_state[cid].next, true, found,", update)
        self.assertIn("!pool_payload_used(payload)", update)
        self.assertIn("stop_topk_approx_worst <= pending_score", update)

    def test_ring_bias_dispatch_and_alpha_zero_policy(self):
        self.assertIn("native_rabitq_search_kernel<QS, false, true, kUseInt8Score, false, true>", SOURCE)
        self.assertIn("launch.template operator()<false, true, kUseInt8Score, false, true>();", SOURCE)
        self.assertIn('v.pending_base_stop=options.rabitq_predicate_mode=="gpu_exact_bias" && options.search_early_stop_mode=="adaptive";', SOURCE)

    def test_prefetch_and_compiled_or_authority(self):
        self.assertIn("!predicate.route_prefetch_disabled[route_idx]", SOURCE)
        self.assertIn("if (gpu_predicate::enabled(predicate, route_idx)) {", SOURCE)
        self.assertIn("if (!predicate.route_compiled || !predicate.route_compiled[route_idx]) {", SOURCE)
        self.assertIn("ctx.io_config.predicate_gpu_verified_routes = &ctx.predicate_verified_routes;", SOURCE)
        self.assertIn("ctx.predicate_view.base_ranges = ctx.d_base_range_values;", SOURCE)

    def test_complete_index_only_for_label_residuals(self):
        begin = SOURCE.index("void init_gpu_predicate_context(")
        end = SOURCE.index("void validate_gpu_predicate_trace(", begin)
        context = SOURCE[begin:end]
        self.assertIn("if (!needed.empty()) {", context)
        self.assertIn("ctx.predicate_index = acquire_gpu_predicate_index(", context)
        shared = SOURCE[SOURCE.index("std::shared_ptr<GpuPredicateDeviceIndex> acquire_gpu_predicate_index("):begin]
        self.assertIn('upload_predicate_buffer(owner->directory,h.directory,"predicate complete directory");', shared)
        self.assertIn('upload_predicate_buffer(owner->payload,h.payload,"predicate complete payload");', shared)
        self.assertIn("verified==graph_routes", context)


if __name__ == "__main__":
    unittest.main()
