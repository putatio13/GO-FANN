#!/usr/bin/env python3

import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2] / "src"
SOURCE = (SRC_DIR / "search/full_gpu/backend_full_gpu.cu").read_text(encoding="utf-8")
MAKEFILE = (SRC_DIR / "Makefile").read_text(encoding="utf-8")


class FullGpuOptimizationSourceContractTest(unittest.TestCase):
    def test_compile_toggles_are_enabled_and_backend_scoped(self) -> None:
        self.assertIn("FULL_GPU_COOPERATIVE_PICKUP ?= 1", MAKEFILE)
        self.assertIn("FULL_GPU_UINT8_DP4A ?= 1", MAKEFILE)
        self.assertIn(
            "$(OBJ_DIR)/search/full_gpu/backend_full_gpu.o: NVCC_FLAGS += $(FULL_GPU_NVCC_FLAGS)", MAKEFILE
        )
        self.assertIn("#define GO_FANN_FULL_GPU_COOPERATIVE_PICKUP 1", SOURCE)
        self.assertIn("#define GO_FANN_FULL_GPU_UINT8_DP4A 1", SOURCE)
        self.assertNotIn("FULL_GPU_SHARED_FRONTIER", MAKEFILE)
        self.assertNotIn("FULL_GPU_ROUTE_LOCAL_SORTED_QUEUE", MAKEFILE)
        self.assertIn("search/full_gpu/full_gpu_sorted_frontier.cuh", MAKEFILE)
        self.assertIn("$(OBJ_DIR)/flags/full_gpu.o", MAKEFILE)

    def test_retired_fixed_route_modes_are_absent(self) -> None:
        self.assertNotIn("full_gpu_highfreq_staged_kernel", SOURCE)
        self.assertNotIn("full_gpu_highfreq_global_frontier_kernel", SOURCE)
        self.assertNotIn("pack_global_frontier_payload", SOURCE)
        self.assertNotIn("highfreq_routes_per_block", SOURCE)
        self.assertNotIn("search_graph_degree", SOURCE)
        self.assertNotIn("FULL_GPU_ROUTE_EXPANSION", MAKEFILE)
        self.assertNotIn("GO_FANN_FULL_GPU_SHARED_FRONTIER", MAKEFILE)

    def test_overlap_frontier_primitives_remain_graph_context_ready(self) -> None:
        self.assertIn("lookup_sample_distance", SOURCE)
        self.assertIn("publish_sample_distance", SOURCE)
        self.assertIn("merge_route_local_sorted", SOURCE)
        self.assertIn("graph_id, local_row", SOURCE)

    def test_frontier_pickup_is_cooperative_and_deterministic(self) -> None:
        self.assertIn("pickup_next_unused_from_pool_block", SOURCE)
        self.assertIn("if constexpr (kFullGpuCooperativePickup)", SOURCE)
        self.assertIn(
            "candidate_distance == selected_distance && candidate_index < selected_index",
            SOURCE,
        )

    def test_uint8_distance_uses_exact_dp4a_identity(self) -> None:
        self.assertIn("if constexpr (kFullGpuUint8Dp4a)", SOURCE)
        self.assertIn("__dp4a", SOURCE)
        self.assertIn("base_squared_norm + query_squared_norm - 2u * base_query_dot", SOURCE)
        self.assertIn("query_squared_norm_device(const uint8_t* query_row", SOURCE)

    def test_candidate_scratch_uses_invalid_local_sentinel(self) -> None:
        self.assertIn(
            "candidate_local[edge] = should_eval ? local : UINT32_MAX;", SOURCE
        )
        self.assertNotIn("candidate_should_eval", SOURCE)
        self.assertNotIn("candidate_sample[kMaxGraphDegree]", SOURCE)

    def test_packed_visited_has_exact_guard_and_uint32_fallback(self) -> None:
        self.assertIn("constexpr uint32_t kPackedVisitedEmpty = 0x00ffffffu;", SOURCE)
        self.assertIn("struct PackedVisitedEntry", SOURCE)
        self.assertIn("sizeof(PackedVisitedEntry) == 3", SOURCE)
        self.assertIn("template <int kQueueSize, typename T, bool kUsePackedVisited", SOURCE)
        self.assertIn("size <= kPackedVisitedEmpty", SOURCE)
        self.assertIn("operator()<uint8_t, true>", SOURCE)
        self.assertIn("operator()<uint8_t, false>", SOURCE)
        self.assertIn("operator()<float, true>", SOURCE)
        self.assertIn("operator()<float, false>", SOURCE)
        self.assertIn("visited_hash_storage[i] = UINT32_MAX;", SOURCE)


if __name__ == "__main__":
    unittest.main()
