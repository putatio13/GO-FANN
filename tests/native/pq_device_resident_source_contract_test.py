#!/usr/bin/env python3

import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2] / "src"


class PqDeviceResidentSourceContractTest(unittest.TestCase):
    def test_config_is_pq_scoped_and_defaults_to_mailbox(self) -> None:
        header = (SRC_DIR / "search/pq/backend_pq.cuh").read_text(encoding="utf-8")
        loader = (SRC_DIR / "app/config_loader.cpp").read_text(encoding="utf-8")
        main = (SRC_DIR / "app/main.cu").read_text(encoding="utf-8")

        self.assertIn('std::string graph_transport = "mailbox"', header)
        self.assertIn('load_optional_value(root, "pq_graph_transport"', loader)
        self.assertIn(
            'load_optional_value(root, "pq", "graph_transport", config.pq_graph_transport)',
            loader,
        )
        self.assertIn('"--pq-graph-transport="', main)
        self.assertIn('options.pq_graph_transport == "device_resident"', main)
        self.assertIn(
            'search_options.graph_transport = options.pq_graph_transport',
            main,
        )

    def test_resident_transport_specializes_the_native_kernel(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        native_begin = source.index(
            "native_warp_independent_search_kernel_impl(const pq_idx_t* d_data"
        )
        native_end = source.index(
            "__global__ void resident_pq_batch_exact_rerank_kernel", native_begin
        )
        native = source[native_begin:native_end]

        self.assertIn("bool kDeviceResident = false", source)
        self.assertIn("kNativeThreadInBlock = kCriticalStep * kMultiQuery", source)
        self.assertIn("kNativeThreadInBlock * kNativeWarpsPerQuery", source)
        self.assertIn("threadIdx.x / kCriticalStep", native)
        self.assertIn("constexpr int kQueryWarps = kDeviceResident", native)
        self.assertIn("warp_in_block / kQueryWarps", native)
        self.assertIn("blockIdx.x * kMultiQuery", native)
        self.assertIn("run_pq_search_one_block<kQueueSize", native)
        self.assertIn("kUseRequestRing,\n                            kDeviceResident", native)
        self.assertIn("kNativeResidentThreadInBlock", source)
        self.assertIn("kNativeCoupledThreadInBlock", source)

    def test_resident_branch_only_replaces_mailbox_transport(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        search_begin = source.index("__device__ void run_pq_search_one_block")
        search_end = source.index(
            "native_warp_independent_search_kernel_impl", search_begin
        )
        search = source[search_begin:search_end]

        self.assertIn("if constexpr (!kDeviceResident)", search)
        self.assertIn("if constexpr (kDeviceResident)", search)
        self.assertIn("d_resident_graph[", search)
        self.assertIn("resident_pq_labels_match", search)
        self.assertIn("profiled_native_hash_test", search)
        self.assertIn("profiled_pickup_next_unused_from_native_frontier", search)
        self.assertIn("early_stop_reached", search)
        self.assertIn("pq_table[query_idx]", search)
        self.assertIn("(max_steps > 0 && step_num >= max_steps)", search)

    def test_resident_launch_has_no_host_scan_and_batches_exact_rerank(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        for function_name in ("void execute_graph_batch", "void execute_graph_query_ids"):
            begin = source.index(function_name)
            end = source.index("copy_time_per_block_to_host", begin)
            execute = source[begin:end]
            self.assertIn("launch.template operator()<false, true>()", execute)
            self.assertIn(
                "ctx.device_resident_graph\n            ? std::thread{}", execute
            )
            self.assertIn("launch_resident_pq_batch_exact_rerank", execute)

        rerank_begin = source.index(
            "__global__ void resident_pq_batch_exact_rerank_kernel"
        )
        rerank_end = source.index("template <int kQueueSize", rerank_begin)
        rerank = source[rerank_begin:rerank_end]
        self.assertIn("threadIdx.x / kCriticalStep", rerank)
        self.assertIn("blockIdx.x * kMultiQuery", rerank)
        self.assertNotIn("kNativeWarpsPerQuery", rerank)
        self.assertIn("d_base_vectors", rerank)
        self.assertIn("d_result", rerank)
        self.assertIn("resident_pq_labels_match", rerank)


if __name__ == "__main__":
    unittest.main()
