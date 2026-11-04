#!/usr/bin/env python3

import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2] / "src"


class PqRequestRingSourceContractTest(unittest.TestCase):
    def test_device_protocol_is_shared_by_pq_and_rabitq(self) -> None:
        helper = (SRC_DIR / "execution/host_request_ring.cuh").read_text(encoding="utf-8")
        pq = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        rabitq = (SRC_DIR / "search/rabitq/backend_rabitq.cu").read_text(encoding="utf-8")

        self.assertIn("publish_host_request", helper)
        self.assertIn("ld.global.cg.u64", helper)
        self.assertIn("st.release.sys.b64", helper)
        self.assertIn('#include "execution/host_request_ring.cuh"', pq)
        self.assertIn('#include "execution/host_request_ring.cuh"', rabitq)
        self.assertNotIn("publish_rabitq_host_request", rabitq)

    def test_pq_coupled_kernel_publishes_demand_and_done(self) -> None:
        makefile = (SRC_DIR / "Makefile").read_text(encoding="utf-8")
        pq = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        publisher_begin = pq.index("__device__ __noinline__ void publish_pq_host_request")
        publisher_end = pq.index("std::chrono::milliseconds finite_no_progress_timeout", publisher_begin)
        publisher = pq[publisher_begin:publisher_end]
        kernel_begin = pq.index("__device__ void run_pq_search_one_block")
        kernel_end = pq.index("native_warp_independent_search_kernel_impl", kernel_begin)
        kernel = pq[kernel_begin:kernel_end]

        self.assertIn("request_ring_device::publish_host_request", publisher)
        self.assertIn("PQ_REQUEST_PUBLISH_TIMING ?= 0", makefile)
        self.assertIn("GO_FANN_PQ_REQUEST_PUBLISH_TIMING", pq)
        self.assertIn("print_pq_request_publish_summary", pq)
        self.assertIn(
            'print_pq_request_publish_summary("PQ benchmark highfreq", highfreq_stats)',
            pq,
        )
        self.assertIn("bool kUseRequestRing", kernel)
        self.assertEqual(kernel.count("publish_profiled_pq_host_request<"), 2)
        self.assertEqual(kernel.count("publish_pq_host_request("), 1)
        self.assertIn("request_publish_ns", kernel)
        self.assertGreaterEqual(kernel.count("HostRequestKind::Demand"), 2)
        self.assertEqual(kernel.count("HostRequestKind::Done"), 1)
        self.assertIn("active_query_idx / request_ring_capacity", kernel)

    def test_pq_host_path_owns_ring_lifecycle_and_cancellation(self) -> None:
        pq = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")

        self.assertIn('ctx.request_ring_enabled = options.request_mode == "batch_ring"', pq)
        self.assertIn("cudaHostAlloc PQ request ring entries", pq)
        self.assertIn("cudaMalloc PQ request ring consumed sequences", pq)
        self.assertIn("gdr_pin_buffer PQ request ring consumed sequences", pq)
        self.assertIn("gdr_map PQ request ring consumed sequences", pq)
        self.assertIn("cudaHostAlloc PQ request ring consumed sequences", pq)
        self.assertIn("cudaHostGetDevicePointer PQ request ring consumed sequences", pq)
        self.assertIn("ctx.cpu_ptr_request_ring_consumed_sequences", pq)
        self.assertIn("ctx.mh_request_ring_consumed_sequences_pinned", pq)
        cleanup_begin = pq.index("void cleanup_graph_context")
        cleanup_end = pq.index("void init_graph_context", cleanup_begin)
        cleanup = pq[cleanup_begin:cleanup_end]
        self.assertLess(
            cleanup.index("gdr_unmap(ctx.g,"),
            cleanup.index("gdr_close(ctx.g)"),
        )
        self.assertLess(
            cleanup.index("gdr_unpin_buffer(ctx.g, ctx.mh_request_ring_consumed_sequences)"),
            cleanup.index("gdr_close(ctx.g)"),
        )
        self.assertIn("cudaFree(ctx.d_request_ring_consumed_sequences_raw)", pq)
        self.assertIn("cudaMemset PQ request ring tails", pq)
        self.assertIn("std::thread(host_request_ring_and_set", pq)
        self.assertIn("request_ring_cancelled_->store(true", pq)
        self.assertGreaterEqual(pq.count("scan_thread.abort_and_join();"), 2)
        self.assertIn("cudaFreeHost(ctx.h_request_ring_entries)", pq)

    def test_consumed_sequence_uses_gdr_writeback_and_release_publication(self) -> None:
        helper = (SRC_DIR / "execution/host_request_ring.cuh").read_text(encoding="utf-8")
        async_io = (SRC_DIR / "execution/async_io.cpp").read_text(encoding="utf-8")
        pq = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        rabitq = (SRC_DIR / "search/rabitq/backend_rabitq.cu").read_text(encoding="utf-8")

        self.assertIn("st.release.sys.b64", helper)
        self.assertNotIn("__threadfence_system", helper)
        store_begin = async_io.index("void store_request_ring_sequence")
        store_end = async_io.index("void host_request_ring_worker", store_begin)
        store = async_io[store_begin:store_end]
        self.assertIn("reinterpret_cast<volatile uint64_t*>", store)
        self.assertIn("_mm_sfence();", store)
        self.assertIn("__atomic_store_n(sequence, value, __ATOMIC_RELEASE)", store)
        self.assertEqual(pq.count("ctx.request_ring_consumed_device_mapped,"), 2)
        self.assertIn("cudaHostAlloc RabitQ request ring consumed sequences", rabitq)

    def test_pq_config_surface_and_rabitq_applicable_defaults(self) -> None:
        main = (SRC_DIR / "app/main.cu").read_text(encoding="utf-8")
        config = (SRC_DIR / "app/config_loader.cpp").read_text(encoding="utf-8")
        pq = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")

        self.assertIn('"--pq-request-mode="', main)
        self.assertIn('"--pq-request-consumed-memory="', main)
        self.assertIn('set("pq_request_mode", options.pq_request_mode)', main)
        self.assertIn(
            'set("pq_request_consumed_memory", options.pq_request_consumed_memory)',
            main,
        )
        self.assertIn('"pq", "request_mode", config.pq_request_mode', config)
        self.assertIn(
            '"pq",\n                        "request_consumed_memory",', config
        )
        self.assertIn(
            "pq-request-consumed-memory must be one of: host, device", main
        )
        self.assertIn(
            "request_consumed_memory must be one of: host, device", pq
        )
        self.assertIn("apply_rabitq_optimized_defaults", main)
        self.assertIn("!options.rabitq_request_mode_explicit", main)
        self.assertIn("!options.rabitq_use_int8_score_explicit", main)
        self.assertIn('options.rabitq_bits_per_dim <= 2', main)

    def test_resident_pq_rerank_uses_direct_row_pointer(self) -> None:
        pq = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        helper_begin = pq.index("float highfreq_exact_l2_for_sample")
        helper_end = pq.index("lowfreq_host_heuristic_rerank_range", helper_begin)
        helper = pq[helper_begin:helper_end]

        self.assertNotIn("base_vectors_on_disk", pq)
        self.assertIn("runtime.base_vectors.row_ptr(sample_id)", helper)
        self.assertNotIn("DiskVectorBatch", pq)


if __name__ == "__main__":
    unittest.main()
