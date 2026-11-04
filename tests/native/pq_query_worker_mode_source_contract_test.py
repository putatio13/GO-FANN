#!/usr/bin/env python3

import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[2] / "src"


class PqQueryWorkerModeSourceContractTest(unittest.TestCase):
    def test_config_cli_option_and_dump_plumbing(self) -> None:
        loader = (SRC_DIR / "app/config_loader.cpp").read_text(encoding="utf-8")
        main = (SRC_DIR / "app/main.cu").read_text(encoding="utf-8")
        header = (SRC_DIR / "search/pq/backend_pq.cuh").read_text(encoding="utf-8")

        self.assertIn('load_optional_value(root, "pq_query_worker_mode"', loader)
        self.assertIn('"pq_finite_token_transport",', loader)
        self.assertIn('"pq_finite_dispatch_mode",', loader)
        self.assertIn(
            '"pq_finite_partial_batch_timeout_us",\n                        config.pq_finite_partial_batch_timeout_us',
            loader,
        )
        self.assertIn(
            'load_optional_value(root, "pq", "query_worker_mode", config.pq_query_worker_mode)',
            loader,
        )
        self.assertIn('"finite_partial_batch_timeout_us",', loader)
        self.assertIn('"finite_token_transport",', loader)
        self.assertIn('"finite_dispatch_mode",', loader)
        self.assertLess(
            loader.index('load_optional_value(root, "pq_query_worker_mode"'),
            loader.index('load_optional_value(root, "pq", "query_worker_mode"'),
        )
        self.assertLess(
            loader.index('"pq_finite_partial_batch_timeout_us"'),
            loader.index('"finite_partial_batch_timeout_us"'),
        )
        self.assertLess(
            loader.index('"pq_finite_token_transport"'),
            loader.index('"finite_token_transport"'),
        )
        self.assertLess(
            loader.index('"pq_finite_dispatch_mode"'),
            loader.index('"finite_dispatch_mode"'),
        )
        self.assertIn("std::string query_worker_mode =", header)
        self.assertIn("std::string finite_token_transport =", header)
        self.assertIn("std::string finite_dispatch_mode =", header)
        self.assertIn("int finite_partial_batch_timeout_us =", header)
        self.assertIn("apply_override(config.pq_query_worker_mode", main)
        self.assertIn("apply_override(config.pq_finite_token_transport", main)
        self.assertIn("apply_override(config.pq_finite_dispatch_mode", main)
        self.assertIn("apply_override(config.pq_finite_partial_batch_timeout_us", main)
        self.assertIn("apply_override(overrides.pq_query_worker_mode", main)
        self.assertIn("apply_override(overrides.pq_finite_token_transport", main)
        self.assertIn("apply_override(overrides.pq_finite_dispatch_mode", main)
        self.assertIn("apply_override(overrides.pq_finite_partial_batch_timeout_us", main)
        self.assertIn('"--pq-query-worker-mode="', main)
        self.assertIn('"--pq-finite-token-transport="', main)
        self.assertIn('"--pq_finite_token_transport="', main)
        self.assertIn('"--pq-finite-dispatch-mode="', main)
        self.assertIn('"--pq_finite_dispatch_mode="', main)
        self.assertIn(
            "[--pq-finite-dispatch-mode=batch|persistent|ready_ring]", main
        )
        self.assertIn(
            '"pq-finite-dispatch-mode must be one of: batch, persistent, ready_ring"',
            main,
        )
        self.assertIn('"--pq-finite-partial-batch-timeout-us="', main)
        self.assertIn('"--pq_finite_partial_batch_timeout_us="', main)
        self.assertIn("search_options.query_worker_mode = options.pq_query_worker_mode", main)
        self.assertIn(
            "search_options.finite_token_transport = options.pq_finite_token_transport",
            main,
        )
        self.assertIn(
            "search_options.finite_dispatch_mode = options.pq_finite_dispatch_mode",
            main,
        )
        self.assertIn(
            "search_options.finite_partial_batch_timeout_us =\n        options.pq_finite_partial_batch_timeout_us",
            main,
        )
        self.assertIn('set("pq_query_worker_mode", options.pq_query_worker_mode)', main)
        self.assertIn(
            'set("pq_finite_token_transport", options.pq_finite_token_transport)',
            main,
        )
        self.assertIn(
            'set("pq_finite_dispatch_mode", options.pq_finite_dispatch_mode)',
            main,
        )
        self.assertIn(
            'set("pq_finite_partial_batch_timeout_us",\n        options.pq_finite_partial_batch_timeout_us)',
            main,
        )
        self.assertIn('{"query_worker_mode", options.pq_query_worker_mode}', main)
        self.assertIn(
            '{"finite_token_transport", options.pq_finite_token_transport}',
            main,
        )
        self.assertIn(
            '{"finite_dispatch_mode", options.pq_finite_dispatch_mode}',
            main,
        )
        self.assertIn(
            '{"finite_partial_batch_timeout_us", options.pq_finite_partial_batch_timeout_us}',
            main,
        )

    def test_validation_limits_finite_to_native_serial_pq(self) -> None:
        main = (SRC_DIR / "app/main.cu").read_text(encoding="utf-8")
        backend = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")

        self.assertIn("is_valid_pq_query_worker_mode(options.pq_query_worker_mode)", main)
        self.assertIn(
            "is_valid_pq_finite_token_transport(\n            options.pq_finite_token_transport)",
            main,
        )
        self.assertIn(
            "is_valid_pq_finite_partial_batch_timeout_us(\n            options.pq_finite_partial_batch_timeout_us)",
            main,
        )
        self.assertIn('options.backend != "pq"', main)
        self.assertIn('options.mode != "search" && options.mode != "benchmark"', main)
        self.assertIn('options.pq_buffer_mode != "double_slot_serial"', main)
        self.assertIn("void validate_pq_query_worker_mode", backend)
        self.assertIn(
            "is_valid_pq_finite_token_transport(options.finite_token_transport)",
            backend,
        )
        self.assertIn(
            "is_valid_pq_finite_dispatch_mode(options.finite_dispatch_mode)",
            backend,
        )
        self.assertIn(
            "is_valid_pq_finite_partial_batch_timeout_us(\n            options.finite_partial_batch_timeout_us)",
            backend,
        )
        self.assertIn('options.buffer_mode != "double_slot_serial"', backend)
        self.assertIn("pq_finite_dispatch_mode_requires_completion_path", main)
        self.assertIn("pq_finite_dispatch_mode_requires_completion_path", backend)
        self.assertIn('options.graph_storage_mode == "memory"', main)
        self.assertIn('options.graph_storage_mode == "memory"', backend)
        self.assertIn(
            '"pq-finite-dispatch-mode=" + options.pq_finite_dispatch_mode', main
        )
        self.assertIn("finite completion dispatch requires memory graph with double_slot_serial", backend)
        self.assertNotIn("options.use_lane_architecture", backend)
        self.assertIn("PQ_MULTI_QUERY=2", backend)
        latency_begin = backend.index("int run_pq_latency_benchmark")
        latency_end = backend.index("int run_pq_search(", latency_begin)
        self.assertIn("query_worker_mode=finite is limited", backend[latency_begin:latency_end])

    def test_control_record_and_state_contract(self) -> None:
        header = (SRC_DIR / "search/pq/backend_pq.cuh").read_text(encoding="utf-8")
        self.assertIn("enum class PqFiniteContextState : uint32_t", header)
        for state in ("UNINITIALIZED", "WAIT_IO", "READY", "RUNNING", "DONE"):
            self.assertIn(state, header)
        self.assertIn("struct alignas(16) PqFiniteContextControl", header)
        self.assertIn("static_assert(sizeof(PqFiniteContextControl) == 96", header)
        self.assertIn("static_assert(alignof(PqFiniteContextControl) == 16", header)
        for field in (
            "state",
            "now_distance",
            "frontier_size",
            "current_buffer",
            "state_since_ns",
        ):
            self.assertIn(f"offsetof(PqFiniteContextControl, {field})", header)
        self.assertIn("struct alignas(64) PqFiniteContextTelemetry", header)
        self.assertIn("static_assert(sizeof(PqFiniteContextTelemetry) == 64", header)
        self.assertIn("static_assert(alignof(PqFiniteContextTelemetry) == 64", header)
        for field in (
            "resumed_step_count",
            "wait_io_ns",
            "resume_count",
            "static_prefetch_step_count",
            "multi_step_resume_count",
            "max_resume_quantum_steps",
        ):
            self.assertIn(f"offsetof(PqFiniteContextTelemetry, {field})", header)
        self.assertIn("struct PqFiniteReadyToken", header)
        self.assertIn("static_assert(sizeof(PqFiniteReadyToken) == 32", header)

    def test_finite_lifecycle_and_byte_accounting_are_context_owned(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        context_begin = source.index("struct PqGraphSearchContext")
        context_end = source.index("struct PqImmutableDeviceState", context_begin)
        context = source[context_begin:context_end]
        for name in (
            "d_finite_contexts",
            "d_finite_context_telemetry",
            "d_finite_frontier_state",
            "d_finite_stop_state",
            "d_finite_result_state",
            "d_finite_ready_tokens",
            "h_finite_ready_tokens_mapped",
            "finite_token_transport",
            "finite_queue_capacity",
            "finite_result_capacity",
            "finite_mutable_bytes_per_query",
            "finite_mutable_bytes_total",
        ):
            self.assertIn(name, context)

        allocate_begin = source.index("void allocate_finite_query_state")
        allocate_end = source.index("struct PqImmutableDeviceState", allocate_begin)
        allocate = source[allocate_begin:allocate_end]
        self.assertLess(
            allocate.index("if (!ctx.finite_query_state_enabled)"),
            allocate.index("cudaMalloc(&ctx.d_finite_contexts"),
        )
        for name in (
            "cudaMalloc(&ctx.d_finite_contexts",
            "cudaMalloc(&ctx.d_finite_context_telemetry",
            "cudaMalloc(&ctx.d_finite_frontier_state",
            "cudaMalloc(&ctx.d_finite_stop_state",
            "cudaMalloc(&ctx.d_finite_result_state",
            "cudaMalloc(&ctx.d_finite_ready_tokens",
            "reinterpret_cast<void**>(&ctx.h_finite_ready_tokens_mapped)",
            "cudaHostGetDevicePointer(",
            "reset_finite_query_state(ctx, query_capacity)",
            "print_finite_query_state_layout(ctx)",
        ):
            self.assertIn(name, allocate)
        self.assertIn("} catch (...) {", allocate)
        rollback = allocate[allocate.index("} catch (...) {") :]
        for name in (
            "cudaFreeHost(ctx.h_finite_ready_tokens_mapped)",
            "cudaFree(ctx.d_finite_ready_tokens)",
            "cudaFree(ctx.d_finite_result_state)",
            "cudaFree(ctx.d_finite_stop_state)",
            "cudaFree(ctx.d_finite_frontier_state)",
            "cudaFree(ctx.d_finite_context_telemetry)",
            "cudaFree(ctx.d_finite_contexts)",
            "throw;",
        ):
            self.assertIn(name, rollback)

        reset_begin = source.index("void reset_finite_query_state")
        reset_end = source.index("void allocate_finite_query_state", reset_begin)
        reset = source[reset_begin:reset_end]
        self.assertIn("if (!ctx.finite_query_state_enabled)", reset)
        for name in (
            "cudaMemset(ctx.d_finite_contexts",
            "cudaMemset(ctx.d_finite_context_telemetry",
            "cudaMemset(ctx.d_finite_frontier_state",
            "cudaMemset(ctx.d_finite_stop_state",
            "cudaMemset(ctx.d_finite_result_state",
        ):
            self.assertIn(name, reset)

        cleanup_begin = source.index("void cleanup_graph_context")
        cleanup_end = source.index("void init_graph_context", cleanup_begin)
        cleanup = source[cleanup_begin:cleanup_end]
        for name in (
            "cudaFree(ctx.d_finite_contexts)",
            "cudaFree(ctx.d_finite_context_telemetry)",
            "cudaFree(ctx.d_finite_frontier_state)",
            "cudaFree(ctx.d_finite_stop_state)",
            "cudaFree(ctx.d_finite_result_state)",
            "cudaFree(ctx.d_finite_ready_tokens)",
            "cudaFreeHost(ctx.h_finite_ready_tokens_mapped)",
        ):
            self.assertIn(name, cleanup)
        self.assertLess(
            cleanup.index("if (ctx.h_finite_ready_tokens_mapped != nullptr)"),
            cleanup.index("cudaFree(ctx.d_finite_ready_tokens)"),
        )

    def test_coupled_path_has_zero_finite_allocation_and_kernels_are_unchanged(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        allocate_begin = source.index("void allocate_finite_query_state")
        allocate_end = source.index("struct PqImmutableDeviceState", allocate_begin)
        allocate = source[allocate_begin:allocate_end]
        coupled_branch = allocate[: allocate.index("ctx.finite_queue_capacity =")]
        self.assertIn("if (!ctx.finite_query_state_enabled)", coupled_branch)
        self.assertIn("print_finite_query_state_layout(ctx)", coupled_branch)
        self.assertIn("return;", coupled_branch)
        self.assertNotIn("cudaMalloc", coupled_branch)

        init_begin = source.index("void init_graph_context(")
        init_end = source.index("void init_graph_context_prepared", init_begin)
        init = source[init_begin:init_end]
        self.assertIn("struct InitRollback", init)
        self.assertIn("if (!committed) cleanup_graph_context(*context)", init)
        self.assertIn("rollback.committed = true", init)
        self.assertLess(
            init.index("struct InitRollback"),
            init.index("allocate_finite_query_state"),
        )
        self.assertLess(
            init.index("allocate_finite_query_state"),
            init.index("rollback.committed = true"),
        )
        self.assertLess(init.index("cudaMalloc(&ctx.d_pq_table"), init.index("allocate_finite_query_state"))
        self.assertLess(init.index("cudaMalloc(&ctx.d_hashdata"), init.index("allocate_finite_query_state"))
        device_begin = source.index("__device__ void run_pq_search_one_block")
        device_end = source.index("struct PqGraphSearchContext", device_begin)
        self.assertNotIn("d_finite_", source[device_begin:device_end])

    def test_finite_capacity_uses_the_dispatch_tiers(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        helper_begin = source.index("constexpr uint32_t specialized_queue_capacity")
        helper_end = source.index("constexpr size_t finite_mutable_bytes_per_query", helper_begin)
        helper = source[helper_begin:helper_end]
        for tier in (64, 128, 256, 512):
            self.assertIn(str(tier), helper)
        dispatch_begin = source.index("void dispatch_kernel_by_queue_size")
        dispatch_end = source.index("void dispatch_graph_kernel", dispatch_begin)
        self.assertIn("specialized_queue_capacity(queue_size)", source[dispatch_begin:dispatch_end])
        self.assertIn("finite_mutable_bytes_per_query=%zu", source)
        self.assertIn("finite_mutable_bytes_total=%zu", source)

    def test_finite_hot_cold_traffic_accounting_is_explicit(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        self.assertIn(
            "sizeof(PqFiniteContextControl) + sizeof(PqFiniteContextTelemetry)",
            source,
        )
        self.assertIn("params.d_context_telemetry = ctx.d_finite_context_telemetry", source)
        self.assertIn("record_finite_state_traffic(batch_stats", source)
        self.assertIn('accounting=logical ', source)
        self.assertIn('operation_unit=record_field_or_array_region ', source)
        self.assertIn('layout_change=access_split_not_capacity_compression ', source)
        for transition in ("resume", "park", "terminal"):
            self.assertIn(f"finite_{transition}_state_traffic_per_hop", source)
            self.assertIn(f"{transition}_hops", source)

    def test_d2_availability_is_measured_before_worker_quantum_changes(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        for contract in (
            "resume_quantum_steps",
            "static_prefetch_steps",
            "print_pq_d2_availability_summary",
            "host_completed_first_row_plus_static_resident_prefetch",
            "qps=%.6f",
            "same_query_multi_token_producer=absent",
            "finite_resumed_step_sum",
            "multi_step_resume_fraction",
        ):
            self.assertIn(contract, source)
        benchmark_begin = source.index("PqIterationResult execute_single_iteration")
        benchmark_end = source.index("void cleanup_benchmark_contexts", benchmark_begin)
        benchmark = source[benchmark_begin:benchmark_end]
        d2_summary = benchmark.index(
            'print_pq_d2_availability_summary("PQ benchmark highfreq",'
        )
        qps_assignment = benchmark.index("const double qps =")
        self.assertLess(qps_assignment, d2_summary)

    def test_finite_init_and_resume_preserve_hash_and_original_query_identity(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        finite_begin = source.index("struct PqFiniteKernelParams")
        finite_end = source.index("__device__ void run_pq_search_one_block")
        finite = source[finite_begin:finite_end]

        self.assertIn("struct PqFiniteFixHashView", finite)
        self.assertIn("new FixHash<uint32_t, kHashTableCapacity>", finite)
        self.assertEqual(finite.count("new FixHash<uint32_t, kHashTableCapacity>"), 1)
        resume_begin = finite.index("finite_pq_resume_kernel")
        self.assertNotIn("new FixHash", finite[resume_begin:])
        self.assertIn("token->query_id", finite[resume_begin:])
        self.assertIn("params.d_contexts[query_idx]", finite)
        self.assertIn("static_cast<size_t>(query_idx) * kGdrMailboxStride", finite)

    def test_resume_validates_complete_mailbox_ownership_before_consuming(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        begin = source.index("if constexpr (!kInitialize)")
        end = source.index("const uint32_t group_id", begin)
        validation = source[begin:end]
        for contract in (
            "control[cid].query_id != query_idx",
            "control[cid].context_epoch != context_epoch",
            "control[cid].state != PqFiniteContextState::WAIT_IO",
            "control[cid].io_generation != token->generation",
            "control[cid].expected_row != token->row",
            "control[cid].current_buffer != token->buffer",
            "REJECT_STATUS",
            "REJECT_MAILBOX",
        ):
            self.assertIn(contract, validation)
        self.assertLess(validation.index("REJECT_STATE"), validation.index("REJECT_MAILBOX"))

        resume_state_begin = source.index(
            "control[cid].state = PqFiniteContextState::READY", end
        )
        resume_state_end = source.index("__syncwarp();", resume_state_begin)
        self.assertIn("index_list_len[cid] = 0", source[resume_state_begin:resume_state_end])

    def test_request_and_terminal_publication_orders_are_fenced(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        helper_begin = source.index("void store_finite_mutable_state")
        helper_end = source.index(
            "template <int kQueueSize, int kGraphPqDim, bool kInitialize, "
            "bool kCollectPerHopTiming>",
            helper_begin,
        )
        helper = source[helper_begin:helper_end]
        for cooperative_loop in (
            "for (int i = subtid; i < kQueueSize + 2; i += step)",
            "for (int i = subtid; i < kTopk + 1; i += step)",
            "for (int i = subtid; i < params.result_width + 1; i += step)",
        ):
            self.assertIn(cooperative_loop, helper)
        self.assertIn("constexpr int step = warp_kernel::kWarpThreads", helper)

        park_begin = source.index("control[cid].state = PqFiniteContextState::WAIT_IO")
        park_end = source.index("return;", park_begin)
        park = source[park_begin:park_end]
        park_store = park.index("store_finite_mutable_state<kQueueSize>")
        park_sync = park.index("__syncwarp();", park_store)
        park_control = park.index("params.d_contexts[query_idx] = control[cid]", park_sync)
        self.assertLess(park_store, park_sync)
        self.assertLess(park_sync, park_control)
        self.assertLess(park.index("params.d_contexts[query_idx] = control[cid]"),
                        park.index("__threadfence_system()"))
        self.assertLess(park.index("__threadfence_system()"),
                        park.index("params.d_status_flag[static_cast<size_t>(query_idx) * 16] = 1"))

        done_begin = source.index("control[cid].state = PqFiniteContextState::DONE")
        done_end = source.index("template <int kQueueSize, int kGraphPqDim>", done_begin)
        done = source[done_begin:done_end]
        flag2 = "params.d_status_flag[static_cast<size_t>(query_idx) * 16] = 2"
        flag3 = "params.d_status_flag[static_cast<size_t>(query_idx) * 16] = 3"
        done_store = done.index("store_finite_mutable_state<kQueueSize>")
        done_sync = done.index("__syncwarp();", done_store)
        done_control = done.index("params.d_contexts[query_idx] = control[cid]", done_sync)
        self.assertLess(done_store, done_sync)
        self.assertLess(done_sync, done_control)
        self.assertLess(done_control, done.index("__threadfence_system()"))
        self.assertLess(done.index(flag2), done.index("params.d_result["))
        self.assertLess(done.index("params.time_per_block["), done.index("__threadfence_system()", done.index(flag2)))
        self.assertLess(done.index("__threadfence_system()", done.index(flag2)), done.index(flag3))

    def test_persistent_fixed_slots_follow_sequence_and_shutdown_contracts(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        header = source[: source.index("struct LatencySummary")]
        for record in (
            "struct alignas(64) PqFinitePersistentRequestControl",
            "struct alignas(64) PqFinitePersistentCompletionControl",
            "struct alignas(16) PqFinitePersistentCompletionNotification",
            "static_assert(sizeof(PqFinitePersistentRequestControl) == 64)",
            "static_assert(sizeof(PqFinitePersistentCompletionControl) == 64)",
            "static_assert(sizeof(PqFinitePersistentCompletionNotification) == 16)",
        ):
            self.assertIn(record, header)

        kernel_begin = source.index("persistent_finite_pq_worker_kernel")
        kernel_end = source.index("persistent_finite_pq_ready_ring_kernel", kernel_begin)
        kernel = source[kernel_begin:kernel_end]
        self.assertIn("blockIdx.x * kMultiQuery", kernel)
        self.assertIn("load_bypass_l1_u32(&request->ready_seq)", kernel)
        self.assertIn("load_bypass_l1_u32(&request->stop)", kernel)
        self.assertIn(
            "run_pq_finite_context<kQueueSize, kGraphPqDim, false, false>",
            kernel,
        )
        self.assertLess(kernel.index("__threadfence_system()"), kernel.index("completion->done_seq"))
        self.assertIn("atomicAdd(completion_tail, 1ULL)", kernel)
        notification_payload = kernel.index("notification->done_seq")
        notification_fence = kernel.index("__threadfence_system()", notification_payload)
        notification_publish = kernel.index("notification->publish_sequence", notification_fence)
        self.assertLess(notification_payload, notification_fence)
        self.assertLess(notification_fence, notification_publish)
        self.assertNotIn("__syncthreads", kernel)

        create_begin = source.index("void create_and_launch_finite_persistent_worker_queue")
        create_end = source.index("uint64_t stop_finite_persistent_worker", create_begin)
        create = source[create_begin:create_end]
        self.assertIn("cudaOccupancyMaxActiveBlocksPerMultiprocessor", create)
        self.assertIn("queue.slot_count = queue.resident_blocks *", create)
        self.assertIn("gdr_pin_buffer", create)
        self.assertIn("cudaHostAlloc", create)
        self.assertIn("cudaHostGetDevicePointer", create)
        self.assertIn("queue.completion_ring_capacity = queue.slot_count", create)
        self.assertIn("cudaMalloc(reinterpret_cast<void**>(&queue.d_completion_tail)", create)
        self.assertIn("queue.h_completion_notifications", create)
        self.assertIn("cudaStreamCreateWithFlags(&queue.stream, cudaStreamNonBlocking)", create)
        self.assertIn("persistent_finite_pq_worker_kernel<QS, PD>", create)

        scheduler_begin = source.index("void drive_finite_persistent_scheduler")
        scheduler_end = source.index("void drive_finite_scheduler", scheduler_begin)
        scheduler = source[scheduler_begin:scheduler_end]
        self.assertIn("completion_queue->try_pop_batch", scheduler)
        self.assertIn("next_finite_persistent_sequence(slot.ready_seq)", scheduler)
        self.assertIn("done_seq != slot.ready_seq", scheduler)
        self.assertIn("completion.query_id != slot.query_id", scheduler)
        self.assertIn("owner_slot[slot.query_id] != slot_idx", scheduler)
        self.assertIn("request.ready_seq = slot.ready_seq", scheduler)
        self.assertIn("expected_publish_sequence = completion_ring_head + 1", scheduler)
        self.assertIn("__atomic_load_n(", scheduler)
        self.assertIn("++completion_ring_head", scheduler)
        self.assertNotIn("for (uint32_t slot_idx = 0; slot_idx < worker_queue.slot_count", scheduler)
        self.assertIn("completion_queue->cancel()", scheduler)
        self.assertIn("stop_finite_persistent_worker(worker_queue)", scheduler)
        self.assertIn("publish_finished_for_host_service(ctx, query_ids)", scheduler)
        self.assertIn("destroy_finite_persistent_worker_queue(worker_queue)", scheduler)
        self.assertLess(
            scheduler.index("stop_finite_persistent_worker(worker_queue)",
                            scheduler.index("} catch (...)")),
            scheduler.index("publish_finished_for_host_service(ctx, query_ids)"),
        )
        self.assertLess(
            scheduler.index("publish_finished_for_host_service(ctx, query_ids)"),
            scheduler.index("destroy_finite_persistent_worker_queue(worker_queue)",
                            scheduler.index("} catch (...)")),
        )
        self.assertNotIn("if (print_summary)", scheduler)
        for field in (
            "worker_blocks=%u",
            "worker_slots=%u",
            "active_blocks_per_sm=%d",
            "active_warps_per_sm=%d",
            "submissions=%zu",
            "completions=%zu",
            "parked=%zu",
            "done=%zu",
            "rejected=%zu",
            "max_busy_slots=%zu",
            "max_ready_backlog=%zu",
            "host_loop_iterations=%zu",
            "idle_yields=%zu",
            "completion_ring_capacity=%u",
            "completion_ring_events=%zu",
            "completion_ring_polls=%zu",
            "completion_ring_empty_polls=%zu",
            "gpu_advance_ms=%.3f",
            "stop_latency_ms=%.3f",
        ):
            self.assertIn(field, scheduler)

    def test_completion_queue_scheduler_telemetry_is_always_emitted(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        begin = source.index("void drive_finite_scheduler")
        end = source.index("void destroy_persistent_latency_queue", begin)
        scheduler = source[begin:end]
        self.assertIn("if (completion_queue != nullptr || print_summary)", scheduler)
        self.assertIn("dispatched_token_count += ready_tokens.size()", scheduler)
        self.assertIn("dispatched_tokens=%zu", scheduler)
        self.assertIn("average_batch=%.2f", scheduler)
        self.assertIn("partial_timeout_us=%lld", scheduler)
        self.assertIn("partial_batch_timeout.count()", scheduler)
        self.assertIn("[pq finite completion queue]", scheduler)
        for field in (
            "duplicates=%llu",
            "invalid_rejections=%llu",
            "terminal_rejections=%llu",
            "cancelled_rejections=%llu",
        ):
            self.assertIn(field, scheduler)
        for field in (
            "transport=%s",
            "host_staging_ms=%.3f",
            "h2d_ms=%.3f",
            "resume_kernel_sync_ms=%.3f",
            "d2h_ms=%.3f",
        ):
            self.assertIn(field, scheduler)

    def test_ready_ring_removes_per_step_host_worker_round_trip(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        scheduler = (SRC_DIR / "search/pq/pq_finite_scheduler.h").read_text(encoding="utf-8")
        header = source[: source.index("struct LatencySummary")]
        for record in (
            "struct alignas(64) PqFiniteReadyRingControl",
            "struct alignas(64) PqFiniteReadyRingEntry",
            "struct alignas(64) PqFiniteReadyRingStatus",
            "static_assert(sizeof(PqFiniteReadyRingEntry) == 64)",
        ):
            self.assertIn(record, header)

        kernel_begin = source.index("persistent_finite_pq_ready_ring_kernel")
        kernel_end = source.index("resident_pq_labels_match", kernel_begin)
        kernel = source[kernel_begin:kernel_end]
        self.assertIn("atomicAdd(consumer_head, 1ULL)", kernel)
        self.assertIn("publish_sequence != expected_publish_sequence", kernel)
        self.assertIn("load_bypass_l1_u32(&control->stop)", kernel)
        self.assertIn("consumed_sequences[entry_idx] = ticket + 1", kernel)
        self.assertIn(
            "run_pq_finite_context<kQueueSize, kGraphPqDim, false, "
            "kFinitePerHopTiming>",
            kernel,
        )
        error_claim = kernel.index("atomicCAS(const_cast<uint32_t*>(&control->error_claimed)")
        error_payload = kernel.index("status->error_query_id", error_claim)
        error_fence = kernel.index("__threadfence_system()", error_payload)
        error_publish = kernel.index("status->error_result", error_fence)
        self.assertLess(error_claim, error_payload)
        self.assertLess(error_payload, error_fence)
        self.assertLess(error_fence, error_publish)
        self.assertNotIn("PqFinitePersistentCompletionControl", kernel)
        self.assertNotIn("__syncthreads", kernel)

        publish_begin = source.index("bool PqFiniteReadyRingSink::publish_ready")
        publish_end = source.index("bool PqFiniteReadyRingSink::publish_done", publish_begin)
        publish = source[publish_begin:publish_end]
        token_write = publish.index("entry.token.query_id")
        first_fence = publish.index("_mm_sfence()", token_write)
        sequence_write = publish.index("entry.publish_sequence", first_fence)
        self.assertLess(token_write, first_fence)
        self.assertLess(first_fence, sequence_write)
        self.assertIn("producer_tail_.fetch_add", publish)
        self.assertIn("expected_consumed_sequence", publish)
        self.assertIn("finite_ready_ring_required_consumed_sequence", publish)
        self.assertIn("queue_.h_consumed_sequences", publish)
        self.assertIn("finite_ready_ring_try_claim_ready(query_state, generation)", publish)
        self.assertIn("finite_ready_ring_finish_ready(query_state, generation)", publish)
        self.assertLess(
            publish.index("context_epoch_.load"),
            publish.index("finite_ready_ring_try_claim_ready"),
        )
        self.assertLess(
            publish.index("entry.publish_sequence"),
            publish.rindex("finite_ready_ring_finish_ready"),
        )
        self.assertNotIn("last_generations_", publish)
        self.assertNotIn("done_queries_", publish)
        self.assertNotIn("ready_enqueued_.fetch_add", publish)

        done_begin = publish_end
        done_end = source.index("void PqFiniteReadyRingSink::set_context_epoch", done_begin)
        done = source[done_begin:done_end]
        self.assertIn("finite_ready_ring_try_claim_done(query_state)", done)
        self.assertIn("std::this_thread::yield()", done)

        self.assertIn("kPqFiniteReadyRingPublishingBit", scheduler)
        self.assertIn("finite_ready_ring_query_publishing(prior_state)", scheduler)
        self.assertIn("return PqFiniteReadyRingClaimResult::PUBLISHING", scheduler)
        self.assertIn("query_state.store", scheduler)
        self.assertIn("std::memory_order_release", scheduler)

        launch_begin = source.index("void PqFiniteReadyRingSink::launch")
        launch_end = source.index("uint64_t PqFiniteReadyRingSink::stop", launch_begin)
        launch = source[launch_begin:launch_end]
        self.assertIn("cudaOccupancyMaxActiveBlocksPerMultiprocessor", launch)
        self.assertIn("persistent_finite_pq_ready_ring_kernel<QS, PD>", launch)
        self.assertIn("queue_.resident_blocks", launch)

        driver_begin = source.index("void drive_finite_ready_ring_scheduler")
        driver_end = source.index("void drive_finite_scheduler", driver_begin)
        driver = source[driver_begin:driver_end]
        self.assertIn("finite PQ ready-ring init completion", driver)
        self.assertIn("ready_ring->launch", driver)
        self.assertIn("ready_ring->done_count()", driver)
        self.assertIn("ready_ring->stop()", driver)
        self.assertNotIn("try_pop_batch", driver)
        self.assertNotIn("wait_pop_batch", driver)
        self.assertIn("ready_timing=unavailable", driver)

    def test_finite_resume_closes_old_mailbox_before_next_request(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        resume_begin = source.index("control[cid].state = PqFiniteContextState::READY;")
        resume_end = source.index("consume_ready_payload = !kInitialize;", resume_begin)
        resume = source[resume_begin:resume_end]
        close_status = (
            "params.d_status_flag[static_cast<size_t>(query_idx) * 16] = 0;"
        )
        self.assertIn(close_status, resume)
        self.assertLess(
            resume.index(close_status), resume.index("__threadfence_system();")
        )

    def test_ready_ring_per_hop_timing_is_compile_time_gated_and_worker_local(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        makefile = (SRC_DIR / "Makefile").read_text(encoding="utf-8")

        self.assertIn("PQ_FINITE_PERHOP_TIMING ?= 0", makefile)
        self.assertIn("struct alignas(64) PqFinitePerHopTiming", source)
        self.assertIn("static_assert(sizeof(PqFinitePerHopTiming) == 64)", source)
        kernel_begin = source.index("persistent_finite_pq_ready_ring_kernel(")
        kernel_end = source.index("resident_pq_labels_match", kernel_begin)
        kernel = source[kernel_begin:kernel_end]
        self.assertIn("__shared__ PqFinitePerHopTiming per_hop_timing", kernel)
        self.assertIn("timing_output[slot_idx] = per_hop_timing[cid]", kernel)
        self.assertNotIn("atomicAdd(&per_hop_timing", kernel)
        self.assertIn("ring_wait_cycles", kernel)
        self.assertIn("ring_metadata_cycles", kernel)
        self.assertIn("restore_cycles", source)
        self.assertIn("payload_cycles", source)
        self.assertIn("writeback_cycles", source)
        self.assertIn("host_publish_sample_rate=1/4096", source)
        self.assertIn("[pq finite per-hop] timing=disabled", source)

    def test_finite_token_transports_share_one_synchronous_stream_owner(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        begin = source.index("void drive_finite_scheduler")
        end = source.index("void destroy_persistent_latency_queue", begin)
        scheduler = source[begin:end]

        self.assertIn("const cudaStream_t scheduler_stream = nullptr", scheduler)
        self.assertEqual(scheduler.count("finite_pq_resume_kernel<QS, PD>"), 1)
        self.assertIn("scheduler_stream>>>", scheduler)
        self.assertIn("cudaStreamSynchronize(scheduler_stream)", scheduler)
        self.assertNotIn("cudaStreamCreate", scheduler)
        mapped = scheduler.index("PqFiniteTokenTransport::Mapped")
        release = scheduler.index(
            "std::atomic_thread_fence(std::memory_order_release)", mapped
        )
        launch = scheduler.index("finite_pq_resume_kernel<QS, PD>", release)
        synchronize = scheduler.index("cudaStreamSynchronize(scheduler_stream)", launch)
        acquire = scheduler.index(
            "std::atomic_thread_fence(std::memory_order_acquire)", synchronize
        )
        self.assertLess(release, launch)
        self.assertLess(launch, synchronize)
        self.assertLess(synchronize, acquire)
        self.assertIn("cudaMemcpyHostToDevice", scheduler)
        self.assertIn("cudaMemcpyDeviceToHost", scheduler)

    def test_host_scheduler_flushes_partial_tail_and_fails_fast_on_reject(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        begin = source.index("void drive_finite_scheduler")
        end = source.index("void destroy_persistent_latency_queue", begin)
        scheduler = source[begin:end]
        self.assertIn(
            "std::chrono::microseconds(options.finite_partial_batch_timeout_us)",
            scheduler,
        )
        self.assertIn(
            "completion_queue->wait_pop_batch(completion_events,\n"
            "                                                  ctx.finite_ready_token_capacity,\n"
            "                                                  partial_batch_timeout)",
            scheduler,
        )
        self.assertIn("partial_timeout_ns", scheduler)
        self.assertNotIn("kFinitePartialBatchTimeout", scheduler)
        self.assertIn("finite_ready_batch_should_flush", scheduler)
        self.assertIn("ready_tokens.size() < ctx.finite_ready_token_capacity", scheduler)
        self.assertIn("finite PQ resume rejected token", scheduler)
        self.assertIn("finite_no_progress_timeout()", scheduler)
        self.assertIn("GO_FANN_FINITE_NO_PROGRESS_TIMEOUT_MS", source)
        self.assertIn("duplicate_rejections=", scheduler)
        self.assertIn("nonterminal_samples=", scheduler)
        self.assertIn("graph_status_flag_finished(flag)", scheduler)

    def test_completion_fast_path_supports_resident_graph(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        begin = source.index("bool use_finite_completion_fast_path")
        end = source.index("size_t checked_finite_state_total_bytes", begin)
        predicate = source[begin:end]
        for condition in (
            "pq_query_worker_mode_is_finite(options.query_worker_mode)",
            "return true",
            "ctx.buffer_mode != PqBufferMode::DoubleSlotSerial",
        ):
            self.assertIn(condition, predicate)

        for function_name in ("void execute_graph_batch", "void execute_graph_query_ids"):
            run_begin = source.index(function_name)
            run_end = source.index("copy_time_per_block_to_host", run_begin)
            run = source[run_begin:run_end]
            self.assertIn("AsyncIoConfig run_io_config = ctx.io_config", run)
            self.assertIn("run_io_config.completion_sink = completion_sink", run)
            self.assertIn("PqFiniteReadyRingSink", run)
            self.assertIn("persistent PQ io_config must not retain a completion sink", run)
            self.assertNotIn("ctx.io_config.completion_sink =", run)

    def test_memory_ready_and_done_publication_order(self) -> None:
        source = (SRC_DIR / "execution/async_io.cpp").read_text(encoding="utf-8")
        check_begin = source.index("SyncPublishCheckResult sync_publish_check")
        check_end = source.index("void publish_in_memory_mailbox_ready", check_begin)
        check = source[check_begin:check_end]
        self.assertIn("config.completion_sink != nullptr", check)
        self.assertIn("? SyncCheckMode::Full", check)

        publish_begin = source.index("void publish_in_memory_mailbox_ready")
        publish_end = source.index("constexpr int64_t kHostHeldStatus", publish_begin)
        publish = source[publish_begin:publish_end]
        gdr_store = publish.index("write_in_memory_mailbox_ready(")
        sink_guard = publish.index("if (config.completion_sink != nullptr)", gdr_store)
        fence = publish.index("_mm_sfence()", sink_guard)
        ready = publish.index("publish_host_ready", fence)
        self.assertLess(gdr_store, sink_guard)
        self.assertLess(sink_guard, fence)
        self.assertLess(fence, ready)

        worker_begin = source.index("void host_scan_and_set_worker")
        worker_end = source.index("AsyncIoConfig::AsyncIoConfig", worker_begin)
        worker = source[worker_begin:worker_end]
        single_begin = source.index("void host_scan_and_set(")
        single_end = source.index("void print_host_filter_summary", single_begin)
        single = source[single_begin:single_end]
        for service in (worker, single):
            terminal = service.index("if (graph_status_flag_finished(flag))")
            done = service.index("publish_host_done(config.completion_sink", terminal)
            self.assertLess(terminal, done)
            self.assertEqual(service.count("publish_host_done(config.completion_sink"), 1)
            self.assertIn("publish_in_memory_mailbox_ready", service)

    def test_obsolete_lane_service_is_removed(self) -> None:
        self.assertFalse((SRC_DIR / "io/compat/lane_cache_inline.cpp").exists())
        self.assertFalse((SRC_DIR / "io/compat/lane_disk_io.cpp").exists())
        self.assertFalse((SRC_DIR / "execution/lane_host_service.cpp").exists())
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        self.assertNotIn("ctx.disk_config", source)
        self.assertNotIn("runtime.graph_on_disk", source)
        self.assertIn("std::thread(host_scan_and_set,", source)

    def test_queue_and_scan_discovery_are_both_explicit_and_failure_safe(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        scheduler_begin = source.index("void drive_finite_scheduler")
        scheduler_end = source.index("void destroy_persistent_latency_queue", scheduler_begin)
        scheduler = source[scheduler_begin:scheduler_end]
        self.assertIn('"completion_queue" : "mailbox_scan"', scheduler)
        self.assertIn("completion_queue->wait_pop_batch", scheduler)
        self.assertIn("load_finite_mailbox_snapshot", scheduler)
        self.assertIn("[pq finite completion queue]", scheduler)

        self.assertGreaterEqual(source.count("completion_sink->cancel()"), 2)
        for catch_pos in (
            source.index("} catch (...)", source.index("void execute_graph_batch")),
            source.index("} catch (...)", source.index("void execute_graph_query_ids")),
        ):
            catch_body = source[catch_pos : catch_pos + 900]
            self.assertLess(
                catch_body.index("completion_sink->cancel()"),
                catch_body.index("publish_finished_for_host_service"),
            )

    def test_null_sink_is_default_and_host_service_borrows_it(self) -> None:
        header = (SRC_DIR / "execution/async_io.h").read_text(encoding="utf-8")
        service = (SRC_DIR / "execution/async_io.cpp").read_text(encoding="utf-8")
        self.assertIn("HostCompletionSink* completion_sink = nullptr", header)
        self.assertIn("Borrowed for one host-service invocation", header)
        self.assertIn("config.completion_sink", service)
        self.assertNotIn("delete config.completion_sink", service)

    def test_original_query_ids_are_validated_before_finite_init_launch(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        begin = source.index("void execute_graph_query_ids")
        end = source.index("struct PqLowfreqSearchContext", begin)
        execute = source[begin:end]
        self.assertLess(
            execute.index("require_valid_finite_query_ids(query_ids"),
            execute.index("launch_finite_init(ctx"),
        )

    def test_scan_thread_guard_and_streaming_cancel_are_failure_safe(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        guard_begin = source.index("class PqHostScanThreadGuard")
        guard_end = source.index("void stop_persistent_latency_worker", guard_begin)
        guard = source[guard_begin:guard_end]
        self.assertIn("abort_and_join();", guard)
        self.assertIn("publish_finished_for_host_service", guard)
        self.assertIn("thread_.join()", guard)
        self.assertIn("cancel_requested->load(std::memory_order_acquire)", source)
        self.assertIn("streaming_rerank_cancelled.store(true", source)

    def test_default_coupled_launch_is_still_the_existing_kernel(self) -> None:
        source = (SRC_DIR / "search/pq/backend_pq.cu").read_text(encoding="utf-8")
        for function_name in ("void execute_graph_batch", "void execute_graph_query_ids"):
            begin = source.index(function_name)
            end = source.index("copy_time_per_block_to_host", begin)
            launch = source[begin:end]
            self.assertIn("pq_query_worker_mode_is_finite(options.query_worker_mode)", launch)
            self.assertIn("launch_finite_init", launch)
            self.assertRegex(
                launch,
                r"native_warp_independent_search_kernel_impl<QS,\s*PD,\s*"
                r"kUseRequestRing,\s*kDeviceResident,\s*kPredicateBias>",
            )
            self.assertIn("launch.template operator()<false, false>()", launch)


if __name__ == "__main__":
    unittest.main()
