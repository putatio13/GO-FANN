#!/usr/bin/env python3
"""Structural contracts for the opt-in RabitQ finite batch scheduler."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "src"
BACKEND = (ROOT / "search/rabitq/backend_rabitq.cu").read_text(encoding="utf-8")
MAIN = (ROOT / "app/main.cu").read_text(encoding="utf-8")
LOADER = (ROOT / "app/config_loader.cpp").read_text(encoding="utf-8")
ASYNC_HEADER = (ROOT / "execution/async_io.h").read_text(encoding="utf-8")
ASYNC_SOURCE = (ROOT / "execution/async_io.cpp").read_text(encoding="utf-8")


def require(text: str, message: str) -> None:
    if text not in BACKEND:
        raise AssertionError(message)


def main() -> None:
    for text, message in (
        ("RabitqFiniteContextControl", "finite context control is missing"),
        ("context_epoch", "finite context must carry an epoch"),
        ("io_generation", "finite context must carry an I/O generation"),
        ("expected_row", "finite context must carry an expected row"),
        ("current_buffer", "finite context must carry an expected buffer"),
        ("control.scratch_slot", "finite mutable state must use compact scratch slots"),
        ("static_cast<size_t>(active_query_idx) * kHashTableCapacity",
         "finite visited state must be initialized by active scratch slot"),
        ("finite_rabitq_init_kernel", "finite initialization kernel is missing"),
        ("finite_rabitq_resume_kernel", "finite resume kernel is missing"),
        ("kRabitqStaticPrefetchStepLimit", "finite static prefetch step bound is missing"),
        ("params.d_prefetch_node_array = ctx.d_prefetch_node_array",
         "finite static node table is not wired into kernel params"),
        ("has_mailbox_payload == 0 && !node_prefetched",
         "finite static misses must fall back to the mailbox path"),
        ("has_mailbox_payload == 0 &&\n            control.step_num < kRabitqStaticPrefetchStepLimit",
         "finite static prefetch must match the coupled first-three-step policy"),
        ("RabitqFiniteAdvanceResult::REJECT_GENERATION", "generation rejection is missing"),
        ("RabitqFiniteAdvanceResult::REJECT_ROW", "row rejection is missing"),
        ("RabitqFiniteAdvanceResult::REJECT_BUFFER", "buffer rejection is missing"),
        ("RabitqFiniteAdvanceResult::REJECT_STATUS", "status rejection is missing"),
        ("RabitqFiniteAdvanceResult::REJECT_MAILBOX", "mailbox readiness check is missing"),
        ("PqFiniteCompletionQueue", "bounded completion queue is missing"),
        ("finite RabitQ scheduler made no progress for 60 s", "finite watchdog is missing"),
        ("finite query workers currently require highfreq_rerank_mode=cpu or off",
         "finite GPU exact rerank must be rejected"),
        ("if constexpr (kUseRequestRing)",
         "scan and request-ring kernels must be compile-time isolated"),
        ("request_ring_capacity > 0 ? active_query_idx / request_ring_capacity",
         "GPU request routing must follow logical service batches"),
        ("host_request_ring_and_set",
         "coupled memory search does not launch the request-ring consumer"),
        ("cudaMalloc RabitQ request ring consumed sequences",
         "RabitQ device consumed-sequence allocation is missing"),
        ("gdr_pin_buffer RabitQ request ring consumed sequences",
         "RabitQ device consumed-sequence pin is missing"),
        ("gdr_map RabitQ request ring consumed sequences",
         "RabitQ device consumed-sequence mapping is missing"),
        ("gdr_unmap(ctx.g,\n                  ctx.mh_request_ring_consumed_sequences",
         "RabitQ device consumed-sequence mapping is not released"),
        ("gdr_unpin_buffer(ctx.g, ctx.mh_request_ring_consumed_sequences)",
         "RabitQ device consumed-sequence pin is not released"),
        ("cudaFree(ctx.d_request_ring_consumed_sequences_raw)",
         "RabitQ device consumed-sequence allocation is not released"),
        ("ctx.request_ring_consumed_device_mapped};",
         "RabitQ request-ring view does not select the configured writeback"),
        ("request_consumed_memory must be one of: host, device",
         "RabitQ backend consumed-memory validation is missing"),
        ("GO_FANN_RABITQ_GRAPH_PROFILE",
         "opt-in RabitQ graph profiling gate is missing"),
        ("[rabitq gpu phases]",
         "GPU graph phase summary is missing"),
        ("highfreq_end_to_end_ms",
         "high-frequency end-to-end wall timing is missing"),
        ("RabitqGpuGraphRowCacheEntry", "GPU graph row cache entry is missing"),
        ("rabitq_gpu_graph_cache_load", "GPU graph row cache lookup is missing"),
        ("rabitq_gpu_graph_cache_store", "GPU graph row cache fill is missing"),
        ("atom.acquire.gpu.global.add.u64",
         "GPU graph row cache lookup must acquire published rows"),
        ("atom.release.gpu.global.add.u64",
         "GPU graph row cache fill must release complete rows"),
        ("RabitqGpuGraphCacheDeviceView",
         "GPU graph cache kernel arguments must use a compact device view"),
        ("__launch_bounds__(kNativeThreadInBlock, kNativeMinBlocksPerSm)",
         "native RabitQ search must preserve its query-warp occupancy target"),
        ("kNativeTargetWarpsPerSm = 24",
         "native RabitQ search occupancy target changed unexpectedly"),
        ("rabitq_gpu_full_labels_match",
         "GPU graph row cache does not preserve full-label filtering"),
        ("GPU graph row cache currently supports graph degree <= 32",
         "unsupported graph degrees must fail fast"),
        ("reset_between_runs=0", "GPU graph row cache warmup policy is not logged"),
        ("RabitqGpuGraphRowCacheView", "GPU graph cache sharing view is missing"),
        ("owns_gpu_graph_row_cache", "shared GPU graph cache ownership is missing"),
        ("clear_rabitq_gpu_graph_row_cache", "GPU graph cache warmup cannot clear stale rows"),
        ("resident_graph", "device-resident RabitQ graph pointer is missing"),
        ("bool kDeviceResident", "device-resident RabitQ kernel specialization is missing"),
        ("graph transport=device_resident", "device-resident RabitQ audit log is missing"),
        ("if (ctx.device_resident_graph)",
         "device-resident RabitQ host service bypass is missing"),
    ):
        require(text, message)

    for text, source, message in (
        ("--rabitq-query-worker-mode=", MAIN, "finite CLI mode is missing"),
        ("--rabitq-graph-transport=", MAIN,
         "device-resident RabitQ CLI mode is missing"),
        ("rabitq finite dispatch currently supports only batch", MAIN,
         "unsupported finite dispatch must fail fast"),
        ("rabitq_query_worker_mode", LOADER, "finite config root key is missing"),
        ("rabitq_graph_transport", LOADER,
         "device-resident RabitQ config root key is missing"),
        ("graph_transport", LOADER,
         "device-resident RabitQ nested config key is missing"),
        ("query_worker_mode", LOADER, "finite nested config key is missing"),
        ("--rabitq-request-mode=", MAIN, "request-ring CLI mode is missing"),
        ("--rabitq-request-consumed-memory=", MAIN,
         "request-ring consumed-memory CLI mode is missing"),
        ("rabitq_request_mode", LOADER, "request-ring config root key is missing"),
        ("rabitq_request_consumed_memory", LOADER,
         "request-ring consumed-memory config root key is missing"),
        ("request_batch_queries", LOADER, "request-ring nested batch key is missing"),
        ("request_consumed_memory", LOADER,
         "request-ring nested consumed-memory key is missing"),
        ("--rabitq-gpu-graph-row-cache-mib=", MAIN,
         "GPU graph row cache CLI size is missing"),
        ("rabitq_gpu_graph_row_cache_mib", LOADER,
         "GPU graph row cache root config key is missing"),
        ("gpu_graph_row_cache_mib", LOADER,
         "GPU graph row cache nested config key is missing"),
        ("static_assert(sizeof(HostRequestRingEntry) == 16", ASYNC_HEADER,
         "request-ring records must contain only the notification payload"),
        ("ring_idx += static_cast<size_t>(worker_count)", ASYNC_SOURCE,
         "request-ring batches must be striped across host workers"),
        ("kDrainBudgetPerRing", ASYNC_SOURCE,
         "request-ring workers must drain multiple active rings fairly"),
        ("cancellation->load(std::memory_order_acquire)", ASYNC_SOURCE,
         "request-ring workers must support exception-path cancellation"),
        ("request_ring_cancelled.store(true, std::memory_order_release)", BACKEND,
         "request-ring cancellation must precede host-service join"),
        ("expected_ring_by_query[event.query_id] != ring_idx", ASYNC_SOURCE,
         "request-ring consumers must reject cross-batch events"),
        ("GO_FANN_RABITQ_GRAPH_PROFILE_SAMPLE_LOG2", ASYNC_SOURCE,
         "request-ring CPU profiling must be sampled"),
        ("[request-ring profile]", ASYNC_SOURCE,
         "request-ring CPU phase summary is missing"),
    ):
        if text not in source:
            raise AssertionError(message)

    for retired in ("benchmark_cache_warmup", "LaneGraphCacheSession", "DiskVectorCache"):
        if retired in BACKEND:
            raise AssertionError("retired cache warmup remains: " + retired)
    require("options.warmup_runs", "ordinary benchmark warmup must remain")
    require("options.timed_runs", "ordinary timed runs must remain")
    print("RabitQ finite source contracts passed")


if __name__ == "__main__":
    main()
