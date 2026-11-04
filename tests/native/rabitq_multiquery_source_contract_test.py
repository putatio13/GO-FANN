#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "src"
SOURCE = (ROOT / "search/rabitq/backend_rabitq.cu").read_text(encoding="utf-8")
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")


def require(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"[OK] {name}")


def source_between(begin_marker: str, end_marker: str) -> str:
    begin = SOURCE.index(begin_marker)
    end = SOURCE.index(end_marker, begin)
    return SOURCE[begin:end]


def main() -> None:
    require("RabitQ MQ build knob defaults to one", "RABITQ_MULTI_QUERY ?= 1" in MAKEFILE)
    require("RabitQ register cap is backend-local", "RABITQ_MAX_REGISTERS ?=" in MAKEFILE)
    require(
        "RabitQ detailed timing knob is backend-local",
        "RABITQ_NATIVE_DETAILED_TIMING ?= 1" in MAKEFILE,
    )
    require(
        "RabitQ cooperative pickup defaults on",
        "RABITQ_NATIVE_COOPERATIVE_PICKUP ?= 1" in MAKEFILE,
    )
    require(
        "RabitQ object receives only RabitQ compile flags",
        "backend_rabitq.o: NVCC_FLAGS += $(RABITQ_NVCC_FLAGS)" in MAKEFILE,
    )
    global_nvcc_assignments = "\n".join(
        line for line in MAKEFILE.splitlines() if line.startswith("NVCC_FLAGS")
    )
    require(
        "RabitQ compile controls do not leak into global CUDA flags",
        "GO_FANN_RABITQ" not in global_nvcc_assignments
        and "RABITQ_MAX_REGISTERS" not in global_nvcc_assignments,
    )
    require(
        "RabitQ cooperative pickup flag is backend-local",
        "-DGO_FANN_RABITQ_NATIVE_COOPERATIVE_PICKUP="
        "$(RABITQ_NATIVE_COOPERATIVE_PICKUP)" in MAKEFILE,
    )
    require(
        "native MQ supports one to four query warps per block",
        "kNativeMultiQuery >= 1 && kNativeMultiQuery <= 4" in SOURCE,
    )
    require(
        "native threads scale by independent query warps",
        "kNativeThreadInBlock = kCriticalStep * kNativeMultiQuery" in SOURCE,
    )
    require(
        "native route index combines block and warp",
        "blockIdx.x * kNativeMultiQuery" in SOURCE,
    )
    require(
        "partial final blocks are bounded by active query count",
        "active_query_idx >= static_cast<uint32_t>(num_query)" in SOURCE,
    )
    require(
        "native launch ceiling-divides the active query count",
        "(active_query_count + kNativeMultiQuery - 1) / kNativeMultiQuery"
        in SOURCE,
    )
    hot_path = source_between(
        "__device__ void run_rabitq_search_one_block",
        "void native_rabitq_search_kernel",
    )
    require(
        "native query hot path has no block-wide barrier",
        "__syncthreads" not in hot_path and "__syncwarp" in hot_path,
    )
    require(
        "graph hash scratch remains active-route indexed",
        "static_cast<size_t>(active_query_idx) * kHashTableCapacity" in SOURCE,
    )
    require(
        "graph distance scratch remains active-route indexed",
        "static_cast<size_t>(active_query_idx) * kGraphDatadistSlotCapacity" in SOURCE,
    )
    require(
        "traversal payload is compact",
        'sizeof(RabitqPoolPair) == 8' in SOURCE,
    )
    require(
        "result payload is compact",
        'sizeof(RabitqResultPair) == 8' in SOURCE,
    )
    require(
        "lane-zero state has an explicit size gate",
        'sizeof(RabitqLane0SearchState) == 28' in SOURCE,
    )
    require(
        "lane-zero heap sizes retain their validated bounded width",
        "uint16_t topk_heap_size" in SOURCE
        and "uint16_t stop_topk_heap_size" in SOURCE
        and "stop_topk_approx_worst_bits" not in hot_path,
    )
    require(
        "native hash avoids device heap allocation",
        "new FixHash" not in hot_path and "delete lane0_state[cid].hash" not in hot_path,
    )
    require(
        "native hash reuses reset scratch without kernel-side full clear",
        "native_visited_hash::add<kHashTableCapacity>(d_hash_scratch" in hot_path
        and "native_visited_hash::clear" not in hot_path,
    )
    require(
        "native result heap is right-sized while frontier stays compile-time shared",
        "rabitq_result_heap_shared_bytes(ctx.result_width)" in SOURCE
        and "__shared__ FrontierPair q[kQueriesPerBlock][kQueueSize + 2]" in SOURCE
        and "using FrontierPair = std::conditional_t<kPredicateBias, gpu_predicate::BiasedPair, RabitqPoolPair>" in SOURCE,
    )
    require(
        "native result heap uses launch-sized dynamic shared memory",
        "<<<grid_dim,kNativeThreadInBlock,rabitq_result_heap_shared_bytes(ctx.result_width)>>>"
        in "".join(SOURCE.split()),
    )
    require(
        "persistent latency remains one-query state",
        "run_rabitq_search_one_block<kQueueSize, 1, true, false, false, false, false, false>"
        in SOURCE,
    )
    require(
        "persistent latency launch remains one warp",
        "persistent_rabitq_latency_kernel<QS><<<queue.slot_count, kThreadInBlock>>>" in SOURCE,
    )
    require(
        "runtime reports query-warp occupancy",
        "active_query_warps_per_sm=%d" in SOURCE,
    )
    require(
        "native search targets 24 resident query warps per SM",
        "kNativeTargetWarpsPerSm = 24" in SOURCE
        and "__launch_bounds__(kNativeThreadInBlock, kNativeMinBlocksPerSm)" in SOURCE,
    )
    pickup = source_between(
        "__device__ bool pickup_next_unused_from_native_frontier",
        "struct RabitqBatchStats",
    )
    require(
        "cooperative pickup scans one frontier stripe per lane",
        "for (int i = lane + 1; i < size; i += warp_kernel::kWarpThreads)" in pickup,
    )
    require(
        "cooperative pickup reduces minima with stable index tie breaking",
        "__shfl_down_sync(warp_kernel::kFullMask" in pickup
        and "other_index < best_index" in pickup,
    )
    require(
        "cooperative pickup has one writer and warp synchronization",
        "if (lane == 0 && found)" in pickup
        and "mark_pool_payload_used" in pickup
        and "__syncwarp" in pickup,
    )
    native_begin = SOURCE.index("void native_rabitq_search_kernel")
    persistent_begin = SOURCE.index("__global__ void persistent_rabitq_latency_kernel")
    native_wrapper = SOURCE[native_begin:persistent_begin]
    require(
        "native wrapper dynamically allocates only result workspaces",
        "extern __shared__ RabitqResultPair topk_workspaces[]" in native_wrapper
        and "frontier_workspaces" not in native_wrapper,
    )
    persistent_end = SOURCE.index("struct RabitqPersistentLatencyQueue", persistent_begin)
    persistent_wrapper = SOURCE[persistent_begin:persistent_end]
    require(
        "persistent latency retains static result storage",
        "__shared__ RabitqResultPair topk_workspace" in persistent_wrapper
        and "extern __shared__" not in persistent_wrapper,
    )
    completion = hot_path[hot_path.rindex("d_status_flag[route_idx * 16] = 2;") :]
    require(
        "completion closes only after a system visibility fence",
        completion.index("d_status_flag[route_idx * 16] = 2;")
        < completion.index("__threadfence_system();")
        < completion.index("d_status_flag[route_idx * 16] = 3;"),
    )


if __name__ == "__main__":
    main()
