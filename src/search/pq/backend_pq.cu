#include "search/pq/backend_pq.cuh"
#include "profiling/expansion_trace.cuh"
#include "profiling/performance_trace.h"
#include "execution/rabitq_cpu_worker_pool.h"
#include "search/pq/pq_gpu_predicate.cuh"
#include "search/pq/pq_resident_posting.cuh"
#include "search/pq/pq_posting_streaming.h"
#include "predicates/gpu_predicate_csr.h"
#include "planning/route_predicate_plan.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <filesystem>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <immintrin.h>

#include <cuda.h>
#include <gdrapi.h>
#include <gdrconfig.h>

#include "bin_heap.h"
#include "config1B.h"
#include "fixhash.h"
#include "smmh2.h"
#include "execution/async_io.h"
#include "data/dataloader.h"
#include "profiling/durable_csv.h"
#include "execution/host_request_ring.cuh"
#include "search/common/native_visited_hash.cuh"
#include "search/pq/pq_finite_scheduler.h"
#include "search/common/posting_shortlist_heap.cuh"
#include "search/rabitq/rabitq_lowfreq_merge.h"
#include "profiling/query_setup_timing.h"
#include "predicates/range_filter.cuh"
#include "search/common/warp_kernel.cuh"

namespace go_fann::backend_pq {
namespace perf_trace = go_fann::performance_trace;

// Reuse the backend-independent scheduler; GPU request responders stay separate.
using CpuRerankWorkerPool = backend_rabitq::RabitqCpuWorkerPool;

constexpr int kGraphMaxDim = 192;
constexpr int kGraphPqDim16 = 16;
constexpr int kGraphPqDim64 = 64;
constexpr int kNumPqCentroids = 256;
constexpr int kTopk = 10;
// Note: kQueueSize is now a template parameter for native_warp_independent_search_kernel_impl
#ifndef GO_FANN_PQ_MULTI_QUERY
#define GO_FANN_PQ_MULTI_QUERY 1
#endif
#ifndef GO_FANN_PQ_WARPS_PER_QUERY
#define GO_FANN_PQ_WARPS_PER_QUERY 1
#endif
#ifndef GO_FANN_PQ_EXPERIMENTAL_COUPLED_MULTIWARP
#define GO_FANN_PQ_EXPERIMENTAL_COUPLED_MULTIWARP 0
#endif
#ifndef GO_FANN_PQ_NATIVE_DETAILED_TIMING
#define GO_FANN_PQ_NATIVE_DETAILED_TIMING 1
#endif
#ifndef GO_FANN_PQ_NATIVE_COOPERATIVE_PICKUP
#define GO_FANN_PQ_NATIVE_COOPERATIVE_PICKUP 1
#endif
#ifndef GO_FANN_PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE
#define GO_FANN_PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE 0
#endif
#ifndef GO_FANN_PQ_NATIVE_MAX_HEAP_FRONTIER
#define GO_FANN_PQ_NATIVE_MAX_HEAP_FRONTIER 0
#endif
#ifndef GO_FANN_PQ_NATIVE_PROFILE_MODE
#define GO_FANN_PQ_NATIVE_PROFILE_MODE 0
#endif
#ifndef GO_FANN_PQ_FINITE_PERHOP_TIMING
#define GO_FANN_PQ_FINITE_PERHOP_TIMING 0
#endif
#ifndef GO_FANN_PQ_REQUEST_PUBLISH_TIMING
#define GO_FANN_PQ_REQUEST_PUBLISH_TIMING 0
#endif
constexpr int kMultiQuery = GO_FANN_PQ_MULTI_QUERY;
constexpr int kNativeWarpsPerQuery = GO_FANN_PQ_WARPS_PER_QUERY;
constexpr bool kExperimentalCoupledMultiwarp =
    GO_FANN_PQ_EXPERIMENTAL_COUPLED_MULTIWARP != 0;
constexpr bool kNativeDetailedTiming = GO_FANN_PQ_NATIVE_DETAILED_TIMING != 0;
constexpr bool kNativeCooperativePickup = GO_FANN_PQ_NATIVE_COOPERATIVE_PICKUP != 0;
constexpr int kNativeProfileMode = GO_FANN_PQ_NATIVE_PROFILE_MODE;
constexpr bool kFinitePerHopTiming = GO_FANN_PQ_FINITE_PERHOP_TIMING != 0;
constexpr bool kRequestPublishTiming = GO_FANN_PQ_REQUEST_PUBLISH_TIMING != 0;
constexpr int kCriticalStep = 32;
constexpr int kThreadInBlock = 32;
constexpr int kNativeThreadInBlock = kCriticalStep * kMultiQuery;
constexpr int kNativeResidentThreadInBlock =
    kNativeThreadInBlock * kNativeWarpsPerQuery;
constexpr int kNativeCoupledWarpsPerQuery =
    kExperimentalCoupledMultiwarp ? kNativeWarpsPerQuery : 1;
constexpr int kNativeCoupledThreadInBlock =
    kNativeThreadInBlock * kNativeCoupledWarpsPerQuery;
static_assert(kMultiQuery >= 1 && kMultiQuery <= 4,
              "GO_FANN_PQ_MULTI_QUERY must be in [1, 4]");
static_assert(kNativeWarpsPerQuery == 1 || kNativeWarpsPerQuery == 2 ||
                  kNativeWarpsPerQuery == 4,
              "GO_FANN_PQ_WARPS_PER_QUERY must be one of: 1, 2, 4");
static_assert(kNativeWarpsPerQuery == 1 || kMultiQuery == 1,
              "multi-warp native PQ currently requires PQ_MULTI_QUERY=1");
static_assert(GO_FANN_PQ_EXPERIMENTAL_COUPLED_MULTIWARP == 0 ||
                  GO_FANN_PQ_EXPERIMENTAL_COUPLED_MULTIWARP == 1,
              "GO_FANN_PQ_EXPERIMENTAL_COUPLED_MULTIWARP must be 0 or 1");
static_assert(kNativeWarpsPerQuery == 1 ||
                  GO_FANN_PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE == 0,
              "multi-warp native PQ does not support cooperative frontier update");
static_assert(GO_FANN_PQ_NATIVE_DETAILED_TIMING == 0 ||
                  GO_FANN_PQ_NATIVE_DETAILED_TIMING == 1,
              "GO_FANN_PQ_NATIVE_DETAILED_TIMING must be 0 or 1");
static_assert(GO_FANN_PQ_NATIVE_COOPERATIVE_PICKUP == 0 ||
                  GO_FANN_PQ_NATIVE_COOPERATIVE_PICKUP == 1,
              "GO_FANN_PQ_NATIVE_COOPERATIVE_PICKUP must be 0 or 1");
static_assert(GO_FANN_PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE == 0 ||
                  GO_FANN_PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE == 1,
              "GO_FANN_PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE must be 0 or 1");
static_assert(GO_FANN_PQ_NATIVE_MAX_HEAP_FRONTIER == 0 ||
                  GO_FANN_PQ_NATIVE_MAX_HEAP_FRONTIER == 1,
              "GO_FANN_PQ_NATIVE_MAX_HEAP_FRONTIER must be 0 or 1");
static_assert(GO_FANN_PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE == 0 ||
                  GO_FANN_PQ_NATIVE_MAX_HEAP_FRONTIER == 0,
              "PQ native frontier experiments are mutually exclusive");
static_assert(kNativeProfileMode >= 0 && kNativeProfileMode <= 2,
              "GO_FANN_PQ_NATIVE_PROFILE_MODE must be in [0, 2]");
static_assert(GO_FANN_PQ_FINITE_PERHOP_TIMING == 0 ||
                  GO_FANN_PQ_FINITE_PERHOP_TIMING == 1,
              "GO_FANN_PQ_FINITE_PERHOP_TIMING must be 0 or 1");
static_assert(GO_FANN_PQ_REQUEST_PUBLISH_TIMING == 0 ||
                  GO_FANN_PQ_REQUEST_PUBLISH_TIMING == 1,
              "GO_FANN_PQ_REQUEST_PUBLISH_TIMING must be 0 or 1");
// Multiwarp execution uses resource-derived occupancy; WPQ1 uses a
// V100 request-ring target of 24 independent blocks per SM.
#if GO_FANN_PQ_WARPS_PER_QUERY > 1
#define GO_FANN_PQ_NATIVE_LAUNCH_BOUNDS(queue_size, use_request_ring, device_resident) \
    __launch_bounds__((device_resident) ? kNativeResidentThreadInBlock \
                                        : kNativeCoupledThreadInBlock, \
                      ((device_resident) || kExperimentalCoupledMultiwarp) \
                          ? 1 \
                          : ((use_request_ring) && (queue_size) <= 256 ? 24 : 1))
#define GO_FANN_PQ_FINITE_LAUNCH_BOUNDS(queue_size)
#elif GO_FANN_PQ_MULTI_QUERY == 1
#define GO_FANN_PQ_NATIVE_LAUNCH_BOUNDS(queue_size, use_request_ring, device_resident) \
    __launch_bounds__(kNativeThreadInBlock, \
                      ((use_request_ring) && (queue_size) <= 256 ? 24 : 1))
#define GO_FANN_PQ_FINITE_LAUNCH_BOUNDS(queue_size)
#elif GO_FANN_PQ_MULTI_QUERY == 2
#define GO_FANN_PQ_NATIVE_LAUNCH_BOUNDS(queue_size, use_request_ring, device_resident) \
    __launch_bounds__(kNativeThreadInBlock, ((queue_size) <= 256 ? 6 : 5))
#define GO_FANN_PQ_FINITE_LAUNCH_BOUNDS(queue_size) \
    __launch_bounds__(kNativeThreadInBlock, ((queue_size) <= 256 ? 6 : 5))
#elif GO_FANN_PQ_MULTI_QUERY == 3
#define GO_FANN_PQ_NATIVE_LAUNCH_BOUNDS(queue_size, use_request_ring, device_resident) \
    __launch_bounds__(kNativeThreadInBlock, ((queue_size) <= 256 ? 6 : 5))
#define GO_FANN_PQ_FINITE_LAUNCH_BOUNDS(queue_size) \
    __launch_bounds__(kNativeThreadInBlock, ((queue_size) <= 256 ? 6 : 5))
#elif GO_FANN_PQ_MULTI_QUERY == 4
#define GO_FANN_PQ_NATIVE_LAUNCH_BOUNDS(queue_size, use_request_ring, device_resident) \
    __launch_bounds__(kNativeThreadInBlock, ((queue_size) <= 256 ? 4 : 2))
#define GO_FANN_PQ_FINITE_LAUNCH_BOUNDS(queue_size) \
    __launch_bounds__(kNativeThreadInBlock, ((queue_size) <= 256 ? 4 : 2))
#endif
constexpr int kMultiprobe = 1;
constexpr int kFinishCount = 1;
constexpr int kHashTableCapacity = 4096;
constexpr int kKernelPrefetchHops = 3;
constexpr size_t kGdrAlignBytes = 65536;
constexpr int kInitialWaitSleepUs = 32;
constexpr int kMinWaitSleepUs = 2;
constexpr int kMaxWaitSleepUs = 64;
constexpr int kLowfreqRerankShortlistMax = defaults::max_lowfreq_rerank_shortlist;
constexpr int kHighfreqRerankShortlistMax = defaults::max_highfreq_rerank_shortlist;
constexpr uint32_t kDefaultQueryBatchSize = 2048;
constexpr uint32_t kPersistentLatencyStopSeq = UINT32_MAX;
constexpr size_t kFiniteReadyBatchCapacity = 1024;
constexpr auto kDefaultFiniteNoProgressTimeout = std::chrono::seconds(120);

__device__ __noinline__ void publish_pq_host_request(
    HostRequestRingEntry* entries,
    const volatile uint64_t* consumed_sequences,
    unsigned long long* tails,
    uint32_t ring_idx,
    uint32_t capacity_per_ring,
    uint32_t query_id,
    HostRequestKind kind)
{
    request_ring_device::publish_host_request(entries,
                                              consumed_sequences,
                                              tails,
                                              ring_idx,
                                              capacity_per_ring,
                                              query_id,
                                              kind);
}

template <bool kCollectTiming>
__device__ __forceinline__ void publish_profiled_pq_host_request(
    HostRequestRingEntry* entries,
    const volatile uint64_t* consumed_sequences,
    unsigned long long* tails,
    uint32_t ring_idx,
    uint32_t capacity_per_ring,
    uint32_t query_id,
    HostRequestKind kind,
    uint64_t& publish_ns)
{
    uint64_t start_ns = 0;
    if constexpr (kCollectTiming) {
        start_ns = warp_kernel::get_global_timer_ns();
    }
    publish_pq_host_request(entries,
                            consumed_sequences,
                            tails,
                            ring_idx,
                            capacity_per_ring,
                            query_id,
                            kind);
    if constexpr (kCollectTiming) {
        publish_ns += warp_kernel::get_global_timer_ns() - start_ns;
    }
}

std::chrono::milliseconds finite_no_progress_timeout()
{
    static const std::chrono::milliseconds timeout = [] {
        const char* value = std::getenv("GO_FANN_FINITE_NO_PROGRESS_TIMEOUT_MS");
        if (value == nullptr || *value == '\0') {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                kDefaultFiniteNoProgressTimeout);
        }
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (end == value || *end != '\0' || parsed == 0 ||
            parsed > static_cast<unsigned long long>(std::numeric_limits<int64_t>::max())) {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                kDefaultFiniteNoProgressTimeout);
        }
        return std::chrono::milliseconds(parsed);
    }();
    return timeout;
}

struct alignas(64) PqPersistentLatencyRequestControl {
    uint32_t ready_seq = 0;
    uint32_t stop = 0;
    uint32_t query_id = UINT32_MAX;
    uint32_t lowfreq_route = 0;
};

struct alignas(64) PqPersistentLatencyCompletionControl {
    uint32_t done_seq = 0;
    uint32_t query_id = UINT32_MAX;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint64_t gpu_start_clock = 0;
    uint64_t gpu_stop_clock = 0;
};

struct alignas(64) PqFinitePersistentRequestControl {
    uint32_t ready_seq = 0;
    uint32_t stop = 0;
    uint32_t query_id = UINT32_MAX;
    uint32_t context_epoch = 0;
    uint32_t generation = 0;
    uint32_t row = UINT32_MAX;
    uint32_t buffer = 0;
    uint32_t reserved0 = 0;
    uint64_t ready_ns = 0;
    uint64_t reserved1[3]{};
};

struct alignas(64) PqFinitePersistentCompletionControl {
    uint32_t done_seq = 0;
    PqFiniteAdvanceResult result = PqFiniteAdvanceResult::PENDING;
    uint32_t query_id = UINT32_MAX;
    uint32_t generation = 0;
    uint64_t gpu_start_ns = 0;
    uint64_t gpu_stop_ns = 0;
    uint64_t reserved[4]{};
};

struct alignas(16) PqFinitePersistentCompletionNotification {
    uint64_t publish_sequence = 0;
    uint32_t slot_idx = UINT32_MAX;
    uint32_t done_seq = 0;
};

struct alignas(64) PqFiniteReadyRingControl {
    uint32_t stop = 0;
    uint32_t error_claimed = 0;
    uint32_t reserved[14]{};
};

struct alignas(64) PqFiniteReadyRingEntry {
    uint64_t publish_sequence = 0;
    PqFiniteReadyToken token{};
    uint64_t reserved[3]{};
};

struct alignas(64) PqFiniteReadyRingStatus {
    uint32_t error_result = 0;
    uint32_t error_query_id = UINT32_MAX;
    uint32_t error_generation = 0;
    uint32_t reserved0 = 0;
    uint64_t error_ticket = 0;
    uint64_t reserved[5]{};
};

struct alignas(64) PqFinitePerHopTiming {
    uint64_t ring_wait_cycles = 0;
    uint64_t ring_metadata_cycles = 0;
    uint64_t restore_cycles = 0;
    uint64_t payload_cycles = 0;
    uint64_t writeback_cycles = 0;
    uint64_t resume_events = 0;
    uint64_t park_events = 0;
    uint64_t terminal_events = 0;
};

static_assert(sizeof(PqFinitePersistentRequestControl) == 64);
static_assert(alignof(PqFinitePersistentRequestControl) == 64);
static_assert(sizeof(PqFinitePersistentCompletionControl) == 64);
static_assert(alignof(PqFinitePersistentCompletionControl) == 64);
static_assert(sizeof(PqFinitePersistentCompletionNotification) == 16);
static_assert(alignof(PqFinitePersistentCompletionNotification) == 16);
static_assert(sizeof(PqFiniteReadyRingControl) == 64);
static_assert(sizeof(PqFiniteReadyRingEntry) == 64);
static_assert(sizeof(PqFiniteReadyRingStatus) == 64);
static_assert(sizeof(PqFinitePerHopTiming) == 64);
static_assert(alignof(PqFiniteReadyRingControl) == 64);
static_assert(alignof(PqFiniteReadyRingEntry) == 64);
static_assert(alignof(PqFiniteReadyRingStatus) == 64);
static_assert(alignof(PqFinitePerHopTiming) == 64);

struct LatencySummary {
    double avg = 0.0;
    double min = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

LatencySummary summarize_latency_values(std::vector<double> values)
{
    LatencySummary summary;
    if (values.empty()) {
        return summary;
    }
    std::sort(values.begin(), values.end());
    const auto percentile = [&values](double p) {
        const double pos = p * static_cast<double>(values.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = static_cast<size_t>(std::ceil(pos));
        if (lo == hi) {
            return values[lo];
        }
        const double frac = pos - static_cast<double>(lo);
        return values[lo] * (1.0 - frac) + values[hi] * frac;
    };
    summary.min = values.front();
    summary.max = values.back();
    summary.p50 = percentile(0.50);
    summary.p90 = percentile(0.90);
    summary.p95 = percentile(0.95);
    summary.p99 = percentile(0.99);
    summary.avg = std::accumulate(values.begin(), values.end(), 0.0) /
                  static_cast<double>(values.size());
    return summary;
}

__device__ __forceinline__ uint32_t load_bypass_l1_u32(const volatile uint32_t* ptr)
{
    uint32_t value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
    return value;
}

__device__ __forceinline__ uint64_t load_bypass_l1_u64(const volatile uint64_t* ptr)
{
    uint64_t value;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(ptr));
    return value;
}

bool pq_hang_debug_enabled();
void pq_debug_log(const char* fmt, ...);
bool pq_mailbox_trace_enabled();

uint64_t read_ordered_tsc()
{
    _mm_lfence();
    const uint64_t cycles = __rdtsc();
    _mm_lfence();
    return cycles;
}

bool pq_hang_debug_enabled()
{
    static const bool enabled = std::getenv("GO_FANN_DEBUG_HANG") != nullptr;
    return enabled;
}

void pq_debug_log(const char* fmt, ...)
{
    if (!pq_hang_debug_enabled()) {
        return;
    }
    std::va_list args;
    va_start(args, fmt);
    std::vfprintf(stdout, fmt, args);
    std::fprintf(stdout, "\n");
    std::fflush(stdout);
    va_end(args);
}

bool pq_mailbox_trace_enabled()
{
    static const bool enabled = std::getenv("GOFANN_DEBUG_RUNTIME") != nullptr;
    return enabled;
}

void trace_query_subset_mapping(const GraphRuntime& runtime,
                                const std::vector<uint32_t>& query_indices,
                                const char* stage)
{
    if (!pq_mailbox_trace_enabled()) {
        return;
    }
    static uint32_t dump_count = 0;
    if (dump_count >= 4) {
        return;
    }
    ++dump_count;

    std::fprintf(stdout,
                 "[runtime-trace][subset] stage=%s query_count=%zu routed_count=%zu\n",
                 stage,
                 query_indices.size(),
                 runtime.query_group_ids.size());
    const size_t sample_count = std::min<size_t>(query_indices.size(), 8);
    for (size_t i = 0; i < sample_count; ++i) {
        const uint32_t query_idx = query_indices[i];
        const uint32_t group_id = query_idx < runtime.query_group_ids.size()
            ? runtime.query_group_ids[query_idx]
            : UINT32_MAX;
        const uint32_t start_local = query_idx < runtime.start_local_ids.size()
            ? runtime.start_local_ids[query_idx]
            : UINT32_MAX;
        std::fprintf(stdout,
                     "[runtime-trace][subset] stage=%s subset_idx=%zu query_idx=%u group=%u start_local=%u\n",
                     stage,
                     i,
                     query_idx,
                     group_id,
                     start_local);
    }
    std::fflush(stdout);
}

__device__ __forceinline__ void maybe_trace_kernel_mailbox_publish(int debug_mailbox,
                                                                    uint32_t query_idx,
                                                                    uint32_t active_query_idx,
                                                                    uint32_t group_id,
                                                                   uint32_t local_row,
                                                                   uint32_t buffer_idx,
                                                                   uint32_t generation,
                                                                   int step_num,
                                                                   int phase)
{
    if (debug_mailbox == 0) {
        return;
    }
    if (query_idx >= 4 || generation >= 4) {
        return;
    }
    if (phase != 0 && phase != 2) {
        return;
    }
    printf("[runtime-trace][kernel-mailbox] phase=%d active_query=%u query=%u group=%u row=%u buffer=%u gen=%u step=%d\n",
           phase,
           active_query_idx,
           query_idx,
           group_id,
           local_row,
           buffer_idx,
           generation,
           step_num);
}

using PoolPair = warp_kernel::KernelPair<dist_t, uint32_t>;
using ResultPair = warp_kernel::KernelPair<dist_t, uint32_t>;
using StopPair = warp_kernel::KernelPair<dist_t, idx_t>;

static_assert(sizeof(PoolPair) == 8, "PQ traversal entries must remain compact");
static_assert(sizeof(ResultPair) == 8, "PQ result heap entries must remain compact");
static_assert(sizeof(StopPair) == 16, "PQ stop heap entries must preserve packed payloads");
static_assert(alignof(ResultPair) >= alignof(dist_t), "PQ result heap entries must be aligned");

constexpr uint32_t specialized_queue_capacity(int queue_size)
{
    return queue_size <= 0 ? 0U
        : queue_size <= 64 ? 64U
        : queue_size <= 128 ? 128U
        : queue_size <= 256 ? 256U
        : queue_size <= 512 ? 512U
        : 0U;
}

constexpr size_t finite_mutable_bytes_per_query(uint32_t queue_capacity,
                                                uint32_t result_capacity)
{
    return sizeof(PqFiniteContextControl) + sizeof(PqFiniteContextTelemetry) +
           (static_cast<size_t>(queue_capacity) + 2) * sizeof(PoolPair) +
           (static_cast<size_t>(kTopk) + 1) * sizeof(StopPair) +
           (static_cast<size_t>(result_capacity) + 1) * sizeof(ResultPair);
}

constexpr size_t finite_hot_state_bytes_per_query(uint32_t queue_capacity,
                                                  uint32_t result_capacity)
{
    return finite_mutable_bytes_per_query(queue_capacity, result_capacity) -
           sizeof(PqFiniteContextTelemetry);
}

struct PqFiniteStateTrafficPerHop {
    uint32_t read_operations = 0;
    uint32_t write_operations = 0;
    size_t read_bytes = 0;
    size_t write_bytes = 0;
};

constexpr size_t finite_resume_quantum_telemetry_bytes = sizeof(uint32_t) * 4;

constexpr PqFiniteStateTrafficPerHop finite_resume_state_traffic_per_hop(
    uint32_t queue_capacity,
    uint32_t result_capacity)
{
    constexpr size_t cold_read_modify_write_bytes =
        sizeof(uint64_t) * 2 + sizeof(float) + sizeof(uint32_t);
    return {
        8,
        6,
        finite_hot_state_bytes_per_query(queue_capacity, result_capacity) +
            cold_read_modify_write_bytes,
        sizeof(PqFiniteContextState) * 2 + cold_read_modify_write_bytes,
    };
}

constexpr PqFiniteStateTrafficPerHop finite_park_state_traffic_per_hop(
    uint32_t queue_capacity,
    uint32_t result_capacity)
{
    constexpr size_t cold_read_modify_write_bytes =
        sizeof(uint64_t) + sizeof(float) * 2 + sizeof(uint32_t);
    return {
        8,
        12,
        cold_read_modify_write_bytes + finite_resume_quantum_telemetry_bytes,
        finite_hot_state_bytes_per_query(queue_capacity, result_capacity) +
            cold_read_modify_write_bytes + finite_resume_quantum_telemetry_bytes,
    };
}

constexpr PqFiniteStateTrafficPerHop finite_terminal_state_traffic_per_hop(
    uint32_t queue_capacity,
    uint32_t result_capacity)
{
    constexpr size_t cold_read_modify_write_bytes =
        sizeof(uint64_t) + sizeof(float) * 2 + sizeof(uint32_t);
    constexpr size_t timing_row_source_bytes =
        sizeof(uint64_t) * 3 + sizeof(float) * 2;
    return {
        13,
        12,
        cold_read_modify_write_bytes + timing_row_source_bytes +
            finite_resume_quantum_telemetry_bytes,
        finite_hot_state_bytes_per_query(queue_capacity, result_capacity) +
            cold_read_modify_write_bytes + finite_resume_quantum_telemetry_bytes,
    };
}

constexpr size_t pq_result_heap_shared_bytes(uint32_t result_width)
{
    return static_cast<size_t>(kMultiQuery) *
           (static_cast<size_t>(result_width) + 1) * sizeof(ResultPair);
}

constexpr size_t native_pq_workspace_shared_bytes(uint32_t queue_limit,
                                                  uint32_t result_width, bool predicate_bias = false)
{
    return static_cast<size_t>(kMultiQuery) *
           ((static_cast<size_t>(queue_limit) + 2) *
                (predicate_bias ? sizeof(gpu_predicate::BiasedPair) : sizeof(PoolPair)) +
            (static_cast<size_t>(result_width) + 1) * sizeof(ResultPair));
}

struct PqNativeProfileRecord {
    uint64_t pickup_elapsed_ns = 0;
    uint64_t hash_test_elapsed_ns = 0;
    uint64_t frontier_update_elapsed_ns = 0;
    uint64_t pickup_invocations = 0;
    uint64_t hash_test_calls = 0;
    uint64_t hash_test_hits = 0;
    uint64_t hash_test_misses = 0;
    uint64_t hash_test_probes = 0;
    uint64_t hash_test_max_probe = 0;
    uint64_t hash_add_calls = 0;
    uint64_t hash_add_insertions = 0;
    uint64_t hash_add_existing = 0;
    uint64_t hash_add_full = 0;
    uint64_t hash_add_probes = 0;
    uint64_t hash_add_max_probe = 0;
    uint64_t neighbor_references = 0;
    uint64_t post_hash_distance_evaluations = 0;
    uint64_t threshold_rejections = 0;
    uint64_t candidate_admissions = 0;
    uint64_t frontier_evictions = 0;
    uint64_t frontier_update_invocations = 0;
    uint64_t traversal_steps = 0;
    uint64_t resident_hash_hits = 0;
    uint64_t nonresident_hash_hits = 0;
};

static_assert(sizeof(PqNativeProfileRecord) == 24 * sizeof(uint64_t),
              "native PQ profile record must remain fixed-width");

struct PqNativeHashTestProbe {
    bool found;
    uint32_t probes;
};

enum class PqNativeHashAddOutcome : uint32_t {
    Inserted,
    Existing,
    Full,
};

struct PqNativeHashAddProbe {
    PqNativeHashAddOutcome outcome;
    uint32_t probes;
};

__device__ __forceinline__ PqNativeHashTestProbe probe_native_hash_test(
    const uint32_t* hash_data,
    uint32_t value)
{
    uint32_t code = native_visited_hash::code<kHashTableCapacity>(value);
    for (uint32_t probes = 1; probes <= kHashTableCapacity; ++probes) {
        if (hash_data[code] == UINT32_MAX) {
            return {false, probes};
        }
        if (hash_data[code] == value) {
            return {true, probes};
        }
        code = (code + 1) % kHashTableCapacity;
    }
    return {false, kHashTableCapacity};
}

__device__ __forceinline__ PqNativeHashAddProbe probe_native_hash_add(
    uint32_t* hash_data,
    uint32_t value)
{
    uint32_t code = native_visited_hash::code<kHashTableCapacity>(value);
    for (uint32_t probes = 1; probes <= kHashTableCapacity; ++probes) {
        if (hash_data[code] == value) {
            return {PqNativeHashAddOutcome::Existing, probes};
        }
        bool available = hash_data[code] == UINT32_MAX;
#ifdef __ENABLE_VISITED_DEL
        available = available || hash_data[code] == static_cast<uint32_t>(-2);
#endif
        if (available) {
            hash_data[code] = value;
            return {PqNativeHashAddOutcome::Inserted, probes};
        }
        code = (code + 1) % kHashTableCapacity;
    }
    return {PqNativeHashAddOutcome::Full, kHashTableCapacity};
}

__host__ __device__ __forceinline__ uint64_t pack_topk_payload(uint32_t local_row, dist_t pq_dist)
{
    union {
        dist_t f;
        uint32_t u;
    } bits{};
    bits.f = pq_dist;
    return (static_cast<uint64_t>(bits.u) << 32) | static_cast<uint64_t>(local_row);
}

__host__ __device__ __forceinline__ uint32_t unpack_topk_local_row(uint64_t payload)
{
    return static_cast<uint32_t>(payload & 0xffffffffULL);
}

__host__ __device__ __forceinline__ dist_t unpack_topk_pq_dist(uint64_t payload)
{
    union {
        dist_t f;
        uint32_t u;
    } bits{};
    bits.u = static_cast<uint32_t>(payload >> 32);
    return bits.f;
}

constexpr uint32_t kPoolUsedMask = 0x80000000u;

__host__ __device__ __forceinline__ uint32_t pack_pool_payload(uint32_t local_row, bool used)
{
    return local_row | (used ? kPoolUsedMask : 0u);
}

__host__ __device__ __forceinline__ uint32_t unpack_pool_local_row(uint32_t payload)
{
    return payload & ~kPoolUsedMask;
}

__host__ __device__ __forceinline__ bool pool_payload_used(uint32_t payload)
{
    return (payload & kPoolUsedMask) != 0u;
}

__host__ __device__ __forceinline__ uint32_t mark_pool_payload_used(uint32_t payload)
{
    return payload | kPoolUsedMask;
}

template <int kProfileMode>
__device__ __forceinline__ bool profiled_native_hash_test(
    const uint32_t* hash_data,
    uint32_t value,
    PqNativeProfileRecord* profile)
{
    if constexpr (kProfileMode == 0) {
        return native_visited_hash::test<kHashTableCapacity>(hash_data, value);
    }

    const PqNativeHashTestProbe result = probe_native_hash_test(hash_data, value);
    ++profile->hash_test_calls;
    profile->hash_test_probes += result.probes;
    if (result.probes > profile->hash_test_max_probe) {
        profile->hash_test_max_probe = result.probes;
    }
    if (result.found) {
        ++profile->hash_test_hits;
    } else {
        ++profile->hash_test_misses;
    }
    return result.found;
}

template <int kProfileMode>
__device__ __forceinline__ void profiled_native_hash_add(
    uint32_t* hash_data,
    uint32_t value,
    PqNativeProfileRecord* profile)
{
    if constexpr (kProfileMode == 0) {
        native_visited_hash::add<kHashTableCapacity>(hash_data, value);
        return;
    }

    const PqNativeHashAddProbe result = probe_native_hash_add(hash_data, value);
    ++profile->hash_add_calls;
    profile->hash_add_probes += result.probes;
    if (result.probes > profile->hash_add_max_probe) {
        profile->hash_add_max_probe = result.probes;
    }
    if (result.outcome == PqNativeHashAddOutcome::Inserted) {
        ++profile->hash_add_insertions;
    } else if (result.outcome == PqNativeHashAddOutcome::Existing) {
        ++profile->hash_add_existing;
    } else {
        ++profile->hash_add_full;
    }
}

template<class Pair>
__device__ __forceinline__ bool native_frontier_contains_local_row(
    const Pair* frontier,
    int size,
    uint32_t local_row)
{
    for (int i = 1; i < size; ++i) {
        if (unpack_pool_local_row(frontier[i].second) == local_row) {
            return true;
        }
    }
    return false;
}

template <class T>
__device__ bool pickup_next_unused_from_pool(T* pool, int size, T& out)
{
    int best = -1;
    for (int i = 1; i < size; ++i) {
        if (pool_payload_used(pool[i].second)) {
            continue;
        }
        if (best < 0 || pool[i].first < pool[best].first) {
            best = i;
        }
    }
    if (best < 0) {
        return false;
    }
    out = pool[best];
    pool[best].second = mark_pool_payload_used(pool[best].second);
    return true;
}

template <bool kCooperative, class T>
__device__ bool pickup_next_unused_from_native_frontier(T* frontier,
                                                        int size,
                                                        T& out,
                                                        int lane)
{
    if constexpr (!kCooperative) {
        return lane == 0 && pickup_next_unused_from_pool(frontier, size, out);
    }

    int best_index = -1;
    dist_t best_distance = FLT_MAX;
    for (int i = lane + 1; i < size; i += warp_kernel::kWarpThreads) {
        if (pool_payload_used(frontier[i].second)) {
            continue;
        }
        const dist_t distance = frontier[i].first;
        if (best_index < 0 || distance < best_distance ||
            (!(best_distance < distance) && !(distance < best_distance) &&
             i < best_index)) {
            best_index = i;
            best_distance = distance;
        }
    }
    for (int offset = warp_kernel::kWarpThreads / 2; offset > 0; offset /= 2) {
        const int other_index = __shfl_down_sync(warp_kernel::kFullMask,
                                                  best_index,
                                                  offset);
        const dist_t other_distance = __shfl_down_sync(warp_kernel::kFullMask,
                                                        best_distance,
                                                        offset);
        if (other_index >= 0 &&
            (best_index < 0 || other_distance < best_distance ||
             (!(best_distance < other_distance) && !(other_distance < best_distance) &&
              other_index < best_index))) {
            best_index = other_index;
            best_distance = other_distance;
        }
    }

    best_index = __shfl_sync(warp_kernel::kFullMask, best_index, 0);
    const bool found = best_index >= 0;
    if (lane == 0 && found) {
        out = frontier[best_index];
        frontier[best_index].second = mark_pool_payload_used(frontier[best_index].second);
    }
    __syncwarp();
    return found;
}

template <bool kCooperative, int kProfileMode, class T>
__device__ bool profiled_pickup_next_unused_from_native_frontier(
    T* frontier,
    int size,
    T& out,
    int lane,
    PqNativeProfileRecord* profile)
{
    uint64_t start_ns = 0;
    if constexpr (kProfileMode > 0) {
        if (lane == 0) {
            start_ns = warp_kernel::get_global_timer_ns();
        }
    }

    bool found = false;
    if constexpr (kCooperative) {
        found = pickup_next_unused_from_native_frontier<true>(frontier, size, out, lane);
    } else if (lane == 0) {
        found = pickup_next_unused_from_pool(frontier, size, out);
    }

    if constexpr (kProfileMode > 0) {
        if (lane == 0) {
            profile->pickup_elapsed_ns += warp_kernel::get_global_timer_ns() - start_ns;
            ++profile->pickup_invocations;
        }
    }
    return found;
}

template <class T>
__device__ __forceinline__ void refresh_native_frontier_worst(
    const T* frontier,
    int size,
    int lane,
    int* shared_worst_index,
    dist_t* shared_worst_distance)
{
    int worst_index = -1;
    dist_t worst_distance = -FLT_MAX;
    for (int i = lane + 1; i < size; i += warp_kernel::kWarpThreads) {
        const dist_t distance = frontier[i].first;
        if (worst_index < 0 || worst_distance < distance ||
            (!(distance < worst_distance) && !(worst_distance < distance) &&
             i < worst_index)) {
            worst_index = i;
            worst_distance = distance;
        }
    }
    for (int offset = warp_kernel::kWarpThreads / 2; offset > 0; offset /= 2) {
        const int other_index = __shfl_down_sync(warp_kernel::kFullMask,
                                                  worst_index,
                                                  offset);
        const dist_t other_distance = __shfl_down_sync(warp_kernel::kFullMask,
                                                        worst_distance,
                                                        offset);
        if (other_index >= 0 &&
            (worst_index < 0 || worst_distance < other_distance ||
             (!(other_distance < worst_distance) && !(worst_distance < other_distance) &&
              other_index < worst_index))) {
            worst_index = other_index;
            worst_distance = other_distance;
        }
    }
    if (lane == 0) {
        *shared_worst_index = worst_index;
        *shared_worst_distance = worst_distance;
    }
    __syncwarp();
}

template<class Pair>
__device__ __forceinline__ void append_native_max_heap(Pair* frontier,
                                                        int& size,
                                                        const Pair& entry)
{
    int index = size++;
    while (index > 1) {
        const int parent = index / 2;
        if (!(frontier[parent].first < entry.first)) {
            break;
        }
        frontier[index] = frontier[parent];
        index = parent;
    }
    frontier[index] = entry;
}

template<class Pair>
__device__ __forceinline__ void replace_native_max_heap_root(Pair* frontier,
                                                              int size,
                                                              const Pair& entry)
{
    int index = 1;
    const int entry_count = size - 1;
    while (index * 2 <= entry_count) {
        int child = index * 2;
        if (child + 1 <= entry_count &&
            frontier[child].first < frontier[child + 1].first) {
            ++child;
        }
        if (!(entry.first < frontier[child].first)) {
            break;
        }
        frontier[index] = frontier[child];
        index = child;
    }
    frontier[index] = entry;
}

__host__ __device__ __forceinline__ uint32_t dist_to_bits(dist_t value)
{
    union {
        dist_t f;
        uint32_t u;
    } bits{};
    bits.f = value;
    return bits.u;
}

[[noreturn]] void fail_pq(const std::string& message);

struct PqBatchStats {
    size_t query_count = 0;
    size_t display_query_count = 0;
    double kernel_ms = 0.0;
    double qps_sum = 0.0;
    size_t run_count = 0;
    double total_time_us_sum = 0.0;
    double io_time_per_step_us_sum = 0.0;
    double cal_time_per_step_us_sum = 0.0;
    double queue_time_per_step_us_sum = 0.0;
    double wait_io_time_us_sum = 0.0;
    double ready_time_us_sum = 0.0;
    double running_time_us_sum = 0.0;
    double request_publish_time_us_sum = 0.0;
    std::vector<double> wait_io_fraction_by_query;
    uint64_t total_step_sum = 0;
    bool detailed_timing_available = true;
    bool finite_mode = false;
    bool finite_state_traffic_available = false;
    uint32_t finite_queue_capacity = 0;
    uint32_t finite_result_capacity = 0;
    uint64_t finite_resume_hops = 0;
    uint64_t finite_park_hops = 0;
    uint64_t finite_terminal_hops = 0;
    uint64_t finite_resumed_step_sum = 0;
    uint64_t finite_static_prefetch_step_sum = 0;
    uint64_t finite_multi_step_resume_count = 0;
    uint32_t finite_max_resume_quantum_steps = 0;
    HostFilterStats host_filter_stats{};
};

void record_pq_query_timing(PqBatchStats& stats, const float* timing_row)
{
    const double resident_time_us = static_cast<double>(timing_row[0]);
    const double step = std::max(1.0, static_cast<double>(timing_row[4]));
    stats.total_time_us_sum += resident_time_us;
    stats.total_step_sum += static_cast<uint64_t>(step);
    if (!stats.detailed_timing_available) {
        return;
    }

    const double parked_time_us = std::max(0.0, static_cast<double>(timing_row[1]));
    const double ready_time_us = stats.finite_mode
        ? std::clamp(static_cast<double>(timing_row[5]), 0.0, parked_time_us)
        : 0.0;
    const double wait_io_time_us = parked_time_us - ready_time_us;
    stats.io_time_per_step_us_sum += parked_time_us / step;
    stats.cal_time_per_step_us_sum += static_cast<double>(timing_row[2]) / step;
    stats.queue_time_per_step_us_sum += static_cast<double>(timing_row[3]) / step;
    stats.wait_io_time_us_sum += wait_io_time_us;
    stats.ready_time_us_sum += ready_time_us;
    stats.running_time_us_sum += std::max(0.0, resident_time_us - parked_time_us);
    if constexpr (kRequestPublishTiming) {
        if (!stats.finite_mode) {
            stats.request_publish_time_us_sum +=
                std::max(0.0, static_cast<double>(timing_row[5]));
        }
    }
    stats.wait_io_fraction_by_query.push_back(
        resident_time_us > 0.0 ? wait_io_time_us / resident_time_us : 0.0);
}

void accumulate_pq_batch_stats(PqBatchStats& total, const PqBatchStats& batch)
{
    total.query_count += batch.query_count;
    total.display_query_count += batch.display_query_count;
    total.kernel_ms += batch.kernel_ms;
    total.qps_sum += batch.qps_sum;
    total.run_count += batch.run_count;
    total.total_time_us_sum += batch.total_time_us_sum;
    total.io_time_per_step_us_sum += batch.io_time_per_step_us_sum;
    total.cal_time_per_step_us_sum += batch.cal_time_per_step_us_sum;
    total.queue_time_per_step_us_sum += batch.queue_time_per_step_us_sum;
    total.wait_io_time_us_sum += batch.wait_io_time_us_sum;
    total.ready_time_us_sum += batch.ready_time_us_sum;
    total.running_time_us_sum += batch.running_time_us_sum;
    total.request_publish_time_us_sum += batch.request_publish_time_us_sum;
    total.wait_io_fraction_by_query.insert(total.wait_io_fraction_by_query.end(),
                                           batch.wait_io_fraction_by_query.begin(),
                                           batch.wait_io_fraction_by_query.end());
    total.total_step_sum += batch.total_step_sum;
    total.detailed_timing_available =
        total.detailed_timing_available && batch.detailed_timing_available;
    total.finite_mode = total.finite_mode || batch.finite_mode;
    if (batch.finite_state_traffic_available) {
        if (!total.finite_state_traffic_available) {
            total.finite_queue_capacity = batch.finite_queue_capacity;
            total.finite_result_capacity = batch.finite_result_capacity;
        } else if (total.finite_queue_capacity != batch.finite_queue_capacity ||
                   total.finite_result_capacity != batch.finite_result_capacity) {
            fail_pq("cannot aggregate finite state traffic across different state layouts");
        }
        total.finite_state_traffic_available = true;
        total.finite_resume_hops += batch.finite_resume_hops;
        total.finite_park_hops += batch.finite_park_hops;
        total.finite_terminal_hops += batch.finite_terminal_hops;
        total.finite_resumed_step_sum += batch.finite_resumed_step_sum;
        total.finite_static_prefetch_step_sum += batch.finite_static_prefetch_step_sum;
        total.finite_multi_step_resume_count += batch.finite_multi_step_resume_count;
        total.finite_max_resume_quantum_steps = std::max(
            total.finite_max_resume_quantum_steps,
            batch.finite_max_resume_quantum_steps);
    }
    total.host_filter_stats.requests_served += batch.host_filter_stats.requests_served;
    total.host_filter_stats.requests_passed += batch.host_filter_stats.requests_passed;
    total.host_filter_stats.requests_filtered += batch.host_filter_stats.requests_filtered;
    total.host_filter_stats.request_ring_demand_events +=
        batch.host_filter_stats.request_ring_demand_events;
    total.host_filter_stats.request_ring_done_events +=
        batch.host_filter_stats.request_ring_done_events;
    total.host_filter_stats.workers_thread = std::max(total.host_filter_stats.workers_thread,
                                                      batch.host_filter_stats.workers_thread);

}

void print_pq_state_summary(const char* label, const PqBatchStats& stats)
{
    const char* state_definitions = stats.finite_mode
        ? "lifetime=time1;READY=time6(host-ready);WAIT_IO=time2-READY;"
          "RUNNING=max(time1-time2,0)"
        : "resident=time1;READY=0(coupled);WAIT_IO=time2(mailbox);"
          "RUNNING=max(time1-time2,0)";
    if (!stats.detailed_timing_available) {
        std::printf("[pq state] label=\"%s\" timing=unavailable query_evaluations=%zu "
                    "resident_us_sum=%.3f READY_us_sum=0.000 RUNNING_us_sum=NA "
                    "WAIT_IO_us_sum=NA READY_fraction=0.000000 RUNNING_fraction=NA "
                    "WAIT_IO_fraction=NA per_query_WAIT_IO_fraction_p50=NA "
                    "per_query_WAIT_IO_fraction_p95=NA per_query_WAIT_IO_fraction_p99=NA "
                    "definitions=\"%s\"\n",
                    label,
                    stats.query_count,
                    stats.total_time_us_sum,
                    state_definitions);
        return;
    }

    const LatencySummary wait_fraction =
        summarize_latency_values(stats.wait_io_fraction_by_query);
    const double resident_time_us = stats.total_time_us_sum;
    const double wait_io_fraction = resident_time_us > 0.0
        ? stats.wait_io_time_us_sum / resident_time_us
        : 0.0;
    const double ready_fraction = resident_time_us > 0.0
        ? stats.ready_time_us_sum / resident_time_us
        : 0.0;
    const double running_fraction = resident_time_us > 0.0
        ? stats.running_time_us_sum / resident_time_us
        : 0.0;
    std::printf("[pq state] label=\"%s\" timing=available query_evaluations=%zu "
                "resident_us_sum=%.3f READY_us_sum=%.3f RUNNING_us_sum=%.3f "
                "WAIT_IO_us_sum=%.3f READY_fraction=%.6f RUNNING_fraction=%.6f "
                "WAIT_IO_fraction=%.6f per_query_WAIT_IO_fraction_p50=%.6f "
                "per_query_WAIT_IO_fraction_p95=%.6f per_query_WAIT_IO_fraction_p99=%.6f "
                "definitions=\"%s\"\n",
                label,
                stats.query_count,
                resident_time_us,
                stats.ready_time_us_sum,
                stats.running_time_us_sum,
                stats.wait_io_time_us_sum,
                ready_fraction,
                running_fraction,
                wait_io_fraction,
                wait_fraction.p50,
                wait_fraction.p95,
                wait_fraction.p99,
                state_definitions);
}

void print_pq_request_publish_summary(const char* label, const PqBatchStats& stats)
{
    if constexpr (kRequestPublishTiming) {
        const uint64_t demand_events =
            stats.host_filter_stats.request_ring_demand_events;
        const double average_publish_us = demand_events > 0
            ? stats.request_publish_time_us_sum / static_cast<double>(demand_events)
            : 0.0;
        std::printf("[pq request publish profile] label=\"%s\" demands=%llu "
                    "total_publish_us=%.3f avg_publish_us=%.6f\n",
                    label,
                    static_cast<unsigned long long>(demand_events),
                    stats.request_publish_time_us_sum,
                    average_publish_us);
    }
}

void print_pq_d2_availability_summary(const char* label,
                                      const PqBatchStats& stats,
                                      double qps)
{
    if (!stats.finite_state_traffic_available) {
        return;
    }
    const double average_steps_per_resume = stats.finite_resume_hops == 0
        ? 0.0
        : static_cast<double>(stats.finite_resumed_step_sum) /
              static_cast<double>(stats.finite_resume_hops);
    const double multi_step_resume_fraction = stats.finite_resume_hops == 0
        ? 0.0
        : static_cast<double>(stats.finite_multi_step_resume_count) /
              static_cast<double>(stats.finite_resume_hops);
    std::printf(
        "[pq d2 availability] label=\"%s\" source="
        "host_completed_first_row_plus_static_resident_prefetch "
        "qps=%.6f resumes=%llu resumed_steps=%llu average_steps_per_resume=%.6f "
        "static_prefetch_steps=%llu multi_step_resumes=%llu "
        "multi_step_resume_fraction=%.6f max_resume_quantum=%u "
        "same_query_multi_token_producer=absent\n",
        label,
        qps,
        static_cast<unsigned long long>(stats.finite_resume_hops),
        static_cast<unsigned long long>(stats.finite_resumed_step_sum),
        average_steps_per_resume,
        static_cast<unsigned long long>(stats.finite_static_prefetch_step_sum),
        static_cast<unsigned long long>(stats.finite_multi_step_resume_count),
        multi_step_resume_fraction,
        stats.finite_max_resume_quantum_steps);
}

void print_pq_batch_stats_summary(const char* label, const PqBatchStats& stats)
{
    if (stats.query_count == 0) {
        return;
    }

    std::printf("%s query evaluations: %zu\n", label, stats.query_count);
    if (stats.display_query_count > 0 && stats.display_query_count != stats.query_count) {
        std::printf("%s displayed queries: %zu\n", label, stats.display_query_count);
    }
    if (stats.run_count > 0) {
        std::printf("%s QPS: %f\n", label, stats.qps_sum / static_cast<double>(stats.run_count));
    }
    if (stats.total_time_us_sum > 0.0) {
        std::printf("%s Total time per query: %f ms\n", label, stats.total_time_us_sum / stats.query_count / 1000.0);
    }
    if (stats.detailed_timing_available) {
        std::printf("%s IO time per query per step: %f ms\n", label, stats.io_time_per_step_us_sum / stats.query_count / 1000.0);
        std::printf("%s Computation time per query per step: %f us\n", label, stats.cal_time_per_step_us_sum / stats.query_count);
        std::printf("%s Queue management time per query per step: %f us\n", label, stats.queue_time_per_step_us_sum / stats.query_count);
    } else {
        std::printf("%s Detailed per-step timing: disabled at compile time\n", label);
    }
    print_pq_state_summary(label, stats);
    print_pq_request_publish_summary(label, stats);
    if (stats.finite_state_traffic_available) {
        const PqFiniteStateTrafficPerHop resume = finite_resume_state_traffic_per_hop(
            stats.finite_queue_capacity, stats.finite_result_capacity);
        const PqFiniteStateTrafficPerHop park = finite_park_state_traffic_per_hop(
            stats.finite_queue_capacity, stats.finite_result_capacity);
        const PqFiniteStateTrafficPerHop terminal = finite_terminal_state_traffic_per_hop(
            stats.finite_queue_capacity, stats.finite_result_capacity);
        std::printf(
            "[pq finite state traffic] label=\"%s\" accounting=logical "
            "operation_unit=record_field_or_array_region "
            "layout_change=access_split_not_capacity_compression "
            "hot_control_bytes=%zu cold_telemetry_bytes=%zu hot_state_bytes_per_query=%zu "
            "persistent_bytes_per_query=%zu "
            "resume_hops=%llu resume_read_ops_per_hop=%u resume_write_ops_per_hop=%u "
            "resume_read_bytes_per_hop=%zu resume_write_bytes_per_hop=%zu "
            "resume_total_read_ops=%llu resume_total_write_ops=%llu "
            "resume_total_read_bytes=%llu resume_total_write_bytes=%llu "
            "park_hops=%llu park_read_ops_per_hop=%u park_write_ops_per_hop=%u "
            "park_read_bytes_per_hop=%zu park_write_bytes_per_hop=%zu "
            "park_total_read_ops=%llu park_total_write_ops=%llu "
            "park_total_read_bytes=%llu park_total_write_bytes=%llu "
            "terminal_hops=%llu terminal_read_ops_per_hop=%u terminal_write_ops_per_hop=%u "
            "terminal_read_bytes_per_hop=%zu terminal_write_bytes_per_hop=%zu "
            "terminal_total_read_ops=%llu terminal_total_write_ops=%llu "
            "terminal_total_read_bytes=%llu terminal_total_write_bytes=%llu\n",
            label,
            sizeof(PqFiniteContextControl),
            sizeof(PqFiniteContextTelemetry),
            finite_hot_state_bytes_per_query(stats.finite_queue_capacity,
                                             stats.finite_result_capacity),
            finite_mutable_bytes_per_query(stats.finite_queue_capacity,
                                           stats.finite_result_capacity),
            static_cast<unsigned long long>(stats.finite_resume_hops),
            resume.read_operations,
            resume.write_operations,
            resume.read_bytes,
            resume.write_bytes,
            static_cast<unsigned long long>(stats.finite_resume_hops * resume.read_operations),
            static_cast<unsigned long long>(stats.finite_resume_hops * resume.write_operations),
            static_cast<unsigned long long>(stats.finite_resume_hops * resume.read_bytes),
            static_cast<unsigned long long>(stats.finite_resume_hops * resume.write_bytes),
            static_cast<unsigned long long>(stats.finite_park_hops),
            park.read_operations,
            park.write_operations,
            park.read_bytes,
            park.write_bytes,
            static_cast<unsigned long long>(stats.finite_park_hops * park.read_operations),
            static_cast<unsigned long long>(stats.finite_park_hops * park.write_operations),
            static_cast<unsigned long long>(stats.finite_park_hops * park.read_bytes),
            static_cast<unsigned long long>(stats.finite_park_hops * park.write_bytes),
            static_cast<unsigned long long>(stats.finite_terminal_hops),
            terminal.read_operations,
            terminal.write_operations,
            terminal.read_bytes,
            terminal.write_bytes,
            static_cast<unsigned long long>(stats.finite_terminal_hops * terminal.read_operations),
            static_cast<unsigned long long>(stats.finite_terminal_hops * terminal.write_operations),
            static_cast<unsigned long long>(stats.finite_terminal_hops * terminal.read_bytes),
            static_cast<unsigned long long>(stats.finite_terminal_hops * terminal.write_bytes));
    }
    const double qps = stats.run_count > 0
        ? stats.qps_sum / static_cast<double>(stats.run_count)
        : 0.0;
    print_pq_d2_availability_summary(label, stats, qps);
    if (stats.total_step_sum > 0) {
        std::printf("%s Average Step Num per query: %f\n", label, static_cast<double>(stats.total_step_sum) / stats.query_count);
    }
    if (stats.host_filter_stats.requests_served > 0) {
        print_host_filter_summary(label, stats.host_filter_stats);
    }
}

[[noreturn]] void fail_pq(const std::string& message)
{
    throw std::runtime_error("PQ backend failed: " + message);
}

void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        fail_pq(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_gdr(int status, const char* what)
{
    if (status != 0) {
        fail_pq(std::string(what) + " failed with code " + std::to_string(status));
    }
}

bool use_adaptive_wait_policy(const std::string& policy)
{
    if (policy == "fixed") {
        return false;
    }
    if (policy == "adaptive") {
        return true;
    }
    fail_pq("unsupported host_reply_wait_policy: " + policy);
}

enum class PqBufferMode : int {
    SingleSlotSerial = 0,
    SingleSlotOverlap = 1,
    DoubleSlotSerial = 2,
    DoubleSlotOverlap = 3,
};

__host__ __device__ constexpr bool pq_buffer_mode_uses_single_slot(PqBufferMode mode)
{
    return static_cast<int>(mode) < static_cast<int>(PqBufferMode::DoubleSlotSerial);
}

__host__ __device__ constexpr bool pq_buffer_mode_overlaps(PqBufferMode mode)
{
    return (static_cast<int>(mode) & 1) != 0;
}

PqBufferMode parse_pq_buffer_mode(const std::string& mode)
{
    if (mode == "single_slot_serial") {
        return PqBufferMode::SingleSlotSerial;
    }
    if (mode == "single_slot_overlap") {
        return PqBufferMode::SingleSlotOverlap;
    }
    if (mode == "double_slot_serial") {
        return PqBufferMode::DoubleSlotSerial;
    }
    if (mode == "double_slot_overlap") {
        return PqBufferMode::DoubleSlotOverlap;
    }
    fail_pq("unsupported pq.buffer_mode: " + mode);
    return PqBufferMode::DoubleSlotOverlap;
}

enum class PqFiniteTokenTransport {
    DeviceCopy,
    Mapped,
};

enum class PqFiniteDispatchMode {
    Batch,
    Persistent,
    ReadyRing,
};

PqFiniteDispatchMode parse_pq_finite_dispatch_mode(const std::string& mode)
{
    if (mode == defaults::pq_finite_dispatch_mode_batch) {
        return PqFiniteDispatchMode::Batch;
    }
    if (mode == defaults::pq_finite_dispatch_mode_persistent) {
        return PqFiniteDispatchMode::Persistent;
    }
    if (mode == defaults::pq_finite_dispatch_mode_ready_ring) {
        return PqFiniteDispatchMode::ReadyRing;
    }
    fail_pq("unsupported pq.finite_dispatch_mode: " + mode);
    return PqFiniteDispatchMode::Batch;
}

const char* pq_finite_dispatch_mode_name(PqFiniteDispatchMode mode)
{
    switch (mode) {
        case PqFiniteDispatchMode::Batch: return "batch";
        case PqFiniteDispatchMode::Persistent: return "persistent";
        case PqFiniteDispatchMode::ReadyRing: return "ready_ring";
    }
    return "unknown";
}

PqFiniteTokenTransport parse_pq_finite_token_transport(const std::string& transport)
{
    if (transport == defaults::pq_finite_token_transport_device_copy) {
        return PqFiniteTokenTransport::DeviceCopy;
    }
    if (transport == defaults::pq_finite_token_transport_mapped) {
        return PqFiniteTokenTransport::Mapped;
    }
    fail_pq("unsupported pq.finite_token_transport: " + transport);
    return PqFiniteTokenTransport::DeviceCopy;
}

const char* pq_finite_token_transport_name(PqFiniteTokenTransport transport)
{
    switch (transport) {
        case PqFiniteTokenTransport::DeviceCopy: return "device_copy";
        case PqFiniteTokenTransport::Mapped: return "mapped";
    }
    return "unknown";
}

void validate_pq_query_worker_mode(const PqSearchOptions& options)
{
    if (options.graph_transport != "mailbox" &&
        options.graph_transport != "device_resident") {
        fail_pq("native PQ graph_transport must be one of: mailbox, device_resident");
    }
    if (!defaults::is_valid_pq_query_worker_mode(options.query_worker_mode)) {
        fail_pq("native PQ search query_worker_mode must be one of: coupled, finite");
    }
    if (!defaults::is_valid_pq_finite_token_transport(options.finite_token_transport)) {
        fail_pq("native PQ search finite_token_transport must be one of: "
                "device_copy, mapped");
    }
    if (!defaults::is_valid_pq_finite_dispatch_mode(options.finite_dispatch_mode)) {
        fail_pq("native PQ search finite_dispatch_mode must be one of: "
                "batch, persistent, ready_ring");
    }
    if (!defaults::is_valid_pq_finite_partial_batch_timeout_us(
            options.finite_partial_batch_timeout_us)) {
        fail_pq("native PQ search finite_partial_batch_timeout_us must be in [1, 1000000]");
    }
    if (defaults::pq_query_worker_mode_is_finite(options.query_worker_mode) &&
        options.buffer_mode != "double_slot_serial") {
        fail_pq("native PQ search query_worker_mode=finite currently requires "
                "pq.buffer_mode=double_slot_serial");
    }
    if (defaults::pq_finite_dispatch_mode_requires_completion_path(
            options.finite_dispatch_mode)) {
        const char* dispatch_mode = options.finite_dispatch_mode.c_str();
        if (!defaults::pq_query_worker_mode_is_finite(options.query_worker_mode)) {
            fail_pq("native PQ search finite_dispatch_mode=" + std::string(dispatch_mode) +
                    " requires query_worker_mode=finite");
        }
        const bool memory_graph = options.graph_storage_mode == "memory";
        if (!memory_graph || options.buffer_mode != "double_slot_serial") {
            fail_pq("finite completion dispatch requires memory graph with double_slot_serial");
        }
        if (kMultiQuery != 2) {
            fail_pq("native PQ search finite_dispatch_mode=" + std::string(dispatch_mode) +
                    " currently requires PQ_MULTI_QUERY=2");
        }
    }
    if (options.request_mode != "scan" && options.request_mode != "batch_ring") {
        fail_pq("native PQ search request_mode must be one of: scan, batch_ring");
    }
    if (options.request_batch_queries == 0 || options.request_batch_queries > 4096) {
        fail_pq("native PQ search request_batch_queries must be in [1, 4096]");
    }
    if (options.request_consumed_memory != "host" &&
        options.request_consumed_memory != "device") {
        fail_pq("native PQ search request_consumed_memory must be one of: host, device");
    }
    if (options.request_mode == "batch_ring") {
        if (defaults::pq_query_worker_mode_is_finite(options.query_worker_mode) ||
            options.graph_storage_mode != "memory") {
            fail_pq("native PQ batch request ring requires coupled memory workers");
        }
        if (options.host_demand_batch_size != 1) {
            fail_pq("native PQ batch request ring requires host_demand_batch_size=1");
        }
    }
    if (options.graph_transport == "device_resident" &&
        (options.graph_storage_mode != "memory" ||
         defaults::pq_query_worker_mode_is_finite(options.query_worker_mode) ||
         options.request_mode != "scan" ||
         !defaults::highfreq_rerank_mode_is_gpu(options.highfreq_rerank_mode))) {
        fail_pq("native PQ device_resident graph transport requires a memory graph, "
                "coupled workers, request_mode=scan, resident base vectors, and GPU rerank");
    }
}

void require_valid_finite_query_ids(const std::vector<uint32_t>& query_ids,
                                    size_t query_capacity)
{
    const PqFiniteQueryIdValidation validation =
        validate_finite_query_ids(query_ids, query_capacity);
    switch (validation.error) {
        case PqFiniteQueryIdError::NONE:
            return;
        case PqFiniteQueryIdError::TOO_MANY:
            fail_pq("finite PQ active context count exceeds context capacity");
        case PqFiniteQueryIdError::OUT_OF_RANGE:
            fail_pq("finite PQ query ID " + std::to_string(validation.query_id) +
                    " at active position " + std::to_string(validation.position) +
                    " exceeds context capacity");
        case PqFiniteQueryIdError::DUPLICATE:
            fail_pq("finite PQ query ID " + std::to_string(validation.query_id) +
                    " appears more than once");
    }
    fail_pq("finite PQ query ID validation returned an unknown result");
}

void require_pq_graph_dim_supported(int dim)
{
    if (dim > kGraphMaxDim) {
        fail_pq("graph-based PQ path supports dim <= " + std::to_string(kGraphMaxDim) +
                " with the current compiled shared-memory query buffer; got dim=" +
                std::to_string(dim));
    }
}

void require_pq_graph_pq_dim_supported(int pq_dim)
{
    if (pq_dim != kGraphPqDim16 && pq_dim != kGraphPqDim64) {
        fail_pq("graph-based PQ path supports pq_dim=16 or pq_dim=64; got pq_dim=" +
                std::to_string(pq_dim));
    }
}

bool pq_graph_use_sample_id_payload(const PqSearchOptions& options)
{
    return options.pq_dim == kGraphPqDim16;
}

bool pq_graph_can_use_implicit_filter_mask(const GraphRuntime& runtime)
{
    if (runtime.group_layout.groups.size() != runtime.metadata.label_count ||
        runtime.metadata.label_filter_bit.size() != runtime.metadata.original_label_count) {
        return false;
    }
    for (uint32_t group_id = 0; group_id < runtime.metadata.label_count; ++group_id) {
        const auto& labels = runtime.group_layout.groups[group_id];
        if (labels.size() != 1) {
            return false;
        }
        const uint32_t original_label = labels.front();
        if (original_label >= runtime.group_layout.label_to_group.size() ||
            runtime.group_layout.label_to_group[original_label] != group_id ||
            runtime.metadata.label_filter_bit[original_label] != 1ULL) {
            return false;
        }
    }
    return true;
}

std::vector<uint32_t> build_start_sample_ids(const GraphRuntime& runtime)
{
    const size_t query_count = runtime.start_local_ids.size();
    std::vector<uint32_t> start_sample_ids(query_count, UINT32_MAX);
    for (size_t query_idx = 0; query_idx < query_count; ++query_idx) {
        const uint32_t group_id = runtime.query_group_ids[query_idx];
        const uint32_t local_row = runtime.start_local_ids[query_idx];
        if (group_id == UINT32_MAX ||
            group_id >= runtime.metadata.label_count ||
            local_row == UINT32_MAX ||
            local_row >= runtime.metadata.label_size[group_id]) {
            continue;
        }
        const uint64_t global_row =
            static_cast<uint64_t>(runtime.metadata.label_offset[group_id]) +
            static_cast<uint64_t>(local_row);
        if (global_row >= runtime.metadata.row_to_sample.size()) {
            continue;
        }
        start_sample_ids[query_idx] = runtime.metadata.row_to_sample[global_row];
    }
    return start_sample_ids;
}

enum SearchEarlyStopMode : int {
    kSearchEarlyStopFixed = 0,
    kSearchEarlyStopOff = 1,
    kSearchEarlyStopAdaptive = 2,
};

int parse_search_early_stop_mode(const std::string& mode)
{
    if (mode == defaults::search_early_stop_mode_fixed) {
        return kSearchEarlyStopFixed;
    }
    if (mode == defaults::search_early_stop_mode_off) {
        return kSearchEarlyStopOff;
    }
    if (mode == defaults::search_early_stop_mode_adaptive) {
        return kSearchEarlyStopAdaptive;
    }
    fail_pq("unsupported search_early_stop_mode: " + mode);
    return kSearchEarlyStopFixed;
}

bool highfreq_gpu_exact_rerank_enabled(const PqSearchOptions& options)
{
    return defaults::highfreq_rerank_mode_is_gpu(options.highfreq_rerank_mode);
}

bool highfreq_cpu_rerank_enabled(const PqSearchOptions& options)
{
    return defaults::highfreq_rerank_mode_is_cpu(options.highfreq_rerank_mode);
}

bool highfreq_cpu_streaming_enabled(const PqSearchOptions& options)
{
    return highfreq_cpu_rerank_enabled(options) && options.highfreq_rerank_streaming;
}

bool pq_device_resident_graph_enabled(const PqSearchOptions& options)
{
    return options.graph_transport == "device_resident";
}

uint32_t effective_highfreq_rerank_shortlist(const PqSearchOptions& options)
{
    return static_cast<uint32_t>(defaults::effective_highfreq_rerank_shortlist(
        options.highfreq_rerank_shortlist,
        options.search_queue_size,
        options.highfreq_rerank_mode,
        options.highfreq_rerank_algorithm));
}

int64_t load_status_flag_word(const int64_t* status_flag, size_t query_idx)
{
    const volatile int64_t* flag_ptr = status_flag + query_idx * 16;
    return *flag_ptr;
}

uint32_t highfreq_result_width(const PqSearchOptions& options)
{
    if ((highfreq_gpu_exact_rerank_enabled(options) &&
         !pq_device_resident_graph_enabled(options)) ||
        defaults::highfreq_rerank_mode_is_off(options.highfreq_rerank_mode)) {
        return static_cast<uint32_t>(options.topk);
    }
    return effective_highfreq_rerank_shortlist(options);
}

uint32_t highfreq_stop_width(const PqSearchOptions& options)
{
    return static_cast<uint32_t>(options.topk);
}

size_t round_up_gdr_bytes(size_t value)
{
    return ((value + kGdrAlignBytes - 1) / kGdrAlignBytes) * kGdrAlignBytes;
}

template <typename T>
T* align_gdr_pointer(T* ptr)
{
    const uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t aligned = (raw + kGdrAlignBytes - 1) & ~(static_cast<uintptr_t>(kGdrAlignBytes) - 1);
    return reinterpret_cast<T*>(aligned);
}

void save_results_to_csv_local(const std::vector<std::vector<idx_t>>& results, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        fail_pq("could not open output file " + filename);
    }

    for (const auto& row : results) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i + 1 < row.size()) {
                file << ',';
            }
        }
        file << '\n';
    }
}

template <typename T>
void load_bin_prefix_local(const std::string& path,
                           std::vector<T>& values,
                           uint64_t& rows,
                           uint64_t& cols,
                           size_t max_points,
                           size_t file_offset = 0)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_pq("unable to open bin file " + path);
    }
    input.seekg(file_offset, std::ios::beg);

    int rows_i32 = 0;
    int cols_i32 = 0;
    input.read(reinterpret_cast<char*>(&rows_i32), sizeof(rows_i32));
    input.read(reinterpret_cast<char*>(&cols_i32), sizeof(cols_i32));
    if (!input) {
        fail_pq("failed to read bin header from " + path);
    }

    rows = static_cast<uint64_t>(std::max(0, rows_i32));
    cols = static_cast<uint64_t>(std::max(0, cols_i32));
    const uint64_t loaded_rows = std::min<uint64_t>(rows, max_points);
    values.resize(static_cast<size_t>(loaded_rows * cols));
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!input) {
        fail_pq("failed to read bin payload from " + path);
    }
    rows = loaded_rows;
}

template <typename T>
void load_bin_local(const std::string& path,
                    std::vector<T>& values,
                    uint64_t& rows,
                    uint64_t& cols,
                    size_t file_offset = 0)
{
    load_bin_prefix_local(path, values, rows, cols, std::numeric_limits<size_t>::max(), file_offset);
}

struct PqPivotData {
    std::vector<uint32_t> chunk_offsets;
    std::vector<float> centroid;
    std::vector<float> tables;
    std::vector<float> tables_tr;
};

PqPivotData load_pq_pivots_local(const std::string& path, size_t num_chunks)
{
    std::vector<size_t> file_offsets;
    uint64_t nr = 0;
    uint64_t nc = 0;
    load_bin_local(path, file_offsets, nr, nc);
    if (nr != 4 && nr != 5) {
        fail_pq("unexpected pq pivots offset table size");
    }

    bool use_old_filetype = nr == 5;
    PqPivotData out;
    load_bin_local(path, out.tables, nr, nc, file_offsets[0]);
    const uint64_t ndims = nc;
    load_bin_local(path, out.centroid, nr, nc, file_offsets[1]);
    int chunk_offsets_index = use_old_filetype ? 3 : 2;
    load_bin_local(path, out.chunk_offsets, nr, nc, file_offsets[chunk_offsets_index]);
    if (nc != 1 || (nr != num_chunks + 1 && num_chunks != 0)) {
        fail_pq("unexpected pq chunk offsets shape");
    }

    out.tables_tr.resize(static_cast<size_t>(256 * ndims));
    for (size_t i = 0; i < 256; ++i) {
        for (size_t j = 0; j < ndims; ++j) {
            out.tables_tr[j * 256 + i] = out.tables[i * ndims + j];
        }
    }
    return out;
}

PqHostIndex load_pq_index(const PqFiles& files, size_t expected_rows, int dim, int pq_dim)
{
    uint64_t npts_u64 = 0;
    uint64_t nchunks_u64 = 0;
    std::vector<pq_idx_t> raw_codes;
    load_bin_prefix_local(files.compressed_path,
                          raw_codes,
                          npts_u64,
                          nchunks_u64,
                          expected_rows);
    if (npts_u64 != expected_rows) {
        fail_pq("compressed codes rows do not match expected rows");
    }

    size_t num_chunks = static_cast<size_t>(pq_dim);
    auto pivots = load_pq_pivots_local(files.pivots_path, num_chunks);

    PqHostIndex index;
    index.num_points = npts_u64;
    index.num_chunks = nchunks_u64;
    index.compressed_codes = std::move(raw_codes);
    index.centroid_table.resize(256ULL * static_cast<size_t>(dim));
    for (int i = 0; i < 256 * dim; ++i) {
        index.centroid_table[static_cast<size_t>(i)] = pivots.tables[static_cast<size_t>(i)] + pivots.centroid[static_cast<size_t>(i % dim)];
    }
    return index;
}

void save_results_to_csv(const std::vector<std::vector<idx_t>>& results,
                         const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        fail_pq("could not open output file " + filename);
    }

    for (const auto& row : results) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i + 1 < row.size()) {
                file << ',';
            }
        }
        file << '\n';
    }
}

void save_step_nums_to_csv(const std::vector<int>& step_nums,
                           const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        fail_pq("could not open output file " + filename);
    }

    for (int step_num : step_nums) {
        file << step_num << '\n';
    }
}

void save_route_flags_to_csv(const std::vector<int>& route_flags,
                             const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        fail_pq("could not open output file " + filename);
    }

    for (int route_flag : route_flags) {
        file << route_flag << '\n';
    }
}

std::string make_sidecar_csv_path(const std::string& csv_path,
                                  const std::string& suffix)
{
    const size_t dot = csv_path.find_last_of('.');
    if (dot == std::string::npos || dot == 0) {
        return csv_path + suffix;
    }
    return csv_path.substr(0, dot) + suffix;
}

std::string make_benchmark_results_path(const std::string& summary_csv_path,
                                        float quick_filter_bonus,
                                        int queue_size,
                                        int finish_count)
{
    char bonus_buf[32];
    std::snprintf(bonus_buf, sizeof(bonus_buf), "%.3f", quick_filter_bonus);
    std::string bonus_tag(bonus_buf);
    std::replace(bonus_tag.begin(), bonus_tag.end(), '.', '_');
    const std::string suffix = ".b" + bonus_tag
                             + ".q" + std::to_string(queue_size)
                             + ".f" + std::to_string(finish_count)
                             + ".results.csv";
    const size_t dot = summary_csv_path.find_last_of('.');
    if (dot == std::string::npos || dot == 0) {
        return summary_csv_path + suffix;
    }
    return summary_csv_path.substr(0, dot) + suffix;
}

std::string make_benchmark_steps_path(const std::string& summary_csv_path,
                                      float quick_filter_bonus,
                                      int queue_size,
                                      int finish_count)
{
    char bonus_buf[32];
    std::snprintf(bonus_buf, sizeof(bonus_buf), "%.3f", quick_filter_bonus);
    std::string bonus_tag(bonus_buf);
    std::replace(bonus_tag.begin(), bonus_tag.end(), '.', '_');
    const std::string suffix = ".b" + bonus_tag
                             + ".q" + std::to_string(queue_size)
                             + ".f" + std::to_string(finish_count)
                             + ".steps.csv";
    return make_sidecar_csv_path(summary_csv_path, suffix);
}

std::string make_benchmark_routes_path(const std::string& summary_csv_path,
                                       float quick_filter_bonus,
                                       int queue_size,
                                       int finish_count)
{
    char bonus_buf[32];
    std::snprintf(bonus_buf, sizeof(bonus_buf), "%.3f", quick_filter_bonus);
    std::string bonus_tag(bonus_buf);
    std::replace(bonus_tag.begin(), bonus_tag.end(), '.', '_');
    const std::string suffix = ".b" + bonus_tag
                             + ".q" + std::to_string(queue_size)
                             + ".f" + std::to_string(finish_count)
                             + ".routes.csv";
    return make_sidecar_csv_path(summary_csv_path, suffix);
}

std::string make_benchmark_timed_runs_path(const std::string& summary_csv_path,
                                           float quick_filter_bonus,
                                           int queue_size,
                                           int finish_count)
{
    char bonus_buf[32];
    std::snprintf(bonus_buf, sizeof(bonus_buf), "%.3f", quick_filter_bonus);
    std::string bonus_tag(bonus_buf);
    std::replace(bonus_tag.begin(), bonus_tag.end(), '.', '_');
    const std::string suffix = ".b" + bonus_tag
                             + ".q" + std::to_string(queue_size)
                             + ".f" + std::to_string(finish_count)
                             + ".timed_runs.csv";
    return make_sidecar_csv_path(summary_csv_path, suffix);
}

std::string csv_field(std::string_view value)
{
    if (value.find_first_of(",\"\r\n") == std::string_view::npos) {
        return std::string(value);
    }
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

uint64_t counter_delta(uint64_t after, uint64_t before) noexcept
{
    return after >= before ? after - before : after;
}

std::string make_latency_results_path(const std::string& summary_csv_path,
                                      int concurrency)
{
    return make_sidecar_csv_path(summary_csv_path,
                                 ".c" + std::to_string(concurrency) + ".results.csv");
}

std::string make_latency_steps_path(const std::string& summary_csv_path,
                                    int concurrency)
{
    return make_sidecar_csv_path(summary_csv_path,
                                 ".c" + std::to_string(concurrency) + ".steps.csv");
}

std::string make_latency_routes_path(const std::string& summary_csv_path,
                                     int concurrency)
{
    return make_sidecar_csv_path(summary_csv_path,
                                 ".c" + std::to_string(concurrency) + ".routes.csv");
}

double average_route_steps(int64_t total_steps, size_t route_count)
{
    return route_count > 0
        ? static_cast<double>(total_steps) / static_cast<double>(route_count)
        : 0.0;
}

__device__ void computePQTable(const pq_value_t* d_pq_centroid,
                               value_t* d_query,
                               pq_value_t* pq_table,
                               int pq_dim,
                               int k,
                               int dim,
                               int tid)
{
    int subvector_dim = dim / pq_dim;
    int step = warp_kernel::kWarpThreads;

    for (int subvector_idx = 0; subvector_idx < pq_dim; ++subvector_idx) {
        for (int centroid_idx = tid; centroid_idx < k; centroid_idx += step) {
            pq_value_t distance = 0;
            for (int dim_idx = 0; dim_idx < subvector_dim; ++dim_idx) {
                int query_dim_idx = subvector_idx * subvector_dim + dim_idx;
                int centroid_dim_idx = centroid_idx * dim + subvector_idx * subvector_dim + dim_idx;
                pq_value_t diff = d_query[query_dim_idx] - d_pq_centroid[centroid_dim_idx];
                distance += diff * diff;
            }
            pq_table[subvector_idx * k + centroid_idx] = distance;
        }
    }
}

__device__ void computePQTableSerial(const pq_value_t* d_pq_centroid,
                                     const value_t* d_query,
                                     pq_value_t* pq_table,
                                     int pq_dim,
                                     int k,
                                     int dim)
{
    int subvector_dim = dim / pq_dim;
    for (int subvector_idx = 0; subvector_idx < pq_dim; ++subvector_idx) {
        for (int centroid_idx = 0; centroid_idx < k; ++centroid_idx) {
            pq_value_t distance = 0;
            for (int dim_idx = 0; dim_idx < subvector_dim; ++dim_idx) {
                int query_dim_idx = subvector_idx * subvector_dim + dim_idx;
                int centroid_dim_idx = centroid_idx * dim + subvector_idx * subvector_dim + dim_idx;
                pq_value_t diff = d_query[query_dim_idx] - d_pq_centroid[centroid_dim_idx];
                distance += diff * diff;
            }
            pq_table[subvector_idx * k + centroid_idx] = distance;
        }
    }
}

__device__ void run_lowfreq_pq_scan_one_block(const pq_idx_t* d_data,
                                              const value_t* d_all_query,
                                              uint32_t query_idx,
                                              uint32_t qid,
                                              idx_t* d_result,
                                              float* d_result_dist,
                                              const pq_value_t* d_pq_centroid,
                                              pq_value_t* d_pq_table,
                                              const uint32_t* d_candidate_ids,
                                              const uint32_t* d_candidate_offsets,
                                              const uint32_t* d_candidate_sizes,
                                              int topk,
                                              int shortlist_k,
                                              int dim,
                                              int pq_dim,
                                              float* time_per_block,
                                              gpu_predicate::View posting_predicate = {},
                                              pq_resident_posting::View resident = {},
                                              bool score_first = false)
{
    const int tid = threadIdx.x;
    __shared__ float best_dist[kLowfreqRerankShortlistMax];
    __shared__ idx_t best_ids[kLowfreqRerankShortlistMax];
    if (tid != 0 || topk > kTopk || shortlist_k > kLowfreqRerankShortlistMax) {
        return;
    }

    const uint64_t start_time = time_per_block != nullptr ? warp_kernel::get_global_timer_ns() : 0;
    for (int i = 0; i < shortlist_k; ++i) {
        best_dist[i] = FLT_MAX;
        best_ids[i] = UINT32_MAX;
    }

    computePQTableSerial(d_pq_centroid,
                         d_all_query + static_cast<size_t>(query_idx) * dim,
                         d_pq_table + static_cast<size_t>(qid) * pq_dim * kNumPqCentroids,
                         pq_dim,
                         kNumPqCentroids,
                         dim);

    uint32_t begin = 0, count = 0;
    if (resident.active) {
        pq_resident_posting::bounds(resident, query_idx, begin, count);
    } else {
        begin = d_candidate_offsets[qid];
        count = d_candidate_sizes[qid];
    }
    for (uint32_t idx = 0; idx < count; ++idx) {
        const uint32_t sample_id = resident.active
            ? pq_resident_posting::candidate(resident, query_idx, begin + idx)
            : d_candidate_ids[begin + idx];
        if (sample_id == UINT32_MAX) continue;
        // query_idx is the routed predicate/vector row; qid is compact batch
        // scratch. Candidate IDs are original samples, never posting-local rows.
        if (!score_first && gpu_predicate::enabled(posting_predicate, query_idx) &&
            !gpu_predicate::matches(posting_predicate, query_idx, sample_id)) continue;
        float dist = 0.0f;
        for (int sub = 0; sub < pq_dim; ++sub) {
            const uint32_t code = d_data[static_cast<uint64_t>(sample_id) * pq_dim + sub];
            dist += d_pq_table[static_cast<size_t>(qid) * pq_dim * kNumPqCentroids + sub * kNumPqCentroids + code];
        }

        if (score_first) {
            if (!posting_shortlist_heap::can_enter<idx_t>(
                    dist, sample_id, best_dist, best_ids, shortlist_k)) continue;
            if (gpu_predicate::enabled(posting_predicate, query_idx) &&
                !gpu_predicate::matches(posting_predicate, query_idx, sample_id)) continue;
        }

        posting_shortlist_heap::insert<idx_t>(
            dist, sample_id, best_dist, best_ids, shortlist_k);
    }

    posting_shortlist_heap::sort(best_dist, best_ids, shortlist_k);
    for (int i = 0; i < shortlist_k; ++i) {
        d_result[static_cast<size_t>(qid) * shortlist_k + i] = best_ids[i];
        d_result_dist[static_cast<size_t>(qid) * shortlist_k + i] = best_dist[i];
    }
    if (time_per_block != nullptr) {
        const uint64_t stop_time = warp_kernel::get_global_timer_ns();
        const float elapsed_us = static_cast<float>(stop_time - start_time) / 1000.0f;
        time_per_block[static_cast<size_t>(query_idx) * 6] = elapsed_us;
        time_per_block[static_cast<size_t>(query_idx) * 6 + 1] = 0.0f;
        time_per_block[static_cast<size_t>(query_idx) * 6 + 2] = elapsed_us;
        time_per_block[static_cast<size_t>(query_idx) * 6 + 3] = 0.0f;
        time_per_block[static_cast<size_t>(query_idx) * 6 + 4] =
            static_cast<float>(count);
    }
}

__global__ void lowfreq_pq_scan_kernel(const pq_idx_t* d_data,
                                       const value_t* d_all_query,
                                       const uint32_t* d_query_ids,
                                       idx_t* d_result,
                                       float* d_result_dist,
                                       const pq_value_t* d_pq_centroid,
                                       pq_value_t* d_pq_table,
                                       const uint32_t* d_candidate_ids,
                                       const uint32_t* d_candidate_offsets,
                                       const uint32_t* d_candidate_sizes,
                                       int num_query,
                                       int topk,
                                       int shortlist_k,
                                       int dim,
                                       int pq_dim,
                                       gpu_predicate::View posting_predicate,
                                       pq_resident_posting::View resident,
                                       bool score_first,
                                       idx_t* mapped_result,
                                       uint32_t* mapped_ready)
{
    const uint32_t qid = static_cast<uint32_t>(blockIdx.x);
    if (qid >= static_cast<uint32_t>(num_query)) {
        return;
    }
    const uint32_t query_idx = d_query_ids[qid];
    run_lowfreq_pq_scan_one_block(d_data,
                                  d_all_query,
                                  query_idx,
                                  qid,
                                  d_result,
                                  d_result_dist,
                                  d_pq_centroid,
                                  d_pq_table,
                                  d_candidate_ids,
                                  d_candidate_offsets,
                                  d_candidate_sizes,
                                  topk,
                                  shortlist_k,
                                  dim,
                                  pq_dim,
                                  nullptr, posting_predicate, resident, score_first);
    if (threadIdx.x == 0 && mapped_ready != nullptr) {
        for (int i = 0; i < shortlist_k; ++i)
            mapped_result[static_cast<size_t>(qid) * shortlist_k + i] =
                d_result[static_cast<size_t>(qid) * shortlist_k + i];
        __threadfence_system();
        *reinterpret_cast<volatile uint32_t*>(mapped_ready + qid) = 1;
    }
}

float exact_l2_u8(const uint8_t* query, const uint8_t* base, uint32_t dim)
{
    float dist = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        float diff = static_cast<float>(query[d]) - static_cast<float>(base[d]);
        dist += diff * diff;
    }
    return dist;
}

std::shared_ptr<ExactDistanceMemo> make_pq_iteration_distance_memo(
    const GraphRuntime& runtime, const PqSearchOptions& options)
{
    if (!options.rerank_distance_reuse) return {};
    if (options.query_worker_mode != "coupled" || options.graph_transport != "mailbox" ||
        options.graph_storage_mode != "memory" ||

        runtime.original_query_count == 0) return {};
    const uint32_t dim = runtime.query_vectors.cols;
    if (runtime.original_query_vectors.rows != runtime.original_query_count ||
        runtime.original_query_vectors.cols != dim || runtime.base_vectors.cols != dim ||
        runtime.route_to_original_query.size() != runtime.query_vectors.rows)
        fail_pq("PQ exact distance memo requires aligned original vector domain");
    std::vector<uint8_t> shared_queries(runtime.original_query_count, 0);
    for (uint32_t route = 0; route < runtime.query_vectors.rows; ++route) {
        const uint32_t original = runtime.route_to_original_query[route];
        if (original >= runtime.original_query_count ||
            std::memcmp(runtime.query_vectors.row_ptr(route),
                        runtime.original_query_vectors.row_ptr(original), dim) != 0)
            fail_pq("PQ exact distance memo route does not match original query vector");
        if ((runtime.route_enabled.empty() || runtime.route_enabled.at(route)) &&
            shared_queries[original] < 2) ++shared_queries[original];
    }
    bool has_sharing = false;
    for (auto& count : shared_queries) { count = count > 1; has_sharing |= count != 0; }
    if (!has_sharing) return {};
    return std::make_shared<ExactDistanceMemo>(ExactDistanceMemo::Domain{
        runtime.original_query_count, runtime.base_vectors.rows, dim, 1,
        &runtime.base_vectors, &runtime.original_query_vectors}, std::move(shared_queries));
}

ExactDistanceMemo::Stats finish_pq_distance_memo(PqSearchOptions& options)
{
    const auto stats = options.exact_distance_memo
        ? options.exact_distance_memo->stats() : ExactDistanceMemo::Stats{};
    options.exact_distance_memo.reset();
    return stats;
}

void report_pq_distance_memo(const ExactDistanceMemo::Stats& stats, bool timed,
                             bool requested, bool enabled)
{
    std::cout << "[exact-distance-policy] backend=pq requested=" << requested
              << " enabled=" << enabled << '\n';
    if (!enabled) return;
    std::cout << "[pq-exact-distance-memo] phase=" << (timed ? "timed" : "warmup")
              << " computations=" << stats.computations << " hits=" << stats.hits
              << " entries=" << stats.entries
              << " adaptive_bypassed_queries=" << stats.adaptive_bypassed_queries
              << " counters=cached_admissions\n";
}

float highfreq_exact_l2_for_sample(const GraphRuntime& runtime,
                                   const uint8_t* query,
                                   uint32_t sample_id)
{
    return exact_l2_u8(query, runtime.base_vectors.row_ptr(sample_id), runtime.query_vectors.cols);
}

void lowfreq_host_heuristic_rerank_range(const GraphRuntime& runtime,
                                         const DenseUint8Matrix& query_vectors,
                                         const std::vector<uint32_t>& query_indices,
                                         const std::vector<idx_t>& gpu_shortlist,
                                         uint32_t shortlist_k,
                                         uint32_t topk,
                                         uint32_t rerank_batch_size,
                                         float rerank_epsilon,
                                         uint32_t rerank_stable_batches,
                                         bool filter_full_labels,
                                         size_t begin_query,
                                         size_t end_query,
                                         std::vector<std::vector<idx_t>>& out_results,
                                         std::vector<int>& out_step_nums,
                                         std::vector<std::vector<float>>* out_distances,
                                         const std::vector<uint8_t>* verified_routes = nullptr,
                                         ExactDistanceMemo* distance_memo = nullptr)
{
    std::vector<uint8_t> base_row(query_vectors.cols);
    std::vector<uint32_t> batch_sample_ids;
    batch_sample_ids.reserve(rerank_batch_size);
    for (size_t q = begin_query; q < end_query; ++q) {
        const uint8_t* query = query_vectors.row_ptr(query_indices[q]);
        std::vector<std::pair<float, idx_t>> heap;
        heap.reserve(topk);
        std::vector<idx_t> prev_topk(topk, UINT32_MAX);
        int stable_count = 0;
        int batch_count = 0;

        for (uint32_t batch_begin = 0; batch_begin < shortlist_k; batch_begin += rerank_batch_size) {
            const uint32_t batch_end = std::min<uint32_t>(shortlist_k, batch_begin + rerank_batch_size);
            ++batch_count;
            batch_sample_ids.clear();

            for (uint32_t idx = batch_begin; idx < batch_end; ++idx) {
                const idx_t sample_id = gpu_shortlist[static_cast<size_t>(q) * shortlist_k + idx];
                if (sample_id == UINT32_MAX || sample_id >= runtime.base_vectors.rows) {
                    continue;
                }
                if (filter_full_labels &&
                    !pq_posting_streaming::predicate_verified(verified_routes, query_indices[q]) &&
                    !sample_matches_lowfreq_full_labels(runtime,
                                                        query_indices[q],
                                                        static_cast<uint32_t>(sample_id))) {
                    continue;
                }
                batch_sample_ids.push_back(static_cast<uint32_t>(sample_id));
            }

            for (uint32_t sample_id : batch_sample_ids) {
                const auto compute = [&] {
                    copy_lowfreq_base_vector_row(runtime, sample_id, base_row.data(),
                                                query_vectors.cols);
                    return exact_l2_u8(query, base_row.data(), query_vectors.cols);
                };
                const float dist = distance_memo
                    ? distance_memo->evaluate(runtime.route_to_original_query.at(query_indices[q]),
                                              sample_id, compute)
                    : compute();

                if (heap.size() < topk) {
                    heap.emplace_back(dist, sample_id);
                    std::push_heap(heap.begin(), heap.end());
                } else if (dist < heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = {dist, sample_id};
                    std::push_heap(heap.begin(), heap.end());
                }
            }

            std::vector<idx_t> curr_topk(topk, UINT32_MAX);
            auto heap_copy = heap;
            std::sort_heap(heap_copy.begin(), heap_copy.end());
            for (size_t i = 0; i < heap_copy.size() && i < topk; ++i) {
                curr_topk[i] = heap_copy[i].second;
            }

            int changed = 0;
            for (uint32_t i = 0; i < topk; ++i) {
                if (curr_topk[i] != prev_topk[i]) {
                    ++changed;
                }
            }
            const float change_rate = static_cast<float>(changed) / static_cast<float>(topk);
            if (heap.size() >= topk && change_rate <= rerank_epsilon) {
                ++stable_count;
                if (stable_count >= static_cast<int>(rerank_stable_batches)) {
                    prev_topk = std::move(curr_topk);
                    break;
                }
            } else {
                stable_count = 0;
            }
            prev_topk = std::move(curr_topk);
        }

        out_results[q] = std::move(prev_topk);
        if (out_distances != nullptr) {
            auto heap_copy = heap;
            std::sort_heap(heap_copy.begin(), heap_copy.end());
            for (size_t i = 0; i < heap_copy.size() && i < topk; ++i) {
                (*out_distances)[q][i] = heap_copy[i].first;
            }
        }
        out_step_nums[q] = batch_count;
    }
}

void lowfreq_host_heuristic_rerank(const GraphRuntime& runtime,
                                   const DenseUint8Matrix& query_vectors,
                                   const std::vector<uint32_t>& query_indices,
                                   const std::vector<idx_t>& gpu_shortlist,
                                   uint32_t shortlist_k,
                                   uint32_t topk,
                                   uint32_t rerank_batch_size,
                                   float rerank_epsilon,
                                   uint32_t rerank_stable_batches,
                                   bool filter_full_labels,
                                   int rerank_threads,
                                   std::vector<std::vector<idx_t>>& out_results,
                                   std::vector<int>& out_step_nums,
                                   std::vector<std::vector<float>>* out_distances,
                                   CpuRerankWorkerPool* cpu_workers = nullptr,
                                   const std::vector<uint8_t>* verified_routes = nullptr,
                                   ExactDistanceMemo* distance_memo = nullptr)
{
    perf_trace::Scope trace_scope("lowfreq_cpu_finalize", &runtime);
    out_results.assign(query_indices.size(), std::vector<idx_t>(topk, UINT32_MAX));
    out_step_nums.assign(query_indices.size(), 0);
    if (out_distances != nullptr) {
        out_distances->assign(query_indices.size(), std::vector<float>(topk, FLT_MAX));
    }
    if (query_indices.empty()) {
        return;
    }

    const size_t requested_threads = static_cast<size_t>(std::max(1, rerank_threads));
    const size_t worker_count = std::min(requested_threads, query_indices.size());
    if (worker_count <= 1) {
        lowfreq_host_heuristic_rerank_range(runtime,
                                            query_vectors,
                                            query_indices,
                                            gpu_shortlist,
                                            shortlist_k,
                                            topk,
                                            rerank_batch_size,
                                            rerank_epsilon,
                                            rerank_stable_batches,
                                            filter_full_labels,
                                            0,
                                            query_indices.size(),
                                            out_results,
                                            out_step_nums,
                                            out_distances, verified_routes, distance_memo);
        return;
    }

    auto rerank_partition = [&](size_t worker) {
        const size_t begin = (query_indices.size() * worker) / worker_count;
        const size_t end = (query_indices.size() * (worker + 1)) / worker_count;
        lowfreq_host_heuristic_rerank_range(runtime,
                                                        query_vectors,
                                                        query_indices,
                                                        gpu_shortlist,
                                                        shortlist_k,
                                                        topk,
                                                        rerank_batch_size,
                                                        rerank_epsilon,
                                                        rerank_stable_batches,
                                                        filter_full_labels,
                                                        begin,
                                                        end,
                                                        out_results,
                                                        out_step_nums,
                                                        out_distances, verified_routes, distance_memo);
    };
    if (cpu_workers != nullptr) {
        cpu_workers->submit(worker_count, rerank_partition)->wait();
        return;
    }

    std::exception_ptr worker_error;
    std::mutex worker_error_mutex;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    try {
        for (size_t worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back([&, worker] {
                try {
                    rerank_partition(worker);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(worker_error_mutex);
                    if (worker_error == nullptr) {
                        worker_error = std::current_exception();
                    }
                }
            });
        }
    } catch (...) {
        for (std::thread& worker : workers) {
            worker.join();
        }
        throw;
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    if (worker_error != nullptr) {
        std::rethrow_exception(worker_error);
    }
}

std::vector<idx_t> sorted_topk_ids_from_heap(std::vector<std::pair<float, idx_t>> heap,
                                             uint32_t topk)
{
    std::sort_heap(heap.begin(), heap.end());
    std::vector<idx_t> ids(topk, UINT32_MAX);
    for (size_t i = 0; i < heap.size() && i < topk; ++i) {
        ids[i] = heap[i].second;
    }
    return ids;
}

void highfreq_host_batch_rerank_range(const GraphRuntime& runtime,
                                      const DenseUint8Matrix& query_vectors,
                                      const std::vector<uint32_t>& query_indices,
                                      const std::vector<std::vector<idx_t>>& gpu_shortlist,
                                      uint32_t shortlist_k,
                                      uint32_t topk,
                                      uint32_t rerank_batch_size,
                                      float rerank_epsilon,
                                      uint32_t rerank_stable_batches,
                                      bool allow_early_stop,
                                      size_t begin_query,
                                      size_t end_query,
                                      std::vector<std::vector<idx_t>>& out_results,
                                      std::vector<std::vector<float>>* out_distances,
                                      bool shortlist_predicate_verified = false,
                                      ExactDistanceMemo* distance_memo = nullptr)
{
    rerank_batch_size = std::max<uint32_t>(1U, rerank_batch_size);
    rerank_stable_batches = std::max<uint32_t>(1U, rerank_stable_batches);

    std::vector<uint32_t> batch_sample_ids;
    batch_sample_ids.reserve(rerank_batch_size);
    for (size_t q = begin_query; q < end_query; ++q) {
        const uint8_t* query = query_vectors.row_ptr(query_indices[q]);
        std::vector<std::pair<float, idx_t>> heap;
        heap.reserve(topk);
        std::vector<idx_t> prev_topk(allow_early_stop ? topk : 0, UINT32_MAX);
        int stable_count = 0;
        const uint32_t effective_shortlist =
            static_cast<uint32_t>(std::min<size_t>(gpu_shortlist[q].size(), shortlist_k));
        std::unordered_set<uint32_t> seen_sample_ids;
        if (allow_early_stop) {
            seen_sample_ids.reserve(effective_shortlist);
        }

        for (uint32_t batch_begin = 0; batch_begin < effective_shortlist; batch_begin += rerank_batch_size) {
            const uint32_t batch_end = std::min<uint32_t>(effective_shortlist, batch_begin + rerank_batch_size);
            batch_sample_ids.clear();

            for (uint32_t idx = batch_begin; idx < batch_end; ++idx) {
                const idx_t sample_id = gpu_shortlist[q][idx];
                if (sample_id == UINT32_MAX || sample_id >= runtime.base_vectors.rows) {
                    continue;
                }
                const auto sample_id_u32 = static_cast<uint32_t>(sample_id);
                if (allow_early_stop &&
                    !seen_sample_ids.insert(sample_id_u32).second) {
                    continue;
                }
                if (!shortlist_predicate_verified && !sample_matches_lowfreq_full_labels(runtime,
                                                        query_indices[q],
                                                        sample_id_u32)) {
                    continue;
                }
                batch_sample_ids.push_back(sample_id_u32);
            }
            if (batch_sample_ids.empty()) {
                continue;
            }
            std::sort(batch_sample_ids.begin(), batch_sample_ids.end());
            batch_sample_ids.erase(std::unique(batch_sample_ids.begin(), batch_sample_ids.end()),
                                   batch_sample_ids.end());

            for (uint32_t sample_id : batch_sample_ids) {
                const auto compute = [&] {
                    return highfreq_exact_l2_for_sample(runtime, query, sample_id);
                };
                const float dist = distance_memo
                    ? distance_memo->evaluate(runtime.route_to_original_query.at(query_indices[q]),
                                              sample_id, compute)
                    : compute();
                if (heap.size() < topk) {
                    heap.emplace_back(dist, sample_id);
                    std::push_heap(heap.begin(), heap.end());
                } else if (dist < heap.front().first ||
                           (dist == heap.front().first && sample_id < heap.front().second)) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = {dist, sample_id};
                    std::push_heap(heap.begin(), heap.end());
                }
            }

            if (allow_early_stop && heap.size() >= topk) {
                std::vector<idx_t> curr_topk = sorted_topk_ids_from_heap(heap, topk);
                int changed = 0;
                for (uint32_t i = 0; i < topk; ++i) {
                    if (curr_topk[i] != prev_topk[i]) {
                        ++changed;
                    }
                }
                const float change_rate = static_cast<float>(changed) / static_cast<float>(topk);
                if (change_rate <= rerank_epsilon) {
                    ++stable_count;
                    if (stable_count >= static_cast<int>(rerank_stable_batches)) {
                        prev_topk = std::move(curr_topk);
                        break;
                    }
                } else {
                    stable_count = 0;
                }
                prev_topk = std::move(curr_topk);
            }
        }

        auto sorted = heap;
        std::sort_heap(sorted.begin(), sorted.end());
        for (size_t i = 0; i < sorted.size() && i < topk; ++i) {
            out_results[q][i] = sorted[i].second;
            if (out_distances != nullptr) {
                (*out_distances)[q][i] = sorted[i].first;
            }
        }
    }
}

void highfreq_host_rerank(const PqSearchOptions& options,
                          const GraphRuntime& runtime,
                          const DenseUint8Matrix& query_vectors,
                          const std::vector<uint32_t>& query_indices,
                          const std::vector<std::vector<idx_t>>& gpu_shortlist,
                          std::vector<std::vector<idx_t>>& out_results,
                          std::vector<std::vector<float>>* out_distances,
                          bool shortlist_predicate_verified = false,
                          CpuRerankWorkerPool* cpu_workers = nullptr)
{
    perf_trace::Scope trace_scope("highfreq_cpu_rerank_batch", &runtime);
    const uint32_t topk = static_cast<uint32_t>(options.topk);
    const uint32_t shortlist_k = effective_highfreq_rerank_shortlist(options);
    const bool heuristic =
        defaults::highfreq_rerank_algorithm_is_heuristic(options.highfreq_rerank_algorithm);
    const uint32_t batch_size = heuristic
        ? static_cast<uint32_t>(options.highfreq_rerank_batch_size)
        : std::max<uint32_t>(1U, shortlist_k);
    const uint32_t stable_batches = heuristic
        ? static_cast<uint32_t>(options.highfreq_rerank_stable_batches)
        : UINT32_MAX;
    const float epsilon = heuristic ? options.highfreq_rerank_epsilon : -1.0f;

    out_results.assign(query_indices.size(), std::vector<idx_t>(topk, UINT32_MAX));
    if (out_distances != nullptr) {
        out_distances->assign(query_indices.size(), std::vector<float>(topk, FLT_MAX));
    }
    if (query_indices.empty()) {
        return;
    }

    const size_t requested_threads = static_cast<size_t>(std::max(1, options.highfreq_rerank_threads));
    const size_t worker_count = std::min(requested_threads, query_indices.size());
    if (worker_count <= 1) {
        highfreq_host_batch_rerank_range(runtime,
                                         query_vectors,
                                         query_indices,
                                         gpu_shortlist,
                                         shortlist_k,
                                         topk,
                                         batch_size,
                                         epsilon,
                                         stable_batches,
                                         heuristic,
                                         0,
                                         query_indices.size(),
                                         out_results,
                                         out_distances, shortlist_predicate_verified, options.exact_distance_memo.get());
        return;
    }

    auto rerank_partition = [&](size_t worker) {
        const size_t begin = (query_indices.size() * worker) / worker_count;
        const size_t end = (query_indices.size() * (worker + 1)) / worker_count;
        highfreq_host_batch_rerank_range(runtime,
                                             query_vectors,
                                             query_indices,
                                             gpu_shortlist,
                                             shortlist_k,
                                             topk,
                                             batch_size,
                                             epsilon,
                                             stable_batches,
                                             heuristic,
                                             begin,
                                             end,
                                             out_results,
                                             out_distances, shortlist_predicate_verified, options.exact_distance_memo.get());
    };
    if (cpu_workers != nullptr) {
        cpu_workers->submit(worker_count, rerank_partition)->wait();
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    std::exception_ptr worker_error;
    std::mutex worker_error_mutex;
    try {
        for (size_t worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back([&, worker] {
                try { rerank_partition(worker); }
                catch (...) {
                    std::lock_guard<std::mutex> lock(worker_error_mutex);
                    if (!worker_error) worker_error = std::current_exception();
                }
            });
        }
    } catch (...) {
        for (auto& worker : workers) worker.join();
        throw;
    }
    for (auto& worker : workers) worker.join();
    if (worker_error) std::rethrow_exception(worker_error);
}

void highfreq_host_rerank_partial(const PqSearchOptions& options,
                                  const GraphRuntime& runtime,
                                  const DenseUint8Matrix& query_vectors,
                                  const std::vector<uint32_t>& query_indices,
                                  const std::vector<std::vector<idx_t>>& gpu_shortlist,
                                  size_t begin_query,
                                  size_t end_query,
                                  std::vector<std::vector<idx_t>>& out_results,
                                  std::vector<std::vector<float>>* out_distances)
{
    perf_trace::Scope trace_scope("highfreq_cpu_rerank_batch", &runtime);
    if (begin_query >= end_query || end_query > query_indices.size() || end_query > gpu_shortlist.size()) {
        return;
    }
    const uint32_t topk = static_cast<uint32_t>(options.topk);
    const uint32_t shortlist_k = effective_highfreq_rerank_shortlist(options);
    const bool heuristic =
        defaults::highfreq_rerank_algorithm_is_heuristic(options.highfreq_rerank_algorithm);
    const uint32_t batch_size = heuristic
        ? static_cast<uint32_t>(options.highfreq_rerank_batch_size)
        : std::max<uint32_t>(1U, shortlist_k);
    const uint32_t stable_batches = heuristic
        ? static_cast<uint32_t>(options.highfreq_rerank_stable_batches)
        : UINT32_MAX;
    const float epsilon = heuristic ? options.highfreq_rerank_epsilon : -1.0f;

    highfreq_host_batch_rerank_range(runtime,
                                     query_vectors,
                                     query_indices,
                                     gpu_shortlist,
                                     shortlist_k,
                                     topk,
                                     batch_size,
                                     epsilon,
                                     stable_batches,
                                     heuristic,
                                     begin_query,
                                     end_query,
                                     out_results,
                                     out_distances);
}

void finalize_highfreq_results(const PqSearchOptions& options,
                               const GraphRuntime& runtime,
                               const std::vector<uint32_t>& query_indices,
                               std::vector<std::vector<idx_t>>& results,
                               std::vector<std::vector<float>>* result_distances,
                               bool shortlist_predicate_verified = false,
                               CpuRerankWorkerPool* cpu_workers = nullptr)
{
    if (highfreq_gpu_exact_rerank_enabled(options)) {
        return;
    }

    if (defaults::highfreq_rerank_mode_is_cpu(options.highfreq_rerank_mode)) {
        std::vector<std::vector<idx_t>> reranked_results;
        std::vector<std::vector<float>> reranked_distances;
        highfreq_host_rerank(options,
                             runtime,
                             runtime.query_vectors,
                             query_indices,
                             results,
                             reranked_results,
                             result_distances != nullptr ? &reranked_distances : nullptr,
                             shortlist_predicate_verified, cpu_workers);
        results = std::move(reranked_results);
        if (result_distances != nullptr) {
            *result_distances = std::move(reranked_distances);
        }
        return;
    }

    for (auto& row : results) {
        row.resize(static_cast<size_t>(options.topk), UINT32_MAX);
    }
    if (result_distances != nullptr) {
        for (auto& row : *result_distances) {
            row.resize(static_cast<size_t>(options.topk), FLT_MAX);
        }
    }
}

template <typename LaunchFunc>
void dispatch_kernel_by_queue_size(int queue_size, LaunchFunc&& launch)
{
    const uint32_t queue_capacity = specialized_queue_capacity(queue_size);
    if (queue_capacity == 64) {
        launch.template operator()<64>();
    } else if (queue_capacity == 128) {
        launch.template operator()<128>();
    } else if (queue_capacity == 256) {
        launch.template operator()<256>();
    } else if (queue_capacity == 512) {
        launch.template operator()<512>();
    } else {
        fail_pq("unsupported queue_size (must be in [1, 512])");
    }
}

template <typename LaunchFunc>
void dispatch_graph_kernel(int queue_size, int pq_dim, LaunchFunc&& launch)
{
    require_pq_graph_pq_dim_supported(pq_dim);
    dispatch_kernel_by_queue_size(queue_size, [&]<int QS>() {
        if (pq_dim == kGraphPqDim16) {
            launch.template operator()<QS, kGraphPqDim16>();
        } else {
            launch.template operator()<QS, kGraphPqDim64>();
        }
    });
}

template <typename KernelFunc>
static uint32_t resident_worker_capacity_for_latency_kernel(KernelFunc kernel,
                                                            int thread_in_block,
                                                            const char* occupancy_context,
                                                            const char* device_context)
{
    int blocks_per_sm = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,
                                                             kernel,
                                                             thread_in_block,
                                                             0),
               occupancy_context);
    cudaDeviceProp prop{};
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), device_context);
    check_cuda(cudaGetDeviceProperties(&prop, device_id), device_context);
    return static_cast<uint32_t>(std::max(0, blocks_per_sm)) *
           static_cast<uint32_t>(prop.multiProcessorCount);
}

__device__ __forceinline__ int adaptive_early_stop_limit(int queue_limit,
                                                         int expanded_count,
                                                         int pass_count,
                                                         int stop_width)
{
    constexpr int kPriorPass = 8;
    constexpr int kPriorTotal = 16;
    constexpr int kBaseNumerator = 1;
    constexpr int kBaseDenominator = 2;
    constexpr int kMinBaseTopkMultiplier = 2;
    constexpr int kExtraNumerator = 1;
    constexpr int kExtraDenominator = 2;
    constexpr int kMaxExtraPatience = 32;
    const int queue_base = queue_limit > 1 ? queue_limit : 1;
    const int base_numerator = queue_base * kBaseNumerator;
    int base_limit = (base_numerator + kBaseDenominator - 1) / kBaseDenominator;
    const int min_base_limit = stop_width > 0 ? stop_width * kMinBaseTopkMultiplier : 1;
    base_limit = base_limit > min_base_limit ? base_limit : min_base_limit;
    const int denom = (expanded_count + kPriorTotal) > 1 ? (expanded_count + kPriorTotal) : 1;
    const int pass_with_prior = pass_count + kPriorPass;
    const int pass_estimate = pass_with_prior < denom ? pass_with_prior : denom;
    const int reject_estimate = (denom - pass_estimate) > 0 ? (denom - pass_estimate) : 0;
    const int64_t numerator = static_cast<int64_t>(queue_limit) *
                              kExtraNumerator *
                              reject_estimate;
    int extra = static_cast<int>((numerator + (kExtraDenominator * denom - 1)) /
                                 (kExtraDenominator * denom));
    extra = extra > 0 ? extra : 0;
    extra = extra < kMaxExtraPatience ? extra : kMaxExtraPatience;
    const int limit = base_limit + extra;
    return limit;
}

__device__ __forceinline__ bool early_stop_reached(int early_stop_mode,
                                                   int bad_count,
                                                   int fixed_finish_limit,
                                                   int queue_limit,
                                                   int expanded_count,
                                                   int pass_count,
                                                   int stop_width)
{
    if (early_stop_mode == kSearchEarlyStopOff) {
        return false;
    }
    if (early_stop_mode == kSearchEarlyStopAdaptive) {
        return bad_count >= adaptive_early_stop_limit(queue_limit,
                                                      expanded_count,
                                                      pass_count,
                                                      stop_width);
    }
    return bad_count >= fixed_finish_limit;
}

template<class Pair>
struct PqLane0SearchStateT {
    Pair now;
    Pair next;
    dist_t stop_topk_pq_worst;
    dist_t current_pq_distance;
    uint16_t stop_topk_heap_size;
    uint16_t topk_heap_size;
    uint8_t now_filter_result;
};

using PqLane0SearchState = PqLane0SearchStateT<PoolPair>;

static_assert(kHighfreqRerankShortlistMax + 1 < (1 << 16),
              "PQ native heap sizes must fit in uint16_t");
static_assert(sizeof(PqLane0SearchState) == 32,
              "PQ lane-0 state must remain compact");

struct PqFiniteKernelParams {
    const pq_idx_t* d_data;
    value_t* d_query;
    idx_t* d_result;
    float* d_result_dist;
    const pq_value_t* d_pq_centroid;
    pq_value_t* d_pq_table;
    uint32_t* d_hashdata;
    float* time_per_block;
    uint32_t* d_neighbor_memory;
    volatile int64_t* d_status_memory;
    volatile int64_t* d_gdr_data;
    volatile int64_t* d_status_flag;
    const Node* d_prefetch_node_array;
    const uint32_t* d_prefetch_index;
    const uint32_t* d_prefetch_begin_by_group;
    const uint32_t* d_prefetch_end_by_group;
    const uint32_t* d_query_group_ids;
    const uint32_t* d_start_local_ids;
    const uint32_t* d_start_sample_ids;
    const uint32_t* d_row_to_sample;
    const uint32_t* d_label_offset;
    const uint64_t* d_row_filter_mask;
    const uint64_t* d_query_filter_bits;
    const uint8_t* d_query_range_modes;
    const float* d_query_range_lower_bounds;
    const float* d_query_range_upper_bounds;
    const float* d_base_range_values;
    PqFiniteContextControl* d_contexts;
    PqFiniteContextTelemetry* d_context_telemetry;
    PoolPair* d_frontier_state;
    StopPair* d_stop_state;
    ResultPair* d_result_state;
    uint32_t query_capacity;
    uint32_t range_count;
    int query_dim;
    int query_pq_dim;
    int queue_limit;
    int finish_limit;
    int early_stop_mode;
    int max_steps;
    int stop_width;
    int result_width;
    uint32_t mailbox_sample_id_word;
    uint32_t mailbox_neighbor_count_word;
    uint32_t mailbox_neighbor_words_offset;
    uint32_t mailbox_neighbor_sample_words_offset;
    uint32_t mailbox_slot_words;
    int use_sample_id_payload;
    int use_implicit_filter_mask;
    int enable_true_distance_rerank;
    float bit_filter_bonus;
    float range_filter_bonus;
    int debug_mailbox;
};

struct PqFiniteFixHashView {
    uint32_t* data = nullptr;

    __device__ __forceinline__ short hash(uint32_t value) const
    {
        constexpr uint64_t kRandom0 = 0x4bcb391f924ed183ULL;
        constexpr uint64_t kRandom1 = 0xa0ab69ccd854fc0aULL;
        return static_cast<short>(
            (value ^ (static_cast<uint64_t>(value >> 16) * kRandom0) ^ kRandom1) %
            kHashTableCapacity);
    }

    __device__ __forceinline__ bool test(uint32_t value) const
    {
        short code = hash(value);
        for (int probes = 0; probes < kHashTableCapacity; ++probes) {
            if (data[code] == UINT32_MAX) return false;
            if (data[code] == value) return true;
            code = static_cast<short>((code + 1) % kHashTableCapacity);
        }
        return false;
    }

    __device__ __forceinline__ void add(uint32_t value) const
    {
        short code = hash(value);
        for (int probes = 0; probes < kHashTableCapacity; ++probes) {
            if (data[code] == value) return;
            if (data[code] == UINT32_MAX) {
                data[code] = value;
                return;
            }
            code = static_cast<short>((code + 1) % kHashTableCapacity);
        }
    }
};

template <int kQueueSize>
__device__ __forceinline__ void store_finite_mutable_state(
    const PqFiniteKernelParams& params,
    uint32_t query_idx,
    int subtid,
    const PoolPair* frontier,
    const StopPair* stop_heap,
    const ResultPair* result_heap)
{
    constexpr int step = warp_kernel::kWarpThreads;
    const size_t frontier_base = static_cast<size_t>(query_idx) * (kQueueSize + 2);
    const size_t stop_base = static_cast<size_t>(query_idx) * (kTopk + 1);
    const size_t result_base = static_cast<size_t>(query_idx) *
        (static_cast<size_t>(params.result_width) + 1);
    for (int i = subtid; i < kQueueSize + 2; i += step) {
        params.d_frontier_state[frontier_base + i] = frontier[i];
    }
    for (int i = subtid; i < kTopk + 1; i += step) {
        params.d_stop_state[stop_base + i] = stop_heap[i];
    }
    for (int i = subtid; i < params.result_width + 1; i += step) {
        params.d_result_state[result_base + i] = result_heap[i];
    }
}

template <int kQueueSize, int kGraphPqDim, bool kInitialize, bool kCollectPerHopTiming>
__device__ void run_pq_finite_context(const PqFiniteKernelParams& params,
                                      uint32_t active_query_idx,
                                      uint32_t query_idx,
                                      uint32_t context_epoch,
                                      PqFiniteReadyToken* token,
                                      ResultPair* topk,
                                      PqFinitePerHopTiming* per_hop_timing)
{
    constexpr int step = warp_kernel::kWarpThreads;
    const int cid = threadIdx.x / kCriticalStep;
    const int subtid = threadIdx.x % kCriticalStep;

    __shared__ PoolPair q[kMultiQuery][kQueueSize + 2];
    __shared__ StopPair stop_topk[kMultiQuery][kTopk + 1];
    __shared__ dist_t dist_list[kMultiQuery][FIXED_DEGREE * kMultiprobe];
    __shared__ idx_t index_list[kMultiQuery][FIXED_DEGREE * kMultiprobe];
    __shared__ uint32_t index_sample_list[kMultiQuery][FIXED_DEGREE * kMultiprobe];
    __shared__ int index_list_len[kMultiQuery];
    __shared__ value_t query_point[kMultiQuery][kGraphMaxDim];
    __shared__ PqFiniteContextControl control[kMultiQuery];
    __shared__ float calculation_time_us[kMultiQuery];
    __shared__ float queue_time_us[kMultiQuery];
    __shared__ uint32_t resume_quantum_steps[kMultiQuery];
    __shared__ uint32_t static_prefetch_steps[kMultiQuery];
    __shared__ uint32_t now_filter_result_shared[kMultiQuery];
    __shared__ PqFiniteFixHashView hash_view[kMultiQuery];
    __shared__ int token_valid[kMultiQuery];
    uint64_t phase_start_cycles = 0;

    if constexpr (kCollectPerHopTiming) {
        if (subtid == 0) phase_start_cycles = clock64();
    }

    if (subtid == 0) {
        token_valid[cid] = query_idx < params.query_capacity ? 1 : 0;
        if constexpr (!kInitialize) {
            if (token_valid[cid] == 0) {
                token->result = PqFiniteAdvanceResult::REJECT_QUERY;
            } else {
                control[cid] = params.d_contexts[query_idx];
                PqFiniteAdvanceResult reject = PqFiniteAdvanceResult::PENDING;
                if (control[cid].query_id != query_idx) {
                    reject = PqFiniteAdvanceResult::REJECT_QUERY;
                } else if (control[cid].context_epoch != context_epoch) {
                    reject = PqFiniteAdvanceResult::REJECT_EPOCH;
                } else if (control[cid].state != PqFiniteContextState::WAIT_IO) {
                    reject = PqFiniteAdvanceResult::REJECT_STATE;
                } else if (control[cid].io_generation != token->generation) {
                    reject = PqFiniteAdvanceResult::REJECT_GENERATION;
                } else if (control[cid].expected_row != token->row) {
                    reject = PqFiniteAdvanceResult::REJECT_ROW;
                } else if (control[cid].current_buffer != token->buffer) {
                    reject = PqFiniteAdvanceResult::REJECT_BUFFER;
                } else if (warp_kernel::load_bypass_l1(
                               &params.d_status_flag[static_cast<size_t>(query_idx) * 16]) != 1 ||
                           static_cast<uint32_t>(warp_kernel::load_bypass_l1(
                               &params.d_status_memory[static_cast<size_t>(query_idx) * 16 + 2])) !=
                               token->generation) {
                    reject = PqFiniteAdvanceResult::REJECT_STATUS;
                } else if (static_cast<uint32_t>(warp_kernel::load_bypass_l1(
                               &params.d_status_memory[static_cast<size_t>(query_idx) * 16])) !=
                               token->row) {
                    reject = PqFiniteAdvanceResult::REJECT_ROW;
                } else if (static_cast<uint32_t>(warp_kernel::load_bypass_l1(
                               &params.d_status_memory[static_cast<size_t>(query_idx) * 16 + 1]) & 1LL) !=
                               token->buffer) {
                    reject = PqFiniteAdvanceResult::REJECT_BUFFER;
                } else if (warp_kernel::load_bypass_l1(
                               &params.d_gdr_data[static_cast<size_t>(query_idx) *
                                                  kGdrMailboxStride]) == 0) {
                    reject = PqFiniteAdvanceResult::REJECT_MAILBOX;
                }
                if (reject != PqFiniteAdvanceResult::PENDING) {
                    token_valid[cid] = 0;
                    token->result = reject;
                }
            }
        }
    }
    __syncwarp();
    if (token_valid[cid] == 0) return;

    const uint32_t group_id = params.d_query_group_ids[query_idx];
    const uint32_t label_offset = params.d_label_offset[group_id];
    const uint64_t query_filter_bit = params.d_query_filter_bits[query_idx];
    const uint32_t active_range_count = go_fann::range_filter::count_active_ranges(
        params.d_query_range_modes, query_idx, params.range_count);
    const uint32_t prefetch_begin = params.d_prefetch_begin_by_group[group_id];
    const uint32_t prefetch_end = params.d_prefetch_end_by_group[group_id];
    const uint32_t prefetch_len = prefetch_end - prefetch_begin;
    pq_value_t (*pq_table)[kGraphPqDim][kNumPqCentroids] =
        reinterpret_cast<pq_value_t (*)[kGraphPqDim][kNumPqCentroids]>(params.d_pq_table);

    hash_view[cid].data = params.d_hashdata +
        static_cast<size_t>(query_idx) * kHashTableCapacity;
    for (int i = subtid; i < params.query_dim; i += step) {
        query_point[cid][i] = params.d_query[static_cast<size_t>(query_idx) * params.query_dim + i];
    }

    if constexpr (kInitialize) {
        computePQTable(params.d_pq_centroid,
                       params.d_query + static_cast<size_t>(query_idx) * params.query_dim,
                       &pq_table[query_idx][0][0],
                       params.query_pq_dim,
                       kNumPqCentroids,
                       params.query_dim,
                       subtid);
        dist_t start_distance_sum = 0.0f;
        const uint32_t begin_id = params.d_start_local_ids[query_idx];
        const uint32_t begin_sample = params.use_sample_id_payload != 0
            ? params.d_start_sample_ids[query_idx]
            : params.d_row_to_sample[label_offset + begin_id];
        for (int i = subtid; i < kGraphPqDim; i += step) {
            start_distance_sum +=
                pq_table[query_idx][i][params.d_data[static_cast<uint64_t>(begin_sample) *
                                                     params.query_pq_dim + i]];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            start_distance_sum += __shfl_xor_sync(
                warp_kernel::kFullMask, start_distance_sum, offset);
        }
        if (subtid == 0) {
            FixHash<uint32_t, kHashTableCapacity>* initializing_hash =
                new FixHash<uint32_t, kHashTableCapacity>(hash_view[cid].data);
            delete initializing_hash;
            control[cid] = PqFiniteContextControl{};
            control[cid].query_id = query_idx;
            control[cid].context_epoch = context_epoch;
            control[cid].state = PqFiniteContextState::RUNNING;
            control[cid].state_since_ns = warp_kernel::get_global_timer_ns();
            control[cid].stop_worst = -FLT_MAX;
            control[cid].current_buffer = 0;
            params.d_context_telemetry[query_idx] = PqFiniteContextTelemetry{};
            calculation_time_us[cid] = 0.0f;
            queue_time_us[cid] = 0.0f;
            resume_quantum_steps[cid] = 0;
            static_prefetch_steps[cid] = 0;
            PoolPair begin;
            begin.first = start_distance_sum;
            begin.second = pack_pool_payload(begin_id, false);
            int frontier_size = 1;
            smmh2::insert(q[cid], frontier_size, begin);
            control[cid].frontier_size = static_cast<uint32_t>(frontier_size);
            hash_view[cid].add(begin_id);
            PoolPair now;
            if (!pickup_next_unused_from_pool(q[cid],
                                              static_cast<int>(control[cid].frontier_size),
                                              now)) {
                control[cid].pool_exhausted = 1;
                now.first = FLT_MAX;
                now.second = pack_pool_payload(UINT32_MAX, true);
            }
            control[cid].now_distance = now.first;
            control[cid].now_payload = now.second;
            params.d_status_memory[static_cast<size_t>(query_idx) * 16] = begin_id;
            __threadfence_system();
            params.d_gdr_data[static_cast<size_t>(query_idx) * kGdrMailboxStride] = 0;
            maybe_trace_kernel_mailbox_publish(params.debug_mailbox,
                                               query_idx,
                                               active_query_idx,
                                               group_id,
                                               begin_id,
                                               0,
                                               0,
                                               0,
                                               0);
        }
    } else {
        const size_t frontier_base = static_cast<size_t>(query_idx) * (kQueueSize + 2);
        const size_t stop_base = static_cast<size_t>(query_idx) * (kTopk + 1);
        const size_t result_base = static_cast<size_t>(query_idx) *
            (static_cast<size_t>(params.result_width) + 1);
        for (int i = subtid; i < kQueueSize + 2; i += step) {
            q[cid][i] = params.d_frontier_state[frontier_base + i];
        }
        for (int i = subtid; i < kTopk + 1; i += step) {
            stop_topk[cid][i] = params.d_stop_state[stop_base + i];
        }
        for (int i = subtid; i < params.result_width + 1; i += step) {
            topk[i] = params.d_result_state[result_base + i];
        }
        if (subtid == 0) {
            PqFiniteContextTelemetry& telemetry = params.d_context_telemetry[query_idx];
            const uint64_t resume_ns = warp_kernel::get_global_timer_ns();
            const uint64_t parked_ns = resume_ns - control[cid].state_since_ns;
            const uint64_t ready_ns = token->ready_ns < parked_ns
                ? token->ready_ns
                : parked_ns;
            telemetry.wait_io_ns += parked_ns - ready_ns;
            telemetry.ready_ns += ready_ns;
            telemetry.io_time_us += static_cast<float>(parked_ns) / 1000.0f;
            ++telemetry.resume_count;
            control[cid].state = PqFiniteContextState::READY;
            params.d_contexts[query_idx].state = PqFiniteContextState::READY;
            __threadfence();
            control[cid].state = PqFiniteContextState::RUNNING;
            control[cid].state_since_ns = resume_ns;
            params.d_contexts[query_idx].state = PqFiniteContextState::RUNNING;
            params.d_status_flag[static_cast<size_t>(query_idx) * 16] = 0;
            __threadfence_system();
            calculation_time_us[cid] = 0.0f;
            queue_time_us[cid] = 0.0f;
            resume_quantum_steps[cid] = 0;
            static_prefetch_steps[cid] = 0;
            index_list_len[cid] = 0;
        }
    }
    __syncwarp();

    if constexpr (kCollectPerHopTiming) {
        if (subtid == 0) {
            const uint64_t restore_done_cycles = clock64();
            per_hop_timing->restore_cycles += restore_done_cycles - phase_start_cycles;
            phase_start_cycles = restore_done_cycles;
        }
    }

    bool consume_ready_payload = !kInitialize;
    while (true) {
        bool node_prefetched = false;
        int prefetch_vector_index = 0;
        if (!consume_ready_payload) {
            if (!(control[cid].step_num == 0 || control[cid].frontier_size > 1)) break;
            if (subtid == 0) {
                ++control[cid].step_num;
                if (control[cid].step_num != 1) {
                    PoolPair next;
                    if (!pickup_next_unused_from_pool(
                            q[cid], static_cast<int>(control[cid].frontier_size), next)) {
                        control[cid].pool_exhausted = 1;
                        next.first = FLT_MAX;
                        next.second = pack_pool_payload(UINT32_MAX, true);
                    } else if (control[cid].stop_heap_size ==
                                   static_cast<uint32_t>(params.stop_width) &&
                               control[cid].stop_worst <= next.first &&
                               params.early_stop_mode != kSearchEarlyStopOff) {
                        ++control[cid].finished;
                    }
                    control[cid].next_distance = next.first;
                    control[cid].next_payload = next.second;
                }
                index_list_len[cid] = 0;
            }
            __syncwarp();
            const uint32_t now_local_row = unpack_pool_local_row(control[cid].now_payload);
            if (control[cid].step_num <= kKernelPrefetchHops && prefetch_len > 0) {
                const uint32_t local_prefetch_index = warp_kernel::n_ary_scan_prefetch(
                    params.d_prefetch_index + prefetch_begin,
                    now_local_row,
                    prefetch_len);
                if (local_prefetch_index != UINT32_MAX) {
                    prefetch_vector_index = static_cast<int>(prefetch_begin + local_prefetch_index);
                    node_prefetched = true;
                }
            }
            if (!node_prefetched) {
                if constexpr (kCollectPerHopTiming) {
                    if (subtid == 0) {
                        const uint64_t payload_done_cycles = clock64();
                        per_hop_timing->payload_cycles +=
                            payload_done_cycles - phase_start_cycles;
                        phase_start_cycles = payload_done_cycles;
                    }
                }
                if (subtid == 0) {
                    PqFiniteContextTelemetry& telemetry =
                        params.d_context_telemetry[query_idx];
                    const uint64_t park_ns = warp_kernel::get_global_timer_ns();
                    telemetry.running_ns += park_ns - control[cid].state_since_ns;
                    telemetry.calculation_time_us += calculation_time_us[cid];
                    telemetry.queue_time_us += queue_time_us[cid];
                    telemetry.resumed_step_count += resume_quantum_steps[cid];
                    telemetry.static_prefetch_step_count += static_prefetch_steps[cid];
                    if (resume_quantum_steps[cid] > 1) {
                        ++telemetry.multi_step_resume_count;
                    }
                    if (resume_quantum_steps[cid] > telemetry.max_resume_quantum_steps) {
                        telemetry.max_resume_quantum_steps = resume_quantum_steps[cid];
                    }
                    ++telemetry.park_count;
                    control[cid].state = PqFiniteContextState::WAIT_IO;
                    control[cid].state_since_ns = park_ns;
                    ++control[cid].io_generation;
                    control[cid].expected_row = now_local_row;
                }
                __syncwarp();
                store_finite_mutable_state<kQueueSize>(
                    params, query_idx, subtid, q[cid], stop_topk[cid], topk);
                __syncwarp();
                if (subtid == 0) {
                    params.d_contexts[query_idx] = control[cid];
                    __threadfence_system();
                    params.d_gdr_data[static_cast<size_t>(query_idx) * kGdrMailboxStride] = 0;
                    params.d_status_memory[static_cast<size_t>(query_idx) * 16] = now_local_row;
                    params.d_status_memory[static_cast<size_t>(query_idx) * 16 + 1] =
                        control[cid].current_buffer;
                    params.d_status_memory[static_cast<size_t>(query_idx) * 16 + 2] =
                        control[cid].io_generation;
                    __threadfence_system();
                    params.d_status_flag[static_cast<size_t>(query_idx) * 16] = 1;
                    maybe_trace_kernel_mailbox_publish(params.debug_mailbox,
                                                       query_idx,
                                                       active_query_idx,
                                                       group_id,
                                                       now_local_row,
                                                       control[cid].current_buffer,
                                                       control[cid].io_generation,
                                                       control[cid].step_num,
                                                       1);
                    if (token != nullptr) token->result = PqFiniteAdvanceResult::PARKED;
                    if constexpr (kCollectPerHopTiming) {
                        per_hop_timing->writeback_cycles += clock64() - phase_start_cycles;
                        ++per_hop_timing->park_events;
                    }
                }
                return;
            }
        }

        const uint32_t now_local_row = unpack_pool_local_row(control[cid].now_payload);
        const size_t payload_slot = (static_cast<size_t>(query_idx) << 1) |
                                    control[cid].current_buffer;
        const size_t payload_base = payload_slot * params.mailbox_slot_words;
        const size_t exact_data_words = static_cast<size_t>(params.query_dim) / 4;
        uint8_t now_filter_result = 1;
        if (subtid == 0) {
            const uint32_t global_row = label_offset + now_local_row;
            const bool bit_match = params.use_implicit_filter_mask != 0
                ? query_filter_bit != 0ULL
                : (params.d_row_filter_mask[global_row] & query_filter_bit) != 0ULL;
            const uint32_t sample_idx = params.use_sample_id_payload != 0
                ? (node_prefetched
                       ? params.d_prefetch_node_array[prefetch_vector_index].sample_id
                       : params.d_neighbor_memory[payload_base + params.mailbox_sample_id_word])
                : params.d_row_to_sample[global_row];
            const uint32_t matched_active_dims =
                go_fann::range_filter::count_matching_active_ranges(
                    params.d_query_range_modes,
                    params.d_query_range_lower_bounds,
                    params.d_query_range_upper_bounds,
                    params.d_base_range_values,
                    query_idx,
                    sample_idx,
                    params.range_count);
            const bool exact_range_match = active_range_count == 0 ||
                                           matched_active_dims == active_range_count;
            const int64_t host_filter_result = node_prefetched
                ? 1
                : warp_kernel::load_bypass_l1(
                      &params.d_gdr_data[static_cast<size_t>(query_idx) * kGdrMailboxStride]);
            now_filter_result = host_filter_result == 1 && bit_match && exact_range_match
                ? static_cast<uint8_t>(1)
                : static_cast<uint8_t>(2);
            now_filter_result_shared[cid] = now_filter_result;
        }
        __syncwarp();
        now_filter_result = static_cast<uint8_t>(now_filter_result_shared[cid]);
        if (subtid == 0 && !kInitialize) {
            ++resume_quantum_steps[cid];
            if (node_prefetched) {
                ++static_prefetch_steps[cid];
            }
        }

        uint32_t node_degree = node_prefetched
            ? params.d_prefetch_node_array[prefetch_vector_index].neighbor_count
            : params.d_neighbor_memory[payload_base + params.mailbox_neighbor_count_word];
        node_degree = node_degree > static_cast<uint32_t>(FIXED_DEGREE)
            ? static_cast<uint32_t>(FIXED_DEGREE)
            : node_degree;
        dist_t current_pq_distance = control[cid].now_distance;
        if (now_filter_result == 1 && params.enable_true_distance_rerank != 0) {
            dist_t true_distance_sum = 0.0f;
            for (size_t word_idx = subtid; word_idx < exact_data_words; word_idx += step) {
                const uint32_t element = node_prefetched
                    ? params.d_prefetch_node_array[prefetch_vector_index].data[word_idx]
                    : params.d_neighbor_memory[payload_base + word_idx];
                const int base_dim = static_cast<int>(word_idx) * 4;
                const uint8_t a = element & 0xff;
                const uint8_t b = (element >> 8) & 0xff;
                const uint8_t c = (element >> 16) & 0xff;
                const uint8_t d = (element >> 24) & 0xff;
                true_distance_sum += (a - query_point[cid][base_dim]) *
                                     (a - query_point[cid][base_dim]);
                true_distance_sum += (b - query_point[cid][base_dim + 1]) *
                                     (b - query_point[cid][base_dim + 1]);
                true_distance_sum += (c - query_point[cid][base_dim + 2]) *
                                     (c - query_point[cid][base_dim + 2]);
                true_distance_sum += (d - query_point[cid][base_dim + 3]) *
                                     (d - query_point[cid][base_dim + 3]);
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                true_distance_sum += __shfl_xor_sync(
                    warp_kernel::kFullMask, true_distance_sum, offset);
            }
            if (subtid == 0) control[cid].now_distance = true_distance_sum;
        }
        __syncwarp();

        if (control[cid].pool_exhausted != 0 ||
            early_stop_reached(params.early_stop_mode,
                               static_cast<int>(control[cid].finished),
                               params.finish_limit,
                               params.queue_limit,
                               static_cast<int>(control[cid].expanded_count),
                               static_cast<int>(control[cid].pass_count),
                               params.stop_width) ||
            (params.max_steps > 0 &&
             control[cid].step_num >= static_cast<uint32_t>(params.max_steps))) {
            break;
        }

        const uint64_t queue_start = warp_kernel::get_global_timer_ns();
        if (subtid == 0) {
            ++control[cid].expanded_count;
            if (now_filter_result == 1) {
                ++control[cid].pass_count;
                const uint32_t current_sample = params.use_sample_id_payload != 0
                    ? (node_prefetched
                           ? params.d_prefetch_node_array[prefetch_vector_index].sample_id
                           : params.d_neighbor_memory[payload_base + params.mailbox_sample_id_word])
                    : params.d_row_to_sample[label_offset + now_local_row];
                ResultPair topk_entry;
                topk_entry.first = params.enable_true_distance_rerank != 0
                    ? control[cid].now_distance
                    : current_pq_distance;
                topk_entry.second = params.use_sample_id_payload != 0
                    ? current_sample
                    : now_local_row;
                StopPair stop_entry;
                stop_entry.first = topk_entry.first;
                stop_entry.second = pack_topk_payload(topk_entry.second, current_pq_distance);
                const uint32_t current_pq_bits = dist_to_bits(current_pq_distance);
                bool stop_topk_improved = false;
                int stop_heap_size = static_cast<int>(control[cid].stop_heap_size);
                if (stop_heap_size < params.stop_width) {
                    stop_topk[cid][stop_heap_size++] = stop_entry;
                    push_heap(stop_topk[cid], stop_topk[cid] + stop_heap_size);
                    stop_topk_improved = true;
                    if (stop_heap_size == 1 || current_pq_distance > control[cid].stop_worst) {
                        control[cid].stop_worst = current_pq_distance;
                        control[cid].stop_worst_bits = current_pq_bits;
                    }
                } else if (stop_entry.first < stop_topk[cid][0].first) {
                    stop_topk[cid][stop_heap_size++] = stop_entry;
                    push_heap(stop_topk[cid], stop_topk[cid] + stop_heap_size);
                    const StopPair removed = pop_heap(stop_topk[cid],
                                                      stop_topk[cid] + stop_heap_size);
                    --stop_heap_size;
                    stop_topk_improved = true;
                    const uint32_t removed_pq_bits = static_cast<uint32_t>(removed.second >> 32);
                    if (current_pq_distance > control[cid].stop_worst) {
                        control[cid].stop_worst = current_pq_distance;
                        control[cid].stop_worst_bits = current_pq_bits;
                    } else if (removed_pq_bits == control[cid].stop_worst_bits) {
                        control[cid].stop_worst = -FLT_MAX;
                        control[cid].stop_worst_bits = 0;
                        for (int i = 0; i < stop_heap_size; ++i) {
                            const dist_t pq_dist = unpack_topk_pq_dist(stop_topk[cid][i].second);
                            if (pq_dist > control[cid].stop_worst) {
                                control[cid].stop_worst = pq_dist;
                                control[cid].stop_worst_bits =
                                    static_cast<uint32_t>(stop_topk[cid][i].second >> 32);
                            }
                        }
                    }
                }
                control[cid].stop_heap_size = static_cast<uint32_t>(stop_heap_size);
                if (params.early_stop_mode == kSearchEarlyStopAdaptive && stop_topk_improved) {
                    control[cid].finished = 0;
                }
                int result_heap_size = static_cast<int>(control[cid].result_heap_size);
                if (result_heap_size < params.result_width) {
                    topk[result_heap_size++] = topk_entry;
                    push_heap(topk, topk + result_heap_size);
                } else if (topk_entry.first < topk[0].first) {
                    topk[result_heap_size++] = topk_entry;
                    push_heap(topk, topk + result_heap_size);
                    pop_heap(topk, topk + result_heap_size);
                    --result_heap_size;
                }
                control[cid].result_heap_size = static_cast<uint32_t>(result_heap_size);
            }
            for (uint32_t i = 0; i < node_degree; ++i) {
                const uint32_t local_row = node_prefetched
                    ? params.d_prefetch_node_array[prefetch_vector_index].neighbor[i]
                    : params.d_neighbor_memory[payload_base +
                                               params.mailbox_neighbor_words_offset + i];
                if (hash_view[cid].test(local_row)) continue;
                const int list_pos = index_list_len[cid]++;
                index_list[cid][list_pos] = local_row;
                if (params.use_sample_id_payload != 0) {
                    index_sample_list[cid][list_pos] = node_prefetched
                        ? params.d_prefetch_node_array[prefetch_vector_index].neighbor_sample[i]
                        : params.d_neighbor_memory[payload_base +
                                                   params.mailbox_neighbor_sample_words_offset + i];
                }
            }
        }
        __syncwarp();
        if (subtid == 0) {
            queue_time_us[cid] +=
                static_cast<float>(warp_kernel::get_global_timer_ns() - queue_start) / 1000.0f;
        }

        if (control[cid].pool_exhausted != 0 ||
            early_stop_reached(params.early_stop_mode,
                               static_cast<int>(control[cid].finished),
                               params.finish_limit,
                               params.queue_limit,
                               static_cast<int>(control[cid].expanded_count),
                               static_cast<int>(control[cid].pass_count),
                               params.stop_width) ||
            (params.max_steps > 0 &&
             control[cid].step_num >= static_cast<uint32_t>(params.max_steps))) {
            break;
        }

        __nanosleep(10 * 1000);
        const uint64_t calculation_start = warp_kernel::get_global_timer_ns();
        const int degree_limit = index_list_len[cid];
        for (int i = 0; i < degree_limit; ++i) {
            dist_t dist_acc = 0.0f;
            const uint32_t sample_idx = params.use_sample_id_payload != 0
                ? index_sample_list[cid][i]
                : params.d_row_to_sample[label_offset + index_list[cid][i]];
            for (int j = subtid; j < kGraphPqDim; j += step) {
                dist_acc += pq_table[query_idx][j][params.d_data[
                    static_cast<uint64_t>(sample_idx) * params.query_pq_dim + j]];
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                dist_acc += __shfl_xor_sync(warp_kernel::kFullMask, dist_acc, offset);
            }
            if (subtid == 0) dist_list[cid][i] = dist_acc;
        }
        __syncwarp();
        __nanosleep(10 * 1000);
        if (subtid == 0) {
            calculation_time_us[cid] +=
                static_cast<float>(warp_kernel::get_global_timer_ns() - calculation_start) /
                1000.0f;
        }
        const uint64_t insert_start = warp_kernel::get_global_timer_ns();
        if (subtid == 0) {
            int frontier_size = static_cast<int>(control[cid].frontier_size);
            for (int i = 0; i < degree_limit; ++i) {
                const uint32_t local_row = index_list[cid][i];
                const uint32_t global_row = label_offset + local_row;
                const bool bit_match = params.use_implicit_filter_mask != 0
                    ? query_filter_bit != 0ULL
                    : (params.d_row_filter_mask[global_row] & query_filter_bit) != 0ULL;
                const uint32_t sample_idx = params.use_sample_id_payload != 0
                    ? index_sample_list[cid][i]
                    : params.d_row_to_sample[global_row];
                const uint32_t matched_active_dims =
                    go_fann::range_filter::count_matching_active_ranges(
                        params.d_query_range_modes,
                        params.d_query_range_lower_bounds,
                        params.d_query_range_upper_bounds,
                        params.d_base_range_values,
                        query_idx,
                        sample_idx,
                        params.range_count);
                PoolPair candidate;
                candidate.first = go_fann::range_filter::apply_filter_bonuses(
                    dist_list[cid][i],
                    bit_match,
                    matched_active_dims,
                    params.bit_filter_bonus,
                    params.range_filter_bonus);
                candidate.second = pack_pool_payload(local_row, false);
                if (frontier_size >= params.queue_limit + 1 &&
                    q[cid][2].first < candidate.first) {
                    continue;
                }
                smmh2::insert(q[cid], frontier_size, candidate);
                hash_view[cid].add(local_row);
                if (frontier_size >= params.queue_limit + 2) {
                    smmh2::pop_max(q[cid], frontier_size);
                }
            }
            control[cid].frontier_size = static_cast<uint32_t>(frontier_size);
            if (control[cid].step_num == 1) {
                PoolPair next;
                if (!pickup_next_unused_from_pool(q[cid], frontier_size, next)) {
                    control[cid].pool_exhausted = 1;
                    next.first = FLT_MAX;
                    next.second = pack_pool_payload(UINT32_MAX, true);
                }
                control[cid].next_distance = next.first;
                control[cid].next_payload = next.second;
            }
            queue_time_us[cid] +=
                static_cast<float>(warp_kernel::get_global_timer_ns() - insert_start) / 1000.0f;
            control[cid].current_buffer ^= 1;
            control[cid].now_distance = control[cid].next_distance;
            control[cid].now_payload = control[cid].next_payload;
        }
        __syncwarp();
        consume_ready_payload = false;
    }

    if constexpr (kCollectPerHopTiming) {
        if (subtid == 0) {
            const uint64_t payload_done_cycles = clock64();
            per_hop_timing->payload_cycles += payload_done_cycles - phase_start_cycles;
            phase_start_cycles = payload_done_cycles;
        }
    }
    if (subtid == 0) {
        PqFiniteContextTelemetry& telemetry = params.d_context_telemetry[query_idx];
        const uint64_t done_ns = warp_kernel::get_global_timer_ns();
        telemetry.running_ns += done_ns - control[cid].state_since_ns;
        telemetry.calculation_time_us += calculation_time_us[cid];
        telemetry.queue_time_us += queue_time_us[cid];
        telemetry.resumed_step_count += resume_quantum_steps[cid];
        telemetry.static_prefetch_step_count += static_prefetch_steps[cid];
        if (resume_quantum_steps[cid] > 1) {
            ++telemetry.multi_step_resume_count;
        }
        if (resume_quantum_steps[cid] > telemetry.max_resume_quantum_steps) {
            telemetry.max_resume_quantum_steps = resume_quantum_steps[cid];
        }
        ++telemetry.terminal_count;
        control[cid].state = PqFiniteContextState::DONE;
    }
    __syncwarp();
    store_finite_mutable_state<kQueueSize>(
        params, query_idx, subtid, q[cid], stop_topk[cid], topk);
    __syncwarp();
    if (subtid == 0) {
        params.d_contexts[query_idx] = control[cid];
        __threadfence_system();
        params.d_status_flag[static_cast<size_t>(query_idx) * 16] = 2;
        for (int i = 0; i < params.result_width; ++i) {
            params.d_result[static_cast<size_t>(query_idx) * params.result_width + i] = UINT32_MAX;
            params.d_result_dist[static_cast<size_t>(query_idx) * params.result_width + i] = FLT_MAX;
        }
        const int valid_topk = static_cast<int>(control[cid].result_heap_size) < params.result_width
            ? static_cast<int>(control[cid].result_heap_size)
            : params.result_width;
        int heap_size = static_cast<int>(control[cid].result_heap_size);
        for (int i = 0; i < valid_topk; ++i) {
            const ResultPair result = pop_heap(topk, topk + heap_size - i);
            const int output_slot = valid_topk - 1 - i;
            if (result.second == UINT32_MAX) continue;
            params.d_result[static_cast<size_t>(query_idx) * params.result_width + output_slot] =
                params.use_sample_id_payload != 0
                    ? result.second
                    : params.d_row_to_sample[label_offset + result.second];
            params.d_result_dist[static_cast<size_t>(query_idx) * params.result_width + output_slot] =
                result.first;
        }
        const PqFiniteContextTelemetry& telemetry = params.d_context_telemetry[query_idx];
        const uint64_t parked_ns = telemetry.wait_io_ns + telemetry.ready_ns;
        params.time_per_block[static_cast<size_t>(query_idx) * 6] =
            static_cast<float>(parked_ns + telemetry.running_ns) / 1000.0f;
        params.time_per_block[static_cast<size_t>(query_idx) * 6 + 1] =
            static_cast<float>(parked_ns) / 1000.0f;
        params.time_per_block[static_cast<size_t>(query_idx) * 6 + 2] =
            telemetry.calculation_time_us;
        params.time_per_block[static_cast<size_t>(query_idx) * 6 + 3] =
            telemetry.queue_time_us;
        params.time_per_block[static_cast<size_t>(query_idx) * 6 + 4] =
            static_cast<float>(control[cid].step_num);
        params.time_per_block[static_cast<size_t>(query_idx) * 6 + 5] =
            static_cast<float>(telemetry.ready_ns) / 1000.0f;
        __threadfence_system();
        params.d_status_flag[static_cast<size_t>(query_idx) * 16] = 3;
        if (token != nullptr) token->result = PqFiniteAdvanceResult::DONE;
        if constexpr (kCollectPerHopTiming) {
            per_hop_timing->writeback_cycles += clock64() - phase_start_cycles;
            ++per_hop_timing->terminal_events;
        }
    }
}

template <int kQueueSize, int kGraphPqDim>
__global__ GO_FANN_PQ_FINITE_LAUNCH_BOUNDS(kQueueSize) void
finite_pq_init_kernel(PqFiniteKernelParams params,
                      const uint32_t* d_query_ids,
                      uint32_t query_count,
                      uint32_t context_epoch)
{
    extern __shared__ ResultPair topk_workspaces[];
    const uint32_t cid = static_cast<uint32_t>(threadIdx.x / kCriticalStep);
    const uint32_t active_query_idx = static_cast<uint32_t>(blockIdx.x * kMultiQuery) + cid;
    if (active_query_idx >= query_count) return;
    const uint32_t query_idx = d_query_ids != nullptr
        ? d_query_ids[active_query_idx]
        : active_query_idx;
    ResultPair* topk = topk_workspaces +
        static_cast<size_t>(cid) * (static_cast<size_t>(params.result_width) + 1);
    run_pq_finite_context<kQueueSize, kGraphPqDim, true, false>(
        params, active_query_idx, query_idx, context_epoch, nullptr, topk, nullptr);
}

template <int kQueueSize, int kGraphPqDim>
__global__ GO_FANN_PQ_FINITE_LAUNCH_BOUNDS(kQueueSize) void
finite_pq_resume_kernel(PqFiniteKernelParams params,
                        PqFiniteReadyToken* tokens,
                        uint32_t token_count)
{
    extern __shared__ ResultPair topk_workspaces[];
    const uint32_t cid = static_cast<uint32_t>(threadIdx.x / kCriticalStep);
    const uint32_t active_token_idx = static_cast<uint32_t>(blockIdx.x * kMultiQuery) + cid;
    if (active_token_idx >= token_count) return;
    PqFiniteReadyToken* token = tokens + active_token_idx;
    ResultPair* topk = topk_workspaces +
        static_cast<size_t>(cid) * (static_cast<size_t>(params.result_width) + 1);
    run_pq_finite_context<kQueueSize, kGraphPqDim, false, false>(
        params,
        active_token_idx,
        token->query_id,
        token->context_epoch,
        token,
        topk,
        nullptr);
}

template <int kQueueSize, int kGraphPqDim>
__global__ GO_FANN_PQ_FINITE_LAUNCH_BOUNDS(kQueueSize) void
persistent_finite_pq_worker_kernel(
    PqFiniteKernelParams params,
    volatile PqFinitePersistentRequestControl* requests,
    volatile PqFinitePersistentCompletionControl* completions,
    unsigned long long* completion_tail,
    volatile PqFinitePersistentCompletionNotification* completion_notifications,
    uint32_t completion_ring_capacity)
{
    extern __shared__ ResultPair topk_workspaces[];
    const uint32_t cid = static_cast<uint32_t>(threadIdx.x / kCriticalStep);
    const uint32_t subtid = static_cast<uint32_t>(threadIdx.x % kCriticalStep);
    const uint32_t slot_idx = static_cast<uint32_t>(blockIdx.x * kMultiQuery) + cid;
    volatile PqFinitePersistentRequestControl* request = requests + slot_idx;
    volatile PqFinitePersistentCompletionControl* completion = completions + slot_idx;
    __shared__ uint32_t current_seq[kMultiQuery];
    __shared__ uint32_t should_stop[kMultiQuery];
    __shared__ PqFiniteReadyToken tokens[kMultiQuery];
    uint32_t last_seq = 0;

    while (true) {
        if (subtid == 0) {
            should_stop[cid] = 0;
            uint32_t seq = last_seq;
            while (seq == last_seq) {
                if (load_bypass_l1_u32(&request->stop) != 0) {
                    should_stop[cid] = 1;
                    break;
                }
                seq = load_bypass_l1_u32(&request->ready_seq);
                if (seq == last_seq) __nanosleep(256);
            }
            current_seq[cid] = seq;
            if (should_stop[cid] == 0 && seq != kPersistentLatencyStopSeq) {
                tokens[cid].query_id = load_bypass_l1_u32(&request->query_id);
                tokens[cid].context_epoch = load_bypass_l1_u32(&request->context_epoch);
                tokens[cid].generation = load_bypass_l1_u32(&request->generation);
                tokens[cid].row = load_bypass_l1_u32(&request->row);
                tokens[cid].buffer = load_bypass_l1_u32(&request->buffer);
                tokens[cid].ready_ns = load_bypass_l1_u64(&request->ready_ns);
                tokens[cid].result = PqFiniteAdvanceResult::PENDING;
            }
        }
        __syncwarp();
        if (should_stop[cid] != 0 || current_seq[cid] == kPersistentLatencyStopSeq) {
            return;
        }

        const uint64_t gpu_start_ns = warp_kernel::get_global_timer_ns();
        ResultPair* topk = topk_workspaces +
            static_cast<size_t>(cid) * (static_cast<size_t>(params.result_width) + 1);
        run_pq_finite_context<kQueueSize, kGraphPqDim, false, false>(
            params,
            slot_idx,
            tokens[cid].query_id,
            tokens[cid].context_epoch,
            &tokens[cid],
            topk,
            nullptr);
        __syncwarp();
        if (subtid == 0) {
            completion->result = tokens[cid].result;
            completion->query_id = tokens[cid].query_id;
            completion->generation = tokens[cid].generation;
            completion->gpu_start_ns = gpu_start_ns;
            completion->gpu_stop_ns = warp_kernel::get_global_timer_ns();
            __threadfence_system();
            completion->done_seq = current_seq[cid];
            last_seq = current_seq[cid];

            const unsigned long long completion_ticket =
                atomicAdd(completion_tail, 1ULL);
            volatile PqFinitePersistentCompletionNotification* notification =
                completion_notifications +
                (completion_ticket % completion_ring_capacity);
            notification->slot_idx = slot_idx;
            notification->done_seq = current_seq[cid];
            __threadfence_system();
            notification->publish_sequence = completion_ticket + 1;
        }
        __syncwarp();
    }
}

template <int kQueueSize, int kGraphPqDim>
__global__ GO_FANN_PQ_FINITE_LAUNCH_BOUNDS(kQueueSize) void
persistent_finite_pq_ready_ring_kernel(
    PqFiniteKernelParams params,
    volatile PqFiniteReadyRingControl* control,
    volatile PqFiniteReadyRingEntry* entries,
    uint32_t ring_capacity,
    unsigned long long* consumer_head,
    volatile unsigned long long* consumed_sequences,
    volatile PqFiniteReadyRingStatus* status
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
    , PqFinitePerHopTiming* timing_output
#endif
    )
{
    extern __shared__ ResultPair topk_workspaces[];
    const uint32_t cid = static_cast<uint32_t>(threadIdx.x / kCriticalStep);
    const uint32_t subtid = static_cast<uint32_t>(threadIdx.x % kCriticalStep);
    const uint32_t slot_idx = static_cast<uint32_t>(blockIdx.x * kMultiQuery) + cid;
    __shared__ uint32_t should_stop[kMultiQuery];
    __shared__ uint64_t current_ticket[kMultiQuery];
    __shared__ PqFiniteReadyToken tokens[kMultiQuery];
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
    __shared__ PqFinitePerHopTiming per_hop_timing[kMultiQuery];
    if (subtid == 0) per_hop_timing[cid] = PqFinitePerHopTiming{};
    __syncwarp();
#endif

    while (true) {
        if (subtid == 0) {
            should_stop[cid] = 0;
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
            const uint64_t ring_wait_start_cycles = clock64();
#endif
            const unsigned long long ticket = atomicAdd(consumer_head, 1ULL);
            current_ticket[cid] = ticket;
            const uint32_t entry_idx = static_cast<uint32_t>(ticket % ring_capacity);
            volatile PqFiniteReadyRingEntry* entry =
                entries + entry_idx;
            const uint64_t expected_publish_sequence = ticket + 1;
            uint64_t publish_sequence = 0;
            while (publish_sequence != expected_publish_sequence) {
                if (load_bypass_l1_u32(&control->stop) != 0) {
                    should_stop[cid] = 1;
                    break;
                }
                publish_sequence = load_bypass_l1_u64(&entry->publish_sequence);
                if (publish_sequence != expected_publish_sequence) __nanosleep(256);
            }
            if (should_stop[cid] == 0) {
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
                const uint64_t publish_observed_cycles = clock64();
                per_hop_timing[cid].ring_wait_cycles +=
                    publish_observed_cycles - ring_wait_start_cycles;
#endif
                tokens[cid].query_id = load_bypass_l1_u32(&entry->token.query_id);
                tokens[cid].context_epoch =
                    load_bypass_l1_u32(&entry->token.context_epoch);
                tokens[cid].generation = load_bypass_l1_u32(&entry->token.generation);
                tokens[cid].row = load_bypass_l1_u32(&entry->token.row);
                tokens[cid].buffer = load_bypass_l1_u32(&entry->token.buffer);
                tokens[cid].ready_ns = 0;
                tokens[cid].result = PqFiniteAdvanceResult::PENDING;
                __threadfence_system();
                consumed_sequences[entry_idx] = ticket + 1;
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
                per_hop_timing[cid].ring_metadata_cycles +=
                    clock64() - publish_observed_cycles;
                ++per_hop_timing[cid].resume_events;
#endif
            }
        }
        __syncwarp();
        if (should_stop[cid] != 0) {
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
            if (subtid == 0) timing_output[slot_idx] = per_hop_timing[cid];
#endif
            return;
        }

        ResultPair* topk = topk_workspaces +
            static_cast<size_t>(cid) * (static_cast<size_t>(params.result_width) + 1);
        run_pq_finite_context<kQueueSize, kGraphPqDim, false, kFinitePerHopTiming>(
            params,
            slot_idx,
            tokens[cid].query_id,
            tokens[cid].context_epoch,
            &tokens[cid],
            topk,
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
            &per_hop_timing[cid]
#else
            nullptr
#endif
            );
        __syncwarp();
        if (subtid == 0 && tokens[cid].result != PqFiniteAdvanceResult::PARKED &&
            tokens[cid].result != PqFiniteAdvanceResult::DONE) {
            if (atomicCAS(const_cast<uint32_t*>(&control->error_claimed), 0U, 1U) == 0U) {
                status->error_query_id = tokens[cid].query_id;
                status->error_generation = tokens[cid].generation;
                status->error_ticket = current_ticket[cid];
                __threadfence_system();
                status->error_result = static_cast<uint32_t>(tokens[cid].result);
            }
            should_stop[cid] = 1;
        }
        __syncwarp();
        if (should_stop[cid] != 0) {
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
            if (subtid == 0) timing_output[slot_idx] = per_hop_timing[cid];
#endif
            return;
        }
    }
}

__device__ __forceinline__ bool resident_pq_labels_match(
    const uint64_t* data_label_indptr,
    const uint32_t* data_label_indices,
    const uint64_t* query_label_indptr,
    const uint32_t* query_label_indices,
    const uint32_t* query_filter_labels,
    uint32_t query_idx,
    uint32_t sample_idx)
{
    const uint64_t sample_begin = data_label_indptr[sample_idx];
    const uint64_t sample_end = data_label_indptr[static_cast<size_t>(sample_idx) + 1];
    const uint64_t query_begin = query_label_indptr[query_idx];
    const uint64_t query_end = query_label_indptr[static_cast<size_t>(query_idx) + 1];
    const uint32_t primary_label = query_filter_labels[query_idx];
    for (uint64_t query_pos = query_begin; query_pos < query_end; ++query_pos) {
        const uint32_t label = query_label_indices[query_pos];
        if (label == primary_label) continue;
        uint64_t lo = sample_begin;
        uint64_t hi = sample_end;
        while (lo < hi) {
            const uint64_t mid = lo + ((hi - lo) >> 1);
            const uint32_t value = data_label_indices[mid];
            if (value < label) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo == sample_end || data_label_indices[lo] != label) return false;
    }
    return true;
}

template <bool kDeviceResident>
__device__ __forceinline__ void sync_native_query()
{
    constexpr int kQueryWarps = kDeviceResident
        ? kNativeWarpsPerQuery
        : kNativeCoupledWarpsPerQuery;
    if constexpr (kQueryWarps == 1) {
        __syncwarp();
    } else {
        __syncthreads();
    }
}

template <int kQueueSize,
          int kGraphPqDim,
          bool kCollectDetailedTiming,
          bool kUseRequestRing,
          bool kDeviceResident,
          bool kUseCooperativePickup,
          int kProfileMode,
          bool kPredicateBias = false>
__device__ void run_pq_search_one_block(const pq_idx_t* d_data,
                                        value_t* d_query,
                                        idx_t* d_result,
                                        float* d_result_dist,
                                        const pq_value_t* d_pq_centroid,
                                        pq_value_t* d_pq_table,
                                        dist_t* d_datadist_list,
                                        uint32_t* d_hashdata,
                                        uint32_t active_query_idx,
                                        uint32_t query_idx,
                                        float* time_per_block,
                                        PqNativeProfileRecord* native_profile,
                                        PqNativeProfileRecord* native_profile_output,
                                        uint32_t* d_neighbor_memory,
                                        const uint32_t* d_resident_graph,
                                        uint32_t resident_graph_degree,
                                        const uint64_t* d_resident_data_label_indptr,
                                        const uint32_t* d_resident_data_label_indices,
                                        const uint64_t* d_resident_query_label_indptr,
                                        const uint32_t* d_resident_query_label_indices,
                                        const uint32_t* d_resident_query_filter_labels,
                                        int resident_full_label_check,
                                        volatile int64_t* d_status_memory,
                                        volatile int64_t* d_gdr_data,
                                        volatile int64_t* d_status_flag,
                                        HostRequestRingEntry* d_request_ring_entries,
                                        const volatile uint64_t* d_request_ring_consumed_sequences,
                                        unsigned long long* d_request_ring_tails,
                                        uint32_t request_ring_capacity,
                                        const Node* d_prefetch_node_array,
                                        const uint32_t* d_prefetch_index,
                                        const uint32_t* d_prefetch_begin_by_group,
                                        const uint32_t* d_prefetch_end_by_group,
                                        const uint32_t* d_query_group_ids,
                                        const uint32_t* d_start_local_ids,
                                        const uint32_t* d_start_sample_ids,
                                        const uint32_t* d_row_to_sample,
                                        const uint32_t* d_label_offset,
                                        const uint64_t* d_row_filter_mask,
                                        const uint64_t* d_query_filter_bits,
                                        const uint8_t* d_query_range_modes,
                                        const float* d_query_range_lower_bounds,
                                        const float* d_query_range_upper_bounds,
                                        const float* d_base_range_values,
                                        uint32_t range_count,
                                        int query_dim,
                                        int query_pq_dim,
                                        int use_adaptive_wait,
                                        int buffer_mode,
                                        int queue_limit,
                                        int finish_limit,
                                        int early_stop_mode,
                                        int max_steps,
                                        int stop_width,
                                        int result_width,
                                        std::conditional_t<kPredicateBias, gpu_predicate::BiasedPair, PoolPair>* frontier,
                                        ResultPair* topk,
                                        uint32_t mailbox_vector_words,
                                        uint32_t mailbox_sample_id_word,
                                        uint32_t mailbox_neighbor_count_word,
                                        uint32_t mailbox_neighbor_words_offset,
                                        uint32_t mailbox_neighbor_sample_words_offset,
                                        uint32_t mailbox_slot_words,
                                        int use_sample_id_payload,
                                        int use_implicit_filter_mask,
                                        int enable_true_distance_rerank,
                                        float bit_filter_bonus,
                                        float range_filter_bonus,
                                        int debug_mailbox,
                                        gpu_predicate::View predicate = {},
                                        expansion_trace::View expansion = {})
{
    using FrontierPair = std::conditional_t<kPredicateBias, gpu_predicate::BiasedPair, PoolPair>;
    constexpr int step = warp_kernel::kWarpThreads;
    constexpr int kQueryWarps = kDeviceResident
        ? kNativeWarpsPerQuery
        : kNativeCoupledWarpsPerQuery;
    int tid = threadIdx.x;
    const int warp_in_block = tid / kCriticalStep;
    const int cid = warp_in_block / kQueryWarps;
    const int query_warp = warp_in_block % kQueryWarps;
    int subtid = tid % kCriticalStep;
    const bool primary_query_warp = query_warp == 0;
    const bool query_leader = primary_query_warp && subtid == 0;
    uint32_t request_ring_idx = 0;
    uint64_t request_publish_ns = 0;
    if constexpr (kUseRequestRing) {
        request_ring_idx = active_query_idx / request_ring_capacity;
    }
    const uint32_t group_id = d_query_group_ids[query_idx];
    const uint32_t label_offset = d_label_offset[group_id];
    const uint64_t query_filter_bit = d_query_filter_bits[query_idx];
    const uint32_t active_range_count = go_fann::range_filter::count_active_ranges(d_query_range_modes,
                                                                                          query_idx,
                                                                                          range_count);
    const uint32_t begin_id = d_start_local_ids[query_idx];
    const uint32_t prefetch_begin = d_prefetch_begin_by_group[group_id];
    const uint32_t prefetch_end = d_prefetch_end_by_group[group_id];
    const uint32_t prefetch_len = prefetch_end - prefetch_begin;
    float time2 = 0.0f;
    float time3 = 0.0f;
    float time4 = 0.0f;
    const uint64_t start1 = warp_kernel::get_global_timer_ns();

    __shared__ warp_kernel::KernelPair<dist_t, idx_t> stop_topk[kMultiQuery][kTopk + 1];
    __shared__ dist_t dist_list[kMultiQuery][FIXED_DEGREE * kMultiprobe];

    __shared__ PqLane0SearchStateT<FrontierPair> lane0_state[kMultiQuery];
    __shared__ uint32_t predicate_pass_masks[kMultiQuery][(FIXED_DEGREE * kMultiprobe + 31) / 32];

    uint32_t* native_hash_data =
        &d_hashdata[static_cast<size_t>(query_idx) * kHashTableCapacity];
    if (primary_query_warp) {
        native_visited_hash::clear<kHashTableCapacity>(native_hash_data, subtid);
    }
    sync_native_query<kDeviceResident>();

    if (query_leader) {
        if constexpr (kProfileMode > 0) {
            *native_profile = PqNativeProfileRecord{};
        }
        if constexpr (!kDeviceResident) {
            if (d_status_memory[query_idx * 16] != -2) {
                printf("query:%u,status error\n", query_idx);
            }
            d_status_memory[query_idx * 16] = begin_id;
            __threadfence_system();
            d_gdr_data[query_idx * kGdrMailboxStride] = 0;
            maybe_trace_kernel_mailbox_publish(debug_mailbox,
                                               query_idx,
                                               active_query_idx,
                                               group_id,
                                               begin_id,
                                               0,
                                               0,
                                               0,
                                               0);
        }
    }

    sync_native_query<kDeviceResident>();

    __shared__ int heap_size[kMultiQuery];
    __shared__ int finished[kMultiQuery];
    __shared__ int pool_exhausted[kMultiQuery];
    __shared__ int expanded_count[kMultiQuery];
    __shared__ int pass_count[kMultiQuery];
    __shared__ idx_t index_list[kMultiQuery][FIXED_DEGREE * kMultiprobe];
    __shared__ uint32_t index_sample_list[kMultiQuery][FIXED_DEGREE * kMultiprobe];
    __shared__ int index_list_len[kMultiQuery];
    __shared__ value_t query_point[kMultiQuery][kGraphMaxDim];
    __shared__ uint32_t current_buffer_idx[kMultiQuery];
    __shared__ uint32_t next_buffer_idx[kMultiQuery];
    __shared__ uint32_t request_generation[kMultiQuery];
    __shared__ int wait_sleep_us[kMultiQuery];
#if GO_FANN_PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE
    __shared__ int frontier_worst_index[kMultiQuery];
    __shared__ dist_t frontier_worst_distance[kMultiQuery];
#endif
    pq_value_t (*pq_table)[kGraphPqDim][kNumPqCentroids] =
        (pq_value_t (*)[kGraphPqDim][kNumPqCentroids])d_pq_table;
    const PqBufferMode mode = static_cast<PqBufferMode>(buffer_mode);

    sync_native_query<kDeviceResident>();

    if (primary_query_warp) {
        computePQTable(d_pq_centroid,
                       d_query + query_idx * query_dim,
                       &pq_table[query_idx][0][0],
                       query_pq_dim,
                       kNumPqCentroids,
                       query_dim,
                       subtid);
    }
    sync_native_query<kDeviceResident>();

    dist_t start_distance = 0.0f;
    int j = cid;
    dist_t start_distance_sum = 0.0f;
    if (primary_query_warp) {
        for (int i = subtid; i < query_dim; i += step) {
            query_point[j][i] = d_query[query_idx * query_dim + i];
        }
    }
    const uint32_t begin_sample = use_sample_id_payload != 0
        ? d_start_sample_ids[query_idx]
        : d_row_to_sample[label_offset + begin_id];
    if (primary_query_warp) {
        for (int i = subtid; i < kGraphPqDim; i += step) {
            start_distance_sum +=
                pq_table[query_idx][i][d_data[1ULL * begin_sample * query_pq_dim + i]];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            start_distance_sum +=
                __shfl_xor_sync(warp_kernel::kFullMask, start_distance_sum, offset);
        }
        if (subtid == 0) {
            start_distance = start_distance_sum;
        }
    }
    sync_native_query<kDeviceResident>();

    float predicate_penalty = 0.0f;
    if constexpr (kPredicateBias) {
        const double wide = double(predicate.alpha) * fmax(double(fabsf(start_distance)), 1e-6);
        predicate_penalty = isfinite(wide) ? float(fmin(wide, double(FLT_MAX))) : 0.0f;
    }
    if (query_leader) {
        heap_size[cid] = 1;
        lane0_state[cid].stop_topk_heap_size = 0;
        lane0_state[cid].stop_topk_pq_worst = -FLT_MAX;
        lane0_state[cid].topk_heap_size = 0;
        lane0_state[cid].now_filter_result = 1;
        finished[cid] = false;
        pool_exhausted[cid] = false;
        expanded_count[cid] = 0;
        pass_count[cid] = 0;
        current_buffer_idx[cid] = 0;
        next_buffer_idx[cid] = pq_buffer_mode_uses_single_slot(mode) ? 0 : 1;
        request_generation[cid] = 0;
        wait_sleep_us[cid] = kInitialWaitSleepUs;
        FrontierPair kp;
        kp.first = start_distance;
        if constexpr (kPredicateBias) kp.base = start_distance;
        kp.second = pack_pool_payload(begin_id, false);
#if GO_FANN_PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE || GO_FANN_PQ_NATIVE_MAX_HEAP_FRONTIER
#if GO_FANN_PQ_NATIVE_MAX_HEAP_FRONTIER
        append_native_max_heap(frontier, heap_size[cid], kp);
#else
        frontier[heap_size[cid]++] = kp;
#endif
#else
        smmh2::insert(frontier, heap_size[cid], kp);
#endif
        profiled_native_hash_add<kProfileMode>(native_hash_data,
                                               begin_id,
                                               native_profile);
    }

    sync_native_query<kDeviceResident>();

#if GO_FANN_PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE
    if (heap_size[cid] >= queue_limit + 1) {
        refresh_native_frontier_worst(frontier,
                                      heap_size[cid],
                                      subtid,
                                      &frontier_worst_index[cid],
                                      &frontier_worst_distance[cid]);
    }
#endif

    if (primary_query_warp) {
        const bool found_next =
            profiled_pickup_next_unused_from_native_frontier<kUseCooperativePickup,
                                                             kProfileMode>(
                frontier,
                heap_size[cid],
                lane0_state[cid].now,
                subtid,
                native_profile);
        if (query_leader && !found_next) {
            pool_exhausted[cid] = true;
            lane0_state[cid].now.first = FLT_MAX;
            lane0_state[cid].now.second = pack_pool_payload(UINT32_MAX, true);
        }
    }
    sync_native_query<kDeviceResident>();

    int step_num = 0;
    while (step_num == 0 || heap_size[cid] > 1) {
        const uint32_t now_local_row = unpack_pool_local_row(lane0_state[cid].now.second);
        step_num++;
        if (query_leader) {
            index_list_len[cid] = 0;
        }
        sync_native_query<kDeviceResident>();

        if (step_num != 1) {
            if (primary_query_warp) {
                const bool found =
                    profiled_pickup_next_unused_from_native_frontier<kUseCooperativePickup,
                                                                     kProfileMode>(
                        frontier,
                        heap_size[cid],
                        lane0_state[cid].next,
                        subtid,
                        native_profile);
                if (query_leader && !found) {
                    pool_exhausted[cid] = true;
                    lane0_state[cid].next.first = FLT_MAX;
                    lane0_state[cid].next.second = pack_pool_payload(UINT32_MAX, true);
                } else if (query_leader && lane0_state[cid].stop_topk_heap_size == stop_width) {
                    float pending_score = lane0_state[cid].next.first;
                    if (predicate.pending_base_stop && gpu_predicate::enabled(predicate, query_idx)) {
                        pending_score = gpu_predicate::minimum_pending_base(
                            frontier, heap_size[cid], lane0_state[cid].now,
                            lane0_state[cid].next, true, found,
                            [] __device__ (uint32_t payload) { return !pool_payload_used(payload); });
                    }
                    if (early_stop_mode != kSearchEarlyStopOff &&
                        lane0_state[cid].stop_topk_pq_worst <= pending_score) ++finished[cid];
                }
            }
            sync_native_query<kDeviceResident>();
        }

        int prefetch_vector_index = 0;
        bool node_prefetched = false;
        if (step_num <= kKernelPrefetchHops && prefetch_len > 0 &&
            (!predicate.route_prefetch_disabled || !predicate.route_prefetch_disabled[query_idx])) {
            uint32_t local_prefetch_index = warp_kernel::n_ary_scan_prefetch(
                d_prefetch_index + prefetch_begin,
                now_local_row,
                prefetch_len);
            if (local_prefetch_index != UINT32_MAX) {
                prefetch_vector_index = static_cast<int>(prefetch_begin + local_prefetch_index);
                node_prefetched = true;
            }
        }

        uint64_t io_start = 0;
        if constexpr (kCollectDetailedTiming) {
            io_start = warp_kernel::get_global_timer_ns();
        }
        if constexpr (!kDeviceResident) {
            if (!node_prefetched) {
                const int sleep_us = wait_sleep_us[cid];
                int wait_loops = 0;
                const bool issue_request_for_now =
                    !pq_buffer_mode_overlaps(mode) || step_num <= kKernelPrefetchHops;

                if (issue_request_for_now) {
                    if (query_leader) {
                        if (pq_buffer_mode_uses_single_slot(mode)) {
                            current_buffer_idx[cid] = 0;
                            next_buffer_idx[cid] = 0;
                        }
                        ++request_generation[cid];
                        d_gdr_data[query_idx * kGdrMailboxStride] = 0;
                        d_status_memory[query_idx * 16] = now_local_row;
                        d_status_memory[query_idx * 16 + 1] = current_buffer_idx[cid];
                        d_status_memory[query_idx * 16 + 2] = request_generation[cid];
                        __threadfence_system();
                        d_status_flag[query_idx * 16] = 1;
                        if constexpr (kUseRequestRing) {
                            publish_profiled_pq_host_request<kRequestPublishTiming>(
                                d_request_ring_entries,
                                d_request_ring_consumed_sequences,
                                d_request_ring_tails,
                                request_ring_idx,
                                request_ring_capacity,
                                query_idx,
                                HostRequestKind::Demand,
                                request_publish_ns);
                        }
                        maybe_trace_kernel_mailbox_publish(debug_mailbox,
                                                           query_idx,
                                                           active_query_idx,
                                                           group_id,
                                                           now_local_row,
                                                           current_buffer_idx[cid],
                                                           request_generation[cid],
                                                           step_num,
                                                           1);
                    }
                    sync_native_query<kDeviceResident>();
                }
                if (use_adaptive_wait) {
                    while (warp_kernel::load_bypass_l1(
                               &d_gdr_data[query_idx * kGdrMailboxStride]) == 0) {
                        warp_kernel::sleep_cuda_us(static_cast<float>(sleep_us));
                        __threadfence_system();
                        ++wait_loops;
                    }
                } else {
                    while (warp_kernel::load_bypass_l1(
                               &d_gdr_data[query_idx * kGdrMailboxStride]) == 0) {
                        warp_kernel::sleep_cuda_us(100);
                        __threadfence_system();
                    }
                }
                if (query_leader) {
                    wait_sleep_us[cid] = use_adaptive_wait
                        ? warp_kernel::next_wait_sleep_us(sleep_us,
                                                          wait_loops,
                                                          kMinWaitSleepUs,
                                                          kMaxWaitSleepUs)
                        : kInitialWaitSleepUs;
                }
                sync_native_query<kDeviceResident>();
            }
        }
        if constexpr (kCollectDetailedTiming) {
            time2 += static_cast<float>(warp_kernel::get_global_timer_ns() - io_start) / 1000.0f;
        }

        const size_t payload_slot = pq_buffer_mode_uses_single_slot(mode)
            ? static_cast<size_t>(query_idx)
            : ((static_cast<size_t>(query_idx) << 1) | current_buffer_idx[cid]);
        const size_t payload_base = payload_slot * static_cast<size_t>(mailbox_slot_words);
        const size_t exact_data_words = static_cast<size_t>(query_dim) / 4;
        const size_t neighbor_count_offset = static_cast<size_t>(mailbox_neighbor_count_word);
        const size_t neighbor_words_offset = static_cast<size_t>(mailbox_neighbor_words_offset);
        const size_t neighbor_sample_words_offset = static_cast<size_t>(mailbox_neighbor_sample_words_offset);

        if (query_leader) {
            const uint32_t global_row = label_offset + now_local_row;
            const bool bit_match = (use_implicit_filter_mask != 0)
                ? (query_filter_bit != 0ULL)
                : ((d_row_filter_mask[global_row] & query_filter_bit) != 0ULL);
            const uint32_t sample_idx = kDeviceResident
                ? d_row_to_sample[global_row]
                : (use_sample_id_payload != 0
                       ? (node_prefetched
                              ? d_prefetch_node_array[prefetch_vector_index].sample_id
                              : d_neighbor_memory[payload_base + mailbox_sample_id_word])
                       : d_row_to_sample[global_row]);
            const uint32_t matched_active_dims = go_fann::range_filter::count_matching_active_ranges(
                d_query_range_modes,
                d_query_range_lower_bounds,
                d_query_range_upper_bounds,
                d_base_range_values,
                query_idx,
                sample_idx,
                range_count);
            const bool exact_range_match = (active_range_count == 0) || (matched_active_dims == active_range_count);
            if (!node_prefetched) {
                if constexpr (kDeviceResident) {
                    lane0_state[cid].now_filter_result =
                        resident_full_label_check == 0 ||
                                resident_pq_labels_match(d_resident_data_label_indptr,
                                                         d_resident_data_label_indices,
                                                         d_resident_query_label_indptr,
                                                         d_resident_query_label_indices,
                                                         d_resident_query_filter_labels,
                                                         query_idx,
                                                         sample_idx)
                            ? static_cast<uint8_t>(1)
                            : static_cast<uint8_t>(2);
                } else {
                    lane0_state[cid].now_filter_result =
                        static_cast<uint8_t>(d_gdr_data[query_idx * kGdrMailboxStride]);
                }
            }
            if (gpu_predicate::enabled(predicate, query_idx)) {
                // Static prefetch supplies adjacency, never a query-specific verdict.
                lane0_state[cid].now_filter_result = gpu_predicate::matches(
                    predicate, query_idx, sample_idx) ? 1 : 2;
            } else if (!predicate.route_compiled || !predicate.route_compiled[query_idx]) {
                lane0_state[cid].now_filter_result =
                    (lane0_state[cid].now_filter_result == 1 && bit_match && exact_range_match) ? 1 : 2;
            }

        }
        sync_native_query<kDeviceResident>();
        const uint8_t now_filter_result = lane0_state[cid].now_filter_result;

        if constexpr (!kDeviceResident) {
            if (pq_buffer_mode_overlaps(mode) && step_num >= 3) {
                if (query_leader && step_num != 1) {
                    next_buffer_idx[cid] = pq_buffer_mode_uses_single_slot(mode)
                        ? 0
                        : (current_buffer_idx[cid] ^ 1);
                    ++request_generation[cid];
                    d_status_flag[query_idx * 16] = 0;
                    d_gdr_data[query_idx * kGdrMailboxStride] = 0;
                    d_status_memory[query_idx * 16] =
                        unpack_pool_local_row(lane0_state[cid].next.second);
                    d_status_memory[query_idx * 16 + 1] = next_buffer_idx[cid];
                    d_status_memory[query_idx * 16 + 2] = request_generation[cid];
                    __threadfence_system();
                    d_status_flag[query_idx * 16] = 1;
                    if constexpr (kUseRequestRing) {
                        publish_profiled_pq_host_request<kRequestPublishTiming>(
                            d_request_ring_entries,
                            d_request_ring_consumed_sequences,
                            d_request_ring_tails,
                            request_ring_idx,
                            request_ring_capacity,
                            query_idx,
                            HostRequestKind::Demand,
                            request_publish_ns);
                    }
                    maybe_trace_kernel_mailbox_publish(debug_mailbox,
                                                       query_idx,
                                                       active_query_idx,
                                                       group_id,
                                                       unpack_pool_local_row(lane0_state[cid].next.second),
                                                       next_buffer_idx[cid],
                                                       request_generation[cid],
                                                       step_num,
                                                       2);
                }
            }
        }

        sync_native_query<kDeviceResident>();

        u_int32_t node_degree;
        if constexpr (kDeviceResident) {
            node_degree = node_prefetched
                ? d_prefetch_node_array[prefetch_vector_index].neighbor_count
                : resident_graph_degree;
            node_degree = node_degree > static_cast<u_int32_t>(FIXED_DEGREE)
                ? static_cast<u_int32_t>(FIXED_DEGREE)
                : node_degree;
        } else {
            if (!node_prefetched) {
                node_degree = d_neighbor_memory[payload_base + neighbor_count_offset];
                node_degree = (node_degree > static_cast<u_int32_t>(FIXED_DEGREE))
                                  ? static_cast<u_int32_t>(FIXED_DEGREE)
                                  : node_degree;
            } else {
                node_degree = d_prefetch_node_array[prefetch_vector_index].neighbor_count;
                node_degree = (node_degree > static_cast<u_int32_t>(FIXED_DEGREE))
                                  ? static_cast<u_int32_t>(FIXED_DEGREE)
                                  : node_degree;
            }
        }

        if (query_leader) {
            lane0_state[cid].current_pq_distance = gpu_predicate::base_score(lane0_state[cid].now);
        }
        if (primary_query_warp && now_filter_result == 1 &&
            enable_true_distance_rerank != 0) {
            dist_t true_distance_sum = 0.0f;
            for (size_t word_idx = subtid; word_idx < exact_data_words; word_idx += step) {
                const uint32_t element = !node_prefetched
                    ? d_neighbor_memory[payload_base + word_idx]
                    : d_prefetch_node_array[prefetch_vector_index].data[word_idx];
                const int base_dim = static_cast<int>(word_idx) * 4;
                const u_int8_t a = element & 0xFF;
                const u_int8_t b = (element >> 8) & 0xFF;
                const u_int8_t c = (element >> 16) & 0xFF;
                const u_int8_t d = (element >> 24) & 0xFF;
                true_distance_sum +=
                    (a - query_point[j][base_dim + 0]) * (a - query_point[j][base_dim + 0]);
                true_distance_sum +=
                    (b - query_point[j][base_dim + 1]) * (b - query_point[j][base_dim + 1]);
                true_distance_sum +=
                    (c - query_point[j][base_dim + 2]) * (c - query_point[j][base_dim + 2]);
                true_distance_sum +=
                    (d - query_point[j][base_dim + 3]) * (d - query_point[j][base_dim + 3]);
            }
            for (int offset2 = 16; offset2 > 0; offset2 /= 2) {
                true_distance_sum +=
                    __shfl_xor_sync(warp_kernel::kFullMask, true_distance_sum, offset2);
            }
            if (query_leader) {
                lane0_state[cid].now.first = true_distance_sum;
            }
        }

        sync_native_query<kDeviceResident>();
        if (pool_exhausted[cid] ||
            early_stop_reached(early_stop_mode,
                               finished[cid],
                               finish_limit,
                               queue_limit,
                               expanded_count[cid],
                               pass_count[cid],
                               stop_width) ||
            (max_steps > 0 && step_num >= max_steps)) {
            break;
        }

        uint64_t queue_start = 0;
        if constexpr (kCollectDetailedTiming) {
            queue_start = warp_kernel::get_global_timer_ns();
        }
        if (query_leader) {
            expansion.append(query_idx, now_local_row);
            ++expanded_count[cid];
            if (now_filter_result == 1) {
                ++pass_count[cid];
                ResultPair topk_entry;
                topk_entry.first = (enable_true_distance_rerank != 0)
                    ? lane0_state[cid].now.first
                    : lane0_state[cid].current_pq_distance;
                const uint32_t current_sample = kDeviceResident
                    ? d_row_to_sample[label_offset + now_local_row]
                    : (use_sample_id_payload != 0
                           ? (node_prefetched
                                  ? d_prefetch_node_array[prefetch_vector_index].sample_id
                                  : d_neighbor_memory[payload_base + mailbox_sample_id_word])
                           : d_row_to_sample[label_offset + now_local_row]);
                const uint32_t topk_payload_id =
                    use_sample_id_payload != 0 ? current_sample : now_local_row;
                topk_entry.second = topk_payload_id;
                warp_kernel::KernelPair<dist_t, idx_t> stop_entry;
                stop_entry.first = topk_entry.first;
                stop_entry.second = pack_topk_payload(
                    topk_payload_id,
                    lane0_state[cid].current_pq_distance);
                bool stop_topk_improved = false;
                if (lane0_state[cid].stop_topk_heap_size < stop_width) {
                    stop_topk[cid][lane0_state[cid].stop_topk_heap_size++] = stop_entry;
                    push_heap(stop_topk[cid],
                              stop_topk[cid] + lane0_state[cid].stop_topk_heap_size);
                    stop_topk_improved = true;
                    if (lane0_state[cid].stop_topk_heap_size == 1 ||
                        lane0_state[cid].current_pq_distance >
                            lane0_state[cid].stop_topk_pq_worst) {
                        lane0_state[cid].stop_topk_pq_worst =
                            lane0_state[cid].current_pq_distance;
                    }
                } else if (stop_entry.first < stop_topk[cid][0].first) {
                    stop_topk[cid][lane0_state[cid].stop_topk_heap_size++] = stop_entry;
                    push_heap(stop_topk[cid],
                              stop_topk[cid] + lane0_state[cid].stop_topk_heap_size);
                    auto removed = pop_heap(stop_topk[cid],
                                            stop_topk[cid] + lane0_state[cid].stop_topk_heap_size);
                    --lane0_state[cid].stop_topk_heap_size;
                    stop_topk_improved = true;

                    const uint32_t removed_pq_bits = static_cast<uint32_t>(removed.second >> 32);
                    if (lane0_state[cid].current_pq_distance >
                        lane0_state[cid].stop_topk_pq_worst) {
                        lane0_state[cid].stop_topk_pq_worst =
                            lane0_state[cid].current_pq_distance;
                    } else if (removed_pq_bits ==
                               dist_to_bits(lane0_state[cid].stop_topk_pq_worst)) {
                        lane0_state[cid].stop_topk_pq_worst = -FLT_MAX;
                        for (int ii = 0; ii < lane0_state[cid].stop_topk_heap_size; ++ii) {
                            const dist_t pq_dist = unpack_topk_pq_dist(stop_topk[cid][ii].second);
                            if (pq_dist > lane0_state[cid].stop_topk_pq_worst) {
                                lane0_state[cid].stop_topk_pq_worst = pq_dist;
                            }
                        }
                    }
                }
                if (early_stop_mode == kSearchEarlyStopAdaptive && stop_topk_improved) {
                    finished[cid] = 0;
                }
                if (lane0_state[cid].topk_heap_size < result_width) {
                    topk[lane0_state[cid].topk_heap_size++] = topk_entry;
                    push_heap(topk, topk + lane0_state[cid].topk_heap_size);
                } else if (topk_entry.first < topk[0].first) {
                    topk[lane0_state[cid].topk_heap_size++] = topk_entry;
                    push_heap(topk, topk + lane0_state[cid].topk_heap_size);
                    pop_heap(topk, topk + lane0_state[cid].topk_heap_size);
                    --lane0_state[cid].topk_heap_size;
                }
            }

            if constexpr (kProfileMode > 0) {
                native_profile->neighbor_references += node_degree;
            }
            uint64_t hash_test_start_ns = 0;
            if constexpr (kProfileMode > 0) {
                hash_test_start_ns = warp_kernel::get_global_timer_ns();
            }
            for (int i = 0; i < node_degree; ++i) {
                u_int32_t idx;
                if constexpr (kDeviceResident) {
                    idx = node_prefetched
                        ? d_prefetch_node_array[prefetch_vector_index].neighbor[i]
                        : d_resident_graph[(static_cast<size_t>(label_offset) + now_local_row) *
                                               resident_graph_degree +
                                           static_cast<uint32_t>(i)];
                    if (idx == UINT32_MAX) {
                        continue;
                    }
                } else {
                    idx = !node_prefetched
                        ? d_neighbor_memory[payload_base + neighbor_words_offset + i]
                        : d_prefetch_node_array[prefetch_vector_index].neighbor[i];
                }
                if (profiled_native_hash_test<kProfileMode>(native_hash_data,
                                                            idx,
                                                            native_profile)) {
                    if constexpr (kProfileMode == 2) {
                        if (native_frontier_contains_local_row(frontier,
                                                               heap_size[cid],
                                                               idx)) {
                            ++native_profile->resident_hash_hits;
                        } else {
                            ++native_profile->nonresident_hash_hits;
                        }
                    }
                    continue;
                }
                const int list_pos = index_list_len[cid]++;
                index_list[cid][list_pos] = idx;
                if (use_sample_id_payload != 0) {
                    index_sample_list[cid][list_pos] = !node_prefetched
                        ? d_neighbor_memory[payload_base + neighbor_sample_words_offset + i]
                        : d_prefetch_node_array[prefetch_vector_index].neighbor_sample[i];
                }
            }
            if constexpr (kProfileMode > 0) {
                native_profile->hash_test_elapsed_ns +=
                    warp_kernel::get_global_timer_ns() - hash_test_start_ns;
            }
        }
        if constexpr (kCollectDetailedTiming) {
            time4 += static_cast<float>(warp_kernel::get_global_timer_ns() - queue_start) / 1000.0f;
        }

        sync_native_query<kDeviceResident>();
        if (pool_exhausted[cid] ||
            early_stop_reached(early_stop_mode,
                               finished[cid],
                               finish_limit,
                               queue_limit,
                               expanded_count[cid],
                               pass_count[cid],
                               stop_width) ||
            (max_steps > 0 && step_num >= max_steps)) {
            break;
        }

        if constexpr (!kDeviceResident) __nanosleep(10 * 1000);
        uint64_t calculation_start = 0;
        if constexpr (kCollectDetailedTiming) {
            calculation_start = warp_kernel::get_global_timer_ns();
        }
        int degree_limit = index_list_len[cid];
        if constexpr (kProfileMode > 0) {
            if (query_leader) {
                native_profile->post_hash_distance_evaluations += degree_limit;
            }
        }
        for (int i = query_warp; i < degree_limit; i += kQueryWarps) {
            dist_t dist_acc = 0.0f;
            const uint32_t sample_idx = use_sample_id_payload != 0
                ? index_sample_list[cid][i]
                : d_row_to_sample[label_offset + index_list[cid][i]];
            for (int j2 = subtid; j2 < kGraphPqDim; j2 += step) {
                dist_acc += pq_table[query_idx][j2][d_data[static_cast<uint64_t>(sample_idx) * query_pq_dim + j2]];
            }
            for (int offset2 = 16; offset2 > 0; offset2 /= 2) {
                dist_acc += __shfl_xor_sync(warp_kernel::kFullMask, dist_acc, offset2);
            }
            if (subtid == 0) {
                dist_list[cid][i] = dist_acc;
            }
        }
        sync_native_query<kDeviceResident>();
        if constexpr (kCollectDetailedTiming) {
            time3 += static_cast<float>(warp_kernel::get_global_timer_ns() - calculation_start) / 1000.0f;
        }

        if constexpr (!kDeviceResident) __nanosleep(10 * 1000);

        if constexpr (kCollectDetailedTiming) {
            queue_start = warp_kernel::get_global_timer_ns();
        }
        uint64_t frontier_update_start_ns = 0;
        if constexpr (kProfileMode > 0) {
            if (query_leader) {
                frontier_update_start_ns = warp_kernel::get_global_timer_ns();
            }
        }
        bool parallel_predicate = false;
        if constexpr (kPredicateBias) {
            if (primary_query_warp) {
                parallel_predicate = pq_gpu_predicate::prepare_candidate_pass_masks(
                    predicate, query_idx, index_list[cid], index_sample_list[cid],
                    degree_limit, label_offset, d_row_to_sample, use_sample_id_payload != 0,
                    predicate_pass_masks[cid], subtid);
            }
        }
        sync_native_query<kDeviceResident>();
#if GO_FANN_PQ_NATIVE_COOPERATIVE_FRONTIER_UPDATE
        for (int i = 0; i < degree_limit; ++i) {
            int refresh_worst = 0;
            if (subtid == 0) {
                FrontierPair kp;
                const uint32_t local_row = index_list[cid][i];
                const uint32_t global_row = label_offset + local_row;
                const uint32_t sample_idx = use_sample_id_payload != 0
                    ? index_sample_list[cid][i]
                    : d_row_to_sample[global_row];
                // Main-path factors are unity: raw distance is also the stop score.
                kp.first = dist_list[cid][i];
                if (bit_filter_bonus != 1.0f || range_filter_bonus != 1.0f) {
                    const bool bit_match = (use_implicit_filter_mask != 0)
                        ? (query_filter_bit != 0ULL)
                        : ((d_row_filter_mask[global_row] & query_filter_bit) != 0ULL);
                    const uint32_t matched_active_dims = go_fann::range_filter::count_matching_active_ranges(
                        d_query_range_modes,
                        d_query_range_lower_bounds,
                        d_query_range_upper_bounds,
                        d_base_range_values,
                        query_idx,
                        sample_idx,
                        range_count);
                    kp.first = go_fann::range_filter::apply_filter_bonuses(dist_list[cid][i],
                                                                                 bit_match,
                                                                                 matched_active_dims,
                                                                                 bit_filter_bonus,
                                                                                 range_filter_bonus);
                }
                if constexpr (kPredicateBias) {
                    kp.base = kp.first;
                    if (gpu_predicate::enabled(predicate, query_idx)) {
                        const bool pass = parallel_predicate
                            ? ((predicate_pass_masks[cid][i / 32] >> (i % 32)) & 1U) != 0
                            : gpu_predicate::matches(predicate, query_idx, sample_idx);
                        bool saturated = false;
                        kp.first = gpu_predicate::priority(kp.base, predicate_penalty, pass, saturated);
                    }
                }
                kp.second = pack_pool_payload(local_row, false);
                if (heap_size[cid] < queue_limit + 1) {
                    frontier[heap_size[cid]++] = kp;
                    profiled_native_hash_add<kProfileMode>(native_hash_data,
                                                           local_row,
                                                           native_profile);
                    if constexpr (kProfileMode > 0) {
                        ++native_profile->candidate_admissions;
                    }
                    refresh_worst = heap_size[cid] == queue_limit + 1;
                } else if (!(frontier_worst_distance[cid] < kp.first)) {
                    frontier[frontier_worst_index[cid]] = kp;
                    profiled_native_hash_add<kProfileMode>(native_hash_data,
                                                           local_row,
                                                           native_profile);
                    if constexpr (kProfileMode > 0) {
                        ++native_profile->candidate_admissions;
                        ++native_profile->frontier_evictions;
                    }
                    refresh_worst = 1;
                } else if constexpr (kProfileMode > 0) {
                    ++native_profile->threshold_rejections;
                }
            }
            refresh_worst = __shfl_sync(warp_kernel::kFullMask, refresh_worst, 0);
            if (refresh_worst != 0) {
                __syncwarp();
                refresh_native_frontier_worst(frontier,
                                              heap_size[cid],
                                              subtid,
                                              &frontier_worst_index[cid],
                                              &frontier_worst_distance[cid]);
            }
        }
#else
        if (query_leader) {
            for (int i = 0; i < degree_limit; ++i) {
                FrontierPair kp;
                const uint32_t local_row = index_list[cid][i];
                const uint32_t global_row = label_offset + local_row;
                const uint32_t sample_idx = use_sample_id_payload != 0
                    ? index_sample_list[cid][i]
                    : d_row_to_sample[global_row];
                // Main-path factors are unity: raw distance is also the stop score.
                kp.first = dist_list[cid][i];
                if (bit_filter_bonus != 1.0f || range_filter_bonus != 1.0f) {
                    const bool bit_match = (use_implicit_filter_mask != 0)
                        ? (query_filter_bit != 0ULL)
                        : ((d_row_filter_mask[global_row] & query_filter_bit) != 0ULL);
                    const uint32_t matched_active_dims = go_fann::range_filter::count_matching_active_ranges(
                        d_query_range_modes,
                        d_query_range_lower_bounds,
                        d_query_range_upper_bounds,
                        d_base_range_values,
                        query_idx,
                        sample_idx,
                        range_count);
                    kp.first = go_fann::range_filter::apply_filter_bonuses(dist_list[cid][i],
                                                                                 bit_match,
                                                                                 matched_active_dims,
                                                                                 bit_filter_bonus,
                                                                                 range_filter_bonus);
                }
                if constexpr (kPredicateBias) {
                    kp.base = kp.first;
                    if (gpu_predicate::enabled(predicate, query_idx)) {
                        const bool pass = parallel_predicate
                            ? ((predicate_pass_masks[cid][i / 32] >> (i % 32)) & 1U) != 0
                            : gpu_predicate::matches(predicate, query_idx, sample_idx);
                        bool saturated = false;
                        kp.first = gpu_predicate::priority(kp.base, predicate_penalty, pass, saturated);
                    }
                }
                kp.second = pack_pool_payload(local_row, false);
#if GO_FANN_PQ_NATIVE_MAX_HEAP_FRONTIER
                if (heap_size[cid] < queue_limit + 1) {
                    append_native_max_heap(frontier, heap_size[cid], kp);
                    if constexpr (kProfileMode > 0) {
                        ++native_profile->candidate_admissions;
                    }
                } else {
                    if (frontier[1].first < kp.first) {
                        if constexpr (kProfileMode > 0) {
                            ++native_profile->threshold_rejections;
                        }
                        continue;
                    }
                    replace_native_max_heap_root(frontier, heap_size[cid], kp);
                    if constexpr (kProfileMode > 0) {
                        ++native_profile->candidate_admissions;
                        ++native_profile->frontier_evictions;
                    }
                }
                profiled_native_hash_add<kProfileMode>(native_hash_data,
                                                       local_row,
                                                       native_profile);
#else
                if (heap_size[cid] >= queue_limit + 1 && frontier[2].first < kp.first) {
                    if constexpr (kProfileMode > 0) {
                        ++native_profile->threshold_rejections;
                    }
                    continue;
                }
                smmh2::insert(frontier, heap_size[cid], kp);
                profiled_native_hash_add<kProfileMode>(native_hash_data,
                                                       local_row,
                                                       native_profile);
                if constexpr (kProfileMode > 0) {
                    ++native_profile->candidate_admissions;
                }
                if (heap_size[cid] >= queue_limit + 2) {
                    smmh2::pop_max(frontier, heap_size[cid]);
                    if constexpr (kProfileMode > 0) {
                        ++native_profile->frontier_evictions;
                    }
                }
#endif
            }
        }
#endif

        if constexpr (kProfileMode > 0) {
            if (query_leader) {
                native_profile->frontier_update_elapsed_ns +=
                    warp_kernel::get_global_timer_ns() - frontier_update_start_ns;
                ++native_profile->frontier_update_invocations;
            }
        }

        if (step_num == 1) {
            if (primary_query_warp) {
                const bool found =
                    profiled_pickup_next_unused_from_native_frontier<kUseCooperativePickup,
                                                                     kProfileMode>(
                        frontier,
                        heap_size[cid],
                        lane0_state[cid].next,
                        subtid,
                        native_profile);
                if (query_leader && !found) {
                    pool_exhausted[cid] = true;
                    lane0_state[cid].next.first = FLT_MAX;
                    lane0_state[cid].next.second = pack_pool_payload(UINT32_MAX, true);
                }
            }
        }
        if constexpr (kCollectDetailedTiming) {
            time4 += static_cast<float>(warp_kernel::get_global_timer_ns() - queue_start) / 1000.0f;
        }
        sync_native_query<kDeviceResident>();

        if (query_leader) {
            if (!pq_buffer_mode_uses_single_slot(mode) &&
                pq_buffer_mode_overlaps(mode) &&
                step_num >= 3 && step_num != 1) {
                current_buffer_idx[cid] = next_buffer_idx[cid];
            } else if (!pq_buffer_mode_uses_single_slot(mode) &&
                       !pq_buffer_mode_overlaps(mode)) {
                current_buffer_idx[cid] ^= 1;
            } else {
                current_buffer_idx[cid] = 0;
            }
            lane0_state[cid].now = lane0_state[cid].next;
        }
        sync_native_query<kDeviceResident>();
    }

    const float time1 = static_cast<float>(warp_kernel::get_global_timer_ns() - start1) / 1000.0f;

    if (query_leader) {
        expansion.finish(query_idx, expanded_count[cid]);
        if constexpr (!kDeviceResident) d_status_flag[query_idx * 16] = 2;
        const int valid_topk = lane0_state[cid].topk_heap_size < result_width
            ? lane0_state[cid].topk_heap_size
            : result_width;
        for (int i = 0; i < result_width; ++i) {
            d_result[query_idx * result_width + i] = UINT32_MAX;
            d_result_dist[query_idx * result_width + i] = FLT_MAX;
        }
        for (int i = 0; i < valid_topk; ++i) {
            auto res = pop_heap(topk, topk + lane0_state[cid].topk_heap_size - i);
            const uint32_t payload_id = res.second;
            const int output_slot = valid_topk - 1 - i;
            if (payload_id == UINT32_MAX) {
                continue;
            }
            d_result[query_idx * result_width + output_slot] = use_sample_id_payload != 0
                ? payload_id
                : d_row_to_sample[label_offset + payload_id];
            d_result_dist[query_idx * result_width + output_slot] = res.first;
        }

        time_per_block[query_idx * 6] = time1;
        time_per_block[query_idx * 6 + 1] = kCollectDetailedTiming ? time2 : -1.0f;
        time_per_block[query_idx * 6 + 2] = kCollectDetailedTiming ? time3 : -1.0f;
        time_per_block[query_idx * 6 + 3] = kCollectDetailedTiming ? time4 : -1.0f;
        time_per_block[query_idx * 6 + 4] = static_cast<float>(step_num);
        if constexpr (kRequestPublishTiming && kUseRequestRing) {
            time_per_block[query_idx * 6 + 5] =
                static_cast<float>(request_publish_ns) / 1000.0f;
        }
        if constexpr (kProfileMode > 0) {
            native_profile->traversal_steps = static_cast<uint64_t>(step_num);
            native_profile_output[active_query_idx] = *native_profile;
        }
        if constexpr (!kDeviceResident) {
            __threadfence_system();
            d_status_flag[query_idx * 16] = 3;
        }
        if constexpr (kUseRequestRing && !kDeviceResident) {
            publish_pq_host_request(
                d_request_ring_entries,
                d_request_ring_consumed_sequences,
                d_request_ring_tails,
                request_ring_idx,
                request_ring_capacity,
                query_idx,
                HostRequestKind::Done);
        }
        return;
    }
}

template <int kQueueSize,
          int kGraphPqDim,
          bool kUseRequestRing = false,
          bool kDeviceResident = false,
          bool kPredicateBias = false>
__global__ GO_FANN_PQ_NATIVE_LAUNCH_BOUNDS(kQueueSize,
                                              kUseRequestRing,
                                              kDeviceResident) void
native_warp_independent_search_kernel_impl(const pq_idx_t* d_data,
                                              const uint32_t* d_query_ids,
                                              value_t* d_query,
                                              idx_t* d_result,
                                              float* d_result_dist,
                                              const pq_value_t* d_pq_centroid,
                                              pq_value_t* d_pq_table,
                                              dist_t* d_datadist_list,
                                              uint32_t* d_hashdata,
                                              int num_query,
                                              float* time_per_block,
                                              PqNativeProfileRecord* native_profile_output,
                                              uint32_t* d_neighbor_memory,
                                              const uint32_t* d_resident_graph,
                                              uint32_t resident_graph_degree,
                                              const uint64_t* d_resident_data_label_indptr,
                                              const uint32_t* d_resident_data_label_indices,
                                              const uint64_t* d_resident_query_label_indptr,
                                              const uint32_t* d_resident_query_label_indices,
                                              const uint32_t* d_resident_query_filter_labels,
                                              int resident_full_label_check,
                                              volatile int64_t* d_status_memory,
                                              volatile int64_t* d_gdr_data,
                                              volatile int64_t* d_status_flag,
                                              HostRequestRingEntry* d_request_ring_entries,
                                              const volatile uint64_t* d_request_ring_consumed_sequences,
                                              unsigned long long* d_request_ring_tails,
                                              uint32_t request_ring_capacity,
                                              const Node* d_prefetch_node_array,
                                              const uint32_t* d_prefetch_index,
                                              const uint32_t* d_prefetch_begin_by_group,
                                              const uint32_t* d_prefetch_end_by_group,
                                              const uint32_t* d_query_group_ids,
                                              const uint32_t* d_start_local_ids,
                                              const uint32_t* d_start_sample_ids,
                                              const uint32_t* d_row_to_sample,
                                              const uint32_t* d_label_offset,
                                              const uint64_t* d_row_filter_mask,
                                              const uint64_t* d_query_filter_bits,
                                              const uint8_t* d_query_range_modes,
                                              const float* d_query_range_lower_bounds,
                                              const float* d_query_range_upper_bounds,
                                              const float* d_base_range_values,
                                              uint32_t range_count,
                                              int query_dim,
                                              int query_pq_dim,
                                              int use_adaptive_wait,
                                              int buffer_mode,
                                              int queue_limit,
                                              int finish_limit,
                                              int early_stop_mode,
                                              int max_steps,
                                              int stop_width,
                                              int result_width,
                                              uint32_t mailbox_vector_words,
                                              uint32_t mailbox_sample_id_word,
                                              uint32_t mailbox_neighbor_count_word,
                                              uint32_t mailbox_neighbor_words_offset,
                                              uint32_t mailbox_neighbor_sample_words_offset,
                                              uint32_t mailbox_slot_words,
                                              int use_sample_id_payload,
                                              int use_implicit_filter_mask,
                                              int enable_true_distance_rerank,
                                              float bit_filter_bonus,
                                              float range_filter_bonus,
                                              int debug_mailbox,
                                        gpu_predicate::View predicate = {},
                                        expansion_trace::View expansion = {})
{
    using FrontierPair = std::conditional_t<kPredicateBias, gpu_predicate::BiasedPair, PoolPair>;
    constexpr int kQueryWarps = kDeviceResident
        ? kNativeWarpsPerQuery
        : kNativeCoupledWarpsPerQuery;
    extern __shared__ __align__(8) unsigned char native_workspace[];
    const uint32_t warp_in_block =
        static_cast<uint32_t>(threadIdx.x / kCriticalStep);
    const uint32_t cid = warp_in_block / kQueryWarps;
    const uint32_t active_query_idx =
        static_cast<uint32_t>(blockIdx.x * kMultiQuery) + cid;
    if (active_query_idx >= static_cast<uint32_t>(num_query)) {
        return;
    }
    const uint32_t query_idx = d_query_ids != nullptr
        ? d_query_ids[active_query_idx]
        : active_query_idx;
    auto* frontier_workspaces = reinterpret_cast<FrontierPair*>(native_workspace);
    FrontierPair* frontier = frontier_workspaces +
        static_cast<size_t>(cid) * (static_cast<size_t>(queue_limit) + 2);
    auto* topk_workspaces = reinterpret_cast<ResultPair*>(
        native_workspace + static_cast<size_t>(kMultiQuery) *
            (static_cast<size_t>(queue_limit) + 2) * sizeof(FrontierPair));
    ResultPair* topk = topk_workspaces +
        static_cast<size_t>(cid) * (static_cast<size_t>(result_width) + 1);
#if GO_FANN_PQ_NATIVE_PROFILE_MODE > 0
    __shared__ PqNativeProfileRecord native_profiles[kMultiQuery];
    PqNativeProfileRecord* native_profile = &native_profiles[cid];
#else
    PqNativeProfileRecord* native_profile = nullptr;
#endif
    run_pq_search_one_block<kQueueSize,
                            kGraphPqDim,
                            kNativeDetailedTiming,
                            kUseRequestRing,
                            kDeviceResident,
                            kNativeCooperativePickup,
                            kNativeProfileMode,
                            kPredicateBias>(
        d_data,
        d_query,
        d_result,
        d_result_dist,
        d_pq_centroid,
        d_pq_table,
        d_datadist_list,
        d_hashdata,
        active_query_idx,
        query_idx,
        time_per_block,
        native_profile,
        native_profile_output,
        d_neighbor_memory,
        d_resident_graph,
        resident_graph_degree,
        d_resident_data_label_indptr,
        d_resident_data_label_indices,
        d_resident_query_label_indptr,
        d_resident_query_label_indices,
        d_resident_query_filter_labels,
        resident_full_label_check,
        d_status_memory,
        d_gdr_data,
        d_status_flag,
        d_request_ring_entries,
        d_request_ring_consumed_sequences,
        d_request_ring_tails,
        request_ring_capacity,
        d_prefetch_node_array,
        d_prefetch_index,
        d_prefetch_begin_by_group,
        d_prefetch_end_by_group,
        d_query_group_ids,
        d_start_local_ids,
        d_start_sample_ids,
        d_row_to_sample,
        d_label_offset,
        d_row_filter_mask,
        d_query_filter_bits,
        d_query_range_modes,
        d_query_range_lower_bounds,
        d_query_range_upper_bounds,
        d_base_range_values,
        range_count,
        query_dim,
        query_pq_dim,
        use_adaptive_wait,
        buffer_mode,
        queue_limit,
        finish_limit,
        early_stop_mode,
        max_steps,
        stop_width,
        result_width,
        frontier,
        topk,
        mailbox_vector_words,
        mailbox_sample_id_word,
        mailbox_neighbor_count_word,
        mailbox_neighbor_words_offset,
        mailbox_neighbor_sample_words_offset,
        mailbox_slot_words,
        use_sample_id_payload,
        use_implicit_filter_mask,
        enable_true_distance_rerank,
        bit_filter_bonus,
        range_filter_bonus,
        debug_mailbox, predicate, expansion);
}

__global__ void resident_pq_batch_exact_rerank_kernel(
    const uint8_t* d_base_vectors,
    const value_t* d_query,
    const uint32_t* d_query_ids,
    const uint64_t* d_data_label_indptr,
    const uint32_t* d_data_label_indices,
    const uint64_t* d_query_label_indptr,
    const uint32_t* d_query_label_indices,
    const uint32_t* d_query_filter_labels,
    idx_t* d_result,
    float* d_result_dist,
    uint32_t base_rows,
    int num_query,
    int query_dim,
    int result_width,
    int topk)
{
    const uint32_t cid = static_cast<uint32_t>(threadIdx.x / kCriticalStep);
    const uint32_t subtid = static_cast<uint32_t>(threadIdx.x % kCriticalStep);
    const uint32_t active_query_idx =
        static_cast<uint32_t>(blockIdx.x * kMultiQuery) + cid;
    if (active_query_idx >= static_cast<uint32_t>(num_query)) return;

    const uint32_t query_idx = d_query_ids != nullptr
        ? d_query_ids[active_query_idx]
        : active_query_idx;
    __shared__ float rerank_best_dist[kMultiQuery][kTopk];
    __shared__ uint32_t rerank_best_id[kMultiQuery][kTopk];
    __shared__ int rerank_best_size[kMultiQuery];
    if (subtid == 0) rerank_best_size[cid] = 0;
    __syncwarp();

    const size_t result_base = static_cast<size_t>(query_idx) * result_width;
    for (int candidate_idx = 0; candidate_idx < result_width; ++candidate_idx) {
        uint32_t sample_id = UINT32_MAX;
        if (subtid == 0) sample_id = d_result[result_base + candidate_idx];
        sample_id = __shfl_sync(warp_kernel::kFullMask, sample_id, 0);
        if (sample_id == UINT32_MAX || sample_id >= base_rows) continue;
        int labels_match = 0;
        if (subtid == 0) {
            labels_match = resident_pq_labels_match(d_data_label_indptr,
                                                    d_data_label_indices,
                                                    d_query_label_indptr,
                                                    d_query_label_indices,
                                                    d_query_filter_labels,
                                                    query_idx,
                                                    sample_id)
                ? 1
                : 0;
        }
        labels_match = __shfl_sync(warp_kernel::kFullMask, labels_match, 0);
        if (labels_match == 0) {
            continue;
        }

        float distance = 0.0f;
        const size_t base_offset = static_cast<size_t>(sample_id) * query_dim;
        const size_t query_offset = static_cast<size_t>(query_idx) * query_dim;
        for (int dim_idx = static_cast<int>(subtid);
             dim_idx < query_dim;
             dim_idx += kCriticalStep) {
            const int diff = static_cast<int>(d_base_vectors[base_offset + dim_idx]) -
                             static_cast<int>(d_query[query_offset + dim_idx]);
            distance += static_cast<float>(diff * diff);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            distance += __shfl_down_sync(warp_kernel::kFullMask, distance, offset);
        }

        if (subtid == 0) {
            int size = rerank_best_size[cid];
            const bool has_room = size < topk;
            const int worst = topk - 1;
            const bool improves_full = !has_room &&
                (distance < rerank_best_dist[cid][worst] ||
                 (distance == rerank_best_dist[cid][worst] &&
                  sample_id < rerank_best_id[cid][worst]));
            if (has_room || improves_full) {
                int insert_pos = has_room ? size : worst;
                if (has_room) ++size;
                while (insert_pos > 0 &&
                       (distance < rerank_best_dist[cid][insert_pos - 1] ||
                        (distance == rerank_best_dist[cid][insert_pos - 1] &&
                         sample_id < rerank_best_id[cid][insert_pos - 1]))) {
                    rerank_best_dist[cid][insert_pos] =
                        rerank_best_dist[cid][insert_pos - 1];
                    rerank_best_id[cid][insert_pos] =
                        rerank_best_id[cid][insert_pos - 1];
                    --insert_pos;
                }
                rerank_best_dist[cid][insert_pos] = distance;
                rerank_best_id[cid][insert_pos] = sample_id;
                rerank_best_size[cid] = size;
            }
        }
        __syncwarp();
    }

    if (subtid == 0) {
        for (int i = 0; i < result_width; ++i) {
            d_result[result_base + i] = UINT32_MAX;
            d_result_dist[result_base + i] = FLT_MAX;
        }
        const int valid_topk = min(rerank_best_size[cid], topk);
        for (int i = 0; i < valid_topk; ++i) {
            d_result[result_base + i] = rerank_best_id[cid][i];
            d_result_dist[result_base + i] = rerank_best_dist[cid][i];
        }
    }
}

template <int kQueueSize, int kGraphPqDim>
__global__ void persistent_pq_latency_kernel(const pq_idx_t* d_data,
                                             value_t* d_query,
                                             idx_t* d_result,
                                             float* d_result_dist,
                                             const pq_value_t* d_pq_centroid,
                                             pq_value_t* d_pq_table,
                                             dist_t* d_datadist_list,
                                             uint32_t* d_hashdata,
                                             float* time_per_block,
                                             uint32_t* d_neighbor_memory,
                                             volatile int64_t* d_status_memory,
                                             volatile int64_t* d_gdr_data,
                                             volatile int64_t* d_status_flag,
                                             const Node* d_prefetch_node_array,
                                             const uint32_t* d_prefetch_index,
                                             const uint32_t* d_prefetch_begin_by_group,
                                             const uint32_t* d_prefetch_end_by_group,
                                             const uint32_t* d_query_group_ids,
                                             const uint32_t* d_start_local_ids,
                                             const uint32_t* d_start_sample_ids,
                                             const uint32_t* d_row_to_sample,
                                             const uint32_t* d_label_offset,
                                             const uint64_t* d_row_filter_mask,
                                             const uint64_t* d_query_filter_bits,
                                             const uint8_t* d_query_range_modes,
                                             const float* d_query_range_lower_bounds,
                                             const float* d_query_range_upper_bounds,
                                             const float* d_base_range_values,
                                             uint32_t range_count,
                                             int query_dim,
                                             int query_pq_dim,
                                             int use_adaptive_wait,
                                             int buffer_mode,
                                             int queue_limit,
                                             int finish_limit,
                                             int early_stop_mode,
                                             int max_steps,
                                             int stop_width,
                                             int result_width,
                                             uint32_t mailbox_vector_words,
                                             uint32_t mailbox_sample_id_word,
                                             uint32_t mailbox_neighbor_count_word,
                                             uint32_t mailbox_neighbor_words_offset,
                                             uint32_t mailbox_neighbor_sample_words_offset,
                                             uint32_t mailbox_slot_words,
                                             int use_sample_id_payload,
                                             int use_implicit_filter_mask,
                                             int enable_true_distance_rerank,
                                             float bit_filter_bonus,
                                             float range_filter_bonus,
                                             int debug_mailbox,
                                             idx_t* d_lowfreq_result,
                                             float* d_lowfreq_distances,
                                             pq_value_t* d_lowfreq_pq_table,
                                             const uint32_t* d_lowfreq_candidate_ids,
                                             const uint32_t* d_lowfreq_candidate_offsets,
                                             const uint32_t* d_lowfreq_candidate_sizes,
                                             int lowfreq_shortlist_k,
                                             volatile PqPersistentLatencyRequestControl* requests,
                                             volatile PqPersistentLatencyCompletionControl* completions)
{
    const uint32_t slot_idx = static_cast<uint32_t>(blockIdx.x);
    volatile PqPersistentLatencyRequestControl* request = requests + slot_idx;
    volatile PqPersistentLatencyCompletionControl* completion = completions + slot_idx;
    __shared__ uint32_t current_seq;
    __shared__ uint32_t query_idx;
    __shared__ uint32_t is_lowfreq_route;
    __shared__ uint32_t should_stop;
    __shared__ PoolPair frontier[kQueueSize + 2];
    __shared__ ResultPair topk[kHighfreqRerankShortlistMax + 1];
    uint32_t last_seq = 0;

    while (true) {
        if (threadIdx.x == 0) {
            should_stop = 0;
            uint32_t seq = last_seq;
            while (seq == last_seq) {
                const uint32_t stop_value = load_bypass_l1_u32(&request->stop);
                if (stop_value != 0) {
                    should_stop = 1;
                    break;
                }
                seq = load_bypass_l1_u32(&request->ready_seq);
                if (seq == last_seq) {
                    __nanosleep(256);
                }
            }
            current_seq = seq;
            query_idx = load_bypass_l1_u32(&request->query_id);
            is_lowfreq_route = load_bypass_l1_u32(&request->lowfreq_route);
        }
        __syncthreads();
        if (should_stop != 0 || current_seq == kPersistentLatencyStopSeq) {
            return;
        }

        const uint64_t gpu_start_clock = warp_kernel::get_global_timer_ns();
        if (is_lowfreq_route != 0) {
            run_lowfreq_pq_scan_one_block(d_data,
                                          d_query,
                                          query_idx,
                                          slot_idx,
                                          d_lowfreq_result,
                                          d_lowfreq_distances,
                                          d_pq_centroid,
                                          d_lowfreq_pq_table,
                                          d_lowfreq_candidate_ids,
                                          d_lowfreq_candidate_offsets,
                                          d_lowfreq_candidate_sizes,
                                          stop_width,
                                          lowfreq_shortlist_k,
                                          query_dim,
                                          query_pq_dim,
                                          time_per_block);
        } else {
            run_pq_search_one_block<kQueueSize, kGraphPqDim, true, false, false, false, 0>(
                d_data,
                d_query,
                d_result,
                d_result_dist,
                d_pq_centroid,
                d_pq_table,
                d_datadist_list,
                d_hashdata,
                slot_idx,
                query_idx,
                time_per_block,
                nullptr,
                nullptr,
                d_neighbor_memory,
                nullptr,
                0,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                0,
                d_status_memory,
                d_gdr_data,
                d_status_flag,
                nullptr,
                nullptr,
                nullptr,
                0,
                d_prefetch_node_array,
                d_prefetch_index,
                d_prefetch_begin_by_group,
                d_prefetch_end_by_group,
                d_query_group_ids,
                d_start_local_ids,
                d_start_sample_ids,
                d_row_to_sample,
                d_label_offset,
                d_row_filter_mask,
                d_query_filter_bits,
                d_query_range_modes,
                d_query_range_lower_bounds,
                d_query_range_upper_bounds,
                d_base_range_values,
                range_count,
                query_dim,
                query_pq_dim,
                use_adaptive_wait,
                buffer_mode,
                queue_limit,
                finish_limit,
                early_stop_mode,
                max_steps,
                stop_width,
                result_width,
                frontier,
                topk,
                mailbox_vector_words,
                mailbox_sample_id_word,
                mailbox_neighbor_count_word,
                mailbox_neighbor_words_offset,
                mailbox_neighbor_sample_words_offset,
                mailbox_slot_words,
                use_sample_id_payload,
                use_implicit_filter_mask,
                enable_true_distance_rerank,
                bit_filter_bonus,
                range_filter_bonus,
                debug_mailbox);
        }
        const uint64_t gpu_stop_clock = warp_kernel::get_global_timer_ns();
        __syncthreads();
        if (threadIdx.x == 0) {
            completion->query_id = query_idx;
            completion->gpu_start_clock = gpu_start_clock;
            completion->gpu_stop_clock = gpu_stop_clock;
            __threadfence_system();
            completion->done_seq = current_seq;
            last_seq = current_seq;
        }
        __syncthreads();
    }
}

std::vector<std::vector<std::pair<int, value_t>>> build_queries_from_dense(
    const DenseUint8Matrix& queries_matrix)
{
    std::vector<std::vector<std::pair<int, value_t>>> queries(queries_matrix.rows);
    for (uint32_t i = 0; i < queries_matrix.rows; ++i) {
        queries[i].reserve(queries_matrix.cols);
        const uint8_t* row = queries_matrix.row_ptr(i);
        for (uint32_t j = 0; j < queries_matrix.cols; ++j) {
            queries[i].push_back(std::make_pair(static_cast<int>(j), static_cast<value_t>(row[j])));
        }
    }
    return queries;
}

std::vector<std::vector<std::pair<int, value_t>>> build_queries_from_dense_subset(
    const DenseUint8Matrix& queries_matrix,
    const std::vector<uint32_t>& query_indices)
{
    std::vector<std::vector<std::pair<int, value_t>>> queries(query_indices.size());
    for (size_t i = 0; i < query_indices.size(); ++i) {
        const uint32_t query_idx = query_indices[i];
        queries[i].reserve(queries_matrix.cols);
        const uint8_t* row = queries_matrix.row_ptr(query_idx);
        for (uint32_t j = 0; j < queries_matrix.cols; ++j) {
            queries[i].push_back(std::make_pair(static_cast<int>(j), static_cast<value_t>(row[j])));
        }
    }
    return queries;
}

GraphRuntime build_query_subset_runtime(const GraphRuntime& runtime,
                                         const std::vector<uint32_t>& query_indices)
{
    GraphRuntime subset = runtime;
    subset.query_vectors.rows = static_cast<uint32_t>(query_indices.size());
    subset.query_vectors.cols = runtime.query_vectors.cols;
    subset.query_vectors.data.resize(static_cast<size_t>(subset.query_vectors.rows) * subset.query_vectors.cols);
    subset.query_group_ids.resize(query_indices.size());
    subset.query_filter_labels.resize(query_indices.size());
    subset.query_filter_bits.resize(query_indices.size());
    subset.query_ranges.rows = static_cast<uint32_t>(query_indices.size());
    subset.query_ranges.range_count = runtime.query_ranges.range_count;
    subset.query_ranges.modes.resize(query_indices.size() * runtime.query_ranges.range_count);
    subset.query_ranges.lower_bounds.resize(query_indices.size() * runtime.query_ranges.range_count);
    subset.query_ranges.upper_bounds.resize(query_indices.size() * runtime.query_ranges.range_count);
    subset.start_local_ids.resize(query_indices.size());
    subset.query_is_lowfreq.resize(query_indices.size());
    subset.route_to_original_query.resize(query_indices.size());
    subset.high_query_indices.clear();
    subset.low_query_indices.clear();
    subset.query_labels_matrix.rows = static_cast<uint32_t>(query_indices.size());
    subset.query_labels_matrix.cols = runtime.query_labels_matrix.cols;
    subset.query_labels_matrix.indptr.clear();
    subset.query_labels_matrix.indices.clear();
    subset.query_labels_matrix.row_labels.clear();
    subset.query_labels_matrix.source_tiers.clear();
    subset.query_labels_matrix.relevance_ranks.clear();
    subset.query_labels_matrix.indptr.push_back(0);

    // Route ownership and dispatch metadata must follow the same permutation
    // as vectors. A copied full-runtime clause CSR would reinterpret subset IDs.
    const auto select_routes = [&](const auto& source, auto& destination) {
        destination.clear();
        if (source.empty()) return;
        if (source.size() != runtime.query_vectors.rows) {
            fail_pq("PQ subset received misaligned route metadata");
        }
        destination.reserve(query_indices.size());
        for (uint32_t route : query_indices) destination.push_back(source.at(route));
    };
    select_routes(runtime.route_to_clause, subset.route_to_clause);
    select_routes(runtime.route_top_branch_ids, subset.route_top_branch_ids);
    select_routes(runtime.route_enabled, subset.route_enabled);
    select_routes(runtime.route_providers, subset.route_providers);
    select_routes(runtime.route_execution_modes, subset.route_execution_modes);
    select_routes(runtime.route_candidate_sources, subset.route_candidate_sources);
    select_routes(runtime.route_range_posting_begins, subset.route_range_posting_begins);
    select_routes(runtime.route_range_posting_sizes, subset.route_range_posting_sizes);
    select_routes(runtime.route_range_posting_indices, subset.route_range_posting_indices);
    subset.route_clause_indptr.clear();
    subset.route_clause_ids.clear();
    if (!runtime.route_clause_indptr.empty()) {
        if (runtime.route_clause_indptr.size() != runtime.query_vectors.rows + size_t{1}) {
            fail_pq("PQ subset received misaligned route clause offsets");
        }
        subset.route_clause_indptr.push_back(0);
        for (uint32_t route : query_indices) {
            const uint64_t begin = runtime.route_clause_indptr.at(route);
            const uint64_t end = runtime.route_clause_indptr.at(route + 1);
            if (begin > end || end > runtime.route_clause_ids.size()) {
                fail_pq("PQ subset received invalid route clause ownership");
            }
            subset.route_clause_ids.insert(subset.route_clause_ids.end(),
                runtime.route_clause_ids.begin() + begin, runtime.route_clause_ids.begin() + end);
            subset.route_clause_indptr.push_back(subset.route_clause_ids.size());
        }
    }

    for (size_t i = 0; i < query_indices.size(); ++i) {
        const uint32_t query_idx = query_indices[i];
        const uint8_t* src = runtime.query_vectors.row_ptr(query_idx);
        std::copy(src,
                  src + subset.query_vectors.cols,
                  subset.query_vectors.data.begin() + static_cast<size_t>(i) * subset.query_vectors.cols);
        subset.query_group_ids[i] = runtime.query_group_ids[query_idx];
        subset.query_filter_labels[i] = runtime.query_filter_labels[query_idx];
        subset.query_filter_bits[i] = runtime.query_filter_bits[query_idx];
        for (uint32_t range_idx = 0; range_idx < runtime.query_ranges.range_count; ++range_idx) {
            const size_t dst_flat = static_cast<size_t>(i) * runtime.query_ranges.range_count + range_idx;
            const size_t src_flat = static_cast<size_t>(query_idx) * runtime.query_ranges.range_count + range_idx;
            subset.query_ranges.modes[dst_flat] = runtime.query_ranges.modes[src_flat];
            subset.query_ranges.lower_bounds[dst_flat] = runtime.query_ranges.lower_bounds[src_flat];
            subset.query_ranges.upper_bounds[dst_flat] = runtime.query_ranges.upper_bounds[src_flat];
        }
        subset.start_local_ids[i] = runtime.start_local_ids[query_idx];
        subset.query_is_lowfreq[i] = runtime.query_is_lowfreq[query_idx];
        subset.route_to_original_query[i] = runtime.route_to_original_query[query_idx];
        const SparseLabelRowSpan source_labels = sparse_label_row_span(
            runtime.query_labels_matrix, query_idx);
        if (!source_labels.valid) fail_pq("PQ subset received invalid route residual labels");
        subset.query_labels_matrix.row_labels.emplace_back();
        auto& labels = subset.query_labels_matrix.row_labels.back();
        if (source_labels.begin != source_labels.end) {
            labels.assign(source_labels.begin, source_labels.end);
        }
        subset.query_labels_matrix.indices.insert(subset.query_labels_matrix.indices.end(),
                                                  labels.begin(),
                                                  labels.end());
        if (!runtime.query_labels_matrix.source_tiers.empty()) {
            const auto begin = runtime.query_labels_matrix.indptr[query_idx];
            const auto end = runtime.query_labels_matrix.indptr[query_idx + 1];
            subset.query_labels_matrix.source_tiers.insert(subset.query_labels_matrix.source_tiers.end(),
                runtime.query_labels_matrix.source_tiers.begin() + begin,
                runtime.query_labels_matrix.source_tiers.begin() + end);
            subset.query_labels_matrix.relevance_ranks.insert(subset.query_labels_matrix.relevance_ranks.end(),
                runtime.query_labels_matrix.relevance_ranks.begin() + begin,
                runtime.query_labels_matrix.relevance_ranks.begin() + end);
        }
        subset.query_labels_matrix.indptr.push_back(subset.query_labels_matrix.indices.size());
        if (!subset.route_enabled.empty() && !subset.route_enabled[i]) continue;
        if (subset.query_is_lowfreq[i] != 0) {
            subset.low_query_indices.push_back(static_cast<uint32_t>(i));
        } else if (subset.query_group_ids[i] != UINT32_MAX) {
            subset.high_query_indices.push_back(static_cast<uint32_t>(i));
        }
    }
#ifndef NDEBUG
    validate_graph_runtime_query_metadata(subset, "build_query_subset_runtime");
#endif
    return subset;
}

// Own every immutable predicate allocation through context teardown/rollback.
struct PqPredicateBuffers {
    std::vector<void*> allocations;
    ~PqPredicateBuffers() {
        for (void* allocation : allocations) cudaFree(allocation);
    }

    template<class T> const T* upload(const std::vector<T>& host, const char* name) {
        if (host.empty()) return nullptr;
        T* device = nullptr;
        check_cuda(cudaMalloc(&device, host.size() * sizeof(T)), name);
        allocations.push_back(device);
        check_cuda(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), name);
        return device;
    }
};

struct PqPredicateCsr : PqPredicateBuffers {
    const uint32_t* offsets = nullptr;
    const uint32_t* labels = nullptr;
    size_t bytes = 0;
};

struct PqPredicateDeviceState : PqPredicateBuffers {
    route_predicate::Plan plan;
    std::vector<uint8_t> verified_routes;
    gpu_predicate::View view{};
    std::shared_ptr<PqPredicateCsr> csr;
    bool bias_frontier = false;
    bool shortlist_verified = false;
};

std::shared_ptr<PqPredicateCsr> acquire_pq_predicate_csr(
    const GraphRuntime& runtime, const PqSearchOptions& options,
    std::map<std::string, std::shared_ptr<PqPredicateCsr>>& csr_cache,
    bool& index_reused)
{
    const auto began = std::chrono::steady_clock::now();
    const auto require = [](bool pass, const char* why) {
        if (!pass) fail_pq(std::string("GPU predicate: ") + why);
    };
    require(std::none_of(runtime.label_intervals.has_interval.begin(),
                         runtime.label_intervals.has_interval.end(),
                         [](uint8_t value) { return value != 0; }),
            "original-token CSR cannot be built from virtual range labels");
    const auto& matrix = runtime.data_labels;
    int device = 0; check_cuda(cudaGetDevice(&device), "PQ predicate device identity");
    const auto source_path = std::filesystem::canonical(options.data_label_spmat_path);
    const std::string key = source_path.string() + ":" +
        std::to_string(std::filesystem::file_size(source_path)) + ":" +
        std::to_string(std::filesystem::last_write_time(source_path).time_since_epoch().count()) + ":" +
        std::to_string(matrix.rows) + ":" + std::to_string(matrix.cols) + ":" +
        std::to_string(matrix.indices.size()) + ":" + std::to_string(device);
    require(matrix.rows == runtime.base_vectors.rows, "CSR sample domain differs from graph");
    auto found = csr_cache.find(key);
    index_reused = found != csr_cache.end();
    if (!index_reused) {
        auto owner = std::make_shared<PqPredicateCsr>();
        auto offsets = gpu_predicate::validate_csr_offsets(matrix.rows, matrix.cols,
            matrix.indptr, matrix.indices, runtime.base_vectors.rows);
        owner->offsets = owner->upload(offsets, "PQ predicate full CSR offsets");
        owner->labels = owner->upload(matrix.indices, "PQ predicate full CSR labels");
        owner->bytes = (offsets.size() + matrix.indices.size()) * sizeof(uint32_t);
        found = csr_cache.emplace(key, std::move(owner)).first;
        std::cout << "[gpu-predicate-index] backend=pq storage=csr rows=" << matrix.rows
                  << " labels=" << matrix.cols << " relations=" << matrix.indices.size()
                  << " offset_bits=32 label_bits=32 resident_bytes=" << found->second->bytes
                  << " load_validate_upload_ms=" << std::chrono::duration<double,std::milli>(
                         std::chrono::steady_clock::now() - began).count() << '\n';
    }
    return found->second;
}

std::unique_ptr<PqPredicateDeviceState> prepare_pq_predicate(
    const GraphRuntime& runtime, const PqSearchOptions& options,
    std::map<std::string, std::shared_ptr<PqPredicateCsr>>& csr_cache)
{
    if (options.predicate_mode == "cpu") return nullptr;
    const auto require = [](bool pass, const char* why) {
        if (!pass) fail_pq(std::string("GPU predicate: ") + why);
    };
    require(options.predicate_mode == "gpu_exact" || options.predicate_mode == "gpu_exact_bias",
            "mode must be cpu, gpu_exact, or gpu_exact_bias (resolve auto before backend dispatch)");
    require(options.predicate_storage == "csr", "PQ supports full CSR storage only");
    require(!options.predicate_diagnostics, "PQ predicate diagnostics are not implemented");
    require(std::isfinite(options.predicate_bias_alpha) && options.predicate_bias_alpha >= 0,
            "bias alpha must be finite and nonnegative");
    require(options.predicate_bias_alpha == 0 || options.predicate_mode == "gpu_exact_bias",
            "positive bias alpha requires gpu_exact_bias mode");
    require(options.predicate_mode != "gpu_exact_bias" || options.search_early_stop_mode == "off" ||
            options.search_early_stop_mode == "adaptive", "bias mode requires adaptive or off early stopping");
    require(options.graph_transport == "mailbox" &&
            !defaults::pq_query_worker_mode_is_finite(options.query_worker_mode) &&
            !options.latency_persistent_queue,
            "requires coupled in-memory mailbox workers");
    require((highfreq_cpu_rerank_enabled(options) || highfreq_gpu_exact_rerank_enabled(options)) &&
            options.highfreq_rerank_algorithm == "exact" &&
            options.full_filter && options.per_step_full_filter,
            "requires full-filter CPU or GPU-inline exact rerank");

    auto state = std::make_unique<PqPredicateDeviceState>();
    state->plan = route_predicate::compile(runtime);
    const auto& plan = state->plan;
    const size_t route_count = runtime.query_vectors.rows;
    state->verified_routes.assign(route_count, 0);
    std::vector<uint8_t> compiled(route_count, 0), prefetch_disabled(route_count, 0);
    size_t graph_routes = 0, entailed = 0, label_routes = 0, range_routes = 0;
    bool needs_csr = false;
    for (uint32_t route : runtime.high_query_indices) {
        if ((!runtime.route_enabled.empty() && !runtime.route_enabled.at(route)) ||
            (!runtime.route_execution_modes.empty() &&
             runtime.route_execution_modes.at(route) != QueryRouteExecution::GraphTraversal)) continue;
        ++graph_routes;
        const auto& residual = plan.routes.at(route);
        compiled[route] = residual.available;
        if (residual.available) {
            state->verified_routes[route] = 1;
            entailed += residual.entailed;
            bool labels = false, ranges = false;
            for (const auto& clause : residual.clauses) {
                labels |= !clause.labels.empty();
                ranges |= !clause.ranges.empty();
                for (uint32_t label : clause.labels)
                    require(label < runtime.data_labels.cols, "residual label outside original CSR domain");
            }
            label_routes += labels; range_routes += ranges;
            needs_csr |= labels;
            state->bias_frontier |= !residual.entailed && options.predicate_mode == "gpu_exact_bias" &&
                options.predicate_bias_alpha > 0;
        } else {
            // Unproven routes retain CPU checks. Static-prefetch verdicts are
            // stale, so obtain a fresh host response on this fallback path.
            prefetch_disabled[route] = 1;
        }
    }
    auto& v = state->view;
    size_t csr_bytes = 0;
    bool index_reused = false;
    if (needs_csr) {
        state->csr = acquire_pq_predicate_csr(runtime, options, csr_cache, index_reused);
        v.csr_offsets = state->csr->offsets;
        v.csr_labels = state->csr->labels;
        csr_bytes = state->csr->bytes;
    }
    std::vector<uint32_t> offsets{0}, handles;
    std::vector<gpu_predicate::Clause> clauses;
    std::vector<gpu_predicate::RangeAtom> ranges;
    for (size_t route = 0; route < route_count; ++route) {
        if (state->verified_routes[route]) {
            for (const auto& c : plan.routes[route].clauses) {
                gpu_predicate::Clause clause{uint32_t(handles.size()), 0, uint32_t(ranges.size()), 0};
                handles.insert(handles.end(), c.labels.begin(), c.labels.end());
                for (const auto& a : c.ranges) ranges.push_back({a.dimension,a.mode,a.lower,a.upper});
                require(handles.size() <= UINT32_MAX && ranges.size() <= UINT32_MAX, "residual atom overflow");
                clause.label_end = handles.size(); clause.range_end = ranges.size();
                clauses.push_back(clause);
            }
        }
        require(clauses.size() <= UINT32_MAX, "residual clause overflow");
        offsets.push_back(clauses.size());
    }
    if (clauses.empty()) clauses.push_back({0,0,0,0});
    v.route_offsets = state->upload(offsets, "PQ predicate route offsets");
    v.route_handles = state->upload(handles, "PQ predicate original labels");
    v.clauses = state->upload(clauses, "PQ predicate clauses");
    v.ranges = state->upload(ranges, "PQ predicate range atoms");
    v.route_enabled = state->upload(state->verified_routes, "PQ predicate verified routes");
    v.route_compiled = state->upload(compiled, "PQ predicate compiled routes");
    v.route_prefetch_disabled = state->upload(prefetch_disabled, "PQ predicate prefetch fallback");
    v.rows = runtime.base_vectors.rows; v.range_stride = runtime.base_range_count;
    v.alpha = options.predicate_bias_alpha;
    v.pending_base_stop = options.predicate_mode == "gpu_exact_bias" && options.search_early_stop_mode == "adaptive";
    const size_t verified = std::count(state->verified_routes.begin(), state->verified_routes.end(), uint8_t{1});
    state->shortlist_verified = graph_routes > 0 && verified == graph_routes;
    std::cout << "[gpu-predicate-plan] backend=pq storage=csr graph_routes=" << graph_routes
              << " entailed_routes=" << entailed << " gpu_verified_routes=" << verified
              << " cpu_fallback_routes=" << graph_routes - verified
              << " remaining_label_routes=" << label_routes << " range_routes=" << range_routes
              << " csr_resident_bytes=" << csr_bytes << " bitmap_resident_bytes=0"
              << " index_reused=" << index_reused
              << " bias_frontier=" << state->bias_frontier << " pending_base_stop=" << v.pending_base_stop
              << " prefetch_disabled_routes=" << std::count(prefetch_disabled.begin(),prefetch_disabled.end(),uint8_t{1})
              << " rerank_predicate=" << (state->shortlist_verified ? "off" : "retained")
              << " candidate_parallel=" << GO_FANN_GPU_PREDICATE_PARALLEL << '\n';
    return state;
}

// The low-frequency scan owns its plan, independently of graph presence and
// queue resets. Full host-prefilter streams keep their existing producer path.
std::unique_ptr<PqPredicateDeviceState> prepare_pq_posting_predicate(
    const GraphRuntime& runtime, const PqSearchOptions& options,
    std::map<std::string, std::shared_ptr<PqPredicateCsr>>& csr_cache)
{
    if (runtime.low_query_indices.empty()) return nullptr;
    const bool supported = (options.predicate_mode == "gpu_exact" ||
                            options.predicate_mode == "gpu_exact_bias") &&
        options.predicate_storage == "csr" && options.query_worker_mode == "coupled" && options.graph_transport == "mailbox" &&
        !options.latency_persistent_queue;
    const bool host_prefilter = defaults::lowfreq_full_filter_mode_is_pre(options.lowfreq_full_filter_mode);
    if (!supported || host_prefilter) {
        std::cout << "[gpu-posting-predicate] backend=pq provider=label mode=legacy reason="
                  << (host_prefilter ? "host_prefilter" : "unsupported_policy") << '\n';
        return nullptr;
    }
    auto state = std::make_unique<PqPredicateDeviceState>();
    state->plan = route_predicate::compile(runtime, route_predicate::Producer::LabelPosting);
    const auto& plan = state->plan;
    state->verified_routes.assign(plan.routes.size(), 0);
    std::vector<uint32_t> offsets{0}, handles;
    std::vector<gpu_predicate::Clause> clauses;
    std::vector<gpu_predicate::RangeAtom> ranges;
    size_t label_routes = 0;
    for (size_t route = 0; route < plan.routes.size(); ++route) {
        const auto& compiled = plan.routes[route];
        if (compiled.available) {
            state->verified_routes[route] = 1;
            bool has_labels = false;
            for (const auto& c : compiled.clauses) {
                gpu_predicate::Clause clause{uint32_t(handles.size()), 0, uint32_t(ranges.size()), 0};
                handles.insert(handles.end(), c.labels.begin(), c.labels.end());
                has_labels |= !c.labels.empty();
                for (const auto& a : c.ranges) ranges.push_back({a.dimension, a.mode, a.lower, a.upper});
                route_predicate::require(handles.size() <= UINT32_MAX && ranges.size() <= UINT32_MAX,
                                       "PQ posting predicate atom overflow");
                clause.label_end = handles.size(); clause.range_end = ranges.size();
                clauses.push_back(clause);
            }
            label_routes += has_labels;
        }
        route_predicate::require(clauses.size() <= UINT32_MAX, "PQ posting clause overflow");
        offsets.push_back(clauses.size());
    }
    bool index_reused = false;
    auto& v = state->view;
    if (!handles.empty()) {
        state->csr = acquire_pq_predicate_csr(runtime, options, csr_cache, index_reused);
        v.csr_offsets = state->csr->offsets; v.csr_labels = state->csr->labels;
    }
    if (clauses.empty()) clauses.push_back({0, 0, 0, 0});
    v.route_offsets = state->upload(offsets, "PQ posting route offsets");
    v.route_handles = state->upload(handles, "PQ posting original labels");
    v.clauses = state->upload(clauses, "PQ posting clauses");
    v.ranges = state->upload(ranges, "PQ posting range atoms");
    v.route_enabled = state->upload(state->verified_routes, "PQ posting enabled routes");
    v.rows = runtime.base_vectors.rows; v.range_stride = runtime.base_range_count;
    if (!ranges.empty()) {
        route_predicate::require(runtime.base_range_values.size() ==
                                   static_cast<size_t>(v.rows) * v.range_stride,
                               "PQ posting numeric domain mismatch");
        v.base_ranges = state->upload(runtime.base_range_values, "PQ posting base numeric values");
    }
    // Alpha, pending-base state and graph shortlist trust remain disabled.
    const size_t verified = std::count(state->verified_routes.begin(), state->verified_routes.end(), uint8_t{1});
    std::cout << "[gpu-posting-predicate] backend=pq provider=label mode="
              << options.posting_predicate_order << " storage=csr"
              << " verified_routes=" << verified << " remaining_label_routes=" << label_routes
              << " cpu_fallback_routes=" << runtime.low_query_indices.size() - verified
              << " csr_resident_bytes=" << (state->csr ? state->csr->bytes : 0)
              << " index_reused=" << index_reused
              << " csr_owner=immutable_shared_cache cpu_final_validation=unverified_routes_only\n";
    return state;
}

struct PqGraphSearchContext {
    std::shared_ptr<CpuRerankWorkerPool> cpu_rerank_workers;

    std::unique_ptr<PqPredicateDeviceState> predicate;
    bool use_adaptive_wait = false;
    PqBufferMode buffer_mode = PqBufferMode::DoubleSlotOverlap;
    AsyncIoConfig io_config{};

    size_t query_capacity = 0;
    size_t total_query_capacity = 0;
    uint32_t stop_width = kTopk;
    uint32_t result_width = kTopk;
    size_t finite_batch_dynamic_shared_bytes = 0;
    size_t mapped_gdr_bytes = 0;
    size_t mapped_neighbor_bytes = 0;
    bool owns_query_inputs = true;
    bool owns_query_metadata = true;
    bool result_buffer_is_host_mapped = false;
    bool time_buffer_is_host_mapped = false;
    bool use_sample_id_payload = false;
    bool use_implicit_filter_mask = false;
    bool finite_query_state_enabled = false;
    bool native_profile_enabled = false;
    bool request_ring_enabled = false;
    bool device_resident_graph = false;
    bool resident_full_label_check = false;
    uint32_t resident_graph_degree = 0;
    uint32_t resident_base_rows = 0;
    size_t request_ring_capacity_per_ring = 0;
    size_t request_ring_capacity_count = 0;
    PqFiniteTokenTransport finite_token_transport = PqFiniteTokenTransport::DeviceCopy;
    PqFiniteDispatchMode finite_dispatch_mode = PqFiniteDispatchMode::Batch;
    uint32_t finite_queue_capacity = 0;
    uint32_t finite_result_capacity = 0;
    size_t finite_mutable_bytes_per_query = 0;
    size_t finite_mutable_bytes_total = 0;
    uint32_t finite_context_epoch = 0;
    size_t finite_ready_token_capacity = 0;

    const pq_idx_t* d_data = nullptr;
    value_t* d_query = nullptr;
    idx_t* d_result = nullptr;
    float* d_result_dist = nullptr;
    const pq_value_t* d_pq_centroid = nullptr;
    pq_value_t* d_pq_table = nullptr;
    dist_t* d_datadist_list = nullptr;
    uint32_t* d_hashdata = nullptr;
    float* d_time_per_block = nullptr;
    PqNativeProfileRecord* d_native_profile = nullptr;
    PqFiniteContextControl* d_finite_contexts = nullptr;
    PqFiniteContextTelemetry* d_finite_context_telemetry = nullptr;
    PoolPair* d_finite_frontier_state = nullptr;
    StopPair* d_finite_stop_state = nullptr;
    ResultPair* d_finite_result_state = nullptr;
    PqFiniteReadyToken* d_finite_ready_tokens = nullptr;
    PqFiniteReadyToken* h_finite_ready_tokens_mapped = nullptr;

    uint32_t* d_query_group_ids = nullptr;
    uint32_t* d_start_local_ids = nullptr;
    uint32_t* d_start_sample_ids = nullptr;
    uint32_t* d_row_to_sample = nullptr;
    uint32_t* d_label_offset = nullptr;
    uint64_t* d_row_filter_mask = nullptr;
    uint64_t* d_query_filter_bits = nullptr;
    uint8_t* d_query_range_modes = nullptr;
    float* d_query_range_lower_bounds = nullptr;
    float* d_query_range_upper_bounds = nullptr;
    float* d_base_range_values = nullptr;
    uint32_t* d_prefetch_index = nullptr;
    uint32_t* d_prefetch_begin_by_group = nullptr;
    uint32_t* d_prefetch_end_by_group = nullptr;
    Node* d_prefetch_node_array = nullptr;
    uint32_t* d_resident_graph = nullptr;
    uint8_t* d_resident_base_vectors = nullptr;
    uint64_t* d_resident_data_label_indptr = nullptr;
    uint32_t* d_resident_data_label_indices = nullptr;
    uint64_t* d_resident_query_label_indptr = nullptr;
    uint32_t* d_resident_query_label_indices = nullptr;
    uint32_t* d_resident_query_filter_labels = nullptr;

    int64_t* d_gdr_data_raw = nullptr;
    int64_t* d_gdr_data = nullptr;
    int64_t* status_memory = nullptr;
    int64_t* status_flag = nullptr;
    int64_t* d_status_memory = nullptr;
    int64_t* d_status_flag = nullptr;
    HostRequestRingEntry* h_request_ring_entries = nullptr;
    HostRequestRingEntry* d_request_ring_entries = nullptr;
    uint64_t* d_request_ring_consumed_sequences_raw = nullptr;
    uint64_t* h_request_ring_consumed_sequences = nullptr;
    uint64_t* d_request_ring_consumed_sequences = nullptr;
    unsigned long long* d_request_ring_tails = nullptr;
    uint32_t* d_neighbor_memory_raw = nullptr;
    uint32_t* d_neighbor_memory = nullptr;
    uint32_t* h_neighbor_memory = nullptr;
    int64_t* h_gdr_data = nullptr;
    idx_t* h_result_mapped = nullptr;
    float* h_result_dist_mapped = nullptr;
    float* h_time_per_block = nullptr;
    float* h_time_per_block_mapped = nullptr;
    PqNativeProfileRecord* h_native_profile = nullptr;
    gdr_t g = nullptr;
    gdr_mh_t mh{};
    gdr_mh_t mh_neighbor{};
    gdr_mh_t mh_request_ring_consumed_sequences{};
    void* cpu_ptr = nullptr;
    void* cpu_ptr_neighbor = nullptr;
    void* cpu_ptr_request_ring_consumed_sequences = nullptr;
    bool mh_pinned = false;
    bool mh_neighbor_pinned = false;
    bool mh_request_ring_consumed_sequences_pinned = false;
    size_t mapped_request_ring_consumed_sequences_bytes = 0;
    bool request_ring_consumed_device_mapped = false;
    bool using_host_mapped_neighbor = false;
    uint32_t* h_neighbor_memory_host = nullptr;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    HostFilterStats host_filter_stats{};
};

struct PqFinitePersistentWorkerQueue {
    uint32_t resident_blocks = 0;
    uint32_t slot_count = 0;
    int active_blocks_per_sm = 0;
    int multiprocessor_count = 0;
    PqFinitePersistentRequestControl* d_requests_raw = nullptr;
    PqFinitePersistentRequestControl* d_requests = nullptr;
    PqFinitePersistentRequestControl* h_requests = nullptr;
    PqFinitePersistentCompletionControl* h_completions = nullptr;
    PqFinitePersistentCompletionControl* d_completions = nullptr;
    unsigned long long* d_completion_tail = nullptr;
    PqFinitePersistentCompletionNotification* h_completion_notifications = nullptr;
    PqFinitePersistentCompletionNotification* d_completion_notifications = nullptr;
    uint32_t completion_ring_capacity = 0;
    gdr_t g = nullptr;
    gdr_mh_t mh{};
    void* cpu_ptr = nullptr;
    size_t mapped_bytes = 0;
    bool pinned = false;
    cudaStream_t stream = nullptr;
    bool launched = false;
    bool stopped = false;
};

struct PqFinitePersistentSlot {
    uint32_t ready_seq = 0;
    uint32_t query_id = UINT32_MAX;
    uint32_t generation = 0;
    bool busy = false;
};

struct PqFiniteReadyRingQueue {
    uint32_t resident_blocks = 0;
    uint32_t worker_slots = 0;
    uint32_t ring_capacity = 0;
    int active_blocks_per_sm = 0;
    int multiprocessor_count = 0;
    uint8_t* d_storage_raw = nullptr;
    PqFiniteReadyRingControl* d_control = nullptr;
    PqFiniteReadyRingEntry* d_entries = nullptr;
    PqFiniteReadyRingControl* h_control = nullptr;
    PqFiniteReadyRingEntry* h_entries = nullptr;
    unsigned long long* d_consumer_head = nullptr;
    unsigned long long* h_consumed_sequences = nullptr;
    unsigned long long* d_consumed_sequences = nullptr;
    PqFiniteReadyRingStatus* h_status = nullptr;
    PqFiniteReadyRingStatus* d_status = nullptr;
    PqFinitePerHopTiming* d_per_hop_timing = nullptr;
    gdr_t g = nullptr;
    gdr_mh_t mh{};
    void* cpu_ptr = nullptr;
    size_t mapped_bytes = 0;
    bool pinned = false;
    cudaStream_t stream = nullptr;
    bool launched = false;
    bool stopped = false;
};

class PqFiniteReadyRingSink final : public HostCompletionSink {
public:
    PqFiniteReadyRingSink(const std::vector<uint32_t>& active_query_ids,
                          size_t query_capacity);
    ~PqFiniteReadyRingSink() override;

    PqFiniteReadyRingSink(const PqFiniteReadyRingSink&) = delete;
    PqFiniteReadyRingSink& operator=(const PqFiniteReadyRingSink&) = delete;

    void cancel() noexcept override;
    bool publish_ready(uint32_t query_id,
                       uint32_t generation,
                       uint32_t row,
                       uint32_t buffer) override;
    bool publish_done(uint32_t query_id) override;

    void set_context_epoch(uint32_t context_epoch);
    void launch(const PqFiniteKernelParams& params,
                const PqSearchOptions& options,
                size_t dynamic_shared_bytes);
    uint64_t stop();
    bool wait_for_terminal_change(std::chrono::milliseconds timeout);
    uint32_t error_result() const noexcept;
    uint32_t error_query_id() const noexcept;
    uint32_t error_generation() const noexcept;
    uint64_t error_ticket() const noexcept;
    size_t active_query_count() const noexcept;
    size_t done_count() const noexcept;
    uint64_t published_ready_count() const noexcept;
    uint64_t observed_consumed_sequence_max() const noexcept;
    uint64_t host_publish_sample_count() const noexcept;
    uint64_t host_publish_sample_cycles() const noexcept;
    PqFiniteCompletionQueueStats stats() const noexcept;
    const PqFiniteReadyRingQueue& queue() const noexcept;

private:
    bool query_is_active(uint32_t query_id) const noexcept;
    void destroy() noexcept;

    PqFiniteReadyRingQueue queue_;
    std::vector<uint8_t> active_queries_;
    std::unique_ptr<std::atomic<uint64_t>[]> query_states_;
    size_t active_query_count_ = 0;
    std::atomic<uint32_t> context_epoch_{0};
    std::atomic<uint64_t> producer_tail_{0};
    std::atomic<uint64_t> done_enqueued_{0};
    std::atomic<uint64_t> duplicate_rejections_{0};
    std::atomic<uint64_t> invalid_rejections_{0};
    std::atomic<uint64_t> terminal_rejections_{0};
    std::atomic<uint64_t> cancelled_rejections_{0};
    std::atomic<uint64_t> backpressure_waits_{0};
    std::atomic<uint64_t> host_publish_sample_count_{0};
    std::atomic<uint64_t> host_publish_sample_cycles_{0};
    std::atomic<bool> cancelled_{false};
    mutable std::mutex terminal_mutex_;
    std::condition_variable terminal_changed_;
};

bool use_finite_completion_fast_path(const PqGraphSearchContext& ctx,
                                     const GraphRuntime& runtime,
                                     const PqSearchOptions& options)
{
    if (!defaults::pq_query_worker_mode_is_finite(options.query_worker_mode) ||
        ctx.buffer_mode != PqBufferMode::DoubleSlotSerial) {
        return false;
    }
    return true;
}

size_t checked_finite_state_total_bytes(size_t bytes_per_query, size_t query_capacity)
{
    if (bytes_per_query != 0 &&
        query_capacity > std::numeric_limits<size_t>::max() / bytes_per_query) {
        fail_pq("finite query state byte count exceeds size_t");
    }
    return bytes_per_query * query_capacity;
}

void print_finite_query_state_layout(const PqGraphSearchContext& ctx)
{
    std::printf("[pq] query worker state: mode=%s dispatch_mode=%s token_transport=%s "
                "queue_capacity=%u result_capacity=%u "
                "finite_mutable_bytes_per_query=%zu finite_mutable_bytes_total=%zu\n",
                ctx.finite_query_state_enabled ? "finite" : "coupled",
                pq_finite_dispatch_mode_name(ctx.finite_dispatch_mode),
                pq_finite_token_transport_name(ctx.finite_token_transport),
                ctx.finite_queue_capacity,
                ctx.finite_result_capacity,
                ctx.finite_mutable_bytes_per_query,
                ctx.finite_mutable_bytes_total);
}

void reset_finite_query_state(PqGraphSearchContext& ctx, size_t query_count)
{
    if (!ctx.finite_query_state_enabled) {
        return;
    }
    const size_t reset_count = std::min(query_count, ctx.total_query_capacity);
    check_cuda(cudaMemset(ctx.d_finite_contexts,
                          0,
                          reset_count * sizeof(PqFiniteContextControl)),
               "cudaMemset d_finite_contexts");
    check_cuda(cudaMemset(ctx.d_finite_context_telemetry,
                          0,
                          reset_count * sizeof(PqFiniteContextTelemetry)),
               "cudaMemset d_finite_context_telemetry");
    check_cuda(cudaMemset(ctx.d_finite_frontier_state,
                          0,
                          reset_count * (static_cast<size_t>(ctx.finite_queue_capacity) + 2) *
                              sizeof(PoolPair)),
               "cudaMemset d_finite_frontier_state");
    check_cuda(cudaMemset(ctx.d_finite_stop_state,
                          0,
                          reset_count * (static_cast<size_t>(kTopk) + 1) * sizeof(StopPair)),
               "cudaMemset d_finite_stop_state");
    check_cuda(cudaMemset(ctx.d_finite_result_state,
                          0,
                          reset_count * (static_cast<size_t>(ctx.finite_result_capacity) + 1) *
                              sizeof(ResultPair)),
               "cudaMemset d_finite_result_state");
}

void allocate_finite_query_state(PqGraphSearchContext& ctx,
                                 size_t query_capacity,
                                 int search_queue_size)
{
    if (!ctx.finite_query_state_enabled) {
        print_finite_query_state_layout(ctx);
        return;
    }

    ctx.finite_queue_capacity = specialized_queue_capacity(search_queue_size);
    if (ctx.finite_queue_capacity == 0) {
        fail_pq("finite query state requires search_queue_size in [1, 512]");
    }
    ctx.finite_result_capacity = ctx.result_width;
    ctx.finite_mutable_bytes_per_query = finite_mutable_bytes_per_query(
        ctx.finite_queue_capacity, ctx.finite_result_capacity);
    ctx.finite_mutable_bytes_total = checked_finite_state_total_bytes(
        ctx.finite_mutable_bytes_per_query, query_capacity);
    ctx.finite_ready_token_capacity =
        std::min(query_capacity, kFiniteReadyBatchCapacity);

    try {
        check_cuda(cudaMalloc(&ctx.d_finite_contexts,
                              query_capacity * sizeof(PqFiniteContextControl)),
                   "cudaMalloc d_finite_contexts");
        check_cuda(cudaMalloc(&ctx.d_finite_context_telemetry,
                              query_capacity * sizeof(PqFiniteContextTelemetry)),
                   "cudaMalloc d_finite_context_telemetry");
        check_cuda(cudaMalloc(&ctx.d_finite_frontier_state,
                              query_capacity *
                                  (static_cast<size_t>(ctx.finite_queue_capacity) + 2) *
                                  sizeof(PoolPair)),
                   "cudaMalloc d_finite_frontier_state");
        check_cuda(cudaMalloc(&ctx.d_finite_stop_state,
                              query_capacity * (static_cast<size_t>(kTopk) + 1) *
                                  sizeof(StopPair)),
                   "cudaMalloc d_finite_stop_state");
        check_cuda(cudaMalloc(&ctx.d_finite_result_state,
                              query_capacity *
                                  (static_cast<size_t>(ctx.finite_result_capacity) + 1) *
                                  sizeof(ResultPair)),
                   "cudaMalloc d_finite_result_state");
        const size_t ready_token_bytes =
            std::max<size_t>(1, ctx.finite_ready_token_capacity) * sizeof(PqFiniteReadyToken);
        if (ctx.finite_token_transport == PqFiniteTokenTransport::Mapped) {
            check_cuda(cudaHostAlloc(
                           reinterpret_cast<void**>(&ctx.h_finite_ready_tokens_mapped),
                           ready_token_bytes,
                           cudaHostAllocMapped),
                       "cudaHostAlloc h_finite_ready_tokens_mapped");
            check_cuda(cudaHostGetDevicePointer(
                           reinterpret_cast<void**>(&ctx.d_finite_ready_tokens),
                           ctx.h_finite_ready_tokens_mapped,
                           0),
                       "cudaHostGetDevicePointer d_finite_ready_tokens");
        } else {
            check_cuda(cudaMalloc(&ctx.d_finite_ready_tokens, ready_token_bytes),
                       "cudaMalloc d_finite_ready_tokens");
        }
        reset_finite_query_state(ctx, query_capacity);
    } catch (...) {
        if (ctx.h_finite_ready_tokens_mapped != nullptr) {
            cudaFreeHost(ctx.h_finite_ready_tokens_mapped);
            ctx.h_finite_ready_tokens_mapped = nullptr;
            ctx.d_finite_ready_tokens = nullptr;
        } else if (ctx.d_finite_ready_tokens != nullptr) {
            cudaFree(ctx.d_finite_ready_tokens);
            ctx.d_finite_ready_tokens = nullptr;
        }
        if (ctx.d_finite_result_state != nullptr) cudaFree(ctx.d_finite_result_state);
        if (ctx.d_finite_stop_state != nullptr) cudaFree(ctx.d_finite_stop_state);
        if (ctx.d_finite_frontier_state != nullptr) cudaFree(ctx.d_finite_frontier_state);
        if (ctx.d_finite_context_telemetry != nullptr) {
            cudaFree(ctx.d_finite_context_telemetry);
        }
        if (ctx.d_finite_contexts != nullptr) cudaFree(ctx.d_finite_contexts);
        ctx.d_finite_result_state = nullptr;
        ctx.d_finite_stop_state = nullptr;
        ctx.d_finite_frontier_state = nullptr;
        ctx.d_finite_context_telemetry = nullptr;
        ctx.d_finite_contexts = nullptr;
        ctx.finite_ready_token_capacity = 0;
        ctx.finite_mutable_bytes_per_query = 0;
        ctx.finite_mutable_bytes_total = 0;
        throw;
    }
    print_finite_query_state_layout(ctx);
}

struct PqImmutableDeviceState {
    std::shared_ptr<CpuRerankWorkerPool> cpu_rerank_workers;
    // One process/run owner retains immutable source/device-specific CSR pools
    // across queue contexts; route descriptors remain context-specific.
    mutable std::map<std::string, std::shared_ptr<PqPredicateCsr>> predicate_csr_cache;
    double upload_ms = 0.0;
    pq_idx_t* d_data = nullptr;
    pq_value_t* d_pq_centroid = nullptr;
};

struct PqPreparedQueryState {
    double buffer_allocate_ms = 0.0;
    double query_upload_ms = 0.0;
    double graph_lowfreq_context_init_ms = 0.0;
    QuerySetupReporter* setup_report = nullptr;
    std::string setup_point_id;
    std::chrono::steady_clock::time_point benchmark_start;
    value_t* d_all_query = nullptr;
    uint32_t* d_query_group_ids = nullptr;
    uint32_t* d_start_local_ids = nullptr;
    uint64_t* d_query_filter_bits = nullptr;
    uint8_t* d_query_range_modes = nullptr;
    float* d_query_range_lower_bounds = nullptr;
    float* d_query_range_upper_bounds = nullptr;
    uint8_t* d_query_is_lowfreq = nullptr;
    uint32_t* d_high_query_ids = nullptr;
    uint32_t* d_low_query_ids = nullptr;
    uint32_t* d_route_counts = nullptr;
    size_t total_queries = 0;
};

struct PqPersistentLatencyQueue {
    uint32_t slot_count = 0;
    PqPersistentLatencyRequestControl* d_requests_raw = nullptr;
    PqPersistentLatencyRequestControl* d_requests = nullptr;
    PqPersistentLatencyRequestControl* h_requests = nullptr;
    PqPersistentLatencyCompletionControl* h_completions = nullptr;
    PqPersistentLatencyCompletionControl* d_completions = nullptr;
    idx_t* d_lowfreq_result = nullptr;
    idx_t* h_lowfreq_result_mapped = nullptr;
    float* d_lowfreq_distances = nullptr;
    float* h_lowfreq_distances_mapped = nullptr;
    pq_value_t* d_lowfreq_pq_table = nullptr;
    uint32_t* d_lowfreq_candidate_ids = nullptr;
    uint32_t* h_lowfreq_candidate_ids_mapped = nullptr;
    uint32_t* d_lowfreq_candidate_offsets = nullptr;
    uint32_t* h_lowfreq_candidate_offsets_mapped = nullptr;
    uint32_t* d_lowfreq_candidate_sizes = nullptr;
    uint32_t* h_lowfreq_candidate_sizes_mapped = nullptr;
    size_t lowfreq_candidate_capacity = 0;
    uint32_t lowfreq_shortlist_k = 0;
    bool lowfreq_candidate_mode = false;
    gdr_t g = nullptr;
    gdr_mh_t mh{};
    void* cpu_ptr = nullptr;
    size_t mapped_bytes = 0;
    bool pinned = false;
};

struct PqPersistentLatencySlot {
    uint32_t slot_index = 0;
    uint32_t ready_seq = 0;
    uint32_t query_id = UINT32_MAX;
    bool lowfreq_route = false;
    bool busy = false;
    bool gpu_done = false;
    bool rerank_done = false;
    double gpu_ms = 0.0;
    int step_num = 0;
    std::vector<idx_t> result_row;
    std::vector<float> distance_row;
    std::chrono::steady_clock::time_point rerank_done_time{};
    std::chrono::steady_clock::time_point submit_time{};
};

struct PqPersistentLatencyMeasurement {
    uint32_t query_id = UINT32_MAX;
    bool lowfreq_route = false;
    double total_ms = 0.0;
    double gpu_ms = 0.0;
};

struct PqLatencyRoutePlan {
    std::vector<uint32_t> route_ids;
    uint32_t selected_high_routes = 0;
    uint32_t selected_low_routes = 0;
    uint32_t unroutable_routes = 0;
};

struct PqLatencyResultSidecars {
    std::vector<std::vector<idx_t>> route_results;
    std::vector<std::vector<float>> route_distances;
    std::vector<int> route_steps;
    std::vector<uint8_t> route_measured;
    std::vector<uint8_t> route_result_ready;
};

void cleanup_prepared_query_state(PqPreparedQueryState& state)
{
    if (state.d_all_query != nullptr) cudaFree(state.d_all_query);
    if (state.d_query_group_ids != nullptr) cudaFree(state.d_query_group_ids);
    if (state.d_start_local_ids != nullptr) cudaFree(state.d_start_local_ids);
    if (state.d_query_filter_bits != nullptr) cudaFree(state.d_query_filter_bits);
    if (state.d_query_range_modes != nullptr) cudaFree(state.d_query_range_modes);
    if (state.d_query_range_lower_bounds != nullptr) cudaFree(state.d_query_range_lower_bounds);
    if (state.d_query_range_upper_bounds != nullptr) cudaFree(state.d_query_range_upper_bounds);
    if (state.d_query_is_lowfreq != nullptr) cudaFree(state.d_query_is_lowfreq);
    if (state.d_high_query_ids != nullptr) cudaFree(state.d_high_query_ids);
    if (state.d_low_query_ids != nullptr) cudaFree(state.d_low_query_ids);
    if (state.d_route_counts != nullptr) cudaFree(state.d_route_counts);
}

void merge_route_results_to_original(const GraphRuntime& runtime,
                                     const std::vector<std::vector<idx_t>>& route_results,
                                     const std::vector<std::vector<float>>& route_distances,
                                     const std::vector<int>& route_steps,
                                     uint32_t topk,
                                     std::vector<std::vector<idx_t>>& merged_results,
                                     std::vector<int>& merged_steps);

void init_prepared_query_state(PqPreparedQueryState& state,
                               const GraphRuntime& runtime,
                               const PqSearchOptions& options)
{
    const auto allocate_start = std::chrono::steady_clock::now();
    state.total_queries = runtime.query_vectors.rows;
    const size_t query_range_elems = static_cast<size_t>(state.total_queries) * runtime.query_ranges.range_count;
    const size_t query_elems = static_cast<size_t>(runtime.query_vectors.rows) * options.dim;
    check_cuda(cudaMalloc(&state.d_all_query, sizeof(value_t) * query_elems), "cudaMalloc d_all_query");
    check_cuda(cudaMalloc(&state.d_query_group_ids, sizeof(uint32_t) * state.total_queries), "cudaMalloc d_query_group_ids all");
    check_cuda(cudaMalloc(&state.d_start_local_ids, sizeof(uint32_t) * state.total_queries), "cudaMalloc d_start_local_ids all");
    check_cuda(cudaMalloc(&state.d_query_filter_bits, sizeof(uint64_t) * state.total_queries), "cudaMalloc d_query_filter_bits all");
    check_cuda(cudaMalloc(&state.d_query_range_modes, sizeof(uint8_t) * query_range_elems), "cudaMalloc d_query_range_modes all");
    check_cuda(cudaMalloc(&state.d_query_range_lower_bounds, sizeof(float) * query_range_elems), "cudaMalloc d_query_range_lower_bounds all");
    check_cuda(cudaMalloc(&state.d_query_range_upper_bounds, sizeof(float) * query_range_elems), "cudaMalloc d_query_range_upper_bounds all");
    check_cuda(cudaMalloc(&state.d_query_is_lowfreq, sizeof(uint8_t) * state.total_queries), "cudaMalloc d_query_is_lowfreq all");
    check_cuda(cudaMalloc(&state.d_high_query_ids, sizeof(uint32_t) * state.total_queries), "cudaMalloc d_high_query_ids");
    check_cuda(cudaMalloc(&state.d_low_query_ids, sizeof(uint32_t) * state.total_queries), "cudaMalloc d_low_query_ids");
    check_cuda(cudaMalloc(&state.d_route_counts, sizeof(uint32_t) * 2), "cudaMalloc d_route_counts");

    const auto upload_start = std::chrono::steady_clock::now();
    state.buffer_allocate_ms = setup_elapsed_ms(allocate_start, upload_start);
    check_cuda(cudaMemcpy(state.d_all_query,
                          runtime.query_vectors.data.data(),
                          sizeof(value_t) * query_elems,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_all_query");
    check_cuda(cudaMemcpy(state.d_query_group_ids,
                          runtime.query_group_ids.data(),
                          sizeof(uint32_t) * state.total_queries,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_query_group_ids all");
    check_cuda(cudaMemcpy(state.d_start_local_ids,
                          runtime.start_local_ids.data(),
                          sizeof(uint32_t) * state.total_queries,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_start_local_ids all");
    check_cuda(cudaMemcpy(state.d_query_filter_bits,
                          runtime.query_filter_bits.data(),
                          sizeof(uint64_t) * state.total_queries,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_query_filter_bits all");
    check_cuda(cudaMemcpy(state.d_query_range_modes,
                          runtime.query_ranges.modes.data(),
                          sizeof(uint8_t) * query_range_elems,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_query_range_modes all");
    check_cuda(cudaMemcpy(state.d_query_range_lower_bounds,
                          runtime.query_ranges.lower_bounds.data(),
                          sizeof(float) * query_range_elems,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_query_range_lower_bounds all");
    check_cuda(cudaMemcpy(state.d_query_range_upper_bounds,
                          runtime.query_ranges.upper_bounds.data(),
                          sizeof(float) * query_range_elems,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_query_range_upper_bounds all");
    check_cuda(cudaMemcpy(state.d_query_is_lowfreq,
                          runtime.query_is_lowfreq.data(),
                          sizeof(uint8_t) * state.total_queries,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_query_is_lowfreq all");
    state.query_upload_ms = setup_elapsed_ms(upload_start, std::chrono::steady_clock::now());
}

__global__ void partition_query_route_kernel(const uint32_t* d_query_group_ids,
                                             const uint8_t* d_query_is_lowfreq,
                                             const uint32_t* d_start_local_ids,
                                             uint32_t total_queries,
                                             uint32_t* d_high_query_ids,
                                             uint32_t* d_low_query_ids,
                                             uint32_t* d_route_counts)
{
    const uint32_t query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= total_queries) {
        return;
    }
    if (d_query_group_ids[query_idx] == UINT32_MAX) {
        return;
    }
    if (d_query_is_lowfreq[query_idx] != 0) {
        const uint32_t pos = atomicAdd(&d_route_counts[1], 1U);
        d_low_query_ids[pos] = query_idx;
        return;
    }
    if (d_start_local_ids[query_idx] != UINT32_MAX) {
        const uint32_t pos = atomicAdd(&d_route_counts[0], 1U);
        d_high_query_ids[pos] = query_idx;
    }
}

void build_timed_query_routes(const PqPreparedQueryState& prepared_state,
                              std::vector<uint32_t>& high_query_ids,
                              std::vector<uint32_t>& low_query_ids)
{
    check_cuda(cudaMemset(prepared_state.d_route_counts, 0, sizeof(uint32_t) * 2), "cudaMemset d_route_counts");
    const int threads = 256;
    const int blocks = static_cast<int>((prepared_state.total_queries + threads - 1) / threads);
    partition_query_route_kernel<<<blocks, threads>>>(prepared_state.d_query_group_ids,
                                                      prepared_state.d_query_is_lowfreq,
                                                      prepared_state.d_start_local_ids,
                                                      static_cast<uint32_t>(prepared_state.total_queries),
                                                      prepared_state.d_high_query_ids,
                                                      prepared_state.d_low_query_ids,
                                                      prepared_state.d_route_counts);
    check_cuda(cudaGetLastError(), "partition_query_route_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize partition_query_route_kernel");

    uint32_t counts[2] = {0, 0};
    check_cuda(cudaMemcpy(counts,
                          prepared_state.d_route_counts,
                          sizeof(counts),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy route counts");
    high_query_ids.resize(counts[0]);
    low_query_ids.resize(counts[1]);
    if (!high_query_ids.empty()) {
        check_cuda(cudaMemcpy(high_query_ids.data(),
                              prepared_state.d_high_query_ids,
                              sizeof(uint32_t) * high_query_ids.size(),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy high_query_ids");
    }
    if (!low_query_ids.empty()) {
        check_cuda(cudaMemcpy(low_query_ids.data(),
                              prepared_state.d_low_query_ids,
                              sizeof(uint32_t) * low_query_ids.size(),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy low_query_ids");
    }
}

void reorder_high_query_ids_by_graph_instance(const GraphRuntime& runtime,
                                              std::vector<uint32_t>& query_ids)
{
    if (query_ids.size() <= 1) {
        return;
    }
    std::stable_sort(query_ids.begin(), query_ids.end(), [&runtime](uint32_t lhs, uint32_t rhs) {
        const uint32_t lhs_group = lhs < runtime.query_group_ids.size()
            ? runtime.query_group_ids[lhs]
            : UINT32_MAX;
        const uint32_t rhs_group = rhs < runtime.query_group_ids.size()
            ? runtime.query_group_ids[rhs]
            : UINT32_MAX;
        if (lhs_group != rhs_group) {
            return lhs_group < rhs_group;
        }
        const uint32_t lhs_start = lhs < runtime.start_local_ids.size()
            ? runtime.start_local_ids[lhs]
            : UINT32_MAX;
        const uint32_t rhs_start = rhs < runtime.start_local_ids.size()
            ? runtime.start_local_ids[rhs]
            : UINT32_MAX;
        if (lhs_start != rhs_start) {
            return lhs_start < rhs_start;
        }
        return lhs < rhs;
    });
}

void sync_query_ids_to_device(const std::vector<uint32_t>& query_ids,
                              uint32_t* d_query_ids,
                              const char* label)
{
    if (query_ids.empty() || d_query_ids == nullptr) {
        return;
    }
    check_cuda(cudaMemcpy(d_query_ids,
                          query_ids.data(),
                          sizeof(uint32_t) * query_ids.size(),
                          cudaMemcpyHostToDevice),
               label);
}

void cleanup_immutable_state(PqImmutableDeviceState& state)
{
    state.cpu_rerank_workers.reset();
    state.predicate_csr_cache.clear();
    if (state.d_data != nullptr) cudaFree(state.d_data);
    if (state.d_pq_centroid != nullptr) cudaFree(state.d_pq_centroid);
}

void init_immutable_state(PqImmutableDeviceState& state,
                          const PqHostIndex& pq_index,
                          const PqSearchOptions& options)
{
    const auto upload_start = std::chrono::steady_clock::now();
    check_cuda(cudaMalloc(&state.d_data,
                          sizeof(pq_idx_t) * pq_index.num_points * options.pq_dim),
               "cudaMalloc d_data");
    check_cuda(cudaMalloc(&state.d_pq_centroid,
                          sizeof(pq_value_t) * kNumPqCentroids * options.dim),
               "cudaMalloc d_pq_centroid");
    check_cuda(cudaMemcpy(state.d_data,
                          pq_index.compressed_codes.data(),
                          sizeof(pq_idx_t) * pq_index.num_points * options.pq_dim,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_data");
    check_cuda(cudaMemcpy(state.d_pq_centroid,
                          pq_index.centroid_table.data(),
                          sizeof(pq_value_t) * kNumPqCentroids * options.dim,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_pq_centroid");
    state.upload_ms = setup_elapsed_ms(upload_start, std::chrono::steady_clock::now());
    if (options.cpu_persistent_workers) {
        state.cpu_rerank_workers = std::make_shared<CpuRerankWorkerPool>(
            static_cast<size_t>(std::max(1, options.highfreq_rerank_threads)));
        std::cout << "[pq-cpu-workers] persistent=1 capacity="
                  << state.cpu_rerank_workers->capacity() << " shared_graph_posting=1\n";
    }
}

void cleanup_graph_context(PqGraphSearchContext& ctx)
{
    if (ctx.g != nullptr && ctx.cpu_ptr_request_ring_consumed_sequences != nullptr) {
        gdr_unmap(ctx.g,
                  ctx.mh_request_ring_consumed_sequences,
                  ctx.cpu_ptr_request_ring_consumed_sequences,
                  ctx.mapped_request_ring_consumed_sequences_bytes);
    }
    if (ctx.g != nullptr && ctx.mh_request_ring_consumed_sequences_pinned) {
        gdr_unpin_buffer(ctx.g, ctx.mh_request_ring_consumed_sequences);
    }
    if (!ctx.using_host_mapped_neighbor) {
        if (ctx.g != nullptr && ctx.cpu_ptr_neighbor != nullptr) {
            gdr_unmap(ctx.g, ctx.mh_neighbor, ctx.cpu_ptr_neighbor, ctx.mapped_neighbor_bytes);
        }
        if (ctx.g != nullptr && ctx.mh_neighbor_pinned) {
            gdr_unpin_buffer(ctx.g, ctx.mh_neighbor);
        }
    }
    if (ctx.g != nullptr && ctx.cpu_ptr != nullptr) {
        gdr_unmap(ctx.g, ctx.mh, ctx.cpu_ptr, ctx.mapped_gdr_bytes);
    }
    if (ctx.g != nullptr && ctx.mh_pinned) {
        gdr_unpin_buffer(ctx.g, ctx.mh);
    }
    if (ctx.g != nullptr) {
        gdr_close(ctx.g);
    }

    if (ctx.start != nullptr) cudaEventDestroy(ctx.start);
    if (ctx.stop != nullptr) cudaEventDestroy(ctx.stop);
    if (ctx.owns_query_inputs && ctx.d_query != nullptr) cudaFree(ctx.d_query);
    if (ctx.result_buffer_is_host_mapped) {
        if (ctx.h_result_mapped != nullptr) cudaFreeHost(ctx.h_result_mapped);
        if (ctx.h_result_dist_mapped != nullptr) cudaFreeHost(ctx.h_result_dist_mapped);
    } else {
        if (ctx.d_result != nullptr) cudaFree(ctx.d_result);
        if (ctx.d_result_dist != nullptr) cudaFree(ctx.d_result_dist);
    }
    if (ctx.d_pq_table != nullptr) cudaFree(ctx.d_pq_table);
    if (ctx.d_hashdata != nullptr) cudaFree(ctx.d_hashdata);
    if (ctx.d_native_profile != nullptr) cudaFree(ctx.d_native_profile);
    if (ctx.d_finite_contexts != nullptr) cudaFree(ctx.d_finite_contexts);
    if (ctx.d_finite_context_telemetry != nullptr) {
        cudaFree(ctx.d_finite_context_telemetry);
    }
    if (ctx.d_finite_frontier_state != nullptr) cudaFree(ctx.d_finite_frontier_state);
    if (ctx.d_finite_stop_state != nullptr) cudaFree(ctx.d_finite_stop_state);
    if (ctx.d_finite_result_state != nullptr) cudaFree(ctx.d_finite_result_state);
    if (ctx.h_finite_ready_tokens_mapped != nullptr) {
        cudaFreeHost(ctx.h_finite_ready_tokens_mapped);
    } else if (ctx.d_finite_ready_tokens != nullptr) {
        cudaFree(ctx.d_finite_ready_tokens);
    }
    if (ctx.d_datadist_list != nullptr) cudaFree(ctx.d_datadist_list);
    if (ctx.time_buffer_is_host_mapped) {
        if (ctx.h_time_per_block_mapped != nullptr) cudaFreeHost(ctx.h_time_per_block_mapped);
    } else {
        if (ctx.d_time_per_block != nullptr) cudaFree(ctx.d_time_per_block);
    }
    if (ctx.status_memory != nullptr) cudaFreeHost(ctx.status_memory);
    if (ctx.status_flag != nullptr) cudaFreeHost(ctx.status_flag);
    if (ctx.h_request_ring_entries != nullptr) cudaFreeHost(ctx.h_request_ring_entries);
    if (ctx.d_request_ring_consumed_sequences_raw != nullptr) {
        cudaFree(ctx.d_request_ring_consumed_sequences_raw);
    } else if (ctx.h_request_ring_consumed_sequences != nullptr) {
        cudaFreeHost(ctx.h_request_ring_consumed_sequences);
    }
    if (ctx.d_request_ring_tails != nullptr) cudaFree(ctx.d_request_ring_tails);
    if (ctx.d_prefetch_node_array != nullptr) cudaFree(ctx.d_prefetch_node_array);
    if (ctx.d_prefetch_index != nullptr) cudaFree(ctx.d_prefetch_index);
    if (ctx.d_prefetch_begin_by_group != nullptr) cudaFree(ctx.d_prefetch_begin_by_group);
    if (ctx.d_prefetch_end_by_group != nullptr) cudaFree(ctx.d_prefetch_end_by_group);
    if (ctx.d_resident_graph != nullptr) cudaFree(ctx.d_resident_graph);
    if (ctx.d_resident_base_vectors != nullptr) cudaFree(ctx.d_resident_base_vectors);
    if (ctx.d_resident_data_label_indptr != nullptr) cudaFree(ctx.d_resident_data_label_indptr);
    if (ctx.d_resident_data_label_indices != nullptr) cudaFree(ctx.d_resident_data_label_indices);
    if (ctx.d_resident_query_label_indptr != nullptr) cudaFree(ctx.d_resident_query_label_indptr);
    if (ctx.d_resident_query_label_indices != nullptr) cudaFree(ctx.d_resident_query_label_indices);
    if (ctx.d_resident_query_filter_labels != nullptr) cudaFree(ctx.d_resident_query_filter_labels);
    if (ctx.owns_query_metadata && ctx.d_query_group_ids != nullptr) cudaFree(ctx.d_query_group_ids);
    if (ctx.owns_query_metadata && ctx.d_start_local_ids != nullptr) cudaFree(ctx.d_start_local_ids);
    if (ctx.d_start_sample_ids != nullptr) cudaFree(ctx.d_start_sample_ids);
    if (ctx.d_row_to_sample != nullptr) cudaFree(ctx.d_row_to_sample);
    if (ctx.d_label_offset != nullptr) cudaFree(ctx.d_label_offset);
    if (ctx.d_row_filter_mask != nullptr) cudaFree(ctx.d_row_filter_mask);
    if (ctx.owns_query_metadata && ctx.d_query_filter_bits != nullptr) cudaFree(ctx.d_query_filter_bits);
    if (ctx.owns_query_metadata && ctx.d_query_range_modes != nullptr) cudaFree(ctx.d_query_range_modes);
    if (ctx.owns_query_metadata && ctx.d_query_range_lower_bounds != nullptr) cudaFree(ctx.d_query_range_lower_bounds);
    if (ctx.owns_query_metadata && ctx.d_query_range_upper_bounds != nullptr) cudaFree(ctx.d_query_range_upper_bounds);
    if (ctx.d_base_range_values != nullptr) cudaFree(ctx.d_base_range_values);
    if (ctx.using_host_mapped_neighbor) {
        if (ctx.h_neighbor_memory_host != nullptr) cudaFreeHost(ctx.h_neighbor_memory_host);
    } else {
        if (ctx.d_neighbor_memory_raw != nullptr) cudaFree(ctx.d_neighbor_memory_raw);
    }
    if (ctx.d_gdr_data_raw != nullptr) cudaFree(ctx.d_gdr_data_raw);
    if (ctx.h_time_per_block != nullptr) free(ctx.h_time_per_block);
    if (ctx.h_native_profile != nullptr) free(ctx.h_native_profile);
    ctx = PqGraphSearchContext{};
}

void init_graph_context(PqGraphSearchContext& ctx,
                        size_t query_capacity,
                        const PqImmutableDeviceState& immutable_state,
                        const GraphRuntime& runtime,
                        const PqSearchOptions& options)
{
    struct InitRollback {
        PqGraphSearchContext* context;
        bool committed = false;

        ~InitRollback()
        {
            if (!committed) cleanup_graph_context(*context);
        }
    } rollback{&ctx};

    if (query_capacity > UINT32_MAX) {
        fail_pq("PQ graph context capacity exceeds 32-bit query indexing");
    }
    ctx.use_adaptive_wait = use_adaptive_wait_policy(options.host_reply_wait_policy);
    ctx.buffer_mode = parse_pq_buffer_mode(options.buffer_mode);
    ctx.cpu_rerank_workers = immutable_state.cpu_rerank_workers;
    ctx.d_data = immutable_state.d_data;
    ctx.d_pq_centroid = immutable_state.d_pq_centroid;
    ctx.query_capacity = query_capacity;
    ctx.total_query_capacity = query_capacity;
    ctx.stop_width = highfreq_stop_width(options);
    ctx.result_width = highfreq_result_width(options);
    ctx.predicate = prepare_pq_predicate(runtime, options, immutable_state.predicate_csr_cache);
    const bool predicate_bias = ctx.predicate && ctx.predicate->bias_frontier;
    const size_t native_batch_dynamic_shared_bytes = native_pq_workspace_shared_bytes(
        options.search_queue_size, ctx.result_width, predicate_bias);
    ctx.finite_batch_dynamic_shared_bytes = pq_result_heap_shared_bytes(ctx.result_width);
    ctx.owns_query_inputs = true;
    ctx.owns_query_metadata = true;
    ctx.device_resident_graph = pq_device_resident_graph_enabled(options);
    if (ctx.device_resident_graph && options.per_step_full_filter) {
        for (uint32_t route : runtime.high_query_indices) {
            if (runtime.route_clause_indptr.size() == runtime.query_vectors.rows + size_t{1} &&
                runtime.route_clause_indptr[route + 1] - runtime.route_clause_indptr[route] > 1) {
                fail_pq("shared OR residuals require mailbox graph transport");
            }
        }
    }
    ctx.resident_full_label_check = options.per_step_full_filter;
    ctx.resident_graph_degree = static_cast<uint32_t>(runtime.graph.cols);
    ctx.resident_base_rows = runtime.base_vectors.rows;
    ctx.use_sample_id_payload = ctx.device_resident_graph
        ? false
        : pq_graph_use_sample_id_payload(options);
    ctx.use_implicit_filter_mask = pq_graph_can_use_implicit_filter_mask(runtime);
    ctx.finite_query_state_enabled =
        defaults::pq_query_worker_mode_is_finite(options.query_worker_mode);
    ctx.native_profile_enabled = kNativeProfileMode > 0 &&
        !ctx.finite_query_state_enabled && !options.latency_persistent_queue;
    ctx.request_ring_enabled = options.request_mode == "batch_ring";
    ctx.request_ring_consumed_device_mapped =
        options.request_consumed_memory == "device";
    if (ctx.request_ring_enabled &&
        (ctx.finite_query_state_enabled || options.request_batch_queries == 0)) {
        fail_pq("PQ batch request ring requires coupled memory workers with a positive batch size");
    }
    if (ctx.request_ring_enabled) {
        ctx.request_ring_capacity_per_ring = options.request_batch_queries;
        ctx.request_ring_capacity_count =
            (ctx.total_query_capacity + ctx.request_ring_capacity_per_ring - 1) /
            ctx.request_ring_capacity_per_ring;
        printf("[pq] request ring: mode=batch_ring batch_queries=%zu rings=%zu "
               "consumed_memory=%s\n",
               ctx.request_ring_capacity_per_ring,
               ctx.request_ring_capacity_count,
               options.request_consumed_memory.c_str());
    }
    ctx.finite_token_transport =
        parse_pq_finite_token_transport(options.finite_token_transport);
    ctx.finite_dispatch_mode =
        parse_pq_finite_dispatch_mode(options.finite_dispatch_mode);
    ctx.io_config = AsyncIoConfig(static_cast<uint32_t>(options.dim),
                                  static_cast<uint32_t>(options.degree),
                                  static_cast<uint32_t>(options.prefetch_hops));
    ctx.io_config.payload_slot_count =
        (ctx.buffer_mode == PqBufferMode::SingleSlotSerial || ctx.buffer_mode == PqBufferMode::SingleSlotOverlap)
        ? 1U
        : 2U;
    ctx.io_config.host_service_threads = options.host_service_threads;
    ctx.io_config.host_service_cpu_start = options.host_service_cpu_start;
    ctx.io_config.host_service_cpu_count = options.host_service_cpu_count;
    ctx.io_config.host_demand_batch_size = options.host_demand_batch_size;
    ctx.io_config.host_demand_batch_timeout_us = options.host_demand_batch_timeout_us;
    ctx.io_config.publish_valid_vector_payload =
        defaults::highfreq_rerank_mode_is_gpu(options.highfreq_rerank_mode) &&
        !ctx.device_resident_graph;
    ctx.io_config.skip_filtered_vector_payload = options.skip_filtered_vector_payload;
    ctx.io_config.per_step_full_filter = options.per_step_full_filter;
    if (ctx.predicate) {
        ctx.io_config.predicate_plan = &ctx.predicate->plan;
        ctx.io_config.predicate_gpu_verified_routes = &ctx.predicate->verified_routes;
        ctx.io_config.enable_full_label_check = options.full_filter;
    }
    ctx.io_config.include_sample_ids = ctx.use_sample_id_payload;
    ctx.io_config.set_reserve_vector_payload(
        defaults::highfreq_rerank_mode_is_gpu(options.highfreq_rerank_mode) &&
        !ctx.device_resident_graph);

    const size_t gdr_size = ctx.total_query_capacity * sizeof(int64_t) * kGdrMailboxStride;
    ctx.mapped_gdr_bytes = round_up_gdr_bytes(gdr_size);
    const size_t neighbor_bytes = ctx.total_query_capacity * ctx.io_config.block_size * ctx.io_config.payload_slot_count;
    ctx.mapped_neighbor_bytes = round_up_gdr_bytes(neighbor_bytes);
    printf("[pq] graph mailbox layout: slot_bytes=%zu slot_words=%zu vector_words=%zu sample_ids=%d elems_per_node=%zu payload_slots=%u neighbor_bytes=%zu mapped_neighbor_bytes=%zu row_to_sample_gpu=%d row_filter_mask_gpu=%d\n",
           ctx.io_config.block_size,
           ctx.io_config.elems_per_block,
           ctx.io_config.vector_payload_words,
           ctx.io_config.include_sample_ids ? 1 : 0,
           ctx.io_config.elems_per_node,
           ctx.io_config.payload_slot_count,
           neighbor_bytes,
           ctx.mapped_neighbor_bytes,
           ctx.use_sample_id_payload ? 0 : 1,
           ctx.use_implicit_filter_mask ? 0 : 1);

    int max_blocks_per_sm = 0;
    dispatch_graph_kernel(options.search_queue_size, options.pq_dim, [&]<int QS, int PD>() {
        if (ctx.finite_query_state_enabled) {
            check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &max_blocks_per_sm,
                           finite_pq_resume_kernel<QS, PD>,
                           kNativeThreadInBlock,
                           ctx.finite_batch_dynamic_shared_bytes),
                       "cudaOccupancyMaxActiveBlocksPerMultiprocessor finite PQ");
        } else if (ctx.device_resident_graph) {
            check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &max_blocks_per_sm,
                           native_warp_independent_search_kernel_impl<QS, PD, false, true>,
                           kNativeResidentThreadInBlock,
                           native_batch_dynamic_shared_bytes),
                       "cudaOccupancyMaxActiveBlocksPerMultiprocessor device-resident PQ");
        } else if (ctx.request_ring_enabled) {
            check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &max_blocks_per_sm,
                           predicate_bias ? native_warp_independent_search_kernel_impl<QS, PD, true, false, true>
                                          : native_warp_independent_search_kernel_impl<QS, PD, true>,
                           kNativeCoupledThreadInBlock,
                           native_batch_dynamic_shared_bytes),
                       "cudaOccupancyMaxActiveBlocksPerMultiprocessor coupled PQ request ring");
        } else {
            check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &max_blocks_per_sm,
                           predicate_bias ? native_warp_independent_search_kernel_impl<QS, PD, false, false, true>
                                          : native_warp_independent_search_kernel_impl<QS, PD>,
                           kNativeCoupledThreadInBlock,
                           native_batch_dynamic_shared_bytes),
                       "cudaOccupancyMaxActiveBlocksPerMultiprocessor coupled PQ");
        }
    });

    const size_t reported_dynamic_shared_bytes = ctx.finite_query_state_enabled
        ? ctx.finite_batch_dynamic_shared_bytes
        : native_batch_dynamic_shared_bytes;
    const int reported_warps_per_query = ctx.finite_query_state_enabled
        ? 1
        : (ctx.device_resident_graph
               ? kNativeWarpsPerQuery
               : kNativeCoupledWarpsPerQuery);
    const int reported_threads_per_block = kNativeThreadInBlock *
        reported_warps_per_query;
    printf("[pq] native multi-query: worker_mode=%s graph_transport=%s "
           "coupled_multiwarp_experimental=%d queries_per_block=%d warps_per_query=%d "
           "threads_per_block=%d dynamic_shared_bytes=%zu active_blocks_per_sm=%d "
           "active_query_contexts_per_sm=%d active_compute_warps_per_sm=%d\n",
           ctx.finite_query_state_enabled ? "finite" : "coupled",
           ctx.device_resident_graph ? "device_resident" : "mailbox",
           kExperimentalCoupledMultiwarp ? 1 : 0,
           kMultiQuery,
           reported_warps_per_query,
           reported_threads_per_block,
           reported_dynamic_shared_bytes,
           max_blocks_per_sm,
           max_blocks_per_sm * kMultiQuery,
           max_blocks_per_sm * kMultiQuery * reported_warps_per_query);
    if (!ctx.finite_query_state_enabled && !ctx.device_resident_graph &&
        kNativeCoupledWarpsPerQuery > 1) {
        printf("[pq][experimental] coupled multiwarp is enabled for ablation only\n");
    }

    printf("Starting GDR setup...\n");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ctx.d_gdr_data_raw),
                          ctx.mapped_gdr_bytes + kGdrAlignBytes),
               "cudaMalloc d_gdr_data_raw");
    ctx.d_gdr_data = align_gdr_pointer(ctx.d_gdr_data_raw);

    ctx.g = gdr_open();
    if (ctx.g == nullptr) fail_pq("gdr_open returned nullptr");
    check_gdr(gdr_pin_buffer(ctx.g,
                             reinterpret_cast<unsigned long>(ctx.d_gdr_data),
                             ctx.mapped_gdr_bytes,
                             0,
                             0,
                             &ctx.mh),
              "gdr_pin_buffer d_gdr_data");
    ctx.mh_pinned = true;
    check_gdr(gdr_map(ctx.g, ctx.mh, &ctx.cpu_ptr, ctx.mapped_gdr_bytes), "gdr_map d_gdr_data");
    ctx.h_gdr_data = reinterpret_cast<int64_t*>(ctx.cpu_ptr);
    printf("GDR setup completed.\n");

    printf("Preparing data on GPU...\n");
    printf("Data preparation completed.\n");

    check_cuda(cudaMalloc(&ctx.d_query, sizeof(value_t) * query_capacity * options.dim), "cudaMalloc d_query");
    ctx.result_buffer_is_host_mapped = highfreq_cpu_streaming_enabled(options);
    if (ctx.result_buffer_is_host_mapped) {
        check_cuda(cudaHostAlloc((void**)&ctx.h_result_mapped,
                                 sizeof(idx_t) * ctx.total_query_capacity * ctx.result_width,
                                 cudaHostAllocMapped),
                   "cudaHostAlloc h_result_mapped");
        check_cuda(cudaHostAlloc((void**)&ctx.h_result_dist_mapped,
                                 sizeof(float) * ctx.total_query_capacity * ctx.result_width,
                                 cudaHostAllocMapped),
                   "cudaHostAlloc h_result_dist_mapped");
        check_cuda(cudaHostGetDevicePointer((void**)&ctx.d_result, ctx.h_result_mapped, 0),
                   "cudaHostGetDevicePointer d_result");
        check_cuda(cudaHostGetDevicePointer((void**)&ctx.d_result_dist, ctx.h_result_dist_mapped, 0),
                   "cudaHostGetDevicePointer d_result_dist");
    } else {
        check_cuda(cudaMalloc(&ctx.d_result, sizeof(idx_t) * ctx.total_query_capacity * ctx.result_width),
                   "cudaMalloc d_result");
        check_cuda(cudaMalloc(&ctx.d_result_dist,
                              sizeof(float) * ctx.total_query_capacity * ctx.result_width),
                   "cudaMalloc d_result_dist");
    }
    check_cuda(cudaMalloc(&ctx.d_pq_table, sizeof(pq_value_t) * options.pq_dim * kNumPqCentroids * query_capacity), "cudaMalloc d_pq_table");
    check_cuda(cudaMalloc(&ctx.d_datadist_list, sizeof(dist_t) * query_capacity * (FIXED_DEGREE * kMultiprobe)), "cudaMalloc d_datadist_list");
    check_cuda(cudaMalloc(&ctx.d_hashdata, sizeof(uint32_t) * query_capacity * kHashTableCapacity), "cudaMalloc d_hashdata");
    if constexpr (kNativeProfileMode > 0) {
        if (ctx.native_profile_enabled) {
            check_cuda(cudaMalloc(&ctx.d_native_profile,
                                  ctx.total_query_capacity * sizeof(PqNativeProfileRecord)),
                       "cudaMalloc d_native_profile");
        }
    }
    allocate_finite_query_state(ctx, query_capacity, options.search_queue_size);
    ctx.time_buffer_is_host_mapped = highfreq_cpu_streaming_enabled(options);
    if (ctx.time_buffer_is_host_mapped) {
        check_cuda(cudaHostAlloc((void**)&ctx.h_time_per_block_mapped,
                                 ctx.total_query_capacity * 6 * sizeof(float),
                                 cudaHostAllocMapped),
                   "cudaHostAlloc h_time_per_block_mapped");
        check_cuda(cudaHostGetDevicePointer((void**)&ctx.d_time_per_block,
                                            ctx.h_time_per_block_mapped,
                                            0),
                   "cudaHostGetDevicePointer d_time_per_block");
    } else {
        check_cuda(cudaMalloc(&ctx.d_time_per_block,
                              ctx.total_query_capacity * 6 * sizeof(float)),
                   "cudaMalloc d_time_per_block");
    }

    printf("Allocating pinned host memory for status arrays...\n");
    check_cuda(cudaHostAlloc((void**)&ctx.status_memory, ctx.total_query_capacity * sizeof(int64_t) * 16, cudaHostAllocMapped), "cudaHostAlloc status_memory");
    check_cuda(cudaHostAlloc((void**)&ctx.status_flag, ctx.total_query_capacity * sizeof(int64_t) * 16, cudaHostAllocMapped), "cudaHostAlloc status_flag");
    check_cuda(cudaHostGetDevicePointer((void**)&ctx.d_status_memory, ctx.status_memory, 0), "cudaHostGetDevicePointer d_status_memory");
    check_cuda(cudaHostGetDevicePointer((void**)&ctx.d_status_flag, ctx.status_flag, 0), "cudaHostGetDevicePointer d_status_flag");
    if (ctx.request_ring_enabled) {
        const size_t ring_slot_count =
            ctx.request_ring_capacity_count * ctx.request_ring_capacity_per_ring;
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ctx.h_request_ring_entries),
                                 ring_slot_count * sizeof(HostRequestRingEntry),
                                 cudaHostAllocMapped),
                   "cudaHostAlloc PQ request ring entries");
        check_cuda(cudaHostGetDevicePointer(
                       reinterpret_cast<void**>(&ctx.d_request_ring_entries),
                       ctx.h_request_ring_entries,
                       0),
                   "cudaHostGetDevicePointer PQ request ring entries");
        if (ctx.request_ring_consumed_device_mapped) {
            ctx.mapped_request_ring_consumed_sequences_bytes = round_up_gdr_bytes(
                ring_slot_count * sizeof(uint64_t));
            check_cuda(cudaMalloc(
                           reinterpret_cast<void**>(&ctx.d_request_ring_consumed_sequences_raw),
                           ctx.mapped_request_ring_consumed_sequences_bytes + kGdrAlignBytes),
                       "cudaMalloc PQ request ring consumed sequences");
            ctx.d_request_ring_consumed_sequences = align_gdr_pointer(
                ctx.d_request_ring_consumed_sequences_raw);
            check_gdr(gdr_pin_buffer(
                          ctx.g,
                          reinterpret_cast<unsigned long>(ctx.d_request_ring_consumed_sequences),
                          ctx.mapped_request_ring_consumed_sequences_bytes,
                          0,
                          0,
                          &ctx.mh_request_ring_consumed_sequences),
                      "gdr_pin_buffer PQ request ring consumed sequences");
            ctx.mh_request_ring_consumed_sequences_pinned = true;
            check_gdr(gdr_map(ctx.g,
                              ctx.mh_request_ring_consumed_sequences,
                              &ctx.cpu_ptr_request_ring_consumed_sequences,
                              ctx.mapped_request_ring_consumed_sequences_bytes),
                      "gdr_map PQ request ring consumed sequences");
            ctx.h_request_ring_consumed_sequences = reinterpret_cast<uint64_t*>(
                ctx.cpu_ptr_request_ring_consumed_sequences);
        } else {
            check_cuda(cudaHostAlloc(
                           reinterpret_cast<void**>(&ctx.h_request_ring_consumed_sequences),
                           ring_slot_count * sizeof(uint64_t),
                           cudaHostAllocMapped),
                       "cudaHostAlloc PQ request ring consumed sequences");
            check_cuda(cudaHostGetDevicePointer(
                           reinterpret_cast<void**>(&ctx.d_request_ring_consumed_sequences),
                           ctx.h_request_ring_consumed_sequences,
                           0),
                       "cudaHostGetDevicePointer PQ request ring consumed sequences");
        }
        check_cuda(cudaMalloc(&ctx.d_request_ring_tails,
                              ctx.request_ring_capacity_count *
                                  sizeof(unsigned long long)),
                   "cudaMalloc PQ request ring tails");
    }

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ctx.d_neighbor_memory_raw),
                          ctx.mapped_neighbor_bytes + kGdrAlignBytes),
               "cudaMalloc d_neighbor_memory_raw");
    ctx.d_neighbor_memory = align_gdr_pointer(ctx.d_neighbor_memory_raw);
    int gdr_status = gdr_pin_buffer(ctx.g,
                                    reinterpret_cast<unsigned long>(ctx.d_neighbor_memory),
                                    ctx.mapped_neighbor_bytes,
                                    0,
                                    0,
                                    &ctx.mh_neighbor);
    if (gdr_status == 0) {
        ctx.mh_neighbor_pinned = true;
        gdr_status = gdr_map(ctx.g,
                             ctx.mh_neighbor,
                             &ctx.cpu_ptr_neighbor,
                             ctx.mapped_neighbor_bytes);
    }
    if (gdr_status == 0) {
        ctx.h_neighbor_memory = reinterpret_cast<uint32_t*>(ctx.cpu_ptr_neighbor);
    } else {
        if (ctx.mh_neighbor_pinned) {
            gdr_unpin_buffer(ctx.g, ctx.mh_neighbor);
            ctx.mh_neighbor_pinned = false;
        }
        cudaFree(ctx.d_neighbor_memory_raw);
        ctx.d_neighbor_memory_raw = nullptr;
        ctx.d_neighbor_memory = nullptr;
        check_cuda(cudaHostAlloc((void**)&ctx.h_neighbor_memory_host, ctx.mapped_neighbor_bytes, cudaHostAllocMapped), "cudaHostAlloc h_neighbor_memory_host");
        std::memset(ctx.h_neighbor_memory_host, 0, ctx.mapped_neighbor_bytes);
        check_cuda(cudaHostGetDevicePointer((void**)&ctx.d_neighbor_memory, ctx.h_neighbor_memory_host, 0), "cudaHostGetDevicePointer d_neighbor_memory");
        ctx.h_neighbor_memory = ctx.h_neighbor_memory_host;
        ctx.using_host_mapped_neighbor = true;
    }

    ctx.h_time_per_block = static_cast<float*>(malloc(ctx.total_query_capacity * 6 * sizeof(float)));
    if (ctx.h_time_per_block == nullptr) {
        fail_pq("malloc h_time_per_block failed");
    }
    if constexpr (kNativeProfileMode > 0) {
        if (ctx.native_profile_enabled) {
            ctx.h_native_profile = static_cast<PqNativeProfileRecord*>(
                malloc(ctx.total_query_capacity * sizeof(PqNativeProfileRecord)));
            if (ctx.h_native_profile == nullptr) {
                fail_pq("malloc h_native_profile failed");
            }
        }
    }

    const size_t prefetch_count = runtime.prefetch_nodes.size();
    const size_t prefetch_index_count = runtime.prefetch_index.size();
    const size_t prefetch_begin_count = runtime.prefetch_begin_by_group.size();
    const size_t prefetch_end_count = runtime.prefetch_end_by_group.size();
    const size_t label_count = runtime.metadata.label_count;
    const size_t row_to_sample_count = runtime.metadata.row_to_sample.size();
    const size_t label_offset_count = runtime.metadata.label_offset.size();
    const size_t row_filter_mask_count = runtime.metadata.row_filter_mask.size();

    if (prefetch_index_count != prefetch_count) {
        fail_pq("prefetch_index size does not match prefetch_nodes size (index=" +
                std::to_string(prefetch_index_count) + ", nodes=" +
                std::to_string(prefetch_count) + ")");
    }
    const bool missing_prefetch_group_bounds =
        (prefetch_begin_count == 0 && prefetch_end_count == 0 && label_count > 0);
    if (!missing_prefetch_group_bounds && prefetch_begin_count != label_count) {
        fail_pq("prefetch_begin_by_group size does not match metadata.label_count (prefetch_begin=" +
                std::to_string(prefetch_begin_count) + ", label_count=" +
                std::to_string(label_count) + ")");
    }
    if (!missing_prefetch_group_bounds && prefetch_end_count != label_count) {
        fail_pq("prefetch_end_by_group size does not match metadata.label_count (prefetch_end=" +
                std::to_string(prefetch_end_count) + ", label_count=" +
                std::to_string(label_count) + ")");
    }
    if (label_offset_count != label_count) {
        fail_pq("label_offset size does not match metadata.label_count (label_offset=" +
                std::to_string(label_offset_count) + ", label_count=" +
                std::to_string(label_count) + ")");
    }
    if (row_to_sample_count != runtime.metadata.total_rows) {
        fail_pq("row_to_sample size does not match metadata.total_rows (row_to_sample=" +
                std::to_string(row_to_sample_count) + ", total_rows=" +
                std::to_string(runtime.metadata.total_rows) + ")");
    }
    if (row_filter_mask_count != runtime.metadata.total_rows) {
        fail_pq("row_filter_mask size does not match metadata.total_rows (row_filter_mask=" +
                std::to_string(row_filter_mask_count) + ", total_rows=" +
                std::to_string(runtime.metadata.total_rows) + ")");
    }

    const size_t prefetch_alloc = std::max<size_t>(1, prefetch_count);
    const size_t label_alloc = std::max<size_t>(1, label_count);
    const size_t row_alloc = std::max<size_t>(1, row_to_sample_count);
    const size_t query_range_elems = query_capacity * runtime.query_ranges.range_count;
    check_cuda(cudaMalloc(&ctx.d_prefetch_node_array, prefetch_alloc * sizeof(Node)), "cudaMalloc d_prefetch_node_array");
    check_cuda(cudaMalloc(&ctx.d_prefetch_index, prefetch_alloc * sizeof(uint32_t)), "cudaMalloc d_prefetch_index");
    check_cuda(cudaMalloc(&ctx.d_prefetch_begin_by_group, label_alloc * sizeof(uint32_t)), "cudaMalloc d_prefetch_begin_by_group");
    check_cuda(cudaMalloc(&ctx.d_prefetch_end_by_group, label_alloc * sizeof(uint32_t)), "cudaMalloc d_prefetch_end_by_group");
    check_cuda(cudaMalloc(&ctx.d_query_group_ids, query_capacity * sizeof(uint32_t)), "cudaMalloc d_query_group_ids");
    check_cuda(cudaMalloc(&ctx.d_start_local_ids, query_capacity * sizeof(uint32_t)), "cudaMalloc d_start_local_ids");
    if (ctx.use_sample_id_payload) {
        check_cuda(cudaMalloc(&ctx.d_start_sample_ids,
                              query_capacity * sizeof(uint32_t)),
                   "cudaMalloc d_start_sample_ids");
    } else {
        check_cuda(cudaMalloc(&ctx.d_row_to_sample, row_alloc * sizeof(uint32_t)), "cudaMalloc d_row_to_sample");
    }
    check_cuda(cudaMalloc(&ctx.d_label_offset, label_alloc * sizeof(uint32_t)), "cudaMalloc d_label_offset");
    if (!ctx.use_implicit_filter_mask) {
        check_cuda(cudaMalloc(&ctx.d_row_filter_mask, row_alloc * sizeof(uint64_t)), "cudaMalloc d_row_filter_mask");
    }
    check_cuda(cudaMalloc(&ctx.d_query_filter_bits, query_capacity * sizeof(uint64_t)), "cudaMalloc d_query_filter_bits");
    check_cuda(cudaMalloc(&ctx.d_query_range_modes, query_range_elems * sizeof(uint8_t)), "cudaMalloc d_query_range_modes");
    check_cuda(cudaMalloc(&ctx.d_query_range_lower_bounds, query_range_elems * sizeof(float)), "cudaMalloc d_query_range_lower_bounds");
    check_cuda(cudaMalloc(&ctx.d_query_range_upper_bounds, query_range_elems * sizeof(float)), "cudaMalloc d_query_range_upper_bounds");
    if (!runtime.base_range_values.empty()) {
        check_cuda(cudaMalloc(&ctx.d_base_range_values,
                              runtime.base_range_values.size() * sizeof(float)),
                   "cudaMalloc d_base_range_values");
    }
    if (ctx.device_resident_graph) {
        if (runtime.graph.data.empty() ||
            runtime.graph.rows != runtime.metadata.total_rows ||
            runtime.graph.cols == 0 || runtime.graph.cols > FIXED_DEGREE) {
            fail_pq("device_resident PQ requires a materialized graph with degree in [1, 64]");
        }
        if (runtime.base_vectors.data.empty() ||
            runtime.base_vectors.rows == 0 ||
            runtime.base_vectors.cols != static_cast<uint32_t>(options.dim)) {
            fail_pq("device_resident PQ requires materialized uint8 base vectors");
        }
        if (runtime.data_labels.indptr.size() !=
                static_cast<size_t>(runtime.data_labels.rows) + 1 ||
            runtime.data_labels.rows != runtime.base_vectors.rows) {
            fail_pq("device_resident PQ requires aligned data-label CSR");
        }
        if (runtime.query_labels_matrix.indptr.size() !=
                static_cast<size_t>(runtime.query_labels_matrix.rows) + 1 ||
            runtime.query_labels_matrix.rows != query_capacity ||
            runtime.query_filter_labels.size() != query_capacity) {
            fail_pq("device_resident PQ requires aligned routed-query label CSR");
        }

        check_cuda(cudaMalloc(&ctx.d_resident_graph,
                              runtime.graph.data.size() * sizeof(uint32_t)),
                   "cudaMalloc resident PQ graph");
        check_cuda(cudaMalloc(&ctx.d_resident_base_vectors,
                              runtime.base_vectors.data.size() * sizeof(uint8_t)),
                   "cudaMalloc resident PQ base vectors");
        check_cuda(cudaMalloc(&ctx.d_resident_data_label_indptr,
                              runtime.data_labels.indptr.size() * sizeof(uint64_t)),
                   "cudaMalloc resident PQ data-label indptr");
        check_cuda(cudaMalloc(&ctx.d_resident_data_label_indices,
                              std::max<size_t>(1, runtime.data_labels.indices.size()) *
                                  sizeof(uint32_t)),
                   "cudaMalloc resident PQ data-label indices");
        check_cuda(cudaMalloc(&ctx.d_resident_query_label_indptr,
                              runtime.query_labels_matrix.indptr.size() * sizeof(uint64_t)),
                   "cudaMalloc resident PQ query-label indptr");
        check_cuda(cudaMalloc(&ctx.d_resident_query_label_indices,
                              std::max<size_t>(1, runtime.query_labels_matrix.indices.size()) *
                                  sizeof(uint32_t)),
                   "cudaMalloc resident PQ query-label indices");
        check_cuda(cudaMalloc(&ctx.d_resident_query_filter_labels,
                              query_capacity * sizeof(uint32_t)),
                   "cudaMalloc resident PQ query primary labels");
    }

    if (prefetch_count > 0) {
        check_cuda(cudaMemcpy(ctx.d_prefetch_node_array, runtime.prefetch_nodes.data(), prefetch_count * sizeof(Node), cudaMemcpyHostToDevice), "cudaMemcpy d_prefetch_node_array");
        check_cuda(cudaMemcpy(ctx.d_prefetch_index, runtime.prefetch_index.data(), prefetch_index_count * sizeof(uint32_t), cudaMemcpyHostToDevice), "cudaMemcpy d_prefetch_index");
    }
    if (prefetch_begin_count > 0) {
        check_cuda(cudaMemcpy(ctx.d_prefetch_begin_by_group,
                              runtime.prefetch_begin_by_group.data(),
                              prefetch_begin_count * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_prefetch_begin_by_group");
        check_cuda(cudaMemcpy(ctx.d_prefetch_end_by_group,
                              runtime.prefetch_end_by_group.data(),
                              prefetch_end_count * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_prefetch_end_by_group");
    } else {
        check_cuda(cudaMemset(ctx.d_prefetch_begin_by_group, 0, label_alloc * sizeof(uint32_t)),
                   "cudaMemset d_prefetch_begin_by_group");
        check_cuda(cudaMemset(ctx.d_prefetch_end_by_group, 0, label_alloc * sizeof(uint32_t)),
                   "cudaMemset d_prefetch_end_by_group");
    }
    if (ctx.d_start_sample_ids != nullptr) {
        const std::vector<uint32_t> start_sample_ids = build_start_sample_ids(runtime);
        if (start_sample_ids.size() != query_capacity) {
            fail_pq("start_sample_ids size does not match query capacity");
        }
        check_cuda(cudaMemcpy(ctx.d_start_sample_ids,
                              start_sample_ids.data(),
                              start_sample_ids.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_start_sample_ids");
    }
    if (ctx.d_row_to_sample != nullptr && row_to_sample_count > 0) {
        check_cuda(cudaMemcpy(ctx.d_row_to_sample,
                              runtime.metadata.row_to_sample.data(),
                              row_to_sample_count * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_row_to_sample");
    }
    if (label_offset_count > 0) {
        check_cuda(cudaMemcpy(ctx.d_label_offset,
                              runtime.metadata.label_offset.data(),
                              label_offset_count * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_label_offset");
    }
    if (ctx.d_row_filter_mask != nullptr && row_filter_mask_count > 0) {
        check_cuda(cudaMemcpy(ctx.d_row_filter_mask,
                              runtime.metadata.row_filter_mask.data(),
                              row_filter_mask_count * sizeof(uint64_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_row_filter_mask");
    }
    if (ctx.d_base_range_values != nullptr) {
        check_cuda(cudaMemcpy(ctx.d_base_range_values,
                              runtime.base_range_values.data(),
                              runtime.base_range_values.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_base_range_values");
    }
    if (ctx.device_resident_graph) {
        check_cuda(cudaMemcpy(ctx.d_resident_graph,
                              runtime.graph.data.data(),
                              runtime.graph.data.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy resident PQ graph");
        check_cuda(cudaMemcpy(ctx.d_resident_base_vectors,
                              runtime.base_vectors.data.data(),
                              runtime.base_vectors.data.size() * sizeof(uint8_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy resident PQ base vectors");
        check_cuda(cudaMemcpy(ctx.d_resident_data_label_indptr,
                              runtime.data_labels.indptr.data(),
                              runtime.data_labels.indptr.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy resident PQ data-label indptr");
        if (!runtime.data_labels.indices.empty()) {
            check_cuda(cudaMemcpy(ctx.d_resident_data_label_indices,
                                  runtime.data_labels.indices.data(),
                                  runtime.data_labels.indices.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy resident PQ data-label indices");
        }
        check_cuda(cudaMemcpy(ctx.d_resident_query_label_indptr,
                              runtime.query_labels_matrix.indptr.data(),
                              runtime.query_labels_matrix.indptr.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy resident PQ query-label indptr");
        if (!runtime.query_labels_matrix.indices.empty()) {
            check_cuda(cudaMemcpy(ctx.d_resident_query_label_indices,
                                  runtime.query_labels_matrix.indices.data(),
                                  runtime.query_labels_matrix.indices.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy resident PQ query-label indices");
        }
        check_cuda(cudaMemcpy(ctx.d_resident_query_filter_labels,
                              runtime.query_filter_labels.data(),
                              query_capacity * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy resident PQ query primary labels");
        printf("[pq] graph transport=device_resident rows=%llu degree=%u "
               "graph_bytes=%zu vector_bytes=%zu rerank=batched_gpu_exact\n",
               static_cast<unsigned long long>(runtime.graph.rows),
               ctx.resident_graph_degree,
               runtime.graph.data.size() * sizeof(uint32_t),
               runtime.base_vectors.data.size() * sizeof(uint8_t));
    }
    if (runtime.base_range_count != runtime.query_ranges.range_count) {
        fail_pq("query/base range dimension mismatch");
    }

    check_cuda(cudaEventCreate(&ctx.start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&ctx.stop), "cudaEventCreate stop");
    if (ctx.predicate) ctx.predicate->view.base_ranges = ctx.d_base_range_values;
    rollback.committed = true;
}

void init_graph_context_prepared(PqGraphSearchContext& ctx,
                                 size_t total_query_capacity,
                                 const PqImmutableDeviceState& immutable_state,
                                 const PqPreparedQueryState& prepared_state,
                                 const GraphRuntime& runtime,
                                 const PqSearchOptions& options)
{
    init_graph_context(ctx,
                       total_query_capacity,
                       immutable_state,
                       runtime,
                       options);
    if (ctx.d_query != nullptr) {
        cudaFree(ctx.d_query);
        ctx.d_query = nullptr;
    }
    if (ctx.d_query_group_ids != nullptr) {
        cudaFree(ctx.d_query_group_ids);
        ctx.d_query_group_ids = nullptr;
    }
    if (ctx.d_start_local_ids != nullptr) {
        cudaFree(ctx.d_start_local_ids);
        ctx.d_start_local_ids = nullptr;
    }
    if (ctx.d_query_filter_bits != nullptr) {
        cudaFree(ctx.d_query_filter_bits);
        ctx.d_query_filter_bits = nullptr;
    }
    if (ctx.d_query_range_modes != nullptr) {
        cudaFree(ctx.d_query_range_modes);
        ctx.d_query_range_modes = nullptr;
    }
    if (ctx.d_query_range_lower_bounds != nullptr) {
        cudaFree(ctx.d_query_range_lower_bounds);
        ctx.d_query_range_lower_bounds = nullptr;
    }
    if (ctx.d_query_range_upper_bounds != nullptr) {
        cudaFree(ctx.d_query_range_upper_bounds);
        ctx.d_query_range_upper_bounds = nullptr;
    }
    ctx.owns_query_inputs = false;
    ctx.owns_query_metadata = false;
    ctx.d_query = prepared_state.d_all_query;
    ctx.d_query_group_ids = prepared_state.d_query_group_ids;
    ctx.d_start_local_ids = prepared_state.d_start_local_ids;
    ctx.d_query_filter_bits = prepared_state.d_query_filter_bits;
    ctx.d_query_range_modes = prepared_state.d_query_range_modes;
    ctx.d_query_range_lower_bounds = prepared_state.d_query_range_lower_bounds;
    ctx.d_query_range_upper_bounds = prepared_state.d_query_range_upper_bounds;
}

void reset_search_state(PqGraphSearchContext& ctx, size_t query_count)
{
    if (ctx.status_memory == nullptr || ctx.status_flag == nullptr || ctx.d_gdr_data == nullptr) {
        return;
    }
    const size_t reset_query_count = std::min(query_count, ctx.total_query_capacity);
    std::fill(ctx.status_memory,
              ctx.status_memory + reset_query_count * 16,
              static_cast<int64_t>(-2));
    std::fill(ctx.status_flag,
              ctx.status_flag + reset_query_count * 16,
              static_cast<int64_t>(-2));
    if (ctx.result_buffer_is_host_mapped) {
        std::fill(ctx.h_result_mapped,
                  ctx.h_result_mapped + reset_query_count * ctx.result_width,
                  UINT32_MAX);
        std::fill(ctx.h_result_dist_mapped,
                  ctx.h_result_dist_mapped + reset_query_count * ctx.result_width,
                  FLT_MAX);
    }
    if (ctx.time_buffer_is_host_mapped && ctx.h_time_per_block_mapped != nullptr) {
        std::fill(ctx.h_time_per_block_mapped,
                  ctx.h_time_per_block_mapped + reset_query_count * 6,
                  0.0f);
    }
    check_cuda(cudaMemset(ctx.d_gdr_data,
                          0,
                          reset_query_count * sizeof(int64_t) * kGdrMailboxStride),
               "cudaMemset d_gdr_data");
    if constexpr (kNativeProfileMode > 0) {
        if (ctx.d_native_profile != nullptr) {
            check_cuda(cudaMemset(ctx.d_native_profile,
                                  0,
                                  reset_query_count * sizeof(PqNativeProfileRecord)),
                       "cudaMemset d_native_profile");
        }
    }
    if (ctx.request_ring_enabled) {
        const size_t ring_slot_count =
            ctx.request_ring_capacity_count * ctx.request_ring_capacity_per_ring;
        std::fill(ctx.h_request_ring_entries,
                  ctx.h_request_ring_entries + ring_slot_count,
                  HostRequestRingEntry{});
        std::fill(ctx.h_request_ring_consumed_sequences,
                  ctx.h_request_ring_consumed_sequences + ring_slot_count,
                  0ULL);
        check_cuda(cudaMemset(ctx.d_request_ring_tails,
                              0,
                              ctx.request_ring_capacity_count *
                                  sizeof(unsigned long long)),
                   "cudaMemset PQ request ring tails");
    }
    reset_finite_query_state(ctx, reset_query_count);
    _mm_sfence();
    ctx.host_filter_stats = HostFilterStats{};
}

PqFiniteKernelParams make_finite_kernel_params(PqGraphSearchContext& ctx,
                                               const GraphRuntime& runtime,
                                               const PqSearchOptions& options)
{
    PqFiniteKernelParams params{};
    params.d_data = ctx.d_data;
    params.d_query = ctx.d_query;
    params.d_result = ctx.d_result;
    params.d_result_dist = ctx.d_result_dist;
    params.d_pq_centroid = ctx.d_pq_centroid;
    params.d_pq_table = ctx.d_pq_table;
    params.d_hashdata = ctx.d_hashdata;
    params.time_per_block = ctx.d_time_per_block;
    params.d_neighbor_memory = ctx.d_neighbor_memory;
    params.d_status_memory = ctx.d_status_memory;
    params.d_gdr_data = ctx.d_gdr_data;
    params.d_status_flag = ctx.d_status_flag;
    params.d_prefetch_node_array = ctx.d_prefetch_node_array;
    params.d_prefetch_index = ctx.d_prefetch_index;
    params.d_prefetch_begin_by_group = ctx.d_prefetch_begin_by_group;
    params.d_prefetch_end_by_group = ctx.d_prefetch_end_by_group;
    params.d_query_group_ids = ctx.d_query_group_ids;
    params.d_start_local_ids = ctx.d_start_local_ids;
    params.d_start_sample_ids = ctx.d_start_sample_ids;
    params.d_row_to_sample = ctx.d_row_to_sample;
    params.d_label_offset = ctx.d_label_offset;
    params.d_row_filter_mask = ctx.d_row_filter_mask;
    params.d_query_filter_bits = ctx.d_query_filter_bits;
    params.d_query_range_modes = ctx.d_query_range_modes;
    params.d_query_range_lower_bounds = ctx.d_query_range_lower_bounds;
    params.d_query_range_upper_bounds = ctx.d_query_range_upper_bounds;
    params.d_base_range_values = ctx.d_base_range_values;
    params.d_contexts = ctx.d_finite_contexts;
    params.d_context_telemetry = ctx.d_finite_context_telemetry;
    params.d_frontier_state = ctx.d_finite_frontier_state;
    params.d_stop_state = ctx.d_finite_stop_state;
    params.d_result_state = ctx.d_finite_result_state;
    params.query_capacity = static_cast<uint32_t>(ctx.total_query_capacity);
    params.range_count = runtime.base_range_count;
    params.query_dim = options.dim;
    params.query_pq_dim = options.pq_dim;
    params.queue_limit = options.search_queue_size;
    params.finish_limit = options.search_finish_count;
    params.early_stop_mode = parse_search_early_stop_mode(options.search_early_stop_mode);
    params.max_steps = options.search_max_steps;
    params.stop_width = static_cast<int>(ctx.stop_width);
    params.result_width = static_cast<int>(ctx.result_width);
    params.mailbox_sample_id_word = static_cast<uint32_t>(ctx.io_config.vector_payload_words);
    params.mailbox_neighbor_count_word = static_cast<uint32_t>(ctx.io_config.neighbor_count_word);
    params.mailbox_neighbor_words_offset = static_cast<uint32_t>(ctx.io_config.neighbor_words_offset);
    params.mailbox_neighbor_sample_words_offset =
        static_cast<uint32_t>(ctx.io_config.neighbor_sample_words_offset);
    params.mailbox_slot_words = static_cast<uint32_t>(ctx.io_config.elems_per_block);
    params.use_sample_id_payload = ctx.use_sample_id_payload ? 1 : 0;
    params.use_implicit_filter_mask = ctx.use_implicit_filter_mask ? 1 : 0;
    params.enable_true_distance_rerank =
        highfreq_gpu_exact_rerank_enabled(options) && !ctx.device_resident_graph ? 1 : 0;
    params.bit_filter_bonus = options.bit_filter_bonus;
    params.range_filter_bonus = options.range_filter_bonus;
    params.debug_mailbox = pq_mailbox_trace_enabled() ? 1 : 0;
    return params;
}

uint32_t launch_finite_init(PqGraphSearchContext& ctx,
                            const GraphRuntime& runtime,
                            const PqSearchOptions& options,
                            const uint32_t* d_query_ids,
                            size_t query_count)
{
    if (++ctx.finite_context_epoch == 0) ++ctx.finite_context_epoch;
    const uint32_t epoch = ctx.finite_context_epoch;
    const PqFiniteKernelParams params = make_finite_kernel_params(ctx, runtime, options);
    const int grid_dim = static_cast<int>((query_count + kMultiQuery - 1) / kMultiQuery);
    dispatch_graph_kernel(options.search_queue_size, options.pq_dim, [&]<int QS, int PD>() {
        finite_pq_init_kernel<QS, PD>
            <<<grid_dim, kNativeThreadInBlock, ctx.finite_batch_dynamic_shared_bytes>>>(
                params,
                d_query_ids,
                static_cast<uint32_t>(query_count),
                epoch);
    });
    check_cuda(cudaGetLastError(), "finite PQ init kernel launch");
    return epoch;
}

int64_t load_finite_mapped_word(const int64_t* base, size_t offset)
{
    return *reinterpret_cast<const volatile int64_t*>(base + offset);
}

struct PqFiniteMailboxSnapshot {
    int64_t flag = 0;
    int64_t row = 0;
    int64_t buffer = 0;
    int64_t generation = 0;
    int64_t ready = 0;

    bool operator==(const PqFiniteMailboxSnapshot& other) const
    {
        return flag == other.flag && row == other.row && buffer == other.buffer &&
               generation == other.generation && ready == other.ready;
    }
};

PqFiniteMailboxSnapshot load_finite_mailbox_snapshot(const PqGraphSearchContext& ctx,
                                                     uint32_t query_id)
{
    const size_t status_base = static_cast<size_t>(query_id) * 16;
    PqFiniteMailboxSnapshot snapshot;
    snapshot.flag = load_finite_mapped_word(ctx.status_flag, status_base);
    snapshot.row = load_finite_mapped_word(ctx.status_memory, status_base);
    snapshot.buffer = load_finite_mapped_word(ctx.status_memory, status_base + 1);
    snapshot.generation = load_finite_mapped_word(ctx.status_memory, status_base + 2);
    snapshot.ready = load_finite_mapped_word(
        ctx.h_gdr_data, static_cast<size_t>(query_id) * kGdrMailboxStride);
    std::atomic_thread_fence(std::memory_order_acquire);
    return snapshot;
}

const char* finite_advance_result_name(PqFiniteAdvanceResult result)
{
    switch (result) {
        case PqFiniteAdvanceResult::PENDING: return "pending";
        case PqFiniteAdvanceResult::PARKED: return "parked";
        case PqFiniteAdvanceResult::DONE: return "done";
        case PqFiniteAdvanceResult::REJECT_QUERY: return "reject_query";
        case PqFiniteAdvanceResult::REJECT_EPOCH: return "reject_epoch";
        case PqFiniteAdvanceResult::REJECT_STATE: return "reject_state";
        case PqFiniteAdvanceResult::REJECT_GENERATION: return "reject_generation";
        case PqFiniteAdvanceResult::REJECT_ROW: return "reject_row";
        case PqFiniteAdvanceResult::REJECT_BUFFER: return "reject_buffer";
        case PqFiniteAdvanceResult::REJECT_STATUS: return "reject_status";
        case PqFiniteAdvanceResult::REJECT_MAILBOX: return "reject_mailbox";
    }
    return "unknown";
}

PqFiniteReadyRingSink::PqFiniteReadyRingSink(
    const std::vector<uint32_t>& active_query_ids,
    size_t query_capacity)
    : active_queries_(query_capacity, 0),
      query_states_(std::make_unique<std::atomic<uint64_t>[]>(query_capacity)),
      active_query_count_(active_query_ids.size())
{
    if (active_query_ids.empty()) {
        fail_pq("finite ready ring requires at least one active query");
    }
    require_valid_finite_query_ids(active_query_ids, query_capacity);
    for (size_t query_id = 0; query_id < query_capacity; ++query_id) {
        query_states_[query_id].store(0, std::memory_order_relaxed);
    }
    for (uint32_t query_id : active_query_ids) active_queries_[query_id] = 1;

    try {
        queue_.ring_capacity = static_cast<uint32_t>(active_query_ids.size());
        const size_t storage_bytes = sizeof(PqFiniteReadyRingControl) +
            static_cast<size_t>(queue_.ring_capacity) * sizeof(PqFiniteReadyRingEntry);
        queue_.mapped_bytes = round_up_gdr_bytes(storage_bytes);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&queue_.d_storage_raw),
                              queue_.mapped_bytes + kGdrAlignBytes),
                   "cudaMalloc PQ finite ready ring storage");
        queue_.d_control = align_gdr_pointer(
            reinterpret_cast<PqFiniteReadyRingControl*>(queue_.d_storage_raw));
        queue_.d_entries = reinterpret_cast<PqFiniteReadyRingEntry*>(
            reinterpret_cast<uint8_t*>(queue_.d_control) + sizeof(PqFiniteReadyRingControl));
        queue_.g = gdr_open();
        if (queue_.g == nullptr) fail_pq("gdr_open returned nullptr for PQ finite ready ring");
        check_gdr(gdr_pin_buffer(queue_.g,
                                 reinterpret_cast<unsigned long>(queue_.d_control),
                                 queue_.mapped_bytes,
                                 0,
                                 0,
                                 &queue_.mh),
                  "gdr_pin_buffer PQ finite ready ring");
        queue_.pinned = true;
        check_gdr(gdr_map(queue_.g,
                          queue_.mh,
                          &queue_.cpu_ptr,
                          queue_.mapped_bytes),
                  "gdr_map PQ finite ready ring");
        queue_.h_control = reinterpret_cast<PqFiniteReadyRingControl*>(queue_.cpu_ptr);
        queue_.h_entries = reinterpret_cast<PqFiniteReadyRingEntry*>(
            reinterpret_cast<uint8_t*>(queue_.cpu_ptr) + sizeof(PqFiniteReadyRingControl));
        *queue_.h_control = PqFiniteReadyRingControl{};
        for (uint32_t entry_idx = 0; entry_idx < queue_.ring_capacity; ++entry_idx) {
            queue_.h_entries[entry_idx] = PqFiniteReadyRingEntry{};
        }
        _mm_sfence();

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&queue_.d_consumer_head),
                              sizeof(*queue_.d_consumer_head)),
                   "cudaMalloc PQ finite ready ring consumer head");
        check_cuda(cudaMemset(queue_.d_consumer_head,
                              0,
                              sizeof(*queue_.d_consumer_head)),
                   "cudaMemset PQ finite ready ring consumer head");
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue_.h_consumed_sequences),
                                 static_cast<size_t>(queue_.ring_capacity) *
                                     sizeof(*queue_.h_consumed_sequences),
                                 cudaHostAllocMapped),
                   "cudaHostAlloc PQ finite ready ring consumed sequences");
        std::fill(queue_.h_consumed_sequences,
                  queue_.h_consumed_sequences + queue_.ring_capacity,
                  0ULL);
        check_cuda(cudaHostGetDevicePointer(
                       reinterpret_cast<void**>(&queue_.d_consumed_sequences),
                       queue_.h_consumed_sequences,
                       0),
                   "cudaHostGetDevicePointer PQ finite ready ring consumed sequences");
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue_.h_status),
                                 sizeof(PqFiniteReadyRingStatus),
                                 cudaHostAllocMapped),
                   "cudaHostAlloc PQ finite ready ring status");
        *queue_.h_status = PqFiniteReadyRingStatus{};
        check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue_.d_status),
                                            queue_.h_status,
                                            0),
                   "cudaHostGetDevicePointer PQ finite ready ring status");
        check_cuda(cudaStreamCreateWithFlags(&queue_.stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags PQ finite ready ring");
    } catch (...) {
        destroy();
        throw;
    }
}

PqFiniteReadyRingSink::~PqFiniteReadyRingSink()
{
    destroy();
}

void PqFiniteReadyRingSink::destroy() noexcept
{
    if (queue_.launched && !queue_.stopped && queue_.h_control != nullptr) {
        queue_.h_control->stop = 1;
        _mm_sfence();
        if (queue_.stream != nullptr) cudaStreamSynchronize(queue_.stream);
    }
    if (queue_.stream != nullptr) cudaStreamDestroy(queue_.stream);
    if (queue_.g != nullptr && queue_.cpu_ptr != nullptr) {
        gdr_unmap(queue_.g, queue_.mh, queue_.cpu_ptr, queue_.mapped_bytes);
    }
    if (queue_.g != nullptr && queue_.pinned) gdr_unpin_buffer(queue_.g, queue_.mh);
    if (queue_.g != nullptr) gdr_close(queue_.g);
    if (queue_.d_storage_raw != nullptr) cudaFree(queue_.d_storage_raw);
    if (queue_.d_consumer_head != nullptr) cudaFree(queue_.d_consumer_head);
    if (queue_.d_per_hop_timing != nullptr) cudaFree(queue_.d_per_hop_timing);
    if (queue_.h_consumed_sequences != nullptr) {
        cudaFreeHost(queue_.h_consumed_sequences);
    }
    if (queue_.h_status != nullptr) cudaFreeHost(queue_.h_status);
    queue_ = PqFiniteReadyRingQueue{};
}

bool PqFiniteReadyRingSink::query_is_active(uint32_t query_id) const noexcept
{
    return query_id < active_queries_.size() && active_queries_[query_id] != 0;
}

void PqFiniteReadyRingSink::cancel() noexcept
{
    cancelled_.store(true, std::memory_order_release);
    if (queue_.h_control != nullptr) {
        queue_.h_control->stop = 1;
        _mm_sfence();
    }
    terminal_changed_.notify_all();
}

bool PqFiniteReadyRingSink::publish_ready(uint32_t query_id,
                                          uint32_t generation,
                                          uint32_t row,
                                          uint32_t buffer)
{
    if (!query_is_active(query_id) || generation == 0 || row == UINT32_MAX || buffer > 1) {
        invalid_rejections_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (cancelled_.load(std::memory_order_acquire)) {
        cancelled_rejections_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    uint32_t context_epoch = context_epoch_.load(std::memory_order_acquire);
    while (context_epoch == 0) {
        if (cancelled_.load(std::memory_order_acquire)) {
            cancelled_rejections_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        std::this_thread::yield();
        context_epoch = context_epoch_.load(std::memory_order_acquire);
    }
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
    constexpr uint32_t kHostPublishSampleMask = (1U << 12) - 1;
    const bool sample_host_publish =
        ((query_id * 2654435761U) ^ generation) % (kHostPublishSampleMask + 1) == 0;
    const uint64_t host_publish_start_cycles = sample_host_publish ? read_ordered_tsc() : 0;
#endif

    std::atomic<uint64_t>& query_state = query_states_[query_id];
    while (true) {
        const PqFiniteReadyRingClaimResult claim =
            finite_ready_ring_try_claim_ready(query_state, generation);
        if (claim == PqFiniteReadyRingClaimResult::CLAIMED) break;
        if (claim == PqFiniteReadyRingClaimResult::TERMINAL) {
            terminal_rejections_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (claim == PqFiniteReadyRingClaimResult::DUPLICATE) {
            duplicate_rejections_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (cancelled_.load(std::memory_order_acquire)) {
            cancelled_rejections_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        std::this_thread::yield();
    }

    if (cancelled_.load(std::memory_order_acquire)) {
        finite_ready_ring_finish_ready(query_state, generation);
        cancelled_rejections_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const uint64_t ticket = producer_tail_.fetch_add(1, std::memory_order_relaxed);
    const uint32_t entry_idx = static_cast<uint32_t>(ticket % queue_.ring_capacity);
    const uint64_t expected_consumed_sequence =
        finite_ready_ring_required_consumed_sequence(ticket, queue_.ring_capacity);
    if (expected_consumed_sequence != 0) {
        bool waited = false;
        while (*reinterpret_cast<volatile unsigned long long*>(
                   queue_.h_consumed_sequences + entry_idx) !=
               expected_consumed_sequence) {
            if (!waited) {
                backpressure_waits_.fetch_add(1, std::memory_order_relaxed);
                waited = true;
            }
            if (cancelled_.load(std::memory_order_acquire)) {
                finite_ready_ring_finish_ready(query_state, generation);
                cancelled_rejections_.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            _mm_pause();
        }
        std::atomic_thread_fence(std::memory_order_acquire);
    }
    PqFiniteReadyRingEntry& entry = queue_.h_entries[entry_idx];
    entry.token.query_id = query_id;
    entry.token.context_epoch = context_epoch;
    entry.token.generation = generation;
    entry.token.row = row;
    entry.token.buffer = buffer;
    entry.token.result = PqFiniteAdvanceResult::PENDING;
    entry.token.ready_ns = 0;
    _mm_sfence();
    entry.publish_sequence = ticket + 1;
    _mm_sfence();
    finite_ready_ring_finish_ready(query_state, generation);
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
    if (sample_host_publish) {
        host_publish_sample_cycles_.fetch_add(
            read_ordered_tsc() - host_publish_start_cycles, std::memory_order_relaxed);
        host_publish_sample_count_.fetch_add(1, std::memory_order_relaxed);
    }
#endif
    return true;
}

bool PqFiniteReadyRingSink::publish_done(uint32_t query_id)
{
    if (!query_is_active(query_id)) {
        invalid_rejections_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (cancelled_.load(std::memory_order_acquire)) {
        cancelled_rejections_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    std::atomic<uint64_t>& query_state = query_states_[query_id];
    while (true) {
        const PqFiniteReadyRingClaimResult claim =
            finite_ready_ring_try_claim_done(query_state);
        if (claim == PqFiniteReadyRingClaimResult::CLAIMED) break;
        if (claim == PqFiniteReadyRingClaimResult::TERMINAL) {
            duplicate_rejections_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (cancelled_.load(std::memory_order_acquire)) {
            cancelled_rejections_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        std::this_thread::yield();
    }
    done_enqueued_.fetch_add(1, std::memory_order_release);
    terminal_changed_.notify_all();
    return true;
}

void PqFiniteReadyRingSink::set_context_epoch(uint32_t context_epoch)
{
    if (context_epoch == 0) fail_pq("finite ready ring context epoch must be nonzero");
    uint32_t expected = 0;
    if (!context_epoch_.compare_exchange_strong(
            expected, context_epoch, std::memory_order_release, std::memory_order_relaxed)) {
        fail_pq("finite ready ring context epoch was already assigned");
    }
}

void PqFiniteReadyRingSink::launch(const PqFiniteKernelParams& params,
                                   const PqSearchOptions& options,
                                   size_t dynamic_shared_bytes)
{
    if (queue_.launched) fail_pq("finite ready ring worker was launched twice");
    dispatch_graph_kernel(options.search_queue_size, options.pq_dim, [&]<int QS, int PD>() {
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &queue_.active_blocks_per_sm,
                       persistent_finite_pq_ready_ring_kernel<QS, PD>,
                       kNativeThreadInBlock,
                       dynamic_shared_bytes),
                   "cudaOccupancyMaxActiveBlocksPerMultiprocessor PQ finite ready ring");
        cudaDeviceProp prop{};
        int device_id = 0;
        check_cuda(cudaGetDevice(&device_id), "cudaGetDevice PQ finite ready ring");
        check_cuda(cudaGetDeviceProperties(&prop, device_id),
                   "cudaGetDeviceProperties PQ finite ready ring");
        queue_.multiprocessor_count = prop.multiProcessorCount;
        if (queue_.active_blocks_per_sm <= 0 || queue_.multiprocessor_count <= 0) {
            fail_pq("finite ready ring has zero resident worker capacity");
        }
        queue_.resident_blocks = static_cast<uint32_t>(queue_.active_blocks_per_sm) *
                                 static_cast<uint32_t>(queue_.multiprocessor_count);
        queue_.worker_slots = queue_.resident_blocks * static_cast<uint32_t>(kMultiQuery);
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&queue_.d_per_hop_timing),
                              static_cast<size_t>(queue_.worker_slots) *
                                  sizeof(PqFinitePerHopTiming)),
                   "cudaMalloc PQ finite per-hop timing");
        check_cuda(cudaMemset(queue_.d_per_hop_timing,
                              0,
                              static_cast<size_t>(queue_.worker_slots) *
                                  sizeof(PqFinitePerHopTiming)),
                   "cudaMemset PQ finite per-hop timing");
#endif
        persistent_finite_pq_ready_ring_kernel<QS, PD>
            <<<queue_.resident_blocks,
               kNativeThreadInBlock,
               dynamic_shared_bytes,
               queue_.stream>>>(params,
                                queue_.d_control,
                                queue_.d_entries,
                                queue_.ring_capacity,
                                queue_.d_consumer_head,
                                queue_.d_consumed_sequences,
                                queue_.d_status
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
                                , queue_.d_per_hop_timing
#endif
                                );
    });
    check_cuda(cudaGetLastError(), "launch PQ finite ready ring worker");
    queue_.launched = true;
}

uint64_t PqFiniteReadyRingSink::stop()
{
    if (!queue_.launched || queue_.stopped) return 0;
    const auto begin = std::chrono::steady_clock::now();
    queue_.h_control->stop = 1;
    _mm_sfence();
    check_cuda(cudaStreamSynchronize(queue_.stream),
               "cudaStreamSynchronize PQ finite ready ring stop");
    queue_.stopped = true;
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count());
}

bool PqFiniteReadyRingSink::wait_for_terminal_change(std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(terminal_mutex_);
    return terminal_changed_.wait_for(lock, timeout) != std::cv_status::timeout;
}

uint32_t PqFiniteReadyRingSink::error_result() const noexcept
{
    const uint32_t result = *reinterpret_cast<volatile uint32_t*>(
        &queue_.h_status->error_result);
    std::atomic_thread_fence(std::memory_order_acquire);
    return result;
}

uint32_t PqFiniteReadyRingSink::error_query_id() const noexcept
{
    return *reinterpret_cast<volatile uint32_t*>(&queue_.h_status->error_query_id);
}

uint32_t PqFiniteReadyRingSink::error_generation() const noexcept
{
    return *reinterpret_cast<volatile uint32_t*>(&queue_.h_status->error_generation);
}

uint64_t PqFiniteReadyRingSink::error_ticket() const noexcept
{
    return *reinterpret_cast<volatile uint64_t*>(&queue_.h_status->error_ticket);
}

size_t PqFiniteReadyRingSink::active_query_count() const noexcept
{
    return active_query_count_;
}

size_t PqFiniteReadyRingSink::done_count() const noexcept
{
    return static_cast<size_t>(done_enqueued_.load(std::memory_order_acquire));
}

uint64_t PqFiniteReadyRingSink::published_ready_count() const noexcept
{
    return producer_tail_.load(std::memory_order_relaxed);
}

uint64_t PqFiniteReadyRingSink::observed_consumed_sequence_max() const noexcept
{
    uint64_t max_sequence = 0;
    for (uint32_t entry_idx = 0; entry_idx < queue_.ring_capacity; ++entry_idx) {
        const uint64_t sequence = *reinterpret_cast<volatile unsigned long long*>(
            queue_.h_consumed_sequences + entry_idx);
        max_sequence = std::max(max_sequence, sequence);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    return max_sequence;
}

uint64_t PqFiniteReadyRingSink::host_publish_sample_count() const noexcept
{
    return host_publish_sample_count_.load(std::memory_order_relaxed);
}

uint64_t PqFiniteReadyRingSink::host_publish_sample_cycles() const noexcept
{
    return host_publish_sample_cycles_.load(std::memory_order_relaxed);
}

PqFiniteCompletionQueueStats PqFiniteReadyRingSink::stats() const noexcept
{
    PqFiniteCompletionQueueStats result;
    result.ready_enqueued = producer_tail_.load(std::memory_order_relaxed);
    result.done_enqueued = done_enqueued_.load(std::memory_order_relaxed);
    result.duplicate_rejections = duplicate_rejections_.load(std::memory_order_relaxed);
    result.invalid_rejections = invalid_rejections_.load(std::memory_order_relaxed);
    result.terminal_rejections = terminal_rejections_.load(std::memory_order_relaxed);
    result.cancelled_rejections = cancelled_rejections_.load(std::memory_order_relaxed);
    result.backpressure_waits = backpressure_waits_.load(std::memory_order_relaxed);
    return result;
}

const PqFiniteReadyRingQueue& PqFiniteReadyRingSink::queue() const noexcept
{
    return queue_;
}

void destroy_finite_persistent_worker_queue(PqFinitePersistentWorkerQueue& queue) noexcept
{
    if (queue.stream != nullptr) cudaStreamDestroy(queue.stream);
    if (queue.g != nullptr && queue.cpu_ptr != nullptr) {
        gdr_unmap(queue.g, queue.mh, queue.cpu_ptr, queue.mapped_bytes);
    }
    if (queue.g != nullptr && queue.pinned) {
        gdr_unpin_buffer(queue.g, queue.mh);
    }
    if (queue.g != nullptr) gdr_close(queue.g);
    if (queue.d_requests_raw != nullptr) cudaFree(queue.d_requests_raw);
    if (queue.d_completion_tail != nullptr) cudaFree(queue.d_completion_tail);
    if (queue.h_completion_notifications != nullptr) {
        cudaFreeHost(queue.h_completion_notifications);
    }
    if (queue.h_completions != nullptr) cudaFreeHost(queue.h_completions);
    queue = PqFinitePersistentWorkerQueue{};
}

void create_and_launch_finite_persistent_worker_queue(
    PqFinitePersistentWorkerQueue& queue,
    const PqFiniteKernelParams& params,
    const PqSearchOptions& options,
    size_t dynamic_shared_bytes)
{
    try {
        dispatch_graph_kernel(options.search_queue_size, options.pq_dim, [&]<int QS, int PD>() {
            check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &queue.active_blocks_per_sm,
                           persistent_finite_pq_worker_kernel<QS, PD>,
                           kNativeThreadInBlock,
                           dynamic_shared_bytes),
                       "cudaOccupancyMaxActiveBlocksPerMultiprocessor PQ finite persistent worker");
            cudaDeviceProp prop{};
            int device_id = 0;
            check_cuda(cudaGetDevice(&device_id), "cudaGetDevice PQ finite persistent worker");
            check_cuda(cudaGetDeviceProperties(&prop, device_id),
                       "cudaGetDeviceProperties PQ finite persistent worker");
            queue.multiprocessor_count = prop.multiProcessorCount;
            if (queue.active_blocks_per_sm <= 0 || queue.multiprocessor_count <= 0) {
                fail_pq("finite persistent worker has zero resident capacity");
            }
            queue.resident_blocks = static_cast<uint32_t>(queue.active_blocks_per_sm) *
                                    static_cast<uint32_t>(queue.multiprocessor_count);
            queue.slot_count = queue.resident_blocks * static_cast<uint32_t>(kMultiQuery);
            // A slot cannot complete again until the host consumes its prior notification.
            queue.completion_ring_capacity = queue.slot_count;
            queue.mapped_bytes = round_up_gdr_bytes(
                static_cast<size_t>(queue.slot_count) *
                sizeof(PqFinitePersistentRequestControl));

            check_cuda(cudaMalloc(reinterpret_cast<void**>(&queue.d_requests_raw),
                                  queue.mapped_bytes + kGdrAlignBytes),
                       "cudaMalloc PQ finite persistent requests");
            queue.d_requests = align_gdr_pointer(queue.d_requests_raw);
            queue.g = gdr_open();
            if (queue.g == nullptr) {
                fail_pq("gdr_open returned nullptr for PQ finite persistent requests");
            }
            check_gdr(gdr_pin_buffer(queue.g,
                                     reinterpret_cast<unsigned long>(queue.d_requests),
                                     queue.mapped_bytes,
                                     0,
                                     0,
                                     &queue.mh),
                      "gdr_pin_buffer PQ finite persistent requests");
            queue.pinned = true;
            check_gdr(gdr_map(queue.g,
                              queue.mh,
                              &queue.cpu_ptr,
                              queue.mapped_bytes),
                      "gdr_map PQ finite persistent requests");
            queue.h_requests =
                reinterpret_cast<PqFinitePersistentRequestControl*>(queue.cpu_ptr);
            for (uint32_t slot_idx = 0; slot_idx < queue.slot_count; ++slot_idx) {
                queue.h_requests[slot_idx] = PqFinitePersistentRequestControl{};
            }
            _mm_sfence();

            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_completions),
                                     static_cast<size_t>(queue.slot_count) *
                                         sizeof(PqFinitePersistentCompletionControl),
                                     cudaHostAllocMapped),
                       "cudaHostAlloc PQ finite persistent completions");
            for (uint32_t slot_idx = 0; slot_idx < queue.slot_count; ++slot_idx) {
                queue.h_completions[slot_idx] = PqFinitePersistentCompletionControl{};
            }
            check_cuda(cudaHostGetDevicePointer(
                           reinterpret_cast<void**>(&queue.d_completions),
                           queue.h_completions,
                           0),
                       "cudaHostGetDevicePointer PQ finite persistent completions");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&queue.d_completion_tail),
                                  sizeof(*queue.d_completion_tail)),
                       "cudaMalloc PQ finite persistent completion tail");
            check_cuda(cudaMemset(queue.d_completion_tail,
                                  0,
                                  sizeof(*queue.d_completion_tail)),
                       "cudaMemset PQ finite persistent completion tail");
            check_cuda(cudaHostAlloc(
                           reinterpret_cast<void**>(&queue.h_completion_notifications),
                           static_cast<size_t>(queue.completion_ring_capacity) *
                               sizeof(PqFinitePersistentCompletionNotification),
                           cudaHostAllocMapped),
                       "cudaHostAlloc PQ finite persistent completion notifications");
            for (uint32_t entry_idx = 0;
                 entry_idx < queue.completion_ring_capacity;
                 ++entry_idx) {
                queue.h_completion_notifications[entry_idx] =
                    PqFinitePersistentCompletionNotification{};
            }
            check_cuda(cudaHostGetDevicePointer(
                           reinterpret_cast<void**>(&queue.d_completion_notifications),
                           queue.h_completion_notifications,
                           0),
                       "cudaHostGetDevicePointer PQ finite persistent completion notifications");
            check_cuda(cudaStreamCreateWithFlags(&queue.stream, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags PQ finite persistent worker");

            persistent_finite_pq_worker_kernel<QS, PD>
                <<<queue.resident_blocks,
                   kNativeThreadInBlock,
                   dynamic_shared_bytes,
                   queue.stream>>>(params,
                                   queue.d_requests,
                                   queue.d_completions,
                                   queue.d_completion_tail,
                                   queue.d_completion_notifications,
                                   queue.completion_ring_capacity);
        });
        check_cuda(cudaGetLastError(), "launch PQ finite persistent worker");
        queue.launched = true;
    } catch (...) {
        destroy_finite_persistent_worker_queue(queue);
        throw;
    }
}

uint64_t stop_finite_persistent_worker(PqFinitePersistentWorkerQueue& queue)
{
    if (!queue.launched || queue.stopped) return 0;
    const auto begin = std::chrono::steady_clock::now();
    for (uint32_t slot_idx = 0; slot_idx < queue.slot_count; ++slot_idx) {
        PqFinitePersistentRequestControl& request = queue.h_requests[slot_idx];
        request.stop = 1;
        request.ready_seq = kPersistentLatencyStopSeq;
    }
    _mm_sfence();
    check_cuda(cudaStreamSynchronize(queue.stream),
               "cudaStreamSynchronize PQ finite persistent worker stop");
    queue.stopped = true;
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count());
}

void publish_finished_for_host_service(PqGraphSearchContext& ctx,
                                       const std::vector<uint32_t>& query_ids);

void drive_finite_persistent_scheduler(PqGraphSearchContext& ctx,
                                       const GraphRuntime& runtime,
                                       const PqSearchOptions& options,
                                       const std::vector<uint32_t>& query_ids,
                                       uint32_t context_epoch,
                                       PqFiniteCompletionQueue* completion_queue,
                                       bool print_summary)
{
    if (completion_queue == nullptr) {
        fail_pq("finite persistent dispatch requires the completion-queue lane fast path");
    }
    require_valid_finite_query_ids(query_ids, ctx.total_query_capacity);
    const PqFiniteKernelParams params = make_finite_kernel_params(ctx, runtime, options);
    PqFinitePersistentWorkerQueue worker_queue;

    struct BacklogItem {
        PqFiniteReadyToken token;
        uint64_t published_ns = 0;
    };
    std::vector<PqFinitePersistentSlot> slots;
    std::vector<uint32_t> free_slots;
    std::deque<BacklogItem> ready_backlog;
    std::vector<HostCompletionEvent> completion_events;
    std::vector<uint8_t> pending;
    std::vector<uint8_t> terminal;
    std::vector<uint8_t> backlogged;
    std::vector<uint32_t> owner_slot;
    std::vector<uint32_t> last_enqueued_generation;

    size_t terminal_count = 0;
    size_t busy_count = 0;
    size_t submissions = 0;
    size_t completions = 0;
    size_t parked_results = 0;
    size_t done_results = 0;
    size_t rejected_results = 0;
    size_t sequence_mismatches = 0;
    size_t max_busy_slots = 0;
    size_t max_ready_backlog = 0;
    size_t host_loop_iterations = 0;
    size_t idle_yields = 0;
    size_t completion_ring_events = 0;
    size_t completion_ring_polls = 0;
    size_t completion_ring_empty_polls = 0;
    uint64_t completion_ring_head = 0;
    uint64_t gpu_advance_ns = 0;
    uint64_t stop_latency_ns = 0;
    auto last_progress = std::chrono::steady_clock::now();
    const auto steady_now_ns = [] {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    };

    try {
        create_and_launch_finite_persistent_worker_queue(
            worker_queue, params, options, ctx.finite_batch_dynamic_shared_bytes);
        slots.resize(worker_queue.slot_count);
        free_slots.reserve(worker_queue.slot_count);
        for (uint32_t slot_idx = worker_queue.slot_count; slot_idx > 0; --slot_idx) {
            free_slots.push_back(slot_idx - 1);
        }
        completion_events.reserve(completion_queue->capacity());
        pending.assign(ctx.total_query_capacity, 0);
        terminal.assign(ctx.total_query_capacity, 0);
        backlogged.assign(ctx.total_query_capacity, 0);
        owner_slot.assign(ctx.total_query_capacity, UINT32_MAX);
        last_enqueued_generation.assign(ctx.total_query_capacity, 0);

        while (terminal_count < query_ids.size() || busy_count != 0) {
            ++host_loop_iterations;
            bool progress = false;

            while (busy_count != 0) {
                const uint64_t expected_publish_sequence = completion_ring_head + 1;
                PqFinitePersistentCompletionNotification& notification =
                    worker_queue.h_completion_notifications[
                        completion_ring_head % worker_queue.completion_ring_capacity];
                ++completion_ring_polls;
                const uint64_t publish_sequence = __atomic_load_n(
                    &notification.publish_sequence, __ATOMIC_ACQUIRE);
                if (publish_sequence != expected_publish_sequence) {
                    ++completion_ring_empty_polls;
                    break;
                }
                const uint32_t slot_idx = notification.slot_idx;
                if (slot_idx >= worker_queue.slot_count) {
                    ++sequence_mismatches;
                    fail_pq("finite persistent completion ring returned an invalid slot");
                }
                PqFinitePersistentSlot& slot = slots[slot_idx];
                if (!slot.busy) {
                    ++sequence_mismatches;
                    fail_pq("finite persistent completion ring returned an idle slot");
                }
                PqFinitePersistentCompletionControl& completion =
                    worker_queue.h_completions[slot_idx];
                const uint32_t done_seq = notification.done_seq;
                if (done_seq != slot.ready_seq) {
                    ++sequence_mismatches;
                    fail_pq("finite persistent worker completion sequence mismatch");
                }
                if (completion.done_seq != done_seq) {
                    ++sequence_mismatches;
                    fail_pq("finite persistent completion payload sequence mismatch");
                }
                if (completion.query_id != slot.query_id ||
                    completion.generation != slot.generation) {
                    ++sequence_mismatches;
                    fail_pq("finite persistent worker completion identity mismatch");
                }
                const PqFiniteAdvanceResult result = completion.result;
                if (result == PqFiniteAdvanceResult::PARKED) {
                    ++parked_results;
                } else if (result == PqFiniteAdvanceResult::DONE) {
                    ++done_results;
                } else {
                    ++rejected_results;
                    fail_pq(std::string("finite persistent worker rejected token: ") +
                            finite_advance_result_name(result));
                }
                if (completion.gpu_stop_ns >= completion.gpu_start_ns) {
                    gpu_advance_ns += completion.gpu_stop_ns - completion.gpu_start_ns;
                }
                if (owner_slot[slot.query_id] != slot_idx || pending[slot.query_id] == 0) {
                    fail_pq("finite persistent worker lost single-owner pending state");
                }
                owner_slot[slot.query_id] = UINT32_MAX;
                if (backlogged[slot.query_id] == 0) pending[slot.query_id] = 0;
                slot.busy = false;
                slot.query_id = UINT32_MAX;
                slot.generation = 0;
                free_slots.push_back(slot_idx);
                --busy_count;
                ++completions;
                ++completion_ring_events;
                ++completion_ring_head;
                progress = true;
            }

            if (!completion_queue->try_pop_batch(completion_events,
                                                 completion_queue->capacity())) {
                fail_pq("finite persistent completion queue was cancelled");
            }
            for (const HostCompletionEvent& event : completion_events) {
                if (event.query_id >= terminal.size()) {
                    fail_pq("finite persistent completion queue returned an out-of-range query");
                }
                if (event.kind == HostCompletionKind::Done) {
                    if (terminal[event.query_id] == 0) {
                        terminal[event.query_id] = 1;
                        ++terminal_count;
                    }
                    progress = true;
                    continue;
                }
                if (terminal[event.query_id] != 0) {
                    fail_pq("finite persistent completion queue returned Ready after Done");
                }
                if (!finite_mailbox_identity_is_valid(event.row,
                                                      event.buffer,
                                                      event.generation)) {
                    fail_pq("finite persistent completion queue returned invalid identity");
                }
                if (event.generation <= last_enqueued_generation[event.query_id]) continue;
                if (backlogged[event.query_id] != 0) {
                    fail_pq("finite persistent scheduler observed duplicate backlog ownership");
                }
                BacklogItem item;
                item.token.query_id = event.query_id;
                item.token.context_epoch = context_epoch;
                item.token.generation = event.generation;
                item.token.row = event.row;
                item.token.buffer = event.buffer;
                item.token.result = PqFiniteAdvanceResult::PENDING;
                item.published_ns = event.published_ns;
                ready_backlog.push_back(item);
                last_enqueued_generation[event.query_id] = event.generation;
                backlogged[event.query_id] = 1;
                pending[event.query_id] = 1;
                progress = true;
            }

            while (!free_slots.empty() && !ready_backlog.empty()) {
                auto ready_it = std::find_if(
                    ready_backlog.begin(),
                    ready_backlog.end(),
                    [&owner_slot](const BacklogItem& item) {
                        return owner_slot[item.token.query_id] == UINT32_MAX;
                    });
                if (ready_it == ready_backlog.end()) break;
                BacklogItem item = *ready_it;
                ready_backlog.erase(ready_it);
                const uint32_t slot_idx = free_slots.back();
                free_slots.pop_back();
                PqFinitePersistentSlot& slot = slots[slot_idx];
                slot.ready_seq = next_finite_persistent_sequence(slot.ready_seq);
                slot.query_id = item.token.query_id;
                slot.generation = item.token.generation;
                slot.busy = true;
                owner_slot[slot.query_id] = slot_idx;
                backlogged[slot.query_id] = 0;

                PqFinitePersistentRequestControl& request =
                    worker_queue.h_requests[slot_idx];
                request.stop = 0;
                request.query_id = item.token.query_id;
                request.context_epoch = item.token.context_epoch;
                request.generation = item.token.generation;
                request.row = item.token.row;
                request.buffer = item.token.buffer;
                const uint64_t now_ns = steady_now_ns();
                request.ready_ns = now_ns >= item.published_ns
                    ? now_ns - item.published_ns
                    : 0;
                _mm_sfence();
                request.ready_seq = slot.ready_seq;
                _mm_sfence();

                ++busy_count;
                ++submissions;
                progress = true;
            }

            max_busy_slots = std::max(max_busy_slots, busy_count);
            max_ready_backlog = std::max(max_ready_backlog, ready_backlog.size());
            if (progress) {
                last_progress = std::chrono::steady_clock::now();
            } else {
                if (std::chrono::steady_clock::now() - last_progress >
                    finite_no_progress_timeout()) {
                    fail_pq("finite persistent scheduler made no progress for " +
                            std::to_string(finite_no_progress_timeout().count()) + " ms");
                }
                ++idle_yields;
                std::this_thread::yield();
            }
        }
        if (!ready_backlog.empty()) {
            fail_pq("finite persistent scheduler terminated with a nonempty ready backlog");
        }
        for (uint32_t query_id : query_ids) {
            if (pending[query_id] != 0 || owner_slot[query_id] != UINT32_MAX) {
                fail_pq("finite persistent scheduler terminated with pending ownership");
            }
        }
        stop_latency_ns = stop_finite_persistent_worker(worker_queue);
    } catch (...) {
        completion_queue->cancel();
        try {
            stop_finite_persistent_worker(worker_queue);
        } catch (...) {
        }
        publish_finished_for_host_service(ctx, query_ids);
        // Lane workers borrow only the completion queue and query mailboxes;
        // after the persistent stream stops they cannot access these slot mappings.
        destroy_finite_persistent_worker_queue(worker_queue);
        throw;
    }

    (void)print_summary;
    std::printf("[pq finite persistent scheduler] worker_blocks=%u worker_slots=%u "
                "active_blocks_per_sm=%d active_warps_per_sm=%d submissions=%zu "
                "completions=%zu parked=%zu done=%zu rejected=%zu "
                "sequence_mismatches=%zu max_busy_slots=%zu max_ready_backlog=%zu "
                "host_loop_iterations=%zu idle_yields=%zu completion_ring_capacity=%u "
                "completion_ring_events=%zu completion_ring_polls=%zu "
                "completion_ring_empty_polls=%zu gpu_advance_ms=%.3f "
                "stop_latency_ms=%.3f\n",
                worker_queue.resident_blocks,
                worker_queue.slot_count,
                worker_queue.active_blocks_per_sm,
                worker_queue.active_blocks_per_sm * kMultiQuery,
                submissions,
                completions,
                parked_results,
                done_results,
                rejected_results,
                sequence_mismatches,
                max_busy_slots,
                max_ready_backlog,
                host_loop_iterations,
                idle_yields,
                worker_queue.completion_ring_capacity,
                completion_ring_events,
                completion_ring_polls,
                completion_ring_empty_polls,
                static_cast<double>(gpu_advance_ns) / 1.0e6,
                static_cast<double>(stop_latency_ns) / 1.0e6);
    const PqFiniteCompletionQueueStats queue_stats = completion_queue->stats();
    std::printf("[pq finite completion queue] capacity=%zu ready=%llu done=%llu "
                "dequeued=%llu duplicates=%llu invalid_rejections=%llu "
                "terminal_rejections=%llu cancelled_rejections=%llu "
                "backpressure_waits=%llu max_depth=%zu\n",
                completion_queue->capacity(),
                static_cast<unsigned long long>(queue_stats.ready_enqueued),
                static_cast<unsigned long long>(queue_stats.done_enqueued),
                static_cast<unsigned long long>(queue_stats.dequeued),
                static_cast<unsigned long long>(queue_stats.duplicate_rejections),
                static_cast<unsigned long long>(queue_stats.invalid_rejections),
                static_cast<unsigned long long>(queue_stats.terminal_rejections),
                static_cast<unsigned long long>(queue_stats.cancelled_rejections),
                static_cast<unsigned long long>(queue_stats.backpressure_waits),
                queue_stats.max_depth);
    destroy_finite_persistent_worker_queue(worker_queue);
}

void drive_finite_ready_ring_scheduler(PqGraphSearchContext& ctx,
                                        const GraphRuntime& runtime,
                                        const PqSearchOptions& options,
                                        PqFiniteReadyRingSink* ready_ring,
                                        bool print_summary)
{
    if (ready_ring == nullptr) fail_pq("finite ready-ring dispatch requires a ready-ring sink");
    check_cuda(cudaStreamSynchronize(nullptr), "finite PQ ready-ring init completion");
    const PqFiniteKernelParams params = make_finite_kernel_params(ctx, runtime, options);
    uint64_t stop_latency_ns = 0;
    uint64_t last_ready_count = ready_ring->published_ready_count();
    size_t last_done_count = ready_ring->done_count();
    size_t host_wait_iterations = 0;
    auto last_progress = std::chrono::steady_clock::now();

    try {
        ready_ring->launch(params, options, ctx.finite_batch_dynamic_shared_bytes);
        while (ready_ring->done_count() < ready_ring->active_query_count()) {
            ++host_wait_iterations;
            const uint32_t error_result = ready_ring->error_result();
            if (error_result != 0) {
                fail_pq("finite ready-ring worker rejected token: result=" +
                        std::string(finite_advance_result_name(
                            static_cast<PqFiniteAdvanceResult>(error_result))) +
                        " query=" + std::to_string(ready_ring->error_query_id()) +
                        " generation=" + std::to_string(ready_ring->error_generation()) +
                        " ticket=" + std::to_string(ready_ring->error_ticket()));
            }
            const uint64_t ready_count = ready_ring->published_ready_count();
            const size_t done_count = ready_ring->done_count();
            if (ready_count != last_ready_count || done_count != last_done_count) {
                last_ready_count = ready_count;
                last_done_count = done_count;
                last_progress = std::chrono::steady_clock::now();
            } else if (std::chrono::steady_clock::now() - last_progress >
                       finite_no_progress_timeout()) {
                const PqFiniteCompletionQueueStats stats = ready_ring->stats();
                std::ostringstream message;
                message << "finite ready-ring scheduler made no progress for "
                        << finite_no_progress_timeout().count() << " ms"
                        << " ready=" << stats.ready_enqueued
                        << " done=" << stats.done_enqueued
                        << " consumed_sequence_max="
                        << ready_ring->observed_consumed_sequence_max()
                        << " duplicate_rejections=" << stats.duplicate_rejections
                        << " invalid_rejections=" << stats.invalid_rejections
                        << " terminal_rejections=" << stats.terminal_rejections
                        << " backpressure_waits=" << stats.backpressure_waits;
                fail_pq(message.str());
            }
            (void)ready_ring->wait_for_terminal_change(std::chrono::milliseconds(1));
        }
        stop_latency_ns = ready_ring->stop();
    } catch (...) {
        ready_ring->cancel();
        try {
            ready_ring->stop();
        } catch (...) {
        }
        throw;
    }

    (void)print_summary;
    const PqFiniteReadyRingQueue& queue = ready_ring->queue();
    const PqFiniteCompletionQueueStats stats = ready_ring->stats();
    const uint64_t parked = stats.ready_enqueued >= stats.done_enqueued
        ? stats.ready_enqueued - stats.done_enqueued
        : 0;
#if GO_FANN_PQ_FINITE_PERHOP_TIMING
    std::vector<PqFinitePerHopTiming> worker_timing(queue.worker_slots);
    check_cuda(cudaMemcpy(worker_timing.data(),
                          queue.d_per_hop_timing,
                          worker_timing.size() * sizeof(PqFinitePerHopTiming),
                          cudaMemcpyDeviceToHost),
               "copy PQ finite per-hop timing to host");
    PqFinitePerHopTiming timing_total{};
    for (const PqFinitePerHopTiming& timing : worker_timing) {
        timing_total.ring_wait_cycles += timing.ring_wait_cycles;
        timing_total.ring_metadata_cycles += timing.ring_metadata_cycles;
        timing_total.restore_cycles += timing.restore_cycles;
        timing_total.payload_cycles += timing.payload_cycles;
        timing_total.writeback_cycles += timing.writeback_cycles;
        timing_total.resume_events += timing.resume_events;
        timing_total.park_events += timing.park_events;
        timing_total.terminal_events += timing.terminal_events;
    }
    const auto average_cycles = [](uint64_t cycles, uint64_t events) {
        return events == 0 ? 0.0
                           : static_cast<double>(cycles) / static_cast<double>(events);
    };
    const uint64_t host_publish_samples = ready_ring->host_publish_sample_count();
    const uint64_t host_publish_cycles = ready_ring->host_publish_sample_cycles();
    std::printf("[pq finite per-hop] timing=enabled clock=sm_cycles worker_records=%zu "
                "resume_events=%llu park_events=%llu terminal_events=%llu "
                "ring_wait_cycles=%llu ring_wait_cycles_per_resume=%.3f "
                "ring_metadata_cycles=%llu ring_metadata_cycles_per_resume=%.3f "
                "restore_cycles=%llu restore_cycles_per_resume=%.3f "
                "payload_cycles=%llu payload_cycles_per_resume=%.3f "
                "writeback_cycles=%llu writeback_cycles_per_transition=%.3f "
                "host_publish_sample_rate=1/4096 host_publish_samples=%llu "
                "host_publish_tsc_cycles=%llu host_publish_tsc_cycles_per_sample=%.3f\n",
                worker_timing.size(),
                static_cast<unsigned long long>(timing_total.resume_events),
                static_cast<unsigned long long>(timing_total.park_events),
                static_cast<unsigned long long>(timing_total.terminal_events),
                static_cast<unsigned long long>(timing_total.ring_wait_cycles),
                average_cycles(timing_total.ring_wait_cycles, timing_total.resume_events),
                static_cast<unsigned long long>(timing_total.ring_metadata_cycles),
                average_cycles(timing_total.ring_metadata_cycles, timing_total.resume_events),
                static_cast<unsigned long long>(timing_total.restore_cycles),
                average_cycles(timing_total.restore_cycles, timing_total.resume_events),
                static_cast<unsigned long long>(timing_total.payload_cycles),
                average_cycles(timing_total.payload_cycles, timing_total.resume_events),
                static_cast<unsigned long long>(timing_total.writeback_cycles),
                average_cycles(timing_total.writeback_cycles,
                               timing_total.park_events + timing_total.terminal_events),
                static_cast<unsigned long long>(host_publish_samples),
                static_cast<unsigned long long>(host_publish_cycles),
                average_cycles(host_publish_cycles, host_publish_samples));
#else
    std::printf("[pq finite per-hop] timing=disabled\n");
#endif
    std::printf("[pq finite ready ring] worker_blocks=%u worker_slots=%u "
                "active_blocks_per_sm=%d active_warps_per_sm=%d ring_capacity=%u "
                "published=%llu parked=%llu done=%llu host_wait_iterations=%zu "
                "duplicates=%llu invalid_rejections=%llu terminal_rejections=%llu "
                "cancelled_rejections=%llu backpressure_waits=%llu "
                "ready_timing=unavailable "
                "stop_latency_ms=%.3f\n",
                queue.resident_blocks,
                queue.worker_slots,
                queue.active_blocks_per_sm,
                queue.active_blocks_per_sm * kMultiQuery,
                queue.ring_capacity,
                static_cast<unsigned long long>(stats.ready_enqueued),
                static_cast<unsigned long long>(parked),
                static_cast<unsigned long long>(stats.done_enqueued),
                host_wait_iterations,
                static_cast<unsigned long long>(stats.duplicate_rejections),
                static_cast<unsigned long long>(stats.invalid_rejections),
                static_cast<unsigned long long>(stats.terminal_rejections),
                static_cast<unsigned long long>(stats.cancelled_rejections),
                static_cast<unsigned long long>(stats.backpressure_waits),
                static_cast<double>(stop_latency_ns) / 1.0e6);
}

void drive_finite_scheduler(PqGraphSearchContext& ctx,
                            const GraphRuntime& runtime,
                            const PqSearchOptions& options,
                            const std::vector<uint32_t>& query_ids,
                            uint32_t context_epoch,
                            PqFiniteCompletionQueue* completion_queue,
                            bool print_summary)
{
    const cudaStream_t scheduler_stream = nullptr;
    check_cuda(cudaStreamSynchronize(scheduler_stream), "finite PQ init kernel completion");
    if (query_ids.empty()) return;
    if (defaults::pq_finite_dispatch_mode_is_ready_ring(options.finite_dispatch_mode)) {
        fail_pq("finite ready-ring dispatch must use its direct scheduler entrypoint");
    }
    if (defaults::pq_finite_dispatch_mode_is_persistent(options.finite_dispatch_mode)) {
        drive_finite_persistent_scheduler(ctx,
                                          runtime,
                                          options,
                                          query_ids,
                                          context_epoch,
                                          completion_queue,
                                          print_summary);
        return;
    }
    if (ctx.finite_ready_token_capacity == 0) {
        fail_pq("finite PQ scheduler has zero ready-token capacity");
    }

    std::vector<uint8_t> pending(ctx.total_query_capacity, 0);
    std::vector<uint8_t> done(ctx.total_query_capacity, 0);
    std::vector<uint32_t> last_dispatched_generation(ctx.total_query_capacity, 0);
    std::vector<std::chrono::steady_clock::time_point> ready_since(ctx.total_query_capacity);
    std::vector<PqFiniteReadyToken> ready_tokens;
    ready_tokens.reserve(ctx.finite_ready_token_capacity);
    std::vector<HostCompletionEvent> completion_events;
    completion_events.reserve(ctx.finite_ready_token_capacity);
    require_valid_finite_query_ids(query_ids, ctx.total_query_capacity);

    size_t done_count = 0;
    size_t launch_count = 0;
    size_t dispatched_token_count = 0;
    size_t partial_launch_count = 0;
    size_t max_batch_size = 0;
    uint64_t host_staging_ns = 0;
    uint64_t h2d_ns = 0;
    uint64_t resume_kernel_sync_ns = 0;
    uint64_t d2h_ns = 0;
    auto first_ready_time = std::chrono::steady_clock::time_point{};
    auto last_progress = std::chrono::steady_clock::now();
    const PqFiniteKernelParams params = make_finite_kernel_params(ctx, runtime, options);
    const auto partial_batch_timeout =
        std::chrono::microseconds(options.finite_partial_batch_timeout_us);
    const uint64_t partial_timeout_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            partial_batch_timeout).count());
    const PqFiniteTokenTransport token_transport =
        parse_pq_finite_token_transport(options.finite_token_transport);
    if (token_transport != ctx.finite_token_transport) {
        fail_pq("finite PQ token transport does not match graph-context allocation");
    }
    const auto elapsed_ns = [](std::chrono::steady_clock::time_point begin,
                               std::chrono::steady_clock::time_point end) {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    };

    while (done_count < query_ids.size()) {
        const auto scan_time = std::chrono::steady_clock::now();
        if (completion_queue != nullptr) {
            if (!completion_queue->wait_pop_batch(completion_events,
                                                  ctx.finite_ready_token_capacity,
                                                  partial_batch_timeout)) {
                fail_pq("finite PQ completion queue was cancelled");
            }
            const auto completion_time = std::chrono::steady_clock::now();
            const uint64_t now_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    completion_time.time_since_epoch()).count());
            for (const HostCompletionEvent& event : completion_events) {
                if (event.query_id >= done.size()) {
                    fail_pq("finite PQ completion queue returned an out-of-range query");
                }
                if (event.kind == HostCompletionKind::Done) {
                    pending[event.query_id] = 0;
                    if (done[event.query_id] == 0) {
                        done[event.query_id] = 1;
                        ++done_count;
                    }
                    last_progress = completion_time;
                    continue;
                }
                if (done[event.query_id] != 0) {
                    fail_pq("finite PQ completion queue returned Ready after Done");
                }
                if (!finite_mailbox_identity_is_valid(event.row,
                                                      event.buffer,
                                                      event.generation)) {
                    fail_pq("finite PQ completion queue returned an invalid mailbox identity");
                }
                if (event.generation <= last_dispatched_generation[event.query_id]) continue;
                PqFiniteReadyToken token;
                token.query_id = event.query_id;
                token.context_epoch = context_epoch;
                token.generation = event.generation;
                token.row = event.row;
                token.buffer = event.buffer;
                token.ready_ns = now_ns >= event.published_ns ? now_ns - event.published_ns : 0;
                token.result = PqFiniteAdvanceResult::PENDING;
                ready_tokens.push_back(token);
                pending[event.query_id] = 1;
            }
        } else {
            for (uint32_t query_id : query_ids) {
                if (done[query_id] != 0) continue;
                const int64_t flag = load_status_flag_word(ctx.status_flag, query_id);
                if (graph_status_flag_finished(flag)) {
                    done[query_id] = 1;
                    pending[query_id] = 0;
                    ++done_count;
                    last_progress = scan_time;
                    continue;
                }
                if (pending[query_id] != 0 || flag != 1) continue;
                const PqFiniteMailboxSnapshot first = load_finite_mailbox_snapshot(ctx, query_id);
                const PqFiniteMailboxSnapshot second = load_finite_mailbox_snapshot(ctx, query_id);
                if (!(first == second) || second.flag != 1 || second.ready == 0) continue;
                if (!finite_mailbox_identity_is_valid(second.row,
                                                      second.buffer,
                                                      second.generation)) {
                    fail_pq("finite PQ scheduler observed an invalid mailbox identity");
                }
                const uint32_t generation = static_cast<uint32_t>(second.generation);
                if (generation <= last_dispatched_generation[query_id]) continue;
                PqFiniteReadyToken token;
                token.query_id = query_id;
                token.context_epoch = context_epoch;
                token.generation = generation;
                token.row = static_cast<uint32_t>(second.row);
                token.buffer = static_cast<uint32_t>(second.buffer);
                token.result = PqFiniteAdvanceResult::PENDING;
                ready_tokens.push_back(token);
                pending[query_id] = 1;
                ready_since[query_id] = scan_time;
                if (ready_tokens.size() == 1) first_ready_time = scan_time;
                if (ready_tokens.size() >= ctx.finite_ready_token_capacity) break;
            }
        }

        const auto now = std::chrono::steady_clock::now();
        const uint64_t oldest_ready_age_ns = ready_tokens.empty()
            ? 0
            : static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                  now - first_ready_time).count());
        const bool should_flush = completion_queue != nullptr
            ? !ready_tokens.empty()
            : finite_ready_batch_should_flush(ready_tokens.size(),
                                              ctx.finite_ready_token_capacity,
                                              oldest_ready_age_ns,
                                              partial_timeout_ns);
        if (should_flush) {
            if (completion_queue == nullptr) {
                for (PqFiniteReadyToken& token : ready_tokens) {
                    token.ready_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            now - ready_since[token.query_id]).count());
                }
            }
            PqFiniteReadyToken* host_result_tokens = nullptr;
            if (token_transport == PqFiniteTokenTransport::Mapped) {
                const auto staging_begin = std::chrono::steady_clock::now();
                std::copy(ready_tokens.begin(),
                          ready_tokens.end(),
                          ctx.h_finite_ready_tokens_mapped);
                std::atomic_thread_fence(std::memory_order_release);
                host_staging_ns += elapsed_ns(staging_begin,
                                              std::chrono::steady_clock::now());
                host_result_tokens = ctx.h_finite_ready_tokens_mapped;
            } else {
                const auto h2d_begin = std::chrono::steady_clock::now();
                check_cuda(cudaMemcpy(ctx.d_finite_ready_tokens,
                                      ready_tokens.data(),
                                      ready_tokens.size() * sizeof(PqFiniteReadyToken),
                                      cudaMemcpyHostToDevice),
                           "copy finite PQ ready tokens to device");
                h2d_ns += elapsed_ns(h2d_begin, std::chrono::steady_clock::now());
            }
            const int grid_dim = static_cast<int>(
                (ready_tokens.size() + kMultiQuery - 1) / kMultiQuery);
            const auto resume_begin = std::chrono::steady_clock::now();
            dispatch_graph_kernel(options.search_queue_size, options.pq_dim, [&]<int QS, int PD>() {
                finite_pq_resume_kernel<QS, PD>
                    <<<grid_dim,
                       kNativeThreadInBlock,
                       ctx.finite_batch_dynamic_shared_bytes,
                       scheduler_stream>>>(
                        params,
                        ctx.d_finite_ready_tokens,
                        static_cast<uint32_t>(ready_tokens.size()));
            });
            check_cuda(cudaGetLastError(), "finite PQ resume kernel launch");
            check_cuda(cudaStreamSynchronize(scheduler_stream),
                       "finite PQ resume kernel completion");
            resume_kernel_sync_ns += elapsed_ns(resume_begin,
                                                std::chrono::steady_clock::now());
            if (token_transport == PqFiniteTokenTransport::DeviceCopy) {
                const auto d2h_begin = std::chrono::steady_clock::now();
                check_cuda(cudaMemcpy(ready_tokens.data(),
                                      ctx.d_finite_ready_tokens,
                                      ready_tokens.size() * sizeof(PqFiniteReadyToken),
                                      cudaMemcpyDeviceToHost),
                           "copy finite PQ ready tokens to host");
                d2h_ns += elapsed_ns(d2h_begin, std::chrono::steady_clock::now());
                host_result_tokens = ready_tokens.data();
            } else {
                std::atomic_thread_fence(std::memory_order_acquire);
            }
            ++launch_count;
            dispatched_token_count += ready_tokens.size();
            if (ready_tokens.size() < ctx.finite_ready_token_capacity) {
                ++partial_launch_count;
            }
            max_batch_size = std::max(max_batch_size, ready_tokens.size());
            for (size_t token_idx = 0; token_idx < ready_tokens.size(); ++token_idx) {
                const PqFiniteReadyToken& token = host_result_tokens[token_idx];
                pending[token.query_id] = 0;
                if (token.result != PqFiniteAdvanceResult::PARKED &&
                    token.result != PqFiniteAdvanceResult::DONE) {
                    fail_pq("finite PQ resume rejected token for query " +
                            std::to_string(token.query_id) + " generation " +
                            std::to_string(token.generation) + ": " +
                            finite_advance_result_name(token.result));
                }
                last_dispatched_generation[token.query_id] = token.generation;
                if (completion_queue == nullptr &&
                    token.result == PqFiniteAdvanceResult::DONE && done[token.query_id] == 0) {
                    done[token.query_id] = 1;
                    ++done_count;
                }
            }
            ready_tokens.clear();
            first_ready_time = {};
            last_progress = std::chrono::steady_clock::now();
            continue;
        }

        if (now - last_progress >= finite_no_progress_timeout()) {
            std::ostringstream message;
            message << "finite PQ scheduler made no progress for "
                    << finite_no_progress_timeout().count() << " ms";
            if (completion_queue != nullptr) {
                const PqFiniteCompletionQueueStats stats = completion_queue->stats();
                message << " ready=" << stats.ready_enqueued
                        << " done=" << stats.done_enqueued
                        << " dequeued=" << stats.dequeued
                        << " duplicate_rejections=" << stats.duplicate_rejections
                        << " invalid_rejections=" << stats.invalid_rejections
                        << " terminal_rejections=" << stats.terminal_rejections;
            }
            const size_t snapshot_count = std::min<size_t>(query_ids.size(), 4);
            for (size_t query_idx = 0; query_idx < snapshot_count; ++query_idx) {
                const uint32_t query_id = query_ids[query_idx];
                const PqFiniteMailboxSnapshot snapshot =
                    load_finite_mailbox_snapshot(ctx, query_id);
                message << " q" << query_id
                        << "={flag:" << snapshot.flag
                        << ",row:" << snapshot.row
                        << ",buffer:" << snapshot.buffer
                        << ",generation:" << snapshot.generation
                        << ",ready:" << snapshot.ready << "}";
            }
            size_t waiting_count = 0;
            size_t terminal_count = 0;
            size_t other_count = 0;
            size_t nonterminal_samples = 0;
            std::ostringstream nonterminal;
            for (uint32_t query_id : query_ids) {
                const PqFiniteMailboxSnapshot snapshot =
                    load_finite_mailbox_snapshot(ctx, query_id);
                if (snapshot.flag == 1) {
                    ++waiting_count;
                } else if (graph_status_flag_finished(snapshot.flag)) {
                    ++terminal_count;
                } else {
                    ++other_count;
                }
                if (!graph_status_flag_finished(snapshot.flag) && nonterminal_samples < 8) {
                    nonterminal << " q" << query_id
                                << "={flag:" << snapshot.flag
                                << ",row:" << snapshot.row
                                << ",buffer:" << snapshot.buffer
                                << ",generation:" << snapshot.generation
                                << ",ready:" << snapshot.ready << "}";
                    ++nonterminal_samples;
                }
            }
            message << " active=" << query_ids.size()
                    << " waiting=" << waiting_count
                    << " terminal=" << terminal_count
                    << " other=" << other_count
                    << " nonterminal_samples=" << nonterminal.str();
            fail_pq(message.str());
        }
        if (completion_queue == nullptr) {
            std::this_thread::sleep_for(std::chrono::microseconds(2));
        }
    }

    if (!ready_tokens.empty()) {
        fail_pq("finite PQ scheduler reached DONE with pending ready ownership");
    }
    if (completion_queue != nullptr || print_summary) {
        const double average_batch_size = launch_count == 0
            ? 0.0
            : static_cast<double>(dispatched_token_count) / static_cast<double>(launch_count);
        std::printf("[pq finite scheduler] epoch=%u discovery=%s contexts=%zu launches=%zu "
                    "partial_launches=%zu dispatched_tokens=%zu max_batch=%zu average_batch=%.2f "
                    "batch_capacity=%zu "
                    "partial_timeout_us=%lld transport=%s host_staging_ms=%.3f h2d_ms=%.3f "
                    "resume_kernel_sync_ms=%.3f d2h_ms=%.3f\n",
                    context_epoch,
                    completion_queue != nullptr ? "completion_queue" : "mailbox_scan",
                    query_ids.size(),
                    launch_count,
                    partial_launch_count,
                    dispatched_token_count,
                    max_batch_size,
                    average_batch_size,
                    ctx.finite_ready_token_capacity,
                    static_cast<long long>(partial_batch_timeout.count()),
                    pq_finite_token_transport_name(token_transport),
                    static_cast<double>(host_staging_ns) / 1.0e6,
                    static_cast<double>(h2d_ns) / 1.0e6,
                    static_cast<double>(resume_kernel_sync_ns) / 1.0e6,
                    static_cast<double>(d2h_ns) / 1.0e6);
        if (completion_queue != nullptr) {
            const PqFiniteCompletionQueueStats queue_stats = completion_queue->stats();
            std::printf("[pq finite completion queue] capacity=%zu ready=%llu done=%llu "
                        "dequeued=%llu duplicates=%llu invalid_rejections=%llu "
                        "terminal_rejections=%llu cancelled_rejections=%llu "
                        "backpressure_waits=%llu max_depth=%zu\n",
                        completion_queue->capacity(),
                        static_cast<unsigned long long>(queue_stats.ready_enqueued),
                        static_cast<unsigned long long>(queue_stats.done_enqueued),
                        static_cast<unsigned long long>(queue_stats.dequeued),
                        static_cast<unsigned long long>(queue_stats.duplicate_rejections),
                        static_cast<unsigned long long>(queue_stats.invalid_rejections),
                        static_cast<unsigned long long>(queue_stats.terminal_rejections),
                        static_cast<unsigned long long>(queue_stats.cancelled_rejections),
                        static_cast<unsigned long long>(queue_stats.backpressure_waits),
                        queue_stats.max_depth);
        }
    }
}

void destroy_persistent_latency_queue(PqPersistentLatencyQueue& queue)
{
    if (queue.g != nullptr && queue.cpu_ptr != nullptr) {
        gdr_unmap(queue.g, queue.mh, queue.cpu_ptr, queue.mapped_bytes);
    }
    if (queue.g != nullptr && queue.pinned) {
        gdr_unpin_buffer(queue.g, queue.mh);
    }
    if (queue.g != nullptr) {
        gdr_close(queue.g);
    }
    if (queue.d_requests_raw != nullptr) {
        cudaFree(queue.d_requests_raw);
    }
    if (queue.h_completions != nullptr) {
        cudaFreeHost(queue.h_completions);
    }
    if (queue.h_lowfreq_result_mapped != nullptr) {
        cudaFreeHost(queue.h_lowfreq_result_mapped);
    }
    if (queue.h_lowfreq_distances_mapped != nullptr) {
        cudaFreeHost(queue.h_lowfreq_distances_mapped);
    }
    if (queue.d_lowfreq_pq_table != nullptr) {
        cudaFree(queue.d_lowfreq_pq_table);
    }
    if (queue.h_lowfreq_candidate_ids_mapped != nullptr) {
        cudaFreeHost(queue.h_lowfreq_candidate_ids_mapped);
    }
    if (queue.h_lowfreq_candidate_offsets_mapped != nullptr) {
        cudaFreeHost(queue.h_lowfreq_candidate_offsets_mapped);
    }
    if (queue.h_lowfreq_candidate_sizes_mapped != nullptr) {
        cudaFreeHost(queue.h_lowfreq_candidate_sizes_mapped);
    }
    queue = PqPersistentLatencyQueue{};
}

PqPersistentLatencyQueue create_persistent_latency_queue(uint32_t slot_count,
                                                         const PqSearchOptions& options,
                                                         size_t lowfreq_candidate_capacity)
{
    PqPersistentLatencyQueue queue;
    queue.slot_count = slot_count;
    queue.lowfreq_shortlist_k = options.enable_lowfreq_rerank
        ? static_cast<uint32_t>(options.lowfreq_rerank_shortlist)
        : static_cast<uint32_t>(options.topk);
    queue.lowfreq_candidate_mode = options.latency_lowfreq_mode == "candidate";
    queue.lowfreq_candidate_capacity = std::max<size_t>(1, lowfreq_candidate_capacity);
    queue.mapped_bytes =
        round_up_gdr_bytes(std::max<size_t>(1, slot_count) *
                           sizeof(PqPersistentLatencyRequestControl));
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&queue.d_requests_raw),
                              queue.mapped_bytes + kGdrAlignBytes),
                   "cudaMalloc PQ persistent latency requests");
        queue.d_requests = align_gdr_pointer(queue.d_requests_raw);
        queue.g = gdr_open();
        if (queue.g == nullptr) {
            fail_pq("gdr_open returned nullptr for PQ persistent latency requests");
        }
        check_gdr(gdr_pin_buffer(queue.g,
                                 reinterpret_cast<unsigned long>(queue.d_requests),
                                 queue.mapped_bytes,
                                 0,
                                 0,
                                 &queue.mh),
                  "gdr_pin_buffer PQ persistent latency requests");
        queue.pinned = true;
        check_gdr(gdr_map(queue.g, queue.mh, &queue.cpu_ptr, queue.mapped_bytes),
                  "gdr_map PQ persistent latency requests");
        queue.h_requests = reinterpret_cast<PqPersistentLatencyRequestControl*>(queue.cpu_ptr);
        for (uint32_t slot_idx = 0; slot_idx < slot_count; ++slot_idx) {
            queue.h_requests[slot_idx] = PqPersistentLatencyRequestControl{};
        }
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_completions),
                                 sizeof(PqPersistentLatencyCompletionControl) *
                                     std::max<size_t>(1, slot_count),
                                 cudaHostAllocMapped),
                   "cudaHostAlloc PQ persistent latency completions");
        for (uint32_t slot_idx = 0; slot_idx < slot_count; ++slot_idx) {
            queue.h_completions[slot_idx] = PqPersistentLatencyCompletionControl{};
        }
        check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue.d_completions),
                                            queue.h_completions,
                                            0),
                   "cudaHostGetDevicePointer PQ persistent latency completions");
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_lowfreq_result_mapped),
                                 sizeof(idx_t) * std::max<size_t>(1, slot_count) *
                                     std::max<uint32_t>(1, queue.lowfreq_shortlist_k),
                                 cudaHostAllocMapped),
                   "cudaHostAlloc PQ persistent lowfreq result");
        check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue.d_lowfreq_result),
                                            queue.h_lowfreq_result_mapped,
                                            0),
                   "cudaHostGetDevicePointer PQ persistent lowfreq result");
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_lowfreq_distances_mapped),
                                 sizeof(float) * std::max<size_t>(1, slot_count) *
                                     std::max<uint32_t>(1, queue.lowfreq_shortlist_k),
                                 cudaHostAllocMapped),
                   "cudaHostAlloc PQ persistent lowfreq distances");
        check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue.d_lowfreq_distances),
                                            queue.h_lowfreq_distances_mapped,
                                            0),
                   "cudaHostGetDevicePointer PQ persistent lowfreq distances");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&queue.d_lowfreq_pq_table),
                              sizeof(pq_value_t) * std::max<size_t>(1, slot_count) *
                                  static_cast<size_t>(options.pq_dim) * kNumPqCentroids),
                   "cudaMalloc PQ persistent lowfreq pq table");
        if (queue.lowfreq_candidate_mode) {
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_lowfreq_candidate_offsets_mapped),
                                     sizeof(uint32_t) * std::max<size_t>(1, slot_count),
                                     cudaHostAllocMapped),
                       "cudaHostAlloc PQ persistent lowfreq candidate offsets");
            check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue.d_lowfreq_candidate_offsets),
                                                queue.h_lowfreq_candidate_offsets_mapped,
                                                0),
                       "cudaHostGetDevicePointer PQ persistent lowfreq candidate offsets");
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_lowfreq_candidate_sizes_mapped),
                                     sizeof(uint32_t) * std::max<size_t>(1, slot_count),
                                     cudaHostAllocMapped),
                       "cudaHostAlloc PQ persistent lowfreq candidate sizes");
            check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue.d_lowfreq_candidate_sizes),
                                                queue.h_lowfreq_candidate_sizes_mapped,
                                                0),
                       "cudaHostGetDevicePointer PQ persistent lowfreq candidate sizes");
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_lowfreq_candidate_ids_mapped),
                                     sizeof(uint32_t) * queue.lowfreq_candidate_capacity *
                                         std::max<size_t>(1, slot_count),
                                     cudaHostAllocMapped),
                       "cudaHostAlloc PQ persistent lowfreq candidate ids");
            check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue.d_lowfreq_candidate_ids),
                                                queue.h_lowfreq_candidate_ids_mapped,
                                                0),
                       "cudaHostGetDevicePointer PQ persistent lowfreq candidate ids");
        }
    } catch (...) {
        destroy_persistent_latency_queue(queue);
        throw;
    }
    return queue;
}

std::vector<PqPersistentLatencySlot> create_persistent_latency_slots(uint32_t slot_count)
{
    std::vector<PqPersistentLatencySlot> slots(slot_count);
    for (uint32_t slot_idx = 0; slot_idx < slot_count; ++slot_idx) {
        slots[slot_idx].slot_index = slot_idx;
    }
    return slots;
}

void publish_finished_for_host_service(PqGraphSearchContext& ctx,
                                       const std::vector<uint32_t>& query_ids)
{
    if (ctx.status_flag == nullptr) {
        return;
    }
    for (uint32_t query_id : query_ids) {
        if (query_id < ctx.total_query_capacity) {
            ctx.status_flag[static_cast<size_t>(query_id) * 16] = 2;
        }
    }
    _mm_sfence();
}

class PqHostScanThreadGuard {
public:
    PqHostScanThreadGuard(PqGraphSearchContext& ctx,
                          const std::vector<uint32_t>& query_ids,
                          std::thread&& thread,
                          HostCompletionSink* completion_sink = nullptr,
                          std::atomic<bool>* request_ring_cancelled = nullptr) noexcept
        : ctx_(&ctx),
          query_ids_(&query_ids),
          thread_(std::move(thread)),
          completion_sink_(completion_sink),
          request_ring_cancelled_(request_ring_cancelled)
    {
    }

    PqHostScanThreadGuard(const PqHostScanThreadGuard&) = delete;
    PqHostScanThreadGuard& operator=(const PqHostScanThreadGuard&) = delete;

    ~PqHostScanThreadGuard()
    {
        abort_and_join();
    }

    bool joinable() const noexcept
    {
        return thread_.joinable();
    }

    void join()
    {
        thread_.join();
    }

    void abort_and_join() noexcept
    {
        if (!thread_.joinable()) {
            return;
        }
        if (request_ring_cancelled_ != nullptr) {
            request_ring_cancelled_->store(true, std::memory_order_release);
        }
        if (completion_sink_ != nullptr) completion_sink_->cancel();
        publish_finished_for_host_service(*ctx_, *query_ids_);
        try {
            thread_.join();
        } catch (...) {
            std::terminate();
        }
    }

private:
    PqGraphSearchContext* ctx_;
    const std::vector<uint32_t>* query_ids_;
    std::thread thread_;
    HostCompletionSink* completion_sink_ = nullptr;
    std::atomic<bool>* request_ring_cancelled_ = nullptr;
};

void stop_persistent_latency_worker(PqPersistentLatencyQueue& queue)
{
    if (queue.h_requests == nullptr) {
        return;
    }
    for (uint32_t slot_idx = 0; slot_idx < queue.slot_count; ++slot_idx) {
        PqPersistentLatencyRequestControl& request = queue.h_requests[slot_idx];
        request.stop = 1;
        request.ready_seq = kPersistentLatencyStopSeq;
    }
    _mm_sfence();
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize PQ persistent latency stop");
}

void launch_persistent_latency_worker(const PqSearchOptions& options,
                                      const GraphRuntime& runtime,
                                      PqGraphSearchContext& ctx,
                                      PqPersistentLatencyQueue& queue)
{
    dispatch_graph_kernel(options.search_queue_size, options.pq_dim, [&]<int QS, int PD>() {
        const uint32_t resident_blocks = resident_worker_capacity_for_latency_kernel(
            persistent_pq_latency_kernel<QS, PD>,
            kThreadInBlock,
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor PQ persistent latency",
            "cudaGetDeviceProperties PQ persistent latency");
        if (queue.slot_count > resident_blocks) {
            fail_pq("latency_persistent_queue concurrency exceeds resident worker capacity");
        }
        persistent_pq_latency_kernel<QS, PD><<<queue.slot_count, kThreadInBlock>>>(
            ctx.d_data,
            ctx.d_query,
            ctx.d_result,
            ctx.d_result_dist,
            ctx.d_pq_centroid,
            ctx.d_pq_table,
            ctx.d_datadist_list,
            ctx.d_hashdata,
            ctx.d_time_per_block,
            ctx.d_neighbor_memory,
            ctx.d_status_memory,
            ctx.d_gdr_data,
            ctx.d_status_flag,
            ctx.d_prefetch_node_array,
            ctx.d_prefetch_index,
            ctx.d_prefetch_begin_by_group,
            ctx.d_prefetch_end_by_group,
            ctx.d_query_group_ids,
            ctx.d_start_local_ids,
            ctx.d_start_sample_ids,
            ctx.d_row_to_sample,
            ctx.d_label_offset,
            ctx.d_row_filter_mask,
            ctx.d_query_filter_bits,
            ctx.d_query_range_modes,
            ctx.d_query_range_lower_bounds,
            ctx.d_query_range_upper_bounds,
            ctx.d_base_range_values,
            runtime.base_range_count,
            options.dim,
            options.pq_dim,
            static_cast<int>(ctx.use_adaptive_wait),
            static_cast<int>(ctx.buffer_mode),
            options.search_queue_size,
            options.search_finish_count,
            parse_search_early_stop_mode(options.search_early_stop_mode),
            options.search_max_steps,
            static_cast<int>(ctx.stop_width),
            static_cast<int>(ctx.result_width),
            static_cast<uint32_t>(ctx.io_config.vector_payload_words),
            static_cast<uint32_t>(ctx.io_config.vector_payload_words),
            static_cast<uint32_t>(ctx.io_config.neighbor_count_word),
            static_cast<uint32_t>(ctx.io_config.neighbor_words_offset),
            static_cast<uint32_t>(ctx.io_config.neighbor_sample_words_offset),
            static_cast<uint32_t>(ctx.io_config.elems_per_block),
            ctx.use_sample_id_payload ? 1 : 0,
            ctx.use_implicit_filter_mask ? 1 : 0,
            highfreq_gpu_exact_rerank_enabled(options) && !ctx.device_resident_graph ? 1 : 0,
            options.bit_filter_bonus,
            options.range_filter_bonus,
            pq_mailbox_trace_enabled() ? 1 : 0,
            queue.d_lowfreq_result,
            queue.d_lowfreq_distances,
            queue.d_lowfreq_pq_table,
            queue.d_lowfreq_candidate_ids,
            queue.d_lowfreq_candidate_offsets,
            queue.d_lowfreq_candidate_sizes,
            static_cast<int>(queue.lowfreq_shortlist_k),
            queue.d_requests,
            queue.d_completions);
    });
    check_cuda(cudaGetLastError(), "launch PQ persistent latency worker");
}

void copy_time_per_block_to_host(PqGraphSearchContext& ctx, size_t float_count, const char* label)
{
    if (ctx.time_buffer_is_host_mapped && ctx.h_time_per_block_mapped != nullptr) {
        std::memcpy(ctx.h_time_per_block,
                    ctx.h_time_per_block_mapped,
                    float_count * sizeof(float));
        return;
    }
    check_cuda(cudaMemcpy(ctx.h_time_per_block,
                          ctx.d_time_per_block,
                          float_count * sizeof(float),
                          cudaMemcpyDeviceToHost),
               label);
}

void launch_resident_pq_batch_exact_rerank(PqGraphSearchContext& ctx,
                                           const uint32_t* d_query_ids,
                                           size_t query_count,
                                           const PqSearchOptions& options)
{
    if (!ctx.device_resident_graph || query_count == 0) return;
    if (ctx.d_resident_base_vectors == nullptr || ctx.result_width < ctx.stop_width) {
        fail_pq("device-resident PQ rerank state is incomplete");
    }
    const int grid_dim = static_cast<int>((query_count + kMultiQuery - 1) / kMultiQuery);
    resident_pq_batch_exact_rerank_kernel<<<grid_dim, kNativeThreadInBlock>>>(
        ctx.d_resident_base_vectors,
        ctx.d_query,
        d_query_ids,
        ctx.d_resident_data_label_indptr,
        ctx.d_resident_data_label_indices,
        ctx.d_resident_query_label_indptr,
        ctx.d_resident_query_label_indices,
        ctx.d_resident_query_filter_labels,
        ctx.d_result,
        ctx.d_result_dist,
        ctx.resident_base_rows,
        static_cast<int>(query_count),
        options.dim,
        static_cast<int>(ctx.result_width),
        options.topk);
    check_cuda(cudaGetLastError(), "launch resident PQ batched exact rerank");
}

void copy_native_profile_to_host(PqGraphSearchContext& ctx,
                                 size_t query_count,
                                 const char* label)
{
    if constexpr (kNativeProfileMode == 0) {
        (void)ctx;
        (void)query_count;
        (void)label;
    } else {
        if (ctx.d_native_profile == nullptr || ctx.h_native_profile == nullptr) {
            fail_pq("native PQ profile buffers are unavailable");
        }
        check_cuda(cudaMemcpy(ctx.h_native_profile,
                              ctx.d_native_profile,
                              query_count * sizeof(PqNativeProfileRecord),
                              cudaMemcpyDeviceToHost),
                   label);
    }
}

void accumulate_native_profile_record(PqNativeProfileRecord& total,
                                      const PqNativeProfileRecord& record)
{
    total.pickup_elapsed_ns += record.pickup_elapsed_ns;
    total.hash_test_elapsed_ns += record.hash_test_elapsed_ns;
    total.frontier_update_elapsed_ns += record.frontier_update_elapsed_ns;
    total.pickup_invocations += record.pickup_invocations;
    total.hash_test_calls += record.hash_test_calls;
    total.hash_test_hits += record.hash_test_hits;
    total.hash_test_misses += record.hash_test_misses;
    total.hash_test_probes += record.hash_test_probes;
    total.hash_test_max_probe = std::max(total.hash_test_max_probe,
                                         record.hash_test_max_probe);
    total.hash_add_calls += record.hash_add_calls;
    total.hash_add_insertions += record.hash_add_insertions;
    total.hash_add_existing += record.hash_add_existing;
    total.hash_add_full += record.hash_add_full;
    total.hash_add_probes += record.hash_add_probes;
    total.hash_add_max_probe = std::max(total.hash_add_max_probe,
                                        record.hash_add_max_probe);
    total.neighbor_references += record.neighbor_references;
    total.post_hash_distance_evaluations += record.post_hash_distance_evaluations;
    total.threshold_rejections += record.threshold_rejections;
    total.candidate_admissions += record.candidate_admissions;
    total.frontier_evictions += record.frontier_evictions;
    total.frontier_update_invocations += record.frontier_update_invocations;
    total.traversal_steps += record.traversal_steps;
    total.resident_hash_hits += record.resident_hash_hits;
    total.nonresident_hash_hits += record.nonresident_hash_hits;
}

double native_profile_ratio(uint64_t numerator, uint64_t denominator)
{
    return denominator > 0
        ? static_cast<double>(numerator) / static_cast<double>(denominator)
        : 0.0;
}

void print_native_profile_summary(const char* run_label,
                                  const PqNativeProfileRecord* records,
                                  size_t query_count,
                                  int queue_size,
                                  double total_query_time_us)
{
    if constexpr (kNativeProfileMode == 0) {
        (void)run_label;
        (void)records;
        (void)query_count;
        (void)queue_size;
        (void)total_query_time_us;
    } else {
        PqNativeProfileRecord total{};
        for (size_t i = 0; i < query_count; ++i) {
            accumulate_native_profile_record(total, records[i]);
        }

        const bool test_partition_ok =
            total.hash_test_hits + total.hash_test_misses == total.hash_test_calls;
        const bool reference_count_ok =
            total.hash_test_calls == total.neighbor_references;
        const bool evaluation_partition_ok =
            total.candidate_admissions + total.threshold_rejections ==
                total.post_hash_distance_evaluations;
        const bool add_count_ok =
            total.hash_add_calls == total.candidate_admissions + query_count;
        const bool add_partition_ok =
            total.hash_add_insertions + total.hash_add_existing + total.hash_add_full ==
                total.hash_add_calls;
        const bool structural_partition_ok = kNativeProfileMode != 2 ||
            total.resident_hash_hits + total.nonresident_hash_hits == total.hash_test_hits;
        const bool invariants_ok = test_partition_ok && reference_count_ok &&
            evaluation_partition_ok && add_count_ok && add_partition_ok &&
            structural_partition_ok;
        const uint64_t measured_phase_ns = total.pickup_elapsed_ns +
            total.hash_test_elapsed_ns + total.frontier_update_elapsed_ns;
        const double total_query_time_ns = total_query_time_us * 1000.0;

        std::ostringstream raw;
        raw << "[pq native profile raw] mode=" << kNativeProfileMode
            << " kind=" << (kNativeProfileMode == 2 ? "structural" : "counters")
            << " label=" << std::quoted(run_label)
            << " queue_size=" << queue_size
            << " query_count=" << query_count
            << " diagnostic_only=1"
            << " invariants_ok=" << (invariants_ok ? 1 : 0)
            << " traversal_steps=" << total.traversal_steps
            << " neighbor_references=" << total.neighbor_references
            << " pickup_invocations=" << total.pickup_invocations
            << " pickup_elapsed_ns=" << total.pickup_elapsed_ns
            << " hash_test_calls=" << total.hash_test_calls
            << " hash_test_hits=" << total.hash_test_hits
            << " hash_test_misses=" << total.hash_test_misses
            << " hash_test_probes=" << total.hash_test_probes
            << " hash_test_max_probe=" << total.hash_test_max_probe
            << " hash_test_elapsed_ns=" << total.hash_test_elapsed_ns
            << " hash_add_calls=" << total.hash_add_calls
            << " hash_add_insertions=" << total.hash_add_insertions
            << " hash_add_existing=" << total.hash_add_existing
            << " hash_add_full=" << total.hash_add_full
            << " hash_add_probes=" << total.hash_add_probes
            << " hash_add_max_probe=" << total.hash_add_max_probe
            << " post_hash_distance_evaluations="
            << total.post_hash_distance_evaluations
            << " threshold_rejections=" << total.threshold_rejections
            << " candidate_admissions=" << total.candidate_admissions
            << " frontier_evictions=" << total.frontier_evictions
            << " frontier_update_invocations=" << total.frontier_update_invocations
            << " frontier_update_elapsed_ns=" << total.frontier_update_elapsed_ns
            << " resident_hash_hits=" << total.resident_hash_hits
            << " nonresident_hash_hits=" << total.nonresident_hash_hits;
        std::cout << raw.str() << '\n';

        std::ostringstream derived;
        derived << std::setprecision(9)
                << "[pq native profile derived] mode=" << kNativeProfileMode
                << " kind=" << (kNativeProfileMode == 2 ? "structural" : "counters")
                << " label=" << std::quoted(run_label)
                << " queue_size=" << queue_size
                << " diagnostic_only=1"
                << " avg_steps_per_query="
                << native_profile_ratio(total.traversal_steps, query_count)
                << " neighbor_references_per_step="
                << native_profile_ratio(total.neighbor_references, total.traversal_steps)
                << " hash_hit_rate="
                << native_profile_ratio(total.hash_test_hits, total.hash_test_calls)
                << " hash_test_probes_per_call="
                << native_profile_ratio(total.hash_test_probes, total.hash_test_calls)
                << " hash_add_probes_per_call="
                << native_profile_ratio(total.hash_add_probes, total.hash_add_calls)
                << " distance_evaluations_per_step="
                << native_profile_ratio(total.post_hash_distance_evaluations,
                                        total.traversal_steps)
                << " candidate_admission_rate="
                << native_profile_ratio(total.candidate_admissions,
                                        total.post_hash_distance_evaluations)
                << " threshold_rejection_rate="
                << native_profile_ratio(total.threshold_rejections,
                                        total.post_hash_distance_evaluations)
                << " frontier_evictions_per_admission="
                << native_profile_ratio(total.frontier_evictions,
                                        total.candidate_admissions)
                << " hash_add_existing_rate="
                << native_profile_ratio(total.hash_add_existing, total.hash_add_calls)
                << " pickup_ns_per_call="
                << native_profile_ratio(total.pickup_elapsed_ns,
                                        total.pickup_invocations)
                << " hash_test_ns_per_call="
                << native_profile_ratio(total.hash_test_elapsed_ns,
                                        total.hash_test_calls)
                << " frontier_update_ns_per_call="
                << native_profile_ratio(total.frontier_update_elapsed_ns,
                                        total.frontier_update_invocations)
                << " measured_phase_share_of_query_time="
                << (total_query_time_ns > 0.0
                        ? static_cast<double>(measured_phase_ns) / total_query_time_ns
                        : 0.0)
                << " nonresident_share_of_hash_hits="
                << native_profile_ratio(total.nonresident_hash_hits,
                                        total.hash_test_hits);
        std::cout << derived.str() << '\n';
        std::cout.flush();
    }
}

void record_finite_state_traffic(PqBatchStats& stats,
                                 const PqGraphSearchContext& ctx,
                                 size_t query_count,
                                 const uint32_t* query_ids = nullptr)
{
    if (!ctx.finite_query_state_enabled || query_count == 0) return;
    if (ctx.d_finite_context_telemetry == nullptr) {
        fail_pq("finite context telemetry is unavailable");
    }

    const size_t copy_count = query_ids == nullptr
        ? query_count
        : ctx.total_query_capacity;
    std::vector<PqFiniteContextTelemetry> telemetry(copy_count);
    check_cuda(cudaMemcpy(telemetry.data(),
                          ctx.d_finite_context_telemetry,
                          copy_count * sizeof(PqFiniteContextTelemetry),
                          cudaMemcpyDeviceToHost),
               "copy finite context telemetry to host");

    stats.finite_state_traffic_available = true;
    stats.finite_queue_capacity = ctx.finite_queue_capacity;
    stats.finite_result_capacity = ctx.finite_result_capacity;
    for (size_t i = 0; i < query_count; ++i) {
        const size_t query_idx = query_ids == nullptr ? i : query_ids[i];
        if (query_idx >= telemetry.size()) {
            fail_pq("finite context telemetry query index is out of range");
        }
        stats.finite_resume_hops += telemetry[query_idx].resume_count;
        stats.finite_park_hops += telemetry[query_idx].park_count;
        stats.finite_terminal_hops += telemetry[query_idx].terminal_count;
        stats.finite_resumed_step_sum += telemetry[query_idx].resumed_step_count;
        stats.finite_static_prefetch_step_sum +=
            telemetry[query_idx].static_prefetch_step_count;
        stats.finite_multi_step_resume_count +=
            telemetry[query_idx].multi_step_resume_count;
        stats.finite_max_resume_quantum_steps = std::max(
            stats.finite_max_resume_quantum_steps,
            telemetry[query_idx].max_resume_quantum_steps);
    }
}

void copy_result_buffer_to_host(PqGraphSearchContext& ctx,
                                size_t value_count,
                                std::vector<idx_t>& host_result,
                                const char* label)
{
    perf_trace::Scope trace_scope("graph_result_d2h", &ctx);
    host_result.resize(value_count, UINT32_MAX);
    if (ctx.result_buffer_is_host_mapped && ctx.h_result_mapped != nullptr) {
        std::memcpy(host_result.data(), ctx.h_result_mapped, value_count * sizeof(idx_t));
        return;
    }
    check_cuda(cudaMemcpy(host_result.data(),
                          ctx.d_result,
                          sizeof(idx_t) * host_result.size(),
                          cudaMemcpyDeviceToHost),
               label);
}

void copy_result_dist_buffer_to_host(PqGraphSearchContext& ctx,
                                     size_t value_count,
                                     std::vector<float>& host_result_dist,
                                     const char* label)
{
    perf_trace::Scope trace_scope("graph_distance_d2h", &ctx);
    host_result_dist.resize(value_count, FLT_MAX);
    if (ctx.result_buffer_is_host_mapped && ctx.h_result_dist_mapped != nullptr) {
        std::memcpy(host_result_dist.data(),
                    ctx.h_result_dist_mapped,
                    value_count * sizeof(float));
        return;
    }
    check_cuda(cudaMemcpy(host_result_dist.data(),
                          ctx.d_result_dist,
                          sizeof(float) * host_result_dist.size(),
                          cudaMemcpyDeviceToHost),
               label);
}

void extract_highfreq_rows_from_host_buffers(const PqGraphSearchContext& ctx,
                                             const std::vector<uint32_t>& query_ids,
                                             std::vector<std::vector<idx_t>>& results,
                                             std::vector<int>& step_num,
                                             std::vector<std::vector<float>>* result_distances)
{
    results.clear();
    step_num.clear();
    if (result_distances != nullptr) {
        result_distances->clear();
    }
    results.reserve(query_ids.size());
    step_num.reserve(query_ids.size());
    if (result_distances != nullptr) {
        result_distances->reserve(query_ids.size());
    }
    for (uint32_t query_idx : query_ids) {
        std::vector<idx_t> v(ctx.result_width, UINT32_MAX);
        std::vector<float> d(ctx.result_width, FLT_MAX);
        const size_t base = static_cast<size_t>(query_idx) * ctx.result_width;
        for (uint32_t j = 0; j < ctx.result_width; ++j) {
            if (ctx.h_result_mapped != nullptr) {
                v[j] = ctx.h_result_mapped[base + j];
            }
            if (ctx.h_result_dist_mapped != nullptr) {
                d[j] = ctx.h_result_dist_mapped[base + j];
            }
        }
        results.push_back(std::move(v));
        if (result_distances != nullptr) {
            result_distances->push_back(std::move(d));
        }
        step_num.push_back(static_cast<int>(ctx.h_time_per_block[6 * query_idx + 4]));
    }
}

void run_highfreq_streaming_rerank(PqGraphSearchContext& ctx,
                                   const PqSearchOptions& options,
                                   const GraphRuntime& runtime,
                                   const std::vector<uint32_t>& query_ids,
                                   std::vector<std::vector<idx_t>>& results,
                                   std::vector<int>& step_num,
                                   std::vector<std::vector<float>>* result_distances,
                                   const std::atomic<bool>* cancel_requested = nullptr)
{
    if (query_ids.empty()) {
        results.clear();
        step_num.clear();
        if (result_distances != nullptr) {
            result_distances->clear();
        }
        return;
    }

    results.assign(query_ids.size(), std::vector<idx_t>(static_cast<size_t>(options.topk), UINT32_MAX));
    step_num.assign(query_ids.size(), 0);
    if (result_distances != nullptr) {
        result_distances->assign(query_ids.size(), std::vector<float>(static_cast<size_t>(options.topk), FLT_MAX));
    }
    const size_t batch_queries = static_cast<size_t>(std::max(1, options.highfreq_rerank_stream_batch_size));
    std::vector<uint8_t> processed(query_ids.size(), 0);

    size_t processed_count = 0;
    while (processed_count < query_ids.size()) {
        if (cancel_requested != nullptr &&
            cancel_requested->load(std::memory_order_acquire)) {
            return;
        }
        std::vector<size_t> ready_positions;
        ready_positions.reserve(batch_queries);
        for (size_t pos = 0; pos < query_ids.size() && ready_positions.size() < batch_queries; ++pos) {
            if (processed[pos] != 0) {
                continue;
            }
            const uint32_t query_idx = query_ids[pos];
            if (!graph_status_flag_result_ready(load_status_flag_word(ctx.status_flag, query_idx))) {
                continue;
            }
            ready_positions.push_back(pos);
        }
        if (ready_positions.empty()) {
            if (cancel_requested != nullptr &&
                cancel_requested->load(std::memory_order_acquire)) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(2));
            continue;
        }

        std::vector<uint32_t> batch_query_ids;
        std::vector<std::vector<idx_t>> batch_shortlists;
        batch_query_ids.reserve(ready_positions.size());
        batch_shortlists.reserve(ready_positions.size());
        for (size_t pos : ready_positions) {
            const uint32_t query_idx = query_ids[pos];
            const size_t row_base = static_cast<size_t>(query_idx) * ctx.result_width;
            batch_query_ids.push_back(query_idx);
            batch_shortlists.emplace_back(ctx.result_width, UINT32_MAX);
            for (uint32_t j = 0; j < ctx.result_width; ++j) {
                batch_shortlists.back()[j] = ctx.h_result_mapped[row_base + j];
            }
        }

        std::vector<std::vector<idx_t>> batch_results;
        std::vector<std::vector<float>> batch_distances;
        highfreq_host_rerank(options,
                             runtime,
                             runtime.query_vectors,
                             batch_query_ids,
                             batch_shortlists,
                             batch_results,
                             result_distances != nullptr ? &batch_distances : nullptr,
                             ctx.predicate && ctx.predicate->shortlist_verified,
                             ctx.cpu_rerank_workers.get());
        for (size_t batch_idx = 0; batch_idx < ready_positions.size(); ++batch_idx) {
            const size_t pos = ready_positions[batch_idx];
            const uint32_t query_idx = query_ids[pos];
            results[pos] = std::move(batch_results[batch_idx]);
            if (result_distances != nullptr) {
                (*result_distances)[pos] = std::move(batch_distances[batch_idx]);
            }
            if (ctx.h_time_per_block_mapped != nullptr) {
                step_num[pos] = static_cast<int>(ctx.h_time_per_block_mapped[6 * query_idx + 4]);
            }
            processed[pos] = 1;
            ++processed_count;
        }
    }
}

void execute_graph_batch(PqGraphSearchContext& ctx,
                         const std::vector<std::vector<std::pair<int, value_t>>>& queries,
                         const GraphRuntime& runtime,
                         std::vector<std::vector<idx_t>>& results,
                         std::vector<int>& step_num,
                         std::vector<std::vector<float>>* result_distances,
                         const PqSearchOptions& options,
                         PqBatchStats* aggregate_stats = nullptr,
                         bool print_summary = true,
                         double* timed_wall_ms = nullptr)
{
    const size_t query_count = queries.size();
    if (query_count == 0) {
        results.clear();
        step_num.clear();
        return;
    }
    if (query_count > ctx.query_capacity) {
        fail_pq("query batch exceeds graph context capacity");
    }
    if (ctx.request_ring_enabled != (options.request_mode == "batch_ring")) {
        fail_pq("PQ graph context request mode does not match execution options");
    }
    if (ctx.request_ring_enabled &&
        ctx.request_ring_consumed_device_mapped !=
            (options.request_consumed_memory == "device")) {
        fail_pq("PQ graph context request consumed memory does not match execution options");
    }

    std::unique_ptr<value_t[]> h_query(new value_t[query_count * options.dim]);
    std::memset(h_query.get(), 0, sizeof(value_t) * query_count * options.dim);
    for (size_t i = 0; i < query_count; ++i) {
        for (auto p : queries[i]) {
            h_query[i * options.dim + p.first] = p.second;
        }
    }
    trace_query_subset_mapping(runtime, {}, "execute_graph_batch(full)");
    std::vector<uint32_t> local_query_indices(query_count, 0);
    for (size_t i = 0; i < query_count; ++i) {
        local_query_indices[i] = static_cast<uint32_t>(i);
    }

    check_cuda(cudaMemcpy(ctx.d_query,
                          h_query.get(),
                          sizeof(value_t) * query_count * options.dim,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_query");
    check_cuda(cudaMemcpy(ctx.d_query_group_ids,
                          runtime.query_group_ids.data(),
                          query_count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_query_group_ids");
    check_cuda(cudaMemcpy(ctx.d_start_local_ids,
                          runtime.start_local_ids.data(),
                          query_count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_start_local_ids");
    check_cuda(cudaMemcpy(ctx.d_query_filter_bits,
                          runtime.query_filter_bits.data(),
                          query_count * sizeof(uint64_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_query_filter_bits");
    const size_t query_range_elems = query_count * runtime.query_ranges.range_count;
    check_cuda(cudaMemcpy(ctx.d_query_range_modes,
                          runtime.query_ranges.modes.data(),
                          query_range_elems * sizeof(uint8_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_query_range_modes");
    check_cuda(cudaMemcpy(ctx.d_query_range_lower_bounds,
                          runtime.query_ranges.lower_bounds.data(),
                          query_range_elems * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_query_range_lower_bounds");
    check_cuda(cudaMemcpy(ctx.d_query_range_upper_bounds,
                          runtime.query_ranges.upper_bounds.data(),
                          query_range_elems * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_query_range_upper_bounds");

    auto execute_run = [&](const char* run_label, bool collect_results) {
        expansion_trace::Invocation expansion(runtime, local_query_indices);
        reset_search_state(ctx, ctx.query_capacity);

        std::vector<int> req_per_loop(20000, 0);
        float milliseconds = 0.0f;
        const bool finite_mode =
            defaults::pq_query_worker_mode_is_finite(options.query_worker_mode);
        const bool completion_queue_mode =
            use_finite_completion_fast_path(ctx, runtime, options);
        const bool ready_ring_mode = completion_queue_mode &&
            defaults::pq_finite_dispatch_mode_is_ready_ring(options.finite_dispatch_mode);
        if (ctx.io_config.completion_sink != nullptr) {
            fail_pq("persistent PQ io_config must not retain a completion sink");
        }
        std::unique_ptr<PqFiniteCompletionQueue> completion_queue =
            completion_queue_mode && !ready_ring_mode
            ? std::make_unique<PqFiniteCompletionQueue>(local_query_indices)
            : nullptr;
        std::unique_ptr<PqFiniteReadyRingSink> ready_ring = ready_ring_mode
            ? std::make_unique<PqFiniteReadyRingSink>(local_query_indices,
                                                      ctx.total_query_capacity)
            : nullptr;
        HostCompletionSink* completion_sink = ready_ring != nullptr
            ? static_cast<HostCompletionSink*>(ready_ring.get())
            : static_cast<HostCompletionSink*>(completion_queue.get());
        AsyncIoConfig run_io_config = ctx.io_config;
        run_io_config.completion_sink = completion_sink;

        if (print_summary) {
            printf("%s\n", run_label);
            printf("Submitting kernel...\n");
        }
        if constexpr (kNativeProfileMode > 0) {
            if (!finite_mode && !ctx.native_profile_enabled) {
                fail_pq("native PQ profiling is unavailable for this execution mode");
            }
        }
        check_cuda(cudaEventRecord(ctx.start), "cudaEventRecord start");
        uint32_t finite_epoch = 0;
        if (finite_mode) {
            finite_epoch = launch_finite_init(ctx, runtime, options, nullptr, query_count);
            if (ready_ring != nullptr) ready_ring->set_context_epoch(finite_epoch);
        } else {
            const int grid_dim = static_cast<int>((query_count + kMultiQuery - 1) / kMultiQuery);
            dispatch_graph_kernel(options.search_queue_size, options.pq_dim, [&]<int QS, int PD>() {
                const auto launch = [&]<bool kUseRequestRing, bool kDeviceResident, bool kPredicateBias = false>() {
                    constexpr int kSearchThreads = kDeviceResident
                        ? kNativeResidentThreadInBlock
                        : kNativeCoupledThreadInBlock;
                    native_warp_independent_search_kernel_impl<QS,
                                                               PD,
                                                               kUseRequestRing,
                                                               kDeviceResident,
                                                               kPredicateBias>
                        <<<grid_dim,
                           kSearchThreads,
                           native_pq_workspace_shared_bytes(options.search_queue_size,
                                                            ctx.result_width, kPredicateBias)>>>(
                ctx.d_data,
                nullptr,
                ctx.d_query,
                ctx.d_result,
                ctx.d_result_dist,
                ctx.d_pq_centroid,
                ctx.d_pq_table,
                ctx.d_datadist_list,
                ctx.d_hashdata,
                static_cast<int>(query_count),
                ctx.d_time_per_block,
                ctx.d_native_profile,
                ctx.d_neighbor_memory,
                ctx.d_resident_graph,
                ctx.resident_graph_degree,
                ctx.d_resident_data_label_indptr,
                ctx.d_resident_data_label_indices,
                ctx.d_resident_query_label_indptr,
                ctx.d_resident_query_label_indices,
                ctx.d_resident_query_filter_labels,
                ctx.resident_full_label_check ? 1 : 0,
                ctx.d_status_memory,
                ctx.d_gdr_data,
                ctx.d_status_flag,
                ctx.d_request_ring_entries,
                ctx.d_request_ring_consumed_sequences,
                ctx.d_request_ring_tails,
                static_cast<uint32_t>(ctx.request_ring_capacity_per_ring),
                ctx.d_prefetch_node_array,
                ctx.d_prefetch_index,
                ctx.d_prefetch_begin_by_group,
                ctx.d_prefetch_end_by_group,
                ctx.d_query_group_ids,
                ctx.d_start_local_ids,
                ctx.d_start_sample_ids,
                ctx.d_row_to_sample,
                ctx.d_label_offset,
                ctx.d_row_filter_mask,
                ctx.d_query_filter_bits,
                ctx.d_query_range_modes,
                ctx.d_query_range_lower_bounds,
                ctx.d_query_range_upper_bounds,
                ctx.d_base_range_values,
                runtime.base_range_count,
                options.dim,
                options.pq_dim,
                static_cast<int>(ctx.use_adaptive_wait),
                static_cast<int>(ctx.buffer_mode),
                options.search_queue_size,
                options.search_finish_count,
                parse_search_early_stop_mode(options.search_early_stop_mode),
                options.search_max_steps,
                static_cast<int>(ctx.stop_width),
                static_cast<int>(ctx.result_width),
                static_cast<uint32_t>(ctx.io_config.vector_payload_words),
                static_cast<uint32_t>(ctx.io_config.vector_payload_words),
                static_cast<uint32_t>(ctx.io_config.neighbor_count_word),
                static_cast<uint32_t>(ctx.io_config.neighbor_words_offset),
                static_cast<uint32_t>(ctx.io_config.neighbor_sample_words_offset),
                static_cast<uint32_t>(ctx.io_config.elems_per_block),
                ctx.use_sample_id_payload ? 1 : 0,
                ctx.use_implicit_filter_mask ? 1 : 0,
                highfreq_gpu_exact_rerank_enabled(options) && !ctx.device_resident_graph ? 1 : 0,
                (options.raw_navigation_scores && !kDeviceResident)
                    ? 1.0f : options.bit_filter_bonus,
                (options.raw_navigation_scores && !kDeviceResident)
                    ? 1.0f : options.range_filter_bonus,
                        pq_mailbox_trace_enabled() ? 1 : 0,
                        ctx.predicate ? ctx.predicate->view : gpu_predicate::View{}, expansion.view());
                };
                if (ctx.device_resident_graph) {
                    launch.template operator()<false, true>();
                } else if (ctx.predicate && ctx.predicate->bias_frontier) {
                    if (ctx.request_ring_enabled) launch.template operator()<true, false, true>();
                    else launch.template operator()<false, false, true>();
                } else if (ctx.request_ring_enabled) {
                    launch.template operator()<true, false>();
                } else {
                    launch.template operator()<false, false>();
                }
            });
            check_cuda(cudaGetLastError(), "kernel launch");
            launch_resident_pq_batch_exact_rerank(ctx, nullptr, query_count, options);
        }
        std::atomic<bool> streaming_rerank_cancelled{false};
        std::future<void> streaming_rerank_future;
        const bool enable_streaming_rerank = collect_results && highfreq_cpu_streaming_enabled(options);

        const GraphNodeView host_view = make_graph_node_view(runtime);
        std::atomic<bool> request_ring_cancelled{false};
        const HostRequestRingView request_ring_view{
            ctx.h_request_ring_entries,
            ctx.h_request_ring_consumed_sequences,
            ctx.request_ring_enabled
                ? (query_count + ctx.request_ring_capacity_per_ring - 1) /
                    ctx.request_ring_capacity_per_ring
                : 0,
            ctx.request_ring_capacity_per_ring,
            ctx.request_ring_consumed_device_mapped,
        };
        PqHostScanThreadGuard scan_thread(
            ctx,
            local_query_indices,
            ctx.device_resident_graph
            ? std::thread{}
            : ctx.request_ring_enabled
            ? std::thread(host_request_ring_and_set,
                          ctx.status_memory,
                          ctx.h_neighbor_memory,
                          ctx.h_gdr_data,
                          ctx.status_flag,
                          host_view,
                          query_count,
                          local_query_indices.data(),
                          local_query_indices.size(),
                          std::cref(request_ring_view),
                          &request_ring_cancelled,
                          run_io_config,
                          &ctx.host_filter_stats)
            : std::thread(host_scan_and_set,
                          ctx.status_memory,
                          ctx.h_neighbor_memory,
                          ctx.h_gdr_data,
                          req_per_loop.data(),
                          ctx.status_flag,
                          host_view,
                          query_count,
                          nullptr,
                          0,
                          run_io_config,
                          &ctx.host_filter_stats),
            completion_sink,
            ctx.request_ring_enabled ? &request_ring_cancelled : nullptr);

        const auto timed_start = std::chrono::steady_clock::now();

        try {
            if (enable_streaming_rerank) {
                streaming_rerank_future = std::async(std::launch::async,
                                                     [&ctx,
                                                      &options,
                                                      &runtime,
                                                      &local_query_indices,
                                                      &results,
                                                      &step_num,
                                                      &result_distances,
                                                      &streaming_rerank_cancelled] {
                                                         run_highfreq_streaming_rerank(
                                                             ctx,
                                                             options,
                                                             runtime,
                                                             local_query_indices,
                                                             results,
                                                             step_num,
                                                             result_distances,
                                                             &streaming_rerank_cancelled);
                                                     });
            }
            if (finite_mode) {
                if (ready_ring != nullptr) {
                    drive_finite_ready_ring_scheduler(
                        ctx, runtime, options, ready_ring.get(), print_summary);
                } else {
                    drive_finite_scheduler(ctx,
                                           runtime,
                                           options,
                                           local_query_indices,
                                           finite_epoch,
                                           completion_queue.get(),
                                           print_summary);
                }
            }
            check_cuda(cudaEventRecord(ctx.stop), "cudaEventRecord stop");
            pq_debug_log("[pq-debug] before event sync label=%s collect=%d", run_label, collect_results ? 1 : 0);
            check_cuda(cudaEventSynchronize(ctx.stop), "cudaEventSynchronize stop");
            pq_debug_log("[pq-debug] after event sync label=%s collect=%d", run_label, collect_results ? 1 : 0);
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");
            pq_debug_log("[pq-debug] after device sync label=%s collect=%d", run_label, collect_results ? 1 : 0);
            check_cuda(cudaEventElapsedTime(&milliseconds, ctx.start, ctx.stop), "cudaEventElapsedTime");
            if (print_summary) {
                printf("Kernel execution time: %f ms\n", milliseconds);
                printf("Kernel execution finished, scanning for remaining requests...\n");
            }
            if (scan_thread.joinable()) scan_thread.join();
        } catch (...) {
            streaming_rerank_cancelled.store(true, std::memory_order_release);
            if (completion_sink != nullptr) completion_sink->cancel();
            publish_finished_for_host_service(ctx, local_query_indices);
            scan_thread.abort_and_join();
            if (streaming_rerank_future.valid()) {
                streaming_rerank_future.wait();
            }
            throw;
        }

        copy_time_per_block_to_host(ctx,
                                    6 * query_count,
                                    "copy_time_per_block_to_host(execute_graph_batch)");
        if constexpr (kNativeProfileMode > 0) {
            if (ctx.native_profile_enabled) {
                copy_native_profile_to_host(
                    ctx,
                    query_count,
                    "copy_native_profile_to_host(execute_graph_batch)");
            }
        }

        PqBatchStats batch_stats;
        batch_stats.query_count = query_count;
        batch_stats.kernel_ms = milliseconds;
        batch_stats.qps_sum = milliseconds > 0.0f ? (static_cast<double>(query_count) / (milliseconds / 1000.0)) : 0.0;
        batch_stats.run_count = 1;
        batch_stats.detailed_timing_available = kNativeDetailedTiming;
        batch_stats.finite_mode = finite_mode;
        record_finite_state_traffic(batch_stats, ctx, query_count);
        batch_stats.wait_io_fraction_by_query.reserve(query_count);
        for (size_t i = 0; i < query_count; i++) {
            record_pq_query_timing(batch_stats, ctx.h_time_per_block + 6 * i);
        }
        batch_stats.host_filter_stats = ctx.host_filter_stats;
        if constexpr (kNativeProfileMode > 0) {
            if (ctx.native_profile_enabled) {
                print_native_profile_summary(run_label,
                                             ctx.h_native_profile,
                                             query_count,
                                             options.search_queue_size,
                                             batch_stats.total_time_us_sum);
            }
        }

        if (collect_results && aggregate_stats != nullptr) {
            accumulate_pq_batch_stats(*aggregate_stats, batch_stats);
        }
        if (print_summary) {
            std::printf("Priority Queue Size per query: %d\n", options.search_queue_size);
            print_pq_batch_stats_summary("PQ", batch_stats);
        }

        if (collect_results) {
            if (enable_streaming_rerank) {
                streaming_rerank_future.get();
            } else {
                std::vector<idx_t> host_result;
                std::vector<float> host_result_dist;
                copy_result_buffer_to_host(ctx,
                                           query_count * ctx.result_width,
                                           host_result,
                                           "copy_result_buffer_to_host(execute_graph_batch)");
                copy_result_dist_buffer_to_host(ctx,
                                                query_count * ctx.result_width,
                                                host_result_dist,
                                                "copy_result_dist_buffer_to_host(execute_graph_batch)");
                results.clear();
                step_num.clear();
                if (result_distances != nullptr) {
                    result_distances->clear();
                }
                for (size_t i = 0; i < query_count; ++i) {
                    std::vector<idx_t> v(ctx.result_width);
                    std::vector<float> d(ctx.result_width, FLT_MAX);
                    for (uint32_t j = 0; j < ctx.result_width; ++j) {
                        v[j] = host_result[i * ctx.result_width + j];
                        d[j] = host_result_dist[i * ctx.result_width + j];
                    }
                    results.push_back(std::move(v));
                    if (result_distances != nullptr) {
                        result_distances->push_back(std::move(d));
                    }
                    step_num.push_back(static_cast<int>(ctx.h_time_per_block[6 * i + 4]));
                }
                finalize_highfreq_results(options,
                                          runtime,
                                          local_query_indices,
                                          results,
                                          result_distances,
                                          ctx.predicate && ctx.predicate->shortlist_verified,
                                          ctx.cpu_rerank_workers.get());
            }
            if (timed_wall_ms != nullptr) {
                const auto timed_end = std::chrono::steady_clock::now();
                *timed_wall_ms += std::chrono::duration<double, std::milli>(timed_end - timed_start).count();
            }
        }
        expansion.write(runtime, local_query_indices, "pq", options.graph_path, options.metadata_sidecar_path);
    };

    for (int run = 0; run < options.warmup_runs; ++run) {
        std::string label = "Warmup run " + std::to_string(run + 1) + "/" + std::to_string(options.warmup_runs);
        execute_run(label.c_str(), false);
    }
    for (int run = 0; run < options.timed_runs; ++run) {
        std::string label = "Timed run " + std::to_string(run + 1) + "/" + std::to_string(options.timed_runs);
        execute_run(label.c_str(), true);
    }
}

void execute_graph_query_ids(PqGraphSearchContext& ctx,
                             const GraphRuntime& runtime,
                             const std::vector<uint32_t>& query_ids,
                             const uint32_t* d_query_ids,
                             std::vector<std::vector<idx_t>>& results,
                             std::vector<int>& step_num,
                             std::vector<std::vector<float>>* result_distances,
                             const PqSearchOptions& options,
                             PqBatchStats* aggregate_stats = nullptr,
                             bool print_summary = true,
                             double* timed_wall_ms = nullptr,
                             bool finalize_results = true)
{
    const size_t query_count = query_ids.size();
    if (query_count == 0) {
        results.clear();
        step_num.clear();
        return;
    }
    if (query_count > ctx.query_capacity) {
        fail_pq("query ID batch exceeds graph context capacity");
    }
    if (ctx.request_ring_enabled != (options.request_mode == "batch_ring")) {
        fail_pq("PQ graph context request mode does not match execution options");
    }
    if (ctx.request_ring_enabled &&
        ctx.request_ring_consumed_device_mapped !=
            (options.request_consumed_memory == "device")) {
        fail_pq("PQ graph context request consumed memory does not match execution options");
    }
    if (defaults::pq_query_worker_mode_is_finite(options.query_worker_mode)) {
        require_valid_finite_query_ids(query_ids, ctx.total_query_capacity);
    }

    auto execute_run = [&](const char* run_label, bool collect_results) {
        const auto prepare_start = perf_trace::enabled() ? perf_trace::Clock::now() : perf_trace::Time{};
        expansion_trace::Invocation expansion(runtime, query_ids);
        pq_debug_log("[pq-debug] execute_graph_query_ids run begin label=%s collect=%d query_count=%zu",
                     run_label,
                     collect_results ? 1 : 0,
                     query_count);
        reset_search_state(ctx, ctx.total_query_capacity);

        std::vector<int> req_per_loop(20000, 0);
        float milliseconds = 0.0f;
        const bool finite_mode =
            defaults::pq_query_worker_mode_is_finite(options.query_worker_mode);
        const bool completion_queue_mode =
            use_finite_completion_fast_path(ctx, runtime, options);
        const bool ready_ring_mode = completion_queue_mode &&
            defaults::pq_finite_dispatch_mode_is_ready_ring(options.finite_dispatch_mode);
        if (ctx.io_config.completion_sink != nullptr) {
            fail_pq("persistent PQ io_config must not retain a completion sink");
        }
        std::unique_ptr<PqFiniteCompletionQueue> completion_queue =
            completion_queue_mode && !ready_ring_mode
            ? std::make_unique<PqFiniteCompletionQueue>(query_ids)
            : nullptr;
        std::unique_ptr<PqFiniteReadyRingSink> ready_ring = ready_ring_mode
            ? std::make_unique<PqFiniteReadyRingSink>(query_ids, ctx.total_query_capacity)
            : nullptr;
        HostCompletionSink* completion_sink = ready_ring != nullptr
            ? static_cast<HostCompletionSink*>(ready_ring.get())
            : static_cast<HostCompletionSink*>(completion_queue.get());
        AsyncIoConfig run_io_config = ctx.io_config;
        run_io_config.completion_sink = completion_sink;

        const GraphNodeView host_view = make_graph_node_view(runtime);
        std::atomic<bool> request_ring_cancelled{false};
        const HostRequestRingView request_ring_view{
            ctx.h_request_ring_entries,
            ctx.h_request_ring_consumed_sequences,
            ctx.request_ring_enabled
                ? (query_count + ctx.request_ring_capacity_per_ring - 1) /
                    ctx.request_ring_capacity_per_ring
                : 0,
            ctx.request_ring_capacity_per_ring,
            ctx.request_ring_consumed_device_mapped,
        };
        PqHostScanThreadGuard scan_thread(
            ctx,
            query_ids,
            ctx.device_resident_graph
            ? std::thread{}
            : ctx.request_ring_enabled
            ? std::thread(host_request_ring_and_set,
                          ctx.status_memory,
                          ctx.h_neighbor_memory,
                          ctx.h_gdr_data,
                          ctx.status_flag,
                          host_view,
                          ctx.total_query_capacity,
                          query_ids.data(),
                          query_ids.size(),
                          std::cref(request_ring_view),
                          &request_ring_cancelled,
                          run_io_config,
                          &ctx.host_filter_stats)
            : std::thread(host_scan_and_set,
                          ctx.status_memory,
                          ctx.h_neighbor_memory,
                          ctx.h_gdr_data,
                          req_per_loop.data(),
                          ctx.status_flag,
                          host_view,
                          ctx.total_query_capacity,
                          query_ids.data(),
                          query_ids.size(),
                          run_io_config,
                          &ctx.host_filter_stats),
            completion_sink,
            ctx.request_ring_enabled ? &request_ring_cancelled : nullptr);

        const auto timed_start = std::chrono::steady_clock::now();
        perf_trace::span("graph_prepare", &runtime, prepare_start, timed_start);

        if (print_summary) {
            printf("%s\n", run_label);
            printf("Submitting kernel...\n");
        }
        if constexpr (kNativeProfileMode > 0) {
            if (!finite_mode && !ctx.native_profile_enabled) {
                fail_pq("native PQ profiling is unavailable for this execution mode");
            }
        }
        check_cuda(cudaEventRecord(ctx.start), "cudaEventRecord start");
        uint32_t finite_epoch = 0;
        if (finite_mode) {
            finite_epoch = launch_finite_init(ctx,
                                              runtime,
                                              options,
                                              d_query_ids,
                                              query_count);
            if (ready_ring != nullptr) ready_ring->set_context_epoch(finite_epoch);
        } else {
            const int grid_dim = static_cast<int>((query_count + kMultiQuery - 1) / kMultiQuery);
            dispatch_graph_kernel(options.search_queue_size, options.pq_dim, [&]<int QS, int PD>() {
                const auto launch = [&]<bool kUseRequestRing, bool kDeviceResident, bool kPredicateBias = false>() {
                    constexpr int kSearchThreads = kDeviceResident
                        ? kNativeResidentThreadInBlock
                        : kNativeCoupledThreadInBlock;
                    native_warp_independent_search_kernel_impl<QS,
                                                               PD,
                                                               kUseRequestRing,
                                                               kDeviceResident,
                                                               kPredicateBias>
                        <<<grid_dim,
                           kSearchThreads,
                           native_pq_workspace_shared_bytes(options.search_queue_size,
                                                            ctx.result_width, kPredicateBias)>>>(
                ctx.d_data,
                d_query_ids,
                ctx.d_query,
                ctx.d_result,
                ctx.d_result_dist,
                ctx.d_pq_centroid,
                ctx.d_pq_table,
                ctx.d_datadist_list,
                ctx.d_hashdata,
                static_cast<int>(query_count),
                ctx.d_time_per_block,
                ctx.d_native_profile,
                ctx.d_neighbor_memory,
                ctx.d_resident_graph,
                ctx.resident_graph_degree,
                ctx.d_resident_data_label_indptr,
                ctx.d_resident_data_label_indices,
                ctx.d_resident_query_label_indptr,
                ctx.d_resident_query_label_indices,
                ctx.d_resident_query_filter_labels,
                ctx.resident_full_label_check ? 1 : 0,
                ctx.d_status_memory,
                ctx.d_gdr_data,
                ctx.d_status_flag,
                ctx.d_request_ring_entries,
                ctx.d_request_ring_consumed_sequences,
                ctx.d_request_ring_tails,
                static_cast<uint32_t>(ctx.request_ring_capacity_per_ring),
                ctx.d_prefetch_node_array,
                ctx.d_prefetch_index,
                ctx.d_prefetch_begin_by_group,
                ctx.d_prefetch_end_by_group,
                ctx.d_query_group_ids,
                ctx.d_start_local_ids,
                ctx.d_start_sample_ids,
                ctx.d_row_to_sample,
                ctx.d_label_offset,
                ctx.d_row_filter_mask,
                ctx.d_query_filter_bits,
                ctx.d_query_range_modes,
                ctx.d_query_range_lower_bounds,
                ctx.d_query_range_upper_bounds,
                ctx.d_base_range_values,
                runtime.base_range_count,
                options.dim,
                options.pq_dim,
                static_cast<int>(ctx.use_adaptive_wait),
                static_cast<int>(ctx.buffer_mode),
                options.search_queue_size,
                options.search_finish_count,
                parse_search_early_stop_mode(options.search_early_stop_mode),
                options.search_max_steps,
                static_cast<int>(ctx.stop_width),
                static_cast<int>(ctx.result_width),
                static_cast<uint32_t>(ctx.io_config.vector_payload_words),
                static_cast<uint32_t>(ctx.io_config.vector_payload_words),
                static_cast<uint32_t>(ctx.io_config.neighbor_count_word),
                static_cast<uint32_t>(ctx.io_config.neighbor_words_offset),
                static_cast<uint32_t>(ctx.io_config.neighbor_sample_words_offset),
                static_cast<uint32_t>(ctx.io_config.elems_per_block),
                ctx.use_sample_id_payload ? 1 : 0,
                ctx.use_implicit_filter_mask ? 1 : 0,
                highfreq_gpu_exact_rerank_enabled(options) && !ctx.device_resident_graph ? 1 : 0,
                (options.raw_navigation_scores && !kDeviceResident)
                    ? 1.0f : options.bit_filter_bonus,
                (options.raw_navigation_scores && !kDeviceResident)
                    ? 1.0f : options.range_filter_bonus,
                        pq_mailbox_trace_enabled() ? 1 : 0,
                        ctx.predicate ? ctx.predicate->view : gpu_predicate::View{}, expansion.view());
                };
                if (ctx.device_resident_graph) {
                    launch.template operator()<false, true>();
                } else if (ctx.predicate && ctx.predicate->bias_frontier) {
                    if (ctx.request_ring_enabled) launch.template operator()<true, false, true>();
                    else launch.template operator()<false, false, true>();
                } else if (ctx.request_ring_enabled) {
                    launch.template operator()<true, false>();
                } else {
                    launch.template operator()<false, false>();
                }
            });
            check_cuda(cudaGetLastError(), "kernel launch");
            launch_resident_pq_batch_exact_rerank(ctx, d_query_ids, query_count, options);
        }
        std::atomic<bool> streaming_rerank_cancelled{false};
        std::future<void> streaming_rerank_future;
        const bool enable_streaming_rerank = collect_results && highfreq_cpu_streaming_enabled(options);

        try {
            if (enable_streaming_rerank) {
                streaming_rerank_future = std::async(std::launch::async,
                                                     [&ctx,
                                                      &options,
                                                      &runtime,
                                                      &query_ids,
                                                      &results,
                                                      &step_num,
                                                      &result_distances,
                                                      &streaming_rerank_cancelled] {
                                                         run_highfreq_streaming_rerank(
                                                             ctx,
                                                             options,
                                                             runtime,
                                                             query_ids,
                                                             results,
                                                             step_num,
                                                             result_distances,
                                                             &streaming_rerank_cancelled);
                                                     });
            }
            if (finite_mode) {
                if (ready_ring != nullptr) {
                    drive_finite_ready_ring_scheduler(
                        ctx, runtime, options, ready_ring.get(), print_summary);
                } else {
                    drive_finite_scheduler(ctx,
                                           runtime,
                                           options,
                                           query_ids,
                                           finite_epoch,
                                           completion_queue.get(),
                                           print_summary);
                }
            }
            check_cuda(cudaEventRecord(ctx.stop), "cudaEventRecord stop");
            check_cuda(cudaEventSynchronize(ctx.stop), "cudaEventSynchronize stop");
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");
            if (perf_trace::enabled())
                perf_trace::span("graph_launch_to_complete", &runtime, timed_start, perf_trace::Clock::now());
            check_cuda(cudaEventElapsedTime(&milliseconds, ctx.start, ctx.stop), "cudaEventElapsedTime");
            if (print_summary) {
                printf("Kernel execution time: %f ms\n", milliseconds);
                printf("Kernel execution finished, scanning for remaining requests...\n");
            }
            pq_debug_log("[pq-debug] before scan join label=%s collect=%d", run_label, collect_results ? 1 : 0);
            if (scan_thread.joinable()) scan_thread.join();
            pq_debug_log("[pq-debug] execute_graph_query_ids run joined label=%s collect=%d kernel_ms=%.3f",
                         run_label,
                         collect_results ? 1 : 0,
                         milliseconds);
        } catch (...) {
            streaming_rerank_cancelled.store(true, std::memory_order_release);
            if (completion_sink != nullptr) completion_sink->cancel();
            publish_finished_for_host_service(ctx, query_ids);
            scan_thread.abort_and_join();
            if (streaming_rerank_future.valid()) {
                streaming_rerank_future.wait();
            }
            throw;
        }

        pq_debug_log("[pq-debug] execute_graph_query_ids run postcopy label=%s collect=%d",
                     run_label,
                     collect_results ? 1 : 0);

        copy_time_per_block_to_host(ctx,
                                    6 * ctx.total_query_capacity,
                                    "copy_time_per_block_to_host(execute_graph_query_ids)");
        if constexpr (kNativeProfileMode > 0) {
            if (ctx.native_profile_enabled) {
                copy_native_profile_to_host(
                    ctx,
                    query_count,
                    "copy_native_profile_to_host(execute_graph_query_ids)");
            }
        }

        PqBatchStats batch_stats;
        batch_stats.query_count = query_count;
        batch_stats.kernel_ms = milliseconds;
        batch_stats.qps_sum = milliseconds > 0.0f ? (static_cast<double>(query_count) / (milliseconds / 1000.0)) : 0.0;
        batch_stats.run_count = 1;
        batch_stats.detailed_timing_available = kNativeDetailedTiming;
        batch_stats.finite_mode = finite_mode;
        record_finite_state_traffic(batch_stats,
                                    ctx,
                                    query_count,
                                    query_ids.data());
        batch_stats.wait_io_fraction_by_query.reserve(query_count);
        for (size_t i = 0; i < query_count; i++) {
            const uint32_t query_idx = query_ids[i];
            record_pq_query_timing(batch_stats,
                                   ctx.h_time_per_block + 6 * query_idx);
        }
        batch_stats.host_filter_stats = ctx.host_filter_stats;
        if constexpr (kNativeProfileMode > 0) {
            if (ctx.native_profile_enabled) {
                print_native_profile_summary(run_label,
                                             ctx.h_native_profile,
                                             query_count,
                                             options.search_queue_size,
                                             batch_stats.total_time_us_sum);
            }
        }

        if (collect_results && aggregate_stats != nullptr) {
            accumulate_pq_batch_stats(*aggregate_stats, batch_stats);
        }
        if (print_summary) {
            std::printf("Priority Queue Size per query: %d\n", options.search_queue_size);
            print_pq_batch_stats_summary("PQ", batch_stats);
        }

        if (collect_results) {
            if (enable_streaming_rerank) {
                streaming_rerank_future.get();
                copy_time_per_block_to_host(ctx,
                                            6 * ctx.total_query_capacity,
                                            "copy_time_per_block_to_host(execute_graph_query_ids_streaming)");
                for (size_t i = 0; i < query_count; ++i) {
                    step_num[i] = static_cast<int>(ctx.h_time_per_block[6 * query_ids[i] + 4]);
                }
            } else {
                std::vector<idx_t> host_result;
                std::vector<float> host_result_dist;
                copy_result_buffer_to_host(ctx,
                                           ctx.total_query_capacity * ctx.result_width,
                                           host_result,
                                           "copy_result_buffer_to_host(execute_graph_query_ids)");
                copy_result_dist_buffer_to_host(ctx,
                                                ctx.total_query_capacity * ctx.result_width,
                                                host_result_dist,
                                                "copy_result_dist_buffer_to_host(execute_graph_query_ids)");
                results.clear();
                step_num.clear();
                if (result_distances != nullptr) {
                    result_distances->clear();
                }
                for (size_t i = 0; i < query_count; ++i) {
                    const uint32_t query_idx = query_ids[i];
                    std::vector<idx_t> v(ctx.result_width);
                    std::vector<float> d(ctx.result_width, FLT_MAX);
                    for (uint32_t j = 0; j < ctx.result_width; ++j) {
                        v[j] = host_result[static_cast<size_t>(query_idx) * ctx.result_width + j];
                        d[j] = host_result_dist[static_cast<size_t>(query_idx) * ctx.result_width + j];
                    }
                    results.push_back(std::move(v));
                    if (result_distances != nullptr) {
                        result_distances->push_back(std::move(d));
                    }
                    step_num.push_back(static_cast<int>(ctx.h_time_per_block[6 * query_idx + 4]));
                }
                if (finalize_results) {
                    finalize_highfreq_results(options,
                                              runtime,
                                              query_ids,
                                              results,
                                              result_distances,
                                              ctx.predicate && ctx.predicate->shortlist_verified,
                                          ctx.cpu_rerank_workers.get());
                }
            }
            if (timed_wall_ms != nullptr) {
                const auto timed_end = std::chrono::steady_clock::now();
                *timed_wall_ms += std::chrono::duration<double, std::milli>(timed_end - timed_start).count();
            }
        }
        expansion.write(runtime, query_ids, "pq", options.graph_path, options.metadata_sidecar_path);
    };

    for (int run = 0; run < options.warmup_runs; ++run) {
        std::string label = "Warmup run " + std::to_string(run + 1) + "/" + std::to_string(options.warmup_runs);
        execute_run(label.c_str(), false);
    }
    for (int run = 0; run < options.timed_runs; ++run) {
        std::string label = "Timed run " + std::to_string(run + 1) + "/" + std::to_string(options.timed_runs);
        execute_run(label.c_str(), true);
    }
}

// Allocated during context setup and retained until every producer/consumer joins.
struct PqPostingMappedBuffers {
    idx_t* host_result = nullptr;
    idx_t* device_result = nullptr;
    uint32_t* host_ready = nullptr;
    uint32_t* device_ready = nullptr;
    size_t rows = 0;
    uint32_t width = 0;
    ~PqPostingMappedBuffers() {
        if (host_result) cudaFreeHost(host_result);
        if (host_ready) cudaFreeHost(host_ready);
    }
};

struct PqLowfreqSearchContext {
    std::unique_ptr<PqPostingMappedBuffers> streaming;
    std::unique_ptr<PqPredicateBuffers> resident_storage;
    pq_resident_posting::View resident{};
    size_t resident_bytes = 0;
    std::unique_ptr<PqPredicateDeviceState> posting_predicate;
    std::shared_ptr<CpuRerankWorkerPool> cpu_rerank_workers;

    const pq_idx_t* d_data = nullptr;
    const pq_value_t* d_pq_centroid = nullptr;
    const value_t* d_all_query = nullptr;
    size_t query_capacity = 0;
    size_t candidate_capacity = 0;
    uint32_t shortlist_capacity = 0;

    idx_t* d_result = nullptr;
    float* d_result_dist = nullptr;
    pq_value_t* d_pq_table = nullptr;
    uint32_t* d_candidate_ids = nullptr;
    uint32_t* d_candidate_offsets = nullptr;
    uint32_t* d_candidate_sizes = nullptr;
};

void cleanup_lowfreq_context(PqLowfreqSearchContext& ctx)
{
    if (ctx.d_result != nullptr) cudaFree(ctx.d_result);
    if (ctx.d_result_dist != nullptr) cudaFree(ctx.d_result_dist);
    if (ctx.d_pq_table != nullptr) cudaFree(ctx.d_pq_table);
    if (ctx.d_candidate_ids != nullptr) cudaFree(ctx.d_candidate_ids);
    if (ctx.d_candidate_offsets != nullptr) cudaFree(ctx.d_candidate_offsets);
    if (ctx.d_candidate_sizes != nullptr) cudaFree(ctx.d_candidate_sizes);
    ctx = PqLowfreqSearchContext{};
}

void init_lowfreq_context(PqLowfreqSearchContext& ctx,
                          const PqImmutableDeviceState& immutable_state,
                          const PqPreparedQueryState& prepared_state,
                          const GraphRuntime& runtime,
                          const PqSearchOptions& options)
{
    ctx.posting_predicate = prepare_pq_posting_predicate(runtime, options, immutable_state.predicate_csr_cache);
    if (options.posting_predicate_order == "score_first" &&
        !runtime.low_query_indices.empty() && !ctx.posting_predicate)
        fail_pq("PQ score-first posting requires supported GPU CSR admission");
    ctx.cpu_rerank_workers = immutable_state.cpu_rerank_workers;
    ctx.d_data = immutable_state.d_data;
    ctx.d_pq_centroid = immutable_state.d_pq_centroid;
    ctx.d_all_query = prepared_state.d_all_query;
    if (options.lowfreq_scan_mode == "resident" && !runtime.low_query_indices.empty()) {
        const auto require = [](bool pass, const char* why) {
            if (!pass) fail_pq(std::string("PQ resident posting: ") + why);
        };
        require(ctx.posting_predicate != nullptr, "requires supported GPU CSR posting predicate");
        for (uint32_t route : runtime.low_query_indices) {
            require(route < runtime.query_vectors.rows &&
                    route < runtime.query_group_ids.size() && route < runtime.query_filter_bits.size(),
                    "route metadata is outside query domain");
            if (!runtime.route_enabled.empty() && !runtime.route_enabled.at(route)) continue;
            require(route < ctx.posting_predicate->verified_routes.size() &&
                    ctx.posting_predicate->verified_routes[route],
                    "active route has no complete GPU predicate plan");
        }
        const auto& index = runtime.lowfreq_index;
        require(index.label_offset.size() == index.label_size.size() &&
                index.sample_ids.size() == index.row_filter_mask.size(), "posting array sizes differ");
        require(index.label_offset.size() <= UINT32_MAX && index.sample_ids.size() <= UINT32_MAX,
                "posting domain exceeds uint32");
        require(runtime.query_group_ids.size() == runtime.query_vectors.rows &&
                runtime.query_filter_bits.size() == runtime.query_vectors.rows,
                "query metadata is not route aligned");
        for (size_t group = 0; group < index.label_offset.size(); ++group) {
            const size_t begin = index.label_offset[group];
            require(begin <= index.sample_ids.size() &&
                    index.label_size[group] <= index.sample_ids.size() - begin,
                    "group row interval is out of bounds");
        }
        ctx.resident_storage = std::make_unique<PqPredicateBuffers>();
        const auto upload = [&](const auto& host, const char* name) {
            ctx.resident_bytes += host.size() * sizeof(typename std::decay_t<decltype(host)>::value_type);
            return ctx.resident_storage->upload(host, name);
        };
        auto& v = ctx.resident;
        v.group_offsets = upload(index.label_offset, "PQ resident group offsets");
        v.group_sizes = upload(index.label_size, "PQ resident group sizes");
        v.sample_ids = upload(index.sample_ids, "PQ resident posting samples");
        v.row_masks = upload(index.row_filter_mask, "PQ resident posting masks");
        v.query_groups = upload(runtime.query_group_ids, "PQ resident query groups");
        v.query_masks = upload(runtime.query_filter_bits, "PQ resident query masks");
        v.groups = index.label_offset.size(); v.rows = index.sample_ids.size();
        v.samples = runtime.base_vectors.rows; v.routes = runtime.query_vectors.rows;
        if (!runtime.base_range_values.empty()) {
            const auto& ranges = runtime.query_ranges;
            require(ranges.range_count == runtime.base_range_count &&
                    runtime.base_range_values.size() == size_t(v.samples) * runtime.base_range_count,
                    "numeric sample domain mismatch");
            const size_t atoms = size_t(v.routes) * ranges.range_count;
            require(ranges.modes.size() == atoms && ranges.lower_bounds.size() == atoms &&
                    ranges.upper_bounds.size() == atoms, "numeric route domain mismatch");
            v.range_count = ranges.range_count;
            v.range_modes = upload(ranges.modes, "PQ resident range modes");
            v.range_lower = upload(ranges.lower_bounds, "PQ resident range lower");
            v.range_upper = upload(ranges.upper_bounds, "PQ resident range upper");
            // Share the full-predicate numeric allocation when it already exists.
            v.base_ranges = ctx.posting_predicate->view.base_ranges;
            if (v.base_ranges == nullptr)
                v.base_ranges = upload(runtime.base_range_values, "PQ resident base numeric values");
        }
        v.active = true;
    }
    if (options.posting_streaming && !runtime.low_query_indices.empty()) {
        ctx.streaming = std::make_unique<PqPostingMappedBuffers>();
        auto& mapped = *ctx.streaming;
        mapped.rows = runtime.low_query_indices.size();
        mapped.width = static_cast<uint32_t>(options.lowfreq_rerank_shortlist);
        check_cuda(cudaHostAlloc(&mapped.host_result, mapped.rows * mapped.width * sizeof(idx_t),
                                 cudaHostAllocMapped), "PQ posting mapped shortlist");
        check_cuda(cudaHostGetDevicePointer(&mapped.device_result, mapped.host_result, 0),
                   "PQ posting mapped shortlist device pointer");
        check_cuda(cudaHostAlloc(&mapped.host_ready, mapped.rows * sizeof(uint32_t),
                                 cudaHostAllocMapped), "PQ posting mapped ready");
        check_cuda(cudaHostGetDevicePointer(&mapped.device_ready, mapped.host_ready, 0),
                   "PQ posting mapped ready device pointer");
    }
    std::cout << "[pq-posting-scan] mode=" << options.lowfreq_scan_mode
              << " predicate_order=" << options.posting_predicate_order
              << " streaming=" << options.posting_streaming
              << " stream_batch=" << options.posting_stream_batch_size
              << " resident_bytes=" << ctx.resident_bytes << '\n';
}

void ensure_lowfreq_context_capacity(PqLowfreqSearchContext& ctx,
                                     size_t query_count,
                                     size_t candidate_count,
                                     const PqSearchOptions& options,
                                     uint32_t shortlist_k)
{
    if (query_count > ctx.query_capacity || shortlist_k > ctx.shortlist_capacity) {
        if (ctx.d_result != nullptr) cudaFree(ctx.d_result);
        if (ctx.d_result_dist != nullptr) cudaFree(ctx.d_result_dist);
        if (ctx.d_pq_table != nullptr) cudaFree(ctx.d_pq_table);
        if (ctx.d_candidate_offsets != nullptr) cudaFree(ctx.d_candidate_offsets);
        if (ctx.d_candidate_sizes != nullptr) cudaFree(ctx.d_candidate_sizes);

        check_cuda(cudaMalloc(&ctx.d_result, sizeof(idx_t) * query_count * shortlist_k), "cudaMalloc lowfreq d_result");
        check_cuda(cudaMalloc(&ctx.d_result_dist, sizeof(float) * query_count * shortlist_k), "cudaMalloc lowfreq d_result_dist");
        check_cuda(cudaMalloc(&ctx.d_pq_table, sizeof(pq_value_t) * query_count * options.pq_dim * kNumPqCentroids), "cudaMalloc lowfreq d_pq_table");
        if (!ctx.resident.active) {
            check_cuda(cudaMalloc(&ctx.d_candidate_offsets, sizeof(uint32_t) * query_count), "cudaMalloc lowfreq d_candidate_offsets");
            check_cuda(cudaMalloc(&ctx.d_candidate_sizes, sizeof(uint32_t) * query_count), "cudaMalloc lowfreq d_candidate_sizes");
        }
        ctx.query_capacity = query_count;
        ctx.shortlist_capacity = shortlist_k;
    }

    if (candidate_count > ctx.candidate_capacity) {
        if (ctx.d_candidate_ids != nullptr) cudaFree(ctx.d_candidate_ids);
        check_cuda(cudaMalloc(&ctx.d_candidate_ids, sizeof(uint32_t) * candidate_count), "cudaMalloc lowfreq d_candidate_ids");
        ctx.candidate_capacity = candidate_count;
    }
}

int run_lowfreq_pq_search_with_context(const PqSearchOptions& options,
                                       const GraphRuntime& runtime,
                                       const std::vector<uint32_t>& query_indices,
                                       const uint32_t* d_query_ids,
                                       PqLowfreqSearchContext& ctx,
                                       std::vector<std::vector<idx_t>>& out_results,
                                       std::vector<int>& out_step_nums,
                                       std::vector<std::vector<float>>* out_distances)
{
    if (query_indices.empty()) {
        out_results.clear();
        out_step_nums.clear();
        return 0;
    }

    const auto candidate_start = perf_trace::enabled() ? perf_trace::Clock::now() : perf_trace::Time{};
    const uint32_t topk = static_cast<uint32_t>(options.topk);
    const uint32_t shortlist_k = options.enable_lowfreq_rerank
        ? static_cast<uint32_t>(options.lowfreq_rerank_shortlist)
        : topk;
    const bool pre_filter_full_labels =
        defaults::lowfreq_full_filter_mode_is_pre(options.lowfreq_full_filter_mode);
    const bool post_filter_full_labels =
        defaults::lowfreq_full_filter_mode_is_post(options.lowfreq_full_filter_mode) ||
        ctx.posting_predicate != nullptr;
    std::vector<uint32_t> candidate_offsets(ctx.resident.active ? 0 : query_indices.size(), 0);
    std::vector<uint32_t> candidate_sizes(ctx.resident.active ? 0 : query_indices.size(), 0);
    std::vector<uint32_t> candidate_ids;

    if (!ctx.resident.active) {
        for (size_t i = 0; i < query_indices.size(); ++i) {
            const uint32_t query_idx = query_indices[i];
            if (query_idx >= runtime.query_vectors.rows) {
                fail_pq("lowfreq query id out of range");
            }
            const uint32_t label = runtime.query_group_ids[query_idx];
            if (label == UINT32_MAX ||
                label >= runtime.lowfreq_index.label_offset.size() ||
                label >= runtime.lowfreq_index.label_size.size()) {
                continue;
            }
            const uint64_t query_filter_bit = runtime.query_filter_bits[query_idx];
            candidate_offsets[i] = static_cast<uint32_t>(candidate_ids.size());
            const uint32_t begin = runtime.lowfreq_index.label_offset[label];
            const uint32_t count = runtime.lowfreq_index.label_size[label];
            for (uint32_t j = 0; j < count; ++j) {
                const uint32_t row = begin + j;
                if (row >= runtime.lowfreq_index.sample_ids.size() ||
                    row >= runtime.lowfreq_index.row_filter_mask.size()) {
                    continue;
                }
                if ((runtime.lowfreq_index.row_filter_mask[row] & query_filter_bit) == 0) {
                    continue;
                }
                const uint32_t sample_id = runtime.lowfreq_index.sample_ids[row];
                if (sample_id >= runtime.base_vectors.rows) {
                    continue;
                }
                if (!runtime.base_range_values.empty()) {
                    if (!go_fann::range_filter::sample_matches_all_active_ranges_host(runtime.query_ranges,
                                                                                            query_idx,
                                                                                            runtime.base_range_values,
                                                                                            runtime.base_range_count,
                                                                                            sample_id)) {
                        continue;
                    }
                }
                if (pre_filter_full_labels &&
                    !sample_matches_route_predicate(runtime, query_idx, sample_id)) {
                    continue;
                }
                if (candidate_ids.size() >= std::numeric_limits<uint32_t>::max()) {
                    fail_pq("lowfreq candidate list exceeds supported size");
                }
                candidate_ids.push_back(sample_id);
                ++candidate_sizes[i];
            }
        }

    } else {
        for (uint32_t route : query_indices) {
            if (route >= ctx.resident.routes) fail_pq("resident posting query ID out of range");
        }
    }

    if (perf_trace::enabled())
        perf_trace::span("lowfreq_candidate_prepare", &runtime, candidate_start, perf_trace::Clock::now());
    {
        perf_trace::Scope preparation("lowfreq_prepare", &runtime);
        ensure_lowfreq_context_capacity(ctx,
                                        query_indices.size(),
                                        candidate_ids.size(),
                                        options,
                                        shortlist_k);

    }
    if (!ctx.resident.active) {
        perf_trace::Scope copy("lowfreq_h2d", &runtime);
        check_cuda(cudaMemcpy(ctx.d_candidate_offsets,
                              candidate_offsets.data(),
                              sizeof(uint32_t) * candidate_offsets.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy lowfreq d_candidate_offsets");
        check_cuda(cudaMemcpy(ctx.d_candidate_sizes,
                              candidate_sizes.data(),
                              sizeof(uint32_t) * candidate_sizes.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy lowfreq d_candidate_sizes");
        if (!candidate_ids.empty()) {
            check_cuda(cudaMemcpy(ctx.d_candidate_ids,
                                  candidate_ids.data(),
                                  sizeof(uint32_t) * candidate_ids.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy lowfreq d_candidate_ids");
        }

    }

    auto consume_stream = [&](const std::atomic<bool>& cancelled) {
        out_results.assign(query_indices.size(), std::vector<idx_t>(topk, UINT32_MAX));
        out_step_nums.assign(query_indices.size(), 0);
        if (out_distances) out_distances->assign(query_indices.size(), std::vector<float>(topk, FLT_MAX));
        pq_posting_streaming::consume_ready(query_indices.size(), options.posting_stream_batch_size,
            [&](size_t pos) {
                return __atomic_load_n(ctx.streaming->host_ready + pos, __ATOMIC_ACQUIRE) != 0;
            }, [&](const std::vector<size_t>& positions) {
                std::vector<uint32_t> routes;
                std::vector<idx_t> shortlists;
                routes.reserve(positions.size());
                shortlists.reserve(positions.size() * shortlist_k);
                for (size_t pos : positions) {
                    routes.push_back(query_indices[pos]);
                    const idx_t* row = ctx.streaming->host_result + pos * shortlist_k;
                    shortlists.insert(shortlists.end(), row, row + shortlist_k);
                }
                std::vector<std::vector<idx_t>> results;
                std::vector<std::vector<float>> distances;
                std::vector<int> steps;
                lowfreq_host_heuristic_rerank(runtime, runtime.query_vectors, routes, shortlists,
                    shortlist_k, topk, options.lowfreq_rerank_batch_size,
                    options.lowfreq_rerank_epsilon, options.lowfreq_rerank_stable_batches,
                    post_filter_full_labels, options.highfreq_rerank_threads,
                    results, steps, out_distances ? &distances : nullptr,
                    ctx.cpu_rerank_workers.get(),
                    ctx.posting_predicate ? &ctx.posting_predicate->verified_routes : nullptr,
                    options.exact_distance_memo.get());
                for (size_t i = 0; i < positions.size(); ++i) {
                    out_results[positions[i]] = std::move(results[i]);
                    out_step_nums[positions[i]] = steps[i];
                    if (out_distances) (*out_distances)[positions[i]] = std::move(distances[i]);
                }
            }, cancelled);
    };

    auto execute_run = [&](bool collect_results) {
        const bool stream = collect_results && options.posting_streaming;
        std::atomic<bool> cancelled{false};
        std::future<void> consumer;
        if (stream) {
            if (!ctx.streaming || query_indices.size() > ctx.streaming->rows ||
                shortlist_k != ctx.streaming->width)
                fail_pq("PQ posting stream buffers do not cover this batch");
            std::fill_n(ctx.streaming->host_ready, query_indices.size(), 0U);
        }
        try {
            const auto kernel_start = perf_trace::enabled() ? perf_trace::Clock::now() : perf_trace::Time{};
            lowfreq_pq_scan_kernel<<<query_indices.size(), kThreadInBlock>>>(
                ctx.d_data,
                ctx.d_all_query,
                d_query_ids,
                ctx.d_result,
                ctx.d_result_dist,
                ctx.d_pq_centroid,
                ctx.d_pq_table,
                candidate_ids.empty() ? nullptr : ctx.d_candidate_ids,
                ctx.d_candidate_offsets,
                ctx.d_candidate_sizes,
                static_cast<int>(query_indices.size()),
                static_cast<int>(topk),
                static_cast<int>(shortlist_k),
                options.dim,
                options.pq_dim,
                ctx.posting_predicate ? ctx.posting_predicate->view : gpu_predicate::View{},
                ctx.resident, options.posting_predicate_order == "score_first",
                stream ? ctx.streaming->device_result : nullptr,
                stream ? ctx.streaming->device_ready : nullptr);
            check_cuda(cudaGetLastError(), "lowfreq kernel launch");
            // Coordinator is outside the shared rerank pool. CPU reads only rows
            // published by the device while remaining blocks continue scanning.
            if (stream) consumer = std::async(std::launch::async, [&] { consume_stream(cancelled); });
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize lowfreq kernel");
            if (perf_trace::enabled())
                perf_trace::span("lowfreq_kernel_launch_to_complete", &runtime, kernel_start, perf_trace::Clock::now());
            if (stream) { consumer.get(); return; }
        } catch (...) {
            cancelled.store(true, std::memory_order_release);
            if (consumer.valid()) consumer.wait();
            // Also drain a launched scan if thread creation or a worker failed.
            cudaDeviceSynchronize();
            throw;
        }

        if (collect_results) {
            std::vector<idx_t> host_result(query_indices.size() * shortlist_k, UINT32_MAX);
            {
                perf_trace::Scope copy("lowfreq_shortlist_d2h", &runtime);
                check_cuda(cudaMemcpy(host_result.data(),
                                      ctx.d_result,
                                      sizeof(idx_t) * host_result.size(),
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy lowfreq result");
            }
            if (options.enable_lowfreq_rerank) {
                lowfreq_host_heuristic_rerank(runtime,
                                              runtime.query_vectors,
                                              query_indices,
                                              host_result,
                                              shortlist_k,
                                              topk,
                                              static_cast<uint32_t>(options.lowfreq_rerank_batch_size),
                                              options.lowfreq_rerank_epsilon,
                                              static_cast<uint32_t>(options.lowfreq_rerank_stable_batches),
                                              post_filter_full_labels,
                                              options.highfreq_rerank_threads,
                                              out_results,
                                              out_step_nums,
                                              out_distances,
                                              ctx.cpu_rerank_workers.get(),
                                              ctx.posting_predicate ? &ctx.posting_predicate->verified_routes : nullptr,
                                              options.exact_distance_memo.get());
            } else {
                out_results.assign(query_indices.size(), std::vector<idx_t>(topk, UINT32_MAX));
                out_step_nums.assign(query_indices.size(), 1);
                std::vector<float> host_distances(query_indices.size() * shortlist_k, FLT_MAX);
                {
                    perf_trace::Scope copy("lowfreq_distance_d2h", &runtime);
                    check_cuda(cudaMemcpy(host_distances.data(),
                                          ctx.d_result_dist,
                                          sizeof(float) * host_distances.size(),
                                          cudaMemcpyDeviceToHost),
                               "cudaMemcpy lowfreq result distances");
                }
                if (out_distances != nullptr) {
                    out_distances->assign(query_indices.size(), std::vector<float>(topk, FLT_MAX));
                }
                for (size_t q = 0; q < query_indices.size(); ++q) {
                    for (uint32_t i = 0; i < topk; ++i) {
                        const size_t offset = q * shortlist_k + i;
                        const idx_t sample = host_result[offset];
                        if (ctx.posting_predicate &&
                            (sample == UINT32_MAX || sample >= runtime.base_vectors.rows ||
                             (!pq_posting_streaming::predicate_verified(
                                 &ctx.posting_predicate->verified_routes, query_indices[q]) &&
                              !sample_matches_lowfreq_full_labels(runtime, query_indices[q], sample)))) continue;
                        out_results[q][i] = sample;
                        if (out_distances != nullptr) {
                            (*out_distances)[q][i] = host_distances[offset];
                        }
                    }
                }
            }
        }
    };

    for (int run = 0; run < options.warmup_runs; ++run) {
        execute_run(false);
    }
    for (int run = 0; run < options.timed_runs; ++run) {
        execute_run(true);
    }

    return 0;
}

void init_benchmark_query_contexts(
    PqPreparedQueryState& prepared,
    PqGraphSearchContext& graph_ctx,
    PqLowfreqSearchContext& lowfreq_ctx,
    const PqImmutableDeviceState& immutable,
    const GraphRuntime& runtime,
    const PqSearchOptions& base_options)
{
    try {
        init_prepared_query_state(prepared, runtime, base_options);
        const auto context_start = std::chrono::steady_clock::now();
        try {
            init_lowfreq_context(lowfreq_ctx, immutable, prepared, runtime, base_options);
            const bool has_potential_high = std::any_of(runtime.query_group_ids.begin(),
                                                        runtime.query_group_ids.end(),
                                                        [](uint32_t label) { return label != UINT32_MAX; }) &&
                std::any_of(runtime.query_is_lowfreq.begin(),
                            runtime.query_is_lowfreq.end(),
                            [](uint8_t flag) { return flag == 0; });
            if (!has_potential_high) {
                prepared.graph_lowfreq_context_init_ms =
                    setup_elapsed_ms(context_start, std::chrono::steady_clock::now());
                return;
            }
            require_pq_graph_dim_supported(base_options.dim);
            require_pq_graph_pq_dim_supported(base_options.pq_dim);
            init_graph_context_prepared(graph_ctx,
                                        runtime.query_vectors.rows,
                                        immutable,
                                        prepared,
                                        runtime,
                                        base_options);
            prepared.graph_lowfreq_context_init_ms =
                setup_elapsed_ms(context_start, std::chrono::steady_clock::now());
        } catch (...) {
            cleanup_graph_context(graph_ctx);
            cleanup_lowfreq_context(lowfreq_ctx);
            throw;
        }
    } catch (...) {
        cleanup_prepared_query_state(prepared);
        throw;
    }
}

void init_benchmark_contexts(PqImmutableDeviceState& immutable,
                             PqPreparedQueryState& prepared,
                             PqGraphSearchContext& graph_ctx,
                             PqLowfreqSearchContext& lowfreq_ctx,
                             const GraphRuntime& runtime,
                             const PqHostIndex& pq_index,
                             const PqSearchOptions& base_options)
{
    init_immutable_state(immutable, pq_index, base_options);
    try {
        init_benchmark_query_contexts(prepared,
                                      graph_ctx,
                                      lowfreq_ctx,
                                      immutable,
                                      runtime,
                                      base_options);
    } catch (...) {
        cleanup_immutable_state(immutable);
        throw;
    }
}

PqIterationResult execute_single_iteration(PqGraphSearchContext& graph_ctx,
                                           PqLowfreqSearchContext& lowfreq_ctx,
                                           const PqPreparedQueryState& prepared,
                                           const PqImmutableDeviceState& immutable,
                                           GraphRuntime& runtime,
                                           const PqSearchOptions& iteration_options,
                                           const std::function<void(int, const PqTimedRunMetrics&)>& after_timed_run)
{
    const auto point_buffer_start = std::chrono::steady_clock::now();
    (void)immutable;
    PqSearchOptions single_run_base_options = iteration_options;
    single_run_base_options.warmup_runs = 0;
    single_run_base_options.timed_runs = 1;
    const size_t highfreq_width = static_cast<size_t>(highfreq_result_width(iteration_options));

    std::vector<std::vector<idx_t>> route_results(runtime.query_vectors.rows,
                                                  std::vector<idx_t>(highfreq_width, UINT32_MAX));
    std::vector<std::vector<float>> route_distances(runtime.query_vectors.rows,
                                                    std::vector<float>(highfreq_width, FLT_MAX));
    std::vector<int> route_steps(runtime.query_vectors.rows, 0);
    std::vector<std::vector<idx_t>> merged_results;
    std::vector<int> merged_steps;
    PqBatchStats highfreq_stats;
    size_t lowfreq_query_count = 0;
    double lowfreq_wall_ms = 0.0;
    double mixed_qps_sum = 0.0;
    size_t mixed_run_count = 0;
    int64_t timed_total_steps_sum = 0;
    int64_t timed_highfreq_steps_sum = 0;
    int64_t timed_lowfreq_steps_sum = 0;
    GraphRouteSplitStats final_split_stats;
    const bool has_graph_ctx = graph_ctx.d_data != nullptr;

    if (prepared.setup_report != nullptr) {
        prepared.setup_report->append({{"record_type", "point_buffers"},
            {"point_id", prepared.setup_point_id},
            {"result_buffers_and_options_ms",
                setup_elapsed_ms(point_buffer_start, std::chrono::steady_clock::now())},
            {"frequency", "once_per_point"}, {"included_in_qps", false}});
    }
    auto execute_iteration = [&](bool collect_results, bool full_rerank_warmup, int run) {
        std::optional<perf_trace::Capture> capture;
        if (collect_results) capture.emplace("pq", prepared.setup_point_id, "timed",
                                             iteration_options.search_queue_size, run);
        PqSearchOptions single_run_options = single_run_base_options;
        const auto pre_qps_start = std::chrono::steady_clock::now();
        PqTimedRunMetrics run_metrics;
        ExactDistanceMemo::Stats memo_stats;
        const bool run_full_results = collect_results || full_rerank_warmup;
        const auto iteration_wall_start = std::chrono::steady_clock::now();
        single_run_options.exact_distance_memo = make_pq_iteration_distance_memo(runtime, single_run_options);
        const bool memo_enabled = single_run_options.exact_distance_memo != nullptr;
        PqSearchOptions graph_run_options = single_run_options;
        if (!run_full_results) {
            graph_run_options.highfreq_rerank_streaming = false;
        }
        std::vector<uint32_t> high_query_ids;
        std::vector<uint32_t> low_query_ids;
        build_timed_query_routes(prepared, high_query_ids, low_query_ids);
        reorder_high_query_ids_by_graph_instance(runtime, high_query_ids);
        sync_query_ids_to_device(high_query_ids,
                                 prepared.d_high_query_ids,
                                 "cudaMemcpy reordered high_query_ids");

        const auto route_partition_end = std::chrono::steady_clock::now();
        perf_trace::span("iteration_prepare", &runtime, iteration_wall_start, route_partition_end);
        double highfreq_timed_ms = 0.0;
        std::vector<uint32_t> pending_high_query_ids;
        std::vector<std::vector<idx_t>> pending_high_results;
        std::vector<std::vector<float>> pending_high_dists;
        std::vector<int> pending_high_steps;
        std::future<void> highfreq_rerank_future;
        if (!high_query_ids.empty()) {
            if (!has_graph_ctx) {
                fail_pq("benchmark graph context is not initialized for high-frequency queries");
            }
            std::vector<std::vector<idx_t>> high_results;
            std::vector<std::vector<float>> high_dists;
            std::vector<int> high_steps;
            PqBatchStats run_highfreq_stats;
            const bool async_highfreq_cpu_rerank =
                collect_results &&
                defaults::highfreq_rerank_mode_is_cpu(graph_run_options.highfreq_rerank_mode) &&
                !highfreq_cpu_streaming_enabled(graph_run_options);
            execute_graph_query_ids(graph_ctx,
                                    runtime,
                                    high_query_ids,
                                    prepared.d_high_query_ids,
                                    high_results,
                                    high_steps,
                                    &high_dists,
                                    graph_run_options,
                                    collect_results ? &run_highfreq_stats : nullptr,
                                    false,
                                    collect_results ? &highfreq_timed_ms : nullptr,
                                    run_full_results && !async_highfreq_cpu_rerank);
            if (collect_results) {
                accumulate_pq_batch_stats(highfreq_stats, run_highfreq_stats);
                if (async_highfreq_cpu_rerank) {
                    pending_high_query_ids = high_query_ids;
                    pending_high_results = std::move(high_results);
                    pending_high_dists = std::move(high_dists);
                    pending_high_steps = std::move(high_steps);
                    highfreq_rerank_future = std::async(cpu_rerank_schedule::host_finalize_launch(single_run_options.cpu_rerank_schedule),
                                                        [&runtime,
                                                         &single_run_options,
                                                         &pending_high_query_ids,
                                                         &pending_high_results,
                                                         &pending_high_dists,
                                                         &graph_ctx,
                                                         shortlist_verified = graph_ctx.predicate && graph_ctx.predicate->shortlist_verified] {
                                                            finalize_highfreq_results(single_run_options,
                                                                                      runtime,
                                                                                      pending_high_query_ids,
                                                                                      pending_high_results,
                                                                                      &pending_high_dists, shortlist_verified,
                                                                                      graph_ctx.cpu_rerank_workers.get());
                                                        });
                } else {
                    merge_subset_results(high_query_ids, route_results, route_steps, high_results, high_steps);
                    for (size_t ii = 0; ii < high_query_ids.size(); ++ii) {
                        route_distances[high_query_ids[ii]] = std::move(high_dists[ii]);
                    }
                }
            }
        }

        double lowfreq_ms = 0.0;
        if (!low_query_ids.empty()) {
            std::vector<std::vector<idx_t>> low_results;
            std::vector<std::vector<float>> low_dists;
            std::vector<int> low_steps;
            const auto lowfreq_start = std::chrono::steady_clock::now();
            run_lowfreq_pq_search_with_context(single_run_options,
                                               runtime,
                                               low_query_ids,
                                               prepared.d_low_query_ids,
                                               lowfreq_ctx,
                                               low_results,
                                               low_steps,
                                               &low_dists);
            const auto lowfreq_end = std::chrono::steady_clock::now();
            lowfreq_ms = std::chrono::duration<double, std::milli>(lowfreq_end - lowfreq_start).count();
            if (collect_results) {
                merge_subset_results(low_query_ids, route_results, route_steps, low_results, low_steps);
                for (size_t ii = 0; ii < low_query_ids.size(); ++ii) {
                    route_distances[low_query_ids[ii]] = std::move(low_dists[ii]);
                }
                lowfreq_query_count += low_query_ids.size();
                lowfreq_wall_ms += lowfreq_ms;
            }
        }

        if (collect_results) {
            if (highfreq_rerank_future.valid()) {
                highfreq_rerank_future.get();
                merge_subset_results(pending_high_query_ids,
                                     route_results,
                                     route_steps,
                                     pending_high_results,
                                     pending_high_steps);
                for (size_t ii = 0; ii < pending_high_query_ids.size(); ++ii) {
                    route_distances[pending_high_query_ids[ii]] = std::move(pending_high_dists[ii]);
                }
            }
            std::vector<std::vector<idx_t>> iteration_merged_results;
            std::vector<int> iteration_merged_steps;
            merge_route_results_to_original(runtime,
                                            route_results,
                                            route_distances,
                                            route_steps,
                                            static_cast<uint32_t>(iteration_options.topk),
                                            iteration_merged_results,
                                            iteration_merged_steps);
            merged_results = std::move(iteration_merged_results);
            graph_run_options.exact_distance_memo.reset();
            memo_stats = finish_pq_distance_memo(single_run_options);
            const auto iteration_wall_end = std::chrono::steady_clock::now();
            perf_trace::span("provider_search_window", &runtime, iteration_wall_start, iteration_wall_end);
            const double run_wall_ms =
                std::chrono::duration<double, std::milli>(iteration_wall_end - iteration_wall_start).count();
            int64_t run_total_steps = 0;
            for (int step : route_steps) {
                run_total_steps += static_cast<int64_t>(step);
            }
            final_split_stats = collect_graph_route_split_stats(runtime, route_steps);
            timed_total_steps_sum += run_total_steps;
            timed_highfreq_steps_sum += final_split_stats.highfreq_total_steps;
            timed_lowfreq_steps_sum += final_split_stats.lowfreq_total_steps;
            const size_t original_query_count = runtime.original_query_count;
            if (original_query_count > 0 && run_wall_ms > 0.0) {
                run_metrics.qps = static_cast<double>(original_query_count) / (run_wall_ms / 1000.0);
                mixed_qps_sum += run_metrics.qps;
                mixed_run_count += 1;
            }
            run_metrics.wall_ms = run_wall_ms;
            run_metrics.query_count = original_query_count;
            run_metrics.total_steps = run_total_steps;
            run_metrics.highfreq_steps = final_split_stats.highfreq_total_steps;
            run_metrics.lowfreq_steps = final_split_stats.lowfreq_total_steps;
        }
        if (!collect_results) {
            graph_run_options.exact_distance_memo.reset();
            memo_stats = finish_pq_distance_memo(single_run_options);
        }
        if (prepared.setup_report != nullptr) {
            prepared.setup_report->append({{"record_type", "iteration"},
                {"point_id", prepared.setup_point_id},
                {"phase", collect_results ? "timed" : "warmup"}, {"run_index", run},
                {"query_count", runtime.original_query_count},
                {"qps_start_since_benchmark_entry_ms",
                    setup_elapsed_ms(prepared.benchmark_start, iteration_wall_start)},
                {"excluded_pre_qps_ms", setup_elapsed_ms(pre_qps_start, iteration_wall_start)},
                {"route_partition_and_upload_ms", setup_elapsed_ms(iteration_wall_start, route_partition_end)},
                {"route_partition_already_in_qps", true},
                {"execution_wall_ms", collect_results ? run_metrics.wall_ms :
                    setup_elapsed_ms(iteration_wall_start, std::chrono::steady_clock::now())},
                {"qps", run_metrics.qps}});
        }
        report_pq_distance_memo(memo_stats, collect_results, single_run_options.rerank_distance_reuse, memo_enabled);
        if (capture) {
            perf_trace::value("reported_search_wall_ms", &runtime, run_metrics.wall_ms);
            capture->finish();
        }
        return run_metrics;
    };

    for (int run = 0; run < iteration_options.warmup_runs; ++run) {

        std::cout << "[benchmark warmup start] queue_size="
                  << iteration_options.search_queue_size
                  << " run=" << run << '/' << iteration_options.warmup_runs << "\n";
        (void)execute_iteration(false, iteration_options.full_rerank_warmup, run);
        std::cout << "[benchmark warmup end] queue_size="
                  << iteration_options.search_queue_size
                  << " run=" << run << '/' << iteration_options.warmup_runs << "\n";
        std::cout.flush();
    }
    for (int run = 0; run < iteration_options.timed_runs; ++run) {


        std::cout << "[benchmark timed-run start] queue_size="
                  << iteration_options.search_queue_size
                  << " run=" << run << '/' << iteration_options.timed_runs << "\n";
        std::cout.flush();
        const PqTimedRunMetrics run_metrics = execute_iteration(true, false, run);
        if (after_timed_run) {
            after_timed_run(run, run_metrics);
        }
        std::cout << "[benchmark timed-run end] queue_size="
                  << iteration_options.search_queue_size
                  << " run=" << run << '/' << iteration_options.timed_runs
                  << " wall_ms=" << run_metrics.wall_ms
                  << " qps=" << run_metrics.qps
                  << " queries=" << run_metrics.query_count << "\n";
        std::cout.flush();
    }

    if (route_results.size() != runtime.query_vectors.rows ||
        route_distances.size() != runtime.query_vectors.rows ||
        route_steps.size() != runtime.query_vectors.rows ||
        runtime.route_to_original_query.size() != runtime.query_vectors.rows) {
        fail_pq("benchmark result routing state has inconsistent route counts before merge");
    }
    std::cout << "[benchmark postprocess start] routed_queries=" << route_results.size()
              << " original_queries=" << runtime.original_query_count << "\n";
    std::cout.flush();
    merge_route_results_to_original(runtime,
                                    route_results,
                                    route_distances,
                                    route_steps,
                                    static_cast<uint32_t>(iteration_options.topk),
                                    merged_results,
                                    merged_steps);
    std::cout << "[benchmark postprocess end] result_rows=" << merged_results.size() << "\n";
    std::cout.flush();

    const double kernel_ms = highfreq_stats.run_count > 0
        ? (highfreq_stats.kernel_ms / static_cast<double>(highfreq_stats.run_count))
        : 0.0;
    const int64_t total_steps = mixed_run_count > 0
        ? (timed_total_steps_sum / static_cast<int64_t>(mixed_run_count))
        : 0;
    if (mixed_run_count == 0) {
        final_split_stats = collect_graph_route_split_stats(runtime, route_steps);
    }
    const int64_t highfreq_total_steps = mixed_run_count > 0
        ? (timed_highfreq_steps_sum / static_cast<int64_t>(mixed_run_count))
        : final_split_stats.highfreq_total_steps;
    const int64_t lowfreq_total_steps = mixed_run_count > 0
        ? (timed_lowfreq_steps_sum / static_cast<int64_t>(mixed_run_count))
        : final_split_stats.lowfreq_total_steps;
    std::vector<int> route_flags = build_graph_original_route_flags(runtime);
    const double qps = mixed_run_count > 0
        ? (mixed_qps_sum / static_cast<double>(mixed_run_count))
        : 0.0;
    if (highfreq_stats.query_count > 0) {
        print_pq_state_summary("PQ benchmark highfreq", highfreq_stats);
        print_pq_request_publish_summary("PQ benchmark highfreq", highfreq_stats);
        print_pq_d2_availability_summary("PQ benchmark highfreq",
                                         highfreq_stats,
                                         qps);
    }
    const auto& host_stats = graph_ctx.host_filter_stats;
    const auto total_requests = host_stats.requests_served;
    float pass_ratio = total_requests > 0
        ? static_cast<float>(host_stats.requests_passed) / static_cast<float>(total_requests)
        : 0.0f;
    int workers_thread = host_stats.workers_thread;
    if (lowfreq_query_count > 0 && !has_graph_ctx) {
        workers_thread = 0;
        pass_ratio = 0.0f;
    }
    return PqIterationResult{
        std::move(merged_results),
        std::move(merged_steps),
        qps,
        kernel_ms,
        mixed_run_count,
        total_steps,
        final_split_stats.highfreq_route_count,
        final_split_stats.lowfreq_route_count,
        final_split_stats.unroutable_route_count,
        highfreq_total_steps,
        lowfreq_total_steps,
        pass_ratio,
        workers_thread,
        std::move(route_flags),
    };
}

void cleanup_benchmark_contexts(PqImmutableDeviceState& immutable,
                                PqPreparedQueryState& prepared,
                                PqGraphSearchContext& graph_ctx,
                                PqLowfreqSearchContext& lowfreq_ctx)
{
    cleanup_lowfreq_context(lowfreq_ctx);
    cleanup_graph_context(graph_ctx);
    cleanup_prepared_query_state(prepared);
    cleanup_immutable_state(immutable);
}

void native_astar_multi_start_search_batch(
    const std::vector<std::vector<std::pair<int, value_t>>>& queries,
    std::vector<std::vector<idx_t>>& results,
    std::vector<int>& step_num,
    std::vector<std::vector<float>>* result_distances,
    const PqImmutableDeviceState& immutable_state,
    const GraphRuntime& runtime,
    const PqSearchOptions& options)
{
    const size_t query_count = queries.size();
    if (query_count == 0) {
        fail_pq("query batch is empty");
    }
    if (query_count != runtime.query_group_ids.size() || query_count != runtime.start_local_ids.size()) {
        fail_pq("query_count does not match runtime arrays");
    }

    PqGraphSearchContext ctx;
    init_graph_context(ctx, query_count, immutable_state, runtime, options);
    try {
        execute_graph_batch(ctx, queries, runtime, results, step_num, result_distances, options);
        cleanup_graph_context(ctx);
    } catch (...) {
        cleanup_graph_context(ctx);
        throw;
    }
}
void validate_pq_search_options(const PqSearchOptions& options)
{
    expansion_trace::validate_options(options);
    if (options.warmup_runs < 0 || options.timed_runs <= 0) {
        fail_pq("native PQ search requires warmup_runs >= 0 and timed_runs >= 1");
    }

    if (options.degree <= 0 || options.degree > FIXED_DEGREE) {
        fail_pq("native PQ search currently supports degree in [1, 64]");
    }
    if (options.topk <= 0 || options.topk > kTopk) {
        fail_pq("native PQ search currently supports topk in [1, 10]");
    }
    if (options.cpu_persistent_workers &&
        (options.query_worker_mode != "coupled" || options.latency_persistent_queue)) {
        fail_pq("CPU persistent workers require coupled PQ search/benchmark");
    }
    if (!defaults::is_valid_highfreq_rerank_mode(options.highfreq_rerank_mode)) {
        fail_pq("native PQ search highfreq_rerank_mode must be one of: gpu, cpu, off");
    }
    if (!defaults::is_valid_highfreq_rerank_algorithm(options.highfreq_rerank_algorithm)) {
        fail_pq("native PQ search highfreq_rerank_algorithm must be one of: exact, heuristic");
    }
    if (options.highfreq_rerank_shortlist < options.topk ||
        options.highfreq_rerank_shortlist > kHighfreqRerankShortlistMax) {
        fail_pq("native PQ search highfreq_rerank_shortlist must be in [topk, 512]");
    }
    if (options.highfreq_rerank_batch_size <= 0) {
        fail_pq("native PQ search highfreq_rerank_batch_size must be >= 1");
    }
    if (options.highfreq_rerank_epsilon < 0.0f) {
        fail_pq("native PQ search highfreq_rerank_epsilon must be >= 0");
    }
    if (options.highfreq_rerank_stable_batches <= 0) {
        fail_pq("native PQ search highfreq_rerank_stable_batches must be >= 1");
    }
    if (options.highfreq_rerank_threads <= 0) {
        fail_pq("native PQ search highfreq_rerank_threads must be >= 1");
    }
    if (options.highfreq_rerank_stream_batch_size <= 0) {
        fail_pq("native PQ search highfreq_rerank_stream_batch_size must be >= 1");
    }
    if (!defaults::highfreq_rerank_mode_is_cpu(options.highfreq_rerank_mode) &&
        (options.highfreq_rerank_streaming ||
         options.highfreq_rerank_stream_batch_size != defaults::default_highfreq_rerank_stream_batch_size)) {
        fail_pq("native PQ search highfreq_rerank_streaming requires highfreq_rerank_mode=cpu");
    }
    if (options.enable_lowfreq_rerank) {
        if (options.lowfreq_rerank_shortlist < options.topk ||
            options.lowfreq_rerank_shortlist > kLowfreqRerankShortlistMax) {
            fail_pq("native PQ search lowfreq_rerank_shortlist must be in [topk, 400]");
        }
        if (options.lowfreq_rerank_batch_size <= 0) {
            fail_pq("native PQ search lowfreq_rerank_batch_size must be >= 1");
        }
        if (options.lowfreq_rerank_epsilon < 0.0f) {
            fail_pq("native PQ search lowfreq_rerank_epsilon must be >= 0");
        }
        if (options.lowfreq_rerank_stable_batches <= 0) {
            fail_pq("native PQ search lowfreq_rerank_stable_batches must be >= 1");
        }
    }
    if (options.posting_streaming &&
        (options.lowfreq_scan_mode != "resident" || options.posting_predicate_order != "filter_first" ||
         !options.enable_lowfreq_rerank ||
         options.posting_stream_batch_size == 0))
        fail_pq("PQ posting streaming requires resident filter-first scan and in-memory CPU rerank with positive batch size");
    if (options.lowfreq_scan_mode != "host" && options.lowfreq_scan_mode != "resident")
        fail_pq("PQ lowfreq_scan_mode must be host or resident");
    if (options.posting_predicate_order != "filter_first" && options.posting_predicate_order != "score_first")
        fail_pq("posting_predicate_order must be filter_first or score_first");
    if (options.lowfreq_scan_mode == "resident" || options.posting_predicate_order == "score_first") {
        if (options.query_worker_mode != "coupled" || options.graph_transport != "mailbox" ||
            options.latency_persistent_queue || options.graph_storage_mode != "memory" ||

            defaults::lowfreq_full_filter_mode_is_pre(options.lowfreq_full_filter_mode) ||
            (options.predicate_mode != "gpu_exact" && options.predicate_mode != "gpu_exact_bias") ||
            options.predicate_storage != "csr")
            fail_pq("PQ resident/score-first posting requires native in-memory coupled GPU CSR, without host prefilter or independent warmup");
    }
    if (!defaults::is_valid_lowfreq_full_filter_mode(options.lowfreq_full_filter_mode)) {
        fail_pq("native PQ search lowfreq_full_filter_mode must be one of: pre, post, off");
    }

    if (options.prefetch_hops < 0 || options.prefetch_hops > kKernelPrefetchHops) {
        fail_pq("native PQ search currently supports prefetch-hops in [0, 3]");
    }
    if (options.quick_filter_bonus <= 0.0f || options.quick_filter_bonus > 1.0f) {
        fail_pq("native PQ search currently supports quick-filter-bonus in (0, 1]");
    }
    if (!defaults::is_valid_search_early_stop_mode(options.search_early_stop_mode)) {
        fail_pq("native PQ search search_early_stop_mode must be one of: fixed, off, adaptive");
    }
    parse_pq_buffer_mode(options.buffer_mode);
    validate_pq_query_worker_mode(options);
}

GraphLoadConfig make_pq_load_config(const PqSearchOptions& options,
                                      const char* vector_consumer)
{
    GraphLoadConfig load_config;
    load_config.graph_path = options.graph_path;
    load_config.graph_storage_mode = options.graph_storage_mode;

    load_config.highfreq_prefetch_with_vector_payload =
        defaults::highfreq_rerank_mode_is_gpu(options.highfreq_rerank_mode) &&
        !pq_device_resident_graph_enabled(options);
    load_config.highfreq_prefetch_with_sample_ids =
        pq_graph_use_sample_id_payload(options);
    load_config.data_label_spmat_path = options.data_label_spmat_path;
    load_config.query_label_spmat_path = options.query_label_spmat_path;
    load_config.query_manifest_jsonl_path = options.query_manifest_jsonl_path;
    load_config.query_manifest_families = options.query_manifest_families;
    load_config.query_manifest_subtypes = options.query_manifest_subtypes;
    load_config.query_manifest_max_or_terms = options.query_manifest_max_or_terms;
    load_config.query_manifest_max_fuzzy_terms = options.query_manifest_max_fuzzy_terms;
    load_config.query_manifest_min_ngram_score = options.query_manifest_min_ngram_score;
    load_config.query_manifest_ngram_size = options.query_manifest_ngram_size;
    load_config.base_vector_path = options.base_vector_path;
    load_config.base_range_fbin_path = options.base_range_fbin_path;
    load_config.query_vector_path = options.query_vector_path;
    load_config.query_range_path = options.query_range_path;
    load_config.range_label_manifest_path = options.range_label_manifest_path;
    load_config.vector_consumer = vector_consumer;
    load_config.supports_shared_label_routes = !pq_device_resident_graph_enabled(options);
    load_config.seed_sidecar_path = options.seed_sidecar_path;
    load_config.original_label_seed_sidecar_path = options.original_label_seed_sidecar_path;
    load_config.metadata_sidecar_path = options.metadata_sidecar_path;
    load_config.seed_selection_mode = options.seed_selection_mode;
    load_config.lowfreq_index_path = options.lowfreq_index_path;
    load_config.group_path = options.group_path;
    load_config.routing_mode = options.routing_mode;
    load_config.and_dual_route_posting_ratio_threshold =
        options.and_dual_route_posting_ratio_threshold;
    load_config.dataset_cutoff = options.dataset_cutoff;
    load_config.specificity_threshold = options.specificity_threshold;
    load_config.prefetch_hops = options.prefetch_hops;
    load_config.dim = options.dim;
    load_config.degree = options.degree;
    load_config.load_data_labels_for_filtering =
        pq_device_resident_graph_enabled(options) ||
        options.full_filter ||
        options.per_step_full_filter ||
        options.lowfreq_full_filter_mode != defaults::lowfreq_full_filter_mode_off ||
        defaults::highfreq_rerank_mode_is_cpu(options.highfreq_rerank_mode);
    return load_config;
}

uint32_t route_original_query_id(const GraphRuntime& runtime, uint32_t route_idx)
{
    if (route_idx < runtime.route_to_original_query.size()) {
        return runtime.route_to_original_query[route_idx];
    }
    return route_idx;
}

bool pq_latency_route_is_lowfreq(const GraphRuntime& runtime, uint32_t route_idx)
{
    return route_idx < runtime.query_is_lowfreq.size() &&
           runtime.query_is_lowfreq[route_idx] != 0;
}

bool pq_latency_route_is_routable(const GraphRuntime& runtime, uint32_t route_idx)
{
    if (route_idx >= runtime.query_group_ids.size() ||
        runtime.query_group_ids[route_idx] == UINT32_MAX) {
        return false;
    }
    if (pq_latency_route_is_lowfreq(runtime, route_idx)) {
        return true;
    }
    return route_idx < runtime.start_local_ids.size() &&
           runtime.start_local_ids[route_idx] != UINT32_MAX;
}

PqLatencyRoutePlan build_pq_latency_routes(const GraphRuntime& runtime,
                                           uint32_t query_window_start,
                                           uint32_t query_window_count)
{
    PqLatencyRoutePlan plan;
    const uint32_t query_window_end = query_window_start + query_window_count;
    plan.route_ids.reserve(runtime.query_vectors.rows);
    for (uint32_t route_idx = 0; route_idx < runtime.query_vectors.rows; ++route_idx) {
        const uint32_t original_query = route_original_query_id(runtime, route_idx);
        if (original_query < query_window_start || original_query >= query_window_end) {
            continue;
        }
        if (!pq_latency_route_is_routable(runtime, route_idx)) {
            ++plan.unroutable_routes;
            continue;
        }
        plan.route_ids.push_back(route_idx);
        if (pq_latency_route_is_lowfreq(runtime, route_idx)) {
            ++plan.selected_low_routes;
        } else {
            ++plan.selected_high_routes;
        }
    }
    return plan;
}

std::vector<uint32_t> build_pq_lowfreq_candidate_ids(const PqSearchOptions& options,
                                                     const GraphRuntime& runtime,
                                                     uint32_t query_idx)
{
    if (query_idx >= runtime.query_vectors.rows ||
        query_idx >= runtime.query_group_ids.size() ||
        query_idx >= runtime.query_filter_bits.size() ||
        query_idx >= runtime.query_filter_labels.size()) {
        fail_pq("PQ latency lowfreq query id out of range");
    }
    const uint32_t label = runtime.query_group_ids[query_idx];
    if (label == UINT32_MAX ||
        label >= runtime.lowfreq_index.label_offset.size() ||
        label >= runtime.lowfreq_index.label_size.size()) {
        return {};
    }

    const bool pre_filter_full_labels =
        defaults::lowfreq_full_filter_mode_is_pre(options.lowfreq_full_filter_mode);
    const uint64_t query_filter_bit = runtime.query_filter_bits[query_idx];
    const uint32_t begin = runtime.lowfreq_index.label_offset[label];
    const uint32_t count = runtime.lowfreq_index.label_size[label];
    std::vector<uint32_t> candidate_ids;
    candidate_ids.reserve(count);

    for (uint32_t j = 0; j < count; ++j) {
        const uint32_t row = begin + j;
        if (row >= runtime.lowfreq_index.sample_ids.size() ||
            row >= runtime.lowfreq_index.row_filter_mask.size()) {
            continue;
        }
        if ((runtime.lowfreq_index.row_filter_mask[row] & query_filter_bit) == 0ULL) {
            continue;
        }
        const uint32_t sample_id = runtime.lowfreq_index.sample_ids[row];
        if (sample_id >= runtime.base_vectors.rows) {
            continue;
        }
        if (!runtime.base_range_values.empty() &&
            go_fann::range_filter::query_has_active_range(runtime.query_ranges.modes.data(),
                                                                query_idx,
                                                                runtime.query_ranges.range_count) &&
            !go_fann::range_filter::sample_matches_all_active_ranges_host(runtime.query_ranges,
                                                                               query_idx,
                                                                               runtime.base_range_values,
                                                                               runtime.base_range_count,
                                                                               sample_id)) {
            continue;
        }
        if (pre_filter_full_labels &&
            !sample_matches_route_predicate(runtime, query_idx, sample_id)) {
            continue;
        }
        candidate_ids.push_back(sample_id);
    }
    return candidate_ids;
}

size_t max_pq_latency_lowfreq_candidate_count(const PqSearchOptions& options,
                                              const GraphRuntime& runtime,
                                              const std::vector<uint32_t>& route_ids)
{
    size_t max_count = 1;
    for (uint32_t query_idx : route_ids) {
        if (!pq_latency_route_is_lowfreq(runtime, query_idx)) {
            continue;
        }
        max_count = std::max(max_count,
                             build_pq_lowfreq_candidate_ids(options, runtime, query_idx).size());
    }
    return max_count;
}

void rerank_pq_latency_lowfreq_shortlist(const PqSearchOptions& options,
                                         const GraphRuntime& runtime,
                                         const PqPersistentLatencyQueue& queue,
                                         PqPersistentLatencySlot& slot)
{
    const uint32_t query_idx = slot.query_id;
    const uint32_t shortlist_k = queue.lowfreq_shortlist_k;
    std::vector<idx_t> shortlist(shortlist_k, UINT32_MAX);
    std::vector<float> shortlist_distances(shortlist_k, FLT_MAX);
    const size_t row_base = static_cast<size_t>(slot.slot_index) * shortlist_k;
    for (uint32_t j = 0; j < shortlist_k; ++j) {
        shortlist[j] = queue.h_lowfreq_result_mapped[row_base + j];
        shortlist_distances[j] = queue.h_lowfreq_distances_mapped[row_base + j];
    }
    if (!options.enable_lowfreq_rerank) {
        slot.result_row.assign(static_cast<size_t>(options.topk), UINT32_MAX);
        slot.distance_row.assign(static_cast<size_t>(options.topk), FLT_MAX);
        for (uint32_t j = 0; j < static_cast<uint32_t>(options.topk) && j < shortlist_k; ++j) {
            slot.result_row[j] = shortlist[j];
            slot.distance_row[j] = shortlist_distances[j];
        }
        slot.step_num = 1;
        return;
    }
    std::vector<uint32_t> query_ids{query_idx};
    std::vector<std::vector<idx_t>> results;
    std::vector<int> step_nums;
    std::vector<std::vector<float>> distances;
    lowfreq_host_heuristic_rerank(runtime,
                                  runtime.query_vectors,
                                  query_ids,
                                  shortlist,
                                  shortlist_k,
                                  static_cast<uint32_t>(options.topk),
                                  static_cast<uint32_t>(options.lowfreq_rerank_batch_size),
                                  options.lowfreq_rerank_epsilon,
                                  static_cast<uint32_t>(options.lowfreq_rerank_stable_batches),
                                  defaults::lowfreq_full_filter_mode_is_post(options.lowfreq_full_filter_mode),
                                  options.highfreq_rerank_threads,
                                  results,
                                  step_nums,
                                  &distances);
    if (!results.empty()) {
        slot.result_row = std::move(results.front());
    }
    if (!distances.empty()) {
        slot.distance_row = std::move(distances.front());
    }
    if (!step_nums.empty()) {
        slot.step_num = step_nums.front();
    }
}

void collect_pq_latency_highfreq_gpu_results(const PqSearchOptions& options,
                                             PqGraphSearchContext& graph_ctx,
                                             PqLatencyResultSidecars& result_sidecars)
{
    const size_t query_count = graph_ctx.total_query_capacity;
    const size_t value_count = query_count * graph_ctx.result_width;
    std::vector<idx_t> host_result(value_count, UINT32_MAX);
    std::vector<float> host_result_dist(value_count, FLT_MAX);
    std::vector<float> host_time(query_count * 6, 0.0f);

    copy_result_buffer_to_host(graph_ctx,
                               value_count,
                               host_result,
                               "cudaMemcpy PQ latency result sidecar ids");
    copy_result_dist_buffer_to_host(graph_ctx,
                                    value_count,
                                    host_result_dist,
                                    "cudaMemcpy PQ latency result sidecar distances");
    if (graph_ctx.time_buffer_is_host_mapped && graph_ctx.h_time_per_block_mapped != nullptr) {
        std::copy(graph_ctx.h_time_per_block_mapped,
                  graph_ctx.h_time_per_block_mapped + query_count * 6,
                  host_time.begin());
    } else {
        check_cuda(cudaMemcpy(host_time.data(),
                              graph_ctx.d_time_per_block,
                              sizeof(float) * host_time.size(),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy PQ latency result sidecar steps");
    }

    for (size_t route_idx = 0; route_idx < result_sidecars.route_results.size(); ++route_idx) {
        if (route_idx >= result_sidecars.route_measured.size() ||
            result_sidecars.route_measured[route_idx] == 0) {
            continue;
        }
        if (route_idx < result_sidecars.route_result_ready.size() &&
            result_sidecars.route_result_ready[route_idx] != 0) {
            continue;
        }
        std::vector<idx_t>& row = result_sidecars.route_results[route_idx];
        row.assign(static_cast<size_t>(options.topk), UINT32_MAX);
        std::vector<float>& dist_row = result_sidecars.route_distances[route_idx];
        dist_row.assign(static_cast<size_t>(options.topk), FLT_MAX);
        const size_t src_base = route_idx * graph_ctx.result_width;
        for (uint32_t j = 0;
             j < graph_ctx.result_width && j < static_cast<uint32_t>(options.topk);
             ++j) {
            row[j] = host_result[src_base + j];
            dist_row[j] = host_result_dist[src_base + j];
        }
        result_sidecars.route_steps[route_idx] = static_cast<int>(host_time[6 * route_idx + 4]);
        if (route_idx < result_sidecars.route_result_ready.size()) {
            result_sidecars.route_result_ready[route_idx] = 1;
        }
    }
}

void write_pq_latency_result_sidecars(const PqSearchOptions& options,
                                      int concurrency,
                                      uint32_t query_window_start,
                                      uint32_t query_window_count,
                                      const GraphRuntime& runtime,
                                      const PqLatencyResultSidecars& result_sidecars)
{
    std::vector<std::vector<idx_t>> merged_results;
    std::vector<int> merged_steps;
    merge_route_results_to_original(runtime,
                                    result_sidecars.route_results,
                                    result_sidecars.route_distances,
                                    result_sidecars.route_steps,
                                    static_cast<uint32_t>(options.topk),
                                    merged_results,
                                    merged_steps);
    const uint32_t query_window_end = std::min<uint32_t>(
        static_cast<uint32_t>(merged_results.size()),
        query_window_start + query_window_count);
    std::vector<std::vector<idx_t>> window_results;
    std::vector<int> window_steps;
    window_results.reserve(query_window_end - query_window_start);
    window_steps.reserve(query_window_end - query_window_start);
    for (uint32_t query_idx = query_window_start; query_idx < query_window_end; ++query_idx) {
        window_results.push_back(std::move(merged_results[query_idx]));
        window_steps.push_back(query_idx < merged_steps.size() ? merged_steps[query_idx] : 0);
    }

    std::vector<int> route_flags = build_graph_original_route_flags(runtime);
    std::vector<int> window_route_flags;
    window_route_flags.reserve(query_window_end - query_window_start);
    for (uint32_t query_idx = query_window_start; query_idx < query_window_end; ++query_idx) {
        window_route_flags.push_back(query_idx < route_flags.size() ? route_flags[query_idx] : 0);
    }

    save_results_to_csv(window_results,
                        make_latency_results_path(options.latency_output_path, concurrency));
    save_step_nums_to_csv(window_steps,
                          make_latency_steps_path(options.latency_output_path, concurrency));
    save_route_flags_to_csv(window_route_flags,
                            make_latency_routes_path(options.latency_output_path, concurrency));
}

void write_pq_latency_header(std::ofstream& csv)
{
    csv << "backend,scheduler,concurrency,warmup_runs,timed_runs,run_all_queries,"
        << "query_window_count,selected_high_routes,selected_low_routes,unroutable_routes,"
        << "measured_requests,closed_loop_ms,qps,effective_inflight,"
        << "total_avg_ms,total_min_ms,total_p50_ms,total_p90_ms,total_p95_ms,total_p99_ms,total_max_ms,"
        << "gpu_avg_ms,gpu_min_ms,gpu_p50_ms,gpu_p90_ms,gpu_p95_ms,gpu_p99_ms,gpu_max_ms,"
        << "host_requests_served,host_requests_passed,host_requests_filtered,workers_thread\n";
}

void write_pq_latency_summary_row(std::ofstream& csv,
                                  int concurrency,
                                  const PqSearchOptions& options,
                                  uint32_t query_window_count,
                                  uint32_t selected_high_routes,
                                  uint32_t selected_low_routes,
                                  uint32_t unroutable_routes,
                                  double closed_loop_ms,
                                  const std::vector<PqPersistentLatencyMeasurement>& measurements,
                                  const HostFilterStats& host_stats)
{
    std::vector<double> total_values;
    std::vector<double> gpu_values;
    total_values.reserve(measurements.size());
    gpu_values.reserve(measurements.size());
    double latency_sum = 0.0;
    for (const auto& measurement : measurements) {
        total_values.push_back(measurement.total_ms);
        gpu_values.push_back(measurement.gpu_ms);
        latency_sum += measurement.total_ms;
    }
    const LatencySummary total = summarize_latency_values(std::move(total_values));
    const LatencySummary gpu = summarize_latency_values(std::move(gpu_values));
    const double qps = closed_loop_ms > 0.0
        ? static_cast<double>(measurements.size()) / (closed_loop_ms / 1000.0)
        : 0.0;
    const double effective_inflight = closed_loop_ms > 0.0 ? latency_sum / closed_loop_ms : 0.0;
    csv << "pq,persistent_queue,"
        << concurrency << ','
        << options.warmup_runs << ','
        << options.timed_runs << ','
        << (options.latency_run_all_queries ? 1 : 0) << ','
        << query_window_count << ','
        << selected_high_routes << ','
        << selected_low_routes << ','
        << unroutable_routes << ','
        << measurements.size() << ','
        << closed_loop_ms << ','
        << qps << ','
        << effective_inflight << ','
        << total.avg << ',' << total.min << ',' << total.p50 << ',' << total.p90 << ','
        << total.p95 << ',' << total.p99 << ',' << total.max << ','
        << gpu.avg << ',' << gpu.min << ',' << gpu.p50 << ',' << gpu.p90 << ','
        << gpu.p95 << ',' << gpu.p99 << ',' << gpu.max << ','
        << host_stats.requests_served << ','
        << host_stats.requests_passed << ','
        << host_stats.requests_filtered << ','
        << host_stats.workers_thread << '\n';
}

void write_pq_latency_trace_header(std::ofstream& csv)
{
    csv << "backend,scheduler,concurrency,request_ordinal,query_id,route_type,total_ms,gpu_ms\n";
}

void append_pq_latency_trace_rows(std::ofstream& csv,
                                  int concurrency,
                                  const std::vector<PqPersistentLatencyMeasurement>& measurements)
{
    for (size_t idx = 0; idx < measurements.size(); ++idx) {
        const auto& measurement = measurements[idx];
        csv << "pq,persistent_queue,"
            << concurrency << ','
            << idx << ','
            << measurement.query_id << ','
            << (measurement.lowfreq_route ? "low" : "high") << ','
            << measurement.total_ms << ','
            << measurement.gpu_ms << '\n';
    }
}

int run_pq_latency_benchmark(const PqSearchOptions& options)
{
    validate_pq_search_options(options);
    if (options.posting_streaming) fail_pq("PQ posting streaming is limited to native search and benchmark modes");
    if (defaults::pq_query_worker_mode_is_finite(options.query_worker_mode)) {
        fail_pq("query_worker_mode=finite is limited to native search and benchmark modes");
    }
    if (!options.latency_staggered || !options.latency_persistent_queue) {
        fail_pq("latency mode for PQ requires --latency-staggered --latency-persistent-queue");
    }
    if (options.latency_online_metadata) {
        fail_pq("PQ persistent latency uses resident query metadata; do not enable latency_online_metadata");
    }
    PqSearchOptions latency_options = options;
    const bool latency_cpu_rerank = highfreq_cpu_rerank_enabled(latency_options);
    if (latency_cpu_rerank) {
        latency_options.highfreq_rerank_streaming = true;
    }
    if (latency_options.latency_lowfreq_mode != "candidate") {
        fail_pq("PQ persistent latency lowfreq restore currently requires latency_lowfreq_mode=candidate");
    }
    if (options.latency_query_start < 0 || options.latency_query_limit < 0) {
        fail_pq("latency query start/limit must be non-negative");
    }
    std::vector<int> concurrency_levels = options.latency_concurrency_levels.empty()
        ? options.latency_batch_sizes
        : options.latency_concurrency_levels;
    if (concurrency_levels.empty()) {
        fail_pq("latency concurrency levels must not be empty");
    }
    for (int concurrency : concurrency_levels) {
        if (concurrency < 0) {
            fail_pq("latency concurrency levels must be >= 0");
        }
    }
    require_pq_graph_dim_supported(options.dim);
    require_pq_graph_pq_dim_supported(options.pq_dim);

    GraphRuntime runtime = load_graph_runtime(make_pq_load_config(latency_options, "native PQ latency"));
    require_uint8_graph_runtime(runtime, "native PQ latency");
    if (runtime.base_vectors.rows == 0 || runtime.query_vectors.rows == 0) {
        fail_pq("loaded empty dataset");
    }
    if (runtime.original_query_count == 0) {
        fail_pq("PQ persistent latency query set is empty");
    }
    if (options.latency_query_start >= static_cast<int>(runtime.original_query_count)) {
        fail_pq("latency_query_start is beyond original query count");
    }
    const uint32_t query_window_start = static_cast<uint32_t>(options.latency_query_start);
    const uint32_t remaining_queries = runtime.original_query_count - query_window_start;
    const uint32_t query_window_count = options.latency_query_limit == 0
        ? remaining_queries
        : std::min(static_cast<uint32_t>(options.latency_query_limit), remaining_queries);
    const PqLatencyRoutePlan route_plan =
        build_pq_latency_routes(runtime, query_window_start, query_window_count);
    if (route_plan.route_ids.empty()) {
        fail_pq("PQ persistent latency query window contains no routable routes");
    }

    PqHostIndex pq_index = load_pq_index(options.files, runtime.base_vectors.rows, options.dim, options.pq_dim);
    PqImmutableDeviceState immutable_state;
    PqPreparedQueryState prepared_state;
    PqGraphSearchContext graph_ctx;
    try {
        init_immutable_state(immutable_state, pq_index, latency_options);
        init_prepared_query_state(prepared_state, runtime, latency_options);
        init_graph_context_prepared(graph_ctx,
                                    runtime.query_vectors.rows,
                                    immutable_state,
                                    prepared_state,
                                    runtime,
                                    latency_options);

        std::ofstream csv(options.latency_output_path);
        if (!csv.is_open()) {
            fail_pq("could not open latency output file " + options.latency_output_path);
        }
        write_pq_latency_header(csv);
        std::ofstream trace_csv;
        if (!options.latency_trace_output_path.empty()) {
            trace_csv.open(options.latency_trace_output_path);
            if (!trace_csv.is_open()) {
                fail_pq("could not open latency trace output file " +
                        options.latency_trace_output_path);
            }
            write_pq_latency_trace_header(trace_csv);
        }

        uint32_t resident_worker_capacity = 0;
        dispatch_graph_kernel(options.search_queue_size, options.pq_dim, [&]<int QS, int PD>() {
            resident_worker_capacity = resident_worker_capacity_for_latency_kernel(
                persistent_pq_latency_kernel<QS, PD>,
                kThreadInBlock,
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor PQ persistent latency",
                "cudaGetDeviceProperties PQ persistent latency");
        });

        for (int concurrency : concurrency_levels) {
            const int resolved_concurrency =
                concurrency == 0 ? static_cast<int>(resident_worker_capacity) : concurrency;
            if (resolved_concurrency <= 0) {
                fail_pq("latency concurrency level resolved to zero");
            }
            const size_t requested_count = options.latency_run_all_queries
                ? route_plan.route_ids.size()
                : std::min(route_plan.route_ids.size(),
                           static_cast<size_t>(std::max(1, options.timed_runs)) *
                               static_cast<size_t>(resolved_concurrency));
            if (requested_count == 0) {
                continue;
            }
            std::vector<uint32_t> measured_ids(route_plan.route_ids.begin(),
                                               route_plan.route_ids.begin() + requested_count);
            const int total_point_runs = options.warmup_runs + 1;
            for (int point_run = 0; point_run < total_point_runs; ++point_run) {
                const bool timed_point_run = point_run == options.warmup_runs;
                PqLatencyResultSidecars result_sidecars;
                result_sidecars.route_results.assign(
                    runtime.query_vectors.rows,
                    std::vector<idx_t>(
                        static_cast<size_t>(latency_options.topk), UINT32_MAX));
                result_sidecars.route_distances.assign(
                    runtime.query_vectors.rows,
                    std::vector<float>(
                        static_cast<size_t>(latency_options.topk), FLT_MAX));
                result_sidecars.route_steps.assign(runtime.query_vectors.rows,
                                                   0);
                result_sidecars.route_measured.assign(
                    runtime.query_vectors.rows, 0);
                result_sidecars.route_result_ready.assign(
                    runtime.query_vectors.rows, 0);
                for (uint32_t route_idx : measured_ids) {
                    if (route_idx < result_sidecars.route_measured.size()) {
                        result_sidecars.route_measured[route_idx] = 1;
                    }
                }
                std::vector<uint32_t> high_route_ids;
                high_route_ids.reserve(measured_ids.size());
                for (uint32_t route_idx : measured_ids) {
                    if (!pq_latency_route_is_lowfreq(runtime, route_idx)) {
                        high_route_ids.push_back(route_idx);
                    }
                }
                reset_search_state(graph_ctx, graph_ctx.total_query_capacity);
                const size_t lowfreq_candidate_capacity =
                    max_pq_latency_lowfreq_candidate_count(
                        latency_options, runtime, measured_ids);
                PqPersistentLatencyQueue queue =
                    create_persistent_latency_queue(
                        static_cast<uint32_t>(resolved_concurrency),
                        latency_options, lowfreq_candidate_capacity);
                std::vector<PqPersistentLatencySlot> slots =
                    create_persistent_latency_slots(
                        static_cast<uint32_t>(resolved_concurrency));
                std::vector<int> req_per_loop(20000, 0);
                const GraphNodeView host_view =
                    make_graph_node_view(runtime);
                auto start_host_service = [&]() {
                    if (high_route_ids.empty()) {
                        return std::thread();
                    }
                    return std::thread(
                                     host_scan_and_set, graph_ctx.status_memory,
                                     graph_ctx.h_neighbor_memory,
                                     graph_ctx.h_gdr_data, req_per_loop.data(),
                                     graph_ctx.status_flag, host_view,
                                     graph_ctx.total_query_capacity,
                                     high_route_ids.data(),
                                     high_route_ids.size(), graph_ctx.io_config,
                                     &graph_ctx.host_filter_stats);
                };
                std::thread scan_thread;

                std::vector<PqPersistentLatencyMeasurement> measurements;
                measurements.reserve(requested_count);
                std::mutex lowfreq_mutex;
                std::condition_variable lowfreq_cv;
                std::deque<size_t> lowfreq_queue;
                bool lowfreq_stop = false;
                std::exception_ptr lowfreq_error;
                std::vector<std::thread> lowfreq_workers;
                auto mark_lowfreq_done = [&](PqPersistentLatencySlot &slot) {
                    std::lock_guard<std::mutex> lock(lowfreq_mutex);
                    slot.rerank_done_time = std::chrono::steady_clock::now();
                    slot.rerank_done = true;
                };
                auto start_lowfreq_workers = [&]() {
                    const size_t worker_count = static_cast<size_t>(
                        std::max(1, latency_options.highfreq_rerank_threads));
                    lowfreq_workers.reserve(worker_count);
                    for (size_t worker_idx = 0; worker_idx < worker_count;
                         ++worker_idx) {
                        lowfreq_workers.emplace_back([&]() {

                            while (true) {
                                size_t slot_index = 0;
                                {
                                    std::unique_lock<std::mutex> lock(
                                        lowfreq_mutex);
                                    lowfreq_cv.wait(lock, [&]() {
                                        return lowfreq_stop ||
                                               !lowfreq_queue.empty();
                                    });
                                    if (lowfreq_stop && lowfreq_queue.empty()) {
                                        return;
                                    }
                                    slot_index = lowfreq_queue.front();
                                    lowfreq_queue.pop_front();
                                }
                                try {
                                    PqPersistentLatencySlot &slot =
                                        slots[slot_index];
                                    rerank_pq_latency_lowfreq_shortlist(
                                        latency_options, runtime, queue, slot);
                                    mark_lowfreq_done(slot);
                                    lowfreq_cv.notify_one();
                                } catch (...) {
                                    {
                                        std::lock_guard<std::mutex> lock(
                                            lowfreq_mutex);
                                        if (lowfreq_error == nullptr) {
                                            lowfreq_error =
                                                std::current_exception();
                                        }
                                        lowfreq_stop = true;
                                    }
                                    lowfreq_cv.notify_all();
                                    return;
                                }
                            }
                        });
                    }
                };
                auto stop_lowfreq_workers = [&]() {
                    {
                        std::lock_guard<std::mutex> lock(lowfreq_mutex);
                        lowfreq_stop = true;
                    }
                    lowfreq_cv.notify_all();
                    for (std::thread &worker : lowfreq_workers) {
                        if (worker.joinable()) {
                            worker.join();
                        }
                    }
                    lowfreq_workers.clear();
                };
                auto enqueue_lowfreq = [&](size_t slot_index) {
                    {
                        std::lock_guard<std::mutex> lock(lowfreq_mutex);
                        lowfreq_queue.push_back(slot_index);
                    }
                    lowfreq_cv.notify_one();
                };
                auto rethrow_lowfreq_error_if_any = [&]() {
                    std::exception_ptr error;
                    {
                        std::lock_guard<std::mutex> lock(lowfreq_mutex);
                        error = lowfreq_error;
                    }
                    if (error != nullptr) {
                        std::rethrow_exception(error);
                    }
                };
                auto submit_slot = [&](PqPersistentLatencySlot &slot,
                                       uint32_t query_id) {
                    slot.query_id = query_id;
                    slot.lowfreq_route =
                        pq_latency_route_is_lowfreq(runtime, query_id);
                    slot.busy = true;
                    slot.gpu_done = false;
                    slot.rerank_done = false;
                    slot.gpu_ms = 0.0;
                    slot.step_num = 0;
                    slot.result_row.clear();
                    slot.distance_row.clear();
                    slot.rerank_done_time = {};
                    slot.submit_time = std::chrono::steady_clock::now();
                    if (queue.lowfreq_candidate_mode && slot.lowfreq_route) {
                        const size_t slot_candidate_offset =
                            static_cast<size_t>(slot.slot_index) *
                            queue.lowfreq_candidate_capacity;
                        const std::vector<uint32_t> candidates =
                            build_pq_lowfreq_candidate_ids(latency_options,
                                                           runtime, query_id);
                        if (candidates.size() >
                            queue.lowfreq_candidate_capacity) {
                            fail_pq("PQ lowfreq latency candidate list exceeds "
                                    "slot "
                                    "capacity");
                        }
                        std::copy(candidates.begin(), candidates.end(),
                                  queue.h_lowfreq_candidate_ids_mapped +
                                      slot_candidate_offset);
                        queue.h_lowfreq_candidate_offsets_mapped
                            [slot.slot_index] =
                            static_cast<uint32_t>(slot_candidate_offset);
                        queue
                            .h_lowfreq_candidate_sizes_mapped[slot.slot_index] =
                            static_cast<uint32_t>(candidates.size());
                    }
                    ++slot.ready_seq;
                    if (slot.ready_seq == 0 ||
                        slot.ready_seq == kPersistentLatencyStopSeq) {
                        slot.ready_seq = 1;
                    }
                    PqPersistentLatencyCompletionControl &completion =
                        queue.h_completions[slot.slot_index];
                    completion.query_id = UINT32_MAX;
                    completion.done_seq = 0;
                    PqPersistentLatencyRequestControl &request =
                        queue.h_requests[slot.slot_index];
                    request.query_id = query_id;
                    request.lowfreq_route = slot.lowfreq_route ? 1u : 0u;
                    request.stop = 0;
                    _mm_sfence();
                    request.ready_seq = slot.ready_seq;
                    _mm_sfence();
                };

                std::mutex rerank_mutex;
                std::condition_variable rerank_cv;
                std::deque<size_t> rerank_queue;
                bool rerank_stop = false;
                std::exception_ptr rerank_error;
                std::vector<std::thread> rerank_workers;
                auto start_rerank_workers = [&]() {
                    if (!latency_cpu_rerank) {
                        return;
                    }
                    if (!graph_ctx.result_buffer_is_host_mapped ||
                        graph_ctx.h_result_mapped == nullptr ||
                        graph_ctx.h_time_per_block_mapped == nullptr) {
                        fail_pq(
                            "PQ CPU-rerank latency requires mapped streaming "
                            "result buffers");
                    }
                    const size_t worker_count = static_cast<size_t>(
                        std::max(1, latency_options.highfreq_rerank_threads));
                    rerank_workers.reserve(worker_count);
                    for (size_t worker_idx = 0; worker_idx < worker_count;
                         ++worker_idx) {
                        rerank_workers.emplace_back([&]() {

                            while (true) {
                                size_t slot_index = 0;
                                {
                                    std::unique_lock<std::mutex> lock(
                                        rerank_mutex);
                                    rerank_cv.wait(lock, [&]() {
                                        return rerank_stop ||
                                               !rerank_queue.empty();
                                    });
                                    if (rerank_stop && rerank_queue.empty()) {
                                        return;
                                    }
                                    slot_index = rerank_queue.front();
                                    rerank_queue.pop_front();
                                }
                                try {
                                    PqPersistentLatencySlot &slot =
                                        slots[slot_index];
                                    const uint32_t query_id = slot.query_id;
                                    std::vector<uint32_t> query_ids{query_id};
                                    std::vector<std::vector<idx_t>> shortlist(
                                        1, std::vector<idx_t>(
                                               graph_ctx.result_width,
                                               UINT32_MAX));
                                    const size_t row_base =
                                        static_cast<size_t>(query_id) *
                                        graph_ctx.result_width;
                                    for (uint32_t j = 0;
                                         j < graph_ctx.result_width; ++j) {
                                        shortlist[0][j] =
                                            graph_ctx
                                                .h_result_mapped[row_base + j];
                                    }
                                    std::vector<std::vector<idx_t>>
                                        reranked_results;
                                    std::vector<std::vector<float>>
                                        reranked_distances;
                                    highfreq_host_rerank(
                                        latency_options, runtime,
                                        runtime.query_vectors, query_ids,
                                        shortlist, reranked_results,
                                        &reranked_distances);
                                    {
                                        std::lock_guard<std::mutex> lock(
                                            rerank_mutex);
                                        if (!reranked_results.empty()) {
                                            slot.result_row = std::move(
                                                reranked_results.front());
                                        }
                                        if (!reranked_distances.empty()) {
                                            slot.distance_row = std::move(
                                                reranked_distances.front());
                                        }
                                        if (graph_ctx.h_time_per_block_mapped !=
                                            nullptr) {
                                            slot.step_num = static_cast<int>(
                                                graph_ctx
                                                    .h_time_per_block_mapped
                                                        [6 * query_id + 4]);
                                        }
                                        slot.rerank_done_time =
                                            std::chrono::steady_clock::now();
                                        slot.rerank_done = true;
                                    }
                                    rerank_cv.notify_one();
                                } catch (...) {
                                    {
                                        std::lock_guard<std::mutex> lock(
                                            rerank_mutex);
                                        if (rerank_error == nullptr) {
                                            rerank_error =
                                                std::current_exception();
                                        }
                                        rerank_stop = true;
                                    }
                                    rerank_cv.notify_all();
                                    return;
                                }
                            }
                        });
                    }
                };
                auto stop_rerank_workers = [&]() {
                    {
                        std::lock_guard<std::mutex> lock(rerank_mutex);
                        rerank_stop = true;
                    }
                    rerank_cv.notify_all();
                    for (std::thread &worker : rerank_workers) {
                        if (worker.joinable()) {
                            worker.join();
                        }
                    }
                    rerank_workers.clear();
                };
                auto enqueue_rerank = [&](size_t slot_index) {
                    {
                        std::lock_guard<std::mutex> lock(rerank_mutex);
                        rerank_queue.push_back(slot_index);
                    }
                    rerank_cv.notify_one();
                };
                auto rethrow_rerank_error_if_any = [&]() {
                    std::exception_ptr error;
                    {
                        std::lock_guard<std::mutex> lock(rerank_mutex);
                        error = rerank_error;
                    }
                    if (error != nullptr) {
                        std::rethrow_exception(error);
                    }
                };

                try {
                    launch_persistent_latency_worker(latency_options, runtime,
                                                     graph_ctx, queue);
                    scan_thread = start_host_service();
                    start_rerank_workers();
                    start_lowfreq_workers();
                    size_t submitted = 0;
                    size_t completed = 0;
                    for (PqPersistentLatencySlot &slot : slots) {
                        if (submitted >= requested_count) {
                            break;
                        }
                        submit_slot(slot, measured_ids[submitted++]);
                    }
                    const auto closed_loop_start =
                        std::chrono::steady_clock::now();
                    while (completed < submitted) {
                        rethrow_rerank_error_if_any();
                        rethrow_lowfreq_error_if_any();
                        bool made_progress = false;
                        for (PqPersistentLatencySlot &slot : slots) {
                            if (!slot.busy) {
                                continue;
                            }
                            volatile PqPersistentLatencyCompletionControl
                                &completion =
                                    queue.h_completions[slot.slot_index];
                            if (!slot.gpu_done &&
                                completion.done_seq == slot.ready_seq) {
                                std::atomic_thread_fence(
                                    std::memory_order_acquire);
                                slot.gpu_ms =
                                    completion.gpu_stop_clock >=
                                            completion.gpu_start_clock
                                        ? static_cast<double>(
                                              completion.gpu_stop_clock -
                                              completion.gpu_start_clock) /
                                              1000000.0
                                        : 0.0;
                                slot.gpu_done = true;
                                if (slot.lowfreq_route) {
                                    enqueue_lowfreq(slot.slot_index);
                                } else if (latency_cpu_rerank) {
                                    enqueue_rerank(slot.slot_index);
                                } else {
                                    slot.rerank_done_time =
                                        std::chrono::steady_clock::now();
                                    slot.rerank_done = true;
                                }
                            }
                            bool slot_complete = false;
                            std::chrono::steady_clock::time_point done_time;
                            if (slot.lowfreq_route) {
                                std::lock_guard<std::mutex> lock(lowfreq_mutex);
                                if (slot.rerank_done) {
                                    slot_complete = true;
                                    done_time = slot.rerank_done_time;
                                }
                            } else if (latency_cpu_rerank) {
                                std::lock_guard<std::mutex> lock(rerank_mutex);
                                if (slot.rerank_done) {
                                    slot_complete = true;
                                    done_time = slot.rerank_done_time;
                                }
                            } else if (slot.rerank_done) {
                                slot_complete = true;
                                done_time = slot.rerank_done_time;
                            }
                            if (!slot_complete) {
                                continue;
                            }
                            PqPersistentLatencyMeasurement measurement;
                            measurement.query_id = slot.query_id;
                            measurement.lowfreq_route = slot.lowfreq_route;
                            measurement.total_ms =
                                std::chrono::duration<double, std::milli>(
                                    done_time - slot.submit_time)
                                    .count();
                            measurement.gpu_ms = slot.gpu_ms;
                            measurements.push_back(measurement);
                            if (slot.query_id <
                                result_sidecars.route_results.size()) {
                                if (!slot.result_row.empty()) {
                                    result_sidecars
                                        .route_results[slot.query_id] =
                                        slot.result_row;
                                }
                                if (!slot.distance_row.empty()) {
                                    result_sidecars
                                        .route_distances[slot.query_id] =
                                        slot.distance_row;
                                }
                                if (slot.step_num != 0) {
                                    result_sidecars.route_steps[slot.query_id] =
                                        slot.step_num;
                                }
                                if (slot.lowfreq_route || latency_cpu_rerank) {
                                    result_sidecars
                                        .route_result_ready[slot.query_id] = 1;
                                }
                            }
                            slot.busy = false;
                            ++completed;
                            made_progress = true;
                            if (submitted < requested_count) {
                                submit_slot(slot, measured_ids[submitted++]);
                            }
                        }
                        if (!made_progress) {
                            std::this_thread::yield();
                        }
                    }
                    const auto closed_loop_stop =
                        std::chrono::steady_clock::now();
                    const double closed_loop_ms =
                        std::chrono::duration<double, std::milli>(
                            closed_loop_stop - closed_loop_start)
                            .count();
                    stop_rerank_workers();
                    stop_lowfreq_workers();
                    stop_persistent_latency_worker(queue);
                    publish_finished_for_host_service(graph_ctx,
                                                      high_route_ids);
                    if (scan_thread.joinable()) {
                        scan_thread.join();
                    }
                    if (timed_point_run) {
                        collect_pq_latency_highfreq_gpu_results(
                            latency_options, graph_ctx, result_sidecars);
                        write_pq_latency_result_sidecars(
                            latency_options, resolved_concurrency,
                            query_window_start, query_window_count, runtime,
                            result_sidecars);
                    }
                    std::vector<double> total_latency_values;
                    total_latency_values.reserve(measurements.size());
                    for (const auto &measurement : measurements) {
                        total_latency_values.push_back(measurement.total_ms);
                    }
                    const LatencySummary total_latency =
                        summarize_latency_values(
                            std::move(total_latency_values));
                    destroy_persistent_latency_queue(queue);
                    if (timed_point_run) {
                        write_pq_latency_summary_row(
                            csv, resolved_concurrency, latency_options,
                            query_window_count,
                            static_cast<uint32_t>(high_route_ids.size()),
                            static_cast<uint32_t>(requested_count -
                                                  high_route_ids.size()),
                            route_plan.unroutable_routes, closed_loop_ms,
                            measurements, graph_ctx.host_filter_stats);
                        if (trace_csv.is_open()) {
                            append_pq_latency_trace_rows(
                                trace_csv, resolved_concurrency, measurements);
                        }
                        std::printf(
                            "[pq latency persistent] concurrency=%d "
                            "requests=%zu avg=%.4fms p95=%.4fms qps=%.2f\n",
                            resolved_concurrency, measurements.size(),
                            total_latency.avg, total_latency.p95,
                            closed_loop_ms > 0.0
                                ? static_cast<double>(measurements.size()) /
                                      (closed_loop_ms / 1000.0)
                                : 0.0);
                    } else {
                        std::printf(
                            "[pq latency persistent warmup] run=%d/%d "
                            "concurrency=%d requests=%zu elapsed=%.4fms\n",
                            point_run + 1, options.warmup_runs,
                            resolved_concurrency, measurements.size(),
                            closed_loop_ms);
                    }
                } catch (...) {
                    stop_rerank_workers();
                    stop_lowfreq_workers();
                    stop_persistent_latency_worker(queue);
                    publish_finished_for_host_service(graph_ctx,
                                                      high_route_ids);
                    if (scan_thread.joinable()) {
                        scan_thread.join();
                    }
                    destroy_persistent_latency_queue(queue);
                    throw;
                }

            }
        }

        cleanup_graph_context(graph_ctx);
        cleanup_prepared_query_state(prepared_state);
        cleanup_immutable_state(immutable_state);
    } catch (...) {
        cleanup_graph_context(graph_ctx);
        cleanup_prepared_query_state(prepared_state);
        cleanup_immutable_state(immutable_state);
        throw;
    }
    std::printf("PQ persistent latency CSV written to %s\n",
                options.latency_output_path.c_str());
    return 0;
}

int run_pq_search(const PqSearchOptions& options)
{
    validate_pq_search_options(options);

    GraphRuntime runtime = load_graph_runtime(make_pq_load_config(options, "native PQ search"));
    require_uint8_graph_runtime(runtime, "native PQ search");
    if (runtime.base_vectors.rows == 0 || runtime.query_vectors.rows == 0) {
        fail_pq("loaded empty dataset");
    }

    PqHostIndex pq_index = load_pq_index(options.files, runtime.base_vectors.rows, options.dim, options.pq_dim);
    PqImmutableDeviceState immutable_state;
    PqPreparedQueryState prepared_state;
    init_immutable_state(immutable_state, pq_index, options);
    init_prepared_query_state(prepared_state, runtime, options);
    PqLowfreqSearchContext lowfreq_ctx;
    init_lowfreq_context(lowfreq_ctx, immutable_state, prepared_state, runtime, options);
    const bool has_potential_high = std::any_of(runtime.query_group_ids.begin(),
                                                runtime.query_group_ids.end(),
                                                [](uint32_t label) { return label != UINT32_MAX; }) &&
        std::any_of(runtime.query_is_lowfreq.begin(), runtime.query_is_lowfreq.end(), [](uint8_t flag) { return flag == 0; });
    PqGraphSearchContext graph_ctx;
    bool graph_ctx_initialized = false;
    if (has_potential_high) {
        require_pq_graph_dim_supported(options.dim);
        require_pq_graph_pq_dim_supported(options.pq_dim);
        init_graph_context_prepared(graph_ctx,
                                    runtime.query_vectors.rows,
                                    immutable_state,
                                    prepared_state,
                                    runtime,
                                    options);
        graph_ctx_initialized = true;
    }

    PqSearchOptions single_run_base_options = options;
    single_run_base_options.warmup_runs = 0;
    single_run_base_options.timed_runs = 1;
    const size_t highfreq_width = static_cast<size_t>(highfreq_result_width(options));

    try {
        std::vector<std::vector<idx_t>> route_results(runtime.query_vectors.rows,
                                                      std::vector<idx_t>(highfreq_width, UINT32_MAX));
        std::vector<std::vector<float>> route_distances(runtime.query_vectors.rows,
                                                        std::vector<float>(highfreq_width, FLT_MAX));
        std::vector<int> route_steps(runtime.query_vectors.rows, 0);
        std::vector<std::vector<idx_t>> merged_results;
        std::vector<int> merged_steps;
        PqBatchStats highfreq_stats;
        size_t lowfreq_query_count = 0;
        double lowfreq_wall_ms = 0.0;
        double lowfreq_qps_sum = 0.0;
        size_t lowfreq_run_count = 0;
        double route_wall_ms = 0.0;
        double total_wall_ms = 0.0;
        double mixed_qps_sum = 0.0;
        size_t mixed_run_count = 0;
        size_t last_high_original_query_count = 0;
        size_t last_high_query_count = 0;
        size_t last_low_query_count = 0;
        size_t last_unroutable_query_count = 0;

        auto execute_iteration = [&](bool collect_results) {
            const auto iteration_wall_start = std::chrono::steady_clock::now();
            PqSearchOptions single_run_options = single_run_base_options;
            ExactDistanceMemo::Stats memo_stats;
            single_run_options.exact_distance_memo = make_pq_iteration_distance_memo(runtime, single_run_options);
            const bool memo_enabled = single_run_options.exact_distance_memo != nullptr;
            std::vector<uint32_t> high_query_ids;
            std::vector<uint32_t> low_query_ids;
            pq_debug_log("[pq-debug] iteration begin collect=%d", collect_results ? 1 : 0);
            const auto route_start = std::chrono::steady_clock::now();
            build_timed_query_routes(prepared_state, high_query_ids, low_query_ids);
            const auto route_end = std::chrono::steady_clock::now();
            reorder_high_query_ids_by_graph_instance(runtime, high_query_ids);
            sync_query_ids_to_device(high_query_ids,
                                     prepared_state.d_high_query_ids,
                                     "cudaMemcpy reordered high_query_ids");
            const double route_ms = std::chrono::duration<double, std::milli>(route_end - route_start).count();
            pq_debug_log("[pq-debug] route done collect=%d high=%zu low=%zu route_ms=%.3f",
                         collect_results ? 1 : 0,
                         high_query_ids.size(),
                         low_query_ids.size(),
                         route_ms);
            if (collect_results) {
                route_wall_ms += route_ms;
            }
            const size_t total_queries = runtime.query_vectors.rows;
            const size_t unroutable_queries = total_queries - high_query_ids.size() - low_query_ids.size();
            last_high_query_count = high_query_ids.size();
            last_low_query_count = low_query_ids.size();
            last_unroutable_query_count = unroutable_queries;
            std::unordered_set<uint32_t> original_high_queries;
            original_high_queries.reserve(high_query_ids.size());
            for (uint32_t route_idx : high_query_ids) {
                if (route_idx < runtime.route_to_original_query.size()) {
                    original_high_queries.insert(runtime.route_to_original_query[route_idx]);
                }
            }
            last_high_original_query_count = original_high_queries.size();

            if (collect_results) {
                std::fill(merged_results.begin(),
                          merged_results.end(),
                          std::vector<idx_t>(static_cast<size_t>(options.topk), UINT32_MAX));
                std::fill(merged_steps.begin(), merged_steps.end(), 0);
            }

            double highfreq_timed_ms = 0.0;
            std::vector<uint32_t> pending_high_query_ids;
            std::vector<std::vector<idx_t>> pending_high_results;
            std::vector<std::vector<float>> pending_high_dists;
            std::vector<int> pending_high_steps;
            std::future<void> highfreq_rerank_future;
            if (!high_query_ids.empty()) {
                pq_debug_log("[pq-debug] highfreq start collect=%d queries=%zu",
                             collect_results ? 1 : 0,
                             high_query_ids.size());
                std::vector<std::vector<idx_t>> high_results;
                std::vector<std::vector<float>> high_dists;
                std::vector<int> high_steps;
                const bool async_highfreq_cpu_rerank =
                    collect_results &&
                    defaults::highfreq_rerank_mode_is_cpu(single_run_options.highfreq_rerank_mode) &&
                    !highfreq_cpu_streaming_enabled(single_run_options);
                execute_graph_query_ids(graph_ctx,
                                        runtime,
                                        high_query_ids,
                                        prepared_state.d_high_query_ids,
                                        high_results,
                                        high_steps,
                                        &high_dists,
                                        single_run_options,
                                        collect_results ? &highfreq_stats : nullptr,
                                        false,
                                        collect_results ? &highfreq_timed_ms : nullptr,
                                        !async_highfreq_cpu_rerank);
                pq_debug_log("[pq-debug] highfreq done collect=%d queries=%zu timed_ms=%.3f",
                             collect_results ? 1 : 0,
                             high_query_ids.size(),
                             highfreq_timed_ms);
                if (collect_results) {
                    highfreq_stats.display_query_count = last_high_original_query_count * static_cast<size_t>(options.timed_runs);
                }
                if (collect_results) {
                    if (async_highfreq_cpu_rerank) {
                        pending_high_query_ids = high_query_ids;
                        pending_high_results = std::move(high_results);
                        pending_high_dists = std::move(high_dists);
                        pending_high_steps = std::move(high_steps);
                        highfreq_rerank_future = std::async(cpu_rerank_schedule::host_finalize_launch(single_run_options.cpu_rerank_schedule),
                                                            [&runtime,
                                                             &single_run_options,
                                                             &pending_high_query_ids,
                                                             &pending_high_results,
                                                             &pending_high_dists,
                                                         &graph_ctx,
                                                         shortlist_verified = graph_ctx.predicate && graph_ctx.predicate->shortlist_verified] {
                                                                finalize_highfreq_results(single_run_options,
                                                                                          runtime,
                                                                                          pending_high_query_ids,
                                                                                          pending_high_results,
                                                                                          &pending_high_dists, shortlist_verified,
                                                                                      graph_ctx.cpu_rerank_workers.get());
                                                            });
                    } else {
                        merge_subset_results(high_query_ids, route_results, route_steps, high_results, high_steps);
                        for (size_t ii = 0; ii < high_query_ids.size(); ++ii) {
                            route_distances[high_query_ids[ii]] = std::move(high_dists[ii]);
                        }
                    }
                }
            }

            double lowfreq_ms = 0.0;
            if (!low_query_ids.empty()) {
                pq_debug_log("[pq-debug] lowfreq start collect=%d queries=%zu",
                             collect_results ? 1 : 0,
                             low_query_ids.size());
                std::vector<std::vector<idx_t>> low_results;
                std::vector<std::vector<float>> low_dists;
                std::vector<int> low_steps;
                const auto lowfreq_start = std::chrono::steady_clock::now();
                run_lowfreq_pq_search_with_context(single_run_options,
                                                   runtime,
                                                   low_query_ids,
                                                   prepared_state.d_low_query_ids,
                                                   lowfreq_ctx,
                                                   low_results,
                                                   low_steps,
                                                   &low_dists);
                const auto lowfreq_end = std::chrono::steady_clock::now();
                pq_debug_log("[pq-debug] lowfreq done collect=%d queries=%zu",
                             collect_results ? 1 : 0,
                             low_query_ids.size());
                if (collect_results) {
                    merge_subset_results(low_query_ids, route_results, route_steps, low_results, low_steps);
                    for (size_t ii = 0; ii < low_query_ids.size(); ++ii) {
                        route_distances[low_query_ids[ii]] = std::move(low_dists[ii]);
                    }
                }
                lowfreq_ms = std::chrono::duration<double, std::milli>(lowfreq_end - lowfreq_start).count();
                if (collect_results) {
                    lowfreq_query_count += low_query_ids.size();
                    lowfreq_wall_ms += lowfreq_ms;
                    lowfreq_qps_sum += lowfreq_ms > 0.0 ? (static_cast<double>(low_query_ids.size()) / (lowfreq_ms / 1000.0)) : 0.0;
                    lowfreq_run_count += 1;
                }
            }

            if (collect_results) {
                if (highfreq_rerank_future.valid()) {
                    highfreq_rerank_future.get();
                    merge_subset_results(pending_high_query_ids,
                                         route_results,
                                         route_steps,
                                         pending_high_results,
                                         pending_high_steps);
                    for (size_t ii = 0; ii < pending_high_query_ids.size(); ++ii) {
                        route_distances[pending_high_query_ids[ii]] = std::move(pending_high_dists[ii]);
                    }
                }
                const size_t processed_queries = high_query_ids.size() + low_query_ids.size();
                const size_t original_query_count = runtime.original_query_count;
                memo_stats = finish_pq_distance_memo(single_run_options);
                const auto iteration_wall_end = std::chrono::steady_clock::now();
                const double run_wall_ms =
                    std::chrono::duration<double, std::milli>(iteration_wall_end - iteration_wall_start).count();
                pq_debug_log("[pq-debug] iteration end collect=%d processed=%zu run_wall_ms=%.3f",
                             collect_results ? 1 : 0,
                             processed_queries,
                             run_wall_ms);
                total_wall_ms += run_wall_ms;
                if (run_wall_ms > 0.0 && original_query_count > 0) {
                    mixed_qps_sum += static_cast<double>(original_query_count) / (run_wall_ms / 1000.0);
                    mixed_run_count += 1;
                }
            }
            if (!collect_results) memo_stats = finish_pq_distance_memo(single_run_options);
            report_pq_distance_memo(memo_stats, collect_results, single_run_options.rerank_distance_reuse, memo_enabled);
        };

        for (int run = 0; run < options.warmup_runs; ++run) {
            execute_iteration(false);
        }
        for (int run = 0; run < options.timed_runs; ++run) {
            execute_iteration(true);
        }

        std::printf("PQ mixed query split per run: total(original)=%u highfreq(route)=%zu lowfreq(route)=%zu unroutable(route)=%zu\n",
                    runtime.original_query_count,
                    last_high_query_count,
                    last_low_query_count,
                    last_unroutable_query_count);
        std::printf("PQ virtual queries per run: %u\n", runtime.query_vectors.rows);
        std::printf("PQ original queries per run: %u\n", runtime.original_query_count);
        if (runtime.original_query_count > 0) {
            std::printf("PQ virtual/original ratio: %f\n",
                        static_cast<double>(runtime.query_vectors.rows) / static_cast<double>(runtime.original_query_count));
        }
        std::printf("PQ timed runs: %d\n", options.timed_runs);
        std::printf("PQ route wall time total: %f ms\n", route_wall_ms);
        if (options.timed_runs > 0) {
            std::printf("PQ route wall time per run: %f ms\n", route_wall_ms / static_cast<double>(options.timed_runs));
        }
        std::printf("PQ highfreq queries per run: %zu\n", last_high_query_count);
        if (last_high_original_query_count > 0 && last_high_original_query_count != last_high_query_count) {
            std::printf("PQ highfreq original queries per run: %zu\n", last_high_original_query_count);
        }
        if (highfreq_stats.query_count > 0) {
            print_pq_batch_stats_summary("PQ highfreq", highfreq_stats);
        }
        if (lowfreq_query_count > 0) {
            std::printf("PQ lowfreq queries per run: %zu\n", last_low_query_count);
            std::printf("PQ lowfreq query evaluations: %zu\n", lowfreq_query_count);
            std::printf("PQ lowfreq wall time: %f ms\n", lowfreq_wall_ms);
            if (lowfreq_run_count > 0) {
                std::printf("PQ lowfreq effective QPS: %f\n", lowfreq_qps_sum / static_cast<double>(lowfreq_run_count));
            }
        }
        const size_t processed_query_count = highfreq_stats.query_count + lowfreq_query_count;
        std::printf("PQ route query evaluations: %zu\n", processed_query_count);
        std::printf("PQ mixed wall time: %f ms\n", total_wall_ms);
        if (mixed_run_count > 0) {
            const double original_qps = runtime.original_query_count > 0
                ? (static_cast<double>(runtime.original_query_count) / static_cast<double>(runtime.query_vectors.rows)) *
                      (mixed_qps_sum / static_cast<double>(mixed_run_count))
                : 0.0;
            std::printf("PQ mixed effective QPS: %f\n", original_qps);
        }

        merge_route_results_to_original(runtime,
                                        route_results,
                                        route_distances,
                                        route_steps,
                                        static_cast<uint32_t>(options.topk),
                                        merged_results,
                                        merged_steps);

        save_results_to_csv(merged_results, options.output_csv_path);
        cleanup_lowfreq_context(lowfreq_ctx);
        if (graph_ctx_initialized) {
            cleanup_graph_context(graph_ctx);
        }
        cleanup_prepared_query_state(prepared_state);
        cleanup_immutable_state(immutable_state);
        return 0;
    } catch (...) {
        cleanup_lowfreq_context(lowfreq_ctx);
        if (graph_ctx_initialized) {
            cleanup_graph_context(graph_ctx);
        }
        cleanup_prepared_query_state(prepared_state);
        cleanup_immutable_state(immutable_state);
        throw;
    }
}

int run_benchmark_sweep(const PqSearchOptions& base_options,
                        const std::vector<float>& quick_filter_bonuses,
                        const std::vector<int>& queue_sizes,
                        const std::string& output_csv_path)
{
    if (base_options.raw_navigation_scores && quick_filter_bonuses.size() > 1) {
        fail_pq("legacy bonus sweep is inactive for raw navigation scores; use one value");
    }

    const auto benchmark_setup_start = std::chrono::steady_clock::now();
    const std::string setup_path = make_sidecar_csv_path(output_csv_path, ".setup.jsonl");
    QuerySetupReporter setup_report(setup_path, "pq");
    auto stage_start = std::chrono::steady_clock::now();
    nlohmann::json setup_stages = nlohmann::json::object();
    validate_pq_search_options(base_options);

    GraphLoadConfig load_config = make_pq_load_config(base_options, "native PQ benchmark");

    GraphRuntime runtime = load_graph_runtime(load_config);
    require_uint8_graph_runtime(runtime, "native PQ benchmark");
    if (runtime.base_vectors.rows == 0 || runtime.query_vectors.rows == 0) {
        fail_pq("loaded empty dataset");
    }

    setup_stages["primary_runtime_load_and_options"] =
        setup_elapsed_ms(stage_start, std::chrono::steady_clock::now());
    stage_start = std::chrono::steady_clock::now();
    PqHostIndex pq_index = load_pq_index(base_options.files, runtime.base_vectors.rows, base_options.dim, base_options.pq_dim);
    setup_stages["artifact_load"] =
        setup_elapsed_ms(stage_start, std::chrono::steady_clock::now());
    stage_start = std::chrono::steady_clock::now();
    PqImmutableDeviceState immutable;
    PqPreparedQueryState prepared;
    PqGraphSearchContext graph_ctx;
    PqLowfreqSearchContext lowfreq_ctx;
    init_benchmark_contexts(immutable, prepared, graph_ctx, lowfreq_ctx, runtime, pq_index, base_options);
    setup_stages["gpu_contexts_init"] =
        setup_elapsed_ms(stage_start, std::chrono::steady_clock::now());
    stage_start = std::chrono::steady_clock::now();
    prepared.setup_report = &setup_report;
    prepared.benchmark_start = benchmark_setup_start;

    const auto gpu_timing_json = [](const PqPreparedQueryState& state) {
        return nlohmann::json{{"query_buffer_allocate_ms", state.buffer_allocate_ms},
            {"query_upload_ms", state.query_upload_ms},
            {"graph_lowfreq_context_init_ms", state.graph_lowfreq_context_init_ms},
            {"pq_table_preprocessing", "inside_search_kernel_included_in_qps"}};
    };
    setup_report.append({{"record_type", "workload_setup"},
        {"summary_csv", output_csv_path},
        {"query_vector_path", base_options.query_vector_path},
        {"query_label_spmat_path", base_options.query_label_spmat_path},
        {"query_manifest_jsonl_path", base_options.query_manifest_jsonl_path},
        {"query_range_path", base_options.query_range_path},
        {"original_queries", runtime.original_query_count},
        {"primary_routes", runtime.query_vectors.rows},
        {"setup_total_before_sweep_ms",
            setup_elapsed_ms(benchmark_setup_start, std::chrono::steady_clock::now())},
        {"setup_stages_ms", setup_stages},
        {"primary_runtime", query_preparation_timing_json(runtime.query_preparation_timing)},
        {"immutable_index_gpu_upload_ms", immutable.upload_ms},
        {"primary_gpu_query", gpu_timing_json(prepared)},

        {"qps_timer", "route_partition_through_final_route_merge"},
        {"nested_details_are_additive", false}});
    std::cout << "[query-setup] path=" << setup_path << "\n";

    std::ofstream csv(output_csv_path);
    if (!csv.is_open()) {

        cleanup_benchmark_contexts(immutable, prepared, graph_ctx, lowfreq_ctx);
        fail_pq("failed to open benchmark output CSV: " + output_csv_path);
    }
    csv << "quick_filter_bonus,queue_size,finish_count,qps,kernel_ms,total_steps,avg_step_num,"
           "highfreq_route_count,lowfreq_route_count,unroutable_route_count,"
           "highfreq_total_steps,highfreq_avg_step_num,lowfreq_total_steps,lowfreq_avg_step_num,"
           "pass_ratio,workers_thread\n";

    try {
        for (float quick_filter_bonus : quick_filter_bonuses) {
            if (quick_filter_bonus <= 0.0f || quick_filter_bonus > 1.0f) {
                fail_pq("benchmark quick_filter_bonus must be in (0, 1]");
            }
            for (int queue_size : queue_sizes) {
                if (queue_size <= 0 || queue_size > 512) {
                    fail_pq("benchmark queue_size must be in [1, 512]");
                }
                if (queue_size < base_options.topk) {
                    fail_pq("benchmark queue_size must be >= topk");
                }
                const auto point_setup_start = std::chrono::steady_clock::now();
                const int finish_count = base_options.search_finish_count;
                PqSearchOptions iteration_options = base_options;
                iteration_options.quick_filter_bonus = quick_filter_bonus;
                iteration_options.bit_filter_bonus = quick_filter_bonus;
                iteration_options.range_filter_bonus = quick_filter_bonus;
                iteration_options.search_queue_size = queue_size;
                iteration_options.print_summary = false;

                if (graph_ctx.d_data != nullptr &&
                    (graph_ctx.result_width != highfreq_result_width(iteration_options) ||
                     graph_ctx.stop_width != highfreq_stop_width(iteration_options) ||
                     (graph_ctx.finite_query_state_enabled &&
                      graph_ctx.finite_queue_capacity !=
                          specialized_queue_capacity(iteration_options.search_queue_size)))) {

                    cleanup_graph_context(graph_ctx);
                    init_graph_context_prepared(graph_ctx,
                                                runtime.query_vectors.rows,
                                                immutable,
                                                prepared,
                                                runtime,
                                                iteration_options);
                }

                const std::string point_results_path =
                    make_benchmark_results_path(output_csv_path,
                                                quick_filter_bonus,
                                                queue_size,
                                                finish_count);
                const std::string point_steps_path =
                    make_benchmark_steps_path(output_csv_path,
                                              quick_filter_bonus,
                                              queue_size,
                                              finish_count);
                const std::string point_routes_path =
                    make_benchmark_routes_path(output_csv_path,
                                               quick_filter_bonus,
                                               queue_size,
                                               finish_count);
                const std::string timed_runs_path =
                    make_benchmark_timed_runs_path(output_csv_path,
                                                   quick_filter_bonus,
                                                   queue_size,
                                                   finish_count);
                go_fann::DurableCsvAppender timed_runs_csv(
                    timed_runs_path,
                    "point_id,phase,run_index,wall_ms,qps,query_count,total_steps,"
                    "highfreq_steps,lowfreq_steps,result_path,steps_path,"
                    "routes_path");
                const std::string point_id =
                    "b" + std::to_string(quick_filter_bonus) +
                    "_q" + std::to_string(queue_size) +
                    "_f" + std::to_string(finish_count);
                const auto after_timed_run = [&](int run,
                                                 const PqTimedRunMetrics& run_metrics) {

                    std::ostringstream row;
                    row << std::setprecision(17)
                        << csv_field(point_id) << ",timed," << run << ','
                        << run_metrics.wall_ms << ',' << run_metrics.qps << ','
                        << run_metrics.query_count << ',' << run_metrics.total_steps << ','
                        << run_metrics.highfreq_steps << ',' << run_metrics.lowfreq_steps << ','

                        << csv_field(point_results_path) << ','
                        << csv_field(point_steps_path) << ','
                        << csv_field(point_routes_path);
                    timed_runs_csv.append_row(row.str());

                };
                reset_search_state(graph_ctx, runtime.query_vectors.rows);
                prepared.setup_point_id = point_id;
                setup_report.append({{"record_type", "point_setup"}, {"point_id", point_id},
                    {"query_setup_reused", true},
                    {"point_context_and_harness_setup_ms",
                        setup_elapsed_ms(point_setup_start, std::chrono::steady_clock::now())}});
                PqIterationResult result = execute_single_iteration(graph_ctx,
                                                                    lowfreq_ctx,
                                                                    prepared,
                                                                    immutable,
                                                                    runtime,
                                                                    iteration_options,
                                                                    after_timed_run);
                const double avg_step_num = runtime.query_vectors.rows > 0
                    ? static_cast<double>(result.total_steps) / static_cast<double>(runtime.query_vectors.rows)
                    : 0.0;
                const double highfreq_avg_step_num =
                    average_route_steps(result.highfreq_total_steps, result.highfreq_route_count);
                const double lowfreq_avg_step_num =
                    average_route_steps(result.lowfreq_total_steps, result.lowfreq_route_count);

                std::cout << "[benchmark] quick_filter_bonus=" << quick_filter_bonus
                          << " queue_size=" << queue_size
                          << " finish_count=" << finish_count
                          << " qps=" << result.qps
                          << " kernel_ms=" << result.kernel_ms
                          << " avg_step_num=" << avg_step_num
                          << " highfreq_avg_step_num=" << highfreq_avg_step_num
                          << " lowfreq_avg_step_num=" << lowfreq_avg_step_num
                          << " pass_ratio=" << result.pass_ratio
                          << " workers_thread=" << result.workers_thread
                          << "\n";
                std::cout.flush();
                csv << quick_filter_bonus << ','
                    << queue_size << ','
                    << finish_count << ','
                    << result.qps << ','
                    << result.kernel_ms << ','
                    << result.total_steps << ','
                    << avg_step_num << ','
                    << result.highfreq_route_count << ','
                    << result.lowfreq_route_count << ','
                    << result.unroutable_route_count << ','
                    << result.highfreq_total_steps << ','
                    << highfreq_avg_step_num << ','
                    << result.lowfreq_total_steps << ','
                    << lowfreq_avg_step_num << ','
                    << result.pass_ratio << ','
                    << result.workers_thread << '\n';

                save_results_to_csv(result.results, point_results_path);
                save_step_nums_to_csv(result.step_nums, point_steps_path);
                save_route_flags_to_csv(result.route_flags, point_routes_path);
                csv.flush();

            }
        }
    } catch (...) {

        cleanup_benchmark_contexts(immutable, prepared, graph_ctx, lowfreq_ctx);
        throw;
    }

    cleanup_benchmark_contexts(immutable, prepared, graph_ctx, lowfreq_ctx);
    return 0;
}

void merge_subset_results(const std::vector<uint32_t>& query_indices,
                         std::vector<std::vector<idx_t>>& merged_results,
                         std::vector<int>& merged_steps,
                         std::vector<std::vector<idx_t>>& subset_results,
                         std::vector<int>& subset_steps)
{
    for (size_t i = 0; i < query_indices.size(); ++i) {
        const uint32_t query_idx = query_indices[i];
        merged_results[query_idx] = std::move(subset_results[i]);
        merged_steps[query_idx] = subset_steps[i];
    }
}

void merge_route_results_to_original(const GraphRuntime& runtime,
                                     const std::vector<std::vector<idx_t>>& route_results,
                                     const std::vector<std::vector<float>>& route_distances,
                                     const std::vector<int>& route_steps,
                                     uint32_t topk,
                                     std::vector<std::vector<idx_t>>& merged_results,
                                     std::vector<int>& merged_steps)
{
    perf_trace::Scope trace_scope("final_route_merge", &runtime);
    if (route_results.size() != route_steps.size()) {
        fail_pq("route result and step counts differ while merging benchmark results");
    }
    go_fann::backend_rabitq::lowfreq_merge::merge_route_results_to_original(
        runtime.route_to_original_query,
        runtime.original_query_count,
        route_results,
        route_distances,
        route_steps,
        topk,
        merged_results,
        merged_steps);
}

int run_pq_search_with_runtime(const PqSearchOptions& options,
                               const GraphRuntime& runtime,
                               const PqHostIndex& pq_index,
                               const std::vector<uint32_t>& query_indices,
                               std::vector<std::vector<idx_t>>& out_results,
                               std::vector<int>& out_step_nums,
                               std::vector<std::vector<float>>* out_distances)
{
    parse_pq_buffer_mode(options.buffer_mode);
    validate_pq_query_worker_mode(options);
    require_uint8_graph_runtime(runtime, "native PQ search");
    require_pq_graph_dim_supported(options.dim);
    require_pq_graph_pq_dim_supported(options.pq_dim);
    if (query_indices.empty()) {
        out_results.clear();
        out_step_nums.clear();
        return 0;
    }

    GraphRuntime subset_runtime = build_query_subset_runtime(runtime, query_indices);
    trace_query_subset_mapping(runtime, query_indices, "run_pq_search_with_runtime");
    auto queries = build_queries_from_dense_subset(runtime.query_vectors, query_indices);

    printf("Start native PQ search\n");
    PqImmutableDeviceState immutable_state;
    try {
        init_immutable_state(immutable_state, pq_index, options);
        native_astar_multi_start_search_batch(queries,
                                              out_results,
                                              out_step_nums,
                                              out_distances,
                                              immutable_state,
                                              subset_runtime,
                                              options);
        cleanup_immutable_state(immutable_state);
    } catch (...) {
        cleanup_immutable_state(immutable_state);
        throw;
    }
    if (highfreq_gpu_exact_rerank_enabled(options) && options.topk < kTopk) {
        for (auto& row : out_results) {
            row.resize(static_cast<size_t>(options.topk));
        }
    }
    return 0;
}

int run_lowfreq_pq_search_with_runtime(const PqSearchOptions& options,
                                       const GraphRuntime& runtime,
                                       const PqHostIndex& pq_index,
                                       const std::vector<uint32_t>& query_indices,
                                       std::vector<std::vector<idx_t>>& out_results,
                                       std::vector<int>& out_step_nums,
                                       std::vector<std::vector<float>>* out_distances)
{
    PqImmutableDeviceState immutable_state;
    PqPreparedQueryState prepared_state;
    PqLowfreqSearchContext lowfreq_ctx;
    try {
        init_immutable_state(immutable_state, pq_index, options);
        init_prepared_query_state(prepared_state, runtime, options);
        init_lowfreq_context(lowfreq_ctx, immutable_state, prepared_state, runtime, options);
        uint32_t* d_query_ids = nullptr;
        if (!query_indices.empty()) {
            check_cuda(cudaMalloc(&d_query_ids, sizeof(uint32_t) * query_indices.size()), "cudaMalloc lowfreq wrapper d_query_ids");
            check_cuda(cudaMemcpy(d_query_ids,
                                  query_indices.data(),
                                  sizeof(uint32_t) * query_indices.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy lowfreq wrapper d_query_ids");
        }
        run_lowfreq_pq_search_with_context(options,
                                           runtime,
                                           query_indices,
                                           d_query_ids,
                                           lowfreq_ctx,
                                           out_results,
                                           out_step_nums,
                                           out_distances);
        if (d_query_ids != nullptr) cudaFree(d_query_ids);
        cleanup_lowfreq_context(lowfreq_ctx);
        cleanup_prepared_query_state(prepared_state);
        cleanup_immutable_state(immutable_state);
    } catch (...) {
        cleanup_lowfreq_context(lowfreq_ctx);
        cleanup_prepared_query_state(prepared_state);
        cleanup_immutable_state(immutable_state);
        throw;
    }
    return 0;
}

}
