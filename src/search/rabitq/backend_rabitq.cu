#include "search/rabitq/backend_rabitq.cuh"
#include "profiling/expansion_trace.cuh"
#include "data/dataloader.h"
#include "profiling/performance_trace.h"
#include "search/rabitq/rabitq_query_rotation.cuh"
#include "predicates/gpu_predicate_index.h"
#include "predicates/gpu_predicate_csr.h"
#include "predicates/gpu_predicate_parallel.cuh"
#include "predicates/gpu_predicate_metadata.h"
#include "planning/route_predicate_plan.h"
namespace gpu_predicate = go_fann::gpu_predicate;

namespace perf_trace = go_fann::performance_trace;

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <fstream>
#include <filesystem>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string_view>
#include <thread>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <sys/resource.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <gdrapi.h>
#include <gdrconfig.h>
#include <immintrin.h>

#include "bin_heap.h"
#include "config1B.h"
#include "fixhash.h"
#include "smmh2.h"
#include "execution/async_io.h"
#include "profiling/durable_csv.h"
#include "execution/host_request_ring.cuh"
#include "search/common/native_visited_hash.cuh"
#include "search/pq/pq_finite_scheduler.h"
#include "search/common/posting_shortlist_heap.cuh"
#include "execution/rabitq_cpu_worker_budget.h"
#include "execution/rabitq_cpu_worker_pool.h"
#include "execution/posting_segment_plan.h"
#include "execution/posting_completion.h"
#include "execution/posting_batch_pipeline.h"
#include "execution/posting_completion.cuh"
#include "search/common/posting_segment_merge.cuh"
#include "search/rabitq/rabitq_lowfreq_merge.h"
#include "profiling/query_setup_timing.h"
#include "profiling/cpu_rerank_profile.h"
#include "execution/exact_distance_memo.h"
#include "execution/incremental_rerank.h"
#include "predicates/range_filter.cuh"
#include "index/range_posting_index.h"

namespace cpu_profile = go_fann::cpu_rerank_profile;

#ifndef GO_FANN_RABITQ_PHASE_SNAPSHOT_PROFILE
#define GO_FANN_RABITQ_PHASE_SNAPSHOT_PROFILE 0
#endif
#include "search/common/range_posting_merge.h"
#include "search/common/warp_kernel.cuh"

namespace go_fann::backend_rabitq {

namespace {

constexpr int kGraphMaxDim = 192;
constexpr int kTopk = 10;
#ifndef GO_FANN_RABITQ_MULTI_QUERY
#define GO_FANN_RABITQ_MULTI_QUERY 1
#endif
#ifndef GO_FANN_RABITQ_NATIVE_DETAILED_TIMING
#define GO_FANN_RABITQ_NATIVE_DETAILED_TIMING 1
#endif
#ifndef GO_FANN_RABITQ_NATIVE_COOPERATIVE_PICKUP
#define GO_FANN_RABITQ_NATIVE_COOPERATIVE_PICKUP 1
#endif
constexpr int kNativeMultiQuery = GO_FANN_RABITQ_MULTI_QUERY;
constexpr bool kNativeDetailedTiming = GO_FANN_RABITQ_NATIVE_DETAILED_TIMING != 0;
constexpr bool kNativeCooperativePickup =
    GO_FANN_RABITQ_NATIVE_COOPERATIVE_PICKUP != 0;
constexpr int kCriticalStep = 32;
constexpr int kThreadInBlock = 32;
constexpr int kNativeThreadInBlock = kCriticalStep * kNativeMultiQuery;
constexpr int kNativeTargetWarpsPerSm = 24;
constexpr int kNativeMinBlocksPerSm = kNativeTargetWarpsPerSm / kNativeMultiQuery;
constexpr int kMultiprobe = 1;
constexpr int kHashTableCapacity = 4096;
constexpr size_t kGraphDatadistSlotCapacity =
    static_cast<size_t>(FIXED_DEGREE) * kMultiprobe;
static_assert(kNativeMultiQuery >= 1 && kNativeMultiQuery <= 4,
              "GO_FANN_RABITQ_MULTI_QUERY must be in [1, 4]");
static_assert(kNativeTargetWarpsPerSm % kNativeMultiQuery == 0,
              "RabitQ native occupancy target must divide by the MQ width");
static_assert(GO_FANN_RABITQ_NATIVE_DETAILED_TIMING == 0 ||
                  GO_FANN_RABITQ_NATIVE_DETAILED_TIMING == 1,
              "GO_FANN_RABITQ_NATIVE_DETAILED_TIMING must be 0 or 1");
static_assert(GO_FANN_RABITQ_NATIVE_COOPERATIVE_PICKUP == 0 ||
                  GO_FANN_RABITQ_NATIVE_COOPERATIVE_PICKUP == 1,
              "GO_FANN_RABITQ_NATIVE_COOPERATIVE_PICKUP must be 0 or 1");
constexpr size_t kBlockSize = 4096;
constexpr size_t kElemsPerBlock = kBlockSize / sizeof(uint32_t);
constexpr size_t kGdrAlignBytes = 65536;

bool cuda_profiler_capture_range_enabled()
{
    static const bool enabled =
        std::getenv("GO_FANN_CUDA_PROFILER_CAPTURE_RANGE") != nullptr;
    return enabled;
}

uint64_t wall_time_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}
constexpr int kInitialWaitSleepUs = 32;
constexpr int kFixedWaitSleepUs = 100;
constexpr int kMinWaitSleepUs = 2;
constexpr int kMaxWaitSleepUs = 64;
constexpr size_t kRabitqMailboxRttBinCount = 9;
constexpr size_t kRabitqMailboxRttRecordWords = kRabitqMailboxRttBinCount + 1;
constexpr size_t kRabitqMailboxRttSumWord = kRabitqMailboxRttBinCount;
constexpr uint64_t kRabitqMailboxRttUpperUs[] = {4, 10, 20, 50, 100, 200, 500, 1000};
static_assert(sizeof(kRabitqMailboxRttUpperUs) / sizeof(kRabitqMailboxRttUpperUs[0]) + 1 ==
                  kRabitqMailboxRttBinCount,
              "mailbox RTT thresholds must cover every bounded histogram bin");
constexpr int kHighfreqRerankShortlistMax = defaults::max_highfreq_rerank_shortlist;
constexpr int kLowfreqRerankShortlistMax = defaults::max_lowfreq_rerank_shortlist;
constexpr int kRangePostingShortlistMax =
    defaults::max_range_posting_postfilter_shortlist;
constexpr uint32_t kRabitqArtifactVersion = 2;
constexpr uint32_t kRabitqLayoutSampleMajor = 1u << 0;
constexpr uint32_t kRabitqLayoutShortCodeMsb32 = 1u << 1;
constexpr uint32_t kRabitqLayoutCuvsBatchFactors = 1u << 2;
constexpr uint32_t kRabitqMaxBitsPerDim = 9;
constexpr uint32_t kCuvsShortFactorCount = 3;
constexpr uint32_t kCuvsExFactorCount = 2;
constexpr uint32_t kRabitqLutChunkBits = 4;
constexpr uint32_t kRabitqLutEntries = 1u << kRabitqLutChunkBits;
constexpr uint32_t kPersistentLatencyStopSeq = UINT32_MAX;

constexpr uint32_t kRabitqGpuGraphRowCacheDegree = 32;

struct alignas(16) RabitqGpuGraphRowCacheEntry {
    unsigned long long sequence = 0;
    uint32_t global_row = UINT32_MAX;
    uint32_t degree = 0;
    uint32_t neighbors[kRabitqGpuGraphRowCacheDegree]{};
};

static_assert(sizeof(RabitqGpuGraphRowCacheEntry) == 144,
              "RabitQ GPU graph row cache entries must remain compact");

struct RabitqGpuLabelCsr {
    std::vector<uint32_t> indptr;
    std::vector<uint32_t> indices;
};

struct RabitqGpuGraphRowCacheView {
    RabitqGpuGraphRowCacheEntry* entries = nullptr;
    uint32_t capacity = 0;
};

struct RabitqGpuGraphCacheDeviceView {
    RabitqGpuGraphRowCacheEntry* entries = nullptr;
    const uint32_t* resident_graph = nullptr;
    const uint32_t* query_label_indptr = nullptr;
    const uint32_t* query_label_indices = nullptr;
    const uint32_t* sample_label_indptr = nullptr;
    const uint32_t* sample_label_indices = nullptr;
    uint32_t capacity = 0;
    uint32_t resident_degree = 0;
};

__device__ __constant__ uint32_t kRabitqDp4aNibbleMasks[kRabitqLutEntries] = {
    0x00000000u, 0x01000000u, 0x00010000u, 0x01010000u,
    0x00000100u, 0x01000100u, 0x00010100u, 0x01010100u,
    0x00000001u, 0x01000001u, 0x00010001u, 0x01010001u,
    0x00000101u, 0x01000101u, 0x00010101u, 0x01010101u,
};

[[noreturn]] void fail_rabitq(const std::string& message)
{
    throw std::runtime_error("rabitq backend failed: " + message);
}

int lowfreq_shortlist_capacity(int shortlist_k)
{
    if (shortlist_k <= 32) return 32;
    if (shortlist_k <= 64) return 64;
    if (shortlist_k <= 100) return 100;
    if (shortlist_k <= 128) return 128;
    if (shortlist_k <= 200) return 200;
    if (shortlist_k <= 256) return 256;
    if (shortlist_k <= kLowfreqRerankShortlistMax) return kLowfreqRerankShortlistMax;
    fail_rabitq("lowfreq shortlist exceeds the compiled maximum");
}

template <typename Launch>
void dispatch_lowfreq_scan_kernel(int shortlist_k, bool use_int8_score, Launch&& launch)
{
    if (shortlist_k <= 32) {
        if (use_int8_score) launch.template operator()<32, true>();
        else launch.template operator()<32, false>();
        return;
    }
    if (shortlist_k <= 64) {
        if (use_int8_score) launch.template operator()<64, true>();
        else launch.template operator()<64, false>();
        return;
    }
    if (shortlist_k <= 100) {
        if (use_int8_score) launch.template operator()<100, true>();
        else launch.template operator()<100, false>();
        return;
    }
    if (shortlist_k <= 128) {
        if (use_int8_score) launch.template operator()<128, true>();
        else launch.template operator()<128, false>();
        return;
    }
    if (shortlist_k <= 200) {
        if (use_int8_score) launch.template operator()<200, true>();
        else launch.template operator()<200, false>();
        return;
    }
    if (shortlist_k <= 256) {
        if (use_int8_score) launch.template operator()<256, true>();
        else launch.template operator()<256, false>();
        return;
    }
    if (shortlist_k <= kLowfreqRerankShortlistMax) {
        if (use_int8_score) {
            launch.template operator()<kLowfreqRerankShortlistMax, true>();
        } else {
            launch.template operator()<kLowfreqRerankShortlistMax, false>();
        }
        return;
    }
    fail_rabitq("lowfreq shortlist exceeds the compiled maximum");
}

int range_posting_shortlist_capacity(int shortlist_k)
{
    if (shortlist_k <= kLowfreqRerankShortlistMax) {
        return lowfreq_shortlist_capacity(shortlist_k);
    }
    if (shortlist_k <= 512) return 512;
    if (shortlist_k <= 1024) return 1024;
    if (shortlist_k <= 2048) return 2048;
    if (shortlist_k <= kRangePostingShortlistMax) return kRangePostingShortlistMax;
    fail_rabitq("range posting shortlist exceeds the compiled maximum");
}

template <typename Launch>
void dispatch_range_posting_scan_kernel(int shortlist_k,
                                        bool use_int8_score,
                                        Launch&& launch)
{
    if (shortlist_k <= kLowfreqRerankShortlistMax) {
        dispatch_lowfreq_scan_kernel(
            shortlist_k, use_int8_score, std::forward<Launch>(launch));
        return;
    }
    if (shortlist_k <= 512) {
        if (use_int8_score) launch.template operator()<512, true>();
        else launch.template operator()<512, false>();
        return;
    }
    if (shortlist_k <= 1024) {
        if (use_int8_score) launch.template operator()<1024, true>();
        else launch.template operator()<1024, false>();
        return;
    }
    if (shortlist_k <= 2048) {
        if (use_int8_score) launch.template operator()<2048, true>();
        else launch.template operator()<2048, false>();
        return;
    }
    if (shortlist_k <= kRangePostingShortlistMax) {
        if (use_int8_score) launch.template operator()<kRangePostingShortlistMax, true>();
        else launch.template operator()<kRangePostingShortlistMax, false>();
        return;
    }
    fail_rabitq("range posting shortlist exceeds the compiled maximum");
}

size_t checked_size_product(size_t lhs, size_t rhs, const char* description)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        fail_rabitq(std::string(description) + " size overflows size_t");
    }
    return lhs * rhs;
}

struct alignas(64) RabitqPersistentLatencyRequestControl {
    uint32_t ready_seq = 0;
    uint32_t stop = 0;
    uint32_t query_id = UINT32_MAX;
    uint32_t lowfreq_route = 0;
};

struct alignas(64) RabitqPersistentLatencyCompletionControl {
    uint32_t done_seq = 0;
    uint32_t query_id = UINT32_MAX;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint64_t gpu_start_clock = 0;
    uint64_t gpu_stop_clock = 0;
};

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
        if (values.empty()) {
            return 0.0;
        }
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

bool lowfreq_profile_enabled()
{
    static const bool enabled = std::getenv("GO_FANN_LOWFREQ_PROFILE") != nullptr;
    return enabled;
}

// A profiler may synchronize inside a kernel launch. The mailbox responder
// must already be running in that diagnostic mode; ordinary launch order stays
// unchanged. Application replay is still required to replay host state.
bool profile_early_host_service_enabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("GO_FANN_PROFILE_HOST_SERVICE_EARLY");
        if (value == nullptr || std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") == 0) return true;
        fail_rabitq("GO_FANN_PROFILE_HOST_SERVICE_EARLY must be 0 or 1");
    }();
    return enabled;
}

bool mailbox_rtt_profile_enabled()
{
    static const bool enabled =
        std::getenv("GO_FANN_RABITQ_MAILBOX_RTT_PROFILE") != nullptr;
    return enabled;
}

bool graph_phase_profile_enabled()
{
    static const bool enabled =
        std::getenv("GO_FANN_RABITQ_GRAPH_PROFILE") != nullptr;
    return enabled;
}

bool graph_phase_snapshot_profile_enabled()
{
#if GO_FANN_RABITQ_PHASE_SNAPSHOT_PROFILE
    static const bool enabled = [] {
        const char* path = std::getenv("GO_FANN_RABITQ_PHASE_SNAPSHOT_DIR");
        return path != nullptr && path[0] != '\0';
    }();
    return enabled;
#else
    return false;
#endif
}

uint32_t graph_phase_snapshot_interval_us()
{
    static const uint32_t interval = [] {
        const char* text = std::getenv("GO_FANN_RABITQ_PHASE_SNAPSHOT_INTERVAL_US");
        if (text == nullptr || text[0] == '\0') return uint32_t{10000};
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(text, &end, 10);
        if (end == text || *end != '\0' || parsed < 1000 || parsed > 1000000) {
            fail_rabitq(
                "GO_FANN_RABITQ_PHASE_SNAPSHOT_INTERVAL_US must be in [1000, 1000000]");
        }
        return static_cast<uint32_t>(parsed);
    }();
    return interval;
}

enum class RabitqWarpPhase : uint16_t {
    Inactive = 0,
    Wait = 1,
    Score = 2,
    Frontier = 3,
    Done = 4,
};

__device__ __forceinline__ void set_rabitq_warp_phase(uint16_t* phase_state,
                                                       uint32_t route_idx,
                                                       int lane,
                                                       RabitqWarpPhase phase)
{
#if GO_FANN_RABITQ_PHASE_SNAPSHOT_PROFILE
    if (phase_state == nullptr || lane != 0) return;
    const uint16_t packed = static_cast<uint16_t>(
        (warp_kernel::get_smid() << 8) | static_cast<uint16_t>(phase));
    reinterpret_cast<volatile uint16_t*>(phase_state)[route_idx] = packed;
#else
    (void)phase_state;
    (void)route_idx;
    (void)lane;
    (void)phase;
#endif
}

RabitqGpuLabelCsr build_rabitq_gpu_sample_label_csr(const SparseLabelMatrix& matrix)
{
    if (matrix.indptr.size() != static_cast<size_t>(matrix.rows) + 1 ||
        matrix.indices.size() > std::numeric_limits<uint32_t>::max()) {
        fail_rabitq("GPU graph row cache requires a uint32-addressable sample label CSR");
    }
    RabitqGpuLabelCsr csr;
    csr.indptr.resize(matrix.indptr.size());
    csr.indices = matrix.indices;
    for (size_t i = 0; i < matrix.indptr.size(); ++i) {
        if (matrix.indptr[i] > std::numeric_limits<uint32_t>::max()) {
            fail_rabitq("GPU graph row cache sample label CSR offset exceeds uint32 range");
        }
        csr.indptr[i] = static_cast<uint32_t>(matrix.indptr[i]);
    }
    return csr;
}

RabitqGpuLabelCsr build_rabitq_gpu_query_label_csr(
    const SparseLabelMatrix& matrix,
    const std::vector<uint32_t>& primary_labels)
{
    if (matrix.indptr.size() != static_cast<size_t>(matrix.rows) + 1 ||
        primary_labels.size() < matrix.rows) {
        fail_rabitq("GPU graph row cache query labels are not aligned with routed queries");
    }
    RabitqGpuLabelCsr csr;
    csr.indptr.reserve(static_cast<size_t>(matrix.rows) + 1);
    csr.indptr.push_back(0);
    for (uint32_t row = 0; row < matrix.rows; ++row) {
        const SparseLabelRowSpan labels = sparse_label_row_span(matrix, row);
        if (!labels.valid) {
            fail_rabitq("GPU graph row cache query label CSR contains an invalid row");
        }
        for (const uint32_t* it = labels.begin; it != labels.end; ++it) {
            if (*it != primary_labels[row]) {
                csr.indices.push_back(*it);
            }
        }
        if (csr.indices.size() > std::numeric_limits<uint32_t>::max()) {
            fail_rabitq("GPU graph row cache query label CSR exceeds uint32 range");
        }
        csr.indptr.push_back(static_cast<uint32_t>(csr.indices.size()));
    }
    return csr;
}

__device__ __forceinline__ size_t rabitq_mailbox_rtt_bin(uint64_t elapsed_ns)
{
    if (elapsed_ns <= 4'000ULL) return 0;
    if (elapsed_ns <= 10'000ULL) return 1;
    if (elapsed_ns <= 20'000ULL) return 2;
    if (elapsed_ns <= 50'000ULL) return 3;
    if (elapsed_ns <= 100'000ULL) return 4;
    if (elapsed_ns <= 200'000ULL) return 5;
    if (elapsed_ns <= 500'000ULL) return 6;
    if (elapsed_ns <= 1'000'000ULL) return 7;
    return 8;
}

bool lowfreq_use_filter_mask_enabled()
{
    static const bool enabled = std::getenv("GO_FANN_LOWFREQ_USE_FILTER_MASK") != nullptr;
    return enabled;
}

bool lowfreq_gpu_posting_enabled()
{
    static const bool enabled = std::getenv("GO_FANN_LOWFREQ_GPU_POSTING") != nullptr;
    return enabled;
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct RabitqCpuPhaseTiming {
    double queue_wait_ms = 0.0;
    double active_ms = 0.0;
    double total_ms = 0.0;
};

template <typename Function>
void run_timed_rabitq_cpu_phase(
    const std::shared_ptr<RabitqCpuPhaseTiming>& timing,
    Function&& function)
{
    const auto active_at = std::chrono::steady_clock::now();
    if (timing != nullptr) {
        timing->queue_wait_ms = 0.0;
    }
    try {
        std::forward<Function>(function)();
    } catch (...) {
        const auto failed_at = std::chrono::steady_clock::now();
        if (timing != nullptr) {
            timing->active_ms = elapsed_ms(active_at, failed_at);
            timing->total_ms = timing->active_ms;
        }
        throw;
    }
    const auto completed_at = std::chrono::steady_clock::now();
    if (timing != nullptr) {
        timing->active_ms = elapsed_ms(active_at, completed_at);
        timing->total_ms = timing->active_ms;
    }
}

__device__ __forceinline__ uint32_t load_bypass_l1_u32(const volatile uint32_t* ptr)
{
    uint32_t value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
    return value;
}

bool use_adaptive_wait_policy(const std::string& policy)
{
    if (policy == "fixed") {
        return false;
    }
    if (policy == "adaptive") {
        return true;
    }
    fail_rabitq("unsupported host_reply_wait_policy: " + policy);
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
    fail_rabitq("unsupported search_early_stop_mode: " + mode);
    return kSearchEarlyStopFixed;
}

int effective_search_max_steps(const RabitqSearchOptions& options)
{
    if (options.search_max_steps > 0) {
        return options.search_max_steps;
    }
    return options.search_queue_size * 2;
}

std::string_view effective_highfreq_rerank_mode(const RabitqSearchOptions& options)
{
    if (!options.enable_highfreq_rerank &&
        (defaults::highfreq_rerank_mode_is_gpu(options.highfreq_rerank_mode) ||
         defaults::highfreq_rerank_mode_is_gpu_post(options.highfreq_rerank_mode))) {
        return defaults::highfreq_rerank_mode_off;
    }
    return options.highfreq_rerank_mode;
}

bool highfreq_gpu_exact_rerank_enabled(const RabitqSearchOptions& options)
{
    const std::string_view mode = effective_highfreq_rerank_mode(options);
    return defaults::highfreq_rerank_mode_is_gpu(mode) ||
        defaults::highfreq_rerank_mode_is_gpu_post(mode);
}

bool highfreq_inline_gpu_exact_rerank_enabled(const RabitqSearchOptions& options,
                                              DenseVectorScalarType scalar_type)
{
    return defaults::highfreq_rerank_mode_is_gpu(
               effective_highfreq_rerank_mode(options)) &&
        (scalar_type == DenseVectorScalarType::Uint8 ||
         dense_vector_scalar_is_float(scalar_type));
}

bool highfreq_staged_float_gpu_exact_rerank_enabled(
    const RabitqSearchOptions& options,
    DenseVectorScalarType scalar_type)
{
    return defaults::highfreq_rerank_mode_is_gpu_post(
               effective_highfreq_rerank_mode(options)) &&
        dense_vector_scalar_is_float(scalar_type);
}

bool highfreq_cpu_rerank_enabled(const RabitqSearchOptions& options)
{
    return defaults::highfreq_rerank_mode_is_cpu(effective_highfreq_rerank_mode(options));
}

bool highfreq_cpu_streaming_enabled(const RabitqSearchOptions& options)
{
    return highfreq_cpu_rerank_enabled(options) && options.highfreq_rerank_streaming;
}

bool rabitq_device_resident_graph_enabled(const RabitqSearchOptions& options)
{
    return options.graph_transport == "device_resident";
}

uint32_t effective_highfreq_rerank_shortlist(const RabitqSearchOptions& options)
{
    return static_cast<uint32_t>(defaults::effective_highfreq_rerank_shortlist(
        options.highfreq_rerank_shortlist,
        options.search_queue_size,
        effective_highfreq_rerank_mode(options),
        options.highfreq_rerank_algorithm));
}

uint32_t highfreq_result_width(const RabitqSearchOptions& options,
                               const GraphRuntime& runtime)
{
    if (highfreq_inline_gpu_exact_rerank_enabled(options, runtime.vector_scalar_type) ||
        defaults::highfreq_rerank_mode_is_off(effective_highfreq_rerank_mode(options))) {
        return static_cast<uint32_t>(options.topk);
    }
    return effective_highfreq_rerank_shortlist(options);
}

uint32_t highfreq_stop_width(const RabitqSearchOptions& options)
{
    return static_cast<uint32_t>(options.topk);
}

bool highfreq_neighbor_mailbox_fits(uint32_t degree)
{
    return static_cast<size_t>(degree) + 1U <= kElemsPerBlock;
}

bool query_ranges_have_active_filters(const QueryRangeTable& ranges)
{
    return std::any_of(ranges.modes.begin(), ranges.modes.end(), [](uint8_t mode) {
        return go_fann::range_filter::range_filter_active(mode);
    });
}

int64_t load_status_flag_word(const int64_t* status_flag, size_t query_idx)
{
    const volatile int64_t* flag_ptr = status_flag + query_idx * 16;
    return *flag_ptr;
}

void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        fail_rabitq(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_gdr(int status, const char* what)
{
    if (status != 0) {
        fail_rabitq(std::string(what) + " failed with code " + std::to_string(status));
    }
}

template <typename LaunchFunc>
void dispatch_rabitq_kernel_by_queue_size(int queue_size, LaunchFunc&& launch)
{
    if (queue_size <= 64) {
        launch.template operator()<64>();
    } else if (queue_size <= 128) {
        launch.template operator()<128>();
    } else if (queue_size <= 256) {
        launch.template operator()<256>();
    } else if (queue_size <= 512) {
        launch.template operator()<512>();
    } else {
        fail_rabitq("unsupported queue_size (must be <= 512)");
    }
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

size_t round_up_gdr_bytes(size_t value)
{
    return ((value + kGdrAlignBytes - 1) / kGdrAlignBytes) * kGdrAlignBytes;
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

template <typename T>
T* align_gdr_pointer(T* ptr)
{
    auto raw = reinterpret_cast<uintptr_t>(ptr);
    raw = (raw + kGdrAlignBytes - 1) & ~(static_cast<uintptr_t>(kGdrAlignBytes) - 1);
    return reinterpret_cast<T*>(raw);
}

using RabitqPoolPair = warp_kernel::KernelPair<dist_t, uint32_t>;
using RabitqResultPair = warp_kernel::KernelPair<dist_t, uint32_t>;
using RabitqStopPair = warp_kernel::KernelPair<dist_t, uint64_t>;

static_assert(sizeof(RabitqPoolPair) == 8,
              "RabitQ traversal entries must remain compact");
static_assert(sizeof(RabitqResultPair) == 8,
              "RabitQ result entries must remain compact");

constexpr size_t rabitq_result_heap_shared_bytes(uint32_t result_width)
{
    return static_cast<size_t>(kNativeMultiQuery) *
           (static_cast<size_t>(result_width) + 1) * sizeof(RabitqResultPair);
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

__host__ __device__ __forceinline__ uint64_t pack_topk_payload(uint32_t local_row, dist_t approx_dist)
{
    union {
        dist_t f;
        uint32_t u;
    } bits{};
    bits.f = approx_dist;
    return (static_cast<uint64_t>(bits.u) << 32) | static_cast<uint64_t>(local_row);
}

__host__ __device__ __forceinline__ dist_t unpack_topk_approx_dist(uint64_t payload)
{
    union {
        dist_t f;
        uint32_t u;
    } bits{};
    bits.u = static_cast<uint32_t>(payload >> 32);
    return bits.f;
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

struct RabitqBatchStats {
    size_t query_count = 0;
    double kernel_ms = 0.0;
    double qps_sum = 0.0;
    size_t run_count = 0;
    uint64_t total_step_sum = 0;
    HostFilterStats host_filter_stats{};
};

void accumulate_host_filter_stats(HostFilterStats& total, const HostFilterStats& batch)
{
    total.requests_served += batch.requests_served;
    total.requests_passed += batch.requests_passed;
    total.requests_filtered += batch.requests_filtered;
    total.workers_thread = std::max(total.workers_thread, batch.workers_thread);
    total.demand_batch_count += batch.demand_batch_count;
    total.demand_partial_batch_count += batch.demand_partial_batch_count;
    total.demand_batch_request_count += batch.demand_batch_request_count;
    total.demand_batch_wait_ns += batch.demand_batch_wait_ns;
    total.demand_batch_max_size = std::max(total.demand_batch_max_size,
                                           batch.demand_batch_max_size);
    total.demand_batch_configured_size = std::max(total.demand_batch_configured_size,
                                                  batch.demand_batch_configured_size);
    total.demand_batch_timeout_us = std::max(total.demand_batch_timeout_us,
                                             batch.demand_batch_timeout_us);
    total.request_ring_batches_claimed += batch.request_ring_batches_claimed;
    total.request_ring_demand_events += batch.request_ring_demand_events;
    total.request_ring_done_events += batch.request_ring_done_events;
    total.request_ring_empty_polls += batch.request_ring_empty_polls;
    total.request_ring_profile_enabled = total.request_ring_profile_enabled ||
        batch.request_ring_profile_enabled;
    total.request_ring_profile_sample_period = std::max(
        total.request_ring_profile_sample_period,
        batch.request_ring_profile_sample_period);
    total.request_ring_profile_samples += batch.request_ring_profile_samples;
    total.request_ring_mailbox_snapshot_ns +=
        batch.request_ring_mailbox_snapshot_ns;
    total.request_ring_filter_ns += batch.request_ring_filter_ns;
    total.request_ring_payload_ns += batch.request_ring_payload_ns;
    total.request_ring_publish_ns += batch.request_ring_publish_ns;
    total.request_ring_service_ns += batch.request_ring_service_ns;
    total.request_ring_worker_wall_ns += batch.request_ring_worker_wall_ns;
    total.request_ring_worker_max_wall_ns = std::max(
        total.request_ring_worker_max_wall_ns,
        batch.request_ring_worker_max_wall_ns);

}

void accumulate_rabitq_batch_stats(RabitqBatchStats& total, const RabitqBatchStats& batch)
{
    total.query_count += batch.query_count;
    total.kernel_ms += batch.kernel_ms;
    total.qps_sum += batch.qps_sum;
    total.run_count += batch.run_count;
    total.total_step_sum += batch.total_step_sum;
    accumulate_host_filter_stats(total.host_filter_stats, batch.host_filter_stats);
}

struct RabitqIterationResult {
    std::vector<std::vector<idx_t>> results;
    std::vector<std::vector<float>> distances;
    std::vector<int> step_nums;
    double qps = 0.0;
    double wall_ms = 0.0;
    size_t query_count = 0;
    double kernel_ms = 0.0;
    int64_t total_steps = 0;
    size_t highfreq_route_count = 0;
    size_t lowfreq_route_count = 0;
    size_t provider_route_count = 0;
    size_t unroutable_route_count = 0;
    int64_t highfreq_total_steps = 0;
    int64_t lowfreq_total_steps = 0;
    float pass_ratio = 0.0f;
    int workers_thread = 0;
    HostFilterStats host_filter_stats{};
    std::vector<int> route_flags;
    double gpu_provider_wall_ms = 0.0;
    double cpu_tail_wall_ms = 0.0;
    double range_posting_rerank_wall_ms = 0.0;
    double route_merge_wall_ms = 0.0;
    double pre_qps_total_ms = 0.0;
    double pre_qps_route_partition_ms = 0.0;
    double pre_qps_result_init_ms = 0.0;
    bool pre_qps_counted_in_wall = false;
    std::chrono::steady_clock::time_point qps_start;
    lowfreq_merge::FinalMergeStrategy route_merge_strategy =
        lowfreq_merge::FinalMergeStrategy::HashDedupSort;
};

void report_iteration_preparation(QuerySetupReporter& report,
                                  const std::string& point_id,
                                  const char* phase,
                                  int run,
                                  const RabitqIterationResult& result,
                                  std::chrono::steady_clock::time_point benchmark_start)
{
    report.append({
        {"record_type", "iteration"}, {"point_id", point_id},
        {"phase", phase}, {"run_index", run}, {"query_count", result.query_count},
        {"qps_start_since_benchmark_entry_ms", elapsed_ms(benchmark_start, result.qps_start)},
        {"pre_qps_total_ms", result.pre_qps_total_ms},
        {"pre_qps_route_partition_ms", result.pre_qps_route_partition_ms},
        {"pre_qps_result_init_ms", result.pre_qps_result_init_ms},
        {"pre_qps_other_ms", result.pre_qps_total_ms -
            result.pre_qps_route_partition_ms - result.pre_qps_result_init_ms},
        {"provider_preparation_already_in_qps", result.pre_qps_counted_in_wall},
        {"excluded_pre_qps_ms", result.pre_qps_counted_in_wall ? 0.0 : result.pre_qps_total_ms},
        {"execution_wall_ms", result.wall_ms}, {"qps", result.qps},
        {"route_merge_ms", result.route_merge_wall_ms},
        {"route_merge_scope", result.pre_qps_counted_in_wall ? "primary_provider" : "all_routes"},
        {"route_merge_strategy", lowfreq_merge::final_merge_strategy_name(result.route_merge_strategy)}});
}

void merge_rabitq_iteration_results(RabitqIterationResult& primary,
                                    const RabitqIterationResult& secondary,
                                    uint32_t topk)
{
    if (primary.query_count != secondary.query_count ||
        primary.results.size() != secondary.results.size() ||
        primary.distances.size() != primary.results.size() ||
        secondary.distances.size() != secondary.results.size()) {
        fail_rabitq("dual-graph iteration results are not query-aligned");
    }
    for (size_t query_idx = 0; query_idx < primary.results.size(); ++query_idx) {
        std::vector<std::pair<float, idx_t>> scored;
        scored.reserve(primary.results[query_idx].size() +
                       secondary.results[query_idx].size());
        const auto append_row = [&](const std::vector<idx_t>& ids,
                                    const std::vector<float>& distances) {
            for (size_t rank = 0; rank < ids.size(); ++rank) {
                const idx_t sample_id = ids[rank];
                if (sample_id == UINT32_MAX) {
                    continue;
                }
                const float distance = rank < distances.size()
                    ? distances[rank]
                    : std::numeric_limits<float>::max();
                const auto existing = std::find_if(
                    scored.begin(), scored.end(), [sample_id](const auto& candidate) {
                        return candidate.second == sample_id;
                    });
                if (existing == scored.end()) {
                    scored.emplace_back(distance, sample_id);
                } else if (distance < existing->first) {
                    existing->first = distance;
                }
            }
        };
        append_row(primary.results[query_idx], primary.distances[query_idx]);
        append_row(secondary.results[query_idx], secondary.distances[query_idx]);
        std::sort(scored.begin(), scored.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });
        primary.results[query_idx].assign(topk, UINT32_MAX);
        primary.distances[query_idx].assign(
            topk, std::numeric_limits<float>::max());
        for (size_t rank = 0; rank < scored.size() && rank < topk; ++rank) {
            primary.distances[query_idx][rank] = scored[rank].first;
            primary.results[query_idx][rank] = scored[rank].second;
        }
        if (query_idx < primary.step_nums.size() &&
            query_idx < secondary.step_nums.size()) {
            const int64_t merged_steps = static_cast<int64_t>(primary.step_nums[query_idx]) +
                static_cast<int64_t>(secondary.step_nums[query_idx]);
            if (merged_steps > std::numeric_limits<int>::max()) {
                fail_rabitq("dual-graph step aggregation overflows int");
            }
            primary.step_nums[query_idx] = static_cast<int>(merged_steps);
        }
        if (query_idx < primary.route_flags.size() &&
            query_idx < secondary.route_flags.size()) {
            primary.route_flags[query_idx] |= secondary.route_flags[query_idx];
        }
    }
    primary.kernel_ms += secondary.kernel_ms;
    primary.total_steps += secondary.total_steps;
    primary.highfreq_route_count += secondary.highfreq_route_count;
    primary.lowfreq_route_count += secondary.lowfreq_route_count;
    primary.provider_route_count += secondary.provider_route_count;
    primary.unroutable_route_count += secondary.unroutable_route_count;
    primary.highfreq_total_steps += secondary.highfreq_total_steps;
    primary.lowfreq_total_steps += secondary.lowfreq_total_steps;
    accumulate_host_filter_stats(primary.host_filter_stats, secondary.host_filter_stats);
    primary.workers_thread = std::max(primary.workers_thread, secondary.workers_thread);
    primary.pass_ratio = primary.host_filter_stats.requests_served > 0
        ? static_cast<float>(primary.host_filter_stats.requests_passed) /
            static_cast<float>(primary.host_filter_stats.requests_served)
        : 0.0f;
}

uint32_t round_up_to_multiple(uint32_t value, uint32_t multiple)
{
    return ((value + multiple - 1u) / multiple) * multiple;
}

uint32_t short_code_words_for_dim(uint32_t padded_dim)
{
    return (padded_dim + 31u) / 32u;
}

uint32_t short_code_bytes_for_padded_dim(uint32_t padded_dim)
{
    return short_code_words_for_dim(padded_dim) * static_cast<uint32_t>(sizeof(uint32_t));
}

uint32_t long_code_bytes_for_spec(uint32_t padded_dim, uint32_t bits_per_dim)
{
    if (bits_per_dim <= 1) {
        return 0;
    }
    const uint64_t ex_bits = static_cast<uint64_t>(bits_per_dim - 1u);
    return static_cast<uint32_t>((static_cast<uint64_t>(padded_dim) * ex_bits) / 8u);
}

size_t short_lut_entries_for_dim(uint32_t padded_dim)
{
    return static_cast<size_t>(padded_dim / kRabitqLutChunkBits) * kRabitqLutEntries;
}

RabitqCodecSpec make_codec_spec(uint32_t dim, uint32_t bits_per_dim)
{
    if (bits_per_dim == 0 || bits_per_dim > kRabitqMaxBitsPerDim) {
        fail_rabitq("bits_per_dim must be in [1, 9]");
    }
    RabitqCodecSpec spec;
    spec.padded_dim = round_up_to_multiple(dim, 64);
    spec.bits_per_dim = bits_per_dim;
    spec.short_code_words = short_code_words_for_dim(spec.padded_dim);
    spec.long_code_bytes = long_code_bytes_for_spec(spec.padded_dim, bits_per_dim);
    spec.layout_flags =
        kRabitqLayoutSampleMajor | kRabitqLayoutShortCodeMsb32 | kRabitqLayoutCuvsBatchFactors;
    spec.short_factor_count = kCuvsShortFactorCount;
    spec.ex_factor_count = kCuvsExFactorCount;
    return spec;
}

void validate_codec_spec(const RabitqArtifactHeader& header, const RabitqCodecSpec& codec)
{
    if (codec.padded_dim < header.dim || codec.padded_dim % 64u != 0) {
        fail_rabitq("artifact codec padded_dim is invalid");
    }
    if (codec.bits_per_dim == 0 || codec.bits_per_dim > kRabitqMaxBitsPerDim) {
        fail_rabitq("artifact codec bits_per_dim is invalid");
    }
    if (codec.short_code_words != short_code_words_for_dim(codec.padded_dim)) {
        fail_rabitq("artifact codec short_code_words is invalid");
    }
    if (codec.long_code_bytes != long_code_bytes_for_spec(codec.padded_dim, codec.bits_per_dim)) {
        fail_rabitq("artifact codec long_code_bytes is invalid");
    }
    const uint32_t required_layout =
        kRabitqLayoutSampleMajor | kRabitqLayoutShortCodeMsb32 | kRabitqLayoutCuvsBatchFactors;
    if ((codec.layout_flags & required_layout) != required_layout) {
        fail_rabitq("artifact codec layout_flags are invalid");
    }
    if (codec.short_factor_count != kCuvsShortFactorCount ||
        codec.ex_factor_count != kCuvsExFactorCount) {
        fail_rabitq("artifact codec factor counts are invalid");
    }
}

void validate_header(const RabitqArtifactHeader& header, const RabitqCodecSpec& codec)
{
    if (header.version != kRabitqArtifactVersion) {
        fail_rabitq("unsupported artifact version");
    }
    if (header.dim == 0 || header.num_anchors == 0 || header.num_points == 0) {
        fail_rabitq("artifact header contains zero-sized fields");
    }
    validate_codec_spec(header, codec);
    if (header.code_bytes != short_code_bytes_for_padded_dim(codec.padded_dim)) {
        fail_rabitq("artifact header code_bytes does not match dim");
    }
}

void save_results_to_csv(const std::vector<std::vector<idx_t>>& results, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        fail_rabitq("could not open output file " + filename);
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

void save_step_nums_to_csv(const std::vector<int>& step_nums, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        fail_rabitq("could not open output file " + filename);
    }
    for (int step_num : step_nums) {
        file << step_num << '\n';
    }
}

void save_route_flags_to_csv(const std::vector<int>& route_flags, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        fail_rabitq("could not open output file " + filename);
    }
    for (int route_flag : route_flags) {
        file << route_flag << '\n';
    }
}

std::string make_sidecar_csv_path(const std::string& csv_path, const std::string& suffix)
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
    return make_sidecar_csv_path(summary_csv_path,
                                 ".b" + bonus_tag
                                     + ".q" + std::to_string(queue_size)
                                     + ".f" + std::to_string(finish_count)
                                     + ".results.csv");
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
    return make_sidecar_csv_path(summary_csv_path,
                                 ".b" + bonus_tag
                                     + ".q" + std::to_string(queue_size)
                                     + ".f" + std::to_string(finish_count)
                                     + ".steps.csv");
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
    return make_sidecar_csv_path(summary_csv_path,
                                 ".b" + bonus_tag
                                     + ".q" + std::to_string(queue_size)
                                     + ".f" + std::to_string(finish_count)
                                     + ".routes.csv");
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
    return make_sidecar_csv_path(summary_csv_path,
                                 ".b" + bonus_tag
                                     + ".q" + std::to_string(queue_size)
                                     + ".f" + std::to_string(finish_count)
                                     + ".timed_runs.csv");
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

struct RabitqProcessPageFaults {
    uint64_t minor = 0;
    uint64_t major = 0;
};

RabitqProcessPageFaults snapshot_process_page_faults()
{
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        fail_rabitq("getrusage failed while collecting RabitQ page-fault diagnostics");
    }
    return RabitqProcessPageFaults{
        static_cast<uint64_t>(std::max<decltype(usage.ru_minflt)>(0, usage.ru_minflt)),
        static_cast<uint64_t>(std::max<decltype(usage.ru_majflt)>(0, usage.ru_majflt))};
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

float exact_l2_u8(const uint8_t* query, const uint8_t* base, uint32_t dim)
{
    float dist = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        const float diff = static_cast<float>(query[d]) - static_cast<float>(base[d]);
        dist += diff * diff;
    }
    return dist;
}

struct NativeIncrementalRerank : go_fann::incremental_rerank::State {
    const GraphRuntime& primary;
    // Only the current per-query scorer initializes/reads this small descriptor.
    std::vector<PreparedFloatL2Query> prepared_queries;
    NativeIncrementalRerank(const GraphRuntime& runtime, size_t cap, size_t topk)
        : State(runtime.original_query_count, cap, topk), primary(runtime),
          prepared_queries(runtime.original_query_count) {}
};

// Borrowed only while iteration-owned contexts and all consumers remain alive.
struct CpuRerankReuse {
    go_fann::ExactDistanceMemo* distances = nullptr;
    const std::vector<uint8_t>* verified_posting_routes = nullptr;
    NativeIncrementalRerank* incremental = nullptr;
    bool posting_verified(uint32_t route) const {
        return verified_posting_routes && route < verified_posting_routes->size() &&
            (*verified_posting_routes)[route] != 0;
    }
};

// Warmup-only collection precedes per-delivery merge, global top-L and dedup.
// The store is shared by both hybrid providers; no distances/results are cached.

uint32_t exact_original_query(const GraphRuntime& runtime, uint32_t route) {
    return runtime.route_to_original_query.empty() ? route : runtime.route_to_original_query.at(route);
}

template <typename Evaluate, typename Select>
void evaluate_rerank_batch(go_fann::ExactDistanceMemo* memo, uint32_t original_query,
                           cpu_profile::Task& profile,
                           const GraphRuntime& runtime,
                           const PreparedFloatL2Query* query,
                           const std::vector<uint32_t>& ids,
                           Evaluate&& evaluate, Select&& select)
{
    if (query && runtime.rerank_vector_store && runtime.rerank_vector_store->uses_batched_l2()) {
        profile.evaluate_native_batch(ids,
            [&](const uint32_t* rows, size_t count, float* distances) {
                auto compute = [&](const uint32_t* missing, size_t size, float* scores) {
                    runtime.rerank_vector_store->l2_float_batch(missing, size, *query, scores);
                };
                if (memo) memo->evaluate_batch(original_query, rows, count, distances, compute);
                else compute(rows, count, distances);
            }, std::forward<Select>(select));
    } else {
        profile.evaluate_batch(ids, [&](uint32_t sample) {
            return memo ? memo->evaluate(original_query, sample, [&] { return evaluate(sample); })
                        : evaluate(sample);
        }, std::forward<Select>(select));
    }
}

float highfreq_exact_l2_for_sample(const GraphRuntime& runtime,
                                   const uint8_t* query_u8,
                                   const PreparedFloatL2Query* query_float,
                                   uint32_t sample_id)
{
    if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        if (query_float == nullptr) {
            fail_rabitq("prepared float query is missing for high-frequency CPU rerank");
        }
        return compute_base_float_vector_l2(runtime,
                                            sample_id,
                                            *query_float);
    }

    return exact_l2_u8(query_u8, runtime.base_vectors.row_ptr(sample_id), runtime.query_vectors.cols);
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

void merge_subset_results(const std::vector<uint32_t>& query_indices,
                          std::vector<std::vector<idx_t>>& merged_results,
                          std::vector<int>& merged_steps,
                          std::vector<std::vector<float>>& merged_distances,
                          std::vector<std::vector<idx_t>>& subset_results,
                          std::vector<int>& subset_steps,
                          std::vector<std::vector<float>>& subset_distances)
{
    for (size_t i = 0; i < query_indices.size(); ++i) {
        const uint32_t query_idx = query_indices[i];
        merged_results[query_idx] = std::move(subset_results[i]);
        merged_steps[query_idx] = subset_steps[i];
        merged_distances[query_idx] = std::move(subset_distances[i]);
    }
}

void highfreq_rabitq_host_batch_rerank_range(const CpuRerankReuse& reuse, const GraphRuntime& runtime,
                                             const std::vector<uint32_t>& query_indices,
                                             const std::vector<std::vector<idx_t>>& gpu_shortlist,
                                             uint32_t shortlist_k,
                                             uint32_t topk,
                                             uint32_t rerank_batch_size,
                                             float rerank_epsilon,
                                             uint32_t rerank_stable_batches,
                                             bool allow_early_stop,
                                             bool filter_full_labels,
                                             size_t begin_query,
                                             size_t end_query,
                                             std::vector<std::vector<idx_t>>& out_results,
                                             std::vector<std::vector<float>>* out_distances)
{
    cpu_profile::Task profile("highfreq_route");

    rerank_batch_size = std::max<uint32_t>(1U, rerank_batch_size);
    rerank_stable_batches = std::max<uint32_t>(1U, rerank_stable_batches);

    std::vector<uint32_t> batch_sample_ids;
    batch_sample_ids.reserve(rerank_batch_size);
    for (size_t q = begin_query; q < end_query; ++q) {
        profile.stage(cpu_profile::Stage::Setup);
        profile.query();
        const uint32_t query_idx = query_indices[q];
        const uint8_t* query_u8 = nullptr;
        PreparedFloatL2Query query_float;
        const PreparedFloatL2Query* prepared_query_float = nullptr;
        if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
            query_u8 = runtime.route_query_u8(query_idx);
        } else if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
            query_float = prepare_base_float_vector_l2_query(
                runtime,
                runtime.route_query_float(query_idx),
                runtime.query_float_vectors.cols);
            prepared_query_float = &query_float;
        } else {
            fail_rabitq("unsupported dense vector scalar type for high-frequency CPU rerank");
        }
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
            profile.stage(cpu_profile::Stage::CandidateCollect);
            profile.candidates(batch_end - batch_begin);
            batch_sample_ids.clear();

            for (uint32_t idx = batch_begin; idx < batch_end; ++idx) {
                const idx_t sample_id = gpu_shortlist[q][idx];
                if (sample_id == UINT32_MAX || sample_id >= runtime.base_vectors.rows) {
                    continue;
                }
                const auto sample_id_u32 = static_cast<uint32_t>(sample_id);
                if (allow_early_stop &&
                    !profile.sample(cpu_profile::Detail::HashLookup, [&] {
                        return seen_sample_ids.insert(sample_id_u32);
                    }).second) {
                    continue;
                }
                if (filter_full_labels &&
                    !profile.sample(cpu_profile::Detail::Predicate, [&] {
                        return sample_matches_lowfreq_full_labels(runtime, query_idx, sample_id_u32);
                    })) {
                    continue;
                }
                batch_sample_ids.push_back(sample_id_u32);
            }
            if (batch_sample_ids.empty()) {
                continue;
            }
            profile.stage(cpu_profile::Stage::CandidateSortDedup);
            std::sort(batch_sample_ids.begin(), batch_sample_ids.end());
            batch_sample_ids.erase(std::unique(batch_sample_ids.begin(), batch_sample_ids.end()),
                                   batch_sample_ids.end());
            profile.stage(cpu_profile::Stage::VectorPrepare);

            evaluate_rerank_batch(reuse.distances, exact_original_query(runtime, query_idx), profile, runtime, prepared_query_float, batch_sample_ids, [&](uint32_t sample_id) {
                const float dist = highfreq_exact_l2_for_sample(runtime,
                                                                query_u8,
                                                                prepared_query_float,
                                                                sample_id);
                return dist;
            }, [&](uint32_t sample_id, float dist) {
                if (heap.size() < topk) {
                    heap.emplace_back(dist, sample_id);
                    std::push_heap(heap.begin(), heap.end());
                } else if (dist < heap.front().first ||
                           (dist == heap.front().first && sample_id < heap.front().second)) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = {dist, sample_id};
                    std::push_heap(heap.begin(), heap.end());
                }
            });
            profile.stage(cpu_profile::Stage::EarlyStop);

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
                    if (rerank_stable_batches != UINT32_MAX &&
                        static_cast<uint32_t>(stable_count) >= rerank_stable_batches) {
                        prev_topk = std::move(curr_topk);
                        break;
                    }
                } else {
                    stable_count = 0;
                }
                prev_topk = std::move(curr_topk);
            }
        }

        profile.stage(cpu_profile::Stage::Writeback);
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

struct HighfreqOriginalRerankGroup {
    uint32_t original_query = UINT32_MAX;
    std::vector<size_t> route_positions;
};

struct HighfreqMergedCandidate {
    uint32_t sample_id = UINT32_MAX;
    uint32_t best_rank = UINT32_MAX;
    float best_score = FLT_MAX;
};

uint32_t highfreq_route_original_query_id(const GraphRuntime& runtime, uint32_t route_idx)
{
    if (route_idx < runtime.route_to_original_query.size()) {
        return runtime.route_to_original_query[route_idx];
    }
    return route_idx;
}

std::vector<HighfreqOriginalRerankGroup> build_highfreq_original_rerank_groups(
    const GraphRuntime& runtime,
    const std::vector<uint32_t>& query_indices)
{
    std::vector<HighfreqOriginalRerankGroup> groups;
    groups.reserve(query_indices.size());
    std::unordered_map<uint32_t, size_t> group_by_original;
    group_by_original.reserve(query_indices.size());
    for (size_t pos = 0; pos < query_indices.size(); ++pos) {
        const uint32_t original_query =
            highfreq_route_original_query_id(runtime, query_indices[pos]);
        auto [it, inserted] = group_by_original.emplace(original_query, groups.size());
        if (inserted) {
            HighfreqOriginalRerankGroup group;
            group.original_query = original_query;
            groups.push_back(std::move(group));
        }
        groups[it->second].route_positions.push_back(pos);
    }
    return groups;
}

void highfreq_rabitq_host_original_rerank_group(const CpuRerankReuse& reuse, const GraphRuntime& runtime,
                                                const std::vector<uint32_t>& query_indices,
                                                const std::vector<std::vector<idx_t>>& gpu_shortlist,
                                                const std::vector<std::vector<float>>* gpu_shortlist_distances,
                                                const HighfreqOriginalRerankGroup& group,
                                                uint32_t shortlist_k,
                                                uint32_t topk,
                                                uint32_t rerank_batch_size,
                                                float rerank_epsilon,
                                                uint32_t rerank_stable_batches,
                                                bool allow_early_stop,
                                                bool filter_full_labels,
                                                std::vector<std::vector<idx_t>>& out_results,
                                                std::vector<std::vector<float>>* out_distances)
{
    if (group.route_positions.empty()) {
        return;
    }
    if (group.route_positions.size() == 1) {
        const size_t route_pos = group.route_positions.front();
        highfreq_rabitq_host_batch_rerank_range(reuse, runtime,
                                                query_indices,
                                                gpu_shortlist,
                                                shortlist_k,
                                                topk,
                                                rerank_batch_size,
                                                rerank_epsilon,
                                                rerank_stable_batches,
                                                allow_early_stop,
                                                filter_full_labels,
                                                route_pos,
                                                route_pos + 1,
                                                out_results,
                                                out_distances);
        return;
    }
    cpu_profile::Task profile("highfreq_merged");
    profile.query();
    const uint32_t representative_route = query_indices[group.route_positions.front()];
    const uint8_t* query_u8 = nullptr;
    PreparedFloatL2Query query_float;
    const PreparedFloatL2Query* prepared_query_float = nullptr;
    if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
        query_u8 = runtime.route_query_u8(representative_route);
    } else if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        query_float = prepare_base_float_vector_l2_query(
            runtime,
            runtime.route_query_float(representative_route),
            runtime.query_float_vectors.cols);
        prepared_query_float = &query_float;
    } else {
        fail_rabitq("unsupported dense vector scalar type for merged high-frequency CPU rerank");
    }

    profile.stage(cpu_profile::Stage::CandidateCollect);
    std::vector<HighfreqMergedCandidate> candidates;
    std::unordered_map<uint32_t, size_t> candidate_by_sample;
    size_t candidate_capacity = 0;
    for (size_t pos : group.route_positions) {
        if (pos >= gpu_shortlist.size()) {
            continue;
        }
        candidate_capacity += std::min<size_t>(gpu_shortlist[pos].size(), shortlist_k);
    }
    profile.candidates(candidate_capacity);
    candidates.reserve(candidate_capacity);
    candidate_by_sample.reserve(candidate_capacity);

    for (size_t pos : group.route_positions) {
        if (pos >= gpu_shortlist.size()) {
            continue;
        }
        const uint32_t route_idx = query_indices[pos];
        const uint32_t effective_shortlist =
            static_cast<uint32_t>(std::min<size_t>(gpu_shortlist[pos].size(), shortlist_k));
        for (uint32_t idx = 0; idx < effective_shortlist; ++idx) {
            const idx_t sample_id = gpu_shortlist[pos][idx];
            if (sample_id == UINT32_MAX || sample_id >= runtime.base_vectors.rows) {
                continue;
            }
            const uint32_t sample_id_u32 = static_cast<uint32_t>(sample_id);
            float score = FLT_MAX;
            if (gpu_shortlist_distances != nullptr &&
                pos < gpu_shortlist_distances->size() &&
                idx < (*gpu_shortlist_distances)[pos].size()) {
                score = (*gpu_shortlist_distances)[pos][idx];
                if (!std::isfinite(score)) {
                    score = FLT_MAX;
                }
            }
            // Preserve route-level full-filter semantics: a sample is valid if it came
            // from at least one source route whose residual label filter accepts it.
            if (filter_full_labels &&
                !profile.sample(cpu_profile::Detail::Predicate, [&] {
                        return sample_matches_lowfreq_full_labels(runtime, route_idx, sample_id_u32);
                    })) {
                continue;
            }
            auto [it, inserted] = profile.sample(cpu_profile::Detail::HashLookup, [&] {
                return candidate_by_sample.emplace(sample_id_u32, candidates.size());
            });
            if (inserted) {
                candidates.push_back({sample_id_u32, idx, score});
            } else {
                HighfreqMergedCandidate& candidate = candidates[it->second];
                candidate.best_rank = std::min(candidate.best_rank, idx);
                candidate.best_score = std::min(candidate.best_score, score);
            }
        }
    }
    profile.stage(cpu_profile::Stage::CandidateRank);
    std::sort(candidates.begin(),
              candidates.end(),
              [](const HighfreqMergedCandidate& lhs, const HighfreqMergedCandidate& rhs) {
                  if (lhs.best_score != rhs.best_score) {
                      return lhs.best_score < rhs.best_score;
                  }
                  if (lhs.best_rank != rhs.best_rank) {
                      return lhs.best_rank < rhs.best_rank;
                  }
                  return lhs.sample_id < rhs.sample_id;
              });
    if (candidates.size() > shortlist_k) {
        candidates.resize(shortlist_k);
    }
    std::vector<uint32_t> candidate_ids;
    candidate_ids.reserve(candidates.size());
    for (const HighfreqMergedCandidate& candidate : candidates) {
        candidate_ids.push_back(candidate.sample_id);
    }

    profile.stage(cpu_profile::Stage::Setup);
    std::vector<std::pair<float, idx_t>> heap;
    heap.reserve(topk);
    std::vector<idx_t> prev_topk(topk, UINT32_MAX);
    int stable_count = 0;
    const bool group_allow_early_stop =
        allow_early_stop && group.route_positions.size() == 1;
    std::vector<uint32_t> batch_sample_ids;
    batch_sample_ids.reserve(rerank_batch_size);

    for (uint32_t batch_begin = 0; batch_begin < candidate_ids.size(); batch_begin += rerank_batch_size) {
        const uint32_t batch_end =
            std::min<uint32_t>(static_cast<uint32_t>(candidate_ids.size()),
                               batch_begin + rerank_batch_size);
        profile.stage(cpu_profile::Stage::CandidateCollect);
        batch_sample_ids.clear();
        batch_sample_ids.insert(batch_sample_ids.end(),
                                candidate_ids.begin() + batch_begin,
                                candidate_ids.begin() + batch_end);
        std::sort(batch_sample_ids.begin(), batch_sample_ids.end());
        batch_sample_ids.erase(std::unique(batch_sample_ids.begin(), batch_sample_ids.end()),
                               batch_sample_ids.end());
        profile.stage(cpu_profile::Stage::VectorPrepare);

        evaluate_rerank_batch(reuse.distances, exact_original_query(runtime, representative_route), profile, runtime, prepared_query_float, batch_sample_ids, [&](uint32_t sample_id) {
            const float dist = highfreq_exact_l2_for_sample(runtime,
                                                            query_u8,
                                                            prepared_query_float,
                                                            sample_id);
            return dist;
        }, [&](uint32_t sample_id, float dist) {
            if (heap.size() < topk) {
                heap.emplace_back(dist, sample_id);
                std::push_heap(heap.begin(), heap.end());
            } else if (dist < heap.front().first ||
                       (dist == heap.front().first && sample_id < heap.front().second)) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = {dist, sample_id};
                std::push_heap(heap.begin(), heap.end());
            }
        });
        profile.stage(cpu_profile::Stage::EarlyStop);

        if (group_allow_early_stop && heap.size() >= topk) {
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
                if (rerank_stable_batches != UINT32_MAX &&
                    static_cast<uint32_t>(stable_count) >= rerank_stable_batches) {
                    prev_topk = std::move(curr_topk);
                    break;
                }
            } else {
                stable_count = 0;
            }
            prev_topk = std::move(curr_topk);
        }
    }

    profile.stage(cpu_profile::Stage::Writeback);
    std::vector<idx_t> group_result(topk, UINT32_MAX);
    std::vector<float> group_distances(topk, FLT_MAX);
    auto sorted = heap;
    std::sort_heap(sorted.begin(), sorted.end());
    for (size_t i = 0; i < sorted.size() && i < topk; ++i) {
        group_result[i] = sorted[i].second;
        group_distances[i] = sorted[i].first;
    }
    for (size_t pos : group.route_positions) {
        out_results[pos] = group_result;
        if (out_distances != nullptr) {
            (*out_distances)[pos] = group_distances;
        }
    }
}

void highfreq_rabitq_host_merged_rerank_range(const CpuRerankReuse& reuse, const GraphRuntime& runtime,
                                              const std::vector<uint32_t>& query_indices,
                                              const std::vector<std::vector<idx_t>>& gpu_shortlist,
                                              const std::vector<std::vector<float>>* gpu_shortlist_distances,
                                              const std::vector<HighfreqOriginalRerankGroup>& groups,
                                              uint32_t shortlist_k,
                                              uint32_t topk,
                                              uint32_t rerank_batch_size,
                                              float rerank_epsilon,
                                              uint32_t rerank_stable_batches,
                                              bool allow_early_stop,
                                              bool filter_full_labels,
                                              size_t begin_group,
                                              size_t end_group,
                                              std::vector<std::vector<idx_t>>& out_results,
                                              std::vector<std::vector<float>>* out_distances)
{
    rerank_batch_size = std::max<uint32_t>(1U, rerank_batch_size);
    rerank_stable_batches = std::max<uint32_t>(1U, rerank_stable_batches);
    for (size_t group_idx = begin_group; group_idx < end_group; ++group_idx) {
        highfreq_rabitq_host_original_rerank_group(reuse, runtime,
                                                   query_indices,
                                                   gpu_shortlist,
                                                   gpu_shortlist_distances,
                                                   groups[group_idx],
                                                   shortlist_k,
                                                   topk,
                                                   rerank_batch_size,
                                                   rerank_epsilon,
                                                   rerank_stable_batches,
                                                   allow_early_stop,
                                                   filter_full_labels,
                                                   out_results,
                                                   out_distances);
    }
}

template <typename Fn>
void dispatch_cpu_rerank_partitions(size_t worker_count,
                                    RabitqCpuWorkerPool* cpu_rerank_workers,
                                    cpu_profile::Task& profile,
                                    Fn&& work)
{
    if (cpu_rerank_workers != nullptr) {
        profile.stage(cpu_profile::Stage::ThreadLaunch);
        auto batch = cpu_rerank_workers->submit(worker_count, std::forward<Fn>(work));
        profile.stage(cpu_profile::Stage::ThreadJoin);
        batch->wait();
        return;
    }
    std::exception_ptr worker_error;
    std::mutex worker_error_mutex;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    try {
        for (size_t worker = 0; worker < worker_count; ++worker) {
            profile.stage(cpu_profile::Stage::ThreadLaunch);
            workers.emplace_back([&, worker] {
                try {
                    work(worker);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(worker_error_mutex);
                    if (!worker_error) worker_error = std::current_exception();
                }
            });
        }
    } catch (...) {
        for (auto& worker : workers) worker.join();
        throw;
    }
    for (auto& worker : workers) {
        profile.stage(cpu_profile::Stage::ThreadJoin);
        worker.join();
    }
    if (worker_error) std::rethrow_exception(worker_error);
}

struct IncrementalGroupScratch {
    std::vector<size_t> slot_by_query;
    std::vector<lowfreq_merge::RouteGroup> groups;
    size_t active = 0;
    void build(const GraphRuntime& runtime, const std::vector<uint32_t>& routes) {
        for (size_t i = 0; i < active; ++i) slot_by_query[groups[i].original_query] = SIZE_MAX;
        active = 0;
        if (slot_by_query.size() < runtime.original_query_count)
            slot_by_query.resize(runtime.original_query_count, SIZE_MAX);
        for (size_t pos = 0; pos < routes.size(); ++pos) {
            const uint32_t query = highfreq_route_original_query_id(runtime, routes[pos]);
            if (query >= runtime.original_query_count) fail_rabitq("incremental query group out of range");
            auto& slot = slot_by_query[query];
            if (slot == SIZE_MAX) {
                slot = active++;
                if (groups.size() < active) groups.emplace_back();
                groups[slot].original_query = query;
                groups[slot].route_positions.clear();
            }
            groups[slot].route_positions.push_back(pos);
        }
    }
};

// Every complete route batch enters the same original-query state. Group only
// the routes already ready in this batch; never wait for remaining producers.
void incremental_rerank_batch(const CpuRerankReuse& reuse,
    const GraphRuntime& runtime, const RabitqSearchOptions& options,
    const std::vector<uint32_t>& routes, const std::vector<idx_t>& ids,
    const std::vector<float>& distances, uint32_t width, bool graph_verified,
    bool posting, std::vector<std::vector<idx_t>>& output,
    std::vector<std::vector<float>>* output_distances, std::vector<int>* steps,
    RabitqCpuWorkerBudget* budget, RabitqCpuWorkerPool* pool)
{
    if (!reuse.incremental) fail_rabitq("incremental rerank state missing");
    output.assign(routes.size(), {});
    if (output_distances) output_distances->assign(routes.size(), {});
    if (steps) steps->assign(routes.size(), 0);
    cpu_profile::Task coordinator("incremental_dispatch", "coordinator");
    coordinator.stage(cpu_profile::Stage::CandidateCollect);
    thread_local IncrementalGroupScratch grouping;
    grouping.build(runtime, routes);
    const auto& ready_groups = grouping.groups;
    const size_t group_count = grouping.active;
    if (!group_count) return;
    const size_t workers = std::min<size_t>(options.highfreq_rerank_threads, group_count);
    dispatch_cpu_rerank_partitions(workers, pool, coordinator, [&](size_t worker) {
        run_with_rabitq_cpu_worker_permit(budget, [&] {
            cpu_profile::Task profile(posting ? "incremental_posting" : "incremental_graph");

            std::vector<uint32_t> pending;
            std::vector<float> scores;
            std::vector<go_fann::incremental_rerank::ScoredId> incoming;
            std::vector<lowfreq_merge::Candidate> candidates;
            pending.reserve(reuse.incremental->cap);
            scores.reserve(reuse.incremental->cap);
            incoming.reserve(reuse.incremental->cap);
            candidates.reserve(reuse.incremental->cap);
            for (size_t i = group_count * worker / workers;
                 i < group_count * (worker + 1) / workers; ++i) {
                const auto& group = ready_groups[i];
                profile.query();
                profile.stage(cpu_profile::Stage::CandidateCollect);
                lowfreq_merge::Diagnostics diagnostics;
                const auto& merge_group = group;
                lowfreq_merge::merge_sorted_group_candidates_into(
                    merge_group, routes, ids, distances, width, runtime.base_vectors.rows,
                    static_cast<uint32_t>(reuse.incremental->cap),
                    [&](uint32_t route, uint32_t sample) {
                        if (posting ? reuse.posting_verified(route) : graph_verified) return true;
                        if (!sample_matches_lowfreq_full_labels(runtime, route, sample)) return false;
                        return &runtime == &reuse.incremental->primary ||
                            sample_matches_range_graph_clause_labels(reuse.incremental->primary,
                                group.original_query, sample);
                    }, diagnostics, candidates);
                profile.candidates(diagnostics.merge_visited_slots);
                incoming.clear();
                // Selector output is sorted and ID-unique; nonfinite input
                // scores have already been normalized to finite FLT_MAX.
                for (const auto& c : candidates)
                    incoming.emplace_back(c.approximate_distance, c.sample_id);
                profile.stage(cpu_profile::Stage::QueryLockWait);
                bool owns_scoring = false;
                reuse.incremental->with_query(group.original_query, [&](auto& state) {
                    profile.stage(cpu_profile::Stage::IncrementalAdmit);
                    state.offer_sorted_unique(incoming, reuse.incremental->cap);
                    owns_scoring = state.acquire(pending);
                });
                if (!owns_scoring) continue;
                profile.stage(cpu_profile::Stage::QueryPrepare);
                const auto& primary = reuse.incremental->primary;
                const uint8_t* query_u8 = nullptr;
                const PreparedFloatL2Query* prepared = nullptr;
                if (primary.vector_scalar_type == DenseVectorScalarType::Uint8) {
                    query_u8 = primary.original_query_vectors.row_ptr(group.original_query);
                } else {
                    auto& descriptor = reuse.incremental->prepared_queries[group.original_query];
                    if (!descriptor.values) descriptor = prepare_base_float_vector_l2_query(primary,
                        primary.original_query_float_vectors.row_ptr(group.original_query),
                        primary.original_query_float_vectors.cols);
                    prepared = &descriptor;
                }
                do {
                    profile.stage(cpu_profile::Stage::VectorPrepare);
                    std::sort(pending.begin(), pending.end());

                    profile.evaluate_into(pending, scores, [&](const uint32_t* rows, size_t count, float* values) {
                        if (prepared && primary.rerank_vector_store && primary.rerank_vector_store->uses_batched_l2())
                            primary.rerank_vector_store->l2_float_batch(rows, count, *prepared, values);
                        else for (size_t row = 0; row < count; ++row)
                            values[row] = highfreq_exact_l2_for_sample(primary, query_u8, prepared, rows[row]);
                    });
                    profile.stage(cpu_profile::Stage::QueryLockWait);
                    reuse.incremental->with_query(group.original_query, [&](auto& state) {
                        profile.stage(cpu_profile::Stage::IncrementalCommit);
                        for (size_t row = 0; row < pending.size(); ++row)
                            state.complete(pending[row], scores[row], reuse.incremental->topk);
                        owns_scoring = state.next(pending);
                    });
                } while (owns_scoring);
                if (steps) for (size_t position : group.route_positions) (*steps)[position] = 1;
            }
        });
    });
}

void highfreq_rabitq_host_rerank(const CpuRerankReuse& reuse, const RabitqSearchOptions& options,
                                 const GraphRuntime& runtime,
                                 const std::vector<uint32_t>& query_indices,
                                 const std::vector<std::vector<idx_t>>& gpu_shortlist,
                                 const std::vector<std::vector<float>>* gpu_shortlist_distances,
                                 std::vector<std::vector<idx_t>>& out_results,
                                 std::vector<std::vector<float>>* out_distances,
                                 RabitqCpuWorkerBudget* cpu_worker_budget = nullptr,
                                 RabitqCpuWorkerPool* cpu_rerank_workers = nullptr,
                                 bool shortlist_predicate_verified = false)
{
    cpu_profile::Task profile("highfreq_dispatch", "coordinator");

    perf_trace::Scope trace_scope("highfreq_cpu_rerank_batch", &runtime);
    if (options.graph_persistent_workers && cpu_rerank_workers == nullptr) {
        fail_rabitq("graph persistent workers were not prepared before execution");
    }
    if (!options.graph_persistent_workers) cpu_rerank_workers = nullptr;
    if (reuse.incremental) {
        if (!gpu_shortlist_distances || gpu_shortlist.size() != query_indices.size() ||
            gpu_shortlist_distances->size() != query_indices.size())
            fail_rabitq("incremental graph rerank needs raw approximate distances for every route");
        const uint32_t width = effective_highfreq_rerank_shortlist(options);
        std::vector<idx_t> ids(query_indices.size() * width, UINT32_MAX);
        std::vector<float> scores(ids.size(), FLT_MAX);
        for (size_t row = 0; row < query_indices.size(); ++row) {
            if (gpu_shortlist[row].size() != (*gpu_shortlist_distances)[row].size())
                fail_rabitq("incremental graph shortlist ID/score widths differ");
            const size_t n = std::min<size_t>(width, gpu_shortlist[row].size());
            std::copy_n(gpu_shortlist[row].data(), n, ids.data() + row * width);
            std::copy_n((*gpu_shortlist_distances)[row].data(), n, scores.data() + row * width);
        }
        incremental_rerank_batch(reuse, runtime, options, query_indices, ids, scores, width,
            shortlist_predicate_verified, false, out_results, out_distances, nullptr,
            cpu_worker_budget, cpu_rerank_workers);
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
    const bool filter_full_labels = options.full_filter && !shortlist_predicate_verified;

    out_results.assign(query_indices.size(), std::vector<idx_t>(topk, UINT32_MAX));
    if (out_distances != nullptr) {
        out_distances->assign(query_indices.size(), std::vector<float>(topk, FLT_MAX));
    }
    if (query_indices.empty()) {
        return;
    }

    const std::vector<HighfreqOriginalRerankGroup> groups =
        build_highfreq_original_rerank_groups(runtime, query_indices);
    const size_t requested_threads = static_cast<size_t>(std::max(1, options.highfreq_rerank_threads));
    const size_t worker_count = std::min(requested_threads, groups.size());
    if (worker_count == 0) {
        return;
    }

    if (worker_count <= 1) {
        profile.stage(cpu_profile::Stage::Delegated);
        run_with_rabitq_cpu_worker_permit(cpu_worker_budget, [&] {
            highfreq_rabitq_host_merged_rerank_range(reuse, runtime,
                                                     query_indices,
                                                     gpu_shortlist,
                                                     gpu_shortlist_distances,
                                                     groups,
                                                     shortlist_k,
                                                     topk,
                                                     batch_size,
                                                     epsilon,
                                                     stable_batches,
                                                     heuristic,
                                                     filter_full_labels,
                                                     0,
                                                     groups.size(),
                                                     out_results,
                                                     out_distances);
        });
        return;
    }

    dispatch_cpu_rerank_partitions(worker_count, cpu_rerank_workers, profile, [&](size_t worker) {
        const size_t begin = (groups.size() * worker) / worker_count;
        const size_t end = (groups.size() * (worker + 1)) / worker_count;
        run_with_rabitq_cpu_worker_permit(cpu_worker_budget, [&] {
            highfreq_rabitq_host_merged_rerank_range(reuse, runtime,
                                                     query_indices,
                                                     gpu_shortlist,
                                                     gpu_shortlist_distances,
                                                     groups,
                                                     shortlist_k,
                                                     topk,
                                                     batch_size,
                                                     epsilon,
                                                     stable_batches,
                                                     heuristic,
                                                     filter_full_labels,
                                                     begin,
                                                     end,
                                                     out_results,
                                                     out_distances);
        });
    });
}

void highfreq_rabitq_host_rerank(const CpuRerankReuse& reuse, const RabitqSearchOptions& options,
                                 const GraphRuntime& runtime,
                                 const std::vector<uint32_t>& query_indices,
                                 const std::vector<std::vector<idx_t>>& gpu_shortlist,
                                 std::vector<std::vector<idx_t>>& out_results,
                                 std::vector<std::vector<float>>* out_distances,
                                 RabitqCpuWorkerBudget* cpu_worker_budget = nullptr,
                                 RabitqCpuWorkerPool* cpu_rerank_workers = nullptr,
                                 bool shortlist_predicate_verified = false)
{
    highfreq_rabitq_host_rerank(reuse, options,
                                runtime,
                                query_indices,
                                gpu_shortlist,
                                nullptr,
                                out_results,
                                out_distances,
                                cpu_worker_budget, cpu_rerank_workers, shortlist_predicate_verified);
}

void finalize_highfreq_rabitq_results(const CpuRerankReuse& reuse, const RabitqSearchOptions& options,
                                      const GraphRuntime& runtime,
                                      const std::vector<uint32_t>& query_indices,
                                      std::vector<std::vector<idx_t>>& results,
                                      std::vector<std::vector<float>>* result_distances,
                                      RabitqCpuWorkerBudget* cpu_worker_budget = nullptr,
                                      RabitqCpuWorkerPool* cpu_rerank_workers = nullptr,
                                 bool shortlist_predicate_verified = false)
{
    if (highfreq_gpu_exact_rerank_enabled(options)) {
        return;
    }

    if (highfreq_cpu_rerank_enabled(options)) {
        std::vector<std::vector<idx_t>> reranked_results;
        std::vector<std::vector<float>> reranked_distances;
        highfreq_rabitq_host_rerank(reuse, options,
                                    runtime,
                                    query_indices,
                                    results,
                                    result_distances,
                                    reranked_results,
                                    result_distances != nullptr ? &reranked_distances : nullptr,
                                    cpu_worker_budget, cpu_rerank_workers, shortlist_predicate_verified);
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

void run_highfreq_rabitq_streaming_rerank(const CpuRerankReuse& reuse, const RabitqSearchOptions& options,
                                          const GraphRuntime& runtime,
                                          const std::vector<uint32_t>& query_ids,
                                          const int64_t* status_flag,
                                          const idx_t* h_result_mapped,
                                          const float* h_result_distances_mapped,
                                          const float* h_time_per_block_mapped,
                                          uint32_t result_width,
                                          std::vector<std::vector<idx_t>>& results,
                                          std::vector<int>& step_nums,
                                          std::vector<std::vector<float>>& result_distances,
                                          RabitqCpuWorkerBudget* cpu_worker_budget,
                                          RabitqCpuWorkerPool* cpu_rerank_workers,
                                          bool shortlist_predicate_verified = false)
{
    cpu_profile::Task profile("highfreq_streaming", "coordinator");
    perf_trace::Scope trace_scope("highfreq_streaming_lifetime", &runtime);
    if (query_ids.empty()) {
        results.clear();
        step_nums.clear();
        result_distances.clear();
        return;
    }

    results.assign(query_ids.size(), std::vector<idx_t>(static_cast<size_t>(options.topk), UINT32_MAX));
    step_nums.assign(query_ids.size(), 0);
    result_distances.assign(query_ids.size(), std::vector<float>(static_cast<size_t>(options.topk), FLT_MAX));

    const size_t batch_queries = static_cast<size_t>(std::max(1, options.highfreq_rerank_stream_batch_size));

    if (reuse.incremental || !options.highfreq_rerank_merge_routes) {
        std::vector<uint8_t> processed(query_ids.size(), 0);
        size_t processed_count = 0;
        std::vector<size_t> ready_positions;
        std::vector<uint32_t> batch_query_ids;
        std::vector<idx_t> flat_ids;
        std::vector<float> flat_scores;
        std::vector<std::vector<idx_t>> batch_results;
        std::vector<std::vector<float>> batch_distances;
        while (processed_count < query_ids.size()) {
            profile.stage(cpu_profile::Stage::ReadyWait);
            ready_positions.clear();
            ready_positions.reserve(batch_queries);
            for (size_t pos = 0; pos < query_ids.size() && ready_positions.size() < batch_queries; ++pos) {
                if (processed[pos] != 0) {
                    continue;
                }
                const uint32_t query_idx = query_ids[pos];
                if (!graph_status_flag_result_ready(load_status_flag_word(status_flag, query_idx))) {
                    continue;
                }
                ready_positions.push_back(pos);
            }
            if (ready_positions.empty()) {
                std::this_thread::sleep_for(std::chrono::microseconds(2));
                continue;
            }

            profile.stage(cpu_profile::Stage::Candidates);
            batch_query_ids.clear();
            std::vector<std::vector<idx_t>> batch_shortlists;
            if (reuse.incremental) {
                if (!h_result_distances_mapped) fail_rabitq("incremental graph scores not mapped");
                flat_ids.resize(ready_positions.size() * result_width);
                flat_scores.resize(flat_ids.size());
            }
            batch_query_ids.reserve(ready_positions.size());
            batch_shortlists.reserve(ready_positions.size());
            for (size_t pos : ready_positions) {
                const uint32_t query_idx = query_ids[pos];
                const size_t row_base = static_cast<size_t>(query_idx) * result_width;
                batch_query_ids.push_back(query_idx);
                if (reuse.incremental) {
                    const size_t target = (batch_query_ids.size() - 1) * result_width;
                    std::copy_n(h_result_mapped + row_base, result_width, flat_ids.data() + target);
                    std::copy_n(h_result_distances_mapped + row_base, result_width, flat_scores.data() + target);
                } else {
                    batch_shortlists.emplace_back(h_result_mapped + row_base,
                                                  h_result_mapped + row_base + result_width);
                }
            }
            profile.stage(cpu_profile::Stage::Delegated);
            if (reuse.incremental) {
                if (options.graph_persistent_workers && !cpu_rerank_workers)
                    fail_rabitq("graph persistent workers were not prepared before execution");

                perf_trace::Scope trace_scope("highfreq_cpu_rerank_batch", &runtime);
                incremental_rerank_batch(reuse, runtime, options, batch_query_ids, flat_ids, flat_scores,
                    result_width, shortlist_predicate_verified, false, batch_results, &batch_distances,
                    nullptr, cpu_worker_budget,
                    options.graph_persistent_workers ? cpu_rerank_workers : nullptr);
            } else {
                highfreq_rabitq_host_rerank(reuse, options, runtime, batch_query_ids, batch_shortlists,
                    nullptr, batch_results, &batch_distances,
                    cpu_worker_budget, cpu_rerank_workers, shortlist_predicate_verified);
            }
            profile.stage(cpu_profile::Stage::Writeback);
            for (size_t batch_idx = 0; batch_idx < ready_positions.size(); ++batch_idx) {
                const size_t pos = ready_positions[batch_idx];
                const uint32_t query_idx = query_ids[pos];
                results[pos] = std::move(batch_results[batch_idx]);
                result_distances[pos] = std::move(batch_distances[batch_idx]);
                if (h_time_per_block_mapped != nullptr) {
                    step_nums[pos] = static_cast<int>(h_time_per_block_mapped[6 * query_idx + 4]);
                }
                processed[pos] = 1;
                ++processed_count;
            }
        }
        return;
    }

    const std::vector<HighfreqOriginalRerankGroup> groups =
        build_highfreq_original_rerank_groups(runtime, query_ids);
    std::vector<size_t> route_to_group(query_ids.size(), SIZE_MAX);
    std::vector<size_t> group_remaining(groups.size(), 0);
    for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
        group_remaining[group_idx] = groups[group_idx].route_positions.size();
        for (size_t pos : groups[group_idx].route_positions) {
            if (pos < route_to_group.size()) {
                route_to_group[pos] = group_idx;
            }
        }
    }

    std::vector<uint8_t> route_ready(query_ids.size(), 0);
    std::vector<uint8_t> group_queued(groups.size(), 0);
    std::vector<std::vector<idx_t>> route_shortlists(query_ids.size());
    std::vector<std::vector<float>> route_shortlist_distances(query_ids.size());
    std::deque<size_t> ready_groups;
    size_t reranked_count = 0;
    while (reranked_count < query_ids.size()) {
        profile.stage(cpu_profile::Stage::ReadyWait);
        std::vector<size_t> ready_positions;
        ready_positions.reserve(batch_queries);
        for (size_t pos = 0; pos < query_ids.size() && ready_positions.size() < batch_queries; ++pos) {
            if (route_ready[pos] != 0) {
                continue;
            }
            const uint32_t query_idx = query_ids[pos];
            if (!graph_status_flag_result_ready(load_status_flag_word(status_flag, query_idx))) {
                continue;
            }
            ready_positions.push_back(pos);
        }
        if (ready_positions.empty() && ready_groups.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(2));
            continue;
        }

        profile.stage(cpu_profile::Stage::Candidates);
        for (size_t pos : ready_positions) {
            const uint32_t query_idx = query_ids[pos];
            const size_t row_base = static_cast<size_t>(query_idx) * result_width;
            route_shortlists[pos].assign(result_width, UINT32_MAX);
            for (uint32_t j = 0; j < result_width; ++j) {
                route_shortlists[pos][j] = h_result_mapped[row_base + j];
            }
            if (h_result_distances_mapped != nullptr) {
                route_shortlist_distances[pos].assign(result_width, FLT_MAX);
                for (uint32_t j = 0; j < result_width; ++j) {
                    route_shortlist_distances[pos][j] = h_result_distances_mapped[row_base + j];
                }
            }
            if (h_time_per_block_mapped != nullptr) {
                step_nums[pos] = static_cast<int>(h_time_per_block_mapped[6 * query_idx + 4]);
            }
            route_ready[pos] = 1;
            const size_t group_idx = route_to_group[pos];
            if (group_idx != SIZE_MAX && group_remaining[group_idx] > 0) {
                --group_remaining[group_idx];
                if (group_remaining[group_idx] == 0 && group_queued[group_idx] == 0) {
                    ready_groups.push_back(group_idx);
                    group_queued[group_idx] = 1;
                }
            }
        }
        if (ready_groups.empty()) {
            continue;
        }

        std::vector<size_t> batch_positions;
        batch_positions.reserve(batch_queries);
        while (!ready_groups.empty() &&
               (batch_positions.empty() || batch_positions.size() < batch_queries)) {
            const size_t group_idx = ready_groups.front();
            ready_groups.pop_front();
            const size_t group_size = groups[group_idx].route_positions.size();
            if (!batch_positions.empty() && batch_positions.size() + group_size > batch_queries) {
                ready_groups.push_front(group_idx);
                break;
            }
            batch_positions.insert(batch_positions.end(),
                                   groups[group_idx].route_positions.begin(),
                                   groups[group_idx].route_positions.end());
        }
        if (batch_positions.empty()) {
            continue;
        }

        std::vector<uint32_t> batch_query_ids;
        std::vector<std::vector<idx_t>> batch_shortlists;
        std::vector<std::vector<float>> batch_shortlist_distances;
        batch_query_ids.reserve(batch_positions.size());
        batch_shortlists.reserve(batch_positions.size());
        if (h_result_distances_mapped != nullptr) {
            batch_shortlist_distances.reserve(batch_positions.size());
        }
        for (size_t pos : batch_positions) {
            batch_query_ids.push_back(query_ids[pos]);
            batch_shortlists.push_back(std::move(route_shortlists[pos]));
            if (h_result_distances_mapped != nullptr) {
                batch_shortlist_distances.push_back(std::move(route_shortlist_distances[pos]));
            }
        }

        std::vector<std::vector<idx_t>> batch_results;
        std::vector<std::vector<float>> batch_distances;
        const std::vector<std::vector<float>>* batch_shortlist_distances_ptr =
            h_result_distances_mapped != nullptr ? &batch_shortlist_distances : nullptr;
        profile.stage(cpu_profile::Stage::Delegated);
        highfreq_rabitq_host_rerank(reuse, options,
                                    runtime,
                                    batch_query_ids,
                                    batch_shortlists,
                                    batch_shortlist_distances_ptr,
                                    batch_results,
                                    &batch_distances,
                                    cpu_worker_budget, cpu_rerank_workers, shortlist_predicate_verified);
        profile.stage(cpu_profile::Stage::Writeback);
        for (size_t batch_idx = 0; batch_idx < batch_positions.size(); ++batch_idx) {
            const size_t pos = batch_positions[batch_idx];
            results[pos] = std::move(batch_results[batch_idx]);
            result_distances[pos] = std::move(batch_distances[batch_idx]);
            ++reranked_count;
        }
    }
}

lowfreq_merge::FinalMergeStrategy merge_route_results_to_original(
    const GraphRuntime& runtime,
    const std::vector<std::vector<idx_t>>& route_results,
    const std::vector<std::vector<float>>& route_distances,
    const std::vector<int>& route_steps,
    uint32_t topk,
    std::vector<std::vector<idx_t>>& merged_results,
    std::vector<int>& merged_steps,
    std::vector<std::vector<float>>* merged_distances = nullptr)
{
    perf_trace::Scope trace_scope("final_route_merge", &runtime);
    return lowfreq_merge::merge_route_results_to_original(runtime.route_to_original_query,
                                                          runtime.original_query_count,
                                                          route_results,
                                                          route_distances,
                                                          route_steps,
                                                          topk,
                                                          merged_results,
                                                          merged_steps,
                                                          merged_distances);
}

template <typename T>
void read_exact(std::ifstream& input, T* values, size_t count, const std::string& label)
{
    static_assert(std::is_trivially_copyable_v<T>, "read_exact requires trivially copyable data");
    const size_t nbytes = count * sizeof(T);
    if (nbytes == 0) {
        return;
    }
    input.read(reinterpret_cast<char*>(values), static_cast<std::streamsize>(nbytes));
    if (!input) {
        fail_rabitq("failed to read " + label);
    }
}

template <typename T>
T read_value(std::ifstream& input, const std::string& label)
{
    T value{};
    read_exact(input, &value, 1, label);
    return value;
}

template <typename T>
std::vector<T> read_values(std::ifstream& input, size_t count, const std::string& label)
{
    std::vector<T> values(count);
    read_exact(input, values.data(), values.size(), label);
    return values;
}

template <typename T>
void copy_sample_major_payload(const std::vector<T>& src,
                               std::vector<T>& dst,
                               size_t src_row,
                               size_t dst_row,
                               size_t values_per_row)
{
    if (values_per_row == 0) {
        return;
    }
    const T* src_ptr = src.data() + src_row * values_per_row;
    T* dst_ptr = dst.data() + dst_row * values_per_row;
    std::copy(src_ptr, src_ptr + values_per_row, dst_ptr);
}

}

RabitqHostIndex import_cuvs_rabitq_index(const std::string& cuvs_artifact_path)
{
    std::ifstream input(cuvs_artifact_path, std::ios::binary);
    if (!input) {
        fail_rabitq("unable to open cuVS RabitQ artifact for reading: " + cuvs_artifact_path);
    }

    const size_t num_vectors = read_value<size_t>(input, "cuVS num_vectors");
    const size_t dim = read_value<size_t>(input, "cuVS num_dimensions");
    const size_t num_centroids = read_value<size_t>(input, "cuVS num_centroids");
    const size_t ex_bits = read_value<size_t>(input, "cuVS ex_bits");
    const bool batch_flag = read_value<bool>(input, "cuVS batch_flag");
    if (num_vectors == 0 || dim == 0 || num_centroids == 0) {
        fail_rabitq("cuVS artifact contains zero-sized metadata");
    }
    if (num_vectors > std::numeric_limits<uint32_t>::max() ||
        dim > std::numeric_limits<uint32_t>::max() ||
        num_centroids > std::numeric_limits<uint32_t>::max() ||
        ex_bits + 1 > kRabitqMaxBitsPerDim) {
        fail_rabitq("cuVS artifact metadata exceeds supported RabitQ limits");
    }
    if (!batch_flag) {
        fail_rabitq("cuVS RabitQ import requires a batch artifact saved with save_batch_flag=true");
    }

    const uint32_t original_dim = static_cast<uint32_t>(dim);
    const uint32_t bits_per_dim = static_cast<uint32_t>(ex_bits + 1);
    RabitqCodecSpec codec = make_codec_spec(original_dim, bits_per_dim);

    RabitqHostIndex index;
    index.header.version = kRabitqArtifactVersion;
    index.header.dim = original_dim;
    index.header.num_anchors = static_cast<uint32_t>(num_centroids);
    index.header.code_bytes = short_code_bytes_for_padded_dim(codec.padded_dim);
    index.header.num_points = num_vectors;
    index.codec = codec;
    index.query_scaling_factors = read_values<float>(input, 2, "cuVS query scaling factors");

    auto cluster_sizes = read_values<size_t>(input, num_centroids, "cuVS cluster sizes");
    const size_t cluster_total =
        std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), static_cast<size_t>(0));
    if (cluster_total != num_vectors) {
        fail_rabitq("cuVS cluster sizes do not sum to num_vectors");
    }

    index.rotation = read_values<float>(
        input, static_cast<size_t>(codec.padded_dim) * codec.padded_dim, "cuVS rotation");
    index.anchors = read_values<float>(
        input, static_cast<size_t>(index.header.num_anchors) * codec.padded_dim, "cuVS centroids");

    const size_t short_code_bytes = index.header.code_bytes;
    const size_t long_code_bytes = index.codec.long_code_bytes;
    const size_t short_factor_count = index.codec.short_factor_count;
    const size_t ex_factor_count = index.codec.ex_factor_count;
    const size_t point_count = static_cast<size_t>(index.header.num_points);

    auto cluster_codes =
        read_values<uint8_t>(input, point_count * short_code_bytes, "cuVS short codes");
    auto cluster_short_factors =
        read_values<float>(input, point_count * short_factor_count, "cuVS short factors");
    auto cluster_long_codes =
        read_values<uint8_t>(input, point_count * long_code_bytes, "cuVS long codes");
    auto cluster_ex_factors =
        read_values<float>(input, point_count * ex_factor_count, "cuVS ex factors");
    auto cluster_ids = read_values<uint32_t>(input, point_count, "cuVS vector ids");

    index.anchor_ids.assign(point_count, std::numeric_limits<uint32_t>::max());
    index.codes.assign(point_count * short_code_bytes, 0);
    index.long_codes.assign(point_count * long_code_bytes, 0);
    index.factors.assign(point_count * short_factor_count, 0.0f);
    index.ex_factors.assign(point_count * ex_factor_count, 0.0f);

    size_t cluster_offset = 0;
    for (uint32_t anchor = 0; anchor < index.header.num_anchors; ++anchor) {
        const size_t cluster_size = cluster_sizes[anchor];
        for (size_t local = 0; local < cluster_size; ++local) {
            const size_t cluster_row = cluster_offset + local;
            const uint32_t sample_id = cluster_ids[cluster_row];
            if (sample_id >= point_count) {
                fail_rabitq("cuVS artifact contains vector id outside num_vectors");
            }
            if (index.anchor_ids[sample_id] != std::numeric_limits<uint32_t>::max()) {
                fail_rabitq("cuVS artifact contains duplicate vector ids");
            }
            index.anchor_ids[sample_id] = anchor;
            copy_sample_major_payload(
                cluster_codes, index.codes, cluster_row, sample_id, short_code_bytes);
            copy_sample_major_payload(
                cluster_long_codes, index.long_codes, cluster_row, sample_id, long_code_bytes);
            copy_sample_major_payload(
                cluster_short_factors, index.factors, cluster_row, sample_id, short_factor_count);
            copy_sample_major_payload(
                cluster_ex_factors, index.ex_factors, cluster_row, sample_id, ex_factor_count);
        }
        cluster_offset += cluster_size;
    }

    if (std::any_of(index.anchor_ids.begin(), index.anchor_ids.end(), [](uint32_t anchor_id) {
            return anchor_id == std::numeric_limits<uint32_t>::max();
        })) {
        fail_rabitq("cuVS artifact did not provide all vector ids");
    }

    input.peek();
    if (input.bad()) {
        fail_rabitq("failed while checking cuVS artifact tail");
    }

    validate_header(index.header, index.codec);
    return index;
}

RabitqHostIndex build_rabitq_index(const DenseUint8Matrix& base_vectors,
                                   const RabitqBuildOptions& options)
{
    (void)base_vectors;
    (void)options;
    fail_rabitq("native RabitQ build was removed; build with cuVS and import with --rabitq-cuvs-artifact-path");
}

RabitqHostIndex build_rabitq_index_from_path(const std::string& base_vector_path,
                                             uint32_t row_limit,
                                             const RabitqBuildOptions& options)
{
    (void)base_vector_path;
    (void)row_limit;
    (void)options;
    fail_rabitq("native RabitQ build was removed; build with cuVS and import with --rabitq-cuvs-artifact-path");
}

void save_rabitq_index(const std::string& path, const RabitqHostIndex& index)
{
    validate_header(index.header, index.codec);
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        fail_rabitq("unable to open artifact for writing");
    }
    output.write(reinterpret_cast<const char*>(&index.header), sizeof(index.header));
    output.write(reinterpret_cast<const char*>(&index.codec), sizeof(index.codec));

    auto write_vector = [&](const auto& values) {
        uint64_t size = values.size();
        output.write(reinterpret_cast<const char*>(&size), sizeof(size));
        if (size > 0) {
            output.write(reinterpret_cast<const char*>(values.data()),
                         static_cast<std::streamsize>(size * sizeof(values[0])));
        }
    };

    write_vector(index.anchors);
    write_vector(index.rotation);
    write_vector(index.query_scaling_factors);
    write_vector(index.anchor_ids);
    write_vector(index.codes);
    write_vector(index.long_codes);
    write_vector(index.factors);
    write_vector(index.ex_factors);
    if (!output) {
        fail_rabitq("failed while writing artifact");
    }
}

RabitqHostIndex load_rabitq_index(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_rabitq("unable to open artifact for reading");
    }

    RabitqHostIndex index;
    input.read(reinterpret_cast<char*>(&index.header), sizeof(index.header));
    if (!input) {
        fail_rabitq("failed to read artifact header");
    }
    input.read(reinterpret_cast<char*>(&index.codec), sizeof(index.codec));
    if (!input) {
        fail_rabitq("failed to read artifact codec spec");
    }
    validate_header(index.header, index.codec);

    auto read_vector = [&](auto& values) {
        uint64_t size = 0;
        input.read(reinterpret_cast<char*>(&size), sizeof(size));
        values.resize(static_cast<size_t>(size));
        if (size > 0) {
            input.read(reinterpret_cast<char*>(values.data()),
                       static_cast<std::streamsize>(size * sizeof(values[0])));
        }
    };

    read_vector(index.anchors);
    read_vector(index.rotation);
    read_vector(index.query_scaling_factors);
    read_vector(index.anchor_ids);
    read_vector(index.codes);
    read_vector(index.long_codes);
    read_vector(index.factors);
    read_vector(index.ex_factors);
    if (!input) {
        fail_rabitq("failed to read artifact payload");
    }

    const size_t expected_anchor_values =
        static_cast<size_t>(index.header.num_anchors) * index.codec.padded_dim;
    const size_t expected_rotation_values =
        static_cast<size_t>(index.codec.padded_dim) * index.codec.padded_dim;
    const size_t expected_query_scaling_values = 2;
    const size_t expected_anchor_ids = static_cast<size_t>(index.header.num_points);
    const size_t expected_codes = expected_anchor_ids * index.header.code_bytes;
    const size_t expected_long_codes = expected_anchor_ids * index.codec.long_code_bytes;
    const size_t expected_factors = expected_anchor_ids * index.codec.short_factor_count;
    const size_t expected_ex_factors = expected_anchor_ids * index.codec.ex_factor_count;
    if (index.anchors.size() != expected_anchor_values ||
        index.rotation.size() != expected_rotation_values ||
        index.query_scaling_factors.size() != expected_query_scaling_values ||
        index.anchor_ids.size() != expected_anchor_ids ||
        index.codes.size() != expected_codes ||
        index.long_codes.size() != expected_long_codes ||
        index.factors.size() != expected_factors ||
        index.ex_factors.size() != expected_ex_factors) {
        fail_rabitq("artifact payload sizes do not match header");
    }
    if (std::any_of(index.anchor_ids.begin(), index.anchor_ids.end(), [&](uint32_t anchor_id) {
            return anchor_id >= index.header.num_anchors;
        })) {
        fail_rabitq("artifact anchor_ids contain an out-of-range anchor id");
    }

    return index;
}

RabitqQueryState prepare_rabitq_query(const uint8_t* query, uint32_t dim)
{
    RabitqQueryState state;
    state.values.resize(dim);
    for (uint32_t d = 0; d < dim; ++d) {
        state.values[d] = static_cast<float>(query[d]);
    }
    return state;
}

std::vector<float> decode_rabitq_point(const RabitqHostIndex& index, uint32_t point_id)
{
    (void)index;
    (void)point_id;
    fail_rabitq("decode_rabitq_point is not implemented for cuVS-imported RabitQ artifacts");
}

float score_rabitq_point(const RabitqHostIndex& index,
                         const RabitqQueryState& query,
                         uint32_t point_id)
{
    (void)index;
    (void)query;
    (void)point_id;
    fail_rabitq("score_rabitq_point is not implemented for cuVS-imported RabitQ artifacts");
}

__global__ void rotate_rabitq_queries_kernel(const float* query,
                                             const float* rotation,
                                             float* rotated_query,
                                             uint32_t query_count,
                                             uint32_t dim,
                                             uint32_t padded_dim)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(query_count) * padded_dim;
    if (idx >= total) {
        return;
    }
    const uint32_t query_idx = static_cast<uint32_t>(idx / padded_dim);
    const uint32_t dst_dim = static_cast<uint32_t>(idx - static_cast<size_t>(query_idx) * padded_dim);
    float acc = 0.0f;
    const float* query_row = query + static_cast<size_t>(query_idx) * dim;
    for (uint32_t src_dim = 0; src_dim < dim; ++src_dim) {
        acc += query_row[src_dim] * rotation[static_cast<size_t>(src_dim) * padded_dim + dst_dim];
    }
    rotated_query[idx] = acc;
}

__global__ void compute_rabitq_query_factors_kernel(const float* rotated_query,
                                                    float* q_k1xsumq,
                                                    float* q_kbxsumq,
                                                    uint32_t query_count,
                                                    uint32_t padded_dim,
                                                    uint32_t bits_per_dim)
{
    const uint32_t query_idx = blockIdx.x;
    if (query_idx >= query_count) {
        return;
    }
    float sum = 0.0f;
    const float* row = rotated_query + static_cast<size_t>(query_idx) * padded_dim;
    for (uint32_t d = threadIdx.x; d < padded_dim; d += blockDim.x) {
        sum += row[d];
    }
    __shared__ float partials[256];
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const uint32_t ex_bits = bits_per_dim - 1u;
        q_k1xsumq[query_idx] = partials[0] * -0.5f;
        q_kbxsumq[query_idx] =
            partials[0] * (-static_cast<float>((1u << (ex_bits + 1u)) - 1u) * 0.5f);
    }
}

__global__ void quantize_rabitq_queries_int8_kernel(const float* rotated_query,
                                                    int8_t* quantized_query,
                                                    float* quantized_query_scales,
                                                    uint32_t query_count,
                                                    uint32_t padded_dim)
{
    const uint32_t query_idx = blockIdx.x;
    if (query_idx >= query_count) {
        return;
    }
    const float* src = rotated_query + static_cast<size_t>(query_idx) * padded_dim;
    int8_t* dst = quantized_query + static_cast<size_t>(query_idx) * padded_dim;
    float max_abs = 0.0f;
    for (uint32_t d = threadIdx.x; d < padded_dim; d += blockDim.x) {
        max_abs = fmaxf(max_abs, fabsf(src[d]));
    }
    __shared__ float partials[256];
    partials[threadIdx.x] = max_abs;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] = fmaxf(partials[threadIdx.x], partials[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float scale = partials[0] > 0.0f ? partials[0] / 127.0f : 1.0f;
    if (threadIdx.x == 0) {
        quantized_query_scales[query_idx] = scale;
    }
    for (uint32_t d = threadIdx.x; d < padded_dim; d += blockDim.x) {
        const int value = max(-127, min(127, __float2int_rn(src[d] / scale)));
        dst[d] = static_cast<int8_t>(value);
    }
}

__global__ void compute_rabitq_centroid_distances_kernel(const float* rotated_query,
                                                         const float* anchors,
                                                         float* centroid_distances,
                                                         uint32_t query_count,
                                                         uint32_t num_anchors,
                                                         uint32_t padded_dim)
{
    const uint32_t pair_idx = blockIdx.x;
    const uint32_t total_pairs = query_count * num_anchors;
    if (pair_idx >= total_pairs) {
        return;
    }
    const uint32_t query_idx = pair_idx / num_anchors;
    const uint32_t anchor_idx = pair_idx - query_idx * num_anchors;
    const float* query_row = rotated_query + static_cast<size_t>(query_idx) * padded_dim;
    const float* anchor_row = anchors + static_cast<size_t>(anchor_idx) * padded_dim;
    float acc = 0.0f;
    for (uint32_t d = threadIdx.x; d < padded_dim; d += blockDim.x) {
        const float diff = query_row[d] - anchor_row[d];
        acc += diff * diff;
    }
    __shared__ float partials[256];
    partials[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        centroid_distances[pair_idx] = partials[0];
    }
}

__global__ void precompute_rabitq_short_lut_kernel(const float* rotated_query,
                                                   float* short_lut,
                                                   uint32_t query_count,
                                                   uint32_t padded_dim)
{
    const uint32_t query_idx = blockIdx.x;
    if (query_idx >= query_count) {
        return;
    }
    const uint32_t chunk_count = padded_dim / kRabitqLutChunkBits;
    const float* query = rotated_query + static_cast<size_t>(query_idx) * padded_dim;
    float* query_lut =
        short_lut + static_cast<size_t>(query_idx) * chunk_count * kRabitqLutEntries;
    for (uint32_t chunk = threadIdx.x; chunk < chunk_count; chunk += blockDim.x) {
        const uint32_t dim_base = chunk * kRabitqLutChunkBits;
        for (uint32_t pattern = 0; pattern < kRabitqLutEntries; ++pattern) {
            float sum = 0.0f;
            for (uint32_t bit = 0; bit < kRabitqLutChunkBits; ++bit) {
                const uint32_t mask = 1u << (kRabitqLutChunkBits - 1u - bit);
                if ((pattern & mask) != 0u) {
                    sum += query[dim_base + bit];
                }
            }
            query_lut[static_cast<size_t>(chunk) * kRabitqLutEntries + pattern] = sum;
        }
    }
}

__device__ __forceinline__ bool rabitq_short_bit_msb(const uint8_t* codes,
                                            uint32_t code_bytes,
                                            uint32_t point_id,
                                            uint32_t dim_idx)
{
    const uint32_t* words = reinterpret_cast<const uint32_t*>(
        codes + static_cast<size_t>(point_id) * code_bytes);
    const uint32_t word = words[dim_idx / 32u];
    const uint32_t bit = 31u - (dim_idx % 32u);
    return ((word >> bit) & 1u) != 0;
}

__device__ __forceinline__ uint32_t rabitq_short_nibble_msb(const uint8_t* codes,
                                                   uint32_t code_bytes,
                                                   uint32_t point_id,
                                                   uint32_t chunk_idx)
{
    const uint8_t* point_code = codes + static_cast<size_t>(point_id) * code_bytes;
    const uint32_t word_idx = chunk_idx / (32u / kRabitqLutChunkBits);
    const uint32_t chunk_in_word = chunk_idx % (32u / kRabitqLutChunkBits);
    const uint32_t byte_in_word = 3u - (chunk_in_word >> 1u);
    const uint8_t byte = point_code[word_idx * sizeof(uint32_t) + byte_in_word];
    return ((chunk_in_word & 1u) == 0u) ? static_cast<uint32_t>(byte >> 4u)
                                        : static_cast<uint32_t>(byte & 0x0Fu);
}

__device__ __forceinline__ uint32_t rabitq_extract_long_code_msb(const uint8_t* codes,
                                                       uint32_t dim_idx,
                                                       uint32_t ex_bits)
{
    const size_t bit_pos = static_cast<size_t>(dim_idx) * ex_bits;
    const size_t byte_idx = bit_pos >> 3;
    const uint32_t bit_offset = static_cast<uint32_t>(bit_pos & 7u);
    uint32_t value = static_cast<uint32_t>(codes[byte_idx]) << 8u;
    if (bit_offset + ex_bits > 8u) {
        value |= static_cast<uint32_t>(codes[byte_idx + 1]);
    }
    const uint32_t shift = 16u - (bit_offset + ex_bits);
    return (value >> shift) & ((1u << ex_bits) - 1u);
}

template <bool kReverseBytesWithinWord>
__device__ __forceinline__ int rabitq_dp4a_binary_dot_lane(const int* query_words,
                                                           const uint8_t* bit_codes,
                                                           uint32_t padded_dim,
                                                           int lane)
{
    const uint32_t byte_chunk_count = padded_dim / 8u;
    int partial = 0;
    for (uint32_t byte_chunk = static_cast<uint32_t>(lane);
         byte_chunk < byte_chunk_count;
         byte_chunk += warp_kernel::kWarpThreads) {
        uint8_t patterns = 0;
        if constexpr (kReverseBytesWithinWord) {
            const uint32_t word_idx = byte_chunk >> 2u;
            const uint32_t byte_in_word = 3u - (byte_chunk & 3u);
            patterns = bit_codes[word_idx * sizeof(uint32_t) + byte_in_word];
        } else {
            patterns = bit_codes[byte_chunk];
        }
        const uint32_t query_word = byte_chunk * 2u;
        partial = __dp4a(query_words[query_word],
                         static_cast<int>(kRabitqDp4aNibbleMasks[patterns >> 4u]),
                         partial);
        partial = __dp4a(query_words[query_word + 1u],
                         static_cast<int>(kRabitqDp4aNibbleMasks[patterns & 0x0Fu]),
                         partial);
    }
    return partial;
}

template <bool kUseInt8Score = false>
__device__ __forceinline__ float score_rabitq_code_device(const uint32_t* anchor_ids,
                                          const uint8_t* codes,
                                          const uint8_t* long_codes,
                                          const float* factors,
                                          const float* ex_factors,
                                          const float* rotated_query,
                                          const float* short_lut,
                                          const float* q_k1xsumq,
                                          const float* q_kbxsumq,
                                          const float* centroid_distances,
                                          uint32_t vector_query_idx,
                                          uint32_t point_id,
                                          uint32_t num_anchors,
                                          uint32_t padded_dim,
                                          uint32_t code_bytes,
                                          uint32_t long_code_bytes,
                                          uint32_t bits_per_dim,
                                          int subtid,
                                          const int8_t* quantized_query = nullptr,
                                          const float* quantized_query_scales = nullptr)
{
    const uint32_t anchor_id = anchor_ids[point_id];
    const float* query = rotated_query + static_cast<size_t>(vector_query_idx) * padded_dim;
    float short_ip = 0.0f;
    float long_ip = 0.0f;
    const uint32_t ex_bits = bits_per_dim - 1u;
    const uint8_t* point_long_code = long_codes + static_cast<size_t>(point_id) * long_code_bytes;
    if constexpr (kUseInt8Score) {
        const int8_t* query_int8 =
            quantized_query + static_cast<size_t>(vector_query_idx) * padded_dim;
        const int* query_words = reinterpret_cast<const int*>(query_int8);
        const uint8_t* point_code = codes + static_cast<size_t>(point_id) * code_bytes;
        int short_ip_int = rabitq_dp4a_binary_dot_lane<true>(
            query_words, point_code, padded_dim, subtid);
        int long_ip_int = ex_bits == 1u
            ? rabitq_dp4a_binary_dot_lane<false>(
                  query_words, point_long_code, padded_dim, subtid)
            : 0;
        if (ex_bits == 0u) {
            for (int offset = 16; offset > 0; offset /= 2) {
                short_ip_int += __shfl_xor_sync(
                    warp_kernel::kFullMask, short_ip_int, offset);
            }
        } else {
            for (int offset = 16; offset > 0; offset /= 2) {
                short_ip_int += __shfl_xor_sync(
                    warp_kernel::kFullMask, short_ip_int, offset);
                long_ip_int += __shfl_xor_sync(
                    warp_kernel::kFullMask, long_ip_int, offset);
            }
        }
        const float query_scale = quantized_query_scales[vector_query_idx];
        short_ip = static_cast<float>(short_ip_int) * query_scale;
        long_ip = static_cast<float>(long_ip_int) * query_scale;
    } else if (short_lut != nullptr) {
        const uint32_t chunk_count = padded_dim / kRabitqLutChunkBits;
        const float* query_lut =
            short_lut + static_cast<size_t>(vector_query_idx) * chunk_count * kRabitqLutEntries;
        for (uint32_t chunk = static_cast<uint32_t>(subtid); chunk < chunk_count;
             chunk += warp_kernel::kWarpThreads) {
            const uint32_t pattern = rabitq_short_nibble_msb(codes, code_bytes, point_id, chunk);
            short_ip += query_lut[static_cast<size_t>(chunk) * kRabitqLutEntries + pattern];
        }
    } else {
        for (uint32_t d = static_cast<uint32_t>(subtid); d < padded_dim; d += warp_kernel::kWarpThreads) {
            if (rabitq_short_bit_msb(codes, code_bytes, point_id, d)) {
                short_ip += query[d];
            }
        }
    }
    if constexpr (!kUseInt8Score) {
        if (ex_bits > 0) {
            for (uint32_t d = static_cast<uint32_t>(subtid);
                 d < padded_dim;
                 d += warp_kernel::kWarpThreads) {
                long_ip += query[d] * static_cast<float>(
                    rabitq_extract_long_code_msb(point_long_code, d, ex_bits));
            }
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            short_ip += __shfl_xor_sync(warp_kernel::kFullMask, short_ip, offset);
            long_ip += __shfl_xor_sync(warp_kernel::kFullMask, long_ip, offset);
        }
    }

    const float q_g_add =
        centroid_distances[static_cast<size_t>(vector_query_idx) * num_anchors + anchor_id];
    if (ex_bits == 0) {
        const float f_add = factors[static_cast<size_t>(point_id) * 3u];
        const float f_rescale = factors[static_cast<size_t>(point_id) * 3u + 1u];
        return f_add + q_g_add + f_rescale * (short_ip + q_k1xsumq[vector_query_idx]);
    }

    const float f_ex_add = ex_factors[static_cast<size_t>(point_id) * 2u];
    const float f_ex_rescale = ex_factors[static_cast<size_t>(point_id) * 2u + 1u];
    return f_ex_add + q_g_add +
           f_ex_rescale *
	               (static_cast<float>(1u << ex_bits) * short_ip + long_ip +
                    q_kbxsumq[vector_query_idx]);
}

__device__ __forceinline__ bool lowfreq_basic_filter_passes(uint64_t row_filter_mask,
                                                            uint64_t query_filter_bit,
                                                            const uint8_t* d_query_range_modes,
                                                            const float* d_query_range_lower_bounds,
                                                            const float* d_query_range_upper_bounds,
                                                            const float* d_base_range_values,
                                                            uint32_t query_idx,
                                                            uint32_t sample_id,
                                                            uint32_t range_count,
                                                            uint32_t active_range_count)
{
    if (query_filter_bit == 0 || (row_filter_mask & query_filter_bit) == 0ULL) {
        return false;
    }
    if (active_range_count == 0) {
        return true;
    }
    if (d_base_range_values == nullptr) {
        return false;
    }
    const uint32_t matched_active_dims =
        go_fann::range_filter::count_matching_active_ranges(d_query_range_modes,
                                                                  d_query_range_lower_bounds,
                                                                  d_query_range_upper_bounds,
                                                                  d_base_range_values,
                                                                  query_idx,
                                                                  sample_id,
                                                                  range_count);
    return matched_active_dims == active_range_count;
}

__device__ __forceinline__ uint64_t graph_row_filter_mask_or_all_ones(
    const uint64_t* d_row_filter_mask,
    uint32_t global_row)
{
    return d_row_filter_mask == nullptr
        ? UINT64_MAX
        : d_row_filter_mask[global_row];
}

__device__ __forceinline__ uint32_t rabitq_gpu_graph_cache_slot(uint32_t global_row,
                                                               uint32_t capacity)
{
    const uint32_t hash = global_row * 2654435761U;
    return static_cast<uint32_t>(
        (static_cast<unsigned long long>(hash) * capacity) >> 32);
}

__device__ __forceinline__ unsigned long long rabitq_gpu_graph_cache_load_acquire(
    unsigned long long* address)
{
    unsigned long long value = 0;
    const unsigned long long zero = 0;
    asm volatile("atom.acquire.gpu.global.add.u64 %0, [%1], %2;"
                 : "=l"(value)
                 : "l"(address), "l"(zero)
                 : "memory");
    return value;
}

__device__ __forceinline__ void rabitq_gpu_graph_cache_store_release_increment(
    unsigned long long* address)
{
    const unsigned long long increment = 1;
    asm volatile("atom.release.gpu.global.add.u64 _, [%0], %1;"
                 :
                 : "l"(address), "l"(increment)
                 : "memory");
}

__device__ __forceinline__ bool rabitq_gpu_graph_cache_load(
    const RabitqGpuGraphCacheDeviceView* view,
    uint32_t global_row,
    uint32_t lane,
    uint32_t* shared_neighbors,
    uint32_t& degree)
{
    if (view == nullptr || view->entries == nullptr || view->capacity == 0) {
        return false;
    }
    RabitqGpuGraphRowCacheEntry* const entries = view->entries;
    const uint32_t capacity = view->capacity;
    RabitqGpuGraphRowCacheEntry* entry =
        entries + rabitq_gpu_graph_cache_slot(global_row, capacity);
    unsigned long long sequence = 0;
    uint32_t cached_row = UINT32_MAX;
    uint32_t cached_degree = 0;
    if (lane == 0) {
        sequence = rabitq_gpu_graph_cache_load_acquire(&entry->sequence);
        cached_row = entry->global_row;
        cached_degree = entry->degree;
    }
    sequence = __shfl_sync(warp_kernel::kFullMask, sequence, 0);
    cached_row = __shfl_sync(warp_kernel::kFullMask, cached_row, 0);
    cached_degree = __shfl_sync(warp_kernel::kFullMask, cached_degree, 0);
    const bool candidate = sequence >= 2 && (sequence & 1ULL) == 0 &&
        cached_row == global_row && cached_degree <= kRabitqGpuGraphRowCacheDegree;
    if (candidate) {
        shared_neighbors[lane] = entry->neighbors[lane];
    }
    __syncwarp();
    bool stable = false;
    if (lane == 0 && candidate) {
        stable = rabitq_gpu_graph_cache_load_acquire(&entry->sequence) == sequence;
    }
    stable = __shfl_sync(warp_kernel::kFullMask, stable ? 1 : 0, 0) != 0;
    if (stable) {
        degree = cached_degree;
    }
    return stable;
}

__device__ __forceinline__ void rabitq_gpu_graph_cache_store(
    const RabitqGpuGraphCacheDeviceView* view,
    uint32_t global_row,
    uint32_t degree,
    const uint32_t* neighbors,
    uint32_t lane)
{
    if (view == nullptr || view->entries == nullptr || view->capacity == 0) {
        return;
    }
    RabitqGpuGraphRowCacheEntry* const entries = view->entries;
    const uint32_t capacity = view->capacity;
    RabitqGpuGraphRowCacheEntry* entry =
        entries + rabitq_gpu_graph_cache_slot(global_row, capacity);
    bool locked = false;
    if (lane == 0) {
        const unsigned long long sequence = atomicAdd(&entry->sequence, 0ULL);
        if ((sequence & 1ULL) == 0) {
            locked = atomicCAS(&entry->sequence, sequence, sequence + 1ULL) == sequence;
        }
    }
    locked = __shfl_sync(warp_kernel::kFullMask, locked ? 1 : 0, 0) != 0;
    if (!locked) {
        return;
    }
    entry->neighbors[lane] = neighbors[lane];
    if (lane == 0) {
        entry->global_row = global_row;
        entry->degree = min(degree, kRabitqGpuGraphRowCacheDegree);
    }
    __syncwarp();
    if (lane == 0) {
        rabitq_gpu_graph_cache_store_release_increment(&entry->sequence);
    }
}

__device__ __forceinline__ bool rabitq_sorted_labels_contain(
    const uint32_t* labels,
    uint32_t begin,
    uint32_t end,
    uint32_t needle)
{
    while (begin < end) {
        const uint32_t mid = begin + ((end - begin) >> 1);
        const uint32_t value = labels[mid];
        if (value == needle) return true;
        if (value < needle) begin = mid + 1;
        else end = mid;
    }
    return false;
}

__device__ __forceinline__ bool rabitq_gpu_full_labels_match(
    const RabitqGpuGraphCacheDeviceView* view,
    uint32_t query_idx,
    uint32_t sample_idx)
{
    if (view == nullptr) {
        return true;
    }
    const uint32_t* const query_label_indptr = view->query_label_indptr;
    const uint32_t* const query_label_indices = view->query_label_indices;
    const uint32_t* const sample_label_indptr = view->sample_label_indptr;
    const uint32_t* const sample_label_indices = view->sample_label_indices;
    if (query_label_indptr == nullptr || sample_label_indptr == nullptr) {
        return true;
    }
    const uint32_t query_begin = query_label_indptr[query_idx];
    const uint32_t query_end = query_label_indptr[query_idx + 1];
    const uint32_t sample_begin = sample_label_indptr[sample_idx];
    const uint32_t sample_end = sample_label_indptr[sample_idx + 1];
    for (uint32_t pos = query_begin; pos < query_end; ++pos) {
        if (!rabitq_sorted_labels_contain(sample_label_indices,
                                          sample_begin,
                                          sample_end,
                                          query_label_indices[pos])) {
            return false;
        }
    }
    return true;
}

__device__ inline bool predicate_match_observed(const gpu_predicate::View& view,
                                                uint32_t route, uint32_t sample,
                                                bool proposed)
{
    const uint64_t begin = view.counters ? warp_kernel::get_global_timer_ns() : 0;
    const bool pass = gpu_predicate::matches(view, route, sample);
    if (view.counters) {
        auto& c = view.counters[route];
        c.lookup_ns += warp_kernel::get_global_timer_ns() - begin;
        if (proposed) ++c.proposed; else { ++c.expanded; c.passed += pass; }
        if (c.trace_count < gpu_predicate::trace_capacity) c.trace[c.trace_count++] = {sample, uint32_t(pass)};
    }
    return pass;
}

template<class Pair> struct RabitqLane0SearchStateT {
    Pair now;
    Pair next;
    dist_t stop_topk_approx_worst;
    uint16_t topk_heap_size;
    uint16_t stop_topk_heap_size;
    uint8_t now_filter_result;
};

static_assert(kHighfreqRerankShortlistMax + 1 < (1 << 16),
              "RabitQ native heap sizes must fit in uint16_t");
using RabitqLane0SearchState = RabitqLane0SearchStateT<RabitqPoolPair>;
static_assert(sizeof(RabitqLane0SearchState) == 28,
              "RabitQ lane-0 state must remain compact");

enum class RabitqFiniteContextState : uint32_t {
    UNINITIALIZED = 0,
    WAIT_IO = 1,
    RUNNING = 2,
    DONE = 3,
};

enum class RabitqFiniteAdvanceResult : uint32_t {
    PENDING = 0,
    PARKED = 1,
    DONE = 2,
    REJECT_QUERY = 3,
    REJECT_EPOCH = 4,
    REJECT_STATE = 5,
    REJECT_GENERATION = 6,
    REJECT_ROW = 7,
    REJECT_BUFFER = 8,
    REJECT_STATUS = 9,
    REJECT_MAILBOX = 10,
};

constexpr uint32_t kRabitqStaticPrefetchStepLimit = 3;

// Control is route-addressable; mutable heaps are indexed by scratch_slot so
// label-expanded route ids do not inflate the high-frequency scratch domain.
struct alignas(16) RabitqFiniteContextControl {
    uint32_t query_id = UINT32_MAX;
    uint32_t context_epoch = 0;
    RabitqFiniteContextState state = RabitqFiniteContextState::UNINITIALIZED;
    uint32_t scratch_slot = UINT32_MAX;
    uint32_t io_generation = 0;
    uint32_t current_buffer = 0;
    uint32_t expected_row = UINT32_MAX;
    uint32_t frontier_size = 0;
    uint32_t stop_heap_size = 0;
    uint32_t result_heap_size = 0;
    uint32_t expanded_count = 0;
    uint32_t pass_count = 0;
    uint32_t step_num = 0;
    uint32_t finished = 0;
    uint32_t pool_exhausted = 0;
    uint32_t stop_worst_bits = 0;
    float now_distance = FLT_MAX;
    uint32_t now_payload = UINT32_MAX;
    float stop_worst = -FLT_MAX;
    uint64_t state_since_ns = 0;
};

struct RabitqFiniteReadyToken {
    uint32_t query_id = UINT32_MAX;
    uint32_t context_epoch = 0;
    uint32_t generation = 0;
    uint32_t row = UINT32_MAX;
    uint32_t buffer = 0;
    RabitqFiniteAdvanceResult result = RabitqFiniteAdvanceResult::PENDING;
    uint64_t ready_ns = 0;
};

static_assert(sizeof(RabitqFiniteReadyToken) == 32,
              "RabitQ finite ready token must remain a compact host/device record");

template <int kQueueSize,
          int kQueriesPerBlock,
          bool kCollectDetailedTiming,
          bool kCollectMailboxRtt,
          bool kUseRequestRing,
          bool kUseInt8Score,
          bool kDeviceResident,
          bool kUseCooperativePickup,
          bool kPredicateBias = false>
__device__ void run_rabitq_search_one_block(const float* d_anchors,
                                            const uint32_t* d_anchor_ids,
                                            const uint8_t* d_codes,
                                            const uint8_t* d_long_codes,
                                            const float* d_factors,
                                            const float* d_ex_factors,
                                            const float* d_rotated_query,
                                            const int8_t* d_quantized_query,
                                            const float* d_quantized_query_scales,
                                            const float* d_short_lut,
                                            const float* d_q_k1xsumq,
                                            const float* d_q_kbxsumq,
                                            const float* d_centroid_distances,
                                            uint32_t route_idx,
                                            uint32_t vector_query_idx,
                                            const value_t* d_query,
                                            const float* d_query_float,
                                            idx_t* d_result,
                                            dist_t* d_result_distances,
                                            dist_t* d_datadist_scratch,
                                            uint32_t* d_hash_scratch,
                                            float* time_per_block,
                                            uint16_t* d_phase_state,
                                            uint64_t* d_mailbox_rtt_hist,
                                            uint32_t* d_neighbor_memory,
                                            volatile int64_t* d_status_memory,
                                            volatile int64_t* d_gdr_data,
                                            volatile int64_t* d_status_flag,
                                            HostRequestRingEntry* d_request_ring_entries,
                                            const volatile uint64_t* d_request_ring_consumed_sequences,
                                            unsigned long long* d_request_ring_tails,
                                            uint32_t request_ring_idx,
                                            uint32_t request_ring_capacity,
                                            const Node* d_prefetch_node_array,
                                            const uint32_t* d_prefetch_index,
                                            const uint32_t* d_prefetch_begin_by_group,
                                            const uint32_t* d_prefetch_end_by_group,
                                            const uint32_t* d_query_group_ids,
                                            const uint32_t* d_start_local_ids,
                                            const uint32_t* d_row_to_sample,
                                            const uint32_t* d_label_offset,
                                            const uint64_t* d_row_filter_mask,
                                            const uint64_t* d_query_filter_bits,
                                            const RabitqGpuGraphCacheDeviceView* d_gpu_graph_cache,
                                            gpu_predicate::View predicate,
                                            const uint8_t* d_query_range_modes,
                                            const float* d_query_range_lower_bounds,
                                            const float* d_query_range_upper_bounds,
                                            const float* d_base_range_values,
                                            uint32_t range_count,
                                            int dim,
                                            uint32_t padded_dim,
                                            uint32_t num_anchors,
                                            uint32_t code_bytes,
                                            uint32_t long_code_bytes,
                                            uint32_t bits_per_dim,
                                            int use_adaptive_wait,
                                            int queue_limit,
                                            int finish_limit,
                                            int early_stop_mode,
                                            int max_steps,
                                            int stop_width,
                                            int result_width,
                                            RabitqResultPair* topk,
                                            uint32_t mailbox_vector_words,
                                            uint32_t mailbox_slot_words,
                                            float bit_filter_bonus,
                                            float range_filter_bonus,
                                            int enable_rerank,
                                            int rerank_payload_float,
                                            expansion_trace::View expansion = {})
{
    // The current kernel keeps distance candidates in shared memory, but the
    // reserved global distance scratch remains graph-slot scoped.
    (void)d_datadist_scratch;
    constexpr int step = warp_kernel::kWarpThreads;
    int tid = threadIdx.x;
    int cid = tid / kCriticalStep;
    int subtid = tid % kCriticalStep;
    const uint32_t group_id = d_query_group_ids[route_idx];
    const uint32_t label_offset = d_label_offset[group_id];
    const uint64_t query_filter_bit = d_query_filter_bits[route_idx];
    const uint32_t active_range_count = go_fann::range_filter::count_active_ranges(
        d_query_range_modes, route_idx, range_count);
    const uint32_t begin_id = d_start_local_ids[route_idx];
    const uint32_t prefetch_begin = d_prefetch_begin_by_group[group_id];
    const uint32_t prefetch_end = d_prefetch_end_by_group[group_id];
    const uint32_t prefetch_len = prefetch_end - prefetch_begin;

    float time1 = 0.0f;
    float time2 = 0.0f;
    float time3 = 0.0f;
    float time4 = 0.0f;
    const uint64_t start1 = warp_kernel::get_global_timer_ns();
    set_rabitq_warp_phase(
        d_phase_state, route_idx, subtid, RabitqWarpPhase::Frontier);

    using FrontierPair = std::conditional_t<kPredicateBias, gpu_predicate::BiasedPair, RabitqPoolPair>;
    __shared__ FrontierPair q[kQueriesPerBlock][kQueueSize + 2];
    __shared__ RabitqStopPair stop_topk[kQueriesPerBlock][kTopk + 1];
    __shared__ dist_t dist_list[kQueriesPerBlock][FIXED_DEGREE * kMultiprobe];
    __shared__ int heap_size[kQueriesPerBlock];
    __shared__ int finished[kQueriesPerBlock];
    __shared__ int pool_exhausted[kQueriesPerBlock];
    __shared__ int expanded_count[kQueriesPerBlock];
    __shared__ int pass_count[kQueriesPerBlock];
    __shared__ uint32_t index_list[kQueriesPerBlock][FIXED_DEGREE * kMultiprobe];
#if GO_FANN_GPU_PREDICATE_PARALLEL
    // One verdict bit per proposed candidate; no per-candidate register array.
    __shared__ uint32_t predicate_pass_masks[kQueriesPerBlock][(FIXED_DEGREE * kMultiprobe + 31) / 32];
#endif
    __shared__ int index_list_len[kQueriesPerBlock];
    __shared__ value_t query_point[kQueriesPerBlock][kGraphMaxDim];
    __shared__ uint32_t current_buffer_idx[kQueriesPerBlock];
    __shared__ uint32_t request_generation[kQueriesPerBlock];
    __shared__ int wait_sleep_us[kQueriesPerBlock];
    __shared__ RabitqLane0SearchStateT<FrontierPair> lane0_state[kQueriesPerBlock];
    __shared__ uint64_t mailbox_request_start_ns[kQueriesPerBlock];
    __shared__ uint64_t mailbox_rtt_sum_ns[kQueriesPerBlock];
    __shared__ uint32_t mailbox_rtt_hist[kQueriesPerBlock][kRabitqMailboxRttBinCount];

    if (subtid == 0) {
        if constexpr (kCollectMailboxRtt) {
            mailbox_request_start_ns[cid] = 0;
            mailbox_rtt_sum_ns[cid] = 0;
            for (size_t bin = 0; bin < kRabitqMailboxRttBinCount; ++bin) {
                mailbox_rtt_hist[cid][bin] = 0;
            }
        }
        d_status_memory[route_idx * 16] = begin_id;
        __threadfence_system();
        d_gdr_data[route_idx * kGdrMailboxStride] = 0;
    }
    __syncwarp();

    if (enable_rerank != 0 && rerank_payload_float == 0) {
        for (int i = subtid; i < dim; i += step) {
            query_point[cid][i] = d_query[static_cast<size_t>(vector_query_idx) * dim + i];
        }
    }

    const uint32_t begin_sample = d_row_to_sample[label_offset + begin_id];
    if (subtid == 0 && predicate.counters) ++predicate.counters[route_idx].scores;
    set_rabitq_warp_phase(d_phase_state, route_idx, subtid, RabitqWarpPhase::Score);
    float start_distance = score_rabitq_code_device<kUseInt8Score>(
        d_anchor_ids,
        d_codes,
        d_long_codes,
        d_factors,
        d_ex_factors,
        d_rotated_query,
        d_short_lut,
        d_q_k1xsumq,
        d_q_kbxsumq,
        d_centroid_distances,
        vector_query_idx,
        begin_sample,
        num_anchors,
        padded_dim,
        code_bytes,
        long_code_bytes,
        bits_per_dim,
        subtid,
        d_quantized_query,
        d_quantized_query_scales);
    float predicate_penalty = 0.0f;
    if constexpr (kPredicateBias) {
        const double penalty_wide = double(predicate.alpha) * fmax(double(fabsf(start_distance)), 1e-6);
        predicate_penalty = isfinite(penalty_wide) ? float(fmin(penalty_wide, double(FLT_MAX))) : 0.0f;
        if (subtid == 0 && predicate.counters && penalty_wide > FLT_MAX) ++predicate.counters[route_idx].saturated;
    }
    set_rabitq_warp_phase(
        d_phase_state, route_idx, subtid, RabitqWarpPhase::Frontier);
    if (subtid == 0) {
        heap_size[cid] = 1;
        finished[cid] = false;
        pool_exhausted[cid] = false;
        expanded_count[cid] = 0;
        pass_count[cid] = 0;
        current_buffer_idx[cid] = 0;
        request_generation[cid] = 0;
        wait_sleep_us[cid] = kInitialWaitSleepUs;
        lane0_state[cid].topk_heap_size = 0;
        lane0_state[cid].stop_topk_heap_size = 0;
        lane0_state[cid].stop_topk_approx_worst = -FLT_MAX;
        lane0_state[cid].now_filter_result = 1;
        FrontierPair kp;
        kp.first = start_distance;
        if constexpr (kPredicateBias) kp.base = start_distance;
        kp.second = pack_pool_payload(begin_id, false);
        smmh2::insert(q[cid], heap_size[cid], kp);
        native_visited_hash::add<kHashTableCapacity>(d_hash_scratch, begin_id);
    }
    __syncwarp();

    bool found_next = false;
    if constexpr (kUseCooperativePickup) {
        found_next = pickup_next_unused_from_native_frontier<true>(
            q[cid], heap_size[cid], lane0_state[cid].now, subtid);
    } else if (subtid == 0) {
        found_next = pickup_next_unused_from_pool(
            q[cid], heap_size[cid], lane0_state[cid].now);
    }
    if (subtid == 0) {
        if (!found_next) {
            pool_exhausted[cid] = true;
            lane0_state[cid].now.first = FLT_MAX;
            lane0_state[cid].now.second = pack_pool_payload(UINT32_MAX, true);
        }
    }
    __syncwarp();

    int step_num = 0;
    uint32_t graph_cache_hits = 0;
    while (step_num == 0 || heap_size[cid] > 1) {
        const uint32_t now_local_row = unpack_pool_local_row(lane0_state[cid].now.second);
        step_num++;
        index_list_len[cid] = 0;
        if (now_local_row == UINT32_MAX) {
            break;
        }

        if (step_num != 1) {
            bool found = false;
            if constexpr (kUseCooperativePickup) {
                found = pickup_next_unused_from_native_frontier<true>(
                    q[cid], heap_size[cid], lane0_state[cid].next, subtid);
            } else if (subtid == 0) {
                found = pickup_next_unused_from_pool(
                    q[cid], heap_size[cid], lane0_state[cid].next);
            }
            if (subtid == 0 && !found) {
                pool_exhausted[cid] = true;
                lane0_state[cid].next.first = FLT_MAX;
                lane0_state[cid].next.second = pack_pool_payload(UINT32_MAX, true);
            } else if (subtid == 0 && lane0_state[cid].stop_topk_heap_size == stop_width) {
                float pending_score = lane0_state[cid].next.first;
                if (predicate.pending_base_stop && gpu_predicate::enabled(predicate, route_idx)) {
                    // now and next are reserved, not processed: pickup has
                    // already marked their copies in q as used. Include both.
                    pending_score = gpu_predicate::minimum_pending_base(
                        q[cid], heap_size[cid], lane0_state[cid].now,
                        lane0_state[cid].next, true, found,
                        [] __device__ (uint32_t payload) { return !pool_payload_used(payload); });
                }
                if (early_stop_mode != kSearchEarlyStopOff &&
                    lane0_state[cid].stop_topk_approx_worst <= pending_score) {
                    ++finished[cid];
                }
            }
        }

        int prefetch_vector_index = 0;
        bool node_prefetched = false;
        if (step_num <= kRabitqStaticPrefetchStepLimit && prefetch_len > 0 &&
            (!predicate.route_prefetch_disabled || !predicate.route_prefetch_disabled[route_idx])) {
            uint32_t local_prefetch_index = warp_kernel::n_ary_scan_prefetch(
                d_prefetch_index + prefetch_begin,
                now_local_row,
                prefetch_len);
            if (local_prefetch_index != UINT32_MAX) {
                prefetch_vector_index = static_cast<int>(prefetch_begin + local_prefetch_index);
                node_prefetched = true;
            }
        }
        const uint32_t global_row = label_offset + now_local_row;
        bool node_resident = false;
        bool node_cached = false;
        uint32_t cached_node_degree = 0;
        if (!node_prefetched) {
            if constexpr (kDeviceResident) {
                cached_node_degree = d_gpu_graph_cache->resident_degree;
                if (static_cast<uint32_t>(subtid) < cached_node_degree) {
                    index_list[cid][subtid] = d_gpu_graph_cache->resident_graph[
                        static_cast<size_t>(global_row) * cached_node_degree +
                        static_cast<uint32_t>(subtid)];
                }
                node_resident = true;
            } else if (d_gpu_graph_cache != nullptr) {
                node_cached = rabitq_gpu_graph_cache_load(
                    d_gpu_graph_cache,
                    global_row,
                    static_cast<uint32_t>(subtid),
                    index_list[cid],
                    cached_node_degree);
                if (subtid == 0 && node_cached) {
                    ++graph_cache_hits;
                }
            }
        }
        const bool node_on_gpu = node_resident || node_cached;

        uint64_t io_start = 0;
        if constexpr (kCollectDetailedTiming) {
            io_start = warp_kernel::get_global_timer_ns();
        }
        if (!node_prefetched && !node_on_gpu) {
            set_rabitq_warp_phase(
                d_phase_state, route_idx, subtid, RabitqWarpPhase::Wait);
            const int sleep_us = wait_sleep_us[cid];
            int wait_loops = 0;
            if (subtid == 0) {
                if constexpr (kCollectMailboxRtt) {
                    mailbox_request_start_ns[cid] = warp_kernel::get_global_timer_ns();
                }
                ++request_generation[cid];
                d_gdr_data[route_idx * kGdrMailboxStride] = 0;
                d_status_memory[route_idx * 16] = now_local_row;
                d_status_memory[route_idx * 16 + 1] = current_buffer_idx[cid];
                d_status_memory[route_idx * 16 + 2] = request_generation[cid];
                __threadfence_system();
                d_status_flag[route_idx * 16] = 1;
                if constexpr (kUseRequestRing) {
                    request_ring_device::publish_host_request(
                        d_request_ring_entries,
                        d_request_ring_consumed_sequences,
                        d_request_ring_tails,
                        request_ring_idx,
                        request_ring_capacity,
                        route_idx,
                        HostRequestKind::Demand);
                }
            }
            __syncwarp();
            if (use_adaptive_wait) {
                while (warp_kernel::load_bypass_l1(&d_gdr_data[route_idx * kGdrMailboxStride]) == 0) {
                    warp_kernel::sleep_cuda_us(static_cast<float>(sleep_us));
                    __threadfence_system();
                    ++wait_loops;
                }
            } else {
                while (warp_kernel::load_bypass_l1(&d_gdr_data[route_idx * kGdrMailboxStride]) == 0) {
                    warp_kernel::sleep_cuda_us(kFixedWaitSleepUs);
                    __threadfence_system();
                }
            }
            if (subtid == 0) {
                wait_sleep_us[cid] = use_adaptive_wait
                    ? warp_kernel::next_wait_sleep_us(sleep_us,
                                                      wait_loops,
                                                      kMinWaitSleepUs,
                                                      kMaxWaitSleepUs)
                    : kInitialWaitSleepUs;
                if constexpr (kCollectMailboxRtt) {
                    const uint64_t elapsed_ns =
                        warp_kernel::get_global_timer_ns() - mailbox_request_start_ns[cid];
                    ++mailbox_rtt_hist[cid][rabitq_mailbox_rtt_bin(elapsed_ns)];
                    mailbox_rtt_sum_ns[cid] += elapsed_ns;
                }
            }
            __syncwarp();
            set_rabitq_warp_phase(
                d_phase_state, route_idx, subtid, RabitqWarpPhase::Frontier);
        }
        if constexpr (kCollectDetailedTiming) {
            time2 += static_cast<float>(warp_kernel::get_global_timer_ns() - io_start) /
                     1000.0f;
        }

        if (subtid == 0) {
            const bool bit_match =
                (graph_row_filter_mask_or_all_ones(d_row_filter_mask, global_row) &
                 query_filter_bit) != 0ULL;
            const uint32_t sample_idx = d_row_to_sample[global_row];
            const uint32_t matched_active_dims = go_fann::range_filter::count_matching_active_ranges(
                d_query_range_modes,
                d_query_range_lower_bounds,
                d_query_range_upper_bounds,
                d_base_range_values,
                route_idx,
                sample_idx,
                range_count);
            const bool exact_range_match = (active_range_count == 0) || (matched_active_dims == active_range_count);
            if (gpu_predicate::enabled(predicate, route_idx)) {
                lane0_state[cid].now_filter_result = predicate_match_observed(predicate, route_idx, sample_idx, false) ? 1 : 2;
            } else if (node_on_gpu) {
                lane0_state[cid].now_filter_result = rabitq_gpu_full_labels_match(
                    d_gpu_graph_cache,
                    route_idx,
                    sample_idx)
                    ? static_cast<uint8_t>(1)
                    : static_cast<uint8_t>(2);
            } else if (!node_prefetched) {
                lane0_state[cid].now_filter_result = static_cast<uint8_t>(
                    d_gdr_data[route_idx * kGdrMailboxStride]);
            } else if (predicate.route_compiled && predicate.route_compiled[route_idx]) {
                // Uncovered compiled residuals disable prefetch. Thus the only
                // CPU-plan route reaching here is an entailed predicate.
                lane0_state[cid].now_filter_result = 1;
            }
            if (!predicate.route_compiled || !predicate.route_compiled[route_idx]) {
                lane0_state[cid].now_filter_result =
                    (lane0_state[cid].now_filter_result == 1 && bit_match && exact_range_match)
                    ? static_cast<uint8_t>(1)
                    : static_cast<uint8_t>(2);
            }
        }
        __syncwarp();
        const uint8_t lane_filter_result = subtid == 0
            ? lane0_state[cid].now_filter_result
            : static_cast<uint8_t>(0);
        const uint8_t now_filter_result = static_cast<uint8_t>(__shfl_sync(
            warp_kernel::kFullMask, lane_filter_result, 0));

        __syncwarp();
        const size_t neighbor_count_offset = static_cast<size_t>(mailbox_vector_words);
        const size_t neighbor_offset = neighbor_count_offset + 1;
        u_int32_t node_degree;
        int element;
        const size_t payload_slot =
            (static_cast<size_t>(route_idx) << 1) | current_buffer_idx[cid];
        const size_t payload_base = payload_slot * static_cast<size_t>(mailbox_slot_words);
        if (node_on_gpu) {
            node_degree = cached_node_degree;
            element = 0;
        } else if (!node_prefetched) {
            node_degree = d_neighbor_memory[payload_base + neighbor_count_offset];
            node_degree = (node_degree > static_cast<u_int32_t>(FIXED_DEGREE))
                              ? static_cast<u_int32_t>(FIXED_DEGREE)
                              : node_degree;
            element = d_neighbor_memory[payload_base + subtid];
        } else {
            node_degree = d_prefetch_node_array[prefetch_vector_index].neighbor_count;
            node_degree = (node_degree > static_cast<u_int32_t>(FIXED_DEGREE))
                              ? static_cast<u_int32_t>(FIXED_DEGREE)
                              : node_degree;
            element = d_prefetch_node_array[prefetch_vector_index].data[subtid];
        }
        if (!node_prefetched && !node_on_gpu) {
            rabitq_gpu_graph_cache_store(
                d_gpu_graph_cache,
                global_row,
                node_degree,
                d_neighbor_memory + payload_base + neighbor_offset,
                static_cast<uint32_t>(subtid));
        }

        dist_t true_distance = 0.0f;
        dist_t true_distance_sum = 0.0f;
        if (now_filter_result == 1 && enable_rerank != 0) {
            if (rerank_payload_float != 0) {
                const float* query = d_query_float +
                    static_cast<size_t>(vector_query_idx) * dim;
                for (int d = subtid; d < dim; d += step) {
                    const float base = __uint_as_float(
                        d_neighbor_memory[payload_base + static_cast<size_t>(d)]);
                    const float diff = base - query[d];
                    true_distance_sum += diff * diff;
                }
            } else {
                const size_t data_words = static_cast<size_t>(dim) / 4;
                for (size_t word_idx = subtid; word_idx < data_words; word_idx += step) {
                    element = !node_prefetched
                        ? d_neighbor_memory[payload_base + word_idx]
                        : d_prefetch_node_array[prefetch_vector_index].data[word_idx];
                    const int base_dim = static_cast<int>(word_idx) * 4;
                    const u_int8_t a = element & 0xFF;
                    const u_int8_t b = (element >> 8) & 0xFF;
                    const u_int8_t c = (element >> 16) & 0xFF;
                    const u_int8_t d = (element >> 24) & 0xFF;
                    true_distance_sum += (a - query_point[cid][base_dim + 0]) *
                                         (a - query_point[cid][base_dim + 0]);
                    true_distance_sum += (b - query_point[cid][base_dim + 1]) *
                                         (b - query_point[cid][base_dim + 1]);
                    true_distance_sum += (c - query_point[cid][base_dim + 2]) *
                                         (c - query_point[cid][base_dim + 2]);
                    true_distance_sum += (d - query_point[cid][base_dim + 3]) *
                                         (d - query_point[cid][base_dim + 3]);
                }
            }
            for (int offset2 = 16; offset2 > 0; offset2 /= 2) {
                true_distance_sum += __shfl_xor_sync(
                    warp_kernel::kFullMask, true_distance_sum, offset2);
            }
            if (subtid == 0) {
                true_distance = true_distance_sum;
            }
        }

        __syncwarp();
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
        if (subtid == 0) {
            expansion.append(route_idx, now_local_row);
            ++expanded_count[cid];
            if (now_filter_result == 1) {
                ++pass_count[cid];
                const dist_t current_approx_distance = [&] {
                    if constexpr (kPredicateBias) return lane0_state[cid].now.base;
                    else return lane0_state[cid].now.first;
                }();
                const dist_t rank_distance =
                    (enable_rerank != 0) ? true_distance : current_approx_distance;
                RabitqResultPair topk_entry;
                topk_entry.first = rank_distance;
                topk_entry.second = now_local_row;
                RabitqStopPair stop_entry;
                stop_entry.first = rank_distance;
                stop_entry.second = pack_topk_payload(now_local_row, current_approx_distance);
                bool stop_topk_improved = false;
                if (lane0_state[cid].stop_topk_heap_size < stop_width) {
                    stop_topk[cid][lane0_state[cid].stop_topk_heap_size++] = stop_entry;
                    push_heap(stop_topk[cid],
                              stop_topk[cid] + lane0_state[cid].stop_topk_heap_size);
                    stop_topk_improved = true;
                    if (lane0_state[cid].stop_topk_heap_size == 1 ||
                        current_approx_distance > lane0_state[cid].stop_topk_approx_worst) {
                        lane0_state[cid].stop_topk_approx_worst = current_approx_distance;
                    }
                } else if (stop_entry.first < stop_topk[cid][0].first) {
                    stop_topk[cid][lane0_state[cid].stop_topk_heap_size++] = stop_entry;
                    push_heap(stop_topk[cid],
                              stop_topk[cid] + lane0_state[cid].stop_topk_heap_size);
                    auto removed = pop_heap(stop_topk[cid],
                                            stop_topk[cid] +
                                                lane0_state[cid].stop_topk_heap_size);
                    --lane0_state[cid].stop_topk_heap_size;
                    stop_topk_improved = true;
                    const uint32_t removed_approx_bits = static_cast<uint32_t>(removed.second >> 32);
                    if (current_approx_distance > lane0_state[cid].stop_topk_approx_worst) {
                        lane0_state[cid].stop_topk_approx_worst = current_approx_distance;
                    } else if (removed_approx_bits ==
                               dist_to_bits(lane0_state[cid].stop_topk_approx_worst)) {
                        lane0_state[cid].stop_topk_approx_worst = -FLT_MAX;
                        for (int ii = 0;
                             ii < lane0_state[cid].stop_topk_heap_size;
                             ++ii) {
                            const dist_t approx_dist = unpack_topk_approx_dist(stop_topk[cid][ii].second);
                            if (approx_dist > lane0_state[cid].stop_topk_approx_worst) {
                                lane0_state[cid].stop_topk_approx_worst = approx_dist;
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
                } else if (rank_distance < topk[0].first) {
                    topk[lane0_state[cid].topk_heap_size++] = topk_entry;
                    push_heap(topk, topk + lane0_state[cid].topk_heap_size);
                    pop_heap(topk, topk + lane0_state[cid].topk_heap_size);
                    --lane0_state[cid].topk_heap_size;
                }
            }

            for (int i = 0; i < node_degree; ++i) {
                u_int32_t idx = node_on_gpu
                    ? index_list[cid][i]
                    : (!node_prefetched
                           ? d_neighbor_memory[payload_base + neighbor_offset + i]
                           : d_prefetch_node_array[prefetch_vector_index].neighbor[i]);
                if (idx == UINT32_MAX ||
                    native_visited_hash::test<kHashTableCapacity>(d_hash_scratch, idx)) {
                    continue;
                }
                index_list[cid][index_list_len[cid]++] = idx;
            }
        }
        if constexpr (kCollectDetailedTiming) {
            time4 += static_cast<float>(warp_kernel::get_global_timer_ns() - queue_start) /
                     1000.0f;
        }

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
        __syncwarp();

        uint64_t calculation_start = 0;
        if constexpr (kCollectDetailedTiming) {
            calculation_start = warp_kernel::get_global_timer_ns();
        }
        set_rabitq_warp_phase(
            d_phase_state, route_idx, subtid, RabitqWarpPhase::Score);
        int degree_limit = index_list_len[cid];
        if (subtid == 0 && predicate.counters) predicate.counters[route_idx].scores += degree_limit;
        for (int i = 0; i < degree_limit; ++i) {
            const uint32_t sample_idx = d_row_to_sample[label_offset + index_list[cid][i]];
            const float dist = score_rabitq_code_device<kUseInt8Score>(
                d_anchor_ids,
                d_codes,
                d_long_codes,
                d_factors,
                d_ex_factors,
                d_rotated_query,
                d_short_lut,
                d_q_k1xsumq,
                d_q_kbxsumq,
                d_centroid_distances,
                vector_query_idx,
                sample_idx,
                num_anchors,
                padded_dim,
                code_bytes,
                long_code_bytes,
                bits_per_dim,
                subtid,
                d_quantized_query,
                d_quantized_query_scales);
            if (subtid == 0) {
                dist_list[cid][i] = dist;
            }
        }
        if constexpr (kCollectDetailedTiming) {
            time3 += static_cast<float>(
                warp_kernel::get_global_timer_ns() - calculation_start) / 1000.0f;
        }
        set_rabitq_warp_phase(
            d_phase_state, route_idx, subtid, RabitqWarpPhase::Frontier);

        __syncwarp();
        uint64_t insert_start = 0;
        if constexpr (kCollectDetailedTiming) {
            insert_start = warp_kernel::get_global_timer_ns();
        }
#if GO_FANN_GPU_PREDICATE_PARALLEL
        bool parallel_predicate = false;
        if constexpr (kPredicateBias) {
            parallel_predicate = gpu_predicate::prepare_candidate_pass_masks(
                predicate, route_idx, index_list[cid], degree_limit,
                label_offset, d_row_to_sample, predicate_pass_masks[cid], subtid);
        }
#endif
        if (subtid == 0) {
            for (int i = 0; i < degree_limit; ++i) {
                FrontierPair kp;
                const uint32_t local_row = index_list[cid][i];
                const uint32_t global_row = label_offset + local_row;
                const uint32_t sample_idx = d_row_to_sample[global_row];
                // Main-path factors are unity: raw distance is also the stop score.
                kp.first = dist_list[cid][i];
                if (bit_filter_bonus != 1.0f || range_filter_bonus != 1.0f) {
                    const uint64_t neighbor_filter_mask =
                        graph_row_filter_mask_or_all_ones(d_row_filter_mask, global_row);
                    const bool bit_match = (neighbor_filter_mask & query_filter_bit) != 0ULL;
                    const uint32_t matched_active_dims = go_fann::range_filter::count_matching_active_ranges(
                        d_query_range_modes,
                        d_query_range_lower_bounds,
                        d_query_range_upper_bounds,
                        d_base_range_values,
                        route_idx,
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
                    // Uncovered routes retain exact CPU admission and their
                    // original navigation priority; no partial-label penalty.
                    const bool pass =
#if GO_FANN_GPU_PREDICATE_PARALLEL
                        parallel_predicate
                        ? ((predicate_pass_masks[cid][i / 32] >> (i % 32)) & 1u) != 0
                        :
#endif
                        (!gpu_predicate::enabled(predicate, route_idx) ||
                         predicate_match_observed(predicate, route_idx, sample_idx, true));
                    bool saturated = false;
                    kp.first = gpu_predicate::priority(kp.base, predicate_penalty, pass, saturated);
                    if (saturated && predicate.counters) ++predicate.counters[route_idx].saturated;
                }
                kp.second = pack_pool_payload(local_row, false);
                if (heap_size[cid] >= queue_limit + 1 && q[cid][2].first < kp.first) {
                    continue;
                }
                smmh2::insert(q[cid], heap_size[cid], kp);
                native_visited_hash::add<kHashTableCapacity>(d_hash_scratch, local_row);
                if (heap_size[cid] >= queue_limit + 2) {
                    smmh2::pop_max(q[cid], heap_size[cid]);
                }
            }
        }
        // Lane 0 mutates the shared frontier and heap_size above. Cooperative
        // pickup reads both on every lane; synchronize BEFORE evaluating its
        // by-value size argument (a barrier inside pickup would be too late).
        __syncwarp();
        if (step_num == 1) {
            bool found = false;
            if constexpr (kUseCooperativePickup) {
                found = pickup_next_unused_from_native_frontier<true>(
                    q[cid], heap_size[cid], lane0_state[cid].next, subtid);
            } else if (subtid == 0) {
                found = pickup_next_unused_from_pool(
                    q[cid], heap_size[cid], lane0_state[cid].next);
            }
            if (subtid == 0 && !found) {
                pool_exhausted[cid] = true;
                lane0_state[cid].next.first = FLT_MAX;
                lane0_state[cid].next.second = pack_pool_payload(UINT32_MAX, true);
            }
        }
        if constexpr (kCollectDetailedTiming) {
            time4 += static_cast<float>(warp_kernel::get_global_timer_ns() - insert_start) /
                     1000.0f;
        }
        __syncwarp();

        if (subtid == 0) {
            current_buffer_idx[cid] ^= 1;
        }

        if (subtid == 0) {
            lane0_state[cid].now = lane0_state[cid].next;
        }
        __syncwarp();
    }

    uint64_t end1 = warp_kernel::get_global_timer_ns();
    time1 = static_cast<float>(end1 - start1) / 1000.0f;
    set_rabitq_warp_phase(d_phase_state, route_idx, subtid, RabitqWarpPhase::Done);
    if (subtid == 0) {
        expansion.finish(route_idx, expanded_count[cid]);
        d_status_flag[route_idx * 16] = 2;
        const int valid_topk = lane0_state[cid].topk_heap_size < result_width
            ? lane0_state[cid].topk_heap_size
            : result_width;
        for (int i = 0; i < result_width; ++i) {
            d_result[static_cast<size_t>(route_idx) * result_width + i] = UINT32_MAX;
            d_result_distances[static_cast<size_t>(route_idx) * result_width + i] = FLT_MAX;
        }
        for (int i = 0; i < valid_topk; ++i) {
            auto res = pop_heap(topk, topk + lane0_state[cid].topk_heap_size - i);
            const uint32_t local_row = res.second;
            const uint32_t global_row = label_offset + local_row;
            const int output_slot = valid_topk - 1 - i;
            if (local_row == UINT32_MAX || global_row < label_offset) {
                continue;
            }
            d_result[static_cast<size_t>(route_idx) * result_width + output_slot] =
                d_row_to_sample[global_row];
            d_result_distances[static_cast<size_t>(route_idx) * result_width + output_slot] =
                res.first;
        }
        time_per_block[static_cast<size_t>(route_idx) * 6] = time1;
        time_per_block[static_cast<size_t>(route_idx) * 6 + 1] =
            kCollectDetailedTiming ? time2 : -1.0f;
        time_per_block[static_cast<size_t>(route_idx) * 6 + 2] =
            kCollectDetailedTiming ? time3 : -1.0f;
        time_per_block[static_cast<size_t>(route_idx) * 6 + 3] =
            kCollectDetailedTiming ? time4 : -1.0f;
        time_per_block[static_cast<size_t>(route_idx) * 6 + 4] = static_cast<float>(step_num);
        time_per_block[static_cast<size_t>(route_idx) * 6 + 5] =
            static_cast<float>(graph_cache_hits);
        if constexpr (kCollectMailboxRtt) {
            const size_t record_base =
                static_cast<size_t>(route_idx) * kRabitqMailboxRttRecordWords;
            for (size_t bin = 0; bin < kRabitqMailboxRttBinCount; ++bin) {
                d_mailbox_rtt_hist[record_base + bin] = mailbox_rtt_hist[cid][bin];
            }
            d_mailbox_rtt_hist[record_base + kRabitqMailboxRttSumWord] =
                mailbox_rtt_sum_ns[cid];
        }
        __threadfence_system();
        d_status_flag[route_idx * 16] = 3;
        if constexpr (kUseRequestRing) {
            request_ring_device::publish_host_request(
                d_request_ring_entries,
                d_request_ring_consumed_sequences,
                d_request_ring_tails,
                request_ring_idx,
                request_ring_capacity,
                route_idx,
                HostRequestKind::Done);
        }
        return;
    }
}

template <int kQueueSize,
          bool kCollectMailboxRtt,
          bool kUseRequestRing,
          bool kUseInt8Score,
          bool kDeviceResident,
          bool kPredicateBias = false>
__global__ __launch_bounds__(kNativeThreadInBlock, kNativeMinBlocksPerSm)
void native_rabitq_search_kernel(const float* d_anchors,
                                            const uint32_t* d_anchor_ids,
                                            const uint8_t* d_codes,
                                            const uint8_t* d_long_codes,
                                            const float* d_factors,
                                            const float* d_ex_factors,
                                            const float* d_rotated_query,
                                            const int8_t* d_quantized_query,
                                            const float* d_quantized_query_scales,
                                            const float* d_short_lut,
                                            const float* d_q_k1xsumq,
                                            const float* d_q_kbxsumq,
                                            const float* d_centroid_distances,
                                            const uint32_t* d_query_ids,
                                            const uint32_t* d_route_to_original_query,
                                            const value_t* d_query,
                                            const float* d_query_float,
                                            idx_t* d_result,
                                            dist_t* d_result_distances,
                                            dist_t* d_datadist_list,
                                            uint32_t* d_hashdata,
                                            uint32_t graph_scratch_capacity,
                                            int num_query,
                                            float* time_per_block,
                                            uint16_t* d_phase_state,
                                            uint64_t* d_mailbox_rtt_hist,
                                            uint32_t* d_neighbor_memory,
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
                                            const uint32_t* d_row_to_sample,
                                            const uint32_t* d_label_offset,
                                            const uint64_t* d_row_filter_mask,
                                            const uint64_t* d_query_filter_bits,
                                            const RabitqGpuGraphCacheDeviceView* d_gpu_graph_cache,
                                            gpu_predicate::View predicate,
                                            const uint8_t* d_query_range_modes,
                                            const float* d_query_range_lower_bounds,
                                            const float* d_query_range_upper_bounds,
                                            const float* d_base_range_values,
                                            uint32_t range_count,
                                            int dim,
                                            uint32_t padded_dim,
                                            uint32_t num_anchors,
                                            uint32_t code_bytes,
                                            uint32_t long_code_bytes,
                                            uint32_t bits_per_dim,
                                            int use_adaptive_wait,
                                            int queue_limit,
                                            int finish_limit,
                                            int early_stop_mode,
                                            int max_steps,
                                            int stop_width,
                                            int result_width,
                                            uint32_t mailbox_vector_words,
                                            uint32_t mailbox_slot_words,
                                            float bit_filter_bonus,
                                            float range_filter_bonus,
                                            int enable_rerank,
                                            int rerank_payload_float,
                                            expansion_trace::View expansion = {})
{
    extern __shared__ RabitqResultPair topk_workspaces[];
    const uint32_t cid = static_cast<uint32_t>(threadIdx.x / kCriticalStep);
    const uint32_t active_query_idx =
        static_cast<uint32_t>(blockIdx.x * kNativeMultiQuery) + cid;
    if (active_query_idx >= static_cast<uint32_t>(num_query)) {
        return;
    }
    if (active_query_idx >= graph_scratch_capacity) {
        return;
    }
    const uint32_t route_idx =
        d_query_ids != nullptr ? d_query_ids[active_query_idx] : active_query_idx;
    const uint32_t vector_query_idx = d_route_to_original_query[route_idx];
    dist_t* d_datadist_scratch =
        d_datadist_list + static_cast<size_t>(active_query_idx) * kGraphDatadistSlotCapacity;
    uint32_t* d_hash_scratch =
        d_hashdata + static_cast<size_t>(active_query_idx) * kHashTableCapacity;
    RabitqResultPair* topk = topk_workspaces +
        static_cast<size_t>(cid) * (static_cast<size_t>(result_width) + 1);
    run_rabitq_search_one_block<kQueueSize,
                                kNativeMultiQuery,
                                kNativeDetailedTiming,
                                kCollectMailboxRtt,
                                kUseRequestRing,
                                kUseInt8Score,
                                kDeviceResident,
                                kNativeCooperativePickup, kPredicateBias>(
        d_anchors,
        d_anchor_ids,
        d_codes,
        d_long_codes,
        d_factors,
        d_ex_factors,
        d_rotated_query,
        d_quantized_query,
        d_quantized_query_scales,
        d_short_lut,
        d_q_k1xsumq,
        d_q_kbxsumq,
        d_centroid_distances,
        route_idx,
        vector_query_idx,
        d_query,
        d_query_float,
        d_result,
        d_result_distances,
        d_datadist_scratch,
        d_hash_scratch,
        time_per_block,
        d_phase_state,
        d_mailbox_rtt_hist,
        d_neighbor_memory,
        d_status_memory,
        d_gdr_data,
        d_status_flag,
        d_request_ring_entries,
        d_request_ring_consumed_sequences,
        d_request_ring_tails,
        request_ring_capacity > 0 ? active_query_idx / request_ring_capacity : 0,
        request_ring_capacity,
        d_prefetch_node_array,
        d_prefetch_index,
        d_prefetch_begin_by_group,
        d_prefetch_end_by_group,
        d_query_group_ids,
        d_start_local_ids,
        d_row_to_sample,
        d_label_offset,
        d_row_filter_mask,
        d_query_filter_bits,
        d_gpu_graph_cache,
        predicate,
        d_query_range_modes,
        d_query_range_lower_bounds,
        d_query_range_upper_bounds,
        d_base_range_values,
        range_count,
        dim,
        padded_dim,
        num_anchors,
        code_bytes,
        long_code_bytes,
        bits_per_dim,
        use_adaptive_wait,
        queue_limit,
        finish_limit,
        early_stop_mode,
        max_steps,
        stop_width,
        result_width,
        topk,
        mailbox_vector_words,
        mailbox_slot_words,
        bit_filter_bonus,
        range_filter_bonus,
        enable_rerank,
        rerank_payload_float, expansion);
}

struct RabitqFiniteKernelParams {
    const uint32_t* d_anchor_ids = nullptr;
    const uint8_t* d_codes = nullptr;
    const uint8_t* d_long_codes = nullptr;
    const float* d_factors = nullptr;
    const float* d_ex_factors = nullptr;
    const float* d_rotated_query = nullptr;
    const float* d_short_lut = nullptr;
    const float* d_q_k1xsumq = nullptr;
    const float* d_q_kbxsumq = nullptr;
    const float* d_centroid_distances = nullptr;
    const uint32_t* d_route_to_original_query = nullptr;
    idx_t* d_result = nullptr;
    dist_t* d_result_distances = nullptr;
    float* d_time_per_block = nullptr;
    uint32_t* d_hashdata = nullptr;
    uint32_t* d_neighbor_memory = nullptr;
    volatile int64_t* d_status_memory = nullptr;
    volatile int64_t* d_gdr_data = nullptr;
    volatile int64_t* d_status_flag = nullptr;
    const Node* d_prefetch_node_array = nullptr;
    const uint32_t* d_prefetch_index = nullptr;
    const uint32_t* d_prefetch_begin_by_group = nullptr;
    const uint32_t* d_prefetch_end_by_group = nullptr;
    const uint32_t* d_query_group_ids = nullptr;
    const uint32_t* d_start_local_ids = nullptr;
    const uint32_t* d_row_to_sample = nullptr;
    const uint32_t* d_label_offset = nullptr;
    const uint64_t* d_row_filter_mask = nullptr;
    const uint64_t* d_query_filter_bits = nullptr;
    const uint8_t* d_query_range_modes = nullptr;
    const float* d_query_range_lower_bounds = nullptr;
    const float* d_query_range_upper_bounds = nullptr;
    const float* d_base_range_values = nullptr;
    RabitqFiniteContextControl* d_contexts = nullptr;
    RabitqPoolPair* d_frontier_state = nullptr;
    RabitqStopPair* d_stop_state = nullptr;
    RabitqResultPair* d_result_state = nullptr;
    uint32_t query_capacity = 0;
    uint32_t graph_scratch_capacity = 0;
    uint32_t range_count = 0;
    uint32_t padded_dim = 0;
    uint32_t num_anchors = 0;
    uint32_t code_bytes = 0;
    uint32_t long_code_bytes = 0;
    uint32_t bits_per_dim = 0;
    uint32_t mailbox_vector_words = 0;
    uint32_t mailbox_slot_words = 0;
    int queue_limit = 0;
    int finish_limit = 0;
    int early_stop_mode = 0;
    int max_steps = 0;
    int stop_width = 0;
    int result_width = 0;
    float bit_filter_bonus = 1.0f;
    float range_filter_bonus = 1.0f;
};

struct RabitqFiniteFixHashView {
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
__device__ __forceinline__ void store_rabitq_finite_mutable_state(
    const RabitqFiniteKernelParams& params,
    uint32_t scratch_slot,
    int lane,
    const RabitqPoolPair* frontier,
    const RabitqStopPair* stop_heap,
    const RabitqResultPair* result_heap)
{
    const size_t frontier_base = static_cast<size_t>(scratch_slot) * (kQueueSize + 2);
    const size_t stop_base = static_cast<size_t>(scratch_slot) * (kTopk + 1);
    const size_t result_base = static_cast<size_t>(scratch_slot) *
        (static_cast<size_t>(params.result_width) + 1);
    for (int i = lane; i < kQueueSize + 2; i += kThreadInBlock) {
        params.d_frontier_state[frontier_base + i] = frontier[i];
    }
    for (int i = lane; i < kTopk + 1; i += kThreadInBlock) {
        params.d_stop_state[stop_base + i] = stop_heap[i];
    }
    for (int i = lane; i < params.result_width + 1; i += kThreadInBlock) {
        params.d_result_state[result_base + i] = result_heap[i];
    }
}

template <int kQueueSize>
__device__ __forceinline__ void load_rabitq_finite_mutable_state(
    const RabitqFiniteKernelParams& params,
    uint32_t scratch_slot,
    int lane,
    RabitqPoolPair* frontier,
    RabitqStopPair* stop_heap,
    RabitqResultPair* result_heap)
{
    const size_t frontier_base = static_cast<size_t>(scratch_slot) * (kQueueSize + 2);
    const size_t stop_base = static_cast<size_t>(scratch_slot) * (kTopk + 1);
    const size_t result_base = static_cast<size_t>(scratch_slot) *
        (static_cast<size_t>(params.result_width) + 1);
    for (int i = lane; i < kQueueSize + 2; i += kThreadInBlock) {
        frontier[i] = params.d_frontier_state[frontier_base + i];
    }
    for (int i = lane; i < kTopk + 1; i += kThreadInBlock) {
        stop_heap[i] = params.d_stop_state[stop_base + i];
    }
    for (int i = lane; i < params.result_width + 1; i += kThreadInBlock) {
        result_heap[i] = params.d_result_state[result_base + i];
    }
}

template <int kQueueSize>
__device__ void finish_rabitq_finite_context(const RabitqFiniteKernelParams& params,
                                             uint32_t route_idx,
                                             int lane,
                                             RabitqFiniteContextControl& control,
                                             RabitqResultPair* result_heap,
                                             RabitqFiniteReadyToken* token)
{
    if (lane != 0) return;
    const uint32_t group_id = params.d_query_group_ids[route_idx];
    const uint32_t label_offset = params.d_label_offset[group_id];
    params.d_status_flag[static_cast<size_t>(route_idx) * 16] = 2;
    const int valid_topk = min(static_cast<int>(control.result_heap_size), params.result_width);
    for (int i = 0; i < params.result_width; ++i) {
        params.d_result[static_cast<size_t>(route_idx) * params.result_width + i] = UINT32_MAX;
        params.d_result_distances[static_cast<size_t>(route_idx) * params.result_width + i] = FLT_MAX;
    }
    for (int i = 0; i < valid_topk; ++i) {
        const auto res = pop_heap(result_heap,
                                  result_heap + static_cast<int>(control.result_heap_size) - i);
        const uint32_t local_row = res.second;
        const int output_slot = valid_topk - 1 - i;
        if (local_row == UINT32_MAX) continue;
        params.d_result[static_cast<size_t>(route_idx) * params.result_width + output_slot] =
            params.d_row_to_sample[label_offset + local_row];
        params.d_result_distances[static_cast<size_t>(route_idx) * params.result_width + output_slot] =
            res.first;
    }
    params.d_time_per_block[static_cast<size_t>(route_idx) * 6] = 0.0f;
    params.d_time_per_block[static_cast<size_t>(route_idx) * 6 + 1] = -1.0f;
    params.d_time_per_block[static_cast<size_t>(route_idx) * 6 + 2] = -1.0f;
    params.d_time_per_block[static_cast<size_t>(route_idx) * 6 + 3] = -1.0f;
    params.d_time_per_block[static_cast<size_t>(route_idx) * 6 + 4] =
        static_cast<float>(control.step_num);
    control.state = RabitqFiniteContextState::DONE;
    params.d_contexts[route_idx] = control;
    __threadfence_system();
    params.d_status_flag[static_cast<size_t>(route_idx) * 16] = 3;
    if (token != nullptr) token->result = RabitqFiniteAdvanceResult::DONE;
}

template <int kQueueSize>
__device__ void park_rabitq_finite_context(const RabitqFiniteKernelParams& params,
                                           uint32_t route_idx,
                                           int lane,
                                           RabitqFiniteContextControl& control,
                                           RabitqPoolPair* frontier,
                                           RabitqStopPair* stop_heap,
                                           RabitqResultPair* result_heap,
                                           RabitqFiniteReadyToken* token)
{
    store_rabitq_finite_mutable_state<kQueueSize>(params,
                                                   control.scratch_slot,
                                                   lane,
                                                   frontier,
                                                   stop_heap,
                                                   result_heap);
    __syncwarp();
    if (lane != 0) return;
    const uint32_t local_row = unpack_pool_local_row(control.now_payload);
    control.state = RabitqFiniteContextState::WAIT_IO;
    control.state_since_ns = warp_kernel::get_global_timer_ns();
    ++control.io_generation;
    control.expected_row = local_row;
    params.d_contexts[route_idx] = control;
    __threadfence_system();
    params.d_gdr_data[static_cast<size_t>(route_idx) * kGdrMailboxStride] = 0;
    params.d_status_memory[static_cast<size_t>(route_idx) * 16] = local_row;
    params.d_status_memory[static_cast<size_t>(route_idx) * 16 + 1] = control.current_buffer;
    params.d_status_memory[static_cast<size_t>(route_idx) * 16 + 2] = control.io_generation;
    __threadfence_system();
    params.d_status_flag[static_cast<size_t>(route_idx) * 16] = 1;
    if (token != nullptr) token->result = RabitqFiniteAdvanceResult::PARKED;
}

template <int kQueueSize, bool kInitialize>
__device__ void run_rabitq_finite_context(const RabitqFiniteKernelParams& params,
                                          uint32_t active_query_idx,
                                          uint32_t route_idx,
                                          uint32_t context_epoch,
                                          RabitqFiniteReadyToken* token,
                                          RabitqResultPair* topk)
{
    const int lane = static_cast<int>(threadIdx.x);
    __shared__ RabitqPoolPair frontier[kQueueSize + 2];
    __shared__ RabitqStopPair stop_heap[kTopk + 1];
    __shared__ uint32_t neighbor_ids[FIXED_DEGREE * kMultiprobe];
    __shared__ dist_t neighbor_distances[FIXED_DEGREE * kMultiprobe];
    __shared__ int neighbor_count;
    __shared__ int valid_neighbor_count;
    __shared__ uint8_t current_filter_result;
    __shared__ uint8_t has_mailbox_payload;
    __shared__ int token_valid;
    __shared__ uint32_t static_prefetch_index;
    __shared__ RabitqFiniteContextControl control;
    __shared__ RabitqPoolPair next;

    if (lane == 0) {
        token_valid = route_idx < params.query_capacity ? 1 : 0;
        if constexpr (!kInitialize) {
            if (token_valid == 0) {
                token->result = RabitqFiniteAdvanceResult::REJECT_QUERY;
            } else {
                control = params.d_contexts[route_idx];
                RabitqFiniteAdvanceResult reject = RabitqFiniteAdvanceResult::PENDING;
                if (control.query_id != route_idx) reject = RabitqFiniteAdvanceResult::REJECT_QUERY;
                else if (control.context_epoch != context_epoch) reject = RabitqFiniteAdvanceResult::REJECT_EPOCH;
                else if (control.state != RabitqFiniteContextState::WAIT_IO) reject = RabitqFiniteAdvanceResult::REJECT_STATE;
                else if (control.io_generation != token->generation) reject = RabitqFiniteAdvanceResult::REJECT_GENERATION;
                else if (control.expected_row != token->row) reject = RabitqFiniteAdvanceResult::REJECT_ROW;
                else if (control.current_buffer != token->buffer) reject = RabitqFiniteAdvanceResult::REJECT_BUFFER;
                else if (warp_kernel::load_bypass_l1(&params.d_status_flag[static_cast<size_t>(route_idx) * 16]) != 1 ||
                         static_cast<uint32_t>(warp_kernel::load_bypass_l1(
                             &params.d_status_memory[static_cast<size_t>(route_idx) * 16 + 2])) != token->generation) {
                    reject = RabitqFiniteAdvanceResult::REJECT_STATUS;
                } else if (static_cast<uint32_t>(warp_kernel::load_bypass_l1(
                               &params.d_status_memory[static_cast<size_t>(route_idx) * 16])) != token->row) {
                    reject = RabitqFiniteAdvanceResult::REJECT_ROW;
                } else if (static_cast<uint32_t>(warp_kernel::load_bypass_l1(
                               &params.d_status_memory[static_cast<size_t>(route_idx) * 16 + 1]) & 1LL) != token->buffer) {
                    reject = RabitqFiniteAdvanceResult::REJECT_BUFFER;
                } else if (warp_kernel::load_bypass_l1(
                               &params.d_gdr_data[static_cast<size_t>(route_idx) * kGdrMailboxStride]) == 0) {
                    reject = RabitqFiniteAdvanceResult::REJECT_MAILBOX;
                }
                if (reject != RabitqFiniteAdvanceResult::PENDING) {
                    token_valid = 0;
                    token->result = reject;
                }
            }
        }
    }
    __syncwarp();
    if (token_valid == 0) return;

    if constexpr (kInitialize) {
        const uint32_t group_id = params.d_query_group_ids[route_idx];
        const uint32_t label_offset = params.d_label_offset[group_id];
        const uint32_t begin_id = params.d_start_local_ids[route_idx];
        const uint32_t vector_query_idx = params.d_route_to_original_query[route_idx];
        const uint32_t begin_sample = params.d_row_to_sample[label_offset + begin_id];
        const dist_t start_distance = score_rabitq_code_device(
            params.d_anchor_ids, params.d_codes, params.d_long_codes, params.d_factors,
            params.d_ex_factors, params.d_rotated_query, params.d_short_lut,
            params.d_q_k1xsumq, params.d_q_kbxsumq, params.d_centroid_distances,
            vector_query_idx, begin_sample, params.num_anchors, params.padded_dim,
            params.code_bytes, params.long_code_bytes, params.bits_per_dim, lane);
        if (lane == 0) {
            control = RabitqFiniteContextControl{};
            control.query_id = route_idx;
            control.context_epoch = context_epoch;
            control.state = RabitqFiniteContextState::RUNNING;
            control.scratch_slot = active_query_idx;
            control.state_since_ns = warp_kernel::get_global_timer_ns();
            control.stop_worst = -FLT_MAX;
            RabitqFiniteFixHashView hash{params.d_hashdata +
                static_cast<size_t>(active_query_idx) * kHashTableCapacity};
            RabitqPoolPair begin;
            begin.first = start_distance;
            begin.second = pack_pool_payload(begin_id, false);
            int frontier_size = 1;
            smmh2::insert(frontier, frontier_size, begin);
            control.frontier_size = static_cast<uint32_t>(frontier_size);
            hash.add(begin_id);
            if (!pickup_next_unused_from_pool(frontier, frontier_size, next)) {
                control.pool_exhausted = 1;
                next.first = FLT_MAX;
                next.second = pack_pool_payload(UINT32_MAX, true);
            }
            control.now_distance = next.first;
            control.now_payload = next.second;
        }
        __syncwarp();
        if (unpack_pool_local_row(control.now_payload) == UINT32_MAX) {
            finish_rabitq_finite_context<kQueueSize>(
                params, route_idx, lane, control, topk, nullptr);
            return;
        }
        if (lane == 0) has_mailbox_payload = 0;
    } else {
        load_rabitq_finite_mutable_state<kQueueSize>(
            params, control.scratch_slot, lane, frontier, stop_heap, topk);
        __syncwarp();
        if (lane == 0) {
            control.state = RabitqFiniteContextState::RUNNING;
            control.state_since_ns = warp_kernel::get_global_timer_ns();
            params.d_contexts[route_idx].state = RabitqFiniteContextState::RUNNING;
            params.d_status_flag[static_cast<size_t>(route_idx) * 16] = 0;
            has_mailbox_payload = 1;
        }
    }
    __syncwarp();

    const uint32_t group_id = params.d_query_group_ids[route_idx];
    const uint32_t label_offset = params.d_label_offset[group_id];
    const uint64_t query_filter_bit = params.d_query_filter_bits[route_idx];
    const uint32_t active_range_count = go_fann::range_filter::count_active_ranges(
        params.d_query_range_modes, route_idx, params.range_count);
    while (true) {
        const uint32_t now_local_row = unpack_pool_local_row(control.now_payload);
        if (now_local_row == UINT32_MAX) {
            finish_rabitq_finite_context<kQueueSize>(params, route_idx, lane, control, topk, token);
            return;
        }

        if (lane == 0) static_prefetch_index = UINT32_MAX;
        __syncwarp();
        if (has_mailbox_payload == 0 &&
            control.step_num < kRabitqStaticPrefetchStepLimit) {
            const uint32_t prefetch_begin = params.d_prefetch_begin_by_group[group_id];
            const uint32_t prefetch_end = params.d_prefetch_end_by_group[group_id];
            const uint32_t prefetch_len = prefetch_end - prefetch_begin;
            if (prefetch_len > 0) {
                const uint32_t local_prefetch_index = warp_kernel::n_ary_scan_prefetch(
                    params.d_prefetch_index + prefetch_begin, now_local_row, prefetch_len);
                if (lane == 0 && local_prefetch_index != UINT32_MAX) {
                    static_prefetch_index = prefetch_begin + local_prefetch_index;
                }
            }
        }
        __syncwarp();
        const bool node_prefetched = static_prefetch_index != UINT32_MAX;
        if (has_mailbox_payload == 0 && !node_prefetched) {
            park_rabitq_finite_context<kQueueSize>(
                params, route_idx, lane, control, frontier, stop_heap, topk, token);
            return;
        }

        if (lane == 0) {
            ++control.step_num;
            if (control.step_num != 1) {
                int frontier_size = static_cast<int>(control.frontier_size);
                if (!pickup_next_unused_from_pool(frontier, frontier_size, next)) {
                    control.pool_exhausted = 1;
                    next.first = FLT_MAX;
                    next.second = pack_pool_payload(UINT32_MAX, true);
                } else if (control.stop_heap_size == static_cast<uint32_t>(params.stop_width) &&
                           control.stop_worst <= next.first &&
                           params.early_stop_mode != kSearchEarlyStopOff) {
                    ++control.finished;
                }
                control.frontier_size = static_cast<uint32_t>(frontier_size);
            }
        }
        __syncwarp();

        const size_t payload_slot =
            (static_cast<size_t>(route_idx) << 1) | control.current_buffer;
        const size_t payload_base = payload_slot * static_cast<size_t>(params.mailbox_slot_words);
        if (lane == 0) {
            const uint32_t global_row = label_offset + now_local_row;
            const uint32_t sample_idx = params.d_row_to_sample[global_row];
            const bool bit_match =
                (graph_row_filter_mask_or_all_ones(params.d_row_filter_mask, global_row) &
                 query_filter_bit) != 0ULL;
            const uint32_t matched_active_dims = go_fann::range_filter::count_matching_active_ranges(
                params.d_query_range_modes, params.d_query_range_lower_bounds,
                params.d_query_range_upper_bounds, params.d_base_range_values,
                route_idx, sample_idx, params.range_count);
            const bool exact_range_match =
                active_range_count == 0 || matched_active_dims == active_range_count;
            if (node_prefetched) {
                current_filter_result = (bit_match && exact_range_match) ? 1 : 0;
                neighbor_count = min(static_cast<int>(
                    params.d_prefetch_node_array[static_prefetch_index].neighbor_count), FIXED_DEGREE);
            } else {
                current_filter_result =
                    (params.d_gdr_data[static_cast<size_t>(route_idx) * kGdrMailboxStride] == 1 &&
                     bit_match && exact_range_match) ? 1 : 0;
                neighbor_count = static_cast<int>(params.d_neighbor_memory[
                    payload_base + static_cast<size_t>(params.mailbox_vector_words)]);
                neighbor_count = min(neighbor_count, FIXED_DEGREE);
            }
        }
        __syncwarp();

        if (control.pool_exhausted != 0 ||
            early_stop_reached(params.early_stop_mode, control.finished, params.finish_limit,
                               params.queue_limit, control.expanded_count, control.pass_count,
                               params.stop_width) ||
            (params.max_steps > 0 && static_cast<int>(control.step_num) >= params.max_steps)) {
            finish_rabitq_finite_context<kQueueSize>(params, route_idx, lane, control, topk, token);
            return;
        }

        if (lane == 0) {
            ++control.expanded_count;
            if (current_filter_result != 0) {
                ++control.pass_count;
                RabitqResultPair topk_entry;
                topk_entry.first = control.now_distance;
                topk_entry.second = now_local_row;
                RabitqStopPair stop_entry;
                stop_entry.first = control.now_distance;
                stop_entry.second = pack_topk_payload(now_local_row, control.now_distance);
                bool stop_improved = false;
                if (control.stop_heap_size < static_cast<uint32_t>(params.stop_width)) {
                    stop_heap[control.stop_heap_size++] = stop_entry;
                    push_heap(stop_heap, stop_heap + control.stop_heap_size);
                    stop_improved = true;
                } else if (stop_entry.first < stop_heap[0].first) {
                    stop_heap[control.stop_heap_size++] = stop_entry;
                    push_heap(stop_heap, stop_heap + control.stop_heap_size);
                    pop_heap(stop_heap, stop_heap + control.stop_heap_size);
                    --control.stop_heap_size;
                    stop_improved = true;
                }
                control.stop_worst = control.stop_heap_size == 0 ? -FLT_MAX : stop_heap[0].first;
                control.stop_worst_bits = dist_to_bits(control.stop_worst);
                if (params.early_stop_mode == kSearchEarlyStopAdaptive && stop_improved) {
                    control.finished = 0;
                }
                if (control.result_heap_size < static_cast<uint32_t>(params.result_width)) {
                    topk[control.result_heap_size++] = topk_entry;
                    push_heap(topk, topk + control.result_heap_size);
                } else if (topk_entry.first < topk[0].first) {
                    topk[control.result_heap_size++] = topk_entry;
                    push_heap(topk, topk + control.result_heap_size);
                    pop_heap(topk, topk + control.result_heap_size);
                    --control.result_heap_size;
                }
            }
            valid_neighbor_count = 0;
            RabitqFiniteFixHashView hash{params.d_hashdata +
                static_cast<size_t>(control.scratch_slot) * kHashTableCapacity};
            const size_t neighbor_offset = static_cast<size_t>(params.mailbox_vector_words) + 1;
            for (int i = 0; i < neighbor_count; ++i) {
                const uint32_t local_row = node_prefetched
                    ? params.d_prefetch_node_array[static_prefetch_index].neighbor[i]
                    : params.d_neighbor_memory[payload_base + neighbor_offset + i];
                if (local_row != UINT32_MAX && !hash.test(local_row)) {
                    neighbor_ids[valid_neighbor_count++] = local_row;
                }
            }
        }
        __syncwarp();

        const uint32_t vector_query_idx = params.d_route_to_original_query[route_idx];
        for (int i = 0; i < valid_neighbor_count; ++i) {
            const uint32_t sample_idx = params.d_row_to_sample[label_offset + neighbor_ids[i]];
            const dist_t distance = score_rabitq_code_device(
                params.d_anchor_ids, params.d_codes, params.d_long_codes, params.d_factors,
                params.d_ex_factors, params.d_rotated_query, params.d_short_lut,
                params.d_q_k1xsumq, params.d_q_kbxsumq, params.d_centroid_distances,
                vector_query_idx, sample_idx, params.num_anchors, params.padded_dim,
                params.code_bytes, params.long_code_bytes, params.bits_per_dim, lane);
            if (lane == 0) neighbor_distances[i] = distance;
        }
        __syncwarp();
        if (lane == 0) {
            RabitqFiniteFixHashView hash{params.d_hashdata +
                static_cast<size_t>(control.scratch_slot) * kHashTableCapacity};
            int frontier_size = static_cast<int>(control.frontier_size);
            for (int i = 0; i < valid_neighbor_count; ++i) {
                const uint32_t local_row = neighbor_ids[i];
                const uint32_t global_row = label_offset + local_row;
                const uint32_t sample_idx = params.d_row_to_sample[global_row];
                const bool bit_match =
                    (graph_row_filter_mask_or_all_ones(params.d_row_filter_mask, global_row) &
                     query_filter_bit) != 0ULL;
                const uint32_t matched_active_dims = go_fann::range_filter::count_matching_active_ranges(
                    params.d_query_range_modes, params.d_query_range_lower_bounds,
                    params.d_query_range_upper_bounds, params.d_base_range_values,
                    route_idx, sample_idx, params.range_count);
                RabitqPoolPair candidate;
                candidate.first = go_fann::range_filter::apply_filter_bonuses(
                    neighbor_distances[i], bit_match, matched_active_dims,
                    params.bit_filter_bonus, params.range_filter_bonus);
                candidate.second = pack_pool_payload(local_row, false);
                if (frontier_size >= params.queue_limit + 1 &&
                    frontier[2].first < candidate.first) {
                    continue;
                }
                smmh2::insert(frontier, frontier_size, candidate);
                hash.add(local_row);
                if (frontier_size >= params.queue_limit + 2) {
                    smmh2::pop_max(frontier, frontier_size);
                }
            }
            if (control.step_num == 1) {
                if (!pickup_next_unused_from_pool(frontier, frontier_size, next)) {
                    control.pool_exhausted = 1;
                    next.first = FLT_MAX;
                    next.second = pack_pool_payload(UINT32_MAX, true);
                }
            }
            control.frontier_size = static_cast<uint32_t>(frontier_size);
            control.current_buffer ^= 1;
            control.now_distance = next.first;
            control.now_payload = next.second;
            has_mailbox_payload = 0;
        }
        __syncwarp();
        if (control.step_num >= kRabitqStaticPrefetchStepLimit &&
            unpack_pool_local_row(control.now_payload) != UINT32_MAX) {
            park_rabitq_finite_context<kQueueSize>(
                params, route_idx, lane, control, frontier, stop_heap, topk, token);
            return;
        }
    }
}

template <int kQueueSize>
__global__ void finite_rabitq_init_kernel(RabitqFiniteKernelParams params,
                                          const uint32_t* d_query_ids,
                                          uint32_t query_count,
                                          uint32_t context_epoch)
{
    extern __shared__ RabitqResultPair topk[];
    const uint32_t active_query_idx = static_cast<uint32_t>(blockIdx.x);
    if (active_query_idx >= query_count) return;
    const uint32_t route_idx = d_query_ids[active_query_idx];
    run_rabitq_finite_context<kQueueSize, true>(
        params, active_query_idx, route_idx, context_epoch, nullptr, topk);
}

template <int kQueueSize>
__global__ void finite_rabitq_resume_kernel(RabitqFiniteKernelParams params,
                                            RabitqFiniteReadyToken* tokens,
                                            uint32_t token_count)
{
    extern __shared__ RabitqResultPair topk[];
    const uint32_t token_idx = static_cast<uint32_t>(blockIdx.x);
    if (token_idx >= token_count) return;
    RabitqFiniteReadyToken* token = tokens + token_idx;
    run_rabitq_finite_context<kQueueSize, false>(
        params, token_idx, token->query_id, token->context_epoch, token, topk);
}

template <int kShortlistCapacity, bool kUseInt8Score>
__device__ void run_lowfreq_rabitq_scan_one_block(const uint32_t* d_anchor_ids,
                                                  const uint8_t* d_codes,
                                                  const uint8_t* d_long_codes,
                                                  const float* d_factors,
                                                  const float* d_ex_factors,
                                                  const float* d_rotated_query,
                                                  const int8_t* d_quantized_query,
                                                  const float* d_quantized_query_scales,
                                                  const float* d_short_lut,
                                                  const float* d_q_k1xsumq,
                                                  const float* d_q_kbxsumq,
                                                  const float* d_centroid_distances,
                                                  uint32_t route_idx,
                                                  uint32_t vector_query_idx,
                                                  uint32_t qid,
                                                  idx_t* d_result,
                                                  float* d_distances,
                                                  const uint32_t* d_candidate_ids,
                                                  const uint32_t* d_candidate_offsets,
                                                  const uint32_t* d_candidate_sizes,
                                                  int topk,
                                                  int shortlist_k,
                                                  uint32_t num_anchors,
                                                  uint32_t padded_dim,
                                                  uint32_t code_bytes,
                                                  uint32_t long_code_bytes,
                                                  uint32_t bits_per_dim,
                                                  float* time_per_block);

template <int kShortlistCapacity, bool kUseInt8Score>
__device__ void run_lowfreq_rabitq_resident_scan_one_block(const uint32_t* d_anchor_ids,
                                                           const uint8_t* d_codes,
                                                           const uint8_t* d_long_codes,
                                                           const float* d_factors,
                                                           const float* d_ex_factors,
                                                           const float* d_rotated_query,
                                                           const int8_t* d_quantized_query,
                                                           const float* d_quantized_query_scales,
                                                           const float* d_short_lut,
                                                           const float* d_q_k1xsumq,
                                                           const float* d_q_kbxsumq,
                                                           const float* d_centroid_distances,
                                                           uint32_t route_idx,
                                                           uint32_t vector_query_idx,
                                                           uint32_t qid,
                                                           idx_t* d_result,
                                                           float* d_distances,
                                                           const uint32_t* d_query_group_ids,
                                                           const uint64_t* d_query_filter_bits,
                                                           const uint8_t* d_query_range_modes,
                                                           const float* d_query_range_lower_bounds,
                                                           const float* d_query_range_upper_bounds,
                                                           const float* d_base_range_values,
                                                           uint32_t range_count,
                                                           const uint32_t* d_lowfreq_label_size,
                                                           const uint32_t* d_lowfreq_label_offset,
                                                           const uint32_t* d_lowfreq_sample_ids,
                                                           const uint64_t* d_lowfreq_row_filter_mask,
                                                           uint32_t lowfreq_group_count,
                                                           uint32_t lowfreq_total_rows,
                                                           uint32_t base_rows,
                                                           int topk,
                                                           int shortlist_k,
                                                           uint32_t num_anchors,
                                                           uint32_t padded_dim,
                                                           uint32_t code_bytes,
                                                           uint32_t long_code_bytes,
                                                           uint32_t bits_per_dim,
                                                           int use_filter_mask,
                                                           float* time_per_block,
                                                           gpu_predicate::View posting_predicate = {},
                                                           bool score_first_requested = false);

template <int kQueueSize>
__global__ void persistent_rabitq_latency_kernel(
                                            const float* d_anchors,
                                            const uint32_t* d_anchor_ids,
                                            const uint8_t* d_codes,
                                            const uint8_t* d_long_codes,
                                            const float* d_factors,
                                            const float* d_ex_factors,
                                            const float* d_rotated_query,
                                            const float* d_short_lut,
                                            const float* d_q_k1xsumq,
                                            const float* d_q_kbxsumq,
                                            const float* d_centroid_distances,
                                            const uint32_t* d_route_to_original_query,
                                            const value_t* d_query,
                                            idx_t* d_result,
                                            dist_t* d_result_distances,
                                            dist_t* d_datadist_list,
                                            uint32_t* d_hashdata,
                                            uint32_t graph_scratch_capacity,
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
                                            const uint32_t* d_row_to_sample,
                                            const uint32_t* d_label_offset,
                                            const uint64_t* d_row_filter_mask,
                                            const uint64_t* d_query_filter_bits,
                                            const uint8_t* d_query_range_modes,
                                            const float* d_query_range_lower_bounds,
                                            const float* d_query_range_upper_bounds,
                                            const float* d_base_range_values,
                                            uint32_t range_count,
                                            int dim,
                                            uint32_t padded_dim,
                                            uint32_t num_anchors,
                                            uint32_t code_bytes,
                                            uint32_t long_code_bytes,
                                            uint32_t bits_per_dim,
                                            int use_adaptive_wait,
                                            int queue_limit,
                                            int finish_limit,
                                            int early_stop_mode,
                                            int max_steps,
                                            int stop_width,
                                            int result_width,
                                            uint32_t mailbox_vector_words,
                                            uint32_t mailbox_slot_words,
                                            float bit_filter_bonus,
                                            float range_filter_bonus,
                                            int enable_rerank,
                                            idx_t* d_lowfreq_result,
                                            float* d_lowfreq_distances,
                                            const uint32_t* d_lowfreq_candidate_ids,
                                            const uint32_t* d_lowfreq_candidate_offsets,
                                            const uint32_t* d_lowfreq_candidate_sizes,
                                            const uint32_t* d_lowfreq_label_size,
                                            const uint32_t* d_lowfreq_label_offset,
                                            const uint32_t* d_lowfreq_sample_ids,
                                            const uint64_t* d_lowfreq_row_filter_mask,
                                            uint32_t lowfreq_group_count,
                                            uint32_t lowfreq_total_rows,
                                            uint32_t base_rows,
                                            int lowfreq_candidate_mode,
                                            int lowfreq_shortlist_k,
                                            int lowfreq_use_filter_mask,
                                            volatile RabitqPersistentLatencyRequestControl* requests,
                                            volatile RabitqPersistentLatencyCompletionControl* completions)
{
    const uint32_t slot_idx = static_cast<uint32_t>(blockIdx.x);
    if (slot_idx >= graph_scratch_capacity) {
        return;
    }
    volatile RabitqPersistentLatencyRequestControl* request = requests + slot_idx;
    volatile RabitqPersistentLatencyCompletionControl* completion = completions + slot_idx;
    __shared__ uint32_t current_seq;
    __shared__ uint32_t route_idx;
    __shared__ uint32_t is_lowfreq_route;
    __shared__ uint32_t should_stop;
    __shared__ RabitqResultPair topk_workspace[kHighfreqRerankShortlistMax + 1];
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
            route_idx = load_bypass_l1_u32(&request->query_id);
            is_lowfreq_route = load_bypass_l1_u32(&request->lowfreq_route);
        }
        __syncthreads();
        if (should_stop != 0 || current_seq == kPersistentLatencyStopSeq) {
            return;
        }

        const uint64_t gpu_start_clock = warp_kernel::get_global_timer_ns();
        const uint32_t vector_query_idx = d_route_to_original_query[route_idx];
        if (is_lowfreq_route != 0) {
            if (lowfreq_candidate_mode != 0) {
                run_lowfreq_rabitq_scan_one_block<kLowfreqRerankShortlistMax, false>(d_anchor_ids,
                                                  d_codes,
                                                  d_long_codes,
                                                  d_factors,
                                                  d_ex_factors,
                                                  d_rotated_query,
                                                  nullptr,
                                                  nullptr,
                                                  d_short_lut,
                                                  d_q_k1xsumq,
                                                  d_q_kbxsumq,
                                                  d_centroid_distances,
                                                  route_idx,
                                                  vector_query_idx,
                                                  slot_idx,
                                                  d_lowfreq_result,
                                                  d_lowfreq_distances,
                                                  d_lowfreq_candidate_ids,
                                                  d_lowfreq_candidate_offsets,
                                                  d_lowfreq_candidate_sizes,
                                                  stop_width,
                                                  lowfreq_shortlist_k,
                                                  num_anchors,
                                                  padded_dim,
                                                  code_bytes,
                                                  long_code_bytes,
                                                  bits_per_dim,
                                                  time_per_block);
            } else {
                run_lowfreq_rabitq_resident_scan_one_block<kLowfreqRerankShortlistMax, false>(
                                                           d_anchor_ids,
                                                           d_codes,
                                                           d_long_codes,
                                                           d_factors,
                                                           d_ex_factors,
                                                           d_rotated_query,
                                                           nullptr,
                                                           nullptr,
                                                           d_short_lut,
                                                           d_q_k1xsumq,
                                                           d_q_kbxsumq,
                                                           d_centroid_distances,
                                                           route_idx,
                                                           vector_query_idx,
                                                           slot_idx,
                                                           d_lowfreq_result,
                                                           d_lowfreq_distances,
                                                           d_query_group_ids,
                                                           d_query_filter_bits,
                                                           d_query_range_modes,
                                                           d_query_range_lower_bounds,
                                                           d_query_range_upper_bounds,
                                                           d_base_range_values,
                                                           range_count,
                                                           d_lowfreq_label_size,
                                                           d_lowfreq_label_offset,
                                                           d_lowfreq_sample_ids,
                                                           d_lowfreq_row_filter_mask,
                                                           lowfreq_group_count,
                                                           lowfreq_total_rows,
                                                           base_rows,
                                                           stop_width,
                                                           lowfreq_shortlist_k,
                                                           num_anchors,
                                                           padded_dim,
                                                           code_bytes,
                                                           long_code_bytes,
                                                           bits_per_dim,
                                                           lowfreq_use_filter_mask,
                                                           time_per_block);
            }
        } else {
            dist_t* d_datadist_scratch =
                d_datadist_list + static_cast<size_t>(slot_idx) * kGraphDatadistSlotCapacity;
            uint32_t* d_hash_scratch =
                d_hashdata + static_cast<size_t>(slot_idx) * kHashTableCapacity;
            run_rabitq_search_one_block<kQueueSize, 1, true, false, false, false, false, false>(
                d_anchors,
                d_anchor_ids,
                d_codes,
                d_long_codes,
                d_factors,
                d_ex_factors,
                d_rotated_query,
                nullptr,
                nullptr,
                d_short_lut,
                d_q_k1xsumq,
                d_q_kbxsumq,
                d_centroid_distances,
                route_idx,
                vector_query_idx,
                d_query,
                nullptr,
                d_result,
                d_result_distances,
                d_datadist_scratch,
                d_hash_scratch,
                time_per_block,
                nullptr,
                nullptr,
                d_neighbor_memory,
                d_status_memory,
                d_gdr_data,
                d_status_flag,
                nullptr,
                nullptr,
                nullptr,
                0,
                0,
                d_prefetch_node_array,
                d_prefetch_index,
                d_prefetch_begin_by_group,
                d_prefetch_end_by_group,
                d_query_group_ids,
                d_start_local_ids,
                d_row_to_sample,
                d_label_offset,
                d_row_filter_mask,
                d_query_filter_bits,
                nullptr,
                gpu_predicate::View{},
                d_query_range_modes,
                d_query_range_lower_bounds,
                d_query_range_upper_bounds,
                d_base_range_values,
                range_count,
                dim,
                padded_dim,
                num_anchors,
                code_bytes,
                long_code_bytes,
                bits_per_dim,
                use_adaptive_wait,
                queue_limit,
                finish_limit,
                early_stop_mode,
                max_steps,
                stop_width,
                result_width,
                topk_workspace,
                mailbox_vector_words,
                mailbox_slot_words,
                bit_filter_bonus,
                range_filter_bonus,
                enable_rerank,
                0);
        }
        const uint64_t gpu_stop_clock = warp_kernel::get_global_timer_ns();
        __syncthreads();
        if (threadIdx.x == 0) {
            completion->query_id = route_idx;
            completion->gpu_start_clock = gpu_start_clock;
            completion->gpu_stop_clock = gpu_stop_clock;
            __threadfence_system();
            completion->done_seq = current_seq;
            last_seq = current_seq;
        }
        __syncthreads();
    }
}

template <int kShortlistCapacity, bool kUseInt8Score>
__device__ void run_lowfreq_rabitq_scan_one_block(const uint32_t* d_anchor_ids,
                                                  const uint8_t* d_codes,
                                                  const uint8_t* d_long_codes,
                                                  const float* d_factors,
                                                  const float* d_ex_factors,
                                                  const float* d_rotated_query,
                                                  const int8_t* d_quantized_query,
                                                  const float* d_quantized_query_scales,
                                                  const float* d_short_lut,
                                                  const float* d_q_k1xsumq,
                                                  const float* d_q_kbxsumq,
                                                  const float* d_centroid_distances,
                                                  uint32_t route_idx,
                                                  uint32_t vector_query_idx,
                                                  uint32_t qid,
                                                  idx_t* d_result,
                                                  float* d_distances,
                                                  const uint32_t* d_candidate_ids,
                                                  const uint32_t* d_candidate_offsets,
                                                  const uint32_t* d_candidate_sizes,
                                                  int topk,
                                                  int shortlist_k,
                                                  uint32_t num_anchors,
                                                  uint32_t padded_dim,
                                                  uint32_t code_bytes,
                                                  uint32_t long_code_bytes,
                                                  uint32_t bits_per_dim,
                                                  float* time_per_block)
{
    const int tid = threadIdx.x;
    const int subtid = tid % warp_kernel::kWarpThreads;
    if (topk > kTopk || shortlist_k > kShortlistCapacity) {
        return;
    }

    __shared__ float best_dist[kShortlistCapacity];
    __shared__ idx_t best_ids[kShortlistCapacity];
    for (int i = tid; i < shortlist_k; i += blockDim.x) {
        best_dist[i] = FLT_MAX;
        best_ids[i] = UINT32_MAX;
    }
    __syncwarp();

    const uint64_t start_time = time_per_block != nullptr ? warp_kernel::get_global_timer_ns() : 0;
    const uint32_t begin = d_candidate_offsets[qid];
    const uint32_t count = d_candidate_sizes[qid];
    for (uint32_t idx = 0; idx < count; ++idx) {
        const uint32_t sample_id = d_candidate_ids[begin + idx];
        const float dist = score_rabitq_code_device<kUseInt8Score>(d_anchor_ids,
                                                    d_codes,
                                                    d_long_codes,
                                                    d_factors,
                                                    d_ex_factors,
                                                    d_rotated_query,
                                                    d_short_lut,
                                                    d_q_k1xsumq,
                                                    d_q_kbxsumq,
                                                    d_centroid_distances,
                                                    vector_query_idx,
                                                    sample_id,
                                                    num_anchors,
                                                    padded_dim,
                                                    code_bytes,
                                                    long_code_bytes,
                                                    bits_per_dim,
                                                    subtid,
                                                    d_quantized_query,
                                                    d_quantized_query_scales);
        if (tid == 0) {
            posting_shortlist_heap::insert<idx_t>(
                dist, sample_id, best_dist, best_ids, shortlist_k);
        }
        __syncwarp();
    }

    if (tid == 0) {
        posting_shortlist_heap::sort(best_dist, best_ids, shortlist_k);
    }
    __syncwarp();
    for (int i = tid; i < shortlist_k; i += blockDim.x) {
        d_result[static_cast<size_t>(qid) * shortlist_k + i] = best_ids[i];
        d_distances[static_cast<size_t>(qid) * shortlist_k + i] = best_dist[i];
    }
    __syncwarp();
    if (tid == 0) {
        if (time_per_block != nullptr) {
            const uint64_t stop_time = warp_kernel::get_global_timer_ns();
            const float elapsed_us = static_cast<float>(stop_time - start_time) / 1000.0f;
            time_per_block[static_cast<size_t>(route_idx) * 6] = elapsed_us;
            time_per_block[static_cast<size_t>(route_idx) * 6 + 1] = 0.0f;
            time_per_block[static_cast<size_t>(route_idx) * 6 + 2] = elapsed_us;
            time_per_block[static_cast<size_t>(route_idx) * 6 + 3] = 0.0f;
            time_per_block[static_cast<size_t>(route_idx) * 6 + 4] = static_cast<float>(count);
        }
    }
}

template <int kShortlistCapacity, bool kUseInt8Score>
__device__ void run_lowfreq_rabitq_resident_scan_one_block(const uint32_t* d_anchor_ids,
                                                           const uint8_t* d_codes,
                                                           const uint8_t* d_long_codes,
                                                           const float* d_factors,
                                                           const float* d_ex_factors,
                                                           const float* d_rotated_query,
                                                           const int8_t* d_quantized_query,
                                                           const float* d_quantized_query_scales,
                                                           const float* d_short_lut,
                                                           const float* d_q_k1xsumq,
                                                           const float* d_q_kbxsumq,
                                                           const float* d_centroid_distances,
                                                           uint32_t route_idx,
                                                           uint32_t vector_query_idx,
                                                           uint32_t qid,
                                                           idx_t* d_result,
                                                           float* d_distances,
                                                           const uint32_t* d_query_group_ids,
                                                           const uint64_t* d_query_filter_bits,
                                                           const uint8_t* d_query_range_modes,
                                                           const float* d_query_range_lower_bounds,
                                                           const float* d_query_range_upper_bounds,
                                                           const float* d_base_range_values,
                                                           uint32_t range_count,
                                                           const uint32_t* d_lowfreq_label_size,
                                                           const uint32_t* d_lowfreq_label_offset,
                                                           const uint32_t* d_lowfreq_sample_ids,
                                                           const uint64_t* d_lowfreq_row_filter_mask,
                                                           uint32_t lowfreq_group_count,
                                                           uint32_t lowfreq_total_rows,
                                                           uint32_t base_rows,
                                                           int topk,
                                                           int shortlist_k,
                                                           uint32_t num_anchors,
                                                           uint32_t padded_dim,
                                                           uint32_t code_bytes,
                                                           uint32_t long_code_bytes,
                                                           uint32_t bits_per_dim,
                                                           int use_filter_mask,
                                                           float* time_per_block,
                                                           gpu_predicate::View posting_predicate,
                                                       bool score_first_requested)
{
    const int tid = threadIdx.x;
    const int subtid = tid % warp_kernel::kWarpThreads;
    if (topk > kTopk || shortlist_k > kShortlistCapacity) {
        return;
    }

    __shared__ float best_dist[kShortlistCapacity];
    __shared__ idx_t best_ids[kShortlistCapacity];
    for (int i = tid; i < shortlist_k; i += blockDim.x) {
        best_dist[i] = FLT_MAX;
        best_ids[i] = UINT32_MAX;
    }
    __syncwarp();

    const uint64_t start_time = time_per_block != nullptr ? warp_kernel::get_global_timer_ns() : 0;
    uint32_t begin = 0;
    uint32_t count = 0;
    if (d_query_group_ids != nullptr &&
        d_lowfreq_label_size != nullptr &&
        d_lowfreq_label_offset != nullptr &&
        d_lowfreq_sample_ids != nullptr) {
        const uint32_t group_id = d_query_group_ids[route_idx];
        if (group_id < lowfreq_group_count) {
            begin = d_lowfreq_label_offset[group_id];
            count = d_lowfreq_label_size[group_id];
            if (begin > lowfreq_total_rows) {
                count = 0;
            } else if (count > lowfreq_total_rows - begin) {
                count = lowfreq_total_rows - begin;
            }
        }
    }

    const uint64_t query_filter_bit = d_query_filter_bits != nullptr
        ? d_query_filter_bits[route_idx]
        : 0ULL;
    uint32_t active_range_count = 0;
    if (subtid == 0) {
        active_range_count =
            go_fann::range_filter::count_active_ranges(d_query_range_modes,
                                                             route_idx,
                                                             range_count);
    }
    active_range_count = __shfl_sync(warp_kernel::kFullMask, active_range_count, 0);

    const bool score_first = score_first_requested &&
        gpu_predicate::enabled(posting_predicate, route_idx);
    for (uint32_t j = 0; j < count; ++j) {
        const uint32_t row = begin + j;
        const uint32_t sample_id = d_lowfreq_sample_ids[row];
        if (sample_id >= base_rows) {
            continue;
        }
        int sample_passes = 1;
        if (subtid == 0 && !score_first) {
            if (gpu_predicate::enabled(posting_predicate, route_idx)) {
                posting_predicate.base_ranges = d_base_range_values;
                sample_passes = gpu_predicate::matches(posting_predicate, route_idx, sample_id);
            } else {
                if (use_filter_mask != 0 && d_lowfreq_row_filter_mask != nullptr &&
                    (query_filter_bit == 0 ||
                     (d_lowfreq_row_filter_mask[row] & query_filter_bit) == 0ULL)) {
                    sample_passes = 0;
                }
                if (sample_passes != 0 && active_range_count != 0 &&
                    !lowfreq_basic_filter_passes(UINT64_MAX,
                                                 UINT64_MAX,
                                                 d_query_range_modes,
                                                 d_query_range_lower_bounds,
                                                 d_query_range_upper_bounds,
                                                 d_base_range_values,
                                                 route_idx,
                                                 sample_id,
                                                 range_count,
                                                 active_range_count)) {
                    sample_passes = 0;
                }
            }
        }
        sample_passes = __shfl_sync(warp_kernel::kFullMask, sample_passes, 0);
        if (sample_passes == 0) {
            continue;
        }
        const float dist = score_rabitq_code_device<kUseInt8Score>(d_anchor_ids,
                                                    d_codes,
                                                    d_long_codes,
                                                    d_factors,
                                                    d_ex_factors,
                                                    d_rotated_query,
                                                    d_short_lut,
                                                    d_q_k1xsumq,
                                                    d_q_kbxsumq,
                                                    d_centroid_distances,
                                                    vector_query_idx,
                                                    sample_id,
                                                    num_anchors,
                                                    padded_dim,
                                                    code_bytes,
                                                    long_code_bytes,
                                                    bits_per_dim,
                                                    subtid,
                                                    d_quantized_query,
                                                    d_quantized_query_scales);
        // The heap contains only predicate-valid rows. A losing score cannot
        // become competitive as that heap improves; all posting rows are still scanned.
        if (score_first) {
            if (subtid == 0) {
                sample_passes = posting_shortlist_heap::can_enter<idx_t>(
                    dist, sample_id, best_dist, best_ids, shortlist_k);
                if (sample_passes != 0) {
                    posting_predicate.base_ranges = d_base_range_values;
                    sample_passes = gpu_predicate::matches(posting_predicate, route_idx, sample_id);
                }
            }
            sample_passes = __shfl_sync(warp_kernel::kFullMask, sample_passes, 0);
            if (sample_passes == 0) continue;
        }
        if (tid == 0) {
            posting_shortlist_heap::insert<idx_t>(
                dist, sample_id, best_dist, best_ids, shortlist_k);
        }
        __syncwarp();
    }

    if (tid == 0) {
        posting_shortlist_heap::sort(best_dist, best_ids, shortlist_k);
    }
    __syncwarp();
    for (int i = tid; i < shortlist_k; i += blockDim.x) {
        d_result[static_cast<size_t>(qid) * shortlist_k + i] = best_ids[i];
        d_distances[static_cast<size_t>(qid) * shortlist_k + i] = best_dist[i];
    }
    __syncwarp();
    if (tid == 0) {
        if (time_per_block != nullptr) {
            const uint64_t stop_time = warp_kernel::get_global_timer_ns();
            const float elapsed_us = static_cast<float>(stop_time - start_time) / 1000.0f;
            time_per_block[static_cast<size_t>(route_idx) * 6] = elapsed_us;
            time_per_block[static_cast<size_t>(route_idx) * 6 + 1] = 0.0f;
            time_per_block[static_cast<size_t>(route_idx) * 6 + 2] = elapsed_us;
            time_per_block[static_cast<size_t>(route_idx) * 6 + 3] = 0.0f;
            time_per_block[static_cast<size_t>(route_idx) * 6 + 4] = static_cast<float>(count);
        }
    }
}

template <int kShortlistCapacity, bool kUseInt8Score>
__global__ void lowfreq_rabitq_scan_kernel(const uint32_t* d_anchor_ids,
                                           const uint8_t* d_codes,
                                           const uint8_t* d_long_codes,
                                           const float* d_factors,
                                           const float* d_ex_factors,
                                           const float* d_rotated_query,
                                           const int8_t* d_quantized_query,
                                           const float* d_quantized_query_scales,
                                           const float* d_short_lut,
                                           const float* d_q_k1xsumq,
                                           const float* d_q_kbxsumq,
                                           const float* d_centroid_distances,
                                           const uint32_t* d_query_ids,
                                           const uint32_t* d_route_to_original_query,
                                           idx_t* d_result,
                                           float* d_distances,
                                           const uint32_t* d_candidate_ids,
                                           const uint32_t* d_candidate_offsets,
                                           const uint32_t* d_candidate_sizes,
                                           int num_query,
                                           int topk,
                                           int shortlist_k,
                                           uint32_t num_anchors,
                                           uint32_t padded_dim,
                                           uint32_t code_bytes,
                                           uint32_t long_code_bytes,
                                           uint32_t bits_per_dim)
{
    const uint32_t qid = static_cast<uint32_t>(blockIdx.x);
    if (qid >= static_cast<uint32_t>(num_query)) {
        return;
    }
    const uint32_t route_idx = d_query_ids[qid];
    const uint32_t vector_query_idx = d_route_to_original_query[route_idx];
    run_lowfreq_rabitq_scan_one_block<kShortlistCapacity, kUseInt8Score>(
                                      d_anchor_ids,
                                      d_codes,
                                      d_long_codes,
                                      d_factors,
                                      d_ex_factors,
                                      d_rotated_query,
                                      d_quantized_query,
                                      d_quantized_query_scales,
                                      d_short_lut,
                                      d_q_k1xsumq,
                                      d_q_kbxsumq,
                                      d_centroid_distances,
                                      route_idx,
                                      vector_query_idx,
                                      qid,
                                      d_result,
                                      d_distances,
                                      d_candidate_ids,
                                      d_candidate_offsets,
                                      d_candidate_sizes,
                                      topk,
                                      shortlist_k,
                                      num_anchors,
                                      padded_dim,
                                      code_bytes,
                                      long_code_bytes,
                                      bits_per_dim,
                                      nullptr);
}

template <int kShortlistCapacity, bool kUseInt8Score>
__global__ void lowfreq_rabitq_resident_posting_kernel(const uint32_t* d_anchor_ids,
                                                       const uint8_t* d_codes,
                                                       const uint8_t* d_long_codes,
                                                       const float* d_factors,
                                                       const float* d_ex_factors,
                                                       const float* d_rotated_query,
                                                       const int8_t* d_quantized_query,
                                                       const float* d_quantized_query_scales,
                                                       const float* d_short_lut,
                                                       const float* d_q_k1xsumq,
                                                       const float* d_q_kbxsumq,
                                                       const float* d_centroid_distances,
                                                       const uint32_t* d_query_ids,
                                                       const uint32_t* d_route_to_original_query,
                                                       idx_t* d_result,
                                                       float* d_distances,
                                                       const uint32_t* d_query_group_ids,
                                                       const uint64_t* d_query_filter_bits,
                                                       const uint8_t* d_query_range_modes,
                                                       const float* d_query_range_lower_bounds,
                                                       const float* d_query_range_upper_bounds,
                                                       const float* d_base_range_values,
                                                       uint32_t range_count,
                                                       const uint32_t* d_lowfreq_label_size,
                                                       const uint32_t* d_lowfreq_label_offset,
                                                       const uint32_t* d_lowfreq_sample_ids,
                                                       const uint64_t* d_lowfreq_row_filter_mask,
                                                       uint32_t lowfreq_group_count,
                                                       uint32_t lowfreq_total_rows,
                                                       uint32_t base_rows,
                                                       int num_query,
                                                       int topk,
                                                       int shortlist_k,
                                                       uint32_t num_anchors,
                                                       uint32_t padded_dim,
                                                       uint32_t code_bytes,
                                                       uint32_t long_code_bytes,
                                                       uint32_t bits_per_dim,
                                                       int use_filter_mask,
                                                       uint32_t* d_ready,
                                                       gpu_predicate::View posting_predicate,
                                                       bool score_first_requested)
{
    const uint32_t qid = static_cast<uint32_t>(blockIdx.x);
    if (qid >= static_cast<uint32_t>(num_query)) {
        return;
    }
    const uint32_t route_idx = d_query_ids[qid];
    const uint32_t vector_query_idx = d_route_to_original_query[route_idx];
    run_lowfreq_rabitq_resident_scan_one_block<kShortlistCapacity, kUseInt8Score>(
        d_anchor_ids,
        d_codes,
        d_long_codes,
        d_factors,
        d_ex_factors,
        d_rotated_query,
        d_quantized_query,
        d_quantized_query_scales,
        d_short_lut,
        d_q_k1xsumq,
        d_q_kbxsumq,
        d_centroid_distances,
        route_idx,
        vector_query_idx,
        qid,
        d_result,
        d_distances,
        d_query_group_ids,
        d_query_filter_bits,
        d_query_range_modes,
        d_query_range_lower_bounds,
        d_query_range_upper_bounds,
        d_base_range_values,
        range_count,
        d_lowfreq_label_size,
        d_lowfreq_label_offset,
        d_lowfreq_sample_ids,
        d_lowfreq_row_filter_mask,
        lowfreq_group_count,
        lowfreq_total_rows,
        base_rows,
        topk,
        shortlist_k,
        num_anchors,
        padded_dim,
        code_bytes,
        long_code_bytes,
        bits_per_dim,
        use_filter_mask,
        nullptr,
        posting_predicate,
        score_first_requested);
    posting_completion::publish(d_ready, qid);
}

template <int kShortlistCapacity, bool kUseInt8Score>
__global__ void range_posting_rabitq_scan_kernel(const uint32_t* d_anchor_ids,
                                                 const uint8_t* d_codes,
                                                 const uint8_t* d_long_codes,
                                                 const float* d_factors,
                                                 const float* d_ex_factors,
                                                 const float* d_rotated_query,
                                                 const int8_t* d_quantized_query,
                                                 const float* d_quantized_query_scales,
                                                 const float* d_short_lut,
                                                 const float* d_q_k1xsumq,
                                                 const float* d_q_kbxsumq,
                                                 const float* d_centroid_distances,
                                                 const uint32_t* d_query_ids,
                                                 const uint32_t* d_route_to_original_query,
                                                 const uint32_t* d_posting_sample_ids,
                                                 const uint64_t* d_posting_offsets,
                                                 const uint64_t* d_posting_sizes,
                                                 const uint8_t* d_query_range_modes,
                                                 const float* d_query_range_lower_bounds,
                                                 const float* d_query_range_upper_bounds,
                                                 const float* d_base_range_values,
                                                 uint32_t range_count,
                                                 uint64_t resident_row_count,
                                                 uint32_t base_rows,
                                                 idx_t* d_result,
                                                 float* d_result_distances,
                                                 uint32_t* d_exact_pass_counts,
                                                 uint32_t query_count,
                                                 int shortlist_k,
                                                 uint32_t num_anchors,
                                                 uint32_t padded_dim,
                                                 uint32_t code_bytes,
                                                 uint32_t long_code_bytes,
                                                 uint32_t bits_per_dim,
                                                 posting_segments::Completion<idx_t> completion,
                                                 gpu_predicate::View posting_predicate,
                                                 bool score_first_requested,
                                                 float* completion_distances)
{
    const uint32_t qid = static_cast<uint32_t>(blockIdx.x);
    if (qid >= query_count || shortlist_k > kShortlistCapacity) {
        return;
    }
    const int tid = threadIdx.x;
    const int lane = tid % warp_kernel::kWarpThreads;
    __shared__ float best_dist[kShortlistCapacity];
    __shared__ idx_t best_ids[kShortlistCapacity];
    for (int i = tid; i < shortlist_k; i += blockDim.x) {
        best_dist[i] = FLT_MAX;
        best_ids[i] = UINT32_MAX;
    }
    __syncwarp();

    const uint32_t route_idx = d_query_ids[qid];
    const uint32_t vector_query_idx = d_route_to_original_query[route_idx];
    uint64_t begin = d_posting_offsets[qid];
    uint64_t count = d_posting_sizes[qid];
    if (begin > resident_row_count) {
        count = 0;
    } else if (count > resident_row_count - begin) {
        count = resident_row_count - begin;
    }
    uint32_t active_range_count = 0;
    if (lane == 0) {
        active_range_count =
            go_fann::range_filter::count_active_ranges(d_query_range_modes,
                                                             route_idx,
                                                             range_count);
    }
    active_range_count = __shfl_sync(warp_kernel::kFullMask, active_range_count, 0);
    const bool score_first = score_first_requested &&
        gpu_predicate::enabled(posting_predicate, route_idx);
    uint32_t exact_pass_count = 0;
    for (uint64_t posting_row = 0; posting_row < count; ++posting_row) {
        const uint32_t sample_id = d_posting_sample_ids[begin + posting_row];
        if (sample_id >= base_rows || active_range_count == 0) {
            continue;
        }
        int exact_pass = 1;
        if (lane == 0 && !score_first && gpu_predicate::enabled(posting_predicate, route_idx)) {
            posting_predicate.base_ranges = d_base_range_values;
            exact_pass = gpu_predicate::matches(posting_predicate, route_idx, sample_id);
        } else if (lane == 0 && !score_first && active_range_count > 1) {
            const uint32_t matched_active_dims =
                go_fann::range_filter::count_matching_active_ranges(
                    d_query_range_modes,
                    d_query_range_lower_bounds,
                    d_query_range_upper_bounds,
                    d_base_range_values,
                    route_idx,
                    sample_id,
                    range_count);
            exact_pass = matched_active_dims == active_range_count ? 1 : 0;
        }
        exact_pass = __shfl_sync(warp_kernel::kFullMask, exact_pass, 0);
        if (exact_pass == 0) {
            continue;
        }
        const float dist = score_rabitq_code_device<kUseInt8Score>(d_anchor_ids,
                                                    d_codes,
                                                    d_long_codes,
                                                    d_factors,
                                                    d_ex_factors,
                                                    d_rotated_query,
                                                    d_short_lut,
                                                    d_q_k1xsumq,
                                                    d_q_kbxsumq,
                                                    d_centroid_distances,
                                                    vector_query_idx,
                                                    sample_id,
                                                    num_anchors,
                                                    padded_dim,
                                                    code_bytes,
                                                    long_code_bytes,
                                                    bits_per_dim,
                                                    lane,
                                                    d_quantized_query,
                                                    d_quantized_query_scales);
        if (score_first) {
            if (lane == 0) {
                exact_pass = posting_shortlist_heap::can_enter<idx_t>(
                    dist, sample_id, best_dist, best_ids, shortlist_k);
                if (exact_pass != 0) {
                    posting_predicate.base_ranges = d_base_range_values;
                    exact_pass = gpu_predicate::matches(posting_predicate, route_idx, sample_id);
                }
            }
            exact_pass = __shfl_sync(warp_kernel::kFullMask, exact_pass, 0);
            if (exact_pass == 0) continue;
        }
        if (tid == 0) {
            ++exact_pass_count;
            posting_shortlist_heap::insert<idx_t>(
                dist, sample_id, best_dist, best_ids, shortlist_k);
        }
        __syncwarp();
    }
    for (int i = tid; i < shortlist_k; i += blockDim.x) {
        const size_t output = static_cast<size_t>(qid) * shortlist_k + i;
        d_result[output] = best_ids[i];
        if (d_result_distances != nullptr) d_result_distances[output] = best_dist[i];
    }
    if (tid == 0) {
        d_exact_pass_counts[qid] = exact_pass_count;
    }
    if (completion.merged_ids != nullptr) {
        // All lanes make their partial shortlist visible before counting this
        // segment complete. The last segment merges and publishes the whole query.
        __threadfence();
        __syncwarp();
        const uint32_t query = completion.query_positions[qid];
        const uint32_t parts = completion.query_indptr[query + 1] - completion.query_indptr[query];
        int last = 0;
        if (tid == 0) last = atomicAdd(completion.counters + query, 1U) + 1 == parts;
        last = __shfl_sync(warp_kernel::kFullMask, last, 0);
        if (last) {
            posting_segments::merge_one_query(query, completion.query_indptr, d_result,
                d_result_distances, d_exact_pass_counts, completion.merged_ids,
                completion.merged_pass_counts, shortlist_k, best_dist, best_ids);
            if (completion_distances != nullptr) {
                for (int i = tid; i < shortlist_k; i += blockDim.x)
                    completion_distances[static_cast<size_t>(query) * shortlist_k + i] = best_dist[i];
            }
            posting_completion::publish(completion.ready, query);
        }
    } else {
        posting_completion::publish(completion.ready, qid);
    }
}

template <int kShortlistCapacity, bool kUseInt8Score>
__global__ void candidate_stream_rabitq_scan_kernel(
    const uint32_t* d_anchor_ids,
    const uint8_t* d_codes,
    const uint8_t* d_long_codes,
    const float* d_factors,
    const float* d_ex_factors,
    const float* d_rotated_query,
    const int8_t* d_quantized_query,
    const float* d_quantized_query_scales,
    const float* d_short_lut,
    const float* d_q_k1xsumq,
    const float* d_q_kbxsumq,
    const float* d_centroid_distances,
    const uint32_t* d_query_ids,
    const uint32_t* d_route_to_original_query,
    uint32_t* d_payload_memory,
    volatile int64_t* d_status_memory,
    volatile int64_t* d_gdr_data,
    volatile int64_t* d_status_flag,
    uint32_t payload_slot_count,
    uint32_t payload_words,
    idx_t* d_result,
    uint32_t* d_candidate_counts,
    uint32_t query_count,
    int shortlist_k,
    uint32_t num_anchors,
    uint32_t padded_dim,
    uint32_t code_bytes,
    uint32_t long_code_bytes,
    uint32_t bits_per_dim)
{
    const uint32_t qid = static_cast<uint32_t>(blockIdx.x);
    if (qid >= query_count || shortlist_k > kShortlistCapacity ||
        payload_slot_count < 2 || payload_words <= kCandidatePayloadIdsWord) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const uint32_t route_idx = d_query_ids[qid];
    const uint32_t vector_query_idx = d_route_to_original_query[route_idx];
    __shared__ float best_dist[kShortlistCapacity];
    __shared__ idx_t best_ids[kShortlistCapacity];
    __shared__ uint32_t candidate_count;
    __shared__ uint32_t payload_flags;
    __shared__ uint32_t payload_generation;
    __shared__ uint64_t next_cursor;
    __shared__ uint32_t protocol_error;
    for (int i = lane; i < shortlist_k; i += blockDim.x) {
        best_dist[i] = FLT_MAX;
        best_ids[i] = UINT32_MAX;
    }
    __syncwarp();

    uint64_t cursor = 0;
    uint32_t buffer_idx = 0;
    uint32_t generation = 0;
    uint32_t total_candidates = 0;
    while (true) {
        if (lane == 0) {
            ++generation;
            d_gdr_data[static_cast<size_t>(route_idx) * kGdrMailboxStride] = 0;
            d_status_memory[static_cast<size_t>(route_idx) * 16] =
                static_cast<int64_t>(cursor);
            d_status_memory[static_cast<size_t>(route_idx) * 16 + 1] = buffer_idx;
            d_status_memory[static_cast<size_t>(route_idx) * 16 + 2] = generation;
            __threadfence_system();
            d_status_flag[static_cast<size_t>(route_idx) * 16] = 1;
        }
        __syncwarp();
        while (warp_kernel::load_bypass_l1(
                   &d_gdr_data[static_cast<size_t>(route_idx) * kGdrMailboxStride]) == 0) {
            warp_kernel::sleep_cuda_us(10.0f);
            __threadfence_system();
        }

        const size_t payload_slot =
            static_cast<size_t>(route_idx) * payload_slot_count + buffer_idx;
        const uint32_t* payload = d_payload_memory + payload_slot * payload_words;
        if (lane == 0) {
            candidate_count = load_bypass_l1_u32(
                payload + kCandidatePayloadCountWord);
            const uint64_t next_low = load_bypass_l1_u32(
                payload + kCandidatePayloadNextCursorLowWord);
            const uint64_t next_high = load_bypass_l1_u32(
                payload + kCandidatePayloadNextCursorHighWord);
            next_cursor = next_low | (next_high << 32U);
            payload_flags = load_bypass_l1_u32(
                payload + kCandidatePayloadFlagsWord);
            payload_generation = load_bypass_l1_u32(
                payload + kCandidatePayloadGenerationWord);
            protocol_error =
                payload_generation != generation || next_cursor < cursor ||
                candidate_count > payload_words - kCandidatePayloadIdsWord
                ? 1U
                : 0U;
        }
        __syncwarp();
        if (protocol_error != 0) {
            if (lane == 0) {
                d_status_flag[static_cast<size_t>(route_idx) * 16] = 2;
            }
            return;
        }

        for (uint32_t candidate_idx = 0; candidate_idx < candidate_count;
             ++candidate_idx) {
            const uint32_t sample_id = load_bypass_l1_u32(
                payload + kCandidatePayloadIdsWord + candidate_idx);
            const float dist = score_rabitq_code_device<kUseInt8Score>(
                d_anchor_ids,
                d_codes,
                d_long_codes,
                d_factors,
                d_ex_factors,
                d_rotated_query,
                d_short_lut,
                d_q_k1xsumq,
                d_q_kbxsumq,
                d_centroid_distances,
                vector_query_idx,
                sample_id,
                num_anchors,
                padded_dim,
                code_bytes,
                long_code_bytes,
                bits_per_dim,
                lane,
                d_quantized_query,
                d_quantized_query_scales);
            if (lane == 0) {
                posting_shortlist_heap::insert<idx_t>(dist,
                                               sample_id,
                                               best_dist,
                                               best_ids,
                                               shortlist_k);
            }
            __syncwarp();
        }
        total_candidates += candidate_count;
        if ((payload_flags & kCandidatePayloadFinal) != 0) break;
        cursor = next_cursor;
        buffer_idx ^= 1U;
    }

    for (int i = lane; i < shortlist_k; i += blockDim.x) {
        const size_t output = static_cast<size_t>(qid) * shortlist_k + i;
        d_result[output] = best_ids[i];
    }
    __syncwarp();
    if (lane == 0) {
        d_candidate_counts[qid] = total_candidates;
        __threadfence_system();
        d_status_flag[static_cast<size_t>(route_idx) * 16] = 3;
    }
}

__global__ void highfreq_float_gpu_l2_kernel(const float* candidate_vectors,
                                             const idx_t* candidate_ids,
                                             const uint32_t* candidate_counts,
                                             const uint32_t* original_query_ids,
                                             const float* original_queries,
                                             float* candidate_distances,
                                             uint32_t batch_size,
                                             uint32_t shortlist_k,
                                             uint32_t dim)
{
    const uint32_t flat_candidate = blockIdx.x;
    const uint32_t query_slot = flat_candidate / shortlist_k;
    const uint32_t candidate_slot = flat_candidate % shortlist_k;
    if (query_slot >= batch_size) {
        return;
    }

    __shared__ float partial[256];
    float sum = 0.0f;
    const bool valid = candidate_slot < candidate_counts[query_slot] &&
        candidate_ids[flat_candidate] != UINT32_MAX;
    if (valid) {
        const float* candidate = candidate_vectors +
            static_cast<size_t>(flat_candidate) * dim;
        const float* query = original_queries +
            static_cast<size_t>(original_query_ids[query_slot]) * dim;
        for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
            const float diff = query[d] - candidate[d];
            sum += diff * diff;
        }
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        candidate_distances[flat_candidate] = valid ? partial[0] : FLT_MAX;
    }
}

__global__ void highfreq_float_gpu_topk_kernel(const idx_t* candidate_ids,
                                               const float* candidate_distances,
                                               const uint32_t* candidate_counts,
                                               idx_t* result_ids,
                                               float* result_distances,
                                               uint32_t batch_size,
                                               uint32_t shortlist_k,
                                               uint32_t topk)
{
    const uint32_t query_slot = blockIdx.x;
    if (query_slot >= batch_size || threadIdx.x != 0) {
        return;
    }

    idx_t best_ids[kTopk];
    float best_distances[kTopk];
    for (uint32_t rank = 0; rank < static_cast<uint32_t>(kTopk); ++rank) {
        best_ids[rank] = UINT32_MAX;
        best_distances[rank] = FLT_MAX;
    }

    const uint32_t candidate_count = min(candidate_counts[query_slot], shortlist_k);
    const size_t candidate_base = static_cast<size_t>(query_slot) * shortlist_k;
    for (uint32_t candidate_slot = 0; candidate_slot < candidate_count; ++candidate_slot) {
        const idx_t sample_id = candidate_ids[candidate_base + candidate_slot];
        const float distance = candidate_distances[candidate_base + candidate_slot];
        if (sample_id == UINT32_MAX || !isfinite(distance)) {
            continue;
        }
        uint32_t insert_at = topk;
        for (uint32_t rank = 0; rank < topk; ++rank) {
            if (distance < best_distances[rank] ||
                (distance == best_distances[rank] && sample_id < best_ids[rank])) {
                insert_at = rank;
                break;
            }
        }
        if (insert_at == topk) {
            continue;
        }
        for (uint32_t rank = topk - 1; rank > insert_at; --rank) {
            best_ids[rank] = best_ids[rank - 1];
            best_distances[rank] = best_distances[rank - 1];
        }
        best_ids[insert_at] = sample_id;
        best_distances[insert_at] = distance;
    }

    const size_t result_base = static_cast<size_t>(query_slot) * topk;
    for (uint32_t rank = 0; rank < topk; ++rank) {
        result_ids[result_base + rank] = best_ids[rank];
        result_distances[result_base + rank] = best_distances[rank];
    }
}

struct RabitqFloatGpuRerankContext {
    bool initialized = false;
    uint32_t batch_capacity = 0;
    uint32_t shortlist_capacity = 0;
    uint32_t topk_capacity = 0;
    uint32_t dim = 0;
    float* h_candidate_vectors = nullptr;
    idx_t* h_candidate_ids = nullptr;
    uint32_t* h_candidate_counts = nullptr;
    uint32_t* h_original_query_ids = nullptr;
    idx_t* h_result_ids = nullptr;
    float* h_result_distances = nullptr;
    float* d_candidate_vectors = nullptr;
    idx_t* d_candidate_ids = nullptr;
    uint32_t* d_candidate_counts = nullptr;
    uint32_t* d_original_query_ids = nullptr;
    float* d_candidate_distances = nullptr;
    idx_t* d_result_ids = nullptr;
    float* d_result_distances = nullptr;
    cudaStream_t stream = nullptr;
};

struct RabitqImmutableDeviceState {
    uint32_t* d_anchor_ids = nullptr;
    uint8_t* d_codes = nullptr;
    uint8_t* d_long_codes = nullptr;
    float* d_factors = nullptr;
    float* d_ex_factors = nullptr;
};

struct RabitqPreparedQueryState {
    value_t* d_query_u8 = nullptr;
    float* d_query_float = nullptr;
    uint32_t* d_route_to_original_query = nullptr;
    float* d_rotation = nullptr;
    float* d_anchors = nullptr;
    float* d_rotated_query = nullptr;
    int8_t* d_quantized_query = nullptr;
    float* d_quantized_query_scales = nullptr;
    float* d_q_k1xsumq = nullptr;
    float* d_q_kbxsumq = nullptr;
    float* d_centroid_distances = nullptr;
    float* d_query_norms = nullptr;
    float* d_anchor_norms = nullptr;
    float* d_short_lut = nullptr;
    size_t routed_query_count = 0;
    size_t original_query_count = 0;
    size_t short_lut_entries_per_query = 0;
    bool owns_shared_query_buffers = true;
};

struct RabitqPreparedQueryTiming {
    double total_ms = 0.0;
    double validation_ms = 0.0;
    double buffer_allocate_upload_ms = 0.0;
    double host_u8_to_float_detail_ms = 0.0;
    double gpu_preprocess_ms = 0.0;
    double rotation_engine_setup_ms = 0.0;
    double rotation_device_ms = 0.0;
    double quantize_device_ms = 0.0;
    double lut_device_ms = 0.0;
    double factors_device_ms = 0.0;
    double centroid_device_ms = 0.0;
    std::string rotation_method;
    std::string centroid_method;
    double static_rotation_anchors_ms = 0.0;
};

struct RabitqPostingStreamBuffer {
    size_t slots = 0;
    size_t rows = 0;
    idx_t* h_ids = nullptr;
    idx_t* d_ids = nullptr;
    float* h_distances = nullptr;
    float* d_distances = nullptr;
    uint32_t* h_ready = nullptr;
    uint32_t* d_ready = nullptr;
    std::vector<uint32_t> routes;
    posting_completion::Groups groups;
};

void cleanup_posting_stream_buffer(RabitqPostingStreamBuffer& buffer)
{
    if (buffer.h_ids != nullptr) cudaFreeHost(buffer.h_ids);
    if (buffer.h_distances != nullptr) cudaFreeHost(buffer.h_distances);
    if (buffer.h_ready != nullptr) cudaFreeHost(buffer.h_ready);
    buffer = RabitqPostingStreamBuffer{};
}

void prepare_posting_stream_buffer(RabitqPostingStreamBuffer& buffer,
                                   const GraphRuntime& runtime,
                                   const std::vector<uint32_t>& routes,
                                   uint32_t shortlist_k)
{
    const size_t slots = checked_size_product(routes.size(), shortlist_k, "posting stream slots");
    if (buffer.slots < slots || buffer.rows < routes.size()) {
        cleanup_posting_stream_buffer(buffer);
        try {
            const auto allocate = [](auto*& host, auto*& device, size_t count) {
                check_cuda(cudaHostAlloc(&host, count * sizeof(*host), cudaHostAllocMapped),
                           "cudaHostAlloc posting completion buffer");
                check_cuda(cudaHostGetDevicePointer(&device, host, 0),
                           "cudaHostGetDevicePointer posting completion buffer");
            };
            allocate(buffer.h_ids, buffer.d_ids, slots);
            allocate(buffer.h_distances, buffer.d_distances, slots);
            allocate(buffer.h_ready, buffer.d_ready, routes.size());
            buffer.slots = slots;
            buffer.rows = routes.size();
        } catch (...) {
            cleanup_posting_stream_buffer(buffer);
            throw;
        }
    }
    if (buffer.routes != routes) {
        std::vector<uint32_t> original_ids;
        original_ids.reserve(routes.size());
        for (uint32_t route : routes) original_ids.push_back(highfreq_route_original_query_id(runtime, route));
        buffer.groups = posting_completion::Groups::make(original_ids, runtime.original_query_count);
        buffer.routes = routes;
    }
    std::fill_n(buffer.h_ready, routes.size(), 0U);
    _mm_sfence();
}

// Cancel a consumer on a failed GPU launch/synchronization, and join before any
// captured buffers can be destroyed. Successful runs may transfer the future.
struct RabitqPostingStreamRun {
    std::shared_ptr<std::atomic<bool>> cancelled = std::make_shared<std::atomic<bool>>(false);
    std::future<void> future;
    ~RabitqPostingStreamRun()
    {
        if (future.valid()) {
            cancelled->store(true, std::memory_order_release);
            future.wait();
        }
    }
};

struct RabitqLowfreqSearchContext {
    RabitqPostingStreamBuffer completion;
    idx_t* d_result = nullptr;
    float* d_distances = nullptr;
    uint32_t* d_query_ids = nullptr;
    uint32_t* d_candidate_ids = nullptr;
    uint32_t* d_candidate_offsets = nullptr;
    uint32_t* d_candidate_sizes = nullptr;
    uint32_t* d_query_group_ids = nullptr;
    uint32_t* d_label_size = nullptr;
    uint32_t* d_label_offset = nullptr;
    uint32_t* d_sample_ids = nullptr;
    size_t query_capacity = 0;
    size_t candidate_capacity = 0;
    uint32_t shortlist_capacity = 0;
    size_t query_group_capacity = 0;
    size_t resident_group_capacity = 0;
    size_t resident_row_capacity = 0;
    uint32_t resident_group_count = 0;
    uint32_t resident_total_rows = 0;
    bool resident_posting_ready = false;
    bool execution_mode_reported = false;
    bool kernel_resources_reported = false;
    std::vector<idx_t> host_shortlist;
    std::vector<float> host_approximate_distances;
};

struct RabitqRangePostingPlan {
    range_posting::Mode mode = range_posting::Mode::Off;
    std::vector<uint32_t> original_query_ids;
    std::vector<uint32_t> representative_route_ids;
    std::vector<uint64_t> posting_offsets;
    std::vector<uint64_t> posting_sizes;
    std::vector<uint32_t> posting_range_indices;
    std::vector<uint32_t> resident_plan_positions;
    std::vector<uint32_t> resident_postfilter_plan_positions;
    std::vector<uint32_t> stream_plan_positions;
    std::vector<uint8_t> selected_original_queries;
    std::vector<uint8_t> selected_routes;
    std::vector<uint8_t> replaced_routes;
    std::vector<uint8_t> dropped_wide_hybrid_or_original_queries;
    std::vector<uint8_t> selected_range_graph_and_original_queries;
    uint64_t dropped_wide_hybrid_or_routes = 0;
    uint64_t dropped_wide_hybrid_or_rows = 0;
    uint64_t selected_range_graph_and_queries = 0;
    uint64_t selected_range_graph_and_rows = 0;
    uint64_t replaced_range_graph_and_label_routes = 0;
    uint64_t total_rows_scanned = 0;
};

struct RabitqPostingSegmentDevicePlan {
    uint32_t count = 0;
    uint32_t* d_route_ids = nullptr;
    uint64_t* d_offsets = nullptr;
    uint64_t* d_sizes = nullptr;
    uint32_t* d_query_indptr = nullptr;
    uint32_t* d_query_positions = nullptr;
};

struct RabitqRangePostingSearchContext {
    RabitqPostingStreamBuffer resident_completion;
    RabitqPostingStreamBuffer postfilter_completion;
    bool initialized = false;
    uint32_t* d_sample_ids = nullptr;
    uint32_t* d_query_ids = nullptr;
    uint64_t* d_posting_offsets = nullptr;
    uint64_t* d_posting_sizes = nullptr;
    uint32_t* d_exact_pass_counts = nullptr;
    idx_t* d_result = nullptr;
    uint32_t shortlist_k = 0;
    uint32_t postfilter_shortlist_k = 0;
    size_t device_query_capacity = 0;
    size_t result_slot_capacity = 0;
    uint64_t resident_row_count = 0;
    RabitqRangePostingPlan plan;
    RabitqPostingSegmentDevicePlan resident_segments;
    RabitqPostingSegmentDevicePlan postfilter_segments;
    idx_t* d_segment_ids = nullptr;
    float* d_segment_distances = nullptr;
    uint32_t* d_segment_pass_counts = nullptr;
    uint32_t* d_segment_completion_counts = nullptr;
    cudaEvent_t segment_start = nullptr;
    cudaEvent_t segment_scan_end = nullptr;
    cudaEvent_t segment_merge_end = nullptr;
    std::vector<idx_t> host_shortlist;
    std::vector<uint32_t> host_exact_pass_counts;
};

struct RabitqRangePostingRunStats {
    uint64_t scanned_rows = 0;
    uint64_t exact_pass_rows = 0;
    bool exact_pass_rows_available = true;
    uint64_t shortlist_rows = 0;
    uint64_t postfilter_shortlist_rows = 0;
    uint64_t postfilter_residual_pass_rows = 0;
    uint64_t exact_rerank_rows = 0;
    uint64_t resident_routes = 0;
    uint64_t streamed_routes = 0;
    uint64_t stream_chunks = 0;
    int ordinary_active_blocks_per_sm = -1;
    int postfilter_active_blocks_per_sm = -1;
    uint32_t ordinary_shortlist_capacity = 0;
    uint32_t postfilter_shortlist_capacity = 0;
    uint64_t ordinary_shared_bytes = 0;
    uint64_t postfilter_shared_bytes = 0;
    double kernel_wall_ms = 0.0;
    double shortlist_d2h_ms = 0.0;
    double rerank_queue_wait_ms = 0.0;
    double rerank_wall_ms = 0.0;
    double total_wall_ms = 0.0;
};

struct RabitqRangePostingPendingSubset {
    std::vector<uint32_t> plan_positions;
    std::vector<uint32_t> route_ids;
    std::vector<idx_t> shortlist;
    uint32_t shortlist_k = 0;
    bool filter_full_labels = false;
    bool gpu_predicate_executed = false;
};

struct RabitqLowfreqHostBufferStatus {
    size_t required_slots = 0;
    bool shortlist_reused = false;
    bool approximate_distances_required = false;
    bool approximate_distances_reused = false;
};

struct GpuPredicateDeviceIndex {
    gpu_predicate::HostIndex host;
    uint32_t* directory = nullptr;
    gpu_predicate::Descriptor* descriptors = nullptr;
    uint16_t* payload = nullptr;
    std::vector<uint32_t> uploaded_labels;
    uint32_t* csr_offsets = nullptr;
    uint32_t* csr_labels = nullptr;
    size_t resident_bytes = 0;
    int device = 0;
    ~GpuPredicateDeviceIndex() {
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        cudaSetDevice(device);
        cudaFree(directory); cudaFree(descriptors); cudaFree(payload);
        cudaFree(csr_offsets); cudaFree(csr_labels);
        cudaSetDevice(previous);
    }
};

struct RabitqPostingPredicateContext {
    std::shared_ptr<GpuPredicateDeviceIndex> index;
    gpu_predicate::View view;
    uint32_t* offsets = nullptr;
    uint32_t* handles = nullptr;
    gpu_predicate::Clause* clauses = nullptr;
    gpu_predicate::RangeAtom* ranges = nullptr;
    uint8_t* enabled = nullptr;
    std::vector<uint8_t> verified_routes;
};

void cleanup_posting_predicate_context(RabitqPostingPredicateContext& ctx)
{
    cudaFree(ctx.offsets); cudaFree(ctx.handles); cudaFree(ctx.clauses);
    cudaFree(ctx.ranges); cudaFree(ctx.enabled);
    ctx = RabitqPostingPredicateContext{};
}

struct RabitqGraphSearchContext {
    RabitqPostingPredicateContext label_posting_predicate;
    RabitqPostingPredicateContext range_posting_predicate;
    std::shared_ptr<GpuPredicateDeviceIndex> predicate_index;
    gpu_predicate::View predicate_view;
    uint32_t* predicate_offsets = nullptr;
    uint32_t* predicate_handles = nullptr;
    gpu_predicate::Clause* predicate_clauses = nullptr;
    gpu_predicate::RangeAtom* predicate_ranges = nullptr;
    uint8_t* predicate_enabled = nullptr;
    uint8_t* predicate_prefetch_disabled = nullptr;
    uint8_t* predicate_compiled = nullptr;
    go_fann::route_predicate::Plan predicate_plan;
    std::vector<uint8_t> predicate_verified_routes;
    bool predicate_bias_frontier = false;
    gpu_predicate::Counters* predicate_counters = nullptr;
    bool shortlist_predicate_verified = false;
    bool initialized = false;
    bool use_inline_gpu_exact_rerank = false;
    bool result_buffer_is_host_mapped = false;
    bool using_host_mapped_neighbor = false;
    bool mailbox_rtt_profile = false;
    bool phase_snapshot_profile = false;
    bool request_ring_enabled = false;
    bool request_ring_consumed_device_mapped = false;
    bool device_resident_graph = false;
    bool gpu_graph_row_cache_enabled = false;
    bool owns_gpu_graph_row_cache = true;
    bool mh_pinned = false;
    bool mh_neighbor_pinned = false;
    bool mh_request_ring_consumed_sequences_pinned = false;
    const GraphRuntime* hybrid_residual_runtime = nullptr;
    std::shared_ptr<RabitqCpuWorkerBudget> cpu_worker_budget;
    std::shared_ptr<RabitqCpuWorkerPool> cpu_rerank_workers;
    std::shared_ptr<go_fann::ExactDistanceMemo> exact_distance_memo;
    std::shared_ptr<NativeIncrementalRerank> incremental_rerank;
    AsyncIoConfig io_config{};

    size_t total_query_capacity = 0;
    size_t graph_scratch_capacity = 0;
    size_t graph_hash_scratch_bytes = 0;
    size_t graph_datadist_scratch_bytes = 0;
    uint32_t result_width = 0;
    uint32_t stop_width = 0;
    uint32_t device_range_count = 0;
    size_t mapped_gdr_bytes = 0;
    size_t mapped_neighbor_bytes = 0;
    size_t mapped_request_ring_consumed_sequences_bytes = 0;
    size_t request_ring_capacity_per_ring = 0;
    size_t request_ring_capacity_count = 0;
    uint32_t gpu_graph_row_cache_capacity = 0;
    uint32_t resident_graph_degree = 0;
    RabitqFloatGpuRerankContext float_gpu_rerank;

    idx_t* d_result = nullptr;
    dist_t* d_result_distances = nullptr;
    idx_t* h_result_mapped = nullptr;
    dist_t* h_result_distances_mapped = nullptr;
    dist_t* d_datadist_list = nullptr;
    uint32_t* d_hashdata = nullptr;
    float* d_time_per_block = nullptr;
    float* h_time_per_block_mapped = nullptr;
    uint16_t* d_phase_state = nullptr;
    uint64_t* d_mailbox_rtt_hist = nullptr;
    uint32_t* d_query_ids = nullptr;
    uint32_t* d_query_group_ids = nullptr;
    uint32_t* d_start_local_ids = nullptr;
    uint32_t* d_row_to_sample = nullptr;
    uint32_t* d_label_offset = nullptr;
    uint64_t* d_row_filter_mask = nullptr;
    uint64_t* h_row_filter_mask_mapped = nullptr;
    uint64_t* d_query_filter_bits = nullptr;
    uint32_t* d_resident_graph = nullptr;
    RabitqGpuGraphRowCacheEntry* d_gpu_graph_row_cache = nullptr;
    uint32_t* d_gpu_query_label_indptr = nullptr;
    uint32_t* d_gpu_query_label_indices = nullptr;
    uint32_t* d_gpu_sample_label_indptr = nullptr;
    uint32_t* d_gpu_sample_label_indices = nullptr;
    RabitqGpuGraphCacheDeviceView* d_gpu_graph_cache_view = nullptr;
    uint8_t* d_query_range_modes = nullptr;
    float* d_query_range_lower_bounds = nullptr;
    float* d_query_range_upper_bounds = nullptr;
    float* d_base_range_values = nullptr;
    size_t base_range_value_count = 0;
    bool owns_base_range_values = true;
    uint32_t* d_prefetch_index = nullptr;
    uint32_t* d_prefetch_begin_by_group = nullptr;
    uint32_t* d_prefetch_end_by_group = nullptr;
    Node* d_prefetch_node_array = nullptr;
    int64_t* d_gdr_data_raw = nullptr;
    int64_t* d_gdr_data = nullptr;
    int64_t* status_memory = nullptr;
    int64_t* status_flag = nullptr;
    int64_t* d_status_memory = nullptr;
    int64_t* d_status_flag = nullptr;
    HostRequestRingEntry* h_request_ring_entries = nullptr;
    HostRequestRingEntry* d_request_ring_entries = nullptr;
    uint64_t* h_request_ring_consumed_sequences = nullptr;
    uint64_t* d_request_ring_consumed_sequences = nullptr;
    uint64_t* d_request_ring_consumed_sequences_raw = nullptr;
    unsigned long long* d_request_ring_tails = nullptr;
    uint32_t* d_neighbor_memory_raw = nullptr;
    uint32_t* d_neighbor_memory = nullptr;
    uint32_t* h_neighbor_memory = nullptr;
    uint32_t* h_neighbor_memory_host = nullptr;
    int64_t* h_gdr_data = nullptr;
    gdr_t g = nullptr;
    gdr_mh_t mh{};
    gdr_mh_t mh_neighbor{};
    gdr_mh_t mh_request_ring_consumed_sequences{};
    void* cpu_ptr = nullptr;
    void* cpu_ptr_neighbor = nullptr;
    void* cpu_ptr_request_ring_consumed_sequences = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    HostFilterStats host_filter_stats{};
    bool row_filter_mask_host_mapped = false;
    bool finite_query_state_enabled = false;
    uint32_t finite_queue_capacity = 0;
    uint32_t finite_result_capacity = 0;
    uint32_t finite_context_epoch = 0;
    size_t finite_mutable_bytes_per_query = 0;
    size_t finite_mutable_bytes_total = 0;
    size_t finite_ready_token_capacity = 0;
    RabitqFiniteContextControl* d_finite_contexts = nullptr;
    RabitqPoolPair* d_finite_frontier_state = nullptr;
    RabitqStopPair* d_finite_stop_state = nullptr;
    RabitqResultPair* d_finite_result_state = nullptr;
    RabitqFiniteReadyToken* d_finite_ready_tokens = nullptr;
};

void prepare_cpu_rerank_workers(RabitqGraphSearchContext& ctx,
                                const RabitqSearchOptions& options,
                                const RabitqGraphSearchContext* shared_ctx = nullptr)
{
    if (!options.graph_persistent_workers && !options.posting_persistent_workers) return;
    if (shared_ctx != nullptr && shared_ctx->cpu_rerank_workers != nullptr) {
        ctx.cpu_rerank_workers = shared_ctx->cpu_rerank_workers;
    }
    if (ctx.cpu_rerank_workers == nullptr) {
        const auto start = std::chrono::steady_clock::now();
        ctx.cpu_rerank_workers = std::make_shared<RabitqCpuWorkerPool>(
            static_cast<size_t>(std::max(1, options.highfreq_rerank_threads)));
        std::printf("[cpu-rerank-worker-pool] threads=%zu setup_ms=%.3f graph=%d posting=%d\n",
                    ctx.cpu_rerank_workers->capacity(),
                    elapsed_ms(start, std::chrono::steady_clock::now()),
                    options.graph_persistent_workers ? 1 : 0,
                    options.posting_persistent_workers ? 1 : 0);
    }
}

struct RabitqPhaseSnapshotSample {
    uint64_t wall_time_ns = 0;
    double copy_us = 0.0;
    std::vector<std::array<uint32_t, 5>> counts_by_sm;
};

struct RabitqPhaseSnapshotSampler {
    std::atomic<bool> stop{false};
    std::thread worker;
    std::exception_ptr error;
    std::vector<uint32_t> query_indices;
    std::vector<RabitqPhaseSnapshotSample> samples;
    uint16_t* device_state = nullptr;
    size_t state_count = 0;
    int device_id = 0;
    uint32_t sm_count = 0;
    uint32_t interval_us = 0;
    std::filesystem::path output_path;
};

std::atomic<uint64_t> rabitq_phase_snapshot_sequence{0};

uint64_t system_wall_time_ns()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

void run_rabitq_phase_snapshot_sampler(RabitqPhaseSnapshotSampler* sampler)
{
    uint16_t* host_state = nullptr;
    cudaStream_t stream = nullptr;
    try {
        check_cuda(cudaSetDevice(sampler->device_id),
                   "cudaSetDevice RabitQ phase snapshot sampler");
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&host_state),
                                 sampler->state_count * sizeof(uint16_t),
                                 cudaHostAllocPortable),
                   "cudaHostAlloc RabitQ phase snapshot sampler");
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreate RabitQ phase snapshot sampler");
        while (!sampler->stop.load(std::memory_order_acquire)) {
            const auto copy_start = std::chrono::steady_clock::now();
            check_cuda(cudaMemcpyAsync(host_state,
                                       sampler->device_state,
                                       sampler->state_count * sizeof(uint16_t),
                                       cudaMemcpyDeviceToHost,
                                       stream),
                       "cudaMemcpyAsync RabitQ phase snapshot sampler");
            check_cuda(cudaStreamSynchronize(stream),
                       "cudaStreamSynchronize RabitQ phase snapshot sampler");
            RabitqPhaseSnapshotSample sample;
            sample.wall_time_ns = system_wall_time_ns();
            sample.copy_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - copy_start).count();
            sample.counts_by_sm.resize(sampler->sm_count);
            for (uint32_t query_idx : sampler->query_indices) {
                const uint16_t packed = host_state[query_idx];
                const uint32_t phase = packed & 0xffU;
                const uint32_t sm_id = packed >> 8;
                if (phase > static_cast<uint32_t>(RabitqWarpPhase::Done) ||
                    sm_id >= sampler->sm_count) {
                    continue;
                }
                ++sample.counts_by_sm[sm_id][phase];
            }
            sampler->samples.push_back(std::move(sample));
            std::this_thread::sleep_for(std::chrono::microseconds(sampler->interval_us));
        }
    } catch (...) {
        sampler->error = std::current_exception();
    }
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (host_state != nullptr) cudaFreeHost(host_state);
}

std::unique_ptr<RabitqPhaseSnapshotSampler> start_rabitq_phase_snapshot_sampler(
    const RabitqGraphSearchContext& ctx,
    const std::vector<uint32_t>& query_indices)
{
    if (!ctx.phase_snapshot_profile || ctx.d_phase_state == nullptr || query_indices.empty()) {
        return nullptr;
    }
    auto sampler = std::make_unique<RabitqPhaseSnapshotSampler>();
    sampler->query_indices = query_indices;
    sampler->device_state = ctx.d_phase_state;
    sampler->state_count = ctx.total_query_capacity;
    sampler->interval_us = graph_phase_snapshot_interval_us();
    check_cuda(cudaGetDevice(&sampler->device_id),
               "cudaGetDevice RabitQ phase snapshot sampler");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, sampler->device_id),
               "cudaGetDeviceProperties RabitQ phase snapshot sampler");
    sampler->sm_count = static_cast<uint32_t>(properties.multiProcessorCount);
    for (uint32_t query_idx : query_indices) {
        if (query_idx >= sampler->state_count) {
            fail_rabitq("RabitQ phase snapshot query index exceeds state capacity");
        }
    }
    const char* output_dir = std::getenv("GO_FANN_RABITQ_PHASE_SNAPSHOT_DIR");
    std::filesystem::create_directories(output_dir);
    const uint64_t sequence = rabitq_phase_snapshot_sequence.fetch_add(
        1, std::memory_order_relaxed);
    sampler->output_path = std::filesystem::path(output_dir) /
        ("phase_snapshot." + std::to_string(static_cast<long long>(getpid())) + "." +
         std::to_string(sequence) + ".csv");
    sampler->worker = std::thread(run_rabitq_phase_snapshot_sampler, sampler.get());
    return sampler;
}

void finish_rabitq_phase_snapshot_sampler(
    std::unique_ptr<RabitqPhaseSnapshotSampler>& sampler)
{
    if (sampler == nullptr) return;
    sampler->stop.store(true, std::memory_order_release);
    if (sampler->worker.joinable()) sampler->worker.join();
    if (sampler->error != nullptr) std::rethrow_exception(sampler->error);
    std::ofstream output(sampler->output_path);
    if (!output) {
        fail_rabitq("failed to open RabitQ phase snapshot output");
    }
    output << "sample,wall_time_ns,copy_us,sm_id,inactive,wait,score,frontier,done\n";
    for (size_t sample_idx = 0; sample_idx < sampler->samples.size(); ++sample_idx) {
        const RabitqPhaseSnapshotSample& sample = sampler->samples[sample_idx];
        for (uint32_t sm_id = 0; sm_id < sampler->sm_count; ++sm_id) {
            const auto& counts = sample.counts_by_sm[sm_id];
            output << sample_idx << ',' << sample.wall_time_ns << ','
                   << std::fixed << std::setprecision(3) << sample.copy_us << ','
                   << sm_id;
            for (uint32_t count : counts) output << ',' << count;
            output << '\n';
        }
    }
    output.close();
    if (!output) fail_rabitq("failed to write RabitQ phase snapshot output");
    std::printf("[rabitq phase snapshot] file=%s samples=%zu active_queries=%zu "
                "state_count=%zu sm_count=%u interval_us=%u\n",
                sampler->output_path.c_str(),
                sampler->samples.size(),
                sampler->query_indices.size(),
                sampler->state_count,
                sampler->sm_count,
                sampler->interval_us);
    sampler.reset();
}

void cancel_rabitq_phase_snapshot_sampler(
    std::unique_ptr<RabitqPhaseSnapshotSampler>& sampler) noexcept
{
    if (sampler == nullptr) return;
    sampler->stop.store(true, std::memory_order_release);
    if (sampler->worker.joinable()) sampler->worker.join();
    sampler.reset();
}

struct RabitqPersistentLatencyQueue {
    uint32_t slot_count = 0;
    RabitqPersistentLatencyRequestControl* d_requests_raw = nullptr;
    RabitqPersistentLatencyRequestControl* d_requests = nullptr;
    RabitqPersistentLatencyRequestControl* h_requests = nullptr;
    RabitqPersistentLatencyCompletionControl* h_completions = nullptr;
    RabitqPersistentLatencyCompletionControl* d_completions = nullptr;
    idx_t* d_lowfreq_result = nullptr;
    idx_t* h_lowfreq_result_mapped = nullptr;
    float* d_lowfreq_distances = nullptr;
    float* h_lowfreq_distances_mapped = nullptr;
    uint32_t* d_lowfreq_candidate_ids = nullptr;
    uint32_t* h_lowfreq_candidate_ids_mapped = nullptr;
    uint32_t* d_lowfreq_candidate_offsets = nullptr;
    uint32_t* h_lowfreq_candidate_offsets_mapped = nullptr;
    uint32_t* d_lowfreq_candidate_sizes = nullptr;
    uint32_t* h_lowfreq_candidate_sizes_mapped = nullptr;
    size_t lowfreq_candidate_capacity = 0;
    uint32_t* d_lowfreq_label_size = nullptr;
    uint32_t* d_lowfreq_label_offset = nullptr;
    uint32_t* d_lowfreq_sample_ids = nullptr;
    uint64_t* d_lowfreq_row_filter_mask = nullptr;
    uint32_t lowfreq_group_count = 0;
    uint32_t lowfreq_total_rows = 0;
    uint32_t base_rows = 0;
    uint32_t lowfreq_shortlist_k = 0;
    bool lowfreq_candidate_mode = false;
    gdr_t g = nullptr;
    gdr_mh_t mh{};
    void* cpu_ptr = nullptr;
    size_t mapped_bytes = 0;
    bool pinned = false;
};

struct RabitqPersistentLatencySlot {
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

struct RabitqPersistentLatencyMeasurement {
    uint32_t query_id = UINT32_MAX;
    bool lowfreq_route = false;
    double total_ms = 0.0;
    double gpu_ms = 0.0;
};

struct RabitqLatencyResultSidecars {
    std::vector<std::vector<idx_t>> route_results;
    std::vector<std::vector<float>> route_distances;
    std::vector<int> route_steps;
    std::vector<uint8_t> route_measured;
    std::vector<uint8_t> route_result_ready;
};

void cleanup_immutable_state(RabitqImmutableDeviceState& state)
{
    if (state.d_anchor_ids != nullptr) cudaFree(state.d_anchor_ids);
    if (state.d_codes != nullptr) cudaFree(state.d_codes);
    if (state.d_long_codes != nullptr) cudaFree(state.d_long_codes);
    if (state.d_factors != nullptr) cudaFree(state.d_factors);
    if (state.d_ex_factors != nullptr) cudaFree(state.d_ex_factors);
    state = RabitqImmutableDeviceState{};
}

void cleanup_prepared_query_state(RabitqPreparedQueryState& state)
{
    if (state.d_route_to_original_query != nullptr) cudaFree(state.d_route_to_original_query);
    if (state.owns_shared_query_buffers) {
        if (state.d_query_u8 != nullptr) cudaFree(state.d_query_u8);
        if (state.d_query_float != nullptr) cudaFree(state.d_query_float);
        if (state.d_rotation != nullptr) cudaFree(state.d_rotation);
        if (state.d_anchors != nullptr) cudaFree(state.d_anchors);
        if (state.d_rotated_query != nullptr) cudaFree(state.d_rotated_query);
        if (state.d_quantized_query != nullptr) cudaFree(state.d_quantized_query);
        if (state.d_quantized_query_scales != nullptr) cudaFree(state.d_quantized_query_scales);
        if (state.d_q_k1xsumq != nullptr) cudaFree(state.d_q_k1xsumq);
        if (state.d_q_kbxsumq != nullptr) cudaFree(state.d_q_kbxsumq);
        if (state.d_centroid_distances != nullptr) cudaFree(state.d_centroid_distances);
        if (state.d_query_norms != nullptr) cudaFree(state.d_query_norms);
        if (state.d_anchor_norms != nullptr) cudaFree(state.d_anchor_norms);
        if (state.d_short_lut != nullptr) cudaFree(state.d_short_lut);
    }
    state = RabitqPreparedQueryState{};
}

void cleanup_float_gpu_rerank_context(RabitqFloatGpuRerankContext& ctx)
{
    if (ctx.stream != nullptr) cudaStreamDestroy(ctx.stream);
    if (ctx.h_candidate_vectors != nullptr) cudaFreeHost(ctx.h_candidate_vectors);
    if (ctx.h_candidate_ids != nullptr) cudaFreeHost(ctx.h_candidate_ids);
    if (ctx.h_candidate_counts != nullptr) cudaFreeHost(ctx.h_candidate_counts);
    if (ctx.h_original_query_ids != nullptr) cudaFreeHost(ctx.h_original_query_ids);
    if (ctx.h_result_ids != nullptr) cudaFreeHost(ctx.h_result_ids);
    if (ctx.h_result_distances != nullptr) cudaFreeHost(ctx.h_result_distances);
    if (ctx.d_candidate_vectors != nullptr) cudaFree(ctx.d_candidate_vectors);
    if (ctx.d_candidate_ids != nullptr) cudaFree(ctx.d_candidate_ids);
    if (ctx.d_candidate_counts != nullptr) cudaFree(ctx.d_candidate_counts);
    if (ctx.d_original_query_ids != nullptr) cudaFree(ctx.d_original_query_ids);
    if (ctx.d_candidate_distances != nullptr) cudaFree(ctx.d_candidate_distances);
    if (ctx.d_result_ids != nullptr) cudaFree(ctx.d_result_ids);
    if (ctx.d_result_distances != nullptr) cudaFree(ctx.d_result_distances);
    ctx = RabitqFloatGpuRerankContext{};
}

void validate_float_gpu_post_vector_storage(const GraphRuntime& runtime)
{

    if (runtime.rerank_vector_store) {
        if (runtime.rerank_vector_store->mode() != "mmap_fp16") {
            fail_rabitq("staged float GPU rerank supports only mmap_fp16 custom vector storage");
        }
        if (runtime.rerank_vector_store->rows() != runtime.base_vectors.rows ||
            runtime.rerank_vector_store->cols() != runtime.base_vectors.cols) {
            fail_rabitq("staged float GPU rerank mmap_fp16 metadata mismatch");
        }
    } else if (runtime.base_float_vectors.data.empty() ||
               runtime.base_float_vectors.rows != runtime.base_vectors.rows ||
               runtime.base_float_vectors.cols != runtime.base_vectors.cols) {
        fail_rabitq("staged float GPU rerank requires a matching resident FP32 base matrix or mmap_fp16 store");
    }
}

void init_float_gpu_rerank_context(RabitqFloatGpuRerankContext& ctx,
                                   const RabitqSearchOptions& options,
                                   const GraphRuntime& runtime)
{
    if (!highfreq_staged_float_gpu_exact_rerank_enabled(
            options, runtime.vector_scalar_type)) {
        return;
    }
    validate_float_gpu_post_vector_storage(runtime);

    ctx.batch_capacity = static_cast<uint32_t>(
        std::max(1, options.highfreq_rerank_batch_size));
    ctx.shortlist_capacity = effective_highfreq_rerank_shortlist(options);
    ctx.topk_capacity = static_cast<uint32_t>(options.topk);
    ctx.dim = runtime.base_vectors.cols;
    const size_t candidate_slots = checked_size_product(
        ctx.batch_capacity, ctx.shortlist_capacity, "float GPU rerank candidate slots");
    const size_t candidate_values = checked_size_product(
        candidate_slots, ctx.dim, "float GPU rerank candidate values");
    const size_t result_slots = checked_size_product(
        ctx.batch_capacity, ctx.topk_capacity, "float GPU rerank result slots");

    try {
        check_cuda(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking),
                   "cudaStreamCreate float GPU rerank");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&ctx.h_candidate_vectors),
                                  candidate_values * sizeof(float)),
                   "cudaMallocHost float GPU rerank candidate vectors");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&ctx.h_candidate_ids),
                                  candidate_slots * sizeof(idx_t)),
                   "cudaMallocHost float GPU rerank candidate ids");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&ctx.h_candidate_counts),
                                  ctx.batch_capacity * sizeof(uint32_t)),
                   "cudaMallocHost float GPU rerank candidate counts");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&ctx.h_original_query_ids),
                                  ctx.batch_capacity * sizeof(uint32_t)),
                   "cudaMallocHost float GPU rerank original query ids");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&ctx.h_result_ids),
                                  result_slots * sizeof(idx_t)),
                   "cudaMallocHost float GPU rerank result ids");
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&ctx.h_result_distances),
                                  result_slots * sizeof(float)),
                   "cudaMallocHost float GPU rerank result distances");
        check_cuda(cudaMalloc(&ctx.d_candidate_vectors, candidate_values * sizeof(float)),
                   "cudaMalloc float GPU rerank candidate vectors");
        check_cuda(cudaMalloc(&ctx.d_candidate_ids, candidate_slots * sizeof(idx_t)),
                   "cudaMalloc float GPU rerank candidate ids");
        check_cuda(cudaMalloc(&ctx.d_candidate_counts,
                              ctx.batch_capacity * sizeof(uint32_t)),
                   "cudaMalloc float GPU rerank candidate counts");
        check_cuda(cudaMalloc(&ctx.d_original_query_ids,
                              ctx.batch_capacity * sizeof(uint32_t)),
                   "cudaMalloc float GPU rerank original query ids");
        check_cuda(cudaMalloc(&ctx.d_candidate_distances, candidate_slots * sizeof(float)),
                   "cudaMalloc float GPU rerank candidate distances");
        check_cuda(cudaMalloc(&ctx.d_result_ids, result_slots * sizeof(idx_t)),
                   "cudaMalloc float GPU rerank result ids");
        check_cuda(cudaMalloc(&ctx.d_result_distances, result_slots * sizeof(float)),
                   "cudaMalloc float GPU rerank result distances");
        ctx.initialized = true;
        std::printf("[rabitq] float GPU rerank: transport=staged_fp32 "
                    "batch_queries=%u shortlist=%u topk=%u dim=%u "
                    "staging_mib=%.2f source=%s source_values=%s device_values=fp32\n",
                    ctx.batch_capacity,
                    ctx.shortlist_capacity,
                    ctx.topk_capacity,
                    ctx.dim,
                    static_cast<double>(candidate_values * sizeof(float)) /
                        static_cast<double>(size_t{1} << 20),
                    runtime.rerank_vector_store ? "mmap_fp16" : "resident_fp32",
                    runtime.rerank_vector_store ? "fp16_exact" : "fp32_exact");
    } catch (...) {
        cleanup_float_gpu_rerank_context(ctx);
        throw;
    }
}

void cleanup_lowfreq_context(RabitqLowfreqSearchContext& ctx)
{
    cleanup_posting_stream_buffer(ctx.completion);
    if (ctx.d_result != nullptr) cudaFree(ctx.d_result);
    if (ctx.d_distances != nullptr) cudaFree(ctx.d_distances);
    if (ctx.d_query_ids != nullptr) cudaFree(ctx.d_query_ids);
    if (ctx.d_candidate_ids != nullptr) cudaFree(ctx.d_candidate_ids);
    if (ctx.d_candidate_offsets != nullptr) cudaFree(ctx.d_candidate_offsets);
    if (ctx.d_candidate_sizes != nullptr) cudaFree(ctx.d_candidate_sizes);
    if (ctx.d_query_group_ids != nullptr) cudaFree(ctx.d_query_group_ids);
    if (ctx.d_label_size != nullptr) cudaFree(ctx.d_label_size);
    if (ctx.d_label_offset != nullptr) cudaFree(ctx.d_label_offset);
    if (ctx.d_sample_ids != nullptr) cudaFree(ctx.d_sample_ids);
    ctx = RabitqLowfreqSearchContext{};
}

void cleanup_range_posting_context(RabitqRangePostingSearchContext& ctx)
{
    cleanup_posting_stream_buffer(ctx.resident_completion);
    cleanup_posting_stream_buffer(ctx.postfilter_completion);
    if (ctx.d_sample_ids != nullptr) cudaFree(ctx.d_sample_ids);
    if (ctx.d_query_ids != nullptr) cudaFree(ctx.d_query_ids);
    if (ctx.d_posting_offsets != nullptr) cudaFree(ctx.d_posting_offsets);
    if (ctx.d_posting_sizes != nullptr) cudaFree(ctx.d_posting_sizes);
    if (ctx.d_exact_pass_counts != nullptr) cudaFree(ctx.d_exact_pass_counts);
    if (ctx.d_result != nullptr) cudaFree(ctx.d_result);
    for (auto* plan : {&ctx.resident_segments, &ctx.postfilter_segments}) {
        if (plan->d_route_ids != nullptr) cudaFree(plan->d_route_ids);
        if (plan->d_offsets != nullptr) cudaFree(plan->d_offsets);
        if (plan->d_sizes != nullptr) cudaFree(plan->d_sizes);
        if (plan->d_query_indptr != nullptr) cudaFree(plan->d_query_indptr);
        if (plan->d_query_positions != nullptr) cudaFree(plan->d_query_positions);
    }
    if (ctx.d_segment_ids != nullptr) cudaFree(ctx.d_segment_ids);
    if (ctx.d_segment_distances != nullptr) cudaFree(ctx.d_segment_distances);
    if (ctx.d_segment_pass_counts != nullptr) cudaFree(ctx.d_segment_pass_counts);
    if (ctx.d_segment_completion_counts != nullptr) cudaFree(ctx.d_segment_completion_counts);
    if (ctx.segment_start != nullptr) cudaEventDestroy(ctx.segment_start);
    if (ctx.segment_scan_end != nullptr) cudaEventDestroy(ctx.segment_scan_end);
    if (ctx.segment_merge_end != nullptr) cudaEventDestroy(ctx.segment_merge_end);
    ctx = RabitqRangePostingSearchContext{};
}

RabitqRangePostingPlan build_range_posting_plan(const range_posting::Index& posting_index,
                                                 const GraphRuntime& runtime,
                                                 const RabitqSearchOptions& options)
{
    RabitqRangePostingPlan plan;
    plan.mode = range_posting::parse_mode(options.range_posting_mode);
    plan.selected_original_queries.assign(runtime.original_query_count, 0);
    plan.selected_routes.assign(runtime.route_to_original_query.size(), 0);
    plan.replaced_routes.assign(runtime.route_to_original_query.size(), 0);
    plan.dropped_wide_hybrid_or_original_queries.assign(runtime.original_query_count, 0);
    plan.selected_range_graph_and_original_queries.assign(runtime.original_query_count, 0);
    if (plan.mode == range_posting::Mode::Off) {
        return plan;
    }
    if (posting_index.row_count() != runtime.base_vectors.rows) {
        fail_rabitq("range posting row count does not match base vectors");
    }
    if (posting_index.range_count() != runtime.base_range_count ||
        posting_index.range_count() != runtime.original_query_ranges.range_count) {
        fail_rabitq("range posting, base range, and query range counts do not match");
    }
    if (runtime.original_query_ranges.rows != runtime.original_query_count) {
        fail_rabitq("original query range table is not aligned with original queries");
    }
    bool has_active_range = false;
    for (uint8_t query_range_mode : runtime.original_query_ranges.modes) {
        if (query_range_mode > 3) {
            fail_rabitq("range posting query table contains an unknown range mode");
        }
        has_active_range = has_active_range || query_range_mode != 0;
    }
    if (!has_active_range) {
        fail_rabitq("enabled range posting requires at least one active range predicate");
    }

    if (runtime.route_providers.size() != runtime.route_to_original_query.size() ||
        runtime.route_execution_modes.size() != runtime.route_to_original_query.size() ||
        runtime.route_candidate_sources.size() != runtime.route_to_original_query.size() ||
        runtime.query_ranges.rows != runtime.route_to_original_query.size()) {
        fail_rabitq("range posting requires provider and range metadata aligned by route");
    }
    std::vector<uint32_t> representative_routes(runtime.original_query_count, UINT32_MAX);
    for (uint32_t route_idx = 0; route_idx < runtime.route_to_original_query.size(); ++route_idx) {
        const uint32_t original_query = runtime.route_to_original_query[route_idx];
        if (original_query >= representative_routes.size()) {
            fail_rabitq("route-to-original map is invalid while planning range posting");
        }
        if (representative_routes[original_query] == UINT32_MAX) {
            representative_routes[original_query] = route_idx;
        }
    }

    const auto append_route_slice = [&](uint32_t route_idx,
                                        const range_posting::Slice& slice) {
        if (route_idx >= runtime.route_to_original_query.size()) {
            fail_rabitq("range posting route id is out of range");
        }
        const uint32_t original_query = runtime.route_to_original_query[route_idx];
        const uint64_t dimension_base =
            static_cast<uint64_t>(slice.range_index) * posting_index.row_count();
        if (dimension_base > UINT64_MAX - slice.begin ||
            plan.total_rows_scanned > UINT64_MAX - slice.count) {
            fail_rabitq("range posting plan size overflow");
        }
        plan.original_query_ids.push_back(original_query);
        plan.representative_route_ids.push_back(route_idx);
        plan.posting_offsets.push_back(dimension_base + slice.begin);
        plan.posting_sizes.push_back(slice.count);
        plan.posting_range_indices.push_back(slice.range_index);
        plan.selected_original_queries[original_query] = 1;
        plan.selected_routes[route_idx] = 1;
        const uint32_t plan_position = static_cast<uint32_t>(
            plan.representative_route_ids.size() - 1);
        if (runtime.route_providers[route_idx] == QueryRouteProvider::RangePosting &&
            runtime.route_execution_modes[route_idx] ==
                QueryRouteExecution::HostCandidateStream) {
            plan.stream_plan_positions.push_back(plan_position);
        } else if (runtime.route_providers[route_idx] == QueryRouteProvider::RangePosting &&
                   runtime.route_execution_modes[route_idx] ==
                       QueryRouteExecution::GpuResidentScanPostFilter) {
            plan.resident_postfilter_plan_positions.push_back(plan_position);
        } else {
            plan.resident_plan_positions.push_back(plan_position);
        }
        plan.total_rows_scanned += slice.count;
    };

    if (runtime.range_access_planner_profile.planned) {
        if (runtime.route_enabled.size() != runtime.route_to_original_query.size() ||
            runtime.route_range_posting_begins.size() !=
                runtime.route_to_original_query.size() ||
            runtime.route_range_posting_sizes.size() !=
                runtime.route_to_original_query.size() ||
            runtime.route_range_posting_indices.size() !=
                runtime.route_to_original_query.size()) {
            fail_rabitq("dataloader range access plan is not route-aligned");
        }
        plan.dropped_wide_hybrid_or_original_queries =
            runtime.range_graph_or_original_queries;
        plan.selected_range_graph_and_original_queries =
            runtime.range_graph_and_original_queries;
        plan.selected_range_graph_and_queries =
            runtime.range_access_planner_profile.range_graph_and_queries;
        std::vector<uint64_t> range_graph_and_rows(runtime.original_query_count, UINT64_MAX);
        std::vector<uint64_t> range_graph_or_rows(runtime.original_query_count, UINT64_MAX);
        for (uint32_t route_idx = 0; route_idx < runtime.route_to_original_query.size();
             ++route_idx) {
            const uint32_t original_query = runtime.route_to_original_query[route_idx];
            if (runtime.route_enabled[route_idx] == 0) {
                plan.replaced_routes[route_idx] = 1;
                if (runtime.route_providers[route_idx] == QueryRouteProvider::RangePosting) {
                    if (original_query < runtime.range_graph_and_original_queries.size() &&
                        runtime.range_graph_and_original_queries[original_query] != 0) {
                        range_graph_and_rows[original_query] = std::min(
                            range_graph_and_rows[original_query],
                            runtime.route_range_posting_sizes[route_idx]);
                    }
                    if (original_query < runtime.range_graph_or_original_queries.size() &&
                        runtime.range_graph_or_original_queries[original_query] != 0) {
                        range_graph_or_rows[original_query] = std::min(
                            range_graph_or_rows[original_query],
                            runtime.route_range_posting_sizes[route_idx]);
                    }
                }
                if (runtime.route_providers[route_idx] == QueryRouteProvider::Label &&
                    original_query < runtime.range_graph_and_original_queries.size() &&
                    runtime.range_graph_and_original_queries[original_query] != 0) {
                    ++plan.replaced_range_graph_and_label_routes;
                }
                continue;
            }
            if (runtime.route_providers[route_idx] != QueryRouteProvider::RangePosting) {
                continue;
            }
            range_posting::Slice slice;
            slice.has_active_range = true;
            slice.range_index = runtime.route_range_posting_indices[route_idx];
            slice.begin = runtime.route_range_posting_begins[route_idx];
            slice.count = runtime.route_range_posting_sizes[route_idx];
            if (slice.range_index >= posting_index.range_count()) {
                fail_rabitq("dataloader range access plan has an invalid range index");
            }
            if (slice.count > options.range_posting_scan_threshold) {
                fail_rabitq(
                    "dataloader enabled a range posting route above the scan threshold");
            }
            append_route_slice(route_idx, slice);
        }
        for (uint32_t original_query = 0;
             original_query < runtime.original_query_count;
             ++original_query) {
            if (runtime.range_graph_or_original_queries[original_query] != 0) {
                ++plan.dropped_wide_hybrid_or_routes;
                if (range_graph_or_rows[original_query] != UINT64_MAX) {
                    plan.dropped_wide_hybrid_or_rows += range_graph_or_rows[original_query];
                }
            }
            if (runtime.range_graph_and_original_queries[original_query] != 0 &&
                range_graph_and_rows[original_query] != UINT64_MAX) {
                plan.selected_range_graph_and_rows += range_graph_and_rows[original_query];
            }
        }
        return plan;
    }

    const auto label_route_rows = [&](uint32_t route_idx) -> uint64_t {
        if (route_idx >= runtime.query_group_ids.size()) return UINT64_MAX;
        const uint32_t group = runtime.query_group_ids[route_idx];
        if (group == UINT32_MAX) return UINT64_MAX;
        if (runtime.query_is_lowfreq[route_idx] != 0 &&
            group < runtime.lowfreq_index.label_size.size()) {
            return runtime.lowfreq_index.label_size[group];
        }
        if (group < runtime.metadata.group_posting_size.size()) {
            return runtime.metadata.group_posting_size[group];
        }
        return UINT64_MAX;
    };
    std::vector<uint64_t> clause_best_label_rows(
        runtime.original_query_clauses.clause_count(), UINT64_MAX);
    std::vector<uint8_t> clause_has_lowfreq_label(
        runtime.original_query_clauses.clause_count(), 0);
    for (uint32_t route_idx = 0; route_idx < runtime.route_providers.size(); ++route_idx) {
        if (runtime.route_providers[route_idx] != QueryRouteProvider::Label ||
            route_idx >= runtime.route_to_clause.size()) {
            continue;
        }
        const uint32_t clause = runtime.route_to_clause[route_idx];
        if (clause == UINT32_MAX || clause >= clause_best_label_rows.size()) {
            continue;
        }
        clause_best_label_rows[clause] =
            std::min(clause_best_label_rows[clause], label_route_rows(route_idx));
        if (route_idx < runtime.query_is_lowfreq.size() &&
            runtime.query_is_lowfreq[route_idx] != 0) {
            clause_has_lowfreq_label[clause] = 1;
        }
    }
    const auto clause_label_route_rows = [&](uint32_t range_route) -> uint64_t {
        if (range_route >= runtime.route_to_clause.size()) return UINT64_MAX;
        const uint32_t clause = runtime.route_to_clause[range_route];
        return clause < clause_best_label_rows.size()
            ? clause_best_label_rows[clause]
            : UINT64_MAX;
    };
    const auto replace_clause_label_routes = [&](uint32_t range_route) {
        const uint32_t clause = runtime.route_to_clause[range_route];
        for (uint32_t route_idx = 0; route_idx < runtime.route_providers.size(); ++route_idx) {
            if (runtime.route_providers[route_idx] == QueryRouteProvider::Label &&
                runtime.route_to_clause[route_idx] == clause) {
                plan.replaced_routes[route_idx] = 1;
            }
        }
    };
    const auto drop_wide_hybrid_or_route = [&](uint32_t route_idx,
                                                bool eligible) {
        if (!options.range_posting_drop_wide_hybrid_or || eligible ||
            runtime.route_execution_modes[route_idx] !=
                QueryRouteExecution::GpuResidentScan) {
            return false;
        }
        const uint32_t original_query = runtime.route_to_original_query[route_idx];
        return original_query < runtime.original_query_label_ops.size() &&
            runtime.original_query_clauses.has_clauses(original_query) &&
            runtime.original_query_label_ops[original_query] == QueryLabelOp::Or;
    };

    if (options.range_graph_hybrid_and_enabled) {
        std::vector<uint64_t> query_range_rows(runtime.original_query_count, UINT64_MAX);
        std::vector<uint64_t> query_best_label_rows(runtime.original_query_count, UINT64_MAX);
        std::vector<uint8_t> query_has_stream(runtime.original_query_count, 0);
        std::vector<uint8_t> query_has_lowfreq(runtime.original_query_count, 0);
        for (uint32_t route_idx = 0; route_idx < runtime.route_providers.size(); ++route_idx) {
            if (runtime.route_providers[route_idx] == QueryRouteProvider::RangePosting &&
                runtime.route_execution_modes[route_idx] ==
                    QueryRouteExecution::HostCandidateStream) {
                const uint32_t original_query = runtime.route_to_original_query[route_idx];
                const range_posting::Slice slice = range_posting::choose_smallest_slice(
                    posting_index, runtime.query_ranges, route_idx);
                query_has_stream[original_query] = 1;
                query_range_rows[original_query] =
                    std::min(query_range_rows[original_query], slice.count);
                const uint32_t clause = runtime.route_to_clause[route_idx];
                if (clause < clause_has_lowfreq_label.size() &&
                    clause_has_lowfreq_label[clause] != 0) {
                    query_has_lowfreq[original_query] = 1;
                }
                query_best_label_rows[original_query] = std::min(
                    query_best_label_rows[original_query],
                    clause_label_route_rows(route_idx));
            }
        }
        for (uint32_t original_query = 0;
             original_query < runtime.original_query_count;
             ++original_query) {
            if (query_has_stream[original_query] == 0 ||
                query_has_lowfreq[original_query] != 0 ||
                query_range_rows[original_query] >= query_best_label_rows[original_query]) {
                continue;
            }
            if (original_query >= runtime.original_query_label_ops.size() ||
                runtime.original_query_label_ops[original_query] != QueryLabelOp::And) {
                fail_rabitq(
                    "range graph hybrid-AND planner received a non-AND host stream");
            }
            plan.selected_range_graph_and_original_queries[original_query] = 1;
            ++plan.selected_range_graph_and_queries;
            plan.selected_range_graph_and_rows += query_range_rows[original_query];
        }
        for (uint32_t route_idx = 0; route_idx < runtime.route_providers.size(); ++route_idx) {
            if (runtime.route_providers[route_idx] != QueryRouteProvider::Label) {
                continue;
            }
            const uint32_t original_query = runtime.route_to_original_query[route_idx];
            if (plan.selected_range_graph_and_original_queries[original_query] == 0) {
                continue;
            }
            const uint32_t clause = runtime.route_to_clause[route_idx];
            if (clause >= runtime.original_query_clauses.clause_count() ||
                !runtime.original_query_clauses.clause_uses_range(clause, original_query)) {
                continue;
            }
            plan.replaced_routes[route_idx] = 1;
            ++plan.replaced_range_graph_and_label_routes;
        }
    }

    // Explicit range providers are either resident scans for range-only
    // clauses or generic host-candidate streams when the clause has residual
    // labels. Select mode chooses the smaller complete enumeration source.
    for (uint32_t route_idx = 0; route_idx < runtime.route_providers.size(); ++route_idx) {
        if (runtime.route_providers[route_idx] != QueryRouteProvider::RangePosting) {
            continue;
        }
        const range_posting::Slice slice = range_posting::choose_smallest_slice(
            posting_index, runtime.query_ranges, route_idx);
        const bool eligible = range_posting::eligible_for_scan(
            slice, options.range_posting_scan_threshold);
        const bool streamed = runtime.route_execution_modes[route_idx] ==
            QueryRouteExecution::HostCandidateStream;
        if (drop_wide_hybrid_or_route(route_idx, eligible)) {
            const uint32_t original_query = runtime.route_to_original_query[route_idx];
            plan.dropped_wide_hybrid_or_original_queries[original_query] = 1;
            ++plan.dropped_wide_hybrid_or_routes;
            plan.dropped_wide_hybrid_or_rows += slice.count;
            continue;
        }
        if (!eligible) {
            const uint32_t original_query = runtime.route_to_original_query[route_idx];
            const bool required_hybrid_or =
                original_query < runtime.original_query_label_ops.size() &&
                runtime.original_query_clauses.has_clauses(original_query) &&
                runtime.original_query_label_ops[original_query] == QueryLabelOp::Or;
            if (required_hybrid_or) {
                fail_rabitq(
                    "wide hybrid range-OR route exceeds the scan threshold without a fallback");
            }
            continue;
        }
        if (streamed) {
            if (options.range_graph_hybrid_and_enabled) {
                continue;
            }
            const uint64_t label_rows = clause_label_route_rows(route_idx);
            const bool range_is_preferred = slice.count < label_rows;
            if ((plan.mode == range_posting::Mode::Select && !range_is_preferred) ||
                (plan.mode == range_posting::Mode::Hybrid &&
                 !eligible && !range_is_preferred)) {
                continue;
            }
            if (plan.mode == range_posting::Mode::Select) {
                replace_clause_label_routes(route_idx);
            }
        }
        append_route_slice(route_idx, slice);
    }

    for (uint32_t original_query = 0; original_query < runtime.original_query_count;
         ++original_query) {
        const SparseLabelRowSpan labels = sparse_label_row_span(
            runtime.original_query_labels_matrix, original_query);
        if (!labels.valid) {
            fail_rabitq("original query label CSR is invalid while planning range posting");
        }
        if (labels.begin != labels.end) {
            continue;
        }
        const uint32_t representative_route = representative_routes[original_query];
        if (representative_route == UINT32_MAX ||
            runtime.route_providers[representative_route] ==
                QueryRouteProvider::RangePosting) {
            continue;
        }
        const range_posting::Slice slice = range_posting::choose_smallest_slice(
            posting_index, runtime.query_ranges, representative_route);
        if (!range_posting::eligible_for_scan(slice,
                                              options.range_posting_scan_threshold)) {
            continue;
        }
        append_route_slice(representative_route, slice);
        if (plan.mode == range_posting::Mode::Select) {
            for (uint32_t route_idx = 0; route_idx < runtime.route_to_original_query.size();
                 ++route_idx) {
                if (runtime.route_to_original_query[route_idx] == original_query) {
                    plan.replaced_routes[route_idx] = 1;
                }
            }
        }
    }
    return plan;
}

void init_posting_segments(RabitqPostingSegmentDevicePlan& device,
                           const RabitqRangePostingPlan& plan,
                           const std::vector<uint32_t>& positions,
                           uint32_t target_rows)
{
    if (target_rows == 0 || positions.empty()) return;
    std::vector<uint32_t> routes;
    std::vector<uint64_t> offsets, sizes;
    for (uint32_t position : positions) {
        routes.push_back(plan.representative_route_ids.at(position));
        offsets.push_back(plan.posting_offsets.at(position));
        sizes.push_back(plan.posting_sizes.at(position));
    }
    const auto segments = posting_segments::make_plan(routes, offsets, sizes, target_rows);
    device.count = static_cast<uint32_t>(segments.route_ids.size());
    const auto upload = [](auto*& destination, const auto& source) {
        const size_t bytes = source.size() * sizeof(source[0]);
        check_cuda(cudaMalloc(&destination, bytes), "cudaMalloc posting segment plan");
        check_cuda(cudaMemcpy(destination, source.data(), bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy posting segment plan");
    };
    upload(device.d_route_ids, segments.route_ids);
    upload(device.d_offsets, segments.offsets);
    upload(device.d_sizes, segments.sizes);
    upload(device.d_query_indptr, segments.query_indptr);
    upload(device.d_query_positions, segments.query_positions);
}

void init_range_posting_context(RabitqRangePostingSearchContext& ctx,
                                const range_posting::Index& posting_index,
                                RabitqRangePostingPlan plan,
                                const RabitqSearchOptions& options)
{
    ctx.plan = std::move(plan);
    ctx.shortlist_k = static_cast<uint32_t>(options.range_posting_shortlist);
    ctx.postfilter_shortlist_k = static_cast<uint32_t>(
        options.range_posting_shortlist *
        options.range_posting_postfilter_shortlist_multiplier);
    if (ctx.plan.mode == range_posting::Mode::Off ||
        ctx.plan.representative_route_ids.empty()) {
        return;
    }
    if (posting_index.range_count() != 0 &&
        posting_index.row_count() >
            std::numeric_limits<uint64_t>::max() / posting_index.range_count()) {
        fail_rabitq("range posting resident row count overflow");
    }
    const uint64_t resident_rows = posting_index.row_count() * posting_index.range_count();
    if (resident_rows > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
        fail_rabitq("range posting resident payload exceeds the host size range");
    }
    const size_t query_count = std::max(
        {ctx.plan.resident_plan_positions.size(),
         ctx.plan.resident_postfilter_plan_positions.size(),
         ctx.plan.stream_plan_positions.size()});
    ctx.device_query_capacity = query_count;
    const size_t ordinary_resident_slots = checked_size_product(
        ctx.plan.resident_plan_positions.size(),
        static_cast<size_t>(ctx.shortlist_k),
        "ordinary resident range posting shortlist");
    const size_t postfilter_slots = checked_size_product(
        ctx.plan.resident_postfilter_plan_positions.size(),
        static_cast<size_t>(ctx.postfilter_shortlist_k),
        "postfilter range posting shortlist");
    const size_t stream_slots = checked_size_product(
        ctx.plan.stream_plan_positions.size(),
        static_cast<size_t>(ctx.shortlist_k),
        "streamed range posting shortlist");
    ctx.result_slot_capacity =
        std::max({ordinary_resident_slots, postfilter_slots, stream_slots});
    try {
        if (!ctx.plan.resident_plan_positions.empty() ||
            !ctx.plan.resident_postfilter_plan_positions.empty()) {
            check_cuda(cudaMalloc(&ctx.d_sample_ids,
                                  static_cast<size_t>(resident_rows) * sizeof(uint32_t)),
                       "cudaMalloc range posting sample ids");
            for (uint32_t range_index = 0; range_index < posting_index.range_count();
                 ++range_index) {
                const range_posting::DimensionView& dimension =
                    posting_index.dimension(range_index);
                check_cuda(cudaMemcpy(ctx.d_sample_ids +
                                          static_cast<size_t>(range_index) * dimension.row_count,
                                      dimension.sample_ids,
                                      static_cast<size_t>(dimension.row_count) * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy range posting sample ids");
            }
        }
        check_cuda(cudaMalloc(&ctx.d_query_ids, query_count * sizeof(uint32_t)),
                   "cudaMalloc range posting query ids");
        check_cuda(cudaMalloc(&ctx.d_posting_offsets, query_count * sizeof(uint64_t)),
                   "cudaMalloc range posting offsets");
        check_cuda(cudaMalloc(&ctx.d_posting_sizes, query_count * sizeof(uint64_t)),
                   "cudaMalloc range posting sizes");
        check_cuda(cudaMalloc(&ctx.d_exact_pass_counts, query_count * sizeof(uint32_t)),
                   "cudaMalloc range posting exact pass counts");
        check_cuda(cudaMalloc(&ctx.d_result, ctx.result_slot_capacity * sizeof(idx_t)),
                   "cudaMalloc range posting results");
        init_posting_segments(ctx.resident_segments, ctx.plan,
                              ctx.plan.resident_plan_positions,
                              options.range_posting_segment_rows);
        init_posting_segments(ctx.postfilter_segments, ctx.plan,
                              ctx.plan.resident_postfilter_plan_positions,
                              options.range_posting_segment_rows);
        const size_t segment_count = std::max(ctx.resident_segments.count,
                                              ctx.postfilter_segments.count);
        if (segment_count > 0) {
            const size_t slots = std::max(
                checked_size_product(ctx.resident_segments.count, ctx.shortlist_k,
                                     "ordinary range segment slots"),
                checked_size_product(ctx.postfilter_segments.count, ctx.postfilter_shortlist_k,
                                     "postfilter range segment slots"));
            check_cuda(cudaMalloc(&ctx.d_segment_ids, slots * sizeof(idx_t)),
                       "cudaMalloc range segment ids");
            check_cuda(cudaMalloc(&ctx.d_segment_distances, slots * sizeof(float)),
                       "cudaMalloc range segment distances");
            check_cuda(cudaMalloc(&ctx.d_segment_pass_counts, segment_count * sizeof(uint32_t)),
                       "cudaMalloc range segment counts");
            check_cuda(cudaMalloc(&ctx.d_segment_completion_counts, query_count * sizeof(uint32_t)),
                       "cudaMalloc range segment completion counters");
            check_cuda(cudaEventCreate(&ctx.segment_start), "cudaEventCreate segment start");
            check_cuda(cudaEventCreate(&ctx.segment_scan_end), "cudaEventCreate segment scan end");
            check_cuda(cudaEventCreate(&ctx.segment_merge_end), "cudaEventCreate segment merge end");
            std::cout << "[range-posting-segments] target_rows=" << options.range_posting_segment_rows
                      << " resident_routes=" << ctx.plan.resident_plan_positions.size()
                      << " resident_tasks=" << ctx.resident_segments.count
                      << " postfilter_routes=" << ctx.plan.resident_postfilter_plan_positions.size()
                      << " postfilter_tasks=" << ctx.postfilter_segments.count
                      << " scratch_bytes=" << slots * (sizeof(idx_t) + sizeof(float)) +
                                                   segment_count * sizeof(uint32_t)
                      << std::endl;
        }
        ctx.host_shortlist.resize(ctx.result_slot_capacity, UINT32_MAX);
        ctx.host_exact_pass_counts.resize(query_count, 0);
        ctx.resident_row_count = resident_rows;
        ctx.initialized = true;
    } catch (...) {
        cleanup_range_posting_context(ctx);
        throw;
    }
}

void cleanup_graph_context(RabitqGraphSearchContext& ctx)
{
    cleanup_float_gpu_rerank_context(ctx.float_gpu_rerank);
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
    if (ctx.result_buffer_is_host_mapped) {
        if (ctx.h_result_mapped != nullptr) cudaFreeHost(ctx.h_result_mapped);
        if (ctx.h_result_distances_mapped != nullptr) cudaFreeHost(ctx.h_result_distances_mapped);
        if (ctx.h_time_per_block_mapped != nullptr) cudaFreeHost(ctx.h_time_per_block_mapped);
    } else {
        if (ctx.d_result != nullptr) cudaFree(ctx.d_result);
        if (ctx.d_result_distances != nullptr) cudaFree(ctx.d_result_distances);
        if (ctx.d_time_per_block != nullptr) cudaFree(ctx.d_time_per_block);
    }
    if (ctx.d_phase_state != nullptr) cudaFree(ctx.d_phase_state);
    if (ctx.d_datadist_list != nullptr) cudaFree(ctx.d_datadist_list);
    if (ctx.d_hashdata != nullptr) cudaFree(ctx.d_hashdata);
    if (ctx.d_mailbox_rtt_hist != nullptr) cudaFree(ctx.d_mailbox_rtt_hist);
    if (ctx.d_finite_contexts != nullptr) cudaFree(ctx.d_finite_contexts);
    if (ctx.d_finite_frontier_state != nullptr) cudaFree(ctx.d_finite_frontier_state);
    if (ctx.d_finite_stop_state != nullptr) cudaFree(ctx.d_finite_stop_state);
    if (ctx.d_finite_result_state != nullptr) cudaFree(ctx.d_finite_result_state);
    if (ctx.d_finite_ready_tokens != nullptr) cudaFree(ctx.d_finite_ready_tokens);
    if (ctx.d_request_ring_tails != nullptr) cudaFree(ctx.d_request_ring_tails);
    if (ctx.d_query_ids != nullptr) cudaFree(ctx.d_query_ids);
    if (ctx.status_memory != nullptr) cudaFreeHost(ctx.status_memory);
    if (ctx.status_flag != nullptr) cudaFreeHost(ctx.status_flag);
    if (ctx.h_request_ring_entries != nullptr) cudaFreeHost(ctx.h_request_ring_entries);
    if (ctx.d_request_ring_consumed_sequences_raw != nullptr) {
        cudaFree(ctx.d_request_ring_consumed_sequences_raw);
    } else if (ctx.h_request_ring_consumed_sequences != nullptr) {
        cudaFreeHost(ctx.h_request_ring_consumed_sequences);
    }
    if (ctx.d_prefetch_node_array != nullptr) cudaFree(ctx.d_prefetch_node_array);
    if (ctx.d_prefetch_index != nullptr) cudaFree(ctx.d_prefetch_index);
    if (ctx.d_prefetch_begin_by_group != nullptr) cudaFree(ctx.d_prefetch_begin_by_group);
    if (ctx.d_prefetch_end_by_group != nullptr) cudaFree(ctx.d_prefetch_end_by_group);
    if (ctx.d_query_group_ids != nullptr) cudaFree(ctx.d_query_group_ids);
    if (ctx.d_start_local_ids != nullptr) cudaFree(ctx.d_start_local_ids);
    if (ctx.d_row_to_sample != nullptr) cudaFree(ctx.d_row_to_sample);
    if (ctx.d_label_offset != nullptr) cudaFree(ctx.d_label_offset);
    if (ctx.row_filter_mask_host_mapped) {
        if (ctx.h_row_filter_mask_mapped != nullptr) cudaFreeHost(ctx.h_row_filter_mask_mapped);
    } else {
        if (ctx.d_row_filter_mask != nullptr) cudaFree(ctx.d_row_filter_mask);
    }
    if (ctx.d_query_filter_bits != nullptr) cudaFree(ctx.d_query_filter_bits);
    if (ctx.d_resident_graph != nullptr) cudaFree(ctx.d_resident_graph);
    if (ctx.owns_gpu_graph_row_cache && ctx.d_gpu_graph_row_cache != nullptr) {
        cudaFree(ctx.d_gpu_graph_row_cache);
    }
    if (ctx.d_gpu_query_label_indptr != nullptr) cudaFree(ctx.d_gpu_query_label_indptr);
    if (ctx.d_gpu_query_label_indices != nullptr) cudaFree(ctx.d_gpu_query_label_indices);
    if (ctx.d_gpu_sample_label_indptr != nullptr) cudaFree(ctx.d_gpu_sample_label_indptr);
    if (ctx.d_gpu_sample_label_indices != nullptr) cudaFree(ctx.d_gpu_sample_label_indices);
    if (ctx.d_gpu_graph_cache_view != nullptr) cudaFree(ctx.d_gpu_graph_cache_view);
    if (ctx.d_query_range_modes != nullptr) cudaFree(ctx.d_query_range_modes);
    if (ctx.d_query_range_lower_bounds != nullptr) cudaFree(ctx.d_query_range_lower_bounds);
    if (ctx.d_query_range_upper_bounds != nullptr) cudaFree(ctx.d_query_range_upper_bounds);
    if (ctx.owns_base_range_values && ctx.d_base_range_values != nullptr) {
        cudaFree(ctx.d_base_range_values);
    }
    if (ctx.using_host_mapped_neighbor) {
        if (ctx.h_neighbor_memory_host != nullptr) cudaFreeHost(ctx.h_neighbor_memory_host);
    } else {
        if (ctx.d_neighbor_memory_raw != nullptr) cudaFree(ctx.d_neighbor_memory_raw);
    }
    if (ctx.d_gdr_data_raw != nullptr) cudaFree(ctx.d_gdr_data_raw);
    cleanup_posting_predicate_context(ctx.label_posting_predicate);
    cleanup_posting_predicate_context(ctx.range_posting_predicate);
    cudaFree(ctx.predicate_offsets);
    cudaFree(ctx.predicate_handles);
    cudaFree(ctx.predicate_clauses);
    cudaFree(ctx.predicate_ranges);
    cudaFree(ctx.predicate_enabled);
    cudaFree(ctx.predicate_prefetch_disabled);
    cudaFree(ctx.predicate_compiled);
    cudaFree(ctx.predicate_counters);
    ctx = RabitqGraphSearchContext{};
}

void init_immutable_state(RabitqImmutableDeviceState& state, const RabitqHostIndex& index)
{
    try {
        check_cuda(cudaMalloc(&state.d_anchor_ids, sizeof(uint32_t) * index.anchor_ids.size()), "cudaMalloc d_anchor_ids");
        check_cuda(cudaMalloc(&state.d_codes, sizeof(uint8_t) * index.codes.size()), "cudaMalloc d_codes");
        check_cuda(cudaMalloc(&state.d_long_codes,
                              sizeof(uint8_t) * std::max<size_t>(1, index.long_codes.size())),
                   "cudaMalloc d_long_codes");
        check_cuda(cudaMalloc(&state.d_factors, sizeof(float) * index.factors.size()), "cudaMalloc d_factors");
        check_cuda(cudaMalloc(&state.d_ex_factors, sizeof(float) * index.ex_factors.size()), "cudaMalloc d_ex_factors");
        check_cuda(cudaMemcpy(state.d_anchor_ids,
                              index.anchor_ids.data(),
                              sizeof(uint32_t) * index.anchor_ids.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_anchor_ids");
        check_cuda(cudaMemcpy(state.d_codes,
                              index.codes.data(),
                              sizeof(uint8_t) * index.codes.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_codes");
        if (!index.long_codes.empty()) {
            check_cuda(cudaMemcpy(state.d_long_codes,
                                  index.long_codes.data(),
                                  sizeof(uint8_t) * index.long_codes.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy d_long_codes");
        }
        check_cuda(cudaMemcpy(state.d_factors,
                              index.factors.data(),
                              sizeof(float) * index.factors.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_factors");
        check_cuda(cudaMemcpy(state.d_ex_factors,
                              index.ex_factors.data(),
                              sizeof(float) * index.ex_factors.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_ex_factors");
    } catch (...) {
        cleanup_immutable_state(state);
        throw;
    }
}

void init_prepared_query_state(RabitqPreparedQueryState& state,
                               const RabitqHostIndex& index,
                               const GraphRuntime& runtime,
                               const RabitqSearchOptions& options,
                               RabitqPreparedQueryTiming* timing = nullptr)
{
    const auto total_start = std::chrono::steady_clock::now();
    const auto validation_start = total_start;
    if (options.query_rotation != "legacy" && options.query_rotation != "cublas_fp32") {
        fail_rabitq("query_rotation must be legacy or cublas_fp32");
    }
    if (options.centroid_distances != "legacy" && options.centroid_distances != "cublas_fp32") {
        fail_rabitq("centroid_distances must be legacy or cublas_fp32");
    }
    const uint32_t padded_dim = index.codec.padded_dim;
    state.routed_query_count = runtime.query_vectors.rows;
    state.original_query_count = runtime.original_query_count;
    if (state.routed_query_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        state.original_query_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fail_rabitq("query batch exceeds CUDA grid and kernel int range");
    }
    if (state.original_query_count == 0 || state.routed_query_count == 0) {
        fail_rabitq("RabitQ query preprocessing requires nonempty original and routed queries");
    }
    if (runtime.query_vectors.cols == 0 || index.codec.padded_dim == 0 ||
        index.header.num_anchors == 0) {
        fail_rabitq("RabitQ query preprocessing requires nonzero dimensions and anchors");
    }
    if (runtime.route_to_original_query.size() != state.routed_query_count) {
        fail_rabitq("route_to_original_query is not aligned with routed RabitQ queries");
    }
    for (uint32_t original_query_idx : runtime.route_to_original_query) {
        if (original_query_idx >= state.original_query_count) {
            fail_rabitq("route_to_original_query contains an out-of-range original query id");
        }
    }

    const size_t route_map_bytes = checked_size_product(
        sizeof(uint32_t), state.routed_query_count, "route-to-original query map");
    const size_t query_value_count = checked_size_product(
        state.original_query_count,
        static_cast<size_t>(runtime.query_vectors.cols),
        "original query matrix");
    const size_t query_u8_bytes =
        checked_size_product(sizeof(value_t), query_value_count, "original uint8 query matrix");
    const size_t query_float_bytes =
        checked_size_product(sizeof(float), query_value_count, "original float query matrix");
    const size_t rotated_value_count = checked_size_product(
        state.original_query_count,
        static_cast<size_t>(index.codec.padded_dim),
        "rotated query matrix");
    const size_t rotated_query_bytes =
        checked_size_product(sizeof(float), rotated_value_count, "rotated query matrix");
    const size_t query_factor_bytes = checked_size_product(
        sizeof(float), state.original_query_count, "RabitQ query factors");
    const uint64_t pair_count64 =
        static_cast<uint64_t>(state.original_query_count) *
        static_cast<uint64_t>(index.header.num_anchors);
    if (pair_count64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        fail_rabitq("query-anchor pair count exceeds CUDA grid range");
    }
    const size_t centroid_distance_bytes = checked_size_product(
        sizeof(float),
        static_cast<size_t>(pair_count64),
        "query-centroid distance matrix");

    if (timing != nullptr) {
        timing->validation_ms = elapsed_ms(validation_start,
                                           std::chrono::steady_clock::now());
    }

    try {
        const auto buffer_start = std::chrono::steady_clock::now();
        check_cuda(cudaMalloc(&state.d_route_to_original_query, route_map_bytes),
                   "cudaMalloc d_route_to_original_query");
        check_cuda(cudaMemcpy(state.d_route_to_original_query,
                              runtime.route_to_original_query.data(),
                              route_map_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_route_to_original_query");
        std::vector<float> query_float_buffer;
        const float* host_query_float = nullptr;
        if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
            if (runtime.original_query_vectors.rows != state.original_query_count ||
                runtime.original_query_vectors.cols != runtime.query_vectors.cols ||
                runtime.original_query_vectors.data.size() != query_value_count) {
                fail_rabitq("original uint8 query matrix is not size-aligned");
            }
            check_cuda(cudaMalloc(&state.d_query_u8, query_u8_bytes),
                       "cudaMalloc d_query_u8");
            check_cuda(cudaMemcpy(state.d_query_u8,
                                  runtime.original_query_vectors.data.data(),
                                  query_u8_bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy d_query_u8");
            const auto host_convert_start = std::chrono::steady_clock::now();
            query_float_buffer.resize(query_value_count);
            for (size_t idx = 0; idx < query_value_count; ++idx) {
                query_float_buffer[idx] =
                    static_cast<float>(runtime.original_query_vectors.data[idx]);
            }
            if (timing != nullptr) {
                timing->host_u8_to_float_detail_ms =
                    elapsed_ms(host_convert_start, std::chrono::steady_clock::now());
            }
            host_query_float = query_float_buffer.data();
        } else if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
            if (runtime.original_query_float_vectors.rows != state.original_query_count ||
                runtime.original_query_float_vectors.cols != runtime.query_vectors.cols ||
                runtime.original_query_float_vectors.data.size() != query_value_count) {
                fail_rabitq("original float query matrix is not size-aligned");
            }
            host_query_float = runtime.original_query_float_vectors.data.data();
        } else {
            fail_rabitq("unsupported dense vector scalar type for RabitQ query preprocessing");
        }
        check_cuda(cudaMalloc(&state.d_query_float, query_float_bytes),
                   "cudaMalloc d_query_float");
        check_cuda(cudaMemcpy(state.d_query_float,
                              host_query_float,
                              query_float_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_query_float");
        const auto static_upload_start = std::chrono::steady_clock::now();
        check_cuda(cudaMalloc(&state.d_rotation, sizeof(float) * index.rotation.size()), "cudaMalloc d_rotation");
        check_cuda(cudaMalloc(&state.d_anchors, sizeof(float) * index.anchors.size()), "cudaMalloc d_anchors");
        check_cuda(cudaMemcpy(state.d_rotation,
                              index.rotation.data(),
                              sizeof(float) * index.rotation.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_rotation");
        check_cuda(cudaMemcpy(state.d_anchors,
                              index.anchors.data(),
                              sizeof(float) * index.anchors.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_anchors");
        const double static_upload_ms =
            elapsed_ms(static_upload_start, std::chrono::steady_clock::now());
        check_cuda(cudaMalloc(&state.d_rotated_query, rotated_query_bytes),
                   "cudaMalloc d_rotated_query");
        check_cuda(cudaMalloc(&state.d_q_k1xsumq, query_factor_bytes),
                   "cudaMalloc d_q_k1xsumq");
        check_cuda(cudaMalloc(&state.d_q_kbxsumq, query_factor_bytes),
                   "cudaMalloc d_q_kbxsumq");
        check_cuda(cudaMalloc(&state.d_centroid_distances, centroid_distance_bytes),
                   "cudaMalloc d_centroid_distances");
        if (options.centroid_distances == "cublas_fp32") {
            check_cuda(cudaMalloc(&state.d_query_norms, query_factor_bytes), "cudaMalloc query norms");
            check_cuda(cudaMalloc(&state.d_anchor_norms, sizeof(float) * index.header.num_anchors), "cudaMalloc anchor norms");
        }

        if (options.use_int8_score) {
            const size_t quantized_query_bytes = checked_size_product(
                sizeof(int8_t), rotated_value_count, "RabitQ int8 query");
            check_cuda(cudaMalloc(&state.d_quantized_query, quantized_query_bytes),
                       "cudaMalloc d_quantized_query");
            check_cuda(cudaMalloc(&state.d_quantized_query_scales, query_factor_bytes),
                       "cudaMalloc d_quantized_query_scales");
        }
        if (options.use_lut) {
            state.short_lut_entries_per_query = short_lut_entries_for_dim(padded_dim);
            const size_t short_lut_value_count = checked_size_product(
                state.original_query_count,
                state.short_lut_entries_per_query,
                "RabitQ short LUT");
            const size_t short_lut_bytes =
                checked_size_product(sizeof(float), short_lut_value_count, "RabitQ short LUT");
            check_cuda(cudaMalloc(&state.d_short_lut, short_lut_bytes),
                       "cudaMalloc d_short_lut");
        }
        if (timing != nullptr) {
            timing->buffer_allocate_upload_ms =
                elapsed_ms(buffer_start, std::chrono::steady_clock::now()) - static_upload_ms;
            timing->static_rotation_anchors_ms = static_upload_ms;
        }
        RabitqPreprocessEvents events(timing != nullptr);
        const auto preprocess_start = std::chrono::steady_clock::now();
        {
            RabitqQueryRotation rotation(options.query_rotation == "cublas_fp32" || options.centroid_distances == "cublas_fp32");
            if (timing != nullptr) {
                timing->rotation_engine_setup_ms = elapsed_ms(preprocess_start, std::chrono::steady_clock::now());
                timing->rotation_method = options.query_rotation;
                timing->centroid_method = options.centroid_distances;
            }
            events.record(0);

            const uint32_t original_query_count_u32 =
                static_cast<uint32_t>(state.original_query_count);
            if (options.query_rotation == "cublas_fp32") {
                rotation.apply(state.d_query_float, state.d_rotation, state.d_rotated_query,
                               original_query_count_u32, static_cast<uint32_t>(options.dim), padded_dim);
            } else {
                const int rotate_threads = 128;
                const size_t rotate_elems = rotated_value_count;
                const size_t rotate_block_count =
                    ((rotate_elems - 1) / static_cast<size_t>(rotate_threads)) + 1;
                if (rotate_block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    fail_rabitq("rotated query launch exceeds CUDA grid range");
                }
                const int rotate_blocks = static_cast<int>(rotate_block_count);
                rotate_rabitq_queries_kernel<<<rotate_blocks, rotate_threads>>>(
                    state.d_query_float,
                    state.d_rotation,
                    state.d_rotated_query,
                    original_query_count_u32,
                    static_cast<uint32_t>(options.dim),
                    padded_dim);
                check_cuda(cudaGetLastError(), "rotate_rabitq_queries_kernel launch");
            }
            events.record(1);

            if (options.use_int8_score) {
                constexpr int quantize_threads = 256;
                quantize_rabitq_queries_int8_kernel<<<original_query_count_u32, quantize_threads>>>(
                    state.d_rotated_query,
                    state.d_quantized_query,
                    state.d_quantized_query_scales,
                    original_query_count_u32,
                    padded_dim);
                check_cuda(cudaGetLastError(), "quantize_rabitq_queries_int8_kernel launch");
            }

            events.record(2);
            if (options.use_lut) {
                constexpr int lut_threads = 128;
                precompute_rabitq_short_lut_kernel<<<original_query_count_u32, lut_threads>>>(
                    state.d_rotated_query,
                    state.d_short_lut,
                    original_query_count_u32,
                    padded_dim);
                check_cuda(cudaGetLastError(), "precompute_rabitq_short_lut_kernel launch");
            }

            events.record(3);
            constexpr int factor_threads = 256;
            compute_rabitq_query_factors_kernel<<<original_query_count_u32, factor_threads>>>(
                state.d_rotated_query,
                state.d_q_k1xsumq,
                state.d_q_kbxsumq,
                original_query_count_u32,
                padded_dim,
                index.codec.bits_per_dim);
            check_cuda(cudaGetLastError(), "compute_rabitq_query_factors_kernel launch");

            events.record(4);
            if (options.centroid_distances == "cublas_fp32") {
                rotation.centroid_distances(state.d_rotated_query, state.d_anchors, state.d_centroid_distances,
                    state.d_query_norms, state.d_anchor_norms, original_query_count_u32, index.header.num_anchors, padded_dim);
            } else {
                compute_rabitq_centroid_distances_kernel<<<static_cast<uint32_t>(pair_count64), factor_threads>>>(
                    state.d_rotated_query,
                    state.d_anchors,
                    state.d_centroid_distances,
                    original_query_count_u32,
                    index.header.num_anchors,
                    padded_dim);
                check_cuda(cudaGetLastError(), "compute_rabitq_centroid_distances_kernel launch");
            }
            events.record(5);
            check_cuda(cudaDeviceSynchronize(), "RabitQ query preprocessing");
        }  // Release the cuBLAS handle inside the preparation wall interval.
        if (timing != nullptr) {
            timing->gpu_preprocess_ms =
                elapsed_ms(preprocess_start, std::chrono::steady_clock::now());
            timing->rotation_device_ms = events.elapsed(0);
            timing->quantize_device_ms = events.elapsed(1);
            timing->lut_device_ms = options.use_lut ? events.elapsed(2) : 0.0;
            timing->factors_device_ms = events.elapsed(3);
            timing->centroid_device_ms = events.elapsed(4);
            timing->total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
        }
        std::printf(
            "[rabitq] prepared query GPU buffers original_queries=%zu routed_queries=%zu "
            "query_dim=%u padded_dim=%u lut=%d int8_score=%d\n",
            state.original_query_count,
            state.routed_query_count,
            runtime.query_vectors.cols,
            padded_dim,
            options.use_lut ? 1 : 0,
            options.use_int8_score ? 1 : 0);
    } catch (...) {
        cleanup_prepared_query_state(state);
        throw;
    }
}

void init_borrowed_prepared_query_state(RabitqPreparedQueryState& state,
                                        const RabitqPreparedQueryState& shared,
                                        const GraphRuntime& runtime)
{
    if (shared.d_query_float == nullptr || shared.d_rotated_query == nullptr ||
        shared.original_query_count == 0) {
        fail_rabitq("cannot borrow an uninitialized RabitQ query state");
    }
    if (runtime.original_query_count != shared.original_query_count) {
        fail_rabitq("secondary graph original query count does not match shared query state");
    }
    if (runtime.route_to_original_query.size() != runtime.query_vectors.rows) {
        fail_rabitq("secondary graph route map is not aligned with routed queries");
    }
    for (uint32_t original_query : runtime.route_to_original_query) {
        if (original_query >= shared.original_query_count) {
            fail_rabitq("secondary graph route map contains an invalid original query id");
        }
    }

    state = RabitqPreparedQueryState{};
    state.owns_shared_query_buffers = false;
    state.d_query_u8 = shared.d_query_u8;
    state.d_query_float = shared.d_query_float;
    state.d_rotation = shared.d_rotation;
    state.d_anchors = shared.d_anchors;
    state.d_rotated_query = shared.d_rotated_query;
    state.d_quantized_query = shared.d_quantized_query;
    state.d_quantized_query_scales = shared.d_quantized_query_scales;
    state.d_q_k1xsumq = shared.d_q_k1xsumq;
    state.d_q_kbxsumq = shared.d_q_kbxsumq;
    state.d_centroid_distances = shared.d_centroid_distances;
    state.d_short_lut = shared.d_short_lut;
    state.routed_query_count = runtime.query_vectors.rows;
    state.original_query_count = shared.original_query_count;
    state.short_lut_entries_per_query = shared.short_lut_entries_per_query;

    const size_t route_map_bytes = checked_size_product(
        sizeof(uint32_t), state.routed_query_count, "secondary route-to-original query map");
    try {
        check_cuda(cudaMalloc(&state.d_route_to_original_query, route_map_bytes),
                   "cudaMalloc secondary d_route_to_original_query");
        check_cuda(cudaMemcpy(state.d_route_to_original_query,
                              runtime.route_to_original_query.data(),
                              route_map_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy secondary d_route_to_original_query");
        std::printf(
            "[rabitq] borrowed shared query GPU buffers original_queries=%zu "
            "secondary_routes=%zu\n",
            state.original_query_count,
            state.routed_query_count);
    } catch (...) {
        cleanup_prepared_query_state(state);
        throw;
    }
}

void ensure_lowfreq_context_capacity(RabitqLowfreqSearchContext& ctx,
                                     size_t query_count,
                                     size_t candidate_count,
                                     uint32_t shortlist_k)
{
    if (query_count > ctx.query_capacity || shortlist_k > ctx.shortlist_capacity) {
        if (ctx.d_result != nullptr) cudaFree(ctx.d_result);
        if (ctx.d_distances != nullptr) cudaFree(ctx.d_distances);
        if (ctx.d_query_ids != nullptr) cudaFree(ctx.d_query_ids);
        if (ctx.d_candidate_offsets != nullptr) cudaFree(ctx.d_candidate_offsets);
        if (ctx.d_candidate_sizes != nullptr) cudaFree(ctx.d_candidate_sizes);

        check_cuda(cudaMalloc(&ctx.d_result, sizeof(idx_t) * query_count * shortlist_k), "cudaMalloc lowfreq d_result");
        check_cuda(cudaMalloc(&ctx.d_distances, sizeof(float) * query_count * shortlist_k), "cudaMalloc lowfreq d_distances");
        check_cuda(cudaMalloc(&ctx.d_query_ids, sizeof(uint32_t) * query_count), "cudaMalloc lowfreq d_query_ids");
        check_cuda(cudaMalloc(&ctx.d_candidate_offsets, sizeof(uint32_t) * query_count), "cudaMalloc lowfreq d_candidate_offsets");
        check_cuda(cudaMalloc(&ctx.d_candidate_sizes, sizeof(uint32_t) * query_count), "cudaMalloc lowfreq d_candidate_sizes");
        ctx.query_capacity = query_count;
        ctx.shortlist_capacity = shortlist_k;
    }

    if (candidate_count > ctx.candidate_capacity) {
        if (ctx.d_candidate_ids != nullptr) cudaFree(ctx.d_candidate_ids);
        check_cuda(cudaMalloc(&ctx.d_candidate_ids,
                              sizeof(uint32_t) * std::max<size_t>(1, candidate_count)),
                   "cudaMalloc lowfreq d_candidate_ids");
        ctx.candidate_capacity = candidate_count;
    }
}

RabitqLowfreqHostBufferStatus ensure_lowfreq_host_buffers(
    RabitqLowfreqSearchContext& ctx,
    size_t required_slots,
    bool require_approximate_distances)
{
    RabitqLowfreqHostBufferStatus status;
    status.required_slots = required_slots;
    status.approximate_distances_required = require_approximate_distances;
    status.shortlist_reused = lowfreq_merge::ensure_reusable_host_staging_slots(
        ctx.host_shortlist, required_slots);
    if (require_approximate_distances) {
        status.approximate_distances_reused =
            lowfreq_merge::ensure_reusable_host_staging_slots(
                ctx.host_approximate_distances, required_slots);
    }
    return status;
}

void ensure_lowfreq_resident_posting(RabitqLowfreqSearchContext& ctx,
                                     const GraphRuntime& runtime)
{
    const LowfreqPostingIndex& index = runtime.lowfreq_index;
    if (index.label_size.size() != index.label_offset.size()) {
        fail_rabitq("lowfreq resident posting label offset/size length mismatch");
    }
    if (index.label_size.size() != index.group_count) {
        fail_rabitq("lowfreq resident posting group count mismatch");
    }
    if (index.sample_ids.size() != index.total_rows) {
        fail_rabitq("lowfreq resident posting sample_ids length mismatch");
    }
    if (runtime.query_group_ids.size() < runtime.query_vectors.rows) {
        fail_rabitq("lowfreq resident posting query_group_ids length mismatch");
    }
    const size_t group_count = index.label_size.size();
    const size_t row_count = index.sample_ids.size();
    const size_t query_group_count = runtime.query_group_ids.size();
    if (ctx.resident_posting_ready &&
        ctx.resident_group_count == index.group_count &&
        ctx.resident_total_rows == index.total_rows &&
        ctx.query_group_capacity >= query_group_count &&
        ctx.resident_group_capacity >= group_count &&
        ctx.resident_row_capacity >= row_count) {
        return;
    }

    if (query_group_count > ctx.query_group_capacity) {
        if (ctx.d_query_group_ids != nullptr) cudaFree(ctx.d_query_group_ids);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ctx.d_query_group_ids),
                              sizeof(uint32_t) * std::max<size_t>(1, query_group_count)),
                   "cudaMalloc lowfreq resident query_group_ids");
        ctx.query_group_capacity = query_group_count;
    }
    if (group_count > ctx.resident_group_capacity) {
        if (ctx.d_label_size != nullptr) cudaFree(ctx.d_label_size);
        if (ctx.d_label_offset != nullptr) cudaFree(ctx.d_label_offset);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ctx.d_label_size),
                              sizeof(uint32_t) * std::max<size_t>(1, group_count)),
                   "cudaMalloc lowfreq resident label_size");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ctx.d_label_offset),
                              sizeof(uint32_t) * std::max<size_t>(1, group_count)),
                   "cudaMalloc lowfreq resident label_offset");
        ctx.resident_group_capacity = group_count;
    }
    if (row_count > ctx.resident_row_capacity) {
        if (ctx.d_sample_ids != nullptr) cudaFree(ctx.d_sample_ids);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ctx.d_sample_ids),
                              sizeof(uint32_t) * std::max<size_t>(1, row_count)),
                   "cudaMalloc lowfreq resident sample_ids");
        ctx.resident_row_capacity = row_count;
    }
    if (!runtime.query_group_ids.empty()) {
        check_cuda(cudaMemcpy(ctx.d_query_group_ids,
                              runtime.query_group_ids.data(),
                              sizeof(uint32_t) * runtime.query_group_ids.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy lowfreq resident query_group_ids");
    }
    if (!index.label_size.empty()) {
        check_cuda(cudaMemcpy(ctx.d_label_size,
                              index.label_size.data(),
                              sizeof(uint32_t) * index.label_size.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy lowfreq resident label_size");
        check_cuda(cudaMemcpy(ctx.d_label_offset,
                              index.label_offset.data(),
                              sizeof(uint32_t) * index.label_offset.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy lowfreq resident label_offset");
    }
    if (!index.sample_ids.empty()) {
        check_cuda(cudaMemcpy(ctx.d_sample_ids,
                              index.sample_ids.data(),
                              sizeof(uint32_t) * index.sample_ids.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy lowfreq resident sample_ids");
    }
    ctx.resident_group_count = index.group_count;
    ctx.resident_total_rows = index.total_rows;
    ctx.resident_posting_ready = true;
}

void reset_graph_context(RabitqGraphSearchContext& ctx);

bool rabitq_route_is_highfreq_graph(const GraphRuntime& runtime, uint32_t route_idx)
{
    return route_idx < runtime.query_group_ids.size() &&
           runtime.query_group_ids[route_idx] != UINT32_MAX &&
           (route_idx >= runtime.query_is_lowfreq.size() ||
            runtime.query_is_lowfreq[route_idx] == 0) &&
           route_idx < runtime.start_local_ids.size() &&
           runtime.start_local_ids[route_idx] != UINT32_MAX;
}

size_t count_rabitq_highfreq_graph_routes(const GraphRuntime& runtime)
{
    size_t count = 0;
    for (uint32_t route_idx = 0; route_idx < runtime.query_vectors.rows; ++route_idx) {
        if (rabitq_route_is_highfreq_graph(runtime, route_idx)) {
            ++count;
        }
    }
    return count;
}

template<class T>
void upload_predicate_buffer(T*& device, const std::vector<T>& host, const char* name)
{
    if (host.empty()) return;
    check_cuda(cudaMalloc(&device, host.size()*sizeof(T)), name);
    check_cuda(cudaMemcpy(device, host.data(), host.size()*sizeof(T), cudaMemcpyHostToDevice), name);
}

std::shared_ptr<GpuPredicateDeviceIndex> acquire_gpu_predicate_index(
    const GraphRuntime& source, const GraphRuntime& runtime,
    const RabitqSearchOptions& options, const GraphRuntime* hybrid,
    bool& index_reused)
{
    using gpu_predicate::require;
    const bool csr = options.rabitq_predicate_storage == "csr";
    const auto began = std::chrono::steady_clock::now();
    // Validate token ownership even when a prior context populated cache.
    if (csr) {
        require(&source == (hybrid ? hybrid : &runtime), "CSR source ownership mismatch");
        require(source.data_labels.rows == runtime.base_vectors.rows,
                "CSR sample domain differs from graph");
        require(std::none_of(source.label_intervals.has_interval.begin(),
                             source.label_intervals.has_interval.end(),
                             [](uint8_t value) { return value != 0; }),
                "full token CSR cannot use a virtual range-label source");
    }
    // Both stores are reusable across primary/secondary contexts. CSR keeps
    // ALL source labels. Empty/numeric residuals need neither allocation.
    static std::map<std::string,std::shared_ptr<GpuPredicateDeviceIndex>> cache;
    int device=0; check_cuda(cudaGetDevice(&device), "predicate device identity");
    const std::string source_path = options.rabitq_predicate_label_source_path.empty()
        ? options.data_label_spmat_path : options.rabitq_predicate_label_source_path;
    const std::string storage_key = csr ? "csr" :
        "roaring:" + std::filesystem::canonical(options.rabitq_predicate_index_path).string();
    std::string key = storage_key + ":" +
        std::filesystem::canonical(source_path).string() + ":" +
        std::to_string(std::filesystem::file_size(source_path)) + ":" +
        std::to_string(std::filesystem::last_write_time(source_path).time_since_epoch().count()) + ":" +
        std::to_string(source.data_labels.rows) + ":" + std::to_string(source.data_labels.cols) + ":" +
        std::to_string(source.data_labels.indices.size()) + ":" +
        std::to_string(options.specificity_threshold) + ":" +
        std::to_string(device);
    auto found = cache.find(key);
    index_reused = found != cache.end();
    if (found == cache.end()) {
        auto owner = std::make_shared<GpuPredicateDeviceIndex>(); owner->device=device;
        if (csr) {
            const auto& matrix = source.data_labels;
            auto offsets = gpu_predicate::validate_csr_offsets(matrix.rows, matrix.cols,
                matrix.indptr, matrix.indices, runtime.base_vectors.rows);
            owner->resident_bytes = (offsets.size() + matrix.indices.size()) * sizeof(uint32_t);
            upload_predicate_buffer(owner->csr_offsets, offsets, "predicate complete CSR offsets");
            upload_predicate_buffer(owner->csr_labels, matrix.indices, "predicate complete CSR labels");
            std::cout << "[gpu-predicate-index] storage=csr rows=" << matrix.rows
                      << " labels=" << matrix.cols << " relations=" << matrix.indices.size()
                      << " offset_bits=32 label_bits=32 resident_bytes=" << owner->resident_bytes
                      << " load_validate_upload_ms=" << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count() << '\n';
        } else {
            owner->host = gpu_predicate::HostIndex::load(options.rabitq_predicate_index_path,
                source_path, source.data_labels.rows, options.specificity_threshold);
            auto& h=owner->host;
            gpu_predicate::validate_identity_cardinalities(h,source.metadata,source.group_layout,source.data_labels.cols);
            owner->uploaded_labels=h.labels;
            owner->resident_bytes=h.directory.size()*4+h.descriptors.size()*sizeof(gpu_predicate::Descriptor)+h.payload.size()*2;
            upload_predicate_buffer(owner->directory,h.directory,"predicate complete directory");
            upload_predicate_buffer(owner->descriptors,h.descriptors,"predicate complete descriptors");
            upload_predicate_buffer(owner->payload,h.payload,"predicate complete payload");
            std::vector<uint32_t>().swap(h.directory);
            std::vector<gpu_predicate::Descriptor>().swap(h.descriptors);
            std::vector<uint16_t>().swap(h.payload);
            std::cout << "[gpu-predicate-index] rows=" << h.rows << " labels=" << h.labels.size()
                      << " artifact_labels=" << h.labels.size() << " resident_bytes=" << owner->resident_bytes
                      << " load_validate_upload_ms=" << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count() << '\n';
        }
        found=cache.emplace(std::move(key),owner).first;
    }
    auto index=found->second;
    if (!csr) gpu_predicate::validate_identity_cardinalities(index->host,source.metadata,source.group_layout,source.data_labels.cols);

    return index;
}

bool gpu_posting_predicate_supported(const GraphRuntime& runtime,
                                     const RabitqSearchOptions& options)
{
    return (options.rabitq_predicate_mode == "gpu_exact" ||
            options.rabitq_predicate_mode == "gpu_exact_bias") &&
        options.rabitq_predicate_storage == "csr" && options.query_worker_mode == "coupled" && options.graph_transport == "mailbox" &&
        !options.latency_persistent_queue && options.gpu_graph_row_cache_mib == 0;
}

bool needs_label_posting_predicate_context(const GraphRuntime& runtime,
                                          const RabitqSearchOptions& options)
{
    return gpu_posting_predicate_supported(runtime, options) &&
        (options.enable_lowfreq_gpu_posting || lowfreq_gpu_posting_enabled()) &&
        !defaults::lowfreq_full_filter_mode_is_pre(options.lowfreq_full_filter_mode) &&
        !lowfreq_use_filter_mask_enabled() && !runtime.low_query_indices.empty();
}

void init_posting_predicate_context(RabitqPostingPredicateContext& ctx,
                                    const go_fann::route_predicate::Plan& plan,
                                    const GraphRuntime& runtime,
                                    const RabitqSearchOptions& options,
                                    const GraphRuntime* hybrid,
                                    const std::shared_ptr<GpuPredicateDeviceIndex>& graph_index,
                                    const char* provider)
{
    using gpu_predicate::require;
    require(ctx.offsets == nullptr && plan.routes.size() == runtime.query_vectors.rows &&
                plan.label_source != nullptr, "invalid posting predicate context");
    const auto began = std::chrono::steady_clock::now();
    std::vector<uint32_t> offsets{0}, handles;
    std::vector<gpu_predicate::Clause> clauses;
    std::vector<gpu_predicate::RangeAtom> ranges;
    ctx.verified_routes.assign(plan.routes.size(), 0);
    size_t label_routes = 0, verified = 0;
    for (uint32_t route = 0; route < plan.routes.size(); ++route) {
        const auto& compiled = plan.routes[route];
        if (compiled.available) {
            ctx.verified_routes[route] = 1;
            ++verified;
            bool has_labels = false;
            for (const auto& c : compiled.clauses) {
                gpu_predicate::Clause clause{uint32_t(handles.size()), 0, uint32_t(ranges.size()), 0};
                handles.insert(handles.end(), c.labels.begin(), c.labels.end());
                has_labels |= !c.labels.empty();
                for (const auto& a : c.ranges) ranges.push_back({a.dimension, a.mode, a.lower, a.upper});
                require(handles.size() <= UINT32_MAX && ranges.size() <= UINT32_MAX,
                        "posting predicate atom overflow");
                clause.label_end = handles.size(); clause.range_end = ranges.size();
                clauses.push_back(clause);
            }
            label_routes += has_labels;
        }
        require(clauses.size() <= UINT32_MAX, "posting predicate clause overflow");
        offsets.push_back(clauses.size());
    }
    bool index_reused = false;
    if (!handles.empty()) {
        ctx.index = acquire_gpu_predicate_index(*plan.label_source, runtime, options,
                                                hybrid, index_reused);
    }
    // Keep an explicit DNF marker for enabled, always-false routes too.
    if (clauses.empty()) clauses.push_back({0, 0, 0, 0});
    upload_predicate_buffer(ctx.offsets, offsets, "posting predicate route offsets");
    upload_predicate_buffer(ctx.handles, handles, "posting predicate labels");
    upload_predicate_buffer(ctx.clauses, clauses, "posting predicate clauses");
    upload_predicate_buffer(ctx.ranges, ranges, "posting predicate ranges");
    upload_predicate_buffer(ctx.enabled, ctx.verified_routes, "posting predicate enabled routes");
    auto& view = ctx.view;
    view.route_offsets = ctx.offsets; view.route_handles = ctx.handles;
    view.clauses = ctx.clauses; view.ranges = ctx.ranges; view.route_enabled = ctx.enabled;
    view.rows = runtime.base_vectors.rows; view.range_stride = runtime.base_range_count;
    if (ctx.index) { view.csr_offsets = ctx.index->csr_offsets; view.csr_labels = ctx.index->csr_labels; }
    // Posting is exact admission only: no navigation alpha, graph counters, or
    // graph-rerank trust is propagated from this separately compiled proof.
    std::cout << "[gpu-posting-predicate] storage=csr mode=" << options.posting_predicate_order
              << " provider=" << provider
              << " verified_routes=" << verified
              << " label_routes=" << (std::string_view(provider) == "label" ? verified : 0)
              << " range_routes=" << (std::string_view(provider) == "range" ? verified : 0)
              << " remaining_label_routes=" << label_routes
              << " csr_resident_bytes=" << (ctx.index ? ctx.index->resident_bytes : 0)
              << " shared_with_graph=" << (ctx.index && ctx.index == graph_index ? 1 : 0)
              << " index_reused=" << index_reused << " cpu_final_validation=unverified_routes_only"
              << " plan_upload_ms=" << elapsed_ms(began, std::chrono::steady_clock::now()) << '\n';
}

void init_range_posting_predicate_context(RabitqGraphSearchContext& ctx,
                                          const RabitqRangePostingPlan& posting_plan,
                                          const GraphRuntime& runtime,
                                          const RabitqSearchOptions& options)
{
    if (!gpu_posting_predicate_supported(runtime, options) ||
        posting_plan.representative_route_ids.empty()) return;
    const auto plan = go_fann::route_predicate::compile(
        runtime, go_fann::route_predicate::Producer::RangePosting,
        ctx.hybrid_residual_runtime, posting_plan.representative_route_ids, posting_plan.original_query_ids);
    init_posting_predicate_context(ctx.range_posting_predicate, plan, runtime, options,
                                  ctx.hybrid_residual_runtime, ctx.predicate_index, "range");
    gpu_predicate::require(ctx.range_posting_predicate.ranges == nullptr ||
                           ctx.d_base_range_values != nullptr,
                           "posting numeric predicate requires resident original numeric values");
}

void init_gpu_predicate_context(RabitqGraphSearchContext& ctx,
                                const GraphRuntime& runtime,
                                const RabitqSearchOptions& options,
                                const GraphRuntime* hybrid)
{
    using gpu_predicate::require;
    namespace rp = go_fann::route_predicate;
    const bool gpu = options.rabitq_predicate_mode == "gpu_exact" ||
                     options.rabitq_predicate_mode == "gpu_exact_bias";
    if (options.rabitq_predicate_diagnostics) {
        require(options.query_worker_mode == "coupled", "predicate diagnostics require coupled kernel");
        if (runtime.query_vectors.rows) {
            check_cuda(cudaMalloc(&ctx.predicate_counters,
                runtime.query_vectors.rows*sizeof(gpu_predicate::Counters)), "predicate counters");
            ctx.predicate_view.counters = ctx.predicate_counters;
        }
    }
    require(options.rabitq_predicate_storage == "roaring" || options.rabitq_predicate_storage == "csr",
            "unknown predicate storage");
    const bool csr = options.rabitq_predicate_storage == "csr";
    if (options.rabitq_predicate_mode == "cpu") return;
    require(gpu || options.rabitq_predicate_mode == "cpu_simplified", "unknown predicate mode");
    require(options.query_worker_mode == "coupled" && options.graph_transport == "mailbox" &&
            options.gpu_graph_row_cache_mib == 0,
            "predicate modes require coupled memory mailbox without graph cache");
    require((highfreq_cpu_rerank_enabled(options) ||
             (gpu && highfreq_inline_gpu_exact_rerank_enabled(options, runtime.vector_scalar_type)) ||
             highfreq_staged_float_gpu_exact_rerank_enabled(options, runtime.vector_scalar_type)) &&
            options.per_step_full_filter && (options.full_filter || hybrid != nullptr),
            "predicate modes require full-filter CPU, GPU-inline or float GPU-post rerank");
    require(options.highfreq_rerank_algorithm == "exact", "heuristic rerank unsupported");
    require(!mailbox_rtt_profile_enabled(), "mailbox RTT instrumentation unsupported with predicate modes");

    const auto began = std::chrono::steady_clock::now();
    ctx.predicate_plan = rp::compile(runtime, rp::Producer::Graph, hybrid);
    const auto& plan = ctx.predicate_plan;
    require(plan.routes.size() == runtime.query_vectors.rows && plan.label_source,
            "route residual plan dimensions/source missing");
    const auto& source = *plan.label_source;
    const uint32_t route_count = runtime.query_vectors.rows;
    ctx.predicate_verified_routes.assign(route_count, 0);
    std::vector<uint8_t> prefetch_disabled(route_count, 0);
    std::vector<uint8_t> compiled(route_count, 0);
    std::vector<uint32_t> needed;
    size_t graph_routes=0, entailed=0, label_routes=0, range_routes=0;
    const auto indexed_label = [&](uint32_t label) {
        if (label >= source.data_labels.cols) return false;
        // Full original-token CSR covers labels independently of graph grouping
        // (including low-frequency residuals). Roaring keeps its index contract.
        if (csr) return true;
        if (options.rabitq_predicate_index_path.empty() ||
            label >= source.group_layout.label_to_group.size()) return false;
        const auto group = source.group_layout.label_to_group[label];
        return group < source.group_layout.groups.size() &&
            source.group_layout.groups[group].size() == 1 &&
            source.group_layout.groups[group][0] == label &&
            group < source.metadata.label_size.size() && source.metadata.label_size[group] > 0;
    };
    for (uint32_t route=0; route<route_count; ++route) {
        if (!rabitq_route_is_highfreq_graph(runtime, route) ||
            (!runtime.route_enabled.empty() && runtime.route_enabled.at(route) == 0) ||
            (!runtime.route_execution_modes.empty() &&
             runtime.route_execution_modes.at(route) != QueryRouteExecution::GraphTraversal)) continue;
        ++graph_routes;
        const auto& r = plan.routes[route];
        compiled[route] = r.available;
        bool has_labels=false, has_ranges=false, supported=r.available;
        for (const auto& c : r.clauses) {
            has_labels |= !c.labels.empty(); has_ranges |= !c.ranges.empty();
            for (uint32_t label : c.labels) supported &= indexed_label(label);
        }
        entailed += r.available && r.entailed;
        label_routes += has_labels; range_routes += has_ranges;
        if (gpu && supported) {
            ctx.predicate_verified_routes[route] = 1;
            for (const auto& c : r.clauses) needed.insert(needed.end(),c.labels.begin(),c.labels.end());
            if (!r.entailed && options.rabitq_predicate_bias_alpha > 0)
                ctx.predicate_bias_frontier = true;
        } else if (!r.available || !r.entailed) {
            // A static-prefetch hit supplies adjacency but no fresh CPU verdict.
            // Keep exact CPU fallback by demanding these residual-bearing rows.
            const auto group = runtime.query_group_ids.at(route);
            prefetch_disabled[route] = group < runtime.prefetch_begin_by_group.size() &&
                group < runtime.prefetch_end_by_group.size() &&
                runtime.prefetch_end_by_group[group] > runtime.prefetch_begin_by_group[group];
        }
    }
    std::sort(needed.begin(),needed.end());
    needed.erase(std::unique(needed.begin(),needed.end()),needed.end());
    bool index_reused = false;
    if (!needed.empty()) {
        ctx.predicate_index = acquire_gpu_predicate_index(source, runtime, options,
                                                          hybrid, index_reused);
    }

    std::vector<uint32_t> offsets{0}, handles;
    std::vector<gpu_predicate::Clause> clauses;
    std::vector<gpu_predicate::RangeAtom> ranges;
    for (uint32_t route=0;route<route_count;++route) {
        if (ctx.predicate_verified_routes[route]) {
            for (const auto& c : plan.routes[route].clauses) {
                gpu_predicate::Clause clause{uint32_t(handles.size()),0,uint32_t(ranges.size()),0};
                for (auto label : c.labels) {
                    if (csr) { handles.push_back(label); continue; }
                    const auto& labels=ctx.predicate_index->uploaded_labels;
                    const auto it=std::lower_bound(labels.begin(),labels.end(),label);
                    require(it!=labels.end() && *it==label,"eligible residual label absent from index");
                    handles.push_back(it-labels.begin());
                }
                for (const auto& a : c.ranges) ranges.push_back({a.dimension,a.mode,a.lower,a.upper});
                require(handles.size()<=UINT32_MAX && ranges.size()<=UINT32_MAX,"residual atom overflow");
                clause.label_end=handles.size(); clause.range_end=ranges.size(); clauses.push_back(clause);
            }
        }
        require(clauses.size()<=UINT32_MAX,"residual clause overflow"); offsets.push_back(clauses.size());
    }
    // A non-null clauses pointer also identifies DNF for all-false plans.
    if (clauses.empty()) clauses.push_back({0,0,0,0});
    upload_predicate_buffer(ctx.predicate_offsets,offsets,"predicate route offsets");
    upload_predicate_buffer(ctx.predicate_handles,handles,"predicate residual handles");
    upload_predicate_buffer(ctx.predicate_clauses,clauses,"predicate DNF clauses");
    upload_predicate_buffer(ctx.predicate_ranges,ranges,"predicate range atoms");
    upload_predicate_buffer(ctx.predicate_enabled,ctx.predicate_verified_routes,"predicate verified routes");
    upload_predicate_buffer(ctx.predicate_prefetch_disabled,prefetch_disabled,"predicate prefetch fallback routes");
    upload_predicate_buffer(ctx.predicate_compiled,compiled,"predicate compiled routes");
    auto& v=ctx.predicate_view;
    if (ctx.predicate_index) {
        v.directory=ctx.predicate_index->directory; v.descriptors=ctx.predicate_index->descriptors;
        v.payload=ctx.predicate_index->payload; v.chunks=ctx.predicate_index->host.chunks;
        v.csr_offsets=ctx.predicate_index->csr_offsets; v.csr_labels=ctx.predicate_index->csr_labels;
    }
    v.route_offsets=ctx.predicate_offsets; v.route_handles=ctx.predicate_handles;
    v.clauses=ctx.predicate_clauses; v.ranges=ctx.predicate_ranges;
    v.route_enabled=ctx.predicate_enabled; v.route_prefetch_disabled=ctx.predicate_prefetch_disabled;
    v.route_compiled=ctx.predicate_compiled;
    v.rows=runtime.base_vectors.rows; v.range_stride=runtime.base_range_count;
    v.alpha=options.rabitq_predicate_bias_alpha;
    v.pending_base_stop=options.rabitq_predicate_mode=="gpu_exact_bias" && options.search_early_stop_mode=="adaptive";
    const size_t verified=std::count(ctx.predicate_verified_routes.begin(),ctx.predicate_verified_routes.end(),uint8_t{1});
    ctx.shortlist_predicate_verified=graph_routes>0 && verified==graph_routes;
    std::cout << "[gpu-predicate-plan] graph_routes=" << graph_routes << " entailed_routes=" << entailed
              << " gpu_verified_routes=" << verified << " cpu_fallback_routes=" << graph_routes-verified
              << " remaining_label_routes=" << label_routes << " range_routes=" << range_routes
              << " prefetch_cached_routes=0 prefetch_disabled_routes=" << std::count(prefetch_disabled.begin(),prefetch_disabled.end(),uint8_t{1})
              << " prefetch_verdict_bytes=0 bitmap_resident_bytes=" << (!csr && ctx.predicate_index?ctx.predicate_index->resident_bytes:0)
              << " storage=" << options.rabitq_predicate_storage
              << " csr_resident_bytes=" << (csr && ctx.predicate_index?ctx.predicate_index->resident_bytes:0)
              << " index_reused=" << index_reused
              << " index_uploaded=" << (ctx.predicate_index?1:0) << " bias_frontier=" << ctx.predicate_bias_frontier << " pending_base_stop=" << v.pending_base_stop
              << " rerank_predicate=" << (ctx.shortlist_predicate_verified?"off":"retained")
              << " plan_upload_ms=" << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count() << '\n';
}

void validate_gpu_predicate_trace(const RabitqGraphSearchContext& ctx,
                                   const GraphRuntime& runtime,
                                   const char* run_label)
{
    if (!ctx.predicate_counters) return;
    std::vector<gpu_predicate::Counters> counters(runtime.query_vectors.rows);
    check_cuda(cudaMemcpy(counters.data(),ctx.predicate_counters,counters.size()*sizeof(counters[0]),cudaMemcpyDeviceToHost), "predicate diagnostic download");
    uint64_t expanded=0, proposed=0, passed=0, ns=0, saturated=0, checked=0, scores=0;
    uint64_t probes=0, compares=0, entailed=0;
    for (uint32_t route=0; route<counters.size(); ++route) {
        const auto& c=counters[route]; expanded+=c.expanded; proposed+=c.proposed;
        passed+=c.passed; ns+=c.lookup_ns; saturated+=c.saturated; scores+=c.scores;
        probes+=c.label_probes; compares+=c.range_compares; entailed+=c.entailed_fastpaths;
        for (uint32_t t=0;t<c.trace_count;++t) {
            gpu_predicate::require(bool(c.trace[t].pass)==go_fann::route_predicate::evaluate(
                ctx.predicate_plan,runtime,route,c.trace[t].sample), "ANN candidate residual CSR/range mismatch");
            ++checked;
        }
    }
    std::cout << "[gpu-predicate-diagnostics] run=" << run_label << " selected_membership_checks=" << expanded
              << " proposed_checks=" << proposed << " selected_membership_pass=" << passed << " lookup_warp_ns_sum=" << ns
              << " approximate_scores=" << scores << " saturation=" << saturated << " csr_trace_checked=" << checked
              << " bitmap_literal_probes=" << probes << " numeric_compares=" << compares
              << " entailed_fastpaths=" << entailed << " mismatches=0\n";
}

void init_graph_context(RabitqGraphSearchContext& ctx,
                        const RabitqHostIndex& index,
                        const GraphRuntime& runtime,
                        const RabitqSearchOptions& options,
                        size_t graph_scratch_capacity,
                        const RabitqGraphSearchContext* shared_base_range_ctx = nullptr,
                        const GraphRuntime* hybrid_residual_runtime = nullptr,
                        const RabitqGpuGraphRowCacheView* shared_gpu_graph_row_cache = nullptr)
{
    if (ctx.initialized) {
        fail_rabitq("RabitQ graph context is already initialized");
    }
    const size_t total_query_count = runtime.query_vectors.rows;
    if (total_query_count == 0) {
        fail_rabitq("query batch is empty");
    }
    if (total_query_count > std::numeric_limits<uint32_t>::max()) {
        fail_rabitq("query batch exceeds supported size");
    }
    const size_t highfreq_graph_route_count =
        count_rabitq_highfreq_graph_routes(runtime);
    if (graph_scratch_capacity < highfreq_graph_route_count) {
        fail_rabitq("graph scratch capacity is smaller than the high-frequency graph route count");
    }
    if (graph_scratch_capacity == 0 ||
        graph_scratch_capacity > std::numeric_limits<uint32_t>::max()) {
        fail_rabitq("graph scratch capacity must be in [1, UINT32_MAX]");
    }
    if (runtime.query_vectors.cols != index.header.dim ||
        static_cast<uint32_t>(options.dim) != index.header.dim) {
        fail_rabitq("query/search dim does not match RabitQ artifact dim");
    }

    if (options.full_filter && (rabitq_device_resident_graph_enabled(options) ||
                                options.gpu_graph_row_cache_mib != 0)) {
        for (uint32_t route : runtime.high_query_indices) {
            if (runtime.route_clause_indptr.size() == total_query_count + 1 &&
                runtime.route_clause_indptr[route + 1] - runtime.route_clause_indptr[route] > 1) {
                fail_rabitq("shared OR residuals require mailbox graph transport without GPU graph row cache");
            }
        }
    }
    init_gpu_predicate_context(ctx, runtime, options, hybrid_residual_runtime);
    if (needs_label_posting_predicate_context(runtime, options)) {
        const auto plan = go_fann::route_predicate::compile(
            runtime, go_fann::route_predicate::Producer::LabelPosting, hybrid_residual_runtime);
        init_posting_predicate_context(ctx.label_posting_predicate, plan, runtime, options,
                                      hybrid_residual_runtime, ctx.predicate_index, "label");
    }
    ctx.use_inline_gpu_exact_rerank =
        highfreq_inline_gpu_exact_rerank_enabled(options, runtime.vector_scalar_type);
    ctx.result_buffer_is_host_mapped = highfreq_cpu_streaming_enabled(options);
    ctx.device_resident_graph = rabitq_device_resident_graph_enabled(options);
    ctx.hybrid_residual_runtime = hybrid_residual_runtime;
    if (shared_base_range_ctx != nullptr && shared_base_range_ctx->cpu_worker_budget != nullptr) {
        ctx.cpu_worker_budget = shared_base_range_ctx->cpu_worker_budget;
    } else {
        ctx.cpu_worker_budget = std::make_shared<RabitqCpuWorkerBudget>(
            static_cast<size_t>(std::max(1, options.highfreq_rerank_threads)));
    }
    prepare_cpu_rerank_workers(ctx, options, shared_base_range_ctx);
    ctx.total_query_capacity = total_query_count;
    ctx.graph_scratch_capacity = graph_scratch_capacity;
    ctx.graph_hash_scratch_bytes = checked_size_product(
        checked_size_product(ctx.graph_scratch_capacity,
                             static_cast<size_t>(kHashTableCapacity),
                             "RabitQ graph hash scratch"),
        sizeof(uint32_t),
        "RabitQ graph hash scratch");
    ctx.graph_datadist_scratch_bytes = checked_size_product(
        checked_size_product(ctx.graph_scratch_capacity,
                             kGraphDatadistSlotCapacity,
                             "RabitQ graph distance scratch"),
        sizeof(dist_t),
        "RabitQ graph distance scratch");
    ctx.result_width = highfreq_result_width(options, runtime);
    ctx.stop_width = highfreq_stop_width(options);
    ctx.mailbox_rtt_profile = mailbox_rtt_profile_enabled() && !ctx.device_resident_graph;
    ctx.phase_snapshot_profile =
        graph_phase_snapshot_profile_enabled() && !ctx.device_resident_graph;
    ctx.finite_query_state_enabled = options.query_worker_mode == "finite";
    ctx.request_ring_enabled = options.request_mode == "batch_ring";
    ctx.request_ring_consumed_device_mapped =
        options.request_consumed_memory == "device";
    const size_t gpu_graph_row_cache_mib = options.gpu_graph_row_cache_mib;
    if (ctx.request_ring_enabled &&
        (ctx.finite_query_state_enabled || options.request_batch_queries == 0)) {
        fail_rabitq("RabitQ batch request ring requires coupled memory workers with a positive batch size");
    }
    if (ctx.request_ring_enabled) {
        ctx.request_ring_capacity_per_ring = options.request_batch_queries;
        ctx.request_ring_capacity_count =
            (ctx.graph_scratch_capacity + ctx.request_ring_capacity_per_ring - 1) /
            ctx.request_ring_capacity_per_ring;
    }
    RabitqGpuLabelCsr gpu_query_label_csr;
    RabitqGpuLabelCsr gpu_sample_label_csr;
    if (ctx.device_resident_graph) {
        const size_t expected_graph_values = checked_size_product(
            static_cast<size_t>(runtime.graph.rows),
            static_cast<size_t>(runtime.graph.cols),
            "resident RabitQ graph");
        if (runtime.graph.data.empty() ||
            runtime.graph.rows != runtime.metadata.total_rows ||
            runtime.graph.cols == 0 || runtime.graph.cols > kRabitqGpuGraphRowCacheDegree ||
            runtime.graph.cols != static_cast<uint32_t>(options.degree) ||
            runtime.graph.data.size() != expected_graph_values ||
            ctx.finite_query_state_enabled || ctx.request_ring_enabled ||
            ctx.use_inline_gpu_exact_rerank || options.latency_persistent_queue ||
            hybrid_residual_runtime != nullptr || gpu_graph_row_cache_mib != 0) {
            fail_rabitq(
                "device_resident RabitQ requires a materialized degree<=32 graph matching "
                "the configured degree, coupled scan workers, CPU rerank/off, non-persistent "
                "execution, no GPU graph row cache, and no hybrid residual");
        }
        ctx.resident_graph_degree = runtime.graph.cols;
        if (options.full_filter) {
            gpu_query_label_csr = build_rabitq_gpu_query_label_csr(
                runtime.query_labels_matrix, runtime.query_filter_labels);
            gpu_sample_label_csr = build_rabitq_gpu_sample_label_csr(runtime.data_labels);
        }
        const size_t graph_bytes = runtime.graph.data.size() * sizeof(uint32_t);
        std::printf(
            "[rabitq] graph transport=device_resident rows=%zu degree=%zu "
            "graph_bytes=%zu graph_gib=%.3f rerank=%s full_filter=%d\n",
            static_cast<size_t>(runtime.graph.rows),
            static_cast<size_t>(runtime.graph.cols),
            graph_bytes,
            static_cast<double>(graph_bytes) /
                static_cast<double>(size_t{1} << 30),
            highfreq_cpu_rerank_enabled(options) ? "cpu_streaming" : "off",
            options.full_filter ? 1 : 0);
    }
    if (gpu_graph_row_cache_mib > 0) {
        if (ctx.finite_query_state_enabled ||
            !ctx.request_ring_enabled || ctx.use_inline_gpu_exact_rerank ||
            options.latency_persistent_queue || hybrid_residual_runtime != nullptr) {
            fail_rabitq(
                "GPU graph row cache currently requires coupled memory graph search with "
                "batch_ring, CPU rerank/off, non-persistent execution, and no hybrid residual");
        }
        if (options.degree > static_cast<int>(kRabitqGpuGraphRowCacheDegree)) {
            fail_rabitq("GPU graph row cache currently supports graph degree <= 32");
        }
        ctx.gpu_graph_row_cache_enabled = true;
        if (shared_gpu_graph_row_cache != nullptr) {
            if (shared_gpu_graph_row_cache->entries == nullptr ||
                shared_gpu_graph_row_cache->capacity == 0) {
                fail_rabitq("shared GPU graph row cache view is empty");
            }
            ctx.d_gpu_graph_row_cache = shared_gpu_graph_row_cache->entries;
            ctx.gpu_graph_row_cache_capacity = shared_gpu_graph_row_cache->capacity;
            ctx.owns_gpu_graph_row_cache = false;
        } else {
            const size_t requested_bytes = checked_size_product(
                gpu_graph_row_cache_mib, size_t{1} << 20, "RabitQ GPU graph row cache");
            const size_t capacity = requested_bytes / sizeof(RabitqGpuGraphRowCacheEntry);
            if (capacity == 0 || capacity > std::numeric_limits<uint32_t>::max()) {
                fail_rabitq("GPU graph row cache capacity is outside the uint32-supported range");
            }
            ctx.gpu_graph_row_cache_capacity = static_cast<uint32_t>(capacity);
        }
        if (options.full_filter) {
            gpu_query_label_csr = build_rabitq_gpu_query_label_csr(
                runtime.query_labels_matrix, runtime.query_filter_labels);
            gpu_sample_label_csr = build_rabitq_gpu_sample_label_csr(runtime.data_labels);
        }
        const size_t actual_bytes = static_cast<size_t>(ctx.gpu_graph_row_cache_capacity) *
            sizeof(RabitqGpuGraphRowCacheEntry);
        std::printf(
            "[rabitq] GPU graph row cache: requested_mib=%zu capacity_rows=%zu "
            "entry_bytes=%zu actual_mib=%.2f full_filter=%d query_label_nnz=%zu "
            "sample_label_nnz=%zu backing=%s reset_between_runs=0\n",
            gpu_graph_row_cache_mib,
            static_cast<size_t>(ctx.gpu_graph_row_cache_capacity),
            sizeof(RabitqGpuGraphRowCacheEntry),
            static_cast<double>(actual_bytes) / static_cast<double>(size_t{1} << 20),
            options.full_filter ? 1 : 0,
            gpu_query_label_csr.indices.size(),
            gpu_sample_label_csr.indices.size(),
            ctx.owns_gpu_graph_row_cache ? "owned" : "shared");
    }
    if (ctx.mailbox_rtt_profile && ctx.finite_query_state_enabled) {
        fail_rabitq("mailbox RTT profiling currently requires coupled query workers");
    }
    if (ctx.finite_query_state_enabled) {
        ctx.finite_queue_capacity = options.search_queue_size <= 64 ? 64U
            : options.search_queue_size <= 128 ? 128U
            : options.search_queue_size <= 256 ? 256U
            : options.search_queue_size <= 512 ? 512U : 0U;
        if (ctx.finite_queue_capacity == 0) {
            fail_rabitq("finite query state requires search_queue_size in [1, 512]");
        }
        ctx.finite_result_capacity = ctx.result_width;
        ctx.finite_ready_token_capacity = std::min<size_t>(ctx.graph_scratch_capacity, 4096);
        ctx.finite_mutable_bytes_per_query =
            sizeof(RabitqFiniteContextControl) +
            (static_cast<size_t>(ctx.finite_queue_capacity) + 2) * sizeof(RabitqPoolPair) +
            (static_cast<size_t>(kTopk) + 1) * sizeof(RabitqStopPair) +
            (static_cast<size_t>(ctx.finite_result_capacity) + 1) * sizeof(RabitqResultPair);
        ctx.finite_mutable_bytes_total = checked_size_product(
            ctx.finite_mutable_bytes_per_query,
            ctx.graph_scratch_capacity,
            "RabitQ finite mutable state");
    }
    if (ctx.request_ring_enabled) {
        printf("[rabitq] request ring: mode=batch_ring capacity_per_ring=%zu max_rings=%zu "
               "entry_bytes=%zu consumed_memory=%s\n",
               ctx.request_ring_capacity_per_ring,
               ctx.request_ring_capacity_count,
               sizeof(HostRequestRingEntry),
               options.request_consumed_memory.c_str());
    }
    const size_t native_batch_dynamic_shared_bytes =
        rabitq_result_heap_shared_bytes(ctx.result_width);
    const uint32_t mailbox_prefetch_hops =
        ctx.use_inline_gpu_exact_rerank &&
            dense_vector_scalar_is_float(runtime.vector_scalar_type)
        ? 0U
        : static_cast<uint32_t>(options.prefetch_hops);
    ctx.io_config = AsyncIoConfig(static_cast<uint32_t>(options.dim),
                                  static_cast<uint32_t>(options.degree),
                                  mailbox_prefetch_hops);
    ctx.io_config.payload_slot_count = 2;
    ctx.io_config.host_service_threads = options.host_service_threads;
    ctx.io_config.host_service_cpu_start = options.host_service_cpu_start;
    ctx.io_config.host_service_cpu_count = options.host_service_cpu_count;
    ctx.io_config.host_demand_batch_size = options.host_demand_batch_size;
    ctx.io_config.host_demand_batch_timeout_us = options.host_demand_batch_timeout_us;
    ctx.io_config.per_step_full_filter = options.per_step_full_filter;
    if (options.rabitq_predicate_mode != "cpu") {
        ctx.io_config.predicate_plan = &ctx.predicate_plan;
        ctx.io_config.predicate_gpu_verified_routes = &ctx.predicate_verified_routes;
    }
    ctx.io_config.enable_full_label_check =
        options.full_filter || hybrid_residual_runtime != nullptr;
    ctx.io_config.publish_valid_vector_payload = ctx.use_inline_gpu_exact_rerank;
    ctx.io_config.skip_filtered_vector_payload = options.skip_filtered_vector_payload;
    ctx.io_config.set_vector_payload_scalar_type(runtime.vector_scalar_type);
    ctx.io_config.set_reserve_vector_payload(ctx.use_inline_gpu_exact_rerank);

    // A shared OR may clear the flat range prefilter while retaining numeric
    // atoms inside individual clauses. Those atoms still need resident values.
    const bool has_active_range_filters = query_ranges_have_active_filters(runtime.query_ranges) ||
        ctx.predicate_ranges != nullptr || ctx.label_posting_predicate.ranges != nullptr;
    gpu_predicate::require(!has_active_range_filters ||
        (runtime.base_range_count == runtime.query_ranges.range_count &&
         runtime.base_range_values.size() ==
             static_cast<size_t>(runtime.base_vectors.rows) * runtime.base_range_count),
        "compiled numeric residual requires aligned original numeric data");
    const uint32_t device_range_count = has_active_range_filters ? runtime.query_ranges.range_count : 0;
    ctx.device_range_count = device_range_count;

    const size_t gdr_bytes = ctx.total_query_capacity * sizeof(int64_t) * kGdrMailboxStride;
    ctx.mapped_gdr_bytes = round_up_gdr_bytes(gdr_bytes);
    const size_t neighbor_bytes =
        ctx.total_query_capacity * ctx.io_config.block_size * ctx.io_config.payload_slot_count;
    ctx.mapped_neighbor_bytes = round_up_gdr_bytes(neighbor_bytes);
    printf("[rabitq] graph mailbox layout: slot_bytes=%zu slot_words=%zu vector_words=%zu vector_scalar=%s elems_per_node=%zu payload_slots=%u neighbor_bytes=%zu mapped_neighbor_bytes=%zu\n",
           ctx.io_config.block_size,
           ctx.io_config.elems_per_block,
           ctx.io_config.vector_payload_words,
           dense_vector_scalar_type_name(ctx.io_config.vector_payload_scalar_type),
           ctx.io_config.elems_per_node,
           ctx.io_config.payload_slot_count,
           neighbor_bytes,
           ctx.mapped_neighbor_bytes);
    printf("[rabitq] graph scratch: routed_queries=%zu highfreq_routes=%zu slots=%zu hash_bytes=%zu datadist_bytes=%zu\n",
           ctx.total_query_capacity,
           highfreq_graph_route_count,
           ctx.graph_scratch_capacity,
           ctx.graph_hash_scratch_bytes,
           ctx.graph_datadist_scratch_bytes);
    if (ctx.finite_query_state_enabled) {
        printf("[rabitq] finite state: queue_capacity=%u result_capacity=%u "
               "bytes_per_slot=%zu total_bytes=%zu ready_batch_capacity=%zu\n",
               ctx.finite_queue_capacity,
               ctx.finite_result_capacity,
               ctx.finite_mutable_bytes_per_query,
               ctx.finite_mutable_bytes_total,
               ctx.finite_ready_token_capacity);
    }

    int max_blocks_per_sm = 0;
    dispatch_rabitq_kernel_by_queue_size(options.search_queue_size, [&]<int QS>() {
        const auto measure_occupancy = [&]<bool kUseInt8Score>() {
            if (ctx.device_resident_graph) {
                check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &max_blocks_per_sm,
                               native_rabitq_search_kernel<QS, false, false, kUseInt8Score, true>,
                               kNativeThreadInBlock,
                               native_batch_dynamic_shared_bytes),
                           "cudaOccupancyMaxActiveBlocksPerMultiprocessor resident native RabitQ");
            } else if (ctx.request_ring_enabled && ctx.mailbox_rtt_profile) {
                check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &max_blocks_per_sm,
                               native_rabitq_search_kernel<QS, true, true, kUseInt8Score, false>,
                               kNativeThreadInBlock,
                               native_batch_dynamic_shared_bytes),
                           "cudaOccupancyMaxActiveBlocksPerMultiprocessor profiled native RabitQ");
            } else if (ctx.request_ring_enabled && ctx.predicate_bias_frontier) {
                check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &max_blocks_per_sm,
                               native_rabitq_search_kernel<QS, false, true, kUseInt8Score, false, true>,
                               kNativeThreadInBlock, native_batch_dynamic_shared_bytes),
                           "cudaOccupancyMaxActiveBlocksPerMultiprocessor biased ring RabitQ");
            } else if (ctx.request_ring_enabled) {
                check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &max_blocks_per_sm,
                               native_rabitq_search_kernel<QS, false, true, kUseInt8Score, false>,
                               kNativeThreadInBlock,
                               native_batch_dynamic_shared_bytes),
                           "cudaOccupancyMaxActiveBlocksPerMultiprocessor request-ring native RabitQ");
            } else if (ctx.mailbox_rtt_profile) {
                check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &max_blocks_per_sm,
                               native_rabitq_search_kernel<QS, true, false, kUseInt8Score, false>,
                               kNativeThreadInBlock,
                               native_batch_dynamic_shared_bytes),
                           "cudaOccupancyMaxActiveBlocksPerMultiprocessor profiled native RabitQ");
            } else if (ctx.predicate_bias_frontier) {
                check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &max_blocks_per_sm,
                               native_rabitq_search_kernel<QS, false, false, kUseInt8Score, false, true>,
                               kNativeThreadInBlock,
                               native_batch_dynamic_shared_bytes),
                           "cudaOccupancyMaxActiveBlocksPerMultiprocessor biased native RabitQ");
            } else {
                check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &max_blocks_per_sm,
                               native_rabitq_search_kernel<QS, false, false, kUseInt8Score, false>,
                               kNativeThreadInBlock,
                               native_batch_dynamic_shared_bytes),
                           "cudaOccupancyMaxActiveBlocksPerMultiprocessor native RabitQ");
            }
        };
        if (options.use_int8_score) {
            measure_occupancy.template operator()<true>();
        } else {
            measure_occupancy.template operator()<false>();
        }
    });
    printf("[rabitq] native multi-query: graph_transport=%s queries_per_block=%d threads_per_block=%d "
           "detailed_timing=%d mailbox_rtt_profile=%d wait_policy=%s dynamic_shared_bytes=%zu "
           "active_blocks_per_sm=%d "
           "active_query_warps_per_sm=%d\n",
           ctx.device_resident_graph ? "device_resident" : "mailbox",
           kNativeMultiQuery,
           kNativeThreadInBlock,
           kNativeDetailedTiming ? 1 : 0,
           ctx.mailbox_rtt_profile ? 1 : 0,
           options.host_reply_wait_policy.c_str(),
           native_batch_dynamic_shared_bytes,
           max_blocks_per_sm,
           max_blocks_per_sm * kNativeMultiQuery);

    try {
        init_float_gpu_rerank_context(ctx.float_gpu_rerank, options, runtime);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ctx.d_gdr_data_raw),
                              ctx.mapped_gdr_bytes + kGdrAlignBytes),
                   "cudaMalloc d_gdr_data_raw");
        ctx.d_gdr_data = align_gdr_pointer(ctx.d_gdr_data_raw);
        ctx.g = gdr_open();
        if (ctx.g == nullptr) fail_rabitq("gdr_open returned nullptr");
        check_gdr(gdr_pin_buffer(ctx.g,
                                 reinterpret_cast<unsigned long>(ctx.d_gdr_data),
                                 ctx.mapped_gdr_bytes,
                                 0,
                                 0,
                                 &ctx.mh),
                  "gdr_pin_buffer d_gdr_data");
        ctx.mh_pinned = true;
        check_gdr(gdr_map(ctx.g, ctx.mh, &ctx.cpu_ptr, ctx.mapped_gdr_bytes),
                  "gdr_map d_gdr_data");
        ctx.h_gdr_data = reinterpret_cast<int64_t*>(ctx.cpu_ptr);

        if (ctx.result_buffer_is_host_mapped) {
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ctx.h_result_mapped),
                                     sizeof(idx_t) * ctx.total_query_capacity * ctx.result_width,
                                     cudaHostAllocMapped),
                       "cudaHostAlloc h_result_mapped");
            check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ctx.d_result),
                                                ctx.h_result_mapped,
                                                0),
                       "cudaHostGetDevicePointer d_result");
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ctx.h_result_distances_mapped),
                                     sizeof(dist_t) * ctx.total_query_capacity * ctx.result_width,
                                     cudaHostAllocMapped),
                       "cudaHostAlloc h_result_distances_mapped");
            check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ctx.d_result_distances),
                                                ctx.h_result_distances_mapped,
                                                0),
                       "cudaHostGetDevicePointer d_result_distances");
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ctx.h_time_per_block_mapped),
                                     ctx.total_query_capacity * 6 * sizeof(float),
                                     cudaHostAllocMapped),
                       "cudaHostAlloc h_time_per_block_mapped");
            check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ctx.d_time_per_block),
                                                ctx.h_time_per_block_mapped,
                                                0),
                       "cudaHostGetDevicePointer d_time_per_block");
        } else {
            check_cuda(cudaMalloc(&ctx.d_result,
                                  sizeof(idx_t) * ctx.total_query_capacity * ctx.result_width),
                       "cudaMalloc d_result");
            check_cuda(cudaMalloc(&ctx.d_result_distances,
                                  sizeof(dist_t) * ctx.total_query_capacity * ctx.result_width),
                       "cudaMalloc d_result_distances");
            check_cuda(cudaMalloc(&ctx.d_time_per_block,
                                  ctx.total_query_capacity * 6 * sizeof(float)),
                       "cudaMalloc d_time_per_block");
        }
        if (ctx.phase_snapshot_profile) {
            check_cuda(cudaMalloc(&ctx.d_phase_state,
                                  ctx.total_query_capacity * sizeof(uint16_t)),
                       "cudaMalloc RabitQ phase snapshot state");
        }
        check_cuda(cudaMalloc(&ctx.d_datadist_list,
                              ctx.graph_datadist_scratch_bytes),
                   "cudaMalloc d_datadist_list");
        check_cuda(cudaMalloc(&ctx.d_hashdata,
                              ctx.graph_hash_scratch_bytes),
                   "cudaMalloc d_hashdata");
        if (ctx.mailbox_rtt_profile) {
            check_cuda(cudaMalloc(
                           &ctx.d_mailbox_rtt_hist,
                           ctx.total_query_capacity * kRabitqMailboxRttRecordWords *
                               sizeof(uint64_t)),
                       "cudaMalloc RabitQ mailbox RTT histogram");
        }
        if (ctx.finite_query_state_enabled) {
            check_cuda(cudaMalloc(&ctx.d_finite_contexts,
                                  ctx.total_query_capacity * sizeof(RabitqFiniteContextControl)),
                       "cudaMalloc RabitQ finite contexts");
            check_cuda(cudaMalloc(&ctx.d_finite_frontier_state,
                                  ctx.graph_scratch_capacity *
                                      (static_cast<size_t>(ctx.finite_queue_capacity) + 2) *
                                      sizeof(RabitqPoolPair)),
                       "cudaMalloc RabitQ finite frontier state");
            check_cuda(cudaMalloc(&ctx.d_finite_stop_state,
                                  ctx.graph_scratch_capacity * (kTopk + 1) *
                                      sizeof(RabitqStopPair)),
                       "cudaMalloc RabitQ finite stop state");
            check_cuda(cudaMalloc(&ctx.d_finite_result_state,
                                  ctx.graph_scratch_capacity *
                                      (static_cast<size_t>(ctx.finite_result_capacity) + 1) *
                                      sizeof(RabitqResultPair)),
                       "cudaMalloc RabitQ finite result state");
            check_cuda(cudaMalloc(&ctx.d_finite_ready_tokens,
                                  std::max<size_t>(1, ctx.finite_ready_token_capacity) *
                                      sizeof(RabitqFiniteReadyToken)),
                       "cudaMalloc RabitQ finite ready tokens");
        }
        check_cuda(cudaMalloc(&ctx.d_query_ids,
                              sizeof(uint32_t) * ctx.total_query_capacity),
                   "cudaMalloc d_query_ids");
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ctx.status_memory),
                                 ctx.total_query_capacity * sizeof(int64_t) * 16,
                                 cudaHostAllocMapped),
                   "cudaHostAlloc status_memory");
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ctx.status_flag),
                                 ctx.total_query_capacity * sizeof(int64_t) * 16,
                                 cudaHostAllocMapped),
                   "cudaHostAlloc status_flag");
        check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ctx.d_status_memory),
                                            ctx.status_memory,
                                            0),
                   "cudaHostGetDevicePointer d_status_memory");
        check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ctx.d_status_flag),
                                            ctx.status_flag,
                                            0),
                   "cudaHostGetDevicePointer d_status_flag");
        if (ctx.request_ring_enabled) {
            const size_t ring_slot_count = checked_size_product(
                ctx.request_ring_capacity_count,
                ctx.request_ring_capacity_per_ring,
                "RabitQ request ring slots");
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ctx.h_request_ring_entries),
                                     ring_slot_count * sizeof(HostRequestRingEntry),
                                     cudaHostAllocMapped),
                       "cudaHostAlloc RabitQ request ring entries");
            check_cuda(cudaHostGetDevicePointer(
                           reinterpret_cast<void**>(&ctx.d_request_ring_entries),
                           ctx.h_request_ring_entries,
                           0),
                       "cudaHostGetDevicePointer RabitQ request ring entries");
            if (ctx.request_ring_consumed_device_mapped) {
                ctx.mapped_request_ring_consumed_sequences_bytes = round_up_gdr_bytes(
                    ring_slot_count * sizeof(uint64_t));
                check_cuda(cudaMalloc(
                               reinterpret_cast<void**>(
                                   &ctx.d_request_ring_consumed_sequences_raw),
                               ctx.mapped_request_ring_consumed_sequences_bytes +
                                   kGdrAlignBytes),
                           "cudaMalloc RabitQ request ring consumed sequences");
                ctx.d_request_ring_consumed_sequences = align_gdr_pointer(
                    ctx.d_request_ring_consumed_sequences_raw);
                check_gdr(gdr_pin_buffer(
                              ctx.g,
                              reinterpret_cast<unsigned long>(
                                  ctx.d_request_ring_consumed_sequences),
                              ctx.mapped_request_ring_consumed_sequences_bytes,
                              0,
                              0,
                              &ctx.mh_request_ring_consumed_sequences),
                          "gdr_pin_buffer RabitQ request ring consumed sequences");
                ctx.mh_request_ring_consumed_sequences_pinned = true;
                check_gdr(gdr_map(
                              ctx.g,
                              ctx.mh_request_ring_consumed_sequences,
                              &ctx.cpu_ptr_request_ring_consumed_sequences,
                              ctx.mapped_request_ring_consumed_sequences_bytes),
                          "gdr_map RabitQ request ring consumed sequences");
                ctx.h_request_ring_consumed_sequences = reinterpret_cast<uint64_t*>(
                    ctx.cpu_ptr_request_ring_consumed_sequences);
            } else {
                check_cuda(cudaHostAlloc(
                               reinterpret_cast<void**>(
                                   &ctx.h_request_ring_consumed_sequences),
                               ring_slot_count * sizeof(uint64_t),
                               cudaHostAllocMapped),
                           "cudaHostAlloc RabitQ request ring consumed sequences");
                check_cuda(cudaHostGetDevicePointer(
                               reinterpret_cast<void**>(
                                   &ctx.d_request_ring_consumed_sequences),
                               ctx.h_request_ring_consumed_sequences,
                               0),
                           "cudaHostGetDevicePointer RabitQ request ring consumed sequences");
            }
            check_cuda(cudaMalloc(&ctx.d_request_ring_tails,
                                  ctx.request_ring_capacity_count *
                                      sizeof(unsigned long long)),
                       "cudaMalloc RabitQ request ring tails");
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
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ctx.h_neighbor_memory_host),
                                     ctx.mapped_neighbor_bytes,
                                     cudaHostAllocMapped),
                       "cudaHostAlloc h_neighbor_memory_host");
            std::memset(ctx.h_neighbor_memory_host, 0, ctx.mapped_neighbor_bytes);
            check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ctx.d_neighbor_memory),
                                                ctx.h_neighbor_memory_host,
                                                0),
                       "cudaHostGetDevicePointer d_neighbor_memory");
            ctx.h_neighbor_memory = ctx.h_neighbor_memory_host;
            ctx.using_host_mapped_neighbor = true;
        }

        const size_t prefetch_count = runtime.prefetch_nodes.size();
        const size_t prefetch_alloc = std::max<size_t>(1, prefetch_count);
        const size_t label_alloc = std::max<size_t>(1, runtime.metadata.label_count);
        const size_t row_alloc = std::max<size_t>(1, runtime.metadata.row_to_sample.size());
        check_cuda(cudaMalloc(&ctx.d_prefetch_node_array,
                              prefetch_alloc * sizeof(Node)),
                   "cudaMalloc d_prefetch_node_array");
        check_cuda(cudaMalloc(&ctx.d_prefetch_index,
                              prefetch_alloc * sizeof(uint32_t)),
                   "cudaMalloc d_prefetch_index");
        check_cuda(cudaMalloc(&ctx.d_prefetch_begin_by_group,
                              label_alloc * sizeof(uint32_t)),
                   "cudaMalloc d_prefetch_begin_by_group");
        check_cuda(cudaMalloc(&ctx.d_prefetch_end_by_group,
                              label_alloc * sizeof(uint32_t)),
                   "cudaMalloc d_prefetch_end_by_group");
        check_cuda(cudaMalloc(&ctx.d_query_group_ids,
                              ctx.total_query_capacity * sizeof(uint32_t)),
                   "cudaMalloc d_query_group_ids");
        check_cuda(cudaMalloc(&ctx.d_start_local_ids,
                              ctx.total_query_capacity * sizeof(uint32_t)),
                   "cudaMalloc d_start_local_ids");
        check_cuda(cudaMalloc(&ctx.d_row_to_sample,
                              row_alloc * sizeof(uint32_t)),
                   "cudaMalloc d_row_to_sample");
        check_cuda(cudaMalloc(&ctx.d_label_offset,
                              label_alloc * sizeof(uint32_t)),
                   "cudaMalloc d_label_offset");
        if (!runtime.metadata.implicit_all_ones_row_filter_mask) {
            const size_t row_filter_mask_bytes = row_alloc * sizeof(uint64_t);
            cudaError_t row_filter_status =
                cudaMalloc(&ctx.d_row_filter_mask, row_filter_mask_bytes);
            if (row_filter_status == cudaErrorMemoryAllocation) {
                cudaGetLastError();
                check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ctx.h_row_filter_mask_mapped),
                                         row_filter_mask_bytes,
                                         cudaHostAllocMapped),
                           "cudaHostAlloc h_row_filter_mask_mapped");
                ctx.row_filter_mask_host_mapped = true;
                check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ctx.d_row_filter_mask),
                                                    ctx.h_row_filter_mask_mapped,
                                                    0),
                           "cudaHostGetDevicePointer d_row_filter_mask");
            } else {
                check_cuda(row_filter_status, "cudaMalloc d_row_filter_mask");
            }
        }
        check_cuda(cudaMalloc(&ctx.d_query_filter_bits,
                              ctx.total_query_capacity * sizeof(uint64_t)),
                   "cudaMalloc d_query_filter_bits");
        if (ctx.device_resident_graph) {
            check_cuda(cudaMalloc(&ctx.d_resident_graph,
                                  runtime.graph.data.size() * sizeof(uint32_t)),
                       "cudaMalloc resident RabitQ graph");
            check_cuda(cudaMemcpy(ctx.d_resident_graph,
                                  runtime.graph.data.data(),
                                  runtime.graph.data.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy resident RabitQ graph");
        }
        if (ctx.gpu_graph_row_cache_enabled || ctx.device_resident_graph) {
            if (ctx.gpu_graph_row_cache_enabled && ctx.owns_gpu_graph_row_cache) {
                check_cuda(cudaMalloc(
                               &ctx.d_gpu_graph_row_cache,
                               static_cast<size_t>(ctx.gpu_graph_row_cache_capacity) *
                                   sizeof(RabitqGpuGraphRowCacheEntry)),
                           "cudaMalloc RabitQ GPU graph row cache");
                check_cuda(cudaMemset(
                               ctx.d_gpu_graph_row_cache,
                               0,
                               static_cast<size_t>(ctx.gpu_graph_row_cache_capacity) *
                                   sizeof(RabitqGpuGraphRowCacheEntry)),
                           "cudaMemset RabitQ GPU graph row cache");
            }
            if (!gpu_query_label_csr.indices.empty()) {
                check_cuda(cudaMalloc(&ctx.d_gpu_query_label_indptr,
                                      gpu_query_label_csr.indptr.size() * sizeof(uint32_t)),
                           "cudaMalloc RabitQ GPU query label indptr");
                check_cuda(cudaMemcpy(ctx.d_gpu_query_label_indptr,
                                      gpu_query_label_csr.indptr.data(),
                                      gpu_query_label_csr.indptr.size() * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy RabitQ GPU query label indptr");
                check_cuda(cudaMalloc(&ctx.d_gpu_query_label_indices,
                                      gpu_query_label_csr.indices.size() * sizeof(uint32_t)),
                           "cudaMalloc RabitQ GPU query label indices");
                check_cuda(cudaMemcpy(ctx.d_gpu_query_label_indices,
                                      gpu_query_label_csr.indices.data(),
                                      gpu_query_label_csr.indices.size() * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy RabitQ GPU query label indices");
                check_cuda(cudaMalloc(&ctx.d_gpu_sample_label_indptr,
                                      gpu_sample_label_csr.indptr.size() * sizeof(uint32_t)),
                           "cudaMalloc RabitQ GPU sample label indptr");
                check_cuda(cudaMemcpy(ctx.d_gpu_sample_label_indptr,
                                      gpu_sample_label_csr.indptr.data(),
                                      gpu_sample_label_csr.indptr.size() * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy RabitQ GPU sample label indptr");
                check_cuda(cudaMalloc(&ctx.d_gpu_sample_label_indices,
                                      gpu_sample_label_csr.indices.size() * sizeof(uint32_t)),
                           "cudaMalloc RabitQ GPU sample label indices");
                check_cuda(cudaMemcpy(ctx.d_gpu_sample_label_indices,
                                      gpu_sample_label_csr.indices.data(),
                                      gpu_sample_label_csr.indices.size() * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy RabitQ GPU sample label indices");
            }
            const RabitqGpuGraphCacheDeviceView host_cache_view{
                ctx.d_gpu_graph_row_cache,
                ctx.d_resident_graph,
                ctx.d_gpu_query_label_indptr,
                ctx.d_gpu_query_label_indices,
                ctx.d_gpu_sample_label_indptr,
                ctx.d_gpu_sample_label_indices,
                ctx.gpu_graph_row_cache_capacity,
                ctx.resident_graph_degree,
            };
            check_cuda(cudaMalloc(&ctx.d_gpu_graph_cache_view,
                                  sizeof(RabitqGpuGraphCacheDeviceView)),
                       "cudaMalloc RabitQ GPU graph cache view");
            check_cuda(cudaMemcpy(ctx.d_gpu_graph_cache_view,
                                  &host_cache_view,
                                  sizeof(host_cache_view),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy RabitQ GPU graph cache view");
        }
        const size_t query_range_elems =
            ctx.total_query_capacity * static_cast<size_t>(device_range_count);
        if (has_active_range_filters) {
            check_cuda(cudaMalloc(&ctx.d_query_range_modes,
                                  query_range_elems * sizeof(uint8_t)),
                       "cudaMalloc d_query_range_modes");
            check_cuda(cudaMalloc(&ctx.d_query_range_lower_bounds,
                                  query_range_elems * sizeof(float)),
                       "cudaMalloc d_query_range_lower_bounds");
            check_cuda(cudaMalloc(&ctx.d_query_range_upper_bounds,
                                  query_range_elems * sizeof(float)),
                       "cudaMalloc d_query_range_upper_bounds");
        }
        ctx.base_range_value_count = runtime.base_range_values.size();
        if (has_active_range_filters && !runtime.base_range_values.empty()) {
            if (shared_base_range_ctx != nullptr) {
                if (shared_base_range_ctx->d_base_range_values == nullptr ||
                    shared_base_range_ctx->base_range_value_count !=
                        runtime.base_range_values.size() ||
                    shared_base_range_ctx->device_range_count != device_range_count) {
                    fail_rabitq("shared graph context has incompatible base range values");
                }
                ctx.d_base_range_values = shared_base_range_ctx->d_base_range_values;
                ctx.owns_base_range_values = false;
            } else {
                check_cuda(cudaMalloc(&ctx.d_base_range_values,
                                      runtime.base_range_values.size() * sizeof(float)),
                           "cudaMalloc d_base_range_values");
            }
        }

        if (prefetch_count > 0) {
            check_cuda(cudaMemcpy(ctx.d_prefetch_node_array,
                                  runtime.prefetch_nodes.data(),
                                  prefetch_count * sizeof(Node),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy d_prefetch_node_array");
            check_cuda(cudaMemcpy(ctx.d_prefetch_index,
                                  runtime.prefetch_index.data(),
                                  prefetch_count * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy d_prefetch_index");
        }
        if (runtime.metadata.label_count > 0) {
            check_cuda(cudaMemcpy(ctx.d_prefetch_begin_by_group,
                                  runtime.prefetch_begin_by_group.data(),
                                  runtime.metadata.label_count * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy d_prefetch_begin_by_group");
            check_cuda(cudaMemcpy(ctx.d_prefetch_end_by_group,
                                  runtime.prefetch_end_by_group.data(),
                                  runtime.metadata.label_count * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy d_prefetch_end_by_group");
            check_cuda(cudaMemcpy(ctx.d_label_offset,
                                  runtime.metadata.label_offset.data(),
                                  runtime.metadata.label_count * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy d_label_offset");
        }
        check_cuda(cudaMemcpy(ctx.d_query_group_ids,
                              runtime.query_group_ids.data(),
                              ctx.total_query_capacity * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_query_group_ids");
        check_cuda(cudaMemcpy(ctx.d_start_local_ids,
                              runtime.start_local_ids.data(),
                              ctx.total_query_capacity * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_start_local_ids");
        if (!runtime.metadata.row_to_sample.empty()) {
            check_cuda(cudaMemcpy(ctx.d_row_to_sample,
                                  runtime.metadata.row_to_sample.data(),
                                  runtime.metadata.row_to_sample.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy d_row_to_sample");
        }
        if (!runtime.metadata.implicit_all_ones_row_filter_mask &&
            !runtime.metadata.row_filter_mask.empty()) {
            const size_t row_filter_mask_payload_bytes =
                runtime.metadata.row_filter_mask.size() * sizeof(uint64_t);
            if (ctx.row_filter_mask_host_mapped) {
                std::memcpy(ctx.h_row_filter_mask_mapped,
                            runtime.metadata.row_filter_mask.data(),
                            row_filter_mask_payload_bytes);
            } else {
                check_cuda(cudaMemcpy(ctx.d_row_filter_mask,
                                      runtime.metadata.row_filter_mask.data(),
                                      row_filter_mask_payload_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy d_row_filter_mask");
            }
        }
        check_cuda(cudaMemcpy(ctx.d_query_filter_bits,
                              runtime.query_filter_bits.data(),
                              ctx.total_query_capacity * sizeof(uint64_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_query_filter_bits");
        if (has_active_range_filters) {
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
        }
        if (ctx.owns_base_range_values && ctx.d_base_range_values != nullptr) {
            check_cuda(cudaMemcpy(ctx.d_base_range_values,
                                  runtime.base_range_values.data(),
                                  runtime.base_range_values.size() * sizeof(float),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy d_base_range_values");
        }
        if (runtime.base_range_count != runtime.query_ranges.range_count) {
            fail_rabitq("query/base range dimension mismatch");
        }

        ctx.predicate_view.base_ranges = ctx.d_base_range_values;
        check_cuda(cudaEventCreate(&ctx.start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&ctx.stop), "cudaEventCreate stop");
        ctx.initialized = true;
        reset_graph_context(ctx);
    } catch (...) {
        cleanup_graph_context(ctx);
        throw;
    }
}

void reset_graph_context(RabitqGraphSearchContext& ctx)
{
    if (!ctx.initialized) {
        return;
    }
    std::fill(ctx.status_memory,
              ctx.status_memory + ctx.total_query_capacity * 16,
              static_cast<int64_t>(-2));
    std::fill(ctx.status_flag,
              ctx.status_flag + ctx.total_query_capacity * 16,
              static_cast<int64_t>(-2));
    std::fill(ctx.h_gdr_data,
              ctx.h_gdr_data + ctx.total_query_capacity * kGdrMailboxStride,
              static_cast<int64_t>(0));
    if (ctx.request_ring_enabled) {
        const size_t ring_slot_count = checked_size_product(
            ctx.request_ring_capacity_count,
            ctx.request_ring_capacity_per_ring,
            "RabitQ request ring reset slots");
        std::fill(ctx.h_request_ring_entries,
                  ctx.h_request_ring_entries + ring_slot_count,
                  HostRequestRingEntry{});
        std::fill(ctx.h_request_ring_consumed_sequences,
                  ctx.h_request_ring_consumed_sequences + ring_slot_count,
                  0ULL);
        _mm_sfence();
        check_cuda(cudaMemset(ctx.d_request_ring_tails,
                              0,
                              ctx.request_ring_capacity_count *
                                  sizeof(unsigned long long)),
                   "cudaMemset RabitQ request ring tails");
    }
    check_cuda(cudaMemset(ctx.d_result,
                          0xFF,
                          sizeof(idx_t) * ctx.total_query_capacity * ctx.result_width),
               "cudaMemset d_result");
    check_cuda(cudaMemset(ctx.d_result_distances,
                          0x7F,
                          sizeof(dist_t) * ctx.total_query_capacity * ctx.result_width),
               "cudaMemset d_result_distances");
    check_cuda(cudaMemset(ctx.d_time_per_block,
                          0,
                          ctx.total_query_capacity * 6 * sizeof(float)),
               "cudaMemset d_time_per_block");
    if (ctx.d_phase_state != nullptr) {
        check_cuda(cudaMemset(ctx.d_phase_state,
                              0,
                              ctx.total_query_capacity * sizeof(uint16_t)),
                   "cudaMemset RabitQ phase snapshot state");
    }
    if (ctx.d_mailbox_rtt_hist != nullptr) {
        check_cuda(cudaMemset(ctx.d_mailbox_rtt_hist,
                              0,
                              ctx.total_query_capacity *
                                  kRabitqMailboxRttRecordWords * sizeof(uint64_t)),
                   "cudaMemset RabitQ mailbox RTT histogram");
    }
    check_cuda(cudaMemset(ctx.d_datadist_list,
                          0,
                          ctx.graph_datadist_scratch_bytes),
               "cudaMemset d_datadist_list");
    check_cuda(cudaMemset(ctx.d_hashdata,
                          0xFF,
                          ctx.graph_hash_scratch_bytes),
               "cudaMemset d_hashdata");
    if (ctx.finite_query_state_enabled) {
        check_cuda(cudaMemset(ctx.d_finite_contexts,
                              0,
                              ctx.total_query_capacity * sizeof(RabitqFiniteContextControl)),
                   "cudaMemset RabitQ finite contexts");
        check_cuda(cudaMemset(ctx.d_finite_frontier_state,
                              0,
                              ctx.graph_scratch_capacity *
                                  (static_cast<size_t>(ctx.finite_queue_capacity) + 2) *
                                  sizeof(RabitqPoolPair)),
                   "cudaMemset RabitQ finite frontier state");
        check_cuda(cudaMemset(ctx.d_finite_stop_state,
                              0,
                              ctx.graph_scratch_capacity * (kTopk + 1) * sizeof(RabitqStopPair)),
                   "cudaMemset RabitQ finite stop state");
        check_cuda(cudaMemset(ctx.d_finite_result_state,
                              0,
                              ctx.graph_scratch_capacity *
                                  (static_cast<size_t>(ctx.finite_result_capacity) + 1) *
                                  sizeof(RabitqResultPair)),
                   "cudaMemset RabitQ finite result state");
    }
    ctx.host_filter_stats = HostFilterStats{};
}

std::vector<HighfreqOriginalRerankGroup> build_float_gpu_rerank_groups(
    const RabitqSearchOptions& options,
    const GraphRuntime& runtime,
    const std::vector<uint32_t>& query_indices)
{
    if (options.highfreq_rerank_merge_routes) {
        return build_highfreq_original_rerank_groups(runtime, query_indices);
    }
    std::vector<HighfreqOriginalRerankGroup> groups;
    groups.reserve(query_indices.size());
    for (size_t pos = 0; pos < query_indices.size(); ++pos) {
        HighfreqOriginalRerankGroup group;
        group.original_query = highfreq_route_original_query_id(runtime, query_indices[pos]);
        group.route_positions.push_back(pos);
        groups.push_back(std::move(group));
    }
    return groups;
}

std::vector<uint32_t> build_float_gpu_rerank_candidates(
    const RabitqSearchOptions& options,
    const GraphRuntime& runtime,
    const std::vector<uint32_t>& query_indices,
    const std::vector<std::vector<idx_t>>& gpu_shortlist,
    const std::vector<std::vector<float>>* gpu_shortlist_distances,
    const HighfreqOriginalRerankGroup& group,
    uint32_t shortlist_k)
{
    std::vector<HighfreqMergedCandidate> candidates;
    std::unordered_map<uint32_t, size_t> candidate_by_sample;
    size_t candidate_capacity = 0;
    for (size_t pos : group.route_positions) {
        if (pos < gpu_shortlist.size()) {
            candidate_capacity += std::min<size_t>(gpu_shortlist[pos].size(), shortlist_k);
        }
    }
    candidates.reserve(candidate_capacity);
    candidate_by_sample.reserve(candidate_capacity);

    for (size_t pos : group.route_positions) {
        if (pos >= gpu_shortlist.size() || pos >= query_indices.size()) {
            continue;
        }
        const uint32_t route_idx = query_indices[pos];
        const uint32_t effective_shortlist = static_cast<uint32_t>(
            std::min<size_t>(gpu_shortlist[pos].size(), shortlist_k));
        for (uint32_t rank = 0; rank < effective_shortlist; ++rank) {
            const idx_t sample_id = gpu_shortlist[pos][rank];
            if (sample_id == UINT32_MAX || sample_id >= runtime.base_vectors.rows) {
                continue;
            }
            const uint32_t sample_id_u32 = static_cast<uint32_t>(sample_id);
            if (options.full_filter &&
                !sample_matches_lowfreq_full_labels(runtime, route_idx, sample_id_u32)) {
                continue;
            }
            float score = FLT_MAX;
            if (gpu_shortlist_distances != nullptr &&
                pos < gpu_shortlist_distances->size() &&
                rank < (*gpu_shortlist_distances)[pos].size() &&
                std::isfinite((*gpu_shortlist_distances)[pos][rank])) {
                score = (*gpu_shortlist_distances)[pos][rank];
            }
            auto [it, inserted] = candidate_by_sample.emplace(
                sample_id_u32, candidates.size());
            if (inserted) {
                candidates.push_back({sample_id_u32, rank, score});
            } else {
                HighfreqMergedCandidate& candidate = candidates[it->second];
                candidate.best_rank = std::min(candidate.best_rank, rank);
                candidate.best_score = std::min(candidate.best_score, score);
            }
        }
    }
    std::sort(candidates.begin(),
              candidates.end(),
              [](const HighfreqMergedCandidate& lhs, const HighfreqMergedCandidate& rhs) {
                  if (lhs.best_score != rhs.best_score) {
                      return lhs.best_score < rhs.best_score;
                  }
                  if (lhs.best_rank != rhs.best_rank) {
                      return lhs.best_rank < rhs.best_rank;
                  }
                  return lhs.sample_id < rhs.sample_id;
              });
    if (candidates.size() > shortlist_k) {
        candidates.resize(shortlist_k);
    }
    std::vector<uint32_t> candidate_ids;
    candidate_ids.reserve(candidates.size());
    for (const HighfreqMergedCandidate& candidate : candidates) {
        candidate_ids.push_back(candidate.sample_id);
    }
    return candidate_ids;
}

void highfreq_rabitq_float_gpu_rerank(
    const RabitqSearchOptions& options,
    const GraphRuntime& runtime,
    const RabitqPreparedQueryState& prepared_state,
    RabitqFloatGpuRerankContext& ctx,
    const std::vector<uint32_t>& query_indices,
    const std::vector<std::vector<idx_t>>& gpu_shortlist,
    const std::vector<std::vector<float>>* gpu_shortlist_distances,
    std::vector<std::vector<idx_t>>& out_results,
    std::vector<std::vector<float>>* out_distances)
{
    if (!ctx.initialized || prepared_state.d_query_float == nullptr) {
        fail_rabitq("staged float GPU rerank context/query matrix is not initialized");
    }
    if (ctx.dim != runtime.base_vectors.cols ||
        ctx.shortlist_capacity != effective_highfreq_rerank_shortlist(options) ||
        ctx.topk_capacity != static_cast<uint32_t>(options.topk)) {
        fail_rabitq("staged float GPU rerank context is incompatible with the current query options");
    }

    const uint32_t topk = static_cast<uint32_t>(options.topk);
    const uint32_t shortlist_k = ctx.shortlist_capacity;
    out_results.assign(query_indices.size(), std::vector<idx_t>(topk, UINT32_MAX));
    if (out_distances != nullptr) {
        out_distances->assign(query_indices.size(), std::vector<float>(topk, FLT_MAX));
    }
    if (query_indices.empty()) {
        return;
    }

    const std::vector<HighfreqOriginalRerankGroup> groups =
        build_float_gpu_rerank_groups(options, runtime, query_indices);
    std::vector<std::vector<uint32_t>> group_candidates;
    group_candidates.reserve(groups.size());
    for (const HighfreqOriginalRerankGroup& group : groups) {
        if (group.original_query >= prepared_state.original_query_count) {
            fail_rabitq("staged float GPU rerank original query id is out of range");
        }
        group_candidates.push_back(build_float_gpu_rerank_candidates(
            options,
            runtime,
            query_indices,
            gpu_shortlist,
            gpu_shortlist_distances,
            group,
            shortlist_k));
    }

    for (size_t batch_begin = 0; batch_begin < groups.size();
         batch_begin += ctx.batch_capacity) {
        const uint32_t batch_size = static_cast<uint32_t>(
            std::min<size_t>(ctx.batch_capacity, groups.size() - batch_begin));
        const size_t candidate_slots = checked_size_product(
            batch_size, shortlist_k, "float GPU rerank active candidate slots");
        const size_t candidate_values = checked_size_product(
            candidate_slots, ctx.dim, "float GPU rerank active candidate values");
        const size_t result_slots = checked_size_product(
            batch_size, topk, "float GPU rerank active result slots");
        std::fill(ctx.h_candidate_ids,
                  ctx.h_candidate_ids + candidate_slots,
                  UINT32_MAX);

        for (uint32_t batch_slot = 0; batch_slot < batch_size; ++batch_slot) {
            const size_t group_idx = batch_begin + batch_slot;
            const std::vector<uint32_t>& candidate_ids = group_candidates[group_idx];
            ctx.h_candidate_counts[batch_slot] = static_cast<uint32_t>(candidate_ids.size());
            ctx.h_original_query_ids[batch_slot] = groups[group_idx].original_query;
            const size_t slot_base = static_cast<size_t>(batch_slot) * shortlist_k;
            for (uint32_t candidate_slot = 0;
                 candidate_slot < candidate_ids.size(); ++candidate_slot) {
                const uint32_t sample_id = candidate_ids[candidate_slot];
                ctx.h_candidate_ids[slot_base + candidate_slot] = sample_id;
                // Keep gathering/conversion in the timed rerank phase. This
                // typed reader also preserves the existing resident FP32 path.
                copy_base_float_vector_row(runtime, sample_id,
                    ctx.h_candidate_vectors +
                        (slot_base + candidate_slot) * static_cast<size_t>(ctx.dim),
                    ctx.dim);
            }
        }

        check_cuda(cudaMemcpyAsync(ctx.d_candidate_vectors,
                                   ctx.h_candidate_vectors,
                                   candidate_values * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   ctx.stream),
                   "cudaMemcpyAsync float GPU rerank candidate vectors");
        check_cuda(cudaMemcpyAsync(ctx.d_candidate_ids,
                                   ctx.h_candidate_ids,
                                   candidate_slots * sizeof(idx_t),
                                   cudaMemcpyHostToDevice,
                                   ctx.stream),
                   "cudaMemcpyAsync float GPU rerank candidate ids");
        check_cuda(cudaMemcpyAsync(ctx.d_candidate_counts,
                                   ctx.h_candidate_counts,
                                   batch_size * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice,
                                   ctx.stream),
                   "cudaMemcpyAsync float GPU rerank candidate counts");
        check_cuda(cudaMemcpyAsync(ctx.d_original_query_ids,
                                   ctx.h_original_query_ids,
                                   batch_size * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice,
                                   ctx.stream),
                   "cudaMemcpyAsync float GPU rerank original query ids");
        highfreq_float_gpu_l2_kernel<<<static_cast<uint32_t>(candidate_slots),
                                       256,
                                       0,
                                       ctx.stream>>>(ctx.d_candidate_vectors,
                                                    ctx.d_candidate_ids,
                                                    ctx.d_candidate_counts,
                                                    ctx.d_original_query_ids,
                                                    prepared_state.d_query_float,
                                                    ctx.d_candidate_distances,
                                                    batch_size,
                                                    shortlist_k,
                                                    ctx.dim);
        check_cuda(cudaGetLastError(), "float GPU rerank L2 kernel launch");
        highfreq_float_gpu_topk_kernel<<<batch_size, 1, 0, ctx.stream>>>(
            ctx.d_candidate_ids,
            ctx.d_candidate_distances,
            ctx.d_candidate_counts,
            ctx.d_result_ids,
            ctx.d_result_distances,
            batch_size,
            shortlist_k,
            topk);
        check_cuda(cudaGetLastError(), "float GPU rerank top-k kernel launch");
        check_cuda(cudaMemcpyAsync(ctx.h_result_ids,
                                   ctx.d_result_ids,
                                   result_slots * sizeof(idx_t),
                                   cudaMemcpyDeviceToHost,
                                   ctx.stream),
                   "cudaMemcpyAsync float GPU rerank result ids");
        check_cuda(cudaMemcpyAsync(ctx.h_result_distances,
                                   ctx.d_result_distances,
                                   result_slots * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   ctx.stream),
                   "cudaMemcpyAsync float GPU rerank result distances");
        check_cuda(cudaStreamSynchronize(ctx.stream),
                   "cudaStreamSynchronize float GPU rerank");

        for (uint32_t batch_slot = 0; batch_slot < batch_size; ++batch_slot) {
            const size_t group_idx = batch_begin + batch_slot;
            const size_t result_base = static_cast<size_t>(batch_slot) * topk;
            for (size_t route_pos : groups[group_idx].route_positions) {
                for (uint32_t rank = 0; rank < topk; ++rank) {
                    out_results[route_pos][rank] = ctx.h_result_ids[result_base + rank];
                    if (out_distances != nullptr) {
                        (*out_distances)[route_pos][rank] =
                            ctx.h_result_distances[result_base + rank];
                    }
                }
            }
        }
    }
}

void clear_rabitq_gpu_graph_row_cache(RabitqGraphSearchContext& ctx)
{
    if (!ctx.gpu_graph_row_cache_enabled || ctx.d_gpu_graph_row_cache == nullptr ||
        ctx.gpu_graph_row_cache_capacity == 0) {
        return;
    }
    check_cuda(cudaMemset(ctx.d_gpu_graph_row_cache,
                          0,
                          static_cast<size_t>(ctx.gpu_graph_row_cache_capacity) *
                              sizeof(RabitqGpuGraphRowCacheEntry)),
               "cudaMemset RabitQ GPU graph row cache clear");
}

void print_rabitq_mailbox_rtt_summary(const RabitqGraphSearchContext& ctx,
                                      const std::vector<uint32_t>& query_indices,
                                      const std::string& wait_policy)
{
    if (!ctx.mailbox_rtt_profile || ctx.d_mailbox_rtt_hist == nullptr) return;

    std::vector<uint64_t> records(
        ctx.total_query_capacity * kRabitqMailboxRttRecordWords, 0);
    check_cuda(cudaMemcpy(records.data(),
                          ctx.d_mailbox_rtt_hist,
                          records.size() * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy RabitQ mailbox RTT histogram");

    uint64_t bins[kRabitqMailboxRttBinCount] = {};
    uint64_t total_ns = 0;
    for (uint32_t query_idx : query_indices) {
        const size_t record_base =
            static_cast<size_t>(query_idx) * kRabitqMailboxRttRecordWords;
        for (size_t bin = 0; bin < kRabitqMailboxRttBinCount; ++bin) {
            bins[bin] += records[record_base + bin];
        }
        total_ns += records[record_base + kRabitqMailboxRttSumWord];
    }
    const uint64_t request_count =
        std::accumulate(bins, bins + kRabitqMailboxRttBinCount, uint64_t{0});
    const double avg_us = request_count == 0
        ? 0.0
        : static_cast<double>(total_ns) / static_cast<double>(request_count) / 1000.0;
    const int initial_sleep_us = use_adaptive_wait_policy(wait_policy)
        ? kInitialWaitSleepUs
        : kFixedWaitSleepUs;
    std::printf("[rabitq mailbox RTT] clock=gpu scope=publish-to-observe "
                "wait_policy=%s initial_sleep_us=%d requests=%llu avg_us=%.3f",
                wait_policy.c_str(),
                initial_sleep_us,
                static_cast<unsigned long long>(request_count),
                avg_us);
    uint64_t cumulative = 0;
    for (size_t bin = 0; bin + 1 < kRabitqMailboxRttBinCount; ++bin) {
        cumulative += bins[bin];
        const double fraction = request_count == 0
            ? 0.0
            : 100.0 * static_cast<double>(cumulative) /
                  static_cast<double>(request_count);
        std::printf(" le_%lluus=%llu(%.2f%%)",
                    static_cast<unsigned long long>(kRabitqMailboxRttUpperUs[bin]),
                    static_cast<unsigned long long>(cumulative),
                    fraction);
    }
    const double overflow_fraction = request_count == 0
        ? 0.0
        : 100.0 * static_cast<double>(bins[kRabitqMailboxRttBinCount - 1]) /
              static_cast<double>(request_count);
    std::printf(" gt_1000us=%llu(%.2f%%)\n",
                static_cast<unsigned long long>(bins[kRabitqMailboxRttBinCount - 1]),
                overflow_fraction);
}

void print_rabitq_gpu_graph_cache_summary(const RabitqGraphSearchContext& ctx,
                                          const std::vector<uint32_t>& query_indices)
{
    if (!ctx.gpu_graph_row_cache_enabled) return;
    std::vector<float> copied_timing;
    const float* timing = ctx.h_time_per_block_mapped;
    if (timing == nullptr) {
        copied_timing.resize(ctx.total_query_capacity * 6, 0.0f);
        check_cuda(cudaMemcpy(copied_timing.data(),
                              ctx.d_time_per_block,
                              copied_timing.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy RabitQ GPU graph cache stats");
        timing = copied_timing.data();
    }
    uint64_t hits = 0;
    for (uint32_t query_idx : query_indices) {
        hits += static_cast<uint64_t>(std::max(
            0.0f, timing[static_cast<size_t>(query_idx) * 6 + 5]));
    }
    const uint64_t misses = ctx.host_filter_stats.request_ring_demand_events;
    const uint64_t lookups = hits + misses;
    const double hit_rate = lookups == 0
        ? 0.0
        : static_cast<double>(hits) / static_cast<double>(lookups);
    std::printf("[rabitq] GPU graph row cache hits=%llu misses=%llu "
                "hit_rate=%.6f capacity_rows=%u\n",
                static_cast<unsigned long long>(hits),
                static_cast<unsigned long long>(misses),
                hit_rate,
                ctx.gpu_graph_row_cache_capacity);
}

void print_rabitq_graph_phase_summary(const RabitqGraphSearchContext& ctx,
                                      const GraphRuntime& runtime,
                                      const std::vector<uint32_t>& query_indices,
                                      const char* run_label,
                                      int max_steps,
                                      double gpu_kernel_ms,
                                      double launch_to_gpu_complete_ms,
                                      double host_join_tail_ms,
                                      double streaming_rerank_tail_ms,
                                      double highfreq_end_to_end_ms)
{
    if (!graph_phase_profile_enabled()) return;
    if (!kNativeDetailedTiming) {
        std::printf("[rabitq graph profile] run=%s unavailable: "
                    "GO_FANN_RABITQ_NATIVE_DETAILED_TIMING=0\n",
                    run_label);
        return;
    }

    std::vector<float> copied_timing;
    const float* timing = ctx.h_time_per_block_mapped;
    if (timing == nullptr) {
        copied_timing.resize(ctx.total_query_capacity * 6, 0.0f);
        check_cuda(cudaMemcpy(copied_timing.data(),
                              ctx.d_time_per_block,
                              copied_timing.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy RabitQ graph phase timing");
        timing = copied_timing.data();
    }

    double route_us = 0.0;
    double io_wait_us = 0.0;
    double score_us = 0.0;
    double frontier_us = 0.0;
    uint64_t steps = 0;
    std::vector<uint32_t> route_steps;
    std::vector<std::pair<uint32_t, uint32_t>> route_step_records;
    route_steps.reserve(query_indices.size());
    route_step_records.reserve(query_indices.size());
    uint64_t capped_routes = 0;
    for (uint32_t query_idx : query_indices) {
        const size_t base = static_cast<size_t>(query_idx) * 6;
        route_us += std::max(0.0, static_cast<double>(timing[base]));
        io_wait_us += std::max(0.0, static_cast<double>(timing[base + 1]));
        score_us += std::max(0.0, static_cast<double>(timing[base + 2]));
        frontier_us += std::max(0.0, static_cast<double>(timing[base + 3]));
        const uint32_t route_step_count = static_cast<uint32_t>(
            std::max(0.0f, timing[base + 4]));
        steps += route_step_count;
        route_steps.push_back(route_step_count);
        route_step_records.emplace_back(route_step_count, query_idx);
        capped_routes += max_steps > 0 && route_step_count >= static_cast<uint32_t>(max_steps);
    }
    const double named_us = io_wait_us + score_us + frontier_us;
    const double other_us = std::max(0.0, route_us - named_us);
    const double route_count = static_cast<double>(query_indices.size());
    const double demand_count = static_cast<double>(
        ctx.host_filter_stats.request_ring_demand_events > 0
            ? ctx.host_filter_stats.request_ring_demand_events
            : ctx.host_filter_stats.requests_served);
    const auto fraction = [route_us](double phase_us) {
        return route_us > 0.0 ? 100.0 * phase_us / route_us : 0.0;
    };
    const double route_parallelism = gpu_kernel_ms > 0.0
        ? route_us / (gpu_kernel_ms * 1000.0)
        : 0.0;

    std::printf("[rabitq graph profile] run=%s routes=%zu steps=%llu "
                "gpu_kernel_ms=%.3f launch_to_gpu_complete_ms=%.3f "
                "host_join_tail_ms=%.3f streaming_rerank_tail_ms=%.3f "
                "highfreq_end_to_end_ms=%.3f summed_route_ms=%.3f "
                "route_parallelism=%.3f\n",
                run_label,
                query_indices.size(),
                static_cast<unsigned long long>(steps),
                gpu_kernel_ms,
                launch_to_gpu_complete_ms,
                host_join_tail_ms,
                streaming_rerank_tail_ms,
                highfreq_end_to_end_ms,
                route_us / 1000.0,
                route_parallelism);
    if (!route_steps.empty()) {
        std::sort(route_steps.begin(), route_steps.end());
        const auto percentile = [&route_steps](double fraction) {
            const size_t index = static_cast<size_t>(
                fraction * static_cast<double>(route_steps.size() - 1));
            return route_steps[index];
        };
        std::printf("[rabitq graph steps] run=%s min=%u p50=%u p90=%u p99=%u "
                    "max=%u max_steps=%d capped_routes=%llu\n",
                    run_label,
                    route_steps.front(),
                    percentile(0.50),
                    percentile(0.90),
                    percentile(0.99),
                    route_steps.back(),
                    max_steps,
                    static_cast<unsigned long long>(capped_routes));
        std::sort(route_step_records.begin(), route_step_records.end(),
                  std::greater<std::pair<uint32_t, uint32_t>>());
        const size_t worst_count = std::min<size_t>(5, route_step_records.size());
        for (size_t rank = 0; rank < worst_count; ++rank) {
            const uint32_t route_idx = route_step_records[rank].second;
            const uint32_t original_query = route_idx < runtime.route_to_original_query.size()
                ? runtime.route_to_original_query[route_idx]
                : UINT32_MAX;
            const uint32_t group_id = route_idx < runtime.query_group_ids.size()
                ? runtime.query_group_ids[route_idx]
                : UINT32_MAX;
            uint32_t active_ranges = 0;
            if (route_idx < runtime.query_ranges.rows) {
                const size_t base = static_cast<size_t>(route_idx) *
                                    runtime.query_ranges.range_count;
                for (uint32_t range_idx = 0;
                     range_idx < runtime.query_ranges.range_count;
                     ++range_idx) {
                    active_ranges += runtime.query_ranges.modes[base + range_idx] != 0;
                }
            }
            const int provider = route_idx < runtime.route_providers.size()
                ? static_cast<int>(runtime.route_providers[route_idx])
                : -1;
            std::printf("[rabitq graph worst-route] rank=%zu route=%u original_query=%u "
                        "group=%u steps=%u active_ranges=%u provider=%d\n",
                        rank + 1,
                        route_idx,
                        original_query,
                        group_id,
                        route_step_records[rank].first,
                        active_ranges,
                        provider);
        }
    }
    std::printf("[rabitq gpu phases] run=%s clock=gpu summed_concurrent_route_time=1 "
                "io_wait_ms=%.3f(%.2f%%) rabitq_score_ms=%.3f(%.2f%%) "
                "frontier_ms=%.3f(%.2f%%) other_ms=%.3f(%.2f%%) "
                "avg_route_us=%.3f avg_io_wait_per_demand_us=%.3f\n",
                run_label,
                io_wait_us / 1000.0,
                fraction(io_wait_us),
                score_us / 1000.0,
                fraction(score_us),
                frontier_us / 1000.0,
                fraction(frontier_us),
                other_us / 1000.0,
                fraction(other_us),
                route_count > 0.0 ? route_us / route_count : 0.0,
                demand_count > 0.0 ? io_wait_us / demand_count : 0.0);
}

void destroy_persistent_latency_queue(RabitqPersistentLatencyQueue& queue)
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
    if (queue.h_lowfreq_candidate_ids_mapped != nullptr) cudaFreeHost(queue.h_lowfreq_candidate_ids_mapped);
    if (queue.h_lowfreq_candidate_offsets_mapped != nullptr) cudaFreeHost(queue.h_lowfreq_candidate_offsets_mapped);
    if (queue.h_lowfreq_candidate_sizes_mapped != nullptr) cudaFreeHost(queue.h_lowfreq_candidate_sizes_mapped);
    if (queue.d_lowfreq_label_size != nullptr) cudaFree(queue.d_lowfreq_label_size);
    if (queue.d_lowfreq_label_offset != nullptr) cudaFree(queue.d_lowfreq_label_offset);
    if (queue.d_lowfreq_sample_ids != nullptr) cudaFree(queue.d_lowfreq_sample_ids);
    if (queue.d_lowfreq_row_filter_mask != nullptr) cudaFree(queue.d_lowfreq_row_filter_mask);
    queue = RabitqPersistentLatencyQueue{};
}

RabitqPersistentLatencyQueue create_persistent_latency_queue(uint32_t slot_count,
                                                             const RabitqSearchOptions& options,
                                                             const GraphRuntime& runtime,
                                                             size_t lowfreq_candidate_capacity)
{
    RabitqPersistentLatencyQueue queue;
    queue.slot_count = slot_count;
    queue.lowfreq_shortlist_k = options.enable_lowfreq_rerank
        ? static_cast<uint32_t>(options.lowfreq_rerank_shortlist)
        : static_cast<uint32_t>(options.topk);
    queue.lowfreq_candidate_mode = options.latency_lowfreq_mode == "candidate";
    queue.lowfreq_candidate_capacity = std::max<size_t>(1, lowfreq_candidate_capacity);
    queue.lowfreq_group_count = runtime.lowfreq_index.group_count;
    queue.lowfreq_total_rows = runtime.lowfreq_index.total_rows;
    queue.base_rows = runtime.base_vectors.rows;
    if (runtime.lowfreq_index.label_offset.size() != runtime.lowfreq_index.label_size.size()) {
        fail_rabitq("RabitQ persistent latency lowfreq index label offset/size length mismatch");
    }
    if (runtime.lowfreq_index.row_filter_mask.size() != runtime.lowfreq_index.sample_ids.size()) {
        fail_rabitq("RabitQ persistent latency lowfreq index sample/filter length mismatch");
    }
    queue.mapped_bytes =
        round_up_gdr_bytes(std::max<size_t>(1, slot_count) *
                           sizeof(RabitqPersistentLatencyRequestControl));
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&queue.d_requests_raw),
                              queue.mapped_bytes + kGdrAlignBytes),
                   "cudaMalloc RabitQ persistent latency requests");
        queue.d_requests = align_gdr_pointer(queue.d_requests_raw);
        queue.g = gdr_open();
        if (queue.g == nullptr) {
            fail_rabitq("gdr_open returned nullptr for RabitQ persistent latency requests");
        }
        check_gdr(gdr_pin_buffer(queue.g,
                                 reinterpret_cast<unsigned long>(queue.d_requests),
                                 queue.mapped_bytes,
                                 0,
                                 0,
                                 &queue.mh),
                  "gdr_pin_buffer RabitQ persistent latency requests");
        queue.pinned = true;
        check_gdr(gdr_map(queue.g, queue.mh, &queue.cpu_ptr, queue.mapped_bytes),
                  "gdr_map RabitQ persistent latency requests");
        queue.h_requests = reinterpret_cast<RabitqPersistentLatencyRequestControl*>(queue.cpu_ptr);
        for (uint32_t slot_idx = 0; slot_idx < slot_count; ++slot_idx) {
            queue.h_requests[slot_idx] = RabitqPersistentLatencyRequestControl{};
        }
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_completions),
                                 sizeof(RabitqPersistentLatencyCompletionControl) *
                                     std::max<size_t>(1, slot_count),
                                 cudaHostAllocMapped),
                   "cudaHostAlloc RabitQ persistent latency completions");
        for (uint32_t slot_idx = 0; slot_idx < slot_count; ++slot_idx) {
            queue.h_completions[slot_idx] = RabitqPersistentLatencyCompletionControl{};
        }
        check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue.d_completions),
                                            queue.h_completions,
                                            0),
                   "cudaHostGetDevicePointer RabitQ persistent latency completions");
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_lowfreq_result_mapped),
                                 sizeof(idx_t) * std::max<size_t>(1, slot_count) *
                                     std::max<uint32_t>(1, queue.lowfreq_shortlist_k),
                                 cudaHostAllocMapped),
                   "cudaHostAlloc RabitQ persistent lowfreq result");
        check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue.d_lowfreq_result),
                                            queue.h_lowfreq_result_mapped,
                                            0),
                   "cudaHostGetDevicePointer RabitQ persistent lowfreq result");
        check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_lowfreq_distances_mapped),
                                 sizeof(float) * std::max<size_t>(1, slot_count) *
                                     std::max<uint32_t>(1, queue.lowfreq_shortlist_k),
                                 cudaHostAllocMapped),
                   "cudaHostAlloc RabitQ persistent lowfreq distances");
        check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue.d_lowfreq_distances),
                                            queue.h_lowfreq_distances_mapped,
                                            0),
                   "cudaHostGetDevicePointer RabitQ persistent lowfreq distances");
        if (queue.lowfreq_candidate_mode) {
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_lowfreq_candidate_offsets_mapped),
                                     sizeof(uint32_t) * std::max<size_t>(1, slot_count),
                                     cudaHostAllocMapped),
                       "cudaHostAlloc RabitQ persistent lowfreq candidate offsets");
            check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue.d_lowfreq_candidate_offsets),
                                                queue.h_lowfreq_candidate_offsets_mapped,
                                                0),
                       "cudaHostGetDevicePointer RabitQ persistent lowfreq candidate offsets");
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_lowfreq_candidate_sizes_mapped),
                                     sizeof(uint32_t) * std::max<size_t>(1, slot_count),
                                     cudaHostAllocMapped),
                       "cudaHostAlloc RabitQ persistent lowfreq candidate sizes");
            check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue.d_lowfreq_candidate_sizes),
                                                queue.h_lowfreq_candidate_sizes_mapped,
                                                0),
                       "cudaHostGetDevicePointer RabitQ persistent lowfreq candidate sizes");
            check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&queue.h_lowfreq_candidate_ids_mapped),
                                     sizeof(uint32_t) * queue.lowfreq_candidate_capacity *
                                         std::max<size_t>(1, slot_count),
                                     cudaHostAllocMapped),
                       "cudaHostAlloc RabitQ persistent lowfreq candidate ids");
            check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&queue.d_lowfreq_candidate_ids),
                                                queue.h_lowfreq_candidate_ids_mapped,
                                                0),
                       "cudaHostGetDevicePointer RabitQ persistent lowfreq candidate ids");
        }
        const size_t lowfreq_group_alloc = std::max<size_t>(1, runtime.lowfreq_index.label_size.size());
        const size_t lowfreq_row_alloc = std::max<size_t>(1, runtime.lowfreq_index.sample_ids.size());
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&queue.d_lowfreq_label_size),
                              sizeof(uint32_t) * lowfreq_group_alloc),
                   "cudaMalloc RabitQ persistent lowfreq label_size");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&queue.d_lowfreq_label_offset),
                              sizeof(uint32_t) * lowfreq_group_alloc),
                   "cudaMalloc RabitQ persistent lowfreq label_offset");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&queue.d_lowfreq_sample_ids),
                              sizeof(uint32_t) * lowfreq_row_alloc),
                   "cudaMalloc RabitQ persistent lowfreq sample_ids");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&queue.d_lowfreq_row_filter_mask),
                              sizeof(uint64_t) * lowfreq_row_alloc),
                   "cudaMalloc RabitQ persistent lowfreq row_filter_mask");
        if (!runtime.lowfreq_index.label_size.empty()) {
            check_cuda(cudaMemcpy(queue.d_lowfreq_label_size,
                                  runtime.lowfreq_index.label_size.data(),
                                  sizeof(uint32_t) * runtime.lowfreq_index.label_size.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy RabitQ persistent lowfreq label_size");
            check_cuda(cudaMemcpy(queue.d_lowfreq_label_offset,
                                  runtime.lowfreq_index.label_offset.data(),
                                  sizeof(uint32_t) * runtime.lowfreq_index.label_offset.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy RabitQ persistent lowfreq label_offset");
        }
        if (!runtime.lowfreq_index.sample_ids.empty()) {
            check_cuda(cudaMemcpy(queue.d_lowfreq_sample_ids,
                                  runtime.lowfreq_index.sample_ids.data(),
                                  sizeof(uint32_t) * runtime.lowfreq_index.sample_ids.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy RabitQ persistent lowfreq sample_ids");
            check_cuda(cudaMemcpy(queue.d_lowfreq_row_filter_mask,
                                  runtime.lowfreq_index.row_filter_mask.data(),
                                  sizeof(uint64_t) * runtime.lowfreq_index.row_filter_mask.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy RabitQ persistent lowfreq row_filter_mask");
        }
    } catch (...) {
        destroy_persistent_latency_queue(queue);
        throw;
    }
    return queue;
}

std::vector<RabitqPersistentLatencySlot> create_persistent_latency_slots(uint32_t slot_count)
{
    std::vector<RabitqPersistentLatencySlot> slots(slot_count);
    for (uint32_t slot_idx = 0; slot_idx < slot_count; ++slot_idx) {
        slots[slot_idx].slot_index = slot_idx;
    }
    return slots;
}

void publish_finished_for_host_service(RabitqGraphSearchContext& ctx,
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

void stop_persistent_latency_worker(RabitqPersistentLatencyQueue& queue)
{
    if (queue.h_requests == nullptr) {
        return;
    }
    for (uint32_t slot_idx = 0; slot_idx < queue.slot_count; ++slot_idx) {
        RabitqPersistentLatencyRequestControl& request = queue.h_requests[slot_idx];
        request.stop = 1;
        request.ready_seq = kPersistentLatencyStopSeq;
    }
    _mm_sfence();
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize RabitQ persistent latency stop");
}

void launch_persistent_latency_worker(const RabitqHostIndex& index,
                                      const RabitqSearchOptions& options,
                                      const RabitqImmutableDeviceState& immutable_state,
                                      const RabitqPreparedQueryState& prepared_state,
                                      RabitqGraphSearchContext& ctx,
                                      RabitqPersistentLatencyQueue& queue)
{
    if (queue.slot_count == 0 || queue.slot_count > ctx.graph_scratch_capacity) {
        fail_rabitq("persistent latency worker slots exceed graph scratch capacity");
    }
    const bool use_adaptive_wait = use_adaptive_wait_policy(options.host_reply_wait_policy);
    const uint32_t result_width = ctx.result_width;
    const uint32_t stop_width = ctx.stop_width;
    dispatch_rabitq_kernel_by_queue_size(options.search_queue_size, [&]<int QS>() {
        int blocks_per_sm = 0;
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,
                                                                 persistent_rabitq_latency_kernel<QS>,
                                                                 kThreadInBlock,
                                                                 0),
                   "cudaOccupancyMaxActiveBlocksPerMultiprocessor RabitQ persistent latency");
        cudaDeviceProp prop{};
        int device_id = 0;
        check_cuda(cudaGetDevice(&device_id), "cudaGetDevice RabitQ persistent latency");
        check_cuda(cudaGetDeviceProperties(&prop, device_id),
                   "cudaGetDeviceProperties RabitQ persistent latency");
        const uint32_t resident_blocks =
            static_cast<uint32_t>(std::max(0, blocks_per_sm)) *
            static_cast<uint32_t>(prop.multiProcessorCount);
        if (queue.slot_count > resident_blocks) {
            fail_rabitq("latency_persistent_queue concurrency exceeds resident worker capacity");
        }
        persistent_rabitq_latency_kernel<QS><<<queue.slot_count, kThreadInBlock>>>(
            prepared_state.d_anchors,
            immutable_state.d_anchor_ids,
            immutable_state.d_codes,
            immutable_state.d_long_codes,
            immutable_state.d_factors,
            immutable_state.d_ex_factors,
            prepared_state.d_rotated_query,
            options.use_lut ? prepared_state.d_short_lut : nullptr,
            prepared_state.d_q_k1xsumq,
            prepared_state.d_q_kbxsumq,
            prepared_state.d_centroid_distances,
            prepared_state.d_route_to_original_query,
            prepared_state.d_query_u8,
            ctx.d_result,
            ctx.d_result_distances,
            ctx.d_datadist_list,
            ctx.d_hashdata,
            static_cast<uint32_t>(ctx.graph_scratch_capacity),
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
            ctx.d_row_to_sample,
            ctx.d_label_offset,
            ctx.d_row_filter_mask,
            ctx.d_query_filter_bits,
            ctx.d_query_range_modes,
            ctx.d_query_range_lower_bounds,
            ctx.d_query_range_upper_bounds,
            ctx.d_base_range_values,
            ctx.device_range_count,
            options.dim,
            index.codec.padded_dim,
            index.header.num_anchors,
            index.header.code_bytes,
            index.codec.long_code_bytes,
            index.codec.bits_per_dim,
            static_cast<int>(use_adaptive_wait),
            options.search_queue_size,
            options.search_finish_count,
            parse_search_early_stop_mode(options.search_early_stop_mode),
            effective_search_max_steps(options),
            static_cast<int>(stop_width),
            static_cast<int>(result_width),
            static_cast<uint32_t>(ctx.io_config.vector_payload_words),
            static_cast<uint32_t>(ctx.io_config.elems_per_block),
            options.bit_filter_bonus,
            options.range_filter_bonus,
            ctx.use_inline_gpu_exact_rerank ? 1 : 0,
            queue.d_lowfreq_result,
            queue.d_lowfreq_distances,
            queue.d_lowfreq_candidate_ids,
            queue.d_lowfreq_candidate_offsets,
            queue.d_lowfreq_candidate_sizes,
            queue.d_lowfreq_label_size,
            queue.d_lowfreq_label_offset,
            queue.d_lowfreq_sample_ids,
            queue.d_lowfreq_row_filter_mask,
            queue.lowfreq_group_count,
            queue.lowfreq_total_rows,
            queue.base_rows,
            queue.lowfreq_candidate_mode ? 1 : 0,
            static_cast<int>(queue.lowfreq_shortlist_k),
            lowfreq_use_filter_mask_enabled() ? 1 : 0,
            queue.d_requests,
            queue.d_completions);
    });
    check_cuda(cudaGetLastError(), "launch RabitQ persistent latency worker");
}

RabitqFiniteKernelParams make_rabitq_finite_kernel_params(
    RabitqGraphSearchContext& ctx,
    const RabitqHostIndex& index,
    const GraphRuntime& runtime,
    const RabitqSearchOptions& options,
    const RabitqImmutableDeviceState& immutable_state,
    const RabitqPreparedQueryState& prepared_state)
{
    RabitqFiniteKernelParams params{};
    params.d_anchor_ids = immutable_state.d_anchor_ids;
    params.d_codes = immutable_state.d_codes;
    params.d_long_codes = immutable_state.d_long_codes;
    params.d_factors = immutable_state.d_factors;
    params.d_ex_factors = immutable_state.d_ex_factors;
    params.d_rotated_query = prepared_state.d_rotated_query;
    params.d_short_lut = options.use_lut ? prepared_state.d_short_lut : nullptr;
    params.d_q_k1xsumq = prepared_state.d_q_k1xsumq;
    params.d_q_kbxsumq = prepared_state.d_q_kbxsumq;
    params.d_centroid_distances = prepared_state.d_centroid_distances;
    params.d_route_to_original_query = prepared_state.d_route_to_original_query;
    params.d_result = ctx.d_result;
    params.d_result_distances = ctx.d_result_distances;
    params.d_time_per_block = ctx.d_time_per_block;
    params.d_hashdata = ctx.d_hashdata;
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
    params.d_row_to_sample = ctx.d_row_to_sample;
    params.d_label_offset = ctx.d_label_offset;
    params.d_row_filter_mask = ctx.d_row_filter_mask;
    params.d_query_filter_bits = ctx.d_query_filter_bits;
    params.d_query_range_modes = ctx.d_query_range_modes;
    params.d_query_range_lower_bounds = ctx.d_query_range_lower_bounds;
    params.d_query_range_upper_bounds = ctx.d_query_range_upper_bounds;
    params.d_base_range_values = ctx.d_base_range_values;
    params.d_contexts = ctx.d_finite_contexts;
    params.d_frontier_state = ctx.d_finite_frontier_state;
    params.d_stop_state = ctx.d_finite_stop_state;
    params.d_result_state = ctx.d_finite_result_state;
    params.query_capacity = static_cast<uint32_t>(ctx.total_query_capacity);
    params.graph_scratch_capacity = static_cast<uint32_t>(ctx.graph_scratch_capacity);
    params.range_count = ctx.device_range_count;
    params.padded_dim = index.codec.padded_dim;
    params.num_anchors = index.header.num_anchors;
    params.code_bytes = index.header.code_bytes;
    params.long_code_bytes = index.codec.long_code_bytes;
    params.bits_per_dim = index.codec.bits_per_dim;
    params.mailbox_vector_words = static_cast<uint32_t>(ctx.io_config.vector_payload_words);
    params.mailbox_slot_words = static_cast<uint32_t>(ctx.io_config.elems_per_block);
    params.queue_limit = options.search_queue_size;
    params.finish_limit = options.search_finish_count;
    params.early_stop_mode = parse_search_early_stop_mode(options.search_early_stop_mode);
    params.max_steps = effective_search_max_steps(options);
    params.stop_width = static_cast<int>(ctx.stop_width);
    params.result_width = static_cast<int>(ctx.result_width);
    params.bit_filter_bonus = options.bit_filter_bonus;
    params.range_filter_bonus = options.range_filter_bonus;

    return params;
}

uint32_t launch_rabitq_finite_init(RabitqGraphSearchContext& ctx,
                                   const RabitqFiniteKernelParams& params,
                                   const RabitqSearchOptions& options,
                                   size_t active_query_count)
{
    if (++ctx.finite_context_epoch == 0) ++ctx.finite_context_epoch;
    const uint32_t epoch = ctx.finite_context_epoch;
    dispatch_rabitq_kernel_by_queue_size(options.search_queue_size, [&]<int QS>() {
        finite_rabitq_init_kernel<QS>
            <<<static_cast<int>(active_query_count),
               kThreadInBlock,
               rabitq_result_heap_shared_bytes(ctx.result_width)>>>(
                params, ctx.d_query_ids, static_cast<uint32_t>(active_query_count), epoch);
    });
    check_cuda(cudaGetLastError(), "finite RabitQ init kernel launch");
    return epoch;
}

const char* rabitq_finite_advance_result_name(RabitqFiniteAdvanceResult result)
{
    switch (result) {
        case RabitqFiniteAdvanceResult::PENDING: return "pending";
        case RabitqFiniteAdvanceResult::PARKED: return "parked";
        case RabitqFiniteAdvanceResult::DONE: return "done";
        case RabitqFiniteAdvanceResult::REJECT_QUERY: return "reject_query";
        case RabitqFiniteAdvanceResult::REJECT_EPOCH: return "reject_epoch";
        case RabitqFiniteAdvanceResult::REJECT_STATE: return "reject_state";
        case RabitqFiniteAdvanceResult::REJECT_GENERATION: return "reject_generation";
        case RabitqFiniteAdvanceResult::REJECT_ROW: return "reject_row";
        case RabitqFiniteAdvanceResult::REJECT_BUFFER: return "reject_buffer";
        case RabitqFiniteAdvanceResult::REJECT_STATUS: return "reject_status";
        case RabitqFiniteAdvanceResult::REJECT_MAILBOX: return "reject_mailbox";
    }
    return "unknown";
}

void drive_rabitq_finite_scheduler(RabitqGraphSearchContext& ctx,
                                   const RabitqFiniteKernelParams& params,
                                   const RabitqSearchOptions& options,
                                   const std::vector<uint32_t>& query_indices,
                                   uint32_t context_epoch,
                                   backend_pq::PqFiniteCompletionQueue& completion_queue,
                                   [[maybe_unused]] bool print_summary)
{
    check_cuda(cudaStreamSynchronize(nullptr), "finite RabitQ init kernel completion");
    std::vector<uint8_t> done(ctx.total_query_capacity, 0);
    std::vector<uint32_t> last_generation(ctx.total_query_capacity, 0);
    std::vector<RabitqFiniteReadyToken> tokens;
    tokens.reserve(ctx.finite_ready_token_capacity);
    std::vector<HostCompletionEvent> events;
    events.reserve(ctx.finite_ready_token_capacity);
    size_t done_count = 0;
    size_t launch_count = 0;
    size_t resumed_contexts = 0;
    const auto partial_timeout = std::chrono::microseconds(
        options.finite_partial_batch_timeout_us);
    auto last_progress = std::chrono::steady_clock::now();

    while (done_count < query_indices.size()) {
        if (!completion_queue.wait_pop_batch(events,
                                             ctx.finite_ready_token_capacity,
                                             partial_timeout)) {
            fail_rabitq("finite RabitQ completion queue was cancelled");
        }
        const auto event_time = std::chrono::steady_clock::now();
        const uint64_t event_time_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                event_time.time_since_epoch()).count());
        for (const HostCompletionEvent& event : events) {
            if (event.query_id >= done.size()) {
                fail_rabitq("finite RabitQ completion queue returned an out-of-range route");
            }
            if (event.kind == HostCompletionKind::Done) {
                if (done[event.query_id] == 0) {
                    done[event.query_id] = 1;
                    ++done_count;
                }
                last_progress = event_time;
                continue;
            }
            if (done[event.query_id] != 0) {
                fail_rabitq("finite RabitQ completion queue returned Ready after Done");
            }
            if (event.row == UINT32_MAX || event.buffer > 1 || event.generation == 0) {
                fail_rabitq("finite RabitQ completion queue returned invalid mailbox identity");
            }
            if (event.generation <= last_generation[event.query_id]) continue;
            RabitqFiniteReadyToken token;
            token.query_id = event.query_id;
            token.context_epoch = context_epoch;
            token.generation = event.generation;
            token.row = event.row;
            token.buffer = event.buffer;
            token.ready_ns = event_time_ns >= event.published_ns
                ? event_time_ns - event.published_ns : 0;
            tokens.push_back(token);
        }

        if (!tokens.empty()) {
            check_cuda(cudaMemcpy(ctx.d_finite_ready_tokens,
                                  tokens.data(),
                                  tokens.size() * sizeof(RabitqFiniteReadyToken),
                                  cudaMemcpyHostToDevice),
                       "copy finite RabitQ ready tokens to device");
            dispatch_rabitq_kernel_by_queue_size(options.search_queue_size, [&]<int QS>() {
                finite_rabitq_resume_kernel<QS>
                    <<<static_cast<int>(tokens.size()),
                       kThreadInBlock,
                       rabitq_result_heap_shared_bytes(ctx.result_width)>>>(
                        params,
                        ctx.d_finite_ready_tokens,
                        static_cast<uint32_t>(tokens.size()));
            });
            check_cuda(cudaGetLastError(), "finite RabitQ resume kernel launch");
            check_cuda(cudaStreamSynchronize(nullptr),
                       "finite RabitQ resume kernel completion");
            check_cuda(cudaMemcpy(tokens.data(),
                                  ctx.d_finite_ready_tokens,
                                  tokens.size() * sizeof(RabitqFiniteReadyToken),
                                  cudaMemcpyDeviceToHost),
                       "copy finite RabitQ ready tokens to host");
            ++launch_count;
            resumed_contexts += tokens.size();
            for (const RabitqFiniteReadyToken& token : tokens) {
                if (token.result != RabitqFiniteAdvanceResult::PARKED &&
                    token.result != RabitqFiniteAdvanceResult::DONE) {
                    fail_rabitq("finite RabitQ resume rejected route " +
                                std::to_string(token.query_id) + " generation " +
                                std::to_string(token.generation) + ": " +
                                rabitq_finite_advance_result_name(token.result));
                }
                last_generation[token.query_id] = token.generation;
            }
            tokens.clear();
            last_progress = std::chrono::steady_clock::now();
        }

        if (std::chrono::steady_clock::now() - last_progress >= std::chrono::seconds(60)) {
            const auto stats = completion_queue.stats();
            std::ostringstream message;
            message << "finite RabitQ scheduler made no progress for 60 s"
                    << " ready=" << stats.ready_enqueued
                    << " done=" << stats.done_enqueued
                    << " dequeued=" << stats.dequeued
                    << " duplicate_rejections=" << stats.duplicate_rejections
                    << " invalid_rejections=" << stats.invalid_rejections;
            fail_rabitq(message.str());
        }
    }
    const auto stats = completion_queue.stats();
    std::printf("[rabitq finite scheduler] launches=%zu resumed_contexts=%zu "
                "ready=%llu done=%llu max_completion_depth=%zu\n",
                launch_count,
                resumed_contexts,
                static_cast<unsigned long long>(stats.ready_enqueued),
                static_cast<unsigned long long>(stats.done_enqueued),
                stats.max_depth);
}

void execute_rabitq_graph_batch(RabitqGraphSearchContext& ctx,
                                const RabitqHostIndex& index,
                                const GraphRuntime& runtime,
                                const RabitqSearchOptions& options,
                                const RabitqImmutableDeviceState& immutable_state,
                                const RabitqPreparedQueryState& prepared_state,
                                const std::vector<uint32_t>& query_indices,
                                const char* run_label,
                                bool collect_results,
                                std::vector<std::vector<idx_t>>& results,
                                std::vector<int>& step_nums,
                                std::vector<std::vector<float>>& result_distances,
                                RabitqBatchStats* aggregate_stats,
                                bool print_summary,
                                double* timed_wall_ms,
                                bool finalize_highfreq_results_now,
                                std::future<void>* deferred_streaming_rerank = nullptr)
{
    if (ctx.predicate_counters) check_cuda(cudaMemset(ctx.predicate_counters, 0,
        runtime.query_vectors.rows * sizeof(gpu_predicate::Counters)), "predicate counters reset");
    perf_trace::Scope trace_scope("graph_batch", &runtime);
    const auto profile_prepare_start = std::chrono::steady_clock::now();
    if (!ctx.initialized) {
        fail_rabitq("RabitQ graph context is not initialized");
    }
    const size_t total_query_count = runtime.query_vectors.rows;
    if (total_query_count != ctx.total_query_capacity ||
        prepared_state.routed_query_count != total_query_count ||
        prepared_state.original_query_count != runtime.original_query_count) {
        fail_rabitq("RabitQ graph context/query state size mismatch");
    }
    if (runtime.query_vectors.cols != index.header.dim ||
        static_cast<uint32_t>(options.dim) != index.header.dim) {
        fail_rabitq("query/search dim does not match RabitQ artifact dim");
    }
    const size_t active_query_count = query_indices.size();
    if (active_query_count == 0) {
        results.clear();
        step_nums.clear();
        result_distances.clear();
        return;
    }
    if (active_query_count > ctx.total_query_capacity) {
        fail_rabitq("active RabitQ query count exceeds graph context route capacity");
    }
    if (active_query_count > ctx.graph_scratch_capacity) {
        fail_rabitq("active high-frequency RabitQ query count exceeds graph scratch capacity");
    }
    for (uint32_t query_idx : query_indices) {
        if (query_idx >= total_query_count) {
            fail_rabitq("active query id out of range");
        }
    }
    if (ctx.result_width != highfreq_result_width(options, runtime) ||
        ctx.stop_width != highfreq_stop_width(options) ||
        ctx.use_inline_gpu_exact_rerank !=
            highfreq_inline_gpu_exact_rerank_enabled(options, runtime.vector_scalar_type) ||
        ctx.result_buffer_is_host_mapped != highfreq_cpu_streaming_enabled(options)) {
        fail_rabitq("RabitQ graph context was initialized with incompatible rerank/result settings");
    }
    const bool finite_mode = ctx.finite_query_state_enabled;
    const bool early_host_service = profile_early_host_service_enabled();
    if (early_host_service &&
        (finite_mode || ctx.device_resident_graph || !ctx.request_ring_enabled)) {
        fail_rabitq("early profiling host service requires a memory graph and coupled request ring");
    }
    if (finite_mode != (options.query_worker_mode == "finite")) {
        fail_rabitq("RabitQ graph context/query worker mode mismatch");
    }
    if (ctx.request_ring_enabled != (options.request_mode == "batch_ring") ||
        (ctx.request_ring_enabled &&
         ctx.request_ring_capacity_per_ring != options.request_batch_queries)) {
        fail_rabitq("RabitQ graph context/request dispatch mode mismatch");
    }
    if (ctx.device_resident_graph != rabitq_device_resident_graph_enabled(options)) {
        fail_rabitq("RabitQ graph context/transport mode mismatch");
    }
    if (ctx.request_ring_enabled &&
        ctx.request_ring_consumed_device_mapped !=
            (options.request_consumed_memory == "device")) {
        fail_rabitq("RabitQ graph context/request consumed memory mismatch");
    }

    const bool use_adaptive_wait = use_adaptive_wait_policy(options.host_reply_wait_policy);
    const uint32_t device_range_count =
        (ctx.d_query_range_modes != nullptr) ? runtime.query_ranges.range_count : 0;
    const uint32_t result_width = ctx.result_width;
    const uint32_t stop_width = ctx.stop_width;
    const bool enable_streaming_rerank =
        collect_results && highfreq_cpu_streaming_enabled(options);
    if (deferred_streaming_rerank != nullptr && deferred_streaming_rerank->valid()) {
        fail_rabitq("deferred high-frequency streaming rerank future is already active");
    }

    expansion_trace::Invocation expansion(runtime, query_indices);
    reset_graph_context(ctx);
    check_cuda(cudaMemcpy(ctx.d_query_ids,
                          query_indices.data(),
                          sizeof(uint32_t) * active_query_count,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy RabitQ graph d_query_ids");
    std::vector<int> req_per_loop(20000, 0);
    float milliseconds = 0.0f;
    float gpu_event_milliseconds = 0.0f;

    if (print_summary) {
        printf("%s\n", run_label);
        printf("Submitting RabitQ kernel...\n");
    }

    std::vector<uint32_t> result_query_ids;
    if (collect_results) {
        result_query_ids = query_indices;
    }
    std::future<void> streaming_rerank_future;
    if (enable_streaming_rerank) {
        streaming_rerank_future = std::async(
            std::launch::async,
            [&options,
             &runtime,
             result_query_ids,
             &ctx,
             result_width,
             &results,
             &step_nums,
             &result_distances,
             cpu_rerank_workers = ctx.cpu_rerank_workers] {
                run_highfreq_rabitq_streaming_rerank({ctx.exact_distance_memo.get(), nullptr, ctx.incremental_rerank.get()}, options,
                                                     runtime,
                                                     result_query_ids,
                                                     ctx.status_flag,
                                                     ctx.h_result_mapped,
                                                     ctx.h_result_distances_mapped,
                                                     ctx.h_time_per_block_mapped,
                                                     result_width,
                                                     results,
                                                     step_nums,
                                                     result_distances,
                                                     ctx.cpu_worker_budget.get(),
                                                     cpu_rerank_workers.get(),
                                                     ctx.shortlist_predicate_verified);
            });
    }

    std::unique_ptr<backend_pq::PqFiniteCompletionQueue> finite_completion_queue;
    AsyncIoConfig run_io_config = ctx.io_config;
    if (finite_mode) {
        finite_completion_queue =
            std::make_unique<backend_pq::PqFiniteCompletionQueue>(query_indices);
        run_io_config.completion_sink = finite_completion_queue.get();
    }

    GraphNodeView host_view = make_graph_node_view(runtime);
    host_view.hybrid_residual_runtime = ctx.hybrid_residual_runtime;
    const HostRequestRingView request_ring_view{
        ctx.h_request_ring_entries,
        ctx.h_request_ring_consumed_sequences,
        ctx.request_ring_enabled
            ? (active_query_count + ctx.request_ring_capacity_per_ring - 1) /
                  ctx.request_ring_capacity_per_ring
            : 0,
        ctx.request_ring_capacity_per_ring,
        ctx.request_ring_consumed_device_mapped};
    std::atomic<bool> request_ring_cancelled{false};
    const auto start_host_service = [&]() -> std::thread {
        if (ctx.device_resident_graph) {
            // The resident specialization reads graph rows and full-label CSR on GPU.
        } else if (ctx.request_ring_enabled) {
            return std::thread(host_request_ring_and_set,
                                      ctx.status_memory,
                                      ctx.h_neighbor_memory,
                                      ctx.h_gdr_data,
                                      ctx.status_flag,
                                      host_view,
                                      total_query_count,
                                      query_indices.data(),
                                      active_query_count,
                                      request_ring_view,
                                      &request_ring_cancelled,
                                      run_io_config,
                                      &ctx.host_filter_stats);
        } else {
            return std::thread(host_scan_and_set,
                                      ctx.status_memory,
                                      ctx.h_neighbor_memory,
                                      ctx.h_gdr_data,
                                      req_per_loop.data(),
                                      ctx.status_flag,
                                      host_view,
                                      total_query_count,
                                      query_indices.data(),
                                      active_query_count,
                                      run_io_config,
                                      &ctx.host_filter_stats);
        }
        return std::thread{};
    };
    std::chrono::steady_clock::time_point timed_start{}, gpu_complete_time{}, host_joined_time{}, timed_end{};
    std::thread scan_thread;
    std::unique_ptr<RabitqPhaseSnapshotSampler> phase_snapshot_sampler;
    try {
        if (early_host_service) scan_thread = start_host_service();
        check_cuda(cudaEventRecord(ctx.start), "cudaEventRecord start");
        timed_start = std::chrono::steady_clock::now();
        gpu_complete_time = host_joined_time = timed_end = timed_start;
        perf_trace::span("graph_prepare", &runtime, profile_prepare_start, timed_start);
        RabitqFiniteKernelParams finite_params{};
        uint32_t finite_context_epoch = 0;
        if (finite_mode) {
            finite_params = make_rabitq_finite_kernel_params(
                ctx, index, runtime, options, immutable_state, prepared_state);
            finite_context_epoch = launch_rabitq_finite_init(
                ctx, finite_params, options, active_query_count);
        } else {
            const int grid_dim = static_cast<int>(
                (active_query_count + kNativeMultiQuery - 1) / kNativeMultiQuery);
            dispatch_rabitq_kernel_by_queue_size(options.search_queue_size, [&]<int QS>() {
                const auto launch = [&]<bool kCollectMailboxRtt,
                                        bool kUseRequestRing,
                                        bool kUseInt8Score,
                                        bool kDeviceResident, bool kPredicateBias = false>() {
                    native_rabitq_search_kernel<QS,
                                                kCollectMailboxRtt,
                                                kUseRequestRing,
                                                kUseInt8Score,
                                                kDeviceResident, kPredicateBias>
                        <<<grid_dim,
                           kNativeThreadInBlock,
                           rabitq_result_heap_shared_bytes(ctx.result_width)>>>(
                prepared_state.d_anchors,
                immutable_state.d_anchor_ids,
                immutable_state.d_codes,
                immutable_state.d_long_codes,
                immutable_state.d_factors,
                immutable_state.d_ex_factors,
                prepared_state.d_rotated_query,
                options.use_int8_score ? prepared_state.d_quantized_query : nullptr,
                options.use_int8_score ? prepared_state.d_quantized_query_scales : nullptr,
                options.use_lut ? prepared_state.d_short_lut : nullptr,
                prepared_state.d_q_k1xsumq,
                prepared_state.d_q_kbxsumq,
                prepared_state.d_centroid_distances,
                ctx.d_query_ids,
                prepared_state.d_route_to_original_query,
                prepared_state.d_query_u8,
                prepared_state.d_query_float,
                ctx.d_result,
                ctx.d_result_distances,
                ctx.d_datadist_list,
                ctx.d_hashdata,
                static_cast<uint32_t>(ctx.graph_scratch_capacity),
                static_cast<int>(active_query_count),
                ctx.d_time_per_block,
                ctx.d_phase_state,
                ctx.d_mailbox_rtt_hist,
                ctx.d_neighbor_memory,
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
                ctx.d_row_to_sample,
                ctx.d_label_offset,
                ctx.d_row_filter_mask,
                ctx.d_query_filter_bits,
                ctx.d_gpu_graph_cache_view,
                ctx.predicate_view,
                ctx.d_query_range_modes,
                ctx.d_query_range_lower_bounds,
                ctx.d_query_range_upper_bounds,
                ctx.d_base_range_values,
                device_range_count,
                options.dim,
                index.codec.padded_dim,
                index.header.num_anchors,
                index.header.code_bytes,
                index.codec.long_code_bytes,
                index.codec.bits_per_dim,
                static_cast<int>(use_adaptive_wait),
                options.search_queue_size,
                options.search_finish_count,
                parse_search_early_stop_mode(options.search_early_stop_mode),
                effective_search_max_steps(options),
                static_cast<int>(stop_width),
                static_cast<int>(result_width),
                static_cast<uint32_t>(ctx.io_config.vector_payload_words),
                static_cast<uint32_t>(ctx.io_config.elems_per_block),
                (options.raw_navigation_scores && !kDeviceResident)
                    ? 1.0f : options.bit_filter_bonus,
                (options.raw_navigation_scores && !kDeviceResident)
                    ? 1.0f : options.range_filter_bonus,
                        ctx.use_inline_gpu_exact_rerank ? 1 : 0,
                        (ctx.use_inline_gpu_exact_rerank &&
                         dense_vector_scalar_is_float(runtime.vector_scalar_type)) ? 1 : 0, expansion.view());
                };
                const auto launch_request_mode = [&]<bool kUseInt8Score>() {
                    if (ctx.device_resident_graph) {
                        launch.template operator()<false, false, kUseInt8Score, true>();
                    } else if (ctx.request_ring_enabled) {
                        if (ctx.mailbox_rtt_profile) {
                            launch.template operator()<true, true, kUseInt8Score, false>();
                        } else if (ctx.predicate_bias_frontier) {
                            launch.template operator()<false, true, kUseInt8Score, false, true>();
                        } else {
                            launch.template operator()<false, true, kUseInt8Score, false>();
                        }
                    } else if (ctx.mailbox_rtt_profile) {
                        launch.template operator()<true, false, kUseInt8Score, false>();
                    } else if (ctx.predicate_bias_frontier) {
                        launch.template operator()<false, false, kUseInt8Score, false, true>();
                    } else {
                        launch.template operator()<false, false, kUseInt8Score, false>();
                    }
                };
                if (options.use_int8_score) {
                    launch_request_mode.template operator()<true>();
                } else {
                    launch_request_mode.template operator()<false>();
                }
            });
        }
        check_cuda(cudaGetLastError(), "kernel launch");

        if (!early_host_service) scan_thread = start_host_service();
        phase_snapshot_sampler = start_rabitq_phase_snapshot_sampler(ctx, query_indices);
        if (finite_mode) {
            drive_rabitq_finite_scheduler(ctx,
                                           finite_params,
                                           options,
                                           query_indices,
                                           finite_context_epoch,
                                           *finite_completion_queue,
                                           print_summary);
        }
        check_cuda(cudaEventRecord(ctx.stop), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(ctx.stop), "cudaEventSynchronize stop");
        finish_rabitq_phase_snapshot_sampler(phase_snapshot_sampler);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");
        check_cuda(cudaEventElapsedTime(&milliseconds, ctx.start, ctx.stop), "cudaEventElapsedTime");
        gpu_event_milliseconds = milliseconds;
        gpu_complete_time = std::chrono::steady_clock::now();
        if (scan_thread.joinable()) scan_thread.join();
        host_joined_time = std::chrono::steady_clock::now();
        if (streaming_rerank_future.valid()) {
            if (deferred_streaming_rerank != nullptr) {
                *deferred_streaming_rerank = std::move(streaming_rerank_future);
            } else {
                streaming_rerank_future.get();
            }
        }
        timed_end = std::chrono::steady_clock::now();
        perf_trace::span("graph_launch_to_complete", &runtime, timed_start, gpu_complete_time);
        perf_trace::span("graph_host_join_tail", &runtime, gpu_complete_time, host_joined_time);
        perf_trace::span("graph_rerank_exposed_tail", &runtime, host_joined_time, timed_end);
        perf_trace::value("graph_cuda_event_ms", &runtime, gpu_event_milliseconds);
        if (finite_mode) {
            // CUDA events exclude host-side completion polling and the time
            // spent parked on memory-graph I/O.  Finite QPS is end-to-end.
            milliseconds = static_cast<float>(
                std::chrono::duration<double, std::milli>(timed_end - timed_start).count());
        }
        if (collect_results && timed_wall_ms != nullptr) {
            *timed_wall_ms += std::chrono::duration<double, std::milli>(timed_end - timed_start).count();
        }
    } catch (...) {
        cancel_rabitq_phase_snapshot_sampler(phase_snapshot_sampler);
        if (finite_completion_queue != nullptr) finite_completion_queue->cancel();
        request_ring_cancelled.store(true, std::memory_order_release);
        publish_finished_for_host_service(ctx, query_indices);
        if (scan_thread.joinable()) scan_thread.join();
        if (streaming_rerank_future.valid()) streaming_rerank_future.get();
        throw;
    }

    expansion.write(runtime, query_indices, "rabitq", options.graph_path, options.metadata_sidecar_path);

    print_rabitq_mailbox_rtt_summary(
        ctx, query_indices, options.host_reply_wait_policy);
    print_rabitq_graph_phase_summary(
        ctx,
        runtime,
        query_indices,
        run_label,
        effective_search_max_steps(options),
        gpu_event_milliseconds,
        elapsed_ms(timed_start, gpu_complete_time),
        elapsed_ms(gpu_complete_time, host_joined_time),
        elapsed_ms(host_joined_time, timed_end),
        elapsed_ms(timed_start, timed_end));
    print_rabitq_gpu_graph_cache_summary(ctx, query_indices);

    if (print_summary) {
        printf("%s execution time: %f ms\n",
               finite_mode ? "Finite end-to-end" : "Kernel",
               milliseconds);
        printf("QPS: %f\n", (active_query_count / (milliseconds / 1000.0f)));
        print_host_filter_summary("RabitQ", ctx.host_filter_stats);
    }

    validate_gpu_predicate_trace(ctx, runtime, run_label);
    if (!collect_results) {
        return;
    }

    if (enable_streaming_rerank) {
        uint64_t total_step_sum = 0;
        if (deferred_streaming_rerank == nullptr) {
            for (int step_num : step_nums) {
                total_step_sum += static_cast<uint64_t>(step_num);
            }
        }
        if (aggregate_stats != nullptr) {
            RabitqBatchStats batch_stats;
            batch_stats.query_count = active_query_count;
            batch_stats.kernel_ms = milliseconds;
            batch_stats.qps_sum = milliseconds > 0.0f
                ? (static_cast<double>(active_query_count) / (milliseconds / 1000.0))
                : 0.0;
            batch_stats.run_count = 1;
            batch_stats.total_step_sum = total_step_sum;
            batch_stats.host_filter_stats = ctx.host_filter_stats;
            accumulate_rabitq_batch_stats(*aggregate_stats, batch_stats);
        }
        return;
    }

    std::vector<idx_t> h_result;
    std::vector<dist_t> h_result_distances;
    if (ctx.result_buffer_is_host_mapped) {
        h_result.assign(ctx.h_result_mapped,
                        ctx.h_result_mapped + total_query_count * result_width);
        h_result_distances.assign(ctx.h_result_distances_mapped,
                                  ctx.h_result_distances_mapped + total_query_count * result_width);
    } else {
        h_result.assign(total_query_count * result_width, UINT32_MAX);
        h_result_distances.assign(total_query_count * result_width, FLT_MAX);
        check_cuda(cudaMemcpy(h_result.data(),
                              ctx.d_result,
                              sizeof(idx_t) * h_result.size(),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy h_result");
        check_cuda(cudaMemcpy(h_result_distances.data(),
                              ctx.d_result_distances,
                              sizeof(dist_t) * h_result_distances.size(),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy h_result_distances");
    }
    results.clear();
    step_nums.clear();
    result_distances.clear();
    results.reserve(active_query_count);
    step_nums.reserve(active_query_count);
    result_distances.reserve(active_query_count);
    std::vector<float> h_time_per_block(total_query_count * 6, 0.0f);
    if (ctx.result_buffer_is_host_mapped) {
        h_time_per_block.assign(ctx.h_time_per_block_mapped,
                                ctx.h_time_per_block_mapped + total_query_count * 6);
    } else {
        check_cuda(cudaMemcpy(h_time_per_block.data(),
                              ctx.d_time_per_block,
                              sizeof(float) * h_time_per_block.size(),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy h_time_per_block");
    }

    uint64_t total_step_sum = 0;
    for (size_t i = 0; i < active_query_count; ++i) {
        const uint32_t query_idx = result_query_ids[i];
        std::vector<idx_t> row(result_width);
        std::vector<float> dist_row(result_width, FLT_MAX);
        for (uint32_t j = 0; j < result_width; ++j) {
            row[j] = h_result[static_cast<size_t>(query_idx) * result_width + j];
            dist_row[j] =
                h_result_distances[static_cast<size_t>(query_idx) * result_width + j];
        }
        results.push_back(std::move(row));
        result_distances.push_back(std::move(dist_row));
        const int step_num = static_cast<int>(
            h_time_per_block[static_cast<size_t>(query_idx) * 6 + 4]);
        step_nums.push_back(step_num);
        total_step_sum += static_cast<uint64_t>(step_num);
    }
    if (aggregate_stats != nullptr) {
        RabitqBatchStats batch_stats;
        batch_stats.query_count = active_query_count;
        batch_stats.kernel_ms = milliseconds;
        batch_stats.qps_sum = milliseconds > 0.0f
            ? (static_cast<double>(active_query_count) / (milliseconds / 1000.0))
            : 0.0;
        batch_stats.run_count = 1;
        batch_stats.total_step_sum = total_step_sum;
        batch_stats.host_filter_stats = ctx.host_filter_stats;
        accumulate_rabitq_batch_stats(*aggregate_stats, batch_stats);
    }
    if (finalize_highfreq_results_now) {
        const auto finalize_start = std::chrono::steady_clock::now();
        if (highfreq_staged_float_gpu_exact_rerank_enabled(
                options, runtime.vector_scalar_type)) {
            std::vector<std::vector<idx_t>> gpu_shortlist = std::move(results);
            std::vector<std::vector<float>> gpu_shortlist_distances =
                std::move(result_distances);
            highfreq_rabitq_float_gpu_rerank(options,
                                             runtime,
                                             prepared_state,
                                             ctx.float_gpu_rerank,
                                             result_query_ids,
                                             gpu_shortlist,
                                             &gpu_shortlist_distances,
                                             results,
                                             &result_distances);
        } else {
            finalize_highfreq_rabitq_results({ctx.exact_distance_memo.get(), nullptr, ctx.incremental_rerank.get()}, options,
                                             runtime,
                                             result_query_ids,
                                             results,
                                             &result_distances,
                                             ctx.cpu_worker_budget.get(),
                                             ctx.cpu_rerank_workers.get(),
                                             ctx.shortlist_predicate_verified);
        }
        const auto finalize_end = std::chrono::steady_clock::now();
        if (timed_wall_ms != nullptr &&
            (highfreq_cpu_rerank_enabled(options) ||
             highfreq_staged_float_gpu_exact_rerank_enabled(
                 options, runtime.vector_scalar_type))) {
            *timed_wall_ms +=
                std::chrono::duration<double, std::milli>(finalize_end - finalize_start).count();
        }
    }
}

// The range boundary is also the handoff point for a future chunked D2H producer.
// Today the batch kernel exposes one query-major shortlist only after synchronization.
void lowfreq_rabitq_host_rerank_range(const CpuRerankReuse& reuse, const GraphRuntime& runtime,
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
                                      std::vector<std::vector<float>>& out_distances)
{
    cpu_profile::Task profile("lowfreq_route");

    std::vector<uint8_t> base_row(runtime.query_vectors.cols);
    std::vector<uint32_t> batch_sample_ids;
    batch_sample_ids.reserve(rerank_batch_size);
    for (size_t q = begin_query; q < end_query; ++q) {
        profile.stage(cpu_profile::Stage::Setup);
        profile.query();
        const uint32_t query_idx = query_indices[q];
        if (query_idx >= runtime.query_vectors.rows) {
            fail_rabitq("lowfreq query id out of range");
        }
        const uint8_t* query_u8 = nullptr;
        PreparedFloatL2Query query_float;
        if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
            query_u8 = runtime.route_query_u8(query_idx);
        } else if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
            query_float = prepare_base_float_vector_l2_query(
                runtime,
                runtime.route_query_float(query_idx),
                runtime.query_float_vectors.cols);
        } else {
            fail_rabitq("unsupported dense vector scalar type for low-frequency CPU rerank");
        }
        std::vector<std::pair<float, idx_t>> heap;
        heap.reserve(topk);
        std::vector<idx_t> prev_topk(topk, UINT32_MAX);
        int stable_count = 0;
        int batch_count = 0;

        for (uint32_t batch_begin = 0; batch_begin < shortlist_k; batch_begin += rerank_batch_size) {
            const uint32_t batch_end = std::min<uint32_t>(shortlist_k, batch_begin + rerank_batch_size);
            ++batch_count;
            profile.stage(cpu_profile::Stage::CandidateCollect);
            profile.candidates(batch_end - batch_begin);
            batch_sample_ids.clear();
            for (uint32_t idx = batch_begin; idx < batch_end; ++idx) {
                const idx_t sample_id = gpu_shortlist[static_cast<size_t>(q) * shortlist_k + idx];
                if (sample_id == UINT32_MAX || sample_id >= runtime.base_vectors.rows) {
                    continue;
                }
                if (filter_full_labels && !reuse.posting_verified(query_idx) &&
                    !profile.sample(cpu_profile::Detail::Predicate, [&] {
                        return sample_matches_lowfreq_full_labels(runtime, query_idx, static_cast<uint32_t>(sample_id));
                    })) {
                    continue;
                }
                batch_sample_ids.push_back(static_cast<uint32_t>(sample_id));
            }

            profile.stage(cpu_profile::Stage::VectorPrepare);

            evaluate_rerank_batch(reuse.distances, exact_original_query(runtime, query_idx), profile, runtime, &query_float, batch_sample_ids, [&](uint32_t sample_id) {
                float dist = 0.0f;
                if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
                    dist = compute_base_float_vector_l2(
                        runtime,
                        sample_id,
                        query_float);
                } else {
                    copy_lowfreq_base_vector_row(runtime,
                                                 sample_id,
                                                 base_row.data(),
                                                 runtime.query_vectors.cols);
                    dist = exact_l2_u8(query_u8, base_row.data(), runtime.query_vectors.cols);
                }
                return dist;
            }, [&](uint32_t sample_id, float dist) {
                if (heap.size() < topk) {
                    heap.emplace_back(dist, sample_id);
                    std::push_heap(heap.begin(), heap.end());
                } else if (dist < heap.front().first ||
                           (dist == heap.front().first && sample_id < heap.front().second)) {
                    std::pop_heap(heap.begin(), heap.end());
                    heap.back() = {dist, sample_id};
                    std::push_heap(heap.begin(), heap.end());
                }
            });
            profile.stage(cpu_profile::Stage::EarlyStop);

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
                if (rerank_stable_batches != UINT32_MAX &&
                    static_cast<uint32_t>(stable_count) >= rerank_stable_batches) {
                    prev_topk = std::move(curr_topk);
                    break;
                }
            } else {
                stable_count = 0;
            }
            prev_topk = std::move(curr_topk);
        }

        profile.stage(cpu_profile::Stage::Writeback);
        auto heap_copy = heap;
        std::sort_heap(heap_copy.begin(), heap_copy.end());
        for (size_t i = 0; i < heap_copy.size() && i < topk; ++i) {
            out_results[q][i] = heap_copy[i].second;
            out_distances[q][i] = heap_copy[i].first;
        }
        out_step_nums[q] = batch_count;
    }
}

void lowfreq_rabitq_host_rerank(const CpuRerankReuse& reuse, const GraphRuntime& runtime,
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
                                std::vector<std::vector<float>>& out_distances,
                                RabitqCpuWorkerBudget* cpu_worker_budget = nullptr,
                                RabitqCpuWorkerPool* cpu_rerank_workers = nullptr)
{
    cpu_profile::Task profile("lowfreq_dispatch", "coordinator");
    out_results.assign(query_indices.size(), std::vector<idx_t>(topk, UINT32_MAX));
    out_step_nums.assign(query_indices.size(), 0);
    out_distances.assign(query_indices.size(), std::vector<float>(topk, FLT_MAX));
    if (query_indices.empty()) {
        return;
    }

    const size_t requested_threads = static_cast<size_t>(std::max(1, rerank_threads));
    const size_t worker_count = std::min(requested_threads, query_indices.size());
    if (worker_count <= 1) {
        profile.stage(cpu_profile::Stage::Delegated);
        run_with_rabitq_cpu_worker_permit(cpu_worker_budget, [&] {
            lowfreq_rabitq_host_rerank_range(reuse, runtime,
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
                                             out_distances);
        });
        return;
    }

    dispatch_cpu_rerank_partitions(worker_count, cpu_rerank_workers, profile, [&](size_t worker) {
        const size_t begin = (query_indices.size() * worker) / worker_count;
        const size_t end = (query_indices.size() * (worker + 1)) / worker_count;
        run_with_rabitq_cpu_worker_permit(cpu_worker_budget, [&] {
            lowfreq_rabitq_host_rerank_range(reuse,
                runtime, query_indices, gpu_shortlist, shortlist_k, topk,
                rerank_batch_size, rerank_epsilon, rerank_stable_batches,
                filter_full_labels, begin, end, out_results, out_step_nums,
                out_distances);
        });
    });

}

void lowfreq_rabitq_host_merged_rerank_group(const CpuRerankReuse& reuse,
    const GraphRuntime& runtime,
    const std::vector<uint32_t>& query_indices,
    const std::vector<idx_t>& gpu_shortlist,
    const std::vector<float>& gpu_shortlist_distances,
    const lowfreq_merge::RouteGroup& group,
    uint32_t shortlist_k,
    uint32_t topk,
    uint32_t rerank_batch_size,
    float rerank_epsilon,
    uint32_t rerank_stable_batches,
    bool filter_full_labels,
    std::vector<std::vector<idx_t>>& out_results,
    std::vector<int>& out_step_nums,
    std::vector<std::vector<float>>& out_distances,
    lowfreq_merge::Diagnostics& diagnostics)
{
    if (group.route_positions.empty()) {
        return;
    }
    cpu_profile::Task profile("lowfreq_merged");
    profile.query();
    const uint32_t original_query = group.original_query;
    const uint8_t* query_u8 = nullptr;
    PreparedFloatL2Query query_float;
    uint32_t query_dim = 0;
    if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
        if (original_query >= runtime.original_query_vectors.rows) {
            fail_rabitq("merged low-frequency uint8 original query id is out of range");
        }
        query_u8 = runtime.original_query_vectors.row_ptr(original_query);
        query_dim = runtime.original_query_vectors.cols;
    } else if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        if (original_query >= runtime.original_query_float_vectors.rows) {
            fail_rabitq("merged low-frequency float original query id is out of range");
        }
        query_dim = runtime.original_query_float_vectors.cols;
        query_float = prepare_base_float_vector_l2_query(
            runtime,
            runtime.original_query_float_vectors.row_ptr(original_query),
            query_dim);
    } else {
        fail_rabitq("unsupported dense vector scalar type for merged low-frequency CPU rerank");
    }

    profile.stage(cpu_profile::Stage::CandidateCollect);
    profile.candidates(group.route_positions.size() * shortlist_k);
    std::vector<lowfreq_merge::Candidate> retained_candidates =
        lowfreq_merge::merge_sorted_group_candidates(
            group,
            query_indices,
            gpu_shortlist,
            gpu_shortlist_distances,
            shortlist_k,
            runtime.base_vectors.rows,
            shortlist_k,
            [&](uint32_t route_id, uint32_t sample_id) {
                return !filter_full_labels || reuse.posting_verified(route_id) ||
                    profile.sample(cpu_profile::Detail::Predicate, [&] {
                        return sample_matches_lowfreq_full_labels(runtime, route_id, sample_id);
                    });
            },
            diagnostics,
            [&](lowfreq_merge::CandidateStage stage) {
                profile.stage(stage == lowfreq_merge::CandidateStage::Collect
                                  ? cpu_profile::Stage::CandidateCollect
                                  : cpu_profile::Stage::CandidateRank);
            },
            [&](auto&& operation) {
                return profile.sample(cpu_profile::Detail::HashLookup, operation);
            });

    profile.stage(cpu_profile::Stage::Setup);
    rerank_batch_size = std::max<uint32_t>(1U, rerank_batch_size);
    rerank_stable_batches = std::max<uint32_t>(1U, rerank_stable_batches);
    std::vector<uint8_t> base_row(query_dim);
    std::vector<uint32_t> batch_sample_ids;
    batch_sample_ids.reserve(rerank_batch_size);
    std::vector<std::pair<float, idx_t>> heap;
    heap.reserve(topk);
    std::vector<idx_t> previous_topk(topk, UINT32_MAX);
    int stable_count = 0;
    int batch_count = 0;

    for (size_t batch_begin = 0;
         batch_begin < retained_candidates.size();
         batch_begin += rerank_batch_size) {
        const size_t batch_end = std::min(retained_candidates.size(),
                                          batch_begin + rerank_batch_size);
        ++batch_count;
        profile.stage(cpu_profile::Stage::CandidateCollect);
        batch_sample_ids.clear();
        for (size_t idx = batch_begin; idx < batch_end; ++idx) {
            batch_sample_ids.push_back(retained_candidates[idx].sample_id);
        }
        profile.stage(cpu_profile::Stage::VectorPrepare);

        evaluate_rerank_batch(reuse.distances, original_query, profile, runtime, &query_float, batch_sample_ids, [&](uint32_t sample_id) {
            float distance = 0.0f;
            if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
                distance = compute_base_float_vector_l2(runtime,
                                                        sample_id,
                                                        query_float);
            } else {
                copy_lowfreq_base_vector_row(runtime,
                                             sample_id,
                                             base_row.data(),
                                             query_dim);
                distance = exact_l2_u8(query_u8, base_row.data(), query_dim);
            }
            return distance;
        }, [&](uint32_t sample_id, float distance) {
            if (heap.size() < topk) {
                heap.emplace_back(distance, sample_id);
                std::push_heap(heap.begin(), heap.end());
            } else if (distance < heap.front().first ||
                       (distance == heap.front().first && sample_id < heap.front().second)) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = {distance, sample_id};
                std::push_heap(heap.begin(), heap.end());
            }
        });
        diagnostics.exact_vector_reads += batch_sample_ids.size();
        profile.stage(cpu_profile::Stage::EarlyStop);

        std::vector<idx_t> current_topk(topk, UINT32_MAX);
        auto heap_copy = heap;
        std::sort_heap(heap_copy.begin(), heap_copy.end());
        for (size_t idx = 0; idx < heap_copy.size() && idx < topk; ++idx) {
            current_topk[idx] = heap_copy[idx].second;
        }
        int changed = 0;
        for (uint32_t idx = 0; idx < topk; ++idx) {
            if (current_topk[idx] != previous_topk[idx]) {
                ++changed;
            }
        }
        const float change_rate = static_cast<float>(changed) / static_cast<float>(topk);
        if (heap.size() >= topk && change_rate <= rerank_epsilon) {
            ++stable_count;
            if (rerank_stable_batches != UINT32_MAX &&
                static_cast<uint32_t>(stable_count) >= rerank_stable_batches) {
                previous_topk = std::move(current_topk);
                break;
            }
        } else {
            stable_count = 0;
        }
        previous_topk = std::move(current_topk);
    }

    profile.stage(cpu_profile::Stage::Writeback);
    auto sorted_heap = heap;
    std::sort_heap(sorted_heap.begin(), sorted_heap.end());
    std::vector<idx_t> group_result;
    std::vector<float> group_distances;
    group_result.reserve(std::min<size_t>(sorted_heap.size(), topk));
    group_distances.reserve(std::min<size_t>(sorted_heap.size(), topk));
    for (size_t idx = 0; idx < sorted_heap.size() && idx < topk; ++idx) {
        group_result.push_back(sorted_heap[idx].second);
        group_distances.push_back(sorted_heap[idx].first);
    }
    lowfreq_merge::write_group_result_to_representative_route(group,
                                                               group_result,
                                                               group_distances,
                                                               batch_count,
                                                               out_results,
                                                               out_step_nums,
                                                               out_distances,
                                                               diagnostics);
}

void lowfreq_rabitq_host_merged_rerank_range(const CpuRerankReuse& reuse,
    const GraphRuntime& runtime,
    const std::vector<uint32_t>& query_indices,
    const std::vector<idx_t>& gpu_shortlist,
    const std::vector<float>& gpu_shortlist_distances,
    const std::vector<lowfreq_merge::RouteGroup>& groups,
    uint32_t shortlist_k,
    uint32_t topk,
    uint32_t rerank_batch_size,
    float rerank_epsilon,
    uint32_t rerank_stable_batches,
    bool filter_full_labels,
    size_t begin_group,
    size_t end_group,
    std::vector<std::vector<idx_t>>& out_results,
    std::vector<int>& out_step_nums,
    std::vector<std::vector<float>>& out_distances,
    lowfreq_merge::Diagnostics& diagnostics)
{
    for (size_t group_idx = begin_group; group_idx < end_group; ++group_idx) {
        lowfreq_rabitq_host_merged_rerank_group(reuse, runtime,
                                                query_indices,
                                                gpu_shortlist,
                                                gpu_shortlist_distances,
                                                groups[group_idx],
                                                shortlist_k,
                                                topk,
                                                rerank_batch_size,
                                                rerank_epsilon,
                                                rerank_stable_batches,
                                                filter_full_labels,
                                                out_results,
                                                out_step_nums,
                                                out_distances,
                                                diagnostics);
    }
}

lowfreq_merge::Diagnostics lowfreq_rabitq_host_merged_rerank(const CpuRerankReuse& reuse,
    const GraphRuntime& runtime,
    const std::vector<uint32_t>& query_indices,
    const std::vector<idx_t>& gpu_shortlist,
    const std::vector<float>& gpu_shortlist_distances,
    uint32_t shortlist_k,
    uint32_t topk,
    uint32_t rerank_batch_size,
    float rerank_epsilon,
    uint32_t rerank_stable_batches,
    bool filter_full_labels,
    int rerank_threads,
    std::vector<std::vector<idx_t>>& out_results,
    std::vector<int>& out_step_nums,
    std::vector<std::vector<float>>& out_distances,
    RabitqCpuWorkerBudget* cpu_worker_budget,
    RabitqCpuWorkerPool* cpu_rerank_workers = nullptr)
{
    cpu_profile::Task profile("lowfreq_merged_dispatch", "coordinator");
    out_results.assign(query_indices.size(), {});
    out_step_nums.assign(query_indices.size(), 0);
    out_distances.assign(query_indices.size(), {});
    lowfreq_merge::Diagnostics diagnostics;
    if (query_indices.empty()) {
        return diagnostics;
    }

    const std::vector<lowfreq_merge::RouteGroup> groups =
        lowfreq_merge::build_route_groups(query_indices,
                                          runtime.route_to_original_query,
                                          runtime.original_query_count);
    diagnostics.original_query_groups = groups.size();
    if (groups.empty()) {
        return diagnostics;
    }

    const size_t requested_threads = static_cast<size_t>(std::max(1, rerank_threads));
    const size_t worker_count = std::min(requested_threads, groups.size());

    std::vector<lowfreq_merge::Diagnostics> worker_diagnostics(worker_count);
    if (worker_count == 1) {
        profile.stage(cpu_profile::Stage::Delegated);
        run_with_rabitq_cpu_worker_permit(cpu_worker_budget, [&] {
            lowfreq_rabitq_host_merged_rerank_range(reuse, runtime,
                                                    query_indices,
                                                    gpu_shortlist,
                                                    gpu_shortlist_distances,
                                                    groups,
                                                    shortlist_k,
                                                    topk,
                                                    rerank_batch_size,
                                                    rerank_epsilon,
                                                    rerank_stable_batches,
                                                    filter_full_labels,
                                                    0,
                                                    groups.size(),
                                                    out_results,
                                                    out_step_nums,
                                                    out_distances,
                                                    worker_diagnostics[0]);
        });
        diagnostics += worker_diagnostics[0];
        return diagnostics;
    }

    dispatch_cpu_rerank_partitions(worker_count, cpu_rerank_workers, profile, [&](size_t worker) {
        const size_t begin = (groups.size() * worker) / worker_count;
        const size_t end = (groups.size() * (worker + 1)) / worker_count;
        run_with_rabitq_cpu_worker_permit(cpu_worker_budget, [&] {
            lowfreq_rabitq_host_merged_rerank_range(reuse,
                runtime, query_indices, gpu_shortlist, gpu_shortlist_distances,
                groups, shortlist_k, topk, rerank_batch_size, rerank_epsilon,
                rerank_stable_batches, filter_full_labels, begin, end,
                out_results, out_step_nums, out_distances,
                worker_diagnostics[worker]);
        });
    });
    for (const lowfreq_merge::Diagnostics& worker_stats : worker_diagnostics) {
        diagnostics += worker_stats;
    }
    return diagnostics;
}

lowfreq_merge::Diagnostics finalize_lowfreq_rabitq_results(const CpuRerankReuse& reuse,
    const GraphRuntime& runtime,
    const RabitqSearchOptions& options,
    const std::vector<uint32_t>& query_indices,
    const RabitqLowfreqSearchContext& ctx,
    uint32_t shortlist_k,
    uint32_t topk,
    bool post_filter_full_labels,
    std::vector<std::vector<idx_t>>& out_results,
    std::vector<int>& out_step_nums,
    std::vector<std::vector<float>>& out_distances,
    RabitqCpuWorkerBudget* cpu_worker_budget,
    RabitqCpuWorkerPool* cpu_rerank_workers = nullptr)
{
    perf_trace::Scope trace_scope("lowfreq_cpu_finalize", &runtime);

    lowfreq_merge::Diagnostics diagnostics;
    if (reuse.incremental) {
        incremental_rerank_batch(reuse, runtime, options, query_indices, ctx.host_shortlist,
            ctx.host_approximate_distances, shortlist_k, false, true, out_results,
            &out_distances, &out_step_nums, cpu_worker_budget, cpu_rerank_workers);
        return diagnostics;
    }
    if (options.enable_lowfreq_rerank) {
        if (options.lowfreq_rerank_merge_routes) {
            return lowfreq_rabitq_host_merged_rerank(reuse,
                runtime,
                query_indices,
                ctx.host_shortlist,
                ctx.host_approximate_distances,
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
                cpu_worker_budget, cpu_rerank_workers);
        }
        lowfreq_rabitq_host_rerank(reuse, runtime,
                                   query_indices,
                                   ctx.host_shortlist,
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
                                   cpu_worker_budget, cpu_rerank_workers);
        return diagnostics;
    }

    out_results.assign(query_indices.size(), std::vector<idx_t>(topk, UINT32_MAX));
    out_distances.assign(query_indices.size(), std::vector<float>(topk, FLT_MAX));
    out_step_nums.assign(query_indices.size(), 1);
    for (size_t q = 0; q < query_indices.size(); ++q) {
        uint32_t write_idx = 0;
        for (uint32_t i = 0; i < topk; ++i) {
            const size_t offset = q * shortlist_k + i;
            const idx_t sample_id = ctx.host_shortlist[offset];
            if (sample_id == UINT32_MAX || sample_id >= runtime.base_vectors.rows) {
                continue;
            }
            if (post_filter_full_labels && !reuse.posting_verified(query_indices[q]) &&
                !sample_matches_lowfreq_full_labels(runtime,
                                                    query_indices[q],
                                                    static_cast<uint32_t>(sample_id))) {
                continue;
            }
            out_results[q][write_idx] = sample_id;
            out_distances[q][write_idx] = ctx.host_approximate_distances[offset];
            ++write_idx;
            if (write_idx >= topk) {
                break;
            }
        }
    }
    return diagnostics;
}

void start_posting_stream(const CpuRerankReuse& reuse, RabitqPostingStreamRun& run,
                          RabitqPostingStreamBuffer& buffer,
                          const GraphRuntime& runtime,
                          RabitqSearchOptions options,
                          uint32_t shortlist_k,
                          bool post_filter_full_labels,
                          std::vector<uint32_t> output_positions,
                          std::vector<std::vector<idx_t>>& out_results,
                          std::vector<int>& out_steps,
                          std::vector<std::vector<float>>& out_distances,
                          std::shared_ptr<RabitqCpuWorkerBudget> cpu_worker_budget,
                          std::shared_ptr<RabitqCpuWorkerPool> cpu_rerank_workers,
                          std::shared_ptr<RabitqCpuPhaseTiming> timing = nullptr)
{
    if (options.posting_persistent_workers && cpu_rerank_workers == nullptr) {
        fail_rabitq("persistent posting workers were not initialized before execution");
    }
    if (!options.posting_persistent_workers) cpu_rerank_workers.reset();
    const size_t inflight = posting_completion::stream_inflight_from_env();
    if (inflight != 0 && (cpu_worker_budget == nullptr || !options.enable_lowfreq_rerank)) {
        fail_rabitq("asynchronous posting batches require CPU rerank and a shared scoring budget");
    }
    run.future = std::async(std::launch::async,
        [reuse, &buffer, &runtime, options, shortlist_k, post_filter_full_labels, inflight,
         output_positions = std::move(output_positions), &out_results, &out_steps, &out_distances,
         cpu_worker_budget = std::move(cpu_worker_budget),
         cpu_rerank_workers = std::move(cpu_rerank_workers), timing, cancelled = run.cancelled] {
            const auto start = std::chrono::steady_clock::now();
            double first_batch_ms = -1.0;
            size_t batches = 0;
            struct BatchScratch {
                RabitqLowfreqSearchContext context;
                std::vector<uint32_t> routes;
                std::vector<std::vector<idx_t>> results;
                std::vector<int> steps;
                std::vector<std::vector<float>> distances;
                double finalize_ms = 0.0;
            };
            std::vector<BatchScratch> scratch(std::max<size_t>(1, inflight));
            const bool need_distances = options.incremental_rerank_shortlist > 0 || options.lowfreq_rerank_merge_routes || !options.enable_lowfreq_rerank;
            auto finalize_batch = [&](size_t slot, const std::vector<size_t>& positions) {
                auto& batch_context = scratch[slot].context;
                auto& batch_routes = scratch[slot].routes;
                auto& batch_results = scratch[slot].results;
                auto& batch_steps = scratch[slot].steps;
                auto& batch_distances = scratch[slot].distances;
                batch_routes.clear();
                batch_context.host_shortlist.resize(positions.size() * shortlist_k);
                if (need_distances) batch_context.host_approximate_distances.resize(positions.size() * shortlist_k);
                for (size_t row = 0; row < positions.size(); ++row) {
                    const size_t source = positions[row];
                    batch_routes.push_back(buffer.routes[source]);
                    std::copy_n(buffer.h_ids + source * shortlist_k, shortlist_k,
                                batch_context.host_shortlist.data() + row * shortlist_k);
                    if (need_distances) std::copy_n(buffer.h_distances + source * shortlist_k, shortlist_k,
                                batch_context.host_approximate_distances.data() + row * shortlist_k);
                }
                const auto finalize_start = std::chrono::steady_clock::now();
                (void)finalize_lowfreq_rabitq_results(reuse, runtime, options, batch_routes, batch_context,
                    shortlist_k, static_cast<uint32_t>(options.topk), post_filter_full_labels,
                    batch_results, batch_steps, batch_distances, cpu_worker_budget.get(), cpu_rerank_workers.get());
                scratch[slot].finalize_ms += elapsed_ms(finalize_start, std::chrono::steady_clock::now());
                for (size_t row = 0; row < positions.size(); ++row) {
                    const size_t output = output_positions.empty() ? positions[row] : output_positions[positions[row]];
                    out_results[output] = std::move(batch_results[row]);
                    out_steps[output] = batch_steps[row];
                    out_distances[output] = std::move(batch_distances[row]);
                }
            };
            // Declared after the callback so exceptional unwinding drains all
            // tasks before destroying their callback or slot-owned scratch.
            std::unique_ptr<posting_completion::BatchPipeline> pipeline;
            if (inflight != 0) pipeline = std::make_unique<posting_completion::BatchPipeline>(inflight, *cancelled);
            posting_completion::Groups route_groups;
            if (reuse.incremental) {
                route_groups.positions.resize(buffer.routes.size());
                route_groups.route_to_group.resize(buffer.routes.size());
                for (size_t row = 0; row < buffer.routes.size(); ++row) {
                    route_groups.positions[row].push_back(row);
                    route_groups.route_to_group[row] = row;
                }
            }
            posting_completion::consume(reuse.incremental ? route_groups : buffer.groups,
                options.posting_stream_batch_size, *cancelled,
                [&buffer](size_t row) { return __atomic_load_n(buffer.h_ready + row, __ATOMIC_ACQUIRE) != 0; },
                [&](const std::vector<size_t>& positions) {
                    if (first_batch_ms < 0.0) first_batch_ms = elapsed_ms(start, std::chrono::steady_clock::now());
                    ++batches;
                    if (pipeline) {
                        pipeline->submit([&, positions](size_t slot) { finalize_batch(slot, positions); });
                    } else {
                        finalize_batch(0, positions);
                    }
                });
            if (pipeline) pipeline->finish();
            double finalize_ms = 0.0;
            for (const auto& slot : scratch) finalize_ms += slot.finalize_ms;
            const double total_ms = elapsed_ms(start, std::chrono::steady_clock::now());
            if (timing != nullptr) {
                timing->active_ms = finalize_ms;
                timing->total_ms = total_ms;
            }
            std::printf("[posting-stream-run] kind=%s rows=%zu groups=%zu batches=%zu first_batch_ms=%.3f finalize_ms=%.3f consumer_ms=%.3f persistent_workers=%d cancelled=%d inflight=%zu peak_inflight=%zu\n",
                        output_positions.empty() ? "label" : "range", buffer.routes.size(),
                        buffer.groups.positions.size(), batches, first_batch_ms, finalize_ms, total_ms,
                        cpu_rerank_workers != nullptr ? 1 : 0,
                        cancelled->load(std::memory_order_acquire) ? 1 : 0,
                        inflight, pipeline ? pipeline->peak() : 0);
        });
}

double run_lowfreq_rabitq_search_with_context(const RabitqHostIndex& index,
                                              const GraphRuntime& runtime,
                                              const RabitqSearchOptions& options,
                                              const std::vector<uint32_t>& query_indices,
                                              const RabitqImmutableDeviceState& immutable_state,
                                              const RabitqPreparedQueryState& prepared_state,
                                              const RabitqGraphSearchContext& graph_ctx,
                                              RabitqLowfreqSearchContext& ctx,
                                              std::vector<std::vector<idx_t>>& out_results,
                                              std::vector<int>& out_step_nums,
                                              std::vector<std::vector<float>>& out_distances,
                                              std::future<void>* deferred_host_finalize)
{
    perf_trace::Scope trace_scope("lowfreq_pipeline", &runtime);
    if (query_indices.empty()) {
        out_results.clear();
        out_step_nums.clear();
        out_distances.clear();
        return 0.0;
    }

    const uint32_t topk = static_cast<uint32_t>(options.topk);
    const uint32_t shortlist_k = options.enable_lowfreq_rerank
        ? static_cast<uint32_t>(options.lowfreq_rerank_shortlist)
        : topk;
    const bool pre_filter_full_labels =
        defaults::lowfreq_full_filter_mode_is_pre(options.lowfreq_full_filter_mode);
    const bool post_filter_full_labels =
        defaults::lowfreq_full_filter_mode_is_post(options.lowfreq_full_filter_mode);
    const bool use_lowfreq_filter_mask = lowfreq_use_filter_mask_enabled();
    const bool has_active_range_filters =
        query_ranges_have_active_filters(runtime.query_ranges);
    const bool resident_range_state_ready =
        !has_active_range_filters ||
        (graph_ctx.initialized &&
         graph_ctx.device_range_count == runtime.query_ranges.range_count &&
         graph_ctx.d_query_range_modes != nullptr &&
         graph_ctx.d_query_range_lower_bounds != nullptr &&
         graph_ctx.d_query_range_upper_bounds != nullptr &&
         graph_ctx.d_base_range_values != nullptr);
    const bool resident_posting_requested =
        options.enable_lowfreq_gpu_posting || lowfreq_gpu_posting_enabled();
    const bool use_resident_posting =
        resident_posting_requested &&
        !pre_filter_full_labels &&
        !use_lowfreq_filter_mask &&
        resident_range_state_ready;
    if (resident_posting_requested && !use_resident_posting) {
        std::string reason;
        if (pre_filter_full_labels) {
            reason = "lowfreq_full_filter_mode=pre";
        } else if (use_lowfreq_filter_mask) {
            reason = "GO_FANN_LOWFREQ_USE_FILTER_MASK is enabled";
        } else if (!resident_range_state_ready) {
            reason = "active range filters have no resident GPU range state";
        } else {
            reason = "unknown resident-posting precondition";
        }
        fail_rabitq(
            "requested low-frequency GPU resident posting is unavailable: " + reason);
    }
    if (!ctx.execution_mode_reported) {
        std::cout << "[rabitq-lowfreq-init] config_requested="
                  << (options.enable_lowfreq_gpu_posting ? 1 : 0)
                  << " env_requested=" << (lowfreq_gpu_posting_enabled() ? 1 : 0)
                  << " effective_mode="
                  << (use_resident_posting ? "gpu_resident" : "host_candidates")
                  << " active_range_filters=" << (has_active_range_filters ? 1 : 0)
                  << " full_filter_mode=" << options.lowfreq_full_filter_mode
                  << "\n";
        ctx.execution_mode_reported = true;
    }
    std::vector<uint32_t> candidate_offsets(query_indices.size(), 0);
    std::vector<uint32_t> candidate_sizes(query_indices.size(), 0);
    std::vector<uint32_t> candidate_ids;
    const bool profile_lowfreq = lowfreq_profile_enabled();
    if (options.posting_streaming && (!use_resident_posting || profile_lowfreq)) {
        fail_rabitq("posting_streaming requires resident posting and disabled legacy lowfreq profiling");
    }
    if (deferred_host_finalize != nullptr && deferred_host_finalize->valid()) {
        fail_rabitq("lowfreq finalize future is already active");
    }
    const auto profile_total_start = std::chrono::steady_clock::now();
    const auto candidate_start = profile_total_start;
    uint64_t posting_rows_scanned = 0;
    uint32_t max_label_posting_count = 0;

    for (size_t i = 0; i < query_indices.size(); ++i) {
        const uint32_t query_idx = query_indices[i];
        if (query_idx >= runtime.query_vectors.rows) {
            fail_rabitq("lowfreq query id out of range");
        }
        const uint32_t label = runtime.query_group_ids[query_idx];
        if (label == UINT32_MAX ||
            label >= runtime.lowfreq_index.label_offset.size() ||
            label >= runtime.lowfreq_index.label_size.size()) {
            continue;
        }

        const uint32_t begin = runtime.lowfreq_index.label_offset[label];
        const uint32_t count = runtime.lowfreq_index.label_size[label];
        posting_rows_scanned += count;
        max_label_posting_count = std::max(max_label_posting_count, count);
        if (use_resident_posting) {
            continue;
        }
        const uint64_t query_filter_bit = runtime.query_filter_bits[query_idx];
        candidate_offsets[i] = static_cast<uint32_t>(candidate_ids.size());
        for (uint32_t j = 0; j < count; ++j) {
            const uint32_t row = begin + j;
            if (row >= runtime.lowfreq_index.sample_ids.size() ||
                row >= runtime.lowfreq_index.row_filter_mask.size()) {
                continue;
            }
            if (use_lowfreq_filter_mask &&
                (runtime.lowfreq_index.row_filter_mask[row] & query_filter_bit) == 0ULL) {
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
            if (candidate_ids.size() >= std::numeric_limits<uint32_t>::max()) {
                fail_rabitq("lowfreq candidate list exceeds supported size");
            }
            candidate_ids.push_back(sample_id);
            ++candidate_sizes[i];
        }
    }
    const auto candidate_end = std::chrono::steady_clock::now();
    perf_trace::span("lowfreq_candidate_prepare", &runtime, candidate_start, candidate_end);
    perf_trace::value("lowfreq_posting_rows", &runtime, posting_rows_scanned);

    const auto h2d_start = std::chrono::steady_clock::now();
    ensure_lowfreq_context_capacity(ctx,
                                    query_indices.size(),
                                    use_resident_posting ? 0 : candidate_ids.size(),
                                    shortlist_k);
    check_cuda(cudaMemcpy(ctx.d_query_ids,
                          query_indices.data(),
                          sizeof(uint32_t) * query_indices.size(),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy lowfreq d_query_ids");
    if (use_resident_posting) {
        ensure_lowfreq_resident_posting(ctx, runtime);
    } else {
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
    const auto h2d_end = std::chrono::steady_clock::now();
    perf_trace::span("lowfreq_h2d", &runtime, h2d_start, h2d_end);

    const size_t host_slot_count = checked_size_product(query_indices.size(),
                                                        shortlist_k,
                                                        "lowfreq host shortlist");
    const bool copy_approximate_distances =
        options.incremental_rerank_shortlist > 0 || options.lowfreq_rerank_merge_routes || !options.enable_lowfreq_rerank;
    const auto host_buffer_prepare_start = std::chrono::steady_clock::now();
    const RabitqLowfreqHostBufferStatus host_buffer_status = ensure_lowfreq_host_buffers(
        ctx, host_slot_count, copy_approximate_distances);
    const auto host_buffer_prepare_end = std::chrono::steady_clock::now();

    double timed_ms_total = 0.0;
    double kernel_wall_ms_total = 0.0;
    double shortlist_d2h_ms_total = 0.0;
    double distance_d2h_ms_total = 0.0;
    double host_finalize_ms_total = 0.0;
    int profiled_collect_runs = 0;
    int lowfreq_active_blocks_per_sm = -1;
    uint64_t lowfreq_shared_bytes = 0;
    lowfreq_merge::Diagnostics merged_rerank_diagnostics;
    const bool defer_host_finalize =
        deferred_host_finalize != nullptr &&
        options.warmup_runs == 0 &&
        options.timed_runs == 1 &&
        !profile_lowfreq;
    auto execute_run = [&](bool collect_results) {
        const auto start = std::chrono::steady_clock::now();
        const bool streaming = options.posting_streaming && collect_results;
        RabitqPostingStreamRun stream_run;
        if (streaming) {
            prepare_posting_stream_buffer(ctx.completion, runtime, query_indices, shortlist_k);
            out_results.resize(query_indices.size());
            out_step_nums.resize(query_indices.size());
            out_distances.resize(query_indices.size());
            start_posting_stream({graph_ctx.exact_distance_memo.get(), use_resident_posting ? &graph_ctx.label_posting_predicate.verified_routes : nullptr, graph_ctx.incremental_rerank.get()}, stream_run, ctx.completion, runtime, options, shortlist_k,
                post_filter_full_labels, {}, out_results, out_step_nums, out_distances,
                graph_ctx.cpu_worker_budget, graph_ctx.cpu_rerank_workers);
        }
        if (use_resident_posting) {
            dispatch_lowfreq_scan_kernel(
                static_cast<int>(shortlist_k),
                options.use_int8_score,
                [&]<int kShortlistCapacity, bool kUseInt8Score>() {
                    cudaFuncAttributes attributes{};
                    check_cuda(cudaFuncGetAttributes(
                                   &attributes,
                                   lowfreq_rabitq_resident_posting_kernel<
                                       kShortlistCapacity,
                                       kUseInt8Score>),
                               "cudaFuncGetAttributes lowfreq resident kernel");
                    lowfreq_shared_bytes = attributes.sharedSizeBytes;
                    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                   &lowfreq_active_blocks_per_sm,
                                   lowfreq_rabitq_resident_posting_kernel<
                                       kShortlistCapacity,
                                       kUseInt8Score>,
                                   kThreadInBlock,
                                   0),
                               "cudaOccupancy lowfreq resident kernel");
                    lowfreq_rabitq_resident_posting_kernel<kShortlistCapacity,
                                                           kUseInt8Score>
                        <<<static_cast<int>(query_indices.size()), kThreadInBlock>>>(
                            immutable_state.d_anchor_ids,
                            immutable_state.d_codes,
                            immutable_state.d_long_codes,
                            immutable_state.d_factors,
                            immutable_state.d_ex_factors,
                            prepared_state.d_rotated_query,
                            kUseInt8Score ? prepared_state.d_quantized_query : nullptr,
                            kUseInt8Score ? prepared_state.d_quantized_query_scales : nullptr,
                            options.use_lut ? prepared_state.d_short_lut : nullptr,
                            prepared_state.d_q_k1xsumq,
                            prepared_state.d_q_kbxsumq,
                            prepared_state.d_centroid_distances,
                            ctx.d_query_ids,
                            prepared_state.d_route_to_original_query,
                            streaming ? ctx.completion.d_ids : ctx.d_result,
                            streaming ? ctx.completion.d_distances : ctx.d_distances,
                            ctx.d_query_group_ids,
                            graph_ctx.d_query_filter_bits,
                            graph_ctx.d_query_range_modes,
                            graph_ctx.d_query_range_lower_bounds,
                            graph_ctx.d_query_range_upper_bounds,
                            graph_ctx.d_base_range_values,
                            graph_ctx.device_range_count,
                            ctx.d_label_size,
                            ctx.d_label_offset,
                            ctx.d_sample_ids,
                            nullptr,
                            ctx.resident_group_count,
                            ctx.resident_total_rows,
                            static_cast<uint32_t>(runtime.base_vectors.rows),
                            static_cast<int>(query_indices.size()),
                            static_cast<int>(topk),
                            static_cast<int>(shortlist_k),
                            index.header.num_anchors,
                            index.codec.padded_dim,
                            index.header.code_bytes,
                            index.codec.long_code_bytes,
                            index.codec.bits_per_dim,
                            0,
                            streaming ? ctx.completion.d_ready : nullptr,
                            graph_ctx.label_posting_predicate.view,
                            options.posting_predicate_order == "score_first");
                });
        } else {
            dispatch_lowfreq_scan_kernel(
                static_cast<int>(shortlist_k),
                options.use_int8_score,
                [&]<int kShortlistCapacity, bool kUseInt8Score>() {
                    cudaFuncAttributes attributes{};
                    check_cuda(cudaFuncGetAttributes(
                                   &attributes,
                                   lowfreq_rabitq_scan_kernel<
                                       kShortlistCapacity,
                                       kUseInt8Score>),
                               "cudaFuncGetAttributes lowfreq candidate kernel");
                    lowfreq_shared_bytes = attributes.sharedSizeBytes;
                    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                   &lowfreq_active_blocks_per_sm,
                                   lowfreq_rabitq_scan_kernel<
                                       kShortlistCapacity,
                                       kUseInt8Score>,
                                   kThreadInBlock,
                                   0),
                               "cudaOccupancy lowfreq candidate kernel");
                    lowfreq_rabitq_scan_kernel<kShortlistCapacity, kUseInt8Score>
                        <<<static_cast<int>(query_indices.size()), kThreadInBlock>>>(
                            immutable_state.d_anchor_ids,
                            immutable_state.d_codes,
                            immutable_state.d_long_codes,
                            immutable_state.d_factors,
                            immutable_state.d_ex_factors,
                            prepared_state.d_rotated_query,
                            kUseInt8Score ? prepared_state.d_quantized_query : nullptr,
                            kUseInt8Score ? prepared_state.d_quantized_query_scales : nullptr,
                            options.use_lut ? prepared_state.d_short_lut : nullptr,
                            prepared_state.d_q_k1xsumq,
                            prepared_state.d_q_kbxsumq,
                            prepared_state.d_centroid_distances,
                            ctx.d_query_ids,
                            prepared_state.d_route_to_original_query,
                            ctx.d_result,
                            ctx.d_distances,
                            candidate_ids.empty() ? nullptr : ctx.d_candidate_ids,
                            ctx.d_candidate_offsets,
                            ctx.d_candidate_sizes,
                            static_cast<int>(query_indices.size()),
                            static_cast<int>(topk),
                            static_cast<int>(shortlist_k),
                            index.header.num_anchors,
                            index.codec.padded_dim,
                            index.header.code_bytes,
                            index.codec.long_code_bytes,
                            index.codec.bits_per_dim);
                });
        }
        check_cuda(cudaGetLastError(), "lowfreq RabitQ kernel launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize lowfreq RabitQ kernel");
        const auto kernel_end = std::chrono::steady_clock::now();
        perf_trace::span("lowfreq_kernel_launch_to_complete", &runtime, start, kernel_end);
        if (collect_results) {
            kernel_wall_ms_total += elapsed_ms(start, kernel_end);
        }

        if (streaming) {
            if (defer_host_finalize) *deferred_host_finalize = std::move(stream_run.future);
            else stream_run.future.get();
            timed_ms_total += elapsed_ms(start, std::chrono::steady_clock::now());
            return;
        }
        if (collect_results) {
            ++profiled_collect_runs;
            const auto shortlist_d2h_start = std::chrono::steady_clock::now();
            check_cuda(cudaMemcpy(ctx.host_shortlist.data(),
                                  ctx.d_result,
                                  sizeof(idx_t) * host_slot_count,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy lowfreq RabitQ shortlist");
            const auto shortlist_d2h_end = std::chrono::steady_clock::now();
            perf_trace::span("lowfreq_shortlist_d2h", &runtime, shortlist_d2h_start, shortlist_d2h_end);
            shortlist_d2h_ms_total += elapsed_ms(shortlist_d2h_start, shortlist_d2h_end);
            if (copy_approximate_distances) {
                const auto distance_d2h_start = std::chrono::steady_clock::now();
                check_cuda(cudaMemcpy(ctx.host_approximate_distances.data(),
                                      ctx.d_distances,
                                      sizeof(float) * host_slot_count,
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy lowfreq RabitQ distances");
                const auto distance_d2h_end = std::chrono::steady_clock::now();
                perf_trace::span("lowfreq_distance_d2h", &runtime, distance_d2h_start, distance_d2h_end);
                distance_d2h_ms_total += elapsed_ms(distance_d2h_start, distance_d2h_end);
            }
            const auto host_finalize_start = std::chrono::steady_clock::now();
            if (defer_host_finalize) {
                const RabitqSearchOptions finalize_options = options;
                *deferred_host_finalize = std::async(
                    cpu_rerank_schedule::host_finalize_launch(options.cpu_rerank_schedule),
                    [&runtime,
                     &query_indices,
                     &ctx,
                     &out_results,
                     &out_step_nums,
                     &out_distances,
                     finalize_options,
                     shortlist_k,
                     topk,
                     post_filter_full_labels,
                     &graph_ctx, use_resident_posting,
                     cpu_worker_budget = graph_ctx.cpu_worker_budget] {
                        (void)finalize_lowfreq_rabitq_results({graph_ctx.exact_distance_memo.get(), use_resident_posting ? &graph_ctx.label_posting_predicate.verified_routes : nullptr, graph_ctx.incremental_rerank.get()},
                            runtime,
                            finalize_options,
                            query_indices,
                            ctx,
                            shortlist_k,
                            topk,
                            post_filter_full_labels,
                            out_results,
                            out_step_nums,
                            out_distances,
                            cpu_worker_budget.get());
                    });
            } else {
                merged_rerank_diagnostics = finalize_lowfreq_rabitq_results({graph_ctx.exact_distance_memo.get(), use_resident_posting ? &graph_ctx.label_posting_predicate.verified_routes : nullptr, graph_ctx.incremental_rerank.get()},
                    runtime,
                    options,
                    query_indices,
                    ctx,
                    shortlist_k,
                    topk,
                    post_filter_full_labels,
                    out_results,
                    out_step_nums,
                    out_distances,
                    graph_ctx.cpu_worker_budget.get());
            }
            const auto end = std::chrono::steady_clock::now();
            if (!defer_host_finalize) {
                host_finalize_ms_total += elapsed_ms(host_finalize_start, end);
            }
            timed_ms_total += std::chrono::duration<double, std::milli>(end - start).count();
        }
    };

    for (int run = 0; run < options.warmup_runs; ++run) {
        execute_run(false);
    }
    for (int run = 0; run < options.timed_runs; ++run) {
        execute_run(true);
    }
    if (!ctx.kernel_resources_reported && lowfreq_active_blocks_per_sm >= 0) {
        std::cout << "[rabitq-lowfreq-kernel] mode="
                  << (use_resident_posting ? "gpu_resident" : "host_candidates")
                  << " shortlist_k=" << shortlist_k
                  << " shortlist_capacity="
                  << lowfreq_shortlist_capacity(static_cast<int>(shortlist_k))
                  << " active_blocks_per_sm=" << lowfreq_active_blocks_per_sm
                  << " shared_bytes=" << lowfreq_shared_bytes
                  << " scorer="
                  << (options.use_int8_score ? "dp4a_int8" : "fp32_bit")
                  << "\n";
        ctx.kernel_resources_reported = true;
    }
    if (profile_lowfreq) {
        const auto profile_total_end = std::chrono::steady_clock::now();
        uint32_t max_candidates = 0;
        uint64_t nonempty_queries = 0;
        for (uint32_t size : candidate_sizes) {
            max_candidates = std::max(max_candidates, size);
            if (size > 0) {
                ++nonempty_queries;
            }
        }
        const double query_count = static_cast<double>(query_indices.size());
        const double avg_label_posting_count = query_count > 0.0
            ? static_cast<double>(posting_rows_scanned) / query_count
            : 0.0;
        const double avg_candidates = query_count > 0.0
            ? static_cast<double>(candidate_ids.size()) / query_count
            : 0.0;
        std::cout << "[lowfreq-profile]"
                  << " queries=" << query_indices.size()
                  << " nonempty_queries=" << nonempty_queries
                  << " shortlist_k=" << shortlist_k
                  << " shortlist_capacity="
                  << lowfreq_shortlist_capacity(static_cast<int>(shortlist_k))
                  << " scorer=" << (options.use_int8_score ? "dp4a_int8" : "fp32_bit")
                  << " timed_runs=" << options.timed_runs
                  << " collect_runs=" << profiled_collect_runs
                  << " rerank=" << (options.enable_lowfreq_rerank ? 1 : 0)
                  << " rerank_merge_routes=" << (options.lowfreq_rerank_merge_routes ? 1 : 0)
                  << " use_filter_mask=" << (use_lowfreq_filter_mask ? 1 : 0)
                  << " gpu_posting=" << (use_resident_posting ? 1 : 0)
                  << " full_filter_mode=" << options.lowfreq_full_filter_mode
                  << " posting_rows_scanned=" << posting_rows_scanned
                  << " avg_label_posting_count=" << avg_label_posting_count
                  << " max_label_posting_count=" << max_label_posting_count
                  << " candidates=" << candidate_ids.size()
                  << " avg_candidates=" << avg_candidates
                  << " max_candidates=" << max_candidates
                  << " original_query_groups=" << merged_rerank_diagnostics.original_query_groups
                  << " route_candidate_slots=" << merged_rerank_diagnostics.route_candidate_slots
                  << " route_valid_candidates=" << merged_rerank_diagnostics.route_valid_candidates
                  << " unique_candidates=" << merged_rerank_diagnostics.unique_candidates
                  << " sorted_merge_groups=" << merged_rerank_diagnostics.sorted_merge_groups
                  << " fallback_merge_groups=" << merged_rerank_diagnostics.fallback_merge_groups
                  << " merge_visited_slots=" << merged_rerank_diagnostics.merge_visited_slots
                  << " incomplete_candidate_count_groups="
                  << merged_rerank_diagnostics.incomplete_candidate_count_groups
                  << " retained_candidates=" << merged_rerank_diagnostics.retained_candidates
                  << " exact_vector_reads=" << merged_rerank_diagnostics.exact_vector_reads
                  << " representative_result_rows="
                  << merged_rerank_diagnostics.representative_result_rows
                  << " compact_empty_result_rows="
                  << merged_rerank_diagnostics.compact_empty_result_rows
                  << " route_step_rows=" << merged_rerank_diagnostics.route_step_rows
                  << " host_shortlist_required_slots=" << host_buffer_status.required_slots
                  << " host_shortlist_buffer_slots=" << ctx.host_shortlist.size()
                  << " host_shortlist_buffer_capacity=" << ctx.host_shortlist.capacity()
                  << " host_shortlist_buffer_reused="
                  << (host_buffer_status.shortlist_reused ? 1 : 0)
                  << " host_approx_distance_required="
                  << (host_buffer_status.approximate_distances_required ? 1 : 0)
                  << " host_approx_distance_buffer_slots="
                  << ctx.host_approximate_distances.size()
                  << " host_approx_distance_buffer_capacity="
                  << ctx.host_approximate_distances.capacity()
                  << " host_approx_distance_buffer_reused="
                  << (host_buffer_status.approximate_distances_reused ? 1 : 0)
                  << " host_buffer_prepare_ms="
                  << elapsed_ms(host_buffer_prepare_start, host_buffer_prepare_end)
                  << " shortlist_d2h_bytes_per_run="
                  << host_slot_count * sizeof(idx_t)
                  << " distance_d2h_bytes_per_run="
                  << (copy_approximate_distances ? host_slot_count * sizeof(float) : 0)
                  << " candidate_build_ms=" << elapsed_ms(candidate_start, candidate_end)
                  << " h2d_ms=" << elapsed_ms(h2d_start, h2d_end)
                  << " timed_ms_total=" << timed_ms_total
                  << " kernel_wall_ms_total=" << kernel_wall_ms_total
                  << " shortlist_d2h_ms_total=" << shortlist_d2h_ms_total
                  << " distance_d2h_ms_total=" << distance_d2h_ms_total
                  << " host_finalize_ms_total=" << host_finalize_ms_total
                  << " total_wall_ms=" << elapsed_ms(profile_total_start, profile_total_end)
                  << "\n";
        std::cout.flush();
    }
    return timed_ms_total;
}

RabitqRangePostingRunStats run_range_posting_rabitq_search(
    const RabitqHostIndex& index,
    const GraphRuntime& runtime,
    const RabitqSearchOptions& options,
    const RabitqImmutableDeviceState& immutable_state,
    const RabitqPreparedQueryState& prepared_state,
    const RabitqGraphSearchContext& graph_ctx,
    const range_posting::Index& posting_index,
    RabitqRangePostingSearchContext& ctx,
    std::vector<std::vector<idx_t>>& out_results,
    std::vector<int>& out_step_nums,
    std::vector<std::vector<float>>& out_distances,
    std::future<void>* deferred_host_finalize = nullptr,
    std::shared_ptr<RabitqCpuPhaseTiming>* deferred_cpu_timing = nullptr)
{
    perf_trace::Scope trace_scope("range_posting_pipeline", &runtime);
    RabitqRangePostingRunStats stats;
    if (deferred_host_finalize != nullptr && deferred_host_finalize->valid()) {
        fail_rabitq("deferred range-posting rerank future is already active");
    }
    if (deferred_cpu_timing != nullptr) {
        *deferred_cpu_timing = nullptr;
    }
    if (deferred_cpu_timing != nullptr && deferred_host_finalize == nullptr) {
        fail_rabitq("deferred range-posting timing requires a deferred future");
    }
    if (!ctx.initialized) {
        out_results.clear();
        out_step_nums.clear();
        out_distances.clear();
        return stats;
    }
    if (!graph_ctx.initialized || graph_ctx.d_query_range_modes == nullptr ||
        graph_ctx.d_query_range_lower_bounds == nullptr ||
        graph_ctx.d_query_range_upper_bounds == nullptr ||
        graph_ctx.d_base_range_values == nullptr) {
        fail_rabitq("range posting scan requires initialized graph range buffers");
    }
    const size_t plan_count = ctx.plan.representative_route_ids.size();
    out_results.assign(plan_count,
                       std::vector<idx_t>(static_cast<size_t>(options.topk), UINT32_MAX));
    out_step_nums.assign(plan_count, 0);
    out_distances.assign(plan_count,
                         std::vector<float>(static_cast<size_t>(options.topk), FLT_MAX));
    const auto total_start = std::chrono::steady_clock::now();
    std::vector<RabitqRangePostingPendingSubset> pending_subsets;
    std::vector<std::future<void>> stream_futures;
    std::vector<std::shared_ptr<RabitqCpuPhaseTiming>> stream_timings;

    const auto execute_subset = [&](const std::vector<uint32_t>& plan_positions,
                                    bool streamed,
                                    uint32_t shortlist_k,
                                    bool postfilter_subset) {
        if (plan_positions.empty()) return;
        if (options.incremental_rerank_shortlist > 0 && (streamed || !options.posting_streaming))
            fail_rabitq("incremental range rerank requires resident posting completion streaming");
        const uint32_t query_count = static_cast<uint32_t>(plan_positions.size());
        if (query_count > ctx.device_query_capacity) {
            fail_rabitq("range posting subset exceeds device query capacity");
        }
        const size_t result_slots = checked_size_product(
            query_count, static_cast<size_t>(shortlist_k), "range posting subset output");
        if (result_slots > ctx.result_slot_capacity) {
            fail_rabitq("range posting subset exceeds result-slot capacity");
        }
        std::vector<uint32_t> route_ids;
        std::vector<uint64_t> posting_offsets;
        std::vector<uint64_t> posting_sizes;
        route_ids.reserve(query_count);
        posting_offsets.reserve(query_count);
        posting_sizes.reserve(query_count);
        for (uint32_t plan_position : plan_positions) {
            if (plan_position >= plan_count) {
                fail_rabitq("range posting plan position is out of range");
            }
            route_ids.push_back(ctx.plan.representative_route_ids[plan_position]);
            posting_offsets.push_back(ctx.plan.posting_offsets[plan_position]);
            posting_sizes.push_back(ctx.plan.posting_sizes[plan_position]);
        }
        check_cuda(cudaMemcpy(ctx.d_query_ids,
                              route_ids.data(),
                              query_count * sizeof(uint32_t),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy range posting subset query ids");
        if (!streamed) {
            check_cuda(cudaMemcpy(ctx.d_posting_offsets,
                                  posting_offsets.data(),
                                  query_count * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy resident range posting offsets");
            check_cuda(cudaMemcpy(ctx.d_posting_sizes,
                                  posting_sizes.data(),
                                  query_count * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy resident range posting sizes");
        }

        const bool result_streaming = options.posting_streaming && !streamed;
        auto& completion_buffer = postfilter_subset ? ctx.postfilter_completion : ctx.resident_completion;
        RabitqPostingStreamRun stream_run;
        if (result_streaming) {
            prepare_posting_stream_buffer(completion_buffer, runtime, route_ids, shortlist_k);
            RabitqSearchOptions rerank_options = options;
            rerank_options.enable_lowfreq_rerank = true;
            rerank_options.lowfreq_rerank_merge_routes = false;
            rerank_options.lowfreq_rerank_batch_size = static_cast<int>(shortlist_k);
            rerank_options.lowfreq_rerank_epsilon = 0.0f;
            rerank_options.lowfreq_rerank_stable_batches = UINT32_MAX;
            auto timing = std::make_shared<RabitqCpuPhaseTiming>();
            stream_timings.push_back(timing);
            start_posting_stream({graph_ctx.exact_distance_memo.get(), &graph_ctx.range_posting_predicate.verified_routes, graph_ctx.incremental_rerank.get()}, stream_run, completion_buffer, runtime, rerank_options, shortlist_k,
                postfilter_subset, plan_positions, out_results, out_step_nums, out_distances,
                graph_ctx.cpu_worker_budget, graph_ctx.cpu_rerank_workers, timing);
        }
        const auto kernel_start = std::chrono::steady_clock::now();
        HostCandidateStreamStats stream_stats;
        int active_blocks_per_sm = -1;
        size_t kernel_shared_bytes = 0;
        if (streamed) {
            if (shortlist_k > kLowfreqRerankShortlistMax) {
                fail_rabitq(
                    "candidate-stream posting only supports the ordinary shortlist");
            }
            std::vector<HostCandidateSourceView> sources;
            sources.reserve(query_count);
            for (uint32_t plan_position : plan_positions) {
                const uint32_t route_idx =
                    ctx.plan.representative_route_ids[plan_position];
                const uint32_t range_index =
                    ctx.plan.posting_range_indices[plan_position];
                const range_posting::DimensionView& dimension =
                    posting_index.dimension(range_index);
                const uint64_t dimension_base =
                    static_cast<uint64_t>(range_index) * posting_index.row_count();
                const uint64_t flat_begin = ctx.plan.posting_offsets[plan_position];
                if (flat_begin < dimension_base ||
                    flat_begin - dimension_base > dimension.row_count ||
                    ctx.plan.posting_sizes[plan_position] >
                        dimension.row_count - (flat_begin - dimension_base)) {
                    fail_rabitq("candidate stream range slice is outside its dimension");
                }
                sources.push_back({route_idx,
                                   UINT32_MAX,
                                   QueryCandidateSource::RangePosting,
                                   dimension.sample_ids + (flat_begin - dimension_base),
                                   ctx.plan.posting_sizes[plan_position]});
                graph_ctx.status_memory[static_cast<size_t>(route_idx) * 16] = 0;
                graph_ctx.status_flag[static_cast<size_t>(route_idx) * 16] = 0;
                graph_ctx.h_gdr_data[static_cast<size_t>(route_idx) *
                                     kGdrMailboxStride] = 0;
            }
            _mm_sfence();
            dispatch_lowfreq_scan_kernel(
                static_cast<int>(shortlist_k),
                options.use_int8_score,
                [&]<int kShortlistCapacity, bool kUseInt8Score>() {
                    cudaFuncAttributes attributes{};
                    check_cuda(cudaFuncGetAttributes(
                                   &attributes,
                                   candidate_stream_rabitq_scan_kernel<
                                       kShortlistCapacity,
                                       kUseInt8Score>),
                               "cudaFuncGetAttributes candidate stream posting kernel");
                    kernel_shared_bytes = attributes.sharedSizeBytes;
                    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                   &active_blocks_per_sm,
                                   candidate_stream_rabitq_scan_kernel<
                                       kShortlistCapacity,
                                       kUseInt8Score>,
                                   kThreadInBlock,
                                   0),
                               "cudaOccupancy candidate stream posting kernel");
                    candidate_stream_rabitq_scan_kernel<kShortlistCapacity, kUseInt8Score>
                        <<<query_count, kThreadInBlock>>>(
                            immutable_state.d_anchor_ids,
                            immutable_state.d_codes,
                            immutable_state.d_long_codes,
                            immutable_state.d_factors,
                            immutable_state.d_ex_factors,
                            prepared_state.d_rotated_query,
                            kUseInt8Score ? prepared_state.d_quantized_query : nullptr,
                            kUseInt8Score ? prepared_state.d_quantized_query_scales : nullptr,
                            prepared_state.d_short_lut,
                            prepared_state.d_q_k1xsumq,
                            prepared_state.d_q_kbxsumq,
                            prepared_state.d_centroid_distances,
                            ctx.d_query_ids,
                            prepared_state.d_route_to_original_query,
                            graph_ctx.d_neighbor_memory,
                            graph_ctx.d_status_memory,
                            graph_ctx.d_gdr_data,
                            graph_ctx.d_status_flag,
                            graph_ctx.io_config.payload_slot_count,
                            static_cast<uint32_t>(graph_ctx.io_config.elems_per_block),
                            ctx.d_result,
                            ctx.d_exact_pass_counts,
                            query_count,
                            static_cast<int>(shortlist_k),
                            index.header.num_anchors,
                            index.codec.padded_dim,
                            index.header.code_bytes,
                            index.codec.long_code_bytes,
                            index.codec.bits_per_dim);
                });
            check_cuda(cudaGetLastError(), "candidate stream RabitQ kernel launch");
            host_candidate_scan_and_set(graph_ctx.status_memory,
                                        graph_ctx.h_neighbor_memory,
                                        graph_ctx.h_gdr_data,
                                        graph_ctx.status_flag,
                                        runtime,
                                        sources.data(),
                                        sources.size(),
                                        graph_ctx.io_config,
                                        &stream_stats);
            check_cuda(cudaDeviceSynchronize(), "candidate stream RabitQ kernel sync");
            for (uint32_t route_idx : route_ids) {
                if (graph_ctx.status_flag[static_cast<size_t>(route_idx) * 16] != 3) {
                    fail_rabitq("candidate stream terminated without a valid GPU result");
                }
            }
            stats.stream_chunks += stream_stats.chunks_published;
            stats.scanned_rows += stream_stats.source_rows_scanned;
        } else {
            const auto& segments = postfilter_subset ? ctx.postfilter_segments : ctx.resident_segments;
            const bool segmented = segments.count != 0;
            if (segmented && result_streaming) {
                check_cuda(cudaMemset(ctx.d_segment_completion_counts, 0, query_count * sizeof(uint32_t)),
                           "reset range segment completion counters");
            }
            if (segmented) check_cuda(cudaEventRecord(ctx.segment_start), "record segment start");
            dispatch_range_posting_scan_kernel(
                static_cast<int>(shortlist_k),
                options.use_int8_score,
                [&]<int kShortlistCapacity, bool kUseInt8Score>() {
                    cudaFuncAttributes attributes{};
                    check_cuda(cudaFuncGetAttributes(
                                   &attributes,
                                   range_posting_rabitq_scan_kernel<
                                       kShortlistCapacity,
                                       kUseInt8Score>),
                               "cudaFuncGetAttributes resident range posting kernel");
                    kernel_shared_bytes = attributes.sharedSizeBytes;
                    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                   &active_blocks_per_sm,
                                   range_posting_rabitq_scan_kernel<
                                       kShortlistCapacity,
                                       kUseInt8Score>,
                                   kThreadInBlock,
                                   0),
                               "cudaOccupancy resident range posting kernel");
                    range_posting_rabitq_scan_kernel<kShortlistCapacity, kUseInt8Score>
                        <<<segmented ? segments.count : query_count, kThreadInBlock>>>(
                            immutable_state.d_anchor_ids,
                            immutable_state.d_codes,
                            immutable_state.d_long_codes,
                            immutable_state.d_factors,
                            immutable_state.d_ex_factors,
                            prepared_state.d_rotated_query,
                            kUseInt8Score ? prepared_state.d_quantized_query : nullptr,
                            kUseInt8Score ? prepared_state.d_quantized_query_scales : nullptr,
                            prepared_state.d_short_lut,
                            prepared_state.d_q_k1xsumq,
                            prepared_state.d_q_kbxsumq,
                            prepared_state.d_centroid_distances,
                            segmented ? segments.d_route_ids : ctx.d_query_ids,
                            prepared_state.d_route_to_original_query,
                            ctx.d_sample_ids,
                            segmented ? segments.d_offsets : ctx.d_posting_offsets,
                            segmented ? segments.d_sizes : ctx.d_posting_sizes,
                            graph_ctx.d_query_range_modes,
                            graph_ctx.d_query_range_lower_bounds,
                            graph_ctx.d_query_range_upper_bounds,
                            graph_ctx.d_base_range_values,
                            graph_ctx.device_range_count,
                            ctx.resident_row_count,
                            runtime.base_vectors.rows,
                            segmented ? ctx.d_segment_ids : (result_streaming ? completion_buffer.d_ids : ctx.d_result),
                            segmented ? ctx.d_segment_distances :
                                (options.incremental_rerank_shortlist > 0 ? completion_buffer.d_distances : nullptr),
                            segmented ? ctx.d_segment_pass_counts : ctx.d_exact_pass_counts,
                            segmented ? segments.count : query_count,
                            static_cast<int>(shortlist_k),
                            index.header.num_anchors,
                            index.codec.padded_dim,
                            index.header.code_bytes,
                            index.codec.long_code_bytes,
                            index.codec.bits_per_dim,
                            posting_segments::Completion<idx_t>{
                                segments.d_query_positions, segments.d_query_indptr,
                                ctx.d_segment_completion_counts,
                                segmented && result_streaming ? completion_buffer.d_ids : nullptr,
                                ctx.d_exact_pass_counts,
                                result_streaming ? completion_buffer.d_ready : nullptr},
                            graph_ctx.range_posting_predicate.view,
                            options.posting_predicate_order == "score_first",
                            options.incremental_rerank_shortlist > 0 ? completion_buffer.d_distances : nullptr);
                    if (segmented) {
                        check_cuda(cudaGetLastError(), "range segment scan launch");
                        check_cuda(cudaEventRecord(ctx.segment_scan_end), "record segment scan end");
                        if (!result_streaming) {
                            posting_segments::merge_shortlists<kShortlistCapacity, idx_t>
                                <<<query_count, kThreadInBlock>>>(segments.d_query_indptr,
                                    ctx.d_segment_ids, ctx.d_segment_distances, ctx.d_segment_pass_counts,
                                    ctx.d_result, ctx.d_exact_pass_counts, static_cast<int>(shortlist_k));
                        }
                        check_cuda(cudaEventRecord(ctx.segment_merge_end), "record segment merge end");
                    }
                });
            check_cuda(cudaGetLastError(), "range_posting_rabitq_scan_kernel launch");
            check_cuda(cudaDeviceSynchronize(), "range_posting_rabitq_scan_kernel sync");
            if (segmented) {
                float scan_ms = 0.0f, merge_ms = 0.0f;
                check_cuda(cudaEventElapsedTime(&scan_ms, ctx.segment_start, ctx.segment_scan_end),
                           "range segment scan time");
                check_cuda(cudaEventElapsedTime(&merge_ms, ctx.segment_scan_end, ctx.segment_merge_end),
                           "range segment merge time");
                std::cout << "[range-posting-segments-run] routes=" << query_count
                          << " tasks=" << segments.count << " shortlist=" << shortlist_k
                          << " fused_completion=" << (result_streaming ? 1 : 0)
                          << " scan_ms=" << scan_ms << " merge_ms=" << merge_ms << std::endl;
            }
            stats.scanned_rows += std::accumulate(posting_sizes.begin(),
                                                  posting_sizes.end(),
                                                  uint64_t{0});
        }
        const auto kernel_end = std::chrono::steady_clock::now();
        stats.kernel_wall_ms += elapsed_ms(kernel_start, kernel_end);
        perf_trace::span("range_posting_kernel_launch_to_complete", &runtime, kernel_start, kernel_end);

        const auto d2h_start = std::chrono::steady_clock::now();
        if (result_streaming) {
            std::copy_n(completion_buffer.h_ids, result_slots, ctx.host_shortlist.data());
        } else {
        check_cuda(cudaMemcpy(ctx.host_shortlist.data(),
                              ctx.d_result,
                              result_slots * sizeof(idx_t),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy range posting subset shortlist");
        }
        check_cuda(cudaMemcpy(ctx.host_exact_pass_counts.data(),
                              ctx.d_exact_pass_counts,
                              query_count * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy range posting subset pass counts");
        const auto d2h_end = std::chrono::steady_clock::now();
        stats.shortlist_d2h_ms += elapsed_ms(d2h_start, d2h_end);
        perf_trace::span("range_posting_d2h", &runtime, d2h_start, d2h_end);
        // Score-gated rows were not predicate-tested, so the total number of
        // matches is unknown. Counts remain diagnostic and never drive results.
        if (!streamed && options.posting_predicate_order == "score_first") {
            stats.exact_pass_rows_available = false;
        }
        stats.exact_pass_rows += std::accumulate(ctx.host_exact_pass_counts.begin(),
                                                ctx.host_exact_pass_counts.begin() + query_count,
                                                uint64_t{0});
        const bool filter_full_labels = std::any_of(
            route_ids.begin(), route_ids.end(), [&runtime](uint32_t route_id) {
                return route_id < runtime.route_execution_modes.size() &&
                    runtime.route_execution_modes[route_id] ==
                        QueryRouteExecution::GpuResidentScanPostFilter;
            });
        if (filter_full_labels != postfilter_subset) {
            fail_rabitq("range posting subset mixes incompatible shortlist policies");
        }
        const uint32_t dispatched_capacity = static_cast<uint32_t>(
            range_posting_shortlist_capacity(static_cast<int>(shortlist_k)));
        if (postfilter_subset) {
            stats.postfilter_active_blocks_per_sm = active_blocks_per_sm;
            stats.postfilter_shortlist_capacity = dispatched_capacity;
            stats.postfilter_shared_bytes = kernel_shared_bytes;
        } else {
            stats.ordinary_active_blocks_per_sm =
                stats.ordinary_active_blocks_per_sm < 0
                ? active_blocks_per_sm
                : std::min(stats.ordinary_active_blocks_per_sm, active_blocks_per_sm);
            stats.ordinary_shortlist_capacity = std::max(
                stats.ordinary_shortlist_capacity, dispatched_capacity);
            stats.ordinary_shared_bytes = std::max(
                stats.ordinary_shared_bytes,
                static_cast<uint64_t>(kernel_shared_bytes));
        }
        for (uint32_t query_idx = 0; query_idx < query_count; ++query_idx) {
            const uint32_t route_id = route_ids[query_idx];
            const bool route_postfilters =
                route_id < runtime.route_execution_modes.size() &&
                runtime.route_execution_modes[route_id] ==
                    QueryRouteExecution::GpuResidentScanPostFilter;
            for (uint32_t shortlist_idx = 0;
                 shortlist_idx < shortlist_k;
                 ++shortlist_idx) {
                const idx_t sample_id =
                    ctx.host_shortlist[static_cast<size_t>(query_idx) *
                                           shortlist_k +
                                       shortlist_idx];
                if (sample_id == UINT32_MAX || sample_id >= runtime.base_vectors.rows) {
                    continue;
                }
                ++stats.shortlist_rows;
                if (route_postfilters) {
                    ++stats.postfilter_shortlist_rows;
                    if (!(!streamed && route_id < graph_ctx.range_posting_predicate.verified_routes.size() &&
                          graph_ctx.range_posting_predicate.verified_routes[route_id]) &&
                        !sample_matches_route_predicate(
                            runtime, route_id, static_cast<uint32_t>(sample_id))) {
                        continue;
                    }
                    ++stats.postfilter_residual_pass_rows;
                }
                ++stats.exact_rerank_rows;
            }
        }

        if (result_streaming) {
            stream_futures.push_back(std::move(stream_run.future));
        } else if (deferred_host_finalize != nullptr) {
            RabitqRangePostingPendingSubset pending;
            pending.plan_positions = plan_positions;
            pending.route_ids = std::move(route_ids);
            pending.shortlist.assign(ctx.host_shortlist.begin(),
                                     ctx.host_shortlist.begin() + result_slots);
            pending.shortlist_k = shortlist_k;
            pending.filter_full_labels = filter_full_labels;
            pending.gpu_predicate_executed = !streamed;
            pending_subsets.push_back(std::move(pending));
        } else {
            std::vector<std::vector<idx_t>> subset_results;
            std::vector<int> subset_steps;
            std::vector<std::vector<float>> subset_distances;
            const auto rerank_start = std::chrono::steady_clock::now();
            lowfreq_rabitq_host_rerank({graph_ctx.exact_distance_memo.get(), !streamed ? &graph_ctx.range_posting_predicate.verified_routes : nullptr}, runtime,
                                       route_ids,
                                       ctx.host_shortlist,
                                       shortlist_k,
                                       static_cast<uint32_t>(options.topk),
                                       shortlist_k,
                                       0.0f,
                                       UINT32_MAX,
                                       filter_full_labels,
                                       options.highfreq_rerank_threads,
                                       subset_results,
                                       subset_steps,
                                       subset_distances,
                                       graph_ctx.cpu_worker_budget.get());
            const auto rerank_end = std::chrono::steady_clock::now();
            stats.rerank_wall_ms += elapsed_ms(rerank_start, rerank_end);
            perf_trace::span("range_posting_cpu_rerank", &runtime, rerank_start, rerank_end);
            for (size_t subset_idx = 0; subset_idx < plan_positions.size(); ++subset_idx) {
                const uint32_t plan_position = plan_positions[subset_idx];
                out_results[plan_position] = std::move(subset_results[subset_idx]);
                out_step_nums[plan_position] = subset_steps[subset_idx];
                out_distances[plan_position] = std::move(subset_distances[subset_idx]);
            }
        }
    };

    execute_subset(ctx.plan.resident_plan_positions, false, ctx.shortlist_k, false);
    execute_subset(ctx.plan.resident_postfilter_plan_positions,
                   false,
                   ctx.postfilter_shortlist_k,
                   true);
    execute_subset(ctx.plan.stream_plan_positions, true, ctx.shortlist_k, false);
    stats.resident_routes = ctx.plan.resident_plan_positions.size() +
        ctx.plan.resident_postfilter_plan_positions.size();
    stats.streamed_routes = ctx.plan.stream_plan_positions.size();
    stats.total_wall_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    if (deferred_host_finalize != nullptr && !pending_subsets.empty()) {
        auto cpu_timing = std::make_shared<RabitqCpuPhaseTiming>();
        if (deferred_cpu_timing != nullptr) {
            *deferred_cpu_timing = cpu_timing;
        }
        const uint32_t topk = static_cast<uint32_t>(options.topk);
        const int rerank_threads = options.highfreq_rerank_threads;
        *deferred_host_finalize = std::async(
            cpu_rerank_schedule::host_finalize_launch(options.cpu_rerank_schedule),
            [&runtime,
             &out_results,
             &out_step_nums,
             &out_distances,
             pending_subsets = std::move(pending_subsets),
             topk,
             rerank_threads,
             cpu_timing, &graph_ctx,
             cpu_worker_budget = graph_ctx.cpu_worker_budget] () mutable {
                perf_trace::Scope trace_scope("range_posting_cpu_rerank", &runtime);
                run_timed_rabitq_cpu_phase(
                    cpu_timing,
                    [&] {
                        for (RabitqRangePostingPendingSubset& pending : pending_subsets) {
                            std::vector<std::vector<idx_t>> subset_results;
                            std::vector<int> subset_steps;
                            std::vector<std::vector<float>> subset_distances;
                            lowfreq_rabitq_host_rerank({graph_ctx.exact_distance_memo.get(), pending.gpu_predicate_executed ? &graph_ctx.range_posting_predicate.verified_routes : nullptr}, runtime,
                                                       pending.route_ids,
                                                       pending.shortlist,
                                                       pending.shortlist_k,
                                                       topk,
                                                       pending.shortlist_k,
                                                       0.0f,
                                                       UINT32_MAX,
                                                       pending.filter_full_labels,
                                                       rerank_threads,
                                                       subset_results,
                                                       subset_steps,
                                                       subset_distances,
                                                       cpu_worker_budget.get());
                            for (size_t subset_idx = 0;
                                 subset_idx < pending.plan_positions.size();
                                 ++subset_idx) {
                                const uint32_t plan_position =
                                    pending.plan_positions[subset_idx];
                                out_results[plan_position] =
                                    std::move(subset_results[subset_idx]);
                                out_step_nums[plan_position] = subset_steps[subset_idx];
                                out_distances[plan_position] =
                                    std::move(subset_distances[subset_idx]);
                            }
                        }
                    });
            });
    }
    if (!stream_futures.empty()) {
        auto timing = std::make_shared<RabitqCpuPhaseTiming>();
        auto collect = [futures = std::move(stream_futures), timings = std::move(stream_timings), timing]() mutable {
            for (auto& future : futures) future.get();
            for (const auto& part : timings) {
                timing->active_ms += part->active_ms;
                timing->total_ms += part->total_ms;
            }
        };
        if (deferred_host_finalize != nullptr) {
            auto prior = std::move(*deferred_host_finalize);
            const auto prior_timing = deferred_cpu_timing == nullptr ? nullptr : *deferred_cpu_timing;
            *deferred_host_finalize = std::async(cpu_rerank_schedule::host_finalize_launch(options.cpu_rerank_schedule),
                [collect = std::move(collect), prior = std::move(prior), prior_timing, timing]() mutable {
                    collect();
                    if (prior.valid()) prior.get();
                    if (prior_timing != nullptr) {
                        timing->active_ms += prior_timing->active_ms;
                        timing->total_ms += prior_timing->total_ms;
                    }
                });
            if (deferred_cpu_timing != nullptr) *deferred_cpu_timing = timing;
        } else {
            collect();
            stats.rerank_wall_ms += timing->active_ms;
            stats.total_wall_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
        }
    }
    return stats;
}

void validate_rabitq_search_options(const RabitqSearchOptions& options)
{
    expansion_trace::validate_options(options);
    if (options.posting_predicate_order != "filter_first" &&
        options.posting_predicate_order != "score_first") {
        fail_rabitq("posting_predicate_order must be filter_first or score_first");
    }
    if (options.posting_predicate_order == "score_first" &&
        (options.query_worker_mode != "coupled" || options.graph_transport != "mailbox" ||
         options.graph_storage_mode != "memory" ||
         options.latency_persistent_queue || options.posting_persistent_workers ||
         options.rabitq_predicate_storage != "csr" ||
         (options.rabitq_predicate_mode != "gpu_exact" &&
          options.rabitq_predicate_mode != "gpu_exact_bias"))) {
        fail_rabitq("score-first posting requires native coupled in-memory mailbox GPU CSR with nonpersistent posting");
    }
    gpu_predicate::require(options.rabitq_predicate_mode == "cpu" || options.rabitq_predicate_mode == "cpu_simplified" ||
        options.rabitq_predicate_mode == "gpu_exact" || options.rabitq_predicate_mode == "gpu_exact_bias", "invalid predicate mode");
    gpu_predicate::require(std::isfinite(options.rabitq_predicate_bias_alpha) && options.rabitq_predicate_bias_alpha >= 0 && options.rabitq_predicate_bias_alpha <= FLT_MAX, "invalid alpha");
    gpu_predicate::require(options.rabitq_predicate_bias_alpha == 0 || options.rabitq_predicate_mode == "gpu_exact_bias", "positive alpha requires bias mode");
    gpu_predicate::require(options.rabitq_predicate_mode != "gpu_exact_bias" ||
        options.search_early_stop_mode == "off" || options.search_early_stop_mode == "adaptive", "bias supports off or adaptive stop");

    if (options.graph_persistent_workers &&
        (options.query_worker_mode != "coupled" || !highfreq_cpu_rerank_enabled(options))) {
        fail_rabitq("graph_persistent_workers requires coupled graph execution with CPU rerank");
    }
    if (options.posting_persistent_workers && !options.posting_streaming) {
        fail_rabitq("posting_persistent_workers requires posting_streaming");
    }
    if (options.posting_stream_batch_size == 0 || options.posting_stream_batch_size > 1048576) {
        fail_rabitq("posting_stream_batch_size must be in [1,1048576]");
    }
    if (options.posting_streaming && options.query_worker_mode != "coupled") {
        fail_rabitq("posting_streaming currently requires coupled graph execution");
    }
    if (options.posting_schedule != "legacy" &&
        options.posting_schedule != "graph_first" &&
        options.posting_schedule != "posting_first") {
        fail_rabitq("posting_schedule must be one of: legacy, graph_first, posting_first");
    }
    if (options.posting_schedule != "legacy" && !highfreq_cpu_rerank_enabled(options)) {
        fail_rabitq("non-legacy posting_schedule requires high-frequency CPU rerank");
    }

    const range_posting::Mode range_posting_mode =
        range_posting::parse_mode(options.range_posting_mode);
    if (options.range_posting_segment_rows != 0 &&
        (options.range_posting_segment_rows < 32 ||
         options.range_posting_segment_rows > 1048576 ||
         range_posting_mode == range_posting::Mode::Off)) {
        fail_rabitq("range_posting_segment_rows requires [32,1048576] and enabled range posting");
    }
    if (options.range_posting_shortlist < options.topk ||
        options.range_posting_shortlist > defaults::max_range_posting_shortlist) {
        fail_rabitq("range posting shortlist must be in [topk, 400]");
    }
    if (options.range_posting_postfilter_shortlist_multiplier <= 0 ||
        options.range_posting_shortlist >
            kRangePostingShortlistMax /
                options.range_posting_postfilter_shortlist_multiplier) {
        fail_rabitq(
            "range posting postfilter shortlist multiplier must be positive and "
            "produce at most 4096 candidates");
    }
    if (options.range_posting_and_filter_mode != "auto" &&
        options.range_posting_and_filter_mode != "pre" &&
        options.range_posting_and_filter_mode != "post" &&
        options.range_posting_and_filter_mode != "label") {
        fail_rabitq(
            "range posting AND filter mode must be one of: auto, pre, post, label");
    }
    if (options.range_label_and_policy != "auto" &&
        options.range_label_and_policy != "label_prefer" &&
        options.range_label_and_policy != "range_prefer" &&
        options.range_label_and_policy != "dual") {
        fail_rabitq(
            "range label AND policy must be one of: auto, label_prefer, range_prefer, dual");
    }
    if ((options.range_label_and_policy == "range_prefer" ||
         options.range_label_and_policy == "dual") &&
        options.range_posting_and_filter_mode == "label") {
        fail_rabitq(
            "range label AND policy range_prefer or dual cannot use AND filter mode label");
    }
    if (options.range_label_and_policy != "auto" &&
        range_posting_mode == range_posting::Mode::Off) {
        fail_rabitq(
            "non-auto range label AND policy requires enabled range posting");
    }
    if (range_posting_mode != range_posting::Mode::Off) {
        if (options.range_posting_index_path.empty() ||
            options.range_posting_scan_threshold == 0) {
            fail_rabitq("enabled range posting requires an index path and positive scan threshold");
        }
        if (!highfreq_cpu_rerank_enabled(options) ||
            !defaults::highfreq_rerank_algorithm_is_exact(
                options.highfreq_rerank_algorithm)) {
            fail_rabitq("enabled range posting requires exact high-frequency CPU rerank");
        }
    }
    const bool range_graph_enabled = options.range_graph_hybrid_or_enabled ||
        options.range_graph_hybrid_and_enabled;
    if (range_graph_enabled) {
        if (range_posting_mode == range_posting::Mode::Off) {
            fail_rabitq(
                "range graph hybrid execution requires enabled range posting");
        }
        if (options.query_worker_mode != "coupled") {
            fail_rabitq("dual-graph RabitQ execution currently requires coupled query workers");
        }
        if (options.range_graph_storage_mode != "memory") {
            fail_rabitq("range_graph_storage_mode must be memory");
        }
        if (options.range_graph_path.empty() ||
            options.range_graph_data_label_spmat_path.empty() ||
            options.range_graph_label_manifest_path.empty() ||
            options.range_graph_seed_sidecar_path.empty() ||
            options.range_graph_original_label_seed_sidecar_path.empty() ||
            options.range_graph_metadata_sidecar_path.empty() ||
            options.range_graph_group_path.empty()) {
            fail_rabitq("dual-graph RabitQ execution requires all range graph sidecar paths");
        }
    }
    if (options.range_graph_hybrid_and_enabled && !options.per_step_full_filter) {
        fail_rabitq(
            "range graph hybrid-AND execution requires per-step full filtering");
    }
    if (options.range_graph_hybrid_and_enabled &&
        range_posting_mode != range_posting::Mode::Select) {
        fail_rabitq(
            "range graph hybrid-AND execution requires range_posting_mode=select");
    }

    if (!defaults::is_valid_rerank_vector_storage_mode(
            options.rerank_vector_storage_mode)) {
        fail_rabitq(
            "native RabitQ rerank_vector_storage_mode must be one of: default, mmap_fp16, memory_int8");
    }
    const bool custom_rerank_vectors =
        !defaults::rerank_vector_storage_is_default(
            options.rerank_vector_storage_mode);
    if (custom_rerank_vectors) {
        const bool fp16_gpu_post =
            defaults::rerank_vector_storage_is_mmap_fp16(options.rerank_vector_storage_mode) &&
            defaults::highfreq_rerank_mode_is_gpu_post(effective_highfreq_rerank_mode(options));
        if (!highfreq_cpu_rerank_enabled(options) && !fp16_gpu_post) {
            fail_rabitq(
                "native RabitQ custom rerank vector storage requires CPU rerank or mmap_fp16 GPU-post");
        }

        if (defaults::rerank_vector_storage_is_memory_int8(
                options.rerank_vector_storage_mode) &&
            options.rerank_vector_artifact_path.empty()) {
            fail_rabitq(
                "native RabitQ memory_int8 requires rerank_vector_artifact_path");
        }
    }
    if (options.dim <= 0 || options.dim % 4 != 0) {
        fail_rabitq("native RabitQ search dim must be positive and divisible by 4; got dim=" +
                    std::to_string(options.dim));
    }
    if (options.degree <= 0 || options.degree > FIXED_DEGREE) {
        fail_rabitq("native RabitQ search currently supports degree in [1, 64]");
    }
    if (options.topk <= 0 || options.topk > kTopk) {
        fail_rabitq("native RabitQ search currently supports topk in [1, 10]");
    }
    if (!defaults::is_valid_highfreq_rerank_mode(options.highfreq_rerank_mode)) {
        fail_rabitq(
            "native RabitQ search highfreq_rerank_mode must be one of: gpu, gpu_post, cpu, off");
    }
    if (!defaults::is_valid_highfreq_rerank_algorithm(options.highfreq_rerank_algorithm)) {
        fail_rabitq("native RabitQ search highfreq_rerank_algorithm must be one of: exact, heuristic");
    }
    if (options.incremental_rerank_shortlist < 0 ||
        (options.incremental_rerank_shortlist > 0 &&
         (options.incremental_rerank_shortlist < options.topk ||
          !highfreq_cpu_rerank_enabled(options) || !options.enable_lowfreq_rerank ||
          !options.highfreq_rerank_streaming || !options.posting_streaming ||
          !options.full_filter || !options.raw_navigation_scores || options.rerank_distance_reuse ||
          options.graph_storage_mode != "memory" || options.query_worker_mode != "coupled")))
        fail_rabitq("incremental rerank requires L>=topk, memory coupled CPU rerank, graph/posting streaming, full filter, and distance reuse off");
    if (options.highfreq_rerank_shortlist < options.topk ||
        options.highfreq_rerank_shortlist > kHighfreqRerankShortlistMax) {
        fail_rabitq("native RabitQ search highfreq_rerank_shortlist must be in [topk, 512]");
    }
    if (options.highfreq_rerank_batch_size <= 0) {
        fail_rabitq("native RabitQ search highfreq_rerank_batch_size must be >= 1");
    }
    if (options.highfreq_rerank_epsilon < 0.0f) {
        fail_rabitq("native RabitQ search highfreq_rerank_epsilon must be >= 0");
    }
    if (options.highfreq_rerank_stable_batches <= 0) {
        fail_rabitq("native RabitQ search highfreq_rerank_stable_batches must be >= 1");
    }
    if (options.highfreq_rerank_threads <= 0) {
        fail_rabitq("native RabitQ search highfreq_rerank_threads must be >= 1");
    }
    if (options.highfreq_rerank_stream_batch_size <= 0) {
        fail_rabitq("native RabitQ search highfreq_rerank_stream_batch_size must be >= 1");
    }
    if (!highfreq_cpu_rerank_enabled(options) &&
        (options.highfreq_rerank_streaming ||
         options.highfreq_rerank_stream_batch_size != defaults::default_highfreq_rerank_stream_batch_size)) {
        fail_rabitq("native RabitQ search highfreq_rerank_streaming requires highfreq_rerank_mode=cpu");
    }
    if (options.enable_lowfreq_rerank) {
        if (options.lowfreq_rerank_shortlist < options.topk ||
            options.lowfreq_rerank_shortlist > kLowfreqRerankShortlistMax) {
            fail_rabitq("native RabitQ search lowfreq_rerank_shortlist must be in [topk, 400]");
        }
        if (options.lowfreq_rerank_batch_size <= 0) {
            fail_rabitq("native RabitQ search lowfreq_rerank_batch_size must be >= 1");
        }
        if (options.lowfreq_rerank_epsilon < 0.0f) {
            fail_rabitq("native RabitQ search lowfreq_rerank_epsilon must be >= 0");
        }
        if (options.lowfreq_rerank_stable_batches <= 0) {
            fail_rabitq("native RabitQ search lowfreq_rerank_stable_batches must be >= 1");
        }
    }
    if (options.lowfreq_rerank_merge_routes && !options.enable_lowfreq_rerank) {
        fail_rabitq("native RabitQ search lowfreq_rerank_merge_routes requires low-frequency rerank");
    }
    if (!defaults::is_valid_lowfreq_full_filter_mode(options.lowfreq_full_filter_mode)) {
        fail_rabitq("native RabitQ search lowfreq_full_filter_mode must be one of: pre, post, off");
    }
    if (options.graph_storage_mode != "memory") {
        fail_rabitq("native RabitQ search graph_storage_mode must be memory");
    }
    if (options.graph_transport != "mailbox" &&
        options.graph_transport != "device_resident") {
        fail_rabitq(
            "native RabitQ search graph_transport must be one of: mailbox, device_resident");
    }
    if (rabitq_device_resident_graph_enabled(options)) {
        const std::string_view rerank_mode = effective_highfreq_rerank_mode(options);
        if (options.graph_storage_mode != "memory" ||
            options.query_worker_mode != "coupled" ||
            options.request_mode != "scan" ||
            options.latency_persistent_queue ||
            options.gpu_graph_row_cache_mib != 0 ||

            options.range_graph_hybrid_or_enabled ||
            options.range_graph_hybrid_and_enabled ||
            (!defaults::highfreq_rerank_mode_is_cpu(rerank_mode) &&
             !defaults::highfreq_rerank_mode_is_off(rerank_mode))) {
            fail_rabitq(
                "device_resident graph transport requires memory graph, coupled scan workers, "
                "CPU rerank/off, non-persistent execution, no GPU graph row cache, "
                "and no dual graph");
        }
        if (options.degree > static_cast<int>(kRabitqGpuGraphRowCacheDegree)) {
            fail_rabitq("device_resident graph transport currently supports degree <= 32");
        }
    }
    if (options.request_mode != "scan" && options.request_mode != "batch_ring") {
        fail_rabitq("native RabitQ search request_mode must be one of: scan, batch_ring");
    }
    if (options.request_batch_queries == 0 || options.request_batch_queries > 4096) {
        fail_rabitq("native RabitQ search request_batch_queries must be in [1, 4096]");
    }
    if (options.request_consumed_memory != "host" &&
        options.request_consumed_memory != "device") {
        fail_rabitq(
            "native RabitQ search request_consumed_memory must be one of: host, device");
    }
    if (options.request_mode == "batch_ring" &&
        (options.query_worker_mode != "coupled" ||
         options.graph_storage_mode != "memory" ||
         options.host_demand_batch_size != 1)) {
        fail_rabitq(
            "native RabitQ batch request ring requires coupled memory workers "
            "with host_demand_batch_size=1");
    }
    if (options.prefetch_hops < 0 || options.prefetch_hops > 3) {
        fail_rabitq("native RabitQ search currently supports prefetch-hops in [0, 3]");
    }
    if (options.quick_filter_bonus <= 0.0f || options.quick_filter_bonus > 1.0f) {
        fail_rabitq("native RabitQ search currently supports quick-filter-bonus in (0, 1]");
    }
    if (!defaults::is_valid_search_early_stop_mode(options.search_early_stop_mode)) {
        fail_rabitq("native RabitQ search search_early_stop_mode must be one of: fixed, off, adaptive");
    }
    if (options.query_worker_mode != "coupled" && options.query_worker_mode != "finite") {
        fail_rabitq("native RabitQ query_worker_mode must be coupled or finite");
    }
    if (options.finite_dispatch_mode != "batch" &&
        options.finite_dispatch_mode != "persistent" &&
        options.finite_dispatch_mode != "ready_ring") {
        fail_rabitq("native RabitQ finite_dispatch_mode must be batch, persistent, or ready_ring");
    }
    if (options.finite_partial_batch_timeout_us < 1 ||
        options.finite_partial_batch_timeout_us > 1000000) {
        fail_rabitq("native RabitQ finite_partial_batch_timeout_us must be in [1, 1000000]");
    }
    if (options.query_worker_mode == "finite") {
        if (options.graph_storage_mode != "memory") {
            fail_rabitq("native RabitQ finite query workers currently require memory graph storage");
        }
        if (options.finite_dispatch_mode != "batch") {
            fail_rabitq("native RabitQ finite dispatch currently supports only batch");
        }
        if (highfreq_gpu_exact_rerank_enabled(options)) {
            fail_rabitq("native RabitQ finite query workers currently require highfreq_rerank_mode=cpu or off");
        }
    }
}

GraphLoadConfig make_rabitq_load_config(const RabitqSearchOptions& options)
{
    GraphLoadConfig load_config;
    load_config.graph_path = options.graph_path;
    load_config.graph_storage_mode = options.graph_storage_mode;

    load_config.rerank_vector_storage_mode = options.rerank_vector_storage_mode;
    load_config.rerank_vector_artifact_path = options.rerank_vector_artifact_path;

    load_config.allow_float_vectors = true;

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
    load_config.vector_consumer = "native RabitQ search";
    load_config.supports_shared_label_routes =
        !rabitq_device_resident_graph_enabled(options) && options.gpu_graph_row_cache_mib == 0;
    load_config.supports_shared_query_vectors = load_config.supports_shared_label_routes;
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
    load_config.supported_query_route_providers = kQueryRouteProviderLabel;
    if (range_posting::parse_mode(options.range_posting_mode) !=
        range_posting::Mode::Off) {
        load_config.supported_query_route_providers |= kQueryRouteProviderRangePosting;
        load_config.range_posting_index_path = options.range_posting_index_path;
        load_config.range_access_planner.mode = options.range_posting_mode;
        load_config.range_access_planner.and_filter_mode =
            options.range_posting_and_filter_mode;
        load_config.range_access_planner.range_label_and_policy =
            options.range_label_and_policy;
        load_config.range_access_planner.scan_threshold =
            options.range_posting_scan_threshold;
        load_config.range_access_planner.drop_wide_hybrid_or =
            options.range_posting_drop_wide_hybrid_or;
        load_config.range_access_planner.range_graph_hybrid_or_enabled =
            options.range_graph_hybrid_or_enabled;
        load_config.range_access_planner.range_graph_hybrid_and_enabled =
            options.range_graph_hybrid_and_enabled;
    }
    const DenseVectorMetadata base_vector_metadata =
        inspect_dense_vector_file(options.base_vector_path, 1);
    const bool inline_gpu_rerank = highfreq_inline_gpu_exact_rerank_enabled(
        options, base_vector_metadata.scalar_type);
    load_config.highfreq_prefetch_with_vector_payload =
        inline_gpu_rerank &&
        base_vector_metadata.scalar_type == DenseVectorScalarType::Uint8;
    if (inline_gpu_rerank &&
        dense_vector_scalar_is_float(base_vector_metadata.scalar_type)) {
        // Static prefetch stores vectors in the fixed uint8 Node payload. FP32
        // inline scoring always uses the dynamically sized demand mailbox.
        load_config.prefetch_hops = 0;
    }
    load_config.load_data_labels_for_filtering =
        options.full_filter ||
        options.per_step_full_filter ||
        options.lowfreq_full_filter_mode != defaults::lowfreq_full_filter_mode_off ||
        highfreq_cpu_rerank_enabled(options);
    return load_config;
}

GraphLoadConfig make_rabitq_range_graph_load_config(
    const RabitqSearchOptions& options,
    const std::vector<uint8_t>& original_query_route_mask)
{
    GraphLoadConfig load_config = make_rabitq_load_config(options);
    load_config.graph_path = options.range_graph_path;
    load_config.graph_storage_mode = options.range_graph_storage_mode;

    load_config.data_label_spmat_path = options.range_graph_data_label_spmat_path;
    load_config.range_label_manifest_path = options.range_graph_label_manifest_path;
    load_config.seed_sidecar_path = options.range_graph_seed_sidecar_path;
    load_config.original_label_seed_sidecar_path =
        options.range_graph_original_label_seed_sidecar_path;
    load_config.metadata_sidecar_path = options.range_graph_metadata_sidecar_path;
    load_config.lowfreq_index_path = options.range_graph_lowfreq_index_path;
    load_config.group_path = options.range_graph_group_path;
    load_config.query_label_spmat_path.clear();
    load_config.query_manifest_jsonl_path.clear();
    load_config.query_manifest_families.clear();
    load_config.query_manifest_subtypes.clear();
    load_config.force_empty_query_labels = true;
    load_config.implicit_all_ones_row_filter_mask = true;
    load_config.defer_rerank_vector_store = true;
    // The secondary graph needs the primary runtime's exact label residual;
    // GPU-side static prefetch cannot perform that check.
    load_config.prefetch_hops = 0;
    load_config.original_query_route_mask = original_query_route_mask;
    load_config.supported_query_route_providers = kQueryRouteProviderLabel;
    load_config.range_posting_index_path.clear();
    load_config.range_access_planner = RangeAccessPlannerConfig{};
    load_config.load_data_labels_for_filtering = false;
    load_config.vector_consumer = "native RabitQ secondary range graph";
    return load_config;
}

RabitqSearchOptions make_rabitq_range_graph_search_options(
    const RabitqSearchOptions& options)
{
    RabitqSearchOptions range_options = options;
    range_options.rabitq_predicate_label_source_path = options.data_label_spmat_path;
    range_options.graph_path = options.range_graph_path;
    range_options.graph_storage_mode = options.range_graph_storage_mode;

    range_options.data_label_spmat_path = options.range_graph_data_label_spmat_path;
    range_options.range_label_manifest_path = options.range_graph_label_manifest_path;
    range_options.seed_sidecar_path = options.range_graph_seed_sidecar_path;
    range_options.original_label_seed_sidecar_path =
        options.range_graph_original_label_seed_sidecar_path;
    range_options.metadata_sidecar_path = options.range_graph_metadata_sidecar_path;
    range_options.lowfreq_index_path = options.range_graph_lowfreq_index_path;
    range_options.group_path = options.range_graph_group_path;
    range_options.query_label_spmat_path.clear();
    range_options.query_manifest_jsonl_path.clear();
    range_options.query_manifest_families.clear();
    range_options.query_manifest_subtypes.clear();
    range_options.full_filter = false;
    range_options.range_posting_mode = "off";
    range_options.range_posting_drop_wide_hybrid_or = false;
    range_options.range_graph_hybrid_or_enabled = false;
    range_options.range_graph_hybrid_and_enabled = false;
    range_options.prefetch_hops = 0;
    return range_options;
}

void make_identity_graph_filter_masks_permissive(GraphRuntime& runtime)
{
    if (runtime.group_layout.groups.size() != runtime.metadata.label_count ||
        runtime.group_layout.label_to_group.size() != runtime.metadata.label_count) {
        fail_rabitq("secondary range graph requires identity group layout");
    }
    for (uint32_t group = 0; group < runtime.metadata.label_count; ++group) {
        const auto& labels = runtime.group_layout.groups[group];
        if (labels.size() != 1 || labels.front() >= runtime.group_layout.label_to_group.size() ||
            runtime.group_layout.label_to_group[labels.front()] != group) {
            fail_rabitq("secondary range graph group layout is not identity");
        }
    }
    if (!runtime.metadata.implicit_all_ones_row_filter_mask ||
        !runtime.metadata.row_filter_mask.empty()) {
        fail_rabitq("secondary range graph must use implicit all-ones row filter mask");
    }

    // Identity groups already establish the route label. The mask cannot add
    // predicate information here; only the exact range residual and the
    // primary runtime's hybrid label residual reject a row during traversal.
    std::cout << "[rabitq-range-graph] identity_group_filter_bypass=1 rows="
              << runtime.graph.rows << " storage=implicit\n";
}

void build_rabitq_query_routes(const GraphRuntime& runtime,
                               std::vector<uint32_t>& high_query_ids,
                               std::vector<uint32_t>& low_query_ids);

bool has_rabitq_highfreq_routes(const GraphRuntime& runtime)
{
    return count_rabitq_highfreq_graph_routes(runtime) != 0;
}

void validate_rabitq_runtime(const RabitqHostIndex& index,
                             const GraphRuntime& runtime,
                             const RabitqSearchOptions& options)
{

    if (options.use_lut && options.use_int8_score) {
        fail_rabitq("RabitQ float LUT and int8 score modes are mutually exclusive");
    }
    if (options.use_int8_score && index.codec.bits_per_dim > 2u) {
        fail_rabitq("RabitQ int8 score mode currently requires a b1 or b2 artifact");
    }
    if (options.use_int8_score && options.query_worker_mode != "coupled") {
        fail_rabitq("RabitQ int8 score mode currently requires coupled query workers");
    }
    if (options.use_int8_score && options.latency_persistent_queue) {
        fail_rabitq("RabitQ int8 score mode is not yet supported by persistent latency execution");
    }
    if (index.header.num_points != runtime.base_vectors.rows) {
        fail_rabitq("artifact num_points does not match base vector rows");
    }
    if (runtime.base_vectors.rows == 0 || runtime.query_vectors.rows == 0) {
        fail_rabitq("loaded empty dataset");
    }
    if (index.header.dim != runtime.query_vectors.cols ||
        index.header.dim != runtime.base_vectors.cols ||
        index.header.dim != static_cast<uint32_t>(options.dim)) {
        fail_rabitq("RabitQ artifact/base/query dimension mismatch");
    }

    const bool has_highfreq_routes = has_rabitq_highfreq_routes(runtime);
    const bool use_inline_gpu_exact_rerank =
        highfreq_inline_gpu_exact_rerank_enabled(options, runtime.vector_scalar_type);
    const bool use_staged_float_gpu_exact_rerank =
        highfreq_staged_float_gpu_exact_rerank_enabled(
            options, runtime.vector_scalar_type);
    if (has_highfreq_routes &&
        defaults::highfreq_rerank_mode_is_gpu_post(
            effective_highfreq_rerank_mode(options)) &&
        !dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        fail_rabitq(
            "native RabitQ highfreq_rerank_mode=gpu_post currently requires float vectors");
    }
    if (range_posting::parse_mode(options.range_posting_mode) != range_posting::Mode::Off) {
        if (runtime.base_range_values.empty() || runtime.base_range_count == 0 ||
            runtime.original_query_ranges.rows != runtime.original_query_count) {
            fail_rabitq("range posting requires aligned base and original-query range metadata");
        }
    }

    if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        if (runtime.original_query_float_vectors.data.empty()) {
            fail_rabitq("float RabitQ search requires loaded original query float vectors");
        }
        if (

            !runtime.rerank_vector_store &&
            runtime.base_float_vectors.data.empty()) {
            fail_rabitq("float RabitQ search requires loaded DenseFloatMatrix base vectors for resident vector storage");
        }
        if (has_highfreq_routes && use_staged_float_gpu_exact_rerank) {
            validate_float_gpu_post_vector_storage(runtime);
        }
        if (has_highfreq_routes && use_inline_gpu_exact_rerank &&
            (
             runtime.rerank_vector_store != nullptr ||
             runtime.base_float_vectors.data.empty())) {
            fail_rabitq(
                "float RabitQ inline GPU rerank requires the default resident FP32 base matrix");
        }
        if (has_highfreq_routes && use_inline_gpu_exact_rerank &&
            (options.query_worker_mode != "coupled" ||
             options.latency_persistent_queue)) {
            fail_rabitq(
                "float RabitQ inline GPU rerank currently requires a resident memory graph, "
                "coupled query workers, and non-persistent execution");
        }
    } else if (runtime.vector_scalar_type != DenseVectorScalarType::Uint8) {
        fail_rabitq("unsupported dense vector scalar type for native RabitQ search");
    }

    if (has_highfreq_routes) {
        if (!highfreq_neighbor_mailbox_fits(static_cast<uint32_t>(options.degree))) {
            fail_rabitq("native RabitQ high-frequency graph mailbox cannot carry degree=" +
                        std::to_string(options.degree) +
                        " because neighbor payloads need degree+1 uint32 words");
        }
        if (use_inline_gpu_exact_rerank) {
            if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8 &&
                static_cast<uint32_t>(options.dim) > kGraphMaxDim) {
                fail_rabitq("native RabitQ high-frequency GPU exact rerank supports dim <= " +
                            std::to_string(kGraphMaxDim) +
                            " for uint8 because Node/query shared buffers are fixed; use CPU/off rerank "
                            "or low-frequency-only routes for larger dimensions");
            }
        }
    }
}

RabitqSearchOptions resolve_rabitq_artifact_defaults(
    const RabitqHostIndex& index,
    const RabitqSearchOptions& options)
{
    RabitqSearchOptions resolved = options;
    if (resolved.use_int8_score_auto && index.codec.bits_per_dim > 2u) {
        resolved.use_int8_score = false;
    }
    return resolved;
}

void build_rabitq_query_routes(const GraphRuntime& runtime,
                               std::vector<uint32_t>& high_query_ids,
                               std::vector<uint32_t>& low_query_ids)
{
    high_query_ids.clear();
    low_query_ids.clear();
    high_query_ids.reserve(runtime.query_vectors.rows);
    low_query_ids.reserve(runtime.query_vectors.rows);
    for (uint32_t query_idx = 0; query_idx < runtime.query_vectors.rows; ++query_idx) {
        if (query_idx < runtime.route_enabled.size() &&
            runtime.route_enabled[query_idx] == 0) {
            continue;
        }
        if (query_idx >= runtime.query_group_ids.size() ||
            runtime.query_group_ids[query_idx] == UINT32_MAX) {
            continue;
        }
        if (query_idx < runtime.query_is_lowfreq.size() &&
            runtime.query_is_lowfreq[query_idx] != 0) {
            low_query_ids.push_back(query_idx);
        } else if (rabitq_route_is_highfreq_graph(runtime, query_idx)) {
            high_query_ids.push_back(query_idx);
        }
    }
    std::stable_sort(high_query_ids.begin(), high_query_ids.end(), [&runtime](uint32_t lhs, uint32_t rhs) {
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

// Created before result futures and destroyed after every consumer has joined.
// The dual-graph owner binds both contexts to the same original-vector domain.

class RabitqIncrementalScope {
    RabitqGraphSearchContext* primary_ = nullptr;
    RabitqGraphSearchContext* secondary_ = nullptr;
    go_fann::incremental_rerank::Stats stats_{};
    size_t cap_ = 0;
public:
    RabitqIncrementalScope(const GraphRuntime& runtime, RabitqGraphSearchContext& graph,
        const RabitqSearchOptions& options, const GraphRuntime* secondary = nullptr,
        RabitqGraphSearchContext* secondary_graph = nullptr, const RabitqSearchOptions* secondary_options = nullptr) {
        if (!options.incremental_rerank_shortlist || graph.incremental_rerank) return;
        if (secondary && (!secondary_graph || !secondary_options ||
            secondary_options->incremental_rerank_shortlist != options.incremental_rerank_shortlist ||
            secondary_options->topk != options.topk ||
            secondary_options->use_int8_score != options.use_int8_score ||
            secondary_options->use_lut != options.use_lut ||
            secondary->original_query_count != runtime.original_query_count ||
            secondary->vector_scalar_type != runtime.vector_scalar_type ||
            secondary->base_vector_metadata.path != runtime.base_vector_metadata.path ||
            secondary->query_vector_metadata.path != runtime.query_vector_metadata.path ||
            secondary->base_vectors.rows != runtime.base_vectors.rows ||
            secondary->query_vectors.cols != runtime.query_vectors.cols ||
            secondary->rerank_vector_store != runtime.rerank_vector_store ||
            secondary_options->artifact_path != options.artifact_path))
            fail_rabitq("incremental rerank requires a shared primary/secondary query, vector and RabitQ score domain");
        auto state = std::make_shared<NativeIncrementalRerank>(runtime,
            options.incremental_rerank_shortlist, options.topk);
        graph.incremental_rerank = state;
        if (secondary_graph) secondary_graph->incremental_rerank = state;
        primary_ = &graph;
        secondary_ = secondary_graph;
        cap_ = options.incremental_rerank_shortlist;
    }
    void publish(std::vector<std::vector<idx_t>>& ids, std::vector<std::vector<float>>& distances) {
        if (primary_) primary_->incremental_rerank->publish(ids, distances);
    }
    void finish() {
        if (!primary_ || !primary_->incremental_rerank) return;
        stats_ = primary_->incremental_rerank->stats();
        primary_->incremental_rerank.reset();
        if (secondary_) secondary_->incremental_rerank.reset();
    }
    ~RabitqIncrementalScope() {
        finish();
        if (primary_) std::printf("[incremental-rerank] L=%zu batches=%llu offered=%llu repeated=%llu issued=%llu completed=%llu coalesced=%llu score_batches=%llu policy=online_topL_independent_exact_topK\n",
            cap_, (unsigned long long)stats_.batches, (unsigned long long)stats_.offered,
            (unsigned long long)stats_.repeated, (unsigned long long)stats_.issued,
            (unsigned long long)stats_.completed, (unsigned long long)stats_.coalesced,
            (unsigned long long)stats_.score_batches);
    }
};

class RabitqExactMemoScope {
public:
    RabitqExactMemoScope(const GraphRuntime& runtime,
                        RabitqGraphSearchContext& graph,
                        const RabitqRangePostingSearchContext& posting,
                        bool reuse_enabled,
                        const GraphRuntime* secondary = nullptr,
                        RabitqGraphSearchContext* secondary_graph = nullptr,
                        const RabitqRangePostingSearchContext* secondary_posting = nullptr) {
        if (graph.exact_distance_memo) return; // enclosed by the dual-graph iteration
        report_ = true;
        requested_ = reuse_enabled;
        if (!reuse_enabled) return; // no provider scan, validation, or allocation

        if (secondary) {
            if (!secondary_graph || !secondary_posting ||
                secondary->original_query_count != runtime.original_query_count ||
                secondary->vector_scalar_type != runtime.vector_scalar_type ||
                secondary->base_vector_metadata.path != runtime.base_vector_metadata.path ||
                secondary->query_vector_metadata.path != runtime.query_vector_metadata.path ||
                secondary->base_vectors.rows != runtime.base_vectors.rows ||
                secondary->query_vectors.cols != runtime.query_vectors.cols ||
                secondary->rerank_vector_store != runtime.rerank_vector_store)
                fail_rabitq("exact distance reuse requires the same primary/secondary vector domain");
        }
        std::vector<uint8_t> providers(runtime.original_query_count, 0);
        auto count = [&](const GraphRuntime& source, const RabitqRangePostingSearchContext& ranges) {
            auto add = [&](uint32_t route) {
                const uint32_t query = exact_original_query(source, route);
                if (query >= providers.size()) fail_rabitq("exact distance memo original query mapping");
                providers[query] = std::min<uint8_t>(2, providers[query] + 1);
            };
            for (uint32_t route = 0; route < source.query_vectors.rows; ++route) {
                if ((route < source.route_enabled.size() && !source.route_enabled[route]) ||
                    route >= source.query_group_ids.size() || source.query_group_ids[route] == UINT32_MAX) continue;
                if (ranges.plan.mode == range_posting::Mode::Select &&
                    route < ranges.plan.replaced_routes.size() && ranges.plan.replaced_routes[route]) continue;
                if ((route < source.query_is_lowfreq.size() && source.query_is_lowfreq[route]) ||
                    rabitq_route_is_highfreq_graph(source, route)) add(route);
            }
            for (uint32_t route : ranges.plan.representative_route_ids) add(route);
        };
        count(runtime, posting);
        if (secondary) count(*secondary, *secondary_posting);
        for (auto& value : providers) value = value > 1;
        const void* base = runtime.rerank_vector_store ? static_cast<const void*>(runtime.rerank_vector_store.get())
            : dense_vector_scalar_is_float(runtime.vector_scalar_type)
                ? static_cast<const void*>(runtime.base_float_vectors.data.data())
                : static_cast<const void*>(runtime.base_vectors.data.data());
        const void* query = dense_vector_scalar_is_float(runtime.vector_scalar_type)
            ? static_cast<const void*>(runtime.original_query_float_vectors.data.data())
            : static_cast<const void*>(runtime.original_query_vectors.data.data());
        // Original row storage may be absent for legacy non-routed singletons.
        if (!query) query = dense_vector_scalar_is_float(runtime.vector_scalar_type)
            ? static_cast<const void*>(runtime.query_float_vectors.data.data())
            : static_cast<const void*>(runtime.query_vectors.data.data());
        if (!base || !query) return;
        memo_ = std::make_shared<go_fann::ExactDistanceMemo>(
            go_fann::ExactDistanceMemo::Domain{runtime.original_query_count,
                runtime.base_vectors.rows, runtime.query_vectors.cols,
                static_cast<uint32_t>(runtime.vector_scalar_type), base, query}, std::move(providers));
        enabled_ = true;
        primary_ = &graph;
        secondary_ = secondary_graph;
        primary_->exact_distance_memo = memo_;
        if (secondary_) secondary_->exact_distance_memo = memo_;
    }
    void finish() {
        if (!memo_) return;
        stats_ = memo_->stats();
        primary_->exact_distance_memo.reset();
        if (secondary_) secondary_->exact_distance_memo.reset();
        memo_.reset(); // include table destruction in execution wall time
    }
    ~RabitqExactMemoScope() {
        finish();
        if (!report_) return;
        std::printf("[exact-distance-policy] backend=rabitq requested=%d enabled=%d\n",
            requested_ ? 1 : 0, enabled_ ? 1 : 0);
        if (!enabled_) return;
        std::printf("[exact-distance-memo] backend=rabitq computations=%llu hits=%llu entries=%llu adaptive_bypassed_queries=%llu scope=iteration_cached_admissions\n",
            static_cast<unsigned long long>(stats_.computations), static_cast<unsigned long long>(stats_.hits),
            static_cast<unsigned long long>(stats_.entries),
            static_cast<unsigned long long>(stats_.adaptive_bypassed_queries));
    }
private:
    bool report_ = false, requested_ = false, enabled_ = false;
    go_fann::ExactDistanceMemo::Stats stats_{};
    std::shared_ptr<go_fann::ExactDistanceMemo> memo_;
    RabitqGraphSearchContext* primary_ = nullptr;
    RabitqGraphSearchContext* secondary_ = nullptr;
};

RabitqIterationResult execute_rabitq_iteration(const RabitqHostIndex& index,
                                               const GraphRuntime& runtime,
                                               const RabitqSearchOptions& options,
                                               const RabitqImmutableDeviceState& immutable_state,
                                               const RabitqPreparedQueryState& prepared_state,
                                               RabitqGraphSearchContext& graph_ctx,
                                               RabitqLowfreqSearchContext& lowfreq_ctx,
                                               const range_posting::Index* posting_index,
                                               RabitqRangePostingSearchContext& range_posting_ctx,
                                               bool print_summary,
                                               bool defer_cpu_rerank = false,
                                               const std::function<void()>&
                                                   gpu_phase_complete =
                                                       std::function<void()>())
{
    const auto pre_qps_start = std::chrono::steady_clock::now();
    perf_trace::Scope trace_scope("provider_iteration", &runtime);
    const auto profile_prepare_start = std::chrono::steady_clock::now();
    RabitqSearchOptions single_run_options = options;
    single_run_options.warmup_runs = 0;
    single_run_options.timed_runs = 1;
    if (graph_ctx.cpu_worker_budget == nullptr) {
        graph_ctx.cpu_worker_budget = std::make_shared<RabitqCpuWorkerBudget>(
            static_cast<size_t>(std::max(1, options.highfreq_rerank_threads)));
    }
    prepare_cpu_rerank_workers(graph_ctx, options);

    const auto route_partition_start = std::chrono::steady_clock::now();
    std::vector<uint32_t> high_query_ids;
    std::vector<uint32_t> low_query_ids;
    build_rabitq_query_routes(runtime, high_query_ids, low_query_ids);
    if (range_posting_ctx.plan.mode == range_posting::Mode::Select) {
        const auto selected_route = [&range_posting_ctx](uint32_t route_idx) {
            return route_idx < range_posting_ctx.plan.replaced_routes.size() &&
                range_posting_ctx.plan.replaced_routes[route_idx] != 0;
        };
        high_query_ids.erase(std::remove_if(high_query_ids.begin(),
                                            high_query_ids.end(),
                                            selected_route),
                             high_query_ids.end());
        low_query_ids.erase(std::remove_if(low_query_ids.begin(),
                                           low_query_ids.end(),
                                           selected_route),
                            low_query_ids.end());
    }
    const auto route_partition_end = std::chrono::steady_clock::now();

    const auto result_init_start = route_partition_end;
    std::vector<std::vector<idx_t>> route_results(runtime.query_vectors.rows);
    std::vector<std::vector<float>> route_distances(runtime.query_vectors.rows);
    std::vector<int> route_steps(runtime.query_vectors.rows, 0);
    const auto result_init_end = std::chrono::steady_clock::now();

    RabitqBatchStats highfreq_stats;
    double mixed_qps_sum = 0.0;
    size_t mixed_run_count = 0;
    int64_t timed_total_steps_sum = 0;
    int64_t timed_highfreq_steps_sum = 0;
    int64_t timed_lowfreq_steps_sum = 0;
    GraphRouteSplitStats final_split_stats;

    const auto iteration_wall_start = std::chrono::steady_clock::now();
    RabitqIncrementalScope incremental_scope(runtime, graph_ctx, options);
    RabitqExactMemoScope memo_scope(runtime, graph_ctx, range_posting_ctx, options.rerank_distance_reuse);
    perf_trace::span("iteration_prepare_outside_search_window", &runtime,
                     profile_prepare_start, iteration_wall_start);
    perf_trace::value("highfreq_routes", &runtime, high_query_ids.size());
    perf_trace::value("lowfreq_routes", &runtime, low_query_ids.size());
    std::vector<uint32_t> pending_high_query_ids;
    std::vector<std::vector<idx_t>> pending_high_results;
    std::vector<std::vector<float>> pending_high_distances;
    std::vector<int> pending_high_steps;
    std::vector<uint32_t> pending_low_query_ids;
    std::vector<std::vector<idx_t>> pending_low_results;
    std::vector<std::vector<float>> pending_low_distances;
    std::vector<int> pending_low_steps;
    std::shared_ptr<RabitqCpuPhaseTiming> range_posting_cpu_timing;
    std::vector<std::vector<idx_t>> posting_results;
    std::vector<int> posting_steps;
    std::vector<std::vector<float>> posting_distances;
    // Destroy/join futures before any captured result containers, including on
    // exception paths. No context or result buffer may be reused while rerank runs.
    std::future<void> highfreq_rerank_future;
    std::future<void> lowfreq_rerank_future;
    std::future<void> range_posting_rerank_future;
    RabitqRangePostingRunStats range_posting_stats;
    const bool defer_provider_cpu_rerank =
        defer_cpu_rerank || options.posting_schedule != "legacy" ||
        cpu_rerank_schedule::no_overlap(options.cpu_rerank_schedule);
    auto gpu_provider_complete_at = iteration_wall_start;

    const auto run_graph = [&] {
        if (!high_query_ids.empty()) {
            if (!graph_ctx.initialized) {
                fail_rabitq("RabitQ graph context is required for high-frequency routes");
            }
            const bool async_highfreq_cpu_rerank =
                highfreq_cpu_rerank_enabled(single_run_options) &&
                !highfreq_cpu_streaming_enabled(single_run_options);
            const bool deferred_streaming_rerank =
                defer_provider_cpu_rerank && highfreq_cpu_streaming_enabled(single_run_options);
            std::vector<std::vector<idx_t>> high_results;
            std::vector<int> high_steps;
            std::vector<std::vector<float>> high_distances;
            std::vector<std::vector<idx_t>>& graph_results = deferred_streaming_rerank
                ? pending_high_results
                : high_results;
            std::vector<int>& graph_steps = deferred_streaming_rerank
                ? pending_high_steps
                : high_steps;
            std::vector<std::vector<float>>& graph_distances = deferred_streaming_rerank
                ? pending_high_distances
                : high_distances;
            if (deferred_streaming_rerank) {
                pending_high_query_ids = high_query_ids;
            }
            double highfreq_wall_ms = 0.0;
            execute_rabitq_graph_batch(graph_ctx,
                                       index,
                                       runtime,
                                       single_run_options,
                                       immutable_state,
                                       prepared_state,
                                       high_query_ids,
                                       "Timed run 1/1",
                                       true,
                                       graph_results,
                                       graph_steps,
                                       graph_distances,
                                       &highfreq_stats,
                                       print_summary,
                                       &highfreq_wall_ms,
                                       !async_highfreq_cpu_rerank,
                                       deferred_streaming_rerank
                                           ? &highfreq_rerank_future
                                           : nullptr);
            if (!async_highfreq_cpu_rerank && !deferred_streaming_rerank &&
                options.topk < kTopk) {
                for (auto& row : high_results) {
                    row.resize(static_cast<size_t>(options.topk));
                }
                for (auto& row : high_distances) {
                    row.resize(static_cast<size_t>(options.topk));
                }
            }
            if (deferred_streaming_rerank) {
                if (!highfreq_rerank_future.valid()) {
                    fail_rabitq("deferred high-frequency streaming rerank did not return a future");
                }
            } else if (async_highfreq_cpu_rerank) {
                pending_high_query_ids = high_query_ids;
                pending_high_results = std::move(high_results);
                pending_high_distances = std::move(high_distances);
                pending_high_steps = std::move(high_steps);
                highfreq_rerank_future = std::async(cpu_rerank_schedule::host_finalize_launch(single_run_options.cpu_rerank_schedule),
                                                    [&runtime,
                                                     &single_run_options,
                                                     &pending_high_query_ids,
                                                     &pending_high_results,
                                                     &pending_high_distances,
                                                     cpu_worker_budget =
                                                         graph_ctx.cpu_worker_budget,
                                                     cpu_rerank_workers =
                                                         graph_ctx.cpu_rerank_workers,
                                                     distance_memo = graph_ctx.exact_distance_memo,
                                                     incremental = graph_ctx.incremental_rerank,
                                                     shortlist_predicate_verified = graph_ctx.shortlist_predicate_verified] {
                                                        finalize_highfreq_rabitq_results({distance_memo.get(), nullptr, incremental.get()},
                                                            single_run_options,
                                                            runtime,
                                                            pending_high_query_ids,
                                                            pending_high_results,
                                                            &pending_high_distances,
                                                            cpu_worker_budget.get(),
                                                            cpu_rerank_workers.get(),
                                                            shortlist_predicate_verified);
                                                    });
            } else {
                merge_subset_results(high_query_ids,
                                     route_results,
                                     route_steps,
                                     route_distances,
                                     high_results,
                                     high_steps,
                                     high_distances);
            }
        }
    };

    const auto run_posting = [&] {
        if (!low_query_ids.empty()) {
            pending_low_query_ids = low_query_ids;
            (void)run_lowfreq_rabitq_search_with_context(index,
                                                         runtime,
                                                         single_run_options,
                                                         pending_low_query_ids,
                                                         immutable_state,
                                                         prepared_state,
                                                         graph_ctx,
                                                         lowfreq_ctx,
                                                         pending_low_results,
                                                         pending_low_steps,
                                                         pending_low_distances,
                                                         &lowfreq_rerank_future);
            if (!lowfreq_rerank_future.valid()) {
                merge_subset_results(pending_low_query_ids,
                                     route_results,
                                     route_steps,
                                     route_distances,
                                     pending_low_results,
                                     pending_low_steps,
                                     pending_low_distances);
                pending_low_query_ids.clear();
            }
        }

        if (range_posting_ctx.initialized) {
            if (posting_index == nullptr) {
                fail_rabitq("initialized range posting context has no host index");
            }
            // Host-fed range posting reuses graph completion mailboxes. In hybrid
            // mode its route IDs may overlap graph routes still consumed by rerank.
            if (!range_posting_ctx.plan.stream_plan_positions.empty() &&
                highfreq_rerank_future.valid() &&
                !cpu_rerank_schedule::no_overlap(options.cpu_rerank_schedule)) {
                highfreq_rerank_future.wait();
            }
            range_posting_stats = run_range_posting_rabitq_search(index,
                                                                   runtime,
                                                                   single_run_options,
                                                                   immutable_state,
                                                                   prepared_state,
                                                                   graph_ctx,
                                                                   *posting_index,
                                                                   range_posting_ctx,
                                                                   posting_results,
                                                                   posting_steps,
                                                                   posting_distances,
                                                                   defer_provider_cpu_rerank
                                                                       ? &range_posting_rerank_future
                                                                       : nullptr,
                                                                   defer_provider_cpu_rerank
                                                                       ? &range_posting_cpu_timing
                                                                       : nullptr);
        }
    };

    if (options.posting_schedule == "posting_first") {
        run_posting();
        run_graph();
    } else {
        run_graph();
        run_posting();
    }

    gpu_provider_complete_at = std::chrono::steady_clock::now();
    if (gpu_phase_complete) {
        gpu_phase_complete();
    }

    if (highfreq_rerank_future.valid()) {
        highfreq_rerank_future.get();
        for (auto& row : pending_high_results) {
            if (row.size() > static_cast<size_t>(options.topk)) {
                row.resize(static_cast<size_t>(options.topk));
            }
        }
        for (auto& row : pending_high_distances) {
            if (row.size() > static_cast<size_t>(options.topk)) {
                row.resize(static_cast<size_t>(options.topk));
            }
        }
        merge_subset_results(pending_high_query_ids,
                             route_results,
                             route_steps,
                             route_distances,
                             pending_high_results,
                             pending_high_steps,
                             pending_high_distances);
    }
    if (lowfreq_rerank_future.valid()) {
        lowfreq_rerank_future.get();
        merge_subset_results(pending_low_query_ids,
                             route_results,
                             route_steps,
                             route_distances,
                             pending_low_results,
                             pending_low_steps,
                             pending_low_distances);
    }
    if (range_posting_rerank_future.valid()) {
        range_posting_rerank_future.get();
        if (range_posting_cpu_timing == nullptr) {
            fail_rabitq("deferred range-posting rerank completed without timing state");
        }
        range_posting_stats.rerank_queue_wait_ms =
            range_posting_cpu_timing->queue_wait_ms;
        range_posting_stats.rerank_wall_ms = range_posting_cpu_timing->active_ms;
        range_posting_stats.total_wall_ms += range_posting_cpu_timing->total_ms;
    }
    for (size_t posting_query = 0;
         posting_query < range_posting_ctx.plan.representative_route_ids.size();
         ++posting_query) {
        const uint32_t route_idx =
            range_posting_ctx.plan.representative_route_ids[posting_query];
        range_posting::merge_exact_result_row(route_results[route_idx],
                                              route_distances[route_idx],
                                              posting_results[posting_query],
                                              posting_distances[posting_query],
                                              static_cast<uint32_t>(options.topk));
        route_steps[route_idx] += posting_steps[posting_query];
    }
    if (print_summary && range_posting_ctx.initialized) {
        std::cout << "[range-posting] mode="
                  << range_posting::mode_name(range_posting_ctx.plan.mode)
                  << " selected_queries="
                  << range_posting_ctx.plan.representative_route_ids.size()
                  << " range_graph_or_handoffs="
                  << range_posting_ctx.plan.dropped_wide_hybrid_or_routes
                  << " range_graph_or_rows="
                  << range_posting_ctx.plan.dropped_wide_hybrid_or_rows
                  << " selected_range_graph_and_queries="
                  << range_posting_ctx.plan.selected_range_graph_and_queries
                  << " selected_range_graph_and_rows="
                  << range_posting_ctx.plan.selected_range_graph_and_rows
                  << " scanned_rows=" << range_posting_stats.scanned_rows
                  << " exact_pass_rows=" << (range_posting_stats.exact_pass_rows_available
                      ? std::to_string(range_posting_stats.exact_pass_rows) : "unavailable")
                  << " exact_pass_rows_available=" << range_posting_stats.exact_pass_rows_available
                  << " shortlist_rows=" << range_posting_stats.shortlist_rows
                  << " postfilter_shortlist_rows="
                  << range_posting_stats.postfilter_shortlist_rows
                  << " postfilter_residual_pass_rows="
                  << range_posting_stats.postfilter_residual_pass_rows
                  << " exact_rerank_rows="
                  << range_posting_stats.exact_rerank_rows
                  << " resident_routes=" << range_posting_stats.resident_routes
                  << " streamed_routes=" << range_posting_stats.streamed_routes
                  << " stream_chunks=" << range_posting_stats.stream_chunks
                  << " ordinary_shortlist_k=" << range_posting_ctx.shortlist_k
                  << " ordinary_shortlist_capacity="
                  << range_posting_stats.ordinary_shortlist_capacity
                  << " ordinary_active_blocks_per_sm="
                  << range_posting_stats.ordinary_active_blocks_per_sm
                  << " ordinary_shared_bytes="
                  << range_posting_stats.ordinary_shared_bytes
                  << " postfilter_shortlist_k="
                  << range_posting_ctx.postfilter_shortlist_k
                  << " postfilter_shortlist_capacity="
                  << range_posting_stats.postfilter_shortlist_capacity
                  << " postfilter_active_blocks_per_sm="
                  << range_posting_stats.postfilter_active_blocks_per_sm
                  << " postfilter_shared_bytes="
                  << range_posting_stats.postfilter_shared_bytes
                  << " scorer="
                  << (single_run_options.use_int8_score ? "dp4a_int8" : "fp32_bit")
                  << " kernel_wall_ms=" << range_posting_stats.kernel_wall_ms
                  << " shortlist_d2h_ms=" << range_posting_stats.shortlist_d2h_ms
                  << " rerank_queue_wait_ms="
                  << range_posting_stats.rerank_queue_wait_ms
                  << " rerank_wall_ms=" << range_posting_stats.rerank_wall_ms
                  << " total_wall_ms=" << range_posting_stats.total_wall_ms
                  << "\n";
    }

    const size_t original_query_count = runtime.original_query_count;
    int64_t run_total_steps = 0;
    for (int step : route_steps) {
        run_total_steps += static_cast<int64_t>(step);
    }
    timed_total_steps_sum += run_total_steps;
    final_split_stats = collect_graph_route_split_stats(runtime, route_steps);
    timed_highfreq_steps_sum += final_split_stats.highfreq_total_steps;
    timed_lowfreq_steps_sum += final_split_stats.lowfreq_total_steps;

    std::vector<std::vector<idx_t>> merged_results;
    std::vector<std::vector<float>> merged_distances;
    std::vector<int> merged_steps;
    const auto route_merge_start = std::chrono::steady_clock::now();
    const lowfreq_merge::FinalMergeStrategy route_merge_strategy =
        merge_route_results_to_original(runtime,
                                        route_results,
                                        route_distances,
                                        route_steps,
                                        static_cast<uint32_t>(options.topk),
                                        merged_results,
                                        merged_steps,
                                        &merged_distances);
    incremental_scope.publish(merged_results, merged_distances);
    const auto route_merge_end = std::chrono::steady_clock::now();

    // Count final route-to-original top-k merge in execution QPS. Query setup
    // and route construction are reported separately.
    incremental_scope.finish();
    memo_scope.finish();
    const auto iteration_wall_end = std::chrono::steady_clock::now();
    perf_trace::span("provider_search_window", &runtime, iteration_wall_start, iteration_wall_end);
    const double run_wall_ms =
        std::chrono::duration<double, std::milli>(iteration_wall_end - iteration_wall_start).count();
    if (original_query_count > 0 && run_wall_ms > 0.0) {
        mixed_qps_sum += static_cast<double>(original_query_count) / (run_wall_ms / 1000.0);
        mixed_run_count += 1;
    }

    RabitqIterationResult result;
    result.results = std::move(merged_results);
    result.distances = std::move(merged_distances);
    result.step_nums = std::move(merged_steps);
    result.kernel_ms = highfreq_stats.run_count > 0
        ? (highfreq_stats.kernel_ms / static_cast<double>(highfreq_stats.run_count))
        : 0.0;
    result.qps = mixed_run_count > 0
        ? mixed_qps_sum / static_cast<double>(mixed_run_count)
        : 0.0;
    result.wall_ms = run_wall_ms;
    result.route_merge_wall_ms = elapsed_ms(route_merge_start, route_merge_end);
    result.pre_qps_route_partition_ms = elapsed_ms(route_partition_start, route_partition_end);
    result.pre_qps_result_init_ms = elapsed_ms(result_init_start, result_init_end);
    result.pre_qps_total_ms = elapsed_ms(pre_qps_start, iteration_wall_start);
    result.qps_start = iteration_wall_start;
    result.route_merge_strategy = route_merge_strategy;
    result.query_count = original_query_count;
    result.total_steps = timed_total_steps_sum;
    result.highfreq_route_count = final_split_stats.highfreq_route_count;
    result.lowfreq_route_count = final_split_stats.lowfreq_route_count;
    result.provider_route_count = final_split_stats.provider_route_count;
    result.unroutable_route_count = final_split_stats.unroutable_route_count;
    result.highfreq_total_steps = timed_highfreq_steps_sum;
    result.lowfreq_total_steps = timed_lowfreq_steps_sum;
    result.route_flags = build_graph_original_route_flags(runtime);
    const auto total_requests = highfreq_stats.host_filter_stats.requests_served;
    result.pass_ratio = total_requests > 0
        ? static_cast<float>(highfreq_stats.host_filter_stats.requests_passed) / static_cast<float>(total_requests)
        : 0.0f;
    result.workers_thread = highfreq_stats.host_filter_stats.workers_thread;
    result.host_filter_stats = highfreq_stats.host_filter_stats;
    result.gpu_provider_wall_ms =
        elapsed_ms(iteration_wall_start, gpu_provider_complete_at);
    result.cpu_tail_wall_ms = elapsed_ms(gpu_provider_complete_at, iteration_wall_end);
    result.range_posting_rerank_wall_ms = range_posting_stats.rerank_wall_ms;
    return result;
}

RabitqIterationResult execute_rabitq_dual_graph_iteration(
    const RabitqHostIndex& index,
    const GraphRuntime& primary_runtime,
    const RabitqSearchOptions& primary_options,
    const RabitqImmutableDeviceState& immutable_state,
    const RabitqPreparedQueryState& primary_prepared_state,
    RabitqGraphSearchContext& primary_graph_ctx,
    RabitqLowfreqSearchContext& primary_lowfreq_ctx,
    const range_posting::Index* posting_index,
    RabitqRangePostingSearchContext& primary_range_posting_ctx,
    const GraphRuntime* secondary_runtime,
    const RabitqSearchOptions* secondary_options,
    const RabitqPreparedQueryState* secondary_prepared_state,
    RabitqGraphSearchContext* secondary_graph_ctx,
    RabitqLowfreqSearchContext* secondary_lowfreq_ctx,
    RabitqRangePostingSearchContext* secondary_range_posting_ctx,
    bool print_summary)
{
    perf_trace::Scope trace_scope("dual_graph_iteration", &primary_runtime);
    const auto iteration_start = std::chrono::steady_clock::now();
    const bool has_secondary = secondary_runtime != nullptr;
    if (has_secondary &&
        (secondary_options == nullptr || secondary_prepared_state == nullptr ||
         secondary_graph_ctx == nullptr || secondary_lowfreq_ctx == nullptr ||
         secondary_range_posting_ctx == nullptr)) {
        fail_rabitq("secondary graph execution context is incomplete");
    }
    if (primary_graph_ctx.cpu_worker_budget == nullptr) {
        primary_graph_ctx.cpu_worker_budget = std::make_shared<RabitqCpuWorkerBudget>(
            static_cast<size_t>(std::max(1, primary_options.highfreq_rerank_threads)));
    }
    prepare_cpu_rerank_workers(primary_graph_ctx, primary_options);
    if (has_secondary) {
        secondary_graph_ctx->cpu_worker_budget = primary_graph_ctx.cpu_worker_budget;
        secondary_graph_ctx->cpu_rerank_workers = primary_graph_ctx.cpu_rerank_workers;
    }
    const bool serial_cpu_rerank = cpu_rerank_schedule::no_overlap(primary_options.cpu_rerank_schedule);
    if (has_secondary && primary_options.cpu_rerank_schedule != secondary_options->cpu_rerank_schedule)
        fail_rabitq("primary and secondary CPU rerank schedules differ");
    const bool overlap_cpu_gpu =
        has_secondary && (primary_options.dual_graph_cpu_gpu_overlap || serial_cpu_rerank);
    if (overlap_cpu_gpu && !serial_cpu_rerank &&
        secondary_options->dual_graph_cpu_gpu_overlap !=
            primary_options.dual_graph_cpu_gpu_overlap) {
        fail_rabitq("primary and secondary dual-graph overlap settings differ");
    }

    if (has_secondary && primary_options.rerank_distance_reuse != secondary_options->rerank_distance_reuse)
        fail_rabitq("primary and secondary exact distance reuse settings differ");
    RabitqIncrementalScope incremental_scope(primary_runtime, primary_graph_ctx, primary_options,
        secondary_runtime, secondary_graph_ctx, secondary_options);
    RabitqExactMemoScope memo_scope(primary_runtime, primary_graph_ctx, primary_range_posting_ctx,
        primary_options.rerank_distance_reuse, secondary_runtime, secondary_graph_ctx, secondary_range_posting_ctx);
    RabitqIterationResult result;
    RabitqIterationResult secondary_result;
    if (!overlap_cpu_gpu) {
        result = execute_rabitq_iteration(index,
                                          primary_runtime,
                                          primary_options,
                                          immutable_state,
                                          primary_prepared_state,
                                          primary_graph_ctx,
                                          primary_lowfreq_ctx,
                                          posting_index,
                                          primary_range_posting_ctx,
                                          print_summary);
        if (has_secondary) {
            secondary_result = execute_rabitq_iteration(index,
                                                        *secondary_runtime,
                                                        *secondary_options,
                                                        immutable_state,
                                                        *secondary_prepared_state,
                                                        *secondary_graph_ctx,
                                                        *secondary_lowfreq_ctx,
                                                        nullptr,
                                                        *secondary_range_posting_ctx,
                                                        print_summary);
        }
    } else {
        cpu_rerank_schedule::GpuPhaseGate all_gpu_complete;
        std::promise<void> primary_gpu_complete_promise;
        std::future<void> primary_gpu_complete =
            primary_gpu_complete_promise.get_future();
        int active_device = 0;
        check_cuda(cudaGetDevice(&active_device),
                   "cudaGetDevice before dual-graph overlap");
        std::future<RabitqIterationResult> primary_iteration_future = std::async(
            std::launch::async,
            [&] {
                bool gpu_completion_published = false;
                try {
                    check_cuda(cudaSetDevice(active_device),
                               "cudaSetDevice for primary dual-graph iteration");
                    RabitqIterationResult primary_result = execute_rabitq_iteration(
                        index,
                        primary_runtime,
                        primary_options,
                        immutable_state,
                        primary_prepared_state,
                        primary_graph_ctx,
                        primary_lowfreq_ctx,
                        posting_index,
                        primary_range_posting_ctx,
                        print_summary,
                        true,
                        [&] {
                            primary_gpu_complete_promise.set_value();
                            gpu_completion_published = true;
                            if (serial_cpu_rerank) all_gpu_complete.wait();
                        });
                    return primary_result;
                } catch (...) {
                    if (!gpu_completion_published) {
                        try {
                            primary_gpu_complete_promise.set_exception(
                                std::current_exception());
                        } catch (const std::future_error&) {
                        }
                    }
                    throw;
                }
            });

        try {
            primary_gpu_complete.get();
        } catch (...) {
            primary_iteration_future.get();
            throw;
        }

        std::exception_ptr secondary_failure;
        try {
            secondary_result = execute_rabitq_iteration(index,
                                                        *secondary_runtime,
                                                        *secondary_options,
                                                        immutable_state,
                                                        *secondary_prepared_state,
                                                        *secondary_graph_ctx,
                                                        *secondary_lowfreq_ctx,
                                                        nullptr,
                                                        *secondary_range_posting_ctx,
                                                        print_summary,
                                                        true,
                                                        [&] { if (serial_cpu_rerank) all_gpu_complete.release(); });
        } catch (...) {
            secondary_failure = std::current_exception();
            if (serial_cpu_rerank) all_gpu_complete.fail(secondary_failure);
        }
        result = primary_iteration_future.get();
        if (secondary_failure != nullptr) {
            std::rethrow_exception(secondary_failure);
        }
    }

    const auto final_merge_start = std::chrono::steady_clock::now();
    if (has_secondary) {
        if (secondary_result.results.size() != secondary_result.distances.size()) {
            fail_rabitq("secondary graph result/distance query count mismatch");
        }
        // Original-query merging loses route provenance. Bypass this final check
        // only if every producer for that query already verified its exact clause.
        std::vector<uint8_t> secondary_verified(secondary_runtime->original_query_count, 1);
        const bool secondary_resident_posting =
            (secondary_options->enable_lowfreq_gpu_posting || lowfreq_gpu_posting_enabled()) &&
            !defaults::lowfreq_full_filter_mode_is_pre(secondary_options->lowfreq_full_filter_mode);
        for (uint32_t route = 0; route < secondary_runtime->query_vectors.rows; ++route) {
            if (route < secondary_runtime->route_enabled.size() && !secondary_runtime->route_enabled[route]) continue;
            if (route < secondary_range_posting_ctx->plan.replaced_routes.size() &&
                secondary_range_posting_ctx->plan.mode == range_posting::Mode::Select &&
                secondary_range_posting_ctx->plan.replaced_routes[route]) continue;
            bool verified = true;
            if (route < secondary_runtime->query_is_lowfreq.size() && secondary_runtime->query_is_lowfreq[route]) {
                verified = secondary_resident_posting &&
                    CpuRerankReuse{nullptr, &secondary_graph_ctx->label_posting_predicate.verified_routes}.posting_verified(route);
            } else if (rabitq_route_is_highfreq_graph(*secondary_runtime, route)) {
                verified = secondary_graph_ctx->shortlist_predicate_verified;
            }
            if (!verified) secondary_verified.at(exact_original_query(*secondary_runtime, route)) = 0;
        }
        // Secondary range posting is not used on the supported path. Retain the
        // fallback for it rather than infer its actual scan mode at merge time.
        for (uint32_t route : secondary_range_posting_ctx->plan.representative_route_ids)
            secondary_verified.at(exact_original_query(*secondary_runtime, route)) = 0;
        for (uint32_t original_query = 0;
             original_query < secondary_result.results.size();
             ++original_query) {
            std::vector<idx_t>& row = secondary_result.results[original_query];
            std::vector<float>& distances = secondary_result.distances[original_query];
            if (row.size() != distances.size()) {
                fail_rabitq("secondary graph result/distance row width mismatch");
            }
            size_t write_idx = 0;
            for (size_t idx = 0; idx < row.size(); ++idx) {
                const idx_t sample_id = row[idx];
                if (sample_id == UINT32_MAX ||
                    sample_id >= primary_runtime.base_vectors.rows ||
                    (!secondary_verified.at(original_query) &&
                     !sample_matches_range_graph_clause_labels(
                        primary_runtime,
                        original_query,
                        static_cast<uint32_t>(sample_id)))) {
                    continue;
                }
                row[write_idx] = sample_id;
                distances[write_idx] = distances[idx];
                ++write_idx;
            }
            std::fill(row.begin() + write_idx, row.end(), UINT32_MAX);
            if (write_idx < distances.size()) {
                std::fill(distances.begin() + write_idx, distances.end(), FLT_MAX);
            }
        }
        merge_rabitq_iteration_results(
            result, secondary_result, static_cast<uint32_t>(primary_options.topk));
        result.pre_qps_route_partition_ms += secondary_result.pre_qps_route_partition_ms;
        result.pre_qps_result_init_ms += secondary_result.pre_qps_result_init_ms;
        result.pre_qps_total_ms += secondary_result.pre_qps_total_ms;
    }
    incremental_scope.publish(result.results, result.distances);
    const auto final_merge_end = std::chrono::steady_clock::now();
    perf_trace::span("dual_graph_final_merge", &primary_runtime, final_merge_start, final_merge_end);

    incremental_scope.finish();
    memo_scope.finish();
    const auto iteration_end = std::chrono::steady_clock::now();
    perf_trace::span("dual_graph_search_window", &primary_runtime, iteration_start, iteration_end);
    result.wall_ms = elapsed_ms(iteration_start, iteration_end);
    result.qps = result.query_count > 0 && result.wall_ms > 0.0
        ? static_cast<double>(result.query_count) / (result.wall_ms / 1000.0)
        : 0.0;
    // The dual-graph outer wall starts before either provider prepares its
    // route lists, so these component times are already counted in QPS.
    result.pre_qps_counted_in_wall = true;
    result.qps_start = iteration_start;
    if (print_summary && has_secondary) {
        std::cout << "[rabitq-dual-graph] original_queries=" << result.query_count
                  << " primary_routes=" << primary_runtime.query_vectors.rows
                  << " secondary_routes=" << secondary_runtime->query_vectors.rows
                  << " cpu_gpu_overlap=" << (overlap_cpu_gpu && !serial_cpu_rerank ? 1 : 0)
                  << " primary_gpu_provider_ms=" << result.gpu_provider_wall_ms
                  << " primary_cpu_tail_ms=" << result.cpu_tail_wall_ms
                  << " primary_range_posting_rerank_ms="
                  << result.range_posting_rerank_wall_ms
                  << " secondary_gpu_provider_ms="
                  << secondary_result.gpu_provider_wall_ms
                  << " secondary_cpu_tail_ms=" << secondary_result.cpu_tail_wall_ms
                  << " final_merge_ms=" << elapsed_ms(final_merge_start, final_merge_end)
                  << " wall_ms=" << result.wall_ms
                  << " qps=" << result.qps << "\n";
    }
    return result;
}

uint32_t route_original_query_id(const GraphRuntime& runtime, uint32_t route_idx)
{
    if (route_idx < runtime.route_to_original_query.size()) {
        return runtime.route_to_original_query[route_idx];
    }
    return route_idx;
}

struct RabitqLatencyRoutePlan {
    std::vector<uint32_t> route_ids;
    uint32_t selected_high_routes = 0;
    uint32_t selected_low_routes = 0;
    uint32_t unroutable_routes = 0;
};

bool rabitq_latency_route_is_lowfreq(const GraphRuntime& runtime, uint32_t route_idx)
{
    return route_idx < runtime.query_is_lowfreq.size() &&
           runtime.query_is_lowfreq[route_idx] != 0;
}

bool rabitq_latency_route_is_routable(const GraphRuntime& runtime, uint32_t route_idx)
{
    if (route_idx >= runtime.query_group_ids.size() ||
        runtime.query_group_ids[route_idx] == UINT32_MAX) {
        return false;
    }
    if (rabitq_latency_route_is_lowfreq(runtime, route_idx)) {
        return true;
    }
    return route_idx < runtime.start_local_ids.size() &&
           runtime.start_local_ids[route_idx] != UINT32_MAX;
}

RabitqLatencyRoutePlan build_rabitq_latency_routes(const GraphRuntime& runtime,
                                                   uint32_t query_window_start,
                                                   uint32_t query_window_count)
{
    RabitqLatencyRoutePlan plan;
    const uint32_t query_window_end = query_window_start + query_window_count;
    plan.route_ids.reserve(runtime.query_vectors.rows);
    for (uint32_t route_idx = 0; route_idx < runtime.query_vectors.rows; ++route_idx) {
        const uint32_t original_query = route_original_query_id(runtime, route_idx);
        if (original_query < query_window_start || original_query >= query_window_end) {
            continue;
        }
        if (!rabitq_latency_route_is_routable(runtime, route_idx)) {
            ++plan.unroutable_routes;
            continue;
        }
        plan.route_ids.push_back(route_idx);
        if (rabitq_latency_route_is_lowfreq(runtime, route_idx)) {
            ++plan.selected_low_routes;
        } else {
            ++plan.selected_high_routes;
        }
    }
    return plan;
}

std::vector<uint32_t> build_rabitq_lowfreq_candidate_ids(const RabitqSearchOptions& options,
                                                         const GraphRuntime& runtime,
                                                         uint32_t query_idx)
{
    if (query_idx >= runtime.query_vectors.rows ||
        query_idx >= runtime.query_group_ids.size() ||
        query_idx >= runtime.query_filter_bits.size() ||
        query_idx >= runtime.query_filter_labels.size()) {
        fail_rabitq("RabitQ latency lowfreq query id out of range");
    }
    const uint32_t label = runtime.query_group_ids[query_idx];
    if (label == UINT32_MAX ||
        label >= runtime.lowfreq_index.label_offset.size() ||
        label >= runtime.lowfreq_index.label_size.size()) {
        return {};
    }

    const bool pre_filter_full_labels =
        defaults::lowfreq_full_filter_mode_is_pre(options.lowfreq_full_filter_mode);
    const bool use_lowfreq_filter_mask = lowfreq_use_filter_mask_enabled();
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
        if (use_lowfreq_filter_mask &&
            (runtime.lowfreq_index.row_filter_mask[row] & query_filter_bit) == 0ULL) {
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

size_t max_rabitq_latency_lowfreq_candidate_count(const RabitqSearchOptions& options,
                                                  const GraphRuntime& runtime,
                                                  const std::vector<uint32_t>& route_ids)
{
    size_t max_count = 1;
    for (uint32_t query_idx : route_ids) {
        if (!rabitq_latency_route_is_lowfreq(runtime, query_idx)) {
            continue;
        }
        max_count = std::max(max_count,
                             build_rabitq_lowfreq_candidate_ids(options, runtime, query_idx).size());
    }
    return max_count;
}

void rerank_rabitq_latency_lowfreq_shortlist(const RabitqSearchOptions& options,
                                             const GraphRuntime& runtime,
                                             const RabitqPersistentLatencyQueue& queue,
                                             RabitqPersistentLatencySlot& slot,
                                             RabitqCpuWorkerBudget* cpu_worker_budget)
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
    lowfreq_rabitq_host_rerank({}, runtime,
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
                               distances,
                               cpu_worker_budget);
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

void collect_rabitq_latency_highfreq_gpu_results(const RabitqSearchOptions& options,
                                                 RabitqGraphSearchContext& graph_ctx,
                                                 RabitqLatencyResultSidecars& result_sidecars)
{
    const size_t query_count = graph_ctx.total_query_capacity;
    const size_t value_count = query_count * graph_ctx.result_width;
    std::vector<idx_t> host_result(value_count, UINT32_MAX);
    std::vector<float> host_result_dist(value_count, FLT_MAX);
    std::vector<float> host_time(query_count * 6, 0.0f);

    if (graph_ctx.result_buffer_is_host_mapped && graph_ctx.h_result_mapped != nullptr) {
        std::copy(graph_ctx.h_result_mapped,
                  graph_ctx.h_result_mapped + value_count,
                  host_result.begin());
    } else {
        check_cuda(cudaMemcpy(host_result.data(),
                              graph_ctx.d_result,
                              sizeof(idx_t) * host_result.size(),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy RabitQ latency result sidecar ids");
    }
    if (graph_ctx.result_buffer_is_host_mapped && graph_ctx.h_result_distances_mapped != nullptr) {
        std::copy(graph_ctx.h_result_distances_mapped,
                  graph_ctx.h_result_distances_mapped + value_count,
                  host_result_dist.begin());
    } else {
        check_cuda(cudaMemcpy(host_result_dist.data(),
                              graph_ctx.d_result_distances,
                              sizeof(dist_t) * host_result_dist.size(),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy RabitQ latency result sidecar distances");
    }
    if (graph_ctx.result_buffer_is_host_mapped && graph_ctx.h_time_per_block_mapped != nullptr) {
        std::copy(graph_ctx.h_time_per_block_mapped,
                  graph_ctx.h_time_per_block_mapped + query_count * 6,
                  host_time.begin());
    } else {
        check_cuda(cudaMemcpy(host_time.data(),
                              graph_ctx.d_time_per_block,
                              sizeof(float) * host_time.size(),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy RabitQ latency result sidecar steps");
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

void write_rabitq_latency_result_sidecars(const RabitqSearchOptions& options,
                                          int concurrency,
                                          uint32_t query_window_start,
                                          uint32_t query_window_count,
                                          const GraphRuntime& runtime,
                                          const RabitqLatencyResultSidecars& result_sidecars)
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

void write_rabitq_latency_header(std::ofstream& csv)
{
    csv << "backend,scheduler,concurrency,warmup_runs,timed_runs,run_all_queries,"
        << "query_window_count,selected_high_routes,selected_low_routes,unroutable_routes,"
        << "measured_requests,closed_loop_ms,qps,effective_inflight,"
        << "total_avg_ms,total_min_ms,total_p50_ms,total_p90_ms,total_p95_ms,total_p99_ms,total_max_ms,"
        << "gpu_avg_ms,gpu_min_ms,gpu_p50_ms,gpu_p90_ms,gpu_p95_ms,gpu_p99_ms,gpu_max_ms,"
        << "host_requests_served,host_requests_passed,host_requests_filtered,workers_thread\n";
}

void write_rabitq_latency_summary_row(std::ofstream& csv,
                                      int concurrency,
                                      const RabitqSearchOptions& options,
                                      uint32_t query_window_count,
                                      uint32_t selected_high_routes,
                                      uint32_t selected_low_routes,
                                      uint32_t unroutable_routes,
                                      double closed_loop_ms,
                                      const std::vector<RabitqPersistentLatencyMeasurement>& measurements,
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
    csv << "rabitq,persistent_queue,"
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

void write_rabitq_latency_trace_header(std::ofstream& csv)
{
    csv << "backend,scheduler,concurrency,request_ordinal,query_id,route_type,total_ms,gpu_ms\n";
}

void append_rabitq_latency_trace_rows(std::ofstream& csv,
                                      int concurrency,
                                      const std::vector<RabitqPersistentLatencyMeasurement>& measurements)
{
    for (size_t idx = 0; idx < measurements.size(); ++idx) {
        const auto& measurement = measurements[idx];
        csv << "rabitq,persistent_queue,"
            << concurrency << ','
            << idx << ','
            << measurement.query_id << ','
            << (measurement.lowfreq_route ? "low" : "high") << ','
            << measurement.total_ms << ','
            << measurement.gpu_ms << '\n';
    }
}

int run_rabitq_latency_benchmark(const RabitqSearchOptions& options)
{
    if (options.incremental_rerank_shortlist) fail_rabitq("incremental rerank does not support latency execution");
    gpu_predicate::require(options.rabitq_predicate_mode == "cpu", "GPU predicate unsupported in latency mode");

    if (options.graph_persistent_workers) {
        fail_rabitq("graph_persistent_workers supports search/benchmark, not persistent latency");
    }
    validate_rabitq_search_options(options);
    if (range_posting::parse_mode(options.range_posting_mode) !=
        range_posting::Mode::Off) {
        fail_rabitq("range posting is not supported by persistent latency execution");
    }
    if (options.lowfreq_rerank_merge_routes) {
        fail_rabitq("lowfreq_rerank_merge_routes is not supported by persistent latency execution");
    }
    if (!options.latency_staggered || !options.latency_persistent_queue) {
        fail_rabitq("latency mode for RabitQ requires --latency-staggered --latency-persistent-queue");
    }
    if (options.latency_online_metadata) {
        fail_rabitq("RabitQ persistent latency uses resident query metadata; do not enable latency_online_metadata");
    }
    RabitqSearchOptions latency_options = options;
    const bool latency_cpu_rerank = highfreq_cpu_rerank_enabled(latency_options);
    if (latency_cpu_rerank) {
        latency_options.highfreq_rerank_streaming = true;
    }
    if (latency_options.lowfreq_full_filter_mode != defaults::lowfreq_full_filter_mode_off &&
        latency_options.latency_lowfreq_mode == "resident") {
        if (!latency_options.enable_lowfreq_rerank) {
            fail_rabitq("RabitQ resident lowfreq latency full-label filtering requires enable_lowfreq_rerank");
        }
        if (defaults::lowfreq_full_filter_mode_is_pre(latency_options.lowfreq_full_filter_mode)) {
            latency_options.lowfreq_full_filter_mode = defaults::lowfreq_full_filter_mode_post;
        }
    }
    if (options.latency_query_start < 0 || options.latency_query_limit < 0) {
        fail_rabitq("latency query start/limit must be non-negative");
    }
    std::vector<int> concurrency_levels = options.latency_concurrency_levels.empty()
        ? options.latency_batch_sizes
        : options.latency_concurrency_levels;
    if (concurrency_levels.empty()) {
        fail_rabitq("latency concurrency levels must not be empty");
    }
    for (int concurrency : concurrency_levels) {
        if (concurrency < 0) {
            fail_rabitq("latency concurrency levels must be >= 0");
        }
    }

    RabitqHostIndex index = load_rabitq_index(options.artifact_path);
    GraphRuntime runtime = load_graph_runtime(make_rabitq_load_config(latency_options));
    validate_rabitq_runtime(index, runtime, latency_options);
    if (!has_rabitq_highfreq_routes(runtime)) {
        fail_rabitq("RabitQ persistent latency requires at least one high-frequency graph route");
    }
    if (runtime.original_query_count == 0) {
        fail_rabitq("RabitQ persistent latency query set is empty");
    }
    if (options.latency_query_start >= static_cast<int>(runtime.original_query_count)) {
        fail_rabitq("latency_query_start is beyond original query count");
    }
    const uint32_t query_window_start = static_cast<uint32_t>(options.latency_query_start);
    const uint32_t remaining_queries = runtime.original_query_count - query_window_start;
    const uint32_t query_window_count = options.latency_query_limit == 0
        ? remaining_queries
        : std::min(static_cast<uint32_t>(options.latency_query_limit), remaining_queries);
    const RabitqLatencyRoutePlan route_plan =
        build_rabitq_latency_routes(runtime, query_window_start, query_window_count);
    if (route_plan.route_ids.empty()) {
        fail_rabitq("RabitQ persistent latency query window contains no routable routes");
    }
    uint32_t resident_worker_capacity = 0;
    dispatch_rabitq_kernel_by_queue_size(options.search_queue_size, [&]<int QS>() {
        resident_worker_capacity = resident_worker_capacity_for_latency_kernel(
            persistent_rabitq_latency_kernel<QS>,
            kThreadInBlock,
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor RabitQ persistent latency",
            "cudaGetDeviceProperties RabitQ persistent latency");
    });
    size_t graph_scratch_capacity = count_rabitq_highfreq_graph_routes(runtime);
    for (int concurrency : concurrency_levels) {
        const int resolved_concurrency =
            concurrency == 0 ? static_cast<int>(resident_worker_capacity) : concurrency;
        if (resolved_concurrency <= 0) {
            fail_rabitq("latency concurrency level resolved to zero");
        }
        if (static_cast<uint32_t>(resolved_concurrency) > resident_worker_capacity) {
            fail_rabitq("latency_persistent_queue concurrency exceeds resident worker capacity");
        }
        graph_scratch_capacity = std::max(
            graph_scratch_capacity, static_cast<size_t>(resolved_concurrency));
    }

    RabitqImmutableDeviceState immutable_state;
    RabitqPreparedQueryState prepared_state;
    RabitqGraphSearchContext graph_ctx;
    try {
        init_immutable_state(immutable_state, index);
        init_prepared_query_state(prepared_state, index, runtime, latency_options);
        init_graph_context(graph_ctx,
                           index,
                           runtime,
                           latency_options,
                           graph_scratch_capacity);

        std::ofstream csv(options.latency_output_path);
        if (!csv.is_open()) {
            fail_rabitq("could not open latency output file " + options.latency_output_path);
        }
        write_rabitq_latency_header(csv);
        std::ofstream trace_csv;
        if (!options.latency_trace_output_path.empty()) {
            trace_csv.open(options.latency_trace_output_path);
            if (!trace_csv.is_open()) {
                fail_rabitq("could not open latency trace output file " +
                            options.latency_trace_output_path);
            }
            write_rabitq_latency_trace_header(trace_csv);
        }

        for (int concurrency : concurrency_levels) {
            const int resolved_concurrency =
                concurrency == 0 ? static_cast<int>(resident_worker_capacity) : concurrency;
            if (resolved_concurrency <= 0) {
                fail_rabitq("latency concurrency level resolved to zero");
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
                RabitqLatencyResultSidecars result_sidecars;
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
                std::vector<uint32_t> low_route_ids;
                high_route_ids.reserve(measured_ids.size());
                low_route_ids.reserve(measured_ids.size());
                for (uint32_t route_idx : measured_ids) {
                    if (rabitq_latency_route_is_lowfreq(runtime, route_idx)) {
                        low_route_ids.push_back(route_idx);
                    } else {
                        high_route_ids.push_back(route_idx);
                    }
                }
                reset_graph_context(graph_ctx);
                const size_t lowfreq_candidate_capacity =
                    latency_options.latency_lowfreq_mode == "candidate"
                        ? max_rabitq_latency_lowfreq_candidate_count(
                              latency_options, runtime, measured_ids)
                        : 1;
                RabitqPersistentLatencyQueue queue =
                    create_persistent_latency_queue(
                        static_cast<uint32_t>(resolved_concurrency),
                        latency_options, runtime, lowfreq_candidate_capacity);
                std::vector<RabitqPersistentLatencySlot> slots =
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

                std::vector<RabitqPersistentLatencyMeasurement> measurements;
                measurements.reserve(requested_count);
                std::mutex lowfreq_mutex;
                std::condition_variable lowfreq_cv;
                std::deque<size_t> lowfreq_queue;
                bool lowfreq_stop = false;
                std::exception_ptr lowfreq_error;
                std::vector<std::thread> lowfreq_workers;
                auto mark_lowfreq_done =
                    [&](RabitqPersistentLatencySlot &slot) {
                        std::lock_guard<std::mutex> lock(lowfreq_mutex);
                        slot.rerank_done_time =
                            std::chrono::steady_clock::now();
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
                                    RabitqPersistentLatencySlot &slot =
                                        slots[slot_index];
                                    rerank_rabitq_latency_lowfreq_shortlist(
                                        latency_options, runtime, queue, slot,
                                        graph_ctx.cpu_worker_budget.get());
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
                auto submit_slot = [&](RabitqPersistentLatencySlot &slot,
                                       uint32_t route_idx) {
                    slot.query_id = route_idx;
                    slot.lowfreq_route =
                        rabitq_latency_route_is_lowfreq(runtime, route_idx);
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
                            build_rabitq_lowfreq_candidate_ids(
                                latency_options, runtime, route_idx);
                        if (candidates.size() >
                            queue.lowfreq_candidate_capacity) {
                            fail_rabitq(
                                "RabitQ lowfreq latency candidate list exceeds "
                                "slot capacity");
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
                    RabitqPersistentLatencyCompletionControl &completion =
                        queue.h_completions[slot.slot_index];
                    completion.query_id = UINT32_MAX;
                    completion.done_seq = 0;
                    RabitqPersistentLatencyRequestControl &request =
                        queue.h_requests[slot.slot_index];
                    request.query_id = route_idx;
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
                        fail_rabitq("RabitQ CPU-rerank latency requires mapped "
                                    "streaming result buffers");
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
                                    RabitqPersistentLatencySlot &slot =
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
                                    highfreq_rabitq_host_rerank({},
                                        latency_options, runtime, query_ids,
                                        shortlist, reranked_results,
                                        &reranked_distances,
                                        graph_ctx.cpu_worker_budget.get());
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
                    launch_persistent_latency_worker(
                        index, latency_options, immutable_state, prepared_state,
                        graph_ctx, queue);
                    scan_thread = start_host_service();
                    start_rerank_workers();
                    start_lowfreq_workers();
                    size_t submitted = 0;
                    size_t completed = 0;
                    for (RabitqPersistentLatencySlot &slot : slots) {
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
                        for (RabitqPersistentLatencySlot &slot : slots) {
                            if (!slot.busy) {
                                continue;
                            }
                            volatile RabitqPersistentLatencyCompletionControl
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
                            RabitqPersistentLatencyMeasurement measurement;
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
                        collect_rabitq_latency_highfreq_gpu_results(
                            latency_options, graph_ctx, result_sidecars);
                        write_rabitq_latency_result_sidecars(
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
                        write_rabitq_latency_summary_row(
                            csv, resolved_concurrency, latency_options,
                            query_window_count,
                            static_cast<uint32_t>(high_route_ids.size()),
                            static_cast<uint32_t>(low_route_ids.size()),
                            route_plan.unroutable_routes, closed_loop_ms,
                            measurements, graph_ctx.host_filter_stats);
                        if (trace_csv.is_open()) {
                            append_rabitq_latency_trace_rows(
                                trace_csv, resolved_concurrency, measurements);
                        }
                        std::printf(
                            "[rabitq latency persistent] concurrency=%d "
                            "requests=%zu avg=%.4fms p95=%.4fms qps=%.2f\n",
                            resolved_concurrency, measurements.size(),
                            total_latency.avg, total_latency.p95,
                            closed_loop_ms > 0.0
                                ? static_cast<double>(measurements.size()) /
                                      (closed_loop_ms / 1000.0)
                                : 0.0);
                    } else {
                        std::printf(
                            "[rabitq latency persistent warmup] run=%d/%d "
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
    std::printf("RabitQ persistent latency CSV written to %s\n",
                options.latency_output_path.c_str());
    return 0;
}

int run_rabitq_search(const RabitqSearchOptions& base_options)
{
    validate_rabitq_search_options(base_options);

    RabitqHostIndex index = load_rabitq_index(base_options.artifact_path);
    const RabitqSearchOptions options =
        resolve_rabitq_artifact_defaults(index, base_options);
    GraphRuntime runtime = load_graph_runtime(make_rabitq_load_config(options));
    const bool range_posting_enabled =
        range_posting::parse_mode(options.range_posting_mode) != range_posting::Mode::Off;
    if (range_posting_enabled && runtime.range_posting_index == nullptr) {
        fail_rabitq("dataloader did not provide the enabled range posting index");
    }
    range_posting::Index disabled_posting_index;
    const range_posting::Index& posting_index = range_posting_enabled
        ? *runtime.range_posting_index
        : disabled_posting_index;
    validate_rabitq_runtime(index, runtime, options);
    const size_t graph_scratch_capacity = std::max<size_t>(
        count_rabitq_highfreq_graph_routes(runtime),
        (range_posting_enabled || needs_label_posting_predicate_context(runtime, options)) ? 1U : 0U);

    printf("Start native RabitQ search\n");
    RabitqImmutableDeviceState immutable_state;
    RabitqPreparedQueryState prepared_state;
    RabitqGraphSearchContext graph_ctx;
    RabitqLowfreqSearchContext lowfreq_ctx;
    RabitqRangePostingSearchContext range_posting_ctx;
    GraphRuntime range_graph_runtime;
    RabitqSearchOptions range_graph_options;
    RabitqPreparedQueryState range_graph_prepared_state;
    RabitqGraphSearchContext range_graph_ctx;
    RabitqLowfreqSearchContext range_graph_lowfreq_ctx;
    RabitqRangePostingSearchContext range_graph_range_posting_ctx;
    bool range_graph_active = false;
    try {
        init_immutable_state(immutable_state, index);
        init_prepared_query_state(prepared_state, index, runtime, options);
        if (has_rabitq_highfreq_routes(runtime) || range_posting_enabled ||
            needs_label_posting_predicate_context(runtime, options)) {
            init_graph_context(graph_ctx,
                               index,
                               runtime,
                               options,
                               graph_scratch_capacity);
        }
        if (range_posting::parse_mode(options.range_posting_mode) !=
            range_posting::Mode::Off) {
            RabitqRangePostingPlan posting_plan =
                build_range_posting_plan(posting_index, runtime, options);
            init_range_posting_context(range_posting_ctx,
                                       posting_index,
                                       std::move(posting_plan),
                                       options);
            init_range_posting_predicate_context(graph_ctx, range_posting_ctx.plan, runtime, options);
            std::cout << "[range-posting-init] mode=" << options.range_posting_mode
                      << " selected_queries="
                      << range_posting_ctx.plan.representative_route_ids.size()
                      << " scan_threshold=" << options.range_posting_scan_threshold
                      << " range_graph_or_handoffs="
                      << range_posting_ctx.plan.dropped_wide_hybrid_or_routes
                      << " range_graph_or_rows="
                      << range_posting_ctx.plan.dropped_wide_hybrid_or_rows
                      << " selected_range_graph_and_queries="
                      << range_posting_ctx.plan.selected_range_graph_and_queries
                      << " selected_range_graph_and_rows="
                      << range_posting_ctx.plan.selected_range_graph_and_rows
                      << " replaced_range_graph_and_label_routes="
                      << range_posting_ctx.plan.replaced_range_graph_and_label_routes
                      << " planned_scan_rows="
                      << range_posting_ctx.plan.total_rows_scanned
                      << " resident_rows=" << range_posting_ctx.resident_row_count
                      << " resident_bytes="
                      << range_posting_ctx.resident_row_count * sizeof(uint32_t)
                      << "\n";
        }

        const bool has_range_graph_or_routes =
            options.range_graph_hybrid_or_enabled &&
            range_posting_ctx.plan.dropped_wide_hybrid_or_routes != 0;
        const bool has_range_graph_and_routes =
            options.range_graph_hybrid_and_enabled &&
            range_posting_ctx.plan.selected_range_graph_and_queries != 0;
        if (has_range_graph_or_routes || has_range_graph_and_routes) {
            std::vector<uint8_t> range_graph_query_mask(runtime.original_query_count, 0);
            for (uint32_t original_query = 0;
                 original_query < runtime.original_query_count;
                 ++original_query) {
                range_graph_query_mask[original_query] =
                    range_posting_ctx.plan
                            .dropped_wide_hybrid_or_original_queries[original_query] != 0 ||
                    range_posting_ctx.plan
                            .selected_range_graph_and_original_queries[original_query] != 0;
            }
            range_graph_options = make_rabitq_range_graph_search_options(options);
            range_graph_runtime = load_graph_runtime(
                make_rabitq_range_graph_load_config(
                    options,
                    range_graph_query_mask));
            make_identity_graph_filter_masks_permissive(range_graph_runtime);
            if (range_graph_runtime.original_query_count != runtime.original_query_count ||
                range_graph_runtime.query_vector_metadata.path !=
                    runtime.query_vector_metadata.path ||
                range_graph_runtime.original_query_ranges.rows !=
                    runtime.original_query_ranges.rows ||
                range_graph_runtime.original_query_ranges.range_count !=
                    runtime.original_query_ranges.range_count) {
                fail_rabitq("secondary range graph does not share the primary query batch");
            }
            if (range_graph_runtime.rerank_vector_store != nullptr) {
                fail_rabitq("secondary range graph initialized a duplicate vector store");
            }
            if (range_graph_runtime.base_vector_metadata.rows !=
                    runtime.base_vector_metadata.rows ||
                range_graph_runtime.base_vector_metadata.cols !=
                    runtime.base_vector_metadata.cols) {
                fail_rabitq("secondary range graph vector shape differs from primary runtime");
            }
            range_graph_runtime.rerank_vector_store = runtime.rerank_vector_store;

            if (range_graph_runtime.rerank_vector_store != nullptr &&
                (range_graph_runtime.rerank_vector_store->rows() !=
                     range_graph_runtime.base_vector_metadata.rows ||
                 range_graph_runtime.rerank_vector_store->cols() !=
                     range_graph_runtime.base_vector_metadata.cols)) {
                fail_rabitq("shared rerank vector store shape differs from secondary runtime");
            }
            const size_t removed_routes = retain_graph_routes_for_original_queries(
                range_graph_runtime,
                range_graph_query_mask);
            if (range_graph_runtime.query_vectors.rows == 0) {
                fail_rabitq("range graph hybrid mask retained no routes");
            }
            validate_rabitq_runtime(index, range_graph_runtime, range_graph_options);
            init_borrowed_prepared_query_state(range_graph_prepared_state,
                                               prepared_state,
                                               range_graph_runtime);
            const size_t range_graph_scratch_capacity =
                count_rabitq_highfreq_graph_routes(range_graph_runtime);
            if (has_rabitq_highfreq_routes(range_graph_runtime)) {
                init_graph_context(range_graph_ctx,
                                   index,
                                   range_graph_runtime,
                                   range_graph_options,
                                   range_graph_scratch_capacity,
                                   &graph_ctx,
                                   &runtime);
            }
            range_graph_active = true;
            std::cout << "[rabitq-dual-graph-init] primary_routes="
                      << runtime.query_vectors.rows
                      << " secondary_routes=" << range_graph_runtime.query_vectors.rows
                      << " removed_secondary_routes=" << removed_routes
                      << " shared_original_queries=" << runtime.original_query_count
                      << " recovered_or_queries="
                      << range_posting_ctx.plan.dropped_wide_hybrid_or_routes
                      << " selected_and_queries="
                      << range_posting_ctx.plan.selected_range_graph_and_queries
                      << " selected_and_rows="
                      << range_posting_ctx.plan.selected_range_graph_and_rows
                      << " replaced_and_label_routes="
                      << range_posting_ctx.plan.replaced_range_graph_and_label_routes
                      << " secondary_graph_rows=" << range_graph_runtime.graph.rows
                      << " secondary_graph_storage="
                      << range_graph_options.graph_storage_mode
                      << " shared_base_range_bytes="
                      << range_graph_ctx.base_range_value_count * sizeof(float)
                      << " shared_rerank_vector_store="
                      << (range_graph_runtime.rerank_vector_store != nullptr ? 1 : 0)
                      << "\n";
        }

        if (graph_ctx.cpu_worker_budget == nullptr) {
            graph_ctx.cpu_worker_budget = std::make_shared<RabitqCpuWorkerBudget>(
                static_cast<size_t>(std::max(1, options.highfreq_rerank_threads)));
        }
        prepare_cpu_rerank_workers(graph_ctx, options);
        if (range_graph_active) {
            range_graph_ctx.cpu_worker_budget = graph_ctx.cpu_worker_budget;
            range_graph_ctx.cpu_rerank_workers = graph_ctx.cpu_rerank_workers;
        }
        const std::shared_ptr<RabitqCpuWorkerBudget> cpu_worker_budget =
            graph_ctx.cpu_worker_budget;

        for (int run = 0; run < options.warmup_runs; ++run) {
            const RabitqProcessPageFaults faults_before =
                snapshot_process_page_faults();
            const RabitqCpuWorkerBudgetStats budget_before =
                cpu_worker_budget->snapshot_stats();
            (void)execute_rabitq_dual_graph_iteration(
                index,
                runtime,
                options,
                immutable_state,
                prepared_state,
                graph_ctx,
                lowfreq_ctx,
                range_posting_enabled ? &posting_index : nullptr,
                range_posting_ctx,
                range_graph_active ? &range_graph_runtime : nullptr,
                range_graph_active ? &range_graph_options : nullptr,
                range_graph_active ? &range_graph_prepared_state : nullptr,
                range_graph_active ? &range_graph_ctx : nullptr,
                range_graph_active ? &range_graph_lowfreq_ctx : nullptr,
                range_graph_active ? &range_graph_range_posting_ctx : nullptr,
                false);
            const RabitqProcessPageFaults faults_after =
                snapshot_process_page_faults();
            const RabitqCpuWorkerBudgetStats budget_after =
                cpu_worker_budget->snapshot_stats();
            std::cout << "[rabitq-warmup] run=" << run + 1 << '/'
                      << options.warmup_runs
                      << " minor_faults="
                      << counter_delta(faults_after.minor, faults_before.minor)
                      << " major_faults="
                      << counter_delta(faults_after.major, faults_before.major)
                      << " cpu_worker_acquisitions="
                      << counter_delta(budget_after.acquisitions,
                                       budget_before.acquisitions)
                      << " cpu_worker_contended="
                      << counter_delta(budget_after.contended_acquisitions,
                                       budget_before.contended_acquisitions)
                      << " cpu_worker_wait_ms="
                      << static_cast<double>(counter_delta(budget_after.wait_ns,
                                                           budget_before.wait_ns)) /
                             1.0e6
                      << " cpu_worker_peak_active="
                      << budget_after.peak_active_workers << '\n';
        }

        RabitqIterationResult result;
        for (int run = 0; run < options.timed_runs; ++run) {
            const RabitqProcessPageFaults faults_before =
                snapshot_process_page_faults();
            const RabitqCpuWorkerBudgetStats budget_before =
                cpu_worker_budget->snapshot_stats();
            result = execute_rabitq_dual_graph_iteration(
                index,
                runtime,
                options,
                immutable_state,
                prepared_state,
                graph_ctx,
                lowfreq_ctx,
                range_posting_enabled ? &posting_index : nullptr,
                range_posting_ctx,
                range_graph_active ? &range_graph_runtime : nullptr,
                range_graph_active ? &range_graph_options : nullptr,
                range_graph_active ? &range_graph_prepared_state : nullptr,
                range_graph_active ? &range_graph_ctx : nullptr,
                range_graph_active ? &range_graph_lowfreq_ctx : nullptr,
                range_graph_active ? &range_graph_range_posting_ctx : nullptr,
                true);
            const RabitqProcessPageFaults faults_after =
                snapshot_process_page_faults();
            const RabitqCpuWorkerBudgetStats budget_after =
                cpu_worker_budget->snapshot_stats();
            printf("[rabitq-e2e] run=%d/%d original_queries=%zu wall_ms=%.3f qps=%.6f "
                   "minor_faults=%llu major_faults=%llu cpu_worker_acquisitions=%llu "
                   "cpu_worker_contended=%llu cpu_worker_wait_ms=%.3f "
                   "cpu_worker_peak_active=%zu\n",
                   run + 1,
                   options.timed_runs,
                   result.query_count,
                   result.wall_ms,
                   result.qps,
                   static_cast<unsigned long long>(
                       counter_delta(faults_after.minor, faults_before.minor)),
                   static_cast<unsigned long long>(
                       counter_delta(faults_after.major, faults_before.major)),
                   static_cast<unsigned long long>(
                       counter_delta(budget_after.acquisitions,
                                     budget_before.acquisitions)),
                   static_cast<unsigned long long>(
                       counter_delta(budget_after.contended_acquisitions,
                                     budget_before.contended_acquisitions)),
                   static_cast<double>(counter_delta(budget_after.wait_ns,
                                                     budget_before.wait_ns)) /
                       1.0e6,
                   budget_after.peak_active_workers);
        }

        printf("RabitQ mixed query split: total(route)=%u highfreq(route)=%zu lowfreq(route)=%zu provider(route)=%zu unroutable(route)=%zu\n",
               runtime.query_vectors.rows,
               result.highfreq_route_count,
               result.lowfreq_route_count,
               result.provider_route_count,
               result.unroutable_route_count);
        printf("RabitQ original queries: %u\n", runtime.original_query_count);
        save_results_to_csv(result.results, options.output_csv_path);
        cleanup_graph_context(range_graph_ctx);
        cleanup_lowfreq_context(range_graph_lowfreq_ctx);
        cleanup_range_posting_context(range_graph_range_posting_ctx);
        cleanup_prepared_query_state(range_graph_prepared_state);
        cleanup_graph_context(graph_ctx);
        cleanup_lowfreq_context(lowfreq_ctx);
        cleanup_range_posting_context(range_posting_ctx);
        cleanup_prepared_query_state(prepared_state);
        cleanup_immutable_state(immutable_state);
    } catch (...) {
        cleanup_graph_context(range_graph_ctx);
        cleanup_lowfreq_context(range_graph_lowfreq_ctx);
        cleanup_range_posting_context(range_graph_range_posting_ctx);
        cleanup_prepared_query_state(range_graph_prepared_state);
        cleanup_graph_context(graph_ctx);
        cleanup_lowfreq_context(lowfreq_ctx);
        cleanup_range_posting_context(range_posting_ctx);
        cleanup_prepared_query_state(prepared_state);
        cleanup_immutable_state(immutable_state);
        throw;
    }
    return 0;
}

int run_benchmark_sweep(const RabitqSearchOptions& base_options,
                        const std::vector<float>& quick_filter_bonuses,
                        const std::vector<int>& queue_sizes,
                        const std::string& output_csv_path)
{
    if (base_options.raw_navigation_scores && quick_filter_bonuses.size() > 1) {
        fail_rabitq("legacy bonus sweep is inactive for raw navigation scores; use one value");
    }

    cpu_profile::Reporter cpu_rerank_report;
    const auto benchmark_setup_start = std::chrono::steady_clock::now();
    const std::string setup_path = make_sidecar_csv_path(output_csv_path, ".setup.jsonl");
    QuerySetupReporter setup_report(setup_path, "rabitq");
    nlohmann::json setup_stages = {{"posting_provider_plan", 0.0}};
    auto setup_stage_start = std::chrono::steady_clock::now();
    validate_rabitq_search_options(base_options);

    RabitqHostIndex index = load_rabitq_index(base_options.artifact_path);
    RabitqSearchOptions benchmark_options =
        resolve_rabitq_artifact_defaults(index, base_options);
    const GraphLoadConfig benchmark_load_config =
        make_rabitq_load_config(benchmark_options);
    setup_stages["artifact_load_and_options"] =
        elapsed_ms(setup_stage_start, std::chrono::steady_clock::now());
    setup_stage_start = std::chrono::steady_clock::now();
    GraphRuntime runtime = load_graph_runtime(benchmark_load_config);
    setup_stages["primary_runtime_load"] =
        elapsed_ms(setup_stage_start, std::chrono::steady_clock::now());
    setup_stage_start = std::chrono::steady_clock::now();
    const bool range_posting_enabled =
        range_posting::parse_mode(benchmark_options.range_posting_mode) !=
        range_posting::Mode::Off;
    if (range_posting_enabled && runtime.range_posting_index == nullptr) {
        fail_rabitq("dataloader did not provide the enabled benchmark range posting index");
    }
    range_posting::Index disabled_posting_index;
    const range_posting::Index& posting_index = range_posting_enabled
        ? *runtime.range_posting_index
        : disabled_posting_index;
    validate_rabitq_runtime(index, runtime, benchmark_options);
    const size_t graph_scratch_capacity = std::max<size_t>(
        count_rabitq_highfreq_graph_routes(runtime),
        (range_posting_enabled || needs_label_posting_predicate_context(runtime, benchmark_options)) ? 1U : 0U);

    RabitqImmutableDeviceState immutable_state;
    RabitqPreparedQueryState prepared_state;
    RabitqPreparedQueryTiming prepared_timing;
    RabitqGraphSearchContext graph_ctx;
    RabitqLowfreqSearchContext lowfreq_ctx;
    RabitqRangePostingSearchContext range_posting_ctx;
    GraphRuntime range_graph_runtime;
    RabitqSearchOptions range_graph_options;
    RabitqPreparedQueryState range_graph_prepared_state;
    RabitqGraphSearchContext range_graph_ctx;
    RabitqLowfreqSearchContext range_graph_lowfreq_ctx;
    RabitqRangePostingSearchContext range_graph_range_posting_ctx;
    size_t range_graph_scratch_capacity = 0;
    bool range_graph_active = false;
    const auto cleanup_range_graph_contexts = [&] {
        cleanup_graph_context(range_graph_ctx);
        cleanup_lowfreq_context(range_graph_lowfreq_ctx);
        cleanup_range_posting_context(range_graph_range_posting_ctx);
        cleanup_prepared_query_state(range_graph_prepared_state);
    };
    try {
        setup_stages["host_validation_and_options"] =
            elapsed_ms(setup_stage_start, std::chrono::steady_clock::now());
        setup_stage_start = std::chrono::steady_clock::now();
        init_immutable_state(immutable_state, index);
        setup_stages["immutable_index_gpu_upload"] =
            elapsed_ms(setup_stage_start, std::chrono::steady_clock::now());
        setup_stage_start = std::chrono::steady_clock::now();
        init_prepared_query_state(prepared_state, index, runtime, benchmark_options,
                                  &prepared_timing);
        setup_stages["primary_query_gpu_prepare"] =
            elapsed_ms(setup_stage_start, std::chrono::steady_clock::now());
        setup_stage_start = std::chrono::steady_clock::now();
        if (has_rabitq_highfreq_routes(runtime) || range_posting_enabled ||
            needs_label_posting_predicate_context(runtime, benchmark_options)) {
            init_graph_context(graph_ctx,
                               index,
                               runtime,
                               benchmark_options,
                               graph_scratch_capacity);
        }
        setup_stages["primary_graph_context_init"] =
            elapsed_ms(setup_stage_start, std::chrono::steady_clock::now());
        setup_stage_start = std::chrono::steady_clock::now();
        if (range_posting::parse_mode(benchmark_options.range_posting_mode) !=
            range_posting::Mode::Off) {
            RabitqRangePostingPlan posting_plan =
                build_range_posting_plan(posting_index, runtime, benchmark_options);
            setup_stages["posting_provider_plan"] =
                elapsed_ms(setup_stage_start, std::chrono::steady_clock::now());
            setup_stage_start = std::chrono::steady_clock::now();
            init_range_posting_context(range_posting_ctx,
                                       posting_index,
                                       std::move(posting_plan),
                                       benchmark_options);
            init_range_posting_predicate_context(graph_ctx, range_posting_ctx.plan, runtime, benchmark_options);
        }
        setup_stages["posting_context_init"] =
            elapsed_ms(setup_stage_start, std::chrono::steady_clock::now());
        setup_stage_start = std::chrono::steady_clock::now();

        const bool has_range_graph_or_routes =
            benchmark_options.range_graph_hybrid_or_enabled &&
            range_posting_ctx.plan.dropped_wide_hybrid_or_routes != 0;
        const bool has_range_graph_and_routes =
            benchmark_options.range_graph_hybrid_and_enabled &&
            range_posting_ctx.plan.selected_range_graph_and_queries != 0;
        if (has_range_graph_or_routes || has_range_graph_and_routes) {
            const auto secondary_query_mask_start = std::chrono::steady_clock::now();
            std::vector<uint8_t> range_graph_query_mask(runtime.original_query_count, 0);
            for (uint32_t original_query = 0;
                 original_query < runtime.original_query_count;
                 ++original_query) {
                range_graph_query_mask[original_query] =
                    range_posting_ctx.plan
                            .dropped_wide_hybrid_or_original_queries[original_query] != 0 ||
                    range_posting_ctx.plan
                            .selected_range_graph_and_original_queries[original_query] != 0;
            }
            range_graph_options =
                make_rabitq_range_graph_search_options(benchmark_options);
            setup_stages["secondary_query_mask_and_options"] =
                elapsed_ms(secondary_query_mask_start, std::chrono::steady_clock::now());
            range_graph_runtime = load_graph_runtime(
                make_rabitq_range_graph_load_config(
                    benchmark_options,
                    range_graph_query_mask));
            const auto secondary_query_finalize_start = std::chrono::steady_clock::now();
            make_identity_graph_filter_masks_permissive(range_graph_runtime);
            if (range_graph_runtime.original_query_count != runtime.original_query_count ||
                range_graph_runtime.query_vector_metadata.path !=
                    runtime.query_vector_metadata.path ||
                range_graph_runtime.original_query_ranges.rows !=
                    runtime.original_query_ranges.rows ||
                range_graph_runtime.original_query_ranges.range_count !=
                    runtime.original_query_ranges.range_count) {
                fail_rabitq("secondary range graph does not share the primary query batch");
            }
            if (range_graph_runtime.rerank_vector_store != nullptr) {
                fail_rabitq("secondary range graph initialized a duplicate vector store");
            }
            if (range_graph_runtime.base_vector_metadata.rows !=
                    runtime.base_vector_metadata.rows ||
                range_graph_runtime.base_vector_metadata.cols !=
                    runtime.base_vector_metadata.cols) {
                fail_rabitq("secondary range graph vector shape differs from primary runtime");
            }
            range_graph_runtime.rerank_vector_store = runtime.rerank_vector_store;

            if (range_graph_runtime.rerank_vector_store != nullptr &&
                (range_graph_runtime.rerank_vector_store->rows() !=
                     range_graph_runtime.base_vector_metadata.rows ||
                 range_graph_runtime.rerank_vector_store->cols() !=
                     range_graph_runtime.base_vector_metadata.cols)) {
                fail_rabitq("shared rerank vector store shape differs from secondary runtime");
            }
            const size_t removed_routes = retain_graph_routes_for_original_queries(
                range_graph_runtime,
                range_graph_query_mask);
            if (range_graph_runtime.query_vectors.rows == 0) {
                fail_rabitq("range graph hybrid mask retained no routes");
            }
            validate_rabitq_runtime(index, range_graph_runtime, range_graph_options);
            init_borrowed_prepared_query_state(range_graph_prepared_state,
                                               prepared_state,
                                               range_graph_runtime);
            range_graph_scratch_capacity =
                count_rabitq_highfreq_graph_routes(range_graph_runtime);
            setup_stages["secondary_query_finalize_and_gpu_bind"] =
                elapsed_ms(secondary_query_finalize_start, std::chrono::steady_clock::now());
            if (has_rabitq_highfreq_routes(range_graph_runtime)) {
                init_graph_context(range_graph_ctx,
                                   index,
                                   range_graph_runtime,
                                   range_graph_options,
                                   range_graph_scratch_capacity,
                                   &graph_ctx,
                                   &runtime);
            }
            range_graph_active = true;
            std::cout << "[rabitq-dual-graph-benchmark-init] primary_routes="
                      << runtime.query_vectors.rows
                      << " secondary_routes=" << range_graph_runtime.query_vectors.rows
                      << " removed_secondary_routes=" << removed_routes
                      << " shared_original_queries=" << runtime.original_query_count
                      << " posting_queries="
                      << range_posting_ctx.plan.representative_route_ids.size()
                      << " planned_scan_rows="
                      << range_posting_ctx.plan.total_rows_scanned
                      << " recovered_or_queries="
                      << range_posting_ctx.plan.dropped_wide_hybrid_or_routes
                      << " selected_and_queries="
                      << range_posting_ctx.plan.selected_range_graph_and_queries
                      << " secondary_graph_rows=" << range_graph_runtime.graph.rows
                      << " shared_rerank_vector_store="
                      << (range_graph_runtime.rerank_vector_store != nullptr ? 1 : 0)
                      << "\n";
        }
        setup_stages["secondary_graph_setup"] =
            elapsed_ms(setup_stage_start, std::chrono::steady_clock::now());
        setup_stage_start = std::chrono::steady_clock::now();

    } catch (...) {

        cleanup_range_graph_contexts();
        cleanup_graph_context(graph_ctx);
        cleanup_lowfreq_context(lowfreq_ctx);
        cleanup_range_posting_context(range_posting_ctx);
        cleanup_prepared_query_state(prepared_state);
        cleanup_immutable_state(immutable_state);
        throw;
    }

    const auto prepared_timing_json = [](const RabitqPreparedQueryTiming& t) {
        return nlohmann::json{{"total_ms", t.total_ms},
            {"validation_ms", t.validation_ms},
            {"query_buffer_allocate_upload_ms", t.buffer_allocate_upload_ms},
            {"host_u8_to_float_detail_ms", t.host_u8_to_float_detail_ms},
            {"static_rotation_anchors_ms", t.static_rotation_anchors_ms},
            {"gpu_preprocess_ms", t.gpu_preprocess_ms},
            {"rotation_method", t.rotation_method},
            {"centroid_method", t.centroid_method},
            {"rotation_engine_setup_ms", t.rotation_engine_setup_ms},
            {"rotation_device_ms", t.rotation_device_ms},
            {"quantize_device_ms", t.quantize_device_ms},
            {"lut_device_ms", t.lut_device_ms},
            {"factors_device_ms", t.factors_device_ms},
            {"centroid_device_ms", t.centroid_device_ms}};
    };
    setup_report.append({
        {"record_type", "workload_setup"},
        {"summary_csv", output_csv_path},
        {"query_vector_path", benchmark_options.query_vector_path},
        {"query_label_spmat_path", benchmark_options.query_label_spmat_path},
        {"query_manifest_jsonl_path", benchmark_options.query_manifest_jsonl_path},
        {"query_range_path", benchmark_options.query_range_path},
        {"original_queries", runtime.original_query_count},
        {"primary_routes", runtime.query_vectors.rows},
        {"secondary_routes", range_graph_runtime.query_vectors.rows},
        {"secondary_active", range_graph_active},
        {"setup_total_before_sweep_ms",
            elapsed_ms(benchmark_setup_start, std::chrono::steady_clock::now())},
        {"setup_stages_ms", setup_stages},
        {"primary_runtime", query_preparation_timing_json(runtime.query_preparation_timing)},
        {"primary_gpu_query", prepared_timing_json(prepared_timing)},
        {"secondary_runtime", query_preparation_timing_json(range_graph_runtime.query_preparation_timing)},

        {"qps_timer", range_graph_active ? "dual_graph_entry_through_final_merge" :
                                           "provider_execution_through_final_route_merge"},
        {"nested_details_are_additive", false}});
    std::cout << "[query-setup] path=" << setup_path
              << " query_prepare_ms=" << runtime.query_preparation_timing.query_stage_sum_ms()
              << " route_generation_ms=" << runtime.query_preparation_timing.route_generation_ms
              << " gpu_query_prepare_ms=" << prepared_timing.total_ms << "\n";

    std::ofstream csv(output_csv_path);
    if (!csv.is_open()) {

        cleanup_range_graph_contexts();
        cleanup_graph_context(graph_ctx);
        cleanup_lowfreq_context(lowfreq_ctx);
        cleanup_range_posting_context(range_posting_ctx);
        cleanup_prepared_query_state(prepared_state);
        cleanup_immutable_state(immutable_state);
        fail_rabitq("failed to open benchmark output CSV: " + output_csv_path);
    }
    csv << "quick_filter_bonus,queue_size,finish_count,qps,kernel_ms,total_steps,avg_step_num,"
           "highfreq_route_count,lowfreq_route_count,provider_route_count,unroutable_route_count,"
           "highfreq_total_steps,highfreq_avg_step_num,lowfreq_total_steps,lowfreq_avg_step_num,"
           "pass_ratio,workers_thread\n";

    try {
        for (float quick_filter_bonus : quick_filter_bonuses) {
            if (quick_filter_bonus <= 0.0f || quick_filter_bonus > 1.0f) {
                fail_rabitq("benchmark quick_filter_bonus must be in (0, 1]");
            }
            for (int queue_size : queue_sizes) {
                if (queue_size <= 0 || queue_size > 512) {
                    fail_rabitq("benchmark queue_size must be in [1, 512]");
                }
                if (queue_size < base_options.topk) {
                    fail_rabitq("benchmark queue_size must be >= topk");
                }
                const auto point_setup_start = std::chrono::steady_clock::now();
                const int finish_count = base_options.search_finish_count;
                RabitqSearchOptions iteration_options = benchmark_options;
                iteration_options.quick_filter_bonus = quick_filter_bonus;
                iteration_options.bit_filter_bonus = quick_filter_bonus;
                iteration_options.range_filter_bonus = quick_filter_bonus;
                iteration_options.search_queue_size = queue_size;
                RabitqSearchOptions range_graph_iteration_options;
                if (range_graph_active) {
                    range_graph_iteration_options = range_graph_options;
                    range_graph_iteration_options.quick_filter_bonus = quick_filter_bonus;
                    range_graph_iteration_options.bit_filter_bonus = quick_filter_bonus;
                    range_graph_iteration_options.range_filter_bonus = quick_filter_bonus;
                    range_graph_iteration_options.search_queue_size = queue_size;
                }

                if (graph_ctx.initialized &&
                    (graph_ctx.result_width != highfreq_result_width(iteration_options, runtime) ||
                     graph_ctx.stop_width != highfreq_stop_width(iteration_options))) {

                    cleanup_graph_context(range_graph_ctx);
                    cleanup_graph_context(graph_ctx);
                    init_graph_context(graph_ctx,
                                       index,
                                       runtime,
                                       iteration_options,
                                       graph_scratch_capacity);
                    init_range_posting_predicate_context(graph_ctx, range_posting_ctx.plan, runtime, iteration_options);
                    if (range_graph_active &&
                        has_rabitq_highfreq_routes(range_graph_runtime)) {
                        init_graph_context(range_graph_ctx,
                                           index,
                                           range_graph_runtime,
                                           range_graph_iteration_options,
                                           range_graph_scratch_capacity,
                                           &graph_ctx,
                                           &runtime);
                    }

                }

                if (graph_ctx.cpu_worker_budget == nullptr) {
                    graph_ctx.cpu_worker_budget = std::make_shared<RabitqCpuWorkerBudget>(
                        static_cast<size_t>(
                            std::max(1, iteration_options.highfreq_rerank_threads)));
                }
                prepare_cpu_rerank_workers(graph_ctx, iteration_options);
                if (range_graph_active) {
                    range_graph_ctx.cpu_worker_budget = graph_ctx.cpu_worker_budget;
                    range_graph_ctx.cpu_rerank_workers = graph_ctx.cpu_rerank_workers;
                }
                const std::shared_ptr<RabitqCpuWorkerBudget> cpu_worker_budget =
                    graph_ctx.cpu_worker_budget;

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
                    "routes_path,"
                    "process_minor_faults_delta,process_major_faults_delta,"
                    "cpu_worker_acquisitions_delta,cpu_worker_contended_delta,"
                    "cpu_worker_wait_ms_delta,cpu_worker_peak_active");
                const std::string point_id =
                    "rabitq_b" + std::to_string(quick_filter_bonus) +
                    "_q" + std::to_string(queue_size) +
                    "_f" + std::to_string(finish_count);

                const auto execute_benchmark_iteration = [&](bool print_summary) {
                    if (!range_graph_active) {
                        return execute_rabitq_iteration(
                            index,
                            runtime,
                            iteration_options,
                            immutable_state,
                            prepared_state,
                            graph_ctx,
                            lowfreq_ctx,
                            range_posting_enabled ? &posting_index : nullptr,
                            range_posting_ctx,
                            print_summary);
                    }
                    return execute_rabitq_dual_graph_iteration(
                        index,
                        runtime,
                        iteration_options,
                        immutable_state,
                        prepared_state,
                        graph_ctx,
                        lowfreq_ctx,
                        range_posting_enabled ? &posting_index : nullptr,
                        range_posting_ctx,
                        &range_graph_runtime,
                        &range_graph_iteration_options,
                        &range_graph_prepared_state,
                        &range_graph_ctx,
                        &range_graph_lowfreq_ctx,
                        &range_graph_range_posting_ctx,
                        print_summary);
                };

                setup_report.append({{"record_type", "point_setup"},
                    {"point_id", point_id}, {"query_setup_reused", true},
                    {"point_context_and_harness_setup_ms",
                        elapsed_ms(point_setup_start, std::chrono::steady_clock::now())}});
                RabitqIterationResult aggregate;
                for (int run = 0; run < iteration_options.warmup_runs; ++run) {

                    std::cout << "[benchmark warmup start] backend=rabitq queue_size="
                              << queue_size << " run=" << run << '/'
                              << iteration_options.warmup_runs << "\n";
                    const RabitqProcessPageFaults faults_before =
                        snapshot_process_page_faults();
                    const RabitqCpuWorkerBudgetStats budget_before =
                        cpu_worker_budget->snapshot_stats();
                    cpu_profile::Iteration cpu_profile_iteration(cpu_rerank_report, "warmup", queue_size, run, point_id);
                    const RabitqIterationResult warmup_result = execute_benchmark_iteration(false);
                    report_iteration_preparation(setup_report, point_id, "warmup", run,
                                                 warmup_result, benchmark_setup_start);
                    const RabitqProcessPageFaults faults_after =
                        snapshot_process_page_faults();
                    const RabitqCpuWorkerBudgetStats budget_after =
                        cpu_worker_budget->snapshot_stats();
                    cpu_profile_iteration.finish();
                    std::cout << "[benchmark warmup end] backend=rabitq queue_size="
                              << queue_size << " run=" << run << '/'
                              << iteration_options.warmup_runs
                              << " minor_faults="
                              << counter_delta(faults_after.minor, faults_before.minor)
                              << " major_faults="
                              << counter_delta(faults_after.major, faults_before.major)
                              << " cpu_worker_acquisitions="
                              << counter_delta(budget_after.acquisitions,
                                               budget_before.acquisitions)
                              << " cpu_worker_contended="
                              << counter_delta(budget_after.contended_acquisitions,
                                               budget_before.contended_acquisitions)
                              << " cpu_worker_wait_ms="
                              << static_cast<double>(counter_delta(budget_after.wait_ns,
                                                                   budget_before.wait_ns)) /
                                     1.0e6
                              << " cpu_worker_peak_active="
                              << budget_after.peak_active_workers << '\n';
                    std::cout.flush();
                }

                for (int run = 0; run < iteration_options.timed_runs; ++run) {

                    const RabitqProcessPageFaults faults_before =
                        snapshot_process_page_faults();
                    const RabitqCpuWorkerBudgetStats budget_before =
                        cpu_worker_budget->snapshot_stats();
                    cpu_profile::Iteration cpu_profile_iteration(cpu_rerank_report, "timed", queue_size, run, point_id);
                    const bool capture_profile_range =
                        cuda_profiler_capture_range_enabled() && run == 0;
                    perf_trace::Capture performance_capture("rabitq", point_id, "timed", queue_size, run);
                    if (capture_profile_range) {
                        check_cuda(cudaProfilerStart(),
                                   "cudaProfilerStart RabitQ benchmark timed run");
                    }
                    const uint64_t timed_start_wall_ns = wall_time_ns();
                    std::cout << "[benchmark timed-run start] backend=rabitq queue_size="
                              << queue_size << " run=" << run << '/'
                              << iteration_options.timed_runs
                              << " wall_time_ns=" << timed_start_wall_ns << "\n";
                    std::cout.flush();
                    RabitqIterationResult result = execute_benchmark_iteration(false);
                    perf_trace::value("reported_search_wall_ms", &runtime, result.wall_ms);
                    const uint64_t timed_end_wall_ns = wall_time_ns();
                    if (capture_profile_range) {
                        check_cuda(cudaProfilerStop(),
                                   "cudaProfilerStop RabitQ benchmark timed run");
                    }

                    const RabitqProcessPageFaults faults_after =
                        snapshot_process_page_faults();
                    const RabitqCpuWorkerBudgetStats budget_after =
                        cpu_worker_budget->snapshot_stats();
                    cpu_profile_iteration.finish();
                    performance_capture.finish();
                    std::ostringstream row;
                    row << std::setprecision(17)
                        << csv_field(point_id) << ",timed," << run << ','
                        << result.wall_ms << ',' << result.qps << ','
                        << result.query_count << ',' << result.total_steps << ','
                        << result.highfreq_total_steps << ',' << result.lowfreq_total_steps << ','

                        << csv_field(point_results_path) << ','
                        << csv_field(point_steps_path) << ','
                        << csv_field(point_routes_path) << ','
                        << counter_delta(faults_after.minor, faults_before.minor) << ','
                        << counter_delta(faults_after.major, faults_before.major) << ','
                        << counter_delta(budget_after.acquisitions,
                                         budget_before.acquisitions) << ','
                        << counter_delta(budget_after.contended_acquisitions,
                                         budget_before.contended_acquisitions) << ','
                        << static_cast<double>(counter_delta(budget_after.wait_ns,
                                                             budget_before.wait_ns)) /
                               1.0e6 << ','
                        << budget_after.peak_active_workers;
                    timed_runs_csv.append_row(row.str());
                    report_iteration_preparation(setup_report, point_id, "timed", run,
                                                 result, benchmark_setup_start);
                    std::cout << "[benchmark timed-run end] backend=rabitq queue_size="
                              << queue_size << " run=" << run << '/'
                              << iteration_options.timed_runs
                              << " wall_ms=" << result.wall_ms
                              << " qps=" << result.qps
                              << " pre_qps_ms=" << result.pre_qps_total_ms
                              << " pre_qps_already_in_wall=" << result.pre_qps_counted_in_wall
                              << " route_merge_ms=" << result.route_merge_wall_ms
                              << " route_merge_strategy="
                              << lowfreq_merge::final_merge_strategy_name(
                                     result.route_merge_strategy)
                              << " queries=" << result.query_count
                              << " minor_faults="
                              << counter_delta(faults_after.minor, faults_before.minor)
                              << " major_faults="
                              << counter_delta(faults_after.major, faults_before.major)
                              << " cpu_worker_acquisitions="
                              << counter_delta(budget_after.acquisitions,
                                               budget_before.acquisitions)
                              << " cpu_worker_contended="
                              << counter_delta(budget_after.contended_acquisitions,
                                               budget_before.contended_acquisitions)
                              << " cpu_worker_wait_ms="
                              << static_cast<double>(counter_delta(budget_after.wait_ns,
                                                                   budget_before.wait_ns)) /
                                     1.0e6
                              << " cpu_worker_peak_active="
                              << budget_after.peak_active_workers
                              << " wall_time_ns=" << timed_end_wall_ns << '\n';
                    std::cout.flush();
                    aggregate.qps += result.qps;
                    aggregate.kernel_ms += result.kernel_ms;
                    aggregate.total_steps += result.total_steps;
                    aggregate.highfreq_route_count = result.highfreq_route_count;
                    aggregate.lowfreq_route_count = result.lowfreq_route_count;
                    aggregate.provider_route_count = result.provider_route_count;
                    aggregate.unroutable_route_count = result.unroutable_route_count;
                    aggregate.highfreq_total_steps += result.highfreq_total_steps;
                    aggregate.lowfreq_total_steps += result.lowfreq_total_steps;
                    aggregate.pass_ratio += result.pass_ratio;
                    aggregate.workers_thread = std::max(aggregate.workers_thread, result.workers_thread);
                    accumulate_host_filter_stats(aggregate.host_filter_stats, result.host_filter_stats);
                    if (run + 1 == iteration_options.timed_runs) {
                        aggregate.results = std::move(result.results);
                        aggregate.step_nums = std::move(result.step_nums);
                        aggregate.route_flags = std::move(result.route_flags);
                    }
                }
                const int timed_runs = std::max(1, iteration_options.timed_runs);
                aggregate.qps /= static_cast<double>(timed_runs);
                aggregate.kernel_ms /= static_cast<double>(timed_runs);
                aggregate.total_steps /= timed_runs;
                aggregate.highfreq_total_steps /= timed_runs;
                aggregate.lowfreq_total_steps /= timed_runs;
                aggregate.pass_ratio /= static_cast<float>(timed_runs);
                const double avg_step_num = runtime.query_vectors.rows > 0
                    ? static_cast<double>(aggregate.total_steps) / static_cast<double>(runtime.query_vectors.rows)
                    : 0.0;
                const double highfreq_avg_step_num =
                    average_route_steps(aggregate.highfreq_total_steps, aggregate.highfreq_route_count);
                const double lowfreq_avg_step_num =
                    average_route_steps(aggregate.lowfreq_total_steps, aggregate.lowfreq_route_count);
                std::cout << "[benchmark] quick_filter_bonus=" << quick_filter_bonus
                          << " queue_size=" << queue_size
                          << " finish_count=" << finish_count
                          << " qps=" << aggregate.qps
                          << " kernel_ms=" << aggregate.kernel_ms
                          << " avg_step_num=" << avg_step_num
                          << " highfreq_avg_step_num=" << highfreq_avg_step_num
                          << " lowfreq_avg_step_num=" << lowfreq_avg_step_num
                          << " pass_ratio=" << aggregate.pass_ratio
                          << " workers_thread=" << aggregate.workers_thread
                          << "\n";
                std::cout.flush();
                if (iteration_options.host_demand_batch_size > 1) {
                    print_host_filter_summary("rabitq", aggregate.host_filter_stats);
                }
                csv << quick_filter_bonus << ','
                    << queue_size << ','
                    << finish_count << ','
                    << aggregate.qps << ','
                    << aggregate.kernel_ms << ','
                    << aggregate.total_steps << ','
                    << avg_step_num << ','
                    << aggregate.highfreq_route_count << ','
                    << aggregate.lowfreq_route_count << ','
                    << aggregate.provider_route_count << ','
                    << aggregate.unroutable_route_count << ','
                    << aggregate.highfreq_total_steps << ','
                    << highfreq_avg_step_num << ','
                    << aggregate.lowfreq_total_steps << ','
                    << lowfreq_avg_step_num << ','
                    << aggregate.pass_ratio << ','
                    << aggregate.workers_thread << '\n';

                save_results_to_csv(aggregate.results, point_results_path);
                save_step_nums_to_csv(aggregate.step_nums, point_steps_path);
                save_route_flags_to_csv(aggregate.route_flags, point_routes_path);
                csv.flush();

            }
        }
    } catch (...) {

        cleanup_range_graph_contexts();
        cleanup_graph_context(graph_ctx);
        cleanup_lowfreq_context(lowfreq_ctx);
        cleanup_range_posting_context(range_posting_ctx);
        cleanup_prepared_query_state(prepared_state);
        cleanup_immutable_state(immutable_state);
        throw;
    }

    cleanup_range_graph_contexts();
    cleanup_graph_context(graph_ctx);
    cleanup_lowfreq_context(lowfreq_ctx);
    cleanup_range_posting_context(range_posting_ctx);
    cleanup_prepared_query_state(prepared_state);
    cleanup_immutable_state(immutable_state);
    return 0;
}

}
