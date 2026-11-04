#include "execution/async_io.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <unordered_set>

#include "predicates/range_filter.cuh"
#include "planning/route_predicate_plan.h"

namespace {

size_t round_up_to_multiple(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

bool& full_filter_override_storage()
{
    static bool enabled = std::getenv("GO_FANN_FORCE_FULL_FILTER") != nullptr;
    return enabled;
}

bool host_stats_enabled()
{
    static const bool enabled = std::getenv("GO_FANN_FULL_DEBUG_STATS") != nullptr &&
        std::getenv("GO_FANN_DISABLE_HOST_STATS") == nullptr;
    return enabled;
}

bool host_stats_collection_enabled()
{
    static const bool enabled = std::getenv("GO_FANN_DISABLE_HOST_STATS_COLLECTION") == nullptr;
    return enabled;
}

bool rabitq_graph_profile_enabled()
{
    static const bool enabled =
        std::getenv("GO_FANN_RABITQ_GRAPH_PROFILE") != nullptr;
    return enabled;
}

uint32_t rabitq_graph_profile_sample_log2()
{
    static const uint32_t sample_log2 = [] {
        constexpr uint32_t kDefaultSampleLog2 = 6;
        constexpr uint32_t kMaxSampleLog2 = 20;
        const char* value =
            std::getenv("GO_FANN_RABITQ_GRAPH_PROFILE_SAMPLE_LOG2");
        if (value == nullptr || value[0] == '\0') return kDefaultSampleLog2;
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value || *end != '\0' || parsed > kMaxSampleLog2) {
            return kDefaultSampleLog2;
        }
        return static_cast<uint32_t>(parsed);
    }();
    return sample_log2;
}

int64_t load_mailbox_word_host(const int64_t* base, size_t offset)
{
    const volatile int64_t* ptr = reinterpret_cast<const volatile int64_t*>(base + offset);
    return *ptr;
}

enum class SyncPublishCheckResult : uint8_t {
    Ok = 0,
    StatusNotWaiting,
    GenerationMismatch,
    RowMismatch,
    BufferMismatch,
    MailboxNotZero,
};

enum class SyncCheckMode : uint8_t {
    Full = 0,
    NoMailbox = 1,
    GenerationOnly = 2,
    Disabled = 3,
};

SyncCheckMode sync_check_mode()
{
    if (std::getenv("GO_FANN_FULL_DEBUG_STATS") != nullptr) {
        return SyncCheckMode::Full;
    }
    const char* value = std::getenv("GO_FANN_SYNC_CHECK_MODE");
    if (value == nullptr) {
        return SyncCheckMode::NoMailbox;
    }
    if (std::strcmp(value, "no_mailbox") == 0) {
        return SyncCheckMode::NoMailbox;
    }
    if (std::strcmp(value, "generation_only") == 0) {
        return SyncCheckMode::GenerationOnly;
    }
    if (std::strcmp(value, "off") == 0 || std::strcmp(value, "disabled") == 0) {
        return SyncCheckMode::Disabled;
    }
    return SyncCheckMode::Full;
}

SyncPublishCheckResult sync_publish_check(int64_t* status_memory,
                                          int64_t* status_flag,
                                          int64_t* h_gdr_data,
                                          size_t query_idx,
                                          uint32_t generation,
                                          uint32_t local_row,
                                          uint32_t buffer_idx,
                                          const AsyncIoConfig& config)
{
    const SyncCheckMode mode = config.completion_sink != nullptr
        ? SyncCheckMode::Full
        : sync_check_mode();
    if (mode == SyncCheckMode::Disabled) {
        return SyncPublishCheckResult::Ok;
    }
    if (load_mailbox_word_host(status_flag, query_idx * 16) != 1) {
        return SyncPublishCheckResult::StatusNotWaiting;
    }
    if (static_cast<uint32_t>(load_mailbox_word_host(status_memory, query_idx * 16 + 2)) != generation) {
        return SyncPublishCheckResult::GenerationMismatch;
    }
    if (mode == SyncCheckMode::GenerationOnly) {
        return SyncPublishCheckResult::Ok;
    }
    if (load_mailbox_word_host(status_memory, query_idx * 16) != static_cast<int64_t>(local_row)) {
        return SyncPublishCheckResult::RowMismatch;
    }
    const uint32_t current_buffer_idx = config.payload_slot_count > 1
        ? static_cast<uint32_t>(load_mailbox_word_host(status_memory, query_idx * 16 + 1) & 1LL)
        : 0U;
    if (current_buffer_idx != buffer_idx) {
        return SyncPublishCheckResult::BufferMismatch;
    }
    if (mode == SyncCheckMode::NoMailbox) {
        return SyncPublishCheckResult::Ok;
    }
    if (load_mailbox_word_host(h_gdr_data, query_idx * kGdrMailboxStride) != 0) {
        return SyncPublishCheckResult::MailboxNotZero;
    }
    return SyncPublishCheckResult::Ok;
}

void write_in_memory_mailbox_ready(int64_t* h_gdr_data,
                                   size_t query_idx,
                                   int64_t filter_result)
{
    h_gdr_data[query_idx * kGdrMailboxStride] = filter_result;
}

void publish_in_memory_mailbox_ready(int64_t* h_gdr_data,
                                     size_t query_idx,
                                     int64_t filter_result,
                                     uint32_t generation,
                                     uint32_t local_row,
                                     uint32_t buffer_idx,
                                     const AsyncIoConfig& config)
{
    write_in_memory_mailbox_ready(h_gdr_data, query_idx, filter_result);
    if (config.completion_sink != nullptr) {
        _mm_sfence();
        (void)publish_host_ready(config.completion_sink,
                                 static_cast<uint32_t>(query_idx),
                                 generation,
                                 local_row,
                                 buffer_idx);
    }
}

constexpr int64_t kHostHeldStatus = -3;
constexpr size_t kHostServiceBatchQueries = 512;

bool host_hang_debug_enabled()
{
    static const bool enabled = std::getenv("GO_FANN_DEBUG_HANG") != nullptr;
    return enabled;
}

bool graph_runtime_debug_enabled()
{
    static const bool enabled = std::getenv("GOFANN_DEBUG_RUNTIME") != nullptr;
    return enabled;
}

void debug_dump_graph_view(const GraphNodeView& view, const char* stage)
{
    if (!graph_runtime_debug_enabled()) {
        return;
    }
    static std::atomic<uint32_t> dump_count{0};
    const uint32_t dump_id = dump_count.fetch_add(1, std::memory_order_relaxed);
    if (dump_id >= 2) {
        return;
    }

    std::fprintf(stdout,
                 "[runtime-trace][view] stage=%s label_count=%u total_rows=%u graph_rows=%llu graph_degree=%llu base_rows=%u query_group_ptr=%p label_size_ptr=%p label_offset_ptr=%p row_to_sample_ptr=%p\n",
                 stage,
                 view.label_count,
                 view.total_rows,
                 static_cast<unsigned long long>(view.graph_rows),
                 static_cast<unsigned long long>(view.graph_degree),
                 view.base_vector_rows,
                 static_cast<const void*>(view.query_group_ids),
                 static_cast<const void*>(view.label_size),
                 static_cast<const void*>(view.label_offset),
                 static_cast<const void*>(view.row_to_sample));
    const uint32_t sample_count = std::min<uint32_t>(view.label_count, 8);
    for (uint32_t group_id = 0; group_id < sample_count; ++group_id) {
        std::fprintf(stdout,
                     "[runtime-trace][view] group=%u label_size=%u label_offset=%u\n",
                     group_id,
                     view.label_size != nullptr ? view.label_size[group_id] : UINT32_MAX,
                     view.label_offset != nullptr ? view.label_offset[group_id] : UINT32_MAX);
    }
    std::fflush(stdout);
}

bool claim_host_batch(size_t loop_query_count,
                      size_t batch_size,
                      int worker_idx,
                      std::atomic<size_t>& next_batch_idx,
                      bool use_static_batch,
                      size_t& batch_begin,
                      size_t& batch_end)
{
    const size_t batch_idx = use_static_batch
        ? static_cast<size_t>(worker_idx)
        : next_batch_idx.fetch_add(1, std::memory_order_relaxed);
    batch_begin = batch_idx * batch_size;
    if (batch_begin >= loop_query_count) {
        return false;
    }
    batch_end = std::min(loop_query_count, batch_begin + batch_size);
    return true;
}

bool full_filter_debug_enabled()
{
    return full_filter_override_storage();
}

bool host_full_label_check_enabled(const GraphNodeView& view,
                                   const AsyncIoConfig& config)
{
    return config.enable_full_label_check &&
        (full_filter_debug_enabled() || view.hybrid_residual_runtime != nullptr);
}

bool host_query_matches_sample(const GraphNodeView& view,
                               uint32_t sample_id,
                               uint32_t query_idx)
{
    if (view.runtime == nullptr) {
        return true;
    }
    if (view.hybrid_residual_runtime != nullptr) {
        if (query_idx >= view.runtime->route_to_original_query.size()) {
            return false;
        }
        return sample_matches_hybrid_range_branch_labels(
            *view.hybrid_residual_runtime,
            view.runtime->route_to_original_query[query_idx],
            sample_id);
    }
    return sample_matches_lowfreq_full_labels(*view.runtime, query_idx, sample_id);
}

bool host_query_matches_ranges(const GraphNodeView& view, uint32_t sample_id, uint32_t query_idx)
{
    if (view.runtime == nullptr) {
        return true;
    }
    const GraphRuntime& runtime = *view.runtime;
    if (runtime.query_ranges.range_count != runtime.base_range_count) {
        return false;
    }
    if (query_idx >= runtime.query_ranges.rows) {
        return false;
    }
    const size_t query_range_count = static_cast<size_t>(runtime.base_range_count);
    const size_t query_flat_end = static_cast<size_t>(query_idx) * query_range_count + query_range_count;
    if (query_flat_end > runtime.query_ranges.modes.size() ||
        query_flat_end > runtime.query_ranges.lower_bounds.size() ||
        query_flat_end > runtime.query_ranges.upper_bounds.size()) {
        return false;
    }

    const bool has_active_range = go_fann::range_filter::query_has_active_range(
        runtime.query_ranges.modes.data(),
        query_idx,
        runtime.base_range_count);
    if (!has_active_range) {
        return true;
    }

    if (sample_id >= runtime.base_vectors.rows) {
        return false;
    }
    const size_t base_flat_begin = static_cast<size_t>(sample_id) * query_range_count;
    if (base_flat_begin + query_range_count > runtime.base_range_values.size()) {
        return false;
    }

    return go_fann::range_filter::sample_matches_all_active_ranges_host(runtime.query_ranges,
                                                                              query_idx,
                                                                              runtime.base_range_values,
                                                                              runtime.base_range_count,
                                                                              sample_id);
}

bool host_candidate_matches_route(const GraphRuntime& runtime,
                                  const HostCandidateSourceView& source,
                                  uint32_t sample_id)
{
    // A physical source may own several bounded DNF clauses. Its flat
    // representative residual is not a conjunction required by every branch.
    return sample_matches_route_predicate(runtime, source.route_id, sample_id);
}

[[noreturn]] void fail_graph_runtime(const char* message,
                                       uint32_t label,
                                       uint32_t local_row)
{
    std::fprintf(stderr,
                 "graph runtime runtime invariant failed: %s (label=%u, local_row=%u)\n",
                 message,
                 label,
                 local_row);
    std::abort();
}

}

void set_full_filter_enabled(bool enabled)
{
    full_filter_override_storage() = enabled;
}

bool full_filter_enabled()
{
    return full_filter_override_storage();
}

void maybe_bind_current_thread(int cpu_start, int cpu_count)
{
    if (cpu_start < 0 || cpu_count <= 0) {
        return;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int cpu = cpu_start; cpu < cpu_start + cpu_count; ++cpu) {
        CPU_SET(cpu, &cpuset);
    }
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void maybe_bind_worker_thread(const AsyncIoConfig& config, int worker_idx)
{
    if (config.host_service_cpu_start < 0 || config.host_service_cpu_count <= 0) {
        return;
    }
    const int cpu = config.host_service_cpu_start + (worker_idx % config.host_service_cpu_count);
    maybe_bind_current_thread(cpu, 1);
}

struct PendingMemoryDemand {
    size_t query_idx = 0;
    uint32_t generation = 0;
    uint32_t local_row = 0;
    uint32_t buffer_idx = 0;
    std::chrono::steady_clock::time_point observed_at;
};

uint64_t pending_memory_demand_age_ns(const std::vector<PendingMemoryDemand>& demands)
{
    if (demands.empty()) return 0;
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - demands.front().observed_at).count());
}

uint64_t host_steady_now_ns()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool append_pending_memory_demand(size_t query_idx,
                                  int64_t row_value,
                                  int64_t buffer_value,
                                  int64_t generation_value,
                                  int64_t* h_gdr_data,
                                  const GraphNodeView& host_view,
                                  const AsyncIoConfig& config,
                                  std::vector<uint32_t>& last_seen_generation,
                                  std::vector<uint32_t>& last_seen_row,
                                  std::vector<uint32_t>& last_seen_buffer,
                                  std::vector<uint8_t>& seen_pending,
                                  HostMailboxSyncStats& sync_stats,
                                  std::vector<PendingMemoryDemand>& demands)
{
    if (row_value < 0) return false;
    const uint32_t generation = generation_value >= 0
        ? static_cast<uint32_t>(generation_value)
        : 0U;
    const uint32_t local_row = static_cast<uint32_t>(row_value);
    const uint32_t buffer_idx = config.payload_slot_count > 1
        ? static_cast<uint32_t>(buffer_value & 1LL)
        : 0U;
    if (seen_pending[query_idx] != 0 &&
        last_seen_generation[query_idx] == generation &&
        last_seen_row[query_idx] == local_row &&
        last_seen_buffer[query_idx] == buffer_idx) {
        if (host_stats_collection_enabled()) {
            ++sync_stats.duplicate_observations;
        }
        return false;
    }
    last_seen_generation[query_idx] = generation;
    last_seen_row[query_idx] = local_row;
    last_seen_buffer[query_idx] = buffer_idx;
    seen_pending[query_idx] = 1;

    const uint32_t group_id = host_view.query_group_ids[query_idx];
    if (group_id >= host_view.label_count) {
        std::fprintf(stderr,
                     "graph runtime runtime invariant failed: group_id out of range (group_id=%u, query=%zu, label_count=%u)\n",
                     group_id,
                     query_idx,
                     host_view.label_count);
        std::fflush(stderr);
        std::abort();
    }
    if (local_row >= host_view.label_size[group_id]) {
        h_gdr_data[query_idx * kGdrMailboxStride] = 2;
        _mm_sfence();
        return false;
    }
    demands.push_back({query_idx,
                       generation,
                       local_row,
                       buffer_idx,
                       std::chrono::steady_clock::now()});
    return true;
}

size_t flush_pending_memory_demands(std::vector<PendingMemoryDemand>& demands,
                                    int64_t* status_memory,
                                    uint32_t* h_neighbor_memory,
                                    int64_t* h_gdr_data,
                                    int64_t* status_flag,
                                    const GraphNodeView& host_view,
                                    const AsyncIoConfig& config,
                                    HostFilterStats* filter_stats,
                                    HostMailboxSyncStats& sync_stats,
                                    bool collect_request_ring_profile = false)
{
    if (demands.empty()) return 0;

    const uint64_t batch_wait_ns = pending_memory_demand_age_ns(demands);
    if (filter_stats != nullptr) {
        ++filter_stats->demand_batch_count;
        filter_stats->demand_batch_request_count += demands.size();
        filter_stats->demand_batch_wait_ns += batch_wait_ns;
        filter_stats->demand_batch_max_size = std::max<uint64_t>(
            filter_stats->demand_batch_max_size, demands.size());
        if (demands.size() < config.host_demand_batch_size) {
            ++filter_stats->demand_partial_batch_count;
        }
    }

    std::vector<int64_t> filter_results(demands.size(), 0);
    uint64_t filter_ns = 0;
    uint64_t payload_ns = 0;
    for (size_t idx = 0; idx < demands.size(); ++idx) {
        const uint64_t filter_start_ns = collect_request_ring_profile
            ? host_steady_now_ns()
            : 0;
        const PendingMemoryDemand& demand = demands[idx];
        const uint32_t group_id = host_view.query_group_ids[demand.query_idx];
        const uint32_t global_row = graph_global_row(host_view, group_id, demand.local_row);
        if (global_row >= host_view.total_rows) {
            std::fprintf(stderr,
                         "graph runtime runtime invariant failed: global_row out of range (group_id=%u, local_row=%u, global_row=%u, total_rows=%u)\n",
                         group_id,
                         demand.local_row,
                         global_row,
                         host_view.total_rows);
            std::fflush(stderr);
            std::abort();
        }
        if (filter_stats != nullptr && host_stats_collection_enabled()) {
            ++filter_stats->requests_served;
        }
        const uint32_t sample_id = host_view.row_to_sample[global_row];
        const uint64_t filter_mask = graph_row_filter_mask_at(host_view, global_row);
        const int64_t filter_result = compute_host_filter_result(host_view,
                                                                 sample_id,
                                                                 filter_mask,
                                                                 static_cast<uint32_t>(demand.query_idx),
                                                                 host_full_label_check_enabled(
                                                                     host_view, config), &config);
        filter_results[idx] = filter_result;
        if (filter_stats != nullptr && host_stats_collection_enabled()) {
            if (filter_result == 1) {
                ++filter_stats->requests_passed;
            } else {
                ++filter_stats->requests_filtered;
            }
        }
        if (collect_request_ring_profile) {
            filter_ns += host_steady_now_ns() - filter_start_ns;
        }
        const uint64_t payload_start_ns = collect_request_ring_profile
            ? host_steady_now_ns()
            : 0;
        const size_t payload_slot = demand.query_idx * config.payload_slot_count + demand.buffer_idx;
        uint32_t* dst_words = h_neighbor_memory + payload_slot * config.elems_per_block;
        if (should_publish_graph_vector_payload(config, filter_result == 1)) {
            const uint32_t* graph_row =
                host_view.graph_data + static_cast<size_t>(global_row) * host_view.graph_degree;
            serialize_graph_node_row_words(host_view,
                                             group_id,
                                             demand.local_row,
                                             graph_row,
                                             dst_words,
                                             config,
                                             static_cast<uint32_t>(host_view.graph_degree),
                                             filter_stats);
        } else {
            const uint32_t* graph_row =
                host_view.graph_data + static_cast<size_t>(global_row) * host_view.graph_degree;
            serialize_graph_neighbor_row_words(host_view,
                                                 group_id,
                                                 demand.local_row,
                                                 graph_row,
                                                 dst_words,
                                                 config,
                                                 static_cast<uint32_t>(host_view.graph_degree));
        }
        if (collect_request_ring_profile) {
            payload_ns += host_steady_now_ns() - payload_start_ns;
        }
    }

    const uint64_t publish_start_ns = collect_request_ring_profile
        ? host_steady_now_ns()
        : 0;
    _mm_sfence();
    std::vector<HostCompletionEvent> ready_events;
    ready_events.reserve(demands.size());
    const uint64_t published_ns = host_steady_now_ns();
    for (size_t idx = 0; idx < demands.size(); ++idx) {
        const PendingMemoryDemand& demand = demands[idx];
        const SyncPublishCheckResult check = sync_publish_check(status_memory,
                                                                status_flag,
                                                                h_gdr_data,
                                                                demand.query_idx,
                                                                demand.generation,
                                                                demand.local_row,
                                                                demand.buffer_idx,
                                                                config);
        switch (check) {
        case SyncPublishCheckResult::Ok:
            write_in_memory_mailbox_ready(h_gdr_data, demand.query_idx, filter_results[idx]);
            ready_events.push_back({HostCompletionKind::Ready,
                                    static_cast<uint32_t>(demand.query_idx),
                                    demand.generation,
                                    demand.local_row,
                                    demand.buffer_idx,
                                    published_ns});
            break;
        case SyncPublishCheckResult::StatusNotWaiting:
            if (host_stats_collection_enabled()) ++sync_stats.stale_status_not_waiting;
            break;
        case SyncPublishCheckResult::GenerationMismatch:
            if (host_stats_collection_enabled()) ++sync_stats.stale_generation_mismatch;
            break;
        case SyncPublishCheckResult::RowMismatch:
            if (host_stats_collection_enabled()) ++sync_stats.stale_row_mismatch;
            break;
        case SyncPublishCheckResult::BufferMismatch:
            if (host_stats_collection_enabled()) ++sync_stats.stale_buffer_mismatch;
            break;
        case SyncPublishCheckResult::MailboxNotZero:
            if (host_stats_collection_enabled()) ++sync_stats.stale_mailbox_not_zero;
            break;
        }
    }
    if (!ready_events.empty() && config.completion_sink != nullptr) {
        _mm_sfence();
        if (!publish_host_ready_batch(config.completion_sink,
                                      ready_events.data(),
                                      ready_events.size())) {
            std::fprintf(stderr, "host demand microbatch completion publication failed\n");
            std::fflush(stderr);
            std::abort();
        }
    }
    if (collect_request_ring_profile && filter_stats != nullptr) {
        ++filter_stats->request_ring_profile_samples;
        filter_stats->request_ring_filter_ns += filter_ns;
        filter_stats->request_ring_payload_ns += payload_ns;
        filter_stats->request_ring_publish_ns +=
            host_steady_now_ns() - publish_start_ns;
    }
    const size_t published_count = ready_events.size();
    demands.clear();
    return published_count;
}

void host_scan_and_set_worker(int64_t* status_memory,
                              uint32_t* h_neighbor_memory,
                              int64_t* h_gdr_data,
                              int* req_per_loop,
                              int64_t* status_flag,
                              const GraphNodeView& host_view,
                              size_t total_queries,
                              const uint32_t* active_query_ids,
                              size_t active_query_count,
                              int worker_idx,
                              int worker_count,
                              std::atomic<size_t>* next_batch_idx,
                              const AsyncIoConfig& config,
                              HostFilterStats* filter_stats,
                              bool record_profile)
{
    maybe_bind_worker_thread(config, worker_idx);
    if (host_hang_debug_enabled()) {
        std::printf("[host-debug] worker=%d start assigned_query_slots=pending\n", worker_idx);
        std::fflush(stdout);
    }

    const size_t loop_query_count = active_query_ids != nullptr ? active_query_count : total_queries;
    const size_t batch_size = std::min(kHostServiceBatchQueries, std::max<size_t>(1, loop_query_count));
    size_t total_assigned_queries = 0;
    size_t total_submissions_all = 0;
    HostMailboxSyncStats sync_stats{};
    if (filter_stats != nullptr) {
        filter_stats->demand_batch_configured_size = config.host_demand_batch_size;
        filter_stats->demand_batch_timeout_us = config.host_demand_batch_timeout_us;
    }
    if (host_hang_debug_enabled()) {
        std::printf("[host-debug] worker=%d batch_size=%zu loop_query_count=%zu worker_count=%d\n",
                    worker_idx,
                    batch_size,
                    loop_query_count,
                    worker_count);
        std::fflush(stdout);
    }

    size_t batch_begin = 0;
    size_t batch_end = 0;
    bool use_static_batch = true;
    int j = 0;
    while (claim_host_batch(loop_query_count,
                            batch_size,
                            worker_idx,
                            *next_batch_idx,
                            use_static_batch,
                            batch_begin,
                            batch_end)) {
        use_static_batch = false;
        total_assigned_queries += (batch_end - batch_begin);
        std::vector<bool> query_finished(batch_end - batch_begin, false);
        std::vector<uint32_t> last_seen_generation(total_queries, UINT32_MAX);
        std::vector<uint32_t> last_seen_row(total_queries, UINT32_MAX);
        std::vector<uint32_t> last_seen_buffer(total_queries, UINT32_MAX);
        std::vector<uint8_t> seen_pending(total_queries, 0);
        size_t finished_count = 0;
        size_t total_submissions = 0;
        size_t no_progress_rounds = 0;
        std::vector<PendingMemoryDemand> pending_demands;
        if (config.host_demand_batch_size > 1) {
            pending_demands.reserve(config.host_demand_batch_size);
        }

        if (host_hang_debug_enabled()) {
            std::printf("[host-debug] worker=%d batch=[%zu,%zu) size=%zu\n",
                        worker_idx,
                        batch_begin,
                        batch_end,
                        batch_end - batch_begin);
            std::fflush(stdout);
        }

        while (finished_count < (batch_end - batch_begin)) {
            j++;
            auto start = std::chrono::high_resolution_clock::now();
            int submitted_this_loop = 0;
            const size_t finished_before = finished_count;

            for (size_t logical_idx = batch_begin; logical_idx < batch_end; ++logical_idx) {
                const size_t batch_idx = logical_idx - batch_begin;
                if (query_finished[batch_idx]) {
                    continue;
                }

                const size_t i = active_query_ids != nullptr
                    ? static_cast<size_t>(active_query_ids[logical_idx])
                    : logical_idx;

                volatile int64_t* flag_ptr = &status_flag[i * 16];
                const int64_t flag = *flag_ptr;

                if (graph_status_flag_finished(flag)) {
                    seen_pending[i] = 0;
                    query_finished[batch_idx] = true;
                    ++finished_count;
                    (void)publish_host_done(config.completion_sink,
                                            static_cast<uint32_t>(i));
                    if (host_hang_debug_enabled() && finished_count % 1000 == 0) {
                        std::printf("[host-debug] worker=%d finished=%zu/%zu\n",
                                    worker_idx,
                                    finished_count,
                                    batch_end - batch_begin);
                        std::fflush(stdout);
                    }
                    continue;
                }

                if (flag == 1) {
                    volatile int64_t* status_ptr = &status_memory[i * 16];
                    volatile int64_t* buffer_ptr = &status_memory[i * 16 + 1];
                    volatile int64_t* generation_ptr = &status_memory[i * 16 + 2];
                    const int64_t val = *status_ptr;
                    const int64_t buffer_val = *buffer_ptr;
                    const int64_t generation_val = *generation_ptr;
                    if (val >= 0) {
                        const uint32_t generation = generation_val >= 0 ? static_cast<uint32_t>(generation_val) : 0U;
                        const uint32_t local_row = static_cast<uint32_t>(val);
                        const uint32_t buffer_idx = config.payload_slot_count > 1
                            ? static_cast<uint32_t>(buffer_val & 1LL)
                            : 0U;
                        if (config.host_demand_batch_size > 1) {
                            (void)append_pending_memory_demand(i,
                                                               val,
                                                               buffer_val,
                                                               generation_val,
                                                               h_gdr_data,
                                                               host_view,
                                                               config,
                                                               last_seen_generation,
                                                               last_seen_row,
                                                               last_seen_buffer,
                                                               seen_pending,
                                                               sync_stats,
                                                               pending_demands);
                            const uint64_t wait_ns = pending_memory_demand_age_ns(pending_demands);
                            const uint64_t timeout_ns = static_cast<uint64_t>(
                                config.host_demand_batch_timeout_us) * 1000ULL;
                            if (host_demand_batch_should_flush(pending_demands.size(),
                                                               config.host_demand_batch_size,
                                                               wait_ns,
                                                               timeout_ns)) {
                                const size_t published = flush_pending_memory_demands(
                                    pending_demands,
                                    status_memory,
                                    h_neighbor_memory,
                                    h_gdr_data,
                                    status_flag,
                                    host_view,
                                    config,
                                    filter_stats,
                                    sync_stats);
                                submitted_this_loop += static_cast<int>(published);
                                total_submissions += published;
                            }
                            continue;
                        }
                        if (seen_pending[i] != 0 &&
                            last_seen_generation[i] == generation &&
                            last_seen_row[i] == local_row &&
                            last_seen_buffer[i] == buffer_idx) {
                            if (host_stats_collection_enabled()) {
                                ++sync_stats.duplicate_observations;
                            }
                        }
                        last_seen_generation[i] = generation;
                        last_seen_row[i] = local_row;
                        last_seen_buffer[i] = buffer_idx;
                        seen_pending[i] = 1;
                        const uint32_t group_id = host_view.query_group_ids[i];
                        if (group_id >= host_view.label_count) {
                            std::fprintf(stderr,
                                         "graph runtime runtime invariant failed: group_id out of range (group_id=%u, query=%zu, label_count=%u)\n",
                                         group_id,
                                         i,
                                         host_view.label_count);
                            std::fflush(stderr);
                            std::abort();
                        }
                        const uint32_t label_size = host_view.label_size[group_id];
                        if (local_row >= label_size) {
                            h_gdr_data[i * kGdrMailboxStride] = 2;
                            _mm_sfence();
                            continue;
                        }
                        if (filter_stats != nullptr) {
                            if (host_stats_collection_enabled()) {
                                ++filter_stats->requests_served;
                            }
                        }
                        const uint32_t global_row = graph_global_row(host_view, group_id, local_row);
                        if (global_row >= host_view.total_rows) {
                            std::fprintf(stderr,
                                         "graph runtime runtime invariant failed: global_row out of range (group_id=%u, local_row=%u, global_row=%u, total_rows=%u)\n",
                                         group_id,
                                         local_row,
                                         global_row,
                                         host_view.total_rows);
                            std::fflush(stderr);
                            std::abort();
                        }
                        const uint32_t sample_id = host_view.row_to_sample[global_row];
                        const uint64_t filter_mask =
                            graph_row_filter_mask_at(host_view, global_row);
                        const int64_t filter_result = compute_host_filter_result(host_view,
                                                                                 sample_id,
                                                                                 filter_mask,
                                                                                 static_cast<uint32_t>(i),
                                                                                 host_full_label_check_enabled(
                                                                                     host_view, config), &config);
                        if (filter_stats != nullptr) {
                            if (filter_result == 1) {
                                if (host_stats_collection_enabled()) {
                                    ++filter_stats->requests_passed;
                                }
                            } else {
                                if (host_stats_collection_enabled()) {
                                    ++filter_stats->requests_filtered;
                                }
                            }
                        }
                        const size_t payload_slot = static_cast<size_t>(i) * config.payload_slot_count +
                            static_cast<size_t>(buffer_idx);
                        uint32_t* dst_words =
                            h_neighbor_memory + payload_slot * config.elems_per_block;
                        if (should_publish_graph_vector_payload(config, filter_result == 1)) {
                            const uint32_t* graph_row =
                                host_view.graph_data + static_cast<size_t>(global_row) * host_view.graph_degree;
                            serialize_graph_node_row_words(
                                host_view,
                                group_id,
                                local_row,
                                graph_row,
                                dst_words,
                                config,
                                static_cast<uint32_t>(host_view.graph_degree),
                                filter_stats);
                        } else {
                            const uint32_t* graph_row =
                                host_view.graph_data + static_cast<size_t>(global_row) * host_view.graph_degree;
                            serialize_graph_neighbor_row_words(host_view,
                                                                 group_id,
                                                                 local_row,
                                                                 graph_row,
                                                                 dst_words,
                                                                 config,
                                                                 static_cast<uint32_t>(host_view.graph_degree));
                        }
                        _mm_sfence();
                        const SyncPublishCheckResult check = sync_publish_check(status_memory,
                                                                                status_flag,
                                                                                h_gdr_data,
                                                                                i,
                                                                                generation,
                                                                                local_row,
                                                                                buffer_idx,
                                                                                config);
                        switch (check) {
                        case SyncPublishCheckResult::Ok:
                            publish_in_memory_mailbox_ready(h_gdr_data,
                                                           i,
                                                           filter_result,
                                                           generation,
                                                           local_row,
                                                           buffer_idx,
                                                           config);
                            submitted_this_loop++;
                            ++total_submissions;
                            break;
                        case SyncPublishCheckResult::StatusNotWaiting:
                            if (host_stats_collection_enabled()) {
                                ++sync_stats.stale_status_not_waiting;
                            }
                            break;
                        case SyncPublishCheckResult::GenerationMismatch:
                            if (host_stats_collection_enabled()) {
                                ++sync_stats.stale_generation_mismatch;
                            }
                            break;
                        case SyncPublishCheckResult::RowMismatch:
                            if (host_stats_collection_enabled()) {
                                ++sync_stats.stale_row_mismatch;
                            }
                            break;
                        case SyncPublishCheckResult::BufferMismatch:
                            if (host_stats_collection_enabled()) {
                                ++sync_stats.stale_buffer_mismatch;
                            }
                            break;
                        case SyncPublishCheckResult::MailboxNotZero:
                            if (host_stats_collection_enabled()) {
                                ++sync_stats.stale_mailbox_not_zero;
                            }
                            break;
                        }
                    }
                }
            }

            if (!pending_demands.empty()) {
                const uint64_t wait_ns = pending_memory_demand_age_ns(pending_demands);
                const uint64_t timeout_ns = static_cast<uint64_t>(
                    config.host_demand_batch_timeout_us) * 1000ULL;
                if (host_demand_batch_should_flush(pending_demands.size(),
                                                   config.host_demand_batch_size,
                                                   wait_ns,
                                                   timeout_ns)) {
                    const size_t published = flush_pending_memory_demands(pending_demands,
                                                                          status_memory,
                                                                          h_neighbor_memory,
                                                                          h_gdr_data,
                                                                          status_flag,
                                                                          host_view,
                                                                          config,
                                                                          filter_stats,
                                                                          sync_stats);
                    submitted_this_loop += static_cast<int>(published);
                    total_submissions += published;
                }
            }

            if (record_profile) {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::micro> elapsed = end - start;
                if (j < 10000 && submitted_this_loop > 0) {
                    req_per_loop[j * 2] = static_cast<int>(elapsed.count());
                    req_per_loop[j * 2 + 1] = submitted_this_loop;
                }
            }

            if (host_hang_debug_enabled()) {
                if (submitted_this_loop == 0 && finished_count == finished_before) {
                    ++no_progress_rounds;
                    if (no_progress_rounds % 200 == 0) {
                        size_t stuck_query = loop_query_count;
                        for (size_t logical_idx = batch_begin; logical_idx < batch_end; ++logical_idx) {
                            const size_t batch_idx = logical_idx - batch_begin;
                            if (!query_finished[batch_idx]) {
                                stuck_query = active_query_ids != nullptr
                                    ? static_cast<size_t>(active_query_ids[logical_idx])
                                    : logical_idx;
                                break;
                            }
                        }
                        std::printf("[host-debug] worker=%d no progress rounds=%zu finished=%zu/%zu stuck_query=%zu status=%lld row=%lld buffer=%lld gen=%lld mailbox=%lld\n",
                                    worker_idx,
                                    no_progress_rounds,
                                    finished_count,
                                    batch_end - batch_begin,
                                    stuck_query,
                                    stuck_query < total_queries ? static_cast<long long>(status_flag[stuck_query * 16]) : -1LL,
                                    stuck_query < total_queries ? static_cast<long long>(status_memory[stuck_query * 16]) : -1LL,
                                    stuck_query < total_queries ? static_cast<long long>(status_memory[stuck_query * 16 + 1]) : -1LL,
                                    stuck_query < total_queries ? static_cast<long long>(status_memory[stuck_query * 16 + 2]) : -1LL,
                                    stuck_query < total_queries ? static_cast<long long>(h_gdr_data[stuck_query * kGdrMailboxStride]) : -1LL);
                        std::fflush(stdout);
                    }
                } else {
                    no_progress_rounds = 0;
                }
            }
        }
        if (!pending_demands.empty()) {
            total_submissions += flush_pending_memory_demands(pending_demands,
                                                              status_memory,
                                                              h_neighbor_memory,
                                                              h_gdr_data,
                                                              status_flag,
                                                              host_view,
                                                              config,
                                                              filter_stats,
                                                              sync_stats);
        }
        total_submissions_all += total_submissions;
    }
    if (host_hang_debug_enabled()) {
        std::printf("[host-debug] worker=%d exit finished_queries=%zu submissions=%zu\n",
                    worker_idx,
                    total_assigned_queries,
                    total_submissions_all);
        std::fflush(stdout);
    }
    print_host_sync_summary("PQ", "in-memory(worker)", sync_stats);
}

AsyncIoConfig::AsyncIoConfig(uint32_t dim_value,
                             uint32_t degree_value,
                             uint32_t prefetch_hops_value,
                             size_t block_size_value,
                             size_t header_size_value)
    : dim(dim_value),
      degree(degree_value),
      prefetch_hops(prefetch_hops_value),
      minimum_block_size(block_size_value),
      block_size(block_size_value),
      header_size(header_size_value)
{
    refresh_layout();
}

void AsyncIoConfig::refresh_layout()
{
    vector_payload_words = reserve_vector_payload
        ? (dense_vector_scalar_is_float(vector_payload_scalar_type)
               ? static_cast<size_t>(dim)
               : static_cast<size_t>(dim) / 4)
        : 0;
    sample_id_words = include_sample_ids ? 1 : 0;
    neighbor_count_word = vector_payload_words + sample_id_words;
    neighbor_words_offset = neighbor_count_word + 1;
    neighbor_sample_words_offset = include_sample_ids
        ? neighbor_words_offset + static_cast<size_t>(degree)
        : 0;
    elems_per_node = neighbor_words_offset + static_cast<size_t>(degree);
    if (include_sample_ids) {
        elems_per_node += static_cast<size_t>(degree);
    }
    const size_t payload_bytes = elems_per_node * sizeof(uint32_t);
    block_size = round_up_to_multiple(std::max(payload_bytes, minimum_block_size),
                                      kMailboxSlotAlignmentBytes);
    elems_per_block = block_size / sizeof(uint32_t);
    nodes_per_block = elems_per_node == 0 ? 0 : elems_per_block / elems_per_node;
}

void AsyncIoConfig::set_reserve_vector_payload(bool reserve)
{
    reserve_vector_payload = reserve;
    refresh_layout();
}

void AsyncIoConfig::set_vector_payload_scalar_type(DenseVectorScalarType scalar_type)
{
    vector_payload_scalar_type = scalar_type;
    refresh_layout();
}

GraphNodeView make_graph_node_view(const GraphRuntime& runtime)
{
#ifndef NDEBUG
    validate_graph_runtime_query_metadata(runtime, "make_graph_node_view");
#endif
    GraphNodeView view;
    view.runtime = &runtime;
    view.base_vector_data = runtime.base_vectors.data.data();
    view.base_vector_rows = runtime.base_vectors.rows;
    view.base_vector_cols = runtime.base_vectors.cols;
    view.graph_data = runtime.graph.data.data();
    view.graph_rows = runtime.graph.rows;
    view.graph_degree = runtime.graph.cols;
    view.label_size = runtime.metadata.label_size.data();
    view.label_offset = runtime.metadata.label_offset.data();
    view.row_to_sample = runtime.metadata.row_to_sample.data();
    view.row_filter_mask = runtime.metadata.implicit_all_ones_row_filter_mask
        ? nullptr
        : runtime.metadata.row_filter_mask.data();
    view.implicit_all_ones_row_filter_mask =
        runtime.metadata.implicit_all_ones_row_filter_mask;
    view.query_filter_bits = runtime.query_filter_bits.data();
    view.query_group_ids = runtime.query_group_ids.data();
    view.query_filter_labels = runtime.query_filter_labels.data();
    view.label_to_group = runtime.group_layout.label_to_group.data();
    view.groups = &runtime.group_layout.groups;
    view.data_labels = &runtime.data_labels;
    view.query_labels_matrix = &runtime.query_labels_matrix;
    view.label_count = runtime.metadata.label_count;
    view.total_rows = runtime.metadata.total_rows;
    if (view.label_count == 0 && view.query_group_ids != nullptr) {
        fail_graph_runtime("label_count is 0 while query_group_ids pointer is non-null", 0, 0);
    }
    debug_dump_graph_view(view, "make_graph_node_view");
    return view;
}

uint32_t graph_global_row(const GraphNodeView& view,
                            uint32_t group_id,
                            uint32_t local_row)
{
    if (group_id >= view.label_count) {
        fail_graph_runtime("group_id out of range", group_id, local_row);
    }
    const uint32_t label_size = view.label_size[group_id];
    if (local_row >= label_size) {
        fail_graph_runtime("local row out of range for group", group_id, local_row);
    }
    const uint64_t global_row =
        static_cast<uint64_t>(view.label_offset[group_id]) + static_cast<uint64_t>(local_row);
    if (global_row >= view.graph_rows) {
        fail_graph_runtime("global row exceeds graph rows", group_id, local_row);
    }
    return static_cast<uint32_t>(global_row);
}

void assemble_graph_node(const GraphNodeView& view,
                           uint32_t group_id,
                           uint32_t local_row,
                           Node& node,
                           const AsyncIoConfig& config,
                           HostFilterStats* filter_stats)
{
    if (view.base_vector_cols != config.dim) {
        fail_graph_runtime("base vector dimension must match dim", group_id, local_row);
    }
    if (view.graph_degree > kNodeNeighborCapacity) {
        fail_graph_runtime("graph degree exceeds node capacity", group_id, local_row);
    }

    std::memset(&node, 0, sizeof(Node));
    node.index = local_row;

    const uint32_t global_row = graph_global_row(view, group_id, local_row);
    const uint32_t sample = view.row_to_sample[global_row];
    if (sample >= view.base_vector_rows) {
        fail_graph_runtime("row_to_sample exceeds base vector rows", group_id, local_row);
    }
    if (view.runtime == nullptr) {
        fail_graph_runtime("missing runtime for base vector access", group_id, local_row);
    }
    node.sample_id = sample;

    copy_base_vector_row(*view.runtime,
                         sample,
                         reinterpret_cast<uint8_t*>(node.data),
                         config.dim);

    const uint32_t label_size = view.label_size[group_id];
    const uint32_t* graph_row =
        view.graph_data + static_cast<size_t>(global_row) * view.graph_degree;
    uint32_t neighbor_count = 0;
    for (uint64_t col = 0; col < view.graph_degree; ++col) {
        const uint32_t local_neighbor = graph_row[col];
        if (local_neighbor == UINT32_MAX || local_neighbor >= label_size) {
            continue;
        }
        if (neighbor_count >= kNodeNeighborCapacity) {
            break;
        }
        node.neighbor[neighbor_count] = local_neighbor;
        if (config.include_sample_ids) {
            const uint32_t neighbor_global_row =
                static_cast<uint32_t>(view.label_offset[group_id] + local_neighbor);
            node.neighbor_sample[neighbor_count] = view.row_to_sample[neighbor_global_row];
        }
        ++neighbor_count;
    }
    node.neighbor_count = neighbor_count;
}

void serialize_graph_neighbors_words(const uint32_t* row_neighbors,
                                       uint32_t label_size,
                                       uint32_t* dst_words,
                                       const AsyncIoConfig& config,
                                       uint32_t graph_degree)
{
    std::memset(dst_words, 0, config.elems_per_node * sizeof(uint32_t));
    const uint32_t scan_degree = std::min<uint32_t>(config.degree, graph_degree);
    uint32_t neighbor_count = 0;
    uint32_t* dst_neighbors = dst_words + config.neighbor_words_offset;
    for (uint32_t col = 0; col < scan_degree; ++col) {
        const uint32_t local_neighbor = row_neighbors[col];
        if (local_neighbor == UINT32_MAX || local_neighbor >= label_size) {
            continue;
        }
        if (neighbor_count >= kNodeNeighborCapacity) {
            break;
        }
        dst_neighbors[neighbor_count++] = local_neighbor;
    }
    dst_words[config.neighbor_count_word] = neighbor_count;
}

void serialize_graph_neighbor_row_words(const GraphNodeView& view,
                                          uint32_t group_id,
                                          uint32_t local_row,
                                          const uint32_t* row_neighbors,
                                          uint32_t* dst_words,
                                          const AsyncIoConfig& config,
                                          uint32_t graph_degree)
{
    std::memset(dst_words, 0, config.elems_per_node * sizeof(uint32_t));
    const uint32_t global_row = graph_global_row(view, group_id, local_row);
    const uint32_t current_sample = view.row_to_sample[global_row];
    if (config.include_sample_ids) {
        dst_words[config.vector_payload_words] = current_sample;
    }
    const uint32_t label_size = view.label_size[group_id];
    const uint32_t scan_degree = std::min<uint32_t>(config.degree, graph_degree);
    uint32_t neighbor_count = 0;
    uint32_t* dst_neighbors = dst_words + config.neighbor_words_offset;
    uint32_t* dst_neighbor_samples = config.include_sample_ids
        ? dst_words + config.neighbor_sample_words_offset
        : nullptr;
    for (uint32_t col = 0; col < scan_degree; ++col) {
        const uint32_t local_neighbor = row_neighbors[col];
        if (local_neighbor == UINT32_MAX || local_neighbor >= label_size) {
            continue;
        }
        if (neighbor_count >= kNodeNeighborCapacity) {
            break;
        }
        dst_neighbors[neighbor_count] = local_neighbor;
        if (dst_neighbor_samples != nullptr) {
            const uint32_t neighbor_global_row =
                static_cast<uint32_t>(view.label_offset[group_id] + local_neighbor);
            dst_neighbor_samples[neighbor_count] = view.row_to_sample[neighbor_global_row];
        }
        ++neighbor_count;
    }
    dst_words[config.neighbor_count_word] = neighbor_count;
}

void serialize_graph_node_words(const Node& node,
                                  uint32_t* dst_words,
                                  const AsyncIoConfig& config)
{
    std::memset(dst_words, 0, config.elems_per_node * sizeof(uint32_t));
    std::memcpy(dst_words, node.data, config.vector_payload_words * sizeof(uint32_t));
    if (config.include_sample_ids) {
        dst_words[config.vector_payload_words] = node.sample_id;
    }
    const uint32_t neighbor_count = std::min<uint32_t>(node.neighbor_count, config.degree);
    dst_words[config.neighbor_count_word] = neighbor_count;
    std::memcpy(dst_words + config.neighbor_words_offset,
                node.neighbor,
                neighbor_count * sizeof(uint32_t));
    if (config.include_sample_ids) {
        std::memcpy(dst_words + config.neighbor_sample_words_offset,
                    node.neighbor_sample,
                    neighbor_count * sizeof(uint32_t));
    }
}

void serialize_graph_float_node_row_words(const GraphNodeView& view,
                                            uint32_t group_id,
                                            uint32_t local_row,
                                            const uint32_t* row_neighbors,
                                            uint32_t* dst_words,
                                            const AsyncIoConfig& config,
                                            uint32_t graph_degree)
{
    if (!config.reserve_vector_payload ||
        !dense_vector_scalar_is_float(config.vector_payload_scalar_type)) {
        fail_graph_runtime("float vector payload serialization requires float payload layout",
                             group_id,
                             local_row);
    }
    if (view.runtime == nullptr) {
        fail_graph_runtime("missing runtime for float vector payload",
                             group_id,
                             local_row);
    }
    std::memset(dst_words, 0, config.elems_per_node * sizeof(uint32_t));
    const uint32_t global_row = graph_global_row(view, group_id, local_row);
    const uint32_t sample = view.row_to_sample[global_row];
    if (sample >= view.base_vector_rows) {
        fail_graph_runtime("row_to_sample exceeds base float vector rows",
                             group_id,
                             local_row);
    }
    copy_base_float_vector_row(*view.runtime,
                               sample,
                               reinterpret_cast<float*>(dst_words),
                               config.dim);
    if (config.include_sample_ids) {
        dst_words[config.vector_payload_words] = sample;
    }
    const uint32_t label_size = view.label_size[group_id];
    const uint32_t scan_degree = std::min<uint32_t>(config.degree, graph_degree);
    uint32_t neighbor_count = 0;
    uint32_t* dst_neighbors = dst_words + config.neighbor_words_offset;
    uint32_t* dst_neighbor_samples = config.include_sample_ids
        ? dst_words + config.neighbor_sample_words_offset
        : nullptr;
    for (uint32_t col = 0; col < scan_degree; ++col) {
        const uint32_t local_neighbor = row_neighbors[col];
        if (local_neighbor == UINT32_MAX || local_neighbor >= label_size) {
            continue;
        }
        if (neighbor_count >= kNodeNeighborCapacity) {
            break;
        }
        dst_neighbors[neighbor_count] = local_neighbor;
        if (dst_neighbor_samples != nullptr) {
            const uint32_t neighbor_global_row =
                static_cast<uint32_t>(view.label_offset[group_id] + local_neighbor);
            dst_neighbor_samples[neighbor_count] = view.row_to_sample[neighbor_global_row];
        }
        ++neighbor_count;
    }
    dst_words[config.neighbor_count_word] = neighbor_count;
}

void serialize_graph_node_row_words(const GraphNodeView& view,
                                      uint32_t group_id,
                                      uint32_t local_row,
                                      const uint32_t* row_neighbors,
                                      uint32_t* dst_words,
                                      const AsyncIoConfig& config,
                                      uint32_t graph_degree,
                                      HostFilterStats* filter_stats)
{
    if (dense_vector_scalar_is_float(config.vector_payload_scalar_type)) {
        serialize_graph_float_node_row_words(view,
                                               group_id,
                                               local_row,
                                               row_neighbors,
                                               dst_words,
                                               config,
                                               graph_degree);
        return;
    }
    Node node{};
    assemble_graph_node(view,
                          group_id,
                          local_row,
                          node,
                          config,
                          filter_stats);
    serialize_graph_node_words(node, dst_words, config);
}

bool host_route_predicate_gpu_verified(const AsyncIoConfig& config, uint32_t route)
{
    if (config.predicate_gpu_verified_routes == nullptr ||
        route >= config.predicate_gpu_verified_routes->size() ||
        (*config.predicate_gpu_verified_routes)[route] == 0) return false;
    go_fann::route_predicate::require(
        config.predicate_plan != nullptr && route < config.predicate_plan->routes.size() &&
            config.predicate_plan->routes[route].available,
        "GPU bypass lacks a compiled graph-route predicate");
    return true;
}

int64_t compute_host_filter_result(const GraphNodeView& view,
                                   uint32_t sample_id,
                                   uint64_t row_filter_mask,
                                   uint32_t query_idx,
                                   bool enable_full_label_check,
                                   const AsyncIoConfig* config)
{
    if (view.runtime == nullptr) {
        return 2;
    }
    const GraphRuntime& runtime = *view.runtime;
    if (query_idx >= runtime.query_filter_bits.size()) {
        return 2;
    }
    if ((row_filter_mask & runtime.query_filter_bits[query_idx]) == 0ULL) {
        return 2;
    }
    if (config != nullptr && config->predicate_plan != nullptr) {
        go_fann::route_predicate::require(
            config->predicate_plan->graph_source == &runtime,
            "host service received a plan for another graph runtime");
        if (host_route_predicate_gpu_verified(*config, query_idx)) {
            return 1; // Adjacency readiness only; device owns exact admission.
        }
        if (query_idx < config->predicate_plan->routes.size() &&
            config->predicate_plan->routes[query_idx].available) {
            return go_fann::route_predicate::evaluate(
                *config->predicate_plan, runtime, query_idx, sample_id) ? 1 : 2;
        }
    }
    if (!host_query_matches_ranges(view, sample_id, query_idx)) {
        return 2;
    }
    if (enable_full_label_check && !host_query_matches_sample(view, sample_id, query_idx)) {
        return 2;
    }
    return 1;
}

bool should_publish_graph_vector_payload(const AsyncIoConfig& config,
                                           bool filter_passed)
{
    if (!config.reserve_vector_payload) {
        return false;
    }
    if (filter_passed) {
        return config.publish_valid_vector_payload;
    }
    return !config.skip_filtered_vector_payload;
}

void build_prefetch_slices(GraphRuntime& runtime,
                           const AsyncIoConfig& config)
{
    runtime.prefetch_begin_by_group.assign(runtime.metadata.label_count, 0);
    runtime.prefetch_end_by_group.assign(runtime.metadata.label_count, 0);
    runtime.prefetch_nodes.clear();
    runtime.prefetch_index.clear();

    const GraphNodeView view = make_graph_node_view(runtime);

    for (uint32_t group_id = 0; group_id < runtime.metadata.label_count; ++group_id) {
        runtime.prefetch_begin_by_group[group_id] = static_cast<uint32_t>(runtime.prefetch_index.size());
        const uint32_t group_size = runtime.metadata.label_size[group_id];
        if (group_size == 0) {
            runtime.prefetch_end_by_group[group_id] = static_cast<uint32_t>(runtime.prefetch_index.size());
            continue;
        }

        const uint32_t row_begin = runtime.metadata.label_offset[group_id];
        const uint32_t* group_graph_rows =
            runtime.graph.data.data() + static_cast<size_t>(row_begin) * runtime.graph.cols;

        std::vector<uint32_t> group_rows;
        std::unordered_set<uint32_t> visited;

        if (runtime.seed_selection_mode == "label") {
            const auto& labels_in_group = runtime.group_layout.groups[group_id];
            for (uint32_t original_label : labels_in_group) {
                const uint32_t seed_local = runtime.seed_by_original_label[original_label];
                if (seed_local == UINT32_MAX || seed_local >= group_size) {
                    continue;
                }
                collect_local_neighbors_from_graph(group_graph_rows,
                                                   runtime.graph.cols,
                                                   seed_local,
                                                   group_rows,
                                                   visited,
                                                   config.prefetch_hops,
                                                   group_size);
            }
        } else {
            const uint32_t seed_local = runtime.seed_local_id_by_label[group_id];
            if (seed_local != UINT32_MAX && seed_local < group_size) {
                collect_local_neighbors_from_graph(group_graph_rows,
                                                   runtime.graph.cols,
                                                   seed_local,
                                                   group_rows,
                                                   visited,
                                                   config.prefetch_hops,
                                                   group_size);
            }
        }

        std::sort(group_rows.begin(), group_rows.end());
        for (uint32_t local_row : group_rows) {
            Node node{};
            if (should_publish_graph_vector_payload(config, true)) {
                assemble_graph_node(view,
                                      group_id,
                                      local_row,
                                      node,
                                      config,
                                      nullptr);
            } else {
                std::memset(&node, 0, sizeof(Node));
                node.index = local_row;
                const uint32_t global_row = graph_global_row(view, group_id, local_row);
                node.sample_id = view.row_to_sample[global_row];
                const uint32_t* graph_row =
                    view.graph_data + static_cast<size_t>(global_row) * view.graph_degree;
                const uint32_t label_size = view.label_size[group_id];
                uint32_t neighbor_count = 0;
                const uint32_t scan_degree =
                    std::min<uint32_t>(config.degree, static_cast<uint32_t>(view.graph_degree));
                for (uint32_t col = 0; col < scan_degree; ++col) {
                    const uint32_t local_neighbor = graph_row[col];
                    if (local_neighbor == UINT32_MAX || local_neighbor >= label_size) {
                        continue;
                    }
                    if (neighbor_count >= kNodeNeighborCapacity) {
                        break;
                    }
                    node.neighbor[neighbor_count] = local_neighbor;
                    if (config.include_sample_ids) {
                        const uint32_t neighbor_global_row =
                            static_cast<uint32_t>(view.label_offset[group_id] + local_neighbor);
                        node.neighbor_sample[neighbor_count] = view.row_to_sample[neighbor_global_row];
                    }
                    ++neighbor_count;
                }
                node.neighbor_count = neighbor_count;
            }
            runtime.prefetch_index.push_back(local_row);
            runtime.prefetch_nodes.push_back(node);
        }
        runtime.prefetch_end_by_group[group_id] = static_cast<uint32_t>(runtime.prefetch_index.size());
    }
}

void host_scan_and_set(int64_t* status_memory,
                       uint32_t* h_neighbor_memory,
                       int64_t* h_gdr_data,
                       int* req_per_loop,
                       int64_t* status_flag,
                       const GraphNodeView& host_view,
                       size_t total_queries,
                       const uint32_t* active_query_ids,
                       size_t active_query_count,
                       const AsyncIoConfig& config,
                       HostFilterStats* filter_stats)
{
    const size_t loop_query_count = active_query_ids != nullptr ? active_query_count : total_queries;
    if (config.host_service_threads > 1 && loop_query_count > 1) {
        const int worker_count = std::min<int>(config.host_service_threads, static_cast<int>(loop_query_count));
        std::atomic<size_t> next_batch_idx(static_cast<size_t>(worker_count));
        std::vector<std::thread> workers;
        std::vector<HostFilterStats> worker_stats(worker_count);
        workers.reserve(worker_count);

        for (int worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back(host_scan_and_set_worker,
                                 status_memory,
                                 h_neighbor_memory,
                                 h_gdr_data,
                                 req_per_loop,
                                 status_flag,
                                 std::cref(host_view),
                                 total_queries,
                                 active_query_ids,
                                 active_query_count,
                                 worker,
                                 worker_count,
                                 &next_batch_idx,
                                 config,
                                 &worker_stats[worker],
                                 worker == 0);
        }

        for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
            if (host_hang_debug_enabled()) {
                std::printf("[host-debug] joining worker=%d/%d\n", worker_idx, worker_count);
                std::fflush(stdout);
            }
            workers[worker_idx].join();
            if (host_hang_debug_enabled()) {
                std::printf("[host-debug] joined worker=%d/%d\n", worker_idx, worker_count);
                std::fflush(stdout);
            }
        }

        if (filter_stats != nullptr) {
            *filter_stats = HostFilterStats{};
            filter_stats->workers_thread = worker_count;
            for (const auto& stat : worker_stats) {
                filter_stats->requests_served += stat.requests_served;
                filter_stats->requests_passed += stat.requests_passed;
                filter_stats->requests_filtered += stat.requests_filtered;
                filter_stats->demand_batch_count += stat.demand_batch_count;
                filter_stats->demand_partial_batch_count += stat.demand_partial_batch_count;
                filter_stats->demand_batch_request_count += stat.demand_batch_request_count;
                filter_stats->demand_batch_wait_ns += stat.demand_batch_wait_ns;
                filter_stats->demand_batch_max_size = std::max(
                    filter_stats->demand_batch_max_size, stat.demand_batch_max_size);
            }
            filter_stats->demand_batch_configured_size = config.host_demand_batch_size;
            filter_stats->demand_batch_timeout_us = config.host_demand_batch_timeout_us;
        }

        std::printf("cpu_thread end\n");
        return;
    }

    maybe_bind_current_thread(config.host_service_cpu_start, config.host_service_cpu_count);
    if (filter_stats != nullptr) {
        filter_stats->workers_thread = loop_query_count > 0 ? 1 : 0;
        filter_stats->demand_batch_configured_size = config.host_demand_batch_size;
        filter_stats->demand_batch_timeout_us = config.host_demand_batch_timeout_us;
    }

    std::vector<bool> query_finished(loop_query_count, false);
    std::vector<uint32_t> last_seen_generation(total_queries, UINT32_MAX);
    std::vector<uint32_t> last_seen_row(total_queries, UINT32_MAX);
    std::vector<uint32_t> last_seen_buffer(total_queries, UINT32_MAX);
    std::vector<uint8_t> seen_pending(total_queries, 0);
    HostMailboxSyncStats sync_stats{};
    size_t finished_count = 0;
    size_t no_progress_rounds = 0;
    std::vector<PendingMemoryDemand> pending_demands;
    if (config.host_demand_batch_size > 1) {
        pending_demands.reserve(config.host_demand_batch_size);
    }

    int j = 0;
    while (finished_count < loop_query_count) {
        j++;
        auto start = std::chrono::high_resolution_clock::now();
        int submitted_this_loop = 0;
        const size_t finished_before = finished_count;

        for (size_t logical_idx = 0; logical_idx < loop_query_count; logical_idx++) {
            if (query_finished[logical_idx]) {
                continue;
            }

            const size_t i = active_query_ids != nullptr
                ? static_cast<size_t>(active_query_ids[logical_idx])
                : logical_idx;

            volatile int64_t* flag_ptr = &status_flag[i * 16];
            const int64_t flag = *flag_ptr;

            if (graph_status_flag_finished(flag)) {
                seen_pending[i] = 0;
                query_finished[logical_idx] = true;
                ++finished_count;
                (void)publish_host_done(config.completion_sink,
                                        static_cast<uint32_t>(i));
                if (host_hang_debug_enabled() && finished_count % 1000 == 0) {
                    std::printf("[host-debug] single finished=%zu/%zu\n",
                                finished_count,
                                loop_query_count);
                    std::fflush(stdout);
                }
                continue;
            }

            if (flag == 1) {
                const int64_t val = status_memory[i * 16];
                const int64_t buffer_val = status_memory[i * 16 + 1];
                const int64_t generation_val = load_mailbox_word_host(status_memory, i * 16 + 2);
                if (val >= 0) {
                        const uint32_t generation = generation_val >= 0 ? static_cast<uint32_t>(generation_val) : 0U;
                        const uint32_t local_row = static_cast<uint32_t>(val);
                        const uint32_t buffer_idx = config.payload_slot_count > 1
                            ? static_cast<uint32_t>(buffer_val & 1LL)
                            : 0U;
                        if (config.host_demand_batch_size > 1) {
                            (void)append_pending_memory_demand(i,
                                                               val,
                                                               buffer_val,
                                                               generation_val,
                                                               h_gdr_data,
                                                               host_view,
                                                               config,
                                                               last_seen_generation,
                                                               last_seen_row,
                                                               last_seen_buffer,
                                                               seen_pending,
                                                               sync_stats,
                                                               pending_demands);
                            const uint64_t wait_ns = pending_memory_demand_age_ns(pending_demands);
                            const uint64_t timeout_ns = static_cast<uint64_t>(
                                config.host_demand_batch_timeout_us) * 1000ULL;
                            if (host_demand_batch_should_flush(pending_demands.size(),
                                                               config.host_demand_batch_size,
                                                               wait_ns,
                                                               timeout_ns)) {
                                submitted_this_loop += static_cast<int>(flush_pending_memory_demands(
                                    pending_demands,
                                    status_memory,
                                    h_neighbor_memory,
                                    h_gdr_data,
                                    status_flag,
                                    host_view,
                                    config,
                                    filter_stats,
                                    sync_stats));
                            }
                            continue;
                        }
                        if (seen_pending[i] != 0 &&
                            last_seen_generation[i] == generation &&
                            last_seen_row[i] == local_row &&
                            last_seen_buffer[i] == buffer_idx) {
                            if (host_stats_collection_enabled()) {
                                ++sync_stats.duplicate_observations;
                            }
                        }
                        last_seen_generation[i] = generation;
                        last_seen_row[i] = local_row;
                        last_seen_buffer[i] = buffer_idx;
                        seen_pending[i] = 1;
                        const uint32_t group_id = host_view.query_group_ids[i];
                        if (group_id >= host_view.label_count) {
                            std::fprintf(stderr,
                                         "graph runtime runtime invariant failed: group_id out of range (group_id=%u, query=%zu, label_count=%u)\n",
                                         group_id,
                                         i,
                                         host_view.label_count);
                            std::fflush(stderr);
                            std::abort();
                        }
                        const uint32_t label_size = host_view.label_size[group_id];
                        if (local_row >= label_size) {
                            h_gdr_data[i * kGdrMailboxStride] = 2;
                            _mm_sfence();
                            continue;
                        }
                    if (filter_stats != nullptr) {
                        if (host_stats_collection_enabled()) {
                            ++filter_stats->requests_served;
                        }
                    }
                        const uint32_t global_row = graph_global_row(host_view, group_id, local_row);
                        if (global_row >= host_view.total_rows) {
                            std::fprintf(stderr,
                                         "graph runtime runtime invariant failed: global_row out of range (group_id=%u, local_row=%u, global_row=%u, total_rows=%u)\n",
                                         group_id,
                                         local_row,
                                         global_row,
                                         host_view.total_rows);
                            std::fflush(stderr);
                            std::abort();
                        }
                        const uint32_t sample_id = host_view.row_to_sample[global_row];
                    const uint64_t filter_mask =
                        graph_row_filter_mask_at(host_view, global_row);
                    const int64_t filter_result = compute_host_filter_result(host_view,
                                                                             sample_id,
                                                                             filter_mask,
                                                                             static_cast<uint32_t>(i),
                                                                             host_full_label_check_enabled(
                                                                                 host_view, config), &config);
                    if (filter_stats != nullptr) {
                        if (filter_result == 1) {
                            if (host_stats_collection_enabled()) {
                                ++filter_stats->requests_passed;
                            }
                        } else {
                            if (host_stats_collection_enabled()) {
                                ++filter_stats->requests_filtered;
                            }
                        }
                    }
                    const size_t payload_slot = static_cast<size_t>(i) * config.payload_slot_count +
                        static_cast<size_t>(buffer_idx);
                    uint32_t* dst_words =
                        h_neighbor_memory + payload_slot * config.elems_per_block;
                    if (should_publish_graph_vector_payload(config, filter_result == 1)) {
                        const uint32_t* graph_row =
                            host_view.graph_data + static_cast<size_t>(global_row) * host_view.graph_degree;
                        serialize_graph_node_row_words(
                            host_view,
                            group_id,
                            local_row,
                            graph_row,
                            dst_words,
                            config,
                            static_cast<uint32_t>(host_view.graph_degree),
                            filter_stats);
                    } else {
                        const uint32_t* graph_row =
                            host_view.graph_data + static_cast<size_t>(global_row) * host_view.graph_degree;
                        serialize_graph_neighbor_row_words(host_view,
                                                             group_id,
                                                             local_row,
                                                             graph_row,
                                                             dst_words,
                                                             config,
                                                             static_cast<uint32_t>(host_view.graph_degree));
                    }
                    _mm_sfence();
                    const SyncPublishCheckResult check = sync_publish_check(status_memory,
                                                                            status_flag,
                                                                            h_gdr_data,
                                                                            i,
                                                                            generation,
                                                                            local_row,
                                                                            buffer_idx,
                                                                            config);
                    switch (check) {
                    case SyncPublishCheckResult::Ok:
                        publish_in_memory_mailbox_ready(h_gdr_data,
                                                       i,
                                                       filter_result,
                                                       generation,
                                                       local_row,
                                                       buffer_idx,
                                                       config);
                        submitted_this_loop++;
                        break;
                    case SyncPublishCheckResult::StatusNotWaiting:
                        if (host_stats_collection_enabled()) {
                            ++sync_stats.stale_status_not_waiting;
                        }
                        break;
                    case SyncPublishCheckResult::GenerationMismatch:
                        if (host_stats_collection_enabled()) {
                            ++sync_stats.stale_generation_mismatch;
                        }
                        break;
                    case SyncPublishCheckResult::RowMismatch:
                        if (host_stats_collection_enabled()) {
                            ++sync_stats.stale_row_mismatch;
                        }
                        break;
                    case SyncPublishCheckResult::BufferMismatch:
                        if (host_stats_collection_enabled()) {
                            ++sync_stats.stale_buffer_mismatch;
                        }
                        break;
                    case SyncPublishCheckResult::MailboxNotZero:
                        if (host_stats_collection_enabled()) {
                            ++sync_stats.stale_mailbox_not_zero;
                        }
                        break;
                    }
                }
            }
        }

        if (!pending_demands.empty()) {
            const uint64_t wait_ns = pending_memory_demand_age_ns(pending_demands);
            const uint64_t timeout_ns = static_cast<uint64_t>(
                config.host_demand_batch_timeout_us) * 1000ULL;
            if (host_demand_batch_should_flush(pending_demands.size(),
                                               config.host_demand_batch_size,
                                               wait_ns,
                                               timeout_ns)) {
                submitted_this_loop += static_cast<int>(flush_pending_memory_demands(
                    pending_demands,
                    status_memory,
                    h_neighbor_memory,
                    h_gdr_data,
                    status_flag,
                    host_view,
                    config,
                    filter_stats,
                    sync_stats));
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;

        if (j < 10000 && submitted_this_loop > 0) {
            req_per_loop[j * 2] = static_cast<int>(elapsed.count());
            req_per_loop[j * 2 + 1] = submitted_this_loop;
        }

        if (host_hang_debug_enabled()) {
            if (submitted_this_loop == 0 && finished_count == finished_before) {
                ++no_progress_rounds;
                if (no_progress_rounds % 200 == 0) {
                    size_t stuck_query = loop_query_count;
                    for (size_t logical_idx = 0; logical_idx < loop_query_count; ++logical_idx) {
                        if (!query_finished[logical_idx]) {
                            stuck_query = active_query_ids != nullptr
                                ? static_cast<size_t>(active_query_ids[logical_idx])
                                : logical_idx;
                            break;
                        }
                    }
                    std::printf("[host-debug] single no progress rounds=%zu finished=%zu/%zu stuck_query=%zu status=%lld row=%lld buffer=%lld gen=%lld mailbox=%lld\n",
                                no_progress_rounds,
                                finished_count,
                                loop_query_count,
                                stuck_query,
                                stuck_query < total_queries ? static_cast<long long>(status_flag[stuck_query * 16]) : -1LL,
                                stuck_query < total_queries ? static_cast<long long>(status_memory[stuck_query * 16]) : -1LL,
                                stuck_query < total_queries ? static_cast<long long>(status_memory[stuck_query * 16 + 1]) : -1LL,
                                stuck_query < total_queries ? static_cast<long long>(status_memory[stuck_query * 16 + 2]) : -1LL,
                                stuck_query < total_queries ? static_cast<long long>(h_gdr_data[stuck_query * kGdrMailboxStride]) : -1LL);
                    std::fflush(stdout);
                }
            } else {
                no_progress_rounds = 0;
            }
        }
    }
    if (!pending_demands.empty()) {
        (void)flush_pending_memory_demands(pending_demands,
                                           status_memory,
                                           h_neighbor_memory,
                                           h_gdr_data,
                                           status_flag,
                                           host_view,
                                           config,
                                           filter_stats,
                                           sync_stats);
    }
    std::printf("cpu_thread end\n");
    print_host_sync_summary("PQ", "in-memory(single)", sync_stats);
}

namespace {

uint64_t load_request_ring_sequence(const uint64_t* sequence)
{
    return __atomic_load_n(sequence, __ATOMIC_ACQUIRE);
}

void store_request_ring_sequence(uint64_t* sequence,
                                 uint64_t value,
                                 bool device_mapped)
{
    if (device_mapped) {
        // sfence drains a potentially write-combining GDR mapping before a GPU
        // polling load can observe this slot as reusable.
        *reinterpret_cast<volatile uint64_t*>(sequence) = value;
        _mm_sfence();
    } else {
        __atomic_store_n(sequence, value, __ATOMIC_RELEASE);
    }
}

void host_request_ring_worker(int64_t* status_memory,
                              uint32_t* h_neighbor_memory,
                              int64_t* h_gdr_data,
                              int64_t* status_flag,
                              const GraphNodeView& host_view,
                              size_t total_queries,
                              size_t active_query_count,
                              const uint32_t* expected_ring_by_query,
                              const HostRequestRingView& request_ring,
                              int worker_idx,
                              int worker_count,
                              const std::atomic<bool>* cancellation,
                              const AsyncIoConfig& config,
                              HostFilterStats* filter_stats)
{
    constexpr size_t kDrainBudgetPerRing = 8;
    maybe_bind_worker_thread(config, worker_idx);
    const bool profile_enabled = rabitq_graph_profile_enabled();
    const uint64_t profile_sample_period =
        uint64_t{1} << rabitq_graph_profile_sample_log2();
    const uint64_t worker_start_ns = profile_enabled ? host_steady_now_ns() : 0;
    HostMailboxSyncStats sync_stats{};
    std::vector<PendingMemoryDemand> pending_demands;
    pending_demands.reserve(1);
    std::vector<uint64_t> next_ticket(request_ring.ring_count, 0);
    std::vector<size_t> done_by_ring(request_ring.ring_count, 0);
    std::vector<size_t> expected_done_by_ring(request_ring.ring_count, 0);
    uint64_t batches_claimed = 0;
    uint64_t demand_events = 0;
    uint64_t done_events = 0;
    uint64_t empty_polls = 0;
    size_t expected_done_total = 0;
    for (size_t ring_idx = static_cast<size_t>(worker_idx);
         ring_idx < request_ring.ring_count;
         ring_idx += static_cast<size_t>(worker_count)) {
        const size_t logical_begin = ring_idx * request_ring.capacity_per_ring;
        expected_done_by_ring[ring_idx] = std::min(
            request_ring.capacity_per_ring, active_query_count - logical_begin);
        expected_done_total += expected_done_by_ring[ring_idx];
        ++batches_claimed;
    }

    while (done_events < expected_done_total) {
        if (cancellation != nullptr &&
            cancellation->load(std::memory_order_acquire)) {
            break;
        }
        bool made_progress = false;
        for (size_t ring_idx = static_cast<size_t>(worker_idx);
             ring_idx < request_ring.ring_count;
             ring_idx += static_cast<size_t>(worker_count)) {
            size_t drained = 0;
            while (done_by_ring[ring_idx] < expected_done_by_ring[ring_idx] &&
                   drained < kDrainBudgetPerRing) {
                const uint64_t ticket = next_ticket[ring_idx];
                const size_t slot = ring_idx * request_ring.capacity_per_ring +
                    static_cast<size_t>(ticket % request_ring.capacity_per_ring);
                HostRequestRingEntry* entry = request_ring.entries + slot;
                if (load_request_ring_sequence(&entry->published_sequence) != ticket + 1) {
                    break;
                }

                const HostRequestRingEntry event = *entry;
                if (event.query_id >= total_queries ||
                    expected_ring_by_query[event.query_id] != ring_idx) {
                    std::fprintf(stderr,
                                 "host request ring identity mismatch "
                                 "(ring=%zu ticket=%llu query=%u expected_ring=%u)\n",
                                 ring_idx,
                                 static_cast<unsigned long long>(ticket),
                                 event.query_id,
                                 event.query_id < total_queries
                                     ? expected_ring_by_query[event.query_id]
                                     : UINT32_MAX);
                    std::fflush(stderr);
                    std::abort();
                }

                const HostRequestKind kind = static_cast<HostRequestKind>(event.kind);
                bool profile_sample = false;
                uint64_t service_start_ns = 0;
                if (kind == HostRequestKind::Demand) {
                    profile_sample = profile_enabled &&
                        ((demand_events & (profile_sample_period - 1)) == 0);
                    service_start_ns = profile_sample ? host_steady_now_ns() : 0;
                    const size_t status_base = static_cast<size_t>(event.query_id) * 16;
                    const uint32_t row = static_cast<uint32_t>(
                        load_mailbox_word_host(status_memory, status_base));
                    const uint32_t buffer = static_cast<uint32_t>(
                        load_mailbox_word_host(status_memory, status_base + 1));
                    const uint32_t generation = static_cast<uint32_t>(
                        load_mailbox_word_host(status_memory, status_base + 2));
                    if (profile_sample && filter_stats != nullptr) {
                        filter_stats->request_ring_mailbox_snapshot_ns +=
                            host_steady_now_ns() - service_start_ns;
                    }
                    const uint32_t group_id = host_view.query_group_ids[event.query_id];
                    if (group_id >= host_view.label_count) {
                        std::fprintf(stderr,
                                     "host request ring group out of range "
                                     "(query=%u group=%u labels=%u)\n",
                                     event.query_id,
                                     group_id,
                                     host_view.label_count);
                        std::fflush(stderr);
                        std::abort();
                    }
                    if (row >= host_view.label_size[group_id]) {
                        h_gdr_data[static_cast<size_t>(event.query_id) * kGdrMailboxStride] = 2;
                        _mm_sfence();
                    } else {
                        pending_demands.push_back(
                            {event.query_id,
                             generation,
                             row,
                             config.payload_slot_count > 1 ? buffer & 1U : 0U,
                             std::chrono::steady_clock::now()});
                        const size_t published = flush_pending_memory_demands(
                            pending_demands,
                            status_memory,
                            h_neighbor_memory,
                            h_gdr_data,
                            status_flag,
                            host_view,
                            config,
                            filter_stats,
                            sync_stats,
                            profile_sample);
                        if (published != 1) {
                            std::fprintf(stderr,
                                         "host request ring rejected current demand "
                                         "(ring=%zu ticket=%llu query=%u generation=%u row=%u buffer=%u)\n",
                                         ring_idx,
                                         static_cast<unsigned long long>(ticket),
                                         event.query_id,
                                         generation,
                                         row,
                                         buffer);
                            std::fflush(stderr);
                            std::abort();
                        }
                    }
                    ++demand_events;
                } else if (kind == HostRequestKind::Done) {
                    if (!graph_status_flag_finished(
                            load_mailbox_word_host(status_flag,
                                                   static_cast<size_t>(event.query_id) * 16))) {
                        std::fprintf(stderr,
                                     "host request ring observed DONE before terminal status "
                                     "(ring=%zu ticket=%llu query=%u)\n",
                                     ring_idx,
                                     static_cast<unsigned long long>(ticket),
                                     event.query_id);
                        std::fflush(stderr);
                        std::abort();
                    }
                    ++done_by_ring[ring_idx];
                    ++done_events;
                } else {
                    std::fprintf(stderr,
                                 "host request ring invalid event kind "
                                 "(ring=%zu ticket=%llu query=%u kind=%u)\n",
                                 ring_idx,
                                 static_cast<unsigned long long>(ticket),
                                 event.query_id,
                                 event.kind);
                    std::fflush(stderr);
                    std::abort();
                }

                store_request_ring_sequence(
                    request_ring.consumed_sequences + slot,
                    ticket + 1,
                    request_ring.consumed_sequences_device_mapped);
                if (profile_sample && filter_stats != nullptr) {
                    filter_stats->request_ring_service_ns +=
                        host_steady_now_ns() - service_start_ns;
                }
                next_ticket[ring_idx] = ticket + 1;
                ++drained;
                made_progress = true;
            }
        }
        if (!made_progress) {
            ++empty_polls;
            _mm_pause();
        }
    }

    if (filter_stats != nullptr) {
        filter_stats->request_ring_profile_enabled = profile_enabled;
        filter_stats->request_ring_profile_sample_period =
            profile_enabled ? profile_sample_period : 0;
        filter_stats->request_ring_batches_claimed = batches_claimed;
        filter_stats->request_ring_demand_events = demand_events;
        filter_stats->request_ring_done_events = done_events;
        filter_stats->request_ring_empty_polls = empty_polls;
        if (profile_enabled) {
            const uint64_t worker_wall_ns = host_steady_now_ns() - worker_start_ns;
            filter_stats->request_ring_worker_wall_ns = worker_wall_ns;
            filter_stats->request_ring_worker_max_wall_ns = worker_wall_ns;
        }
    }

    print_host_sync_summary("memory", "request_ring", sync_stats);
}

}  // namespace

void host_request_ring_and_set(int64_t* status_memory,
                               uint32_t* h_neighbor_memory,
                               int64_t* h_gdr_data,
                               int64_t* status_flag,
                               const GraphNodeView& host_view,
                               size_t total_queries,
                               const uint32_t* active_query_ids,
                               size_t active_query_count,
                               const HostRequestRingView& request_ring,
                               const std::atomic<bool>* cancellation,
                               const AsyncIoConfig& config,
                               HostFilterStats* filter_stats)
{
    if (active_query_ids == nullptr || active_query_count == 0) {
        std::fprintf(stderr, "host request ring requires a non-empty active query list\n");
        std::abort();
    }
    if (request_ring.entries == nullptr || request_ring.consumed_sequences == nullptr ||
        request_ring.capacity_per_ring == 0) {
        std::fprintf(stderr, "host request ring requires a complete non-empty layout\n");
        std::abort();
    }
    const size_t expected_ring_count =
        (active_query_count + request_ring.capacity_per_ring - 1) /
        request_ring.capacity_per_ring;
    if (request_ring.ring_count != expected_ring_count) {
        std::fprintf(stderr,
                     "host request ring layout mismatch "
                     "(rings=%zu expected=%zu capacity=%zu)\n",
                     request_ring.ring_count,
                     expected_ring_count,
                     request_ring.capacity_per_ring);
        std::abort();
    }

    std::vector<uint32_t> expected_ring_by_query(total_queries, UINT32_MAX);
    for (size_t logical_idx = 0; logical_idx < active_query_count; ++logical_idx) {
        const uint32_t query_id = active_query_ids[logical_idx];
        if (query_id >= total_queries || expected_ring_by_query[query_id] != UINT32_MAX) {
            std::fprintf(stderr,
                         "host request ring active query list is invalid "
                         "(logical=%zu query=%u total=%zu)\n",
                         logical_idx,
                         query_id,
                         total_queries);
            std::abort();
        }
        expected_ring_by_query[query_id] = static_cast<uint32_t>(
            logical_idx / request_ring.capacity_per_ring);
    }

    const int worker_count = std::min<int>(
        config.host_service_threads, static_cast<int>(request_ring.ring_count));
    std::vector<std::thread> workers;
    std::vector<HostFilterStats> worker_stats(worker_count);
    workers.reserve(worker_count);
    for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
        workers.emplace_back(host_request_ring_worker,
                             status_memory,
                             h_neighbor_memory,
                             h_gdr_data,
                             status_flag,
                             std::cref(host_view),
                             total_queries,
                             active_query_count,
                             expected_ring_by_query.data(),
                             std::cref(request_ring),
                             worker_idx,
                             worker_count,
                             cancellation,
                             std::cref(config),
                             &worker_stats[worker_idx]);
    }
    for (std::thread& worker : workers) {
        worker.join();
    }

    if (filter_stats != nullptr) {
        *filter_stats = HostFilterStats{};
        filter_stats->workers_thread = worker_count;
        filter_stats->demand_batch_configured_size = config.host_demand_batch_size;
        filter_stats->demand_batch_timeout_us = config.host_demand_batch_timeout_us;
        for (const HostFilterStats& stat : worker_stats) {
            filter_stats->requests_served += stat.requests_served;
            filter_stats->requests_passed += stat.requests_passed;
            filter_stats->requests_filtered += stat.requests_filtered;
            filter_stats->demand_batch_count += stat.demand_batch_count;
            filter_stats->demand_partial_batch_count += stat.demand_partial_batch_count;
            filter_stats->demand_batch_request_count += stat.demand_batch_request_count;
            filter_stats->demand_batch_wait_ns += stat.demand_batch_wait_ns;
            filter_stats->demand_batch_max_size = std::max(
                filter_stats->demand_batch_max_size, stat.demand_batch_max_size);
            filter_stats->request_ring_batches_claimed +=
                stat.request_ring_batches_claimed;
            filter_stats->request_ring_demand_events +=
                stat.request_ring_demand_events;
            filter_stats->request_ring_done_events += stat.request_ring_done_events;
            filter_stats->request_ring_empty_polls += stat.request_ring_empty_polls;
            filter_stats->request_ring_profile_enabled =
                filter_stats->request_ring_profile_enabled ||
                stat.request_ring_profile_enabled;
            filter_stats->request_ring_profile_sample_period = std::max(
                filter_stats->request_ring_profile_sample_period,
                stat.request_ring_profile_sample_period);
            filter_stats->request_ring_profile_samples +=
                stat.request_ring_profile_samples;
            filter_stats->request_ring_mailbox_snapshot_ns +=
                stat.request_ring_mailbox_snapshot_ns;
            filter_stats->request_ring_filter_ns += stat.request_ring_filter_ns;
            filter_stats->request_ring_payload_ns += stat.request_ring_payload_ns;
            filter_stats->request_ring_publish_ns += stat.request_ring_publish_ns;
            filter_stats->request_ring_service_ns += stat.request_ring_service_ns;
            filter_stats->request_ring_worker_wall_ns +=
                stat.request_ring_worker_wall_ns;
            filter_stats->request_ring_worker_max_wall_ns = std::max(
                filter_stats->request_ring_worker_max_wall_ns,
                stat.request_ring_worker_max_wall_ns);
        }
    }
    std::printf("[request-ring] rings=%zu capacity_per_ring=%zu workers=%d "
                "batches=%llu demands=%llu done=%llu empty_polls=%llu\n",
                request_ring.ring_count,
                request_ring.capacity_per_ring,
                worker_count,
                static_cast<unsigned long long>(
                    filter_stats != nullptr ? filter_stats->request_ring_batches_claimed : 0),
                static_cast<unsigned long long>(
                    filter_stats != nullptr ? filter_stats->request_ring_demand_events : 0),
                static_cast<unsigned long long>(
                    filter_stats != nullptr ? filter_stats->request_ring_done_events : 0),
                static_cast<unsigned long long>(
                    filter_stats != nullptr ? filter_stats->request_ring_empty_polls : 0));
    if (filter_stats != nullptr && filter_stats->request_ring_profile_enabled) {
        const double samples = static_cast<double>(
            filter_stats->request_ring_profile_samples);
        const auto average_us = [samples](uint64_t total_ns) {
            return samples > 0.0
                ? static_cast<double>(total_ns) / samples / 1000.0
                : 0.0;
        };
        const uint64_t named_ns = filter_stats->request_ring_mailbox_snapshot_ns +
            filter_stats->request_ring_filter_ns +
            filter_stats->request_ring_payload_ns +
            filter_stats->request_ring_publish_ns;
        const uint64_t other_ns = filter_stats->request_ring_service_ns > named_ns
            ? filter_stats->request_ring_service_ns - named_ns
            : 0;
        std::printf("[request-ring profile] clock=cpu sampling=1/%llu samples=%llu "
                    "avg_service_us=%.4f avg_mailbox_snapshot_us=%.4f "
                    "avg_filter_us=%.4f avg_payload_us=%.4f "
                    "avg_publish_us=%.4f avg_other_us=%.4f "
                    "worker_wall_sum_ms=%.3f worker_max_wall_ms=%.3f\n",
                    static_cast<unsigned long long>(
                        filter_stats->request_ring_profile_sample_period),
                    static_cast<unsigned long long>(
                        filter_stats->request_ring_profile_samples),
                    average_us(filter_stats->request_ring_service_ns),
                    average_us(filter_stats->request_ring_mailbox_snapshot_ns),
                    average_us(filter_stats->request_ring_filter_ns),
                    average_us(filter_stats->request_ring_payload_ns),
                    average_us(filter_stats->request_ring_publish_ns),
                    average_us(other_ns),
                    static_cast<double>(filter_stats->request_ring_worker_wall_ns) /
                        1.0e6,
                    static_cast<double>(filter_stats->request_ring_worker_max_wall_ns) /
                        1.0e6);
        for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
            const HostFilterStats& stat = worker_stats[worker_idx];
            std::printf("[request-ring profile worker] worker=%d rings=%llu "
                        "demands=%llu done=%llu empty_polls=%llu wall_ms=%.3f\n",
                        worker_idx,
                        static_cast<unsigned long long>(
                            stat.request_ring_batches_claimed),
                        static_cast<unsigned long long>(
                            stat.request_ring_demand_events),
                        static_cast<unsigned long long>(
                            stat.request_ring_done_events),
                        static_cast<unsigned long long>(
                            stat.request_ring_empty_polls),
                        static_cast<double>(stat.request_ring_worker_wall_ns) /
                            1.0e6);
        }
    }
}

void host_candidate_scan_and_set(int64_t* status_memory,
                                 uint32_t* h_payload_memory,
                                 int64_t* h_gdr_data,
                                 int64_t* status_flag,
                                 const GraphRuntime& runtime,
                                 const HostCandidateSourceView* sources,
                                 size_t source_count,
                                 const AsyncIoConfig& config,
                                 HostCandidateStreamStats* stats)
{
    if (source_count == 0) {
        if (stats != nullptr) *stats = HostCandidateStreamStats{};
        return;
    }
    if (status_memory == nullptr || h_payload_memory == nullptr || h_gdr_data == nullptr ||
        status_flag == nullptr || sources == nullptr ||
        config.payload_slot_count < 2 ||
        config.elems_per_block <= kCandidatePayloadIdsWord) {
        std::fprintf(stderr, "candidate stream received an invalid communication layout\n");
        std::abort();
    }
    for (size_t source_idx = 0; source_idx < source_count; ++source_idx) {
        const HostCandidateSourceView& source = sources[source_idx];
        if (source.route_id >= runtime.query_vectors.rows ||
            (source.sample_count != 0 && source.sample_ids == nullptr) ||
            source.source_kind == QueryCandidateSource::None) {
            std::fprintf(stderr,
                         "candidate stream source is invalid (source=%zu route=%u rows=%llu kind=%u)\n",
                         source_idx,
                         source.route_id,
                         static_cast<unsigned long long>(source.sample_count),
                         static_cast<unsigned>(source.source_kind));
            std::abort();
        }
    }

    const int worker_count = std::max(
        1, std::min(config.host_service_threads, static_cast<int>(source_count)));
    std::atomic<uint64_t> source_rows_scanned{0};
    std::atomic<uint64_t> candidates_emitted{0};
    std::atomic<uint64_t> chunks_published{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(worker_count));
    for (int worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
        workers.emplace_back([&, worker_idx] {
            maybe_bind_worker_thread(config, worker_idx);
            std::vector<uint8_t> finished(source_count, 0);
            std::vector<uint32_t> payload(config.elems_per_block, 0);
            size_t remaining = 0;
            for (size_t source_idx = static_cast<size_t>(worker_idx);
                 source_idx < source_count;
                 source_idx += static_cast<size_t>(worker_count)) {
                ++remaining;
            }
            while (remaining != 0) {
                bool made_progress = false;
                for (size_t source_idx = static_cast<size_t>(worker_idx);
                     source_idx < source_count;
                     source_idx += static_cast<size_t>(worker_count)) {
                    if (finished[source_idx] != 0) continue;
                    const HostCandidateSourceView& source = sources[source_idx];
                    const size_t status_base = static_cast<size_t>(source.route_id) * 16;
                    const int64_t flag = load_mailbox_word_host(status_flag, status_base);
                    if (graph_status_flag_finished(flag)) {
                        finished[source_idx] = 1;
                        --remaining;
                        made_progress = true;
                        continue;
                    }
                    if (flag != 1) continue;

                    const uint64_t cursor = static_cast<uint64_t>(
                        load_mailbox_word_host(status_memory, status_base));
                    const uint32_t buffer_idx = static_cast<uint32_t>(
                        load_mailbox_word_host(status_memory, status_base + 1)) & 1U;
                    const uint32_t generation = static_cast<uint32_t>(
                        load_mailbox_word_host(status_memory, status_base + 2));
                    if (cursor > source.sample_count ||
                        load_mailbox_word_host(
                            h_gdr_data,
                            static_cast<size_t>(source.route_id) * kGdrMailboxStride) != 0) {
                        continue;
                    }

                    std::fill(payload.begin(),
                              payload.begin() + kCandidatePayloadIdsWord,
                              0U);
                    const uint32_t candidate_capacity = static_cast<uint32_t>(
                        config.elems_per_block - kCandidatePayloadIdsWord);
                    uint64_t next_cursor = cursor;
                    uint32_t candidate_count = 0;
                    while (next_cursor < source.sample_count &&
                           candidate_count < candidate_capacity) {
                        const uint32_t sample_id = source.sample_ids[next_cursor++];
                        if (!host_candidate_matches_route(runtime, source, sample_id)) {
                            continue;
                        }
                        payload[kCandidatePayloadIdsWord + candidate_count] = sample_id;
                        ++candidate_count;
                    }
                    payload[kCandidatePayloadCountWord] = candidate_count;
                    payload[kCandidatePayloadNextCursorLowWord] =
                        static_cast<uint32_t>(next_cursor);
                    payload[kCandidatePayloadNextCursorHighWord] =
                        static_cast<uint32_t>(next_cursor >> 32U);
                    payload[kCandidatePayloadFlagsWord] =
                        next_cursor == source.sample_count ? kCandidatePayloadFinal : 0U;
                    payload[kCandidatePayloadGenerationWord] = generation;

                    const size_t payload_slot =
                        static_cast<size_t>(source.route_id) * config.payload_slot_count +
                        buffer_idx;
                    uint32_t* destination =
                        h_payload_memory + payload_slot * config.elems_per_block;
                    std::memcpy(destination,
                                payload.data(),
                                (static_cast<size_t>(candidate_count) +
                                 kCandidatePayloadIdsWord) * sizeof(uint32_t));
                    _mm_sfence();
                    if (load_mailbox_word_host(status_flag, status_base) != 1 ||
                        static_cast<uint32_t>(load_mailbox_word_host(
                            status_memory, status_base + 2)) != generation ||
                        (static_cast<uint32_t>(load_mailbox_word_host(
                             status_memory, status_base + 1)) & 1U) != buffer_idx ||
                        static_cast<uint64_t>(load_mailbox_word_host(
                            status_memory, status_base)) != cursor) {
                        continue;
                    }
                    h_gdr_data[static_cast<size_t>(source.route_id) * kGdrMailboxStride] = 1;
                    _mm_sfence();
                    source_rows_scanned.fetch_add(next_cursor - cursor,
                                                  std::memory_order_relaxed);
                    candidates_emitted.fetch_add(candidate_count,
                                                 std::memory_order_relaxed);
                    chunks_published.fetch_add(1, std::memory_order_relaxed);
                    made_progress = true;
                }
                if (!made_progress) {
                    std::this_thread::yield();
                }
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
    if (stats != nullptr) {
        stats->source_rows_scanned = source_rows_scanned.load(std::memory_order_relaxed);
        stats->candidates_emitted = candidates_emitted.load(std::memory_order_relaxed);
        stats->chunks_published = chunks_published.load(std::memory_order_relaxed);
    }
}

void print_host_filter_summary(const char* backend_name, const HostFilterStats& filter_stats)
{
    const uint64_t total = filter_stats.requests_passed + filter_stats.requests_filtered;
    const double pass_ratio = total > 0
        ? static_cast<double>(filter_stats.requests_passed) / static_cast<double>(total)
        : 0.0;

    std::printf("%s host filter requests served: %llu\n",
                backend_name,
                static_cast<unsigned long long>(filter_stats.requests_served));
    std::printf("%s host filter pass: %llu\n",
                backend_name,
                static_cast<unsigned long long>(filter_stats.requests_passed));
    std::printf("%s host filter fail: %llu\n",
                backend_name,
                static_cast<unsigned long long>(filter_stats.requests_filtered));
    std::printf("%s host filter pass_ratio: %.4f\n",
                backend_name,
                pass_ratio);
    std::printf("%s host filter workers_thread: %d\n",
                backend_name,
                filter_stats.workers_thread);
    if (filter_stats.demand_batch_configured_size > 1) {
        const double average_batch_size = filter_stats.demand_batch_count > 0
            ? static_cast<double>(filter_stats.demand_batch_request_count) /
                  static_cast<double>(filter_stats.demand_batch_count)
            : 0.0;
        const double average_wait_us = filter_stats.demand_batch_count > 0
            ? static_cast<double>(filter_stats.demand_batch_wait_ns) /
                  static_cast<double>(filter_stats.demand_batch_count) / 1000.0
            : 0.0;
        std::printf("%s host demand batch configured_size: %llu\n",
                    backend_name,
                    static_cast<unsigned long long>(
                        filter_stats.demand_batch_configured_size));
        std::printf("%s host demand batch timeout_us: %d\n",
                    backend_name,
                    filter_stats.demand_batch_timeout_us);
        std::printf("%s host demand batch count: %llu\n",
                    backend_name,
                    static_cast<unsigned long long>(filter_stats.demand_batch_count));
        std::printf("%s host demand batch requests: %llu\n",
                    backend_name,
                    static_cast<unsigned long long>(filter_stats.demand_batch_request_count));
        std::printf("%s host demand batch partial_count: %llu\n",
                    backend_name,
                    static_cast<unsigned long long>(filter_stats.demand_partial_batch_count));
        std::printf("%s host demand batch max_size: %llu\n",
                    backend_name,
                    static_cast<unsigned long long>(filter_stats.demand_batch_max_size));
        std::printf("%s host demand batch average_size: %.4f\n",
                    backend_name,
                    average_batch_size);
        std::printf("%s host demand batch average_oldest_wait_us: %.4f\n",
                    backend_name,
                    average_wait_us);
    }
}

void print_host_sync_summary(const char* backend_name, const char* path_name, const HostMailboxSyncStats& stats)
{
    if (!host_stats_enabled()) {
        return;
    }
    const uint64_t stale_total = stats.stale_status_not_waiting +
        stats.stale_generation_mismatch +
        stats.stale_row_mismatch +
        stats.stale_buffer_mismatch +
        stats.stale_mailbox_not_zero;
    std::printf("%s %s duplicate observations: %llu\n",
                backend_name,
                path_name,
                static_cast<unsigned long long>(stats.duplicate_observations));
    std::printf("%s %s stale total: %llu\n",
                backend_name,
                path_name,
                static_cast<unsigned long long>(stale_total));
    std::printf("%s %s stale status_not_waiting: %llu\n",
                backend_name,
                path_name,
                static_cast<unsigned long long>(stats.stale_status_not_waiting));
    std::printf("%s %s stale generation_mismatch: %llu\n",
                backend_name,
                path_name,
                static_cast<unsigned long long>(stats.stale_generation_mismatch));
    std::printf("%s %s stale row_mismatch: %llu\n",
                backend_name,
                path_name,
                static_cast<unsigned long long>(stats.stale_row_mismatch));
    std::printf("%s %s stale buffer_mismatch: %llu\n",
                backend_name,
                path_name,
                static_cast<unsigned long long>(stats.stale_buffer_mismatch));
    std::printf("%s %s stale mailbox_not_zero: %llu\n",
                backend_name,
                path_name,
                static_cast<unsigned long long>(stats.stale_mailbox_not_zero));
}
