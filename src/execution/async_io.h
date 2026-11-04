#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <immintrin.h>

#include "data/dataloader.h"

namespace go_fann::route_predicate { struct Plan; }

constexpr size_t kGdrMailboxStride = 16;
constexpr size_t kMailboxSlotAlignmentBytes = 256;

enum class HostRequestKind : uint32_t {
    Demand = 1,
    Done = 2,
};

constexpr uint32_t kCandidatePayloadCountWord = 0;
constexpr uint32_t kCandidatePayloadNextCursorLowWord = 1;
constexpr uint32_t kCandidatePayloadNextCursorHighWord = 2;
constexpr uint32_t kCandidatePayloadFlagsWord = 3;
constexpr uint32_t kCandidatePayloadGenerationWord = 4;
constexpr uint32_t kCandidatePayloadIdsWord = 5;
constexpr uint32_t kCandidatePayloadFinal = 1U;

struct alignas(16) HostRequestRingEntry {
    uint64_t published_sequence = 0;
    uint32_t query_id = UINT32_MAX;
    uint32_t kind = 0;
};

static_assert(sizeof(HostRequestRingEntry) == 16,
              "host request ring entries must remain compact");

struct HostRequestRingView {
    HostRequestRingEntry* entries = nullptr;
    uint64_t* consumed_sequences = nullptr;
    size_t ring_count = 0;
    size_t capacity_per_ring = 0;
    // Selects volatile BAR writeback for GDR-mapped device storage.
    bool consumed_sequences_device_mapped = false;
};

enum class HostCompletionKind : uint8_t {
    Ready = 0,
    Done = 1,
};

struct HostCompletionEvent {
    HostCompletionKind kind = HostCompletionKind::Ready;
    uint32_t query_id = UINT32_MAX;
    uint32_t generation = 0;
    uint32_t row = UINT32_MAX;
    uint32_t buffer = 0;
    uint64_t published_ns = 0;
};

class HostCompletionSink {
public:
    virtual ~HostCompletionSink() = default;

    virtual void cancel() noexcept = 0;

    virtual bool publish_ready(uint32_t query_id,
                               uint32_t generation,
                               uint32_t row,
                               uint32_t buffer) = 0;
    virtual bool publish_ready_batch(const HostCompletionEvent* events, size_t count)
    {
        if (events == nullptr && count != 0) return false;
        bool accepted = true;
        for (size_t idx = 0; idx < count; ++idx) {
            const HostCompletionEvent& event = events[idx];
            accepted = publish_ready(event.query_id,
                                     event.generation,
                                     event.row,
                                     event.buffer) && accepted;
        }
        return accepted;
    }
    virtual bool publish_done(uint32_t query_id) = 0;
};

inline bool publish_host_ready(HostCompletionSink* sink,
                               uint32_t query_id,
                               uint32_t generation,
                               uint32_t row,
                               uint32_t buffer)
{
    return sink != nullptr && sink->publish_ready(query_id, generation, row, buffer);
}

inline bool publish_host_ready_batch(HostCompletionSink* sink,
                                     const HostCompletionEvent* events,
                                     size_t count)
{
    return sink != nullptr && sink->publish_ready_batch(events, count);
}

inline bool host_demand_batch_should_flush(size_t batch_size,
                                           size_t configured_batch_size,
                                           uint64_t oldest_wait_ns,
                                           uint64_t timeout_ns) noexcept
{
    return batch_size != 0 &&
           (batch_size >= configured_batch_size || oldest_wait_ns >= timeout_ns);
}

inline bool publish_host_done(HostCompletionSink* sink, uint32_t query_id)
{
    return sink != nullptr && sink->publish_done(query_id);
}

struct AsyncIoConfig {
    uint32_t dim = 128;
    uint32_t degree = 64;
    uint32_t prefetch_hops = 3;
    uint32_t payload_slot_count = 1;
    int host_service_threads = 1;
    int host_service_cpu_start = -1;
    int host_service_cpu_count = 0;
    size_t host_demand_batch_size = 1;
    int host_demand_batch_timeout_us = 0;
    bool per_step_full_filter = true;
    bool enable_full_label_check = true;
    // Borrowed context-owned state; low-frequency/provider candidates never
    // inherit graph implication or a GPU shortlist's verified provenance.
    const go_fann::route_predicate::Plan* predicate_plan = nullptr;
    const std::vector<uint8_t>* predicate_gpu_verified_routes = nullptr;
    bool reserve_vector_payload = true;
    bool publish_valid_vector_payload = true;
    bool skip_filtered_vector_payload = true;
    bool include_sample_ids = false;
    DenseVectorScalarType vector_payload_scalar_type = DenseVectorScalarType::Uint8;
    size_t minimum_block_size = 0;
    size_t block_size = 4096;
    size_t header_size = 4096;
    size_t vector_payload_words = 0;
    size_t sample_id_words = 0;
    size_t neighbor_count_word = 0;
    size_t neighbor_words_offset = 0;
    size_t neighbor_sample_words_offset = 0;
    size_t elems_per_node = 0;
    size_t elems_per_block = 0;
    size_t nodes_per_block = 0;
    // Borrowed for one host-service invocation. Persistent contexts keep this null.
    HostCompletionSink* completion_sink = nullptr;

    AsyncIoConfig(uint32_t dim_value = 128,
                  uint32_t degree_value = 64,
                  uint32_t prefetch_hops_value = 3,
                  size_t block_size_value = 0,
                  size_t header_size_value = 4096);
    void refresh_layout();
    void set_reserve_vector_payload(bool reserve);
    void set_vector_payload_scalar_type(DenseVectorScalarType scalar_type);
};

struct HostCandidateSourceView {
    uint32_t route_id = UINT32_MAX;
    uint32_t primary_label = UINT32_MAX;
    QueryCandidateSource source_kind = QueryCandidateSource::None;
    const uint32_t* sample_ids = nullptr;
    uint64_t sample_count = 0;
};

struct HostCandidateStreamStats {
    uint64_t source_rows_scanned = 0;
    uint64_t candidates_emitted = 0;
    uint64_t chunks_published = 0;
};

struct GraphNodeView {
    const GraphRuntime* runtime = nullptr;
    // Optional primary-runtime predicate view for a secondary range graph.
    // Graph rows/ranges still come from runtime; only hybrid label residuals
    // are evaluated against this base-row CSR and original DNF table.
    const GraphRuntime* hybrid_residual_runtime = nullptr;
    const uint8_t* base_vector_data = nullptr;
    uint32_t base_vector_rows = 0;
    uint32_t base_vector_cols = 0;
    const uint32_t* graph_data = nullptr;
    uint64_t graph_rows = 0;
    uint64_t graph_degree = 0;
    const uint32_t* label_size = nullptr;
    const uint32_t* label_offset = nullptr;
    const uint32_t* row_to_sample = nullptr;
    const uint64_t* row_filter_mask = nullptr;
    bool implicit_all_ones_row_filter_mask = false;
    const uint64_t* query_filter_bits = nullptr;
    const uint32_t* query_group_ids = nullptr;
    const uint32_t* query_filter_labels = nullptr;
    const uint32_t* label_to_group = nullptr;
    const std::vector<std::vector<uint32_t>>* groups = nullptr;
    const SparseLabelMatrix* data_labels = nullptr;
    const SparseLabelMatrix* query_labels_matrix = nullptr;
    uint32_t label_count = 0;
    uint32_t total_rows = 0;
};

inline uint64_t graph_row_filter_mask_at(const GraphNodeView& view,
                                           uint32_t global_row)
{
    return view.implicit_all_ones_row_filter_mask
        ? UINT64_MAX
        : view.row_filter_mask[global_row];
}

struct HostFilterStats {
    uint64_t requests_served = 0;
    uint64_t requests_passed = 0;
    uint64_t requests_filtered = 0;
    int workers_thread = 0;
    uint64_t demand_batch_count = 0;
    uint64_t demand_partial_batch_count = 0;
    uint64_t demand_batch_request_count = 0;
    uint64_t demand_batch_wait_ns = 0;
    uint64_t demand_batch_max_size = 0;
    uint64_t demand_batch_configured_size = 1;
    int demand_batch_timeout_us = 0;
    uint64_t request_ring_batches_claimed = 0;
    uint64_t request_ring_demand_events = 0;
    uint64_t request_ring_done_events = 0;
    uint64_t request_ring_empty_polls = 0;
    bool request_ring_profile_enabled = false;
    uint64_t request_ring_profile_sample_period = 0;
    uint64_t request_ring_profile_samples = 0;
    uint64_t request_ring_mailbox_snapshot_ns = 0;
    uint64_t request_ring_filter_ns = 0;
    uint64_t request_ring_payload_ns = 0;
    uint64_t request_ring_publish_ns = 0;
    uint64_t request_ring_service_ns = 0;
    uint64_t request_ring_worker_wall_ns = 0;
    uint64_t request_ring_worker_max_wall_ns = 0;
};

struct HostMailboxSyncStats {
    uint64_t duplicate_observations = 0;
    uint64_t stale_status_not_waiting = 0;
    uint64_t stale_generation_mismatch = 0;
    uint64_t stale_row_mismatch = 0;
    uint64_t stale_buffer_mismatch = 0;
    uint64_t stale_mailbox_not_zero = 0;
};

inline bool graph_status_flag_finished(int64_t flag)
{
    return flag == 2 || flag == 3;
}

inline bool graph_status_flag_result_ready(int64_t flag)
{
    return flag == 3;
}

GraphNodeView make_graph_node_view(const GraphRuntime& runtime);
uint32_t graph_global_row(const GraphNodeView& view,
                            uint32_t group_id,
                            uint32_t local_row);
void assemble_graph_node(const GraphNodeView& view,
                           uint32_t group_id,
                           uint32_t local_row,
                           Node& node,
                           const AsyncIoConfig& config = AsyncIoConfig(),
                           HostFilterStats* filter_stats = nullptr);
void serialize_graph_neighbors_words(const uint32_t* row_neighbors,
                                       uint32_t label_size,
                                       uint32_t* dst_words,
                                       const AsyncIoConfig& config,
                                       uint32_t graph_degree);
void serialize_graph_neighbor_row_words(const GraphNodeView& view,
                                          uint32_t group_id,
                                          uint32_t local_row,
                                          const uint32_t* row_neighbors,
                                          uint32_t* dst_words,
                                          const AsyncIoConfig& config,
                                          uint32_t graph_degree);
void serialize_graph_node_words(const Node& node,
                                  uint32_t* dst_words,
                                  const AsyncIoConfig& config = AsyncIoConfig());
void serialize_graph_float_node_row_words(const GraphNodeView& view,
                                            uint32_t group_id,
                                            uint32_t local_row,
                                            const uint32_t* row_neighbors,
                                            uint32_t* dst_words,
                                            const AsyncIoConfig& config,
                                            uint32_t graph_degree);
void serialize_graph_node_row_words(const GraphNodeView& view,
                                      uint32_t group_id,
                                      uint32_t local_row,
                                      const uint32_t* row_neighbors,
                                      uint32_t* dst_words,
                                      const AsyncIoConfig& config,
                                      uint32_t graph_degree,
                                      HostFilterStats* filter_stats = nullptr);
int64_t compute_host_filter_result(const GraphNodeView& view,
                                   uint32_t sample_id,
                                   uint64_t row_filter_mask,
                                   uint32_t query_idx,
                                   bool enable_full_label_check,
                                   const AsyncIoConfig* config = nullptr);
bool host_route_predicate_gpu_verified(const AsyncIoConfig& config, uint32_t route);
bool should_publish_graph_vector_payload(const AsyncIoConfig& config,
                                           bool filter_passed);
void build_prefetch_slices(GraphRuntime& runtime,
                           const AsyncIoConfig& config = AsyncIoConfig());

void host_scan_and_set(int64_t* status_memory,
                       uint32_t* h_neighbor_memory,
                       int64_t* h_gdr_data,
                       int* req_per_loop,
                       int64_t* status_flag,
                       const GraphNodeView& host_view,
                       size_t total_queries,
                       const uint32_t* active_query_ids = nullptr,
                       size_t active_query_count = 0,
                       const AsyncIoConfig& config = AsyncIoConfig(),
                       HostFilterStats* filter_stats = nullptr);

void host_request_ring_and_set(int64_t* status_memory,
                               uint32_t* h_neighbor_memory,
                               int64_t* h_gdr_data,
                               int64_t* status_flag,
                               const GraphNodeView& host_view,
                               size_t total_queries,
                               const uint32_t* active_query_ids,
                               size_t active_query_count,
                               const HostRequestRingView& request_ring,
                               const std::atomic<bool>* cancellation = nullptr,
                               const AsyncIoConfig& config = AsyncIoConfig(),
                               HostFilterStats* filter_stats = nullptr);

void host_candidate_scan_and_set(int64_t* status_memory,
                                 uint32_t* h_payload_memory,
                                 int64_t* h_gdr_data,
                                 int64_t* status_flag,
                                 const GraphRuntime& runtime,
                                 const HostCandidateSourceView* sources,
                                 size_t source_count,
                                 const AsyncIoConfig& config,
                                 HostCandidateStreamStats* stats = nullptr);

void set_full_filter_enabled(bool enabled);
bool full_filter_enabled();

void print_host_filter_summary(const char* backend_name, const HostFilterStats& filter_stats);
void print_host_sync_summary(const char* backend_name, const char* path_name, const HostMailboxSyncStats& stats);
