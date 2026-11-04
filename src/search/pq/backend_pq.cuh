#pragma once

#include "execution/cpu_rerank_schedule.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "data/dataloader.h"
#include "execution/exact_distance_memo.h"
#include "app/runtime_defaults.h"
#include "type_def.h"

namespace go_fann::backend_pq {

struct PqFiles {
    std::string compressed_path;
    std::string pivots_path;
};

struct PqHostIndex {
    uint64_t num_points = 0;
    uint64_t num_chunks = 0;
    std::vector<pq_idx_t> compressed_codes;
    std::vector<pq_value_t> centroid_table;
};

struct PqImmutableDeviceState;
struct PqPreparedQueryState;
struct PqGraphSearchContext;
struct PqLowfreqSearchContext;

enum class PqFiniteContextState : uint32_t {
    UNINITIALIZED = 0,
    WAIT_IO = 1,
    READY = 2,
    RUNNING = 3,
    DONE = 4,
};

struct alignas(16) PqFiniteContextControl {
    uint32_t query_id = 0;
    uint32_t context_epoch = 0;
    PqFiniteContextState state = PqFiniteContextState::UNINITIALIZED;
    uint32_t io_generation = 0;
    float now_distance = 0.0f;
    uint32_t now_payload = 0;
    float next_distance = 0.0f;
    uint32_t next_payload = 0;
    uint32_t frontier_size = 0;
    uint32_t stop_heap_size = 0;
    uint32_t result_heap_size = 0;
    uint32_t expanded_count = 0;
    uint32_t pass_count = 0;
    uint32_t step_num = 0;
    uint32_t finished = 0;
    uint32_t pool_exhausted = 0;
    uint32_t current_buffer = 0;
    uint32_t expected_row = 0;
    float stop_worst = 0.0f;
    uint32_t stop_worst_bits = 0;
    uint64_t state_since_ns = 0;
};

struct alignas(64) PqFiniteContextTelemetry {
    float io_time_us = 0.0f;
    float calculation_time_us = 0.0f;
    float queue_time_us = 0.0f;
    uint32_t resumed_step_count = 0;
    uint64_t wait_io_ns = 0;
    uint64_t ready_ns = 0;
    uint64_t running_ns = 0;
    uint32_t resume_count = 0;
    uint32_t park_count = 0;
    uint32_t terminal_count = 0;
    uint32_t static_prefetch_step_count = 0;
    uint32_t multi_step_resume_count = 0;
    uint32_t max_resume_quantum_steps = 0;
};

enum class PqFiniteAdvanceResult : uint32_t {
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

struct PqFiniteReadyToken {
    uint32_t query_id = 0;
    uint32_t context_epoch = 0;
    uint32_t generation = 0;
    uint32_t row = 0;
    uint32_t buffer = 0;
    PqFiniteAdvanceResult result = PqFiniteAdvanceResult::PENDING;
    uint64_t ready_ns = 0;
};

static_assert(sizeof(PqFiniteContextControl) == 96,
              "PQ finite hot control must remain a 96-byte record");
static_assert(alignof(PqFiniteContextControl) == 16,
              "PQ finite hot control must remain 16-byte aligned");
static_assert(offsetof(PqFiniteContextControl, state) == 8);
static_assert(offsetof(PqFiniteContextControl, now_distance) == 16);
static_assert(offsetof(PqFiniteContextControl, frontier_size) == 32);
static_assert(offsetof(PqFiniteContextControl, current_buffer) == 64);
static_assert(offsetof(PqFiniteContextControl, state_since_ns) == 80);
static_assert(sizeof(PqFiniteContextTelemetry) == 64,
              "PQ finite cold telemetry must remain one 64-byte record");
static_assert(alignof(PqFiniteContextTelemetry) == 64,
              "PQ finite cold telemetry must remain 64-byte aligned");
static_assert(offsetof(PqFiniteContextTelemetry, resumed_step_count) == 12);
static_assert(offsetof(PqFiniteContextTelemetry, wait_io_ns) == 16);
static_assert(offsetof(PqFiniteContextTelemetry, resume_count) == 40);
static_assert(offsetof(PqFiniteContextTelemetry, static_prefetch_step_count) == 52);
static_assert(offsetof(PqFiniteContextTelemetry, multi_step_resume_count) == 56);
static_assert(offsetof(PqFiniteContextTelemetry, max_resume_quantum_steps) == 60);
static_assert(sizeof(PqFiniteReadyToken) == 32,
              "PQ finite ready token must remain a compact 32-byte record");

struct PqSearchOptions {
    cpu_rerank_schedule::Mode cpu_rerank_schedule = cpu_rerank_schedule::Mode::Native;
    // Internal application policy: native in-memory search uses raw base scores.
    bool raw_navigation_scores = false;
    std::string lowfreq_scan_mode = "host";
    std::string posting_predicate_order = "filter_first";
    bool posting_streaming = false;
    bool rerank_distance_reuse = false;
    uint32_t posting_stream_batch_size = 1024;
    // Internal iteration-owned distance sharing, never a persisted query cache.
    std::shared_ptr<ExactDistanceMemo> exact_distance_memo;
    // Main resolves automatic defaults against supported execution paths.
    std::string predicate_mode = "cpu";
    std::string predicate_storage = "csr";
    float predicate_bias_alpha = 0.0f;
    bool predicate_diagnostics = false;
    PqFiles files;
    std::string graph_transport = "mailbox";
    std::string query_worker_mode = std::string(defaults::default_pq_query_worker_mode);
    std::string request_mode = "scan";
    size_t request_batch_queries = 128;
    std::string request_consumed_memory = "device";
    std::string finite_token_transport =
        std::string(defaults::default_pq_finite_token_transport);
    std::string finite_dispatch_mode =
        std::string(defaults::default_pq_finite_dispatch_mode);
    int finite_partial_batch_timeout_us =
        defaults::default_pq_finite_partial_batch_timeout_us;
    std::string graph_path;
    std::string graph_storage_mode = defaults::graph_storage_mode;
    bool skip_filtered_vector_payload = true;
    bool per_step_full_filter = true;
    std::string highfreq_rerank_mode = std::string(defaults::default_highfreq_rerank_mode);
    std::string highfreq_rerank_algorithm = std::string(defaults::default_highfreq_rerank_algorithm);
    bool highfreq_rerank_streaming = defaults::default_highfreq_rerank_streaming;
    std::string data_label_spmat_path;
    std::string query_label_spmat_path;
    std::string query_manifest_jsonl_path;
    std::string query_manifest_families;
    std::string query_manifest_subtypes;
    uint32_t query_manifest_max_or_terms = 64;
    uint32_t query_manifest_max_fuzzy_terms = 64;
    float query_manifest_min_ngram_score = 0.25f;
    uint32_t query_manifest_ngram_size = 3;
    std::string base_vector_path;
    std::string base_range_fbin_path;
    std::string query_vector_path;
    std::string query_range_path;
    std::string range_label_manifest_path;
    std::string seed_sidecar_path;
    std::string original_label_seed_sidecar_path;
    std::string metadata_sidecar_path;
    std::string seed_selection_mode = "group";
    std::string lowfreq_index_path;
    std::string group_path;
    std::string output_csv_path = "results_100.csv";
    bool full_filter = false;
    uint32_t dataset_cutoff = 0;
    int specificity_threshold = 2000;
    int dim = 128;
    int pq_dim = 64;
    int degree = 64;
    int topk = 10;
    int highfreq_rerank_shortlist = defaults::default_highfreq_rerank_shortlist;
    int highfreq_rerank_batch_size = defaults::default_lowfreq_rerank_batch_size;
    float highfreq_rerank_epsilon = defaults::default_lowfreq_rerank_epsilon;
    int highfreq_rerank_stable_batches = defaults::default_lowfreq_rerank_stable_batches;
    int highfreq_rerank_threads = defaults::default_highfreq_rerank_threads;
    bool cpu_persistent_workers = false;
    int highfreq_rerank_stream_batch_size = defaults::default_highfreq_rerank_stream_batch_size;
    int lowfreq_rerank_shortlist = defaults::default_lowfreq_rerank_shortlist;
    int lowfreq_rerank_batch_size = defaults::default_lowfreq_rerank_batch_size;
    float lowfreq_rerank_epsilon = defaults::default_lowfreq_rerank_epsilon;
    int lowfreq_rerank_stable_batches = defaults::default_lowfreq_rerank_stable_batches;
    bool enable_lowfreq_rerank = true;
    std::string lowfreq_full_filter_mode = defaults::default_lowfreq_full_filter_mode;
    int prefetch_hops = 3;
    float quick_filter_bonus = 0.95f;
    float bit_filter_bonus = 0.95f;
    float range_filter_bonus = 0.95f;
    double and_dual_route_posting_ratio_threshold = defaults::default_and_dual_route_posting_ratio_threshold;
    int warmup_runs = 0;
    int timed_runs = 1;
    bool full_rerank_warmup = false;
    std::vector<int> latency_batch_sizes;
    std::vector<int> latency_concurrency_levels;
    std::string latency_output_path = "latency_sweep.csv";
    std::string latency_trace_output_path;
    int latency_query_start = 0;
    int latency_query_limit = 0;
    bool latency_run_all_queries = false;
    bool latency_time_planner = false;
    bool latency_online_metadata = false;
    bool latency_staggered = false;
    bool latency_persistent_queue = false;
    std::string latency_lowfreq_mode = "resident";
    int process_cpu_start = -1;
    int process_cpu_count = 0;
    int host_service_threads = 1;
    int host_service_cpu_start = -1;
    int host_service_cpu_count = 0;
    size_t host_demand_batch_size = 1;
    int host_demand_batch_timeout_us = 0;
    int search_queue_size = 50;
    int search_finish_count = 1;
    int search_max_steps = 0;
    std::string search_early_stop_mode = defaults::default_search_early_stop_mode;
    bool print_summary = true;
    std::string buffer_mode = "double_slot_serial";
    std::string host_reply_wait_policy = "fixed";
    std::string routing_mode = "greedy_min_posting";
};

struct PqIterationResult {
    std::vector<std::vector<idx_t>> results;
    std::vector<int> step_nums;
    double qps = 0.0;
    double kernel_ms = 0.0;
    size_t qps_run_count = 0;
    int64_t total_steps = 0;
    size_t highfreq_route_count = 0;
    size_t lowfreq_route_count = 0;
    size_t unroutable_route_count = 0;
    int64_t highfreq_total_steps = 0;
    int64_t lowfreq_total_steps = 0;
    float pass_ratio = 0.0f;
    int workers_thread = 0;
    std::vector<int> route_flags;
};

struct PqTimedRunMetrics {
    double wall_ms = 0.0;
    double qps = 0.0;
    size_t query_count = 0;
    int64_t total_steps = 0;
    int64_t highfreq_steps = 0;
    int64_t lowfreq_steps = 0;
};

PqHostIndex load_pq_index(const PqFiles& files, size_t expected_rows, int dim = 128, int pq_dim = 64);
int run_pq_search(const PqSearchOptions& options);
int run_pq_latency_benchmark(const PqSearchOptions& options);
int run_benchmark_sweep(const PqSearchOptions& base_options,
                        const std::vector<float>& quick_filter_bonuses,
                        const std::vector<int>& queue_sizes,
                        const std::string& output_csv_path);
int run_pq_search_with_runtime(const PqSearchOptions& options,
                               const GraphRuntime& runtime,
                               const PqHostIndex& pq_index,
                               const std::vector<uint32_t>& query_indices,
                               std::vector<std::vector<idx_t>>& out_results,
                               std::vector<int>& out_step_nums,
                               std::vector<std::vector<float>>* out_distances = nullptr);
int run_lowfreq_pq_search_with_runtime(const PqSearchOptions& options,
                                       const GraphRuntime& runtime,
                                       const PqHostIndex& pq_index,
                                       const std::vector<uint32_t>& query_indices,
                                       std::vector<std::vector<idx_t>>& out_results,
                                       std::vector<int>& out_step_nums,
                                       std::vector<std::vector<float>>* out_distances = nullptr);
void merge_subset_results(const std::vector<uint32_t>& query_indices,
                          std::vector<std::vector<idx_t>>& merged_results,
                          std::vector<int>& merged_steps,
                          std::vector<std::vector<idx_t>>& subset_results,
                          std::vector<int>& subset_steps);
void init_benchmark_contexts(PqImmutableDeviceState& immutable,
                             PqPreparedQueryState& prepared,
                             PqGraphSearchContext& graph_ctx,
                             PqLowfreqSearchContext& lowfreq_ctx,
                             const GraphRuntime& runtime,
                             const PqHostIndex& pq_index,
                             const PqSearchOptions& base_options);
void reset_search_state(PqGraphSearchContext& ctx, size_t query_count);
PqIterationResult execute_single_iteration(PqGraphSearchContext& graph_ctx,
                                           PqLowfreqSearchContext& lowfreq_ctx,
                                           const PqPreparedQueryState& prepared,
                                           const PqImmutableDeviceState& immutable,
                                           GraphRuntime& runtime,
                                           const PqSearchOptions& iteration_options,
                                           const std::function<void(int)>& before_timed_run = {},
                                           const std::function<void(int, const PqTimedRunMetrics&)>&
                                               after_timed_run = {});
void cleanup_benchmark_contexts(PqImmutableDeviceState& immutable,
                                PqPreparedQueryState& prepared,
                                PqGraphSearchContext& graph_ctx,
                                PqLowfreqSearchContext& lowfreq_ctx);
void save_results_to_csv(const std::vector<std::vector<idx_t>>& results, const std::string& filename);

__device__ void computePQTable(const pq_value_t* d_pq_centroid,
                               const value_t* d_query,
                               pq_value_t* pq_table,
                               int pq_dim,
                               int k,
                               int dim,
                               int tid);

}
