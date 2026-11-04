#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <optional>
#include <cstdlib>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <nlohmann/json.hpp>

#include "search/full_gpu/backend_full_gpu.cuh"
#include "profiling/expansion_trace.h"
#include "search/pq/backend_pq.cuh"
#include "search/rabitq/backend_rabitq.cuh"
#include "execution/async_io.h"
#include "app/config_loader.h"
#include "data/dataloader.h"
#include "index/graph_builder.h"
#include "app/runtime_defaults.h"
#include "predicates/predicate_defaults.h"
#include "search/common/navigation_score_policy.h"

namespace go_fann {

struct AppOptions {
    std::string mode = "search";
    std::string backend = "pq";
    std::string cpu_rerank_schedule = "native";
    std::string graph_builder = "cagra";
    std::string graph_path = defaults::graph_path;
    std::string graph_file_path;
    std::string graph_storage_mode = defaults::graph_storage_mode;
    std::string rerank_vector_storage_mode =
        std::string(defaults::rerank_vector_storage_mode_default);
    std::string rerank_vector_artifact_path;
    bool skip_filtered_vector_payload = true;
    bool per_step_full_filter = true;
    std::string highfreq_rerank_mode = std::string(defaults::default_highfreq_rerank_mode);
    std::string highfreq_rerank_algorithm = std::string(defaults::default_highfreq_rerank_algorithm);
    bool highfreq_rerank_streaming = defaults::default_highfreq_rerank_streaming;
    bool highfreq_rerank_merge_routes = true;
    bool rerank_distance_reuse = false;
    int rabitq_incremental_rerank_shortlist = 0;
    bool lowfreq_rerank_merge_routes = defaults::default_lowfreq_rerank_merge_routes;
    std::string lowfreq_index_path;
    std::string base_vector_path = defaults::base_vector_path;
    std::string base_range_fbin_path;
    std::string range_posting_index_path;
    std::string range_posting_mode = "off";
    uint64_t range_posting_scan_threshold = 0;
    int range_posting_shortlist = defaults::default_range_posting_shortlist;
    int range_posting_postfilter_shortlist_multiplier =
        defaults::default_range_posting_postfilter_shortlist_multiplier;
    std::string range_posting_and_filter_mode = "auto";
    std::string range_label_and_policy = "auto";
    bool range_posting_drop_wide_hybrid_or = false;
    bool range_graph_hybrid_or_enabled = false;
    bool range_graph_hybrid_and_enabled = false;
    bool rabitq_dual_graph_cpu_gpu_overlap = false;
    std::string rabitq_posting_schedule = "legacy";
    uint32_t rabitq_range_posting_segment_rows = 0;
    bool rabitq_posting_streaming = false;
    bool rabitq_graph_persistent_workers = false;
    bool rabitq_posting_persistent_workers = false;
    uint32_t rabitq_posting_stream_batch_size = 1024;
    std::string range_graph_path;
    std::string range_graph_storage_mode = "memory";
    std::string range_graph_data_label_spmat_path;
    std::string range_graph_label_manifest_path;
    std::string range_graph_seed_sidecar_path;
    std::string range_graph_original_label_seed_sidecar_path;
    std::string range_graph_metadata_sidecar_path;
    std::string range_graph_lowfreq_index_path;
    std::string range_graph_group_path;
    std::string data_label_spmat_path = defaults::data_label_spmat_path;
    std::string query_vector_path = defaults::query_vector_path;
    std::string query_label_spmat_path = defaults::query_label_spmat_path;
    std::string query_manifest_jsonl_path;
    std::string query_manifest_families;
    std::string query_manifest_subtypes;
    uint32_t query_manifest_max_or_terms = 64;
    uint32_t query_manifest_max_fuzzy_terms = 64;
    std::string query_manifest_fuzzy_cap_policy = "legacy";
    uint32_t query_prepare_threads = 1;
    bool query_prepare_optimized = true;
    float query_manifest_min_ngram_score = 0.25f;
    uint32_t query_manifest_ngram_size = 3;
    std::string query_manifest_operator_index_path;
    std::string query_manifest_operator_fill_mode = "legacy_full_fallback";
    bool query_manifest_operator_parity_check = false;
    std::string or_route_order_mode = "legacy";
    std::string or_route_sketch_sidecar_path;
    std::string or_route_accuracy_cost_model_path;
    std::string or_route_calibration_trace_path;
    std::string or_route_calibration_run_id;
    uint32_t or_route_sketch_k = kOrRouteSketchDefaultK;
    uint64_t or_route_sketch_hash_seed = kOrRouteSketchDefaultHashSeed;
    bool or_route_allow_runtime_sketch_fallback = false;
    uint32_t or_route_runtime_sketch_fallback_max_rows = 100000;
    bool or_route_verify_sketch_payload_checksum = true;
    std::string query_range_path;
    std::string range_label_manifest_path;
    std::string seed_sidecar_path = defaults::seed_sidecar_path;
    std::string original_label_seed_sidecar_path = defaults::original_label_seed_sidecar_path;
    std::string metadata_sidecar_path;
    std::string seed_selection_mode = defaults::default_seed_selection_mode;
    std::string pq_compressed_path = defaults::pq_compressed_path;
    std::string pq_pivots_path = defaults::pq_pivots_path;
    std::string pq_buffer_mode = "double_slot_serial";
    std::string pq_graph_transport = "mailbox";
    std::string pq_query_worker_mode = defaults::default_pq_query_worker_mode;
    bool pq_cpu_persistent_workers = false;
    std::string pq_request_mode = "scan";
    size_t pq_request_batch_queries = 128;
    std::string pq_request_consumed_memory = "device";
    std::string pq_finite_token_transport = defaults::default_pq_finite_token_transport;
    std::string pq_finite_dispatch_mode = defaults::default_pq_finite_dispatch_mode;
    int pq_finite_partial_batch_timeout_us =
        defaults::default_pq_finite_partial_batch_timeout_us;
    std::string host_reply_wait_policy = "fixed";
    std::string routing_mode = "greedy_min_posting";
    std::string group_path;
    std::string rabitq_artifact_path = defaults::rabitq_artifact_path;
    std::string rabitq_cuvs_artifact_path = defaults::rabitq_cuvs_artifact_path;
    std::string pq_lowfreq_scan_mode = "auto";
    bool pq_posting_streaming = false;
    uint32_t pq_posting_stream_batch_size = 1024;
    std::string posting_predicate_order = "filter_first";
    std::string rabitq_predicate_mode = "auto";
    std::string rabitq_predicate_storage = "csr";
    std::string rabitq_predicate_index_path;
    double rabitq_predicate_bias_alpha = 0.15;
    bool predicate_alpha_explicit = false;
    bool rabitq_predicate_diagnostics = false;
    std::string rabitq_query_worker_mode = "coupled";
    std::string rabitq_finite_dispatch_mode = "batch";
    int rabitq_finite_partial_batch_timeout_us = 50;
    std::string rabitq_graph_transport = "mailbox";
    std::string rabitq_request_mode = "scan";
    size_t rabitq_request_batch_queries = 128;
    std::string rabitq_request_consumed_memory = "host";
    size_t rabitq_gpu_graph_row_cache_mib = 0;
    bool rabitq_request_mode_explicit = false;
    std::string output_csv_path = "results_100.csv";
    std::string config_path;
    bool full_filter = false;
    bool full_gpu_fp16 = false;
    std::string overlap_graph_path;
    std::string overlap_manifest_path;
    uint32_t overlap_max_bridges = 1;

    std::string lowfreq_full_filter_mode = defaults::default_lowfreq_full_filter_mode;
    bool force_rebuild = false;
    bool run_analyze = false;
    uint32_t row_limit = defaults::default_row_limit;
    bool row_limit_explicit = false;
    uint32_t cagra_build_worker_count = 4;
    uint32_t stream_graph_worker_count = 1;
    bool stream_graph_auto_workers = false;
    uint64_t stream_graph_memory_budget_bytes = 0;
    uint64_t stream_graph_per_worker_memory_cap_bytes = 0;
    bool stream_graph_allow_concurrent_cagra = false;
    std::string cagra_ace_build_dir;
    bool cagra_ace_force_disk = false;
    int specificity_threshold = defaults::default_specificity_threshold;
    int prefetch_hops = defaults::default_prefetch_hops;
    float quick_filter_bonus = defaults::default_quick_filter_bonus;
    float bit_filter_bonus = defaults::default_quick_filter_bonus;
    float range_filter_bonus = defaults::default_quick_filter_bonus;
    double and_dual_route_posting_ratio_threshold =
        defaults::default_and_dual_route_posting_ratio_threshold;
    int warmup_runs = 0;
    int timed_runs = 1;
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
    bool predicate_early_stop_explicit = false;
    std::string benchmark_quick_filter_bonuses;
    std::string benchmark_queue_sizes = "64,128,256,512";
    std::string benchmark_output = "benchmark_sweep.csv";
    std::string latency_queue_sizes;
    std::string latency_batch_sizes = "1,2,4,8,16,32,64";
    std::string latency_output = "latency_sweep.csv";
    std::string latency_trace_output;
    int latency_query_start = 0;
    int latency_query_limit = 0;
    std::string latency_concurrency_levels;
    bool latency_run_all_queries = false;
    bool latency_time_planner = false;
    bool latency_online_metadata = false;
    bool latency_staggered = false;
    bool latency_persistent_queue = false;
    bool latency_queue_sweep_full_warmup = false;
    std::string latency_lowfreq_mode = "resident";
    int degree = defaults::default_degree;
    int topk = defaults::default_topk;
    int highfreq_rerank_shortlist = defaults::default_highfreq_rerank_shortlist;
    int highfreq_rerank_batch_size = defaults::default_lowfreq_rerank_batch_size;
    float highfreq_rerank_epsilon = defaults::default_lowfreq_rerank_epsilon;
    int highfreq_rerank_stable_batches = defaults::default_lowfreq_rerank_stable_batches;
    int highfreq_rerank_threads = defaults::default_highfreq_rerank_threads;
    int highfreq_rerank_stream_batch_size = defaults::default_highfreq_rerank_stream_batch_size;
    int lowfreq_rerank_shortlist = defaults::default_lowfreq_rerank_shortlist;
    int lowfreq_rerank_batch_size = defaults::default_lowfreq_rerank_batch_size;
    float lowfreq_rerank_epsilon = defaults::default_lowfreq_rerank_epsilon;
    int lowfreq_rerank_stable_batches = defaults::default_lowfreq_rerank_stable_batches;
    bool enable_lowfreq_rerank = true;
    int dim = defaults::default_dim;
    int pq_dim = defaults::default_pq_dim;
    uint32_t rabitq_num_anchors = defaults::default_rabitq_num_anchors;
    uint32_t rabitq_bits_per_dim = defaults::default_rabitq_bits_per_dim;
    bool rabitq_enable_highfreq_rerank = true;
    bool rabitq_enable_lowfreq_rerank = true;
    bool rabitq_lowfreq_gpu_posting = false;
    bool rabitq_use_lut = false;
    std::string rabitq_query_rotation = "cublas_fp32";
    std::string rabitq_centroid_distances = "cublas_fp32";
    bool rabitq_use_int8_score = false;
    bool rabitq_use_int8_score_explicit = false;
    bool help = false;
};

struct AppOptionOverrides {
    std::optional<std::string> mode;
    std::optional<std::string> backend;
    std::optional<std::string> cpu_rerank_schedule;
    std::optional<std::string> graph_builder;
    std::optional<std::string> graph_path;
    std::optional<std::string> graph_file_path;
    std::optional<std::string> graph_storage_mode;
    std::optional<std::string> rerank_vector_storage_mode;
    std::optional<std::string> rerank_vector_artifact_path;
    std::optional<bool> skip_filtered_vector_payload;
    std::optional<bool> per_step_full_filter;
    std::optional<std::string> highfreq_rerank_mode;
    std::optional<std::string> highfreq_rerank_algorithm;
    std::optional<bool> highfreq_rerank_streaming;
    std::optional<bool> highfreq_rerank_merge_routes;
    std::optional<bool> rerank_distance_reuse;
    std::optional<int> rabitq_incremental_rerank_shortlist;
    std::optional<bool> lowfreq_rerank_merge_routes;
    std::optional<std::string> lowfreq_index_path;
    std::optional<std::string> base_vector_path;
    std::optional<std::string> base_range_fbin_path;
    std::optional<std::string> range_posting_index_path;
    std::optional<std::string> range_posting_mode;
    std::optional<uint64_t> range_posting_scan_threshold;
    std::optional<int> range_posting_shortlist;
    std::optional<int> range_posting_postfilter_shortlist_multiplier;
    std::optional<std::string> range_posting_and_filter_mode;
    std::optional<std::string> range_label_and_policy;
    std::optional<bool> range_posting_drop_wide_hybrid_or;
    std::optional<bool> range_graph_hybrid_or_enabled;
    std::optional<bool> range_graph_hybrid_and_enabled;
    std::optional<std::string> range_graph_path;
    std::optional<std::string> range_graph_storage_mode;
    std::optional<std::string> range_graph_data_label_spmat_path;
    std::optional<std::string> range_graph_label_manifest_path;
    std::optional<std::string> range_graph_seed_sidecar_path;
    std::optional<std::string> range_graph_original_label_seed_sidecar_path;
    std::optional<std::string> range_graph_metadata_sidecar_path;
    std::optional<std::string> range_graph_lowfreq_index_path;
    std::optional<std::string> range_graph_group_path;
    std::optional<std::string> data_label_spmat_path;
    std::optional<std::string> query_vector_path;
    std::optional<std::string> query_label_spmat_path;
    std::optional<std::string> query_manifest_jsonl_path;
    std::optional<std::string> query_manifest_families;
    std::optional<std::string> query_manifest_subtypes;
    std::optional<uint32_t> query_manifest_max_or_terms;
    std::optional<uint32_t> query_manifest_max_fuzzy_terms;
    std::optional<std::string> query_manifest_fuzzy_cap_policy;
    std::optional<uint32_t> query_prepare_threads;
    std::optional<bool> query_prepare_optimized;
    std::optional<float> query_manifest_min_ngram_score;
    std::optional<uint32_t> query_manifest_ngram_size;
    std::optional<std::string> query_range_path;
    std::optional<std::string> range_label_manifest_path;
    std::optional<std::string> seed_sidecar_path;
    std::optional<std::string> original_label_seed_sidecar_path;
    std::optional<std::string> metadata_sidecar_path;
    std::optional<std::string> seed_selection_mode;
    std::optional<std::string> pq_compressed_path;
    std::optional<std::string> pq_pivots_path;
    std::optional<std::string> pq_buffer_mode;
    std::optional<std::string> pq_graph_transport;
    std::optional<std::string> pq_query_worker_mode;
    std::optional<bool> pq_cpu_persistent_workers;
    std::optional<std::string> pq_request_mode;
    std::optional<size_t> pq_request_batch_queries;
    std::optional<std::string> pq_request_consumed_memory;
    std::optional<std::string> pq_finite_token_transport;
    std::optional<std::string> pq_finite_dispatch_mode;
    std::optional<int> pq_finite_partial_batch_timeout_us;
    std::optional<std::string> pq_lowfreq_scan_mode;
    std::optional<bool> pq_posting_streaming;
    std::optional<uint32_t> pq_posting_stream_batch_size;
    std::optional<std::string> posting_predicate_order;
    std::optional<std::string> rabitq_predicate_mode;
    std::optional<std::string> rabitq_predicate_storage;
    std::optional<std::string> rabitq_predicate_index_path;
    std::optional<double> rabitq_predicate_bias_alpha;
    std::optional<bool> rabitq_predicate_diagnostics;
    std::optional<std::string> rabitq_query_worker_mode;
    std::optional<std::string> rabitq_finite_dispatch_mode;
    std::optional<int> rabitq_finite_partial_batch_timeout_us;
    std::optional<std::string> rabitq_graph_transport;
    std::optional<std::string> rabitq_posting_schedule;
    std::optional<uint32_t> rabitq_range_posting_segment_rows;
    std::optional<bool> rabitq_posting_streaming;
    std::optional<bool> rabitq_graph_persistent_workers;
    std::optional<bool> rabitq_posting_persistent_workers;
    std::optional<uint32_t> rabitq_posting_stream_batch_size;
    std::optional<std::string> host_reply_wait_policy;
    std::optional<std::string> routing_mode;
    std::optional<std::string> group_path;
    std::optional<std::string> rabitq_artifact_path;
    std::optional<std::string> rabitq_cuvs_artifact_path;
    std::optional<std::string> output_csv_path;
    std::optional<std::string> config_path;
    std::optional<bool> full_filter;
    std::optional<bool> full_gpu_fp16;
    std::optional<std::string> overlap_graph_path;
    std::optional<std::string> overlap_manifest_path;
    std::optional<uint32_t> overlap_max_bridges;

    std::optional<std::string> lowfreq_full_filter_mode;
    std::optional<bool> force_rebuild;
    std::optional<bool> run_analyze;
    std::optional<uint32_t> row_limit;
    std::optional<uint32_t> cagra_build_worker_count;
    std::optional<uint32_t> stream_graph_worker_count;
    std::optional<bool> stream_graph_auto_workers;
    std::optional<uint64_t> stream_graph_memory_budget_bytes;
    std::optional<uint64_t> stream_graph_per_worker_memory_cap_bytes;
    std::optional<bool> stream_graph_allow_concurrent_cagra;
    std::optional<int> specificity_threshold;
    std::optional<int> prefetch_hops;
    std::optional<float> quick_filter_bonus;
    std::optional<float> bit_filter_bonus;
    std::optional<float> range_filter_bonus;
    std::optional<double> and_dual_route_posting_ratio_threshold;
    std::optional<int> warmup_runs;
    std::optional<int> timed_runs;
    std::optional<int> process_cpu_start;
    std::optional<int> process_cpu_count;
    std::optional<int> host_service_threads;
    std::optional<int> host_service_cpu_start;
    std::optional<int> host_service_cpu_count;
    std::optional<size_t> host_demand_batch_size;
    std::optional<int> host_demand_batch_timeout_us;
    std::optional<std::string> rabitq_request_mode;
    std::optional<size_t> rabitq_request_batch_queries;
    std::optional<std::string> rabitq_request_consumed_memory;
    std::optional<size_t> rabitq_gpu_graph_row_cache_mib;
    std::optional<int> search_queue_size;
    std::optional<int> search_finish_count;
    std::optional<int> search_max_steps;
    std::optional<std::string> search_early_stop_mode;
    std::optional<std::string> benchmark_quick_filter_bonuses;
    std::optional<std::string> benchmark_queue_sizes;
    std::optional<std::string> benchmark_output;
    std::optional<std::string> latency_queue_sizes;
    std::optional<std::string> latency_batch_sizes;
    std::optional<std::string> latency_output;
    std::optional<std::string> latency_trace_output;
    std::optional<int> latency_query_start;
    std::optional<int> latency_query_limit;
    std::optional<std::string> latency_concurrency_levels;
    std::optional<bool> latency_run_all_queries;
    std::optional<bool> latency_time_planner;
    std::optional<bool> latency_online_metadata;
    std::optional<bool> latency_staggered;
    std::optional<bool> latency_persistent_queue;
    std::optional<bool> latency_queue_sweep_full_warmup;
    std::optional<std::string> latency_lowfreq_mode;
    std::optional<int> degree;
    std::optional<int> topk;
    std::optional<int> highfreq_rerank_shortlist;
    std::optional<int> highfreq_rerank_batch_size;
    std::optional<float> highfreq_rerank_epsilon;
    std::optional<int> highfreq_rerank_stable_batches;
    std::optional<int> highfreq_rerank_threads;
    std::optional<int> highfreq_rerank_stream_batch_size;
    std::optional<int> lowfreq_rerank_shortlist;
    std::optional<int> lowfreq_rerank_batch_size;
    std::optional<float> lowfreq_rerank_epsilon;
    std::optional<int> lowfreq_rerank_stable_batches;
    std::optional<bool> enable_lowfreq_rerank;
    std::optional<int> dim;
    std::optional<int> pq_dim;
    std::optional<uint32_t> rabitq_num_anchors;
    std::optional<uint32_t> rabitq_bits_per_dim;
    std::optional<bool> rabitq_enable_rerank;
    std::optional<bool> rabitq_enable_highfreq_rerank;
    std::optional<bool> rabitq_enable_lowfreq_rerank;
    std::optional<bool> rabitq_lowfreq_gpu_posting;
    std::optional<bool> rabitq_use_lut;
    std::optional<std::string> rabitq_query_rotation;
    std::optional<std::string> rabitq_centroid_distances;
    std::optional<bool> rabitq_use_int8_score;
    bool help = false;
};

template <typename T>
static void apply_override(const std::optional<T>& override_value, T& target)
{
    if (override_value.has_value()) {
        target = *override_value;
    }
}

static bool starts_with_option(const std::string& arg,
                               const std::string& hyphen_form,
                               const std::string& underscore_form)
{
    return arg.rfind(hyphen_form, 0) == 0 || arg.rfind(underscore_form, 0) == 0;
}

static std::string option_value(const std::string& arg,
                                const std::string& hyphen_form,
                                const std::string& underscore_form)
{
    if (arg.rfind(hyphen_form, 0) == 0) {
        return arg.substr(hyphen_form.size());
    }
    return arg.substr(underscore_form.size());
}

static std::vector<int> parse_int_list(const std::string& s);

static void apply_config_values(const AppConfigValues& config, AppOptions& options)
{
    apply_override(config.mode, options.mode);
    apply_override(config.backend, options.backend);
    apply_override(config.cpu_rerank_schedule, options.cpu_rerank_schedule);
    apply_override(config.graph_builder, options.graph_builder);
    apply_override(config.graph_path, options.graph_path);
    apply_override(config.graph_file_path, options.graph_file_path);
    apply_override(config.graph_storage_mode, options.graph_storage_mode);
    apply_override(config.rerank_vector_storage_mode, options.rerank_vector_storage_mode);
    apply_override(config.rerank_vector_artifact_path, options.rerank_vector_artifact_path);
    apply_override(config.skip_filtered_vector_payload, options.skip_filtered_vector_payload);
    apply_override(config.per_step_full_filter, options.per_step_full_filter);
    apply_override(config.highfreq_rerank_mode, options.highfreq_rerank_mode);
    apply_override(config.highfreq_rerank_algorithm, options.highfreq_rerank_algorithm);
    apply_override(config.highfreq_rerank_streaming, options.highfreq_rerank_streaming);
    apply_override(config.highfreq_rerank_merge_routes, options.highfreq_rerank_merge_routes);
    apply_override(config.rerank_distance_reuse, options.rerank_distance_reuse);
    apply_override(config.rabitq_incremental_rerank_shortlist, options.rabitq_incremental_rerank_shortlist);
    apply_override(config.lowfreq_rerank_merge_routes, options.lowfreq_rerank_merge_routes);
    apply_override(config.lowfreq_index_path, options.lowfreq_index_path);
    apply_override(config.base_vector_path, options.base_vector_path);
    apply_override(config.base_range_fbin_path, options.base_range_fbin_path);
    apply_override(config.range_posting_index_path, options.range_posting_index_path);
    apply_override(config.range_posting_mode, options.range_posting_mode);
    apply_override(config.range_posting_scan_threshold, options.range_posting_scan_threshold);
    apply_override(config.range_posting_shortlist, options.range_posting_shortlist);
    apply_override(config.range_posting_postfilter_shortlist_multiplier,
                   options.range_posting_postfilter_shortlist_multiplier);
    apply_override(config.range_posting_and_filter_mode,
                   options.range_posting_and_filter_mode);
    apply_override(config.range_label_and_policy,
                   options.range_label_and_policy);
    apply_override(config.range_posting_drop_wide_hybrid_or,
                   options.range_posting_drop_wide_hybrid_or);
    apply_override(config.range_graph_hybrid_or_enabled,
                   options.range_graph_hybrid_or_enabled);
    apply_override(config.range_graph_hybrid_and_enabled,
                   options.range_graph_hybrid_and_enabled);
    apply_override(config.range_graph_path, options.range_graph_path);
    apply_override(config.range_graph_storage_mode, options.range_graph_storage_mode);
    apply_override(config.range_graph_data_label_spmat_path,
                   options.range_graph_data_label_spmat_path);
    apply_override(config.range_graph_label_manifest_path,
                   options.range_graph_label_manifest_path);
    apply_override(config.range_graph_seed_sidecar_path,
                   options.range_graph_seed_sidecar_path);
    apply_override(config.range_graph_original_label_seed_sidecar_path,
                   options.range_graph_original_label_seed_sidecar_path);
    apply_override(config.range_graph_metadata_sidecar_path,
                   options.range_graph_metadata_sidecar_path);
    apply_override(config.range_graph_lowfreq_index_path,
                   options.range_graph_lowfreq_index_path);
    apply_override(config.range_graph_group_path, options.range_graph_group_path);
    apply_override(config.data_label_spmat_path, options.data_label_spmat_path);
    apply_override(config.query_vector_path, options.query_vector_path);
    apply_override(config.query_label_spmat_path, options.query_label_spmat_path);
    apply_override(config.query_manifest_jsonl_path, options.query_manifest_jsonl_path);
    apply_override(config.query_manifest_families, options.query_manifest_families);
    apply_override(config.query_manifest_subtypes, options.query_manifest_subtypes);
    apply_override(config.query_manifest_max_or_terms, options.query_manifest_max_or_terms);
    apply_override(config.query_manifest_max_fuzzy_terms, options.query_manifest_max_fuzzy_terms);
    apply_override(config.query_manifest_min_ngram_score, options.query_manifest_min_ngram_score);
    apply_override(config.query_manifest_ngram_size, options.query_manifest_ngram_size);
    apply_override(config.query_manifest_operator_index_path,
                   options.query_manifest_operator_index_path);
    apply_override(config.query_manifest_operator_fill_mode,
                   options.query_manifest_operator_fill_mode);
    apply_override(config.query_manifest_operator_parity_check,
                   options.query_manifest_operator_parity_check);
    apply_override(config.or_route_order_mode, options.or_route_order_mode);
    apply_override(config.query_prepare_threads, options.query_prepare_threads);
    apply_override(config.query_prepare_optimized, options.query_prepare_optimized);
    apply_override(config.query_manifest_fuzzy_cap_policy,
                   options.query_manifest_fuzzy_cap_policy);
    apply_override(config.or_route_sketch_sidecar_path, options.or_route_sketch_sidecar_path);
    apply_override(config.or_route_accuracy_cost_model_path,
                   options.or_route_accuracy_cost_model_path);
    apply_override(config.or_route_calibration_trace_path,
                   options.or_route_calibration_trace_path);
    apply_override(config.or_route_calibration_run_id,
                   options.or_route_calibration_run_id);
    apply_override(config.or_route_sketch_k, options.or_route_sketch_k);
    apply_override(config.or_route_sketch_hash_seed, options.or_route_sketch_hash_seed);
    apply_override(config.or_route_allow_runtime_sketch_fallback,
                   options.or_route_allow_runtime_sketch_fallback);
    apply_override(config.or_route_runtime_sketch_fallback_max_rows,
                   options.or_route_runtime_sketch_fallback_max_rows);
    apply_override(config.or_route_verify_sketch_payload_checksum,
                   options.or_route_verify_sketch_payload_checksum);
    apply_override(config.query_range_path, options.query_range_path);
    apply_override(config.range_label_manifest_path, options.range_label_manifest_path);
    apply_override(config.seed_sidecar_path, options.seed_sidecar_path);
    apply_override(config.original_label_seed_sidecar_path, options.original_label_seed_sidecar_path);
    apply_override(config.metadata_sidecar_path, options.metadata_sidecar_path);
    apply_override(config.seed_selection_mode, options.seed_selection_mode);
    apply_override(config.pq_compressed_path, options.pq_compressed_path);
    apply_override(config.pq_pivots_path, options.pq_pivots_path);
    apply_override(config.pq_buffer_mode, options.pq_buffer_mode);
    apply_override(config.pq_graph_transport, options.pq_graph_transport);
    apply_override(config.pq_query_worker_mode, options.pq_query_worker_mode);
    apply_override(config.pq_cpu_persistent_workers, options.pq_cpu_persistent_workers);
    apply_override(config.pq_request_mode, options.pq_request_mode);
    apply_override(config.pq_request_batch_queries, options.pq_request_batch_queries);
    apply_override(config.pq_request_consumed_memory,
                   options.pq_request_consumed_memory);
    apply_override(config.pq_finite_token_transport, options.pq_finite_token_transport);
    apply_override(config.pq_finite_dispatch_mode, options.pq_finite_dispatch_mode);
    apply_override(config.pq_finite_partial_batch_timeout_us,
                   options.pq_finite_partial_batch_timeout_us);
    apply_override(config.pq_lowfreq_scan_mode, options.pq_lowfreq_scan_mode);
    apply_override(config.pq_posting_streaming, options.pq_posting_streaming);
    apply_override(config.pq_posting_stream_batch_size, options.pq_posting_stream_batch_size);
    apply_override(config.posting_predicate_order, options.posting_predicate_order);
    apply_override(config.rabitq_predicate_mode, options.rabitq_predicate_mode);
    apply_override(config.rabitq_predicate_storage, options.rabitq_predicate_storage);
    apply_override(config.rabitq_predicate_index_path, options.rabitq_predicate_index_path);
    apply_override(config.rabitq_predicate_bias_alpha, options.rabitq_predicate_bias_alpha);
    if (config.rabitq_predicate_bias_alpha) options.predicate_alpha_explicit = true;
    apply_override(config.rabitq_predicate_diagnostics, options.rabitq_predicate_diagnostics);
    apply_override(config.rabitq_query_worker_mode, options.rabitq_query_worker_mode);
    apply_override(config.rabitq_finite_dispatch_mode, options.rabitq_finite_dispatch_mode);
    apply_override(config.rabitq_finite_partial_batch_timeout_us,
                   options.rabitq_finite_partial_batch_timeout_us);
    apply_override(config.rabitq_graph_transport, options.rabitq_graph_transport);
    apply_override(config.rabitq_dual_graph_cpu_gpu_overlap,
                   options.rabitq_dual_graph_cpu_gpu_overlap);
    apply_override(config.rabitq_posting_schedule, options.rabitq_posting_schedule);
    apply_override(config.rabitq_range_posting_segment_rows, options.rabitq_range_posting_segment_rows);
    apply_override(config.rabitq_posting_stream_batch_size, options.rabitq_posting_stream_batch_size);
    apply_override(config.rabitq_posting_streaming, options.rabitq_posting_streaming);
    apply_override(config.rabitq_graph_persistent_workers, options.rabitq_graph_persistent_workers);
    apply_override(config.rabitq_posting_persistent_workers, options.rabitq_posting_persistent_workers);
    if (config.rabitq_request_mode.has_value()) {
        options.rabitq_request_mode_explicit = true;
    }
    apply_override(config.rabitq_request_mode, options.rabitq_request_mode);
    apply_override(config.rabitq_request_batch_queries,
                   options.rabitq_request_batch_queries);
    apply_override(config.rabitq_request_consumed_memory,
                   options.rabitq_request_consumed_memory);
    apply_override(config.rabitq_gpu_graph_row_cache_mib,
                   options.rabitq_gpu_graph_row_cache_mib);
    apply_override(config.host_reply_wait_policy, options.host_reply_wait_policy);
    apply_override(config.routing_mode, options.routing_mode);
    apply_override(config.group_path, options.group_path);
    apply_override(config.rabitq_artifact_path, options.rabitq_artifact_path);
    apply_override(config.rabitq_cuvs_artifact_path, options.rabitq_cuvs_artifact_path);
    apply_override(config.output_csv_path, options.output_csv_path);
    apply_override(config.full_filter, options.full_filter);
    apply_override(config.full_gpu_fp16, options.full_gpu_fp16);
    apply_override(config.overlap_graph_path, options.overlap_graph_path);
    apply_override(config.overlap_manifest_path, options.overlap_manifest_path);
    apply_override(config.overlap_max_bridges, options.overlap_max_bridges);

    apply_override(config.lowfreq_full_filter_mode, options.lowfreq_full_filter_mode);
    apply_override(config.force_rebuild, options.force_rebuild);
    apply_override(config.run_analyze, options.run_analyze);
    if (config.row_limit.has_value()) {
        options.row_limit_explicit = true;
    }
    apply_override(config.row_limit, options.row_limit);
    apply_override(config.cagra_build_worker_count, options.cagra_build_worker_count);
    apply_override(config.stream_graph_worker_count, options.stream_graph_worker_count);
    apply_override(config.stream_graph_auto_workers, options.stream_graph_auto_workers);
    apply_override(config.stream_graph_memory_budget_bytes, options.stream_graph_memory_budget_bytes);
    apply_override(config.stream_graph_per_worker_memory_cap_bytes,
                   options.stream_graph_per_worker_memory_cap_bytes);
    apply_override(config.stream_graph_allow_concurrent_cagra,
                   options.stream_graph_allow_concurrent_cagra);
    apply_override(config.cagra_ace_build_dir, options.cagra_ace_build_dir);
    apply_override(config.cagra_ace_force_disk, options.cagra_ace_force_disk);
    apply_override(config.specificity_threshold, options.specificity_threshold);
    apply_override(config.prefetch_hops, options.prefetch_hops);
    apply_override(config.quick_filter_bonus, options.quick_filter_bonus);
    apply_override(config.bit_filter_bonus, options.bit_filter_bonus);
    apply_override(config.range_filter_bonus, options.range_filter_bonus);
    apply_override(config.and_dual_route_posting_ratio_threshold,
                   options.and_dual_route_posting_ratio_threshold);
    apply_override(config.warmup_runs, options.warmup_runs);
    apply_override(config.timed_runs, options.timed_runs);
    apply_override(config.process_cpu_start, options.process_cpu_start);
    apply_override(config.process_cpu_count, options.process_cpu_count);
    apply_override(config.host_service_threads, options.host_service_threads);
    apply_override(config.host_service_cpu_start, options.host_service_cpu_start);
    apply_override(config.host_service_cpu_count, options.host_service_cpu_count);
    apply_override(config.host_demand_batch_size, options.host_demand_batch_size);
    apply_override(config.host_demand_batch_timeout_us, options.host_demand_batch_timeout_us);
    apply_override(config.search_queue_size, options.search_queue_size);
    apply_override(config.search_finish_count, options.search_finish_count);
    apply_override(config.search_max_steps, options.search_max_steps);
    apply_override(config.search_early_stop_mode, options.search_early_stop_mode);
    if (config.search_early_stop_mode) options.predicate_early_stop_explicit = true;
    apply_override(config.benchmark_quick_filter_bonuses, options.benchmark_quick_filter_bonuses);
    apply_override(config.benchmark_queue_sizes, options.benchmark_queue_sizes);
    apply_override(config.benchmark_output, options.benchmark_output);
    apply_override(config.latency_queue_sizes, options.latency_queue_sizes);
    apply_override(config.latency_batch_sizes, options.latency_batch_sizes);
    apply_override(config.latency_output, options.latency_output);
    apply_override(config.latency_trace_output, options.latency_trace_output);
    apply_override(config.latency_query_start, options.latency_query_start);
    apply_override(config.latency_query_limit, options.latency_query_limit);
    apply_override(config.latency_concurrency_levels, options.latency_concurrency_levels);
    apply_override(config.latency_run_all_queries, options.latency_run_all_queries);
    apply_override(config.latency_time_planner, options.latency_time_planner);
    apply_override(config.latency_online_metadata, options.latency_online_metadata);
    apply_override(config.latency_staggered, options.latency_staggered);
    apply_override(config.latency_persistent_queue, options.latency_persistent_queue);
    apply_override(config.latency_queue_sweep_full_warmup,
                   options.latency_queue_sweep_full_warmup);
    apply_override(config.latency_lowfreq_mode, options.latency_lowfreq_mode);
    apply_override(config.degree, options.degree);
    apply_override(config.topk, options.topk);
    apply_override(config.highfreq_rerank_shortlist, options.highfreq_rerank_shortlist);
    apply_override(config.highfreq_rerank_batch_size, options.highfreq_rerank_batch_size);
    apply_override(config.highfreq_rerank_epsilon, options.highfreq_rerank_epsilon);
    apply_override(config.highfreq_rerank_stable_batches, options.highfreq_rerank_stable_batches);
    apply_override(config.highfreq_rerank_threads, options.highfreq_rerank_threads);
    apply_override(config.highfreq_rerank_stream_batch_size, options.highfreq_rerank_stream_batch_size);
    apply_override(config.lowfreq_rerank_shortlist, options.lowfreq_rerank_shortlist);
    apply_override(config.lowfreq_rerank_batch_size, options.lowfreq_rerank_batch_size);
    apply_override(config.lowfreq_rerank_epsilon, options.lowfreq_rerank_epsilon);
    apply_override(config.lowfreq_rerank_stable_batches, options.lowfreq_rerank_stable_batches);
    apply_override(config.enable_lowfreq_rerank, options.enable_lowfreq_rerank);
    apply_override(config.dim, options.dim);
    apply_override(config.pq_dim, options.pq_dim);
    apply_override(config.rabitq_num_anchors, options.rabitq_num_anchors);
    apply_override(config.rabitq_bits_per_dim, options.rabitq_bits_per_dim);
    if (config.rabitq_enable_rerank.has_value()) {
        options.rabitq_enable_highfreq_rerank = *config.rabitq_enable_rerank;
        options.rabitq_enable_lowfreq_rerank = *config.rabitq_enable_rerank;
        options.enable_lowfreq_rerank = *config.rabitq_enable_rerank;
    }
    apply_override(config.rabitq_enable_highfreq_rerank, options.rabitq_enable_highfreq_rerank);
    apply_override(config.rabitq_enable_lowfreq_rerank, options.rabitq_enable_lowfreq_rerank);
    if (config.rabitq_enable_lowfreq_rerank.has_value()) {
        options.enable_lowfreq_rerank = *config.rabitq_enable_lowfreq_rerank;
    }
    apply_override(config.rabitq_lowfreq_gpu_posting, options.rabitq_lowfreq_gpu_posting);
    apply_override(config.rabitq_use_lut, options.rabitq_use_lut);
    if (config.rabitq_use_int8_score.has_value()) {
        options.rabitq_use_int8_score_explicit = true;
    }
    apply_override(config.rabitq_query_rotation, options.rabitq_query_rotation);
    apply_override(config.rabitq_centroid_distances, options.rabitq_centroid_distances);
    apply_override(config.rabitq_use_int8_score, options.rabitq_use_int8_score);
}

static void apply_cli_overrides(const AppOptionOverrides& overrides, AppOptions& options)
{
    apply_override(overrides.mode, options.mode);
    apply_override(overrides.backend, options.backend);
    apply_override(overrides.cpu_rerank_schedule, options.cpu_rerank_schedule);
    apply_override(overrides.graph_builder, options.graph_builder);
    apply_override(overrides.graph_path, options.graph_path);
    apply_override(overrides.graph_file_path, options.graph_file_path);
    apply_override(overrides.graph_storage_mode, options.graph_storage_mode);
    apply_override(overrides.rerank_vector_storage_mode, options.rerank_vector_storage_mode);
    apply_override(overrides.rerank_vector_artifact_path, options.rerank_vector_artifact_path);
    apply_override(overrides.skip_filtered_vector_payload, options.skip_filtered_vector_payload);
    apply_override(overrides.per_step_full_filter, options.per_step_full_filter);
    apply_override(overrides.highfreq_rerank_mode, options.highfreq_rerank_mode);
    apply_override(overrides.highfreq_rerank_algorithm, options.highfreq_rerank_algorithm);
    apply_override(overrides.highfreq_rerank_streaming, options.highfreq_rerank_streaming);
    apply_override(overrides.highfreq_rerank_merge_routes, options.highfreq_rerank_merge_routes);
    apply_override(overrides.rerank_distance_reuse, options.rerank_distance_reuse);
    apply_override(overrides.rabitq_incremental_rerank_shortlist, options.rabitq_incremental_rerank_shortlist);
    apply_override(overrides.rabitq_posting_schedule, options.rabitq_posting_schedule);
    apply_override(overrides.rabitq_range_posting_segment_rows, options.rabitq_range_posting_segment_rows);
    apply_override(overrides.rabitq_posting_stream_batch_size, options.rabitq_posting_stream_batch_size);
    apply_override(overrides.rabitq_posting_streaming, options.rabitq_posting_streaming);
    apply_override(overrides.rabitq_graph_persistent_workers, options.rabitq_graph_persistent_workers);
    apply_override(overrides.rabitq_posting_persistent_workers, options.rabitq_posting_persistent_workers);
    apply_override(overrides.lowfreq_rerank_merge_routes, options.lowfreq_rerank_merge_routes);
    apply_override(overrides.lowfreq_index_path, options.lowfreq_index_path);
    apply_override(overrides.base_vector_path, options.base_vector_path);
    apply_override(overrides.base_range_fbin_path, options.base_range_fbin_path);
    apply_override(overrides.range_posting_index_path, options.range_posting_index_path);
    apply_override(overrides.range_posting_mode, options.range_posting_mode);
    apply_override(overrides.range_posting_scan_threshold, options.range_posting_scan_threshold);
    apply_override(overrides.range_posting_shortlist, options.range_posting_shortlist);
    apply_override(overrides.range_posting_postfilter_shortlist_multiplier,
                   options.range_posting_postfilter_shortlist_multiplier);
    apply_override(overrides.range_posting_and_filter_mode,
                   options.range_posting_and_filter_mode);
    apply_override(overrides.range_label_and_policy,
                   options.range_label_and_policy);
    apply_override(overrides.range_posting_drop_wide_hybrid_or,
                   options.range_posting_drop_wide_hybrid_or);
    apply_override(overrides.range_graph_hybrid_or_enabled,
                   options.range_graph_hybrid_or_enabled);
    apply_override(overrides.range_graph_hybrid_and_enabled,
                   options.range_graph_hybrid_and_enabled);
    apply_override(overrides.range_graph_path, options.range_graph_path);
    apply_override(overrides.range_graph_storage_mode, options.range_graph_storage_mode);
    apply_override(overrides.range_graph_data_label_spmat_path,
                   options.range_graph_data_label_spmat_path);
    apply_override(overrides.range_graph_label_manifest_path,
                   options.range_graph_label_manifest_path);
    apply_override(overrides.range_graph_seed_sidecar_path,
                   options.range_graph_seed_sidecar_path);
    apply_override(overrides.range_graph_original_label_seed_sidecar_path,
                   options.range_graph_original_label_seed_sidecar_path);
    apply_override(overrides.range_graph_metadata_sidecar_path,
                   options.range_graph_metadata_sidecar_path);
    apply_override(overrides.range_graph_lowfreq_index_path,
                   options.range_graph_lowfreq_index_path);
    apply_override(overrides.range_graph_group_path, options.range_graph_group_path);
    apply_override(overrides.data_label_spmat_path, options.data_label_spmat_path);
    apply_override(overrides.query_vector_path, options.query_vector_path);
    apply_override(overrides.query_label_spmat_path, options.query_label_spmat_path);
    apply_override(overrides.query_manifest_jsonl_path, options.query_manifest_jsonl_path);
    apply_override(overrides.query_manifest_families, options.query_manifest_families);
    apply_override(overrides.query_manifest_subtypes, options.query_manifest_subtypes);
    apply_override(overrides.query_manifest_max_or_terms, options.query_manifest_max_or_terms);
    apply_override(overrides.query_manifest_max_fuzzy_terms, options.query_manifest_max_fuzzy_terms);
    apply_override(overrides.query_prepare_threads, options.query_prepare_threads);
    apply_override(overrides.query_prepare_optimized, options.query_prepare_optimized);
    apply_override(overrides.query_manifest_fuzzy_cap_policy,
                   options.query_manifest_fuzzy_cap_policy);
    apply_override(overrides.query_manifest_min_ngram_score, options.query_manifest_min_ngram_score);
    apply_override(overrides.query_manifest_ngram_size, options.query_manifest_ngram_size);
    apply_override(overrides.query_range_path, options.query_range_path);
    apply_override(overrides.range_label_manifest_path, options.range_label_manifest_path);
    apply_override(overrides.seed_sidecar_path, options.seed_sidecar_path);
    apply_override(overrides.original_label_seed_sidecar_path, options.original_label_seed_sidecar_path);
    apply_override(overrides.metadata_sidecar_path, options.metadata_sidecar_path);
    apply_override(overrides.seed_selection_mode, options.seed_selection_mode);
    apply_override(overrides.pq_compressed_path, options.pq_compressed_path);
    apply_override(overrides.pq_pivots_path, options.pq_pivots_path);
    apply_override(overrides.pq_buffer_mode, options.pq_buffer_mode);
    apply_override(overrides.pq_graph_transport, options.pq_graph_transport);
    apply_override(overrides.pq_query_worker_mode, options.pq_query_worker_mode);
    apply_override(overrides.pq_cpu_persistent_workers, options.pq_cpu_persistent_workers);
    apply_override(overrides.pq_request_mode, options.pq_request_mode);
    apply_override(overrides.pq_request_batch_queries, options.pq_request_batch_queries);
    apply_override(overrides.pq_request_consumed_memory,
                   options.pq_request_consumed_memory);
    apply_override(overrides.pq_finite_token_transport, options.pq_finite_token_transport);
    apply_override(overrides.pq_finite_dispatch_mode, options.pq_finite_dispatch_mode);
    apply_override(overrides.pq_finite_partial_batch_timeout_us,
                   options.pq_finite_partial_batch_timeout_us);
    apply_override(overrides.pq_lowfreq_scan_mode, options.pq_lowfreq_scan_mode);
    apply_override(overrides.pq_posting_streaming, options.pq_posting_streaming);
    apply_override(overrides.pq_posting_stream_batch_size, options.pq_posting_stream_batch_size);
    apply_override(overrides.posting_predicate_order, options.posting_predicate_order);
    apply_override(overrides.rabitq_predicate_mode, options.rabitq_predicate_mode);
    apply_override(overrides.rabitq_predicate_storage, options.rabitq_predicate_storage);
    apply_override(overrides.rabitq_predicate_index_path, options.rabitq_predicate_index_path);
    apply_override(overrides.rabitq_predicate_bias_alpha, options.rabitq_predicate_bias_alpha);
    if (overrides.rabitq_predicate_bias_alpha) options.predicate_alpha_explicit = true;
    apply_override(overrides.rabitq_predicate_diagnostics, options.rabitq_predicate_diagnostics);
    apply_override(overrides.rabitq_query_worker_mode, options.rabitq_query_worker_mode);
    apply_override(overrides.rabitq_finite_dispatch_mode,
                   options.rabitq_finite_dispatch_mode);
    apply_override(overrides.rabitq_finite_partial_batch_timeout_us,
                   options.rabitq_finite_partial_batch_timeout_us);
    apply_override(overrides.rabitq_graph_transport, options.rabitq_graph_transport);
    if (overrides.rabitq_request_mode.has_value()) {
        options.rabitq_request_mode_explicit = true;
    }
    apply_override(overrides.rabitq_request_mode, options.rabitq_request_mode);
    apply_override(overrides.rabitq_request_batch_queries,
                   options.rabitq_request_batch_queries);
    apply_override(overrides.rabitq_request_consumed_memory,
                   options.rabitq_request_consumed_memory);
    apply_override(overrides.rabitq_gpu_graph_row_cache_mib,
                   options.rabitq_gpu_graph_row_cache_mib);
    apply_override(overrides.host_reply_wait_policy, options.host_reply_wait_policy);
    apply_override(overrides.routing_mode, options.routing_mode);
    apply_override(overrides.group_path, options.group_path);
    apply_override(overrides.rabitq_artifact_path, options.rabitq_artifact_path);
    apply_override(overrides.rabitq_cuvs_artifact_path, options.rabitq_cuvs_artifact_path);
    apply_override(overrides.output_csv_path, options.output_csv_path);
    apply_override(overrides.config_path, options.config_path);
    apply_override(overrides.full_filter, options.full_filter);
    apply_override(overrides.full_gpu_fp16, options.full_gpu_fp16);
    apply_override(overrides.overlap_graph_path, options.overlap_graph_path);
    apply_override(overrides.overlap_manifest_path, options.overlap_manifest_path);
    apply_override(overrides.overlap_max_bridges, options.overlap_max_bridges);

    apply_override(overrides.lowfreq_full_filter_mode, options.lowfreq_full_filter_mode);
    apply_override(overrides.force_rebuild, options.force_rebuild);
    apply_override(overrides.run_analyze, options.run_analyze);
    if (overrides.row_limit.has_value()) {
        options.row_limit_explicit = true;
    }
    apply_override(overrides.row_limit, options.row_limit);
    apply_override(overrides.cagra_build_worker_count, options.cagra_build_worker_count);
    apply_override(overrides.stream_graph_worker_count, options.stream_graph_worker_count);
    apply_override(overrides.stream_graph_auto_workers, options.stream_graph_auto_workers);
    apply_override(overrides.stream_graph_memory_budget_bytes, options.stream_graph_memory_budget_bytes);
    apply_override(overrides.stream_graph_per_worker_memory_cap_bytes,
                   options.stream_graph_per_worker_memory_cap_bytes);
    apply_override(overrides.stream_graph_allow_concurrent_cagra,
                   options.stream_graph_allow_concurrent_cagra);
    apply_override(overrides.specificity_threshold, options.specificity_threshold);
    apply_override(overrides.prefetch_hops, options.prefetch_hops);
    apply_override(overrides.quick_filter_bonus, options.quick_filter_bonus);
    apply_override(overrides.bit_filter_bonus, options.bit_filter_bonus);
    apply_override(overrides.range_filter_bonus, options.range_filter_bonus);
    apply_override(overrides.and_dual_route_posting_ratio_threshold,
                   options.and_dual_route_posting_ratio_threshold);
    apply_override(overrides.warmup_runs, options.warmup_runs);
    apply_override(overrides.timed_runs, options.timed_runs);
    apply_override(overrides.process_cpu_start, options.process_cpu_start);
    apply_override(overrides.process_cpu_count, options.process_cpu_count);
    apply_override(overrides.host_service_threads, options.host_service_threads);
    apply_override(overrides.host_service_cpu_start, options.host_service_cpu_start);
    apply_override(overrides.host_service_cpu_count, options.host_service_cpu_count);
    apply_override(overrides.host_demand_batch_size, options.host_demand_batch_size);
    apply_override(overrides.host_demand_batch_timeout_us, options.host_demand_batch_timeout_us);
    apply_override(overrides.search_queue_size, options.search_queue_size);
    apply_override(overrides.search_finish_count, options.search_finish_count);
    apply_override(overrides.search_max_steps, options.search_max_steps);
    apply_override(overrides.search_early_stop_mode, options.search_early_stop_mode);
    if (overrides.search_early_stop_mode) options.predicate_early_stop_explicit = true;
    apply_override(overrides.benchmark_quick_filter_bonuses, options.benchmark_quick_filter_bonuses);
    apply_override(overrides.benchmark_queue_sizes, options.benchmark_queue_sizes);
    apply_override(overrides.benchmark_output, options.benchmark_output);
    apply_override(overrides.latency_queue_sizes, options.latency_queue_sizes);
    apply_override(overrides.latency_batch_sizes, options.latency_batch_sizes);
    apply_override(overrides.latency_output, options.latency_output);
    apply_override(overrides.latency_trace_output, options.latency_trace_output);
    apply_override(overrides.latency_query_start, options.latency_query_start);
    apply_override(overrides.latency_query_limit, options.latency_query_limit);
    apply_override(overrides.latency_concurrency_levels, options.latency_concurrency_levels);
    apply_override(overrides.latency_run_all_queries, options.latency_run_all_queries);
    apply_override(overrides.latency_time_planner, options.latency_time_planner);
    apply_override(overrides.latency_online_metadata, options.latency_online_metadata);
    apply_override(overrides.latency_staggered, options.latency_staggered);
    apply_override(overrides.latency_persistent_queue, options.latency_persistent_queue);
    apply_override(overrides.latency_queue_sweep_full_warmup,
                   options.latency_queue_sweep_full_warmup);
    apply_override(overrides.latency_lowfreq_mode, options.latency_lowfreq_mode);
    apply_override(overrides.degree, options.degree);
    apply_override(overrides.topk, options.topk);
    apply_override(overrides.highfreq_rerank_shortlist, options.highfreq_rerank_shortlist);
    apply_override(overrides.highfreq_rerank_batch_size, options.highfreq_rerank_batch_size);
    apply_override(overrides.highfreq_rerank_epsilon, options.highfreq_rerank_epsilon);
    apply_override(overrides.highfreq_rerank_stable_batches, options.highfreq_rerank_stable_batches);
    apply_override(overrides.highfreq_rerank_threads, options.highfreq_rerank_threads);
    apply_override(overrides.highfreq_rerank_stream_batch_size, options.highfreq_rerank_stream_batch_size);
    apply_override(overrides.lowfreq_rerank_shortlist, options.lowfreq_rerank_shortlist);
    apply_override(overrides.lowfreq_rerank_batch_size, options.lowfreq_rerank_batch_size);
    apply_override(overrides.lowfreq_rerank_epsilon, options.lowfreq_rerank_epsilon);
    apply_override(overrides.lowfreq_rerank_stable_batches, options.lowfreq_rerank_stable_batches);
    apply_override(overrides.enable_lowfreq_rerank, options.enable_lowfreq_rerank);
    apply_override(overrides.dim, options.dim);
    apply_override(overrides.pq_dim, options.pq_dim);
    apply_override(overrides.rabitq_num_anchors, options.rabitq_num_anchors);
    apply_override(overrides.rabitq_bits_per_dim, options.rabitq_bits_per_dim);
    if (overrides.rabitq_enable_rerank.has_value()) {
        options.rabitq_enable_highfreq_rerank = *overrides.rabitq_enable_rerank;
        options.rabitq_enable_lowfreq_rerank = *overrides.rabitq_enable_rerank;
        options.enable_lowfreq_rerank = *overrides.rabitq_enable_rerank;
    }
    apply_override(overrides.rabitq_enable_highfreq_rerank, options.rabitq_enable_highfreq_rerank);
    apply_override(overrides.rabitq_enable_lowfreq_rerank, options.rabitq_enable_lowfreq_rerank);
    if (overrides.rabitq_enable_lowfreq_rerank.has_value()) {
        options.enable_lowfreq_rerank = *overrides.rabitq_enable_lowfreq_rerank;
    }
    apply_override(overrides.rabitq_lowfreq_gpu_posting, options.rabitq_lowfreq_gpu_posting);
    apply_override(overrides.rabitq_use_lut, options.rabitq_use_lut);
    if (overrides.rabitq_use_int8_score.has_value()) {
        options.rabitq_use_int8_score_explicit = true;
    }
    apply_override(overrides.rabitq_query_rotation, options.rabitq_query_rotation);
    apply_override(overrides.rabitq_centroid_distances, options.rabitq_centroid_distances);
    apply_override(overrides.rabitq_use_int8_score, options.rabitq_use_int8_score);
    options.help = overrides.help;
}

static std::optional<std::string> default_config_path_if_exists()
{
    const std::filesystem::path config_path = std::filesystem::current_path() / "config.json";
    if (std::filesystem::exists(config_path)) {
        return config_path.string();
    }
    return std::nullopt;
}

static void maybe_bind_process_cpus(const AppOptions& options)
{
    if (options.process_cpu_start < 0 || options.process_cpu_count <= 0) {
        return;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int cpu = options.process_cpu_start; cpu < options.process_cpu_start + options.process_cpu_count; ++cpu) {
        CPU_SET(cpu, &cpuset);
    }
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

static bool uses_lowfreq_rerank_controls(const AppOptions& options)
{
    if (options.mode != "search" && options.mode != "benchmark" && options.mode != "latency") {
        return false;
    }
    if (options.backend == "pq") {
        return options.enable_lowfreq_rerank;
    }
    return options.backend == "rabitq" && options.enable_lowfreq_rerank;
}

static bool uses_lowfreq_full_filter_mode(const AppOptions& options)
{
    if (options.mode != "search" && options.mode != "benchmark") {
        return false;
    }
    return options.backend == "pq" || options.backend == "rabitq";
}

static bool uses_highfreq_rerank_controls(const AppOptions& options)
{
    if (options.mode != "search" && options.mode != "benchmark" && options.mode != "latency") {
        return false;
    }
    return options.backend == "pq" || options.backend == "rabitq";
}

static void apply_legacy_rabitq_highfreq_rerank_mode(AppOptions& options)
{
    if (options.backend != "rabitq") {
        return;
    }
    if (!options.rabitq_enable_highfreq_rerank &&
        (defaults::highfreq_rerank_mode_is_gpu(options.highfreq_rerank_mode) ||
         defaults::highfreq_rerank_mode_is_gpu_post(options.highfreq_rerank_mode))) {
        options.highfreq_rerank_mode = std::string(defaults::highfreq_rerank_mode_off);
    }
}

static void resolve_cpu_rerank_schedule(AppOptions& options)
{
    const auto mode = cpu_rerank_schedule::parse(options.cpu_rerank_schedule);
    const bool pq = options.backend == "pq";
    cpu_rerank_schedule::apply(mode, options.backend, options.mode,
        pq ? options.pq_query_worker_mode : options.rabitq_query_worker_mode,
        pq ? options.pq_graph_transport : options.rabitq_graph_transport,
        options.highfreq_rerank_mode, options.highfreq_rerank_algorithm,
        options.highfreq_rerank_streaming,
        pq ? options.pq_posting_streaming : options.rabitq_posting_streaming);
    if (mode != cpu_rerank_schedule::Mode::Native && options.latency_persistent_queue)
        throw std::invalid_argument("cpu_rerank_schedule does not support persistent queues");
    if (mode != cpu_rerank_schedule::Mode::Native && options.backend == "rabitq" &&
        std::getenv("GO_FANN_LOWFREQ_PROFILE") != nullptr)
        throw std::invalid_argument(
            "cpu_rerank_schedule requires GO_FANN_LOWFREQ_PROFILE to be unset");
    if ((mode == cpu_rerank_schedule::Mode::NoOverlap ||
         mode == cpu_rerank_schedule::Mode::BatchOverlap) &&
        options.rabitq_incremental_rerank_shortlist > 0)
        throw std::invalid_argument(
            "cpu_rerank_schedule without streaming does not support incremental rerank");
    std::cout << "[cpu-rerank-schedule] mode=" << options.cpu_rerank_schedule
              << " graph_streaming=" << options.highfreq_rerank_streaming
              << " posting_streaming=" << (pq ? options.pq_posting_streaming : options.rabitq_posting_streaming)
              << " deferred_finalize=" << cpu_rerank_schedule::no_overlap(mode) << '\n';
}

static void apply_rabitq_optimized_defaults(AppOptions& options)
{
    if (options.backend != "rabitq") {
        return;
    }

    const bool native_throughput_mode =
        options.mode == "search" || options.mode == "benchmark";
    const bool coupled_memory_path = native_throughput_mode &&
        options.graph_storage_mode == "memory" &&
        options.rabitq_query_worker_mode == "coupled";

    if (!options.rabitq_request_mode_explicit) {
        options.rabitq_request_mode = coupled_memory_path &&
                options.rabitq_graph_transport != "device_resident" &&
                options.host_demand_batch_size == 1
            ? "batch_ring"
            : "scan";
    }
    if (!options.rabitq_use_int8_score_explicit) {
        options.rabitq_use_int8_score = coupled_memory_path &&
            options.rabitq_bits_per_dim <= 2 && !options.rabitq_use_lut;
    }
}

static void resolve_predicate_defaults(AppOptions& options)
{
    predicate_defaults::Context context;
    context.backend = options.backend;
    context.operation = options.mode;
    const bool pq = options.backend == "pq";
    context.worker = pq ? options.pq_query_worker_mode : options.rabitq_query_worker_mode;
    context.transport = pq ? options.pq_graph_transport : options.rabitq_graph_transport;
    context.graph_storage = options.graph_storage_mode;
    context.early_stop = options.search_early_stop_mode;
    context.early_stop_explicit = options.predicate_early_stop_explicit;
    context.cpu_exact_rerank = defaults::highfreq_rerank_mode_is_cpu(options.highfreq_rerank_mode) &&
        options.highfreq_rerank_algorithm == "exact" &&
        (pq || options.rabitq_enable_highfreq_rerank);
    context.gpu_inline_exact_rerank = defaults::highfreq_rerank_mode_is_gpu(options.highfreq_rerank_mode) &&
        options.highfreq_rerank_algorithm == "exact" &&
        (pq || options.rabitq_enable_highfreq_rerank);
    context.gpu_post_exact_rerank = !pq && options.backend == "rabitq" &&
        defaults::highfreq_rerank_mode_is_gpu_post(options.highfreq_rerank_mode) &&
        options.highfreq_rerank_algorithm == "exact" && options.rabitq_enable_highfreq_rerank;
    context.full_filter = options.full_filter;
    context.per_step_full_filter = options.per_step_full_filter;
    context.graph_cache = !pq && options.rabitq_gpu_graph_row_cache_mib != 0;
    context.persistent_queue = options.latency_persistent_queue;
    context.mailbox_profile = !pq && std::getenv("GO_FANN_RABITQ_MAILBOX_RTT_PROFILE") != nullptr;
    const auto resolved = predicate_defaults::resolve(
        options.rabitq_predicate_mode, options.rabitq_predicate_storage,
        options.predicate_alpha_explicit ? std::optional<double>(options.rabitq_predicate_bias_alpha) : std::nullopt,
        options.rabitq_predicate_diagnostics, context);
    std::cout << "[predicate-policy] requested=" << options.rabitq_predicate_mode
              << " effective=" << resolved.mode << " storage=" << options.rabitq_predicate_storage
              << " alpha=" << resolved.alpha << " early_stop=" << resolved.early_stop << " reason=" << resolved.reason << '\n';
    // Auto fallback must not enable unsupported observers; explicit CPU
    // diagnostics retain the legacy RabitQ control behavior.
    if (options.rabitq_predicate_mode == "auto" && resolved.mode == "cpu")
        options.rabitq_predicate_diagnostics = false;
    options.search_early_stop_mode = resolved.early_stop;
    options.rabitq_predicate_mode = resolved.mode;
    options.rabitq_predicate_bias_alpha = resolved.alpha;
}

// Keep the application default within the same bounds as native posting
// validation. The backend receives only concrete host/resident choices.
static const char* posting_policy_unsupported_reason(const AppOptions& options)
{
    const bool pq = options.backend == "pq";
    if (!pq && options.backend != "rabitq") return "backend";
    if (options.mode != "search" && options.mode != "benchmark") return "operation";
    if (options.graph_storage_mode != "memory")
        return "graph_storage";
    if ((pq ? options.pq_query_worker_mode : options.rabitq_query_worker_mode) != "coupled")
        return "worker";
    if ((pq ? options.pq_graph_transport : options.rabitq_graph_transport) != "mailbox")
        return "transport";
    if (options.latency_persistent_queue || (!pq && options.rabitq_posting_persistent_workers))
        return "persistent_posting";
    if (pq && defaults::lowfreq_full_filter_mode_is_pre(options.lowfreq_full_filter_mode))
        return "host_prefilter";
    if (options.rabitq_predicate_storage != "csr" ||
        (options.rabitq_predicate_mode != "gpu_exact" && options.rabitq_predicate_mode != "gpu_exact_bias"))
        return "gpu_csr_predicate";
    return nullptr;
}

static void resolve_posting_defaults(AppOptions& options)
{
    const std::string requested = options.pq_lowfreq_scan_mode;
    if (requested != "auto" && requested != "host" && requested != "resident")
        throw std::runtime_error("pq-lowfreq-scan-mode must be one of: auto, host, resident");
    const char* reason = "explicit";
    if (requested == "auto") {
        // This PQ-specific option never changes RabitQ's existing scan policy.
        const char* unsupported = options.backend == "pq"
            ? posting_policy_unsupported_reason(options) : "not_pq";
        options.pq_lowfreq_scan_mode = unsupported ? "host" : "resident";
        reason = unsupported ? unsupported : "supported_native_gpu_csr";
    }
    std::cout << "[pq-posting-policy] requested=" << requested
              << " effective=" << options.pq_lowfreq_scan_mode
              << " reason=" << reason << '\n';
}

static void validate_app_options(const AppOptions& options)
{
    if (expansion_trace::enabled() &&
        ((options.mode != "search" && options.mode != "benchmark") ||
         (options.backend != "pq" && options.backend != "rabitq")))
        throw std::runtime_error("expansion trace supports native PQ/RabitQ search or single-point benchmark");
    if (options.rabitq_incremental_rerank_shortlist < 0 ||
        (options.rabitq_incremental_rerank_shortlist > 0 &&
         (options.backend != "rabitq" || (options.mode != "search" && options.mode != "benchmark") ||
          options.rabitq_incremental_rerank_shortlist < options.topk)))
        throw std::invalid_argument("incremental rerank requires RabitQ search/benchmark and L >= topk (0 disables)");
    if (options.posting_predicate_order != "filter_first" && options.posting_predicate_order != "score_first")
        throw std::runtime_error("posting-predicate-order must be one of: filter_first, score_first");
    if (options.pq_posting_stream_batch_size == 0 || options.pq_posting_stream_batch_size > 1048576)
        throw std::runtime_error("pq-posting-stream-batch-size must be in [1,1048576]");
    if (options.pq_posting_streaming &&
        (options.backend != "pq" || posting_policy_unsupported_reason(options) != nullptr ||
         options.pq_lowfreq_scan_mode != "resident" || !options.enable_lowfreq_rerank ||
         options.posting_predicate_order != "filter_first"))
        throw std::runtime_error("pq-posting-streaming requires native resident filter-first GPU CSR posting with CPU lowfreq rerank");
    if (options.pq_lowfreq_scan_mode == "resident" && options.backend != "pq")
        throw std::runtime_error("pq-lowfreq-scan-mode=resident requires backend=pq");
    if (options.pq_lowfreq_scan_mode == "resident" || options.posting_predicate_order == "score_first") {
        if (const char* reason = posting_policy_unsupported_reason(options))
            throw std::runtime_error(std::string("resident/score-first posting requires native coupled in-memory mailbox search/benchmark with GPU CSR predicate and nonpersistent posting; unsupported: ") + reason);
    }
    std::cout << "[rerank-distance-reuse] enabled=" << options.rerank_distance_reuse << '\n';
    std::cout << "[posting-policy] pq_lowfreq_scan_mode=" << options.pq_lowfreq_scan_mode
              << " predicate_order=" << options.posting_predicate_order
              << " posting_streaming=" << (options.backend == "pq" ? options.pq_posting_streaming : options.rabitq_posting_streaming) << '\n';

    if (!defaults::is_valid_rerank_vector_storage_mode(
            options.rerank_vector_storage_mode)) {
        throw std::runtime_error(
            "rerank-vector-storage-mode must be one of: default, mmap_fp16, memory_int8");
    }
    const bool custom_rerank_vectors =
        !defaults::rerank_vector_storage_is_default(options.rerank_vector_storage_mode);
    if (custom_rerank_vectors) {
        if (options.mode != "search" && options.mode != "benchmark" &&
            options.mode != "latency") {
            throw std::runtime_error(
                "custom rerank vector storage is supported only by search, benchmark, and latency");
        }
        if (options.backend != "rabitq") {
            throw std::runtime_error(
                "custom rerank vector storage currently requires --backend=rabitq");
        }
        const bool fp16_gpu_post =
            defaults::rerank_vector_storage_is_mmap_fp16(options.rerank_vector_storage_mode) &&
            defaults::highfreq_rerank_mode_is_gpu_post(options.highfreq_rerank_mode) &&
            (options.mode == "search" || options.mode == "benchmark");
        if (!defaults::highfreq_rerank_mode_is_cpu(options.highfreq_rerank_mode) &&
            !fp16_gpu_post) {
            throw std::runtime_error(
                "custom rerank vector storage requires CPU rerank or mmap_fp16 GPU-post search/benchmark");
        }
        if (defaults::rerank_vector_storage_is_memory_int8(
                options.rerank_vector_storage_mode) &&
            options.rerank_vector_artifact_path.empty()) {
            throw std::runtime_error(
                "memory_int8 requires --rerank-vector-artifact-path");
        }
    }
    if (options.mode == "build-rerank-int8" &&
        options.rerank_vector_artifact_path.empty()) {
        throw std::runtime_error(
            "build-rerank-int8 requires --rerank-vector-artifact-path");
    }
    if (options.pq_cpu_persistent_workers &&
        (options.backend != "pq" || (options.mode != "search" && options.mode != "benchmark") ||
         options.pq_query_worker_mode != "coupled")) {
        throw std::runtime_error("pq-cpu-persistent-workers requires coupled PQ search/benchmark");
    }
    if (!defaults::is_valid_pq_query_worker_mode(options.pq_query_worker_mode)) {
        throw std::runtime_error("pq-query-worker-mode must be one of: coupled, finite");
    }
    if (options.pq_graph_transport != "mailbox" &&
        options.pq_graph_transport != "device_resident") {
        throw std::runtime_error(
            "pq-graph-transport must be one of: mailbox, device_resident");
    }
    if (options.pq_graph_transport == "device_resident" &&
        (options.backend != "pq" ||
         (options.mode != "search" && options.mode != "benchmark") ||
         options.graph_storage_mode != "memory" ||
         options.pq_query_worker_mode != "coupled" ||
         options.pq_request_mode != "scan" ||

         !defaults::highfreq_rerank_mode_is_gpu(options.highfreq_rerank_mode))) {
        throw std::runtime_error(
            "pq-graph-transport=device_resident requires PQ search|benchmark, "
            "memory graph, coupled workers, request_mode=scan, resident base vectors, "
            "and highfreq_rerank_mode=gpu");
    }
    if (!defaults::is_valid_pq_finite_token_transport(
            options.pq_finite_token_transport)) {
        throw std::runtime_error(
            "pq-finite-token-transport must be one of: device_copy, mapped");
    }
    if (!defaults::is_valid_pq_finite_dispatch_mode(options.pq_finite_dispatch_mode)) {
        throw std::runtime_error(
            "pq-finite-dispatch-mode must be one of: batch, persistent, ready_ring");
    }
    if (!defaults::is_valid_pq_finite_partial_batch_timeout_us(
            options.pq_finite_partial_batch_timeout_us)) {
        throw std::runtime_error(
            "pq-finite-partial-batch-timeout-us must be in [1, 1000000]");
    }
    if (defaults::pq_query_worker_mode_is_finite(options.pq_query_worker_mode)) {
        if (options.backend != "pq" ||
            (options.mode != "search" && options.mode != "benchmark")) {
            throw std::runtime_error(
                "pq-query-worker-mode=finite currently requires native --mode=search|benchmark --backend=pq");
        }
        if (options.pq_buffer_mode != "double_slot_serial") {
            throw std::runtime_error(
                "pq-query-worker-mode=finite currently requires --pq-buffer-mode=double_slot_serial");
        }
    }
    if (defaults::pq_finite_dispatch_mode_requires_completion_path(
            options.pq_finite_dispatch_mode)) {
        const bool memory_graph = options.graph_storage_mode == "memory";
        if (!defaults::pq_query_worker_mode_is_finite(options.pq_query_worker_mode) ||
            !memory_graph ||
            options.pq_buffer_mode != "double_slot_serial") {
            throw std::runtime_error(
                "pq-finite-dispatch-mode=" + options.pq_finite_dispatch_mode +
                " requires finite memory graph, "
                "with double_slot_serial");
        }
    }
    if (options.pq_request_mode != "scan" &&
        options.pq_request_mode != "batch_ring") {
        throw std::runtime_error(
            "pq-request-mode must be one of: scan, batch_ring");
    }
    if (options.pq_request_batch_queries == 0 ||
        options.pq_request_batch_queries > 4096) {
        throw std::runtime_error(
            "pq-request-batch-queries must be in [1, 4096]");
    }
    if (options.pq_request_consumed_memory != "host" &&
        options.pq_request_consumed_memory != "device") {
        throw std::runtime_error(
            "pq-request-consumed-memory must be one of: host, device");
    }
    if (options.pq_request_mode == "batch_ring") {
        if (options.backend != "pq" ||
            (options.mode != "search" && options.mode != "benchmark") ||
            options.graph_storage_mode != "memory" ||
            options.pq_query_worker_mode != "coupled") {
            throw std::runtime_error(
                "pq-request-mode=batch_ring requires native PQ search|benchmark "
                "with coupled workers and a memory graph");
        }
        if (options.host_demand_batch_size != 1) {
            throw std::runtime_error(
                "pq-request-mode=batch_ring currently requires "
                "host-demand-batch-size=1");
        }
    }
    if (options.rabitq_query_worker_mode != "coupled" &&
        options.rabitq_query_worker_mode != "finite") {
        throw std::runtime_error("rabitq-query-worker-mode must be one of: coupled, finite");
    }
    if (options.rabitq_finite_dispatch_mode != "batch" &&
        options.rabitq_finite_dispatch_mode != "persistent" &&
        options.rabitq_finite_dispatch_mode != "ready_ring") {
        throw std::runtime_error(
            "rabitq-finite-dispatch-mode must be one of: batch, persistent, ready_ring");
    }
    if (options.rabitq_finite_partial_batch_timeout_us < 1 ||
        options.rabitq_finite_partial_batch_timeout_us > 1000000) {
        throw std::runtime_error(
            "rabitq-finite-partial-batch-timeout-us must be in [1, 1000000]");
    }
    if (options.rabitq_query_worker_mode == "finite") {
        if (options.backend != "rabitq" ||
            (options.mode != "search" && options.mode != "benchmark")) {
            throw std::runtime_error(
                "rabitq-query-worker-mode=finite requires native --mode=search|benchmark --backend=rabitq");
        }
        if (options.graph_storage_mode != "memory") {
            throw std::runtime_error(
                "rabitq-query-worker-mode=finite currently requires --graph-storage-mode=memory");
        }
        if (options.rabitq_finite_dispatch_mode != "batch") {
            throw std::runtime_error(
                "rabitq finite dispatch currently supports only batch; persistent and ready_ring are deferred");
        }
    }
    if (options.rabitq_graph_transport != "mailbox" &&
        options.rabitq_graph_transport != "device_resident") {
        throw std::runtime_error(
            "rabitq-graph-transport must be one of: mailbox, device_resident");
    }
    if (options.rabitq_graph_transport == "device_resident") {
        const bool cpu_or_off_rerank =
            defaults::highfreq_rerank_mode_is_cpu(options.highfreq_rerank_mode) ||
            defaults::highfreq_rerank_mode_is_off(options.highfreq_rerank_mode);
        if (options.backend != "rabitq" ||
            (options.mode != "search" && options.mode != "benchmark") ||
            options.graph_storage_mode != "memory" ||
            options.rabitq_query_worker_mode != "coupled" ||
            options.rabitq_request_mode != "scan" ||
            options.latency_persistent_queue ||
            options.rabitq_gpu_graph_row_cache_mib != 0 ||

            options.range_graph_hybrid_or_enabled ||
            options.range_graph_hybrid_and_enabled ||
            !cpu_or_off_rerank) {
            throw std::runtime_error(
                "rabitq-graph-transport=device_resident requires RabitQ search|benchmark, "
                "memory graph, coupled workers, request_mode=scan, CPU/off rerank, "
                "non-persistent execution, no GPU graph row cache, and no dual graph");
        }
        if (options.degree > 32) {
            throw std::runtime_error(
                "rabitq-graph-transport=device_resident currently supports degree <= 32");
        }
    }
    if (options.rabitq_request_mode != "scan" &&
        options.rabitq_request_mode != "batch_ring") {
        throw std::runtime_error(
            "rabitq-request-mode must be one of: scan, batch_ring");
    }
    if (options.rabitq_request_batch_queries == 0 ||
        options.rabitq_request_batch_queries > 4096) {
        throw std::runtime_error(
            "rabitq-request-batch-queries must be in [1, 4096]");
    }
    if (options.rabitq_request_consumed_memory != "host" &&
        options.rabitq_request_consumed_memory != "device") {
        throw std::runtime_error(
            "rabitq-request-consumed-memory must be one of: host, device");
    }
    if (options.rabitq_request_mode == "batch_ring") {
        if (options.backend != "rabitq" ||
            (options.mode != "search" && options.mode != "benchmark") ||
            options.graph_storage_mode != "memory" ||
            options.rabitq_query_worker_mode != "coupled") {
            throw std::runtime_error(
                "rabitq-request-mode=batch_ring requires native RabitQ "
                "search|benchmark with coupled workers and a memory graph");
        }
        if (options.host_demand_batch_size != 1) {
            throw std::runtime_error(
                "rabitq-request-mode=batch_ring currently requires "
                "host-demand-batch-size=1");
        }
    }
    if (options.degree <= 0 || options.degree > defaults::max_degree) {
        throw std::runtime_error("degree must be in [1, 64] for the current compiled kernels");
    }
    if (options.topk <= 0 || options.topk > defaults::max_topk) {
        throw std::runtime_error("topk must be in [1, 10] for the current compiled kernels");
    }
    const bool range_posting_enabled = options.range_posting_mode != "off";
    if (options.range_posting_mode != "off" &&
        options.range_posting_mode != "select" &&
        options.range_posting_mode != "hybrid") {
        throw std::runtime_error("range-posting-mode must be one of: off, select, hybrid");
    }
    if (options.range_posting_shortlist < options.topk ||
        options.range_posting_shortlist > defaults::max_range_posting_shortlist) {
        throw std::runtime_error("range-posting-shortlist must be in [topk, 400]");
    }
    if (options.range_posting_postfilter_shortlist_multiplier <= 0 ||
        options.range_posting_shortlist >
            defaults::max_range_posting_postfilter_shortlist /
                options.range_posting_postfilter_shortlist_multiplier) {
        throw std::runtime_error(
            "range-posting postfilter shortlist multiplier must be positive and "
            "produce at most 4096 candidates");
    }
    if (options.range_posting_and_filter_mode != "auto" &&
        options.range_posting_and_filter_mode != "pre" &&
        options.range_posting_and_filter_mode != "post" &&
        options.range_posting_and_filter_mode != "label") {
        throw std::runtime_error(
            "range-posting-and-filter-mode must be one of: auto, pre, post, label");
    }
    if (options.range_label_and_policy != "auto" &&
        options.range_label_and_policy != "label_prefer" &&
        options.range_label_and_policy != "range_prefer" &&
        options.range_label_and_policy != "dual") {
        throw std::runtime_error(
            "range-label-and-policy must be one of: auto, label_prefer, range_prefer, dual");
    }
    if ((options.range_label_and_policy == "range_prefer" ||
         options.range_label_and_policy == "dual") &&
        options.range_posting_and_filter_mode == "label") {
        throw std::runtime_error(
            "range-label-and-policy range_prefer or dual cannot use range-posting-and-filter-mode=label");
    }
    if (options.range_label_and_policy != "auto" && !range_posting_enabled) {
        throw std::runtime_error(
            "non-auto range-label-and-policy requires range-posting-mode=select|hybrid");
    }
    if (range_posting_enabled) {
        if ((options.backend != "rabitq" && options.backend != "full_gpu") ||
            (options.mode != "search" && options.mode != "benchmark")) {
            throw std::runtime_error("range-posting-mode=select|hybrid requires search or benchmark mode");
        }
        if (options.range_posting_index_path.empty() ||
            options.range_posting_scan_threshold == 0) {
            throw std::runtime_error(
                "enabled range posting requires range-posting-index-path and a positive scan threshold");
        }
        if (options.backend == "rabitq" &&
            (!defaults::highfreq_rerank_mode_is_cpu(options.highfreq_rerank_mode) ||
             !defaults::highfreq_rerank_algorithm_is_exact(options.highfreq_rerank_algorithm))) {
            throw std::runtime_error(
                "enabled range posting requires highfreq-rerank-mode=cpu and highfreq-rerank-algorithm=exact");
        }
    }
    if (options.range_posting_drop_wide_hybrid_or && !range_posting_enabled) {
        throw std::runtime_error(
            "range-posting-drop-wide-hybrid-or requires range-posting-mode=select|hybrid");
    }
    const bool range_graph_enabled = options.range_graph_hybrid_or_enabled ||
        options.range_graph_hybrid_and_enabled;
    if (range_graph_enabled &&
        (options.backend != "rabitq" ||
         (options.mode != "search" && options.mode != "benchmark") ||
         !range_posting_enabled)) {
        throw std::runtime_error(
            "range-graph hybrid execution requires RabitQ search or benchmark mode and "
            "range-posting-mode=select|hybrid");
    }
    if (options.rabitq_posting_stream_batch_size == 0 ||
        options.rabitq_posting_stream_batch_size > 1048576) {
        throw std::runtime_error("rabitq-posting-stream-batch-size must be in [1,1048576]");
    }
    if (options.rabitq_graph_persistent_workers &&
        (options.backend != "rabitq" ||
         (options.mode != "search" && options.mode != "benchmark") ||
         options.rabitq_query_worker_mode != "coupled" ||
         !defaults::highfreq_rerank_mode_is_cpu(options.highfreq_rerank_mode))) {
        throw std::runtime_error(
            "rabitq-graph-persistent-workers requires coupled RabitQ search/benchmark with CPU rerank");
    }
    if (options.rabitq_posting_persistent_workers && !options.rabitq_posting_streaming) {
        throw std::runtime_error("rabitq-posting-persistent-workers requires posting streaming");
    }
    if (options.rabitq_posting_streaming &&
        (options.backend != "rabitq" || (options.mode != "search" && options.mode != "benchmark"))) {
        throw std::runtime_error("rabitq-posting-streaming requires RabitQ search/benchmark");
    }
    if (options.rabitq_range_posting_segment_rows != 0 &&
        (options.rabitq_range_posting_segment_rows < 32 ||
         options.rabitq_range_posting_segment_rows > 1048576 ||
         options.backend != "rabitq" || !range_posting_enabled ||
         (options.mode != "search" && options.mode != "benchmark"))) {
        throw std::runtime_error(
            "rabitq-range-posting-segment-rows must be 0 or [32,1048576] with enabled RabitQ range posting search/benchmark");
    }
    if (options.rabitq_posting_schedule != "legacy" &&
        options.rabitq_posting_schedule != "graph_first" &&
        options.rabitq_posting_schedule != "posting_first") {
        throw std::runtime_error(
            "rabitq-posting-schedule must be one of: legacy, graph_first, posting_first");
    }
    if (options.rabitq_posting_schedule != "legacy" &&
        (options.backend != "rabitq" ||
         (options.mode != "search" && options.mode != "benchmark") ||
         !defaults::highfreq_rerank_mode_is_cpu(options.highfreq_rerank_mode))) {
        throw std::runtime_error(
            "non-legacy rabitq-posting-schedule requires RabitQ search or benchmark with CPU rerank");
    }
    if (options.rabitq_dual_graph_cpu_gpu_overlap && !range_graph_enabled) {
        throw std::runtime_error(
            "rabitq-dual-graph-cpu-gpu-overlap requires range-graph hybrid execution");
    }
    if (options.range_graph_hybrid_and_enabled && !options.per_step_full_filter) {
        throw std::runtime_error(
            "range-graph hybrid-AND execution requires per-step-full-filter=true");
    }
    if (options.range_graph_hybrid_and_enabled && options.range_posting_mode != "select") {
        throw std::runtime_error(
            "range-graph hybrid-AND execution requires range-posting-mode=select");
    }
    if (!defaults::is_valid_highfreq_rerank_mode(options.highfreq_rerank_mode)) {
        throw std::runtime_error("highfreq-rerank-mode must be one of: gpu, gpu_post, cpu, off");
    }
    if (defaults::highfreq_rerank_mode_is_gpu_post(options.highfreq_rerank_mode) &&
        options.backend != "rabitq") {
        throw std::runtime_error(
            "highfreq-rerank-mode=gpu_post currently requires --backend=rabitq");
    }
    if (!defaults::is_valid_highfreq_rerank_algorithm(options.highfreq_rerank_algorithm)) {
        throw std::runtime_error("highfreq-rerank-algorithm must be one of: exact, heuristic");
    }
    if (options.highfreq_rerank_shortlist < options.topk ||
        options.highfreq_rerank_shortlist > defaults::max_highfreq_rerank_shortlist) {
        throw std::runtime_error("highfreq-rerank-shortlist must be in [topk, 512] for the current compiled kernels");
    }
    if (options.highfreq_rerank_batch_size <= 0) {
        throw std::runtime_error("highfreq-rerank-batch-size must be >= 1");
    }
    if (options.highfreq_rerank_epsilon < 0.0f) {
        throw std::runtime_error("highfreq-rerank-epsilon must be >= 0");
    }
    if (options.highfreq_rerank_stable_batches <= 0) {
        throw std::runtime_error("highfreq-rerank-stable-batches must be >= 1");
    }
    if (options.highfreq_rerank_threads <= 0) {
        throw std::runtime_error("highfreq-rerank-threads must be >= 1");
    }
    if (options.highfreq_rerank_stream_batch_size <= 0) {
        throw std::runtime_error("highfreq-rerank-stream-batch-size must be >= 1");
    }
    if (!uses_highfreq_rerank_controls(options) &&
        (!defaults::highfreq_rerank_mode_is_gpu(options.highfreq_rerank_mode) ||
         options.highfreq_rerank_algorithm != defaults::default_highfreq_rerank_algorithm ||
         options.highfreq_rerank_shortlist != defaults::default_highfreq_rerank_shortlist ||
         options.highfreq_rerank_batch_size != defaults::default_lowfreq_rerank_batch_size ||
         options.highfreq_rerank_epsilon != defaults::default_lowfreq_rerank_epsilon ||
         options.highfreq_rerank_stable_batches != defaults::default_lowfreq_rerank_stable_batches ||
         options.highfreq_rerank_threads != defaults::default_highfreq_rerank_threads ||
         options.highfreq_rerank_streaming != defaults::default_highfreq_rerank_streaming ||
         options.highfreq_rerank_stream_batch_size != defaults::default_highfreq_rerank_stream_batch_size)) {
        throw std::runtime_error(
            "highfreq-rerank-mode, highfreq-rerank-algorithm, highfreq-rerank-shortlist, "
            "highfreq-rerank-batch-size, highfreq-rerank-epsilon, highfreq-rerank-stable-batches, "
            "highfreq-rerank-threads, highfreq-rerank-streaming, and "
            "highfreq-rerank-stream-batch-size are supported by --backend=pq|rabitq only");
    }
    if (!defaults::highfreq_rerank_mode_is_cpu(options.highfreq_rerank_mode) &&
        (options.highfreq_rerank_streaming != defaults::default_highfreq_rerank_streaming ||
         options.highfreq_rerank_stream_batch_size != defaults::default_highfreq_rerank_stream_batch_size)) {
        throw std::runtime_error(
            "highfreq-rerank-streaming and highfreq-rerank-stream-batch-size require "
            "--highfreq-rerank-mode=cpu");
    }
    if (uses_lowfreq_full_filter_mode(options) &&
        !defaults::is_valid_lowfreq_full_filter_mode(options.lowfreq_full_filter_mode)) {
        throw std::runtime_error("lowfreq-full-filter-mode must be one of: pre, post, off");
    }
    if (uses_lowfreq_rerank_controls(options)) {
        if (options.lowfreq_rerank_shortlist < options.topk ||
            options.lowfreq_rerank_shortlist > defaults::max_lowfreq_rerank_shortlist) {
            throw std::runtime_error("lowfreq-rerank-shortlist must be in [topk, 400] for the current compiled kernels");
        }
        if (options.lowfreq_rerank_batch_size <= 0) {
            throw std::runtime_error("lowfreq-rerank-batch-size must be >= 1");
        }
        if (options.lowfreq_rerank_epsilon < 0.0f) {
            throw std::runtime_error("lowfreq-rerank-epsilon must be >= 0");
        }
        if (options.lowfreq_rerank_stable_batches <= 0) {
            throw std::runtime_error("lowfreq-rerank-stable-batches must be >= 1");
        }
    }
    if (options.lowfreq_rerank_merge_routes && options.backend != "rabitq") {
        throw std::runtime_error("lowfreq-rerank-merge-routes is supported by --backend=rabitq only");
    }
    if (options.lowfreq_rerank_merge_routes && !options.enable_lowfreq_rerank) {
        throw std::runtime_error("lowfreq-rerank-merge-routes requires low-frequency rerank to be enabled");
    }
    if (options.lowfreq_rerank_merge_routes &&
        options.mode != "search" && options.mode != "benchmark") {
        throw std::runtime_error("lowfreq-rerank-merge-routes is supported by search and benchmark batch execution only");
    }
    if (options.prefetch_hops < 0 || options.prefetch_hops > defaults::max_prefetch_hops) {
        throw std::runtime_error("prefetch-hops must be in [0, 3] for the current compiled kernels");
    }
    if (options.quick_filter_bonus <= 0.0f || options.quick_filter_bonus > 1.0f) {
        throw std::runtime_error("quick-filter-bonus must be in (0, 1]");
    }
    if (options.bit_filter_bonus <= 0.0f || options.bit_filter_bonus > 1.0f) {
        throw std::runtime_error("bit-filter-bonus must be in (0, 1]");
    }
    if (options.range_filter_bonus <= 0.0f || options.range_filter_bonus > 1.0f) {
        throw std::runtime_error("range-filter-bonus must be in (0, 1]");
    }
    if (!std::isfinite(options.and_dual_route_posting_ratio_threshold) ||
        options.and_dual_route_posting_ratio_threshold <= 0.0) {
        throw std::runtime_error("and-dual-route-posting-ratio-threshold must be a positive finite double");
    }
    if (!std::isfinite(options.query_manifest_min_ngram_score) ||
        options.query_manifest_min_ngram_score < 0.0f ||
        options.query_manifest_min_ngram_score > 1.0f) {
        throw std::runtime_error("query-manifest-min-ngram-score must be finite and in [0, 1]");
    }
    if (options.query_manifest_max_or_terms == 0 ||
        options.query_manifest_max_fuzzy_terms == 0 ||
        options.query_manifest_ngram_size == 0) {
        throw std::runtime_error("query manifest term caps and ngram size must be positive");
    }
    if (!options.query_manifest_operator_index_path.empty() &&
        options.query_manifest_jsonl_path.empty()) {
        throw std::runtime_error(
            "query_manifest_operator_index_path requires query_manifest_jsonl_path");
    }
    if (options.query_manifest_operator_parity_check &&
        options.query_manifest_operator_index_path.empty()) {
        throw std::runtime_error(
            "query_manifest_operator_parity_check requires query_manifest_operator_index_path");
    }
    validate_query_manifest_operator_rewrite_options(
        options.query_manifest_operator_fill_mode,
        options.query_manifest_operator_parity_check);
    if (query_manifest_operator_fill_mode_requires_operator_index(
            options.query_manifest_operator_fill_mode) &&
        options.query_manifest_operator_index_path.empty()) {
        throw std::runtime_error(
            "query_manifest_operator_fill_mode=" +
            options.query_manifest_operator_fill_mode + " requires "
            "query_manifest_operator_index_path");
    }
    if (!is_valid_or_route_order_mode(options.or_route_order_mode)) {
        throw std::runtime_error(
            "or_route_order_mode must be one of: legacy, manifest, cost_aware, "
            "coverage_cost, relevance_coverage_cost, marginal_accuracy_cost");
    }
    if (!is_valid_query_manifest_fuzzy_cap_policy(
            options.query_manifest_fuzzy_cap_policy)) {
        throw std::runtime_error(
            "query_manifest_fuzzy_cap_policy must be one of: legacy, cost_aware, "
            "coverage_cost, relevance_coverage_cost");
    }
    if (options.or_route_sketch_k < kOrRouteSketchMinK ||
        options.or_route_sketch_k > kOrRouteSketchMaxK) {
        throw std::runtime_error("or_route_sketch_k must be in [4, 4096]");
    }
    if (options.or_route_runtime_sketch_fallback_max_rows == 0 ||
        options.or_route_runtime_sketch_fallback_max_rows > 1000000) {
        throw std::runtime_error(
            "or_route_runtime_sketch_fallback_max_rows must be in [1, 1000000]");
    }
    if (!options.or_route_calibration_trace_path.empty() &&
        options.or_route_calibration_run_id.empty()) {
        throw std::runtime_error(
            "or_route_calibration_trace_path requires nonempty or_route_calibration_run_id");
    }
    if (options.or_route_order_mode == "marginal_accuracy_cost") {
        if (options.or_route_accuracy_cost_model_path.empty()) {
            throw std::runtime_error(
                "marginal_accuracy_cost requires or_route_accuracy_cost_model_path");
        }
        if (options.or_route_sketch_sidecar_path.empty() &&
            !options.or_route_allow_runtime_sketch_fallback) {
            throw std::runtime_error(
                "marginal_accuracy_cost requires or_route_sketch_sidecar_path; "
                "runtime sketch fallback is only for explicit small fixtures");
        }
    }
    if (options.warmup_runs < 0) {
        throw std::runtime_error("warmup-runs must be >= 0");
    }
    if (options.timed_runs <= 0) {
        throw std::runtime_error("timed-runs must be >= 1");
    }
    if (options.cagra_build_worker_count == 0 ||
        options.cagra_build_worker_count > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("cagra-build-workers must be between1 and INT_MAX");
    }
    if (options.stream_graph_worker_count == 0) {
        throw std::runtime_error("stream-graph-workers must be >= 1");
    }
    if (options.stream_graph_per_worker_memory_cap_bytes > 0 &&
        options.stream_graph_memory_budget_bytes > 0 &&
        options.stream_graph_per_worker_memory_cap_bytes > options.stream_graph_memory_budget_bytes) {
        throw std::runtime_error(
            "stream-graph-per-worker-memory-cap-bytes must be <= stream-graph-memory-budget-bytes");
    }
    if (options.host_service_threads <= 0) {
        throw std::runtime_error("host-service-threads must be >= 1");
    }
    if (options.host_demand_batch_size == 0) {
        throw std::runtime_error("host-demand-batch-size must be >= 1");
    }
    if (options.host_demand_batch_timeout_us < 0 ||
        options.host_demand_batch_timeout_us > 1000000) {
        throw std::runtime_error("host-demand-batch-timeout-us must be in [0, 1000000]");
    }
    if (options.host_demand_batch_size > 1 && options.host_demand_batch_timeout_us == 0) {
        throw std::runtime_error(
            "host-demand-batch-timeout-us must be >= 1 when host-demand-batch-size > 1");
    }
    if (options.graph_storage_mode != "memory" || options.range_graph_storage_mode != "memory") {
        throw std::runtime_error("graph storage mode must be memory; SSD search is no longer supported");
    }
    if (options.search_queue_size <= 0 || options.search_queue_size > 512) {
        throw std::runtime_error("search-queue-size must be in [1, 512]");
    }
    if (options.search_queue_size < options.topk) {
        throw std::runtime_error("search-queue-size must be >= topk");
    }
    if (options.search_finish_count <= 0) {
        throw std::runtime_error("search-finish-count must be >= 1");
    }
    if (options.search_max_steps < 0) {
        throw std::runtime_error("search-max-steps must be >= 0");
    }
    if (!defaults::is_valid_search_early_stop_mode(options.search_early_stop_mode)) {
        throw std::runtime_error("search-early-stop-mode must be one of: fixed, off, adaptive");
    }
    if (options.mode == "benchmark") {
        if (options.backend != "pq" && options.backend != "rabitq") {
            throw std::runtime_error("benchmark mode currently supports --backend=pq|rabitq");
        }
    }
    if (options.mode == "latency") {
        if (options.backend != "full_gpu" && options.backend != "rabitq" && options.backend != "pq") {
            throw std::runtime_error("latency mode currently supports --backend=full_gpu|rabitq|pq");
        }
        if (options.latency_query_start < 0) {
            throw std::runtime_error("latency-query-start must be >= 0");
        }
        if (options.latency_query_limit < 0) {
            throw std::runtime_error("latency-query-limit must be >= 0");
        }
        if (options.latency_queue_sweep_full_warmup && options.latency_queue_sizes.empty()) {
            throw std::runtime_error("latency-queue-sweep-full-warmup requires --latency-queue-sizes");
        }
        if (!options.latency_queue_sizes.empty()) {
            if (options.backend == "full_gpu") {
                throw std::runtime_error("latency-queue-sizes currently supports --backend=pq|rabitq only");
            }
            if (options.backend == "pq" && options.latency_lowfreq_mode != "candidate") {
                throw std::runtime_error("latency-queue-sizes with --backend=pq requires --latency-lowfreq-mode=candidate");
            }
            const std::vector<int> latency_queue_sizes = parse_int_list(options.latency_queue_sizes);
            for (int queue_size : latency_queue_sizes) {
                if (queue_size <= 0 || queue_size > 512) {
                    throw std::runtime_error("latency queue sizes must be in [1, 512]");
                }
                if (queue_size < options.topk) {
                    throw std::runtime_error("latency queue sizes must be >= topk");
                }
            }
        }
        if (options.backend == "full_gpu" && options.latency_batch_sizes.empty()) {
            throw std::runtime_error("latency-batch-sizes must not be empty");
        }
        if ((options.backend == "rabitq" || options.backend == "pq") &&
            (!options.latency_staggered || !options.latency_persistent_queue)) {
            throw std::runtime_error(options.backend + " latency mode requires --latency-staggered --latency-persistent-queue");
        }
        if ((options.backend == "rabitq" || options.backend == "pq") &&
            options.latency_lowfreq_mode != "resident" &&
            options.latency_lowfreq_mode != "candidate") {
            throw std::runtime_error("latency-lowfreq-mode must be one of: resident, candidate");
        }
    }
    (void)options;
}

static std::vector<int> parse_int_list(const std::string& s)
{
    std::vector<int> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        result.push_back(std::stoi(item));
    }
    if (result.empty()) {
        throw std::runtime_error("expected at least one integer in comma-separated list");
    }
    return result;
}

static std::vector<float> parse_float_list(const std::string& s)
{
    std::vector<float> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        result.push_back(std::stof(item));
    }
    if (result.empty()) {
        throw std::runtime_error("expected at least one float in comma-separated list");
    }
    return result;
}

static bool uses_raw_navigation_scores(const AppOptions& options)
{
    const bool pq = options.backend == "pq";
    return navigation_score_policy::uses_raw_scores(
        options.backend, options.mode,
        pq ? options.pq_query_worker_mode : options.rabitq_query_worker_mode,
        pq ? options.pq_graph_transport : options.rabitq_graph_transport,
        options.graph_storage_mode);
}

static void resolve_navigation_score_policy(const AppOptions& options)
{
    const bool raw_scores = uses_raw_navigation_scores(options);
    if (!raw_scores) return;
    if (options.mode == "benchmark" && !options.benchmark_quick_filter_bonuses.empty()) {
        navigation_score_policy::validate_bonus_axis(
            raw_scores, parse_float_list(options.benchmark_quick_filter_bonuses).size());
    }
    std::cout << "[navigation-score] priority=raw_approx+predicate_penalty"
              << " alpha=" << options.rabitq_predicate_bias_alpha
              << " score_stop=raw_approx early_stop=" << options.search_early_stop_mode
              << " legacy_bonus=disabled"
              << " quick_filter_bonus=" << options.quick_filter_bonus
              << " bit_filter_bonus=" << options.bit_filter_bonus
              << " range_filter_bonus=" << options.range_filter_bonus
              << " legacy_values=compatibility_metadata_only"
              << " benchmark_bonus_axis=" << (options.benchmark_quick_filter_bonuses.empty()
                    ? "inherited_single_value" : options.benchmark_quick_filter_bonuses)
              << '\n';
}

static AppOptionOverrides parse_app_options(int argc, char** argv)
{
    AppOptionOverrides options;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        reject_obsolete_cli_option(arg);
        if (starts_with_option(arg, "--graph-file-path=", "--graph_file_path=")) {
            options.graph_file_path = option_value(arg, "--graph-file-path=", "--graph_file_path=");
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            options.help = true;
            continue;
        }
        if (arg.rfind("--mode=", 0) == 0) {
            options.mode = arg.substr(7);
            continue;
        }
        if (starts_with_option(arg, "--cpu-rerank-schedule=", "--cpu_rerank_schedule=")) {
            options.cpu_rerank_schedule = option_value(arg, "--cpu-rerank-schedule=", "--cpu_rerank_schedule=");
            continue;
        }
        if (arg.rfind("--backend=", 0) == 0) {
            options.backend = arg.substr(10);
            continue;
        }
        if (arg == "--runtime-build-only") {
            options.mode = "build-graph";
            continue;
        }
        if (arg == "--runtime-force-rebuild") {
            options.force_rebuild = true;
            continue;
        }
        if (arg.rfind("--runtime-graph-builder=", 0) == 0) {
            options.mode = "build-graph";
            options.graph_builder = arg.substr(24);
            continue;
        }
        if (arg.rfind("--graph-builder=", 0) == 0) {
            options.graph_builder = arg.substr(16);
            continue;
        }
        if (arg.rfind("--config=", 0) == 0) {
            options.config_path = arg.substr(9);
            continue;
        }
        if (arg.rfind("--group-path=", 0) == 0) {
            options.group_path = arg.substr(13);
            continue;
        }
        if (arg.rfind("--query-vector-path=", 0) == 0) {
            options.query_vector_path = arg.substr(20);
            continue;
        }
        if (arg.rfind("--query-label-spmat-path=", 0) == 0) {
            options.query_label_spmat_path = arg.substr(25);
            continue;
        }
        if (starts_with_option(arg, "--query-manifest-jsonl-path=", "--query_manifest_jsonl_path=")) {
            options.query_manifest_jsonl_path =
                option_value(arg, "--query-manifest-jsonl-path=", "--query_manifest_jsonl_path=");
            continue;
        }
        if (starts_with_option(arg, "--query-manifest-families=", "--query_manifest_families=")) {
            options.query_manifest_families =
                option_value(arg, "--query-manifest-families=", "--query_manifest_families=");
            continue;
        }
        if (starts_with_option(arg, "--query-manifest-subtypes=", "--query_manifest_subtypes=")) {
            options.query_manifest_subtypes =
                option_value(arg, "--query-manifest-subtypes=", "--query_manifest_subtypes=");
            continue;
        }
        if (starts_with_option(arg, "--query-manifest-max-or-terms=", "--query_manifest_max_or_terms=")) {
            options.query_manifest_max_or_terms = static_cast<uint32_t>(
                std::stoul(option_value(arg,
                                        "--query-manifest-max-or-terms=",
                                        "--query_manifest_max_or_terms=")));
            continue;
        }
        if (starts_with_option(arg, "--query-manifest-max-fuzzy-terms=", "--query_manifest_max_fuzzy_terms=")) {
            options.query_manifest_max_fuzzy_terms = static_cast<uint32_t>(
                std::stoul(option_value(arg,
                                        "--query-manifest-max-fuzzy-terms=",
                                        "--query_manifest_max_fuzzy_terms=")));
            continue;
        }
        if (starts_with_option(arg,
                               "--query-manifest-fuzzy-cap-policy=",
                               "--query_manifest_fuzzy_cap_policy=")) {
            options.query_manifest_fuzzy_cap_policy = option_value(
                arg,
                "--query-manifest-fuzzy-cap-policy=",
                "--query_manifest_fuzzy_cap_policy=");
            continue;
        }
        if (starts_with_option(arg, "--query-prepare-threads=", "--query_prepare_threads=")) {
            const auto value = std::stoul(option_value(arg, "--query-prepare-threads=", "--query_prepare_threads="));
            if (value > 256) throw std::invalid_argument("query_prepare_threads must be in [0, 256]");
            options.query_prepare_threads = static_cast<uint32_t>(value);
            continue;
        }
        if (starts_with_option(arg, "--query-prepare-optimized=", "--query_prepare_optimized=")) {
            const auto value = option_value(arg, "--query-prepare-optimized=", "--query_prepare_optimized=");
            if (value != "0" && value != "1" && value != "false" && value != "true")
                throw std::invalid_argument("query_prepare_optimized must be 0, 1, false or true");
            options.query_prepare_optimized = value == "1" || value == "true";
            continue;
        }
        if (starts_with_option(arg, "--query-manifest-min-ngram-score=", "--query_manifest_min_ngram_score=")) {
            options.query_manifest_min_ngram_score = std::stof(
                option_value(arg,
                             "--query-manifest-min-ngram-score=",
                             "--query_manifest_min_ngram_score="));
            continue;
        }
        if (starts_with_option(arg, "--query-manifest-ngram-size=", "--query_manifest_ngram_size=")) {
            options.query_manifest_ngram_size = static_cast<uint32_t>(
                std::stoul(option_value(arg,
                                        "--query-manifest-ngram-size=",
                                        "--query_manifest_ngram_size=")));
            continue;
        }
        if (arg.rfind("--query-range-path=", 0) == 0) {
            options.query_range_path = arg.substr(19);
            continue;
        }
        if (arg.rfind("--range-label-manifest-path=", 0) == 0) {
            options.range_label_manifest_path = arg.substr(28);
            continue;
        }
        if (arg.rfind("--seed-sidecar-path=", 0) == 0) {
            options.seed_sidecar_path = arg.substr(20);
            continue;
        }
        if (arg.rfind("--original-label-seed-sidecar-path=", 0) == 0) {
            options.original_label_seed_sidecar_path =
                arg.substr(std::string("--original-label-seed-sidecar-path=").size());
            continue;
        }
        if (starts_with_option(arg, "--metadata-sidecar-path=", "--metadata_sidecar_path=")) {
            options.metadata_sidecar_path =
                option_value(arg, "--metadata-sidecar-path=", "--metadata_sidecar_path=");
            continue;
        }
        if (arg.rfind("--seed-selection-mode=", 0) == 0) {
            options.seed_selection_mode = arg.substr(22);
            continue;
        }
        if (arg.rfind("--pq-compressed-path=", 0) == 0) {
            options.pq_compressed_path = arg.substr(21);
            continue;
        }
        if (arg.rfind("--pq-pivots-path=", 0) == 0) {
            options.pq_pivots_path = arg.substr(17);
            continue;
        }
        if (arg.rfind("--pq-buffer-mode=", 0) == 0) {
            options.pq_buffer_mode = arg.substr(17);
            continue;
        }
        if (starts_with_option(arg,
                               "--pq-graph-transport=",
                               "--pq_graph_transport=")) {
            options.pq_graph_transport = option_value(arg,
                                                      "--pq-graph-transport=",
                                                      "--pq_graph_transport=");
            continue;
        }
        if (starts_with_option(arg,
                               "--pq-query-worker-mode=",
                               "--pq_query_worker_mode=")) {
            options.pq_query_worker_mode =
                option_value(arg,
                             "--pq-query-worker-mode=",
                             "--pq_query_worker_mode=");
            continue;
        }
        if (starts_with_option(arg,
                               "--pq-request-mode=",
                               "--pq_request_mode=")) {
            options.pq_request_mode = option_value(arg,
                                                   "--pq-request-mode=",
                                                   "--pq_request_mode=");
            continue;
        }
        if (starts_with_option(arg,
                               "--pq-request-batch-queries=",
                               "--pq_request_batch_queries=")) {
            options.pq_request_batch_queries = static_cast<size_t>(
                std::stoull(option_value(arg,
                                         "--pq-request-batch-queries=",
                                         "--pq_request_batch_queries=")));
            continue;
        }
        if (starts_with_option(arg,
                               "--pq-request-consumed-memory=",
                               "--pq_request_consumed_memory=")) {
            options.pq_request_consumed_memory =
                option_value(arg,
                             "--pq-request-consumed-memory=",
                             "--pq_request_consumed_memory=");
            continue;
        }
        if (starts_with_option(arg,
                               "--pq-finite-partial-batch-timeout-us=",
                               "--pq_finite_partial_batch_timeout_us=")) {
            options.pq_finite_partial_batch_timeout_us =
                std::stoi(option_value(arg,
                                       "--pq-finite-partial-batch-timeout-us=",
                                       "--pq_finite_partial_batch_timeout_us="));
            continue;
        }
        if (starts_with_option(arg,
                               "--pq-finite-token-transport=",
                               "--pq_finite_token_transport=")) {
            options.pq_finite_token_transport =
                option_value(arg,
                             "--pq-finite-token-transport=",
                             "--pq_finite_token_transport=");
            continue;
        }
        if (starts_with_option(arg,
                               "--pq-finite-dispatch-mode=",
                               "--pq_finite_dispatch_mode=")) {
            options.pq_finite_dispatch_mode =
                option_value(arg,
                             "--pq-finite-dispatch-mode=",
                             "--pq_finite_dispatch_mode=");
            continue;
        }
        if (starts_with_option(arg,
                               "--rabitq-graph-transport=",
                               "--rabitq_graph_transport=")) {
            options.rabitq_graph_transport = option_value(
                arg,
                "--rabitq-graph-transport=",
                "--rabitq_graph_transport=");
            continue;
        }
        // Backend-neutral spellings share the legacy option destination.
        if (arg.rfind("--predicate-", 0) == 0 || arg.rfind("--predicate_", 0) == 0) {
            arg.insert(2, arg[11] == '_' ? "rabitq_" : "rabitq-");
        }
        if (starts_with_option(arg, "--pq-lowfreq-scan-mode=", "--pq_lowfreq_scan_mode=")) {
            options.pq_lowfreq_scan_mode = option_value(arg, "--pq-lowfreq-scan-mode=", "--pq_lowfreq_scan_mode=");
            continue;
        }
        if (starts_with_option(arg, "--pq-posting-streaming=", "--pq_posting_streaming=")) {
            const auto value = option_value(arg, "--pq-posting-streaming=", "--pq_posting_streaming=");
            if (value != "0" && value != "1" && value != "false" && value != "true" && value != "off" && value != "on")
                throw std::runtime_error("invalid pq-posting-streaming boolean");
            options.pq_posting_streaming = value == "1" || value == "true" || value == "on";
            continue;
        }
        if (starts_with_option(arg, "--pq-posting-stream-batch-size=", "--pq_posting_stream_batch_size=")) {
            const auto value = std::stoull(option_value(arg, "--pq-posting-stream-batch-size=", "--pq_posting_stream_batch_size="));
            if (value == 0 || value > 1048576)
                throw std::runtime_error("pq-posting-stream-batch-size must be in [1,1048576]");
            options.pq_posting_stream_batch_size = static_cast<uint32_t>(value);
            continue;
        }
        if (starts_with_option(arg, "--posting-predicate-order=", "--posting_predicate_order=")) {
            options.posting_predicate_order = option_value(arg, "--posting-predicate-order=", "--posting_predicate_order=");
            continue;
        }
        if (starts_with_option(arg, "--rabitq-predicate-mode=", "--rabitq_predicate_mode=")) {
            const std::string value = option_value(arg, "--rabitq-predicate-mode=", "--rabitq_predicate_mode=");
            options.rabitq_predicate_mode = value;
            continue;
        }
        if (starts_with_option(arg, "--rabitq-predicate-storage=", "--rabitq_predicate_storage=")) {
            options.rabitq_predicate_storage = option_value(arg, "--rabitq-predicate-storage=", "--rabitq_predicate_storage=");
            continue;
        }
        if (starts_with_option(arg, "--rabitq-predicate-index-path=", "--rabitq_predicate_index_path=")) {
            const std::string value = option_value(arg, "--rabitq-predicate-index-path=", "--rabitq_predicate_index_path=");
            options.rabitq_predicate_index_path = value;
            continue;
        }
        if (starts_with_option(arg, "--rabitq-predicate-bias-alpha=", "--rabitq_predicate_bias_alpha=")) {
            const std::string value = option_value(arg, "--rabitq-predicate-bias-alpha=", "--rabitq_predicate_bias_alpha=");
            size_t parsed = 0;
            const double alpha = std::stod(value, &parsed);
            if (parsed != value.size()) {
                throw std::runtime_error("invalid rabitq-predicate-bias-alpha number");
            }
            options.rabitq_predicate_bias_alpha = alpha;
            continue;
        }
        if (starts_with_option(arg, "--rabitq-predicate-diagnostics=", "--rabitq_predicate_diagnostics=")) {
            const std::string value = option_value(arg, "--rabitq-predicate-diagnostics=", "--rabitq_predicate_diagnostics=");
            if (value != "0" && value != "1" && value != "false" && value != "true" &&
                value != "off" && value != "on") {
                throw std::runtime_error("invalid rabitq-predicate-diagnostics boolean");
            }
            options.rabitq_predicate_diagnostics = value == "1" || value == "true" || value == "on";
            continue;
        }
        if (starts_with_option(arg,
                               "--rabitq-query-worker-mode=",
                               "--rabitq_query_worker_mode=")) {
            options.rabitq_query_worker_mode = option_value(arg,
                                                            "--rabitq-query-worker-mode=",
                                                            "--rabitq_query_worker_mode=");
            continue;
        }
        if (starts_with_option(arg,
                               "--rabitq-finite-dispatch-mode=",
                               "--rabitq_finite_dispatch_mode=")) {
            options.rabitq_finite_dispatch_mode = option_value(arg,
                                                               "--rabitq-finite-dispatch-mode=",
                                                               "--rabitq_finite_dispatch_mode=");
            continue;
        }
        if (starts_with_option(arg,
                               "--rabitq-finite-partial-batch-timeout-us=",
                               "--rabitq_finite_partial_batch_timeout_us=")) {
            options.rabitq_finite_partial_batch_timeout_us = std::stoi(option_value(
                arg,
                "--rabitq-finite-partial-batch-timeout-us=",
                "--rabitq_finite_partial_batch_timeout_us="));
            continue;
        }
        if (starts_with_option(arg,
                               "--rabitq-request-mode=",
                               "--rabitq_request_mode=")) {
            options.rabitq_request_mode = option_value(arg,
                                                       "--rabitq-request-mode=",
                                                       "--rabitq_request_mode=");
            continue;
        }
        if (starts_with_option(arg,
                               "--rabitq-request-batch-queries=",
                               "--rabitq_request_batch_queries=")) {
            options.rabitq_request_batch_queries = static_cast<size_t>(
                std::stoull(option_value(arg,
                                         "--rabitq-request-batch-queries=",
                                         "--rabitq_request_batch_queries=")));
            continue;
        }
        if (starts_with_option(arg,
                               "--rabitq-request-consumed-memory=",
                               "--rabitq_request_consumed_memory=")) {
            options.rabitq_request_consumed_memory = option_value(
                arg,
                "--rabitq-request-consumed-memory=",
                "--rabitq_request_consumed_memory=");
            continue;
        }
        if (starts_with_option(arg,
                               "--rabitq-gpu-graph-row-cache-mib=",
                               "--rabitq_gpu_graph_row_cache_mib=")) {
            options.rabitq_gpu_graph_row_cache_mib = static_cast<size_t>(
                std::stoull(option_value(arg,
                                         "--rabitq-gpu-graph-row-cache-mib=",
                                         "--rabitq_gpu_graph_row_cache_mib=")));
            continue;
        }
        if (arg.rfind("--host-reply-wait-policy=", 0) == 0) {
            options.host_reply_wait_policy = arg.substr(25);
            continue;
        }
        if (arg.rfind("--routing-mode=", 0) == 0) {
            options.routing_mode = arg.substr(15);
            continue;
        }
        if (starts_with_option(arg,
                               "--and-dual-route-posting-ratio-threshold=",
                               "--and_dual_route_posting_ratio_threshold=")) {
            options.and_dual_route_posting_ratio_threshold =
                std::stod(option_value(arg,
                                       "--and-dual-route-posting-ratio-threshold=",
                                       "--and_dual_route_posting_ratio_threshold="));
            continue;
        }
        if (arg.rfind("--output-csv-path=", 0) == 0) {
            options.output_csv_path = arg.substr(18);
            continue;
        }
        if (arg == "--full-filter") {
            options.full_filter = true;
            continue;
        }
        if (starts_with_option(arg, "--overlap-graph-path=", "--overlap_graph_path=")) {
            options.overlap_graph_path = option_value(arg, "--overlap-graph-path=", "--overlap_graph_path=");
            continue;
        }
        if (starts_with_option(arg, "--overlap-manifest-path=", "--overlap_manifest_path=")) {
            options.overlap_manifest_path = option_value(arg, "--overlap-manifest-path=", "--overlap_manifest_path=");
            continue;
        }
        if (starts_with_option(arg, "--overlap-max-bridges=", "--overlap_max_bridges=")) {
            options.overlap_max_bridges = static_cast<uint32_t>(std::stoul(option_value(arg, "--overlap-max-bridges=", "--overlap_max_bridges=")));
            continue;
        }
        if (arg == "--full-gpu-fp16" || arg == "--use-fp16") {
            options.full_gpu_fp16 = true;
            continue;
        }
        if (starts_with_option(arg, "--full-gpu-fp16=", "--full_gpu_fp16=")) {
            const std::string value = option_value(arg, "--full-gpu-fp16=", "--full_gpu_fp16=");
            options.full_gpu_fp16 = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg == "--no-full-filter") {
            options.full_filter = false;
            continue;
        }
        if (arg.rfind("--full-filter=", 0) == 0) {
            const std::string value = arg.substr(14);
            options.full_filter = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (starts_with_option(arg, "--lowfreq-full-filter-mode=", "--lowfreq_full_filter_mode=")) {
            options.lowfreq_full_filter_mode =
                option_value(arg, "--lowfreq-full-filter-mode=", "--lowfreq_full_filter_mode=");
            continue;
        }
        if (arg.rfind("--rabitq-artifact-path=", 0) == 0) {
            options.rabitq_artifact_path = arg.substr(23);
            continue;
        }
        if (arg.rfind("--rabitq-cuvs-artifact-path=", 0) == 0) {
            options.rabitq_cuvs_artifact_path = arg.substr(28);
            continue;
        }
        if (arg.rfind("--rabitq-num-anchors=", 0) == 0) {
            options.rabitq_num_anchors = static_cast<uint32_t>(std::stoul(arg.substr(21)));
            continue;
        }
        if (arg.rfind("--rabitq-bits-per-dim=", 0) == 0) {
            options.rabitq_bits_per_dim = static_cast<uint32_t>(std::stoul(arg.substr(22)));
            continue;
        }
        if (arg == "--enable-rabitq-rerank") {
            options.rabitq_enable_rerank = true;
            options.enable_lowfreq_rerank = true;
            continue;
        }
        if (arg == "--disable-rabitq-rerank") {
            options.rabitq_enable_rerank = false;
            options.enable_lowfreq_rerank = false;
            continue;
        }
        if (starts_with_option(arg, "--rabitq-enable-rerank=", "--rabitq_enable_rerank=")) {
            const std::string value = option_value(arg, "--rabitq-enable-rerank=", "--rabitq_enable_rerank=");
            options.rabitq_enable_rerank = (value == "1" || value == "true" || value == "on");
            options.enable_lowfreq_rerank = *options.rabitq_enable_rerank;
            continue;
        }
        if (starts_with_option(arg, "--rabitq-rerank=", "--rabitq_rerank=")) {
            const std::string value = option_value(arg, "--rabitq-rerank=", "--rabitq_rerank=");
            options.rabitq_enable_rerank = (value == "1" || value == "true" || value == "on");
            options.enable_lowfreq_rerank = *options.rabitq_enable_rerank;
            continue;
        }
        if (arg == "--enable-rabitq-highfreq-rerank") {
            options.rabitq_enable_highfreq_rerank = true;
            continue;
        }
        if (arg == "--disable-rabitq-highfreq-rerank") {
            options.rabitq_enable_highfreq_rerank = false;
            continue;
        }
        if (starts_with_option(arg, "--rabitq-enable-highfreq-rerank=", "--rabitq_enable_highfreq_rerank=")) {
            const std::string value = option_value(arg,
                                                   "--rabitq-enable-highfreq-rerank=",
                                                   "--rabitq_enable_highfreq_rerank=");
            options.rabitq_enable_highfreq_rerank = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (starts_with_option(arg, "--rabitq-highfreq-rerank=", "--rabitq_highfreq_rerank=")) {
            const std::string value = option_value(arg, "--rabitq-highfreq-rerank=", "--rabitq_highfreq_rerank=");
            options.rabitq_enable_highfreq_rerank = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg == "--enable-rabitq-lowfreq-rerank") {
            options.rabitq_enable_lowfreq_rerank = true;
            options.enable_lowfreq_rerank = true;
            continue;
        }
        if (arg == "--disable-rabitq-lowfreq-rerank") {
            options.rabitq_enable_lowfreq_rerank = false;
            options.enable_lowfreq_rerank = false;
            continue;
        }
        if (starts_with_option(arg, "--rabitq-enable-lowfreq-rerank=", "--rabitq_enable_lowfreq_rerank=")) {
            const std::string value = option_value(arg,
                                                   "--rabitq-enable-lowfreq-rerank=",
                                                   "--rabitq_enable_lowfreq_rerank=");
            options.rabitq_enable_lowfreq_rerank = (value == "1" || value == "true" || value == "on");
            options.enable_lowfreq_rerank = options.rabitq_enable_lowfreq_rerank;
            continue;
        }
        if (starts_with_option(arg, "--rabitq-lowfreq-rerank=", "--rabitq_lowfreq_rerank=")) {
            const std::string value = option_value(arg, "--rabitq-lowfreq-rerank=", "--rabitq_lowfreq_rerank=");
            options.rabitq_enable_lowfreq_rerank = (value == "1" || value == "true" || value == "on");
            options.enable_lowfreq_rerank = options.rabitq_enable_lowfreq_rerank;
            continue;
        }
        if (arg == "--enable-rabitq-lowfreq-gpu-posting") {
            options.rabitq_lowfreq_gpu_posting = true;
            continue;
        }
        if (arg == "--disable-rabitq-lowfreq-gpu-posting") {
            options.rabitq_lowfreq_gpu_posting = false;
            continue;
        }
        if (starts_with_option(arg,
                               "--rabitq-lowfreq-gpu-posting=",
                               "--rabitq_lowfreq_gpu_posting=")) {
            const std::string value = option_value(arg,
                                                   "--rabitq-lowfreq-gpu-posting=",
                                                   "--rabitq_lowfreq_gpu_posting=");
            options.rabitq_lowfreq_gpu_posting =
                (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg == "--enable-lowfreq-rerank") {
            options.enable_lowfreq_rerank = true;
            options.rabitq_enable_lowfreq_rerank = true;
            continue;
        }
        if (arg == "--disable-lowfreq-rerank") {
            options.enable_lowfreq_rerank = false;
            options.rabitq_enable_lowfreq_rerank = false;
            continue;
        }
        if (starts_with_option(arg, "--enable-lowfreq-rerank=", "--enable_lowfreq_rerank=")) {
            const std::string value =
                option_value(arg, "--enable-lowfreq-rerank=", "--enable_lowfreq_rerank=");
            options.enable_lowfreq_rerank = (value == "1" || value == "true" || value == "on");
            options.rabitq_enable_lowfreq_rerank = options.enable_lowfreq_rerank;
            continue;
        }
        if (starts_with_option(arg, "--lowfreq-rerank=", "--lowfreq_rerank=")) {
            const std::string value = option_value(arg, "--lowfreq-rerank=", "--lowfreq_rerank=");
            options.enable_lowfreq_rerank = (value == "1" || value == "true" || value == "on");
            options.rabitq_enable_lowfreq_rerank = options.enable_lowfreq_rerank;
            continue;
        }
        if (starts_with_option(arg,
                               "--lowfreq-rerank-merge-routes=",
                               "--lowfreq_rerank_merge_routes=")) {
            const std::string value = option_value(arg,
                                                   "--lowfreq-rerank-merge-routes=",
                                                   "--lowfreq_rerank_merge_routes=");
            options.lowfreq_rerank_merge_routes =
                (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg == "--lowfreq-rerank-merge-routes") {
            options.lowfreq_rerank_merge_routes = true;
            continue;
        }
        if (arg == "--no-lowfreq-rerank-merge-routes") {
            options.lowfreq_rerank_merge_routes = false;
            continue;
        }
        if (arg == "--enable-rabitq-lut") {
            options.rabitq_use_lut = true;
            continue;
        }
        if (arg == "--disable-rabitq-lut") {
            options.rabitq_use_lut = false;
            continue;
        }
        if (starts_with_option(arg, "--rabitq-use-lut=", "--rabitq_use_lut=")) {
            const std::string value = option_value(arg, "--rabitq-use-lut=", "--rabitq_use_lut=");
            options.rabitq_use_lut = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (starts_with_option(arg, "--rabitq-query-rotation=", "--rabitq_query_rotation=")) {
            options.rabitq_query_rotation = option_value(arg, "--rabitq-query-rotation=", "--rabitq_query_rotation=");
            continue;
        }
        if (starts_with_option(arg, "--rabitq-centroid-distances=", "--rabitq_centroid_distances=")) {
            options.rabitq_centroid_distances = option_value(arg, "--rabitq-centroid-distances=", "--rabitq_centroid_distances=");
            continue;
        }
        if (arg == "--enable-rabitq-int8-score") {
            options.rabitq_use_int8_score = true;
            continue;
        }
        if (arg == "--disable-rabitq-int8-score") {
            options.rabitq_use_int8_score = false;
            continue;
        }
        if (starts_with_option(arg,
                               "--rabitq-use-int8-score=",
                               "--rabitq_use_int8_score=")) {
            const std::string value = option_value(arg,
                                                   "--rabitq-use-int8-score=",
                                                   "--rabitq_use_int8_score=");
            options.rabitq_use_int8_score =
                (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg.rfind("--graph-path=", 0) == 0) {
            options.graph_path = arg.substr(13);
            continue;
        }
        if (arg.rfind("--graph-storage-mode=", 0) == 0) {
            options.graph_storage_mode = arg.substr(21);
            continue;
        }

        if (arg.rfind("--lowfreq-index-path=", 0) == 0) {
            options.lowfreq_index_path = arg.substr(21);
            continue;
        }
        if (arg.rfind("--base-vector-path=", 0) == 0) {
            options.base_vector_path = arg.substr(19);
            continue;
        }

        if (starts_with_option(arg,
                               "--rerank-vector-storage-mode=",
                               "--rerank_vector_storage_mode=")) {
            options.rerank_vector_storage_mode = option_value(
                arg,
                "--rerank-vector-storage-mode=",
                "--rerank_vector_storage_mode=");
            continue;
        }
        if (starts_with_option(arg,
                               "--rerank-vector-artifact-path=",
                               "--rerank_vector_artifact_path=")) {
            options.rerank_vector_artifact_path = option_value(
                arg,
                "--rerank-vector-artifact-path=",
                "--rerank_vector_artifact_path=");
            continue;
        }

        if (starts_with_option(arg, "--skip-filtered-vector-payload=", "--skip_filtered_vector_payload=")) {
            const std::string value =
                option_value(arg, "--skip-filtered-vector-payload=", "--skip_filtered_vector_payload=");
            options.skip_filtered_vector_payload = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (starts_with_option(arg, "--per-step-full-filter=", "--per_step_full_filter=")) {
            const std::string value = option_value(arg, "--per-step-full-filter=", "--per_step_full_filter=");
            options.per_step_full_filter = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (starts_with_option(arg, "--highfreq-rerank-mode=", "--highfreq_rerank_mode=")) {
            options.highfreq_rerank_mode =
                option_value(arg, "--highfreq-rerank-mode=", "--highfreq_rerank_mode=");
            continue;
        }
        if (starts_with_option(arg, "--highfreq-rerank-algorithm=", "--highfreq_rerank_algorithm=")) {
            options.highfreq_rerank_algorithm =
                option_value(arg, "--highfreq-rerank-algorithm=", "--highfreq_rerank_algorithm=");
            continue;
        }
        if (starts_with_option(arg, "--rabitq-incremental-rerank-shortlist=", "--rabitq_incremental_rerank_shortlist=")) {
            const auto value = option_value(arg, "--rabitq-incremental-rerank-shortlist=", "--rabitq_incremental_rerank_shortlist=");
            size_t consumed = 0;
            options.rabitq_incremental_rerank_shortlist = std::stoi(value, &consumed);
            if (consumed != value.size()) throw std::invalid_argument("invalid incremental shortlist");
            continue;
        }
        if (starts_with_option(arg, "--rerank-distance-reuse=", "--rerank_distance_reuse=")) {
            const auto value = option_value(arg, "--rerank-distance-reuse=", "--rerank_distance_reuse=");
            if (value != "0" && value != "1" && value != "false" && value != "true" && value != "off" && value != "on")
                throw std::runtime_error("invalid rerank-distance-reuse boolean");
            options.rerank_distance_reuse = value == "1" || value == "true" || value == "on";
            continue;
        }
        if (starts_with_option(arg, "--highfreq-rerank-streaming=", "--highfreq_rerank_streaming=")) {
            const std::string value =
                option_value(arg, "--highfreq-rerank-streaming=", "--highfreq_rerank_streaming=");
            options.highfreq_rerank_streaming = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (starts_with_option(arg, "--highfreq-rerank-merge-routes=", "--highfreq_rerank_merge_routes=")) {
            const std::string value =
                option_value(arg, "--highfreq-rerank-merge-routes=", "--highfreq_rerank_merge_routes=");
            options.highfreq_rerank_merge_routes = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (starts_with_option(arg, "--pq-cpu-persistent-workers=", "--pq_cpu_persistent_workers=")) {
            const std::string value = option_value(
                arg, "--pq-cpu-persistent-workers=", "--pq_cpu_persistent_workers=");
            if (value != "0" && value != "1" && value != "false" && value != "true" && value != "off" && value != "on")
                throw std::runtime_error("invalid pq-cpu-persistent-workers boolean");
            options.pq_cpu_persistent_workers = value == "1" || value == "true" || value == "on";
            continue;
        }
        if (starts_with_option(arg, "--rabitq-graph-persistent-workers=", "--rabitq_graph_persistent_workers=")) {
            const std::string value = option_value(
                arg, "--rabitq-graph-persistent-workers=", "--rabitq_graph_persistent_workers=");
            if (value != "0" && value != "1" && value != "false" && value != "true" && value != "off" && value != "on")
                throw std::runtime_error("invalid rabitq-graph-persistent-workers boolean");
            options.rabitq_graph_persistent_workers = value == "1" || value == "true" || value == "on";
            continue;
        }
        if (starts_with_option(arg, "--rabitq-posting-persistent-workers=", "--rabitq_posting_persistent_workers=")) {
            const std::string value = option_value(
                arg, "--rabitq-posting-persistent-workers=", "--rabitq_posting_persistent_workers=");
            if (value != "0" && value != "1" && value != "false" && value != "true" && value != "off" && value != "on")
                throw std::runtime_error("invalid rabitq-posting-persistent-workers boolean");
            options.rabitq_posting_persistent_workers = value == "1" || value == "true" || value == "on";
            continue;
        }
        if (starts_with_option(arg, "--rabitq-posting-streaming=", "--rabitq_posting_streaming=")) {
            const std::string value = option_value(
                arg, "--rabitq-posting-streaming=", "--rabitq_posting_streaming=");
            if (value != "0" && value != "1" && value != "false" && value != "true" && value != "off" && value != "on")
                throw std::runtime_error("invalid rabitq-posting-streaming boolean");
            options.rabitq_posting_streaming = value == "1" || value == "true" || value == "on";
            continue;
        }
        if (starts_with_option(arg, "--rabitq-posting-stream-batch-size=", "--rabitq_posting_stream_batch_size=")) {
            const auto value = std::stoull(option_value(
                arg, "--rabitq-posting-stream-batch-size=", "--rabitq_posting_stream_batch_size="));
            if (value == 0 || value > 1048576) throw std::runtime_error("invalid posting stream batch size");
            options.rabitq_posting_stream_batch_size = static_cast<uint32_t>(value);
            continue;
        }
        if (starts_with_option(arg, "--rabitq-range-posting-segment-rows=", "--rabitq_range_posting_segment_rows=")) {
            const auto value = std::stoull(option_value(
                arg, "--rabitq-range-posting-segment-rows=", "--rabitq_range_posting_segment_rows="));
            if (value > 1048576) throw std::runtime_error("rabitq-range-posting-segment-rows exceeds 1048576");
            options.rabitq_range_posting_segment_rows = static_cast<uint32_t>(value);
            continue;
        }
        if (starts_with_option(arg, "--rabitq-posting-schedule=", "--rabitq_posting_schedule=")) {
            options.rabitq_posting_schedule = option_value(
                arg, "--rabitq-posting-schedule=", "--rabitq_posting_schedule=");
            continue;
        }
        if (arg == "--skip-filtered-vector-payload") {
            options.skip_filtered_vector_payload = true;
            continue;
        }
        if (arg == "--highfreq-rerank-streaming") {
            options.highfreq_rerank_streaming = true;
            continue;
        }
        if (arg == "--no-highfreq-rerank-streaming") {
            options.highfreq_rerank_streaming = false;
            continue;
        }
        if (arg == "--highfreq-rerank-merge-routes") {
            options.highfreq_rerank_merge_routes = true;
            continue;
        }
        if (arg == "--no-highfreq-rerank-merge-routes") {
            options.highfreq_rerank_merge_routes = false;
            continue;
        }
        if (arg == "--no-skip-filtered-vector-payload") {
            options.skip_filtered_vector_payload = false;
            continue;
        }
        if (arg == "--per-step-full-filter") {
            options.per_step_full_filter = true;
            continue;
        }
        if (arg == "--no-per-step-full-filter") {
            options.per_step_full_filter = false;
            continue;
        }

        if (arg.rfind("--base-range-fbin-path=", 0) == 0) {
            options.base_range_fbin_path = arg.substr(23);
            continue;
        }
        if (starts_with_option(arg,
                               "--range-posting-index-path=",
                               "--range_posting_index_path=")) {
            options.range_posting_index_path = option_value(arg,
                                                            "--range-posting-index-path=",
                                                            "--range_posting_index_path=");
            continue;
        }
        if (starts_with_option(arg,
                               "--range-posting-mode=",
                               "--range_posting_mode=")) {
            options.range_posting_mode = option_value(arg,
                                                      "--range-posting-mode=",
                                                      "--range_posting_mode=");
            continue;
        }
        if (starts_with_option(arg,
                               "--range-posting-scan-threshold=",
                               "--range_posting_scan_threshold=")) {
            options.range_posting_scan_threshold = std::stoull(option_value(
                arg,
                "--range-posting-scan-threshold=",
                "--range_posting_scan_threshold="));
            continue;
        }
        if (starts_with_option(arg,
                               "--range-posting-shortlist=",
                               "--range_posting_shortlist=")) {
            options.range_posting_shortlist = std::stoi(option_value(
                arg,
                "--range-posting-shortlist=",
                "--range_posting_shortlist="));
            continue;
        }
        if (starts_with_option(
                arg,
                "--range-posting-postfilter-shortlist-multiplier=",
                "--range_posting_postfilter_shortlist_multiplier=")) {
            options.range_posting_postfilter_shortlist_multiplier = std::stoi(option_value(
                arg,
                "--range-posting-postfilter-shortlist-multiplier=",
                "--range_posting_postfilter_shortlist_multiplier="));
            continue;
        }
        if (starts_with_option(arg,
                               "--range-posting-and-filter-mode=",
                               "--range_posting_and_filter_mode=")) {
            options.range_posting_and_filter_mode = option_value(
                arg,
                "--range-posting-and-filter-mode=",
                "--range_posting_and_filter_mode=");
            continue;
        }
        if (starts_with_option(arg,
                               "--range-label-and-policy=",
                               "--range_label_and_policy=")) {
            options.range_label_and_policy = option_value(
                arg,
                "--range-label-and-policy=",
                "--range_label_and_policy=");
            continue;
        }
        if (arg == "--range-posting-drop-wide-hybrid-or") {
            options.range_posting_drop_wide_hybrid_or = true;
            continue;
        }
        if (arg == "--no-range-posting-drop-wide-hybrid-or") {
            options.range_posting_drop_wide_hybrid_or = false;
            continue;
        }
        if (starts_with_option(arg,
                               "--range-posting-drop-wide-hybrid-or=",
                               "--range_posting_drop_wide_hybrid_or=")) {
            const std::string value = option_value(
                arg,
                "--range-posting-drop-wide-hybrid-or=",
                "--range_posting_drop_wide_hybrid_or=");
            options.range_posting_drop_wide_hybrid_or =
                (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg.rfind("--data-label-spmat-path=", 0) == 0) {
            options.data_label_spmat_path = arg.substr(24);
            continue;
        }
        if (arg.rfind("--row-limit=", 0) == 0) {
            options.row_limit = static_cast<uint32_t>(std::stoul(arg.substr(12)));
            continue;
        }
        if (starts_with_option(arg, "--cagra-build-workers=", "--cagra_build_worker_count=")) {
            const auto value = std::stoull(option_value(
                arg, "--cagra-build-workers=", "--cagra_build_worker_count="));
            if (value == 0 || value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("cagra-build-workers must be between1 and INT_MAX");
            }
            options.cagra_build_worker_count = static_cast<uint32_t>(value);
            continue;
        }
        if (starts_with_option(arg, "--stream-graph-workers=", "--stream_graph_workers=")) {
            options.stream_graph_worker_count = static_cast<uint32_t>(
                std::stoul(option_value(arg, "--stream-graph-workers=", "--stream_graph_workers=")));
            continue;
        }
        if (starts_with_option(arg,
                               "--stream-graph-worker-count=",
                               "--stream_graph_worker_count=")) {
            options.stream_graph_worker_count = static_cast<uint32_t>(
                std::stoul(option_value(arg,
                                        "--stream-graph-worker-count=",
                                        "--stream_graph_worker_count=")));
            continue;
        }
        if (starts_with_option(arg, "--stream-graph-auto-workers=", "--stream_graph_auto_workers=")) {
            const std::string value =
                option_value(arg, "--stream-graph-auto-workers=", "--stream_graph_auto_workers=");
            options.stream_graph_auto_workers = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg == "--stream-graph-auto-workers") {
            options.stream_graph_auto_workers = true;
            continue;
        }
        if (arg == "--no-stream-graph-auto-workers") {
            options.stream_graph_auto_workers = false;
            continue;
        }
        if (starts_with_option(arg,
                               "--stream-graph-memory-budget-bytes=",
                               "--stream_graph_memory_budget_bytes=")) {
            options.stream_graph_memory_budget_bytes =
                std::stoull(option_value(arg,
                                         "--stream-graph-memory-budget-bytes=",
                                         "--stream_graph_memory_budget_bytes="));
            continue;
        }
        if (starts_with_option(arg,
                               "--stream-graph-per-worker-memory-cap-bytes=",
                               "--stream_graph_per_worker_memory_cap_bytes=")) {
            options.stream_graph_per_worker_memory_cap_bytes =
                std::stoull(option_value(arg,
                                         "--stream-graph-per-worker-memory-cap-bytes=",
                                         "--stream_graph_per_worker_memory_cap_bytes="));
            continue;
        }
        if (starts_with_option(arg,
                               "--stream-graph-allow-concurrent-cagra=",
                               "--stream_graph_allow_concurrent_cagra=")) {
            const std::string value =
                option_value(arg,
                             "--stream-graph-allow-concurrent-cagra=",
                             "--stream_graph_allow_concurrent_cagra=");
            options.stream_graph_allow_concurrent_cagra =
                (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg == "--stream-graph-allow-concurrent-cagra") {
            options.stream_graph_allow_concurrent_cagra = true;
            continue;
        }
        if (arg.rfind("--specificity-threshold=", 0) == 0) {
            options.specificity_threshold = std::stoi(arg.substr(24));
            continue;
        }
        if (arg.rfind("--degree=", 0) == 0) {
            options.degree = std::stoi(arg.substr(9));
            continue;
        }
        if (arg.rfind("--topk=", 0) == 0) {
            options.topk = std::stoi(arg.substr(7));
            continue;
        }
        if (starts_with_option(arg, "--highfreq-rerank-shortlist=", "--highfreq_rerank_shortlist=")) {
            options.highfreq_rerank_shortlist =
                std::stoi(option_value(arg, "--highfreq-rerank-shortlist=", "--highfreq_rerank_shortlist="));
            continue;
        }
        if (starts_with_option(arg, "--highfreq-rerank-batch-size=", "--highfreq_rerank_batch_size=")) {
            options.highfreq_rerank_batch_size =
                std::stoi(option_value(arg, "--highfreq-rerank-batch-size=", "--highfreq_rerank_batch_size="));
            continue;
        }
        if (starts_with_option(arg, "--highfreq-rerank-epsilon=", "--highfreq_rerank_epsilon=")) {
            options.highfreq_rerank_epsilon =
                std::stof(option_value(arg, "--highfreq-rerank-epsilon=", "--highfreq_rerank_epsilon="));
            continue;
        }
        if (starts_with_option(arg, "--highfreq-rerank-stable-batches=", "--highfreq_rerank_stable_batches=")) {
            options.highfreq_rerank_stable_batches =
                std::stoi(option_value(arg,
                                       "--highfreq-rerank-stable-batches=",
                                       "--highfreq_rerank_stable_batches="));
            continue;
        }
        if (starts_with_option(arg, "--highfreq-rerank-threads=", "--highfreq_rerank_threads=")) {
            options.highfreq_rerank_threads =
                std::stoi(option_value(arg, "--highfreq-rerank-threads=", "--highfreq_rerank_threads="));
            continue;
        }
        if (starts_with_option(arg,
                               "--highfreq-rerank-stream-batch-size=",
                               "--highfreq_rerank_stream_batch_size=")) {
            options.highfreq_rerank_stream_batch_size =
                std::stoi(option_value(arg,
                                       "--highfreq-rerank-stream-batch-size=",
                                       "--highfreq_rerank_stream_batch_size="));
            continue;
        }
        if (starts_with_option(arg, "--lowfreq-rerank-shortlist=", "--lowfreq_rerank_shortlist=")) {
            options.lowfreq_rerank_shortlist =
                std::stoi(option_value(arg, "--lowfreq-rerank-shortlist=", "--lowfreq_rerank_shortlist="));
            continue;
        }
        if (starts_with_option(arg, "--lowfreq-rerank-batch-size=", "--lowfreq_rerank_batch_size=")) {
            options.lowfreq_rerank_batch_size =
                std::stoi(option_value(arg, "--lowfreq-rerank-batch-size=", "--lowfreq_rerank_batch_size="));
            continue;
        }
        if (starts_with_option(arg, "--lowfreq-rerank-epsilon=", "--lowfreq_rerank_epsilon=")) {
            options.lowfreq_rerank_epsilon =
                std::stof(option_value(arg, "--lowfreq-rerank-epsilon=", "--lowfreq_rerank_epsilon="));
            continue;
        }
        if (starts_with_option(arg, "--lowfreq-rerank-stable-batches=", "--lowfreq_rerank_stable_batches=")) {
            options.lowfreq_rerank_stable_batches =
                std::stoi(option_value(arg,
                                       "--lowfreq-rerank-stable-batches=",
                                       "--lowfreq_rerank_stable_batches="));
            continue;
        }
        if (arg.rfind("--dim=", 0) == 0) {
            options.dim = std::stoi(arg.substr(6));
            continue;
        }
        if (arg.rfind("--pq-dim=", 0) == 0) {
            options.pq_dim = std::stoi(arg.substr(9));
            continue;
        }
        if (arg.rfind("--prefetch-hops=", 0) == 0) {
            options.prefetch_hops = std::stoi(arg.substr(16));
            continue;
        }
        if (starts_with_option(arg, "--quick-filter-bonus=", "--quick_filter_bonus=")) {
            options.quick_filter_bonus = std::stof(option_value(arg, "--quick-filter-bonus=", "--quick_filter_bonus="));
            continue;
        }
        if (starts_with_option(arg, "--bit-filter-bonus=", "--bit_filter_bonus=")) {
            options.bit_filter_bonus = std::stof(option_value(arg, "--bit-filter-bonus=", "--bit_filter_bonus="));
            continue;
        }
        if (starts_with_option(arg, "--range-filter-bonus=", "--range_filter_bonus=")) {
            options.range_filter_bonus = std::stof(option_value(arg, "--range-filter-bonus=", "--range_filter_bonus="));
            continue;
        }
        if (arg.rfind("--process-cpu-start=", 0) == 0) {
            options.process_cpu_start = std::stoi(arg.substr(20));
            continue;
        }
        if (arg.rfind("--process-cpu-count=", 0) == 0) {
            options.process_cpu_count = std::stoi(arg.substr(20));
            continue;
        }
        if (starts_with_option(arg, "--host-service-threads=", "--host_service_threads=")) {
            options.host_service_threads = std::stoi(option_value(arg, "--host-service-threads=", "--host_service_threads="));
            continue;
        }
        if (arg.rfind("--host-service-cpu-start=", 0) == 0) {
            options.host_service_cpu_start = std::stoi(arg.substr(25));
            continue;
        }
        if (arg.rfind("--host-service-cpu-count=", 0) == 0) {
            options.host_service_cpu_count = std::stoi(arg.substr(25));
            continue;
        }
        if (starts_with_option(arg,
                               "--host-demand-batch-size=",
                               "--host_demand_batch_size=")) {
            options.host_demand_batch_size = static_cast<size_t>(std::stoull(option_value(
                arg,
                "--host-demand-batch-size=",
                "--host_demand_batch_size=")));
            continue;
        }
        if (starts_with_option(arg,
                               "--host-demand-batch-timeout-us=",
                               "--host_demand_batch_timeout_us=")) {
            options.host_demand_batch_timeout_us = std::stoi(option_value(
                arg,
                "--host-demand-batch-timeout-us=",
                "--host_demand_batch_timeout_us="));
            continue;
        }

        if (starts_with_option(arg, "--search-queue-size=", "--search_queue_size=")) {
            options.search_queue_size = std::stoi(option_value(arg, "--search-queue-size=", "--search_queue_size="));
            continue;
        }
        if (starts_with_option(arg, "--search-finish-count=", "--search_finish_count=")) {
            options.search_finish_count = std::stoi(option_value(arg, "--search-finish-count=", "--search_finish_count="));
            continue;
        }
        if (starts_with_option(arg, "--search-max-steps=", "--search_max_steps=")) {
            options.search_max_steps = std::stoi(option_value(arg, "--search-max-steps=", "--search_max_steps="));
            continue;
        }
        if (starts_with_option(arg, "--search-early-stop-mode=", "--search_early_stop_mode=")) {
            options.search_early_stop_mode =
                option_value(arg, "--search-early-stop-mode=", "--search_early_stop_mode=");
            continue;
        }
        if (starts_with_option(arg, "--search-early-stop=", "--search_early_stop=")) {
            options.search_early_stop_mode = option_value(arg, "--search-early-stop=", "--search_early_stop=");
            continue;
        }
        if (starts_with_option(arg, "--benchmark-queue-sizes=", "--benchmark_queue_sizes=")) {
            options.benchmark_queue_sizes = option_value(arg, "--benchmark-queue-sizes=", "--benchmark_queue_sizes=");
            continue;
        }
        if (starts_with_option(arg, "--benchmark-quick-filter-bonuses=", "--benchmark_quick_filter_bonuses=")) {
            options.benchmark_quick_filter_bonuses = option_value(arg, "--benchmark-quick-filter-bonuses=", "--benchmark_quick_filter_bonuses=");
            continue;
        }
        if (starts_with_option(arg, "--benchmark-finish-counts=", "--benchmark_finish_counts=")) {
            throw std::runtime_error(
                "benchmark-finish-counts was removed; use --search-finish-count for the fixed early-stop patience");
        }
        if (starts_with_option(arg, "--benchmark-output=", "--benchmark_output=")) {
            options.benchmark_output = option_value(arg, "--benchmark-output=", "--benchmark_output=");
            continue;
        }

        if (starts_with_option(arg, "--latency-queue-sizes=", "--latency_queue_sizes=")) {
            options.latency_queue_sizes =
                option_value(arg, "--latency-queue-sizes=", "--latency_queue_sizes=");
            continue;
        }
        if (starts_with_option(arg, "--latency-batch-sizes=", "--latency_batch_sizes=")) {
            options.latency_batch_sizes =
                option_value(arg, "--latency-batch-sizes=", "--latency_batch_sizes=");
            continue;
        }
        if (starts_with_option(arg, "--latency-output=", "--latency_output=")) {
            options.latency_output = option_value(arg, "--latency-output=", "--latency_output=");
            continue;
        }
        if (starts_with_option(arg, "--latency-trace-output=", "--latency_trace_output=")) {
            options.latency_trace_output =
                option_value(arg, "--latency-trace-output=", "--latency_trace_output=");
            continue;
        }
        if (starts_with_option(arg, "--latency-query-start=", "--latency_query_start=")) {
            options.latency_query_start =
                std::stoi(option_value(arg, "--latency-query-start=", "--latency_query_start="));
            continue;
        }
        if (starts_with_option(arg, "--latency-query-limit=", "--latency_query_limit=")) {
            options.latency_query_limit =
                std::stoi(option_value(arg, "--latency-query-limit=", "--latency_query_limit="));
            continue;
        }
        if (starts_with_option(arg, "--latency-concurrency-levels=", "--latency_concurrency_levels=")) {
            options.latency_concurrency_levels =
                option_value(arg, "--latency-concurrency-levels=", "--latency_concurrency_levels=");
            continue;
        }
        if (starts_with_option(arg, "--latency-staggered=", "--latency_staggered=")) {
            const std::string value = option_value(arg, "--latency-staggered=", "--latency_staggered=");
            options.latency_staggered = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg == "--latency-staggered") {
            options.latency_staggered = true;
            continue;
        }
        if (starts_with_option(arg, "--latency-persistent-queue=", "--latency_persistent_queue=")) {
            const std::string value =
                option_value(arg, "--latency-persistent-queue=", "--latency_persistent_queue=");
            options.latency_persistent_queue = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg == "--latency-persistent-queue") {
            options.latency_persistent_queue = true;
            continue;
        }
        if (starts_with_option(arg,
                               "--latency-queue-sweep-full-warmup=",
                               "--latency_queue_sweep_full_warmup=")) {
            const std::string value = option_value(
                arg,
                "--latency-queue-sweep-full-warmup=",
                "--latency_queue_sweep_full_warmup=");
            options.latency_queue_sweep_full_warmup = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg == "--latency-queue-sweep-full-warmup") {
            options.latency_queue_sweep_full_warmup = true;
            continue;
        }
        if (starts_with_option(arg, "--latency-lowfreq-mode=", "--latency_lowfreq_mode=")) {
            options.latency_lowfreq_mode =
                option_value(arg, "--latency-lowfreq-mode=", "--latency_lowfreq_mode=");
            continue;
        }
        if (starts_with_option(arg, "--latency-run-all-queries=", "--latency_run_all_queries=")) {
            const std::string value =
                option_value(arg, "--latency-run-all-queries=", "--latency_run_all_queries=");
            options.latency_run_all_queries = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg == "--latency-run-all-queries" || arg == "--latency-run-all") {
            options.latency_run_all_queries = true;
            continue;
        }
        if (starts_with_option(arg, "--latency-time-planner=", "--latency_time_planner=")) {
            const std::string value = option_value(arg, "--latency-time-planner=", "--latency_time_planner=");
            options.latency_time_planner = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg == "--latency-time-planner") {
            options.latency_time_planner = true;
            continue;
        }
        if (starts_with_option(arg, "--latency-online-metadata=", "--latency_online_metadata=")) {
            const std::string value =
                option_value(arg, "--latency-online-metadata=", "--latency_online_metadata=");
            options.latency_online_metadata = (value == "1" || value == "true" || value == "on");
            continue;
        }
        if (arg == "--latency-online-metadata") {
            options.latency_online_metadata = true;
            continue;
        }
        if (arg.rfind("--warmup-runs=", 0) == 0) {
            options.warmup_runs = std::stoi(arg.substr(std::string("--warmup-runs=").size()));
            continue;
        }
        if (arg.rfind("--timed-runs=", 0) == 0) {
            options.timed_runs = std::stoi(arg.substr(std::string("--timed-runs=").size()));
            continue;
        }
        if (arg == "--run-analyze") {
            options.run_analyze = true;
            continue;
        }
        if (arg.rfind("--run-analyze=", 0) == 0) {
            const std::string value = arg.substr(14);
            options.run_analyze = (value == "1" || value == "true" || value == "on");
            continue;
        }
    }
    return options;
}

static int print_usage(const char* argv0)
{
    std::cout << "Usage: " << argv0 << " [--mode=search|benchmark|latency|build-graph|build-rabitq|build-rerank-int8]"
              << " [--backend=pq|rabitq|full_gpu]\n"
              << " [--cpu-rerank-schedule=native|no_overlap|batch_overlap|streaming]\n"
              << "\n"
              << "Config:      [--config=PATH] (defaults to ./config.json if present)\n"
              << "GPU filter:  [--predicate-mode=auto|cpu|cpu_simplified|gpu_exact|gpu_exact_bias (default auto; supported paths use bias)] [--predicate-storage=roaring|csr (default csr)] [--rabitq-predicate-index-path=PATH (Roaring label residuals only)] [--predicate-bias-alpha=F>=0 (default bias 0.15; requires off/adaptive)] [--predicate-diagnostics=0|1] (legacy --rabitq-predicate-* aliases retained)\n"
              << "PQ CPU:      [--pq-cpu-persistent-workers=0|1] (coupled search/benchmark)\n"
              << "Navigation:  native coupled mailbox memory search/benchmark uses raw approximate distance plus predicate penalty. quick/bit/range-filter-bonus are inactive compatibility values there; multiple benchmark bonus values are rejected. Legacy paths retain bonus behavior.\n"
              << "RabitQ:      [--rabitq-graph-persistent-workers=0|1] [--rabitq-posting-persistent-workers=0|1] [--rabitq-posting-streaming=0|1] [--rabitq-posting-stream-batch-size=N] [--rabitq-range-posting-segment-rows=N] [--rabitq-posting-schedule=legacy|graph_first|posting_first] [--rabitq-graph-transport=mailbox|device_resident] [--rabitq-request-mode=scan|batch_ring] [--rabitq-request-batch-queries=N] [--rabitq-request-consumed-memory=host|device] [--rabitq-gpu-graph-row-cache-mib=N]\n"
              << "Range policy: [--range-label-and-policy=auto|label_prefer|range_prefer|dual] [--range-posting-and-filter-mode=auto|pre|post|label] [--range-posting-postfilter-shortlist-multiplier=N] [--range-posting-drop-wide-hybrid-or[=0|1]] [--no-range-posting-drop-wide-hybrid-or]\n"
              << "Search:      --mode=search --backend=pq|rabitq|full_gpu [--pq-graph-transport=mailbox|device_resident] [--range-posting-index-path=PATH --range-posting-mode=off|select|hybrid --range-posting-scan-threshold=N --range-posting-shortlist=N<=400 --range-posting-postfilter-shortlist-multiplier=N] [--rerank-vector-storage-mode=default|mmap_fp16|memory_int8] [--rerank-vector-artifact-path=PATH] [--pq-query-worker-mode=coupled|finite] [--pq-request-mode=scan|batch_ring] [--pq-request-batch-queries=N] [--pq-request-consumed-memory=host|device] [--pq-finite-token-transport=device_copy|mapped] [--pq-finite-dispatch-mode=batch|persistent|ready_ring] [--pq-finite-partial-batch-timeout-us=N] [--degree=N<=64] [--topk=N<=10] [--highfreq-rerank-mode=gpu|gpu_post|cpu|off --highfreq-rerank-algorithm=exact|heuristic --highfreq-rerank-shortlist=N<=512 --highfreq-rerank-batch-size=N --highfreq-rerank-epsilon=F>=0 --highfreq-rerank-stable-batches=N --highfreq-rerank-threads=N --highfreq-rerank-streaming[=0|1] --no-highfreq-rerank-streaming --highfreq-rerank-merge-routes[=0|1] --no-highfreq-rerank-merge-routes --highfreq-rerank-stream-batch-size=N] [--rerank-distance-reuse=0|1] [--pq-lowfreq-scan-mode=auto|host|resident] [--pq-posting-streaming=0|1] [--pq-posting-stream-batch-size=N] [--posting-predicate-order=filter_first|score_first] [--lowfreq-rerank-shortlist=N<=400] [--lowfreq-rerank-batch-size=N] [--lowfreq-rerank-epsilon=F>=0] [--lowfreq-rerank-stable-batches=N] [--lowfreq-rerank-merge-routes[=0|1]] [--no-lowfreq-rerank-merge-routes] [--lowfreq-full-filter-mode=pre|post|off] [--prefetch-hops=N<=3] [--quick-filter-bonus=F<=1] [--bit-filter-bonus=F<=1] [--range-filter-bonus=F<=1] [--and-dual-route-posting-ratio-threshold=F>0] [--base-range-fbin-path=PATH] [--metadata-sidecar-path=PATH] [--skip-filtered-vector-payload=0|1] [--per-step-full-filter=0|1] [--query-range-path=PATH] [--range-label-manifest-path=PATH] [--graph-storage-mode=memory] [--warmup-runs=N] [--timed-runs=N] [--process-cpu-start=N] [--process-cpu-count=N] [--host-service-threads=N] [--host-service-cpu-start=N] [--host-service-cpu-count=N] [--host-demand-batch-size=N --host-demand-batch-timeout-us=N] [--search-queue-size=N<=512] [--search-finish-count=N] [--search-max-steps=N] [--search-early-stop-mode=fixed|off|adaptive] [--host-reply-wait-policy=fixed|adaptive] [--routing-mode=greedy_min_posting|greedy_max_posting|per_label_multi_route] [--full-filter|--no-full-filter|--full-filter=0|1] [--disable-rabitq-rerank] [--disable-rabitq-highfreq-rerank] [--disable-rabitq-lowfreq-rerank] [--enable-rabitq-lut] [--enable-rabitq-int8-score]\n"
              << "Query manifest: [--query-manifest-jsonl-path=PATH] [--query-manifest-families=csv] [--query-manifest-subtypes=csv] [--query-manifest-max-or-terms=N] [--query-manifest-max-fuzzy-terms=N] [--query-manifest-fuzzy-cap-policy=legacy|cost_aware|coverage_cost|relevance_coverage_cost] [--query-manifest-min-ngram-score=F] [--query-manifest-ngram-size=N] [config: query_manifest_operator_index_path, query_manifest_operator_fill_mode=literal_only, query_manifest_operator_parity_check]\n"
              << "Graph files: --graph-path=PATH loads adjacency into memory; --graph-file-path=PATH selects the optional serialized build output.\n"
              << "Benchmark:   --mode=benchmark --backend=pq|rabitq [--benchmark-quick-filter-bonuses=0.95] [--benchmark-queue-sizes=64,128,256,512] [--benchmark-output=PATH] [search options reused for runtime setup]\n"
              << "Latency:     --mode=latency --backend=full_gpu|rabitq|pq [full_gpu: --latency-batch-sizes=1,2,4,8 --latency-concurrency-levels=0,32,64,128 (0=auto)] [rabitq|pq: --latency-staggered --latency-persistent-queue --latency-concurrency-levels=0,32,64,128 (0=auto) [--latency-queue-sizes=64,128,256] [--latency-queue-sweep-full-warmup]] [--latency-output=PATH] [--latency-trace-output=PATH] [--latency-query-start=N] [--latency-query-limit=N] [--latency-run-all-queries] [--latency-time-planner] [--latency-online-metadata] [--warmup-runs=N] [--timed-runs=N]\n"
              << "Build graph: --mode=build-graph [--graph-file-path=PATH] [--graph-builder=cagra|cagra_stream|vamana|seed_sidecars] [--graph-path=PATH] [--lowfreq-index-path=PATH] [--base-vector-path=PATH] [--data-label-spmat-path=PATH] [--group-path=PATH] [--row-limit=N] [--specificity-threshold=N] [--degree=N<=64] [--cagra-build-workers=N] [--stream-graph-workers=N] [--stream-graph-auto-workers=0|1] [--stream-graph-memory-budget-bytes=N] [--stream-graph-per-worker-memory-cap-bytes=N] [--stream-graph-allow-concurrent-cagra=0|1] [--run-analyze=0|1]\n"
              << "Build RabitQ: --mode=build-rabitq --rabitq-cuvs-artifact-path=CUVS_INDEX [--rabitq-artifact-path=PATH]\n"
              << "Build INT8 rerank: --mode=build-rerank-int8 --base-vector-path=PATH --rerank-vector-artifact-path=PATH [--row-limit=N] [--force-rebuild]\n"
              << "Debug search: add [--prefetch-hops=N] to control prefetch depth\n";
    return 0;
}

static backend_pq::PqSearchOptions make_pq_search_options(const AppOptions& options);
static backend_full_gpu::FullGpuSearchOptions make_full_gpu_search_options(const AppOptions& options);
static backend_rabitq::RabitqSearchOptions make_rabitq_search_options(const AppOptions& options);

static int run_native_pq_search(const AppOptions& options)
{
    return backend_pq::run_pq_search(make_pq_search_options(options));
}

static int run_full_gpu_search(const AppOptions& options)
{
    return backend_full_gpu::run_full_gpu_search(make_full_gpu_search_options(options));
}

static int run_full_gpu_latency(const AppOptions& options)
{
    return backend_full_gpu::run_full_gpu_latency_benchmark(make_full_gpu_search_options(options));
}

static int run_rabitq_latency(const AppOptions& options)
{
    return backend_rabitq::run_rabitq_latency_benchmark(make_rabitq_search_options(options));
}

static int run_pq_latency(const AppOptions& options)
{
    return backend_pq::run_pq_latency_benchmark(make_pq_search_options(options));
}

static backend_full_gpu::FullGpuSearchOptions make_full_gpu_search_options(const AppOptions& options)
{
    backend_full_gpu::FullGpuSearchOptions search_options;
    search_options.graph_path = options.graph_path;
    search_options.use_fp16 = options.full_gpu_fp16;
    search_options.overlap_graph_path = options.overlap_graph_path;
    search_options.overlap_manifest_path = options.overlap_manifest_path;
    search_options.overlap_max_bridges = options.overlap_max_bridges;

    search_options.graph_storage_mode = options.graph_storage_mode;
    search_options.lowfreq_index_path = options.lowfreq_index_path;
    search_options.data_label_spmat_path = options.data_label_spmat_path;
    search_options.query_label_spmat_path = options.query_label_spmat_path;
    search_options.query_manifest_jsonl_path = options.query_manifest_jsonl_path;
    search_options.query_manifest_families = options.query_manifest_families;
    search_options.query_manifest_subtypes = options.query_manifest_subtypes;
    search_options.query_manifest_max_or_terms = options.query_manifest_max_or_terms;
    search_options.query_manifest_max_fuzzy_terms = options.query_manifest_max_fuzzy_terms;
    search_options.query_manifest_min_ngram_score = options.query_manifest_min_ngram_score;
    search_options.query_manifest_ngram_size = options.query_manifest_ngram_size;
    search_options.base_vector_path = options.base_vector_path;
    search_options.base_range_fbin_path = options.base_range_fbin_path;
    search_options.range_posting_index_path = options.range_posting_index_path;
    search_options.range_posting_mode = options.range_posting_mode;
    search_options.range_posting_scan_threshold = options.range_posting_scan_threshold;
    search_options.range_posting_shortlist = options.range_posting_shortlist;
    search_options.range_posting_and_filter_mode = options.range_posting_and_filter_mode;
    search_options.range_label_and_policy = options.range_label_and_policy;
    search_options.range_posting_drop_wide_hybrid_or = options.range_posting_drop_wide_hybrid_or;
    search_options.range_graph_hybrid_or_enabled = options.range_graph_hybrid_or_enabled;
    search_options.range_graph_hybrid_and_enabled = options.range_graph_hybrid_and_enabled;
    search_options.query_vector_path = options.query_vector_path;
    search_options.query_range_path = options.query_range_path;
    search_options.range_label_manifest_path = options.range_label_manifest_path;
    search_options.seed_sidecar_path = options.seed_sidecar_path;
    search_options.original_label_seed_sidecar_path = options.original_label_seed_sidecar_path;
    search_options.metadata_sidecar_path = options.metadata_sidecar_path;
    search_options.seed_selection_mode = options.seed_selection_mode;
    search_options.group_path = options.group_path;
    search_options.output_csv_path = options.output_csv_path;
    search_options.routing_mode = options.routing_mode;
    search_options.and_dual_route_posting_ratio_threshold =
        options.and_dual_route_posting_ratio_threshold;
    search_options.search_early_stop_mode = options.search_early_stop_mode;
    search_options.dataset_cutoff = options.row_limit;
    search_options.specificity_threshold = options.specificity_threshold;
    search_options.dim = options.dim;
    search_options.degree = options.degree;
    search_options.topk = options.topk;
    search_options.search_queue_size = options.search_queue_size;
    search_options.search_finish_count = options.search_finish_count;
    search_options.search_max_steps = options.search_max_steps;
    search_options.warmup_runs = options.warmup_runs;
    search_options.timed_runs = options.timed_runs;
    search_options.latency_batch_sizes = parse_int_list(options.latency_batch_sizes);
    search_options.latency_output_path = options.latency_output;
    search_options.latency_trace_output_path = options.latency_trace_output;
    search_options.latency_query_start = options.latency_query_start;
    search_options.latency_query_limit = options.latency_query_limit;
    search_options.latency_concurrency_levels = options.latency_concurrency_levels.empty()
        ? std::vector<int>()
        : parse_int_list(options.latency_concurrency_levels);
    search_options.latency_run_all_queries = options.latency_run_all_queries;
    search_options.latency_time_planner = options.latency_time_planner;
    search_options.latency_online_metadata = options.latency_online_metadata;
    search_options.latency_staggered = options.latency_staggered;
    search_options.latency_persistent_queue = options.latency_persistent_queue;
    return search_options;
}

static backend_pq::PqSearchOptions make_pq_search_options(const AppOptions& options)
{
    backend_pq::PqSearchOptions search_options;
    search_options.cpu_rerank_schedule = cpu_rerank_schedule::parse(options.cpu_rerank_schedule);
    search_options.raw_navigation_scores = uses_raw_navigation_scores(options);
    search_options.lowfreq_scan_mode = options.pq_lowfreq_scan_mode;
    search_options.posting_streaming = options.pq_posting_streaming;
    search_options.posting_stream_batch_size = options.pq_posting_stream_batch_size;
    search_options.posting_predicate_order = options.posting_predicate_order;
    search_options.predicate_mode = options.rabitq_predicate_mode;
    search_options.predicate_storage = options.rabitq_predicate_storage;
    search_options.predicate_bias_alpha = options.rabitq_predicate_bias_alpha;
    search_options.predicate_diagnostics = options.rabitq_predicate_diagnostics;
    search_options.files.compressed_path = options.pq_compressed_path;
    search_options.files.pivots_path = options.pq_pivots_path;
    search_options.graph_transport = options.pq_graph_transport;
    search_options.query_worker_mode = options.pq_query_worker_mode;
    search_options.cpu_persistent_workers = options.pq_cpu_persistent_workers;
    search_options.request_mode = options.pq_request_mode;
    search_options.request_batch_queries = options.pq_request_batch_queries;
    search_options.request_consumed_memory = options.pq_request_consumed_memory;
    search_options.finite_token_transport = options.pq_finite_token_transport;
    search_options.finite_dispatch_mode = options.pq_finite_dispatch_mode;
    search_options.finite_partial_batch_timeout_us =
        options.pq_finite_partial_batch_timeout_us;
    search_options.buffer_mode = options.pq_buffer_mode;
    search_options.host_reply_wait_policy = options.host_reply_wait_policy;
    search_options.routing_mode = options.routing_mode;
    search_options.graph_path = options.graph_path;
    search_options.graph_storage_mode = options.graph_storage_mode;
    search_options.skip_filtered_vector_payload = options.skip_filtered_vector_payload;
    search_options.per_step_full_filter = options.per_step_full_filter;
    search_options.highfreq_rerank_mode = options.highfreq_rerank_mode;
    search_options.highfreq_rerank_algorithm = options.highfreq_rerank_algorithm;
    search_options.highfreq_rerank_streaming = options.highfreq_rerank_streaming;
    search_options.rerank_distance_reuse = options.rerank_distance_reuse;
    search_options.lowfreq_index_path = options.lowfreq_index_path;
    search_options.data_label_spmat_path = options.data_label_spmat_path;
    search_options.query_label_spmat_path = options.query_label_spmat_path;
    search_options.query_manifest_jsonl_path = options.query_manifest_jsonl_path;
    search_options.query_manifest_families = options.query_manifest_families;
    search_options.query_manifest_subtypes = options.query_manifest_subtypes;
    search_options.query_manifest_max_or_terms = options.query_manifest_max_or_terms;
    search_options.query_manifest_max_fuzzy_terms = options.query_manifest_max_fuzzy_terms;
    search_options.query_manifest_min_ngram_score = options.query_manifest_min_ngram_score;
    search_options.query_manifest_ngram_size = options.query_manifest_ngram_size;
    search_options.base_vector_path = options.base_vector_path;
    search_options.base_range_fbin_path = options.base_range_fbin_path;
    search_options.query_vector_path = options.query_vector_path;
    search_options.query_range_path = options.query_range_path;
    search_options.range_label_manifest_path = options.range_label_manifest_path;
    search_options.seed_sidecar_path = options.seed_sidecar_path;
    search_options.original_label_seed_sidecar_path = options.original_label_seed_sidecar_path;
    search_options.metadata_sidecar_path = options.metadata_sidecar_path;
    search_options.seed_selection_mode = options.seed_selection_mode;
    search_options.group_path = options.group_path;
    search_options.output_csv_path = options.output_csv_path;
    search_options.full_filter = options.full_filter;
    search_options.dataset_cutoff = options.row_limit;
    search_options.specificity_threshold = options.specificity_threshold;
    search_options.dim = options.dim;
    search_options.pq_dim = options.pq_dim;
    search_options.degree = options.degree;
    search_options.topk = options.topk;
    search_options.highfreq_rerank_shortlist = options.highfreq_rerank_shortlist;
    search_options.highfreq_rerank_batch_size = options.highfreq_rerank_batch_size;
    search_options.highfreq_rerank_epsilon = options.highfreq_rerank_epsilon;
    search_options.highfreq_rerank_stable_batches = options.highfreq_rerank_stable_batches;
    search_options.highfreq_rerank_threads = options.highfreq_rerank_threads;
    search_options.highfreq_rerank_stream_batch_size = options.highfreq_rerank_stream_batch_size;
    search_options.lowfreq_rerank_shortlist = options.lowfreq_rerank_shortlist;
    search_options.lowfreq_rerank_batch_size = options.lowfreq_rerank_batch_size;
    search_options.lowfreq_rerank_epsilon = options.lowfreq_rerank_epsilon;
    search_options.lowfreq_rerank_stable_batches = options.lowfreq_rerank_stable_batches;
    search_options.enable_lowfreq_rerank = options.enable_lowfreq_rerank;
    search_options.lowfreq_full_filter_mode = options.lowfreq_full_filter_mode;
    search_options.prefetch_hops = options.prefetch_hops;
    search_options.quick_filter_bonus = options.quick_filter_bonus;
    search_options.bit_filter_bonus = options.bit_filter_bonus;
    search_options.range_filter_bonus = options.range_filter_bonus;
    search_options.and_dual_route_posting_ratio_threshold =
        options.and_dual_route_posting_ratio_threshold;
    search_options.warmup_runs = options.warmup_runs;
    search_options.timed_runs = options.timed_runs;
    search_options.latency_batch_sizes = parse_int_list(options.latency_batch_sizes);
    search_options.latency_output_path = options.latency_output;
    search_options.latency_trace_output_path = options.latency_trace_output;
    search_options.latency_query_start = options.latency_query_start;
    search_options.latency_query_limit = options.latency_query_limit;
    search_options.latency_concurrency_levels = options.latency_concurrency_levels.empty()
        ? std::vector<int>()
        : parse_int_list(options.latency_concurrency_levels);
    search_options.latency_run_all_queries = options.latency_run_all_queries;
    search_options.latency_time_planner = options.latency_time_planner;
    search_options.latency_online_metadata = options.latency_online_metadata;
    search_options.latency_staggered = options.latency_staggered;
    search_options.latency_persistent_queue = options.latency_persistent_queue;
    search_options.latency_lowfreq_mode = options.latency_lowfreq_mode;
    search_options.process_cpu_start = options.process_cpu_start;
    search_options.process_cpu_count = options.process_cpu_count;
    search_options.host_service_threads = options.host_service_threads;
    search_options.host_service_cpu_start = options.host_service_cpu_start;
    search_options.host_service_cpu_count = options.host_service_cpu_count;
    search_options.host_demand_batch_size = options.host_demand_batch_size;
    search_options.host_demand_batch_timeout_us = options.host_demand_batch_timeout_us;
    search_options.skip_filtered_vector_payload = options.skip_filtered_vector_payload;
    search_options.per_step_full_filter = options.per_step_full_filter;
    search_options.search_queue_size = options.search_queue_size;
    search_options.search_finish_count = options.search_finish_count;
    search_options.search_max_steps = options.search_max_steps;
    search_options.search_early_stop_mode = options.search_early_stop_mode;
    return search_options;
}

static void configure_benchmark_runtime_environment()
{
    setenv("GO_FANN_BENCHMARK_QUIET", "1", 1);
    unsetenv("GO_FANN_ASYNC_ROW_TRACE_PREFIX");
    unsetenv("GOFANN_DEBUG_RUNTIME");
    unsetenv("GO_FANN_DEBUG_DISK_ASYNC");
    unsetenv("GO_FANN_DEBUG_HANG");
    unsetenv("GO_FANN_FULL_DEBUG_STATS");
}

static std::string append_suffix_before_extension(const std::string& path, const std::string& suffix)
{
    const size_t slash = path.find_last_of("/\\");
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot <= slash)) {
        return path + suffix;
    }
    return path.substr(0, dot) + suffix + path.substr(dot);
}

static std::string replace_extension_with_suffix(const std::string& path,
                                                 const std::string& suffix,
                                                 const std::string& extension)
{
    std::filesystem::path output_path(path);
    if (output_path.has_extension()) {
        output_path.replace_extension("");
    }
    output_path += suffix + extension;
    return output_path.string();
}

static std::string shell_quote(const std::string& value)
{
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

static std::string make_command_line(int argc, char** argv)
{
    std::ostringstream command;
    for (int i = 0; i < argc; ++i) {
        if (i > 0) {
            command << ' ';
        }
        command << shell_quote(argv[i]);
    }
    return command.str();
}

static nlohmann::json make_effective_config_json(const AppOptions& options)
{
    nlohmann::json config;
    auto set = [&](const char* key, const auto& value) {
        config[key] = value;
    };

    set("mode", options.mode);
    set("backend", options.backend);
    set("cpu_rerank_schedule", options.cpu_rerank_schedule);
    set("graph_builder", options.graph_builder);
    set("graph_path", options.graph_path);
    set("graph_file_path", options.graph_file_path);
    set("graph_storage_mode", options.graph_storage_mode);
    set("rerank_vector_storage_mode", options.rerank_vector_storage_mode);
    set("rerank_vector_artifact_path", options.rerank_vector_artifact_path);
    set("skip_filtered_vector_payload", options.skip_filtered_vector_payload);
    set("per_step_full_filter", options.per_step_full_filter);
    set("highfreq_rerank_mode", options.highfreq_rerank_mode);
    set("highfreq_rerank_algorithm", options.highfreq_rerank_algorithm);
    set("highfreq_rerank_streaming", options.highfreq_rerank_streaming);
    set("highfreq_rerank_merge_routes", options.highfreq_rerank_merge_routes);
    set("rerank_distance_reuse", options.rerank_distance_reuse);
    set("rabitq_incremental_rerank_shortlist", options.rabitq_incremental_rerank_shortlist);
    set("lowfreq_rerank_merge_routes", options.lowfreq_rerank_merge_routes);
    set("lowfreq_index_path", options.lowfreq_index_path);
    set("base_vector_path", options.base_vector_path);
    set("base_range_fbin_path", options.base_range_fbin_path);
    set("range_posting_index_path", options.range_posting_index_path);
    set("range_posting_mode", options.range_posting_mode);
    set("range_posting_scan_threshold", options.range_posting_scan_threshold);
    set("range_posting_shortlist", options.range_posting_shortlist);
    set("range_posting_postfilter_shortlist_multiplier",
        options.range_posting_postfilter_shortlist_multiplier);
    set("range_posting_and_filter_mode", options.range_posting_and_filter_mode);
    set("range_label_and_policy", options.range_label_and_policy);
    set("range_posting_drop_wide_hybrid_or",
        options.range_posting_drop_wide_hybrid_or);
    set("range_graph_hybrid_or_enabled", options.range_graph_hybrid_or_enabled);
    set("range_graph_hybrid_and_enabled", options.range_graph_hybrid_and_enabled);
    set("rabitq_dual_graph_cpu_gpu_overlap",
        options.rabitq_dual_graph_cpu_gpu_overlap);
    set("rabitq_posting_schedule", options.rabitq_posting_schedule);
    set("rabitq_range_posting_segment_rows", options.rabitq_range_posting_segment_rows);
    set("rabitq_posting_stream_batch_size", options.rabitq_posting_stream_batch_size);
    set("rabitq_posting_streaming", options.rabitq_posting_streaming);
    set("rabitq_graph_persistent_workers", options.rabitq_graph_persistent_workers);
    set("rabitq_posting_persistent_workers", options.rabitq_posting_persistent_workers);
    set("range_graph_path", options.range_graph_path);
    set("range_graph_storage_mode", options.range_graph_storage_mode);
    set("range_graph_data_label_spmat_path", options.range_graph_data_label_spmat_path);
    set("range_graph_label_manifest_path", options.range_graph_label_manifest_path);
    set("range_graph_seed_sidecar_path", options.range_graph_seed_sidecar_path);
    set("range_graph_original_label_seed_sidecar_path",
        options.range_graph_original_label_seed_sidecar_path);
    set("range_graph_metadata_sidecar_path", options.range_graph_metadata_sidecar_path);
    set("range_graph_lowfreq_index_path", options.range_graph_lowfreq_index_path);
    set("range_graph_group_path", options.range_graph_group_path);
    set("data_label_spmat_path", options.data_label_spmat_path);
    set("query_vector_path", options.query_vector_path);
    set("query_label_spmat_path", options.query_label_spmat_path);
    set("query_manifest_jsonl_path", options.query_manifest_jsonl_path);
    set("query_manifest_families", options.query_manifest_families);
    set("query_manifest_subtypes", options.query_manifest_subtypes);
    set("query_manifest_max_or_terms", options.query_manifest_max_or_terms);
    set("query_manifest_max_fuzzy_terms", options.query_manifest_max_fuzzy_terms);
    set("query_manifest_fuzzy_cap_policy", options.query_manifest_fuzzy_cap_policy);
    set("query_prepare_threads", options.query_prepare_threads);
    set("query_prepare_optimized", options.query_prepare_optimized);
    set("query_manifest_min_ngram_score", options.query_manifest_min_ngram_score);
    set("query_manifest_ngram_size", options.query_manifest_ngram_size);
    set("query_manifest_operator_index_path", options.query_manifest_operator_index_path);
    set("query_manifest_operator_fill_mode", options.query_manifest_operator_fill_mode);
    set("query_manifest_operator_parity_check", options.query_manifest_operator_parity_check);
    set("or_route_order_mode", options.or_route_order_mode);
    set("or_route_sketch_sidecar_path", options.or_route_sketch_sidecar_path);
    set("or_route_accuracy_cost_model_path", options.or_route_accuracy_cost_model_path);
    set("or_route_calibration_trace_path", options.or_route_calibration_trace_path);
    set("or_route_calibration_run_id", options.or_route_calibration_run_id);
    set("or_route_sketch_k", options.or_route_sketch_k);
    set("or_route_sketch_hash_seed", options.or_route_sketch_hash_seed);
    set("or_route_allow_runtime_sketch_fallback",
        options.or_route_allow_runtime_sketch_fallback);
    set("or_route_runtime_sketch_fallback_max_rows",
        options.or_route_runtime_sketch_fallback_max_rows);
    set("or_route_verify_sketch_payload_checksum",
        options.or_route_verify_sketch_payload_checksum);
    set("query_range_path", options.query_range_path);
    set("range_label_manifest_path", options.range_label_manifest_path);
    set("seed_sidecar_path", options.seed_sidecar_path);
    set("original_label_seed_sidecar_path", options.original_label_seed_sidecar_path);
    set("metadata_sidecar_path", options.metadata_sidecar_path);
    set("seed_selection_mode", options.seed_selection_mode);
    set("pq_compressed_path", options.pq_compressed_path);
    set("pq_pivots_path", options.pq_pivots_path);
    set("pq_buffer_mode", options.pq_buffer_mode);
    set("pq_graph_transport", options.pq_graph_transport);
    set("pq_query_worker_mode", options.pq_query_worker_mode);
    set("pq_cpu_persistent_workers", options.pq_cpu_persistent_workers);
    set("pq_request_mode", options.pq_request_mode);
    set("pq_request_batch_queries", options.pq_request_batch_queries);
    set("pq_request_consumed_memory", options.pq_request_consumed_memory);
    set("pq_finite_token_transport", options.pq_finite_token_transport);
    set("pq_finite_dispatch_mode", options.pq_finite_dispatch_mode);
    set("pq_finite_partial_batch_timeout_us",
        options.pq_finite_partial_batch_timeout_us);
    set("host_reply_wait_policy", options.host_reply_wait_policy);
    set("routing_mode", options.routing_mode);
    set("group_path", options.group_path);
    set("rabitq_artifact_path", options.rabitq_artifact_path);
    set("rabitq_cuvs_artifact_path", options.rabitq_cuvs_artifact_path);
    set("pq_lowfreq_scan_mode", options.pq_lowfreq_scan_mode);
    set("pq_posting_streaming", options.pq_posting_streaming);
    set("pq_posting_stream_batch_size", options.pq_posting_stream_batch_size);
    set("posting_predicate_order", options.posting_predicate_order);
    set("predicate_mode", options.rabitq_predicate_mode);
    set("predicate_storage", options.rabitq_predicate_storage);
    set("predicate_bias_alpha", options.rabitq_predicate_bias_alpha);
    set("predicate_diagnostics", options.rabitq_predicate_diagnostics);
    set("rabitq_predicate_mode", options.rabitq_predicate_mode);
    set("rabitq_predicate_storage", options.rabitq_predicate_storage);
    set("rabitq_predicate_index_path", options.rabitq_predicate_index_path);
    set("rabitq_predicate_bias_alpha", options.rabitq_predicate_bias_alpha);
    set("rabitq_predicate_diagnostics", options.rabitq_predicate_diagnostics);
    set("rabitq_query_worker_mode", options.rabitq_query_worker_mode);
    set("rabitq_finite_dispatch_mode", options.rabitq_finite_dispatch_mode);
    set("rabitq_finite_partial_batch_timeout_us",
        options.rabitq_finite_partial_batch_timeout_us);
    set("rabitq_graph_transport", options.rabitq_graph_transport);
    set("rabitq_request_mode", options.rabitq_request_mode);
    set("rabitq_request_batch_queries", options.rabitq_request_batch_queries);
    set("rabitq_request_consumed_memory", options.rabitq_request_consumed_memory);
    set("rabitq_gpu_graph_row_cache_mib", options.rabitq_gpu_graph_row_cache_mib);
    set("output_csv_path", options.output_csv_path);
    set("config_path", options.config_path);
    set("full_filter", options.full_filter);
    set("full_gpu_fp16", options.full_gpu_fp16);
    set("overlap_graph_path", options.overlap_graph_path);
    set("overlap_manifest_path", options.overlap_manifest_path);
    set("overlap_max_bridges", options.overlap_max_bridges);

    set("lowfreq_full_filter_mode", options.lowfreq_full_filter_mode);
    set("force_rebuild", options.force_rebuild);
    set("run_analyze", options.run_analyze);
    set("row_limit", options.row_limit);
    set("cagra_build_worker_count", options.cagra_build_worker_count);
    set("stream_graph_worker_count", options.stream_graph_worker_count);
    set("stream_graph_auto_workers", options.stream_graph_auto_workers);
    set("stream_graph_memory_budget_bytes", options.stream_graph_memory_budget_bytes);
    set("stream_graph_per_worker_memory_cap_bytes", options.stream_graph_per_worker_memory_cap_bytes);
    set("stream_graph_allow_concurrent_cagra", options.stream_graph_allow_concurrent_cagra);
    set("cagra_ace_build_dir", options.cagra_ace_build_dir);
    set("cagra_ace_force_disk", options.cagra_ace_force_disk);
    set("specificity_threshold", options.specificity_threshold);
    set("prefetch_hops", options.prefetch_hops);
    set("quick_filter_bonus", options.quick_filter_bonus);
    set("bit_filter_bonus", options.bit_filter_bonus);
    set("range_filter_bonus", options.range_filter_bonus);
    set("and_dual_route_posting_ratio_threshold",
        options.and_dual_route_posting_ratio_threshold);
    set("warmup_runs", options.warmup_runs);
    set("timed_runs", options.timed_runs);
    set("process_cpu_start", options.process_cpu_start);
    set("process_cpu_count", options.process_cpu_count);
    set("host_service_threads", options.host_service_threads);
    set("host_service_cpu_start", options.host_service_cpu_start);
    set("host_service_cpu_count", options.host_service_cpu_count);
    set("host_demand_batch_size", options.host_demand_batch_size);
    set("host_demand_batch_timeout_us", options.host_demand_batch_timeout_us);

    set("search_queue_size", options.search_queue_size);
    set("search_finish_count", options.search_finish_count);
    set("search_max_steps", options.search_max_steps);
    set("search_early_stop_mode", options.search_early_stop_mode);
    set("benchmark_quick_filter_bonuses", options.benchmark_quick_filter_bonuses);
    set("benchmark_queue_sizes", options.benchmark_queue_sizes);
    set("benchmark_output", options.benchmark_output);
    set("latency_queue_sizes", options.latency_queue_sizes);
    set("latency_batch_sizes", options.latency_batch_sizes);
    set("latency_output", options.latency_output);
    set("latency_trace_output", options.latency_trace_output);
    set("latency_query_start", options.latency_query_start);
    set("latency_query_limit", options.latency_query_limit);
    set("latency_concurrency_levels", options.latency_concurrency_levels);
    set("latency_run_all_queries", options.latency_run_all_queries);
    set("latency_time_planner", options.latency_time_planner);
    set("latency_online_metadata", options.latency_online_metadata);
    set("latency_staggered", options.latency_staggered);
    set("latency_persistent_queue", options.latency_persistent_queue);
    set("latency_queue_sweep_full_warmup", options.latency_queue_sweep_full_warmup);
    set("latency_lowfreq_mode", options.latency_lowfreq_mode);
    set("degree", options.degree);
    set("topk", options.topk);
    set("highfreq_rerank_shortlist", options.highfreq_rerank_shortlist);
    set("highfreq_rerank_batch_size", options.highfreq_rerank_batch_size);
    set("highfreq_rerank_epsilon", options.highfreq_rerank_epsilon);
    set("highfreq_rerank_stable_batches", options.highfreq_rerank_stable_batches);
    set("highfreq_rerank_threads", options.highfreq_rerank_threads);
    set("highfreq_rerank_stream_batch_size", options.highfreq_rerank_stream_batch_size);
    set("lowfreq_rerank_shortlist", options.lowfreq_rerank_shortlist);
    set("lowfreq_rerank_batch_size", options.lowfreq_rerank_batch_size);
    set("lowfreq_rerank_epsilon", options.lowfreq_rerank_epsilon);
    set("lowfreq_rerank_stable_batches", options.lowfreq_rerank_stable_batches);
    set("enable_lowfreq_rerank", options.enable_lowfreq_rerank);
    set("dim", options.dim);
    set("pq_dim", options.pq_dim);
    set("rabitq_num_anchors", options.rabitq_num_anchors);
    set("rabitq_bits_per_dim", options.rabitq_bits_per_dim);
    set("rabitq_enable_highfreq_rerank", options.rabitq_enable_highfreq_rerank);
    set("rabitq_enable_lowfreq_rerank", options.rabitq_enable_lowfreq_rerank);
    set("rabitq_lowfreq_gpu_posting", options.rabitq_lowfreq_gpu_posting);
    set("rabitq_use_lut", options.rabitq_use_lut);
    set("rabitq_query_rotation", options.rabitq_query_rotation);
    set("rabitq_centroid_distances", options.rabitq_centroid_distances);
    set("rabitq_use_int8_score", options.rabitq_use_int8_score);

    config["pq"] = nlohmann::json{
        {"compressed_path", options.pq_compressed_path},
        {"pivots_path", options.pq_pivots_path},
        {"buffer_mode", options.pq_buffer_mode},
        {"graph_transport", options.pq_graph_transport},
        {"query_worker_mode", options.pq_query_worker_mode},
        {"cpu_persistent_workers", options.pq_cpu_persistent_workers},
        {"request_mode", options.pq_request_mode},
        {"request_batch_queries", options.pq_request_batch_queries},
        {"request_consumed_memory", options.pq_request_consumed_memory},
        {"finite_token_transport", options.pq_finite_token_transport},
        {"finite_dispatch_mode", options.pq_finite_dispatch_mode},
        {"finite_partial_batch_timeout_us", options.pq_finite_partial_batch_timeout_us},
        {"host_reply_wait_policy", options.host_reply_wait_policy},
        {"pq_dim", options.pq_dim},
        {"lowfreq_scan_mode", options.pq_lowfreq_scan_mode},
        {"posting_streaming", options.pq_posting_streaming},
        {"posting_stream_batch_size", options.pq_posting_stream_batch_size},
        {"routing_mode", options.routing_mode},
    };
    config["rabitq"] = nlohmann::json{
        {"artifact_path", options.rabitq_artifact_path},
        {"cuvs_artifact_path", options.rabitq_cuvs_artifact_path},
        {"num_anchors", options.rabitq_num_anchors},
        {"bits_per_dim", options.rabitq_bits_per_dim},
        {"enable_highfreq_rerank", options.rabitq_enable_highfreq_rerank},
        {"enable_lowfreq_rerank", options.rabitq_enable_lowfreq_rerank},
        {"lowfreq_gpu_posting", options.rabitq_lowfreq_gpu_posting},
        {"lowfreq_rerank_merge_routes", options.lowfreq_rerank_merge_routes},
        {"use_lut", options.rabitq_use_lut},
        {"query_rotation", options.rabitq_query_rotation},
        {"centroid_distances", options.rabitq_centroid_distances},
        {"use_int8_score", options.rabitq_use_int8_score},
        {"predicate_mode", options.rabitq_predicate_mode},
        {"predicate_storage", options.rabitq_predicate_storage},
        {"predicate_index_path", options.rabitq_predicate_index_path},
        {"predicate_bias_alpha", options.rabitq_predicate_bias_alpha},
        {"predicate_diagnostics", options.rabitq_predicate_diagnostics},
        {"query_worker_mode", options.rabitq_query_worker_mode},
        {"finite_dispatch_mode", options.rabitq_finite_dispatch_mode},
        {"finite_partial_batch_timeout_us", options.rabitq_finite_partial_batch_timeout_us},
        {"graph_transport", options.rabitq_graph_transport},
        {"dual_graph_cpu_gpu_overlap", options.rabitq_dual_graph_cpu_gpu_overlap},
        {"posting_schedule", options.rabitq_posting_schedule},
        {"range_posting_segment_rows", options.rabitq_range_posting_segment_rows},
        {"posting_stream_batch_size", options.rabitq_posting_stream_batch_size},
        {"posting_streaming", options.rabitq_posting_streaming},
        {"graph_persistent_workers", options.rabitq_graph_persistent_workers},
        {"posting_persistent_workers", options.rabitq_posting_persistent_workers},
        {"request_mode", options.rabitq_request_mode},
        {"request_batch_queries", options.rabitq_request_batch_queries},
        {"request_consumed_memory", options.rabitq_request_consumed_memory},
        {"gpu_graph_row_cache_mib", options.rabitq_gpu_graph_row_cache_mib},
    };
    return config;
}

static void write_latency_queue_sweep_audit_files(const AppOptions& options,
                                                  const std::string& command_line)
{
    const std::string config_path =
        replace_extension_with_suffix(options.latency_output, ".effective-config", ".json");
    const std::string command_path =
        replace_extension_with_suffix(options.latency_output, ".command", ".txt");

    std::ofstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("failed to open latency effective config output: " + config_path);
    }
    config_file << make_effective_config_json(options).dump(2) << '\n';

    std::ofstream command_file(command_path);
    if (!command_file.is_open()) {
        throw std::runtime_error("failed to open latency command output: " + command_path);
    }
    command_file << "cwd=" << std::filesystem::current_path().string() << '\n'
                 << "command=" << command_line << '\n';

    std::cout << "Latency queue sweep effective config written to " << config_path << "\n";
    std::cout << "Latency queue sweep command written to " << command_path << "\n";
}

static void append_latency_queue_sweep_rows(const std::string& combined_csv_path,
                                            const std::string& point_csv_path,
                                            int queue_size,
                                            bool write_header)
{
    std::ifstream input(point_csv_path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open latency sweep point CSV: " + point_csv_path);
    }

    std::string header;
    if (!std::getline(input, header)) {
        throw std::runtime_error("latency sweep point CSV is empty: " + point_csv_path);
    }

    std::ofstream output;
    if (write_header) {
        output.open(combined_csv_path, std::ios::out | std::ios::trunc);
    } else {
        output.open(combined_csv_path, std::ios::out | std::ios::app);
    }
    if (!output.is_open()) {
        throw std::runtime_error("failed to open latency sweep combined CSV: " + combined_csv_path);
    }
    if (write_header) {
        output << "queue_size," << header << '\n';
    }

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        output << queue_size << ',' << line << '\n';
    }
}

static int run_latency_queue_sweep(const AppOptions& options, const std::string& command_line)
{
    configure_benchmark_runtime_environment();
    const std::vector<int> queue_sizes = parse_int_list(options.latency_queue_sizes);
    if (queue_sizes.empty()) {
        throw std::runtime_error("latency-queue-sizes must not be empty");
    }
    if (options.backend != "pq" && options.backend != "rabitq") {
        throw std::runtime_error("latency queue sweep currently supports --backend=pq|rabitq");
    }

    write_latency_queue_sweep_audit_files(options, command_line);

    bool combined_header_written = false;
    for (size_t point_index = 0; point_index < queue_sizes.size(); ++point_index) {
        const int queue_size = queue_sizes[point_index];
        AppOptions point_options = options;
        point_options.search_queue_size = queue_size;
        if (options.latency_queue_sweep_full_warmup) {
            point_options.warmup_runs = point_index == 0 ? 1 : 0;
        }
        point_options.latency_output =
            append_suffix_before_extension(options.latency_output, ".q" + std::to_string(queue_size));
        if (!options.latency_trace_output.empty()) {
            point_options.latency_trace_output = append_suffix_before_extension(
                options.latency_trace_output,
                ".q" + std::to_string(queue_size));
        }

        const int point_rc = options.backend == "pq"
            ? run_pq_latency(point_options)
            : run_rabitq_latency(point_options);
        if (point_rc != 0) {
            return point_rc;
        }

        append_latency_queue_sweep_rows(options.latency_output,
                                        point_options.latency_output,
                                        queue_size,
                                        !combined_header_written);
        combined_header_written = true;
        std::error_code remove_error;
        std::filesystem::remove(point_options.latency_output, remove_error);
        std::cout << "[latency-sweep] queue_size=" << queue_size
                  << " point_output=" << point_options.latency_output << "\n";
    }

    std::cout << "Latency queue sweep CSV written to " << options.latency_output << "\n";
    return 0;
}

static int run_benchmark_sweep(const AppOptions& options)
{
    configure_benchmark_runtime_environment();
    const std::vector<float> quick_filter_bonuses = options.benchmark_quick_filter_bonuses.empty()
        ? std::vector<float>{options.quick_filter_bonus}
        : parse_float_list(options.benchmark_quick_filter_bonuses);
    const std::vector<int> queue_sizes = parse_int_list(options.benchmark_queue_sizes);
    if (expansion_trace::enabled() &&
        (quick_filter_bonuses.size() != 1 || queue_sizes.size() != 1))
        throw std::runtime_error("expansion trace benchmark requires exactly one queue and bonus point");
    if (options.backend == "pq") {
        backend_pq::run_benchmark_sweep(make_pq_search_options(options),
                                        quick_filter_bonuses,
                                        queue_sizes,
                                        options.benchmark_output);
    } else if (options.backend == "rabitq") {
        backend_rabitq::run_benchmark_sweep(make_rabitq_search_options(options),
                                            quick_filter_bonuses,
                                            queue_sizes,
                                            options.benchmark_output);
    } else {
        throw std::runtime_error("benchmark mode currently supports --backend=pq|rabitq");
    }
    std::cout << "Benchmark sweep CSV written to " << options.benchmark_output << "\n";
    return 0;
}

static backend_rabitq::RabitqSearchOptions make_rabitq_search_options(const AppOptions& options)
{
    backend_rabitq::RabitqSearchOptions search_options;
    search_options.cpu_rerank_schedule = cpu_rerank_schedule::parse(options.cpu_rerank_schedule);
    search_options.raw_navigation_scores = uses_raw_navigation_scores(options);
    search_options.artifact_path = options.rabitq_artifact_path;
    search_options.host_reply_wait_policy = options.host_reply_wait_policy;
    search_options.routing_mode = options.routing_mode;
    search_options.posting_predicate_order = options.posting_predicate_order;
    search_options.rabitq_predicate_mode = options.rabitq_predicate_mode;
    search_options.rabitq_predicate_storage = options.rabitq_predicate_storage;
    search_options.rabitq_predicate_index_path = options.rabitq_predicate_index_path;
    search_options.rabitq_predicate_bias_alpha = options.rabitq_predicate_bias_alpha;
    search_options.rabitq_predicate_diagnostics = options.rabitq_predicate_diagnostics;
    search_options.query_worker_mode = options.rabitq_query_worker_mode;
    search_options.finite_dispatch_mode = options.rabitq_finite_dispatch_mode;
    search_options.finite_partial_batch_timeout_us =
        options.rabitq_finite_partial_batch_timeout_us;
    search_options.request_mode = options.rabitq_request_mode;
    search_options.request_batch_queries = options.rabitq_request_batch_queries;
    search_options.request_consumed_memory = options.rabitq_request_consumed_memory;
    search_options.gpu_graph_row_cache_mib = options.rabitq_gpu_graph_row_cache_mib;
    search_options.graph_path = options.graph_path;
    search_options.graph_storage_mode = options.graph_storage_mode;
    search_options.rerank_vector_storage_mode = options.rerank_vector_storage_mode;
    search_options.rerank_vector_artifact_path = options.rerank_vector_artifact_path;
    search_options.lowfreq_index_path = options.lowfreq_index_path;
    search_options.data_label_spmat_path = options.data_label_spmat_path;
    search_options.query_label_spmat_path = options.query_label_spmat_path;
    search_options.query_manifest_jsonl_path = options.query_manifest_jsonl_path;
    search_options.query_manifest_families = options.query_manifest_families;
    search_options.query_manifest_subtypes = options.query_manifest_subtypes;
    search_options.query_manifest_max_or_terms = options.query_manifest_max_or_terms;
    search_options.query_manifest_max_fuzzy_terms = options.query_manifest_max_fuzzy_terms;
    search_options.query_manifest_min_ngram_score = options.query_manifest_min_ngram_score;
    search_options.query_manifest_ngram_size = options.query_manifest_ngram_size;
    search_options.base_vector_path = options.base_vector_path;
    search_options.base_range_fbin_path = options.base_range_fbin_path;
    search_options.range_posting_index_path = options.range_posting_index_path;
    search_options.range_posting_mode = options.range_posting_mode;
    search_options.range_posting_scan_threshold = options.range_posting_scan_threshold;
    search_options.range_posting_shortlist = options.range_posting_shortlist;
    search_options.range_posting_postfilter_shortlist_multiplier =
        options.range_posting_postfilter_shortlist_multiplier;
    search_options.range_posting_and_filter_mode =
        options.range_posting_and_filter_mode;
    search_options.range_label_and_policy = options.range_label_and_policy;
    search_options.range_posting_drop_wide_hybrid_or =
        options.range_posting_drop_wide_hybrid_or;
    search_options.range_graph_hybrid_or_enabled = options.range_graph_hybrid_or_enabled;
    search_options.range_graph_hybrid_and_enabled = options.range_graph_hybrid_and_enabled;
    search_options.dual_graph_cpu_gpu_overlap =
        options.rabitq_dual_graph_cpu_gpu_overlap;
    search_options.posting_schedule = options.rabitq_posting_schedule;
    search_options.range_posting_segment_rows = options.rabitq_range_posting_segment_rows;
    search_options.posting_stream_batch_size = options.rabitq_posting_stream_batch_size;
    search_options.posting_streaming = options.rabitq_posting_streaming;
    search_options.graph_persistent_workers = options.rabitq_graph_persistent_workers;
    search_options.posting_persistent_workers = options.rabitq_posting_persistent_workers;
    search_options.range_graph_path = options.range_graph_path;
    search_options.range_graph_storage_mode = options.range_graph_storage_mode;
    search_options.range_graph_data_label_spmat_path =
        options.range_graph_data_label_spmat_path;
    search_options.range_graph_label_manifest_path =
        options.range_graph_label_manifest_path;
    search_options.range_graph_seed_sidecar_path = options.range_graph_seed_sidecar_path;
    search_options.range_graph_original_label_seed_sidecar_path =
        options.range_graph_original_label_seed_sidecar_path;
    search_options.range_graph_metadata_sidecar_path =
        options.range_graph_metadata_sidecar_path;
    search_options.range_graph_lowfreq_index_path =
        options.range_graph_lowfreq_index_path;
    search_options.range_graph_group_path = options.range_graph_group_path;
    search_options.query_vector_path = options.query_vector_path;
    search_options.query_range_path = options.query_range_path;
    search_options.range_label_manifest_path = options.range_label_manifest_path;
    search_options.seed_sidecar_path = options.seed_sidecar_path;
    search_options.original_label_seed_sidecar_path = options.original_label_seed_sidecar_path;
    search_options.metadata_sidecar_path = options.metadata_sidecar_path;
    search_options.seed_selection_mode = options.seed_selection_mode;
    search_options.group_path = options.group_path;
    search_options.output_csv_path = options.output_csv_path;
    search_options.full_filter = options.full_filter;
    search_options.dataset_cutoff = options.row_limit;
    search_options.specificity_threshold = options.specificity_threshold;
    search_options.dim = options.dim;
    search_options.degree = options.degree;
    search_options.topk = options.topk;
    search_options.lowfreq_rerank_shortlist = options.lowfreq_rerank_shortlist;
    search_options.lowfreq_rerank_batch_size = options.lowfreq_rerank_batch_size;
    search_options.lowfreq_rerank_epsilon = options.lowfreq_rerank_epsilon;
    search_options.lowfreq_rerank_stable_batches = options.lowfreq_rerank_stable_batches;
    search_options.lowfreq_full_filter_mode = options.lowfreq_full_filter_mode;
    search_options.enable_lowfreq_gpu_posting = options.rabitq_lowfreq_gpu_posting;
    search_options.prefetch_hops = options.prefetch_hops;
    search_options.quick_filter_bonus = options.quick_filter_bonus;
    search_options.bit_filter_bonus = options.bit_filter_bonus;
    search_options.range_filter_bonus = options.range_filter_bonus;
    search_options.and_dual_route_posting_ratio_threshold =
        options.and_dual_route_posting_ratio_threshold;
    search_options.warmup_runs = options.warmup_runs;
    search_options.timed_runs = options.timed_runs;
    search_options.latency_batch_sizes = parse_int_list(options.latency_batch_sizes);
    search_options.latency_output_path = options.latency_output;
    search_options.latency_trace_output_path = options.latency_trace_output;
    search_options.latency_query_start = options.latency_query_start;
    search_options.latency_query_limit = options.latency_query_limit;
    search_options.latency_concurrency_levels = options.latency_concurrency_levels.empty()
        ? std::vector<int>()
        : parse_int_list(options.latency_concurrency_levels);
    search_options.latency_run_all_queries = options.latency_run_all_queries;
    search_options.latency_time_planner = options.latency_time_planner;
    search_options.latency_online_metadata = options.latency_online_metadata;
    search_options.latency_staggered = options.latency_staggered;
    search_options.latency_persistent_queue = options.latency_persistent_queue;
    search_options.latency_lowfreq_mode = options.latency_lowfreq_mode;
    search_options.process_cpu_start = options.process_cpu_start;
    search_options.process_cpu_count = options.process_cpu_count;
    search_options.host_service_threads = options.host_service_threads;
    search_options.host_service_cpu_start = options.host_service_cpu_start;
    search_options.host_service_cpu_count = options.host_service_cpu_count;
    search_options.host_demand_batch_size = options.host_demand_batch_size;
    search_options.host_demand_batch_timeout_us = options.host_demand_batch_timeout_us;
    search_options.search_queue_size = options.search_queue_size;
    search_options.search_finish_count = options.search_finish_count;
    search_options.search_max_steps = options.search_max_steps;
    search_options.search_early_stop_mode = options.search_early_stop_mode;
    search_options.graph_transport = options.rabitq_graph_transport;
    search_options.skip_filtered_vector_payload = options.skip_filtered_vector_payload;
    search_options.per_step_full_filter = options.per_step_full_filter;
    search_options.highfreq_rerank_mode = options.highfreq_rerank_mode;
    search_options.highfreq_rerank_algorithm = options.highfreq_rerank_algorithm;
    search_options.highfreq_rerank_streaming = options.highfreq_rerank_streaming;
    search_options.rerank_distance_reuse = options.rerank_distance_reuse;
    search_options.incremental_rerank_shortlist = options.rabitq_incremental_rerank_shortlist;
    search_options.highfreq_rerank_merge_routes = options.highfreq_rerank_merge_routes;
    search_options.lowfreq_rerank_merge_routes = options.lowfreq_rerank_merge_routes;
    search_options.highfreq_rerank_shortlist = options.highfreq_rerank_shortlist;
    search_options.highfreq_rerank_batch_size = options.highfreq_rerank_batch_size;
    search_options.highfreq_rerank_epsilon = options.highfreq_rerank_epsilon;
    search_options.highfreq_rerank_stable_batches = options.highfreq_rerank_stable_batches;
    search_options.highfreq_rerank_threads = options.highfreq_rerank_threads;
    search_options.highfreq_rerank_stream_batch_size = options.highfreq_rerank_stream_batch_size;
    search_options.enable_highfreq_rerank = options.rabitq_enable_highfreq_rerank;
    search_options.enable_lowfreq_rerank = options.enable_lowfreq_rerank;
    search_options.use_lut = options.rabitq_use_lut;
    if (options.rabitq_query_rotation != "legacy" && options.rabitq_query_rotation != "cublas_fp32") {
        throw std::invalid_argument("rabitq_query_rotation must be legacy or cublas_fp32");
    }
    if (options.rabitq_centroid_distances != "legacy" && options.rabitq_centroid_distances != "cublas_fp32") {
        throw std::invalid_argument("rabitq_centroid_distances must be legacy or cublas_fp32");
    }
    search_options.query_rotation = options.rabitq_query_rotation;
    search_options.centroid_distances = options.rabitq_centroid_distances;
    search_options.use_int8_score = options.rabitq_use_int8_score;
    search_options.use_int8_score_auto = !options.rabitq_use_int8_score_explicit;
    return search_options;
}

static int run_native_rabitq_search(const AppOptions& options)
{
    return backend_rabitq::run_rabitq_search(make_rabitq_search_options(options));
}

static int run_build_graph(const AppOptions& options)
{
    graph_builder::GraphBuildOptions build_options;
    build_options.graph_path = options.graph_path;
    build_options.graph_file_path = options.graph_file_path;
    build_options.lowfreq_index_path = options.lowfreq_index_path;
    build_options.seed_sidecar_path = options.seed_sidecar_path;
    build_options.original_label_seed_sidecar_path = options.original_label_seed_sidecar_path;
    build_options.metadata_sidecar_path = options.metadata_sidecar_path;
    build_options.base_vector_path = options.base_vector_path;
    build_options.data_label_spmat_path = options.data_label_spmat_path;
    build_options.group_path = options.group_path;
    build_options.builder = options.graph_builder;
    build_options.force_rebuild = options.force_rebuild;
    build_options.run_analyze = options.run_analyze;
    build_options.row_limit = options.row_limit;
    build_options.cagra_build_worker_count = options.cagra_build_worker_count;
    build_options.stream_worker_count = options.stream_graph_worker_count;
    build_options.stream_auto_workers = options.stream_graph_auto_workers;
    build_options.stream_memory_budget_bytes = options.stream_graph_memory_budget_bytes;
    build_options.stream_per_worker_memory_cap_bytes =
        options.stream_graph_per_worker_memory_cap_bytes;
    build_options.stream_allow_concurrent_cagra = options.stream_graph_allow_concurrent_cagra;
    build_options.cagra_ace_build_dir = options.cagra_ace_build_dir;
    build_options.cagra_ace_force_disk = options.cagra_ace_force_disk;
    build_options.specificity_threshold = options.specificity_threshold;
    build_options.dim = options.dim;
    build_options.degree = options.degree;
    return graph_builder::run_graph_builder(build_options);
}

static int run_build_rabitq(const AppOptions& options)
{
    if (options.rabitq_cuvs_artifact_path.empty()) {
        throw std::runtime_error("build-rabitq requires --rabitq-cuvs-artifact-path");
    }
    backend_rabitq::RabitqBuildOptions build_options;
    build_options.dim = options.dim;
    build_options.num_anchors = options.rabitq_num_anchors;
    build_options.bits_per_dim = options.rabitq_bits_per_dim;
    build_options.artifact_path = options.rabitq_artifact_path;
    build_options.cuvs_artifact_path = options.rabitq_cuvs_artifact_path;

    auto index = backend_rabitq::import_cuvs_rabitq_index(build_options.cuvs_artifact_path);
    backend_rabitq::save_rabitq_index(options.rabitq_artifact_path, index);
    std::cout << "Imported cuVS RabitQ artifact from " << options.rabitq_cuvs_artifact_path
              << " and saved FlashANNS artifact to " << options.rabitq_artifact_path << "\n";
    return 0;
}

static int run_build_rerank_int8(const AppOptions& options)
{
    const uint32_t row_limit = options.row_limit_explicit ? options.row_limit : 0;
    const Int8RerankBuildStats stats = build_int8_rerank_artifact(
        options.base_vector_path,
        options.rerank_vector_artifact_path,
        row_limit,
        options.force_rebuild);
    std::cout << "Built INT8 rerank artifact " << options.rerank_vector_artifact_path
              << " rows=" << stats.rows
              << " cols=" << stats.cols
              << " source_bytes=" << stats.source_bytes
              << " artifact_bytes=" << stats.artifact_bytes << "\n";
    return 0;
}

}

int main(int argc, char** argv)
{
    try {
        if (argc == 1) {
            return go_fann::print_usage(argv[0]);
        }

        go_fann::AppOptions options;
        if (auto default_config_path = go_fann::default_config_path_if_exists()) {
            options.config_path = *default_config_path;
            go_fann::apply_config_values(go_fann::load_config_file(*default_config_path), options);
        }

        const auto cli_overrides = go_fann::parse_app_options(argc, argv);
        if (cli_overrides.help) {
            return go_fann::print_usage(argv[0]);
        }
        if (cli_overrides.config_path.has_value()) {
            options = go_fann::AppOptions{};
            options.config_path = *cli_overrides.config_path;
            go_fann::apply_config_values(go_fann::load_config_file(*cli_overrides.config_path), options);
        }
        go_fann::apply_cli_overrides(cli_overrides, options);
        go_fann::apply_legacy_rabitq_highfreq_rerank_mode(options);
        go_fann::resolve_cpu_rerank_schedule(options);
        go_fann::apply_rabitq_optimized_defaults(options);
        go_fann::resolve_predicate_defaults(options);
        go_fann::resolve_posting_defaults(options);
        go_fann::resolve_navigation_score_policy(options);
        go_fann::validate_app_options(options);
        configure_query_manifest_operator_rewrite(options.query_manifest_operator_index_path,
                                                  options.query_manifest_operator_parity_check,
                                                  options.query_manifest_operator_fill_mode);
        OrRoutePlannerConfig planner_config;
        planner_config.order_mode = options.or_route_order_mode;
        planner_config.fuzzy_cap_policy = options.query_manifest_fuzzy_cap_policy;
        configure_query_preparation({options.query_prepare_threads, options.query_prepare_optimized});
        planner_config.sketch_sidecar_path = options.or_route_sketch_sidecar_path;
        planner_config.accuracy_cost_model_path = options.or_route_accuracy_cost_model_path;
        planner_config.calibration_trace_path = options.or_route_calibration_trace_path;
        planner_config.calibration_run_id = options.or_route_calibration_run_id;
        planner_config.sketch_k = options.or_route_sketch_k;
        planner_config.sketch_hash_seed = options.or_route_sketch_hash_seed;
        planner_config.allow_runtime_sketch_fallback =
            options.or_route_allow_runtime_sketch_fallback;
        planner_config.runtime_sketch_fallback_max_rows =
            options.or_route_runtime_sketch_fallback_max_rows;
        planner_config.verify_sketch_payload_checksum =
            options.or_route_verify_sketch_payload_checksum;
        configure_or_route_planner(planner_config);
        set_full_filter_enabled(options.full_filter);
        go_fann::maybe_bind_process_cpus(options);
        const std::string command_line = go_fann::make_command_line(argc, argv);

        if (options.mode == "build-graph") {
            return go_fann::run_build_graph(options);
        }
        if (options.mode == "build-rabitq") {
            return go_fann::run_build_rabitq(options);
        }
        if (options.mode == "build-rerank-int8") {
            return go_fann::run_build_rerank_int8(options);
        }
        if (options.mode == "benchmark") {
            return go_fann::run_benchmark_sweep(options);
        }
        if (options.mode == "latency") {
            if (options.backend == "full_gpu") {
                return go_fann::run_full_gpu_latency(options);
            }
            if (!options.latency_queue_sizes.empty()) {
                return go_fann::run_latency_queue_sweep(options, command_line);
            }
            if (options.backend == "rabitq") {
                return go_fann::run_rabitq_latency(options);
            }
            if (options.backend == "pq") {
                return go_fann::run_pq_latency(options);
            }
            std::cerr << "Latency backend '" << options.backend << "' is not wired yet.\n";
            return 2;
        }
        if (options.mode != "search") {
            std::cerr << "Mode '" << options.mode << "' is not wired yet in src/main.cu.\n";
            return 2;
        }
        if (options.backend == "pq") {
            return go_fann::run_native_pq_search(options);
        }
        if (options.backend == "rabitq") {
            return go_fann::run_native_rabitq_search(options);
        }
        if (options.backend == "full_gpu") {
            return go_fann::run_full_gpu_search(options);
        }

        std::cerr << "Backend '" << options.backend << "' is not wired yet in src/main.cu.\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "GO-FANN failed: " << e.what() << '\n';
        return 1;
    }
}
