#pragma once

#include "execution/cpu_rerank_schedule.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "app/runtime_defaults.h"

struct DenseUint8Matrix;

namespace go_fann::backend_rabitq {

struct RabitqArtifactHeader {
    uint32_t version = 2;
    uint32_t dim = 128;
    uint32_t num_anchors = 16;
    uint32_t code_bytes = 16;
    uint64_t num_points = 0;
};

struct RabitqCodecSpec {
    uint32_t padded_dim = 128;
    uint32_t bits_per_dim = 1;
    uint32_t short_code_words = 4;
    uint32_t long_code_bytes = 0;
    uint32_t layout_flags = 0;
    uint32_t short_factor_count = 3;
    uint32_t ex_factor_count = 2;
};

struct RabitqBuildOptions {
    uint32_t dim = 128;
    uint32_t num_anchors = 16;
    uint32_t bits_per_dim = 1;
    std::string artifact_path;
    std::string cuvs_artifact_path;
};

struct RabitqSearchOptions {
    cpu_rerank_schedule::Mode cpu_rerank_schedule = cpu_rerank_schedule::Mode::Native;
    // Internal application policy: native in-memory search uses raw base scores.
    bool raw_navigation_scores = false;
    std::string posting_predicate_order = "filter_first";
    // Direct backend APIs use concrete modes; main resolves the automatic GPU policy.
    std::string rabitq_predicate_mode = "cpu";
    std::string rabitq_predicate_storage = "csr";
    std::string rabitq_predicate_index_path;
    // Primary original-label domain retained when options are adapted to the
    // secondary range graph. Internal plumbing, not a separate user option.
    std::string rabitq_predicate_label_source_path;
    double rabitq_predicate_bias_alpha = 0.0;
    bool rabitq_predicate_diagnostics = false;
    std::string artifact_path;
    std::string graph_path;
    std::string graph_storage_mode = go_fann::defaults::graph_storage_mode;
    std::string rerank_vector_storage_mode = "default";
    std::string rerank_vector_artifact_path;
    std::string highfreq_rerank_mode = std::string(go_fann::defaults::default_highfreq_rerank_mode);
    std::string highfreq_rerank_algorithm = std::string(go_fann::defaults::default_highfreq_rerank_algorithm);
    bool highfreq_rerank_streaming = go_fann::defaults::default_highfreq_rerank_streaming;
    bool highfreq_rerank_merge_routes = true;
    bool lowfreq_rerank_merge_routes =
        go_fann::defaults::default_lowfreq_rerank_merge_routes;
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
    std::string range_posting_index_path;
    std::string range_posting_mode = "off";
    uint64_t range_posting_scan_threshold = 0;
    int range_posting_shortlist = go_fann::defaults::default_range_posting_shortlist;
    int range_posting_postfilter_shortlist_multiplier =
        go_fann::defaults::default_range_posting_postfilter_shortlist_multiplier;
    std::string range_posting_and_filter_mode = "auto";
    std::string range_label_and_policy = "auto";
    bool range_posting_drop_wide_hybrid_or = false;
    bool range_graph_hybrid_or_enabled = false;
    bool range_graph_hybrid_and_enabled = false;
    bool dual_graph_cpu_gpu_overlap = false;
    std::string posting_schedule = "legacy";
    uint32_t range_posting_segment_rows = 0;
    bool posting_streaming = false;
    bool rerank_distance_reuse = false;
    int incremental_rerank_shortlist = 0;
    bool graph_persistent_workers = false;
    bool posting_persistent_workers = false;
    uint32_t posting_stream_batch_size = 1024;
    std::string range_graph_path;
    std::string range_graph_storage_mode = "memory";
    std::string range_graph_data_label_spmat_path;
    std::string range_graph_label_manifest_path;
    std::string range_graph_seed_sidecar_path;
    std::string range_graph_original_label_seed_sidecar_path;
    std::string range_graph_metadata_sidecar_path;
    std::string range_graph_lowfreq_index_path;
    std::string range_graph_group_path;
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
    int degree = 64;
    int topk = 10;
    int highfreq_rerank_shortlist = go_fann::defaults::default_highfreq_rerank_shortlist;
    int highfreq_rerank_batch_size = go_fann::defaults::default_lowfreq_rerank_batch_size;
    float highfreq_rerank_epsilon = go_fann::defaults::default_lowfreq_rerank_epsilon;
    int highfreq_rerank_stable_batches = go_fann::defaults::default_lowfreq_rerank_stable_batches;
    int highfreq_rerank_threads = go_fann::defaults::default_highfreq_rerank_threads;
    int highfreq_rerank_stream_batch_size =
        go_fann::defaults::default_highfreq_rerank_stream_batch_size;
    int lowfreq_rerank_shortlist = go_fann::defaults::default_lowfreq_rerank_shortlist;
    int lowfreq_rerank_batch_size = go_fann::defaults::default_lowfreq_rerank_batch_size;
    float lowfreq_rerank_epsilon = go_fann::defaults::default_lowfreq_rerank_epsilon;
    int lowfreq_rerank_stable_batches = go_fann::defaults::default_lowfreq_rerank_stable_batches;
    std::string lowfreq_full_filter_mode = go_fann::defaults::default_lowfreq_full_filter_mode;
    bool enable_lowfreq_gpu_posting = false;
    int prefetch_hops = 3;
    float quick_filter_bonus = 0.95f;
    float bit_filter_bonus = 0.95f;
    float range_filter_bonus = 0.95f;
    double and_dual_route_posting_ratio_threshold =
        go_fann::defaults::default_and_dual_route_posting_ratio_threshold;
    int warmup_runs = 0;
    int timed_runs = 1;
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
    bool skip_filtered_vector_payload = true;
    bool per_step_full_filter = true;
    int search_queue_size = 50;
    int search_finish_count = 1;
    int search_max_steps = 0;
    std::string search_early_stop_mode = go_fann::defaults::default_search_early_stop_mode;
    std::string host_reply_wait_policy = "fixed";
    std::string routing_mode = "greedy_min_posting";
    // Stage 2 finite scheduling is intentionally opt-in.  The coupled kernel
    // remains the default until the finite path has equivalent validation.
    std::string query_worker_mode = "coupled";
    std::string finite_dispatch_mode = "batch";
    int finite_partial_batch_timeout_us = 50;
    std::string graph_transport = "mailbox";
    std::string request_mode = "scan";
    size_t request_batch_queries = 128;
    std::string request_consumed_memory = "host";
    size_t gpu_graph_row_cache_mib = 0;
    bool enable_highfreq_rerank = true;
    bool enable_lowfreq_rerank = true;
    bool use_lut = false;
    std::string query_rotation = "cublas_fp32";
    std::string centroid_distances = "cublas_fp32";
    bool use_int8_score = false;
    bool use_int8_score_auto = false;
};

struct RabitqQueryState {
    std::vector<float> values;
};

struct RabitqHostIndex {
    RabitqArtifactHeader header;
    RabitqCodecSpec codec;
    std::vector<float> anchors;
    std::vector<float> rotation;
    std::vector<float> query_scaling_factors;
    std::vector<uint32_t> anchor_ids;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> long_codes;
    std::vector<float> factors;
    std::vector<float> ex_factors;
};

RabitqHostIndex import_cuvs_rabitq_index(const std::string& cuvs_artifact_path);
RabitqHostIndex build_rabitq_index(const DenseUint8Matrix& base_vectors,
                                   const RabitqBuildOptions& options);
RabitqHostIndex build_rabitq_index_from_path(const std::string& base_vector_path,
                                             uint32_t row_limit,
                                             const RabitqBuildOptions& options);
void save_rabitq_index(const std::string& path, const RabitqHostIndex& index);
RabitqHostIndex load_rabitq_index(const std::string& path);
RabitqQueryState prepare_rabitq_query(const uint8_t* query, uint32_t dim);
float score_rabitq_point(const RabitqHostIndex& index,
                         const RabitqQueryState& query,
                         uint32_t point_id);
std::vector<float> decode_rabitq_point(const RabitqHostIndex& index, uint32_t point_id);
int run_rabitq_search(const RabitqSearchOptions& options);
int run_rabitq_latency_benchmark(const RabitqSearchOptions& options);
int run_benchmark_sweep(const RabitqSearchOptions& base_options,
                        const std::vector<float>& quick_filter_bonuses,
                        const std::vector<int>& queue_sizes,
                        const std::string& output_csv_path);

}
