#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "type_def.h"
#include "data/dataloader.h"
#include "app/runtime_defaults.h"

namespace go_fann::backend_full_gpu {

struct FullGpuSearchOptions {
    std::string graph_path;
    std::string graph_storage_mode = defaults::graph_storage_mode;
    std::string lowfreq_index_path;
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
    int range_posting_shortlist = defaults::default_range_posting_shortlist;
    std::string range_posting_and_filter_mode = "auto";
    std::string range_label_and_policy = "auto";
    bool range_posting_drop_wide_hybrid_or = false;
    bool range_graph_hybrid_or_enabled = false;
    bool range_graph_hybrid_and_enabled = false;
    std::string query_vector_path;
    std::string query_range_path;
    std::string range_label_manifest_path;
    std::string seed_sidecar_path;
    std::string original_label_seed_sidecar_path;
    std::string metadata_sidecar_path;
    std::string seed_selection_mode = "group";
    std::string group_path;
    std::string output_csv_path = "results_100.csv";
    std::string routing_mode = "greedy_min_posting";
    std::string search_early_stop_mode = defaults::default_search_early_stop_mode;
    uint32_t dataset_cutoff = 0;
    int specificity_threshold = 2000;
    int dim = defaults::default_dim;
    int degree = defaults::default_degree;
    int topk = defaults::default_topk;
    int search_queue_size = 50;
    int search_finish_count = 1;
    int search_max_steps = 0;
    int warmup_runs = 0;
    int timed_runs = 1;
    double and_dual_route_posting_ratio_threshold = defaults::default_and_dual_route_posting_ratio_threshold;
    std::vector<int> latency_batch_sizes;
    std::string latency_output_path = "latency_sweep.csv";
    std::string latency_trace_output_path;
    int latency_query_start = 0;
    int latency_query_limit = 0;
    std::vector<int> latency_concurrency_levels;
    bool latency_run_all_queries = false;
    bool latency_time_planner = false;
    bool latency_online_metadata = false;
    bool latency_staggered = false;
    bool latency_persistent_queue = false;
    bool print_summary = true;
    // Store float vectors as FP16 on device while accumulating L2 in FP32.
    bool use_fp16 = false;
    // Optional query-level traversal across primary and finest-boundary graphs.
    std::string overlap_graph_path;
    std::string overlap_manifest_path;
    uint32_t overlap_max_bridges = 1;

};

int run_full_gpu_search(const FullGpuSearchOptions& options);
int run_full_gpu_latency_benchmark(const FullGpuSearchOptions& options);

}  // namespace go_fann::backend_full_gpu
