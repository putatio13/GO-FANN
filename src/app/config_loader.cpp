#include <limits>
#include "app/config_loader.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace go_fann {

namespace {

using json = nlohmann::json;

// Explicitly supplied removed controls must fail even when false or null: silently
// ignoring them could turn an SSD workload into an unintended resident allocation.
bool is_obsolete_runtime_option(const std::string& key)
{
    static const std::unordered_set<std::string> obsolete = {
        "avoid_direct_io_for_block_cache",
        "disable_block_cache",
        "disable_lane_architecture",
        "disable_lock_free_cache",
        "disable_row_cache",
        "disable_vector_cache",
        "base_vectors_on_disk",
        "benchmark_cache_warmup",
        "benchmark_cache_warmup_once",
        "benchmark_invalidate_caches_between_points",
        "benchmark_invalidate_vector_cache_between_runs",
        "benchmark_warmup_query_label_spmat_path",
        "benchmark_warmup_query_range_path",
        "benchmark_warmup_query_vector_path",
        "block_cache_capacity",
        "block_cache_capacity_blocks",
        "block_cache_enabled",
        "block_cache_shard_count",
        "block_cache_trace_path",
        "collect_cache_stats",
        "collect_lane_io_depth_stats",
        "disk_io",
        "disk_io_mode",
        "disk_queue_depth",
        "disk_worker_threads",
        "enable_block_cache",
        "enable_row_cache",
        "enable_vector_cache",
        "graph_disk_path",
        "io_mode",
        "lane_batch_size",
        "lane_cache_policy",
        "lane_cache_soft_group_quota",
        "lane_count",
        "lane_group_window_size",
        "lane_io_submit_batch_size",
        "lane_schedule_mode",
        "lowfreq_vectors_on_disk",
        "prefer_direct_io_for_block_cache",
        "range_graph_disk_path",
        "row_cache_capacity_rows",
        "row_cache_shard_count",
        "use_direct_io",
        "use_lane_architecture",
        "use_lock_free_cache",
        "vecflow",
        "vector_cache_block_rows",
        "vector_cache_capacity_blocks",
        "vector_cache_capacity_rows",
        "vector_cache_io_lane_count",
        "vector_cache_mode",
        "vector_cache_shard_count",
        "vector_cache_trace_path",
        "vector_use_direct_io",
    };
    return obsolete.count(key) != 0;
}

void reject_obsolete_config_options(const json& value, const std::string& path = "config")
{
    if (value.is_array()) {
        for (size_t i = 0; i < value.size(); ++i)
            reject_obsolete_config_options(value[i], path + "[" + std::to_string(i) + "]");
        return;
    }
    if (!value.is_object()) return;
    for (auto it = value.begin(); it != value.end(); ++it) {
        const std::string location = path + "." + it.key();
        if (is_obsolete_runtime_option(it.key())) {
            throw std::runtime_error("unsupported retired runtime option " + location +
                                     "; use resident graph storage and resident or mmap_fp16 rerank vectors");
        }
        if (it.key() == "graph_storage_mode" || it.key() == "range_graph_storage_mode" ||
            it.key() == "graph_payload_mode") {
            if (!it->is_string() ||
                (*it != "memory" && !(it.key() == "graph_payload_mode" && *it == "in_memory")))
                throw std::runtime_error("unsupported " + location +
                                         ": only memory graph storage is supported");
        }
        reject_obsolete_config_options(*it, location);
    }
}

template <typename T>
void load_optional_value(const json& source, const char* key, std::optional<T>& target)
{
    auto it = source.find(key);
    if (it != source.end() && !it->is_null()) {
        try {
            target = it->get<T>();
        } catch (const json::type_error&) {
        }
    }
}

template <typename T>
void load_optional_value(const json& source,
                         const char* object_key,
                         const char* key,
                         std::optional<T>& target)
{
    auto object_it = source.find(object_key);
    if (object_it == source.end() || !object_it->is_object()) {
        return;
    }
    load_optional_value(*object_it, key, target);
}

// An explicitly requested offload mode must never disappear on a type error.
template <typename T>
void load_predicate_option(const json& source, const char* key, std::optional<T>& target)
{
    auto it = source.find(key);
    if (it == source.end()) {
        return;
    }
    try {
        target = it->get<T>();
    } catch (const json::exception& error) {
        throw std::runtime_error(std::string("invalid ") + key + ": " + error.what());
    }
}

void load_posting_stream_batch_size(const json& source, const char* key,
                                    std::optional<uint32_t>& target)
{
    const auto it = source.find(key);
    if (it == source.end()) return;
    if (!it->is_number_integer() || *it < 1 || *it > 1048576) {
        throw std::runtime_error(std::string("invalid ") + key +
                                 ": expected an integer in [1, 1048576]");
    }
    target = it->get<uint32_t>();
}

void load_graph_payload_mode_alias(const json& source, AppConfigValues& config)
{
    std::optional<std::string> payload_mode;
    load_optional_value(source, "graph_payload_mode", payload_mode);
    if (!payload_mode) {
        return;
    }
    if (*payload_mode == "in_memory" || *payload_mode == "memory") {
        config.graph_storage_mode = "memory";
        return;
    }
    config.graph_storage_mode = *payload_mode;
}

std::string sidecar_path_from_graph_path(const std::string& graph_path, const std::string& extension)
{
    std::filesystem::path path(graph_path);
    if (path.extension() == ".bin") {
        path.replace_extension(extension);
        return path.string();
    }
    return graph_path + extension;
}

}

void reject_obsolete_cli_option(const std::string& argument)
{
    if (argument.rfind("--", 0) != 0) return;
    const size_t equals = argument.find('=');
    std::string key = argument.substr(2, equals == std::string::npos ? equals : equals - 2);
    std::replace(key.begin(), key.end(), '-', '_');
    if (key.rfind("no_", 0) == 0) key.erase(0, 3);
    if (is_obsolete_runtime_option(key))
        throw std::runtime_error("unsupported retired runtime option " + argument +
                                 "; use resident graph storage and resident or mmap_fp16 rerank vectors");
    if (equals != std::string::npos &&
        (key == "graph_storage_mode" || key == "range_graph_storage_mode" || key == "graph_payload_mode") &&
        argument.substr(equals + 1) != "memory")
        throw std::runtime_error("unsupported " + argument + ": only memory graph storage is supported");
}

AppConfigValues load_config_file(const std::string& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("unable to open config file " + path);
    }

    json root;
    input >> root;
    if (!root.is_object()) {
        throw std::runtime_error("config file root must be a JSON object");
    }

    reject_obsolete_config_options(root);

    AppConfigValues config;
    load_optional_value(root, "mode", config.mode);
    load_optional_value(root, "backend", config.backend);
    load_predicate_option(root, "cpu_rerank_schedule", config.cpu_rerank_schedule);
    load_optional_value(root, "graph_builder", config.graph_builder);
    load_optional_value(root, "graph_builder", "builder", config.graph_builder);
    load_optional_value(root, "graph_builder", "cagra_build_worker_count", config.cagra_build_worker_count);
    load_optional_value(root, "cagra_build_worker_count", config.cagra_build_worker_count);

    load_optional_value(root, "graph_path", config.graph_path);
    load_optional_value(root, "graph_file_path", config.graph_file_path);
    load_graph_payload_mode_alias(root, config);
    load_optional_value(root, "graph_storage_mode", config.graph_storage_mode);
    load_optional_value(root, "lowfreq_index_path", config.lowfreq_index_path);
    load_optional_value(root, "base_vector_path", config.base_vector_path);
    load_optional_value(root, "rerank_vector_storage_mode", config.rerank_vector_storage_mode);
    load_optional_value(root, "rerank_vector_artifact_path", config.rerank_vector_artifact_path);
    load_optional_value(root, "skip_filtered_vector_payload", config.skip_filtered_vector_payload);
    load_optional_value(root, "per_step_full_filter", config.per_step_full_filter);
    load_optional_value(root, "highfreq_rerank_mode", config.highfreq_rerank_mode);
    load_optional_value(root, "highfreq_rerank_algorithm", config.highfreq_rerank_algorithm);
    load_optional_value(root, "highfreq_rerank_shortlist", config.highfreq_rerank_shortlist);
    load_optional_value(root, "highfreq_rerank_batch_size", config.highfreq_rerank_batch_size);
    load_optional_value(root, "highfreq_rerank_epsilon", config.highfreq_rerank_epsilon);
    load_optional_value(root, "highfreq_rerank_stable_batches", config.highfreq_rerank_stable_batches);
    load_optional_value(root, "highfreq_rerank_threads", config.highfreq_rerank_threads);
    load_optional_value(root, "highfreq_rerank_streaming", config.highfreq_rerank_streaming);
    load_optional_value(root, "highfreq_rerank_merge_routes", config.highfreq_rerank_merge_routes);
    load_optional_value(root, "lowfreq_rerank_merge_routes", config.lowfreq_rerank_merge_routes);
    load_optional_value(root,
                        "highfreq_rerank_stream_batch_size",
                        config.highfreq_rerank_stream_batch_size);
    load_optional_value(root, "highfreq_max_live_routes", config.highfreq_max_live_routes);

    load_optional_value(root, "base_range_fbin_path", config.base_range_fbin_path);
    load_optional_value(root, "range_posting_index_path", config.range_posting_index_path);
    load_optional_value(root, "range_posting_mode", config.range_posting_mode);
    load_optional_value(root, "range_posting_scan_threshold", config.range_posting_scan_threshold);
    load_optional_value(root, "range_posting_shortlist", config.range_posting_shortlist);
    load_optional_value(root,
                        "range_posting_postfilter_shortlist_multiplier",
                        config.range_posting_postfilter_shortlist_multiplier);
    load_optional_value(root,
                        "range_posting_and_filter_mode",
                        config.range_posting_and_filter_mode);
    load_optional_value(root,
                        "range_label_and_policy",
                        config.range_label_and_policy);
    load_optional_value(root,
                        "range_posting_drop_wide_hybrid_or",
                        config.range_posting_drop_wide_hybrid_or);
    load_optional_value(root,
                        "range_graph_hybrid_or_enabled",
                        config.range_graph_hybrid_or_enabled);
    load_optional_value(root,
                        "range_graph_hybrid_and_enabled",
                        config.range_graph_hybrid_and_enabled);
    load_optional_value(root, "range_graph_path", config.range_graph_path);
    load_optional_value(root,
                        "range_graph_storage_mode",
                        config.range_graph_storage_mode);
    load_optional_value(root,
                        "range_graph_data_label_spmat_path",
                        config.range_graph_data_label_spmat_path);
    load_optional_value(root,
                        "range_graph_label_manifest_path",
                        config.range_graph_label_manifest_path);
    load_optional_value(root,
                        "range_graph_seed_sidecar_path",
                        config.range_graph_seed_sidecar_path);
    load_optional_value(root,
                        "range_graph_original_label_seed_sidecar_path",
                        config.range_graph_original_label_seed_sidecar_path);
    load_optional_value(root,
                        "range_graph_metadata_sidecar_path",
                        config.range_graph_metadata_sidecar_path);
    load_optional_value(root,
                        "range_graph_lowfreq_index_path",
                        config.range_graph_lowfreq_index_path);
    load_optional_value(root, "range_graph_group_path", config.range_graph_group_path);
    load_optional_value(root, "data_label_spmat_path", config.data_label_spmat_path);
    load_optional_value(root, "query_vector_path", config.query_vector_path);
    load_optional_value(root, "query_label_spmat_path", config.query_label_spmat_path);
    load_optional_value(root, "query_manifest_jsonl_path", config.query_manifest_jsonl_path);
    load_optional_value(root, "query_manifest_families", config.query_manifest_families);
    load_optional_value(root, "query_manifest_subtypes", config.query_manifest_subtypes);
    load_optional_value(root, "query_manifest_max_or_terms", config.query_manifest_max_or_terms);
    load_optional_value(root, "query_prepare_threads", config.query_prepare_threads);
    load_optional_value(root, "query_prepare_optimized", config.query_prepare_optimized);
    load_optional_value(root, "query_manifest_max_fuzzy_terms", config.query_manifest_max_fuzzy_terms);
    load_optional_value(root,
                        "query_manifest_fuzzy_cap_policy",
                        config.query_manifest_fuzzy_cap_policy);
    load_optional_value(root, "query_manifest_min_ngram_score", config.query_manifest_min_ngram_score);
    load_optional_value(root, "query_manifest_ngram_size", config.query_manifest_ngram_size);
    load_optional_value(root, "query_manifest_operator_index_path", config.query_manifest_operator_index_path);
    load_optional_value(root,
                        "query_manifest_operator_fill_mode",
                        config.query_manifest_operator_fill_mode);
    load_optional_value(root,
                        "query_manifest_operator_parity_check",
                        config.query_manifest_operator_parity_check);
    load_optional_value(root, "or_route_order_mode", config.or_route_order_mode);
    load_optional_value(root, "or_route_sketch_sidecar_path", config.or_route_sketch_sidecar_path);
    load_optional_value(root, "or_route_accuracy_cost_model_path", config.or_route_accuracy_cost_model_path);
    load_optional_value(root, "or_route_calibration_trace_path", config.or_route_calibration_trace_path);
    load_optional_value(root, "or_route_calibration_run_id", config.or_route_calibration_run_id);
    load_optional_value(root, "or_route_sketch_k", config.or_route_sketch_k);
    load_optional_value(root, "or_route_sketch_hash_seed", config.or_route_sketch_hash_seed);
    load_optional_value(root,
                        "or_route_allow_runtime_sketch_fallback",
                        config.or_route_allow_runtime_sketch_fallback);
    load_optional_value(root,
                        "or_route_runtime_sketch_fallback_max_rows",
                        config.or_route_runtime_sketch_fallback_max_rows);
    load_optional_value(root,
                        "or_route_verify_sketch_payload_checksum",
                        config.or_route_verify_sketch_payload_checksum);
    load_optional_value(root, "query_range_path", config.query_range_path);
    load_optional_value(root, "range_label_manifest_path", config.range_label_manifest_path);
    load_optional_value(root, "seed_sidecar_path", config.seed_sidecar_path);
    load_optional_value(root, "original_label_seed_sidecar_path", config.original_label_seed_sidecar_path);
    load_optional_value(root, "metadata_sidecar_path", config.metadata_sidecar_path);
    load_optional_value(root, "seed_selection_mode", config.seed_selection_mode);
    load_optional_value(root, "group_path", config.group_path);
    load_optional_value(root, "rabitq_artifact_path", config.rabitq_artifact_path);
    load_optional_value(root, "rabitq_cuvs_artifact_path", config.rabitq_cuvs_artifact_path);
    load_predicate_option(root, "rabitq_predicate_mode", config.rabitq_predicate_mode);
    load_predicate_option(root, "rabitq_predicate_storage", config.rabitq_predicate_storage);
    load_predicate_option(root, "rabitq_predicate_index_path", config.rabitq_predicate_index_path);
    load_predicate_option(root, "rabitq_predicate_bias_alpha", config.rabitq_predicate_bias_alpha);
    load_predicate_option(root, "rabitq_predicate_diagnostics", config.rabitq_predicate_diagnostics);
    load_optional_value(root, "rabitq_query_worker_mode", config.rabitq_query_worker_mode);
    load_optional_value(root,
                        "rabitq_finite_dispatch_mode",
                        config.rabitq_finite_dispatch_mode);
    load_optional_value(root,
                        "rabitq_finite_partial_batch_timeout_us",
                        config.rabitq_finite_partial_batch_timeout_us);
    load_optional_value(root, "rabitq_graph_transport", config.rabitq_graph_transport);
    load_optional_value(root,
                        "rabitq_dual_graph_cpu_gpu_overlap",
                        config.rabitq_dual_graph_cpu_gpu_overlap);
    load_optional_value(root, "rabitq_posting_schedule", config.rabitq_posting_schedule);
    load_optional_value(root, "rabitq_range_posting_segment_rows", config.rabitq_range_posting_segment_rows);
    load_optional_value(root, "rabitq_posting_stream_batch_size", config.rabitq_posting_stream_batch_size);
    load_optional_value(root, "rabitq_posting_streaming", config.rabitq_posting_streaming);
    load_optional_value(root, "rabitq_graph_persistent_workers", config.rabitq_graph_persistent_workers);
    load_optional_value(root, "rabitq_posting_persistent_workers", config.rabitq_posting_persistent_workers);
    load_optional_value(root, "rabitq_request_mode", config.rabitq_request_mode);
    load_optional_value(root,
                        "rabitq_request_batch_queries",
                        config.rabitq_request_batch_queries);
    load_optional_value(root,
                        "rabitq_request_consumed_memory",
                        config.rabitq_request_consumed_memory);
    load_optional_value(root,
                        "rabitq_gpu_graph_row_cache_mib",
                        config.rabitq_gpu_graph_row_cache_mib);
    load_optional_value(root, "rabitq_query_rotation", config.rabitq_query_rotation);
    load_optional_value(root, "rabitq_centroid_distances", config.rabitq_centroid_distances);
    load_optional_value(root, "rabitq_use_int8_score", config.rabitq_use_int8_score);
    load_optional_value(root, "output_csv_path", config.output_csv_path);
    load_optional_value(root, "full_filter", config.full_filter);
    load_optional_value(root, "full_gpu_fp16", config.full_gpu_fp16);
    load_optional_value(root, "overlap_graph_path", config.overlap_graph_path);
    load_optional_value(root, "overlap_manifest_path", config.overlap_manifest_path);
    load_optional_value(root, "overlap_max_bridges", config.overlap_max_bridges);

    load_optional_value(root, "use_fp16", config.full_gpu_fp16);
    load_optional_value(root, "lowfreq_full_filter_mode", config.lowfreq_full_filter_mode);
    load_optional_value(root, "pq_graph_transport", config.pq_graph_transport);
    load_optional_value(root, "pq_query_worker_mode", config.pq_query_worker_mode);
    load_optional_value(root, "pq_cpu_persistent_workers", config.pq_cpu_persistent_workers);
    load_optional_value(root, "pq_request_mode", config.pq_request_mode);
    load_optional_value(root,
                        "pq_request_batch_queries",
                        config.pq_request_batch_queries);
    load_optional_value(root,
                        "pq_request_consumed_memory",
                        config.pq_request_consumed_memory);
    load_optional_value(root,
                        "pq_finite_token_transport",
                        config.pq_finite_token_transport);
    load_optional_value(root,
                        "pq_finite_dispatch_mode",
                        config.pq_finite_dispatch_mode);
    load_optional_value(root,
                        "pq_finite_partial_batch_timeout_us",
                        config.pq_finite_partial_batch_timeout_us);
    load_optional_value(root, "host_reply_wait_policy", config.host_reply_wait_policy);

    load_optional_value(root, "pq", "compressed_path", config.pq_compressed_path);
    load_optional_value(root, "pq", "pivots_path", config.pq_pivots_path);
    load_optional_value(root, "pq", "buffer_mode", config.pq_buffer_mode);
    load_optional_value(root, "pq", "graph_transport", config.pq_graph_transport);
    load_optional_value(root, "pq", "query_worker_mode", config.pq_query_worker_mode);
    load_optional_value(root, "pq", "cpu_persistent_workers", config.pq_cpu_persistent_workers);
    load_optional_value(root, "pq", "request_mode", config.pq_request_mode);
    load_optional_value(root,
                        "pq",
                        "request_batch_queries",
                        config.pq_request_batch_queries);
    load_optional_value(root,
                        "pq",
                        "request_consumed_memory",
                        config.pq_request_consumed_memory);
    load_optional_value(root,
                        "pq",
                        "finite_token_transport",
                        config.pq_finite_token_transport);
    load_optional_value(root,
                        "pq",
                        "finite_dispatch_mode",
                        config.pq_finite_dispatch_mode);
    load_optional_value(root,
                        "pq",
                        "finite_partial_batch_timeout_us",
                        config.pq_finite_partial_batch_timeout_us);
    load_optional_value(root, "pq", "host_reply_wait_policy", config.host_reply_wait_policy);
    load_optional_value(root, "pq", "pq_dim", config.pq_dim);
    load_optional_value(root, "pq", "routing_mode", config.routing_mode);
    load_optional_value(root,
                        "rabitq",
                        "query_worker_mode",
                        config.rabitq_query_worker_mode);
    load_optional_value(root,
                        "rabitq",
                        "finite_dispatch_mode",
                        config.rabitq_finite_dispatch_mode);
    load_optional_value(root,
                        "rabitq",
                        "finite_partial_batch_timeout_us",
                        config.rabitq_finite_partial_batch_timeout_us);
    load_optional_value(root,
                        "rabitq",
                        "graph_transport",
                        config.rabitq_graph_transport);
    load_optional_value(root,
                        "rabitq",
                        "dual_graph_cpu_gpu_overlap",
                        config.rabitq_dual_graph_cpu_gpu_overlap);
    load_optional_value(root, "rabitq", "posting_schedule", config.rabitq_posting_schedule);
    load_optional_value(root, "rabitq", "range_posting_segment_rows", config.rabitq_range_posting_segment_rows);
    load_optional_value(root, "rabitq", "posting_stream_batch_size", config.rabitq_posting_stream_batch_size);
    load_optional_value(root, "rabitq", "posting_streaming", config.rabitq_posting_streaming);
    load_optional_value(root, "rabitq", "graph_persistent_workers", config.rabitq_graph_persistent_workers);
    load_optional_value(root, "rabitq", "posting_persistent_workers", config.rabitq_posting_persistent_workers);
    load_optional_value(root,
                        "rabitq",
                        "request_mode",
                        config.rabitq_request_mode);
    load_optional_value(root,
                        "rabitq",
                        "request_batch_queries",
                        config.rabitq_request_batch_queries);
    load_optional_value(root,
                        "rabitq",
                        "request_consumed_memory",
                        config.rabitq_request_consumed_memory);
    load_optional_value(root,
                        "rabitq",
                        "gpu_graph_row_cache_mib",
                        config.rabitq_gpu_graph_row_cache_mib);

    auto rabitq_it = root.find("rabitq");
    if (rabitq_it != root.end() && rabitq_it->is_object()) {
        load_predicate_option(*rabitq_it, "predicate_mode", config.rabitq_predicate_mode);
        load_predicate_option(*rabitq_it, "predicate_storage", config.rabitq_predicate_storage);
        load_predicate_option(*rabitq_it, "predicate_index_path", config.rabitq_predicate_index_path);
        load_predicate_option(*rabitq_it, "predicate_bias_alpha", config.rabitq_predicate_bias_alpha);
        load_predicate_option(*rabitq_it, "predicate_diagnostics", config.rabitq_predicate_diagnostics);
    }

    auto graph_it = root.find("runtime");
    if (graph_it != root.end() && graph_it->is_object()) {
        load_graph_payload_mode_alias(*graph_it, config);
    }
    load_optional_value(root, "runtime", "graph_path", config.graph_path);
    load_optional_value(root, "runtime", "graph_file_path", config.graph_file_path);
    load_optional_value(root, "runtime", "graph_storage_mode", config.graph_storage_mode);
    load_optional_value(root, "runtime", "lowfreq_index_path", config.lowfreq_index_path);
    load_optional_value(root, "runtime", "base_vector_path", config.base_vector_path);
    load_optional_value(root,
                        "runtime",
                        "rerank_vector_storage_mode",
                        config.rerank_vector_storage_mode);
    load_optional_value(root,
                        "runtime",
                        "rerank_vector_artifact_path",
                        config.rerank_vector_artifact_path);
    load_optional_value(root, "runtime", "skip_filtered_vector_payload", config.skip_filtered_vector_payload);
    load_optional_value(root, "runtime", "per_step_full_filter", config.per_step_full_filter);
    load_optional_value(root, "runtime", "base_range_fbin_path", config.base_range_fbin_path);
    load_optional_value(root, "runtime", "range_posting_index_path", config.range_posting_index_path);
    load_optional_value(root, "runtime", "range_posting_mode", config.range_posting_mode);
    load_optional_value(root,
                        "runtime",
                        "range_posting_scan_threshold",
                        config.range_posting_scan_threshold);
    load_optional_value(root, "runtime", "range_posting_shortlist", config.range_posting_shortlist);
    load_optional_value(root,
                        "runtime",
                        "range_posting_postfilter_shortlist_multiplier",
                        config.range_posting_postfilter_shortlist_multiplier);
    load_optional_value(root,
                        "runtime",
                        "range_posting_and_filter_mode",
                        config.range_posting_and_filter_mode);
    load_optional_value(root,
                        "runtime",
                        "range_label_and_policy",
                        config.range_label_and_policy);
    load_optional_value(root,
                        "runtime",
                        "range_posting_drop_wide_hybrid_or",
                        config.range_posting_drop_wide_hybrid_or);
    load_optional_value(root,
                        "runtime",
                        "range_graph_hybrid_or_enabled",
                        config.range_graph_hybrid_or_enabled);
    load_optional_value(root,
                        "runtime",
                        "range_graph_hybrid_and_enabled",
                        config.range_graph_hybrid_and_enabled);
    load_optional_value(root, "runtime", "range_graph_path", config.range_graph_path);
    load_optional_value(root,
                        "runtime",
                        "range_graph_storage_mode",
                        config.range_graph_storage_mode);
    load_optional_value(root,
                        "runtime",
                        "range_graph_data_label_spmat_path",
                        config.range_graph_data_label_spmat_path);
    load_optional_value(root,
                        "runtime",
                        "range_graph_label_manifest_path",
                        config.range_graph_label_manifest_path);
    load_optional_value(root,
                        "runtime",
                        "range_graph_seed_sidecar_path",
                        config.range_graph_seed_sidecar_path);
    load_optional_value(root,
                        "runtime",
                        "range_graph_original_label_seed_sidecar_path",
                        config.range_graph_original_label_seed_sidecar_path);
    load_optional_value(root,
                        "runtime",
                        "range_graph_metadata_sidecar_path",
                        config.range_graph_metadata_sidecar_path);
    load_optional_value(root,
                        "runtime",
                        "range_graph_lowfreq_index_path",
                        config.range_graph_lowfreq_index_path);
    load_optional_value(root,
                        "runtime",
                        "range_graph_group_path",
                        config.range_graph_group_path);
    load_optional_value(root, "runtime", "data_label_spmat_path", config.data_label_spmat_path);
    load_optional_value(root, "runtime", "query_vector_path", config.query_vector_path);
    load_optional_value(root, "runtime", "query_label_spmat_path", config.query_label_spmat_path);
    load_optional_value(root, "runtime", "query_manifest_jsonl_path", config.query_manifest_jsonl_path);
    load_optional_value(root, "runtime", "query_manifest_families", config.query_manifest_families);
    load_optional_value(root, "runtime", "query_manifest_subtypes", config.query_manifest_subtypes);
    load_optional_value(root, "runtime", "query_manifest_max_or_terms", config.query_manifest_max_or_terms);
    load_optional_value(root, "runtime", "query_prepare_threads", config.query_prepare_threads);
    load_optional_value(root, "runtime", "query_prepare_optimized", config.query_prepare_optimized);
    load_optional_value(root, "runtime", "query_manifest_max_fuzzy_terms", config.query_manifest_max_fuzzy_terms);
    load_optional_value(root,
                        "runtime",
                        "query_manifest_fuzzy_cap_policy",
                        config.query_manifest_fuzzy_cap_policy);
    load_optional_value(root, "runtime", "query_manifest_min_ngram_score", config.query_manifest_min_ngram_score);
    load_optional_value(root, "runtime", "query_manifest_ngram_size", config.query_manifest_ngram_size);
    load_optional_value(root,
                        "runtime",
                        "query_manifest_operator_index_path",
                        config.query_manifest_operator_index_path);
    load_optional_value(root,
                        "runtime",
                        "query_manifest_operator_fill_mode",
                        config.query_manifest_operator_fill_mode);
    load_optional_value(root,
                        "runtime",
                        "query_manifest_operator_parity_check",
                        config.query_manifest_operator_parity_check);
    load_optional_value(root, "runtime", "or_route_order_mode", config.or_route_order_mode);
    load_optional_value(root,
                        "runtime",
                        "or_route_sketch_sidecar_path",
                        config.or_route_sketch_sidecar_path);
    load_optional_value(root,
                        "runtime",
                        "or_route_accuracy_cost_model_path",
                        config.or_route_accuracy_cost_model_path);
    load_optional_value(root,
                        "runtime",
                        "or_route_calibration_trace_path",
                        config.or_route_calibration_trace_path);
    load_optional_value(root,
                        "runtime",
                        "or_route_calibration_run_id",
                        config.or_route_calibration_run_id);
    load_optional_value(root, "runtime", "or_route_sketch_k", config.or_route_sketch_k);
    load_optional_value(root,
                        "runtime",
                        "or_route_sketch_hash_seed",
                        config.or_route_sketch_hash_seed);
    load_optional_value(root,
                        "runtime",
                        "or_route_allow_runtime_sketch_fallback",
                        config.or_route_allow_runtime_sketch_fallback);
    load_optional_value(root,
                        "runtime",
                        "or_route_runtime_sketch_fallback_max_rows",
                        config.or_route_runtime_sketch_fallback_max_rows);
    load_optional_value(root,
                        "runtime",
                        "or_route_verify_sketch_payload_checksum",
                        config.or_route_verify_sketch_payload_checksum);
    load_optional_value(root, "runtime", "query_range_path", config.query_range_path);
    load_optional_value(root, "runtime", "range_label_manifest_path", config.range_label_manifest_path);
    load_optional_value(root, "runtime", "seed_sidecar_path", config.seed_sidecar_path);
    load_optional_value(root, "runtime", "original_label_seed_sidecar_path", config.original_label_seed_sidecar_path);
    load_optional_value(root, "runtime", "metadata_sidecar_path", config.metadata_sidecar_path);
    load_optional_value(root, "runtime", "seed_selection_mode", config.seed_selection_mode);
    load_optional_value(root, "runtime", "group_path", config.group_path);
    load_optional_value(root, "runtime", "routing_mode", config.routing_mode);
    load_optional_value(root, "runtime", "full_filter", config.full_filter);
    load_optional_value(root, "runtime", "lowfreq_full_filter_mode", config.lowfreq_full_filter_mode);
    load_optional_value(root, "runtime", "row_limit", config.row_limit);
    load_optional_value(root, "runtime", "stream_graph_worker_count", config.stream_graph_worker_count);
    load_optional_value(root, "runtime", "stream_graph_workers", config.stream_graph_worker_count);
    load_optional_value(root, "runtime", "stream_graph_auto_workers", config.stream_graph_auto_workers);
    load_optional_value(root,
                        "runtime",
                        "stream_graph_memory_budget_bytes",
                        config.stream_graph_memory_budget_bytes);
    load_optional_value(root,
                        "runtime",
                        "stream_graph_per_worker_memory_cap_bytes",
                        config.stream_graph_per_worker_memory_cap_bytes);
    load_optional_value(root,
                        "runtime",
                        "stream_graph_allow_concurrent_cagra",
                        config.stream_graph_allow_concurrent_cagra);
    load_optional_value(root, "runtime", "cagra_ace_build_dir", config.cagra_ace_build_dir);
    load_optional_value(root, "runtime", "cagra_ace_force_disk", config.cagra_ace_force_disk);
    load_optional_value(root, "runtime", "specificity_threshold", config.specificity_threshold);
    load_optional_value(root, "runtime", "prefetch_hops", config.prefetch_hops);
    load_optional_value(root, "runtime", "quick_filter_bonus", config.quick_filter_bonus);
    load_optional_value(root, "runtime", "bit_filter_bonus", config.bit_filter_bonus);
    load_optional_value(root, "runtime", "range_filter_bonus", config.range_filter_bonus);
    load_optional_value(root,
                        "runtime",
                        "and_dual_route_posting_ratio_threshold",
                        config.and_dual_route_posting_ratio_threshold);
    load_optional_value(root, "runtime", "degree", config.degree);
    load_optional_value(root, "runtime", "topk", config.topk);
    load_optional_value(root, "runtime", "highfreq_rerank_mode", config.highfreq_rerank_mode);
    load_optional_value(root, "runtime", "highfreq_rerank_algorithm", config.highfreq_rerank_algorithm);
    load_optional_value(root, "runtime", "highfreq_rerank_shortlist", config.highfreq_rerank_shortlist);
    load_optional_value(root, "runtime", "highfreq_rerank_batch_size", config.highfreq_rerank_batch_size);
    load_optional_value(root, "runtime", "highfreq_rerank_epsilon", config.highfreq_rerank_epsilon);
    load_optional_value(root, "runtime", "highfreq_rerank_stable_batches", config.highfreq_rerank_stable_batches);
    load_optional_value(root, "runtime", "highfreq_rerank_threads", config.highfreq_rerank_threads);
    load_optional_value(root, "runtime", "highfreq_rerank_streaming", config.highfreq_rerank_streaming);
    load_optional_value(root, "runtime", "highfreq_rerank_merge_routes", config.highfreq_rerank_merge_routes);
    load_optional_value(root,
                        "runtime",
                        "lowfreq_rerank_merge_routes",
                        config.lowfreq_rerank_merge_routes);
    load_optional_value(root,
                        "runtime",
                        "highfreq_rerank_stream_batch_size",
                        config.highfreq_rerank_stream_batch_size);
    load_optional_value(root, "runtime", "highfreq_max_live_routes", config.highfreq_max_live_routes);
    load_optional_value(root, "runtime", "lowfreq_rerank_shortlist", config.lowfreq_rerank_shortlist);
    load_optional_value(root, "runtime", "lowfreq_rerank_batch_size", config.lowfreq_rerank_batch_size);
    load_optional_value(root, "runtime", "lowfreq_rerank_epsilon", config.lowfreq_rerank_epsilon);
    load_optional_value(root, "runtime", "lowfreq_rerank_stable_batches", config.lowfreq_rerank_stable_batches);
    load_optional_value(root, "runtime", "enable_lowfreq_rerank", config.enable_lowfreq_rerank);
    load_optional_value(root, "runtime", "lowfreq_rerank", config.enable_lowfreq_rerank);
    load_optional_value(root, "runtime", "dim", config.dim);

    load_optional_value(root, "rabitq", "artifact_path", config.rabitq_artifact_path);
    load_optional_value(root, "rabitq", "cuvs_artifact_path", config.rabitq_cuvs_artifact_path);
    load_optional_value(root, "rabitq", "num_anchors", config.rabitq_num_anchors);
    load_optional_value(root, "rabitq", "bits_per_dim", config.rabitq_bits_per_dim);
    load_optional_value(root, "rabitq", "enable_rerank", config.rabitq_enable_rerank);
    load_optional_value(root, "rabitq", "rerank", config.rabitq_enable_rerank);
    load_optional_value(root, "rabitq", "enable_highfreq_rerank", config.rabitq_enable_highfreq_rerank);
    load_optional_value(root, "rabitq", "highfreq_rerank", config.rabitq_enable_highfreq_rerank);
    load_optional_value(root, "rabitq", "enable_lowfreq_rerank", config.rabitq_enable_lowfreq_rerank);
    load_optional_value(root, "rabitq", "lowfreq_rerank", config.rabitq_enable_lowfreq_rerank);
    load_optional_value(root, "rabitq", "lowfreq_gpu_posting", config.rabitq_lowfreq_gpu_posting);
    load_optional_value(root,
                        "rabitq",
                        "lowfreq_rerank_merge_routes",
                        config.lowfreq_rerank_merge_routes);
    load_optional_value(root, "rabitq", "query_rotation", config.rabitq_query_rotation);
    load_optional_value(root, "rabitq", "centroid_distances", config.rabitq_centroid_distances);
    load_optional_value(root, "rabitq", "use_lut", config.rabitq_use_lut);
    load_optional_value(root,
                        "rabitq",
                        "use_int8_score",
                        config.rabitq_use_int8_score);

    load_optional_value(root, "force_rebuild", config.force_rebuild);
    load_optional_value(root, "run_analyze", config.run_analyze);
    load_optional_value(root, "full_filter", config.full_filter);
    load_optional_value(root, "lowfreq_full_filter_mode", config.lowfreq_full_filter_mode);
    load_optional_value(root, "row_limit", config.row_limit);
    load_optional_value(root, "stream_graph_worker_count", config.stream_graph_worker_count);
    load_optional_value(root, "stream_graph_workers", config.stream_graph_worker_count);
    load_optional_value(root, "stream_graph_auto_workers", config.stream_graph_auto_workers);
    load_optional_value(root, "stream_graph_memory_budget_bytes", config.stream_graph_memory_budget_bytes);
    load_optional_value(root,
                        "stream_graph_per_worker_memory_cap_bytes",
                        config.stream_graph_per_worker_memory_cap_bytes);
    load_optional_value(root, "stream_graph_allow_concurrent_cagra", config.stream_graph_allow_concurrent_cagra);
    load_optional_value(root, "cagra_ace_build_dir", config.cagra_ace_build_dir);
    load_optional_value(root, "cagra_ace_force_disk", config.cagra_ace_force_disk);
    load_optional_value(root, "specificity_threshold", config.specificity_threshold);
    load_optional_value(root, "prefetch_hops", config.prefetch_hops);
    load_optional_value(root, "quick_filter_bonus", config.quick_filter_bonus);
    load_optional_value(root, "bit_filter_bonus", config.bit_filter_bonus);
    load_optional_value(root, "range_filter_bonus", config.range_filter_bonus);
    load_optional_value(root,
                        "and_dual_route_posting_ratio_threshold",
                        config.and_dual_route_posting_ratio_threshold);
    load_optional_value(root, "routing_mode", config.routing_mode);
    load_optional_value(root, "query_manifest_jsonl_path", config.query_manifest_jsonl_path);
    load_optional_value(root, "query_manifest_families", config.query_manifest_families);
    load_optional_value(root, "query_manifest_subtypes", config.query_manifest_subtypes);
    load_optional_value(root, "query_manifest_max_or_terms", config.query_manifest_max_or_terms);
    load_optional_value(root, "query_manifest_max_fuzzy_terms", config.query_manifest_max_fuzzy_terms);
    load_optional_value(root,
                        "query_manifest_fuzzy_cap_policy",
                        config.query_manifest_fuzzy_cap_policy);
    load_optional_value(root, "query_manifest_min_ngram_score", config.query_manifest_min_ngram_score);
    load_optional_value(root, "query_manifest_ngram_size", config.query_manifest_ngram_size);
    load_optional_value(root, "query_manifest_operator_index_path", config.query_manifest_operator_index_path);
    load_optional_value(root,
                        "query_manifest_operator_fill_mode",
                        config.query_manifest_operator_fill_mode);
    load_optional_value(root,
                        "query_manifest_operator_parity_check",
                        config.query_manifest_operator_parity_check);
    load_optional_value(root, "or_route_order_mode", config.or_route_order_mode);
    load_optional_value(root, "or_route_sketch_sidecar_path", config.or_route_sketch_sidecar_path);
    load_optional_value(root, "or_route_accuracy_cost_model_path", config.or_route_accuracy_cost_model_path);
    load_optional_value(root, "or_route_calibration_trace_path", config.or_route_calibration_trace_path);
    load_optional_value(root, "or_route_calibration_run_id", config.or_route_calibration_run_id);
    load_optional_value(root, "or_route_sketch_k", config.or_route_sketch_k);
    load_optional_value(root, "or_route_sketch_hash_seed", config.or_route_sketch_hash_seed);
    load_optional_value(root,
                        "or_route_allow_runtime_sketch_fallback",
                        config.or_route_allow_runtime_sketch_fallback);
    load_optional_value(root,
                        "or_route_runtime_sketch_fallback_max_rows",
                        config.or_route_runtime_sketch_fallback_max_rows);
    load_optional_value(root,
                        "or_route_verify_sketch_payload_checksum",
                        config.or_route_verify_sketch_payload_checksum);
    load_optional_value(root, "warmup_runs", config.warmup_runs);
    load_optional_value(root, "timed_runs", config.timed_runs);
    load_optional_value(root, "process_cpu_start", config.process_cpu_start);
    load_optional_value(root, "process_cpu_count", config.process_cpu_count);
    load_optional_value(root, "host_service_threads", config.host_service_threads);
    load_optional_value(root, "host_service_cpu_start", config.host_service_cpu_start);
    load_optional_value(root, "host_service_cpu_count", config.host_service_cpu_count);
    load_optional_value(root, "host_demand_batch_size", config.host_demand_batch_size);
    load_optional_value(root,
                        "host_demand_batch_timeout_us",
                        config.host_demand_batch_timeout_us);
    load_optional_value(root, "search_queue_size", config.search_queue_size);
    load_optional_value(root, "search_finish_count", config.search_finish_count);
    load_optional_value(root, "search_max_steps", config.search_max_steps);
    load_optional_value(root, "search_early_stop_mode", config.search_early_stop_mode);
    load_optional_value(root, "benchmark_quick_filter_bonuses", config.benchmark_quick_filter_bonuses);
    load_optional_value(root, "benchmark_queue_sizes", config.benchmark_queue_sizes);
    load_optional_value(root, "benchmark_output", config.benchmark_output);
    load_optional_value(root, "latency_queue_sizes", config.latency_queue_sizes);
    load_optional_value(root, "latency_batch_sizes", config.latency_batch_sizes);
    load_optional_value(root, "latency_output", config.latency_output);
    load_optional_value(root, "latency_trace_output", config.latency_trace_output);
    load_optional_value(root, "latency_query_start", config.latency_query_start);
    load_optional_value(root, "latency_query_limit", config.latency_query_limit);
    load_optional_value(root, "latency_concurrency_levels", config.latency_concurrency_levels);
    load_optional_value(root, "latency_run_all_queries", config.latency_run_all_queries);
    load_optional_value(root, "latency_time_planner", config.latency_time_planner);
    load_optional_value(root, "latency_online_metadata", config.latency_online_metadata);
    load_optional_value(root, "latency_staggered", config.latency_staggered);
    load_optional_value(root, "latency_persistent_queue", config.latency_persistent_queue);
    load_optional_value(root,
                        "latency_queue_sweep_full_warmup",
                        config.latency_queue_sweep_full_warmup);
    load_optional_value(root, "latency_lowfreq_mode", config.latency_lowfreq_mode);
    load_optional_value(root, "degree", config.degree);
    load_optional_value(root, "topk", config.topk);
    load_optional_value(root, "highfreq_rerank_mode", config.highfreq_rerank_mode);
    load_optional_value(root, "highfreq_rerank_algorithm", config.highfreq_rerank_algorithm);
    load_optional_value(root, "highfreq_rerank_shortlist", config.highfreq_rerank_shortlist);
    load_optional_value(root, "highfreq_rerank_batch_size", config.highfreq_rerank_batch_size);
    load_optional_value(root, "highfreq_rerank_epsilon", config.highfreq_rerank_epsilon);
    load_optional_value(root, "highfreq_rerank_stable_batches", config.highfreq_rerank_stable_batches);
    load_optional_value(root, "highfreq_rerank_threads", config.highfreq_rerank_threads);
    load_optional_value(root, "highfreq_rerank_streaming", config.highfreq_rerank_streaming);
    load_optional_value(root, "highfreq_rerank_merge_routes", config.highfreq_rerank_merge_routes);
    load_optional_value(root, "lowfreq_rerank_merge_routes", config.lowfreq_rerank_merge_routes);
    load_optional_value(root,
                        "highfreq_rerank_stream_batch_size",
                        config.highfreq_rerank_stream_batch_size);
    load_optional_value(root, "highfreq_max_live_routes", config.highfreq_max_live_routes);
    load_optional_value(root, "lowfreq_rerank_shortlist", config.lowfreq_rerank_shortlist);
    load_optional_value(root, "lowfreq_rerank_batch_size", config.lowfreq_rerank_batch_size);
    load_optional_value(root, "lowfreq_rerank_epsilon", config.lowfreq_rerank_epsilon);
    load_optional_value(root, "lowfreq_rerank_stable_batches", config.lowfreq_rerank_stable_batches);
    load_optional_value(root, "enable_lowfreq_rerank", config.enable_lowfreq_rerank);
    load_optional_value(root, "lowfreq_rerank", config.enable_lowfreq_rerank);
    load_optional_value(root, "dim", config.dim);
    load_optional_value(root, "rabitq_num_anchors", config.rabitq_num_anchors);
    load_optional_value(root, "rabitq_bits_per_dim", config.rabitq_bits_per_dim);
    load_optional_value(root, "rabitq_enable_rerank", config.rabitq_enable_rerank);
    load_optional_value(root, "rabitq_enable_highfreq_rerank", config.rabitq_enable_highfreq_rerank);
    load_optional_value(root, "rabitq_enable_lowfreq_rerank", config.rabitq_enable_lowfreq_rerank);
    load_optional_value(root, "rabitq_lowfreq_gpu_posting", config.rabitq_lowfreq_gpu_posting);
    load_optional_value(root, "rabitq_use_lut", config.rabitq_use_lut);
    load_optional_value(root, "rabitq_cuvs_artifact_path", config.rabitq_cuvs_artifact_path);

    if (config.graph_path) {
        if (!config.seed_sidecar_path) {
            config.seed_sidecar_path = sidecar_path_from_graph_path(*config.graph_path, ".seed");
        }
        if (!config.original_label_seed_sidecar_path) {
            config.original_label_seed_sidecar_path =
                sidecar_path_from_graph_path(*config.graph_path, ".orig_label.seed");
        }
    }

    // Neutral root keys take precedence over legacy RabitQ aliases. Backend
    // nested PQ keys override aliases only when this config selects PQ.
    const auto pq = root.find("pq");
    if (config.backend.value_or("pq") == "pq" && pq != root.end() && pq->is_object()) {
        load_predicate_option(*pq, "predicate_mode", config.rabitq_predicate_mode);
        load_predicate_option(*pq, "predicate_storage", config.rabitq_predicate_storage);
        load_predicate_option(*pq, "predicate_bias_alpha", config.rabitq_predicate_bias_alpha);
        load_predicate_option(*pq, "predicate_diagnostics", config.rabitq_predicate_diagnostics);
    }
    load_predicate_option(root, "rerank_distance_reuse", config.rerank_distance_reuse);
    if (const auto it = root.find("rabitq_incremental_rerank_shortlist"); it != root.end()) {
        if (!it->is_number_integer() || *it < 0 || *it > std::numeric_limits<int>::max())
            throw std::runtime_error("invalid rabitq_incremental_rerank_shortlist: expected nonnegative int");
        config.rabitq_incremental_rerank_shortlist = it->get<int>();
    }
    load_predicate_option(root, "pq_lowfreq_scan_mode", config.pq_lowfreq_scan_mode);
    load_predicate_option(root, "pq_posting_streaming", config.pq_posting_streaming);
    load_posting_stream_batch_size(root, "pq_posting_stream_batch_size", config.pq_posting_stream_batch_size);
    load_predicate_option(root, "posting_predicate_order", config.posting_predicate_order);
    // Parse explicit PQ policy even on another backend so validation can reject it.
    if (pq != root.end() && pq->is_object()) {
        load_predicate_option(*pq, "lowfreq_scan_mode", config.pq_lowfreq_scan_mode);
        load_predicate_option(*pq, "posting_streaming", config.pq_posting_streaming);
        load_posting_stream_batch_size(*pq, "posting_stream_batch_size", config.pq_posting_stream_batch_size);
    }
    load_predicate_option(root, "predicate_mode", config.rabitq_predicate_mode);
    load_predicate_option(root, "predicate_storage", config.rabitq_predicate_storage);
    load_predicate_option(root, "predicate_bias_alpha", config.rabitq_predicate_bias_alpha);
    load_predicate_option(root, "predicate_diagnostics", config.rabitq_predicate_diagnostics);

    return config;
}

}
