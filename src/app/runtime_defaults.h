#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

namespace go_fann::defaults {

inline const std::string model_prefix = "artifacts/";
inline const std::string pq_pivots_path = model_prefix + "pq_pivots.bin";
inline const std::string pq_compressed_path = model_prefix + "pq_compressed.bin";

inline const std::string dataset_prefix = "data/";
inline const std::string graph_path = model_prefix + "graph.bin";
inline const std::string graph_storage_mode = "memory";
inline const std::string data_label_spmat_path = dataset_prefix + "base.spmat";
inline const std::string query_label_spmat_path = dataset_prefix + "query.spmat";
inline const std::string base_vector_path = dataset_prefix + "base.u8bin";
inline const std::string query_vector_path = dataset_prefix + "query.u8bin";
inline const std::string seed_sidecar_path = model_prefix + "graph.seed";
inline const std::string original_label_seed_sidecar_path = model_prefix + "graph.orig_label.seed";
inline const std::string rabitq_artifact_path = model_prefix + "rabitq_global.bin";
inline const std::string rabitq_cuvs_artifact_path = "";

inline constexpr uint32_t default_row_limit = 5000000;
inline constexpr int default_specificity_threshold = 2000;
inline constexpr int default_prefetch_hops = 3;
inline constexpr int max_prefetch_hops = 3;
inline constexpr float default_quick_filter_bonus = 0.95f;
inline constexpr double default_and_dual_route_posting_ratio_threshold = 1e30;

inline constexpr int default_dim = 128;
inline constexpr int default_pq_dim = 64;
inline constexpr int default_degree = 64;
inline constexpr int max_degree = 64;

inline constexpr int default_topk = 10;
inline constexpr int max_topk = 10;
inline constexpr int default_highfreq_rerank_shortlist = 100;
inline constexpr int default_lowfreq_rerank_shortlist = 100;
inline constexpr int max_highfreq_rerank_shortlist = 512;
inline constexpr int max_lowfreq_rerank_shortlist = 400;
inline constexpr int default_range_posting_shortlist = 200;
inline constexpr int max_range_posting_shortlist = 400;
inline constexpr int default_range_posting_postfilter_shortlist_multiplier = 10;
inline constexpr int max_range_posting_postfilter_shortlist = 4096;
inline constexpr int default_lowfreq_rerank_batch_size = 40;
inline constexpr float default_lowfreq_rerank_epsilon = 0.05f;
inline constexpr int default_lowfreq_rerank_stable_batches = 2;
inline constexpr int default_highfreq_rerank_threads = 1;
inline constexpr bool default_highfreq_rerank_streaming = false;
inline constexpr int default_highfreq_rerank_stream_batch_size = 1024;
inline constexpr bool default_lowfreq_rerank_merge_routes = false;
inline constexpr std::string_view search_early_stop_mode_fixed = "fixed";
inline constexpr std::string_view search_early_stop_mode_off = "off";
inline constexpr std::string_view search_early_stop_mode_adaptive = "adaptive";
inline const std::string default_search_early_stop_mode(search_early_stop_mode_fixed);
inline constexpr std::string_view rerank_vector_storage_mode_default = "default";
inline constexpr std::string_view rerank_vector_storage_mode_mmap_fp16 = "mmap_fp16";
inline constexpr std::string_view rerank_vector_storage_mode_memory_int8 = "memory_int8";
inline constexpr std::string_view pq_query_worker_mode_coupled = "coupled";
inline constexpr std::string_view pq_query_worker_mode_finite = "finite";
inline const std::string default_pq_query_worker_mode(pq_query_worker_mode_coupled);
inline constexpr std::string_view pq_finite_token_transport_device_copy = "device_copy";
inline constexpr std::string_view pq_finite_token_transport_mapped = "mapped";
inline const std::string default_pq_finite_token_transport(
    pq_finite_token_transport_device_copy);
inline constexpr std::string_view pq_finite_dispatch_mode_batch = "batch";
inline constexpr std::string_view pq_finite_dispatch_mode_persistent = "persistent";
inline constexpr std::string_view pq_finite_dispatch_mode_ready_ring = "ready_ring";
inline const std::string default_pq_finite_dispatch_mode(pq_finite_dispatch_mode_batch);
inline constexpr int default_pq_finite_partial_batch_timeout_us = 50;
inline constexpr int min_pq_finite_partial_batch_timeout_us = 1;
inline constexpr int max_pq_finite_partial_batch_timeout_us = 1000000;
inline constexpr std::string_view highfreq_rerank_mode_gpu = "gpu";
inline constexpr std::string_view highfreq_rerank_mode_gpu_post = "gpu_post";
inline constexpr std::string_view highfreq_rerank_mode_cpu = "cpu";
inline constexpr std::string_view highfreq_rerank_mode_off = "off";
inline constexpr std::string_view highfreq_rerank_algorithm_exact = "exact";
inline constexpr std::string_view highfreq_rerank_algorithm_heuristic = "heuristic";
inline const std::string default_highfreq_rerank_mode(highfreq_rerank_mode_gpu);
inline const std::string default_highfreq_rerank_algorithm(highfreq_rerank_algorithm_exact);
inline constexpr std::string_view lowfreq_full_filter_mode_pre = "pre";
inline constexpr std::string_view lowfreq_full_filter_mode_post = "post";
inline constexpr std::string_view lowfreq_full_filter_mode_off = "off";
inline const std::string default_lowfreq_full_filter_mode(lowfreq_full_filter_mode_pre);
inline constexpr uint32_t default_rabitq_num_anchors = 16;
inline constexpr uint32_t default_rabitq_bits_per_dim = 1;

inline const std::string default_seed_selection_mode = "group";

inline bool benchmark_quiet_enabled()
{
    static const bool enabled = std::getenv("GO_FANN_BENCHMARK_QUIET") != nullptr;
    return enabled;
}

inline bool is_valid_lowfreq_full_filter_mode(std::string_view mode)
{
    return mode == lowfreq_full_filter_mode_pre ||
           mode == lowfreq_full_filter_mode_post ||
           mode == lowfreq_full_filter_mode_off;
}

inline bool is_valid_highfreq_rerank_mode(std::string_view mode)
{
    return mode == highfreq_rerank_mode_gpu ||
           mode == highfreq_rerank_mode_gpu_post ||
           mode == highfreq_rerank_mode_cpu ||
           mode == highfreq_rerank_mode_off;
}

inline bool is_valid_search_early_stop_mode(std::string_view mode)
{
    return mode == search_early_stop_mode_fixed ||
           mode == search_early_stop_mode_off ||
           mode == search_early_stop_mode_adaptive;
}

inline bool is_valid_rerank_vector_storage_mode(std::string_view mode)
{
    return mode == rerank_vector_storage_mode_default ||
           mode == rerank_vector_storage_mode_mmap_fp16 ||
           mode == rerank_vector_storage_mode_memory_int8;
}

inline bool rerank_vector_storage_is_default(std::string_view mode)
{
    return mode == rerank_vector_storage_mode_default;
}

inline bool rerank_vector_storage_is_mmap_fp16(std::string_view mode)
{
    return mode == rerank_vector_storage_mode_mmap_fp16;
}

inline bool rerank_vector_storage_is_memory_int8(std::string_view mode)
{
    return mode == rerank_vector_storage_mode_memory_int8;
}

inline bool is_valid_pq_query_worker_mode(std::string_view mode)
{
    return mode == pq_query_worker_mode_coupled ||
           mode == pq_query_worker_mode_finite;
}

inline bool pq_query_worker_mode_is_finite(std::string_view mode)
{
    return mode == pq_query_worker_mode_finite;
}

inline bool is_valid_pq_finite_token_transport(std::string_view transport)
{
    return transport == pq_finite_token_transport_device_copy ||
           transport == pq_finite_token_transport_mapped;
}

inline bool is_valid_pq_finite_dispatch_mode(std::string_view mode)
{
    return mode == pq_finite_dispatch_mode_batch ||
           mode == pq_finite_dispatch_mode_persistent ||
           mode == pq_finite_dispatch_mode_ready_ring;
}

inline bool pq_finite_dispatch_mode_is_persistent(std::string_view mode)
{
    return mode == pq_finite_dispatch_mode_persistent;
}

inline bool pq_finite_dispatch_mode_is_ready_ring(std::string_view mode)
{
    return mode == pq_finite_dispatch_mode_ready_ring;
}

inline bool pq_finite_dispatch_mode_requires_completion_path(std::string_view mode)
{
    return pq_finite_dispatch_mode_is_persistent(mode) ||
           pq_finite_dispatch_mode_is_ready_ring(mode);
}

inline constexpr bool is_valid_pq_finite_partial_batch_timeout_us(int timeout_us)
{
    return timeout_us >= min_pq_finite_partial_batch_timeout_us &&
           timeout_us <= max_pq_finite_partial_batch_timeout_us;
}

inline bool highfreq_rerank_mode_is_gpu(std::string_view mode)
{
    return mode == highfreq_rerank_mode_gpu;
}

inline bool highfreq_rerank_mode_is_gpu_post(std::string_view mode)
{
    return mode == highfreq_rerank_mode_gpu_post;
}

inline bool highfreq_rerank_mode_is_cpu(std::string_view mode)
{
    return mode == highfreq_rerank_mode_cpu;
}

inline bool highfreq_rerank_mode_is_off(std::string_view mode)
{
    return mode == highfreq_rerank_mode_off;
}

inline bool is_valid_highfreq_rerank_algorithm(std::string_view algorithm)
{
    return algorithm == highfreq_rerank_algorithm_exact ||
           algorithm == highfreq_rerank_algorithm_heuristic;
}

inline bool highfreq_rerank_algorithm_is_exact(std::string_view algorithm)
{
    return algorithm == highfreq_rerank_algorithm_exact;
}

inline bool highfreq_rerank_algorithm_is_heuristic(std::string_view algorithm)
{
    return algorithm == highfreq_rerank_algorithm_heuristic;
}

inline constexpr int effective_highfreq_rerank_shortlist(int configured_shortlist,
                                                         int search_queue_size,
                                                         std::string_view mode,
                                                         std::string_view algorithm)
{
    return mode == highfreq_rerank_mode_cpu &&
                   algorithm == highfreq_rerank_algorithm_heuristic &&
                   search_queue_size > configured_shortlist
        ? search_queue_size
        : configured_shortlist;
}

inline bool lowfreq_full_filter_mode_is_pre(std::string_view mode)
{
    return mode == lowfreq_full_filter_mode_pre;
}

inline bool lowfreq_full_filter_mode_is_post(std::string_view mode)
{
    return mode == lowfreq_full_filter_mode_post;
}

}
